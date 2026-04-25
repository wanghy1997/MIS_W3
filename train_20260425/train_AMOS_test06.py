import argparse
import logging
import os
import shutil
import sys

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from loss_amos import GADice, GACE, ContrastiveLoss
from dataloaders.dataset import *
from networks.amos_magicnet_test06 import *
from utils import ramps, test_amos, cube_losses, cube_utils

from torch.nn.parallel import DataParallel as DP
# from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--data_path', type=str, default='/data/why/Datasets/amos/', help='data_path')
parser.add_argument('--log_path', type=str, default='/data/why/logs/', help='log_path')
parser.add_argument('--exp', type=str, default='Ours', help='exp_name')
parser.add_argument('--remark', type=str, default='ours1', help='remark')
# parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--deterministic', type=bool, default=True, help='whether to use deterministic training')
parser.add_argument('--label_num', type=int, default=10, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')

parser.add_argument('--max_epoch', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base_lr')
parser.add_argument('--max_lr_iters', type=int, default=70000, help='max_lr_iters')

parser.add_argument('--contrast_w', type=float, default=0.1, help='contrastive loss weight')
parser.add_argument('--contrast_sample_num', type=int, default=10, help='contrastive sample_num')
parser.add_argument('--start_contrast_epoch', type=int, default=20, help='start contrastive loss epoch')
parser.add_argument('--entropy_max_epoch', type=int, default=100, help='entropy_max_epoch')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_w', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
args = parser.parse_args()

gpu_num = torch.cuda.device_count()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# gpu_num = len(args.gpu.split(','))



"""李雅普诺夫更新-gpt5.1"""
def _iter_params(student_model, teacher_model):
    # 兼容 DDP/DP
    s = student_model.module if hasattr(student_model, "module") else student_model
    t = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
    return s.parameters(), t.parameters(), s.buffers(), t.buffers()

@torch.no_grad()
def lyapunov_V(student_model, teacher_model):
    """
    V = 1/2 ||theta_s - theta_t||_2^2  (Eq.(6))
    """
    s_params, t_params, _, _ = _iter_params(student_model, teacher_model)
    sq_sum = None
    for ps, pt in zip(s_params, t_params):
        if not ps.dtype.is_floating_point:
            continue
        diff = (ps.data - pt.data).float()
        val = diff.pow(2).sum()
        sq_sum = val if sq_sum is None else (sq_sum + val)
    if sq_sum is None:
        return torch.tensor(0.0, device=next(student_model.parameters()).device)
    return 0.5 * sq_sum

@torch.no_grad()
def lyap_student_pull(student_model, teacher_model, lr: float, lam: float):
    """
    Eq.(8): theta_s <- theta_s + lr * lam * (theta_t - theta_s)
    注意：论文写的是 +λ(θ_t - θ_s)，这里用 lr*lam 把步长吸收进去更像真实训练。
    """
    s_params, t_params, _, _ = _iter_params(student_model, teacher_model)
    coef = lr * lam
    for ps, pt in zip(s_params, t_params):
        if not ps.dtype.is_floating_point:
            continue
        ps.data.add_(pt.data - ps.data, alpha=coef)

@torch.no_grad()
def lyapema_teacher_update(student_model, teacher_model, gamma: float = 1.0, sync_buffers: bool = True):
    """
    Eq.(9): alpha_t = sigmoid(gamma * ||theta_s - theta_t||_2)
    Eq.(5): theta_t <- alpha_t * theta_t + (1-alpha_t) * theta_s
    """
    s_params, t_params, s_bufs, t_bufs = _iter_params(student_model, teacher_model)

    # 计算 ||theta_s - theta_t||_2
    sq_sum = None
    for ps, pt in zip(s_params, t_params):
        if not ps.dtype.is_floating_point:
            continue
        diff = (ps.data - pt.data).float()
        val = diff.pow(2).sum()
        sq_sum = val if sq_sum is None else (sq_sum + val)

    if sq_sum is None:
        return

    norm = torch.sqrt(sq_sum + 1e-12)
    alpha_t = torch.sigmoid(gamma * norm).item()  # (0,1)

    # EMA 更新 teacher
    # 重新取一遍 params（因为上面迭代器耗尽了）
    s_params, t_params, s_bufs, t_bufs = _iter_params(student_model, teacher_model)
    for pt, ps in zip(t_params, s_params):
        if not ps.dtype.is_floating_point:
            continue
        pt.data.mul_(alpha_t).add_(ps.data, alpha=1.0 - alpha_t)

    # 建议同步 buffers（BN running stats 等），否则 teacher 可能“脏”
    if sync_buffers:
        for bt, bs in zip(t_bufs, s_bufs):
            bt.data.copy_(bs.data)


def get_current_consistency_weight(epoch):
    return args.consistency_w * ramps.sigmoid_rampup(epoch, args.max_epoch)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def create_model(n_classes=16, cube_size=32, patch_size=96, ema=False):
    model = VNet_Magic_CLIP_2p_Contrast(n_channels=1,
                                        n_classes=n_classes,
                                        cube_size=cube_size,
                                        patch_size=patch_size,
                                        )
    if ema:
        for param in model.parameters():
            param.detach_()

    model = DP(model, device_ids=[i for i in range(gpu_num)]).cuda()
    return model


def read_list(split):
    ids_list = np.loadtxt(
        os.path.join(args.data_path, 'split_txts/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


if args.label_num == 4:
    # 2%, 4 labeld
    labeled_list = read_list('labeled_2p')
    unlabeled_list = read_list('unlabeled_2p')
elif args.label_num == 10:
    # 5%, 10 labeld
    labeled_list = read_list('labeled_5p')
    unlabeled_list = read_list('unlabeled_5p')
elif args.label_num == 21:
    # 10%, 21 labeld
    labeled_list = read_list('labeled_10p')
    unlabeled_list = read_list('unlabeled_10p')
else:
    print('Error label_num!')
    os.exit()

eval_list = read_list('eval')
test_list = read_list('test')

snapshot_path = args.log_path + "/{}_{}_{}_{}".format(args.dataset_name, args.exp, args.label_num, args.remark)

num_classes = 16
patch_size = (96, 96, 96)

train_data_path = args.data_path
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size
con_w = args.contrast_w

if args.deterministic:
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + f"/log_{typename}.txt", mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(labeled_list, unlabeled_list, eval_list):
    scaler = GradScaler()

    train_list = labeled_list + unlabeled_list
    handler, sh = config_log(snapshot_path, 'train')
    logging.info(str(args))

    model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0])
    ema_model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0], ema=True)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)

    db_train = AMOS_fast(labeled_list, unlabeled_list,
                         base_dir=train_data_path,
                         transform=transforms.Compose([
                             RandomCrop(patch_size),
                             ToTensor(),
                         ]))

    labeled_idxs = list(range(len(unlabeled_list) * 2))
    unlabeled_idxs = list(range(len(unlabeled_list) * 2, len(unlabeled_list) * 4))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=8,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    writer = SummaryWriter(snapshot_path)
    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("Logs files: {} ".format(snapshot_path))

    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)
    con_loss = ContrastiveLoss()

    ema_model.train()

    iter_num = 0
    best_dice_avg = 0

    lr_ = base_lr
    metric_all_cases = None

    loc_list = cube_utils.get_loc_mask([4, 1, 96, 96, 96], cube_size)

    for epoch_num in tqdm(range(args.max_epoch + 1), dynamic_ncols=True, position=0):

        do_contrast = epoch_num > args.start_contrast_epoch
        # do_contrast = True

        for i_batch, sampled_batch in enumerate(trainloader):

            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                # 1️⃣  输入 
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()  # 已标注数据 及其标签

                unlabeled_volume_batch = volume_batch[labeled_bs:]  # 得到未标注数据

                model.train()
                outputs, features_embedding_list, text_embedding_list = model(volume_batch, do_contrast)  # 模型得到预测结果、特征嵌入、文本嵌入

                # 2️⃣ Cross-image Partition-and-Recovery
                batch_size, c, w, h, d = volume_batch.shape
                nb_cubes = h // cube_size
                cube_part_ind, cube_rec_ind, rand_loc_ind = cube_utils.get_part_and_rec_ind(
                    volume_shape=volume_batch.shape,
                    nb_cubes=nb_cubes,
                    nb_chnls=16,
                    rand_loc_ind=None
                )  # 这是得到打乱和恢复的index

                img_cross_mix = volume_batch.view(batch_size, c, w, h, d)
                img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)  # 这是打乱后的输入

                mix_features, mix_features_embedding_list, _, clip_embedding = model.module.forward_encoder(img_cross_mix, do_contrast)  # 得到打乱后输入的特征、特征嵌入、文本嵌入
                del img_cross_mix, _

                # unmix features
                # dim:64
                cube_rec_ind_3 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-3].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=64,
                                                                 rand_loc_ind=rand_loc_ind)[1]  # 获取恢复index

                # dim:128
                cube_rec_ind_2 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-2].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=128,
                                                                 rand_loc_ind=rand_loc_ind)[1]  # 获取恢复index

                # dim:256
                cube_rec_ind_1 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-1].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=256,
                                                                 rand_loc_ind=rand_loc_ind)[1]  # 获取恢复index

                mix_features_rec = [
                    torch.gather(mix_features[-3], dim=0, index=cube_rec_ind_3),
                    torch.gather(mix_features[-2], dim=0, index=cube_rec_ind_2),
                    torch.gather(mix_features[-1], dim=0, index=cube_rec_ind_1)
                ]  # 恢复特征

                if do_contrast:  # 这部分代码是为了对比学习准备特征嵌入
                    features_embedding_list[0] = torch.cat(
                        [features_embedding_list[0],
                         torch.gather(mix_features_embedding_list[0], dim=0, index=cube_rec_ind_3)])

                    features_embedding_list[1] = torch.cat(
                        [features_embedding_list[1],
                         torch.gather(mix_features_embedding_list[1], dim=0, index=cube_rec_ind_2)])

                    features_embedding_list[2] = torch.cat(
                        [features_embedding_list[2],
                         torch.gather(mix_features_embedding_list[2], dim=0, index=cube_rec_ind_1)])

                mix_embedding = model.module.forward_decoder(mix_features)  # 解码打乱后的特征得到嵌入

                embedding_rec = torch.gather(mix_embedding, dim=0, index=cube_rec_ind)  # 这是恢复嵌入
                del cube_part_ind, cube_rec_ind, cube_rec_ind_3, cube_rec_ind_2, cube_rec_ind_1

                outputs_unmix = model.module.forward_prediction_head(mix_features_rec,clip_embedding,embedding_rec)  # 得到恢复嵌入的预测结果

                # 3️⃣ Get pseudo-label from Teacher model  
                noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                entropy_threshold = np.log(2) * (0.75 + 0.25 * ramps.sigmoid_rampup(epoch_num,args.entropy_max_epoch))

                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)[0]
                    unlab_pl_soft = F.softmax(ema_output, dim=1)

                    # [2, 1, 96, 96, 96]
                    pseudo_label = torch.argmax(unlab_pl_soft, dim=1, keepdim=True).long()
                    entropy_map = -torch.sum(unlab_pl_soft * torch.log(unlab_pl_soft + 1e-6), dim=1, keepdim=True)

                loss_seg = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))# 已标注的CE损失
                outputs_soft = F.softmax(outputs, dim=1)# （已标注，未标注）的预测结果
                outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)# （已标注，未标注）cube 后的预测结果
                loss_seg_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs])# 已标注的dice损失
                loss_unmix_dice = dice_loss(outputs_unmix_soft[:labeled_bs], label_batch[:labeled_bs])# 已标注的cube dice损失
                supervised_loss = loss_seg + loss_seg_dice + loss_unmix_dice  # 已标注的总损失
                count_ss = 3

                # 4️⃣ Magic-cube Location Reasoning
                # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
                patch_list = cube_losses.get_patch_list(volume_batch, cube_size=cube_size)  # 得到所有cube块  patch_list[0].shape:  27 * torch.Size([4, 1, 1, 32, 32, 32])
                # idx = 27
                idx = torch.randperm(len(patch_list)).cuda()  # random, 这是打乱index
                # cube location loss
                loc_loss, feat_list, feat_embed_list = cube_losses.cube_location_loss(model.module, loc_list, patch_list, idx, do_contrast)  # 每个块的位置损失  
                """
                特征形状
                    # feat_list shape:  4 * torch.Size([27, 16, 32, 32, 32])
                    # feat_embed_list[0] shape: 4 * torch.Size([27, 64, 8, 8, 8])
                """

                # print("feat_embed_list shape:", feat_embed_list.__len__(), feat_embed_list[0].__len__(), feat_embed_list[0][0].shape)

                consistency_loss = 0
                count_consist = 1

                # 5️⃣ 已标注部分，Within-image Partition-and-Recovery 
                embed_list = []
                for i in range(labeled_bs):
                    embed_tmp = model.module.forward_decoder(feat_list[i])
                    embed_list.append(embed_tmp)
                embed_all = torch.stack(embed_list, dim=0)  # torch.Size([2, 27, 16, 32, 32, 32])
                embed_all_unmix = cube_losses.unmix_tensor(embed_all, volume_batch[:labeled_bs].shape)  # 2, 16, 96, 96, 96。 复原 已标注 

                feat_list_unmix = []
                for i in [3, 2, 1]:
                    f_list = []
                    for feat in feat_list[:labeled_bs]:
                        f_list.append(feat[-i])
                    f_list = torch.stack(f_list, dim=0)  # [2, 27, C, W/3, H/3, D/3]
                    feat_list_unmix.append(cube_losses.unmix_tensor(f_list, [labeled_bs, 1,
                                                                             f_list.shape[3] * 3,
                                                                             f_list.shape[4] * 3,
                                                                             f_list.shape[5] * 3]))

                pred_all_unmix = model.module.forward_prediction_head(feat_list_unmix,
                                                                      clip_embedding,
                                                                      embed_all_unmix)
                """
                Constructed feat_list_unmix, lengths:   3 torch.Size([2, 64, 24, 24, 24]) torch.Size([2, 128, 12, 12, 12]) torch.Size([2, 256, 6, 6, 6])
                clip_embedding.shape                    torch.Size([16, 256])
                """

                unmix_pred_soft = F.softmax(pred_all_unmix, dim=1)
                loss_lab_local_dice = dice_loss(unmix_pred_soft[:labeled_bs], label_batch[:labeled_bs])
                supervised_loss += loss_lab_local_dice
                count_ss += 1

                # 6️⃣ Cube-wise Pseudo-label Blending 
                pred_class_mix = None
                with torch.no_grad():
                    # To store some class pixels at the beginning of training to calculate the organ-class dist
                    if iter_num > 100:
                        # Get organ-class distribution
                        current_organ_dist = dist_logger.get_class_dist().cuda()  # (1, C)
                        # Normalize
                        current_organ_dist = current_organ_dist ** (1. / args.T_dist)
                        current_organ_dist = current_organ_dist / current_organ_dist.sum()
                        current_organ_dist = current_organ_dist / current_organ_dist.max()

                        weight_map = current_organ_dist[pseudo_label.squeeze()].unsqueeze(1).repeat(1, num_classes, 1, 1, 1)

                        unmix_pl = cube_losses.get_mix_pl(ema_model.module, feat_list[labeled_bs:],clip_embedding,volume_batch[labeled_bs:].shape, batch_size - labeled_bs)
                        unlab_pl_mix = (1. - weight_map) * ema_output + weight_map * unmix_pl
                        unlab_pl_mix_soft = F.softmax(unlab_pl_mix, dim=1)
                        mix_entropy_map = -torch.sum(unlab_pl_mix_soft * torch.log(unlab_pl_mix_soft + 1e-6), dim=1, keepdim=True)  # 2, 1, 96, 96, 96

                        _, pred_class_mix = torch.max(unlab_pl_mix_soft, dim=1)  # 2, 96, 96, 96

                        conf, pr_class = torch.max(unlab_pl_mix_soft.detach(), dim=1)  # 2, 96, 96, 96
                        dist_logger.append_class_list(pr_class.view(-1, 1))
                    else:
                        conf, pr_class = torch.max(unlab_pl_soft.detach(), dim=1)
                        dist_logger.append_class_list(pr_class.view(-1, 1))

                del volume_batch

                if iter_num % 20 == 0 and len(dist_logger.class_total_pixel_store):
                    dist_logger.update_class_dist()

                consistency_weight = get_current_consistency_weight(epoch_num)

                # debiase the pseudo-label: blend ema and unmixed_within pseudo-label
                if pred_class_mix is None:
                    consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pseudo_label)
                else:
                    consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pred_class_mix)

                consistency_loss += consistency_loss_unmix

                supervised_loss /= count_ss
                consistency_loss /= count_consist

                if do_contrast:
                    feat_embed_list_ = [torch.stack(feat_embed) for feat_embed in zip(feat_embed_list[0],
                                                                                      feat_embed_list[1],
                                                                                      feat_embed_list[2],
                                                                                      feat_embed_list[3])]
                    organized_idx = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
                    for i in range(len(features_embedding_list)):
                        feat_embed_list_[i] = cube_losses.unmix_tensor(feat_embed_list_[i],
                                                                       [1, feat_embed_list_[i].shape[2],
                                                                        feat_embed_list_[i].shape[3] * 3,
                                                                        feat_embed_list_[i].shape[4] * 3,
                                                                        feat_embed_list_[i].shape[5] * 3])
                        features_embedding_list[i] = torch.cat([features_embedding_list[i],
                                                                feat_embed_list_[i]], dim=0)[organized_idx]

                    # Dumb DP
                    for i in range(len(text_embedding_list)):
                        text_embedding_list[i] = text_embedding_list[i][:text_embedding_list[i].shape[0] // gpu_num]

                    loss_con = con_loss(features_embedding_list,
                                        torch.cat([
                                            label_batch[:labeled_bs].unsqueeze(1).repeat(3, 1, 1, 1, 1),
                                            pseudo_label.repeat(2, 1, 1, 1, 1), pred_class_mix.unsqueeze(1)
                                        ], dim=0),
                                        text_embedding_list,
                                        torch.cat([entropy_map.repeat(2, 1, 1, 1, 1), mix_entropy_map], dim=0),
                                        entropy_threshold,
                                        sample_num=args.contrast_sample_num
                                        )  # 可以考虑换对比学习的部分。
                else:
                    loss_con = 0

                # Final Loss
                loss = supervised_loss + \
                       0.1 * loc_loss + \
                       consistency_weight * consistency_loss + \
                       con_w * loss_con

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)  # EMA update
            
            # # LyapEMA update by gpt5.2
            # cur_lr = optimizer.param_groups[0]["lr"]
            # lyap_student_pull(model, ema_model, lr=cur_lr, lam=1.0)  # lam_stability 就是论文里的 λ
            # lyapema_teacher_update(model, ema_model, gamma=1e-4, sync_buffers=True)   

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info('iteration {}:, loss: {:.4f}, loss_sup: {:.4f}, '
                             'loss_consistency: {:.4f}, loss_loc: {:.4f}, loss_contrast: {:.4f}, '
                             'threshold: {:.6f}, lr: {:.6f}'.format(iter_num,
                                                                    loss,
                                                                    supervised_loss,
                                                                    consistency_weight * consistency_loss,
                                                                    0.1 * loc_loss,
                                                                    con_w * loss_con,
                                                                    entropy_threshold,
                                                                    lr_))
            lr_ = base_lr * (1.0 - iter_num / args.max_lr_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        # if epoch_num % 5 == 0 and epoch_num > 19:
        if epoch_num % 5 == 0:

            model.eval()
            dice_all, std_all, metric_all_cases = test_amos.validation_all_case_fast(model,
                                                                                     num_classes=num_classes,
                                                                                     base_dir=train_data_path,
                                                                                     image_list=eval_list,
                                                                                     patch_size=patch_size,
                                                                                     stride_xy=90,
                                                                                     stride_z=80)
            dice_avg = dice_all.mean()

            logging.info('epoch {}, '
                         'average DSC: {:.4f}, '
                         'spleen: {:.4f}, '
                         'r.kidney: {:.4f}, '
                         'l.kidney: {:.4f}, '
                         'gallbladder: {:.4f}, '
                         'esophagus: {:.4f}, '
                         'liver: {:.4f}, '
                         'stomach: {:.4f}, '
                         'aorta: {:.4f}, '
                         'inferior vena cava: {:.4f}, '
                         'pancreas: {:.4f}, '
                         'r.adrenal gland: {:.4f}, '
                         'l.adrenal gland: {:.4f}, '
                         'duodenum: {:.4f}, '
                         'bladder: {:.4f}, '
                         'prostate/uterus: {:.4f}'
                         .format(epoch_num,
                                 dice_avg,
                                 dice_all[0],
                                 dice_all[1],
                                 dice_all[2],
                                 dice_all[3],
                                 dice_all[4],
                                 dice_all[5],
                                 dice_all[6],
                                 dice_all[7],
                                 dice_all[8],
                                 dice_all[9],
                                 dice_all[10],
                                 dice_all[11],
                                 dice_all[12],
                                 dice_all[13],
                                 dice_all[14]))

            if dice_avg > best_dice_avg:
                best_dice_avg = dice_avg
                best_model_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}_best.pth'.format(epoch_num,
                                                                                                 str(best_dice_avg)[:6]
                                                                                                 ))
                torch.save(model.module.state_dict(), best_model_path)
                logging.info("save best model to {}".format(best_model_path))
            else:
                save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(epoch_num,
                                                                                           str(dice_avg)[:6]))
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            model.train()

    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle(f'amos{args.label_num}_{args.remark}')

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        print("Create new folder {}".format(snapshot_path))
    else:
        shutil.rmtree(snapshot_path)
        os.makedirs(snapshot_path)
    shutil.copytree(
            '.',snapshot_path + '/codes',ignore=shutil.ignore_patterns(
                '.git',
                '__pycache__',
                'CLIP',
                'data',
                'TAK_logs',
                '*.pt'
            )
        )
    metric_final, best_model_path = train(labeled_list, unlabeled_list, eval_list)

    save_best_path = best_model_path
    # save_best_path = "/home/why/TAK-Semi-main/TAK_logs/AMOS_MagicNet_GA_TAK_10_fuxian/epoch_145_dice_0.5449_best.pth"
    # save_best_path = "/home/why/TAK-Semi-main/TAK_logs/AMOS_MagicNet_GA_TAK_10_fuxian/epoch_145_dice_0.5449_best.pth"
    # save_best_path = "/data/why/logs//AMOS_Ours_10_fuse_new/epoch_85_dice_0.5426_best.pth"

    model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0])
    model.module.load_state_dict(torch.load(save_best_path, weights_only=True))
    model.eval()
    _, _, metric_final = test_amos.validation_all_case(model, num_classes=num_classes, base_dir=train_data_path,
                                                       image_list=test_list, patch_size=patch_size, stride_xy=32,
                                                       stride_z=16)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, \n'
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'r.adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'l.adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'duodenum: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'bladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'prostate/uterus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0],
                         metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1],
                         metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2],
                         metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3],
                         metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4],
                         metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5],
                         metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6],
                         metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7],
                         metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8],
                         metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9],
                         metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10],
                         metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11],
                         metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12],
                         metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12],
                         metric_mean[0][13], metric_std[0][13], metric_mean[1][13], metric_std[1][13],
                         metric_mean[2][13], metric_std[2][13], metric_mean[3][13], metric_std[3][13],
                         metric_mean[0][14], metric_std[0][14], metric_mean[1][14], metric_std[1][14],
                         metric_mean[2][14], metric_std[2][14], metric_mean[3][14], metric_std[3][14]))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
