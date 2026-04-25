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

from loss_amos import GADice, GACE, ContrastiveLoss, AnatomyWeightScheduler
from dataloaders.dataset import *
from networks.baseline import *
from utils import ramps, test_amos, cube_losses, cube_utils

from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--data_path', type=str, default='/data/why/Datasets/amos/', help='data_path')
parser.add_argument('--log_path', type=str, default='/data/why/logs/', help='log_path')
parser.add_argument('--exp', type=str, default='baseline', help='exp_name')
parser.add_argument('--remark', type=str, default='ours1', help='remark')
# parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--deterministic', type=bool, default=True, help='whether to use deterministic training')
parser.add_argument('--label_num', type=int, default=10, help='labeled trained samples ratios')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')

parser.add_argument('--max_epoch', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base_lr')
parser.add_argument('--sem_align_w', type=float, default=0.1, help='semantic alignment weight')
parser.add_argument('--sem_temp', type=float, default=0.07, help='sem_temp weight')
parser.add_argument('--sem_pl_w', type=float, default=1.0, help='sem_pl_w weight')
parser.add_argument('--sem_pl_min', type=float, default=0.2, help='sem_pl_min weight')
parser.add_argument('--sem_pl_max', type=float, default=1.0, help='sem_pl_max weight')
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
        # os.path.join(args.data_path, 'split_txts/', f'{split}.txt'),
        os.path.join(args.data_path, 'split_txt/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


if args.label_num == 2:
    # 2%, 4 labeld
    labeled_list = read_list('labeled_2p')
    unlabeled_list = read_list('unlabeled_2p')
elif args.label_num == 5:
    # 5%, 10 labeld
    labeled_list = read_list('labeled_5p')
    unlabeled_list = read_list('unlabeled_5p')
elif args.label_num == 10:
    # 10%, 21 labeld
    labeled_list = read_list('labeled_10p')
    unlabeled_list = read_list('unlabeled_10p')
elif args.label_num == 20:
    # 20%, 38 labeld 
    labeled_list = read_list('labeled_20p')
    unlabeled_list = read_list('unlabeled_20p')
elif args.label_num == 50:
    # 50%, 96 labeld 
    labeled_list = read_list('labeled_50p')
    unlabeled_list = read_list('unlabeled_50p')
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

    weight_scheduler = AnatomyWeightScheduler(
        total_epochs=args.max_epoch, 
        warmup_epochs=args.start_contrast_epoch, # 假设前 20% 的时间用来打基础
        # warmup_epochs=10, # 假设前 20% 的时间用来打基础
        target_rel_weight=0.5 # 最终关系损失占类别损失的一半权重
    )

    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)

    ema_model.train()

    iter_num = 0
    best_dice_avg = 0

    lr_ = base_lr
    metric_all_cases = None

    loc_list = cube_utils.get_loc_mask([4, 1, 96, 96, 96], cube_size)

    for epoch_num in tqdm(range(args.max_epoch + 1), dynamic_ncols=True, position=0):

        do_contrast = False
        for i_batch, sampled_batch in enumerate(trainloader):

            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                # 输入 1️⃣
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                unlabeled_volume_batch = volume_batch[labeled_bs:]

                model.train()
                outputs, _, _ = model(volume_batch, do_contrast)

                # Cross-image Partition-and-Recovery  2️⃣
                batch_size, c, w, h, d = volume_batch.shape
                nb_cubes = h // cube_size
                cube_part_ind, cube_rec_ind, rand_loc_ind = cube_utils.get_part_and_rec_ind(
                    volume_shape=volume_batch.shape,
                    nb_cubes=nb_cubes,
                    nb_chnls=16,
                    rand_loc_ind=None
                )

                img_cross_mix = volume_batch.view(batch_size, c, w, h, d)
                img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)

                mix_features, _, _, _ = model.module.forward_encoder(img_cross_mix, do_contrast)  # 做混合后的预测
                del img_cross_mix, _

                # unmix features
                # dim:64
                cube_rec_ind_3 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-3].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=64,
                                                                 rand_loc_ind=rand_loc_ind)[1]

                # dim:128
                cube_rec_ind_2 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-2].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=128,
                                                                 rand_loc_ind=rand_loc_ind)[1]

                # dim:256
                cube_rec_ind_1 = cube_utils.get_part_and_rec_ind(volume_shape=mix_features[-1].shape,
                                                                 nb_cubes=nb_cubes,
                                                                 nb_chnls=256,
                                                                 rand_loc_ind=rand_loc_ind)[1]

                mix_embedding = model.module.forward_decoder(mix_features)

                embedding_rec = torch.gather(mix_embedding, dim=0, index=cube_rec_ind)
                del cube_part_ind, cube_rec_ind, cube_rec_ind_3, cube_rec_ind_2, cube_rec_ind_1

                outputs_unmix = model.module.out_(embedding_rec)

                # Get pseudo-label from teacher model   3️⃣
                noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                entropy_threshold = np.log(2) * (0.75 + 0.25 * ramps.sigmoid_rampup(epoch_num,args.entropy_max_epoch))

                with torch.no_grad():
                    ema_output, _, _ = ema_model(ema_inputs, False)
                    unlab_pl_soft = F.softmax(ema_output, dim=1)
                    pseudo_label = torch.argmax(unlab_pl_soft, dim=1, keepdim=True).long()

                loss_seg = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))# 已标注的CE损失
                outputs_soft = F.softmax(outputs, dim=1)# （已标注，未标注）的预测结果
                outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)# （已标注，未标注）cube 后的预测结果
                loss_seg_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs])# 已标注的dice损失
                loss_unmix_dice = dice_loss(outputs_unmix_soft[:labeled_bs], label_batch[:labeled_bs])# 已标注的cube dice损失
                supervised_loss = loss_seg + loss_seg_dice + loss_unmix_dice  # 已标注的总损失
                count_ss = 3

                # Magic-cube Location Reasoning  4️⃣
                # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
                patch_list = cube_losses.get_patch_list(volume_batch, cube_size=cube_size)
                # idx = 27
                idx = torch.randperm(len(patch_list)).cuda()  # random
                # cube location loss
                loc_loss, feat_list, feat_embed_list = cube_losses.cube_location_loss(model.module, loc_list,
                                                                                      patch_list, idx, do_contrast)

                consistency_loss = 0
                count_consist = 1

                # Within-image Partition-and-Recovery  5️⃣
                embed_list = []
                for i in range(labeled_bs):
                    embed_tmp = model.module.forward_decoder(feat_list[i])
                    embed_list.append(embed_tmp)
                embed_all = torch.stack(embed_list, dim=0)  # torch.Size([2, 27, 16, 32, 32, 32])
                embed_all_unmix = cube_losses.unmix_tensor(embed_all, volume_batch[:labeled_bs].shape)  # 2, 16, 96, 96, 96

                pred_all_unmix = model.module.out_(embed_all_unmix)
                
                unmix_pred_soft = F.softmax(pred_all_unmix, dim=1)
                loss_lab_local_dice = dice_loss(unmix_pred_soft[:labeled_bs], label_batch[:labeled_bs])
                supervised_loss += loss_lab_local_dice
                count_ss += 1

                # Cube-wise Pseudo-label Blending  6️⃣
                pred_class_mix = None
                with torch.no_grad():
                    # To store some class pixels at the beginning of training to calculate the organ-class dist
                    if iter_num > 100:
                    # if iter_num > 5:
                        # Get organ-class distribution
                        current_organ_dist = dist_logger.get_class_dist().cuda()  # (1, C)
                        # Normalize
                        current_organ_dist = current_organ_dist ** (1. / args.T_dist)
                        current_organ_dist = current_organ_dist / current_organ_dist.sum()
                        current_organ_dist = current_organ_dist / current_organ_dist.max()

                        weight_map = current_organ_dist[pseudo_label.squeeze()].unsqueeze(1).repeat(1, num_classes,
                                                                                                    1, 1, 1)

                        unmix_pl = cube_losses.get_mix_pl(ema_model.module, feat_list[labeled_bs:],None,
                                                          volume_batch[labeled_bs:].shape, batch_size - labeled_bs)
                        unlab_pl_mix = (1. - weight_map) * ema_output + weight_map * unmix_pl
                        unlab_pl_mix_soft = F.softmax(unlab_pl_mix, dim=1)

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

                consistency_loss_unmix = consistency_loss_unmix
                consistency_loss += consistency_loss_unmix
                
                supervised_loss /= count_ss
                consistency_loss /= count_consist

                loss_con = 0

                # Final Loss
                loss = supervised_loss + 0.1*loc_loss + consistency_weight*consistency_loss + con_w*loss_con

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)  # EMA update
            
            iter_num = iter_num + 1

            if iter_num % 100 == 0:
            # if iter_num % 10 == 0:
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
        if epoch_num % 20 == 0:

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
    # import setproctitle
    # setproctitle.setproctitle(f'amos{args.label_num}_{args.remark}')

    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    # else:
    #     shutil.rmtree(snapshot_path)
    #     os.makedirs(snapshot_path)
    # shutil.copytree(
    #         '.',snapshot_path + '/codes',ignore=shutil.ignore_patterns(
    #             '.git',
    #             '__pycache__',
    #             'CLIP',
    #             'data',
    #             'TAK_logs',
    #             '*.pt'
    #         )
    #     )
    
    # metric_final, best_model_path = train(labeled_list, unlabeled_list, eval_list)

    # save_best_path = best_model_path
    # save_best_path = "/data/why/logs/AMOS_baseline_10_Baseline/epoch_140_dice_0.7012_best.pth"
    save_best_path = "/data/why/logs/AMOS_baseline_50_Baseline/epoch_140_dice_0.7411_best.pth"

    model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0])
    model.module.load_state_dict(torch.load(save_best_path, weights_only=True))
    model.eval()

    _, _, metric_final = test_amos.validation_all_case(model, num_classes=num_classes, base_dir=train_data_path,
                                                       image_list=test_list, patch_size=patch_size, stride_xy=32,
                                                       stride_z=16, save_nii_dir=snapshot_path)
    # -------------------------------------------------------------------
    # 1. 计算每个器官的统计量 (保持你原有的逻辑，用于展示每个器官的详细数据)
    # -------------------------------------------------------------------
    # 形状: (N_metrics, N_classes)
    class_mean = np.mean(metric_final, axis=0) 
    class_std  = np.std(metric_final, axis=0)
    
    # -------------------------------------------------------------------
    # 2. 计算【Final Average】的正确逻辑 (修正点)
    # -------------------------------------------------------------------
    # 第一步：先计算每个病人(Case)的平均分 (沿着器官维度 axis=2 取平均)
    # 形状变成: (N_cases, N_metrics) -> 每个病人在 Dice/HD95 上的平均表现
    case_level_mean = np.mean(metric_final, axis=2) 
    
    # 第二步：计算所有病人平均分的 均值 和 标准差
    # 形状变成: (N_metrics, )
    final_mean = np.mean(case_level_mean, axis=0)  # 这是最终的平均 Dice
    final_std  = np.std(case_level_mean, axis=0)   # 这是最终的 Dice 标准差 (对应表格中的 ±11.35)

    # -------------------------------------------------------------------
    # 3. 保存与打印
    # -------------------------------------------------------------------
    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    
    # 优化后的打印代码 (使用 final_mean 和 final_std)
    logging.info('Final Average DSC:{:.4f}±{:.4f}, HD95: {:.4f}±{:.4f}, JI: {:.4f}±{:.4f}, ASD: {:.4f}±{:.4f}'.format(
        final_mean[0], final_std[0],  # Dice
        final_mean[1], final_std[1],  # HD95
        final_mean[2], final_std[2],  # JI
        final_mean[3], final_std[3]   # ASD
    ))
    
    # 打印每个器官的详细数据 (这部分逻辑保持不变，依然使用 class_mean 和 class_std)
    organ_names = [
        'spleen', 'r.kidney', 'l.kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 
        'aorta', 'ivc', 'pancreas', 'right adrenal gland', 'left adrenal gland', 
        'duodenum', 'bladder', 'prostate/uterus'
    ]
    
    log_str = ""
    for i, organ in enumerate(organ_names):
        # 确保索引不越界 (防止 organ_names 比实际类别多)
        if i < class_mean.shape[1]: 
            log_str += '\n{}: {:.4f}±{:.4f}, {:.4f}±{:.4f}, {:.4f}±{:.4f}, {:.4f}±{:.4f},'.format(
                organ,
                class_mean[0][i], class_std[0][i], # Dice
                class_mean[1][i], class_std[1][i], # HD95
                class_mean[2][i], class_std[2][i], # JI
                class_mean[3][i], class_std[3][i]  # ASD
            )
    logging.info(log_str)

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

# CUDA_VISIBLE_DEVICES=0 python train_AMOS_Baseline.py --seed 1337 --label_num 10 --remark "Baseline"
# CUDA_VISIBLE_DEVICES=1 python train_AMOS_Baseline.py --seed 1337 --label_num 50 --remark "Baseline"
