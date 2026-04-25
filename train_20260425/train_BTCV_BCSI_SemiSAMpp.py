import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from utils import ramps, cube_losses, cube_utils, test_util
from utils import cube_losses_ori as cube_losses
from dataloaders.dataset import *
from dataloaders.mix_up import generate_mask_3D, get_entropy_map, get_cut_mask, multi_class_weit_loss, PolyWarmRestartScheduler, DiceLoss
from networks.BCSI import VNet_MoE
from loss_amos import GADice, GACE
from code_semisampp.semisam_plus import SAM_branch, SAM_init


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BTCV', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/BTCV/', help='Name of Dataset')
parser.add_argument('--log_path', type=str, default='/data/why/logs/', help='path to save')
parser.add_argument('--exp', type=str, default='BCSI', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=35000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=90, help='total samples of the dataset')
parser.add_argument('--max_epoch', type=int, default=7000, help='max_epoch')
parser.add_argument('--max_train_samples', type=int, default=66, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=22, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--label_num', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
parser.add_argument('--remark', type=str, default='baseline', help='exp_name')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def read_list(split):
    ids_list = np.loadtxt(
        # os.path.join(args.root_path, 'split_txts/', f'{split}.txt'),
        os.path.join(args.root_path, 'split_txt/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)

if args.label_num == 30:
    labeled_list = read_list('labeled_30p')
    unlabeled_list = read_list('unlabeled_30p')
elif args.label_num == 40:
    labeled_list = read_list('labeled_40p')
    unlabeled_list = read_list('unlabeled_40p')
elif args.label_num == 10:
    labeled_list = read_list('labeled_10p')
    unlabeled_list = read_list('unlabeled_10p')
elif args.label_num == 50:
    labeled_list = read_list('labeled_50p')
    unlabeled_list = read_list('unlabeled_50p')
else:
    print('Error label_num!')
    os.exit()

test_list = read_list('test')

snapshot_path = args.log_path + "/BCSI_{}_{}_{}".format(args.dataset_name, args.label_num, args.remark)

num_classes = 14
class_momentum = 0.999
patch_size = (96, 96, 96)

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(labeled_list, unlabeled_list):
    train_list = labeled_list + unlabeled_list
    handler, sh = config_log(snapshot_path, 'train')
    logging.info(str(args))
    model = VNet_MoE(n_channels=1, n_classes=num_classes)
    model = model.cuda()
    db_train = BTCV_ws(train_list,
                    base_dir=train_data_path,
                    patch_size=patch_size)
    labeled_idxs = list(range(0, len(labeled_list)))
    unlabeled_idxs = list(range(len(labeled_list), len(train_list)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, 
        unlabeled_idxs, 
        args.batch_size, 
        args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    sam_finetune = SAM_init(generalist='SAM-Med3D', device='cuda')

    writer = SummaryWriter(snapshot_path)
    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("Logs files: {} ".format(snapshot_path))

    # dice_loss = GADice()
    # ce_loss = GACE(k=10, gama=0.5)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.00001)
        
    scheduler = PolyWarmRestartScheduler(
        optimizer,
        base_lr=args.base_lr,
        max_iters=args.max_iteration,
        power=0.9,
        warm_restart_iters=8000
    )

    dice_loss = DiceLoss(num_classes)
    ce_loss = nn.CrossEntropyLoss()
    pixel_ce_loss = nn.CrossEntropyLoss(reduction='none')


    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    lr_ = base_lr
    loc_list = None

    for epoch_num in tqdm(range(args.max_epoch + 1), dynamic_ncols=True, position=0):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, strong_aug_volume_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['strong_aug']

            weak_lab_img = volume_batch[:labeled_bs].cuda()
            weak_lab_label = label_batch[:labeled_bs].long().cuda()
            weak_unlab_img = volume_batch[labeled_bs:].cuda()

            # 这里如果你后面真的有 strong_aug 的 labeled 图，建议改成 strong_aug_volume_batch[:labeled_bs]
            strong_lab_img = volume_batch[:labeled_bs].cuda()
            strong_lab_label = label_batch[:labeled_bs].long().cuda()
            strong_unlab_img = strong_aug_volume_batch[labeled_bs:].cuda()

            
            model.train()
            
            # copy paste
            mask, _ = generate_mask_3D(weak_lab_img, mask_ratio=2/3)  # [B_l,1,D,H,W] 最好保证是这个形状

            # lab-unlab copy paste
            weak_lab_strong_unlab_img = weak_lab_img * mask + strong_unlab_img * (1 - mask)
            weak_unlab_strong_lab_img = weak_unlab_img * mask + strong_lab_img * (1 - mask)
            lab_unlab_cp_img = torch.cat([weak_lab_strong_unlab_img, weak_unlab_strong_lab_img], dim=0)

            # --------------------------------------------------
            # 1) weak forward：先跑，用于生成 unlabeled pseudo
            #    这里 lab_label 可以传，unlab_pseudo 先没有
            # --------------------------------------------------
            weak_input = torch.cat([weak_lab_img, weak_unlab_img], dim=0)
            weak_pred, weak_features, weak_mask = model(
                weak_input,
                labeled_bs=labeled_bs,
                lab_label=weak_lab_label,
                unlab_pseudo=None,
                is_training=True
            )

            # weak_uncertainty_map = get_entropy_map(weak_pred)

            # pseudo map：多类别版本建议直接 argmax，不要再用老的二分类 get_cut_mask
            # SAM
            samseg_soft, uncsam = SAM_branch(weak_unlab_img, weak_pred[labeled_bs:].detach(), sam_finetune, prompt=args.prompt)
            weak_prob = torch.softmax(weak_pred, dim=1)
            weak_uncertainty_map = torch.mean((weak_prob - samseg_soft)**2, dim=1)  # 不一致性作为 uncertainty
            weak_prob = torch.softmax(weak_pred[labeled_bs:].detach(), dim=1)
            w = torch.exp(-weak_uncertainty_map[labeled_bs:])  # UNet confidence
            fused_prob = w * weak_prob + (1 - w) * samseg_soft
            unlab_pseudo = torch.argmax(fused_prob, dim=1)

            # --------------------------------------------------
            # 2) strong forward：现在可以把 unlab_pseudo 传进 model
            # --------------------------------------------------
            strong_input = torch.cat([strong_lab_img, strong_unlab_img], dim=0)
            strong_pred, strong_features, strong_mask = model(
                strong_input,
                labeled_bs=labeled_bs,
                lab_label=strong_lab_label,
                unlab_pseudo=unlab_pseudo,
                is_training=True
            )

            strong_uncertainty_map = get_entropy_map(strong_pred)

            # --------------------------------------------------
            # 3) mixed forward：同样传入 lab_label / unlab_pseudo
            # --------------------------------------------------
            mixed_pred, mixed_features, mixed_mask = model(
                lab_unlab_cp_img,
                labeled_bs=labeled_bs,
                lab_label=weak_lab_label,
                unlab_pseudo=unlab_pseudo,
                is_training=True
            )

            mixed_uncertainty_map = get_entropy_map(mixed_pred)

            # --------------------------------------------------
            # 4) restore mixed pred
            # --------------------------------------------------
            mixed_lab_pred = mixed_pred[:labeled_bs] * mask + mixed_pred[labeled_bs:] * (1 - mask)
            mixed_unlab_pred = mixed_pred[:labeled_bs] * (1 - mask) + mixed_pred[labeled_bs:] * mask

            mask_u = mask.squeeze(1)   # [B_l,D,H,W]
            mixed_lab_uncertainty_map = mixed_uncertainty_map[:labeled_bs] * mask_u + \
                                        mixed_uncertainty_map[labeled_bs:] * (1 - mask_u)

            mixed_unlab_uncertainty_map = mixed_uncertainty_map[:labeled_bs] * (1 - mask_u) + \
                                        mixed_uncertainty_map[labeled_bs:] * mask_u
            
            

            # --------------------------------------------------
            # 5) supervised loss
            # --------------------------------------------------
            sup_loss = 0.0
            sup_loss += multi_class_weit_loss(
                weak_pred[:labeled_bs], weak_lab_label, weak_uncertainty_map[:labeled_bs]
            )
            sup_loss += multi_class_weit_loss(
                strong_pred[:labeled_bs], strong_lab_label, strong_uncertainty_map[:labeled_bs]
            )
            sup_loss += multi_class_weit_loss(
                mixed_lab_pred, weak_lab_label, mixed_lab_uncertainty_map
            )

            # --------------------------------------------------
            # 6) unsupervised loss
            # --------------------------------------------------
            unsup_loss = 0.0
            unsup_loss += multi_class_weit_loss(
                strong_pred[labeled_bs:], unlab_pseudo, strong_uncertainty_map[labeled_bs:]
            )
            unsup_loss += multi_class_weit_loss(
                mixed_unlab_pred, unlab_pseudo, mixed_unlab_uncertainty_map
            )

            # --------------------------------------------------
            # 7) consistency loss
            # --------------------------------------------------
            mixed_pred_soft = torch.softmax(torch.cat([mixed_lab_pred, mixed_unlab_pred], dim=0), dim=1)
            cons_loss = F.mse_loss(mixed_pred_soft, torch.softmax(strong_pred, dim=1))
            sam_loss = F.mse_loss(
                torch.softmax(weak_pred[labeled_bs:], dim=1),
                samseg_soft.detach()
            )
            # --------------------------------------------------
            # 8) total loss
            # --------------------------------------------------
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = sup_loss + cons_loss + consistency_weight * unsup_loss + sam_loss

            # --------------------------------------------------
            # 9) save to queue
            # --------------------------------------------------
            model.interpolation_save(
                weak_features[:labeled_bs], weak_mask[:labeled_bs], weak_lab_label, is_lab=True
            )
            model.interpolation_save(
                weak_features[labeled_bs:], weak_mask[labeled_bs:], unlab_pseudo, is_lab=False
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.3f}, '
                             'current_lr: {:.3f}, loss_weight: {:f}, '.format(1, iter_num,
                                                       loss,
                                                       current_lr,
                                                       consistency_weight))

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            if epoch_num % 100 == 0 and epoch_num > 190:
                model.eval()
                dice_all, std_all, metric_all_cases = test_util.validation_all_case_btcv(model,
                                                                                    num_classes=num_classes,
                                                                                    base_dir=train_data_path,
                                                                                    image_list=test_list,
                                                                                    patch_size=patch_size,
                                                                                    stride_xy=90,
                                                                                     stride_z=80)
                dice_avg = dice_all.mean()

                logging.info('iteration {}, '
                             'average DSC: {:.3f}, '
                             'spleen: {:.3f}, '
                             'r.kidney: {:.3f}, '
                             'l.kidney: {:.3f}, '
                             'gallbladder: {:.3f}, '
                             'esophagus: {:.3f}, '
                             'liver: {:.3f}, '
                             'stomach: {:.3f}, '
                             'aorta: {:.3f}, '
                             'inferior vena cava: {:.3f}'
                             'portal vein and splenic vein: {:.3f}, '
                             'pancreas: {:.3f}, '
                             'right adrenal gland: {:.3f}, '
                             'left adrenal gland: {:.3f}'
                             .format(iter_num,
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
                                     dice_all[12]))

                if dice_avg > best_dice_avg:
                    best_dice_avg = dice_avg
                    best_model_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_best.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model.state_dict(), best_model_path)
                    logging.info("save best model to {}".format(best_model_path))
                else:
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                
                model.train()
    
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle(f'BCSI_BTCV_{args.label_num}_{args.remark}')

    # 1. 确保实验目录存在
    os.makedirs(snapshot_path, exist_ok=True)

    code_path = os.path.join(snapshot_path, 'code')

    # 2. 如果 code 目录存在，先删掉
    if os.path.exists(code_path):
        shutil.rmtree(code_path)

    # 3. 重新创建空的 code 目录
    os.makedirs(code_path)

    print("Prepare code folder: {}".format(code_path))

    # 4. 复制代码文件
    shutil.copyfile("/home/why/TAK-Semi-main/networks/BCSI.py",
                    os.path.join(code_path, "BCSI.py"))
    shutil.copyfile("/home/why/TAK-Semi-main/train_BTCV_BCSI.py",
                    os.path.join(code_path, "train_BTCV_BCSI.py"))
    
    metric_final, best_model_path = train(labeled_list, unlabeled_list)

    save_best_path = best_model_path
    # save_best_path = '/data/why/logs//GA_BTCV_GA_10_new_data/iter_05201_dice_0.591254_best.pth' 

    model = VNet_MoE(n_channels=1, n_classes=num_classes)
    model.load_state_dict(torch.load(save_best_path, weights_only=True))
    model.eval()
    _, _, metric_final = test_util.validation_all_case_btcv(model, num_classes=num_classes, base_dir=train_data_path,
                                                       image_list=test_list, patch_size=patch_size, stride_xy=16,
                                                            stride_z=16, save_nii_dir=snapshot_path)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}+-{:.4f},, HD95: {:.4f}+-{:.4f},, JI: {:.4f}+-{:.4f},, ASD: {:.4f}+-{:.4f},, '
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'portal and splenic vein: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'right adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'Left adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_std[0].mean(), metric_mean[1].mean(), metric_std[1].mean(), metric_mean[2].mean(), metric_std[2].mean(), metric_mean[3].mean(), metric_std[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0], metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1], metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2], metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3], metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4], metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5], metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6], metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7], metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8], metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9], metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10], metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11], metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12], metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12]))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

# CUDA_VISIBLE_DEVICES=0 python train_BTCV_BCSI.py --seed 1337 --label_num 10 --remark "baseline"
# CUDA_VISIBLE_DEVICES=0 python train_BTCV_BCSI.py --seed 1337 --label_num 50 --remark "baseline"