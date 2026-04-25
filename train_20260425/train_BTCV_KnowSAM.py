import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloaders.dataset import *
from torch.utils.data import DataLoader
from trainer import Trainer
from torchvision import transforms




parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/BTCV',
                    help='Name of Experiment')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='Percentage of label quantity')
parser.add_argument('--dataset_name', type=str, default='BTCV',
                    help='Name of Experiment')

parser.add_argument('--remark', type=str,  default="TMI25", help='remark')
parser.add_argument('--num_classes', type=int,  default=14,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=3,
                    help='input channel of network')
parser.add_argument('--exp', type=str,
                    default='KnowSAM', help='experiment_name')
parser.add_argument('--save_best_path', type=str,
                    default='', help='model path')

parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-UNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-VNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--image_size', type=int, default=128, help='image_size')
parser.add_argument('--point_nums', type=int, default=5, help='points number')
parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
parser.add_argument('-thd', type=bool, default=False, help='3d or not')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--batch_size_all', type=int, default=4,
                    help='batch_size per gpu on 体素')
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_bs_all', type=int, default=2,
                    help='labeled_batch_size per gpu on 体素')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--mixed_iterations', type=int, default=12000,
                    help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="sam checkpoint")


args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def read_list(split):  # 对应 DVCL 的数据标准
    ids_list = np.loadtxt(
        os.path.join(args.root_path, 'split_txt/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)

def read_list_s(split):  # 对应 MagicNet 的数据标准
    ids_list = np.loadtxt(
        os.path.join(args.root_path, 'split_txts/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)




def _sample_fixed_indices(indices, k, fallback_indices=None):
    """
    从 indices 中固定采样 k 个。
    - 如果 indices 足够，随机无放回采样
    - 如果不足，先全部保留，再重复已有样本补齐
    - 如果 indices 为空，则从 fallback_indices 里采样
    """
    indices = list(indices)

    if len(indices) >= k:
        return random.sample(indices, k)

    if len(indices) > 0:
        out = indices.copy()
        while len(out) < k:
            out.append(random.choice(indices))
        return out

    # indices 为空时，用 fallback
    fallback_indices = list(fallback_indices) if fallback_indices is not None else []
    if len(fallback_indices) == 0:
        raise ValueError("Both indices and fallback_indices are empty.")

    if len(fallback_indices) >= k:
        return random.sample(fallback_indices, k)

    out = fallback_indices.copy()
    while len(out) < k:
        out.append(random.choice(fallback_indices))
    return out


def build_2d_batch_from_3d(
    volume_batch,              # [B,1,Z,H,W]
    label_batch,               # [B,Z,H,W]
    num_labeled_volumes=2,
    k_labeled=8,
    k_unlabeled=8,
):
    """
    固定输出数量的 2D slice batch。

    返回:
        image_slices:      [N,3,H,W]
        label_slices:      [N,H,W]
        labeled_slice_bs:  int

    规则:
    - 前 num_labeled_volumes 个 volume 视为 labeled
    - 每个 labeled volume 固定取 k_labeled 张
      优先取前景 slice；不够则重复前景 slice；若完全没有前景，则从全 volume 补
    - 每个 unlabeled volume 固定取 k_unlabeled 张
      优先取前景 slice；不够则重复；若完全没有前景，则从全 volume 补
    """
    B, C, Z, H, W = volume_batch.shape
    assert C == 1, f"Expected single-channel volume, got C={C}"

    labeled_images = []
    labeled_labels = []
    unlabeled_images = []
    unlabeled_labels = []

    all_z = list(range(Z))

    for b in range(B):
        vol = volume_batch[b]   # [1,Z,H,W]
        lab = label_batch[b]    # [Z,H,W]

        # 有前景的 slice
        fg_z = [z for z in range(Z) if torch.any(lab[z] > 0)]

        if b < num_labeled_volumes:
            chosen_z = _sample_fixed_indices(
                indices=fg_z,
                k=k_labeled,
                fallback_indices=all_z
            )

            for z in chosen_z:
                img_slice = vol[:, z, :, :].repeat(3, 1, 1)   # [3,H,W]
                lbl_slice = lab[z]                             # [H,W]
                labeled_images.append(img_slice)
                labeled_labels.append(lbl_slice)

        else:
            chosen_z = _sample_fixed_indices(
                indices=fg_z,
                k=k_unlabeled,
                fallback_indices=all_z
            )

            for z in chosen_z:
                img_slice = vol[:, z, :, :].repeat(3, 1, 1)   # [3,H,W]
                lbl_slice = lab[z]                             # [H,W]
                unlabeled_images.append(img_slice)
                unlabeled_labels.append(lbl_slice)

    image_slices = torch.stack(labeled_images + unlabeled_images, dim=0)   # [N,3,H,W]
    label_slices = torch.stack(labeled_labels + unlabeled_labels, dim=0)   # [N,H,W]
    labeled_slice_bs = len(labeled_images)

    return image_slices, label_slices, labeled_slice_bs


# def build_2d_batch_from_3d(volume_batch, label_batch, num_labeled_volumes=2, k_labeled=8, k_unlabeled=8):
#     """
#     volume_batch: [B,1,Z,H,W]
#     label_batch:  [B,Z,H,W]
#     return:
#         image_slices: [N,3,H,W]
#         label_slices: [N,H,W]
#         labeled_slice_bs: int
#     """
#     B, C, Z, H, W = volume_batch.shape

#     labeled_images = []
#     labeled_labels = []
#     unlabeled_images = []
#     unlabeled_labels = []

#     for b in range(B):
#         vol = volume_batch[b]        # [1,Z,H,W]
#         lab = label_batch[b]         # [Z,H,W]

#         if b < num_labeled_volumes:
#             valid_z = [z for z in range(Z) if torch.any(lab[z] > 0)]
#             if len(valid_z) == 0:
#                 valid_z = [Z // 2]

#             if len(valid_z) > k_labeled:
#                 step = max(1, len(valid_z) // k_labeled)
#                 valid_z = valid_z[::step][:k_labeled]

#             for z in valid_z:
#                 img_slice = vol[:, z, :, :].repeat(3, 1, 1)   # [3,H,W]
#                 lbl_slice = lab[z]                             # [H,W]
#                 labeled_images.append(img_slice)
#                 labeled_labels.append(lbl_slice)

#         else:
#             sample_z = list(range(0, Z, max(1, Z // k_unlabeled)))[:k_unlabeled]
#             for z in sample_z:
#                 img_slice = vol[:, z, :, :].repeat(3, 1, 1)
#                 lbl_slice = lab[z]
#                 unlabeled_images.append(img_slice)
#                 unlabeled_labels.append(lbl_slice)

#     image_slices = torch.stack(labeled_images + unlabeled_images, dim=0)   # [N,3,H,W]
#     label_slices = torch.stack(labeled_labels + unlabeled_labels, dim=0)   # [N,H,W]
#     labeled_slice_bs = len(labeled_images)

#     return image_slices, label_slices, labeled_slice_bs

def train(args, snapshot_path):

    max_iterations = args.max_iterations
    # model
    train_data_path = args.root_path
    trainer = Trainer(args)
    train_list = labeled_list + unlabeled_list
    db_train = BTCV(train_list,
                    base_dir=train_data_path,
                    transform=transforms.Compose([
                        RandomCrop(args.patch_size),
                        ToTensor(),
                    ]))
    labeled_idxs = list(range(0, len(labeled_list)))
    unlabeled_idxs = list(range(len(labeled_list), len(train_list)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, 
        unlabeled_idxs, 
        args.batch_size_all, 
        args.batch_size_all - args.labeled_bs_all
    )

    # dataloader
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    logging.info("{} iterations per epoch".format(len(train_loader)))

    # 训练时，不注释，测试时注释
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = 0
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            image_slices, label_slices, _ = build_2d_batch_from_3d(
                    volume_batch, label_batch,
                    num_labeled_volumes=args.labeled_bs_all,
                    k_labeled=8,
                    k_unlabeled=8,
                )

            trainer.train(image_slices, label_slices, iter_num)
            if iter_num > 0 and iter_num % 5000 == 0:
                trainer.save_model(snapshot_path, f"{iter_num}")
            iter_num = iter_num + 1

    logging.info(f"Test....")
    _, _, metric_final = trainer.val_BTCV(args.root_path, test_list, args.num_classes, patch_size=(96, 96, 96), stride_xy=16, stride_z=16, save_nii_dir=snapshot_path)
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)
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
    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_final)

if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        torch.autograd.set_detect_anomaly(True)
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        torch.cuda.manual_seed(2024)
        train_data_path = args.root_path

        base_lr = args.lr
        labeled_bs = args.labeled_bs

        labeled_list, unlabeled_list =[], []
        if args.labeled_num == 30:
            labeled_list = read_list_s('labeled_30p')
            unlabeled_list = read_list_s('unlabeled_30p')
        elif args.labeled_num == 40:
            labeled_list = read_list_s('labeled_40p')
            unlabeled_list = read_list_s('unlabeled_40p')
        elif args.labeled_num == 10:
            labeled_list = read_list('labeled_10p')
            unlabeled_list = read_list('unlabeled_10p')
        elif args.labeled_num == 50:
            labeled_list = read_list('labeled_50p')
            unlabeled_list = read_list('unlabeled_50p')
        else:
            print('Error labeled_num!')
            os.exit()

        eval_list = read_list('eval')
        test_list = read_list('test')
        snapshot_path = "/data/why/logs_SAM/KnowSAM_{}_{}_{}_{}".format(args.exp, args.labeled_num, fold, args.remark)
        # if not os.path.exists(snapshot_path):
        #     os.makedirs(snapshot_path)
        # if os.path.exists(snapshot_path + '/code'):
        #     shutil.rmtree(snapshot_path + '/code')
        # if not os.path.exists(snapshot_path + '/code'):
        #     os.makedirs(snapshot_path + '/code')

        # shutil.copyfile("./train_BTCV_KnowSAM.py", snapshot_path + "/code/train_BTCV_KnowSAM.py")
        # shutil.copyfile("./trainer.py", snapshot_path + "/code/trainer.py")

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path)


# CUDA_VISIBLE_DEVICES=1 python train_BTCV_KnowSAM.py --batch_size_all 2 --labeled_bs_all 1 --labeled_num 10 --remark TMI25
# CUDA_VISIBLE_DEVICES=0 python train_BTCV_KnowSAM.py --batch_size_all 2 --labeled_bs_all 1 --labeled_num 30 --remark TMI25

# CUDA_VISIBLE_DEVICES=1 python train_BTCV_KnowSAM.py --batch_size_all 2 --labeled_bs_all 1 --labeled_num 10 --remark TMI25 --save_best_path /data/why/logs_SAM/KnowSAM_KnowSAM_10_0_TMI25/SGDL.pth

