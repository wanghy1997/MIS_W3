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
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils import metrics, ramps, test_util, cube_losses, cube_utils
from dataloaders.dataset import *
from networks.magicnet import VNet_Magic
from loss_amos import GADice, GACE
from code_semisampp.semisam_plus import SAM_branch, SAM_init
from sam2.build_sam import build_sam2
from SAM2_builder import CrispSam2
from torch.nn.parallel import DataParallel as DP


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BTCV', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/BTCV/', help='Name of Dataset')
parser.add_argument('--log_path', type=str, default='/data/why/logs/', help='path to save')
parser.add_argument('--exp', type=str, default='SAM2', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=35000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=90, help='total samples of the dataset')
parser.add_argument('--max_epoch', type=int, default=10000, help='max_epoch')
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
parser.add_argument('--remark', type=str, default='ours_1', help='exp_name')

parser.add_argument("--pretrain", type=str, default='')
parser.add_argument("--resume", type=str, default='')
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--dataset_codes", type=list, default=['0003'])
parser.add_argument("--mode", type=str, default='train')

# config
parser.add_argument("--patch_size", default=(96, 96, 96), type=tuple)
parser.add_argument("--spatial_size", default=(32, 512, 512), type=tuple)
parser.add_argument("--rand_flipped_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--rand_scale_intensityd_prob", default=0.1, type=float,
                    help="RandScaleIntensityd aug probability")
parser.add_argument("--rand_shift_intensityd_prob", default=0.1, type=float,
                    help="RandShiftIntensityd aug probability")
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument("--clip_text_ckpt", type=str, default='./path/to/clip_text_ckpt')
parser.add_argument("--clip_image_ckpt", type=str, default='./path/to/clip_image_ckpt')
parser.add_argument("--config_file", type=str, default='./path/to/config_file')
parser.add_argument("--sam2_ckpt", type=str, default='./path/to/sam2_ckpt')
# parser.add_argument("--model_id", type=str, default='hf_id')
parser.add_argument('--num_workers', type=int, default=8)

# dist
parser.add_argument('--dist', dest='dist', type=bool, default=True,
                    help='distributed segment_anything_training or not')
parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
parser.add_argument('--init_method', type=str, default="env://")
parser.add_argument('--bucket_cap_mb', type=int, default=25,
                    help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')

# key params
parser.add_argument('--lr_one', type=float, default=1e-4)
parser.add_argument('--lr_two', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--stage_one_frozen', type=list)
parser.add_argument('--stage_two_frozen', type=list)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--stage_one', type=int, default=120)
parser.add_argument('--stage_two', type=int, default=280)
parser.add_argument('--warmup_epoch', type=int, default=80)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)

args = parser.parse_args()


def create_model(ema=False):
    sam_model = build_sam2(config_file=args.config_file, checkpoint=args.sam2_ckpt, mode="train")  # checkpoint for pretrained sam2
    model = CrispSam2(
        image_encoder=sam_model.image_encoder,
        memory_attention=sam_model.memory_attention,
        memory_encoder=sam_model.memory_encoder,
        clip_text_ckpt=args.clip_text_ckpt,
        clip_image_ckpt=args.clip_image_ckpt,
    ).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    # model = DP(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    return model

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
        os.path.join(f'{args.root_path}/split_txts/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


if args.label_num == 30:
    # 30%
    labeled_list = read_list('labeled_30p')
    unlabeled_list = read_list('unlabeled_30p')
elif args.label_num == 40:
    # 40%, 10 labeld
    labeled_list = read_list('labeled_40p')
    unlabeled_list = read_list('unlabeled_40p')
else:
    print('Error label_num!')
    os.exit()

eval_list = read_list('eval')
test_list = read_list('test')

snapshot_path = args.log_path + "/SAM2_{}_{}_{}_{}".format(args.dataset_name, args.exp, args.label_num, args.remark)

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
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)
    db_train = BTCV(train_list,
                    base_dir=train_data_path,
                    transform=transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                    ]))
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

    writer = SummaryWriter(snapshot_path)
    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("Logs files: {} ".format(snapshot_path))

    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)      

    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    lr_ = base_lr
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)

    for epoch_num in tqdm(range(args.max_epoch + 1), dynamic_ncols=True, position=0):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            labeled_volume_batch = volume_batch[:labeled_bs]

            model.train()
            outputs = model(volume_batch)[0] # Original Model Outputs
            
            

            # Final Loss
            loss = supervised_loss + 0.1 * loc_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            if iter_num % 20 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.3f}, '
                             'cons_dist: {:.3f}, loss_weight: {:f}, '
                             'loss_loc: {:.3f}'.format(1, iter_num,
                                                       loss,
                                                       consistency_loss,
                                                       consistency_weight,
                                                       0.1 * loc_loss))

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            if iter_num >= 400 and iter_num % 500 == 0:
            # if iter_num >= max_iterations:
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
    setproctitle.setproctitle(f'amos{args.label_num}_{args.remark}')

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        print("Create new folder {}".format(snapshot_path))
    else:
        shutil.rmtree(snapshot_path)
        os.makedirs(snapshot_path)

    _, best_model_path = train(labeled_list, unlabeled_list)

    save_best_path = best_model_path

    model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0])
    model.load_state_dict(torch.load(save_best_path, weights_only=True))
    model.eval()
    _, _, metric_all_cases = test_util.validation_all_case_btcv(model,
                                                            num_classes=num_classes,
                                                            base_dir=train_data_path,
                                                            image_list=test_list,
                                                            patch_size=patch_size,
                                                            stride_xy=16,
                                                            stride_z=16)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_all_cases, axis=0), np.std(metric_all_cases, axis=0)

    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_all_cases)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, '
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
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
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
