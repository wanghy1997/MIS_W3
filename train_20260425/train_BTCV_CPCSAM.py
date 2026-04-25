import argparse
import logging
import os
import random
import shutil
import sys
import time
import os
import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from utils import test_util
from dataloaders.dataset import *
from networks.magicnet import VNet_Magic
from loss_amos import GADice, GACE
from segment_anything import sam_model_registry, sam_lora_image_encoder_prompt
from importlib import import_module
from utils_ori import ramps



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BTCV', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/BTCV/', help='Name of Dataset')
parser.add_argument('--log_path', type=str, default='/data/why/logs/', help='path to save')
parser.add_argument('--exp', type=str, default='SAM2', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=28000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=90, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=66, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=22, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labeled_num', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
parser.add_argument('--remark', type=str, default='ours_1', help='exp_name')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr', default=True)
parser.add_argument("--pretrain", type=str, default='')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
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
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
# dist
parser.add_argument('--dist', dest='dist', type=bool, default=True,
                    help='distributed segment_anything_training or not')
parser.add_argument('--vit_name', type=str,
                    default='vit_b_dualmask_same_prompt_class_random_large', help='select one vit model')
parser.add_argument('--warm_iter', type=int, default=2000, help='labeled data') 
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--promptmode', type=str, default='point',help='prompt')
parser.add_argument('--coe', type=float,
                    default=0.4, help='coe')
parser.add_argument('--coe2', type=float,
                    default=0.05, help='coe')
args = parser.parse_args()


snapshot_path = args.log_path + "/CPCSAM_{}_{}_{}_{}".format(args.dataset_name, args.exp, args.labeled_num, args.remark)

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


num_classes = 14
class_momentum = 0.999
patch_size = (128, 128, 128)

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size
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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def create_model(n_classes=14, cube_size=32, patch_size=96, ema=False):
    # Network definition
    net = VNet_Magic(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patch_size)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
    

def calc_loss_labeled(low_res_logits, low_res_label_batch, ce_loss, dice_loss, labeled_bs,dice_weight:float=0.8):
    low_res_logits_labeled = low_res_logits[:labeled_bs]
    low_res_label_batch_labeled = low_res_label_batch[:labeled_bs]

    loss_ce = ce_loss(low_res_logits_labeled, low_res_label_batch_labeled[:].long())
    loss_dice = dice_loss(low_res_logits_labeled, low_res_label_batch_labeled)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def train(labeled_list, unlabeled_list):
    base_lr = args.base_lr
    train_data_path = args.root_path
    
    train_list = labeled_list + unlabeled_list
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
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=128,
                                                                num_classes=num_classes,
                                                                checkpoint='/data/why/pretrain/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    low_res = img_embedding_size * 4   
    multimask_output = True
    model = sam_lora_image_encoder_prompt.LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        model.load_lora_parameters(args.lora_ckpt)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)  
    model.train()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    max_epoch = 2000
    best_dice_avg = 0.0
    # iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in tqdm(range(max_epoch + 1), dynamic_ncols=True, position=0):
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            volume_loss1 = 0.0
            volume_loss2 = 0.0
            volume_loss3 = 0.0

            # 如果你还想保留统计项
            volume_loss_ce1 = 0.0
            volume_loss_dice1 = 0.0
            volume_loss_ce2 = 0.0
            volume_loss_dice2 = 0.0

            valid_slice_count = 0
            print("iter_num:", iter_num, "epoch_num:", epoch_num, "i_batch-->", i_batch)
            for z in range(128):
                
                image_slice = image_batch[:, :, z, :, :]   # [B,1,H,W]
                label_slice = label_batch[:, z, :, :]      # [B,H,W]
                # 如果当前 batch 这张 slice 全是背景，直接跳过
                if not torch.any(label_slice > 0):
                    continue
                # assert image_slice.max() <= 3, f'image_slice max: {image_slice.max()}'
                # # 若模型要求 3 通道输入
                # image_slice = image_slice.repeat(1, 3, 1, 1)

                # --------------------------
                # first round
                # --------------------------
                outputs = model(image_slice, multimask_output, 128, -1, args.promptmode)

                outputs1 = outputs['low_res_logits1']
                outputs2 = outputs['low_res_logits2']

                supervised_loss1, loss_ce1, loss_dice1 = calc_loss_labeled(
                    outputs1, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                )
                supervised_loss2, loss_ce2, loss_dice2 = calc_loss_labeled(
                    outputs2, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                )

                loss1_slice = supervised_loss1 + supervised_loss2
                volume_loss1 += loss1_slice.item()

                volume_loss_ce1 += loss_ce1
                volume_loss_dice1 += loss_dice1
                volume_loss_ce2 += loss_ce2
                volume_loss_dice2 += loss_dice2

                # --------------------------
                # second / third round
                # 先按你原逻辑逐 slice 做
                # --------------------------
                if iter_num < args.warm_iter:
                    loss2_slice = torch.tensor(0.0, device=image_batch.device)
                    loss3_slice = torch.tensor(0.0, device=image_batch.device)
                else:
                    # round2
                    outputs_round2 = model(image_slice, multimask_output, 128, 1, args.promptmode)
                    outputs_round2_1 = outputs_round2['low_res_logits1']
                    outputs_round2_1_r = outputs_round2['low_res_logits1_r']
                    outputs_round2_2 = outputs_round2['low_res_logits2']

                    outputs_round2_soft1 = torch.softmax(outputs_round2_1, dim=1)
                    outputs_round2_soft1_r = torch.softmax(outputs_round2_1_r, dim=1)

                    supervised_round2_loss1, _, _ = calc_loss_labeled(
                        outputs_round2_1, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                    )
                    supervised_round2_loss1_r, _, _ = calc_loss_labeled(
                        outputs_round2_1_r, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                    )

                    outputs_round2_soft1 = (outputs_round2_soft1 + outputs_round2_soft1_r) / 2.0
                    pseudo_outputs1 = torch.argmax(outputs_round2_soft1[labeled_bs:].detach(), dim=1)

                    if pseudo_outputs1.numel() > 0:
                        consistency_loss2 = 0.5 * (
                            ce_loss(outputs_round2_2[labeled_bs:], pseudo_outputs1.long())
                            + dice_loss(outputs_round2_2[labeled_bs:], pseudo_outputs1, softmax=True)
                        )
                        consistency_loss1_r = 0.5 * (
                            ce_loss(outputs_round2_1_r[labeled_bs:], pseudo_outputs1.long())
                            + dice_loss(outputs_round2_1_r[labeled_bs:], pseudo_outputs1, softmax=True)
                        )
                    else:
                        consistency_loss2 = torch.tensor(0.0, device=image_batch.device)
                        consistency_loss1_r = torch.tensor(0.0, device=image_batch.device)

                    loss2_slice = supervised_round2_loss1 + supervised_round2_loss1_r + args.coe * consistency_loss2 + args.coe2 * consistency_loss1_r

                    # round3
                    outputs_round3 = model(image_slice, multimask_output, 128, 0, args.promptmode)
                    outputs_round3_1 = outputs_round3['low_res_logits1']
                    outputs_round3_2 = outputs_round3['low_res_logits2']
                    outputs_round3_2_r = outputs_round3['low_res_logits2_r']

                    outputs_round3_soft2 = torch.softmax(outputs_round3_2, dim=1)
                    outputs_round3_soft2_r = torch.softmax(outputs_round3_2_r, dim=1)

                    supervised_round3_loss1, _, _ = calc_loss_labeled(
                        outputs_round3_2, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                    )
                    supervised_round3_loss1_r, _, _ = calc_loss_labeled(
                        outputs_round3_2_r, label_slice, ce_loss, dice_loss, labeled_bs, args.dice_param
                    )

                    outputs_round3_soft2 = (outputs_round3_soft2 + outputs_round3_soft2_r) / 2.0
                    pseudo_outputs2 = torch.argmax(outputs_round3_soft2[labeled_bs:].detach(), dim=1)

                    if pseudo_outputs2.numel() > 0:
                        consistency_loss1 = 0.5 * (
                            ce_loss(outputs_round3_1[labeled_bs:], pseudo_outputs2.long())
                            + dice_loss(outputs_round3_1[labeled_bs:], pseudo_outputs2, softmax=True)
                        )
                        consistency_loss2_r = 0.5 * (
                            ce_loss(outputs_round3_2_r[labeled_bs:], pseudo_outputs2.long())
                            + dice_loss(outputs_round3_2_r[labeled_bs:], pseudo_outputs2, softmax=True)
                        )
                    else:
                        consistency_loss1 = torch.tensor(0.0, device=image_batch.device)
                        consistency_loss2_r = torch.tensor(0.0, device=image_batch.device)

                    loss3_slice = supervised_round3_loss1 + supervised_round3_loss1_r + args.coe * consistency_loss1 + args.coe2 * consistency_loss2_r

                volume_loss2 += loss2_slice.item()
                volume_loss3 += loss3_slice.item()
                valid_slice_count += 1
                loss = loss1_slice + loss2_slice + loss3_slice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if iter_num < args.warm_iter + 1:
                if iter_num < args.warm_iter + 1:
                    logging.info('iteration %d : loss : %f, loss1: %f' % (iter_num, loss.item(), volume_loss1))
                else:
                    logging.info('iteration %d : loss : %f, loss1: %f, loss2: %f, loss3: %f' % (iter_num, loss.item(), volume_loss1, volume_loss2, volume_loss3))

            # volume-level aggregate
            volume_loss1 = volume_loss1 / valid_slice_count
            volume_loss2 = volume_loss2 / valid_slice_count
            volume_loss3 = volume_loss3 / valid_slice_count
            loss_all = volume_loss1 + volume_loss2 + volume_loss3
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # the first round
            writer.add_scalar('info/total_loss', loss_all, iter_num)

            

            if iter_num > 0 and iter_num % 400 == 0:  # evaluation
                model.eval()
                # metric_list = 0.0
                # for i_batch, sampled_batch in enumerate(valloader):
                #     metric_i = test_single_volume(
                #         sampled_batch["image"], sampled_batch["label"], model, classes=num_classes+1) 
                #     metric_list += np.array(metric_i)
                # metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes):   
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                     metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                     metric_list[class_i, 1], iter_num)

                # performance = np.mean(metric_list, axis=0)[0]

                # mean_hd95 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                # if performance > best_performance:
                #     best_performance = performance
                #     save_mode_path = os.path.join(snapshot_path,
                #                                 'iter_{}_dice_{}.pth'.format(
                #                                     iter_num, round(best_performance, 4)))
                #     save_best = os.path.join(snapshot_path,
                #                             'best_model.pth')
                #     try:
                #         model.save_lora_parameters(save_best)
                #         model.save_lora_parameters(save_mode_path)
                #     except:
                #         model.module.save_lora_parameters(save_best)
                #         model.module.save_lora_parameters(save_mode_path)

                # logging.info(
                #     'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
    

        save_interval = 400 # int(max_epoch/6)   # 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= 10000 - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break
    model.train()
    
    writer.close()
    # logging.getLogger().removeHandler(handler)
    # logging.getLogger().removeHandler(sh)

    return None, best_model_path


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle(f'amos{args.labeled_num}_{args.remark}')
    print(snapshot_path)
    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    #     print("Create new folder {}".format(snapshot_path))
    # else:
    #     shutil.rmtree(snapshot_path)
    #     os.makedirs(snapshot_path)

    # _, best_model_path = train(labeled_list, unlabeled_list)
    best_model_path = '/data/why/logs/CPCSAM_BTCV_SAM2_10_ours_1/epoch_1999.pth'
    save_best_path = best_model_path

    model = create_model(n_classes=num_classes, cube_size=cube_size, patch_size=patch_size[0])
    model.load_state_dict(torch.load(save_best_path, weights_only=True))
    model.eval()
    _, _, metric_final = test_util.validation_all_case_btcv_2d(
                                                    model=model,
                                                    num_classes=num_classes,
                                                    base_dir=train_data_path,
                                                    image_list=test_list,
                                                    image_size=128,
                                                    multimask_output=True,
                                                    promptmode=args.promptmode,
                                                    branch='branch2',
                                                    save_nii_dir=snapshot_path,
                                                )

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_log_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_log_path, metric_final)

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


# CUDA_VISIBLE_DEVICES=0 python train_BTCV_CPCSAM.py --batch_size 2 --labeled_bs 1 --labeled_num 10