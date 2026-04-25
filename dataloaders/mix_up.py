import torch
import numpy as np
import torch.nn.functional as F
from skimage.measure import label
import torch.nn as nn
from skimage.measure import label


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.unsqueeze(1))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes
    

class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super(PolyWarmRestartScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]
    

def weit_loss(pred, mask, weit, weit_coef=2, smooth=1e-6):
        weit = weit * weit_coef + 1
        mask_one_hot = F.one_hot(mask.long(), num_classes=2).permute(0, 4,  1, 2, 3).float()

        wbce = F.binary_cross_entropy_with_logits(pred, mask_one_hot, reduction='none')
        wbce = (weit * wbce).mean()
        
        # IOU
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * mask_one_hot * weit).sum(dim=(2, 3, 4))
        union = (pred_sig * weit).sum(dim=(2, 3, 4)) + (mask_one_hot * weit).sum(dim=(2, 3, 4)) - inter
        wiou = 1 - ((inter + smooth)/(union + smooth)).mean(dim=1)
        
        if wiou.mean() < 0:
            print("Error", wiou.mean())
        
        loss = wbce + wiou.mean()
        return loss


def multi_class_weit_loss(pred, mask, weit=None, num_classes=None, weit_coef=2, smooth=1e-6):
    """
    pred: [B, C, D, H, W]
    mask: [B, D, H, W]，取值范围 0 ~ C-1
    weit: [B, D, H, W] 或 [B, 1, D, H, W]，像素级权重图
    """
    if num_classes is None:
        num_classes = pred.shape[1]

    if weit is None:
        weit = torch.ones_like(mask).float()
    else:
        weit = weit.float()

    if weit.dim() == 5 and weit.shape[1] == 1:
        weit = weit.squeeze(1)  # [B, D, H, W]

    weit = weit * weit_coef + 1.0   # [B, D, H, W]

    # -------------------------
    # 1. Weighted Cross Entropy
    # -------------------------
    ce = F.cross_entropy(pred, mask.long(), reduction='none')   # [B, D, H, W]
    wce = (ce * weit).mean()

    # -------------------------
    # 2. Weighted Multi-class IoU
    # -------------------------
    pred_prob = F.softmax(pred, dim=1)   # [B, C, D, H, W]
    mask_one_hot = F.one_hot(mask.long(), num_classes=num_classes) \
                    .permute(0, 4, 1, 2, 3).float()             # [B, C, D, H, W]

    weit_expand = weit.unsqueeze(1)      # [B, 1, D, H, W]

    inter = (pred_prob * mask_one_hot * weit_expand).sum(dim=(2, 3, 4))   # [B, C]
    union = (
        (pred_prob * weit_expand).sum(dim=(2, 3, 4)) +
        (mask_one_hot * weit_expand).sum(dim=(2, 3, 4)) -
        inter
    )   # [B, C]

    wiou = 1.0 - (inter + smooth) / (union + smooth)   # [B, C]
    wiou = wiou.mean()   # 所有 batch、所有类别取平均

    loss = wce + wiou
    return loss



def get_entropy_map(p, softmax=True):
    if softmax:
        p = torch.softmax(p, dim=1)
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


# def get_cut_mask(out, thres=0.5, nms=0):
#     """
#     pred: [B, C, D, H, W]
#     return: [B, D, H, W]
#     """
#     prob = torch.softmax(out, dim=1)
#     pseudo = torch.argmax(prob, dim=1).long()
#     return pseudo


def keep_largest_cc_per_class(mask_np):
    """
    对单张多类别图像进行后处理：
    针对图像中出现的每一个类别，只保留其最大的连通域。
    """
    # 创建一个全黑的底图，用于存放处理后的结果
    cleaned_mask = np.zeros_like(mask_np)
    
    # 获取图中存在的所有类别（排除背景 0）
    unique_classes = np.unique(mask_np)
    unique_classes = unique_classes[unique_classes > 0]
    
    for cls in unique_classes:
        # 1. 提取当前类别的二值掩码
        class_mask = (mask_np == cls).astype(np.uint8)
        
        # 2. 对当前类别进行连通域标记
        labeled_mask = label(class_mask)
        
        if labeled_mask.max() > 0:
            # 3. 统计各连通域的大小（跳过背景索引 0）
            counts = np.bincount(labeled_mask.flat)
            # 找到最大的连通域索引
            largest_cc_idx = np.argmax(counts[1:]) + 1
            
            # 4. 将该类别最大的连通域填入结果图中，保持原有的类别标签 cls
            cleaned_mask[labeled_mask == largest_cc_idx] = cls
            
    return cleaned_mask


def get_cut_mask(out, thres=0.5, apply_nms=True):
    """
    out: 模型的输出 Tensor, 形状为 [B, C, H, W] 或 [B, C, D, H, W]
    """
    # 1. 获取初步的类别预测
    probs = F.softmax(out, dim=1)
    # 得到 [B, H, W] 或 [B, D, H, W] 的类别索引图
    masks = torch.argmax(probs, dim=1)
    
    if not apply_nms:
        return masks

    # 2. 转换到 CPU 进行连通域处理
    masks_np = masks.detach().cpu().numpy()
    batch_size = masks_np.shape[0]
    processed_list = []
    
    for i in range(batch_size):
        # 对 Batch 中的每一例进行独立的多类别后处理
        processed_mask = keep_largest_cc_per_class(masks_np[i])
        processed_list.append(processed_mask)
    
    # 3. 转回 Tensor 并移动到原始设备
    return torch.from_numpy(np.array(processed_list)).to(out.device)


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:] ) +1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def generate_mask_3D(img, mask_ratio=2/3):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()

    patch_x, patch_y, patch_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    d = np.random.randint(0, img_z - patch_z)
    mask[w:w+patch_x, h:h+patch_y, d:d+patch_z] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y, d:d+patch_z] = 0
    # mask[w:w+patch_x, h:h+patch_y, ...] = 0
    # loss_mask[:, w:w+patch_x, h:h+patch_y, ...] = 0
    return mask.long(), loss_mask.long()
