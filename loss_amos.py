import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AnatomyWeightScheduler:
    def __init__(self, total_epochs, warmup_epochs=10, target_rel_weight=0.5, mode='cosine'):
        """
        Args:
            total_epochs: 总训练轮数
            warmup_epochs: 预热轮数，在此期间只训练类别对齐。
            target_rel_weight: 训练结束时关系对齐达到的最大权重 (建议 < 1.0，因为类别基础不能丢)。
            mode: 'linear' 或 'cosine' 爬升策略。
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.target_rel_weight = target_rel_weight
        self.mode = mode

    def get_weights(self, current_epoch):
        """根据当前 Epoch 返回权重字典"""
        # 阶段 1: 预热期
        if current_epoch < self.warmup_epochs:
            w_cat = 1.0
            w_rel = 0.0
        # 阶段 2: 爬升期
        else:
            # 计算爬升进度 (0.0 -> 1.0)
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = max(0.0, min(1.0, progress)) # 确保在 [0, 1] 之间

            # 使用反向余弦曲线来实现平滑上升
            ramp = 1.0 - 0.5 * (1.0 + math.cos(math.pi * progress))
            
            w_rel = ramp * self.target_rel_weight
            
            # 策略 A: 保持类别权重恒定为 1.0 (推荐，更稳定)
            # w_cat = 1.0
            # 策略 B: 保持总权重和为 1.0 (w_cat 下降)
            w_cat = 1.0 - w_rel 

        return {'w_cat': w_cat, 'w_rel': w_rel}


class ContrastiveLoss(nn.Module):
    def __init__(self, ignore_index=-1, prompt_num=2):
        super(ContrastiveLoss, self).__init__()
        self.ignore_label = ignore_index
        self.prompt_num = prompt_num # 默认为 2：一个类别描述，一个关系描述

    def sample_anchor(self, features, label, n_view):
        # ... (保持你原来的 sample_anchor 代码逻辑不变)
        # 该函数返回 anchor (N, C) 和 target (N,)
        batch_size, feat_dim = features.shape[0], features.shape[-1]
        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]
            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x != self.ignore_label]
            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0: return None, None

        anc_features, anc_labels = [], []
        for i in range(batch_size):
            i_label, i_classes = label[i], classes[i]
            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))
        
        return torch.cat(anc_features, dim=0), torch.cat(anc_labels).to(features.device)

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask
        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        return -mean_log_prob_pos.mean()

    def forward(self, features_embedding_all, label, text_embedding_all, entropy_map, threshold, sample_num=10, weights={'w_cat': 1.0, 'w_rel': 0.0}):
        """
        features_embedding_all: [feat_64, feat_128, feat_256]
        text_embedding_all: [text_64, text_128, text_256] 
                           (假设每个尺度的文本特征都包含 class_text 和 rel_text)
        """
        loss = 0.0
        n_view_list = [int(x * sample_num) for x in [4, 2, 1]]

        # 1. 处理伪标签 (保持原有逻辑)
        mask = entropy_map > threshold
        # 这里的 label[6:] 假设 batch_size=labeled_bs(6) + unlabeled_bs
        pseudo_label = label[6:].clone() 
        pseudo_label[mask] = self.ignore_label
        full_label = torch.cat([label[:6], pseudo_label], dim=0)

        loss_cat_total = 0.0
        loss_rel_total = 0.0

        # 2. 遍历多尺度特征
        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            # 标签插值到当前特征图尺寸
            i_label = F.interpolate(full_label.float(), (features.shape[2], features.shape[3], features.shape[4]), mode='nearest').long()

            batch_size = features.shape[0]
            i_label_flat = i_label.view(batch_size, -1)
            features_flat = features.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, features.shape[1])

            # 采样视觉锚点
            anchor_v, target_v = self.sample_anchor(features_flat, i_label_flat, n_view)
            if anchor_v is None: continue

            # --- 文本特征拆解 ---
            # 根据 prompt_num=2, 前一半是类别文本, 后一半是关系文本
            num_classes = text_embedding.shape[0] // self.prompt_num
            # print('text_embedding.shape[0]', text_embedding.shape[0])
            class_text = text_embedding[:num_classes]   # [16, C]  # 器官自身的文本描述
            rel_text = text_embedding[num_classes:]     # [16, C]  # 邻接关系的文本描述
            text_target = torch.arange(num_classes).to(target_v.device)

            # --- 策略 2: 聚焦类别的对齐 (Visual Anchor + Class Text) ---
            anchor_cat = torch.cat([anchor_v, class_text], dim=0)
            target_cat = torch.cat([target_v, text_target], dim=0)
            anchor_cat = F.normalize(anchor_cat, p=2, dim=1)
            loss_category = self.contrastive_loss(anchor_cat, target_cat)

            # --- 策略 3: 聚焦邻接关系的对齐 (Visual Anchor + Relation Text) ---
            anchor_rel = torch.cat([anchor_v, rel_text], dim=0)
            target_rel = torch.cat([target_v, text_target], dim=0)
            anchor_rel = F.normalize(anchor_rel, p=2, dim=1)
            loss_relation = self.contrastive_loss(anchor_rel, target_rel)

            # 累加各个尺度的损失
            loss_cat_total += loss_category
            loss_rel_total += loss_relation

        # 累加双重任务损失
        # 获取当前时刻的权重
        w_cat = weights.get('w_cat', 1.0)
        w_rel = weights.get('w_rel', 0.0)

        return w_cat * loss_cat_total + w_rel * loss_rel_total


class ContrastiveLoss_AB(nn.Module):
    def __init__(self, ignore_index=-1, prompt_num=2):
        super(ContrastiveLoss_AB, self).__init__()
        self.ignore_label = ignore_index
        self.prompt_num = prompt_num # 默认为 2：一个类别描述，一个关系描述

    def sample_anchor(self, features, label, n_view):
        # ... (保持你原来的 sample_anchor 代码逻辑不变)
        # 该函数返回 anchor (N, C) 和 target (N,)
        batch_size, feat_dim = features.shape[0], features.shape[-1]
        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]
            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x != self.ignore_label]
            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0: return None, None

        anc_features, anc_labels = [], []
        for i in range(batch_size):
            i_label, i_classes = label[i], classes[i]
            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))
        
        return torch.cat(anc_features, dim=0), torch.cat(anc_labels).to(features.device)

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask
        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        return -mean_log_prob_pos.mean()

    def forward(self, features_embedding_all, label, text_embedding_all, entropy_map, threshold, sample_num=10, weights={'w_cat': 1.0, 'w_rel': 0.0}):
        """
        features_embedding_all: [feat_64, feat_128, feat_256]
        text_embedding_all: [text_64, text_128, text_256] 
                           (假设每个尺度的文本特征都包含 class_text 和 rel_text)
        """
        loss = 0.0
        n_view_list = [int(x * sample_num) for x in [4, 2, 1]]

        # 1. 处理伪标签 (保持原有逻辑)
        mask = entropy_map > threshold
        # 这里的 label[6:] 假设 batch_size=labeled_bs(6) + unlabeled_bs
        pseudo_label = label[6:].clone() 
        pseudo_label[mask] = self.ignore_label
        full_label = torch.cat([label[:6], pseudo_label], dim=0)

        loss_cat_total = 0.0
        loss_rel_total = 0.0

        # 2. 遍历多尺度特征
        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            # 标签插值到当前特征图尺寸
            i_label = F.interpolate(full_label.float(), (features.shape[2], features.shape[3], features.shape[4]), mode='nearest').long()

            batch_size = features.shape[0]
            i_label_flat = i_label.view(batch_size, -1)
            features_flat = features.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, features.shape[1])

            # 采样视觉锚点
            anchor_v, target_v = self.sample_anchor(features_flat, i_label_flat, n_view)
            if anchor_v is None: continue

            # --- 文本特征拆解 ---
            # 根据 prompt_num=2, 前一半是类别文本, 后一半是关系文本
            num_classes = text_embedding.shape[0] // self.prompt_num
            # print('text_embedding.shape[0]', text_embedding.shape[0])
            class_text = text_embedding[:num_classes]   # [16, C]  # 器官自身的文本描述
            # rel_text = text_embedding[num_classes:]     # [16, C]  # 邻接关系的文本描述
            text_target = torch.arange(num_classes).to(target_v.device)

            # --- 策略 2: 聚焦类别的对齐 (Visual Anchor + Class Text) ---
            anchor_cat = torch.cat([anchor_v, class_text], dim=0)
            target_cat = torch.cat([target_v, text_target], dim=0)
            anchor_cat = F.normalize(anchor_cat, p=2, dim=1)
            loss_category = self.contrastive_loss(anchor_cat, target_cat)

            # # --- 策略 3: 聚焦邻接关系的对齐 (Visual Anchor + Relation Text) ---
            # anchor_rel = torch.cat([anchor_v, rel_text], dim=0)
            # target_rel = torch.cat([target_v, text_target], dim=0)
            # anchor_rel = F.normalize(anchor_rel, p=2, dim=1)
            # loss_relation = self.contrastive_loss(anchor_rel, target_rel)

            # 累加各个尺度的损失
            loss_cat_total += loss_category
            # loss_rel_total += loss_relation

        # 累加双重任务损失
        # 获取当前时刻的权重
        # w_cat = weights.get('w_cat', 1.0)
        # w_rel = weights.get('w_rel', 0.0)

        return loss_cat_total

class GADice(nn.Module):
    def __init__(self, GA=True):
        self.GA = GA
        super(GADice, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, cls, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10

        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union

        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        self.n_classes = inputs.size()[1]
        if len(inputs.size()) == len(target.size()) + 1:
            target = target.unsqueeze(1)

        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict & target shape do not match'
        smooth = 1e-10
        loss = 0.0
        for i in range(0, self.n_classes):
            if torch.sum(target[:, i]) > 0:
                dice_loss = self._dice_loss(i, inputs[:, i], target[:, i], weighted_pixel_map)
            else:
                if self.GA:
                    beta = inputs[:, i] / (torch.sum(1 - target[:, i]))
                    dice_loss = torch.sum(beta.detach() * inputs[:, i])
            loss += dice_loss * weight[i]

        return loss / self.n_classes


class GACE(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1, )
        res = super(GACE, self).forward(inp, target)

        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1,)), int(n_instance * self.k / 100), sorted=False)
        target = torch.gather(target, 0, indices)
        assert res.size() == target.size(), 'predict & target shape do not match'

        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        for i in range(0, self.n_classes):
            target_cls = (target == i).float()
            w = torch.pow(torch.sum(target_cls) + smooth, 1 - self.gama) * bg_w
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss


class ContrastiveLoss_(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 prompt_num=2):
        super(ContrastiveLoss_, self).__init__()

        self.ignore_label = ignore_index
        self.prompt_num = prompt_num

    def sample_anchor(self, features, label, n_view):
        batch_size, feat_dim = features.shape[0], features.shape[-1]

        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]

            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x != self.ignore_label]

            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0:
            return None, None

        anc_features = []
        anc_labels = []

        for i in range(batch_size):
            i_label = label[i]
            i_classes = classes[i]

            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))

        anc_features = torch.cat(anc_features, dim=0)
        anc_labels = torch.cat(anc_labels).to(features.device)

        return anc_features, anc_labels

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        # temperature = self.temperature_list[target].unsqueeze(1)
        # sim = torch.div(torch.matmul(anchor, anchor.T), temperature)

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask

        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        loss = -mean_log_prob_pos.mean()

        return loss

    def forward(self, features_embedding_all, label, text_embedding_all, entropy_map, threshold, sample_num = 10):
        loss = 0.0
        # n_view_list = [40, 25, 10]
        n_view_list = [4, 2, 1]
        n_view_list = [int(x * sample_num) for x in n_view_list]

        mask = entropy_map > threshold
        pseudo_label = label[6:]
        pseudo_label[mask] = -1
        label = torch.cat([label[:6], pseudo_label], dim=0)

        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            i_label = label.float()
            i_label = F.interpolate(i_label, (features.shape[2], features.shape[3], features.shape[4]), mode='nearest')
            i_label = i_label.long()
            assert i_label.shape[-1] == features.shape[-1], '{} {}'.format(i_label.shape, features.shape)

            batch_size = features.shape[0]
            i_label = i_label.view(batch_size, -1)
            features = features.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, features.shape[1])

            anchor, target = self.sample_anchor(features, i_label, n_view)
            anchor = torch.cat([anchor, text_embedding], dim=0)
            anchor = F.normalize(anchor, p=2, dim=1)
            # text_embedding_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8,
            #                                      9, 10, 11, 12, 13, 14, 15]).repeat(self.prompt_num).to(target.device)
            # 1. 根据 text_embedding 的总长度推算类别数
            # text_embedding.shape[0] 是总的文本特征数量 (Class_Num * Prompt_Num)
            current_num_classes = text_embedding.shape[0] // self.prompt_num

            # 2. 动态生成 0 到 N-1 的标签
            text_embedding_label = torch.arange(current_num_classes).repeat(self.prompt_num).to(target.device)
            target = torch.cat([target, text_embedding_label], dim=0)
            loss += self.contrastive_loss(anchor, target)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10
        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i], weighted_pixel_map)
            # class_wise_dice.append(dice_loss)
            loss += dice_loss * weight[i]

        return loss / self.n_classes


@torch.no_grad()
def build_cam(
    feat_3d: torch.Tensor,
    clip_embedding: torch.Tensor,
    attn_pred: torch.Tensor = None,
    eps: float = 1e-6
):
    """
    Build 3D CAM using CLIP text embedding and optional token affinity.

    Args:
        feat_3d:        [B, C, D, H, W]   (e.g. [2, 256, 6, 6, 6])
        clip_embedding: [num_classes, C] (e.g. [16, 256])
        attn_pred:      [num_classes, T, T] or None
        eps:            numerical stability

    Returns:
        cam_region:     [B, num_classes, D, H, W]
    """

    B, C, D, H, W = feat_3d.shape
    num_classes = clip_embedding.shape[0]

    # -------------------------------------------------
    # 1️⃣ Normalize features and text embeddings
    # -------------------------------------------------
    feat_norm = F.normalize(feat_3d, dim=1)                 # [B, C, D, H, W]
    text_norm = F.normalize(clip_embedding, dim=1)          # [C_cls, C]

    # -------------------------------------------------
    # 2️⃣ Compute voxel–class similarity (raw CAM)
    #     CAM[b, k, d, h, w] = <f_b(d,h,w), t_k>
    # -------------------------------------------------
    cam = torch.einsum(
        "bcdhw,kc->bkdhw",
        feat_norm,
        text_norm
    )                                                        # [B, K, D, H, W]

    # ReLU: only keep positive evidence (standard CAM practice)
    cam = F.relu(cam)

    # -------------------------------------------------
    # 3️⃣ Normalize CAM spatially (per class, per sample)
    # -------------------------------------------------
    cam_flat = cam.view(B, num_classes, -1)
    cam_min = cam_flat.min(dim=-1, keepdim=True)[0]
    cam_max = cam_flat.max(dim=-1, keepdim=True)[0]
    cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + eps)
    cam_norm = cam_norm.view(B, num_classes, D, H, W)

    # -------------------------------------------------
    # 4️⃣ Optional: class-wise smoothing using token affinity
    # -------------------------------------------------
    if attn_pred is not None:
        # attn_pred: [K, T, T] → reduce to [K, K] affinity
        # 简单做法：token 相似性求均值，得到 class-to-class affinity
        class_affinity = attn_pred.mean(dim=-1)              # [K, T]
        class_affinity = class_affinity @ class_affinity.t() # [K, K]
        class_affinity = F.softmax(class_affinity, dim=-1)

        # CAM smoothing across classes
        cam_smooth = torch.einsum(
            "ij,bjdhw->bidhw",
            class_affinity,
            cam_norm
        )
        cam_norm = cam_smooth

    return cam_norm



@torch.no_grad()
def refine_cam_with_affinity_3d(
    cam: torch.Tensor,
    feat_3d: torch.Tensor = None,
    affinity: torch.Tensor = None,
    class_mask: torch.Tensor = None,
    num_iter: int = 4,
    alpha: float = 0.7,
    local_radius: int = None,
    seed_thre: float = 0.7,
    keep_seeds: bool = True,
    eps: float = 1e-6,
):
    """
    ViT-agnostic CAM refinement via region affinity propagation (3D-ready).

    This abstracts ExCEL's `refine_cams_with_aff` into a generic graph propagation
    that only needs:
      - cam:          [B, K, D, H, W]  (or [B, K, N])
      - affinity:     [B, N, N] (optional) region similarity / adjacency
      - feat_3d:      [B, C, D, H, W] (optional) used to compute affinity if not provided
      - class_mask:   [B, K] (optional) indicates which classes exist in each sample

    Output:
      - cam_refined:  [B, K, D, H, W]  refined CAMs in [0,1] (per class, per sample)
    """
    assert cam.dim() in (3, 5), f"cam must be [B,K,N] or [B,K,D,H,W], got {cam.shape}"

    # ---- shape handling ----
    if cam.dim() == 5:
        B, K, D, H, W = cam.shape
        N = D * H * W
        cam_flat = cam.view(B, K, N)
    else:
        B, K, N = cam.shape
        cam_flat = cam

    # ---- class existence mask ----
    if class_mask is not None:
        # class_mask: [B, K] -> [B, K, 1]
        cm = class_mask.to(dtype=cam_flat.dtype, device=cam_flat.device).unsqueeze(-1)
        cam_flat = cam_flat * cm

    # ---- normalize CAM to [0,1] per (B,K) ----
    cam_min = cam_flat.amin(dim=-1, keepdim=True)
    cam_max = cam_flat.amax(dim=-1, keepdim=True)
    cam0 = (cam_flat - cam_min) / (cam_max - cam_min + eps)

    # ---- build affinity A [B,N,N] ----
    if affinity is None:
        assert feat_3d is not None, "Provide either affinity or feat_3d to compute affinity."
        assert feat_3d.dim() == 5, f"feat_3d must be [B,C,D,H,W], got {feat_3d.shape}"
        Bf, Cf, Df, Hf, Wf = feat_3d.shape
        assert Bf == B, "feat_3d batch size must match cam batch size"
        Nf = Df * Hf * Wf
        assert Nf == N, "feat_3d spatial tokens must match cam tokens (same D,H,W) for refinement"
        f = feat_3d.view(B, Cf, Nf)
        f = F.normalize(f, dim=1)
        A = torch.matmul(f.transpose(1, 2), f)  # [B, N, N], cosine sim
        A = F.relu(A)
    else:
        A = affinity
        assert A.dim() == 3 and A.shape[0] == B and A.shape[1] == N and A.shape[2] == N, \
            f"affinity must be [B,N,N] matching cam tokens, got {A.shape}"

    # ---- optional local neighborhood mask (prevents dense propagation) ----
    if local_radius is not None and local_radius > 0:
        # Build a (N,N) boolean mask based on 3D grid distance
        # Only feasible when we know D,H,W (cam provided as 5D)
        assert cam.dim() == 5, "local_radius requires cam as [B,K,D,H,W] to infer grid."
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=A.device),
            torch.arange(H, device=A.device),
            torch.arange(W, device=A.device),
            indexing="ij"
        )
        coords = torch.stack([zz, yy, xx], dim=-1).view(N, 3).to(torch.float32)  # [N,3]
        # squared euclidean distance
        dist2 = torch.cdist(coords, coords, p=2.0)  # [N,N]
        neigh = (dist2 <= float(local_radius)).to(A.dtype)
        A = A * neigh.unsqueeze(0)

    # ---- row-normalize affinity to be stochastic ----
    A = A / (A.sum(dim=-1, keepdim=True) + eps)

    # ---- seed mask (keep strong CAM evidence fixed) ----
    if keep_seeds:
        seeds = cam0 > seed_thre  # [B,K,N]
    else:
        seeds = None

    # ---- iterative propagation: cam <- (1-a)cam0 + a(cam @ A) ----
    cam_ref = cam0
    for _ in range(int(num_iter)):
        cam_ref = (1.0 - alpha) * cam0 + alpha * torch.matmul(cam_ref, A)  # [B,K,N]
        if seeds is not None:
            cam_ref = torch.where(seeds, cam0, cam_ref)

    # ---- re-normalize to [0,1] per (B,K) ----
    cam_min2 = cam_ref.amin(dim=-1, keepdim=True)
    cam_max2 = cam_ref.amax(dim=-1, keepdim=True)
    cam_ref = (cam_ref - cam_min2) / (cam_max2 - cam_min2 + eps)

    if cam.dim() == 5:
        cam_ref = cam_ref.view(B, K, D, H, W)
    return cam_ref


def cam_structure_kl_3d(
    pred_logits: torch.Tensor,
    cam: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
):
    """
    Coarse structural consistency between student prediction and CAM prior.

    pred_logits: [B, K, d, h, w]  (student logits at CAM resolution)
    cam:         [B, K, d, h, w]  (CAM prior, MUST be detached by caller)
    mask:        [B, 1, d, h, w]  (optional, e.g., uncertain regions)
    """
    assert pred_logits.dim() == 5 and cam.dim() == 5
    B, K, d, h, w = pred_logits.shape
    assert cam.shape == (B, K, d, h, w)

    pred_prob = F.softmax(pred_logits, dim=1)
    cam_prob = cam / (cam.sum(dim=1, keepdim=True) + eps)

    # KL(cam || pred) per voxel
    kl = (cam_prob * (torch.log(cam_prob + eps) - torch.log(pred_prob + eps))).sum(dim=1, keepdim=True)  # [B,1,d,h,w]

    if mask is not None:
        assert mask.shape == (B, 1, d, h, w)
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0)
        return kl.sum() / denom

    return kl.mean()


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                     F.softmax(out_t / self.T, dim=1), reduction="batchmean") # , reduction="batchmean"
            * self.T
            * self.T
        )
        return loss