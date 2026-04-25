import torch
from torch import nn
import torch.nn.functional as F
from .hyperbolic import BlockDiagonalLinear, BlockDiagonalLinear_text
from .ca_doubleSpace import FeatureFusionGate, HyperMultiAttention1, HyperTextGuidedVisualBlock, InterSliceTextGuidedAttention


class HyperbolicTextAdapter(nn.Module):
    def __init__(self, dim, block_size=32, curvature=1e-3, alpha=0.1):
        super(HyperbolicTextAdapter, self).__init__()
        self.dim = dim
        self.block = BlockDiagonalLinear_text(block_size, dim, dim, curvature=curvature)
        self.alpha = alpha  # 控制双曲扰动强度

    def forward(self, x):
        """
        x: [..., dim]  (比如 [32, 512])
        return: 同形状，沿最后一维做 hyper block-diagonal 变换
        """
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])   # [N, dim]
        x_h = self.block(x_flat)              # expmap -> Möbius -> logmap
        # 残差形式：避免完全替换掉 CLIP 的原始文本语义
        x_h = x_flat + self.alpha * (x_h - x_flat)
        x_h = x_h.view(*orig_shape)
        return x_h


class HyperbolicFeatureAdapter(nn.Module):
    def __init__(self, channels, block_size=32, curvature=0.01):
        super(HyperbolicFeatureAdapter, self).__init__()
        self.channels = channels
        self.block = BlockDiagonalLinear(block_size, channels, channels, curvature=curvature)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        assert C == self.channels
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*D*H*W, C]
        x_h = self.block(x_flat)                          # 超曲 exp -> Möbius mat -> log
        x_h = x_h.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x_h
    

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:

            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class FcLayer(nn.Module):
    def __init__(self, ts=32, patch_size=96, n_filters=16):
        super(FcLayer, self).__init__()
        nt = patch_size // ts
        self.fc_layer = nn.Sequential(
            nn.Linear((n_filters * 16) * ((ts // 16) ** 3), 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, nt ** 3)
        )

    def forward(self, x):
        return self.fc_layer(x)



class VNet_Magic_CLIP_2p_Contrast(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, cube_size=32, patch_size=96, n_filters=16,
                 normalization='instancenorm', has_dropout=False, has_residual=False):
        super(VNet_Magic_CLIP_2p_Contrast, self).__init__()

        self.num_classes = n_classes
        self.has_dropout = has_dropout
        text_embedding_dim = 512

        self.fc_layer = FcLayer(cube_size, patch_size)

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        # Encoder
        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        # Decoder
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)


        # # CLIP
        if n_classes == 16: 
            # self.text_embedding = torch.load(self.text_path, weights_only=True).float()
            # # print('loaded shape&location embedding:', self.text_embedding.shape)  # [32, 512]
            # text_embedding_dim = self.text_embedding.shape[-1]
            
            # 适用于 AMOS 的 CLIP 文本嵌入，16 类别
            # ckpt = torch.load("/home/why/UniMed-CLIP-main/text_embeddings/biomedbert_text_embeddings_16_baseline.pt")
            ckpt = torch.load("/home/why/UniMed-CLIP-main/text_embeddings/biomedbert_text_embeddings_16_ours.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [32, 512]
            text_embedding_dim = self.text_embedding.shape[-1]
        elif n_classes == 14:
            # UniCLIP BiomedBERT  适用于 Synapse 和 FLARE 13 类 + 背景
            ckpt = torch.load("/home/why/UniMed-CLIP-main/text_embeddings/biomedbert_text_embeddings_14_baseline.pt")
            # ckpt = torch.load("/home/why/UniMed-CLIP-main/text_embeddings/biomedbert_text_embeddings_14_ours.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [28, 512]
            text_embedding_dim = self.text_embedding.shape[-1]
        
        self.text_to_64 = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.ReLU(inplace=True),
        )

        self.text_to_128 = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(inplace=True),
        )

        self.text_to_256 = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(inplace=True),
        )

        # hyper_text
        self.text_hyper = HyperMultiAttention1(embedding_dim=256,
                 num_heads=8,
                 n_layers=6,
                 dropout_p=0.0,
                 c=1.0)
        
        self.text_to_vision = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
        )

        self.Gap_64 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(64, 256, kernel_size=1, stride=1, padding=0),
        )

        self.GAP_128 = nn.Sequential(
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0),
        )

        self.GAP_256 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0),
        )

        # hyper
        self.hyper_block_64 = HyperTextGuidedVisualBlock(
            img_size=(24, 24, 24),
            patch_size=(3, 3, 3),      # 24 / 3 = 8 tokens per dim
            in_chans=64,
            embed_dim=256,
            num_heads=4,
            c=0.01
        )
        self.hyper_block_64_32 = HyperTextGuidedVisualBlock(
            img_size=(8, 8, 8),      # cube 内特征图
            patch_size=(4, 4, 4),    # 8 / 4 = 2 tokens per dim → 2×2×2 = 8 tokens
            in_chans=64,
            embed_dim=256,           # ⬅️ 降维，避免 token attention 过拟合
            num_heads=2,             # ⬅️ 减少头数
            c=0.005                  # ⬅️ 更平的双曲空间
        )

        self.hyper_block_128 = HyperTextGuidedVisualBlock(
            img_size=(12, 12, 12),
            patch_size=(3, 3, 3),      # 12 / 3 = 4
            in_chans=128,
            embed_dim=256,
            num_heads=4,
            c=0.01
        )
        self.hyper_block_128_32 = HyperTextGuidedVisualBlock(
            img_size=(4, 4, 4),      # cube feature map
            patch_size=(2, 2, 2),    # 4 / 2 = 2 → 2×2×2 = 8 tokens
            in_chans=128,
            embed_dim=256,
            num_heads=2,
            c=0.005
        )


        self.hyper_block_256 = HyperTextGuidedVisualBlock(
            img_size=(6, 6, 6),
            patch_size=(3, 3, 3),      # 6 / 3 = 2
            in_chans=256,
            embed_dim=256,
            num_heads=4,
            c=0.01
        )
        self.hyper_block_256_32 = HyperTextGuidedVisualBlock(
            img_size=(2, 2, 2),
            patch_size=(1, 1, 1),    # 2 / 1 = 2 → 2×2×2 = 8 tokens
            in_chans=256,
            embed_dim=256,
            num_heads=2,
            c=0.005
        )

        

        # 邻接关系
        self.slice_attn_64_D = InterSliceTextGuidedAttention(
            in_dim=64,
            q_k_dim=32,
            patch_ini=(24, 24, 24),
            axis='D',
            text_dim=256
        )
        self.slice_attn_64_D_32 = InterSliceTextGuidedAttention(
            in_dim=64,
            q_k_dim=16,              # ⬅️ 明显减小
            patch_ini=(8, 8, 8),     # cube feature map
            axis='D',
            text_dim=256
        )

        self.slice_attn_128_D = InterSliceTextGuidedAttention(
            in_dim=128,
            q_k_dim=64,
            patch_ini=(12, 12, 12),
            axis='D',
            text_dim=256
        )
        self.slice_attn_128_D_32 = InterSliceTextGuidedAttention(
            in_dim=128,
            q_k_dim=8,               # ⬅️ 再降一档
            patch_ini=(4, 4, 4),
            axis='D',
            text_dim=256
        )

        self.slice_attn_256_D = InterSliceTextGuidedAttention(
            in_dim=256,
            q_k_dim=128,
            patch_ini=(6, 6, 6),
            axis='D',
            text_dim=256
        )

        # 融合
        self.fusion_64 = FeatureFusionGate(
            channels=64,
            use_proj=True
        )

        self.fusion_128 = FeatureFusionGate(
            channels=128,
            use_proj=True
        )

        self.fusion_256 = FeatureFusionGate(
            channels=256,
            use_proj=True
        )

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256 + 256, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=1)
        )

        self.features_to_64 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=1)
        )

        self.features_to_128 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1)
        )

        self.features_to_256 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=1)
        )

    def forward_encoder(self, x, output_embedding=False):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)


        self.text_embedding = self.text_embedding.cuda()
        # ① 对文本做双曲，然后融合原始文本特征
        
        ori_embedding_256 = self.text_to_256(self.text_embedding)  # [32, 256]
        text_embedding_256 = self.text_hyper(ori_embedding_256)          # [32, 512]
        # text_embedding_256 = self.text_to_256(text_embedding_hyper)  # [32, 256]
        clip_embedding = self.text_to_vision(torch.cat([text_embedding_256[:self.num_classes],
                                                        text_embedding_256[self.num_classes:]], dim=1))  # [16, 256]
        if x5.shape[0] == 4 or x5.shape[0] == 15: # 常规是 batch size 4
            # print('All:', 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape)
            # 2️⃣ 融合层次关系
            feat_64_hyper = self.hyper_block_64(visual_feat=x3,text_emb=text_embedding_256)
            feat_128_hyper = self.hyper_block_128(visual_feat=x4,text_emb=text_embedding_256)
            feat_256_hyper = self.hyper_block_256(visual_feat=x5,text_emb=text_embedding_256)
            # res = [x1, x2, x3+feat_64_hyper, x4+feat_128_hyper, x5+feat_256_hyper]

            # 3️⃣ 融合邻切片关系
            feat_64_slice = self.slice_attn_64_D(x3, text_embedding_256)
            feat_128_slice = self.slice_attn_128_D(x4, text_embedding_256)
            feat_256_slice = self.slice_attn_256_D(x5, text_embedding_256)

            # 4️⃣ 融合 层次关系和邻切片关系
            feat_64 = self.fusion_64(feat_64_hyper, feat_64_slice)
            feat_128 = self.fusion_128(feat_128_hyper, feat_128_slice)
            feat_256 = self.fusion_256(feat_256_hyper, feat_256_slice)
            res = [x1, x2, x3+feat_64, x4+feat_128, x5+feat_256]
        elif x5.shape[0] == 27:  # cube 输入时的 batch size
            # print('cube:', 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape)
            feat_64_hyper = self.hyper_block_64_32(visual_feat=x3,text_emb=text_embedding_256)
            feat_128_hyper = self.hyper_block_128_32(visual_feat=x4,text_emb=text_embedding_256)
            feat_256_hyper = self.hyper_block_256_32(visual_feat=x5,text_emb=text_embedding_256)

            feat_64_slice  = self.slice_attn_64_D_32(x3, text_embedding_256)
            feat_128_slice = self.slice_attn_128_D_32(x4, text_embedding_256)

            feat_64  = self.fusion_64(feat_64_hyper,  feat_64_slice)
            feat_128 = self.fusion_128(feat_128_hyper, feat_128_slice)
            feat_256 = self.fusion_256(feat_256_hyper, x5)
            res = [x1, x2, x3+feat_64, x4+feat_128, x5+feat_256]
        else:
            print('batch=2:', 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape)
            # 2️⃣ 融合层次关系
            feat_64_hyper = self.hyper_block_64(visual_feat=torch.cat([x3, x3], dim=0),text_emb=text_embedding_256)
            feat_128_hyper = self.hyper_block_128(visual_feat=torch.cat([x4, x4], dim=0),text_emb=text_embedding_256)
            feat_256_hyper = self.hyper_block_256(visual_feat=torch.cat([x5, x5], dim=0),text_emb=text_embedding_256)
            # res = [x1, x2, x3+feat_64_hyper, x4+feat_128_hyper, x5+feat_256_hyper]

            # 3️⃣ 融合邻切片关系
            feat_64_slice = self.slice_attn_64_D(torch.cat([x3, x3], dim=0), text_embedding_256)
            feat_128_slice = self.slice_attn_128_D(torch.cat([x4, x4], dim=0), text_embedding_256)
            feat_256_slice = self.slice_attn_256_D(torch.cat([x5, x5], dim=0), text_embedding_256)

            # 4️⃣ 融合 层次关系和邻切片关系
            feat_64 = self.fusion_64(feat_64_hyper, feat_64_slice)[:x3.shape[0]-1]
            feat_128 = self.fusion_128(feat_128_hyper, feat_128_slice)[:x3.shape[0]-1]
            feat_256 = self.fusion_256(feat_256_hyper, feat_256_slice)[:x3.shape[0]-1]

            res = [x1, x2, x3+feat_64, x4+feat_128, x5+feat_256]

        if output_embedding:
            features_embedding_list = [   #                              原始尺寸的输入             方块的输入
                    self.features_to_64(x3+feat_64),  # 改变通道数到 64， 原本是 64。   [4, 64, 24, 24, 24]     [27, 64, 8, 8, 8]
                    self.features_to_128(x4+feat_128),  # 改变通道数到 128， 原本是 128。[4, 128, 12, 12, 12]    [27, 128, 4, 4, 4]
                    self.features_to_256(x5+feat_256)   # 改变通道数到 256， 原本是 256。 [4, 256, 6, 6, 6]       [27, 256, 2, 2, 2]
                ]
            # if x5.shape[0] == 4 or x5.shape[0] == 27:
            #     features_embedding_list = [   #                              原始尺寸的输入             方块的输入
            #         self.features_to_64(x3+feat_64),  # 改变通道数到 64， 原本是 64。   [4, 64, 24, 24, 24]     [27, 64, 8, 8, 8]
            #         self.features_to_128(x4+feat_128),  # 改变通道数到 128， 原本是 128。[4, 128, 12, 12, 12]    [27, 128, 4, 4, 4]
            #         self.features_to_256(x5+feat_256)   # 改变通道数到 256， 原本是 256。 [4, 256, 6, 6, 6]       [27, 256, 2, 2, 2]
            #     ]
            # else:  
            #     features_embedding_list = [   #                              原始尺寸的输入             方块的输入
            #         self.features_to_64(x3),  # 改变通道数到 64， 原本是 64。   [4, 64, 24, 24, 24]     [27, 64, 8, 8, 8]
            #         self.features_to_128(x4),  # 改变通道数到 128， 原本是 128。[4, 128, 12, 12, 12]    [27, 128, 4, 4, 4]
            #         self.features_to_256(x5)   # 改变通道数到 256， 原本是 256。 [4, 256, 6, 6, 6]       [27, 256, 2, 2, 2]
            #     ]

            text_embedding_list = [
                self.text_to_64(self.text_embedding),
                self.text_to_128(self.text_embedding),
                text_embedding_256,
            ]
        else:
            features_embedding_list = None
            text_embedding_list = None

        return res, features_embedding_list, text_embedding_list, clip_embedding

    def forward_decoder(self, features):
        x5_up = self.block_five_up(features[4])
        x5_up = x5_up + features[3]

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + features[2]

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + features[1]

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + features[0]
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        return x9

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_classes = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_classes * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_classes * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_classes):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_classes
            )
            if i < n_layers - 1:
                # x = self.head_norm(x)
                x = F.relu(x, inplace=True)
        return x

    def forward_prediction_head(self, feature_list, clip_embedding, decoder_output):
        # original
        feat = (self.Gap_64(feature_list[-3]) + self.GAP_128(feature_list[-2]) + self.GAP_256(feature_list[-1])) / 3

        batch_size = feat.shape[0]
        logits_array = []

        for i in range(batch_size):
            vision_language_embedding = torch.cat([feat[i].unsqueeze(0).repeat(self.num_classes, 1, 1, 1, 1),
                                                   clip_embedding.unsqueeze(2).unsqueeze(2).unsqueeze(2)], dim=1)
            params = self.controller(vision_language_embedding)
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1)

            head_inputs = self.precls_conv(decoder_output[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.num_classes, 1, 1, 1, 1)  # [16, 8, D, H, W]

            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W)  # [1, 16*8, D, H, W]
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))

        out_seg = torch.cat(logits_array, dim=0)

        return out_seg

    def forward(self, input, output_embedding=False):
        features, features_embedding_list, text_embedding_list, clip_embedding = self.forward_encoder(input,
                                                                                                      output_embedding)
        decoder_output = self.forward_decoder(features)

        return (self.forward_prediction_head(features, clip_embedding, decoder_output),
                features_embedding_list, text_embedding_list)





if __name__ == '__main__':
    pass
