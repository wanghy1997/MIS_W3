import torch
from torch import nn
import torch.nn.functional as F
from networks.semantic_extraction_all import SemanticInteraction
from networks.semantic_extraction_2 import MultiScaleFusion3D, CA_TransformerDecoderLayer


"""
此段代码的思路：
1. 单个器官的文本与视觉特征进行交互，
2. 多个器官的文本与视觉特征进行交互，
"""
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
        self.out_ = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)


        # # CLIP
        if n_classes == 16: 
            # self.text_embedding = torch.load(self.text_path, weights_only=True).float()
            # # print('loaded shape&location embedding:', self.text_embedding.shape)  # [32, 512]
            # text_embedding_dim = self.text_embedding.shape[-1]

            # 适用于 AMOS 的 CLIP 文本嵌入，16 类别
            # ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_16_baseline.pt")
            ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_16_ours.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [32, 512]
            self.text_affinity = ckpt["tokens"].float()   # [32, 256, 256]
            text_embedding_dim = self.text_embedding.shape[-1]
            self.text_affinity = self.text_affinity.cuda()
        elif n_classes == 14:
            # UniCLIP BiomedBERT  适用于 Synapse 和 FLARE 13 类 + 背景
            # ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_14_baseline.pt")
            ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_14_ours.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [28, 512]
            self.text_affinity = ckpt["tokens"].float()   # [28, 256, 256]
            text_embedding_dim = self.text_embedding.shape[-1]
            self.text_affinity = self.text_affinity.cuda()
        elif n_classes == 9:
            # UniCLIP BiomedBERT  适用于 Synapse 和 FLARE 13 类 + 背景
            ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_9_ours.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [18, 512]
            self.text_affinity = ckpt["tokens"].float()   # [18, 256, 256]
            text_embedding_dim = self.text_embedding.shape[-1]
            self.text_affinity = self.text_affinity.cuda()
        elif n_classes == 8:
            # UniCLIP BiomedBERT  适用于 Synapse 和 FLARE 13 类 + 背景
            ckpt = torch.load("/home/why/TAK-Semi-main/CLIP/biomedbert_text_embeddings_MMWHS.pt")
            self.text_embedding = ckpt["embeddings"].float()   # [16, 512]
            self.text_affinity = ckpt["tokens"].float()   # [16, 256, 256]
            text_embedding_dim = self.text_embedding.shape[-1]
            self.text_affinity = self.text_affinity.cuda()
        
        self.text_to_256 = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.text_to_vision = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
        )


        self.text_to_64 = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.ReLU(inplace=True),
        )

        self.text_to_128 = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
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

        self.SemInter1 = SemanticInteraction(  # 文本-视觉语义交互模块
                image_dim=256, text_dim=256,
                num_heads=4,      # 我更推荐 4：对 Nv=8 不会“分头太细”，Nv=216 也够用
                drop=0.0, attn_drop=0.0, drop_path=0.0,
                has_mlp=False
            )
        # self.attn_x5 = CA_TransformerDecoderLayer(in_channels=256, d_model=512, nhead=8, dropout=0.1, patch_size=(6, 6, 6))
        # self.ms_fusion = MultiScaleFusion3D([64, 128, 256],256)
        """
        raw1, out1 = self.SemInter(torch.randn(4, 256, 6, 6, 6),  text)  # OK
        raw2, out2 = self.SemInter(torch.randn(27,256, 2, 2, 2),  text)  # OK
        print(raw1.shape, out1.shape)  # [4,256], [4,128]
        print(raw2.shape, out2.shape)  # [27,256], [27,128]
        """


    # def forward_encoder(self, x, output_embedding=False):
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
        
        res = [x1, x2, x3, x4, x5]

        self.text_embedding = self.text_embedding.cuda()
        self.text_embedding_256 = self.text_to_256(self.text_embedding).cuda()  # [32, 256]
        self.clip_embedding = self.text_to_vision(torch.cat([self.text_embedding_256[:self.num_classes],
                                                        self.text_embedding_256[self.num_classes:]], dim=1))  # [16, 256]
        # 只用 class tokens：前 num_classes
        T_class = self.text_embedding_256[:self.num_classes]          # [K,256]
        vis_256 = self.features_to_256(x5)                       # [B,256,D,H,W]
        v_raw, v_out = self.SemInter1(vis_256, T_class)           # v_raw[B,256], v_out[B,128]

        sem_pack = (v_raw, v_out, T_class)   # 注意：DP 下 T_class 可能会被 gather 拼接，训练里再去重


        if output_embedding:
            features_embedding_list = [   #                              原始尺寸的输入             方块的输入
                self.features_to_64(x3),  # 改变通道数到 64， 原本是 64。   [4, 64, 24, 24, 24]     [27, 64, 8, 8, 8]
                self.features_to_128(x4),  # 改变通道数到 128， 原本是 128。[4, 128, 12, 12, 12]    [27, 128, 4, 4, 4]
                self.features_to_256(x5)  # 改变通道数到 256， 原本是 256。 [4, 256, 6, 6, 6]       [27, 256, 2, 2, 2]
            ]

            text_embedding_list = [
                self.text_to_64(self.text_embedding),
                self.text_to_128(self.text_embedding),
                self.text_embedding_256,
            ]
        else:
            features_embedding_list = None
            text_embedding_list = None

        # return res, features_embedding_list, text_embedding_list, clip_embedding
        return res, features_embedding_list, text_embedding_list, self.clip_embedding, sem_pack

    def forward_decoder(self, features):
        # x_rel = self.ms_fusion([features[2], features[3], features[4]])        # 多尺度
        # txt_embed = self.text_embedding[self.num_classes:]
        # txt_embed = txt_embed.unsqueeze(0).expand(features[4].shape[0], -1, -1).contiguous()
        # if features[4].shape[-3:] != (6, 6, 6):  # 多视角 × 多器官文本  torch.Size([4, 256, 6, 6, 6]) torch.Size([4, 16, 512])
        #     x_rel = F.interpolate(features[4], size=(6, 6, 6), mode="trilinear", align_corners=True)
        #     x_rel = self.attn_x5(x_rel, txt_embed)
        #     x_rel = F.interpolate(x_rel, size=features[4].shape[-3:], mode="trilinear", align_corners=True)
        # else:
        #     x_rel = self.attn_x5(x_rel, txt_embed)
        # features[4] = features[4] + x_rel                              # 只增强特征

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


    def forward(self, input, output_embedding=False):
        features, features_embedding_list, text_embedding_list, _, sem_pack = self.forward_encoder(
            input, output_embedding
        )
        decoder_output = self.forward_decoder(features)

        out_seg = self.out_(decoder_output)
        return out_seg, features_embedding_list, text_embedding_list, sem_pack



if __name__ == '__main__':
    pass
