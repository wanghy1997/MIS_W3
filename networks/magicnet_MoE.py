import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


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


class TokenLearner_Local(nn.Module):
    """
    Image to local patch tokens
    Input : [B*C, 1, D, H, W]
    Output: [B*C, N, token_dim]
    """
    def __init__(self, img_size=(4, 8, 8), patch_size=(2, 2, 2), in_chans=1, embed_dim=8):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * \
                      (img_size[1] // patch_size[1]) * \
                      (img_size[2] // patch_size[2])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):  # x: [B*C, 1, D, H, W]
        x = self.proj(x)          # [B*C, embed_dim, D/p1, H/p2, W/p3]
        x = x.flatten(2)          # [B*C, embed_dim, N]
        x = x.transpose(1, 2)     # [B*C, N, embed_dim]
        return x
    

class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
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
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock  ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x9


class FcLayer(nn.Module):
    def __init__(self, ts=32, patch_size=96, n_filters=16):
        super(FcLayer, self).__init__()
        nt = patch_size // ts
        self.fc_layer = nn.Sequential(
            nn.Linear((n_filters * 16) * ((ts // 16) ** 3), 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, nt ** 3)
        )

    def forward(self, x):
        return self.fc_layer(x)


class LocalFeatureExtractor3D(nn.Module):
    """
    Local feature extractor adapted from SKCDF local tokenization.

    Input:
        x: [B, C, D, H, W]
    Output:
        local_feat: [B, C, D, H, W]

    Purpose:
        Extract dense local anatomical features while keeping the same
        spatial size and channel size as the encoder last-layer feature.
    """
    def __init__(
        self,
        in_channels=256,
        img_size=(4, 8, 8),
        patch_size=(2, 2, 2),
        token_dim=None,
        dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.token_dim = token_dim if token_dim is not None else int(np.prod(patch_size))

        # local tokenization
        self.token_learner = TokenLearner_Local(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=self.token_dim
        )

        # token refinement
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, self.token_dim)
        )

        # dense reconstruction + local fusion
        self.local_fuse = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        b, c, d, h, w = x.shape
        p1, p2, p3 = self.patch_size

        assert (d, h, w) == self.img_size, \
            f"Expected input size {self.img_size}, but got {(d, h, w)}"
        assert d % p1 == 0 and h % p2 == 0 and w % p3 == 0, \
            f"Input size {(d, h, w)} must be divisible by patch size {self.patch_size}"

        dg, hg, wg = d // p1, h // p2, w // p3

        # [B, C, D, H, W] -> [B*C, 1, D, H, W]
        x_bc = x.contiguous().view(b * c, d, h, w).unsqueeze(1)

        # local tokens: [B*C, N, token_dim]
        tokens = self.token_learner(x_bc)

        # token refinement
        tokens = self.token_norm(tokens)
        tokens = tokens + self.token_mlp(tokens)

        # restore dense 3D map
        # [B*C, N, token_dim] -> [B, C, D, H, W]
        local_feat = rearrange(
            tokens,
            '(b c) (dg hg wg) (p1 p2 p3) -> b c (dg p1) (hg p2) (wg p3)',
            b=b, c=c, dg=dg, hg=hg, wg=wg, p1=p1, p2=p2, p3=p3
        )

        # local fusion + residual
        local_feat = self.local_fuse(local_feat)
        local_feat = local_feat + x

        return local_feat


class AnatomyAwareRouter(nn.Module):
    def __init__(self, in_channels=273, hidden_channels=64, num_experts=4, topk=2):
        super().__init__()
        self.topk = topk
        self.num_experts = num_experts

        self.router_net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, num_experts, kernel_size=1, bias=True)
        )

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        return:
            gates_soft: [B, E, D, H, W]
            gates_hard: [B, E, D, H, W]  (top-k masked)
        """
        logits = self.router_net(x)  # [B, E, D, H, W]
        gates_soft = torch.softmax(logits, dim=1)

        if self.topk is not None and self.topk < self.num_experts:
            topk_val, topk_idx = torch.topk(gates_soft, k=self.topk, dim=1)
            hard_mask = torch.zeros_like(gates_soft).scatter_(1, topk_idx, 1.0)
            gates_hard = gates_soft * hard_mask
            gates_hard = gates_hard / (gates_hard.sum(dim=1, keepdim=True) + 1e-6)
        else:
            gates_hard = gates_soft

        return gates_soft, gates_hard


class GeometryFeatureExtractor3D(nn.Module):
    """
    Extract geometry-aware cues from encoder/local features.

    Input:
        x: [B, C, D, H, W]
    Output:
        geometry_map: [B, out_channels, D, H, W]

    Design:
        1) Use anisotropic convolutions along three axes to capture directional continuity.
        2) Use gradient-like responses to highlight local boundary and structure variation.
        3) Fuse them into a compact geometry cue map.
    """
    def __init__(self, in_channels=256, out_channels=16):
        super().__init__()

        # axis-aware anisotropic branches
        self.branch_d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.branch_h = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.branch_w = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

        # local gradient-like structure response
        self.grad_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels if in_channels % out_channels == 0 else 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        gd = self.branch_d(x)
        gh = self.branch_h(x)
        gw = self.branch_w(x)
        gg = self.grad_branch(x)

        geometry_map = torch.cat([gd, gh, gw, gg], dim=1)
        geometry_map = self.fuse(geometry_map)
        return geometry_map
    

class SimpleExpert3D(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)
    

class VNet_Magic(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, cube_size=32, patch_size=96, n_filters=16, normalization='instancenorm',
                 has_dropout=False, has_residual=False):
        super(VNet_Magic, self).__init__()
        self.num_classes = n_classes
        self.topk = 4
        self.num_experts = 5
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.fc_layer = FcLayer(cube_size, patch_size)
        self.local_feature_extractor = LocalFeatureExtractor3D(
                                                    in_channels=256,
                                                    img_size=(6, 6, 6),      # 这里改成你 x5 的真实空间尺寸
                                                    patch_size=(2, 2, 2),
                                                    token_dim=8
                                                )
        self.geometry_extractor = GeometryFeatureExtractor3D(
                            in_channels=256,
                            out_channels=16
                        )
        self.anatomy_router = AnatomyAwareRouter(
                            in_channels=256+16+1,  # encoder feature + geometry cue + uncertainty map
                            hidden_channels=64,
                            num_experts=self.num_experts,
                            topk=self.topk
                        )
        self.experts = nn.ModuleList([SimpleExpert3D(256) for _ in range(self.num_experts)])

    def forward_prediction_head(self, feat):
        return self.decoder.out_conv(feat)

    def forward_encoder(self, x):
        # 4x1x96x96x96 -> 4x256x6x6x6
        # 4x1x32x32x32 -> 4x256x2x2x2(4, 2048)
        return self.encoder(x)

    def forward_decoder(self, feat_list):
        return self.decoder(feat_list)

    def compute_entropy_map(self, prob, eps=1e-6):
        # prob: [B, C, D, H, W]
        entropy = -(prob * torch.log(prob + eps)).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
        return entropy
    
    def forward(self, input, is_training=True):
        features = self.encoder(input)
        # # MoE
        # ori_feats = features[-1].clone()
        # ori_feats = self.local_feature_extractor(ori_feats)
        # # print("ori_feats shape:", ori_feats.shape)  # torch.Size([4, 256, 8, 8, 8])

        # with torch.no_grad():  # 得到不确定性 map
        #     pred, _ = self.decoder(features)
        #     uncertainty_map = self.compute_entropy_map(F.softmax(pred, dim=1))
        #     uncertainty_map = F.interpolate(
        #                             uncertainty_map,
        #                             size=ori_feats.shape[2:],
        #                             mode='trilinear',
        #                             align_corners=False
        #                         )   # [B,1,8,8,8]
        #     geometry_map = self.geometry_extractor(ori_feats)
        #     # print("pred shape:", pred.shape)         # 期望: [4, 16, 128, 128, 128]
        #     # print("ori_feats shape:", ori_feats.shape)
        #     # print("geometry_map shape:", geometry_map.shape)
        #     # print("uncertainty_map shape:", uncertainty_map.shape)
        # _, gates_hard = self.anatomy_router(torch.cat([ori_feats, geometry_map, uncertainty_map], dim=1))   # [B,273,8,8,8]  [B,4,8,8,8]
        
        # if is_training:
        #     expert_outputs = [expert(ori_feats) for expert in self.experts]
        #     expert_stack = torch.stack(expert_outputs, dim=1)   # [B,4,256,8,8,8]
        #     fused_feat = (expert_stack * gates_hard.unsqueeze(2)).sum(dim=1)
        #     refined_feats = ori_feats + fused_feat
        #     features[-1] = refined_feats
        
        #Decoder 
        out_seg, embedding = self.decoder(features)
        
        #for i in range(len(features)):
        #    print(features[i].shape)
        #print(out_seg.shape, embedding.shape)
        #os.exit()
        return out_seg, embedding  # 4, 16, 96, 96, 96


if __name__ == '__main__':
    pass
