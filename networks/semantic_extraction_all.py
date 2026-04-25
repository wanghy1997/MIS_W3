import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class SemanticInteraction(nn.Module):
    """
    适配你的输入：
      - image: [B, C, D, H, W]  (例如 [4,256,6,6,6] 或 [27,256,2,2,2])
              或者已经展平的 [B, N, C]
      - text : [T, C] (例如 [16,256]) 或 [B, T, C]

    计算流程（两级交互，序列版 cross-attention）：
      1) 图像 tokens <- text tokens (f_vt)
      2) text tokens <- 图像 tokens (f_tv)
      3) 取图像/text 的全局 token（mean pooling），融合得到 f_c
      4) f_c 分别去 attend f_vt / f_tv，再相加得到 raw_out

    输出：
      - raw_out: [B, image_dim]
      - out    : [B, image_dim//2]
    """

    def __init__(
        self,
        image_dim: int = 256,
        text_dim: int = 256,
        num_heads: int = 8,
        stem_channel: int = 16,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        has_mlp: bool = False,
    ):
        super().__init__()

        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim)
        self.num_heads = int(num_heads)

        # 统一到同一维度上做 cross-attention
        if self.text_dim != self.image_dim:
            self.text_proj = nn.Linear(self.text_dim, self.image_dim)
        else:
            self.text_proj = nn.Identity()

        # level-1 cross attention
        self.cross_att_one_level_v = CrossAttentionBlock(
            self.image_dim,
            num_heads=self.num_heads,
            stem_channel=stem_channel,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            has_mlp=has_mlp,
        )
        self.cross_att_one_level_t = CrossAttentionBlock(
            self.image_dim,
            num_heads=self.num_heads,
            stem_channel=stem_channel,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            has_mlp=has_mlp,
        )

        # 融合全局 token（替代原来的 2D co_conv）
        self.fuse_linear = nn.Linear(self.image_dim * 2, self.image_dim)

        # level-2 cross attention (query=1 token)
        self.cross_att_two_level_v = CrossAttentionBlock(
            self.image_dim,
            num_heads=self.num_heads,
            stem_channel=stem_channel,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            has_mlp=has_mlp,
        )
        self.cross_att_two_level_t = CrossAttentionBlock(
            self.image_dim,
            num_heads=self.num_heads,
            stem_channel=stem_channel,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            has_mlp=has_mlp,
        )

        # 输出头：image_dim -> 2*image_dim -> image_dim//2
        self.linear_1 = nn.Linear(self.image_dim, self.image_dim * 2)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.image_dim * 2, self.image_dim // 2)

    @staticmethod
    def _image_to_tokens(image: torch.Tensor) -> torch.Tensor:
        """[B,C,D,H,W] -> [B,N,C]  or keep [B,N,C]."""
        if image.dim() == 4:
            # [C,D,H,W] -> [1,N,C]
            image = image.unsqueeze(0)
        if image.dim() == 5:
            b, c, d, h, w = image.shape
            x = image.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)
            return x
        if image.dim() == 3:
            # [B,N,C]
            return image
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    @staticmethod
    def _text_to_tokens(text: torch.Tensor, batch_size: int) -> torch.Tensor:
        """[T,C] -> [B,T,C]  or keep [B,T,C]."""
        if text.dim() == 2:
            text = text.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            return text
        if text.dim() == 3:
            if text.shape[0] != batch_size:
                # 允许传入 [1,T,C]
                if text.shape[0] == 1:
                    return text.expand(batch_size, -1, -1).contiguous()
                raise ValueError(
                    f"Batch mismatch: image batch={batch_size}, text batch={text.shape[0]}"
                )
            return text
        raise ValueError(f"Unsupported text shape: {tuple(text.shape)}")

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        # 1) token 化
        f_v = self._image_to_tokens(image)  # [B,Nv,C]
        b = f_v.shape[0]
        f_t = self._text_to_tokens(text, b)  # [B,Nt,?]
        f_t = self.text_proj(f_t)  # -> [B,Nt,C]

        # 2) level-1 cross attention
        f_vt = self.cross_att_one_level_v(f_v, f_t)  # [B,Nv,C]
        f_tv = self.cross_att_one_level_t(f_t, f_v)  # [B,Nt,C]

        # 3) global token 融合得到 f_c (query token)
        v_global = f_v.mean(dim=1, keepdim=True)  # [B,1,C]
        t_global = f_t.mean(dim=1, keepdim=True)  # [B,1,C]
        f_c = self.fuse_linear(torch.cat([v_global, t_global], dim=-1))  # [B,1,C]

        # 4) level-2：f_c 去 attend 两个方向的交互结果
        f_vt_dot = self.cross_att_two_level_v(f_c, f_vt)  # [B,1,C]
        f_tv_dot = self.cross_att_two_level_t(f_c, f_tv)  # [B,1,C]

        raw_out = (f_vt_dot + f_tv_dot).squeeze(1)  # [B,C]

        out = self.linear_1(raw_out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.linear_2(out)  # [B,C//2]

        return raw_out, out


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CrossAttention(nn.Module):
    """Sequence-based cross attention.

    x: [B, Nx, C] as query
    y: [B, Ny, C] as key/value
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        stem_channel=16,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: [B,Nx,C], y: [B,Ny,C]
        out, _ = self.mha(query=x, key=y, value=y, need_weights=False)
        out = self.proj_drop(out)
        return out


class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        stem_channel,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        has_mlp=False,
    ):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            stem_channel=stem_channel,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            # 保持维度不变，便于残差
            self.mlp = MLP(dim, mlp_hidden_dim, dim, num_layers=2)

    def forward(self, x, y):
        # x attends to y
        out = x + self.drop_path(self.attn(self.norm_x(x), self.norm_y(y)))
        if self.has_mlp:
            out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out
