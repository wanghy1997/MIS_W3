import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        M = k.shape[1]

        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, text_tokens, img_tokens):
        # text_tokens: [B, T, C]
        # img_tokens:  [B, N, C]
        q = rearrange(self.w_q(text_tokens), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.w_k(img_tokens),  'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.w_v(img_tokens),  'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.einsum('b h t d, b h n d -> b h t n', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h t n, b h n d -> b h t d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class CA_TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        in_channels=256,
        d_model=256,
        nhead=8,
        dropout=0.1,
        patch_size=(2, 2)
    ):
        super().__init__()

        self.to_patch = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(in_channels, d_model)
        )

        self.img_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.txt_attn = CA_Attention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.to_3d = nn.Sequential(
            nn.Linear(d_model, in_channels),
            Rearrange('b (h w) c -> b c h w', h=patch_size[0])
        )

        self.view_weights = nn.Parameter(torch.ones(3))

    def _view_interaction(self, view_3d, txt_embed):
        B, C, H, W = view_3d.shape
        tokens = self.to_patch(view_3d)                # [B, HW, C]
        tokens = tokens + self.img_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens)

        txt_enh = self.txt_attn(txt_embed, tokens)     # [B, T, C]
        tokens = tokens + txt_enh.mean(dim=1, keepdim=True)
        tokens = tokens + self.mlp(self.norm2(tokens))

        out = self.to_3d(tokens)
        return out

    def forward(self, img_feats, txt_embed):
        assert img_feats.dim() == 5
        assert txt_embed.dim() == 3

        B, C, D, H, W = img_feats.shape

        # ====== 三个解剖视角 ======
        axial    = F.adaptive_avg_pool3d(img_feats, (1, H, W)).squeeze(2)
        sagittal = F.adaptive_avg_pool3d(img_feats, (D, H, 1)).squeeze(4)
        coronal  = F.adaptive_avg_pool3d(img_feats, (D, 1, W)).squeeze(3)

        axial_enh    = self._view_interaction(axial, txt_embed).unsqueeze(2)
        sagittal_enh = self._view_interaction(sagittal, txt_embed).unsqueeze(4)
        coronal_enh  = self._view_interaction(coronal, txt_embed).unsqueeze(3)

        view_3d = (
            self.view_weights[0] * axial_enh +
            self.view_weights[1] * sagittal_enh +
            self.view_weights[2] * coronal_enh
        )

        return img_feats + view_3d
    

class MultiScaleFusion3D(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv3d(c, out_channels, kernel_size=1)
            for c in in_channels_list
        ])
        self.weight = nn.Parameter(torch.ones(len(in_channels_list)))

    def forward(self, feats):
        base_size = feats[-1].shape[2:]
        fused = 0
        for i, f in enumerate(feats):
            if f.shape[2:] != base_size:
                f = F.interpolate(f, size=base_size,
                                  mode='trilinear', align_corners=False)
            fused = fused + self.weight[i] * self.proj[i](f)
        return fused / self.weight.sum()


if __name__ == "__main__":
    # test
    attn = CA_TransformerDecoderLayer(in_channels=256, d_model=512, nhead=8, dropout=0.1, patch_size=(6, 6, 6))
    txt = torch.randn(4, 16, 512)
    img = torch.randn(4, 256, 6, 6, 6)  
    out = attn(img, txt)
    print(out.shape)  # torch.Size([4, 256, 6, 6, 6])