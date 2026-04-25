import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=7):
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
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)

        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        project_out = not (num_heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, p1, p2):
        q_p1 = self.w_q(p1)
        k_p2 = self.w_k(p2)
        v_p2 = self.w_v(p2)
        q_p1 = rearrange(q_p1, 'b n (h d) -> b h n d', h=self.num_heads)
        k_p2 = rearrange(k_p2, 'b n (h d) -> b h n d', h=self.num_heads)
        v_p2 = rearrange(v_p2, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_p1p2 = einsum('b h i d, b h j d -> b h i j', q_p1, k_p2) * self.scale
        attn_p1p2 = attn_p1p2.softmax(dim=-1)

        attn_p1p2 = einsum('b h i j, b h j d -> b h j d', attn_p1p2, v_p2)
        attn_p1p2 = rearrange(attn_p1p2, 'b h n d -> b n (h d)')

        attn_p1p2 = self.to_out(attn_p1p2)
        return attn_p1p2


class MultiPerspectiveFusion(nn.Module):
    def __init__(
            self,
            input_size,
            in_channels,
            d_model,

            nhead=8,
            dropout=0.1,
            patch_size=[7, 7, 5],
    ):
        super().__init__()

        # patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding_Axial = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[1], p2=patch_size[2]),
            nn.Linear(in_channels * patch_size[1] * patch_size[2], d_model)
        )
        self.to_patch_embedding_Sagittal = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[2]),
            nn.Linear(in_channels * patch_size[0] * patch_size[2], d_model)
        )
        self.to_patch_embedding_Coronal = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(in_channels * patch_size[0] * patch_size[1], d_model)
        )
        
        # Positional encodings for Axial, Sagittal, Coronal views
        self.positional_encoding_Axial = nn.Parameter(torch.randn(1, (input_size[1] // patch_size[1]) * (input_size[2] // patch_size[2]), d_model))
        self.positional_encoding_Sagittal = nn.Parameter(torch.randn(1, (input_size[0] // patch_size[0]) * (input_size[2] // patch_size[2]), d_model))
        self.positional_encoding_Coronal = nn.Parameter(torch.randn(1, (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1]), d_model))
        
        self.view_weights = nn.Parameter(torch.ones(3))

        self.img_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.img_txt_attn = CA_Attention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.to_out_Axial = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[1] * patch_size[2]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[1] // patch_size[1]),
                      w=(input_size[2] // patch_size[2]), p1=patch_size[1], p2=patch_size[2]),
        )
        self.to_out_Sagittal = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[0] * patch_size[2]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[0] // patch_size[0]),
                      w=(input_size[2] // patch_size[2]), p1=patch_size[0], p2=patch_size[2]),
        )
        self.to_out_Coronal = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[0] * patch_size[1]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[0] // patch_size[0]),
                      w=(input_size[1] // patch_size[1]), p1=patch_size[0], p2=patch_size[1]),
        )


        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1)

    def cross_attn(self, x, text_feature, view):
        
        if view == "Axial":
            q = self.to_patch_embedding_Axial(x)
            q = q + self.positional_encoding_Axial
        elif view == "Sagittal":
            q = self.to_patch_embedding_Sagittal(x)
            q = q + self.positional_encoding_Sagittal
        else:  # Coronal 
            q = self.to_patch_embedding_Coronal(x)
            q = q + self.positional_encoding_Coronal

            
        q = k = v = self.norm1(q)
        q = q + self.img_attn(q, k, v)
        q = self.norm2(q)


        x = q + self.img_txt_attn(text_feature, q)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        if view == "Axial":
            x = self.to_out_Axial(x)
        elif view == "Sagittal":
            x = self.to_out_Sagittal(x)
        elif view == "Coronal":
            x = self.to_out_Coronal(x)
        return x, q

    def forward(self, img_feats, txt_emebd):
        # diff views
        # Axial_view = img_feats.mean(dim=2)
        # Sagittal_view = img_feats.mean(dim=3)
        # Coronal_view = img_feats.mean(dim=4)
        Axial_view = F.adaptive_avg_pool3d(img_feats, (1, img_feats.size(3), img_feats.size(4))).squeeze(2)
        Sagittal_view = F.adaptive_avg_pool3d(img_feats, (img_feats.size(2), 1, img_feats.size(4))).squeeze(3)
        Coronal_view = F.adaptive_avg_pool3d(img_feats, (img_feats.size(2), img_feats.size(3), 1)).squeeze(4)


        # views attn
        Axial_view_attn_map, _ = self.cross_attn(Axial_view, txt_emebd, view="Axial")
        Sagittal_view_attn_map, _ = self.cross_attn(Sagittal_view, txt_emebd, view="Sagittal")
        Coronal_view_attn_map, _ = self.cross_attn(Coronal_view, txt_emebd, view="Coronal")

        # view_3D = Axial_view_attn_map.unsqueeze(2) + Sagittal_view_attn_map.unsqueeze(3) + Coronal_view_attn_map.unsqueeze(4)
        view_3D = (
                    self.view_weights[0] * Axial_view_attn_map.unsqueeze(2) +
                    self.view_weights[1] * Sagittal_view_attn_map.unsqueeze(3) +
                    self.view_weights[2] * Coronal_view_attn_map.unsqueeze(4)
                )
        
        return img_feats + view_3D