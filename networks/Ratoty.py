import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助函数 (保持不变) ---
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, freqs):
    # q: [..., Head_Dim], freqs: [..., Head_Dim]
    # 自动广播机制会处理中间维度的差异
    cos = freqs.cos()
    sin = freqs.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

class VisualFeatureFuser(nn.Module):
    def __init__(self, target_dim=256, target_size=(6, 6, 6)):
        super().__init__()
        self.target_size = target_size
        self.proj1 = nn.Conv3d(64, target_dim, kernel_size=1)
        self.proj2 = nn.Conv3d(128, target_dim, kernel_size=1)
        self.proj3 = nn.Conv3d(256, target_dim, kernel_size=1)

    def forward(self, feats):
        f1, f2, f3 = feats
        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)
        f1 = F.interpolate(f1, size=self.target_size, mode='trilinear', align_corners=False)
        f2 = F.interpolate(f2, size=self.target_size, mode='trilinear', align_corners=False)
        return f1 + f2 + f3

class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, head_dim, max_shape=(6, 6, 6), base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.z_dim = head_dim // 3
        self.y_dim = head_dim // 3
        self.x_dim = head_dim - self.z_dim - self.y_dim
        self.base = base
        self.register_buffer('inv_freq_z', self._get_inv_freq(self.z_dim))
        self.register_buffer('inv_freq_y', self._get_inv_freq(self.y_dim))
        self.register_buffer('inv_freq_x', self._get_inv_freq(self.x_dim))

    def _get_inv_freq(self, dim):
        return 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x):
        # x: [B, nH, D, H, W, hD]
        D, H, W = x.shape[2:5]
        device = x.device
        z_pos = torch.arange(D, device=device).type_as(self.inv_freq_z)
        y_pos = torch.arange(H, device=device).type_as(self.inv_freq_y)
        x_pos = torch.arange(W, device=device).type_as(self.inv_freq_x)
        
        freqs_z = torch.einsum('i,j->ij', z_pos, self.inv_freq_z)[:, None, None, :]
        freqs_y = torch.einsum('i,j->ij', y_pos, self.inv_freq_y)[None, :, None, :]
        freqs_x = torch.einsum('i,j->ij', x_pos, self.inv_freq_x)[None, None, :, :]
        
        emb_z = torch.cat((freqs_z, freqs_z), dim=-1).expand(D, H, W, -1)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1).expand(D, H, W, -1)
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1).expand(D, H, W, -1)
        
        emb = torch.cat([emb_z, emb_y, emb_x], dim=-1)
        return emb[None, None, ...] # [1, 1, D, H, W, hD]


# --- 核心修正模块 ---

class TextGeoMapper(nn.Module):
    """
    修正后的文本几何解析器
    """
    def __init__(self, text_dim, head_dim, num_heads):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),  # 512 -> 256
            nn.GELU(),
            # 修正点1: 输入是 text_dim // 2 (256)
            # 修正点2: 输出是 num_heads * head_dim，确保每个 Head 有独立的空间感知
            nn.Linear(text_dim // 2, num_heads * head_dim) 
        )

    def forward(self, text_emb):
        """
        text_emb: [B, Seq, C]
        Return: [B, nH, Seq, hD] (对应 K 的形状)
        """
        B, Seq, _ = text_emb.shape
        
        # 1. 预测位置/相位偏置
        # Output: [B, Seq, num_heads * head_dim]
        bias = self.proj(text_emb) 
        
        # 2. 调整形状以匹配 Multi-Head Attention 的 Key
        # [B, Seq, nH, hD] -> [B, nH, Seq, hD]
        bias = bias.view(B, Seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        return bias

class AnatomicalSpaceAttention(nn.Module):
    def __init__(self, visual_dim=256, text_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 3D 旋转位置编码 (用于 Visual)
        self.rope3d = RotaryPositionEmbedding3D(self.head_dim, max_shape=(6, 6, 6))
        
        # [新增] 文本几何解析器 (用于 Text)
        # 传入 num_heads，确保多头多样性
        self.text_geo_mapper = TextGeoMapper(text_dim, self.head_dim, num_heads)
        
        self.q_proj = nn.Linear(visual_dim, visual_dim)
        self.k_proj = nn.Linear(text_dim, visual_dim)
        self.v_proj = nn.Linear(text_dim, visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, fused_visual, text_embedding):
        B = text_embedding.shape[0]
        
        # --- 1. 视觉特征处理 ---
        # fused_visual = self.fuser(visual_feats) # [B, 256, 6, 6, 6]
        fused_visual = fused_visual.permute(0, 2, 3, 4, 1) # [B, D, H, W, C]
        D, H, W, C = fused_visual.shape[1:]
        
        # 生成 Visual Query: [B, nH, D, H, W, hD]
        q = self.q_proj(fused_visual).view(B, D, H, W, self.num_heads, self.head_dim)
        q = q.permute(0, 4, 1, 2, 3, 5) 
        
        # --- 2. 文本特征处理 ---
        # 生成 Semantic Key: [B, nH, Seq, hD]
        k = self.k_proj(text_embedding).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 生成 Value: [B, nH, Seq, hD]
        v = self.v_proj(text_embedding).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # --- 3. 注入位置信息 ---
        
        # A. 视觉侧：物理坐标旋转
        visual_freqs = self.rope3d(q) # [1, 1, D, H, W, hD]
        q_rotated = apply_rotary_pos_emb(q, visual_freqs)
        
        # B. 文本侧：预测的语义坐标旋转
        # text_phase: [B, nH, Seq, hD]
        text_phase = self.text_geo_mapper(text_embedding) 
        
        # 对 K 进行旋转。注意：这里不需要 squeeze/unsqueeze 那些复杂的 1,1,1
        # 因为 text_phase 和 k 的形状完全一致 [B, nH, Seq, hD]
        # apply_rotary_pos_emb 会直接进行 element-wise 运算
        k_rotated = apply_rotary_pos_emb(k, text_phase)
        
        # --- 4. 交叉注意力 ---
        # 展平 Visual: [B, nH, D*H*W, hD]
        q_flat = q_rotated.flatten(2, 4) 
        
        # 计算 Attention Score
        # Q (Visual Grid Pos) dot K (Text Predicted Pos)
        attn_weights = torch.matmul(q_flat, k_rotated.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 聚合
        attn_out = torch.matmul(attn_weights, v)
        
        # 恢复形状
        attn_out = attn_out.view(B, self.num_heads, D, H, W, self.head_dim)
        attn_out = attn_out.permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, C)
        output = self.out_proj(attn_out).permute(0, 4, 1, 2, 3)
        
        return output

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入数据
    feat3 = torch.randn(4, 256, 6, 6, 6)
    
    # 模拟文本 Embedding
    text_emb = torch.randn(4, 16, 512)
    
    # 实例化模块
    model = AnatomicalSpaceAttention(visual_dim=256, text_dim=512)
    
    # 前向传播
    output = model(feat3, text_emb)
    
    print("输入特征尺寸:", feat3.shape)
    print("文本特征尺寸:", text_emb.shape)
    print("输出特征尺寸:", output.shape) # 预期: [4, 256, 6, 6, 6]