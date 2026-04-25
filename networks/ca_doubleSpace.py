
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
import geoopt
import numpy as np


"""
attention_fromHCMA
"""
class InterSliceSelfAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, axis='D', use_hyper: bool = False, c: float = 1.0):
        """
        初始化方法，定义了卷积层和位置嵌入。
        Parameters:
        in_dim : int # 输入张量的通道数
        q_k_dim : int # Q 和 K 向量的通道数
        axis : str # 注意力计算的轴 ('D', 'H', 'W')
        """
        super(InterSliceSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        self.use_hyper = use_hyper
        self.c = c
        D, H, W = patch_ini[0], patch_ini[1], patch_ini[2]

        # 定义卷积层
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 根据轴选择不同的位置信息嵌入
        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))  # 深度方向嵌入
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))  # 高度方向嵌入
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))  # 宽度方向嵌入
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")  # 如果轴不是 'D', 'H', 'W' 则报错

        # 使用 Xavier 初始化位置嵌入
        nn.init.xavier_uniform_(self.pos_embed)

        self.softmax = nn.Softmax(dim=-1)  # 定义 softmax 层
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练的缩放参数

    def forward(self, x, processed):
        """
        前向传播方法，计算注意力机制。
        参数：
        x : Tensor # 输入的 5D 张量 (batch, channels, depth, height, width)
        processed : Tensor # 处理过的输入张量，形状与 x 相同
        """
        B, C, D, H, W = x.size()

        processed_t = processed

        # 计算 Q, K, V（在欧式/切空间）
        Q = self.query_conv(processed_t) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        K = self.key_conv(processed_t) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        V = self.value_conv(processed_t)  # (B, in_dim, D, H, W)
        scale = math.sqrt(self.q_k_dim)  # 缩放因子

        # 根据注意力轴 ('D', 'H', 'W') 进行不同维度的处理
        if self.axis == 'D':  # 如果是深度方向
            Q = Q.permute(0, 3, 4, 2, 1).contiguous()  # 重新排列维度为 (B, H, W, D, q_k_dim)
            Q = Q.view(B * H * W, D, self.q_k_dim)  # 展平为 (B*H*W, D, q_k_dim)

            K = K.permute(0, 3, 4, 1, 2).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim, D)
            K = K.view(B * H * W, self.q_k_dim, D)  # 展平为 (B*H*W, q_k_dim, D)

            V = V.permute(0, 3, 4, 2, 1).contiguous()  # 重新排列维度为 (B, H, W, D, in_dim)
            V = V.view(B * H * W, D, self.in_dim)  # 展平为 (B*H*W, D, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*H*W, D, D)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*H*W, D, in_dim)
            out = out.view(B, H, W, D, self.in_dim)  # 恢复为原始形状 (B, H, W, D, in_dim)
            out = out.permute(0, 4, 3, 1, 2).contiguous()  # 最终输出形状 (B, C, D, H, W)

        elif self.axis == 'H':  # 如果是高度方向
            Q = Q.permute(0, 2, 4, 3, 1).contiguous()  # 重新排列维度为 (B, D, W, H, q_k_dim)
            Q = Q.view(B * D * W, H, self.q_k_dim)  # 展平为 (B*D*W, H, q_k_dim)

            K = K.permute(0, 2, 4, 1, 3).contiguous()  # 重新排列维度为 (B, D, W, q_k_dim, H)
            K = K.view(B * D * W, self.q_k_dim, H)  # 展平为 (B*D*W, q_k_dim, H)

            V = V.permute(0, 2, 4, 3, 1).contiguous()  # 重新排列维度为 (B, D, W, H, in_dim)
            V = V.view(B * D * W, H, self.in_dim)  # 展平为 (B*D*W, H, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*D*W, H, H)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*D*W, H, in_dim)
            out = out.view(B, D, W, H, self.in_dim)  # 恢复为原始形状 (B, D, W, H, in_dim)
            out = out.permute(0, 4, 1, 3, 2).contiguous()  # 最终输出形状 (B, C, D, H, W)

        else:  # 如果是宽度方向
            Q = Q.permute(0, 2, 3, 4, 1).contiguous()  # 重新排列维度为 (B, D, H, W, q_k_dim)
            Q = Q.view(B * D * H, W, self.q_k_dim)  # 展平为 (B*D*H, W, q_k_dim)

            K = K.permute(0, 2, 3, 1, 4).contiguous()  # 重新排列维度为 (B, D, H, q_k_dim, W)
            K = K.view(B * D * H, self.q_k_dim, W)  # 展平为 (B*D*H, q_k_dim, W)

            V = V.permute(0, 2, 3, 4, 1).contiguous()  # 重新排列维度为 (B, D, H, W, in_dim)
            V = V.view(B * D * H, W, self.in_dim)  # 展平为 (B*D*H, W, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*D*H, W, W)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*D*H, W, in_dim)
            out = out.view(B, D, H, W, self.in_dim)  # 恢复为原始形状 (B, D, H, W, in_dim)
            out = out.permute(0, 4, 1, 2, 3).contiguous()  # 最终输出形状 (B, C, D, H, W)

        # 使用 gamma 在欧式/切空间融合输入与注意力结果
        gamma = torch.sigmoid(self.gamma)
        out_t = gamma * out + (1 - gamma) * x  # 切空间中的加权

        return out_t


class InterSliceTextGuidedAttention(nn.Module):
    """
    Text-Guided Inter-Slice Attention

    功能：
        - 在 D / H / W 轴上进行 slice-wise self-attention
        - 使用文本 embedding 对 Q / K 进行语义调制
        - 文本不参与 slice 排列，只提供全局语义先验

    输入：
        x:        (B, C, D, H, W)
        text_emb: (T, C_text)  例如 [32, 256]

    输出：
        out:      (B, C, D, H, W)
    """

    def __init__(self,
                 in_dim,
                 q_k_dim,
                 patch_ini,
                 axis='D',
                 text_dim=256,
                 use_hyper=False,
                 c=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        self.use_hyper = use_hyper

        # -----------------------------
        # Visual projections
        # -----------------------------
        self.query_conv = nn.Conv3d(in_dim, q_k_dim, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_dim, q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, kernel_size=1)

        # -----------------------------
        # Text → Q/K modulation
        # -----------------------------
        self.text_to_qk = nn.Linear(text_dim, q_k_dim)

        # -----------------------------
        # Positional embedding (slice-wise)
        # -----------------------------
        D, H, W = patch_ini
        if axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))
        elif axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))
        elif axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))
        else:
            raise ValueError("axis must be 'D', 'H', or 'W'")

        nn.init.xavier_uniform_(self.pos_embed)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual gate

    def forward(self, x, text_emb):
        """
        x:        (B, C, D, H, W)
        text_emb: (T, text_dim)
        """
        B, C, D, H, W = x.shape
        residual = x

        # ----------------------------------
        # 1️⃣ Text → global Q/K bias
        # ----------------------------------
        # [T, text_dim] → [T, q_k_dim] → [q_k_dim]
        text_bias = self.text_to_qk(text_emb).mean(dim=0)
        text_bias = text_bias.view(1, self.q_k_dim, 1, 1, 1)

        # ----------------------------------
        # 2️⃣ Visual Q / K / V
        # ----------------------------------
        Q = self.query_conv(x) + self.pos_embed + text_bias
        K = self.key_conv(x)   + self.pos_embed + text_bias
        V = self.value_conv(x)

        scale = math.sqrt(self.q_k_dim)

        # ----------------------------------
        # 3️⃣ Slice-wise attention
        # ----------------------------------
        if self.axis == 'D':
            Q = Q.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, self.q_k_dim)
            K = K.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.q_k_dim, D)
            V = V.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)

            attn = self.softmax(torch.bmm(Q, K) / scale)
            out  = torch.bmm(attn, V).view(B, H, W, D, C)
            out  = out.permute(0, 4, 3, 1, 2)

        elif self.axis == 'H':
            Q = Q.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, self.q_k_dim)
            K = K.permute(0, 2, 4, 1, 3).reshape(B * D * W, self.q_k_dim, H)
            V = V.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)

            attn = self.softmax(torch.bmm(Q, K) / scale)
            out  = torch.bmm(attn, V).view(B, D, W, H, C)
            out  = out.permute(0, 4, 1, 3, 2)

        else:  # W
            Q = Q.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, self.q_k_dim)
            K = K.permute(0, 2, 3, 1, 4).reshape(B * D * H, self.q_k_dim, W)
            V = V.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)

            attn = self.softmax(torch.bmm(Q, K) / scale)
            out  = torch.bmm(attn, V).view(B, D, H, W, C)
            out  = out.permute(0, 4, 1, 2, 3)

        # ----------------------------------
        # 4️⃣ Residual fusion
        # ----------------------------------
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * residual

        return out
    

"""
if __name__ == '__main__':
    # 设置输入参数
    batch_size = 4  # 批次大小
    in_channels = 512  # 输入通道数
    q_k_dim = 16  # Q, K 向量的通道数
    input_resolution = (4, 8, 8)  # 输入张量的分辨率
    axis = 'D'  # 在深度方向进行注意力操作

    # 创建随机输入张量 (batch_size, channels, depth, height, width)
    x = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1], input_resolution[2]).cuda()
    processed = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1],
                            input_resolution[2]).cuda()

    # 创建 InterSliceSelfAttention 模块
    model = InterSliceSelfAttention(in_dim=512, q_k_dim=16, patch_ini=(4, 8, 8), axis='D', use_hyper=True, c=1.0).cuda()

    # 打印模型结构
    print(model)

    # 前向传播
    output = model(x, processed)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
"""


"""
双曲注意力
"""
class FastHyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(FastHyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights on the manifold
        weight = torch.Tensor(out_features, in_features)
        nn.init.xavier_uniform_(weight)
        self.weight = geoopt.ManifoldParameter(weight, manifold=manifold)
        if bias:
            bias = torch.zeros(out_features)
            self.bias = geoopt.ManifoldParameter(bias, manifold=manifold)
        else:
            self.bias = None

    def forward(self, input):
        # Combine Mobius matvec and addition into a single operation
        output = self.manifold.mobius_matvec(self.weight, input)
        if self.bias is not None:
            output = self.manifold.mobius_add(output, self.bias)
        return output


class FastHyperbolicMultiheadCrossAttention(nn.Module):
    """
    Fast Hyperbolic Multi-Head Attention
    Supports both self-attention and cross-attention in Poincaré Ball.
    """

    def __init__(self, embedding_dim, num_heads, manifold):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = embedding_dim // num_heads

        assert embedding_dim % num_heads == 0

        self.query_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.key_proj   = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.value_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.out_proj   = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)

    def forward(self, query, key=None, value=None):
        """
        query: (B, Nq, C) hyperbolic
        key:   (B, Nk, C) hyperbolic (optional)
        value: (B, Nk, C) hyperbolic (optional)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        B, Nq, _ = query.shape
        Nk = key.shape[1]

        # Hyperbolic linear projections
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        # Split heads
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Hyperbolic attention scores
        attn_scores = self._hyperbolic_attention_scores(q, k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted aggregation
        out = torch.matmul(attn_weights, v)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.embedding_dim)
        out = self.out_proj(out)

        return out

    def _hyperbolic_attention_scores(self, q, k):
        """
        q: (B, H, Nq, Dh)
        k: (B, H, Nk, Dh)
        """
        eps = 1e-5

        q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True).clamp_max(1 - eps)
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True).clamp_max(1 - eps)

        qk = torch.matmul(q, k.transpose(-2, -1))

        denom = (1 - q_norm_sq) * (1 - k_norm_sq.transpose(-2, -1)) + eps
        delta = 2 * (qk - q_norm_sq * k_norm_sq.transpose(-2, -1)) / denom

        dist = torch.sqrt(torch.clamp(delta, min=eps))
        return -dist / (self.head_dim ** 0.5)
    

class FastHyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold):
        super(FastHyperbolicMultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = embedding_dim // num_heads

        assert (
                embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        # Use FastHyperbolicLinear
        self.query_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.key_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.value_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.out_proj = FastHyperbolicLinear(embedding_dim, embedding_dim, manifold)

        # Simplify head scaling
        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Project inputs to queries, keys, and values
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = self._approximate_hyperbolic_attention_scores(q, k) / (self.head_dim ** 0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        output = self.out_proj(attn_output)

        return output

    def _approximate_hyperbolic_attention_scores(self, q, k):
        # Approximate the hyperbolic distance using a first-order Taylor expansion
        # This avoids the expensive computation of acosh
        epsilon = 1e-5
        q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True).clamp_max(1 - epsilon)
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True).clamp_max(1 - epsilon)

        # Efficiently compute inner product
        qk_inner = torch.matmul(q, k.transpose(-2, -1))  # Shape: (B, H, S, S)

        # Approximate hyperbolic distance squared
        denom = (1 - q_norm_sq) * (1 - k_norm_sq).transpose(-2, -1) + epsilon
        delta = 2 * ((qk_inner - q_norm_sq * k_norm_sq.transpose(-2, -1)) / denom)

        # Since acosh(1 + x) ≈ sqrt(2x) for small x
        dist_sq = torch.sqrt(torch.clamp(delta, min=epsilon))
        attn_scores = -dist_sq  # Negative distance as scores
        return attn_scores


class FastHyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, manifold):
        super(FastHyperbolicTransformerLayer, self).__init__()
        self.manifold = manifold

        self.self_attn = FastHyperbolicMultiheadAttention(embedding_dim, num_heads, manifold)
        self.linear1 = FastHyperbolicLinear(embedding_dim, embedding_dim * 4, manifold)
        self.linear2 = FastHyperbolicLinear(embedding_dim * 4, embedding_dim, manifold)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output = self.self_attn(x)
        x = self.manifold.mobius_add(x, self.dropout(attn_output))

        # Feedforward network with Möbius ReLU
        x2 = self.linear1(x)
        x2 = self.mobius_relu(x2)
        x2 = self.linear2(x2)
        x = self.manifold.mobius_add(x, self.dropout(x2))

        return x

    def mobius_relu(self, x):
        # Avoid transformations by applying ReLU in the tangent space at 0
        x_euclidean = self.manifold.logmap0(x)
        x_euclidean = F.relu(x_euclidean)
        return self.manifold.expmap0(x_euclidean)


class FastHyperbolicConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            c=1.0,
    ):
        super(FastHyperbolicConv2d, self).__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weight on the manifold
        weight = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        nn.init.xavier_uniform_(weight)
        self.weight = geoopt.ManifoldParameter(weight, manifold=self.manifold)
        if bias:
            bias = torch.zeros(out_channels)
            self.bias = geoopt.ManifoldParameter(bias, manifold=self.manifold)
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        # Transform input and weights to Euclidean space once
        input_euclidean = self.manifold.logmap0(input)
        weight_euclidean = self.manifold.logmap0(self.weight)

        # Perform Euclidean convolution
        output_euclidean = F.conv2d(
            input_euclidean,
            weight_euclidean,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            # Add bias in Euclidean space
            output_euclidean += self.bias.view(1, -1, 1, 1)

        # Map the output back to hyperbolic space once
        output_hyperbolic = self.manifold.expmap0(output_euclidean)
        return output_hyperbolic


class FastHyperbolicLearnedPositionEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim, manifold):
        super(FastHyperbolicLearnedPositionEncoding, self).__init__()
        self.manifold = manifold
        position_embeddings = torch.zeros(1, num_patches, embedding_dim)
        nn.init.xavier_uniform_(position_embeddings)
        self.position_embeddings = geoopt.ManifoldParameter(position_embeddings, manifold=manifold)

    def forward(self, x):
        # Position embeddings addition without scaling
        return self.manifold.mobius_add(x, self.position_embeddings)


class Net(nn.Module):
    def __init__(
            self,
            # img_size=32,
            # patch_size=4,
            # in_channels=3,
            # num_classes=10,
            # embedding_dim=128,
            # num_heads=8,
            # num_layers=4,  # Reduced number of layers
            # dropout=0.1,
            # manifold=None
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            embedding_dim=768,
            num_heads=12,
            num_layers=12,
            dropout=0.1,
            manifold=None
    ):
        super(Net, self).__init__()
        if manifold is None:
            manifold = geoopt.PoincareBall(c=1.0)
        self.manifold = manifold

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = FastHyperbolicConv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            c=1.0,
        )

        self.position_embeddings = FastHyperbolicLearnedPositionEncoding(
            self.num_patches, embedding_dim, manifold
        )

        self.layers = nn.ModuleList(
            [
                FastHyperbolicTransformerLayer(embedding_dim, num_heads, dropout, manifold)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)  # Shape: (batch_size, embedding_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)

        x = self.position_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        # Map back to Euclidean space once for classification
        x = self.manifold.logmap0(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x



# Hyperbolic Layer Normalization
class HyperbolicLayerNorm(nn.Module):
    def __init__(self, embedding_dim, manifold, eps=1e-5):
        super(HyperbolicLayerNorm, self).__init__()
        self.manifold = manifold
        self.eps = eps
        self.normalized_shape = (embedding_dim,)  # Normalize over embedding_dim only
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # Map to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        # Apply LayerNorm over the last dimension (embedding_dim)
        x_norm = F.layer_norm(
            x_tangent,
            self.normalized_shape,
            self.gamma,
            self.beta,
            self.eps
        )
        # Map back to manifold
        return self.manifold.expmap0(x_norm)


# Parametric ReLU in Hyperbolic Space with Shared Alpha
class MobiusPReLU(nn.Module):
    def __init__(self, manifold):
        super(MobiusPReLU, self).__init__()
        self.manifold = manifold
        # Initialize alpha as a single parameter shared across all channels
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Map to tangent space at origin
        x_euclidean = self.manifold.logmap0(x)
        # Apply PReLU with shared alpha
        x_relu = F.prelu(x_euclidean, self.alpha)
        # Map back to manifold
        return self.manifold.expmap0(x_relu)


# Hyperbolic Linear Layer with He Initialization
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        self.weight = geoopt.ManifoldParameter(
            torch.Tensor(out_features, in_features), manifold=manifold
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He Initialization
        if bias:
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features), manifold=manifold
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, input):
        original_shape = input.shape
        in_features = input.shape[-1]

        if in_features != self.in_features:
            raise ValueError(
                f"Incompatible shapes: input shape {input.size()} and weight shape {self.weight.size()}"
            )

        input_flat = input.reshape(-1, in_features)
        output_flat = self.manifold.mobius_matvec(self.weight, input_flat)
        output = output_flat.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            bias_unsqueezed = self.bias.view(
                *([1] * (output.dim() - 1)),
                self.out_features
            )
            output = self.manifold.mobius_add(output, bias_unsqueezed)

        return output


# Hyperbolic Learned Position Encoding
class HyperbolicLearnedPositionEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim, manifold):
        super(HyperbolicLearnedPositionEncoding, self).__init__()
        self.manifold = manifold
        self.position_embeddings = geoopt.ManifoldParameter(
            torch.zeros(1, num_patches, embedding_dim), manifold=manifold
        )
        nn.init.xavier_uniform_(self.position_embeddings)
        self.curvature = nn.Parameter(torch.tensor(1.0))  # Learnable curvature

    def forward(self, x):
        scaled_embeddings = self.position_embeddings * self.curvature
        return self.manifold.mobius_add(x, scaled_embeddings)


# Hyperbolic Patch Embedding without nn.Conv2d
class HyperbolicPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embedding_dim, manifold):
        super(HyperbolicPatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.manifold = manifold

        # Calculate the flattened patch size
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear layer to project patches into embedding space
        self.proj = HyperbolicLinear(self.patch_dim, embedding_dim, manifold)

    def forward(self, x):
        batch_size = x.size(0)
        # Unfold the image to extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (batch_size, in_channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size,
            self.in_channels,
            -1,
            self.patch_size,
            self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4)  # (batch_size, num_patches, in_channels, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)  # (batch_size, num_patches, patch_dim)

        # Project patches to hyperbolic embedding space
        x = self.proj(patches)  # (batch_size, num_patches, embedding_dim)

        return x


class HyperbolicPatchEmbedding3D(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embedding_dim, manifold):
        super(HyperbolicPatchEmbedding3D, self).__init__()
        assert all(i % p == 0 for i, p in zip(img_size, patch_size)), \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.img_size = img_size          # e.g., (D=32, H=128, W=128)
        self.patch_size = patch_size      # e.g., (4, 16, 16)
        self.in_channels = in_channels    # e.g., 1
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])

        self.patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        self.proj = HyperbolicLinear(self.patch_dim, embedding_dim, manifold)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size

        # Step 1: unfold along depth
        x = x.unfold(2, pd, pd)  # → (B, C, D//pd, H, W, pd)
        x = x.unfold(3, ph, ph)  # → (B, C, D//pd, H//ph, W, pd, ph)
        x = x.unfold(4, pw, pw)  # → (B, C, D//pd, H//ph, W//pw, pd, ph, pw)

        # Step 2: reshape to (B, num_patches, patch_dim)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # (B, D', H', W', C, pd, ph, pw)
        x = x.contiguous().view(B, -1, self.patch_dim)  # (B, N, patch_dim)

        # Step 3: project to hyperbolic space
        x = self.proj(x)  # (B, N, embedding_dim)

        return x


class TokenLearner(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(4,8,8), patch_size=(4,8,8), in_chans=1, embed_dim=256): # 8 512
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1] * (img_size[2] // patch_size[2])) #32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),nn.GELU())

    def forward(self, x): #bc,1,dhw,
        x = self.proj(x) # bc,dhw,1,1,1
        x = x.flatten(2) # bc,dhw,1
        x = x.transpose(1, 2) # bc,1,dhw
        return x


# Hyperbolic Multihead Attention with DropConnect
class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold, dropconnect_prob=0.1):
        super(HyperbolicMultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = embedding_dim // num_heads

        self.query_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.key_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.value_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.out_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.curvature = manifold.c if hasattr(manifold, 'c') else 1.0
        self.dropconnect_prob = dropconnect_prob

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        q = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]  # (batch_size, num_heads, seq_length, head_dim)

        # Compute hyperbolic distance-based attention scores
        # Map to tangent space
        q_tangent = self.manifold.logmap0(q)
        k_tangent = self.manifold.logmap0(k)

        # Compute dot product attention in tangent space
        attn_scores = torch.matmul(q_tangent, k_tangent.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores / self.head_scaling

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.training and self.dropconnect_prob > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropconnect_prob, training=True)

        # Compute attention output
        attn_output_tangent = torch.matmul(attn_weights, self.manifold.logmap0(v))
        attn_output = self.manifold.expmap0(attn_output_tangent)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output


# Hyperbolic Transformer Layer with Enhancements
class HyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, manifold):
        super(HyperbolicTransformerLayer, self).__init__()
        self.manifold = manifold

        self.self_attn = HyperbolicMultiheadAttention(embedding_dim, num_heads, manifold)
        self.norm1 = HyperbolicLayerNorm(embedding_dim, manifold)
        self.norm2 = HyperbolicLayerNorm(embedding_dim, manifold)

        self.linear1 = HyperbolicLinear(embedding_dim, embedding_dim * 4, manifold)
        self.activation = MobiusPReLU(manifold)
        self.linear2 = HyperbolicLinear(embedding_dim * 4, embedding_dim, manifold)
        self.dropout = nn.Dropout(dropout)
        self.layer_scaling = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Self-attention with normalization
        attn_output = self.self_attn(x)
        x = self.norm1(self.manifold.mobius_add(x, self.dropout(attn_output).mul(self.layer_scaling)))

        # Feedforward network with Mobius PReLU and normalization
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x = self.norm2(self.manifold.mobius_add(x, self.dropout(x2).mul(self.layer_scaling)))

        return x



"""
if __name__ == '__main__':
    # 准备参数
    batch_size = 2
    seq_len = 16
    embed_dim = 512
    num_heads = 4

    # 创建 manifold（庞加莱球）
    manifold = geoopt.PoincareBall(c=1.0)

    # 创建超曲多头注意力模块
    # attn = HyperbolicMultiheadAttention(
    #     embedding_dim=512,
    #     num_heads=4,
    #     manifold=manifold,
    #     dropconnect_prob=0
    # )
    attn = FastHyperbolicMultiheadAttention(
        embedding_dim=512,
        num_heads=4,
        manifold=manifold,
    )

    # 模拟输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 映射到双曲空间（模块内部也会 logmap0/expmap0）
    x_hyp = manifold.expmap0(x)

    # 前向计算
    output = attn(x_hyp)

    print("输入形状:", x_hyp.shape)
    print("输出形状:", output.shape)
"""

"""
合并后的综合双空间注意力
"""

class StructureAwareBottleneck(nn.Module):
    """
    Local → Global → Dual-space (Euclidean + Hyperbolic) structure modeling.
    This block is MT / Teacher–Student friendly:
    - no pred_type
    - no batch split
    - pure feature transformation
    """

    def __init__(self,
                 num_heads,
                 embedding_channels,
                 attention_dropout_rate,
                 parallel_block_cfg: dict):
        super().__init__()

        self.local_attn = Cross_Attention_Local_Block(
            num_heads=num_heads,
            embedding_channels=embedding_channels,
            attention_dropout_rate=attention_dropout_rate,
        )

        self.global_attn = Cross_Attention_Global_Block(
            num_heads=num_heads,
            embedding_channels=embedding_channels,
            attention_dropout_rate=attention_dropout_rate,
        )

        self.parallel_block = ParallelISSAHMA(**parallel_block_cfg)

    def forward(self, x):
        x = self.local_attn(x)
        x = self.global_attn(x)
        x = self.parallel_block(x)
        return x
    

class ParallelISSAHMA(nn.Module):
    """
    Parallel fusion of axis-wise InterSliceSelfAttention (Euclidean) and HyperMultiAttention (Hyperbolic).
    - ISSA branch runs in Euclidean/tangent space (use_hyper=False) to avoid extra exp/log.
    - HMA branch performs the only exp/log mapping internally (single mapping round overall).
    - Outputs are fused in Euclidean space via gated residuals + optional 1x1x1 projection.
    """

    def __init__(self,
                 in_dim: int,
                 q_k_dim: int,
                 img_size,  # (D, H, W) at bottleneck, e.g., (4, 8, 8)
                 axes=("D", "H", "W"),
                 c: float = 0.5,
                 hma_heads: int = 8,
                 hma_layers: int = 1,
                 hma_dropout: float = 0.0,
                 use_proj: bool = True,
                 num_heads=2,
                 embedding_channels=16,
                 attention_dropout_rate=0.1
                 ):
        super().__init__()
        self.axes = axes
        self.issas = nn.ModuleList([
            InterSliceSelfAttention(in_dim=in_dim,
                                    q_k_dim=q_k_dim,
                                    patch_ini=img_size,
                                    axis=ax,
                                    use_hyper=False,  # Euclidean
                                    c=c)
            for ax in self.axes
        ])
        self.issa_gate = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))  # sigmoid -> ~0.57 start
        self.alpha = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))  # sigmoid -> ~0.57 start

        self.hma = HyperMultiAttention(embedding_dim=in_dim,
                                       num_heads=hma_heads,
                                       img_size=img_size,
                                       patch_size=(1, 1, 1),
                                       in_chans=in_dim,
                                       c=c,
                                       n_layers=hma_layers,
                                       dropout_p=hma_dropout)
        self.hma_gate = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Conv3d(in_dim, in_dim, kernel_size=1, bias=True)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_dim, max(8, in_dim // 16), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(max(8, in_dim // 16), in_dim, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.reduc_map_euro = nn.Sequential(nn.Conv3d(embedding_channels // 2, embedding_channels // 4, kernel_size=1, padding=0), nn.GELU())  


    def forward(self, x):
        y_issa = x
        for m in self.issas:
            y_issa = m(y_issa, y_issa)
        g_issa = torch.sigmoid(self.issa_gate)
        y_issa = x + g_issa * (y_issa - x)  # gated residual
        y_hma = self.hma(x)
        y = self.alpha * y_hma + (1 - self.alpha) * y_issa
        if self.use_proj:
            y = self.proj(y)
            y = self.se(y) * y

        return y


class FeatureFusionGate(nn.Module):
    """
    Gated Feature Fusion Block

    功能：
        - 对两个同 shape 的 3D 特征进行 gated 融合
        - 支持：
            * 残差式对齐
            * α 加权融合
            * 可选的 1x1x1 投影 + SE 通道重标定

    输入：
        x1, x2: (B, C, D, H, W)

    输出：
        y: (B, C, D, H, W)
    """

    def __init__(self,
                 channels: int,
                 use_proj: bool = True,
                 init_gate: float = 0.3,
                 init_alpha: float = 0.3):
        super().__init__()

        # gated residual between x1 and x2
        self.gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, max(8, channels // 16), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(max(8, channels // 16), channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x1, x2):
        """
        x1, x2: (B, C, D, H, W)
        """
        # 1️⃣ gated residual alignment
        g = torch.sigmoid(self.gate)
        x2_aligned = x1 + g * (x2 - x1)

        # 2️⃣ weighted fusion
        a = torch.sigmoid(self.alpha)
        y = a * x2_aligned + (1 - a) * x1

        # 3️⃣ optional projection & channel reweight
        if self.use_proj:
            y = self.proj(y)
            y = self.se(y) * y

        return y
    

class HyperMultiAttention1(nn.Module):
    """
    Hyperbolic semantic relation modeling for CLIP tokens
    ------------------------------------------------
    Input:
        x: (N, C), e.g. (32, 512)

    Function:
        - Input is a batch-free CLIP token embedding (no batch dimension)
        - Temporarily add batch dimension for FastHyperbolicMultiheadAttention
        - Map tokens to hyperbolic space
        - Apply n_layers hyperbolic multi-head attention
        - Map back to Euclidean space
        - Gated residual fusion

    Output:
        x_out: (N, C)
    """

    def __init__(self,
                 embedding_dim=512,
                 num_heads=8,
                 n_layers=1,
                 dropout_p=0.0,
                 c=0.5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = geoopt.PoincareBall(c=c)

        self.layers = nn.ModuleList([
            FastHyperbolicMultiheadAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                manifold=self.manifold,
            )
            for _ in range(n_layers)
        ])

        self.norm = HyperbolicLayerNorm(embedding_dim, self.manifold)
        self.dropout = nn.Dropout(dropout_p)

        # gated residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        """
        x: (N, C), e.g. (32, 512)
        CLIP token embeddings (no batch dimension)
        """
        assert x.dim() == 2, f"Expected (N, C), got {x.shape}"
        N, C = x.shape
        assert C == self.embedding_dim, \
            f"Channel mismatch: expect {self.embedding_dim}, got {C}"

        x_residual = x

        # ---- add dummy batch dimension ----
        x = x.unsqueeze(0)  # (1, N, C)

        # Euclidean → Hyperbolic
        x = self.manifold.expmap0(x)

        # Hyperbolic MHSA
        for layer in self.layers:
            x = layer(x)

        # Norm + Hyperbolic → Euclidean
        x = self.norm(x)
        x = self.manifold.logmap0(x)
        x = self.dropout(x)

        # remove dummy batch dimension
        x = x.squeeze(0)  # (N, C)

        # Gated residual fusion (Euclidean)
        x = x_residual + self.alpha * x

        return x


class HyperbolicCrossAttention(nn.Module):
    """
    Hyperbolic Cross-Attention:
        Q from visual tokens
        K/V from text tokens
    """

    def __init__(self, dim, num_heads, manifold):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0

        self.q_proj = FastHyperbolicLinear(dim, dim, manifold)
        self.k_proj = FastHyperbolicLinear(dim, dim, manifold)
        self.v_proj = FastHyperbolicLinear(dim, dim, manifold)
        self.out_proj = FastHyperbolicLinear(dim, dim, manifold)

    def forward(self, query, key, value):
        """
        query: (B, Nq, C)
        key:   (B, Nk, C)
        value: (B, Nk, C)
        """
        B, Nq, C = query.shape
        Nk = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Hyperbolic distance-based attention
        attn_scores = self._hyperbolic_attention(q, k)
        attn = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)

        return self.out_proj(out)

    def _hyperbolic_attention(self, q, k):
        eps = 1e-5
        q_norm = torch.sum(q * q, dim=-1, keepdim=True).clamp_max(1 - eps)
        k_norm = torch.sum(k * k, dim=-1, keepdim=True).clamp_max(1 - eps)

        qk = torch.matmul(q, k.transpose(-2, -1))
        denom = (1 - q_norm) * (1 - k_norm.transpose(-2, -1)) + eps
        delta = 2 * (qk - q_norm * k_norm.transpose(-2, -1)) / denom

        dist = torch.sqrt(torch.clamp(delta, min=eps))
        return -dist


class HyperTextGuidedVisualBlock(nn.Module):
    """
    Hyperbolic Text-Guided Visual Block (Final Version)

    Visual:
        (B, C, D, H, W)
          → patch tokens
          → hyperbolic cross-attention (text-guided)
          → token grid
          → 1×1×1 projection
          → gated residual

    Text:
        (32, 256)  (16 self + 16 adjacency)
        → Euclidean → Hyperbolic (once)
    """

    def __init__(self,
                 img_size,        # (D, H, W)
                 patch_size,      # (pd, ph, pw)
                 in_chans,        # C
                 embed_dim=256,
                 num_heads=4,
                 c=0.5,
                 dropout=0.0):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # ---------- Hyperbolic manifold ----------
        self.manifold = geoopt.PoincareBall(c=c)

        # ---------- Patch embedding ----------
        self.patch_embed = HyperbolicPatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embedding_dim=embed_dim,
            manifold=self.manifold
        )

        # token grid size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        # ---------- Hyperbolic cross-attention ----------
        self.cross_attn = HyperbolicCrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            manifold=self.manifold
        )

        self.norm = HyperbolicLayerNorm(embed_dim, self.manifold)
        self.dropout = nn.Dropout(dropout)

        # ---------- Token → voxel projection ----------
        self.token_to_feat = nn.Conv3d(embed_dim, in_chans, kernel_size=1, bias=True)

        # ---------- Gated residual ----------
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, visual_feat, text_emb):
        """
        visual_feat: (B, C, D, H, W)
        text_emb:    (32, 256)  Euclidean CLIP embedding
        """
        B, C, D, H, W = visual_feat.shape
        x_res = visual_feat

        # ===== Text: Euclidean → Hyperbolic (once) =====
        text_tokens_h = self.manifold.expmap0(text_emb)  # (32, 256)
        text_tokens_h = text_tokens_h.unsqueeze(0).expand(B, -1, -1)  # (B, 32, 256)

        # ===== Visual: Euclidean → Hyperbolic =====
        x_h = self.manifold.expmap0(visual_feat)

        # ===== Patch → tokens =====
        vis_tokens = self.patch_embed(x_h)  # (B, N, 256)

        # ===== Hyperbolic cross-attention (Q: visual, K/V: text) =====
        vis_tokens = self.cross_attn(
            query=vis_tokens,
            key=text_tokens_h,
            value=text_tokens_h
        )  # (B, N, 256)

        # ===== Norm + Hyperbolic → Euclidean =====
        vis_tokens = self.norm(vis_tokens)
        vis_tokens = self.manifold.logmap0(vis_tokens)
        vis_tokens = self.dropout(vis_tokens)

        # ===== Tokens → grid =====
        gd, gh, gw = self.grid_size
        vis_tokens = vis_tokens.transpose(1, 2).contiguous()  # (B, 256, N)
        vis_grid = vis_tokens.view(B, self.embed_dim, gd, gh, gw)

        # ===== Project back to feature channels =====
        vis_grid = self.token_to_feat(vis_grid)  # (B, C, gd, gh, gw)

        # ===== Upsample to original resolution if needed =====
        if (gd, gh, gw) != (D, H, W):
            vis_grid = torch.nn.functional.interpolate(
                vis_grid,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )

        # ===== Gated residual =====
        alpha = torch.sigmoid(self.alpha)
        out = x_res + alpha * (vis_grid - x_res)

        return out
    

class HyperMultiAttention(nn.Module):
    """
    hmat
    双曲多头注意力模块，用于对 3D 特征进行 token 化、位置编码以及双曲注意力操作。

    参数说明:
        embedding_dim: 输出 token 的特征维度（C）
        num_heads: 多头注意力的头数
        img_size: 输入特征图的大小 (D, H, W)，例如 (4, 8, 8)
        patch_size: patch 的切分大小 (pd, ph, pw)，例如 (2, 2, 2)
        in_chans: 输入特征的通道数（例如 ResNet 中间层为 n_filters * k）
        c: 双曲空间的曲率

    注意：要求 img_size 必须能被 patch_size 整除
    """

    def __init__(self,
                 embedding_dim,
                 num_heads,
                 img_size,
                 patch_size,
                 in_chans,
                 c=0.5,
                 n_layers=1,
                 dropout_p=0.0,
                 learnable=True):
        super(HyperMultiAttention, self).__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        # self.log_c = nn.Parameter(torch.log(torch.tensor(c))) if learnable else torch.tensor(c)
        # 计算 token 数量 = patch 数
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])

        # 定义双曲空间流形（Poincaré ball）
        self.manifold = geoopt.PoincareBall(c=c)

        # 将 3D 输入划分为 patch
        self.tokenlenrner = HyperbolicPatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embedding_dim=embedding_dim,
            manifold=self.manifold
        )

        # 双曲空间多头注意力机制
        self.hyperMultiAtten = FastHyperbolicMultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            manifold=self.manifold,
        )

        self.layers = nn.ModuleList([
            FastHyperbolicMultiheadAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                manifold=self.manifold,
            )
            for _ in range(n_layers)
        ])

        self.norm = HyperbolicLayerNorm(embedding_dim, self.manifold)
        self.dropout = nn.Dropout(dropout_p)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        """
        【输入】
        x: Tensor，形状为 (B, C, D, H, W)
            - B: 批大小
            - C: 通道数，建议与 embedding_dim 一致
            - D/H/W: 输入特征图的空间维度

        【前向步骤】
        1. 映射输入到双曲空间（Poincaré Ball）→ (B, C, D, H, W)
        2. 对输入进行 patch 切分和展平 → (B, N, embedding_dim)，其中 N 为 patch 数量
        3. 为每个 patch 添加 learnable 的双曲位置编码
        4. 输入多个 FastHyperbolicMultiheadAttention 层，进行 token 间的双曲注意力建模
        5. 输出特征做 LayerNorm 并投影回欧几里得空间
        6. reshape 回原来的 shape → (B, C, D, H, W)

        【输出】
        - x: Tensor，形状为 (B, C, D, H, W)，与输入尺寸一致（如果 embedding_dim == in_chans）

        【备注】
        - 输出的空间尺寸 (D, H, W) 实际为 patch 数目，必须满足：
            D = img_size[0] // patch_size[0]
            H = img_size[1] // patch_size[1]
            W = img_size[2] // patch_size[2]
        - 如果 embedding_dim ≠ C，则需要额外线性变换才能融合原网络结构。
        """
        B, C, D, H, W = x.shape  # 例如 (4, 512, 4, 8, 8)
        x_residual = x  # 试用残差为方式 1，不适用残差为方式 2

        # Step 0:  European 2 Hyperbolic
        x = self.manifold.expmap0(x)

        # Step 1: token embedding → (B, N, embedding_dim)
        x = self.tokenlenrner(x)

        # Step 3: hyperbolic multi-head attention → (B, N, embedding_dim)
        for layer in self.layers:
            x = layer(x)

        # Step 4: LayerNorm + Hyperbolic 2 European
        x = self.norm(x)
        x = self.manifold.logmap0(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).view(B, C, D, H, W)
        # Residual: keep the original Euclidean feature and add a gated hyperbolic-attention update
        x = x_residual + self.alpha * x  # 方式 1
        return x


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


class TokenLearner_Global(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x = self.proj(x)                # (B, embed_dim, D, H, W)
        x = x.mean(dim=[2,3,4])         # (B, embed_dim)
        x = x.unsqueeze(1)              # (B, 1, embed_dim)
        return x


class TokenLearner_Local(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(6, 6, 6), patch_size=(2, 2, 2), in_chans=1, embed_dim=8):  
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (
                    img_size[1] // patch_size[1] * (img_size[2] // patch_size[2]))  # 32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
                                    nn.GELU())

    def forward(self, x):  # bc,1,d,h,w
        x = self.proj(x)  # bc,p1p2p3,d/p1,h/p2,w/p3
        x = x.flatten(2)  # bc,p1p2p3,dhw/p1p2p3
        x = x.transpose(1, 2)  # bc,dhw/p1p2p3,p1p2p3
        return x


class Cross_Attention_Global(nn.Module):  #
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads

        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.psi = nn.InstanceNorm2d(1)

        self.softmax = nn.Softmax(dim=3)

        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):  # (b,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (
            self.num_heads, self.attention_head_size)  # (b,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b,self.num_heads,hw,embedding_channels)

    def forward(self, emb):
        _, N, C = emb.size()

        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)

        mh_q = self.multi_head_rep(q).transpose(-1, -2)
        mh_k = self.multi_head_rep(k)
        mh_v = self.multi_head_rep(v).transpose(-1, -2)

        self_attn = torch.matmul(mh_q, mh_k)
        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v)

        self_attn = self_attn.permute(0, 3, 2, 1).contiguous()
        new_shape = self_attn.size()[:-2] + (self.KV_size,)
        self_attn = self_attn.view(*new_shape)

        out = self.out(self_attn)
        out = self.proj_dropout(out)
        return out


class Cross_Attention_Global_Block(nn.Module):  # 4,512,4,8,8
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, img_size=(6, 6, 6)):
        super().__init__()
        self.token_learner = TokenLearner_Global(in_chans=1, embed_dim=256)
        self.attn_norm = nn.LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Cross_Attention_Global(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = nn.LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # ---- 1. global token ----
        x_token = x.view(B * C, D, H, W).unsqueeze(1)   # (B*C, 1, D, H, W)
        x_token = self.token_learner(x_token)           # (B*C, 1, C)
        x_token = x_token.view(B, C, C).mean(dim=1)     # (B, C)
        x_token = x_token.unsqueeze(1)                  # (B, 1, C)

        # ---- 2. global attention ----
        res = x_token
        x_token = self.attn_norm(x_token)
        x_token = self.attn(x_token)
        x_token = x_token + res
        x_token = self.ffn_norm(x_token)                # (B, 1, C)

        # ---- 3. broadcast back to space ----
        x_out = x_token.squeeze(1).view(B, C, 1, 1, 1)
        x_out = x_out.expand(-1, -1, D, H, W)

        x_out = self.map_out(x_out)
        return x_out


class Cross_Attention_Local(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, patch_size=(2, 2, 2)):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.patch_size = patch_size
        self.embedding_channels = embedding_channels
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(1)

        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):  # (b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (self.num_heads,
                                       self.attention_head_size)  # (b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b/2,self.num_heads,hw,embedding_channels)

    def forward(self, emb):  # (b d/p1 h/p2 w/p3) (p1 p2 p3) c
        _, N, C = emb.size()  # (b d/p1 h/p2 w/p3) (p1 p2 p3) c

        # 在不同通道的相同位置做注意力
        q = self.q(emb)  # (b d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)
        k = self.k(emb)
        v = self.v(emb)

        # (2,1,embedding_channels * self.num_heads,hw)
        mh_q = rearrange(q, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',
                         p1=2, p2=2, p3=2, d=3, h=3, w=3, c=self.embedding_channels, heads=self.num_heads)
        mh_k = rearrange(k, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads (p1 p2 p3) c',
                         p1=2, p2=2, p3=2, d=3, h=3, w=3, c=self.embedding_channels, heads=self.num_heads)
        mh_v = rearrange(v, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',
                         p1=2, p2=2, p3=2, d=3, h=3, w=3, c=self.embedding_channels, heads=self.num_heads)

        self_attn = torch.matmul(mh_q, mh_k)
        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v)

        self_attn = rearrange(self_attn.squeeze(1),
                              '(b d h w) heads c (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
                              p1=2, p2=2, p3=2, d=3, h=3, w=3, c=self.embedding_channels,
                              heads=self.num_heads)
        out = self.out(self_attn)  # (b,hw,embedding_channels)
        return out


class Cross_Attention_Local_Block(nn.Module):
    def __init__(self, num_heads, embedding_channels,
                 attention_dropout_rate, ):
        super().__init__()
        self.token_learner = TokenLearner_Local(img_size=(4, 8, 8), patch_size=(2, 2, 2), in_chans=1, embed_dim=8)
        self.attn_norm = nn.LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Cross_Attention_Local(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = nn.LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.contiguous().view(b * c, d, h, w).unsqueeze(1)  # bc,1,d,h,w
        x = self.token_learner(x)  # bc,dhw/p1p2p3,p1p2p3  不同通道的相同位置进行注意力   b(dhw/p1p2p3),p1p2p3,c
        res = rearrange(x, '(b c)  (d h w)(p1 p2 p3) -> b (d p1 h p2 w p3) c', b=b, c=c, d=3, h=3, w=3, p1=2, p2=2,
                        p3=2)
        x = rearrange(x, '(b c) (d h w) (p1 p2 p3) -> (b d h w) (p1 p2 p3) c', b=b, c=c, d=3, h=3, w=3, p1=2, p2=2,
                      p3=2)
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + res  # residual
        x = self.ffn_norm(x)
        B, n_patch, hidden = x.size()  # 4, 256, 512
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        return x



"""

def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, num_heads=2):


self.struct_bottleneck_x5 = StructureAwareBottleneck(
            num_heads=num_heads,
            embedding_channels=n_filters * 16,   # = 256
            attention_dropout_rate=0.1,
            parallel_block_cfg=dict(
                in_dim=n_filters * 16,           # 256
                q_k_dim=16,
                img_size=(6, 6, 6),              # ★ 只锁定 x5
                axes=("D", "H", "W"),
                c=0.01,
                hma_heads=8,
                hma_layers=4,
                hma_dropout=0.5,
                use_proj=True,
                num_heads=num_heads,
                embedding_channels=n_filters * 16 * 2,
                attention_dropout_rate=0.1
            )
        )

x5 = self.struct_bottleneck(x5)

"""

if __name__ == '__main__':
    # ====== StructureAwareBottleneck sanity check ======
    batch_size = 4
    in_channels = 256
    D, H, W = 6, 6, 6

    x = torch.randn(batch_size, in_channels, D, H, W)

    model = StructureAwareBottleneck(
        num_heads=2,
        embedding_channels=in_channels,
        attention_dropout_rate=0.1,
        parallel_block_cfg=dict(
            in_dim=in_channels,
            q_k_dim=16,
            img_size=(D, H, W),        # ★ 必须与输入一致
            axes=("D", "H", "W"),
            c=0.5,
            hma_heads=8,
            hma_layers=1,
            hma_dropout=0.0,
            use_proj=True,
            num_heads=2,
            embedding_channels=in_channels * 2,
            attention_dropout_rate=0.1
        )
    )

    model.eval()  # 测试阶段
    with torch.no_grad():
        out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)