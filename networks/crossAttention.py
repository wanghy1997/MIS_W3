import torch
from torch import nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import math


class TokenLearner_Global(nn.Module):
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
        # print('TokenLearner_Global x shape:', x.shape)
        return x


class TokenLearner_Local(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(4,8,8), patch_size=(2,2,2), in_chans=1, embed_dim=8): # 8 512
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1] * (img_size[2] // patch_size[2])) #32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),nn.GELU())

    def forward(self, x): #bc,1,d,h,w
        x = self.proj(x) # bc,p1p2p3,d/p1,h/p2,w/p3
        x = x.flatten(2) # bc,p1p2p3,dhw/p1p2p3
        x = x.transpose(1, 2) # bc,dhw/p1p2p3,p1p2p3
        return x


class Cross_Attention_Global(nn.Module): #
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads


        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.psi = nn.InstanceNorm2d(1)

        self.softmax = Softmax(dim=3)

        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):  # (b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (
        self.num_heads, self.attention_head_size)  # (b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b/2,self.num_heads,hw,embedding_channels)



    def forward(self, emb):
        # print('Cross_Attention_Global emb shape:', emb.shape)
        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)

        _, N, C = emb_u.size() #(b/2,hwd,c)



        q_u2l = self.q(emb_u.detach())  # (2,hw,embedding_channels * self.num_heads)
        k_u2l = self.k(emb_l)  # (2,hw,embedding_channels * self.num_heads)
        v_u2l = self.v(emb_l)  # (2,hw,embedding_channels * self.num_heads)

        batch_size = q_u2l.size(0)

        k_u2l = rearrange(k_u2l, 'b n c -> n (b c)')  # (hw,2 *embedding_channels * self.num_heads)
        v_u2l = rearrange(v_u2l, 'b n c -> n (b c)')  # (hw,2 *embedding_channels * self.num_heads)

        k_u2l = repeat(k_u2l, 'n bc -> r n bc', r=batch_size)  # (2,hw,2 *embedding_channels * self.num_heads)
        v_u2l = repeat(v_u2l, 'n bc -> r n bc', r=batch_size)  # (2,hw,2 *embedding_channels * self.num_heads)



        q_u2l = q_u2l.unsqueeze(1).transpose(-1, -2)  # (2,1,embedding_channels * self.num_heads,hw)
        k_u2l = k_u2l.unsqueeze(1)  # (2 , 1 , hw, 2 * embedding_channels * self.num_heads)
        v_u2l = v_u2l.unsqueeze(1).transpose(-1, -2)  # (2,1,embedding_channels * self.num_heads,hw)


        cross_attn_u2l = torch.matmul(q_u2l, k_u2l)  # (2,1,embedding_channels * self.num_heads,embedding_channels * self.num_heads)
        cross_attn_u2l = self.attn_dropout(self.softmax(self.psi(cross_attn_u2l)))

        cross_attn_u2l = torch.matmul(cross_attn_u2l, v_u2l)  # (2,self.num_heads,embedding_channels,hw)

        cross_attn_u2l = cross_attn_u2l.permute(0, 3, 2, 1).contiguous()  # (2,hw,embedding_channels,self.num_heads)
        new_shape_u2l = cross_attn_u2l.size()[:-2] + (self.KV_size,)  # (2,hw,embedding*num_heads)
        cross_attn_u2l = cross_attn_u2l.view(*new_shape_u2l)  # (2,hw,embedding*num_heads)

        out_u2l = self.out(cross_attn_u2l)
        out_u2l = self.proj_dropout(out_u2l)

        # ==========================================================

        q_l2u = self.q(emb_l)
        k_l2u = self.k(emb_u.detach())
        v_l2u = self.v(emb_u.detach())


        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)


        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u,v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        out = torch.cat([out_l2u, out_u2l], dim=0)
        return out
        

class Cross_Attention_Global_Block(nn.Module): #4,512,4,8,8
    def __init__(self, num_heads, img_size, patch_size, embedding_channels, attention_dropout_rate, embed_dim):
        super().__init__()
        self.token_learner = TokenLearner_Global(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Cross_Attention_Global(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)

        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels,embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self.training: #推理模式
            x = torch.cat((x, x))


        b, c, d, h, w = x.shape
        x = x.contiguous().view(b*c, d, h, w).unsqueeze(1)  #bc,1,d,h,w
        x = self.token_learner(x) #bc,1,dhw

        x = rearrange(x, '(b c) 1 (d h w) -> b (d h w) c',b = b, c = c, d = d, h = h, w = w)

        res = x

        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + res #residual



        x = self.ffn_norm(x)


        B, n_patch, hidden = x.size() # 4, 256, 512


        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training: #推理模式
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x
    

class ParallelISSAHMA(nn.Module):
    """
    先全局注意力，再轴向注意力。
    """
    def __init__(self,
                 in_dim: int = 256,
                 q_k_dim: int = 16,
                 axes=("D", "H", "W"),
                 img_size=(6,6,6),
                 patch_size=(6,6,6),
                 embed_dim: int = 216,
                 num_heads=8,
                 embedding_channels=256,
                 attention_dropout_rate=0.1
                 ):
        super().__init__()
        self.axes = axes
        self.issas = nn.ModuleList([
            InterSliceSelfAttention(in_dim=in_dim,
                                    q_k_dim=q_k_dim,
                                    patch_ini=img_size,
                                    axis=ax)
            for ax in self.axes
        ])
        self.globle_attn = Cross_Attention_Global_Block(num_heads=num_heads,
                                                        img_size=img_size,
                                                        patch_size=patch_size,
                                                        embedding_channels=embedding_channels,
                                                        attention_dropout_rate=attention_dropout_rate,
                                                        embed_dim=embed_dim)
        self.issa_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 全局注意力
        x = self.globle_attn(x)
        y_issa = x
        for m in self.issas:
            y_issa = m(y_issa, y_issa)
        y_issa = x + torch.sigmoid(self.issa_gate) * (y_issa - x)  # gated residual
        return y_issa


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

        self.softmax = Softmax(dim=-1)  # 定义 softmax 层
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
    

if __name__ == "__main__":
    model2 = ParallelISSAHMA(in_dim=256, q_k_dim=16, axes=("D", "D", "D"), img_size=(6,6,6), patch_size=(6,6,6), embed_dim=216, num_heads=8, embedding_channels=256, attention_dropout_rate=0.1)
    input = torch.randn(4,256,6,6,6) # 6,6,6  216   4,8,8 256
    output = model2(input)
    print(output.shape)
