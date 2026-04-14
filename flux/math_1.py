'''
 本文件核心:
 RoPE(旋转位置编码)的实现
 '''
import torch
from einops import rearrange
from torch import Tensor

#实现:带有旋转位置编码（RoPE）的多头注意力机制（Multi-Head Attention）
#输入是孤立的词向量（不知道上下文），而输出是融合了上下文信息的词向量
def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe) # v(值) 是纯粹的物质/信息载荷, 所以v不需要被旋转
        #输入: q:查询张量，k:键张量，v:值张量，pe:位置编码张量
        #输出:注意力机制的输出张量
        #步骤:
        #1.将输入的查询 (q) 和键 (k) 张量与生成的旋转位置编码进行结合, 实现:输入绝对位置 -> 算力呈现相对位置
        #2.使用 PyTorch 的 scaled_dot_product_attention 函数计算注意力权重和加权值, 实现:输入绝对位置 -> 算力呈现相对位置
        #3.将多头注意力的输出重新排列回原来的形状, 实现:输入绝对位置 -> 算力呈现相对位置
    
        #scaled_dot_product_attention函数计算查询、键和值之间的注意力权重，并返回加权后的值。它会自动处理缩放和掩码等细节。
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)#计算注意力并提取信息, Q 和 K 越相似，得分越高
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

# RoPE 旋转矩阵生成
#只依赖于位置索引 (pos) 和维度配置 (dim), 实现:输入绝对位置 -> 算力呈现相对位置
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
     #输入:pos:位置序列张量，通常代表 token 在序列中的绝对位置索引
     # dim:注意力头(Attention Head)的特征维度, 必须是偶数
     #theta:频率计算的基数(Base)
    assert dim % 2 == 0

     #定义转速:
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

     #构造二维空间旋转算符:
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

# RoPE 旋转位置编码应用
#将输入的查询 (q) 和键 (k) 张量与生成的旋转位置编码进行结合, 实现:输入绝对位置 -> 算力呈现相对位置
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
     #把输入的q,k的最后一个维度（128）拆成64个二维平面，最后一个维度变成2，倒数第二个维度变成64  128=64*1*2(-1->64)
     #-1是为了适配后续的频率矩阵乘法，64个二维平面分别乘以不同的频率，达到旋转位置编码的效果 
     # 1是为了后续的广播机制，方便频率矩阵和输入的q,k进行乘法运算 
     # 2是因为旋转矩阵是二维的，将128维的q,k拆成64个二维平面，每个平面对应一个旋转位置编码的频率，进行旋转矩阵乘法后再拼回原来的128维
     #xq,xk即输入的q,k  *xq.shape[:-1]  *xk.shape[:-1] 提取q,k的维度
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
     #旋转矩阵乘法，频率矩阵*输入的q,k   旋转角度=Token位置*频率
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
     #把刚才拆解的 64 个二维平面，重新展平拼回到原来的 128 维
