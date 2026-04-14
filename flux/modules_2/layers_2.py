"""
本文件核心：
FLUX 模型的去噪与生成的“核心中央大脑”，负责在图像特征 (img)、文本特征 (txt) 和时间条件 (vec) 之间进行复杂的交互计算。
整个 FLUX 是一个基于 Transformer 的扩散模型 (DiT)。这个文件里定义了所有的“神经元网络层”
双流架构(Double Stream Block) :前期的核心层，让图像特征和文本特征分别进行计算，但在注意力计算时交织在一起，确保图文精准对齐
单流架构(Single Stream Block) :后期的核心层，将图文特征粗暴地拼在一起处理，以极高的效率完成最终的细节雕琢
位置与时间嵌入：处理图像的空间位置信息和扩散过程的时间信息，让模型知道“现在在画图的哪个位置”和“扩散过程进行到了哪一步”。
AdaLN 调制机制：根据时间步条件动态调整每一层的特征变换参数，让模型在不同的扩散阶段表现出不同的行为
"""

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope

" ========== 位置与时间嵌入模块 ========== "

# 多维位置编码：结合了 math_1.py 中的 RoPE，处理多维度的位置信息
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        # ids 包含了各个维度的坐标轴索引
        n_axes = ids.shape[-1]
        # 在多个轴（例如高度、宽度维度）上分别计算旋转位置编码，然后拼接起来
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


# 时间步嵌入：把标量的时间 t (代表扩散过程到了哪一步) 转换成模型能理解的高维向量
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    # 生成不同频率的正弦/余弦波参数
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    # 正弦波和余弦波拼接，将一维的时间变成了 dim 维度的特征表示
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: # 奇数维度补零对齐
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


" ========== 基础网络组件 ========== "

# 多层感知机 (MLP)：基础的非线性变换层，负责特征的升维和降维
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU() # 使用 SiLU 激活函数
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


# 均方根归一化 (RMSNorm)：比传统的 LayerNorm 更高效的归一化方式，用来稳定网络训练
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        # 计算均方根的倒数 (rsqrt)，加上 1e-6 防止除零导致的 NaN 错误
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


# 查询/键归一化：专门用于平滑 Q 和 K 矩阵，防止注意力分数过大导致梯度消失
class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


# 自注意力层：调用 math_1.py 里的核心 attention 算法
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        # 将合并的 QKV 线性层输出拆分为独立的 Query, Key, Value，并按多头格式重排
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        # 执行带有旋转位置编码 (pe) 的注意力计算
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


" ========== AdaLN 调制机制 ========== "

# 记录调制输出的三个关键参数
@dataclass
class ModulationOut:
    shift: Tensor  # 平移量
    scale: Tensor  # 缩放量
    gate: Tensor   # 门控值 (决定输出保留多少比例)


# 调制层 (Modulation)：根据时间步条件 (vec) 动态计算特征变换的参数
# 这是 DiT 架构的核心：把“当前是第几步去噪”以及“全图的条件”强行注入到网络的每一层里
class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3 # 双流架构需要控制两条通路，产出两套参数，所以是6
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        # vec 通常是 时间步嵌入特征 加上 汇聚(pooled)的文本特征
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


" ========== FLUX 的灵魂骨架：双流与单流 Transformer 块 ========== "

# 双流块 (MM-DiT)：FLUX 前期的核心层
# 为什么叫双流？因为图像(img)和文本(txt)的模态不同，前期需要用独立的全连接层处理，
# 但在注意力层(Attention)时，把它们拼在一起计算，实现图文跨模态对齐。
class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # --- 图像专属通道 (img) 硬件配置 ---
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # --- 文本专属通道 (txt) 硬件配置 ---
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # 1. 调制信号生成：根据全局条件 vec，生成给图像和文本的调制参数
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # 2. 图像通道 Attention 前置处理：注入 AdaLN 调制参数，计算 QKV
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # 3. 文本通道 Attention 前置处理：同理计算 QKV
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # 4. 跨模态融合计算 (核心逻辑)：把 txt 和 img 的 QKV 拼接在一起！
        # 这样文本就能看到图像，图像也能看到文本，实现注意力交织
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        # 算完之后，根据原本的序列长度把它们切开，重新分发给 txt 和 img 通道
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # 5. 图像残差与独立 MLP 计算：分别经过门控 (gate) 和前馈网络
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # 6. 文本残差与独立 MLP 计算：同理
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt


# 单流块：FLUX 后期的核心层
# 经过前面多层的双流处理后，图文特征已经高度融合。为了节省算力，
# 后期不再区分 img 和 txt，而是把它们提前拼在一起成为 x，共用一个 Transformer 通道。
class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 一次性计算 qkv 和 mlp 的输入 (并行化提升计算速度)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # 一次性处理 proj 和 mlp 的输出
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        # 单流块只需要一套调制参数 (double=False)
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        # x 此时包含了图像和文本融合后的特征
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        
        # 拆分出用于注意力的 qkv 和用于前馈网络的 mlp 特征
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # 执行注意力计算
        attn = attention(q, k, v, pe=pe)
        # 将注意力结果和激活后的 MLP 结果拼接，通过第二层线性网络，再用门控 (gate) 控制残差接入量
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


# 最终映射层：将 Transformer 计算完的深层特征，映射回图像区块 (Patch) 的大小，用来预测噪声。
class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 最后一次通过 vec 进行 AdaLN 调制
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x