"""
本文件核心：
变分自编码器 (VAE) 的实现。
FLUX 并非直接在几百万个像素上进行扩散生图（算力会爆炸），
而是将图像压缩到了一个更小、更密集的“潜空间 (Latent Space) ”里进行生成, 最后再解压出来
本文件定义了完整的压缩(Encoder)和解压(Decoder)流水线以及相应的卷积组件
"""
#训练时（起点）：现实中的高清图片像素太多了，直接在上面训练扩散模型会非常慢，甚至无法训练。
# Encoder 会把这些高维的像素图压缩成一个更小的潜空间表示，这个表示虽然维度更低，但保留了图像的核心语义信息。
#推理/生成时（终点）：我们在潜空间里进行扩散生成，得到一个新的潜空间向量
# Decoder 会把这个向量解压回高清图片

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn


# 定义 VAE 的基本参数图纸：比如输入输出通道数、通道倍乘率、潜空间维度等
@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


# Swish 激活函数：在 VAE 的卷积网络中表现良好的平滑非线性函数
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


" ========== 卷积组件块 ========== "

# 自注意力层 (空间维度的 Attention)：用于捕捉 2D 图像特征图上的全局语义依赖
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # 使用组归一化 (GroupNorm)，这对生成模型中的卷积层很有效
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 将 2D 特征图 (B, C, H, W) 展平为序列 (B, 1, H*W, C) 才能做点积注意力
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 算完之后再折叠回图像格式
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        # 残差连接
        return x + self.proj_out(self.attention(x))


# 残差块 (ResNet Block)：经典的图像特征提取块，通过卷积层提取纹理，同时引入跳跃连接防止特征丢失
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            # 如果输入输出通道数不一致，使用 1x1 卷积调整通道数以匹配残差相加的维度
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


# 下采样层 (缩小尺寸)：用步长为 2 的卷积，让图像的长宽减半，起到“空间压缩”作用
class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # PyTorch 的卷积不支持不对称的 padding，必须手动填充
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1) # 右边和下边填充 1 个像素
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


# 上采样层 (放大尺寸)：用临近插值法把长宽放大一倍，再过一次卷积平滑边缘，起到“空间解压”作用
class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


" ========== 核心流水线 ========== "

# 编码器 (Encoder)：负责把 像素世界 (高维度) 压缩到 潜空间 (低维度)
class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        # 初始特征提取
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        
        # 1. 逐步下采样阶段 (Downsampling)：层层缩小尺寸，增加通道数
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # 2. 中间瓶颈层 (Middle)：在最低分辨率下进行深层语义处理，引入注意力机制
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 3. 输出层 (End)：映射到潜空间的最终维度。注意这里输出是 2*z_channels，因为要给高斯分布输出均值和方差
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # hs 列表用来存储中间特征
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 执行中间层
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        # 执行输出层
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


# 解码器 (Decoder)：负责把 潜空间 (低维度) 解压回 像素世界 (高维度)
class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # 算出潜空间的起始通道数和分辨率
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in: 接收潜空间特征
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间瓶颈层 (Middle)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 逐步上采样阶段 (Upsampling)：不断放大特征图尺寸
        self.up = nn.ModuleList()
        # 逆向遍历分辨率配置
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # 从前面插入以保持顺序一致

        # 输出层 (End)：映射回图像的 3 通道 (RGB)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # 获取后续网络层的精度类型，为了保证数据兼容性
        upscale_dtype = next(self.up.parameters()).dtype

        # 接收潜变量
        h = self.conv_in(z)

        # 中间层处理
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 转换精度
        h = h.to(upscale_dtype)
        
        # 逐步上采样解压
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 映射回 RGB 空间
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


# 对角高斯分布采样器：VAE 的核心数学技巧 (Reparameterization Trick)
# Encoder 输出的并不是确定的特征图，而是正态分布的“均值”和“方差的对数”。
# 这个类负责从这个分布中随机采样出一个样本交给模型。这赋予了模型创造新图像的能力。
class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        # 把刚才 Encoder 输出的 2倍 通道切开，一半当做均值(mean)，一半当做对数方差(logvar)
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar) # 还原标准差
            # 均值 + 标准差 * 随机噪声 = 采样结果
            return mean + std * torch.randn_like(mean)
        else:
            return mean


# VAE 终极包装盒：统筹编码、采样、解码的完整流程
class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams, sample_z: bool = False):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian(sample=sample_z)

        # 潜空间的标准化系数：因为模型喜欢处理 0 附近的数值，如果不进行缩放，方差过大会让扩散模型崩溃
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    # 完整压缩动作
    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        # 根据经验设定的死常数进行数值标准化
        z = self.scale_factor * (z - self.shift_factor)
        return z

    # 完整解压动作
    def decode(self, z: Tensor) -> Tensor:
        # 逆标准化
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    # 连贯前向：压进去再弹出来算重构误差 (只在单独训练 VAE 模块本身时用到)
    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))