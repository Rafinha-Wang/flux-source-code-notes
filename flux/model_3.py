"""
本文件核心：
FLUX 核心大模型 (The Mainboard / Backbone) 的总装配厂。
它负责把 `layers.py` 里定义的各种神经元组件(如 RoPE、双流块、单流块) 组装成一个完整的 Transformer。
它定义了数据从输入（噪声图像、文本向量、时间步）到输出（预测的噪声/图像速度场）的完整流向。
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora


" ========== 架构蓝图 ========== "
# 定义了实例化一台 FLUX 主机所需的全部超参数图纸
@dataclass
class FluxParams:
    in_channels: int          # 图像输入通道数 (通常是被 VAE 压缩后的潜空间通道)
    out_channels: int         # 图像输出通道数
    vec_in_dim: int           #  pooled 文本向量的维度 (来自 CLIP)
    context_in_dim: int       # 序列化文本向量的维度 (来自 T5)
    hidden_size: int          # Transformer 内部隐藏层的核心宽度 (神经元数量)
    mlp_ratio: float          # 前馈网络放大的倍数
    num_heads: int            # 注意力头的数量
    depth: int                # 前期：双流块 (DoubleStreamBlock) 的层数
    depth_single_blocks: int  # 后期：单流块 (SingleStreamBlock) 的层数
    axes_dim: list[int]       # 多维位置编码 (RoPE) 各个维度的分配
    theta: int                # RoPE 的基数
    qkv_bias: bool            # QKV 线性层是否使用偏置
    guidance_embed: bool      # 是否把 CFG (提示词引导系数) 作为一个可学习的条件嵌入模型


" ========== 核心大模型 ========== "
class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    (用于序列上流匹配的 Transformer 模型)
    注: FLUX 使用的是 Flow Matching (流匹配) 算法，这是比早期 DDPM 扩散模型更先进、训练更稳定的数学框架。
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        
        # 严谨的尺寸校验：隐藏层宽度必须能被注意力头数整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        
        # 1. 挂载位置编码器 (RoPE)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
        # 2. 输入特征的投影层 (统一将各类异构数据映射到 hidden_size 这个统一的维度空间)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)     # 处理图像潜变量
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)        # 处理时间步 t
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)          # 处理 CLIP 提取的文本摘要 (vec)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )                                                                          # 处理提示词引导强度
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)           # 处理 T5 提取的长文本序列 (txt)

        # 3. 挂载核心计算主板：前期的双流架构 (分别处理但相互注意的图文流水线)
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        # 4. 挂载核心计算主板：后期的单流架构 (图文拼在一起粗暴处理的高效流水线)
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 5. 输出映射层：把思考完毕的高维特征重新翻译回图像的像素/噪声维度
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)


    # 核心数据流向：每一次去噪步骤，数据都要经历以下的完整流转
    def forward(
        self,
        img: Tensor,          # 图像特征序列 (带噪声)
        img_ids: Tensor,      # 图像的空间位置坐标
        txt: Tensor,          # 文本特征序列 (T5)
        txt_ids: Tensor,      # 文本的位置坐标
        timesteps: Tensor,    # 当前所处的时间步/去噪阶段
        y: Tensor,            # 文本摘要向量 (CLIP)
        guidance: Tensor | None = None, # 引导强度
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # --- 阶段 1：特征嵌入与对齐 (Embeddings) ---
        img = self.img_in(img) # 将图像拉伸到模型内部的隐藏层维度
        
        # 组装全局控制条件 (vec)：包含时间进度、引导强度、整体文本氛围
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y) # 把这三个要素累加，作为一个统一的控制信号 (AdaLN 调制的来源)
        
        txt = self.txt_in(txt) # 将文本拉伸到统一维度

        # 生成位置编码：把文本的 1D 坐标和图像的 2D 坐标拼接，计算出旋转位置编码 (RoPE)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # --- 阶段 2：双流计算 (Double Blocks) ---
        # 图像和文本分别走各自的线性层，但在 Attention 时互相“看着”对方
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # --- 阶段 3：单流计算 (Single Blocks) ---
        # 前期对齐做得差不多了，把文本和图像沿着序列长度维度 (1) 直接暴力拼接在一起
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
            
        # 算完之后，把前缀的文本部分砍掉，只保留后面图像部分的数据
        img = img[:, txt.shape[1] :, ...]

        # --- 阶段 4：输出映射 (Final Layer) ---
        img = self.final_layer(img, vec)  # 输出形状: (N, 序列长度, patch_size ** 2 * out_channels)
        return img


" ========== LoRA 外挂包装器 ========== "
# 这个类巧妙地继承了上面的 Flux 大模型，但是在初始化之后，
# 直接调用了 `lora.py` 里的偷天换日函数，把体内所有的线性层换成了支持外挂微调的 LoRA 层。
class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        # 核心：自动化遍历替换
        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    # 允许在推理时随时动态拨动拉杆，调整 LoRA 的生效强度
    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)