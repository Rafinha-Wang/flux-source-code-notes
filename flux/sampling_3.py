"""
本文件核心：
生成管线的中枢指挥室 (The Orchestrator / Inference Engine)。
它不定义网络结构，而是负责在实际生图时调用所有的组件。
包含：制作最初的纯噪声画布、翻译文本提示词、构建时间步调度表 (Scheduler)、以及执行核心的循环去噪动作。
"""

import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS

" ========== 准备画布 ========== "
# 在潜空间 (Latent Space) 里生成一块指定分辨率的纯随机噪声图。
# 这就是模型画画的“原始白纸”。
def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16, # VAE 的潜空间通道数
        # 尺寸除以 16 是因为 VAE 会把图像的宽高压缩 8 倍，加上 Patch 化再除以 2
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed), # 固定种子确保出图可复现
    ).to(device)


" ========== 准备给模型的粮草 (数据格式化) ========== "
# 这个函数负责把人类的输入 (文本) 和纯噪声画布 (img)，打包整理成 model.py 能直接吃的格式
# u can see 大量的repeat :广播与对齐 (Batch Broadcasting)
def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # 1. 图像展平：Transformer 不能直接吃 2D 图片，必须把它切成小块 (Patch) 然后排成一排 (Sequence)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # 2. 建立图像的三维坐标系：为了让模型记住展平前谁挨着谁，这里创建了 (0, y坐标, x坐标) 的空间网格
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # 3. 文本处理：送入 T5，得到详细的上下文序列特征
    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # 文本是一维序列，所以 y 和 x 坐标都是 0

    # 4. 文本处理：送入 CLIP，得到浓缩的全局摘要特征
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    # 打包返回，这五样东西就是 Flux.forward 所需的全部基础原料
    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


" ========== 特殊条件的扩展准备函数 ========== "
# 以下这些 prepare_XXX 函数，都是基于基础的 prepare，额外塞入了 ControlNet 或 Inpainting 需要的参考图特征

# 控制网图 (Canny/Depth) 的预处理：提特征 -> 压缩过 VAE -> 和噪声图像在通道维度上合并
def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)  # 提取深度或边缘
        img_cond = ae.encode(img_cond) # 压入潜空间

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict

# 局部重绘 (Inpainting/Fill) 的预处理：除了参考图，还要处理黑白 Mask 遮罩，告诉模型“只能画这块区域”
def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        # 用 mask 抠出需要保留的原图部分
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        
        # 调整 Mask 尺寸对齐潜空间
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # 将原图特征和 mask 拼接在一起，强塞给模型
    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict

# 风格垫图 (Redux) 的预处理：利用 SigLIP 提取整图特征，然后把它伪装成“文本词汇”强行拼接在 T5 的文本序列后面
def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    
    # 核心：将垫图特征 img_cond 直接与文本向量 txt 拼接 (Concat)，实现了“图就是文，文就是图”
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

# 上下文垫图预处理
def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    width, height = img_cond.size
    aspect_ratio = width / height
    # Kontext 必须强制采用特定黄金比例进行对齐
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    width = 2 * int(width / 16)
    height = 2 * int(height / 16)

    img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    img_cond_orig = img_cond.clone()

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # 独立的位置编码处理，防止与生成的图像位置冲突
    img_cond_ids = torch.zeros(height // 2, width // 2, 3)
    img_cond_ids[..., 0] = 1
    img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

    if target_width is None:
        target_width = 8 * width
    if target_height is None:
        target_height = 8 * height

    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond_seq"] = img_cond
    return_dict["img_cond_seq_ids"] = img_cond_ids.to(device)
    return_dict["img_cond_orig"] = img_cond_orig
    return return_dict, target_height, target_width


" ========== 核心降噪调度表 (Scheduler) ========== "

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

# 确定去噪进度的“时间表” (例如分成 50 步，每一步减去多少比例的噪声)
def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # 生成一个从 1.0 (全噪声) 到 0.0 (无噪声) 的均匀下降序列
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # 关键创新点 (Time Shifting)：FLUX 发现，如果你画的图分辨率特别大，
    # 那么在刚开始全是噪声的阶段，模型需要花费更多的心智去构图。
    # 所以通过一个动态函数，把时间步向早期挤压，让模型在“大局构图”阶段多算几步，在“抠细节”阶段少算几步。
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


" ========== 终极去噪循环 (The Euler Loop) ========== "
# 这里是奇迹发生的地方。它将噪声图像送入 Flux 大脑，拿到预测结果后，不断逼近最终的清晰图像。
def denoise(
    model: Flux,
    # 模型所需的全部准备好的输入
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # 控制去噪节奏的参数
    timesteps: list[float],
    guidance: float = 4.0, # CFG 强度，告诉模型需要多大程度上听从提示词
    
    img_cond: Tensor | None = None,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    # 准备一整批相同的引导强度张量
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    # 核心循环：在刚刚生成的时间表里一步步往下走
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # 告诉模型现在进行到哪一步了
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        # 组装当前的输入（如果存在额外的条件特征，就在此合并进去）
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
            
        # 1. 唤醒模型进行预测 (Forward Pass)
        # 注意：这里的 pred 预测出的不是“图像本身”，而是去噪的“速度/方向” (Vector Field)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        # 2. 欧拉步进 (Euler Step)：基于 Flow Matching 的数学更新公式
        # 新图像 = 当前图像 + 时间步差值 * 预测出的速度场
        # 这个操作会将雪花噪声一点点雕刻出具体的图案
        img = img + (t_prev - t_curr) * pred

    # 循环走完，img 就从一堆乱码变成了含有完整语义的潜空间特征
    return img


" ========== 完工打包 ========== "
# 把刚才展平成 1D 长条的潜空间序列，重新折叠拼接回 2D 的矩形矩阵格式
# 为下一步交给 autoencoder.py 的 Decoder (解压为 RGB 像素) 做好准备
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )