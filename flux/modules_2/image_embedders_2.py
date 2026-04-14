"""
本文件核心：
图像条件编码器 (Image Embedders) 的实现。
当 FLUX 不仅仅依赖“文本”生图，还需要额外的“图像”作为条件控制时（例如 ControlNet 常见的玩法: Canny 边缘控制、Depth 深度图控制，或 Redux 风格参考），
这个文件负责把用户上传的“参考图”转换成模型能直接使用的特征张量。
没有这些 Embedder, FLUX 就像在蒙眼狂奔, 你只能靠抽卡(改随机种子)来碰运气。有了它们, FLUX 就变成了真正的商业级设计软件。
你可以用画笔随便勾勒几根线条(Canny), 或者用简单的 3D 软件拉一个白模(Depth), FLUX 就能精准地为你渲染出大片
"""

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file as load_sft
from torch import nn
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel

from flux.util import print_load_warning

" ========== 深度图特征提取器 ========== "
# 作用：将输入的普通 RGB 图像，转换成黑白灰的“深度图（Depth Map）”特征，用来控制生成图像的三维空间景深。
class DepthImageEncoder:
    # 使用预训练的 Depth Anything 模型来识别图像深度
    depth_model_name = "LiheYoung/depth-anything-large-hf"

    def __init__(self, device):
        self.device = device
        # 加载深度估计模型及其对应的图像处理器
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(self.depth_model_name)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # 记录输入图像的原始宽高，方便后面还原尺寸
        hw = img.shape[-2:]

        # 将张量数值范围限制在 [-1, 1] 之间，并转换为适合模型输入的 uint8 字节格式 (0-255)
        img = torch.clamp(img, -1.0, 1.0)
        img_byte = ((img + 1.0) * 127.5).byte()

        # 1. 预处理：将其转为 HuggingFace 模型需要的格式
        img = self.processor(img_byte, return_tensors="pt")["pixel_values"]
        # 2. 推理：丢给深度模型，拿到单通道的深度预测结果
        depth = self.depth_model(img.to(self.device)).predicted_depth
        # 3. 后处理：单通道复制成 3 通道（为了适配后续网络），并用双三次插值缩放回原始宽高
        depth = repeat(depth, "b h w -> b 3 h w")
        depth = torch.nn.functional.interpolate(depth, hw, mode="bicubic", antialias=True)

        # 重新归一化到 [-1, 1] 的数值范围，交给主网络
        depth = depth / 127.5 - 1.0
        return depth


" ========== 边缘线条特征提取器 ========== "
# 作用：提取输入图像的骨架线条（Canny 边缘），常用于线稿上色或严格保持原始物体的轮廓。
class CannyImageEncoder:
    def __init__(
        self,
        device,
        min_t: int = 50,  # Canny 算法的下阈值
        max_t: int = 200, # Canny 算法的上阈值
    ):
        self.device = device
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Canny 算法通常基于 CPU 端的 OpenCV 运行，所以目前强制要求 batch size 为 1
        assert img.shape[0] == 1, "Only batch size 1 is supported"

        # PyTorch 张量 (C, H, W) 转换回传统的图像矩阵 (H, W, C)，并还原成 0-255 的 numpy 数组
        img = rearrange(img[0], "c h w -> h w c")
        img = torch.clamp(img, -1.0, 1.0)
        img_np = ((img + 1.0) * 127.5).numpy().astype(np.uint8)

        # 核心：调用 OpenCV 的 Canny 函数进行边缘检测，得到一张非黑即白的线稿图
        canny = cv2.Canny(img_np, self.min_t, self.max_t)

        # 把 Numpy 生成的单通道线稿图变回 PyTorch 张量
        # 转换回 [-1, 1] 的范围，并将单通道复制扩展为 3 通道，送入设备
        canny = torch.from_numpy(canny).float() / 127.5 - 1.0
        canny = rearrange(canny, "h w -> 1 1 h w")
        canny = repeat(canny, "b 1 ... -> b 3 ...")
        return canny.to(self.device)


" ========== 全局视觉语义提取器 (Redux) ========== "
# 作用：这是一个强大的“图像转特征”模块，用于实现“垫图/风格参考（Image Prompt）”功能。
# 它不需要提取线条或深度，而是直接理解原图的“氛围、风格、主体”，将其转换为类似文本提示词的高维向量，去影响画面的生成。
class ReduxImageEncoder(nn.Module):
    # 使用 Google 强大的 SigLIP 模型（CLIP 的进化版）作为视觉理解的大脑
    siglip_model_name = "google/siglip-so400m-patch14-384"

    def __init__(
        self,
        device,
        redux_path: str,
        redux_dim: int = 1152,        # SigLIP 提取出来的特征维度
        txt_in_features: int = 4096,  # FLUX 主网络期望接收的特征维度 (要和文本 T5 的维度对齐)
        dtype=torch.bfloat16,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        with self.device:
            # 这是一个关键的适配器 (Adapter) 网络：
            # 由于视觉模型(SigLIP)输出的特征形状，与 FLUX 主干网络能听懂的特征形状不匹配，
            # 需要用线性层进行维度映射：先升维(up)再降维(down)
            self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
            self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

            # 加载适配器权重
            sd = load_sft(redux_path, device=str(device))
            missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
            print_load_warning(missing, unexpected)

            # 加载基础的 SigLIP 视觉模型
            self.siglip = SiglipVisionModel.from_pretrained(self.siglip_model_name).to(dtype=dtype)
        # 加载对应的图像处理器
        self.normalize = SiglipImageProcessor.from_pretrained(self.siglip_model_name)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        # 1. 把普通的 PIL 图片裁剪缩放并标准化
        imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)

        # 2. 扔给 SigLIP，提取出代表整张图片深层语义的隐藏状态 (last_hidden_state)
        _encoded_x = self.siglip(**imgs.to(device=self.device, dtype=self.dtype)).last_hidden_state

        # 3. 通过适配器 (Adapter)：把 SigLIP 的特征映射成 FLUX Transformer 能够理解和接收的特征维度
        # 中间使用了 SiLU 激活函数增加非线性
        projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))

        return projected_x