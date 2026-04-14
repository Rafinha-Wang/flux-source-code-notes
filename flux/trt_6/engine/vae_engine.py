"""
本文件核心：
VAE (潜空间图像压缩/解压模块) 的 TRT 适配器。
因为 VAE 包括 Encoder (图变特征) 和 Decoder (特征变图) 两个完全独立的子网络，
所以这里定义了两个独立的 Engine，并用一个 VAEEngine 把它们打包起来，伪装成原本的 PyTorch VAE 模样。
同时，它在这里补齐了原本 autoencoder.py 里处理的数学归一化操作 (scale 和 shift)。
"""
#
# 法律与开源许可证声明:
# SPDX-FileCopyrightText: 版权所有 (c) 1993-2024 英伟达公司及其附属公司。保留所有权利。
# SPDX-License-Identifier: Apache-2.0
#
# 根据 Apache 许可证 2.0 版（以下简称“许可证”）获得授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样（AS IS）”的基础提供的，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证下管理权限和限制的具体条款，请参阅许可证文本。

import torch

from flux.trt.engine.base_engine import BaseEngine, Engine
from flux.trt.trt_config import VAEDecoderConfig, VAEEncoderConfig

" ========== 解码器引擎 ========== "
class VAEDecoder(Engine):
    def __init__(self, trt_config: VAEDecoderConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @torch.inference_mode()
    def __call__(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        z = z.to(dtype=self.get_dtype("latent"))
        # 核心算术：在交由 TRT 算庞大的反卷积之前，先在 PyTorch 层用纯数学操作进行逆标准化
        z = (z / self.trt_config.scale_factor) + self.trt_config.shift_factor
        
        feed_dict = {"latent": z}
        # 极速解码出 RGB 图像
        images = self.infer(feed_dict=feed_dict)["images"]
        return images

" ========== 编码器引擎 ========== "
class VAEEncoder(Engine):
    def __init__(self, trt_config: VAEEncoderConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feed_dict = {"images": x.to(dtype=self.get_dtype("images"))}
        # 极速卷积提取潜空间特征
        latent = self.infer(feed_dict=feed_dict)["latent"]
        
        # 核心算术：生成完后，在 PyTorch 层加上固定常数的标准化缩放和平移，防止数值爆炸
        latent = self.trt_config.scale_factor * (latent - self.trt_config.shift_factor)
        return latent

" ========== 终极打包器 ========== "
# 这个类实现了一个 Facade 模式（外观模式）。
# 上层的 sampling.py 调用 `ae.decode()` 或 `ae.encode()` 时，根本不会察觉到底层已经换成了 TensorRT 加速器，
# 实现了业务逻辑与底层加速框架的完美解耦。
class VAEEngine(BaseEngine):
    def __init__(
        self,
        decoder: VAEDecoder,
        encoder: VAEEncoder | None = None, # 纯文生图时不需要 Encoder，只有图生图才需要
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None, "An encoder is needed to encode an image"
        return self.encoder(x)

    # 包装设备转移：一口气把两个引擎都踢到 CPU 或者拉回 GPU
    def cpu(self):
        self.decoder = self.decoder.cpu()
        if self.encoder is not None:
            self.encoder = self.encoder.cpu()
        return self

    def cuda(self):
        self.decoder = self.decoder.cuda()
        if self.encoder is not None:
            self.encoder = self.encoder.cuda()
        return self

    def to(self, device):
        self.decoder = self.decoder.to(device)
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self

    # 显存计算：以其中占用显存最大的那个网络为准，作为当前 VAE 组件向 SharedMemory 共享显存池申报的大小
    @property
    def device_memory_size(self):
        device_memory = self.decoder.device_memory_size
        if self.encoder is not None:
            device_memory = max(device_memory, self.encoder.device_memory_size)
        return device_memory