"""
本文件核心：
CLIP 文本编码器的 TRT 适配器。
它是一个混合体：先用原生的 HuggingFace 分词器在 CPU 上把文本变成数字 Token，
然后把计算量极大的 Embedding 提取过程，外包给上面定义好的高性能 TensorRT 引擎 (infer)。
"""

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
from transformers import CLIPTokenizer

from flux.trt.engine import Engine
from flux.trt.trt_config import ClipConfig

class CLIPEngine(Engine):
    def __init__(self, trt_config: ClipConfig, stream: torch.cuda.Stream, **kwargs):
        # 继承 Engine 基类，完成底层内存和上下文的初始化
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)
        # 加载分词字典，这部分计算量极小，直接使用标准的 HF 库即可
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            max_length=self.trt_config.text_maxlen,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
        with torch.inference_mode():
            # 1. CPU 端查字典：将字符串列表转化为 Token ID 矩阵
            feed_dict = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.trt_config.text_maxlen,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            # 2. 准备传给 GPU 的弹药：转换数据类型对齐 TRT 引擎的期望
            feed_dict = {"input_ids": feed_dict["input_ids"].to(dtype=self.get_dtype("input_ids"))}

            # 3. 极速推理：调用 BaseEngine 的 infer 跑 CUDA，提取出结果字典中的 pooled_embeddings
            pooled_embeddings = self.infer(feed_dict)["pooled_embeddings"]

        return pooled_embeddings