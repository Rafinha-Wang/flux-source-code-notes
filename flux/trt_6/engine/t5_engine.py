"""
本文件核心：
T5 文本编码器的 TRT 适配器。
逻辑与 clip_engine.py 几乎完全一致，只是查的字典变成了 T5，返回的是一整个序列的特征 (text_embeddings)。
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
from transformers import T5Tokenizer

from flux.trt.engine import Engine
from flux.trt.trt_config import T5Config


class T5Engine(Engine):
    def __init__(self, trt_config: T5Config, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl",
            max_length=self.trt_config.text_maxlen,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
        with torch.inference_mode():
            feed_dict = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.trt_config.text_maxlen,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            feed_dict = {"input_ids": feed_dict["input_ids"].to(dtype=self.get_dtype("input_ids"))}

            # 调用 TRT 执行 Transformer 计算，获取庞大的隐状态特征
            text_embeddings = self.infer(feed_dict)["text_embeddings"]

        return text_embeddings