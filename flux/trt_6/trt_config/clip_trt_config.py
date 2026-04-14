"""
本文件核心：
CLIP 模块的 TRT 编译图纸。
向 TRT 编译器声明了 CLIP 期望的 Token 最大长度 (77)。
"""

#
# SPDX-FileCopyrightText: 版权所有 (c) 1993-2025 英伟达公司及其附属公司。保留所有权利。
# SPDX-License-Identifier: Apache-2.0
#
# 根据 Apache 许可证 2.0 版（以下简称“许可证”）获得授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样（AS IS）”的基础提供的，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证下管理权限和限制的具体条款，请参阅许可证文本。
#

from dataclasses import dataclass

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, register_config
from flux.util import configs


@register_config(module_name=ModuleName.CLIP, precision="bf16")
@dataclass
class ClipConfig(TRTBaseConfig):
    text_maxlen: int | None = None
    hidden_size: int | None = None
    trt_tf32: bool = True
    trt_bf16: bool = False
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = True

    @classmethod
    def from_args(
        cls,
        model_name: str,
        **kwargs,
    ):
        return cls(
            text_maxlen=77, # CLIP 标准输入长度
            hidden_size=configs[model_name].params.vec_in_dim,
            model_name=model_name,
            module_name=ModuleName.CLIP,
            **kwargs,
        )

    def check_dims(self, batch_size: int) -> None:
        self._check_batch(batch_size)

    # 关键：告诉编译器输入张量 (input_ids) 的形状变化范围
    def get_input_profile(
        self,
        batch_size: int,
        image_height=None,
        image_width=None,
    ):
        min_batch = batch_size if self.trt_static_batch else self.min_batch
        max_batch = batch_size if self.trt_static_batch else self.max_batch

        self.check_dims(batch_size)
        return {
            "input_ids": [
                (min_batch, self.text_maxlen), # Min
                (batch_size, self.text_maxlen), # Opt (最优)
                (max_batch, self.text_maxlen), # Max
            ]
        }