"""
本文件核心：
T5 模块的 TRT 编译图纸。
不仅支持默认的 bf16 精度，为了极限压榨显存，还支持将巨型 T5 模型量化编译为 fp8 精度格式。
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


import os
from dataclasses import dataclass

from huggingface_hub import snapshot_download

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, register_config
from flux.util import configs


# 注册双重精度图纸
@register_config(module_name=ModuleName.T5, precision="bf16")
@register_config(module_name=ModuleName.T5, precision="fp8")
@dataclass
class T5Config(TRTBaseConfig):
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
            # schnell 版本因为精简，仅需 256 长度，dev 满血版为 512
            text_maxlen=256 if model_name == "flux-schnell" else 512,
            hidden_size=configs[model_name].params.context_in_dim,
            model_name=model_name,
            module_name=ModuleName.T5,
            **kwargs,
        )

    def check_dims(self, batch_size: int) -> None:
        self._check_batch(batch_size)

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
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    # 重写寻址逻辑：如果是 fp8 精度，需要去 HF 下载专门转换好的 fp8 ONNX 模型
    def _get_onnx_path(self) -> str:
        if self.custom_onnx_path:
            return self.custom_onnx_path

        if self.precision == "fp8":
            repo_id = self._get_repo_id(self.model_name)
            snapshot_path = snapshot_download(repo_id, allow_patterns=["t5-fp8.opt/*"])
            onnx_model_path = os.path.join(snapshot_path, "t5-fp8.opt/model.onnx")
            return onnx_model_path

        else:
            return super()._get_onnx_path()