"""
本文件核心：
FLUX 主大脑 (Diffusion Transformer / DiT) 的 TRT 适配器。
由于 NVIDIA 官方转导 ONNX/TRT 模型时，张量节点的命名规范可能与 FLUX 原生 Python 代码不一致，
这个类充当了“双向翻译机”，负责将外部传入的参数名转换为引擎底层认识的命名，然后执行高强度的去噪计算。
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

from flux.trt.engine import Engine
from flux.trt.trt_config import TransformerConfig


class TransformerEngine(Engine):
    # 张量名称映射表：[TRT/ONNX 底层期望的名字] -> [FLUX Python 代码里的名字]
    __dd_to_flux__ = {
        "hidden_states": "img",            # 图像噪声特征
        "img_ids": "img_ids",
        "encoder_hidden_states": "txt",    # T5 文本特征
        "pooled_projections": "y",         # CLIP 全局向量
        "txt_ids": "txt_ids",
        "timestep": "timesteps",           # 时间步
        "guidance": "guidance",            # CFG 指导系数
        "latent": "latent",                # 预测输出
    }

    # 反向映射表：[FLUX Python 名字] -> [TRT 底层名字]
    __flux_to_dd__ = {
        "img": "hidden_states",
        "img_ids": "img_ids",
        "txt": "encoder_hidden_states",
        "y": "pooled_projections",
        "txt_ids": "txt_ids",
        "timesteps": "timestep",
        "guidance": "guidance",
        "latent": "latent",
    }

    def __init__(self, trt_config: TransformerConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @property
    def dd_to_flux(self):
        return TransformerEngine.__dd_to_flux__

    @property
    def flux_to_dd(self):
        return TransformerEngine.__flux_to_dd__

    @torch.inference_mode()
    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        feed_dict = {}

        # 逻辑分支：schnell 是加速版，它在底层设计上就去掉了 guidance 引导计算，所以要把这个参数扔掉
        if self.trt_config.model_name == "flux-schnell":
            # remove guidance
            kwargs.pop("guidance")

        # 遍历外部传进来的所有参数 (img, txt, y 等)
        for tensor_name, tensor_value in kwargs.items():
            if tensor_name == "latent":
                continue
            # 根据字典，把它们改名为底层引擎认识的名字，并严格对齐数据精度
            dd_name = self.flux_to_dd[tensor_name]
            feed_dict[dd_name] = tensor_value.to(dtype=self.get_dtype(dd_name))

        # 维度兼容性修正：由于 Demo-Diffusion 格式的 TRT 引擎不接受位置编码前方的 Batch 维度，
        # 所以硬编码截取掉第一个维度 [0]，只保留核心数据传递给底层。
        feed_dict["img_ids"] = feed_dict["img_ids"][0]
        feed_dict["txt_ids"] = feed_dict["txt_ids"][0]

        # 开火！执行几十层的 Transformer 注意力计算
        latent = self.infer(feed_dict=feed_dict)["latent"]

        return latent