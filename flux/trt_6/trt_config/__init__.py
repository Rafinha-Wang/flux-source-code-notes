"""
config (配置文件) 和 engine (引擎文件) 的核心区别在于它们所处的程序生命周期不同：
config 负责在前期“编译转换”模型，而 engine 负责在后期“加载运行”模型
config 文件（编译与配置）: 核心职责：定义规则并生成经过优化的模型文件
engine 文件（执行与显存管理）: 核心职责：加载编译好的模型，并在生图时执行具体的矩阵计算
"""


"""
本文件核心：
TRT 配置生成器的初始化文件。
统一对外导出各个子模块的模型编译参数与配置类。
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

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, get_config, register_config
from flux.trt.trt_config.clip_trt_config import ClipConfig
from flux.trt.trt_config.t5_trt_config import T5Config
from flux.trt.trt_config.transformer_trt_config import TransformerConfig
from flux.trt.trt_config.vae_trt_config import VAEDecoderConfig, VAEEncoderConfig

__all__ = [
    "register_config",
    "get_config",
    "ModuleName",
    "TRTBaseConfig",
    "ClipConfig",
    "T5Config",
    "TransformerConfig",
    "VAEDecoderConfig",
    "VAEEncoderConfig",
]