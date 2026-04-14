"""
config (配置文件) 和 engine (引擎文件) 的核心区别在于它们所处的程序生命周期不同：
config 负责在前期“编译转换”模型，而 engine 负责在后期“加载运行”模型
config 文件（编译与配置）: 核心职责：定义规则并生成经过优化的模型文件
engine 文件（执行与显存管理）: 核心职责：加载编译好的模型，并在生图时执行具体的矩阵计算
"""

"""
本文件核心：
这是 Python 包的初始化文件。它唯一的代码逻辑就是把另外五个文件里定义的各种复杂类（如 BaseEngine, TransformerEngine, SharedMemory 等）
全部 import 进来，然后通过 __all__ 列表统一打包暴露给外部
TRT 引擎模块的初始化文件。
向外暴露（导出）了所有经过 TensorRT 加速重构的模块。
主程序 (如 cli.py) 只需要从这里导入对应的 Engine, 就能像使用原生 PyTorch 模型一样使用它们。
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

from flux.trt.engine.base_engine import BaseEngine, Engine, SharedMemory
from flux.trt.engine.clip_engine import CLIPEngine
from flux.trt.engine.t5_engine import T5Engine
from flux.trt.engine.transformer_engine import TransformerEngine
from flux.trt.engine.vae_engine import VAEDecoder, VAEEncoder, VAEEngine

# 定义当其他文件使用 `from flux.trt.engine import *` 时，具体导出哪些类
__all__ = [
    "BaseEngine",
    "Engine",
    "SharedMemory",
    "CLIPEngine",
    "TransformerEngine",
    "T5Engine",
    "VAEEngine",
    "VAEDecoder",
    "VAEEncoder",
]
