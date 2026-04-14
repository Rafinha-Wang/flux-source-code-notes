"""
本文件核心：
TensorRT 引擎的“超级组装工厂”。
它不仅仅存储变量，最核心的是 `build_trt_engine` 方法，该方法通过在系统底层拼装一段复杂的 CLI 命令行，
调用 Nvidia 强大的 `polygraphy` 工具，将普通的 ONNX 模型文件，硬核编译（AOT编译）成针对当前显卡绝对优化的 .plan (Engine) 文件。
同时它实现了一个注册中心 (Registry)，方便根据不同精度 (fp8, bf16) 动态获取对应的配置图纸。
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
import subprocess
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from colored import fore, style
from huggingface_hub import snapshot_download
from tensorrt import __version__ as trt_version

# 枚举：定义当前系统支持加速的核心组件
class ModuleName(Enum):
    CLIP = "clip"
    T5 = "t5"
    TRANSFORMER = "transformer"
    VAE = "vae"
    VAE_ENCODER = "vae_encoder"


# 全局注册字典，用于存储不同模块和精度组合下的配置类
registry = {}


@dataclass
class TRTBaseConfig:
    engine_dir: str                 # 编译好的引擎存放目录
    precision: str                  # 目标计算精度 (如 bf16, fp8)
    trt_verbose: bool               # 是否打印编译过程中的海量日志
    trt_static_batch: bool          # 是否锁死 Batch Size（锁死能提升极限性能，但不灵活）
    trt_static_shape: bool          # 是否锁死图像分辨率
    model_name: str
    module_name: ModuleName
    onnx_path: str = field(init=False)   # 下载/指定的源 ONNX 模型路径
    engine_path: str = field(init=False) # 最终生成的 .plan 路径
    
    # TRT 底层硬件特性开关
    trt_tf32: bool
    trt_bf16: bool
    trt_fp8: bool
    trt_fp4: bool
    trt_build_strongly_typed: bool
    
    custom_onnx_path: str | None = None
    trt_update_output_names: list[str] | None = None
    trt_enable_all_tactics: bool = False
    trt_timing_cache: str | None = None          # 编译缓存（TRT编译巨慢，有了它下次编译能快很多）
    trt_native_instancenorm: bool = True
    trt_builder_optimization_level: int = 3      # 编译器优化等级 (3 代表相当高的优化)
    trt_precision_constraints: str = "none"

    min_batch: int = 1
    max_batch: int = 4

    # ★ 核心编译器 ★ ：拼装命令行，调用系统进程执行底层编译
    @staticmethod
    def build_trt_engine(
        engine_path: str,
        onnx_path: str,
        strongly_typed=False,
        tf32=True,
        bf16=False,
        fp8=False,
        fp4=False,
        input_profile: dict[str, Any] | None = None,
        update_output_names: list[str] | None = None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache: str | None = None,
        native_instancenorm=True,
        builder_optimization_level=3,
        precision_constraints="none",
        verbose=False,
    ):
        """(省略原版英文文档，逻辑是调用 polygraphy 将 ONNX 编译为 TRT)"""
        print(f"Building TensorRT engine for {onnx_path}: {engine_path}")

        # 基础命令：调用 polygraphy
        build_command = [f"polygraphy convert {onnx_path} --convert-to trt --output {engine_path}"]

        # 精度控制标志
        build_args = [
            "--bf16" if bf16 else "",
            "--tf32" if tf32 else "",
            "--fp8" if fp8 else "",
            "--fp4" if fp4 else "",
            "--strongly-typed" if strongly_typed else "",
        ]

        # 附加优化参数
        build_args.extend(
            [
                "--refittable" if enable_refit else "",
                "--tactic-sources" if not enable_all_tactics else "",
                "--onnx-flags native_instancenorm" if native_instancenorm else "",
                f"--builder-optimization-level {builder_optimization_level}",
                f"--precision-constraints {precision_constraints}",
            ]
        )

        # 编译时序缓存
        if timing_cache:
            build_args.extend([f"--load-timing-cache {timing_cache}", f"--save-timing-cache {timing_cache}"])

        # 详细日志控制
        verbosity = "extra_verbose" if verbose else "error"
        build_args.append(f"--verbosity {verbosity}")

        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            build_args.append(f"--trt-outputs {' '.join(update_output_names)}")

        # ★ 动态形状配置 (Dynamic Profiles) ★
        # TensorRT 需要知道输入张量的【最小、最优、最大】尺寸范围，以便在编译时分配合适的显存策略
        if input_profile:
            profile_args = defaultdict(str)
            for name, dims in input_profile.items():
                assert len(dims) == 3
                profile_args["--trt-min-shapes"] += f"{name}:{str(list(dims[0])).replace(' ', '')} "
                profile_args["--trt-opt-shapes"] += f"{name}:{str(list(dims[1])).replace(' ', '')} "
                profile_args["--trt-max-shapes"] += f"{name}:{str(list(dims[2])).replace(' ', '')} "

            build_args.extend(f"{k} {v}" for k, v in profile_args.items())

        # 拼接成最终的 Linux 终端执行命令
        build_args = [arg for arg in build_args if arg]
        final_command = " \\\n".join(build_command + build_args)

        # 唤醒子进程执行编译，并进行错误拦截
        try:
            print(f"Engine build command:{fore('yellow')}\n{final_command}\n{style('reset')}")
            subprocess.run(final_command, check=True, shell=True)
        except subprocess.CalledProcessError as exc:
            error_msg = f"Failed to build TensorRT engine. Error details:\nCommand: {exc.cmd}\n"
            raise RuntimeError(error_msg) from exc

    @classmethod
    @abstractmethod
    def from_args(cls, model_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError("Factory method is missing")

    # 要求子类必须提供当前组件的“动态尺寸配置表”
    @abstractmethod
    def get_input_profile(
        self,
        batch_size: int,
        image_height: int | None,
        image_width: int | None,
    ) -> dict[str, Any]:
        """(省略原版英文文档)"""
        pass

    @abstractmethod
    def check_dims(self, *args, **kwargs) -> None | tuple[int, int] | int:
        pass

    def _check_batch(self, batch_size):
        assert (
            self.min_batch <= batch_size <= self.max_batch
        ), f"Batch size {batch_size} must be between {self.min_batch} and {self.max_batch}"

    def __post_init__(self):
        # 自动推导和校验模型路径
        self.onnx_path = self._get_onnx_path()
        self.engine_path = self._get_engine_path()
        assert os.path.isfile(self.onnx_path), "onnx_path do not exists: {}".format(self.onnx_path)

    # 如果本地没有指定的 ONNX，就去 HuggingFace 自动下载对应组件的 ONNX 模型
    def _get_onnx_path(self) -> str:
        if self.custom_onnx_path:
            return self.custom_onnx_path

        repo_id = self._get_repo_id(self.model_name)
        snapshot_path = snapshot_download(repo_id, allow_patterns=[f"{self.module_name.value}.opt/*"])
        onnx_model_path = os.path.join(snapshot_path, f"{self.module_name.value}.opt/model.onnx")
        return onnx_model_path

    def _get_engine_path(self) -> str:
        return os.path.join(
            self.engine_dir,
            self.model_name,
            f"{self.module_name.value}_{self.precision}.trt_{trt_version}.plan",
        )

    # HF 仓库寻址表
    @staticmethod
    def _get_repo_id(model_name: str) -> str:
        if model_name == "flux-dev":
            return "black-forest-labs/FLUX.1-dev-onnx"
        elif model_name == "flux-schnell":
            return "black-forest-labs/FLUX.1-schnell-onnx"
        elif model_name == "flux-dev-canny":
            return "black-forest-labs/FLUX.1-Canny-dev-onnx"
        elif model_name == "flux-dev-depth":
            return "black-forest-labs/FLUX.1-Depth-dev-onnx"
        elif model_name == "flux-dev-kontext":
            return "black-forest-labs/FLUX.1-Kontext-dev-onnx"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

# 装饰器：用于在子类定义时，自动将其注册到大字典中，方便按需加载
def register_config(module_name: ModuleName, precision: str):
    def decorator(cls):
        key = f"module={module_name.value}_dtype={precision}"
        registry[key] = cls
        return cls

    return decorator

def get_config(module_name: ModuleName, precision: str) -> TRTBaseConfig:
    key = f"module={module_name.value}_dtype={precision}"
    return registry[key]