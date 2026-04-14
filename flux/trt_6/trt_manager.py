"""
本文件核心：
这是整个 TRT 模块中最核心的调度文件。
它的唯一核心任务就是提供一个 load_engines() 方法，帮你把所有的脏活累活全干了。
trt_manager.py 就是一个高度自动化的脚本。
外部代码不需要知道模型该怎么编译、显存该怎么共享、CUDA 流该怎么开启。
外部代码只需要告诉 TRTManager 需要 fp8 精度的 T5 和 bf16 精度的 Transformer”, 就会自动把一切准备就绪, 出一套可以直接用来算图的引擎组合
"""
#
# SPDX-FileCopyrightText: 版权所有 (c) 1993-2024 英伟达公司及其附属公司。保留所有权利。
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


import gc
import os
import sys
import warnings

import tensorrt as trt
import torch

from flux.trt.engine import (
    BaseEngine,
    CLIPEngine,
    Engine,
    SharedMemory,
    T5Engine,
    TransformerEngine,
    VAEDecoder,
    VAEEncoder,
    VAEEngine,
)
from flux.trt.trt_config import (
    ModuleName,
    TRTBaseConfig,
    get_config,
)

TRT_LOGGER = trt.Logger()
# 定义 Transformer 和 T5 允许的硬件加速精度
VALID_TRANSFORMER_PRECISIONS = {"bf16", "fp8", "fp4", "fp4_svd32"}
VALID_T5_PRECISIONS = {"bf16", "fp8"}


class TRTManager:
    # 映射表：把枚举名称映射到具体的运行期 Engine 类
    @property
    def module_to_engine_class(self) -> dict[ModuleName, type[Engine]]:
        return {
            ModuleName.CLIP: CLIPEngine,
            ModuleName.TRANSFORMER: TransformerEngine,
            ModuleName.T5: T5Engine,
            ModuleName.VAE: VAEDecoder,
            ModuleName.VAE_ENCODER: VAEEncoder,
        }

    def __init__(
        self,
        trt_transformer_precision: str,
        trt_t5_precision: str,
        max_batch=2,
        verbose=False,
    ):
        self.max_batch = max_batch
        # 解析并校验用户要求的精度
        self.precisions = self._parse_models_precisions(
            trt_transformer_precision=trt_transformer_precision,
            trt_t5_precision=trt_t5_precision,
        )
        self.verbose = verbose
        self.runtime: trt.Runtime = None
        
        # ★ 显存管理核心 ★
        # 在这里实例化之前讲过的 SharedMemory 单例。
        # 接下来创建的所有 Engine 实例，都会被强制注入这同一个显存池对象，从而实现显存轮流复用。
        self.device_memory = SharedMemory(1024)

        assert torch.cuda.is_available(), "No cuda device available"

    @staticmethod
    def _parse_models_precisions(
        trt_transformer_precision: str, trt_t5_precision: str
    ) -> dict[ModuleName, str]:
        # CLIP 和 VAE 通常计算量不是瓶颈，为了保证图像质量，强制锁定在 bf16 精度
        precisions = {
            ModuleName.CLIP: "bf16",
            ModuleName.VAE: "bf16",
            ModuleName.VAE_ENCODER: "bf16",
        }

        assert (
            trt_transformer_precision in VALID_TRANSFORMER_PRECISIONS
        ), f"Invalid precision for flux-transformer `{trt_transformer_precision}`. Possible value are {VALID_TRANSFORMER_PRECISIONS}"
        precisions[ModuleName.TRANSFORMER] = (
            trt_transformer_precision if trt_transformer_precision != "fp4_svd32" else "fp4"
        )

        assert (
            trt_t5_precision in VALID_T5_PRECISIONS
        ), f"Invalid precision for T5 `{trt_t5_precision}`. Possible value are {VALID_T5_PRECISIONS}"
        precisions[ModuleName.T5] = trt_t5_precision
        return precisions

    @staticmethod
    def _parse_custom_onnx_path(custom_onnx_paths: str) -> dict[ModuleName, str]:
        """(解析逗号分隔的自定义 ONNX 模型路径，方便用户使用自己微调的模型)"""
        parsed = {}

        for key_value_pair in custom_onnx_paths.split(","):
            if not key_value_pair:
                continue

            key_value_pair = key_value_pair.split(":")
            if len(key_value_pair) != 2:
                raise ValueError(f"Invalid key-value pair: {key_value_pair}. Must have length 2.")
            key, value = key_value_pair
            key = ModuleName(key)
            parsed[key] = value

        return parsed

    @staticmethod
    def _create_directories(engine_dir: str):
        print(f"[I] Create directory: {engine_dir} if not existing")
        os.makedirs(engine_dir, exist_ok=True)

    # 第一阶段：准备图纸。
    # 遍历需要加载的模块（比如 t5, vae），根据之前注册的 get_config 工厂方法，提取对应的 Config 类并实例化。
    def _get_trt_configs(
        self,
        model_name: str,
        module_names: set[ModuleName],
        engine_dir: str,
        custom_onnx_paths: dict[ModuleName, str],
        trt_static_batch: bool,
        trt_static_shape: bool,
        trt_enable_all_tactics: bool,
        trt_timing_cache: str | None,
        trt_native_instancenorm: bool,
        trt_builder_optimization_level: int,
        trt_precision_constraints: str,
        **kwargs,
    ) -> dict[ModuleName, TRTBaseConfig]:
        trt_configs = {}
        for module_name in module_names:
            config_cls = get_config(module_name=module_name, precision=self.precisions[module_name])
            custom_onnx_path = custom_onnx_paths.get(module_name, None)

            trt_config = config_cls.from_args(
                model_name=model_name,
                max_batch=self.max_batch,
                custom_onnx_path=custom_onnx_path,
                engine_dir=engine_dir,
                trt_verbose=self.verbose,
                precision=self.precisions[module_name],
                trt_static_batch=trt_static_batch,
                trt_static_shape=trt_static_shape,
                trt_enable_all_tactics=trt_enable_all_tactics,
                trt_timing_cache=trt_timing_cache,
                trt_native_instancenorm=trt_native_instancenorm,
                trt_builder_optimization_level=trt_builder_optimization_level,
                trt_precision_constraints=trt_precision_constraints,
                **kwargs,
            )

            trt_configs[module_name] = trt_config

        # 确保 Transformer 接收的文本长度与 T5 解析出的长度严格一致
        if ModuleName.TRANSFORMER in trt_configs and ModuleName.T5 in trt_configs:
            trt_configs[ModuleName.TRANSFORMER].text_maxlen = trt_configs[ModuleName.T5].text_maxlen
        else:
            warnings.warn("`text_maxlen` attribute of flux-trasformer is not update. Default value is used.")

        return trt_configs

    # 第二阶段：执行编译。
    # 这里体现了“惰性编译（Lazy Build）”逻辑。
    @staticmethod
    def _build_engine(
        trt_config: TRTBaseConfig,
        batch_size: int,
        image_height: int | None,
        image_width: int | None,
    ):
        # 检查硬盘上是否已经存在对应的 .plan 引擎文件
        already_build = os.path.exists(trt_config.engine_path)
        if already_build:
            return  # 已经有了就直接跳过编译，节省大量时间

        # 没有的话，就调用 config 里的 build_trt_engine 拼接命令行，执行极度耗时的底层编译转换
        trt_config.build_trt_engine(
            engine_path=trt_config.engine_path,
            onnx_path=trt_config.onnx_path,
            strongly_typed=trt_config.trt_build_strongly_typed,
            tf32=trt_config.trt_tf32,
            bf16=trt_config.trt_bf16,
            fp8=trt_config.trt_fp8,
            fp4=trt_config.trt_fp4,
            input_profile=trt_config.get_input_profile(
                batch_size=batch_size,
                image_height=image_height,
                image_width=image_width,
            ),
            enable_all_tactics=trt_config.trt_enable_all_tactics,
            timing_cache=trt_config.trt_timing_cache,
            update_output_names=trt_config.trt_update_output_names,
            builder_optimization_level=trt_config.trt_builder_optimization_level,
            verbose=trt_config.trt_verbose,
        )

        TRTManager._clean_memory()

    # ★ 核心公开接口 ★ ：上层代码 (cli) 直接调用的方法。
    def load_engines(
        self,
        model_name: str,
        module_names: set[ModuleName],
        engine_dir: str,
        trt_image_height: int | None,
        trt_image_width: int | None,
        trt_batch_size=1,
        trt_static_batch=True,
        trt_static_shape=True,
        trt_enable_all_tactics=False,
        trt_timing_cache: str | None = None,
        trt_native_instancenorm=True,
        trt_builder_optimization_level=3,
        trt_precision_constraints="none",
        custom_onnx_paths="",
        **kwargs,
    ) -> dict[ModuleName, BaseEngine]:
        TRTManager._clean_memory()
        TRTManager._create_directories(engine_dir)
        custom_onnx_paths = TRTManager._parse_custom_onnx_path(custom_onnx_paths)

        # 步骤 1：生成配置图纸
        trt_configs = self._get_trt_configs(
            model_name,
            module_names,
            engine_dir=engine_dir,
            custom_onnx_paths=custom_onnx_paths,
            trt_static_batch=trt_static_batch,
            trt_static_shape=trt_static_shape,
            trt_enable_all_tactics=trt_enable_all_tactics,
            trt_timing_cache=trt_timing_cache,
            trt_native_instancenorm=trt_native_instancenorm,
            trt_builder_optimization_level=trt_builder_optimization_level,
            trt_precision_constraints=trt_precision_constraints,
            **kwargs,
        )

        # 步骤 2：检查并执行缺失的 TRT 引擎编译任务
        for module_name, trt_config in trt_configs.items():
            self._build_engine(
                trt_config=trt_config,
                batch_size=trt_batch_size,
                image_height=trt_image_height,
                image_width=trt_image_width,
            )

        # 步骤 3：初始化底层的 C++ Runtime 运行时环境
        self.init_runtime()
        
        # 步骤 4：加载编译好的引擎文件，并实例化 Engine 对象
        engines = {}
        for module_name, trt_config in trt_configs.items():
            engine_class = self.module_to_engine_class[module_name]
            # 关键操作：在此处将统一的 self.device_memory (共享显存池) 注入给所有引擎！
            engine = engine_class(
                trt_config=trt_config,
                stream=self.stream,
                context_memory=self.device_memory,
                allocation_policy=os.getenv("TRT_ALLOCATION_POLICY", "global"),
            )
            engines[module_name] = engine

        # 步骤 5：如果是 VAE，因为其分为编码和解码两部分，将其打包为一个 VAEEngine 外壳以对齐 PyTorch 接口
        if ModuleName.VAE in engines:
            engines[ModuleName.VAE] = VAEEngine(
                decoder=engines.pop(ModuleName.VAE),
                encoder=engines.pop(ModuleName.VAE_ENCODER, None),
            )
            
        self._clean_memory()
        # 返回准备就绪的、随时可以开始计算的引擎字典
        return engines

    @staticmethod
    def _clean_memory():
        gc.collect()
        torch.cuda.empty_cache()

    # 初始化 TRT 运行时环境，并拉起一个专门的 CUDA Stream 用来进行无阻塞的高速矩阵计算
    def init_runtime(self):
        print("[I] Init TRT runtime")
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)
        self.stream = torch.cuda.current_stream()

    # 销毁 TRT 运行时环境，释放底层指针和显存
    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
        del self.stream
        del self.device_memory
        print("[I] Stop TRT runtime")