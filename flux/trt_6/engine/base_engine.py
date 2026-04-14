"""
本文件核心：
TensorRT 引擎的底层驱动与显存管家。
在深度学习推理中，最怕的就是每个模型各自分配显存导致 OOM (显存溢出)。
这个文件实现了一个全局的 SharedMemory (共享内存池)，让 T5、CLIP、Transformer、VAE 轮流复用同一块显存区域。
同时定义了 Engine 基类，封装了 TensorRT 极其繁琐的数据绑定、形状推断和异步执行逻辑。
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

import gc
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import tensorrt as trt
import torch
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

from flux.trt.trt_config import TRTBaseConfig

# 关闭 TRT 冗长的日志，只打印 ERROR 级别的错误
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

" ========== 显存共享池 (单例模式) ========== "
class SharedMemory(object):
    # 使用 __new__ 实现单例模式 (Singleton)：确保整个程序运行期间，显卡上只有这一个显存池管家
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(SharedMemory, cls).__new__(cls)
            cls.instance.__init__(*args, **kwargs)
        return cls.instance

    def __init__(self, size: int, device=torch.device("cuda")):
        self.allocations = {} # 记录每个组件向管家申请了多少显存
        # 预先在 GPU 上开辟一块连续的、指定大小的原始字节空间
        self._buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
            memory_format=torch.contiguous_format,
        )

    # 动态扩容机制：如果当前模型申请的显存超过了现有池子，就重新分配更大的空间
    def resize(self, name: str, size: int):
        self.allocations[name] = size
        if max(self.allocations.values()) > self._buffer.numel():
            self.buffer = self._buffer.resize_(size)
            torch.cuda.empty_cache()

    # 重置/缩容机制
    def reset(self, name: str):
        self.allocations.pop(name)
        new_max = max(self.allocations.values())
        if new_max < self._buffer.numel():
            self.buffer = self._buffer.resize_(new_max)
            torch.cuda.empty_cache()

    # 彻底释放显存池，恢复到初始的 1024 字节状态
    def deallocate(self):
        del self._buffer
        torch.cuda.empty_cache()
        self._buffer = torch.empty(
            1024,
            dtype=torch.uint8,
            device="cuda",
            memory_format=torch.contiguous_format,
        )

    # 返回这块物理显存的指针地址，供 TRT 底层绑定使用
    @property
    def shared_device_memory(self):
        return self._buffer.data_ptr()

    # 方便调试的打印格式，把 Bytes 换算成 MB/GB
    def __str__(self):
        def human_readable_size(size):
            for unit in ["B", "KiB", "MiB", "GiB"]:
                if size < 1024.0:
                    return size, unit
                size /= 1024.0
            return size, unit

        allocations_str = []

        for name, size_bytes in self.allocations.items():
            size, unit = human_readable_size(size_bytes)
            allocations_str.append(f"\t{name}: {size:.2f} {unit}\n")
        allocations_output = "".join(allocations_str)

        size, unit = human_readable_size(self._buffer.numel())
        allocations_buffer = f"{size:.2f} {unit}"
        return f"Shared Memory Allocations: \n{allocations_output} \n\tCurrent: {allocations_buffer}"


TRT_ALLOCATION_POLICY = {"global", "dynamic"}
# 内存极客策略：当模型闲置时，引擎文件是放在 CPU 内存的 bytes 格式，还是直接常驻显存？这里选了放内存。
TRT_OFFLOAD_POLICY = "cpu_buffer"


" ========== TRT 引擎基类 ========== "
class BaseEngine(ABC):
    # 类型翻译官：把 TensorRT 底层的 C++ 数据类型枚举，映射成 PyTorch 能识别的数据类型
    @staticmethod
    def trt_datatype_to_torch(datatype):
        datatype_mapping = {
            trt.DataType.BOOL: torch.bool,
            trt.DataType.UINT8: torch.uint8,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.BF16: torch.bfloat16,
        }
        if datatype not in datatype_mapping:
            raise ValueError(f"No PyTorch equivalent for TensorRT data type: {datatype}")

        return datatype_mapping[datatype]

    @abstractmethod
    def cpu(self) -> "BaseEngine":
        pass

    @abstractmethod
    def cuda(self) -> "BaseEngine":
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> "BaseEngine":
        pass


" ========== TRT 引擎标准实现 ========== "
# 所有具体的模块 (CLIP, T5, Transformer) 都要继承这个类。
# 它负责处理最让人头疼的 C++ 级显存指针绑定和执行上下文 (ExecutionContext) 管理。
class Engine(BaseEngine):
    def __init__(
        self,
        trt_config: TRTBaseConfig,
        stream: torch.cuda.Stream, # CUDA 流，用于实现非阻塞的异步计算
        context_memory: SharedMemory | None = None,
        allocation_policy: str = "global",
    ):
        self.trt_config = trt_config
        self.stream = stream
        self.context = None # TRT 执行上下文
        self.tensors = OrderedDict() # 存放输入输出张量的字典
        self.context_memory = context_memory
        self.device: torch.device = torch.device("cpu")

        # 初始化时根据策略读取硬盘上的 .engine 模型文件
        # cpu_buffer 意味着只读到内存里当做字节流，不占显存
        if TRT_OFFLOAD_POLICY == "cpu_buffer":
            self.engine: trt.ICudaEngine | bytes = None
            self.cpu_engine_buffer: bytes = bytes_from_path(self.trt_config.engine_path)
        else:
            self.engine: trt.ICudaEngine | bytes = bytes_from_path(self.trt_config.engine_path)

        assert allocation_policy in TRT_ALLOCATION_POLICY
        self.allocation_policy = allocation_policy
        self.current_input_hash = None
        self.cuda_graph = None

    @abstractmethod
    def __call__(self, *args, **Kwargs) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor]:
        pass

    # 卸载到 CPU：销毁执行上下文，将显卡上的引擎序列化回内存字节流
    def cpu(self) -> "Engine":
        if self.device == torch.device("cpu"):
            return self
        self.deactivate()
        if TRT_OFFLOAD_POLICY == "cpu_buffer":
            del self.engine
            return self
        self.engine = memoryview(self.engine.serialize())
        return self

    # 加载到 GPU：反序列化引擎，创建执行上下文，并向共享显存池挂载指针
    def cuda(self) -> "Engine":
        if self.device == torch.device("cuda"):
            return self
        buffer = self.cpu_engine_buffer if TRT_OFFLOAD_POLICY == "cpu_buffer" else self.engine
        self.engine = engine_from_bytes(buffer)
        gc.collect() # 强制垃圾回收
        # 创建上下文，但不让 TRT 自己乱申请显存 (without_device_memory)
        self.context = self.engine.create_execution_context_without_device_memory()
        # 统一使用我们一开始创建的 SharedMemory 池子
        self.context_memory.resize(self.__class__.__name__, self.device_memory_size)
        self.context.device_memory = self.context_memory.shared_device_memory
        return self

    # 设备切换包装器
    def to(self, device: str | torch.device) -> "Engine":
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if self.device == device:
            return self
        if device == torch.device("cpu"):
            self.cpu()
        else:
            self.cuda()
        self.device = device
        return self

    def deactivate(self):
        del self.context
        self.context = None

    # 初始化并分配输入/输出张量所需的显存空间
    def allocate_buffers(
        self,
        shape_dict: dict[str, tuple],
        device: str | torch.device = "cuda",
    ):
        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            tensor_shape = shape_dict[tensor_name]

            if tensor_name in self.tensors and self.tensors[tensor_name].shape == tensor_shape:
                continue

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, tensor_shape)
            tensor_dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(tensor_name))
            # 申请连续的显存块
            tensor = torch.empty(
                size=tensor_shape,
                dtype=tensor_dtype,
                memory_format=torch.contiguous_format,
            ).to(device=device)
            self.tensors[tensor_name] = tensor

    def get_dtype(self, tensor_name: str):
        return self.trt_datatype_to_torch(self.engine.get_tensor_dtype(tensor_name))

    # 动态形状支持 (Dynamic Shapes)：每次输入图像尺寸改变时，重新计算显存地址和绑定
    def override_shapes(self, feed_dict: Dict[str, torch.Tensor]):
        for name, tensor in feed_dict.items():
            shape = tensor.shape
            # 严格的数据类型断言防御
            assert tensor.dtype == self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name)), (
                f"Debug: Mismatched data types for tensor '{name}'. "
                f"Expected: {self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))}, "
                f"Found: {tensor.dtype} "
                f"in {self.__class__.__name__}"
            )
            self.context.set_input_shape(name, shape)

        assert self.context.all_binding_shapes_specified
        # 让 TRT 内部根据输入推断出输出的形状
        self.context.infer_shapes()
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                raise Exception("Unspecified shape identified for tensor {}: {} ".format(name, shape))
            # 重新分配零矩阵，并将底层指针 (data_ptr) 交给 TRT 上下文
            self.tensors[name] = torch.zeros(tuple(shape), dtype=dtype, device=self.device).contiguous()
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())

        if self.allocation_policy == "dynamic":
            self.context_memory.resize(self.__class__.__name__, self.device_memory_size)
        # 计算当前输入的哈希值，作为缓存标志
        self.current_input_hash = self.calculate_input_hash(feed_dict)

    def deallocate_buffers(self):
        if len(self.tensors) == 0:
            return

        del self.tensors
        self.tensors = OrderedDict()
        torch.cuda.empty_cache()

    @property
    def device_memory_size(self):
        if self.allocation_policy == "global":
            return self.engine.device_memory_size
        else:
            if not self.context.all_binding_shapes_specified:
                return 0
            return self.context.update_device_memory_size_for_shapes()

    @staticmethod
    def calculate_input_hash(feed_dict: Dict[str, torch.Tensor]):
        return hash(tuple(feed_dict[key].shape for key in sorted(feed_dict.keys())))

    # CUDA 图捕获 (高级优化技术，可降低 CPU 调度开销，这里暂时保留结构未使用)
    def _capture_cuda_graph(self):
        self.cuda_graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.graph(self.cuda_graph, stream=s):
            noerror = self.context.execute_async_v3(s.cuda_stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
        # self.cuda_graph.replay()

    # ★ 核心计算触发点 ★
    def infer(
        self,
        feed_dict: dict[str, torch.Tensor],
    ):
        # 如果当前输入的张量形状跟上次不一样，就触发上面的 override_shapes 重新绑定内存
        if self.current_input_hash != self.calculate_input_hash(feed_dict):
            self.override_shapes(feed_dict)

        # 确保池子挂载正确
        self.context.device_memory = self.context_memory.shared_device_memory
        
        # 将 PyTorch 传进来的真实数据，拷贝进我们预先分配好的、已经和 TRT 绑定的内存块里
        for name, tensor in feed_dict.items():
            self.tensors[name].copy_(tensor, non_blocking=True)

        # 发射指令！利用 CUDA Stream 进行极速的异步 C++ 推理计算
        noerror = self.context.execute_async_v3(self.stream.cuda_stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        # 返回计算结果（此时 TRT 已经把输出写进了 self.tensors 的相应输出槽位里）
        return self.tensors

    def __str__(self):
        if self.engine is None:
            return "Engine has not been initialized"
        out = ""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            out += f"\t{mode.name}: {name}={shape} {dtype.__str__()}\n"
        return out