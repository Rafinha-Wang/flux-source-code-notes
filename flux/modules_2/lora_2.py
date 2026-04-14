"""
本文件核心：
低秩微调技术 (LoRA - Low-Rank Adaptation) 的底层逻辑与网络层替换实现。
FLUX 原始模型极其庞大（百亿参数），如果想教它画特定的人脸或特定的画风，普通玩家的显卡根本无法进行全量参数训练 (Full Fine-tuning)
彻底平民化了 AI 的训练门槛，它是整个开源 AI 绘画社区繁荣的基石
LoRA 的巧妙思路是：彻底冻结模型原本庞大的权重矩阵 (Base Weights)
而在原本的网络旁边“外挂”两条并行的极小单行道(A矩阵降维, B矩阵升维)
训练时只更新这个“外挂”小分支，推理时，把外挂分支计算出的微调增量叠加回原本的结果上。从而实现极低显存成本的模型微调。
"""

import torch
from torch import nn

" ========== 偷天换日：自动化替换函数 ========== "
# 作用：遍历一个庞大的神经网络模型（比如 FLUX 的 Transformer），
# 找到里面所有的普通线性层 (nn.Linear)，然后不动声色地把它们全替换成我们自定义的带有 LoRA 旁路的 `LinearLora` 层。
def replace_linear_with_lora(
    module: nn.Module,
    max_rank: int,  # LoRA 的秩 (Rank)，决定了“外挂”通道的宽度，值越大能学的信息越多，但显存开销也越大
    scale: float = 1.0, # 融合强度控制
) -> None:
    # 递归遍历模型的所有子模块
    for name, child in module.named_children():
        # 如果碰到了标准的全连接线性层
        if isinstance(child, nn.Linear):
            # 实例化我们特制的 LoRA 线性层，继承老层的输入输出维度和设备信息
            new_lora = LinearLora(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias,
                rank=max_rank,
                scale=scale,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )

            # 核心步骤：把老层原本的权重和偏置“借”过来，直接赋给新层
            # 这样网络在不经过 LoRA 旁路时，输出的还是原来的结果，画图能力没有任何损失
            new_lora.weight = child.weight
            new_lora.bias = child.bias if child.bias is not None else None

            # 执行替换：把原来的老模块踢掉，换成带 LoRA 的新模块
            setattr(module, name, new_lora)
        else:
            # 如果是个嵌套的复杂模块，就递归钻进去继续找 nn.Linear
            replace_linear_with_lora(
                module=child,
                max_rank=max_rank,
                scale=scale,
            )

" ========== LoRA 层核心实现 ========== "
# 作用：一个伪装成普通 nn.Linear 的高级线性层。
# 它的内部不仅有原始的线性计算，还藏着 A 和 B 两个极其轻量级的小矩阵。
class LinearLora(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        # 首先调用父类 (nn.Linear) 的初始化，建立基础层
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        assert isinstance(scale, float), "scale must be a float"

        self.scale = scale
        self.rank = rank
        self.lora_bias = lora_bias
        self.dtype = dtype
        self.device = device

        # 防御性编程：防止配置的 rank 超过了原本矩阵的长宽极限
        if rank > (new_rank := min(self.out_features, self.in_features)):
            self.rank = new_rank

        # 核心：建立旁路 (Bypass) 的降维矩阵 A
        # 输入维度 -> rank (极小的数值，比如 16)，起到剧烈压缩信息的作用，不带偏置
        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=self.rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # 核心：建立旁路 (Bypass) 的升维矩阵 B
        # rank -> 输出维度，把压缩的信息还原回去，这组小矩阵才是真正被反向传播“微调”的对象
        self.lora_B = nn.Linear(
            in_features=self.rank,
            out_features=out_features,
            bias=self.lora_bias,
            dtype=dtype,
            device=device,
        )

    # 动态调整 LoRA 强度权重的接口 (类似 WebUI 里的 Lora Weight 拉杆)
    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    # 前向传播：数据是如何同时流过两条通道的
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 通道 1 (主路)：走原始 FLUX 的大模型权重矩阵。
        # 这里直接调用了父类的 forward，相当于 原权重 * 输入
        base_out = super().forward(input)

        # 通道 2 (旁路)：走 LoRA 微调矩阵。
        # 先用 lora_A 极致压缩数据，再用 lora_B 还原并提炼出我们微调时新学的特征
        _lora_out_B = self.lora_B(self.lora_A(input))
        
        # 乘以调节比例，决定这个新特征在画面中的“浓度”
        lora_update = _lora_out_B * self.scale

        # 最终汇报：将原始大模型的雄厚基础能力，和我们微调出来的小范围特征，直接相加合并。
        return base_out + lora_update