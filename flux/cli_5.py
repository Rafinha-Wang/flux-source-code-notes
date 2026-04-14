"""
本文件核心：
标准文生图模式,负责最基础的 Text-to-Image (文生图) 任务
对应 flux-dev 和 flux-schnell 模型
负责串联起加载模型、解析用户指令、执行 Sampling 管线、以及将结果保存到硬盘的整个全生命周期。
特别注意: 它实现了基于终端的交互式会话 (loop 模式), 以及极端压榨显存的 Offload 机制。
"""

import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    check_onnx_access_for_trt,
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)

NSFW_THRESHOLD = 0.85

# 记录生成参数的数据结构
@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

# 交互式指令解析器：当开启 loop 模式时，它会让终端变成一个聊天框，允许你用斜杠命令 (如 /w 1024) 动态修改参数
# 接收一个 SamplingOptions 数据类实例(包含了当前所有的生图参数，如宽高、步数、提示词等)，并返回一个修改后的 SamplingOptions 实例或者 None (如果用户选择退出)
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    # 循环读取终端输入，只要以 '/' 开头就被视为系统命令而不是画图提示词
    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16) # 强制对齐 16 倍数
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        # 输入 /q 时，直接返回 None。主函数 main() 里的 while opts is not None: 
        # 循环收到这个信号后就会彻底终止，进而关闭 TensorRT 引擎或释放 PyTorch 显存
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    # 状态保留：如果你什么都不敲，直接按了回车键（prompt == ""），这段代码会直接跳过赋值
    # 意味着 options.prompt 会原封不动地保留上一轮的提示词
    if prompt != "":
        options.prompt = prompt
    #配合前面用 /s 修改种子的指令，这非常适合用来做“固定提示词，疯狂抽卡刷种子”的自动化测试操作
    return options

# 主控入口函数
@torch.inference_mode()
def main(
    name: str = "flux-dev-krea",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,       # 显存救星：是否将不计算的模型踢回内存
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    trt: bool = False,           # 速度狂魔：是否启用 Nvidia 的 TensorRT 加速引擎
    trt_transformer_precision: str = "bf16",
    track_usage: bool = False,
):
    """Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: 要加载的模型的名称
        height: 样品的像素高度(应为16的倍数)
        width: 样品的像素宽度(应为16的倍数)
        seed: 设置采样的随机种子
        output_name: 保存输出图像的路径，`{idx}` 将被样本的索引替换
        prompt: 用于采样的提示词
        device: Pytorch device
        num_steps: 采样步骤的个数 (default 4 for schnell, 50 for guidance distilled)
        loop: 启动交互式会话并多次进行样本采集
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: 将提示信息添加至图像的Exif元数据中
        trt: 使用TensorRT后端进行优化推理
        trt_transformer_precision: 指定推理中变换器的精确度
        track_usage: 用于许可目的跟踪模型的使用情况
        """

    # 支持用 '|' 分割多个 prompt 进行批量连续生成
    prompt = prompt.split("|")
    if len(prompt) == 1:
        prompt = prompt[0]
        additional_prompts = None
    else:
        additional_prompts = prompt[1:]
        prompt = prompt[0]

    assert not (
        (additional_prompts is not None) and loop
    ), "Do not provide additional prompts and set loop to True"

    # 加载鉴黄模型
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    # dev 版本默认跑 50 步，schnell (蒸馏提速版) 只需要跑 4 步
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    height = 16 * (height // 16)
    width = 16 * (width // 16)

    # 智能创建输出目录和确定下一张图的序号 (避免覆盖老图片)
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # --- 硬件加载分支 ---
    if not trt:
        # 常规 PyTorch 加载模式
        # 如果开启了 offload，初始状态全放进 cpu 内存，只有计算时才搬去显卡
        t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
        clip = load_clip(torch_device)
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        ae = load_ae(name, device="cpu" if offload else torch_device)
    else:
        # TensorRT 极限推理模式：需要将模型编译成针对特定显卡架构优化的引擎文件 (Engine)
        from flux.trt.trt_manager import ModuleName, TRTManager

        onnx_dir = check_onnx_access_for_trt(name, trt_transformer_precision)

        trt_ctx_manager = TRTManager(
            trt_transformer_precision=trt_transformer_precision,
            trt_t5_precision=os.getenv("TRT_T5_PRECISION", "bf16"),
        )
        engines = trt_ctx_manager.load_engines(...) # 省略大量传参

        ae = engines[ModuleName.VAE].to(device="cpu" if offload else torch_device)
        model = engines[ModuleName.TRANSFORMER].to(device="cpu" if offload else torch_device)
        clip = engines[ModuleName.CLIP].to(torch_device)
        t5 = engines[ModuleName.T5].to(device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt, width=width, height=height, num_steps=num_steps, guidance=guidance, seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    # --- 核心推理生命周期 ---
    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # 1. 初始化纯噪声画布
        x = get_noise(
            1, opts.height, opts.width, device=torch_device, dtype=torch.bfloat16, seed=opts.seed,
        )
        opts.seed = None
        
        # 显存微操 1：把不需要用的 VAE (ae) 踢回 CPU，把需要读懂文字的 T5 和 CLIP 请上显卡
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache() # 清理显存碎片
            t5, clip = t5.to(torch_device), clip.to(torch_device)
            
        # 2. 准备所有管线数据
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # 显存微操 2：文本读完了，把 T5 和 CLIP 踢回 CPU，把负责画图的主干大脑 (model) 请上显卡
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # 3. 核心大循环去噪
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # 显存微操 3：图画完了，主大脑 (model) 踢回 CPU，把负责解压变回彩色图片的 VAE (ae) 请上显卡
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # 4. 图像解压
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize() # 确保显卡完全算完再计秒
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

        # 5. 鉴黄与存盘
        idx = save_image(
            nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
        )

        # 处理多步连拍
        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        elif additional_prompts:
            next_prompt = additional_prompts.pop(0)
            opts.prompt = next_prompt
        else:
            opts = None

    if trt:
        trt_ctx_manager.stop_runtime()

if __name__ == "__main__":
    Fire(main)