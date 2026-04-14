"""
本文件核心：
负责挂载 ControlNet(Canny 线稿 / Depth 深度图)和 LoRA 微调权重
与普通文生图相比，这里多了解析参考图 (img_cond_path) 以及控制 LoRA 权重浓度 (lora_scale) 的交互逻辑。
实现对图像空间结构和画风的绝对控制
对应 flux-dev-canny、flux-dev-depth、flux-dev-canny-lora、flux-dev-depth-lora 四个模型"""
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import CannyImageEncoder, DepthImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_control, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image


@dataclass
class SamplingOptions:
    # 比普通生图多了这两个参数
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str         # 垫图的路径
    lora_scale: float | None   # LoRA 强度的滑块

# (省略重复的 parse_prompt 注释，逻辑与 cli.py 一致)
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None: ...

# 交互式指令解析器：用于在 Loop 模式下不断更换垫图
def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning image (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning image or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_cond_path = input(user_question)
        if img_cond_path.startswith("/"):
            if img_cond_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_cond_path.startswith("/h"):
                    print(f"Got invalid command '{img_cond_path}'\n{usage}")
                print(usage)
            continue

        if img_cond_path == "":
            break

        # 检查图片是否存在以及格式是否合法
        if not os.path.isfile(img_cond_path) or not img_cond_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_cond_path}' does not exist or is not a valid image file")
            continue

        options.img_cond_path = img_cond_path
        break

    return options

# 交互式指令解析器：用于在 Loop 模式下调节 LoRA 权重
def parse_lora_scale(options: SamplingOptions | None) -> tuple[SamplingOptions | None, bool]:
    changed = False

    if options is None:
        return None, changed

    user_question = "Next lora scale (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the lora scale or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/q"):
            print("Quitting")
            return None, changed
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.lora_scale = float(prompt)
        changed = True
    return options, changed


@torch.inference_mode()
def main(
    name: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    prompt: str = "a robot made out of gold",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float | None = None,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
    lora_scale: float | None = 0.85,
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    track_usage: bool = False,
    **kwargs: dict | None,
):
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if "lora" in name:
        assert not trt, "TRT does not support LORA"
    # 安全性检查：控制网必须挂载特定的针对性模型
    assert name in [
        "flux-dev-canny",
        "flux-dev-depth",
        "flux-dev-canny-lora",
        "flux-dev-depth-lora",
    ], f"Got unknown model name: {name}"

    # Canny 边缘和 Depth 深度图，它们对原本提示词的覆盖强度不一样，所以有不同的指导系数经验值
    if guidance is None:
        if name in ["flux-dev-canny", "flux-dev-canny-lora"]:
            guidance = 30.0
        elif name in ["flux-dev-depth", "flux-dev-depth-lora"]:
            guidance = 10.0
        else:
            raise NotImplementedError()

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)

    # (省略文件保存编号逻辑，同上)
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

    # 加载额外的视觉感知器官：让模型能看懂深度图或边缘线稿
    if name in ["flux-dev-depth", "flux-dev-depth-lora"]:
        img_embedder = DepthImageEncoder(torch_device)
    elif name in ["flux-dev-canny", "flux-dev-canny-lora"]:
        img_embedder = CannyImageEncoder(torch_device)
    else:
        raise NotImplementedError()

    # (省略常规硬件加载逻辑，同上)
    if not trt:
        t5 = load_t5(torch_device, max_length=512)
        clip = load_clip(torch_device)
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        ae = load_ae(name, device="cpu" if offload else torch_device)
    else:
        from flux.trt.trt_manager import ModuleName, TRTManager
        trt_ctx_manager = TRTManager(...)
        engines = trt_ctx_manager.load_engines(...)
        ae = engines[ModuleName.VAE].to(device="cpu" if offload else torch_device)
        model = engines[ModuleName.TRANSFORMER].to(device="cpu" if offload else torch_device)
        clip = engines[ModuleName.CLIP].to(torch_device)
        t5 = engines[ModuleName.T5].to(device="cpu" if offload else torch_device)

    # 给大模型打满 LoRA 药剂：遍历所有子模块，如果是 lora 层，设置融合权重
    if "lora" in name and lora_scale is not None:
        for _, module in model.named_modules():
            if hasattr(module, "set_scale"):
                module.set_scale(lora_scale)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt, width=width, height=height, num_steps=num_steps, guidance=guidance,
        seed=seed, img_cond_path=img_cond_path, lora_scale=lora_scale,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)
        if "lora" in name:
            opts, changed = parse_lora_scale(opts)
            if changed:
                # 用户在交互界面调了 LoRA 浓度，立刻生效到网络层里
                for _, module in model.named_modules():
                    if hasattr(module, "set_scale"):
                        module.set_scale(opts.lora_scale)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        x = get_noise(1, opts.height, opts.width, device=torch_device, dtype=torch.bfloat16, seed=opts.seed)
        opts.seed = None
        
        if offload:
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
            
        # 核心差异：调用 prepare_control，它不仅解析文字，还会把参考图变成特征强塞进去
        inp = prepare_control(
            t5, clip, x, prompt=opts.prompt, ae=ae, encoder=img_embedder, img_cond_path=opts.img_cond_path,
        )
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        if offload:
            t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        idx = save_image(
            nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
        )

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
            opts = parse_img_cond_path(opts)
            if "lora" in name:
                opts, changed = parse_lora_scale(opts)
                if changed:
                    for _, module in model.named_modules():
                        if hasattr(module, "set_scale"):
                            module.set_scale(opts.lora_scale)
        else:
            opts = None

    if trt:
        trt_ctx_manager.stop_runtime()

if __name__ == "__main__":
    Fire(main)