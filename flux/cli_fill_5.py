"""
本文件核心：
局部重绘 (Inpainting/Fill) 的终端实现。专门调用 flux-dev-fill 模型
与前面的生图不同，这里不仅需要“原图”作为垫底，还需要一张“黑白遮罩图 (Mask)”，
它会告诉 AI: “黑色区域保留原样，白色区域请你根据我的提示词瞎编重画”
可以理解为局部重绘/消除模式
"""
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from PIL import Image
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str   # 原图路径
    img_mask_path: str   # 遮罩图路径

# (省略 parse_prompt 和 parse_img_cond_path，逻辑同上，只为 Loop 模式服务)
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None: ...
def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None: ...

# 交互式指令解析器：专门用来解析遮罩图 (Mask) 的路径
def parse_img_mask_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning mask (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning mask or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_mask_path = input(user_question)

        if img_mask_path.startswith("/"):
            if img_mask_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_mask_path.startswith("/h"):
                    print(f"Got invalid command '{img_mask_path}'\n{usage}")
                print(usage)
            continue

        if img_mask_path == "":
            break

        if not os.path.isfile(img_mask_path) or not img_mask_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_mask_path}' does not exist or is not a valid image file")
            continue
        else:
            with Image.open(img_mask_path) as img:
                width, height = img.size

            # 严格对齐校验：Vae 会以 32 倍率对图像切块，所以像素宽高必须被 32 整除
            if width % 32 != 0 or height % 32 != 0:
                print(f"Image dimensions must be divisible by 32, got {width}x{height}")
                continue
            else:
                with Image.open(options.img_cond_path) as img_cond:
                    img_cond_width, img_cond_height = img_cond.size

                # 严格对齐校验：遮罩的长宽必须和原图完全一比一匹配
                if width != img_cond_width or height != img_cond_height:
                    print(
                        f"Mask dimensions must match conditioning image, got {width}x{height} and {img_cond_width}x{img_cond_height}"
                    )
                    continue

        options.img_mask_path = img_mask_path
        break

    return options


@torch.inference_mode()
def main(
    seed: int | None = None,
    prompt: str = "a white paper cup",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float = 30.0,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
    img_mask_path: str = "assets/cup_mask.png",
    track_usage: bool = False,
):
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    # 强制挂载专用的 fill (局部重绘) 模型权重
    name = "flux-dev-fill"
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

    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    # 因为要填补原图，所以生成的画布长宽必须读取自原图的长宽
    with Image.open(img_cond_path) as img:
        width, height = img.size
        
    opts = SamplingOptions(
        prompt=prompt, width=width, height=height, num_steps=num_steps, guidance=guidance,
        seed=seed, img_cond_path=img_cond_path, img_mask_path=img_mask_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)
        with Image.open(opts.img_cond_path) as img:
            width, height = img.size
        opts.height = height
        opts.width = width
        opts = parse_img_mask_path(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        x = get_noise(1, opts.height, opts.width, device=torch_device, dtype=torch.bfloat16, seed=opts.seed)
        opts.seed = None
        if offload:
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
            
        # 核心差异：调用 prepare_fill，将原图信息、涂黑的遮罩信息、文本需求 一起融合进管线
        inp = prepare_fill(
            t5, clip, x, prompt=opts.prompt, ae=ae, img_cond_path=opts.img_cond_path, mask_path=opts.img_mask_path,
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

            with Image.open(opts.img_cond_path) as img:
                width, height = img.size
            opts.height = height
            opts.width = width

            opts = parse_img_mask_path(opts)
        else:
            opts = None

if __name__ == "__main__":
    Fire(main)