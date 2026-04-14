"""
本文件核心：
风格迁移与图像提示词 (Redux / Image Prompt) 的终端实现。
Redux 的特点是它不要求你提供线稿或者黑白遮罩图，它会直接阅读你提供的一张“风格参考图”，
然后“模仿”那种感觉、配色或者主体结构，结合你的文字画出全新的图片。
"""
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_redux, unpack
from flux.util import (
    get_checkpoint_path,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str

# (省略与基础文生图完全一致的解析器注释)
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None: ...
def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None: ...


@torch.inference_mode()
def main(
    name: str = "flux-dev",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
    track_usage: bool = False,
):
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    # Redux 属于标准模型的附加功能，因此可以在 dev 和 schnell 上跑
    if name not in (available := ["flux-dev", "flux-schnell"]):
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

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

    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # 核心差异 1：额外下载并挂载强大的 Redux 图像编码器 (本质上是一层连接着 Google SigLIP 模型特征的映射层)
    redux_path = str(
        get_checkpoint_path("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors", "FLUX_REDUX")
    )
    img_embedder = ReduxImageEncoder(torch_device, redux_path=redux_path)

    rng = torch.Generator(device="cpu")
    prompt = "" # Redux 模式下，允许甚至鼓励不写任何文本 prompt，全靠图像特征撑场面
    opts = SamplingOptions(
        prompt=prompt, width=width, height=height, num_steps=num_steps, guidance=guidance,
        seed=seed, img_cond_path=img_cond_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        x = get_noise(1, opts.height, opts.width, device=torch_device, dtype=torch.bfloat16, seed=opts.seed)
        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
            
        # 核心差异 2：调用 prepare_redux 进行打包。它内部会把读取出来的图片特征，拼接在 T5 提取出来的文本特征末尾。
        inp = prepare_redux(
            t5, clip, x, prompt=opts.prompt, encoder=img_embedder, img_cond_path=opts.img_cond_path,
        )
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        if offload:
            t5, clip = t5.cpu(), clip.cpu()
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
        else:
            opts = None

if __name__ == "__main__":
    Fire(main)