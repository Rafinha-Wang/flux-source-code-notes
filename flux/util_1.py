" 本文件核心 """"
这个 util_1.py 文件是整个 FLUX 项目的工具箱, 里面装满了各种各样的函数和配置, 负责处理模型下载、加载、图像保存、商用追踪、烙AI水印等琐碎但地基性质的任务
还有很大一部分是一个庞大的模型配置字典, 里面详细记录了每个版本 FLUX 模型的设计参数和它们对应的 Hugging Face 仓库地址
它就像是一个全能的瑞士军刀, 虽然不直接参与图像生成的核心算法, 但却在背后默默地支持着整个系统的顺利运行
"""

import getpass
import math
import os
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download, login
from imwatermark import WatermarkEncoder
from PIL import ExifTags, Image
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxLoraWrapper, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder

#创建基础文件夹 checkpoints
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)
#获取 API 密钥
BFL_API_KEY = os.getenv("BFL_API_KEY")

#设置 TensorRT 环境变量            #创建 TensorRT 子文件夹
os.environ.setdefault("TRT_ENGINE_DIR", str(CHECKPOINTS_DIR / "trt_engines"))
(CHECKPOINTS_DIR / "trt_engines").mkdir(exist_ok=True)


"--------hf前科函数--------"
#确保当前环境已经成功登录（鉴权）了 Hugging Face (HF) 平台:
def ensure_hf_auth():
    #试图通过环境变量获取 Token
    hf_token = os.environ.get("HF_TOKEN")
    #试图使用环境变量进行登录
    if hf_token:
        print("Trying to authenticate to HuggingFace with the HF_TOKEN environment variable.")
        try:
            login(token=hf_token)
            print("Successfully authenticated with HuggingFace using HF_TOKEN")
            return True
        except Exception as e:
            print(f"Warning: Failed to authenticate with HF_TOKEN: {e}")
     #如果环境变量登录失败，继续往下走正常的登录流程（可能是之前已经登录过了，或者用户需要手动登录）
     #检查本地缓存的 Token 文件，判断是否已经登录过 Hugging Face,
     #当你曾在终端/命令行里运行过 huggingface-cli login 命令并成功登录后，Hugging Face 会把你的 Token 悄悄保存在这个默认路径下
    if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        print("Already authenticated with HuggingFace")
        return True
    return False


"--------hf处女函数--------"
#通过终端交互的方式，手动提示用户输入 Hugging Face 的 Token 进行登录
#是ensure_hf_auth()的备胎
def prompt_for_hf_auth():
    try:
        #用 Python 内置的 getpass 模块来接收用户的输入  .strip() 用于去掉多复制的空格或换行符
        #不用普通的 input(): getpass 让你输入或粘贴的内容在屏幕上不可见, 防窥
        token = getpass.getpass("HF Token (hidden input): ").strip()
        #检查输入是否为空:
        if not token:
            print("No token provided. Aborting.")
            return False
         #执行登录操作:
        login(token=token)
        print("Successfully authenticated!")
        return True
    
     #Ctrl+C 主动取消:
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user.")
        return False
     #登录失败:
    except Exception as auth_e:
        print(f"Authentication failed: {auth_e}")
        print("Tip: You can also run 'huggingface-cli login' or set HF_TOKEN environment variable")
        return False


" ==========hf核心函数========== "
#利用前两个函数，完成整个模型下载和加载流程的核心调度, 负责协调检查本地文件、下载权重、加载模型等
def get_checkpoint_path(repo_id: str, filename: str, env_var: str) -> Path:
    #优先使用环境变量指定的路径, 避免了重复下载浪费时间和硬盘空间
    if os.environ.get(env_var) is not None:
        local_path = os.environ[env_var]
        if os.path.exists(local_path):
            return Path(local_path)

        print(
            f"Trying to load model {repo_id}, {filename} from environment "
            f"variable {env_var}. But file {local_path} does not exist. "
            "Falling back to default location."
        )

     #构建安全的默认本地存储路径, 将hf的repo_id转换成一个安全的文件夹名称
    safe_repo_name = repo_id.replace("/", "_")
    checkpoint_dir = CHECKPOINTS_DIR / safe_repo_name
    checkpoint_dir.mkdir(exist_ok=True)

    local_path = checkpoint_dir / filename
     #判断是否需要下载
    if not local_path.exists():
        print(f"Downloading {filename} from {repo_id} to {local_path}")
        #试图下载:
        try:
            ensure_hf_auth()  #检查前科
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
        #例外: 处理hf权限问题 (Gated Repo)
        except Exception as e:
            if "gated repo" in str(e).lower() or "restricted" in str(e).lower():
                print(f"\nError: Cannot access {repo_id} -- this is a gated repository.")

                #试图登录进行身份验证 authenticate
                if prompt_for_hf_auth():
                    #再次试图下载
                    print("Retrying download...")
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
                else:
                    print("Authentication failed or cancelled.")
                    print("You can also run 'huggingface-cli login' or set HF_TOKEN environment variable")
                    raise RuntimeError(f"Authentication required for {repo_id}")
            else:
                raise e
    return local_path



" ========== 配置和寻址-工具人函数 ========== "
# 输入:模型名称和Transformer模块的精度要求（默认为bf16）
#返回值： 返回下载保存的目录路径（字符串），如果不需要转换则返回 None
def download_onnx_models_for_trt(model_name: str, trt_transformer_precision: str = "bf16") -> str | None:
    #建立仓库映射字典表，将模型名称映射到对应的 Hugging Face 仓库 ID:
    onnx_repo_map = {
        "flux-dev": "black-forest-labs/FLUX.1-dev-onnx",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell-onnx",
        "flux-dev-canny": "black-forest-labs/FLUX.1-Canny-dev-onnx",
        "flux-dev-depth": "black-forest-labs/FLUX.1-Depth-dev-onnx",
        "flux-dev-redux": "black-forest-labs/FLUX.1-Redux-dev-onnx",
        "flux-dev-fill": "black-forest-labs/FLUX.1-Fill-dev-onnx",
        "flux-dev-kontext": "black-forest-labs/FLUX.1-Kontext-dev-onnx",
    }
     #过滤不支持的模型，如果模型名称不在映射表中，则返回 None，表示不需要下载 ONNX 模型
    if model_name not in onnx_repo_map:
        return None  # No ONNX repository required for this model
    #构建安全的本地存储路径, 获取真实的仓库 ID，并将其转换为本地文件夹路径。
    repo_id = onnx_repo_map[model_name]
    safe_repo_name = repo_id.replace("/", "_")
    onnx_dir = CHECKPOINTS_DIR / safe_repo_name

    # 构建精确的组件下载清单
    onnx_file_map = {
        "clip": "clip.opt/model.onnx",
        "transformer": f"transformer.opt/{trt_transformer_precision}/model.onnx",
        "transformer_data": f"transformer.opt/{trt_transformer_precision}/backbone.onnx_data",
        "t5": "t5.opt/model.onnx",
        "t5_data": "t5.opt/backbone.onnx_data",
        "vae": "vae.opt/model.onnx",
    }
    #clip 和 t5：用于理解用户输入的文本提示词（Text Encoders）
    #vae：用于图像潜空间和像素空间转换的编解码器
    #transformer：真正负责核心图像生成的扩散模型骨干

    # 遍历之前整理好的清单 onnx_file_map, 如果本地已经存在所有需要的ONNX文件，就不再执行任何下载，直接打包路径并返回
    if onnx_dir.exists():
        all_files_exist = True
        custom_paths = []
        for module, onnx_file in onnx_file_map.items():
            if module.endswith("_data"):
                continue  # 跳过 _data 文件，因为它们不是 TRT 需要的实际模型文件
            local_path = onnx_dir / onnx_file
            if not local_path.exists():
                all_files_exist = False
                break
            custom_paths.append(f"{module}:{local_path}")
        if all_files_exist:
            print(f"ONNX models ready in {onnx_dir}")
            return ",".join(custom_paths)

    #如果并非所有文件都已存在，请将其下载到本地，并处理可能出现的权限问题
    print(f"Downloading ONNX models from {repo_id} to {onnx_dir}")
    print(f"Using transformer precision: {trt_transformer_precision}")
    onnx_dir.mkdir(exist_ok=True)

    #下载所有 ONNX 文件
    for module, onnx_file in onnx_file_map.items():
        local_path = onnx_dir / onnx_file
        if local_path.exists():
            continue  #已下载完毕
        #创建父目录, 防止下载写入时报“找不到路径”的错误
        local_path.parent.mkdir(parents=True, exist_ok=True)

        #执行实际的 hf_hub_download 下载动作，并处理可能的报错
        try:
            print(f"Downloading {onnx_file}")
            hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=onnx_dir)
        #如果报的是 does not exist（文件不存在），代码选择 continue 忽略, 不影响大局
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                continue
            #用户必须亲自打开浏览器，登录网页，阅读一长串法律协议，然后点击“I Accept（我同意）”按钮，才能获得访问权限
            elif "gated repo" in str(e).lower() or "restricted" in str(e).lower():
                print(f"Cannot access {repo_id} - requires license acceptance")
                print("Please follow these steps:")
                print(f"   1. Visit: https://huggingface.co/{repo_id}")
                print("   2. Log in to your HuggingFace account")
                print("   3. Accept the license terms and conditions")
                print("   4. Then retry this command")
                raise RuntimeError(f"License acceptance required for {model_name}")
            else:
                #抛锚: 重新处理其他错误
                raise
    print(f"ONNX models ready in {onnx_dir}")

    #基本重复164行-175行的逻辑，确保返回的路径是最新的（即使之前已经下载过了部分文件）
    custom_paths = []
    for module, onnx_file in onnx_file_map.items():
        if module.endswith("_data"):
            continue  #跳过返回路径中的数据文件
        full_path = onnx_dir / onnx_file
        if full_path.exists():
            custom_paths.append(f"{module}:{full_path}")
    return ",".join(custom_paths)
    #返回 TRT 期望的自定义 ONNX 路径格式：“模块1:路径1,模块2:路径2”
    #注意：只返回实际的模块路径，不要返回数据文件

" ========= 前台接待函数 ========== "
def check_onnx_access_for_trt(model_name: str, trt_transformer_precision: str = "bf16") -> str | None:
    """检查 ONNX 访问权限并下载 TRT 模型 返回 ONNX 目录路径"""
    return download_onnx_models_for_trt(model_name, trt_transformer_precision)


" ========== 防君子更防小人-商用函数 ========= "
def track_usage_via_api(name: str, n=1) -> None:
    """
    Track usage of licensed models via the BFL API for commercial licensing compliance.
    For more information on licensing BFL's models for commercial use and usage reporting,
    see the README.md or visit: https://dashboard.bfl.ai/licensing/subscriptions?showInstructions=true
    """
    #断言强校验: 如果没有配置 API Key，程序会瞬间崩溃并抛出红字大错, 别想白嫖商用
    assert BFL_API_KEY is not None, "BFL_API_KEY is not set"
    #建立了一个内部小名到官方计费代号（slug）的映射表:
    model_slug_map = {
        "flux-dev": "flux-1-dev",
        "flux-dev-kontext": "flux-1-kontext-dev",
        "flux-dev-fill": "flux-tools",
        "flux-dev-depth": "flux-tools",
        "flux-dev-canny": "flux-tools",
        "flux-dev-canny-lora": "flux-tools",
        "flux-dev-depth-lora": "flux-tools",
        "flux-dev-redux": "flux-tools",
        "flux-dev-krea": "flux-1-krea-dev",
    }
    #免费模型不报错, “跳过计费追踪”
    if name not in model_slug_map:
        print(f"Skipping tracking usage for {name}, as it cannot be tracked. Please check the model name.")
        return

    #发送“计费电报”
    model_slug = model_slug_map[name]
    url = f"https://api.bfl.ai/v1/licenses/models/{model_slug}/usage"
    headers = {"x-key": BFL_API_KEY, "Content-Type": "application/json"}
    payload = {"number_of_generations": n}
    # 核实汇报结果
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to track usage: {response.status_code} {response.text}")
    else:
        print(f"Successfully tracked usage for {name} with {n} generations")


" ========== 出厂质检函数 ========== "
#把数学矩阵冲洗成真正的图片、打上不可见的官方钢印、进行色情暴力审查、贴上产品说明书，最后存入你的硬盘
def save_image(
    nsfw_classifier,
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
    track_usage: bool = False,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # 把AI生成的Tensor转换为 PIL 格式并保存:
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float()) #强制插入了一步不可见的ai隐形水印
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # NSFW鉴黄:
    if nsfw_classifier is not None:
        nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    else:
        nsfw_score = nsfw_threshold - 1.0
    #合格品放行: ( 同时贴出厂标签（EXIF）)
    if nsfw_score < nsfw_threshold:
        exif_data = Image.Exif()
        if name in ["flux-dev", "flux-schnell"]:
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        else:
            exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        if track_usage:
            track_usage_via_api(name, 1)
        idx += 1
    #违规品扣押：直接销毁:
    else:
        print("Your generated image may contain NSFW content.")
    return idx


@dataclass # 表明：“这个类不负责执行复杂的动作，它纯粹就是一个装数据的容器”
class ModelSpec:
    params: FluxParams  # FLUX 核心 Transformer 模型的设计图纸
    ae_params: AutoEncoderParams  # VAE（图像自动编码器）的参数
    #寻找“货源”的地址（Hugging Face 仓库）:
    repo_id: str
    repo_flow: str
    repo_ae: str
    #加载LoRA:
    lora_repo_id: str | None = None
    lora_filename: str | None = None
    #None = None - 这代表这两个参数是可选的


" ========== 模型档案柜 ========== "
#名为 configs 的“档案柜”，里面存放着不同版本 FLUX 模型的神经网络架构的超参数（Hyper-parameters）
#在此给第一个模型数据稍作阐释, 后续略了
configs = {
    "flux-dev": ModelSpec(
        #权重文件定位:
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        #正经 Transformer 参数 (FluxParams):
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,   # CLIP 文本编码器的维度
            context_in_dim=4096,   # T5 文本编码器的维度
            hidden_size=3072,   # 核心神经元的宽度（极度粗壮，吃显存的大头）
            mlp_ratio=4.0,
            num_heads=24,    # 24个注意力头（多线思考能力）
            depth=19,   # 19层双流 Transformer 块
            depth_single_blocks=38,   # 38层单流 Transformer 块
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,   # 是否支持提示词引导系数（CFG/Guidance Scale）
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,   # 输入是 RGB 3通道
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,   # 压缩后的潜空间是 16 通道
            #两个死常数:
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-krea": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        repo_flow="flux1-krea-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Canny-dev-lora",
        lora_filename="flux1-canny-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Depth-dev-lora",
        lora_filename="flux1-depth-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-redux": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        repo_flow="flux1-redux-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}
#终于，归档了不同版本的 FLUX 模型的设计图纸（超参数）和它们对应的 Hugging Face 仓库地址
#后续的加载函数会根据这个配置来下载和构建模型


" ========== 黄金尺寸表 ========== "
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]
#无论长宽比例怎么变，它们相乘的总像素数都死死地卡在 1,050,000 左右（ 100 万像素，1 Megapixel）
#这个列表确保了无论你画多宽或多高的图，AI 的“脑力负荷”始终保持在最完美的舒适区，既不会过载也不会闲置


" =========== 绣花匠抡大锤 - 画布裁剪函数 ========== "
def aspect_ratio_to_height_width(aspect_ratio: str, area: int = 1024**2) -> tuple[int, int]:
    #提取比例:
    width = float(aspect_ratio.split(":")[0])
    height = float(aspect_ratio.split(":")[1])
    ratio = width / height
    width = round(math.sqrt(area * ratio))
    height = round(math.sqrt(area / ratio))
    return 16 * (width // 16), 16 * (height // 16)
    #强制变 16 的倍数，因为大多数 AI 模型都喜欢 16 的倍数作为输入尺寸


" ========== 捉奸函数 ========== "
#通过合理的换行、缩进和分割线，让开发者能一眼看清到底是哪些具体的层没有加载成功，从而快速定位是代码写错了，还是模型文件下错了
#输入: 模型加载过程中遇到的缺失键和意外键列表
def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        #一行最多显示 80 个字符, 打 79 个减号，刚好能铺满一整行又不会触发自动换行
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


" ========== flux 模型装配函数 ========== "
def load_flow_model(name: str, device: str | torch.device = "cuda", verbose: bool = True) -> Flux:
    # 加载 Flux
    print("Init model")
    #根据传入的模型名称，从全局配置字典 configs 中查找到对应的模型配置, 获取模型权重文件（Checkpoint）的本地或下载路径:
    config = configs[name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))
    #使用 Meta 设备初始化“空壳”模型:
    #在 meta 设备下实例化模型，PyTorch 只会记录模型的架构、层的形状（Shape），但不会为它们分配真实的内存或显存
    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
        else:
            model = Flux(config.params).to(torch.bfloat16)
    #加载基础模型权重并填充到“空壳”中:
    print(f"Loading checkpoint: {ckpt_path}")
    # load_sft 不支持 torch.device
    sd = load_sft(ckpt_path, device=str(device))  # 读取权重文件
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)

    #挂载 LoRA 权重（如果配置存在）:
    if config.lora_repo_id is not None and config.lora_filename is not None:
        print("Loading LoRA")
        lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
        lora_sd = load_sft(lora_path, device=str(device))
        # 加载 Lora 参数 + 重置规范中的比例值
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model


" ========== 在 Flux 架构中的双剑合璧 (T5 + CLIP) ========== "
#加载 T5 文本编码器（Text Encoder）:
def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # 最大长度为 64、128、256 和 512 均应适用（前提是您的序列长度足够短）
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)
    #显存救星：强制使用 bfloat16, 显存占用直接砍半，同时几乎不会影响它理解文本的能力
#加载 CLIP 文本编码器 :
def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


#加载 Flux 模型生态中的 AutoEncoder（自动编码器，在图像生成领域通常被称为 VAE）:
def load_ae(name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    # config 准备参数与路径:
    config = configs[name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE"))
    # 加载 autoencoder:
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)
    print(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    #调用捉奸函数, 发出报错提醒, 快速定位错误:
    print_load_warning(missing, unexpected)
    return ae

" ========== Zero-initialization - 备份存档函数 ========== "
#这个填 0 扩展的机制，开发者可以冻结（Freeze）或保留那些原来不为 0 的参数，只针对那些填了 0 的新参数进行反向传播训练（梯度更新）
#能让模型在不丢失原有画图能力的前提下，学会新加入的特定功能  即选择性扩展(模型的功能)的含义
def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """可选择性地扩展状态字典，使其与模型的参数形状相匹配。"""
    #遍历并寻找需要干预的参数
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                #创建一个大号零矩阵:     #device=...：确保这个全是 0 的新矩阵和旧权重在同一个设备上
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device) 
                #将旧数据“无缝嵌入”到新矩阵中: 通过切片操作，旧权重被放置在新矩阵的左上角（或前面），而新增的部分则保持为零
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight
    return state_dict


" ========== 水印函数 =========="
#通过类, 给不同模型生成的图像打上 同一套不可见的数字ai水印, 以证明它们都是由这个模型生成的, 也可以用来追踪泄密和盗版
class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """为输入图像添加预定义的水印   参数 Args:image: ([N,] B, RGB, H, W) in range [-1, 1]
        Returns:与输入相同，但带有水印"""
        #预处理：调整数值范围和维度:
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        #核心转换：PyTorch Tensor 转为 OpenCV 格式:
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # 注入水印: 对每张图像进行循环，调用水印编码器的 encode 方法将水印嵌入到图像中
        # 水印库期望输入格式为 cv2 BGR 格式
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        #原路返回，变回 PyTorch 格式:
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# 一个固定长度的 48 位消息，系随机选定
# flux官方签名:
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] 将 x 的位转换为字符串形式，使用 int 将其转换为 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
#实例化类中的水印嵌入器对象，以便后续调用:
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
