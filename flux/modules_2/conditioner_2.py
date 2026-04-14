"""
本文件核心：
文本编码器封装模块 (Conditioner)。
由于 FLUX 同时使用了 CLIP (擅长图文对齐) 和 T5 (擅长复杂逻辑和长难句) 作为文本编码器，
这个文件提供了一个统一的包装类 HFEmbedder, 用于自动下载、加载和推理这两种截然不同的模型结构。
将人类的自然语言提示词(Prompt)翻译成神经网络能听懂的数学语言 (高维词向量 Embeddings)
"""

from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

class HFEmbedder(nn.Module):
    # 初始化：根据传入的 version 名字，自动判断该挂载什么模型
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        # 通过判断名字里有没有 "openai" 来区分是不是 CLIP
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        # CLIP 输出的是池化后的单维向量(pooler_output)，T5 输出的是整个序列的隐藏状态(last_hidden_state)
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 分别实例化对应的分词器 (Tokenizer) 和模型主体 (TextModel)
        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        # 文本编码器只做推理，不参与画图时的权重更新，因此关闭梯度以节省巨量显存
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    # 前向传播：将字符串翻译成向量
    def forward(self, text: list[str]) -> Tensor:
        # 1. 查字典：把文字切割成 Token ID 数组，并填充或截断到指定的最大长度
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # 2. 过模型：将 Token ID 送入模型，提取深层语义特征
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        
        # 返回指定格式的张量，并强制转换为 bfloat16 精度（节省一半显存）
        return outputs[self.output_key].bfloat16()