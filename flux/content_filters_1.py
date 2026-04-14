"""
本文件核心：
FLUX 项目的内容安全与合规过滤模块
该模块集成了多模态大模型 (Pixtral) 和专用的 NSFW (Not Safe For Work) 图像分类器
主要功能是在生成管线中, 对用户输入的文本提示词(Prompt)以及最终生成的图像进行安全审查,
防止系统生成涉黄、侵犯版权（如受保护的 IP 或商标）或不当描绘真实公众人物的内容
"""

import torch
from einops import rearrange
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline

" ========== 图像合规检测系统提示词 ========== "
# 用于指导多模态大模型 (Pixtral) 审查生成图像的 Prompt
PROMPT_IMAGE_INTEGRITY = """
Task: Analyze an image to identify potential copyright concerns or depictions of public figures.
Output: Respond with only "yes" or "no"

Criteria for "yes":
- The image contains a recognizable character from copyrighted media (movies, TV, comics, games, etc.)
- The image displays a trademarked logo or brand
- The image depicts a recognizable public figure (celebrities, politicians, athletes, influencers, historical figures, etc.)

Criteria for "no":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person depicted
2. General references to demographics or characteristics are not sufficient
3. Base your decision solely on visual content, not interpretation
4. Provide only the one-word answer: "yes" or "no"
""".strip()
#翻译:
# 任务：分析图像，识别潜在的版权问题或对公众人物的描绘。
# 输出：仅回答 "yes" 或 "no"

# 判定为违规的标准:
# - 图像包含受版权保护媒体中可识别的角色，如电影、电视、漫画、游戏等
# - 图像展示了受商标保护的标识或品牌
# - 图像描绘了可识别的公众人物，如名人、政客、运动员、网红、历史人物等

# 判定为安全的标准:
# - 所有其他情况
# - 无法具体识别出版权作品或个人身份时

# 核心要求:
# 1. 必须能够确切指出被描绘的版权作品或具体人物
# 2. 仅具备普遍的人口统计特征或外貌特征不足以构成违规
# 3. 仅基于视觉内容做出判定，不要过度推测
# 4. 仅提供单字回答: yes 或 no


PROMPT_IMAGE_INTEGRITY_FOLLOW_UP = "Does this image have copyright concerns or includes public figures?"
# 这张图片是否存在版权问题，或者是否包含公众人物？

PROMPT_TEXT_INTEGRITY = """
Task: Analyze a text prompt to identify potential copyright concerns or requests to depict living public figures.

Output: Respond with only "yes" or "no"

Criteria for "Yes":
- The prompt explicitly names a character from copyrighted media (movies, TV, comics, games, etc.)
- The prompt explicitly mentions a trademarked logo or brand
- The prompt names or describes a specific living public figure (celebrities, politicians, athletes, influencers, etc.)

Criteria for "No":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person referenced
2. General demographic descriptions or characteristics are not sufficient
3. Analyze only the prompt text, not potential image outcomes
4. Provide only the one-word answer: "yes" or "no"

The prompt to check is:
-----
{prompt}
-----

Does this prompt have copyright concerns or includes public figures?
""".strip()
#翻译:
# 任务：分析文本提示词，识别潜在的版权问题或描绘在世公众人物的请求。
# 输出：仅回答 "yes" 或 "no"
# 判定为违规的标准:
# - 提示词明确指出了受版权保护媒体中的角色（如电影、电视、漫画、游戏等）
# - 提示词明确提及了受商标保护的标识或品牌
# - 提示词点名或描绘了特定的在世公众人物（如名人、政客、运动员、网红等）
# 判定为安全的标准:
# - 所有其他情况
# - 无法具体识别出版权作品或个人身份时
# 核心要求:
# 1. 必须能够确切指出被提及的版权作品或具体人物
# 2. 仅提供普遍的人口统计特征或外貌描述不足以构成违规
# 3. 仅分析提示词文本本身，不预测可能生成的图像结果
# 4. 仅提供单字回答：yes 或 no
# 待检查的提示词为：
# {prompt}
# 该提示词是否存在版权问题或包含公众人物？


#主要功能是对图像进行两阶段的内容安全审核（过滤违规/敏感图像）:
class PixtralContentFilter(torch.nn.Module):

    #搭建一个双层架构的图像内容安全审核（特别是 NSFW 过滤）系统:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        nsfw_threshold: float = 0.85,
    ):
        super().__init__()
        #加载多模态大模型
        model_id = "mistral-community/pixtral-12b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=device)
        self.yes_token, self.no_token = self.processor.tokenizer.encode(["yes", "no"])
        #加载轻量级 NSFW 分类器, 并设定了拦截阈值 默认 0.85
        self.nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=device
        )
        self.nsfw_threshold = nsfw_threshold
    
    " ========== “yes”或“no”函数 ========== "
    # 这个函数是一个 Logit 处理器，强制它在面对图片审查时，下一句只能说出“yes”或者“no”
    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Sets all tokens but yes/no to the minimum.
        """
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    " ========== 正经干活的 - 图像测试函数 ========== "
    def test_image(self, image: Image.Image | str | torch.Tensor) -> bool:
        #万能格式兼容 (Input Normalization): 支持 PIL 图像对象、图像文件路径字符串、以及 PyTorch 张量格式的图像输入
        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, str):
            image = Image.open(image)

        #第一道防线 —— 快速“秒杀” (Fast NSFW Check):
        classification = next(c for c in self.nsfw_classifier(image) if c["label"] == "nsfw")
        if classification["score"] > self.nsfw_threshold:
            return True

        # 动态等比缩放 (VRAM Optimization) 512^2 pixels are enough for checking
        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        #第二道防线 —— 深度语义审查 (VLM Prompting):
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "content": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        #强制单选与最终宣判: 调用“yes”或“no”函数，并根据输出结果返回布尔值
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token

    
    " ========== 文本测试函数 ========== "
    #拦截用户输入的违规提示词（Prompt）
    def test_txt(self, txt: str) -> bool:
        # chat - 将用户输入的待审核文本 (txt) 嵌入到了一个预设的系统模板
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            }
        ]
        #文本向量化: 
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        # 强制单选推理: 依旧调用“yes”或“no”函数，并根据输出结果返回布尔值
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token
