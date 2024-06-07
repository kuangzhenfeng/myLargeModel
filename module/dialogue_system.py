import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore
from .web_scraper import fetch_web_content_by_query

class DialogueSystem:
    """
    对话系统类，用于生成对话回复和处理对话历史。

    参数：
    model_name (str): 使用的预训练模型的名称。
    mode (str): 对话系统的默认模式，可以是 "normal"（普通模式）或 "web"（网络模式）。
    """
    def __init__(self, model_name, mode="normal"):
        """
        初始化对话系统。

        参数：
        model_name (str): 使用的预训练模型的名称。
        mode (str): 对话系统的默认模式，可以是 "normal"（普通模式）或 "web"（网络模式）。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.mode = mode
        # 对话历史记录，初始化为系统问候语
        self.dialogue_history = [{"role": "system", "content": "你是一个乐于助人的助手。"}]

    def generate_response(self, user_input):
        """
        生成对话回复。

        参数：
        user_input (str): 用户输入的文本。

        返回：
        str: 生成的对话回复。
        """
        self.dialogue_history.append({"role": "user", "content": user_input})

        if self.mode == "normal":
            return self._generate_normal_response(user_input)
        elif self.mode == "web":
            return self._generate_web_response(user_input)

    def _generate_normal_response(self, text):
        """
        生成普通模式下的对话回复。

        参数：
        text (str): 要生成回复的文本。

        返回：
        str: 生成的对话回复。
        """
        response = self._model_response(text)
        self._add_to_dialogue_history("assistant", response)
        return response

    def _generate_web_response(self, text):
        """
        生成网络模式下的对话回复。

        参数：
        text (str): 要生成回复的文本。

        返回：
        str: 生成的对话回复。
        """
        keywords = self._extract_keywords(text)
        self._add_to_dialogue_history("assistant", f"关键词是：{keywords}")
        print(f"{Fore.YELLOW}关键词提取: {keywords}{Fore.RESET}")
        web_content = fetch_web_content_by_query(keywords)
        self._add_to_dialogue_history("assistant", f"联网搜索结果:{web_content}")
        print(f"{Fore.GREEN}联网搜索结果: {web_content}{Fore.RESET}")
        cleaned_web_content = self._clean_web_content(web_content)
        self._add_to_dialogue_history("assistant", f"整理后回答是：{cleaned_web_content}")
        return cleaned_web_content

    def _model_response(self, text, record=True):
        """
        使用模型生成对话回复。

        参数：
        text (str): 要生成回复的文本。
        record (bool): 是否记录对话历史，默认为True。

        返回：
        str: 生成的对话回复。
        """
        if record:
            text = self.tokenizer.apply_chat_template(
                self.dialogue_history,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if record:
            self._add_to_dialogue_history("assistant", response)
        return response

    def _extract_keywords(self, text):
        """
        从文本中提取关键词。

        参数：
        text (str): 要提取关键词的文本。

        返回：
        str: 提取的关键词。
        """
        return text  # 暂不进行关键词提取
        text = "请提取关键词，仅包含问题内容，不要进行回答。文本：\n" + text
        # keywords = self._model_response(text, record=False)  # 模型提取为单次有效，不依赖历史记录
        self._add_to_dialogue_history("assistant", f"提取的关键词是：{keywords}")
        return keywords

    def _clean_web_content(self, text):
        """
        整理从网页中提取的内容。

        参数：
        text (str): 要整理的网页内容。

        返回：
        str: 整理后的网页内容。
        """
        text = "请整理以下网页内容，并在此次回答中仅包含内容：\n" + text
        cleaned_web_content = self._model_response(text)
        self._add_to_dialogue_history("assistant", f"整理后的网页内容是：{cleaned_web_content}")
        return cleaned_web_content

    def _add_to_dialogue_history(self, role, content):
        """
        将对话历史记录添加到历史记录中。

        参数：
        role (str): 角色，可以是 "user"（用户）或 "assistant"（助手）。
        content (str): 对话内容。
        """
        self.dialogue_history.append({"role": role, "content": content})
