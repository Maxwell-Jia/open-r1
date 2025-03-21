import json
import re
from typing import Optional

import requests
from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_output import GenerativeResponse
from lighteval.tasks.requests import GreedyUntilRequest
from transformers import AutoTokenizer


class CustomAPIMLModel(LightevalModel):
    """自定义 API 模型封装，用于 LightEval 评估"""

    def __init__(self, api_url, api_key, model_name="gpt2"):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def model_name(self):
        """返回自定义 API 模型名称"""
        return "CustomAPIModel"

    def get_next_words(self, *args, **kwargs):
        """API 模型不需要实现 get_next_words"""
        raise NotImplementedError("CustomAPIModel 不支持 get_next_words")

    def generate(self, prompt):
        """调用 API 生成响应"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": {},
            "response_mode": "streaming",
            "conversation_id": "",
            "user": "abc-123",
            "query": prompt
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=6000
            )

            messages = []

            for line in response.iter_lines():
                if line:
                    try:
                        # 去掉 "data: " 前缀
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            line = line[6:]  # 去掉 "data: "

                        # 解析 JSON
                        data = json.loads(line)
                        event_type = data.get("event", "")

                        # 只处理感兴趣的事件
                        if event_type == "message":
                            messages.append(data['answer'])
                    except json.JSONDecodeError as e:
                        pass

            print(''.join(messages))

            match = re.search(r'answer:\s*([A-Z])', ''.join(messages), re.IGNORECASE)

            if match:
                print(f"匹配的答案是：{match.group(0)}")
                return 'Answer: %s' % match.group(1)
            else:
                print("没有找到匹配的答案")
                return "No Answer"
        except requests.Timeout:
            print("API 请求超时，请检查网络或 API 状态")
            return "No Answer"
        except requests.RequestException as e:
            print(f"API 请求失败：{e}")
            return "No Answer"
        except ValueError:
            print("API 响应解析失败，返回数据格式不正确")
            return "No Answer"

    @property
    def add_special_tokens(self) -> bool:
        return False

    def greedy_until(
            self,
            requests: list[GreedyUntilRequest],
            override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for c in dataset:
            print(c.context)
            result = self.generate(c.context)

            cur_response = GenerativeResponse(
                result=[result],
                logits=None,
                generated_tokens=[],
                input_tokens=[],
            )
            results.append(cur_response)

        return dataset.get_original_order(results)

    def loglikelihood(self, *args, **kwargs):
        """不支持 loglikelihood"""
        raise NotImplementedError("loglikelihood 不适用于 API 模型")

    def loglikelihood_rolling(self, *args, **kwargs):
        """不支持 loglikelihood_rolling"""
        raise NotImplementedError("loglikelihood_rolling 不适用于 API 模型")

    def loglikelihood_single_token(self, *args, **kwargs):
        """不支持 loglikelihood_single_token"""
        raise NotImplementedError("loglikelihood_single_token 不适用于 API 模型")

    @property
    def max_length(self):
        """设置 API 模型最大长度"""
        return 1024

    @property
    def tokenizer(self):
        """API 模型不使用 tokenizer"""
        return self._tokenizer

    @property
    def model_info(self):
        """返回模型的基本信息，供 evaluation tracker 记录"""
        return ModelInfo(
            model_name=self.model_name
        )
