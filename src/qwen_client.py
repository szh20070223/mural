# src/qwen_client.py
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL
)


class QwenClient:

    def __init__(self):
        self.model = QWEN_MODEL

        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL
        )

    def encode_image(self, image_path: str) -> str:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")

        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type not in {
            "image/png",
            "image/jpeg",
            "image/webp"
        }:
            raise ValueError(f"暂不支持的图片格式：{mime_type}")

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        image_url = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Qwen 返回了空结果")

        return content