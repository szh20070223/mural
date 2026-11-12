import os
import base64
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL")
)

MODEL = os.getenv("QWEN_MODEL")


def encode_image(image_path):
    """将本地图片编码为 Base64"""
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }.get(suffix)

    if mime_type is None:
        raise ValueError("暂不支持该图片格式")

    return f"data:{mime_type};base64,{image_data}"


def analyze_image(image_path):
    image_url = encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        请分析这张壁画图片。

                        重点观察以下病害：
                        1. 裂隙
                        2. 起甲
                        3. 颜料层脱落
                        4. 地仗层脱落
                        5. 粉化
                        6. 褪色
                        7. 水渍

                        请区分明确发现、疑似发现和无法判断的情况。
                        不要将正常绘画纹理直接认定为病害。
                        请说明判断依据和大致位置。
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )

    return response


if __name__ == "__main__":
    result = analyze_image("../data/images/test_mural1.png")
    print(result.choices[0].message.content)