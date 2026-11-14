# src/agents/vision_agent.py
from pathlib import Path

from src.qwen_client import QwenClient


class VisionAgent:

    def __init__(self, client: QwenClient):
        self.client = client
        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        self.prompt_path = PROJECT_ROOT / "prompts" / "vision_analysis.txt"

        if not self.prompt_path.is_file():
            raise FileNotFoundError(
                f"找不到提示词文件：{self.prompt_path}"
            )

        self.prompt = self.prompt_path.read_text(
            encoding="utf-8"
        )

    def run(self, image_path: str) -> str:
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"图片不存在：{image_path}"
            )

        # 将图片文件名传给模型作为 image_id
        prompt = (
            self.prompt
            + f"\n\n本次图片的 image_id 必须设置为：{image_path.stem}"
        )

        raw_result = self.client.analyze_image(
            image_path=str(image_path),
            prompt=prompt
        )

        return raw_result