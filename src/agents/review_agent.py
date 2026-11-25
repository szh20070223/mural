# src/agents/review_agent.py
# =============================================================================
# B 审查官：看着原图，复核 A 给出的候选区域
# =============================================================================
# 为什么 B 必须也看到图片：
#   如果只把 A 的文本丢给 B，它能做的只有"语言润色"——
#   改改措辞、调调置信度，本质上是在复述 A 的判断。
#   真正的复核必须回到原图：这个框里到底有没有东西？框住的是不是那块白？
#
# 这是整条链路里唯一"两遍看图"的环节，也是最能说明
# "AI 能力边界"和"人工复核点"的地方——评审要看的正是这个。
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import PROJECT_ROOT
from src.qwen_client import ModelReply, QwenClient
from src.schemas import Observation
from src.trace import sha256_short

DEFAULT_REVIEW_PROMPT = PROJECT_ROOT / "prompts" / "review_agent.txt"


class ReviewAgent:
    """B 阶段的执行者。"""

    def __init__(self, client: QwenClient, prompt_path: str | Path | None = None) -> None:
        self.client = client
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_REVIEW_PROMPT

        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"找不到审查提示词文件：{self.prompt_path}")

        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.prompt_sha = sha256_short(self.prompt)

        match = re.search(r"_v(\d+)$", self.prompt_path.stem)
        self.version = f"v{float(match.group(1)):g}" if match else "v1"

    def build_prompt(self, observation: Observation, image_id: str | None = None) -> str:
        """把 A 的结果整理成"待审清单"，接到提示词后面。

        几个刻意的处理：
          - bbox 用数组而不是 {"x1": ...} 对象：少一层嵌套，模型少犯一次错，
            也和它要输出的格式保持一致；
          - 把 summary 和 limitations 一起给它：判断一个区域值不值得保留，
            需要知道整体情况（比如"这张图本来就缺侧光"）；
          - 明确告诉它"每条都要裁"，否则它会挑几条裁、剩下的不提。
        """
        payload = {
            "image_id": observation.image_id,
            "summary": observation.summary,
            "limitations": observation.limitations,
            "candidates": [
                {
                    "region_id": finding.region_id,
                    "disease": finding.disease,
                    "bbox": finding.bbox.as_list(),
                    "observation": finding.observation,
                    "impact": finding.impact,
                    "confidence": finding.confidence,
                    "limitations": finding.limitations,
                }
                for finding in observation.findings
            ],
        }

        return (
            f"{self.prompt}\n\n"
            f"本次图片的 image_id 必须设置为：{image_id or observation.image_id}\n\n"
            "以下是观察员给出的候选区域清单，请对每一条给出裁决：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "再次强调：只返回 JSON，不要 Markdown 代码围栏，不要任何解释文字。"
        )

    def run(
        self,
        image_path: str | Path,
        observation: Observation,
        history: list[dict] | None = None,
    ) -> ModelReply:
        """跑一次审查。history 用于自修复重试。"""
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")

        prompt = self.build_prompt(observation, image_id=path.stem)
        return self.client.analyze_image(str(path), prompt, history=history)

    def correction_message(self, error: str) -> str:
        """解析失败时的纠正说明，和 A 的版本略有不同。"""
        return (
            f"你上一次的回复无法被解析，原因是：{error}\n\n"
            "请重新输出，严格遵守给定的 JSON 格式：\n"
            "1. 只返回一个 JSON 对象，不要 Markdown 代码围栏，不要任何解释文字；\n"
            "2. verdicts 必须覆盖候选清单里的**每一条**，region_id 不能新造；\n"
            "3. decision 只能是 keep、adjust、drop、merge；\n"
            "4. decision=adjust 必须给出 adjusted_bbox（归一化 [x1,y1,x2,y2]）；\n"
            "5. decision=merge 必须给出 merge_into，且不能等于自己；\n"
            "6. 每条都要写 reason，不要只写'证据不足'。"
        )
