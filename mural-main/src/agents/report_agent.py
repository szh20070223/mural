# src/agents/report_agent.py
# =============================================================================
# C 成稿：把审查后的区域整理成卡片上的文字
# =============================================================================
# C 是唯一**不看图**的一环，这是故意的：
#   看图判断已经在 A 和 B 做完了。如果 C 也看图，它就有机会推翻前面的结论，
#   而它并没有 B 那样的裁决职责，结果只会变成一个没人负责的第三意见。
#
#   更实际的理由：C 只做文字整理，纯文本调用比带图快得多、也便宜得多。
#
# 它能改的东西也被刻意限制：只有总述、建议、局限。
# 区域的位置、病害、观察、影响，一律不许动。
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import PROJECT_ROOT
from src.guidelines import guidelines_for
from src.qwen_client import ModelReply, QwenClient
from src.schemas import Region
from src.trace import sha256_short

DEFAULT_REPORT_PROMPT = PROJECT_ROOT / "prompts" / "report_agent_v2.txt"


class ReportAgent:
    """C 阶段的执行者。"""

    def __init__(self, client: QwenClient, prompt_path: str | Path | None = None) -> None:
        self.client = client
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_REPORT_PROMPT

        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"找不到成稿提示词文件：{self.prompt_path}")

        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.prompt_sha = sha256_short(self.prompt)

        match = re.search(r"_v(\d+)$", self.prompt_path.stem)
        self.version = f"v{float(match.group(1)):g}" if match else "v1"

    # ------------------------------------------------------------------
    # 把结构化数据摊成模型好读的文本
    # ------------------------------------------------------------------

    @staticmethod
    def _format_guidelines(diseases: list[str]) -> str:
        """把处置原则摊成"可建议 / 必须说的风险"两行。

        只取卡片上真实出现的病害——把 8 条全给过去，模型会忍不住
        给没上卡的病害也写建议，结果建议和区域对不上。
        """
        selected = guidelines_for(diseases)

        if not selected:
            # 理论上不会发生（病害名来自白名单），但真发生了要能看出来，
            # 而不是让模型凭空发挥
            return "（本次没有匹配到可用的处置原则，请只写记录与监测类建议）"

        lines: list[str] = []
        for disease, rule in selected.items():
            lines.append(f"- {disease}")
            lines.append(f"    可建议的做法：{rule['principle']}")
            lines.append(f"    必须说明的风险：{rule['risk']}")
        return "\n".join(lines)

    @staticmethod
    def _format_regions(regions: list[Region]) -> str:
        """把区域摊成清单。框也给它：写建议时"范围大小"会影响判断，
        比如一个覆盖全图的裂隙和一个局部小裂隙，建议是不一样的。"""
        lines: list[str] = []
        for region in regions:
            box = ", ".join(f"{value:.0%}" for value in region.bbox.as_list())
            # 待复核的类型要带上标记，否则 C 会把它当成已确认的病害来写
            disease_label = region.disease + (
                "（待复核）" if region.pending_review else ""
            )
            lines.append(
                f"- {region.region_id}　{disease_label}"
                f"（置信度：{region.confidence}，范围：[{box}]）"
            )
            lines.append(f"    观察：{region.observation}")
            lines.append(f"    影响：{region.impact}")
            if region.review_notes:
                lines.append(f"    需人工复核：{'；'.join(region.review_notes)}")
        return "\n".join(lines)

    def build_prompt(self, regions: list[Region], limitations: list[str]) -> str:
        """拼出 C 的完整输入。"""
        guidelines = self._format_guidelines([region.disease for region in regions])
        region_text = self._format_regions(regions)

        limitation_text = (
            "\n".join(f"- {item}" for item in limitations) if limitations else "（无）"
        )

        return (
            self.prompt.replace("{guidelines}", guidelines)
            .replace("{regions}", region_text)
            .replace("{limitations}", limitation_text)
        )

    def run(
        self,
        regions: list[Region],
        limitations: list[str],
        history: list[dict] | None = None,
    ) -> ModelReply:
        """跑一次成稿。注意这里**不传图片**，是纯文本调用。"""
        prompt = self.build_prompt(regions, limitations)

        messages: list[dict] = [{"role": "user", "content": prompt}]
        if history:
            messages.extend(history)

        return self.client.chat(messages)

    def correction_message(self, error: str) -> str:
        """解析失败时的纠正说明。

        关键是把"修正"和"重写"分开：模型收到格式错误时，最容易的做法是
        重新组织一遍内容——于是两条建议被并成一条、某个区域被漏掉，
        格式是过了，内容却退化了。所以这里明确要求：只改报错的那个字段。
        """
        return (
            f"你上一次的回复无法被解析，原因是：{error}\n\n"
            "这是**修正**，不是重写：\n"
            "1. 只改报错的那个字段，其他字段保持原样；\n"
            "2. 建议的条数不要减少，也不要合并；\n"
            "3. 如果报错是字数超限，就把那句话压缩到上限以内，不要删掉要点；\n"
            "4. 只返回一个 JSON 对象，不要 Markdown 代码围栏，不要任何解释文字；\n"
            "5. 每条建议必须有 action、basis、risk，risk 不能为空；\n"
            "6. basis 里必须写明对应的区域编号（R1/R2/R3）；\n"
            "7. summary 不超过 80 字。"
        )
