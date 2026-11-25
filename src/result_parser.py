# src/result_parser.py
# =============================================================================
# 把模型的自由文本变成合法数据
# =============================================================================
# 解析分三层，任何一层失败都要给出**能直接反馈给模型**的短错误信息：
#   ① 文本 -> dict      ：处理代码围栏、尾逗号、全角引号
#   ② dict -> 类型      ：pydantic 校验（字段、枚举、字数、坐标范围）
#   ③ 类型 -> 业务规则  ：候选超过 5 个就截断、千分制坐标换算
#
# 为什么错误信息要短：解析失败后我们会把它塞回给模型让它重写，
# pydantic 默认那几行带 URL 的报错，模型看了反而更糊涂。
#
# 同时存在的两套格式：
#   - Observation     ：v2 提示词，带坐标框，是现在的主线
#   - AnalysisResult  ：v1 提示词，只有文字位置，保留下来给旧文件做对照
# parse_any() 会按内容自动判断是哪一套，这样 v1/v2 能并排跑。
# =============================================================================

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from src.schemas import AnalysisResult, Observation, ReportDraft, ReviewReport

# A 阶段最多返回几个候选区域。
# 这个数必须和 Observation.findings 的 max_length 一致，改一处就要改另一处。
# 为什么不是 3：卡片上只放 3 个，但候选要比最终名额多，
# 否则 B 审查官剔除一个之后就没有备选了。
MAX_CANDIDATES = 5

# 坐标换算的兜底阈值。
# 有些视觉模型习惯输出 0-1000 的整数坐标（千分制），而不是 0-1。
# 如果发现最大值超过 1、但不超过 1000，就认为它是千分制，统一除以 1000。
# 这一步会记进日志（bbox_rescaled），不会被悄悄吞掉。
_LIKELY_THOUSAND_SCALE_MAX = 1000.0


# =============================================================================
# 第一层：文本 -> dict
# =============================================================================


def _strip_code_fence(text: str) -> str:
    """去掉 Markdown 代码围栏。

    提示词里明确说了不要围栏，但模型经常不听。这里兜一下：
    既能处理 ``` 包住整个回复的情况，也能处理只有开头有 ```json 的情况。
    """
    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # 去掉第一行（``` 或 ```json）
    lines = lines[1:]
    # 去掉可能的结尾 ```
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _remove_trailing_commas(text: str) -> str:
    """删掉对象/数组结尾多余的逗号。

    JSON 不允许 {"a": 1,} 这种写法，但模型非常爱这么写，
    是所有格式错误里最常见的一种。
    """
    return re.sub(r",(\s*[}\]])", r"\1", text)


def extract_json(text: str) -> dict:
    """从模型回复里抠出第一个 JSON 对象。

    按"最宽松到最严格"的顺序尝试，每次失败才进入下一档修复：
      原始 -> 去尾逗号 -> 再把全角引号换成半角
    全角引号那一步放最后，因为它有误伤风险（正文里本来就有全角引号时），
    只有在别的办法都失败后才值得一试。
    """
    cleaned = _strip_code_fence(text)

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("模型回复中没有找到 JSON 对象（找不到左花括号）")

    candidate = cleaned[start:]

    attempts = (
        candidate,
        _remove_trailing_commas(candidate),
        _remove_trailing_commas(candidate)
        .replace("“", '"')
        .replace("”", '"'),
    )

    last_error: json.JSONDecodeError | None = None

    decoder = json.JSONDecoder()
    for attempt in attempts:
        try:
            data, _ = decoder.raw_decode(attempt)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            raise ValueError("解析结果不是一个 JSON 对象")
        return data

    # 三次都失败：把位置报出来，方便定位是截断还是语法错
    assert last_error is not None
    raise ValueError(
        f"JSON 解析失败（第 {last_error.lineno} 行第 {last_error.colno} 列："
        f"{last_error.msg}）"
    )


# =============================================================================
# 第二层：业务规则修补（在 pydantic 校验之前做）
# =============================================================================


@dataclass
class ParseMeta:
    """解析过程中的"异常但可接受"的情况，用来写日志。

    这些数字很值钱：比如连续几次 truncated 都大于 0，说明提示词里的
    "最多 5 个"根本没约束住模型，得在答辩里说明或者改提示词。
    """

    raw_findings: int = 0      # 模型原本给了几个候选
    kept: int = 0              # 实际保留几个
    truncated: int = 0         # 被硬截掉几个
    bbox_rescaled: int = 0     # 有几个框被从千分制换算成 0-1


def _is_bbox_array(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 4


def _maybe_rescale_bbox(bbox: Any) -> tuple[Any, bool]:
    """如果是千分制坐标，换算成 0-1。

    只在"明确越界但看起来像千分制"时才动手：最大值大于 1、且所有值都在
    0 到 1000 之间。真正的脏数据（比如 5000、负数）不会被这里悄悄改掉，
    交给 pydantic 报错，走重试。
    """
    if not _is_bbox_array(bbox):
        return bbox, False

    try:
        values = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return bbox, False

    if max(values) <= 1.0:
        return bbox, False

    if all(0.0 <= value <= _LIKELY_THOUSAND_SCALE_MAX for value in values):
        return [round(value / 1000.0, 4) for value in values], True

    return bbox, False


def _compact_error(exc: ValidationError) -> str:
    """把 pydantic 的错误压成一句话，用来回喂给模型。

    只取第一条：模型一次改一个错就够了，一次甩五条它反而改不干净。
    """
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "<根>"
    return f"{location}: {first['msg']}"


# =============================================================================
# 第三层：对外接口
# =============================================================================


def parse_observation(text: str) -> tuple[Observation, ParseMeta]:
    """v2 的输出解析：文本 -> Observation。

    顺序很重要：**先截断，再校验**。
    Observation.findings 有 max_length=5，如果先校验，
    模型给 8 条会直接抛错；但我们其实只想保留前 5 条，不需要为这个重试一次。
    """
    data = extract_json(text)
    meta = ParseMeta()

    raw_findings = data.get("findings")
    if not isinstance(raw_findings, list):
        # 模型有时会把 findings 写成 null 或字符串，这里兜成空列表，
        # 空列表是合法的（"这张图没看出问题"本来就是一种合法输出）
        raw_findings = []
        data["findings"] = raw_findings

    meta.raw_findings = len(raw_findings)

    # ---- 业务规则 1：候选超量直接截断 ----
    if len(raw_findings) > MAX_CANDIDATES:
        raw_findings = raw_findings[:MAX_CANDIDATES]
        data["findings"] = raw_findings
        meta.truncated = meta.raw_findings - MAX_CANDIDATES

    # ---- 业务规则 2：千分制坐标换算 ----
    for finding in raw_findings:
        if not isinstance(finding, dict):
            continue
        new_bbox, rescaled = _maybe_rescale_bbox(finding.get("bbox"))
        if rescaled:
            finding["bbox"] = new_bbox
            meta.bbox_rescaled += 1

    try:
        observation = Observation.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_compact_error(exc)) from exc

    meta.kept = len(observation.findings)
    return observation, meta


def parse_analysis_result(text: str) -> AnalysisResult:
    """【旧版】v1 的输出解析。

    保留它有两个用处：读历史文件、和 v2 做对照。
    主线流程已经不用了，新代码别再调用。
    """
    data = extract_json(text)
    try:
        return AnalysisResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_compact_error(exc)) from exc


def parse_review(text: str) -> ReviewReport:
    """B 审查官的输出解析：文本 -> ReviewReport。

    比 A 那边少一层修补：裁决结果里没有坐标换算的问题，
    bbox 只有在 adjust 的时候才出现，格式错一次就重试，不值当再写特例。
    """
    data = extract_json(text)

    # verdicts 给 null 时兜成空列表：这和"一条都不裁"不是一回事，
    # 但至少不会因为 None 直接崩，后面 apply_verdicts 会把未裁决的候选
    # 全部保守保留并记 unjudged，日志里看得见。
    if data.get("verdicts") is None:
        data["verdicts"] = []

    try:
        return ReviewReport.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_compact_error(exc)) from exc


def _strip_label_prefix(text: str, *labels: str) -> str:
    """去掉模型自己加的"依据：""风险："前缀。

    卡片排版时字段名是代码画的，模型再写一遍就会变成
    "依据：依据R1的裂隙……"。提示词里已经要求不要写，但模型很爱照着
    示例抄，所以在解析层再兜一次——比在渲染层做字符串替换干净。
    """
    cleaned = text.strip()

    # 循环剥，直到没有前缀为止。
    # "依据：依据R1的裂隙" 这种双层前缀确实出现过——模型既照抄了示例里的
    # "依据："，又在正文里重复了一次。剥一次不够。
    changed = True
    while changed:
        changed = False
        for label in labels:
            for separator in ("：", ":"):
                prefix = f"{label}{separator}"
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True

    return cleaned


def parse_report(text: str) -> ReportDraft:
    """C 成稿的输出解析：文本 -> ReportDraft。

    这里要多做一件事：检查 basis 里有没有提到区域编号。
    赛题要求"建议要说明依据"，而依据如果连"对应哪个区域"都不写，
    就是一句放之四海皆准的套话，等于没有依据。
    这一步拦在解析层，比等卡片印出来再发现要好。
    """
    data = extract_json(text)

    try:
        draft = ReportDraft.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_compact_error(exc)) from exc

    # 去掉模型照抄示例带上的字段名前缀，避免卡片上出现"依据：依据……"
    for rec in draft.recommendations:
        rec.action = _strip_label_prefix(rec.action, "建议", "action")
        rec.basis = _strip_label_prefix(rec.basis, "依据", "basis")
        rec.risk = _strip_label_prefix(rec.risk, "风险", "risk")

    missing_region = [
        index + 1
        for index, rec in enumerate(draft.recommendations)
        if not re.search(r"R[1-9]", rec.basis)
    ]
    if missing_region:
        # 依据里不写对应哪个区域，就只是一句放之四海皆准的套话，
        # 等于没有依据——而"建议及依据合理"是评分维度里的原话。
        # 在这里抛错会触发一次重试，比等卡片印出来再发现便宜。
        raise ValueError(
            "recommendations.{}: basis 里必须写明对应的区域编号（如 R1）".format(
                missing_region[0] - 1
            )
        )

    return draft


def _looks_legacy(data: dict) -> bool:
    """判断一份输出是 v1 还是 v2 的格式。

    判据是"有没有 status 字段"：v1 的每条 finding 必带 status，
    v2 的候选必须带 bbox 且没有 status。两套格式在这点上完全不重叠，
    所以按内容判断是安全的，不会误判。
    """
    findings = data.get("findings")
    if not isinstance(findings, list) or not findings:
        # findings 为空时看不出差别，按 v2 处理（v2 允许空列表）
        return False

    first = findings[0]
    if not isinstance(first, dict):
        return False

    return "status" in first and "bbox" not in first


def parse_any(
    text: str,
) -> tuple[Literal["observation", "analysis"], Observation | AnalysisResult, ParseMeta]:
    """按内容自动判断版本，返回 (阶段, 解析结果, 解析元信息)。

    给 workflow 用：同一套代码既能跑 v2 也能跑 v1，
    这样对比两种提示词时不用切分支。
    """
    data = extract_json(text)

    if _looks_legacy(data):
        try:
            analysis = AnalysisResult.model_validate(data)
        except ValidationError as exc:
            raise ValueError(_compact_error(exc)) from exc
        return "analysis", analysis, ParseMeta(
            raw_findings=len(analysis.findings),
            kept=len(analysis.findings),
        )

    # 走 v2：复用 parse_observation，但数据已经解析过了，
    # 这里重新走一遍 extract_json 成本可以忽略（就一次 json 解析）
    observation, meta = parse_observation(text)
    return "observation", observation, meta
