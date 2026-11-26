# src/review_rules.py
# =============================================================================
# B 之后的确定性处理：把模型的裁决落成卡片上的 1-3 个区域
# =============================================================================
# 分工原则：**能算的交给代码，要判断的交给模型。**
#   - 两个框重叠多少才算重叠？这是 IoU，能算。让模型算，它既不准也不稳定。
#   - 留几个、怎么去重、怎么编号？这是规则，不需要模型参与。
#   - "这一处到底算不算病害"？这才是模型该回答的。
#
# 把这两类混在一起，就会得到一个"看起来在审查、其实全靠模型自觉"的环节。
# 而模型是记不住"最多保留 3 个"这种硬约束的——它上一轮刚给了 4 个候选。
#
# 这一层完全可测试、可复现：对同一输入可复现；更换模型输出仍可能改变最终结果。
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import BoundingBox, ObservedRegion, Observation, Region, ReviewReport

# 卡片上最多放几个区域。赛题原文："在原图上标出 1—3 处值得关注的区域"
MAX_CARD_REGIONS = 3

# 同类框的重叠阈值沿用 0.6，属于待验证的工程启发式，不是病害鉴定标准。
IOU_MERGE_THRESHOLD = 0.6

# 置信度排序；合并不会提高模型自评等级
_CONFIDENCE_RANK = {"低": 0, "中": 1, "高": 2}


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """两个框的交并比（Intersection over Union），取值 0-1。

    这是目标检测里衡量"两个框重合多少"的标准做法：
    交集面积 / 并集面积。完全重合是 1，完全不沾边是 0。
    """
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    # 没有交集（其中一个框完全在另一个的左边/上边）
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area_ratio() + b.area_ratio() - intersection

    return intersection / union if union > 0 else 0.0


def union_box(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    """把两个框合成一个能同时罩住它们的最小框。

    合并两条候选时用这个，而不是简单保留其中一个：
    只保留大的会丢掉另一条覆盖的边缘部分，只保留小的会漏掉。
    """
    return BoundingBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def _lower_confidence(first: str, second: str) -> str:
    """重复模型意见不是独立证据，合并时保留较低置信度。"""
    return first if _CONFIDENCE_RANK[first] <= _CONFIDENCE_RANK[second] else second


def validate_review(observation: Observation, report: ReviewReport) -> ReviewReport:
    """在重试边界校验跨阶段引用；漏判不能变成默认肯定。"""
    findings = {item.region_id: item for item in observation.findings}
    verdicts = {item.region_id: item for item in report.verdicts}
    if set(findings) != set(verdicts):
        raise ValueError(f"裁决必须完整且仅覆盖候选：缺少 {sorted(set(findings)-set(verdicts))}；"
                         f"未知 {sorted(set(verdicts)-set(findings))}")
    for verdict in report.verdicts:
        if verdict.decision != "merge":
            continue
        target = verdicts.get(verdict.merge_into)
        if target is None or target.decision not in {"keep", "adjust", "defer"}:
            raise ValueError("merge 必须直接指向保留项，不允许合并链、循环或被剔除目标")
        source = findings[verdict.region_id]
        destination = findings[target.region_id]
        disease = verdict.adjusted_disease or source.disease
        target_disease = "待判读" if target.decision == "defer" else (target.adjusted_disease or destination.disease)
        if disease != target_disease:
            raise ValueError("不同类别不能合并；请先对类别作出一致裁决或独立保留")
        if iou(verdict.adjusted_bbox or source.bbox, target.adjusted_bbox or destination.bbox) <= 0:
            raise ValueError("不相交的区域不能合并，请独立保留")
    return report


@dataclass
class ReviewStats:
    """决策过程的统计，写进日志用。

    这几个数字是答辩时的好材料：模型提了几条、审查砍了几条、
    有几条是合并掉的、最后进卡几条。一条完整的"人机协同"证据链。
    """

    proposed: int = 0          # A 提出的候选数
    unjudged: int = 0          # 旧日志兼容字段；当前漏判会重试/失败
    dropped: int = 0           # 被判 drop 的
    adjusted: int = 0          # 被判 adjust 的
    merged_by_model: int = 0   # 模型主动要求合并的
    merge_orphan: int = 0      # 旧日志兼容字段；当前孤立目标会拒绝
    merge_box_mismatch: int = 0  # 旧日志兼容字段；当前不相交合并会拒绝
    merged_by_iou: int = 0     # 代码兜底按重合度合并的
    truncated: int = 0         # 超过 3 个被截掉的
    final: int = 0             # 最终进卡数量


@dataclass
class _Candidate:
    """处理过程中的中间形态：从 ObservedRegion 出发，叠加裁决结果。"""

    source: ObservedRegion
    box: BoundingBox
    disease: str
    confidence: str
    observation: str | None = None
    impact: str | None = None
    source_ids: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)


def apply_verdicts(
    observation: Observation,
    report: ReviewReport,
    *,
    max_regions: int = MAX_CARD_REGIONS,
    iou_threshold: float = IOU_MERGE_THRESHOLD,
) -> tuple[list[Region], ReviewStats]:
    """把 A 的候选 + B 的裁决，落成卡片上的 1-3 个 Region。

    处理顺序（这个顺序有讲究）：
      ① 按裁决分流：drop 丢掉，merge 先挂起，keep/adjust 进保留列表
      ② 执行 merge：把挂起的并到目标框上
      ③ 按 IoU 兜底去重：模型没看出来的重复，代码补一刀
      ④ 截断到 3 个、重新编号 R1-R3

    为什么要③：模型经常嘴上说"这两条重复"，实际漏掉；也经常两条都 keep
    但框明显重叠。兜底比信任稳定。
    """
    if not 1 <= max_regions <= 3 or not 0 < iou_threshold <= 1:
        raise ValueError("max_regions 必须为 1–3，iou_threshold 必须在 (0,1]")
    validate_review(observation, report)
    stats = ReviewStats(proposed=len(observation.findings))

    by_id = {finding.region_id: finding for finding in observation.findings}
    verdict_by_id = {verdict.region_id: verdict for verdict in report.verdicts}
    unknown_ids = set(verdict_by_id) - set(by_id)
    if unknown_ids:
        # 未知 ID 不能悄悄进入合并逻辑；它们不对应图上的任何候选。
        raise ValueError(f"审查结果包含未知候选编号：{sorted(unknown_ids)}")

    kept: list[_Candidate] = []
    index: dict[str, int] = {}
    pending_merge: list = []

    # ---- ① 按裁决分流 ----
    for reviewed in report.verdicts:
        finding = by_id[reviewed.region_id]
        verdict = verdict_by_id.get(finding.region_id)

        if verdict.decision == "drop":
            stats.dropped += 1
            continue

        if verdict.decision == "merge":
            # 先挂起：目标可能排在后面，等所有保留项都入列再处理
            pending_merge.append(verdict)
            continue

        # keep / adjust / defer
        deferred = verdict.decision == "defer"
        candidate = _Candidate(
            source=finding,
            box=verdict.adjusted_bbox or finding.bbox,
            disease=("待判读" if deferred else (verdict.adjusted_disease or finding.disease)),
            confidence=("低" if deferred else (verdict.adjusted_confidence or finding.confidence)),
            review_notes=list(verdict.review_notes),
            observation=verdict.adjusted_observation or finding.observation,
            impact=verdict.adjusted_impact or finding.impact,
            source_ids=[finding.region_id],
        )
        if verdict.decision in {"adjust", "defer"}:
            stats.adjusted += 1

        kept.append(candidate)
        index[finding.region_id] = len(kept) - 1

    # ---- ② 执行 merge ----
    for verdict in pending_merge:
        target_index = index.get(verdict.merge_into or "")

        target = kept[target_index]
        source = by_id.get(verdict.region_id)
        if source is None:
            raise ValueError(f"merge 来源候选不存在：{verdict.region_id}")
        source_box = verdict.adjusted_bbox or source.bbox

        # validate_review 已验证类别、交集和目标；保留合并前的来源映射。
        target.box = union_box(target.box, source_box)
        notes = list(dict.fromkeys(target.review_notes + verdict.review_notes))
        target.review_notes = notes if len(notes) <= 2 else [notes[0], "；".join(notes[1:])]
        target.source_ids.append(verdict.region_id)
        target.confidence = _lower_confidence(target.confidence, verdict.adjusted_confidence or source.confidence)
        stats.merged_by_model += 1

    # ---- ③ 按 IoU 兜底去重 ----
    deduped: list[_Candidate] = []
    for candidate in kept:
        for existing in deduped:
            if existing.disease == candidate.disease and iou(existing.box, candidate.box) >= iou_threshold:
                existing.box = union_box(existing.box, candidate.box)
                existing.confidence = _lower_confidence(existing.confidence, candidate.confidence)
                notes = list(dict.fromkeys(existing.review_notes + candidate.review_notes))
                existing.review_notes = notes if len(notes) <= 2 else [notes[0], "；".join(notes[1:])]
                existing.source_ids.extend(candidate.source_ids)
                stats.merged_by_iou += 1
                break
        else:
            deduped.append(candidate)

    # ---- ④ 截断 + 重新编号 ----
    if len(deduped) > max_regions:
        stats.truncated = len(deduped) - max_regions
        deduped = deduped[:max_regions]

    regions: list[Region] = []
    for order, candidate in enumerate(deduped, start=1):
        regions.append(
            Region(
                region_id=f"R{order}",
                disease=candidate.disease,
                bbox=candidate.box,
                observation=candidate.observation or candidate.source.observation,
                impact=candidate.impact or candidate.source.impact,
                confidence=candidate.confidence,
                review_notes=candidate.review_notes[:2],
                ai_generated=True,
                source_candidate_ids=sorted(set(candidate.source_ids)),
            )
        )

    stats.final = len(regions)
    return regions, stats
