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
# 这一层完全可测试、可复现：换模型、换提示词，都不会影响这里的判定。
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import BoundingBox, ObservedRegion, Observation, Region, ReviewReport

# 卡片上最多放几个区域。赛题原文："在原图上标出 1—3 处值得关注的区域"
MAX_CARD_REGIONS = 3

# 两个框的重叠度超过这个值，就认为它们是同一处病害，合并掉。
# 0.6 是个经验值：低于它通常是"相邻但不相同"（比如两块挨着脱落的颜料），
# 高于它基本可以确定是同一个东西被标了两遍。
IOU_MERGE_THRESHOLD = 0.6

# 置信度强弱，用于合并时取更高的那个
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


def _stronger(first: str, second: str) -> str:
    """取两个置信度里更高的那个。"""
    return first if _CONFIDENCE_RANK.get(first, 0) >= _CONFIDENCE_RANK.get(second, 0) else second


@dataclass
class ReviewStats:
    """决策过程的统计，写进日志用。

    这几个数字是答辩时的好材料：模型提了几条、审查砍了几条、
    有几条是合并掉的、最后进卡几条。一条完整的"人机协同"证据链。
    """

    proposed: int = 0          # A 提出的候选数
    unjudged: int = 0          # 模型漏判、被保守保留的
    dropped: int = 0           # 被判 drop 的
    adjusted: int = 0          # 被判 adjust 的
    merged_by_model: int = 0   # 模型主动要求合并的
    merge_orphan: int = 0      # 要合并但目标不在保留列表里，只好独立保留的
    merge_box_mismatch: int = 0  # 说是同一处、但两个框根本不重叠的
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
    stats = ReviewStats(proposed=len(observation.findings))

    by_id = {finding.region_id: finding for finding in observation.findings}
    verdict_by_id = {verdict.region_id: verdict for verdict in report.verdicts}

    kept: list[_Candidate] = []
    index: dict[str, int] = {}
    pending_merge: list = []

    # ---- ① 按裁决分流 ----
    for finding in observation.findings:
        verdict = verdict_by_id.get(finding.region_id)

        if verdict is None:
            # 模型漏判了一条。这里**保守保留**：
            # 宁可多留一个让 B 之外的人去砍，也不要因为模型偷懒丢掉真病害。
            # 漏判数会进日志，连续出现说明提示词没说清"每条都要裁"。
            stats.unjudged += 1
            kept.append(
                _Candidate(
                    source=finding,
                    box=finding.bbox,
                    disease=finding.disease,
                    confidence=finding.confidence,
                )
            )
            index[finding.region_id] = len(kept) - 1
            continue

        if verdict.decision == "drop":
            stats.dropped += 1
            continue

        if verdict.decision == "merge":
            # 先挂起：目标可能排在后面，等所有保留项都入列再处理
            pending_merge.append(verdict)
            continue

        # keep / adjust
        candidate = _Candidate(
            source=finding,
            box=verdict.adjusted_bbox or finding.bbox,
            disease=verdict.adjusted_disease or finding.disease,
            confidence=verdict.adjusted_confidence or finding.confidence,
            review_notes=list(verdict.review_notes),
        )
        if verdict.decision == "adjust":
            stats.adjusted += 1

        kept.append(candidate)
        index[finding.region_id] = len(kept) - 1

    # ---- ② 执行 merge ----
    for verdict in pending_merge:
        target_index = index.get(verdict.merge_into or "")

        if target_index is None:
            # 要合并的目标自己也被 drop 了，或者压根不存在。
            # 这时候不能把这条悄悄丢掉——它可能是真的，
            # 只是模型给错了合并目标。独立保留，记一笔。
            source = by_id.get(verdict.region_id)
            if source is not None:
                kept.append(
                    _Candidate(
                        source=source,
                        box=verdict.adjusted_bbox or source.bbox,
                        disease=verdict.adjusted_disease or source.disease,
                        confidence=verdict.adjusted_confidence or source.confidence,
                        review_notes=list(verdict.review_notes),
                    )
                )
                index[verdict.region_id] = len(kept) - 1
            stats.merge_orphan += 1
            continue

        target = kept[target_index]
        source_box = verdict.adjusted_bbox or by_id[verdict.region_id].bbox

        # 只有两个框真的重叠时才取并集。
        # 如果模型说"这是同一处"但两个框根本不沾边，取并集的结果会是一个
        # 横跨两处的巨大框——比不合还糟。这种情况保留目标框，记一笔。
        if iou(target.box, source_box) > 0.0:
            target.box = union_box(target.box, source_box)
        else:
            stats.merge_box_mismatch += 1

        # 置信度保持目标自己的：B 已经对目标做过判断，
        # 合并进来的那条不构成新证据，不该悄悄把等级改高。
        for note in verdict.review_notes:
            if note not in target.review_notes and len(target.review_notes) < 2:
                target.review_notes.append(note)
        stats.merged_by_model += 1

    # ---- ③ 按 IoU 兜底去重 ----
    deduped: list[_Candidate] = []
    for candidate in kept:
        for existing in deduped:
            if iou(existing.box, candidate.box) >= iou_threshold:
                existing.box = union_box(existing.box, candidate.box)
                existing.confidence = _stronger(existing.confidence, candidate.confidence)
                for note in candidate.review_notes:
                    if note not in existing.review_notes and len(existing.review_notes) < 2:
                        existing.review_notes.append(note)
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
                observation=candidate.source.observation,
                impact=candidate.source.impact,
                confidence=candidate.confidence,
                review_notes=candidate.review_notes[:2],
                ai_generated=True,
            )
        )

    stats.final = len(regions)
    return regions, stats
