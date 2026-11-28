"""逐项验证 B 的分级，再按病害类型和置信度选择观察卡的代表区域。"""

from dataclasses import dataclass, field
from itertools import combinations

from src.schemas import BoundingBox, Observation, Region, ReviewReport

MAX_CARD_REGIONS = 3
DEFERRED_DISEASE = "待判读"
IOU_MERGE_THRESHOLD = 0.6


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


@dataclass
class ReviewStats:
    proposed: int = 0
    unjudged: int = 0
    dropped: int = 0
    adjusted: int = 0
    deferred: int = 0
    merged_by_model: int = 0
    merged_by_iou: int = 0
    truncated: int = 0
    final: int = 0
    grades: dict[str, int] = field(default_factory=lambda: {"高": 0, "中": 0, "低": 0})
    selected_ids: list[str] = field(default_factory=list)


def _disease(finding, verdict):
    if verdict.decision == "defer":
        return verdict.adjusted_disease or DEFERRED_DISEASE
    return verdict.adjusted_disease or finding.disease


def validate_review(observation: Observation, report: ReviewReport) -> ReviewReport:
    sources = {f.region_id: f for f in observation.findings}
    verdicts = {v.region_id: v for v in report.verdicts}
    if sources.keys() != verdicts.keys():
        raise ValueError(f"候选编号必须逐一对应，缺少 {sorted(sources.keys()-verdicts.keys())}，"
                         f"多出 {sorted(verdicts.keys()-sources.keys())}")
    for v in report.verdicts:
        source = sources[v.region_id]
        if v.adjusted_confidence is None:
            raise ValueError(f"{v.region_id} 缺少 B 的 adjusted_confidence，须显式给出高、中或低")
        disease = _disease(source, v)
        if v.decision != "drop" and (v.decision == "defer" or disease == DEFERRED_DISEASE) and v.adjusted_confidence != "低":
            raise ValueError(f"{v.region_id} 类型待复核，置信度应为低")
        if v.decision == "merge":
            target = verdicts.get(v.merge_into)
            if target is None or target.decision not in {"keep", "adjust", "defer"}:
                raise ValueError(f"{v.region_id} 须直接合并到一个保留项")
            target_source = sources[target.region_id]
            if (disease, v.adjusted_confidence) != (_disease(target_source, target), target.adjusted_confidence):
                raise ValueError(f"{v.region_id} 与合并目标的类型或等级不同，请分别裁决")
            if iou(v.adjusted_bbox or source.bbox, target.adjusted_bbox or target_source.bbox) == 0:
                raise ValueError(f"{v.region_id} 与合并目标没有重叠，请分别裁决")
    return report


def _compose_card(pool, max_regions):
    # pool follows B's priority order. Within a type/grade pair retain its first representative.
    representatives = {}
    for source_id, region in pool:
        representatives.setdefault((region.disease, region.confidence), (source_id, region))
    size = min(max(0, max_regions), MAX_CARD_REGIONS, len(representatives))
    def diversity(group):
        named = [r.disease for _, r in group if r.disease != DEFERRED_DISEASE]
        return len(set(named)), len({r.confidence for _, r in group}), len(named)
    # At most C(10,3) combinations. Stable ties keep the earliest B priorities.
    return max(combinations(representatives.values(), size), key=diversity)


def apply_verdicts(observation: Observation, report: ReviewReport, *,
                   max_regions: int = MAX_CARD_REGIONS,
                   iou_threshold: float = IOU_MERGE_THRESHOLD) -> tuple[list[Region], ReviewStats]:
    validate_review(observation, report)
    sources = {f.region_id: f for f in observation.findings}
    stats = ReviewStats(proposed=len(sources))
    kept = {}
    for verdict in report.verdicts:
        grade = verdict.adjusted_confidence
        stats.grades[grade] += 1
        if verdict.decision == "drop":
            stats.dropped += 1
            continue
        if verdict.decision == "merge":
            continue
        f = sources[verdict.region_id]
        kept[f.region_id] = Region(
            region_id="R1",  # final R numbering is assigned only after selection
            bbox=verdict.adjusted_bbox or f.bbox, disease=_disease(f, verdict),
            confidence=grade, pending_review=grade == "低",
            observation=verdict.adjusted_observation or f.observation,
            impact=verdict.adjusted_impact or f.impact, review_notes=verdict.review_notes,
        )
        stats.adjusted += verdict.decision == "adjust"
        stats.deferred += verdict.decision == "defer"

    def combine(target, box, notes):
        target.bbox = union_box(target.bbox, box)
        target.review_notes = list(dict.fromkeys([*target.review_notes, *notes]))[:2]

    for verdict in report.verdicts:
        if verdict.decision == "merge":
            combine(kept[verdict.merge_into], verdict.adjusted_bbox or sources[verdict.region_id].bbox,
                    verdict.review_notes)
            stats.merged_by_model += 1

    distinct = []
    for source_id, region in kept.items():
        for _, earlier in distinct:
            if (region.disease == earlier.disease and region.confidence == earlier.confidence
                    and iou(region.bbox, earlier.bbox) >= iou_threshold):
                combine(earlier, region.bbox, region.review_notes)
                stats.merged_by_iou += 1
                break
        else:
            distinct.append((source_id, region))

    selected = _compose_card(distinct, max_regions)
    regions = []
    for order, (source_id, region) in enumerate(selected, 1):
        region.region_id = f"R{order}"
        stats.selected_ids.append(source_id)
        regions.append(region)
    stats.final = len(regions)
    stats.truncated = len(distinct) - stats.final
    return regions, stats
