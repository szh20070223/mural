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

# "待判读"：异常存在但类型判断不了。它不是病害，是审查环节的结论。
DEFERRED_DISEASE = "待判读"

# 卡片上最多留几处"待判读"。
# 三处里两处写"不知道"，读者会觉得这一页什么都没看出来——
# 这是展示问题，不是证据问题，所以由代码统一控制，不交给模型自觉。
MAX_DEFERRED_ON_CARD = 1

# 例外：如果卡片上"有明确类型"的区域不到这个数，允许再补一处待判读，
# 让卡片保持 3 处（而不是缩成 2 处）。
#
# 规则合起来是：
#   有 2 个以上明确病害 → 待判读最多 1 个（理想形态：两个不同 + 一个待判读）
#   只有 1 个明确病害   → 允许 1 个明确 + 2 个待判读，卡片仍凑满 3 处
#   一个明确病害都没有 → 只保留 1 处待判读（整页"不知道"没有意义）
MIN_TYPED_FOR_NO_FILL = 2

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
    deferred: int = 0          # 位置保留、类型待判的（defer）
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
    observation: str
    impact: str
    review_notes: list[str] = field(default_factory=list)
    # 类型有倾向但未核实 -> 卡片上写成"X（待复核）"
    pending_review: bool = False


def validate_review(observation: Observation, report: ReviewReport) -> ReviewReport:
    """检查 B 的裁决和 A 的候选对不对得上。

    三件事，缺一不可——这是"B 是裁决者"而不是"B 是过滤器"的前提：
      1. 裁决里的编号必须真的在候选里（B 编出 C7 是常见的）
      2. 编号不能重复
      3. **每一条候选都必须有裁决**。漏一条就报错，触发一次格式纠正重试。

    为什么第 3 条要用强约束：漏裁的候选会被"保守保留"混进卡片，
    看起来一切正常，但那条其实**没被审过**。B 的职责是逐条裁决，
    少答一条就等于审查没做完——这种半成品不该进入下游。
    """
    known = [finding.region_id for finding in observation.findings]
    judged = [verdict.region_id for verdict in report.verdicts]

    unknown = sorted(set(judged) - set(known))
    if unknown:
        raise ValueError(
            f"裁决里出现了不存在的候选编号：{unknown}；A 实际只给了 {known}"
        )

    missing = [region_id for region_id in known if region_id not in set(judged)]
    if missing:
        raise ValueError(
            f"漏裁了候选 {missing}；必须对 {known} 每一条都给出裁决，一条都不能少"
        )

    return report


def _bbox_center_distance(a: BoundingBox, b: BoundingBox) -> float:
    """两个框中心点的距离。归一化坐标下，最大值约 1.41（对角线）。"""
    ax, ay = (a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2
    bx, by = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _diversity_score(candidate: _Candidate, chosen: list[_Candidate]) -> tuple:
    """给"下一个该上卡谁"打分，越大越优先。

    排序键（从主到次）：
      0. 是否带类型                   —— "X（待复核）"比"待判读"信息量大，不该排在它后面
      1. 病害类型是否与已选的不同     —— 一张卡全是一种病，读者会以为图上只有这一种问题
      2. 置信度是否与已选的不同       —— 同类型但一处"中"一处"待复核·低"，是不同风险等级，可并列
      3. 与已选区域的中心距离         —— 越远越不像是同一处的重复框选
      4. 面积                         —— 前面都持平时，大的更值得上卡

    注意这只是"在都站得住的候选之间怎么排"，不改变任何一条的判断结果。
    """
    has_type = 0 if candidate.disease == DEFERRED_DISEASE else 1
    same_type = any(item.disease == candidate.disease for item in chosen)
    same_confidence = any(item.confidence == candidate.confidence for item in chosen)
    distance = min(_bbox_center_distance(candidate.box, item.box) for item in chosen)
    return (
        has_type,
        0 if same_type else 1,
        0 if same_confidence else 1,
        distance,
        candidate.box.area_ratio(),
    )


def _compose_card(
    candidates: list[_Candidate],
    *,
    max_regions: int = MAX_CARD_REGIONS,
) -> list[_Candidate]:
    """卡片编排：从 B 排好序的候选里挑出最终上卡的 1-3 个。

    B 负责"哪些站得住、谁更重要"，这一步负责"卡片上放哪几个"。
    后者是展示问题——用代码做比让模型记住更可靠，也不会污染判断。

    规则：
    1. **B 排第一的那条一定进卡**，它是"最要紧的一处"，不由展示规则决定。
    2. 后面的位置按 `_diversity_score` 贪心挑：类型 > 置信度 > 位置分离度 > 面积。
    3. **证据强度优先于多样性**：只要还有"中"及以上的候选，就不会为了凑类型
       去选一个"低"的；"低"只在没有更强的可选时才进来。
    4. **待判读最多 1 处**（纯未知，连倾向都没有的那种）。
       "X（待复核）"不算待判读，它带着类型倾向。
    5. 例外：有类型的区域不足 2 个时，允许补一处待判读凑满 max_regions，
       否则卡片会缩成 2 处。
    """
    if not candidates:
        return []

    chosen: list[_Candidate] = [candidates[0]]
    chosen_ids = {candidates[0].source.region_id}
    deferred = 1 if chosen[0].disease == DEFERRED_DISEASE else 0

    while len(chosen) < max_regions:
        pool = [
            item
            for item in candidates
            if item.source.region_id not in chosen_ids
            and not (
                item.disease == DEFERRED_DISEASE and deferred >= MAX_DEFERRED_ON_CARD
            )
        ]
        if not pool:
            break

        best = max(pool, key=lambda item: _diversity_score(item, chosen))
        chosen.append(best)
        chosen_ids.add(best.source.region_id)
        if best.disease == DEFERRED_DISEASE:
            deferred += 1

    # 明确病害不足 2 个时，允许再补待判读凑满（"放开到 3 处"的规则）
    typed = sum(1 for item in chosen if item.disease != DEFERRED_DISEASE)
    if 1 <= typed < MIN_TYPED_FOR_NO_FILL and len(chosen) < max_regions:
        for candidate in candidates:
            if len(chosen) >= max_regions:
                break
            if candidate.source.region_id in chosen_ids:
                continue
            if candidate.disease != DEFERRED_DISEASE:
                continue
            chosen.append(candidate)
            chosen_ids.add(candidate.source.region_id)

    return chosen


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

    # B 的输出顺序就是优先级。
    # A 排出的是"哪里最可疑"，B 排出的是"哪里最该进卡"——这两件事不一样，
    # 而最终卡片应该听 B 的：它看过图，还做过取舍。
    priority = {
        verdict.region_id: index for index, verdict in enumerate(report.verdicts)
    }

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
                    observation=finding.observation,
                    impact=finding.impact,
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

        # keep / adjust / defer
        # 文字先取裁决里的版本，没有就沿用 A 的。
        # 这一句是"B 的改判能落到卡片上"的关键：以前这里直接取
        # candidate.source.observation，B 说什么都传不下去。
        observation = verdict.adjusted_observation or finding.observation
        impact = verdict.adjusted_impact or finding.impact

        if verdict.decision == "defer":
            # 类型判不了，但没必要把"它倾向于是什么"也一起丢掉。
            # 第一位观察员给过类型的话就保留，标成"待复核"——
            # 卡片上写"颜料层脱落（待复核）"，比单写"待判读"信息量大得多。
            # 只有第一位观察员自己都写的是"待判读"时，才真的是纯未知。
            deferred_disease = verdict.adjusted_disease or finding.disease
            candidate = _Candidate(
                source=finding,
                box=verdict.adjusted_bbox or finding.bbox,
                disease=deferred_disease,
                confidence=verdict.adjusted_confidence or "低",
                observation=observation,
                impact=impact,
                review_notes=list(verdict.review_notes),
                pending_review=deferred_disease != DEFERRED_DISEASE,
            )
            stats.deferred += 1
        else:
            candidate = _Candidate(
                source=finding,
                box=verdict.adjusted_bbox or finding.bbox,
                disease=verdict.adjusted_disease or finding.disease,
                confidence=verdict.adjusted_confidence or finding.confidence,
                observation=observation,
                impact=impact,
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
                        observation=verdict.adjusted_observation or source.observation,
                        impact=verdict.adjusted_impact or source.impact,
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
            # 必须是**同一种病害**才合并。
            # 这条判断不是形式主义：裂缝穿过颜料脱落区时，两个框必然大面积重叠，
            # 不分类型合并就会把其中一条的类型直接抹掉——
            # 卡片上剩一条，看起来很正常，但类型可能正好错了。
            if (
                existing.disease == candidate.disease
                and iou(existing.box, candidate.box) >= iou_threshold
            ):
                existing.box = union_box(existing.box, candidate.box)
                existing.confidence = _stronger(existing.confidence, candidate.confidence)
                for note in candidate.review_notes:
                    if note not in existing.review_notes and len(existing.review_notes) < 2:
                        existing.review_notes.append(note)
                stats.merged_by_iou += 1
                break
        else:
            deduped.append(candidate)

    # ---- ④ 按 B 的优先级排，再按卡片展示规则挑最终 1-3 个 ----
    # 排序放在去重之后：合并会把两条并成一条，先排序再合并会白排。
    deduped.sort(key=lambda item: priority.get(item.source.region_id, 10_000))

    before = len(deduped)
    deduped = _compose_card(deduped, max_regions=max_regions)
    if before > len(deduped):
        stats.truncated = before - len(deduped)

    regions: list[Region] = []
    for order, candidate in enumerate(deduped, start=1):
        regions.append(
            Region(
                region_id=f"R{order}",
                disease=candidate.disease,
                bbox=candidate.box,
                observation=candidate.observation,
                impact=candidate.impact,
                confidence=candidate.confidence,
                review_notes=candidate.review_notes[:2],
                pending_review=candidate.pending_review,
                ai_generated=True,
            )
        )

    stats.final = len(regions)
    return regions, stats
