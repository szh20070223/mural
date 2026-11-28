# src/card_builder.py
# =============================================================================
# 组装卡片：把 A/B/C 三段的产物拼成一张 Card
# =============================================================================
# 这一层刻意做成**纯函数、没有模型参与**。原因：
#   卡片里最不能出错的两块是 provenance（来源披露）和 attribution（AI/人工归属）。
#   这两块如果让模型写，它会编——编一个不存在的图片来源、编一个"已人工复核"。
#   赛题明确写了"隐瞒第三方工具""伪造记录"会被取消成绩，
#   所以这两块必须由代码按事实填，一个字都不交给模型。
#
# 模型只负责它擅长的：区域怎么描述（A）、留哪几个（B）、建议怎么写（C）。
# =============================================================================

from __future__ import annotations

import re

from src.schemas import (
    Attribution,
    Card,
    Observation,
    Provenance,
    Region,
    ReportDraft,
)

# 人工复核清单最多列几条。卡片空间有限，列太多等于没重点。
MAX_REVIEW_ITEMS = 6


def _build_attribution(regions: list[Region]) -> Attribution:
    """按事实填 AI/人工归属。

    三个列表的语义必须分清楚，答辩时就是靠它们回答"哪句话是谁写的"：
      - ai_generated        ：确定由模型产出的内容
      - human_review_required：必须人看过才能用的部分
      - human_reviewed      ：已经人看过的（这里留空——确实还没人复核）
    """
    region_ids = "、".join(region.region_id for region in regions)

    ai_generated = [
        f"区域 {region_ids} 的位置框、观察描述与影响判断（A 观察官生成）",
        "区域的保留与剔除、人工复核提示（B 审查官生成）",
        "总述与保护建议的文字（C 成稿生成）",
    ]

    # 复核点来自 B 给每个区域写的 review_notes —— 那是全链路里最具体的
    # "人应该做什么"，直接搬过来，不重新措辞
    review_items: list[str] = []
    for region in regions:
        for note in region.review_notes:
            if note not in review_items:
                review_items.append(note)

    # 再补两条与具体区域无关、但必须有人负责的
    review_items.append("病害类型的最终判定（AI 给出的是疑似判断，不是鉴定结论）")
    review_items.append("图片来源与授权信息的核对")

    return Attribution(
        ai_generated=ai_generated,
        human_review_required=review_items[:MAX_REVIEW_ITEMS],
        # 留空是有意的：现在确实没有任何人复核过。
        # 标成"已复核"而实际没有，比不标更糟——那叫伪造记录。
        human_reviewed=[],
        reviewer="",
    )


def _dedupe_limitations(items: list[str]) -> list[str]:
    """按"开头十来个字"去重。

    同一条局限经常被三拨人换个说法各写一遍：
      A："粉化需近距离观察表面质感，仅凭此图无法确认"
      B："粉化需近距离观察表面质感，当前分辨率下无法判断"
    精确匹配抓不住，按开头去重能抓住绝大多数重复。

    只用于卡片的显示层——observation.json / review.json 里的原文一个字不动，
    想追溯完整表述去那两个文件。
    """
    seen: set[str] = set()
    kept: list[str] = []

    for item in items:
        text = item.strip()
        if not text:
            continue
        # 去掉标点再取前 12 字做 key，避免"逗号位置不同"被当成两条
        key = re.sub(r"[\s，。；、：,.;:（）()「」]", "", text)[:12]
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)

    return kept


def build_card(
    *,
    image_id: str,
    regions: list[Region],
    draft: ReportDraft,
    observation: Observation,
    image_source: str,
    image_license: str,
    models: list[str],
    prompt_versions: dict[str, str],
    tools: list[str],
    run_id: str,
    generated_at: str,
    scene_note: str = "",
    platform: str = "阿里云百炼（DashScope）",
    upstream_limitations: list[str] | None = None,
) -> Card:
    """把三段产物拼成一张 Card。

    参数里的 image_source / image_license 必须由调用方提供真实值：
    命题方材料就写"命题方提供"，自摄就写拍摄信息。
    这个函数不提供默认值，就是为了逼调用方想清楚"这张图哪来的"。
    """
    if not regions:
        # 一张没有任何区域的卡片不符合赛题要求（必须标 1-3 处），
        # 与其生成一张空卡，不如在这里明确失败，
        # 让上层去决定是重跑、换图，还是出"未发现明显病害"的变体版式。
        raise ValueError("没有可用于卡片的区域：全部候选都被审查剔除了")

    if len(regions) > 3:
        raise ValueError(f"卡片最多 3 个区域，收到 {len(regions)} 个")

    # 局限必须以**上游**为主，C 只能补充，不能替换。
    #
    # 旧写法：draft.limitations or observation.limitations
    #   C 只要给出一条，A 的"缺侧光、无法判断起甲"就整个消失——
    #   而那句恰恰是"AI 能力边界"最硬的证据，直接影响评分。
    #
    # 现在的规则：上游（A 的整体局限 + B 的总体意见）在前，C 的补充在后，
    # 去重后最多 3 条。超过时优先保留上游、截掉 C 的——
    # 因为上游是"看图的一线判断"，C 只是文字整理。
    #
    # 不再把 6 条压成 3 组长句：那样虽然一条不丢，但每组长达一百多字，
    # 复核的人根本读不完——"写全"不等于"说明清楚"。
    upstream = (
        list(upstream_limitations)
        if upstream_limitations is not None
        else list(observation.limitations)
    )
    limitations = _dedupe_limitations([*upstream, *draft.limitations])[:3]

    provenance = Provenance(
        image_source=image_source,
        image_license=image_license,
        scene_note=scene_note,
        platform=platform,
        models=models,
        prompt_version=" / ".join(
            f"{key.upper()}:{value}" for key, value in prompt_versions.items()
        ),
        tools=tools,
        generated_at=generated_at,
        run_id=run_id,
    )

    return Card(
        schema_version="1.0",
        image_id=image_id,
        summary=draft.summary,
        regions=regions,
        recommendations=draft.recommendations,
        limitations=limitations[:3],
        provenance=provenance,
        attribution=_build_attribution(regions),
        # 默认"待人工复核"。只有真的有人看过、并在 attribution 里写清楚是谁，
        # 才允许改这个状态——schema 里有硬校验拦着。
        review_status="待人工复核",
    )
