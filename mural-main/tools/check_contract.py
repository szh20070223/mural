#!/usr/bin/env python
# tools/check_contract.py
# =============================================================================
# 数据契约自检脚本
# =============================================================================
# 作用：把"文档里写的规范"变成"能跑的命令"。
# 文档只能靠人自觉，脚本能挡住错误——这是两回事。
#
# 它检查三样东西：
#   1. docs/card.example.json 符不符合 Card 契约（示例跑不过 = 契约写错了）
#   2. data/labels.example.jsonl 符不符合标注规范（格式示例本身要是对的）
#   3. data/labels.jsonl 如果存在，逐行校验，并统计类别分布、负样本数、框的大小
#
# 用法（在项目根目录下执行）：
#   python tools/check_contract.py                        # 检查上面三样
#   python tools/check_contract.py --card path/card.json  # 只查某张卡
#   python tools/check_contract.py --labels data/labels.jsonl
#
# 退出码：0 = 全过；1 = 有硬错误。警告不影响退出码，但会打印出来。
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# 让脚本不管是 `python tools/check_contract.py` 还是在 PyCharm 里点运行，
# 都能 import 到项目自己的 src 包（否则会报 ModuleNotFoundError: src）。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402  （必须放在 sys.path 设置之后）

from src.schemas import DISEASE_TYPES, AnnotationRecord, Card  # noqa: E402


def use_utf8_stdout() -> None:
    """把控制台输出切成 UTF-8。

    Windows 默认控制台编码是 GBK，脚本里的中文（✓ ✗ 之类）会变成乱码。
    PyCharm 和 Windows Terminal 默认就是 UTF-8，切一下最省事。
    如果你的终端反而显示乱了，把这行注释掉即可，不影响校验结果。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # 某些环境（比如输出被重定向到管道）没有 reconfigure，忽略即可
        pass


def rel(path: Path) -> str:
    """打印时尽量用相对路径，短一点好看。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def format_error(error: dict) -> str:
    """把 pydantic 的错误压成一行。

    原始错误信息很长（JSON Schema 一坨），终端里根本看不完，
    这里只留"哪个字段 + 为什么错"，够定位就行。
    """
    location = ".".join(str(part) for part in error["loc"]) or "<根>"
    return f"{location}: {error['msg']}"


# =============================================================================
# 卡片校验
# =============================================================================


def check_card(path: Path) -> bool:
    print(f"\n[卡片] {rel(path)}")

    if not path.is_file():
        print("  ✗ 文件不存在")
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # JSON 语法错比字段错更常见（少个逗号、多个尾逗号），单独报出来
        print(f"  ✗ JSON 语法错误：第 {exc.lineno} 行第 {exc.colno} 列 —— {exc.msg}")
        return False

    try:
        card = Card.model_validate(data)
    except ValidationError as exc:
        print(f"  ✗ 不符合契约，共 {exc.error_count()} 处：")
        for error in exc.errors()[:8]:  # 最多列 8 条，再多就刷屏了
            print(f"     - {format_error(error)}")
        return False

    print("  ✓ 契约校验通过")
    print(
        "    区域："
        + "、".join(f"{region.region_id}={region.disease}" for region in card.regions)
    )
    print(f"    建议：{len(card.recommendations)} 条   局限：{len(card.limitations)} 条")
    print(f"    复核状态：{card.review_status}（复核人：{card.attribution.reviewer or '未填'}）")
    print(f"    模型：{', '.join(card.provenance.models)}   提示词：{card.provenance.prompt_version}")

    # ---- 软检查：不合法但不算错，只是很可能不好看 / 不加分 ----
    # 这些是"能过校验但拿不到分"的典型情况，早点提示比事后返工便宜。
    warnings: list[str] = []

    for region in card.regions:
        ratio = region.bbox.area_ratio()
        if ratio > 0.8:
            # 注意：这不是必然的错误。裂隙、褪色这类"遍布型"病害，
            # 本来就该用一个几乎覆盖全图的大框；只有局部型病害
            # （起甲、水渍、脱落）框这么大才说明标得不够细。
            warnings.append(
                f"{region.region_id} 的框占全图 {ratio:.0%}，定位意义有限——"
                "遍布型病害（裂隙、褪色、粉化）可以这样标，"
                "局部型病害（起甲、水渍、脱落）建议收紧到具体范围"
            )
        if ratio < 0.002:
            warnings.append(f"{region.region_id} 的框只有 {ratio:.2%}，可能小到看不见")
        if not region.review_notes:
            # 复核点是"结果质量"这个维度的得分点，空着很可惜
            warnings.append(f"{region.region_id} 没写人工复核点")

    for index, rec in enumerate(card.recommendations, start=1):
        if not rec.risk.strip():
            warnings.append(f"建议 {index} 的风险一栏是空的")
        if not rec.needs_human_review:
            # 没有任何一条建议应该绕过人工复核，这是赛题的原话
            warnings.append(f"建议 {index} 标了不需要人工复核，请确认是否有意为之")

    if not card.attribution.human_review_required:
        warnings.append("attribution.human_review_required 是空的，赛题明确要求列出需人工复核的内容")

    for message in warnings:
        print(f"  ! 提醒：{message}")

    return True


# =============================================================================
# 标注集校验
# =============================================================================


def check_labels(path: Path) -> bool:
    print(f"\n[标注] {rel(path)}")

    if not path.is_file():
        print("  ✗ 文件不存在")
        return False

    # 示例文件里的 image_id 是编的，不要求真有图；
    # 真实标注文件里每一行都必须对应 data/images/ 下的一张图。
    is_example = path.name.endswith(".example.jsonl")

    lines = path.read_text(encoding="utf-8").splitlines()
    total = 0                 # 有效记录数
    negatives = 0             # 负样本数
    unassessable = 0          # 无法判读的（既不是负样本，也不该算正样本）
    failed = 0                # 解析/校验失败的行数
    disease_counter: Counter[str] = Counter()   # 每类病害出现了几次
    area_ratios: list[float] = []               # 框面积占比，用来看框是不是画太大
    missing_images: list[str] = []              # 找不到图片文件的 image_id

    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # 允许空行，方便手动编辑

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failed += 1
            print(f"  ✗ 第 {lineno} 行 JSON 语法错误：{exc.msg}")
            continue

        try:
            record = AnnotationRecord.model_validate(raw)
        except ValidationError as exc:
            failed += 1
            image_id = raw.get("image_id", "?") if isinstance(raw, dict) else "?"
            # 一行只报第一条错误，避免一个错误刷十几行
            print(f"  ✗ 第 {lineno} 行（{image_id}）：{format_error(exc.errors()[0])}")
            continue

        total += 1
        if record.assessment == "无法判断":
            unassessable += 1
        elif record.negative:
            negatives += 1
        for region in record.regions:
            disease_counter[region.disease] += 1
            area_ratios.append(region.bbox.area_ratio())

        # 图片是否存在（只对真实标注文件检查）
        if not is_example:
            matches = list((ROOT / "data" / "images").glob(f"{record.image_id}.*"))
            if not matches:
                missing_images.append(record.image_id)

    if failed:
        print(f"  ✗ {failed} 行未通过校验（共 {len([x for x in lines if x.strip()])} 行）")
    else:
        print("  ✓ 全部行解析通过")

    if not total:
        print("  ✗ 没有任何有效记录")
        return False

    # ---- 统计信息：直接回答"这份数据够不够用" ----
    # 「无法判读」单独列出来：它既不是负样本，也不进召回/误报统计，
    # 混进去会让误报率失真（看不清 ≠ 确认没病害）。
    positive = total - negatives - unassessable
    print(
        f"    图片：{total} 张（正样本 {positive}，负样本 {negatives}，"
        f"无法判读 {unassessable}）"
    )
    print(f"    标注框：{len(area_ratios)} 个")

    if disease_counter:
        distribution = "  ".join(
            f"{disease}×{disease_counter[disease]}" for disease in DISEASE_TYPES
        )
        print(f"    类别分布：{distribution}")

    if area_ratios:
        average = sum(area_ratios) / len(area_ratios)
        print(f"    框平均面积占比：{average:.1%}（超过 60% 基本可以断定框画大了）")

    # ---- 分层验收：按 docs/annotation.md 第 7 节的标准提示 ----
    if not is_example:
        if negatives < 20:
            print(f"  ! 提醒：负样本只有 {negatives} 张，规范要求 ≥20")
        # 病害白名单只有 8 个真实病害；"待判读"是审查环节的结论，
        # 用了独立的类型（ObservationDisease），不会出现在标注集里。
        for disease in DISEASE_TYPES:
            if disease_counter[disease] < 3:
                print(f"  ! 提醒：{disease} 只有 {disease_counter[disease]} 个框，规范要求 ≥3")
        if missing_images:
            print(f"  ✗ 有 {len(missing_images)} 张图的文件找不到：")
            for image_id in missing_images[:5]:
                print(f"     - data/images/{image_id}.*")
            return False

    return failed == 0


# =============================================================================
# 入口
# =============================================================================


def main() -> int:
    use_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="校验卡片契约与标注格式是否符合 docs/ 里的规范"
    )
    parser.add_argument(
        "--card",
        type=Path,
        help="只校验指定的卡片 JSON；不填则默认校验 docs/card.example.json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="只校验指定的标注 JSONL；不填则默认校验示例文件 + data/labels.jsonl（若存在）",
    )
    args = parser.parse_args()

    results: list[bool] = []

    if args.card or not args.labels:
        target = args.card or (ROOT / "docs" / "card.example.json")
        results.append(check_card(target))

    if args.labels or not args.card:
        if args.labels:
            results.append(check_labels(args.labels))
        else:
            results.append(check_labels(ROOT / "data" / "labels.example.jsonl"))
            real_labels = ROOT / "data" / "labels.jsonl"
            if real_labels.is_file():
                results.append(check_labels(real_labels))

    print()
    if all(results):
        print("全部通过。")
        return 0

    print(f"有 {results.count(False)} 项未通过，见上面的 ✗ 行。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
