# main.py
# =============================================================================
# 命令行入口
# =============================================================================
# 用法：
#   python main.py                                        # 自动用 data/images/ 下第一张
#   python main.py data/images/test_mural1.png
#   python main.py data/images/test_mural1.png --prompt prompts/vision_analysis.txt
#
# 输出说明：
#   这是 A 阶段的产物——若干"候选区域"，带坐标框。
#   最终观察卡上只放 1-3 个，由 B 审查官裁决决定；B/C 还没接。
# =============================================================================

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT
from src.workflow import run_vision_workflow

# 不传图片参数时，默认从 data/images/ 下挑第一张。
# 这样在 PyCharm 里点绿色三角就能跑，不用每次去改 Run Configuration。
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _pick_default_image() -> Path | None:
    """挑一张默认图片。

    按文件名排序，不按修改时间——顺序必须稳定，
    否则演示时点一下运行，跑的可能和上一次不是同一张图。
    """
    if not IMAGE_DIR.is_dir():
        return None

    try:
        entries = list(IMAGE_DIR.iterdir())
    except OSError:
        # 目录存在但读不了（权限问题）。这里不抛异常，
        # 让调用方走"没有可用图片"的提示——报错信息比 traceback 有用得多。
        return None

    candidates = sorted(
        path
        for path in entries
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return candidates[0] if candidates else None


def _resolve_image(raw: str | None) -> Path:
    """把命令行参数或默认图片解析成一个路径。"""
    if raw:
        return Path(raw)

    picked = _pick_default_image()
    if picked is None:
        raise SystemExit(
            f"没有传入图片路径，而且 {IMAGE_DIR} 里也没有可用的图片。\n"
            "用法：python main.py data/images/你的图.png"
        )

    print(f"（未传入图片路径，自动使用 {picked.name}）")
    return picked


def _print_observation(outcome: dict) -> None:
    """打印 v2（带坐标框）的结果。"""
    result = outcome["result"]
    findings = result.get("findings", [])

    print("\n分析完成（A 阶段：候选区域）")
    print("图片编号：", outcome["image_id"])
    print(f"候选区域：{len(findings)} 个（卡片最终只取 1-3 个，由 B 审查官裁决）")

    if not findings:
        print("  （模型没有发现值得关注的区域——这是合法结果，不是报错）")

    for finding in findings:
        bbox = finding["bbox"]
        # 坐标是归一化的，直接换算成百分比更好读
        box = ", ".join(f"{value:.0%}" for value in bbox.values())
        print(
            f"\n  {finding['region_id']}  {finding['disease']}"
            f"  置信度:{finding['confidence']}"
        )
        print(f"      位置：[{box}]")
        print(f"      观察：{finding['observation']}")
        print(f"      影响：{finding['impact']}")

    if result.get("limitations"):
        print("\n  看不清的地方：")
        for item in result["limitations"]:
            print(f"    - {item}")

    print("\n总述：", result.get("summary", ""))


def _print_analysis(outcome: dict) -> None:
    """打印 v1（只有文字位置）的结果，用于和旧版本对照。"""
    result = outcome["result"]

    print("\n分析完成（A 阶段 v1：病害清单，无坐标框）")
    print("图片编号：", outcome["image_id"])
    print("分析摘要：", result.get("summary", ""))
    print()

    for finding in result.get("findings", []):
        print(f"  {finding['disease']:<8} {finding['status']:<6} 置信度:{finding['confidence']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="壁画病害视觉分析（A 阶段）")

    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="需要分析的图片路径。省略时会自动使用 data/images/ 下的第一张图",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="指定提示词文件（相对于项目根目录）。"
        "不填则使用 config 里的默认值，可用环境变量 VISION_PROMPT_FILE 覆盖",
    )

    args = parser.parse_args()

    image_path = _resolve_image(args.image)

    try:
        outcome = run_vision_workflow(str(image_path), prompt_file=args.prompt)
    except Exception as exc:
        # 这里把异常原样抛出去：堆栈对排查更有用，
        # 只留一句"分析失败"会把真正的原因藏起来
        print(f"\n分析失败：{exc}")
        raise

    if outcome["stage"] == "observation":
        _print_observation(outcome)
    else:
        _print_analysis(outcome)

    print("\n" + "-" * 60)
    print("尝试次数：", outcome["attempts"])

    meta = outcome["meta"]
    if meta.get("truncated"):
        # 这一行值得盯：连续出现说明提示词没约束住模型，
        # 要靠代码兜底，答辩时也是个可以讲的点
        print(f"注意：模型多给了 {meta['truncated']} 个候选，已被截断")
    if meta.get("bbox_rescaled"):
        print(f"注意：{meta['bbox_rescaled']} 个坐标框按千分制自动换算为 0-1")

    print("结果文件：", outcome["output_path"])
    print("运行日志：", outcome["trace_path"])


if __name__ == "__main__":
    main()
