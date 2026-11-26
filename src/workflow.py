# src/workflow.py
# =============================================================================
# 编排：一次完整运行的流程
# =============================================================================
# 现在的流程（A + B）：
#   读图
#     -> A 观察官：找出最多 5 个候选区域（带坐标框）
#     -> B 审查官：看着原图逐条裁决，砍到 1-3 个
#     -> 确定性后处理：合并重叠、截断、重新编号
#     -> 落盘 + 写日志
#
# 还差 C（建议）和渲染（一页 PDF）。接的时候直接在 B 后面插一步就行，
# 所以这里刻意保持线性、每一步一个函数调用。
#
# 自修复重试：
#   模型偶尔会输出不能解析的内容（多写一段解释、坐标越界、漏掉字段）。
#   直接报错让演示中断是最差的选择。这里给它第二次机会，
#   而且是把"它自己上一次的输出 + 具体错在哪"一起喂回去——
#   重新问一遍它大概率还是同样的错。
# =============================================================================

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from src.agents.report_agent import ReportAgent
from src.agents.review_agent import ReviewAgent
from src.agents.vision_agent import VisionAgent
from src.card_builder import build_card
from src.config import (
    CARD_DIR,
    OBSERVATION_DIR,
    PROJECT_ROOT,
    QWEN_BASE_URL,
    QWEN_REVIEW_CROPS,
    RAW_DIR,
    REVIEW_DIR,
)
from src.qwen_client import ModelReply, QwenClient
from src.render import render_card
from src.result_parser import parse_any, parse_report, parse_review
from src.review_rules import apply_verdicts, validate_review
from src.schemas import Region, ReviewReport
from src.trace import RunTrace, now_iso

# 最多尝试几次。2 次是权衡：第一次失败后带着错误信息重问，成功率很高；
# 再加就是提示词或模型能力的问题了，多花的调用换不来质量。
MAX_ATTEMPTS = 2


def _image_size(path: Path) -> tuple[int | None, int | None]:
    """读图片宽高，只用于写日志。

    PIL 万一不可用，不应该让整个流程挂掉——尺寸只是参考信息。
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def _platform_name() -> str:
    """按实际配置的 endpoint 推断平台名，**不写死**。

    如果 QWEN_BASE_URL 不是 dashscope.aliyuncs.com（比如某个第三方中转），
    卡片上就应该如实写那台主机。赛题里"隐瞒第三方平台/工具"是取消成绩
    级别的问题，而"顺手写死成百炼"恰恰是最容易犯的那个错。
    """
    host = urlsplit(QWEN_BASE_URL).hostname or ""
    if host.startswith("dashscope.") and host.endswith("aliyuncs.com"):
        return "阿里云百炼（DashScope）"
    return f"OpenAI 兼容服务（{host or '未知主机'}）"


def _run_with_retry(
    *,
    trace: RunTrace,
    stage: str,
    call: Callable[[list[dict] | None], ModelReply],
    parse: Callable[[str], Any],
    correction: Callable[[str], str],
    raw_stem: str,
) -> tuple[Any, ModelReply, int]:
    """调用一次模型 + 解析，失败就带着错误信息重试一次。

    把这段抽出来的原因：A 和 B 的重试逻辑一模一样，
    差别只在"传什么图、怎么解析、怎么纠正"。复制两遍的话，
    以后改重试策略要改两处，早晚会不一致。
    """
    history: list[dict] | None = None
    attempts = 0
    reply: ModelReply | None = None

    while attempts < MAX_ATTEMPTS:
        attempts += 1

        try:
            reply = call(history)
        except Exception as exc:
            # 网络/鉴权/超时归 QwenClient 那层处理，这里只记录再抛
            trace.log(
                "model_error",
                stage=stage,
                attempt=attempts,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        trace.log(
            "model_call",
            stage=stage,
            attempt=attempts,
            model=reply.model,
            latency_ms=reply.latency_ms,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            finish_reason=reply.finish_reason,
            response_chars=len(reply.content),
            # 记的是"关掉了没有"，不是"发没发参数"——
            # 之前那个字段名让人以为 true = 开着思考，实际正相反
            thinking="off" if reply.thinking_disabled else "on",
        )

        # 原始回复一定要留：出问题时要看的是"模型到底说了什么"，
        # 而不是解析之后的东西。
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{raw_stem}.attempt{attempts}.txt").write_text(
            reply.content, encoding="utf-8"
        )

        try:
            # finish_reason 不是 stop，说明输出被截断或异常结束了。
            # 这时 JSON 通常也是断的，但如果直接往下解析，报出来的会是
            # "第 37 行解析失败"——排查方向完全错。先报真正的原因。
            if reply.finish_reason not in (None, "stop"):
                raise ValueError(
                    f"模型未正常结束（finish_reason={reply.finish_reason}），已经截断，需要完整 JSON"
                )
            return parse(reply.content), reply, attempts
        except ValueError as exc:
            trace.log("parse_fail", stage=stage, attempt=attempts, error=str(exc))

            if attempts >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"[{stage}] 模型连续 {attempts} 次输出都无法解析，"
                    f"最后一次的错误：{exc}\n"
                    f"原始回复见：{RAW_DIR / f'{raw_stem}.attempt{attempts}.txt'}"
                ) from exc

            history = [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": correction(str(exc))},
            ]
            trace.log("retry", stage=stage, attempt=attempts + 1, reason=str(exc))

    # 循环里要么 return 要么 raise，走不到这里；留个保险
    raise RuntimeError(f"[{stage}] 重试循环异常结束")


def run_vision_workflow(
    image_path: str,
    prompt_file: str | None = None,
    review: bool = True,
    review_prompt_file: str | None = None,
    compose: bool = True,
    report_prompt_file: str | None = None,
    render: bool = True,
    image_source: str = "命题方提供的比赛材料",
    image_license: str = "仅限本次比赛使用",
    scene_note: str = "",
) -> dict:
    """跑一次完整流程（A 观察 → B 审查 → C 成稿 → 卡片 → PDF）。

    参数：
        image_path          ：图片路径，相对路径按项目根目录解析
        prompt_file         ：可选，A 的提示词（评测时用来对比 v1/v2）
        review              ：是否跑 B 审查。调 A 的提示词时可以关掉省钱
        review_prompt_file  ：可选，B 的提示词
        compose             ：是否跑 C 成稿并组装卡片。只关心候选时可关
        report_prompt_file  ：可选，C 的提示词
        render              ：是否渲染一页 PDF。关掉只出 card.json
        image_source        ：图片来源，会原样写进卡片的来源区
        image_license       ：授权说明，同上
        scene_note          ：命题方给的场景说明，同上

    返回：
        {
          "image_id", "stage", "attempts", "result", "meta",
          "output_path", "trace_path", "run_id",
          "review": {...} 或 None,
          "card": {"json_path", "pdf_path", "summary", "recommendations"} 或 None
        }

    注意 image_source / image_license 没有真正的默认值可言：
    这里给的默认值只是"命题方比赛材料"的假设，自摄或公开数据集必须显式传。
    来源写错的后果是"规范合规"这个维度直接失分，别依赖默认值。
    """

    # ---- 0. 路径处理 ----
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")

    image_id = path.stem
    width, height = _image_size(path)

    # 先确认图片真能解码，再开始花钱调模型。
    # 不然会拿着一张坏图跑完 A、B、C 三次调用，最后渲染时才炸——
    # 那时已经付过费了，而且报错指向的是渲染器，方向完全错。
    if width is None or height is None:
        raise ValueError(f"图片无法解码（可能已损坏或格式不支持）：{path}")

    # ---- 1. 开日志 ----
    trace = RunTrace(image_id=image_id)
    trace.log(
        "run_start",
        image_path=str(path),
        image_width=width,
        image_height=height,
        review_enabled=review,
    )

    # ---- 2. 初始化 ----
    client = QwenClient()
    agent = VisionAgent(client, prompt_path=prompt_file)

    trace.log(
        "agent_ready",
        stage="a",
        model=client.model,
        prompt_file=agent.prompt_path.name,
        prompt_version=agent.version,
        prompt_sha256=agent.prompt_sha,
        thinking_requested=client.thinking_enabled,
    )

    # ---- 3. A：观察 ----
    # _run_with_retry 返回 (解析结果, 回执, 尝试次数)，
    # 而 parse_any 本身返回的是 (阶段, 对象, 元信息)，所以要解两层。
    parsed, a_reply, a_attempts = _run_with_retry(
        trace=trace,
        stage="a",
        call=lambda history: agent.run(path, history=history),
        parse=parse_any,
        correction=agent.correction_message,
        raw_stem=image_id,
    )
    stage, observation_or_analysis, meta = parsed

    trace.log(
        "parse_ok",
        stage="a",
        attempt=a_attempts,
        result_stage=stage,
        raw_findings=meta.raw_findings,
        kept=meta.kept,
        truncated=meta.truncated,
        bbox_rescaled=meta.bbox_rescaled,
    )

    # 以本地文件名为准：模型可能把 image_id 写成中文描述或带扩展名
    observation_or_analysis.image_id = image_id

    # ---- 4. 落盘 A 的结果 ----
    if stage == "observation":
        OBSERVATION_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OBSERVATION_DIR / f"{image_id}.json"
    else:
        # v1 的老位置，保持兼容，方便和以前的结果对比
        output_path = PROJECT_ROOT / "output" / "results" / f"{image_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        observation_or_analysis.model_dump_json(indent=2), encoding="utf-8"
    )

    # ---- 5. B：审查（v1 的输出没有坐标框，没法审查，直接跳过）----
    review_payload: dict | None = None
    final_regions: list[Region] = []
    # 提前声明：C 阶段要用 B 的 reviewer_notes（并进局限性）。
    # 不能只在 if 分支里定义——那会变成"靠 final_regions 非空隐式保证
    # review_report 存在"的写法，以后一改流程就是 NameError。
    review_report: ReviewReport | None = None

    if review and stage == "observation":
        observation = observation_or_analysis

        if not observation.findings:
            # A 什么都没找到，B 没有可裁的东西，不用花这次钱
            trace.log("review_skipped", reason="no_candidates")
        else:
            reviewer = ReviewAgent(client, prompt_path=review_prompt_file)

            trace.log(
                "review_start",
                stage="b",
                prompt_file=reviewer.prompt_path.name,
                prompt_version=reviewer.version,
                prompt_sha256=reviewer.prompt_sha,
                candidates=len(observation.findings),
                crops_sent=QWEN_REVIEW_CROPS,
            )

            review_report, _review_reply, review_attempts = _run_with_retry(
                trace=trace,
                stage="b",
                call=lambda history: reviewer.run(path, observation, history=history),
                # 解析之后立刻做一次跨阶段校验：B 编出来的编号在这里就炸，
                # 而不是等到 apply_verdicts 里被静默忽略
                parse=lambda text: validate_review(observation, parse_review(text)),
                correction=reviewer.correction_message,
                raw_stem=f"{image_id}.review",
            )

            review_report.image_id = image_id

            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            review_path = REVIEW_DIR / f"{image_id}.json"
            review_path.write_text(
                review_report.model_dump_json(indent=2), encoding="utf-8"
            )

            # 确定性后处理：合并、去重、截断、编号
            regions, stats = apply_verdicts(observation, review_report)
            final_regions = regions

            # 把"候选 → 裁决 → 最终区域"的完整映射单独落一份，
            # 和原始裁决分开存：原始裁决是模型说的，这份是代码处理后的，
            # 两者放在一起才解释得清"为什么最后剩这三条"。
            applied_path = REVIEW_DIR / f"{image_id}.applied.json"
            applied_path.write_text(
                json.dumps(
                    {
                        "report": json.loads(review_report.model_dump_json()),
                        "regions": [json.loads(region.model_dump_json()) for region in regions],
                        "stats": asdict(stats),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            trace.log("review_applied", stage="b", **asdict(stats))

            if not regions:
                # 全部被砍掉了。这在数据上是合法的（这张图确实没问题），
                # 但卡片契约要求至少 1 个区域，所以这里必须显式记一笔，
                # 后面做 C 的时候要决定：是出"未发现明显病害"的变体卡，
                # 还是回到 A 请它重新看。现在先如实暴露，不假装成功。
                trace.log("review_all_dropped", stage="b")

            review_payload = {
                "attempts": review_attempts,
                "report": json.loads(review_report.model_dump_json()),
                "regions": [json.loads(region.model_dump_json()) for region in regions],
                "stats": asdict(stats),
                "output_path": str(review_path),
                "applied_path": str(applied_path),
            }

    # ---- 6. C：成稿 + 组装卡片 + 渲染 ----
    card_payload: dict | None = None

    if compose and final_regions:
        observation = observation_or_analysis
        reporter = ReportAgent(client, prompt_path=report_prompt_file)

        # 上游局限合并传给 C，而不是只给 A 那一份：
        #   A 的整体局限 + B 的总体意见
        # （每条候选自带的局限**不往卡上放**：它用的是 C1/C2 编号，
        #   而卡片上只有 R1–R3，读者对不上；完整内容都在 observation.json 里。）
        # 旧写法是 draft.limitations or A.limitations——C 一开口，
        # A 的"缺侧光无法判断起甲"就没了，而那正是 AI 边界最硬的证据。
        upstream_limitations = list(
            dict.fromkeys(
                [
                    *observation.limitations,
                    *(review_report.reviewer_notes if review_report else []),
                ]
            )
        )

        trace.log(
            "report_start",
            stage="c",
            prompt_file=reporter.prompt_path.name,
            prompt_version=reporter.version,
            prompt_sha256=reporter.prompt_sha,
            regions=len(final_regions),
            upstream_limitations=len(upstream_limitations),
        )

        draft, _report_reply, report_attempts = _run_with_retry(
            trace=trace,
            stage="c",
            call=lambda history: reporter.run(
                final_regions, upstream_limitations, history=history
            ),
            # 把最终区域的编号传进去：C 引用了不存在的 R5 会被当场拦住
            parse=lambda text: parse_report(
                text, {region.region_id for region in final_regions}
            ),
            correction=reporter.correction_message,
            raw_stem=f"{image_id}.report",
        )

        card = build_card(
            image_id=image_id,
            regions=final_regions,
            draft=draft,
            observation=observation,
            image_source=image_source,
            image_license=image_license,
            scene_note=scene_note,
            models=[client.model],
            prompt_versions={
                "a": agent.version,
                "b": reviewer.version,
                "c": reporter.version,
            },
            tools=["Pillow（卡片渲染）", f"Python {sys.version.split()[0]}"],
            run_id=trace.run_id,
            generated_at=now_iso(),
            upstream_limitations=upstream_limitations,
            platform=_platform_name(),
        )

        CARD_DIR.mkdir(parents=True, exist_ok=True)
        card_json_path = CARD_DIR / f"{image_id}.json"
        card_json_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

        trace.log(
            "card_built",
            stage="c",
            attempts=report_attempts,
            regions=len(card.regions),
            recommendations=len(card.recommendations),
        )

        # 渲染失败不应该让整次运行作废：card.json 已经落盘了，
        # 至少数据还在，可以事后补渲染。
        pdf_path: str | None = None
        try:
            rendered = render_card(card, path, CARD_DIR / f"{image_id}.pdf")
            pdf_path = str(rendered)
            trace.log("rendered", stage="c", pdf_path=pdf_path)
        except Exception as exc:
            trace.log("render_failed", stage="c", error=f"{type(exc).__name__}: {exc}")

        card_payload = {
            "attempts": report_attempts,
            "summary": card.summary,
            "recommendations": [rec.model_dump() for rec in card.recommendations],
            "json_path": str(card_json_path),
            "pdf_path": pdf_path,
        }

    # ---- 7. 收尾 ----
    trace.log(
        "run_end",
        ok=True,
        stage=stage,
        output_path=str(output_path),
        regions=len(final_regions) if final_regions else len(observation_or_analysis.findings),
        review_ran=review_payload is not None,
        card_built=card_payload is not None,
        a_attempts=a_attempts,
        a_latency_ms=a_reply.latency_ms,
    )

    return {
        "image_id": image_id,
        "stage": stage,
        "attempts": a_attempts,
        "result": json.loads(observation_or_analysis.model_dump_json()),
        "meta": asdict(meta),
        "output_path": str(output_path),
        "trace_path": str(trace.path),
        "run_id": trace.run_id,
        "review": review_payload,
        "card": card_payload,
    }
