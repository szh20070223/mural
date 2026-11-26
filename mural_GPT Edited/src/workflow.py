# src/workflow.py
# =============================================================================
# 编排：一次完整运行的流程
# =============================================================================
# A 观察 → B 审查 → 确定性处理 → C 成稿 → 卡片与单页渲染。
# 格式错误携带原回复与校验原因重试一次，网络重试交给 SDK。

from __future__ import annotations

import json
import hashlib
import shutil
import sys
from importlib.metadata import version
from urllib.parse import urlsplit
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.agents.report_agent import ReportAgent
from src.agents.review_agent import ReviewAgent
from src.agents.vision_agent import VisionAgent
from src.card_builder import build_card
from src.config import CARD_DIR, OBSERVATION_DIR, PROJECT_ROOT, RAW_DIR, REVIEW_DIR, INPUT_DIR
from src.config import QWEN_BASE_URL, QWEN_IMAGE_MAX_EDGE, QWEN_REVIEW_CROPS, QWEN_TIMEOUT, QWEN_MAX_RETRIES
from src.qwen_client import ModelReply, QwenClient
from src.render import render_card
from src.result_parser import parse_any, parse_report, parse_review
from src.review_rules import apply_verdicts, validate_review
from src.images import load_image
from src.schemas import Region
from src.trace import RunTrace, now_iso

# 最多尝试几次。2 次是权衡：第一次失败后带着错误信息重问，成功率很高；
# 再加就是提示词或模型能力的问题了，多花的调用换不来质量。
MAX_ATTEMPTS = 2


def _image_size(path: Path) -> tuple[int | None, int | None]:
    """读图片宽高，只用于写日志。

    PIL 万一不可用，不应该让整个流程挂掉——尺寸只是参考信息。
    """
    try:
        return load_image(path).size
    except Exception:
        return None, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            thinking_param_sent=reply.thinking_param_sent,
        )

        # 原始回复一定要留：出问题时要看的是"模型到底说了什么"，
        # 而不是解析之后的东西。
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{raw_stem}.attempt{attempts}.txt").write_text(
            reply.content, encoding="utf-8"
        )

        try:
            if reply.finish_reason not in (None, "stop"):
                raise ValueError(f"模型未正常结束：finish_reason={reply.finish_reason}，需完整 JSON")
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
    image_source: str = "待核实：未提供图片来源",
    image_license: str = "待核实：未提供授权说明",
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
    默认标明待核实；正式交付时必须显式填写真实来源和授权。
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
    if not review:
        compose = False

    # ---- 1. 开日志 ----
    trace = RunTrace(image_id=image_id)
    client = None
    try:
        trace.log(
            "run_start",
            image_path=str(path),
            image_width=width,
            image_height=height,
            review_enabled=review,
            compose_enabled=compose,
            render_enabled=render,
        )

        input_snapshot = INPUT_DIR / trace.run_id / path.name
        input_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, input_snapshot)
        path = input_snapshot
        input_sha256 = _sha256_file(path)
        # 提前解码，损坏的图像在消耗 API 费用前失败；坐标与渲染统一 EXIF 方向。
        width, height = load_image(path).size
        trace.log("input_saved", path=str(path), sha256=input_sha256, width=width, height=height)
        prompt_snapshots = INPUT_DIR / trace.run_id / "prompts"
        prompt_snapshots.mkdir()
        dependency_versions = {name: version(name) for name in ("Pillow", "pydantic", "openai", "python-dotenv")}
        endpoint_host = urlsplit(QWEN_BASE_URL).hostname or "未指定主机"
        platform = ("阿里云百炼（DashScope）" if endpoint_host.startswith("dashscope.")
                    and endpoint_host.endswith("aliyuncs.com") else f"OpenAI 兼容服务（{endpoint_host}）")
        manifest = {
            "image_sha256": input_sha256, "image_source": image_source,
            "image_license": image_license, "scene_note": scene_note,
            "image_max_edge": QWEN_IMAGE_MAX_EDGE, "review_crops": QWEN_REVIEW_CROPS,
            "timeout": QWEN_TIMEOUT, "sdk_max_retries": QWEN_MAX_RETRIES,
            "endpoint_host": endpoint_host, "dependencies": dependency_versions,
            "source_hashes": {str(file.relative_to(PROJECT_ROOT)): _sha256_file(file)
                              for file in (PROJECT_ROOT / "src").rglob("*.py")},
        }
        (input_snapshot.parent / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(PROJECT_ROOT / "src" / "guidelines.py", prompt_snapshots / "guidelines.py")

        def snapshot_prompt(agent, stage):
            (prompt_snapshots / f"{stage}.txt").write_text(agent.prompt, encoding="utf-8")

        # ---- 2. 初始化 ----
        client = QwenClient()
        agent = VisionAgent(client, prompt_path=prompt_file)
        snapshot_prompt(agent, "a")

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
            call=lambda history: agent.run(path, history=history, scene_note=scene_note),
            parse=parse_any,
            correction=agent.correction_message,
            raw_stem=trace.run_id,
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
            output_path = OBSERVATION_DIR / f"{trace.run_id}.json"
        else:
            # v1 的老位置，保持兼容，方便和以前的结果对比
            output_path = PROJECT_ROOT / "output" / "results" / f"{trace.run_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            observation_or_analysis.model_dump_json(indent=2), encoding="utf-8"
        )

        # ---- 5. B：审查（v1 的输出没有坐标框，没法审查，直接跳过）----
        review_payload: dict | None = None
        final_regions: list[Region] = []

        if review and stage == "observation":
            observation = observation_or_analysis

            if not observation.findings:
                # A 什么都没找到，B 没有可裁的东西，不用花这次钱
                trace.log("review_skipped", reason="no_candidates")
            else:
                reviewer = ReviewAgent(client, prompt_path=review_prompt_file)
                snapshot_prompt(reviewer, "b")

                trace.log(
                    "review_start",
                    stage="b",
                    prompt_file=reviewer.prompt_path.name,
                    prompt_version=reviewer.version,
                    prompt_sha256=reviewer.prompt_sha,
                    candidates=len(observation.findings),
                )

                review_report, _review_reply, review_attempts = _run_with_retry(
                    trace=trace,
                    stage="b",
                    call=lambda history: reviewer.run(path, observation, history=history, scene_note=scene_note),
                    parse=lambda text: validate_review(observation, parse_review(text)),
                    correction=reviewer.correction_message,
                    raw_stem=f"{trace.run_id}.review",
                )

                review_report.image_id = image_id

                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                review_path = REVIEW_DIR / f"{trace.run_id}.json"
                review_path.write_text(
                    review_report.model_dump_json(indent=2), encoding="utf-8"
                )

                # 确定性后处理：合并、去重、截断、编号
                regions, stats = apply_verdicts(observation, review_report)
                final_regions = regions
                applied_path = REVIEW_DIR / f"{trace.run_id}.applied.json"
                applied_path.write_text(json.dumps({
                    "report": review_report.model_dump(),
                    "regions": [region.model_dump() for region in regions],
                    "stats": asdict(stats),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

                trace.log("review_applied", stage="b", **asdict(stats))

                if not regions:
                    # 全部被砍掉了。这在数据上是合法的（不能据此认定壁画完好），
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
            snapshot_prompt(reporter, "c")
            upstream_limitations = list(dict.fromkeys(
                observation.limitations + review_report.reviewer_notes +
                [f"{f.region_id}：{note}" for f in observation.findings for note in f.limitations]
            ))

            trace.log(
                "report_start",
                stage="c",
                prompt_file=reporter.prompt_path.name,
                prompt_version=reporter.version,
                prompt_sha256=reporter.prompt_sha,
                regions=len(final_regions),
            )

            draft, _report_reply, report_attempts = _run_with_retry(
                trace=trace,
                stage="c",
                call=lambda history: reporter.run(
                    final_regions, upstream_limitations, history=history, scene_note=scene_note
                ),
                parse=lambda text: parse_report(text, {r.region_id for r in final_regions}),
                correction=reporter.correction_message,
                raw_stem=f"{trace.run_id}.report",
            )

            card = build_card(
                image_id=image_id,
                regions=final_regions,
                draft=draft,
                observation=observation,
                image_source=image_source,
                image_license=image_license,
                scene_note=scene_note,
                models=list(dict.fromkeys([a_reply.model, _review_reply.model, _report_reply.model])),
                prompt_versions={
                    "a": agent.prompt_sha,
                    "b": reviewer.prompt_sha,
                    "c": reporter.prompt_sha,
                },
                tools=[f"{name} {value}" for name, value in dependency_versions.items()] + [f"Python {sys.version.split()[0]}"],
                run_id=trace.run_id,
                generated_at=now_iso(),
                input_sha256=input_sha256,
                evidence_path=f"output/reviews/{applied_path.name}",
                upstream_limitations=upstream_limitations,
                platform=platform,
            )

            CARD_DIR.mkdir(parents=True, exist_ok=True)
            card_json_path = CARD_DIR / f"{trace.run_id}.json"
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
            render_error = None
            if render:
                try:
                    rendered = render_card(card, path, CARD_DIR / f"{trace.run_id}.pdf")
                    pdf_path = str(rendered)
                    trace.log("rendered", stage="c", pdf_path=pdf_path)
                except Exception as exc:
                    render_error = f"{type(exc).__name__}: {exc}"
                    trace.log("render_failed", stage="c", error=render_error)
            else:
                trace.log("render_skipped", stage="c", reason="render=False")

            card_payload = {
                "attempts": report_attempts,
                "summary": card.summary,
                "recommendations": [rec.model_dump() for rec in card.recommendations],
                "json_path": str(card_json_path),
                "pdf_path": pdf_path,
                "render_status": "failed" if render_error else ("completed" if render else "skipped"),
                "render_error": render_error,
            }

        # ---- 7. 收尾 ----
        status = "completed"
        reason = ""
        if compose and not card_payload:
            status = "no_card"
            reason = ("审查关闭，未进行成稿" if not review else
                      "旧版无坐标结果不能成卡" if stage != "observation" else
                      "没有通过审查的关注区域；这不等于壁画完好")
        elif card_payload and card_payload["render_status"] == "failed":
            status, reason = "render_failed", card_payload["render_error"]

        trace.log(
            "run_end",
            ok=status == "completed",
            status=status,
            reason=reason,
            stage=stage,
            output_path=str(output_path),
            regions=len(final_regions),
            review_ran=review_payload is not None,
            card_built=card_payload is not None,
            a_attempts=a_attempts,
            a_latency_ms=a_reply.latency_ms,
        )

        return {
            "image_id": image_id,
            "status": status,
            "reason": reason,
            "input_path": str(path),
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
    except Exception as exc:
        trace.log("run_end", ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if client is not None:
            client.close()
