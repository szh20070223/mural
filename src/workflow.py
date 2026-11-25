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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.agents.review_agent import ReviewAgent
from src.agents.vision_agent import VisionAgent
from src.config import OBSERVATION_DIR, PROJECT_ROOT, RAW_DIR, REVIEW_DIR
from src.qwen_client import ModelReply, QwenClient
from src.result_parser import parse_any, parse_review
from src.review_rules import apply_verdicts
from src.trace import RunTrace

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
) -> dict:
    """跑一次完整流程（A 观察 + 可选 B 审查）。

    参数：
        image_path          ：图片路径，相对路径按项目根目录解析
        prompt_file         ：可选，A 的提示词（评测时用来对比 v1/v2）
        review              ：是否跑 B 审查。调 A 的提示词时可以关掉省钱
        review_prompt_file  ：可选，B 的提示词

    返回：
        {
          "image_id", "stage", "attempts", "result", "meta",
          "output_path", "trace_path", "run_id",
          "review": {                          # review=False 或 A 没找到东西时为 None
              "attempts":     B 调了几次,
              "report":       裁决原文（含每条的理由）,
              "regions":      最终进卡的 1-3 个区域,
              "stats":        裁掉几条、合并几条、截断几条,
              "output_path":  裁决落盘位置,
          }
        }
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
            )

            report, _review_reply, review_attempts = _run_with_retry(
                trace=trace,
                stage="b",
                call=lambda history: reviewer.run(path, observation, history=history),
                parse=parse_review,
                correction=reviewer.correction_message,
                raw_stem=f"{image_id}.review",
            )

            report.image_id = image_id

            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            review_path = REVIEW_DIR / f"{image_id}.json"
            review_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

            # 确定性后处理：合并、去重、截断、编号
            regions, stats = apply_verdicts(observation, report)

            trace.log("review_applied", stage="b", **asdict(stats))

            if not regions:
                # 全部被砍掉了。这在数据上是合法的（这张图确实没问题），
                # 但卡片契约要求至少 1 个区域，所以这里必须显式记一笔，
                # 后面做 C 的时候要决定：是出"未发现明显病害"的变体卡，
                # 还是回到 A 请它重新看。现在先如实暴露，不假装成功。
                trace.log("review_all_dropped", stage="b")

            review_payload = {
                "attempts": review_attempts,
                "report": json.loads(report.model_dump_json()),
                "regions": [json.loads(region.model_dump_json()) for region in regions],
                "stats": asdict(stats),
                "output_path": str(review_path),
            }

    # ---- 6. 收尾 ----
    trace.log(
        "run_end",
        ok=True,
        stage=stage,
        output_path=str(output_path),
        regions=(
            len(review_payload["regions"])
            if review_payload
            else len(observation_or_analysis.findings)
        ),
        review_ran=review_payload is not None,
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
    }
