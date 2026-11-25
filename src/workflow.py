# src/workflow.py
# =============================================================================
# 编排：一次完整运行的流程
# =============================================================================
# 现在的流程（A 阶段）：
#   读图 -> A 观察 -> 解析 -> 落盘 -> 写日志
#
# 还差的两段（B 审查、C 成稿）接进来之后，这里会变成：
#   A -> B -> C -> Card -> 渲染 PDF
# 所以本文件刻意保持"每一步一个函数调用"的线性结构，方便往里插环节。
#
# 自修复重试：
#   模型偶尔会输出不能解析的内容（多写一句解释、坐标越界、用了 status 老格式）。
#   直接报错让演示中断，是最差的选择。这里给它第二次机会，
#   而且是把"它自己上一次的输出 + 具体错在哪"一起喂回去让它改——
#   重新问一遍它大概率还是同样的错。
# =============================================================================

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.agents.vision_agent import VisionAgent
from src.config import OBSERVATION_DIR, PROJECT_ROOT, RAW_DIR
from src.qwen_client import QwenClient
from src.result_parser import parse_any
from src.trace import RunTrace

# 最多尝试几次。2 次是权衡：第一次失败后带着错误信息重问，
# 成功率很高；再往上加，多半说明是提示词或模型能力的问题，
# 多花的那几次调用换不来质量。
MAX_ATTEMPTS = 2


def _image_size(path: Path) -> tuple[int | None, int | None]:
    """读图片宽高，只用于写日志。

    PIL 万一不可用（换环境没装），不应该让整个流程挂掉——
    尺寸只是个参考信息，缺了不影响出结果。
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def run_vision_workflow(
    image_path: str,
    prompt_file: str | None = None,
) -> dict:
    """跑一次 A 阶段流程，返回结果字典。

    参数：
        image_path  ：图片路径，相对路径按项目根目录解析
        prompt_file ：可选，指定提示词的相对路径（评测时用来对比 v1/v2）

    返回：
        {
          "image_id":    图片名（不含扩展名）,
          "stage":       "observation"（v2）或 "analysis"（v1）,
          "attempts":    实际调用了几次模型,
          "result":      结构化结果（dict）,
          "meta":        解析过程中的修补记录,
          "output_path": 结构化结果落盘位置,
          "trace_path":  本次运行的日志文件,
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
    # 日志要在第一行业务代码之前开，否则前面失败就什么都查不到
    trace = RunTrace(image_id=image_id)
    trace.log(
        "run_start",
        image_path=str(path),
        image_width=width,
        image_height=height,
    )

    # ---- 2. 初始化 ----
    client = QwenClient()
    agent = VisionAgent(client, prompt_path=prompt_file)

    trace.log(
        "agent_ready",
        model=client.model,
        prompt_file=agent.prompt_path.name,
        prompt_version=agent.version,
        prompt_sha256=agent.prompt_sha,
        # 思考链开没开，直接影响耗时（实测差 9 倍），必须记下来，
        # 否则以后看到某次特别慢，会误以为是网络问题
        thinking_requested=client.thinking_enabled,
    )

    # ---- 3. 调模型 + 解析（带一次自修复）----
    result = None
    stage = None
    meta = None
    reply = None
    attempts = 0
    history: list[dict] | None = None

    while attempts < MAX_ATTEMPTS:
        attempts += 1

        try:
            reply = agent.run(path, history=history)
        except Exception as exc:
            # 网络/鉴权/超时这类错误，重试逻辑归 QwenClient 那一层，
            # 这里只负责记下来再抛出去，不要在这里吞掉
            trace.log("model_error", attempt=attempts, error=f"{type(exc).__name__}: {exc}")
            raise

        trace.log(
            "model_call",
            attempt=attempts,
            model=reply.model,
            latency_ms=reply.latency_ms,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            finish_reason=reply.finish_reason,
            response_chars=len(reply.content),
            thinking_param_sent=reply.thinking_param_sent,
        )

        # 保存原始回复：不管是第几次尝试，原文都要留。
        # 出问题时要看的是"模型到底说了什么"，而不是解析后的结果。
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{image_id}.attempt{attempts}.txt").write_text(
            reply.content, encoding="utf-8"
        )

        try:
            stage, result, meta = parse_any(reply.content)
        except ValueError as exc:
            # 解析失败：记下来，把错误回喂给模型再来一次
            trace.log("parse_fail", attempt=attempts, error=str(exc))

            if attempts >= MAX_ATTEMPTS:
                trace.log("run_end", ok=False, attempts=attempts, reason="parse_failed")
                raise RuntimeError(
                    f"模型连续 {attempts} 次输出都无法解析，最后一次的错误：{exc}\n"
                    f"原始回复见：{RAW_DIR / f'{image_id}.attempt{attempts}.txt'}"
                ) from exc

            history = [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": agent.correction_message(str(exc))},
            ]
            trace.log("retry", attempt=attempts + 1, reason=str(exc))
            continue

        trace.log(
            "parse_ok",
            attempt=attempts,
            stage=stage,
            raw_findings=meta.raw_findings,
            kept=meta.kept,
            truncated=meta.truncated,
            bbox_rescaled=meta.bbox_rescaled,
        )
        break

    if result is None or stage is None or meta is None:
        # 理论上走不到这里（循环里要么 break 要么 raise），留个保险
        raise RuntimeError("流程异常结束：没有拿到解析结果")

    # ---- 4. 以本地文件名为准 ----
    # 模型可能把 image_id 写成中文描述或者带扩展名，
    # 下游全部按文件名索引，所以这里统一覆盖，不信任模型。
    result.image_id = image_id

    # ---- 5. 落盘 ----
    if stage == "observation":
        OBSERVATION_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OBSERVATION_DIR / f"{image_id}.json"
    else:
        # v1 的老位置，保持兼容，方便和以前的结果对比
        output_path = PROJECT_ROOT / "output" / "results" / f"{image_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # ---- 6. 收尾日志 ----
    elapsed_note = {
        "attempts": attempts,
        "output_path": str(output_path),
        "regions": len(result.findings),
    }
    if reply is not None:
        elapsed_note["total_model_latency_ms"] = reply.latency_ms

    trace.log("run_end", ok=True, **elapsed_note)

    return {
        "image_id": image_id,
        "stage": stage,
        "attempts": attempts,
        "result": json.loads(result.model_dump_json()),
        "meta": asdict(meta),
        "output_path": str(output_path),
        "trace_path": str(trace.path),
        "run_id": trace.run_id,
    }
