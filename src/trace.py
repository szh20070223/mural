# src/trace.py
# =============================================================================
# 运行轨迹日志（一条运行 = 一个 JSONL 文件，一行一个事件）
# =============================================================================
# 为什么要专门做这个：
#   1. 现场赛的评分项里有"关键过程可追溯"，评审组有权核查模型调用清单。
#      能拿出"什么时候、用哪个模型、什么提示词、耗时多久、返回了多少字"的记录，
#      和"我们跑了，效果不错"是完全两回事。
#   2. 演示时如果结果和预期不一样，第一件事是翻日志，不是重跑。
#   3. 后面写评测脚本时，误报/漏报能顺着 run_id 追到那一次的具体输入输出。
#
# 设计取舍：
#   - 用 JSONL 而不是数据库：一行一条，能用记事本打开，也能直接被 pandas 读。
#   - 每次写入都重新打开文件（append 模式）：不持有文件句柄，
#     进程被 Ctrl+C 打断也不会丢已经写进去的内容。
#   - 日志写失败**不能**让主流程崩。演示当天，日志挂了也得把卡片出出来。
# =============================================================================

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import TRACE_DIR


def now_iso() -> str:
    """带时区的 ISO8601 时间。不带时区的话，跨机器对日志会差 9 小时。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_short(text: str, length: int = 12) -> str:
    """取文本的短哈希。

    用来标记"提示词的哪一版"：比手写版本号可靠，改了任何一个字哈希都会变。
    写进日志和卡片的 prompt_version，就能证明当时的提示词长什么样。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


class RunTrace:
    """一次运行一个实例。

    用法：
        trace = RunTrace(image_id="test_mural1")
        trace.log("model_call", attempt=1, latency_ms=3210)
        print(trace.path)
    """

    def __init__(self, image_id: str, trace_dir: Path | None = None) -> None:
        self.image_id = image_id
        self.trace_dir = Path(trace_dir) if trace_dir else TRACE_DIR

        # run_id 里同时带图片名、时间、随机后缀：
        # 图片名方便肉眼找，时间方便排序，随机后缀防止同一秒跑两次撞名。
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:4]
        self.run_id = f"{image_id}-{stamp}-{suffix}"

        self.path = self.trace_dir / f"{self.run_id}.jsonl"

        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # 目录建不出来就算了，日志不是主流程
            print(f"[trace] 警告：无法创建日志目录 {self.trace_dir}：{exc}")

    def log(self, event: str, **fields: object) -> None:
        """写一条事件。

        event 用英文短名（run_start / model_call / parse_fail ...），
        字段名也尽量固定，方便以后用脚本统计——中文键名在表格里对不齐。
        """
        record = {"ts": now_iso(), "run_id": self.run_id, "event": event}
        record.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # 这里绝不能抛异常：日志失败不能连累出卡
            print(f"[trace] 警告：写日志失败：{exc}")
