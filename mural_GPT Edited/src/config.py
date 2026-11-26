# src/config.py
# 统一管理配置：环境变量都在这里读一次，别的地方不要再去碰 os.getenv
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 显式指定 .env 的位置，不依赖当前工作目录。
#
# 踩过的坑：PyCharm 里点"运行"时工作目录通常是项目根，没问题；
# 但如果从别的目录执行 `python E:\py\mural-api-test\main.py`，
# load_dotenv() 会去**那个**目录找 .env，找不到就报"没有配置 DASHSCOPE_API_KEY"，
# 报错信息会把你引向一个完全无关的方向。
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# 模型接入
# ---------------------------------------------------------------------------

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.getenv("QWEN_MODEL")

# 单次请求超时（秒）。不设的话，网络卡住时脚本会一直挂着——
# 演示当天最怕的就是"看起来在跑，其实早就死了"。
QWEN_TIMEOUT = float(os.getenv("QWEN_TIMEOUT", "120"))
QWEN_MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "2"))
# 0 保留全图尺寸。启用缩放时仍保留原始文件用于复核。
QWEN_IMAGE_MAX_EDGE = int(os.getenv("QWEN_IMAGE_MAX_EDGE", "0"))
QWEN_REVIEW_CROPS = os.getenv("QWEN_REVIEW_CROPS", "false").lower() in {"true", "1", "yes"}

# 思考链开关（Qwen3 系列支持 enable_thinking）。
#
# 默认关闭是原项目配置选择，不是准确性结论；不同模型需独立验证。
QWEN_ENABLE_THINKING = os.getenv("QWEN_ENABLE_THINKING", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ---------------------------------------------------------------------------
# 提示词版本
# ---------------------------------------------------------------------------
# 用环境变量切换，而不是改代码里的字符串。这样 v1/v2 可以并排跑、对比效果：
#   PowerShell:  $env:VISION_PROMPT_FILE="vision_analysis.txt"
#
# 默认 v2：带坐标框的版本，能出观察卡。
# v1 是早期只出"文字位置"的老版本，只保留作对照，出不了卡片。
VISION_PROMPT_FILE = os.getenv("VISION_PROMPT_FILE", "vision_analysis_v2.txt")

VISION_PROMPT_PATH = PROJECT_ROOT / "prompts" / VISION_PROMPT_FILE

# 版本号从文件名里取：vision_analysis_v2.txt -> "v2"。
# 这个值会写进卡片的 provenance.prompt_version，用来回答
# "这张卡是哪个版本的提示词生成的"——答辩时被问到就靠它。
_version_match = re.search(r"_v(\d+)$", VISION_PROMPT_PATH.stem)
VISION_PROMPT_VERSION = f"v{_version_match.group(1)}" if _version_match else "v1"

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
# 分三类放：
#   raw/          模型原始回复，出问题时第一时间看这里
#   observations/ A 阶段解析后的结构化结果（中间产物）
#   trace/        每次运行的完整日志，一行一个事件
RAW_DIR = PROJECT_ROOT / "output" / "raw"
OBSERVATION_DIR = PROJECT_ROOT / "output" / "observations"
REVIEW_DIR = PROJECT_ROOT / "output" / "reviews"
CARD_DIR = PROJECT_ROOT / "output" / "cards"
TRACE_DIR = PROJECT_ROOT / "output" / "trace"
INPUT_DIR = PROJECT_ROOT / "output" / "inputs"

# ---------------------------------------------------------------------------
# 启动自检
# ---------------------------------------------------------------------------
# 凭据在创建客户端时校验，使 --help、离线校验与渲染无需 API 配置。
