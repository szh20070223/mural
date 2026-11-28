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

# 思考链开关（Qwen3 系列支持 enable_thinking）。
#
# 实测数据（**qwen3.8-max**，1123x756 的壁画图，同一份提示词）：
#   开着：109 秒，completion 4955 token，只给出 1 个候选区域
#   关掉： 12 秒，completion  557 token，给出 3 个候选区域
#
# 换到 qwen4-max 之后没有重测这两组数字，但"关掉"这个选择仍然成立：
# 当前单次调用稳定在 8 秒左右（含思考关闭）。换模型时请顺手记一组新数字。
#
# 默认关闭。想要长链推理时：$env:QWEN_ENABLE_THINKING="true"
QWEN_ENABLE_THINKING = os.getenv("QWEN_ENABLE_THINKING", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# 审查环节是否附带局部裁剪图（多图输入）。
#
# 它解决的是一个物理问题：整图进模型时会被视觉编码器缩到固定边长，
# 面部那种 16%×9% 的区域缩完只剩一百多像素，边缘锐利还是渐变根本分不出来。
# 单独发一张裁剪，等于把该区域以更高的有效分辨率送进去。
#
# **已验证**（2026-11-26，qwen3.8-max）：
#   同一张 1280×1634 的壁画，问"整图里能看清面部斑块的边缘吗 / 放大图里边缘什么样"，
#   模型回答"整图不能"、"放大图里是渐变的、没有翘起亮边"。
#   说明它真的读到了裁剪图，不是在复述。
#   代价：B 的输入 token 从 4155 涨到 6706（约 +60%），耗时基本不变（7 秒）。
#
# 关掉的方式：$env:QWEN_REVIEW_CROPS="false"
QWEN_REVIEW_CROPS = os.getenv("QWEN_REVIEW_CROPS", "true").strip().lower() in {
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
# 默认 v3：带坐标框 + 逐病种识别标准 + "待判读"兜底。
# v2 是更早期的一版，保留下来做对照；v1 只出文字位置，做不出卡片。
VISION_PROMPT_FILE = os.getenv("VISION_PROMPT_FILE", "vision_analysis_v3.txt")

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

# ---------------------------------------------------------------------------
# 启动自检
# ---------------------------------------------------------------------------
if not DASHSCOPE_API_KEY:
    raise RuntimeError("没有配置 DASHSCOPE_API_KEY，请检查 .env 文件")

if not QWEN_MODEL:
    raise RuntimeError("没有配置 QWEN_MODEL，请检查 .env 文件")
