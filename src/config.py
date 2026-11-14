# src/config.py
# 统一管理配置
import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_MODEL")

if not DASHSCOPE_API_KEY:
    raise RuntimeError("没有配置 DASHSCOPE_API_KEY，请检查 .env 文件")

if not QWEN_MODEL:
    raise RuntimeError("没有配置 QWEN_MODEL，请检查 .env 文件")