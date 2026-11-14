#把模型回复变成合法数据

"""
这里有两个层次：
- json：检查文本是不是合法 JSON。
- Pydantic：检查 JSON 是否符合约定的数据结构。
"""
# src/result_parser.py
import json

from src.schemas import AnalysisResult


def extract_json(text: str) -> dict:
    text = text.strip()

    # 处理 Markdown 代码围栏
    if text.startswith("```"):
        lines = text.splitlines()

        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])

    # 找到 JSON 对象的起始位置
    start = text.find("{")

    if start == -1:
        raise ValueError("模型回复中没有找到 JSON 对象")

    # 从第一个 { 开始解析完整 JSON
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text[start:])

    if not isinstance(data, dict):
        raise ValueError("分析结果必须是 JSON 对象")

    return data


def parse_analysis_result(text: str) -> AnalysisResult:
    data = extract_json(text)

    # 使用 Pydantic 检查字段和字段类型
    return AnalysisResult.model_validate(data)