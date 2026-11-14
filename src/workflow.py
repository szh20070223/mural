# src/workflow.py
import json
from pathlib import Path

from src.qwen_client import QwenClient
from src.agents.vision_agent import VisionAgent
from src.result_parser import parse_analysis_result
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_vision_workflow(image_path: str) -> dict:

    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_path = image_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    # 1. 初始化客户端和 Agent
    client = QwenClient()
    agent = VisionAgent(client)

    # 2. 调用 Agent A，获得原始结果
    raw_result = agent.run(str(image_path))

    # 3. 先保存原始回复，方便以后检查模型输出
    raw_dir = PROJECT_ROOT / "output" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{image_path.stem}.txt"
    raw_path.write_text(raw_result, encoding="utf-8")

    # 4. 解析并校验结果
    result = parse_analysis_result(raw_result)

    # 5. 以本地文件名为准，避免模型写错 image_id
    result.image_id = image_path.stem

    # 6. 保存结构化结果
    result_dir = PROJECT_ROOT / "output" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{image_path.stem}.json"

    result_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8"
    )

    # 7. 返回普通字典，供其他模块调用
    return result.model_dump()