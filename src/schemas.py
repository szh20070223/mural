# src/schemas.py
# 约定模型最终要交付什么数据
from typing import Literal
from pydantic import BaseModel, Field


DiseaseStatus = Literal[
    "明确发现",
    "疑似发现",
    "无法判断",
    "未发现"
]

Confidence = Literal["高", "中", "低", "无法评估"]


class DiseaseFinding(BaseModel):
    disease: str
    status: DiseaseStatus
    location: list[str] = Field(default_factory=list)
    evidence: str
    confidence: Confidence = "无法评估"
    limitations: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    image_id: str
    findings: list[DiseaseFinding]
    limitations: list[str] = Field(default_factory=list)
    summary: str