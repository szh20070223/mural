# src/qwen_client.py
# =============================================================================
# 通义千问（DashScope 兼容 OpenAI 接口）的最小封装
# =============================================================================
# 这一层的职责只有三件：
#   1. 把本地图片读成 base64 data URL
#   2. 发请求
#   3. 把回复连同「模型名 / 耗时 / token 数 / 结束原因」一起返回
#
# 第 3 点是这次改的重点：以前只返回一段字符串，日志里没有耗时和 token，
# 出问题只能靠猜。现在返回 ModelReply，trace 才有东西可记。
#
# 图像统一编码、可选缩放与 SDK 网络重试在本层实现。

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    QWEN_ENABLE_THINKING,
    QWEN_MODEL,
    QWEN_TIMEOUT,
    QWEN_MAX_RETRIES,
    QWEN_IMAGE_MAX_EDGE,
)
from src.images import load_image
from PIL import Image

def _looks_like_unsupported_param(exc: Exception) -> bool:
    """判断这次报错是不是"模型不认 enable_thinking 这个参数"。

    不按异常类型判断，是因为 openai SDK 的异常类在不同大版本之间搬过家；
    按状态码 + 报错文字判断更稳，升级 SDK 也不会失效。
    """
    status = getattr(exc, "status_code", None)
    if status not in (400, 422):
        return False

    text = str(exc).lower()
    return "enable_thinking" in text and any(
        token in text for token in ("unsupported", "not support", "unknown", "unrecognized", "unexpected", "不支持")
    )


@dataclass
class ModelReply:
    """一次模型调用的完整回执。

    为什么不直接返回字符串：调用日志需要知道"花了多久、用了多少 token、
    是不是被截断了"。这些信息只在 response 对象里，一旦丢成字符串就再也拿不回来。
    """

    content: str                 # 模型回复的正文
    model: str                   # 实际使用的模型名（以服务端返回为准）
    latency_ms: int              # 本次调用耗时
    prompt_tokens: int | None    # 输入 token（接口没返回就是 None）
    completion_tokens: int | None
    finish_reason: str | None    # "stop" 正常结束；"length" 说明被截断
    # 这次请求有没有真的带上 enable_thinking 参数。
    # 注意语义：关闭思考链也必须**显式**发 false——不发参数的话，
    # 服务端会按模型默认值来（多半是开着），耗时差 6 倍以上。
    # 这个坑吃过一次，专门留个字段记着。
    thinking_param_sent: bool = False


class QwenClient:
    """DashScope 的 OpenAI 兼容客户端。"""

    def __init__(self) -> None:
        if not DASHSCOPE_API_KEY or not DASHSCOPE_API_KEY.strip():
            raise ValueError("没有配置 DASHSCOPE_API_KEY，请检查项目 .env 文件")
        if not QWEN_MODEL or not QWEN_MODEL.strip():
            raise ValueError("没有配置 QWEN_MODEL，请检查项目 .env 文件")
        if not math.isfinite(QWEN_TIMEOUT) or QWEN_TIMEOUT <= 0:
            raise ValueError("QWEN_TIMEOUT 必须是有限正数")
        if QWEN_MAX_RETRIES < 0 or QWEN_IMAGE_MAX_EDGE < 0:
            raise ValueError("重试次数与图像边长配置不能为负数")
        self.model = QWEN_MODEL
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
            # 超时：宁可报错重跑，也不要挂在那里让人以为还在跑
            timeout=QWEN_TIMEOUT,
            max_retries=QWEN_MAX_RETRIES,
        )
        self._image_cache: tuple[tuple, str] | None = None

        # 是否请求关闭思考链，来自配置
        self.thinking_enabled = QWEN_ENABLE_THINKING
        # 当前模型是否支持这个参数。第一次调用失败后会被置为 False，
        # 后面的调用就不再带它，避免每次都先错一次。
        self._thinking_param_supported = True

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------
    # 图片编码
    # ------------------------------------------------------------------

    def encode_image(self, image_path: str) -> str:
        """把图片读成 data URL。

        格式：data:image/png;base64,xxxxx
        这是 OpenAI 视觉接口的通用写法，DashScope 兼容模式也认。
        """
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")

        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if self._image_cache and self._image_cache[0] == cache_key:
            return self._image_cache[1]
        photo = load_image(path)
        if QWEN_IMAGE_MAX_EDGE:
            photo.thumbnail((QWEN_IMAGE_MAX_EDGE, QWEN_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
        encoded = self._encode_photo(photo)
        self._image_cache = (cache_key, encoded)
        return encoded

    @staticmethod
    def _encode_photo(photo: Image.Image) -> str:
        buffer = io.BytesIO()
        photo.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    # ------------------------------------------------------------------
    # 消息构造
    # ------------------------------------------------------------------

    def build_image_messages(
        self,
        image_path: str,
        prompt: str,
        history: list[dict] | None = None,
        regions: list | None = None,
    ) -> list[dict]:
        """拼出一条完整的消息列表。

        history 用来做"自修复重试"：第一次输出格式不对时，把模型自己的
        错误回复和一句纠正说明接在后面，让它看着自己的输出改。
        比重新问一遍有效得多——重问它大概率还是同样的错。
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.encode_image(image_path)},
                    },
                ],
            }
        ]

        if regions:
            photo = load_image(image_path)
            for label, box in regions:
                crop = photo.crop(box.to_pixels(*photo.size))
                crop.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                messages[0]["content"].extend([
                    {"type": "text", "text": f"{label} 局部图，原图归一化范围 {box.as_list()}。所有输出坐标仍相对完整原图。"},
                    {"type": "image_url", "image_url": {"url": self._encode_photo(crop)}},
                ])
        if history:
            messages.extend(history)

        return messages

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------

    def _create(self, messages: list[dict], send_thinking_flag: bool):
        """真正发请求的地方，把"带不带思考链参数"抽成一个开关。"""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            # 温度压到 0.1：这个任务要的是稳定复现，不是文采。
            # 演示时同一张图跑两次结果差太多，评委问起来很难解释。
            "temperature": 0.1,
        }
        if send_thinking_flag:
            # DashScope 的扩展参数必须走 extra_body，
            # 直接放进 kwargs 会被 openai SDK 当成未知参数拒掉。
            kwargs["extra_body"] = {"enable_thinking": self.thinking_enabled}
        return self.client.chat.completions.create(**kwargs)

    def chat(self, messages: list[dict]) -> ModelReply:
        """发一次请求，返回结构化的回执。"""
        started = time.perf_counter()

        send_flag = self._thinking_param_supported

        if not send_flag:
            response = self._create(messages, False)
        else:
            try:
                response = self._create(messages, True)
            except Exception as exc:
                # 换模型时最常见的坑：新模型不认 enable_thinking 这个参数。
                # 现场赛允许使用的模型清单可能和现在不一样，所以这里不硬失败：
                # 去掉参数重发一次，并记住"这个模型不支持"，后面不再带它。
                if not _looks_like_unsupported_param(exc):
                    raise
                self._thinking_param_supported = False
                response = self._create(messages, False)

        latency_ms = int((time.perf_counter() - started) * 1000)

        if not response.choices:
            raise RuntimeError("模型没有返回任何候选结果（choices 为空）")

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)

        if not isinstance(content, str) or not content.strip():
            # 把 finish_reason 一起报出来：如果是 "length"，
            # 说明是输出被截断，不是模型不想说话，排查方向完全不同。
            raise RuntimeError(
                f"模型返回了空内容（finish_reason={finish_reason}）"
            )

        usage = getattr(response, "usage", None)

        return ModelReply(
            content=content,
            # 以服务端返回的模型名为准：它可能带日期后缀，
            # 披露到卡片上的名字必须和实际调用的一致
            model=getattr(response, "model", None) or self.model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            finish_reason=finish_reason,
            # 兜底之后 _thinking_param_supported 会变成 False，
            # 所以这里能如实反映"这次请求到底带没带上这个参数"
            thinking_param_sent=send_flag and self._thinking_param_supported,
        )

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        history: list[dict] | None = None,
        regions: list | None = None,
    ) -> ModelReply:
        """一步到位：图 + 提示词 -> 回执。日常调用走这个就行。"""
        messages = self.build_image_messages(image_path, prompt, history=history, regions=regions)
        return self.chat(messages)
