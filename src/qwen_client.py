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
# 明确划出去的边界（属于客户端硬化那条线，见 README 任务清单 R1 待办）：
#   - 超大图片的缩放 / 压缩
#   - 网络失败的重试与退避
#   - 更多图片格式的兼容
# 这里只留一个超时，避免网络卡死时整个演示挂住。
# =============================================================================

from __future__ import annotations

import base64
import io
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image

from src.config import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    QWEN_ENABLE_THINKING,
    QWEN_MODEL,
    QWEN_TIMEOUT,
)

# 支持的图片格式。注意：这里的值必须和 encode_image 里判断的完全一致，
# 之前出现过一处写 image/webp、另一处没写，改的时候两边一起改。
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}


def _looks_like_unsupported_param(exc: Exception) -> bool:
    """判断这次报错是不是"模型不认 enable_thinking 这个参数"。

    不按异常类型判断，是因为 openai SDK 的异常类在不同大版本之间搬过家；
    按状态码 + 报错文字判断更稳，升级 SDK 也不会失效。
    """
    status = getattr(exc, "status_code", None)
    if status not in (400, 422):
        return False

    text = str(exc).lower()
    return any(
        token in text for token in ("thinking", "extra_body", "parameter", "invalid")
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
    # 这次请求有没有**真的把思考链关掉**。
    #
    # 注意语义：关闭思考链也必须**显式**发 enable_thinking=false——
    # 不发参数的话服务端会按模型默认值来（多半是开着），耗时差好几倍。
    # 所以不能写成"有没有发参数"，那会让人误以为 true = 开着思考。
    # True = 已显式要求关闭；False = 没关成（配置要求开、或模型不认这个参数）。
    thinking_disabled: bool = False


class QwenClient:
    """DashScope 的 OpenAI 兼容客户端。"""

    def __init__(self) -> None:
        self.model = QWEN_MODEL
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
            # 超时：宁可报错重跑，也不要挂在那里让人以为还在跑
            timeout=QWEN_TIMEOUT,
        )

        # 是否请求关闭思考链，来自配置
        self.thinking_enabled = QWEN_ENABLE_THINKING
        # 当前模型是否支持这个参数。第一次调用失败后会被置为 False，
        # 后面的调用就不再带它，避免每次都先错一次。
        self._thinking_param_supported = True

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

        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type not in ALLOWED_MIME_TYPES:
            raise ValueError(
                f"暂不支持的图片格式：{mime_type}。"
                f"支持的类型：{sorted(ALLOWED_MIME_TYPES)}"
            )

        with open(path, "rb") as handle:
            image_data = base64.b64encode(handle.read()).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"

    # ------------------------------------------------------------------
    # 消息构造
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_crop(photo: "Image.Image", bbox, max_side: int = 800) -> str:
        """把一个候选区域裁出来、留一圈边、放大，编码成 data URL。

        两个细节：
          - **必须留边**：只裁框内会丢掉"边界在哪"的上下文，
            而判断颜料层脱落最关键的恰恰就是边界。
            这里按框宽高的 15% 外扩，最多到图片边界。
          - **小图要放大**：面部那种 16%×9% 的框在原图上只有两百来像素，
            视觉编码器对几十像素的细节基本无感。放大不增加信息，
            但能让模型真的"看见"边缘。
        """
        width, height = photo.size
        x1, y1, x2, y2 = bbox.to_pixels(width, height)

        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

        crop = photo.crop(box)

        scale = min(4.0, max_side / max(crop.size))
        if scale != 1:
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS
            )

        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def build_image_messages(
        self,
        image_path: str,
        prompt: str,
        history: list[dict] | None = None,
        crops: list[tuple[str, object]] | None = None,
    ) -> list[dict]:
        """拼出一条完整的消息列表。

        crops：可选的 [(region_id, BoundingBox), ...]。
        传了的话，除了整张图，还会按顺序附上每个候选的局部裁剪，
        并在每张裁剪前面标一句"这是同一张图的局部放大"——
        不标的话模型很容易当成"又是另一张壁画"。

        history 用来做"自修复重试"：第一次输出格式不对时，把模型自己的
        错误回复和一句纠正说明接在后面，让它看着自己的输出改。
        比重新问一遍有效得多——重问它大概率还是同样的错。
        """
        content: list[dict] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": self.encode_image(image_path)},
            },
        ]

        if crops:
            with Image.open(image_path) as source:
                photo = source.convert("RGB")

            for region_id, bbox in crops:
                content.append(
                    {
                        "type": "text",
                        "text": f"裁剪图 {region_id}：这是同一张图的局部放大，"
                        f"用来核对框位和边缘细节，不是另一张图。",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_crop(photo, bbox)},
                    }
                )

        messages: list[dict] = [{"role": "user", "content": content}]

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

        if not content:
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
            model=getattr(response, "model", self.model),
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            finish_reason=finish_reason,
            # 兜底之后 _thinking_param_supported 会变成 False，
            # 所以这里能如实反映"这次到底有没有把思考关掉"
            thinking_disabled=send_flag
            and self._thinking_param_supported
            and not self.thinking_enabled,
        )

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        history: list[dict] | None = None,
        crops: list[tuple[str, object]] | None = None,
    ) -> ModelReply:
        """一步到位：图 + 提示词 -> 回执。日常调用走这个就行。"""
        messages = self.build_image_messages(
            image_path, prompt, history=history, crops=crops
        )
        return self.chat(messages)
