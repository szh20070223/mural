"""Serve the observation-card prototype and bridge it to the existing workflow."""

import base64
import binascii
import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_BODY = 14 * 1024 * 1024
FORMATS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_POST(self):
        if self.path != "/api/analyze":
            return self.reply(404, {"error": "接口不存在"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.reply(413, {"error": "图片过大，最大 10 MB"})
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                return self.reply(400, {"error": "请求格式应为 JSON 对象"})
            mime = payload.get("mime")
            source = str(payload.get("image_source", "")).strip()
            license_note = str(payload.get("image_license", "")).strip()
            if not source or not license_note:
                return self.reply(400, {"error": "请填写图片来源与授权范围后再调用模型"})
            if mime not in FORMATS:
                return self.reply(400, {"error": "仅支持 PNG、JPEG、WebP 图片"})
            image = base64.b64decode(payload.get("image", ""), validate=True)
            if not image or len(image) > 10 * 1024 * 1024:
                return self.reply(413, {"error": "图片过大，最大 10 MB"})
            load_dotenv(dotenv_path=ROOT / ".env")
            if not os.getenv("DASHSCOPE_API_KEY"):
                return self.reply(503, {"error": "尚未配置 DASHSCOPE_API_KEY"})
            if not os.getenv("QWEN_MODEL"):
                return self.reply(503, {"error": "尚未配置 QWEN_MODEL"})
            with tempfile.NamedTemporaryFile(suffix=FORMATS[mime], delete=False) as tmp:
                tmp.write(image)
                image_path = Path(tmp.name)
            try:
                from src.workflow import run_vision_workflow
                result = run_vision_workflow(
                    str(image_path), image_source=source,
                    image_license=license_note,
                    scene_note=str(payload.get("scene_note", "")),
                )
            finally:
                image_path.unlink(missing_ok=True)
            card_info = result.get("card")
            batch = result["candidate_status"]
            stats = (result.get("review") or {}).get("stats")
            metadata = {"run_id": result["run_id"], "candidate_status": batch, "review_stats": stats}
            if not card_info:
                return self.reply(200, {
                    "card": None,
                    **metadata,
                    "message": (f"A 补查后有 {batch['count']} 处可定位候选，未达到 {batch['minimum']} 处，已停止送审。"
                                if batch and not batch["ready"]
                                else "本次没有可上卡的关注区域，请对照原图检查。"),
                })
            card = json.loads(Path(card_info["json_path"]).read_text(encoding="utf-8"))
            return self.reply(200, {"card": card, **metadata})
        except (ValueError, KeyError, binascii.Error, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": f"请求数据无效：{exc}"})
        except Exception as exc:
            return self.reply(500, {"error": f"分析失败：{exc}"})

    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    address = ("127.0.0.1", int(os.getenv("PORT", "8000")))
    print(f"打开 http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()
