"""Render the reviewed web observation card as a downloadable one-page PDF."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from src.render import _font, _wrap

PAGE = (1240, 1754)  # A4 at 150 dpi
INK = (55, 49, 39)
GREEN = (43, 82, 74)
RUST = (151, 79, 51)
MUTED = (99, 88, 72)


def _text(draw, value, x, y, width, *, size=21, lines=3, leading=30, color=INK, bold=False):
    font = _font(size, bold)
    wrapped = _wrap(draw, str(value or "未填写"), font, width)
    if len(wrapped) > lines:
        raise ValueError("卡片文字超出单页版式，请缩短较长的文字后再导出")
    for line in wrapped:
        draw.text((x, y), line, font=font, fill=color)
        y += leading
    return y


def _heading(draw, title, x, y, width):
    draw.text((x, y), title, font=_font(28, True), fill=GREEN)
    draw.line((x, y + 42, x + width, y + 42), fill=(185, 151, 104), width=2)
    return y + 60


def render_web_card(payload: dict, image_bytes: bytes, background: Path) -> bytes:
    with Image.open(background) as source:
        canvas = source.convert("RGB").resize(PAGE, Image.Resampling.LANCZOS)
    # A translucent paper panel keeps the supplied mural visible around the card.
    overlay = Image.new("RGBA", PAGE, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle((72, 260, 1168, 1530), radius=16,
                                               fill=(255, 249, 235, 232), outline=(162, 120, 74, 255), width=2)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    draw.text((118, 305), "壁画智护 · AI 壁画观察卡", font=_font(46, True), fill=GREEN)
    draw.text((120, 373), "图像观察 · 人工复核 · 来源可追溯", font=_font(21), fill=RUST)
    draw.rounded_rectangle((944, 371, 1120, 412), radius=12, outline=GREEN, width=2)
    draw.text((964, 380), "已人工复核", font=_font(19, True), fill=GREEN)
    _text(draw, "观察摘要：" + (payload.get("summary") or "未填写"), 120, 427, 1000,
          size=20, lines=2, leading=28)

    x, top, box_w, box_h = 118, 518, 480, 355
    draw.rounded_rectangle((x, top, x + box_w, top + box_h), radius=9,
                           fill=(239, 231, 211), outline=(195, 168, 128), width=2)
    with Image.open(BytesIO(image_bytes)) as source:
        mural = source.convert("RGB")
    mural.thumbnail((box_w - 18, box_h - 18), Image.Resampling.LANCZOS)
    ix = x + (box_w - mural.width) // 2
    iy = top + (box_h - mural.height) // 2
    canvas.paste(mural, (ix, iy))
    draw = ImageDraw.Draw(canvas)
    for index, marker in enumerate(payload.get("markers", [])[:3], 1):
        mx = ix + int(mural.width * marker["x"] / 100)
        my = iy + int(mural.height * marker["y"] / 100)
        mw = max(8, int(mural.width * marker["w"] / 100))
        mh = max(8, int(mural.height * marker["h"] / 100))
        draw.rectangle((mx, my, mx + mw, my + mh), outline="white", width=5)
        draw.rectangle((mx, my, mx + mw, my + mh), outline=RUST, width=3)
        draw.rectangle((mx, my, mx + 28, my + 28), fill=RUST)
        draw.text((mx + 7, my + 2), str(index), font=_font(20, True), fill="white")
    _text(draw, "原图与关注区域（编号对应右侧记录）", x, 884, box_w, size=17, lines=1, leading=24, color=MUTED)

    rx, rw = 640, 480
    y = _heading(draw, "关注区域", rx, 515, rw)
    for index, marker in enumerate(payload.get("markers", [])[:3], 1):
        draw.text((rx, y), f"{index:02d}  {marker.get('status') or '待复核'}", font=_font(20, True), fill=RUST)
        y += 31
        y = _text(draw, marker.get("phenomenon"), rx, y, rw, size=19, lines=2, leading=26)
        y = _text(draw, "依据：" + (marker.get("evidence") or "未填写"), rx, y, rw,
                  size=17, lines=1, leading=24, color=MUTED)
        y += 8
        if y > 895:
            break

    y = _heading(draw, "建议与依据", 118, 950, 1002)
    for index, key in enumerate(("advice1", "advice2"), 1):
        value = payload.get(key)
        if value:
            y = _text(draw, f"{index}. {value}", 120, y, 995, size=20, lines=2, leading=29)
            y += 9
    if y == 1010:
        y = _text(draw, "未填写建议。", 120, y, 995, size=20, lines=1)

    y = _heading(draw, "不确定性与人工复核", 118, 1168, 1002)
    _text(draw, payload.get("limitations"), 120, y, 995, size=19, lines=2, leading=28)

    y = _heading(draw, "来源披露", 118, 1288, 1002)
    source_rows = [
        f"图片：{payload.get('imageSource') or '未填写'}  ·  授权：{payload.get('imageLicense') or '未填写'}",
        f"场景：{payload.get('sceneNote') or '无'}",
        f"AI／工具：{payload.get('toolSource') or '未填写'}",
        f"公开资料：{payload.get('referenceSource') or '无'}",
        f"运行编号：{payload.get('runId') or '未调用模型；观察内容由人工填写'}",
    ]
    for row in source_rows:
        y = _text(draw, row, 120, y, 995, size=17, lines=1, leading=28, color=MUTED)
    draw.line((118, 1494, 1120, 1494), fill=(185, 151, 104), width=2)
    _text(draw, "本卡为 AI 辅助图像观察与方案表达，不作为专业鉴定或修复施工依据。",
          120, 1502, 995, size=16, lines=1, color=MUTED)

    result = BytesIO()
    canvas.save(result, format="PDF", resolution=150)
    return result.getvalue()
