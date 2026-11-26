# src/render.py
# =============================================================================
# 渲染：Card -> 一页 A4 PDF
# =============================================================================
# 保持 Pillow/A4 路线：完整测量后布局，溢出报错，不静默截断。

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.schemas import Card
from src.images import load_image

# A4 @ 300dpi。写成固定尺寸而不是可配置，是为了保证"永远一页"。
PAGE_WIDTH = 2480
PAGE_HEIGHT = 3508
MARGIN = 150

# 配色：深蓝主色 + 灰正文 + 橙色"待复核"徽章。
# 别用太鲜艳的颜色——这是一张材料，不是海报。
COLOR_MAIN = (26, 60, 143)
COLOR_TEXT = (34, 34, 34)
COLOR_MUTED = (110, 110, 110)
COLOR_LINE = (210, 214, 222)
COLOR_WARN = (214, 122, 30)
COLOR_OK = (39, 139, 96)
COLOR_BG_SOFT = (246, 248, 252)

# 区域标记颜色，按顺序取。和图上的框、卡片上的编号一一对应。
REGION_COLORS = [
    (214, 48, 49),
    (41, 128, 185),
    (39, 174, 96),
    (142, 68, 173),
]

# Windows 自带的中文字体，按优先级找第一个存在的。
_FONT_CANDIDATES = {
    False: ["msyh.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc"],
    True: ["msyhbd.ttc", "msyh.ttc", "simhei.ttf", "simsun.ttc"],
}


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """按字号取字体。带缓存——一张卡要取几十次，不缓存会明显变慢。"""
    configured = os.getenv("MURAL_FONT_PATH")
    if configured:
        return ImageFont.truetype(configured, size)
    for linux_font in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ):
        if Path(linux_font).is_file():
            return ImageFont.truetype(linux_font, size)
    fonts_dir = Path("C:/Windows/Fonts")

    for name in _FONT_CANDIDATES[bold]:
        candidate = fonts_dir / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)

    raise RuntimeError(
        "找不到中文字体。请确认 C:/Windows/Fonts 下有 msyh.ttc 或 simhei.ttf；"
        "换到没有这些字体的机器上时，需要把字体文件一起带上"
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """按像素宽度折行。

    中文没有空格，不能按词折，只能一个字一个字量宽度。
    效率不高但足够——一张卡最多几百个字。
    文本里原有的换行会被保留。
    """
    lines: list[str] = []

    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue

        current = ""
        for char in paragraph:
            trial = current + char
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)

    return lines


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    x: int,
    y: int,
    max_width: int,
    max_lines: int,
    line_height: int,
    fill: tuple[int, int, int] = COLOR_TEXT,
) -> int:
    """画一段折行文本，返回画完之后的 y。

    超过 max_lines 明确报错；不丢弃正文。
    """
    lines = _wrap(draw, text, font, max_width)

    if len(lines) > max_lines:
        raise ValueError("文字超过指定版面行数，未输出截断内容")

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height

    return y


def _draw_section_title(
    draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int
) -> int:
    """小节标题：左侧一根竖线 + 标题文字，返回下一行的 y。"""
    draw.rectangle([x, y + 6, x + 8, y + 44], fill=COLOR_MAIN)
    draw.text((x + 24, y), text, font=_font(34, bold=True), fill=COLOR_MAIN)
    return y + 62


def _paste_thumbnail(
    canvas: Image.Image,
    image_path: Path,
    card: Card,
    box: tuple[int, int, int, int],
) -> None:
    """把原图等比缩放塞进指定区域，并在上面画区域框和编号标签。"""
    left, top, right, bottom = box
    max_w, max_h = right - left, bottom - top

    photo = load_image(image_path)

    scale = min(max_w / photo.width, max_h / photo.height)
    width, height = max(1, int(photo.width * scale)), max(1, int(photo.height * scale))
    photo = photo.resize((width, height), Image.LANCZOS)

    # 居中放置
    offset_x = left + (max_w - width) // 2
    offset_y = top + (max_h - height) // 2

    canvas.paste(photo, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)
    # 细边框，让缩略图从白底上分出来
    draw.rectangle([offset_x, offset_y, offset_x + width, offset_y + height], outline=COLOR_LINE, width=2)

    for index, region in enumerate(card.regions):
        color = REGION_COLORS[index % len(REGION_COLORS)]
        x1, y1, x2, y2 = region.bbox.to_pixels(width, height)
        x1, y1 = x1 + offset_x, y1 + offset_y
        x2, y2 = x2 + offset_x, y2 + offset_y

        # 画三遍不同粗细，让框在深浅底色上都看得清
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=9)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)

        # 编号标签贴在框的左上角；如果框太靠上，就贴到框内侧
        label = region.region_id
        label_font = _font(26, bold=True)
        label_w = int(draw.textlength(label, font=label_font)) + 20
        label_h = 34
        label_y = y1 - label_h if y1 - label_h >= offset_y else y1

        label_x = min(x1, right - label_w)
        label_y = min(label_y, bottom - label_h)
        draw.rectangle([label_x, label_y, label_x + label_w, label_y + label_h], fill=color)
        draw.text((label_x + 10, label_y + 3), label, font=label_font, fill=(255, 255, 255))


def _draw_badge(draw: ImageDraw.ImageDraw, text: str, right: int, y: int, ok: bool) -> None:
    """右上角的状态徽章（待人工复核 / 已复核）。"""
    font = _font(26, bold=True)
    width = int(draw.textlength(text, font=font)) + 44
    height = 52
    color = COLOR_OK if ok else COLOR_WARN

    draw.rounded_rectangle(
        [right - width, y, right, y + height], radius=10, outline=color, width=3
    )
    draw.text((right - width + 22, y + 9), text, font=font, fill=color)


def _content_blocks(card: Card, size: int):
    """返回全部文字及样式；不做切片、省略或语义压缩。"""
    small = max(23, size - 4)
    blocks = []
    def text(value, font_size=size, color=COLOR_TEXT, bold=False, gap=3):
        blocks.append((value, font_size, bold, color, gap))
    def section(value):
        text(value, size + 5, COLOR_MAIN, True, 12)
    section("总述")
    text(card.summary, gap=18)
    section("值得关注的区域")
    for i, region in enumerate(card.regions):
        text(f"{region.region_id}　{region.disease}　模型自评置信度：{region.confidence}",
             size + 2, REGION_COLORS[i], True, 5)
        text("观察：" + region.observation)
        text("影响：" + region.impact)
        if region.review_notes:
            text("复核：" + "；".join(region.review_notes), small, COLOR_WARN)
        text("", small, gap=5)
    section("保护与数字展示建议")
    for i, rec in enumerate(card.recommendations, 1):
        text(f"建议 {i}：{rec.action}")
        text("依据：" + rec.basis, small, COLOR_MUTED)
        text("风险：" + rec.risk, small, COLOR_WARN, gap=12)
    section("不确定性说明")
    for item in card.limitations or ["模型未列出额外局限，不代表不存在不确定性"]:
        text("· " + item, small, COLOR_MUTED)
    section("AI 生成与人工复核")
    text("AI 生成：" + "；".join(card.attribution.ai_generated), small)
    text("需人工复核：" + "；".join(card.attribution.human_review_required), small)
    reviewed = "；".join(card.attribution.human_reviewed) or "无"
    text(f"已人工复核：{reviewed}；复核人：{card.attribution.reviewer or '未填写'}", small)
    section("图片与工具来源")
    p = card.provenance
    text(f"图片来源：{p.image_source}；授权：{p.image_license}", small, COLOR_MUTED)
    if p.scene_note:
        text("场景资料：" + p.scene_note, small, COLOR_MUTED)
    text(f"模型：{'、'.join(p.models)}；平台：{p.platform}", small, COLOR_MUTED)
    text(f"提示词版本：{p.prompt_version}；工具：{'、'.join(p.tools) or '无'}", small, COLOR_MUTED)
    text(f"运行编号：{p.run_id}", small, COLOR_MUTED)
    return blocks


def _measure_blocks(draw, blocks, width):
    measured = []
    height = 0
    for text, size, bold, color, gap in blocks:
        font = _font(size, bold)
        lines = _wrap(draw, text, font, width) if text else []
        line_height = size + 10
        measured.append((lines, font, line_height, color, gap))
        height += len(lines) * line_height + gap
    return measured, height


def render_card(card: Card, image_path: str | Path, out_pdf: str | Path) -> Path:
    """先测量全部正文，再排一页 A4。无法容纳时失败，绝不静默截字。"""
    image_path, out_pdf = Path(image_path), Path(out_pdf)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    width = PAGE_WIDTH - 2 * MARGIN
    right = PAGE_WIDTH - MARGIN
    draw.text((MARGIN, 110), "AI 壁画观察卡", font=_font(56, True), fill=COLOR_MAIN)
    _draw_badge(draw, card.review_status, right, 120, card.review_status != "待人工复核")
    header, header_height = _measure_blocks(draw, [
        (f"{card.image_id} · {card.provenance.generated_at[:10]}", 26, False, COLOR_MUTED, 12)
    ], width)
    image_top = 200 + header_height + 20
    footer, footer_height = _measure_blocks(draw, [(card.disclaimer, 23, False, COLOR_MUTED, 0)], width)
    footer_y = PAGE_HEIGHT - MARGIN - footer_height
    for size in (30, 28, 26):
        measured, body_height = _measure_blocks(draw, _content_blocks(card, size), width)
        thumbnail_height = min(1040, footer_y - 50 - image_top - 75 - body_height)
        if thumbnail_height >= 450:
            break
    else:
        raise ValueError("完整内容超出一页 A4 的可读版面；请缩短文字后重新渲染。完整 JSON 已保留，未截断导出。")
    def paint(parts, y):
        for lines, font, line_height, color, gap in parts:
            for line in lines:
                draw.text((MARGIN, y), line, font=font, fill=color)
                y += line_height
            y += gap
        return y
    paint(header, 200)
    _paste_thumbnail(canvas, image_path, card, (MARGIN, image_top, right, image_top + thumbnail_height))
    caption = "缩略图按比例缩放；区域框为 AI 辅助定位，复核状态见页首；置信度未经统计校准。"
    draw.text((MARGIN, image_top + thumbnail_height + 10), caption, font=_font(23), fill=COLOR_MUTED)
    end_y = paint(measured, image_top + thumbnail_height + 65)
    if end_y > footer_y - 30:
        raise ValueError("正文与页脚重叠，未输出截断内容")
    draw.line((MARGIN, footer_y - 20, right, footer_y - 20), fill=COLOR_LINE, width=2)
    paint(footer, footer_y)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_pdf, "PDF", resolution=300.0)
    canvas.save(out_pdf.with_suffix(".png"), "PNG")
    return out_pdf
