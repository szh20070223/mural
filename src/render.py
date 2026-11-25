# src/render.py
# =============================================================================
# 渲染：Card -> 一页 A4 PDF
# =============================================================================
# 为什么用 Pillow 手绘，而不是 HTML 转 PDF：
#   比赛要求"PDF 超过 1 页只评第 1 页"，所以版面必须**确定能塞进一页**。
#   用 HTML 排版，内容一多就会溢出到第二页，而且不同机器渲染结果不一致。
#   Pillow 画布是固定像素，超了就截断、不够就留白，永远不会变成两页。
#   另外它不依赖浏览器/GTK 之类的运行库，现场换台电脑也能跑。
#
# 版式按 docs/card-contract.md 第 4 节的排版预算来：
#   页眉 8% / 缩略图 38% / 区域说明 22% / 建议 12% / 其他 18% / 页脚 2%
# =============================================================================

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.schemas import Card, Region

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

    超过 max_lines 就截断并加省略号——这是"保证一页"的最后一道闸。
    宁可少显示几个字，也不能让内容溢出到第二页。
    """
    lines = _wrap(draw, text, font, max_width)

    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"

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

    with Image.open(image_path) as source:
        photo = source.convert("RGB")

    scale = min(max_w / photo.width, max_h / photo.height)
    width, height = int(photo.width * scale), int(photo.height * scale)
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

        draw.rectangle([x1, label_y, x1 + label_w, label_y + label_h], fill=color)
        draw.text((x1 + 10, label_y + 3), label, font=label_font, fill=(255, 255, 255))


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


def render_card(card: Card, image_path: str | Path, out_pdf: str | Path) -> Path:
    """把一张 Card 画成 A4 PDF，同时输出一张 PNG 方便快速预览。

    返回 PDF 路径。
    """
    image_path = Path(image_path)
    out_pdf = Path(out_pdf)

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    content_width = PAGE_WIDTH - MARGIN * 2
    right_edge = PAGE_WIDTH - MARGIN

    # ------------------------------------------------------------------
    # 页眉
    # ------------------------------------------------------------------
    draw.text((MARGIN, 130), "AI 壁画观察卡", font=_font(56, bold=True), fill=COLOR_MAIN)

    subtitle = f"{card.image_id}　·　{card.provenance.generated_at[:10]}"
    draw.text((MARGIN, 214), subtitle, font=_font(28), fill=COLOR_MUTED)

    _draw_badge(
        draw,
        card.review_status,
        right_edge,
        132,
        ok=card.review_status != "待人工复核",
    )

    draw.line([MARGIN, 286, right_edge, 286], fill=COLOR_LINE, width=3)

    # ------------------------------------------------------------------
    # 缩略图（占 38%）
    # ------------------------------------------------------------------
    thumb_box = (MARGIN, 316, right_edge, 1356)
    _paste_thumbnail(canvas, image_path, card, thumb_box)

    # 图注：提醒这不是原尺寸，避免被当成"精确测量图"
    draw.text(
        (MARGIN, 1368),
        "缩略图（按比例缩放）；彩色框对应下方 R1–R3 区域，坐标由模型给出并已人工复核前校准",
        font=_font(22),
        fill=COLOR_MUTED,
    )

    y = 1416

    # ------------------------------------------------------------------
    # 总述
    # ------------------------------------------------------------------
    y = _draw_section_title(draw, "总述", MARGIN, y, content_width)
    y = _draw_text_block(draw, card.summary, _font(30), MARGIN, y, content_width, 3, 44)

    y += 24

    # ------------------------------------------------------------------
    # 区域说明（1-3 个）
    # ------------------------------------------------------------------
    y = _draw_section_title(draw, "值得关注的区域", MARGIN, y, content_width)

    for index, region in enumerate(card.regions):
        color = REGION_COLORS[index % len(REGION_COLORS)]
        block_top = y

        # 左侧色条，和缩略图上的框同色
        draw.rectangle([MARGIN, block_top, MARGIN + 10, block_top + 150], fill=color)

        # 编号 + 病害 + 置信度（紧挨着排，不放到最右边——
        # 放右边会孤零零地飘着，看起来像另一栏的东西）
        title_text = f"{region.region_id}　{region.disease}"
        title_font = _font(32, bold=True)
        draw.text((MARGIN + 30, block_top - 4), title_text, font=title_font, fill=COLOR_TEXT)

        title_width = int(draw.textlength(title_text, font=title_font))
        draw.text(
            (MARGIN + 30 + title_width + 30, block_top + 6),
            f"置信度：{region.confidence}",
            font=_font(24),
            fill=COLOR_MUTED,
        )

        text_x = MARGIN + 30
        text_width = right_edge - text_x
        cursor = block_top + 48

        cursor = _draw_text_block(
            draw,
            f"观察：{region.observation}",
            _font(27),
            text_x,
            cursor,
            text_width,
            2,
            38,
        )
        cursor = _draw_text_block(
            draw,
            f"影响：{region.impact}",
            _font(27),
            text_x,
            cursor,
            text_width,
            2,
            38,
        )
        if region.review_notes:
            cursor = _draw_text_block(
                draw,
                "复核：" + "；".join(region.review_notes),
                _font(25),
                text_x,
                cursor,
                text_width,
                2,
                34,
                fill=COLOR_WARN,
            )

        y = max(cursor, block_top + 150) + 26

    # ------------------------------------------------------------------
    # 建议
    # ------------------------------------------------------------------
    y = _draw_section_title(draw, "保护与数字展示建议", MARGIN, y, content_width)

    for index, rec in enumerate(card.recommendations, start=1):
        number = f"建议 {index}"
        draw.text((MARGIN, y - 2), number, font=_font(28, bold=True), fill=COLOR_MAIN)

        text_x = MARGIN + 150
        text_width = right_edge - text_x
        cursor = y - 4

        cursor = _draw_text_block(draw, rec.action, _font(28), text_x, cursor, text_width, 2, 40)
        cursor = _draw_text_block(
            draw, f"依据：{rec.basis}", _font(25), text_x, cursor, text_width, 2, 34, COLOR_MUTED
        )
        cursor = _draw_text_block(
            draw, f"风险：{rec.risk}", _font(25), text_x, cursor, text_width, 2, 34, COLOR_WARN
        )

        y = max(cursor, y + 120) + 16

    # ------------------------------------------------------------------
    # 局限 + 归属 + 来源
    # ------------------------------------------------------------------
    y += 10
    y = _draw_section_title(draw, "不确定性说明", MARGIN, y, content_width)

    # 一条一行，比挤成一句用分号连起来好读得多。
    # 每条限一行：字段本身有 40 字上限，这里宽度放得下。
    for item in card.limitations or ["无"]:
        y = _draw_text_block(
            draw, f"· {item}", _font(25), MARGIN, y, content_width, 1, 34, COLOR_MUTED
        )
    y += 20

    # 归属框：用一个浅色底块把它和普通正文分开，这是评分点，要显眼
    box_top = y
    padding = 26
    inner_x = MARGIN + padding
    inner_width = content_width - padding * 2

    reviewed = (
        "；".join(card.attribution.human_reviewed)
        if card.attribution.human_reviewed
        else "无（本卡尚未经人工复核）"
    )

    # 先把四段文字折行算出来、量好总高度，再画底块。
    # 如果底块高度写死，内容短了会留一大片空白，长了又盖不住文字——
    # 这个版式要一页装完，任何写死的数字最后都会出问题。
    blocks = [
        ("AI 生成／人工复核：", _font(27, bold=True), 1, 36, COLOR_MAIN),
        (
            "· AI 生成：" + "；".join(card.attribution.ai_generated),
            _font(23),
            3,
            32,
            COLOR_TEXT,
        ),
        (
            "· 必须人工复核：" + "；".join(card.attribution.human_review_required),
            _font(23),
            3,
            32,
            COLOR_TEXT,
        ),
        (f"· 已人工复核：{reviewed}", _font(23), 2, 32, COLOR_MUTED),
    ]

    measured: list[tuple[list[str], ImageFont.FreeTypeFont, int, tuple[int, int, int]]] = []
    box_height = padding * 2
    for text, font, max_lines, line_height, color in blocks:
        lines = _wrap(draw, text, font, inner_width)[:max_lines]
        measured.append((lines, font, line_height, color))
        box_height += len(lines) * line_height

    draw.rectangle([MARGIN, box_top, right_edge, box_top + box_height], fill=COLOR_BG_SOFT)

    cursor = box_top + padding
    for lines, font, line_height, color in measured:
        for line in lines:
            draw.text((inner_x, cursor), line, font=font, fill=color)
            cursor += line_height

    y = box_top + box_height + 34

    # ------------------------------------------------------------------
    # 来源披露
    # ------------------------------------------------------------------
    y = _draw_section_title(draw, "图片与工具来源", MARGIN, y, content_width)

    provenance_lines = [
        f"图片来源：{card.provenance.image_source}",
        f"授权说明：{card.provenance.image_license}",
        f"模型：{'、'.join(card.provenance.models)}　·　平台：{card.provenance.platform}",
        f"提示词版本：{card.provenance.prompt_version}",
        f"其他工具：{'、'.join(card.provenance.tools) if card.provenance.tools else '无'}",
        f"运行编号：{card.provenance.run_id}",
    ]
    if card.provenance.scene_note:
        provenance_lines.insert(2, f"场景说明：{card.provenance.scene_note}")

    for line in provenance_lines:
        draw.text((MARGIN, y), line, font=_font(23), fill=COLOR_MUTED)
        y += 34

    # ------------------------------------------------------------------
    # 页脚免责声明
    # ------------------------------------------------------------------
    footer_y = PAGE_HEIGHT - 150
    draw.line([MARGIN, footer_y - 24, right_edge, footer_y - 24], fill=COLOR_LINE, width=2)
    _draw_text_block(draw, card.disclaimer, _font(21), MARGIN, footer_y, content_width, 2, 30, COLOR_MUTED)

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_pdf, "PDF", resolution=300.0)

    # PNG 预览：PDF 在有的阅读器里不方便截图，做视频时更需要 PNG
    canvas.save(out_pdf.with_suffix(".png"), "PNG")

    return out_pdf
