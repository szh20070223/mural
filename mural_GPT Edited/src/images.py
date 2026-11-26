"""供模型输入和卡片渲染共用的图像方向及颜色处理。"""

from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    """按 EXIF 方向读取，透明区域合成到白底，返回独立 RGB 图像。"""
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("请先从多帧图片中选出单帧再分析")
        photo = ImageOps.exif_transpose(source)
        if "A" in photo.getbands() or "transparency" in photo.info:
            rgba = photo.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return photo.convert("RGB")
