"""
Image annotation helpers used across all three agents.
"""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


# ── coordinate grid annotation ─────────────────────────────────────────────


def annotate_grid(
    image: Image.Image,
    intervals: float = 0.1,
    line_color: str = "blue",
    line_width: int = 1,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw a semi-transparent coordinate grid on *image* and label every
    intersection with its relative (x, y) position (e.g. "0.3,0.2").
    Returns a copy of the image – the original is not modified.
    """
    img = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    steps = [round(i * intervals, 2) for i in range(int(1 / intervals) + 1)]

    # vertical lines
    for x_frac in steps:
        px = int(x_frac * W)
        draw.line([(px, 0), (px, H)], fill=(*_hex_to_rgb(line_color), 120), width=line_width)

    # horizontal lines
    for y_frac in steps:
        py = int(y_frac * H)
        draw.line([(0, py), (W, py)], fill=(*_hex_to_rgb(line_color), 120), width=line_width)

    # labels at intersections
    for x_frac in steps:
        for y_frac in steps:
            px = int(x_frac * W)
            py = int(y_frac * H)
            label = f"{x_frac},{y_frac}"
            draw.text((px + 2, py + 2), label, fill=(0, 0, 255, 200), font=font)

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


# ── thumbnail with patch-ID labels ─────────────────────────────────────────


def create_thumbnail_with_ids(
    thumbnail: Image.Image,
    tiles: List[Tuple[int, float, float, float, float]],
    font_size: int = 14,
    border_color: str = "red",
) -> Image.Image:
    """
    Draw patch borders and IDs onto *thumbnail*.

    tiles : list of (patch_id, x1, y1, x2, y2)  – normalised coords
    """
    img = thumbnail.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for pid, x1, y1, x2, y2 in tiles:
        px1, py1 = int(x1 * W), int(y1 * H)
        px2, py2 = int(x2 * W), int(y2 * H)
        draw.rectangle([px1, py1, px2, py2], outline=border_color, width=1)
        # ID label at top-left of each tile
        draw.text((px1 + 2, py1 + 2), str(pid), fill=border_color, font=font)

    return img


# ── crop a view by normalised coords ───────────────────────────────────────


def crop_view(
    image: Image.Image,
    x1: float, y1: float, x2: float, y2: float,
    target_size: int = 1008,
) -> Image.Image:
    """
    Crop *image* by normalised bbox and resize to *target_size* × *target_size*.
    """
    W, H = image.size
    box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    cropped = image.crop(box)
    return cropped.resize((target_size, target_size), Image.LANCZOS)


# ── uniform resize ─────────────────────────────────────────────────────────


def resize_for_model(image: Image.Image, target_size: int = 1008) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)


# ── helper ─────────────────────────────────────────────────────────────────


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    named = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    if color.lower() in named:
        return named[color.lower()]
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore
