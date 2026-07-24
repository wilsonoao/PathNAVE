"""
WSI loading utilities.

Tries openslide first (supports .svs / .ndpi / .tiff / .mrxs …).
Falls back to PIL for standard image formats.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
    _OPENSLIDE = True
except ImportError:
    _OPENSLIDE = False

# Extensions that openslide can handle (prefer openslide for these when available)
_OPENSLIDE_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu", ".bif"}

# Extensions handled only by PIL (small standard images)
_PIL_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class WSILoader:
    """
    Thin wrapper around either openslide or PIL, exposing:
      - get_thumbnail()
      - get_region(x1, y1, x2, y2)   ← normalised [0, 1] coords
      - partition_grid(patch_size_px) ← list of (patch_id,x1,y1,x2,y2) tiles with overlap
    """

    def __init__(self, path: str, thumbnail_scale: int = 32):
        self.path = Path(path)
        self.thumbnail_scale = thumbnail_scale

        ext = self.path.suffix.lower()

        # Prefer openslide for WSI formats when available; fall back to PIL otherwise
        use_openslide = _OPENSLIDE and (ext in _OPENSLIDE_EXTS)

        if use_openslide:
            self._backend = "openslide"
            self._slide = openslide.OpenSlide(path)
            self.width, self.height = self._slide.dimensions
        else:
            self._backend = "pil"
            # Disable PIL decompression bomb check for large-but-legitimate images
            import PIL.Image as _PILImage
            _PILImage.MAX_IMAGE_PIXELS = None
            self._img = Image.open(path).convert("RGB")
            self.width, self.height = self._img.size

    # ── thumbnail ─────────────────────────────────────────────────────────────

    def get_thumbnail(self, max_dim: int | None = None) -> Image.Image:
        tw = self.width // self.thumbnail_scale
        th = self.height // self.thumbnail_scale
        if max_dim:
            scale = max_dim / max(tw, th)
            tw, th = int(tw * scale), int(th * scale)

        if self._backend == "pil":
            return self._img.resize((tw, th), Image.LANCZOS)
        else:
            return self._slide.get_thumbnail((tw, th)).convert("RGB")

    # ── region by normalised coords ───────────────────────────────────────────

    def get_region(
        self,
        x1: float, y1: float, x2: float, y2: float,
        target_size: int = 1008,
    ) -> Image.Image:
        """
        x1, y1, x2, y2 are in [0, 1].
        Returns an RGB PIL image resized to (target_size × target_size).
        """
        px1 = int(x1 * self.width)
        py1 = int(y1 * self.height)
        px2 = int(x2 * self.width)
        py2 = int(y2 * self.height)
        w = max(px2 - px1, 1)
        h = max(py2 - py1, 1)

        if self._backend == "pil":
            region = self._img.crop((px1, py1, px2, py2))
        else:
            region = self._slide.read_region((px1, py1), 0, (w, h)).convert("RGB")

        return region.resize((target_size, target_size), Image.LANCZOS)

    # ── grid partition ────────────────────────────────────────────────────────

    def partition_grid(
        self,
        patch_size_px: int = 512,
        overlap: float = 0.05,
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Tile the thumbnail into non-fixed-count patches of *patch_size_px × patch_size_px*
        (in thumbnail pixel space) with *overlap* fraction of boundary overlap.

        Thumbnail size = full-WSI size / thumbnail_scale.

        Returns a list of (patch_id, x1, y1, x2, y2) tuples where coordinates
        are normalised to [0, 1] relative to the **full** WSI dimensions.
        """
        # Thumbnail dimensions (pixels)
        tw = self.width  // self.thumbnail_scale
        th = self.height // self.thumbnail_scale

        overlap_px = int(patch_size_px * overlap)
        step_px    = patch_size_px - overlap_px          # e.g. 512 - 25 = 487

        tiles = []
        pid = 0
        y = 0
        while y < th:
            x = 0
            while x < tw:
                # Clamp to thumbnail boundary
                x2 = min(x + patch_size_px, tw)
                y2 = min(y + patch_size_px, th)
                # Normalise to [0, 1] of full WSI
                tiles.append((
                    pid,
                    x  / tw,
                    y  / th,
                    x2 / tw,
                    y2 / th,
                ))
                pid += 1
                x  += step_px
            y += step_px
        return tiles

    def close(self):
        if self._backend == "openslide":
            self._slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
