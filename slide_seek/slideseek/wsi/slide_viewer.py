"""
WSI slide navigation wrapper using OpenSlide.
Provides region extraction at specified coordinates and magnification levels.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TissueBBox:
    """Bounding box for a tissue region."""
    region_id: int
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int

    @property
    def center_x(self) -> int:
        return (self.top_left_x + self.bottom_right_x) // 2

    @property
    def center_y(self) -> int:
        return (self.top_left_y + self.bottom_right_y) // 2

    @property
    def width(self) -> int:
        return self.bottom_right_x - self.top_left_x

    @property
    def height(self) -> int:
        return self.bottom_right_y - self.top_left_y

    def __str__(self):
        return (
            f"Region {self.region_id}: "
            f"top-left ({self.top_left_x}, {self.top_left_y}), "
            f"bottom-right ({self.bottom_right_x}, {self.bottom_right_y})"
        )


@dataclass
class ROI:
    """A region of interest extracted from a WSI."""
    x: int
    y: int
    magnification: float
    size: int
    image: np.ndarray   # HxWx3 uint8
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"region-mag_{self.magnification}-x_{self.x}-y_{self.y}"

    def to_pil(self):
        from PIL import Image
        return Image.fromarray(self.image)


class SlideViewer:
    """
    OpenSlide-based WSI viewer.
    Handles coordinate mapping between different magnification levels.
    """

    def __init__(self, slide_path: str, roi_size: int = 896):
        self.slide_path = Path(slide_path)
        self.roi_size = roi_size
        self._slide = None
        self._base_magnification: Optional[float] = None
        self._tissue_bboxes: list[TissueBBox] = []
        self._thumbnail: Optional[np.ndarray] = None
        self._load()

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #
    def _load(self):
        try:
            import openslide
            self._slide = openslide.OpenSlide(str(self.slide_path))
            self._base_magnification = self._get_base_magnification()
            logger.info(
                f"Loaded WSI: {self.slide_path.name} | "
                f"Dimensions: {self._slide.dimensions} | "
                f"Base mag: {self._base_magnification}x"
            )
            self._compute_tissue_bboxes()
            self._thumbnail = self._extract_thumbnail()
        except ImportError:
            logger.error("openslide-python not installed. Install with: pip install openslide-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load slide {self.slide_path}: {e}")
            raise

    def _get_base_magnification(self) -> float:
        """Extract objective magnification from slide metadata."""
        props = self._slide.properties
        for key in [
            "openslide.objective-power",
            "aperio.AppMag",
            "hamamatsu.XResolution",
        ]:
            val = props.get(key)
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    pass
        logger.warning("Could not determine base magnification, assuming 40x")
        return 40.0

    def _compute_tissue_bboxes(self):
        """
        Detect tissue regions using basic thresholding (no TRIDENT dependency).
        For production, replace with TRIDENT or your tissue detection pipeline.
        """
        try:
            thumbnail = self._extract_thumbnail(size=1024)
            import cv2

            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
            # Otsu threshold to find tissue
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Remove small noise
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            scale_x = self._slide.dimensions[0] / thumbnail.shape[1]
            scale_y = self._slide.dimensions[1] / thumbnail.shape[0]

            min_area = 0.001 * thumbnail.shape[0] * thumbnail.shape[1]

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                self._tissue_bboxes.append(TissueBBox(
                    region_id=i,
                    top_left_x=int(x * scale_x),
                    top_left_y=int(y * scale_y),
                    bottom_right_x=int((x + w) * scale_x),
                    bottom_right_y=int((y + h) * scale_y),
                ))

            logger.info(f"Detected {len(self._tissue_bboxes)} tissue regions")

        except ImportError:
            # Fallback: single bbox covering whole slide
            w, h = self._slide.dimensions
            self._tissue_bboxes = [TissueBBox(0, 0, 0, w, h)]
            logger.warning("cv2 not available; using full-slide bounding box")

    def _extract_thumbnail(self, size: int = 512) -> np.ndarray:
        """Extract a thumbnail of the whole slide."""
        thumb = self._slide.get_thumbnail((size, size))
        return np.array(thumb.convert("RGB"))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    @property
    def dimensions(self) -> tuple[int, int]:
        """(width, height) of the slide at base magnification."""
        return self._slide.dimensions

    @property
    def tissue_bboxes(self) -> list[TissueBBox]:
        return self._tissue_bboxes

    @property
    def thumbnail(self) -> np.ndarray:
        if self._thumbnail is None:
            self._thumbnail = self._extract_thumbnail()
        return self._thumbnail

    def get_slide_context(self) -> str:
        """Returns a text description of the slide for the supervisor agent."""
        w, h = self.dimensions
        bbox_descriptions = "\n".join(f"  - {bbox}" for bbox in self._tissue_bboxes)
        return (
            f"Slide dimensions: {w} x {h} pixels (at base {self._base_magnification}x)\n"
            f"Detected tissue regions ({len(self._tissue_bboxes)}):\n{bbox_descriptions}"
        )

    def navigate(self, x: int, y: int, magnification: float) -> ROI:
        """
        Extract a region of interest from the slide.

        Args:
            x, y:          Center coordinates at BASE magnification.
            magnification: Target magnification (e.g. 1.25, 5, 20).

        Returns:
            ROI object with the extracted image.
        """
        level, actual_mag = self._mag_to_level(magnification)
        # Compute pixel size at this level
        downsample = self._slide.level_downsamples[level]
        size_at_level = self.roi_size
        # Convert center to top-left at level 0
        half = int((size_at_level * downsample) / 2)
        read_x = max(0, x - half)
        read_y = max(0, y - half)

        region = self._slide.read_region(
            location=(read_x, read_y),
            level=level,
            size=(size_at_level, size_at_level),
        )
        img = np.array(region.convert("RGB"))

        return ROI(
            x=x, y=y,
            magnification=actual_mag,
            size=size_at_level,
            image=img,
        )

    def _mag_to_level(self, target_mag: float) -> tuple[int, float]:
        """Find the best OpenSlide level for the target magnification."""
        target_downsample = self._base_magnification / target_mag
        best_level = self._slide.get_best_level_for_downsample(target_downsample)
        actual_downsample = self._slide.level_downsamples[best_level]
        actual_mag = self._base_magnification / actual_downsample
        return best_level, actual_mag

    def close(self):
        if self._slide is not None:
            self._slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------ #
#  Mock viewer for testing without a real WSI                         #
# ------------------------------------------------------------------ #
class MockSlideViewer:
    """Drop-in replacement for testing without a real WSI file."""

    def __init__(self, width: int = 50000, height: int = 40000):
        self._w = width
        self._h = height
        self._tissue_bboxes = [
            TissueBBox(0, 1000, 1000, 25000, 20000),
            TissueBBox(1, 26000, 1000, 49000, 39000),
        ]
        self._thumbnail = np.random.randint(180, 230, (512, 512, 3), dtype=np.uint8)

    @property
    def dimensions(self):
        return (self._w, self._h)

    @property
    def tissue_bboxes(self):
        return self._tissue_bboxes

    @property
    def thumbnail(self):
        return self._thumbnail

    def get_slide_context(self) -> str:
        return (
            f"[MOCK] Slide dimensions: {self._w} x {self._h}\n"
            f"Tissue regions:\n"
            + "\n".join(f"  - {b}" for b in self._tissue_bboxes)
        )

    def navigate(self, x: int, y: int, magnification: float) -> ROI:
        """Returns a random RGB patch."""
        img = np.random.randint(150, 255, (896, 896, 3), dtype=np.uint8)
        return ROI(x=x, y=y, magnification=magnification, size=896, image=img)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
