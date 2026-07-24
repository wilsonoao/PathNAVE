import time
from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class BaseModel(ABC):
    """Common interface for all model backends."""

    def __init__(self):
        self._stats = {"llm_calls": 0, "vlm_calls": 0, "llm_time": 0.0, "vlm_time": 0.0}

    def reset_stats(self):
        self._stats = {"llm_calls": 0, "vlm_calls": 0, "llm_time": 0.0, "vlm_time": 0.0}

    def get_stats(self) -> dict:
        s = self._stats
        return {
            "llm_calls": s["llm_calls"],
            "vlm_calls": s["vlm_calls"],
            "llm_time_s": round(s["llm_time"], 3),
            "vlm_time_s": round(s["vlm_time"], 3),
            "total_inference_time_s": round(s["llm_time"] + s["vlm_time"], 3),
        }

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Run a text-only prompt through the text model, tracking call count and time."""
        t0 = time.time()
        result = self._generate_text(prompt, **kwargs)
        self._stats["llm_calls"] += 1
        self._stats["llm_time"] += time.time() - t0
        return result

    def generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        """Run a multimodal prompt through the VLM, tracking call count and time."""
        t0 = time.time()
        result = self._generate_with_images(prompt, images, **kwargs)
        self._stats["vlm_calls"] += 1
        self._stats["vlm_time"] += time.time() - t0
        return result

    @abstractmethod
    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Run a text-only prompt through the **text model** (Qwen3 4B)."""

    @abstractmethod
    def _generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        """Run a multimodal prompt through the **VLM** (patho_r1)."""
