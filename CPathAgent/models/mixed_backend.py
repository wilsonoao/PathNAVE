from typing import List
from PIL import Image

from .base_model import BaseModel


class MixedBackend(BaseModel):
    """
    Delegates text-only calls to one backend and vision calls to another.

    Example:
        text  → HFBackend  (Qwen3-4B local)
        image → OpenAIBackend (gpt-4o)
    """

    def __init__(self, text_backend: BaseModel, vlm_backend: BaseModel):
        super().__init__()
        self._text = text_backend
        self._vlm = vlm_backend

    def _generate_text(self, prompt: str, **kwargs) -> str:
        return self._text._generate_text(prompt, **kwargs)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        return self._vlm._generate_with_images(prompt, images, **kwargs)
