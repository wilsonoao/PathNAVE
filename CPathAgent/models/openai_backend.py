import base64
import os
from io import BytesIO
from typing import List, Optional

from PIL import Image

from .base_model import BaseModel


class OpenAIBackend(BaseModel):
    """
    Backend that routes calls to the OpenAI API (or any OpenAI-compatible endpoint).

    text  → text_model  (e.g. gpt-4o-mini)
    image → vlm_model   (e.g. gpt-4o  –  vision-capable)

    API key is read from:
      1. Constructor argument ``api_key``
      2. Environment variable ``OPENAI_API_KEY``
    """

    def __init__(
        self,
        text_model: str = "gpt-4o-mini",
        vlm_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with:  pip install openai"
            ) from e

        self.text_model = text_model
        self.vlm_model = vlm_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Pass it via --openai-api-key or set the OPENAI_API_KEY environment variable."
            )

        client_kwargs = {"api_key": resolved_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url

        super().__init__()
        self.client = OpenAI(**client_kwargs)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _img_to_b64_url(image: Image.Image) -> str:
        """Encode a PIL Image as a base64 data URL (PNG)."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def _chat(self, model: str, messages: list, max_new_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            # temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    # ── public API ────────────────────────────────────────────────────────────

    def _generate_text(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        print(f"[LLM] text → openai/{self.text_model}  (max_tokens={max_tokens})")
        messages = [{"role": "user", "content": prompt}]
        return self._chat(self.text_model, messages, max_new_tokens=max_tokens)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        # content: list = [{"type": "text", "text": prompt}]
        # for img in images:
        #     content.append({
        #         "type": "image_url",
        #         "image_url": {"url": self._img_to_b64_url(img)},
        #     })
        # messages = [{"role": "user", "content": content}]
        return None

    # ── convenience ───────────────────────────────────────────────────────────

    def list_models(self) -> List[str]:
        """Return available model IDs from the API."""
        models = self.client.models.list()
        return sorted(m.id for m in models.data)

    def check_availability(self) -> dict:
        """Verify that the configured models are accessible."""
        try:
            available = self.list_models()
            text_ok = self.text_model in available
            vlm_ok = self.vlm_model in available
        except Exception as exc:
            return {
                "text_model": self.text_model,
                "text_model_ok": False,
                "vlm_model": self.vlm_model,
                "vlm_model_ok": False,
                "error": str(exc),
            }
        return {
            "text_model": self.text_model,
            "text_model_ok": text_ok,
            "vlm_model": self.vlm_model,
            "vlm_model_ok": vlm_ok,
            "available_models": available,
        }
