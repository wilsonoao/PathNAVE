import base64
import json
import re
from io import BytesIO
from typing import List

import requests
from PIL import Image

from .base_model import BaseModel


class OllamaBackend(BaseModel):
    """
    Backend that routes calls to a locally-running Ollama server.

    text  → text_model  (e.g. qwen3:4b)
    image → vlm_model   (e.g. patho_r1)
    """

    def __init__(
        self,
        text_model: str,
        vlm_model: str,
        host: str = "http://localhost:11434",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        qwen3_thinking: bool = False,
    ):
        super().__init__()
        self.text_model = text_model
        self.vlm_model = vlm_model
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.qwen3_thinking = qwen3_thinking

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _img_to_b64(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _call(
        self,
        model: str,
        prompt: str,
        images_b64: List[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                "temperature": self.temperature,
            },
        }
        if images_b64:
            payload["images"] = images_b64

        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        text = resp.json()["response"]

        # Strip Qwen3 thinking tokens if present (<think>…</think>)
        if not self.qwen3_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    # ── public API ────────────────────────────────────────────────────────────

    def _generate_text(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        print(f"[LLM] text → ollama/{self.text_model}  (max_tokens={max_tokens})")
        # Qwen3 in Ollama defaults to thinking mode.  For structured text tasks
        # (JSON clustering, repair) thinking consumes all of max_new_tokens before
        # producing any output.  /no_think disables the thinking block entirely.
        if not self.qwen3_thinking:
            prompt = "/no_think\n" + prompt
        return self._call(self.text_model, prompt, max_new_tokens=max_tokens)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        print(f"[LLM] vision → ollama/{self.vlm_model}")
        b64_list = [self._img_to_b64(img) for img in images]
        return self._call(self.vlm_model, prompt, images_b64=b64_list)

    # ── convenience ───────────────────────────────────────────────────────────

    def list_models(self) -> List[str]:
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def check_availability(self) -> dict:
        available = self.list_models()
        return {
            "text_model": self.text_model,
            "text_model_ok": self.text_model in available,
            "vlm_model": self.vlm_model,
            "vlm_model_ok": self.vlm_model in available,
            "available_models": available,
        }
