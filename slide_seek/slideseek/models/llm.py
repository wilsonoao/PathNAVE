"""
LLM wrappers — Ollama and HuggingFace backends.
Default model: Qwen/Qwen3-4B (HF) or qwen3:4b (Ollama)
"""
import json
import logging
import re
import threading
from typing import Any, Optional
import requests

from slideseek.config import OllamaConfig, HFLLMConfig, HFVisionLLMConfig, OpenAILLMConfig

logger = logging.getLogger(__name__)

# Global lock: ensures only one model occupies GPU at a time.
# patho_model._inference_lock also acquires this before moving to GPU.
_gpu_lock = threading.Lock()

# Forward declaration so _fix_json_with_qwen3 can reference it at call time.
_openaillm_instance = None


_JSON_FIX_PROMPT = (
    "The following text was supposed to be a valid JSON object but could not be parsed:\n\n"
    "{bad_output}\n\n"
    "Rewrite it as a valid JSON object only. "
    "No markdown code fences, no extra text, no trailing commas."
)


def _fix_json_with_qwen3(bad_output: str, source: str) -> dict:
    """Send malformed JSON to an available LLM to fix. Returns parsed dict or {"raw": bad_output}."""
    try:
        # Prefer OpenAI if it's already initialised (avoids loading a local model)
        if _openaillm_instance is not None:
            fixer = _openaillm_instance
        else:
            fixer = get_hfllm()
        fix_messages = [{"role": "user", "content": _JSON_FIX_PROMPT.format(bad_output=bad_output[:800])}]
        raw2 = fixer.chat(fix_messages, json_mode=True)
        return _parse_json_robust(raw2, source=f"{source}-fix")
    except Exception as e:
        logger.error(f"{source}: JSON fix attempt failed: {e}")
        return {"raw": bad_output}


def _parse_json_robust(raw: str, source: str = "LLM") -> dict:
    """
    Try multiple strategies to extract a JSON dict from raw LLM output.
    Returns {"raw": raw} if all strategies fail.
    """
    # 1. Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Extract first {...} block (greedy — outermost braces)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 4. Try to fix truncated JSON by closing open braces/brackets
    candidate = match.group() if match else stripped
    open_braces   = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    if open_braces > 0 or open_brackets > 0:
        fixed = candidate + ("]" * max(open_brackets, 0)) + ("}" * max(open_braces, 0))
        try:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # logger.warning(f"Could not parse JSON from {source} response:\n{raw[:300]}")
    return {"raw": raw}


class OllamaLLM:
    """Simple wrapper around Ollama's /api/chat endpoint."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._check_connection()

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #
    def _check_connection(self):
        try:
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.config.model not in models:
                logger.warning(
                    f"Model '{self.config.model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self.config.model}"
                )
            else:
                logger.info(f"Ollama connected — using model: {self.config.model}")
        except Exception as e:
            logger.warning(f"Cannot reach Ollama at {self.config.base_url}: {e}")

    # ------------------------------------------------------------------ #
    #  Core chat                                                           #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat request to Ollama.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system:   Optional system prompt prepended automatically.
            json_mode: If True, instructs the model to reply in JSON.

        Returns:
            The assistant's reply as a string.
        """
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        # Qwen3 /think suppression (no extra chain-of-thought tokens by default)
        options: dict[str, Any] = {"temperature": self.config.temperature}

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": payload_messages,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        # Qwen3 thinking toggle
        if not self.config.enable_thinking:
            # Append /no_think suffix to last user message to suppress <think> tokens
            for msg in reversed(payload_messages):
                if msg["role"] == "user":
                    if "/no_think" not in msg["content"] and "/think" not in msg["content"]:
                        msg["content"] += " /no_think"
                    break

        try:
            resp = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            # Strip any <think>...</think> blocks that still leak through
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  JSON helpers                                                        #
    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> dict:
        """Like chat() but parses and returns a JSON dict. Retries once if output is not valid JSON."""
        raw = self.chat(messages, system=system, json_mode=True)
        result = _parse_json_robust(raw, source="OllamaLLM")
        if "raw" in result and len(result) == 1:
            # Fallback: retry with doubled token budget via options num_predict
            original_opts = self.config.timeout
            try:
                payload_messages_fallback = []
                if system:
                    payload_messages_fallback.append({"role": "system", "content": system})
                payload_messages_fallback.extend(messages)
                import requests as _req
                fallback_payload = {
                    "model": self.config.model,
                    "messages": payload_messages_fallback,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": self.config.temperature, "num_predict": 4096},
                }
                resp2 = _req.post(
                    f"{self.config.base_url}/api/chat",
                    json=fallback_payload,
                    timeout=self.config.timeout,
                )
                resp2.raise_for_status()
                raw2 = resp2.json()["message"]["content"]
                raw2 = re.sub(r"<think>.*?</think>", "", raw2, flags=re.DOTALL).strip()
                result = _parse_json_robust(raw2, source="OllamaLLM-fallback")
                if "raw" in result and len(result) == 1:
                    result = _fix_json_with_qwen3(raw2, source="OllamaLLM-fallback")
            except Exception as e:
                logger.error(f"OllamaLLM fallback retry failed: {e}")
                result = _fix_json_with_qwen3(raw, source="OllamaLLM")
        return result


class PathoR1LLM:
    """
    Thin wrapper around the already-loaded patho_r1 model.
    Reuses patho_model._model / _processor — no extra VRAM.
    Exposes chat() / chat_json() compatible with HFLLM / HFVisionLLM.
    """

    def __init__(self, max_new_tokens: int = 512, temperature: float = 0.1):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _get_model(self):
        from slideseek.models import patho_model
        patho_model.ensure_model_loaded()
        if patho_model._model_type != "patho_r1":
            raise RuntimeError(
                f"PathoR1LLM requires patho_r1 model but model_type is '{patho_model._model_type}'. "
                "Check that the model loaded successfully."
            )
        return patho_model._model, patho_model._processor

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        images: Optional[list] = None,
        json_mode: bool = False,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        from slideseek.models import patho_model as _patho_module

        model, processor = self._get_model()
        with _patho_module._inference_lock:
            with _gpu_lock:
                _patho_module.to_gpu()
                try:
                    return self._chat_inner(model, processor, messages, system, images, json_mode)
                finally:
                    _patho_module.to_cpu()

    def _chat_inner(self, model, processor, messages, system, images, json_mode) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        payload = []
        if system:
            payload.append({"role": "system", "content": system})

        for i, msg in enumerate(messages):
            is_last_user = (msg["role"] == "user" and i == len(messages) - 1)
            text = msg["content"]
            if json_mode and is_last_user:
                text += "\nReply with valid JSON only, no extra text."

            if i == 0 and images and msg["role"] == "user":
                content = [{"type": "image", "image": img} for img in images]
                content.append({"type": "text", "text": text})
                payload.append({"role": "user", "content": content})
            else:
                payload.append({"role": msg["role"], "content": text})

        text_input = processor.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(payload)
        inputs = processor(
            text=[text_input],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        content = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def chat_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        images: Optional[list] = None,
    ) -> dict:
        raw = self.chat(messages, system=system, images=images, json_mode=True)
        result = _parse_json_robust(raw, source="PathoR1LLM")
        if "raw" in result and len(result) == 1:
            # Fallback: retry with doubled token budget
            original_tokens = self.max_new_tokens
            self.max_new_tokens = original_tokens * 2
            try:
                raw2 = self.chat(messages, system=system, images=images, json_mode=True)
                result = _parse_json_robust(raw2, source="PathoR1LLM-fallback")
                if "raw" in result and len(result) == 1:
                    result = _fix_json_with_qwen3(raw2, source="PathoR1LLM-fallback")
            finally:
                self.max_new_tokens = original_tokens
        return result


_patho_r1_llm_instance: Optional["PathoR1LLM"] = None


def get_patho_r1_llm(max_new_tokens: int = 512, temperature: float = 0.1) -> "PathoR1LLM":
    """Return a shared PathoR1LLM instance."""
    global _patho_r1_llm_instance
    if _patho_r1_llm_instance is None:
        _patho_r1_llm_instance = PathoR1LLM(max_new_tokens=max_new_tokens, temperature=temperature)
    return _patho_r1_llm_instance


_hfllm_instance: Optional["HFLLM"] = None


def get_hfllm(config: Optional[HFLLMConfig] = None) -> "HFLLM":
    """Return a shared HFLLM instance, loading the model only once."""
    global _hfllm_instance
    if _hfllm_instance is None:
        _hfllm_instance = HFLLM(config)
    return _hfllm_instance


class HFLLM:
    """HuggingFace transformers wrapper — loads model locally (e.g. Qwen/Qwen3-4B)."""

    def __init__(self, config: Optional[HFLLMConfig] = None):
        self.config = config or HFLLMConfig()
        self._model = None
        self._tokenizer = None
        self._quantized = False
        self._load_model()

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #
    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading HF model: {self.config.model_id} ...")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self._quantized = True
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self._quantized = True

        # Without quantization: load to CPU and offload to GPU per call
        device = self.config.device if self._quantized else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype if quantization_config is None else None,
            device_map=device,
            quantization_config=quantization_config,
        )
        logger.info(f"HF model loaded to {'GPU (quantized)' if self._quantized else 'CPU (offload mode)'}: {self.config.model_id}")

    # ------------------------------------------------------------------ #
    #  GPU offload helpers                                                 #
    # ------------------------------------------------------------------ #
    def _to_gpu(self):
        import torch
        if not self._quantized and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._model = self._model.to("cuda")
            # logger.debug(f"HFLLM moved to GPU")

    def _to_cpu(self):
        import torch
        if not self._quantized and torch.cuda.is_available():
            self._model = self._model.to("cpu")
            torch.cuda.empty_cache()
            # logger.debug(f"HFLLM moved to CPU")

    # ------------------------------------------------------------------ #
    #  Core chat                                                           #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        import torch

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        if json_mode:
            # Inject JSON instruction into last user message
            for msg in reversed(payload_messages):
                if msg["role"] == "user":
                    msg = dict(msg)
                    msg["content"] += "\nReply with valid JSON only, no extra text."
                    # Replace in list
                    idx = payload_messages.index(msg) if msg in payload_messages else -1
                    # rebuild safely
                    break
            # Rebuild with modified last user message
            modified = []
            found = False
            for m in reversed(payload_messages):
                if not found and m["role"] == "user":
                    modified.insert(0, {**m, "content": m["content"] + "\nReply with valid JSON only, no extra text."})
                    found = True
                else:
                    modified.insert(0, m)
            payload_messages = modified

        with _gpu_lock:
            self._to_gpu()
            try:
                text = self._tokenizer.apply_chat_template(
                    payload_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.config.enable_thinking,
                )
                inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                content = self._tokenizer.decode(new_ids, skip_special_tokens=True)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
            finally:
                self._to_cpu()

    # ------------------------------------------------------------------ #
    #  JSON helpers                                                        #
    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> dict:
        """Like chat() but parses and returns a JSON dict. Retries once if output is not valid JSON."""
        raw = self.chat(messages, system=system, json_mode=True)
        result = _parse_json_robust(raw, source="HFLLM")
        if "raw" in result and len(result) == 1:
            # Fallback: retry with doubled token budget
            original_tokens = self.config.max_new_tokens
            self.config.max_new_tokens = original_tokens * 2
            try:
                raw2 = self.chat(messages, system=system, json_mode=True)
                result = _parse_json_robust(raw2, source="HFLLM-fallback")
                if "raw" in result and len(result) == 1:
                    result = _fix_json_with_qwen3(raw2, source="HFLLM-fallback")
            finally:
                self.config.max_new_tokens = original_tokens
        return result


_hf_vision_llm_instance: Optional["HFVisionLLM"] = None


def get_hf_vision_llm(config: Optional[HFVisionLLMConfig] = None) -> "HFVisionLLM":
    """Return a shared HFVisionLLM instance, loading the model only once."""
    global _hf_vision_llm_instance
    if _hf_vision_llm_instance is None:
        _hf_vision_llm_instance = HFVisionLLM(config)
    return _hf_vision_llm_instance


def unload_hf_vision_llm():
    """Unload HFVisionLLM from VRAM and free memory."""
    global _hf_vision_llm_instance
    if _hf_vision_llm_instance is not None:
        import torch
        instance = _hf_vision_llm_instance
        _hf_vision_llm_instance = None
        if instance._model is not None:
            instance._model.cpu()
            del instance._model
        if instance._processor is not None:
            del instance._processor
        del instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFVisionLLM unloaded from VRAM.")


class HFVisionLLM:
    """
    Vision-language LLM wrapper for Qwen2.5-VL.
    Supports optional image input alongside text messages.
    Interface mirrors HFLLM (chat / chat_json), with an extra `images` param.
    """

    def __init__(self, config: Optional[HFVisionLLMConfig] = None):
        from slideseek.config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG.hf_vision_llm
        self._model = None
        self._processor = None
        self._quantized = False
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        logger.info(f"Loading HF vision LLM: {self.config.model_id} ...")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        load_kwargs = {"torch_dtype": torch_dtype}
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.pop("torch_dtype", None)
            load_kwargs["device_map"] = self.config.device
            self._quantized = True
        elif self.config.load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs.pop("torch_dtype", None)
            load_kwargs["device_map"] = self.config.device
            self._quantized = True
        else:
            load_kwargs["device_map"] = "cpu"  # CPU offload, move to GPU on demand

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        logger.info(f"HF vision LLM loaded to {'GPU (quantized)' if self._quantized else 'CPU (offload mode)'}.")

    def _to_gpu(self):
        import torch
        if not self._quantized and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._model = self._model.to("cuda")
            logger.debug("HFVisionLLM moved to GPU")

    def _to_cpu(self):
        import torch
        if not self._quantized and torch.cuda.is_available():
            self._model = self._model.to("cpu")
            torch.cuda.empty_cache()
            logger.debug("HFVisionLLM moved to CPU")

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        images: Optional[list] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Args:
            messages:  Standard chat messages (role/content dicts).
            system:    Optional system prompt.
            images:    Optional list of PIL images to attach to the first user message.
            json_mode: Append JSON instruction to last user message.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        payload = []
        if system:
            payload.append({"role": "system", "content": system})

        for i, msg in enumerate(messages):
            if i == 0 and images and msg["role"] == "user":
                content = []
                for img in images:
                    content.append({"type": "image", "image": img})
                text = msg["content"]
                if json_mode and i == len(messages) - 1:
                    text += "\nReply with valid JSON only, no extra text."
                content.append({"type": "text", "text": text})
                payload.append({"role": "user", "content": content})
            else:
                text = msg["content"]
                if json_mode and i == len(messages) - 1 and msg["role"] == "user":
                    text += "\nReply with valid JSON only, no extra text."
                payload.append({"role": msg["role"], "content": text})

        with _gpu_lock:
            self._to_gpu()
            try:
                text_input = self._processor.apply_chat_template(
                    payload, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(payload)
                inputs = self._processor(
                    text=[text_input],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                )
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=self.config.temperature > 0,
                    )

                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                content = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
            finally:
                self._to_cpu()

    def chat_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        images: Optional[list] = None,
    ) -> dict:
        raw = self.chat(messages, system=system, images=images, json_mode=True)
        result = _parse_json_robust(raw, source="HFVisionLLM")
        if "raw" in result and len(result) == 1:
            # Fallback: retry with doubled token budget
            original_tokens = self.config.max_new_tokens
            self.config.max_new_tokens = original_tokens * 2
            try:
                raw2 = self.chat(messages, system=system, images=images, json_mode=True)
                result = _parse_json_robust(raw2, source="HFVisionLLM-fallback")
                if "raw" in result and len(result) == 1:
                    result = _fix_json_with_qwen3(raw2, source="HFVisionLLM-fallback")
            finally:
                self.config.max_new_tokens = original_tokens
        return result


# ------------------------------------------------------------------ #
#  OpenAI backend                                                      #
# ------------------------------------------------------------------ #

def get_openaillm(config: Optional["OpenAILLMConfig"] = None) -> "OpenAILLM":
    """Return a shared OpenAILLM instance."""
    global _openaillm_instance
    if _openaillm_instance is None:
        _openaillm_instance = OpenAILLM(config)
    return _openaillm_instance


class OpenAILLM:
    """
    OpenAI API wrapper (text-only).
    Exposes chat() / chat_json() compatible with HFLLM / OllamaLLM.

    API key is read from:
      1. config.api_key
      2. Environment variable OPENAI_API_KEY
    """

    def __init__(self, config: Optional[OpenAILLMConfig] = None):
        from slideseek.config import OpenAILLMConfig as _Cfg
        self.config = config or _Cfg()
        self._client = self._build_client()

    def _build_client(self):
        import os
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install it with: pip install openai"
            ) from exc

        try:
            from dotenv import load_dotenv
            _env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
            load_dotenv(_env_path)
        except ImportError as e:
            print("ImportError:", e)
            print("env path:", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
            pass

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set it via --openai-api-key or the OPENAI_API_KEY environment variable."
            )
        kwargs = {"api_key": api_key}
        if self.config.api_base_url:
            kwargs["base_url"] = self.config.api_base_url
        client = OpenAI(**kwargs)
        logger.info(f"OpenAI LLM initialised — model: {self.config.model}")
        return client

    # ------------------------------------------------------------------ #
    #  Core chat                                                           #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        kwargs: dict = {
            "model": self.config.model,
            "messages": payload,
            "max_completion_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Log every API call so it's visible in console output
        user_preview = next(
            (m["content"][:80] for m in reversed(payload) if m["role"] == "user"), ""
        )
        msg = (
            f"[OpenAI API] model={self.config.model}  "
            f"json_mode={json_mode}  "
            f"prompt_preview={user_preview!r}..."
        )
        logger.info(msg)
        print(msg)

        def _do_create(kw: dict) -> str:
            response = self._client.chat.completions.create(**kw)
            usage = response.usage
            result_msg = (
                f"[OpenAI API] done — "
                f"prompt_tokens={usage.prompt_tokens}  "
                f"completion_tokens={usage.completion_tokens}  "
                f"total_tokens={usage.total_tokens}"
            )
            logger.info(result_msg)
            print(result_msg)
            return response.choices[0].message.content or ""

        try:
            return _do_create(kwargs)
        except Exception as e:
            # Some models (o-series) reject custom temperature; retry without it
            if "temperature" in str(e):
                logger.warning(f"Model rejected temperature param, retrying without it: {e}")
                kw2 = {k: v for k, v in kwargs.items() if k != "temperature"}
                try:
                    return _do_create(kw2)
                except Exception as e2:
                    logger.error(f"OpenAI chat error (retry): {e2}")
                    raise
            logger.error(f"OpenAI chat error: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  JSON helpers                                                        #
    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> dict:
        raw = self.chat(messages, system=system, json_mode=True)
        result = _parse_json_robust(raw, source="OpenAILLM")
        if "raw" in result and len(result) == 1:
            # Fallback: retry with doubled token budget
            original_tokens = self.config.max_new_tokens
            self.config.max_new_tokens = original_tokens * 2
            try:
                raw2 = self.chat(messages, system=system, json_mode=True)
                result = _parse_json_robust(raw2, source="OpenAILLM-fallback")
                if "raw" in result and len(result) == 1:
                    result = _fix_json_with_qwen3(raw2, source="OpenAILLM-fallback")
            finally:
                self.config.max_new_tokens = original_tokens
        return result
