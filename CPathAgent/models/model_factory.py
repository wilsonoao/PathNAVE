from config import ModelConfig
from .base_model import BaseModel


def _build_single(backend: str, cfg: ModelConfig) -> BaseModel:
    """Instantiate one backend by name using settings from cfg."""
    if backend == "ollama":
        print("ollama")
        from .ollama_backend import OllamaBackend
        return OllamaBackend(
            text_model=cfg.text_model_ollama,
            vlm_model=cfg.vlm_model_ollama,
            host=cfg.ollama_host,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            qwen3_thinking=cfg.qwen3_thinking,
        )

    elif backend == "hf":
        print("hf")
        from .hf_backend import HFBackend
        return HFBackend(
            text_model_id=cfg.text_model_hf,
            vlm_model_id=cfg.vlm_model_hf,
            device=cfg.hf_device,
            dtype=cfg.hf_dtype,
            load_in_8bit=cfg.hf_load_in_8bit,
            load_in_4bit=cfg.hf_load_in_4bit,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            qwen3_thinking=cfg.qwen3_thinking,
            use_flash_attn=cfg.hf_use_flash_attn,
            use_sdpa=cfg.hf_use_sdpa,
            enable_tf32=cfg.hf_enable_tf32,
            use_compile=cfg.hf_use_compile,
        )

    elif backend == "openai":
        print("openai")
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(
            text_model=cfg.text_model_openai,
            vlm_model=cfg.vlm_model_openai,
            api_key=cfg.openai_api_key,
            api_base_url=cfg.openai_api_base_url,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'ollama', 'hf', or 'openai'.")


def build_model(cfg: ModelConfig) -> BaseModel:
    """
    Factory: return the correct backend given cfg.

    Mixed-backend mode
    ──────────────────
    If cfg.text_backend and cfg.vlm_backend are both set (and differ from each
    other or from cfg.backend), a MixedBackend is returned that routes
    text-only calls to text_backend and vision calls to vlm_backend.

    Single-backend mode
    ───────────────────
    Otherwise cfg.backend is used for both roles.
    """
    text_be = cfg.text_backend or cfg.backend
    vlm_be  = cfg.vlm_backend  or cfg.backend

    if text_be == vlm_be:
        return _build_single(text_be, cfg)

    # Mixed: build each role separately and wrap them
    from .mixed_backend import MixedBackend
    text_model = _build_single(text_be, cfg)
    vlm_model  = _build_single(vlm_be,  cfg)
    return MixedBackend(text_backend=text_model, vlm_backend=vlm_model)
