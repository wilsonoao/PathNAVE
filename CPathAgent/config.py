from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    # Which backend to use for BOTH text and vision (single-backend mode)
    backend: Literal["ollama", "hf", "openai"] = "ollama"

    # Optional per-role overrides (mixed-backend mode)
    # When set, text_backend is used for generate_text()
    # and vlm_backend is used for generate_with_images().
    # Either can differ from `backend`.
    text_backend: Optional[Literal["ollama", "hf", "openai"]] = None
    vlm_backend: Optional[Literal["ollama", "hf", "openai"]] = None

    # ── Text model (Qwen3 4B) – pure-text tasks ──────────────────────────────
    text_model_ollama: str = "qwen3:4b"
    text_model_hf: str = "Qwen/Qwen3-4B"
    # text_model_ollama: str = ""
    # text_model_hf: str = ""

    # ── VLM model (patho_r1) – pathology image description ───────────────────
    # For Ollama: the model tag as listed by `ollama list`
    vlm_model_ollama: str = "patho_r1"
    # For HuggingFace: the repo ID, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    # patho_r1 HF repo – update if the actual HF ID differs
    vlm_model_hf: str = "patho_r1"

    # ── Ollama settings ───────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"

    # ── HuggingFace settings ──────────────────────────────────────────────────
    hf_device: str = "cuda"
    hf_dtype: str = "float16"       # "float16" | "bfloat16" | "float32"
    hf_load_in_8bit: bool = False
    hf_load_in_4bit: bool = False
    hf_use_flash_attn: bool = False  # Flash Attention 2 (requires flash-attn package, fp16/bf16 only)
    hf_use_sdpa: bool = True        # SDPA attention kernel (no extra package needed)
    hf_enable_tf32: bool = True     # TF32 matmul on Ampere+ tensor cores
    hf_use_compile: bool = False    # torch.compile (warmup cost, faster after first call)

    # ── OpenAI / GPT settings ─────────────────────────────────────────────────
    # Text-only tasks (e.g. JSON extraction, prompt repair)
    text_model_openai: str = "gpt-4o-mini"
    # Vision-language tasks (must be a vision-capable model)
    vlm_model_openai: str = "gpt-4o"
    # API key – if None, falls back to the OPENAI_API_KEY environment variable
    openai_api_key: Optional[str] = None
    # Override base URL for Azure OpenAI or other OpenAI-compatible endpoints
    openai_api_base_url: Optional[str] = None

    # ── Generation ────────────────────────────────────────────────────────────
    max_new_tokens: int = 2048
    temperature: float = 0.1
    # Qwen3 supports thinking tokens – set True to enable chain-of-thought
    qwen3_thinking: bool = False


@dataclass
class WSIConfig:
    thumbnail_scale: int = 32       # Thumbnail = original / 32
    region_size_px: int = 16000     # Each region is 16 000 × 16 000 px at 40 ×
    overlap_ratio: float = 0.05     # 5 % boundary overlap between adjacent patches
    patch_size_px: int = 512        # Patch size in thumbnail-pixel space (512 × 512)
    model_input_size: int = 1008    # Resize cropped views to this before sending to VLM
    max_nav_steps: int = 8          # Max navigation steps per region
    coord_grid_intervals: float = 0.1   # Grid lines every 10 % of image width/height


@dataclass
class CPathAgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    wsi: WSIConfig = field(default_factory=WSIConfig)
    output_dir: str = "outputs"
    save_images: bool = True
    verbose: bool = True
