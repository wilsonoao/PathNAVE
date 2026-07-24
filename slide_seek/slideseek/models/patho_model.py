"""
Pathology vision model wrapper.
Primary model: patho_r1 (linjc16/Patho-R1-7B) from HuggingFace.
Used for ROI-level morphological captioning (replaces PathChat+ in SlideSeek).
"""
import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Lazy imports — only loaded when model is first used               #
# ------------------------------------------------------------------ #
_model = None
_processor = None
_tokenizer = None
_model_type = None   # "patho_r1" | "mock"
_is_quantized = False

import threading
_inference_lock = threading.Lock()
_load_lock = threading.Lock()


def to_gpu():
    """Move patho_r1 to CUDA for inference (no-op if quantized or no CUDA)."""
    global _model
    import torch
    if _model is None or _model_type != "patho_r1" or _is_quantized:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # flush any residual cached VRAM before loading
        _model = _model.to("cuda")
        logger.debug("patho_r1 moved to GPU")


def to_cpu():
    """Move patho_r1 back to CPU and free VRAM."""
    global _model
    import torch
    if _model is None or _model_type != "patho_r1" or _is_quantized:
        return
    _model = _model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("patho_r1 moved to CPU")


def _load_patho_r1(config):
    """Load patho_r1 (Qwen2-VL based pathology reasoning model)."""
    global _model, _processor, _tokenizer, _model_type, _is_quantized
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading patho_r1 from {config.model_id} ...")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

        load_kwargs = {"torch_dtype": torch_dtype}

        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = config.device  # bitsandbytes needs GPU
            _is_quantized = True
        elif config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = config.device
            _is_quantized = True
        else:
            load_kwargs["device_map"] = "cpu"  # CPU offload, move to GPU on demand
            _is_quantized = False

        # Determine attention backend: flash_attention_2 → sdpa → default
        attn_candidates = []
        if getattr(config, "use_flash_attn", True):
            attn_candidates.append("flash_attention_2")
        attn_candidates += ["sdpa", None]

        _model = None
        for attn_impl in attn_candidates:
            kwargs = dict(load_kwargs)
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            try:
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    config.model_id, **kwargs
                )
                label = attn_impl if attn_impl else "default"
                logger.info(f"patho_r1: using {label} attention")
                break
            except Exception as attn_err:
                err_msg = str(attn_err).lower()
                if attn_impl and (
                    "attn_implementation" in err_msg
                    or attn_impl.replace("_", "") in err_msg.replace("_", "")
                    or "flash" in err_msg
                    or "sdpa" in err_msg
                    or "not installed" in err_msg
                    or "importerror" in err_msg
                ):
                    logger.warning(
                        f"patho_r1: {attn_impl} not available, trying next fallback. "
                        f"Reason: {attn_err}"
                    )
                    continue
                raise
        if _model is None:
            raise RuntimeError("Failed to load patho_r1 with any attention implementation.")

        _processor = AutoProcessor.from_pretrained(config.model_id)
        _model_type = "patho_r1"
        logger.info(f"patho_r1 loaded to {'GPU (quantized)' if _is_quantized else 'CPU (offload mode)'}.")
    except Exception as e:
        logger.error(f"Failed to load patho_r1: {e}", exc_info=True)
        raise


def _load_mock():
    """Fallback mock model for testing without GPU."""
    global _model, _processor, _model_type
    _model = "mock"
    _processor = "mock"
    _model_type = "mock"
    logger.warning("Using MOCK pathology model — output will be placeholder text.")


def ensure_model_loaded(config=None):
    global _model_type
    if _model is not None:
        return
    with _load_lock:
        # Re-check after acquiring lock — another thread may have loaded it already
        if _model is not None:
            return
        from slideseek.config import DEFAULT_CONFIG
        cfg = config or DEFAULT_CONFIG.patho_model
        try:
            _load_patho_r1(cfg)
        except Exception as e:
            logger.error(
                f"patho_r1 load FAILED — falling back to MOCK model. "
                f"ROI captions will be placeholder text. Error: {e}"
            )
            _load_mock()


# ------------------------------------------------------------------ #
#  Image helpers                                                       #
# ------------------------------------------------------------------ #
def _to_pil(image: Union[np.ndarray, "Image.Image", str, Path]):
    """Convert various image types to PIL Image."""
    from PIL import Image
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return image.convert("RGB")


# ------------------------------------------------------------------ #
#  Inference                                                           #
# ------------------------------------------------------------------ #
CAPTION_PROMPT = (
    "You are an expert surgical pathologist. "
    "Carefully examine this histopathology image and describe the key morphological features observed. "
    "Focus on: tissue architecture, cellular characteristics, nuclear features, stroma, "
    "and any abnormal findings. Be concise and precise. Do not suggest a diagnosis. "
    "Limit your response to 30 words or fewer."
)

DIAGNOSIS_PROMPT_TEMPLATE = (
    "You are an expert surgical pathologist. "
    "Tissue site: {tissue_site}. Patient sex: {patient_sex}. "
    "Review these regions of interest from a histology slide and provide "
    "the most likely primary diagnosis. Also suggest 2 other possibilities to consider.\n"
    "Your response must have these sections:\n"
    "Primary Diagnosis:\n"
    "Other Possibilities:\n"
    "1.\n2."
)


def caption_roi(
    image: Union[np.ndarray, "Image.Image", str, Path],
    prompt: Optional[str] = None,
    config=None,
) -> str:
    """
    Generate a morphological description for a single pathology ROI image.

    Args:
        image:  ROI image (numpy array, PIL Image, or file path).
        prompt: Custom prompt (default: standard morphology caption prompt).
        config: PathoModelConfig (uses default if None).

    Returns:
        Morphological description string.
    """
    ensure_model_loaded(config)
    prompt = prompt or CAPTION_PROMPT

    if _model_type == "mock":
        return (
            "[MOCK] The image shows tissue with moderate cellularity. "
            "Cells exhibit round nuclei with moderate nuclear-to-cytoplasmic ratios. "
            "No definitive architectural abnormalities identified in this mock response."
        )

    if _model_type == "patho_r1":
        import torch, time
        from slideseek.models.llm import _gpu_lock
        oom_retries = 10
        for attempt in range(1, oom_retries + 1):
            with _inference_lock:
                with _gpu_lock:
                    try:
                        to_gpu()
                        return _infer_patho_r1(image, prompt, config)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        logger.warning(
                            f"caption_roi: CUDA OOM on attempt {attempt}/{oom_retries}, "
                            f"retrying in 30s..."
                        )
                    finally:
                        to_cpu()
            if attempt < oom_retries:
                time.sleep(30)
        raise RuntimeError(f"caption_roi: CUDA OOM after {oom_retries} attempts, giving up.")

    raise ValueError(f"Unknown model type: {_model_type}")


def caption_multiple_rois(
    images: list,
    tissue_site: str = "unknown",
    patient_sex: str = "unknown",
    config=None,
) -> str:
    """
    Generate a differential diagnosis from multiple ROI images.
    Used at the end of SlideSeek's pipeline (final diagnosis step).
    """
    ensure_model_loaded(config)
    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
        tissue_site=tissue_site, patient_sex=patient_sex
    )

    if _model_type == "mock":
        return (
            "Primary Diagnosis:\nAdenocarcinoma [MOCK]\n\n"
            "Other Possibilities:\n1. Squamous cell carcinoma [MOCK]\n2. Neuroendocrine tumor [MOCK]"
        )

    if _model_type == "patho_r1":
        import torch, time
        from slideseek.models.llm import _gpu_lock
        oom_retries = 10
        for attempt in range(1, oom_retries + 1):
            with _inference_lock:
                with _gpu_lock:
                    try:
                        to_gpu()
                        return _infer_patho_r1_multi(images, prompt, config)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        logger.warning(
                            f"caption_multiple_rois: CUDA OOM on attempt {attempt}/{oom_retries}, "
                            f"retrying in 30s..."
                        )
                    finally:
                        to_cpu()
            if attempt < oom_retries:
                time.sleep(30)
        raise RuntimeError(f"caption_multiple_rois: CUDA OOM after {oom_retries} attempts, giving up.")

    raise ValueError(f"Unknown model type: {_model_type}")


# ------------------------------------------------------------------ #
#  patho_r1 inference internals                                        #
# ------------------------------------------------------------------ #
def _extract_answer(text: str) -> str:
    """Extract content inside <answer>...</answer>; fall back to full text if absent."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _truncate_to_words(text: str, max_words: int) -> str:
    """Hard-truncate text to at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _infer_patho_r1(image, prompt: str, config) -> str:
    import torch

    pil_img = _to_pil(image)
    logger.info(f"patho_r1 input | image size: {pil_img.size} (W x H) | prompt: {prompt[:120]}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # qwen2-vl processor
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    from slideseek.config import DEFAULT_CONFIG
    cfg = config or DEFAULT_CONFIG.patho_model

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Decode only newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    result = _extract_answer(raw)
    result = _truncate_to_words(result, cfg.max_output_words)
    logger.info(f"patho_r1 output: {result}")
    return result


def _infer_patho_r1_multi(images: list, prompt: str, config) -> str:
    """Multi-image inference for final differential diagnosis."""
    import torch

    pil_images = [_to_pil(img) for img in images]
    sizes = [img.size for img in pil_images]
    logger.info(f"patho_r1 multi input | {len(pil_images)} images | sizes: {sizes} | prompt: {prompt[:120]}")

    content = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    text_input = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    from slideseek.config import DEFAULT_CONFIG
    cfg = config or DEFAULT_CONFIG.patho_model

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    result = _extract_answer(raw)
    result = _truncate_to_words(result, cfg.max_output_words)
    logger.info(f"patho_r1 multi output: {result}")
    return result
