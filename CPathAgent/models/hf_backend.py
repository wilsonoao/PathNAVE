"""
HuggingFace backend.

Text model : Qwen3-4B  (AutoModelForCausalLM)
VLM model  : patho_r1  (supports Qwen2.5-VL-style and generic LLaVA-style processors)

The VLM is loaded lazily on the first multimodal call to avoid unnecessary
GPU memory allocation if only text inference is needed.

Inference acceleration flags
─────────────────────────────
use_flash_attn: use Flash Attention 2 (attn_implementation="flash_attention_2").
                Requires the `flash-attn` package and fp16/bf16 dtype.
                Fastest option; takes priority over use_sdpa when enabled.
use_sdpa      : use PyTorch SDPA kernel (attn_implementation="sdpa") – no extra
                package needed, significant speedup on Ampere/Ada GPUs.
enable_tf32   : enable TF32 matmul on Ampere+ tensor cores (free throughput boost).
use_compile   : apply torch.compile(dynamic=True) to the VLM after loading.
                First call pays a compilation cost; subsequent calls are faster.
"""
from __future__ import annotations

import re
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseModel


class HFBackend(BaseModel):

    def __init__(
        self,
        text_model_id: str,
        vlm_model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        qwen3_thinking: bool = False,
        use_flash_attn: bool = False,
        use_sdpa: bool = True,
        enable_tf32: bool = True,
        use_compile: bool = False,
    ):
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.qwen3_thinking = qwen3_thinking
        self.use_flash_attn = use_flash_attn
        self.use_sdpa = use_sdpa
        self.use_compile = use_compile

        # TF32 lets tensor cores operate on float32 with ~3× throughput at
        # negligible precision cost; safe to enable globally for inference.
        super().__init__()

        if enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ── Text model ────────────────────────────────────────────────────────
        # print(f"[HF] Loading text model: {text_model_id}")
        # quant_kwargs = {}
        # if load_in_4bit:
        #     from transformers import BitsAndBytesConfig
        #     quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        # elif load_in_8bit:
        #     from transformers import BitsAndBytesConfig
        #     quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        # self.text_model = AutoModelForCausalLM.from_pretrained(
        #     text_model_id,
        #     torch_dtype=self.torch_dtype,
        #     device_map=device,
        #     **quant_kwargs,
        # )
        # self.text_model.eval()

        # ── VLM (lazy) ────────────────────────────────────────────────────────
        self.vlm_model_id = vlm_model_id
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_type: Optional[str] = None   # "qwen25vl" | "llava" | "generic"

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_vlm_type(model_id: str) -> str:
        """
        Detect VLM architecture by reading the actual model config first,
        then fall back to model-ID string matching.

        Returns: "qwen2vl" | "qwen25vl" | "llava" | "generic"
        """
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            mt = getattr(cfg, "model_type", "").lower()
            if mt == "qwen2_5_vl":
                return "qwen25vl"
            if mt == "qwen2_vl":
                return "qwen2vl"
            if "llava" in mt:
                return "llava"
            if mt:  # known model_type but not VLM-specific above
                # Still might be a VLM – fall through to string matching
                pass
        except Exception:
            pass

        # Fallback: string matching (check more-specific strings first)
        mid = model_id.lower()
        if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid:
            return "qwen25vl"
        if "qwen2-vl" in mid or ("qwen2" in mid and "vl" in mid):
            return "qwen2vl"
        if "llava" in mid:
            return "llava"
        return "generic"

    def _load_vlm(self):
        if self._vlm_model is not None:
            return

        print(f"[HF] Loading VLM: {self.vlm_model_id}")

        quant_kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Attention implementation priority: flash_attention_2 > sdpa > eager.
        # Flash Attention 2 requires fp16/bf16 and no quantization.
        # SDPA requires no quantization.
        is_quantized = self.load_in_4bit or self.load_in_8bit
        _fa2_dtypes = (torch.float16, torch.bfloat16)
        if self.use_flash_attn and not is_quantized and self.torch_dtype in _fa2_dtypes:
            attn_impl = "flash_attention_2"
        elif self.use_flash_attn and not is_quantized:
            print("[HF] flash_attention_2 requires float16 or bfloat16; falling back to sdpa")
            attn_impl = "sdpa" if self.use_sdpa else "eager"
        elif self.use_sdpa and not is_quantized:
            attn_impl = "sdpa"
        else:
            attn_impl = "eager"

        self._vlm_type = self._detect_vlm_type(self.vlm_model_id)
        print(f"[HF] VLM type detected: {self._vlm_type}")
        print(f"[HF] attn_implementation: {attn_impl}")

        from transformers import AutoProcessor
        self._vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_model_id, trust_remote_code=True
        )

        # Qwen2/2.5-VL: set pixel bounds to prevent the processor from computing
        # a visual-token grid that exceeds the model's position-embedding table,
        # which causes a CUDA device-side assert during .to(device).
        # 256 * 28² = 200 704 px minimum (~448×448)
        # 1280 * 28² = 1 003 520 px maximum (~1002×1002)
        if self._vlm_type in ("qwen25vl", "qwen2vl"):
            ip = getattr(self._vlm_processor, "image_processor", None)
            if ip is not None:
                ip.min_pixels = 256 * 28 * 28
                ip.max_pixels = 1280 * 28 * 28
                print(f"[HF] Qwen-VL pixel bounds set: "
                      f"min={ip.min_pixels}, max={ip.max_pixels}")

        if self._vlm_type == "qwen25vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            self._vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                attn_implementation=attn_impl,
                **quant_kwargs,
            ).eval()

        elif self._vlm_type == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration
            self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.vlm_model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                attn_implementation=attn_impl,
                **quant_kwargs,
            ).eval()

        elif self._vlm_type == "llava":
            from transformers import LlavaForConditionalGeneration
            self._vlm_model = LlavaForConditionalGeneration.from_pretrained(
                self.vlm_model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                attn_implementation=attn_impl,
                **quant_kwargs,
            ).eval()

        else:
            # Generic: use AutoModelForVision2Seq which covers most VLMs,
            # fall back to AutoModelForCausalLM with trust_remote_code
            try:
                from transformers import AutoModelForVision2Seq
                self._vlm_model = AutoModelForVision2Seq.from_pretrained(
                    self.vlm_model_id,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                    **quant_kwargs,
                ).eval()
            except (ValueError, ImportError):
                self._vlm_model = AutoModelForCausalLM.from_pretrained(
                    self.vlm_model_id,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    **quant_kwargs,
                ).eval()

        # torch.compile: traces the model graph for kernel fusion and reduced
        # Python overhead.  dynamic=True handles varying image/sequence lengths
        # without recompilation.  First call pays the compilation cost (~60 s);
        # every subsequent call is faster.
        if self.use_compile:
            print("[HF] Compiling VLM with torch.compile(dynamic=True) …")
            self._vlm_model = torch.compile(
                self._vlm_model, dynamic=True, fullgraph=False
            )

    def _strip_thinking(self, text: str) -> str:
        if not self.qwen3_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def _generate_text(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        print(f"[LLM] text → hf/{self.text_model_id}  (max_tokens={max_tokens})")
        messages = [{"role": "user", "content": prompt}]

        # Build the formatted text string (never tokenize inside apply_chat_template
        # because newer transformers return BatchEncoding which lacks .shape)
        tmpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if hasattr(self.text_tokenizer, "apply_chat_template"):
            try:
                text = self.text_tokenizer.apply_chat_template(
                    messages, enable_thinking=self.qwen3_thinking, **tmpl_kwargs
                )
            except TypeError:
                # Older tokenizer – enable_thinking not supported
                text = self.text_tokenizer.apply_chat_template(messages, **tmpl_kwargs)
        else:
            text = prompt

        encoded = self.text_tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encoded.input_ids          # always a plain tensor

        with torch.no_grad():
            output_ids = self.text_model.generate(
                input_ids=input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        result = self.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._strip_thinking(result)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], **kwargs) -> str:
        self._load_vlm()

        if self._vlm_type in ("qwen25vl", "qwen2vl"):
            return self._generate_qwen25vl(prompt, images)
        elif self._vlm_type == "llava":
            return self._generate_llava(prompt, images)
        else:
            return self._generate_generic(prompt, images)

    # ── Model-specific generation ─────────────────────────────────────────────

    def _generate_qwen25vl(self, prompt: str, images: List[Image.Image]) -> str:
        from qwen_vl_utils import process_vision_info  # installed with Qwen2-VL

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        try:
            inputs = self._vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except RuntimeError as e:
            if "device-side assert" in str(e) or "CUDA" in str(e):
                # CUDA assertion usually means the image caused an out-of-range
                # visual token index.  Reset CUDA state and return empty string
                # so the caller can continue with the next patch.
                print(f"[HF] CUDA error during input processing – skipping image: {e}")
                torch.cuda.empty_cache()
                return ""
            raise

        try:
            with torch.inference_mode():
                output_ids = self._vlm_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )
        except RuntimeError as e:
            if "device-side assert" in str(e) or "CUDA" in str(e):
                print(f"[HF] CUDA error during generation – skipping image: {e}")
                torch.cuda.empty_cache()
                return ""
            raise

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        result = self._vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return self._strip_thinking(result)

    def _generate_llava(self, prompt: str, images: List[Image.Image]) -> str:
        # Build prompt with <image> tokens
        image_tokens = "\n".join(["<image>"] * len(images))
        full_prompt = f"{image_tokens}\n{prompt}"

        inputs = self._vlm_processor(
            text=full_prompt, images=images, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self._vlm_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, use_cache=True
            )
        result = self._vlm_processor.decode(output_ids[0], skip_special_tokens=True)
        # Strip the echoed prompt
        if result.startswith(full_prompt):
            result = result[len(full_prompt):]
        return self._strip_thinking(result.strip())

    def _generate_generic(self, prompt: str, images: List[Image.Image]) -> str:
        inputs = self._vlm_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                use_cache=True,
            )

        result = self._vlm_processor.decode(output_ids[0], skip_special_tokens=True)
        return self._strip_thinking(result)
