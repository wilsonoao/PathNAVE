#!/usr/bin/env python3
"""
CPathAgent – Command Line Interface

Usage examples
──────────────

# Full WSI (Ollama, default models)
python main.py wsi --input slide.svs --tissue breast

# Full WSI (HuggingFace backend)
python main.py wsi --input slide.svs --tissue lung --backend hf \
    --vlm-hf Qwen/Qwen2.5-VL-7B-Instruct

# Single region image
python main.py region --input region.png --tissue colon \
    --question "What is the most likely diagnosis?"

# List available Ollama models
python main.py list-models

# Check model availability
python main.py check
"""
from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv
load_dotenv(override=False)

from config import CPathAgentConfig, ModelConfig, WSIConfig


# ── Argument Parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpathagent",
        description="CPathAgent – Agent-based Foundation Model for Pathology Image Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── shared model args ──────────────────────────────────────────────────────
    def add_model_args(p: argparse.ArgumentParser):
        g = p.add_argument_group("Model backend")
        g.add_argument(
            "--backend", choices=["ollama", "hf", "openai"], default="ollama",
            help="Inference backend for both text and vision (default: ollama)",
        )
        g.add_argument(
            "--text-backend", choices=["ollama", "hf", "openai"], default=None,
            help="Override backend for text-only tasks (enables mixed mode)",
        )
        g.add_argument(
            "--vlm-backend", choices=["ollama", "hf", "openai"], default=None,
            help="Override backend for vision tasks (enables mixed mode)",
        )
        # Ollama
        g.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama server URL")
        g.add_argument("--text-ollama", default="qwen3:4b",
                       help="Ollama tag for text model (default: qwen3:4b)")
        g.add_argument("--vlm-ollama", default="patho_r1",
                       help="Ollama tag for VLM (default: patho_r1)")
        # HuggingFace
        g.add_argument("--text-hf", default="Qwen/Qwen3-4B",
                       help="HF repo ID for text model (default: Qwen/Qwen3-4B)")
        g.add_argument("--vlm-hf", default="patho_r1",
                       help="HF repo ID for VLM (default: patho_r1)")
        g.add_argument("--hf-device", default="cuda",
                       help="torch device for HF models (default: cuda)")
        g.add_argument("--hf-dtype", default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="dtype for HF models (default: float16)")
        g.add_argument("--hf-flash-attn", action="store_true",
                       help="Use Flash Attention 2 for HF models (requires flash-attn, fp16/bf16)")
        g.add_argument("--hf-8bit", action="store_true",
                       help="Load HF models in 8-bit (bitsandbytes)")
        g.add_argument("--hf-4bit", action="store_true",
                       help="Load HF models in 4-bit (bitsandbytes)")
        # OpenAI / GPT
        g.add_argument("--text-openai", default="gpt-4o-mini",
                       help="OpenAI model for text-only tasks (default: gpt-4o-mini)")
        g.add_argument("--vlm-openai", default="gpt-4o",
                       help="OpenAI model for vision tasks (default: gpt-4o)")
        g.add_argument("--openai-api-key", default=None,
                       help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
        g.add_argument("--openai-base-url", default=None,
                       help="Override OpenAI base URL (e.g. for Azure or compatible APIs)")
        # Generation
        g.add_argument("--max-tokens", type=int, default=2048,
                       help="Max new tokens per generation")
        g.add_argument("--temperature", type=float, default=0.1)
        g.add_argument("--thinking", action="store_true",
                       help="Enable Qwen3 chain-of-thought thinking tokens")

    # ── wsi command ────────────────────────────────────────────────────────────
    p_wsi = sub.add_parser("wsi", help="Run full pipeline on a WSI file")
    add_model_args(p_wsi)
    p_wsi.add_argument("--input", required=True, help="Path to WSI file (.svs/.tiff/…)")
    p_wsi.add_argument("--tissue", default="unknown", help="Tissue type (e.g. breast, lung)")
    p_wsi.add_argument("--question", default=None, help="VQA question (optional)")
    p_wsi.add_argument("--output-dir", default="outputs")
    p_wsi.add_argument("--prefix", default="result", help="Output file prefix")
    p_wsi.add_argument("--no-save-images", action="store_true")
    p_wsi.add_argument("--patch-size", type=int, default=512,
                       help="Patch size in thumbnail pixels (default: 512)")
    p_wsi.add_argument("--max-nav-steps", type=int, default=8)
    p_wsi.add_argument("--quiet", action="store_true")

    # ── region command ─────────────────────────────────────────────────────────
    p_reg = sub.add_parser("region", help="Run stages 2+3 on a pre-cropped region image")
    add_model_args(p_reg)
    p_reg.add_argument("--input", required=True, help="Path to region image")
    p_reg.add_argument("--tissue", default="unknown")
    p_reg.add_argument("--question", default=None, help="VQA question (optional)")
    p_reg.add_argument("--output-dir", default="outputs")
    p_reg.add_argument("--prefix", default="region_result")
    p_reg.add_argument("--no-save-images", action="store_true")
    p_reg.add_argument("--max-nav-steps", type=int, default=8)
    p_reg.add_argument("--quiet", action="store_true")

    # ── list-models command ────────────────────────────────────────────────────
    p_list = sub.add_parser("list-models", help="List models available in Ollama")
    p_list.add_argument("--ollama-host", default="http://localhost:11434")

    # ── check command ──────────────────────────────────────────────────────────
    p_check = sub.add_parser("check", help="Check model availability")
    add_model_args(p_check)

    return parser


# ── Config builders ───────────────────────────────────────────────────────────

def args_to_model_config(args) -> ModelConfig:
    return ModelConfig(
        backend=args.backend,
        text_backend=args.text_backend,
        vlm_backend=args.vlm_backend,
        text_model_ollama=args.text_ollama,
        vlm_model_ollama=args.vlm_ollama,
        text_model_hf=args.text_hf,
        vlm_model_hf=args.vlm_hf,
        ollama_host=args.ollama_host,
        hf_device=args.hf_device,
        hf_dtype=args.hf_dtype,
        hf_use_flash_attn=args.hf_flash_attn,
        hf_load_in_8bit=args.hf_8bit,
        hf_load_in_4bit=args.hf_4bit,
        text_model_openai=args.text_openai,
        vlm_model_openai=args.vlm_openai,
        openai_api_key=args.openai_api_key,
        openai_api_base_url=args.openai_base_url,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        qwen3_thinking=args.thinking,
    )


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_wsi(args):
    from pipeline import CPathAgentPipeline

    model_cfg = args_to_model_config(args)
    wsi_cfg = WSIConfig(
        patch_size_px=args.patch_size,
        max_nav_steps=args.max_nav_steps,
    )
    cfg = CPathAgentConfig(
        model=model_cfg,
        wsi=wsi_cfg,
        output_dir=args.output_dir,
        save_images=not args.no_save_images,
        verbose=not args.quiet,
    )

    pipeline = CPathAgentPipeline(cfg)
    result = pipeline.run_wsi(
        wsi_path=args.input,
        tissue_type=args.tissue,
        question=args.question,
        save_prefix=args.prefix,
    )

    print("\n" + "=" * 60)
    print("FINAL PATHOLOGICAL REPORT")
    print("=" * 60)
    print(result["final_report"])


def cmd_region(args):
    from pipeline import CPathAgentPipeline

    model_cfg = args_to_model_config(args)
    wsi_cfg = WSIConfig(max_nav_steps=args.max_nav_steps)
    cfg = CPathAgentConfig(
        model=model_cfg,
        wsi=wsi_cfg,
        output_dir=args.output_dir,
        save_images=not args.no_save_images,
        verbose=not args.quiet,
    )

    pipeline = CPathAgentPipeline(cfg)
    result = pipeline.run_region(
        region_path=args.input,
        tissue_type=args.tissue,
        question=args.question,
        save_prefix=args.prefix,
    )

    print("\n" + "=" * 60)
    if args.question:
        print("ANSWER:", result.get("answer", "N/A"))
    print("PATHOLOGICAL REPORT")
    print("=" * 60)
    print(result.get("report", result.get("reasoning", "")[:800]))


def cmd_list_models(args):
    import requests
    host = args.ollama_host.rstrip("/")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Models available at {host}:")
        for m in sorted(models):
            print(f"  • {m}")
    except Exception as e:
        print(f"[Error] Cannot reach Ollama at {host}: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_check(args):
    if args.backend == "ollama":
        from models.ollama_backend import OllamaBackend
        backend = OllamaBackend(
            text_model=args.text_ollama,
            vlm_model=args.vlm_ollama,
            host=args.ollama_host,
        )
        info = backend.check_availability()
        print(f"Ollama host   : {args.ollama_host}")
        print(f"Text model    : {info['text_model']}  {'✓' if info['text_model_ok'] else '✗ NOT FOUND'}")
        print(f"VLM model     : {info['vlm_model']}   {'✓' if info['vlm_model_ok'] else '✗ NOT FOUND'}")
        print(f"All models    : {', '.join(info['available_models'])}")
    elif args.backend == "openai":
        from models.openai_backend import OpenAIBackend
        try:
            backend = OpenAIBackend(
                text_model=args.text_openai,
                vlm_model=args.vlm_openai,
                api_key=args.openai_api_key,
                api_base_url=args.openai_base_url,
            )
            info = backend.check_availability()
            print(f"OpenAI backend")
            print(f"  Text model : {info['text_model']}  {'✓' if info['text_model_ok'] else '✗ NOT FOUND'}")
            print(f"  VLM model  : {info['vlm_model']}  {'✓' if info['vlm_model_ok'] else '✗ NOT FOUND'}")
            if "error" in info:
                print(f"  Error      : {info['error']}")
        except ValueError as e:
            print(f"[Error] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"HF backend selected.")
        print(f"  Text model: {args.text_hf}")
        print(f"  VLM model : {args.vlm_hf}")
        print(f"  Device    : {args.hf_device}  |  dtype: {args.hf_dtype}")
        print("  (Models will be downloaded from HuggingFace Hub on first run)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "wsi": cmd_wsi,
        "region": cmd_region,
        "list-models": cmd_list_models,
        "check": cmd_check,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
