#!/usr/bin/env python3
"""
SlideSeek CLI — run the full pipeline from the command line.

Examples
--------
# Mock run (no real WSI, tests the whole pipeline end-to-end)
python run_slideseek.py --mock --tissue-site lung --sex male

# Real WSI
python run_slideseek.py --wsi path/to/slide.svs --tissue-site thyroid --sex female

# Real WSI with 4-bit quantization (low VRAM)
python run_slideseek.py --wsi path/to/slide.svs --tissue-site breast --sex female --load-4bit

# Use a different Ollama model
python run_slideseek.py --mock --tissue-site brain --ollama-model qwen3:8b
"""
import argparse
import logging
import sys
import os

# Load .env before anything else so env vars are available project-wide
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slideseek import SlideSeekPipeline, SlideSeekConfig
from slideseek.config import (
    OllamaConfig, OpenAILLMConfig, PathoModelConfig, SlideConfig, AgentConfig
)


def build_config(args) -> SlideSeekConfig:
    return SlideSeekConfig(
        use_openai_llm=args.use_openai,
        use_hf_llm=(not args.use_openai),
        openai_llm=OpenAILLMConfig(
            model=args.openai_model,
            api_key=args.openai_api_key,
            api_base_url=args.openai_base_url,
        ),
        ollama=OllamaConfig(
            base_url=args.ollama_url,
            model=args.ollama_model,
            enable_thinking=args.enable_thinking,
        ),
        patho_model=PathoModelConfig(
            model_id=args.patho_model,
            device=args.device,
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
        ),
        slide=SlideConfig(
            roi_size=args.roi_size,
        ),
        agent=AgentConfig(
            max_supervisor_iterations=args.max_iter,
            max_parallel_explorers=args.max_explorers,
        ),
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SlideSeek — Multi-agent WSI pathology diagnosis"
    )

    # Slide input
    parser.add_argument("--wsi", type=str, default=None,
                        help="Path to WSI file (.svs, .ndpi, .tif)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock slide viewer (no real WSI needed)")

    # Clinical context
    parser.add_argument("--tissue-site", type=str, default="unknown",
                        help="Tissue site (e.g. lung, breast, thyroid)")
    parser.add_argument("--sex", type=str, default="unknown",
                        choices=["male", "female", "unknown"],
                        help="Patient sex")
    parser.add_argument("--clinical-context", type=str,
                        default="No additional clinical context provided.",
                        help="Free-text clinical notes")

    # Model config
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default="qwen3:4b",
                        help="Ollama model name")
    parser.add_argument("--use-openai", action="store_true",
                        help="Use OpenAI API for the text LLM (overrides HF/Ollama)")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini",
                        help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--openai-base-url", type=str, default=None,
                        help="Override OpenAI base URL (e.g. for Azure OpenAI)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 chain-of-thought /think mode")
    parser.add_argument("--patho-model", type=str, default="linjc16/Patho-R1-7B",
                        help="HuggingFace model ID for pathology captioning")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device (auto, cuda, cpu)")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load patho model in 4-bit (low VRAM)")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load patho model in 8-bit")

    # Pipeline config
    parser.add_argument("--roi-size", type=int, default=896,
                        help="ROI patch size in pixels")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Max supervisor iterations")
    parser.add_argument("--max-explorers", type=int, default=3,
                        help="Max parallel explorer agents")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for reports")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    if not args.mock and args.wsi is None:
        parser.error("Either --wsi or --mock is required.")

    config = build_config(args)
    pipeline = SlideSeekPipeline(config)

    print(f"\n{'='*60}")
    print("SlideSeek Pipeline")
    print(f"{'='*60}")
    print(f"  Tissue site:    {args.tissue_site}")
    print(f"  Patient sex:    {args.sex}")
    print(f"  Patho model:    {args.patho_model}")
    if args.use_openai:
        print(f"  LLM:            {args.openai_model} (OpenAI API)")
    else:
        print(f"  LLM:            {args.ollama_model} @ {args.ollama_url}")
    print(f"  WSI:            {'[MOCK]' if args.mock else args.wsi}")
    print(f"{'='*60}\n")

    result = pipeline.run(
        wsi_path=args.wsi,
        tissue_site=args.tissue_site,
        patient_sex=args.sex,
        clinical_context=args.clinical_context,
        mock=args.mock,
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Primary Diagnosis:     {result.primary_diagnosis}")
    print(f"Differential:          {result.differential_diagnoses}")
    print(f"Confidence:            {result.confidence}")
    print(f"ROIs examined:         {result.num_rois_examined}")
    print(f"Supervisor iterations: {result.num_iterations}")
    print("\n--- REPORT ---")
    print(result.report)
    print("="*60)
    print(f"\nFull report saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
