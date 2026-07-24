"""
CPathAgent – CAMELYON16 benchmark evaluation script.

Reads a QA dataset JSON (e.g. roi_qa_dataset.json), finds the corresponding
slide file for each entry, runs the full CPathAgent pipeline, and saves per-case
results with full intermediate-step traces.

Output layout
─────────────
  {output_dir}/{qa_set}/{case_id}_{type}.json   ← main result + step trace
  {output_dir}/{qa_set}/{case_id}_{type}_lesions.json  ← raw reasoning per region

Usage examples
──────────────
# Ollama backend (default)
python evaluate_camelyon16.py \\
    --dataset /work/Agent_dataset/CAMELYON16/roi_qa_dataset.json \\
    --slide-dir /work/Agent_dataset/CAMELYON16/slides \\
    --qa-set CAMELYON16 \\
    --tissue breast

# HuggingFace backend
python evaluate_camelyon16.py \\
    --dataset /work/Agent_dataset/CAMELYON16/roi_qa_dataset.json \\
    --slide-dir /work/Agent_dataset/CAMELYON16/slides \\
    --qa-set CAMELYON16 \\
    --backend hf --vlm-hf Qwen/Qwen2.5-VL-7B-Instruct

# Resume interrupted run (already-completed cases are skipped automatically)
# Just re-run the same command – skips existing output files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv(override=False)

# ── ensure repo root is on sys.path ─────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CPathAgentConfig, ModelConfig, WSIConfig
from pipeline import CPathAgentPipeline


# ── Slide discovery ──────────────────────────────────────────────────────────

_SLIDE_EXTS = [".tif", ".tiff", ".svs", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu"]


def find_slide(slide_dir: str, case_id: str) -> Optional[str]:
    """Return path to slide file for *case_id*, or None if not found."""
    for ext in _SLIDE_EXTS:
        p = Path(slide_dir) / f"{case_id}{ext}"
        if p.exists():
            return str(p)
    # Case-insensitive fallback
    try:
        for f in Path(slide_dir).iterdir():
            stem = f.stem.lower()
            if stem == case_id.lower() and f.suffix.lower() in _SLIDE_EXTS:
                return str(f)
    except OSError:
        pass
    return None


# ── Result serialisation ─────────────────────────────────────────────────────

def build_result_record(
    item: Dict[str, Any],
    pipeline_result: Dict[str, Any],
    inference_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge QA item with full pipeline output into the saved record.

    Record structure
    ────────────────
    {
      "case_id"       : str,
      "type"          : str,          # DetectionLocalization | MultiLesion
      "question"      : str,
      "ground_truth"  : dict/str,

      "process": {
        "stage1_screening": {
          "groups_all"   : [...],      # all groups before background filter
          "groups"       : [...],      # groups with severity > 0
          "high_priority": [...]       # groups selected for deep analysis
        },
        "regions": [                   # one entry per high-priority region
          {
            "region_index" : int,
            "group"        : dict,
            "region_bbox"  : [x1,y1,x2,y2],
            "stage2_navigation": {
              "steps": [              # NavigateAgent output (validated steps)
                {
                  "step"             : int,
                  "magnification"    : float,
                  "region_coordinate": [x1,y1,x2,y2],
                  "reasoning"        : str,
                  "need_to_see"      : str
                }, ...
              ]
            },
            "stage3_reasoning": {
              "raw_reasoning" : str,   # full VLM output
              "report"        : str,   # extracted **Pathological Report:** section
              "answer"        : str    # VQA answer letter (if question provided)
            }
          }, ...
        ],
        "final_report": str
      },

      "prediction"    : str,           # extracted answer (VQA) or "" for report-only
      "status"        : "success" | "error",
      "error"         : str | null
    }
    """
    case_id = item["Id"]
    q_type  = item.get("type", "Unknown")
    question = item.get("Question", "")
    _answer_raw = item.get("Answer", "")
    if isinstance(_answer_raw, dict):
        answer_gt = _answer_raw.get("presence", "").lower()
    else:
        answer_gt = str(_answer_raw).lower()

    # ── build per-region trace ──────────────────────────────────────────────
    # Stage 3 runs once for all regions combined; reasoning lives at the top level.
    combined_reasoning = pipeline_result.get("reasoning", {})
    region_trace = []
    for idx, r in enumerate(pipeline_result.get("regions", [])):
        nav_steps = r.get("nav_steps", [])
        region_trace.append({
            "region_index": idx,
            "group": r.get("group", {}),
            "region_bbox": r.get("region_bbox", []),
            "stage2_navigation": {
                "steps": nav_steps,
            },
            "stage3_reasoning": {
                "raw_reasoning":      combined_reasoning.get("reasoning", ""),
                "report":             combined_reasoning.get("report", ""),
                "patch_descriptions": combined_reasoning.get("patch_descriptions", []),
                "diagnosis":          combined_reasoning.get("diagnosis", ""),
                "answer":             combined_reasoning.get("answer", ""),
            },
        })

    # ── aggregate prediction ────────────────────────────────────────────────
    # For VQA: use the extracted answer letter/option.
    # For diagnosis tasks: use the extracted short diagnosis term (e.g. "IDC").
    prediction = (
        combined_reasoning.get("answer", "")
        or combined_reasoning.get("diagnosis", "")
    )

    screening = pipeline_result.get("screening", {})

    return {
        "case_id":      case_id,
        "type":         q_type,
        "question":     question,
        "ground_truth": answer_gt,
        "process": {
            "stage1_screening": {
                "groups_all":        screening.get("groups_all", []),
                "groups":            screening.get("groups", []),
                "high_priority":     [r.get("group", {}) for r in pipeline_result.get("regions", [])],
                "descriptions":      screening.get("descriptions", {}),
                "raw_cluster_output": screening.get("raw_cluster_output", ""),
            },
            "regions":      region_trace,
            "final_report": pipeline_result.get("final_report", ""),
        },
        "prediction": prediction,
        "status":     "success",
        "error":      None,
        "inference_stats": inference_stats or {},
    }


def build_error_record(item: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    return {
        "case_id":      item["Id"],
        "type":         item.get("type", "Unknown"),
        "question":     item.get("Question", ""),
        "ground_truth": (item.get("Answer", {}).get("presence", "").lower()
                         if isinstance(item.get("Answer"), dict)
                         else str(item.get("Answer", "")).lower()),
        "process":      None,
        "prediction":   "",
        "status":       "error",
        "error":        error_msg,
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(args):
    # ── Load dataset ─────────────────────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)
    print(f"[Eval] Loaded {len(dataset)} QA items from {args.dataset}")

    # ── Slice dataset (range mode) ───────────────────────────────────────────
    total = len(dataset)
    start = max(0, args.start)
    end   = min(total, args.end) if args.end is not None else total
    dataset = dataset[start:end]
    if start != 0 or end != total:
        print(f"[Eval] Running slice [{start}:{end}] ({len(dataset)} items out of {total})")

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.qa_set
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval] Output directory: {out_dir}")

    # ── Build pipeline (model loaded once, reused for all cases) ─────────────
    model_cfg = ModelConfig(
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
        hf_load_in_8bit=args.hf_8bit,
        hf_load_in_4bit=args.hf_4bit,
        hf_use_flash_attn=args.hf_flash_attn,
        hf_use_sdpa=not args.no_sdpa,
        hf_enable_tf32=not args.no_tf32,
        hf_use_compile=args.hf_compile,
        text_model_openai=args.text_openai,
        vlm_model_openai=args.vlm_openai,
        openai_api_key=args.openai_api_key,
        openai_api_base_url=args.openai_base_url,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        qwen3_thinking=args.thinking,
    )
    wsi_cfg = WSIConfig(
        patch_size_px=args.patch_size,
        max_nav_steps=args.max_nav_steps,
    )
    cfg = CPathAgentConfig(
        model=model_cfg,
        wsi=wsi_cfg,
        output_dir=str(out_dir / "_tmp_images"),  # intermediate images go here
        save_images=args.save_images,
        verbose=not args.quiet,
    )

    pipeline = CPathAgentPipeline(cfg)

    # ── Iterate ───────────────────────────────────────────────────────────────
    skipped = 0
    success = 0
    errors  = 0

    n = len(dataset)
    for idx, item in enumerate(dataset):
        case_id = item["Id"]
        q_type  = item.get("type", "Unknown")
        out_path = out_dir / f"{case_id}_{q_type}.json"
        global_idx = start + idx + 1

        # Resume: skip if already done
        if out_path.exists() and not args.overwrite:
            skipped += 1
            if not args.quiet:
                print(f"[{global_idx}/{total}] Skip (exists): {out_path.name}")
            continue

        print(f"\n[{global_idx}/{total}] {case_id}  type={q_type}")

        # Find slide
        slide_path = find_slide(args.slide_dir, case_id)
        if slide_path is None:
            msg = f"Slide not found for {case_id} in {args.slide_dir}"
            print(f"  [WARN] {msg}")
            record = build_error_record(item, msg)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False, default=str)
            errors += 1
            continue

        print(f"  Slide: {slide_path}")

        # Run pipeline
        pipeline.model.reset_stats()
        qa_start = time.time()
        try:
            result = pipeline.run_wsi(
                wsi_path=slide_path,
                tissue_type=args.tissue,
                question=item.get("Question") or None,
                save_prefix=f"{case_id}_{q_type}",
            )
            stats = pipeline.model.get_stats()
            stats["total_elapsed_s"] = round(time.time() - qa_start, 3)
            print(
                f"  [Stats] LLM calls={stats['llm_calls']}  VLM calls={stats['vlm_calls']}"
                f"  inference={stats['total_inference_time_s']}s"
                f"  elapsed={stats['total_elapsed_s']}s"
            )
            record = build_result_record(item, result, inference_stats=stats)
            success += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] {e}\n{tb}")
            record = build_error_record(item, f"{e}\n{tb}")
            errors += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluation complete")
    print(f"  Success : {success}")
    print(f"  Errors  : {errors}")
    print(f"  Skipped : {skipped}")
    print(f"  Output  : {out_dir}")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_camelyon16",
        description="CPathAgent evaluation on CAMELYON16 (or any roi_qa_dataset.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset / IO
    io = p.add_argument_group("Dataset / IO")
    io.add_argument("--dataset", required=True,
                    help="Path to roi_qa_dataset.json")
    io.add_argument("--slide-dir", required=True,
                    help="Directory containing slide files (.tif/.svs/…)")
    io.add_argument("--qa-set", default="CAMELYON16",
                    help="Name used as subdirectory under output-dir (default: CAMELYON16)")
    io.add_argument("--output-dir", default="./output",
                    help="Root output directory (default: ./output)")
    io.add_argument("--tissue", default="breast",
                    help="Tissue type passed to the pipeline (default: breast)")
    io.add_argument("--overwrite", action="store_true",
                    help="Re-run even if output file already exists")
    io.add_argument("--start", type=int, default=0,
                    help="Start index (0-based, inclusive) of the dataset slice (default: 0)")
    io.add_argument("--end", type=int, default=None,
                    help="End index (exclusive) of the dataset slice; omit to run to the end")
    io.add_argument("--save-images", action="store_true",
                    help="Save annotated thumbnail / region / nav overlay images")
    io.add_argument("--quiet", action="store_true",
                    help="Suppress verbose pipeline output")

    # WSI processing
    wsi = p.add_argument_group("WSI processing")
    wsi.add_argument("--patch-size", type=int, default=512,
                     help="Patch size in thumbnail pixels (default: 512)")
    wsi.add_argument("--max-nav-steps", type=int, default=8)

    # Model backend
    m = p.add_argument_group("Model backend")
    m.add_argument("--backend", choices=["ollama", "hf", "openai"], default="ollama",
                   help="Backend for both text and vision (default: ollama)")
    m.add_argument("--text-backend", choices=["ollama", "hf", "openai"], default=None,
                   help="Override backend for text-only tasks (enables mixed mode)")
    m.add_argument("--vlm-backend", choices=["ollama", "hf", "openai"], default=None,
                   help="Override backend for vision tasks (enables mixed mode)")
    m.add_argument("--ollama-host", default="http://localhost:11434")
    m.add_argument("--text-ollama", default="qwen3:4b")
    m.add_argument("--vlm-ollama", default="patho_r1")
    m.add_argument("--text-hf", default="Qwen/Qwen3-4B")
    m.add_argument("--vlm-hf", default="patho_r1")
    m.add_argument("--hf-device", default="cuda")
    m.add_argument("--hf-dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    m.add_argument("--hf-8bit", action="store_true")
    m.add_argument("--hf-4bit", action="store_true")
    m.add_argument("--hf-flash-attn", action="store_true",
                   help="Use Flash Attention 2 (requires flash-attn, fp16/bf16 only)")
    m.add_argument("--no-sdpa", action="store_true",
                   help="Disable SDPA attention kernel (fall back to eager)")
    m.add_argument("--no-tf32", action="store_true",
                   help="Disable TF32 matmul on tensor cores")
    m.add_argument("--hf-compile", action="store_true",
                   help="Apply torch.compile to VLM (warmup cost, faster thereafter)")
    m.add_argument("--text-openai", default="gpt-4o-mini",
                   help="OpenAI model for text-only tasks (default: gpt-4o-mini)")
    m.add_argument("--vlm-openai", default="",
                   help="OpenAI model for vision tasks (default: gpt-4o)")
    m.add_argument("--openai-api-key", default=None,
                   help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    m.add_argument("--openai-base-url", default=None,
                   help="Override OpenAI base URL (e.g. for Azure or compatible APIs)")
    m.add_argument("--max-tokens", type=int, default=2048)
    m.add_argument("--temperature", type=float, default=0.1)
    m.add_argument("--thinking", action="store_true",
                   help="Enable Qwen3 chain-of-thought thinking tokens")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
