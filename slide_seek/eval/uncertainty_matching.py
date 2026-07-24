"""
uncertainty_matching.py — Uncertainty-aware evidence matching for TCGA_LUNG_Classification.

For each *_matching.json in --matching_dir:
  - correct=True  → copy to output_dir with effective_key=gt_key (no LLM call needed)
  - correct=False → re-run evidence matching against the PREDICTED label's checklist
                    (LUAD→LUSC, LUSC→LUAD) and keep correct=False in the output

This reveals whether a wrong prediction was at least internally consistent with
the agent's own extracted facts.

Usage:
  python uncertainty_matching.py --provider openai --model gpt-4o \\
    --matching_dir ../post_eval/TCGA_LUNG_Classification/Evidence/Matching \\
    --atom_dir     ../post_eval/TCGA_LUNG_Classification/Evidence/Atom \\
    --output_dir   ../post_eval/TCGA_LUNG_Classification/Evidence/Uncertainty_matching
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Re-use helpers from fact_matching in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from fact_matching import (
    PROMPT_TEMPLATE,
    LLMResult,
    call_openai,
    call_ollama,
    collect_facts,
    extract_json,
    format_checklist,
)

# Only binary LUNG subtypes are supported by this script
OPPOSITE = {"LUAD": "LUSC", "LUSC": "LUAD", "IDC": "ILC", "ILC": "IDC"}


def _parse_correct(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Uncertainty evidence matching for TCGA_LUNG_Classification"
    )
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True, help="e.g. gpt-4o (OpenAI) or llama3 (Ollama)")
    parser.add_argument("--matching_dir", required=True,
                        help="Directory with existing *_matching.json files (Evidence/Matching)")
    parser.add_argument("--atom_dir", required=True,
                        help="Directory with *_facts.json files (Evidence/Atom)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write uncertainty *_matching.json files")
    parser.add_argument("--gt_file", default="/work/Agent_dataset/LLM_GT/GT.json",
                        help="Path to GT.json")
    parser.add_argument("--ollama_url", default="http://localhost:11434")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else script_dir / pp

    matching_dir = resolve(args.matching_dir)
    atom_dir = resolve(args.atom_dir)
    output_dir = resolve(args.output_dir)
    gt_file = Path(args.gt_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_file, encoding="utf-8") as f:
        gt_db = json.load(f)
    print(f"Loaded GT keys: {list(gt_db.keys())}")

    # Build LLM caller (only needed for incorrect cases)
    if args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("openai package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key required. Set OPENAI_API_KEY in .env or pass --api_key",
                  file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        llm_call = lambda prompt: call_openai(client, args.model, prompt, args.reasoning_effort)
    else:
        llm_call = lambda prompt: call_ollama(args.ollama_url, args.model, prompt)

    matching_files = sorted(matching_dir.glob("*_matching.json"))
    if not matching_files:
        print(f"No *_matching.json found in {matching_dir}", file=sys.stderr)
        sys.exit(1)

    total = len(matching_files)
    copied = 0
    rematched = 0
    skipped = 0
    skipped_null = 0
    skipped_unsupported = 0
    no_facts = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, mf in enumerate(matching_files, 1):
        out_file = output_dir / mf.name
        if out_file.exists():
            skipped += 1
            print(f"[SKIP] {mf.name}")
            continue

        try:
            obj = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERR] {mf.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        gt_key = obj.get("gt_key")
        correct = _parse_correct(obj.get("correct"))
        case_id = obj.get("case_id", "")

        if correct is None:
            print(f"[SKIP_NULL] {mf.name} — correct is None, skipping")
            skipped_null += 1
            continue

        # ── Correct case: copy with effective_key added ───────────────────────
        if correct is True:
            obj["effective_key"] = gt_key
            out_file.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            copied += 1
            print(f"[COPY] [{idx}/{total}] {mf.name}  (gt_key={gt_key})")
            continue

        # ── Incorrect case: re-match against predicted label's checklist ──────
        effective_key = OPPOSITE.get(gt_key)
        if effective_key is None:
            print(f"[SKIP_KEY] {mf.name} — gt_key='{gt_key}' not in LUAD/LUSC, skipping",
                  file=sys.stderr)
            skipped_unsupported += 1
            continue

        gt_entry = gt_db.get(effective_key)
        if gt_entry is None:
            print(f"[NO_GT] {mf.name} — effective_key '{effective_key}' not in GT.json",
                  file=sys.stderr)
            skipped_unsupported += 1
            continue

        # Derive atom filename: strip "_matching" from stem, add "_facts.json"
        stem = mf.stem  # e.g. "TCGA-xx-xx_..._matching"
        if stem.endswith("_matching"):
            facts_stem = stem[:-9]  # remove "_matching"
        else:
            facts_stem = stem
        facts_file = atom_dir / (facts_stem + "_facts.json")

        if not facts_file.exists():
            print(f"[NO_FACTS_FILE] {mf.name} — {facts_file.name} not found", file=sys.stderr)
            no_facts += 1
            continue

        with open(facts_file, encoding="utf-8") as f:
            facts_data = json.load(f)

        facts_list = collect_facts(facts_data)
        if not facts_list:
            print(f"[NO_FACTS] {mf.name} — no present facts extracted")
            no_facts += 1
            continue

        checklist_text = format_checklist(gt_entry)
        facts_text = "\n".join(f"- {f}" for f in facts_list)
        prompt = PROMPT_TEMPLATE.format(FACTS=facts_text, CHECKLIST=checklist_text)

        print(f"\n[REMATCH] [{idx}/{total}] {mf.name}")
        print(f"  case_id      : {case_id}")
        print(f"  gt_key       : {gt_key}  →  effective_key: {effective_key}")
        print(f"  facts        : {len(facts_list)}")

        try:
            result = llm_call(prompt)
        except Exception as e:
            print(f"  !! LLM error: {e}", file=sys.stderr)
            result = LLMResult(text=f"ERROR: {e}")

        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        reasoning_str = f"  (reasoning={result.reasoning_tokens})" if result.reasoning_tokens else ""
        print(
            f"  tokens: prompt={result.prompt_tokens}  "
            f"completion={result.completion_tokens}{reasoning_str}  "
            f"total={result.total_tokens}"
        )

        parsed = extract_json(result.text)
        if parsed is None:
            print(f"  !! Could not parse JSON from response", file=sys.stderr)
            parsed = {"parse_error": True, "raw_response": result.text}

        output = {
            "case_id": case_id,
            "gt_key": gt_key,
            "effective_key": effective_key,
            "correct": False,
            "facts_file": facts_file.name,
            "facts_count": len(facts_list),
            "facts": facts_list,
            "matching": parsed,
        }

        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  => saved {out_file.name}")
        rematched += 1

    grand_total_tokens = total_prompt_tokens + total_completion_tokens
    print(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  Total input files   : {total}\n"
        f"  Copied (correct)    : {copied}\n"
        f"  Re-matched (wrong)  : {rematched}\n"
        f"  Skipped (exist)     : {skipped}\n"
        f"  Skipped (null corr) : {skipped_null}\n"
        f"  Skipped (unsupported): {skipped_unsupported}\n"
        f"  Skipped (no facts)  : {no_facts}\n"
        f"  Prompt tokens       : {total_prompt_tokens:,}\n"
        f"  Completion tokens   : {total_completion_tokens:,}\n"
        f"  Grand total tokens  : {grand_total_tokens:,}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
