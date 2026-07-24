"""
Extract atomic pathology facts from SlideSeek output JSON files.

For each output JSON (non-lesions, non-manifest):
  - visited_rois[*].description is sent to LLM with the fact-extraction prompt
  - Provenance is recorded: slide_id, roi_index, x, y, magnification, size

Input formats supported:
  - CAMELYON16_detection/  test_XXX_DetectionLocalization.json
  - tcga_brca_subtype/     TCGA-XXX_Subtype.json

Usage:
  python extract_facts_slideseek.py --provider openai --model gpt-4o \
    --input_dir ../CAMELYON16_detection \
    --output_dir result_facts_camelyon16

  python extract_facts_slideseek.py --provider openai --model gpt-4o \
    --input_dir ../tcga_brca_subtype \
    --output_dir result_facts_tcga_brca
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROMPT_TEMPLATE = """You are a pathology expert.

Your task is to extract atomic pathology facts from a given description.

Requirements:

1. Each fact must describe a single, independent pathological observation.
2. Each fact must be written in a standardized binary form:

   * Use the format: "<feature> present" or "<feature> absent"
3. Do NOT combine multiple findings into one sentence.
4. Do NOT include explanations, reasoning, or interpretations.
5. Do NOT include uncertainty words (e.g., "suggests", "likely").
6. Use consistent and canonical pathology terminology.
7. Remove redundancy (no duplicate facts).
8. Each fact must be independently verifiable from image evidence.

---

## Example 1

Input:
The tissue shows invasive carcinoma with irregular ductal structures,
nuclear pleomorphism, and increased mitotic figures.

Output:
* stromal invasion present
* irregular ductal structures present
* nuclear pleomorphism present
* high mitotic activity present

---

Now process the following input:

Input:
{TEXT}

Output:"""


# ── Data helpers ──────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # o-series only: hidden CoT tokens inside completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def load_slideseek_results(input_dir: Path):
    """Yield (json_path, list[dict]) for every SlideSeek output JSON.

    Skips _lesions.json files (same visited_rois as main file) and
    manifest_*.json files (aggregate summaries without descriptions).

    Each dict has keys:
      slide_id, roi_index, x, y, magnification, size, text
    """
    for json_file in sorted(input_dir.glob("*.json")):
        stem = json_file.stem
        # Skip lesion annotation files and manifest summaries
        if stem.endswith("_lesions") or stem.startswith("manifest"):
            continue

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        visited_rois = data.get("visited_rois", [])
        if not visited_rois:
            continue

        slide_id = data.get("slide_id", stem)
        texts = []

        for roi_idx, roi in enumerate(visited_rois):
            desc = roi.get("description", "")
            if not desc:
                continue
            texts.append({
                "slide_id": slide_id,
                "roi_index": roi_idx,
                "x": roi.get("x"),
                "y": roi.get("y"),
                "magnification": roi.get("magnification"),
                "size": roi.get("size"),
                "text": desc,
            })

        if texts:
            yield json_file, texts


# ── LLM backends ─────────────────────────────────────────────────────────────

def call_openai(client, model: str, prompt: str, reasoning_effort: Optional[str] = None) -> LLMResult:
    is_reasoning_model = model.startswith(("o1", "o3", "o4"))
    kwargs = {}
    if not is_reasoning_model:
        kwargs["temperature"] = 0
    if is_reasoning_model and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    usage = response.usage
    reasoning = 0
    if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    return LLMResult(
        text=response.choices[0].message.content.strip(),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=reasoning,
    )


def call_ollama(base_url: str, model: str, prompt: str) -> LLMResult:
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    return LLMResult(
        text=result["message"]["content"].strip(),
        prompt_tokens=result.get("prompt_eval_count", 0),
        completion_tokens=result.get("eval_count", 0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract atomic pathology facts from SlideSeek outputs")
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-4o (OpenAI) or llama3 (Ollama)")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing SlideSeek output JSON files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write facts.json output files")
    parser.add_argument("--ollama_url", default="http://localhost:11434",
                        help="Ollama server base URL")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API key (falls back to OPENAI_API_KEY in .env)")
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort for o-series models (default: low)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else script_dir / pp

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build LLM caller ──
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

    # ── Process results ──
    total_files = 0
    skipped_files = 0
    total_descs = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for json_file, texts in load_slideseek_results(input_dir):
        total_files += 1
        out_file = output_dir / (json_file.stem + "_facts.json")

        if out_file.exists():
            skipped_files += 1
            print(f"[SKIP] {json_file.name}")
            continue

        print(f"\n[{total_files}] {json_file.name}  ({len(texts)} ROIs)")
        results = []
        file_prompt_tokens = 0
        file_completion_tokens = 0

        for entry in texts:
            label = f"roi={entry['roi_index']} ({entry['x']},{entry['y']}) mag={entry['magnification']}"
            preview = entry["text"][:80].replace("\n", " ")
            print(f"  [{label}] {preview}...")

            prompt = PROMPT_TEMPLATE.replace("{TEXT}", entry["text"])
            try:
                result = llm_call(prompt)
            except Exception as e:
                print(f"  !! LLM error: {e}", file=sys.stderr)
                result = LLMResult(text=f"ERROR: {e}")

            file_prompt_tokens += result.prompt_tokens
            file_completion_tokens += result.completion_tokens
            reasoning_str = f"  (reasoning={result.reasoning_tokens})" if result.reasoning_tokens else ""
            print(
                f"           tokens: prompt={result.prompt_tokens}  "
                f"completion={result.completion_tokens}{reasoning_str}  "
                f"total={result.total_tokens}"
            )
            print(result.text)

            results.append({
                "slide_id": entry["slide_id"],
                "roi_index": entry["roi_index"],
                "x": entry["x"],
                "y": entry["y"],
                "magnification": entry["magnification"],
                "size": entry["size"],
                "source_text": entry["text"],
                "extracted_facts": result.text,
            })
            total_descs += 1

        total_prompt_tokens += file_prompt_tokens
        total_completion_tokens += file_completion_tokens
        file_total = file_prompt_tokens + file_completion_tokens
        print(
            f"  => file tokens: prompt={file_prompt_tokens}  "
            f"completion={file_completion_tokens}  total={file_total}"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  => saved {out_file}")

    grand_total = total_prompt_tokens + total_completion_tokens
    processed = total_files - skipped_files
    print(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  Files processed   : {processed}  (skipped: {skipped_files})\n"
        f"  Total ROIs        : {total_descs}\n"
        f"  Prompt tokens     : {total_prompt_tokens:,}\n"
        f"  Completion tokens : {total_completion_tokens:,}\n"
        f"  Grand total tokens: {grand_total:,}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
