"""
Extract atomic pathology facts from PathAgent result JSON files.

For each result JSON:
  - Regular patches: look up descriptions from patches_descriptions*.json by long_id + patch_name
  - Zoom patch: use the zoom_patch_desc field directly from the result JSON
  - Each description is sent to LLM with the fact-extraction prompt

Usage:
  python extract_facts_pathagent.py --provider openai --model gpt-4o
  python extract_facts_pathagent.py --provider ollama --model llama3
  python extract_facts_pathagent.py --provider openai --model gpt-4o \
    --input_dir ../result/CAMELYON16/results \
    --desc_dir ../result/CAMELYON16/desc \
    --output_dir result_facts_pathagent
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


def load_desc_files(desc_dir: Path) -> dict:
    """Merge all patches_descriptions*.json into one nested dict: long_id -> patch_name -> text."""
    merged: dict = {}
    for desc_file in sorted(desc_dir.glob("patches_descriptions*.json")):
        with open(desc_file, encoding="utf-8") as f:
            data = json.load(f)
        for long_id, patches in data.items():
            if long_id not in merged:
                merged[long_id] = {}
            for patch_name, text in patches.items():
                if patch_name not in merged[long_id]:
                    merged[long_id][patch_name] = text
    return merged


def load_pathagent_results(input_dir: Path, desc_dir: Path):
    """Yield (json_path, list[dict]) for every result JSON found.

    Each dict has keys: patch_name, source, text, and optionally attempt.
    """
    all_descs = load_desc_files(desc_dir)

    for json_file in sorted(input_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        long_id = data.get("long_id", "")
        proc = data.get("process", [])
        if not proc:
            continue

        slide_descs = all_descs.get(long_id, {})
        texts = []
        seen_patches: set = set()

        # Regular patches: prefer accumulated_patch_names from the last attempt
        # (CAMELYON16 style — cumulative list). Fall back to collecting
        # evaluated_patches_this_round from all attempts (TCGA_BRCA style).
        last_attempt = proc[-1]
        all_patch_names = last_attempt.get("accumulated_patch_names") or []
        if not all_patch_names:
            for attempt in proc:
                all_patch_names.extend(attempt.get("evaluated_patches_this_round", []))

        for patch_name in all_patch_names:
            if patch_name in seen_patches:
                continue
            seen_patches.add(patch_name)
            desc_text = slide_descs.get(patch_name)
            if desc_text:
                texts.append({
                    "patch_name": patch_name,
                    "source": "patches_descriptions",
                    "text": desc_text,
                })

        # Zoom patch: collect from every attempt that has one
        for attempt in proc:
            zoom_desc = attempt.get("zoom_patch_desc", "")
            if not zoom_desc:
                continue
            zoom_patch = attempt.get("selected_zoom_patch", "")
            texts.append({
                "patch_name": zoom_patch,
                "source": "zoom_patch_desc",
                "attempt": attempt.get("attempt", 1),
                "text": zoom_desc,
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
    parser = argparse.ArgumentParser(description="Extract atomic pathology facts from PathAgent results")
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-4o (OpenAI) or llama3 (Ollama)")
    parser.add_argument("--input_dir", default="../result/CAMELYON16/results",
                        help="Directory containing PathAgent result JSON files")
    parser.add_argument("--desc_dir", default="../result/CAMELYON16/desc",
                        help="Directory containing patches_descriptions*.json files")
    parser.add_argument("--output_dir", default="result_facts_pathagent",
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
    desc_dir = resolve(args.desc_dir)
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

    for json_file, texts in load_pathagent_results(input_dir, desc_dir):
        total_files += 1
        out_file = output_dir / (json_file.stem + "_facts.json")

        if out_file.exists():
            skipped_files += 1
            print(f"[SKIP] {json_file.name}")
            continue

        print(f"\n[{total_files}] {json_file.name}  ({len(texts)} descriptions)")
        results = []
        file_prompt_tokens = 0
        file_completion_tokens = 0

        for entry in texts:
            source_label = f"{entry['source']}"
            if entry["source"] == "zoom_patch_desc":
                source_label += f" (attempt {entry.get('attempt', '?')})"
            preview = entry["text"][:80].replace("\n", " ")
            print(f"  [{entry['patch_name']} | {source_label}] {preview}...")

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

            out_entry = {
                "patch_name": entry["patch_name"],
                "source": entry["source"],
                "source_text": entry["text"],
                "extracted_facts": result.text,
            }
            if entry["source"] == "zoom_patch_desc":
                out_entry["attempt"] = entry.get("attempt", 1)
            results.append(out_entry)
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
        f"  Total descriptions: {total_descs}\n"
        f"  Prompt tokens     : {total_prompt_tokens:,}\n"
        f"  Completion tokens : {total_completion_tokens:,}\n"
        f"  Grand total tokens: {grand_total:,}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
