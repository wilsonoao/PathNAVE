"""
fact_matching.py — Compare extracted atomic pathology facts against GT evidence checklists.

For each *_facts.json in --atom_dir:
  1. Extract case_id from filename by matching known dataset case IDs
  2. Look up the GT category (LUAD / LUSC / IDC / ILC / Detection)
  3. Collect and deduplicate all extracted_facts bullet points across all entries
  4. Call LLM with the structured matching prompt
  5. Parse JSON response and save to --output_dir

Usage:
  python fact_matching.py --provider openai --model gpt-4o \\
    --atom_dir ../post_eval/CAMELYON16_detection/Evidence/Atom \\
    --output_dir ../post_eval/CAMELYON16_detection/Evidence/Matching

  python fact_matching.py --provider openai --model gpt-4o \\
    --atom_dir ../post_eval/TCGA_LUNG_Classification/Evidence/Atom \\
    --output_dir ../post_eval/TCGA_LUNG_Classification/Evidence/Matching
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROMPT_TEMPLATE = """You are given a pathology evidence sufficiency evaluation task.

Semantically compare the extracted atomic pathology facts with the curated reference evidence checklist and assign a case-level evidence sufficiency score.

Extracted atomic pathology facts:
{FACTS}

Reference evidence checklist:
{CHECKLIST}

Rules:
- Use semantic matching, not exact string matching.
- Match only findings explicitly stated as present in the extracted facts.
- Do not infer unstated findings.
- Negated or absent findings must not count as matched present evidence.
- A vague or overly general fact should not match a specific checklist item.
- Use only morphology-related evidence unless the question requires otherwise.
- Use the maximum applicable score.
- Do not add scores across matched items.

Scoring:
- 1.0: any Strong support item is matched, or any Combination set is fully matched.
- 0.5: only partial Combination evidence is matched, or only Insufficient if appearing alone evidence is matched.
- 0.0: no task-relevant checklist evidence is matched.

Return only valid JSON with the following schema:

{{
  "strong_support": [
    {{
      "item": "...",
      "matched": true,
      "matched_fact": "...",
      "reason": "..."
    }}
  ],
  "combination": [
    {{
      "set": ["finding A", "finding B"],
      "items": [
        {{
          "item": "finding A",
          "matched": true,
          "matched_fact": "..."
        }},
        {{
          "item": "finding B",
          "matched": false,
          "matched_fact": null
        }}
      ],
      "status": "complete | partial | not_matched",
      "score": 1.0
    }}
  ],
  "insufficient_if_appearing_alone": [
    {{
      "item": "...",
      "matched": false,
      "matched_fact": null,
      "reason": "..."
    }}
  ],
  "final_score": 1.0
}}"""


# ── Answer → GT key mapping ───────────────────────────────────────────────────

ANSWER_TO_GT = {
    "Invasive Ductal Carcinoma (IDC)": "IDC",
    "Invasive Lobular Carcinoma (ILC)": "ILC",
    "Lung Adenocarcinoma (LUAD)": "LUAD",
    "Lung Squamous Cell Carcinoma (LUSC)": "LUSC",
}


def load_case_gt_map(dataset_dir: Path) -> dict:
    """Build {case_id: gt_key} from all qa dataset JSONs under dataset_dir."""
    case_to_gt = {}

    datasets = [
        dataset_dir / "TCGA" / "BRCA" / "subtype_qa_dataset.json",
        dataset_dir / "TCGA" / "LUNG" / "subtype_qa_dataset.json",
        dataset_dir / "TCGA" / "LUNG_LUAD" / "subtype_qa_dataset.json",
        dataset_dir / "CAMELYON16" / "subtype_qa_dataset.json",
    ]

    for path in datasets:
        if not path.exists():
            print(f"[WARN] dataset not found: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            cid = entry["Id"]
            answer = entry.get("Answer")
            if isinstance(answer, dict):
                # CAMELYON16: only include presence=Yes
                if answer.get("presence") == "Yes":
                    case_to_gt[cid] = "Detection"
            elif isinstance(answer, str):
                gt_key = ANSWER_TO_GT.get(answer)
                if gt_key:
                    case_to_gt[cid] = gt_key

    return case_to_gt


def extract_case_id(stem: str, known_ids: list) -> Optional[str]:
    """Return the known case ID that is a prefix of stem (followed by _ or end)."""
    for cid in known_ids:
        if stem.startswith(cid):
            rest = stem[len(cid):]
            if rest == "" or rest.startswith("_"):
                return cid
    return None


# ── Fact parsing ──────────────────────────────────────────────────────────────

def collect_facts(facts_data: list) -> list:
    """Pool and deduplicate all extracted_facts bullet points from a facts file."""
    seen = set()
    facts = []
    for entry in facts_data:
        raw = entry.get("extracted_facts", "")
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("*"):
                fact = line.lstrip("*").strip()
                # Only include "present" facts; skip "absent" facts
                if fact and "absent" not in fact.lower() and fact not in seen:
                    seen.add(fact)
                    facts.append(fact)
    return facts


# ── GT checklist formatting ───────────────────────────────────────────────────

def format_checklist(gt_entry: dict) -> str:
    lines = []

    strong = gt_entry.get("strong_support", [])
    if strong:
        lines.append("Strong support (any one matched = score 1.0):")
        for item in strong:
            lines.append(f"  - {item}")

    combos = gt_entry.get("combination", [])
    if combos:
        lines.append("\nCombination sets (any complete set matched = score 1.0):")
        for i, combo in enumerate(combos, 1):
            lines.append(f"  Set {i}: {' + '.join(combo)}")

    insuff = gt_entry.get("insufficient_if_alone", [])
    if insuff:
        lines.append("\nInsufficient if appearing alone (only these matched = score 0.5):")
        for item in insuff:
            lines.append(f"  - {item}")

    return "\n".join(lines)


# ── Correct-flag lookup ───────────────────────────────────────────────────────

def _parse_correct(val) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return None


def lookup_correct(facts_stem: str, case_id: str, result_dir: Path) -> Optional[bool]:
    """Find the correct field for this case from the original result files.

    Strategy:
      1. Exact stem match: search for {facts_stem}.json recursively in result_dir.
         If found and has a non-None correct field, return it.
      2. Case-ID fallback: search any JSON whose case_id/long_id/slide_id == case_id.
         Useful when the result filename hash differs from the facts filename hash.
    """
    ID_FIELDS = ("case_id", "long_id", "slide_id")

    def read_json(path: Path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # Pass 1: exact filename match
    for result_file in result_dir.rglob(f"{facts_stem}.json"):
        if "_tmp_images" in result_file.parts:
            continue
        data = read_json(result_file)
        if data is None:
            continue
        val = _parse_correct(data.get("correct"))
        if val is not None:
            return val

    # Pass 2: case_id-based fallback
    for result_file in sorted(result_dir.rglob("*.json")):
        if "_tmp_images" in result_file.parts:
            continue
        data = read_json(result_file)
        if data is None:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            for field in ID_FIELDS:
                if item.get(field) == case_id:
                    val = _parse_correct(item.get("correct"))
                    if val is not None:
                        return val
                    break

    return None


# ── LLM backends ─────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


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


# ── JSON extraction from LLM response ────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Try to parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Match extracted facts against GT evidence checklists")
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-4o (OpenAI) or llama3 (Ollama)")
    parser.add_argument("--atom_dir", required=True,
                        help="Directory with *_facts.json files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write *_matching.json files")
    parser.add_argument("--gt_file", default="/work/Agent_dataset/LLM_GT/GT.json",
                        help="Path to GT.json")
    parser.add_argument("--dataset_dir", default="/work/Agent_dataset",
                        help="Root dir containing TCGA/ and CAMELYON16/ subdirs")
    parser.add_argument("--result_dir", default=None,
                        help="Root directory of original agent result files (used to look up correct field)")
    parser.add_argument("--ollama_url", default="http://localhost:11434")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else script_dir / pp

    atom_dir = resolve(args.atom_dir)
    output_dir = resolve(args.output_dir)
    gt_file = Path(args.gt_file)
    dataset_dir = Path(args.dataset_dir)
    result_dir = resolve(args.result_dir) if args.result_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT checklists
    with open(gt_file, encoding="utf-8") as f:
        gt_db = json.load(f)
    print(f"Loaded GT keys: {list(gt_db.keys())}")

    # Load case → GT key mapping
    case_to_gt = load_case_gt_map(dataset_dir)
    print(f"Loaded {len(case_to_gt)} case-GT mappings")

    # Sort known IDs by length descending for prefix matching (longest first)
    known_ids = sorted(case_to_gt.keys(), key=len, reverse=True)

    # Build LLM caller
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

    # Process facts files
    fact_files = sorted(atom_dir.glob("*_facts.json"))
    if not fact_files:
        print(f"No *_facts.json found in {atom_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    skipped = 0
    no_gt = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for facts_file in fact_files:
        total += 1
        stem = facts_file.stem  # e.g. "test_001_a9247a2a_facts" without .json
        # stem ends with "_facts"; strip it
        if stem.endswith("_facts"):
            stem = stem[:-6]

        out_file = output_dir / (stem + "_matching.json")

        # If matching file already exists but lacks correct, patch it in-place
        if out_file.exists():
            if result_dir is not None:
                try:
                    existing = json.loads(out_file.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                if "correct" not in existing:
                    existing_case_id = existing.get("case_id") or extract_case_id(stem, known_ids)
                    if existing_case_id:
                        correct = lookup_correct(stem, existing_case_id, result_dir)
                        existing["correct"] = correct
                        out_file.write_text(
                            json.dumps(existing, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        print(f"[PATCH] {facts_file.name}  correct={correct}")
                        skipped += 1
                        continue
            skipped += 1
            print(f"[SKIP] {facts_file.name}")
            continue

        # Extract case ID
        case_id = extract_case_id(stem, known_ids)
        if case_id is None:
            print(f"[NO_ID] {facts_file.name} — could not match to a known case ID", file=sys.stderr)
            no_gt += 1
            continue

        gt_key = case_to_gt.get(case_id)
        if gt_key is None:
            # Negative CAMELYON16 case (presence=No) — skip
            print(f"[SKIP_NEG] {facts_file.name} — case {case_id} has no GT (likely negative)")
            no_gt += 1
            continue

        gt_entry = gt_db.get(gt_key)
        if gt_entry is None:
            print(f"[NO_GT] {facts_file.name} — GT key '{gt_key}' not found", file=sys.stderr)
            no_gt += 1
            continue

        with open(facts_file, encoding="utf-8") as f:
            facts_data = json.load(f)

        facts_list = collect_facts(facts_data)
        if not facts_list:
            print(f"[NO_FACTS] {facts_file.name} — no present facts found")
            no_gt += 1
            continue

        checklist_text = format_checklist(gt_entry)
        facts_text = "\n".join(f"- {f}" for f in facts_list)
        prompt = PROMPT_TEMPLATE.format(FACTS=facts_text, CHECKLIST=checklist_text)

        # Look up whether the agent answered correctly
        correct = None
        if result_dir is not None:
            correct = lookup_correct(stem, case_id, result_dir)

        print(f"\n[{total}] {facts_file.name}")
        print(f"  case_id : {case_id}")
        print(f"  gt_key  : {gt_key}")
        print(f"  correct : {correct}")
        print(f"  facts   : {len(facts_list)}")

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
            "correct": correct,
            "facts_file": str(facts_file.name),
            "facts_count": len(facts_list),
            "facts": facts_list,
            "matching": parsed,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  => saved {out_file.name}")

    grand_total = total_prompt_tokens + total_completion_tokens
    processed = total - skipped - no_gt
    print(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  Files total     : {total}\n"
        f"  Processed       : {processed}\n"
        f"  Skipped (exist) : {skipped}\n"
        f"  Skipped (no GT) : {no_gt}\n"
        f"  Prompt tokens   : {total_prompt_tokens:,}\n"
        f"  Completion tokens: {total_completion_tokens:,}\n"
        f"  Grand total tokens: {grand_total:,}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
