"""
Generate diagnostic fact-atom criteria for each pathology Q&A.

For each input JSON file:
  - Extracts the question (from first user message) and ground_truth answer
  - Sends to LLM with the fact-atom prompt
  - Saves result to <output_dir>/fact_atom/<stem>_fact_atom.json

Usage:
  python generate_fact_atoms.py --provider openai --model gpt-4o \
      --input_dir result/TCGA_BRCA/test_total \
      --output_dir post_eval/TCGA_BRCA

  python generate_fact_atoms.py --provider ollama --model llama3 \
      --input_dir result/TCGA_BRCA/test_total \
      --output_dir post_eval/TCGA_BRCA
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROMPT_TEMPLATE = """You are a pathology expert.

Your task is to define diagnostic evidence criteria for supporting a specific answer to a histopathology question.

You must express all evidence as atomic pathology facts.

----------------------------------------
Example
----------------------------------------
Example 1:
Question:
Is this slide consistent with breast carcinoma (BRCA)?

Answer:
Yes

<single_sufficient>
- stromal invasion present
</single_sufficient>

<minimal_sets>

Set 1:
- nuclear pleomorphism present
- high mitotic activity present

Set 2:
- irregular ductal structures present
- architectural distortion present

</minimal_sets>

<proportional_sets>

Group 1:
Atoms:
- nuclear pleomorphism present
- high mitotic activity present
- hyperchromatic nuclei present
Threshold: 2 / 3

</proportional_sets>

Example 2:
Question:
Does this slide contain more than one tumor region?

Answer:
Yes

<single_sufficient>
- more than one tumor region present
</single_sufficient>

<minimal_sets>

Set 1:
- tumor region present in one location
- tumor region present in a separate location

</minimal_sets>

<proportional_sets>

Group 1:
Atoms:
- tumor region present in one location
- tumor region present in a separate location
- tumor regions are spatially distinct
Threshold: 2 / 3

</proportional_sets>


----------------------------------------
Now your task
----------------------------------------

Question:
{YOUR_QUESTION}

Answer:
{TARGET_ANSWER}

Requirements:

[Task Awareness]
1. Identify the question type:

   (A) Morphology → cellular / architectural features
   (B) Spatial → number, location, separation
   (C) Feature Presence → whether a specific feature exists
   (D) Classification → subtype or category decision
   (E) Descriptive → summarize key pathological findings

2. Adapt evidence accordingly:

   - Morphology → atypia, mitosis, structure
   - Spatial → region count, separation
   - Feature Presence → direct presence/absence of feature
   - Classification → combination of discriminative features
   - Descriptive → key representative findings only

[Atomicity]
3. Each atom must express ONE fact only

[Non-redundancy]
4. Do NOT repeat the same concept

[Set Design]
5. Use minimal structure:
   - Simple tasks → 1 minimal set
   - Complex tasks → up to 2 minimal sets

[Proportional Sets]
6. Use at most ONE proportional group if needed

[Limits]
7. Maximum:
   - 1 single_sufficient
   - 1–2 minimal sets
   - 1 proportional group

----------------------------------------
Output format (STRICT):

<single_sufficient>
- ...
</single_sufficient>

<minimal_sets>

Set 1:
- ...

</minimal_sets>

<proportional_sets>

Group 1:
Atoms:
- ...
Threshold: X / N

</proportional_sets>"""


# ── Data helpers ──────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _extract_question_from_message(content) -> str:
    """Pull the Question: ... block out of the first user message."""
    if isinstance(content, list):
        text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = str(content)
    m = re.search(r"Question:\s*\n(.+?)(?:\n\n|\nFormat|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def load_cases(input_dir: Path):
    """Yield (json_path, question, answer) for every .json file."""
    for json_file in sorted(input_dir.rglob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        # Extract ground-truth answer
        answer = data.get("ground_truth", "").strip()
        if not answer:
            continue

        # Extract question from first user turn
        chat = data.get("chat_history", data.get("messages", []))
        user_turns = [m for m in chat if m.get("role") == "user"]
        if not user_turns:
            continue
        question = _extract_question_from_message(user_turns[0]["content"])
        if not question:
            continue

        yield json_file, question, answer


# ── LLM backends ─────────────────────────────────────────────────────────────

def call_openai(client, model: str, prompt: str, reasoning_effort: str | None = None) -> LLMResult:
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
    parser = argparse.ArgumentParser(description="Generate diagnostic fact-atom criteria via LLM")
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-4o (OpenAI) or llama3 (Ollama)")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input .json files")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory; results go into <output_dir>/fact_atom/")
    parser.add_argument("--ollama_url", default="http://localhost:11434")
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
    output_dir = resolve(args.output_dir) / "fact_atom"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build LLM caller ──
    if args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("openai not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key required. Set OPENAI_API_KEY in .env or pass --api_key",
                  file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        llm_call = lambda p: call_openai(client, args.model, p, args.reasoning_effort)
    else:
        llm_call = lambda p: call_ollama(args.ollama_url, args.model, p)

    # ── Process cases ──
    total = skipped = 0
    total_prompt = total_completion = total_reasoning = 0

    for json_file, question, answer in load_cases(input_dir):
        total += 1
        out_file = output_dir / (json_file.stem + "_fact_atom.json")

        if out_file.exists():
            skipped += 1
            print(f"[SKIP] {json_file.name}")
            continue

        print(f"\n[{total}] {json_file.name}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

        prompt = (PROMPT_TEMPLATE
                  .replace("{YOUR_QUESTION}", question)
                  .replace("{TARGET_ANSWER}", answer))

        try:
            result = llm_call(prompt)
        except Exception as e:
            print(f"  !! LLM error: {e}", file=sys.stderr)
            result = LLMResult(text=f"ERROR: {e}")

        reasoning_str = f"  (reasoning={result.reasoning_tokens})" if result.reasoning_tokens else ""
        print(
            f"  tokens: prompt={result.prompt_tokens}  "
            f"completion={result.completion_tokens}{reasoning_str}  "
            f"total={result.total_tokens}"
        )
        print(result.text)

        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens
        total_reasoning += result.reasoning_tokens

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": json_file.name,
                "question": question,
                "answer": answer,
                "fact_atoms": result.text,
                "tokens": {
                    "prompt": result.prompt_tokens,
                    "completion": result.completion_tokens,
                    "reasoning": result.reasoning_tokens,
                    "total": result.total_tokens,
                },
            }, f, indent=2, ensure_ascii=False)
        print(f"  => saved {out_file}")

    processed = total - skipped
    grand_total = total_prompt + total_completion
    print(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  Files processed  : {processed}  (skipped: {skipped})\n"
        f"  Prompt tokens    : {total_prompt:,}\n"
        f"  Completion tokens: {total_completion:,}\n"
        f"  Reasoning tokens : {total_reasoning:,}\n"
        f"  Grand total      : {grand_total:,}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
