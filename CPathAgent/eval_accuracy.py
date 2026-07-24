"""
Evaluate prediction accuracy from CPathAgent output JSON files.

- Reads all *.json files in the given folder (or nested with --recursive)
- Compares `prediction` vs `ground_truth` case-insensitively
- Treats known abbreviations and their full names as equivalent
  (e.g. "IDC" == "invasive ductal carcinoma")
- Prints per-class breakdown and overall accuracy

Usage:
    python eval_accuracy.py <output_folder>
    python eval_accuracy.py <output_folder> --recursive
    python eval_accuracy.py <output_folder> --errors-only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Alias table ──────────────────────────────────────────────────────────────
# Each entry: canonical key (lowercase) → set of accepted surface forms (lowercase)
# Add more aliases here as needed.
ALIASES: Dict[str, List[str]] = {
    "idc": [
        "idc",
        "invasive ductal carcinoma",
        "invasive ductal carcinoma (idc)",
        "infiltrating ductal carcinoma",
        "infiltrating ductal carcinoma (idc)",
        "invasive ductal",
        "ductal carcinoma",
    ],
    "ilc": [
        "ilc",
        "invasive lobular carcinoma",
        "invasive lobular carcinoma (ilc)",
        "infiltrating lobular carcinoma",
        "infiltrating lobular carcinoma (ilc)",
        "invasive lobular",
        "lobular carcinoma",
    ],
    "dcis": [
        "dcis",
        "ductal carcinoma in situ",
        "ductal carcinoma in situ (dcis)",
    ],
    "lcis": [
        "lcis",
        "lobular carcinoma in situ",
        "lobular carcinoma in situ (lcis)",
    ],
    "luad": [
        "luad",
        "lung adenocarcinoma",
        "lung adenocarcinoma (luad)",
        "adenocarcinoma",
        "adenocarcinoma (luad)",
    ],
    "lusc": [
        "lusc",
        "lung squamous cell carcinoma",
        "lung squamous cell carcinoma (lusc)",
        "squamous cell carcinoma",
        "squamous cell carcinoma (lusc)",
    ],
    "normal": [
        "normal",
        "normal tissue",
        "benign",
        "benign tissue",
        "no malignancy",
        "no tumor",
        "non-tumor",
        "non-malignant",
    ],
}

# Build reverse lookup: surface_form → canonical
_SURFACE_TO_CANONICAL: Dict[str, str] = {}
for _canon, _forms in ALIASES.items():
    for _form in _forms:
        _SURFACE_TO_CANONICAL[_form.lower()] = _canon


def canonicalise(text: str) -> str:
    """Lowercase and map to canonical alias if one exists."""
    t = text.strip().lower()
    return _SURFACE_TO_CANONICAL.get(t, t)


def is_correct(prediction: str, ground_truth: str) -> bool:
    return canonicalise(prediction) == canonicalise(ground_truth)


# ── JSON loading ─────────────────────────────────────────────────────────────

def load_records(folder: Path, recursive: bool) -> List[dict]:
    pattern = "**/*.json" if recursive else "*.json"
    records = []
    for path in sorted(folder.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            # Only keep records that have both fields
            if "prediction" in d and "ground_truth" in d:
                if "correct" not in d:
                    d["correct"] = is_correct(str(d["prediction"]), str(d["ground_truth"]))
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(d, f, ensure_ascii=False, indent=2)
                d["_file"] = path.name
                records.append(d)
        except Exception as e:
            print(f"[WARN] Could not read {path.name}: {e}")
    return records


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(records: List[dict], errors_only: bool, show_errors: int) -> None:
    total = 0
    correct = 0
    skipped_error = 0

    # per-class: {gt_canonical: [correct_count, total_count]}
    per_class: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    wrong_rows: List[Tuple[str, str, str, str]] = []  # (file, gt, pred, gt_canon, pred_canon)

    for d in records:
        status = d.get("status", "success")
        gt_raw  = str(d.get("ground_truth", "")).strip()
        pred_raw = str(d.get("prediction", "")).strip()
        fname = d.get("_file", "?")

        if status == "error":
            skipped_error += 1
            if errors_only:
                print(f"  [ERROR] {fname}  gt={gt_raw!r}  error={d.get('error', '')[:80]}")
            continue

        total += 1
        gt_canon   = canonicalise(gt_raw)
        pred_canon = canonicalise(pred_raw)
        print(f"gt: {gt_canon}")
        print(f"pred: {pred_canon}")
        print("==============")
        ok = bool(d.get("correct", gt_canon == pred_canon))

        per_class[gt_canon][1] += 1
        if ok:
            correct += 1
            per_class[gt_canon][0] += 1
        else:
            wrong_rows.append((fname, gt_raw, pred_raw, gt_canon, pred_canon))

    # ── Summary ───────────────────────────────────────────────────────────────
    acc = correct / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"Overall accuracy : {correct}/{total}  ({acc:.1%})")
    if skipped_error:
        print(f"Skipped (error)  : {skipped_error}")
    print(f"{'='*60}")

    # Per-class breakdown
    print("\nPer-class accuracy:")
    for cls in sorted(per_class):
        c, t = per_class[cls]
        print(f"  {cls:<35}  {c}/{t}  ({c/t:.1%})" if t else f"  {cls:<35}  -")

    # Wrong predictions
    if wrong_rows and not errors_only:
        n_show = min(show_errors, len(wrong_rows)) if show_errors >= 0 else len(wrong_rows)
        print(f"\nWrong predictions ({len(wrong_rows)} total"
              + (f", showing {n_show}" if n_show < len(wrong_rows) else "") + "):")
        for fname, gt_raw, pred_raw, gt_c, pred_c in wrong_rows[:n_show]:
            print(f"  {fname}")
            print(f"    gt  : {gt_raw!r}  → {gt_c}")
            print(f"    pred: {pred_raw!r}  → {pred_c}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_accuracy",
        description="Compute prediction accuracy from CPathAgent output JSONs.",
    )
    p.add_argument("folder", help="Folder containing result *.json files")
    p.add_argument("--recursive", action="store_true",
                   help="Search subfolders recursively")
    p.add_argument("--errors-only", action="store_true",
                   help="Only list error-status records; skip accuracy report")
    p.add_argument("--show-errors", type=int, default=20, metavar="N",
                   help="Max wrong predictions to print (-1 = all, default: 20)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        raise SystemExit(1)

    records = load_records(folder, args.recursive)
    print(f"[Eval] Loaded {len(records)} records from {folder}")

    evaluate(records, errors_only=args.errors_only, show_errors=args.show_errors)
