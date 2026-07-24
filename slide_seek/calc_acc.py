import json
import re
from pathlib import Path
from typing import Dict, List


# ── Alias table ──────────────────────────────────────────────────────────────
# Each entry: canonical key (lowercase) → list of accepted surface forms (lowercase)
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
    "present": [
        "present",
        "positive",
        "yes",
        "true",
        "detected",
        "found",
    ],
    "absent": [
        "absent",
        "negative",
        "no",
        "false",
        "not detected",
        "not found",
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


# Words that negate or deny a finding
_NEGATIVE_WORDS = {"no", "not", "none", "negative", "absent", "without", "non", "free", "never"}

# Tumor-related words that are NOT inherently malignant
_TUMOR_WORDS = {
    "tumor", "tumour", "mass", "lesion", "neoplasm", "nodule",
    "growth", "metastasis", "metastatic",
}

# Words that imply malignancy → always "present"
_MALIGNANT_WORDS = {
    "malignant", "malignancy", "cancer", "cancerous",
    "carcinoma", "sarcoma", "adenocarcinoma", "squamous",
}


def _detect_presence(text: str) -> str:
    """Map free-text prediction to 'present' or 'absent'.

    Rules (in priority order):
      1. Exact alias match → use that canonical label
      2. 'yes' → present
      3. Malignant word → present
      4. Tumor word + negative word → absent  (e.g. "no tumor", "tumor absent")
      5. Tumor word (no negation) → present   (e.g. "tumor found")
      6. 'present' + negative word → absent   (e.g. "no present tumor")
      7. 'present' (no negation) → present
      8. Fall back to alias canonicalisation
    """
    t = text.strip().lower()
    words = set(re.findall(r"\w+", t))

    exact = _SURFACE_TO_CANONICAL.get(t)
    if exact in ("present", "absent"):
        return exact

    has_neg = bool(words & _NEGATIVE_WORDS)
    has_tumor = bool(words & _TUMOR_WORDS)
    has_malignant = bool(words & _MALIGNANT_WORDS)

    if "yes" in words:
        return "present"
    if has_malignant:
        return "present"
    if has_tumor and has_neg:
        return "absent"
    if has_tumor:
        return "present"
    if "present" in words and has_neg:
        return "absent"
    if "present" in words:
        return "present"
    return canonicalise(text)


def _normalise_presence(canon: str) -> str:
    """Map absent→normal since both mean no tumour."""
    return "normal" if canon == "absent" else canon


def is_correct(predicted: str, gt: str) -> bool:
    canon_gt = _normalise_presence(canonicalise(gt))
    use_presence = canon_gt in ("present", "normal")
    canon_pred = _normalise_presence(_detect_presence(predicted) if use_presence else canonicalise(predicted))
    print(f"gt:   {gt} → {canon_gt}")
    print(f"pred: {predicted} → {canon_pred}")
    print("==============")
    return canon_pred == canon_gt


def calc_acc(json_files: list[Path], overwrite: bool = False) -> dict:
    total = 0
    correct = 0
    missing = 0
    details = []

    for f in sorted(json_files):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[WARN] Could not parse {f}: {e}")
            continue

        # Support both single-record dicts and list-of-records (e.g. merged files)
        records = data if isinstance(data, list) else [data]

        file_modified = False

        for record in records:
            if not isinstance(record, dict):
                continue

            pred = record.get("predicted_answer_text")
            gt = record.get("gt_presence")

            if pred is None or gt is None:
                missing += 1
                continue

            total += 1

            if "correct" in record and not overwrite:
                match = record["correct"]
            else:
                match = is_correct(pred, gt)
                record["correct"] = match
                file_modified = True

            if match:
                correct += 1

            details.append({
                "file": f.name,
                "slide_id": record.get("slide_id", ""),
                "predicted": pred,
                "gt": gt,
                "correct": match,
            })

        if file_modified:
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    acc = correct / total if total > 0 else 0.0
    return {"total": total, "correct": correct, "missing_fields": missing, "accuracy": acc, "details": details}


def find_json_files(root: Path) -> list[Path]:
    return [
        f for f in root.rglob("*.json")
        if not f.stem.endswith("_lesions")
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate presence accuracy from JSON result files.")
    parser.add_argument("root", nargs="?", default=".", help="Root directory to search for JSON files (default: current dir)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-file results")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Recompute and overwrite existing 'correct' fields")
    args = parser.parse_args()

    root = Path(args.root)
    files = find_json_files(root)
    print(f"Found {len(files)} JSON files (excluding *_lesions.json) under '{root}'")

    if not files:
        print("No files found.")
        exit(0)

    result = calc_acc(files, overwrite=args.overwrite)

    if args.verbose:
        print(f"\n{'FILE':<50} {'PRED':<35} {'GT':<35} {'OK'}")
        print("-" * 130)
        for d in result["details"]:
            tick = "Y" if d["correct"] else "N"
            print(f"{d['file']:<50} {d['predicted']:<35} {d['gt']:<35} {tick}")

    print(f"\n=== Results ===")
    print(f"Total samples : {result['total']}")
    print(f"Correct       : {result['correct']}")
    print(f"Missing fields: {result['missing_fields']}")
    print(f"Accuracy      : {result['accuracy']:.4f}  ({result['accuracy']*100:.2f}%)")
