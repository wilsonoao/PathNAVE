import os
import re
import json
import argparse
from pathlib import Path


def extract_paren_content(text: str):
    """Return the content inside the first pair of parentheses, or None."""
    m = re.search(r"\(([^)]+)\)", text)
    return m.group(1).strip() if m else None


def strip_paren(text: str) -> str:
    """Remove parenthesised portion and collapse whitespace."""
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def is_correct(gt: str, pred: str) -> bool:
    gt = gt.strip()
    pred = pred.strip()

    # 1. Exact match (case-insensitive)
    if gt.lower() == pred.lower():
        return True

    # 2. pred matches the abbreviation / content inside GT's parentheses
    gt_paren = extract_paren_content(gt)
    if gt_paren and gt_paren.lower() == pred.lower():
        return True

    # 3. pred matches the part of GT *outside* the parentheses
    gt_no_paren = strip_paren(gt)
    if gt_no_paren.lower() == pred.lower():
        return True

    # 4. Symmetric: GT matches content inside pred's parentheses
    pred_paren = extract_paren_content(pred)
    if pred_paren and pred_paren.lower() == gt.lower():
        return True

    pred_no_paren = strip_paren(pred)
    if pred_no_paren.lower() == gt.lower():
        return True

    return False


def evaluate_folder(folder: str, verbose: bool = False) -> dict:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {folder}")
        return {}

    total = correct = skipped = 0
    wrong_cases = []

    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Cannot parse {fpath}, skipping.")
                skipped += 1
                continue

        # ground_truth may be a dict (detection tasks) or a plain string
        gt_raw = data.get("ground_truth", "")
        if isinstance(gt_raw, dict):
            gt = gt_raw.get("presence", "")
        else:
            gt = str(gt_raw)

        pred = str(data.get("pred_answer", ""))

        if not gt or not pred:
            skipped += 1
            continue

        total += 1
        ok = is_correct(gt, pred)
        if ok:
            correct += 1
        else:
            wrong_cases.append({
                "file": str(fpath.relative_to(folder)),
                "ground_truth": gt,
                "pred_answer": pred,
            })

        if "correct" not in data:
            data["correct"] = ok
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        if verbose:
            status = "OK" if ok else "WRONG"
            print(f"[{status}] {fpath.name}  GT={gt!r}  PRED={pred!r}")

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Folder   : {folder}")
    print(f"Total    : {total}")
    print(f"Correct  : {correct}")
    print(f"Wrong    : {total - correct}")
    print(f"Skipped  : {skipped}")
    print(f"Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # if wrong_cases:
    #     print("\nWrong cases:")
    #     for i, case in enumerate(wrong_cases, 1):
    #         print(f"  [{i}] {case['file']}")
    #         print(f"       GT  : {case['ground_truth']}")
    #         print(f"       PRED: {case['pred_answer']}")

    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "skipped": skipped,
        "accuracy": accuracy,
        "wrong_cases": wrong_cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ground_truth vs pred_answer in JSON result files."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing result JSON files (searched recursively).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print every file's result.",
    )
    args = parser.parse_args()

    evaluate_folder(args.folder, verbose=args.verbose)
