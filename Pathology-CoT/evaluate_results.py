#!/usr/bin/env python3
"""
Evaluate analysis_results JSONs in a given folder and report accuracy.
Usage:
    python evaluate_results.py -d result/TCGA_BRCA/test_total
"""
import json
import os
import glob
import argparse


def _rank(v) -> int:
    if v is True:
        return 2
    if v is False:
        return 1
    return 0


def evaluate(folder: str, prune_duplicates: bool = False):
    files = glob.glob(os.path.join(folder, "*.json"))
    if not files:
        print(f"No JSON files found in: {folder}")
        return

    # Group by {slide_id}_{question} (strip last _{hash} suffix); track all (value, path) per key.
    groups: dict[str, list[tuple[object, str]]] = {}

    for path in files:
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[Parse error] {path}: {e}")
                continue

        base = os.path.splitext(os.path.basename(path))[0]
        key = base.split("_")[0]
        value = data.get("correct", None)
        groups.setdefault(key, []).append((value, path))

    if prune_duplicates:
        deleted = 0
        for key, entries in groups.items():
            if len(entries) <= 1:
                continue
            # Keep the entry with the highest-ranked value; break ties by keeping first found.
            best_entry = max(entries, key=lambda e: _rank(e[0]))
            for val, path in entries:
                if path != best_entry[1]:
                    os.remove(path)
                    deleted += 1
        if deleted:
            print(f"Pruned {deleted} duplicate file(s), kept best per case.")

    # Best value per group for stats
    best_values = {k: max(entries, key=lambda e: _rank(e[0]))[0] for k, entries in groups.items()}

    total = len(best_values)
    correct = sum(1 for v in best_values.values() if v is True)
    no_key = sum(1 for v in best_values.values() if v is None)
    evaluated = total - no_key
    wrong = evaluated - correct

    print(f"\n=== Evaluation Results ===")
    print(f"Folder           : {folder}")
    print(f"Total (deduplicated) : {total}")
    print(f"correct = True   : {correct}")
    print(f"correct = False  : {wrong}")
    print(f"missing 'correct': {no_key}")
    if evaluated > 0:
        print(f"Accuracy         : {correct / evaluated:.2%}  ({correct}/{evaluated})")
    else:
        print("Accuracy         : N/A (no evaluable results)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate correctness of analysis result JSONs.")
    parser.add_argument(
        "-d", "--dir",
        default="result/TCGA_BRCA/test_total",
        help="Folder containing result JSON files (default: result/TCGA_BRCA/test_total)"
    )
    parser.add_argument(
        "--prune-duplicates", action="store_true",
        help="Delete duplicate case files from disk, keeping only the best result per case"
    )
    args = parser.parse_args()
    evaluate(args.dir, prune_duplicates=args.prune_duplicates)
