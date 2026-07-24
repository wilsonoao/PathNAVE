import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

# ------------------------------------------------------------------ #
#  Question-type classification                                        #
# ------------------------------------------------------------------ #
DETECTION_LOCALIZE_TEMPLATES = [
    "Is there any metastatic tumor present in this slide? If yes, indicate its approximate location.",
    "Does this slide contain tumor regions? If so, where are they located?",
    "Determine whether tumor is present. If present, provide the approximate location.",
]

MULTI_LESION_TEMPLATES = [
    "Are there multiple distinct tumor regions in this slide?",
    "Does this slide contain more than one tumor region?",
    "Is there evidence of multiple tumor foci in this slide?",
]

_DL_SET = {q.strip().lower() for q in DETECTION_LOCALIZE_TEMPLATES}
_ML_SET = {q.strip().lower() for q in MULTI_LESION_TEMPLATES}


def classify_question(item: dict) -> str:
    """
    Return 'DetectionLocalization', 'MultiLesion', or 'Unknown'.
    Search order:
      1. top-level question_type
      2. ground_truth.question_type
      3. top-level question text match
      4. ground_truth.question text match
    """
    gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}

    for qt in (item.get("question_type", ""), gt.get("question_type", "")):
        if qt in ("DetectionLocalization", "MultiLesion"):
            return qt

    for q_raw in (item.get("question", ""), gt.get("question", "")):
        q = q_raw.strip().lower()
        if q in _DL_SET:
            return "DetectionLocalization"
        if q in _ML_SET:
            return "MultiLesion"

    return "Unknown"


def get_presence(item: dict) -> str:
    """
    Return normalised ground-truth presence label: 'yes' or 'no'.
    Search order:
      1. top-level gt_presence
      2. top-level presence
      3. ground_truth (string directly, e.g. "yes" / "no")
      4. ground_truth.gt_presence
      5. ground_truth.presence
    """
    gt_raw = item.get("ground_truth")

    candidates = [
        item.get("gt_presence"),
        item.get("presence"),
    ]

    # ground_truth 可能直接是字串 "yes"/"no"
    if isinstance(gt_raw, str):
        candidates.append(gt_raw)
    elif isinstance(gt_raw, dict):
        candidates.append(gt_raw.get("gt_presence"))
        candidates.append(gt_raw.get("presence"))

    for raw in candidates:
        if raw is not None:
            return str(raw).strip().lower()

    return "unknown"


# ------------------------------------------------------------------ #
#  Field extraction                                                    #
# ------------------------------------------------------------------ #
def extract_fields(item: dict) -> dict:
    """
    Keep only: question, ground_truth, thinking_steps.
    thinking_steps is collected from process[*].evaluation_result.thinking_steps
    and joined in order (one entry per attempt).
    """
    steps = []
    for attempt in item.get("process", []):
        er = attempt.get("evaluation_result") or {}
        ts = er.get("thinking_steps")
        if ts:
            steps.append(ts)

    gt_raw = item.get("ground_truth")
    if isinstance(gt_raw, dict):
        gt_out = gt_raw.get("presence") or gt_raw.get("gt_presence") or gt_raw
    else:
        gt_out = gt_raw

    return {
        "question": item.get("question", ""),
        "ground_truth": gt_out,
        "thinking_steps": steps if len(steps) > 1 else (steps[0] if steps else None),
    }


# ------------------------------------------------------------------ #
#  I/O helpers                                                         #
# ------------------------------------------------------------------ #
def load_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[Skip] {path} load error: {e}")
        return None


def remove_lesions_key(data_list):
    """如果 ground_truth 有 lesions，就刪掉該 key"""
    modified = 0
    for item in data_list:
        gt = item.get("ground_truth", {})
        if isinstance(gt, dict) and "lesions" in gt:
            del gt["lesions"]
            modified += 1
    return data_list, modified


def collect_all_data(input_dir):
    all_data = []
    for fname in tqdm(sorted(os.listdir(input_dir))):
        if not fname.endswith(".json"):
            continue
        # Skip companion lesion files (e.g. *_lesions.json) — they only contain
        # gt_lesions / visited_rois and have no question/question_type field.
        if fname.endswith("_lesions.json"):
            continue
        fpath = os.path.join(input_dir, fname)
        data = load_json(fpath)
        if data is None:
            continue
        if isinstance(data, dict):
            data = [data]
        all_data.extend(data)
    return all_data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def split_and_save(data, output_dir, base_name, split_size):
    os.makedirs(output_dir, exist_ok=True)
    total = len(data)
    num_splits = (total + split_size - 1) // split_size
    for i in range(num_splits):
        chunk = data[i * split_size: (i + 1) * split_size]
        out_path = os.path.join(output_dir, f"{base_name}_{i}.json")
        save_json(chunk, out_path)
        print(f"  Saved: {out_path} ({len(chunk)} samples)")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main(args):
    print("Loading data...")
    all_data = collect_all_data(args.input_dir)
    print(f"Total loaded: {len(all_data)}")

    print("Removing lesions key from ground_truth...")
    all_data, modified = remove_lesions_key(all_data)
    print(f"Modified samples (lesions removed): {modified}")

    # ---- Classify into buckets: {question_type: {presence: [items]}} ----
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    unknown_items = []

    for item in all_data:
        qt = classify_question(item)
        presence = get_presence(item)
        extracted = extract_fields(item)
        if qt == "Unknown":
            unknown_items.append(extracted)
            continue
        buckets[qt][presence].append(extracted)

    # ---- Report counts ----
    print("\n=== Split summary ===")
    for qt in sorted(buckets.keys()):
        for presence in sorted(buckets[qt].keys()):
            n = len(buckets[qt][presence])
            max_splits = n  # max: 1 sample per file
            print(f"  {qt} / gt_presence={presence}: {n} samples  (max {max_splits} splits)")
    if unknown_items:
        print(f"  Unknown question type: {len(unknown_items)} samples (saved separately)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Save each bucket ----
    for qt in sorted(buckets.keys()):
        for presence in sorted(buckets[qt].keys()):
            items = buckets[qt][presence]
            base = f"{qt}_{presence}"

            if args.split_size is not None:
                split_and_save(items, args.output_dir, base, args.split_size)
            else:
                out_path = os.path.join(args.output_dir, f"{base}.json")
                save_json(items, out_path)
                print(f"  Saved: {out_path} ({len(items)} samples)")

    # ---- Save unknowns if any ----
    if unknown_items:
        out_path = os.path.join(args.output_dir, "Unknown.json")
        save_json(unknown_items, out_path)
        print(f"  Saved: {out_path} ({len(unknown_items)} samples)")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect SlideSeek QA results and split by question type + gt_presence."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="資料夾路徑（含各 slide 的 .json 結果）")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="輸出資料夾")
    parser.add_argument("--split_size", type=int, default=None,
                        help="每個 JSON 最多幾筆（不填=不切，每個 bucket 存成一個檔案）")
    args = parser.parse_args()
    main(args)
