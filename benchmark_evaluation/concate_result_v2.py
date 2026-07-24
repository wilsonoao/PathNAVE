import os
import json
import argparse
import re
from tqdm import tqdm


def load_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[Skip] {path} load error: {e}")
        return None


# =========================
# Clean logic
# =========================
import re


import re

def extract_question_from_text(text):
    if not text:
        return None

    match = re.search(r"Question:\s*(.*?)\n", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def extract_answer_and_clean(content):
    removed_think = 0

    if not content:
        return "", None, removed_think

    # remove <think>
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        removed_think = 1

    # extract <answer>
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return f"<answer>{answer}</answer>", answer, removed_think

    return "", None, removed_think


def clean_data(data_list):
    removed_lesions = 0
    removed_user = 0
    removed_think = 0
    extracted_answer = 0
    extracted_question = 0

    for item in data_list:

        # ---------- 1. remove lesions ----------
        gt = item.get("ground_truth", {})
        if isinstance(gt, dict) and "lesions" in gt:
            del gt["lesions"]
            removed_lesions += 1

        # ---------- 2. extract question ----------
        if "question" not in item and "chat_history" in item:
            for msg in item["chat_history"]:
                if msg.get("role") == "user":
                    content = msg.get("content", [])

                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") == "text":
                                q = extract_question_from_text(c.get("text", ""))
                                if q:
                                    item["question"] = q
                                    extracted_question += 1
                                    break

                    elif isinstance(content, str):
                        q = extract_question_from_text(content)
                        if q:
                            item["question"] = q
                            extracted_question += 1

                if "question" in item:
                    break

        # ---------- 3. clean chat_history ----------
        if "chat_history" in item:
            new_history = []

            for msg in item["chat_history"]:

                # ❌ remove user
                if msg.get("role") == "user":
                    removed_user += 1
                    continue

                if msg.get("role") == "assistant":
                    content = msg.get("content", "")

                    cleaned_content, answer, think_flag = extract_answer_and_clean(content)
                    removed_think += think_flag

                    msg["content"] = cleaned_content

                    if answer and "pred_answer" not in item:
                        item["pred_answer"] = answer
                        extracted_answer += 1

                new_history.append(msg)

            item["chat_history"] = new_history

        # ---------- 4. final_summary ----------
        if "final_summary" in item:
            cleaned_content, answer, think_flag = extract_answer_and_clean(item["final_summary"])
            removed_think += think_flag

            item["final_summary"] = cleaned_content

            if answer and "pred_answer" not in item:
                item["pred_answer"] = answer
                extracted_answer += 1

        # ---------- 5. initial_response ----------
        if "initial_response" in item:
            cleaned_content, answer, think_flag = extract_answer_and_clean(item["initial_response"])
            removed_think += think_flag

            item["initial_response"] = cleaned_content

            if answer and "pred_answer" not in item:
                item["pred_answer"] = answer
                extracted_answer += 1

    return (
        data_list,
        removed_lesions,
        removed_user,
        removed_think,
        extracted_answer,
        extracted_question
    )


# =========================
# Load all json
# =========================
def collect_all_data(input_dir):
    all_data = []

    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".json"):
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


# =========================
# Task type classification
# =========================
MULTI_LESION_KEYWORDS = [
    "more than one",
    "multiple",
    "multiple foci",
    "multiple distinct",
    "multiple tumor",
]

def classify_task_type(item):
    question = (item.get("question") or "").lower()
    for kw in MULTI_LESION_KEYWORDS:
        if kw in question:
            return "MultiLesion"
    return "DetectionLocalization"


def classify_and_add_task_type(data_list):
    for item in data_list:
        item["task_type"] = classify_task_type(item)
    return data_list


SLIM_FIELDS = ("case_id", "question", "pred_answer", "ground_truth")


def to_slim(items):
    return [{k: item.get(k) for k in SLIM_FIELDS} for item in items]


def save_by_task_type(data, output_dir, split_size):
    from collections import defaultdict
    groups = defaultdict(list)
    for item in data:
        groups[item["task_type"]].append(item)

    for task_type, items in groups.items():
        print(f"\nTask type '{task_type}': {len(items)} samples")
        if split_size is not None:
            total = len(items)
            num_splits = (total + split_size - 1) // split_size
            for i in range(num_splits):
                chunk = items[i * split_size: (i + 1) * split_size]
                out_path = os.path.join(output_dir, f"{task_type}_{i}.json")
                save_json(chunk, out_path)
                print(f"  Saved: {out_path} ({len(chunk)} samples)")
                slim_path = os.path.join(output_dir, f"{task_type}_{i}_slim.json")
                save_json(to_slim(chunk), slim_path)
                print(f"  Saved: {slim_path} (slim)")
        else:
            out_path = os.path.join(output_dir, f"{task_type}.json")
            save_json(items, out_path)
            print(f"  Saved: {out_path} ({len(items)} samples)")
            slim_path = os.path.join(output_dir, f"{task_type}_slim.json")
            save_json(to_slim(items), slim_path)
            print(f"  Saved: {slim_path} (slim)")


# =========================
# Ground truth
# =========================
def load_ground_truth(gt_path):
    """Build lookup: {(case_id, task_type): answer} from roi_qa_dataset.json."""
    with open(gt_path, "r") as f:
        records = json.load(f)

    lookup = {}
    for rec in records:
        key = (rec["Id"], rec["type"])
        answer = rec["Answer"]
        # strip lesions to keep ground_truth compact
        if isinstance(answer, dict) and "lesions" in answer:
            answer = {k: v for k, v in answer.items() if k != "lesions"}
        lookup[key] = answer

    return lookup


def attach_ground_truth(data_list, gt_path):
    lookup = load_ground_truth(gt_path)
    matched = 0
    missing = 0

    for item in data_list:
        key = (item.get("case_id"), item.get("task_type"))
        if key in lookup:
            item["ground_truth"] = lookup[key]
            matched += 1
        else:
            missing += 1

    print(f"  Ground truth matched: {matched}, missing: {missing}")
    return data_list


def split_and_save(data, output_dir, base_name, split_size):
    os.makedirs(output_dir, exist_ok=True)

    total = len(data)
    num_splits = (total + split_size - 1) // split_size

    for i in range(num_splits):
        chunk = data[i * split_size: (i + 1) * split_size]
        out_path = os.path.join(output_dir, f"{base_name}_{i}.json")
        save_json(chunk, out_path)
        print(f"Saved: {out_path} ({len(chunk)} samples)")


# =========================
# Main
# =========================
def main(args):
    print("Loading data...")
    all_data = collect_all_data(args.input_dir)
    print(f"Total loaded: {len(all_data)}")

    print("Cleaning data...")
    cleaned, r_lesions, r_user, r_think, r_ans, r_q = clean_data(all_data)

    print(f"Removed lesions: {r_lesions}")
    print(f"Removed user msgs: {r_user}")
    print(f"Removed think: {r_think}")
    print(f"Extracted answers: {r_ans}")
    print(f"Extracted questions: {r_q}")

    print("Classifying task types...")
    classify_and_add_task_type(cleaned)
    from collections import Counter
    type_counts = Counter(item["task_type"] for item in cleaned)
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    if args.gt_path:
        print("Attaching ground truth...")
        attach_ground_truth(cleaned, args.gt_path)

    # print(cleaned)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nSaving merged output...")
    if args.max_per_file is not None:
        split_and_save(
            cleaned,
            args.output_dir,
            args.base_name,
            args.max_per_file
        )
    else:
        out_path = os.path.join(args.output_dir, f"{args.base_name}.json")
        save_json(cleaned, out_path)
        print(f"Saved: {out_path}")

    print("\nSaving by task type...")
    save_by_task_type(cleaned, args.output_dir, args.max_per_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--base_name", type=str, default="merged")
    parser.add_argument("--max_per_file", type=int, default=None,
                        help="Max number of cases per output JSON file.")
    parser.add_argument("--split_size", type=int, default=None,
                        help="Alias for --max_per_file (deprecated, use --max_per_file).")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to roi_qa_dataset.json for attaching ground truth.")

    args = parser.parse_args()

    # --max_per_file takes precedence; fall back to --split_size for compatibility
    if args.max_per_file is None and args.split_size is not None:
        args.max_per_file = args.split_size

    main(args)