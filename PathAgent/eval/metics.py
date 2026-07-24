import os
import json
import hashlib
import difflib
import argparse
from collections import defaultdict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor


# ====== Utility ======
def make_unique_id(long_id, question_text):
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"


def normalize_id(long_id):
    """TCGA-A1-A0SF-01Z-... → TCGA-A1-A0SF"""
    return "-".join(long_id.split("-")[:3])


def normalize_question(q):
    """remove extra spaces + lower"""
    return " ".join(q.lower().strip().split())


DETECTION_LOCALIZE_TEMPLATES = [
    "Is there any metastatic tumor present in this slide? If yes, indicate its approximate location.",
    "Does this slide contain tumor regions? If so, where are they located?",
    "Determine whether tumor is present. If present, provide the approximate location.",
]

_NORMALIZED_TEMPLATES = {normalize_question(t) for t in DETECTION_LOCALIZE_TEMPLATES}


def is_detection_localize(question: str) -> bool:
    return normalize_question(question) in _NORMALIZED_TEMPLATES


def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]
    eval_res = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def acc_of_seq(choices, gt, res):
    if not choices:
        return None
    score = difflib.SequenceMatcher(None, res, gt).quick_ratio()
    for item in choices:
        tmp = difflib.SequenceMatcher(None, res, item).quick_ratio()
        if tmp > score:
            return False
    return True


# ====== Build Type Mapping ======
def build_type_mapping(type_json_path):
    print(f"Loading type mapping from {type_json_path}")
    with open(type_json_path, "r") as f:
        data = json.load(f)

    mapping = {}
    for item in data:
        key = (item["Id"], normalize_question(item["Question"]))
        mapping[key] = item.get("type", None)

    print(f"Loaded {len(mapping)} entries")
    return mapping


# ====== Main Evaluation ======
def evaluate_results(results_dir, use_type=False, type_mapping=None, debug=False):
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return None, None, None, None

    gts = {}
    res = {}

    total_mcq = 0
    correct_mcq = 0

    type_stats = defaultdict(lambda: {"total": 0, "correct": 0}) if use_type else None
    unmatched = []

    file_list = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    for fname in file_list:
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        long_id = data["long_id"]
        question_raw = data["question"]

        # short_id = normalize_id(long_id)
        question = normalize_question(question_raw)

        case_id = make_unique_id(long_id, question_raw)
        
        if isinstance(data.get("ground_truth", ""), dict):
            gt = data.get("ground_truth", "").get("presence", "").strip()
        else:
            gt = data.get("ground_truth", "").strip()
        pred = data.get("pred_answer", "").strip()
        choices = data.get("choices", None)

        

        # ===== type lookup =====
        q_type = None
        if use_type and type_mapping is not None:
            q_type = type_mapping.get((short_id, question), None)

            if q_type is None and debug:
                unmatched.append({
                    "id": short_id,
                    "question": question_raw
                })

        if not is_detection_localize(question_raw):
            continue

        print(f"gt: {gt}, pred: {pred}")

        # ===== NLU =====
        if gt and pred:
            gts[case_id] = [gt]
            res[case_id] = [pred]

        # ===== MCQ =====
        if choices and isinstance(choices, list) and len(choices) > 0:
            total_mcq += 1

            correct = acc_of_seq(choices, gt, pred)
            if correct:
                correct_mcq += 1

            # ===== per-type =====
            if use_type and q_type is not None:
                type_stats[q_type]["total"] += 1
                if correct:
                    type_stats[q_type]["correct"] += 1

    print(f"Collected {len(gts)} valid samples")

    # ===== NLU =====
    nlu_scores = compute_scores(gts, res)

    # ===== Overall =====
    acc_score = correct_mcq / total_mcq if total_mcq > 0 else 0.0
    print(f"\nTotal MCQ: {total_mcq}, Correct: {correct_mcq}, Accuracy = {acc_score:.4f}")

    # ===== Per-Type =====
    if use_type and type_stats:
        print("\n===== Per-Type Accuracy =====")
        for t, stats in sorted(type_stats.items()):
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            print(f"{t:15s} | {correct:4d}/{total:4d} = {acc:.4f}")

    # ===== Debug =====
    if use_type and debug and len(unmatched) > 0:
        print(f"\n⚠️ Unmatched samples: {len(unmatched)}")
        for i, item in enumerate(unmatched[:10]):  # 只印前10筆
            print(f"[{i}] {item['id']} | {item['question']}")

    return nlu_scores, acc_score, gts, res


# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Results")

    parser.add_argument("--results_dir", type=str, required=True)

    parser.add_argument(
        "--use_type",
        action="store_true",
        help="Enable per-type accuracy"
    )

    parser.add_argument(
        "--type_json",
        type=str,
        default=None,
        help="Original dataset with type"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print unmatched type samples"
    )

    args = parser.parse_args()

    print(f"Evaluating: {args.results_dir}")
    print(f"Use type: {args.use_type}")

    type_mapping = None
    if args.use_type:
        if args.type_json is None:
            raise ValueError("❌ --use_type requires --type_json")

        type_mapping = build_type_mapping(args.type_json)

    scores, acc, gts, res = evaluate_results(
        args.results_dir,
        use_type=args.use_type,
        type_mapping=type_mapping,
        debug=args.debug
    )

    if scores:
        print("\nNLU Evaluation Results:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        print(f"\nMCQ Accuracy: {acc:.4f}")