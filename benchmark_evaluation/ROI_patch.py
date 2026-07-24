import os
import json
import argparse


def get_case_id(filename):
    # test_002_189d8e80.json → test_002
    return "_".join(filename.split("_")[:2])


def build_mask_lookup(mask_data):
    mask_lookup = {}
    for k, v in mask_data.items():
        filename = os.path.basename(k)
        mask_lookup[filename] = v["is_tumor"]
    return mask_lookup

def compute_mrfh(matched):
    for i, m in enumerate(matched):
        if m["is_tumor"]:
            return 1.0 / (i + 1)
    return 0.0


import math

def compute_dcg(labels, k):
    dcg = 0.0
    for i in range(min(k, len(labels))):
        rel = labels[i]
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(matched, k=10):
    labels = [1 if m["is_tumor"] else 0 for m in matched]

    dcg = compute_dcg(labels, k)

    # ideal（全部 tumor 排前面）
    ideal_labels = sorted(labels, reverse=True)
    idcg = compute_dcg(ideal_labels, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def compute_metrics(matched, k=10):
    return {
        "MRFH": compute_mrfh(matched),
        f"nDCG@{k}": compute_ndcg(matched, k)
    }


def extract_patches(result_data):
    patches = []

    for step in result_data.get("process", []):
        patches.extend(step.get("evaluated_patches_this_round", []))

        zoom_patch = step.get("selected_zoom_patch")
        if zoom_patch:
            patches.append(zoom_patch + ".jpg")

    # ✅ 保留順序去重
    seen = set()
    ordered = []
    for p in patches:
        if p not in seen:
            ordered.append(p)
            seen.add(p)

    return ordered


def evaluate_one(result_path, mask_path):
    with open(result_path, "r") as f:
        result_data = json.load(f)

    with open(mask_path, "r") as f:
        mask_data = json.load(f)

    mask_lookup = build_mask_lookup(mask_data)
    patches = extract_patches(result_data)

    matched = []
    for patch in patches:
        npy_name = os.path.splitext(patch)[0] + ".jpg"
        is_tumor = mask_lookup.get(npy_name, None)

        matched.append({
            "patch": patch,
            "jpg": npy_name,
            "is_tumor": is_tumor
        })

    # ===== 原本的統計（保留）=====
    valid = [m for m in matched if m["is_tumor"] is not None]
    tumor = sum(m["is_tumor"] for m in valid)
    total = len(valid)
    tumor_ratio = tumor / total if total > 0 else 0

    # ===== 新 metrics =====
    metrics = compute_metrics(matched, k=10)

    # ===== 新增：region usage =====
    total_regions = len(mask_lookup)  # 所有可用 patch（region）
    used_regions = len(valid)         # 有 match 到 mask 的 patch

    region_usage_ratio = used_regions / total_regions if total_regions > 0 else 0

    return {
        "matched": matched,

        # 🔹 保留原本統計
        "tumor_count": tumor,
        "total_valid": total,
        "tumor_ratio": tumor_ratio,

        "region_usage_ratio": region_usage_ratio,

        # 🔹 新增 navigation metrics
        **metrics
    }


def main(args):
    results = {}

    all_mrfh = []
    all_ndcg = []
    all_ratio = []
    all_usage = []
    all_hit = []

    for fname in os.listdir(args.result_dir):
        if not fname.endswith(".json"):
            continue

        case_id = get_case_id(fname)

        result_path = os.path.join(args.result_dir, fname)
        mask_path = os.path.join(args.mask_dir, f"{case_id}.json")

        if not os.path.exists(mask_path):
            if args.verbose:
                print(f"[WARNING] mask not found: {mask_path}")
            continue

        output = evaluate_one(result_path, mask_path)
        results[fname] = output
        all_mrfh.append(output["MRFH"])
        all_ndcg.append(output["nDCG@10"])
        all_ratio.append(output["tumor_ratio"])
        all_hit.append(1 if output["tumor_count"] > 0 else 0)
        all_usage.append(output["region_usage_ratio"])

        if args.verbose:
            print(f"\n=== {fname} ===")
            print(f"tumor: {output['tumor_count']}/{output['total_valid']}")
            print(f"ratio: {output['tumor_ratio']:.4f}")
            print(f"MRFH: {output['MRFH']:.4f}")
            print(f"nDCG@10: {output['nDCG@10']:.4f}")
            


    if len(all_mrfh) > 0:
        mean_mrfh = sum(all_mrfh) / len(all_mrfh)
        mean_ndcg = sum(all_ndcg) / len(all_ndcg)
        mean_ratio = sum(all_ratio) / len(all_ratio)
        hit_rate = sum(all_hit) / len(all_hit)
        mean_usage = sum(all_usage) / len(all_usage)

        print("\n===== Global Summary =====")
        print(f"Mean MRFH: {mean_mrfh:.4f}")
        print(f"Mean nDCG@10: {mean_ndcg:.4f}")
        print(f"Mean Tumor Ratio: {mean_ratio:.4f}")
        print(f"Hit Rate: {hit_rate:.4f}")
        print(f"Mean Region Usage: {mean_usage:.4f}")


    summary = {
        "mean_MRFH": mean_mrfh,
        "mean_nDCG@10": mean_ndcg,
        "mean_tumor_ratio": mean_ratio,
        "mean_region_usage": mean_usage,
        "hit_rate": hit_rate
    }

    final_output = {
        "summary": summary
    }

    with open(args.output_summary, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved to {args.output}")

    # ===== save =====
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, required=True,
                        help="path to results folder")

    parser.add_argument("--mask_dir", type=str, required=True,
                        help="path to mask_tumor folder")

    parser.add_argument("--output", type=str, default=None,
                        help="save output json")

    parser.add_argument("--output_summary", type=str, default=None,
                        help="save output json")    

    parser.add_argument("--verbose", action="store_true",
                        help="print details")

    args = parser.parse_args()
    main(args)