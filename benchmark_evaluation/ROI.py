import os
import json
import math
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from shapely.geometry import Polygon, Point, box
import openslide


# =========================
# Parse XML → polygons
# =========================
def parse_xml_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    for ann in root.findall(".//Annotation"):
        group = ann.attrib.get("PartOfGroup", "").lower()

        if "tumor" not in group and "_0" not in group:
            continue

        coords = []
        for coord in ann.findall(".//Coordinate"):
            x = float(coord.attrib["X"])
            y = float(coord.attrib["Y"])
            coords.append((x, y))

        if len(coords) > 2:
            polygons.append(Polygon(coords))

    return polygons


# =========================
# ROI vs tumor 判斷
# =========================
def evaluate_roi_with_polygon(roi_json_path, polygons, slide_area, use_intersection=True):
    with open(roi_json_path, "r") as f:
        roi_data = json.load(f)

    matched = []
    total_roi_area = 0
    tumor_area = 0
    normal_area = 0

    for roi_name, coords in roi_data.items():
        x1, y1, x2, y2 = coords
        roi_box = box(x1, y1, x2, y2)

        roi_area = abs((x2 - x1) * (y2 - y1))
        total_roi_area += roi_area

        if len(polygons) == 0:
            is_tumor = False
        else:
            if use_intersection:
                is_tumor = any(poly.intersects(roi_box) for poly in polygons)
            else:
                center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                is_tumor = any(poly.contains(center) for poly in polygons)

        if is_tumor:
            tumor_area += roi_area
        else:
            normal_area += roi_area

        matched.append({
            "roi": roi_name,
            "coords": coords,
            "is_tumor": is_tumor
        })

    # ===== 排序 =====
    matched = sorted(matched, key=lambda x: int(x["roi"].split("_")[-1]))

    # ===== 基本統計 =====
    tumor = sum(m["is_tumor"] for m in matched)
    total = len(matched)
    tumor_ratio = tumor / total if total > 0 else 0

    # =========================
    # 🔥 Coverage metrics（重點🔥）
    # =========================

    coverage_ratio = total_roi_area / slide_area if slide_area > 0 else 0
    tumor_coverage = tumor_area / slide_area if slide_area > 0 else 0
    normal_coverage = normal_area / slide_area if slide_area > 0 else 0

    # ⭐ 最重要
    efficiency_area = tumor_coverage - normal_coverage

    # ===== navigation metrics =====
    metrics = compute_metrics(matched, k=10)

    return {
        "matched": matched,

        # 原本
        "tumor_count": tumor,
        "total_valid": total,
        "tumor_ratio": tumor_ratio,

        # 🔥 空間指標
        "coverage_ratio": coverage_ratio,
        "tumor_coverage": tumor_coverage,
        "normal_coverage": normal_coverage,
        "efficiency_area": efficiency_area,

        # navigation
        **metrics
    }


# =========================
# Metrics
# =========================
def compute_mrfh(matched):
    for i, m in enumerate(matched):
        if m["is_tumor"]:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(labels, k):
    dcg = 0.0
    for i in range(min(k, len(labels))):
        dcg += labels[i] / math.log2(i + 2)
    return dcg


def compute_ndcg(matched, k=10):
    labels = [1 if m["is_tumor"] else 0 for m in matched]

    dcg = compute_dcg(labels, k)
    ideal = sorted(labels, reverse=True)
    idcg = compute_dcg(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_metrics(matched, k=10):
    return {
        "MRFH": compute_mrfh(matched),
        f"nDCG@{k}": compute_ndcg(matched, k)
    }


# =========================
# Evaluate single case
# =========================
def evaluate_one_case(case_dir, xml_path, slide_dir):
    roi_json = os.path.join(case_dir, "roi_coords.json")

    if not os.path.exists(roi_json):
        return None

    case_name = os.path.basename(case_dir)
    slide_path = os.path.join(slide_dir, case_name + ".tif")

    if not os.path.exists(slide_path):
        print(f"[Skip] No slide for {case_name}")
        return None

    # 👉 讀 slide size
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    slide_area = width * height

    polygons = parse_xml_polygons(xml_path)

    result = evaluate_roi_with_polygon(
        roi_json,
        polygons,
        slide_area=slide_area,   # 🔥 傳進去
        use_intersection=True
    )

    return result


# =========================
# Evaluate all
# =========================
def evaluate_all(result_root, annotation_dir, slide_dir, output_path):
    results = {}

    all_mrfh = []
    all_ndcg = []
    all_ratio = []
    all_hit = []

    # 🔥 新增
    all_coverage = []
    all_tumor_cov = []
    all_normal_cov = []
    all_eff_area = []

    cases = [d for d in os.listdir(result_root)
             if os.path.isdir(os.path.join(result_root, d))]

    for case in tqdm(cases):
        case_dir = os.path.join(result_root, case)
        xml_path = os.path.join(annotation_dir, case + ".xml")

        if not os.path.exists(xml_path):
            print(f"[Skip] No XML for {case}")
            continue

        output = evaluate_one_case(case_dir, xml_path, slide_dir)

        if output is None:
            continue

        results[case] = output

        all_mrfh.append(output["MRFH"])
        all_ndcg.append(output["nDCG@10"])
        all_ratio.append(output["tumor_ratio"])
        all_hit.append(1 if output["tumor_count"] > 0 else 0)

        # 🔥 coverage
        all_coverage.append(output["coverage_ratio"])
        all_tumor_cov.append(output["tumor_coverage"])
        all_normal_cov.append(output["normal_coverage"])
        all_eff_area.append(output["efficiency_area"])

    # ===== summary =====
    if len(all_mrfh) > 0:
        summary = {
            "mean_MRFH": sum(all_mrfh) / len(all_mrfh),
            "mean_nDCG@10": sum(all_ndcg) / len(all_ndcg),
            "mean_tumor_ratio": sum(all_ratio) / len(all_ratio),
            "hit_rate": sum(all_hit) / len(all_hit),

            # 🔥 空間指標
            "mean_coverage": sum(all_coverage) / len(all_coverage),
            "mean_tumor_coverage": sum(all_tumor_cov) / len(all_tumor_cov),
            "mean_normal_coverage": sum(all_normal_cov) / len(all_normal_cov),
            "mean_efficiency_area": sum(all_eff_area) / len(all_eff_area),
        }
    else:
        summary = {}

    final_output = {
        "summary": summary,
        "cases": results
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n===== Global Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved to {output_path}")


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_root", type=str, required=True,
                        help="path to result/CAMELYON16")

    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="path to xml annotations")

    parser.add_argument("--output", type=str, default="roi_eval.json",
                        help="output json")

    parser.add_argument("--slide_dir", type=str, required=True,
                    help="path to slides (.tif)")

    args = parser.parse_args()

    evaluate_all(
        args.result_root,
        args.annotation_dir,
        args.slide_dir,
        args.output
    )