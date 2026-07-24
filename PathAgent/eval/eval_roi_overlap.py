"""
Evaluate overlap between predicted ROI bounding boxes and ground truth tumor polygons.

GT CSV format: X_base, Y_base polygon points; first == last point closes each polygon.
ROI JSON format: {"roi_1": [x1, y1, x2, y2], ...}

Usage (single folder):
    python eval_roi_overlap.py result_json_dir --gt-dir gt_dir

Usage (multiple folders):
    python eval_roi_overlap.py result_dir1 result_dir2 --gt-dir gt_dir1 gt_dir2

Each result JSON must have a 'process' key. evaluated_patches_this_round entries are
converted to bounding boxes at BASE_ZOOM (40x) using:
    pixels_at_40x = patch_size * (40 / zoom_level)
zoom_level absent or 'Uncertain' → defaults to 40x.
"""

import os
import csv
import glob
import json
import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union


DEFAULT_GT_DIRS = ["/work/data/TCGA-LUSC-FS/Tumor_annotation/LUSC"]
BASE_ZOOM = 40  # reference magnification for all coordinate/pixel calculations

HIT_THRESHOLD_PATH = "/work/Agent_benchmark/Analyze/hit_threshold.json"


def _load_hit_threshold_cache() -> dict:
    """Load cached hit thresholds from JSON. Returns empty dict on error or if file absent."""
    try:
        if os.path.exists(HIT_THRESHOLD_PATH):
            with open(HIT_THRESHOLD_PATH) as f:
                return json.load(f)
    except Exception as e:
        print(f"[hit_threshold] load error: {e}")
    return {}


def _save_hit_threshold_cache(cache: dict) -> None:
    """Persist hit thresholds to JSON, creating parent dirs as needed."""
    try:
        os.makedirs(os.path.dirname(HIT_THRESHOLD_PATH), exist_ok=True)
        with open(HIT_THRESHOLD_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[hit_threshold] save error: {e}")


def _make_size_key(w: float, h: float) -> str:
    return f"{int(round(w))}x{int(round(h))}"


def _make_mag_key(mag: float) -> str:
    return f"{int(round(mag))}x"


def _compute_random_overlap_threshold(
    gt_union,
    roi_w: float,
    roi_h: float,
    slide_bounds: tuple[float, float, float, float],
    n_samples: int = 1000,
    percentile: float = 95.0,
) -> tuple[float, dict]:
    """Sample N random ROIs within slide_bounds; return (Nth-percentile overlap ratio, debug stats)."""
    x_min, y_min, x_max, y_max = slide_bounds
    sample_x_max = x_max - roi_w
    sample_y_max = y_max - roi_h
    if sample_x_max <= x_min or sample_y_max <= y_min:
        return 0.0, {}

    rng = np.random.default_rng(42)
    xs = rng.uniform(x_min, sample_x_max, n_samples)
    ys = rng.uniform(y_min, sample_y_max, n_samples)

    ratios = []
    for x, y in zip(xs, ys):
        roi_poly = box(x, y, x + roi_w, y + roi_h)
        roi_area = roi_poly.area
        if roi_area <= 0:
            ratios.append(0.0)
            continue
        if gt_union is not None and not gt_union.is_empty:
            ratios.append(roi_poly.intersection(gt_union).area / roi_area)
        else:
            ratios.append(0.0)

    if not ratios:
        return 0.0, {}

    arr = np.array(ratios)
    threshold = float(np.percentile(arr, percentile))
    debug = {
        "n_samples": int(len(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "q95": float(np.percentile(arr, 95)),
        "q97_5": float(np.percentile(arr, 97.5)),
        "q99": float(np.percentile(arr, 99)),
        "q99_5": float(np.percentile(arr, 99.5)),
        "max": float(np.max(arr)),
        "num_eq_1": int(np.sum(arr >= 0.999999)),
        "ratio_eq_1": float(np.mean(arr >= 0.999999)),
    }
    return threshold, debug


def _get_hit_threshold(
    cache: dict,
    case_id: str,
    mag_str: str,
    size_str: str,
    gt_union,
    slide_bounds: tuple[float, float, float, float],
    n_samples: int = 1000,
) -> tuple[float, bool, dict]:
    """Return (threshold, was_newly_computed, debug). Uses cache; computes and caches on miss."""
    try:
        entry = cache[case_id][mag_str][size_str]
        # Backward compat: old cache entries stored a plain float
        if isinstance(entry, dict):
            return entry["threshold"], False, entry.get("debug", {})
        return float(entry), False, {}
    except KeyError:
        pass
    w, h = (float(v) for v in size_str.split("x"))
    threshold, debug = _compute_random_overlap_threshold(gt_union, w, h, slide_bounds, n_samples)
    cache.setdefault(case_id, {}).setdefault(mag_str, {})[size_str] = {
        "threshold": threshold,
        "debug": debug,
    }
    return threshold, True, debug


def _update_case_debug(
    cache: dict,
    case_id: str,
    gt_union,
    slide_bounds: tuple[float, float, float, float],
) -> None:
    """Store GT/WSI ratio debug info in cache at case level (under '_debug' key)."""
    if "_debug" in cache.get(case_id, {}):
        return
    x_min, y_min, x_max, y_max = slide_bounds
    slide_area = (x_max - x_min) * (y_max - y_min)
    gt_area = gt_union.area if gt_union and not gt_union.is_empty else 0.0
    cache.setdefault(case_id, {})["_debug"] = {
        "gt_area": gt_area,
        "slide_area": slide_area,
        "gt_ratio": gt_area / slide_area if slide_area > 0 else None,
        "slide_bounds": list(slide_bounds),
    }


def _bonferroni_threshold(debug: dict, k: int) -> float | None:
    """Interpolate Bonferroni-corrected threshold (α=0.05/k) from stored debug percentiles.

    With k ROIs per case the per-test level is α/k, so the threshold is the
    (1 - 0.05/k)*100th percentile of the null overlap distribution.
    """
    if not debug or k <= 0:
        return None
    p = (1.0 - 0.05 / k) * 100.0
    knots = [
        (0.0, "min"), (25.0, "q25"), (50.0, "median"), (75.0, "q75"),
        (90.0, "q90"), (95.0, "q95"), (97.5, "q97_5"),
        (99.0, "q99"), (99.5, "q99_5"), (100.0, "max"),
    ]
    pts = [(pct, debug[key]) for pct, key in knots if key in debug]
    if not pts:
        return None
    p = max(pts[0][0], min(pts[-1][0], p))
    for i in range(len(pts) - 1):
        p0, v0 = pts[i]
        p1, v1 = pts[i + 1]
        if p0 <= p <= p1:
            if p1 == p0:
                return v0
            return v0 + (p - p0) / (p1 - p0) * (v1 - v0)
    return pts[-1][1]


def _slide_bounds_fallback(
    gt_union,
    total_size: int | None = None,
    slide_w: float | None = None,
    slide_h: float | None = None,
) -> tuple[float, float, float, float]:
    """Get level-0 slide bounds for random ROI sampling.

    Priority: explicit dims > sqrt(total_size) square approx > GT polygon bounds.
    """
    if slide_w and slide_h and slide_w > 0 and slide_h > 0:
        return (0.0, 0.0, float(slide_w), float(slide_h))
    if total_size and total_size > 0:
        # FALLBACK: approximates a square slide from total area; inaccurate for non-square slides
        side = total_size ** 0.5
        return (0.0, 0.0, side, side)
    if gt_union is not None and not gt_union.is_empty:
        # FALLBACK: uses GT annotation bounds as proxy; underestimates true slide extent
        return gt_union.bounds
    return (0.0, 0.0, 100000.0, 100000.0)


def extract_patches_as_rois(
    result_json_path: str,
    patch_size: int = 4096,
    base_zoom: int = BASE_ZOOM,
) -> dict[str, list]:
    """Convert evaluated_patches_this_round in a result JSON to ROI boxes at base_zoom.

    For each step in process:
      - zoom_level defaults to base_zoom when absent, "Uncertain", or non-numeric.
      - pixels_at_base_zoom = patch_size * (base_zoom / zoom_level)
      - bbox = [x, y, x + pixels_at_base_zoom, y + pixels_at_base_zoom]

    Returns {roi_name: [x1, y1, x2, y2]} where coordinates are at base_zoom resolution.
    """
    with open(result_json_path) as f:
        data = json.load(f)

    rois: dict[str, list] = {}
    roi_metas: dict[str, dict] = {}
    for step in data.get("process", []):
        attempt = step.get("attempt", "?")
        zoom_raw = step.get("zoom_level")
        if zoom_raw is None or zoom_raw == "Uncertain":
            zoom = base_zoom
        else:
            try:
                zoom = int(zoom_raw)
            except (ValueError, TypeError):
                zoom = base_zoom

        px = patch_size * base_zoom / zoom  # patch side length in base_zoom pixels

        for patch_name in step.get("evaluated_patches_this_round", []):
            stem = os.path.splitext(patch_name)[0]
            try:
                x, y = map(int, stem.split("_"))
            except ValueError:
                continue
            roi_name = f"attempt{attempt}_{stem}"
            rois[roi_name] = [x, y, x + px, y + px]
            roi_metas[roi_name] = {"mag": float(zoom), "roi_w": float(px), "roi_h": float(px)}

    return rois, roi_metas


def estimate_total_slide_px(data: dict, patch_size: int, base_zoom: int = BASE_ZOOM) -> float | None:
    """Estimate total slide area at base_zoom using remaining_patch_count.

    Finds the first process step that has remaining_patch_count, then:
        total_patches = len(accumulated_patch_names) + remaining_patch_count
        total_slide_px = total_patches * patch_size^2
    patch_size here is the side length at base_zoom (no zoom conversion needed because
    remaining_patch_count counts base-pool patches, which are all at base_zoom).
    Returns None if no step has remaining_patch_count.
    """
    for step in data.get("process", []):
        remaining = step.get("remaining_patch_count")
        if remaining is None:
            continue
        accumulated = step.get("accumulated_patch_names", [])
        total_patches = len(accumulated) + remaining
        return total_patches * patch_size * patch_size
    return None


def _qa_rank(v) -> int:
    if v is True:
        return 2
    if v is False:
        return 1
    return 0


def build_qa_index(qa_folders: list[str]) -> dict[str, bool | None]:
    """Map case_name -> best 'correct' value from one or more QA result JSON folders.

    Filename convention: <case_name>[_suffix].json
    Groups by the part before the first '_', same as evaluate_results.py.
    """
    groups: dict[str, list] = {}
    for qa_folder in qa_folders:
        if not os.path.isdir(qa_folder):
            print(f"Warning: QA folder not found, skipping: {qa_folder}")
            continue
        for path in glob.glob(os.path.join(qa_folder, "*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[QA parse error] {path}: {e}")
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"^(test_\d+)", base)
            key = m.group(1) if m else base.split("_")[0]
            groups.setdefault(key, []).append(data.get("correct", None))
    return {k: max(vals, key=_qa_rank) for k, vals in groups.items()}


def build_gt_index(gt_dirs: list[str]) -> dict[str, str]:
    """Build a mapping of case_name -> csv_path from all GT directories."""
    index = {}
    for gt_dir in gt_dirs:
        if not os.path.isdir(gt_dir):
            print(f"Warning: GT dir not found, skipping: {gt_dir}")
            continue
        for fn in os.listdir(gt_dir):
            if fn.endswith(".csv") or fn.endswith(".xml"):
                case_name = fn[:-4]  # both extensions are 4 chars
                if case_name in index:
                    print(f"Warning: duplicate case {case_name} in {gt_dir}, keeping first match")
                else:
                    index[case_name] = os.path.join(gt_dir, fn)
    return index


def parse_gt_polygons(csv_path: str) -> list[Polygon]:
    """Parse closed polygons from GT CSV. Each polygon closes when first point repeats."""
    polygons = []
    current = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                pt = (float(row[0]), float(row[1]))
            except ValueError:
                continue  # skip header or malformed rows
            if not current:
                current.append(pt)
            elif pt == current[0] and len(current) >= 3:
                current.append(pt)
                poly = Polygon(current)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
                elif not poly.is_valid:
                    poly = poly.buffer(0)
                    if poly.area > 0:
                        polygons.append(poly)
                current = []
            else:
                current.append(pt)

    # Handle unclosed trailing polygon
    if len(current) >= 3:
        poly = Polygon(current)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            polygons.append(poly)

    return polygons


def parse_gt_polygons_xml(xml_path: str) -> list[Polygon]:
    """Parse tumor polygons from an ASAP XML annotation file.

    Skips annotations whose PartOfGroup is 'Exclusion'; includes all others
    (Tumor, _0, _1, _2, ...) as positive tumor regions.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []

    for annotation in root.iter("Annotation"):
        if annotation.get("PartOfGroup", "").lower() == "exclusion":
            continue
        coords_elem = annotation.find("Coordinates")
        if coords_elem is None:
            continue
        pts = sorted(coords_elem.findall("Coordinate"),
                     key=lambda c: int(c.get("Order", 0)))
        try:
            coords = [(float(c.get("X")), float(c.get("Y"))) for c in pts]
        except (TypeError, ValueError):
            continue
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            polygons.append(poly)

    return polygons


def parse_gt_polygons_any(gt_path: str) -> list[Polygon]:
    """Dispatch to CSV or XML parser based on file extension."""
    if gt_path.endswith(".xml"):
        return parse_gt_polygons_xml(gt_path)
    return parse_gt_polygons(gt_path)


def read_slide_info_from_csv(csv_path: str) -> tuple[int | None, float | None, float | None]:
    """Read total_size, wsi_40x_width, wsi_40x_height from GT CSV.

    Returns (total_size, wsi_w, wsi_h). wsi_w/h are None when those columns are
    absent (older CSV format that only has X_base, Y_base, total_size).
    """
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items() if k}
                try:
                    total_size = int(float(row["total_size"]))
                except (KeyError, ValueError):
                    continue
                wsi_w = float(row["wsi_40x_width"]) if "wsi_40x_width" in row else None
                wsi_h = float(row["wsi_40x_height"]) if "wsi_40x_height" in row else None
                return total_size, wsi_w, wsi_h
    except Exception:
        pass
    return None, None, None


def compute_overlap(roi_box: list, gt_union) -> dict:
    """Compute overlap metrics between a ROI box [x1,y1,x2,y2] and GT polygon union."""
    x1, y1, x2, y2 = roi_box
    roi_poly = box(x1, y1, x2, y2)
    roi_area = roi_poly.area

    if gt_union is None or gt_union.is_empty:
        return {
            "roi_area": roi_area,
            "gt_area": 0,
            "intersection": 0,
            "union": roi_area,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    intersection = roi_poly.intersection(gt_union).area
    gt_area = gt_union.area
    union = roi_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / roi_area if roi_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0

    return {
        "roi_area": roi_area,
        "gt_area": gt_area,
        "intersection": intersection,
        "union": union,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def evaluate_case(case_dir: str, gt_index: dict[str, str],
                  qa_index: dict[str, bool | None] | None = None) -> dict | None:
    """Evaluate one case. Returns None if no GT or no ROI file."""
    case_name = os.path.basename(case_dir)
    roi_path = os.path.join(case_dir, "roi_coords.json")

    if not os.path.exists(roi_path):
        return None
    gt_path = gt_index.get(case_name)
    if gt_path is None:
        return None

    with open(roi_path) as f:
        roi_data = json.load(f)

    polygons = parse_gt_polygons_any(gt_path)
    gt_union = unary_union(polygons) if polygons else None

    qa_correct = qa_index.get(case_name) if qa_index is not None else None
    results = {"case": case_name, "num_gt_polygons": len(polygons),
               "qa_correct": qa_correct, "rois": {}}

    for roi_name, coords in roi_data.items():
        metrics = compute_overlap(coords, gt_union)
        results["rois"][roi_name] = {**metrics, "coords": coords}

    # Aggregate: union of all ROIs vs GT
    if roi_data:
        all_roi_boxes = [box(*coords) for coords in roi_data.values()]
        all_roi_union = unary_union(all_roi_boxes)
        roi_union_area = all_roi_union.area
        gt_area = gt_union.area if gt_union and not gt_union.is_empty else 0
        intersection = all_roi_union.intersection(gt_union).area if gt_union and not gt_union.is_empty else 0
        union_area = roi_union_area + gt_area - intersection
        results["aggregate"] = {
            "num_rois": len(roi_data),
            "roi_union_area": roi_union_area,
            "gt_area": gt_area,
            "intersection": intersection,
            "union": union_area,
            "iou": intersection / union_area if union_area > 0 else 0.0,
            "precision": intersection / roi_union_area if roi_union_area > 0 else 0.0,
            "recall": intersection / gt_area if gt_area > 0 else 0.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json_dirs", nargs="+",
                        help="Folders of result JSONs containing 'process' key with "
                             "evaluated_patches_this_round.")
    parser.add_argument("--patch-size", type=int, default=4096,
                        help=f"Patch size in pixels at the recorded zoom_level (default 4096). "
                             f"Absent or 'Uncertain' zoom_level defaults to {BASE_ZOOM}x.")
    parser.add_argument("--gt-dir", nargs="+", default=DEFAULT_GT_DIRS,
                        help="One or more GT directories containing case CSVs")
    parser.add_argument("--output", default="roi_overlap_results.csv")
    parser.add_argument("--summary-output", default="roi_overlap_summary.csv")
    parser.add_argument("--mrfh-threshold", type=float, default=0.0,
                        help="Recall threshold for MRFH (Mean Reciprocal First Hit), default 0.0")
    parser.add_argument("--mrfh-threshold-output", default="mrfh_by_threshold.csv",
                        help="Output CSV for MRFH at precision thresholds 0.00-1.00 step 0.05")
    parser.add_argument("--hit-rate-output", default="hit_rate_by_threshold.csv",
                        help="Output CSV for hit rate (any ROI precision > threshold) at 0.00-1.00 step 0.05")
    parser.add_argument("--qa-folder", nargs="+", default=None,
                        help="One or more folders of QA result JSONs (each with a 'correct' field). "
                             "When provided, precision/recall are also reported weighted by QA correctness.")
    args = parser.parse_args()

    for out_path in (args.output, args.summary_output, args.mrfh_threshold_output, args.hit_rate_output):
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    gt_index = build_gt_index(args.gt_dir)
    print(f"GT index built: {len(gt_index)} cases from {len(args.gt_dir)} director(y/ies)")

    qa_index: dict | None = None
    if args.qa_folder:
        qa_index = build_qa_index(args.qa_folder)
        print(f"QA index built: {len(qa_index)} cases from {len(args.qa_folder)} folder(s)")

    all_results = []
    skipped_no_gt = 0
    skipped_no_roi = 0
    processed = 0

    hit_threshold_cache = _load_hit_threshold_cache()

    print(f"\nLoading result JSONs (patch_size={args.patch_size}, base_zoom={BASE_ZOOM}x)")
    for rj_dir in args.result_json_dirs:
        print(f"  {rj_dir}")
        for fname in sorted(glob.glob(os.path.join(rj_dir, "*.json"))):
            try:
                with open(fname) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[parse error] {fname}: {e}")
                continue

            long_id = meta.get("long_id", "")
            base = os.path.splitext(os.path.basename(fname))[0]
            m = re.match(r"^(test_\d+)", base)
            case_name = m.group(1) if m else base.split("_")[0]
            if not case_name:
                case_name = long_id.rsplit(".", 1)[0]

            gt_path = gt_index.get(case_name)
            if gt_path is None:
                skipped_no_gt += 1
                continue

            roi_data, roi_metas = extract_patches_as_rois(fname, patch_size=args.patch_size)
            if not roi_data:
                skipped_no_roi += 1
                continue

            total_slide_px = estimate_total_slide_px(meta, patch_size=args.patch_size)
            total_size, csv_wsi_w, csv_wsi_h = (
                read_slide_info_from_csv(gt_path) if not gt_path.endswith(".xml") else (None, None, None)
            )

            polygons = parse_gt_polygons_any(gt_path)
            gt_union = unary_union(polygons) if polygons else None
            qa_correct = qa_index.get(case_name) if qa_index is not None else None

            # Determine slide bounds for conditional hit threshold sampling.
            # Priority: CSV wsi_40x_width/height > sqrt(total_slide_px) approx > GT bounds.
            slide_bounds = _slide_bounds_fallback(
                gt_union,
                total_size,
                csv_wsi_w if csv_wsi_w else (total_slide_px ** 0.5 if total_slide_px else None),
                csv_wsi_h if csv_wsi_h else (total_slide_px ** 0.5 if total_slide_px else None),
            )
            _update_case_debug(hit_threshold_cache, case_name, gt_union, slide_bounds)

            res = {"case": case_name, "num_gt_polygons": len(polygons),
                   "qa_correct": qa_correct, "total_slide_px": total_slide_px,
                   "total_size": total_size, "rois": {}}
            k_rois = len(roi_data)
            for roi_name, coords in roi_data.items():
                metrics = compute_overlap(coords, gt_union)
                roi_meta = roi_metas.get(roi_name, {})
                mag = roi_meta.get("mag", float(BASE_ZOOM))
                roi_w = roi_meta.get("roi_w", abs(coords[2] - coords[0]))
                roi_h = roi_meta.get("roi_h", abs(coords[3] - coords[1]))
                threshold, _, entry_debug = _get_hit_threshold(
                    hit_threshold_cache, case_name,
                    _make_mag_key(mag), _make_size_key(roi_w, roi_h),
                    gt_union, slide_bounds,
                )
                threshold_bonf = _bonferroni_threshold(entry_debug, k_rois)
                res["rois"][roi_name] = {
                    **metrics, "coords": coords, "hit_threshold": threshold,
                    "hit": metrics["precision"] > threshold,
                    "hit_bonferroni": (metrics["precision"] > threshold_bonf
                                      if threshold_bonf is not None else None),
                }

            if roi_data:
                all_roi_boxes = [box(*coords) for coords in roi_data.values()]
                all_roi_union = unary_union(all_roi_boxes)
                roi_union_area = all_roi_union.area
                gt_area = gt_union.area if gt_union and not gt_union.is_empty else 0
                intersection = (all_roi_union.intersection(gt_union).area
                                if gt_union and not gt_union.is_empty else 0)
                union_area = roi_union_area + gt_area - intersection
                efficiency_pct = (roi_union_area / total_slide_px * 100
                                  if total_slide_px else None)
                res["aggregate"] = {
                    "num_rois": len(roi_data),
                    "roi_union_area": roi_union_area,
                    "gt_area": gt_area,
                    "intersection": intersection,
                    "union": union_area,
                    "iou": intersection / union_area if union_area > 0 else 0.0,
                    "precision": intersection / roi_union_area if roi_union_area > 0 else 0.0,
                    "recall": intersection / gt_area if gt_area > 0 else 0.0,
                    "efficiency_pct": efficiency_pct,
                }

            all_results.append(res)
            processed += 1

    print(f"\nProcessed: {processed} cases")
    print(f"Skipped (no GT): {skipped_no_gt}")
    print(f"Skipped (no ROI): {skipped_no_roi}")

    _save_hit_threshold_cache(hit_threshold_cache)

    # Write per-ROI CSV
    rows = []
    for res in all_results:
        agg = res.get("aggregate", {})
        qc = res["qa_correct"]  # True / False / None
        qa_mul = 1 if qc is True else (0 if qc is False else None)
        for roi_name, metrics in res["rois"].items():
            row = {
                "case": res["case"],
                "num_gt_polygons": res["num_gt_polygons"],
                "roi": roi_name,
                "roi_coords": str(metrics["coords"]),
                "roi_area": f"{metrics['roi_area']:.0f}",
                "gt_area": f"{metrics['gt_area']:.0f}",
                "intersection": f"{metrics['intersection']:.0f}",
                "iou": f"{metrics['iou']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "hit_threshold": f"{metrics.get('hit_threshold', 0.0):.6f}",
                "hit": int(metrics.get("hit", False)),
                "agg_iou": f"{agg.get('iou', 0):.4f}",
                "agg_precision": f"{agg.get('precision', 0):.4f}",
                "agg_recall": f"{agg.get('recall', 0):.4f}",
                "agg_coverage_pct": (f"{agg['efficiency_pct']:.4f}"
                                     if agg.get("efficiency_pct") is not None else ""),
            }
            if qa_index is not None:
                row["qa_correct"] = "" if qc is None else int(qc)
                row["qa_precision"] = "" if qa_mul is None else f"{metrics['precision'] * qa_mul:.4f}"
                row["qa_recall"]    = "" if qa_mul is None else f"{metrics['recall'] * qa_mul:.4f}"
                row["qa_agg_precision"] = "" if qa_mul is None else f"{agg.get('precision', 0) * qa_mul:.4f}"
                row["qa_agg_recall"]    = "" if qa_mul is None else f"{agg.get('recall', 0) * qa_mul:.4f}"
            rows.append(row)

    with open(args.output, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nPer-ROI results written to: {args.output}")

    # Compute summary statistics and write to summary CSV
    if all_results:
        # --- Case-level micro stats (union-of-ROIs per case pooled globally) ---
        case_inter = sum(r["aggregate"]["intersection"] for r in all_results if "aggregate" in r)
        case_roi_area = sum(r["aggregate"]["roi_union_area"] for r in all_results if "aggregate" in r)
        case_gt_area = sum(r["aggregate"]["gt_area"] for r in all_results if "aggregate" in r)
        case_union = case_roi_area + case_gt_area - case_inter
        case_iou = case_inter / case_union if case_union > 0 else 0.0
        case_prec = case_inter / case_roi_area if case_roi_area > 0 else 0.0
        case_recall = case_inter / case_gt_area if case_gt_area > 0 else 0.0

        # --- Per-ROI micro stats (every individual ROI pooled globally) ---
        roi_inter = sum(m["intersection"] for res in all_results for m in res["rois"].values())
        roi_area = sum(m["roi_area"] for res in all_results for m in res["rois"].values())
        # GT area repeated per ROI within same case; use case-level GT for recall
        roi_union = sum(m["union"] for res in all_results for m in res["rois"].values())
        roi_iou = roi_inter / roi_union if roi_union > 0 else 0.0
        roi_prec = roi_inter / roi_area if roi_area > 0 else 0.0
        roi_recall = roi_inter / case_gt_area if case_gt_area > 0 else 0.0

        total_rois = sum(len(res["rois"]) for res in all_results)
        total_16x16_subpatches = sum(
            (abs(m["coords"][2] - m["coords"][0]) // 16) * (abs(m["coords"][3] - m["coords"][1]) // 16)
            for res in all_results for m in res["rois"].values()
            if "coords" in m
        )
        thresholds = [round(x * 0.1, 1) for x in range(10)]

        print("\n=== Case-level Micro Statistics (union-of-ROIs vs GT, pixels pooled across cases) ===")
        print(f"Total intersection (px²): {case_inter:.0f}")
        print(f"Total ROI area     (px²): {case_roi_area:.0f}")
        print(f"Total GT area      (px²): {case_gt_area:.0f}")
        print(f"IoU:       {case_iou:.4f}")
        print(f"Precision: {case_prec:.4f}")
        print(f"Recall:    {case_recall:.4f}")

        print(f"\n=== Per-ROI Micro Statistics ({total_rois} ROIs, pixels pooled globally) ===")
        print(f"Total intersection (px²): {roi_inter:.0f}")
        print(f"Total ROI area     (px²): {roi_area:.0f}")
        print(f"IoU:       {roi_iou:.4f}")
        print(f"Precision: {roi_prec:.4f}")
        print(f"Recall:    {roi_recall:.4f}")
        for t in thresholds:
            n = sum(1 for res in all_results for m in res["rois"].values()
                    if m["iou"] > t)
            print(f"ROIs with IoU > {t:.1f}: {n} ({100*n/total_rois:.1f}%)")

        # --- Conditional hit rate (per-case/per-mag/per-size threshold) ---
        conditional_hit_cases = sum(
            1 for res in all_results
            if any(m.get("hit", False) for m in res["rois"].values())
        )
        conditional_hit_rate = conditional_hit_cases / processed if processed > 0 else 0.0

        def _roi_rank_key(name: str) -> int:
            tail = name.rsplit("_", 1)[-1]
            return int(tail) if tail.isdigit() else 0

        cond_rr_list = []
        for res in all_results:
            rois_sorted = sorted(res["rois"].items(), key=lambda kv: _roi_rank_key(kv[0]))
            rr = 0.0
            for rank, (_, m) in enumerate(rois_sorted, start=1):
                if m.get("hit", False):
                    rr = 1.0 / rank
                    break
            cond_rr_list.append(rr)
        cond_mrfh = sum(cond_rr_list) / len(cond_rr_list) if cond_rr_list else 0.0

        print(f"\n=== Conditional Hit (95th-pct random-overlap threshold per case/mag/size) ===")
        print(f"Cases with a conditional hit: {conditional_hit_cases} ({100*conditional_hit_rate:.1f}%)")
        print(f"Conditional MRFH: {cond_mrfh:.4f}")

        # --- Bonferroni hit rate (Bonferroni-corrected per-case/mag/size threshold) ---
        bonferroni_hit_cases = sum(
            1 for res in all_results
            if any(m.get("hit_bonferroni") is True for m in res["rois"].values())
        )
        bonferroni_hit_rate = bonferroni_hit_cases / processed if processed > 0 else 0.0

        bonf_rr_list = []
        for res in all_results:
            rois_sorted = sorted(res["rois"].items(), key=lambda kv: _roi_rank_key(kv[0]))
            rr = 0.0
            for rank, (_, m) in enumerate(rois_sorted, start=1):
                if m.get("hit_bonferroni") is True:
                    rr = 1.0 / rank
                    break
            bonf_rr_list.append(rr)
        bonf_mrfh = sum(bonf_rr_list) / len(bonf_rr_list) if bonf_rr_list else 0.0

        print(f"\n=== Bonferroni Hit (Bonferroni-corrected random-overlap threshold per case/mag/size) ===")
        print(f"Cases with a Bonferroni hit: {bonferroni_hit_cases} ({100*bonferroni_hit_rate:.1f}%)")
        print(f"Bonferroni MRFH: {bonf_mrfh:.4f}")

        # --- MRFH: Mean Reciprocal First Hit (case-level) ---
        # For each case, find the rank of the first ROI (sorted numerically) with IoU > threshold.
        # Reciprocal rank = 1/rank; 0 if no ROI hits. MRFH = mean across cases.

        mrfh_threshold = args.mrfh_threshold
        rr_list = []
        for res in all_results:
            rois_sorted = sorted(res["rois"].items(), key=lambda kv: _roi_rank_key(kv[0]))
            rr = 0.0
            for rank, (_, metrics) in enumerate(rois_sorted, start=1):
                if metrics["precision"] >= mrfh_threshold:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)
        mrfh = sum(rr_list) / len(rr_list) if rr_list else 0.0
        mrfh_hit_cases = sum(1 for rr in rr_list if rr > 0)

        print(f"\n=== MRFH (precision threshold={mrfh_threshold}, {processed} cases) ===")
        print(f"Cases with a hit: {mrfh_hit_cases} ({100*mrfh_hit_cases/processed:.1f}%)")
        print(f"MRFH: {mrfh:.4f}")

        # --- MRFH sweep: precision threshold 0.00 to 1.00, step 0.05 ---
        precision_thresholds = [round(t * 0.05, 2) for t in range(21)]
        mrfh_sweep_rows = []
        for t in precision_thresholds:
            rr_list_t = []
            for res in all_results:
                rois_sorted = sorted(res["rois"].items(), key=lambda kv: _roi_rank_key(kv[0]))
                rr = 0.0
                for rank, (_, m) in enumerate(rois_sorted, start=1):
                    if m["precision"] > t:
                        rr = 1.0 / rank
                        break
                rr_list_t.append(rr)
            mrfh_t = sum(rr_list_t) / len(rr_list_t) if rr_list_t else 0.0
            hit_t = sum(1 for rr in rr_list_t if rr > 0)
            mrfh_sweep_rows.append({
                "precision_threshold": f"{t:.2f}",
                "mrfh": f"{mrfh_t:.4f}",
                "hit_cases": hit_t,
                "hit_rate_pct": f"{100*hit_t/processed:.1f}" if processed > 0 else "0.0",
            })
        with open(args.mrfh_threshold_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["precision_threshold", "mrfh", "hit_cases", "hit_rate_pct"])
            writer.writeheader()
            writer.writerows(mrfh_sweep_rows)
        print(f"MRFH by precision threshold written to: {args.mrfh_threshold_output}")

        # --- Hit rate sweep: cases where ANY ROI has precision > threshold ---
        hit_rate_rows = []
        for t in precision_thresholds:
            hit_t = sum(
                1 for res in all_results
                if any(m["precision"] > t for m in res["rois"].values())
            )
            hit_rate_rows.append({
                "precision_threshold": f"{t:.2f}",
                "hit_cases": hit_t,
                "hit_rate_pct": f"{100*hit_t/processed:.1f}" if processed > 0 else "0.0",
            })
        with open(args.hit_rate_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["precision_threshold", "hit_cases", "hit_rate_pct"])
            writer.writeheader()
            writer.writerows(hit_rate_rows)
        print(f"Hit rate by precision threshold written to: {args.hit_rate_output}")

        # --- Efficiency: evaluated pixels vs total slide pixels ---
        # roi_area      = raw sum (same patch counted once per round it appears)
        # case_roi_area = union per case (deduped within each case)
        cases_with_slide_px = [r for r in all_results if r.get("total_slide_px")]
        total_slide_px_sum = sum(r["total_slide_px"] for r in cases_with_slide_px)
        avg_px_per_case = case_roi_area / processed if processed > 0 else 0.0
        overall_efficiency_pct = (case_roi_area / total_slide_px_sum * 100
                                  if total_slide_px_sum > 0 else None)
        avg_efficiency_pct = (
            sum(r["aggregate"]["efficiency_pct"]
                for r in all_results
                if r.get("aggregate", {}).get("efficiency_pct") is not None)
            / len(cases_with_slide_px)
            if cases_with_slide_px else None
        )

        print(f"\n=== Efficiency — Evaluated Pixels at {BASE_ZOOM}x ===")
        print(f"Total patches evaluated (all rounds): {total_rois}")
        print(f"Raw total patch area   (px², with duplicates): {roi_area:.0f}")
        print(f"Unique patch area      (px², union per case):  {case_roi_area:.0f}")
        print(f"Total slide area       (px², all cases):       {total_slide_px_sum:.0f}"
              + (" [estimated from remaining_patch_count]" if cases_with_slide_px else " [N/A]"))
        if overall_efficiency_pct is not None:
            print(f"Coverage (unique / total slide): {overall_efficiency_pct:.4f}%")
        if avg_efficiency_pct is not None:
            print(f"Avg coverage per case:           {avg_efficiency_pct:.4f}%")

        # --- QA-weighted stats (only when --qa-folder provided) ---
        qa_stats_rows: list[tuple] = []
        if qa_index is not None:
            qa_correct_cases = sum(1 for r in all_results if r["qa_correct"] is True)
            # Case-level: sum intersection only from QA-correct cases, divide by all-case areas
            qa_case_inter = sum(
                r["aggregate"]["intersection"] for r in all_results
                if "aggregate" in r and r["qa_correct"] is True
            )
            qa_case_prec = qa_case_inter / case_roi_area if case_roi_area > 0 else 0.0
            qa_case_recall = qa_case_inter / case_gt_area if case_gt_area > 0 else 0.0
            # Per-ROI: same idea
            qa_roi_inter = sum(
                m["intersection"] for res in all_results if res["qa_correct"] is True
                for m in res["rois"].values()
            )
            qa_roi_prec = qa_roi_inter / roi_area if roi_area > 0 else 0.0
            qa_roi_recall = qa_roi_inter / case_gt_area if case_gt_area > 0 else 0.0

            # Acc among annotated cases (all_results are already filtered to have GT)
            qa_has_result = sum(1 for r in all_results if r["qa_correct"] is not None)
            acc_annotated = qa_correct_cases / qa_has_result if qa_has_result > 0 else 0.0

            # Acc where QA-correct AND any ROI overlaps GT (aggregate intersection > 0)
            correct_with_overlap = sum(
                1 for r in all_results
                if r["qa_correct"] is True and r.get("aggregate", {}).get("intersection", 0) > 0
            )
            acc_overlap_and_correct = correct_with_overlap / qa_has_result if qa_has_result > 0 else 0.0

            print(f"\n=== QA-Weighted Statistics ({qa_correct_cases}/{processed} QA-correct cases) ===")
            print(f"Acc among annotated cases:        {acc_annotated:.4f}"
                  f"  ({qa_correct_cases}/{qa_has_result})")
            print(f"Acc (correct AND overlap > 0):    {acc_overlap_and_correct:.4f}"
                  f"  ({correct_with_overlap}/{qa_has_result})")
            print(f"Case-level QA-weighted precision: {qa_case_prec:.4f}")
            print(f"Case-level QA-weighted recall:    {qa_case_recall:.4f}")
            print(f"Per-ROI  QA-weighted precision:   {qa_roi_prec:.4f}")
            print(f"Per-ROI  QA-weighted recall:      {qa_roi_recall:.4f}")

            qa_stats_rows = [
                ("qa_correct_cases", qa_correct_cases,
                 f"{100*qa_correct_cases/processed:.1f}% of processed cases"),
                ("qa_has_result", qa_has_result,
                 "cases with GT annotation that also have a QA result"),
                ("acc_annotated", f"{acc_annotated:.4f}",
                 f"QA accuracy among annotated cases ({qa_correct_cases}/{qa_has_result})"),
                ("acc_overlap_and_correct", f"{acc_overlap_and_correct:.4f}",
                 f"QA-correct AND ROI overlaps GT ({correct_with_overlap}/{qa_has_result})"),
                ("qa_case_weighted_precision", f"{qa_case_prec:.4f}",
                 "intersection of QA-correct cases / all-case ROI area"),
                ("qa_case_weighted_recall", f"{qa_case_recall:.4f}",
                 "intersection of QA-correct cases / all-case GT area"),
                ("qa_roi_weighted_precision", f"{qa_roi_prec:.4f}",
                 "intersection of QA-correct cases (per ROI) / all-ROI area"),
                ("qa_roi_weighted_recall", f"{qa_roi_recall:.4f}",
                 "intersection of QA-correct cases (per ROI) / all-case GT area"),
            ]

        # --- Efficiency: unique ROI pixels / total_size from annotation CSV ---
        cases_with_size = [r for r in all_results if r.get("total_size")]
        total_size_sum = sum(r["total_size"] for r in cases_with_size)
        case_roi_area_eff = sum(
            r["aggregate"]["roi_union_area"] for r in cases_with_size if "aggregate" in r
        )
        overall_efficiency = case_roi_area_eff / total_size_sum if total_size_sum > 0 else None
        per_case_eff = [
            r["aggregate"]["roi_union_area"] / r["total_size"]
            for r in cases_with_size
            if "aggregate" in r and r["total_size"] > 0
        ]
        avg_efficiency = sum(per_case_eff) / len(per_case_eff) if per_case_eff else None

        print(f"\n=== Efficiency ({len(cases_with_size)}/{processed} cases have total_size) ===")
        if overall_efficiency is not None:
            print(f"Overall efficiency (unique ROI px / total slide px): {overall_efficiency:.6f}")
        if avg_efficiency is not None:
            print(f"Average per-case efficiency: {avg_efficiency:.6f}")

        # ---------- Write summary CSV ----------
        denom_16x16_multiscale = (
            total_size_sum / 256 +           # 40x: full-res pixels / 16²
            (total_size_sum / 4) / 256 +     # 20x: 40x/4 pixels / 16²
            (total_size_sum / 64) / 256      # 5x:  40x/64 pixels / 16²
        ) if total_size_sum > 0 else 0
        coverage_16x16_multiscale = total_16x16_subpatches / denom_16x16_multiscale if denom_16x16_multiscale > 0 else None
        global_stats = [
            ("processed_cases", processed, ""),
            ("skipped_no_gt", skipped_no_gt, ""),
            ("skipped_no_roi", skipped_no_roi, ""),
            ("case_total_intersection_px2", f"{case_inter:.0f}", "union-of-ROIs vs GT, summed across cases"),
            ("case_total_roi_area_px2", f"{case_roi_area:.0f}", ""),
            ("case_total_gt_area_px2", f"{case_gt_area:.0f}", ""),
            ("case_micro_iou", f"{case_iou:.4f}", ""),
            ("case_micro_precision", f"{case_prec:.4f}", ""),
            ("case_micro_recall", f"{case_recall:.4f}", ""),
            ("per_roi_total", total_rois, "total ROI patches used across all processed cases"),
            ("total_16x16_subpatches", total_16x16_subpatches,
             "total 16x16 sub-patches across all ROI boxes (floor(w/16) * floor(h/16))"),
            ("coverage_16x16_multiscale",
             f"{coverage_16x16_multiscale:.6f}" if coverage_16x16_multiscale is not None else "N/A",
             "total_16x16_subpatches / (full-slide 16x16 tiles at 40x + 20x + 5x)"),
            ("per_roi_total_intersection_px2", f"{roi_inter:.0f}", "each ROI vs GT, summed globally"),
            ("per_roi_total_roi_area_px2", f"{roi_area:.0f}", ""),
            ("per_roi_micro_iou", f"{roi_iou:.4f}", ""),
            ("per_roi_micro_precision", f"{roi_prec:.4f}", ""),
            ("per_roi_micro_recall", f"{roi_recall:.4f}", ""),
            *[
                (f"per_roi_iou_gt{t:.1f}",
                 n := sum(1 for res in all_results for m in res["rois"].values() if m["iou"] > t),
                 f"{100*n/total_rois:.1f}%")
                for t in thresholds
            ],
            ("conditional_hit_cases", conditional_hit_cases,
             f"{100*conditional_hit_rate:.1f}% — cases where any ROI passes 95th-pct random threshold"),
            ("conditional_hit_rate", f"{conditional_hit_rate:.4f}", ""),
            ("conditional_mrfh", f"{cond_mrfh:.4f}", "MRFH using per-case/mag/size conditional threshold"),
            ("bonferroni_hit_cases", bonferroni_hit_cases,
             f"{100*bonferroni_hit_rate:.1f}% — cases where any ROI passes Bonferroni-corrected threshold"),
            ("bonferroni_hit_rate", f"{bonferroni_hit_rate:.4f}", ""),
            ("bonferroni_mrfh", f"{bonf_mrfh:.4f}", "MRFH using Bonferroni-corrected threshold"),
            (f"mrfh_threshold", mrfh_threshold, "precision threshold for first-hit detection"),
            ("mrfh_hit_cases", mrfh_hit_cases, f"{100*mrfh_hit_cases/processed:.1f}% of cases have a hit"),
            ("mrfh", f"{mrfh:.4f}", "Mean Reciprocal First Hit across cases"),
            *qa_stats_rows,
            ("efficiency_cases_with_total_size", len(cases_with_size),
             "cases with total_size in GT CSV"),
            ("efficiency_total_slide_px2",
             f"{total_size_sum:.0f}" if total_size_sum > 0 else "N/A",
             "sum of total_size (WSI width*height) from GT CSV"),
            ("efficiency",
             f"{overall_efficiency:.6f}" if overall_efficiency is not None else "N/A",
             "unique ROI px / total slide px (sum across cases with total_size)"),
            ("efficiency_avg_per_case",
             f"{avg_efficiency:.6f}" if avg_efficiency is not None else "N/A",
             "per-case efficiency averaged across cases with total_size"),
        ]
        with open(args.summary_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value", "note"])
            writer.writeheader()
            writer.writerows({"metric": m, "value": v, "note": n} for m, v, n in global_stats)

        print(f"\nSummary written to: {args.summary_output}")


if __name__ == "__main__":
    main()
