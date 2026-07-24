import os
import openslide
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from pathlib import Path
import json


# ========= 設定 =========
WSI_DIR = "/work/Agent_dataset/CAMELYON16/slides"
OUTPUT_DIR = "/work/Agent_benchmark/Pathology-CoT/result/CAMELYON16"
YOLO_MODEL_PATH = "/work/Agent_benchmark/Pathology-CoT/pathology-o3/best.pt"

THUMB_SIZE = 1024
CONF_THRES = 0.01
MAX_ROIS = 10
ROI_LEVEL = 0
# ========================


def create_thumbnail(slide):
    w, h = slide.dimensions
    thumb_w = THUMB_SIZE
    thumb_h = int(h * thumb_w / w)
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    return thumbnail


def run_yolo(model, image, conf=None):
    if conf is None:
        conf = CONF_THRES
    results = model(image, imgsz=THUMB_SIZE, conf=conf, iou=0.3, verbose=False)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return [], []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    return boxes, scores


# 🔥 畫全部 ROI（含 index + score）
def draw_all_boxes(image, boxes, scores, save_path):
    img = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{i}:{scores[i-1]:.2f}"
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, img)


# 🔥 畫單一 ROI
def draw_single_box(image, box, save_path):
    img = image.copy()
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(save_path, img)


def map_to_wsi_coords(boxes, slide, thumbnail):
    w, h = slide.dimensions
    scale_x = w / thumbnail.width
    scale_y = h / thumbnail.height

    mapped_boxes = []
    for (x1, y1, x2, y2) in boxes:
        X1 = int(x1 * scale_x)
        Y1 = int(y1 * scale_y)
        X2 = int(x2 * scale_x)
        Y2 = int(y2 * scale_y)
        mapped_boxes.append((X1, Y1, X2, Y2))

    return mapped_boxes

def save_roi_bbox(case_dir, boxes):
    data = {}

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        data[f"roi_{i}"] = [
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ]

    json_path = os.path.join(case_dir, "roi_coords.json")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def crop_rois(slide, boxes, save_dir):
    roi_paths = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        try:
            region = slide.read_region((x1, y1), ROI_LEVEL, (w, h)).convert("RGB")
            roi_path = os.path.join(save_dir, f"roi_{i}.jpg")
            region.save(roi_path)
            roi_paths.append(roi_path)
        except Exception as e:
            print(f"[Warning] ROI crop failed: {e}")

    return roi_paths


def process_slide(slide_path, model, case_id):
    slide = openslide.OpenSlide(slide_path)
    filename = Path(slide_path).stem

    case_dir = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(case_dir, exist_ok=True)

    # --- thumbnail ---
    thumbnail = create_thumbnail(slide)
    thumb_path = os.path.join(case_dir, "thumbnail.jpg")
    thumbnail.save(thumb_path)

    thumbnail_np = np.array(thumbnail)

    # --- YOLO (降低 conf 直到找到至少一個 ROI) ---
    conf = CONF_THRES
    boxes, scores = run_yolo(model, thumbnail_np, conf)
    while len(boxes) == 0 and conf > 0.00005:
        conf = round(conf - 0.00005, 5)
        print(f"[Info] No ROI found, retrying with conf={conf:.6f}: {filename}")
        boxes, scores = run_yolo(model, thumbnail_np, conf)

    if len(boxes) == 0:
        print(f"[Warning] No ROI found even at conf={conf:.3f}: {slide_path}")
        slide.close()
        return

    # --- sort top-K ---
    idx = np.argsort(scores)[::-1][:MAX_ROIS]
    boxes = boxes[idx]
    scores = scores[idx]

    # 🔥 畫全部 ROI
    draw_all_boxes(
        thumbnail_np,
        boxes,
        scores,
        os.path.join(case_dir, "all_rois.jpg")
    )

    # 🔥 畫 top-1
    draw_single_box(
        thumbnail_np,
        boxes[0],
        os.path.join(case_dir, "top1_roi.jpg")
    )

    # 🔥 畫第10個 ROI（如果有）
    if len(boxes) >= 10:
        draw_single_box(
            thumbnail_np,
            boxes[9],
            os.path.join(case_dir, "roi_10.jpg")
        )

    # --- mapping ---
    mapped_boxes = map_to_wsi_coords(boxes, slide, thumbnail)

    # --- save mapping ---
    save_roi_bbox(case_dir, mapped_boxes)

    # --- crop ---
    crop_rois(slide, mapped_boxes, case_dir)

    slide.close()


def load_test_total_cases(test_total_dir):
    cases = set()
    if os.path.isdir(test_total_dir):
        for fname in os.listdir(test_total_dir):
            case_id = fname.split("_")[0]
            if case_id:
                cases.add(case_id)
    return cases


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL_PATH)

    test_total_dir = os.path.join(OUTPUT_DIR, "test_total")
    test_total_cases = load_test_total_cases(test_total_dir)
    print(len(test_total_cases))

    tif_files = [f for f in os.listdir(WSI_DIR) if f.endswith(".tif")]
    # tif_files = [f for f in os.listdir(WSI_DIR) if f.endswith(".svs")]
    print(len(tif_files))
    for i, fname in enumerate(tqdm(tif_files)):
        filename = Path(fname).stem
        if filename in test_total_cases:
            # print(f"[Skip] Already in test_total: {filename}")
            continue
        case_dir = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(os.path.join(case_dir, "roi_1.jpg")):
            print(f"[Warning] Already in test_total: {filename}")
            continue
        slide_path = os.path.join(WSI_DIR, fname)
        process_slide(slide_path, model, i + 1)


if __name__ == "__main__":
    main()