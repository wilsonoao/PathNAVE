import os
import json
import random
import pickle
import argparse
import xml.etree.ElementTree as ET

# ===== Paths =====
SLIDE_DIR = "/work/Agent_dataset/CAMELYON16/slides"
ANNOTATION_DIR = "/work/Agent_dataset/CAMELYON16/annotations"
OUTPUT_PATH = "/work/Agent_dataset/CAMELYON16/roi_qa_dataset.json"

BRCA_SLIDE_DIR = "/work/data/TCGA-BRCA-FS/CHIEF/picture_wsi"
BRCA_LABEL_PATH = "../data/subtype/BRCA.pkl"
BRCA_OUTPUT_PATH = "/work/Agent_dataset/TCGA/BRCA/subtype_qa_dataset_choiceV2.json"
BRCA_ANNOTATION_DIR = "/work/data/TCGA-BRCA-FS/Tumor_annotation/BRCA"

LUNG_SLIDE_DIRS = {
    "LUAD": "/work/data/TCGA-LUAD-FS/CHIEF/picture_wsi",
    "LUSC": "/work/data/TCGA-LUSC-FS/CHIEF/picture_wsi",
}
LUNG_OUTPUT_PATH = "/work/Agent_dataset/TCGA/LUNG/subtype_qa_dataset.json"

# ===== Prompt pools =====
DETECTION_LOCALIZE_TEMPLATES = [
    "Is there any metastatic tumor present in this slide?",
    "Does this slide contain tumor regions?",
    "Determine whether tumor is present.",
]

# MULTI_LESION_TEMPLATES = [
#     "Are there multiple distinct tumor regions in this slide?",
#     "Does this slide contain more than one tumor region?",
#     "Is there evidence of multiple tumor foci in this slide?",
# ]

# SUBTYPE_TEMPLATES = [
#     "Does this slide indicate which breast cancer subtype?\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
#     "Classify this breast tumor.\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
#     "Which subtype is most consistent with the carcinoma pattern?\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
# ]

SUBTYPE_TEMPLATES = [
    "As an expert pathologist with extensive expertise in evaluating histopathological slides, could you determine the histological subtype of the breast tumor depicted in this image?\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
    "As a pathologist with vast expertise in examining histopathological slides, can you determine the histological subtype of the breast tumor depicted in the image provided, which is specifically from a case of breast cancer?\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
    "As an expert pathologist with a wealth of experience in evaluating histopathological slides, can you identify the histological subtype of the breast tumor depicted in the image?\n(A) IDC\n(B) ILC\nAnswer with A or B only.",
]

SUBTYPE_LABEL_MAP = {
    "IDC": "Invasive Ductal Carcinoma (IDC)",
    "ILC": "Invasive Lobular Carcinoma (ILC)",
}

LUNG_SUBTYPE_TEMPLATES = [
    "Is this tumor LUAD or LUSC?",
    "Does this slide indicate LUAD or LUSC subtype?",
    "Does this slide show LUAD or LUSC?",
]

LUNG_SUBTYPE_LABEL_MAP = {
    "LUAD": "Lung Adenocarcinoma (LUAD)",
    "LUSC": "Lung Squamous Cell Carcinoma (LUSC)",
}


# ===== 解析 XML lesion（回傳完整座標）=====
def parse_lesions(xml_path):
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    lesions = []

    for anno in root.findall(".//Annotation"):
        group = anno.attrib.get("PartOfGroup", "")

        # 只取 tumor
        if group.lower() != "tumor" and group.lower() != "_0":
            print(group.lower())
            continue

        coords = []
        for c in anno.findall(".//Coordinate"):
            x = float(c.attrib["X"])
            y = float(c.attrib["Y"])
            coords.append({"x": x, "y": y})

        if len(coords) == 0:
            continue

        xs = [p["x"] for p in coords]
        ys = [p["y"] for p in coords]

        bbox = {
            "xmin": min(xs),
            "ymin": min(ys),
            "xmax": max(xs),
            "ymax": max(ys)
        }

        center = {
            "x": (bbox["xmin"] + bbox["xmax"]) / 2,
            "y": (bbox["ymin"] + bbox["ymax"]) / 2
        }

        lesions.append({
            "bbox": bbox,
            "center": center,
            "polygon": coords
        })

    return lesions


def load_brca_labels(pkl_path):
    """Load BRCA subtype labels; return dict: case_submitter_id -> subtype."""
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    return dict(zip(df["case_submitter_id"], df["subtype"]))


def match_case_id(slide_id, label_map):
    """
    TCGA slide filename contains the case_submitter_id as the first three
    hyphen-separated fields (e.g. TCGA-A7-A0DA-...).  Try progressively
    shorter prefixes until we find a match.
    """
    parts = slide_id.split("-")
    for n in range(len(parts), 2, -1):
        candidate = "-".join(parts[:n])
        if candidate in label_map:
            return candidate
    return None


# ===== Task: detection =====
def run_detection():
    dataset = []

    for slide_file in os.listdir(SLIDE_DIR):
        if not slide_file.endswith(".tif"):
            continue

        slide_id = os.path.splitext(slide_file)[0]
        xml_path = os.path.join(ANNOTATION_DIR, slide_id + ".xml")

        if not os.path.exists(xml_path):
            print(f"[Normal] {slide_id} (no XML found, treated as normal slide)")
            detection_answer = {
                "presence": "No",
                "num_lesions": 0,
                "lesions": []
            }
        else:
            lesions = parse_lesions(xml_path)
            num_lesions = len(lesions)

            if num_lesions > 0:
                detection_answer = {
                    "presence": "Yes",
                    "num_lesions": num_lesions,
                    "lesions": lesions
                }
            else:
                detection_answer = {
                    "presence": "No",
                    "num_lesions": 0,
                    "lesions": []
                }

        dataset.append({
            "Id": slide_id,
            "Question": random.choice(DETECTION_LOCALIZE_TEMPLATES),
            "Answer": detection_answer,
            "type": "DetectionLocalization"
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} QA pairs to {OUTPUT_PATH}")


# ===== Task: brca_subtype =====
def run_brca_subtype():
    if not os.path.exists(BRCA_SLIDE_DIR):
        print(f"[BRCA] Slide dir not found: {BRCA_SLIDE_DIR}")
        return
    if not os.path.exists(BRCA_LABEL_PATH):
        print(f"[BRCA] Label file not found: {BRCA_LABEL_PATH}")
        return

    label_map = load_brca_labels(BRCA_LABEL_PATH)

    # Build allowed set from annotation directory (strip .csv extension)
    # annotation_ids = {
    #     os.path.splitext(f)[0].lower()
    #     for f in os.listdir(BRCA_ANNOTATION_DIR)
    #     if f.endswith(".csv")
    # }

    # print(len(annotation_ids))

    brca_dataset = []
    skipped = 0
    skipped_no_annotation = 0

    print(len(os.listdir(BRCA_SLIDE_DIR)))
    for slide_file in os.listdir(BRCA_SLIDE_DIR):
        if not slide_file.lower().endswith((".tif", ".svs", ".ndpi")):
            continue

        slide_id = os.path.splitext(slide_file)[0]

        # if slide_id.lower() not in annotation_ids:
        #     skipped_no_annotation += 1
        #     continue

        case_id = match_case_id(slide_id, label_map)

        if case_id is None:
            print(case_id)
            skipped += 1
            continue

        subtype = label_map[case_id]
        answer_label = SUBTYPE_LABEL_MAP.get(subtype, subtype)

        brca_dataset.append({
            "Id": slide_id,
            "Question": random.choice(SUBTYPE_TEMPLATES),
            "Answer": answer_label,
            "type": "Subtype"
        })

    os.makedirs(os.path.dirname(BRCA_OUTPUT_PATH), exist_ok=True)
    with open(BRCA_OUTPUT_PATH, "w") as f:
        json.dump(brca_dataset, f, indent=2)

    answer_counts = {}
    for item in brca_dataset:
        ans = item["Answer"]
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    print(f"[BRCA] Saved {len(brca_dataset)} subtype QA pairs to {BRCA_OUTPUT_PATH} (skipped no_label={skipped}, no_annotation={skipped_no_annotation})")
    print("[BRCA] Answer distribution:")
    for ans, count in sorted(answer_counts.items()):
        print(f"  {ans}: {count}")


# ===== Task: lung_subtype =====
def run_lung_classification():
    lung_dataset = []

    for subtype, slide_dir in LUNG_SLIDE_DIRS.items():
        if not os.path.exists(slide_dir):
            print(f"[LUNG] Slide dir not found: {slide_dir}")
            continue

        answer_label = LUNG_SUBTYPE_LABEL_MAP.get(subtype, subtype)
        svs_files = [f for f in os.listdir(slide_dir) if f.lower().endswith(".svs")]
        print(f"[LUNG] {subtype}: {len(svs_files)} slides")

        for slide_file in svs_files:
            slide_id = os.path.splitext(slide_file)[0]
            lung_dataset.append({
                "Id": slide_id,
                "Question": random.choice(LUNG_SUBTYPE_TEMPLATES),
                "Answer": answer_label,
                "type": "Subtype"
            })

    os.makedirs(os.path.dirname(LUNG_OUTPUT_PATH), exist_ok=True)
    with open(LUNG_OUTPUT_PATH, "w") as f:
        json.dump(lung_dataset, f, indent=2)

    answer_counts = {}
    for item in lung_dataset:
        ans = item["Answer"]
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    print(f"[LUNG] Saved {len(lung_dataset)} subtype QA pairs to {LUNG_OUTPUT_PATH}")
    print("[LUNG] Answer distribution:")
    for ans, count in sorted(answer_counts.items()):
        print(f"  {ans}: {count}")


# ===== Entry point =====
TASKS = {
    "detection": run_detection,
    "brca_subtype": run_brca_subtype,
    "lung_classification": run_lung_classification,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA datasets for WSI tasks.")
    parser.add_argument(
        "task",
        choices=list(TASKS.keys()),
        help="QA task to run: " + ", ".join(TASKS.keys()),
    )
    args = parser.parse_args()
    TASKS[args.task]()
