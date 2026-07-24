import os
import json
import argparse
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon, box


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
# Generate mask for ONE case
# =========================
def generate_patch_mask(
    feature_dir,
    polygons,
    patch_size=224,
    use_intersection=False,
    scale=1.0
):
    mask_dict = {}
    total = 0
    tumor_count = 0

    for fname in os.listdir(feature_dir):
        if not fname.endswith(".npy"):
            continue

        total += 1

        try:
            x, y = map(int, fname.replace(".npy", "").split("_"))
        except:
            continue

        x = x * scale
        y = y * scale

        if len(polygons) == 0:
            is_tumor = False
        else:
            if use_intersection:
                patch_box = box(x, y, x + patch_size, y + patch_size)
                is_tumor = any(poly.intersects(patch_box) for poly in polygons)
            else:
                center = Point(x + patch_size // 2, y + patch_size // 2)
                is_tumor = any(poly.contains(center) for poly in polygons)

        if is_tumor:
            tumor_count += 1

        mask_dict[fname.replace(".npy", ".jpg")] = {"is_tumor": is_tumor}

    return mask_dict, tumor_count, total


# =========================
# MAIN (batch version)
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_root", type=str, required=True,
                        help="Root dir: contains many feature folders (e.g., test_001, test_002...)")

    parser.add_argument("--xml_root", type=str, required=True,
                        help="Root dir: contains XML files")

    parser.add_argument("--output_root", type=str, required=True,
                        help="Where to save mask jsons")

    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--use_intersection", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    cases = sorted(os.listdir(args.feature_root))

    print(f"Found {len(cases)} cases")

    for case_name in cases:
        feature_dir = os.path.join(args.feature_root, case_name)

        if not os.path.isdir(feature_dir):
            continue

        # 👉 對應 XML（你可以依 naming 改）
        xml_path = os.path.join(args.xml_root, f"{case_name}.xml")

        print(f"\nProcessing {case_name}")

        if os.path.exists(xml_path):
            polygons = parse_xml_polygons(xml_path)
            print(f"  Loaded {len(polygons)} polygons")
        else:
            print("  ⚠️ No XML found → treat as normal slide")
            polygons = []

        mask_dict, tumor_count, total = generate_patch_mask(
            feature_dir=feature_dir,
            polygons=polygons,
            patch_size=args.patch_size,
            use_intersection=args.use_intersection,
            scale=args.scale
        )

        output_path = os.path.join(args.output_root, f"{case_name}.json")

        with open(output_path, "w") as f:
            json.dump(mask_dict, f, indent=2)

        print(f"  Saved → {output_path}")
        print(f"  Tumor patches: {tumor_count}/{total}")


if __name__ == "__main__":
    main()