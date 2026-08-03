"""
Convert ASAP-format XML annotation files to CSV.

Each output CSV has columns: X_base, Y_base, total_size
where total_size = width * height of the matching WSI at highest magnification.

Usage:
    python xml_to_csv.py <xml_dir> <wsi_dir> [output_dir]

If output_dir is omitted, CSVs are written alongside the XML files.
WSI files (.svs / .tif / .tiff) are matched by stem name (e.g. test_108.xml -> test_108.tif).
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

try:
    import openslide
    _OPENSLIDE = True
except ImportError:
    _OPENSLIDE = False

_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs"}


def build_wsi_index(wsi_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in _WSI_EXTS:
        for p in wsi_dir.rglob(f"*{ext}"):
            index.setdefault(p.stem, p)
    return index


def get_total_pixels(wsi_path: Path) -> int:
    if _OPENSLIDE:
        slide = openslide.OpenSlide(str(wsi_path))
        w, h = slide.dimensions
        slide.close()
        return w * h
    from PIL import Image
    with Image.open(str(wsi_path)) as img:
        return img.width * img.height


def parse_xml(xml_path: Path) -> list[tuple[float, float]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    coords: list[tuple[float, float]] = []
    for coord in root.iter("Coordinate"):
        x = float(coord.attrib["X"])
        y = float(coord.attrib["Y"])
        coords.append((x, y))
    return coords


def process_xml(xml_path: Path, wsi_index: dict[str, Path], output_dir: Path) -> None:
    stem = xml_path.stem
    wsi_path = wsi_index.get(stem)
    if wsi_path is None:
        print(f"  [WARN] {xml_path.name}: no matching WSI found, skipping")
        return

    coords = parse_xml(xml_path)
    if not coords:
        print(f"  [WARN] {xml_path.name}: no coordinates found, skipping")
        return

    total_size = get_total_pixels(wsi_path)
    print(f"  [{stem}] {len(coords)} coords, total_size={total_size}")

    df = pd.DataFrame(coords, columns=["X_base", " Y_base"])
    df["total_size"] = total_size

    out_path = output_dir / (stem + ".csv")
    df.to_csv(out_path, index=False)
    print(f"  -> saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ASAP XML annotations to CSV with WSI total_size."
    )
    parser.add_argument("xml_dir", help="Directory containing .xml annotation files.")
    parser.add_argument("wsi_dir", help="Directory containing .svs/.tif/.tiff WSI files.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for CSV files (default: same as xml_dir).",
    )
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    wsi_dir = Path(args.wsi_dir)
    output_dir = Path(args.output_dir) if args.output_dir else xml_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_index = build_wsi_index(wsi_dir)
    print(f"Indexed {len(wsi_index)} WSI files from {wsi_dir}")

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {xml_dir}")

    for xml_path in xml_files:
        print(f"\nProcessing {xml_path.name}")
        process_xml(xml_path, wsi_index, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
