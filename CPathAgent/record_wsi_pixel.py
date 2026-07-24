"""
For each CSV in csv_dir, find the matching .svs or .tiff file in input_dir by case_id,
open it with WSILoader, compute total pixels (width * height), and write the
result back into the same CSV as new columns: 'total_size', 'wsi_40x_width', 'wsi_40x_height'.

Usage:
    python record_wsi_pixel.py <input_dir> <csv_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import WSILoader

try:
    import openslide
    _OPENSLIDE = True
except ImportError:
    _OPENSLIDE = False

_TARGET_MPP = 0.25  # μm/px at 40×


def get_wsi_info(wsi_path: Path) -> Tuple[int, int, int]:
    """Return (total_pixels, width_40x, height_40x) for a WSI."""
    with WSILoader(str(wsi_path)) as loader:
        w, h = loader.width, loader.height
        total = w * h

        mpp = None
        if _OPENSLIDE and loader._backend == "openslide":
            try:
                mpp = float(loader._slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0) or 0)
            except (ValueError, TypeError):
                mpp = None

        if mpp and mpp > 0:
            scale = mpp / _TARGET_MPP
            w40 = round(w * scale)
            h40 = round(h * scale)
        else:
            w40, h40 = w, h

        return total, w40, h40


WSI_EXTENSIONS = ("*.svs", "*.tiff", "*.tif")


def build_wsi_index(input_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for pattern in WSI_EXTENSIONS:
        for p in input_dir.rglob(pattern):
            index.setdefault(p.stem, p)
    return index


def process_csv(csv_path: Path, svs_index: dict[str, Path]) -> None:
    case_id = csv_path.stem.split("_")[0]
    wsi_path = svs_index.get(case_id)
    if wsi_path is None:
        print(f"  [WARN] {csv_path.name}: no .svs/.tiff found for case_id={case_id}")
        return

    total, w40, h40 = get_wsi_info(wsi_path)
    print(f"  [{case_id}] total_size={total}  40x=({w40} x {h40})")

    if csv_path.stat().st_size == 0:
        print(f"  [WARN] {csv_path.name}: file is empty, skipping")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"  [WARN] {csv_path.name}: no columns to parse, skipping")
        return

    new_cols = {"total_size", "wsi_40x_width", "wsi_40x_height"}
    if new_cols.issubset(df.columns) and df[list(new_cols)].notna().all(axis=None):
        print(f"  [SKIP] {csv_path.name}: columns already present")
        return

    df["total_size"] = total
    df["wsi_40x_width"] = w40
    df["wsi_40x_height"] = h40
    df.to_csv(csv_path, index=False)
    print(f"  -> saved {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record total WSI pixel count (width*height) into CSV files by case_id."
    )
    parser.add_argument(
        "input_dir",
        help="Root directory; .svs and .tiff/.tif files are searched recursively.",
    )
    parser.add_argument(
        "csv_dir",
        help="Directory containing CSV files with a 'case_id' column.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_dir = Path(args.csv_dir)

    svs_index = build_wsi_index(input_dir)
    print(f"Indexed {len(svs_index)} WSI files (.svs/.tiff) from {input_dir}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {csv_dir}")

    for csv_path in csv_files:
        print(f"\nProcessing {csv_path.name}")
        process_csv(csv_path, svs_index)

    print("\nDone.")


if __name__ == "__main__":
    main()
