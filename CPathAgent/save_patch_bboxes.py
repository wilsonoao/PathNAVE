"""
Save the bounding box of every patch produced by partition_grid to JSON files.

For each .svs or .tiff file found in the input directory, one JSON file named
<case_id>.json is written to the output directory.

Output format per file:
    {
        "0": [x1, y1, x2, y2],
        "1": [x1, y1, x2, y2],
        ...
    }

Coordinates are absolute pixel positions in the full WSI space (level-0).

Usage:
    python save_patch_bboxes.py <input_dir> <output_dir> [--patch-size 512]
                                                          [--overlap 0.05]
                                                          [--thumbnail-scale 32]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import WSILoader


def process_wsi(
    wsi_path: Path,
    output_dir: Path,
    patch_size_px: int,
    overlap: float,
    thumbnail_scale: int,
) -> None:
    case_id = wsi_path.stem
    out_file = output_dir / f"{case_id}.json"

    with WSILoader(str(wsi_path), thumbnail_scale=thumbnail_scale) as loader:
        tiles = loader.partition_grid(
            patch_size_px=patch_size_px,
            overlap=overlap,
        )
        W, H = loader.width, loader.height

    bboxes: dict[str, list[int]] = {
        str(pid): [
            int(x1 * W), int(y1 * H),
            int(x2 * W), int(y2 * H),
        ]
        for pid, x1, y1, x2, y2 in tiles
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)

    print(f"  [{case_id}] {len(bboxes)} patches -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save patch bounding boxes for all .svs/.tiff files in a directory."
    )
    parser.add_argument("input_dir", help="Directory containing .svs or .tiff/.tif files.")
    parser.add_argument("output_dir", help="Directory where <case_id>.json files are written.")
    parser.add_argument(
        "--patch-size", type=int, default=512,
        help="Patch size in thumbnail pixels (default: 512).",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.05,
        help="Overlap fraction between adjacent patches (default: 0.05).",
    )
    parser.add_argument(
        "--thumbnail-scale", type=int, default=32,
        help="Downscale factor used to build the thumbnail (default: 32).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_files = sorted({
        p.stem: p
        for pattern in ("*.svs", "*.tiff", "*.tif")
        for p in input_dir.glob(pattern)
    }.values())
    if not wsi_files:
        print(f"No .svs/.tiff files found in {input_dir}")
        return

    print(f"Found {len(wsi_files)} WSI files in {input_dir}")
    for wsi_path in wsi_files:
        process_wsi(
            wsi_path=wsi_path,
            output_dir=output_dir,
            patch_size_px=args.patch_size,
            overlap=args.overlap,
            thumbnail_scale=args.thumbnail_scale,
        )

    print(f"\nDone. {len(wsi_files)} JSON files saved to {output_dir}")


if __name__ == "__main__":
    main()
