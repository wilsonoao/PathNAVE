"""
Save the bounding box of every patch produced by partition_grid to JSON files.

input_dir is a directory whose subdirectories each contain .svs files.
done_dir  is a directory whose entries (subdirs or files) represent already-reviewed
          case IDs — only .svs files whose stem matches a case ID in done_dir are processed.

For each matched .svs file, one JSON file named <case_id>.json is written to output_dir.

Output format per file:
    {
        "0": [x1, y1, x2, y2],
        "1": [x1, y1, x2, y2],
        ...
    }

Coordinates are absolute pixel positions in the full WSI space (level-0).

Usage:
    python save_patch_bboxes_v2.py <input_dir> <done_dir> <output_dir>
                                   [--patch-size 512] [--overlap 0.05]
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


def collect_done_ids(done_dir: Path) -> set[str]:
    """Return the set of case IDs found in done_dir, taking the part before the first '_'."""
    ids: set[str] = set()
    for entry in done_dir.iterdir():
        raw = entry.stem if entry.is_file() else entry.name
        ids.add(raw.split("_")[0])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save patch bounding boxes for .svs files whose case ID appears in done_dir."
    )
    parser.add_argument(
        "input_dir",
        help="Root directory; .svs files are searched recursively inside its subdirectories.",
    )
    parser.add_argument(
        "done_dir",
        help="Directory whose entries represent already-reviewed case IDs.",
    )
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
    done_dir = Path(args.done_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_ids = collect_done_ids(done_dir)
    print(f"Done cases loaded: {len(done_ids)} IDs from {done_dir}")

    all_svs = sorted(input_dir.rglob("*.svs"))
    matched = [p for p in all_svs if p.stem in done_ids]

    print(f"Found {len(all_svs)} .svs files total; {len(matched)} match done cases.")
    if not matched:
        return

    for wsi_path in matched:
        process_wsi(
            wsi_path=wsi_path,
            output_dir=output_dir,
            patch_size_px=args.patch_size,
            overlap=args.overlap,
            thumbnail_scale=args.thumbnail_scale,
        )

    print(f"\nDone. {len(matched)} JSON files saved to {output_dir}")


if __name__ == "__main__":
    main()
