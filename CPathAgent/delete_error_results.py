"""
Delete JSON result files where the "error" field is not null.

Usage:
    python delete_error_results.py <directory> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Delete JSON files where error != null")
    p.add_argument("directory", help="Directory to scan for .json files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without actually deleting")
    args = p.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(root.rglob("*.json"))
    print(f"Found {len(json_files)} JSON file(s) in {root}")

    deleted = 0
    skipped = 0
    parse_errors = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP] Cannot parse {path.name}: {e}")
            parse_errors += 1
            continue

        if data.get("error") is not None or data.get("prediction") is "":
            if args.dry_run:
                print(f"  [DRY-RUN] Would delete: {path}")
            else:
                path.unlink()
                # print(f"  [DELETED] {path}")
            deleted += 1
        else:
            skipped += 1

    print(f"\nDone.")
    print(f"  {'Would delete' if args.dry_run else 'Deleted'}: {deleted}")
    print(f"  Kept        : {skipped}")
    print(f"  Parse errors: {parse_errors}")


if __name__ == "__main__":
    main()
