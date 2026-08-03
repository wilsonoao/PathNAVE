#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import urllib.request

BASE_URL = "s3://camelyon-dataset/CAMELYON16/images"
HTTP_BASE = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images"
OUT_DIR = os.path.join(os.path.dirname(__file__), "slides")
CSV_PATH = os.path.join(os.path.dirname(__file__), "reference.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def remote_size(name):
    """Return expected file size from S3 via HTTP HEAD, or -1 on error."""
    try:
        req = urllib.request.Request(f"{HTTP_BASE}/{name}.tif", method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(r.headers.get("Content-Length", -1))
    except Exception:
        return -1


normal_slides = []
with open(CSV_PATH, newline="") as f:
    for row in csv.reader(f):
        if len(row) >= 2 and row[1].strip() == "Normal":
            normal_slides.append(row[0].strip())

print(f"Found {len(normal_slides)} Normal slides in reference.csv")
# normal_slides = normal_slides[:48]
print(f"Downloading first 48 Normal slides")

failed = []
for i, name in enumerate(normal_slides, 1):
    dest = os.path.join(OUT_DIR, f"{name}.tif")

    if os.path.exists(dest):
        local_size = os.path.getsize(dest)
        expected = remote_size(name)
        if expected > 0 and local_size == expected:
            print(f"[{i}/{len(normal_slides)}] SKIP {name}.tif (already complete, {local_size // 1024**2} MB)")
            continue
        else:
            print(f"[{i}/{len(normal_slides)}] Re-downloading {name}.tif (incomplete: {local_size // 1024**2} MB / {expected // 1024**2} MB)")

    print(f"[{i}/{len(normal_slides)}] Downloading {name}.tif ...", flush=True)
    result = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request",
         f"{BASE_URL}/{name}.tif", dest],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: failed to download {name}.tif", file=sys.stderr)
        failed.append(name)

print(f"\nDone. {len(normal_slides) - len(failed)} downloaded, {len(failed)} failed.")
if failed:
    print("Failed slides:", failed)
