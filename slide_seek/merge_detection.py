import os
import json
import glob

QA_DIR = os.path.join(os.path.dirname(__file__), "qa_results")
OUTPUT = os.path.join(os.path.dirname(__file__), "DetectionLocalization_merged.json")

files = sorted(glob.glob(os.path.join(QA_DIR, "*_DetectionLocalization.json")))

merged = []
for fpath in files:
    with open(fpath) as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            merged.append({
                "slide_id": item.get("slide_id"),
                "predicted_presence": item.get("predicted_presence"),
                "gt_presence": item.get("gt_presence"),
            })
    else:
        merged.append({
            "slide_id": data.get("slide_id"),
            "predicted_presence": data.get("predicted_presence"),
            "gt_presence": data.get("gt_presence"),
        })

with open(OUTPUT, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(files)} files → {len(merged)} records")
print(f"Saved: {OUTPUT}")
