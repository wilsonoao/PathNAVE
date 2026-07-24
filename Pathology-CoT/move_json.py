import os
import shutil
import uuid
import argparse

def process(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for d in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, d)

        if not os.path.isdir(dir_path):
            continue

        src_file = os.path.join(dir_path, "analysis_results.json")
        if not os.path.exists(src_file):
            continue

        # 取 test_001
        parts = d.split("_")
        if len(parts) < 2:
            continue

        prefix = parts[0] + "_" + parts[1]

        # unique id 防重複
        unique_id = uuid.uuid4().hex[:6]

        new_filename = f"{prefix}_{unique_id}.json"
        dst_file = os.path.join(output_dir, new_filename)

        shutil.copy2(src_file, dst_file)

        # print(f"[OK] {d} -> {new_filename}")

    print(len(os.listdir(root_dir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    process(args.input_dir, args.output_dir)