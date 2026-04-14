import os
import hashlib
import time
import json

def get_dir_snapshot(path):
    snapshot = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            stat = os.stat(file_path)
            # Calculate MD5 for absolute verification
            with open(file_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            snapshot.append({
                "rel_path": os.path.relpath(file_path, path),
                "size": stat.st_size,
                "mtime": time.ctime(stat.st_mtime),
                "md5": md5
            })
    return snapshot

def create_pre_op_snapshot():
    plots_path = "plots"
    snapshot = get_dir_snapshot(plots_path)
    with open("pre_op_manifest.json", "w") as f:
        json.dump(snapshot, f, indent=4)
    print(f"Pre-operation manifest created for {len(snapshot)} files.")

if __name__ == "__main__":
    create_pre_op_snapshot()
