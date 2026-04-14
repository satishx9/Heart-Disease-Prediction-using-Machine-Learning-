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

def verify_retention():
    plots_path = "plots"
    post_snapshot = get_dir_snapshot(plots_path)
    
    with open("post_op_manifest.json", "w") as f:
        json.dump(post_snapshot, f, indent=4)
        
    with open("pre_op_manifest.json", "r") as f:
        pre_snapshot = json.load(f)
        
    # Create comparison maps
    pre_map = {item['rel_path']: item for item in pre_snapshot}
    post_map = {item['rel_path']: item for item in post_snapshot}
    
    errors = []
    
    # Check for missing or changed files
    for rel_path, pre_item in pre_map.items():
        if rel_path not in post_map:
            errors.append(f"MISSING: {rel_path}")
        else:
            post_item = post_map[rel_path]
            if pre_item['md5'] != post_item['md5']:
                errors.append(f"MODIFIED: {rel_path}")
            elif pre_item['size'] != post_item['size']:
                errors.append(f"SIZE CHANGE: {rel_path}")
                
    # Check for new files
    for rel_path in post_map:
        if rel_path not in pre_map:
            errors.append(f"NEW FILE: {rel_path}")
            
    if not errors:
        print("VERIFICATION SUCCESS: Zero changes detected. All plots are intact.")
        print(f"Verified {len(post_snapshot)} files.")
    else:
        print("VERIFICATION FAILURE: Changes detected!")
        for err in errors:
            print(f" - {err}")

if __name__ == "__main__":
    verify_retention()
