"""
Upload experimental tracking artifacts to the model repo.

Target repo : Shiggii/qwen-incident-response-grpo  (repo_type="model")
Files       : trainer_state.json, training_args.bin (if present)

Usage:
    # Put trainer_state.json (and optionally training_args.bin) in one of:
    #   - C:\\Users\\shikh\\Downloads\\
    #   - C:\\Users\\shikh\\Downloads\\kaggle_output\\
    #   - C:\\Users\\shikh\\Downloads\\kaggle_output\\checkpoint-400\\
    #   - Repo root (same folder as this script)
    # Then run (HF_TOKEN required — do not hardcode tokens in this file):
    HF_TOKEN=hf_... python upload_trainer_state.py
"""

import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID   = "Shiggii/qwen-incident-response-grpo"
REPO_TYPE = "model"
TOKEN     = os.environ["HF_TOKEN"]

# ── Search for files in likely locations ──────────────────────────────────────

SEARCH_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),              # repo root
    r"C:\Users\shikh\Downloads",
    r"C:\Users\shikh\Downloads\kaggle_output",
    r"C:\Users\shikh\Downloads\kaggle_output\checkpoint-400",
    r"C:\Users\shikh\Downloads\kaggle_output\checkpoint-384",
]

TARGET_FILES = {
    "trainer_state.json": None,
    "training_args.bin":  None,
}

for fname in TARGET_FILES:
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            TARGET_FILES[fname] = candidate
            break

print("=== File discovery ===")
for fname, path in TARGET_FILES.items():
    if path:
        size_kb = os.path.getsize(path) / 1024
        print(f"  FOUND   {fname:25s}  {size_kb:7.1f} KB  at {path}")
    else:
        print(f"  MISSING {fname}")

found = {k: v for k, v in TARGET_FILES.items() if v is not None}

if not found:
    print("\nNo files found. Download them from the Kaggle notebook Output tab first.")
    print("Expected locations: C:\\Users\\shikh\\Downloads\\ or subdirectories.")
    sys.exit(1)

# ── Confirm before uploading ──────────────────────────────────────────────────

print(f"\nTarget : {REPO_ID}  (repo_type={REPO_TYPE})")
print(f"Will upload: {list(found.keys())}")
confirm = input("\nProceed with upload? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)

# ── Upload ────────────────────────────────────────────────────────────────────

api = HfApi(token=TOKEN)

for fname, local_path in found.items():
    size_kb = os.path.getsize(local_path) / 1024
    print(f"\nUploading {fname}  ({size_kb:.1f} KB) ...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add trainer_state.json - experimental tracking artifacts",
        )
        print("OK")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")

print("\nDone. Verify at:")
print(f"  https://huggingface.co/Shiggii/qwen-incident-response-grpo/tree/main")
