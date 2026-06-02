"""
Upload experimental tracking artifacts to the model repo.

Target repo : Shiggii/qwen-incident-response-grpo  (repo_type="model")
Files       : trainer_state.json, training_args.bin (if present)

Usage:
    # Place trainer_state.json (and optionally training_args.bin) in the repo root
    # or pass extra directories to search:
    HF_TOKEN=hf_... python upload_trainer_state.py
    HF_TOKEN=hf_... python upload_trainer_state.py --search-dir ~/Downloads --search-dir ~/kaggle_output
"""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID   = "Shiggii/qwen-incident-response-grpo"
REPO_TYPE = "model"

TARGET_FILES = {
    "trainer_state.json": None,
    "training_args.bin":  None,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload trainer_state.json to the HF model repo.")
    ap.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Additional directories to search for files (repeatable)",
    )
    return ap.parse_args()


def build_search_dirs(extra_dirs: list[str]) -> list[str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [repo_root]
    for d in extra_dirs:
        search_dirs.append(os.path.abspath(os.path.expanduser(d)))
    return search_dirs


def discover_files(search_dirs: list[str]) -> dict[str, str | None]:
    found_paths: dict[str, str | None] = {fname: None for fname in TARGET_FILES}
    for fname in found_paths:
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                found_paths[fname] = candidate
                break
    return found_paths


args = parse_args()
SEARCH_DIRS = build_search_dirs(args.search_dir)
TARGET_FILES = discover_files(SEARCH_DIRS)

print("=== File discovery ===")
print(f"Search paths: {SEARCH_DIRS}")
for fname, path in TARGET_FILES.items():
    if path:
        size_kb = os.path.getsize(path) / 1024
        print(f"  FOUND   {fname:25s}  {size_kb:7.1f} KB  at {path}")
    else:
        print(f"  MISSING {fname}")

found = {k: v for k, v in TARGET_FILES.items() if v is not None}

if not found:
    print("\nNo files found. Download them from the Kaggle notebook Output tab first.")
    print("Place files in the repo root or pass --search-dir <path> (repeatable).")
    sys.exit(1)

# ── Confirm before uploading ──────────────────────────────────────────────────

print(f"\nTarget : {REPO_ID}  (repo_type={REPO_TYPE})")
print(f"Will upload: {list(found.keys())}")
confirm = input("\nProceed with upload? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)

# ── Upload ────────────────────────────────────────────────────────────────────

try:
    token = os.environ["HF_TOKEN"]
except KeyError:
    print("HF_TOKEN environment variable is required.")
    sys.exit(1)

api = HfApi(token=token)

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
