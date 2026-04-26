"""
Upload training plot PNGs to the HuggingFace Space repo via the Hub API.

Usage:
    HF_TOKEN=hf_... python upload_pngs.py
    -- or --
    python upload_pngs.py          (will prompt for token interactively)
"""

import os
import sys

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID   = "Shiggii/incident-response-detective"
REPO_TYPE = "space"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "reward_curve.png",
    "before_after.png",
    "loss_curve.png",
]

# ── Token ─────────────────────────────────────────────────────────────────────
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    print("HF_TOKEN not set in environment.")
    token = input("Paste your HuggingFace token (hf_...): ").strip()
if not token:
    print("No token provided. Exiting.")
    sys.exit(1)

# ── Upload ────────────────────────────────────────────────────────────────────
api = HfApi(token=token)

print(f"\nTarget : {REPO_ID}  (repo_type={REPO_TYPE})")
print(f"Files  : {FILES}\n")

for filename in FILES:
    local_path = os.path.join(REPO_ROOT, filename)

    if not os.path.exists(local_path):
        print(f"  SKIP  {filename}  (not found on disk)")
        continue

    size_kb = os.path.getsize(local_path) / 1024
    print(f"  Uploading {filename}  ({size_kb:.1f} KB) ...", end=" ", flush=True)

    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,       # repo root
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add training plot: {filename}",
        )
        print("OK")
    except Exception as e:
        print(f"FAILED\n    Error: {e}")

print("\nDone.")
