"""
Download training artifacts from the Hugging Face model repo for local plotting / analysis.

Usage:
    HF_TOKEN=hf_...  python scripts/download_training_data.py
    (Token optional if the repo is public and no rate limit issues.)
"""
from __future__ import annotations

import os
import sys

REPO_ID = "Shiggii/qwen-incident-response-grpo"
FILENAME = "trainer_state.json"
REPO_TYPE = "model"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from e

    import shutil

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, FILENAME)

    token = os.environ.get("HF_TOKEN", "").strip() or None

    # Download to HF cache, then copy to repo so path is always data/trainer_state.json
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type=REPO_TYPE,
        token=token,
    )
    shutil.copy2(cached, target)

    size_kb = os.path.getsize(target) / 1024
    print(f"OK: {target}  ({size_kb:.1f} KB)")
    print(f"Source: {REPO_ID} / {FILENAME}")


if __name__ == "__main__":
    main()
