#!/usr/bin/env python3
"""
FrameVision - FLUX.2 klein 4B downloader

Downloads model snapshot into:
  <FrameVisionRoot>/models/klein4b/

Notes:
- If the repo is gated, you must accept the license on Hugging Face and provide a token.
  Set one of these env vars before running:
    HF_TOKEN or HUGGINGFACE_TOKEN

This script is meant to be launched by klein_install.py *inside* the venv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = "black-forest-labs/FLUX.2-klein-4B"

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "models" / "klein4b"

def _env_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

def main() -> int:
    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Download target: {DEST}")

    DEST.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[ERROR] huggingface_hub not installed in this env.")
        print("        Re-run klein_install.py.")
        return 1

    token = _env_token()
    if not token:
        print("[NOTE] No HF token found (HF_TOKEN / HUGGINGFACE_TOKEN).")
        print("       If this model is gated for you, the download will fail.")
        print("       You can still proceed; if it fails, set the token and rerun.")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        print("\n[DONE] Model snapshot downloaded.")
        return 0
    except Exception as e:
        print("\n[ERROR] Download failed.")
        print("Reason:", e)
        print("\nTips:")
        print(" - Make sure you accepted the model license on Hugging Face.")
        print(" - Set HF_TOKEN (or HUGGINGFACE_TOKEN) to a valid token with access.")
        print(" - Ensure you have enough disk space.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
