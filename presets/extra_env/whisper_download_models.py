#!/usr/bin/env python3
"""Download Faster-Whisper models into FrameVision's local models folder.

Defaults to downloading 'medium' into:
  <project_root>/models/faster_whisper/medium/

Usage:
  python whisper_download_models.py
  python whisper_download_models.py --model small
  python whisper_download_models.py --model large-v3 --force

Models map to the Systran faster-whisper repos on Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise SystemExit(
        "Missing dependency 'huggingface_hub'.\n"
        "Install it with: pip install huggingface_hub\n\n"
        f"Error: {type(e).__name__}: {e}"
    )

MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # presets/extra_env -> project root

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", choices=sorted(MODEL_REPOS.keys()))
    ap.add_argument("--force", action="store_true", help="re-download even if folder exists")
    args = ap.parse_args()

    root = project_root()
    models_dir = root / "models" / "faster_whisper"
    out_dir = models_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = MODEL_REPOS[args.model]

    print(f"[Whisper] Downloading model '{args.model}' ({repo_id})")
    print(f"[Whisper] Target folder: {out_dir}")

    if any(out_dir.iterdir()) and not args.force:
        print("[Whisper] Folder is not empty; skipping. Use --force to re-download.")
        return 0

    # Download snapshot into the target folder
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*"],
    )

    print("[Whisper] Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
