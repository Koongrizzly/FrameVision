#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download ACE-Step 1.5 models into FrameVision's portable folders.

Targets:
  <root>\models\ace_step_15\repo\ACE-Step-1.5\checkpoints\

Always downloads (DiT path):
  - ACE-Step/Ace-Step1.5 : acestep-v15-turbo/**, vae/**, Qwen3-Embedding-0.6B/**

Optional LM:
  - 0: none
  - 1: ACE-Step/acestep-5Hz-lm-0.6B
  - 2: Ace-Step1.5 includes acestep-5Hz-lm-1.7B/** (download that folder)
  - 3: ACE-Step/acestep-5Hz-lm-4B
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lm", required=True, choices=["0","1","2","3"])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    # Store checkpoints inside the repo folder so ACE-Step can run fully portable
    # without needing a separate checkpoints directory.
    ckpt = root / "models" / "ace_step_15" / "repo" / "ACE-Step-1.5" / "checkpoints"
    cache_root = root / "cache" / "ace_step_15"
    ensure_dir(ckpt)
    ensure_dir(cache_root)

    # Force HF caches inside FrameVision
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hf_home" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf_home" / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch_home"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    # Import here so installer can ensure dependencies exist first
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[ERROR] huggingface_hub not available in this env. Install step may have failed.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    def snap(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
        ensure_dir(local_dir)
        print(f"\n[DOWNLOAD] {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        )

    # Base (DiT) bundle + (optionally) LM 1.7B folder
    patterns = [
        "acestep-v15-turbo/**",
        "vae/**",
        "Qwen3-Embedding-0.6B/**",
    ]
    if args.lm == "2":
        patterns.append("acestep-5Hz-lm-1.7B/**")

    # Skip if already present
    base_ok = (ckpt / "acestep-v15-turbo").exists() and (ckpt / "vae").exists() and (ckpt / "Qwen3-Embedding-0.6B").exists()
    if base_ok and (args.lm != "2" or (ckpt / "acestep-5Hz-lm-1.7B").exists()):
        print("[SKIP] Base models already present.")
    else:
        snap("ACE-Step/Ace-Step1.5", ckpt, allow_patterns=patterns)

    # Optional LM sizes not included in Ace-Step1.5 bundle
    if args.lm == "1":
        if (ckpt / "acestep-5Hz-lm-0.6B").exists():
            print("[SKIP] LM 0.6B already present.")
        else:
            snap("ACE-Step/acestep-5Hz-lm-0.6B", ckpt / "acestep-5Hz-lm-0.6B")
    elif args.lm == "3":
        if (ckpt / "acestep-5Hz-lm-4B").exists():
            print("[SKIP] LM 4B already present.")
        else:
            snap("ACE-Step/acestep-5Hz-lm-4B", ckpt / "acestep-5Hz-lm-4B")

    print("\n[OK] Model download finished.")
    print(f"Models are in: {ckpt}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
