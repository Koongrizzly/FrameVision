#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Headless verification for ACE-Step 1.5 install.
- No Gradio
- Confirms: imports + torch CUDA visibility + required model folders present

Exit codes:
  0 OK
  1 Generic failure
  2 Missing models
  3 Torch/CUDA not available (warning for GPU-only requirement)
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()

    # Checkpoints location can vary depending on installer version.
    # Old: <root>\\models\\ace_step_15\\checkpoints
    # New: <root>\\models\\ace_step_15\\repo\\ACE-Step-1.5\\checkpoints
    ckpt_old = root / "models" / "ace_step_15" / "checkpoints"
    ckpt_new = root / "models" / "ace_step_15" / "repo" / "ACE-Step-1.5" / "checkpoints"

    def has_required(base: Path) -> bool:
        return all((base / name).exists() for name in (
            "acestep-v15-turbo",
            "vae",
            "Qwen3-Embedding-0.6B",
        ))

    if has_required(ckpt_new):
        ckpt = ckpt_new
    elif has_required(ckpt_old):
        ckpt = ckpt_old
    else:
        ckpt = ckpt_new if ckpt_new.exists() else ckpt_old
    out_dir = root / "output" / "audio"
    cache_root = root / "cache" / "ace_step_15"

    # Ensure portable cache env vars are set even during verify
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hf_home" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf_home" / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch_home"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("PYTHONUTF8", "1")

    required = [
        ckpt / "acestep-v15-turbo",
        ckpt / "vae",
        ckpt / "Qwen3-Embedding-0.6B",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[ERROR] Missing required model folders:")
        for p in missing:
            print(" -", p)
        print("\n[INFO] Checked checkpoints base:")
        print(" -", ckpt)
        print("[INFO] Candidate locations:")
        print(" -", ckpt_new)
        print(" -", ckpt_old)
        return 2

    print("[OK] Required model folders found.")
    print("[INFO] Using checkpoints base:", ckpt)
    print(" -", ckpt / "acestep-v15-turbo")
    print(" -", ckpt / "vae")
    print(" -", ckpt / "Qwen3-Embedding-0.6B")

    # Imports
    try:
        import torch
    except Exception as e:
        print("[ERROR] Failed to import torch:", str(e), file=sys.stderr)
        return 1

    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    print(f"[INFO] torch.__version__ = {getattr(torch, '__version__', 'unknown')}")
    print(f"[INFO] CUDA available     = {cuda_ok}")
    if cuda_ok:
        try:
            print(f"[INFO] CUDA device count  = {torch.cuda.device_count()}")
            print(f"[INFO] CUDA device        = {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        print("[WARN] CUDA is NOT available in this environment.")
        print("       If you intended GPU-only install, check:")
        print("       - NVIDIA driver installed")
        print("       - torch CUDA build installed (cu128)")
        print("       - No conflicting torch packages")
        return 3

    try:
        import acestep  # noqa: F401
        print("[OK] Imported 'acestep'")
    except Exception as e:
        print("[ERROR] Failed to import 'acestep':", str(e), file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Output folder ready:", out_dir)
    print("[SUCCESS] Headless verification passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
