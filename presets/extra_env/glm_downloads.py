"""
FrameVision GLM-Image downloads helper.

- Clones the upstream repo (for examples/resources) into:
    <framevision_root>/models/glm-image/repo/GLM-Image

- Downloads the Hugging Face model snapshot into:
    <framevision_root>/models/glm-image/model/GLM-Image

You can provide a HF token (if ever required) via:
    set HF_TOKEN=...

This script is designed to be called from presets/extra_env/glm_install.bat.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def ensure_git() -> None:
    try:
        subprocess.check_output(["git", "--version"], stderr=subprocess.STDOUT)
    except Exception:
        print("ERROR: 'git' was not found on PATH. Install Git for Windows, then re-run.", file=sys.stderr)
        raise

def clone_repo(dest_dir: Path) -> None:
    ensure_git()
    url = "https://github.com/zai-org/GLM-Image.git"
    if (dest_dir / ".git").exists():
        print(f"Repo already present: {dest_dir}")
        return
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {url} -> {dest_dir}")
    run(["git", "clone", "--depth", "1", url, str(dest_dir)])

def download_model(dest_dir: Path) -> None:
    """
    Uses huggingface_hub snapshot_download (HTTP, no git-lfs required).
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        print("ERROR: huggingface_hub not installed in this environment.", file=sys.stderr)
        print("Re-run the installer so requirements install first.", file=sys.stderr)
        raise

    model_id = "zai-org/GLM-Image"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hugging Face model snapshot: {model_id}")
    print(f"Target: {dest_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    print("Model download complete.")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framevision-root", required=True, help="Absolute path to FrameVision root folder")
    args = ap.parse_args()

    fv_root = Path(args.framevision_root).resolve()
    models_root = fv_root / "models" / "glm-image"
    repo_dir = models_root / "repo" / "GLM-Image"
    model_dir = models_root / "model" / "GLM-Image"

    print(f"FrameVision root: {fv_root}")
    print(f"Models root:      {models_root}")

    # Ensure base dirs exist
    models_root.mkdir(parents=True, exist_ok=True)

    # Clone repo (optional but useful for examples/scripts)
    try:
        clone_repo(repo_dir)
    except Exception:
        # Don't fail hard if git isn't present; repo is optional
        print("WARNING: Repo clone failed (git missing?). Continuing without repo.", file=sys.stderr)

    # Download model (required)
    download_model(model_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
