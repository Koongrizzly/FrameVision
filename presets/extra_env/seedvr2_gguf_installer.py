#!/usr/bin/env python3
"""
SeedVR2 GGUF Installer (FrameVision optional install prep)

Creates:
- Environment: <framevision_root>/environments/.seedvr2
- Models:      <framevision_root>/models/SEEDVR2
- Source:      <framevision_root>/presets/extra_env/seedvr2_src  (SeedVR2 CLI/node code)

This script is designed to be runnable standalone.
Place this file in: <framevision_root>/presets/extra_env/
Then run: python seedvr2_gguf_installer.py
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---- Constants (URLs / repos) ----
SEEDVR2_NODE_REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_NODE_ZIP  = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/archive/refs/heads/main.zip"
SEEDVR2_REQ_RAW   = "https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/main/requirements.txt"

HF_GGUF_REPO = "cmeka/SeedVR2-GGUF"
HF_MAIN_REPO = "numz/SeedVR2_comfyUI"

GGUF_FILES_3B = {
    "Q3": "seedvr2_ema_3b-Q3_K_M.gguf",
    "Q4": "seedvr2_ema_3b-Q4_K_M.gguf",
    "Q5": "seedvr2_ema_3b-Q5_K_M.gguf",
    "Q6": "seedvr2_ema_3b-Q6_K.gguf",
    "Q8": "seedvr2_ema_3b-Q8_0.gguf",
}
VAE_FILE = "ema_vae_fp16.safetensors"

# ---- Helpers ----
def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")

def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")

def _venv_python(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def _pip(venv_dir: Path, args: List[str]) -> None:
    py = _venv_python(venv_dir)
    _run([str(py), "-m", "pip"] + args)

def _try_install_torch(venv_dir: Path) -> None:
    """
    Install PyTorch.
    Preference: CUDA wheels if an NVIDIA GPU seems present; fallback to CPU wheels.
    """
    _print_header("Installing PyTorch")
    # Heuristic: if nvidia-smi exists and returns 0, assume CUDA-capable GPU
    has_nvidia = False
    try:
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_nvidia = (p.returncode == 0)
    except Exception:
        has_nvidia = False

    # Try CUDA 12.1 wheels first when NVIDIA present, else CPU wheels.
    # (Keeps the installer simple and avoids requiring local CUDA toolkits.)
    if has_nvidia:
        print("Detected NVIDIA GPU (via nvidia-smi). Trying CUDA PyTorch wheels (cu121).")
        try:
            _pip(venv_dir, ["install", "--upgrade", "torch", "torchvision", "torchaudio",
                            "--index-url", "https://download.pytorch.org/whl/cu121"])
            return
        except subprocess.CalledProcessError:
            print("CUDA wheel install failed; falling back to CPU wheels.")
    else:
        print("No NVIDIA GPU detected. Installing CPU PyTorch wheels.")

    _pip(venv_dir, ["install", "--upgrade", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"])

def _download_file(url: str, dest: Path) -> None:
    """
    Download via python stdlib only (urllib) to avoid bootstrap dependency problems.
    """
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def _ensure_git() -> bool:
    return shutil.which("git") is not None

def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def _detect_framevision_root() -> Path:
    """
    Installer is intended to live at: <root>/presets/extra_env/seedvr2_gguf_installer.py
    So root is 2 levels up from this file's directory.
    """
    here = Path(__file__).resolve()
    # .../presets/extra_env/<this_file>
    return here.parents[2]

def _install_requirements(venv_dir: Path, req_path: Path) -> None:
    _print_header("Installing Python requirements")
    _pip(venv_dir, ["install", "--upgrade", "pip", "setuptools", "wheel"])
    _try_install_torch(venv_dir)
    _pip(venv_dir, ["install", "--upgrade",
                    "numpy", "tqdm", "pillow", "opencv-python", "safetensors",
                    "einops", "huggingface_hub", "requests", "packaging"])
    # gguf package is used by ComfyUI-GGUF and SeedVR2 GGUF flows
    _pip(venv_dir, ["install", "--upgrade", "gguf"])

    if req_path.exists():
        _pip(venv_dir, ["install", "-r", str(req_path)])
    else:
        print("WARNING: requirements.txt not found; skipping -r install.")


def _hf_download(venv_dir: Path, repo_id: str, filename: str, dest_dir: Path) -> Path:
    """
    Download a single file from Hugging Face with a simple stdout progress indicator.

    We intentionally use stdlib-only networking here so progress remains visible even
    when this installer is run from inside FrameVision (where TTY-based tqdm bars
    often don't render).
    """
    import urllib.request

    _print_header(f"Downloading model: {filename}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already exists: {dest_path}")
        return dest_path

    # Public Hugging Face resolve URL (works for public repos)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    def _fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
            n /= 1024
        return f"{n:.1f}TB"

    print(f"URL: {url}")
    print(f"Saving to: {dest_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "FrameVision-SeedVR2-Installer"})
    start_t = time.time()
    last_t = 0.0
    downloaded = 0
    total = 0

    with urllib.request.urlopen(req) as r:
        try:
            total = int(r.headers.get("Content-Length") or 0)
        except Exception:
            total = 0

        with open(tmp_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_t >= 0.2:
                    last_t = now
                    if total > 0:
                        pct = (downloaded / total) * 100.0
                        elapsed = max(0.001, now - start_t)
                        speed = downloaded / elapsed
                        sys.stdout.write(
                            f"\r  {pct:6.2f}%  ({_fmt_bytes(downloaded)}/{_fmt_bytes(total)})  {_fmt_bytes(int(speed))}/s"
                        )
                    else:
                        elapsed = max(0.001, now - start_t)
                        speed = downloaded / elapsed
                        sys.stdout.write(
                            f"\r  {_fmt_bytes(downloaded)} downloaded  {_fmt_bytes(int(speed))}/s"
                        )
                    sys.stdout.flush()

    # Newline after progress
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Finalize
    try:
        if dest_path.exists():
            dest_path.unlink()
    except Exception:
        pass
    tmp_path.replace(dest_path)

    print(f"Done: {dest_path} ({_fmt_bytes(dest_path.stat().st_size)})")
    return dest_path


def _clone_or_download_seedvr2_src(src_dir: Path) -> Tuple[Path, Path]:
    """
    Ensure SeedVR2 node repo is present. Returns (repo_root, requirements_txt_path).
    """
    _print_header("Fetching SeedVR2 source (CLI/node)")
    src_dir.mkdir(parents=True, exist_ok=True)

    repo_root = src_dir / "ComfyUI-SeedVR2_VideoUpscaler"
    if repo_root.exists():
        print(f"Source already present: {repo_root}")
    else:
        if _ensure_git():
            try:
                _run(["git", "clone", "--depth", "1", SEEDVR2_NODE_REPO, str(repo_root)])
            except subprocess.CalledProcessError:
                print("Git clone failed; falling back to zip download.")
        if not repo_root.exists():
            tmp_zip = src_dir / "seedvr2_node.zip"
            _download_file(SEEDVR2_NODE_ZIP, tmp_zip)
            _extract_zip(tmp_zip, src_dir)
            # github zip extracts as .../ComfyUI-SeedVR2_VideoUpscaler-main
            extracted = src_dir / "ComfyUI-SeedVR2_VideoUpscaler-main"
            if extracted.exists() and not repo_root.exists():
                extracted.rename(repo_root)
            try:
                tmp_zip.unlink()
            except Exception:
                pass

    req = repo_root / "requirements.txt"
    if not req.exists():
        # As a fallback, download raw requirements into repo_root
        try:
            _download_file(SEEDVR2_REQ_RAW, req)
        except Exception:
            pass
    return repo_root, req

def _write_marker(root_dir: Path, models_dir: Path, venv_dir: Path, src_dir: Path, quant: str) -> None:
    marker = root_dir / "presets" / "extra_env" / "seedvr2_gguf_install.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        textwrap.dedent(f"""\
        {{
          "engine": "SeedVR2",
          "format": "GGUF",
          "quant": "{quant}",
          "models_dir": "{models_dir.as_posix()}",
          "env_dir": "{venv_dir.as_posix()}",
          "src_dir": "{src_dir.as_posix()}",
          "note": "Created by seedvr2_gguf_installer.py"
        }}
        """),
        encoding="utf-8"
    )
    print(f"Wrote marker: {marker}")

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="seedvr2_gguf_installer.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Install SeedVR2 GGUF runtime + models into FrameVision folders."
    )
    parser.add_argument("--quant", default="Q4", choices=["Q3", "Q4", "Q5", "Q6", "Q8"],
                        help="GGUF quant level for the 3B model (default: Q4).")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads (env + deps only).")
    parser.add_argument("--force", action="store_true", help="Recreate venv even if it already exists.")
    args = parser.parse_args()

    _print_header("SeedVR2 GGUF Installer")
    root_dir = _detect_framevision_root()
    print("FrameVision root:", root_dir)

    # Target dirs (as requested)
    env_dir = root_dir / "environments" / ".seedvr2"
    models_dir = root_dir / "models" / "SEEDVR2"
    src_dir = root_dir / "presets" / "extra_env" / "seedvr2_src"

    print("\nTargets:")
    print("  Env:   ", env_dir)
    print("  Models:", models_dir)
    print("  Source:", src_dir)

    # Create / recreate env
    if env_dir.exists() and args.force:
        _print_header("Removing existing environment (force)")
        shutil.rmtree(env_dir, ignore_errors=True)

    if not env_dir.exists():
        _print_header("Creating virtual environment")
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "venv", str(env_dir)])
    else:
        print("Environment already exists; reusing.")

    # Fetch source code and requirements
    repo_root, req_path = _clone_or_download_seedvr2_src(src_dir)

    # Install deps
    _install_requirements(env_dir, req_path)

    # Models
    if not args.skip_models:
        models_dir.mkdir(parents=True, exist_ok=True)
        gguf_name = GGUF_FILES_3B[args.quant]
        # GGUF DiT
        _hf_download(env_dir, HF_GGUF_REPO, gguf_name, models_dir)
        # VAE
        _hf_download(env_dir, HF_MAIN_REPO, VAE_FILE, models_dir)

        # Optional: patch file helpful for GGUF workflows (stored alongside)
        try:
            _hf_download(env_dir, HF_GGUF_REPO, "lcpp-seedvr.patch", models_dir)
        except Exception:
            print("Could not download lcpp-seedvr.patch (non-fatal).")

    _write_marker(root_dir, models_dir, env_dir, src_dir, args.quant)

    _print_header("Done")
    print("Next steps (manual test idea):")
    print("1) Verify models exist in models/SEEDVR2/")
    print("2) Verify env imports: environments/.seedvr2/Scripts/python -c \"import torch; import gguf\"")
    print("3) Later: hook FrameVision to call the SeedVR2 CLI/node code in presets/extra_env/seedvr2_src")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
