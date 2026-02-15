#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACE-Step 1.5 one-click installer for FrameVision (Windows).

BAT does one thing: runs this file.
This script auto-detects FrameVisionRoot based on its own location:
  FrameVisionRoot\presets\extra_env\ace_step_15_installer.py

It then installs (portable/offline runtime expectations):
- venv:    FrameVisionRoot\environments\.ace_15
- repo:    FrameVisionRoot\models\ace_step_15\repo\ACE-Step-1.5
- models:  FrameVisionRoot\models\ace_step_15\checkpoints\...
- caches:  FrameVisionRoot\cache\ace_step_15\...
- output:  FrameVisionRoot\output\audio\

LM is OPTIONAL:
- Default is DiT-only (lm_choice=0).
- Change presets\setsave\ace_step_15.json to set lm_choice to 1/2/3 if desired.
"""

from __future__ import annotations
import os, sys, subprocess, shutil, textwrap, argparse, zipfile, urllib.request, re
from pathlib import Path

TORCH_INDEX_CU129 = "https://download.pytorch.org/whl/cu129"
TORCH_PIN = "2.8.0"
NANO_VLLM_GIT_URL = "https://github.com/GeeeekExplorer/nano-vllm.git"
NANO_VLLM_ZIP_URL = "https://github.com/GeeeekExplorer/nano-vllm/archive/refs/heads/main.zip"

LM_MAP = {
    0: ("dit_only", None),
    1: ("lm_0_6B", "ACE-Step/acestep-5Hz-lm-0.6B"),
    2: ("lm_1_7B", "ACE-Step/Ace-Step1.5"),  # folder inside base repo snapshot
    3: ("lm_4B",   "ACE-Step/acestep-5Hz-lm-4B"),
}

def log(msg: str) -> None:
    print(msg, flush=True)

def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log(">>> " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")

def which(exe: str) -> str | None:
    return shutil.which(exe)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_zip(url: str, out_zip: Path) -> None:
    log(f"Downloading: {url}")
    with urllib.request.urlopen(url) as r, open(out_zip, "wb") as f:
        shutil.copyfileobj(r, f)

def extract_zip(zip_path: Path, dest: Path) -> None:
    log(f"Extracting: {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

def portable_cache_env(root: Path) -> dict[str, str]:
    cache_root = root / "cache" / "ace_step_15"
    ensure_dir(cache_root)
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(cache_root / "hf_home"),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hf_home" / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "hf_home" / "transformers"),
        "XDG_CACHE_HOME": str(cache_root / "xdg_cache"),
        "TORCH_HOME": str(cache_root / "torch_home"),
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PIP_CACHE_DIR": str(cache_root / "pip_cache"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONUTF8": "1",
        "TEMP": str(cache_root / "tmp"),
        "TMP": str(cache_root / "tmp"),
    })
    return env

def detect_root_from_self() -> Path:
    # This file should live in: <root>\presets\extra_env\
    here = Path(__file__).resolve()
    extra_env = here.parent
    presets = extra_env.parent
    root = presets.parent
    return root

def load_lm_choice(root: Path) -> int:
    cfg = root / "presets" / "setsave" / "ace_step_15.json"
    if cfg.exists():
        try:
            import json
            data = json.loads(cfg.read_text(encoding="utf-8"))
            v = int(data.get("lm_choice", 0))
            return v if v in (0,1,2,3) else 0
        except Exception:
            return 0
    return 0



def patch_ace_pyproject_windows_compat(repo_dir: Path) -> None:
    """
    Patch ACE-Step's local pyproject.toml for Windows + FrameVision headless runtime.

    Fixes:
    - Remove nano-vllm dependency (pulls official triton>=3 and other deps that break pip on Windows).
    - Remove gradio / fastapi (not required for FrameVision headless subprocess runner).
    - Remove strict torch pin like 'torch==2.7.1+cu128; sys_platform == "win32"' so it won't fight our pinned Torch.
    We install our Torch stack explicitly (cu129) before installing ACE-Step.
    """
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return

    s = pyproject.read_text(encoding="utf-8")
    original = s

    # Drop specific problematic deps
    for dep in ("nano-vllm", "gradio", "fastapi"):
        s = re.sub(r'\n\s*"' + re.escape(dep) + r'[^"]*"\s*,?\s*', "\n", s)

    # Drop strict torch/vision/audio pins with CUDA local version markers
    # Examples:
    #   "torch==2.7.1+cu128; sys_platform == \"win32\"",
    #   "torchvision==0.xx+cu128; sys_platform == \"win32\"",
    #   "torchaudio==...+cu128; sys_platform == \"win32\"",
    s = re.sub(r'\n\s*"torch[^"]*\+cu\d+[^"]*"\s*,?\s*', "\n", s)
    s = re.sub(r'\n\s*"torchvision[^"]*\+cu\d+[^"]*"\s*,?\s*', "\n", s)
    s = re.sub(r'\n\s*"torchaudio[^"]*\+cu\d+[^"]*"\s*,?\s*', "\n", s)

    # Also remove any plain torch==x.y.z+cuNNN that might appear without quotes variants
    s = re.sub(r'\n\s*"torch==[^"]*"\s*,?\s*', lambda m: "\n" if "+cu" in m.group(0) else m.group(0), s)

    if s != original:
        pyproject.write_text(s, encoding="utf-8")
        log("[PATCH] Patched ACE-Step pyproject.toml (Windows): removed nano-vllm/gradio/fastapi and torch+cu pins.")

def ensure_nano_vllm(root: Path, vpip: list[str], env: dict[str, str]) -> None:
    """
    ACE-Step's pyproject depends on 'nano-vllm', but it's not published on PyPI.
    Install it from GitHub so pip can satisfy the dependency on Windows.
    """
    # If already importable, skip.
    try:
        import nano_vllm  # type: ignore
        log("[SKIP] nano-vllm already installed.")
        return
    except Exception:
        pass

    # Windows workaround:
    # - nano-vllm requires 'triton>=3', but the official 'triton' package has no Windows wheels.
    # - Many Windows AI stacks use 'triton-windows', which provides the 'triton' module.
    # - pip dependency resolution won't match 'triton-windows' to 'triton', so we install nano-vllm with --no-deps.
    if sys.platform.startswith("win"):
        # triton-windows installed earlier in core deps step
        log('[STEP] Installing nano-vllm from GitHub (no-deps, Windows)...')
        tmp = root / 'cache' / 'ace_step_15' / 'tmp'
        ensure_dir(tmp)
        git = which('git')
        if git:
            run(vpip + ['install', '--no-deps', f'git+{NANO_VLLM_GIT_URL}'], env=env)
            return
        zip_path = tmp / 'nano-vllm_main.zip'
        if not zip_path.exists():
            download_zip(NANO_VLLM_ZIP_URL, zip_path)
        extract_zip(zip_path, tmp)
        extracted = tmp / 'nano-vllm-main'
        if not extracted.exists():
            raise RuntimeError('Unexpected nano-vllm zip layout: nano-vllm-main folder not found.')
        run(vpip + ['install', '--no-deps', str(extracted)], env=env)
        return

    log("[STEP] Installing nano-vllm from GitHub (required by ACE-Step)...")
    tmp = root / "cache" / "ace_step_15" / "tmp"
    ensure_dir(tmp)

    git = which("git")
    if git:
        run(vpip + ["install", f"git+{NANO_VLLM_GIT_URL}"], env=env)
        return

    # No git: download zip and install from extracted folder
    zip_path = tmp / "nano-vllm_main.zip"
    if not zip_path.exists():
        download_zip(NANO_VLLM_ZIP_URL, zip_path)
    extract_zip(zip_path, tmp)
    extracted = tmp / "nano-vllm-main"
    if not extracted.exists():
        raise RuntimeError("Unexpected nano-vllm zip layout: nano-vllm-main folder not found.")
    run(vpip + ["install", str(extracted)], env=env)

def ensure_repo(root: Path, repo_dir: Path, env: dict[str, str]) -> None:
    if repo_dir.exists() and ((repo_dir / ".git").exists() or (repo_dir / "pyproject.toml").exists()):
        log(f"[SKIP] Repo already present: {repo_dir}")
        return
    ensure_dir(repo_dir.parent)
    git = which("git")
    if git:
        log("[STEP] Cloning ACE-Step-1.5 repo (depth=1)...")
        run([git, "clone", "--depth", "1", "https://github.com/ace-step/ACE-Step-1.5.git", str(repo_dir)], env=env)
    else:
        log("[WARN] git not found - using zip download.")
        tmp = root / "cache" / "ace_step_15" / "tmp"
        ensure_dir(tmp)
        zip_url = "https://github.com/ace-step/ACE-Step-1.5/archive/refs/heads/main.zip"
        zip_path = tmp / "ACE-Step-1.5_main.zip"
        if not zip_path.exists():
            download_zip(zip_url, zip_path)
        extract_zip(zip_path, tmp)
        extracted = tmp / "ACE-Step-1.5-main"
        if not extracted.exists():
            raise RuntimeError("Unexpected zip layout: ACE-Step-1.5-main folder not found.")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.move(str(extracted), str(repo_dir))

def download_models(root: Path, vpy: Path, lm_choice: int, env: dict[str, str]) -> None:
    dl = (Path(__file__).parent / "download_models_ace_step_15.py").resolve()
    if not dl.exists():
        raise RuntimeError(f"Missing downloader script: {dl}")
    run([str(vpy), str(dl), "--root", str(root), "--lm", str(lm_choice)], env=env)

def verify(root: Path, vpy: Path, env: dict[str, str]) -> None:
    vf = (Path(__file__).parent / "verify_ace_step_15.py").resolve()
    if not vf.exists():
        raise RuntimeError(f"Missing verifier script: {vf}")
    run([str(vpy), str(vf), "--root", str(root)], env=env)

def main() -> int:
    root = detect_root_from_self()
    log(f"[INFO] Detected FrameVision root: {root}")

    if not (root / "models").exists():
        log("[ERROR] Root detection failed (missing 'models' folder).")
        log("Make sure this installer is located at:")
        log(r"  <FrameVisionRoot>\presets\extra_env\ace_step_15_installer.py")
        return 2

    env = portable_cache_env(root)

    lm_choice = load_lm_choice(root)
    log(f"[INFO] lm_choice (from presets/setsave/ace_step_15.json): {lm_choice} ({LM_MAP[lm_choice][0]})")
    log("       (LM is optional. Keep 0 for DiT-only.)")

    env_dir = root / "environments" / ".ace_15"
    models_root = root / "models" / "ace_step_15"
    repo_dir = models_root / "repo" / "ACE-Step-1.5"
    ensure_dir(models_root)

    # 1) venv
    vpy = env_dir / "Scripts" / "python.exe"
    if vpy.exists():
        log(f"[SKIP] Venv already exists: {env_dir}")
    else:
        log(f"[STEP] Creating venv: {env_dir}")
        run([sys.executable, "-m", "venv", str(env_dir)], env=env)

    vpy = env_dir / "Scripts" / "python.exe"
    vpip = [str(vpy), "-m", "pip"]

    # 2) pip tooling
    log("[STEP] Upgrading pip/setuptools/wheel...")
    run(vpip + ["install", "--upgrade", "pip", "setuptools", "wheel"], env=env)

    # 3) repo
    ensure_repo(root, repo_dir, env)
    if sys.platform.startswith('win'):
        patch_ace_pyproject_windows_compat(repo_dir)


    # 4) deps (pip only).
    if not sys.platform.startswith('win'):
        ensure_nano_vllm(root, vpip, env)
    # Normalize core deps for Windows compatibility (known-good recipe)
    log("[STEP] Ensuring Torch CUDA stack (pinned) + torchsde + triton-windows...")
    # Uninstall any existing torch stack (ignore failures)
    try:
        run(vpip + ["uninstall", "-y", "torch", "torchvision", "torchaudio"], env=env)
    except Exception:
        pass
    # Install pinned torch stack from CUDA 12.9 index
    run(vpip + ["install", f"torch=={TORCH_PIN}", "torchvision", "torchaudio", "--index-url", TORCH_INDEX_CU129], env=env)
    # Extra deps commonly needed by Triton/nano-vllm stacks
    run(vpip + ["install", "torchsde"], env=env)
    run(vpip + ["install", f"triton-windows<3.5"], env=env)

    log("[STEP] Installing ACE-Step (pip, editable)...")
    try:
        run(vpip + ["install", "--extra-index-url", TORCH_INDEX_CU129, "-e", "."], cwd=repo_dir, env=env)
    except Exception:
        log("[WARN] Editable install failed. Trying to install torch CUDA stack first, then retry...")
        # best-effort torch install (cu128 index)
        run(vpip + ["install", "--index-url", TORCH_INDEX_CU129, "torch", "torchvision", "torchaudio"], env=env)
        run(vpip + ["install", "--extra-index-url", TORCH_INDEX_CU129, "-e", "."], cwd=repo_dir, env=env)

    # 4b) Stabilize Diffusers/TorchAO on Windows.
    # Recent Diffusers releases introduced a TorchAO quantizer import path that can crash at import-time
    # when torchao is present but incompatible (e.g., torchao 0.16.x with torch 2.8.0+cu129),
    # resulting in: ModuleNotFoundError (torchao...uint4_layout) -> NameError: logger is not defined.
    # ACE-Step does NOT require torchao for FrameVision headless generation, so we remove it and pin diffusers.
    if sys.platform.startswith("win"):
        log("[STEP] Stabilizing Diffusers (remove torchao + pin diffusers==0.35.1)...")
        try:
            run(vpip + ["uninstall", "-y", "torchao"], env=env)
        except Exception:
            pass
        try:
            run(vpip + ["uninstall", "-y", "torchao-nightly"], env=env)
        except Exception:
            pass
        # Pin to a known-good pre-0.36 series to avoid the logger bug.
        run(vpip + ["install", "--upgrade", "--force-reinstall", "diffusers==0.35.1"], env=env)


        # Transformers in ACE-Step expects huggingface-hub < 1.0. Some installs accidentally pull hub 1.x,
        # causing ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found huggingface-hub==1.x
        log("[STEP] Stabilizing HuggingFace Hub (pin huggingface-hub==0.35.1)...")
        run(vpip + ["install", "--upgrade", "--force-reinstall", "huggingface-hub==0.35.1"], env=env)


        # Numba (used by ACE-Step alignment scorer) currently requires NumPy <= 2.3.
        # Some installs pull NumPy 2.4+, causing: ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.
        log("[STEP] Stabilizing NumPy/Numba (pin numpy==2.3.3 + reinstall numba)...")
        run(vpip + ["install", "--upgrade", "--force-reinstall", "numpy==2.3.3"], env=env)
        run(vpip + ["install", "--upgrade", "--force-reinstall", "numba"], env=env)



    # 5) models (base always; LM optional)
    log("[STEP] Downloading models (base always; LM optional)...")
    download_models(root, vpy, lm_choice, env)

    # 6) headless verify
    log("[STEP] Headless verification (no Gradio)...")
    verify(root, vpy, env)

    log("\n[OK] ACE-Step 1.5 install complete.")
    return 0

if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except KeyboardInterrupt:
        log("\n[ABORTED] Cancelled by user.")
        rc = 130
    except Exception as e:
        log("\n[FAILED] " + str(e))
        rc = 1

    # Keep the window open for double-click runs
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass
    raise SystemExit(rc)
