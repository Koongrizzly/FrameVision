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
import json
import os
import platform
import re
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

# Stable SeedVR2 runtime pins.
# Modern FrameVision baseline: Torch 2.8.0 with CUDA 12.8 wheels.
# Keep Transformers below the 4.57+ finegrained-FP8 import path that caused
# torch.float8_e8m0fnu crashes in SeedVR2.
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
TRANSFORMERS_VERSION = "4.55.4"
DIFFUSERS_VERSION = "0.35.1"
ACCELERATE_VERSION = "1.10.1"
TRITON_WINDOWS_SPEC = "triton-windows>=3.4,<3.5"
FLASH_ATTN_WIN_TORCH28_CU128_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2%2Bcu128torch2.8-cp311-cp311-win_amd64.whl"
SAGE_ATTENTION_WIN_TORCH28_CU128_URL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0%2Bcu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"

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

def _try_install_torch(venv_dir: Path) -> str:
    """
    Install a *matching* PyTorch/torchvision/torchaudio set.
    We pin companion versions to avoid pip backtracking into incompatible newer wheels.
    Returns torch index tag string (e.g. "cu124", "cu121", "cpu").
    """
    _print_header("Installing PyTorch")
    has_nvidia = False
    try:
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_nvidia = (p.returncode == 0)
    except Exception:
        has_nvidia = False

    # torch companion versions
    trio = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}"]

    if has_nvidia:
        print("Detected NVIDIA GPU (via nvidia-smi). Trying modern CUDA PyTorch wheels (preferred order: cu128 -> cu126).")
        for tag in ("cu128", "cu126"):
            try:
                _pip(venv_dir, [
                    "install", "--upgrade", "--force-reinstall",
                    *trio,
                    "--index-url", f"https://download.pytorch.org/whl/{tag}"
                ])
                return tag
            except subprocess.CalledProcessError:
                print(f"PyTorch install failed for {tag}; trying next option...")
    else:
        print("No NVIDIA GPU detected. Installing CPU PyTorch wheels.")

    _pip(venv_dir, ["install", "--upgrade", "--force-reinstall", *trio,
                    "--index-url", "https://download.pytorch.org/whl/cpu"])
    return "cpu"


def _detect_torch_env(venv_dir: Path) -> dict:
    py = _venv_python(venv_dir)
    code = (
        "import json, sys\n"
        "try:\n"
        " import torch\n"
        " d={\"python\":sys.version.split()[0],\"torch\":getattr(torch,\"__version__\",None),"
        "\"torch_cuda\":getattr(torch.version,\"cuda\",None),\"cuda_available\":bool(torch.cuda.is_available())}\n"
        " print(json.dumps(d))\n"
        "except Exception as e:\n"
        " print(json.dumps({\"error\":str(e),\"python\":sys.version.split()[0]}))\n"
    )
    cp = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    out = (cp.stdout or "").strip().splitlines()
    if not out:
        return {"error": (cp.stderr or "No output").strip()}
    try:
        return json.loads(out[-1])
    except Exception:
        return {"raw": out[-1], "stderr": (cp.stderr or "").strip()}


def _ask_yes_no(question: str, *, default: bool = False) -> bool:
    """Ask before optional installs. Non-interactive runs use the default."""
    if not sys.stdin or not sys.stdin.isatty():
        print(f"{question} {'[default: yes]' if default else '[default: no]'}")
        print("Non-interactive installer run detected; skipping optional install.")
        return default
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(question + suffix).strip().lower()
    except Exception:
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def _module_imports(venv_dir: Path, module_name: str) -> bool:
    py = _venv_python(venv_dir)
    cp = subprocess.run([str(py), "-c", f"import {module_name}; print('{module_name}: OK')"], capture_output=True, text=True)
    if cp.returncode == 0:
        print((cp.stdout or "").strip())
        return True
    return False


def _write_constraints(root_dir: Path, torch_tag: str) -> Path:
    """Prevent pip from drifting SeedVR2 into an incompatible runtime stack."""
    c = root_dir / "presets" / "extra_env" / "_seedvr2_constraints.txt"
    c.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if torch_tag == "cpu" else f"+{torch_tag}"
    c.write_text(
        "\n".join([
            f"torch=={TORCH_VERSION}{suffix}",
            f"torchvision=={TORCHVISION_VERSION}{suffix}",
            f"torchaudio=={TORCHAUDIO_VERSION}{suffix}",
            f"transformers=={TRANSFORMERS_VERSION}",
            f"diffusers=={DIFFUSERS_VERSION}",
            f"accelerate=={ACCELERATE_VERSION}",
            "",
        ]),
        encoding="utf-8",
    )
    return c


def _detect_triton_version(venv_dir: Path) -> dict:
    """Return Triton import/version info from the target env."""
    py = _venv_python(venv_dir)
    code = r"""
import json
try:
    import importlib.metadata as md
    import triton
    version = None
    for pkg in ("triton-windows", "triton"):
        try:
            version = md.version(pkg)
            break
        except Exception:
            pass
    print(json.dumps({"ok": True, "version": version or getattr(triton, "__version__", None)}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
""".strip()
    cp = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    out = (cp.stdout or "").strip().splitlines()
    if not out:
        return {"ok": False, "error": (cp.stderr or "No output").strip()}
    try:
        return json.loads(out[-1])
    except Exception:
        return {"ok": False, "raw": out[-1], "stderr": (cp.stderr or "").strip()}


def _version_tuple(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", str(v or ""))
    return tuple(int(x) for x in nums[:3]) if nums else (0,)


def _triton_meets_target(venv_dir: Path) -> tuple[bool, dict]:
    info = _detect_triton_version(venv_dir)
    if not info.get("ok"):
        return False, info
    version = str(info.get("version") or "")
    # Current FrameVision SeedVR2 Torch 2.8/cu128 target is triton-windows >=3.4,<3.5.
    ok = _version_tuple(version) >= (3, 4) and _version_tuple(version) < (3, 5)
    return ok, info


def _try_install_triton_windows(venv_dir: Path) -> bool:
    """Optional Triton install/upgrade. Only called after user approval."""
    _print_header("Installing Triton for Windows (optional)")
    try:
        _pip(venv_dir, ["uninstall", "-y", "triton"])
        _pip(venv_dir, ["uninstall", "-y", "triton-windows"])
        _pip(venv_dir, ["install", "--upgrade", "--force-reinstall", TRITON_WINDOWS_SPEC])
        ok, info = _triton_meets_target(venv_dir)
        if ok:
            print(f"triton: OK ({info.get('version')})")
            return True
        print(f"Triton installed/imported but version does not meet target {TRITON_WINDOWS_SPEC}: {info}")
        return False
    except Exception as exc:
        print(f"Triton install failed/skipped (non-fatal): {type(exc).__name__}: {exc}")
        return False


def _verify_seedvr2_stack(venv_dir: Path) -> None:
    """Fail during install if the core stack is incompatible."""
    py = _venv_python(venv_dir)
    code = f"""
import torch
import torchvision
import torchaudio
import transformers
import diffusers
import accelerate
from transformers import AutoImageProcessor

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
if torch.__version__.split("+", 1)[0] != "2.8.0":
    raise SystemExit("Wrong torch version; expected 2.8.0")
if torchvision.__version__.split("+", 1)[0] != "0.23.0":
    raise SystemExit("Wrong torchvision version; expected 0.23.0")
if torchaudio.__version__.split("+", 1)[0] != "2.8.0":
    raise SystemExit("Wrong torchaudio version; expected 2.8.0")
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("accelerate:", accelerate.__version__)
print("AutoImageProcessor: OK")

if transformers.__version__.split("+", 1)[0] != "{TRANSFORMERS_VERSION}":
    raise SystemExit("Wrong transformers version; expected {TRANSFORMERS_VERSION}")
if diffusers.__version__.split("+", 1)[0] != "{DIFFUSERS_VERSION}":
    raise SystemExit("Wrong diffusers version; expected {DIFFUSERS_VERSION}")
if accelerate.__version__.split("+", 1)[0] != "{ACCELERATE_VERSION}":
    raise SystemExit("Wrong accelerate version; expected {ACCELERATE_VERSION}")
"""
    _run([str(py), "-c", code])



def _try_install_flash_attn(venv_dir: Path) -> bool:
    """
    Optional FlashAttention install using direct prebuilt wheel URLs only.
    No generic pip package fallback and no source-build attempts.
    """
    _print_header("Attempting FlashAttention install (optional)")
    info = _detect_torch_env(venv_dir)
    print("Torch env info:", info)
    if info.get("error"):
        print("Skipping flash-attn install (could not inspect torch env).")
        return False
    if not info.get("cuda_available") or not info.get("torch_cuda"):
        print("Skipping flash-attn install (CUDA torch runtime not detected).")
        return False

    pyver = str(info.get("python", ""))
    tor = str(info.get("torch", ""))
    tcu = str(info.get("torch_cuda", ""))

    wheel_candidates = []
    if _is_windows() and pyver.startswith("3.11") and tor.startswith("2.8.0") and str(tcu).startswith("12.8"):
        wheel_candidates.append(FLASH_ATTN_WIN_TORCH28_CU128_URL)

    if not wheel_candidates:
        print("No direct prebuilt flash-attn wheel rule for this platform/combo. Skipping (non-fatal).")
        return False

    py = _venv_python(venv_dir)
    for url in wheel_candidates:
        try:
            print(f"Trying direct prebuilt FlashAttention wheel: {url}")
            _pip(venv_dir, ["uninstall", "-y", "flash-attn", "flash_attn"])
            _pip(venv_dir, ["install", "--no-deps", "--force-reinstall", "--no-cache-dir", url])
            cp = subprocess.run(
                [str(py), "-c", "import flash_attn; print(getattr(flash_attn,'__version__','unknown'))"],
                capture_output=True, text=True
            )
            if cp.returncode == 0:
                print("flash-attn installed OK (direct wheel):", (cp.stdout or "").strip())
                return True
            print("flash-attn import check failed after direct wheel install:", (cp.stderr or "").strip())
        except subprocess.CalledProcessError:
            print("Direct prebuilt FlashAttention wheel install failed for this URL.")

    print("FlashAttention direct-wheel install not available/working for this exact combo (non-fatal).")
    return False


def _try_install_sage_attention(venv_dir: Path) -> bool:
    """
    Optional SageAttention install using direct prebuilt wheel URL only.
    """
    _print_header("Installing SageAttention (optional)")
    info = _detect_torch_env(venv_dir)
    print("Torch env info:", info)
    pyver = str(info.get("python", ""))
    tor = str(info.get("torch", ""))
    tcu = str(info.get("torch_cuda", ""))
    if not (_is_windows() and pyver.startswith("3.11") and tor.startswith("2.8.0") and str(tcu).startswith("12.8")):
        print("No direct prebuilt SageAttention wheel rule for this platform/combo. Skipping (non-fatal).")
        return False
    try:
        _pip(venv_dir, ["uninstall", "-y", "sageattention"])
        _pip(venv_dir, ["install", "--no-deps", "--force-reinstall", "--no-cache-dir", SAGE_ATTENTION_WIN_TORCH28_CU128_URL])
        return _module_imports(venv_dir, "sageattention")
    except Exception as exc:
        print(f"SageAttention install failed/skipped (non-fatal): {type(exc).__name__}: {exc}")
        return False


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

def _install_requirements(venv_dir: Path, req_path: Path, *, try_flash_attn: bool = True, install_flash_attn: bool = False, install_triton: bool = False, install_sage: bool = False) -> dict:
    _print_header("Installing Python requirements")
    _pip(venv_dir, ["install", "--upgrade", "pip", "setuptools", "wheel"])

    torch_tag = _try_install_torch(venv_dir)
    root_dir = _detect_framevision_root()
    constraints = _write_constraints(root_dir, torch_tag)
    torch_index = "https://download.pytorch.org/whl/cpu" if torch_tag == "cpu" else f"https://download.pytorch.org/whl/{torch_tag}"

    _pip(venv_dir, [
        "install", "--upgrade", "--upgrade-strategy", "only-if-needed",
        "--extra-index-url", torch_index,
        "-c", str(constraints),
        "numpy", "tqdm", "pillow", "opencv-python", "safetensors",
        "einops", "huggingface_hub", "requests", "packaging",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"diffusers=={DIFFUSERS_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
    ])

    # gguf package is used by ComfyUI-GGUF and SeedVR2 GGUF flows.
    _pip(venv_dir, ["install", "--upgrade", "--upgrade-strategy", "only-if-needed", "-c", str(constraints), "gguf"])

    if req_path.exists():
        _pip(venv_dir, [
            "install", "--upgrade", "--upgrade-strategy", "only-if-needed",
            "--extra-index-url", torch_index,
            "-c", str(constraints),
            "-r", str(req_path),
        ])
    else:
        print("WARNING: requirements.txt not found; skipping -r install.")

    # Re-apply the sensitive pins after requirements.txt in case the repo file has loose deps.
    _pip(venv_dir, [
        "install", "--upgrade", "--upgrade-strategy", "only-if-needed",
        "--extra-index-url", torch_index,
        "-c", str(constraints),
        f"transformers=={TRANSFORMERS_VERSION}",
        f"diffusers=={DIFFUSERS_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
    ])

    triton_ok, triton_info = _triton_meets_target(venv_dir)
    if triton_ok:
        print(f"triton: OK ({triton_info.get('version')})")
    else:
        print(f"Triton missing or below target {TRITON_WINDOWS_SPEC}: {triton_info}")
        if install_triton or _ask_yes_no("Triton is missing or outdated. Install/upgrade optional triton-windows now?", default=False):
            triton_ok = _try_install_triton_windows(venv_dir)
        else:
            print("Triton optional install/upgrade skipped.")

    flash_ok = _module_imports(venv_dir, "flash_attn")
    if not flash_ok and try_flash_attn:
        if install_flash_attn or _ask_yes_no("FlashAttention is missing. Install optional FlashAttention now?", default=False):
            flash_ok = _try_install_flash_attn(venv_dir)
        else:
            print("FlashAttention optional install skipped.")

    sage_ok = _module_imports(venv_dir, "sageattention")
    if not sage_ok:
        if install_sage or _ask_yes_no("SageAttention is missing. Install optional SageAttention now?", default=False):
            sage_ok = _try_install_sage_attention(venv_dir)
        else:
            print("SageAttention optional install skipped.")

    _verify_seedvr2_stack(venv_dir)

    return {
        "torch_tag": torch_tag,
        "flash_attn": flash_ok,
        "triton": triton_ok,
        "triton_info": triton_info,
        "sageattention": sage_ok,
        "torch_info": _detect_torch_env(venv_dir),
        "constraints": str(constraints),
        "transformers": TRANSFORMERS_VERSION,
        "diffusers": DIFFUSERS_VERSION,
        "accelerate": ACCELERATE_VERSION,
    }

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

def _write_marker(root_dir: Path, models_dir: Path, venv_dir: Path, src_dir: Path, quant: str, install_info: Optional[dict] = None) -> None:
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
          "install_info": {json.dumps(install_info or {}, ensure_ascii=False)},
          "note": "Created by seedvr2_cu128_gguf_installer.py"
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
    parser.add_argument("--skip-flash-attn", action="store_true", help="Do not ask/install optional flash-attn.")
    parser.add_argument("--install-flash-attn", action="store_true", help="Explicitly install optional flash-attn if missing.")
    parser.add_argument("--install-triton", action="store_true", help="Explicitly install optional triton-windows if missing.")
    parser.add_argument("--install-sage", action="store_true", help="Explicitly install optional SageAttention if missing.")
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
    install_info = _install_requirements(env_dir, req_path, try_flash_attn=not args.skip_flash_attn, install_flash_attn=args.install_flash_attn, install_triton=args.install_triton, install_sage=args.install_sage)

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

    _write_marker(root_dir, models_dir, env_dir, src_dir, args.quant, install_info=install_info)

    _print_header("Done")
    print("Next steps (manual test idea):")
    print("1) Verify models exist in models/SEEDVR2/")
    print("2) Verify env imports: environments/.seedvr2/Scripts/python -c \"import torch; import gguf\"")
    print("3) Later: hook FrameVision to call the SeedVR2 CLI/node code in presets/extra_env/seedvr2_src")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
