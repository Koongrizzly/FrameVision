from __future__ import annotations

"""Unified FrameVision LTX 2.5 installer / repair script.

This is the single authoritative installer for:
  - native FP16/BF16 distilled LTX 2.5
  - W4A8 ConvRot + isolated ComfyUI backend
  - INT4 ConvRot + isolated ComfyUI backend
  - Licon Multiple Subject Reference (MSR) model

All Python environments are UV-managed. Conda is intentionally not used.
The FP16 layout remains environments/ltx25 + models/ltx-2.5.
The ConvRot layout remains environments/ltx25_convrot + models/ltx_2_5_convrot.
"""

import argparse
import concurrent.futures
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import threading
import time
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENV_RELATIVE = Path("environments") / "ltx25"
MODEL_ROOT_RELATIVE = Path("models") / "ltx-2.5"
REPO_RELATIVE = MODEL_ROOT_RELATIVE / "LTX-2"
FFMPEG_BIN_RELATIVE = Path("presets") / "bin"
TEMP_RELATIVE = Path("temp")

OFFICIAL_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
HF_REPO = "Lightricks/LTX-2.5"

PYTHON_VERSION = "3.12"

# Native FP16 follows the known-working UV sync path used by FrameVision's
# previous install_ltx_2_5_distilled.bat.  The official LTX repo resolves its
# own Torch-compatible dependency set; we do not force the obsolete Conda/Torch
# 2.7 branch here.
SAGEATTN_WINDOWS_WHEEL = (
    "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/"
    "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
)

BASE_PACKAGES: Sequence[str] = (
    "huggingface_hub>=0.34",
    "opencv-python",
    "PySide6",
)


# Strict BF16 allow-list. No repository snapshot_download is used anywhere.
DEFAULT_MODEL_FILES: Sequence[str] = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)

OPTIONAL_CONV_VAE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
OPTIONAL_TEMPORAL_UPSCALER = "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
OPTIONAL_DURATION_HEAD = "model_patches/ltx-2.5-duration-head-bf16.safetensors"

# Hard-deny tokens provide an additional guard against accidentally adding an
# unwanted quantized/distilled-LoRA file to the allow-list later.
FORBIDDEN_MODEL_TOKENS: Sequence[str] = (
    "int8",
    "int4",
    "fp8",
    "convrot",
    "dev-transformer",
    "distilled-lora",
    "/loras/",
    "ic-lora",
)

FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
MODEL_DOWNLOAD_WORKERS_DEFAULT = 3
MODEL_DOWNLOAD_RETRIES = 3
_PRINT_LOCK = threading.Lock()

# Canonical ConvRot layout. Never shares the FP16 model/env folders.
CONVROT_ENV_RELATIVE = Path("environments") / "ltx25_convrot"
CONVROT_MODEL_ROOT_RELATIVE = Path("models") / "ltx_2_5_convrot"
CONVROT_REPO = "Winnougan/ltx-2.5-w4a8-convrot-int4-convrot-Winnougan-Blessing"

CONVROT_W4A8_FILES: Sequence[str] = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors",
)
CONVROT_INT4_FILES: Sequence[str] = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-int4_convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-int4_convrot.safetensors",
)
CONVROT_COMMON_FILES: Sequence[str] = (
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)

MSR_REPO = "LiconStudio/LTX-2.5-Multiple-Subject-Reference"
MSR_FILE = "LTX-2.5-Licon-MSR-V1.safetensors"
COMFY_ZIP_URL = "https://github.com/Comfy-Org/ComfyUI/archive/refs/heads/master.zip"
COMFY_EXCLUDE_EXACT = {
    "torch", "torchvision", "torchaudio",
    "comfyui_frontend_package", "comfyui_workflow_templates", "comfyui_embedded_docs",
}

def _installer_uv(root: Path) -> Path:
    """Return FrameVision's portable uv, bootstrapping it when missing."""
    candidates = [
        root / "presets" / "bin" / "uv" / "uv.exe",
        root / "presets" / "extra_env" / "tools" / "uv" / "uv.exe",
    ]
    for uv in candidates:
        if uv.is_file():
            status("FOUND", f"uv: {uv}")
            return uv

    target_dir = candidates[0].parent
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = root / TEMP_RELATIVE / "ltx25_uv_bootstrap"
    temp_dir.mkdir(parents=True, exist_ok=True)
    archive = temp_dir / "uv_windows.zip"
    url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    status("DOWNLOADING", "Portable uv")
    req = urllib.request.Request(url, headers={"User-Agent": "FrameVision-LTX25-Installer/3.0"})
    with urllib.request.urlopen(req, timeout=180) as response, open(archive, "wb") as fh:
        shutil.copyfileobj(response, fh, length=8 * 1024 * 1024)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    try:
        archive.unlink()
    except OSError:
        pass
    uv = target_dir / "uv.exe"
    if not uv.is_file():
        # Some release archives may contain one top-level directory.
        found = list(target_dir.rglob("uv.exe"))
        if found:
            found[0].replace(uv)
    if not uv.is_file():
        raise RuntimeError(f"uv bootstrap completed but uv.exe was not found under {target_dir}")
    status("OK", f"Portable uv installed: {uv}")
    return uv

def _hf_direct_url(repo: str, rel: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{rel}?download=true"

def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "FrameVision-LTX25-Installer/2.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        try:
            token = token_file.read_text(encoding="utf-8").strip()
        except Exception:
            token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _remote_file_size(url: str) -> tuple[Optional[int], str]:
    req = urllib.request.Request(url, headers=_hf_headers(), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            size = response.headers.get("Content-Length")
            return (int(size) if size else None), response.geturl()
    except Exception:
        req = urllib.request.Request(
            url,
            headers={**_hf_headers(), "Range": "bytes=0-0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1]), response.geturl()
            size = response.headers.get("Content-Length")
            return (int(size) if size else None), response.geturl()

def _download_direct(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, final_url = _remote_file_size(url)
    if dest.is_file() and size is not None and dest.stat().st_size == size:
        status("FOUND", f"Model file: {dest}")
        return

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = _hf_headers()
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(final_url, headers=headers)
    status("DOWNLOADING", f"{dest.name}")
    with urllib.request.urlopen(request, timeout=180) as response, open(part, "ab") as fh:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            fh.write(block)
    part.replace(dest)
    if size is not None and dest.stat().st_size != size:
        raise RuntimeError(
            f"Downloaded size mismatch for {dest.name}: "
            f"{dest.stat().st_size} != {size}"
        )
    status("OK", f"Model file: {dest}")

def _ensure_convrot_repo(root: Path, model_root: Path) -> Path:
    repo_path = model_root / "LTX-2"
    if repo_has_expected_packages(repo_path):
        status("FOUND", f"LTX 2.5 repo: {repo_path}")
        return repo_path
    return ensure_repo(root, repo_path, update_repo=False)

def install_convrot(root: Path, mode: str) -> int:
    model_root = root / CONVROT_MODEL_ROOT_RELATIVE
    env_path = root / CONVROT_ENV_RELATIVE
    model_root.mkdir(parents=True, exist_ok=True)

    status("OK", f"FrameVision root: {root}")
    status("OK", f"ConvRot env: {env_path}")
    status("OK", f"ConvRot model root: {model_root}")
    status("OK", "FP16 paths are not touched by ConvRot installation")

    _ensure_convrot_repo(root, model_root)
    files = CONVROT_W4A8_FILES if mode == "w4a8" else CONVROT_INT4_FILES

    checked = []
    missing = []
    for rel in files:
        url = _hf_direct_url(CONVROT_REPO, rel)
        try:
            _remote_file_size(url)
            checked.append((rel, url))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(rel)
            else:
                raise

    if missing:
        status("UNAVAILABLE", f"{mode.upper()} is not fully published upstream")
        for rel in missing:
            status("MISSING", rel)
        return 2

    for rel, url in checked:
        _download_direct(url, model_root / rel)
    for rel in CONVROT_COMMON_FILES:
        _download_direct(_hf_direct_url(HF_REPO, rel), model_root / rel)

    uv = _installer_uv(root)
    py = env_path / "Scripts" / "python.exe"
    if not py.is_file():
        run([uv, "venv", "--python", "3.12", env_path], cwd=root, check=True)

    run(
        [uv, "pip", "install", "--python", py,
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu130"],
        cwd=root, check=True,
    )
    run(
        [uv, "pip", "install", "--python", py,
         "comfy-kitchen>=0.2.24", "comfy-aimdo", "torchsde",
         "safetensors", "transformers", "sentencepiece", "protobuf",
         "numpy", "pillow", "einops", "psutil", "pyyaml", "av",
         "soundfile", "scipy", "tqdm"],
        cwd=root, check=True,
    )
    install_comfy_backend(root, model_root, env_path)
    status("OK", f"{mode.upper()} ConvRot install/repair complete (models + isolated ComfyUI backend)")
    return 0


def root_from_script() -> Path:
    # Expected normal location: <root>/presets/extra_env/ltx2_5_install.py
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


def status(kind: str, msg: str) -> None:
    with _PRINT_LOCK:
        print(f"[{kind}] {msg}", flush=True)


def quote_cmd(cmd: Sequence[object]) -> str:
    out = []
    for part in cmd:
        text = str(part)
        out.append(f'"{text}"' if any(ch in text for ch in " \t&()") else text)
    return " ".join(out)


def run(cmd: Sequence[object], *, cwd: Path, env: Optional[dict[str, str]] = None, check: bool = False) -> int:
    print("\n>>> " + quote_cmd(cmd), flush=True)
    completed = subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, text=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {quote_cmd(cmd)}")
    return int(completed.returncode)


def run_capture(cmd: Sequence[object], *, cwd: Path, env: Optional[dict[str, str]] = None) -> tuple[int, str, str]:
    completed = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return int(completed.returncode), completed.stdout.strip(), completed.stderr.strip()


def portable_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    temp_dir = root / TEMP_RELATIVE
    cache_dir = temp_dir / "cache"
    for p in (temp_dir, cache_dir, cache_dir / "hf", cache_dir / "torch", cache_dir / "pip"):
        p.mkdir(parents=True, exist_ok=True)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(cache_dir / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hf" / "hub")
    env["TORCH_HOME"] = str(cache_dir / "torch")
    env["PIP_CACHE_DIR"] = str(cache_dir / "pip")
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    return env


def find_env_python(env_path: Path) -> Optional[Path]:
    candidates = [
        env_path / "python.exe",
        env_path / "Scripts" / "python.exe",
        env_path / "bin" / "python",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def assert_nvidia_present() -> None:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        raise RuntimeError("nvidia-smi was not found. Refusing CPU-only Torch fallback.")
    rc = subprocess.run([nvidia], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        raise RuntimeError("nvidia-smi exists but failed. Check the NVIDIA driver before installing LTX 2.5.")
    status("OK", "NVIDIA driver probe passed")


def create_or_repair_env(root: Path, env_path: Path, *, recreate: bool) -> Path:
    """Create/repair a plain UV venv. Conda is never used."""
    uv = _installer_uv(root)
    existing = find_env_python(env_path)
    if existing and not recreate:
        status("FOUND", f"LTX 2.5 env Python: {existing}")
        return existing
    if recreate and env_path.exists():
        status("WARN", f"Deleting ONLY the LTX 2.5 env: {env_path}")
        shutil.rmtree(env_path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    status("DOWNLOADING", f"Creating UV environment with Python {PYTHON_VERSION}: {env_path}")
    run([uv, "python", "install", PYTHON_VERSION], cwd=root, env=portable_env(root), check=True)
    run([uv, "venv", "--python", PYTHON_VERSION, env_path], cwd=root, env=portable_env(root), check=True)
    py = find_env_python(env_path)
    if not py:
        raise RuntimeError(f"UV created the environment but Python was not found under {env_path}")
    return py

def _uv_project_env(root: Path, env_path: Path) -> dict[str, str]:
    env = portable_env(root)
    cache = root / MODEL_ROOT_RELATIVE / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    env.update({
        "UV_PROJECT_ENVIRONMENT": str(env_path),
        "UV_PYTHON_INSTALL_DIR": str(root / "environments" / "uv-python"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_XET_CACHE": str(cache / "huggingface" / "xet"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "XDG_CACHE_HOME": str(cache),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    return env


def install_fp16_uv_runtime(root: Path, env_path: Path, repo_path: Path, *, recreate: bool) -> Path:
    """Port of the previously working UV BAT installer into Python."""
    uv = _installer_uv(root)
    if recreate and env_path.exists():
        status("WARN", f"Deleting ONLY the LTX 2.5 env: {env_path}")
        shutil.rmtree(env_path)
    env = _uv_project_env(root, env_path)
    status("DOWNLOADING", f"Preparing UV Python {PYTHON_VERSION}")
    run([uv, "python", "install", PYTHON_VERSION], cwd=root, env=env, check=True)
    status("DOWNLOADING", "Synchronizing official LTX-2 workspace into environments/ltx25")
    run([uv, "sync", "--python", PYTHON_VERSION], cwd=repo_path, env=env, check=True)
    py = find_env_python(env_path)
    if not py:
        raise RuntimeError(f"UV sync completed but Python was not found in {env_path}")

    # Keep the exact acceleration path that the working BAT used.  These are
    # optional optimizations; PySide6 is required for standalone helper use.
    rc, _, _ = run_capture([py, "-c", "import triton"], cwd=root, env=env)
    if rc != 0:
        status("DOWNLOADING", "Triton-Windows")
        run([uv, "pip", "install", "--python", py, "triton-windows<3.7"], cwd=root, env=env, check=True)
    else:
        status("FOUND", "Triton-Windows")

    rc, _, _ = run_capture([py, "-c", "import sageattention"], cwd=root, env=env)
    if rc != 0:
        status("DOWNLOADING", "SageAttention Windows wheel")
        run([uv, "pip", "install", "--python", py, SAGEATTN_WINDOWS_WHEEL], cwd=root, env=env, check=True)
    else:
        status("FOUND", "SageAttention")

    rc, _, _ = run_capture([py, "-c", "import PySide6"], cwd=root, env=env)
    if rc != 0:
        status("DOWNLOADING", "PySide6")
        run([uv, "pip", "install", "--python", py, "PySide6"], cwd=root, env=env, check=True)

    run([py, "-c", "import torch, triton, sageattention; print('[VERIFY] torch', torch.__version__); print('[VERIFY] cuda', torch.version.cuda); print('[VERIFY] triton', triton.__version__)"], cwd=root, env=env, check=True)
    return py


def _requirement_name(line: str) -> str:
    text = line.strip()
    if not text or text.startswith("#") or text.startswith("-"):
        return ""
    left = text.split(";", 1)[0].strip()
    for sep in (" @ ", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in left:
            left = left.split(sep, 1)[0].strip()
            break
    if "[" in left:
        left = left.split("[", 1)[0].strip()
    return left.lower().replace("-", "_")


def install_comfy_backend(root: Path, model_root: Path, env_path: Path) -> None:
    """Install/update the isolated ComfyUI backend required by ConvRot."""
    uv = _installer_uv(root)
    py = find_env_python(env_path)
    if not py:
        raise RuntimeError(f"ConvRot environment Python not found: {env_path}")
    comfy_dir = model_root / "ComfyUI"
    status("DOWNLOADING", "Current isolated ComfyUI backend for LTX 2.5 ConvRot")
    req = urllib.request.Request(COMFY_ZIP_URL, headers={"User-Agent": "FrameVision-LTX25-Installer/3.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        payload = response.read()
    with tempfile.TemporaryDirectory(prefix="ltx25_comfy_", dir=str(model_root)) as td:
        temp = Path(td)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(temp)
        roots = [x for x in temp.iterdir() if x.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Unexpected ComfyUI archive layout")
        staging = model_root / "ComfyUI.new"
        old = model_root / "ComfyUI.old"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(roots[0], staging)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if comfy_dir.exists():
            comfy_dir.replace(old)
        staging.replace(comfy_dir)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)

    req_file = comfy_dir / "requirements.txt"
    if not req_file.is_file():
        raise RuntimeError(f"ComfyUI requirements.txt not found: {req_file}")
    selected = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _requirement_name(line) in COMFY_EXCLUDE_EXACT:
            continue
        selected.append(line)
    existing = {_requirement_name(x) for x in selected}
    for needed in ("comfy-kitchen>=0.2.24", "comfy-aimdo>=0.4.13", "torchsde"):
        if _requirement_name(needed) not in existing:
            selected.append(needed)
    filtered = model_root / "_comfy_backend_requirements.txt"
    filtered.write_text("\n".join(selected) + "\n", encoding="utf-8")
    try:
        run([uv, "pip", "install", "--python", py, "-r", filtered], cwd=root, check=True)
    finally:
        try:
            filtered.unlink()
        except OSError:
            pass
    run([py, "-c", "import torch, comfy_kitchen, comfy_aimdo, torchsde; print('[VERIFY] ConvRot Comfy backend imports OK; torch=' + torch.__version__)"], cwd=root, check=True)
    status("OK", f"ConvRot ComfyUI backend ready: {comfy_dir}")


def install_msr(root: Path) -> int:
    """Install/repair the Licon Multiple Subject Reference model."""
    model_root = root / MODEL_ROOT_RELATIVE
    dest = model_root / "msr" / MSR_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    status("OK", "Installing LTX 2.5 Licon Multiple Subject Reference model")
    _download_direct(_hf_direct_url(MSR_REPO, MSR_FILE), dest)
    if not dest.is_file() or dest.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"MSR model is missing or unexpectedly small: {dest}")
    status("OK", f"Licon MSR ready: {dest}")
    return 0


def repo_has_expected_packages(path: Path) -> bool:
    return (path / "packages" / "ltx-core").is_dir() and (path / "packages" / "ltx-pipelines").is_dir()


def ensure_repo(root: Path, repo_path: Path, *, update_repo: bool) -> Path:
    git = shutil.which("git")
    if repo_has_expected_packages(repo_path):
        status("FOUND", f"LTX 2.5 repo: {repo_path}")
        if update_repo:
            if not git:
                status("WARN", "Git is not available, so the existing LTX repo cannot be updated")
            elif (repo_path / ".git").exists():
                status("DOWNLOADING", "Updating official LTX-2 repository")
                run([git, "fetch", "--depth", "1", "origin", "main"], cwd=repo_path, env=portable_env(root), check=True)
                run([git, "reset", "--hard", "origin/main"], cwd=repo_path, env=portable_env(root), check=True)
            else:
                status("WARN", "Existing repo is valid but is not a Git checkout; leaving it unchanged")
        return repo_path

    if not git:
        raise RuntimeError(f"Git was not found and the LTX-2 repo is missing. Expected: {repo_path}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if repo_path.exists() and any(repo_path.iterdir()):
        raise RuntimeError(f"Repo path exists but is not a valid LTX-2 repo. Refusing to delete it: {repo_path}")
    status("DOWNLOADING", "Cloning current official Lightricks/LTX-2 repository")
    run([git, "clone", "--depth", "1", OFFICIAL_REPO_URL, repo_path], cwd=root, env=portable_env(root), check=True)
    if not repo_has_expected_packages(repo_path):
        raise RuntimeError(f"Downloaded repository is missing ltx-core/ltx-pipelines: {repo_path}")
    return repo_path


def _check_allowed_model_file(filename: str) -> None:
    norm = "/" + filename.replace("\\", "/").lower()
    for token in FORBIDDEN_MODEL_TOKENS:
        if token in norm:
            raise RuntimeError(f"Safety guard refused non-BF16/default model file: {filename} (matched {token!r})")
    if not filename.lower().endswith(".safetensors"):
        raise RuntimeError(f"Model allow-list contains an unexpected non-safetensors file: {filename}")


def hf_file_present(model_root: Path, filename: str) -> bool:
    p = model_root / Path(filename)
    return p.exists() and p.stat().st_size > 1024 * 1024


def _hf_access_probe(py: Path, root: Path, filename: str, *, hf_token: Optional[str]) -> tuple[bool, str]:
    """HEAD-only access check for one gated file; does not start the large download."""
    token_expr = repr(hf_token) if hf_token else "True"
    code = f"""
from huggingface_hub import get_hf_file_metadata, hf_hub_url
url = hf_hub_url(repo_id={HF_REPO!r}, filename={filename!r})
meta = get_hf_file_metadata(url, token={token_expr})
print(getattr(meta, 'size', None) or 'access-ok')
"""
    env = portable_env(root)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    rc, out, err = run_capture([py, "-c", code], cwd=root, env=env)
    return rc == 0, out or err


def ensure_hf_auth(py: Path, root: Path, files: Sequence[str], *, supplied_token: Optional[str]) -> Optional[str]:
    """Authenticate once before parallel downloads and verify gated LTX-2.5 access."""
    probe_file = files[0]

    # Explicit --hf-token wins. We validate it but never print it.
    if supplied_token:
        status("AUTH", "Checking supplied Hugging Face token against gated LTX 2.5 files")
        ok, detail = _hf_access_probe(py, root, probe_file, hf_token=supplied_token)
        if ok:
            status("OK", "Hugging Face authentication verified; LTX 2.5 gated files are accessible")
            return supplied_token
        raise RuntimeError(
            "The supplied Hugging Face token cannot access Lightricks/LTX-2.5. "
            "Make sure it belongs to the account that accepted the LTX 2.5 terms and has read access. "
            f"Access probe failed: {detail.splitlines()[-1] if detail else 'unknown error'}"
        )

    # First try the portable HF_HOME token saved by an earlier run/login.
    status("AUTH", "Checking saved Hugging Face login in FrameVision's portable HF cache")
    ok, _detail = _hf_access_probe(py, root, probe_file, hf_token=None)
    if ok:
        status("OK", "Saved Hugging Face login verified; LTX 2.5 gated files are accessible")
        return None

    status("AUTH", "LTX 2.5 access is approved on the website, but this installer is not authenticated yet")
    status("AUTH", "Starting one-time Hugging Face login. The login is saved under FrameVision temp\\cache\\hf and reused automatically.")

    env = portable_env(root)
    # Prefer the official CLI. Current huggingface_hub uses browser/device auth and stores the token in HF_HOME.
    candidates: list[list[object]] = []
    if os.name == "nt":
        hf_exe = py.parent / "Scripts" / "hf.exe"
        if hf_exe.exists():
            candidates.append([hf_exe, "auth", "login"])
    candidates.append([py, "-c", "from huggingface_hub import login; login(skip_if_logged_in=False)"])

    login_succeeded = False
    for cmd in candidates:
        try:
            rc = run(cmd, cwd=root, env=env, check=False)
        except Exception:
            rc = 1
        if rc == 0:
            login_succeeded = True
            break

    if not login_succeeded:
        raise RuntimeError(
            "Hugging Face login was not completed. LTX 2.5 is gated, so the installer needs a one-time authenticated "
            "HF session. Re-run the installer and complete the login prompt, or pass --hf-token with a personal read token."
        )

    ok, detail = _hf_access_probe(py, root, probe_file, hf_token=None)
    if not ok:
        tail = detail.splitlines()[-1] if detail else "unknown error"
        raise RuntimeError(
            "Hugging Face login succeeded, but this account still cannot read Lightricks/LTX-2.5. "
            "Use the same Hugging Face account that accepted the LTX 2.5 access terms. "
            f"Access probe failed: {tail}"
        )

    status("OK", "Hugging Face login and gated LTX 2.5 access verified")
    return None


def download_hf_file(py: Path, root: Path, model_root: Path, filename: str, *, hf_token: Optional[str]) -> None:
    _check_allowed_model_file(filename)
    target = model_root / Path(filename)
    if hf_file_present(model_root, filename):
        status("FOUND", f"Model file: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    token_expr = repr(hf_token) if hf_token else "True"
    code = f"""
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id={HF_REPO!r},
    filename={filename!r},
    local_dir={str(model_root)!r},
    token={token_expr},
)
print(p)
"""

    last_error: Optional[Exception] = None
    for attempt in range(1, MODEL_DOWNLOAD_RETRIES + 1):
        status("DOWNLOADING", f"BF16 whitelist file [{attempt}/{MODEL_DOWNLOAD_RETRIES}]: {filename}")
        try:
            dl_env = portable_env(root)
            if hf_token:
                dl_env["HF_TOKEN"] = hf_token
            run([py, "-c", code], cwd=root, env=dl_env, check=True)
            if hf_file_present(model_root, filename):
                status("OK", f"Downloaded: {filename}")
                return
            last_error = RuntimeError(f"Hugging Face returned successfully but expected file is missing: {target}")
        except Exception as exc:
            last_error = exc
        if attempt < MODEL_DOWNLOAD_RETRIES:
            status("WARN", f"Download failed; retrying with resume: {filename}")
            time.sleep(min(2 * attempt, 5))

    raise RuntimeError(
        f"Failed to download {filename} after {MODEL_DOWNLOAD_RETRIES} attempts. "
        f"Authentication was already verified before the download pool started, so this is likely a network/download error. "
        f"Original error: {last_error}"
    )



def selected_model_files(args: argparse.Namespace) -> list[str]:
    files = list(DEFAULT_MODEL_FILES)
    if args.with_conv_vae:
        files.append(OPTIONAL_CONV_VAE)
    if args.with_temporal_upscaler:
        files.append(OPTIONAL_TEMPORAL_UPSCALER)
    if args.with_duration_head:
        files.append(OPTIONAL_DURATION_HEAD)
    for filename in files:
        _check_allowed_model_file(filename)
    return files


def write_manifest(model_root: Path, files: Sequence[str]) -> None:
    manifest = model_root / "FRAMEVISION_LTX2_5_BF16_INSTALL.txt"
    text = [
        "FrameVision LTX 2.5 BF16 model manifest",
        "",
        "Downloaded/required by this install:",
    ]
    text += [f"  {f}" for f in files]
    text += [
        "",
        "Deliberately excluded:",
        "  dev/full transformer",
        "  INT8 / INT4 / FP8 / ConvRot variants",
        "  distilled LoRA",
        "  IC-LoRAs",
        "  repository snapshot downloads",
        "",
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(text), encoding="utf-8")


def ensure_models(py: Path, root: Path, model_root: Path, args: argparse.Namespace) -> None:
    files = selected_model_files(args)
    pending = [f for f in files if not hf_file_present(model_root, f)]
    for filename in files:
        if filename not in pending:
            status("FOUND", f"Model file: {model_root / Path(filename)}")

    workers = max(1, min(int(args.model_download_workers), len(pending) if pending else 1))
    status("OK", f"Strict BF16 whitelist; no snapshot download. Parallel model downloads: {workers}")

    active_token = args.hf_token
    if pending:
        active_token = ensure_hf_auth(py, root, pending, supplied_token=args.hf_token)

    errors: list[tuple[str, Exception]] = []
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ltx25-hf") as pool:
            future_map = {
                pool.submit(download_hf_file, py, root, model_root, filename, hf_token=active_token): filename
                for filename in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                filename = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append((filename, exc))
                    status("FAILED", f"Model download failed: {filename}: {exc}")

    if errors:
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(f"One or more model downloads failed after retry/resume: {names}")

    # Native LTX 2.5 spatial latent upscaler is part of DEFAULT_MODEL_FILES
    # and is downloaded into models/ltx-2.5/latent_upscale_models.
    write_manifest(model_root, files)


def ensure_ffmpeg(root: Path, *, skip_downloads: bool) -> None:
    bin_dir = root / FFMPEG_BIN_RELATIVE
    needed = [bin_dir / "ffmpeg.exe", bin_dir / "ffprobe.exe", bin_dir / "ffplay.exe"]
    if all(p.exists() and p.stat().st_size > 0 for p in needed):
        status("FOUND", "FFmpeg tools in presets\\bin")
        return
    if os.name != "nt":
        status("WARN", "Portable FFmpeg auto-download is Windows-only in this installer")
        return
    if skip_downloads:
        status("SKIPPED", "FFmpeg download skipped")
        return
    temp_dir = root / TEMP_RELATIVE / "ffmpeg_ltx2_5"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_dir / "ffmpeg-release-essentials.zip"
    status("DOWNLOADING", "FFmpeg essentials")
    urllib.request.urlretrieve(FFMPEG_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Optional[Path]] = {name: None for name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")}
    for p in temp_dir.rglob("*.exe"):
        low = p.name.lower()
        if low in found and found[low] is None:
            found[low] = p
    for name, src in found.items():
        if src is None:
            raise RuntimeError(f"Could not find {name} inside the FFmpeg bundle")
        shutil.copy2(src, bin_dir / name)
    shutil.rmtree(temp_dir, ignore_errors=True)
    status("OK", "Portable FFmpeg tools ready")


def verify(py: Path, root: Path, repo_path: Path, model_root: Path, files: Sequence[str]) -> int:
    failed = False

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failed
        status("OK" if ok else "FAILED", label + (f": {detail}" if detail else ""))
        if not ok:
            failed = True

    check("Separate env exists", py.exists(), str(py))
    check("Official LTX repo exists", repo_has_expected_packages(repo_path), str(repo_path))

    checks = {
        "Torch CUDA available": "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)",
        "ltx_core import": "import ltx_core; print('ltx_core OK')",
        "ltx_pipelines import": "import ltx_pipelines; print('ltx_pipelines OK')",
        "Gemma4-capable Transformers": "import transformers; from transformers import Gemma4UnifiedTextModel; print(transformers.__version__, Gemma4UnifiedTextModel.__name__)",
        "OpenImageIO import": "import OpenImageIO as oiio; print(getattr(oiio, '__version__', 'OK'))",
        "OpenCV import": "import cv2; print(cv2.__version__)",
        "Triton import": "import triton; print(getattr(triton, '__version__', 'unknown'))",
        "SageAttention import": "import sageattention; from importlib.metadata import version; print(version('sageattention'))",
    }
    for label, code in checks.items():
        rc, out, err = run_capture([py, "-c", code], cwd=root, env=_uv_project_env(root, root / ENV_RELATIVE))
        detail = out or (err.splitlines()[-1] if err else "")
        # Torch/LTX/model imports are required. Triton/Sage are installed by the
        # normal path too, so verification treats them as required as the old BAT did.
        check(label, rc == 0, detail)

    for filename in files:
        fp = model_root / Path(filename)
        check(f"Model {filename}", fp.exists() and fp.stat().st_size > 1024 * 1024, str(fp))

    return 20 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified FrameVision LTX 2.5 installer / repair")
    ap.add_argument(
        "--action",
        choices=("install-fp16", "install-w4a8", "install-int4", "install-msr", "verify-fp16"),
        default="install-fp16",
        help="Installer action. One Python file owns FP16, ConvRot/ComfyUI and MSR.",
    )
    ap.add_argument("--root", default=None, help="FrameVision root. Normally auto-detected from presets/extra_env.")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--skip-model-downloads", action="store_true")
    ap.add_argument("--skip-ffmpeg", action="store_true")
    ap.add_argument("--update-repo", action="store_true", help="Update an existing LTX-2 checkout to current origin/main")
    ap.add_argument("--danger-recreate-env", action="store_true", help="Delete/recreate ONLY the UV environment environments/ltx25; never deletes models")
    ap.add_argument("--confirm-recreate-env", default="")
    ap.add_argument("--hf-token", default=None, help="Optional Hugging Face read token. Prefer normal HF login/cache when possible.")
    ap.add_argument("--model-download-workers", type=int, default=MODEL_DOWNLOAD_WORKERS_DEFAULT, help="Number of simultaneous whitelisted model-file downloads (default: 3)")
    ap.add_argument("--with-conv-vae", action="store_true", help="Also download the optional lighter BF16 convolutional video VAE")
    ap.add_argument("--with-temporal-upscaler", action="store_true", help="Also download the optional BF16 temporal latent upscaler")
    ap.add_argument("--with-duration-head", action="store_true", help="Also download the optional BF16 duration head")
    args = ap.parse_args()
    if args.model_download_workers < 1 or args.model_download_workers > 8:
        raise RuntimeError("--model-download-workers must be between 1 and 8")

    # Root comes from this Python file when --root is omitted.
    # The caller's current working directory never changes installation paths.
    root = Path(args.root).resolve() if args.root else root_from_script()

    legacy_locations = (
        root / "models" / "ltx_2_5",
        root / "environments" / ".ltx2_5",
        root / "environments" / "ltx25_fp16",
    )
    for legacy in legacy_locations:
        if legacy.exists():
            status("LEGACY", f"Ignored old path (never created/used): {legacy}")

    if args.action == "install-w4a8":
        return install_convrot(root, "w4a8")
    if args.action == "install-int4":
        return install_convrot(root, "int4")
    if args.action == "install-msr":
        return install_msr(root)
    if args.action == "verify-fp16":
        args.verify_only = True

    env_path = root / ENV_RELATIVE
    model_root = root / MODEL_ROOT_RELATIVE
    repo_path = root / REPO_RELATIVE
    files = selected_model_files(args)

    for d in (root / TEMP_RELATIVE, model_root, repo_path.parent, root / FFMPEG_BIN_RELATIVE):
        d.mkdir(parents=True, exist_ok=True)

    status("OK", f"FrameVision root: {root}")
    status("OK", f"Separate LTX 2.5 env: {env_path}")
    status("OK", f"Separate LTX 2.5 model root: {model_root}")
    status("OK", f"Separate LTX 2.5 repo: {repo_path}")
    status("OK", "Model policy: distilled BF16 transformer only; no dev transformer, no INT8/INT4/FP8, no distilled LoRA")

    if args.danger_recreate_env and args.confirm_recreate_env != "DELETE_ENV_ONLY":
        raise RuntimeError("Recreate refused. Add --confirm-recreate-env DELETE_ENV_ONLY. Models are never deleted.")

    try:
        assert_nvidia_present()
        repo = ensure_repo(root, repo_path, update_repo=args.update_repo)
        if args.verify_only:
            py = find_env_python(env_path)
            if not py:
                raise RuntimeError(f"LTX 2.5 FP16 environment is missing: {env_path}")
        else:
            if args.skip_deps:
                status("SKIPPED", "UV workspace/dependency synchronization skipped")
                py = create_or_repair_env(root, env_path, recreate=args.danger_recreate_env)
            else:
                py = install_fp16_uv_runtime(root, env_path, repo, recreate=args.danger_recreate_env)

        if not args.verify_only:
            if not args.skip_model_downloads:
                ensure_models(py, root, model_root, args)
            else:
                status("SKIPPED", "Model downloads skipped")
                write_manifest(model_root, files)
            if not args.skip_ffmpeg:
                ensure_ffmpeg(root, skip_downloads=False)
            else:
                status("SKIPPED", "FFmpeg check/download skipped")

        rc = verify(py, root, repo_path, model_root, files)
        if rc == 0:
            status("OK", "LTX 2.5 BF16 installer verification passed")
        else:
            status("FAILED", "LTX 2.5 BF16 verification found missing/broken pieces")
        return rc
    except Exception as exc:
        status("FAILED", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
