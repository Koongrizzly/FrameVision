from __future__ import annotations

"""FrameVision LTX 2.5 BF16 installer / repair script.

Purpose
-------
Creates a completely separate LTX 2.5 runtime without touching LTX 2.3:
  <root>/environments/ltx25
  <root>/models/ltx-2.5/LTX-2
  <root>/models/ltx-2.5/...

Important download policy
-------------------------
This installer NEVER snapshots the whole Hugging Face LTX-2.5 repository.
It downloads an explicit BF16 allow-list only, with multiple files in parallel.

Default model files:
  - distilled BF16 transformer
  - Gemma 4 12B BF16 text encoder + LTX projection
  - DiffVAE BF16 video VAE
  - BF16 audio VAE
  - shared LTX 2.3 BF16 spatial upscaler used by LTX 2.5 distilled

Explicitly NOT downloaded by the default installer:
  - INT8 / Comfy INT8 / ConvRot files
  - FP8 model files
  - distilled transformer
  - distilled LoRA
  - IC-LoRAs
  - training/pre-trained packs
  - temporal upscaler (optional switch)
  - convolutional video VAE (optional switch)
  - duration head (optional switch)

The distilled LoRA is intentionally not used by this FrameVision installer.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENV_RELATIVE = Path("environments") / "ltx25"
MODEL_ROOT_RELATIVE = Path("models") / "ltx-2.5"
REPO_RELATIVE = MODEL_ROOT_RELATIVE / "LTX-2"
FFMPEG_BIN_RELATIVE = Path("presets") / "bin"
TEMP_RELATIVE = Path("temp")

OFFICIAL_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
HF_REPO = "Lightricks/LTX-2.5"

# LTX 2.5 DiffVAE uses torch.compiler.nested_compile_region, introduced in PyTorch 2.8.
# Keep CUDA 12.8 for Windows/RTX 30-series compatibility.
PYTHON_VERSION = "3.12"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES: Sequence[str] = (
    "torch==2.7.0",
    "torchaudio==2.7.0",
)

# We deliberately install the native repository packages, not Diffusers/SDNQ.
BASE_PACKAGES: Sequence[str] = (
    "huggingface_hub>=0.34",
    "opencv-python",
    "PySide6",
)

# Optional native LTX 2.5 acceleration on Windows.
# These are exact user-supplied wheels for Python 3.12 / Torch 2.7 / CUDA 12.8.
# Do not substitute guessed package versions.
USER_NATTEN_WHEEL = (
    "https://huggingface.co/lldacing/NATTEN-windows/resolve/main/"
    "natten-0.17.5+torch270cu128-cp312-cp312-win_amd64.whl"
)
USER_FLASH_ATTN_WHEEL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/"
    "flash_attn-2.8.2+cu128torch2.7-cp312-cp312-win_amd64.whl"
)
# SageAttention wheel supplied by the user targets Torch 2.7.1, not this 2.7.0
# runtime, so it is deliberately skipped.

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

def _installer_uv(root: Path) -> Path:
    uv = root / "presets" / "bin" / "uv" / "uv.exe"
    if not uv.is_file():
        raise RuntimeError(f"Portable uv.exe was not found: {uv}")
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
    status("OK", f"{mode.upper()} ConvRot install/repair complete")
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


def find_conda(root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    if os.environ.get("CONDA_EXE"):
        candidates.append(Path(os.environ["CONDA_EXE"]))
    if os.name == "nt":
        candidates += [
            root / "_miniconda" / "Scripts" / "conda.exe",
            root / "_miniconda" / "condabin" / "conda.bat",
            root / "_miniconda3" / "Scripts" / "conda.exe",
            root / "miniconda3" / "Scripts" / "conda.exe",
        ]
    else:
        candidates += [
            root / "_miniconda" / "bin" / "conda",
            root / "_miniconda3" / "bin" / "conda",
            root / "miniconda3" / "bin" / "conda",
        ]
    for c in candidates:
        if c.exists():
            status("FOUND", f"Conda: {c}")
            return c
    found = shutil.which("conda")
    if found:
        status("FOUND", f"Conda on PATH: {found}")
        return Path(found)
    return None


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
    conda = find_conda(root)
    if not conda:
        raise RuntimeError("Conda was not found in the portable FrameVision locations or on PATH.")

    existing = find_env_python(env_path)
    if existing and not recreate:
        status("FOUND", f"LTX 2.5 env Python: {existing}")
        return existing

    if recreate and env_path.exists():
        status("WARN", f"Deleting ONLY the LTX 2.5 env: {env_path}")
        shutil.rmtree(env_path)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    status("DOWNLOADING", f"Creating separate LTX 2.5 conda env with Python {PYTHON_VERSION}: {env_path}")
    run(
        [conda, "create", "--yes", "--prefix", env_path, f"python={PYTHON_VERSION}", "pip"],
        cwd=root,
        env=portable_env(root),
        check=True,
    )
    py = find_env_python(env_path)
    if not py:
        raise RuntimeError(f"Conda created the environment but Python was not found under {env_path}")
    return py


def pip_install(py: Path, root: Path, packages: Iterable[object], *, label: str, extra_args: Optional[list[object]] = None) -> None:
    cmd: list[object] = [py, "-m", "pip", "install", "--no-warn-script-location"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    status("DOWNLOADING", label)
    run(cmd, cwd=root, env=portable_env(root), check=True)


def pip_install_optional(
    py: Path,
    root: Path,
    packages: Iterable[object],
    *,
    label: str,
    extra_args: Optional[list[object]] = None,
) -> bool:
    try:
        pip_install(py, root, packages, label=label, extra_args=extra_args)
        status("OK", f"{label} installed")
        return True
    except Exception as exc:
        status("WARN", f"{label} could not be installed; LTX will keep its compatibility fallback: {exc}")
        return False


def natten_cuda_usable(py: Path, root: Path) -> tuple[bool, str]:
    """Verify NATTEN by actually running a tiny CUDA neighborhood-attention op.

    Older Windows wheels such as NATTEN 0.17.5 do not reliably expose the
    newer HAS_LIBNATTEN flag, so import-only or flag-only checks are not enough.
    """
    code = r"""
import sys
import torch
import natten

print("natten", getattr(natten, "__version__", "unknown"))

if not torch.cuda.is_available():
    print("CUDA unavailable")
    raise SystemExit(4)

device = torch.device("cuda")

# NATTEN APIs changed across releases. Try the older functional entry points
# first, then the newer na3d API if present. A real CUDA result is the pass
# condition.
errors = []

try:
    from natten.functional import na3d_qk, na3d_av
    q = torch.randn((1, 2, 4, 4, 4, 8), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attn = na3d_qk(q, k, kernel_size=(3, 3, 3), dilation=(1, 1, 1))
    out = na3d_av(attn, v, kernel_size=(3, 3, 3), dilation=(1, 1, 1))
    torch.cuda.synchronize()
    print("NATTEN CUDA smoke test OK via na3d_qk/na3d_av", tuple(out.shape))
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as exc:
    errors.append("legacy na3d_qk/na3d_av: %s: %s" % (type(exc).__name__, exc))

try:
    from natten.functional import na3d
    q = torch.randn((1, 2, 4, 4, 4, 8), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        out = na3d(q, k, v, kernel_size=(3, 3, 3), dilation=(1, 1, 1))
    except TypeError:
        out = na3d(q, k, v, kernel_size=(3, 3, 3))
    torch.cuda.synchronize()
    print("NATTEN CUDA smoke test OK via na3d", tuple(out.shape))
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as exc:
    errors.append("na3d: %s: %s" % (type(exc).__name__, exc))

print("NATTEN CUDA smoke test FAILED")
for err in errors:
    print(err)
raise SystemExit(5)
"""
    rc, out, err = run_capture([py, "-c", code], cwd=root, env=portable_env(root))
    detail = out or err
    if detail:
        detail = " | ".join(line.strip() for line in detail.splitlines() if line.strip())
    return rc == 0, detail or f"exit code {rc}"


def flash_attn_usable(py: Path, root: Path) -> tuple[bool, str]:
    # LTX native 2.5 transformer still chooses its own backend, but verify the
    # user's wheel is actually importable before reporting success.
    code = (
        "import flash_attn; "
        "print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"
    )
    rc, out, err = run_capture([py, "-c", code], cwd=root, env=portable_env(root))
    return rc == 0, (out or (err.splitlines()[-1] if err else ""))


def uninstall_package(py: Path, root: Path, package: str, *, reason: str) -> None:
    status("WARN", f"Removing {package}: {reason}")
    run(
        [py, "-m", "pip", "uninstall", "-y", package],
        cwd=root,
        env=portable_env(root),
        check=False,
    )


def install_dependencies(py: Path, root: Path) -> None:
    status("OK", "Installing native LTX 2.5 BF16 runtime dependencies")
    run(
        [py, "-m", "pip", "install", "--no-warn-script-location", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=root,
        env=portable_env(root),
        check=True,
    )

    # Clear accelerator wheels tied to the previous Torch 2.8 experiment before
    # switching to the user's known Torch 2.7 wheel stack.
    if os.name == "nt":
        run(
            [py, "-m", "pip", "uninstall", "-y", "triton-windows", "natten", "flash-attn", "flash_attn"],
            cwd=root,
            env=portable_env(root),
            check=False,
        )

    pip_install(
        py,
        root,
        TORCH_PACKAGES,
        label="PyTorch 2.7.0 CUDA 12.8 tuple",
        extra_args=["--index-url", PYTORCH_CUDA_INDEX],
    )
    pip_install(py, root, BASE_PACKAGES, label="FrameVision/LTX helper dependencies")

    if os.name == "nt":
        # Exact user-supplied NATTEN wheel. No public-index guessing or source build.
        pip_install(
            py,
            root,
            [USER_NATTEN_WHEEL],
            label="User NATTEN 0.17.5 wheel (Torch 2.7 / CUDA 12.8 / Python 3.12)",
        )
        natten_ok, natten_detail = natten_cuda_usable(py, root)
        if not natten_ok:
            uninstall_package(py, root, "natten", reason=natten_detail or "CUDA smoke test failed")
            raise RuntimeError(
                "The user-supplied NATTEN wheel installed but its CUDA neighborhood-attention backend failed: "
                + (natten_detail or "HAS_LIBNATTEN=False")
            )
        status("OK", "NATTEN CUDA backend verified: " + natten_detail)

        # Exact user-supplied FlashAttention wheel for the same Torch/CUDA/Python tuple.
        pip_install(
            py,
            root,
            [USER_FLASH_ATTN_WHEEL],
            label="User FlashAttention 2.8.2 wheel (Torch 2.7 / CUDA 12.8 / Python 3.12)",
        )
        flash_ok, flash_detail = flash_attn_usable(py, root)
        if not flash_ok:
            uninstall_package(py, root, "flash-attn", reason=flash_detail or "import failed")
            raise RuntimeError(
                "The user-supplied FlashAttention wheel installed but could not be imported: "
                + (flash_detail or "import failed")
            )
        status("OK", "FlashAttention verified: " + flash_detail)

        status(
            "FOUND",
            "SageAttention intentionally skipped: available user wheel targets Torch 2.7.1, "
            "while this LTX environment is pinned to Torch 2.7.0.",
        )


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


def install_ltx_packages(py: Path, root: Path, repo_path: Path) -> None:
    core = repo_path / "packages" / "ltx-core"
    pipelines = repo_path / "packages" / "ltx-pipelines"
    # Install with dependencies. This intentionally lets the current official repo
    # choose its compatible transformers/accelerate/etc ranges while Torch is pinned
    # to the required 2.8 CUDA tuple above.
    status("DOWNLOADING", "Installing official ltx-core and ltx-pipelines from the local LTX 2.5 repo")
    run(
        [py, "-m", "pip", "install", "--no-warn-script-location", "-e", core, "-e", pipelines],
        cwd=root,
        env=portable_env(root),
        check=True,
    )


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

    rc, out, err = run_capture(
        [py, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        cwd=root,
        env=portable_env(root),
    )
    check("Torch CUDA available", rc == 0, out or err.splitlines()[-1] if err else "")

    rc, out, err = run_capture(
        [py, "-c", "import torch; fn=getattr(torch.compiler, 'nested_compile_region', None); print(torch.__version__, 'nested_compile_region=' + ('YES' if callable(fn) else 'NO'))"],
        cwd=root,
        env=portable_env(root),
    )
    nested_detail = out or (err.splitlines()[-1] if err else "")
    if rc == 0 and "nested_compile_region=YES" in nested_detail:
        status("OK", "PyTorch nested_compile_region API" + (f": {nested_detail}" if nested_detail else ""))
    else:
        # Torch 2.7 is within the official LTX-core torch~=2.7 requirement.
        # nested_compile_region is only needed by optional torch.compile paths.
        # FrameVision's normal native LTX 2.5 run is eager unless --compile is requested.
        status(
            "WARN",
            "PyTorch nested_compile_region API"
            + (f": {nested_detail}" if nested_detail else "")
            + " | unavailable on Torch 2.7; FrameVision installs an eager-mode compatibility shim before LTX imports",
        )

    checks = {
        "ltx_core import": "import ltx_core; print('ltx_core OK')",
        "ltx_pipelines import": "import ltx_pipelines; print('ltx_pipelines OK')",
        "Gemma4-capable Transformers": "import transformers; from transformers import Gemma4UnifiedTextModel; print(transformers.__version__, Gemma4UnifiedTextModel.__name__)",
        "OpenImageIO import": "import OpenImageIO as oiio; print(getattr(oiio, '__version__', 'OK'))",
        "OpenCV import": "import cv2; print(cv2.__version__)",
    }
    for label, code in checks.items():
        rc, out, err = run_capture([py, "-c", code], cwd=root, env=portable_env(root))
        check(label, rc == 0, out or (err.splitlines()[-1] if err else ""))

    flash_ok, flash_detail = flash_attn_usable(py, root)
    status("OK" if flash_ok else "WARN", "FlashAttention acceleration" + (f": {flash_detail}" if flash_detail else ""))

    natten_ok, natten_detail = natten_cuda_usable(py, root)
    if natten_ok:
        status("OK", "NATTEN DiffVAE CUDA acceleration: " + natten_detail)
    else:
        status(
            "WARN",
            "NATTEN DiffVAE CUDA acceleration unavailable"
            + (f": {natten_detail}" if natten_detail else ""),
        )

    # Triton is intentionally not installed automatically in the Torch 2.7 stack.
    rc, out, err = run_capture([py, "-c", "import triton; print('triton', getattr(triton, '__version__', 'unknown'))"], cwd=root, env=portable_env(root))
    detail = out or (err.splitlines()[-1] if err else "")
    status("FOUND" if rc == 0 else "WARN", "Triton status (not auto-installed)" + (f": {detail}" if detail else ""))

    for filename in files:
        p = model_root / Path(filename)
        check(f"Model {filename}", p.exists() and p.stat().st_size > 1024 * 1024, str(p))

    # Verify we did not accidentally create obvious unwanted model variants.
    accidental = []
    if model_root.exists():
        for p in model_root.rglob("*.safetensors"):
            rel = "/" + p.relative_to(model_root).as_posix().lower()
            if any(tok in rel for tok in FORBIDDEN_MODEL_TOKENS):
                accidental.append(str(p))
    if accidental:
        status("WARN", "Unwanted/quantized/distilled files already exist in the LTX 2.5 model folder; installer did not delete them:")
        for p in accidental:
            print("  " + p)
    else:
        status("OK", "No INT8/INT4/FP8/distilled-LoRA files found in the LTX 2.5 model folder")

    return 20 if failed else 0



def install_native_workspace_packages(root: Path, env_python: Path, repo_path: Path) -> None:
    """Install LTX workspace packages and their declared dependencies with uv.

    A fresh `uv venv` does not need pip inside the environment. `uv pip install
    --python <env-python>` installs directly into that environment and resolves
    package dependencies from the local LTX-2 pyprojects.
    """
    uv = _installer_uv(root)
    core_pkg = repo_path / "packages" / "ltx-core"
    pipelines_pkg = repo_path / "packages" / "ltx-pipelines"

    missing = [p for p in (core_pkg, pipelines_pkg) if not (p / "pyproject.toml").is_file()]
    if missing:
        raise RuntimeError(
            "Native LTX-2 repo is incomplete; missing workspace package(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    status("OK", "Installing native LTX workspace packages + dependencies")
    status("OK", f"ltx-core source: {core_pkg}")
    status("OK", f"ltx-pipelines source: {pipelines_pkg}")

    run(
        [
            uv, "pip", "install",
            "--python", env_python,
            "--upgrade",
            str(core_pkg),
            str(pipelines_pkg),
        ],
        cwd=repo_path,
        check=True,
    )

    # Verify the environment itself can import the packages and a core dependency.
    verify_code = (
        "import einops, ltx_core, ltx_pipelines; "
        "print('[VERIFY] native LTX imports OK')"
    )
    run([env_python, "-c", verify_code], cwd=repo_path, check=True)



def repair_native_torch_abi(root: Path, env_python: Path) -> None:
    """Restore a matching Torch ABI after workspace dependency resolution.

    ltx-core declares torch~=2.7, so a generic dependency resolver may select a
    newer 2.x release. TorchVision is not required by LTX and an older
    torchvision wheel beside a newer torch produces Windows _C.pyd entry-point
    failures. Remove torchvision and reinstall the known matching native pair.
    """
    uv = _installer_uv(root)
    status("OK", "Finalizing native Torch ABI: torch 2.7.0 + torchaudio 2.7.0 CUDA 12.8")
    run(
        [uv, "pip", "uninstall", "--python", env_python, "torchvision"],
        cwd=root,
        check=False,
    )
    run(
        [
            uv, "pip", "install", "--python", env_python,
            "--reinstall",
            "torch==2.7.0", "torchaudio==2.7.0",
            "--index-url", PYTORCH_CUDA_INDEX,
        ],
        cwd=root,
        check=True,
    )
    verify = (
        "import torch, torchaudio; "
        "print('[VERIFY] torch', torch.__version__); "
        "print('[VERIFY] torchaudio', torchaudio.__version__)"
    )
    run([env_python, "-c", verify], cwd=root, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="FrameVision LTX 2.5 native BF16 installer / repair")
    ap.add_argument(
        "--action",
        choices=("install-fp16", "install-w4a8", "install-int4", "verify-fp16"),
        default="install-fp16",
        help="Installer action. The GUI launches this Python file directly.",
    )
    ap.add_argument("--root", default=None, help="FrameVision root. Normally auto-detected from presets/extra_env.")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--skip-model-downloads", action="store_true")
    ap.add_argument("--skip-ffmpeg", action="store_true")
    ap.add_argument("--update-repo", action="store_true", help="Update an existing LTX-2 checkout to current origin/main")
    ap.add_argument("--danger-recreate-env", action="store_true", help="Delete/recreate ONLY environments/ltx25; never deletes models")
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
        py = create_or_repair_env(root, env_path, recreate=args.danger_recreate_env)

        if not args.verify_only and not args.skip_deps:
            install_dependencies(py, root)
        elif args.skip_deps:
            status("SKIPPED", "Dependency installation skipped")

        if not args.verify_only:
            repo = ensure_repo(root, repo_path, update_repo=args.update_repo)

            # Install the two local monorepo packages only after the repository
            # is known to exist, using the actual environment Python returned
            # by create_or_repair_env().
            install_native_workspace_packages(root, py, repo)

            install_ltx_packages(py, root, repo)

            # Workspace dependency resolution is allowed to install newer torch
            # because ltx-core uses torch~=2.7. Restore the exact matching native
            # ABI after that step and remove unneeded torchvision.
            repair_native_torch_abi(root, py)
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
