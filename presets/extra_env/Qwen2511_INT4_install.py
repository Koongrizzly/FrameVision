#!/usr/bin/env python3
"""FrameVision installer for Qwen-Image-Edit-2511 Nunchaku INT4.

Place at:
    <FrameVision root>/presets/extra_env/Qwen2511_INT4_install.py

Creates:
    <root>/environments/.qwen2511_int
    <root>/models/qwen2511_int
    <root>/temp/qwen2511_int4
    <root>/presets/setsave/qwen2511_int4_install.json
    <root>/presets/setsave/qwen2511_int4_settings.json

The included INT4 checkpoints work with Nunchaku on RTX 30/40-series GPUs.
Nunchaku requires FP4 checkpoints on Blackwell RTX 50-series GPUs, so this
installer deliberately rejects RTX 50-series instead of creating a broken setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, NoReturn, Optional

INSTALLER_VERSION = "1.0.0"

NUNCHAKU_GITHUB = "https://github.com/nunchaku-ai/nunchaku"
NUNCHAKU_RELEASES_API = (
    "https://api.github.com/repos/nunchaku-ai/nunchaku/releases?per_page=30"
)
NUNCHAKU_FALLBACK_TAG = "v1.2.1"
QWEN_BASE_REPO = "Qwen/Qwen-Image-Edit-2511"
INT4_REPO = "LAXMAYDAY/Qwen_image_edit_2511_int4"

# Prefer the same cu128/Torch 2.8 stack already used by other FrameVision engines.
# Newer stacks are fallbacks if a matching prebuilt Nunchaku wheel is unavailable.
TORCH_STACKS = (
    {
        "torch_tag": "2.8",
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "index": "https://download.pytorch.org/whl/cu128",
        "cuda": "12.8",
    },
    {
        "torch_tag": "2.9",
        "torch": "2.9.0",
        "torchvision": "0.24.0",
        "index": "https://download.pytorch.org/whl/cu128",
        "cuda": "12.8",
    },
    {
        "torch_tag": "2.10",
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "index": "https://download.pytorch.org/whl/cu128",
        "cuda": "12.8",
    },
)

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "recommended": {
        "label": "Recommended: quality-r64 Lightning 4-step",
        "filename": (
            "nunchaku_qwen_image_edit_2511_quality_r64_"
            "lightning4steps_q128o64m64_int4.safetensors"
        ),
        "steps": 4,
        "rank": 64,
        "size_gib": 11.26,
        "role": "Recommended balance of speed, footprint and edit quality.",
        "aliases": ("quality64", "quality-r64", "r64", "default"),
    },
    "fastest": {
        "label": "Fastest / lowest footprint: balanced-r32 Lightning 4-step",
        "filename": (
            "nunchaku_qwen_image_edit_2511_balanced_r32_"
            "lightning4steps_q128o64m64_int4.safetensors"
        ),
        "steps": 4,
        "rank": 32,
        "size_gib": 11.26,
        "role": "Fastest candidate and low-VRAM baseline.",
        "aliases": (
            "balanced32",
            "balanced-r32",
            "r32",
            "low-vram",
            "lowest-vram",
        ),
    },
    "best-low-step": {
        "label": "Best low-step candidate: quality-r128-b15 Lightning 4-step",
        "filename": (
            "nunchaku_qwen_image_edit_2511_quality_r128_b15_"
            "lightning4steps_q128o128m128_int4.safetensors"
        ),
        "steps": 4,
        "rank": 128,
        "size_gib": 11.79,
        "role": "Highest-rank four-step quality candidate.",
        "aliases": ("best", "quality128", "quality-r128", "r128-4"),
    },
    "fidelity": {
        "label": "Best measured fidelity: mid-r128 Lightning 8-step",
        "filename": (
            "nunchaku_qwen_image_edit_2511_mid_r128_"
            "lightning8steps_q128o128m128_int4.safetensors"
        ),
        "steps": 8,
        "rank": 128,
        "size_gib": 11.79,
        "role": "Best measured PSNR/SSIM/LPIPS among benchmarked Lightning rows.",
        "aliases": ("mid128", "mid-r128", "r128-8", "best-measured"),
    },
}


def log(message: str = "") -> None:
    print(f"[Qwen2511 INT4] {message}", flush=True)


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * max(8, len(title))}", flush=True)


def fail(message: str, code: int = 1) -> NoReturn:
    print("\n[Qwen2511 INT4] ERROR", file=sys.stderr, flush=True)
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def quote_cmd(cmd: Iterable[object]) -> str:
    result = []
    for value in cmd:
        text = str(value)
        result.append(f'"{text}"' if any(c.isspace() for c in text) else text)
    return " ".join(result)


def run(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    log(f"Running: {quote_cmd(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def detect_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    here = Path(__file__).resolve()
    candidate = here.parents[2]  # .../presets/extra_env/file.py -> root
    if (candidate / "presets" / "extra_env").exists():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "presets" / "extra_env").exists():
        return cwd
    return candidate


def venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if is_windows() else "bin/python")


def pip_install(py: Path, args: list[str], env: dict[str, str]) -> None:
    run([str(py), "-m", "pip", *args], env=env)


def ensure_supported_host() -> None:
    version = sys.version_info
    if not (3, 10) <= (version.major, version.minor) <= (3, 13):
        fail(
            "Run this installer with Python 3.10, 3.11, 3.12 or 3.13. "
            f"Detected {version.major}.{version.minor}.{version.micro}."
        )
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        fail(f"Only x86-64 is supported. Detected: {platform.machine()}")
    if platform.system().lower() not in {"windows", "linux"}:
        fail(f"Only Windows and Linux are supported. Detected: {platform.system()}")


def ensure_environment(env_dir: Path, force: bool, runtime_env: dict[str, str]) -> Path:
    section("Python environment")
    if force and env_dir.exists():
        log(f"Removing existing environment: {env_dir}")
        shutil.rmtree(env_dir, ignore_errors=True)

    py = venv_python(env_dir)
    if not py.exists():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(env_dir)])
    else:
        log(f"Reusing environment: {env_dir}")

    if not py.exists():
        fail(f"Environment Python was not created: {py}")
    pip_install(py, ["install", "--upgrade", "pip", "setuptools", "wheel"], runtime_env)
    return py


def request_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FrameVision-Qwen2511-INT4-Installer",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def host_wheel_tags() -> tuple[str, str]:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = "win_amd64" if is_windows() else "linux_x86_64"
    return py_tag, platform_tag


def wheel_matches(
    name: str,
    *,
    py_tag: str,
    platform_tag: str,
    torch_tag: str,
) -> bool:
    lower = name.lower()
    return (
        lower.endswith(".whl")
        and lower.startswith("nunchaku-")
        and f"torch{torch_tag}" in lower
        and f"-{py_tag}-{py_tag}-{platform_tag}.whl" in lower
        and ("cu12.8" in lower or "+torch" in lower)
    )


def discover_nunchaku_wheel() -> tuple[dict[str, Any], dict[str, str], str]:
    section("Selecting Nunchaku wheel")
    py_tag, platform_tag = host_wheel_tags()

    try:
        releases = request_json(NUNCHAKU_RELEASES_API)
    except Exception as exc:
        log(f"GitHub release discovery failed: {exc}")
        releases = []

    ordered: list[dict[str, Any]] = []
    if isinstance(releases, list):
        ordered.extend(r for r in releases if not r.get("draft") and not r.get("prerelease"))
        ordered.extend(r for r in releases if not r.get("draft") and r.get("prerelease"))

    # Prefer the newest stable Nunchaku release first. Inside that release, use
    # the oldest compatible Torch stack from our tested list to avoid unnecessary
    # dependency churn. This prevents an old Nunchaku release from winning merely
    # because it still has a Torch 2.8 wheel.
    for release in ordered:
        for stack in TORCH_STACKS:
            for asset in release.get("assets", []):
                name = str(asset.get("name", ""))
                if wheel_matches(
                    name,
                    py_tag=py_tag,
                    platform_tag=platform_tag,
                    torch_tag=str(stack["torch_tag"]),
                ):
                    url = str(asset.get("browser_download_url", ""))
                    if url:
                        tag = str(release.get("tag_name") or NUNCHAKU_FALLBACK_TAG)
                        log(f"Selected: {name}")
                        return {"name": name, "url": url, "tag": tag}, stack, tag

    # v1.2.1 is confirmed to publish cu12.8/Torch 2.10 wheels for Python
    # 3.10-3.13 on Windows and Linux.
    stack = next(item for item in TORCH_STACKS if item["torch_tag"] == "2.10")
    version = NUNCHAKU_FALLBACK_TAG.lstrip("v")
    filename = (
        f"nunchaku-{version}+cu12.8torch{stack['torch_tag']}-"
        f"{py_tag}-{py_tag}-{platform_tag}.whl"
    )
    url = f"{NUNCHAKU_GITHUB}/releases/download/{NUNCHAKU_FALLBACK_TAG}/{filename}"
    log(f"Using stable fallback wheel URL: {filename}")
    return {"name": filename, "url": url, "tag": NUNCHAKU_FALLBACK_TAG}, stack, NUNCHAKU_FALLBACK_TAG


def format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{number:.1f} TiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        log(f"Already downloaded: {destination}")
        return destination

    section(f"Downloading {destination.name}")
    log(f"Source: {url}")
    log(f"Target: {destination}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "FrameVision-Qwen2511-INT4-Installer"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            started = time.time()
            last_update = 0.0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_update >= 0.5:
                        speed = downloaded / max(0.001, now - started)
                        if total:
                            text = (
                                f"\r  {100.0 * downloaded / total:6.2f}%  "
                                f"{format_bytes(downloaded)}/{format_bytes(total)}  "
                                f"{format_bytes(int(speed))}/s"
                            )
                        else:
                            text = f"\r  {format_bytes(downloaded)}  {format_bytes(int(speed))}/s"
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        last_update = now
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        fail(
            f"Download failed with HTTP {exc.code}: {url}\n"
            "No matching Nunchaku wheel was found. Run this installer with "
            "FrameVision's Python 3.10 or Python 3.11, or update the wheel table."
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    sys.stdout.write("\n")
    partial.replace(destination)
    log(f"Downloaded: {destination} ({format_bytes(destination.stat().st_size)})")
    return destination


def make_runtime_env(root: Path, model_root: Path, temp_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    hf_temp = temp_dir / "huggingface"
    hf_temp.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HF_HOME": str(hf_temp / "home"),
            "HF_HUB_CACHE": str(hf_temp / "hub"),
            "HF_XET_CACHE": str(hf_temp / "xet"),
            "HUGGINGFACE_HUB_CACHE": str(hf_temp / "hub"),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "FRAMEVISION_ROOT": str(root),
            "QWEN2511_INT4_ROOT": str(model_root),
        }
    )
    return env


def install_stack(
    py: Path,
    stack: dict[str, str],
    wheel: dict[str, Any],
    temp_dir: Path,
    runtime_env: dict[str, str],
) -> Path:
    section("Installing PyTorch and dependencies")
    pip_install(
        py,
        [
            "install",
            "--upgrade",
            f"torch=={stack['torch']}",
            f"torchvision=={stack['torchvision']}",
            "--index-url",
            stack["index"],
        ],
        runtime_env,
    )
    pip_install(
        py,
        [
            "install",
            "--upgrade",
            "diffusers==0.39.0",
            "transformers>=4.57,<5",
            "accelerate>=1.9",
            "huggingface_hub>=0.34",
            "hf_xet>=1.1",
            "safetensors>=0.4.5",
            "peft>=0.17",
            "sentencepiece>=0.2",
            "protobuf>=5,<7",
            "einops>=0.8",
            "Pillow>=10",
            "numpy<2.3",
            "requests>=2.32",
            "packaging>=24",
        ],
        runtime_env,
    )
    wheel_path = download_url(str(wheel["url"]), temp_dir / str(wheel["name"]))
    pip_install(py, ["install", "--upgrade", "--no-deps", str(wheel_path)], runtime_env)
    return wheel_path


def python_json(py: Path, code: str, runtime_env: dict[str, str]) -> Any:
    result = run([str(py), "-c", code], env=runtime_env, capture=True)
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def verify_gpu_and_runtime(
    py: Path,
    *,
    allow_no_gpu: bool,
    runtime_env: dict[str, str],
) -> dict[str, Any]:
    section("Verifying runtime and GPU")
    code = r'''
import json, platform, sys
import torch, diffusers, transformers, nunchaku
info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "nunchaku": getattr(nunchaku, "__version__", "unknown"),
}
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    info.update({
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": [int(capability[0]), int(capability[1])],
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    })
print(json.dumps(info))
'''
    try:
        info = python_json(py, code, runtime_env) or {}
    except subprocess.CalledProcessError as exc:
        fail(f"Runtime import verification failed: {exc}")
    log(json.dumps(info, indent=2))

    if not info.get("cuda_available"):
        if allow_no_gpu:
            log("WARNING: CUDA not detected; continuing without a GPU load test.")
            return info
        fail(
            "CUDA is unavailable in the new environment. Install a current NVIDIA "
            "driver and rerun. --allow-no-gpu only prepares files; it does not add CPU inference."
        )

    capability = tuple(info.get("capability") or (0, 0))
    gpu_name = str(info.get("gpu_name") or "Unknown NVIDIA GPU")
    sm = int(capability[0]) * 10 + int(capability[1])

    if sm in {120, 121} or "RTX 50" in gpu_name.upper():
        fail(
            f"Detected Blackwell GPU: {gpu_name} (sm_{sm}).\n\n"
            "Nunchaku requires FP4 checkpoints on RTX 50-series GPUs and rejects "
            "these INT4 files. A separate Qwen2511 FP4 source is needed for 50-series."
        )
    if sm not in {75, 80, 86, 89}:
        fail(
            f"Unsupported architecture: {gpu_name} (sm_{sm}). "
            "This installer expects a Nunchaku-supported Turing, Ampere or Ada GPU."
        )
    if sm < 86:
        log("WARNING: GPU is older than the requested RTX 30-series baseline.")
    return info


def clone_or_download_repo(repo_dir: Path, temp_dir: Path, tag: str) -> Path:
    section("Installing Nunchaku source repository")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if (repo_dir / ".git").exists():
        try:
            run(["git", "fetch", "--tags", "--depth", "1"], cwd=repo_dir)
            run(["git", "checkout", "--force", tag], cwd=repo_dir)
        except Exception as exc:
            log(f"Repository update failed; keeping installed source: {exc}")
        return repo_dir
    if repo_dir.exists() and any(repo_dir.iterdir()):
        log(f"Source directory already exists: {repo_dir}")
        return repo_dir

    if shutil.which("git"):
        try:
            run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    tag,
                    "--recurse-submodules",
                    NUNCHAKU_GITHUB + ".git",
                    str(repo_dir),
                ]
            )
            return repo_dir
        except Exception as exc:
            log(f"Git clone failed; falling back to source zip: {exc}")
            shutil.rmtree(repo_dir, ignore_errors=True)

    zip_path = download_url(
        f"{NUNCHAKU_GITHUB}/archive/refs/tags/{tag}.zip",
        temp_dir / f"nunchaku-{tag}.zip",
    )
    extract_dir = temp_dir / f"nunchaku-extract-{tag}"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    candidates = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        fail(f"Could not identify extracted Nunchaku source in {extract_dir}")
    shutil.move(str(candidates[0]), str(repo_dir))
    return repo_dir


def download_base_assets(py: Path, base_dir: Path, runtime_env: dict[str, str]) -> None:
    required = [
        base_dir / "model_index.json",
        base_dir / "scheduler" / "scheduler_config.json",
        base_dir / "vae" / "config.json",
        base_dir / "text_encoder" / "config.json",
        base_dir / "tokenizer" / "tokenizer_config.json",
        base_dir / "processor" / "preprocessor_config.json",
    ]
    if all(path.exists() and path.stat().st_size > 0 for path in required):
        section("Qwen-Image-Edit-2511 shared assets")
        log(f"Shared assets already installed; skipping download: {base_dir}")
        return

    section("Downloading Qwen-Image-Edit-2511 shared assets")
    log(
        "Downloading processor, tokenizer, Qwen2.5-VL text encoder, scheduler, VAE "
        "and configs. The original BF16 transformer is excluded."
    )
    code = textwrap.dedent(
        f'''
        import json, os, shutil
        from pathlib import Path
        from huggingface_hub import snapshot_download
        destination = Path({str(base_dir)!r})
        destination.mkdir(parents=True, exist_ok=True)
        cache_root = Path(os.environ["HF_HUB_CACHE"]) / "qwen2511_base_download"
        path = snapshot_download(
            repo_id={QWEN_BASE_REPO!r},
            cache_dir=str(cache_root),
            ignore_patterns=[
                "transformer/*", "transformer/**", "*.md", "README*", ".gitattributes"
            ],
            max_workers=4,
        )
        shutil.copytree(path, destination, dirs_exist_ok=True, symlinks=False)
        shutil.rmtree(cache_root, ignore_errors=True)
        print(json.dumps({{"path": str(destination)}}))
        '''
    )
    python_json(py, code, runtime_env)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail("Shared Qwen download is incomplete. Missing:\n  - " + "\n  - ".join(missing))
    transformer_dir = base_dir / "transformer"
    if transformer_dir.exists() and list(transformer_dir.glob("*.safetensors")):
        fail(
            f"Original BF16 transformer weights appeared in {transformer_dir}. "
            "Delete that folder and rerun; the INT4 checkpoint replaces it."
        )


def download_int4_checkpoint(
    py: Path,
    int4_dir: Path,
    option_key: str,
    runtime_env: dict[str, str],
) -> Path:
    option = MODEL_OPTIONS[option_key]
    filename = str(option["filename"])
    final_path = int4_dir / filename

    # Nunchaku INT4 checkpoints are larger than 11 GiB. Reuse an already completed
    # file instead of downloading it into a disposable cache on every installer run.
    minimum_complete_size = int(float(option.get("size_gib", 0.0)) * 1024**3 * 0.90)
    if final_path.exists() and final_path.stat().st_size >= max(minimum_complete_size, 1024 * 1024):
        section(f"INT4 model: {option['label']}")
        log(
            f"Checkpoint already installed; skipping download: {final_path} "
            f"({final_path.stat().st_size / 1024**3:.2f} GiB)"
        )
        return final_path

    if final_path.exists():
        log(
            f"Existing checkpoint looks incomplete "
            f"({final_path.stat().st_size / 1024**3:.2f} GiB); redownloading it."
        )
        final_path.unlink(missing_ok=True)

    section(f"Downloading INT4 model: {option['label']}")
    code = textwrap.dedent(
        f'''
        import json, os, shutil
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        destination = Path({str(int4_dir)!r})
        destination.mkdir(parents=True, exist_ok=True)
        final_path = destination / {filename!r}
        cache_root = Path(os.environ["HF_HUB_CACHE"]) / ("checkpoint_" + final_path.stem)
        cached_path = hf_hub_download(
            repo_id={INT4_REPO!r},
            filename={filename!r},
            cache_dir=str(cache_root),
        )
        shutil.copy2(cached_path, final_path)
        shutil.rmtree(cache_root, ignore_errors=True)
        print(json.dumps({{"path": str(final_path)}}))
        '''
    )
    result = python_json(py, code, runtime_env) or {}
    path = Path(result.get("path") or (int4_dir / filename))
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        fail(f"INT4 checkpoint is missing or incomplete: {path}")
    return path


def download_repo_metadata(py: Path, metadata_dir: Path, runtime_env: dict[str, str]) -> None:
    section("Downloading INT4 repository metadata")
    code = textwrap.dedent(
        f'''
        import os, shutil
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        destination = Path({str(metadata_dir)!r})
        destination.mkdir(parents=True, exist_ok=True)
        cache_root = Path(os.environ["HF_HUB_CACHE"]) / "int4_metadata_download"
        for name in ("README.md", "manifest.json", "metrics.json"):
            try:
                cached = hf_hub_download(
                    repo_id={INT4_REPO!r}, filename=name, cache_dir=str(cache_root)
                )
                shutil.copy2(cached, destination / name)
            except Exception as exc:
                print(f"Optional metadata not downloaded: {{name}}: {{exc}}")
        shutil.rmtree(cache_root, ignore_errors=True)
        '''
    )
    run([str(py), "-c", code], env=runtime_env)


def resolve_model_key(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in MODEL_OPTIONS:
        return normalized
    for key, option in MODEL_OPTIONS.items():
        if normalized in {str(alias).lower() for alias in option.get("aliases", ())}:
            return key
    raise ValueError(
        f"Unknown model option {value!r}. Valid: {', '.join(MODEL_OPTIONS)}"
    )


def choose_models(args: argparse.Namespace) -> list[str]:
    if args.env_only or args.skip_models:
        return []
    selected: list[str] = []
    if args.all_models:
        return list(MODEL_OPTIONS)
    if args.model:
        for value in args.model:
            key = resolve_model_key(value)
            if key not in selected:
                selected.append(key)
        return selected
    if args.non_interactive:
        return ["recommended"]

    section("Choose INT4 checkpoint(s)")
    keys = list(MODEL_OPTIONS)
    for index, key in enumerate(keys, 1):
        option = MODEL_OPTIONS[key]
        print(
            f"  {index}) {option['label']}\n"
            f"     {option['role']} Approx. {option['size_gib']} GiB"
        )
    print("  5) All four models")
    print("  6) Environment and shared assets only")
    answer = input("\nSelection [1]: ").strip() or "1"
    if answer.lower() in {"5", "all", "a"}:
        return keys
    if answer.lower() in {"6", "none", "n", "env"}:
        return []
    for token in (part.strip() for part in answer.split(",") if part.strip()):
        key = keys[int(token) - 1] if token.isdigit() and 1 <= int(token) <= 4 else resolve_model_key(token)
        if key not in selected:
            selected.append(key)
    return selected


def warn_disk_space(root: Path, selected: list[str], skip_base: bool) -> None:
    free_gib = shutil.disk_usage(root).free / 1024**3
    estimate = 8.0 + (0.0 if skip_base else 17.0)
    estimate += sum(float(MODEL_OPTIONS[key]["size_gib"]) for key in selected)
    estimate += 5.0
    log(f"Estimated requirement: {estimate:.1f} GiB; available: {free_gib:.1f} GiB")
    if free_gib < estimate:
        fail("Not enough free disk space. Free space or select fewer checkpoints.")


def profile_payload() -> dict[str, Any]:
    models = {}
    for key, option in MODEL_OPTIONS.items():
        models[key] = {
            "label": option["label"],
            "filename": option["filename"],
            "relative_path": f"int4/{option['filename']}",
            "steps": option["steps"],
            "rank": option["rank"],
            "true_cfg_scale": 1.0,
            "guidance_scale": 1.0,
            "size_gib": option["size_gib"],
            "role": option["role"],
        }
    return {
        "schema_version": 1,
        "engine": "Qwen-Image-Edit-2511",
        "backend": "Nunchaku",
        "precision": "INT4",
        "base_repo": QWEN_BASE_REPO,
        "checkpoint_repo": INT4_REPO,
        "default_model": "recommended",
        "models": models,
        "vram_modes": {
            "auto": "Model CPU offload above 18 GiB; official low-VRAM path at 18 GiB or below.",
            "balanced": "Diffusers model CPU offload.",
            "low": "One Nunchaku block on GPU plus sequential CPU offload.",
            "full": "Whole pipeline on CUDA; explicit testing only.",
        },
    }


RUNTIME_TEMPLATE = r'''#!/usr/bin/env python3
"""FrameVision runtime adapter for Qwen-Image-Edit-2511 Nunchaku INT4."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

MODEL_ROOT = Path(__file__).resolve().parent
BASE_DIR = MODEL_ROOT / "base" / "Qwen-Image-Edit-2511"
PROFILES_PATH = MODEL_ROOT / "model_profiles.json"

LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def load_profiles() -> dict[str, Any]:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def available_models() -> dict[str, dict[str, Any]]:
    result = {}
    for key, profile in load_profiles()["models"].items():
        path = MODEL_ROOT / profile["relative_path"]
        if path.exists():
            item = dict(profile)
            item["path"] = str(path)
            result[key] = item
    return result


def gpu_memory_gib() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def resolve_model(model_key: Optional[str] = None):
    profiles = load_profiles()
    key = model_key or profiles.get("default_model") or "recommended"
    if key not in profiles["models"]:
        raise KeyError(f"Unknown Qwen2511 INT4 profile: {key}")
    profile = dict(profiles["models"][key])
    checkpoint = MODEL_ROOT / profile["relative_path"]
    if not checkpoint.exists():
        installed = ", ".join(available_models()) or "none"
        raise FileNotFoundError(
            f"Checkpoint not installed: {checkpoint}. Installed profiles: {installed}"
        )
    return key, profile, checkpoint


def append_offload_exclusion(pipeline: Any, name: str) -> None:
    current = getattr(pipeline, "_exclude_from_cpu_offload", None)
    if current is None:
        pipeline._exclude_from_cpu_offload = [name]
    elif name not in current:
        current.append(name)


def apply_vram_mode(pipeline: Any, transformer: Any, mode: str = "auto") -> str:
    normalized = (mode or "auto").strip().lower().replace("_", "-")
    if normalized in {"lowest", "low-vram", "minimum"}:
        normalized = "low"
    if normalized == "auto":
        normalized = "balanced" if gpu_memory_gib() > 18.0 else "low"
    if normalized == "balanced":
        pipeline.enable_model_cpu_offload()
    elif normalized == "low":
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        append_offload_exclusion(pipeline, "transformer")
        pipeline.enable_sequential_cpu_offload()
    elif normalized == "full":
        pipeline.to("cuda")
    else:
        raise ValueError(f"Unknown VRAM mode: {mode}; use auto, balanced, low or full")
    return normalized


def load_pipeline(model_key: Optional[str] = None, *, vram_mode: str = "auto"):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from nunchaku import NunchakuQwenImageTransformer2DModel

    key, profile, checkpoint = resolve_model(model_key)
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(checkpoint))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER_CONFIG)
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        str(BASE_DIR),
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    applied_mode = apply_vram_mode(pipeline, transformer, vram_mode)
    pipeline.set_progress_bar_config(disable=False)
    info = {
        "model_key": key,
        "checkpoint": str(checkpoint),
        "steps": int(profile["steps"]),
        "true_cfg_scale": float(profile.get("true_cfg_scale", 1.0)),
        "guidance_scale": float(profile.get("guidance_scale", 1.0)),
        "vram_mode": applied_mode,
    }
    return pipeline, info


def edit_images(
    images: Iterable[Any],
    prompt: str,
    *,
    model_key: Optional[str] = None,
    vram_mode: str = "auto",
    seed: int = -1,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    import torch

    image_list = list(images)
    if not image_list:
        raise ValueError("At least one input image is required")
    if not prompt.strip():
        raise ValueError("Edit prompt is empty")
    pipeline, info = load_pipeline(model_key, vram_mode=vram_mode)
    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = int(torch.seed() % (2**63 - 1))
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
    kwargs: dict[str, Any] = {
        "image": image_list if len(image_list) > 1 else image_list[0],
        "prompt": prompt,
        "num_inference_steps": int(steps or info["steps"]),
        "true_cfg_scale": float(info["true_cfg_scale"]),
        "guidance_scale": float(info["guidance_scale"]),
        "generator": generator,
    }
    if width:
        kwargs["width"] = int(width)
    if height:
        kwargs["height"] = int(height)
    result = pipeline(**kwargs)
    info["seed"] = actual_seed
    info["effective_steps"] = kwargs["num_inference_steps"]
    return result.images, info, pipeline


def unload_pipeline(pipeline: Any) -> None:
    try:
        pipeline.to("cpu")
    except Exception:
        pass
    del pipeline
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
'''

CLI_TEMPLATE = r'''#!/usr/bin/env python3
"""Standalone FrameVision CLI for Qwen-Image-Edit-2511 Nunchaku INT4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image
from qwen2511_int4_runtime import available_models, edit_images, load_profiles, unload_pipeline


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit images with Qwen2511 INT4")
    parser.add_argument("--input", action="append", default=[], help="Repeat for multiple images")
    parser.add_argument("--output")
    parser.add_argument("--prompt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--vram-mode", default="auto", choices=("auto", "balanced", "low", "full"))
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--print-info", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.print_info:
        payload = load_profiles()
        payload["installed_models"] = available_models()
        print(json.dumps(payload, indent=2))
        return 0
    if not args.input:
        raise SystemExit("At least one --input image is required")
    if not args.output:
        raise SystemExit("--output is required")
    if not args.prompt:
        raise SystemExit("--prompt is required")

    images = []
    for value in args.input:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Input image not found: {path}")
        with Image.open(path) as opened:
            images.append(opened.convert("RGB"))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = None
    try:
        results, info, pipeline = edit_images(
            images,
            args.prompt,
            model_key=args.model,
            vram_mode=args.vram_mode,
            seed=args.seed,
            steps=args.steps,
            width=args.width,
            height=args.height,
        )
        results[0].save(output)
        info["output"] = str(output)
        print("FRAMEVISION_RESULT=" + json.dumps(info))
        return 0
    finally:
        if pipeline is not None:
            unload_pipeline(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
'''

SMOKE_TEMPLATE = r'''#!/usr/bin/env python3
"""Non-generating environment and asset smoke test."""

import json
import diffusers
import nunchaku
import torch
import transformers
from qwen2511_int4_runtime import BASE_DIR, available_models, load_profiles

required = [
    BASE_DIR / "model_index.json",
    BASE_DIR / "scheduler" / "scheduler_config.json",
    BASE_DIR / "vae" / "config.json",
    BASE_DIR / "text_encoder" / "config.json",
    BASE_DIR / "tokenizer" / "tokenizer_config.json",
    BASE_DIR / "processor" / "preprocessor_config.json",
]
missing = [str(path) for path in required if not path.exists()]
payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "nunchaku": getattr(nunchaku, "__version__", "unknown"),
    "installed_models": available_models(),
    "missing_shared_assets": missing,
    "profiles": load_profiles(),
}
if torch.cuda.is_available():
    payload["gpu"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
    payload["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
print(json.dumps(payload, indent=2))
if missing:
    raise SystemExit(2)
if not payload["installed_models"]:
    raise SystemExit(3)
'''


def write_runtime_files(model_root: Path) -> None:
    section("Writing FrameVision runtime adapter")
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "model_profiles.json").write_text(
        json.dumps(profile_payload(), indent=2), encoding="utf-8"
    )
    for name, content in {
        "qwen2511_int4_runtime.py": RUNTIME_TEMPLATE,
        "qwen2511_int4_cli.py": CLI_TEMPLATE,
        "smoke_test.py": SMOKE_TEMPLATE,
    }.items():
        path = model_root / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        log(f"Wrote: {path}")


def validate_headers(py: Path, checkpoints: list[Path], runtime_env: dict[str, str]) -> None:
    if not checkpoints:
        return
    section("Validating checkpoint headers")
    payload = json.dumps([str(path) for path in checkpoints])
    code = textwrap.dedent(
        f'''
        import json
        from safetensors import safe_open
        paths = json.loads({payload!r})
        result = {{}}
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                result[path] = {{"tensor_count": len(keys), "first_keys": keys[:5]}}
        print(json.dumps(result))
        '''
    )
    log(json.dumps(python_json(py, code, runtime_env), indent=2)[:8000])


def verify_transformer_load(
    py: Path,
    model_root: Path,
    model_key: str,
    runtime_env: dict[str, str],
) -> None:
    section("Optional transformer load verification")
    code = textwrap.dedent(
        f'''
        import json, sys
        from pathlib import Path
        root = Path({str(model_root)!r})
        sys.path.insert(0, str(root))
        from qwen2511_int4_runtime import resolve_model
        from nunchaku import NunchakuQwenImageTransformer2DModel
        key, profile, checkpoint = resolve_model({model_key!r})
        model = NunchakuQwenImageTransformer2DModel.from_pretrained(str(checkpoint))
        print(json.dumps({{"ok": True, "model_key": key, "class": type(model).__name__}}))
        '''
    )
    log(json.dumps(python_json(py, code, runtime_env), indent=2))


def write_state(
    root: Path,
    env_dir: Path,
    model_root: Path,
    selected: list[str],
    gpu_info: dict[str, Any],
    wheel: dict[str, Any],
    stack: dict[str, str],
    repo_tag: str,
    wheel_path: Path,
) -> tuple[Path, Path]:
    section("Writing FrameVision install state")
    save_dir = root / "presets" / "setsave"
    save_dir.mkdir(parents=True, exist_ok=True)
    installed = [
        key
        for key, option in MODEL_OPTIONS.items()
        if (model_root / "int4" / option["filename"]).exists()
    ]
    default_model = "recommended" if "recommended" in installed else (
        selected[0] if selected else (installed[0] if installed else "recommended")
    )
    settings = {
        "schema_version": 1,
        "engine": "qwen2511_int4",
        "default_model": default_model,
        "vram_mode": "auto",
        "steps_locked_to_profile": True,
        "last_updated_unix": int(time.time()),
    }
    settings_path = save_dir / "qwen2511_int4_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    marker = {
        "schema_version": 1,
        "installer_version": INSTALLER_VERSION,
        "installed_at_unix": int(time.time()),
        "engine": "Qwen-Image-Edit-2511",
        "backend": "Nunchaku",
        "precision": "INT4",
        "framevision_root": str(root),
        "environment_dir": str(env_dir),
        "environment_python": str(venv_python(env_dir)),
        "model_root": str(model_root),
        "base_model_dir": str(model_root / "base" / "Qwen-Image-Edit-2511"),
        "int4_dir": str(model_root / "int4"),
        "nunchaku_repo_dir": str(model_root / "repo" / "nunchaku"),
        "runtime_cli": str(model_root / "qwen2511_int4_cli.py"),
        "runtime_adapter": str(model_root / "qwen2511_int4_runtime.py"),
        "installed_models": installed,
        "default_model": default_model,
        "gpu": gpu_info,
        "torch_stack": stack,
        "nunchaku_release_tag": repo_tag,
        "nunchaku_wheel": {
            **wheel,
            "local_path": str(wheel_path),
            "sha256": sha256_file(wheel_path),
        },
        "sources": {
            "nunchaku": NUNCHAKU_GITHUB,
            "base_model": QWEN_BASE_REPO,
            "int4_models": INT4_REPO,
        },
        "compatibility": {
            "rtx_30_series": True,
            "rtx_40_series": True,
            "rtx_50_series": False,
            "rtx_50_reason": "Nunchaku requires FP4 checkpoints on Blackwell.",
        },
    }
    marker_path = save_dir / "qwen2511_int4_install.json"
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    log(f"Wrote: {settings_path}")
    log(f"Wrote: {marker_path}")
    return settings_path, marker_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Qwen2511_INT4_install.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Install Qwen-Image-Edit-2511 Nunchaku INT4 for FrameVision.\n\n"
            "Models:\n"
            "  recommended   quality-r64 Lightning 4-step\n"
            "  fastest       balanced-r32 Lightning 4-step\n"
            "  best-low-step quality-r128-b15 Lightning 4-step\n"
            "  fidelity      mid-r128 Lightning 8-step"
        ),
    )
    parser.add_argument("--root", help="FrameVision root; normally auto-detected")
    parser.add_argument("--model", action="append", help="Repeat to install multiple models")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-repo", action="store_true")
    parser.add_argument("--force-env", action="store_true")
    parser.add_argument("--verify-load", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def print_models() -> None:
    for key, option in MODEL_OPTIONS.items():
        print(
            json.dumps(
                {
                    "key": key,
                    "label": option["label"],
                    "filename": option["filename"],
                    "steps": option["steps"],
                    "rank": option["rank"],
                    "size_gib": option["size_gib"],
                    "aliases": option["aliases"],
                }
            )
        )


def main() -> int:
    args = parse_args()
    if args.list_models:
        print_models()
        return 0

    ensure_supported_host()
    root = detect_root(args.root)
    env_dir = root / "environments" / ".qwen2511_int"
    model_root = root / "models" / "qwen2511_int"
    base_dir = model_root / "base" / "Qwen-Image-Edit-2511"
    int4_dir = model_root / "int4"
    repo_dir = model_root / "repo" / "nunchaku"
    metadata_dir = model_root / "int4_repo_metadata"
    temp_dir = root / "temp" / "qwen2511_int4"

    for path in (
        env_dir.parent,
        model_root,
        base_dir.parent,
        int4_dir,
        repo_dir.parent,
        metadata_dir,
        temp_dir,
        root / "presets" / "setsave",
    ):
        path.mkdir(parents=True, exist_ok=True)

    selected = choose_models(args)
    warn_disk_space(root, selected, args.skip_base or args.env_only)

    section("Installation paths")
    log(f"FrameVision root: {root}")
    log(f"Environment:      {env_dir}")
    log(f"Models/runtime:   {model_root}")
    log(f"Temporary files:  {temp_dir}")
    log(f"Saved settings:   {root / 'presets' / 'setsave'}")

    runtime_env = make_runtime_env(root, model_root, temp_dir)
    py = ensure_environment(env_dir, args.force_env, runtime_env)
    wheel, stack, repo_tag = discover_nunchaku_wheel()
    wheel_path = install_stack(py, stack, wheel, temp_dir, runtime_env)
    gpu_info = verify_gpu_and_runtime(
        py, allow_no_gpu=args.allow_no_gpu, runtime_env=runtime_env
    )

    if not args.skip_repo:
        clone_or_download_repo(repo_dir, temp_dir, repo_tag)
    write_runtime_files(model_root)

    checkpoints: list[Path] = []
    if not args.env_only:
        if not args.skip_base:
            download_base_assets(py, base_dir, runtime_env)
        else:
            log("Skipping shared Qwen assets by request")
        download_repo_metadata(py, metadata_dir, runtime_env)
        for key in selected:
            checkpoints.append(download_int4_checkpoint(py, int4_dir, key, runtime_env))

    validate_headers(py, checkpoints, runtime_env)
    settings_path, marker_path = write_state(
        root,
        env_dir,
        model_root,
        selected,
        gpu_info,
        wheel,
        stack,
        repo_tag,
        wheel_path,
    )

    if args.verify_load and checkpoints:
        verify_transformer_load(py, model_root, selected[0], runtime_env)

    section("Installation complete")
    log(f"Environment Python: {py}")
    log(f"Runtime CLI:        {model_root / 'qwen2511_int4_cli.py'}")
    log(f"Settings:           {settings_path}")
    log(f"Install marker:     {marker_path}")
    print("\nFrameVision subprocess pattern:")
    print(
        f'  "{py}" "{model_root / "qwen2511_int4_cli.py"} '
        '--input "input.png" --output "output.png" '
        '--prompt "your edit instruction" --model recommended --vram-mode auto'
    )
    print("\nCompatibility: INT4 is enabled for RTX 30/40-series. RTX 50-series requires FP4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
