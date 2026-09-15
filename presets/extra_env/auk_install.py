#!/usr/bin/env python3
"""
FrameVision AuK-Flash installer
==============================

Intended destination:
    /presets/extra_env/auk_install.py

Installs:
    /environments/.auk/
    /models/AuK/repo/
    /models/AuK/ckpts/AuK-Flash/
    /models/AuK/ckpts/Qwen2.5-Omni-3B/

Logs:
    /logs/auk_install.log

Temporary installer files:
    /temp/auk_install/
    (removed after a successful install)

Design goals:
- Windows / FrameVision friendly.
- Uses uv and a dedicated Python 3.10 environment.
- Installs core AuK inference only.
- Uses aria2c with multi-connection + resume for fast model downloads.
- Downloads/keeps a local portable aria2c.exe if one is not already available.
- Falls back to Hugging Face hf_xet/snapshot_download if aria2 cannot download
  a model repository reliably.
- Safe to re-run: existing valid files are reused and partial downloads resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# FrameVision paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()

# Expected location is <FrameVision>/presets/extra_env/auk_install.py.
# Keep a fallback so the installer can also be tested/copied elsewhere.
if SCRIPT_PATH.parent.name.lower() == "extra_env" and SCRIPT_PATH.parent.parent.name.lower() == "presets":
    FRAMEVISION_ROOT = SCRIPT_PATH.parents[2]
else:
    FRAMEVISION_ROOT = Path.cwd().resolve()

ENV_DIR = FRAMEVISION_ROOT / "environments" / ".auk"
MODELS_ROOT = FRAMEVISION_ROOT / "models" / "AuK"
REPO_DIR = MODELS_ROOT / "repo"
CKPTS_DIR = MODELS_ROOT / "ckpts"
AUK_FLASH_DIR = CKPTS_DIR / "AuK-Flash"
QWEN_DIR = CKPTS_DIR / "Qwen2.5-Omni-3B"

LOG_DIR = FRAMEVISION_ROOT / "logs"
LOG_FILE = LOG_DIR / "auk_install.log"
TEMP_ROOT = FRAMEVISION_ROOT / "temp" / "auk_install"

# Portable aria2 is retained because it is useful for repairs/re-runs.
ARIA2_DIR = FRAMEVISION_ROOT / "presets" / "extra_env" / "aria2"
ARIA2_EXE = ARIA2_DIR / "aria2c.exe"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

AUK_GIT_URL = "https://github.com/Tencent-Hunyuan/AuK.git"
AUK_GITHUB_ZIP = "https://github.com/Tencent-Hunyuan/AuK/archive/refs/heads/main.zip"

AUK_FLASH_REPO = "tencent/AuK-Flash"
QWEN_REPO = "Qwen/Qwen2.5-Omni-3B"
HF_REVISION = "main"

# aria2 1.37.0 is still the latest official release at installer creation time.
ARIA2_VERSION = "1.37.0"
ARIA2_ZIP_URL = (
    "https://github.com/aria2/aria2/releases/download/"
    "release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
)
ARIA2_ZIP_SHA256 = "67d015301eef0b612191212d564c5bb0a14b5b9c4796b76454276a4d28d9b288"

# AuK-Flash has exactly these runtime files at the repo root.
AUK_FLASH_REQUIRED = (
    "auk_flash.safetensors",
    "config.yaml",
    "vae.safetensors",
)

# Files we do not need from model repositories.
HF_SKIP_NAMES = {
    ".gitattributes",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
}
HF_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp4", ".mov",
}

# Runtime file types used by Transformers/Qwen checkpoints.
HF_RUNTIME_SUFFIXES = {
    ".json", ".txt", ".model", ".safetensors", ".bin",
    ".tiktoken", ".jinja", ".py", ".yaml", ".yml",
}


# ---------------------------------------------------------------------------
# Logging / process helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [AuK install] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def log(msg: str) -> None:
    logging.info(msg)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    pretty = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    log(f"> {pretty}")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
    )


def exe_name(base: str) -> str:
    return f"{base}.exe" if os.name == "nt" else base


def env_python() -> Path:
    if os.name == "nt":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def env_executable(name: str) -> Path:
    if os.name == "nt":
        return ENV_DIR / "Scripts" / f"{name}.exe"
    return ENV_DIR / "bin" / name


def find_command(name: str) -> str | None:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def urlopen_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FrameVision-AuK-Installer/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_small(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FrameVision-AuK-Installer/1.0"},
    )
    log(f"Downloading: {url}")
    with urllib.request.urlopen(req, timeout=timeout) as src, open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------

def uv_prefix() -> list[str]:
    direct = find_command("uv")
    if direct:
        return [direct]

    # uv installed into the Python running this installer.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "uv"]
    except Exception:
        pass

    log("uv was not found. Installing uv into the current Python...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
    return [sys.executable, "-m", "uv"]


def ensure_uv_environment() -> None:
    uv = uv_prefix()

    if env_python().exists():
        log(f"AuK environment already exists: {ENV_DIR}")
        return

    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    log("Creating dedicated AuK Python 3.10 environment with uv...")

    # uv can download/manage Python 3.10 itself if it is not already installed.
    run(uv + ["python", "install", "3.10"])
    run(uv + ["venv", str(ENV_DIR), "--python", "3.10"])

    if not env_python().exists():
        raise RuntimeError(f"uv finished but environment Python was not created: {env_python()}")


# ---------------------------------------------------------------------------
# aria2
# ---------------------------------------------------------------------------

def locate_aria2() -> Path | None:
    # Prefer a user/system installation.
    found = find_command("aria2c")
    if found:
        return Path(found)

    if ARIA2_EXE.exists():
        return ARIA2_EXE

    return None


def install_portable_aria2() -> Path | None:
    if os.name != "nt":
        found = locate_aria2()
        if found:
            return found
        log("aria2 auto-bootstrap is Windows-only; continuing without aria2.")
        return None

    existing = locate_aria2()
    if existing:
        log(f"Using aria2c: {existing}")
        return existing

    log("aria2c not found. Downloading the official portable 64-bit aria2 build...")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_ROOT / f"aria2-{ARIA2_VERSION}-win-64bit.zip"

    try:
        download_small(ARIA2_ZIP_URL, zip_path)
        digest = sha256_file(zip_path)
        if digest.lower() != ARIA2_ZIP_SHA256.lower():
            raise RuntimeError(
                "aria2 archive SHA256 mismatch.\n"
                f"Expected: {ARIA2_ZIP_SHA256}\n"
                f"Got:      {digest}"
            )

        extract_dir = TEMP_ROOT / "aria2_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        candidates = list(extract_dir.rglob("aria2c.exe"))
        if not candidates:
            raise RuntimeError("aria2c.exe was not found inside the downloaded archive.")

        ARIA2_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], ARIA2_EXE)

        if not ARIA2_EXE.exists():
            raise RuntimeError("Failed to copy portable aria2c.exe.")

        log(f"Portable aria2c installed: {ARIA2_EXE}")
        return ARIA2_EXE
    except Exception as exc:
        logging.warning(f"Could not bootstrap aria2c: {exc}")
        logging.warning("The installer will use Hugging Face hf_xet as fallback.")
        return None


def aria2_download(url: str, destination: Path, aria2: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    # -c resumes partial files. Multiple segments/connections are particularly
    # useful on large weight shards when the server accepts byte ranges.
    cmd = [
        str(aria2),
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=16M",
        "--piece-length=4M",
        "--file-allocation=none",
        "--max-tries=8",
        "--retry-wait=2",
        "--timeout=60",
        "--connect-timeout=20",
        "--lowest-speed-limit=10K",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--remote-time=true",
        "--console-log-level=warn",
        "--summary-interval=5",
        "--dir", str(destination.parent),
        "--out", destination.name,
        url,
    ]
    run(cmd)


# ---------------------------------------------------------------------------
# AuK source repository
# ---------------------------------------------------------------------------

def ensure_auk_repo(aria2: Path | None) -> None:
    git = find_command("git")

    if (REPO_DIR / ".git").exists() and git:
        log("AuK repository already exists; updating with git pull --ff-only...")
        result = run(
            [git, "-C", str(REPO_DIR), "pull", "--ff-only"],
            check=False,
        )
        if result.returncode != 0:
            logging.warning("AuK git pull failed. Keeping the existing repository checkout.")
        return

    if (REPO_DIR / "pyproject.toml").exists():
        log(f"AuK source already exists: {REPO_DIR}")
        return

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    if git:
        log("Cloning official AuK repository...")
        run([git, "clone", "--depth", "1", AUK_GIT_URL, str(REPO_DIR)])
        return

    log("git was not found; downloading the AuK GitHub source archive instead...")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    source_zip = TEMP_ROOT / "AuK-main.zip"

    if aria2:
        try:
            aria2_download(AUK_GITHUB_ZIP, source_zip, aria2)
        except Exception as exc:
            logging.warning(f"aria2 source download failed: {exc}; using urllib.")
            download_small(AUK_GITHUB_ZIP, source_zip)
    else:
        download_small(AUK_GITHUB_ZIP, source_zip)

    extract_dir = TEMP_ROOT / "auk_source_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip, "r") as zf:
        zf.extractall(extract_dir)

    candidates = [p for p in extract_dir.iterdir() if p.is_dir() and (p / "pyproject.toml").exists()]
    if not candidates:
        raise RuntimeError("Downloaded AuK source archive does not contain pyproject.toml.")

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    shutil.move(str(candidates[0]), str(REPO_DIR))


# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------

def install_python_dependencies() -> None:
    uv = uv_prefix()
    py = env_python()

    log("Updating pip tooling in AuK environment...")
    run(uv + ["pip", "install", "--python", str(py), "--upgrade", "pip", "setuptools", "wheel"])

    if os.name == "nt":
        # AuK is tested on the PyTorch 2.7 ABI line. Use the CUDA 12.8 wheels on
        # Windows so FrameVision does not accidentally receive CPU-only torch.
        log("Installing tested PyTorch 2.7 CUDA stack...")
        run(
            uv + [
                "pip", "install",
                "--python", str(py),
                "--index-url", "https://download.pytorch.org/whl/cu128",
                "torch==2.7.1",
                "torchvision==0.22.1",
                "torchaudio==2.7.1",
            ]
        )

    log("Installing AuK core inference package and dependencies...")
    run(
        uv + [
            "pip", "install",
            "--python", str(py),
            "-e", str(REPO_DIR),
        ],
        cwd=REPO_DIR,
    )

    # Standalone AuK helper GUI dependencies. FrameVision itself may already
    # have these, but the dedicated AuK environment must be self-contained so
    # helpers/auk_helper.py can also run outside FrameVision.
    log("Installing standalone AuK GUI dependencies (PySide6 + pyqtgraph)...")
    run(
        uv + [
            "pip", "install",
            "--python", str(py),
            "PySide6>=6.7,<7",
            "pyqtgraph>=0.13.7,<0.14",
        ]
    )

    # Keep huggingface_hub inside the version range required by AuK's
    # Transformers dependency. Newer 1.x huggingface_hub releases are not
    # compatible with the Transformers version currently installed by AuK.
    # hf_xet still provides the supported fast Hugging Face transfer backend.
    log("Installing compatible Hugging Face downloader with hf_xet support...")
    run(
        uv + [
            "pip", "install",
            "--python", str(py),
            "--upgrade",
            "huggingface_hub[hf_xet]>=0.34.0,<1.0",
        ]
    )

    # Verify the environment after applying the explicit hub constraint.
    # This also repairs existing AuK installs that received hub 1.x.
    log("Checking AuK dependency compatibility...")
    run(
        uv + [
            "pip", "check",
            "--python", str(py),
        ]
    )


# ---------------------------------------------------------------------------
# Hugging Face downloads
# ---------------------------------------------------------------------------

def hf_api_files(repo_id: str) -> list[str]:
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    url = f"https://huggingface.co/api/models/{encoded}?revision={urllib.parse.quote(HF_REVISION)}"
    data = urlopen_json(url)

    files: list[str] = []
    for sibling in data.get("siblings", []):
        name = sibling.get("rfilename")
        if isinstance(name, str) and name:
            files.append(name)
    if not files:
        raise RuntimeError(f"Hugging Face API returned no files for {repo_id}.")
    return files


def hf_resolve_url(repo_id: str, filename: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    file_path = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{repo}/resolve/{HF_REVISION}/{file_path}?download=true"


def is_runtime_model_file(filename: str) -> bool:
    p = Path(filename)
    if p.name in HF_SKIP_NAMES:
        return False
    if p.suffix.lower() in HF_SKIP_SUFFIXES:
        return False

    # Hidden metadata folders are not needed in FrameVision's runtime copy.
    if any(part.startswith(".") for part in p.parts):
        return False

    return p.suffix.lower() in HF_RUNTIME_SUFFIXES


def aria2_download_repo(
    repo_id: str,
    destination: Path,
    aria2: Path,
    *,
    exact_files: Iterable[str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    if exact_files is not None:
        files = list(exact_files)
    else:
        files = [name for name in hf_api_files(repo_id) if is_runtime_model_file(name)]

    if not files:
        raise RuntimeError(f"No runtime files selected for {repo_id}.")

    log(f"aria2 selected {len(files)} runtime file(s) from {repo_id}.")

    for index, filename in enumerate(files, start=1):
        target = destination / Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)

        # aria2's -c option will resume a partial file. Existing files are still
        # passed through aria2 so the remote size/continuation logic can validate
        # completion rather than blindly assuming a truncated file is valid.
        log(f"[{index}/{len(files)}] {repo_id}: {filename}")
        aria2_download(hf_resolve_url(repo_id, filename), target, aria2)


def hf_snapshot_download(repo_id: str, destination: Path) -> None:
    py = env_python()
    code = (
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={repo_id!r}, revision={HF_REVISION!r}, "
        f"local_dir={str(destination)!r}, "
        "ignore_patterns=['*.md','.gitattributes','*.png','*.jpg','*.jpeg','*.gif','*.webp','*.svg'])\n"
    )

    child_env = os.environ.copy()
    child_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # hf_xet is automatically used by modern huggingface_hub.
    run([str(py), "-c", code], env=child_env)


def download_model_repo(
    repo_id: str,
    destination: Path,
    aria2: Path | None,
    *,
    exact_files: Iterable[str] | None = None,
) -> None:
    if aria2:
        try:
            aria2_download_repo(
                repo_id,
                destination,
                aria2,
                exact_files=exact_files,
            )
            return
        except Exception as exc:
            logging.warning(f"aria2 download path failed for {repo_id}: {exc}")
            logging.warning("Falling back to Hugging Face snapshot_download + hf_xet.")

    hf_snapshot_download(repo_id, destination)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def require_file(path: Path, min_size: int = 1) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    if path.stat().st_size < min_size:
        raise RuntimeError(f"Required file looks incomplete: {path}")


def validate_install() -> None:
    log("Validating AuK installation...")

    if not env_python().exists():
        raise RuntimeError(f"AuK environment is missing: {ENV_DIR}")

    require_file(REPO_DIR / "pyproject.toml")
    require_file(AUK_FLASH_DIR / "config.yaml", 100)
    require_file(AUK_FLASH_DIR / "auk_flash.safetensors", 100 * 1024 * 1024)
    require_file(AUK_FLASH_DIR / "vae.safetensors", 100 * 1024 * 1024)

    require_file(QWEN_DIR / "config.json", 100)
    require_file(QWEN_DIR / "tokenizer_config.json", 100)
    require_file(QWEN_DIR / "model.safetensors.index.json", 100)

    shards = sorted(QWEN_DIR.glob("model-*-of-*.safetensors"))
    if not shards:
        raise RuntimeError(f"No Qwen safetensor shards found in {QWEN_DIR}")

    # Real inference import smoke test from the dedicated environment.
    # Importing only `auk` is not sufficient because Transformers is first
    # imported by auk.infer.infer_auk. This catches dependency mismatches
    # before the GUI reports a generation failure.
    code = (
        "import torch\n"
        "import huggingface_hub\n"
        "import transformers\n"
        "import PySide6\n"
        "import pyqtgraph\n"
        "from auk.infer.infer_auk import AukInfer, save_audio\n"
        "print('AUK_INFERENCE_IMPORT_OK')\n"
        "print('pyside6', PySide6.__version__)\n"
        "print('pyqtgraph', pyqtgraph.__version__)\n"
        "print('torch', torch.__version__)\n"
        "print('transformers', transformers.__version__)\n"
        "print('huggingface_hub', huggingface_hub.__version__)\n"
        "print('cuda_available', torch.cuda.is_available())\n"
        "if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))\n"
    )
    run([str(env_python()), "-c", code], cwd=MODELS_ROOT)

    log("Validation passed.")


def write_install_manifest() -> None:
    manifest = {
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "framevision_root": str(FRAMEVISION_ROOT),
        "environment": str(ENV_DIR),
        "repo": str(REPO_DIR),
        "models_root": str(MODELS_ROOT),
        "auk_flash": str(AUK_FLASH_DIR),
        "qwen": str(QWEN_DIR),
        "aria2": str(locate_aria2() or ""),
        "python": str(env_python()),
        "status": "ok",
    }
    path = MODELS_ROOT / "install_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install AuK-Flash for FrameVision.")
    parser.add_argument(
        "--no-aria2",
        action="store_true",
        help="Skip aria2 and use Hugging Face hf_xet downloads only.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-run installation/download validation even if files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    log("=" * 72)
    log("FrameVision AuK-Flash installer")
    log(f"FrameVision root : {FRAMEVISION_ROOT}")
    log(f"Environment      : {ENV_DIR}")
    log(f"AuK source       : {REPO_DIR}")
    log(f"Checkpoints      : {CKPTS_DIR}")
    log(f"Log              : {LOG_FILE}")
    log("=" * 72)

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        aria2 = None if args.no_aria2 else install_portable_aria2()

        ensure_uv_environment()
        ensure_auk_repo(aria2)
        install_python_dependencies()

        log("Downloading AuK-Flash checkpoint...")
        download_model_repo(
            AUK_FLASH_REPO,
            AUK_FLASH_DIR,
            aria2,
            exact_files=AUK_FLASH_REQUIRED,
        )

        log("Downloading Qwen2.5-Omni-3B encoder...")
        download_model_repo(
            QWEN_REPO,
            QWEN_DIR,
            aria2,
        )

        validate_install()
        write_install_manifest()

        log("=" * 72)
        log("AuK-Flash installation completed successfully.")
        log(f"Environment : {ENV_DIR}")
        log(f"Model root  : {MODELS_ROOT}")
        log("=" * 72)

        # Only clean temporary installer data after the entire installation has
        # validated. Partial downloads live in their final destinations so aria2
        # can resume them on the next run.
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
        return 0

    except KeyboardInterrupt:
        logging.error("Installation cancelled by user. Partial downloads were kept for resume.")
        return 130
    except Exception:
        logging.exception("AuK installation failed. Re-run the installer to resume/repair.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
