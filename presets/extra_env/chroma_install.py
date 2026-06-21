"""
FrameVision SPARK.Chroma optional installer.

Environment behavior:
- If <FrameVision root>/environments/.images_models already exists, reuse it and skip
  environment creation and package installation.
- Otherwise create/use a dedicated portable environment at:
      <FrameVision root>/environments/.chroma

Downloads model files to:
    <FrameVision root>/models/chroma/SPARK.Chroma_v1

This installer intentionally does NOT install Gradio and does NOT launch a server/browser.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_REPO = "SG161222/SPARK.Chroma_v1"
MODEL_SUBDIR = Path("models") / "chroma" / "SPARK.Chroma_v1"
SHARED_ENV_SUBDIR = Path("environments") / ".images_models"
DEDICATED_ENV_SUBDIR = Path("environments") / ".chroma"
REQ_REL = Path("presets") / "extra_env" / "chroma_requirements.txt"

ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

REQUIRED_FILES = [
    Path("model_index.json"),
    Path("scheduler") / "scheduler_config.json",
    Path("text_encoder") / "model.safetensors.index.json",
    Path("tokenizer") / "tokenizer_config.json",
    Path("transformer") / "diffusion_pytorch_model.safetensors.index.json",
    Path("vae") / "diffusion_pytorch_model.safetensors",
]


def framevision_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_candidates_for_env(env_dir: Path) -> list[Path]:
    return [
        env_dir / "python.exe",              # Windows conda prefix
        env_dir / "Scripts" / "python.exe",  # Windows venv
        env_dir / "bin" / "python",          # Linux/macOS conda or venv
    ]


def first_existing_python(env_dir: Path) -> Path | None:
    for py in python_candidates_for_env(env_dir):
        if py.exists():
            return py
    return None


def shared_env_dir(root: Path) -> Path:
    return root / SHARED_ENV_SUBDIR


def dedicated_env_dir(root: Path) -> Path:
    return root / DEDICATED_ENV_SUBDIR


def preferred_env_dir(root: Path) -> Path:
    shared = shared_env_dir(root)
    if first_existing_python(shared):
        return shared
    return dedicated_env_dir(root)


def env_python(root: Path) -> Path:
    env_dir = preferred_env_dir(root)
    py = first_existing_python(env_dir)
    if py is not None:
        return py
    candidates = python_candidates_for_env(env_dir)
    return candidates[0] if os.name == "nt" else candidates[-1]


def env_label(root: Path) -> str:
    env_dir = preferred_env_dir(root)
    if env_dir == shared_env_dir(root):
        return ".images_models"
    return ".chroma"


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[chroma-install] " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.check_call(cmd, cwd=str(cwd or framevision_root()), env=merged_env)


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def create_or_pick_env(root: Path, *, force_venv: bool = False) -> tuple[Path, str]:
    shared_dir = shared_env_dir(root)
    shared_py = first_existing_python(shared_dir)
    if shared_py is not None:
        print(f"[chroma-install] Found shared image environment, reusing it: {shared_dir}", flush=True)
        print("[chroma-install] Skipping environment creation and dependency install.", flush=True)
        return shared_py, "shared"

    env_dir = dedicated_env_dir(root)
    py = first_existing_python(env_dir)
    if py is not None:
        print(f"[chroma-install] Dedicated Chroma environment already exists: {env_dir}", flush=True)
        return py, "dedicated"

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    conda = shutil.which("conda")
    if conda and not force_venv:
        print(f"[chroma-install] Creating dedicated conda env at: {env_dir}", flush=True)
        run([conda, "create", "-y", "-p", str(env_dir), "python=3.11", "pip"])
    else:
        print(f"[chroma-install] Creating dedicated venv at: {env_dir}", flush=True)
        run([sys.executable, "-m", "venv", str(env_dir)])

    py = first_existing_python(env_dir)
    if py is None:
        checked = ", ".join(str(c) for c in python_candidates_for_env(env_dir))
        raise RuntimeError(f"Environment was created, but Python was not found. Checked: {checked}")
    print(f"[chroma-install] Dedicated environment Python: {py}", flush=True)
    return py, "dedicated"


def install_deps(root: Path, *, force_venv: bool = False) -> None:
    py, env_kind = create_or_pick_env(root, force_venv=force_venv)
    if env_kind == "shared":
        return

    req = root / REQ_REL
    if not req.exists():
        raise FileNotFoundError(f"Missing requirements file: {req}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    if not has_nvidia_gpu():
        raise RuntimeError(
            "No NVIDIA GPU was detected with nvidia-smi. "
            "Stopping instead of installing CPU-only Torch."
        )

    print("[chroma-install] Installing CUDA PyTorch cu128. No CPU fallback will be used.", flush=True)
    run([str(py), "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"])
    run([str(py), "-m", "pip", "install", "-r", str(req)])


def model_ready(model_dir: Path) -> bool:
    return all((model_dir / rel).exists() for rel in REQUIRED_FILES)


def download_model(root: Path) -> None:
    py = env_python(root)
    if not py.exists():
        raise RuntimeError(
            "No usable Chroma/image environment was found. "
            "Run with --install-deps first or create /environments/.images_models/."
        )

    helper = root / "helpers" / "chroma.py"
    if not helper.exists():
        raise FileNotFoundError(f"Missing helper file: {helper}")

    print(f"[chroma-install] Using environment: {env_label(root)} ({py})", flush=True)
    env = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    run([str(py), str(helper), "--download-only"], env=env, cwd=root)

    model_dir = root / MODEL_SUBDIR
    if not model_ready(model_dir):
        missing = [str(rel) for rel in REQUIRED_FILES if not (model_dir / rel).exists()]
        raise RuntimeError("Download finished, but required files are missing: " + ", ".join(missing))

    print(f"[chroma-install] Model ready at: {model_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameVision SPARK.Chroma installer")
    parser.add_argument("--install-deps", action="store_true", help="Create/update environment and install packages")
    parser.add_argument("--download-model", action="store_true", help="Download/repair SPARK.Chroma model")
    parser.add_argument("--force-venv", action="store_true", help="Use venv even if conda exists when creating a dedicated Chroma env")
    args = parser.parse_args()

    if not args.install_deps and not args.download_model:
        args.install_deps = True
        args.download_model = True

    root = framevision_root()
    print(f"[chroma-install] FrameVision root: {root}", flush=True)
    print(f"[chroma-install] Preferred environment: {env_label(root)}", flush=True)

    if args.install_deps:
        install_deps(root, force_venv=args.force_venv)

    if args.download_model:
        download_model(root)

    print("[chroma-install] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
