#!/usr/bin/env python3
"""
HiAR installer for FrameVision.

What it does:
- Creates a dedicated FrameVision environment at /environments/.hiar
- Clones or updates the HiAR repo into /models/hiar/HiAR
- Installs Python requirements from the repo
- Optionally installs flash-attn separately
- Downloads the HiAR checkpoint (hiar.pt)
- Downloads the Wan2.1-T2V-1.3B base model into the repo's expected wan_models path

The script is intentionally conservative:
- No hidden external cache paths are used for model storage
- Hugging Face downloads go directly into FrameVision folders
- It supports skip flags so pieces can be retried independently
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional


HIAR_GIT_URL = "https://github.com/Jacky-hate/HiAR.git"
HIAR_HF_REPO = "jackyhate/HiAR"
HIAR_CKPT_NAME = "hiar.pt"
WAN_BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
ENV_NAME = ".hiar"
MODEL_ROOT_NAME = "hiar"


class InstallError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = message.rstrip()
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


LOG: Optional[Logger] = None


def log(message: str) -> None:
    if LOG is None:
        print(message)
    else:
        LOG.write(message)


def run(cmd: Iterable[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    cmd_list = [str(x) for x in cmd]
    shown = " ".join(f'"{c}"' if " " in c else c for c in cmd_list)
    log(f"[run] {shown}")
    result = subprocess.run(cmd_list, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        raise InstallError(f"Command failed with exit code {result.returncode}: {shown}")


def capture(cmd: Iterable[str], *, cwd: Optional[Path] = None) -> str:
    cmd_list = [str(x) for x in cmd]
    result = subprocess.run(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise InstallError(result.stdout.strip() or f"Command failed: {' '.join(cmd_list)}")
    return result.stdout.strip()


def find_framevision_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "helpers").exists() or (candidate / "presets").exists() or (candidate / "models").exists():
            return candidate
    return start.resolve()


def pick_python(python_hint: Optional[str]) -> str:
    if python_hint:
        return python_hint

    candidates = []
    if os.name == "nt":
        candidates.extend([
            ["py", "-3.10", "-c", "import sys; print(sys.executable)"],
            ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
            ["py", "-3", "-c", "import sys; print(sys.executable)"],
        ])
    candidates.append([sys.executable, "-c", "import sys; print(sys.executable)"])

    for cmd in candidates:
        try:
            found = capture(cmd).strip()
            if found:
                log(f"[info] Using base Python: {found}")
                return found
        except Exception:
            continue

    raise InstallError("Could not find a usable Python interpreter. Pass --python explicitly.")


def create_venv(base_python: str, env_dir: Path) -> Path:
    python_exe = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python_exe.exists():
        log(f"[skip] Environment already exists: {env_dir}")
        return python_exe

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    run([base_python, "-m", "venv", str(env_dir)])
    if not python_exe.exists():
        raise InstallError(f"Environment was created but python was not found at: {python_exe}")
    return python_exe


def ensure_git() -> None:
    if shutil.which("git"):
        return
    raise InstallError("git was not found on PATH. Install Git first, then rerun this installer.")


def clone_or_update_repo(repo_dir: Path, branch: str = "main") -> None:
    ensure_git()
    if (repo_dir / ".git").exists():
        log(f"[info] Updating existing repo: {repo_dir}")
        run(["git", "fetch", "origin"], cwd=repo_dir)
        run(["git", "checkout", branch], cwd=repo_dir)
        run(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--branch", branch, HIAR_GIT_URL, str(repo_dir)])


def pip_install(python_exe: Path, args: list[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    run([str(python_exe), "-m", "pip"] + args, cwd=cwd, env=env)



def install_cuda_torch(python_exe: Path) -> None:
    print("[info] Installing CUDA-enabled PyTorch (cu128)")
    # HiAR/Wan requires a CUDA-enabled PyTorch build. Do not let pip keep a CPU wheel.
    extra_index = "https://download.pytorch.org/whl/cu128"

    # Remove any existing CPU-only installs first.
    subprocess.call([str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    subprocess.check_call([
        str(python_exe), "-m", "pip", "install",
        "--index-url", extra_index,
        "--force-reinstall",
        "--no-cache-dir",
        "torch", "torchvision", "torchaudio",
    ])

    # Verify that the installed torch build has CUDA support.
    subprocess.check_call([
        str(python_exe), "-u", "-c",
        "import torch; "
        "print('torch_version=', torch.__version__); "
        "print('torch_cuda_version=', getattr(torch.version, 'cuda', None)); "
        "print('cuda_available=', torch.cuda.is_available()); "
        "raise SystemExit(0 if getattr(torch.version, 'cuda', None) else 1)"
    ])





def install_flash_attn_wheel(python_exe: Path) -> None:
    print("[info] Installing Flash Attention wheel at end of installer")
    subprocess.check_call([
        str(python_exe), "-m", "pip", "install",
        "--no-cache-dir",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3%2Bcu128torch2.10-cp311-cp311-win_amd64.whl",
    ])


def install_repo_requirements(python_exe: Path, repo_dir: Path, install_flash_attn: bool) -> None:
    pip_install(python_exe, ["install", "--upgrade", "pip", "setuptools", "wheel"])
    install_cuda_torch(python_exe)
    pip_install(python_exe, ["install", "huggingface_hub", "hf_transfer"])

    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        raise InstallError(f"Missing requirements file: {req_file}")

    raw = req_file.read_text(encoding="utf-8")
    filtered = []
    for token in raw.split():
        if token.strip().lower() == "flash-attn":
            continue
        filtered.append(token)

    if filtered:
        pip_install(python_exe, ["install"] + filtered)

    # Extra runtime deps used by HiAR/Wan imports that may not be pulled in
    # reliably by the repo requirements alone.
    pip_install(python_exe, ["install", "easydict", "ftfy"])

    # Install the known-good Windows prebuilt Flash Attention wheel last so it
    # does not get overwritten by other package operations.
    install_flash_attn_wheel(python_exe)

    if install_flash_attn:
        log("[info] Installing flash-attn separately with --no-build-isolation")
        pip_install(python_exe, ["install", "flash-attn", "--no-build-isolation"])
    else:
        log("[info] Skipping flash-attn install. Use --install-flash-attn when your Python/CUDA/PyTorch tuple is confirmed compatible.")


def huggingface_env(base_env: Optional[dict], target_root: Path) -> dict:
    env = dict(base_env or os.environ)
    # Keep cache/download state inside FrameVision for portability.
    env["HF_HOME"] = str(target_root / ".hf_home")
    env["HF_HUB_CACHE"] = str(target_root / ".hf_home" / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(target_root / ".hf_home" / "hub")
    env["HF_XET_CACHE"] = str(target_root / ".hf_home" / "xet")
    env["TRANSFORMERS_CACHE"] = str(target_root / ".hf_home" / "transformers")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    return env


HUGGINGFACE_SNIPPET = r'''
from huggingface_hub import hf_hub_download, snapshot_download
import os

mode = os.environ["HIAR_DL_MODE"]
repo_id = os.environ["HIAR_REPO_ID"]
out_dir = os.environ["HIAR_OUT_DIR"]
local_dir = os.environ["HIAR_LOCAL_DIR"]
filename = os.environ.get("HIAR_FILENAME", "")
revision = os.environ.get("HIAR_REVISION") or None

if mode == "file":
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=revision,
    )
elif mode == "snapshot":
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=revision,
        resume_download=True,
    )
else:
    raise SystemExit(f"Unknown mode: {mode}")
'''


def hf_download_file(python_exe: Path, repo_id: str, filename: str, local_dir: Path, framevision_root: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    env = huggingface_env(os.environ, framevision_root)
    env["HIAR_DL_MODE"] = "file"
    env["HIAR_REPO_ID"] = repo_id
    env["HIAR_FILENAME"] = filename
    env["HIAR_LOCAL_DIR"] = str(local_dir)
    env["HIAR_OUT_DIR"] = str(local_dir)
    run([str(python_exe), "-c", HUGGINGFACE_SNIPPET], env=env)


def hf_snapshot_download(python_exe: Path, repo_id: str, local_dir: Path, framevision_root: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    env = huggingface_env(os.environ, framevision_root)
    env["HIAR_DL_MODE"] = "snapshot"
    env["HIAR_REPO_ID"] = repo_id
    env["HIAR_LOCAL_DIR"] = str(local_dir)
    env["HIAR_OUT_DIR"] = str(local_dir)
    run([str(python_exe), "-c", HUGGINGFACE_SNIPPET], env=env)


def write_readme_hint(repo_dir: Path, env_python: Path) -> None:
    hint = textwrap.dedent(
        f"""
        HiAR was installed for FrameVision.

        Repo:
          {repo_dir}

        Environment Python:
          {env_python}

        Typical inference example:
          {env_python} inference.py --config_path configs/hiar.yaml --checkpoint_path ckpts/hiar.pt --data_path data/prompts.txt --output_folder outputs/ --num_output_frames 21 --use_ema --inference_method timestep_first
        """
    ).strip() + "\n"
    (repo_dir / "FRAMEVISION_HIAR_INSTALL.txt").write_text(hint, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install HiAR into FrameVision folders.")
    parser.add_argument("--root", type=str, default=None, help="FrameVision root folder. Defaults to auto-detect from this script location.")
    parser.add_argument("--python", type=str, default=None, help="Base Python to use for the venv, e.g. C:/Python310/python.exe")
    parser.add_argument("--branch", type=str, default="main", help="Git branch for the HiAR repo")
    parser.add_argument("--skip-env", action="store_true", help="Skip creating the virtual environment")
    parser.add_argument("--skip-repo", action="store_true", help="Skip cloning/updating the HiAR repo")
    parser.add_argument("--skip-pip", action="store_true", help="Skip pip dependency installation")
    parser.add_argument("--skip-checkpoint", action="store_true", help="Skip downloading hiar.pt")
    parser.add_argument("--skip-base-model", action="store_true", help="Skip downloading Wan2.1-T2V-1.3B")
    parser.add_argument("--install-flash-attn", action="store_true", help="Also install flash-attn with --no-build-isolation")
    return parser.parse_args()


def main() -> int:
    global LOG
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    framevision_root = Path(args.root).resolve() if args.root else find_framevision_root(script_dir)

    logs_dir = framevision_root / "logs"
    LOG = Logger(logs_dir / "hiar_install.log")

    log("=" * 72)
    log("HiAR installer for FrameVision")
    log("=" * 72)
    log(f"[info] Platform: {platform.platform()}")
    log(f"[info] FrameVision root: {framevision_root}")

    models_root = framevision_root / "models" / MODEL_ROOT_NAME
    repo_dir = models_root / "HiAR"
    env_dir = framevision_root / "environments" / ENV_NAME
    env_python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    base_python = pick_python(args.python)

    if not args.skip_env:
        env_python = create_venv(base_python, env_dir)
    elif not env_python.exists():
        raise InstallError(f"--skip-env was used, but env python does not exist: {env_python}")

    if not args.skip_repo:
        clone_or_update_repo(repo_dir, branch=args.branch)
    elif not repo_dir.exists():
        raise InstallError(f"--skip-repo was used, but repo does not exist: {repo_dir}")

    if not args.skip_pip:
        install_repo_requirements(env_python, repo_dir, install_flash_attn=args.install_flash_attn)

    if not args.skip_checkpoint:
        ckpt_dir = repo_dir / "ckpts"
        log(f"[info] Downloading HiAR checkpoint into: {ckpt_dir}")
        hf_download_file(env_python, HIAR_HF_REPO, HIAR_CKPT_NAME, ckpt_dir, framevision_root)

    if not args.skip_base_model:
        wan_dir = repo_dir / "wan_models" / "Wan2.1-T2V-1.3B"
        log(f"[info] Downloading Wan base model into: {wan_dir}")
        hf_snapshot_download(env_python, WAN_BASE_REPO, wan_dir, framevision_root)

    write_readme_hint(repo_dir, env_python)

    log("[done] HiAR install steps completed.")
    log(f"[done] Repo: {repo_dir}")
    log(f"[done] Env Python: {env_python}")
    log(f"[done] Log file: {logs_dir / 'hiar_install.log'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as exc:
        log(f"[error] {exc}")
        raise SystemExit(1)
