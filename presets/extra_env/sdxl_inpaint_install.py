#!/usr/bin/env python3
"""FrameVision SDXL Inpaint installer.

Creates/updates a portable conda environment at:
    <FrameVisionRoot>/environments/.sdxl_inpaint

This replaces the old root-level venv:
    <FrameVisionRoot>/.sdxl_inpaint

All model/download/cache paths stay inside the FrameVision folder.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV_NAME = ".sdxl_inpaint"
PYTHON_VERSION = "3.11"
TORCH_PACKAGES = [
    "torch==2.6.0+cu124",
    "torchvision==0.21.0+cu124",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


def root_dir() -> Path:
    # Expected location: <root>/presets/extra_env/sdxl_inpaint_install.py
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd().resolve()


def env_dir(root: Path) -> Path:
    return root / "environments" / ENV_NAME


def env_python(root: Path) -> Path:
    e = env_dir(root)
    if os.name == "nt":
        return e / "python.exe"  # conda prefix env
    return e / "bin" / "python"


def main_python(root: Path) -> Path | None:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    return None


def find_conda(root: Path) -> str | None:
    env = os.environ.get("CONDA_EXE")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        root / "miniconda3" / "Scripts" / "conda.exe",
        root / "Miniconda3" / "Scripts" / "conda.exe",
        root / "tools" / "miniconda3" / "Scripts" / "conda.exe",
        root / "presets" / "extra_env" / "miniconda3" / "Scripts" / "conda.exe",
    ])
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            pass
    found = shutil.which("conda")
    return found


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n> " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def pip_install(py: Path, args: list[str]) -> None:
    run([str(py), "-m", "pip", *args])


def ensure_env(root: Path) -> Path:
    e = env_dir(root)
    py = env_python(root)
    conda = find_conda(root)
    print("\n===============================")
    print(" SDXL Inpaint Conda Installer")
    print("===============================")
    print(f"Root: {root}")
    print(f"Env : {e}")
    print(f"Py  : {py}")
    if not conda:
        print("\nERROR: conda was not found.")
        print("This installer now creates a conda env under /environments/.")
        print("Install/ship Miniconda, or make CONDA_EXE point to conda.exe, then rerun.")
        raise SystemExit(1)
    print(f"Conda: {conda}")

    if not py.exists():
        e.parent.mkdir(parents=True, exist_ok=True)
        run([conda, "create", "-y", "-p", str(e), f"python={PYTHON_VERSION}", "pip"])
    else:
        print("Env already exists. Updating packages...")
    if not py.exists():
        print(f"\nERROR: env python was not created: {py}")
        raise SystemExit(1)
    return py


def install_requirements(root: Path, py: Path) -> None:
    req = root / "presets" / "extra_env" / "sdxl_inpaint_req.txt"
    if not req.exists():
        print(f"\nERROR: requirements file not found: {req}")
        raise SystemExit(1)

    pip_install(py, ["install", "--upgrade", "pip", "setuptools", "wheel"])
    pip_install(py, ["install", "--index-url", TORCH_INDEX, *TORCH_PACKAGES])
    pip_install(py, ["install", "-r", str(req)])


def run_background_download(root: Path) -> None:
    bg = root / "scripts" / "background_download.py"
    if not bg.exists():
        print(f"\nWARNING: background_download.py not found: {bg}")
        return
    py = main_python(root)
    if not py:
        print("\nWARNING: no main Python found for background downloads.")
        return
    print("\nRunning background downloads...")
    try:
        run([str(py), str(bg)], cwd=root)
    except SystemExit as e:
        print(f"WARNING: background download returned {e.code}. You can run it again later from the app.")


def main() -> int:
    root = root_dir()
    os.environ.setdefault("HF_HOME", str(root / "models" / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "models" / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "models" / "hf_cache"))
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    py = ensure_env(root)
    install_requirements(root, py)
    run_background_download(root)
    print("\nDone.")
    print(f"SDXL Inpaint env ready: {py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
