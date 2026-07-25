from __future__ import annotations

"""FrameVision installer/downloader for Microsoft Mage-Flow models.

Install layout
--------------
<root>/presets/extra_env/Mage_edit.py              (this installer)
<root>/environments/.mage_edit/                    (shared isolated environment)
<root>/models/mage_edit/repo/Mage/                 (official Microsoft source)
<root>/models/mage_edit/Mage-Flow-Turbo/           (text-to-image Turbo, optional)
<root>/models/mage_edit/Mage-Flow-Edit-Turbo/      (image-edit Turbo, optional)
<root>/models/mage_edit/Mage-Flow-Edit/            (image-edit RL model, optional)

The Base checkpoints are intentionally not offered. The installer can download
one, several, or all three supported checkpoints while sharing one environment
and one official source repository.

Safe behaviour
--------------
- Reuses and repairs existing files by default.
- Never deletes model files.
- Refuses a CPU-only Torch fallback.
- Uses the exact Windows wheel-compatible stack selected for FrameVision:
  Python 3.13, PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130.
- Installs Wildminder's dated FlashAttention 2.8.3 stable-Torch-ABI wheel.
- Never compiles FlashAttention and never silently falls back to SDPA.
- Recreates only the dedicated .mage_edit environment when explicitly requested
  or when its Python version is incompatible; all model folders are preserved.
- Keeps Hugging Face, Torch, pip, and temporary caches inside FrameVision.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENV_RELATIVE = Path("environments") / ".mage_edit"
MODEL_ROOT_RELATIVE = Path("models") / "mage_edit"
REPO_RELATIVE = MODEL_ROOT_RELATIVE / "repo" / "Mage"
TEMP_RELATIVE = Path("temp") / "mage_edit"
STATE_RELATIVE = MODEL_ROOT_RELATIVE / "framevision_mage_install.json"

OFFICIAL_REPO_URL = "https://github.com/microsoft/Mage.git"
OFFICIAL_REPO_ZIP = "https://github.com/microsoft/Mage/archive/refs/heads/main.zip"

MODEL_SPECS: dict[str, dict[str, object]] = {
    "turbo": {
        "label": "Mage-Flow Turbo",
        "repo_id": "microsoft/Mage-Flow-Turbo",
        "folder": "Mage-Flow-Turbo",
        "task": "text-to-image",
        "steps": 4,
        "cfg": 1.0,
    },
    "edit-turbo": {
        "label": "Mage-Flow Edit Turbo",
        "repo_id": "microsoft/Mage-Flow-Edit-Turbo",
        "folder": "Mage-Flow-Edit-Turbo",
        "task": "image-edit",
        "steps": 4,
        "cfg": 1.0,
    },
    "edit": {
        "label": "Mage-Flow Edit",
        "repo_id": "microsoft/Mage-Flow-Edit",
        "folder": "Mage-Flow-Edit",
        "task": "image-edit",
        "steps": 30,
        "cfg": 5.0,
    },
}
MODEL_ORDER: tuple[str, ...] = ("turbo", "edit-turbo", "edit")
DEFAULT_MODEL_KEYS: tuple[str, ...] = ("edit-turbo",)
MODEL_ALIASES: dict[str, str] = {
    "turbo": "turbo",
    "normal-turbo": "turbo",
    "mage-flow-turbo": "turbo",
    "microsoft/mage-flow-turbo": "turbo",
    "edit-turbo": "edit-turbo",
    "mage-flow-edit-turbo": "edit-turbo",
    "microsoft/mage-flow-edit-turbo": "edit-turbo",
    "edit": "edit",
    "normal-edit": "edit",
    "mage-flow-edit": "edit",
    "microsoft/mage-flow-edit": "edit",
}

PYTHON_VERSION = "3.13"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_PACKAGES: Sequence[str] = (
    "torch==2.10.0+cu130",
    "torchvision==0.25.0+cu130",
)
FLASH_ATTN_WHEEL_URL = (
    "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/"
    "flash_attn-2.8.3%2Bd20260121.cu130torch2.10.0cxx11abiTRUE-"
    "cp313-cp313-win_amd64.whl"
    "#sha256=fa072159b8a8d04aa44c1df224ca1b332dbf886484b3d6ecf9d74bc02ddd9b5c"
)
EXPECTED_PYTHON = (3, 13)
EXPECTED_TORCH_PREFIX = "2.10.0+cu130"
EXPECTED_TORCHVISION_PREFIX = "0.25.0+cu130"
EXPECTED_CUDA_RUNTIME = "13.0"
EXPECTED_FLASH_ATTN_PREFIX = "2.8.3"
INSTALLER_REVISION = "4.0-three-model-selection"

MIN_MODEL_WEIGHT_BYTES = 1_000_000_000
MIN_FREE_BYTES = 25 * 1024**3
WARN_FREE_BYTES = 40 * 1024**3


def root_from_script() -> Path:
    # <root>/presets/extra_env/Mage_edit.py
    return Path(__file__).resolve().parents[2]


def status(kind: str, message: str) -> None:
    print(f"[{kind}] {message}", flush=True)


def quote_cmd(cmd: Sequence[object]) -> str:
    rendered: list[str] = []
    for part in cmd:
        text = str(part)
        rendered.append(f'"{text}"' if any(ch in text for ch in " \t&()") else text)
    return " ".join(rendered)


def portable_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    temp_dir = root / TEMP_RELATIVE
    cache_dir = temp_dir / "cache"
    hf_dir = cache_dir / "huggingface"
    for path in (
        temp_dir,
        cache_dir,
        hf_dir,
        hf_dir / "hub",
        cache_dir / "torch",
        cache_dir / "pip",
    ):
        path.mkdir(parents=True, exist_ok=True)

    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["HF_HOME"] = str(hf_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_dir / "hub")
    env["TRANSFORMERS_CACHE"] = str(hf_dir / "transformers")
    env["TORCH_HOME"] = str(cache_dir / "torch")
    env["PIP_CACHE_DIR"] = str(cache_dir / "pip")
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run(
    cmd: Sequence[object],
    *,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    check: bool = False,
) -> int:
    print("\n>>> " + quote_cmd(cmd), flush=True)
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {quote_cmd(cmd)}"
        )
    return int(completed.returncode)


def run_capture(
    cmd: Sequence[object],
    *,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
) -> tuple[int, str, str]:
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
    )
    return (
        int(completed.returncode),
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def env_python(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / "python.exe"
    return env_path / "bin" / "python"


def find_existing_env_python(env_path: Path) -> Optional[Path]:
    candidates = [
        env_python(env_path),
        env_path / "Scripts" / "python.exe",
        env_path / "python.exe",
        env_path / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_conda(root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    if os.environ.get("CONDA_EXE"):
        candidates.append(Path(os.environ["CONDA_EXE"]))

    if os.name == "nt":
        candidates.extend(
            [
                root / "_miniconda" / "Scripts" / "conda.exe",
                root / "_miniconda" / "condabin" / "conda.bat",
                root / "_miniconda3" / "Scripts" / "conda.exe",
                root / "_miniconda3" / "condabin" / "conda.bat",
                root / "miniconda3" / "Scripts" / "conda.exe",
                root / "miniconda3" / "condabin" / "conda.bat",
                root / "installer_files" / "conda" / "Scripts" / "conda.exe",
                root / "installer_files" / "conda" / "condabin" / "conda.bat",
            ]
        )
    else:
        candidates.extend(
            [
                root / "_miniconda" / "bin" / "conda",
                root / "_miniconda3" / "bin" / "conda",
                root / "miniconda3" / "bin" / "conda",
                root / "installer_files" / "conda" / "bin" / "conda",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            status("FOUND", f"Conda: {candidate}")
            return candidate

    found = shutil.which("conda")
    if found:
        status("FOUND", f"Conda on PATH: {found}")
        return Path(found)
    return None


def run_conda(
    conda: Path,
    args: Sequence[object],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool = False,
) -> int:
    if os.name == "nt" and conda.suffix.lower() in {".bat", ".cmd"}:
        command = subprocess.list2cmdline([str(conda), *[str(x) for x in args]])
        return run(
            ["cmd.exe", "/d", "/s", "/c", f'call {command}'],
            cwd=cwd,
            env=env,
            check=check,
        )
    return run([conda, *args], cwd=cwd, env=env, check=check)


def assert_nvidia_present() -> None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError(
            "nvidia-smi was not found. Refusing a CPU-only Torch installation. "
            "Install or repair the NVIDIA driver first."
        )

    rc, out, err = run_capture(
        [
            executable,
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        cwd=Path.cwd(),
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(
            "nvidia-smi exists but the NVIDIA driver version could not be read. "
            f"Details: {err or out}"
        )

    driver_text = out.splitlines()[0].strip()

    # Accept normal NVIDIA Windows versions such as 581.80, 580.97,
    # and optional extra components such as 581.80.01.
    match = re.search(r"(?<!\d)(\d{3,4})(?:\.\d+)*(?!\d)", driver_text)
    if not match:
        raise RuntimeError(
            f"Unrecognized NVIDIA driver version returned by nvidia-smi: {driver_text!r}"
        )

    driver_major = int(match.group(1))
    if driver_major < 580:
        raise RuntimeError(
            f"NVIDIA driver {driver_text} is too old for the required CUDA 13.0 "
            "runtime. Install an NVIDIA 580-series or newer driver first. "
            "The Mage environment and downloaded models were not deleted."
        )

    status("OK", f"NVIDIA driver {driver_text} supports CUDA 13.x")



def environment_python_version(python: Path, root: Path) -> tuple[int, int] | None:
    rc, out, _ = run_capture(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        cwd=root,
        env=portable_env(root),
    )
    if rc != 0:
        return None
    try:
        major, minor = out.strip().split(".", 1)
        return int(major), int(minor)
    except Exception:
        return None


def create_or_repair_env(
    root: Path,
    env_path: Path,
    *,
    recreate: bool = False,
) -> Path:
    conda = find_conda(root)
    if not conda:
        raise RuntimeError(
            "Conda was not found. Expected FrameVision's root-local Miniconda, "
            "CONDA_EXE, or conda on PATH."
        )

    existing = find_existing_env_python(env_path)
    if existing and not recreate:
        version = environment_python_version(existing, root)
        if version == EXPECTED_PYTHON:
            status("FOUND", f"Mage environment Python {version[0]}.{version[1]}: {existing}")
            return existing

        found = "unknown" if version is None else f"{version[0]}.{version[1]}"
        status(
            "WARN",
            f"Existing .mage_edit uses Python {found}; the FlashAttention wheel requires "
            f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.",
        )
        status(
            "WARN",
            "Recreating only environments/.mage_edit. Models and repository are untouched.",
        )
        recreate = True
    elif env_path.exists() and not existing:
        status(
            "WARN",
            "The dedicated .mage_edit environment is incomplete and has no usable Python. "
            "It will be recreated; models and repository are untouched.",
        )
        recreate = True

    if env_path.exists() and recreate:
        shutil.rmtree(env_path)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    status("MISSING", f"Creating Python {PYTHON_VERSION} conda-prefix environment: {env_path}")
    run_conda(
        conda,
        [
            "create",
            "--yes",
            "--prefix",
            env_path,
            f"python={PYTHON_VERSION}",
            "pip",
        ],
        cwd=root,
        env=portable_env(root),
        check=True,
    )

    python = find_existing_env_python(env_path)
    if not python:
        raise RuntimeError(
            f"Conda reported success, but Python was not found under {env_path}"
        )

    version = environment_python_version(python, root)
    if version != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"but conda created {version!r}."
        )
    return python


def pip_install(
    python: Path,
    packages: Iterable[object],
    root: Path,
    *,
    label: str,
    extra_args: Optional[Sequence[object]] = None,
    check: bool = True,
) -> int:
    command: list[object] = [
        python,
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
    ]
    if extra_args:
        command.extend(extra_args)
    command.extend(packages)
    status("DOWNLOADING", label)
    return run(
        command,
        cwd=root,
        env=portable_env(root),
        check=check,
    )


def repo_is_valid(repo_path: Path) -> bool:
    package_root = repo_path / "mage_flow"
    return (
        (package_root / "pyproject.toml").exists()
        and (package_root / "pipeline.py").exists()
        and (package_root / "models" / "mage_flow.py").exists()
    )


def download_repo_zip(root: Path, repo_path: Path) -> None:
    temp_dir = root / TEMP_RELATIVE / "repo_download"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    archive = temp_dir / "Mage-main.zip"

    status("DOWNLOADING", f"Microsoft Mage source archive: {OFFICIAL_REPO_ZIP}")
    urllib.request.urlretrieve(OFFICIAL_REPO_ZIP, archive)

    with zipfile.ZipFile(archive) as handle:
        handle.extractall(temp_dir)

    extracted = temp_dir / "Mage-main"
    if not repo_is_valid(extracted):
        raise RuntimeError("Downloaded Mage archive does not contain the expected source tree")

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted), str(repo_path))
    shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_repo(
    root: Path,
    repo_path: Path,
    *,
    skip_downloads: bool,
    update_repo: bool,
) -> Path:
    if repo_is_valid(repo_path):
        status("FOUND", f"Microsoft Mage repository: {repo_path}")
        if update_repo:
            git = shutil.which("git")
            if git and (repo_path / ".git").exists():
                status("DOWNLOADING", "Updating existing Mage repository with git pull --ff-only")
                rc = run(
                    [git, "-C", repo_path, "pull", "--ff-only"],
                    cwd=root,
                    env=portable_env(root),
                )
                if rc != 0:
                    status("WARN", "Repository update failed; keeping the existing valid checkout")
            else:
                status("WARN", "Existing repository is not a git checkout; update skipped")
        else:
            status("SKIPPED", "Repository already exists")
        return repo_path

    status("MISSING", f"Microsoft Mage repository: {repo_path}")
    if skip_downloads:
        return repo_path

    if repo_path.exists() and any(repo_path.iterdir()):
        raise RuntimeError(
            "The Mage repository path exists but is not a valid checkout. "
            f"It was not deleted: {repo_path}"
        )

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git:
        status("DOWNLOADING", "Cloning the official Microsoft Mage repository")
        rc = run(
            [git, "clone", "--depth", "1", OFFICIAL_REPO_URL, repo_path],
            cwd=root,
            env=portable_env(root),
        )
        if rc == 0 and repo_is_valid(repo_path):
            return repo_path
        status("WARN", "git clone failed; trying the official source ZIP instead")
        if repo_path.exists() and not any(repo_path.iterdir()):
            repo_path.rmdir()

    download_repo_zip(root, repo_path)
    if not repo_is_valid(repo_path):
        raise RuntimeError(f"Repository verification failed: {repo_path}")
    return repo_path


def check_free_space(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(model_root)
    free_gib = usage.free / 1024**3
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB is free on the model drive. "
            "At least 25 GiB is required for the checkpoint and environment."
        )
    if usage.free < WARN_FREE_BYTES:
        status(
            "WARN",
            f"Free disk space is {free_gib:.1f} GiB. The full model is about 17.5 GB.",
        )
    else:
        status("OK", f"Free disk space: {free_gib:.1f} GiB")


def write_filtered_requirements(root: Path, repo_path: Path) -> Path:
    source = repo_path / "mage_flow" / "requirements.txt"
    target = root / TEMP_RELATIVE / "requirements_without_torch.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    if source.exists():
        kept: list[str] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            package_name = re.split(r"[<>=!~\s\[]", stripped, maxsplit=1)[0].lower()
            if package_name in {"torch", "torchvision", "torchaudio"}:
                kept.append(f"# FrameVision installer already pinned CUDA build: {line}")
            else:
                kept.append(line)
        target.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return target

    # Current official fallback pins, used only if the checkout unexpectedly lacks
    # requirements.txt but otherwise contains a valid package.
    fallback = """\
numpy==2.4.3
diffusers==0.38.0
transformers==5.5.0
accelerate==1.13.0
safetensors==0.8.0
huggingface_hub>=0.20
einops==0.8.2
pydantic==2.12.5
pillow==12.3.0
loguru==0.7.3
typing_extensions==4.15.0
gradio>=6.0
"""
    target.write_text(fallback, encoding="utf-8")
    status("WARN", "Official requirements.txt was missing; using embedded current pins")
    return target



def torch_cuda_probe(python: Path, root: Path, *, require_version: bool = True) -> bool:
    code = r"""
import sys
import torch
import torchvision

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

expected = {
    "python": (3, 13),
    "torch": "2.10.0+cu130",
    "torchvision": "0.25.0+cu130",
    "cuda": "13.0",
}
ok = torch.cuda.is_available()
if __import__("os").environ.get("FRAMEVISION_REQUIRE_EXACT_MAGE_STACK") == "1":
    ok = ok and sys.version_info[:2] == expected["python"]
    ok = ok and torch.__version__.startswith(expected["torch"])
    ok = ok and torchvision.__version__.startswith(expected["torchvision"])
    ok = ok and str(torch.version.cuda) == expected["cuda"]
raise SystemExit(0 if ok else 77)
"""
    env = portable_env(root)
    if require_version:
        env["FRAMEVISION_REQUIRE_EXACT_MAGE_STACK"] = "1"
    rc = run([python, "-c", code], cwd=root, env=env)
    return rc == 0



def patch_mage_dependency_metadata(repo_path: Path) -> None:
    """Patch only Mage's local package metadata to describe the selected wheel stack."""
    pyproject = repo_path / "mage_flow" / "pyproject.toml"
    if not pyproject.exists():
        raise RuntimeError(f"Mage pyproject.toml is missing: {pyproject}")

    original = pyproject.read_text(encoding="utf-8")
    patched = original
    patched = patched.replace(
        '"torch>=2.13.0"',
        '"torch>=2.10.0,<2.11"',
    )
    patched = patched.replace(
        '"torchvision>=0.28.0"',
        '"torchvision>=0.25.0,<0.26"',
    )

    if patched == original:
        already_ok = (
            '"torch>=2.10.0,<2.11"' in original
            and '"torchvision>=0.25.0,<0.26"' in original
        )
        if already_ok:
            status("FOUND", "Mage package metadata already matches the FrameVision wheel stack")
            return
        raise RuntimeError(
            "Microsoft changed Mage's dependency declarations. The compatibility "
            "patch did not match, so installation stopped instead of guessing."
        )

    backup = pyproject.with_suffix(".toml.framevision_original")
    if not backup.exists():
        shutil.copy2(pyproject, backup)
    pyproject.write_text(patched, encoding="utf-8")
    status(
        "OK",
        "Patched local Mage metadata for Torch 2.10 / torchvision 0.25 compatibility",
    )


def flash_attention_probe(python: Path, root: Path) -> tuple[bool, str]:
    code = r"""
import torch
import flash_attn
version = getattr(flash_attn, "__version__", "unknown")
print("flash_attn", version)
print("torch", torch.__version__)
print("CUDA", torch.version.cuda)
ok = (
    str(version).startswith("2.8.3")
    and torch.__version__.startswith("2.10.0+cu130")
    and str(torch.version.cuda) == "13.0"
    and torch.cuda.is_available()
)
raise SystemExit(0 if ok else 78)
"""
    rc, out, err = run_capture(
        [python, "-c", code],
        cwd=root,
        env=portable_env(root),
    )
    detail = out if out else err
    return rc == 0, detail.strip()


def install_flash_attention_wheel(python: Path, root: Path) -> None:
    ok, detail = flash_attention_probe(python, root)
    if ok:
        status("FOUND", f"Matching prebuilt FlashAttention wheel: {detail}")
        return

    if os.name != "nt":
        raise RuntimeError(
            "This installer is pinned to a Windows cp313 prebuilt FlashAttention wheel."
        )

    status("MISSING", f"Matching FlashAttention wheel is not active: {detail or 'not installed'}")
    status(
        "DOWNLOADING",
        "Installing prebuilt FlashAttention 2.8.3 wheel "
        "(Python 3.13 / Torch 2.10 stable / CUDA 13.0); no compilation",
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--no-deps",
            "--force-reinstall",
            FLASH_ATTN_WHEEL_URL,
        ],
        cwd=root,
        env=portable_env(root),
        check=True,
    )

    ok, detail = flash_attention_probe(python, root)
    if not ok:
        raise RuntimeError(
            "The exact FlashAttention wheel installed but failed its import/ABI check. "
            f"Details: {detail}"
        )
    status("OK", f"FlashAttention wheel verified: {detail}")


def install_dependencies(python: Path, root: Path, repo_path: Path) -> None:
    pip_install(
        python,
        ["pip", "setuptools", "wheel", "packaging"],
        root,
        label="pip packaging tools",
        extra_args=["--upgrade"],
    )

    pip_install(
        python,
        TORCH_PACKAGES,
        root,
        label=(
            "Exact CUDA stack: Torch 2.10.0 / torchvision 0.25.0 "
            "(CUDA 13.0, Python 3.13)"
        ),
        extra_args=["--index-url", PYTORCH_INDEX, "--upgrade"],
    )
    if not torch_cuda_probe(python, root):
        raise RuntimeError(
            "The required Python 3.13 / Torch 2.10.0+cu130 / torchvision "
            "0.25.0+cu130 stack is not active or CUDA is unavailable."
        )

    requirements = write_filtered_requirements(root, repo_path)
    pip_install(
        python,
        ["-r", requirements],
        root,
        label=f"Official Mage inference dependencies ({requirements.name})",
        extra_args=["--upgrade-strategy", "only-if-needed"],
    )

    pip_install(
        python,
        ["hf_xet"],
        root,
        label="Hugging Face Xet transfer helper",
        extra_args=["--upgrade"],
    )

    patch_mage_dependency_metadata(repo_path)

    # Clear the previous editable metadata so pip check sees the patched requirements.
    run(
        [python, "-m", "pip", "uninstall", "--yes", "mage-flow"],
        cwd=root,
        env=portable_env(root),
    )

    package_root = repo_path / "mage_flow"
    status("OK", "Installing the local official mage-flow package in editable mode")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--no-deps",
            "-e",
            package_root,
        ],
        cwd=root,
        env=portable_env(root),
        check=True,
    )

    install_flash_attention_wheel(python, root)

    if not torch_cuda_probe(python, root):
        raise RuntimeError("The exact CUDA Torch stack became unhealthy after installation")

    rc = run(
        [python, "-m", "pip", "check"],
        cwd=root,
        env=portable_env(root),
    )
    if rc != 0:
        raise RuntimeError("pip check found an incompatible Mage dependency set")


def verify_import(
    python: Path,
    root: Path,
    module_name: str,
) -> tuple[bool, str]:
    code = f"""
import importlib
module = importlib.import_module({module_name!r})
print(getattr(module, "__version__", getattr(module, "version", "import OK")))
"""
    rc, out, err = run_capture(
        [python, "-c", code],
        cwd=root,
        env=portable_env(root),
    )
    if rc == 0:
        return True, out or "import OK"
    detail = err or out or "import failed"
    return False, detail.splitlines()[-1]



def configure_attention_backend(repo_path: Path, backend: str = "flash2") -> None:
    if backend != "flash2":
        raise ValueError("This installer requires FlashAttention and does not use SDPA")

    source = repo_path / "mage_flow" / "models" / "mage_flow.py"
    if not source.exists():
        raise RuntimeError(f"Cannot configure attention backend; missing: {source}")

    original = source.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(attn_type\s*:\s*str\s*=\s*Field\(\s*default\s*=\s*)'
        r'["\'](?:flash2|sdpa)["\']',
        re.MULTILINE,
    )
    patched, count = pattern.subn(r'\1"flash2"', original, count=1)
    if count != 1:
        raise RuntimeError(
            "Microsoft changed the ModelConfig attention field. Installation stopped "
            "instead of silently selecting another backend."
        )

    if patched != original:
        backup = source.with_suffix(source.suffix + ".framevision_original")
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_text(patched, encoding="utf-8")
        status("OK", "Configured Mage to use FlashAttention 2")
    else:
        status("FOUND", "Mage is already configured for FlashAttention 2")


def configured_attention_backend(repo_path: Path) -> str:
    return "flash2"


def choose_attention_backend(
    python: Path,
    root: Path,
    repo_path: Path,
) -> tuple[str, bool]:
    install_flash_attention_wheel(python, root)
    configure_attention_backend(repo_path, "flash2")
    return "flash2", True


def model_path_for(root: Path, model_key: str) -> Path:
    spec = MODEL_SPECS[model_key]
    return root / MODEL_ROOT_RELATIVE / str(spec["folder"])


def normalize_model_token(token: str) -> str:
    normalized = token.strip().lower().replace("_", "-")
    if normalized in {"all", "*"}:
        return "all"
    try:
        return MODEL_ALIASES[normalized]
    except KeyError as exc:
        valid = ", ".join(MODEL_ORDER)
        raise ValueError(f"Unknown Mage model selection {token!r}. Valid choices: {valid}, all") from exc


def parse_model_values(values: Optional[Sequence[str]]) -> list[str]:
    if not values:
        return []
    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in re.split(r"[,;+]", str(value)) if part.strip())

    selected: list[str] = []
    for token in tokens:
        key = normalize_model_token(token)
        if key == "all":
            return list(MODEL_ORDER)
        if key not in selected:
            selected.append(key)
    return [key for key in MODEL_ORDER if key in selected]


def discover_installed_model_keys(root: Path) -> list[str]:
    return [key for key in MODEL_ORDER if model_is_complete(model_path_for(root, key))]


def interactive_model_selection() -> list[str]:
    print("", flush=True)
    print("Select one or more Mage checkpoints to install:", flush=True)
    print("  1. Mage-Flow Turbo       — text to image, 4 steps, CFG 1", flush=True)
    print("  2. Mage-Flow Edit Turbo  — image editing, 4 steps, CFG 1", flush=True)
    print("  3. Mage-Flow Edit        — higher-quality image editing, 30 steps, CFG 5", flush=True)
    print("  4. All three", flush=True)
    print("", flush=True)
    raw = input("Selection [default: 2]: ").strip()
    if not raw:
        return list(DEFAULT_MODEL_KEYS)

    numeric = {"1": "turbo", "2": "edit-turbo", "3": "edit"}
    selected: list[str] = []
    for token in re.split(r"[,;+\s]+", raw):
        if not token:
            continue
        if token == "4" or token.lower() == "all":
            return list(MODEL_ORDER)
        key = numeric[token] if token in numeric else normalize_model_token(token)
        if key not in selected:
            selected.append(key)
    return [key for key in MODEL_ORDER if key in selected]


def resolve_model_selection(
    values: Optional[Sequence[str]],
    *,
    root: Path,
    verify_only: bool,
) -> list[str]:
    selected = parse_model_values(values)
    if selected:
        return selected
    if verify_only:
        installed = discover_installed_model_keys(root)
        return installed or list(DEFAULT_MODEL_KEYS)
    if sys.stdin is not None and sys.stdin.isatty():
        return interactive_model_selection()
    # Preserve the original installer behaviour for non-interactive callers.
    return list(DEFAULT_MODEL_KEYS)


def model_is_complete(model_path: Path) -> bool:
    required = [
        model_path / "model_index.json",
        model_path / "transformer" / "config.json",
        model_path / "scheduler" / "scheduler_config.json",
        model_path / "text_encoder" / "config.json",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False

    transformer_weights = list((model_path / "transformer").glob("*.safetensors"))
    if not transformer_weights:
        return False
    if sum(path.stat().st_size for path in transformer_weights) < MIN_MODEL_WEIGHT_BYTES:
        return False

    vae_weights = list((model_path / "vae").glob("*.safetensors"))
    text_weights = list((model_path / "text_encoder").glob("*.safetensors"))
    text_index = model_path / "text_encoder" / "model.safetensors.index.json"
    return bool(vae_weights) and (bool(text_weights) or text_index.exists())


def ensure_model(
    python: Path,
    root: Path,
    model_key: str,
    *,
    skip_downloads: bool,
) -> Path:
    spec = MODEL_SPECS[model_key]
    label = str(spec["label"])
    repo_id = str(spec["repo_id"])
    model_path = model_path_for(root, model_key)

    if model_is_complete(model_path):
        status("FOUND", f"{label} checkpoint: {model_path}")
        status("SKIPPED", "Complete checkpoint already exists")
        return model_path

    status("MISSING", f"{label} checkpoint: {model_path}")
    if skip_downloads:
        return model_path

    check_free_space(root / MODEL_ROOT_RELATIVE)
    model_path.mkdir(parents=True, exist_ok=True)
    code = f"""
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id={repo_id!r},
    local_dir={str(model_path)!r},
    max_workers=4,
)
print(path)
"""
    status("DOWNLOADING", f"Complete {repo_id} repository into {model_path}")
    run(
        [python, "-c", code],
        cwd=root,
        env=portable_env(root),
        check=True,
    )
    if not model_is_complete(model_path):
        raise RuntimeError(
            "Hugging Face download finished, but required Mage model files are "
            f"missing or incomplete under {model_path}"
        )
    return model_path


def runtime_bootstrap_code(backend: str) -> str:
    hf_backend = "flash_attention_2" if backend == "flash2" else "sdpa"
    return f"""\
import os
os.environ.setdefault("VF_HF_ATTN_IMPL", {hf_backend!r})

def configure_mage_backend():
    from mage_flow.models import mage_flow as _mage_model
    from mage_flow.models.modules._attn_backend import set_attn_backend
    field = _mage_model.ModelConfig.model_fields.get("attn_type")
    if field is not None:
        field.default = {backend!r}
    set_attn_backend({backend!r})
"""


def installed_model_registry(root: Path) -> dict[str, Path]:
    return {
        key: model_path_for(root, key)
        for key in MODEL_ORDER
        if model_is_complete(model_path_for(root, key))
    }


def preferred_default_model_key(registry: dict[str, Path]) -> str:
    for key in ("edit-turbo", "edit", "turbo"):
        if key in registry:
            return key
    raise RuntimeError("No complete Mage checkpoint is installed")


def write_runtime_helpers(
    root: Path,
    model_root: Path,
    registry: dict[str, Path],
    backend: str,
) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    if not registry:
        status("SKIPPED", "Runtime helpers were not written because no complete model is installed")
        return

    bootstrap = runtime_bootstrap_code(backend)
    default_key = preferred_default_model_key(registry)
    paths_payload = {key: str(path) for key, path in registry.items()}
    tasks_payload = {key: str(MODEL_SPECS[key]["task"]) for key in registry}
    defaults_payload = {
        key: {
            "steps": int(MODEL_SPECS[key]["steps"]),
            "cfg": float(MODEL_SPECS[key]["cfg"]),
        }
        for key in registry
    }

    runtime_py = model_root / "framevision_mage_runtime.py"
    runtime_py.write_text(
        bootstrap
        + f"""
from pathlib import Path

MODEL_PATHS = {{key: Path(value) for key, value in {paths_payload!r}.items()}}
MODEL_TASKS = {tasks_payload!r}
MODEL_DEFAULTS = {defaults_payload!r}
DEFAULT_MODEL_KEY = {default_key!r}
MODEL_PATH = MODEL_PATHS[DEFAULT_MODEL_KEY]  # backwards compatibility

def load_pipeline(model=DEFAULT_MODEL_KEY, device="cuda"):
    if model not in MODEL_PATHS:
        raise KeyError(f"Mage model {{model!r}} is not installed. Available: {{', '.join(MODEL_PATHS)}}")
    configure_mage_backend()
    from mage_flow import MageFlowPipeline
    return MageFlowPipeline.from_pretrained(str(MODEL_PATHS[model]), device=device)
""",
        encoding="utf-8",
    )

    app_py = model_root / "framevision_mage_app.py"
    app_py.write_text(
        bootstrap
        + """
def main():
    configure_mage_backend()
    from mage_flow.app import main as mage_main
    mage_main()

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    edit_registry = {key: str(path) for key, path in registry.items() if MODEL_SPECS[key]["task"] == "image-edit"}
    if edit_registry:
        edit_default = "edit-turbo" if "edit-turbo" in edit_registry else next(iter(edit_registry))
        edit_defaults = {
            key: {
                "steps": int(MODEL_SPECS[key]["steps"]),
                "cfg": float(MODEL_SPECS[key]["cfg"]),
            }
            for key in edit_registry
        }
        cli_py = model_root / "mage_edit_cli.py"
        cli_py.write_text(
            bootstrap
            + f"""
import argparse
from pathlib import Path
from PIL import Image

MODEL_PATHS = {{key: Path(value) for key, value in {edit_registry!r}.items()}}
MODEL_DEFAULTS = {edit_defaults!r}
DEFAULT_MODEL = {edit_default!r}

def main():
    parser = argparse.ArgumentParser(description="Local Mage image-edit test")
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), default=DEFAULT_MODEL)
    parser.add_argument("--input", required=True, help="Source image")
    parser.add_argument("--prompt", required=True, help="Edit instruction")
    parser.add_argument("--output", required=True, help="Output image")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    args = parser.parse_args()

    configure_mage_backend()
    from mage_flow import MageFlowPipeline

    defaults = MODEL_DEFAULTS[args.model]
    steps = defaults["steps"] if args.steps is None else args.steps
    cfg = defaults["cfg"] if args.cfg is None else args.cfg
    source = Image.open(args.input).convert("RGB")
    pipe = MageFlowPipeline.from_pretrained(str(MODEL_PATHS[args.model]), device="cuda")
    result = pipe.edit(
        [args.prompt],
        [[source]],
        neg_prompts=[" "],
        seeds=[args.seed],
        steps=steps,
        cfg=cfg,
        max_size=args.max_size,
    )[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(output)

if __name__ == "__main__":
    main()
""",
            encoding="utf-8",
        )
        status("OK", f"Local edit CLI: {cli_py}")

    if "turbo" in registry:
        generate_py = model_root / "mage_generate_cli.py"
        generate_py.write_text(
            bootstrap
            + f"""
import argparse
from pathlib import Path

MODEL_PATH = Path({str(registry['turbo'])!r})

def main():
    parser = argparse.ArgumentParser(description="Local Mage-Flow Turbo text-to-image test")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_mage_backend()
    from mage_flow import MageFlowPipeline
    pipe = MageFlowPipeline.from_pretrained(str(MODEL_PATH), device="cuda")
    result = pipe.generate(
        [args.prompt],
        heights=[args.height],
        widths=[args.width],
        seeds=[args.seed],
        steps=4,
        cfg=1.0,
    )[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(output)

if __name__ == "__main__":
    main()
""",
            encoding="utf-8",
        )
        status("OK", f"Local text-to-image CLI: {generate_py}")

    if os.name == "nt":
        hf_backend = "flash_attention_2" if backend == "flash2" else "sdpa"
        app_bat = model_root / "run_mage_app.bat"
        app_bat.write_text(
            f"""@echo off
setlocal
set "ROOT={root}"
set "PY={root / ENV_RELATIVE / 'python.exe'}"
set "MAGEFLOW_HF_DIR={model_root}"
set "VF_HF_ATTN_IMPL={hf_backend}"
set "PYTHONNOUSERSITE=1"
"%PY%" "{app_py}" --host 127.0.0.1 --port 7860
endlocal
""",
            encoding="utf-8",
        )

    status("OK", f"Runtime helper: {runtime_py}")


def read_state(root: Path) -> dict:
    path = root / STATE_RELATIVE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(
    root: Path,
    python: Path,
    repo_path: Path,
    registry: dict[str, Path],
    selected_keys: Sequence[str],
    backend: str,
    flash_available: bool,
) -> None:
    state_path = root / STATE_RELATIVE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    default_key = preferred_default_model_key(registry)
    model_payload = {
        key: {
            "label": MODEL_SPECS[key]["label"],
            "task": MODEL_SPECS[key]["task"],
            "repo_id": MODEL_SPECS[key]["repo_id"],
            "path": str(path),
            "steps": MODEL_SPECS[key]["steps"],
            "cfg": MODEL_SPECS[key]["cfg"],
        }
        for key, path in registry.items()
    }
    payload = {
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(python.parent),
        "python": str(python),
        "repository": str(repo_path),
        "selected_models": list(selected_keys),
        "models": model_payload,
        "default_model_key": default_key,
        # Backwards-compatible single-model fields used by older helpers.
        "model": str(registry[default_key]),
        "model_repo_id": MODEL_SPECS[default_key]["repo_id"],
        "attention_backend": backend,
        "flash_attention_available": flash_available,
        "torch_index": PYTORCH_INDEX,
        "python_version": PYTHON_VERSION,
        "torch_packages": list(TORCH_PACKAGES),
        "flash_attention_wheel": FLASH_ATTN_WHEEL_URL.split("#", 1)[0],
        "installer_revision": INSTALLER_REVISION,
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status("OK", f"Install state: {state_path}")


def verify_runtime(
    python: Path,
    root: Path,
    repo_path: Path,
    selected_keys: Sequence[str],
    backend: str,
) -> int:
    failed: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status("OK" if condition else "FAILED", f"{name}{': ' + detail if detail else ''}")
        if not condition:
            failed.append(name)

    check("environment Python exists", python.exists() and python.stat().st_size > 0, str(python))
    check("official Mage repository exists", repo_is_valid(repo_path), str(repo_path))
    for key in selected_keys:
        path = model_path_for(root, key)
        check(f"complete {MODEL_SPECS[key]['label']} checkpoint exists", model_is_complete(path), str(path))
    check("configured attention backend", backend == "flash2", backend)

    stack_ok = torch_cuda_probe(python, root)
    check(
        "exact Python/Torch/CUDA stack",
        stack_ok,
        "Python 3.13, Torch 2.10.0+cu130, torchvision 0.25.0+cu130",
    )

    flash_ok, flash_detail = flash_attention_probe(python, root)
    check("prebuilt FlashAttention 2.8.3 wheel", flash_ok, flash_detail)

    env = portable_env(root)
    env["VF_HF_ATTN_IMPL"] = "flash_attention_2"
    code = r"""
import torch
import diffusers
import transformers
import mage_flow
from mage_flow.models import mage_flow as mm
from mage_flow.models.modules._attn_backend import set_attn_backend
field = mm.ModelConfig.model_fields.get("attn_type")
if field is not None:
    field.default = "flash2"
set_attn_backend("flash2")
print("mage_flow", getattr(mage_flow, "__version__", "unknown"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
raise SystemExit(0 if torch.cuda.is_available() else 77)
"""
    rc = run([python, "-c", code], cwd=root, env=env)
    check("Mage imports with FlashAttention backend", rc == 0)

    rc = run([python, "-m", "pip", "check"], cwd=root, env=env)
    check("pip dependency check", rc == 0)
    return 0 if not failed else 20


def verify_full_model_load(
    python: Path,
    root: Path,
    model_key: str,
    backend: str,
) -> int:
    model_path = model_path_for(root, model_key)
    status("OK", f"Running optional full CUDA load verification: {MODEL_SPECS[model_key]['label']}")
    env = portable_env(root)
    env["VF_HF_ATTN_IMPL"] = "flash_attention_2" if backend == "flash2" else "sdpa"
    code = f"""
import gc
import torch
from mage_flow.models import mage_flow as mm
from mage_flow.models.modules._attn_backend import set_attn_backend
field = mm.ModelConfig.model_fields.get("attn_type")
if field is not None:
    field.default = {backend!r}
set_attn_backend({backend!r})
from mage_flow import MageFlowPipeline
pipe = MageFlowPipeline.from_pretrained({str(model_path)!r}, device="cuda")
print("FULL_MODEL_LOAD_OK", {model_key!r})
print("allocated_GiB", torch.cuda.memory_allocated() / 1024**3)
print("reserved_GiB", torch.cuda.memory_reserved() / 1024**3)
del pipe
gc.collect()
torch.cuda.empty_cache()
"""
    return run([python, "-c", code], cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install/repair selected Microsoft Mage-Flow models for FrameVision"
    )
    parser.add_argument("--root", default=None, help="FrameVision application root")
    parser.add_argument("--repair", action="store_true", help="Safe create/repair mode")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Models to install, separated by spaces or commas: turbo, edit-turbo, edit, all. "
            "Without this option an interactive selector is shown when possible."
        ),
    )
    parser.add_argument("--list-models", action="store_true", help="List supported model choices and exit")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-load", action="store_true", help="Also load each selected full model on CUDA")
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--skip-model-downloads", action="store_true")
    parser.add_argument("--skip-repo-download", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--update-repo", action="store_true")
    parser.add_argument("--danger-recreate-env", action="store_true")
    parser.add_argument(
        "--confirm-recreate-env",
        default="",
        help="Must be DELETE_ENV_ONLY when recreating the environment",
    )
    args = parser.parse_args()

    if args.list_models:
        for key in MODEL_ORDER:
            spec = MODEL_SPECS[key]
            print(f"{key:10} | {spec['label']} | {spec['task']} | {spec['steps']} steps | CFG {spec['cfg']} | {spec['repo_id']}")
        return 0

    root = Path(args.root).resolve() if args.root else root_from_script()
    env_path = root / ENV_RELATIVE
    repo_path = root / REPO_RELATIVE
    model_root = root / MODEL_ROOT_RELATIVE
    selected_keys = resolve_model_selection(args.models, root=root, verify_only=args.verify_only)
    if not selected_keys:
        raise RuntimeError("No Mage models were selected")

    for directory in (
        root / "environments",
        model_root,
        root / TEMP_RELATIVE,
        root / "presets" / "extra_env",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    status("OK", f"Mage installer revision: {INSTALLER_REVISION}")
    status("OK", "Required stack: Python 3.13 / Torch 2.10.0+cu130 / FlashAttention 2.8.3 stable-ABI wheel")
    status("OK", f"FrameVision root: {root}")
    status("OK", f"Environment target: {env_path}")
    status("OK", f"Repository target: {repo_path}")
    status("OK", "Selected models: " + ", ".join(str(MODEL_SPECS[key]["label"]) for key in selected_keys))
    for key in selected_keys:
        status("OK", f"Model target [{key}]: {model_path_for(root, key)}")

    if args.danger_recreate_env and args.confirm_recreate_env != "DELETE_ENV_ONLY":
        raise RuntimeError(
            "Environment recreation refused. Add --confirm-recreate-env DELETE_ENV_ONLY. "
            "Models are never deleted."
        )

    try:
        assert_nvidia_present()

        if args.verify_only:
            python = find_existing_env_python(env_path) or env_python(env_path)
            backend = "flash2"
            rc = verify_runtime(python, root, repo_path, selected_keys, backend)
            if rc == 0 and args.verify_load:
                for key in selected_keys:
                    rc = verify_full_model_load(python, root, key, backend)
                    if rc != 0:
                        break
            return rc

        repo = ensure_repo(
            root,
            repo_path,
            skip_downloads=args.skip_downloads or args.skip_repo_download,
            update_repo=args.update_repo,
        )
        if not repo_is_valid(repo):
            raise RuntimeError("Mage source repository is missing. Downloads were skipped or failed.")

        python = create_or_repair_env(root, env_path, recreate=args.danger_recreate_env)

        if args.skip_deps:
            status("SKIPPED", "Dependency installation skipped")
        else:
            install_dependencies(python, root, repo)

        if args.skip_deps:
            patch_mage_dependency_metadata(repo)
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-warn-script-location",
                    "--no-deps",
                    "-e",
                    repo / "mage_flow",
                ],
                cwd=root,
                env=portable_env(root),
                check=True,
            )

        backend, flash_available = choose_attention_backend(python, root, repo)

        for key in selected_keys:
            ensure_model(
                python,
                root,
                key,
                skip_downloads=args.skip_downloads or args.skip_model_downloads,
            )

        registry = installed_model_registry(root)
        missing_selected = [key for key in selected_keys if key not in registry]
        if missing_selected:
            labels = ", ".join(str(MODEL_SPECS[key]["label"]) for key in missing_selected)
            raise RuntimeError(
                "Selected model download was skipped or incomplete: " + labels
            )
        write_runtime_helpers(root, model_root, registry, backend)
        write_state(root, python, repo, registry, selected_keys, backend, flash_available)

        rc = verify_runtime(python, root, repo, selected_keys, backend)
        if rc == 0 and args.verify_load:
            for key in selected_keys:
                rc = verify_full_model_load(python, root, key, backend)
                if rc != 0:
                    break

        if rc == 0:
            labels = ", ".join(str(MODEL_SPECS[key]["label"]) for key in selected_keys)
            status("OK", f"Mage installation and verification passed: {labels}")
        else:
            status("FAILED", "Verification found missing or broken components")
        return rc
    except Exception as exc:
        status("FAILED", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
