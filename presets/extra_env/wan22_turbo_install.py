#!/usr/bin/env python3
r"""FrameVision WAN 2.2 Turbo optional installer.

Installs the combined WAN 2.2 TI2V 5B + WAN 2.2 Turbo setup.

This installer is intentionally a full combined installer, not a small Turbo-only
installer with hidden console arguments. The default optional-install button run
installs/repairs normal WAN 2.2 first, then adds Turbo on top.

What this installer does by default:
  1) Creates/reuses APP_ROOT\environments\.wan22_i2v, preferably as a local conda env.
  2) Installs the Python deps used by the working WAN/Turbo tests.
  3) Downloads the complete Wan-AI/Wan2.2-TI2V-5B model snapshot into models\wan22,
     including the large diffusion_pytorch_model*.safetensors shards.
  4) Downloads the official Wan2.2 GitHub repo and merges its scripts into models\wan22.
  5) Downloads the Turbo repo and the Turbo model.pt.
  6) Applies small FrameVision compatibility patches to the Turbo repo.
  7) Creates/repairs wan_models junctions/symlinks inside the Turbo repo.
  8) Extracts presets\extra_env\wan22.zip into APP_ROOT\models at the end, matching
     the original normal WAN 2.2 installer repair step.

Expected location:
  APP_ROOT\presets\extra_env\wan22_turbo_install.py

Run from FrameVision optional installs or manually:
  python presets\extra_env\wan22_turbo_install.py

Notes:
  - Hugging Face model access may require login. For conda envs the CLI is usually:
      APP_ROOT\environments\.wan22_i2v\huggingface-cli.exe login
    For venv fallback it is usually:
      APP_ROOT\environments\.wan22_i2v\Scripts\huggingface-cli.exe login
  - No CPU-only torch fallback is used. If CUDA torch install fails, this script
    stops with an error.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Version pins copied from the working WAN installer style.
# Adjust these constants in one place if the main WAN env is upgraded later.
# ---------------------------------------------------------------------------
PYTHON_VERSION = "3.11"
TORCH_PACKAGES = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
TRITON_PACKAGE = "triton-windows>=3.2,<3.3"
FLASH_ATTN_WHEEL = (
    "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/"
    "flash_attn-2.7.4+cu126torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl?download=true"
)

BASE_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
TURBO_REPO_ID = "quanhaol/Wan2.2-TI2V-5B-Turbo"
TURBO_GITHUB_ZIP = "https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/archive/refs/heads/main.zip"
WAN_GITHUB_ZIP = "https://github.com/Wan-Video/Wan2.2/archive/refs/heads/main.zip"

# Minimal shared WAN files. This avoids the big FP16 diffusion shards.
BASE_ALLOW_PATTERNS = [
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/**",
    "*.json",
    "*.txt",
    "*.model",
]
BASE_IGNORE_PATTERNS = [
    "diffusion_pytorch_model*.safetensors",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "!Wan2.2_VAE.pth",
    "!models_t5_umt5-xxl-enc-bf16.pth",
]
# snapshot_download ignore_patterns does not support "!" unignore rules.  The
# actual download call below uses allow_patterns for the required .pth files and
# small metadata, so the broad ignore list is only used when full-base mode is off
# and after allow_patterns has already constrained the candidate set.

TURBO_ALLOW_PATTERNS = ["model.pt", "*.json", "*.txt", "*.md"]

PIP_PACKAGES = [
    "Pillow==12.2.0",
    "tqdm==4.67.3",
    "imageio==2.37.2",
    "PySide6==6.10.2",
    "huggingface_hub==0.36.2",
    "safetensors==0.7.0",
    "lmdb==2.2.0",
    "pandas==3.0.3",
    "omegaconf==2.3.0",
    "einops==0.8.2",
    "decord==0.6.0",
    "librosa==0.11.0",
    "opencv-python==4.11.0.86",
    "tokenizers==0.22.2",
    "accelerate==1.12.0",
    "imageio[ffmpeg]==2.37.2",
    "easydict==1.13",
    "ftfy==6.3.1",
    "dashscope==1.25.12",
    "imageio-ffmpeg==0.6.0",
    "numpy==1.26.4",
    "regex==2026.1.15",
    "psutil==7.2.2",
    "sageattention==1.0.6",
    "sentencepiece",
    "protobuf",
]
PINNED_NO_DEPS = [
    "transformers==4.57.6",
    "diffusers==0.35.2",
    "peft==0.18.0",
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    log("\n[FATAL] " + msg)
    raise SystemExit(code)


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("[RUN] " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if p.returncode != 0:
        fail(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


def capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def script_app_root() -> Path:
    here = Path(__file__).resolve()
    # Expected: APP_ROOT/presets/extra_env/wan22_turbo_install.py
    try:
        if here.parent.name.lower() == "extra_env" and here.parent.parent.name.lower() == "presets":
            return here.parent.parent.parent
    except Exception:
        pass
    return Path.cwd().resolve()


def env_python(env_dir: Path) -> Path:
    r"""Return the Python executable for either a local conda env or venv.

    Important on Windows:
    - conda envs created with `conda create -p environments/.wan22_i2v` place python.exe
      directly in `environments\.wan22_i2v\python.exe`
    - venvs place python.exe in `environments\.wan22_i2v\Scripts\python.exe`

    The first Turbo installer only checked Scripts/python.exe, so fresh conda
    installs created correctly but failed immediately when pip was launched.
    """
    if os.name == "nt":
        candidates = [
            env_dir / "python.exe",              # conda -p env
            env_dir / "Scripts" / "python.exe", # venv
        ]
        for c in candidates:
            if c.exists():
                return c
        # Fresh conda envs will create this path, so prefer it before creation.
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def env_exe(env_dir: Path, name: str) -> Path:
    """Return executable path for conda/venv layouts."""
    if os.name == "nt":
        suffix = ".exe" if not name.endswith(".exe") else ""
        exe_name = f"{name}{suffix}"
        candidates = [
            env_dir / exe_name,
            env_dir / "Scripts" / exe_name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return env_dir / exe_name
    return env_dir / "bin" / name


def find_conda() -> str | None:
    candidates = ["conda"]
    for env_name in ("CONDA_EXE", "MAMBA_EXE"):
        value = os.environ.get(env_name)
        if value:
            candidates.insert(0, value)
    for c in candidates:
        try:
            out = capture([c, "--version"])
            if out:
                return c
        except Exception:
            pass
    return None


def create_env(app_root: Path, env_dir: Path, mode: str) -> None:
    py = env_python(env_dir)
    if py.exists():
        log(f"[INFO] Reusing existing environment: {env_dir}")
        return

    if mode in ("auto", "conda"):
        conda = find_conda()
        if conda:
            log(f"[INFO] Creating local conda env: {env_dir}")
            run([conda, "create", "-y", "-p", str(env_dir), f"python={PYTHON_VERSION}"])
            created_py = env_python(env_dir)
            if not created_py.exists():
                fail(
                    "Conda environment was created, but Python was not found. Checked: "
                    f"{env_dir / 'python.exe'} and {env_dir / 'Scripts' / 'python.exe'}"
                )
            log(f"[INFO] Environment Python: {created_py}")
            return
        if mode == "conda":
            fail("Conda was requested, but conda was not found. Install Miniconda/Conda or run with --env-mode venv.")

    log(f"[INFO] Creating venv fallback: {env_dir}")
    run([sys.executable, "-m", "venv", str(env_dir)])
    py = env_python(env_dir)
    if not py.exists():
        fail(f"Environment Python was not created: {py}")
    log(f"[INFO] Environment Python: {py}")


def install_env_packages(py: Path, skip_flash: bool = False) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    log("\n[INFO] Installing CUDA PyTorch. No CPU fallback will be used.")
    run([str(py), "-m", "pip", "install", "--upgrade", "--force-reinstall", *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL])

    log("\n[INFO] Verifying CUDA PyTorch...")
    run([str(py), "-c", "import torch; print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 2)"])

    log("\n[INFO] Installing WAN/Turbo Python packages (without torch pins)...")
    run([str(py), "-m", "pip", "install", *PIP_PACKAGES])

    log("\n[INFO] Pinning core Hugging Face/diffusers deps without touching torch...")
    run([str(py), "-m", "pip", "install", "--upgrade", "--no-deps", *PINNED_NO_DEPS])
    run([str(py), "-m", "pip", "install", "huggingface_hub==0.36.2"])

    if skip_flash:
        log("[WARN] Skipping Triton/FlashAttention because --skip-flash was used.")
        verify_runtime_imports(py)
        return

    log("\n[INFO] Installing Triton-Windows and FlashAttention...")
    run([str(py), "-m", "pip", "install", "--upgrade", TRITON_PACKAGE])
    run([str(py), "-m", "pip", "install", "--upgrade", FLASH_ATTN_WHEEL])
    run([str(py), "-c", "import flash_attn; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"])
    verify_runtime_imports(py)




def verify_runtime_imports(py: Path) -> None:
    """Fail during install if the WAN/Turbo runtime stack is incomplete or drifted."""
    code = r"""
from importlib import import_module
from importlib.metadata import version

required_imports = [
    ("torch", "torch"),
    ("PIL", "Pillow"),
    ("regex", "regex"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("accelerate", "accelerate"),
    ("omegaconf", "omegaconf"),
    ("einops", "einops"),
    ("decord", "decord"),
    ("cv2", "opencv-python"),
    ("peft", "peft"),
    ("psutil", "psutil"),
]

expected_versions = {
    "torch": "2.6.0+cu126",
    "torchvision": "0.21.0+cu126",
    "torchaudio": "2.6.0+cu126",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "diffusers": "0.35.2",
    "accelerate": "1.12.0",
    "numpy": "1.26.4",
    "huggingface_hub": "0.36.2",
}

missing = []
for module, package in required_imports:
    try:
        import_module(module)
    except Exception as exc:
        missing.append(f"{package} ({module}): {type(exc).__name__}: {exc}")

version_errors = []
for package, expected in expected_versions.items():
    try:
        got = version(package)
        if got != expected:
            version_errors.append(f"{package}: expected {expected}, got {got}")
    except Exception as exc:
        version_errors.append(f"{package}: version check failed: {type(exc).__name__}: {exc}")

try:
    import torch
    if not torch.cuda.is_available():
        version_errors.append("torch.cuda.is_available() is False")
except Exception as exc:
    version_errors.append(f"torch CUDA check failed: {type(exc).__name__}: {exc}")

if missing or version_errors:
    if missing:
        print("Missing/failed runtime imports:")
        for item in missing:
            print(" - " + item)
    if version_errors:
        print("Runtime version mismatches:")
        for item in version_errors:
            print(" - " + item)
    raise SystemExit(2)

print("Runtime import/version check: OK")
"""
    log("\n[INFO] Verifying WAN/Turbo runtime imports and key versions...")
    run([str(py), "-c", code])



def py_download_script() -> str:
    return r'''
import os
from huggingface_hub import snapshot_download
repo_id = os.environ["FV_REPO_ID"]
local_dir = os.environ["FV_LOCAL_DIR"]
patterns = [p for p in os.environ.get("FV_ALLOW", "").split("||") if p]
token = os.environ.get("HF_TOKEN") or None
print(f"[HF] repo={repo_id}")
print(f"[HF] local_dir={local_dir}")
print(f"[HF] allow_patterns={patterns if patterns else 'ALL'}")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=patterns or None,
    token=token,
)
'''


def snapshot_download(py: Path, repo_id: str, local_dir: Path, allow_patterns: list[str] | None, hf_token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FV_REPO_ID"] = repo_id
    env["FV_LOCAL_DIR"] = str(local_dir)
    env["FV_ALLOW"] = "||".join(allow_patterns or [])
    # Do not force hf_transfer. Some user/global environments set
    # HF_HUB_ENABLE_HF_TRANSFER=1, but the fresh WAN env may not have hf_transfer
    # installed yet, which makes snapshot_download crash before it can fall back.
    # Keep downloads reliable by disabling that optional fast path here.
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Also disable Xet unless the user installed hf_xet; regular HTTP is slower
    # but avoids fresh-install failures.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    run([str(py), "-c", py_download_script()], env=env)


def required_base_files_exist(base_dir: Path, minimal: bool) -> bool:
    needed = [
        base_dir / "Wan2.2_VAE.pth",
        base_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        base_dir / "google" / "umt5-xxl",
    ]
    if minimal:
        needed.append(base_dir / "config.json")
    else:
        needed.extend([
            base_dir / "diffusion_pytorch_model.safetensors.index.json",
        ])
    return all(p.exists() for p in needed)


def download_models(py: Path, model_dir: Path, turbo_model_dir: Path, hf_token: str | None, full_base: bool) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    turbo_model_dir.mkdir(parents=True, exist_ok=True)

    if full_base:
        if required_base_files_exist(model_dir, minimal=False):
            log("\n[INFO] Full WAN 2.2 base files already exist; checking/repairing snapshot if needed.")
        else:
            log("\n[INFO] Downloading complete Wan-AI/Wan2.2-TI2V-5B snapshot.")
            log("       This includes the large diffusion safetensor shards required by normal WAN 2.2.")
            log("       If this fails with 401/RepositoryNotFound, login to Hugging Face and accept model access.")
        snapshot_download(py, BASE_REPO_ID, model_dir, None, hf_token)
    elif required_base_files_exist(model_dir, minimal=True):
        log("\n[WARN] --minimal-base used: normal WAN 2.2 will NOT be fully installed/repaired.")
        log("[INFO] Minimal shared WAN files already exist; skipping base download.")
    else:
        log("\n[WARN] --minimal-base used: downloading only shared WAN files for Turbo.")
        log("       This mode is for debugging only and is not used by the optional-install button.")
        snapshot_download(py, BASE_REPO_ID, model_dir, BASE_ALLOW_PATTERNS, hf_token)

    if (turbo_model_dir / "model.pt").exists():
        log("\n[INFO] Turbo model.pt already exists; skipping Turbo checkpoint download.")
    else:
        log("\n[INFO] Downloading Turbo checkpoint model.pt.")
        snapshot_download(py, TURBO_REPO_ID, turbo_model_dir, TURBO_ALLOW_PATTERNS, hf_token)


def download_url(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[DOWNLOAD] {url}")
    log(f"           -> {out_path}")
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def copytree_merge(src: Path, dst: Path) -> None:
    """Merge a downloaded repo folder into an existing model folder."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def install_normal_wan_repo(app_root: Path, model_dir: Path, force: bool = False) -> None:
    """Download and merge the official Wan2.2 GitHub repo into models/wan22.

    This is the Python equivalent of the original wan22_setup.bat repo step.
    The repo files are needed for normal WAN 2.2, not only for Turbo.
    """
    generate_py = model_dir / "generate.py"
    if generate_py.exists() and not force:
        log(f"\n[INFO] Normal Wan2.2 repo files already exist: {generate_py}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = app_root / "wan22_repo.zip"
    tmp_extract = app_root / "wan22_repo"
    if tmp_zip.exists():
        tmp_zip.unlink()
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)

    log("\n[INFO] Downloading official Wan2.2 GitHub repo for normal WAN 2.2...")
    download_url(WAN_GITHUB_ZIP, tmp_zip)
    tmp_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(tmp_extract)

    candidates = [p for p in tmp_extract.iterdir() if p.is_dir() and p.name.startswith("Wan2.2")]
    if not candidates:
        fail("Could not find extracted Wan2.2 repo folder in downloaded zip.")

    log(f"[INFO] Merging Wan2.2 repo files into: {model_dir}")
    copytree_merge(candidates[0], model_dir)

    tmp_zip.unlink(missing_ok=True)
    shutil.rmtree(tmp_extract, ignore_errors=True)
    if not generate_py.exists():
        fail(f"Normal Wan2.2 repo merge did not create expected file: {generate_py}")
    log(f"[OK] Normal Wan2.2 repo installed/repaired: {model_dir}")


def apply_wan22_patch_bundle(app_root: Path) -> None:
    """Extract presets/extra_env/wan22.zip into APP_ROOT/models.

    The old normal WAN installer did this as the final repair step. Keep it here
    so combining Turbo does not silently remove the normal WAN patches.
    """
    patch_zip = app_root / "presets" / "extra_env" / "wan22.zip"
    models_root = app_root / "models"
    log("\n[INFO] Applying normal WAN patch bundle...")
    log(f"       Source: {patch_zip}")
    log(f"       Target: {models_root}")
    if not patch_zip.exists():
        log(f"[WARN] wan22.zip not found, skipping patch bundle: {patch_zip}")
        return
    models_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(patch_zip, "r") as zf:
        zf.extractall(models_root)
    log(f"[OK] wan22.zip extracted into: {models_root}")


def install_turbo_repo(turbo_root: Path, force: bool = False) -> None:
    repo_dir = turbo_root / "Wan2.2-TI2V-5B-Turbo-main"
    if repo_dir.exists() and not force:
        log(f"\n[INFO] Turbo repo already exists: {repo_dir}")
        return

    turbo_root.mkdir(parents=True, exist_ok=True)
    tmp_zip = turbo_root / "_Wan2.2-TI2V-5B-Turbo-main.zip"
    tmp_extract = turbo_root / "_repo_extract"
    if tmp_zip.exists():
        tmp_zip.unlink()
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    log("\n[INFO] Downloading Turbo GitHub repo...")
    download_url(TURBO_GITHUB_ZIP, tmp_zip)
    tmp_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(tmp_extract)

    candidates = [p for p in tmp_extract.iterdir() if p.is_dir() and p.name.startswith("Wan2.2-TI2V-5B-Turbo")]
    if not candidates:
        fail("Could not find extracted Turbo repo folder in downloaded zip.")
    shutil.move(str(candidates[0]), str(repo_dir))
    tmp_zip.unlink(missing_ok=True)
    shutil.rmtree(tmp_extract, ignore_errors=True)
    log(f"[OK] Turbo repo installed: {repo_dir}")


def patch_file(path: Path, replacements: list[tuple[str, str]], marker: str) -> None:
    """Apply text replacements even when an older FrameVision marker exists.

    Earlier installer builds skipped a file as soon as they saw the marker. That
    made reruns unable to repair older/wrong WAN wrapper patches. This version is
    intentionally idempotent: it still checks every replacement, only prepends the
    marker once, and writes a numbered backup before changing anything.
    """
    if not path.exists():
        log(f"[WARN] Patch target missing: {path}")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text
    marker_present = marker in text
    changed_any = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed_any = True
        elif new in text:
            log(f"[INFO] Patch already present in {path.name}: {new[:80]!r}")
        else:
            log(f"[WARN] Pattern not found in {path.name}: {old[:80]!r}")
    if text != original:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.with_suffix(path.suffix + f".fvbak_{stamp}").write_text(original, encoding="utf-8")
        if not marker_present:
            text = marker + "\n" + text
        path.write_text(text, encoding="utf-8")
        log(f"[OK] Patched/repaired {path}")
    elif marker_present:
        log(f"[INFO] Already patched and no repair needed: {path.name}")
    else:
        log(f"[WARN] No changes made to {path}")


def patch_turbo_repo(repo_dir: Path, use_full_base: bool) -> None:
    log("\n[INFO] Applying FrameVision Turbo repo compatibility patches...")

    utils_file = repo_dir / "utils" / "wan_wrapper.py"
    marker = "# FrameVision Turbo installer patch applied"

    old_import = "from wan.modules.model import WanModel, CausalWanModel, RegisterTokens, GanAttentionBlock"
    new_import = "from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock\n# FrameVision: import CausalWanModel lazily only if causal mode is used.\nCausalWanModel = None"

    old_causal = """        if is_causal:\n            self.model = CausalWanModel.from_pretrained(\n                f\"wan_models/{model_name}/\", local_attn_size=local_attn_size, sink_size=sink_size)\n        else:\n            if \"2.2\" in model_name:\n                self.model = Wan22Model.from_pretrained(f\"wan_models/{model_name}/\")\n                self.seq_len = 27280  # [1, 31, 48, 44, 80]\n            else:\n                self.model = WanModel.from_pretrained(f\"wan_models/{model_name}/\")\n                self.seq_len = 32760  # [1, 21, 16, 60, 104]\n"""
    new_causal = """        if is_causal:\n            global CausalWanModel\n            if CausalWanModel is None:\n                from wan.modules.causal_model import CausalWanModel as _CausalWanModel\n                CausalWanModel = _CausalWanModel\n            self.model = CausalWanModel.from_pretrained(\n                f\"wan_models/{model_name}/\", local_attn_size=local_attn_size, sink_size=sink_size)\n        else:\n            if \"2.2\" in model_name:\n                model_dir = f\"wan_models/{model_name}/\"\n                use_base = os.environ.get(\"FV_WAN_TURBO_USE_BASE_WEIGHTS\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n                if use_base:\n                    self.model = Wan22Model.from_pretrained(model_dir)\n                else:\n                    # FrameVision Turbo minimal install: build the Wan 2.2 architecture\n                    # from config only, then wan2.2_fewstep.py loads the full Turbo\n                    # model.pt into pipe.generator. This avoids downloading the big\n                    # base diffusion_pytorch_model*.safetensors shards.\n                    self.model = Wan22Model.from_config(Wan22Model.load_config(model_dir))\n                self.seq_len = 27280  # [1, 31, 48, 44, 80]\n            else:\n                self.model = WanModel.from_pretrained(f\"wan_models/{model_name}/\")\n                self.seq_len = 32760  # [1, 21, 16, 60, 104]\n"""

    replacements = [(old_import, new_import)]
    if not use_full_base:
        replacements.append((old_causal, new_causal))
    else:
        replacements.append((old_causal, old_causal.replace("        if is_causal:", "        # FrameVision full-base mode\n        if is_causal:")))
    patch_file(utils_file, replacements, marker)

    # Some repo zips already contain this inference entry. Keep the check explicit
    # so optional installs can fail early instead of creating a broken UI entry.
    fewstep = repo_dir / "wan2.2_fewstep.py"
    if not fewstep.exists():
        fail(f"Turbo inference script missing: {fewstep}")


def is_windows_reparse_point(path: Path) -> bool:
    """Return True for Windows junctions/symlinks without following the target."""
    if os.name != "nt":
        return path.is_symlink()
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except Exception:
        return False


def safe_backup_existing_real_dir(path: Path, reason: str) -> Path:
    r"""Rename a wrong real folder instead of deleting model data.

    The fast WAN Turbo layout uses junctions under Turbo-main\wan_models. A
    fresh/broken install may contain real nested folders there. Deleting them is
    risky and can waste a huge download, so the installer renames them out of the
    way and then creates the correct junction.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}_wrong_nested_{stamp}")
    n = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}_wrong_nested_{stamp}_{n}")
        n += 1
    log(f"[WARN] {reason}")
    log(f"[WARN] Renaming existing real folder instead of deleting it:")
    log(f"       {path}")
    log(f"    -> {backup}")
    path.rename(backup)
    return backup


def remove_wrong_link_or_file(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if is_windows_reparse_point(path):
        # rmdir removes the junction itself, not the target.
        path.rmdir()
        return
    safe_backup_existing_real_dir(path, "Existing path is a real folder, not the expected junction.")


def create_dir_link(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        try:
            same_target = link.resolve() == target.resolve()
        except Exception:
            same_target = False
        if same_target and (link.is_symlink() or is_windows_reparse_point(link)):
            log(f"[INFO] Junction/link already OK: {link} -> {target}")
            return
        if same_target and link.is_dir() and not is_windows_reparse_point(link):
            # A real directory at this location can work functionally, but it is
            # exactly the layout that caused the slow/full-load branch. Replace it.
            safe_backup_existing_real_dir(
                link,
                "Existing path resolves to the right files but is a real folder; WAN Turbo expects the old fast junction layout.",
            )
        else:
            remove_wrong_link_or_file(link)

    if os.name == "nt":
        cmd = ["cmd", "/c", "mklink", "/J", str(link), str(target)]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            fail(f"Could not create junction: {link} -> {target}")
    else:
        os.symlink(target, link, target_is_directory=True)
    log(f"[OK] Junction/link created: {link} -> {target}")


def verify_dir_link(link: Path, target: Path) -> None:
    if not link.exists():
        fail(f"WAN Turbo model link is missing: {link}")
    try:
        resolved = link.resolve()
        same_target = resolved == target.resolve()
    except Exception:
        resolved = "<resolve failed>"
        same_target = False
    if not same_target:
        fail(
            "WAN Turbo model link points to the wrong target:\n"
            f"  {link}\n"
            "expected:\n"
            f"  {target}\n"
            "resolved:\n"
            f"  {resolved}"
        )
    if os.name == "nt" and not is_windows_reparse_point(link):
        fail(
            "WAN Turbo model path exists but is not a junction. This is the slow layout.\n"
            f"Path: {link}\n"
            f"Target should be: {target}"
        )


def create_turbo_wan_models_links(repo_dir: Path, base_model_dir: Path, turbo_model_dir: Path) -> None:
    log("\n[INFO] Creating/repairing Turbo repo wan_models junctions (old fast layout)...")
    wan_models = repo_dir / "wan_models"
    wan_models.mkdir(parents=True, exist_ok=True)
    base_link = wan_models / "Wan2.2-TI2V-5B"
    turbo_link = wan_models / "Wan2.2-TI2V-5B-Turbo"
    create_dir_link(base_link, base_model_dir)
    create_dir_link(turbo_link, turbo_model_dir)
    verify_dir_link(base_link, base_model_dir)
    verify_dir_link(turbo_link, turbo_model_dir)
    log("[OK] Turbo wan_models junction layout verified.")


def verify_install(py: Path, repo_dir: Path, model_dir: Path, turbo_model_dir: Path, full_base: bool) -> None:
    log("\n[INFO] Verifying installed files...")
    required = [
        repo_dir / "wan2.2_fewstep.py",
        repo_dir / "configs" / "inference" / "wan22.yaml",
        model_dir / "generate.py",
        model_dir / "wan" / "modules" / "model.py",
        model_dir / "Wan2.2_VAE.pth",
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "google" / "umt5-xxl",
        model_dir / "config.json",
        turbo_model_dir / "model.pt",
    ]
    if full_base:
        required.append(model_dir / "diffusion_pytorch_model.safetensors.index.json")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        fail("Install is incomplete. Missing:\n  " + "\n  ".join(missing))

    # Make sure the old fast Turbo layout is still present. Without these
    # junctions, the repo can fall into the slow nested/full-load layout.
    verify_dir_link(repo_dir / "wan_models" / "Wan2.2-TI2V-5B", model_dir)
    verify_dir_link(repo_dir / "wan_models" / "Wan2.2-TI2V-5B-Turbo", turbo_model_dir)

    log("[INFO] Verifying key imports in WAN env...")
    run([str(py), "-c", "import torch, transformers, diffusers, omegaconf, huggingface_hub; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); import flash_attn; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install FrameVision WAN 2.2 normal + Turbo optional model.")
    p.add_argument("--app-root", type=Path, default=None, help="FrameVision root. Defaults to two levels above presets/extra_env.")
    p.add_argument("--env-mode", choices=["auto", "conda", "venv"], default="auto", help="Environment creation mode. auto prefers conda and falls back to venv.")
    p.add_argument("--skip-env", action="store_true", help="Do not create/install Python env; only download models/repo and patch.")
    p.add_argument("--skip-flash", action="store_true", help="Skip Triton/FlashAttention install.")
    p.add_argument("--skip-downloads", action="store_true", help="Do not download HF/GitHub files; only patch/link/verify existing files.")
    p.add_argument("--force-repo", action="store_true", help="Redownload both normal Wan2.2 and Turbo GitHub repos even if they exist.")
    p.add_argument("--minimal-base", action="store_true", help="Advanced/debug only: skip the large normal WAN diffusion shards. Normal WAN 2.2 will not be fully installed.")
    p.add_argument("--download-full-base", action="store_true", help=argparse.SUPPRESS)  # backward compatible no-op; full base is now the default.
    p.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Normally use huggingface-cli login instead.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app_root = (args.app_root or script_app_root()).resolve()
    env_dir = app_root / "environments" / ".wan22_i2v"
    py = env_python(env_dir)
    model_dir = app_root / "models" / "wan22"
    turbo_root = model_dir / "wan_turbo"
    turbo_repo_dir = turbo_root / "Wan2.2-TI2V-5B-Turbo-main"
    turbo_model_dir = turbo_root / "Wan2.2-TI2V-5B-Turbo"

    log("\n====================================================")
    log("  FrameVision WAN 2.2 Turbo Optional Installer")
    log("====================================================\n")
    log(f"[INFO] App root       : {app_root}")
    log(f"[INFO] Env dir        : {env_dir}")
    log(f"[INFO] Base model dir : {model_dir}")
    log(f"[INFO] Turbo repo dir : {turbo_repo_dir}")
    log(f"[INFO] Turbo model dir: {turbo_model_dir}")
    full_base = not args.minimal_base
    log(f"[INFO] Base mode      : {'FULL normal WAN 2.2 + Turbo' if full_base else 'MINIMAL Turbo-only debug mode'}")

    if not args.skip_env:
        create_env(app_root, env_dir, args.env_mode)
        py = env_python(env_dir)
        install_env_packages(py, skip_flash=args.skip_flash)
    elif not py.exists():
        fail(f"--skip-env was used but env python does not exist: {py}")

    if not args.skip_downloads:
        download_models(py, model_dir, turbo_model_dir, args.hf_token, full_base)
        if full_base:
            install_normal_wan_repo(app_root, model_dir, force=args.force_repo)
        else:
            log("[WARN] Skipping normal Wan2.2 repo repair because --minimal-base was used.")
        install_turbo_repo(turbo_root, force=args.force_repo)
    else:
        log("[INFO] --skip-downloads used; using existing model/repo files.")

    patch_turbo_repo(turbo_repo_dir, use_full_base=full_base)
    create_turbo_wan_models_links(turbo_repo_dir, model_dir, turbo_model_dir)
    if full_base:
        apply_wan22_patch_bundle(app_root)
    verify_install(py, turbo_repo_dir, model_dir, turbo_model_dir, full_base)

    log("\n[SUCCESS] WAN 2.2 combined normal + Turbo optional install completed.")
    log(f"          Normal WAN 2.2 dir: {model_dir}")
    log(f"          Turbo repo       : {turbo_repo_dir}")
    log(f"          Turbo model      : {turbo_model_dir / 'model.pt'}")
    log("          Full base diffusion safetensors installed/repaired by default.")
    log("          wan22.zip patch bundle applied when present.")
    log("          Turbo wan_models junctions repaired/verified for the old fast layout.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        fail("Interrupted by user.", code=130)
