
from __future__ import annotations

"""Standalone FrameVision-style LTX 2.3 installer / repair script.

Safe defaults:
- Creates/repairs env at <root>/environments/.ltx23.
- Keeps official LTX repo at <root>/models/ltx23/repos/LTX-2.
- Keeps models at <root>/models/ltx23.
- Keeps FFmpeg at <root>/presets/bin.
- Uses <root>/temp and local HF/Torch caches only.
- Does not delete models.
- Refuses CPU-only Torch fallback.
- Installs optional FlashAttention, Triton, and SageAttention by default; use skip flags to disable.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENV_RELATIVE = Path("environments") / ".ltx23"
REPO_RELATIVE = Path("models") / "ltx23" / "repos" / "LTX-2"
CHECKPOINT_RELATIVE = Path("models") / "ltx23" / "distilled-1.1" / "ltx-2.3-22b-distilled-1.1.safetensors"
FP8_CHECKPOINT_RELATIVE = Path("models") / "ltx23" / "fp8" / "ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA_RELATIVE = Path("models") / "ltx23" / "text_encoder" / "lightricks_gemma_original"
SPLIT_RELATIVE = Path("models") / "ltx23" / "split"
FFMPEG_BIN_RELATIVE = Path("presets") / "bin"
TEMP_RELATIVE = Path("temp")
OFFICIAL_REPO_URL = "https://github.com/Lightricks/LTX-2"
HF_LTX23_REPO = "Lightricks/LTX-2.3"
HF_LTX23_FP8_REPO = "Lightricks/LTX-2.3-fp8"
HF_LTX23_SPLIT_REPO = "koongrizzly/ltx23_extracted_parts"
CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled-1.1.safetensors"
FP8_CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled-fp8.safetensors"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES: Sequence[str] = ("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0")
FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FLASH_ATTN_WHEEL_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl"
SAGE_ATTN_WHEEL_URL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"
TRITON_WINDOWS_RELEASES_URL = "https://github.com/woct0rdho/triton-windows/releases"
TRITON_WINDOWS_PACKAGE = "triton-windows"
SPLIT_PART_FILENAMES: Sequence[str] = (
    "vocoder.safetensors",
    "audio_vae.safetensors",
    "vae.safetensors",
)



def root_from_script() -> Path:
    # <root>/presets/extra_env/ltx23_install.py
    return Path(__file__).resolve().parents[2]


def status(kind: str, msg: str) -> None:
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


def portable_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    temp_dir = root / TEMP_RELATIVE
    cache_dir = temp_dir / "cache"
    for p in [temp_dir, cache_dir, cache_dir / "hf", cache_dir / "torch", cache_dir / "pip"]:
        p.mkdir(parents=True, exist_ok=True)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(cache_dir / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hf" / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache_dir / "hf" / "transformers")
    env["TORCH_HOME"] = str(cache_dir / "torch")
    env["PIP_CACHE_DIR"] = str(cache_dir / "pip")
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    return env


def env_python(env_path: Path) -> Path:
    return env_path / ("python.exe" if os.name == "nt" else "bin/python")


def find_existing_env_python(env_path: Path) -> Optional[Path]:
    candidates = [env_python(env_path), env_path / "Scripts" / "python.exe", env_path / "python.exe", env_path / "bin" / "python"]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def find_conda(root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    if os.environ.get("CONDA_EXE"):
        candidates.append(Path(os.environ["CONDA_EXE"]))
    if os.name == "nt":
        candidates += [root / "_miniconda" / "Scripts" / "conda.exe", root / "_miniconda" / "condabin" / "conda.bat", root / "_miniconda3" / "Scripts" / "conda.exe", root / "miniconda3" / "Scripts" / "conda.exe"]
    else:
        candidates += [root / "_miniconda" / "bin" / "conda", root / "_miniconda3" / "bin" / "conda", root / "miniconda3" / "bin" / "conda"]
    for c in candidates:
        if c.exists():
            status("FOUND", f"Conda: {c}")
            return c
    found = shutil.which("conda")
    if found:
        status("FOUND", f"Conda on PATH: {found}")
        return Path(found)
    return None


def assert_nvidia_present() -> None:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        raise RuntimeError("nvidia-smi was not found. Refusing CPU Torch fallback. Install/check NVIDIA driver first.")
    code = subprocess.run([nvidia], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if code != 0:
        raise RuntimeError("nvidia-smi exists but failed. Refusing CPU Torch fallback. Check NVIDIA driver first.")
    status("OK", "NVIDIA driver probe passed")


def create_or_repair_env(root: Path, env_path: Path, *, recreate: bool = False) -> Path:
    conda = find_conda(root)
    if not conda:
        raise RuntimeError("Conda was not found. Expected root\\_miniconda, root\\miniconda3, CONDA_EXE, or conda on PATH.")
    existing = find_existing_env_python(env_path)
    if existing and not recreate:
        status("FOUND", f"Env Python: {existing}")
        return existing
    if env_path.exists() and recreate:
        status("MISSING", "Danger recreate selected: deleting env only, not models")
        shutil.rmtree(env_path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    status("MISSING", f"Creating conda env: {env_path}")
    run([conda, "create", "--yes", "--prefix", env_path, "python=3.11", "pip"], cwd=root, env=portable_env(root), check=True)
    py = find_existing_env_python(env_path)
    if not py:
        raise RuntimeError(f"Conda env was created but python.exe was not found under {env_path}")
    return py


def pip_install(py: Path, packages: Iterable[object], root: Path, *, label: str, extra_args: Optional[list[object]] = None) -> None:
    cmd: list[object] = [py, "-m", "pip", "install", "--no-warn-script-location"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    status("DOWNLOADING", label)
    run(cmd, cwd=root, env=portable_env(root), check=True)


def ensure_openimageio(py: Path, root: Path) -> None:
    """OpenImageIO is required by ltx_pipelines.utils.media_io at import time."""
    ok, detail = verify_import(py, root, "OpenImageIO")
    if ok:
        status("FOUND", f"OpenImageIO already importable: {detail}")
        status("SKIPPED", "OpenImageIO repair install skipped")
        return
    status("MISSING", f"OpenImageIO not importable yet: {detail}")
    pip_install(py, ["OpenImageIO"], root, label="OpenImageIO runtime dependency")
    ok, detail = verify_import(py, root, "OpenImageIO")
    if not ok:
        raise RuntimeError(f"OpenImageIO install/repair failed: {detail}")
    status("OK", f"OpenImageIO import check: {detail}")




def ensure_opencv(py: Path, root: Path) -> None:
    """OpenCV is required for LTX/MSR reference-video preview export.

    The normal MSR reference frames can be built without OpenCV, but the
    ComfyUI-style preview/video export path uses cv2. Install it in the LTX
    environment so --msr-save-video works from the UI and CLI.
    """
    ok, detail = verify_import(py, root, "cv2")
    if ok:
        status("FOUND", f"OpenCV already importable: {detail}")
        status("SKIPPED", "OpenCV repair install skipped")
        return
    status("MISSING", f"OpenCV not importable yet: {detail}")
    pip_install(py, ["opencv-python"], root, label="OpenCV runtime dependency for LTX/MSR reference video export")
    ok, detail = verify_import(py, root, "cv2")
    if not ok:
        raise RuntimeError(f"OpenCV install/repair failed: {detail}")
    status("OK", f"OpenCV import check: {detail}")

def install_deps(py: Path, root: Path) -> None:
    script_dir = Path(__file__).resolve().parent
    req = script_dir / "req_ltx.txt"
    if not req.exists():
        req = script_dir / "requirements_ltx_drama_tuple.txt"
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=root, env=portable_env(root), check=True)
    pip_install(py, TORCH_PACKAGES, root, label="CUDA Torch tuple 2.8.0/cu128", extra_args=["--index-url", PYTORCH_CUDA_INDEX])
    pip_install(py, ["-r", req], root, label=f"LTX + standalone UI requirements ({req.name})")
    ensure_openimageio(py, root)
    ensure_opencv(py, root)


def repo_has_expected_packages(path: Path) -> bool:
    return (path / "packages" / "ltx-core").exists() and (path / "packages" / "ltx-pipelines").exists()


def ensure_repo(root: Path, repo_path: Path) -> Path:
    if repo_has_expected_packages(repo_path):
        status("FOUND", f"LTX repo: {repo_path}")
        status("SKIPPED", "Repo already exists")
        return repo_path
    git = shutil.which("git")
    if not git:
        raise RuntimeError(f"Git was not found and LTX repo is missing. Place official repo at: {repo_path}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if repo_path.exists() and any(repo_path.iterdir()):
        raise RuntimeError(f"Repo path exists but is not a valid LTX-2 repo. Not deleting it: {repo_path}")
    status("MISSING", f"LTX repo: {repo_path}")
    status("DOWNLOADING", "Cloning official LTX-2 repo code")
    run([git, "clone", "--depth", "1", OFFICIAL_REPO_URL, repo_path], cwd=root, env=portable_env(root), check=True)
    if not repo_has_expected_packages(repo_path):
        raise RuntimeError(f"Cloned repo does not contain expected package folders: {repo_path}")
    return repo_path


def install_ltx_packages(py: Path, root: Path, repo_path: Path) -> None:
    core = repo_path / "packages" / "ltx-core"
    pipes = repo_path / "packages" / "ltx-pipelines"
    status("OK", "Installing official ltx-core / ltx-pipelines from local repo with --no-deps")
    run([py, "-m", "pip", "install", "--no-warn-script-location", "--no-deps", "-e", core, "-e", pipes], cwd=root, env=portable_env(root), check=True)


def model_variant_paths(root: Path, variant: str) -> tuple[Path, str, str, str]:
    """Return the checkpoint path/repo/filename for the selected LTX model variant.

    fp16 keeps the existing tested distilled 1.1 layout.
    fp8 downloads only the official distilled FP8 checkpoint into a separate
    models/ltx23/fp8 folder while reusing the same env, repo, Gemma text encoder,
    FFmpeg tools, and other shared LTX files.
    """
    v = (variant or "fp16").strip().lower()
    if v in {"fp8", "distilled-fp8", "distilled_fp8"}:
        return (
            root / FP8_CHECKPOINT_RELATIVE,
            HF_LTX23_FP8_REPO,
            FP8_CHECKPOINT_FILENAME,
            "LTX 2.3 distilled FP8",
        )
    if v in {"fp16", "distilled-1.1", "distilled_1.1", "default"}:
        return (
            root / CHECKPOINT_RELATIVE,
            HF_LTX23_REPO,
            CHECKPOINT_FILENAME,
            "LTX 2.3 distilled 1.1 FP16",
        )
    raise RuntimeError(f"Unknown LTX model variant: {variant!r}. Use fp16 or fp8.")


def ensure_checkpoint(
    py: Path,
    root: Path,
    ckpt: Path,
    *,
    repo_id: str,
    filename: str,
    label: str,
    skip_downloads: bool,
) -> None:
    if ckpt.exists() and ckpt.stat().st_size > 1024 * 1024:
        status("FOUND", f"{label} checkpoint: {ckpt}")
        status("SKIPPED", "Big safetensors already exists")
        return
    status("MISSING", f"{label} checkpoint: {ckpt}")
    if skip_downloads:
        return
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    code = f"""
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, local_dir={str(ckpt.parent)!r}, local_dir_use_symlinks=False)
print(p)
"""
    status("DOWNLOADING", f"{label} checkpoint from Hugging Face repo {repo_id}")
    run([py, "-c", code], cwd=root, env=portable_env(root), check=True)
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint download finished but expected file was not found: {ckpt}")



def ensure_ltx23_split_parts(py: Path, root: Path, split_dir: Path, *, skip_downloads: bool) -> None:
    """Download/repair the default LTX 2.3 split components.

    These files are used by FrameVision's late final-load split routing so long
    480p runs do not need to reopen the full 42.98 GB checkpoint for final
    decoder/audio/vocoder pieces.

    Required files:
    - vocoder.safetensors -> Vocoder
    - audio_vae.safetensors -> AudioDecoder
    - vae.safetensors -> VideoDecoder
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    for target_name in SPLIT_PART_FILENAMES:
        target = split_dir / target_name
        if target.exists() and target.stat().st_size > 1024:
            status("FOUND", f"LTX split part: {target} ({target.stat().st_size / (1024 ** 2):.2f} MB)")
            continue
        missing.append(target_name)

    if not missing:
        status("SKIPPED", "All required LTX split parts already exist")
        return

    for target_name in missing:
        target = split_dir / target_name
        status("MISSING", f"LTX split part: {target}")

    if skip_downloads:
        status("SKIPPED", "LTX split part downloads skipped")
        return

    for target_name in missing:
        target = split_dir / target_name
        code = f"""
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id={HF_LTX23_SPLIT_REPO!r},
    filename={target_name!r},
    local_dir={str(split_dir)!r},
    local_dir_use_symlinks=False,
)
print(p)
"""
        status("DOWNLOADING", f"LTX split part {target_name} from Hugging Face repo {HF_LTX23_SPLIT_REPO}")
        run([py, "-c", code], cwd=root, env=portable_env(root), check=True)

        if target.exists() and target.stat().st_size > 1024:
            status("FOUND", f"Split part: {target.name} ({target.stat().st_size / (1024 ** 2):.2f} MB)")
        else:
            raise RuntimeError(f"LTX split download finished but expected file was not found: {target}")


def gemma_present(gemma: Path) -> bool:
    return gemma.exists() and any((gemma / n).exists() for n in ["config.json", "model.safetensors.index.json", "tokenizer.json"])


def ensure_gemma(py: Path, root: Path, gemma: Path, *, skip_downloads: bool) -> None:
    if gemma_present(gemma):
        status("FOUND", f"Gemma root: {gemma}")
        status("SKIPPED", "Text encoder folder already exists")
        return
    status("MISSING", f"Gemma root: {gemma}")
    if skip_downloads:
        return
    gemma.parent.mkdir(parents=True, exist_ok=True)
    # First try the official LTX repo layout. If the exact folder is not present upstream,
    # the error is shown clearly and user can copy the existing folder manually.
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id={HF_LTX23_REPO!r}, local_dir={str(root / 'models' / 'ltx23')!r}, local_dir_use_symlinks=False, allow_patterns=['text_encoder/lightricks_gemma_original/**'])
print('done')
"""
    status("DOWNLOADING", f"Text encoder from Hugging Face repo {HF_LTX23_REPO}")
    run([py, "-c", code], cwd=root, env=portable_env(root), check=True)
    if not gemma_present(gemma):
        raise RuntimeError("Gemma text encoder download did not produce expected folder. Copy your existing lightricks_gemma_original folder to models\\ltx23\\text_encoder\\lightricks_gemma_original.")


def ensure_ffmpeg(root: Path, *, skip_downloads: bool) -> None:
    bin_dir = root / FFMPEG_BIN_RELATIVE
    needed = [bin_dir / "ffmpeg.exe", bin_dir / "ffprobe.exe", bin_dir / "ffplay.exe"]
    if all(p.exists() and p.stat().st_size > 0 for p in needed):
        status("FOUND", "FFmpeg tools in presets\\bin")
        status("SKIPPED", "FFmpeg already exists")
        return
    status("MISSING", "One or more FFmpeg tools are missing")
    if skip_downloads:
        return
    temp_dir = root / TEMP_RELATIVE / "ffmpeg_download"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_dir / "ffmpeg-release-essentials.zip"
    status("DOWNLOADING", FFMPEG_ZIP_URL)
    urllib.request.urlretrieve(FFMPEG_ZIP_URL, zip_path)
    status("OK", "Unpacking FFmpeg bundle")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    found = {name: None for name in ["ffmpeg.exe", "ffprobe.exe", "ffplay.exe"]}
    for p in temp_dir.rglob("*.exe"):
        if p.name.lower() in found:
            found[p.name.lower()] = p
    for name, src in found.items():
        if src is None:
            raise RuntimeError(f"Could not find {name} inside FFmpeg bundle")
        shutil.copy2(src, bin_dir / name)
        status("OK", f"Copied {name}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    status("OK", "Cleaned FFmpeg temp files")



def run_python_capture(py: Path, code: str, root: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        [str(py), "-c", code],
        cwd=str(root),
        env=portable_env(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return int(completed.returncode), completed.stdout.strip(), completed.stderr.strip()


def verify_import(py: Path, root: Path, module_name: str, import_name: Optional[str] = None) -> tuple[bool, str]:
    target = import_name or module_name
    code = f"""
import importlib
m = importlib.import_module({target!r})
print(getattr(m, '__version__', getattr(m, 'version', 'version unavailable')))
"""
    rc, out, err = run_python_capture(py, code, root)
    if rc == 0:
        return True, out or "import OK"
    return False, (err or out or "import failed").splitlines()[-1]


def torch_cuda_is_healthy(py: Path, root: Path) -> bool:
    code = """
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
"""
    rc, out, err = run_python_capture(py, code, root)
    if out:
        status("OK" if rc == 0 else "FAILED", out)
    if err and rc != 0:
        status("FAILED", err.splitlines()[-1])
    return rc == 0


def install_optional_flash_attention(py: Path, root: Path) -> bool:
    status("OK", "Optional FlashAttention requested")
    if not torch_cuda_is_healthy(py, root):
        status("FAILED", "Torch CUDA is broken; refusing to install optional FlashAttention")
        return False
    ok, detail = verify_import(py, root, "flash_attn")
    if ok:
        status("FOUND", f"flash_attn already importable: {detail}")
        status("SKIPPED", "FlashAttention install skipped")
        return True
    status("MISSING", f"flash_attn not importable yet: {detail}")
    status("DOWNLOADING", "Installing exact FlashAttention wheel for Python 3.11 + Torch 2.8 + CUDA 12.8")
    rc = run(
        [py, "-m", "pip", "install", "--no-warn-script-location", "--no-deps", FLASH_ATTN_WHEEL_URL],
        cwd=root,
        env=portable_env(root),
    )
    if rc != 0:
        status("FAILED", "FlashAttention wheel install failed")
        return False
    ok, detail = verify_import(py, root, "flash_attn")
    status("OK" if ok else "FAILED", f"flash_attn import check: {detail}")
    if ok:
        status("WARN", "Installed/importable does not guarantee LTX is using this backend. Use the LTX runtime profiler to confirm the active attention path.")
    return ok


def install_optional_sage_attention(py: Path, root: Path) -> bool:
    status("OK", "Optional SageAttention requested")
    if not torch_cuda_is_healthy(py, root):
        status("FAILED", "Torch CUDA is broken; refusing to install optional SageAttention")
        return False
    ok, detail = verify_import(py, root, "sageattention")
    if ok:
        status("FOUND", f"sageattention already importable: {detail}")
        status("SKIPPED", "SageAttention install skipped")
        return True
    status("MISSING", f"sageattention not importable yet: {detail}")
    status("DOWNLOADING", "Installing exact SageAttention wheel for Python 3.9+ ABI3 + Torch 2.8 + CUDA 12.8")
    rc = run(
        [py, "-m", "pip", "install", "--no-warn-script-location", "--no-deps", SAGE_ATTN_WHEEL_URL],
        cwd=root,
        env=portable_env(root),
    )
    if rc != 0:
        status("FAILED", "SageAttention wheel install failed")
        return False
    ok, detail = verify_import(py, root, "sageattention")
    status("OK" if ok else "FAILED", f"sageattention import check: {detail}")
    if ok:
        status("WARN", "Installed/importable does not guarantee LTX is using Sage. Use the LTX runtime profiler to confirm the active attention path.")
    return ok


def install_optional_triton(py: Path, root: Path) -> bool:
    status("OK", "Optional Triton requested")
    if not torch_cuda_is_healthy(py, root):
        status("FAILED", "Torch CUDA is broken; refusing to install optional Triton")
        return False
    ok, detail = verify_import(py, root, "triton")
    if ok:
        status("FOUND", f"triton already importable: {detail}")
        status("SKIPPED", "Triton install skipped")
        return True

    probe = """
import platform, torch
print(platform.system(), platform.machine(), platform.python_version(), torch.__version__, torch.version.cuda)
"""
    rc, out, err = run_python_capture(py, probe, root)
    status("OK" if rc == 0 else "WARN", f"Triton compatibility probe: {out or err}")

    # Keep this conservative: install the Windows package only as a binary package,
    # with --no-deps so it cannot downgrade/replace the known-good CUDA Torch tuple.
    # pip still filters by Python ABI / Windows wheel tag. Import verification below
    # decides whether it is usable. Failure is non-fatal.
    status("DOWNLOADING", f"Trying safe binary-only install: {TRITON_WINDOWS_PACKAGE}")
    rc = run(
        [py, "-m", "pip", "install", "--no-warn-script-location", "--only-binary", ":all:", "--no-deps", TRITON_WINDOWS_PACKAGE],
        cwd=root,
        env=portable_env(root),
    )
    if rc != 0:
        status("WARN", f"Triton install failed or no compatible wheel was selected automatically. Check manually: {TRITON_WINDOWS_RELEASES_URL}")
        return False
    ok, detail = verify_import(py, root, "triton")
    status("OK" if ok else "WARN", f"triton import check: {detail}")
    if ok:
        status("WARN", "Installed/importable does not guarantee LTX is using this backend. Use the LTX runtime profiler to confirm the active attention path.")
        return True
    status("WARN", f"Triton package installed but import failed. Check manually: {TRITON_WINDOWS_RELEASES_URL}")
    return False


def verify_kernel_stack(py: Path, root: Path) -> int:
    status("OK", "Optional kernel stack verification")
    code = r"""
import importlib
import torch

def line(kind, msg):
    print(f"[{kind}] {msg}")

try:
    line("OK", f"torch version: {torch.__version__}")
    line("OK", f"torch CUDA version: {torch.version.cuda}")
    cuda_ok = bool(torch.cuda.is_available())
    line("OK" if cuda_ok else "FAILED", f"torch.cuda.is_available(): {cuda_ok}")
    if cuda_ok:
        line("OK", f"GPU name: {torch.cuda.get_device_name(0)}")
        line("OK", f"CUDA capability: {torch.cuda.get_device_capability(0)}")
except Exception as exc:
    line("FAILED", f"torch probe failed: {exc}")
    raise SystemExit(10)

for module_name in ["flash_attn", "triton", "xformers", "sageattention"]:
    try:
        m = importlib.import_module(module_name)
        version = getattr(m, "__version__", getattr(m, "version", "version unavailable"))
        line("OK", f"{module_name}: importable / {version}")
    except ModuleNotFoundError:
        line("WARN", f"{module_name}: MISSING optional kernel")
    except Exception as exc:
        line("WARN", f"{module_name}: FAILED optional import: {exc}")

print("[WARN] Installed/importable does not guarantee LTX is using this backend. Use the LTX runtime profiler to confirm the active attention path.")
raise SystemExit(0 if torch.cuda.is_available() else 11)
"""
    return run([py, "-c", code], cwd=root, env=portable_env(root))


def verify(py: Path, root: Path, repo: Path, ckpt: Path, gemma: Path) -> int:
    failed: list[str] = []
    def check(name: str, ok: bool, detail: str = "") -> None:
        status("OK" if ok else "FAILED", f"{name}{(': ' + detail) if detail else ''}")
        if not ok: failed.append(name)
    check("python.exe exists and is not 0 KB", py.exists() and py.stat().st_size > 0, str(py))
    check("LTX repo exists", repo_has_expected_packages(repo), str(repo))
    check("checkpoint exists", ckpt.exists() and ckpt.stat().st_size > 1024 * 1024, str(ckpt))
    check("Gemma root exists", gemma_present(gemma), str(gemma))
    split_dir = root / SPLIT_RELATIVE
    check("LTX split parts exist", all((split_dir / n).exists() and (split_dir / n).stat().st_size > 1024 for n in SPLIT_PART_FILENAMES), str(split_dir))
    check("FFmpeg exists", all((root / FFMPEG_BIN_RELATIVE / n).exists() for n in ["ffmpeg.exe", "ffprobe.exe", "ffplay.exe"]), str(root / FFMPEG_BIN_RELATIVE))
    checks = [
        ("Python launches", "import sys; print(sys.version)"),
        ("torch imports + CUDA true", "import torch; print(torch.__version__, torch.version.cuda); raise SystemExit(0 if torch.cuda.is_available() else 77)"),
        ("ltx_core imports", "import ltx_core; print(getattr(ltx_core, '__file__', 'ok'))"),
        ("OpenImageIO imports", "import OpenImageIO; print(getattr(OpenImageIO, '__version__', 'import OK'))"),
        ("OpenCV imports", "import cv2; print(cv2.__version__)"),
        ("ltx_pipelines imports", "import importlib.util; s=importlib.util.find_spec('ltx_pipelines'); print(s); raise SystemExit(0 if s else 78)"),
        ("PySide6 imports", "import PySide6; print(PySide6.__version__)"),
    ]
    env = portable_env(root)
    for label, code in checks:
        rc = run([py, "-c", code], cwd=root, env=env)
        check(label, rc == 0)
    return 0 if not failed else 20


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone FrameVision-style LTX 2.3 installer / repair")
    ap.add_argument("--root", default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--repair", action="store_true", help="Safe default: create/update missing pieces without deleting models")
    ap.add_argument("--model-variant", choices=("fp16", "fp8"), default="fp16", help="Checkpoint to install/verify. fp16 is the existing distilled 1.1 model; fp8 downloads the official distilled FP8 checkpoint.")
    ap.add_argument("--fp8", action="store_true", help="Shortcut for --model-variant fp8")
    ap.add_argument("--skip-downloads", action="store_true", help="Only verify/report missing downloads")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--skip-model-downloads", action="store_true")
    ap.add_argument("--danger-recreate-env", action="store_true", help="Delete and recreate env only. Never deletes models.")
    ap.add_argument("--confirm-recreate-env", default="", help="Must be DELETE_ENV_ONLY when using --danger-recreate-env")
    ap.add_argument("--install-flash-attn", action="store_true", help="Legacy/no-op: FlashAttention installs by default unless --skip-flash-attn is used")
    ap.add_argument("--install-triton", action="store_true", help="Legacy/no-op: Triton installs by default unless --skip-triton is used")
    ap.add_argument("--install-sage-attn", action="store_true", help="Legacy/no-op: SageAttention installs by default unless --skip-sage-attn is used")
    ap.add_argument("--skip-flash-attn", action="store_true", help="Disable default FlashAttention install/verify")
    ap.add_argument("--skip-sage-attn", action="store_true", help="Disable default SageAttention install/verify")
    ap.add_argument("--skip-triton", action="store_true", help="Disable default Triton install/verify")
    ap.add_argument("--verify-kernels", action="store_true", help="Verify optional acceleration kernels without requiring them; normal install verifies after attempting them")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else root_from_script()
    env_path = root / ENV_RELATIVE
    repo_path = root / REPO_RELATIVE
    variant = "fp8" if args.fp8 else args.model_variant
    ckpt, ckpt_repo, ckpt_filename, ckpt_label = model_variant_paths(root, variant)
    gemma = root / GEMMA_RELATIVE
    for d in [root / TEMP_RELATIVE, root / "models" / "ltx23", root / SPLIT_RELATIVE, ckpt.parent, root / "presets" / "bin", root / "tools" / "vram_lab", root / "helpers", root / "presets" / "setsave"]:
        d.mkdir(parents=True, exist_ok=True)

    status("OK", f"Standalone root: {root}")
    status("OK", f"Selected LTX model variant: {ckpt_label}")
    if args.danger_recreate_env and args.confirm_recreate_env != "DELETE_ENV_ONLY":
        raise RuntimeError("Danger recreate refused. Add --confirm-recreate-env DELETE_ENV_ONLY. This deletes env only, never models.")

    try:
        assert_nvidia_present()
        py = create_or_repair_env(root, env_path, recreate=args.danger_recreate_env)
        if not args.verify_only and not args.skip_deps:
            install_deps(py, root)
        elif args.skip_deps:
            status("SKIPPED", "Dependency install skipped")
        # Optional acceleration kernels are now attempted by default during a normal install/repair.
        # Use --skip-flash-attn / --skip-triton / --skip-sage-attn to disable them.
        # Keep --install-* as legacy no-op flags.
        if args.verify_only:
            if args.install_flash_attn or args.install_sage_attn or args.install_triton:
                status("WARN", "--verify-only is active, so optional kernel installs are not performed")
        else:
            if not args.skip_flash_attn:
                install_optional_flash_attention(py, root)
            else:
                status("SKIPPED", "FlashAttention optional install skipped by --skip-flash-attn")
            if not args.skip_triton:
                install_optional_triton(py, root)
            else:
                status("SKIPPED", "Triton optional install skipped by --skip-triton")
            if not args.skip_sage_attn:
                install_optional_sage_attention(py, root)
            else:
                status("SKIPPED", "SageAttention optional install skipped by --skip-sage-attn")

        if args.verify_kernels or (not args.verify_only and not (args.skip_flash_attn and args.skip_sage_attn and args.skip_triton)):
            krc = verify_kernel_stack(py, root)
            if krc != 0:
                status("FAILED", "Kernel verification found fatal Torch CUDA failure")
                return krc
        repo = ensure_repo(root, repo_path) if not args.verify_only else repo_path
        if not args.verify_only:
            install_ltx_packages(py, root, repo)
            ensure_checkpoint(py, root, ckpt, repo_id=ckpt_repo, filename=ckpt_filename, label=ckpt_label, skip_downloads=args.skip_downloads or args.skip_model_downloads)
            ensure_ltx23_split_parts(py, root, root / SPLIT_RELATIVE, skip_downloads=args.skip_downloads or args.skip_model_downloads)
            ensure_gemma(py, root, gemma, skip_downloads=args.skip_downloads or args.skip_model_downloads)
            ensure_ffmpeg(root, skip_downloads=args.skip_downloads)
        rc = verify(py, root, repo_path, ckpt, gemma)
        if rc == 0:
            status("OK", "Verification passed")
        else:
            status("FAILED", "Verification found missing/broken pieces")
        return rc
    except Exception as exc:
        status("FAILED", str(exc))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
