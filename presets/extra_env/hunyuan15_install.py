from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENV_NAME = ".hunyuan15_official"
REQ_NAME = "hunyuan15_req.txt"
DEFAULT_MODEL = "480p_t2v_distilled"
# Updated Hunyuan runtime target: Python 3.11 + PyTorch 2.6.0 + CUDA 12.6.
# These Windows wheels are installed after the base Diffusers runtime verifies.
FLASH_WHEEL_CU126_TORCH260_PY311 = (
    "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/"
    "flash_attn-2.7.4+cu126torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl?download=true"
)
SAGE_WHEEL_CU126_TORCH260_PY311 = (
    "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/"
    "sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl"
)
# woct0rdho's Triton Windows 3.2.x line is the PyTorch 2.6-compatible line.
TRITON_WINDOWS_SPEC_TORCH260 = "triton-windows>=3.2,<3.3"

REQ_FALLBACK = [
    "huggingface_hub>=0.24.6",
    "diffusers>=0.33.0",
    "transformers>=4.45.0",
    "accelerate>=0.33.0",
    "safetensors>=0.4.5",
    "numpy>=1.26",
    "pillow>=10.0",
    "tqdm>=4.66",
    "requests>=2.31",
    "packaging>=24.0",
    "pyyaml>=6.0",
    "imageio>=2.34",
    "imageio-ffmpeg>=0.5.1",
    "opencv-python>=4.8",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25",
    "einops>=0.8",
]


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def is_windows() -> bool:
    return os.name == "nt"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if (" " in str(c) or "\t" in str(c)) else str(c) for c in cmd)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, env: dict[str, str] | None = None) -> int:
    _print(f"[cmd] {quote_cmd([str(x) for x in cmd])}")
    proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {quote_cmd([str(x) for x in cmd])}")
    return int(proc.returncode)


def run_capture(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        return int(proc.returncode), proc.stdout or ""
    except Exception as exc:
        return 999, str(exc)


def project_root_from_script(script_path: Path) -> Path:
    # Expected: <root>/presets/extra_env/hunyuan15_install.py
    p = script_path.resolve()
    if p.parent.name.lower() == "extra_env" and p.parent.parent.name.lower() == "presets":
        return p.parent.parent.parent
    return Path.cwd().resolve()


def env_python(env_dir: Path) -> Path:
    if is_windows():
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def venv_python(env_dir: Path) -> Path:
    if is_windows():
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def find_conda(explicit: str | None = None) -> Path | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for name in ("CONDA_EXE", "MAMBA_EXE", "MICROMAMBA_EXE"):
        val = os.environ.get(name)
        if val:
            candidates.append(val)
    for exe in ("conda", "mamba", "micromamba"):
        found = shutil.which(exe)
        if found:
            candidates.append(found)
    seen = set()
    for c in candidates:
        try:
            p = Path(c).expanduser().resolve()
        except Exception:
            p = Path(c)
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.exists() or shutil.which(str(c)):
            return p
    return None


def conda_kind(conda_exe: Path) -> str:
    name = conda_exe.name.lower()
    if "micromamba" in name:
        return "micromamba"
    if "mamba" in name:
        return "mamba"
    return "conda"


def is_conda_env(env_dir: Path) -> bool:
    return (env_dir / "conda-meta").exists() and env_python(env_dir).exists()


def backup_non_conda_env(env_dir: Path, force_backup: bool = True) -> Path | None:
    """If env_dir is an old venv/broken folder, move it aside so conda can create the prefix."""
    if not env_dir.exists():
        return None
    if is_conda_env(env_dir):
        return None
    if env_python(env_dir).exists() and (env_dir / "conda-meta").exists():
        return None
    # A non-empty venv folder under the same target blocks conda create -p.
    if not force_backup:
        raise RuntimeError(
            f"Target env exists but is not a conda env: {env_dir}\n"
            f"Delete it or rerun without --no-backup-existing."
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = env_dir.with_name(env_dir.name + f"_venv_backup_{stamp}")
    _print(f"[repair] Existing target is not a conda env; moving it aside:")
    _print(f"         {env_dir}")
    _print(f"      -> {backup}")
    shutil.move(str(env_dir), str(backup))
    return backup


def create_or_reuse_conda_env(root: Path, env_dir: Path, conda_exe: Path, python_version: str, no_backup_existing: bool = False) -> Path:
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    py = env_python(env_dir)
    if is_conda_env(env_dir):
        _print(f"[ok] conda env exists: {env_dir}")
        return py

    backup_non_conda_env(env_dir, force_backup=not no_backup_existing)
    kind = conda_kind(conda_exe)
    _print(f"[1/8] Creating conda env: {env_dir}")
    _print(f"      Conda tool: {conda_exe} ({kind})")
    if kind == "micromamba":
        cmd = [str(conda_exe), "create", "-y", "-p", str(env_dir), f"python={python_version}", "pip"]
    else:
        cmd = [str(conda_exe), "create", "-y", "-p", str(env_dir), f"python={python_version}", "pip"]
    run(cmd, cwd=root)
    if not py.exists():
        raise RuntimeError(f"Conda env was created but root python is missing: {py}")
    return py


def pip_install(py: Path, args: list[str]) -> None:
    final_args = list(args)
    if final_args and final_args[0] == "install" and "--no-warn-script-location" not in final_args:
        final_args.insert(1, "--no-warn-script-location")
    run([str(py), "-m", "pip", *final_args])


def install_torch(py: Path, cuda_tag: str, torch_version: str) -> None:
    _print("[3/8] Installing CUDA PyTorch")
    torch_version = str(torch_version or "2.6.0").strip()
    if torch_version != "2.6.0":
        raise RuntimeError(
            f"Unsupported Hunyuan torch version for this installer: {torch_version}. "
            "This installer is pinned to torch 2.6.0 so the cu126 Flash/Sage wheels match."
        )
    pip_install(py, [
        "install", "--upgrade", "--force-reinstall",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    ])

def verify_cuda_torch(py: Path) -> None:
    _print("[guard] Verifying CUDA torch")
    run([str(py), "-c", "import torch,sys; print('python', sys.executable); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)"])


def uninstall_kernels(py: Path) -> None:
    """Remove kernels-hub package because it breaks HunyuanVideo 1.5 Diffusers imports on this stack."""
    _print("[4/8] Removing kernels package if present (Hunyuan import guard)")
    run([str(py), "-m", "pip", "uninstall", "-y", "kernels"], check=False)


def install_attention_addons(py: Path, cuda_tag: str, torch_version: str) -> None:
    """Install only Triton by default. Keep Sage/Flash code disabled for now.

    HunyuanVideo 1.5 currently passes attn_mask through the Diffusers attention
    path. The cu126/torch2.6 Sage and Flash wheels import correctly, but both
    fail during generation with attn_mask errors. Keep their install code gated
    behind HY15_ENABLE_FLASH_SAGE=1 so it is easy to re-enable later without
    silently breaking normal installs.
    """
    if os.environ.get("HY15_NO_ATTENTION_ADDONS", "").strip().lower() in {"1", "true", "yes", "on"}:
        _print("[7/8] Skipping Triton/Sage/Flash (HY15_NO_ATTENTION_ADDONS=1)")
        return

    cuda_tag = str(cuda_tag or "").lower().strip()
    torch_version = str(torch_version or "").strip()
    if cuda_tag != "cu126" or torch_version != "2.6.0":
        raise RuntimeError(
            "Triton/Sage/Flash add-ons are pinned to Python 3.11 + torch 2.6.0 + CUDA 12.6. "
            f"Current request: torch={torch_version}, cuda={cuda_tag}."
        )

    probe = (
        "import sys, torch; "
        "ok=(sys.version_info[:2]==(3,11) and getattr(torch,'__version__','')=='2.6.0+cu126'); "
        "print('python', sys.version.split()[0]); print('torch', getattr(torch,'__version__','')); "
        "print('cuda_build', getattr(torch.version,'cuda',None)); "
        "sys.exit(0 if ok else 2)"
    )
    rc, out = run_capture([str(py), "-c", probe])
    if out.strip():
        _print(out.strip())
    if rc != 0:
        raise RuntimeError("Attention add-ons require Python 3.11 + torch 2.6.0+cu126 before installation.")

    _print("[7/8] Installing Triton Windows for torch 2.6")
    pip_install(py, ["install", "--upgrade", TRITON_WINDOWS_SPEC_TORCH260])
    verify_attention_addons(py, include_flash_sage=False)

    if os.environ.get("HY15_ENABLE_FLASH_SAGE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        _print("[7/8] SageAttention and FlashAttention installs are disabled for Hunyuan by default.")
        _print("      Reason: both currently fail this Hunyuan Diffusers path with attn_mask errors.")
        _print("      Set HY15_ENABLE_FLASH_SAGE=1 only for manual experiments.")
        return

    _print("[7/8] EXPERIMENTAL: Installing SageAttention cu126/torch2.6 wheel")
    pip_install(py, ["install", "--upgrade", "--force-reinstall", "--no-deps", SAGE_WHEEL_CU126_TORCH260_PY311])

    _print("[7/8] EXPERIMENTAL: Installing FlashAttention cu126/torch2.6 wheel")
    pip_install(py, ["install", "--upgrade", "--force-reinstall", "--no-deps", FLASH_WHEEL_CU126_TORCH260_PY311])

    verify_attention_addons(py, include_flash_sage=True)


def verify_attention_addons(py: Path, include_flash_sage: bool = False) -> None:
    _print("[guard] Verifying Triton import" + (" + Sage/Flash imports" if include_flash_sage else ""))
    code = """
import torch
print('torch', torch.__version__)
print('cuda_build', torch.version.cuda)
import triton
print('triton', getattr(triton, '__version__', 'unknown'))
""".strip()
    if include_flash_sage:
        code += """
import sageattention
print('sageattention OK')
import flash_attn
print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))
"""
    run([str(py), "-c", code])


def requirements_file(root: Path) -> Path:
    return root / "presets" / "extra_env" / REQ_NAME


def install_requirements(py: Path, root: Path) -> None:
    _print("[5/8] Installing Hunyuan Diffusers runtime requirements")
    req = requirements_file(root)
    if req.exists():
        pip_install(py, ["install", "--upgrade", "-r", str(req)])
    else:
        _print(f"[WARN] requirements file missing: {req}")
        _print("       Installing built-in fallback requirement list.")
        pip_install(py, ["install", "--upgrade", *REQ_FALLBACK])
    # Keep helper CLI tools available too.
    pip_install(py, ["install", "-U", "huggingface_hub[cli]"])


def verify_imports(py: Path) -> None:
    _print("[6/8] Verifying imports used by FrameVision Hunyuan")
    code = r'''
import sys
print('python', sys.executable)
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
import diffusers, huggingface_hub, transformers, accelerate
try:
    import kernels
    raise RuntimeError('kernels package is installed; remove it for HunyuanVideo 1.5')
except ModuleNotFoundError:
    print('kernels not installed OK')
print('diffusers', diffusers.__version__)
print('huggingface_hub', huggingface_hub.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
print('HunyuanVideo15Pipeline OK')
print('HunyuanVideo15ImageToVideoPipeline OK')
sys.exit(0 if torch.cuda.is_available() else 2)
'''.strip()
    run([str(py), "-c", code])


def download_default_model(py: Path, root: Path, model: str) -> None:
    if os.environ.get("SKIP_HUNYUAN15_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}:
        _print("[8/8] Skipping model download (SKIP_HUNYUAN15_MODEL=1)")
        return
    cli = root / "helpers" / "hunyuan15_cli.py"
    if not cli.exists():
        raise RuntimeError(f"Cannot download default model because helper is missing: {cli}")
    _print(f"[8/8] Downloading/reusing default model: {model}")
    run([str(py), str(cli), "download", "--model", str(model)], cwd=root)


def write_manifest(root: Path, payload: dict) -> Path:
    out = root / "logs" / "hunyuan15_conda_install_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameVision HunyuanVideo 1.5 conda installer")
    parser.add_argument("--root", type=str, default=None, help="FrameVision root. Defaults to auto-detect from script location.")
    parser.add_argument("--env-dir", type=str, default=None, help="Override env dir. Default: <root>/environments/.hunyuan15_official")
    parser.add_argument("--conda", type=str, default=None, help="Path to conda/mamba/micromamba executable. Defaults to CONDA_EXE/MAMBA_EXE/MICROMAMBA_EXE/PATH.")
    parser.add_argument("--python-version", type=str, default="3.11", help="Conda Python version. Default: 3.11")
    parser.add_argument("--torch-version", type=str, default="2.6.0", help="PyTorch version. Default: 2.6.0 (required for bundled cu126 Flash/Sage wheels).")
    parser.add_argument("--default-model", type=str, default=DEFAULT_MODEL, help=f"Default model to download. Default: {DEFAULT_MODEL}")
    parser.add_argument("--no-backup-existing", action="store_true", help="Do not move aside a non-conda env folder at the target location; fail instead.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else project_root_from_script(script_path)
    env_dir = Path(args.env_dir).resolve() if args.env_dir else (root / "environments" / ENV_NAME)
    cuda_tag = os.environ.get("PYTORCH_CUDA", "cu126").strip() or "cu126"

    _print("")
    _print("===========================")
    _print(" HunyuanVideo-1.5 Installer")
    _print(" FrameVision conda runtime")
    _print("===========================")
    _print(f"Root: {root}")
    _print(f"Env:  {env_dir}")
    _print(f"Python target: {env_python(env_dir)}")
    _print(f"Torch: {args.torch_version}")
    _print(f"Torch CUDA wheels: {cuda_tag}")
    _print("")

    if not root.exists():
        raise RuntimeError(f"Root folder not found: {root}")

    conda_exe = find_conda(args.conda)
    if conda_exe is None:
        raise RuntimeError(
            "Could not find conda/mamba/micromamba.\n"
            "Install Miniconda/Mambaforge, or pass --conda C:/path/to/conda.exe.\n"
            "This installer intentionally creates a real conda env, not a venv."
        )

    py = create_or_reuse_conda_env(root, env_dir, conda_exe, args.python_version, no_backup_existing=args.no_backup_existing)

    _print("[2/8] Upgrading pip/setuptools/wheel")
    pip_install(py, ["install", "--upgrade", "pip", "setuptools", "wheel"])

    install_torch(py, cuda_tag, args.torch_version)
    verify_cuda_torch(py)
    uninstall_kernels(py)
    install_requirements(py, root)
    verify_imports(py)
    install_attention_addons(py, cuda_tag, args.torch_version)
    download_default_model(py, root, args.default_model)

    manifest = {
        "root": str(root),
        "env_dir": str(env_dir),
        "python": str(py),
        "conda": str(conda_exe),
        "torch_cuda": cuda_tag,
        "torch_version": str(args.torch_version),
        "attention_addons": "triton-windows only; SageAttention/FlashAttention disabled unless HY15_ENABLE_FLASH_SAGE=1",
        "default_model": str(args.default_model),
        "requirements_file": str(requirements_file(root)),
        "layout": "conda-prefix",
        "expected_runtime_python": str(py),
    }
    manifest_path = write_manifest(root, manifest)

    _print("")
    _print("[OK] Hunyuan 1.5 conda environment is ready.")
    _print(f"Python: {py}")
    _print(f"Manifest: {manifest_path}")
    _print("")
    _print("You can test with:")
    _print(f"  {py} -c \"from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline; print('OK')\"")
    _print(f"  {py} {root / 'helpers' / 'hunyuan15_cli.py'} download --model {args.default_model}")
    _print("")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _print("")
        _print(f"[ERROR] {exc}")
        raise
