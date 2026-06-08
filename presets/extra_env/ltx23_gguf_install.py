from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INSTALLER_VERSION = "2026-06-06.3"

# IMPORTANT: deliberately limited. Do not expand this list to every file in the repo.
LTX_REPO_ID = "unsloth/LTX-2.3-GGUF"
GEMMA_GGUF_REPO_ID = "unsloth/gemma-3-12b-it-qat-GGUF"
LTX_OFFICIAL_REPO_ID = "Lightricks/LTX-2.3"

LTX_MAIN_FILE = "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
LTX_SUPPORT_FILES = [
    "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
    "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
    "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
    "README.md",
    "LICENSE",
]

# The Unsloth model card examples use this Gemma 3 12B GGUF encoder + mmproj for LTX GGUF workflows.
GEMMA_TEXT_ENCODER_FILES = [
    "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
    "mmproj-BF16.gguf",
]

# Support model for LTX two-stage / upscale workflows. This is not a LoRA fuse file.
OPTIONAL_LTX_SUPPORT_FILES = [
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
]

# Pinned CUDA runtime stack for this isolated GGUF test environment.
# Order matters: Torch first, then Triton, then Flash Attention, then SageAttention.
TORCH_PACKAGES = [
    "torch==2.8.0+cu128",
    "torchvision==0.23.0+cu128",
    "torchaudio==2.8.0+cu128",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TRITON_PACKAGE = "triton-windows==3.4.0.post21"
FLASH_ATTN_WHEEL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl"
SAGE_ATTN_WHEEL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"

TOOL_REPOS = {
    "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
}

_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parents[2]
ENV_DIR = ROOT / "environments" / ".ltx23gguf"
MODEL_DIR = ROOT / "models" / "ltx23_gguf"
HF_LOCAL_DIR = MODEL_DIR / "unsloth_LTX-2.3-GGUF"
GEMMA_LOCAL_DIR = MODEL_DIR / "text_encoders" / "gemma-3-12b-it-qat-GGUF"
SUPPORT_LOCAL_DIR = MODEL_DIR / "support_models"
REPOS_DIR = MODEL_DIR / "repos"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "ltx23_gguf_install.log"
MANIFEST_PATH = MODEL_DIR / "install_manifest.json"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text = f"[{_now()}] {msg}"
    print(text, flush=True)
    with LOG_PATH.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text + "\n")


def run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, check: bool = True) -> subprocess.CompletedProcess:
    shown = subprocess.list2cmdline([str(x) for x in cmd])
    log(f"RUN: {shown}")
    p = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {shown}")
    return p


def pip_cmd(*args: str) -> List[str]:
    """Build a pip command for the isolated env with noisy PATH warnings suppressed."""
    return [str(env_python()), "-m", "pip", *args, "--no-warn-script-location"]


def verify_torch_stack() -> None:
    """Fail loudly if any package changed the pinned CUDA Torch stack."""
    py = env_python()
    code = (
        "import sys\n"
        "import torch, torchvision, torchaudio\n"
        "expected = {\n"
        "    'torch': '2.8.0+cu128',\n"
        "    'torchvision': '0.23.0+cu128',\n"
        "    'torchaudio': '2.8.0+cu128',\n"
        "}\n"
        "actual = {\n"
        "    'torch': torch.__version__,\n"
        "    'torchvision': torchvision.__version__,\n"
        "    'torchaudio': torchaudio.__version__,\n"
        "}\n"
        "print('[torch-check]', actual)\n"
        "bad = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}\n"
        "if bad:\n"
        "    raise SystemExit('Pinned Torch stack mismatch: ' + repr(bad))\n"
        "print('[torch-check] pinned CUDA 12.8 / Torch 2.8 stack OK')\n"
    )
    run([str(py), "-c", code])


def find_conda() -> str:
    candidates: List[str] = []
    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe:
        candidates.append(conda_exe)
    found = shutil.which("conda")
    if found:
        candidates.append(found)
    user = os.environ.get("USERPROFILE", "")
    for base in [
        Path(user) / "miniconda3",
        Path(user) / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
    ]:
        candidates.extend([str(base / "Scripts" / "conda.exe"), str(base / "condabin" / "conda.bat")])
    seen = set()
    for c in candidates:
        if not c:
            continue
        p = str(Path(c))
        key = os.path.normcase(os.path.normpath(p))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(p):
            return p
    return ""


def env_python() -> Path:
    if os.name == "nt":
        return ENV_DIR / "python.exe"
    return ENV_DIR / "bin" / "python"


def ensure_conda_env() -> None:
    py = env_python()
    if py.is_file():
        log(f"Environment already exists: {ENV_DIR}")
        return
    conda = find_conda()
    if not conda:
        raise RuntimeError(
            "Conda was not found. Install Miniconda/Anaconda or make conda available on PATH. "
            f"Expected env location: {ENV_DIR}"
        )
    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"Creating conda env: {ENV_DIR}")
    run([conda, "create", "-y", "-p", str(ENV_DIR), "python=3.11", "pip"])
    if not py.is_file():
        raise RuntimeError(f"Conda env was created but python.exe was not found: {py}")


def pip_install_base(skip_torch: bool = False, skip_attention: bool = False) -> None:
    py = env_python()
    run(pip_cmd("install", "--upgrade", "pip", "setuptools", "wheel"))
    run(pip_cmd("install", "--upgrade", "huggingface_hub[hf_xet]", "hf_xet", "safetensors", "psutil", "numpy", "pillow", "tqdm", "requests"))
    if not skip_torch:
        # Pinned CUDA 12.8 / Torch 2.8 stack for repeatable FrameVision installs.
        run(pip_cmd(
            "install", "--force-reinstall",
            *TORCH_PACKAGES,
            "--index-url", TORCH_INDEX_URL,
        ))
        verify_torch_stack()
        if not skip_attention:
            # Triton first: SageAttention needs the compatible Triton/Torch stack to exist.
            run(pip_cmd("install", "--force-reinstall", TRITON_PACKAGE))
            # Flash/Sage wheels are already built for CUDA 12.8 + Torch 2.8.
            # Use --no-deps so pip cannot replace torch 2.8.0+cu128 with a newer incompatible torch.
            run(pip_cmd("install", "--force-reinstall", "--no-deps", FLASH_ATTN_WHEEL))
            verify_torch_stack()
            run(pip_cmd("install", "--force-reinstall", "--no-deps", SAGE_ATTN_WHEEL))
            verify_torch_stack()


def stage_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    GEMMA_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    SUPPORT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    downloads: List[Dict[str, str]] = []

    def download(repo_id: str, filename: str, local_dir: Path, label: str) -> str:
        log(f"Downloading {label}: {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        downloads.append({"repo_id": repo_id, "filename": filename, "local_path": str(Path(path).resolve()), "label": label})
        return path

    download(LTX_REPO_ID, LTX_MAIN_FILE, HF_LOCAL_DIR, "LTX distilled-1.1 Q4_K_M GGUF")
    for fn in LTX_SUPPORT_FILES:
        download(LTX_REPO_ID, fn, HF_LOCAL_DIR, "LTX repo support file")
    for fn in GEMMA_TEXT_ENCODER_FILES:
        download(GEMMA_GGUF_REPO_ID, fn, GEMMA_LOCAL_DIR, "Gemma 3 12B GGUF text encoder")
    if not args.skip_spatial_upscaler:
        for fn in OPTIONAL_LTX_SUPPORT_FILES:
            download(LTX_OFFICIAL_REPO_ID, fn, SUPPORT_LOCAL_DIR, "LTX optional spatial upscaler")

    clone_tool_repos(install_requirements=not args.skip_tool_requirements)

    manifest = {
        "installer_version": INSTALLER_VERSION,
        "created_at": _now(),
        "framevision_root": str(ROOT),
        "environment": str(ENV_DIR),
        "model_root": str(MODEL_DIR),
        "selected_ltx_model": LTX_MAIN_FILE,
        "runtime_stack": {
            "python": "3.11",
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "torchaudio": "2.8.0+cu128",
            "triton_windows": TRITON_PACKAGE,
            "flash_attention_wheel": FLASH_ATTN_WHEEL,
            "sageattention_wheel": SAGE_ATTN_WHEEL,
        },
        "policy": "Only distilled-1.1 Q4_K_M is downloaded. Other LTX GGUF quant variants are intentionally ignored.",
        "downloads": downloads,
        "tool_repos": {name: str((REPOS_DIR / name).resolve()) for name in TOOL_REPOS},
        "notes": [
            "This installer prepares files for a GGUF runtime/probe. It does not wire Planner/UI generation by itself.",
            "No LoRA fuse cache is created by this installer.",
            "If the future runtime dequantizes GGUF to BF16/FP16 at load time, memory can still explode; test with a memory probe first.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote manifest: {MANIFEST_PATH}")


def clone_tool_repos(*, install_requirements: bool) -> None:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if not git:
        log("WARNING: git.exe not found on PATH. Tool repos were not cloned.")
        return
    for name, url in TOOL_REPOS.items():
        dest = REPOS_DIR / name
        if (dest / ".git").exists():
            log(f"Updating tool repo: {name}")
            run([git, "pull", "--ff-only"], cwd=dest, check=False)
        elif dest.exists() and any(dest.iterdir()):
            log(f"WARNING: tool repo folder exists but is not a git repo, leaving unchanged: {dest}")
        else:
            log(f"Cloning tool repo: {name}")
            run([git, "clone", "--depth", "1", url, str(dest)])
        if install_requirements:
            req = dest / "requirements.txt"
            if req.is_file():
                log(f"Installing requirements for {name}")
                run(pip_cmd("install", "-r", str(req)), check=False)


def write_model_index() -> None:
    index = {
        "name": "LTX 2.3 GGUF distilled-1.1 Q4_K_M",
        "type": "ltx23_gguf",
        "env_path": str(ENV_DIR),
        "model_root": str(MODEL_DIR),
        "ltx_gguf": str((HF_LOCAL_DIR / LTX_MAIN_FILE).resolve()),
        "text_encoder_gemma_gguf": str((GEMMA_LOCAL_DIR / "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf").resolve()),
        "text_encoder_mmproj": str((GEMMA_LOCAL_DIR / "mmproj-BF16.gguf").resolve()),
        "embeddings_connector": str((HF_LOCAL_DIR / "text_encoders" / "ltx-2.3-22b-distilled_embeddings_connectors.safetensors").resolve()),
        "audio_vae": str((HF_LOCAL_DIR / "vae" / "ltx-2.3-22b-distilled_audio_vae.safetensors").resolve()),
        "video_vae": str((HF_LOCAL_DIR / "vae" / "ltx-2.3-22b-distilled_video_vae.safetensors").resolve()),
        "spatial_upscaler": str((SUPPORT_LOCAL_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors").resolve()),
        "runtime_stack": {
            "python": "3.11",
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "torchaudio": "2.8.0+cu128",
            "triton_windows": TRITON_PACKAGE,
            "flash_attention_wheel": FLASH_ATTN_WHEEL,
            "sageattention_wheel": SAGE_ATTN_WHEEL,
        },
        "repos": {name: str((REPOS_DIR / name).resolve()) for name in TOOL_REPOS},
    }
    path = MODEL_DIR / "framevision_ltx23_gguf_index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote FrameVision index: {path}")


def relaunch_inside_env(args: argparse.Namespace) -> int:
    py = env_python()
    cmd = [str(py), str(_THIS_FILE), "--stage", "download"]
    if args.skip_torch:
        cmd.append("--skip-torch")
    if args.skip_spatial_upscaler:
        cmd.append("--skip-spatial-upscaler")
    if args.skip_tool_requirements:
        cmd.append("--skip-tool-requirements")
    if args.skip_attention:
        cmd.append("--skip-attention")
    return run(cmd, check=False).returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrameVision LTX 2.3 GGUF one-click installer")
    p.add_argument("--stage", choices=["all", "download"], default="all")
    p.add_argument("--skip-torch", action="store_true", help="Do not install isolated CUDA torch wheels into the GGUF env.")
    p.add_argument("--skip-spatial-upscaler", action="store_true", help="Do not download the optional LTX spatial upscaler support model.")
    p.add_argument("--skip-tool-requirements", action="store_true", help="Clone tool repos but do not pip-install their requirements.")
    p.add_argument("--skip-attention", action="store_true", help="Install pinned Torch but skip Triton/Flash/Sage attention wheels.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log(f"FrameVision LTX GGUF installer v{INSTALLER_VERSION}")
    log(f"Root: {ROOT}")
    log(f"Env: {ENV_DIR}")
    log(f"Models: {MODEL_DIR}")
    log(f"Selected LTX file only: {LTX_MAIN_FILE}")
    try:
        if args.stage == "all":
            ensure_conda_env()
            pip_install_base(skip_torch=args.skip_torch, skip_attention=args.skip_attention)
            rc = relaunch_inside_env(args)
            if rc != 0:
                return rc
            write_model_index()
            log("Installer completed successfully.")
            return 0
        if args.stage == "download":
            stage_download(args)
            write_model_index()
            return 0
        raise RuntimeError(f"Unknown stage: {args.stage}")
    except Exception as exc:
        log("ERROR: " + str(exc))
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
