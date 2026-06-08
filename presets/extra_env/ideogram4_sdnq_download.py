from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ID = "WaveCut/ideogram-4-sdnq-uint4"
MODEL_SUBDIR = Path("models") / "ideogram4" / "sdnq_uint4"

FLASH_ATTN_TORCH28_CU128 = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl"
SAGE_ATTN_TORCH28_CU128 = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"


def framevision_root() -> Path:
    # <root>/presets/extra_env/ideogram4_sdnq_download.py
    return Path(__file__).resolve().parents[2]


def log_path() -> Path:
    path = framevision_root() / "logs" / "ideogram4_sdnq_install.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_cmd(cmd: list[str], allow_fail: bool = False) -> int:
    write_log("RUN: " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(framevision_root()))
    if proc.returncode and not allow_fail:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if proc.returncode:
        write_log(f"WARN: command failed with code {proc.returncode}; continuing")
    return proc.returncode


def run_pip(args: list[str], allow_fail: bool = False) -> int:
    return run_cmd([sys.executable, "-m", "pip", *args, "--no-warn-script-location"], allow_fail=allow_fail)


def torch_version_tuple() -> tuple[int, int]:
    try:
        import torch
        base = torch.__version__.split("+")[0]
        major, minor, *_ = base.split(".")
        return int(major), int(minor)
    except Exception:
        return (0, 0)


def install_attention(profile: str) -> None:
    write_log(f"Installing attention helpers for profile: {profile}")
    run_pip(["install", "--upgrade", "psutil"])

    if profile == "torch28-cu128-attn":
        # PyTorch 2.8 pairs with Triton 3.4. Pin <3.5 so pip does not later pull a Triton meant for newer Torch.
        run_pip(["install", "--upgrade", "triton-windows<3.5"])
        # Use direct Windows wheels. Do NOT use `pip install flash-attn`, because that tries source build on Windows.
        run_pip(["install", "--upgrade", FLASH_ATTN_TORCH28_CU128])
        run_pip(["install", "--upgrade", SAGE_ATTN_TORCH28_CU128])
    elif profile == "torch211-cu128-safe":
        # PyTorch 2.11 pairs with newer Triton. There are no known matching user-provided Flash/Sage wheels here.
        run_pip(["install", "--upgrade", "triton-windows<3.7"])
        write_log("Skipping FlashAttention/SageAttention direct wheels: supplied wheels are for torch 2.8 + cu128 only.")
    else:
        raise SystemExit(f"Unknown attention profile: {profile}")

    verify_attention_imports(soft=True)


def install_runtime() -> None:
    write_log("Installing Ideogram SDNQ runtime packages...")
    run_pip(["install", "--upgrade", "huggingface_hub", "hf_transfer", "hf_xet", "psutil"])

    # Install the normal runtime deps first. Avoid replacing the selected Torch profile.
    run_pip([
        "install", "--upgrade",
        "diffusers", "transformers", "accelerate", "safetensors", "pillow", "numpy",
        "sdnq", "bitsandbytes", "einops", "sentencepiece", "requests",
    ])

    # ideogram4 currently declares torch>=2.11. For the torch2.8 attention profile, install it without deps
    # so pip does not silently upgrade Torch and break the Flash/Sage wheels.
    tv = torch_version_tuple()
    if tv and tv < (2, 11):
        write_log(f"Torch {tv[0]}.{tv[1]} detected; installing ideogram4 with --no-deps to preserve torch2.8 attention wheels.")
        run_pip(["install", "--upgrade", "--no-deps", "git+https://github.com/ideogram-oss/ideogram4"])
    else:
        run_pip(["install", "--upgrade", "git+https://github.com/ideogram-oss/ideogram4"])


def cuda_test() -> None:
    write_log("Running CUDA self-test...")
    import torch

    write_log(f"Python: {sys.version.split()[0]}")
    write_log(f"Torch: {torch.__version__}")
    write_log(f"Torch CUDA build: {torch.version.cuda}")
    write_log(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CPU fallback detected. Aborting install.")
    write_log(f"GPU: {torch.cuda.get_device_name(0)}")


def download_model(force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    root = framevision_root()
    model_dir = root / MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    required = [
        model_dir / "model_index.json",
        model_dir / "ideogram4_sdnq_pipeline.py",
        model_dir / "quantization_manifest.json",
    ]
    if not force and all(p.exists() for p in required):
        write_log(f"Model repo already present: {model_dir}")
    else:
        write_log(f"Downloading {REPO_ID} into {model_dir}")
        write_log("Repo size shown on Hugging Face is about 14.8 GB. Existing files should be reused.")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=[".git*", "*.md.tmp", "__pycache__/*"],
        )

    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Model download incomplete. Missing: " + ", ".join(missing))

    settings_dir = root / "presets" / "setsave"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "repo_id": REPO_ID,
        "model_dir": str(model_dir),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    (settings_dir / "ideogram4_sdnq_location.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_log(f"Model ready: {model_dir}")
    return model_dir


def verify_attention_imports(soft: bool = False) -> None:
    write_log("Verifying attention imports...")
    checks = ["triton", "flash_attn", "sageattention"]
    for name in checks:
        try:
            __import__(name)
            write_log(f"Import OK: {name}")
        except Exception as exc:
            msg = f"Import failed: {name}: {exc}"
            if soft:
                write_log("WARN: " + msg)
            else:
                raise RuntimeError(msg) from exc


def verify_imports() -> None:
    write_log("Verifying runtime imports...")
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    import accelerate  # noqa: F401
    import safetensors  # noqa: F401
    import sdnq  # noqa: F401
    import ideogram4  # noqa: F401
    write_log("Runtime imports OK.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download/install WaveCut Ideogram 4 SDNQ UInt4 for FrameVision")
    p.add_argument("--install-runtime", action="store_true", help="Install Python runtime packages in the current env")
    p.add_argument("--install-attention", action="store_true", help="Install Triton/Flash/Sage for the selected profile")
    p.add_argument("--profile", default="torch28-cu128-attn", help="torch28-cu128-attn or torch211-cu128-safe")
    p.add_argument("--download-model", action="store_true", help="Download the model repo to models/ideogram4/sdnq_uint4")
    p.add_argument("--force", action="store_true", help="Force snapshot_download even if required files exist")
    p.add_argument("--cuda-test", action="store_true", help="Refuse CPU fallback")
    p.add_argument("--verify-imports", action="store_true", help="Verify runtime imports")
    p.add_argument("--verify-attention", action="store_true", help="Verify attention imports")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        write_log("Ideogram 4 SDNQ installer helper started.")
        if args.cuda_test:
            cuda_test()
        if args.install_attention:
            install_attention(args.profile)
        if args.install_runtime:
            install_runtime()
        if args.verify_imports or args.install_runtime:
            verify_imports()
        if args.verify_attention:
            verify_attention_imports(soft=False)
        if args.download_model:
            download_model(force=args.force)
        write_log("Ideogram 4 SDNQ installer helper finished.")
        return 0
    except subprocess.CalledProcessError as exc:
        write_log(f"ERROR: command failed with code {exc.returncode}")
        return exc.returncode or 1
    except Exception as exc:
        write_log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
