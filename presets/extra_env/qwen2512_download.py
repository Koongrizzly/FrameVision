#!/usr/bin/env python3
"""
Qwen-Image-2512 (GGUF) downloader for FrameVision.

By default, this downloads the "baseline" Q4_K_M model, plus the companion
Qwen2.5-VL GGUF and the VAE safetensors.

You can also download other quantizations (Q2/Q3/Q5/Q6/Q8) one-by-one via
--qwen-quant, without including the huge FP16/BF16 variants.

Designed to be called from presets/extra_env/qwen2512_install.bat
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Iterable

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    print("[QWEN2512] ERROR: requests is required. Please install it in your venv.", file=sys.stderr)
    raise

# tqdm is optional
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# --- Qwen-Image-2512 GGUF quantizations (NO FP16/BF16) ---
# We expose ONE selectable file per Q-level, matching your request:
#   q2, q3, q4, q5, q6, q8
QWEN_IMAGE_QUANTS = {
    "q2": (
        "qwen-image-2512-Q2_K.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q2_K.gguf",
    ),
    "q3": (
        "qwen-image-2512-Q3_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q3_K_M.gguf",
    ),
    "q4": (
        "qwen-image-2512-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf",
    ),
    "q5": (
        "qwen-image-2512-Q5_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q5_K_M.gguf",
    ),
    "q6": (
        "qwen-image-2512-Q6_K.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q6_K.gguf",
    ),
    "q8": (
        "qwen-image-2512-Q8_0.gguf",
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf",
    ),
}

# --- Always-needed companion files ---
BASE_DOWNLOADS = [
    (
        "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
    ),
    (
        "qwen_image_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    ),
]

GITHUB_LATEST = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"


def _human_gb(n: int) -> str:
    return f"{n / (1024**3):.2f} GB"


def _head_content_length(url: str, timeout: int = 30) -> Optional[int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def download_with_resume(url: str, out_path: Path, *, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = _head_content_length(url)
    existing = out_path.stat().st_size if out_path.exists() else 0

    # If already complete, skip.
    if total is not None and existing == total and total > 0:
        print(f"[QWEN2512] OK  {out_path.name} already downloaded ({_human_gb(total)})")
        return

    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()

        # If server ignored Range, restart.
        if existing > 0 and r.status_code == 200:
            print(f"[QWEN2512] Server did not resume; restarting {out_path.name}")
            existing = 0
            mode = "wb"

        if tqdm is not None:
            bar_total = total if total is not None else None
            pbar = tqdm(
                total=bar_total,
                initial=existing,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=out_path.name,
                leave=True,
            )
        else:
            pbar = None

        with open(out_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))

        if pbar is not None:
            pbar.close()

    # Basic integrity check if we know the size
    if total is not None:
        got = out_path.stat().st_size
        if got != total:
            raise RuntimeError(f"Downloaded size mismatch for {out_path.name}: got {got}, expected {total}")


def _download_file(url: str, out_path: Path) -> None:
    """Backward-compatible alias used by older installer scripts."""
    print(f"[QWEN2512] Downloading: {url}")
    download_with_resume(url, out_path)


def _score_asset(name: str, want: str) -> int:
    """
    Heuristic scorer for stable-diffusion.cpp release assets.

    IMPORTANT: We intentionally *avoid* HIP/ROCm builds because they depend on
    external AMD runtime DLLs (e.g. amdhip64_*.dll) that are often not shipped
    alongside sd-cli.exe, leading to "amdhip64_6.dll was not found" errors.
    """
    n = name.lower()
    score = 0

    # Must be an archive we can extract.
    if not (n.endswith(".zip") or n.endswith(".7z")):
        return -10**9

    # Platform
    if "win" in n or "windows" in n:
        score += 200
    else:
        score -= 2000

    if "x64" in n or "amd64" in n:
        score += 80
    if "arm" in n or "aarch64" in n:
        score -= 5000

    # Avoid debug/minimal/unsupported backends.
    if "debug" in n:
        score -= 5000
    if "hip" in n or "rocm" in n:
        score -= 100000  # hard veto
    if "opencl" in n:
        score -= 5000

    # Backend preference
    if want == "vulkan":
        # Prefer pure Vulkan builds; avoid CUDA-labeled bundles.
        if "vulkan" in n:
            score += 500
        if "cuda" in n:
            score -= 2000
        if "cpu" in n and "vulkan" not in n:
            score -= 200
    elif want == "cuda":
        if "cuda" in n:
            score += 500
        if "vulkan" in n:
            score -= 200
        if "cpu" in n and "cuda" not in n:
            score -= 200
    elif want == "cpu":
        if "cpu" in n:
            score += 400
        if "vulkan" in n or "cuda" in n:
            score -= 200

    # Prefer full bundles over "tiny" ones.
    if "minimal" in n or "lite" in n:
        score -= 2000

    # Mild preference for anything that looks like a packaged binary build.
    if "bin" in n:
        score += 20
    if "cli" in n:
        score += 20

    return score



def _candidate_assets(assets: list[dict], want: str) -> list[dict]:
    """Return assets sorted best→worst for the requested backend."""
    scored: list[tuple[int, dict]] = []
    for a in assets:
        name = str(a.get("name", ""))
        if not name:
            continue
        s = _score_asset(name, want)
        scored.append((s, a))
    scored.sort(key=lambda t: t[0], reverse=True)
    # Filter out hard-veto scores.
    return [a for s, a in scored if s > -10**8]


def ensure_sd_cli(bin_dir: Path, *, backend: str = "vulkan") -> None:
    """
    Install a *known-working* stable-diffusion.cpp Windows bundle into bin_dir.

    We intentionally pin to the same CUDA12 bundle used by the Z-Image installer:
      https://github.com/leejet/stable-diffusion.cpp/releases/download/master-445-860a78e/sd-master-860a78e-bin-win-cuda12-x64.zip

    Reason: "latest" assets frequently change names/layout and may ship "mini" builds.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)

    sdcli = bin_dir / "sd-cli.exe"
    sdexe = bin_dir / "sd.exe"
    diffdll = bin_dir / "diffusers.dll"
    stabledll = bin_dir / "stable-diffusion.dll"

    # If it already exists and looks complete, keep it.
    if sdcli.exists() and (diffdll.exists() or stabledll.exists()) and (sdexe.exists() or sdcli.exists()):
        print(f"[QWEN2512] OK  sd-cli already present: {sdcli}")
        return

    # Download the pinned bundle (cache it under bin_dir).
    bundle_url = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-445-860a78e/sd-master-860a78e-bin-win-cuda12-x64.zip"
    bundle_name = os.path.basename(bundle_url)
    archive = bin_dir / bundle_name

    print("[QWEN2512] Installing stable-diffusion.cpp pinned bundle…")
    print(f"[QWEN2512] URL: {bundle_url}")
    print(f"[QWEN2512] Target: {archive}")

    # Always download if missing or suspiciously small (< 50MB). Prevents bad partial downloads.
    need_dl = True
    if archive.exists():
        try:
            if archive.stat().st_size > 50 * 1024 * 1024:
                need_dl = False
        except Exception:
            need_dl = True

    if need_dl:
        _download_file(bundle_url, archive)
    else:
        gb = archive.stat().st_size / (1024**3)
        print(f"[QWEN2512] OK  {bundle_name} already downloaded ({gb:.2f} GB)")

    # Extract only relevant payload into bin_dir (flat), regardless of subfolders inside the zip.
    import zipfile
    extracted = 0
    with zipfile.ZipFile(archive, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            bn = os.path.basename(info.filename)
            if not bn:
                continue
            low = bn.lower()

            # Keep ONLY the runtime files we care about.
            if low not in ("sd.exe", "sd-cli.exe", "diffusers.dll", "stable-diffusion.dll"):
                continue

            outp = bin_dir / bn
            with z.open(info, "r") as src, open(outp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    if extracted == 0:
        raise RuntimeError(f"Pinned sd bundle extracted 0 files. Unexpected zip layout: {archive.name}")

    # Normalize naming differences.
    if not diffdll.exists() and stabledll.exists():
        # Some builds renamed the dll. FrameVision expects diffusers.dll, so provide it.
        try:
            shutil.copy2(stabledll, diffdll)
            print("[QWEN2512] OK  Created diffusers.dll from stable-diffusion.dll")
        except Exception as e:
            raise RuntimeError(f"Failed to create diffusers.dll from stable-diffusion.dll: {e}")

    # Ensure we have both exes (some builds may ship only one).
    if not sdcli.exists() and sdexe.exists():
        shutil.copy2(sdexe, sdcli)
        print("[QWEN2512] OK  Created sd-cli.exe from sd.exe")
    if not sdexe.exists() and sdcli.exists():
        shutil.copy2(sdcli, sdexe)
        print("[QWEN2512] OK  Created sd.exe from sd-cli.exe")

    # Final validation.
    if not sdcli.exists():
        raise RuntimeError("sd-cli.exe is missing after extraction.")
    if not sdexe.exists():
        raise RuntimeError("sd.exe is missing after extraction.")
    if not diffdll.exists():
        raise RuntimeError("diffusers.dll is missing after extraction (and stable-diffusion.dll was not present).")

    # Guard against mini/partial builds (tiny dll).
    dll_size = diffdll.stat().st_size if diffdll.exists() else 0
    if dll_size < 100 * 1024 * 1024:
        raise RuntimeError(
            f"Installed diffusers.dll looks too small ({dll_size/1024/1024:.1f} MB) — likely a mini/partial build. "
            "Pinned bundle should provide a large CUDA build; if this persists, the download may be corrupted."
        )

    print(f"[QWEN2512] OK  Installed stable-diffusion.cpp files to: {bin_dir}")
    print(f"[QWEN2512] OK  sd-cli: {sdcli}")
    print(f"[QWEN2512] OK  sd.exe:  {sdexe}")
    print(f"[QWEN2512] OK  dll:    {diffdll} ({dll_size/1024/1024:.0f} MB)")

    # Cleanup: delete the downloaded archive after a successful extract + validation.
    # This saves disk space; the installer can always re-download if needed.
    try:
        if archive.exists():
            archive.unlink()
            print(f"[QWEN2512] OK  Deleted downloaded archive: {archive.name}")
    except Exception as e:
        print(f"[QWEN2512] WARN  Could not delete archive {archive.name}: {e}")

def _print_quants() -> None:
    print("[QWEN2512] Available Qwen-Image-2512 quantizations (select with --qwen-quant):")
    for k in ("q2", "q3", "q4", "q5", "q6", "q8"):
        fn, url = QWEN_IMAGE_QUANTS[k]
        print(f"  - {k}: {fn}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FrameVision root path")
    ap.add_argument("--models-dir", required=True, help="Destination folder for GGUF/weights")
    ap.add_argument("--bin-dir", default="", help="Destination folder for sd-cli.exe + DLLs (optional)")
    ap.add_argument("--ensure-cli", action="store_true", help="(Legacy) Explicitly ensure sd-cli into bin-dir")
    ap.add_argument("--no-cli", action="store_true", help="Do not download sd-cli/dll (models only)")
    ap.add_argument(
        "--cli-backend",
        default="vulkan",
        choices=["vulkan", "cuda", "cpu"],
        help="Which sd-cli build to prefer when downloading.",
    )

    ap.add_argument(
        "--qwen-quant",
        action="append",
        choices=["q2", "q3", "q4", "q5", "q6", "q8"],
        help="Download a specific Qwen-Image-2512 GGUF quantization. Can be passed multiple times.",
    )
    ap.add_argument(
        "--list-quants",
        action="store_true",
        help="Print available quantizations and exit.",
    )

    args = ap.parse_args(argv)

    if args.list_quants:
        _print_quants()
        return 0

    # Default behavior remains: download Q4 if user didn't specify anything.
    quants: list[str] = args.qwen_quant or ["q4"]

    root = Path(args.root).resolve()
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QWEN2512] Root:       {root}")
    print(f"[QWEN2512] Models dir: {models_dir}")

    # Download chosen Qwen-Image GGUF quant(s)
    for q in quants:
        fn, url = QWEN_IMAGE_QUANTS[q]
        out = models_dir / fn
        print(f"[QWEN2512] GET {fn} ({q})")
        download_with_resume(url, out)

    # Download always-needed companion files (VL + VAE)
    for fn, url in BASE_DOWNLOADS:
        out = models_dir / fn
        print(f"[QWEN2512] GET {fn}")
        download_with_resume(url, out)
    # Download sd-cli + DLLs
    # NOTE: Optional-installs runner historically called this script without --ensure-cli.
    # We now ensure the CLI by default unless the caller explicitly requests --no-cli.
    if not args.no_cli:
        # Default bin dir: <root>/presets/bin (shared)
        bin_dir = Path(args.bin_dir).resolve() if args.bin_dir else (root / "presets" / "bin")
        print(f"[QWEN2512] Bin dir:   {bin_dir}")
        ensure_sd_cli(bin_dir, backend=args.cli_backend)

    print("[QWEN2512] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
