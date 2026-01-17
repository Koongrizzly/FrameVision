"""
Qwen-Image-Edit-2511 (GGUF) downloader for FrameVision / sd-cli.exe (stable-diffusion.cpp).

- Downloads selected UNet GGUF variants (excluding the huge BF16/F16).
- Ensures shared required files exist only once:
    * Qwen2.5-VL text encoder GGUF
    * mmproj (vision tower) GGUF
    * qwen_image_vae.safetensors
- Installs into: <project_root>/models/qwen2511gguf/
  with subfolders: unet/, text_encoders/, vae/, input/

Designed to be usable:
1) Standalone: python qwen2511_download.py --variants Q4_K_M,Q5_K_S
2) Imported by app: call list_options(), ensure_download(...)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# Optional override when the script is run from outside presets/extra_env.
_ROOT_OVERRIDE: Optional[Path] = None

# ----------------------------
# Model sources (public)
# ----------------------------

UNET_REPO = "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/"
TEXT_ENCODER_REPO = "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/"
VAE_REPO = "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/"

# Required shared files (download once)
REQUIRED_TEXT_ENCODER = "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
REQUIRED_MMPROJ_SRC = "mmproj-BF16.gguf"
REQUIRED_MMPROJ_DST = "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
REQUIRED_VAE = "qwen_image_vae.safetensors"

# UNet variants (exclude BF16/F16 per request)
UNET_FILES: Dict[str, str] = {
    "Q2_K":   "qwen-image-edit-2511-Q2_K.gguf",
    "Q3_K_S": "qwen-image-edit-2511-Q3_K_S.gguf",
    "Q3_K_M": "qwen-image-edit-2511-Q3_K_M.gguf",
    "Q3_K_L": "qwen-image-edit-2511-Q3_K_L.gguf",
    "Q4_0":   "qwen-image-edit-2511-Q4_0.gguf",
    "Q4_1":   "qwen-image-edit-2511-Q4_1.gguf",
    "Q4_K_S": "qwen-image-edit-2511-Q4_K_S.gguf",
    "Q4_K_M": "qwen-image-edit-2511-Q4_K_M.gguf",
    "Q5_0":   "qwen-image-edit-2511-Q5_0.gguf",
    "Q5_1":   "qwen-image-edit-2511-Q5_1.gguf",
    "Q5_K_S": "qwen-image-edit-2511-Q5_K_S.gguf",
    "Q5_K_M": "qwen-image-edit-2511-Q5_K_M.gguf",
    "Q6_K":   "qwen-image-edit-2511-Q6_K.gguf",
    "Q8_0":   "qwen-image-edit-2511-Q8_0.gguf",
}

@dataclass(frozen=True)
class DownloadItem:
    kind: str                 # "unet" | "text_encoder" | "mmproj" | "vae"
    name: str                 # filename to save as
    url: str                  # download url
    subdir: str               # relative dir under models/qwen2511gguf/

ProgressCb = Callable[[str, int, Optional[int]], None]
# callback signature: (label, bytes_done, bytes_total_or_None)


def _project_root() -> Path:
    """Return the FrameVision project root.

    Default expectation:
        presets/extra_env/qwen2511_download.py -> <root>/presets/extra_env
        parents[0]=extra_env, [1]=presets, [2]=root

    If the script is copied elsewhere, pass --root <root> to override.
    """
    global _ROOT_OVERRIDE
    if _ROOT_OVERRIDE is not None:
        return _ROOT_OVERRIDE
    return Path(__file__).resolve().parents[2]



def _shared_sdcli_dir(root: Path) -> Path:
    return (root / "presets" / "bin").resolve()


def _sdcli_present(bin_dir: Path) -> bool:
    return (bin_dir / "sd-cli.exe").exists() and ((bin_dir / "stable-diffusion.dll").exists() or (bin_dir / "diffusers.dll").exists())


def _models_root() -> Path:
    return _project_root() / "models" / "qwen2511gguf"


def list_options() -> List[str]:
    """Return available UNet variant keys."""
    return list(UNET_FILES.keys())


def build_download_plan(variants: Sequence[str]) -> List[DownloadItem]:
    variants_norm = []
    for v in variants:
        vv = v.strip()
        if not vv:
            continue
        if vv not in UNET_FILES:
            raise ValueError(f"Unknown variant '{vv}'. Options: {', '.join(list_options())}")
        variants_norm.append(vv)

    plan: List[DownloadItem] = []

    # Shared required components
    plan.append(DownloadItem(
        kind="text_encoder",
        name=REQUIRED_TEXT_ENCODER,
        url=TEXT_ENCODER_REPO + REQUIRED_TEXT_ENCODER,
        subdir="text_encoders",
    ))
    plan.append(DownloadItem(
        kind="mmproj",
        name=REQUIRED_MMPROJ_DST,
        url=TEXT_ENCODER_REPO + REQUIRED_MMPROJ_SRC,
        subdir="text_encoders",
    ))
    plan.append(DownloadItem(
        kind="vae",
        name=REQUIRED_VAE,
        url=VAE_REPO + REQUIRED_VAE,
        subdir="vae",
    ))

    # Selected UNets
    for v in variants_norm:
        fn = UNET_FILES[v]
        plan.append(DownloadItem(
            kind="unet",
            name=fn,
            url=UNET_REPO + fn,
            subdir="unet",
        ))
    return plan


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _http_head(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            if cl is None:
                return None
            return int(cl)
    except Exception:
        return None


def _download_with_resume(url: str, dest: Path, label: str, progress: Optional[ProgressCb] = None, max_retries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    total = _http_head(url)
    # If already complete, skip
    if dest.exists():
        if total is None:
            # Can't verify; trust existing file
            if progress:
                progress(f"{label} (exists)", 0, total)
            return
        if dest.stat().st_size == total:
            if progress:
                progress(f"{label} (exists)", total, total)
            return

    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                # If server doesn't honor Range, restart
                if downloaded > 0 and resp.status == 200:
                    downloaded = 0
                    headers.pop("Range", None)
                    if tmp.exists():
                        tmp.unlink()

                mode = "ab" if downloaded > 0 else "wb"
                with tmp.open(mode) as f:
                    done = downloaded
                    if progress:
                        progress(label, done, total)
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(label, done, total)

            # Finalize
            tmp.replace(dest)
            return

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed downloading {label} after {max_retries} attempts: {e}") from e
            time.sleep(min(10, 1.5 ** attempt))


def ensure_download(
    variants: Sequence[str],
    models_dir: Optional[Path] = None,
    progress: Optional[ProgressCb] = None,
) -> List[Path]:
    """
    Download variants + required files.
    Returns list of downloaded/verified file paths.
    """
    if models_dir is None:
        models_dir = _models_root()

    plan = build_download_plan(variants)
    saved: List[Path] = []

    # Create folder structure
    (models_dir / "unet").mkdir(parents=True, exist_ok=True)
    (models_dir / "text_encoders").mkdir(parents=True, exist_ok=True)
    (models_dir / "vae").mkdir(parents=True, exist_ok=True)
    (models_dir / "input").mkdir(parents=True, exist_ok=True)

    for item in plan:
        out_path = models_dir / item.subdir / item.name
        _download_with_resume(item.url, out_path, f"{item.kind}: {item.name}", progress=progress)
        saved.append(out_path)

    return saved


def _cli_progress(label: str, done: int, total: Optional[int]) -> None:
    if total:
        pct = (done / total) * 100.0 if total else 0.0
        sys.stdout.write(f"\r{label}  {done/1e9:.2f} / {total/1e9:.2f} GB  ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r{label}  {done/1e9:.2f} GB")
    sys.stdout.flush()
    if total and done >= total:
        sys.stdout.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download Qwen-Image-Edit-2511 GGUF models + required components.")
    p.add_argument("--variants", type=str, default="Q4_K_M",
                   help="Comma-separated UNet variants to download, e.g. Q4_K_M,Q5_K_S (default: Q4_K_M).")
    p.add_argument("--list", action="store_true", help="List available variants and exit.")
    p.add_argument("--models-dir", type=str, default="", help="Override destination folder.")
    p.add_argument("--root", type=str, default="", help="Override FrameVision project root (useful if script is relocated).")
    p.add_argument("--ensure-cli", action="store_true", help="Check shared sd-cli/dlls exist in <root>/presets/bin (no download).")
    p.add_argument("--bin-dir", type=str, default="", help="Override shared bin dir for sd-cli check.")
    args = p.parse_args(list(argv) if argv is not None else None)

    global _ROOT_OVERRIDE
    if args.root:
        _ROOT_OVERRIDE = Path(args.root).resolve()

    if args.list:
        for k in list_options():
            print(k)
        return 0

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None

    print("Destination:", models_dir if models_dir else _models_root())
    print("Variants:", ", ".join(variants))
    print()

    if args.ensure_cli:
        root = _project_root()
        bin_dir = Path(args.bin_dir).resolve() if args.bin_dir else _shared_sdcli_dir(root)
        if _sdcli_present(bin_dir):
            print(f"[QWEN2511] OK  sd-cli present: {bin_dir / 'sd-cli.exe'}")
        else:
            print(f"[QWEN2511] WARN  sd-cli not found in: {bin_dir}")
            print("[QWEN2511]       (Install it once via Z-Image GGUF installer or Qwen2512 installer; it is shared.)")

    try:
        ensure_download(variants, models_dir=models_dir, progress=_cli_progress)
        print("\nDone.")
        return 0
    except Exception as e:
        print("\nERROR:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
