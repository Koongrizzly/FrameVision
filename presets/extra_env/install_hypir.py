from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def log(msg: str) -> None:
    print(f"[HYPIR INSTALL] {msg}", flush=True)


def retry(fn, tries=4, delay=3):
    last = None
    for n in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if n == tries:
                raise
            log(f"Attempt {n}/{tries} failed: {exc}")
            time.sleep(delay * n)
    raise last  # pragma: no cover


def download_repo(repo_dir: Path, cache_dir: Path) -> None:
    marker = repo_dir / "HYPIR" / "enhancer" / "sd2.py"
    if marker.exists():
        log(f"Repository already present: {repo_dir}")
        return

    src_root = repo_dir.parent
    src_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / "HYPIR-main.zip"
    url = "https://github.com/XPixelGroup/HYPIR/archive/refs/heads/main.zip"

    log("Downloading official XPixelGroup/HYPIR repository...")
    retry(lambda: urllib.request.urlretrieve(url, zpath))

    unpack = cache_dir / "repo_unpack"
    if unpack.exists():
        shutil.rmtree(unpack, ignore_errors=True)
    unpack.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(unpack)
    extracted = unpack / "HYPIR-main"
    if not extracted.exists():
        raise RuntimeError("Downloaded HYPIR archive did not contain HYPIR-main.")
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    shutil.move(str(extracted), str(repo_dir))
    shutil.rmtree(unpack, ignore_errors=True)
    try:
        zpath.unlink()
    except OSError:
        pass
    log(f"Repository installed: {repo_dir}")


def main() -> int:
    extra_env = Path(__file__).resolve().parent
    root = extra_env.parent.parent
    models = root / "models" / "hypir"
    repo_dir = models / "HYPIR"
    dl_cache = extra_env / "hypir_installer_cache"
    models.mkdir(parents=True, exist_ok=True)

    log(f"FrameVision root: {root}")
    log(f"Installer/support files: {extra_env}")
    log(f"HYPIR repository: {repo_dir}")
    log(f"Models: {models}")

    download_repo(repo_dir, dl_cache)

    weight = models / "HYPIR_sd2.pth"
    if weight.exists() and weight.stat().st_size > 100_000_000:
        log(f"HYPIR weights already present: {weight}")
    else:
        log("Downloading HYPIR_sd2.pth...")
        retry(lambda: hf_hub_download(
            repo_id="lxq007/HYPIR",
            filename="HYPIR_sd2.pth",
            local_dir=str(models),
        ))
        if not weight.exists():
            raise RuntimeError(f"HYPIR weights were not created at {weight}")

    sd2_dir = models / "stable-diffusion-2-1-base"
    model_index = sd2_dir / "model_index.json"
    unet_dir = sd2_dir / "unet"
    if model_index.exists() and unet_dir.exists():
        log(f"Stable Diffusion 2.1 base already present: {sd2_dir}")
    else:
        log("Downloading local Stable Diffusion 2.1 base model (parallel file downloads)...")
        retry(lambda: snapshot_download(
            repo_id="sd2-community/stable-diffusion-2-1-base",
            local_dir=str(sd2_dir),
            max_workers=8,
        ), tries=4, delay=5)
        if not model_index.exists() or not unet_dir.exists():
            raise RuntimeError(f"Incomplete Stable Diffusion 2.1 download: {sd2_dir}")

    # Keep support/cache data under presets/extra_env, but remove temporary archive material.
    try:
        if dl_cache.exists() and not any(dl_cache.iterdir()):
            dl_cache.rmdir()
    except OSError:
        pass

    log("All HYPIR runtime files are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
