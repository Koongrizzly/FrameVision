#!/usr/bin/env python3
# Boogu_download.py
#
# FrameVision helper downloader for Boogu Image.
# Place this file in:
#   <FrameVision root>\presets\extra_env\Boogu_download.py
#
# It downloads:
#   - latest stable-diffusion.cpp Windows CUDA 12 sd-cli release
#   - selected Boogu Image diffusion model(s)
#   - FLUX VAE
#   - Qwen3-VL 8B GGUF text encoder
#   - Qwen3-VL mmproj file when edit is selected
#
# Downloader output is intentionally neutral and direct.

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
SDCPP_RELEASES_PAGE = "https://github.com/leejet/stable-diffusion.cpp/releases"

BOOGU_REPO = "Comfy-Org/Boogu-Image"
QWEN_REPO = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
FLUX_VAE_REPO = "Comfy-Org/Boogu-Image"

# Default folder layout for FrameVision.
# Keep file names unchanged so helpers can detect them by their original names.
BOOGU_MODEL_DIR = Path("models") / "boogu_image" / "diffusion_models"
BOOGU_VAE_DIR = Path("models") / "boogu_image" / "vae"
BOOGU_LLM_DIR = Path("models") / "boogu_image" / "llm"
BIN_DIR = Path("presets") / "bin"

# User-facing choices.
PRECISION_CHOICES = {
    "1": ("fp8", "FP8 scaled"),
    "2": ("fp16", "FP16 / BF16"),
}

MODEL_CHOICES = {
    "1": "turbo",
    "2": "edit",
    "3": "both",
}

# For FP16, the turbo hotfix is preferred because the upstream repo currently provides it.
DIFFUSION_FILES = {
    ("turbo", "fp8"): "diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
    ("turbo", "fp16"): "diffusion_models/boogu_image_turbo_hotfix_bf16.safetensors",
    ("edit", "fp8"): "diffusion_models/boogu_image_edit_fp8_scaled.safetensors",
    ("edit", "fp16"): "diffusion_models/boogu_image_edit_bf16.safetensors",
}

EXTRA_FILES = {
    "vae": (
        FLUX_VAE_REPO,
        "vae/flux1_vae_bf16.safetensors",
        BOOGU_VAE_DIR / "ae.safetensors",
    ),
    "qwen_llm": (
        QWEN_REPO,
        "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        BOOGU_LLM_DIR / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
    ),
    "qwen_mmproj": (
        QWEN_REPO,
        "mmproj-BF16.gguf",
        BOOGU_LLM_DIR / "mmproj-BF16.gguf",
    ),
}


def app_root_from_script() -> Path:
    """
    Expected script location:
      <FrameVision root>/presets/extra_env/Boogu_download.py

    If run from somewhere else, still try to resolve a sensible root.
    """
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "extra_env" and here.parent.parent.name.lower() == "presets":
        return here.parent.parent.parent
    return Path.cwd().resolve()


APP_ROOT = app_root_from_script()


def log(message: str) -> None:
    print(f"[Boogu] {message}", flush=True)


def ask_choice(title: str, choices: Dict[str, object]) -> str:
    print()
    print(title)
    for key, value in choices.items():
        if isinstance(value, tuple):
            label = value[1]
        else:
            label = str(value).title()
        print(f"  {key}) {label}")

    while True:
        answer = input("Select option: ").strip().lower()
        if answer in choices:
            return answer
        print("Invalid selection.")


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} ({suffix}): ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes or num_bytes < 0:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def request_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "FrameVision-Boogu-Downloader",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def request_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FrameVision-Boogu-Downloader"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def find_latest_sd_cli_asset() -> Tuple[str, str]:
    """
    Returns (asset_name, download_url) for the latest:
      sd-master-XXXXXX-bin-win-cuda12-x64.zip
    """
    pattern = re.compile(r"^sd-master-.+?-bin-win-cuda12-x64\.zip$", re.IGNORECASE)

    # Primary: GitHub releases API.
    try:
        data = request_json(GITHUB_LATEST_RELEASE_API)
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if pattern.match(name):
                url = asset.get("browser_download_url")
                if url:
                    return name, url
    except Exception as exc:
        log(f"GitHub API lookup did not complete: {exc}")

    # Fallback: releases page scrape.
    page = request_text(SDCPP_RELEASES_PAGE)
    names = pattern.findall(page)
    if names:
        name = names[0]
        tag_match = re.search(r"/leejet/stable-diffusion\.cpp/releases/tag/([^\"/]+)", page)
        if tag_match:
            tag = tag_match.group(1)
        else:
            # The filename contains the commit, but not the release counter.
            # Fall back to latest/download path, which GitHub supports.
            tag = "latest/download"
        return name, f"https://github.com/leejet/stable-diffusion.cpp/releases/{tag}/{name}"

    raise RuntimeError("Could not find a Windows CUDA 12 sd-cli release asset.")


def hf_resolve_url(repo: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path_in_repo}?download=true"


def download_file(url: str, target: Path, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "FrameVision-Boogu-Downloader"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    req = urllib.request.Request(url, headers=headers)

    log(f"Downloading {label}")
    if resume_from:
        log(f"Resuming from {human_size(resume_from)}")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None

            if response.status == 206 and total is not None:
                expected_total = resume_from + total
            else:
                # Server did not honor Range; restart the partial file.
                expected_total = total
                resume_from = 0

            mode = "ab" if response.status == 206 and resume_from else "wb"
            downloaded = resume_from
            started = time.time()
            last_print = 0.0

            with open(tmp, mode) as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print >= 1.0:
                        last_print = now
                        elapsed = max(now - started, 0.1)
                        speed = (downloaded - resume_from) / elapsed
                        if expected_total:
                            pct = min(downloaded / expected_total * 100.0, 100.0)
                            print(
                                f"  {pct:6.2f}%  {human_size(downloaded)} / {human_size(expected_total)}  "
                                f"{human_size(int(speed))}/s",
                                end="\r",
                                flush=True,
                            )
                        else:
                            print(
                                f"  {human_size(downloaded)}  {human_size(int(speed))}/s",
                                end="\r",
                                flush=True,
                            )
            print(" " * 100, end="\r", flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and tmp.exists():
            # Range not satisfiable can mean the partial is already complete.
            tmp.replace(target)
            log(f"Kept existing complete file: {target.name}")
            return
        raise

    tmp.replace(target)
    log(f"Saved: {target}")


def should_skip(target: Path) -> bool:
    return target.exists() and target.stat().st_size > 1024 * 1024


def download_if_missing(url: str, target: Path, label: str) -> None:
    abs_target = APP_ROOT / target
    if should_skip(abs_target):
        log(f"Already present: {abs_target}")
        return
    download_file(url, abs_target, label)


def unpack_sd_cli(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sdcpp_unpack_") as tmp_name:
        tmp_dir = Path(tmp_name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Copy everything to presets/bin while flattening the common release layout.
        # If the archive already has files at root, this still works.
        files = [p for p in tmp_dir.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError("Downloaded sd-cli archive did not contain files.")

        copied = 0

        # Prefer the folder that contains sd-cli.exe as the copy root.
        sd_cli_candidates = [p for p in files if p.name.lower() == "sd-cli.exe"]
        if sd_cli_candidates:
            copy_root = sd_cli_candidates[0].parent
            for src in copy_root.iterdir():
                if src.is_file():
                    shutil.copy2(src, target_dir / src.name)
                    copied += 1
                elif src.is_dir():
                    dst = target_dir / src.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    copied += 1
        else:
            # Fallback: preserve relative paths from extracted archive.
            common_root = tmp_dir
            top_dirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
            top_files = [p for p in tmp_dir.iterdir() if p.is_file()]
            if len(top_dirs) == 1 and not top_files:
                common_root = top_dirs[0]

            for src in common_root.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(common_root)
                    dst = target_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1

        if copied == 0:
            raise RuntimeError("No sd-cli files were copied.")

    sd_cli = target_dir / "sd-cli.exe"
    if sd_cli.exists():
        log(f"sd-cli available: {sd_cli}")
    else:
        log("sd-cli archive unpacked, but sd-cli.exe was not found at presets/bin root.")


def install_latest_sd_cli() -> None:
    bin_dir = APP_ROOT / BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    name, url = find_latest_sd_cli_asset()
    zip_target = bin_dir / name

    download_file(url, zip_target, name)
    log("Unpacking sd-cli")
    unpack_sd_cli(zip_target, bin_dir)

    try:
        zip_target.unlink()
        log("Removed downloaded sd-cli zip")
    except OSError as exc:
        log(f"Could not remove zip: {exc}")


def selected_model_names(selection: str) -> List[str]:
    chosen = MODEL_CHOICES[selection]
    if chosen == "both":
        return ["turbo", "edit"]
    return [chosen]


def planned_downloads(precision: str, model_names: List[str]) -> List[Tuple[str, str, Path, str]]:
    """
    Returns list of:
      (repo, path_in_repo, relative_target, label)
    """
    downloads: List[Tuple[str, str, Path, str]] = []

    for model_name in model_names:
        model_file = DIFFUSION_FILES[(model_name, precision)]
        target = BOOGU_MODEL_DIR / Path(model_file).name
        downloads.append((BOOGU_REPO, model_file, target, f"Boogu Image {model_name} {precision}"))

    vae_repo, vae_path, vae_target = EXTRA_FILES["vae"]
    downloads.append((vae_repo, vae_path, vae_target, "FLUX VAE"))

    qwen_repo, qwen_path, qwen_target = EXTRA_FILES["qwen_llm"]
    downloads.append((qwen_repo, qwen_path, qwen_target, "Qwen3-VL 8B text encoder"))

    if "edit" in model_names:
        mm_repo, mm_path, mm_target = EXTRA_FILES["qwen_mmproj"]
        downloads.append((mm_repo, mm_path, mm_target, "Qwen3-VL vision projector"))

    return downloads


def write_manifest(precision: str, model_names: List[str], files: Iterable[Path]) -> None:
    manifest = {
        "name": "Boogu Image",
        "precision": precision,
        "models": model_names,
        "paths": {
            "sd_cli": str((APP_ROOT / BIN_DIR / "sd-cli.exe").resolve()),
            "diffusion_models": str((APP_ROOT / BOOGU_MODEL_DIR).resolve()),
            "vae": str((APP_ROOT / BOOGU_VAE_DIR).resolve()),
            "llm": str((APP_ROOT / BOOGU_LLM_DIR).resolve()),
        },
        "files": [str((APP_ROOT / p).resolve()) for p in files],
        "notes": [
            "Edit mode uses mmproj-BF16.gguf with --llm_vision.",
            "The selected diffusion files keep their upstream filenames.",
        ],
    }
    manifest_path = APP_ROOT / "models" / "boogu_image" / "boogu_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {manifest_path}")


def main() -> int:
    print()
    print("Boogu Image downloader")
    print("======================")
    print(f"App root: {APP_ROOT}")
    print()

    if not ask_yes_no("Download or update sd-cli first", True):
        log("Skipping sd-cli download")
    else:
        install_latest_sd_cli()

    precision_key = ask_choice("Choose model precision", PRECISION_CHOICES)
    precision = PRECISION_CHOICES[precision_key][0]

    model_key = ask_choice("Choose Boogu Image model", MODEL_CHOICES)
    models = selected_model_names(model_key)

    downloads = planned_downloads(precision, models)

    print()
    print("Download plan")
    for repo, path_in_repo, target, label in downloads:
        print(f"  - {label}: {target}")
    print()

    if not ask_yes_no("Start download", True):
        log("Cancelled by user")
        return 0

    completed_targets: List[Path] = []
    for repo, path_in_repo, target, label in downloads:
        url = hf_resolve_url(repo, path_in_repo)
        download_if_missing(url, target, label)
        completed_targets.append(target)

    write_manifest(precision, models, completed_targets)

    print()
    log("Done")
    print()
    print("Installed locations:")
    print(f"  sd-cli:           {APP_ROOT / BIN_DIR / 'sd-cli.exe'}")
    print(f"  diffusion models: {APP_ROOT / BOOGU_MODEL_DIR}")
    print(f"  vae:              {APP_ROOT / BOOGU_VAE_DIR}")
    print(f"  text encoder:     {APP_ROOT / BOOGU_LLM_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        log("Interrupted")
        raise SystemExit(130)
    except Exception as exc:
        print()
        log(f"Error: {exc}")
        raise SystemExit(1)
