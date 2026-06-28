#!/usr/bin/env python3
"""
FrameVision tiny whisper.cpp installer - ROOTFIX BUILD

Expected location:
    <FrameVision root>/presets/extra_env/tiny_whisper_rootfix.py

Installs latest Windows x64 whisper.cpp binary into:
    <FrameVision root>/presets/bin/whisper/

Downloads whisper.cpp GGML models into:
    <FrameVision root>/models/faster_whisper/base/ggml-base.en.bin
    <FrameVision root>/models/faster_whisper/tiny/ggml-tiny.en.bin

No external Python packages needed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

APP_NAME = "FrameVision whisper.cpp installer"
FIXED_VERSION = "VERSION THAT IS FIXED - rootfix-2026-06-28-008 - NEW FILE NAME"

GITHUB_API_LATEST = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest"
GITHUB_RELEASES_LATEST = "https://github.com/ggml-org/whisper.cpp/releases/latest"

ASSET_PREFERENCES = [
    "whisper-bin-x64.zip",
]

MODELS = [
    {
        "name": "base.en",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "folder": Path("models") / "faster_whisper" / "base",
        "filename": "ggml-base.en.bin",
    },
    {
        "name": "tiny.en",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        "folder": Path("models") / "faster_whisper" / "tiny",
        "filename": "ggml-tiny.en.bin",
    },
]


def script_file() -> Path:
    return Path(__file__).resolve()


def script_dir() -> Path:
    return script_file().parent


def app_root() -> Path:
    """
    Resolve FrameVision root from the actual script location, not from the
    current working directory.

    Expected:
        C:/FrameVision-main/presets/extra_env/tiny_whisper_rootfix.py

    Then:
        script_dir = C:/FrameVision-main/presets/extra_env
        root       = C:/FrameVision-main
    """
    here = script_dir()

    if here.name.lower() == "extra_env" and here.parent.name.lower() == "presets":
        return here.parent.parent

    # Fallback if someone places this script directly in FrameVision root.
    if (here / "presets").exists() or (here / "models").exists():
        return here

    # Last-resort fallback for the intended /presets/extra_env/ placement.
    return here.parent.parent


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def print_header(root: Path) -> None:
    whisper_target = root / "presets" / "bin" / "whisper"
    base_target = root / "models" / "faster_whisper" / "base" / "ggml-base.en.bin"
    tiny_target = root / "models" / "faster_whisper" / "tiny" / "ggml-tiny.en.bin"

    print("=" * 72)
    print(APP_NAME)
    print(FIXED_VERSION)
    print("=" * 72)
    print(f"Script file:      {script_file()}")
    print(f"Script dir:       {script_dir()}")
    print(f"FrameVision root: {root}")
    print(f"Whisper target:   {whisper_target}")
    print(f"Base model:       {base_target}")
    print(f"Tiny model:       {tiny_target}")
    print()

    lower_target = str(whisper_target).lower().replace("/", "\\")
    if "\\presets\\extra_env\\presets\\" in lower_target:
        raise RuntimeError(
            "BAD ROOT DETECTED. Refusing to install into presets\\extra_env\\presets.\n"
            f"Resolved target was: {whisper_target}"
        )


def remove_bad_old_folder(root: Path) -> None:
    # Remove folder created by the broken old installer:
    # <root>/presets/extra_env/presets
    bad = root / "presets" / "extra_env" / "presets"
    if bad.exists():
        print(f"[cleanup] Removing wrong old installer folder: {bad}")
        shutil.rmtree(bad, ignore_errors=True)


def request_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FrameVision-whisper-installer",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def find_release_asset() -> tuple[str, str, str]:
    print("Checking latest whisper.cpp release...")
    data = request_json(GITHUB_API_LATEST)
    tag = data.get("tag_name", "latest")
    assets = data.get("assets", []) or []

    by_name = {asset.get("name", ""): asset for asset in assets}

    for wanted in ASSET_PREFERENCES:
        asset = by_name.get(wanted)
        if asset and asset.get("browser_download_url"):
            return tag, wanted, asset["browser_download_url"]

    candidates = []
    for asset in assets:
        name = asset.get("name", "")
        lower = name.lower()
        if lower.endswith(".zip") and "x64" in lower and "whisper" in lower:
            if "macos" not in lower and "ubuntu" not in lower and "arm" not in lower:
                candidates.append(asset)

    if candidates:
        asset = sorted(candidates, key=lambda a: a.get("name", ""))[0]
        return tag, asset.get("name", "whisper-bin-x64.zip"), asset["browser_download_url"]

    raise RuntimeError(
        "Could not find a Windows x64 whisper.cpp zip in the latest release.\n"
        f"Open manually: {GITHUB_RELEASES_LATEST}"
    )


def download_file(url: str, dest: Path, label: str, overwrite: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        print(f"[skip] {label}: already present ({human_size(dest.stat().st_size)})")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"[download] {label}")
    print(f"           {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "FrameVision-whisper-installer"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response, tmp.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            downloaded = 0
            last_print = 0.0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print > 0.75:
                    if total:
                        pct = downloaded * 100.0 / total
                        print(f"           {pct:5.1f}%  {human_size(downloaded)} / {human_size(total)}", end="\r")
                    else:
                        print(f"           {human_size(downloaded)}", end="\r")
                    last_print = now

            print(" " * 80, end="\r")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty file for {label}")

    if dest.exists():
        dest.unlink()
    tmp.replace(dest)
    print(f"[ok] {label}: {human_size(dest.stat().st_size)}")


def clear_folder(folder: Path) -> None:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return

    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def install_whisper_cpp(root: Path) -> None:
    tag, asset_name, url = find_release_asset()
    target_dir = root / "presets" / "bin" / "whisper"
    target_dir.mkdir(parents=True, exist_ok=True)

    lower_target = str(target_dir).lower().replace("/", "\\")
    if "\\presets\\extra_env\\presets\\" in lower_target:
        raise RuntimeError(f"Refusing bad whisper.cpp install target: {target_dir}")

    with tempfile.TemporaryDirectory(prefix="framevision_whisper_") as tmp_name:
        tmp_dir = Path(tmp_name)
        zip_path = tmp_dir / asset_name

        download_file(url, zip_path, f"whisper.cpp {tag} / {asset_name}", overwrite=True)

        print(f"[install] Extracting to: {target_dir}")
        print("[install] Cleaning old whisper.cpp files first...")
        clear_folder(target_dir)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()
                print("[cleanup] Removed downloaded zip")

    exe_candidates = [
        target_dir / "whisper-cli.exe",
        target_dir / "main.exe",
    ]
    found = next((p for p in exe_candidates if p.exists()), None)

    if found:
        print(f"[ok] Installed executable: {found.relative_to(root)}")
    else:
        found_any = list(target_dir.rglob("whisper-cli.exe")) + list(target_dir.rglob("main.exe"))
        if found_any:
            print(f"[ok] Installed executable: {found_any[0].relative_to(root)}")
        else:
            print("[warn] Extracted zip, but did not find whisper-cli.exe or main.exe.")
            print("       Check the release zip contents if whisper.cpp changed its layout.")


def install_models(root: Path) -> None:
    print()
    print("Installing whisper.cpp GGML models...")

    for model in MODELS:
        folder = root / model["folder"]
        dest = folder / model["filename"]

        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            print(f"[skip] {model['name']}: already present at {dest.relative_to(root)} ({human_size(dest.stat().st_size)})")
            continue

        download_file(model["url"], dest, f"model {model['name']}", overwrite=True)


def write_info_file(root: Path) -> None:
    info_path = root / "presets" / "bin" / "whisper" / "FRAMEVISION_WHISPER_CPP.txt"
    info = f"""FrameVision whisper.cpp helper

Installed by tiny_whisper_rootfix.py
{FIXED_VERSION}

Default binary folder:
  presets/bin/whisper/

Models:
  models/faster_whisper/base/ggml-base.en.bin
  models/faster_whisper/tiny/ggml-tiny.en.bin

Example command:
  presets\\bin\\whisper\\Release\\whisper-cli.exe -m models\\faster_whisper\\tiny\\ggml-tiny.en.bin -f input.wav -l en -otxt -nt

Notes:
  - This installer uses the normal Windows x64 whisper.cpp release zip when available.
  - The downloaded models are GGML whisper.cpp models, not Python faster-whisper models.
  - They are stored under models/faster_whisper because FrameVision already has that folder family.
"""
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(info, encoding="utf-8")


def main() -> int:
    root = app_root()

    try:
        print_header(root)
        remove_bad_old_folder(root)
        install_whisper_cpp(root)
        install_models(root)
        write_info_file(root)
    except urllib.error.HTTPError as e:
        print()
        print(f"[error] HTTP {e.code}: {e.reason}")
        print("        Download failed. Check internet connection or release availability.")
        return 1
    except urllib.error.URLError as e:
        print()
        print(f"[error] Network error: {e}")
        return 1
    except Exception as e:
        print()
        print(f"[error] {e}")
        return 1

    print()
    print("Done.")
    print("Fast test model:")
    print(r"  models\faster_whisper\tiny\ggml-tiny.en.bin")
    print("Better still-fast model:")
    print(r"  models\faster_whisper\base\ggml-base.en.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
