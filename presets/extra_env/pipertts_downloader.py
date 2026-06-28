#!/usr/bin/env python3
"""
FrameVision lightweight TTS downloader - Piper + Kokoro INT8

Expected location:
    <FrameVision root>/presets/extra_env/tts_downloader_rootfix.py

Installs / downloads:
    Piper Windows x64 helper:
        <FrameVision root>/presets/bin/piper/

    Piper voices:
        <FrameVision root>/models/tts/piper/en_US-lessac-low/
        <FrameVision root>/models/tts/piper/en_US-lessac-medium/

    Kokoro ONNX INT8 optional files:
        <FrameVision root>/models/tts/kokoro/kokoro-v1.0.int8.onnx
        <FrameVision root>/models/tts/kokoro/voices-v1.0.bin

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

APP_NAME = "FrameVision lightweight TTS downloader"
FIXED_VERSION = "VERSION THAT IS FIXED - tts-downloader-2026-06-28-001 - NEW FILE NAME"

PIPER_GITHUB_API_LATEST = "https://api.github.com/repos/rhasspy/piper/releases/latest"
PIPER_RELEASES_LATEST = "https://github.com/rhasspy/piper/releases/latest"
PIPER_ASSET_NAME = "piper_windows_amd64.zip"

KOKORO_RELEASE_TAG = "model-files-v1.0"
KOKORO_BASE_URL = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/{KOKORO_RELEASE_TAG}"

PIPER_VOICES = [
    {
        "name": "en_US-lessac-low",
        "folder": Path("models") / "tts" / "piper" / "en_US-lessac-low",
        "files": [
            (
                "en_US-lessac-low.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx",
            ),
            (
                "en_US-lessac-low.onnx.json",
                "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
            ),
        ],
    },
    {
        "name": "en_US-lessac-medium",
        "folder": Path("models") / "tts" / "piper" / "en_US-lessac-medium",
        "files": [
            (
                "en_US-lessac-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            ),
            (
                "en_US-lessac-medium.onnx.json",
                "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            ),
        ],
    },
]

KOKORO_FILES = [
    {
        "name": "kokoro-v1.0.int8.onnx",
        "url": f"{KOKORO_BASE_URL}/kokoro-v1.0.int8.onnx",
        "folder": Path("models") / "tts" / "kokoro",
        "filename": "kokoro-v1.0.int8.onnx",
    },
    {
        "name": "voices-v1.0.bin",
        "url": f"{KOKORO_BASE_URL}/voices-v1.0.bin",
        "folder": Path("models") / "tts" / "kokoro",
        "filename": "voices-v1.0.bin",
    },
]


def script_file() -> Path:
    return Path(__file__).resolve()


def script_dir() -> Path:
    return script_file().parent


def app_root() -> Path:
    here = script_dir()

    # Correct intended location:
    #   FrameVision-main/presets/extra_env/tts_downloader_rootfix.py
    if here.name.lower() == "extra_env" and here.parent.name.lower() == "presets":
        return here.parent.parent

    # Also allow direct root placement.
    if (here / "presets").exists() or (here / "models").exists():
        return here

    # Last fallback for unusual placement.
    return here.parent.parent


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def print_header(root: Path) -> None:
    piper_target = root / "presets" / "bin" / "piper"
    piper_models = root / "models" / "tts" / "piper"
    kokoro_models = root / "models" / "tts" / "kokoro"

    print("=" * 72)
    print(APP_NAME)
    print(FIXED_VERSION)
    print("=" * 72)
    print(f"Script file:      {script_file()}")
    print(f"Script dir:       {script_dir()}")
    print(f"FrameVision root: {root}")
    print(f"Piper target:     {piper_target}")
    print(f"Piper voices:     {piper_models}")
    print(f"Kokoro files:     {kokoro_models}")
    print()

    for target in [piper_target, piper_models, kokoro_models]:
        lower = str(target).lower().replace("/", "\\")
        if "\\presets\\extra_env\\presets\\" in lower or "\\presets\\extra_env\\models\\" in lower:
            raise RuntimeError(
                "BAD ROOT DETECTED. Refusing to install into presets\\extra_env.\n"
                f"Bad target was: {target}"
            )


def request_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FrameVision-tts-downloader",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def find_piper_asset() -> tuple[str, str, str]:
    print("Checking latest Piper release...")
    data = request_json(PIPER_GITHUB_API_LATEST)
    tag = data.get("tag_name", "latest")
    assets = data.get("assets", []) or []

    for asset in assets:
        name = asset.get("name", "")
        if name == PIPER_ASSET_NAME and asset.get("browser_download_url"):
            return tag, name, asset["browser_download_url"]

    for asset in assets:
        name = asset.get("name", "")
        lower = name.lower()
        if lower.endswith(".zip") and "windows" in lower and "amd64" in lower and asset.get("browser_download_url"):
            return tag, name, asset["browser_download_url"]

    raise RuntimeError(
        f"Could not find {PIPER_ASSET_NAME} in latest Piper release.\n"
        f"Open manually: {PIPER_RELEASES_LATEST}"
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

    req = urllib.request.Request(url, headers={"User-Agent": "FrameVision-tts-downloader"})

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


def install_piper_binary(root: Path) -> None:
    tag, asset_name, url = find_piper_asset()
    target_dir = root / "presets" / "bin" / "piper"
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="framevision_piper_") as tmp_name:
        tmp_dir = Path(tmp_name)
        zip_path = tmp_dir / asset_name

        download_file(url, zip_path, f"Piper {tag} / {asset_name}", overwrite=True)

        print(f"[install] Extracting Piper to: {target_dir}")
        print("[install] Cleaning old Piper files first...")
        clear_folder(target_dir)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()
                print("[cleanup] Removed downloaded zip")

    found_any = list(target_dir.rglob("piper.exe"))
    if found_any:
        print(f"[ok] Installed Piper executable: {found_any[0].relative_to(root)}")
    else:
        print("[warn] Extracted Piper zip, but did not find piper.exe.")


def install_piper_voices(root: Path) -> None:
    print()
    print("Installing Piper voices...")

    for voice in PIPER_VOICES:
        folder = root / voice["folder"]
        folder.mkdir(parents=True, exist_ok=True)

        print(f"[voice] {voice['name']}")

        for filename, url in voice["files"]:
            dest = folder / filename
            # Voice files are stable. Skip existing good files.
            if dest.exists() and dest.stat().st_size > 256:
                print(f"[skip] {filename}: already present at {dest.relative_to(root)} ({human_size(dest.stat().st_size)})")
                continue

            download_file(url, dest, filename, overwrite=True)


def install_kokoro_files(root: Path) -> None:
    print()
    print("Installing optional Kokoro ONNX INT8 files...")

    for item in KOKORO_FILES:
        folder = root / item["folder"]
        dest = folder / item["filename"]
        folder.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            print(f"[skip] {item['name']}: already present at {dest.relative_to(root)} ({human_size(dest.stat().st_size)})")
            continue

        download_file(item["url"], dest, item["name"], overwrite=True)


def write_info_file(root: Path) -> None:
    info_path = root / "presets" / "bin" / "piper" / "FRAMEVISION_TTS_HELPERS.txt"
    info = f"""FrameVision lightweight TTS helpers

Installed by:
  tts_downloader_rootfix.py
{FIXED_VERSION}

Piper:
  presets/bin/piper/
  models/tts/piper/en_US-lessac-low/
  models/tts/piper/en_US-lessac-medium/

Kokoro optional:
  models/tts/kokoro/kokoro-v1.0.int8.onnx
  models/tts/kokoro/voices-v1.0.bin

Fast Piper test command:
  echo Hello from FrameVision. | presets\\bin\\piper\\piper.exe --model models\\tts\\piper\\en_US-lessac-low\\en_US-lessac-low.onnx --output_file output\\audio\\piper_test.wav

Notes:
  - Piper is the simple fast/small CPU helper path.
  - Kokoro INT8 is downloaded as optional better-quality ONNX model data.
  - Kokoro still needs a runner/integration script later; this downloader only fetches model files.
"""
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(info, encoding="utf-8")


def main() -> int:
    root = app_root()

    try:
        print_header(root)
        install_piper_binary(root)
        install_piper_voices(root)
        install_kokoro_files(root)
        write_info_file(root)
    except urllib.error.HTTPError as e:
        print()
        print(f"[error] HTTP {e.code}: {e.reason}")
        print("        Download failed. Check internet connection, GitHub/Hugging Face availability, or asset names.")
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
    print("Fastest Piper test voice:")
    print(r"  models\tts\piper\en_US-lessac-low\en_US-lessac-low.onnx")
    print("Better Piper voice:")
    print(r"  models\tts\piper\en_US-lessac-medium\en_US-lessac-medium.onnx")
    print("Optional Kokoro INT8:")
    print(r"  models\tts\kokoro\kokoro-v1.0.int8.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
