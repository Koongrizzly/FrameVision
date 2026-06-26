#!/usr/bin/env python3
"""
Krea 2 Turbo GGUF downloader.

Installs/updates the newest stable-diffusion.cpp CUDA12 Windows sd-cli release
into /presets/bin and downloads selected Krea 2 Turbo GGUF assets into
/models/krea2.

No Python ML environment is created. This is only a downloader/unpacker.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_ROOT = Path(__file__).resolve().parents[2]
PRESETS_BIN = APP_ROOT / "presets" / "bin"
MODEL_DIR = APP_ROOT / "models" / "krea2"

GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
KREA_TURBO_REPO = "realrebelai/KREA-2_GGUFs"
QWEN_GGUF_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
WAN_VAE_URL = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
WAN_VAE_NAME = "wan_2.1_vae.safetensors"

USER_AGENT = "FrameVision-Krea2-Downloader/1.0"
CHUNK = 1024 * 1024 * 4


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def header() -> None:
    print("=" * 72)
    print(" FrameVision - Krea 2 Turbo GGUF downloader")
    print("=" * 72)
    print(f" App root : {APP_ROOT}")
    print(f" sd-cli   : {PRESETS_BIN}")
    print(f" models   : {MODEL_DIR}")
    print("=" * 72)


def request_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_head_length(url: str) -> Optional[int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            value = resp.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def fmt_size(n: Optional[int]) -> str:
    if n is None:
        return "unknown size"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GB"


def download(url: str, dest: Path, label: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total else None
        done = 0
        name = label or dest.name
        print(f"\nDownloading {name}")
        print(f" -> {dest}")
        if total_i:
            print(f" Size: {fmt_size(total_i)}")
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total_i:
                pct = done * 100 / total_i
                print(f"\r {pct:6.2f}%  {fmt_size(done)} / {fmt_size(total_i)}", end="")
            else:
                print(f"\r {fmt_size(done)}", end="")
        print()
    tmp.replace(dest)


def latest_sd_cpp_cuda12_asset() -> Tuple[str, str, str]:
    data = request_json(GITHUB_LATEST_RELEASE_API)
    tag = data.get("tag_name", "latest")
    assets = data.get("assets", [])
    candidates = []
    for asset in assets:
        name = asset.get("name", "")
        url = asset.get("browser_download_url", "")
        if not name.lower().endswith(".zip"):
            continue
        low = name.lower()
        if "win" in low and "cuda12" in low and "x64" in low:
            candidates.append((name, url))
    if not candidates:
        raise RuntimeError("Could not find a Windows CUDA12 x64 ZIP asset in the latest stable-diffusion.cpp release.")
    # Prefer the normal binary package if multiple variants ever show up.
    candidates.sort(key=lambda x: ("bin" not in x[0].lower(), x[0]))
    name, url = candidates[0]
    return tag, name, url


def extract_sd_cpp(zip_path: Path) -> None:
    PRESETS_BIN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fv_sdcpp_") as td:
        tmpdir = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Copy all extracted files into presets/bin. This overwrites the old sd-cli release,
        # but avoids deleting unrelated tools the user may keep in presets/bin.
        copied = 0
        for src in tmpdir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(tmpdir)
            # Some GitHub zips have a top-level folder. Flatten one wrapper folder when present.
            parts = rel.parts
            if len(parts) > 1 and parts[0].lower().startswith("sd-"):
                rel = Path(*parts[1:])
            dest = PRESETS_BIN / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        print(f"Extracted {copied} file(s) to {PRESETS_BIN}")
    zip_path.unlink(missing_ok=True)
    print("Deleted downloaded ZIP after unpacking.")


def hf_siblings(repo_id: str) -> List[Dict[str, Any]]:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo_id, safe="/") + "?recursive=1"
    data = request_json(url)
    return data.get("siblings", [])


def hf_file_url(repo_id: str, path: str) -> str:
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + urllib.parse.quote(path, safe="/")


def list_krea_turbo_ggufs() -> List[Dict[str, Any]]:
    out = []
    for item in hf_siblings(KREA_TURBO_REPO):
        path = item.get("rfilename") or item.get("path") or ""
        if path.startswith("TURBO/") and path.lower().endswith(".gguf"):
            out.append({"path": path, "name": Path(path).name, "size": item.get("size"), "url": hf_file_url(KREA_TURBO_REPO, path)})
    return sort_ggufs(out)


def list_qwen_ggufs() -> List[Dict[str, Any]]:
    out = []
    for item in hf_siblings(QWEN_GGUF_REPO):
        path = item.get("rfilename") or item.get("path") or ""
        if path.lower().endswith(".gguf"):
            out.append({"path": path, "name": Path(path).name, "size": item.get("size"), "url": hf_file_url(QWEN_GGUF_REPO, path)})
    return sort_ggufs(out)


def quant_rank(name: str) -> Tuple[int, str]:
    low = name.lower()
    order = [
        "q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0",
        "f16", "bf16", "f32",
    ]
    for i, key in enumerate(order):
        if key in low:
            return i, name
    return 999, name


def sort_ggufs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: quant_rank(x["name"]))


def choose_from_list(title: str, items: List[Dict[str, Any]], default_index: int = 0, allow_skip: bool = True) -> Optional[Dict[str, Any]]:
    print("\n" + title)
    print("-" * len(title))
    if not items:
        print("No files found.")
        return None
    for i, item in enumerate(items, 1):
        default = "  [recommended]" if i - 1 == default_index else ""
        print(f" {i:2d}) {item['name']}  ({fmt_size(item.get('size'))}){default}")
    if allow_skip:
        print("  0) Skip")
    while True:
        raw = input(f"Select 1-{len(items)}" + (" or 0 to skip" if allow_skip else "") + f" [default {default_index+1}]: ").strip()
        if raw == "":
            return items[default_index]
        if allow_skip and raw == "0":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("Invalid selection.")


def download_if_needed(url: str, dest: Path, label: str, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"\nAlready exists, skipping: {dest.name}")
        return
    download(url, dest, label=label)


def select_turbo_gguf() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items = list_krea_turbo_ggufs()
    choice = choose_from_list("Select Krea 2 Turbo GGUF", items, default_index=0, allow_skip=False)
    if not choice:
        raise RuntimeError("No Krea 2 Turbo GGUF selected.")
    download_if_needed(choice["url"], MODEL_DIR / choice["name"], choice["name"])


def select_text_encoder() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items = list_qwen_ggufs()
    default = 0
    for i, item in enumerate(items):
        if "q4_k_m" in item["name"].lower():
            default = i
            break
    choice = choose_from_list("Select Qwen3-VL 4B GGUF text encoder", items, default_index=default, allow_skip=False)
    if not choice:
        raise RuntimeError("No Qwen3-VL text encoder selected.")
    download_if_needed(choice["url"], MODEL_DIR / choice["name"], choice["name"])


def install_sd_cli() -> None:
    tag, name, url = latest_sd_cpp_cuda12_asset()
    print(f"\nLatest stable-diffusion.cpp release: {tag}")
    print(f"Selected asset: {name}")
    PRESETS_BIN.mkdir(parents=True, exist_ok=True)
    zip_path = PRESETS_BIN / name
    download(url, zip_path, label=name)
    extract_sd_cpp(zip_path)


def download_vae() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_if_needed(WAN_VAE_URL, MODEL_DIR / WAN_VAE_NAME, WAN_VAE_NAME)


def run_installer() -> None:
    print("\nThis installer will:")
    print(" - update stable-diffusion.cpp sd-cli CUDA12 in presets/bin")
    print(" - ask you to select one Krea 2 Turbo GGUF")
    print(" - ask you to select one Qwen3-VL 4B GGUF text encoder")
    print(" - automatically download Wan 2.1 VAE")
    print("\nNo Python ML environment is created.")
    ans = input("\nContinue? [Y/n]: ").strip().lower()
    if ans not in ("", "y", "yes"):
        print("Cancelled.")
        return

    install_sd_cli()
    select_turbo_gguf()
    select_text_encoder()
    download_vae()
    write_notes()
    print("\nDone. Krea 2 Turbo files are ready in models/krea2.")


def write_notes() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    note = MODEL_DIR / "KREA2_TURBO_NOTES.txt"
    note.write_text(
        "Krea 2 Turbo GGUF files for FrameVision/sd-cli\n"
        "\n"
        "Files should live here:\n"
        f"  {MODEL_DIR}\n"
        "\n"
        "Basic sd-cli test shape:\n"
        "  sd-cli.exe --diffusion-model <Krea-2-Turbo GGUF> --llm <Qwen3-VL GGUF> --vae wan_2.1_vae.safetensors -p \"test prompt\" --steps 8 --cfg-scale 0 --width 1024 --height 1024 --diffusion-fa --offload-to-cpu -v\n"
        "\n"
        "Recommended first test: 1024x1024, 8 steps, CFG 0.\n",
        encoding="utf-8",
    )
    print(f"\nWrote notes: {note}")


def show_paths() -> None:
    print("\nPaths")
    print("-----")
    print(f"App root   : {APP_ROOT}")
    print(f"presets/bin: {PRESETS_BIN}")
    print(f"models     : {MODEL_DIR}")


def main() -> int:
    try:
        PRESETS_BIN.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clear()
        header()
        run_installer()
        input("\nPress Enter to exit...")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except urllib.error.HTTPError as e:
        print(f"\nHTTP error: {e.code} {e.reason}")
        print(e.geturl())
        input("\nPress Enter to exit...")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        input("\nPress Enter to exit...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
