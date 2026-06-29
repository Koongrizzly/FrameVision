#!/usr/bin/env python3
"""
FrameVision Krea 2 Turbo GGUF downloader.

Downloads/updates:
- newest stable-diffusion.cpp Windows CUDA12 sd-cli into /presets/bin
- selected Krea 2 Turbo GGUF into /models/krea2
- Qwen3-VL 4B GGUF text encoder into /models/krea2
- Wan 2.1 VAE into /models/krea2

No Python ML environment is created. This is only a downloader/unpacker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_APP_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = DEFAULT_APP_ROOT
PRESETS_BIN = APP_ROOT / "presets" / "bin"
MODEL_DIR = APP_ROOT / "models" / "krea2"

GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
KREA_TURBO_REPO = "realrebelai/KREA-2_GGUFs"
QWEN_GGUF_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
WAN_VAE_URL = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=1"
WAN_VAE_NAME = "wan_2.1_vae.safetensors"

USER_AGENT = "FrameVision-Krea2-Downloader/1.1"
CHUNK = 1024 * 1024 * 4

# Turbo only. The BASE folder is intentionally ignored.
# The API listing is used first; this list is the offline/stale fallback and defines
# what Optional Installs can call directly.
TURBO_GGUF_FALLBACK = [
    "TURBO/Krea-2-Turbo-Q3_K_S.gguf",
    "TURBO/Krea-2-Turbo-Q3_K_M.gguf",
    "TURBO/Krea-2-Turbo-Q4_K_M.gguf",
    "TURBO/Krea-2-Turbo-Q5_K_S.gguf",
    "TURBO/Krea-2-Turbo-Q6_K.gguf",
    "TURBO/Krea-2-Turbo-Q8_0.gguf",
]


def set_app_root(root: Path) -> None:
    global APP_ROOT, PRESETS_BIN, MODEL_DIR
    APP_ROOT = root.resolve()
    PRESETS_BIN = APP_ROOT / "presets" / "bin"
    MODEL_DIR = APP_ROOT / "models" / "krea2"


def clear(enabled: bool = True) -> None:
    if enabled:
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


def download_if_needed(url: str, dest: Path, label: str, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"\nAlready exists, skipping: {dest.name}")
        return
    download(url, dest, label=label)


def latest_sd_cpp_cuda12_asset() -> Tuple[str, str, str]:
    data = request_json(GITHUB_LATEST_RELEASE_API)
    tag = data.get("tag_name", "latest")
    assets = data.get("assets", [])
    candidates: List[Tuple[str, str]] = []
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
    candidates.sort(key=lambda x: ("bin" not in x[0].lower(), x[0]))
    name, url = candidates[0]
    return tag, name, url


def extract_sd_cpp(zip_path: Path) -> None:
    PRESETS_BIN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fv_sdcpp_") as td:
        tmpdir = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        copied = 0
        for src in tmpdir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(tmpdir)
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


def install_sd_cli(force: bool = True) -> None:
    tag, name, url = latest_sd_cpp_cuda12_asset()
    print(f"\nLatest stable-diffusion.cpp release: {tag}")
    print(f"Selected asset: {name}")
    PRESETS_BIN.mkdir(parents=True, exist_ok=True)
    zip_path = PRESETS_BIN / name
    if zip_path.exists() and not force:
        extract_sd_cpp(zip_path)
        return
    download(url, zip_path, label=name)
    extract_sd_cpp(zip_path)


def hf_file_url(repo_id: str, path: str) -> str:
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + urllib.parse.quote(path, safe="/") + "?download=1"


def hf_siblings(repo_id: str) -> List[Dict[str, Any]]:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo_id, safe="/") + "?recursive=1"
    data = request_json(url)
    return data.get("siblings", [])


def quant_rank(name: str) -> Tuple[int, str]:
    low = name.lower()
    order = ["q3_k_s", "q3_k_m", "q4_k_m", "q4_k_s", "q5_k_s", "q5_k_m", "q6_k", "q8_0"]
    for i, key in enumerate(order):
        if key in low:
            return i, name
    return 999, name


def sort_ggufs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: quant_rank(x["name"]))


def list_krea_turbo_ggufs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        siblings = hf_siblings(KREA_TURBO_REPO)
    except Exception as e:
        print(f"\nCould not query Hugging Face file list, using built-in Turbo list: {e}")
        siblings = []

    for item in siblings:
        path = item.get("rfilename") or item.get("path") or ""
        if path.startswith("TURBO/") and path.lower().endswith(".gguf"):
            out.append({"path": path, "name": Path(path).name, "size": item.get("size"), "url": hf_file_url(KREA_TURBO_REPO, path)})

    if not out:
        out = [{"path": p, "name": Path(p).name, "size": None, "url": hf_file_url(KREA_TURBO_REPO, p)} for p in TURBO_GGUF_FALLBACK]
    return sort_ggufs(out)


def list_qwen_ggufs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in hf_siblings(QWEN_GGUF_REPO):
        path = item.get("rfilename") or item.get("path") or ""
        if path.lower().endswith(".gguf"):
            out.append({"path": path, "name": Path(path).name, "size": item.get("size"), "url": hf_file_url(QWEN_GGUF_REPO, path)})
    return sort_ggufs(out)


def find_item_by_name_or_quant(items: List[Dict[str, Any]], selector: str, default_quant: str = "Q4_K_M") -> Dict[str, Any]:
    if not items:
        raise RuntimeError("No GGUF files found.")
    want = selector.strip().lower()
    if want.endswith(".gguf"):
        for item in items:
            if item["name"].lower() == want or item["path"].lower() == want:
                return item
        raise RuntimeError(f"Requested GGUF not found: {selector}")

    if want in ("", "default"):
        want = default_quant.lower()
    for item in items:
        if want in item["name"].lower():
            return item
    raise RuntimeError(f"Requested GGUF quant not found: {selector}")


def choose_from_list(title: str, items: List[Dict[str, Any]], default_index: int = 0) -> Dict[str, Any]:
    print("\n" + title)
    print("-" * len(title))
    for i, item in enumerate(items, 1):
        default = "  [recommended]" if i - 1 == default_index else ""
        print(f" {i:2d}) {item['name']}  ({fmt_size(item.get('size'))}){default}")
    while True:
        raw = input(f"Select 1-{len(items)} [default {default_index + 1}]: ").strip()
        if raw == "":
            return items[default_index]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("Invalid selection.")


def select_turbo_gguf(selector: Optional[str] = None) -> Dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items = list_krea_turbo_ggufs()
    default = 0
    for i, item in enumerate(items):
        if "q4_k_m" in item["name"].lower():
            default = i
            break
    choice = find_item_by_name_or_quant(items, selector, "Q4_K_M") if selector else choose_from_list("Select Krea 2 Turbo GGUF", items, default_index=default)
    download_if_needed(choice["url"], MODEL_DIR / choice["name"], choice["name"])
    return choice


def select_text_encoder(selector: str = "Q4_K_M") -> Dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items = list_qwen_ggufs()
    choice = find_item_by_name_or_quant(items, selector, "Q4_K_M")
    download_if_needed(choice["url"], MODEL_DIR / choice["name"], choice["name"])
    return choice


def download_vae() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_if_needed(WAN_VAE_URL, MODEL_DIR / WAN_VAE_NAME, WAN_VAE_NAME)


def write_notes(turbo_name: str = "<Krea-2-Turbo GGUF>", qwen_name: str = "<Qwen3-VL GGUF>") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    note = MODEL_DIR / "KREA2_TURBO_NOTES.txt"
    note.write_text(
        "Krea 2 Turbo GGUF files for FrameVision/sd-cli\n"
        "\n"
        "Only the TURBO GGUF model files are downloaded by this installer.\n"
        "BASE GGUF files are intentionally ignored.\n"
        "\n"
        "Files should live here:\n"
        f"  {MODEL_DIR}\n"
        "\n"
        "Installed diffusion model:\n"
        f"  {turbo_name}\n"
        "Installed text encoder:\n"
        f"  {qwen_name}\n"
        "Installed VAE:\n"
        f"  {WAN_VAE_NAME}\n"
        "\n"
        "Basic sd-cli test shape:\n"
        f"  sd-cli.exe --diffusion-model {turbo_name} --llm {qwen_name} --vae {WAN_VAE_NAME} -p \"test prompt\" --steps 8 --cfg-scale 0 --width 1024 --height 1024 --diffusion-fa --offload-to-cpu -v\n"
        "\n"
        "Recommended first test: 1024x1024, 8 steps, CFG 0.\n",
        encoding="utf-8",
    )
    print(f"\nWrote notes: {note}")


def run_installer(args: argparse.Namespace) -> None:
    print("\nThis installer will:")
    print(" - update stable-diffusion.cpp sd-cli CUDA12 in presets/bin")
    print(" - download one selected Krea 2 Turbo GGUF")
    print(" - download one Qwen3-VL 4B GGUF text encoder")
    print(" - automatically download Wan 2.1 VAE")
    print("\nNo Python ML environment is created.")

    if not args.yes:
        ans = input("\nContinue? [Y/n]: ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return

    if not args.skip_sd_cli:
        install_sd_cli(force=True)
    turbo = select_turbo_gguf(args.turbo_gguf or args.gguf_quant)
    text_encoder = select_text_encoder(args.text_encoder_quant)
    download_vae()
    write_notes(turbo["name"], text_encoder["name"])
    print("\nDone. Krea 2 Turbo files are ready in models/krea2.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FrameVision Krea 2 Turbo GGUF downloader")
    p.add_argument("--app-root", default=None, help="FrameVision root folder. Default: two folders above this script.")
    p.add_argument("--yes", action="store_true", help="Run without confirmation prompts.")
    p.add_argument("--no-clear", action="store_true", help="Do not clear the console at startup.")
    p.add_argument("--skip-sd-cli", action="store_true", help="Do not update sd-cli.")
    p.add_argument("--gguf-quant", default=None, help="Turbo quant selector, for example Q4_K_M, Q5_K_S, Q8_0.")
    p.add_argument("--turbo-gguf", default=None, help="Exact Turbo GGUF filename or TURBO/path to download.")
    p.add_argument("--text-encoder-quant", default="Q4_K_M", help="Qwen3-VL GGUF text encoder quant. Default: Q4_K_M.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.app_root:
        set_app_root(Path(args.app_root))
    try:
        PRESETS_BIN.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clear(not args.no_clear)
        header()
        run_installer(args)
        if not args.yes:
            input("\nPress Enter to exit...")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except urllib.error.HTTPError as e:
        print(f"\nHTTP error: {e.code} {e.reason}")
        print(e.geturl())
        if not args.yes:
            input("\nPress Enter to exit...")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        if not args.yes:
            input("\nPress Enter to exit...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
