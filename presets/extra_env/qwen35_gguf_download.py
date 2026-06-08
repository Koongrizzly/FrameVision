#!/usr/bin/env python3
"""
Qwen3.5 GGUF downloader for FrameVision.

- Lists available GGUF files in:
    lmstudio-community/Qwen3.5-4B-GGUF
    lmstudio-community/Qwen3.5-9B-GGUF
- Lets you select one or more GGUF variants to download
- Skips already-downloaded files (size match)
- Auto-downloads the matching mmproj file when you select a model

Destinations (relative to FrameVision root):
  models/qwen35_gguf/4B/
  models/qwen35_gguf/9B/

Usage examples:
  python qwen35_gguf_download.py --list
  python qwen35_gguf_download.py --model 4B --select Q4_K_M,Q5_K_M
  python qwen35_gguf_download.py --model 9B --all
  python qwen35_gguf_download.py            (interactive)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REPO_4B = "lmstudio-community/Qwen3.5-4B-GGUF"
REPO_9B = "lmstudio-community/Qwen3.5-9B-GGUF"

MMPROJ_4B = "mmproj-Qwen3.5-4B-BF16.gguf"
MMPROJ_9B = "mmproj-Qwen3.5-9B-BF16.gguf"

DEST_4B_REL = os.path.join("models", "qwen35_gguf", "4B")
DEST_9B_REL = os.path.join("models", "qwen35_gguf", "9B")

HF_API_TREE = "https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"

USER_AGENT = "FrameVision-Qwen35-GGUF-Downloader/1.0"

# -----------------------------
# Optional: llama.cpp runner installer (portable, no Python env)
# -----------------------------
# We use GitHub Releases API to fetch the latest prebuilt Windows binaries.
# Installed to: <FrameVisionRoot>/presets/bin/llama/<release_tag>/<backend>/
#
# Backends:
#   cpu      -> llama-<tag>-bin-win-cpu-x64.zip
#   vulkan   -> llama-<tag>-bin-win-vulkan-x64.zip
#   cuda12   -> llama-<tag>-bin-win-cuda-12.4-x64.zip  (+ cudart-llama-bin-win-cuda-12.4-x64.zip)
#   cuda13   -> llama-<tag>-bin-win-cuda-13.1-x64.zip  (+ cudart-llama-bin-win-cuda-13.1-x64.zip)
#
# NOTE: CUDA builds require the matching CUDA runtime DLL bundle (cudart-llama-bin-...).
# We download both zips for CUDA backends and extract into the same folder.

GH_LLAMA_REPO = "ggml-org/llama.cpp"
GH_API_LATEST = f"https://api.github.com/repos/{GH_LLAMA_REPO}/releases/latest"

LLAMA_BACKENDS = ["cpu", "vulkan", "cuda12", "cuda13"]

def _http_json_github(url: str, timeout: int = 60) -> object:
    # GitHub API can rate-limit; use a clear UA.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))

def _download_to_file(url: str, dest_path: str, overwrite: bool = False) -> None:
    ensure_dir(os.path.dirname(dest_path))
    if os.path.exists(dest_path) and not overwrite:
        return
    tmp = dest_path + ".part"
    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        while True:
            buf = resp.read(1024 * 1024 * 8)
            if not buf:
                break
            f.write(buf)
    try:
        os.replace(tmp, dest_path)
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(tmp, dest_path)

def _extract_zip(zip_path: str, dest_dir: str) -> None:
    import zipfile
    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def _llama_install_root(fv_root: str) -> str:
    return os.path.join(fv_root, "presets", "bin", "llama")

def _find_llama_cli_anywhere(fv_root: str) -> Optional[str]:
    base = _llama_install_root(fv_root)
    if not os.path.isdir(base):
        return None
    for r, ds, fs in os.walk(base):
        for f in fs:
            if f.lower() == "llama-cli.exe":
                return os.path.join(r, f)
    return None

def install_llama_cpp(fv_root: str, backend: str = "cuda12", overwrite: bool = False) -> Optional[str]:
    backend = (backend or "").strip().lower()
    if backend not in LLAMA_BACKENDS:
        raise ValueError(f"Unknown llama backend: {backend}. Choose from: {', '.join(LLAMA_BACKENDS)}")

    try:
        meta = _http_json_github(GH_API_LATEST)
    except Exception as e:
        print(f"ERROR: could not query llama.cpp latest release: {e}")
        return None

    tag = meta.get("tag_name") or "latest"
    assets = meta.get("assets", []) if isinstance(meta, dict) else []
    if not isinstance(assets, list):
        assets = []

    def pick_asset(name_contains: str) -> Optional[dict]:
        for a in assets:
            if not isinstance(a, dict):
                continue
            nm = (a.get("name") or "")
            if name_contains in nm:
                return a
        return None

    # Map backend to expected asset names
    if backend == "cpu":
        main_name = f"llama-{tag}-bin-win-cpu-x64.zip"
        cudart_name = None
    elif backend == "vulkan":
        main_name = f"llama-{tag}-bin-win-vulkan-x64.zip"
        cudart_name = None
    elif backend == "cuda12":
        main_name = f"llama-{tag}-bin-win-cuda-12.4-x64.zip"
        cudart_name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
    else:  # cuda13
        main_name = f"llama-{tag}-bin-win-cuda-13.1-x64.zip"
        cudart_name = "cudart-llama-bin-win-cuda-13.1-x64.zip"

    main_asset = pick_asset(main_name)
    if not main_asset:
        # As a fallback, try substring match (some tags may differ slightly)
        main_asset = pick_asset("bin-win-cpu-x64.zip" if backend == "cpu" else
                                "bin-win-vulkan-x64.zip" if backend == "vulkan" else
                                "bin-win-cuda-12.4-x64.zip" if backend == "cuda12" else
                                "bin-win-cuda-13.1-x64.zip")
    if not main_asset:
        print(f"ERROR: could not find llama.cpp Windows asset for backend '{backend}' in latest release ({tag}).")
        return None

    cudart_asset = None
    if cudart_name:
        cudart_asset = pick_asset(cudart_name)
        if not cudart_asset:
            cudart_asset = pick_asset("cudart-llama-bin-win-cuda-12.4-x64.zip" if backend == "cuda12" else
                                      "cudart-llama-bin-win-cuda-13.1-x64.zip")

    install_dir = os.path.join(_llama_install_root(fv_root), str(tag), backend)
    ensure_dir(install_dir)
    cache_dir = os.path.join(_llama_install_root(fv_root), str(tag), "_downloads")
    ensure_dir(cache_dir)

    # Download + extract main
    main_url = main_asset.get("browser_download_url")
    main_zip = os.path.join(cache_dir, os.path.basename(main_asset.get("name") or "llama.zip"))
    if main_url:
        print(f"\nInstalling llama.cpp runner ({backend}) -> {install_dir}")
        print(f"↓ {main_asset.get('name')}")
        _download_to_file(main_url, main_zip, overwrite=overwrite)
        _extract_zip(main_zip, install_dir)
    else:
        print("ERROR: missing browser_download_url for main asset.")
        return None

    # Download + extract cudart (CUDA only)
    if cudart_asset and cudart_asset.get("browser_download_url"):
        cu_url = cudart_asset["browser_download_url"]
        cu_zip = os.path.join(cache_dir, os.path.basename(cudart_asset.get("name") or "cudart.zip"))
        print(f"↓ {cudart_asset.get('name')}")
        _download_to_file(cu_url, cu_zip, overwrite=overwrite)
        _extract_zip(cu_zip, install_dir)

    # Find llama-cli.exe in extracted output (sometimes nested in a folder)
    cli = None
    for r, ds, fs in os.walk(install_dir):
        for f in fs:
            if f.lower() == "llama-cli.exe":
                cli = os.path.join(r, f)
                break
        if cli:
            break

    if cli:
        print(f"✓ llama-cli.exe found: {cli}")
    else:
        print("WARNING: llama-cli.exe not found after extraction (unexpected).")
    return cli

def ensure_llama_installed_if_needed(fv_root: str, backend: str = "cuda12", overwrite: bool = False) -> Optional[str]:
    existing = _find_llama_cli_anywhere(fv_root)
    if existing and not overwrite:
        return existing
    return install_llama_cpp(fv_root, backend=backend, overwrite=overwrite)

@dataclass
class RemoteFile:
    path: str
    size: Optional[int] = None  # bytes

def _framevision_root_from_script() -> str:
    # Expected location: <root>/presets/extra_env/qwen35_gguf_download.py
    here = os.path.abspath(os.path.dirname(__file__))
    # go up: extra_env -> presets -> root
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    return root

def _http_json(url: str, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))

def _http_head_content_length(url: str, timeout: int = 60) -> Optional[int]:
    # Follow redirects to get final content-length where possible
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None

def _human_size(n: Optional[int]) -> str:
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{n} B"

def list_repo_files(repo: str, rev: str = "main") -> List[RemoteFile]:
    url = HF_API_TREE.format(repo=repo, rev=urllib.parse.quote(rev, safe=""))
    try:
        items = _http_json(url)
    except Exception:
        # Fallback: use model metadata which includes siblings (filenames + sizes)
        fallback_url = f"https://huggingface.co/api/models/{repo}"
        meta = _http_json(fallback_url)
        siblings = meta.get("siblings", []) if isinstance(meta, dict) else []
        items = [{"type":"file","path": s.get("rfilename"), "size": s.get("size")} for s in siblings if isinstance(s, dict) and s.get("rfilename")]
    files: List[RemoteFile] = []
    for it in items:
        # entries include "path", "type", "size" (sometimes absent)
        if isinstance(it, dict) and it.get("type") == "file":
            p = it.get("path")
            if isinstance(p, str):
                files.append(RemoteFile(path=p, size=it.get("size") if isinstance(it.get("size"), int) else None))
    return files

def filter_ggufs(files: List[RemoteFile]) -> List[RemoteFile]:
    out = []
    for f in files:
        if f.path.lower().endswith(".gguf"):
            out.append(f)
    return out

def resolve_url(repo: str, path: str, rev: str = "main") -> str:
    return HF_RESOLVE.format(repo=repo, rev=rev, path=urllib.parse.quote(path, safe="/"))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def file_ok(path: str, expected_size: Optional[int]) -> bool:
    if not os.path.exists(path):
        return False
    if expected_size is None:
        # if we can't verify size, treat existing file as OK to avoid re-downloading
        return True
    try:
        return os.path.getsize(path) == expected_size
    except Exception:
        return False

def download_file(url: str, dest_path: str, expected_size: Optional[int], overwrite: bool = False) -> None:
    ensure_dir(os.path.dirname(dest_path))

    if (not overwrite) and file_ok(dest_path, expected_size):
        print(f"✓ Exists, skipping: {os.path.basename(dest_path)}")
        return

    tmp_path = dest_path + ".part"
    # Simple resume support if partial exists and server accepts Range
    resume_from = 0
    if os.path.exists(tmp_path) and (not overwrite):
        try:
            resume_from = os.path.getsize(tmp_path)
        except Exception:
            resume_from = 0

    headers = {"User-Agent": USER_AGENT}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    req = urllib.request.Request(url, headers=headers)
    print(f"↓ Downloading: {os.path.basename(dest_path)}")
    if expected_size is not None:
        print(f"  Size: {_human_size(expected_size)}")

    start = time.time()
    downloaded = resume_from
    chunk = 1024 * 1024 * 8  # 8MB

    mode = "ab" if resume_from > 0 else "wb"
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, mode) as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if expected_size:
                    pct = (downloaded / expected_size) * 100.0
                    speed = downloaded / max(time.time() - start, 1e-6)
                    sys.stdout.write(f"\r  {pct:6.2f}%  {_human_size(downloaded)}  ({_human_size(int(speed))}/s)")
                else:
                    speed = downloaded / max(time.time() - start, 1e-6)
                    sys.stdout.write(f"\r  {_human_size(downloaded)}  ({_human_size(int(speed))}/s)")
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write("\n")
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    # Finalize
    try:
        os.replace(tmp_path, dest_path)
    except Exception:
        # fallback
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(tmp_path, dest_path)

    # Verify size if known
    if expected_size is not None:
        actual = os.path.getsize(dest_path)
        if actual != expected_size:
            raise RuntimeError(f"Size mismatch for {dest_path}: got {actual}, expected {expected_size}")

    print(f"✓ Saved: {dest_path}")

def pick_dest(model: str, fv_root: str) -> Tuple[str, str, str]:
    m = model.strip().upper()
    if m == "4B":
        return REPO_4B, MMPROJ_4B, os.path.join(fv_root, DEST_4B_REL)
    if m == "9B":
        return REPO_9B, MMPROJ_9B, os.path.join(fv_root, DEST_9B_REL)
    raise ValueError("Model must be 4B or 9B")

def _match_any(name: str, patterns: List[str]) -> bool:
    ln = name.lower()
    for p in patterns:
        p = p.strip()
        if not p:
            continue
        if p.lower() in ln:
            return True
    return False

def interactive_select(files: List[RemoteFile]) -> List[RemoteFile]:
    if not files:
        return []
    print("\nAvailable GGUF files:")
    for i, f in enumerate(files, 1):
        print(f"  [{i:2d}] {f.path}  ({_human_size(f.size)})")
    print("\nSelect one or more by number, comma-separated (e.g. 1,3,5).")
    print("Or type 'all' to select all GGUFs, or press Enter to cancel.")
    s = input("> ").strip()
    if not s:
        return []
    if s.lower() == "all":
        return files
    idxs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
            if 1 <= idx <= len(files):
                idxs.append(idx - 1)
        except Exception:
            pass
    idxs = sorted(set(idxs))
    return [files[i] for i in idxs]

def main() -> int:
    ap = argparse.ArgumentParser(description="Download Qwen3.5 GGUF models (4B/9B) into FrameVision's models/qwen35_gguf folder.")
    ap.add_argument("--model", choices=["4B", "9B", "both"], default=None, help="Which model repo to use.")
    ap.add_argument("--list", action="store_true", help="List available GGUF files and exit.")
    ap.add_argument("--all", action="store_true", help="Download all GGUFs for the selected model(s).")
    ap.add_argument("--select", default="", help="Comma-separated substrings to match GGUF filenames (e.g. Q4_K_M,Q5_K_M).")
    ap.add_argument("--rev", default="main", help="HuggingFace revision (default: main).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files (otherwise skip).")
    ap.add_argument("--no-mmproj", action="store_true", help="Do not auto-download mmproj file.")
    ap.add_argument("--install-llama", choices=["off","cpu","vulkan","cuda12","cuda13","auto"], default="auto",
                    help="Install llama.cpp runner into presets/bin/llama if missing. auto defaults to cuda12 on Windows.")
    ap.add_argument("--llama-overwrite", action="store_true", help="Re-download/extract llama.cpp runner even if already installed.")

    args = ap.parse_args()

    fv_root = _framevision_root_from_script()
    print(f"FrameVision root: {fv_root}")

    # Determine models to process
    models: List[str]
    if args.model is None:
        # interactive default: ask
        print("\nChoose model to download:")
        print("  [1] 4B")
        print("  [2] 9B")
        print("  [3] both")
        choice = input("> ").strip()
        models = ["4B"] if choice == "1" else ["9B"] if choice == "2" else ["4B", "9B"] if choice == "3" else []
        if not models:
            print("Cancelled.")
            return 0
    elif args.model == "both":
        models = ["4B", "9B"]
    else:
        models = [args.model]

    sel_patterns = [p.strip() for p in args.select.split(",") if p.strip()]
    any_selected = False

    # Optional: ensure llama.cpp runner exists for text LLM GGUF usage
    if (not args.list) and (args.install_llama or "auto"):
        install_choice = (args.install_llama or "auto").lower()
        if install_choice != "off":
            backend = "cuda12" if install_choice == "auto" else install_choice
            # Install only if missing (unless llama-overwrite)
            _ = ensure_llama_installed_if_needed(fv_root, backend=backend, overwrite=bool(args.llama_overwrite))

    for m in models:
        repo, mmproj, dest = pick_dest(m, fv_root)
        ensure_dir(dest)

        print(f"\n=== {m} :: {repo} ===")
        try:
            all_files = list_repo_files(repo, rev=args.rev)
        except Exception as e:
            print(f"ERROR: failed to list files for {repo}: {e}")
            continue

        ggufs = filter_ggufs(all_files)

        # Split out mmproj (we'll handle it separately)
        mmproj_entry = None
        other_ggufs: List[RemoteFile] = []
        for f in ggufs:
            base = os.path.basename(f.path)
            if base == mmproj:
                mmproj_entry = f
            else:
                other_ggufs.append(f)

        other_ggufs.sort(key=lambda x: x.path.lower())

        if args.list:
            print("GGUFs:")
            for f in other_ggufs:
                print(f"  - {f.path} ({_human_size(f.size)})")
            if mmproj_entry:
                print(f"mmproj: {mmproj_entry.path} ({_human_size(mmproj_entry.size)})")
            else:
                print(f"mmproj: {mmproj} (not found in repo listing)")
            continue

        selected: List[RemoteFile] = []
        if args.all:
            selected = other_ggufs
        elif sel_patterns:
            selected = [f for f in other_ggufs if _match_any(os.path.basename(f.path), sel_patterns)]
        else:
            # interactive selection
            selected = interactive_select(other_ggufs)

        if not selected:
            print("No GGUF variants selected for this model.")
            continue

        any_selected = True

        # Auto mmproj if needed
        if not args.no_mmproj:
            mmproj_dest = os.path.join(dest, mmproj)
            if os.path.exists(mmproj_dest):
                print(f"✓ mmproj exists: {mmproj}")
            else:
                if mmproj_entry is None:
                    # If it wasn't in tree listing, still attempt to download by name
                    mm_url = resolve_url(repo, mmproj, rev=args.rev)
                    mm_size = _http_head_content_length(mm_url)  # best effort
                    download_file(mm_url, mmproj_dest, mm_size, overwrite=args.overwrite)
                else:
                    mm_url = resolve_url(repo, mmproj_entry.path, rev=args.rev)
                    mm_size = mmproj_entry.size if mmproj_entry.size else _http_head_content_length(mm_url)
                    download_file(mm_url, mmproj_dest, mm_size, overwrite=args.overwrite)

        # Download selected GGUFs
        for f in selected:
            url = resolve_url(repo, f.path, rev=args.rev)
            size = f.size if f.size else _http_head_content_length(url)
            dest_path = os.path.join(dest, os.path.basename(f.path))
            download_file(url, dest_path, size, overwrite=args.overwrite)

    if args.list:
        return 0

    if not any_selected:
        print("\nNothing downloaded.")
        return 0

    print("\nDone.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
