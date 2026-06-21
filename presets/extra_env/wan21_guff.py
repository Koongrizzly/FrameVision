#!/usr/bin/env python
"""
WAN 2.1 GGUF helper

Goals:
- Keep a catalog of GGUF (and required companion) model files.
- Download exactly one chosen file (not a whole snapshot).
- Optionally grab required companion files (VAE / CLIP-Vision / UMT5 encoder).
- Download a Windows stable-diffusion.cpp build (sd-cli.exe) for local inference.

This is designed to be called from:
- presets/extra_env/wan21_guff_install.bat (one click installer)
- later: FrameVision optional installs (download one selected model)

Sources for model locations:
- GGUF diffusion models (Wan2.1 T2V 14B, I2V 14B 480P): city96 repos on Hugging Face.
- UMT5 encoder GGUF: city96/umt5-xxl-encoder-gguf on Hugging Face.
- VAE + CLIP-Vision safetensors: Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ------------------------------ HF direct download -----------------------

def _hf_resolve_url(repo_id: str, filename: str) -> str:
    # Works for public repos (follows redirects to LFS storage/CDN)
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _download_stream(url: str, out_path: Path, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    r = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", "0") or "0")

    pbar = None
    try:
        from tqdm import tqdm  # optional
        pbar = tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=out_path.name)
    except Exception:
        pbar = None

    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            if pbar is not None:
                pbar.update(len(chunk))
    if pbar is not None:
        pbar.close()

    tmp.replace(out_path)

# ------------------------------ Catalog ---------------------------------

@dataclass(frozen=True)
class CatalogItem:
    key: str
    repo_id: str
    filename: str
    subdir: str  # relative under <root>/models/wan21gguf/
    kind: str    # "diffusion_t2v" | "diffusion_i2v" | "t5xxl" | "vae" | "clip_vision"


CATALOG: List[CatalogItem] = [
    # ---- Diffusion GGUFs (common picks) ----
    CatalogItem("wan21_t2v14b_q4_k_m", "city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q4_K_M.gguf", "diffusion_models", "diffusion_t2v"),
    CatalogItem("wan21_t2v14b_q5_k_m", "city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q5_K_M.gguf", "diffusion_models", "diffusion_t2v"),
    CatalogItem("wan21_t2v14b_q6_k",   "city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q6_K.gguf",   "diffusion_models", "diffusion_t2v"),
    CatalogItem("wan21_t2v14b_q8_0",   "city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q8_0.gguf",   "diffusion_models", "diffusion_t2v"),

    CatalogItem("wan21_i2v14b_480p_q4_k_m", "city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q4_K_M.gguf", "diffusion_models", "diffusion_i2v"),
    CatalogItem("wan21_i2v14b_480p_q5_k_m", "city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q5_K_M.gguf", "diffusion_models", "diffusion_i2v"),
    CatalogItem("wan21_i2v14b_480p_q6_k",   "city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q6_K.gguf",   "diffusion_models", "diffusion_i2v"),
    CatalogItem("wan21_i2v14b_480p_q8_0",   "city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q8_0.gguf",   "diffusion_models", "diffusion_i2v"),

    # ---- UMT5 encoder GGUF (pick one; Q6/Q8 are typical) ----
    CatalogItem("umt5_xxl_q6_k", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q6_K.gguf", "text_encoders", "t5xxl"),
    CatalogItem("umt5_xxl_q8_0", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf", "text_encoders", "t5xxl"),

    # ---- Companion safetensors (required by Wan2.1 pipelines) ----
    CatalogItem("wan21_vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae", "vae"),
    CatalogItem("clip_vision_h", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision", "clip_vision"),
]


DEFAULT_BUNDLE_T2V = ["wan21_t2v14b_q4_k_m", "umt5_xxl_q6_k", "wan21_vae", "clip_vision_h"]
DEFAULT_BUNDLE_I2V = ["wan21_i2v14b_480p_q4_k_m", "umt5_xxl_q6_k", "wan21_vae", "clip_vision_h"]


def catalog_by_key() -> Dict[str, CatalogItem]:
    return {c.key: c for c in CATALOG}


# ------------------------------ Paths -----------------------------------

def project_root_from_arg(root: Optional[str]) -> Path:
    if root:
        return Path(root).resolve()
    # default: two levels up from presets/extra_env/
    return Path(__file__).resolve().parents[2]


def models_root(root: Path) -> Path:
    return root / "models" / "wan21gguf"


def sdcpp_root(root: Path) -> Path:
    return root / ".wan21gguf_env" / "sdcpp"


def ensure_dirs(root: Path) -> None:
    (root / ".wan21gguf_env").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "presets" / "setsave").mkdir(parents=True, exist_ok=True)
    (root / "presets" / "extra_env").mkdir(parents=True, exist_ok=True)

    mr = models_root(root)
    for sub in ["diffusion_models", "text_encoders", "vae", "clip_vision", "loras"]:
        (mr / sub).mkdir(parents=True, exist_ok=True)


# ------------------------------ Download helpers -------------------------

def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        with tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def download_catalog_item(root: Path, key: str) -> Path:
    c = catalog_by_key().get(key)
    if not c:
        raise SystemExit(f"Unknown key: {key}")

    mr = models_root(root)
    target_dir = mr / c.subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    # If filename includes subfolders, preserve only the leaf name in our folder.
    leaf = Path(c.filename).name
    dest = target_dir / leaf
    if dest.exists():
        print(f"[OK] Already exists: {dest}")
        return dest

    print(f"[DL] {c.repo_id} :: {c.filename}")
    print(f" -> {dest}")

    url = _hf_resolve_url(c.repo_id, c.filename)
    _download_stream(url, dest, timeout=120)
    return dest


def download_bundle(root: Path, keys: List[str]) -> None:
    for k in keys:
        download_catalog_item(root, k)


# ------------------------------ sd.cpp downloader ------------------------

def github_latest_release_assets(owner: str, repo: str) -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("assets", [])


def pick_windows_asset(assets: List[dict]) -> Optional[Tuple[str, str]]:
    """
    Return (name, url) of a good Windows x64 zip containing sd-cli.exe.
    Preference: avx2 > avx > generic.
    """
    # best-effort heuristics
    candidates = []
    for a in assets:
        name = a.get("name", "") or ""
        url = a.get("browser_download_url", "") or ""
        if not name.lower().endswith(".zip"):
            continue
        if "win" not in name.lower():
            continue
        # stable-diffusion.cpp zips are usually "sd-...-bin-win-...-x64.zip"
        score = 0
        n = name.lower()
        if "x64" in n or "x86_64" in n:
            score += 10
        if "avx2" in n:
            score += 5
        elif "avx" in n:
            score += 3
        if "bin" in n:
            score += 2
        candidates.append((score, name, url))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, name, url = candidates[0]
    return name, url


def ensure_sdcpp(root: Path) -> Path:
    out_dir = sdcpp_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdcli = None
    for cand in ["sd-cli.exe", "bin/Release/sd-cli.exe", "bin\\Release\\sd-cli.exe"]:
        p = out_dir / cand
        if p.exists():
            sdcli = p
            break
    if sdcli:
        print(f"[OK] sd-cli already present: {sdcli}")
        return sdcli

    print("[DL] Fetching stable-diffusion.cpp latest release assets...")
    assets = github_latest_release_assets("leejet", "stable-diffusion.cpp")
    pick = pick_windows_asset(assets)
    if not pick:
        raise SystemExit("Could not find a suitable Windows build asset in latest release.")

    name, url = pick
    zip_path = out_dir / name
    print(f"[DL] {name}")
    _download_file(url, zip_path)

    print("[EXTRACT] ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Try locate sd-cli.exe
    found = list(out_dir.rglob("sd-cli.exe"))
    if not found:
        raise SystemExit(f"Downloaded/extracted build but could not find sd-cli.exe under: {out_dir}")

    sdcli = found[0]
    # For convenience, place a copy at out_dir/sd-cli.exe
    if sdcli.parent != out_dir:
        shutil.copy2(sdcli, out_dir / "sd-cli.exe")
        sdcli = out_dir / "sd-cli.exe"

    print(f"[OK] sd-cli ready: {sdcli}")
    return sdcli


# ------------------------------ Settings --------------------------------

def write_default_settings(root: Path) -> Path:
    settings_path = root / "presets" / "setsave" / "wan21gguf.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        print(f"[OK] Settings already exist: {settings_path}")
        return settings_path

    data = {
        "backend": "gguf",
        "mode": "t2v",
        "width": 640,
        "height": 360,
        "num_frames": 49,
        "fps": 16,
        "guidance_scale": 6.0,
        "num_inference_steps": 12,
        "use_random_seed": True,
        "seed": 0,
        "gguf": {
            "t2v_model_key": "wan21_t2v14b_q4_k_m",
            "i2v_model_key": "wan21_i2v14b_480p_q4_k_m",
            "t5_key": "umt5_xxl_q6_k",
            "vae_key": "wan21_vae",
            "clip_vision_key": "clip_vision_h",
        },
        "output_dir": "output/wan21",
    }
    settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] Wrote default settings: {settings_path}")
    return settings_path


# ------------------------------ CLI -------------------------------------

def cmd_list(_: argparse.Namespace) -> None:
    print("WAN 2.1 GGUF catalog keys:\n")
    for c in CATALOG:
        print(f"- {c.key:26s}  ({c.kind})  {c.repo_id} :: {c.filename}")
    print("\nBundles:")
    print("  bundle_t2v_default  -> " + ", ".join(DEFAULT_BUNDLE_T2V))
    print("  bundle_i2v_default  -> " + ", ".join(DEFAULT_BUNDLE_I2V))


def cmd_ensure_modeldirs(args: argparse.Namespace) -> None:
    root = project_root_from_arg(args.root)
    ensure_dirs(root)
    print(f"[OK] Ensured model directories at: {models_root(root)}")


def cmd_write_default_settings(args: argparse.Namespace) -> None:
    root = project_root_from_arg(args.root)
    ensure_dirs(root)
    write_default_settings(root)


def cmd_ensure_sdcpp(args: argparse.Namespace) -> None:
    root = project_root_from_arg(args.root)
    ensure_dirs(root)
    ensure_sdcpp(root)


def cmd_download(args: argparse.Namespace) -> None:
    root = project_root_from_arg(args.root)
    ensure_dirs(root)

    key = args.key.strip()
    if key == "bundle_t2v_default":
        download_bundle(root, DEFAULT_BUNDLE_T2V)
        return
    if key == "bundle_i2v_default":
        download_bundle(root, DEFAULT_BUNDLE_I2V)
        return

    c = catalog_by_key().get(key)
    if not c:
        raise SystemExit(f"Unknown key: {key}")

    # If user asked for a diffusion model, auto-pull required companions (best default).
    keys_to_get = [key]
    if args.with_deps and c.kind in ("diffusion_t2v", "diffusion_i2v"):
        # Pull common companions. We never overwrite files already present on disk.
        if getattr(args, "t5_key", None):
            keys_to_get.append(args.t5_key)
        if getattr(args, "vae_key", None):
            keys_to_get.append(args.vae_key)
        # CLIP-Vision is required for I2V and harmless to have for T2V; users expect the stack to be complete.
        if getattr(args, "clip_vision_key", None):
            keys_to_get.append(args.clip_vision_key)

    # Deduplicate, keep order.
    seen = set()
    uniq = []
    for k in keys_to_get:
        if k not in seen:
            uniq.append(k)
            seen.add(k)

    download_bundle(root, uniq)
    print("[OK] Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wan21_guff.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_list = sub.add_parser("list", help="List catalog keys")
    s_list.set_defaults(func=cmd_list)

    s_md = sub.add_parser("ensure-modeldirs", help="Create model folders under /models/wan21gguf/")
    s_md.add_argument("--root", default=None, help="Project root (auto-detected if omitted)")
    s_md.set_defaults(func=cmd_ensure_modeldirs)

    s_sd = sub.add_parser("ensure-sdcpp", help="Download stable-diffusion.cpp Windows build (sd-cli.exe)")
    s_sd.add_argument("--root", default=None, help="Project root (auto-detected if omitted)")
    s_sd.set_defaults(func=cmd_ensure_sdcpp)

    s_ws = sub.add_parser("write-default-settings", help="Write presets/setsave/wan21gguf.json if missing")
    s_ws.add_argument("--root", default=None, help="Project root (auto-detected if omitted)")
    s_ws.set_defaults(func=cmd_write_default_settings)

    s_dl = sub.add_parser("download", help="Download one model by key (optionally with required companions)")
    s_dl.add_argument("key", help="Catalog key (or bundle_t2v_default / bundle_i2v_default)")
    s_dl.add_argument("--root", default=None, help="Project root (auto-detected if omitted)")
    s_dl.add_argument("--with-deps", action="store_true", help="Auto-download companions (UMT5 + VAE + CLIP-Vision if needed)")
    s_dl.add_argument("--t5-key", default="umt5_xxl_q6_k")
    s_dl.add_argument("--vae-key", default="wan21_vae")
    s_dl.add_argument("--clip-vision-key", default="clip_vision_h")
    s_dl.set_defaults(func=cmd_download)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
