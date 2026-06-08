#!/usr/bin/env python3
"""
FrameVision helper: download UnsLoTH FLUX.2-klein UNet (GGUF) + optional VAE + optional text encoder.

NEW:
  - Supports both FLUX.2-klein-4B and FLUX.2-klein-9B.
  - In interactive mode, you pick 4B vs 9B at the start.

Important:
- The UnsLoTH repo you linked contains the *UNet diffusion model* in GGUF form.
- Flux/Klein also requires a VAE and a text encoder, which are typically hosted separately.
  This script lets you download those *only if you choose them* (nothing huge auto-downloads).

Where to place this script:
  <FRAMEVISION_ROOT>/presets/extra_env/klein_gguf_download.py

Where it downloads to (kept fully portable inside FrameVision):
  <FRAMEVISION_ROOT>/models/klein4b_gguf/   (if variant=4B)
  <FRAMEVISION_ROOT>/models/klein9b_gguf/   (if variant=9B)
    unet/           (the FLUX.2-klein-4B GGUF files)
    vae/            (flux2-vae.safetensors)
    text_encoders/  (one or more text encoders you select)

Usage:
  python presets/extra_env/klein_gguf_download.py --variant 4b --list
  python presets/extra_env/klein_gguf_download.py --variant 9b --list
  python presets/extra_env/klein_gguf_download.py --variant 9b --download-unet Q4_K_M
  python presets/extra_env/klein_gguf_download.py --variant 9b --download-textenc gguf:Q5_K_M
  python presets/extra_env/klein_gguf_download.py --download-vae
  python presets/extra_env/klein_gguf_download.py --interactive

Auth (optional):
  - set HF_TOKEN env var if you hit rate limits or need auth:
      set HF_TOKEN=hf_...
Notes:
  - Cordux text encoder repo is *gated* (requires accepting terms on Hugging Face).
    If listing/downloading fails, open the repo page in a browser and accept the terms.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def _norm_variant(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("4", "4b", "klein4b", "klein-4b"):
        return "4b"
    if v in ("9", "9b", "klein9b", "klein-9b"):
        return "9b"
    raise ValueError("Variant must be 4b or 9b")

UNET_REV  = "main"

# VAE (single file):
VAE_REPO = "Comfy-Org/flux2-dev"
VAE_REV  = "main"
VAE_FILE = "split_files/vae/flux2-vae.safetensors"

# Text encoder for Klein 4B (Qwen3 4B), safetensors (Comfy-Org bundle):
TE_COMFY_REPO = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
TE_COMFY_REV  = "main"
TE_COMFY_PREFIX = "split_files/text_encoders/"

# Text encoder GGUF repos:
TE_GGUF_4B_REPO = "unsloth/Qwen3-4B-GGUF"
TE_GGUF_9B_REPO = "unsloth/Qwen3-8B-GGUF"  # requested: use these files as text encoders for 9B
TE_GGUF_REV  = "main"

# Requested curated lists for 9B (keeps list clean & matches your requirements)
KLEIN9B_UNET_FILES = [
    "flux-2-klein-9b-Q2_K.gguf",
    "flux-2-klein-9b-Q3_K_M.gguf",
    "flux-2-klein-9b-Q4_K_M.gguf",
    "flux-2-klein-9b-Q5_K_M.gguf",
    "flux-2-klein-9b-Q6_K.gguf",
    "flux-2-klein-9b-Q8_0.gguf",
]

KLEIN9B_TE_GGUF_FILES = [
    "Qwen3-8B-Q2_K.gguf",
    "Qwen3-8B-Q4_K_M.gguf",
    "Qwen3-8B-Q5_K_M.gguf",
    "Qwen3-8B-Q6_K.gguf",
    "Qwen3-8B-Q8_0.gguf",
]

# Text encoder option: Cordux "uncensored" (GATED repo)
TE_CORDUX_REPO = "Cordux/flux2-klein-4B-uncensored-text-encoder"
TE_CORDUX_REV  = "main"

# Repo “small files” (downloaded automatically; excludes large model weights unless selected)
SMALL_ALLOW_PATTERNS = [
    "README.md", "LICENSE*", ".gitattributes",
    "assets/*", "*.png", "*.jpg", "*.jpeg", "*.webp",
]

def _framevision_root_from_script(script_path: Path) -> Path:
    # Expect: <root>/presets/extra_env/klein4b_gguf_download.py -> root is parents[2]
    try:
        return script_path.resolve().parents[2]
    except Exception:
        return Path.cwd().resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes or num_bytes <= 0:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{num_bytes} B"

def _try_import_hfhub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        return HfApi, hf_hub_download, snapshot_download
    except Exception:
        return None

def _list_repo_files(repo_id: str, revision: str, suffix: Optional[str]=None, prefix: Optional[str]=None) -> List[Dict]:
    imported = _try_import_hfhub()
    if not imported:
        raise RuntimeError(
            "huggingface_hub is not installed in this Python environment.\n"
            "Install it in your FrameVision extra env, e.g.:\n"
            "  pip install -U huggingface_hub\n"
        )
    HfApi, _, _ = imported
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    out: List[Dict] = []
    for sib in getattr(info, "siblings", []) or []:
        name = getattr(sib, "rfilename", None) or getattr(sib, "path", None)
        size = getattr(sib, "size", None)
        if not name:
            continue
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.lower().endswith(suffix.lower()):
            continue
        out.append({"name": name, "size": size})
    out.sort(key=lambda x: x["name"].lower())
    return out

def _print_component_list(variant: str, unet_gguf: List[Dict], te_comfy: List[Dict], te_gguf: List[Dict], te_cordux: List[Dict], cordux_note: str) -> None:
    print(f"\n=== UNet (FLUX.2-klein-{variant.upper()}) GGUF files ===")
    if not unet_gguf:
        print("  (none found)")
    else:
        for i, f in enumerate(unet_gguf, start=1):
            print(f"  [U{i:02d}] {f['name']}   ({_human_size(f.get('size'))})")

    print("\n=== Text encoder (Comfy-Org safetensors) ===")
    if not te_comfy:
        print("  (none found)")
    else:
        for i, f in enumerate(te_comfy, start=1):
            print(f"  [C{i:02d}] {f['name']}   ({_human_size(f.get('size'))})")

    te_label = "Unsloth Qwen3-4B" if variant == "4b" else "Unsloth Qwen3-8B"
    print(f"\n=== Text encoder ({te_label}) GGUF ===")
    if not te_gguf:
        print("  (none found)")
    else:
        for i, f in enumerate(te_gguf, start=1):
            print(f"  [G{i:02d}] {f['name']}   ({_human_size(f.get('size'))})")

    print("\n=== Text encoder (Cordux uncensored) ===")
    if cordux_note:
        print(f"  NOTE: {cordux_note}")
    if not te_cordux:
        print("  (none listed)")
    else:
        for i, f in enumerate(te_cordux, start=1):
            print(f"  [X{i:02d}] {f['name']}   ({_human_size(f.get('size'))})")

    print("\n=== VAE ===")
    print(f"  [V]  {VAE_FILE}   (from {VAE_REPO})")

def _select_by_tokens(files: List[Dict], tokens: List[str], label: str) -> List[Dict]:
    """
    tokens can be:
      - indices: "1", "2", ...
      - exact filenames
      - substrings: e.g. "Q4_K_M"
    """
    chosen: List[Dict] = []
    by_name = {f["name"]: f for f in files}

    def add(f: Dict):
        if f not in chosen:
            chosen.append(f)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= len(files):
                add(files[idx - 1])
                continue
            raise ValueError(f"{label}: index out of range: {tok} (1..{len(files)})")

        if tok in by_name:
            add(by_name[tok])
            continue

        t = tok.lower()
        hits = [f for f in files if t in f["name"].lower()]
        if not hits:
            raise ValueError(f"{label}: no match for token: {tok}")
        for f in hits:
            add(f)

    chosen.sort(key=lambda x: x["name"].lower())
    return chosen

def _download_small_repo_files(repo_id: str, revision: str, dest_dir: Path) -> None:
    imported = _try_import_hfhub()
    if not imported:
        raise RuntimeError("huggingface_hub is required to download from Hugging Face.")
    _, _, snapshot_download = imported

    _ensure_dir(dest_dir)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        allow_patterns=SMALL_ALLOW_PATTERNS,
        ignore_patterns=["*.gguf", "*.safetensors", "*.bin", "*.pt"],
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )

def _download_file(repo_id: str, revision: str, filename: str, dest_dir: Path) -> Path:
    imported = _try_import_hfhub()
    if not imported:
        raise RuntimeError("huggingface_hub is required to download from Hugging Face.")
    _, hf_hub_download, _ = imported

    _ensure_dir(dest_dir)
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return Path(out)

def main() -> int:
    p = argparse.ArgumentParser(
        description="Download FLUX.2-klein (4B/9B) GGUF UNet + optional VAE + optional text encoder (selective; portable paths)."
    )
    p.add_argument("--variant", type=str, default="4b", help="Pick Klein variant: 4b or 9b. Default: 4b.")
    p.add_argument("--list", action="store_true", help="List UNet GGUFs + text encoder options + VAE.")
    p.add_argument("--download-unet", type=str, default="", help="UNet GGUF selection: indices/filenames/substrings (comma-separated).")
    p.add_argument("--download-textenc", type=str, default="", help="Text encoder selection. Use 'comfy:<...>' or 'gguf:<...>' or 'cordux:<...>' (e.g. comfy:1 or gguf:Q4_K_M or cordux:1).")
    p.add_argument("--download-vae", action="store_true", help="Download the Flux2 VAE (flux2-vae.safetensors).")
    p.add_argument("--all-unet", action="store_true", help="Download ALL UNet GGUFs (explicit action).")
    p.add_argument("--skip-repo", action="store_true", help="Skip downloading small repo files (README/LICENSE/assets).")
    p.add_argument("--interactive", action="store_true", help="Interactive picker (recommended).")
    args = p.parse_args()

    variant_explicit = any(a.startswith("--variant") for a in sys.argv[1:])

    try:
        variant = _norm_variant(args.variant)
    except Exception as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    # Variant-specific repos + base folders
    UNET_REPO = "unsloth/FLUX.2-klein-4B-GGUF" if variant == "4b" else "unsloth/FLUX.2-klein-9B-GGUF"
    TE_GGUF_REPO = TE_GGUF_4B_REPO if variant == "4b" else TE_GGUF_9B_REPO

    fv_root = _framevision_root_from_script(Path(__file__))
    base = fv_root / "models" / ("klein4b_gguf" if variant == "4b" else "klein9b_gguf")
    unet_dir = base / "unet"
    te_dir = base / "text_encoders"
    vae_dir = base / "vae"

    print(f"FrameVision root: {fv_root}")
    print(f"Base folder:      {base}")

    try:
        unet_gguf = _list_repo_files(UNET_REPO, UNET_REV, suffix=".gguf")
        te_comfy: List[Dict] = []
        if variant == "4b":
            te_comfy = _list_repo_files(TE_COMFY_REPO, TE_COMFY_REV, suffix=".safetensors", prefix=TE_COMFY_PREFIX)
        te_gguf  = _list_repo_files(TE_GGUF_REPO, TE_GGUF_REV, suffix=".gguf")
    except Exception as e:
        print(f"\nERROR listing repos: {e}\n", file=sys.stderr)
        return 2

    # Curate 9B lists to exactly what you requested (keeps output tidy)
    if variant == "9b":
        unet_by_name = {f["name"]: f for f in unet_gguf}
        unet_gguf = [unet_by_name.get(n, {"name": n, "size": None}) for n in KLEIN9B_UNET_FILES]

        te_by_name = {f["name"]: f for f in te_gguf}
        te_gguf = [te_by_name.get(n, {"name": n, "size": None}) for n in KLEIN9B_TE_GGUF_FILES]

    # Cordux is gated; list may fail unless terms accepted (and/or token provided).
    te_cordux: List[Dict] = []
    cordux_note = ""
    try:
        # Cordux repo contains GGUF(s) and/or other weights; list both .gguf and .safetensors
        te_cordux = _list_repo_files(TE_CORDUX_REPO, TE_CORDUX_REV)
        # Keep just plausible weight files for display/selection:
        te_cordux = [f for f in te_cordux if f["name"].lower().endswith((".gguf", ".safetensors"))]
    except Exception as e:
        cordux_note = f"Could not list (likely gated). Open the repo in a browser and accept terms, then retry. Details: {e}"

    if args.list:
        _print_component_list(variant, unet_gguf, te_comfy, te_gguf, te_cordux, cordux_note)
        return 0

    if not args.skip_repo:
        try:
            print("\nDownloading small repo files (metadata only, no large weights)...")
            _download_small_repo_files(UNET_REPO, UNET_REV, base)
        except Exception as e:
            print(f"\nERROR downloading small repo files: {e}\n", file=sys.stderr)
            return 3

    # Interactive mode if requested or if no actions were passed
    if args.interactive or (not args.all_unet and not args.download_unet and not args.download_textenc and not args.download_vae):
        # Requested behavior: at start, let user choose 4B vs 9B (unless --variant was provided).
        if args.interactive and not variant_explicit:
            try:
                pick = input("Select Flux Klein variant (4B/9B) [4B]: ").strip()
                if pick:
                    variant = _norm_variant(pick)
            except KeyboardInterrupt:
                print("\nCanceled.")
                return 0

            # If user changed variant, recompute repos + base + lists
            UNET_REPO = "unsloth/FLUX.2-klein-4B-GGUF" if variant == "4b" else "unsloth/FLUX.2-klein-9B-GGUF"
            TE_GGUF_REPO = TE_GGUF_4B_REPO if variant == "4b" else TE_GGUF_9B_REPO
            base = fv_root / "models" / ("klein4b_gguf" if variant == "4b" else "klein9b_gguf")
            unet_dir = base / "unet"
            te_dir = base / "text_encoders"
            vae_dir = base / "vae"
            print(f"\nFrameVision root: {fv_root}")
            print(f"Base folder:      {base}")

            try:
                unet_gguf = _list_repo_files(UNET_REPO, UNET_REV, suffix=".gguf")
                te_comfy = []
                if variant == "4b":
                    te_comfy = _list_repo_files(TE_COMFY_REPO, TE_COMFY_REV, suffix=".safetensors", prefix=TE_COMFY_PREFIX)
                te_gguf = _list_repo_files(TE_GGUF_REPO, TE_GGUF_REV, suffix=".gguf")
            except Exception as e:
                print(f"\nERROR listing repos: {e}\n", file=sys.stderr)
                return 2

            if variant == "9b":
                unet_by_name = {f["name"]: f for f in unet_gguf}
                unet_gguf = [unet_by_name.get(n, {"name": n, "size": None}) for n in KLEIN9B_UNET_FILES]
                te_by_name = {f["name"]: f for f in te_gguf}
                te_gguf = [te_by_name.get(n, {"name": n, "size": None}) for n in KLEIN9B_TE_GGUF_FILES]

        _print_component_list(variant, unet_gguf, te_comfy, te_gguf, te_cordux, cordux_note)

        try:
            raw_unet = input("\nSelect UNet GGUFs (comma tokens like 1,3 or Q4_K_M). Blank to skip: ").strip()
            raw_te   = input("Select Text Encoder (prefix comfy:, gguf:, or cordux:). Example 'comfy:1' / 'gguf:Q4_K_M' / 'cordux:1'. Blank to skip: ").strip()
            raw_vae  = input("Download VAE? (y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\nCanceled.")
            return 0

        if raw_unet:
            args.download_unet = raw_unet
        if raw_te:
            args.download_textenc = raw_te
        if raw_vae in ("y", "yes", "1", "true"):
            args.download_vae = True

    # --- UNet downloads ---
    selections_unet: List[Dict] = []
    if args.all_unet:
        selections_unet = list(unet_gguf)
    elif args.download_unet.strip():
        tokens = [t.strip() for t in args.download_unet.split(",") if t.strip()]
        try:
            selections_unet = _select_by_tokens(unet_gguf, tokens, "UNet")
        except Exception as e:
            print(f"\nUNet selection error: {e}\nTip: use --list to see options.", file=sys.stderr)
            return 4

    # --- Text encoder downloads ---
    selections_te: List[Tuple[str, Dict]] = []  # (source, file)
    if args.download_textenc.strip():
        raw = args.download_textenc.strip()
        if ":" not in raw:
            print("\nText encoder selection must start with comfy: or gguf: or cordux:\n  --download-textenc comfy:1\n  --download-textenc gguf:Q4_K_M\n  --download-textenc cordux:1\n", file=sys.stderr)
            return 4
        src, rest = raw.split(":", 1)
        src = src.strip().lower()
        rest = rest.strip()
        if not rest:
            print("\nText encoder selection missing after prefix (comfy:/gguf:/cordux:).", file=sys.stderr)
            return 4

        if src == "comfy":
            tokens = [t.strip() for t in rest.split(",") if t.strip()]
            try:
                chosen = _select_by_tokens(te_comfy, tokens, "TextEnc(comfy)")
                selections_te = [("comfy", f) for f in chosen]
            except Exception as e:
                print(f"\nTextEnc(comfy) selection error: {e}\nTip: use --list to see options.", file=sys.stderr)
                return 4
        elif src == "gguf":
            tokens = [t.strip() for t in rest.split(",") if t.strip()]
            try:
                chosen = _select_by_tokens(te_gguf, tokens, "TextEnc(gguf)")
                selections_te = [("gguf", f) for f in chosen]
            except Exception as e:
                print(f"\nTextEnc(gguf) selection error: {e}\nTip: use --list to see options.", file=sys.stderr)
                return 4
        elif src == "cordux":
            # List may be empty if gated; still allow user to specify exact filename.
            if not te_cordux:
                # If user typed a filename, attempt direct download.
                # Treat each token as exact filename (or substring match on a best-effort list if any).
                tokens = [t.strip() for t in rest.split(",") if t.strip()]
                for t in tokens:
                    selections_te.append(("cordux", {"name": t, "size": None}))
            else:
                tokens = [t.strip() for t in rest.split(",") if t.strip()]
                try:
                    chosen = _select_by_tokens(te_cordux, tokens, "TextEnc(cordux)")
                    selections_te = [("cordux", f) for f in chosen]
                except Exception as e:
                    print(f"\nTextEnc(cordux) selection error: {e}\nTip: use --list to see options (after accepting the gate).", file=sys.stderr)
                    return 4
        else:
            print("\nUnknown text encoder source. Use comfy: or gguf: or cordux:.", file=sys.stderr)
            return 4

    # Summary
    print("\nPlanned downloads:")
    if selections_unet:
        print("  UNet GGUF:")
        for f in selections_unet:
            print(f"    - {f['name']} ({_human_size(f.get('size'))})")
    else:
        print("  UNet GGUF: (none)")

    if selections_te:
        print("  Text Encoder:")
        for src, f in selections_te:
            print(f"    - [{src}] {f['name']} ({_human_size(f.get('size'))})")
    else:
        print("  Text Encoder: (none)")

    if args.download_vae:
        print(f"  VAE: {VAE_FILE} (from {VAE_REPO})")
    else:
        print("  VAE: (none)")

    if not selections_unet and not selections_te and not args.download_vae:
        print("\nNothing selected. Exiting.")
        return 0

    errors = 0

    # Download UNet GGUFs
    for f in selections_unet:
        try:
            print(f"\nDownloading UNet: {f['name']}")
            out = _download_file(UNET_REPO, UNET_REV, f["name"], unet_dir)
            print(f"Saved: {out}")
        except Exception as e:
            errors += 1
            print(f"\nERROR downloading UNet {f['name']}: {e}\n", file=sys.stderr)

    # Download Text Encoder(s)
    for src, f in selections_te:
        try:
            if src == "comfy":
                repo, rev = TE_COMFY_REPO, TE_COMFY_REV
            elif src == "gguf":
                repo, rev = TE_GGUF_REPO, TE_GGUF_REV
            else:
                repo, rev = TE_CORDUX_REPO, TE_CORDUX_REV
            print(f"\nDownloading Text Encoder [{src}]: {f['name']}")
            out = _download_file(repo, rev, f["name"], te_dir)
            print(f"Saved: {out}")
        except Exception as e:
            errors += 1
            print(f"\nERROR downloading Text Encoder [{src}] {f['name']}: {e}\n", file=sys.stderr)
            if src == "cordux":
                print("Cordux is gated — open the repo in a browser, accept the terms, and (optionally) set HF_TOKEN before retrying.", file=sys.stderr)

    # Download VAE
    if args.download_vae:
        try:
            print(f"\nDownloading VAE: {VAE_FILE}")
            out = _download_file(VAE_REPO, VAE_REV, VAE_FILE, vae_dir)
            print(f"Saved: {out}")
        except Exception as e:
            errors += 1
            print(f"\nERROR downloading VAE: {e}\n", file=sys.stderr)

    if errors:
        print(f"\nDone with {errors} error(s).", file=sys.stderr)
        return 5

    print("\nAll selected downloads complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
