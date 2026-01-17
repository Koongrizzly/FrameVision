#!/usr/bin/env python3
"""
background_download.py — v1.1

Extracted from downloadbg.py (v3.9) and kept ONLY the background-removal + inpaint models.

Final locations (project root):
- Background models:  models/bg
- Inpaint models:     models/inpaint

Downloads:
- MODNet ONNX (required)
- BiRefNet ONNX (optional via --pro, or included in --all/default)
- SDXL Juggernaut Inpaint safetensors (optional via --sd15-inpaint, or included in --all/default)

Notes:
- Pre-check destination folders BEFORE any download (prevents re-fetching huge files)
- Adds --force to re-download even if destination already has the file
- Supports Hugging Face tokens via --hf-token, HF_TOKEN, or HUGGINGFACE_TOKEN
"""

import argparse
import fnmatch
import hashlib
import math
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request

# Ensure Windows consoles using cp1252 don't crash on Unicode output
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


VERSION = "v1.1"

# ------------------------------
# Paths and helpers
# ------------------------------

def _get_project_root() -> pathlib.Path:
    """
    If the script is inside a 'scripts' folder, project root is its parent.
    Otherwise, project root is the script folder.
    """
    this = pathlib.Path(__file__).resolve()
    scripts_dir = this.parent
    return scripts_dir.parent if scripts_dir.name.lower() == "scripts" else scripts_dir

ROOT = _get_project_root()
ROOT_MODELS = ROOT / "models"
ROOT_BG = ROOT_MODELS / "bg"
ROOT_INPAINT = ROOT_MODELS / "inpaint"

# Legacy location used by some older layouts (scripts/models/*)
SCRIPTS_MODELS = pathlib.Path(__file__).resolve().parent / "models"


def _human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n_bytes)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:,.1f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024

def _eta_str(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def sha256sum(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _build_request(url: str, token: str | None):
    headers = {
        "User-Agent": f"background_download/{VERSION} (python-urllib)",
        "Accept": "*/*",
    }
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return urllib.request.Request(url, headers=headers)

def _print_progress(prefix: str, downloaded: int, total: int | None, t0: float):
    width = 100
    try:
        width = max(40, min((shutil.get_terminal_size().columns or 80), 140))
    except Exception:
        pass

    elapsed = max(1e-6, time.time() - t0)
    speed = downloaded / elapsed  # bytes/sec

    if total:
        pct = downloaded / total
        eta = (total - downloaded) / max(1e-6, speed)
        bar_w = max(10, min(30, width - 60))
        filled = int(bar_w * pct)
        bar = "#" * filled + "-" * (bar_w - filled)
        line = (
            f"{prefix} |{bar}| {pct*100:5.1f}%  "
            f"{_human(downloaded)} / {_human(total)}  {_human(speed)}/s  ETA {_eta_str(eta)}"
        )
    else:
        line = f"{prefix} {_human(downloaded)}  {_human(speed)}/s"

    sys.stdout.write("\x1b[2K\r" + line[:width])
    sys.stdout.flush()

def _fetch_single(url: str, dest_path: pathlib.Path, expected_sha256: str | None = None, token: str | None = None):
    req = _build_request(url, token)
    tmp = str(dest_path) + ".part"
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None
            downloaded = 0
            t0 = time.time()
            last = 0.0
            _print_progress("[dl  ]", 0, total, t0)

            while True:
                chunk = r.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last >= 0.25 or (total and downloaded >= total):
                    last = now
                    _print_progress("[dl  ]", downloaded, total, t0)

            _print_progress("[dl  ]", downloaded, total, t0)
            sys.stdout.write("\n")
            sys.stdout.flush()
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

    os.replace(tmp, dest_path)

    if expected_sha256:
        got = sha256sum(dest_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(f"SHA256 mismatch for {dest_path.name}: {got} != {expected_sha256}")

def fetch_with_fallback(url_or_list, dest_path: pathlib.Path, expected_sha256: str | None = None, token: str | None = None):
    urls = url_or_list if isinstance(url_or_list, (list, tuple)) else [url_or_list]
    last_err = None

    for i, url in enumerate(urls, 1):
        try:
            print(f"[try ] {dest_path.name}  mirror {i}/{len(urls)}")
            _fetch_single(url, dest_path, expected_sha256=expected_sha256, token=token)
            return
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"[warn] {dest_path.name} from mirror {i}: HTTP {e.code} {e.reason}")
            if e.code in (401, 403) and "huggingface.co" in url:
                print("      Requires HF token and (sometimes) license acceptance. Set HF_TOKEN or pass --hf-token.")
        except Exception as e:
            last_err = e
            print(f"[warn] {dest_path.name} from mirror {i}: {e}")

    if last_err:
        raise last_err

def exists_any(root: pathlib.Path, patterns: list[str]) -> bool:
    for pat in patterns:
        for _ in root.glob(pat):
            return True
    return False

def list_matches(root: pathlib.Path, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                hits.append(p.name)
    return sorted(set(hits))


# ------------------------------
# Background + inpaint model sources
# ------------------------------

MODELS = {
    "modnet_onnx": {
        "target": "bg",
        "url": [
            "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx?download=true",
            "https://github.com/Technopagan/ai-assets/releases/download/models/modnet_photographic_portrait_matting.onnx",
            "https://raw.githubusercontent.com/Technopagan/ai-assets/main/models/modnet_photographic_portrait_matting.onnx",
        ],
        "sha256": "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
        "filename": "modnet_photographic_portrait_matting.onnx",
        "size_hint": "~ 100 MB",
        "optional": False,
    },
    "birefnet_onnx": {
        "target": "bg",
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx",
        "sha256": None,
        "filename": "BiRefNet-COD-epoch_125.onnx",
        "size_hint": "~ 900 MB",
        "optional": True,
    },
    # Note: the flag name stays --sd15-inpaint for backward compatibility.
    "sd15_inpaint_fp16": {
        "target": "inpaint",
        "url": "https://civitai.com/api/download/models/449759?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "sha256": None,
        "filename": "SDXL-Juggernaut-V9RDphoto2-inpaint.safetensors",
        "size_hint": "~ 6.5 GB",
        "optional": True,
        "present_globs": ["SDXL-Juggernaut-V9RDphoto2*.*"],
    },
}


def _target_root_for_model(m: dict) -> pathlib.Path:
    return ROOT_INPAINT if str(m.get("target", "bg")).lower() == "inpaint" else ROOT_BG

def _other_root(target_root: pathlib.Path) -> pathlib.Path:
    return ROOT_BG if target_root.resolve() == ROOT_INPAINT.resolve() else ROOT_INPAINT

def _safe_move(src: pathlib.Path, dst: pathlib.Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                os.chmod(dst, 0o666)
                dst.unlink()
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"[warn] Move failed for {src.name}: {e}")
        return False

def _build_known_patterns() -> list[tuple[str, pathlib.Path]]:
    """Return [(pattern, target_root), ...] for strict filenames + present_globs."""
    pats: list[tuple[str, pathlib.Path]] = []
    for m in MODELS.values():
        target = _target_root_for_model(m)
        pats.append((m["filename"], target))
        for pat in m.get("present_globs", []) or []:
            pats.append((pat, target))
    return pats

KNOWN_PATTERNS = _build_known_patterns()


def reconcile_legacy_models() -> None:
    """
    Move legacy model files from scripts/models (or scripts/models/bg|inpaint) into
    project_root/models/bg or project_root/models/inpaint, but ONLY for files that match
    MODELS' filenames/patterns. This avoids moving unrelated files.
    """
    legacy = SCRIPTS_MODELS
    if not legacy.exists():
        return

    ROOT_MODELS.mkdir(parents=True, exist_ok=True)
    ROOT_BG.mkdir(parents=True, exist_ok=True)
    ROOT_INPAINT.mkdir(parents=True, exist_ok=True)

    # Collect candidate files from:
    # - scripts/models/*
    # - scripts/models/bg/*
    # - scripts/models/inpaint/*
    candidates: list[pathlib.Path] = []
    for d in [legacy, legacy / "bg", legacy / "inpaint"]:
        if d.exists() and d.is_dir():
            candidates.extend([p for p in d.glob("*") if p.is_file()])

    moved_any = False
    for p in candidates:
        target_root = None
        for pat, trg in KNOWN_PATTERNS:
            if fnmatch.fnmatch(p.name, pat):
                target_root = trg
                break
        if target_root is None:
            continue

        dest = target_root / p.name
        if dest.exists():
            print(f"[skip] {dest} already present")
            continue

        if _safe_move(p, dest):
            print(f"[move] {p} -> {dest}")
            moved_any = True

    # Best-effort cleanup (only if empty after moves)
    try:
        for d in [legacy / "bg", legacy / "inpaint"]:
            if d.exists():
                try:
                    d.rmdir()
                except Exception:
                    pass
        legacy.rmdir()
        if moved_any:
            print(f"[clean] Deleted folder {legacy}")
    except Exception:
        pass


def reconcile_wrong_folder_models() -> None:
    """
    If a known model was downloaded into the wrong final folder (bg vs inpaint),
    move it to the correct one.
    """
    ROOT_BG.mkdir(parents=True, exist_ok=True)
    ROOT_INPAINT.mkdir(parents=True, exist_ok=True)

    for m in MODELS.values():
        target = _target_root_for_model(m)
        other = _other_root(target)

        # exact filename
        src = other / m["filename"]
        dst = target / m["filename"]
        if src.exists() and not dst.exists():
            if _safe_move(src, dst):
                print(f"[move] {src} -> {dst}")

        # glob variants
        pats = m.get("present_globs", []) or []
        if pats:
            if exists_any(target, pats):
                continue
            for pat in pats:
                for p in other.glob(pat):
                    if not p.is_file():
                        continue
                    dstp = target / p.name
                    if dstp.exists():
                        continue
                    if _safe_move(p, dstp):
                        print(f"[move] {p} -> {dstp}")


def sweep_temp_to_targets(temp_dest: pathlib.Path) -> int:
    """Move any known model files found in temp_dest into their correct final folder."""
    if not temp_dest.exists():
        return 0

    moved = 0
    for p in temp_dest.glob("*"):
        if not p.is_file():
            continue

        target_root = None
        for pat, trg in KNOWN_PATTERNS:
            if fnmatch.fnmatch(p.name, pat):
                target_root = trg
                break
        if target_root is None:
            continue

        dst = target_root / p.name
        if dst.exists():
            try:
                p.unlink()
            except Exception:
                pass
            continue

        if _safe_move(p, dst):
            moved += 1
    return moved


# ------------------------------
# Main
# ------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dest",
        default="models/bg",
        help="Temporary download folder (files are moved into models/bg and/or models/inpaint at the end)",
    )
    ap.add_argument("--pro", action="store_true", help="Include BiRefNet (~900 MB) when not using --all/--only")
    ap.add_argument("--sd15-inpaint", action="store_true", help="Include SDXL-Juggernaut-V9RDphoto2-inpaint (~6.5 GB) when not using --all/--only")
    ap.add_argument("--all", action="store_true", help="Download ALL background/inpaint models (also the default when no flags are passed)")
    ap.add_argument("--only", nargs="+", choices=list(MODELS.keys()), help="Download only these model keys (space-separated)")
    ap.add_argument("--hf-token", default=None, help="Hugging Face access token (overrides HF_TOKEN/HUGGINGFACE_TOKEN env vars)")
    ap.add_argument("--ignore-errors", action="store_true", help="Return success even if some downloads fail")
    ap.add_argument("--force", action="store_true", help="Re-download files even if they seem present already")
    args = ap.parse_args()

    print(f"[background_download] {VERSION}")
    print(f"[root] {ROOT.resolve()}")

    # Reconcile any legacy model locations before presence checks
    reconcile_legacy_models()
    reconcile_wrong_folder_models()

    temp_dest = pathlib.Path(args.dest)
    temp_dest.mkdir(parents=True, exist_ok=True)
    ROOT_BG.mkdir(parents=True, exist_ok=True)
    ROOT_INPAINT.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    all_keys = list(MODELS.keys())
    selected_default_all = False

    if args.only:
        to_get = args.only
        reason = "--only"
    elif args.all or (not args.pro and not args.sd15_inpaint and not args.only and not args.force and not args.hf_token):
        # Match old behavior: no flags -> "all"
        # (The additional checks here avoid treating '--force' or '--hf-token' as "model selection flags".)
        to_get = all_keys
        reason = "--all(default)"
        selected_default_all = True
    else:
        # "lite" selection: always MODNet + optional extras
        to_get = ["modnet_onnx"]
        reason = "--pro/--sd15-inpaint"
        if args.pro:
            to_get.append("birefnet_onnx")
        if args.sd15_inpaint:
            to_get.append("sd15_inpaint_fp16")

    # If user passed --all explicitly, prefer it even if it overlaps other logic
    if args.all and not selected_default_all:
        to_get = all_keys
        reason = "--all"

    print(f"[select] reason={reason} -> {to_get}")

    any_fail = False

    for key in to_get:
        m = MODELS[key]
        target_root = _target_root_for_model(m)
        other_root = _other_root(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        out = temp_dest / m["filename"]
        final_exact = target_root / m["filename"]

        if not args.force:
            # temp dest present
            if out.exists():
                print(f"[skip] {out.name} already exists in {temp_dest}")
                continue

            # exact file present at correct final folder
            if final_exact.exists():
                print(f"[skip] {final_exact.name} already exists in {target_root.relative_to(ROOT)}")
                continue

            # exact file present in the wrong folder -> move it and continue
            wrong_exact = other_root / m["filename"]
            if wrong_exact.exists() and not final_exact.exists():
                if _safe_move(wrong_exact, final_exact):
                    print(f"[move] Fixed location: {wrong_exact} -> {final_exact}")
                    continue

            # extra globs for certain models (e.g., Juggernaut variants)
            present_globs = m.get("present_globs", []) or []
            if present_globs:
                if exists_any(target_root, present_globs):
                    hits = list_matches(target_root, present_globs)
                    print(f"[skip] Found existing files in {target_root.relative_to(ROOT)} matching {present_globs}: {hits}")
                    continue
                # if found in the wrong folder, move the first match (and continue)
                if exists_any(other_root, present_globs):
                    hits = list_matches(other_root, present_globs)
                    moved_one = False
                    for name in hits:
                        src = other_root / name
                        dst = target_root / name
                        if src.exists() and not dst.exists():
                            if _safe_move(src, dst):
                                print(f"[move] Fixed location: {src} -> {dst}")
                                moved_one = True
                    if moved_one:
                        continue

        print(f"[get ] {m['filename']}  {m['size_hint']}  -> {target_root.relative_to(ROOT)}")
        try:
            fetch_with_fallback(m["url"], out, expected_sha256=m["sha256"], token=token)
            # Move into correct final folder if temp_dest isn't already the target folder.
            if out.resolve() != final_exact.resolve():
                if _safe_move(out, final_exact):
                    print(f"[ ok ] Saved to {final_exact}")
                else:
                    raise RuntimeError(f"Downloaded but failed to move into {final_exact}")
            else:
                print(f"[ ok ] Saved to {out}")
        except Exception as e:
            any_fail = True
            print(f"[fail] {m['filename']}: {e}")

    # Sweep any known leftovers from temp_dest into the proper final folders
    moved = sweep_temp_to_targets(temp_dest)
    if moved:
        print(f"[move] Swept {moved} file(s) from {temp_dest} into final model folders")

    if any_fail and not args.ignore_errors:
        print("[done] Completed with errors.", file=sys.stderr)
        return 1

    print("[done] Models ready:")
    print("       bg     =", ROOT_BG.resolve())
    print("       inpaint=", ROOT_INPAINT.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
