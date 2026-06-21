#!/usr/bin/env python3
"""
background_download.py — v1.5

Downloads only the FrameVision background-removal models.

Final location (project root):
- Background models: models/bg

Downloads:
- MODNet ONNX (required)
- BiRefNet ONNX (optional via --pro, or included in default)
- RMBG-1.4 ONNX / FP16 ONNX (optional via --rmbg or --all)

Notes:
- Pre-checks destination folders before downloading.
- Adds --force to re-download even if destination already has the file.
- Does not create or repair any Python environment.
- Keeps --sd15-inpaint as a no-op compatibility flag so old launchers do not crash,
  but it no longer downloads any SDXL/Juggernaut inpaint model.
"""

from __future__ import annotations

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

# Ensure Windows consoles using cp1252 do not crash on Unicode output.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VERSION = "v1.5"


# ------------------------------
# Paths and helpers
# ------------------------------

def _looks_like_root(path: pathlib.Path) -> bool:
    return (path / "helpers").exists() or (path / "presets").exists() or (path / "models").exists()


def _get_project_root() -> pathlib.Path:
    """Resolve FrameVision root from scripts/, presets/extra_env/, or current working dir."""
    this = pathlib.Path(__file__).resolve()
    candidates: list[pathlib.Path] = []

    # Optional installs normally run from FrameVision root.
    candidates.append(pathlib.Path.cwd().resolve())

    # Legacy layout: <root>/scripts/background_download.py
    if this.parent.name.lower() == "scripts":
        candidates.append(this.parent.parent)

    # Current optional-install layout: <root>/presets/extra_env/background_download.py
    if this.parent.name.lower() == "extra_env" and this.parent.parent.name.lower() == "presets":
        candidates.append(this.parent.parent.parent)

    # Fallbacks.
    candidates.append(this.parent)
    candidates.append(this.parent.parent)

    for cand in candidates:
        try:
            cand = cand.resolve()
        except Exception:
            continue
        if _looks_like_root(cand):
            return cand

    return pathlib.Path.cwd().resolve()


ROOT = _get_project_root()
ROOT_MODELS = ROOT / "models"
ROOT_BG = ROOT_MODELS / "bg"

# Legacy location used by some older layouts (scripts/models/*)
SCRIPTS_MODELS = pathlib.Path(__file__).resolve().parent / "models"


def _human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n_bytes)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:,.1f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024
    return f"{n_bytes} B"


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
    speed = downloaded / elapsed

    if total:
        pct = downloaded / total
        eta = (total - downloaded) / max(1e-6, speed)
        bar_w = max(10, min(30, width - 60))
        filled = int(bar_w * pct)
        bar = "#" * filled + "-" * (bar_w - filled)
        line = (
            f"{prefix} |{bar}| {pct*100:5.1f}%  "
            f"{_human(downloaded)} / {_human(total)}  {_human(int(speed))}/s  ETA {_eta_str(eta)}"
        )
    else:
        line = f"{prefix} {_human(downloaded)}  {_human(int(speed))}/s"

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
                chunk = r.read(1024 * 1024)
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
                print("      Requires HF token and sometimes license acceptance. Set HF_TOKEN or pass --hf-token.")
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


def _same_path(a: pathlib.Path, b: pathlib.Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)


def _looks_like_model_file(path: pathlib.Path, model: dict) -> bool:
    """Accept only plausible files for THIS model, never just any file in the folder."""
    if (not path.exists()) or (not path.is_file()):
        return False
    name = path.name
    lower = name.lower()
    # Do not count interrupted downloads or temporary leftovers as installed models.
    if lower.endswith(".part") or lower.endswith(".tmp") or lower.endswith(".download"):
        return False

    patterns = [model.get("filename", ""), *(model.get("present_globs", []) or [])]
    if not any(fnmatch.fnmatch(name, pat) for pat in patterns if pat):
        return False

    min_bytes = int(model.get("min_bytes") or 1)
    try:
        if path.stat().st_size < min_bytes:
            print(f"[warn] Ignoring suspicious small file for {model.get('filename')}: {path.name} ({_human(path.stat().st_size)})")
            return False
    except Exception:
        return False
    return True


def find_existing_model(root: pathlib.Path, model: dict) -> pathlib.Path | None:
    """Find a plausible already-installed file for this model by exact name or safe alias."""
    exact = root / model["filename"]
    if _looks_like_model_file(exact, model):
        return exact
    for pat in model.get("present_globs", []) or []:
        for p in sorted(root.glob(pat)):
            if _looks_like_model_file(p, model):
                return p
    return None


# ------------------------------
# Background-removal model sources
# ------------------------------

MODELS = {
    "modnet_onnx": {
        "url": [
            "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx?download=true",
            "https://github.com/Technopagan/ai-assets/releases/download/models/modnet_photographic_portrait_matting.onnx",
            "https://raw.githubusercontent.com/Technopagan/ai-assets/main/models/modnet_photographic_portrait_matting.onnx",
        ],
        "sha256": "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
        "filename": "modnet_photographic_portrait_matting.onnx",
        "present_globs": ["modnet*.onnx", "*portrait*matting*.onnx"],
        "min_bytes": 10 * 1024 * 1024,
        "size_hint": "~ 100 MB",
        "optional": False,
    },
    "birefnet_onnx": {
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx",
        "sha256": None,
        "filename": "BiRefNet-COD-epoch_125.onnx",
        "present_globs": ["BiRefNet-COD-epoch_125.onnx", "BiRefNet*.onnx", "birefnet*.onnx"],
        "min_bytes": 100 * 1024 * 1024,
        "size_hint": "~ 900 MB",
        "optional": True,
    },
    "rmbg14_onnx": {
        # BRIA RMBG-1.4 original ONNX. The Hugging Face filename is model.onnx,
        # but FrameVision saves it with a unique name so it cannot collide with
        # other models called model.onnx in models/bg.
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true",
        "sha256": None,
        "filename": "RMBG-14.onnx",
        "present_globs": ["RMBG-14.onnx", "rmbg-14.onnx", "RMBG_14.onnx", "rmbg_14.onnx"],
        "min_bytes": 100 * 1024 * 1024,
        "size_hint": "~ 176 MB",
        "optional": True,
    },
    "rmbg14_fp16_onnx": {
        # BRIA RMBG-1.4 FP16 ONNX. Saved separately from the FP32/original model.
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_fp16.onnx?download=true",
        "sha256": None,
        "filename": "RMBG-14_fp16.onnx",
        "present_globs": ["RMBG-14_fp16.onnx", "rmbg-14_fp16.onnx", "RMBG_14_fp16.onnx", "rmbg_14_fp16.onnx"],
        "min_bytes": 50 * 1024 * 1024,
        "size_hint": "~ 88 MB",
        "optional": True,
    },
}


def _target_root_for_model(_m: dict) -> pathlib.Path:
    return ROOT_BG


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
    pats: list[tuple[str, pathlib.Path]] = []
    for m in MODELS.values():
        target = _target_root_for_model(m)
        pats.append((m["filename"], target))
        for pat in m.get("present_globs", []) or []:
            pats.append((pat, target))
    return pats


KNOWN_PATTERNS = _build_known_patterns()


def reconcile_legacy_models() -> None:
    """Move known old downloads from scripts/models or scripts/models/bg into models/bg."""
    legacy = SCRIPTS_MODELS
    if not legacy.exists():
        return

    ROOT_MODELS.mkdir(parents=True, exist_ok=True)
    ROOT_BG.mkdir(parents=True, exist_ok=True)

    candidates: list[pathlib.Path] = []
    for d in [legacy, legacy / "bg"]:
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

    try:
        if (legacy / "bg").exists():
            try:
                (legacy / "bg").rmdir()
            except Exception:
                pass
        legacy.rmdir()
        if moved_any:
            print(f"[clean] Deleted folder {legacy}")
    except Exception:
        pass


def sweep_temp_to_targets(temp_dest: pathlib.Path) -> int:
    """Move known background model files from a temp folder into models/bg.

    Important: when --dest is the final models/bg folder, the source and destination
    can be the same path. In that case we must not delete the file we just downloaded.
    """
    if not temp_dest.exists():
        return 0

    moved = 0
    for p in temp_dest.glob("*"):
        if not p.is_file():
            continue

        matched_model = None
        target_root = None
        for m in MODELS.values():
            if _looks_like_model_file(p, m):
                matched_model = m
                target_root = _target_root_for_model(m)
                break
        if matched_model is None or target_root is None:
            continue

        dst = target_root / matched_model["filename"]
        if _same_path(p, dst):
            # Already in the final folder under the expected name. Leave it alone.
            continue

        existing = find_existing_model(target_root, matched_model)
        if existing is not None:
            print(f"[skip] {matched_model['filename']} already present as {existing.name}; leaving temp duplicate: {p}")
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
        help="Temporary download folder. Known files are moved into models/bg at the end.",
    )
    ap.add_argument("--pro", action="store_true", help="Include BiRefNet (~900 MB) when not using --all/--only")
    ap.add_argument("--rmbg", action="store_true", help="Include RMBG-1.4 ONNX models when not using --all/--only")
    ap.add_argument("--rmbg-fp32", action="store_true", help="Include only the RMBG-1.4 original/FP32 ONNX model when not using --all/--only")
    ap.add_argument("--rmbg-fp16", action="store_true", help="Include only the RMBG-1.4 FP16 ONNX model when not using --all/--only")
    ap.add_argument(
        "--sd15-inpaint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--all", action="store_true", help="Download all background-removal models; also the default when no flags are passed")
    ap.add_argument("--only", nargs="+", choices=list(MODELS.keys()), help="Download only these model keys (space-separated)")
    ap.add_argument("--hf-token", default=None, help="Hugging Face access token (overrides HF_TOKEN/HUGGINGFACE_TOKEN env vars)")
    ap.add_argument("--ignore-errors", action="store_true", help="Return success even if some downloads fail")
    ap.add_argument("--force", action="store_true", help="Re-download files even if they seem present already")
    args = ap.parse_args()

    print(f"[background_download] {VERSION}")
    print(f"[root] {ROOT.resolve()}")

    # Reconcile any legacy model locations before presence checks.
    reconcile_legacy_models()

    temp_dest = pathlib.Path(args.dest)
    if not temp_dest.is_absolute():
        temp_dest = ROOT / temp_dest
    temp_dest.mkdir(parents=True, exist_ok=True)
    ROOT_BG.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    all_keys = list(MODELS.keys())
    default_keys = ["modnet_onnx", "birefnet_onnx"]

    if args.only:
        to_get = list(args.only)
        reason = "--only"
    elif args.all:
        to_get = all_keys
        reason = "--all"
    elif not any((args.pro, args.rmbg, args.rmbg_fp32, args.rmbg_fp16, args.force, args.hf_token)):
        # Keep the old default lightweight: MODNet + BiRefNet only.
        # RMBG is still experimental for the FrameVision UI, so it stays opt-in.
        to_get = list(default_keys)
        reason = "default"
    else:
        to_get = ["modnet_onnx"]
        reason = "flags"
        if args.pro:
            to_get.append("birefnet_onnx")
        if args.rmbg or args.rmbg_fp32:
            to_get.append("rmbg14_onnx")
        if args.rmbg or args.rmbg_fp16:
            to_get.append("rmbg14_fp16_onnx")

    # Preserve order while removing duplicates from combined flags.
    to_get = list(dict.fromkeys(to_get))

    print(f"[select] reason={reason} -> {to_get}")

    any_fail = False

    for key in to_get:
        m = MODELS[key]
        target_root = _target_root_for_model(m)
        target_root.mkdir(parents=True, exist_ok=True)

        out = temp_dest / m["filename"]
        final_exact = target_root / m["filename"]

        if not args.force:
            existing_final = find_existing_model(target_root, m)
            if existing_final is not None:
                print(f"[skip] {m['filename']} already present in {target_root.relative_to(ROOT)} as {existing_final.name}")
                continue

            # If a previous complete download sits in a separate temp folder, move it
            # instead of downloading again. Do not count random files as success.
            if (not _same_path(out, final_exact)) and _looks_like_model_file(out, m):
                if _safe_move(out, final_exact):
                    print(f"[move] Reused completed temp download: {out} -> {final_exact}")
                    continue
                print(f"[warn] Found completed temp file but could not move it: {out}")

        print(f"[get ] {m['filename']}  {m['size_hint']}  -> {target_root.relative_to(ROOT)}")
        try:
            fetch_with_fallback(m["url"], out, expected_sha256=m["sha256"], token=token)
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

    moved = sweep_temp_to_targets(temp_dest)
    if moved:
        print(f"[move] Swept {moved} file(s) from {temp_dest} into models/bg")

    if any_fail and not args.ignore_errors:
        print("[done] Completed with errors.", file=sys.stderr)
        return 1

    print("[done] Background-removal models ready:")
    print("       bg =", ROOT_BG.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
