#!/usr/bin/env python3
"""
downloadbg.py — v3.3
- Clean, single‑line progress (ETA, speed, human sizes)
- Moves finished files to models/bg, cleans stray models/*.zip
- Hugging Face auth support (token or env)
- **MODNet fallback mirrors**: tries each URL in order until one works
- **Default = ALL models** when no flags are passed

Flags:
    --all / --only <keys...> / --pro / --sd15-inpaint
    --hf-token TOKEN
    --ignore-errors
"""
import argparse, hashlib, os, sys, urllib.request, urllib.error, pathlib, shutil, time, math, shutil as _shutil

VERSION = "v3.3"

# Some entries provide a list of URLs (mirrors). We'll try them in order.
MODELS = {
    "modnet_onnx": {
        "url": [
            # Primary (Hugging Face mirror sometimes 404s; try alternates)
            "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx?download=true",
            # Alternate mirrors (community mirrors; feel free to replace with your own trusted mirrors)
            "https://github.com/Technopagan/ai-assets/releases/download/models/modnet_photographic_portrait_matting.onnx",
            "https://raw.githubusercontent.com/Technopagan/ai-assets/main/models/modnet_photographic_portrait_matting.onnx",
        ],
        "sha256": "07C308CF0FC7E6E8B2065A12ED7FC07E1DE8FEBB7DC7839D7B7F15DD66584DF9".lower(),
        "filename": "modnet_photographic_portrait_matting.onnx",
        "size_hint": "≈ 100 MB",
        "optional": False
    },
    "birefnet_onnx": {
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx",
        "sha256": None,
        "filename": "BiRefNet-COD-epoch_125.onnx",
        "size_hint": "≈ 900 MB",
        "optional": True
    },
    "sd15_inpaint_fp16": {
        # If you prefer the GitHub link you found, replace this with it.
        "url": "https://huggingface.co/webui/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.safetensors?download=true",
        "sha256": None,
        "filename": "sd-v1-5-inpainting.safetensors",
        "size_hint": "≈ 4.0 GB",
        "optional": True
    },
}

def _human(n_bytes: int) -> str:
    units = ["B","KB","MB","GB","TB"]
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

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _build_request(url: str, token: str | None):
    headers = {
        "User-Agent": f"downloadbg/{VERSION} (python-urllib)",
        "Accept": "*/*",
    }
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return urllib.request.Request(url, headers=headers)

def _print_progress(prefix: str, downloaded: int, total: int | None, t0: float):
    width = max(40, min((_shutil.get_terminal_size().columns or 80), 140))
    elapsed = max(1e-6, time.time() - t0)
    speed = downloaded / elapsed  # bytes/sec
    if total:
        pct = downloaded / total
        eta = (total - downloaded) / max(1e-6, speed)
        bar_w = max(10, min(30, width - 60))
        filled = int(bar_w * pct)
        bar = "#" * filled + "-" * (bar_w - filled)
        line = f"{prefix} |{bar}| {pct*100:5.1f}%  { _human(downloaded) } / { _human(total) }  { _human(speed) }/s  ETA { _eta_str(eta) }"
    else:
        line = f"{prefix} { _human(downloaded) }  { _human(speed) }/s"
    sys.stdout.write("\x1b[2K\r" + line[:width])
    sys.stdout.flush()

def _fetch_single(url, dest_path, expected_sha256=None, token: str | None = None):
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
            sys.stdout.write("\n"); sys.stdout.flush()
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

def fetch_with_fallback(url_or_list, dest_path, expected_sha256=None, token: str | None = None):
    urls = url_or_list if isinstance(url_or_list, (list, tuple)) else [url_or_list]
    last_err = None
    for i, url in enumerate(urls, 1):
        try:
            print(f"[try ] {dest_path.name}  mirror {i}/{len(urls)}")
            _fetch_single(url, dest_path, expected_sha256, token=token)
            return
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"[warn] {dest_path.name} from mirror {i}: HTTP {e.code} {e.reason}")
            if e.code in (401, 403) and "huggingface.co" in url:
                print("      Requires HF token and license acceptance. Set HF_TOKEN or pass --hf-token.")
        except Exception as e:
            last_err = e
            print(f"[warn] {dest_path.name} from mirror {i}: {e}")
    # If all mirrors failed
    if last_err:
        raise last_err

def move_all_to_root_models_bg(temp_dest: pathlib.Path) -> int:
    final_root = pathlib.Path("models/bg")
    final_root.mkdir(parents=True, exist_ok=True)
    try:
        same_dir = temp_dest.resolve() == final_root.resolve()
    except Exception:
        same_dir = False
    moved = 0
    for p in temp_dest.glob("*"):
        if not p.is_file():
            continue
        target = final_root / p.name
        try:
            if same_dir and p.resolve() == target.resolve():
                continue
        except Exception:
            pass
        if target.exists():
            try:
                target.unlink()
            except Exception:
                try:
                    os.chmod(target, 0o666)
                    target.unlink()
                except Exception as e:
                    print(f"[warn] Could not replace existing {target}: {e}")
                    continue
        try:
            shutil.move(str(p), str(target))
            moved += 1
        except Exception as e:
            print(f"[warn] Move failed for {p.name}: {e}")
    return moved

def cleanup_model_root_zips() -> int:
    root = pathlib.Path("models")
    if not root.exists():
        return 0
    deleted = 0
    for z in root.glob("*.zip"):
        try:
            z.unlink()
            deleted += 1
        except PermissionError:
            try:
                os.chmod(z, 0o666)
                z.unlink()
                deleted += 1
            except Exception as e:
                print(f"[warn] Could not delete {z.name}: {e}")
        except Exception as e:
            print(f"[warn] Could not delete {z.name}: {e}")
    return deleted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="models/bg", help="Temporary download folder (files will be moved to project-root models/bg at the end)")
    ap.add_argument("--pro", action="store_true", help="Include BiRefNet (≈900 MB) when not using --all/--only")
    ap.add_argument("--sd15-inpaint", action="store_true", help="Include SD 1.5 Inpainting (≈4.0 GB) when not using --all/--only")
    ap.add_argument("--all", action="store_true", help="Download ALL supported models")
    ap.add_argument("--only", nargs="+", choices=list(MODELS.keys()), help="Download only these model keys (space-separated)")
    ap.add_argument("--hf-token", default=None, help="Hugging Face access token (overrides HF_TOKEN/HUGGINGFACE_TOKEN env vars)")
    ap.add_argument("--ignore-errors", action="store_true", help="Return success even if some downloads fail")
    args = ap.parse_args()

    print(f"[downloadbg] {VERSION}")
    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    all_keys = list(MODELS.keys())
    if args.only:
        to_get = args.only
        reason = "--only"
    elif args.all or (not args.pro and not args.sd15_inpaint and not args.only):
        # Default to ALL
        to_get = all_keys
        reason = "--all(default)"
    else:
        to_get = ["modnet_onnx"]
        if args.pro:
            to_get.append("birefnet_onnx")
        if args.sd15_inpaint:
            to_get.append("sd15_inpaint_fp16")
        reason = "--pro/--sd15-inpaint"

    print(f"[select] reason={reason} -> {to_get}")
    any_fail = False
    for key in to_get:
        m = MODELS[key]
        out = dest / m["filename"]
        if out.exists():
            print(f"[skip] {out.name} already exists in {dest}")
            continue
        print(f"[get ] {out.name}  {m['size_hint']}")
        try:
            fetch_with_fallback(m["url"], out, m["sha256"], token=token)
        except Exception as e:
            any_fail = True
            print(f"[fail] {out.name}: {e}")
        else:
            print(f"[ ok ] Saved to {out}")

    moved = move_all_to_root_models_bg(dest)
    final_root = pathlib.Path("models/bg").resolve()
    if moved:
        print(f"[move] Moved {moved} file(s) to {final_root}")
    else:
        print(f"[move] No files moved; they were already in {final_root}")

    removed = cleanup_model_root_zips()
    if removed:
        print(f"[clean] Removed {removed} zip file(s) from {pathlib.Path('models').resolve()}")
    else:
        print(f"[clean] No zip files to remove in {pathlib.Path('models').resolve()}")

    if any_fail and not args.ignore_errors:
        print("[done] Completed with errors.", file=sys.stderr)
        return 1

    print("[done] Background models ready in", final_root)
    return 0

if __name__ == "__main__":
    sys.exit(main())
