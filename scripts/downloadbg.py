#!/usr/bin/env python3

"""
downloadbg.py — v3.8
- Pre-check destination folders BEFORE any download (prevents re-fetching huge files)
- Reconciliation FIRST: move scripts/models -> project root models/, then clean
- Adds --force to re-download even if destination already has the file
- UltraSharp NCNN (Hugging Face) + SRMD (GitHub) upscaler downloads to models/realesrgan
- NEW: Treat any file matching 'sd-v1-5-inpainting*.*' in models/bg as present (skip download)
- Default = ALL models when no flags passed
"""
import argparse, hashlib, os, sys, urllib.request, urllib.error, pathlib, shutil, time, math, zipfile, io

VERSION = "v3.8"

# ------------------------------
# Paths and helpers
# ------------------------------

def _get_project_root() -> pathlib.Path:
    this = pathlib.Path(__file__).resolve()
    scripts_dir = this.parent
    return scripts_dir.parent if scripts_dir.name.lower() == "scripts" else scripts_dir

ROOT = _get_project_root()
ROOT_MODELS = ROOT / "models"
ROOT_BG = ROOT_MODELS / "bg"
ROOT_REALESRGAN = ROOT_MODELS / "realesrgan"
SCRIPTS_MODELS = pathlib.Path(__file__).resolve().parent / "models"

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
    if last_err:
        raise last_err

def move_all_to(temp_dest: pathlib.Path, final_root: pathlib.Path) -> int:
    final_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in temp_dest.glob("*"):
        if not p.is_file():
            continue
        target = final_root / p.name
        try:
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    os.chmod(target, 0o666)
                    target.unlink()
            shutil.move(str(p), str(target))
            moved += 1
        except Exception as e:
            print(f"[warn] Move failed for {p.name}: {e}")
    return moved

def cleanup_model_root_zips() -> int:
    root = ROOT_MODELS
    if not root.exists():
        return 0
    deleted = 0
    for z in root.glob("*.zip"):
        try:
            z.unlink()
            deleted += 1
        except Exception as e:
            print(f"[warn] Could not delete {z.name}: {e}")
    return deleted

def reconcile_scripts_models() -> None:
    legacy = SCRIPTS_MODELS
    if not legacy.exists():
        return
    ROOT_MODELS.mkdir(parents=True, exist_ok=True)
    ROOT_BG.mkdir(parents=True, exist_ok=True)

    for p in list(legacy.glob("*")):
        try:
            if p.is_dir():
                dest = ROOT_MODELS / p.name
                if not dest.exists():
                    shutil.move(str(p), str(dest))
                    print(f"[move] {p} -> {dest}")
                else:
                    print(f"[skip] {dest} already present")
                continue
            if p.suffix.lower() == ".zip":
                dest = ROOT_MODELS / p.name
                if not dest.exists():
                    shutil.move(str(p), str(dest))
                    print(f"[move] {p} -> {dest}")
                else:
                    print(f"[skip] {dest} already present")
            else:
                dest = ROOT_BG / p.name
                if not dest.exists():
                    shutil.move(str(p), str(dest))
                    print(f"[move] {p} -> {dest}")
                else:
                    print(f"[skip] {dest} already present")
        except Exception as e:
            print(f"[warn] Could not reconcile {p}: {e}")
    try:
        shutil.rmtree(legacy)
        print(f"[clean] Deleted folder {legacy}")
    except Exception as e:
        print(f"[warn] Could not delete {legacy}: {e}")

def exists_any(root: pathlib.Path, patterns: list[str]) -> bool:
    for pat in patterns:
        for _ in root.glob(pat):
            return True
    return False

def list_matches(root: pathlib.Path, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                hits.append(p.name)
    return sorted(set(hits))

# ------------------------------
# Model sources
# ------------------------------

MODELS = {
    "modnet_onnx": {
        "url": [
            "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx?download=true",
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
        "url": "https://huggingface.co/webui/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.safetensors?download=true",
        "sha256": None,
        "filename": "sd-v1-5-inpainting.safetensors",
        "size_hint": "≈ 4.0 GB",
        "optional": True,
        # NEW: treat alt names as present
        "present_globs": ["sd-v1-5-inpainting*.*"]
    },
    "realsr_ncnn_zip": {
        "url": "https://github.com/nihui/realsr-ncnn-vulkan/releases/download/20220728/realsr-ncnn-vulkan-20220728-windows.zip",
        "sha256": None,
        "filename": "realsr-ncnn-vulkan-20220728-windows.zip",
        "size_hint": "≈ 20 MB",
        "optional": False,
        "extract_to": "models",
        "exists_path": "models/realsr-ncnn-vulkan-20220728-windows"
    }
}

# Upscaler extras
ULTRASHARP_FILES = [
    "4x-UltraSharp-fp16.bin",
    "4x-UltraSharp-fp16.param",
    "4x-UltraSharp-fp32.bin",
    "4x-UltraSharp-fp32.param",
]
ULTRASHARP_BASE = "https://huggingface.co/Kim2091/UltraSharp/resolve/main/NCNN"

def download_ultrasharp_ncnn(temp_dest: pathlib.Path, token: str | None = None) -> tuple[int,int]:
    temp_dest.mkdir(parents=True, exist_ok=True)
    ROOT_REALESRGAN.mkdir(parents=True, exist_ok=True)
    # If ALL files already exist in final folder, skip
    all_present = all((ROOT_REALESRGAN / f).exists() for f in ULTRASHARP_FILES)
    if all_present:
        print(f"[skip] UltraSharp NCNN already present in {ROOT_REALESRGAN}")
        return (0,0)
    downloaded = 0
    for fname in ULTRASHARP_FILES:
        final_path = ROOT_REALESRGAN / fname
        if final_path.exists():
            print(f"[skip] {fname} already exists in {ROOT_REALESRGAN}")
            continue
        out = temp_dest / fname
        if out.exists():
            print(f"[skip] {out.name} already exists in {temp_dest}")
        else:
            url = f"{ULTRASHARP_BASE}/{fname}?download=true"
            try:
                print(f"[get ] {fname} (UltraSharp NCNN)")
                fetch_with_fallback(url, out, expected_sha256=None, token=token)
                print(f"[ ok ] Saved to {out}")
                downloaded += 1
            except Exception as e:
                print(f"[fail] {fname}: {e}")
    moved = move_all_to(temp_dest, ROOT_REALESRGAN)
    if moved:
        print(f"[move] UltraSharp -> {ROOT_REALESRGAN} ({moved} file(s))")
    return (downloaded, moved)

def download_srmd_models(temp_dest: pathlib.Path) -> tuple[int,int]:
    temp_dest.mkdir(parents=True, exist_ok=True)
    ROOT_REALESRGAN.mkdir(parents=True, exist_ok=True)
    zip_name = "srmd-ncnn-vulkan-master.zip"
    zip_path = temp_dest / zip_name
    repo_zip_url = "https://github.com/nihui/srmd-ncnn-vulkan/archive/refs/heads/master.zip"
    try:
        print(f"[get ] SRMD models (GitHub archive)")
        fetch_with_fallback(repo_zip_url, zip_path, expected_sha256=None, token=None)
        print(f"[ ok ] Saved to {zip_path}")
    except Exception as e:
        print(f"[fail] SRMD archive: {e}")
        return (0,0)

    extracted = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            sub_prefix = "srmd-ncnn-vulkan-master/models/models-srmd/"
            select = [m for m in members if m.startswith(sub_prefix) and (m.endswith(".bin") or m.endswith(".param")) and not m.endswith("/") ]
            if not select:
                print(f"[warn] SRMD archive: no files found in {sub_prefix}")
            for m in select:
                fname = os.path.basename(m)
                final_path = ROOT_REALESRGAN / fname
                if final_path.exists():
                    print(f"[skip] {fname} already exists in {ROOT_REALESRGAN}")
                    continue
                data = zf.read(m)
                out = temp_dest / fname
                with open(out, "wb") as f:
                    f.write(data)
                extracted += 1
        try:
            zip_path.unlink()
        except Exception:
            pass
    except Exception as e:
        print(f"[fail] Extract SRMD archive: {e}")
        return (0,0)

    moved = move_all_to(temp_dest, ROOT_REALESRGAN)
    if moved:
        print(f"[move] SRMD -> {ROOT_REALESRGAN} ({moved} file(s))")
    return (extracted, moved)

# ------------------------------
# Main
# ------------------------------

def main():
    import shutil as _shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="models/bg", help="Temporary download folder for background models (files will be moved to project-root models/bg at the end)")
    ap.add_argument("--pro", action="store_true", help="Include BiRefNet (≈900 MB) when not using --all/--only")
    ap.add_argument("--sd15-inpaint", action="store_true", help="Include SD 1.5 Inpainting (≈4.0 GB) when not using --all/--only")
    ap.add_argument("--realsr", action="store_true", help="Include RealSR 2x/4x (realsr-ncnn-vulkan) zip")
    ap.add_argument("--ultrasharp", action="store_true", help="Download UltraSharp NCNN files to models/realesrgan")
    ap.add_argument("--srmd", action="store_true", help="Download SRMD models (nihui) to models/realesrgan")
    ap.add_argument("--all", action="store_true", help="Download ALL supported models (includes UltraSharp+SRMD)")
    ap.add_argument("--only", nargs="+", choices=list(MODELS.keys()), help="Download only these model keys (space-separated)")
    ap.add_argument("--hf-token", default=None, help="Hugging Face access token (overrides HF_TOKEN/HUGGINGFACE_TOKEN env vars)")
    ap.add_argument("--ignore-errors", action="store_true", help="Return success even if some downloads fail")
    ap.add_argument("--force", action="store_true", help="Re-download files even if they seem present already")
    args = ap.parse_args()

    print(f"[downloadbg] {VERSION}")

    # 1) Reconcile legacy first so presence checks see the files in their final place
    reconcile_scripts_models()

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # 2) Determine selection
    all_keys = list(MODELS.keys())
    selected_default_all = False
    if args.only:
        to_get = args.only
        reason = "--only"
    elif args.all or (not args.pro and not args.sd15_inpaint and not args.only and not args.realsr and not args.ultrasharp and not args.srmd):
        to_get = all_keys
        reason = "--all(default)"
        selected_default_all = True
    else:
        to_get = ["modnet_onnx"]
        reason = "--pro/--sd15-inpaint"
        if args.pro:
            to_get.append("birefnet_onnx")
        if args.sd15_inpaint and "sd15_inpaint_fp16" not in to_get:
            to_get.append("sd15_inpaint_fp16")
        if args.realsr and "realsr_ncnn_zip" not in to_get:
            to_get.append("realsr_ncnn_zip")
    if args.realsr and "realsr_ncnn_zip" not in to_get:
        to_get.append("realsr_ncnn_zip")
        reason += "+realsr"

    print(f"[select] reason={reason} -> {to_get}")

    # 3) Pre-checks before downloading any background model
    any_fail = False
    for key in to_get:
        m = MODELS[key]
        out = dest / m["filename"]

        # Skip logic (unless --force)
        if not args.force:
            # temp dest present
            if out.exists():
                print(f"[skip] {out.name} already exists in {dest}")
                continue

            # exact file present at final bg folder
            final_exact = ROOT_BG / m["filename"]
            if final_exact.exists():
                print(f"[skip] {final_exact.name} already exists in models/bg")
                continue

            # extra globs for certain models (e.g., sd-v1-5-inpainting*.*)
            present_globs = m.get("present_globs", [])
            if present_globs and exists_any(ROOT_BG, present_globs):
                hits = list_matches(ROOT_BG, present_globs)
                print(f"[skip] Found existing files in models/bg matching {present_globs}: {hits}")
                continue

            # extracted path present (e.g., realsr zip)
            exists_path = m.get("exists_path")
            if exists_path:
                exists_path = (ROOT / exists_path) if not pathlib.Path(exists_path).is_absolute() else pathlib.Path(exists_path)
                if exists_path.exists():
                    print(f"[skip] {exists_path} already present")
                    continue

        # If we arrive here, proceed with fetching
        print(f"[get ] {out.name}  {m['size_hint']}")
        try:
            fetch_with_fallback(m["url"], out, m["sha256"], token=token)
        except Exception as e:
            any_fail = True
            print(f"[fail] {out.name}: {e}")
            continue
        print(f"[ ok ] Saved to {out}")

        # Unzip & delete, if applicable
        if m.get("extract_to") and out.suffix.lower() == ".zip":
            try:
                target_root = (ROOT / m["extract_to"]) if not pathlib.Path(m["extract_to"]).is_absolute() else pathlib.Path(m["extract_to"])
                _ = 0
                with zipfile.ZipFile(out, 'r') as zf:
                    members = zf.namelist()
                    print(f"[unzip] {out.name} -> {target_root.resolve()} ({len(members)} entries)")
                    zf.extractall(target_root)
                try:
                    out.unlink()
                    print(f"[clean] Deleted zip {out.name} after extraction")
                except Exception as e:
                    print(f"[warn] Could not delete zip {out.name}: {e}")
            except Exception as e:
                any_fail = True
                print(f"[fail] Extract {out.name}: {e}")

    # 4) Move any remaining tmp bg files into final bg folder
    moved_bg = move_all_to(dest, ROOT_BG)
    final_bg = ROOT_BG.resolve()
    if moved_bg:
        print(f"[move] Moved {moved_bg} file(s) to {final_bg}")
    else:
        print(f"[move] No files moved; they were already in {final_bg}")

    removed = cleanup_model_root_zips()
    if removed:
        print(f"[clean] Removed {removed} zip file(s) from {ROOT_MODELS.resolve()}")
    else:
        print(f"[clean] No zip files to remove in {ROOT_MODELS.resolve()}")

    # 5) Upscaler extras (UltraSharp + SRMD)
    include_ultrasharp = args.ultrasharp or selected_default_all or args.all
    include_srmd = args.srmd or selected_default_all or args.all
    realesr_tmp = pathlib.Path("scripts") / "_tmp_realesrgan"
    try:
        if include_ultrasharp:
            dl, mv = download_ultrasharp_ncnn(realesr_tmp, token=token)
            if dl == 0 and mv == 0:
                print("[info] UltraSharp NCNN done (nothing to do)")
        if include_srmd:
            ex, mv = download_srmd_models(realesr_tmp)
            if ex == 0 and mv == 0:
                print("[info] SRMD models done (nothing to do)")
    finally:
        try:
            if realesr_tmp.exists():
                import shutil as _shutil2
                _shutil2.rmtree(realesr_tmp)
        except Exception:
            pass

    if any_fail and not args.ignore_errors:
        print("[done] Completed with errors.", file=sys.stderr)
        return 1

    print("[done] Background + upscaler models ready in", ROOT_BG.resolve(), "and", ROOT_REALESRGAN.resolve())
    return 0

if __name__ == "__main__":
    sys.exit(main())
