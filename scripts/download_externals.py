from __future__ import annotations
import os
import urllib.request, urllib.error, sys, io, json, time, shutil, zipfile, argparse, urllib.request, glob
from pathlib import Path
from typing import Optional, Any, List
# --- begin: FFmpeg & RIFE installers (minimal) ---
import zipfile as _zipfile
import urllib.request as _url
from pathlib import Path as _P

def _fv_dl(_urlsrc: str, _dst: _P) -> bool:
    try:
        print("[externals] download:", _urlsrc)
        _dst.parent.mkdir(parents=True, exist_ok=True)
        with _url.urlopen(_urlsrc, timeout=180) as r, open(_dst, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print("[externals] download failed:", e, "@", _urlsrc)
        try: _dst.unlink()
        except Exception: pass
        return False

def ensure_ffmpeg_bins():
    # Root: parent of this script (…/scripts/download_externals.py -> project root)
    root = (globals().get("ROOT") or _P(__file__).resolve().parent.parent)
    bin_dir = root / "presets" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    targets = ["ffmpeg.exe", "ffplay.exe", "ffprobe.exe"]
    if all((bin_dir / t).exists() for t in targets):
        print("[externals] ffmpeg bins already present")
        return
    cache = root / ".dl_cache"
    cache.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
    zpath = cache / "ffmpeg-master-latest-win64-gpl.zip"
    if not zpath.exists() and not _fv_dl(url, zpath):
        print("[externals] FFmpeg ZIP fetch failed; skipping")
        return
    with _zipfile.ZipFile(zpath, "r") as z:
        names = z.namelist()
        for w in targets:
            cand = [n for n in names if n.lower().endswith("/"+w) or n.lower().endswith("\\"+w) or n.lower().endswith(w)]

            if cand:
                with z.open(cand[0]) as src, open(bin_dir / w, "wb") as out:
                    out.write(src.read())
    print("[externals] ffmpeg bins installed to", bin_dir)

def ensure_rife_pack():
    root = (globals().get("ROOT") or _P(__file__).resolve().parent.parent)
    target = root / "models" / "rife-ncnn-vulkan"
    want_dirs = {"rife", "rife-anime", "rife-HD", "rife-UHD", "rife-v4.6", "rife-v4"}
    want_files = {"rife-ncnn-vulkan.exe", "vcomp140.dll", "LICENSE", "README.md"}
    if (target / "rife-ncnn-vulkan.exe").exists():
        print("[externals] RIFE pack already present")
        return
    cache = root / ".dl_cache"
    cache.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"
    zpath = cache / "rife-ncnn-vulkan-20221029-windows.zip"
    if not zpath.exists() and not _fv_dl(url, zpath):
        print("[externals] RIFE ZIP fetch failed; skipping")
        return
    target.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(zpath, "r") as z:
        for name in z.namelist():
            norm = name.replace("\\", "/")
            base = norm.split("/")[-1]
            parts = norm.split("/")
            # copy wanted top-level files
            if base in want_files and not norm.endswith("/"):
                out = target / base
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as f:
                    f.write(src.read())
            # copy everything under wanted dirs
            for d in list(want_dirs):
                if f"/{d}/" in "/" + norm and not norm.endswith("/"):
                    idx = parts.index(d)
                    sub = "/".join(parts[idx:])
                    out = target / sub
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as src, open(out, "wb") as f:
                        f.write(src.read())
                    break
    print("[externals] RIFE pack installed to", target)
# --- end: FFmpeg & RIFE installers (minimal) ---

def move_folders_to_models():
    """
    Mirrors these batch commands on Windows (no-ops on non-Windows):
        if not exist "models" mkdir "models"
        robocopy ".\externals" ".\models" /E /MOVE /R:2 /W:0 >nul
        robocopy ".\.cache" ".\models" /E /MOVE /R:2 /W:0 >nul
        robocopy ".\.dl_cache" ".\models" /E /MOVE /R:2 /W:0 >nul
        exit /b 0
    """
    import sys, subprocess, shutil
    root = (globals().get("ROOT") or _P(__file__).resolve().parent.parent)
    models = root / "models"
    import os
    VERBOSE = os.environ.get("FV_VERBOSE_MOVES", "").strip() not in ("", "0", "false", "False")
    models.mkdir(parents=True, exist_ok=True)

    # Only attempt robocopy on Windows
    if sys.platform.startswith("win"):
        sources = [root / "externals", root / ".cache", root / ".dl_cache"]
        for src_dir in sources:
            if not src_dir.exists():
                continue
            try:
                cmd = ["robocopy", str(src_dir), str(models), "/E", "/MOVE", "/R:2", "/W:0", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"]
                print("[externals] robocopy:", " ".join(cmd)) if VERBOSE else None
                # robocopy uses special exit codes: 0-7 indicate success (>=8 failure)
                import subprocess as _sp
                completed = _sp.run(cmd, shell=False, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                code = completed.returncode
                if code >= 8:
                    print(f"[externals] robocopy failed ({code}), falling back to Python move for", src_dir) if VERBOSE else None
                    # Fallback: move the tree manually
                    for path in src_dir.iterdir():
                        dst = models / path.name
                        if path.is_dir():
                            # merge move
                            if dst.exists():
                                # move contents
                                for child in path.iterdir():
                                    shutil.move(str(child), str(dst / child.name))
                                # remove now-empty dir
                                try: path.rmdir()
                                except Exception: pass
                            else:
                                shutil.move(str(path), str(dst))
                        else:
                            shutil.move(str(path), str(dst))
                    # remove empty src_dir
                    try: src_dir.rmdir()
                    except Exception: pass
            except FileNotFoundError:
                # robocopy missing; fallback to Python move
                print("[externals] robocopy not found; falling back to Python move for", src_dir) if VERBOSE else None
                if src_dir.exists():
                    for path in src_dir.iterdir():
                        dst = models / path.name
                        if path.is_dir():
                            shutil.move(str(path), str(dst))
                        else:
                            shutil.move(str(path), str(dst))
            except Exception as e:
                print("[externals] move error:", e) if VERBOSE else None
    else:
        print("[externals] move skipped (non-Windows platform)") if VERBOSE else None


# >>> FRAMEVISION_QWEN_BEGIN
# Qwen20B local registrar/validator (no network)
import os, sys, json

def _qwen20b_validate(path):
    req_files = [
        "qwen_image_20B_quanto_bf16_int8.safetensors",
        "qwen_vae.safetensors",
        "qwen_image_20B.json",
        "qwen_vae_config.json",
        "qwen_scheduler_config.json",
    ]
    missing = [f for f in req_files if not os.path.exists(os.path.join(path, f))]
    pipe_py = os.path.join(path, "qwen", "pipeline_qwenimage.py")
    ok_pipe = os.path.exists(pipe_py)
    if missing or not ok_pipe:
        print("[qwen20b] missing:", ", ".join(missing) if missing else "pipeline_qwenimage.py not found")
        raise SystemExit(2)
    print("[qwen20b] OK at", path)
    raise SystemExit(0)

# Pre-argparse intercept (insert-only; does not alter other flags)
if "--component" in sys.argv:
    try:
        i = sys.argv.index("--component")
        comp = sys.argv[i+1] if i+1 < len(sys.argv) else ""
        if comp == "qwen20b":
            p = ".\\models\\Qwen20B"
            if "--path" in sys.argv:
                j = sys.argv.index("--path")
                if j+1 < len(sys.argv): p = sys.argv[j+1]
            _qwen20b_validate(p)
    except SystemExit:
        raise
    except Exception as e:
        print("[qwen20b] registrar error:", e)
        raise SystemExit(3)
# <<< FRAMEVISION_QWEN_END









try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

ROOT     = Path(__file__).resolve().parents[1]
MODELS   = ROOT / "models"
EXTERNAL = ROOT / "externals"
CACHE    = ROOT / ".dl_cache"
HF_CACHE = ROOT / ".hf_cache"
URLS_DIR = ROOT / ".urls"
for p in (MODELS, EXTERNAL, CACHE, HF_CACHE, URLS_DIR):
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def log(m: str): print(f"[externals] {m}", flush=True)

# --------------------------- HTTP helpers ---------------------------
def http_json(url: str, retries: int = 3, timeout: int = 30) -> Any:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    last = None
    for i in range(retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last = e; time.sleep(1.5*(i+1))
    raise RuntimeError(f"GET {url} failed: {last}")

def http_fetch(url: str, out: Path, retries: int = 3, timeout: int = 240) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
                shutil.copyfileobj(r, f)
            # validate ZIP signature
            with open(out, "rb") as f:
                sig = f.read(4)
            if sig != b"PK\x03\x04":
                raise RuntimeError("not a zip (bad signature)")
            return out
        except Exception as e:
            last = e; time.sleep(1.5*(i+1))
    raise RuntimeError(f"download failed: {last} @ {url}")

# --------------------------- ZIP helpers ---------------------------
def extract_to_temp(zip_path: Path) -> Path:
    tmp = CACHE / f"__unpack_{zip_path.stem}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    # If single top-level dir, descend
    entries = [p for p in tmp.iterdir()]
    src = entries[0] if len(entries)==1 and entries[0].is_dir() else tmp
    return src  # caller removes tmp.parent

def copy_models_anywhere(from_dir: Path, to_dir: Path) -> tuple[int,int]:
    """Copy ANY *.bin/*.param under from_dir to to_dir.
    Returns (added, skipped_existing).
    """
    to_dir.mkdir(parents=True, exist_ok=True)
    added = 0; skipped = 0

    # Prefer structured copy from directories named models*
    did_structured = False
    for d in from_dir.rglob("*"):
        if d.is_dir() and d.name.lower().startswith("models"):
            did_structured = True
            for root,_,files in os.walk(d):
                rel = Path(root).relative_to(d)
                out = to_dir / rel
                out.mkdir(parents=True, exist_ok=True)
                for f in files:
                    if f.lower().endswith(".bin") or f.lower().endswith(".param"):
                        src = Path(root)/f
                        dst = out/f
                        if dst.exists():
                            skipped += 1
                        else:
                            shutil.copy2(src, dst); added += 1

    # Fallback: gather all loose *.bin/*.param anywhere
    if not did_structured:
        for p in from_dir.rglob("*"):
            if p.is_file() and (p.suffix.lower() in (".bin",".param")):
                dst = to_dir / p.name
                if dst.exists():
                    skipped += 1
                else:
                    shutil.copy2(p, dst); added += 1
    return added, skipped

def merge_tools_to_externals(from_dir: Path, family: str) -> None:
    out = EXTERNAL / family
    out.mkdir(parents=True, exist_ok=True)
    for item in from_dir.iterdir():
        tgt = out / item.name
        if item.is_dir():
            shutil.copytree(item, tgt, dirs_exist_ok=True)
        else:
            try: shutil.copy2(item, tgt)
            except Exception: pass

def present_any_models(folder: Path) -> bool:
    return any(folder.rglob("*.bin")) or any(folder.rglob("*.param"))

# --------------------------- Source picking ---------------------------
def pick_github_asset(owner: str, repo: str, name_contains: str="windows.zip") -> Optional[str]:
    endpoints = [f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
                 f"https://api.github.com/repos/{owner}/{repo}/releases"]
    for url in endpoints:
        try:
            data = http_json(url)
            rels = data if isinstance(data, list) else [data]
            for rel in rels:
                for asset in rel.get("assets", []):
                    name = (asset.get("name") or "").lower()
                    if name_contains in name:
                        return asset.get("browser_download_url")
        except Exception:
            continue
    return None


def _download_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent":"framevision-installer"})
    with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
        f.write(r.read())
def first_line(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    return s
    except Exception:
        pass
    return None

def url_overrides_for(family: str, cli_url: Optional[str]) -> List[str]:
    urls: List[str] = []
    # CLI has top priority
    if cli_url:
        urls.append(cli_url)
    # ENV vars
    env = os.environ.get(f"{family.upper()}_ZIP_URL")
    if env:
        urls.append(env)
    # .urls/<family>.txt
    cfg = first_line(URLS_DIR / f"{family}.txt")
    if cfg:
        urls.append(cfg)
    return urls

# --------------------------- Families ---------------------------
def pull_qwen2_vl_2b(base: Path) -> None:
    """Deprecated: replaced by pull_qwen3_vl_2b. No action taken."""
    log("Qwen2 downloader is deprecated; using Qwen3 instead.")
    return None

def pull_qwen3_vl_2b(base: Path) -> None:
    """
    Download Qwen3-VL-2B-Instruct into models/describe/default/qwen3vl2b.
    Prefers huggingface_hub.snapshot_download when available, otherwise falls back to:
        huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir <dest> --resume-download
    """
    dest = base / "describe" / "default" / "qwen3vl2b"
    log(f"pulling Qwen/Qwen3-VL-2B-Instruct -> {dest}")
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    used = None
    # Try Python API first if available
    try:
        if snapshot_download is not None:
            snapshot_download(
                repo_id="Qwen/Qwen3-VL-2B-Instruct",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["*.json","*.safetensors","*.bin","*.model","*.txt","*.tiktoken","*.py","LICENSE*","*.md"],
                ignore_patterns=["**/raw/**","**/.git/**"]
            )
            used = "hub"
    except Exception as e:
        log(f"Qwen3 snapshot_download failed ({e}); will try huggingface-cli.")

    if used is None:
        # Fallback to huggingface-cli (as requested)
        import shutil as _sh, subprocess as _sp
        cli = _sh.which("huggingface-cli") or _sh.which("hf")
        if not cli:
            log("huggingface-cli not found; skipping Qwen3-VL-2B fetch.")
            return
        cmd = [cli, "download", "Qwen/Qwen3-VL-2B-Instruct", "--local-dir", str(dest), "--resume-download"]
        try:
            _sp.run(cmd, check=True)
        except Exception as e:
            log(f"huggingface-cli download failed: {e}")
            return

    log("Qwen3-VL-2B ready.")

def install_family(owner: str, repo: str, family: str, fixed_candidates: list[str] | None = None, manual_url: Optional[str]=None) -> bool:
    # 0) Try local zips
    local = find_local_zip_candidates(family)
    for z in local:
        log(f"{family}: using local zip {z.name}")
        try:
            src = extract_to_temp(z); added, skipped = copy_models_anywhere(src, MODELS / family)
            merge_tools_to_externals(src, family); shutil.rmtree(src.parent, ignore_errors=True)
            if present_any_models(MODELS / family):
                log(f"{family}: installed from local zip {z.name} (added {added}, kept {skipped})")
                return True
        except Exception as e:
            log(f"{family}: local zip failed: {e}")

    # 1) Try overrides (CLI/ENV/.urls)
    urls = url_overrides_for(family, manual_url)

    # 2) Fixed candidates + API pick
    fixed_candidates = fixed_candidates or []
    urls.extend(fixed_candidates)
    pick = pick_github_asset(owner, repo, "windows.zip")
    if pick: urls.append(pick)

    if not urls:
        log(f"{family}: no candidate assets found; set {family.upper()}_ZIP_URL, create .urls/{family}.txt, or drop a zip into .dl_cache/.")
        return False

    last = None
    for u in urls:
        log(f"{family}: trying {u}")
        try:
            z = http_fetch(u, CACHE / f"{family}-{Path(u).name}")
            src = extract_to_temp(z)
            added, skipped = copy_models_anywhere(src, MODELS / family)
            merge_tools_to_externals(src, family)
            shutil.rmtree(src.parent, ignore_errors=True)
            if present_any_models(MODELS / family):
                log(f"{family}: installed ({added} added, {skipped} existing) from {u}")
                return True
            else:
                last = RuntimeError("no models found after extraction")
        except Exception as e:
            last = e; continue
    log(f"WARNING: {family} install failed. Last error: {last}")
    return False

def find_local_zip_candidates(family: str) -> list[Path]:
    patterns = [
        CACHE / f"{family}-*.zip",
        CACHE / f"*{family}*.zip",
        ROOT / ".cache" / f"{family}-*.zip",
        ROOT / ".cache" / f"*{family}*.zip",
        ROOT / "manual_models" / f"{family}.zip",
        ROOT / "manual_models" / f"*{family}*.zip",
        ROOT / "models" / f"{family}.zip",
    ]
    found = []
    for pat in patterns:
        for p in glob.glob(str(pat)):
            try:
                pth = Path(p)
                with open(pth, "rb") as fh:
                    if fh.read(4) == b"PK\x03\x04":
                        found.append(pth)
            except Exception:
                continue
    return found

def pull_realesrgan(manual_url: Optional[str]=None) -> None:
    install_family(
        "xinntao", "Real-ESRGAN",
        "realesrgan",
        fixed_candidates=[
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
        ],
        manual_url=manual_url
    ) or install_family("xinntao", "Real-ESRGAN-ncnn-vulkan", "realesrgan", manual_url=manual_url)

def pull_waifu2x(manual_url: Optional[str]=None) -> None:
    install_family("nihui","waifu2x-ncnn-vulkan","waifu2x",
        fixed_candidates=[
            "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-windows.zip",
            "https://sourceforge.net/projects/waifu2x-ncnn-vulkan.mirror/files/20220728/waifu2x-ncnn-vulkan-20220728-windows.zip/download"
        ],
        manual_url=manual_url)



def pull_realesr_general_v3(param_url: Optional[str]=None, bin_url: Optional[str]=None) -> None:
    """
    Download NCNN models for:
      - realesr-general-x4v3.param/bin
      - realesr-general-wdn-x4v3.param/bin
    Sources checked in order:
      1) CLI overrides (param_url/bin_url) — for non-WDN
      2) Environment (REALS_GENERAL_PARAM_URL / REALS_GENERAL_BIN_URL)
      3) .urls overrides (.urls/realesr-general.param.txt and .urls/realesr-general.bin.txt)
      4) Best-effort mirrors (Hugging Face resolve paths)
    """
    ROOT = Path(__file__).resolve().parents[1]
    MODELS = ROOT / "models" / "realesrgan"
    URLS   = ROOT / ".urls"
    MODELS.mkdir(parents=True, exist_ok=True)
    URLS.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("realesr-general-x4v3.param", "realesr-general-x4v3.bin"),
        ("realesr-general-wdn-x4v3.param", "realesr-general-wdn-x4v3.bin"),
    ]

    # Build candidate lists
    cand_param = []
    cand_bin = []

    # 1) CLI overrides for non-WDN (param_url/bin_url)
    if param_url: cand_param.append(param_url)
    if bin_url:   cand_bin.append(bin_url)

    # 2) Environment
    env_p = os.environ.get("REALS_GENERAL_PARAM_URL")
    env_b = os.environ.get("REALS_GENERAL_BIN_URL")
    if env_p: cand_param.append(env_p)
    if env_b: cand_bin.append(env_b)

    # 3) .urls overrides (single-line)
    fp = first_line(URLS / "realesr-general.param.txt")
    fb = first_line(URLS / "realesr-general.bin.txt")
    if fp: cand_param.append(fp)
    if fb: cand_bin.append(fb)

    # 4) Best-effort mirrors for both normal & WDN
    for name in ["realesr-general-x4v3.param", "realesr-general-wdn-x4v3.param"]:
        cand_param.append(f"https://huggingface.co/nihui/realesrgan-ncnn-vulkan/resolve/main/models/{name}")
    for name in ["realesr-general-x4v3.bin", "realesr-general-wdn-x4v3.bin"]:
        cand_bin.append(f"https://huggingface.co/nihui/realesrgan-ncnn-vulkan/resolve/main/models/{name}")

    # Attempt downloads
    missing = []
    for pa, bi in pairs:
        dest_pa = MODELS / pa
        dest_bi = MODELS / bi
        if not dest_pa.exists():
            tried = [u for u in cand_param if u.lower().endswith(pa)]
            ok = False
            for u in tried:
                try:
                    log(f"realesr-general: fetching {u}")
                    _download_to_file(u, dest_pa)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                missing.append(pa)
        if not dest_bi.exists():
            tried = [u for u in cand_bin if u.lower().endswith(bi)]
            ok = False
            for u in tried:
                try:
                    log(f"realesr-general: fetching {u}")
                    _download_to_file(u, dest_bi)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                missing.append(bi)

    if missing:
        log(f"WARNING: realesr-general not fully installed. Missing: {missing}. Supply direct links via .urls or CLI.")
    else:
        log("realesr-general-x4v3: installed")


def pull_upscayl(manual_url: Optional[str]=None) -> None:
    """
    Fetch Upscayl portable ZIP into .dl_cache, extract it, and copy all models (*.param/*.bin)
    under any resources/models subfolder into models/upscayl. Also mirrors the extracted tree
    to externals/upscayl so the GUI app is available if desired.
    """
    install_family(
        "upscayl", "upscayl",
        "upscayl",
        fixed_candidates=[],
        manual_url=manual_url
    )


# --- FrameVision: Upscayl v2.15.0 portable ZIP fetch to externals/upscayl (no cache kept) ---
def pull_upscayl_v215(manual_url: str | None = None) -> None:
    """Download Upscayl portable ZIP and extract to externals/upscayl.
    Removes the zip afterwards. Also copies model files (resources/models/*.param|.bin)
    into models/upscayl for Real-ESRGAN integration.
    """
    try:
        ROOT = Path(__file__).resolve().parents[1]
    except Exception:
        ROOT = Path('.').resolve()
    EXTS = ROOT / 'externals' / 'upscayl'
    MODELS = ROOT / 'models' / 'upscayl'
    EXTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    url = manual_url or 'https://github.com/upscayl/upscayl/releases/download/v2.15.0/upscayl-2.15.0-win.zip'
    zip_path = ROOT / 'externals' / 'upscayl-portable.zip'
    try:
        log(f'upscayl: downloading {url}')
    except Exception:
        print('[externals] upscayl: downloading', url)
    _download_to_file(url, zip_path)

    # Extract to externals/upscayl
    added = 0
    import zipfile as _zf
    with _zf.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            norm = name.replace('\\','/')
            target = EXTS / norm.split('/', 1)[-1] if '/' in norm else EXTS / norm
            if name.endswith('/'):
                target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as srcf, open(target, 'wb') as dstf:
                dstf.write(srcf.read())
            added += 1
            if '/resources/models/' in norm.lower() and (norm.lower().endswith('.param') or norm.lower().endswith('.bin')):
                from shutil import copy2
                copy2(target, MODELS / Path(norm).name)
    # Remove zip
    try: zip_path.unlink()
    except Exception: pass
    try:
        log(f'upscayl: installed ({added} files extracted) to {EXTS}')
    except Exception:
        print(f'[externals] upscayl: installed ({added} files) -> {EXTS}')
# --- end FrameVision block ---
# --------------------------- CLI ---------------------------

# --- begin: Background remover models (MODNet, BiRefNet) ----------------------------------------
def pull_bg_models(dest_folder: str | None = None, pro: bool = False, extra_urls: list[str] | None = None, manual_url: str | None = None) -> None:
    """
    Download background-removal models into root \\models\\ (default subfolder: 'bg').
      - Always fetches: MODNet (ONNX) — good portrait matting.
      - With --bg-pro: also fetches BiRefNet (ONNX) — large, high-quality general SOD.
    Also accepts overrides via:
      - CLI: --bg-url (can be repeated)
      - ENV: BG_URL (single URL)   [handled via url_overrides_for("bg", manual_url)]
      - .urls/bg.txt (first non-empty line)
    If a downloaded file is a .zip, it is extracted to the destination and the .zip removed.
    """
    import os, shutil, zipfile
    root = (globals().get("ROOT") or Path(__file__).resolve().parent.parent)
    models_root = root / "models"

    # Destination under models/
    sub = (dest_folder or ".").strip("\\/")
    if sub in (".", ""):
        sub = ""
    dest = models_root / sub if sub else models_root
    dest.mkdir(parents=True, exist_ok=True)

    # Default model URLs
    urls: list[str] = []
    # CLI/ENV/.urls overrides have priority
    urls.extend(url_overrides_for("bg", manual_url))

    # Built-in defaults (safe mirrors)
    # MODNet ONNX (portrait matting) — ~100MB
    
    # Primary: DavG25 (correct path includes /models/)
    urls.append("https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.onnx")
    # Fallback 1: gradio/Modnet
    urls.append("https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx")
    # Fallback 2: onnx-community/modnet-webnn
    urls.append("https://huggingface.co/onnx-community/modnet-webnn/resolve/main/onnx/model.onnx")

    # BiRefNet ONNX (large) — only when --bg-pro
    if pro:
        urls.append("https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx")

    # Extra URLs from CLI
    if extra_urls:
        urls.extend([u for u in extra_urls if isinstance(u, str) and u.strip()])

    # De-duplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        key = u.split("?")[0]
        if key not in seen:
            seen.add(key); unique_urls.append(u)

    added = 0; skipped = 0; failed = 0
    for url in unique_urls:
        # Resolve filename
        base = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
        out = dest / base
        # Skip existing non-zip file
        if out.exists() and not out.suffix.lower() == ".zip":
            print(f"[bg] skip (exists): {out.name}")
            skipped += 1
            continue

        # Download to cache first
        cache_dir = (globals().get("CACHE") or (root / ".dl_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / base
        try:
            ok = _fv_dl(url, tmp)
            if not ok:
                print(f"[bg] fail: {url}")
                failed += 1
                continue
        except Exception as e:
            print(f"[bg] error downloading {url}: {e}")
            failed += 1
            continue

        # If zip -> extract to destination, then delete zip
        if tmp.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(tmp, "r") as z:
                    z.extractall(dest)
                try: tmp.unlink()
                except Exception: pass
                print(f"[bg] extracted zip -> {dest}")
                added += 1
            except Exception as e:
                print(f"[bg] zip error for {tmp.name}: {e}")
                failed += 1
        else:
            # Normal file: move into dest (overwrite only if --bg-pro explicitly asked for a replacement)
            if out.exists():
                # keep existing; place as versioned copy
                out_ver = dest / f"{tmp.stem}_new{tmp.suffix}"
                shutil.move(str(tmp), str(out_ver))
                print(f"[bg] kept existing {out.name}; saved new as {out_ver.name}")
                skipped += 1
            else:
                shutil.move(str(tmp), str(out))
                print(f"[bg] saved {out.name}")
                added += 1

    print(f"[bg] done: {added} added, {skipped} skipped, {failed} failed; dest={dest}")
# --- end: Background remover models --------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--upscayl-v215-only', action='store_true')
    ap.add_argument('--upscayl-v215-url', type=str)
    ap.add_argument("--upscayl-only", action="store_true")
    ap.add_argument("--upscayl-url", type=str)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--describe-only", action="store_true")
    ap.add_argument("--realesrgan-only", action="store_true")
    ap.add_argument("--waifu2x-only", action="store_true")
    ap.add_argument("--lapsrn-only", action="store_true")
    # URL overrides
    ap.add_argument("--realesrgan-url", type=str)
    ap.add_argument("--waifu2x-url", type=str)

    # New: background remover models
    ap.add_argument("--bg-only", action="store_true", help="Download background-removal models (MODNet ONNX; unzip if needed) into models/bg")
    ap.add_argument("--bg-pro", action="store_true", help="Also fetch BiRefNet ONNX (~900MB)")
    ap.add_argument("--bg-url", action="append", help="Additional BG model URL(s); repeatable. Zips are extracted; zips deleted.")
    ap.add_argument("--bg-dest", type=str, default="bg", help="Destination subfolder under models/ (default: bg)")

    args = ap.parse_args(argv)

    # If no specific flags, behave like --all
    do_all = args.all or not any([
        args.describe_only, args.realesrgan_only, args.waifu2x_only, args.lapsrn_only, args.upscayl_only,
        args.bg_only, args.bg_pro
    ])

    if do_all or args.describe_only:
        try: pull_qwen3_vl_2b(MODELS)
        except Exception as e: log(f"Qwen error: {e}")

    if do_all or args.realesrgan_only:
        pull_realesrgan(args.realesrgan_url)
    if do_all or args.waifu2x_only:
        pull_waifu2x(args.waifu2x_url)

    if do_all or args.upscayl_v215_only:
        pull_upscayl_v215(args.upscayl_v215_url)

    if do_all or args.upscayl_only:
        pull_upscayl(args.upscayl_url)

    # New: background remover models
    if do_all or args.bg_only or args.bg_pro:
        try:
            pull_bg_models(dest_folder=args.bg_dest, pro=args.bg_pro, extra_urls=args.bg_url)
        except Exception as e:
            print("[externals] BG models error:", e)

    # FFmpeg & RIFE (minimal) — safe to skip if offline
    try:
        ensure_ffmpeg_bins()
        ensure_rife_pack()
    except Exception as __e:
        print("[externals] optional installers error:", __e)

    # Move externals/cache into models (Windows)
    move_folders_to_models()

if __name__ == "__main__":
    main()
# --- begin: Move folders to models ---
import subprocess

def move_to_models():
    try:
        cmds = [
            'if not exist "models" mkdir "models"',
            'robocopy ".\\externals" ".\\models" /E /MOVE /R:2 /W:0 >nul',
            'robocopy ".\\.cache" ".\\models" /E /MOVE /R:2 /W:0 >nul',
            'robocopy ".\\.dl_cache" ".\\models" /E /MOVE /R:2 /W:0 >nul',
            'exit /b 0'
        ]
        for cmd in cmds:
            subprocess.call(cmd, shell=True)
    except Exception as e:
        print("[externals] move_to_models error:", e)

# Call automatically at end of script
if __name__ == "__main__":
    move_to_models()
# --- end: Move folders to models ---

# --- begin: TXT->IMG model families (SD1.5 / SDXL) --------------------------------------------
# Moved to separate module: download_sd_models.py
# Keeping download_externals.py focused on non-SD externals, while TXT->IMG (SD15/SDXL) remains optional.
try:
    # When executed as scripts/download_externals.py
    from download_sd_models import run_txt2img_auto as _fv_run_txt2img_auto
except Exception:
    try:
        # When executed as a module package (rare)
        from .download_sd_models import run_txt2img_auto as _fv_run_txt2img_auto
    except Exception:
        _fv_run_txt2img_auto = None

if __name__ == "__main__":
    if _fv_run_txt2img_auto is None:
        print("[externals][txt2img] download_sd_models.py not found; skipping TXT→IMG downloads")
    else:
        try:
            _fv_run_txt2img_auto()
        except Exception as e:
            print("[externals][txt2img] error:", e)
# --- end: TXT->IMG model families -------------------------------------------------------------
# --- begin: run downloadbg.py at the very end ----------------------------------
def _run_downloadbg_after_all():
    try:
        import sys, subprocess
        from pathlib import Path as _Path
        _root = (globals().get("ROOT") or _Path(__file__).resolve().parents[1])
        _script = _root / "scripts" / "downloadbg.py"
        if not _script.exists():
            print("[externals] downloadbg.py not found at", _script)
            return
        print("[externals] starting downloadbg.py …")
        # Use project root as cwd; run with current interpreter
        subprocess.run([sys.executable, str(_script)], cwd=str(_root), check=False)
    except Exception as e:
        print("[externals] downloadbg runner error:", e)

if __name__ == "__main__":
    # Ensure this fires after all other __main__ blocks above
    _run_downloadbg_after_all()
# --- end: run downloadbg.py at the very end ------------------------------------
