from __future__ import annotations
import os
import urllib.request, urllib.error, sys, io, json, time, shutil, zipfile, argparse, urllib.request, glob
from pathlib import Path
from typing import Optional, Any, List
import subprocess, hashlib, datetime, re
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

    cache = root / ".dl_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # Keep this as "latest" if you prefer always-up-to-date builds.
    # Note: the installer records the exact build info (ffmpeg -version + sha256) into presets/info/
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
    zpath = cache / Path(url).name

    # Install bins if missing
    if not all((bin_dir / t).exists() for t in targets):
        if not zpath.exists() and not _fv_dl(url, zpath):
            print("[externals] FFmpeg ZIP fetch failed; skipping")
            return
        with _zipfile.ZipFile(zpath, "r") as z:
            names = z.namelist()
            for w in targets:
                cand = [n for n in names if n.lower().endswith("/" + w) or n.lower().endswith("\\" + w) or n.lower().endswith(w)]
                if cand:
                    with z.open(cand[0]) as src, open(bin_dir / w, "wb") as out:
                        out.write(src.read())
        print("[externals] ffmpeg bins installed to", bin_dir)
    else:
        print("[externals] ffmpeg bins already present")

    # Always record build info + update license metadata
    _fv_record_ffmpeg_build(root=root, bin_dir=bin_dir, ffmpeg_url=url, zip_path=zpath if zpath.exists() else None)

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

def _fv_sha256(p: _P) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _fv_record_ffmpeg_build(root: _P, bin_dir: _P, ffmpeg_url: str, zip_path: Optional[_P] = None) -> None:
    """
    Writes presets/info/ffmpeg_build_info.txt and updates presets/info/3rd_party_licenses.json
    with the exact installed FFmpeg build info (version + sha256).
    Safe to call repeatedly.
    """
    try:
        ffmpeg_exe = bin_dir / "ffmpeg.exe"
        if not ffmpeg_exe.exists():
            return

        info_dir = root / "presets" / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        build_info_path = info_dir / "ffmpeg_build_info.txt"

        # Capture "ffmpeg -version" output
        try:
            cp = subprocess.run(
                [str(ffmpeg_exe), "-version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=20,
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
            out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        except Exception as e:
            out = f"[error] failed to run ffmpeg -version: {e}\n"

        # Write build info log (always overwrite with latest info)
        stamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        header = f"Captured at (UTC): {stamp}\nFFmpeg URL: {ffmpeg_url}\n"
        if zip_path and zip_path.exists():
            header += f"ZIP SHA256: {_fv_sha256(zip_path)}\nZIP Path: {zip_path}\n"
        header += f"ffmpeg.exe SHA256: {_fv_sha256(ffmpeg_exe)}\n\n"
        build_info_path.write_text(header + out, encoding="utf-8", errors="replace")

        # Parse the version token from first line
        version_token = ""
        first_line = (out.splitlines()[0] if out else "").strip()
        m = re.search(r"ffmpeg\\s+version\\s+(\\S+)", first_line, flags=re.IGNORECASE)
        if m:
            version_token = m.group(1)

        # Update 3rd_party_licenses.json if present
        candidates = [
            root / "presets" / "info" / "3rd_party_licenses.json",
            root / "presets" / "info" / "3rd_party_licences.json",
            root / "3rd_party_licenses.json",
            root / "3rd_party_licences.json",
        ]
        lic_path = next((p for p in candidates if p.exists()), None)
        if not lic_path:
            print("[externals] (info) no 3rd_party_licenses.json found to update")
            return

        try:
            data = json.loads(lic_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            print("[externals] (warn) failed to parse licenses json:", e)
            return

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            print("[externals] (warn) licenses json missing 'items' list:", lic_path)
            return

        # Locate FFmpeg entry (by id/name)
        entry = None
        for it in items:
            if not isinstance(it, dict):
                continue
            _id = str(it.get("id", "")).lower()
            _name = str(it.get("name", "")).lower()
            if "ffmpeg" in _id or _name.startswith("ffmpeg"):
                entry = it
                break
        if not entry:
            print("[externals] (warn) no FFmpeg entry found in licenses json:", lic_path)
            return

        entry["download_url"] = ffmpeg_url
        entry["source_url"] = ffmpeg_url
        if version_token:
            entry["installed_version"] = version_token
        entry["installed_at_utc"] = stamp
        entry["installed_ffmpeg_sha256"] = _fv_sha256(ffmpeg_exe)
        entry["build_info_path"] = str(build_info_path.relative_to(root)).replace("\\\\", "/")

        # Touch generator timestamp if present
        if isinstance(data, dict):
            data["generated_at_utc"] = stamp

        lic_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[externals] updated license info:", lic_path)
    except Exception as e:
        print("[externals] (warn) failed recording FFmpeg build info:", e)

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


# Qwen20B local registrar/validator removed: obsolete optional install hook.










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

    # Former scripts/downloadbg.py is embedded below so scripts/ only needs this one downloader.
    try:
        _run_embedded_downloadbg()
    except Exception as e:
        print("[externals] embedded downloadbg error:", e)


# ---------------------------------------------------------------------------
# Embedded former scripts/downloadbg.py
# ---------------------------------------------------------------------------
# Kept in this file so the installer no longer needs to call a second script.
# Faster-Whisper medium is still opt-in via the embedded downloadbg --fw-medium
# behavior; the normal/default run skips Faster-Whisper, same as before.
_DOWNLOADBG_EMBEDDED_SOURCE = '\n#!/usr/bin/env python3\n\n"""\ndownloadbg.py — v3.10\n- Pre-check destination folders BEFORE any download (prevents re-fetching huge files)\n- Reconciliation FIRST: move scripts/models -> project root models/, then clean\n- Adds --force to re-download even if destination already has the file\n- UltraSharp NCNN (Hugging Face) + SRMD (GitHub) upscaler downloads to models/realesrgan\n- Default = ALL models when no flags passed (except Faster-Whisper)\n- Faster-Whisper "medium" model downloader is opt-in via --fw-medium (downloads to models/faster_whisper/medium)\n"""\nimport argparse, hashlib, os, sys, urllib.request, urllib.error, pathlib, shutil, time, math, zipfile, io\n\nVERSION = "v3.10"\n\n# ------------------------------\n# Model registries\n# ------------------------------\n# NOTE:\n# - Zip-based packages live in MODELS and are referenced by --only / --realsr.\n# - File-based extras (UltraSharp / SRMD / RealESRGAN extras / Faster-Whisper) are handled by dedicated functions below.\n\n# Zip-based packages (downloaded into --dest, optionally extracted)\nMODELS = {\n    # RealSR 2x/4x package (nihui/realsr-ncnn-vulkan) — Windows build\n    # The GitHub release page lists this asset as ~61 MB. \n    "realsr_ncnn_zip": {\n        "filename": "realsr-ncnn-vulkan-20220728-windows.zip",\n        "url": "https://github.com/nihui/realsr-ncnn-vulkan/releases/download/20220728/realsr-ncnn-vulkan-20220728-windows.zip",\n        "sha256": None,          # optional; leave None to skip hash check\n        "size_hint": "≈61 MB",\n        # Extract under models/ so the app can keep everything self-contained.\n        "extract_to": "models/realsr_ncnn_vulkan",\n        # Presence check (best-effort; different zips sometimes vary slightly)\n        "exists_path": [\n            "models/realsr_ncnn_vulkan/realsr-ncnn-vulkan.exe",\n            "models/realsr_ncnn_vulkan/realsr-ncnn-vulkan",\n        ],\n    },\n}\n\n# UltraSharp NCNN model files (Kim2091/UltraSharp on Hugging Face) \nULTRASHARP_BASE = "https://huggingface.co/Kim2091/UltraSharp/resolve/main/NCNN"\nULTRASHARP_FILES = [\n    "4x-UltraSharp-fp16.bin",\n    "4x-UltraSharp-fp16.param",\n    "4x-UltraSharp-fp32.bin",\n    "4x-UltraSharp-fp32.param",\n]\n\n\n# ------------------------------\n# Paths and helpers\n# ------------------------------\n\ndef _get_project_root() -> pathlib.Path:\n    this = pathlib.Path(__file__).resolve()\n    scripts_dir = this.parent\n    return scripts_dir.parent if scripts_dir.name.lower() == "scripts" else scripts_dir\n\nROOT = _get_project_root()\nROOT_MODELS = ROOT / "models"\nROOT_REALESRGAN = ROOT_MODELS / "realesrgan"\nROOT_FWHISPER = ROOT_MODELS / "faster_whisper"\nROOT_FWHISPER_MEDIUM = ROOT_FWHISPER / "medium"\nSCRIPTS_MODELS = pathlib.Path(__file__).resolve().parent / "models"\n\ndef _human(n_bytes: int) -> str:\n    units = ["B","KB","MB","GB","TB"]\n    f = float(n_bytes)\n    for u in units:\n        if f < 1024 or u == units[-1]:\n            return f"{f:,.1f} {u}" if u != "B" else f"{int(f)} {u}"\n        f /= 1024\n\ndef _eta_str(seconds: float) -> str:\n    if not math.isfinite(seconds) or seconds <= 0:\n        return "--:--"\n    m, s = divmod(int(seconds), 60)\n    h, m = divmod(m, 60)\n    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"\n\ndef sha256sum(path):\n    h = hashlib.sha256()\n    with open(path, "rb") as f:\n        for chunk in iter(lambda: f.read(1024*1024), b""):\n            h.update(chunk)\n    return h.hexdigest()\n\ndef _build_request(url: str, token: str | None):\n    headers = {\n        "User-Agent": f"downloadbg/{VERSION} (python-urllib)",\n        "Accept": "*/*",\n    }\n    if token and "huggingface.co" in url:\n        headers["Authorization"] = f"Bearer {token.strip()}"\n    return urllib.request.Request(url, headers=headers)\n\ndef _print_progress(prefix: str, downloaded: int, total: int | None, t0: float):\n    width = 100\n    try:\n        width = max(40, min((shutil.get_terminal_size().columns or 80), 140))\n    except Exception:\n        pass\n    elapsed = max(1e-6, time.time() - t0)\n    speed = downloaded / elapsed  # bytes/sec\n    if total:\n        pct = downloaded / total\n        eta = (total - downloaded) / max(1e-6, speed)\n        bar_w = max(10, min(30, width - 60))\n        filled = int(bar_w * pct)\n        bar = "#" * filled + "-" * (bar_w - filled)\n        line = f"{prefix} |{bar}| {pct*100:5.1f}%  { _human(downloaded) } / { _human(total) }  { _human(speed) }/s  ETA { _eta_str(eta) }"\n    else:\n        line = f"{prefix} { _human(downloaded) }  { _human(speed) }/s"\n    sys.stdout.write("\\x1b[2K\\r" + line[:width])\n    sys.stdout.flush()\n\ndef _fetch_single(url, dest_path, expected_sha256=None, token: str | None = None):\n    req = _build_request(url, token)\n    tmp = str(dest_path) + ".part"\n    try:\n        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:\n            total = r.headers.get("Content-Length")\n            total = int(total) if total is not None else None\n            downloaded = 0\n            t0 = time.time()\n            last = 0.0\n            _print_progress("[dl  ]", 0, total, t0)\n            while True:\n                chunk = r.read(1024 * 1024)  # 1 MB\n                if not chunk:\n                    break\n                f.write(chunk)\n                downloaded += len(chunk)\n                now = time.time()\n                if now - last >= 0.25 or (total and downloaded >= total):\n                    last = now\n                    _print_progress("[dl  ]", downloaded, total, t0)\n            _print_progress("[dl  ]", downloaded, total, t0)\n            sys.stdout.write("\\n"); sys.stdout.flush()\n    except Exception:\n        try:\n            if os.path.exists(tmp):\n                os.remove(tmp)\n        except Exception:\n            pass\n        raise\n\n    os.replace(tmp, dest_path)\n    if expected_sha256:\n        got = sha256sum(dest_path)\n        if got.lower() != expected_sha256.lower():\n            raise RuntimeError(f"SHA256 mismatch for {dest_path.name}: {got} != {expected_sha256}")\n\ndef fetch_with_fallback(url_or_list, dest_path, expected_sha256=None, token: str | None = None):\n    urls = url_or_list if isinstance(url_or_list, (list, tuple)) else [url_or_list]\n    last_err = None\n    for i, url in enumerate(urls, 1):\n        try:\n            print(f"[try ] {dest_path.name}  mirror {i}/{len(urls)}")\n            _fetch_single(url, dest_path, expected_sha256, token=token)\n            return\n        except urllib.error.HTTPError as e:\n            last_err = e\n            print(f"[warn] {dest_path.name} from mirror {i}: HTTP {e.code} {e.reason}")\n            if e.code in (401, 403) and "huggingface.co" in url:\n                print("      Requires HF token and license acceptance. Set HF_TOKEN or pass --hf-token.")\n        except Exception as e:\n            last_err = e\n            print(f"[warn] {dest_path.name} from mirror {i}: {e}")\n    if last_err:\n        raise last_err\n\ndef move_all_to(temp_dest: pathlib.Path, final_root: pathlib.Path) -> int:\n    final_root.mkdir(parents=True, exist_ok=True)\n    try:\n        # Safety: if the temp folder is the same as the final folder, there\'s nothing to move.\n        if temp_dest.resolve() == final_root.resolve():\n            # This prevents accidental deletion when source == destination.\n            return 0\n    except Exception:\n        # If resolve() fails for any reason, continue without the guard (best-effort).\n        pass\n    moved = 0\n    for p in temp_dest.glob("*"):\n        if not p.is_file():\n            continue\n        target = final_root / p.name\n        try:\n            if target.exists():\n                try:\n                    target.unlink()\n                except Exception:\n                    os.chmod(target, 0o666)\n                    target.unlink()\n            shutil.move(str(p), str(target))\n            moved += 1\n        except Exception as e:\n            print(f"[warn] Move failed for {p.name}: {e}")\n    return moved\n\ndef cleanup_model_root_zips() -> int:\n    root = ROOT_MODELS\n    if not root.exists():\n        return 0\n    deleted = 0\n    for z in root.glob("*.zip"):\n        try:\n            z.unlink()\n            deleted += 1\n        except Exception as e:\n            print(f"[warn] Could not delete {z.name}: {e}")\n    return deleted\n\ndef reconcile_scripts_models() -> None:\n    legacy = SCRIPTS_MODELS\n    if not legacy.exists():\n        return\n    ROOT_MODELS.mkdir(parents=True, exist_ok=True)\n\n    for p in list(legacy.glob("*")):\n        try:\n            if p.is_dir():\n                dest = ROOT_MODELS / p.name\n                if not dest.exists():\n                    shutil.move(str(p), str(dest))\n                    print(f"[move] {p} -> {dest}")\n                else:\n                    print(f"[skip] {dest} already present")\n                continue\n            if p.suffix.lower() == ".zip":\n                dest = ROOT_MODELS / p.name\n                if not dest.exists():\n                    shutil.move(str(p), str(dest))\n                    print(f"[move] {p} -> {dest}")\n                else:\n                    print(f"[skip] {dest} already present")\n            else:\n                dest = ROOT_MODELS / p.name\n                if not dest.exists():\n                    shutil.move(str(p), str(dest))\n                    print(f"[move] {p} -> {dest}")\n                else:\n                    print(f"[skip] {dest} already present")\n        except Exception as e:\n            print(f"[warn] Could not reconcile {p}: {e}")\n    try:\n        shutil.rmtree(legacy)\n        print(f"[clean] Deleted folder {legacy}")\n    except Exception as e:\n        print(f"[warn] Could not delete {legacy}: {e}")\n\ndef download_ultrasharp_ncnn(temp_dest: pathlib.Path, token: str | None = None) -> tuple[int,int]:\n    temp_dest.mkdir(parents=True, exist_ok=True)\n    ROOT_REALESRGAN.mkdir(parents=True, exist_ok=True)\n    # If ALL files already exist in final folder, skip\n    all_present = all((ROOT_REALESRGAN / f).exists() for f in ULTRASHARP_FILES)\n    if all_present:\n        print(f"[skip] UltraSharp NCNN already present in {ROOT_REALESRGAN}")\n        return (0,0)\n    downloaded = 0\n    for fname in ULTRASHARP_FILES:\n        final_path = ROOT_REALESRGAN / fname\n        if final_path.exists():\n            print(f"[skip] {fname} already exists in {ROOT_REALESRGAN}")\n            continue\n        out = temp_dest / fname\n        if out.exists():\n            print(f"[skip] {out.name} already exists in {temp_dest}")\n        else:\n            url = f"{ULTRASHARP_BASE}/{fname}?download=true"\n            try:\n                print(f"[get ] {fname} (UltraSharp NCNN)")\n                fetch_with_fallback(url, out, expected_sha256=None, token=token)\n                print(f"[ ok ] Saved to {out}")\n                downloaded += 1\n            except Exception as e:\n                print(f"[fail] {fname}: {e}")\n    moved = move_all_to(temp_dest, ROOT_REALESRGAN)\n    if moved:\n        print(f"[move] UltraSharp -> {ROOT_REALESRGAN} ({moved} file(s))")\n    return (downloaded, moved)\n\ndef download_srmd_models(temp_dest: pathlib.Path) -> tuple[int,int]:\n    temp_dest.mkdir(parents=True, exist_ok=True)\n    ROOT_REALESRGAN.mkdir(parents=True, exist_ok=True)\n    zip_name = "srmd-ncnn-vulkan-master.zip"\n    zip_path = temp_dest / zip_name\n    repo_zip_url = "https://github.com/nihui/srmd-ncnn-vulkan/archive/refs/heads/master.zip"\n    try:\n        print(f"[get ] SRMD models (GitHub archive)")\n        fetch_with_fallback(repo_zip_url, zip_path, expected_sha256=None, token=None)\n        print(f"[ ok ] Saved to {zip_path}")\n    except Exception as e:\n        print(f"[fail] SRMD archive: {e}")\n        return (0,0)\n\n    extracted = 0\n    try:\n        with zipfile.ZipFile(zip_path, \'r\') as zf:\n            members = zf.namelist()\n            sub_prefix = "srmd-ncnn-vulkan-master/models/models-srmd/"\n            select = [m for m in members if m.startswith(sub_prefix) and (m.endswith(".bin") or m.endswith(".param")) and not m.endswith("/") ]\n            if not select:\n                print(f"[warn] SRMD archive: no files found in {sub_prefix}")\n            for m in select:\n                fname = os.path.basename(m)\n                final_path = ROOT_REALESRGAN / fname\n                if final_path.exists():\n                    print(f"[skip] {fname} already exists in {ROOT_REALESRGAN}")\n                    continue\n                data = zf.read(m)\n                out = temp_dest / fname\n                with open(out, "wb") as f:\n                    f.write(data)\n                extracted += 1\n        try:\n            zip_path.unlink()\n        except Exception:\n            pass\n    except Exception as e:\n        print(f"[fail] Extract SRMD archive: {e}")\n        return (0,0)\n\n    moved = move_all_to(temp_dest, ROOT_REALESRGAN)\n    if moved:\n        print(f"[move] SRMD -> {ROOT_REALESRGAN} ({moved} file(s))")\n    return (extracted, moved)\n\n# RealESRGAN extra models (Remacri + RealeSR-general-v3)\nREALESR_REM_BASE = "https://huggingface.co/tumuyan2/realsr-models/resolve/main/models-ESRGAN-Remacri"\nREALESR_GEN_BASE = "https://huggingface.co/tumuyan2/realsr-models/resolve/main/models-RealeSR-general-v3"\n\nREALESR_EXTRA_FILES = {\n    "realesr-remacri_x4.bin": f"{REALESR_REM_BASE}/x4.bin?download=true",\n    "realesr-remacri_x4.param": f"{REALESR_REM_BASE}/x4.param?download=true",\n    "realesr-general-v3_x4.bin": f"{REALESR_GEN_BASE}/x4.bin?download=true",\n    "realesr-general-v3_x4.param": f"{REALESR_GEN_BASE}/x4.param?download=true",\n}\n\ndef download_realesr_extra_models(temp_dest: pathlib.Path, token: str | None = None, force: bool = False) -> tuple[int,int]:\n    """\n    Download extra RealESRGAN NCNN models:\n    - realesr-remacri_x4\n    - realesr-general-v3_x4\n\n    Files are fetched from tumuyan2/realsr-models on HuggingFace\n    into a temporary folder and renamed to the expected NCNN filenames\n    before being moved into models/realesrgan.\n    """\n    temp_dest.mkdir(parents=True, exist_ok=True)\n    ROOT_REALESRGAN.mkdir(parents=True, exist_ok=True)\n\n    downloaded = 0\n    for final_name, url in REALESR_EXTRA_FILES.items():\n        final_path = ROOT_REALESRGAN / final_name\n        if final_path.exists() and not force:\n            print(f"[skip] {final_name} already exists in {ROOT_REALESRGAN}")\n            continue\n        out = temp_dest / final_name\n        if out.exists() and not force:\n            print(f"[skip] {out.name} already exists in {temp_dest}")\n        else:\n            try:\n                print(f"[get ] {final_name} (RealESRGAN extra)")\n                fetch_with_fallback(url, out, expected_sha256=None, token=token)\n                print(f"[ ok ] Saved to {out}")\n                downloaded += 1\n            except Exception as e:\n                print(f"[fail] {final_name}: {e}")\n\n    moved = move_all_to(temp_dest, ROOT_REALESRGAN)\n    if moved:\n        print(f"[move] RealESRGAN extras -> {ROOT_REALESRGAN} ({moved} file(s))")\n    return (downloaded, moved)\n\n\n# Faster-Whisper "medium" model (Systran/faster-whisper-medium)\nFWHISPER_MEDIUM_BASE = "https://huggingface.co/Systran/faster-whisper-medium/resolve/main"\nFWHISPER_MEDIUM_FILES = [\n    "config.json",\n    "model.bin",\n    "tokenizer.json",\n    "vocabulary.txt",\n]\n\ndef download_fasterwhisper_medium(temp_dest: pathlib.Path, token: str | None = None, force: bool = False) -> tuple[int,int]:\n    """\n    Download the Faster-Whisper \'medium\' CTranslate2 model from HuggingFace into\n    models/faster_whisper/medium.\n\n    - Uses temp_dest as temporary holder then moves into ROOT_FWHISPER_MEDIUM.\n    - Skips download if all expected files already exist (unless force=True).\n    """\n    temp_dest.mkdir(parents=True, exist_ok=True)\n    ROOT_FWHISPER_MEDIUM.mkdir(parents=True, exist_ok=True)\n\n    # Skip if all target files already exist in final folder\n    if not force and all((ROOT_FWHISPER_MEDIUM / f).exists() for f in FWHISPER_MEDIUM_FILES):\n        print(f"[skip] Faster-Whisper medium already present in {ROOT_FWHISPER_MEDIUM}")\n        return (0,0)\n\n    downloaded = 0\n    for fname in FWHISPER_MEDIUM_FILES:\n        final_path = ROOT_FWHISPER_MEDIUM / fname\n        if final_path.exists() and not force:\n            print(f"[skip] {fname} already exists in {ROOT_FWHISPER_MEDIUM}")\n            continue\n        out = temp_dest / fname\n        if out.exists() and not force:\n            print(f"[skip] {out.name} already exists in {temp_dest}")\n        else:\n            url = f"{FWHISPER_MEDIUM_BASE}/{fname}?download=true"\n            try:\n                print(f"[get ] {fname} (Faster-Whisper medium)")\n                fetch_with_fallback(url, out, expected_sha256=None, token=token)\n                print(f"[ ok ] Saved to {out}")\n                downloaded += 1\n            except Exception as e:\n                print(f"[fail] {fname}: {e}")\n\n    moved = move_all_to(temp_dest, ROOT_FWHISPER_MEDIUM)\n    if moved:\n        print(f"[move] Faster-Whisper medium -> {ROOT_FWHISPER_MEDIUM} ({moved} file(s))")\n    return (downloaded, moved)\n\n# ------------------------------\n# Main\n# ------------------------------\n\ndef main():\n    import shutil as _shutil\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--dest", default="scripts/_tmp_downloads", help="Temporary download folder for zips/files (safe to delete)")\n    ap.add_argument("--realsr", action="store_true", help="Include RealSR 2x/4x (realsr-ncnn-vulkan) zip")\n    ap.add_argument("--ultrasharp", action="store_true", help="Download UltraSharp NCNN files to models/realesrgan")\n    ap.add_argument("--srmd", action="store_true", help="Download SRMD models (nihui) to models/realesrgan")\n    ap.add_argument("--fw-medium", action="store_true", help="Download Faster-Whisper medium model (≈1.6 GB) to models/faster_whisper/medium")\n    ap.add_argument("--all", action="store_true", help="Download ALL supported models (includes UltraSharp+SRMD+RealESRGAN extras; excludes Faster-Whisper unless --fw-medium)")\n    only_choices = list(MODELS.keys()) if isinstance(globals().get("MODELS"), dict) else None\n    ap.add_argument("--only", nargs="+", choices=only_choices, help="Download only these model keys (space-separated)")\n    ap.add_argument("--hf-token", default=None, help="Hugging Face access token (overrides HF_TOKEN/HUGGINGFACE_TOKEN env vars)")\n    ap.add_argument("--ignore-errors", action="store_true", help="Return success even if some downloads fail")\n    ap.add_argument("--force", action="store_true", help="Re-download files even if they seem present already")\n    args = ap.parse_args()\n\n    print(f"[downloadbg] {VERSION}")\n\n    # 1) Reconcile legacy first so presence checks see the files in their final place\n    reconcile_scripts_models()\n\n    dest = pathlib.Path(args.dest)\n    dest.mkdir(parents=True, exist_ok=True)\n\n    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")\n\n    any_fail = False\n\n    # 2) Determine selection for zip-based tools (e.g., RealSR package)\n    all_keys = list(MODELS.keys())\n    selected_default_all = False\n    if args.only:\n        to_get = args.only\n        reason = "--only"\n    elif args.all or (not args.realsr and not args.ultrasharp and not args.srmd and not args.fw_medium):\n        # Default: all supported downloads when no flags are passed\n        to_get = all_keys\n        reason = "--all(default)"\n        selected_default_all = True\n    else:\n        to_get = []\n        reason = "flags"\n        if args.realsr:\n            to_get.append("realsr_ncnn_zip")\n\n    # Ensure realsr is included when explicitly requested\n    if args.realsr and "realsr_ncnn_zip" not in to_get:\n        to_get.append("realsr_ncnn_zip")\n        reason += "+realsr"\n    for key in to_get:\n        m = MODELS.get(key) if isinstance(MODELS, dict) else None\n        if not m:\n            any_fail = True\n            print(f"[fail] Unknown model key: {key}")\n            continue\n        out = dest / m["filename"]\n\n        # Skip logic (unless --force)\n        if not args.force:\n            # temp dest present\n            if out.exists():\n                print(f"[skip] {out.name} already exists in {dest}")\n                continue            # extracted path present (e.g., RealSR extracted folder)\n            exists_path = m.get("exists_path")\n            if exists_path:\n                candidates = exists_path if isinstance(exists_path, (list, tuple)) else [exists_path]\n                exists_found = False\n                for c in candidates:\n                    if not c:\n                        continue\n                    exists_path_p = (ROOT / c) if not pathlib.Path(c).is_absolute() else pathlib.Path(c)\n                    if exists_path_p.exists():\n                        print(f"[skip] {exists_path_p} already present")\n                        exists_found = True\n                        break\n                if exists_found:\n                    continue\n\n        # If we arrive here, proceed with fetching\n        print(f"[get ] {out.name}  {m[\'size_hint\']}")\n        try:\n            fetch_with_fallback(m["url"], out, m["sha256"], token=token)\n        except Exception as e:\n            any_fail = True\n            print(f"[fail] {out.name}: {e}")\n            continue\n        print(f"[ ok ] Saved to {out}")\n\n        # Unzip & delete, if applicable\n        if m.get("extract_to") and out.suffix.lower() == ".zip":\n            try:\n                target_root = (ROOT / m["extract_to"]) if not pathlib.Path(m["extract_to"]).is_absolute() else pathlib.Path(m["extract_to"])\n                _ = 0\n                with zipfile.ZipFile(out, \'r\') as zf:\n                    members = zf.namelist()\n                    print(f"[unzip] {out.name} -> {target_root.resolve()} ({len(members)} entries)")\n                    zf.extractall(target_root)\n                try:\n                    out.unlink()\n                    print(f"[clean] Deleted zip {out.name} after extraction")\n                except Exception as e:\n                    print(f"[warn] Could not delete zip {out.name}: {e}")\n            except Exception as e:\n                any_fail = True\n                print(f"[fail] Extract {out.name}: {e}")\n\n    # 5) Upscaler extras (UltraSharp + SRMD + RealESRGAN extras)\n    include_ultrasharp = args.ultrasharp or selected_default_all or args.all\n    include_srmd = args.srmd or selected_default_all or args.all\n    include_realesr_extras = selected_default_all or args.all or args.ultrasharp or args.srmd\n    realesr_tmp = pathlib.Path("scripts") / "_tmp_realesrgan"\n    try:\n        if include_ultrasharp:\n            dl, mv = download_ultrasharp_ncnn(realesr_tmp, token=token)\n            if dl == 0 and mv == 0:\n                print("[info] UltraSharp NCNN done (nothing to do)")\n        if include_srmd:\n            ex, mv = download_srmd_models(realesr_tmp)\n            if ex == 0 and mv == 0:\n                print("[info] SRMD models done (nothing to do)")\n        if include_realesr_extras:\n            dl2, mv2 = download_realesr_extra_models(realesr_tmp, token=token, force=args.force)\n            if dl2 == 0 and mv2 == 0:\n                print("[info] RealESRGAN extras (Remacri + RealeSR-general-v3) done (nothing to do)")\n    finally:\n        try:\n            if realesr_tmp.exists():\n                import shutil as _shutil2\n                _shutil2.rmtree(realesr_tmp)\n        except Exception:\n            pass\n\n    # 6) Faster-Whisper medium model\n    include_fw_medium = args.fw_medium\n    fw_tmp = pathlib.Path("scripts") / "_tmp_fasterwhisper"\n    try:\n        if include_fw_medium:\n            dl, mv = download_fasterwhisper_medium(fw_tmp, token=token, force=args.force)\n            if dl == 0 and mv == 0:\n                print("[info] Faster-Whisper medium done (nothing to do)")\n    finally:\n        try:\n            if fw_tmp.exists():\n                import shutil as _shutil3\n                _shutil3.rmtree(fw_tmp)\n        except Exception:\n            pass\n\n    if any_fail and not args.ignore_errors:\n        print("[done] Completed with errors.", file=sys.stderr)\n        return 1\n\n    parts = [str(ROOT_MODELS.resolve()), str(ROOT_REALESRGAN.resolve())]\n    if include_fw_medium:\n        parts.append(str(ROOT_FWHISPER_MEDIUM.resolve()))\n        print("[done] Models ready in " + ", ".join(parts))\n    else:\n        print("[done] Models ready in " + ", ".join(parts) + " (Faster-Whisper skipped; use --fw-medium to download)")\n    return 0\n\nif __name__ == "__main__":\n    sys.exit(main())\n'

def _run_embedded_downloadbg() -> int:
    import sys as _sys
    _ns = {
        "__name__": "_framevision_embedded_downloadbg",
        "__file__": __file__,
        "__package__": None,
    }
    exec(_DOWNLOADBG_EMBEDDED_SOURCE, _ns)
    _main = _ns.get("main")
    if not callable(_main):
        print("[externals] embedded downloadbg has no main(); skipping")
        return 0
    _old_argv = list(_sys.argv)
    try:
        # Match the previous behavior: download_externals.py launched downloadbg.py with no extra args.
        _sys.argv = [str(__file__)]
        rc = _main()
        return int(rc or 0)
    finally:
        _sys.argv = _old_argv


if __name__ == "__main__":
    raise SystemExit(main())
