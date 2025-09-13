
# FrameVision worker V1.0 — NCNN wiring
import json, time, subprocess, os, re
from pathlib import Path
try:
    from PIL import Image
except Exception:
    Image = None

ROOT = Path(".").resolve()
BASE = ROOT

# ---- Media type helpers ----
IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff",".gif"}
VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".webm",".m4v"}

def is_image_path(p: Path) -> bool:
    try:
        return p.suffix.lower() in IMAGE_EXTS
    except Exception:
        return False

def is_video_path(p: Path) -> bool:
    try:
        return p.suffix.lower() in VIDEO_EXTS
    except Exception:
        return False

# ---- Executable resolution (sanity-filtered) ----
def resolve_upscaler_exe(cfg: dict, mani: dict, model_name: str):
    """Return (canonical_model_name, exe_path) for upscalers only (NO RIFE)."""
    # 1) Manifest-relative exe path
    try:
        root = Path(mani.get("root")) if mani.get("root") else ROOT
    except Exception:
        root = ROOT
    entry = (mani.get("models") or {}).get("upscalers", {}).get(model_name) if mani else None
    if entry and isinstance(entry, dict):
        exe_rel = (entry or {}).get('exe') or ''
        if exe_rel:
            exe_path = (root / exe_rel)
            if exe_path.exists() and exe_path.is_file():
                return (model_name, exe_path)

    # 2) Search models folder for known upscaler executables
    models_dir = _resolve_models_folder(cfg)
    if models_dir and models_dir.exists():
        candidates = [
            "realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan",
            "swinir-ncnn-vulkan.exe",     "swinir-ncnn-vulkan",
            "waifu2x-ncnn-vulkan.exe",    "waifu2x-ncnn-vulkan",
            "lapsrn-ncnn-vulkan.exe",     "lapsrn-ncnn-vulkan",
        ]
        try:
            for name in candidates:
                for p in models_dir.rglob(name):
                    if p.is_file():
                        return (model_name, p)
        except Exception:
            pass
    return (model_name, None)

def resolve_rife_exe(cfg: dict, mani: dict):
    """Locate rife executable for interpolation jobs only."""
    models_dir = _resolve_models_folder(cfg)
    if models_dir and models_dir.exists():
        for name in ["rife-ncnn-vulkan.exe","rife-ncnn-vulkan"]:
            try:
                for p in models_dir.rglob(name):
                    if p.is_file():
                        return p
            except Exception:
                pass
    # Fallback to manifest exe if provided
    try:
        root = Path(mani.get("root")) if mani.get("root") else ROOT
        entry = (mani.get("models") or {}).get("interpolators", {}).get("rife") if mani else None
        if entry and isinstance(entry, dict):
            exe_rel = (entry or {}).get('exe') or ''
            if exe_rel:
                exe_path = (root / exe_rel)
                if exe_path.exists() and exe_path.is_file():
                    return exe_path
    except Exception:
        pass
    return None
def _resolve_models_folder(cfg: dict) -> Path:
    # Prefer config if valid; otherwise fall back to typical locations.
    try:
        cand = Path(cfg.get("models_folder", "")).expanduser()
        if str(cand).strip() and cand.exists():
            return cand
    except Exception:
        pass
    # Common fallbacks
    for p in [BASE/'models', Path('.')/'FrameVision'/'models', Path('.')/'models']:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return BASE/'models'


LEGACY_BASES = [ROOT / "FrameVision", ROOT / "framevision", ROOT / "FrameLab"]
def _migrate_legacy_tree():
    base = BASE
    for p in ["output/video","output/trims","output/screenshots","output/descriptions","output/_temp",
              "jobs/pending","jobs/running","jobs/done","jobs/failed","logs"]:
        (base / p).mkdir(parents=True, exist_ok=True)
    for legacy in LEGACY_BASES:
        if not legacy.exists() or legacy == base:
            continue
        for rel in ["output/video","output/trims","output/screenshots","output/descriptions","output/_temp",
                    "jobs/pending","jobs/running","jobs/done","jobs/failed","logs"]:
            src = legacy / rel
            dst = base / rel
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for pth in src.rglob("*"):
                if pth.is_dir():
                    continue
                relp = pth.relative_to(src)
                target = dst / relp
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if not target.exists():
                        pth.replace(target)
                    else:
                        stem, suff = target.stem, target.suffix
                        alt = target.with_name(f"{stem}_migrated{suff}")
                        pth.replace(alt)
                except Exception:
                    pass

_migrate_legacy_tree()

JOBS = { "pending": BASE/"jobs"/"pending", "running": BASE/"jobs"/"running", "done": BASE/"jobs"/"done", "failed": BASE/"jobs"/"failed" }
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT = LOGS_DIR / "worker_heartbeat.txt"
PROGRESS_FILE = None
for p in JOBS.values(): p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE / "config.json"
MANIFEST_PATH = ROOT / "models_manifest.json"

# --- Robust model resolution helpers (idempotent) ---
def _find_manifest_entry(model_name: str, mani: dict):
    if not model_name:
        return None, None
    n = str(model_name).strip()
    if not n:
        return None, None
    nl = n.lower()
    for k, v in mani.items():
        if str(k).lower() == nl:
            return k, v
    if 'realesrgan' in nl or 'realesr' in nl: return 'RealESR-general-x4v3', mani.get('RealESR-general-x4v3', {})
    if 'swinir' in nl: return 'SwinIR-x4', mani.get('SwinIR-x4', {})
    if 'lapsrn' in nl: return 'LapSRN-x4', mani.get('LapSRN-x4', {})
    return None, None

def _resolve_model_exe(cfg: dict, mani: dict, model_name: str):
    # Prefer manifest; if not found, auto-detect under models_folder.
    # Works even when model_name is missing or doesn't include "realesr"/"swinir".
    canon, entry = _find_manifest_entry(model_name, mani)
    try:
        root = _resolve_models_folder(cfg)
    except Exception:
        root = ROOT

    # 1) Manifest-relative exe path
    exe_rel = (entry or {}).get('exe') or ''
    if exe_rel:
        exe_path = (root / exe_rel)
        if exe_path.exists() and exe_path.is_file():
            return (canon or model_name), exe_path

    # 2) Auto-detect executables under models folder
    # Candidate tags and common filenames
    CANDS = [
        ("realesrgan", ["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"]),
        ("swinir",    ["swinir-ncnn-vulkan.exe", "swinir-ncnn-vulkan"]),
        ("waifu2x",   ["waifu2x-ncnn-vulkan.exe", "waifu2x-ncnn-vulkan"]),
        ("lapsrn",    ["lapsrn-ncnn-vulkan.exe", "lapsrn"]),
    ]
    m = (canon or model_name or "").lower()
    # Prioritize by requested model tag if present
    ordered = CANDS
    if m:
        ordered = [c for c in CANDS if c[0] in m] + [c for c in CANDS if c[0] not in m]

    # Search by common names first
    try:
        for tag, names in ordered:
            for name in names:
                # exact file
                for p in root.rglob(name):
                    try:
                        if p.is_file() and (p.suffix.lower()=='.exe' or os.name!='nt'):
                            return (canon or tag), p
                    except Exception:
                        pass
                # wildcard around basename
                base = name.split('.')[0]
                for p in root.rglob(f"*{base}*"):
                    try:
                        if p.is_file() and (p.suffix.lower()=='.exe' or os.name!='nt'):
                            return (canon or tag), p
                    except Exception:
                        pass
    except Exception:
        pass

    # 3) Last resort: any executable under models folder
    try:
        any_pat = "*.exe" if os.name=="nt" else "*"
        for p in root.rglob(any_pat):
            try:
                if p.is_file():
                    return (canon or model_name or "realesrgan"), p
            except Exception:
                pass
    except Exception:
        pass

    return (canon or model_name), None


def load_config():
    try: return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception: return {}

def manifest():
    try: return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception: return {}

def run(cmd):
    try:
        LOGS = ROOT/"logs"; LOGS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = LOGS/f"run_{stamp}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join([str(x) for x in cmd]) + "\n\n")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                try: f.write(line)
                except Exception: pass
            code = p.wait()
            f.write(f"\nEXIT CODE: {code}\n")
        return code
    except Exception:
        return subprocess.call(cmd)


def _progress_set(pct: int):
    try:
        global PROGRESS_FILE
        if not PROGRESS_FILE:
            return
        p = Path(PROGRESS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        data = json.dumps({"pct": int(max(0, min(100, pct)))}, ensure_ascii=False)
        tmp.write_text(data, encoding="utf-8")
        try:
            tmp.replace(p)
        except Exception:
            # fallback write directly
            p.write_text(data, encoding="utf-8")
    except Exception:
        pass
def ffmpeg_path():
    cand = [ROOT/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), ROOT/"presets"/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), "ffmpeg"]
    for c in cand:
        try: subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
        except Exception: continue
        return str(c)
    return "ffmpeg"

FFMPEG = ffmpeg_path()


def _normalize_realesr_model(model_name: str, factor: int|float|str):
    """
    Accepts names like:
      - "realesrgan-x4plus", "realesrgan-x4plus-anime"
      - "realesr-general-x4v3", "realesr-general-wdn-x4v3"
      - "realesr-animevideov3-x2", "realesr-animevideov3-x3", "realesr-animevideov3-x4"
    Returns (base_name_without_scale_suffix, scale_int)
    """
    try:
        name = (model_name or "").strip()
    except Exception:
        name = ""
    # Pull trailing -xN if present
    m = re.search(r"(?i)(.*?)(?:-x(\d+))?$", name)
    base = (m.group(1) if m else name) or ""
    scale = int(m.group(2)) if (m and m.group(2)) else int(float(factor or 4))
    # Canonicalization for some common aliases
    aliases = {
        "RealESRGAN-x4plus": "realesrgan-x4plus",
        "RealESRGAN-anime-x4plus": "realesrgan-x4plus-anime",
        "RealESR-general-x4v3": "realesr-general-x4v3",
        "RealESR-general-wdn-x4v3": "realesr-general-wdn-x4v3",
    }
    base = aliases.get(base, base).lower()
    return base, int(scale)

def build_realesrgan_cmd(exe, inp, out, factor, model_name, models_dir=None, is_dir=False):
    base, s = _normalize_realesr_model(model_name, factor)
    cmd = [exe, "-i", str(inp), "-o", str(out), "-s", str(int(s)), "-n", base]
    if models_dir:
        cmd += ["-m", str(models_dir)]
    if is_dir:
        cmd += ["-f", "png"]
    return cmd


def build_swinir_cmd(exe, inp, out, factor):
    s = int(factor); return [exe, "-i", str(inp), "-o", str(out), "-s", str(s)]

def build_lapsrn_cmd(exe, inp, out, factor):
    s = int(factor); return [exe, "-i", str(inp), "-o", str(out), "-s", str(s)]


def upscale_video(job, cfg, mani):
    print("[worker] upscale_video: start", job.get("input"))
    inp = Path(job["input"]); out_dir = Path(job["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    factor = int(job["args"].get("factor", 4))
    model_name = job["args"].get("model","RealESRGAN-x4plus")
    model_name, exe_path = resolve_upscaler_exe(cfg, mani, model_name)
    out = out_dir / f"{inp.stem}_x{factor}.mp4"

    if exe_path and exe_path.exists() and exe_path.is_file():
        print(f"[worker] Using model '{model_name}' at {exe_path}")
        # 1) Extract frames
        frames = out_dir / f"{inp.stem}_x{factor}_frames"; frames.mkdir(parents=True, exist_ok=True)
        if run([FFMPEG,"-y","-i",str(inp), str(frames/"%06d.png")])!=0:
            return 1
        # 2) Upscale frames (batch dir-mode for RealESRGAN; per-frame for others)
        up = out_dir / f"{inp.stem}_x{factor}_up"; up.mkdir(parents=True, exist_ok=True)
        m = str(model_name).lower()
        if ("realesr" in m) or ("realesrgan" in m):
            cmd = build_realesrgan_cmd(str(exe_path), frames, up, factor, model_name, models_dir=Path(exe_path).parent, is_dir=True)
            if run(cmd)!=0:
                return 1
        else:
            for png in sorted(frames.glob("*.png")):
                if "swinir" in m:
                    cmd = build_swinir_cmd(str(exe_path), png, up/png.name, factor)
                elif "lapsrn" in m:
                    cmd = build_lapsrn_cmd(str(exe_path), png, up/png.name, factor)
                else:
                    cmd = [FFMPEG,"-y","-i",str(png),"-vf",f"scale=iw*{factor}:ih*{factor}:flags=lanczos","-frames:v","1",str(up/png.name)]
                if run(cmd)!=0:
                    return 1
        # 3) Re-encode (preserve input FPS when possible)
        enc = [FFMPEG,"-y","-i",str(up/"%06d.png"),"-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
        if run(enc)!=0:
            return 1
        try:
            _progress_set(100)
        except Exception:
            pass
        try:
            job['produced'] = str(out)
        except Exception:
            pass
        return 0
    else:
        # Fallback: ffmpeg scale (no model)
        print("[worker] No model exe found, using ffmpeg scale fallback")
        code = run([FFMPEG,"-y","-i",str(inp),"-vf",f"scale=iw*{factor}:ih*{factor}:flags=lanczos", str(out)])
        try:
            _progress_set(100)
        except Exception:
            pass
        if code == 0:
            try:
                job['produced'] = str(out)
            except Exception:
                pass
        return code
def tools_ffmpeg(job, cfg, mani):

    try:
        import shlex
        import pathlib  # ensure available even if not imported at module level
        args = job.get("args", {}) or {}
        cmd = args.get("ffmpeg_cmd") or args.get("cmd") or job.get("cmd")
        # Normalize to list
        if isinstance(cmd, str):
            try:
                cmd = shlex.split(cmd)
            except Exception:
                cmd = cmd.strip().split()
        if not isinstance(cmd, (list, tuple)) or len(cmd) == 0:
            try:
                job['error'] = "No ffmpeg command provided (expected args.ffmpeg_cmd, args.cmd, or job.cmd)."
            except Exception:
                pass
            return 1

        # Ensure output directory exists if job provides one
        try:
            out_dir = job.get("out_dir")
            if out_dir:
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # If args['outfile'] is set, create its parent
        try:
            outfile = args.get("outfile")
            if outfile:
                pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Record the normalized command for visibility
        try:
            job['cmd'] = " ".join(str(x) for x in cmd)
        except Exception:
            pass

        return run([str(x) for x in cmd])
    except Exception as e:
        try:
            job['error'] = f"tools_ffmpeg exception: {e}"
        except Exception:
            pass
        return 1
def upscale_photo(job, cfg, mani):
    print("[worker] upscale_photo: start", job.get("input"))
    inp = Path(job["input"]); out_dir = Path(job["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    factor = int(job["args"].get("factor", 4)); fmt = (job["args"].get("format") or "png").lower()
    model_name = job["args"].get("model","RealESRGAN-x4plus")
    model_name, exe_path = resolve_upscaler_exe(cfg, mani, model_name)
    out = out_dir / f"{inp.stem}_x{factor}.{fmt}"
    try:
        _progress_set(5)
    except Exception:
        pass

    # Decode-prep still images to a temp RGB PNG to avoid alpha/codec quirks
    src_in = inp
    try:
        ext = inp.suffix.lower()
        if Image is not None and ext in IMAGE_EXTS:
            tmp_rgb = out_dir / f"{inp.stem}_probe_rgb.png"
            im = Image.open(str(inp))
            try:
                im.seek(0)  # GIF: first frame
            except Exception:
                pass
            im = im.convert("RGB")
            im.save(str(tmp_rgb), format="PNG")
            if tmp_rgb.exists():
                src_in = tmp_rgb
        elif Image is None and ext in IMAGE_EXTS:
            # Fallback to ffmpeg if Pillow unavailable
            tmp_rgb = out_dir / f"{inp.stem}_probe_rgb.png"
            code = run([FFMPEG,"-y","-i",str(inp),"-vf","format=rgb24","-frames:v","1",str(tmp_rgb)])
            if code == 0 and tmp_rgb.exists():
                src_in = tmp_rgb
    except Exception:
        pass

    if exe_path and exe_path.exists() and exe_path.is_file():
        m = str(model_name).lower()
        if ("realesr" in m) or ("realesrgan" in m):
            models_dir_guess = None
            try:
                models_dir_guess = Path(exe_path).parent
            except Exception:
                models_dir_guess = None
            try:
                # If not clearly in a realsr folder, fall back to models/realesrgan under the configured models folder
                if not models_dir_guess or all(tag not in str(models_dir_guess).lower() for tag in ("realesr", "realesrgan")):
                    models_dir_guess = _resolve_models_folder(cfg) / "realesrgan"
            except Exception:
                pass
            cmd = build_realesrgan_cmd(str(exe_path), src_in, out, factor, model_name, models_dir=models_dir_guess)
        elif "swinir" in m:
            cmd = build_swinir_cmd(str(exe_path), src_in, out, factor)
        elif "lapsrn" in m:
            cmd = build_lapsrn_cmd(str(exe_path), src_in, out, factor)
        else:
            # Unknown model tag, fallback to ffmpeg scaling
            cmd = [FFMPEG,"-y","-i",str(src_in),"-vf",f"scale=iw*{factor}:ih*{factor}:flags=lanczos","-frames:v","1",str(out)]
        code = run(cmd)
        try:
            job['cmd'] = ' '.join([str(x) for x in cmd])
        except Exception:
            pass
        if code == 0 and not out.exists():
            _mark_error(job, 'Upscale finished but output file missing.')
            return 2
        return code

    # No model exe -> fallback upscale with ffmpeg
    cmd = [FFMPEG,"-y","-i",str(src_in),"-vf",f"scale=iw*{factor}:ih*{factor}:flags=lanczos","-frames:v","1",str(out)]
    code = run(cmd)
    try:
        job['cmd'] = ' '.join(str(x) for x in cmd)
    except Exception:
        pass
    if code == 0:
        try: job['produced'] = str(out)
        except Exception: pass
    return code

def handle_job(jpath: Path):
    job = json.loads(jpath.read_text(encoding="utf-8"))
    cfg = load_config(); mani = manifest()
    running = JOBS["running"] / jpath.name; jpath.rename(running)

    # progress sidecar path
    try:
        global PROGRESS_FILE
        PROGRESS_FILE = str(running.with_suffix(".progress.json"))
    except Exception:
        pass

    # Mark started
    try:
        job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        running.write_text(json.dumps(job, indent=2), encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()

    # Validate job type early
    t = job.get("type")
    if not t:
        print("WARN: job missing 'type' — skipping:", running)
        return 1

    # Smart reroute if extension says otherwise
    try:
        inp = Path(job.get("input",""))
        if t == "upscale_video" and is_image_path(inp):
            print("[worker] Reroute: image input detected; using upscale_photo")
            t = "upscale_photo"
        elif t == "upscale_photo" and is_video_path(inp):
            print("[worker] Reroute: video input detected; using upscale_video")
            t = "upscale_video"
        job["type"] = t
    except Exception:
        pass

    try:
        if t=="upscale_video": code = upscale_video(job, cfg, mani)
        elif t=="upscale_photo": code = upscale_photo(job, cfg, mani)
        elif t=='tools_ffmpeg': code = tools_ffmpeg(job, cfg, mani)
        elif t=='rife_interpolate':
            code = rife_interpolate(job, cfg, mani)
        else:
            _mark_error(job, f"Unknown job type: {t}")
            code = 2

        # mark finished
        try:
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job["duration_sec"] = int(time.time()-t0)
            running.write_text(json.dumps(job, indent=2), encoding="utf-8")
        except Exception:
            pass
        dest = JOBS["done"] if code==0 else JOBS["failed"]
        try:
            _progress_set(100)
        except Exception:
            pass
        (dest / running.name).write_text(json.dumps(job, indent=2), encoding="utf-8")
        try:
            running.unlink()
        except Exception:
            pass
        return code
    except Exception as e:
        try:
            _mark_error(job, str(e))
        except Exception:
            pass
        return 1

def main():
    print("FrameVision Worker V1.0. Watching jobs/pending in", JOBS["pending"])
    while True:
        try:
            HEARTBEAT.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        except Exception:
            pass
        items = sorted(JOBS["pending"].glob("*.json"))
        if not items:
            time.sleep(1.0)
            continue
        handle_job(items[0])

if __name__ == "__main__":
    main()

def _find_rife_exe(cfg: dict, mani: dict):
    names = ["rife-ncnn-vulkan.exe","rife-ncnn-vulkan"]
    cands = []
    try:
        td = cfg.get("tools_dir","")
        if td:
            for n in names:
                cands.append(Path(td)/"rife"/n)
    except Exception:
        pass
    for n in names:
        cands += [ROOT/'bin'/n, ROOT/'presets'/'bin'/n, ROOT/'tools'/'rife'/n, ROOT/n]
    for c in cands:
        if Path(c).exists():
            return str(c)
    return "rife-ncnn-vulkan.exe"



def build_rife_cmd(exe: str, inp: Path, out: Path, args: dict):
    cmd = [str(exe), "-i", str(inp), "-o", str(out)]
    g = int(args.get("gpu", 0)); th = int(args.get("threads", 0))
    if g >= 0: cmd += ["-g", str(g)]
    tfps = int(args.get("target_fps", 0)); fac = int(args.get("factor", 0))
    if tfps > 0: cmd += ["-r", str(tfps)]
    elif fac >= 2: cmd += ["-f", str(fac)]
    net = (args.get("network") or "").strip()
    if net: cmd += ["-n", net]
    models_dir = (args.get("models_dir") or "").strip()
    if models_dir: cmd += ["-m", models_dir]
    if th > 0: cmd += ["-j", str(th)]
    return cmd




def _build_rife_cmd_fallback(exe: str, inp: Path, outp: Path, args: dict, models_dir: str | None) -> list:
    cmd = [exe, "-i", str(inp), "-o", str(outp)]
    net = (args.get("network") or "").strip()
    if net: cmd += ["-n", net]
    gpu = int(args.get("gpu", 0) or 0)
    cmd += ["-g", str(gpu)]
    tfps = int(args.get("target_fps") or 0)
    fac = int(args.get("factor") or 0)
    if tfps > 0:
        cmd += ["-r", str(tfps)]
    elif fac > 0:
        cmd += ["-f", str(fac)]
    th = int(args.get("threads") or 0)
    if th > 0:
        cmd += ["-j", f"{th}:{max(1,th)}:{max(1,th)}"]
    if models_dir:
        cmd += ["-m", str(models_dir)]
    return cmd

def _deep_find_models_dir(root: Path, exe: str | None) -> str | None:
    # Look for a folder that contains rife-v4*/uhd/anime subfolders.
    allowed = ("rife-v4.6","rife-v4","rife-uhd","rife-anime")
    def contains_models(d: Path) -> bool:
        try:
            kids = [x.name.lower() for x in d.iterdir() if x.is_dir()]
        except Exception:
            return False
        return any(any(k.startswith(a) for k in kids) for a in allowed)
    # 1) ROOT/models
    m = root / "models"
    if m.exists() and contains_models(m): return str(m)
    # 2) any nested folder under ROOT/models
    if m.exists():
        for p in m.rglob("*"):
            if p.is_dir() and contains_models(p): return str(p)
    # 3) next to exe
    if exe:
        exedir = Path(exe).parent
        for cand in [exedir / "models"] + [p for p in (exedir).rglob("*") if p.is_dir()]:
            if contains_models(cand): return str(cand)
    return None
def _resolve_rife_exe(exe_override, cfg_root: Path) -> Path | None:
    # Try explicit override
    if exe_override:
        p = Path(exe_override)
        if p.exists():
            return p
        # Windows convenience: allow missing .exe
        try:
            if os.name == "nt" and not p.suffix:
                px = p.with_suffix(".exe")
                if px.exists():
                    return px
        except Exception:
            pass
    # Fallback to existing finder
    try:
        ex = _find_rife_exe(cfg_root, {})  # mani not needed for local search
        if ex and Path(ex).exists():
            return Path(ex)
    except Exception:
        pass
    return None
def rife_interpolate(job: dict, cfg: dict, mani: dict):
    inp = Path(job.get("input",""))
    out_dir = Path(job.get("out_dir",".")); out_dir.mkdir(parents=True, exist_ok=True)
    args = job.get("args",{}) or {}
    fmt = str(args.get("format","mp4")).lower()
    preview_sec = int(args.get("preview_seconds", 0))

    # Guard
    if not inp.exists():
        _mark_error(job, "Input file not found.")
        return 2
    if inp.suffix.lower() in (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"):
        _mark_error(job, "Selected file is an image, not a video.")
        return 2

    exe_override = (args.get('exe') or '').strip()
    # Try ONNX backend (future hook)
    try:
        from helpers.rife_core import ensure_bootstrap as _rife_bootstrap  # noqa
        _ = _rife_bootstrap(Path("."))
    except Exception:
        pass

    # If no exe available, fallback to FFmpeg minterpolate (CPU-only)
    if (not exe_override) and (not Path(exe).exists() if isinstance(exe, (str, bytes, os.PathLike)) else True):
        return _fallback_minterpolate_ffmpeg(job, inp, out_dir, args)

    exe = Path(exe_override) if (exe_override and os.path.exists(exe_override)) else _find_rife_exe(cfg, mani)
    if not exe or not Path(exe).exists():
        _mark_error(job, f"RIFE executable not found: {job.get('args',{}).get('exe')}")
        return 127
    job["tool_path"] = str(exe)

    # Resolve output name with pattern
    pattern = (args.get("filename_pattern") or "{name}_rife").strip()
    name = inp.stem
    base_name = (pattern.replace("{name}", name) or f"{name}_rife") + f".{fmt}"
    out_path = out_dir / base_name

    ow = str(args.get("overwrite","ask")).lower()
    if out_path.exists():
        if ow == "skip":
            job["skipped"] = True
            return 0
        if ow != "overwrite":
            # pick unique name
            i = 1
            while out_path.exists():
                out_path = out_dir / (base_name.replace(f".{fmt}", f"_{i}.{fmt}"))
                i += 1

    # If preview, build a clip first
    clip_in = inp
    if preview_sec > 0:
        try:
            FF = ffmpeg_path()
            tmp = out_dir / f"preview_src_{job.get('id','tmp')}.mp4"
            code = run([FF, "-y", "-ss", "0", "-t", str(preview_sec), "-i", str(inp), "-an", "-c", "copy", str(tmp)])
            if code != 0:
                _mark_error(job, "FFmpeg failed to create preview clip.")
                return code
            clip_in = tmp
        except Exception:
            pass

    # Streaming or direct
    streaming = bool(args.get("streaming", False))
    chunk = int(args.get("chunk_seconds", 0))
    if not streaming:
        cmd = build_rife_cmd(exe, clip_in, out_path, args)
        job["cmd"] = " ".join([str(x) for x in cmd])
        code = run(cmd)
        if code == 0:
            try: job['produced'] = str(out_path)
            except Exception: pass
        return code

    # streaming mode
    FF = ffmpeg_path()
    try:
        import subprocess as _sp
        _sp.check_output([FF, "-version"], stderr=_sp.STDOUT)
    except Exception:
        _mark_error(job, "FFmpeg not found but Streaming is ON.")
        return 126
    tmp = out_dir / f"tmp_rife_{job.get('id','tmp')}"; tmp.mkdir(parents=True, exist_ok=True)
    segtime = max(2, int(chunk))
    code = run([FF, "-y", "-i", str(clip_in), "-c","copy","-map","0","-segment_time", str(segtime), "-f","segment", str(tmp/"part_%04d.mp4")])
    if code != 0:
        _mark_error(job, "FFmpeg segmenting failed.")
        return code
    seglist = []
    for p in sorted(tmp.glob("part_*.mp4")):
        seg_out = tmp / f"out_{p.stem}.mp4"
        cmd_seg = build_rife_cmd(exe, p, seg_out, args)
        if (args.get('models_dir') or models_dir_resolved) and '-m' not in cmd_seg:
            cmd_seg += ['-m', str(args.get('models_dir') or models_dir_resolved)]
        code = run(cmd_seg)
        if code != 0:
            _mark_error(job, f"RIFE failed on segment {p.name}.")
            return code
        seglist.append(seg_out)
    listfile = tmp / "list.txt"
    listfile.write_text("\n".join([f"file '{s.as_posix()}'" for s in seglist]), encoding="utf-8")
    code = run([FF, "-y", "-f","concat","-safe","0","-i", str(listfile), "-c","copy", str(out_path)])
    if code != 0:
        _mark_error(job, "FFmpeg concat failed.")
    else:
        try: job['produced'] = str(out_path)
        except Exception: pass
    return code


def _fallback_minterpolate_ffmpeg(job: dict, inp: Path, out_dir: Path, args: dict) -> int:
    """CPU fallback using FFmpeg's minterpolate filter when rife exe is unavailable.
    This ensures out-of-box interpolation without extra downloads.
    """
    FF = ffmpeg_path()
    # Determine target FPS
    target_fps = int(args.get("target_fps") or 0)
    factor = int(args.get("factor") or 2)
    # Try to probe input fps using ffprobe
    def _ffprobe_path():
        cand = [ROOT/"bin"/('ffprobe.exe' if os.name=='nt' else 'ffprobe'), 'ffprobe']
        for c in cand:
            try:
                subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            except Exception:
                continue
            return str(c)
        return None
    if target_fps <= 0:
        fps = None
        FP = _ffprobe_path()
        if FP:
            try:
                out = subprocess.check_output([FP, "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", str(inp)], stderr=subprocess.STDOUT)
                txt = out.decode("utf-8", errors="ignore").strip()
                if "/" in txt:
                    a,b = txt.split("/"); fps = float(a)/float(b) if float(b)!=0 else None
                else:
                    fps = float(txt)
            except Exception:
                fps = None
        if fps and factor >= 2:
            target_fps = int(round(fps * factor))
        elif fps:
            target_fps = int(round(fps))
        else:
            target_fps = 60 if factor>=2 else 30

    # Output path (atomic write)
    out_name = (args.get("out_name") or (inp.stem + "_interp")).strip()
    fmt = str(args.get("format","mp4")).lower()
    out_path = out_dir / f"{out_name}.{fmt}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Construct ffmpeg command
    vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    cmd = [FF, "-y", "-i", str(inp), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "copy", str(tmp_path)]
    code = run(cmd)
    if code == 0:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        try:
            tmp_path.replace(out_path)
        except Exception:
            pass
        try: job['produced'] = str(out_path)
        except Exception: pass
    else:
        _mark_error(job, "FFmpeg minterpolate fallback failed.")
    return code


def _mark_error(job, msg):
    try:
        job['error'] = str(msg)
    except Exception:
        pass
