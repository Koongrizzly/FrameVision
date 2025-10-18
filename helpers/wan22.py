#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wan22.py — Offline runner for WAN 2.2 (TI2V-5B) on Windows/Python 3.11/CUDA.

This script is self-contained (no project-specific imports) and designed to run WAN 2.2 locally
without ComfyUI or any network access. It supports text2video and image2video modes, persistent
settings, probing, dry-run checks, and robust path resolution for models and ffmpeg.

Exit codes:
0 OK
2 missing weights/config
3 ffmpeg/ffprobe not found
4 CUDA requested but unavailable
5 WAN import failure (API not found)
6 OOM
7 invalid args
8 probe/dry-run failed
1 unexpected error
"""

# ---------- Offline mode must be set at the very, very top ----------
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Will be set later as well when parsing args to honor --cache-dir:
#   TRANSFORMERS_CACHE, HF_HOME

import sys
import argparse
import json
import logging
import logging.handlers
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Allowed third-party libs (must be installed locally, no network):
try:
    import torch
except Exception as e:  # pragma: no cover
    print("ERROR: PyTorch must be installed locally.", file=sys.stderr)
    sys.exit(7)

import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional libs
try:
    import imageio
except Exception:
    imageio = None  # type: ignore

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None  # type: ignore

# We will not require decord/cv2; PIL is enough for single image input.

# ----------------------------- Defaults -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Built-in defaults (these are merged with saved JSON, then CLI)
BUILTIN_DEFAULTS: Dict[str, Any] = {
    # Core generation
    "size": "480p",
    "width": 854,
    "height": 480,
    "frames": 48,
    "fps": 24,
    "seed": None,
    "dtype": "fp16",
    "device": "auto",  # auto->cuda if available else cpu
    "offload_model": False,
    "t5_cpu": False,
    "attn": "sdpa",  # sdpa|flashattn
    "chunk_frames": 8,

    # Paths & cache
    "model_root": str((SCRIPT_DIR / "models" / "wan22").as_posix()),
    "vae_root": None,  # default inside model_root
    "text_encoder_root": None,  # default inside model_root
    "cache_dir": str((SCRIPT_DIR / "hf_cache").as_posix()),

    # Binaries
    "ffmpeg": None,
    "ffprobe": None,

    # Logging prefs
    "log_level": "INFO",
    "json_logs": False,
    "no_file_log": False,

    # Prompt persistence (opt-in only)
    # "last_prompt": None,
    # "last_image": None,
}

# Settings JSON default path (resolved relative to script dir; with upward search)
DEFAULT_SETTINGS_PATH = SCRIPT_DIR / "presets" / "setsave" / "wan22.json"

# Rotating log defaults
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / "wan22.log"
CRASH_DIR = LOG_DIR / "crash"

# --------------------------- Utilities ---------------------------

def upward_search_for_folder(start: Path, target_rel: Path, max_hops: int = 3) -> Optional[Path]:
    """
    Search upward from 'start' for 'target_rel' subfolder (e.g., Path('models/wan22')).
    Returns the first found absolute path or None.
    """
    cur = start
    for _ in range(max_hops + 1):
        candidate = cur / target_rel
        if candidate.exists():
            return candidate.resolve()
        cur = cur.parent
    return None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B","KB","MB","GB","TB"]
    i = 0
    val = float(n)
    while val >= step and i < len(units)-1:
        val /= step
        i += 1
    return f"{val:.2f} {units[i]}"

# --------------------------- Logging ---------------------------

class NDJSONFormatter(logging.Formatter):
    """Simple JSON-lines formatter for file logs (console stays human)."""
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def setup_logging(level: str = "INFO", json_logs: bool = False, no_file_log: bool = False) -> logging.Logger:
    ensure_dir(LOG_DIR)
    logger = logging.getLogger("wan22")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    if not no_file_log:
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(NDJSONFormatter() if json_logs else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    return logger

# ------------------------ Settings I/O ------------------------

def default_settings_path_from_cli(cli_path: Optional[str]) -> Path:
    """
    Resolve settings path. If cli_path is relative, resolve to SCRIPT_DIR.
    Else, search upward for presets/setsave; else create it at SCRIPT_DIR.
    """
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = (SCRIPT_DIR / p).resolve()
        ensure_dir(p.parent)
        return p

    # Use default: SCRIPT_DIR/presets/setsave/wan22.json, with upward fallback
    default_dir = SCRIPT_DIR / "presets" / "setsave"
    found = upward_search_for_folder(SCRIPT_DIR, Path("presets") / "setsave", max_hops=3)
    if found is None:
        ensure_dir(default_dir)
        return default_dir / "wan22.json"
    else:
        return found / "wan22.json"

def load_settings(path: Path) -> Dict[str, Any]:
    """
    Load settings JSON if exists and merge over BUILTIN_DEFAULTS.
    """
    merged = dict(BUILTIN_DEFAULTS)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update({k: data[k] for k in data})
                merged["_loaded_settings"] = True
            else:
                merged["_loaded_settings"] = False
        except Exception:
            merged["_loaded_settings"] = False
    else:
        merged["_loaded_settings"] = False
    return merged

def save_settings(path: Path, settings: Dict[str, Any]) -> None:
    """
    Save selected keys to pretty JSON. Do not store prompt/image unless explicitly asked.
    """
    ensure_dir(path.parent)
    keys_to_store = [
        "size","width","height","frames","fps","dtype","device","offload_model","t5_cpu","attn","chunk_frames",
        "model_root","vae_root","text_encoder_root","cache_dir",
        "ffmpeg","ffprobe",
        "log_level","json_logs","no_file_log",
    ]
    out: Dict[str, Any] = {k: settings.get(k) for k in keys_to_store}

    # Optionally persist last prompt/image
    if settings.get("_remember_prompt", False):
        if settings.get("last_prompt") is not None:
            out["last_prompt"] = settings.get("last_prompt")
        if settings.get("last_image") is not None:
            out["last_image"] = settings.get("last_image")

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

# ------------------------ CLI ------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN 2.2 (TI2V-5B) offline runner")

    subparsers = parser.add_subparsers(dest="mode", required=False)
    # Modes: text2video | image2video
    parser_t2v = subparsers.add_parser("text2video", help="Generate video from text prompt")
    parser_i2v = subparsers.add_parser("image2video", help="Generate video from a starting image")

    # Shared/Global options (attach to root parser)
    parser.add_argument("--prompt", type=str, help="Text prompt (required for text2video)")
    parser.add_argument("--image", type=str, help="Path to starting image (required for image2video)")
    parser.add_argument("--out", type=str, default=str((SCRIPT_DIR / "output" / "wan22_out.mp4").as_posix()), help="Output mp4 path")

    parser.add_argument("--size", type=str, choices=["480p","720p"], default=None, help="Size preset (480p=854x480, 720p=1280x720)")
    parser.add_argument("--width", type=int, default=None, help="Width override")
    parser.add_argument("--height", type=int, default=None, help="Height override")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default random)")

    parser.add_argument("--dtype", type=str, choices=["fp16","bf16","fp32"], default=None, help="Computation dtype")
    parser.add_argument("--device", type=str, choices=["auto","cuda","cpu"], default=None, help="Device")
    parser.add_argument("--offload-model", action="store_true", help="Enable model offloading")
    parser.add_argument("--t5-cpu", action="store_true", help="Run T5/text-encoder on CPU")
    parser.add_argument("--attn", type=str, choices=["sdpa","flashattn"], default=None, help="Attention backend")
    parser.add_argument("--chunk-frames", type=int, default=None, help="Frames per generation chunk")

    parser.add_argument("--model-root", type=str, default=None, help="Path to WAN 2.2 model root (default ./models/wan22)")
    parser.add_argument("--vae-root", type=str, default=None, help="VAE root (default <model-root>/vae)")
    parser.add_argument("--text-encoder-root", type=str, default=None, help="Text encoder root (default <model-root>/text-encoder)")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache dir (no network; default ./hf_cache)")

    parser.add_argument("--ffmpeg", type=str, default=None, help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", type=str, default=None, help="Path to ffprobe executable")

    parser.add_argument("--settings", type=str, default=None, help="Path to settings JSON")
    parser.add_argument("--no-save-settings", action="store_true", help="Do not write settings after run")
    parser.add_argument("--reset-settings", action="store_true", help="Ignore and delete existing settings JSON for this run")
    parser.add_argument("--remember-prompt", action="store_true", help="Also store last prompt/image path in settings")
    parser.add_argument("--forget-prompt", action="store_true", help="Remove last prompt/image from settings")

    parser.add_argument("--probe", action="store_true", help="Probe environment and exit")
    parser.add_argument("--dry-run", action="store_true", help="Initialize and generate 1–2 frames to temp, then exit")

    parser.add_argument("--log-level", type=str, choices=["DEBUG","INFO","WARNING","ERROR"], default=None, help="Logging level")
    parser.add_argument("--no-file-log", action="store_true", help="Console-only logging")
    parser.add_argument("--json-logs", action="store_true", help="File logs as NDJSON")
    parser.add_argument("--trace", action="store_true", help="Include full traceback in errors and crash reports")

    parser.add_argument("--preset", type=str, choices=["lowvram","highvram"], default=None, help="Convenience presets")

    args = parser.parse_args(argv)

    # Mode inference if omitted:
    if args.mode is None:
        if args.prompt and not args.image:
            args.mode = "text2video"
        elif args.image and not args.prompt:
            args.mode = "image2video"
        # else remain None for probe/dry-run only

    return args

# ---------------------- Path resolution ----------------------

def which_exe(candidate: Optional[Union[str, Path]]) -> Optional[Path]:
    """Validate executable path or search in PATH."""
    if candidate:
        p = Path(candidate)
        if p.suffix == "" and sys.platform.startswith("win"):
            p = p.with_suffix(".exe")
        if p.exists() and p.is_file():
            return p.resolve()

    # Fallback to PATH
    if candidate:
        found = shutil.which(str(candidate))
        if found:
            return Path(found).resolve()
    return None

def resolve_ffmpeg_and_ffprobe(settings: Dict[str, Any], logger: logging.Logger) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    """
    Resolve ffmpeg and ffprobe according to the search order.
    Returns (ffmpeg_path, ffprobe_path, debug_notes).
    """
    notes: List[str] = []
    ffmpeg = settings.get("ffmpeg")
    ffprobe = settings.get("ffprobe")

    # 1) CLI-provided path (already in settings merged)
    if ffmpeg:
        p = which_exe(ffmpeg)
        if p:
            notes.append(f"ffmpeg: using CLI path {p}")
            ffmpeg_path = p
        else:
            notes.append(f"ffmpeg: CLI path invalid: {ffmpeg}")
            ffmpeg_path = None
    else:
        ffmpeg_path = None

    if ffprobe:
        p = which_exe(ffprobe)
        if p:
            notes.append(f"ffprobe: using CLI path {p}")
            ffprobe_path = p
        else:
            notes.append(f"ffprobe: CLI path invalid: {ffprobe}")
            ffprobe_path = None
    else:
        ffprobe_path = None

    # 2) script_dir/presets/bin
    preset_bin = SCRIPT_DIR / "presets" / "bin"
    for hop in range(0, 4):  # current + up to 3 parents
        base = SCRIPT_DIR if hop == 0 else SCRIPT_DIR.parents[hop-1]
        cand_dir = base / "presets" / "bin"
        if cand_dir.exists():
            if not ffmpeg_path:
                cand = cand_dir / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
                ffmpeg_path = cand if cand.exists() else ffmpeg_path
                if cand.exists():
                    notes.append(f"ffmpeg: found in {cand_dir}")
            if not ffprobe_path:
                cand = cand_dir / ("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe")
                ffprobe_path = cand if cand.exists() else ffprobe_path
                if cand.exists():
                    notes.append(f"ffprobe: found in {cand_dir}")
            break

    # 3) imageio_ffmpeg.get_ffmpeg_exe() (ffmpeg only)
    if not ffmpeg_path and imageio_ffmpeg is not None:
        try:
            cand = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if cand.exists():
                ffmpeg_path = cand.resolve()
                notes.append(f"ffmpeg: imageio_ffmpeg provided {ffmpeg_path}")
        except Exception:
            pass

    # 4) System PATH
    if not ffmpeg_path:
        found = shutil.which("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
        if found:
            ffmpeg_path = Path(found).resolve()
            notes.append(f"ffmpeg: found on PATH: {ffmpeg_path}")

    if not ffprobe_path:
        found = shutil.which("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe")
        if found:
            ffprobe_path = Path(found).resolve()
            notes.append(f"ffprobe: found on PATH: {ffprobe_path}")

    return ffmpeg_path, ffprobe_path, notes

def resolve_paths(settings: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Resolve model roots, cache, ffmpeg/ffprobe, and validate existence as required.
    """
    resolved = dict(settings)

    # Model root discovery
    model_root = settings.get("model_root") or BUILTIN_DEFAULTS["model_root"]
    model_root_path = Path(model_root)
    if not model_root_path.is_absolute():
        model_root_path = (SCRIPT_DIR / model_root_path).resolve()

    if not model_root_path.exists():
        # Upward search for models/wan22
        found = upward_search_for_folder(SCRIPT_DIR, Path("models") / "wan22", max_hops=3)
        if found:
            model_root_path = found
            logger.info(f"Auto-discovered model_root at {model_root_path}")
        else:
            # keep non-existing for now; will error later with exit 2
            pass
    resolved["model_root"] = str(model_root_path)

    # VAE and text encoder roots default inside model_root when omitted
    vae_root = settings.get("vae_root")
    if vae_root is None:
        vae_root_path = model_root_path / "vae"
    else:
        vae_root_path = Path(vae_root)
        if not vae_root_path.is_absolute():
            vae_root_path = (SCRIPT_DIR / vae_root_path).resolve()
    resolved["vae_root"] = str(vae_root_path)

    te_root = settings.get("text_encoder_root")
    if te_root is None:
        te_root_path = model_root_path / "text-encoder"
    else:
        te_root_path = Path(te_root)
        if not te_root_path.is_absolute():
            te_root_path = (SCRIPT_DIR / te_root_path).resolve()
    resolved["text_encoder_root"] = str(te_root_path)

    # Cache dir
    cache_dir = settings.get("cache_dir") or BUILTIN_DEFAULTS["cache_dir"]
    cache_dir_path = Path(cache_dir)
    if not cache_dir_path.is_absolute():
        cache_dir_path = (SCRIPT_DIR / cache_dir_path).resolve()
    ensure_dir(cache_dir_path)
    resolved["cache_dir"] = str(cache_dir_path)

    # Set relevant envs for offline/cache
    _os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir_path))
    _os.environ.setdefault("HF_HOME", str(cache_dir_path))

    # FFmpeg and ffprobe
    ffmpeg_path, ffprobe_path, notes = resolve_ffmpeg_and_ffprobe(resolved, logger)
    for n in notes:
        logger.debug(n)

    resolved["ffmpeg"] = str(ffmpeg_path) if ffmpeg_path else None
    resolved["ffprobe"] = str(ffprobe_path) if ffprobe_path else None

    return resolved

# ---------------------- Probe & device ----------------------

def probe_environment(resolved: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Gather environment info and resolved paths; also list top-level files in model dirs.
    """
    os_name = _os.name
    py = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    vram_free = vram_total = None
    if cuda_ok:
        try:
            free, total = torch.cuda.mem_get_info()
            vram_free, vram_total = free, total
        except Exception:
            pass

    def list_top(p: Union[str, Path]) -> List[str]:
        try:
            path = Path(p)
            if path.exists():
                return [x.name for x in path.iterdir()]
        except Exception:
            return []
        return []

    info = {
        "os": os_name,
        "python": py,
        "torch": torch_ver,
        "cuda_available": cuda_ok,
        "gpu": gpu_name,
        "vram_free": human_bytes(vram_free) if vram_free is not None else None,
        "vram_total": human_bytes(vram_total) if vram_total is not None else None,
        "model_root": resolved.get("model_root"),
        "vae_root": resolved.get("vae_root"),
        "text_encoder_root": resolved.get("text_encoder_root"),
        "model_root_files": list_top(resolved.get("model_root")),
        "vae_root_files": list_top(resolved.get("vae_root")),
        "text_encoder_root_files": list_top(resolved.get("text_encoder_root")),
        "ffmpeg": resolved.get("ffmpeg"),
        "ffprobe": resolved.get("ffprobe"),
        "settings_path": resolved.get("_settings_path"),
        "settings_loaded": bool(resolved.get("_loaded_settings")),
    }
    logger.info("Probe summary: " + json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return info

def select_device(settings: Dict[str, Any], logger: logging.Logger) -> Tuple[torch.device, torch.dtype]:
    """
    Decide on device and dtype; configure attention backend hints.
    """
    # Device
    want = (settings.get("device") or "auto").lower()
    if want == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif want == "cuda":
        if not torch.cuda.is_available():
            logger.error("CUDA requested but torch.cuda.is_available() is False.")
            sys.exit(4)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dtype
    dtype_str = (settings.get("dtype") or "fp16").lower()
    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Attention backend
    attn = (settings.get("attn") or "sdpa").lower()
    if attn == "flashattn":
        # Try to force PyTorch flash SDP; if unsupported, fallback to math/sdp
        try:
            # PyTorch 2.x flags
            _os.environ["PYTORCH_FORCE_SDP_KERNEL"] = "flash"
            if not torch.cuda.is_available():
                raise RuntimeError("Flash attention requires CUDA.")
            # Probe by calling dummy sdp?
            logger.info("Requested attention backend: flashattn. If unsupported, will fallback to SDPA.")
        except Exception as e:
            logger.warning(f"Flash attention unavailable ({e}); falling back to SDPA.")
            _os.environ["PYTORCH_FORCE_SDP_KERNEL"] = "math"
            settings["attn"] = "sdpa"
    else:
        _os.environ["PYTORCH_FORCE_SDP_KERNEL"] = "default"

    # Seed
    seed = settings.get("seed")
    if seed is not None:
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random as _random
            _random.seed(seed)
        except Exception:
            pass

    # Snapshot VRAM
    if device.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info()
            logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)}), VRAM free/total: {human_bytes(free)}/{human_bytes(total)}")
        except Exception:
            logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("Device: CPU")

    return device, dtype

# ---------------------- WAN model loading ----------------------

class WANHandle:
    """
    Minimal adapter wrapper around an imported WAN 2.2 API object/module.
    """
    def __init__(self, module: Any, entry_obj: Any, meta: Dict[str, Any]):
        self.module = module
        self.entry = entry_obj
        self.meta = meta

def try_import_wan(module_search_paths: List[Path], logger: logging.Logger) -> Optional[Any]:
    """
    Try to import a WAN module by temporarily extending sys.path with module_search_paths.
    """
    original = list(sys.path)
    try:
        for p in module_search_paths:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        # Try some likely module names; the actual local API should provide one of them.
        candidates = ["wan", "wan22", "wan_22", "ti2v", "wan_ti2v"]
        for name in candidates:
            try:
                mod = __import__(name)
                logger.debug(f"Imported WAN module '{name}' from {getattr(mod, '__file__', '?')}")
                return mod
            except Exception:
                continue
        return None
    finally:
        # keep added paths in sys.path for subsequent imports if needed; do not restore
        pass

def load_models(settings: Dict[str, Any], device: torch.device, dtype: torch.dtype, logger: logging.Logger) -> WANHandle:
    """
    Attempt to import and initialize the actual WAN 2.2 (TI2V-5B) API from local installation.
    If not found, exit code 5 with a clear message.
    """
    model_root = Path(settings["model_root"])
    vae_root = Path(settings["vae_root"])
    te_root = Path(settings["text_encoder_root"])

    # Basic validation for weights/config presence
    if not model_root.exists():
        logger.error(f"Model root not found: {model_root}")
        sys.exit(2)
    # It's hard to know exact files; require at least something inside
    try:
        if not any(model_root.iterdir()):
            logger.error(f"Model root is empty: {model_root}")
            sys.exit(2)
    except Exception:
        logger.error(f"Cannot read model root: {model_root}")
        sys.exit(2)

    # Try to import local WAN API by adding model_root and its parent to sys.path
    search_paths = [
        model_root,
        model_root.parent,
        SCRIPT_DIR,
    ]
    wan_mod = try_import_wan(search_paths, logger)
    if wan_mod is None:
        logger.error("WAN 2.2 API not found; install it locally and place weights under --model-root.")
        sys.exit(5)

    # Heuristics to get an entry object/class
    entry = None
    entry_names = [
        "TI2V_5B", "WAN22_T2V", "WAN_T2V", "WAN", "Pipeline", "load_ti2v_5b", "load"
    ]
    for name in entry_names:
        if hasattr(wan_mod, name):
            entry = getattr(wan_mod, name)
            break

    if entry is None:
        logger.error("WAN 2.2 API module found, but expected entry point not found "
                     "(tried: TI2V_5B, WAN22_T2V, WAN_T2V, WAN, Pipeline, load_ti2v_5b, load).")
        sys.exit(5)

    meta = {
        "module": getattr(wan_mod, "__name__", "unknown"),
        "module_file": getattr(wan_mod, "__file__", "unknown"),
        "entry": getattr(entry, "__name__", str(entry)),
    }

    # Instantiate or call factory if callable
    try:
        # Many WAN APIs use a factory function e.g. load_ti2v_5b(...)
        if callable(entry):
            try:
                model = entry(
                    model_root=str(model_root),
                    vae_root=str(vae_root),
                    text_encoder_root=str(te_root),
                    device=str(device),
                    dtype=str(dtype).split(".")[-1],
                    offload=settings.get("offload_model", False),
                    t5_cpu=settings.get("t5_cpu", False),
                    cache_dir=str(settings.get("cache_dir")),
                )
            except TypeError:
                # Fall back to more generic kwargs
                model = entry(
                    model_root=str(model_root),
                    device=str(device),
                )
        else:
            # If entry is a class, instantiate with minimal args
            model = entry(
                model_root=str(model_root),
                device=str(device),
            )
    except Exception as e:
        logger.error(f"WAN 2.2 API entry exists but could not be initialized: {e}")
        sys.exit(5)

    return WANHandle(wan_mod, model, meta)

# ---------------------- Generation helpers ----------------------

def _pil_from_any(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if x.ndim == 2:
            return Image.fromarray(x, mode="L")
        elif x.ndim == 3 and x.shape[2] in (1,3,4):
            mode = "L" if x.shape[2] == 1 else ("RGB" if x.shape[2] == 3 else "RGBA")
            return Image.fromarray(x, mode=mode)
    raise TypeError("Unsupported frame type; expected PIL.Image or HxWxC uint8 numpy array.")

def generate_text2video(prompt: str, handle: WANHandle, settings: Dict[str, Any], logger: logging.Logger) -> List[Image.Image]:
    """
    Generate frames via WAN text2video. Attempts a few common call signatures.
    """
    num_frames = int(settings["frames"])
    width = int(settings["width"])
    height = int(settings["height"])
    fps = int(settings["fps"])
    chunk = max(1, int(settings.get("chunk_frames", 8)))

    frames: List[Image.Image] = []
    remain = num_frames
    start_idx = 0

    logger.info(f"Generating {num_frames} frames at {width}x{height} in chunks of {chunk}...")

    while remain > 0:
        n = min(chunk, remain)
        ok = False
        # Try multiple APIs
        try_calls = [
            # Common-style
            dict(fn="generate", kwargs=dict(prompt=prompt, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="generate_video", kwargs=dict(prompt=prompt, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="text2video", kwargs=dict(prompt=prompt, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="sample", kwargs=dict(prompt=prompt, frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
        ]
        for spec in try_calls:
            fn_name = spec["fn"]
            if hasattr(handle.entry, fn_name):
                fn = getattr(handle.entry, fn_name)
                try:
                    out = fn(**spec["kwargs"])
                    ok = True
                    break
                except TypeError:
                    # try without start_frame
                    kw = dict(spec["kwargs"])
                    kw.pop("start_frame", None)
                    try:
                        out = fn(**kw)
                        ok = True
                        break
                    except Exception:
                        continue
                except Exception:
                    continue
        if not ok:
            logger.error("Could not find a working text2video API on the WAN handle.")
            sys.exit(5)

        # Normalize output to list[Image]
        if isinstance(out, (list, tuple)):
            chunk_frames = [_pil_from_any(x) for x in out]
        else:
            # Some APIs return an iterator/generator
            try:
                chunk_frames = [_pil_from_any(x) for x in list(out)]
            except Exception:
                logger.error("WAN returned unsupported output type for frames.")
                sys.exit(5)

        frames.extend(chunk_frames)
        remain -= len(chunk_frames)
        start_idx += len(chunk_frames)
        tqdm.update = lambda *a, **k: None  # no-op if tqdm not used
        logger.debug(f"Generated {len(chunk_frames)} frames (total {len(frames)}/{num_frames})")

        if len(chunk_frames) == 0:
            logger.error("WAN returned zero frames; aborting.")
            sys.exit(5)

    return frames[:num_frames]

def generate_image2video(image_path: Path, handle: WANHandle, settings: Dict[str, Any], logger: logging.Logger) -> List[Image.Image]:
    """
    Generate frames from a starting image (conditioning).
    """
    num_frames = int(settings["frames"])
    width = int(settings["width"])
    height = int(settings["height"])
    fps = int(settings["fps"])
    chunk = max(1, int(settings.get("chunk_frames", 8)))

    try:
        init_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {image_path} ({e})")
        sys.exit(7)

    frames: List[Image.Image] = []
    remain = num_frames
    start_idx = 0

    logger.info(f"Generating {num_frames} frames from image '{image_path.name}' at {width}x{height}...")

    while remain > 0:
        n = min(chunk, remain)
        ok = False
        try_calls = [
            dict(fn="image2video", kwargs=dict(prompt=None, image=init_img, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="generate", kwargs=dict(prompt=None, init_image=init_img, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="generate_video", kwargs=dict(prompt=None, init_image=init_img, num_frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
            dict(fn="sample", kwargs=dict(init_image=init_img, frames=n, width=width, height=height, fps=fps, start_frame=start_idx)),
        ]
        for spec in try_calls:
            fn_name = spec["fn"]
            if hasattr(handle.entry, fn_name):
                fn = getattr(handle.entry, fn_name)
                try:
                    out = fn(**spec["kwargs"])
                    ok = True
                    break
                except TypeError:
                    # Drop start_frame
                    kw = dict(spec["kwargs"])
                    kw.pop("start_frame", None)
                    try:
                        out = fn(**kw)
                        ok = True
                        break
                    except Exception:
                        continue
                except Exception:
                    continue
        if not ok:
            logger.error("Could not find a working image2video API on the WAN handle.")
            sys.exit(5)

        # Normalize output to list[Image]
        if isinstance(out, (list, tuple)):
            chunk_frames = [_pil_from_any(x) for x in out]
        else:
            try:
                chunk_frames = [_pil_from_any(x) for x in list(out)]
            except Exception:
                logger.error("WAN returned unsupported output type for frames.")
                sys.exit(5)

        frames.extend(chunk_frames)
        remain -= len(chunk_frames)
        start_idx += len(chunk_frames)
        logger.debug(f"Generated {len(chunk_frames)} frames (total {len(frames)}/{num_frames})")

        if len(chunk_frames) == 0:
            logger.error("WAN returned zero frames; aborting.")
            sys.exit(5)

    return frames[:num_frames]

# ---------------------- Video writing ----------------------

def write_video(frames: List[Image.Image], out_path: Path, fps: int, ffmpeg_path: Optional[Path], logger: logging.Logger) -> None:
    """
    Save frames to disk and mux with ffmpeg. If ffmpeg is None, attempt imageio fallback.
    """
    ensure_dir(out_path.parent)
    # Temp frames dir next to output
    frames_dir = out_path.parent / (out_path.stem + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    # Save frames to numbered PNGs
    for i, im in enumerate(frames):
        fn = frames_dir / f"{i:06d}.png"
        im.save(fn, format="PNG")

    if ffmpeg_path and ffmpeg_path.exists():
        cmd = [
            str(ffmpeg_path),
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_path),
        ]
        logger.info("Muxing with ffmpeg: " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed with code {e.returncode}")
            logger.error(e.stderr.decode("utf-8", errors="ignore"))
            raise
        finally:
            # Clean up frames
            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass
    else:
        logger.warning("ffmpeg not available; attempting imageio fallback (files may be larger).")
        if imageio is None:
            logger.error("imageio is not installed; cannot write video.")
            raise RuntimeError("No muxer available.")
        writer = imageio.get_writer(str(out_path), fps=fps)
        for im in frames:
            writer.append_data(np.asarray(im.convert("RGB")))
        writer.close()
        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

# ---------------------- Main ----------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Determine settings path and load
    settings_path = default_settings_path_from_cli(args.settings)
    settings = load_settings(settings_path)
    settings["_settings_path"] = str(settings_path)

    # Apply presets first (over defaults + loaded settings)
    if args.preset == "lowvram":
        settings.update({"size":"480p","width":854,"height":480,"frames":32,"chunk_frames":8,"t5_cpu":True,"attn":"sdpa"})
    elif args.preset == "highvram":
        settings.update({"size":"720p","width":1280,"height":720,"frames":48,"chunk_frames":12,"attn":"sdpa"})

    # Merge CLI overrides
    def set_if(name: str, val: Any):
        if val is not None:
            settings[name] = val

    set_if("prompt", args.prompt)
    set_if("image", args.image)
    set_if("out", args.out)
    set_if("size", args.size)
    set_if("width", args.width)
    set_if("height", args.height)
    set_if("frames", args.frames)
    set_if("fps", args.fps)
    set_if("seed", args.seed)
    set_if("dtype", args.dtype)
    set_if("device", args.device)
    if args.offload_model: settings["offload_model"] = True
    if args.t5_cpu: settings["t5_cpu"] = True
    set_if("attn", args.attn)
    set_if("chunk_frames", args.chunk_frames)

    set_if("model_root", args.model_root)
    set_if("vae_root", args.vae_root)
    set_if("text_encoder_root", args.text_encoder_root)
    set_if("cache_dir", args.cache_dir)

    set_if("ffmpeg", args.ffmpeg)
    set_if("ffprobe", args.ffprobe)

    set_if("log_level", args.log_level)
    if args.json_logs: settings["json_logs"] = True
    if args.no_file_log: settings["no_file_log"] = True
    settings["_remember_prompt"] = bool(args.remember_prompt)
    settings["_forget_prompt"] = bool(args.forget_prompt)
    settings["_no_save_settings"] = bool(args.no_save_settings)
    settings["_reset_settings"] = bool(args.reset_settings)
    settings["_probe"] = bool(args.probe)
    settings["_dry_run"] = bool(args.dry_run)
    settings["_mode"] = args.mode

    # Apply size presets if set
    if settings.get("size") == "480p":
        settings.setdefault("width", 854); settings.setdefault("height", 480)
    elif settings.get("size") == "720p":
        settings.setdefault("width", 1280); settings.setdefault("height", 720)

    # Ensure width/height present
    settings["width"] = int(settings.get("width") or 854)
    settings["height"] = int(settings.get("height") or 480)
    settings["frames"] = int(settings.get("frames") or 48)
    settings["fps"] = int(settings.get("fps") or 24)
    if settings.get("seed", None) is None:
        settings["seed"] = int(time.time() * 1000) % 2**31

    # Logging
    logger = setup_logging(settings.get("log_level","INFO"), bool(settings.get("json_logs", False)), bool(settings.get("no_file_log", False)))
    logger.debug("Initialized logger.")

    # Reset settings handling
    if settings.get("_reset_settings"):
        try:
            if settings_path.exists():
                settings_path.unlink()
                logger.info(f"Deleted settings: {settings_path}")
        except Exception as e:
            logger.warning(f"Could not delete settings file: {e}")

    # Resolve paths
    resolved = resolve_paths(settings, logger)
    resolved["_settings_path"] = str(settings_path)
    resolved["_loaded_settings"] = settings.get("_loaded_settings", False)

    # Validate ffmpeg & ffprobe presence as per spec
    # For probe/dry-run we still validate, otherwise exit 3
    ffmpeg_path = Path(resolved["ffmpeg"]) if resolved.get("ffmpeg") else None
    ffprobe_path = Path(resolved["ffprobe"]) if resolved.get("ffprobe") else None
    if ffmpeg_path is None or not ffmpeg_path.exists() or ffprobe_path is None or not ffprobe_path.exists():
        logger.error("ffmpeg/ffprobe not found or invalid.")
        sys.exit(3)

    # Probe only
    if settings.get("_probe"):
        info = probe_environment(resolved, logger)
        # Exit 0 on success
        sys.exit(0)

    # Mode validation
    mode = settings.get("_mode")
    if mode not in ("text2video","image2video"):
        logger.error("Mode not specified. Use 'text2video' or 'image2video', or provide --probe / --dry-run.")
        sys.exit(7)

    # Validate required inputs
    if mode == "text2video" and not settings.get("prompt"):
        logger.error("--prompt is required for text2video.")
        sys.exit(7)
    if mode == "image2video" and not settings.get("image"):
        logger.error("--image is required for image2video.")
        sys.exit(7)

    # Device & dtype
    device, dtype = select_device(settings, logger)

    # Load WAN model
    try:
        handle = load_models(resolved, device, dtype, logger)
        logger.info(f"WAN module '{handle.meta['module']}' entry '{handle.meta['entry']}' loaded from {handle.meta['module_file']}")
    except SystemExit:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.error(f"OOM or CUDA error while loading model: {e}")
            sys.exit(6)
        logger.error(f"Error while loading model: {e}")
        sys.exit(5)
    except Exception as e:
        logger.error(f"Unhandled error while loading model: {e}")
        sys.exit(5)

    # Dry run: generate 1–2 frames to temp dir, then clean up
    if settings.get("_dry_run"):
        try:
            test_frames = 2
            resolved["frames"] = test_frames
            tmp_out = Path(settings.get("out") or (SCRIPT_DIR / "output" / "wan22_out.mp4"))
            tmp_out = Path(tmp_out).with_name("wan22_dryrun.mp4")
            if mode == "text2video":
                frames = generate_text2video(settings["prompt"], handle, {**resolved, **settings, "frames": test_frames}, logger)
            else:
                frames = generate_image2video(Path(settings["image"]), handle, {**resolved, **settings, "frames": test_frames}, logger)
            # Write then delete
            write_video(frames, tmp_out, int(settings["fps"]), ffmpeg_path, logger)
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except Exception:
                    pass
            logger.info("Dry run successful.")
            # Save settings unless disabled
            if not settings.get("_no_save_settings"):
                # Handle remember/forget prompt
                if settings.get("_remember_prompt"):
                    if mode == "text2video":
                        settings["last_prompt"] = settings.get("prompt")
                        settings["last_image"] = None
                    else:
                        settings["last_image"] = str(Path(settings.get("image")).resolve())
                        settings["last_prompt"] = settings.get("prompt")
                if settings.get("_forget_prompt"):
                    settings.pop("last_prompt", None)
                    settings.pop("last_image", None)
                save_settings(settings_path, {**resolved, **settings})
            sys.exit(0)
        except SystemExit:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory during dry run: {e}")
                sys.exit(6)
            logger.error(f"Dry run failed: {e}")
            sys.exit(8)
        except Exception as e:
            logger.error(f"Dry run failed: {e}")
            sys.exit(8)

    # Full generation
    out_path = Path(settings.get("out") or (SCRIPT_DIR / "output" / "wan22_out.mp4"))
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    ensure_dir(out_path.parent)

    try:
        if mode == "text2video":
            frames = generate_text2video(settings["prompt"], handle, {**resolved, **settings}, logger)
        else:
            frames = generate_image2video(Path(settings["image"]), handle, {**resolved, **settings}, logger)
    except SystemExit:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Out of memory while generating: {e}")
            sys.exit(6)
        logger.error(f"Generation failed: {e}")
        sys.exit(5)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(5)

    try:
        write_video(frames, out_path, int(settings["fps"]), ffmpeg_path, logger)
        logger.info(f"Wrote video: {out_path}")
    except Exception as e:
        logger.error(f"Muxing failed: {e}")
        sys.exit(5)

    # Persist settings unless disabled
    if not settings.get("_no_save_settings"):
        if settings.get("_remember_prompt"):
            if mode == "text2video":
                settings["last_prompt"] = settings.get("prompt")
                settings["last_image"] = None
            else:
                settings["last_image"] = str(Path(settings.get("image")).resolve())
                settings["last_prompt"] = settings.get("prompt")
        if settings.get("_forget_prompt"):
            settings.pop("last_prompt", None)
            settings.pop("last_image", None)
        save_settings(settings_path, {**resolved, **settings})

    sys.exit(0)

# ---------------------- Entrypoint with crash report ----------------------

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # Allow our sys.exit codes to pass through without crash report
        raise
    except Exception as ex:
        # Crash report
        try:
            ensure_dir(CRASH_DIR)
            ts = time.strftime("%Y%m%d_%H%M%S")
            crash = {
                "time": ts,
                "args": sys.argv[1:],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "resolved_settings_path": str(default_settings_path_from_cli(None)),
                "exception": str(ex),
            }
            import traceback
            crash["traceback"] = traceback.format_exc()
            crash_path = CRASH_DIR / f"wan22_{ts}.json"
            with crash_path.open("w", encoding="utf-8") as f:
                json.dump(crash, f, indent=2)
            print(f"Crash report written to: {crash_path}", file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)



# === FrameVision UI pane: Wan22Pane ==========================================
try:
    from PySide6.QtCore import Qt, Signal, QProcess
    from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                                   QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                                   QTextEdit, QFileDialog, QCheckBox, QMessageBox)
except Exception:
    QWidget = object  # type: ignore

class Wan22Pane(QWidget):
    """
    Minimal GUI wrapper to run this script (wan22.py) for text→video or image→video.
    """
    fileReady = Signal(object)  # emits Path of the produced file

    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["text2video","image2video"])
        top.addWidget(QLabel("Mode:")); top.addWidget(self.cmb_mode); top.addStretch(1)
        v.addLayout(top)

        form = QFormLayout()
        self.ed_prompt = QLineEdit(); self.ed_prompt.setPlaceholderText("Write a clear, visual prompt...")
        self.ed_image = QLineEdit(); btn_img = QPushButton("Browse")
        def _pick_img():
            fn, _ = QFileDialog.getOpenFileName(self, "Pick image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)")
            if fn: self.ed_image.setText(fn)
        btn_img.clicked.connect(_pick_img)

        self.cmb_size = QComboBox(); self.cmb_size.addItems(["480p","720p"])
        self.spn_frames = QSpinBox(); self.spn_frames.setRange(1, 7200); self.spn_frames.setValue(48)
        self.spn_fps = QSpinBox(); self.spn_fps.setRange(1, 120); self.spn_fps.setValue(24)

        self.ed_out = QLineEdit()
        btn_out = QPushButton("Browse")
        def _pick_out():
            fn, _ = QFileDialog.getSaveFileName(self, "Output MP4", "", "Video (*.mp4);;All files (*.*)")
            if fn: self.ed_out.setText(fn)
        btn_out.clicked.connect(_pick_out)

        form.addRow("Prompt:", self.ed_prompt)
        row_img = QHBoxLayout(); row_img.addWidget(self.ed_image); row_img.addWidget(btn_img); w_img = QWidget(); w_img.setLayout(row_img)
        form.addRow("Start image:", w_img)
        form.addRow("Size preset:", self.cmb_size)
        row_frames = QHBoxLayout(); row_frames.addWidget(QLabel("Frames")); row_frames.addWidget(self.spn_frames); row_frames.addSpacing(16); row_frames.addWidget(QLabel("FPS")); row_frames.addWidget(self.spn_fps); w_frames = QWidget(); w_frames.setLayout(row_frames)
        form.addRow("Timeline:", w_frames)
        row_out = QHBoxLayout(); row_out.addWidget(self.ed_out); row_out.addWidget(btn_out); w_out = QWidget(); w_out.setLayout(row_out)
        form.addRow("Output:", w_out)
        v.addLayout(form)

        rowb = QHBoxLayout()
        self.btn_probe = QPushButton("Probe")
        self.btn_dry = QPushButton("Dry run")
        self.btn_run = QPushButton("Run")
        rowb.addWidget(self.btn_probe); rowb.addWidget(self.btn_dry); rowb.addStretch(1); rowb.addWidget(self.btn_run)
        v.addLayout(rowb)
        self.log = QTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log)

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(lambda: self._append_log(bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")))
        self.proc.finished.connect(self._on_finished)

        def _toggle():
            is_img = (self.cmb_mode.currentText() == "image2video")
            self.ed_image.setEnabled(is_img)
        self.cmb_mode.currentTextChanged.connect(lambda _: _toggle())
        _toggle()

        self.btn_probe.clicked.connect(self._do_probe)
        self.btn_dry.clicked.connect(lambda: self._launch(dry=True))
        self.btn_run.clicked.connect(lambda: self._launch(dry=False))

    def _append_log(self, s: str):
        try:
            self.log.append(s.rstrip())
        except Exception:
            pass

    def _script_path(self):
        from pathlib import Path
        return str(Path(__file__).resolve())

    def _launch(self, dry: bool):
        mode = self.cmb_mode.currentText()
        args = [self._script_path(), mode]
        if mode == "text2video":
            prompt = (self.ed_prompt.text() or "").strip()
            if not prompt:
                QMessageBox.information(self, "Prompt needed", "Please enter a prompt.")
                return
            args += ["--prompt", prompt]
        else:
            img = (self.ed_image.text() or "").strip()
            if not img:
                QMessageBox.information(self, "Image needed", "Please choose a starting image.")
                return
            args += ["--image", img]
        args += ["--size", self.cmb_size.currentText()]
        args += ["--frames", str(int(self.spn_frames.value())), "--fps", str(int(self.spn_fps.value()))]
        outp = (self.ed_out.text() or "").strip()
        if outp:
            args += ["--out", outp]
        if dry:
            args += ["--dry-run"]
        import sys
        self.log.clear()
        self._append_log("Launching: python " + " ".join(args))
        self.proc.start(sys.executable, args)

    def _do_probe(self):
        import sys
        self.log.clear()
        args = [self._script_path(), "--probe"]
        self.proc.start(sys.executable, args)

    def _on_finished(self, code, status):
        self._append_log(f"[process finished] code={code}")
        if code == 0:
            outp = (self.ed_out.text() or "").strip()
            if outp:
                from pathlib import Path
                p = Path(outp)
                if p.exists():
                    try:
                        self.fileReady.emit(p)
                    except Exception:
                        pass
