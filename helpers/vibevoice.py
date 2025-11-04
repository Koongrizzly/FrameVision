#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vibevoice.py — Run VibeVoice 1.5B fully offline (Windows-friendly, Python 3.11, CUDA).
No ComfyUI.

Features
- Offline-only: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE set; respects --cache-dir.
- Model/discovery: auto-discover ./models/vibevoice (or up to 3 parents) + optional 1.5B subfolder.
- FFmpeg/ffprobe discovery: CLI path → presets/bin → up to 3 parents → PATH.
- Persistent settings: load→merge→CLI override→(optional) save to ./presets/setsave/vibevoice.json.
- Modes: tts (single speaker) and multitts (timeline from JSON script).
- Probe: environment snapshot; Dry-run: 1s synth test and cleanup.
- Logging: console + rotating file; NDJSON option; crash reports with context.
- Post-processing (optional): normalize (peak), EBU loudnorm, arnndn denoise (if model provided).
- WAV output: always write PCM16 (or better via soundfile/torchaudio if available), mono by default.
- Exit codes: 0 OK; 2 missing weights; 3 ffmpeg needed; 4 CUDA requested but unavailable; 5 VibeVoice import failure; 6 OOM; 7 invalid args; 8 probe/dry-run failed.
"""
# --- Offline mode (critical) --------------------------------------------------
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Respect cache dir later (set via resolve_paths). We also set TRANSFORMERS_CACHE/HF_HOME when known.

# --- Standard / allowed libs --------------------------------------------------
import sys
import json
import time
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import subprocess
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# Heavy libs (allowed)
try:
    import torch
except Exception as _e:
    print("Warning: torch could not be imported:", _e, file=sys.stderr)
    torch = None  # type: ignore

# Optional audio libs (graceful fallback order)
_sf = None
_taudio = None
try:
    import soundfile as _sf  # highest quality / simplest
except Exception:
    _sf = None
try:
    import torchaudio as _taudio  # decent fallback
except Exception:
    _taudio = None

import numpy as np

# transformers/accelerate allowed for local model utilities if needed
try:
    import transformers  # noqa: F401
except Exception:
    transformers = None  # type: ignore
try:
    import accelerate  # noqa: F401
except Exception:
    accelerate = None  # type: ignore

# tqdm optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ----------------------------------------------------------------------------------
APP_NAME = "vibevoice"
APP_TITLE = "VibeVoice 1.5B Offline Runner"
DEFAULT_SAMPLE_RATE = 24000
VALID_SAMPLE_RATES = {22050, 24000, 44100, 48000}

# Exit codes
EXIT_OK = 0
EXIT_MISSING_WEIGHTS = 2
EXIT_FFMPEG_REQUIRED = 3
EXIT_CUDA_UNAVAILABLE = 4
EXIT_IMPORT_FAIL = 5
EXIT_OOM = 6
EXIT_INVALID_ARGS = 7
EXIT_PROBE_FAIL = 8

# -----------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO",
                  json_logs: bool = False,
                  no_file_log: bool = False,
                  script_dir: Optional[Path] = None) -> logging.Logger:
    """
    Setup console + optional rotating file logging.
    File logs go to ./logs/vibevoice.log (1MB x 5). If json_logs=True, file is NDJSON.
    """
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    if not no_file_log:
        base = script_dir or Path(__file__).resolve().parent
        log_dir = (base / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh_path = log_dir / f"{APP_NAME}.log"
        fh = RotatingFileHandler(fh_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        if json_logs:
            class NDJSONFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    data = {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
                        "level": record.levelname,
                        "name": record.name,
                        "msg": record.getMessage(),
                        "module": record.module,
                        "funcName": record.funcName,
                        "line": record.lineno,
                    }
                    return json.dumps(data, ensure_ascii=False)
            fh.setFormatter(NDJSONFormatter())
        else:
            fh_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

    return logger


def find_upward(start: Path, relative: Union[str, Path], max_hops: int = 3) -> Optional[Path]:
    """
    From 'start', search up to max_hops parents for 'relative' path existence.
    Returns the first existing path or None.
    """
    rel = Path(relative)
    cur = start
    for _ in range(max_hops + 1):
        candidate = (cur / rel)
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def is_executable(path: Optional[Union[str, Path]]) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and _is_exec(p)


def _is_exec(p: Path) -> bool:
    try:
        return _is_windows_exec(p) or _is_posix_exec(p)
    except Exception:
        return False


def _is_windows_exec(p: Path) -> bool:
    if _os.name == "nt":
        return p.suffix.lower() in (".exe", ".bat", ".cmd") and _can_run(p)
    return False


def _is_posix_exec(p: Path) -> bool:
    if _os.name != "nt":
        return _can_run(p)
    return False


def _can_run(p: Path) -> bool:
    try:
        return _os.access(str(p), _os.X_OK)
    except Exception:
        return False


def resolve_paths(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """
    Resolve script_dir, model_root, selected_model_dir (prefers '1.5B' if exists),
    cache_dir, settings_path, ffmpeg, ffprobe, logs/crash dir.
    Also sets TRANSFORMERS_CACHE/HF_HOME to cache_dir for offline use.
    """
    script_dir = Path(__file__).resolve().parent

    # Model root default: ./models/vibevoice, with upward auto-discovery if not present.
    default_model_root = script_dir / "models" / "vibevoice"
    model_root: Path
    if args.model_root:
        model_root = Path(args.model_root).resolve()
    else:
        if default_model_root.exists():
            model_root = default_model_root
        else:
            found = find_upward(script_dir, Path("models") / "vibevoice", 3)
            model_root = found.resolve() if found else default_model_root.resolve()

    # Selected subfolder: prefer '1.5B' under model_root if it exists
    selected_model_dir = model_root
    sub_15b = model_root / "1.5B"
    if sub_15b.exists():
        selected_model_dir = sub_15b

    # Cache dir
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
    else:
        cache_dir = (script_dir / "hf_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    _os.environ.setdefault("HF_HOME", str(cache_dir))
    _os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))

    # Settings path
    settings_path: Path
    if args.settings:
        settings_path = Path(args.settings)
        if not settings_path.is_absolute():
            # Resolve relative to script_dir or search upward for presets/setsave
            presets_save = find_upward(script_dir, Path("presets") / "setsave", 3)
            base = presets_save if presets_save else (script_dir / "presets" / "setsave")
            base.mkdir(parents=True, exist_ok=True)
            settings_path = (base / settings_path.name).resolve()
        else:
            settings_path = settings_path.resolve()
    else:
        presets_save = find_upward(script_dir, Path("presets") / "setsave", 3)
        base = presets_save if presets_save else (script_dir / "presets" / "setsave")
        base.mkdir(parents=True, exist_ok=True)
        settings_path = (base / "vibevoice.json").resolve()

    # FFmpeg / FFprobe resolution order
    def resolve_ff(bin_name: str, cli_path: Optional[str]) -> Optional[Path]:
        # 1) CLI if provided
        cand: Optional[Path] = None
        if cli_path:
            cp = Path(cli_path)
            if cp.exists():
                cand = cp
        # 2) presets/bin under script_dir
        if cand is None:
            exe = f"{bin_name}.exe" if _os.name == "nt" else bin_name
            local = script_dir / "presets" / "bin" / exe
            if local.exists():
                cand = local
        # 3) Search up to 3 parents for presets/bin
        if cand is None:
            exe = f"{bin_name}.exe" if _os.name == "nt" else bin_name
            up = find_upward(script_dir, Path("presets") / "bin" / exe, 3)
            if up and up.exists():
                cand = up
        # 4) PATH
        if cand is None:
            which = shutil.which(bin_name)
            if which:
                cand = Path(which)
        # Validate executable
        if cand and is_executable(cand):
            return cand.resolve()
        return None

    ffmpeg_path = resolve_ff("ffmpeg", args.ffmpeg)
    ffprobe_path = resolve_ff("ffprobe", args.ffprobe)

    # Logs dir for crash reports
    logs_dir = (script_dir / "logs")
    crash_dir = logs_dir / "crash"
    crash_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "script_dir": script_dir,
        "model_root": model_root,
        "selected_model_dir": selected_model_dir,
        "cache_dir": cache_dir,
        "settings_path": settings_path,
        "ffmpeg": ffmpeg_path,
        "ffprobe": ffprobe_path,
        "logs_dir": logs_dir,
        "crash_dir": crash_dir,
    }
    logger.debug(f"Resolved paths: { {k: str(v) if isinstance(v, Path) else v for k, v in info.items()} }")
    return info


def load_settings(settings_path: Path, defaults: Dict[str, Any], reset: bool, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load defaults, optionally merge settings file (unless reset), and return dict.
    """
    eff = defaults.copy()
    loaded = False
    if reset and settings_path.exists():
        try:
            settings_path.unlink()
            logger.info(f"Removed settings file: {settings_path}")
        except Exception as e:
            logger.warning(f"Could not delete settings file {settings_path}: {e}")

    if not reset and settings_path.exists():
        try:
            with settings_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                eff.update(data)
                loaded = True
        except Exception as e:
            logger.warning(f"Failed to read settings {settings_path}: {e}")
    logger.info(f"Settings loaded: {loaded} ({settings_path})")
    return eff


def save_settings(settings_path: Path, settings: Dict[str, Any], no_save: bool, logger: logging.Logger) -> None:
    """
    Persist settings as pretty JSON unless disabled.
    """
    if no_save:
        logger.info("--no-save-settings set; not writing settings JSON.")
        return
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info(f"Saved settings to {settings_path}")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")


def select_device(arg_device: Optional[str], logger: logging.Logger) -> str:
    """
    Choose 'cuda' if available (or requested), otherwise 'cpu'.
    Error when cuda requested explicitly but unavailable.
    """
    if torch is None:
        logger.warning("torch not available; falling back to CPU.")
        return "cpu"

    if arg_device in ("cuda", "cpu"):
        if arg_device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA requested but unavailable.")
            sys.exit(EXIT_CUDA_UNAVAILABLE)
        return arg_device

    # default auto→cuda
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype_from_str(dtype_str: str) -> "torch.dtype":
    if torch is None:
        return None  # type: ignore
    m = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return m.get(dtype_str.lower(), torch.float16)


def vram_snapshot(logger: logging.Logger) -> Dict[str, Any]:
    info = {"cuda_available": bool(torch and torch.cuda.is_available())}
    if torch and torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            name = torch.cuda.get_device_name(dev)
            free, total = torch.cuda.mem_get_info()
            info.update({
                "device_index": dev,
                "device_name": name,
                "vram_free": int(free),
                "vram_total": int(total),
            })
        except Exception as e:
            logger.debug(f"VRAM snapshot failed: {e}")
    return info


def import_vibevoice(logger: logging.Logger):
    """
    Import the actual VibeVoice API. Tries vendor paths first (project-local),
    then common package names. If all fail, exits with code 5.
    """
    # Prepend vendor paths (project local) so we don't need to pip-install
    try:
        root = Path(__file__).resolve().parents[1]
        for rel in ("third_party/vibevoice", "external/vibevoice", "deps/vibevoice"):
            p = (root / rel).resolve()
            if p.exists() and p.is_dir():
                sp = str(p)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                    try:
                        logger.debug(f"Added vendor path: {sp}")
                    except Exception:
                        pass
    except Exception:
        pass

    candidates = [
        "vibevoice",
        "vibevoice_tts",
        "vibevoice_1_5b",
        "VibeVoice",
    ]
    here = Path(__file__).resolve()
    last_err = None
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            # Self-import guard: skip if this file got imported as 'vibevoice'
            try:
                mod_file = Path(getattr(mod, "__file__", "") or "").resolve()
                if mod_file == here:
                    logger.debug(f"Skipped self-imported module '{name}' at {mod_file}")
                    continue
            except Exception:
                pass
            logger.info(f"Imported VibeVoice module: {name}")
            try:
                logger.debug(f"VibeVoice module path: {mod.__file__}")
            except Exception:
                pass
            return mod
        except Exception as e:
            last_err = e
            continue
    logger.error("VibeVoice 1.5B API not found; install locally and place weights under --model-root.")
    if last_err:
        logger.debug(f"Last import error: {last_err}")
    sys.exit(EXIT_IMPORT_FAIL)

def _ensure_top_level_api(vv_mod: Any, logger: logging.Logger):
    """
    If the imported 'vibevoice' package doesn't expose VibeVoiceTTS/from_pretrained
    at the top level, add thin wrappers from common submodules.
    """
    try:
        had_vvt = hasattr(vv_mod, "VibeVoiceTTS")
        had_fp  = hasattr(vv_mod, "from_pretrained")
        # Try common submodule
        tts_mod = None
        if not had_vvt or not had_fp:
            try:
                import importlib
                tts_mod = importlib.import_module(vv_mod.__name__ + ".tts")
            except Exception:
                tts_mod = None
        if not had_vvt and tts_mod and hasattr(tts_mod, "VibeVoiceTTS"):
            setattr(vv_mod, "VibeVoiceTTS", getattr(tts_mod, "VibeVoiceTTS"))
            logger.debug("Added alias: vibevoice.VibeVoiceTTS -> vibevoice.tts.VibeVoiceTTS")
        if not hasattr(vv_mod, "from_pretrained") and hasattr(vv_mod, "VibeVoiceTTS") and hasattr(vv_mod.VibeVoiceTTS, "from_pretrained"):
            def _vv_from_pretrained(pretrained_model_name_or_path, torch_dtype=None, device_map=None,
                                    local_files_only=True, cache_dir=None, attn_backend=None):
                return vv_mod.VibeVoiceTTS.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    local_files_only=local_files_only,
                    cache_dir=cache_dir,
                    attn_backend=attn_backend,
                )
            setattr(vv_mod, "from_pretrained", _vv_from_pretrained)
            logger.debug("Added wrapper: vibevoice.from_pretrained(...)")
    except Exception as e:
        logger.debug(f"Top-level API shim skipped: {e}")
    return vv_mod

def load_model(vv_mod,
               model_dir: Path,
               device: str,
               dtype_str: str,
               attn: str,
               cache_dir: Path,
               logger: logging.Logger):
    """
    Prefer TTS-capable classes (with synth methods) over base PreTrainedModel.
    Auto-discover across submodules and validate capabilities before accepting.
    """
    import importlib, inspect, pkgutil

    def _is_modelish(obj):
        return hasattr(obj, "to") or hasattr(obj, "eval")

    def _has_tts_api(obj):
        # Accept if any common TTS entrypoint exists
        for name in [
            "tts", "speak", "synthesize", "synthesise", "infer",
            "generate_audio", "generate_wav", "generate_speech",
            "text_to_speech", "forward_tts"
        ]:
            if hasattr(obj, name) and callable(getattr(obj, name)):
                return True
        return False

    dtype = dtype_from_str(dtype_str)
    dev = torch.device(device)

    model = None
    last_err = None

    # 1) Known entrypoints (fast path) — but require TTS API
    try_order = [
        ("VibeVoiceTTS.from_pretrained", lambda: getattr(vv_mod, "VibeVoiceTTS").from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            dtype=dtype,  # use new Transformers kw
            device_map=None,
            local_files_only=True,
            cache_dir=str(cache_dir),
            attn_backend=attn,
        )),
        ("VibeVoiceTTS.load", lambda: getattr(vv_mod, "VibeVoiceTTS").load(
            model_dir=str(model_dir),
            dtype=dtype_str,
            device=device,
            attn=attn,
        )),
        ("from_pretrained", lambda: getattr(vv_mod, "from_pretrained")(
            pretrained_model_name_or_path=str(model_dir),
            dtype=dtype,
            device_map=None,
            local_files_only=True,
            cache_dir=str(cache_dir),
            attn_backend=attn,
        )),
        ("load_pretrained", lambda: getattr(vv_mod, "load_pretrained")(
            model_dir=str(model_dir),
            dtype=dtype_str,
            device=device,
            attn=attn,
        )),
        ("load_tts", lambda: getattr(vv_mod, "load_tts")(
            model_dir=str(model_dir),
            dtype=dtype_str,
            device=device,
            attn=attn,
            cache_dir=str(cache_dir),
        )),
    ]
    for label, fn in try_order:
        try:
            candidate = fn()
            if not _is_modelish(candidate):
                raise TypeError(f"{label} returned non-model: {type(candidate).__name__}")
            # If it has a clear TTS API, accept immediately
            if _has_tts_api(candidate):
                model = candidate
                logger.info(f"Model loaded via {label}")
                break
            else:
                # Keep as fallback, but keep searching for a better TTS-capable class
                if model is None:
                    model = candidate
                    logger.info(f"Loaded base model via {label}; searching for TTS-capable wrapper...")
        except Exception as e:
            last_err = e
            continue

    # 2) If we don't yet have a TTS-capable model, auto-discover across submodules
    if model is None or not _has_tts_api(model):
        logger.debug("Auto-discovering TTS-capable loader in vibevoice package...")
        candidates = []

        def add_candidate(score, label, maker):
            candidates.append((score, label, maker))

        # Top-level classes
        try:
            for name, obj in inspect.getmembers(vv_mod):
                if inspect.isclass(obj):
                    n = name.lower()
                    score = 0
                    if "tts" in n: score += 5
                    if "voice" in n or "model" in n: score += 2
                    has_fp = hasattr(obj, "from_pretrained")
                    has_ld = hasattr(obj, "load")
                    if has_fp:
                        add_candidate(score + 2, f"{vv_mod.__name__}.{name}.from_pretrained",
                                      lambda o=obj: o.from_pretrained(
                                          pretrained_model_name_or_path=str(model_dir),
                                          dtype=dtype,
                                          device_map=None,
                                          local_files_only=True,
                                          cache_dir=str(cache_dir),
                                          attn_backend=attn))
                    if has_ld:
                        add_candidate(score + 1, f"{vv_mod.__name__}.{name}.load",
                                      lambda o=obj: o.load(
                                          model_dir=str(model_dir),
                                          dtype=dtype_str,
                                          device=device,
                                          attn=attn))
        except Exception as e:
            last_err = e

        # Submodules (prefer *.tts.* first)
        try:
            if hasattr(vv_mod, "__path__"):
                for m in pkgutil.walk_packages(vv_mod.__path__, vv_mod.__name__ + "."):
                    subname = m.name
                    try:
                        sm = importlib.import_module(subname)
                    except Exception:
                        continue
                    for name, obj in inspect.getmembers(sm):
                        if inspect.isclass(obj):
                            n = f"{subname}.{name}".lower()
                            score = 0
                            if ".tts" in subname.lower(): score += 6
                            if "tts" in n: score += 5
                            if "voice" in n or "model" in n: score += 2
                            has_fp = hasattr(obj, "from_pretrained")
                            has_ld = hasattr(obj, "load")
                            if has_fp:
                                add_candidate(score + 2, f"{subname}.{name}.from_pretrained",
                                              lambda o=obj: o.from_pretrained(
                                                  pretrained_model_name_or_path=str(model_dir),
                                                  dtype=dtype,
                                                  device_map=None,
                                                  local_files_only=True,
                                                  cache_dir=str(cache_dir),
                                                  attn_backend=attn))
                            if has_ld:
                                add_candidate(score + 1, f"{subname}.{name}.load",
                                              lambda o=obj: o.load(
                                                  model_dir=str(model_dir),
                                                  dtype=dtype_str,
                                                  device=device,
                                                  attn=attn))
        except Exception as e:
            last_err = e

        # Best-first
        candidates.sort(key=lambda x: -x[0])
        for _, label, maker in candidates:
            try:
                candidate = maker()
                if not _is_modelish(candidate):
                    raise TypeError(f"{label} returned non-model: {type(candidate).__name__}")
                if _has_tts_api(candidate):
                    model = candidate
                    logger.info(f"Model loaded via {label}")
                    break
                else:
                    # Keep only if we still don't have any model
                    if model is None:
                        model = candidate
                        logger.info(f"Loaded base model via {label}; still looking for TTS-capable wrapper...")
            except Exception as e:
                last_err = e
                continue

    if model is None:
        logger.error(f"Failed to load VibeVoice model from {model_dir}")
        if last_err:
            logger.debug(f"Last load error: {last_err}")
        sys.exit(EXIT_INVALID_ARGS)

    # Final guard: if we ended up with a base PreTrainedModel, try to attach a simple .tts alias if a suitable method exists
    if not _has_tts_api(model):
        for name in ["generate_audio", "generate_wav", "generate_speech", "infer"]:
            if hasattr(model, name) and callable(getattr(model, name)):
                setattr(model, "tts", getattr(model, name))
                logger.info(f"Attached tts -> {name} on model")
                break

    if hasattr(model, "eval"):
        model.eval()
    return model.to(dev)
def _ensure_mono(np_audio: np.ndarray) -> np.ndarray:
    if np_audio.ndim == 1:
        return np_audio
    if np_audio.ndim == 2:
        # Average channels
        return np.mean(np_audio, axis=0).astype(np.float32, copy=False)
    # Fallback: flatten
    return np_audio.reshape(-1).astype(np.float32, copy=False)


def _resample_if_needed(wave: np.ndarray, src_sr: int, tgt_sr: int, logger: logging.Logger) -> np.ndarray:
    if src_sr == tgt_sr:
        return wave
    # Try torchaudio first
    if _taudio is not None and torch is not None:
        try:
            tensor = torch.tensor(wave).unsqueeze(0)  # [1, T]
            resampler = _taudio.transforms.Resample(orig_freq=src_sr, new_freq=tgt_sr)
            out = resampler(tensor).squeeze(0).cpu().numpy()
            return out.astype(np.float32, copy=False)
        except Exception as e:
            logger.debug(f"torchaudio resample failed: {e}")
    # Numpy linear interpolation fallback
    try:
        x_old = np.linspace(0, 1, num=wave.shape[-1], endpoint=False, dtype=np.float64)
        x_new = np.linspace(0, 1, num=int(round(wave.shape[-1] * (tgt_sr / src_sr))), endpoint=False, dtype=np.float64)
        out = np.interp(x_new, x_old, wave.astype(np.float64)).astype(np.float32)
        return out
    except Exception as e:
        logger.warning(f"Resample failed; returning original SR {src_sr}: {e}")
        return wave


def _call_model_tts(model, text: str, sample_rate: int, speed: float, pitch: float, energy: float,
                    ref: Optional[Path], speaker: Optional[int], logger: logging.Logger) -> Tuple[np.ndarray, int]:
    """
    Attempt a few common method names to synthesize audio with VibeVoice.
    Returns (audio_float32_mono, sr).
    """
    kwargs_common = {
        "text": text,
        "sample_rate": sample_rate,
        "speed": speed,
        "pitch": pitch,
        "energy": energy,
    }
    if ref:
        kwargs_common["voice_ref"] = str(ref)

    # Some APIs use 'speaker' / 'speaker_id'
    if speaker is not None:
        kwargs_common["speaker"] = speaker
        kwargs_common["speaker_id"] = speaker

    call_order = [
        "tts",
        "generate",
        "synthesize",
        "infer",
    ]
    last_err = None
    for name in call_order:
        if hasattr(model, name):
            try:
                out = getattr(model, name)(**kwargs_common)
                # normalize output to float32 mono numpy and SR
                if isinstance(out, tuple) and len(out) == 2:
                    audio, sr = out
                else:
                    audio, sr = out, sample_rate
                if torch is not None and isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().float().numpy()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                audio = _ensure_mono(audio)
                return audio, int(sr)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.getLogger(APP_NAME).error("CUDA OOM during synthesis.")
                    sys.exit(EXIT_OOM)
                last_err = e
            except Exception as e:
                last_err = e
                continue
    logger.error("Could not find a working synthesis method on VibeVoice model.")
    if last_err:
        logger.debug(f"Last synthesis error: {last_err}")
    # If we get here, it's a fatal usage error with the local API
    sys.exit(EXIT_IMPORT_FAIL)


def synthesize_single(model,
                      text: str,
                      sample_rate: int,
                      speed: float,
                      pitch: float,
                      energy: float,
                      ref: Optional[Path],
                      speaker: Optional[int],
                      logger: logging.Logger) -> np.ndarray:
    audio, src_sr = _call_model_tts(model, text, sample_rate, speed, pitch, energy, ref, speaker, logger)
    audio = _resample_if_needed(audio, src_sr, sample_rate, logger)
    audio = _ensure_mono(audio)
    return audio


def synthesize_multi(model,
                     script: Dict[str, Any],
                     sample_rate: int,
                     speed: float,
                     pitch: float,
                     energy: float,
                     logger: logging.Logger) -> np.ndarray:
    """
    script format: {"segments": [{"speaker": 1, "text": "Hello", "ref": "path.wav", "pause": 0.3}, ...]}
    """
    segments = script.get("segments", [])
    if not isinstance(segments, list) or len(segments) == 0:
        logger.error("Script JSON missing or empty 'segments'.")
        sys.exit(EXIT_INVALID_ARGS)
    out: List[np.ndarray] = []
    for idx, seg in enumerate(tqdm(segments, desc="Synthesizing segments")):
        if not isinstance(seg, dict) or "text" not in seg:
            logger.error(f"Invalid segment at index {idx}: {seg}")
            sys.exit(EXIT_INVALID_ARGS)
        text = str(seg.get("text", ""))
        if not text.strip():
            pause = float(seg.get("pause", 0.0) or 0.0)
            if pause > 0:
                out.append(np.zeros(int(round(pause * sample_rate)), dtype=np.float32))
            continue
        speaker = seg.get("speaker", None)
        ref = seg.get("ref", None)
        ref_path = Path(ref).resolve() if ref else None
        audio = synthesize_single(model, text, sample_rate, speed, pitch, energy, ref_path, speaker, logger)
        out.append(audio)
        pause = float(seg.get("pause", 0.0) or 0.0)
        if pause > 0:
            out.append(np.zeros(int(round(pause * sample_rate)), dtype=np.float32))
    if len(out) == 0:
        return np.zeros(int(sample_rate * 0.1), dtype=np.float32)
    return np.concatenate(out, axis=0)


def write_wav(out_path: Path, audio: np.ndarray, sample_rate: int, logger: logging.Logger) -> None:
    """
    Write audio to WAV at PCM16 (or better, if backend available), ensuring mono and SR.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = _ensure_mono(np.asarray(audio, dtype=np.float32))

    # Prefer soundfile -> torchaudio -> wave (builtin) with int16 conversion
    if _sf is not None:
        try:
            _sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
            logger.info(f"Wrote WAV via soundfile: {out_path}")
            return
        except Exception as e:
            logger.warning(f"soundfile write failed: {e}")

    if _taudio is not None and torch is not None:
        try:
            tens = torch.tensor(audio).unsqueeze(0)  # [1, T]
            _taudio.save(str(out_path), tens, sample_rate, bits_per_sample=16, encoding="PCM_S")
            logger.info(f"Wrote WAV via torchaudio: {out_path}")
            return
        except Exception as e:
            logger.warning(f"torchaudio write failed: {e}")

    # Fallback: wave
    try:
        import wave
        # float32 -> int16 with clipping
        a = np.clip(audio, -1.0, 1.0)
        a = (a * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(a.tobytes())
        logger.info(f"Wrote WAV via wave: {out_path}")
    except Exception as e:
        logger.error(f"Failed to write WAV: {e}")
        raise


def _ffmpeg_check(ffmpeg: Optional[Path], steps_needed: List[str], logger: logging.Logger):
    if any(steps_needed) and not ffmpeg:
        logger.error(f"FFmpeg required for: {', '.join([s for s in steps_needed if s])}, but not found.")
        sys.exit(EXIT_FFMPEG_REQUIRED)


def _run_ffmpeg(ffmpeg: Path, cmd_args: List[str], logger: logging.Logger) -> None:
    cmd = [str(ffmpeg)] + cmd_args
    logger.debug(f"Running ffmpeg: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        logger.error(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")


def _ff_volumedetect(ffmpeg: Path, in_path: Path, logger: logging.Logger) -> float:
    """
    Run a volumedetect pass to find max_volume in dB. Returns needed gain in dB to normalize to 0 dBFS.
    """
    # Use null muxer; on Windows, output can be NUL
    null_sink = "NUL" if _os.name == "nt" else "/dev/null"
    cmd = [
        "-hide_banner", "-y",
        "-i", str(in_path),
        "-af", "volumedetect",
        "-f", "null",
        null_sink
    ]
    proc = subprocess.run([str(ffmpeg)] + cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    # Parse "max_volume: -X.XX dB"
    import re
    m = re.search(r"max_volume:\s*([-\d\.]+)\s*dB", stderr)
    if not m:
        logger.warning("volumedetect did not report max_volume; skipping peak normalize.")
        return 0.0
    max_db = float(m.group(1))
    gain_db = -max_db  # bring peak to 0 dBFS
    # small safety to avoid clipping
    gain_db = min(gain_db, 0.0)
    return gain_db


def post_process_ffmpeg(ffmpeg: Optional[Path],
                        in_wav: Path,
                        sample_rate: int,
                        normalize: bool,
                        loudnorm: bool,
                        denoise: bool,
                        denoise_model: Optional[Path],
                        logger: logging.Logger) -> Path:
    """
    Run optional post-processing in sequence: denoise -> loudnorm -> normalize.
    Returns path to final file (may be the same as input if nothing done).
    """
    needed = []
    if normalize:
        needed.append("normalize")
    if loudnorm:
        needed.append("loudnorm")
    if denoise:
        needed.append("denoise")
    if needed:
        _ffmpeg_check(ffmpeg, needed, logger)

    if not ffmpeg or not needed:
        return in_wav

    work_in = in_wav
    tmp_files: List[Path] = []

    def mktemp(suffix: str) -> Path:
        p = in_wav.with_name(in_wav.stem + f"_{random.randint(1000,9999)}{suffix}")
        tmp_files.append(p)
        return p

    try:
        # 1) Denoise
        if denoise:
            if denoise_model and Path(denoise_model).exists():
                out1 = mktemp("_den.wav")
                _run_ffmpeg(ffmpeg, [
                    "-hide_banner", "-y",
                    "-i", str(work_in),
                    "-af", "arnndn=m='" + str(denoise_model).replace("'", "\\'") + "'",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(out1)
                ], logger)
                work_in = out1
            else:
                logger.info("Denoise requested but no arnndn model provided; skipping denoise. Use --denoise-model path.")

        # 2) Loudness normalization (EBU R128, -23LUFS)
        if loudnorm:
            out2 = mktemp("_ln.wav")
            _run_ffmpeg(ffmpeg, [
                "-hide_banner", "-y",
                "-i", str(work_in),
                "-af", "loudnorm=I=-23:TP=-2:LRA=11",
                "-ar", str(sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(out2)
            ], logger)
            work_in = out2
        # 3) Peak normalize
        elif normalize:
            gain_db = _ff_volumedetect(ffmpeg, work_in, logger)
            if gain_db < 0.0:  # need to raise
                out3 = mktemp("_pn.wav")
                _run_ffmpeg(ffmpeg, [
                    "-hide_banner", "-y",
                    "-i", str(work_in),
                    "-af", f"volume={gain_db}dB",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(out3)
                ], logger)
                work_in = out3
            else:
                logger.info("Peak normalize: no positive gain needed; skipping.")

        # Replace input with final processed file
        if work_in != in_wav:
            shutil.move(str(work_in), str(in_wav))
        # Cleanup temps (already moved final)
        for p in tmp_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        return in_wav
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise


def probe_environment(paths: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Probe: OS, Python, Torch/CUDA, VRAM, resolved paths, short model listing.
    """
    env = {
        "os_name": _os.name,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "torch_available": bool(torch),
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "device_info": vram_snapshot(logger),
        "model_root": str(paths["model_root"]),
        "selected_model_dir": str(paths["selected_model_dir"]),
        "model_dir_exists": paths["selected_model_dir"].exists(),
        "ffmpeg": str(paths["ffmpeg"]) if paths["ffmpeg"] else None,
        "ffprobe": str(paths["ffprobe"]) if paths["ffprobe"] else None,
        "settings_path": str(paths["settings_path"]),
        "settings_exists": Path(paths["settings_path"]).exists(),
        "cache_dir": str(paths["cache_dir"]),
    }
    # short file listing
    try:
        mdl = paths["selected_model_dir"]
        if mdl.exists():
            files = [p.name for p in list(mdl.glob("*"))[:20]]
            env["model_dir_listing"] = files
    except Exception:
        pass
    return env


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vibevoice.py",
        description="VibeVoice 1.5B offline runner (tts | multitts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", nargs="?", choices=["tts", "multitts"],
                   help="Operation mode. Use --probe for environment checks or --dry-run for a quick synthesis test.")
    # Inputs
    p.add_argument("--text", type=str, default=None, help="Text to synthesize (UTF-8).")
    p.add_argument("--text-file", type=str, default=None, help="Path to UTF-8 text file.")
    p.add_argument("--script", type=str, default=None, help="Path to multitts script JSON.")
    p.add_argument("--out", type=str, default=str(Path("output") / "vibevoice_out.wav"), help="Output WAV path.")

    # Core synthesis controls
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Output sample rate (22050/24000/44100/48000).")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Computation dtype.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Compute device (default auto→cuda if available).")
    p.add_argument("--attn", type=str, default="sdpa", choices=["sdpa", "flashattn"], help="Attention backend.")
    p.add_argument("--speed", type=float, default=1.0, help="Prosody speed ratio (1.0=normal).")
    p.add_argument("--pitch", type=float, default=0.0, help="Prosody pitch shift in semitones (0.0=neutral).")
    p.add_argument("--energy", type=float, default=1.0, help="Prosody energy/emphasis (1.0=neutral).")

    # References / speakers
    p.add_argument("--ref", type=str, default=None, help="Reference voice path for single-speaker tts (optional).")
    p.add_argument("--speaker-count", type=int, default=1, choices=[1, 2, 3, 4], help="Number of speakers to consider.")

    # Post-processing
    p.add_argument("--normalize", action="store_true", help="Simple peak normalization via ffmpeg.")
    p.add_argument("--loudnorm", action="store_true", help="EBU R128 loudness normalization (-23 LUFS) via ffmpeg.")
    p.add_argument("--denoise", action="store_true", help="Light spectral denoise via ffmpeg arnndn (requires model).")
    p.add_argument("--denoise-model", type=str, default=None, help="Path to ffmpeg arnndn model (.onnx).")

    # Probe / logging / misc
    p.add_argument("--probe", action="store_true", help="Probe environment and exit 0 on success.")
    p.add_argument("--dry-run", action="store_true", help="Initialize model and synthesize ~1 second test audio.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Log verbosity.")
    p.add_argument("--no-file-log", action="store_true", help="Disable file logging (console only).")
    p.add_argument("--json-logs", action="store_true", help="Write NDJSON logs to file (console stays human-readable).")
    p.add_argument("--trace", action="store_true", help="Include full traceback in crash reports.")

    # Paths / settings
    p.add_argument("--model-root", type=str, default=None, help="Root folder for models (expects vibevoice/[1.5B]).")
    p.add_argument("--cache-dir", type=str, default=None, help="Transformers/HF cache directory (offline).")
    p.add_argument("--ffmpeg", type=str, default=None, help="Path to ffmpeg executable.")
    p.add_argument("--ffprobe", type=str, default=None, help="Path to ffprobe executable.")
    p.add_argument("--settings", type=str, default=None, help="Path to persistent settings JSON.")

    # Settings toggles
    p.add_argument("--no-save-settings", action="store_true", help="Do not write settings JSON after run.")
    p.add_argument("--reset-settings", action="store_true", help="Ignore and delete existing settings JSON for this run.")
    p.add_argument("--remember-text", action="store_true", help="Also store last text/script path into settings.")
    p.add_argument("--forget-text", action="store_true", help="Remove stored last text/script from settings.")

    # Repro
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (if used by model).")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    logger = setup_logging(args.log_level, args.json_logs, args.no_file_log, script_dir)

    try:
        paths = resolve_paths(args, logger)

        # Prepare defaults (merged with any existing settings then overridden by CLI).
        defaults: Dict[str, Any] = {
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "dtype": "fp16",
            "device": "cuda" if (torch and torch.cuda.is_available()) else "cpu",
            "attn": "sdpa",
            "speed": 1.0,
            "pitch": 0.0,
            "energy": 1.0,
            "denoise": False,
            "normalize": False,
            "voice_ref_paths": [],
            "speaker_count": 1,
            "model_root": str(paths["model_root"]),
            "cache_dir": str(paths["cache_dir"]),
            "ffmpeg": str(paths["ffmpeg"]) if paths["ffmpeg"] else "",
            "ffprobe": str(paths["ffprobe"]) if paths["ffprobe"] else "",
            "log_level": args.log_level,
            "json_logs": bool(args.json_logs),
            "no_file_log": bool(args.no_file_log),
        }

        settings = load_settings(paths["settings_path"], defaults, args.reset_settings, logger)

        # CLI overrides
        if args.sample_rate:
            settings["sample_rate"] = int(args.sample_rate)
        if args.dtype:
            settings["dtype"] = args.dtype
        if args.device:
            settings["device"] = args.device
        if args.attn:
            settings["attn"] = args.attn
        if args.speed is not None:
            settings["speed"] = float(args.speed)
        if args.pitch is not None:
            settings["pitch"] = float(args.pitch)
        if args.energy is not None:
            settings["energy"] = float(args.energy)
        if args.speaker_count:
            settings["speaker_count"] = int(args.speaker_count)
        if args.ffmpeg:
            settings["ffmpeg"] = args.ffmpeg
        if args.ffprobe:
            settings["ffprobe"] = args.ffprobe
        if args.model_root:
            settings["model_root"] = str(Path(args.model_root).resolve())
        if args.cache_dir:
            settings["cache_dir"] = str(Path(args.cache_dir).resolve())

        # Save/forget last text/script toggles applied later after success
        last_text: Optional[str] = None
        last_script: Optional[str] = None

        # Probe mode (no model init or synthesis required)
        if args.probe:
            env_info = probe_environment(paths, logger)
            print(json.dumps(env_info, ensure_ascii=False, indent=2))
            return EXIT_OK

        # Validate sample rate
        if int(settings["sample_rate"]) not in VALID_SAMPLE_RATES:
            logger.error(f"Invalid sample rate: {settings['sample_rate']}")
            return EXIT_INVALID_ARGS

        # Device selection
        device = select_device(args.device, logger)
        logger.info(f"Selected device: {device}")
        dev_info = vram_snapshot(logger)
        logger.info(f"Device snapshot: {dev_info}")

        # Seed
        if args.seed is not None and torch is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            logger.info(f"Seed set to {args.seed}")

        # Import model API (must exist locally) and load model
        vv_mod = import_vibevoice(logger)
        vv_mod = _ensure_top_level_api(vv_mod, logger)
        model = load_model(vv_mod,
                           paths["selected_model_dir"],
                           device,
                           settings["dtype"],
                           settings["attn"],
                           Path(settings["cache_dir"]),
                           logger)

        # Dry run: do a quick 1-second synth
        if args.dry_run:
            test_text = "Vibe voice test."
            audio = synthesize_single(model, test_text, int(settings["sample_rate"]),
                                      float(settings["speed"]), float(settings["pitch"]), float(settings["energy"]),
                                      None, None, logger)
            # Write to temp then delete
            temp_out = Path("output") / "vibevoice_dryrun.wav"
            write_wav(temp_out, audio[: int(settings["sample_rate"])], int(settings["sample_rate"]), logger)
            try:
                temp_out.unlink()
            except Exception:
                pass
            # Persist settings after dry-run success (unless --no-save-settings)
            if args.remember_text:
                settings["last_text"] = test_text
            if args.forget_text:
                settings.pop("last_text", None)
                settings.pop("last_script", None)
            save_settings(paths["settings_path"], settings, args.no_save_settings, logger)
            return EXIT_OK

        # Validate mode and inputs
        mode = args.mode
        if mode not in ("tts", "multitts"):
            logger.error("You must provide a mode: tts | multitts (or use --probe / --dry-run).")
            return EXIT_INVALID_ARGS

        out_path = Path(args.out if args.out else (Path("output") / "vibevoice_out.wav")).resolve()

        if mode == "tts":
            # Get text
            text = None
            if args.text is not None:
                text = args.text
            elif args.text_file is not None:
                tf = Path(args.text_file)
                if not tf.exists():
                    logger.error(f"--text-file not found: {tf}")
                    return EXIT_INVALID_ARGS
                text = tf.read_text(encoding="utf-8")
            else:
                logger.error("tts mode requires --text or --text-file.")
                return EXIT_INVALID_ARGS
            # Reference voice
            ref_path = Path(args.ref).resolve() if args.ref else None
            audio = synthesize_single(model, text, int(settings["sample_rate"]),
                                      float(settings["speed"]), float(settings["pitch"]), float(settings["energy"]),
                                      ref_path, None, logger)
            write_wav(out_path, audio, int(settings["sample_rate"]), logger)

            # Post-processing (optional)
            ffmpeg = Path(settings["ffmpeg"]) if settings.get("ffmpeg") else paths["ffmpeg"]
            denoise_model = Path(args.denoise_model).resolve() if args.denoise_model else None
            post_process_ffmpeg(ffmpeg, out_path, int(settings["sample_rate"]),
                                bool(args.normalize), bool(args.loudnorm), bool(args.denoise), denoise_model, logger)

            # Persist settings after success
            if args.ref:
                vlist = settings.get("voice_ref_paths", [])
                if isinstance(vlist, list) and args.ref not in vlist:
                    vlist.append(args.ref)
                    settings["voice_ref_paths"] = vlist[:16]  # keep recent up to 16
            settings["device"] = device  # persist effective
            if args.remember_text:
                settings["last_text"] = text
            if args.forget_text:
                settings.pop("last_text", None)
                settings.pop("last_script", None)
            save_settings(paths["settings_path"], settings, args.no_save_settings, logger)
            logger.info(f"Done. Output: {out_path}")
            return EXIT_OK

        elif mode == "multitts":
            if not args.script:
                logger.error("multitts mode requires --script path.json")
                return EXIT_INVALID_ARGS
            s_path = Path(args.script)
            if not s_path.exists():
                logger.error(f"Script JSON not found: {s_path}")
                return EXIT_INVALID_ARGS
            try:
                script_obj = json.loads(s_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to parse script JSON: {e}")
                return EXIT_INVALID_ARGS

            audio = synthesize_multi(model, script_obj, int(settings["sample_rate"]),
                                     float(settings["speed"]), float(settings["pitch"]), float(settings["energy"]),
                                     logger)
            write_wav(out_path, audio, int(settings["sample_rate"]), logger)

            # Post-processing
            ffmpeg = Path(settings["ffmpeg"]) if settings.get("ffmpeg") else paths["ffmpeg"]
            denoise_model = Path(args.denoise_model).resolve() if args.denoise_model else None
            post_process_ffmpeg(ffmpeg, out_path, int(settings["sample_rate"]),
                                bool(args.normalize), bool(args.loudnorm), bool(args.denoise), denoise_model, logger)

            # Persist settings after success
            settings["device"] = device
            if args.remember_text:
                settings["last_script"] = str(s_path)
            if args.forget_text:
                settings.pop("last_text", None)
                settings.pop("last_script", None)
            save_settings(paths["settings_path"], settings, args.no_save_settings, logger)
            logger.info(f"Done. Output: {out_path}")
            return EXIT_OK

        else:
            logger.error("Unknown mode.")
            return EXIT_INVALID_ARGS

    except SystemExit as e:
        # argparse or explicit sys.exit codes propagate unchanged
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return EXIT_OOM
        raise
    except Exception as e:
        # Crash report
        try:
            paths = locals().get("paths")  # may be missing if early failure
            script_dir = Path(__file__).resolve().parent
            logs_dir = (paths["logs_dir"] if paths and "logs_dir" in paths else (script_dir / "logs"))
            crash_dir = (paths["crash_dir"] if paths and "crash_dir" in paths else (logs_dir / "crash"))
            crash_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            report = {
                "args": vars(args) if 'args' in locals() else None,
                "paths": {k: (str(v) if isinstance(v, Path) else v) for k, v in (paths.items() if paths else [])},
                "device_snapshot": vram_snapshot(logging.getLogger(APP_NAME)) if torch else {},
                "exception": {
                    "type": type(e).__name__,
                    "message": str(e),
                }
            }
            if 'args' in locals() and getattr(args, "trace", False):
                import traceback
                report["exception"]["traceback"] = traceback.format_exc()
            out = crash_dir / f"{APP_NAME}_{ts}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logging.getLogger(APP_NAME).error(f"Crash report written to {out}")
        except Exception as e2:
            print("Failed to write crash report:", e2, file=sys.stderr)
        return EXIT_PROBE_FAIL


if __name__ == "__main__":
    sys.exit(main())



# === FrameVision UI pane: VibeVoicePane ======================================
try:
    from PySide6.QtCore import Qt, Signal, QProcess
    from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                                   QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox,
                                   QSpinBox, QFileDialog, QMessageBox)
except Exception:
    QWidget = object  # type: ignore

class VibeVoicePane(QWidget):
    """Minimal GUI wrapper to run this script (vibevoice.py) for TTS or multitts."""
    fileReady = Signal(object)  # Path of produced WAV

    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["tts","multitts"])
        top.addWidget(QLabel("Mode:")); top.addWidget(self.cmb_mode); top.addStretch(1)
        v.addLayout(top)

        form = QFormLayout()
        self.txt_text = QTextEdit(); self.txt_text.setPlaceholderText("Type something to speak...")
        self.ed_script = QLineEdit(); btn_script = QPushButton("Browse")
        def _pick_script():
            fn, _ = QFileDialog.getOpenFileName(self, "Pick JSON script", "", "JSON (*.json);;All files (*.*)")
            if fn: self.ed_script.setText(fn)
        btn_script.clicked.connect(_pick_script)

        self.ed_ref = QLineEdit(); btn_ref = QPushButton("Browse")
        def _pick_ref():
            fn, _ = QFileDialog.getOpenFileName(self, "Pick voice reference", "", "Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.opus);;All files (*.*)")
            if fn: self.ed_ref.setText(fn)
        btn_ref.clicked.connect(_pick_ref)

        self.spn_sr = QSpinBox(); self.spn_sr.setRange(8000, 48000); self.spn_sr.setSingleStep(1000); self.spn_sr.setValue(24000)
        self.ed_out = QLineEdit(); btn_out = QPushButton("Browse")
        def _pick_out():
            fn, _ = QFileDialog.getSaveFileName(self, "Output WAV", "", "Audio (*.wav);;All files (*.*)")
            if fn: self.ed_out.setText(fn)
        btn_out.clicked.connect(_pick_out)

        form.addRow("Text:", self.txt_text)
        row_script = QHBoxLayout(); row_script.addWidget(self.ed_script); row_script.addWidget(btn_script); w_script = QWidget(); w_script.setLayout(row_script)
        form.addRow("Script JSON:", w_script)
        row_ref = QHBoxLayout(); row_ref.addWidget(self.ed_ref); row_ref.addWidget(btn_ref); w_ref = QWidget(); w_ref.setLayout(row_ref)
        form.addRow("Reference voice:", w_ref)
        form.addRow("Sample rate:", self.spn_sr)
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
            is_tts = (self.cmb_mode.currentText() == "tts")
            self.txt_text.setEnabled(is_tts)
            self.ed_script.setEnabled(not is_tts)
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
        args = [self._script_path(), mode, "--sample-rate", str(int(self.spn_sr.value()))]
        if mode == "tts":
            text = (self.txt_text.toPlainText() or "").strip()
            if not text:
                QMessageBox.information(self, "Text needed", "Please enter text.")
                return
            args += ["--text", text]
            ref = (self.ed_ref.text() or "").strip()
            if ref:
                args += ["--ref", ref]
        else:
            script = (self.ed_script.text() or "").strip()
            if not script:
                QMessageBox.information(self, "Script JSON needed", "Please pick a script file.")
                return
            args += ["--script", script]
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