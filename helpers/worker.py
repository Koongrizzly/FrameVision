

def _infer_image_format_from_input(job_args):
    try:
        p = str(job_args.get("input_path") or job_args.get("source") or "")
        suf = Path(p).suffix.lower().lstrip(".")
        if suf == "jpeg":
            suf = "jpg"
        # Allow common still formats
        if suf in ("jpg","png","webp","bmp","tif","tiff"):
            return "tiff" if suf == "tif" else suf
    except Exception:
        pass
    return None
# FrameVision worker - NCNN wiring
import json, time, subprocess, os, re, shutil, sys
from pathlib import Path
try:
    from PIL import Image
except Exception:
    Image = None

# --- No-overwrite helper: pick a new name if target exists ---
def _unique_path(p: Path) -> Path:
    try:
        p = Path(p)
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix
        i = 1
        while True:
            cand = p.with_name(f"{stem}_{i:03d}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    except Exception:
        return Path(p)


# --- Salvage helper: treat non-zero exit as success if output exists ---
def _salvage_nonzero_output(code, outfile, job, label="Command"):
    """If a tool exits non-zero but the expected output file exists and looks non-empty,
    treat it as success (queue UX) and attach a warning + original exit code."""
    try:
        code_i = int(code)
    except Exception:
        code_i = 1
    try:
        if code_i in (0, 130):
            return code_i
        if not outfile:
            return code_i
        op = Path(str(outfile))
        if not op.exists():
            return code_i
        try:
            sz = int(op.stat().st_size or 0)
        except Exception:
            sz = 0
        if sz <= 4096:
            return code_i

        # Mark as salvaged success
        try:
            job["exit_code"] = int(code_i)
        except Exception:
            pass
        try:
            prev_err = job.get("error")
            msg = f"{label} exited with code {code_i} but output exists; marking job as done."
            if prev_err:
                msg = f"{msg}  (original error: {prev_err})"
            job["warning"] = msg
            try:
                job.pop("error", None)
            except Exception:
                job["error"] = ""
        except Exception:
            pass
        return 0
    except Exception:
        return code_i



# --- Quiet mode: suppress CLIP 77-token complaints & other noisy logs ---
try:
    import os as _w_os, warnings as _w_warnings
    _w_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _w_os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    _w_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from transformers.utils import logging as _w_hf_logging
        _w_hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from diffusers.utils import logging as _w_df_logging
        _w_df_logging.set_verbosity_error()
        try:
            _w_df_logging.disable_progress_bar()
        except Exception:
            pass
    except Exception:
        pass
    try:
        import logging as _w_pylogging
        _w_pylogging.getLogger("transformers").setLevel(_w_pylogging.ERROR)
        _w_pylogging.getLogger("transformers.tokenization_utils_base").setLevel(_w_pylogging.ERROR)
    except Exception:
        pass
    try:
        _w_warnings.filterwarnings(
            "ignore",
            message=r".*CLIP can only handle sequences up to 77 tokens.*",
            category=UserWarning,
        )
        _w_warnings.filterwarnings(
            "ignore",
            message=r"The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens:.*",
            category=UserWarning,
        )
    except Exception:
        pass
except Exception:
    pass

import warnings

# 1) Suppress the original scary warning
warnings.filterwarnings(
    "ignore",
    message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead",
    category=UserWarning,
)

# 2) Print your own friendlier note once at startup
print("[Worker info] loaded all Queue settings, hiding useless/harmless warnings. Do NOT close this window.")


# -----------------------------------------------------------------------

ROOT = Path(".").resolve()
BASE = ROOT

def _safe_unlink(p):
    try:
        Path(p).unlink(missing_ok=True)
    except Exception:
        try:
            if Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass

def _safe_rmtree(p):
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


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


LEGACY_BASES = [ROOT / "FrameVision", ROOT / "framevision", ROOT / "Framevis*"]
def _migrate_legacy_tree():
    base = BASE
    for p in ["output/video","output/video/trims","output/trims","output/screenshots","output/descriptions","output/_temp",
              "jobs/pending","jobs/running","jobs/done","jobs/failed","jobs/stale","logs"]:
        (base / p).mkdir(parents=True, exist_ok=True)
    for legacy in LEGACY_BASES:
        if not legacy.exists() or legacy == base:
            continue
        for rel in ["output/video","output/video/trims","output/trims","output/screenshots","output/descriptions","output/_temp",
                    "jobs/pending","jobs/running","jobs/done","jobs/failed","jobs/stale","logs"]:
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

JOBS = { "pending": BASE/"jobs"/"pending", "running": BASE/"jobs"/"running", "done": BASE/"jobs"/"done", "failed": BASE/"jobs"/"failed", "stale": BASE/"jobs"/"stale" }
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT = LOGS_DIR / "worker_heartbeat.txt"
PROGRESS_FILE = None
RUNNING_JSON_FILE = None
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


# --- LLM/queue GPU safety gate ----------------------------------------------
LLM_LOCK_PATH = BASE / "temp" / "runtime" / "gpu_locks" / "llm.lock"

def _pid_is_running(pid: int) -> bool:
    """Cheap Windows-first PID check. Used only at the worker start gate."""
    try:
        pid = int(pid or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return int(code.value) == 259  # STILL_ACTIVE
                return True
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _read_llm_lock() -> dict:
    try:
        if not LLM_LOCK_PATH.exists():
            return {}
        # Tiny JSON file; cap the read so a corrupt file can never become expensive.
        raw = LLM_LOCK_PATH.read_text(encoding="utf-8", errors="replace")[:4096]
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"kind": "llm", "pid": 0, "error": "unreadable lock"}

def _active_llm_lock_info() -> dict:
    """Return active LLM lock info, or {}. Deletes stale lock files."""
    if not LLM_LOCK_PATH.exists():
        return {}
    info = _read_llm_lock()
    pid = 0
    try:
        pid = int(info.get("pid") or 0)
    except Exception:
        pid = 0
    if pid > 0 and _pid_is_running(pid):
        return info
    # Stale lock: remove once, then allow jobs to start.
    try:
        LLM_LOCK_PATH.unlink()
        print("[worker] removed stale LLM GPU lock")
    except Exception:
        pass
    return {}


# --- Stale running-job cleanup ------------------------------------------------
def _is_queue_sidecar(path: Path) -> bool:
    try:
        n = path.name.lower()
        return (
            n.endswith(".progress.json") or
            n.endswith(".json.progress") or
            n.endswith(".meta.json") or
            n.endswith(".cancel") or
            n.startswith("_")
        )
    except Exception:
        return False


def _unique_job_path(dst_dir: Path, name: str) -> Path:
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    dst = dst_dir / name
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix or ".json"
    for i in range(1, 1000):
        cand = dst_dir / f"{stem}_stale_{int(time.time())}_{i}{suffix}"
        if not cand.exists():
            return cand
    return dst_dir / f"{stem}_stale_{uuid.uuid4().hex[:8]}{suffix}"


def _read_job_json_limited(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[:1024 * 1024]
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _job_has_live_owner(data: dict) -> bool:
    """Return True if a running job still appears to belong to a live process."""
    for key in ("worker_pid", "running_worker_pid", "framevision_worker_pid"):
        try:
            pid = int((data or {}).get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and _pid_is_running(pid):
            return True
    for key in ("active_pid", "pid", "process_pid", "child_pid"):
        try:
            pid = int((data or {}).get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and _pid_is_running(pid):
            return True
    return False


def _running_job_looks_finished_or_dead(path: Path, data: dict) -> tuple[bool, str]:
    """Decide whether a jobs/running JSON is stale and safe to move away."""
    try:
        if _is_queue_sidecar(path):
            return False, "sidecar"
        if RUNNING_JSON_FILE and os.path.normcase(os.path.abspath(str(path))) == os.path.normcase(os.path.abspath(str(RUNNING_JSON_FILE))):
            return False, "current-worker-job"
    except Exception:
        pass

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    if size <= 0:
        return True, "empty-running-file"

    if not isinstance(data, dict) or not data:
        # Broken JSON in running can never be a trustworthy active job.
        return True, "invalid-running-json"

    if _job_has_live_owner(data):
        return False, "live-owner"

    status = str(data.get("status") or data.get("stage") or data.get("state") or "").strip().lower()
    if status in {"done", "finished", "complete", "completed", "failed", "error", "cancelled", "canceled", "cancel requested", "cancelled by user", "cancelled by user."}:
        return True, f"terminal-status:{status}"

    if data.get("finished_at") or data.get("ended_at") or data.get("completed_at"):
        return True, "has-finished-timestamp"

    try:
        pct = data.get("pct", data.get("progress", data.get("percent")))
        if pct is not None and float(pct) >= 100.0:
            return True, "progress-100"
    except Exception:
        pass

    if bool(data.get("cancel_requested")) or bool(data.get("cancelled")) or bool(data.get("canceled")):
        return True, "cancel-flag-without-live-owner"

    # Jobs from older builds may not have pid fields. Give very fresh files a grace
    # period so a just-started old worker is not moved while it is still writing.
    try:
        age = max(0.0, time.time() - float(path.stat().st_mtime))
    except Exception:
        age = 999999.0
    has_any_pid_key = any(k in data for k in ("worker_pid", "running_worker_pid", "framevision_worker_pid", "active_pid", "pid", "process_pid", "child_pid"))
    if not has_any_pid_key:
        if age < 600.0:
            return False, "fresh-no-pid-grace"
        return True, "old-no-live-owner"

    return True, "dead-owner"


def cleanup_stale_running_jobs(quiet: bool = False) -> int:
    """Move stale files from jobs/running to jobs/stale.

    This keeps jobs/running meaningful for safeguards: files in that folder should
    represent jobs that are actually owned by a live worker/process.
    """
    moved = 0
    running_dir = JOBS.get("running")
    stale_dir = JOBS.get("stale") or (BASE / "jobs" / "stale")
    try:
        running_dir.mkdir(parents=True, exist_ok=True)
        stale_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        items = [p for p in Path(running_dir).glob("*.json") if p.is_file()]
    except Exception:
        items = []
    for path in items:
        try:
            if _is_queue_sidecar(path):
                continue
            data = _read_job_json_limited(path)
            stale, reason = _running_job_looks_finished_or_dead(path, data)
            if not stale:
                continue
            try:
                if isinstance(data, dict):
                    data["stale_recovered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    data["stale_reason"] = reason
                    data["status"] = "stale"
                    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
            dst = _unique_job_path(Path(stale_dir), path.name)
            try:
                shutil.move(str(path), str(dst))
            except Exception:
                try:
                    dst.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                    path.unlink()
                except Exception:
                    continue
            # Clean known sidecars/cancel markers for the old running file.
            for side in (path.with_suffix(path.suffix + ".cancel"), path.with_suffix(".progress.json"), path.with_suffix(path.suffix + ".progress.json")):
                try:
                    if side.exists():
                        side.unlink()
                except Exception:
                    pass
            moved += 1
            if not quiet:
                print(f"[worker] moved stale running job to jobs/stale: {path.name} ({reason})")
        except Exception as exc:
            if not quiet:
                try:
                    print(f"[worker] stale cleanup skipped {path.name}: {exc}")
                except Exception:
                    pass
    return moved

def _llm_safety_override_enabled(cfg: dict) -> bool:
    try:
        return bool((cfg or {}).get("queue_override_llm_safety_check", False))
    except Exception:
        return False

def _job_waiting_for_llm_patch(job_path: Path, info: dict) -> None:
    try:
        data = json.loads(job_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        msg = "Waiting for LLM to unload"
        model = str((info or {}).get("model") or "").strip()
        if model:
            msg += f" ({model})"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        # Avoid noisy writes: update only when message changes or every ~30 seconds.
        last_msg = str(data.get("waiting_reason") or "")
        last_t = str(data.get("waiting_since") or "")
        should_write = (last_msg != msg) or not last_t
        if not should_write:
            try:
                last_epoch = time.mktime(time.strptime(last_t, "%Y-%m-%d %H:%M:%S"))
                should_write = (time.time() - last_epoch) > 30.0
            except Exception:
                should_write = False
        if not should_write:
            return
        data["waiting_reason"] = msg
        data["waiting_for"] = "llm_unload"
        data["waiting_since"] = last_t or now
        data["stage"] = msg
        job_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

def _clear_job_waiting_for_llm_patch(job_path: Path) -> None:
    try:
        data = json.loads(job_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        if data.get("waiting_for") != "llm_unload" and not data.get("waiting_reason"):
            return
        for k in ("waiting_reason", "waiting_for", "waiting_since"):
            data.pop(k, None)
        if str(data.get("stage") or "").lower().startswith("waiting for llm"):
            data.pop("stage", None)
        job_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

def _can_start_job_under_llm_safety(job_path: Path, cfg: dict) -> bool:
    if _llm_safety_override_enabled(cfg):
        try:
            _clear_job_waiting_for_llm_patch(job_path)
        except Exception:
            pass
        return True
    info = _active_llm_lock_info()
    if not info:
        try:
            _clear_job_waiting_for_llm_patch(job_path)
        except Exception:
            pass
        return True
    _job_waiting_for_llm_patch(job_path, info)
    model = str(info.get("model") or "local LLM").strip() or "local LLM"
    print(f"[worker] waiting: LLM is loaded ({model}); pending job stays queued")
    return False


# --- Cancel / stop support (running jobs) -----------------------------------
import threading as _threading
import queue as _queue
import signal as _signal

_EOF = object()

def _cancel_marker_for(rj: Path) -> Path:
    try:
        return rj.with_suffix(rj.suffix + ".cancel")
    except Exception:
        return Path(str(rj) + ".cancel")

def _cancel_requested() -> bool:
    global RUNNING_JSON_FILE
    try:
        if not RUNNING_JSON_FILE:
            return False
        rj = Path(RUNNING_JSON_FILE)
        try:
            if _cancel_marker_for(rj).exists():
                return True
        except Exception:
            pass
        try:
            j = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
            return bool(j.get("cancel_requested"))
        except Exception:
            return False
    except Exception:
        return False

def _patch_running_json(patch: dict):
    global RUNNING_JSON_FILE
    try:
        if not RUNNING_JSON_FILE:
            return
        rj = Path(RUNNING_JSON_FILE)
        if not rj.exists():
            return
        try:
            j = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            j = {}
        changed = False
        for k, v in (patch or {}).items():
            if j.get(k) != v:
                j[k] = v
                changed = True
        if not changed:
            return
        tmp = rj.with_suffix(rj.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(j, indent=2), encoding="utf-8")
            tmp.replace(rj)
        except Exception:
            try:
                rj.write_text(json.dumps(j, indent=2), encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

def _note_cancel(reason: str = "Cancelled by user"):
    try:
        _patch_running_json({"cancel_requested": True, "cancelled": True, "error": reason, "status": reason})
    except Exception:
        pass
    # Best-effort: remove cancel marker so it doesn't linger in jobs/running
    try:
        if RUNNING_JSON_FILE:
            rj = Path(RUNNING_JSON_FILE)
            mk = _cancel_marker_for(rj)
            if mk.exists():
                mk.unlink()
    except Exception:
        pass

def _kill_process_tree(pid: int):
    """Best-effort kill of a subprocess tree."""
    try:
        pid = int(pid)
    except Exception:
        return
    if pid <= 0:
        return
    try:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        # POSIX (if process group exists)
        try:
            os.killpg(pid, _signal.SIGTERM)
            return
        except Exception:
            pass
        try:
            os.kill(pid, _signal.SIGTERM)
        except Exception:
            pass
    except Exception:
        pass

def _start_proc_reader(proc):
    """Read proc stdout in a daemon thread and push lines into a queue."""
    q = _queue.Queue()
    def _reader():
        try:
            if getattr(proc, "stdout", None) is None:
                q.put(_EOF)
                return
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                q.put(line)
        except Exception:
            pass
        finally:
            try:
                q.put(_EOF)
            except Exception:
                pass
    try:
        t = _threading.Thread(target=_reader, daemon=True)
        t.start()
    except Exception:
        t = None
        try:
            q.put(_EOF)
        except Exception:
            pass
    return q, t

def run(cmd):
    """Run a command, logging stdout/stderr, with best-effort cancellation support."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        LOGS = ROOT/"logs"; LOGS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = LOGS/f"run_{stamp}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join([str(x) for x in cmd]) + "\n\n")
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            try:
                _patch_running_json({"active_pid": int(getattr(p, "pid", 0) or 0), "active_cmd": " ".join([str(x) for x in cmd])})
            except Exception:
                pass
            q, _t = _start_proc_reader(p)
            was_cancel = False
            while True:
                if _cancel_requested():
                    was_cancel = True
                    _note_cancel("Cancelled by user")
                    try:
                        _kill_process_tree(getattr(p, "pid", 0) or 0)
                    except Exception:
                        pass
                    break
                try:
                    item = q.get_nowait()
                except _queue.Empty:
                    item = None
                if item is None:
                    if p.poll() is not None:
                        break
                    time.sleep(0.2)
                    continue
                if item is _EOF:
                    break
                line = item
                try:
                    f.write(line)
                except Exception:
                    pass
            try:
                code = p.wait(timeout=1.5) if was_cancel else p.wait()
            except Exception:
                try:
                    code = p.wait()
                except Exception:
                    code = 1
            if was_cancel:
                code = 130
            f.write(f"\nEXIT CODE: {code}\n")
        return int(code)
    except Exception:
        try:
            return int(subprocess.call(cmd, env=env))
        except Exception:
            return 1
def _progress_set(pct: int):
    try:
        global PROGRESS_FILE, RUNNING_JSON_FILE
        # Write sidecar progress file (optional consumer)
        if PROGRESS_FILE:
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

        # Patch running job JSON for UI (pct + elapsed + eta)
        if RUNNING_JSON_FILE and int(pct) >= 0:
            try:
                rj = Path(RUNNING_JSON_FILE)
                j = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
                ipct = int(max(0, min(100, pct)))
                changed = (j.get("pct") != ipct)
                j["pct"] = ipct

                # Compute timing fields from started_at
                start = j.get("started_at")
                if start:
                    # Parse "YYYY-mm-dd HH:MM:SS"
                    import time as _t
                    try:
                        t0 = _t.mktime(_t.strptime(str(start), "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        # Try ISO format fallback
                        try:
                            from datetime import datetime as _dt
                            t0 = _dt.fromisoformat(str(start)).timestamp()
                        except Exception:
                            t0 = None
                    if t0:
                        elapsed = int(max(0, _t.time() - t0))
                        if j.get("elapsed_sec") != elapsed:
                            changed = True
                        j["elapsed_sec"] = elapsed
                        if 0 < ipct < 100:
                            try:
                                total_est = elapsed / (ipct / 100.0)
                                eta = int(max(1, total_est - elapsed))
                            except Exception:
                                eta = None
                            if eta is not None and j.get("eta_sec") != eta:
                                changed = True
                            j["eta_sec"] = eta
                        else:
                            if "eta_sec" in j:
                                changed = True
                            j["eta_sec"] = 0 if ipct >= 100 else None

                if changed:
                    tmpj = rj.with_suffix(rj.suffix + ".tmp")
                    tmpj.write_text(json.dumps(j, indent=2), encoding="utf-8")
                    try:
                        tmpj.replace(rj)
                    except Exception:
                        rj.write_text(json.dumps(j, indent=2), encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass


# --- Worker log helpers ------------------------------------------------------

def _fmt_dur_short(seconds: int | float | None) -> str:
    """Human-friendly duration (H:MM:SS or M:SS)."""
    try:
        if seconds is None:
            return ""
        s = int(max(0, float(seconds)))
        h, rem = divmod(s, 3600)
        m, s2 = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s2:02d}"
        return f"{m:d}:{s2:02d}"
    except Exception:
        try:
            return str(seconds)
        except Exception:
            return ""

def _now_ts() -> float:
    try:
        import time as _t
        return float(_t.time())
    except Exception:
        return 0.0

def _print_once(key: str, state: dict, msg: str) -> None:
    """Best-effort: print a message only if it changed since last time."""
    try:
        if not isinstance(state, dict):
            print(msg)
            return
        last = state.get(key)
        if last != msg:
            state[key] = msg
            print(msg)
    except Exception:
        try:
            print(msg)
        except Exception:
            pass

def _worker_app_root():
    """Return the FrameVision root even when the worker is launched from another cwd."""
    try:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "helpers":
            return here.parent.parent
        return here.parent
    except Exception:
        return ROOT


def _first_working_exe(candidates, version_arg="-version"):
    for c in candidates:
        try:
            if c is None:
                continue
            c_str = str(c).strip().strip('"')
            if not c_str:
                continue
            # Absolute/relative paths should exist. Bare names may be found by PATH.
            if ("/" in c_str or "\\" in c_str) and not Path(c_str).exists():
                continue
            subprocess.check_output([c_str, version_arg], stderr=subprocess.STDOUT, timeout=8)
            return c_str
        except Exception:
            continue
    return ""


def ffmpeg_path():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    app_root = _worker_app_root()
    candidates = [
        app_root / "presets" / "bin" / exe,
        app_root / "bin" / exe,
        ROOT / "presets" / "bin" / exe,
        ROOT / "bin" / exe,
        shutil.which(exe),
        "ffmpeg",
    ]
    found = _first_working_exe(candidates)
    return found or "ffmpeg"

FFMPEG = ffmpeg_path()

def _ffprobe_bin():
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    app_root = _worker_app_root()
    cands = [
        app_root / "presets" / "bin" / exe,
        app_root / "bin" / exe,
        ROOT / "presets" / "bin" / exe,
        ROOT / "bin" / exe,
        str(FFMPEG).replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe"),
        shutil.which(exe),
        "ffprobe",
    ]
    found = _first_working_exe(cands)
    return found or "ffprobe"

def _probe_src_fps(p: Path) -> str:
    """Best-effort source FPS for video->frames->encode pipelines.

    Prefer avg_frame_rate; fall back to nb_frames/duration; last resort r_frame_rate; default 30.
    Returns an ffmpeg-friendly value (ratio like 24000/1001 or decimal string).
    """
    def _parse_ratio(s: str):
        try:
            s = (s or "").strip()
            if not s or s in ("0/0", "0", "N/A"):
                return None
            if "/" in s:
                a, b = s.split("/", 1)
                a_i = int(float(a))
                b_i = int(float(b))
                if b_i == 0:
                    return None
                return (a_i, b_i)
            f = float(s)
            if f <= 0:
                return None
            den = 1_000_000
            num = int(round(f * den))
            if num <= 0:
                return None
            return (num, den)
        except Exception:
            return None

    def _ratio_to_str(r):
        try:
            n, d = r
            return str(n) if d == 1 else f"{n}/{d}"
        except Exception:
            return ""

    def _fps_from_ratio(r):
        try:
            n, d = r
            return float(n) / float(d) if d else 0.0
        except Exception:
            return 0.0

    try:
        FP = _ffprobe_bin()
        out = subprocess.check_output(
            [FP, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
             "-show_entries", "format=duration",
             "-of", "json",
             str(p)],
            stderr=subprocess.STDOUT
        ).decode("utf-8", "ignore").strip()

        j = {}
        try:
            j = json.loads(out or "{}")
        except Exception:
            j = {}
        streams = j.get("streams") or []
        s0 = streams[0] if streams else {}
        fmt = j.get("format") or {}

        avg = _parse_ratio(str(s0.get("avg_frame_rate") or ""))
        rfr = _parse_ratio(str(s0.get("r_frame_rate") or ""))
        nb_frames = s0.get("nb_frames")
        try:
            nb = int(nb_frames) if nb_frames not in (None, "", "N/A") else None
        except Exception:
            nb = None
        dur_s = s0.get("duration")
        if dur_s in (None, "", "N/A"):
            dur_s = fmt.get("duration")
        try:
            dur = float(dur_s) if dur_s not in (None, "", "N/A") else None
        except Exception:
            dur = None

        candidates = []
        if avg:
            candidates.append((_ratio_to_str(avg), _fps_from_ratio(avg)))
        if nb and dur and dur > 0:
            f = float(nb) / float(dur)
            candidates.append((f"{f:.6f}".rstrip("0").rstrip("."), f))
        if rfr:
            candidates.append((_ratio_to_str(rfr), _fps_from_ratio(rfr)))

        for val_str, fps in candidates:
            if fps and 1.0 <= fps <= 240.0:
                return val_str or "30"
    except Exception:
        pass
    return "30"



def _bytes_human(n: int | float | None) -> str:
    try:
        if n is None:
            return ""
        n = float(n)
        if n < 1024:
            return f"{int(n)} B"
        for unit in ["KB","MB","GB","TB"]:
            n /= 1024.0
            if n < 1024.0:
                return f"{n:.2f} {unit}"
        return f"{n:.2f} PB"
    except Exception:
        try:
            return str(n)
        except Exception:
            return ""

def _probe_video_info(p: Path) -> dict:
    """Best-effort ffprobe info for video files.
    Returns dict with: width,height,duration,fps_str,fps,nb_frames,vcodec,acodec,bit_rate,a_channels,a_rate
    """
    info = {}
    try:
        FP = _ffprobe_bin()
        out = subprocess.check_output(
            [FP, "-v", "error",
             "-show_streams", "-show_format",
             "-of", "json",
             str(p)],
            stderr=subprocess.STDOUT
        ).decode("utf-8", "ignore").strip()
        j = json.loads(out or "{}")
        streams = j.get("streams") or []
        fmt = j.get("format") or {}

        v0 = None
        a0 = None
        for s in streams:
            if not isinstance(s, dict):
                continue
            if s.get("codec_type") == "video" and v0 is None:
                v0 = s
            elif s.get("codec_type") == "audio" and a0 is None:
                a0 = s

        if v0:
            try: info["width"] = int(v0.get("width") or 0)
            except Exception: pass
            try: info["height"] = int(v0.get("height") or 0)
            except Exception: pass
            try: info["vcodec"] = str(v0.get("codec_name") or "")
            except Exception: pass
            try:
                afr = str(v0.get("avg_frame_rate") or "") or ""
                info["fps_str"] = afr
                # Convert to float if possible
                if "/" in afr:
                    a,b = afr.split("/",1)
                    a = float(a); b = float(b)
                    if b:
                        info["fps"] = a/b
                else:
                    f = float(afr) if afr else 0.0
                    if f:
                        info["fps"] = f
            except Exception:
                pass
            try:
                nb = v0.get("nb_frames")
                if nb not in (None, "", "N/A"):
                    info["nb_frames"] = int(float(nb))
            except Exception:
                pass
            try:
                br = v0.get("bit_rate")
                if br not in (None, "", "N/A"):
                    info["v_bit_rate"] = int(float(br))
            except Exception:
                pass

        if a0:
            try: info["acodec"] = str(a0.get("codec_name") or "")
            except Exception: pass
            try:
                ch = a0.get("channels")
                if ch not in (None,"","N/A"):
                    info["a_channels"] = int(float(ch))
            except Exception:
                pass
            try:
                sr = a0.get("sample_rate")
                if sr not in (None,"","N/A"):
                    info["a_rate"] = int(float(sr))
            except Exception:
                pass
            try:
                br = a0.get("bit_rate")
                if br not in (None,"","N/A"):
                    info["a_bit_rate"] = int(float(br))
            except Exception:
                pass

        try:
            dur = fmt.get("duration")
            if dur not in (None,"","N/A"):
                info["duration"] = float(dur)
        except Exception:
            pass
        try:
            br = fmt.get("bit_rate")
            if br not in (None,"","N/A"):
                info["bit_rate"] = int(float(br))
        except Exception:
            pass
    except Exception:
        return info

    # Fill nb_frames estimate if missing
    try:
        if not info.get("nb_frames"):
            fps = float(info.get("fps") or 0.0)
            dur = float(info.get("duration") or 0.0)
            if fps > 0 and dur > 0:
                info["nb_frames"] = int(round(fps * dur))
    except Exception:
        pass
    return info



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
    inp = Path(job["input"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    args = (job.get("args") or {})
    # If UI provided a model-specific scale, honor it (prevents mismatched x2/x4 models).
    factor = int(args.get("model_scale") or args.get("factor") or 4)
    model_name_req = args.get("model", "RealESRGAN-x4plus")
    model_name, exe_path = resolve_upscaler_exe(cfg, mani, model_name_req)

    out = out_dir / f"{inp.stem}_x{factor}.mp4"
    out = _unique_path(out)

    # Prepare potential temp dirs (some may not be used; we'll still try to delete them in finally)
    frames = out_dir / f"{inp.stem}_x{factor}_frames"
    up = out_dir / f"{inp.stem}_x{factor}_up"
    work = out_dir / f"{inp.stem}_x{factor}_work"  # also clean UI-created temp dirs if present

    # --- Helpful upfront info ---
    src = {}
    try:
        if is_video_path(inp):
            src = _probe_video_info(inp)
    except Exception:
        src = {}

    try:
        in_size = int(inp.stat().st_size or 0)
    except Exception:
        in_size = 0

    iw = int(src.get("width") or 0) if isinstance(src, dict) else 0
    ih = int(src.get("height") or 0) if isinstance(src, dict) else 0
    ow = iw * factor if iw else 0
    oh = ih * factor if ih else 0

    try:
        fps_str = _probe_src_fps(inp)
    except Exception:
        fps_str = "30"

    try:
        dur = float(src.get("duration") or 0.0)
    except Exception:
        dur = 0.0

    try:
        nb_est = int(src.get("nb_frames") or 0)
    except Exception:
        nb_est = 0

    try:
        vcodec = (src.get("vcodec") or "").strip()
        acodec = (src.get("acodec") or "").strip()
    except Exception:
        vcodec = ""
        acodec = ""

    try:
        print(
            "[worker][upscale_video] input:",
            str(inp),
            "| size",
            _bytes_human(in_size),
            "| src",
            (f"{iw}x{ih}" if iw and ih else "unknown"),
            "| fps",
            fps_str,
            "| dur",
            (f"{dur:.2f}s" if dur else "unknown"),
            "| vcodec",
            (vcodec or "unknown"),
            "| audio",
            (acodec or "none"),
        )
        if ow and oh:
            print(f"[worker][upscale_video] output: {ow}x{oh}  |  factor x{factor}  |  out_file: {out.name}")
    except Exception:
        pass

    # Patch queue JSON with richer metadata (best-effort)
    try:
        _patch_running_json({
            "stage": "Preparing video upscale",
            "input_path": str(inp),
            "out_path": str(out),
            "factor": int(factor),
            "model": str(model_name),
            "src_w": iw,
            "src_h": ih,
            "out_w": ow,
            "out_h": oh,
            "src_fps": str(fps_str),
            "duration_sec": float(dur) if dur else None,
            "frames_est": int(nb_est) if nb_est else None,
            "src_size_bytes": int(in_size) if in_size else None,
            "vcodec": vcodec or None,
            "acodec": acodec or None,
        })
    except Exception:
        pass

    _hb_thr = None
    _hb_stop = {"run": False}

    def _stop_hb():
        try:
            _hb_stop["run"] = False
            if _hb_thr is not None:
                _hb_thr.join(timeout=0.2)
        except Exception:
            pass

    def _count_pngs(d: Path) -> int:
        try:
            c = 0
            with os.scandir(str(d)) as it:
                for e in it:
                    try:
                        if e.is_file() and e.name.lower().endswith(".png"):
                            c += 1
                    except Exception:
                        continue
            return int(c)
        except Exception:
            try:
                return len(list(Path(d).glob("*.png")))
            except Exception:
                return 0

    try:
        if exe_path and exe_path.exists() and exe_path.is_file():
            print(f"[worker] Using model '{model_name}' at {exe_path}")

            # 1) Extract frames
            try:
                _patch_running_json({"stage": "Extracting frames"})
            except Exception:
                pass
            print("[worker][upscale_video] Stage 1/3: extracting frames...")

            frames.mkdir(parents=True, exist_ok=True)
            code = run([FFMPEG, "-y", "-i", str(inp), str(frames / "%06d.png")])
            if code != 0:
                return 1

            # 2) Upscale frames
            try:
                _patch_running_json({"stage": "Counting frames"})
            except Exception:
                pass

            try:
                total_frames = int(nb_est) if nb_est else 0
            except Exception:
                total_frames = 0
            if total_frames <= 0:
                total_frames = _count_pngs(frames)

            try:
                print(f"[worker][upscale_video] extracted frames: {total_frames}")
            except Exception:
                pass

            up.mkdir(parents=True, exist_ok=True)
            try:
                _progress_set(10)
            except Exception:
                pass

            import threading as _th, time as _time
            _hb_stop["run"] = True
            _hb_state = {
                "last_pct": -1,
                "last_bucket": -1,
                "last_done": 0,
                "last_ts": _time.time(),
                "t0": _time.time(),
                "last_json_ts": 0.0,
            }

            def _hb_loop():
                while _hb_stop.get("run", False):
                    try:
                        done = _count_pngs(up)
                        now = _time.time()
                        dt = max(0.001, now - float(_hb_state.get("last_ts") or now))
                        dframes = max(0, int(done) - int(_hb_state.get("last_done") or 0))
                        speed = float(dframes) / float(dt) if dt > 0 else 0.0
                        _hb_state["last_done"] = int(done)
                        _hb_state["last_ts"] = now

                        if total_frames > 0:
                            pct = 10.0 + min(85.0, (float(done) / float(total_frames)) * 85.0)
                            ipct = int(pct)
                            if ipct != int(_hb_state.get("last_pct") or -1):
                                _progress_set(ipct)
                                _hb_state["last_pct"] = ipct

                            # Print every 10% bucket so the console stays readable
                            bucket = int(ipct / 10)
                            if bucket != int(_hb_state.get("last_bucket") or -1):
                                _hb_state["last_bucket"] = bucket
                                eta_s = None
                                if speed > 0 and done < total_frames:
                                    eta_s = float(total_frames - done) / float(speed)
                                msg = f"[worker][upscale_video] upscaling: {done}/{total_frames}  |  {speed:.2f} frames/s"
                                if eta_s is not None:
                                    msg += f"  |  eta {_fmt_dur_short(eta_s)}"
                                try:
                                    print(msg)
                                except Exception:
                                    pass

                        # Patch running JSON at most ~2 Hz
                        if (now - float(_hb_state.get("last_json_ts") or 0.0)) >= 0.5:
                            _hb_state["last_json_ts"] = now
                            try:
                                eta_s = None
                                if total_frames > 0 and speed > 0 and done < total_frames:
                                    eta_s = float(total_frames - done) / float(speed)
                                _patch_running_json({
                                    "stage": "Upscaling frames",
                                    "done_frames": int(done),
                                    "total_frames": int(total_frames) if total_frames else None,
                                    "speed_fps": float(speed) if speed else None,
                                    "eta_frames_sec": float(eta_s) if eta_s is not None else None,
                                })
                            except Exception:
                                pass
                    except Exception:
                        pass
                    _time.sleep(1.0)

            _hb_thr = _th.Thread(target=_hb_loop, daemon=True)
            try:
                _hb_thr.start()
            except Exception:
                pass

            try:
                _patch_running_json({"stage": "Upscaling frames", "total_frames": int(total_frames) if total_frames else None})
            except Exception:
                pass
            print("[worker][upscale_video] Stage 2/3: upscaling frames...")

            m = str(model_name).lower()
            if ("realesr" in m) or ("realesrgan" in m):
                cmd = build_realesrgan_cmd(
                    str(exe_path),
                    frames,
                    up,
                    factor,
                    model_name,
                    models_dir=Path((job.get('args') or {}).get('models_dir') or Path(exe_path).parent),
                    is_dir=True
                )
                if run(cmd) != 0:
                    return 1
            else:
                # Per-frame upscale for other engines / fallback
                for png in sorted(frames.glob("*.png")):
                    if _cancel_requested():
                        _note_cancel("Cancelled by user")
                        return 130
                    if "swinir" in m:
                        cmd = build_swinir_cmd(str(exe_path), png, up / png.name, factor)
                    elif "lapsrn" in m:
                        cmd = build_lapsrn_cmd(str(exe_path), png, up / png.name, factor)
                    else:
                        cmd = [
                            FFMPEG, "-y", "-i", str(png),
                            "-vf", f"scale=iw*{factor}:ih*{factor}:flags=lanczos",
                            "-frames:v", "1",
                            str(up / png.name)
                        ]
                    if run(cmd) != 0:
                        return 1

            # 3) Re-encode
            _stop_hb()
            try:
                _patch_running_json({"stage": "Encoding video"})
            except Exception:
                pass

            try:
                _progress_set(90)
            except Exception:
                pass

            src_fps = fps_str or _probe_src_fps(inp)
            print("[worker][upscale_video] Stage 3/3: encoding video...")
            try:
                print(f"[worker][upscale_video] encoder: libx264 preset=veryfast | fps={src_fps} | audio=copy")
            except Exception:
                pass

            enc = [
                FFMPEG, "-y",
                "-framerate", str(src_fps),
                "-i", str(up / "%06d.png"),
                "-i", str(inp),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "copy",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-vsync", "cfr",
                "-r", str(src_fps),
                "-movflags", "+faststart",
                str(out)
            ]
            code = run(enc)
            try:
                code = _salvage_nonzero_output(code, out, job, label="FFmpeg encode")
            except Exception:
                pass
            if int(code) != 0:
                return 1

            try:
                _progress_set(100)
            except Exception:
                pass
            try:
                job["produced"] = str(out)
            except Exception:
                pass
            try:
                _patch_running_json({"stage": "Done", "produced": str(out)})
            except Exception:
                pass
            return 0

        # Fallback: ffmpeg scale (no model)
        print("[worker] No model exe found, using ffmpeg scale fallback")
        try:
            _patch_running_json({"stage": "FFmpeg scale fallback", "model": None})
        except Exception:
            pass

        cmd = [FFMPEG, "-y", "-i", str(inp), "-vf", f"scale=iw*{factor}:ih*{factor}:flags=lanczos", str(out)]
        code = run(cmd)

        try:
            code = _salvage_nonzero_output(code, out, job, label="FFmpeg scale")
        except Exception:
            pass

        try:
            _progress_set(100)
        except Exception:
            pass

        if code == 0:
            try:
                job["produced"] = str(out)
            except Exception:
                pass
            try:
                _patch_running_json({"stage": "Done", "produced": str(out)})
            except Exception:
                pass
        return int(code)

    finally:
        # Always try to clean temp dirs; ignore errors
        _stop_hb()
        for d in (frames, up, work / "in", work / "out", work):
            try:
                if d.exists():
                    _safe_rmtree(d)
            except Exception:
                pass



def _ltx_unique_glue_output_path(generated_path):
    try:
        generated_path = Path(str(generated_path))
        base = generated_path.with_name(generated_path.stem + "_glued" + generated_path.suffix)
        if not base.exists():
            return base
        for index in range(1, 1000):
            candidate = generated_path.with_name(f"{generated_path.stem}_glued_{index:03d}{generated_path.suffix}")
            if not candidate.exists():
                return candidate
        return generated_path.with_name(f"{generated_path.stem}_glued_{int(time.time())}{generated_path.suffix}")
    except Exception:
        return Path(str(generated_path) + "_glued.mp4")


def _ltx_escape_concat_path(path):
    try:
        return str(Path(str(path))).replace("\\", "/").replace("'", "'\\''")
    except Exception:
        return str(path).replace("\\", "/").replace("'", "'\\''")


def _ltx_probe_video_geometry(video_path, ffprobe=None):
    ffprobe = ffprobe or _ffprobe_bin()
    try:
        out = subprocess.check_output(
            [
                str(ffprobe),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                "-of", "json",
                str(video_path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        data = json.loads(out or "{}")
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width") or 1280)
        height = int(stream.get("height") or 704)
        fps = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30").strip()
        if not fps or fps in ("0/0", "0", "N/A"):
            fps = "30"
        return width, height, fps
    except Exception:
        return 1280, 704, "30"


def _ltx_run_glue_subprocess(cmd, cwd=None, log_prefix="[worker]"):
    try:
        print(log_prefix, " ".join(str(x) for x in cmd), flush=True)
    except Exception:
        pass
    try:
        completed = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.stdout:
            try:
                for line in completed.stdout.splitlines():
                    print("[TOOLS]", line, flush=True)
            except Exception:
                pass
        return int(completed.returncode) == 0
    except Exception as exc:
        try:
            print(f"[worker] LTX glue command failed: {type(exc).__name__}: {exc}", flush=True)
        except Exception:
            pass
        return False


def _ltx_resolve_glue_ffmpeg(user_value=None):
    """Resolve bundled ffmpeg for LTX glue; never rely on a missing PATH entry first."""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    app_root = _worker_app_root()
    candidates = [
        user_value,
        app_root / "presets" / "bin" / exe,
        app_root / "bin" / exe,
        ROOT / "presets" / "bin" / exe,
        ROOT / "bin" / exe,
        FFMPEG,
        shutil.which(exe),
        "ffmpeg",
    ]
    found = _first_working_exe(candidates)
    if found:
        return found
    return str(app_root / "presets" / "bin" / exe)


def _ltx_resolve_glue_ffprobe(ffmpeg_value=None):
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    app_root = _worker_app_root()
    paired = ""
    try:
        paired = str(ffmpeg_value or "").replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    except Exception:
        paired = ""
    candidates = [
        paired,
        app_root / "presets" / "bin" / exe,
        app_root / "bin" / exe,
        ROOT / "presets" / "bin" / exe,
        ROOT / "bin" / exe,
        _ffprobe_bin(),
        shutil.which(exe),
        "ffprobe",
    ]
    found = _first_working_exe(candidates)
    if found:
        return found
    return str(app_root / "presets" / "bin" / exe)


def _ltx_cleanup_temp_video_frames(args):
    """Delete only LTX temporary extracted video frames after a successful queued run."""
    cleaned = 0
    paths = []
    try:
        raw = (args or {}).get("ltx_temp_video_frame_paths") or []
        if isinstance(raw, str):
            paths = [raw]
        else:
            paths = list(raw)
    except Exception:
        paths = []

    # Fallback for older queued jobs: detect temp PNGs used directly in the CLI command.
    try:
        cmd = (args or {}).get("cmd") or (args or {}).get("ffmpeg_cmd") or []
        cmd_list = list(cmd) if isinstance(cmd, (list, tuple)) else []
        for item in cmd_list:
            text = str(item or "")
            low = text.replace("\\", "/").lower()
            if "/temp/ltx23_video_frames/" in low and text.lower().endswith(".png"):
                paths.append(text)
    except Exception:
        pass

    root = _worker_app_root()
    temp_dir = (root / "temp" / "ltx23_video_frames").resolve()
    for value in dict.fromkeys(str(x).strip() for x in paths if str(x).strip()):
        try:
            p = Path(value).expanduser()
            if p.exists() and p.is_file() and p.resolve().parent == temp_dir:
                p.unlink()
                cleaned += 1
        except Exception:
            pass
    try:
        temp_dir.rmdir()
    except Exception:
        pass
    if cleaned:
        try:
            print(f"[worker] LTX cleanup removed {cleaned} temporary video frame(s).", flush=True)
        except Exception:
            pass
    return cleaned


def _ltx_cleanup_glue_temp_files(*paths):
    """Best-effort cleanup for tiny glue sidecars; never touches videos."""
    cleaned = 0
    for value in paths:
        if not value:
            continue
        try:
            p = Path(value)
            if p.exists() and p.is_file() and p.suffix.lower() in {".txt", ".tmp"}:
                p.unlink()
                cleaned += 1
        except Exception:
            pass
    return cleaned


def _ltx_glue_videos_after_generation(args, generated_output, cwd=None):
    """Combine Continue video + generated LTX output + End with video when requested."""
    try:
        if not bool((args or {}).get("ltx_glue_input_videos")):
            return str(generated_output or "")
        generated = Path(str(generated_output or (args or {}).get("outfile") or ""))
        if not (generated.exists() and generated.stat().st_size > 4096):
            return str(generated)

        start_video = str((args or {}).get("ltx_start_video_path") or "").strip()
        end_video = str((args or {}).get("ltx_end_video_path") or "").strip()
        videos = []
        for item in (start_video, str(generated), end_video):
            if not item:
                continue
            p = Path(item)
            if p.exists() and p.is_file():
                videos.append(p)
        if len(videos) < 2:
            return str(generated)

        ffmpeg = _ltx_resolve_glue_ffmpeg((args or {}).get("ffmpeg_path"))
        ffprobe = _ltx_resolve_glue_ffprobe(ffmpeg)
        try:
            print(f"[worker] LTX glue ffmpeg: {ffmpeg}", flush=True)
            print(f"[worker] LTX glue ffprobe: {ffprobe}", flush=True)
        except Exception:
            pass
        final_path = _ltx_unique_glue_output_path(generated)
        concat_list = final_path.with_suffix(".concat.txt")
        concat_list.write_text("".join(f"file '{_ltx_escape_concat_path(p)}'\n" for p in videos), encoding="utf-8")

        copy_cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(final_path),
        ]
        if _ltx_run_glue_subprocess(copy_cmd, cwd=cwd, log_prefix="[worker] LTX glue stream-copy:") and final_path.exists() and final_path.stat().st_size > 4096:
            try:
                concat_list.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[worker] LTX glued output: {final_path}", flush=True)
            return str(final_path)

        try:
            final_path.unlink(missing_ok=True)
        except Exception:
            pass

        width, height, fps = _ltx_probe_video_geometry(generated, ffprobe=ffprobe)
        inputs = []
        filters = []
        labels = []
        for index, video in enumerate(videos):
            inputs.extend(["-i", str(video)])
            label = f"v{index}"
            filters.append(
                f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[{label}]"
            )
            labels.append(f"[{label}]")
        filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={len(videos)}:v=1:a=0[outv]"
        reencode_cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final_path),
        ]
        if _ltx_run_glue_subprocess(reencode_cmd, cwd=cwd, log_prefix="[worker] LTX glue re-encode:") and final_path.exists() and final_path.stat().st_size > 4096:
            try:
                concat_list.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[worker] LTX glued output: {final_path}", flush=True)
            return str(final_path)

        try:
            concat_list.unlink(missing_ok=True)
        except Exception:
            pass
        print("[worker] LTX glue failed; keeping original output.", flush=True)
        return str(generated)
    except Exception as exc:
        try:
            print(f"[worker] LTX glue skipped: {type(exc).__name__}: {exc}", flush=True)
        except Exception:
            pass
        return str(generated_output or "")



def tools_ffmpeg(job, cfg, mani):
    """Run an external command job (legacy job_type name: tools_ffmpeg).

    Adds best-effort progress updates by parsing stdout for common patterns:
      - tqdm: "10%|...| 3/30"
      - "step 3/30" or "3/30"
      - "10%" (fallback)

    Supports job["args"] keys:
      - ffmpeg_cmd / cmd: list or string command
      - cwd: working directory
      - outfile: expected output file (marked as job["produced"] on success)
      - log_file: optional path for stdout capture (defaults to ./logs/tools_<id>.log)
    """
    try:
        import shlex
        import pathlib  # ensure available even if not imported at module level
        import subprocess as _sp
        import time as _time
        import re as _re
        import os as _os

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
                job["error"] = "No command provided (expected args.ffmpeg_cmd, args.cmd, or job.cmd)."
            except Exception:
                pass
            return 1

        _h15_python_rewrite_log = ""

        # Safety rewrite for Hunyuan 1.5 envs.
        # This is deliberately broad: any Hunyuan job should launch with the
        # first local Hunyuan Python that can import diffusers, even if an old
        # job/helper wrote the wrong Scripts/python.exe into the command.
        try:
            if isinstance(cmd, (list, tuple)) and len(cmd) >= 1:
                _cmd_txt = ' '.join(str(x) for x in cmd[:12])
                _engine = str(args.get('engine') or job.get('engine') or '').lower().strip()
                _is_h15 = (_engine == 'hunyuan15') or ('hunyuan15_cli.py' in _cmd_txt.replace('\\', '/'))
                if _is_h15:
                    from pathlib import Path as _Path
                    import subprocess as _subprocess
                    # Use the job CWD, not the worker's current directory. The previous
                    # patch ran before cwd was normalized, so it often scanned the wrong
                    # folder and never found the real Hunyuan conda env.
                    _root = _Path(str(args.get('cwd') or job.get('cwd') or ROOT)).resolve()
                    _env_dirs = [
                        _root / 'environments' / '.hunyuan15_official',
                        _root / 'environments' / '.hunyuan15',
                        _root / '.hunyuan15_env',
                    ]
                    _cands = []
                    for _env in _env_dirs:
                        _cands.extend([
                            _env / 'python.exe',
                            _env / 'Scripts' / 'python.exe',
                            _env / 'bin' / 'python',
                        ])
                    _selected = None
                    _checked = []
                    for _py in _cands:
                        try:
                            if not (_py.exists() and _py.is_file()):
                                continue
                            _ok = False
                            try:
                                _pr = _subprocess.run([str(_py), '-c', 'import diffusers'], cwd=str(_root), stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL, timeout=12, check=False)
                                _ok = (int(getattr(_pr, 'returncode', 1)) == 0)
                            except Exception:
                                _ok = False
                            _checked.append(f"{_py} => {'diffusers OK' if _ok else 'no diffusers'}")
                            if _ok:
                                _selected = _py
                                break
                        except Exception:
                            pass
                    if _selected is None:
                        # Fallback: still prefer a root conda python over Scripts if it exists.
                        for _py in _cands:
                            try:
                                if _py.exists() and _py.is_file() and str(_py).replace('\\','/').lower().endswith('/python.exe') and '/scripts/' not in str(_py).replace('\\','/').lower():
                                    _selected = _py
                                    _checked.append(f"{_py} => fallback root python")
                                    break
                            except Exception:
                                pass
                    if _selected is not None and str(_selected) != str(cmd[0]):
                        cmd = list(cmd)
                        _old = str(cmd[0])
                        cmd[0] = str(_selected)
                        try:
                            args['ffmpeg_cmd'] = cmd
                            job.setdefault('args', args)
                            job['hunyuan15_python_rewrite'] = str(_selected)
                            job['hunyuan15_python_rewrite_old'] = _old
                        except Exception:
                            pass
                        _h15_python_rewrite_log = f"[worker] Hunyuan Python rewrite: {_old} -> {_selected}; root={_root}"
                        print(_h15_python_rewrite_log, flush=True)
                    elif _is_h15:
                        _h15_python_rewrite_log = "[worker] Hunyuan Python rewrite: no better interpreter found; root=" + str(_root) + "; checked: " + " | ".join(_checked[:10])
                        print(_h15_python_rewrite_log, flush=True)
        except Exception as _h15_py_rewrite_error:
            try:
                _h15_python_rewrite_log = f"[worker] Hunyuan Python rewrite failed: {_h15_py_rewrite_error}"
                print(_h15_python_rewrite_log, flush=True)
            except Exception:
                pass

        # CWD
        cwd = args.get("cwd") or job.get("cwd") or None
        try:
            if cwd:
                cwd = str(pathlib.Path(str(cwd)).resolve())
        except Exception:
            cwd = None

        # Ensure output directory exists if job provides one
        try:
            out_dir = job.get("out_dir")
            if out_dir:
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # If args['outfile'] is set, create its parent
        outfile = None
        try:
            outfile = args.get("outfile")
            if outfile:
                pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            outfile = None

        # Record the normalized command for visibility
        try:
            job["cmd"] = " ".join(str(x) for x in cmd)
        except Exception:
            pass

        # Nice title for the queue row (if caller provided a label)
        try:
            label = args.get("label") or args.get("title")
            if label:
                job["title"] = str(label)
        except Exception:
            pass

        # Default log file
        try:
            log_dir = ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_dir = None

        log_file = None
        try:
            log_file = args.get("log_file")
        except Exception:
            log_file = None
        if not log_file and log_dir is not None:
            stamp = _time.strftime("%Y%m%d_%H%M%S")
            jid = job.get("id") or "job"
            log_file = str(log_dir / f"tools_{jid}_{stamp}.log")

        # Progress parsing
        tqdm_re = _re.compile(r"(?P<pct>\d{1,3})%\|.*?(?P<cur>\d+)\s*/\s*(?P<tot>\d+)")
        step_re = _re.compile(r"(?i)(?:step[^0-9]*)?(?P<cur>\d+)\s*/\s*(?P<tot>\d+)")
        pct_re = _re.compile(r"(?<!\d)(?P<pct>\d{1,3})\s*%")

        start_ts = _time.time()
        last_pct = -1
        last_touch = start_ts

        def _set_pct(p):
            nonlocal last_pct, last_touch
            try:
                ip = int(max(0, min(99, int(p))))
            except Exception:
                return
            if ip != last_pct:
                try:
                    _progress_set(ip)
                except Exception:
                    pass
                last_pct = ip
                last_touch = _time.time()

        def _idle_tick():
            # If we have no measurable progress, at least show "alive" movement.
            nonlocal last_touch
            now = _time.time()
            if (now - last_touch) < 3.0:
                return
            elapsed = max(0.0, now - start_ts)
            # Slowly ramp to 25% over ~60s, then hover.
            p = min(25.0, (elapsed / 60.0) * 25.0)
            if last_pct < 0:
                p = max(0.0, p)
            _set_pct(p)

        env = _os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        # Optional env overrides from job args (useful for tools that need custom PATH/vars)
        try:
            extra_env = args.get("env")
            if isinstance(extra_env, dict):
                for k, v in extra_env.items():
                    if k is None or v is None:
                        continue
                    env[str(k)] = str(v)
        except Exception:
            pass

        # Optional PATH prepend (string or list/tuple of paths)
        try:
            pp = args.get("prepend_path") or args.get("prepend_PATH")
            if pp:
                if isinstance(pp, (list, tuple)):
                    pp = _os.pathsep.join([str(x) for x in pp if x])
                pp = str(pp)
                oldp = env.get("PATH", "")
                env["PATH"] = pp + (_os.pathsep + oldp if oldp else "")
        except Exception:
            pass

        # Start
        try:
            _progress_set(0)
        except Exception:
            pass


        # Fix for planner queue-lock: python -c cannot contain a 'while' compound statement after semicolons.
        # If cmd looks like: python -c "import ...;...;while not os.path.exists(done_flag): time.sleep(0.5);..."
        # rewrite the script arg into a multiline program so Python can execute it.
        try:
            if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                exe = str(cmd[0]).lower()
                if (exe.endswith("python") or exe.endswith("python.exe")) and str(cmd[1]) == "-c":
                    script = str(cmd[2])
                    if ";while " in script and "os.path.exists(done_flag)" in script:
                        script = _re.sub(
                            r";while\s+not\s+os\.path\.exists\(done_flag\):\s*time\.sleep\(0\.5\);",
                            "\nwhile not os.path.exists(done_flag):\n    time.sleep(0.5)\n",
                            script,
                        )
                        cmd = list(cmd)
                        cmd[2] = script
        except Exception:
            pass

        try:
            lf = open(log_file, "w", encoding="utf-8") if log_file else None
            if lf:
                try:
                    lf.write("CMD: " + " ".join(str(x) for x in cmd) + "\n")
                    if cwd:
                        lf.write("CWD: " + str(cwd) + "\n")
                    try:
                        if _h15_python_rewrite_log:
                            lf.write(_h15_python_rewrite_log + "\n")
                        # Also show queue-adapter diagnostics if present in the job args.
                        _qa = args.get("hunyuan15_python_rewrite_checked") or args.get("hunyuan15_python_rewrite_error") or ""
                        if _qa:
                            lf.write("[queue_adapter] Hunyuan Python checked: " + str(_qa) + "\n")
                    except Exception:
                        pass
                    lf.write("\n")
                    lf.flush()
                except Exception:
                    pass

            p = _sp.Popen(
                [str(x) for x in cmd],
                cwd=cwd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            try:
                _patch_running_json({'active_pid': int(getattr(p,'pid',0) or 0), 'active_cmd': ' '.join([str(x) for x in cmd])})
            except Exception:
                pass

            q, _t = _start_proc_reader(p)
            was_cancel = False
            while True:
                if _cancel_requested():
                    was_cancel = True
                    _note_cancel("Cancelled by user")
                    try:
                        _kill_process_tree(getattr(p, 'pid', 0) or 0)
                    except Exception:
                        pass
                    break
                try:
                    item = q.get_nowait()
                except _queue.Empty:
                    item = None
                if item is None:
                    if p.poll() is not None:
                        break
                    _idle_tick()
                    _time.sleep(0.35)
                    continue
                if item is _EOF:
                    break
                line = item
                if not line:
                    continue

                # tqdm often uses carriage returns; split them into logical "lines"
                parts = str(line).replace("\r", "\n").split("\n")
                for part in parts:
                    if part == "":
                        continue

                    if lf:
                        try:
                            lf.write(part + "\n")
                        except Exception:
                            pass

                    # Console echo (useful when worker is run standalone)
                    try:
                        print("[TOOLS]", part)
                    except Exception:
                        pass

                    # Progress patterns
                    m = None
                    try:
                        m = tqdm_re.search(part)
                    except Exception:
                        m = None
                    if m:
                        try:
                            pct = int(m.group("pct"))
                            _set_pct(pct)
                            continue
                        except Exception:
                            pass

                    try:
                        m = step_re.search(part)
                    except Exception:
                        m = None
                    if m:
                        try:
                            cur = int(m.group("cur"))
                            tot = int(m.group("tot"))
                            if tot > 0:
                                pct = int(100.0 * float(max(0, min(cur, tot))) / float(tot))
                                _set_pct(pct)
                                continue
                        except Exception:
                            pass

                    try:
                        m = pct_re.search(part)
                    except Exception:
                        m = None
                    if m:
                        try:
                            pct = int(m.group("pct"))
                            if 0 <= pct <= 100:
                                _set_pct(pct)
                                continue
                        except Exception:
                            pass

                _idle_tick()

            code = p.wait()
            if was_cancel: code = 130
            try:
                if lf:
                    lf.write(f"\nEXIT CODE: {code}\n")
                    lf.close()
            except Exception:
                pass

        except Exception:
            # Fallback to old behavior without streaming/progress.
            code = run([str(x) for x in cmd])

        # Finalize progress
        try:
            _progress_set(100 if int(code) == 0 else max(0, last_pct))
        except Exception:
            pass

        # Normalize return code
        try:
            code_i = int(code)
        except Exception:
            code_i = 1

        # Salvage: some helper scripts exit non-zero even though the expected output was produced.
        # For queue UX, it's more useful to mark these as DONE (with a warning) than FAILED.
        salvaged = False
        try:
            if code_i not in (0, 130) and outfile:
                op = pathlib.Path(str(outfile))
                if op.exists():
                    try:
                        sz = int(op.stat().st_size or 0)
                    except Exception:
                        sz = 0
                    # Small threshold avoids "empty placeholder file" false-positives.
                    if sz > 4096:
                        salvaged = True
        except Exception:
            salvaged = False

        if salvaged:
            try:
                job["exit_code"] = int(code_i)
            except Exception:
                pass
            try:
                prev_err = job.get("error")
                msg = f"Command exited with code {code_i} but output exists; marking job as done."
                if prev_err:
                    msg = f"{msg}  (original error: {prev_err})"
                job["warning"] = msg
                try:
                    job.pop("error", None)
                except Exception:
                    job["error"] = ""
            except Exception:
                pass
            code_i = 0

        if int(code_i) == 0:
            # LTX 2.3 optional video glue: Continue video + generated result + End with video.
            try:
                if bool(args.get("ltx_glue_input_videos")) and outfile and pathlib.Path(str(outfile)).exists():
                    glued_out = _ltx_glue_videos_after_generation(args, outfile, cwd=cwd)
                    if glued_out and pathlib.Path(str(glued_out)).exists():
                        outfile = str(glued_out)
                        args["outfile"] = str(glued_out)
                        args["scan_dir"] = str(pathlib.Path(str(glued_out)).parent)
                        args["scan_ext"] = pathlib.Path(str(glued_out)).suffix or ".mp4"
                        job["args"] = args
            except Exception as _ltx_glue_error:
                try:
                    print(f"[worker] LTX glue postprocess failed: {_ltx_glue_error}", flush=True)
                except Exception:
                    pass

            # The extracted Continue/End PNGs are only needed during generation.
            # Clean them on successful completion, but keep them after failures for debugging/retry clues.
            try:
                if str(args.get("engine") or "").lower() == "ltx23" or args.get("ltx_temp_video_frame_paths"):
                    _ltx_cleanup_temp_video_frames(args)
            except Exception as _ltx_cleanup_error:
                try:
                    print(f"[worker] LTX cleanup skipped: {_ltx_cleanup_error}", flush=True)
                except Exception:
                    pass

            # Mark produced output if known
            try:
                if outfile and pathlib.Path(str(outfile)).exists():
                    job["produced"] = str(outfile)
                    job["files"] = [str(outfile)]
            except Exception:
                pass

            # Planner lock / generic tools jobs: if caller provided a scan_dir + scan_ext,
            # pick the newest matching file and mark it as produced so Queue's "Play last result" works.
            try:
                scan_dir = args.get("scan_dir") or args.get("out_scan_dir")
                scan_ext = args.get("scan_ext") or args.get("out_scan_ext") or ".mp4"
                if scan_dir:
                    sd = pathlib.Path(str(scan_dir))
                    if sd.exists() and sd.is_dir():
                        ext = str(scan_ext).lower()
                        if not ext.startswith("."):
                            ext = "." + ext
                        cand = list(sd.glob(f"*{ext}"))
                        if cand:
                            newest = max(cand, key=lambda p: p.stat().st_mtime)
                            job["produced"] = str(newest)
                            job["files"] = [str(newest)]
                            # also set outfile for UI/compat
                            try:
                                args["outfile"] = str(newest)
                                job["args"] = args
                            except Exception:
                                pass
            except Exception:
                pass

        else:
            try:
                if not job.get("error"):
                    job["error"] = f"tools_ffmpeg failed (code {code_i})."
            except Exception:
                pass

        return int(code_i)

    except Exception as e:
        try:
            job["error"] = f"tools_ffmpeg exception: {e}"
        except Exception:
            pass
        return 1


def upscale_photo(job, cfg, mani):
    print("[worker] upscale_photo: start", job.get("input"))
    inp = Path(job["input"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    args = job.get("args") or {}
    # If UI provided a model-specific scale, honor it (prevents mismatched x2/x4 models).
    factor = int(args.get("model_scale") or args.get("factor") or 4)
    fmt = (args.get("format") or _infer_image_format_from_input(args) or "png").lower()
    model_name = args.get("model", "RealESRGAN-x4plus")
    model_name, exe_path = resolve_upscaler_exe(cfg, mani, model_name)
    out = out_dir / f"{inp.stem}_x{factor}.{fmt}"
    out = _unique_path(out)
    try:
        _progress_set(5)
    except Exception:
        pass

    # Decode-prep still images to a temp RGB PNG to avoid alpha/codec quirks
    src_in = inp
    tmp_rgb = None
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

    try:
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
            try:
                code = _salvage_nonzero_output(code, out, job, label="Upscaler")
            except Exception:
                pass

            # ## PATCH set produced after model-exe
            try:
                if code == 0:
                    job['produced'] = str(out)
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
        try:
            code = _salvage_nonzero_output(code, out, job, label="Upscaler")
        except Exception:
            pass

        if code == 0:
            try: job['produced'] = str(out)
            except Exception: pass
        return code
    finally:
        # Remove temp RGB probe image if we created one
        if tmp_rgb:
            _safe_unlink(tmp_rgb)


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


def txt2img_generate(job, cfg, mani):
    """Queue worker entry for txt2img (SD15/SDXL/Z-Image via helpers.txt2img).

    Strong-fail for Z-Image: when engine == "zimage", we never fall back to the
    legacy Diffusers path. For all other engines we still allow a legacy
    fallback so existing SD15/SDXL jobs keep working.
    """
    # Try the shared helpers.txt2img implementation first
    try:
        from helpers import txt2img as _txt2img
    except BaseException as _imp_err:
        # Option B: if the helpers package is not available (e.g. standalone worker),
        # fall back to importing a local txt2img.py that lives next to worker.py.
        try:
            import txt2img as _txt2img  # type: ignore
            try:
                print("[worker txt2img] Using local txt2img.py (OK)")
            except Exception:
                pass
        except BaseException as _imp_err2:
            try:
                print("[worker txt2img] helpers.txt2img and local txt2img imports failed; falling back to legacy:",
                      _imp_err, "/", _imp_err2)
            except Exception:
                pass
            return _txt2img_generate_legacy(job, cfg, mani)

    # Pull args produced by queue_adapter / enqueue_txt2img
    try:
        args = job.get("args") or {}
    except Exception:
        args = {}

    # Resolve output directory (queue JSON uses job["out_dir"])
    try:
        from pathlib import Path as _P
        out_dir = _P(job.get("out_dir") or "./output/photo/txt2img")
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = None

    # Build a UI-style job dict for helpers.txt2img
    helper_job = dict(args)
    if out_dir is not None:
        helper_job.setdefault("output", str(out_dir))

    # Normalise engine selector
    try:
        engine = (str(args.get("engine") or job.get("engine") or "diffusers")).strip().lower()
    except Exception:
        engine = "diffusers"
    helper_job["engine"] = engine
    try:
        print(f"[worker txt2img] job id={job.get('id','?')} engine={engine!r}")
    except Exception:
        pass

    res = None

    # Strict Z-Image path (Option A: no SDXL fallback)
    if engine in ("zimage", "zimage_gguf"):
        try:
            _z_state = {"t0": _now_ts(), "last_stage": None, "last_step": None, "last_pct": -1, "last_line": None}
            try:
                _steps = int(args.get("steps") or helper_job.get("steps") or 0)
            except Exception:
                _steps = 0
            try:
                _w = int(args.get("width") or args.get("w") or helper_job.get("width") or helper_job.get("w") or 0)
                _h = int(args.get("height") or args.get("h") or helper_job.get("height") or helper_job.get("h") or 0)
            except Exception:
                _w = _h = 0
            try:
                _seed = args.get("seed") or helper_job.get("seed")
            except Exception:
                _seed = None
            try:
                _batch = int(args.get("batch") or helper_job.get("batch") or 1)
            except Exception:
                _batch = 1
            try:
                _model = args.get("model") or args.get("ai_model") or helper_job.get("model") or "Z-Image-Turbo"
            except Exception:
                _model = "Z-Image-Turbo"
            try:
                print(f"[txt2img][Z-Image] start  |  engine={engine}  |  model={_model}  |  {_w}x{_h}  |  steps={_steps}  |  batch={_batch}  |  seed={_seed}")
            except Exception:
                pass

            def _zprog(p):
                """Progress callback for txt2img engines (pct or step/total dict).

                For Z-Image-Turbo this adds human-readable worker logs (stage + step)
                instead of only raw numbers.
                """
                try:
                    elapsed = None
                    try:
                        elapsed = _now_ts() - float(_z_state.get("t0") or _now_ts())
                    except Exception:
                        elapsed = None

                    # ---- Numeric progress ----
                    if isinstance(p, (int, float)):
                        ip = int(p)
                        _progress_set(ip)
                        # Don't spam: print every 5% or on finish
                        try:
                            lastp = int(_z_state.get("last_pct") or -999)
                        except Exception:
                            lastp = -999
                        if ip >= 100 or (ip - lastp) >= 5:
                            _z_state["last_pct"] = ip
                            msg = f"[txt2img][Z-Image] {ip}%  |  elapsed {_fmt_dur_short(elapsed)}"
                            _print_once("last_line", _z_state, msg)
                        return

                    if not isinstance(p, dict):
                        return

                    stage = (p.get("stage") or p.get("status") or "").strip()
                    status = (p.get("status") or "").strip()

                    pct = None
                    if "pct" in p:
                        try:
                            pct = int(float(p.get("pct") or 0))
                        except Exception:
                            pct = None
                        if pct is not None:
                            _progress_set(pct)

                    step = None
                    total = None
                    if "step" in p and "total" in p:
                        try:
                            step = int(float(p.get("step") or 0))
                            total = int(float(p.get("total") or 0))
                        except Exception:
                            step = None
                            total = None
                        if total and total > 0:
                            if pct is None:
                                try:
                                    pct = int(100.0 * float(step) / float(total))
                                except Exception:
                                    pct = None
                            try:
                                _progress_set(int(100.0 * float(step) / float(total)))
                            except Exception:
                                pass

                    # Infer a reasonable stage if backend didn't send one.
                    if not stage:
                        if step is not None and total is not None:
                            stage = "Denoising"
                        elif pct is not None and pct < 5:
                            stage = "Preparing"
                        elif pct is not None and pct >= 95:
                            stage = "Saving"
                        else:
                            stage = "Working"

                    img_idx = None
                    for k in ("idx", "image_idx", "image_index", "batch_idx"):
                        if k in p:
                            try:
                                img_idx = int(p.get(k))
                            except Exception:
                                img_idx = None
                            break

                    # Decide if we should print a line (avoid spam for 50+ steps).
                    should_print = False
                    try:
                        last_stage = _z_state.get("last_stage")
                        if stage and stage != last_stage:
                            should_print = True
                            _z_state["last_stage"] = stage
                    except Exception:
                        pass
                    try:
                        last_step = _z_state.get("last_step")
                        if step is not None and step != last_step:
                            if total and total > 0:
                                if total <= 20:
                                    should_print = True
                                else:
                                    stride = max(1, int(total / 10))
                                    if step in (1, total) or (step % stride) == 0:
                                        should_print = True
                            else:
                                should_print = True
                            _z_state["last_step"] = step
                    except Exception:
                        pass

                    if not should_print:
                        return

                    parts = [f"[txt2img][Z-Image] {stage}"]
                    if img_idx is not None:
                        parts.append(f"img {img_idx}")
                    if step is not None and total is not None and total > 0:
                        parts.append(f"{step}/{total}")
                    elif pct is not None:
                        parts.append(f"{pct}%")
                    if status and status != stage:
                        parts.append(status)

                    msg = "  |  ".join(parts)
                    if elapsed is not None:
                        msg += f"  |  elapsed {_fmt_dur_short(elapsed)}"
                    _print_once("last_line", _z_state, msg)

                    # Patch running job JSON with richer fields (best-effort)
                    try:
                        if RUNNING_JSON_FILE:
                            rj = Path(RUNNING_JSON_FILE)
                            j = json.loads(rj.read_text(encoding='utf-8')) if rj.exists() else {}
                            changed = False
                            for k in ("step", "total", "status", "stage"):
                                if k in p and j.get(k) != p.get(k):
                                    j[k] = p.get(k)
                                    changed = True
                            # Also store our inferred stage if backend didn't send one.
                            if stage and j.get("stage") != stage:
                                j["stage"] = stage
                                changed = True
                            if changed:
                                j["status_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                                rj.write_text(json.dumps(j, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                except Exception:
                    return
            res = _txt2img.generate_qwen_images(helper_job, progress_cb=_zprog, cancel_event=None)
        except BaseException as e:
            try:
                _mark_error(job, f"Z-Image txt2img failed: {e}")
            except Exception:
                pass
            try:
                print("[worker txt2img] Z-Image backend raised; NOT falling back to legacy SDXL.")
            except Exception:
                pass
            return 2

        if not res or not res.get("files"):
            try:
                _mark_error(job, "Z-Image backend produced no images.")
            except Exception:
                pass
            try:
                print("[worker txt2img] Z-Image backend returned no files; NOT falling back to legacy SDXL.")
            except Exception:
                pass
            return 2
    else:
        # Non-Z-Image engines: allow helper + legacy Diffusers fallback
        try:
            # Special: Qwen-Image-2512 (GGUF via sd-cli) — print richer debug info like the Z-Image path does.
            if engine in ("qwen2512", "qwen_2512", "qwenimage2512", "qwen-image-2512"):
                _q_state = {"t0": _now_ts(), "last_stage": None, "last_step": None, "last_pct": -1, "last_line": None}

                # Pull common settings (best-effort; keys vary a bit across UI versions).
                try:
                    _steps = int(args.get("steps") or helper_job.get("steps") or 0)
                except Exception:
                    _steps = 0
                try:
                    _w = int(args.get("width") or args.get("w") or helper_job.get("width") or helper_job.get("w") or 0)
                    _h = int(args.get("height") or args.get("h") or helper_job.get("height") or helper_job.get("h") or 0)
                except Exception:
                    _w = _h = 0
                try:
                    _cfg = float(args.get("cfg") or args.get("cfg_scale") or helper_job.get("cfg") or helper_job.get("cfg_scale") or 0.0)
                except Exception:
                    _cfg = 0.0
                try:
                    _sampler = (args.get("sampler") or helper_job.get("sampler") or "").strip()
                except Exception:
                    _sampler = ""
                try:
                    _flow = int(args.get("flow_shift", helper_job.get("flow_shift", 0)) or 0)
                except Exception:
                    _flow = 0
                try:
                    _seed = args.get("seed") or helper_job.get("seed")
                except Exception:
                    _seed = None
                try:
                    _offload_raw = args.get("offload_cpu", None)
                    if _offload_raw is None:
                        _offload_raw = args.get("offload_to_cpu", None)
                    if _offload_raw is None:
                        _offload_raw = helper_job.get("offload_cpu", None)
                    if _offload_raw is None:
                        _offload_raw = helper_job.get("offload_to_cpu", None)
                    _offload = bool(_offload_raw)
                except Exception:
                    _offload = False
                try:
                    _batch = int(args.get("batch") or helper_job.get("batch") or 1)
                except Exception:
                    _batch = 1

                # Prompt visibility (trimmed so we don't spam the console).
                try:
                    _prompt = str(args.get("prompt") or helper_job.get("prompt") or "").strip()
                except Exception:
                    _prompt = ""
                _prompt_short = (_prompt[:120] + "…") if (len(_prompt) > 120) else _prompt

                try:
                    print(f"[txt2img][Qwen2512] start  |  {_w}x{_h}  |  steps={_steps}  |  cfg={_cfg}  |  batch={_batch}  |  seed={_seed if str(_seed).strip() else 'random'}")
                except Exception:
                    pass
                try:
                    if _sampler or _flow or _offload:
                        print(f"[txt2img][Qwen2512] sampler={_sampler or 'default'}  |  flow_shift={_flow}  |  offload_cpu={_offload}")
                except Exception:
                    pass
                try:
                    if _prompt_short:
                        print(f"[txt2img][Qwen2512] prompt: {_prompt_short}")
                        # LoRA (optional)
                        try:
                            _lp = (args.get("lora_path") or helper_job.get("lora_path") or "").strip()
                            _ls = args.get("lora_scale", None)
                            if _ls is None:
                                _ls = helper_job.get("lora_scale", None)
                            _lp2 = (args.get("lora2_path") or helper_job.get("lora2_path") or "").strip()
                            _ls2 = args.get("lora2_scale", None)
                            if _ls2 is None:
                                _ls2 = helper_job.get("lora2_scale", None)
                            if _lp:
                                print(f"[txt2img][Qwen2512] LoRA1: {_lp}  |  scale={_ls if _ls is not None else 1.0}")
                            if _lp2:
                                print(f"[txt2img][Qwen2512] LoRA2: {_lp2}  |  scale={_ls2 if _ls2 is not None else 1.0}")
                        except Exception:
                            pass
                except Exception:
                    pass

                # Try to resolve sd-cli + model file paths (best-effort) so you can spot wrong installs immediately.
                try:
                    sd_cli = None
                    diffusion = llm = vae = None
                    try:
                        from helpers import qwen2512 as _qtool  # type: ignore
                        try:
                            sd_cli = _qtool._find_sd_cli(BASE)  # type: ignore[attr-defined]
                        except Exception:
                            sd_cli = None
                        try:
                            diffusion, llm, vae = _qtool._model_paths(BASE)  # type: ignore[attr-defined]
                        except Exception:
                            diffusion = llm = vae = None
                    except Exception:
                        sd_cli = None
                        diffusion = llm = vae = None

                    def _pinfo(pth):
                        try:
                            if not pth:
                                return None
                            pp = Path(str(pth))
                            if not pp.exists():
                                return str(pp)
                            sz = int(pp.stat().st_size or 0)
                            return f"{pp} ({_bytes_human(sz)})"
                        except Exception:
                            try:
                                return str(pth)
                            except Exception:
                                return None

                    if sd_cli:
                        try:
                            print(f"[txt2img][Qwen2512] sd-cli: {sd_cli}")
                        except Exception:
                            pass
                    if diffusion or llm or vae:
                        try:
                            print("[txt2img][Qwen2512] models:",
                                  "\n  diffusion:", _pinfo(diffusion),
                                  "\n  llm:", _pinfo(llm),
                                  "\n  vae:", _pinfo(vae))
                        except Exception:
                            pass

                    # Patch running JSON so the Queue row can show more than just a percent.
                    try:
                        _patch_running_json({
                            "stage": "Preparing (Qwen2512)",
                            "engine": "qwen2512",
                            "w": _w or None,
                            "h": _h or None,
                            "steps": _steps or None,
                            "cfg": _cfg or None,
                            "sampler": _sampler or None,
                            "flow_shift": _flow or None,
                            "seed": _seed if str(_seed).strip() else None,
                            "batch": _batch or None,
                            "offload_cpu": bool(_offload),
                            "sd_cli": str(sd_cli) if sd_cli else None,
                            "diffusion_model": str(diffusion) if diffusion else None,
                            "llm_model": str(llm) if llm else None,
                            "vae_model": str(vae) if vae else None,

"lora_path": str(args.get("lora_path") or helper_job.get("lora_path") or "") or None,
"lora_scale": (args.get("lora_scale") if args.get("lora_scale", None) is not None else helper_job.get("lora_scale", None)),
"lora2_path": str(args.get("lora2_path") or helper_job.get("lora2_path") or "") or None,
"lora2_scale": (args.get("lora2_scale") if args.get("lora2_scale", None) is not None else helper_job.get("lora2_scale", None)),
                        })
                    except Exception:
                        pass
                except Exception:
                    pass

                def _qprog(p):
                    """Progress callback for Qwen2512 txt2img (pct or step/total dict)."""
                    try:
                        elapsed = None
                        try:
                            elapsed = _now_ts() - float(_q_state.get("t0") or _now_ts())
                        except Exception:
                            elapsed = None

                        # ---- Numeric progress ----
                        if isinstance(p, (int, float)):
                            ip = int(p)
                            _progress_set(ip)
                            # Print every 10% or on finish
                            try:
                                lastp = int(_q_state.get("last_pct") or -999)
                            except Exception:
                                lastp = -999
                            if ip >= 100 or (ip - lastp) >= 10:
                                _q_state["last_pct"] = ip
                                msg = f"[txt2img][Qwen2512] {ip}%"
                                if elapsed is not None:
                                    msg += f"  |  elapsed {_fmt_dur_short(elapsed)}"
                                _print_once("last_line", _q_state, msg)
                            return

                        if not isinstance(p, dict):
                            return

                        stage = (p.get("stage") or p.get("status") or "").strip()
                        status = (p.get("status") or "").strip()
                        line = (p.get("line") or p.get("msg") or p.get("message") or "").strip()

                        pct = None
                        if "pct" in p:
                            try:
                                pct = int(float(p.get("pct") or 0))
                            except Exception:
                                pct = None
                            if pct is not None:
                                _progress_set(pct)

                        step = None
                        total = None
                        if "step" in p and "total" in p:
                            try:
                                step = int(float(p.get("step") or 0))
                                total = int(float(p.get("total") or 0))
                            except Exception:
                                step = None
                                total = None
                            if total and total > 0:
                                if pct is None:
                                    try:
                                        pct = int(100.0 * float(step) / float(total))
                                    except Exception:
                                        pct = None
                                try:
                                    _progress_set(int(100.0 * float(step) / float(total)))
                                except Exception:
                                    pass

                        # Reasonable stage inference
                        if not stage:
                            if step is not None and total is not None:
                                stage = "Denoising"
                            elif pct is not None and pct < 5:
                                stage = "Preparing"
                            elif pct is not None and pct >= 95:
                                stage = "Saving"
                            else:
                                stage = "Working"

                        # Print only on stage changes, or every ~10% bucket, or when we get a meaningful line.
                        should_print = False
                        try:
                            last_stage = _q_state.get("last_stage")
                            if stage and stage != last_stage:
                                should_print = True
                                _q_state["last_stage"] = stage
                        except Exception:
                            pass
                        try:
                            lastp = int(_q_state.get("last_pct") or -999)
                        except Exception:
                            lastp = -999
                        if pct is not None:
                            bucket = int(max(0, min(100, int(pct))) / 10)
                            last_bucket = int(max(0, min(100, int(lastp))) / 10) if lastp >= 0 else -999
                            if bucket != last_bucket:
                                should_print = True
                                _q_state["last_pct"] = int(pct)

                        if line:
                            # Avoid printing identical lines over and over.
                            if line != _q_state.get("last_line"):
                                _q_state["last_line"] = line
                                # Keep this conservative: only print short-ish lines.
                                if len(line) <= 180:
                                    should_print = True

                        if not should_print:
                            return

                        parts = [f"[txt2img][Qwen2512] {stage}"]
                        if step is not None and total is not None and total > 0:
                            parts.append(f"{step}/{total}")
                        elif pct is not None:
                            parts.append(f"{pct}%")
                        if status and status != stage:
                            parts.append(status)
                        if line and line not in (stage, status):
                            parts.append(line)

                        msg = "  |  ".join([str(x) for x in parts if x])
                        if elapsed is not None:
                            msg += f"  |  elapsed {_fmt_dur_short(elapsed)}"
                        _print_once("last_line", _q_state, msg)

                        # Patch running job JSON with richer fields (best-effort)
                        try:
                            if RUNNING_JSON_FILE:
                                rj = Path(RUNNING_JSON_FILE)
                                j = json.loads(rj.read_text(encoding='utf-8')) if rj.exists() else {}
                                changed = False
                                for k in ('step','total','status','stage','pct'):
                                    if k in p and j.get(k) != p.get(k):
                                        j[k] = p.get(k)
                                        changed = True
                                if stage and j.get("stage") != stage:
                                    j["stage"] = stage
                                    changed = True
                                if pct is not None and j.get("pct") != int(pct):
                                    j["pct"] = int(pct)
                                    changed = True
                                if line and j.get("last_line") != line:
                                    j["last_line"] = line
                                    changed = True
                                if changed:
                                    j['status_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                                    rj.write_text(json.dumps(j, indent=2), encoding='utf-8')
                        except Exception:
                            pass
                    except Exception:
                        return

                res = _txt2img.generate_qwen_images(helper_job, progress_cb=_qprog, cancel_event=None)

            else:
                def _zprog(p):
                    """Progress callback for txt2img engines (pct or step/total dict)."""
                    try:
                        if isinstance(p, (int, float)):
                            _progress_set(int(p))
                            return
                        if not isinstance(p, dict):
                            return
                        if 'pct' in p:
                            try:
                                _progress_set(int(p.get('pct') or 0))
                            except Exception:
                                pass
                        elif 'step' in p and 'total' in p:
                            try:
                                step = float(p.get('step') or 0)
                                total = float(p.get('total') or 0)
                                if total > 0:
                                    _progress_set(int(100.0 * step / total))
                            except Exception:
                                pass
                        # Patch running job JSON with richer fields (best-effort)
                        try:
                            if RUNNING_JSON_FILE:
                                rj = Path(RUNNING_JSON_FILE)
                                j = json.loads(rj.read_text(encoding='utf-8')) if rj.exists() else {}
                                changed = False
                                for k in ('step','total','status','stage'):
                                    if k in p and j.get(k) != p.get(k):
                                        j[k] = p.get(k)
                                        changed = True
                                if changed:
                                    j['status_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                                    rj.write_text(json.dumps(j, indent=2), encoding='utf-8')
                        except Exception:
                            pass
                    except Exception:
                        return
                res = _txt2img.generate_qwen_images(helper_job, progress_cb=_zprog, cancel_event=None)
        except Exception as e:
            try:
                _mark_error(job, f"txt2img helper failed: {e}")
            except Exception:
                pass
            try:
                print("[worker txt2img] helper path failed; falling back to legacy diffusers.")
            except Exception:
                pass
            return _txt2img_generate_legacy(job, cfg, mani)

        if not res or not res.get("files"):
            try:
                _mark_error(job, "No images produced (helpers.txt2img returned empty result).")
            except Exception:
                pass
            try:
                print("[worker txt2img] helper produced no images; falling back to legacy diffusers.")
            except Exception:
                pass
            return _txt2img_generate_legacy(job, cfg, mani)

    # Map result back onto the queue job for UI / JSON
    try:
        files = res.get("files") or []
        job["files"] = files
        if files:
            job["produced"] = files[-1]
        backend = res.get("engine") or res.get("backend")
        if backend:
            job["backend"] = backend
        model = res.get("model")
        if model:
            job["model"] = model
    except Exception:
        pass

    try:
        _progress_set(100)
    except Exception:
        pass
    return 0


def _txt2img_generate_legacy(job, cfg, mani):
    # Offline txt2img using local diffusers pipeline
    try:
        import importlib, time
        from pathlib import Path as P

        args = job.get("args") or {}
        try:
            job["title"] = args.get("label") or (args.get("prompt","")[:80] or "txt2img")
        except Exception:
            pass

        out_dir = P(job.get("out_dir") or "."); out_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import torch/diffusers so worker can start even if not installed
        torch = importlib.import_module("torch")
        from diffusers import (
            StableDiffusionPipeline, StableDiffusionXLPipeline,
            EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
            HeunDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler
        )

        prompt   = args.get("prompt","") or ""
        negative = args.get("negative","") or ""
        steps    = int(args.get("steps") or 25)
        seed     = int(args.get("seed") or 0)
        batch    = int(args.get("batch") or 1)
        width    = int(args.get("width") or 1024)
        height   = int(args.get("height") or 768)
        cfg_scale= float(args.get("cfg_scale") or 7.5)
        sampler  = (args.get("sampler") or "").strip().lower()
        attn_slice = bool(args.get("attn_slicing"))
        fmt = (args.get("format") or _infer_image_format_from_input(args) or "png").lower()
        name_tmpl= args.get("filename_template") or f"sd_{{seed}}_{{idx:03d}}.{fmt}"

        ROOT = P(__file__).resolve().parent.parent
        default_model = ROOT / "models" / "SD15" / "DreamShaper_8_pruned.safetensors"
        model_path = args.get("model_path") or str(default_model)
        mp = P(model_path)
        if not mp.exists():
            model_path = str(default_model)

        device = "cuda" if getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        is_sdxl = ("sdxl" in model_path.lower()) or ("sd_xl" in model_path.lower())
        try:
            if is_sdxl:
                try:
                    pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype, local_files_only=True)
                except Exception:
                    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
            else:
                pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=dtype, local_files_only=True)
        except Exception as e:
            _mark_error(job, f"Model load failed: {e}")
            return 2

        try: pipe = pipe.to(device)
        except Exception: pass

        try:
            sched_map = {
                "euler a": EulerAncestralDiscreteScheduler,
                "dpm++ 2m": DPMSolverMultistepScheduler,
                "heun": HeunDiscreteScheduler,
                "unipc": UniPCMultistepScheduler,
                "ddim": DDIMScheduler,
            }
            if sampler in sched_map:
                pipe.scheduler = sched_map[sampler].from_config(pipe.scheduler.config)
        except Exception:
            pass

        try:
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = (lambda images, **kwargs: (images, [False]*len(images)))
            if attn_slice and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        except Exception:
            pass

        sp = (args.get("seed_policy") or "fixed").lower()
        if sp == "increment":
            seeds = [seed + i for i in range(max(1, batch))]
        elif sp == "random":
            import random
            rng = random.Random(seed if seed else int(time.time()))
            seeds = [rng.randint(0, 2_147_483_647) for _ in range(max(1, batch))]
        else:
            seeds = [seed for _ in range(max(1, batch))]
        try: job["seeds"] = seeds
        except Exception: pass

        files = []
        for idx, s in enumerate(seeds):
            try:
                g = torch.Generator(device=device)
                if hasattr(g, "manual_seed"):
                    g = g.manual_seed(int(s))
            except Exception:
                g = None

            try:
                out = pipe(
                    prompt=prompt, negative_prompt=negative,
                    width=width, height=height,
                    num_inference_steps=steps, guidance_scale=cfg_scale,
                    generator=g,
                )
                img = out.images[0]
            except Exception as e:
                _mark_error(job, f"inference failed: {e}")
                return 2

            name = name_tmpl.format(seed=s, idx=idx)
            if not name.lower().endswith((".png",".jpg",".jpeg",".webp")):
                name += ".png"
            outp = out_dir / name
            try:
                outp = _unique_path(outp)
                img.save(str(outp))
            except Exception as e:
                _mark_error(job, f"save failed: {e}")
                return 2

            files.append(str(outp))
            try: job["files"] = files[:]
            except Exception: pass
            try: _progress_set(int(100 * (idx+1) / max(1, len(seeds))))
            except Exception: pass

        if files:
            try:
                job["produced"] = files[-1]
                job["backend"] = "diffusers"
            except Exception:
                pass
            return 0

        _mark_error(job, "No images produced.")
        return 2

    except Exception as e:
        try: _mark_error(job, f"txt2img exception: {e}")
        except Exception: pass
        return 2


def ace_generate(job, cfg, mani):
    """Queue worker entry for ACE-Step (text-to-music or audio-to-audio).

    Accepts job types:
      - "ace_text2music"
      - "ace_audio2audio"
      - "ace" / "ace_step" / "ace_music" (aliases)

    This mirrors the logic of helpers/ace.py:AceWorker.run but uses the
    job['args'] dict plus Ace config JSON (presets/setsave/ace.json).
    """
    import subprocess as _subprocess, tempfile as _tempfile, time as _time
    from pathlib import Path as _Path

    # Pull args safely
    try:
        args = job.get("args") or {}
    except Exception:
        args = {}

    # Friendly title / label for the queue row
    try:
        title = args.get("label") or (args.get("prompt", "")[:80] or "ACE-Step")
        job["title"] = title
        try:
            job["label"] = title
        except Exception:
            pass
        try:
            a = job.get("args") or {}
            if not a.get("label"):
                a["label"] = title
            job["args"] = a
        except Exception:
            pass
    except Exception:
        pass

    # Resolve FrameVision root
    root = BASE

    # Load ACE config JSON (same file AceConfig uses)
    ace_cfg = {}
    try:
        cfg_path = root / "presets" / "setsave" / "ace.json"
        if cfg_path.exists():
            ace_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        ace_cfg = {}

    def _cfg(key, default=None):
        """Prefer explicit job args, then ace.json, then default."""
        v = args.get(key, None)
        if v is None:
            v = ace_cfg.get(key, default)
        return v if v is not None else default

    # Core generation parameters
    prompt = _cfg("prompt", "")
    negative_prompt = _cfg("negative_prompt", "")
    lyrics = _cfg("lyrics", "")
    audio_duration = float(_cfg("audio_duration", 60.0) or 60.0)
    infer_step = int(_cfg("infer_step", 60) or 60)
    guidance_scale = float(_cfg("guidance_scale", 15.0) or 15.0)
    scheduler_type = _cfg("scheduler_type", "euler") or "euler"
    cfg_type = _cfg("cfg_type", "apg") or "apg"

    # Advanced guidance / ERG settings
    omega_scale = float(_cfg("omega_scale", 10.0) or 10.0)
    guidance_interval = float(_cfg("guidance_interval", 0.5) or 0.5)
    guidance_interval_decay = float(_cfg("guidance_interval_decay", 0.0) or 0.0)
    min_guidance_scale = float(_cfg("min_guidance_scale", 3.0) or 3.0)
    guidance_scale_text = float(_cfg("guidance_scale_text", 5.0) or 5.0)
    guidance_scale_lyric = float(_cfg("guidance_scale_lyric", 1.5) or 1.5)
    use_erg_tag = bool(_cfg("use_erg_tag", True))
    use_erg_lyric = bool(_cfg("use_erg_lyric", False))
    use_erg_diffusion = bool(_cfg("use_erg_diffusion", True))

    # Device / precision
    bf16 = bool(_cfg("bf16", True))
    cpu_offload = bool(_cfg("cpu_offload", False))
    overlapped_decode = bool(_cfg("overlapped_decode", False))
    device_id = int(_cfg("device_id", 0) or 0)

    # Seed handling: single-seed text or audio2audio job
    try:
        seed_val = int(_cfg("seed", 0) or 0)
    except Exception:
        seed_val = 0
    if seed_val == 0:
        try:
            import random as _rnd
            seed_val = _rnd.randint(0, 2_147_483_647)
        except Exception:
            seed_val = 0
    actual_seeds = [int(seed_val)]
    manual_seeds = ", ".join(map(str, actual_seeds))

    # OSS steps: list or string
    oss_steps = _cfg("oss_steps", [])
    if isinstance(oss_steps, (list, tuple)):
        oss_steps_str = ", ".join(map(str, oss_steps))
    else:
        oss_steps_str = str(oss_steps or "")

    # Reference audio
    ref_audio_input = str(_cfg("ref_audio_input", "") or "").strip()
    ref_audio_strength = float(_cfg("ref_audio_strength", 0.5) or 0.5)
    if ref_audio_input:
        try:
            p = _Path(ref_audio_input)
            if not p.is_absolute():
                p = (root / p).resolve()
            ref_audio_input = str(p)
        except Exception:
            ref_audio_input = str(ref_audio_input)
        audio2audio_enable = True
    else:
        ref_audio_input = None
        audio2audio_enable = bool(_cfg("audio2audio_enable", False))

    # Output directory
    try:
        out_dir = _Path(job.get("out_dir") or (root / "output" / "ace"))
    except Exception:
        out_dir = root / "output" / "ace"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Build descriptive filename using user track name, seed and preset
    try:
        track_name_raw = str(_cfg("track_name", "") or "").strip()
    except Exception:
        track_name_raw = ""
    try:
        preset_name_raw = str(_cfg("preset_name", "") or "").strip()
    except Exception:
        preset_name_raw = ""
    import re as _re
    def _slugify_name(value: str, default: str) -> str:
        value = (value or "").strip()
        if not value:
            return default
        value = _re.sub(r"[^a-zA-Z0-9_]+", "_", value)
        value = value.strip("_") or default
        return value.lower()
    track_slug = _slugify_name(track_name_raw, "track")
    preset_slug = _slugify_name(preset_name_raw, "preset")

    base_seed = actual_seeds[0] if actual_seeds else int(seed_val or 0)
    filename = f"{track_slug}_{base_seed}_{preset_slug}.wav"
    output_path = str(out_dir / filename)

    # Checkpoint directory: use ace.json if it overrides, else default under presets/extra_env
    checkpoint_rel = ace_cfg.get("checkpoint_path", ".ace_env/ACE-Step/checkpoints")
    try:
        checkpoint_path = str((root / checkpoint_rel).resolve())
    except Exception:
        # Fallback: standard location under presets/extra_env
        checkpoint_path = str((root / "presets" / "extra_env" / ".ace_env" / "ACE-Step" / "checkpoints").resolve())

    # Build job payload for the ACE runner
    jobj = {
        "checkpoint_path": checkpoint_path,
        "dtype": "bfloat16" if bf16 else "float32",
        "torch_compile": False,
        "cpu_offload": bool(cpu_offload),
        "overlapped_decode": bool(overlapped_decode),
        "device_id": int(device_id),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "lyrics": lyrics,
        "audio_duration": float(audio_duration),
        "infer_step": int(infer_step),
        "guidance_scale": float(guidance_scale),
        "scheduler_type": scheduler_type,
        "cfg_type": cfg_type,
        "manual_seeds": manual_seeds,
        "omega_scale": float(omega_scale),
        "guidance_interval": float(guidance_interval),
        "guidance_interval_decay": float(guidance_interval_decay),
        "min_guidance_scale": float(min_guidance_scale),
        "use_erg_tag": bool(use_erg_tag),
        "use_erg_lyric": bool(use_erg_lyric),
        "use_erg_diffusion": bool(use_erg_diffusion),
        "oss_steps": oss_steps_str,
        "guidance_scale_text": float(guidance_scale_text),
        "guidance_scale_lyric": float(guidance_scale_lyric),
        "audio2audio_enable": bool(audio2audio_enable),
        "ref_audio_strength": float(ref_audio_strength),
        "ref_audio_input": ref_audio_input,
        "output_path": output_path,
    }

    # Write temporary job JSON
    tmp_dir = _Path(_tempfile.gettempdir())
    tmp_path = tmp_dir / f"framevision_ace_job_{os.getpid()}_{int(_time.time())}.json"
    try:
        tmp_path.write_text(json.dumps(jobj), encoding="utf-8")
    except Exception as e:
        _mark_error(job, f"Could not write ACE job file: {e}")
        return 2

    # Determine ACE env Python and runner script (same layout as in helpers/ace.py)
    ace_env_dir = root / "presets" / "extra_env" / ".ace_env"
    ace_repo_dir = ace_env_dir / "ACE-Step"
    if os.name == "nt":
        ace_python = ace_env_dir / "Scripts" / "python.exe"
    else:
        ace_python = ace_env_dir / "bin" / "python"

    runner_script = ace_repo_dir / "framevision_ace_runner.py"
    if not ace_python.exists():
        _mark_error(job, f"ACE env python not found at: {ace_python}")
        return 2
    if not runner_script.exists():
        _mark_error(job, f"ACE runner script not found at: {runner_script}")
        return 2

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    cmd = [str(ace_python), str(runner_script), str(tmp_path)]

    # Progress/ETA: stream ACE logs and infer "X/Y" style step progress when possible.
    try:
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ace_{stamp}.log"
    except Exception:
        log_file = None

    try:
        _progress_set(5)
    except Exception:
        pass

    import re as _re

    step_re = _re.compile(r"(\d+)\s*/\s*(\d+)")
    total_steps = None
    last_step = 0
    last_pct = -1
    start_ts = _time.time()

    def _update_progress():
        nonlocal last_pct
        pct = 5.0
        if total_steps and total_steps > 0:
            frac = max(0.0, min(1.0, float(last_step) / float(total_steps)))
            pct = 5.0 + 90.0 * frac
        else:
            # Fallback: small time-based ramp up to ~25% so very fast jobs aren't stuck at 5%.
            elapsed = max(0.0, _time.time() - start_ts)
            pct = 5.0 + min(20.0, (elapsed / 20.0) * 20.0)
        ipct = int(max(5, min(98, round(pct))))
        if ipct != last_pct:
            try:
                _progress_set(ipct)
            except Exception:
                pass
            last_pct = ipct

    code = 0
    try:
        lf = None
        if log_file is not None:
            try:
                lf = open(log_file, "w", encoding="utf-8")
                lf.write("CMD: " + " ".join(str(x) for x in cmd) + "\n\n")
                lf.flush()
            except Exception:
                lf = None

        p = _subprocess.Popen(
            cmd,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            _patch_running_json({'active_pid': int(getattr(p,'pid',0) or 0), 'active_cmd': ' '.join([str(x) for x in cmd])})
        except Exception:
            pass
        q, _t = _start_proc_reader(p)
        was_cancel = False
        while True:
            if _cancel_requested():
                was_cancel = True
                _note_cancel("Cancelled by user")
                try:
                    _kill_process_tree(getattr(p, 'pid', 0) or 0)
                except Exception:
                    pass
                break
            try:
                item = q.get_nowait()
            except _queue.Empty:
                item = None
            if item is None:
                if p.poll() is not None:
                    break
                _update_progress()
                _time.sleep(0.3)
                continue
            if item is _EOF:
                break
            line = item
            if not line:
                continue

            if lf is not None:
                try:
                    lf.write(line)
                except Exception:
                    pass
            try:
                print("[ACE]", line, end="")
            except Exception:
                pass

            try:
                m = step_re.search(line)
            except Exception:
                m = None
            if m:
                try:
                    cur = int(m.group(1))
                    tot = int(m.group(2))
                    if tot > 0:
                        if total_steps is None or total_steps != tot:
                            total_steps = tot
                        if cur > last_step:
                            last_step = min(cur, tot)
                except Exception:
                    pass

            _update_progress()

        code = p.wait()
        if was_cancel: code = 130
        if lf is not None:
            try:
                lf.write(f"\nEXIT CODE: {code}\n")
                lf.close()
            except Exception:
                pass
    except Exception as e:
        _mark_error(job, f"ACE-Step runner exception: {e}")
        return 2

    if code != 0:
        _mark_error(job, f"ACE-Step runner failed (code {code}).")
        return 2
    # Success: record produced file and mark 100%
    try:
        job["produced"] = output_path
    except Exception:
        pass
    try:
        _progress_set(100)
    except Exception:
        pass
    return 0


def ace15_generate(job, cfg, mani):
    """Queue worker entry for Ace-Step 1.5 (runs cli.py with a generated TOML).

    Expected job shape (from helpers/ace_step_15.py queue button):
      type: "ace_step_15"
      out_dir: output folder (defaults to presets config if missing)
      args: {
        "env_python": ".../environments/.ace_15/Scripts/python.exe",
        "cli_py": ".../ACE-Step/cli.py",
        "project_root": ".../ACE-Step",
        "cfg_path": ".../output/.../ace_step_run_YYYYmmdd_HHMMSS.toml",
        "hide_console": bool,
        "label": str (optional)
      }

    This function is intentionally independent from PySide so it can run inside the worker.
    """
    import subprocess as _subprocess
    import time as _time
    import tempfile as _tempfile
    from pathlib import Path as _Path

    try:
        args = job.get("args") or {}
    except Exception:
        args = {}

    # Friendly title in queue row
    try:
        title = args.get("label") or "Ace-Step 1.5"
        job["title"] = title
        try:
            job["label"] = title
        except Exception:
            pass
        try:
            a = job.get("args") or {}
            if not a.get("label"):
                a["label"] = title
            job["args"] = a
        except Exception:
            pass
    except Exception:
        pass

    env_python = str(args.get("env_python") or "").strip()
    cli_py = str(args.get("cli_py") or "").strip()
    project_root = str(args.get("project_root") or "").strip()
    cfg_path = str(args.get("cfg_path") or "").strip()

    if not env_python or not _Path(env_python).exists():
        _mark_error(job, f"Ace-Step 1.5 env python not found: {env_python}")
        return 2
    if not cli_py or not _Path(cli_py).exists():
        _mark_error(job, f"Ace-Step 1.5 cli.py not found: {cli_py}")
        return 2
    if not project_root or not _Path(project_root).exists():
        # Some installs pass a file path; recover its parent.
        try:
            pr = _Path(project_root)
            project_root = str(pr.parent if pr.is_file() else pr)
        except Exception:
            pass
    if not project_root or not _Path(project_root).exists():
        _mark_error(job, f"Ace-Step 1.5 project_root not found: {project_root}")
        return 2

    # Output folder
    try:
        out_dir = _Path(job.get("out_dir") or "").resolve()
    except Exception:
        out_dir = _Path(BASE / "output" / "audio" / "ace15").resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Config path must exist; if not, fail (keeps job reproducible)
    if not cfg_path or not _Path(cfg_path).exists():
        _mark_error(job, f"Ace-Step 1.5 config TOML not found: {cfg_path}")
        return 2


    # --- Post-processing helpers (matches ace_step_15.py behavior) ---
    def _sanitize_part(s: str) -> str:
        try:
            s = (s or "").strip()
        except Exception:
            s = ""
        if not s:
            return ""
        out = []
        for ch in s:
            if ch.isalnum() or ch in {"-", "_"}:
                out.append(ch)
            elif ch.isspace() or ch in {"/", "\\", ":", "|", "*", "?", '"', "<", ">"}:
                out.append("_")
            else:
                out.append("_")
        cleaned = "".join(out)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("._ ")

    def _derive_stamp(cfg_path_s: str) -> str:
        try:
            stem = _Path(cfg_path_s).stem
        except Exception:
            stem = ""
        stamp = ""
        if stem:
            if stem.startswith("ace_step_run_"):
                stamp = stem.replace("ace_step_run_", "", 1)
            else:
                stamp = stem
        if not stamp:
            stamp = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
        return stamp

    def _extract_sub_seed(label_s: str):
        # label: "Ace-Step 1.5: <sub> (seed <seed>)"
        sub = "Custom"
        seed = "AUTO"
        try:
            s = str(label_s or "")
        except Exception:
            s = ""
        try:
            mm = re.search(r"^\s*Ace-Step\s*1\.5\s*:\s*(.*?)\s*\(\s*seed\s+([^\)]+)\)\s*$", s, re.IGNORECASE)
            if mm:
                sub = (mm.group(1) or "").strip() or sub
                seed = (mm.group(2) or "").strip() or seed
        except Exception:
            pass
        return sub, seed

    def _move_instruction_txt(project_root_s: str, out_dir_p: _Path, stamp: str, exit_code: int) -> None:
        try:
            proj = _Path(project_root_s)
            instr = proj / "instruction.txt"
            if not instr.exists() or (not instr.is_file()):
                return
            try:
                out_dir_p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            base_name = f"instruction_used_{stamp}.txt" if exit_code == 0 else f"instruction_error_{stamp}.txt"
            dst = out_dir_p / base_name
            if dst.exists():
                i = 2
                while True:
                    cand = out_dir_p / f"{dst.stem}_{i}{dst.suffix}"
                    if not cand.exists():
                        dst = cand
                        break
                    i += 1
            try:
                shutil.move(str(instr), str(dst))
            except Exception:
                # Fall back to copy+delete if move fails across volumes.
                try:
                    shutil.copy2(str(instr), str(dst))
                    _safe_unlink(instr)
                except Exception:
                    pass
        except Exception:
            return

    def _rename_outputs(out_dir_p: _Path, files_list: list, sub: str, seed: str, stamp: str) -> list:
        if not files_list:
            return files_list
        sub_s = _sanitize_part(sub) or "Custom"
        seed_s = _sanitize_part(seed) or "AUTO"
        renamed = []
        multi = len(files_list) > 1
        for idx2, fp in enumerate(files_list, start=1):
            try:
                src = _Path(fp)
                if not src.exists():
                    continue
                ext = src.suffix or ""
                counter = f"_{idx2}" if multi else ""
                dst = out_dir_p / f"{sub_s}__seed{seed_s}__{stamp}{counter}{ext}"
                dst = _unique_path(dst)
                try:
                    src.rename(dst)
                    renamed.append(str(dst.resolve()))
                except Exception:
                    renamed.append(str(src.resolve()))
            except Exception:
                continue
        return renamed

    # Snapshot existing audio outputs
    def _list_audio(d: _Path):
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        try:
            return [p for p in d.glob("*") if p.is_file() and p.suffix.lower() in exts]
        except Exception:
            return []

    before = {str(p.resolve()): float(getattr(p.stat(), "st_mtime", 0.0)) for p in _list_audio(out_dir)}
    start_ts = _time.time()

    cmd = [str(env_python), str(cli_py), "-c", str(cfg_path)]

    # Console hiding flag (Windows only)
    creationflags = 0
    try:
        if os.name == "nt" and bool(args.get("hide_console")):
            creationflags = _subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    except Exception:
        creationflags = 0

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")

    try:
        _progress_set(5)
    except Exception:
        pass

    # Lightweight time-based progress ramp until process exits.
    last_pct = -1
    def _update_progress():
        nonlocal last_pct
        elapsed = max(0.0, _time.time() - start_ts)
        pct = 5.0 + min(90.0, (elapsed / 60.0) * 90.0)  # ~60s to reach 95%
        ipct = int(max(5, min(95, round(pct))))
        if ipct != last_pct:
            try:
                _progress_set(ipct)
            except Exception:
                pass
            last_pct = ipct

    # Run
    try:
        p = _subprocess.Popen(
            cmd,
            cwd=str(_Path(project_root)),
            stdout=_subprocess.PIPE,
            stdin=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            env=env,
        )
        try:
            _patch_running_json({'active_pid': int(getattr(p,'pid',0) or 0), 'active_cmd': ' '.join([str(x) for x in cmd])})
        except Exception:
            pass

        q, _t = _start_proc_reader(p)
        was_cancel = False
        while True:
            if _cancel_requested():
                was_cancel = True
                _note_cancel("Cancelled by user")
                try:
                    _kill_process_tree(getattr(p, 'pid', 0) or 0)
                except Exception:
                    pass
                break

            try:
                item = q.get_nowait()
            except _queue.Empty:
                item = None

            if item is None:
                if p.poll() is not None:
                    break
                _update_progress()
                _time.sleep(0.3)
                continue

            if item is _EOF:
                break

            line = item
            if not line:
                continue

            try:
                print("[ACE15]", line, end="")
            except Exception:
                pass

            # Auto-continue interactive draft prompt
            if "Press Enter when ready to continue." in line:
                try:
                    if p.stdin:
                        p.stdin.write("\n")
                        p.stdin.flush()
                except Exception:
                    pass

            _update_progress()

        code = p.wait()
        if was_cancel:
            code = 130
    except Exception as e:
        _mark_error(job, f"Ace-Step 1.5 runner exception: {e}")
        return 2

    stamp = _derive_stamp(cfg_path)
    sub_for_name, seed_for_name = _extract_sub_seed(job.get('title') or job.get('label') or (args.get('label') if isinstance(args, dict) else '') or '')

    if code != 0:
        # Always move instruction.txt so the next run doesn't re-use it.
        try:
            _move_instruction_txt(project_root, out_dir, stamp, int(code))
        except Exception:
            pass
        _mark_error(job, f"Ace-Step 1.5 failed (code {code}).")
        return 2

    # Detect produced audio files
    after = _list_audio(out_dir)
    new_files = []
    for pth in after:
        try:
            rp = str(pth.resolve())
            mt = float(pth.stat().st_mtime)
            if rp not in before or mt > (before.get(rp, 0.0) + 0.0001):
                new_files.append((mt, rp))
        except Exception:
            continue
    new_files.sort(key=lambda t: t[0])
    files = [rp for _mt, rp in new_files]
    if not files and after:
        # Fallback: newest file in folder
        try:
            newest = max(after, key=lambda p: p.stat().st_mtime)
            files = [str(newest.resolve())]
        except Exception:
            files = []

    # Post-process: rename newly generated outputs to a human-friendly name (matches direct-run UI).
    try:
        if files:
            files = _rename_outputs(out_dir, files, sub_for_name, seed_for_name, stamp)
    except Exception:
        pass

    # Always move instruction.txt so the next run doesn't re-use it.
    try:
        _move_instruction_txt(project_root, out_dir, stamp, 0)
    except Exception:
        pass

    if files:
        job["files"] = files
        job["produced"] = files[-1]

    try:
        _progress_set(100)
    except Exception:
        pass
    return 0


def wan22_generate(job, cfg, mani):
    """
    Queue worker entry for Wan 2.2 TI2V (text2video / image2video) with
    best-effort progress + ETA reporting based on stdout "step" logs.

    Expected job shape:
      type: "wan22_text2video" | "wan22_image2video" | "wan22_ti2v" | "wan22"
      input: optional path to start image for image2video
      out_dir: base output folder (optional – defaults to ./output/video/wan22)
      args: {
          "prompt": str,
          "mode": "text2video" | "image2video" (optional, inferred from type),
          "image": str (start image, for image2video),
          "size": "1280*704" | "704*1280",
          "steps": int,
          "guidance": float | int,
          "guidance_scale": float | int,
          "frames": int,
          "frame_num": int,
          "seed": int,
          "base_seed": int,
          "random_seed": bool | str,
          "save_file": str,
          "output_path": str,
      }
    """
    from pathlib import Path as _Path
    import time as _time
    import re as _re
    global ROOT, BASE

    try:
        args = job.get("args") or {}
    except Exception:
        args = {}

    # Nice title in the queue row
    try:
        title = args.get("label") or (args.get("prompt", "")[:80] or "WAN 2.2")
        job["title"] = title
        # Mirror onto common fields some UIs may read
        try:
            job["label"] = title
        except Exception:
            pass
        try:
            a = job.get("args") or {}
            if not a.get("label"):
                a["label"] = title
            job["args"] = a
        except Exception:
            pass
    except Exception:
        pass

    root = BASE

    # Turbo is a real WAN engine now. The queue must preserve that selection
    # instead of falling back to the original Wan 2.2 generate.py path.
    try:
        engine_key = str(args.get("engine") or "").strip().lower()
    except Exception:
        engine_key = ""
    try:
        turbo_job = bool(args.get("turbo_model")) or engine_key in ("wan22_turbo", "wan 2.2 turbo") or "turbo" in str(job.get("type") or "").lower()
    except Exception:
        turbo_job = "turbo" in str(job.get("type") or "").lower()

    # Resolve Wan Python. Prefer the new portable env under /environments/.wan22_i2v.
    # Supports local conda layout (python.exe directly in env root) and venv layout
    # (Scripts/python.exe). The old root .wan_venv is only a legacy fallback.
    try:
        if os.name == "nt":
            candidates = [
                root / "environments" / ".wan22_i2v" / "python.exe",
                root / "environments" / ".wan22_i2v" / "Scripts" / "python.exe",
                root / ".wan_venv" / "python.exe",
                root / ".wan_venv" / "Scripts" / "python.exe",
            ]
        else:
            candidates = [
                root / "environments" / ".wan22_i2v" / "bin" / "python",
                root / "environments" / ".wan22_i2v" / "python",
                root / ".wan_venv" / "bin" / "python",
            ]
        py = ""
        for cand in candidates:
            try:
                if cand.exists():
                    py = str(cand)
                    break
            except Exception:
                pass
        if not py:
            py = sys.executable or "python"
    except Exception:
        try:
            py = sys.executable
        except Exception:
            py = "python"

    model_root = root / "models" / "wan22"
    gen = model_root / "generate.py"
    helper_wrapper = root / "helpers" / "wan22_vram_lab_cli.py"

    turbo_repo_dir = model_root / "wan_turbo" / "Wan2.2-TI2V-5B-Turbo-main"
    turbo_model_dir = model_root / "wan_turbo" / "Wan2.2-TI2V-5B-Turbo"
    turbo_gen = turbo_repo_dir / "wan2.2_fewstep.py"
    turbo_config = turbo_repo_dir / "configs" / "inference" / "wan22.yaml"
    turbo_helper_wrapper = root / "helpers" / "wan22_turbo_vramlab_cli.py"
    firstlast_helper = root / "helpers" / "wan_firstlast.py"

    if turbo_job:
        if not turbo_gen.exists():
            _mark_error(job, f"Wan2.2 Turbo script not found at {turbo_gen}")
            return 2
        if not turbo_config.exists():
            _mark_error(job, f"Wan2.2 Turbo config not found at {turbo_config}")
            return 2
        if not (turbo_model_dir / "model.pt").exists():
            _mark_error(job, f"Wan2.2 Turbo model.pt not found at {turbo_model_dir / 'model.pt'}")
            return 2
    else:
        if not gen.exists():
            _mark_error(job, f"Wan2.2 generate.py not found at {gen}")
            return 2

    if firstlast_enabled and not turbo_job:
        _mark_error(job, "First/last frame is currently supported for WAN 2.2 Turbo only.")
        return 2

    # Mode inference
    mode = (str(args.get("mode") or "") or "").strip().lower()
    t = str(job.get("type") or "").lower()
    if not mode:
        if "image" in t:
            mode = "image2video"
        else:
            mode = "text2video"
    if turbo_job:
        # Preserve queued Turbo mode. Turbo supports text2video by omitting --image,
        # and image2video/TI2V by passing --image.
        if "image" in t:
            mode = "image2video"
        elif "text" in t:
            mode = "text2video"
        elif mode not in ("text2video", "image2video"):
            mode = "text2video"

    # Core params
    size_str = (args.get("size") or "1280*704").strip() or "1280*704"
    steps = int(args.get("steps") or args.get("sample_steps") or 30)
    guidance = float(args.get("guidance") or args.get("guidance_scale") or 7)
    frames = int(args.get("frames") or args.get("frame_num") or 121)
    fps = int(args.get("fps") or args.get("sample_fps") or 24)
    if turbo_job:
        steps = 4
        guidance = 0.0

    base_seed = int(args.get("seed") or args.get("base_seed") or 42)
    rs = args.get("random_seed")
    if rs in (True, 1, "1", "true", "True", "yes", "on"):
        import random as _rnd
        try:
            base_seed = _rnd.randint(0, 2147483647)
        except Exception:
            pass

    prompt = args.get("prompt", "") or ""
    image = (
        args.get("image")
        or args.get("input_image")
        or args.get("image_path")
        or job.get("input")
        or ""
    )

    # Queue text2video jobs use a small wan22_text_prompt.txt dummy as job["input"]
    # so the queue row has a file-like input. Turbo text2video must NOT treat that
    # dummy file as --image; the Turbo repo switches to its text-only branch only
    # when --image is omitted.
    if turbo_job and mode == "text2video":
        image = ""

    try:
        firstlast_enabled = bool(args.get("firstlast_enabled", False))
    except Exception:
        firstlast_enabled = False
    try:
        firstlast_end_image = str(args.get("firstlast_end_image") or "").strip()
    except Exception:
        firstlast_end_image = ""
    try:
        firstlast_end_timing = str(args.get("firstlast_end_timing") or "late").strip().lower()
    except Exception:
        firstlast_end_timing = "late"
    try:
        firstlast_end_strength = str(args.get("firstlast_end_strength") or "high").strip().lower()
    except Exception:
        firstlast_end_strength = "high"
    try:
        firstlast_force_exact = bool(args.get("firstlast_force_exact", False))
    except Exception:
        firstlast_force_exact = False

    def _firstlast_start_fraction(mode_name):
        m = str(mode_name or "late").strip().lower()
        if m == "balanced":
            return 0.55
        if m in ("very_late", "very late", "verylate"):
            return 0.82
        return 0.70

    def _firstlast_strength_value(mode_name):
        m = str(mode_name or "high").strip().lower()
        if m == "low":
            return 0.55
        if m == "medium":
            return 0.75
        return 1.00

    # Output folder / file
    try:
        out_dir = _Path(job.get("out_dir") or (BASE / "output" / "video" / "wan22"))
    except Exception:
        out_dir = BASE / "output" / "video" / "wan22"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file_arg = args.get("save_file") or args.get("output_path") or ""
    if save_file_arg:
        save_path = _Path(save_file_arg)
        if not save_path.is_absolute():
            save_path = out_dir / save_path
    else:
        base_name = args.get("filename") or f"wan22_{job.get('id') or int(time.time())}.mp4"
        if not str(base_name).lower().endswith(".mp4"):
            base_name = f"{base_name}.mp4"
        save_path = out_dir / base_name
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _ensure_turbo_links():
        """Create the minimal wan_models layout expected by the Turbo repo.

        Do not junction Wan2.2-TI2V-5B back to models\\wan22. That creates a
        recursive wan_turbo loop and also breaks when the shared T5 encoder has
        been moved to models\\shared.
        """
        try:
            import shutil as _shutil
            import subprocess as _subp

            wm = turbo_repo_dir / "wan_models"
            wm.mkdir(parents=True, exist_ok=True)

            expected_base = wm / "Wan2.2-TI2V-5B"
            expected_turbo = wm / "Wan2.2-TI2V-5B-Turbo"

            def _remove_empty_or_link_dir(p):
                """Remove only an empty dir or a Windows junction/symlink-looking dir."""
                try:
                    if not p.exists():
                        return
                    # If this was the old junction, rmdir without /s removes only the link.
                    if os.name == "nt":
                        try:
                            _subp.run(["cmd", "/c", "rmdir", str(p)], stdout=_subp.PIPE, stderr=_subp.PIPE, text=True, timeout=20)
                        except Exception:
                            pass
                    else:
                        try:
                            if p.is_symlink():
                                p.unlink()
                        except Exception:
                            pass
                    # For a real empty folder, normal rmdir is safe. It will fail if non-empty.
                    try:
                        if p.exists():
                            p.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass

            def _link_or_copy_file(src, dst):
                try:
                    src = Path(src)
                    dst = Path(dst)
                    if not src.exists() or not src.is_file():
                        return False
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        try:
                            if dst.stat().st_size == src.stat().st_size:
                                return True
                        except Exception:
                            pass
                        try:
                            dst.unlink()
                        except Exception:
                            return False
                    try:
                        if os.name == "nt":
                            r = _subp.run(["cmd", "/c", "mklink", "/H", str(dst), str(src)], stdout=_subp.PIPE, stderr=_subp.PIPE, text=True, timeout=20)
                            if int(getattr(r, "returncode", 1) or 0) == 0 and dst.exists():
                                return True
                        else:
                            try:
                                os.link(str(src), str(dst))
                                if dst.exists():
                                    return True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        _shutil.copy2(str(src), str(dst))
                        return dst.exists()
                    except Exception:
                        return False
                except Exception:
                    return False

            # If the expected base path exists but does not contain the shared/local T5,
            # it is probably the old junction or a stale broken folder. Remove only if safe.
            shared_dir = root / "models" / "shared"
            t5_name = "models_t5_umt5-xxl-enc-bf16.pth"
            t5_src = shared_dir / t5_name
            if not t5_src.exists():
                t5_src = model_root / t5_name

            try:
                if expected_base.exists() and not (expected_base / t5_name).exists() and t5_src.exists():
                    _remove_empty_or_link_dir(expected_base)
            except Exception:
                pass

            try:
                expected_base.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Root files the Turbo repo expects under wan_models/Wan2.2-TI2V-5B.
            try:
                _link_or_copy_file(t5_src, expected_base / t5_name)
            except Exception:
                pass
            for name in ("Wan2.2_VAE.pth", "diffusion_pytorch_model.safetensors.index.json"):
                try:
                    _link_or_copy_file(model_root / name, expected_base / name)
                except Exception:
                    pass

            # Keep only the Turbo model folder link/junction, not the base folder.
            try:
                if not expected_turbo.exists():
                    if os.name == "nt":
                        _subp.run(["cmd", "/c", "mklink", "/J", str(expected_turbo), str(turbo_model_dir)], cwd=str(turbo_repo_dir), stdout=_subp.PIPE, stderr=_subp.PIPE, text=True, timeout=20)
                    else:
                        os.symlink(str(turbo_model_dir), str(expected_turbo), target_is_directory=True)
            except Exception:
                pass
        except Exception:
            pass

    # Build Wan command args (without python executable).
    # Turbo uses the separate few-step repo/CLI; original Wan keeps generate.py.
    cwd_run = None
    if turbo_job:
        if mode == "image2video" and not image:
            _mark_error(job, "Wan2.2 Turbo image2video / TI2V queue jobs require a start image.")
            return 2
        if mode not in ("text2video", "image2video"):
            _mark_error(job, f"Wan2.2 Turbo queue jobs support text2video or image2video, not {mode!r}.")
            return 2
        if (int(frames) - 1) % 4 != 0:
            _mark_error(job, "Wan2.2 Turbo frame count must be 1 more than a multiple of 4. Examples: 33, 61, 81, 101, 121, 161, 201, 241.")
            return 2
        try:
            width_s, height_s = str(size_str).replace("x", "*").split("*", 1)
            width = int(width_s.strip())
            height = int(height_s.strip().split(",", 1)[0])
        except Exception:
            width, height = 1280, 704
        _ensure_turbo_links()
        if firstlast_enabled:
            if not firstlast_helper.exists():
                _mark_error(job, f"WAN first/last helper is missing: {firstlast_helper}")
                return 2
            if not firstlast_end_image:
                _mark_error(job, "Use last frame is enabled, but no Last frame image is selected.")
                return 2
            wan_args = [
                "--repo_root", str(turbo_repo_dir),
                "--config_path", str(turbo_config),
                "--checkpoint_folder", str(turbo_model_dir),
                "--output_path", str(save_path),
                "--prompt", prompt or "A cinematic video, realistic motion",
                "--h", str(height),
                "--w", str(width),
                "--num_frames", str(frames),
                "--fps", str(fps),
                "--seed", str(base_seed),
                "--end_image", str(firstlast_end_image),
                "--end-influence-start", str(_firstlast_start_fraction(firstlast_end_timing)),
                "--end-influence-strength", str(_firstlast_strength_value(firstlast_end_strength)),
            ]
            if firstlast_force_exact:
                wan_args.append("--force-exact-last-frame")
            if mode == "image2video" and image:
                wan_args += ["--start_image", str(image)]
        else:
            wan_args = [
                "--config_path", str(turbo_config.relative_to(turbo_repo_dir)) if str(turbo_config).startswith(str(turbo_repo_dir)) else str(turbo_config),
                "--checkpoint_folder", str(_Path("wan_models") / "Wan2.2-TI2V-5B-Turbo"),
                "--output_path", str(save_path),
                "--prompt", prompt or "A cinematic video, realistic motion",
                "--h", str(height),
                "--w", str(width),
                "--num_frames", str(frames),
                "--seed", str(base_seed),
            ]
            if mode == "image2video":
                wan_args += ["--image", str(image)]
    else:
        wan_args = [
            "--task", "ti2v-5B",
            "--size", size_str,
            "--sample_steps", str(steps),
            "--sample_guide_scale", str(guidance),
            "--base_seed", str(base_seed),
            "--frame_num", str(frames),
            "--fps", str(fps),
            "--ckpt_dir", str(model_root),
            "--convert_model_dtype",
        ]
        if prompt:
            wan_args += ["--prompt", prompt]
        if mode == "image2video":
            if not image:
                _mark_error(job, "Wan2.2 image2video mode requires an image path (args['image'] or job['input']).")
                return 2
            wan_args += ["--image", str(image)]

        # Optional model offload override: only pass when enabled.
        try:
            if bool(args.get("offload_model")):
                wan_args += ["--offload_model", "True"]
        except Exception:
            pass

        # Optional T5 CPU flag.
        try:
            if bool(args.get("t5_cpu")):
                wan_args += ["--t5_cpu"]
        except Exception:
            pass

        wan_args += ["--save_file", str(save_path)]

    # Route queued Wan jobs through VRAM Lab helper when the UI requested it.
    try:
        vram_mode = str(args.get("vram_lab") or "off").strip().lower()
    except Exception:
        vram_mode = "off"
    try:
        vram_profile = str(args.get("vram_profile") or "24").strip()
    except Exception:
        vram_profile = "24"
    if vram_profile not in ("12", "16", "24"):
        vram_profile = "24"

    # FlashAttention toggle from UI/queue. Missing value defaults to ON for
    # backward-compatible old jobs. Accept both old/new key names.
    try:
        if "flash_attention" in args:
            flash_attention = bool(args.get("flash_attention"))
        elif "flash_attn" in args:
            flash_attention = bool(args.get("flash_attn"))
        else:
            flash_attention = True
    except Exception:
        flash_attention = True
    try:
        crawl_guard = bool(args.get("crawl_guard", False))
    except Exception:
        crawl_guard = False

    if vram_mode != "off":
        if turbo_job:
            if not turbo_helper_wrapper.exists():
                _mark_error(job, f"Wan 2.2 Turbo VRAM Lab helper is missing: {turbo_helper_wrapper}")
                return 2
            turbo_target = firstlast_helper if firstlast_enabled else turbo_gen
            cmd = [
                py,
                str(turbo_helper_wrapper),
                "--vram-lab", "safe",
                "--vram-profile", vram_profile,
                "--wan-generate", str(turbo_target),
                "--wan-root", str(turbo_repo_dir),
                "--base-model-dir", str(model_root),
                "--turbo-model-dir", str(turbo_model_dir),
            ]
            if not flash_attention:
                cmd.append("--disable-flash-attention")
            if crawl_guard:
                cmd.append("--enable-crawl-guard")
            cmd += ["--"] + list(wan_args)
            cwd_run = str(root)
        else:
            if not helper_wrapper.exists():
                _mark_error(job, f"Wan VRAM Lab helper is missing: {helper_wrapper}")
                return 2
            cmd = [
                py,
                str(helper_wrapper),
                "--vram-lab", "safe",
                "--vram-profile", vram_profile,
                "--wan-generate", str(gen),
                "--wan-root", str(model_root),
            ]
            if not flash_attention:
                cmd.append("--disable-flash-attention")
            cmd += ["--"] + list(wan_args)
            cwd_run = str(root)
        try:
            job["wan_vram_lab_active"] = True
            job["wan_vram_lab_profile"] = vram_profile
            job["wan_flash_attention"] = bool(flash_attention)
            job["wan_crawl_guard"] = bool(crawl_guard)
            job["wan_engine"] = "wan22_turbo" if turbo_job else "wan22"
            job["wan_firstlast"] = bool(firstlast_enabled)
        except Exception:
            pass
    else:
        if turbo_job:
            if firstlast_enabled:
                cmd = [py, str(firstlast_helper)] + list(wan_args)
                cwd_run = str(root)
            else:
                cmd = [py, str(turbo_gen)] + list(wan_args)
                cwd_run = str(turbo_repo_dir)
        else:
            cmd = [py, str(gen)] + list(wan_args)
            cwd_run = str(model_root)
        try:
            job["wan_vram_lab_active"] = False
            job["wan_flash_attention"] = bool(flash_attention)
            job["wan_crawl_guard"] = bool(crawl_guard)
            job["wan_engine"] = "wan22_turbo" if turbo_job else "wan22"
            job["wan_firstlast"] = bool(firstlast_enabled)
        except Exception:
            pass

    # Expose command for debugging
    try:
        job["cmd"] = " ".join(str(x) for x in cmd)
    except Exception:
        pass

    # --- Progress-aware run: 2-phase estimate (steps + post) ---
    # We treat the second phase as roughly equal length to the sampling steps.
    # Progress is estimated from stdout "step X/Y" logs and wall-clock.
    log_dir = ROOT / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"wan22_{stamp}.log"

    import subprocess as _sp

    # Step / ETA tracking
    start_ts = _time.time()
    first_step_ts = None
    est_step_sec = None
    total_steps = None
    last_step_idx = 0
    est_total_dur = None
    last_pct = -1

    step_re = _re.compile(r"(?i)(?:step[^0-9]*)?(\d+)\s*/\s*(\d+)")

    def _update_progress_from_state():
        nonlocal last_pct, est_total_dur
        now = _time.time()
        elapsed = max(0.0, now - start_ts)

        if total_steps and est_step_sec:
            # Estimated duration for one "phase" (sampling) based on steps.
            phase1_dur = est_step_sec * float(total_steps)
            # Assume phase2 ~= phase1.
            est_total_dur = max(phase1_dur * 2.0, phase1_dur + 1.0)
            if elapsed <= phase1_dur:
                # First phase: clamp by observed step index.
                if total_steps > 0:
                    frac1 = max(0.0, min(1.0, float(last_step_idx) / float(total_steps)))
                else:
                    frac1 = 0.0
                # Map to 0–50%.
                pct = 50.0 * frac1
            else:
                # Second phase: 50–100% over remaining time.
                rem = max(0.1, est_total_dur - phase1_dur)
                frac2 = max(0.0, min(1.0, (elapsed - phase1_dur) / rem))
                pct = 50.0 + 50.0 * frac2
        elif total_steps:
            # We know total steps but not timing yet – just use steps → 0–50%.
            if total_steps > 0:
                frac1 = max(0.0, min(1.0, float(last_step_idx) / float(total_steps)))
            else:
                frac1 = 0.0
            pct = 50.0 * frac1
        else:
            # No info – keep a small "spinner" effect.
            # Do NOT over-commit ETA; just bump to 5% after a bit.
            if elapsed > 5.0:
                pct = 5.0
            else:
                pct = 0.0

        ipct = int(max(0, min(99, round(pct))))
        if ipct != last_pct:
            try:
                _progress_set(ipct)
            except Exception:
                pass
            last_pct = ipct

    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("CMD: " + " ".join([str(x) for x in cmd]) + "\n\n")
            lf.flush()
            try:
                _progress_set(0)
            except Exception:
                pass

            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            try:
                if not flash_attention:
                    env["FV_WAN_DISABLE_FLASH_ATTENTION"] = "1"
                else:
                    env.pop("FV_WAN_DISABLE_FLASH_ATTENTION", None)
                if crawl_guard:
                    env["FV_WAN_SHARED_MEM_GUARD"] = "1"
                else:
                    env["FV_WAN_SHARED_MEM_GUARD"] = "0"
            except Exception:
                pass

            p = _sp.Popen(
                cmd,
                cwd=cwd_run,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            try:
                _patch_running_json({'active_pid': int(getattr(p,'pid',0) or 0), 'active_cmd': ' '.join([str(x) for x in cmd])})
            except Exception:
                pass
            q, _t = _start_proc_reader(p)
            was_cancel = False
            while True:
                if _cancel_requested():
                    was_cancel = True
                    _note_cancel("Cancelled by user")
                    try:
                        _kill_process_tree(getattr(p, 'pid', 0) or 0)
                    except Exception:
                        pass
                    break
                try:
                    item = q.get_nowait()
                except _queue.Empty:
                    item = None
                if item is None:
                    if p.poll() is not None:
                        break
                    _update_progress_from_state()
                    _time.sleep(0.5)
                    continue
                if item is _EOF:
                    break
                line = item
                if not line:
                    continue

                try:
                    lf.write(line)
                except Exception:
                    pass
                try:
                    print("[WAN22]", line, end="")
                except Exception:
                    pass

                # Try to parse "step X/Y" style logs.
                try:
                    m = step_re.search(line)
                except Exception:
                    m = None
                if m:
                    try:
                        cur = int(m.group(1))
                        tot = int(m.group(2))
                        if tot > 0:
                            if total_steps is None or total_steps != tot:
                                total_steps = tot
                            now = _time.time()
                            if first_step_ts is None:
                                first_step_ts = now
                            if cur > last_step_idx:
                                # Update step-rate estimate from timing.
                                if last_step_idx > 0:
                                    dt = max(0.01, now - step_ts)  # step_ts set below
                                    step_count = max(1, cur - last_step_idx)
                                    sec_per_step = dt / float(step_count)
                                    if est_step_sec is None:
                                        est_step_sec = sec_per_step
                                    else:
                                        # Smooth the estimate.
                                        est_step_sec = (est_step_sec * 0.7) + (sec_per_step * 0.3)
                                last_step_idx = cur
                                step_ts = now
                    except Exception:
                        pass

                # Update progress view
                _update_progress_from_state()

            code = p.wait()
            if was_cancel: code = 130
            try:
                _progress_set(100)
            except Exception:
                pass
            lf.write(f"\nEXIT CODE: {code}\n")
    except Exception:
        # As a fallback, run without streaming/progress.
        code = run(cmd)

    if code == 0:
        if save_path.exists():
            try:
                job["produced"] = str(save_path)
            except Exception:
                pass
        else:
            _mark_error(job, f"Wan2.2 finished but output file is missing: {save_path}")
            return 2
    return code



def qwen2511_image_edit(job, cfg, mani):
    """Run a Qwen2511 (stable-diffusion.cpp sd-cli) image edit job via the queue.

    This builds an sd-cli command using helpers.qwen2511.build_sdcli_cmd and executes it
    through tools_ffmpeg so we reuse streaming + cancel + progress handling.
    """
    import os
    import time as _time
    from pathlib import Path as _P

    args = job.get("args", {}) or {}

    # Resolve output dir
    try:
        out_dir = job.get("out_dir") or str((ROOT / "output" / "edits" / "qwen_2511"))
    except Exception:
        out_dir = str(_P(".").resolve() / "output" / "edits" / "qwen_2511")
    try:
        _P(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Input image
    init_img = args.get("init_img") or job.get("input") or ""
    init_img = str(init_img).strip()
    if not init_img or not _P(init_img).is_file():
        _mark_error(job, "Qwen2511: input image missing or not found.")
        return 2

    # Mask
    mask_img = str(args.get("mask_img") or args.get("mask") or "").strip()
    invert_mask = bool(args.get("invert_mask") or False)

    # Invert mask in the worker (no Qt dependency)
    if mask_img and invert_mask and _P(mask_img).is_file():
        try:
            from PIL import Image, ImageOps
            im = Image.open(mask_img)
            # force grayscale
            im = im.convert("L")
            im = ImageOps.invert(im)
            tmp = _P(out_dir) / f"_mask_inverted_{job.get('id','job')}_{int(_time.time())}.png"
            im.save(tmp)
            mask_img = str(tmp)
            try:
                args["mask_img_inverted"] = mask_img
            except Exception:
                pass
        except Exception:
            # If inversion fails, continue with original mask
            pass

    # Reference images
    ref_images = []
    try:
        ref_images = args.get("ref_images") or []
        if isinstance(ref_images, str):
            ref_images = [ref_images]
        ref_images = [str(x).strip() for x in ref_images if str(x).strip()]
    except Exception:
        ref_images = []
    # allow legacy keys
    for k in ("ref_img_2", "ref_img_3", "ref_img_4"):
        try:
            p = str(args.get(k) or "").strip()
        except Exception:
            p = ""
        if p:
            ref_images.append(p)

    # sd-cli path
    sdcli_path = str(args.get("sdcli_path") or args.get("sdcli") or "").strip()
    if not sdcli_path:
        try:
            sdcli_path = str((ROOT / ".qwen2512" / "bin" / "sd-cli.exe"))
        except Exception:
            sdcli_path = "sd-cli.exe"

    # Output file
    out_file = str(args.get("out_file") or args.get("outfile") or "").strip()
    if not out_file:
        out_file = str(_P(out_dir) / f"qwen_image_edit_2511_{int(_time.time())}.png")

    # Models
    unet_path = str(args.get("unet_path") or "").strip()
    llm_path = str(args.get("llm_path") or "").strip()
    mmproj_path = str(args.get("mmproj_path") or "").strip()
    vae_path = str(args.get("vae_path") or "").strip()

    # Prompts / settings
    prompt = str(args.get("prompt") or "").strip()
    negative = str(args.get("negative") or "").strip()
    steps = int(args.get("steps") or 28)
    cfg_scale = float(args.get("cfg") or 4.5)
    seed = int(args.get("seed") if args.get("seed") is not None else -1)
    width = int(args.get("width") or 1024)
    height = int(args.get("height") or 576)
    strength = float(args.get("strength") or 1.0)
    sampling_method = str(args.get("sampling_method") or "euler_a")
    shift = float(args.get("shift") or 12.5)

    # Options
    use_increase_ref_index = bool(args.get("use_increase_ref_index") or False)
    disable_auto_resize_ref_images = bool(args.get("disable_auto_resize_ref_images") or False)

    use_vae_tiling = bool(args.get("use_vae_tiling") or False)
    vae_tile_size = str(args.get("vae_tile_size") or "256x256").strip()
    vae_tile_overlap = float(args.get("vae_tile_overlap") or 0.50)
    use_offload = bool(args.get("use_offload") or False)
    use_mmap = bool(args.get("use_mmap") or False)
    use_vae_on_cpu = bool(args.get("use_vae_on_cpu") or False)
    use_clip_on_cpu = bool(args.get("use_clip_on_cpu") or False)
    use_diffusion_fa = bool(args.get("use_diffusion_fa") or False)

    # LoRA
    lora_dir = str(args.get("lora_dir") or "").strip()
    if not lora_dir:
        try:
            lora_dir = str((ROOT / "models" / "lora" / "qwen2511"))
        except Exception:
            lora_dir = ""
    lora_name = str(args.get("lora_name") or "").strip()
    try:
        lora_strength = float(args.get("lora_strength") if args.get("lora_strength") is not None else 1.0)
    except Exception:
        lora_strength = 1.0

    # Build command
    try:
        try:
            from helpers.qwen2511 import detect_sdcli_caps, build_sdcli_cmd
        except Exception:
            from qwen2511 import detect_sdcli_caps, build_sdcli_cmd
        caps = detect_sdcli_caps(sdcli_path)
        cmd = build_sdcli_cmd(
            sdcli_path=sdcli_path,
            caps=caps,
            init_img=init_img,
            mask_path=mask_img,
            ref_images=ref_images,
            use_increase_ref_index=use_increase_ref_index,
            disable_auto_resize_ref_images=disable_auto_resize_ref_images,
            prompt=prompt,
            negative=negative,
            unet_path=unet_path,
            llm_path=llm_path,
            mmproj_path=mmproj_path,
            vae_path=vae_path,
            steps=steps,
            cfg=cfg_scale,
            seed=seed,
            width=width,
            height=height,
            strength=strength,
            sampling_method=sampling_method,
            shift=shift,
            out_file=out_file,
            lora_model_dir=lora_dir,
            lora_name=lora_name,
            lora_strength=lora_strength,
            use_vae_tiling=use_vae_tiling,
            vae_tile_size=vae_tile_size,
            vae_tile_overlap=vae_tile_overlap,
            use_offload=use_offload,
            use_mmap=use_mmap,
            use_vae_on_cpu=use_vae_on_cpu,
            use_clip_on_cpu=use_clip_on_cpu,
            use_diffusion_fa=use_diffusion_fa,
        )
    except Exception as e:
        _mark_error(job, "Qwen2511: failed to build sd-cli command: " + str(e))
        return 2

    # Populate args for tools_ffmpeg executor
    try:
        label = args.get("label") or ("Qwen2511: " + (prompt.replace("\n", " ").strip()[:80] if prompt else "image edit"))
        args["label"] = label
        args["cmd"] = cmd
        args["cwd"] = str(ROOT)
        args["outfile"] = out_file
        args["out_file"] = out_file
        job["args"] = args
    except Exception:
        pass

    # Metadata for UI
    try:
        job["backend"] = "qwen2511"
    except Exception:
        pass
    try:
        if unet_path:
            job["model"] = os.path.basename(unet_path)
    except Exception:
        pass

    return tools_ffmpeg(job, cfg, mani)





def flux_klein_image_edit(job, cfg, mani):
    """Run a Flux Klein GGUF image edit job via the queue using sd-cli."""
    import os
    import re
    import time as _time
    from pathlib import Path as _P

    args = job.get('args', {}) or {}

    try:
        out_dir = job.get('out_dir') or str((ROOT / 'output' / 'edits' / 'flux_klein'))
    except Exception:
        out_dir = str(_P('.').resolve() / 'output' / 'edits' / 'flux_klein')
    try:
        _P(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _rel_lora_name(lora_path: str, lora_model_dir: str) -> str:
        if not lora_path:
            return ''
        lp = _P(lora_path)
        try:
            base = _P(lora_model_dir) if lora_model_dir else lp.parent
            rel = lp.relative_to(base)
        except Exception:
            rel = _P(lp.name)
        return rel.with_suffix('').as_posix().strip()

    def _append_lora_tag(prompt: str, lora_name: str, strength: float) -> str:
        if not lora_name:
            return prompt
        tag = f'<lora:{lora_name}:{strength:g}>'
        if tag in (prompt or ''):
            return prompt
        if re.search(r'<\s*lora\s*:[^>]+>', prompt or '', flags=re.IGNORECASE):
            return prompt
        prompt = (prompt or '').rstrip()
        return f'{prompt} {tag}'.strip()

    sdcli_path = str(args.get('sdcli_path') or '').strip()
    diffusion_model = str(args.get('diffusion_model') or '').strip()
    llm_model = str(args.get('llm_model') or '').strip()
    vae_file = str(args.get('vae_file') or '').strip()
    lora_file = str(args.get('lora_file') or '').strip()

    prompt = str(args.get('prompt') or '').strip()
    negative = str(args.get('negative') or '').strip()
    ref_images = args.get('ref_images') or []
    if isinstance(ref_images, str):
        ref_images = [ref_images]
    ref_images = [str(x).strip() for x in ref_images if str(x).strip()]

    width = int(args.get('width') or 1024)
    height = int(args.get('height') or 1024)
    steps = int(args.get('steps') or 4)
    cfg_scale = float(args.get('cfg_scale') if args.get('cfg_scale') is not None else 1.0)
    sampling_method = str(args.get('sampling_method') or 'euler').strip() or 'euler'
    random_seed = bool(args.get('random_seed') if args.get('random_seed') is not None else True)
    seed = -1 if random_seed else int(args.get('seed') if args.get('seed') is not None else 0)
    diffusion_fa = bool(args.get('diffusion_fa') if args.get('diffusion_fa') is not None else True)
    offload_to_cpu = bool(args.get('offload_to_cpu') or False)
    vae_tiling = bool(args.get('vae_tiling') or False)

    try:
        lora_strength = float(args.get('lora_strength') if args.get('lora_strength') is not None else 1.0)
    except Exception:
        lora_strength = 1.0

    out_file = str(args.get('out_file') or args.get('outfile') or '').strip()
    if not out_file:
        out_file = str(_P(out_dir) / f"klein_{_time.strftime('%Y%m%d_%H%M%S')}.png")

    if not sdcli_path or not _P(sdcli_path).is_file():
        _mark_error(job, 'Flux Klein: sd-cli missing or not found.')
        return 2
    if not diffusion_model or not _P(diffusion_model).is_file():
        _mark_error(job, 'Flux Klein: diffusion model missing or not found.')
        return 2
    if not llm_model or not _P(llm_model).is_file():
        _mark_error(job, 'Flux Klein: text encoder missing or not found.')
        return 2
    if not vae_file or not _P(vae_file).is_file():
        _mark_error(job, 'Flux Klein: VAE missing or not found.')
        return 2
    if lora_file and not _P(lora_file).is_file():
        _mark_error(job, 'Flux Klein: LoRA file not found.')
        return 2
    if not prompt:
        _mark_error(job, 'Flux Klein: prompt is empty.')
        return 2
    for rp in ref_images:
        if not _P(rp).is_file():
            _mark_error(job, 'Flux Klein: reference image not found: ' + rp)
            return 2

    cmd = [sdcli_path]
    cmd += ['--diffusion-model', diffusion_model]
    cmd += ['--vae', vae_file]
    cmd += ['--llm', llm_model]

    prompt_for_cmd = prompt
    if lora_file:
        lora_dir = str(_P(lora_file).parent)
        lora_name = _rel_lora_name(lora_file, lora_dir)
        prompt_for_cmd = _append_lora_tag(prompt_for_cmd, lora_name, lora_strength)
        cmd += ['--lora-model-dir', lora_dir]

    cmd += ['-p', prompt_for_cmd]
    if negative:
        cmd += ['-n', negative]
    for r in ref_images:
        cmd += ['-r', r]
    cmd += ['-W', str(width), '-H', str(height)]
    cmd += ['--steps', str(steps)]
    cmd += ['--cfg-scale', str(cfg_scale)]
    cmd += ['--sampling-method', sampling_method]
    cmd += ['-s', str(seed)]
    if diffusion_fa:
        cmd += ['--diffusion-fa']
    if offload_to_cpu:
        cmd += ['--offload-to-cpu']
    if vae_tiling:
        cmd += ['--vae-tiling']
    cmd += ['-o', str(out_file)]
    cmd += ['-v']

    try:
        label = args.get('label') or ('Flux Klein: ' + (prompt.replace('\n', ' ').strip()[:80] if prompt else 'image edit'))
        args['label'] = label
        args['cmd'] = cmd
        args['cwd'] = str(_P(out_dir))
        args['outfile'] = out_file
        args['out_file'] = out_file
        job['args'] = args
        job['backend'] = 'flux_klein'
        job['model'] = os.path.basename(diffusion_model) if diffusion_model else 'Flux Klein'
    except Exception:
        pass

    return tools_ffmpeg(job, cfg, mani)
def firered_image_edit(job, cfg, mani):
    """Run a FireRed image edit job via the queue using sd-cli."""
    import os
    from pathlib import Path as _P

    args = job.get('args', {}) or {}

    try:
        out_dir = job.get('out_dir') or str((ROOT / 'output' / 'edits' / 'firered'))
    except Exception:
        out_dir = str(_P('.').resolve() / 'output' / 'edits' / 'firered')
    try:
        _P(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    sdcli_path = str(args.get('sdcli_path') or '').strip()
    model_path = str(args.get('model_path') or '').strip()
    vae_path = str(args.get('vae_path') or '').strip()
    llm_path = str(args.get('llm_path') or '').strip()
    lora_path = str(args.get('lora_path') or '').strip()
    prompt = str(args.get('prompt') or '').strip()
    negative = str(args.get('negative') or '').strip()
    images = [str(x).strip() for x in (args.get('images') or []) if str(x).strip()]
    width = int(args.get('width') or 1024)
    height = int(args.get('height') or 1024)
    steps = int(args.get('steps') or 8)
    cfg_scale = int(args.get('cfg_scale') or 4)
    strength = float(args.get('strength') or 0.75)
    seed = str(args.get('seed') or '-1').strip() or '-1'
    sampler = str(args.get('sampler') or 'euler').strip() or 'euler'
    batch = int(args.get('batch') or 1)
    offload_cpu = bool(args.get('offload_cpu'))
    flash_attn = bool(args.get('flash_attn'))
    vae_tiling = bool(args.get('vae_tiling'))
    verbose = bool(args.get('verbose', True))
    out_file = str(args.get('out_file') or args.get('outfile') or '').strip()

    if not out_file:
        out_file = str(_P(out_dir) / 'firered_output.png')

    if not sdcli_path or not _P(sdcli_path).is_file():
        _mark_error(job, 'FireRed: sd-cli missing or not found.')
        return 2
    if not model_path or not _P(model_path).is_file():
        _mark_error(job, 'FireRed: diffusion model missing or not found.')
        return 2
    if not vae_path or not _P(vae_path).is_file():
        _mark_error(job, 'FireRed: VAE missing or not found.')
        return 2
    if not llm_path or not _P(llm_path).is_file():
        _mark_error(job, 'FireRed: LLM missing or not found.')
        return 2
    if lora_path and not _P(lora_path).is_file():
        _mark_error(job, 'FireRed: LoRA file not found.')
        return 2
    if not prompt:
        _mark_error(job, 'FireRed: prompt is empty.')
        return 2
    if not images:
        _mark_error(job, 'FireRed: no input images were provided.')
        return 2
    for rp in images:
        if not _P(rp).is_file():
            _mark_error(job, 'FireRed: input image not found: ' + rp)
            return 2

    cmd = [sdcli_path]
    cmd += ['--diffusion-model', model_path]
    cmd += ['--vae', vae_path]
    cmd += ['--llm', llm_path]
    cmd += ['-p', prompt]
    cmd += ['-W', str(width), '-H', str(height)]
    cmd += ['--steps', str(steps)]
    cmd += ['--cfg-scale', str(cfg_scale)]
    cmd += ['--strength', str(strength)]
    cmd += ['--sampling-method', sampler]
    cmd += ['-s', seed]
    cmd += ['-o', str(out_file)]
    for r in images:
        cmd += ['-r', r]
    if lora_path:
        cmd += ['--lora-model-dir', str(_P(lora_path).parent)]
    if negative:
        cmd += ['-n', negative]
    if offload_cpu:
        cmd += ['--offload-to-cpu']
    if flash_attn:
        cmd += ['--diffusion-fa']
    if vae_tiling:
        cmd += ['--vae-tiling']
    if verbose:
        cmd += ['-v']
    if batch > 1:
        cmd += ['-b', str(batch)]

    try:
        label = args.get('label') or ('FireRed: ' + (prompt.replace('\n', ' ').strip()[:80] if prompt else 'image edit'))
        args['label'] = label
        args['cmd'] = cmd
        args['cwd'] = str(_P(sdcli_path).parent)
        args['outfile'] = out_file
        args['out_file'] = out_file
        job['args'] = args
        job['backend'] = 'firered'
        job['model'] = os.path.basename(model_path) if model_path else 'FireRed'
    except Exception:
        pass

    return tools_ffmpeg(job, cfg, mani)


def heartmula_generate(job, cfg, mani):
    """Run a HeartMuLa music generation job via the queue.

    The UI enqueues this as type=heartmula_generate with args.cmd built
    to call the upstream heartlib example script. We execute through
    tools_ffmpeg so we reuse streaming logs + cancel support.
    """
    try:
        from pathlib import Path as _P
        import time as _time
        import os as _os
    except Exception:
        return 1

    args = job.get("args", {}) or {}

    # Output folder fallback
    try:
        out_dir = job.get("out_dir") or str((_P(".").resolve() / "output" / "music" / "heartmula"))
    except Exception:
        out_dir = str(_P("output") / "music" / "heartmula")
    try:
        _P(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Label + metadata
    try:
        tags = str(args.get("tags") or "").strip()
    except Exception:
        tags = ""
    label = args.get("label") or ("HeartMuLa: " + (tags[:60] if tags else "music"))
    args["label"] = label
    job["title"] = label
    try:
        job["backend"] = "heartmula"
    except Exception:
        pass
    try:
        job["model"] = str(args.get("version") or "3B")
    except Exception:
        pass

    # Ensure cmd exists
    cmd = args.get("cmd") or args.get("ffmpeg_cmd") or job.get("cmd")
    if not cmd:
        _mark_error(job, "HeartMuLa job missing args.cmd")
        return 2

    # Expected outfile
    outfile = args.get("outfile") or args.get("save_path") or None
    if not outfile:
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        outfile = str(_P(out_dir) / f"heartmula_{stamp}.mp3")
    args["outfile"] = str(outfile)
    args["cwd"] = args.get("cwd") or str(_P(".").resolve())
    # Make bundled binaries discoverable
    if not args.get("prepend_path"):
        try:
            args["prepend_path"] = str((_P(args["cwd"]) / "presets" / "bin").resolve())
        except Exception:
            pass
    args["cmd"] = cmd
    job["args"] = args

    return tools_ffmpeg(job, cfg, mani)

def planner_lock(job: dict, cfg: dict, mani: dict):
    """
    Queue lock job used by Planner to reserve the worker while the Planner runs locally in the app.
    The Planner enqueues this job, then runs its pipeline locally, and finally writes args.done_flag.

    args:
      done_flag: path to a flag file that will be created by the Planner when finished
      scan_dir: directory to scan for newest .mp4 output (optional)
      scan_ext: extension to scan (default .mp4)
    """
    args = job.get("args") or {}
    done_flag = str(args.get("done_flag") or "").strip()
    if not done_flag:
        _mark_error(job, "planner_lock missing args.done_flag")
        return 2
    scan_dir = str(args.get("scan_dir") or job.get("out_dir") or "").strip()
    scan_ext = str(args.get("scan_ext") or ".mp4").strip().lower()
    tail_lines = int(args.get("log_tail_lines") or 120)

    # Wait loop
    start = time.time()
    while True:
        # Cancel marker support
        try:
            mk = Path(RUNNING_JSON_FILE).with_suffix(Path(RUNNING_JSON_FILE).suffix + ".cancel")
            if mk.exists():
                _mark_error(job, "Cancelled")
                return 130
        except Exception:
            pass
        if Path(done_flag).exists():
            break
        # update a tiny heartbeat/log tail
        try:
            job["waiting_for"] = done_flag
            job["elapsed_sec"] = int(time.time() - start)
            Path(RUNNING_JSON_FILE).write_text(json.dumps(job, indent=2), encoding="utf-8")
        except Exception:
            pass
        time.sleep(0.5)

    # Planner finished; try to locate final video
    final_video = ""
    newest = None
    newest_mtime = -1.0
    try:
        if scan_dir and Path(scan_dir).exists():
            for p in Path(scan_dir).rglob(f"*{scan_ext}"):
                try:
                    mt = p.stat().st_mtime
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = p
                except Exception:
                    continue
    except Exception:
        newest = None

    if newest and newest.exists():
        final_video = str(newest)
        job["produced"] = final_video
        job["files"] = [final_video]
    return 0





def planner_generate(job: dict, cfg: dict, mani: dict):
    """Run a full Planner pipeline as a real FrameVision queue job.

    The old Planner queue path only enqueued a waiting/lock row while the Planner
    UI did the real work locally.  This worker entry imports helpers/planner.py,
    rebuilds the PlannerJob payload, and runs PipelineWorker inside the worker
    process so the queue owns the actual job.
    """
    import copy as _copy
    import importlib.util as _importlib_util
    import threading as _threading_local
    import traceback as _traceback
    from pathlib import Path as _P

    args = job.get('args') or {}
    planner_payload = args.get('planner_job') or args.get('job_config') or args.get('job') or {}
    if not isinstance(planner_payload, dict):
        _mark_error(job, 'Planner queue job missing planner_job payload.')
        return 2

    out_dir = str(args.get('out_dir') or job.get('out_dir') or '').strip()
    if not out_dir:
        _mark_error(job, 'Planner queue job missing output directory.')
        return 2
    try:
        _P(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    label = str(args.get('label') or args.get('title') or 'Planner').strip() or 'Planner'
    try:
        job['title'] = label
        job['label'] = label
        job['backend'] = 'planner'
        job['model'] = str(((planner_payload.get('encoding') or {}) if isinstance(planner_payload.get('encoding'), dict) else {}).get('video_model') or 'planner')
        _patch_running_json({'title': label, 'label': label, 'stage': 'Starting Planner', 'backend': 'planner'})
    except Exception:
        pass

    root = _P(__file__).resolve().parents[1] if _P(__file__).resolve().parent.name.lower() == 'helpers' else _P('.').resolve()
    planner_path = (_P(__file__).resolve().parent / 'planner.py').resolve()
    if not planner_path.exists():
        planner_path = (root / 'helpers' / 'planner.py').resolve()
    if not planner_path.exists():
        _mark_error(job, f'Planner helper not found: {planner_path}')
        return 2

    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    except Exception:
        pass

    log_dir = root / 'logs'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_file = log_dir / f"planner_queue_{str(job.get('id') or planner_payload.get('job_id') or int(time.time()))}.log"

    def _append_log(msg):
        try:
            s = str(msg)
        except Exception:
            s = repr(msg)
        try:
            print('[planner queue] ' + s, flush=True)
        except Exception:
            pass
        try:
            with open(log_file, 'a', encoding='utf-8', errors='replace') as f:
                f.write(s + '\n')
        except Exception:
            pass
        try:
            _patch_running_json({'last_log': s[-1000:]})
        except Exception:
            pass

    try:
        spec = _importlib_util.spec_from_file_location('helpers.planner', str(planner_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Could not load planner module spec: {planner_path}')
        mod = _importlib_util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        _mark_error(job, f'Could not import Planner helper: {e}')
        try:
            _append_log(_traceback.format_exc())
        except Exception:
            pass
        return 2

    try:
        fields = getattr(getattr(mod, 'PlannerJob'), '__dataclass_fields__', {}) or {}
        clean = {k: _copy.deepcopy(planner_payload.get(k)) for k in fields.keys() if k in planner_payload}
        missing = [k for k in fields.keys() if k not in clean]
        if missing:
            raise RuntimeError('Planner job payload missing fields: ' + ', '.join(missing[:8]))
        planner_job = mod.PlannerJob(**clean)
    except Exception as e:
        _mark_error(job, f'Could not rebuild PlannerJob: {e}')
        return 2

    # Real queued jobs cannot wait for UI review dialogs. Disable interactive gates
    # even if the UI setting was on when the job was created.
    try:
        enc = getattr(planner_job, 'encoding', None)
        if not isinstance(enc, dict):
            enc = {}
        enc = dict(enc)
        enc['framevision_queue_worker_run'] = True
        enc['allow_edit_while_running'] = False
        enc.pop('planner_resume_review_existing', None)
        enc.pop('planner_resume_pre_review_images_done', None)
        if args.get('title'):
            enc.setdefault('title', str(args.get('title')))
        if args.get('slug'):
            enc.setdefault('slug', str(args.get('slug')))
        planner_job.encoding = enc
    except Exception:
        pass

    try:
        payload_copy = dict(planner_payload)
        payload_copy['encoding'] = getattr(planner_job, 'encoding', payload_copy.get('encoding'))
        (_P(out_dir) / 'planner_queued_job.json').write_text(json.dumps(payload_copy, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass

    result_box = {'result': None, 'error': None, 'assets': []}
    done = {'value': False}

    def _stage(s):
        try:
            ss = str(s)
        except Exception:
            ss = 'Planner'
        _append_log('[STAGE] ' + ss)
        try:
            _patch_running_json({'stage': ss, 'status': ss})
        except Exception:
            pass

    def _progress(p):
        try:
            _progress_set(int(p))
        except Exception:
            pass

    def _finished(res):
        result_box['result'] = res if isinstance(res, dict) else {'result': res}

    def _failed(err):
        result_box['error'] = str(err)

    def _asset(path):
        try:
            p = str(path or '').strip()
            if p:
                result_box.setdefault('assets', []).append(p)
                _patch_running_json({'files': result_box.get('assets', [])})
        except Exception:
            pass

    class _Sig:
        def __init__(self, fn=None):
            self._fn = fn
        def emit(self, *a, **k):
            if self._fn:
                return self._fn(*a, **k)
            return None

    class _Signals:
        pass

    try:
        runner = mod.PipelineWorker(planner_job, out_dir)
        sigs = _Signals()
        sigs.log = _Sig(lambda s: _append_log(s))
        sigs.stage = _Sig(_stage)
        sigs.progress = _Sig(_progress)
        sigs.finished = _Sig(_finished)
        sigs.failed = _Sig(_failed)
        sigs.asset_created = _Sig(_asset)
        # Safety: if an old/resume flag still reaches a review gate, auto-continue
        # instead of hanging the worker forever.
        sigs.request_image_review = _Sig(lambda payload: runner.post_review_command({'type': 'continue'}))
        sigs.request_video_review = _Sig(lambda payload: runner.post_review_command({'type': 'continue'}))
        sigs.image_regen_started = _Sig()
        sigs.image_regen_done = _Sig()
        sigs.image_regen_failed = _Sig()
        sigs.clip_regen_started = _Sig()
        sigs.clip_regen_done = _Sig()
        sigs.clip_regen_failed = _Sig()
        runner.signals = sigs
    except Exception as e:
        _mark_error(job, f'Could not start Planner pipeline runner: {e}')
        return 2

    def _cancel_watch():
        while not done.get('value'):
            try:
                if _cancel_requested():
                    _append_log('Cancel requested from FrameVision queue.')
                    try:
                        runner.request_stop()
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            time.sleep(0.5)

    try:
        th = _threading_local.Thread(target=_cancel_watch, daemon=True)
        th.start()
    except Exception:
        pass

    try:
        _progress_set(0)
        _append_log('Starting real Planner queue job.')
        runner.run()
    except BaseException as e:
        result_box['error'] = str(e)
        try:
            _append_log(_traceback.format_exc())
        except Exception:
            pass
    finally:
        done['value'] = True

    if result_box.get('error'):
        msg = str(result_box.get('error') or 'Planner failed')
        _mark_error(job, msg)
        if 'cancel' in msg.lower() or _cancel_requested():
            return 130
        return 2

    final_video = ''
    try:
        res = result_box.get('result') or {}
        final_video = str(res.get('final_video') or '') if isinstance(res, dict) else ''
    except Exception:
        final_video = ''
    if not final_video:
        try:
            mp = _P(out_dir) / 'manifest.json'
            if mp.exists():
                man = json.loads(mp.read_text(encoding='utf-8', errors='replace'))
                final_video = str((man.get('paths') or {}).get('final_video') or man.get('final_video') or '')
        except Exception:
            final_video = ''
    if not final_video:
        try:
            vids = list((_P(out_dir) / 'final').glob('*.mp4')) + list(_P(out_dir).rglob('*final*.mp4'))
            vids = [p for p in vids if p.exists() and p.is_file()]
            if vids:
                vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                final_video = str(vids[0])
        except Exception:
            final_video = ''

    files = []
    try:
        files.extend([str(x) for x in (result_box.get('assets') or []) if x])
    except Exception:
        pass
    if final_video:
        try:
            if final_video not in files:
                files.append(final_video)
            job['produced'] = final_video
        except Exception:
            pass
    if files:
        try:
            job['files'] = files
        except Exception:
            pass
    try:
        _progress_set(100)
        _patch_running_json({'stage': 'Done', 'status': 'Done', 'produced': final_video, 'files': files})
    except Exception:
        pass
    _append_log('Planner queue job finished.' + (f' Final: {final_video}' if final_video else ''))
    return 0

def hiar_generate(job, cfg, mani):
    """Run a queued HiAR (Wan 2.1 based) video generation job."""
    import os
    import re
    import json as _json
    import tempfile as _tempfile
    import shutil as _shutil
    import time as _time
    from datetime import datetime as _dt
    from pathlib import Path as _P

    args = job.get('args', {}) or {}

    repo_root = _P(str(args.get('repo_root') or '').strip())
    python_path = _P(str(args.get('python_path') or '').strip())
    config_path = _P(str(args.get('config_path') or '').strip())
    checkpoint_path = _P(str(args.get('checkpoint_path') or '').strip())
    output_folder = _P(str(args.get('output_folder') or (ROOT / 'output' / 'hiar')).strip())
    prompt_file = str(args.get('prompt_file') or '').strip()
    prompt_text = str(args.get('prompt_text') or '').strip()
    extended_prompt_path = str(args.get('extended_prompt_path') or '').strip()
    negative_prompt = str(args.get('negative_prompt') or '').strip()
    guidance = float(args.get('guidance_scale') if args.get('guidance_scale') is not None else 3.0)
    frames = int(args.get('num_output_frames') or 66)
    seed = int(args.get('seed') or 0)
    samples = int(args.get('num_samples') or 1)
    inference_method = str(args.get('inference_method') or 'timestep_first').strip() or 'timestep_first'
    frame_first_blocks = int(args.get('num_frame_first_blocks') or 1)
    use_ema = bool(args.get('use_ema'))
    save_with_index = bool(args.get('save_with_index', True))

    if not repo_root.exists():
        _mark_error(job, f'HiAR: repo root not found: {repo_root}')
        return 2
    if not (repo_root / 'inference.py').exists():
        _mark_error(job, f'HiAR: inference.py not found in repo root: {repo_root}')
        return 2
    if not python_path.exists():
        _mark_error(job, f'HiAR: Python exe not found: {python_path}')
        return 2
    if not config_path.exists():
        _mark_error(job, f'HiAR: config not found: {config_path}')
        return 2
    if not checkpoint_path.exists():
        _mark_error(job, f'HiAR: checkpoint not found: {checkpoint_path}')
        return 2
    if prompt_file and not _P(prompt_file).exists() and not prompt_text:
        _mark_error(job, f'HiAR: prompt file not found: {prompt_file}')
        return 2
    if not prompt_text and not prompt_file:
        _mark_error(job, 'HiAR: provide prompt text or prompt file.')
        return 2

    output_folder.mkdir(parents=True, exist_ok=True)
    run_stamp = _dt.now().strftime('%Y%m%d_%H%M%S_%f')
    run_temp_output_dir = output_folder / f'_hiar_run_{run_stamp}'
    run_temp_output_dir.mkdir(parents=True, exist_ok=True)
    existing_mp4s = {str(x.resolve()) for x in output_folder.glob('*.mp4')}

    temp_paths = []

    def _replace_yaml_scalar(text: str, key: str, value_literal: str) -> str:
        pattern = rf'^(?P<indent>[ 	]*){re.escape(key)}\s*:\s*(?P<value>.*)$'
        replacement = rf'\g<indent>{key}: {value_literal}'
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            new_text = text.rstrip() + f'\n{key}: {value_literal}\n'
        return new_text

    def _format_yaml_float(value: float) -> str:
        text = f'{value:.2f}'.rstrip('0').rstrip('.')
        if '.' not in text:
            text += '.0'
        return text

    def _prompt_words_for_filename(source: str, max_words: int = 5) -> str:
        source = (source or '').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip().lower()
        cleaned = []
        for part in source.split():
            word = ''.join(ch for ch in part if ch.isalnum())
            if word:
                cleaned.append(word)
            if len(cleaned) >= max_words:
                break
        return '_'.join(cleaned) if cleaned else 'output'

    def _read_prompt_lines_from_source(prompt_text_value: str, prompt_file_value: str) -> list[str]:
        raw = ''
        if prompt_text_value:
            raw = str(prompt_text_value)
        elif prompt_file_value:
            try:
                raw = _P(prompt_file_value).read_text(encoding='utf-8', errors='replace')
            except Exception:
                raw = ''
        lines = []
        for line in str(raw).replace('\r\n', '\n').replace('\r', '\n').split('\n'):
            cleaned = ' '.join(str(line).split()).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _build_output_name_bases(prompt_lines: list[str], num_samples: int, file_count: int) -> list[str]:
        if not prompt_lines:
            return []
        prompt_bases = [_prompt_words_for_filename(line) for line in prompt_lines]
        if file_count == len(prompt_bases):
            return prompt_bases
        if num_samples > 1 and file_count == (len(prompt_bases) * num_samples):
            names = []
            for base in prompt_bases:
                for sample_idx in range(1, num_samples + 1):
                    names.append(f'{base}_{sample_idx:02d}')
            return names
        if len(prompt_bases) == 1 and file_count > 1:
            return [prompt_bases[0]] * file_count
        return []

    try:
        data_path = ''
        if prompt_text:
            fd, temp_prompt = _tempfile.mkstemp(prefix='hiar_prompts_', suffix='.txt', dir=str((BASE / 'temp').resolve()))
            os.close(fd)
            with open(temp_prompt, 'w', encoding='utf-8', newline='\n') as f:
                txt = prompt_text.replace('\r\n', '\n').replace('\r', '\n')
                f.write(txt)
                if not txt.endswith('\n'):
                    f.write('\n')
            data_path = temp_prompt
            temp_paths.append(temp_prompt)
        else:
            data_path = prompt_file

        final_config_path = str(config_path)
        custom_needed = (abs(guidance - 3.0) > 1e-9) or bool(negative_prompt)
        if custom_needed:
            temp_dir = (BASE / 'temp').resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_cfg = _tempfile.mkstemp(prefix='hiar_config_', suffix='.yaml', dir=str(temp_dir))
            os.close(fd)
            original = config_path.read_text(encoding='utf-8')
            updated = _replace_yaml_scalar(original, 'guidance_scale', _format_yaml_float(guidance))
            if negative_prompt:
                updated = _replace_yaml_scalar(updated, 'negative_prompt', _json.dumps(negative_prompt, ensure_ascii=False))
            with open(temp_cfg, 'w', encoding='utf-8', newline='\n') as f:
                f.write(updated)
            final_config_path = temp_cfg
            temp_paths.append(temp_cfg)

        cmd = [
            str(python_path), '-u', str(repo_root / 'inference.py'),
            '--config_path', str(final_config_path),
            '--checkpoint_path', str(checkpoint_path),
            '--data_path', str(data_path),
            '--output_folder', str(run_temp_output_dir),
            '--num_output_frames', str(frames),
            '--seed', str(seed),
            '--num_samples', str(samples),
            '--inference_method', inference_method,
        ]
        if extended_prompt_path:
            cmd += ['--extended_prompt_path', extended_prompt_path]
        if use_ema:
            cmd.append('--use_ema')
        if save_with_index:
            cmd.append('--save_with_index')
        if inference_method == 'hybrid_block0':
            cmd += ['--num_frame_first_blocks', str(frame_first_blocks)]

        env = os.environ.copy()
        repo_root_str = str(repo_root)
        fv_root_str = str(BASE)
        env['PYTHONPATH'] = repo_root_str + (os.pathsep + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
        env['HF_HOME'] = str(BASE / 'models' / 'hiar' / 'hf_cache')
        env['HUGGINGFACE_HUB_CACHE'] = str(BASE / 'models' / 'hiar' / 'hf_cache' / 'hub')
        env['TRANSFORMERS_CACHE'] = str(BASE / 'models' / 'hiar' / 'hf_cache' / 'transformers')
        env['TORCH_HOME'] = str(BASE / 'models' / 'hiar' / 'torch_cache')
        env['XDG_CACHE_HOME'] = str(BASE / 'models' / 'hiar' / 'cache')
        env['FRAMEVISION_ROOT'] = fv_root_str
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['USERNAME'] = env.get('USERNAME', 'FrameVision')
        env['USER'] = env.get('USER', env.get('USERNAME', 'FrameVision'))
        env['USERPROFILE'] = env.get('USERPROFILE', fv_root_str)
        env['HOME'] = env.get('HOME', env.get('USERPROFILE', fv_root_str))
        env['TEMP'] = env.get('TEMP', str(BASE / 'temp'))
        env['TMP'] = env.get('TMP', env.get('TEMP', str(BASE / 'temp')))

        try:
            label = args.get('label') or ('HiAR: ' + _prompt_words_for_filename(prompt_text or prompt_file, max_words=8)[:80])
            args['label'] = label
            args['cmd'] = cmd
            args['cwd'] = str(repo_root)
            args['env'] = env
            args['log_file'] = str(LOGS_DIR / f"hiar_{job.get('id','job')}_{_time.strftime('%Y%m%d_%H%M%S')}.log")
            job['args'] = args
            job['backend'] = 'hiar'
            job['model'] = 'HiAR / Wan 2.1'
        except Exception:
            pass

        code = tools_ffmpeg(job, cfg, mani)

        new_files = []
        try:
            source_dir = run_temp_output_dir
            if source_dir.exists():
                new_files = list(source_dir.glob('*.mp4'))
            if not new_files:
                new_files = [x for x in output_folder.glob('*.mp4') if str(x.resolve()) not in existing_mp4s]
        except Exception:
            new_files = []

        if code != 0 and not new_files:
            return code

        if not new_files:
            _mark_error(job, 'HiAR: process finished but no new mp4 output was found.')
            return 1

        prompt_lines = _read_prompt_lines_from_source(prompt_text, prompt_file)
        name_bases = _build_output_name_bases(prompt_lines, samples, len(new_files))
        fallback_prompt_part = _prompt_words_for_filename((prompt_lines[0] if prompt_lines else (prompt_text or prompt_file)))
        stamp = _dt.now().strftime('%Y%m%d_%H%M%S')
        produced_files = []
        new_files = sorted(new_files, key=lambda x: x.stat().st_mtime if x.exists() else 0)
        for idx, src_path in enumerate(new_files, 1):
            if idx - 1 < len(name_bases):
                prefix = f"hiar_{name_bases[idx - 1]}_{stamp}"
                target_name = f'{prefix}.mp4'
            else:
                prefix = f'hiar_{fallback_prompt_part}_{stamp}'
                if len(new_files) == 1:
                    target_name = f'{prefix}.mp4'
                else:
                    target_name = f'{prefix}_{idx:02d}.mp4'
            target_path = output_folder / target_name
            n = 2
            while target_path.exists():
                stem = target_path.stem
                suff = target_path.suffix
                target_path = output_folder / f'{stem}_{n:02d}{suff}'
                n += 1
            try:
                if src_path.resolve() != target_path.resolve():
                    src_path.replace(target_path)
                produced_files.append(str(target_path))
            except Exception:
                try:
                    _shutil.move(str(src_path), str(target_path))
                    produced_files.append(str(target_path))
                except Exception:
                    produced_files.append(str(src_path))

        try:
            if produced_files:
                job['files'] = produced_files
                job['produced'] = produced_files[-1]
                job.pop('error', None)
                if code != 0:
                    job['warning'] = f'HiAR exited with code {code} but output exists; marking job as done.'
            code = 0
        except Exception:
            pass
        return code
    finally:
        try:
            for tp in temp_paths:
                try:
                    _P(tp).unlink(missing_ok=True)
                except Exception:
                    pass
            if run_temp_output_dir.exists() and not any(run_temp_output_dir.iterdir()):
                run_temp_output_dir.rmdir()
        except Exception:
            pass




def _hidream_runner_supports_option(runner_path: str, option: str) -> bool:
    try:
        return bool(runner_path and option and option in Path(runner_path).read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return False



def ideogram4_generate(job, cfg, mani):
    """Run a queued Ideogram 4 GGUF image generation job using helpers.ideogram4."""
    try:
        args = dict(job.get('args') or {})
    except Exception:
        args = {}

    try:
        from helpers.ideogram4 import GenerateConfig as _IdeoGenerateConfig
        from helpers.ideogram4 import generate_once as _ideo_generate_once
        from helpers.ideogram4 import default_model_dir as _ideo_default_model_dir
        from helpers.ideogram4 import default_gguf_dir as _ideo_default_gguf_dir
        from helpers.ideogram4 import default_sd_cli_path as _ideo_default_sd_cli_path
        from helpers.ideogram4 import default_output_dir as _ideo_default_output_dir
        from helpers.ideogram4 import output_path as _ideo_output_path
    except Exception:
        try:
            from ideogram4 import GenerateConfig as _IdeoGenerateConfig  # type: ignore
            from ideogram4 import generate_once as _ideo_generate_once  # type: ignore
            from ideogram4 import default_model_dir as _ideo_default_model_dir  # type: ignore
            from ideogram4 import default_gguf_dir as _ideo_default_gguf_dir  # type: ignore
            from ideogram4 import default_sd_cli_path as _ideo_default_sd_cli_path  # type: ignore
            from ideogram4 import default_output_dir as _ideo_default_output_dir  # type: ignore
            from ideogram4 import output_path as _ideo_output_path  # type: ignore
        except Exception as exc:
            _mark_error(job, f'Could not import Ideogram 4 helper: {exc}')
            return 2

    prompt = str(args.get('prompt') or '').strip()
    if not prompt:
        _mark_error(job, 'Ideogram 4 prompt is empty.')
        return 2

    try:
        out_dir = Path(str(job.get('out_dir') or args.get('output_dir') or _ideo_default_output_dir())).resolve()
    except Exception:
        out_dir = Path(str(_ideo_default_output_dir())).resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        out_file = str(args.get('out_file') or args.get('outfile') or '')
    except Exception:
        out_file = ''
    out_path = Path(out_file).resolve() if out_file else _ideo_output_path(out_dir, 'ideogram4_gguf')

    try:
        title = str(args.get('label') or ('Ideogram 4 GGUF: ' + prompt.replace('\n', ' ')[:80])).strip()
        job['title'] = title
        job['label'] = title
        job['backend'] = 'ideogram4_gguf'
        job['model'] = str(args.get('gguf_diffusion_file') or 'ideogram4 GGUF')
        job['args'] = args
    except Exception:
        pass

    try:
        _progress_set(2)
        _patch_running_json({'stage': 'Ideogram 4 generating'})
    except Exception:
        pass

    import re as _re
    _ansi = _re.compile(r'\[[0-9;]*[A-Za-z]')
    last_pct = 2

    def _ideo_log(line):
        nonlocal last_pct
        try:
            msg = str(line or '')
            clean = _ansi.sub('', msg)
            print(f'[worker][ideogram4] {clean}')
            matches = _re.findall(r'(\d+)\s*/\s*(\d+)', clean)
            if matches:
                cur, total = matches[-1]
                cur = int(cur); total = int(total)
                if total > 0:
                    pct = max(2, min(98, int(2 + (float(cur) / float(total)) * 96.0)))
                    if pct != last_pct:
                        last_pct = pct
                        try:
                            _progress_set(pct)
                            _patch_running_json({'stage': f'Ideogram 4 generating {cur}/{total}', 'step': int(cur), 'steps_total': int(total)})
                        except Exception:
                            pass
        except Exception:
            pass

    try:
        cfg_obj = _IdeoGenerateConfig(
            prompt=prompt,
            negative=str(args.get('negative') or ''),
            model_dir=_ideo_default_model_dir(),
            output=out_path,
            width=int(args.get('width') or 1024),
            height=int(args.get('height') or 1024),
            steps=int(args.get('steps') or 20),
            guidance=float(args.get('guidance') or 3.5),
            preset='Custom',
            seed=int(args.get('seed') or -1),
            compile_sdnq=False,
            raw_prompt=bool(args.get('raw_prompt', False)),
            text_encoder_cpu_offload=False,
            backend='gguf',
            gguf_dir=Path(str(args.get('gguf_dir') or _ideo_default_gguf_dir())),
            gguf_diffusion_file=str(args.get('gguf_diffusion_file') or ''),
            gguf_unconditional_file=str(args.get('gguf_unconditional_file') or ''),
            gguf_llm_file=str(args.get('gguf_llm_file') or ''),
            gguf_vae_file=str(args.get('gguf_vae_file') or ''),
            sd_cli_path=Path(str(args.get('sd_cli_path') or _ideo_default_sd_cli_path())),
            gguf_max_vram=0.0,
            gguf_stream_layers=bool(args.get('gguf_stream_layers', False)),
        )
        out, used_seed, notes = _ideo_generate_once(cfg_obj, log=_ideo_log)
        files = [str(out)]
        job['files'] = files
        job['produced'] = files[-1]
        job['output'] = files[-1]
        args['seed_used'] = int(used_seed)
        job['args'] = args
        try:
            _progress_set(100)
            _patch_running_json({'stage': 'Ideogram 4 done', 'produced': files[-1]})
        except Exception:
            pass
        try:
            print('[worker][ideogram4] done:', files[-1])
            if notes:
                print('[worker][ideogram4] notes:', str(notes).replace('\n', ' | '))
        except Exception:
            pass
        return 0
    except Exception as exc:
        try:
            import traceback as _traceback
            print('[worker][ideogram4] ERROR:', exc)
            print(_traceback.format_exc())
        except Exception:
            pass
        _mark_error(job, f'Ideogram 4 GGUF failed: {exc}')
        return 1


def lens_turbo_u4_generate(job, cfg, mani):
    """Run a queued Lens Turbo U4 image job using the existing Lens helper backend."""
    try:
        args = dict(job.get('args') or {})
    except Exception:
        args = {}
    try:
        from helpers.lens_turbo_u4_ui import run_lens_worker_subprocess
    except Exception:
        try:
            from lens_turbo_u4_ui import run_lens_worker_subprocess  # type: ignore
        except Exception as exc:
            _mark_error(job, f'Could not import Lens Turbo U4 helper: {exc}')
            return 2

    try:
        out_dir = str(job.get('out_dir') or args.get('output_dir') or (ROOT / 'output' / 'lens_turbo_u4'))
        args['output_dir'] = out_dir
    except Exception:
        pass

    prompt = str(args.get('prompt') or '').strip()
    if not prompt:
        _mark_error(job, 'Lens Turbo U4 prompt is empty.')
        return 2

    try:
        title = str(args.get('label') or ('Lens Turbo U4: ' + prompt.replace('\n', ' ')[:80])).strip()
        job['title'] = title
        job['label'] = title
        job['backend'] = 'lens_turbo_u4'
        job['model'] = str(args.get('repo_id') or 'WaveCut/Lens-Turbo-SDNQ-uint4-static')
        job['args'] = args
    except Exception:
        pass

    try:
        _progress_set(2)
        _patch_running_json({'stage': 'Lens Turbo U4 generating'})
    except Exception:
        pass

    try:
        print('[worker][lens] starting Lens Turbo U4')
        print('[worker][lens] output:', str(args.get('output_dir') or ''))
        print('[worker][lens] prompt:', prompt[:220])
    except Exception:
        pass

    try:
        def _lens_log(line):
            try:
                if line:
                    print(f'[worker][lens] {line}')
            except Exception:
                pass

        paths = run_lens_worker_subprocess(ROOT, args, log_callback=_lens_log)
        files = [str(x) for x in (paths or [])]
        if not files:
            _mark_error(job, 'Lens Turbo U4 produced no images.')
            return 1
        job['files'] = files
        job['produced'] = files[-1]
        try:
            _progress_set(100)
            _patch_running_json({'stage': 'Lens Turbo U4 done', 'produced': files[-1]})
        except Exception:
            pass
        print('[worker][lens] done:', files[-1])
        return 0
    except Exception as exc:
        try:
            import traceback as _traceback
            traceback_txt = _traceback.format_exc()
            print('[worker][lens] ERROR:', exc)
            print(traceback_txt)
        except Exception:
            pass
        _mark_error(job, f'Lens Turbo U4 failed: {exc}')
        return 1

def hidream_generate(job, cfg, mani):
    """Run a HiDream BF16 queued job by invoking run_hidream.py directly."""
    import time as _time
    from pathlib import Path as _P

    try:
        args = job.get('args') or {}
    except Exception:
        args = {}

    model_key = str(args.get('model_key') or 'base').strip() or 'base'
    prompt = str(args.get('prompt') or '').strip()
    settings = args.get('settings') or {}
    refs = args.get('refs') or args.get('ref_images') or []
    if isinstance(refs, str):
        refs = [refs]
    refs = [str(x).strip() for x in refs if str(x).strip()]
    # If an older queued job only stored the first reference as the job input, recover it.
    if not refs:
        try:
            inp_ref = str(job.get('input') or job.get('input_path') or '').strip()
            if inp_ref:
                refs = [inp_ref]
        except Exception:
            pass
    keep_original_aspect = bool(args.get('keep_original_aspect') or False)

    try:
        out_file = str(args.get('output_path') or args.get('out_file') or args.get('outfile') or '').strip()
        if not out_file:
            out_file = str(_P(job.get('out_dir') or (ROOT / 'models' / 'hidream_bf16' / 'results')) / f'hidream_{model_key}_{int(_time.time())}.png')
    except Exception:
        out_file = str(_P(ROOT / 'models' / 'hidream_bf16' / 'results') / f'hidream_{model_key}_{int(_time.time())}.png')

    out_path = _P(out_file)
    out_dir = out_path.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _resolve_hidream_python() -> str:
        # Prefer HiDream's own private env. Keep shared/private alternates only as fallbacks.
        supplied = str(args.get('env_python') or '').strip()
        candidates = []
        if supplied:
            candidates.append(_P(supplied))
        candidates.extend([
            ROOT / 'environments' / '.hidream_dev' / 'python.exe',
            ROOT / 'environments' / '.hidream_bf16' / 'python.exe',
            ROOT / '.hidream_dev' / 'python.exe',
            ROOT / 'environments' / '.images_models' / 'python.exe',
            ROOT / 'environments' / '.image_models' / 'python.exe',
        ])
        for cand in candidates:
            try:
                if cand.exists():
                    return str(cand)
            except Exception:
                pass
        return supplied or str(ROOT / 'environments' / '.hidream_dev' / 'python.exe')

    env_python = _resolve_hidream_python()
    runner = str(args.get('runner') or (ROOT / 'helpers' / 'hidream_cli.py')).strip()
    if not _P(env_python).exists():
        _mark_error(job, f'HiDream env python not found: {env_python}')
        return 2
    if not _P(runner).exists():
        _mark_error(job, f'HiDream runner not found: {runner}')
        return 2
    if not prompt:
        _mark_error(job, 'HiDream prompt is empty.')
        return 2
    for rp in refs:
        if not _P(rp).is_file():
            _mark_error(job, 'HiDream reference image not found: ' + rp)
            return 2

    def _sg(name, default):
        try:
            value = settings.get(name, default)
            return value
        except Exception:
            return default

    cmd = [
        env_python, '-u', runner,
        '--resolution_mode', 'framevision',
        '--model_key', model_key,
        '--width', str(int(_sg('width', 1280))),
        '--height', str(int(_sg('height', 720))),
        '--steps', str(int(_sg('steps', 28))),
        '--guidance_scale', str(float(_sg('guidance_scale', 0.0))),
        '--shift', str(float(_sg('shift', 1.0))),
        '--seed', str(int(_sg('seed', -1))),
        '--scheduler_name', str(_sg('scheduler_name', 'flash')),
        '--timesteps', str(_sg('timesteps', 'none')),
        '--noise_scale_start', str(float(_sg('noise_scale_start', 7.5))),
        '--noise_scale_end', str(float(_sg('noise_scale_end', 7.5))),
        '--noise_clip_std', str(float(_sg('noise_clip_std', 2.5))),
        '--output_image', str(out_path),
        '--prompt', prompt,
    ]

    offload = settings.get('offload_settings', {}) if isinstance(settings.get('offload_settings'), dict) else {}
    try_auto_offload = bool(offload.get('try_auto_cpu_offload', False))
    if _hidream_runner_supports_option(runner, '--device_map'):
        if try_auto_offload:
            try:
                offload_folder = _P(str(offload.get('offload_folder') or (ROOT / 'temp' / 'hidream_offload'))).expanduser()
                offload_folder.mkdir(parents=True, exist_ok=True)
            except Exception:
                offload_folder = _P(ROOT / 'temp' / 'hidream_offload')
            cmd.extend(['--device_map', 'auto', '--offload_folder', str(offload_folder)])
        else:
            cmd.extend(['--device_map', 'cuda'])

    negative_prompt = str(_sg('negative_prompt', '') or '').strip()
    if model_key == 'base' and negative_prompt and _hidream_runner_supports_option(runner, '--negative_prompt'):
        cmd.extend(['--negative_prompt', negative_prompt])

    if refs:
        cmd.extend(['--ref_images', *refs])
    if keep_original_aspect:
        cmd.append('--keep_original_aspect')

    try:
        title = str(args.get('label') or f'HiDream {model_key}').strip()
        job['title'] = title
        job['label'] = title
        args['label'] = title
        args['cmd'] = cmd
        args['outfile'] = str(out_path)
        args['out_file'] = str(out_path)
        job['args'] = args
        job['backend'] = 'hidream'
        job['model'] = model_key
    except Exception:
        pass

    try:
        print('[worker][hidream] mode:', str(args.get('mode') or 'create'), '| refs:', len(refs), '| output:', str(out_path))
        if refs:
            for i, rp in enumerate(refs, start=1):
                print(f'[worker][hidream] ref {i}: {rp}')
        print('[worker][hidream] CMD:', ' '.join([str(x) for x in cmd]))
    except Exception:
        pass
    try:
        _progress_set(2)
        _patch_running_json({'stage': 'HiDream generating'})
    except Exception:
        pass

    # Stream runner output live so the Queue/worker log shows the current steps,
    # and keep the queue progress bar alive instead of jumping from start to end.
    import re as _re
    env = os.environ.copy()
    # Avoid leaking FrameVision's Python import paths into the HiDream environment.
    # The CLI inserts the HiDream repo path itself, so PYTHONPATH/PYTHONHOME are not needed here.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    code = 1
    was_cancel = False
    last_pct = 2
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            cwd=str(_P(runner).parent),
            bufsize=1,
        )
        try:
            _patch_running_json({'active_pid': int(getattr(p, 'pid', 0) or 0), 'active_cmd': ' '.join([str(x) for x in cmd])})
        except Exception:
            pass

        q, _t = _start_proc_reader(p)
        while True:
            if _cancel_requested():
                was_cancel = True
                _note_cancel('Cancelled by user')
                try:
                    _kill_process_tree(getattr(p, 'pid', 0) or 0)
                except Exception:
                    pass
                break
            try:
                item = q.get_nowait()
            except _queue.Empty:
                item = None
            if item is None:
                if p.poll() is not None:
                    break
                time.sleep(0.15)
                continue
            if item is _EOF:
                break
            line = str(item or '')
            stripped = line.rstrip('\r\n')
            if stripped:
                try:
                    print(stripped)
                except Exception:
                    pass
                matches = _re.findall(r'(\d+)\s*/\s*(\d+)', stripped)
                if matches:
                    try:
                        cur, total = map(int, matches[-1])
                        if total > 0:
                            pct = max(2, min(98, int(2 + (float(cur) / float(total)) * 96.0)))
                            if pct != last_pct:
                                last_pct = pct
                                _progress_set(pct)
                            try:
                                _patch_running_json({'stage': f'HiDream generating {cur}/{total}', 'step': int(cur), 'steps_total': int(total)})
                            except Exception:
                                pass
                    except Exception:
                        pass
        try:
            code = p.wait(timeout=1.5) if was_cancel else p.wait()
        except Exception:
            try:
                code = p.wait()
            except Exception:
                code = 1
        if was_cancel:
            code = 130
    except Exception as e:
        _mark_error(job, f'HiDream launch failed: {e}')
        return 2

    if code == 130:
        _note_cancel('Cancelled by user')
        return 130
    if code != 0:
        _mark_error(job, f'HiDream failed (code {code}).')
        return int(code or 1)
    if not out_path.exists():
        _mark_error(job, f'HiDream finished but output file is missing: {out_path}')
        return 1
    try:
        _progress_set(100)
        _patch_running_json({'stage': 'HiDream finished'})
    except Exception:
        pass
    return 0


def qwentts_generate(job, cfg, mani):
    """Run a queued Qwen TTS job by invoking qwentts_ui.py in --worker mode."""
    import subprocess as _subprocess
    from pathlib import Path as _P

    try:
        args = job.get('args') or {}
    except Exception:
        args = {}

    try:
        title = str(args.get('label') or 'Qwen TTS').strip() or 'Qwen TTS'
        job['title'] = title
        job['label'] = title
        args['label'] = title
        job['args'] = args
    except Exception:
        pass

    env_python = str(args.get('env_python') or '').strip()
    ui_script = str(args.get('ui_script') or '').strip()
    mode = str(args.get('mode') or '').strip()
    payload = args.get('payload') or {}

    if not env_python or not _P(env_python).exists():
        _mark_error(job, f'Qwen TTS env python not found: {env_python}')
        return 2
    if not ui_script or not _P(ui_script).exists():
        _mark_error(job, f'Qwen TTS ui script not found: {ui_script}')
        return 2
    if mode not in ('custom', 'clone', 'design'):
        _mark_error(job, f'Qwen TTS mode invalid: {mode}')
        return 2

    try:
        out_dir = _P(job.get('out_dir') or (BASE / 'output' / 'audio' / 'qwen3tts')).resolve()
    except Exception:
        out_dir = _P(BASE / 'output' / 'audio' / 'qwen3tts').resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    result = None
    try:
        cmd = [env_python, '-u', ui_script, '--worker', '--task', 'generate']
        env = os.environ.copy()
        env.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
        proc = _subprocess.Popen(
            cmd,
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            cwd=str(BASE),
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        payload_str = json.dumps({'mode': mode, 'payload': payload}, ensure_ascii=False)
        if proc.stdin:
            proc.stdin.write(payload_str)
            proc.stdin.close()
        if proc.stdout:
            for line in proc.stdout:
                line = (line or '').rstrip('\n')
                if not line:
                    continue
                if line.startswith('__LOG__'):
                    print(line[len('__LOG__'):].lstrip())
                elif line.startswith('__RESULT__'):
                    js = line[len('__RESULT__'):].strip()
                    try:
                        result = json.loads(js)
                    except Exception:
                        result = {'ok': False, 'error': f'Failed to parse result JSON: {js}'}
                else:
                    print(line)
        rc = int(proc.wait() or 0)
    except Exception as e:
        _mark_error(job, f'Qwen TTS launch failed: {e}')
        return 2

    if not isinstance(result, dict):
        result = {'ok': False, 'error': 'No result returned from Qwen TTS worker.'}

    if not result.get('ok', False):
        _mark_error(job, str(result.get('error') or f'Qwen TTS failed (exit {rc}).'))
        return rc or 1

    out_path = str(result.get('out_path') or '').strip()
    if not out_path:
        _mark_error(job, 'Qwen TTS finished but did not report an output WAV path.')
        return 1

    try:
        op = _P(out_path)
        if not op.is_absolute():
            op = (BASE / op).resolve()
        out_path = str(op)
    except Exception:
        pass

    if not _P(out_path).exists():
        _mark_error(job, f'Qwen TTS output not found: {out_path}')
        return 1

    try:
        job['produced'] = out_path
        job['output'] = out_path
        job['files'] = [out_path]
    except Exception:
        pass
    return 0

def handle_job(jpath: Path):
    job = json.loads(jpath.read_text(encoding="utf-8"))
    cfg = load_config(); mani = manifest()
    running = JOBS["running"] / jpath.name; jpath.rename(running)

    # progress sidecar path
    try:
        global PROGRESS_FILE
        global RUNNING_JSON_FILE
        RUNNING_JSON_FILE = str(running)
        PROGRESS_FILE = str(running.with_suffix(".progress.json"))
    except Exception:
        pass

    # Mark started
    try:
        job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job["worker_pid"] = os.getpid()
        job["status"] = "running"
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
        elif t in ('ltx23_generate','ltx23','ltx23_tools'):
            # LTX 2.5 is queued as a normal external command. No special worker
            # launch mode, no detached process, no LTX-specific subprocess tricks.
            code = tools_ffmpeg(job, cfg, mani)
        # SeedVR2: external CLI runner (python inference_cli.py ...)
        # Queue writes type="seedvr2" with args.cmd + args.outfile.
        # Reuse the robust external runner (progress parsing + log capture).
        elif t in ('seedvr2','seedvr2_upscale','seedvr2_video'):
            code = tools_ffmpeg(job, cfg, mani)
        elif t=='rife_interpolate':
            code = rife_interpolate(job, cfg, mani)
        elif t in ('txt2img','txt2img_qwen'):
            code = txt2img_generate(job, cfg, mani)
        elif t in ('lens_turbo_u4','lens_turbo','lens'):
            code = lens_turbo_u4_generate(job, cfg, mani)
        elif t in ('chroma_generate','chroma','spark_chroma'):
            # SPARK.Chroma is queued as an isolated helper command using
            # environments/.images_models, so reuse the generic streaming runner.
            code = tools_ffmpeg(job, cfg, mani)
        elif t in ('ideogram4_generate','ideogram4','ideogram4_gguf'):
            code = ideogram4_generate(job, cfg, mani)
        elif t in ("qwen2511_image_edit","qwen2511_edit","qwen2511"):
            code = qwen2511_image_edit(job, cfg, mani)
        elif t in ("flux_klein_image_edit","flux_klein_edit","flux_klein","klein_image_edit"):
            code = flux_klein_image_edit(job, cfg, mani)
        elif t in ("firered_image_edit","firered_edit","firered"):
            code = firered_image_edit(job, cfg, mani)
        elif t in ("wan22_text2video","wan22_image2video","wan22_ti2v","wan22","wan22_turbo_text2video","wan22_turbo_image2video","wan22_turbo_ti2v","wan22_turbo"):
            code = wan22_generate(job, cfg, mani)
        elif t in ("hiar_generate","hiar","wan21_hiar"):
            code = hiar_generate(job, cfg, mani)
        elif t in ("ace_step_15","ace_step15","ace15"):
            code = ace15_generate(job, cfg, mani)
        elif t in ("ace_text2music","ace_audio2audio","ace","ace_step","ace_music"):
            code = ace_generate(job, cfg, mani)
        elif t in ("heartmula_generate","heartmula","heartmula_music"):
            code = heartmula_generate(job, cfg, mani)
        elif t in ("qwentts_generate","qwen_tts","qwen3tts"):
            code = qwentts_generate(job, cfg, mani)
        elif t in ("hidream_generate","hidream","hidream_bf16"):
            code = hidream_generate(job, cfg, mani)
        elif t in ('planner_generate','planner','planner_pipeline'):
            code = planner_generate(job, cfg, mani)
        elif t in ('planner_lock',):
            code = planner_lock(job, cfg, mani)
        else:
            _mark_error(job, f"Unknown job type: {t}")
            code = 2
        # mark finished
        # Preserve any output paths computed by the job runner (some runners only update the in-memory job dict).
        _out_patch = {}
        try:
            if job.get("produced"):
                _out_patch["produced"] = job.get("produced")
            try:
                _files = job.get("files")
                if isinstance(_files, (list, tuple)) and _files:
                    _out_patch["files"] = list(_files)
                    if not _out_patch.get("produced"):
                        _out_patch["produced"] = _out_patch["files"][-1]
            except Exception:
                pass
            for _k in ("backend", "model"):
                try:
                    if job.get(_k):
                        _out_patch[_k] = job.get(_k)
                except Exception:
                    pass
        except Exception:
            _out_patch = {}

        # Re-read running JSON in case long-running helpers patched fields (progress/cancel/error).
        # Then merge output fields back in so the UI does not need to "guess" by scanning huge output folders.
        try:
            if running.exists():
                _disk = json.loads(running.read_text(encoding="utf-8"))
                try:
                    if _out_patch:
                        # files: prefer the longer list
                        if "files" in _out_patch:
                            if (not isinstance(_disk.get("files"), list)) or (len(_disk.get("files") or []) < len(_out_patch["files"])):
                                _disk["files"] = _out_patch["files"]
                        # produced/backend/model: only fill if missing
                        if _out_patch.get("produced") and not _disk.get("produced"):
                            _disk["produced"] = _out_patch["produced"]
                        for _k in ("backend", "model"):
                            if _out_patch.get(_k) and not _disk.get(_k):
                                _disk[_k] = _out_patch[_k]
                        # final fallback: produced from files
                        if not _disk.get("produced"):
                            try:
                                _f = _disk.get("files") or []
                                if isinstance(_f, list) and _f:
                                    _disk["produced"] = _f[-1]
                            except Exception:
                                pass
                except Exception:
                    pass
                job = _disk
        except Exception:
            pass

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
            try:
                mk = running.with_suffix(running.suffix + ".cancel")
                if mk.exists():
                    mk.unlink()
            except Exception:
                pass
            running.unlink()
        except Exception:
            pass

        # Console end notice (helps when running worker in a terminal)
        try:
            _dur = job.get("duration_sec")
            _typ = job.get("type") or t
            _jid = job.get("id", "?")
            if int(code) == 0:
                print(f"[worker] finished {_typ} job {_jid} in {_fmt_dur_short(_dur)}")
            else:
                print(f"[worker] failed {_typ} job {_jid} in {_fmt_dur_short(_dur)} (code={code})")
            _outp = job.get("produced")
            if not _outp:
                try:
                    _files = job.get("files") or []
                    if _files:
                        _outp = _files[-1]
                except Exception:
                    _outp = None
            if _outp:
                print(f"[worker] output: {_outp}")
        except Exception:
            pass
        return code
    except BaseException as e:
        try:
            _mark_error(job, str(e))
        except Exception:
            pass
        # Do not leave a crashed/errored job stuck in jobs/running when the
        # worker caught the exception and can still recover.
        try:
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job["duration_sec"] = int(time.time()-t0)
            job["status"] = "failed"
            job["error"] = str(e)
            JOBS["failed"].mkdir(parents=True, exist_ok=True)
            (JOBS["failed"] / running.name).write_text(json.dumps(job, indent=2), encoding="utf-8")
            if running.exists():
                running.unlink()
        except Exception:
            pass
        return 1

def main():
    print("FrameVision Worker V2.5 Waiting for jobs in", JOBS["pending"])
    try:
        n = cleanup_stale_running_jobs(quiet=False)
        if n:
            print(f"[worker] recovered {n} stale running job(s) on startup")
    except Exception:
        pass
    while True:
        try:
            HEARTBEAT.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        except Exception:
            pass
        try:
            cleanup_stale_running_jobs(quiet=True)
        except Exception:
            pass
        items = []
        try:
            cand = [p for p in JOBS["pending"].glob("*.json") if p.is_file()]
            for p in cand:
                n = p.name
                if (n.endswith(".progress.json") or n.endswith(".json.progress") or n.endswith(".meta.json") or n.startswith("_")):
                    continue
                items.append(p)
            # Oldest first = FIFO (matches UI + supports reordering via file mtime)
            items.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, p.name))
        except Exception:
            items = []
        if not items:
            time.sleep(1.0)
            continue
        try:
            cfg = load_config()
        except Exception:
            cfg = {}
        if not _can_start_job_under_llm_safety(items[0], cfg):
            time.sleep(2.0)
            continue
        handle_job(items[0])

if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        try:
            print("[worker main] fatal error caught:", e)
        except Exception:
            pass

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
