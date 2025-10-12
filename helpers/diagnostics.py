# helpers/diagnostics.py — diagnostics with timer/handle probe (drop-in)
from __future__ import annotations
import os, sys, time, platform, subprocess, json, threading, traceback, weakref, ctypes
from typing import Optional, Dict
from PySide6 import QtCore
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import QApplication

APP_ORG = "FrameVision"; APP_NAME = "FrameVision"
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "framevision.log")  # keep existing path for compatibility

# ----------------- logging core -----------------
def _ensure_logfile() -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8"): pass
    except Exception: pass

def diag_enabled() -> bool:
    try:
        s = QSettings(APP_ORG, APP_NAME)
        val = s.value("diagnostics_enabled", "true")
        return str(val).lower() in ("1","true","yes","on")
    except Exception:
        return True

def _fmt(*args) -> str:
    parts = []
    for a in args:
        try: parts.append(str(a))
        except Exception:
            try: parts.append(repr(a))
            except Exception: parts.append("<unprintable>")
    return " ".join(parts)

def log(*args) -> None:
    if not diag_enabled(): return
    _ensure_logfile()
    line = f"[FrameVision DIAG] {time.strftime('%Y-%m-%d %H:%M:%S')} " + _fmt(*args)
    try: print(line, flush=True)
    except Exception: pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(line + "\n")
    except Exception: pass

# ----------------- Windows handle helpers -----------------
try:
    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32
    _GetGuiResources = _user32.GetGuiResources
    _GetCurrentProcess = _kernel32.GetCurrentProcess
    def _get_handle_counts():
        h = _GetCurrentProcess()
        gdi = int(_GetGuiResources(h, 0))  # 0=GDI
        usr = int(_GetGuiResources(h, 1))  # 1=USER
        return gdi, usr
except Exception:
    def _get_handle_counts():
        return -1, -1

# ----------------- timer guard (optional) -----------------
_GUARD_INSTALLED = False
_LIVE_TIMERS = weakref.WeakSet()
_TIMER_ORIGINS: Dict[int, str] = {}
_TIMER_LOCK = threading.Lock()

def _origin_from_stack()->str:
    try:
        for fr in traceback.extract_stack(limit=80)[:-2]:
            fn = str(fr.filename).replace("\\", "/")
            if "PySide6" in fn or "site-packages" in fn:
                continue
            return f"{fr.filename}:{fr.lineno} in {fr.name}"
    except Exception:
        pass
    return "<unknown origin>"

def _install_timer_guard(threshold:int=4000):
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    QTimerCls = QtCore.QTimer
    orig_init = QTimerCls.__init__

    def wrapped_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            with _TIMER_LOCK:
                _LIVE_TIMERS.add(self)
                _TIMER_ORIGINS[id(self)] = _origin_from_stack()
                n = len(_LIVE_TIMERS)
                if n % 500 == 0 or n > threshold:
                    counts: Dict[str,int] = {}
                    for t in list(_LIVE_TIMERS):
                        o = _TIMER_ORIGINS.get(id(t), "?")
                        counts[o] = counts.get(o, 0) + 1
                    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    log(f"[timer_guard] Live QTimers={n}  top={top}")
        except Exception as e:
            log("[timer_guard] wrap error:", e)

    QTimerCls.__init__ = wrapped_init  # type: ignore[assignment]
    _GUARD_INSTALLED = True
    log(f"[timer_guard] Installed; threshold={threshold}")

# ----------------- handle logger (optional; no Qt timers used) -----------------
_HANDLE_THREAD = None
_HANDLE_STOP = threading.Event()

def _handle_loop(period_ms:int):
    while not _HANDLE_STOP.is_set():
        g,u = _get_handle_counts()
        log(f"[handles] GDI={g} USER={u}")
        _HANDLE_STOP.wait(max(0.2, period_ms/1000.0))

def _start_handle_logger(period_ms:int=1000):
    global _HANDLE_THREAD
    if _HANDLE_THREAD and _HANDLE_THREAD.is_alive():
        return
    _HANDLE_THREAD = threading.Thread(target=_handle_loop, name="fv_handle_logger", args=(period_ms,), daemon=True)
    _HANDLE_THREAD.start()
    log(f"[handles] logger started @{period_ms}ms")

# ----------------- Qt message hook (stack dump on registerTimer; suppress QPainter spam) -----------------
_prev_qt_handler = None
_logged_qpainter_once = False
def _qt_message_handler(mode, context, message):
    try: m = str(message)
    except Exception: m = message

    # Suppress noisy QPainter spew entirely (configurable to log once)
    if isinstance(m, str) and "QPainter::" in m:
        try:
            s = QSettings(APP_ORG, APP_NAME)
            if str(s.value("diagnostics_log_qpainter","false")).lower() in ("1","true","yes","on"):
                global _logged_qpainter_once
                if not _logged_qpainter_once:
                    log(f"Qt(QPainter suppressed): {m}")
                    _logged_qpainter_once = True
        except Exception: pass
        return

    # Suppress specific harmless messages
    try:
        if isinstance(m, str):
            if "Could not parse stylesheet of object" in m and "FvSettingsContent" in m:
                return
            if "QObject::startTimer: Timers can only be used with threads started with QThread" in m:
                return
    except Exception:
        pass

    # Mirror to previous handler/stdout for normal flow
    try:
        if _prev_qt_handler is not None: _prev_qt_handler(mode, context, message)
        else: 
            try: print(m)
            except Exception: pass
    except Exception:
        pass

    # When timer creation fails, dump stack + handle counts
    try:
        if isinstance(m, str) and ("registerTimer" in m or "Failed to create a timer" in m):
            stack = "".join(traceback.format_stack(limit=80))
            gdi, usr = _get_handle_counts()
            log(f"[QtTimer] registerTimer warning; USER={usr} GDI={gdi}\n{stack}\n{'-'*70}")
    except Exception as e:
        log("registerTimer diagnostic failed:", e)

# Install the message handler ASAP (module import time), but it's cheap
try:
    _prev_qt_handler = QtCore.qInstallMessageHandler(_qt_message_handler)
except Exception:
    pass

# ----------------- helpers -----------------
NOISY_KEYS = {"intro_cached_urls","intro_gallery_queue","dad_jokes_queue","dad_jokes_seen","dad_jokes_used"}
def _summarize_value(v):
    try:
        s = v if isinstance(v, str) else str(v)
        try:
            obj = json.loads(s)
            if isinstance(obj,(list,tuple)): return f"[redacted list: {len(obj)} items]"
            if isinstance(obj,dict): return f"[redacted dict: {len(obj)} keys]"
        except Exception: pass
        if "http" in s or len(s) > 120: return f"[redacted string: len={len(s)}]"
        return "[redacted]"
    except Exception: return "[redacted]"
def _maybe_redact(k,v):
    k=str(k); lk=k.lower()
    return _summarize_value(v) if (k in NOISY_KEYS or any(p in lk for p in ("urls","gallery","jokes"))) else v

def dump_qsettings(prefix: Optional[str]=None) -> None:
    try:
        s = QSettings(APP_ORG, APP_NAME); keys = sorted(list(s.allKeys()))
        log(f"QSettings dump ({len(keys)} keys){' — ' + prefix if prefix else ''}:")
        for k in keys:
            try: v = _maybe_redact(k, s.value(k))
            except Exception: v = "<error>"
            log(f" - {k} = {v!r}")
    except Exception as e: log("QSettings dump failed:", e)

def log_environment() -> None:
    try:
        log(f"Python: {sys.version.split()[0]} @ {sys.executable}")
        log(f"Platform: {platform.platform()}")
        log(f"Qt/PySide6: Qt={QtCore.qVersion()} PySide6={getattr(QtCore,'__version__','unknown')}")
        in_venv = hasattr(sys,"real_prefix") or (getattr(sys,"base_prefix",sys.prefix) != sys.prefix)
        log(f"Virtualenv: {'yes' if in_venv else 'no'} prefix={sys.prefix}")
    except Exception as e: log("Environment log failed:", e)

def log_packages() -> None:
    def _try(name, attr="__version__"):
        try:
            m = __import__(name, fromlist=['*']); return True, getattr(m, attr, "<no __version__>")
        except Exception as e: return False, f"<missing: {e.__class__.__name__}>"
    for mod in ["torch","transformers","onnxruntime_genai","onnxruntime","onnxruntime_directml","huggingface_hub","safetensors"]:
        ok, ver = _try(mod); log(f"Package {mod}: {'OK' if ok else 'MISSING'} {ver}")

def log_ffmpeg() -> None:
    try:
        out = subprocess.run(["ffmpeg","-hide_banner","-version"], capture_output=True, text=True, timeout=5)
        line1 = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "<no output>"
        log("ffmpeg:", line1)
    except FileNotFoundError: log("ffmpeg: not found on PATH")
    except Exception as e: log("ffmpeg: error probing:", e)

# ----------------- installer (post-start dumps + optional probes) -----------------
def _flag_true(x) -> bool:
    return str(x).lower() in ("1","true","yes","on")

def install_all(main_window=None) -> None:
    try:
        # Optional probes (controlled by env or QSettings)
        s = QSettings(APP_ORG, APP_NAME)
        want_guard = _flag_true(os.environ.get("PROBE_QT_TIMERS", "0")) or _flag_true(s.value("diagnostics_timer_guard","false"))
        want_handles = _flag_true(os.environ.get("PROBE_QT_HANDLES", "0")) or _flag_true(s.value("diagnostics_handle_log","false"))
        thr = int(os.environ.get("PROBE_QT_TIMERS_THRESHOLD", s.value("diagnostics_probe_threshold", 4000)))
        period = int(os.environ.get("PROBE_QT_HANDLES_MS", s.value("diagnostics_handle_log_ms", 1000)))
        if want_guard:
            _install_timer_guard(threshold=thr)
        if want_handles:
            _start_handle_logger(period)

        # Delay dumps a bit to avoid flashing intro
        QTimer.singleShot(1500, log_environment)
        QTimer.singleShot(2000, log_packages)
        QTimer.singleShot(2300, log_ffmpeg)
        QTimer.singleShot(2600, lambda: dump_qsettings("startup"))
        log("Diagnostics installed (post-start dumps).")
    except Exception as e:
        log("Diagnostics install failed:", e)

# Kick installer once after import
try:
    QTimer.singleShot(300, install_all)
except Exception:
    pass
