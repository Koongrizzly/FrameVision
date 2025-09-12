# helpers/diagnostics.py — post-start dump + hard QPainter suppress
from __future__ import annotations
import os, sys, time, platform, subprocess, json
from typing import Optional
from PySide6 import QtCore
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import QApplication

APP_ORG = "FrameVision"; APP_NAME = "FrameVision"
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "framevision.log")

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

# ----------------- Qt message hook (install ASAP, fully suppress QPainter) -----------------
_prev_qt_handler = None
_logged_qpainter_once = False
def _qt_message_handler(mode, context, message):
    try: m = str(message)
    except Exception: m = message
    if isinstance(m, str) and "QPainter::" in m:
        # fallthrough handled below
        pass
        # Fully suppress. If ever needed, set diagnostics_log_qpainter=true to log first occurrence only.
        try:
            s = QSettings(APP_ORG, APP_NAME)
            if (str(s.value("diagnostics_log_qpainter","false")).lower() in ("1","true","yes","on")):
                global _logged_qpainter_once
                if not _logged_qpainter_once:
                    log(f"Qt(QPainter suppressed): {m}"); _logged_qpainter_once = True
        except Exception: pass
        return
    # Suppress a couple of known-harmless messages
    try:
        if isinstance(m, str):
            if "Could not parse stylesheet of object" in m and "FvSettingsContent" in m:
                return
            if "QObject::startTimer: Timers can only be used with threads started with QThread" in m:
                return
        
        if _prev_qt_handler is not None: _prev_qt_handler(mode, context, message)
        else: print(m)
    except Exception: pass

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

# ----------------- installer (post-start dumps) -----------------
def install_all(main_window=None) -> None:
    try:
        # Delay all visible dumps to avoid flashing the console before intro appears
        QTimer.singleShot(1500, log_environment)
        QTimer.singleShot(2000, log_packages)
        QTimer.singleShot(2300, log_ffmpeg)
        QTimer.singleShot(2600, lambda: dump_qsettings("startup"))
        log("Diagnostics installed (post-start dumps).")
    except Exception as e:
        log("Diagnostics install failed:", e)

try:
    QTimer.singleShot(300, install_all)
except Exception:
    pass
