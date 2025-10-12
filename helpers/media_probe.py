
import os, sys, logging
from pathlib import Path
from datetime import datetime

def _resolve_log_path():
    # Prefer ./logs next to the app; otherwise fall back to %TEMP%\FrameVision
    try:
        base = Path(getattr(sys, "_MEIPASS", os.getcwd()))
        log_dir = base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "media_debug.log"
    except Exception:
        pass
    # Fallback
    temp = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".") / "FrameVision" / "logs"
    try:
        temp.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return temp / "media_debug.log"

_LOGGER = None
_LOG_PATH = None

def _get_logger():
    global _LOGGER, _LOG_PATH
    if _LOGGER:
        return _LOGGER
    try:
        _LOG_PATH = _resolve_log_path()
        lg = logging.getLogger("FrameVision.MediaProbe")
        if not lg.handlers:
            fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            lg.addHandler(fh)
            lg.setLevel(logging.DEBUG)
        _LOGGER = lg
        return lg
    except Exception:
        return None

def _status_name(st):
    try:
        from PySide6.QtMultimedia import QMediaPlayer as _MP
        for name in dir(_MP.MediaStatus):
            val = getattr(_MP.MediaStatus, name)
            if isinstance(val, int) and val == st:
                return name
    except Exception:
        pass
    return str(st)

def _state_name(st):
    try:
        from PySide6.QtMultimedia import QMediaPlayer as _MP
        for name in dir(_MP.PlaybackState):
            val = getattr(_MP.PlaybackState, name)
            if isinstance(val, int) and val == st:
                return name
    except Exception:
        pass
    return str(st)

def _safe(obj, attr, default=None):
    try:
        return getattr(obj, attr) if obj is not None else default
    except Exception:
        return default

def _posdur(player):
    try:
        return _safe(player, "position", lambda: None)(), _safe(player, "duration", lambda: None)()
    except Exception:
        return None, None

def attach_probe(video_pane, player, sink=None, slider=None):
    """
    Attach logging to the given player/sink/slider.
    Safe to call multiple times. No behavior changes.
    """
    lg = _get_logger()
    # Always print to console where logs are going
    try:
        print(f"[MediaProbe] attached. Logging to: {_LOG_PATH}", file=sys.stderr)
    except Exception:
        pass
    if not lg:
        return
    if getattr(video_pane, "_probe_attached", False):
        return
    video_pane._probe_attached = True
    video_pane._probe_logger = lg

    lg.debug("=== attach_probe === Py %s  PID %s", sys.version.split()[0], os.getpid())
    try:
        from PySide6 import QtCore as _QC
        lg.debug("Qt %s  Platform %s  Backend=%s", _QC.QT_VERSION_STR, _safe(_QC.QGuiApplication, "platformName", lambda: "?")(), os.environ.get("QT_MEDIA_BACKEND","?"))
    except Exception:
        pass

    # Helper: small wrapper to throttle positionChanged
    last_pos_log = {"t": 0}
    def _pos_changed(pos):
        try:
            from time import monotonic
            now = monotonic()
            if now - last_pos_log["t"] < 0.25:
                return
            last_pos_log["t"] = now
            d = _safe(player, "duration", lambda: 0)() or 0
            lg.debug("signal: positionChanged -> %s / %s", pos, d)
        except Exception:
            pass

    def _status(st):
        lg.debug("signal: mediaStatusChanged -> %s (%s)", _status_name(st), st)

    def _state(st):
        lg.debug("signal: playbackStateChanged -> %s (%s)", _state_name(st), st)

    def _dur(d):
        lg.debug("signal: durationChanged -> %s", d)

    # Wire signals (no UniqueConnection needed here; harmless if duplicated)
    try: player.mediaStatusChanged.connect(_status)
    except Exception: pass
    try: player.playbackStateChanged.connect(_state)
    except Exception: pass
    try: player.positionChanged.connect(_pos_changed)
    except Exception: pass
    try: player.durationChanged.connect(_dur)
    except Exception: pass

    # One-shot snapshot
    try:
        src = _safe(player, "source", lambda: None)()
        pos, dur = _posdur(player)
        lg.debug("snapshot: status=%s  state=%s  pos=%s  dur=%s  src=%s",
                 _status_name(_safe(player, "mediaStatus", lambda: None)()),
                 _state_name(_safe(player, "playbackState", lambda: None)()),
                 pos, dur, src.toString() if hasattr(src,'toString') else src)
    except Exception:
        pass

def log_open(pathlike, note="open()"):
    lg = _get_logger()
    try:
        print(f"[MediaProbe] {note}: {pathlike}", file=sys.stderr)
    except Exception:
        pass
    if not lg:
        return
    try:
        lg.debug("%s: %s", note, str(pathlike))
    except Exception:
        pass
