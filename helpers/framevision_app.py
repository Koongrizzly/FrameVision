from tools.diag_probe import init_diagnostics as _fv_diag_init
from typing import Optional

# --- Prefer FFmpeg backend to avoid WMF first-play stalls ---
try:
    import os
    os.environ.setdefault('QT_MEDIA_BACKEND', 'ffmpeg')
except Exception:
    pass

_fv_diag_init()
# --- BEGIN: FrameVision log silencer (auto-injected) ---
# Hide harmless multimedia/FFmpeg + thread shutdown warnings.
# Scope:
#   1) Tighten QT_LOGGING_RULES to silence qt.multimedia + ffmpeg categories.
#   2) Install a Qt message handler that ignores known-noisy messages.
#   3) Wrap sys.stderr so raw FFmpeg lines like "[mp3 @ ...]" are filtered out.
try:
    import os as __fv_os, sys as __fv_sys, re as __fv_re
    # 1) Expand QT_LOGGING_RULES
    __rules = __fv_os.environ.get("QT_LOGGING_RULES","")
    __extra = "qt.multimedia.*=false;qt.multimedia.warning=false;qt.multimedia.debug=false;qt.multimedia.ffmpeg.*=false"
    if __rules:
        __parts = {x.strip() for x in __rules.split(";") if x.strip()}
    else:
        __parts = set()
    for __piece in __extra.split(";"):
        if __piece and __piece not in __parts:
            __parts.add(__piece)
    __fv_os.environ["QT_LOGGING_RULES"] = ";".join(sorted(__parts))

    # 2) Qt message filter (if PySide6 available)
    try:
        from PySide6 import QtCore as __fv_QtCore
        __prev = __fv_QtCore.qInstallMessageHandler(None)  # grab current (if any) then clear
        __silence_substrings = (
            "CreateFontFaceFromHDC",      # font noise on some Windows setups
            "MS Sans Serif",
            "QThread: Destroyed while thread is still running",  # harmless during shutdown
            "Could not update timestamps for skipped samples",   # FFmpeg mp3float notice
            "Could not find codec parameters for stream",        # cover-art streams in mp3
            "analyzeduration", "probesize"
        )
        def __fv_qt_msg_filter(mode, ctx, msg):
            try:
                __s = str(msg)
                if any(sub in __s for sub in __silence_substrings):
                    return
            except Exception:
                pass
            if __prev:
                try:
                    __prev(mode, ctx, msg)
                except Exception:
                    pass
        __fv_QtCore.qInstallMessageHandler(__fv_qt_msg_filter)
    except Exception:
        pass

    # 3) stderr wrapper to hide raw FFmpeg lines not routed via Qt logging
    class __FVFilteredStderr:
        __slots__ = ("_orig", "_subs")
        def __init__(self, orig, subs):
            self._orig = orig
            self._subs = tuple(subs)
        def write(self, data):
            try:
                s = str(data)
                if any(sub in s for sub in self._subs):
                    return
                self._orig.write(data)
            except Exception:
                try:
                    self._orig.write(data)
                except Exception:
                    pass
        def flush(self):
            try:
                self._orig.flush()
            except Exception:
                pass
        def fileno(self):
            try:
                return self._orig.fileno()
            except Exception:
                raise
        def isatty(self):
            try:
                return self._orig.isatty()
            except Exception:
                return False
        def __getattr__(self, name):
            return getattr(self._orig, name)

    try:
        __patterns = (
            "[mp3 @", "[mp3float @", "Could not update timestamps for skipped samples",
            "Could not find codec parameters for stream", "analyzeduration", "probesize",
            "QThread: Destroyed while thread is still running"
        )
        __fv_sys.stderr = __FVFilteredStderr(__fv_sys.stderr, __patterns)
    except Exception:
        pass

except Exception:
    pass
# --- END: FrameVision log silencer (auto-injected) ---

import os as _os
_rules = _os.environ.get("QT_LOGGING_RULES","")
_append = "qt.qpa.fonts=false;qt.fonts.warning=false;qt.fonts.debug=false"
if not _rules:
    _os.environ["QT_LOGGING_RULES"] = _append
else:
    parts = {x.strip() for x in _rules.split(";") if x.strip()}
    for piece in _append.split(";"):
        if piece and piece not in parts:
            parts.add(piece)
    _os.environ["QT_LOGGING_RULES"] = ";".join(sorted(parts))

# --- compare page opener (module-level) ---
def _open_compare_page():
    try:
        path = ROOT / "assets" / "compare.html"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
    except Exception:
        pass




# FrameVision — Full-classic UI + NCNN upscalers + branding
# Classic layout (big player, control bar, seek slider, fullscreen) + click-to-play/pause
# Auto Theme (Day/Evening/Night), Session Restore, Instant Tools, Presets, Describe-on-Pause,
# Queue + Worker, Models Manager, Upscale Video/Photo buttons (queue) wired to NCNN CLIs.
# Branding: Title bar, About text, splash placeholder, default output under FrameVision/

import os, sys, json, subprocess, re, hashlib
from helpers.collapsible_compat import CollapsibleSection
from helpers.ask_popup import AskPopup
from helpers import state_persist
from helpers.img_fallback import load_pixmap
from helpers.worker_led import WorkerStatusWidget

try:
    from helpers.queue_pane import QueuePane
except Exception:
    from queue_pane import QueuePane

from helpers.tools_tab import CollapsibleSection as ToolsCollapsibleSection
from pathlib import Path
# ---- Begin: Quiet specific Qt warnings ----
_prev_qt_handler = None
def _fv_qt_msg_filter(mode, ctx, msg):
    s = str(msg)
    if ("CreateFontFaceFromHDC" in s) or ("MS Sans Serif" in s):
        return
    if _prev_qt_handler:
        _prev_qt_handler(mode, ctx, msg)

try:
    from PySide6 import QtCore as _QtCore
    _prev_qt_handler = _QtCore.qInstallMessageHandler(_fv_qt_msg_filter)
except Exception:
    pass
# ---- End: Quiet specific Qt warnings ----


# from helpers.quick_action_button import QuickActionDriver  # removed
# --- Block creation of any directory containing "framelab" (legacy) ---------
try:
    import os as _os
    _orig_makedirs = getattr(_os, "makedirs", None)
    def _fv_safe_makedirs(path, *args, **kwargs):
        try:
            if path and ("framelab" in str(path).lower()):
                # Skip creating legacy framelab folders
                return None
        except Exception:
            pass
        if _orig_makedirs:
            return _orig_makedirs(path, *args, **kwargs)
    if _orig_makedirs:
        _os.makedirs = _fv_safe_makedirs  # monkeypatch
except Exception:
    pass
# ---------------------------------------------------------------------------
from pathlib import Path
import os
from datetime import datetime
from PySide6.QtCore import QUrl, Qt, QTimer, QUrl, Signal, QRect, QEasingCurve, QPropertyAnimation, QByteArray, QEvent
from PySide6.QtCore import QUrl, QSettings
from PySide6.QtGui import QAction, QPixmap, QImage, QKeySequence, QColor, QDesktopServices, QShortcut

# --- BEGIN: Image allocation limit bump ---
try:
    from PySide6.QtGui import QImageReader
    # Raise per-image allocation limit to 1.5 GB (prevents 'Rejecting image ... 256 megabytes')
    try:
        QImageReader.setAllocationLimit(1536)  # megabytes
    except Exception:
        pass
    try:
        import os as __img_env_os
        # Some plugins also read this env var; set as bytes for compatibility.
        __img_env_os.environ.setdefault("QT_IMAGEIO_MAXALLOC", str(1536 * 1024 * 1024))
    except Exception:
        pass
except Exception:
    pass
# --- END: Image allocation limit bump ---

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFileDialog, QTabWidget, QSplitter, QStackedWidget, QListWidget, QListWidgetItem, QLineEdit, QFormLayout, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QStyle, QSlider, QToolButton, QSizePolicy, QScrollArea, QFrame, QGroupBox, QScrollArea, QFrame)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from helpers.tools_tab import InstantToolsPane
from helpers.themes import QSS_DAY, QSS_EVENING, QSS_NIGHT
from helpers.mediainfo import AUDIO_EXTS, probe_media_all, show_info_popup
from helpers.volume_new import add_new_volume_popup
from helpers.background import install_background_tool


# >>> FRAMEVISION_TXT2IMG_BEGIN
# Safe import of the txt2img pane; never crash app on failure.
try:
    from helpers.txt2img import Txt2ImgPane
except Exception as _e:
    print("[framevision] txt2img tab import failed:", _e)
    Txt2ImgPane = None
# <<< FRAMEVISION_TXT2IMG_END

# >>> FRAMEVISION_PLANNER_TAB_BEGIN
# Safe import of the Planner tab; never crash app on failure.
try:
    from helpers.planner import PlannerPane
except Exception as _e:
    print("[framevision] planner tab import failed:", _e)
    PlannerPane = None
# <<< FRAMEVISION_PLANNER_TAB_END


# >>> FRAMEVISION_MUSICCLIP_BEGIN
# Safe import of the Music Clip Creator tab; never crash app on failure.
try:
    from helpers.auto_music_sync import OneClickVideoClipTab as MusicClipCreatorTab
except Exception as _e:
    print("[framevision] Music Clip Creator tab import failed:", _e)
    MusicClipCreatorTab = None
# <<< FRAMEVISION_MUSICCLIP_END

# >>> CORRECT FRAMEVISION_EDITOR_IMPORT BEGIN
# Safe import of the Editor pane; never crash app on failure.
#try:
#    from helpers.editor import EditorPane
#except Exception as _e:
#    print("[framevision] editor tab import failed:", _e)
#    EditorPane = None
# <<< FRAMEVISION_EDITOR_END

# >>> FRAMEVISION_WAN22_BEGIN
# Safe import of the WAN 2.2 pane; never crash app on failure.
# We import the module first so that a class name mismatch is easier to debug,
# and we try a few common pane class names before giving up.
Wan22Pane = None
try:
    import helpers.wan22 as _wan22_mod
    for _name in ("Wan22Pane", "WAN22Pane", "WanPane", "WanTab", "Wan22Tab"):
        Wan22Pane = getattr(_wan22_mod, _name, None)
        if Wan22Pane is not None:
            break
    if Wan22Pane is None:
        print("[framevision] WAN22: module imported but no suitable pane class found. "
              "Expected one of Wan22Pane/WAN22Pane/WanPane/WanTab/Wan22Tab.")
except Exception as _e:
    print("[framevision] WAN22 tab import failed:", _e)
    Wan22Pane = None
# <<< FRAMEVISION_WAN22_END

# >>> FRAMEVISION_QWEN2511_BEGIN
# Safe import of the Qwen2511 pane; never crash app on failure.
# We import the module first so that a class name mismatch is easier to debug,
# and we try a few common pane class names before giving up.
Qwen2511Pane = None
try:
    import helpers.qwen2511 as _qwen2511_mod
    for _name in ("Qwen2511Pane", "Qwen2511Tab", "QwenPane", "Qwen2511Widget"):
        Qwen2511Pane = getattr(_qwen2511_mod, _name, None)
        if Qwen2511Pane is not None:
            break
    if Qwen2511Pane is None:
        print("[framevision] Qwen2511: module imported but no suitable pane class found. "
              "Expected one of Qwen2511Pane/Qwen2511Tab/QwenPane/Qwen2511Widget.")
except Exception as _e:
    print("[framevision] Qwen2511 tab import failed:", _e)
    Qwen2511Pane = None
# <<< FRAMEVISION_QWEN2511_END



# >>> FRAMEVISION_ace_BEGIN
# Safe import of the ace pane; never crash app on failure.
try:
    from helpers.ace import acePane
except Exception as _e:
    print("[framevision] ace tab import failed:", _e)
    acePane = None
# <<< FRAMEVISION_ace_END

# >>> FRAMEVISION_ACE_STEP_15_BEGIN
# Safe import of Ace-Step 1.5 pane; never crash app on failure.
AceStep15Pane = None
try:
    import helpers.ace_step_15 as _ace15_mod
    # Prefer an embeddable QWidget pane if provided.
    for _name in ("AceStep15Pane", "AceStep15Tab", "AceStep15Widget"):
        AceStep15Pane = getattr(_ace15_mod, _name, None)
        if AceStep15Pane is not None:
            break
    # Fallback: factory function
    if AceStep15Pane is None:
        _factory = getattr(_ace15_mod, "create_pane", None)
        if callable(_factory):
            AceStep15Pane = _factory
    if AceStep15Pane is None:
        print("[framevision] Ace-Step 1.5: module imported but no suitable pane found. "
              "Expected AceStep15Pane (or create_pane()).")
except Exception as _e:
    print("[framevision] Ace-Step 1.5 tab import failed:", _e)
    AceStep15Pane = None
# <<< FRAMEVISION_ACE_STEP_15_END



try:
    # Replace None placeholders with imported QSS constants
    if QSS_DAY is None or QSS_EVENING is None or QSS_NIGHT is None:
        from helpers import themes as _themes
        QSS_DAY = getattr(_themes, "QSS_DAY", "")
        QSS_EVENING = getattr(_themes, "QSS_EVENING", "")
        QSS_NIGHT = getattr(_themes, "QSS_NIGHT", "")
except Exception:
    pass

from helpers.queue_system import QueueSystem
from helpers.mods import ModelsPane
from helpers.interp import InterpPane
import psutil

APP_NAME = "FrameVision"
TAGLINE  = "All-in-one Image-Video-Sound Tool"
ROOT = Path(".").resolve()

# Check if extra environment is installed for WAN 2.2 and AceMusic virtualenv folders at app root
WAN22_ENV_DIR_LEGACY = ROOT / ".wan_venv"
ACE_ENV_DIR = ROOT / "presets" / "extra_env" / ".ace_env"


# --- Grabbable Splitter with themed hover/arrow & cursor ----------------------
from PySide6.QtWidgets import QSplitter, QSplitterHandle
from PySide6.QtGui import QPainter, QPen, QBrush, QPixmap, QCursor
from PySide6.QtCore import QUrl, QPoint

# --- Animated, theme-aware tab-nav button --------------------------------------
from PySide6.QtCore import Property, QEasingCurve
from PySide6.QtGui import QColor

class _TabNavButton(QToolButton):
    def __init__(self, glyph: str, parent=None):
        super().__init__(parent)
        self.setText(glyph)
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._bg = QColor(0,0,0,0)
        self._base = self.palette().button().color()
        # Use the theme highlight for "selected tab" look
        self._hi = self.palette().highlight().color()
        # Start transparent so it blends with the tabbar
        self._anim = QPropertyAnimation(self, b"bgColor", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.setStyleSheet("QToolButton{border-radius:8px;padding:4px 10px;}")
        self.pressed.connect(self._pulse_in)
        self.released.connect(self._pulse_out)

    def _pulse_in(self):
        # animate to highlight
        try:
            self._anim.stop()
            self._anim.setStartValue(self._bg)
            self._anim.setEndValue(self._hi)
            self._anim.start()
        except Exception:
            pass

    def _pulse_out(self):
        # animate back to transparent/base
        try:
            self._anim.stop()
            end = QColor(self._base)
            end.setAlpha(0)
            self._anim.setStartValue(self._bg)
            self._anim.setEndValue(end)
            self._anim.start()
        except Exception:
            pass

    def get_bg(self): 
        return self._bg

    def set_bg(self, c: QColor):
        try:
            self._bg = QColor(c)
            # Apply via stylesheet so it remains theme-friendly.
            # We don't set any hard-coded colors for text; let the theme/palette handle it.
            rgba = f"rgba({self._bg.red()},{self._bg.green()},{self._bg.blue()},{self._bg.alpha()})"
            self.setStyleSheet("QToolButton{border-radius:8px;padding:4px 10px;background:%s;}" % rgba)
        except Exception:
            pass

    bgColor = Property(QColor, get_bg, set_bg)


def _format_temp_units(celsius_int: int) -> str:
    try:
        from helpers.framevision_app import config as _cfg
        units = (_cfg.get('temp_units','C') or 'C').upper()
    except Exception:
        units = 'C'
    try:
        c = float(celsius_int)
    except Exception:
        c = float(celsius_int or 0)
    if units == 'F':
        return f"{int(round((c * 9/5) + 32))}F"
    return f"{int(round(c))}C"

def current_theme_name():
    try:
        name = config.get("theme", "Auto")
    except Exception:
        name = "Auto"
    if name == "Auto":
        try:
            return pick_auto_theme()
        except Exception:
            return "Day"
    return name

def theme_arrow_color():
    name = current_theme_name()
    # Match your theme palette: bright blue at night/evening, black at day
    if name in ("Evening", "Night", "Slate", "High Contrast"):
        return QColor("#00A3FF")
    return QColor("#111111")

def _make_cursor_pix(color: 'QColor') -> 'QCursor':
    w = h = 24
    pm = QPixmap(w, h); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(color, 2); p.setPen(pen)
    cx, cy = w//2, h//2
    # left arrow
    p.drawLine(cx-8, cy, cx-2, cy-6); p.drawLine(cx-8, cy, cx-2, cy+6)
    # right arrow
    p.drawLine(cx+8, cy, cx+2, cy-6); p.drawLine(cx+8, cy, cx+2, cy+6)
    p.end()
    return QCursor(pm, cx, cy)

class GrabbableHandle(QSplitterHandle):
    _cursor_day = None
    _cursor_night = None

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._hover = False
        self.setMouseTracking(True)
        # default split cursor (OS), will be replaced on hover
        self.setCursor(Qt.SplitHCursor)

    def _apply_theme_cursor(self):
        name = current_theme_name()
        if name in ("Evening", "Night", "Slate", "High Contrast"):
            if not GrabbableHandle._cursor_night:
                GrabbableHandle._cursor_night = _make_cursor_pix(theme_arrow_color())
            self.setCursor(GrabbableHandle._cursor_night)
        else:
            if not GrabbableHandle._cursor_day:
                GrabbableHandle._cursor_day = _make_cursor_pix(theme_arrow_color())
            self.setCursor(GrabbableHandle._cursor_day)

    def enterEvent(self, e):
        self._hover = True
        self._apply_theme_cursor()
        super().enterEvent(e)
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        # back to standard split cursor when not hovering
        self.setCursor(Qt.SplitHCursor)
        super().leaveEvent(e)
        self.update()

    def paintEvent(self, ev):
        r = self.rect()
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        # subtle background on hover
        if self._hover:
            p.fillRect(r, QColor(0, 0, 0, 40))
        # draw double arrow centered (matches cursor color)
        col = theme_arrow_color()
        pen = QPen(col, 2); p.setPen(pen)
        cx = r.center().x(); cy = r.center().y()
        # left arrow
        p.drawLine(cx-8, cy, cx-2, cy-6); p.drawLine(cx-8, cy, cx-2, cy+6)
        # right arrow
        p.drawLine(cx+8, cy, cx+2, cy-6); p.drawLine(cx+8, cy, cx+2, cy+6)
        # grip dots
        p.setPen(QPen(col, 1))
        for dy in (-6, 0, 6):
            p.drawPoint(cx, cy+dy)
        p.end()

class GrabbableSplitter(QSplitter):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        try:
            self.setHandleWidth(14)
        except Exception:
            pass

    def createHandle(self):
        return GrabbableHandle(self.orientation(), self)

# --- Unified BASE at project root + legacy migration (FrameVision/, framevision/, FrameLab/)
LEGACY_BASES = [ROOT / "FrameVision", ROOT / "framevision", ROOT / "FrameLab"]

def _migrate_legacy_tree():
    base = ROOT
    # Create target dirs first
    for p in ["output/video","output/trims","output/screenshots","output/descriptions","output/_temp",
              "jobs/pending","jobs/running","jobs/done","jobs/failed","logs","models","helpers"]:
        (base / p).mkdir(parents=True, exist_ok=True)
    # Move files from legacy trees into the root-based structure
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
                        # avoid overwrite: tag as migrated
                        stem, suff = target.stem, target.suffix
                        alt = target.with_name(f"{stem}_migrated{suff}")
                        pth.replace(alt)
                except Exception:
                    # best-effort; skip on error
                    pass

def ensure_out_root():
    base = ROOT
    for p in ["output/video","output/trims","output/screenshots","output/descriptions","output/_temp",
              "jobs/pending","jobs/running","jobs/done","jobs/failed","logs","models","helpers"]:
        (base / p).mkdir(parents=True, exist_ok=True)
    return base

# One-time migration then standardize to ROOT as BASE
_migrate_legacy_tree()
BASE = ensure_out_root()


OUT_VIDEOS = BASE / "output" / "video"
OUT_TRIMS  = BASE / "output" / "trims"
OUT_SHOTS  = BASE / "output" / "screenshots"
OUT_DESCR  = BASE / "output" / "descriptions"
OUT_TEMP   = BASE / "output" / "_temp"
JOBS_DIRS = {
    "pending": BASE / "jobs" / "pending",
    "running": BASE / "jobs" / "running",
    "done": BASE / "jobs" / "done",
    "failed": BASE / "jobs" / "failed",
}
HELPERS   = BASE / "helpers"
LOGS      = BASE / "logs"
MODELS_DIR= BASE / "models"

CONFIG_PATH   = BASE / "config.json"
PRESETS_PATH  = BASE / "presets.json"
MANIFEST_PATH = ROOT / "models_manifest.json"  # keep shared at root

def load_json(path: Path, default):
    if not path.exists():
        path.write_text(json.dumps(default, indent=2), encoding="utf-8")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

config = load_json(CONFIG_PATH, {
    "models_folder": str(MODELS_DIR),
    "last_open_dir": str(ROOT),
    "caption_model":"Model 1",
    "auto_on_pause": True,
    "models_status": {},
    "show_toasts": True,
    "planner_last_choice": "",

    "theme": "Auto",
    "session_restore": { "last_file": "", "last_position_ms": 0, "active_tab": 0,
                         "win_geometry_b64":"", "splitter_state_b64":"" }
})
presets = load_json(PRESETS_PATH, {"crop": {}, "resize": {}, "export": {}})

# Default manifest if missing (at project root so it persists across brand folders)
if not MANIFEST_PATH.exists():
    MANIFEST_PATH.write_text(json.dumps({
        "RealESR-general-x4v3": {"exe":"realesrgan-ncnn-vulkan.exe","url":"", "checksum":""},
        "RealESRGAN-x4plus": {"exe":"realesrgan-ncnn-vulkan.exe","url":"", "checksum":""},
        "SwinIR-x4": {"exe":"swinir-ncnn-vulkan.exe","url":"", "checksum":""},
        "LapSRN-x4": {"exe":"lapsrn-ncnn-vulkan.exe","url":"", "checksum":""},
        "RIFE": {"exe":"rife-ncnn-vulkan.exe","url":"", "checksum":""}
    }, indent=2), encoding="utf-8")

def save_config(): CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
def save_presets(): PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")

# --- Startup logo rotation (presets/logo) ---
def _fv_select_startup_logo():
    try:
        folder = ROOT / "presets" / "logo"
        if not folder.exists():
            return None
        files = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
        files.sort()
        if not files:
            return None
        idx = int(config.get("logo_cycle_index", 0) or 0)
        chosen = files[idx % len(files)]
        # advance index once per app start
        config["logo_cycle_index"] = (idx + 1) % len(files)
        try:
            save_config()
        except Exception:
            pass
        return chosen
    except Exception:
        return None

STARTUP_LOGO_PATH = _fv_select_startup_logo()

def now_stamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def _sec_to_mss(sec_val):
    """Return M:SS or H:MM:SS for a duration in seconds (float or str).
    If sec_val is None/invalid, return empty string."""
    try:
        total = int(float(sec_val or 0))
    except Exception:
        return ""
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def compose_video_info_text(path: Path) -> str:
    """Build the first info line under the player.

    User-facing intent:
    - Keep it short.
    - For videos: show filename • WxH • size on disk (e.g., 5.5mb)
    - For audio: show filename • duration • size
    - For still images: show filename • size • WxH (when available)
    """
    try:
        inf = probe_media(path)
        ext = str(path.suffix or "").lower()

        # raw bytes from ffprobe, fallback to filesystem size
        size_b = inf.get("size")
        try:
            if not size_b:
                size_b = path.stat().st_size
        except Exception:
            size_b = size_b or 0

        def _fmt_mb_simple(b):
            try:
                mb = float(b) / (1024.0 * 1024.0)
            except Exception:
                mb = 0.0
            # one decimal, trim trailing .0
            s = f"{mb:.1f}"
            if s.endswith(".0"):
                s = s[:-2]
            return f"{s}mb"

        dur_txt = _sec_to_mss(inf.get("duration"))

        audio_exts = set(x.lower() for x in list(AUDIO_EXTS))
        image_exts = {
            ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
            ".gif", ".avif", ".heic", ".heif"
        }

        w = inf.get("width")
        h = inf.get("height")

        # Audio file
        if ext in audio_exts:
            parts = [path.name]
            if dur_txt:
                parts.append(dur_txt)
            parts.append(_fmt_mb_simple(size_b))
            return " • ".join(parts)

        # Still images (including gif treated as image for info line)
        if ext in image_exts:
            parts = [path.name, _fmt_mb_simple(size_b)]
            if w and h:
                parts.append(f"{w}x{h}")
            return " • ".join(parts)

        # Video / default
        parts = [path.name]
        if w and h:
            parts.append(f"{w}x{h}")
        parts.append(_fmt_mb_simple(size_b))
        return " • ".join(parts)
    except Exception:
        try:
            return str(path.name)
        except Exception:
            return ""
def human_mb(b):
    try: return round(b/(1024*1024),2)
    except: return 0.0

def ffmpeg_path():
    candidates = [ROOT/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), "ffmpeg"]
    for c in candidates:
        try: subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT); return str(c)
        except Exception: continue
    return "ffmpeg"

def ffprobe_path():
    candidates = [ROOT/"bin"/("ffprobe.exe" if os.name=="nt" else "ffprobe"), "ffprobe"]
    for c in candidates:
        try: subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT); return str(c)
        except Exception: continue
    return "ffprobe"

def probe_media(path: Path):
    info = {"width": None, "height": None, "fps": None, "duration": None, "size": None}
    try:
        out = subprocess.check_output([ ffprobe_path(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-show_entries", "format=duration,size",
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(path) ], stderr=subprocess.STDOUT, universal_newlines=True)
        for line in out.splitlines():
            if line.startswith("width="): info["width"] = int(line.split("=")[1])
            if line.startswith("height="): info["height"] = int(line.split("=")[1])
            if line.startswith("avg_frame_rate="):
                fr = line.split("=")[1]
                if "/" in fr:
                    n,d = fr.split("/")
                    if float(d)!=0: info["fps"] = round(float(n)/float(d),2)
            if line.startswith("duration="):
                try: info["duration"] = float(line.split("=")[1])
                except: pass
            if line.startswith("size="):
                try: info["size"] = int(line.split("=")[1])
                except: pass
    except Exception:
        pass
    return info

# --- Themes (QSS condensed for size)
QSS_DAY = None  # moved to helpers.themes
QSS_EVENING = None  # moved to helpers.themes
QSS_NIGHT = None  # moved to helpers.themes

def pick_auto_theme():
    t = datetime.now().strftime("%H:%M"); h, m = map(int, t.split(":")); mins = h*60 + m
    if 420 <= mins <= 1019: return "Day"
    if 1020 <= mins <= 1409: return "Evening"
    return "Night"

def apply_theme(app, name: str) -> None:
    try:
        from helpers.themes import apply_theme as _themes_apply
    except Exception:
        return
    try:
        _themes_apply(app, name)
    except Exception:
        pass

    def __init__(self, title: str, parent=None, expanded=False):
        super().__init__(parent)
        self._expanded = bool(expanded)
        self.toggle = QToolButton(self)
        self.toggle.setStyleSheet("QToolButton { border:none; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)

        self.content = QWidget(self)
        self.content.setLayout(QVBoxLayout()); self.content.layout().setSpacing(8)
        self.content.layout().setContentsMargins(12, 6, 12, 12)
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.content.setVisible(self._expanded)
        self.content.setMaximumHeight(16777215 if self._expanded else 0)

        self.anim = QPropertyAnimation(self.content, b"maximumHeight", self)
        self.anim.setDuration(160)
        self.anim.finished.connect(self._on_anim_finished)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

        def _on_toggled(on):
            self._expanded = on
            self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            self.content.setVisible(True)  # keep visible during animation
            start = self.content.maximumHeight()
            end = self.content.sizeHint().height() if on else 0
            self.anim.stop()
            self.anim.setStartValue(start)
            self.anim.setEndValue(end)
            self.anim.start()
        self.toggle.toggled.connect(_on_toggled)

    def _on_anim_finished(self):
        # After expanding, let content take natural height so scrollbars can appear
        if self._expanded:
            self.content.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self.content.setMaximumHeight(0)
            self.content.setVisible(False)

    def setContentLayout(self, layout):
        # Replace inner layout safely
        QWidget().setLayout(self.content.layout())
        self.content.setLayout(layout)
        if self._expanded:
            self.content.setMaximumHeight(16777215)
        else:
            self.content.setMaximumHeight(0)

class Toast(QWidget):
    def __init__(self, text: str, parent=None, ms=3000):
        super().__init__(parent, flags=Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        w = QWidget(self); w.setObjectName("toast"); lay = QHBoxLayout(w); lay.setContentsMargins(12,8,12,8)
        ico = QLabel(); ico.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxInformation).pixmap(16,16))
        lbl = QLabel(text); lay.addWidget(ico); lay.addWidget(lbl)
        outer = QVBoxLayout(self); outer.addWidget(w)
        QTimer.singleShot(ms, self.close)
        self.setStyleSheet("#toast {background: rgba(30,30,30,220); color: white; border-radius: 8px; font-size: 12px;}")
    def show_at(self, parent: QWidget, margin=20):
        parent_rect = parent.geometry(); self.adjustSize(); size = self.size()
        x = parent_rect.x()+parent_rect.width()-size.width()-margin
        y = parent_rect.y()+parent_rect.height()-size.height()-margin
        self.setGeometry(QRect(x,y,size.width(),size.height())); self.show()

# --- Video Pane with classic controls
IMAGE_EXTS = {'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff','.gif'}

class VideoPane(QWidget):

    # Emitted when the player is paused and we have a captured frame (used by Describer / pause-capture).
    # Keep signature generic to avoid Qt type mismatches across backends.
    frameCaptured = Signal(object)

    # --- Rebuild player/sinks to avoid backend deadlocks (no-bump edition) ---
    def _rebuild_player(self):
        try:
            try: self.player.stop()
            except Exception: pass
            try: self.sink.videoFrameChanged.disconnect(self._on_frame)
            except Exception: pass
            try: self.player.positionChanged.disconnect(self.on_pos)
            except Exception: pass
            try: self.player.durationChanged.disconnect(self.on_dur)
            except Exception: pass
            try: self.player.playbackStateChanged.disconnect(self._on_playback_state)
            except Exception: pass
            try: self.player.mediaStatusChanged.disconnect(self._on_media_status)
            except Exception: pass
            try: self.player.setVideoSink(None)
            except Exception: pass
            try: self.player.setAudioOutput(None)
            except Exception: pass
            try: self.audio.deleteLater()
            except Exception: pass
            try: self.sink.deleteLater()
            except Exception: pass
            try: self.player.deleteLater()
            except Exception: pass
        except Exception:
            pass
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
            self.player = QMediaPlayer(self)
            self.audio = QAudioOutput(self)
            self.player.setAudioOutput(self.audio)
            self.sink = QVideoSink(self)
            self.player.setVideoSink(self.sink)
            # If a tool has redirected video output, honor it after rebuild.
            try:
                ov = getattr(self, "_video_output_override", None)
                if ov:
                    mode, target = ov
                    if str(mode) == "sink":
                        self.player.setVideoSink(target)
                    elif str(mode) == "output":
                        try:
                            self.player.setVideoSink(None)
                        except Exception:
                            pass
                        self.player.setVideoOutput(target)
            except Exception:
                pass
            self.sink.videoFrameChanged.connect(self._on_frame)
            self.player.positionChanged.connect(self.on_pos)
            self.player.durationChanged.connect(self.on_dur)
            try:
                self.player.playbackStateChanged.connect(lambda *_: self._sync_play_button())
                self.player.playbackStateChanged.connect(self._on_playback_state)
            except Exception:
                pass
            try:
                self.player.mediaStatusChanged.connect(self._on_media_status)
            except Exception:
                pass
        except Exception:
            pass

        # Inform MainWindow that the internal player was rebuilt so global hooks can rebind safely.
        try:
            hook = getattr(self.main, "_hook_video_player", None)
            if callable(hook):
                hook(self.player)
        except Exception:
            pass
    def _open_ask_popup(self):
        """Open the Ask chat popup (lazy-create)."""
        if not hasattr(self, "_ask_popup") or self._ask_popup is None:
            self._ask_popup = AskPopup(parent=self.window())
        self._ask_popup.show()
        try:
            self._ask_popup.raise_()
            self._ask_popup.activateWindow()
        except Exception:
            pass



    



    def _set_repeat_available(self, is_video: bool):
        try:
            if hasattr(self, 'btn_repeat'):
                self.btn_repeat.setVisible(bool(is_video))
                try:
                    from PySide6.QtCore import QTimer as _QTimer
                    _QTimer.singleShot(0, self._update_compact_button_labels)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_repeat_style(self):
        try:
            enabled = bool(getattr(self, '_repeat_enabled', False))
            if not hasattr(self, 'btn_repeat') or self.btn_repeat is None:
                return

            compact = bool(getattr(self.btn_repeat, "_fv_compact_label", False))
            pad = "4px 10px" if compact else "4px 14px"
            fw  = "900" if compact else "600"

            base = self.palette().button().color()
            hi = self.palette().highlight().color()
            txt_base = self.palette().buttonText().color()
            txt_hi = self.palette().highlightedText().color()
            bg = hi if enabled else base
            fg = txt_hi if enabled else txt_base
            css = (
                f"QPushButton#btn_repeat {{ padding:{pad}; border-radius:8px; font-weight:{fw};"
                f" background: rgba({bg.red()},{bg.green()},{bg.blue()},255);"
                f" color: rgba({fg.red()},{fg.green()},{fg.blue()},255); }}"
                "QPushButton#btn_repeat:hover { opacity: 0.9; }"
            )
            self.btn_repeat.setStyleSheet(css)
        except Exception:
            pass

    def _toggle_repeat(self):
        try:
            self._repeat_enabled = not bool(getattr(self, '_repeat_enabled', False))
            self._update_repeat_style()
        except Exception:
            pass

    # --- Compact label logic (auto) ---
    def _apply_compact_label(self, btn, compact: bool, full_text: str, compact_text: str, style_full: str, style_compact: str):
        try:
            if btn is None:
                return
            want = bool(compact)
            cur = bool(getattr(btn, "_fv_compact_label", False))
            if want == cur:
                return
            btn._fv_compact_label = want
            if want:
                btn.setText(str(compact_text or "")[:1].upper())
                try:
                    btn.setStyleSheet(style_compact)
                except Exception:
                    pass
                try:
                    btn.setToolTip(full_text)
                except Exception:
                    pass
            else:
                btn.setText(full_text)
                try:
                    btn.setStyleSheet(style_full)
                except Exception:
                    pass
                try:
                    btn.setToolTip("")
                except Exception:
                    pass
        except Exception:
            pass

    def _update_compact_button_labels(self):
        """Swap Upscale/Ask Framie/Repeat to single-letter mode when there isn't room for full labels."""
        try:
            bu = getattr(self, "btn_upscale", None)
            ba = getattr(self, "btn_ask", None)
            br = getattr(self, "btn_repeat", None)
            if bu is None or ba is None:
                return
            include_repeat = bool(br is not None and br.isVisible())

            # Layout can report 0 sizes early in init; retry once we're laid out.
            if self.width() <= 10:
                try:
                    from PySide6.QtCore import QTimer as _QTimer
                    _QTimer.singleShot(0, self._update_compact_button_labels)
                except Exception:
                    pass
                return

            full_u = getattr(self, "_btn_upscale_full_text", "Upscale")
            full_a = getattr(self, "_btn_ask_full_text", "Ask Framie")
            full_r = getattr(self, "_btn_repeat_full_text", "Repeat")

            fm_u = bu.fontMetrics()
            fm_a = ba.fontMetrics()

            # Approximate full-label width as (text + padding/border).
            need_u = int(fm_u.horizontalAdvance(full_u)) + 28
            need_a = int(fm_a.horizontalAdvance(full_a)) + 28

            need_r = 0
            if include_repeat:
                try:
                    fm_r = br.fontMetrics()
                    need_r = int(fm_r.horizontalAdvance(full_r)) + 28
                except Exception:
                    need_r = 0

            # Available width in the bar: subtract some spacing/margins.
            avail = int(self.width()) - 40

            need_full = need_u + need_a + (need_r if include_repeat else 0)

            want_compact = need_full > avail

            # Cache styles for toggle (only once).
            try:
                if not hasattr(self, "_btn_upscale_style_full"):
                    self._btn_upscale_style_full = bu.styleSheet() or ""
                if not hasattr(self, "_btn_ask_style_full"):
                    self._btn_ask_style_full = ba.styleSheet() or ""
                if include_repeat and br is not None and not hasattr(self, "_btn_repeat_style_full"):
                    self._btn_repeat_style_full = br.styleSheet() or ""
            except Exception:
                pass

            if want_compact:
                try:
                    bu.setText(getattr(self, "_btn_upscale_compact_text", "U"))
                    ba.setText(getattr(self, "_btn_ask_compact_text", "A"))
                    if include_repeat and br is not None:
                        br.setText(getattr(self, "_btn_repeat_compact_text", "R"))
                except Exception:
                    pass
                # Optional compact styles (if present)
                try:
                    if hasattr(self, "_btn_upscale_style_compact"):
                        bu.setStyleSheet(getattr(self, "_btn_upscale_style_compact", ""))
                    if hasattr(self, "_btn_ask_style_compact"):
                        ba.setStyleSheet(getattr(self, "_btn_ask_style_compact", ""))
                    if include_repeat and br is not None and hasattr(self, "_btn_repeat_style_compact"):
                        br.setStyleSheet(getattr(self, "_btn_repeat_style_compact", ""))
                except Exception:
                    pass
            else:
                try:
                    bu.setText(full_u)
                    ba.setText(full_a)
                    if include_repeat and br is not None:
                        br.setText(full_r)
                except Exception:
                    pass
                try:
                    bu.setStyleSheet(getattr(self, "_btn_upscale_style_full", bu.styleSheet()))
                    ba.setStyleSheet(getattr(self, "_btn_ask_style_full", ba.styleSheet()))
                    if include_repeat and br is not None:
                        br.setStyleSheet(getattr(self, "_btn_repeat_style_full", br.styleSheet()))
                except Exception:
                    pass
        except Exception:
            pass

    def _set_zoom(self, new_zoom: float):
        try:
            z = float(new_zoom)
        except Exception:
            z = 1.0
        z = max(0.5, min(self._max_zoom, z))
        z = round(z / self._zoom_step) * self._zoom_step
        if z < 0.5: z = 0.5
        if z > self._max_zoom: z = self._max_zoom
        if abs(z - getattr(self, '_zoom', 1.0)) < 1e-6:
            return
        self._zoom = z
        try:
            if z <= 1.0:
                self._pan_cx = 0.5; self._pan_cy = 0.5
        except Exception:
            pass
        self._clamp_pan()
        try:
            self._refresh_label_pixmap()
        except Exception:
            pass

    def _clamp_pan(self):
        try:
            z = float(getattr(self, '_zoom', 1.0))
            if z < 1.0:
                vw = 1.0; vh = 1.0
            else:
                vw = 1.0 / z; vh = 1.0 / z
            cx = float(getattr(self, '_pan_cx', 0.5))
            cy = float(getattr(self, '_pan_cy', 0.5))
            cx = max(vw/2.0, min(1.0 - vw/2.0, cx))
            cy = max(vh/2.0, min(1.0 - vh/2.0, cy))
            self._pan_cx = cx
            self._pan_cy = cy
        except Exception:
            self._pan_cx, self._pan_cy = 0.5, 0.5


    def _apply_zoom_and_pan(self, pm):
        # Crop for zoom>1 before scaling (used in Fit/Fill/Full modes)
        try:
            if pm is None or pm.isNull():
                # draw background logo if empty
                try:
                    self._render_background_logo()
                except Exception:
                    pass
                return pm
            z = float(getattr(self, '_zoom', 1.0) or 1.0)
            if z <= 1.0:
                return pm
            w = pm.width(); h = pm.height()
            vw = int(max(1, round(w / z)))
            vh = int(max(1, round(h / z)))
            self._clamp_pan()
            cx = self._pan_cx; cy = self._pan_cy
            left = int(round(cx * w - vw / 2.0))
            top  = int(round(cy * h - vh / 2.0))
            left = max(0, min(w - vw, left))
            top  = max(0, min(h - vh, top))
            return pm.copy(left, top, vw, vh)
        except Exception:
            return pm

    def _center_zoom(self, pm, target):
        # Center mode: z<1 scales down (no crop); z>1 scales up then crops to label size
        try:
            if pm is None or pm.isNull():
                # draw background logo if empty
                try:
                    self._render_background_logo()
                except Exception:
                    pass
                return pm
            z = float(getattr(self, '_zoom', 1.0) or 1.0)
            from PySide6.QtCore import QUrl, QSize
            if z < 1.0:
                sw = max(1, int(round(pm.width() * z)))
                sh = max(1, int(round(pm.height() * z)))
                return pm.scaled(QSize(sw, sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if z == 1.0:
                return pm
            sw = max(1, int(round(pm.width() * z)))
            sh = max(1, int(round(pm.height() * z)))
            spm = pm.scaled(QSize(sw, sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            tw = max(1, int(self.label.contentsRect().width()))
            th = max(1, int(self.label.contentsRect().height()))
            self._clamp_pan()
            cx = self._pan_cx * spm.width()
            cy = self._pan_cy * spm.height()
            left = int(round(cx - tw/2)); top = int(round(cy - th/2))
            left = max(0, min(spm.width() - tw, left))
            top  = max(0, min(spm.height() - th, top))
            if spm.width() <= tw and spm.height() <= th:
                return spm
            return spm.copy(left, top, min(tw, spm.width()), min(th, spm.height()))
        except Exception:
            return pm

    def _show_zoom_overlay(self, pos=None):
        """Show a floating reset button near the cursor with current zoom; auto-hide after 1.5s."""
        try:
            if not getattr(self, '_zoomOverlayBtn', None):
                return
            z = float(getattr(self, '_zoom', 1.0) or 1.0)
            self._zoomOverlayBtn.setText(f"{z:.2f}×  Reset")
            if pos is None:
                pos = self.label.rect().center()
            try:
                x, y = (pos.x(), pos.y()) if hasattr(pos, 'x') else (int(pos[0]), int(pos[1]))
            except Exception:
                x = self.label.rect().center().x(); y = self.label.rect().center().y()
            x += 12; y += 12
            w = self._zoomOverlayBtn.sizeHint().width(); h = self._zoomOverlayBtn.sizeHint().height()
            if x + w > self.label.width() - 6: x = self.label.width() - w - 6
            if y + h > self.label.height() - 6: y = self.label.height() - h - 6
            if x < 6: x = 6
            if y < 6: y = 6
            self._zoomOverlayBtn.move(int(x), int(y))
            self._zoomOverlayBtn.show(); self._zoomOverlayBtn.raise_()
            if getattr(self, '_zoomOverlayTimer', None):
                self._zoomOverlayTimer.start(1500)
        except Exception:
            pass

    # --- Tool video output override (for temporary UI takeovers) ---
    def set_video_output_override(self, mode: str, target) -> None:
        """mode='sink' uses QVideoSink. mode='output' uses Qt's video output (e.g., QGraphicsVideoItem)."""
        try:
            self._video_output_override = (str(mode or ""), target)
        except Exception:
            self._video_output_override = None
        try:
            m = str(mode or "")
            if m == "sink":
                try:
                    self.player.setVideoOutput(None)
                except Exception:
                    pass
                self.player.setVideoSink(target)
            elif m == "output":
                try:
                    self.player.setVideoSink(None)
                except Exception:
                    pass
                self.player.setVideoOutput(target)
        except Exception:
            pass

    def clear_video_output_override(self) -> None:
        try:
            self._video_output_override = None
        except Exception:
            pass
        try:
            try:
                self.player.setVideoOutput(None)
            except Exception:
                pass
            self.player.setVideoSink(self.sink)
        except Exception:
            pass


    def __init__(self,parent=None):
        # TODO: Future: audio visualizer / track info / thumbnail for audio-only playback
        # FPS throttle stripped — keep inert placeholders
        self._render_timer = None
        self._target_fps = None
        # Playback FPS cap: keep original FPS for info display, but cap render/processing.
        self._fps_cap = 30.0
        self._fps_target = 30.0
        self._src_fps = None
        self._last_frame_accept_ts = 0.0
        self._last_present_ts = 0.0
        self._compare_last_accept_ts = 0.0
        self._compare_last_present_ts = 0.0
        super().__init__(parent)
        # Compare: default scaling mode (used by the picker dialog)
        try:
            self._compare_scale_mode_default = str(config.get("compare_scale_mode", "fill") or "fill").strip().lower()
        except Exception:
            self._compare_scale_mode_default = "fill"
        self._video_output_override = None
        self.player = QMediaPlayer(self); self.audio = QAudioOutput(self); self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self); self.player.setVideoSink(self.sink); self.sink.videoFrameChanged.connect(self._on_frame)
        # Autoplay state
        self._autoplay_request = False
        try:
            self.player.mediaStatusChanged.connect(self._on_media_status)
        except Exception:
            pass

        self.currentFrame=None
        self.label=QLabel("FrameVision — Drop a video here or File → Open"); self.label.setAlignment(Qt.AlignCenter); self.label.setMinimumSize(1,1)
        # background logo path selected at app start
        try:
            self._bg_logo_path = STARTUP_LOGO_PATH
        except Exception:
            self._bg_logo_path = None
        # show empty background initially
        try:
            self._show_empty_background()
        except Exception:
            pass

        try:
            self.label.mouseDoubleClickEvent = lambda e: (self.toggle_fullscreen())
        except Exception:
            pass

        self.label.installEventFilter(self)
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,0); self.slider.sliderMoved.connect(self.seek)
        # Compare wipe slider (hidden unless compare mode is active)
        self.compare_slider = QSlider(Qt.Horizontal)
        self.compare_slider.setRange(0, 1000)
        self.compare_slider.setValue(500)
        self.compare_slider.setToolTip("Compare wipe (left ↔ right)")
        try:
            self.compare_slider.sliderMoved.connect(self._on_compare_wipe_moved)
            self.compare_slider.valueChanged.connect(self._on_compare_wipe_changed)
        except Exception:
            pass
        try:
            self.compare_slider.hide()
        except Exception:
            pass
        try:
            self.slider.setStyleSheet('QSlider::groove:horizontal{height:12px;} QSlider::sub-page:horizontal{height:12px;} QSlider::handle:horizontal{width:18px;height:18px;margin:-7px 0;}')
        except Exception:
            pass
        self.player.positionChanged.connect(self.on_pos); self.player.durationChanged.connect(self.on_dur)
        try:
            self.player.playbackStateChanged.connect(lambda *_: self._sync_play_button())
            self.player.playbackStateChanged.connect(self._on_playback_state)
            self._sync_play_button()
        except Exception:
            pass
        bar = QHBoxLayout()
        self._controls_bar_layout = bar
        self.btn_open=QPushButton("📂"); self.btn_open.setToolTip(""); self.btn_play=QPushButton("▶"); self.btn_play.setToolTip(""); self.btn_pause=QPushButton("▮▮"); self.btn_pause.setToolTip(""); self.btn_pause.setVisible(False); self.btn_pause.setEnabled(False); self.btn_pause.setStyleSheet("background: transparent; border: none; padding: 0;")
        self.btn_stop=QPushButton("■"); self.btn_stop.setToolTip(""); self.btn_info=QPushButton("ℹ"); self.btn_info.setToolTip(""); self.btn_fs=QPushButton("⛶"); self.btn_fs.setToolTip("")
        self.btn_ratio = None  # removed; # ratio button removed
        for b in [self.btn_open, self.btn_play, self.btn_stop, self.btn_info, self.btn_fs]: bar.addWidget(b)
        self.btn_compare = QPushButton("▷│◁"); self.btn_compare.setToolTip(""); bar.addWidget(self.btn_compare)
        # Quick Upscale button
        self.btn_upscale = QPushButton("Upscale"); self.btn_upscale.setToolTip("")
        self.btn_upscale.setObjectName("btn_upscale_quick")
        try:
            self.btn_upscale.setStyleSheet(
                "QPushButton#btn_upscale_quick { padding:4px 14px; background:#3b82f6; color:white; border-radius:8px; font-weight:600; }"
                "QPushButton#btn_upscale_quick:hover { background:#2563eb; }"
            )
        except Exception:
            pass
        bar.addWidget(self.btn_upscale)
        # --- Ask button (opens chat popup) ---  # ASK_INSERTED
        self.btn_ask = QPushButton("Ask Framie")
        self.btn_ask.setObjectName("btn_ask_chat")
        self.btn_ask.setStyleSheet(
            "QPushButton#btn_ask_chat { padding:4px 14px; background:#f59e0b; color:black; border-radius:8px; font-weight:600; }"
            "QPushButton#btn_ask_chat:hover { background:#d97706; }"
        )
        bar.addWidget(self.btn_ask)

        # --- Compact button labels (auto) ---
        try:
            # When the window is narrow, these two buttons can get "half-rendered" text.
            # Swap to a single bold capital letter (U/A) until there's room again.
            self._btn_upscale_full_text = "Upscale"
            self._btn_ask_full_text = "Ask Framie"
            self._btn_repeat_full_text = "Repeat"
            self._btn_repeat_compact_text = "R"
            self._btn_upscale_compact_text = "U"
            self._btn_ask_compact_text = "A"

            self._btn_upscale_style_full = (
                "QPushButton#btn_upscale_quick { padding:4px 14px; background:#3b82f6; color:white; border-radius:8px; font-weight:600; }"
                "QPushButton#btn_upscale_quick:hover { background:#2563eb; }"
            )
            self._btn_upscale_style_compact = (
                "QPushButton#btn_upscale_quick { padding:4px 10px; background:#3b82f6; color:white; border-radius:8px; font-weight:900; }"
                "QPushButton#btn_upscale_quick:hover { background:#2563eb; }"
            )

            self._btn_ask_style_full = (
                "QPushButton#btn_ask_chat { padding:4px 14px; background:#f59e0b; color:black; border-radius:8px; font-weight:600; }"
                "QPushButton#btn_ask_chat:hover { background:#d97706; }"
            )
            self._btn_ask_style_compact = (
                "QPushButton#btn_ask_chat { padding:4px 10px; background:#f59e0b; color:black; border-radius:8px; font-weight:900; }"
                "QPushButton#btn_ask_chat:hover { background:#d97706; }"
            )


            from PySide6.QtCore import QTimer as _QTimer
            _QTimer.singleShot(0, self._update_compact_button_labels)
        except Exception:
            pass


        # Repeat button (video loop)
        self.btn_repeat = QPushButton("Repeat")
        self.btn_repeat.setObjectName("btn_repeat")
        self._repeat_enabled = False
        bar.addWidget(self.btn_repeat)
        self.btn_repeat.setVisible(False)
        try:
            self._update_repeat_style()
        except Exception:
            pass

        # Volume/EQ popup button (new)
        try:
            add_new_volume_popup(self, bar)
        except Exception:
            pass
        try:
            self.btn_repeat.clicked.connect(self._toggle_repeat, Qt.ConnectionType.UniqueConnection)
        except Exception:
            pass
        self.btn_ask.clicked.connect(self._open_ask_popup, Qt.ConnectionType.UniqueConnection)

        # --- Toolbar styling: uniform height, transparent, larger glyphs ---
        try:
            _all_ctrls = [self.btn_open, self.btn_play, self.btn_stop, self.btn_info, self.btn_fs, self.btn_compare]
            for _b in _all_ctrls:
                _b.setFixedHeight(48)
                _b.setFlat(True)
                _b.setAutoDefault(False)
                try: _b.setDefault(False)
                except Exception: pass
            # Sizes: play/pause larger
            self.btn_play.setStyleSheet("font-size: 34px; padding: 0; background: transparent; border: none;")
            for _b in [self.btn_open, self.btn_stop, self.btn_info, self.btn_fs, self.btn_compare]:
                _b.setStyleSheet("font-size: 24px; padding: 0; background: transparent; border: none;")
            # Keep Play/Pause compact so bars don't look like multiple stops
            try: self.btn_play.setFixedWidth(48)
            except Exception: pass
        except Exception:
            pass

        # --- Uniform control sizing: keep all buttons the same height ---
        try:
            _all_ctrls = [self.btn_open, self.btn_play, self.btn_pause, self.btn_stop, self.btn_info, self.btn_ratio, self.btn_fs, self.btn_compare, self.btn_upscale, getattr(self,'btn_ask',None), getattr(self,'btn_repeat',None)]
            for _b in _all_ctrls:
                if _b is None:
                    continue
                _b.setFixedHeight(48)
            # Keep play glyph larger but do NOT change height
            self.btn_play.setStyleSheet("font-size: 30px; padding: 0;")
            # Match Upscale to the raster/row height
            self.btn_upscale.setMinimumHeight(48)
            self.btn_upscale.setMaximumHeight(48)
        except Exception:
            pass


        bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.label, 1)

        self.info_label = QLabel("—")
        self.info_label.setObjectName("videoInfo")
        f = self.info_label.font()
        f.setPointSize(max(10, f.pointSize()))
        self.info_label.setFont(f)
        lay.addWidget(self.info_label)

        # Compare info (only visible in compare mode)
        self.compare_info_label = QLabel("")
        self.compare_info_label.setObjectName("compareInfo")
        try:
            f2 = self.compare_info_label.font()
            f2.setPointSize(max(9, f2.pointSize() - 1))
            self.compare_info_label.setFont(f2)
        except Exception:
            pass
        try:
            self.compare_info_label.setWordWrap(True)
        except Exception:
            pass
        try:
            self.compare_info_label.hide()
        except Exception:
            pass
        lay.addWidget(self.compare_info_label)

        self.time_label = QLabel("—")
        self.time_label.setObjectName("videoTime")
        self.time_label.setFont(f)
        lay.addWidget(self.time_label)

        lay.addWidget(self.slider)
        lay.addWidget(self.compare_slider)
        lay.addLayout(bar)
        self.btn_open.clicked.connect(self._open_via_dialog, Qt.ConnectionType.UniqueConnection); self.btn_play.clicked.connect(self._toggle_play_pause, Qt.ConnectionType.UniqueConnection)
        # pause button hidden; no click
        self.btn_stop.clicked.connect(self._handle_stop, Qt.ConnectionType.UniqueConnection)
        self.btn_info.clicked.connect(self._show_info_popup, Qt.ConnectionType.UniqueConnection)
        # ratio button removed
                # compare button: open in-app dialog (replaces external HTML page)
        try:
            self.btn_compare.clicked.connect(self._open_compare_dialog, Qt.ConnectionType.UniqueConnection)
        except Exception:
            pass
        self.btn_upscale.clicked.connect(self.quick_upscale, Qt.ConnectionType.UniqueConnection); self.btn_fs.clicked.connect(self.toggle_fullscreen, Qt.ConnectionType.UniqueConnection)
        self.is_fullscreen=False
        # Ratio modes: 0=Center, 1=Fit, 2=Fill, 3=Full(stretch)
        self.ratio_mode = 1  # force FIT
        # _update_ratio_button removed
# --- injected: zoom overlay button ---
        try:
            from PySide6.QtWidgets import QToolButton
            self._zoomOverlayBtn = QToolButton(self.label)
            self._zoomOverlayBtn.setAutoRaise(True)
            self._zoomOverlayBtn.setVisible(False)
            self._zoomOverlayBtn.setText("1.00×  Reset")
            self._zoomOverlayBtn.setStyleSheet(
                "QToolButton{background:rgba(20,20,20,180);color:white;border-radius:12px;padding:6px 10px;font-weight:600;}"
                "QToolButton:hover{background:rgba(40,40,40,210);}"
            )
            self._zoomOverlayBtn.clicked.connect(lambda: (self._set_zoom(1.0), self._zoomOverlayBtn.hide()))
        except Exception:
            self._zoomOverlayBtn = None
        try:
            self._zoomOverlayTimer = QTimer(self)
            self._zoomOverlayTimer.setSingleShot(True)
            self._zoomOverlayTimer.timeout.connect(lambda: self._zoomOverlayBtn and self._zoomOverlayBtn.hide())
        except Exception:
            self._zoomOverlayTimer = None
# --- zoom/pan state + mode flag ---
        self._zoom = 1.0
        self._max_zoom =    50.0
        self._zoom_step = 0.50
        self._pan_cx = 0.5
        self._pan_cy = 0.5
        self._dragging = False
        self._last_pos = None
        self._mode = 'video'
        self._compare_dragging = False
        # Accept drops on the entire pane
        self.setAcceptDrops(True)

    
    def _on_frame(self, frame):
        if getattr(self, '_mode', None) != 'video':
            return
        # Lightweight: store the image and schedule a coalesced present (zoom untouched)
        try:
            if not self.isVisible() or not self.label.isVisible():
                return
        except Exception:
            pass

        # Lazy-init presenter flags to avoid __init__ edits
        if not hasattr(self, '_present_pending'):
            self._present_pending = False
        if not hasattr(self, '_present_busy'):
            self._present_busy = False
        # Early FPS cap: drop frames BEFORE converting to QImage (prevents 60/120fps stutter).
        try:
            import time as _t
            _tgt = float(getattr(self, '_fps_target', 30.0) or 30.0)
            if _tgt > 0:
                if not hasattr(self, '_last_frame_accept_ts'):
                    self._last_frame_accept_ts = 0.0
                _now = _t.perf_counter()
                _interval = max(0.001, 1.0 / float(_tgt))
                if (_now - float(self._last_frame_accept_ts or 0.0)) < _interval:
                    return
                self._last_frame_accept_ts = _now
        except Exception:
            pass

        # Use existing render timer if present for FPS cap
        try:
            allow = True
            if getattr(self, '_render_timer', None) is not None:
                elapsed = self._render_timer.elapsed()  # ms
                min_interval = int(1000 / max(1, int(getattr(self, '_target_fps', 30))))
                if elapsed < min_interval:
                    allow = False
                else:
                    self._render_timer.restart()
            if not allow:
                return
        except Exception:
            pass

        # Convert to QImage quickly and release frame
        try:
            img = frame.toImage()
            if img and not img.isNull():
                self.currentFrame = img
        except Exception:
            return

        # Coalesce UI updates
        if self._present_pending:
            return
        self._present_pending = True
        try:
            from PySide6.QtCore import QUrl, QTimer
            QTimer.singleShot(0, self._present_current_frame)
        except Exception:
            try:
                self._present_current_frame()
            except Exception:
                pass

    def _present_current_frame(self):
        # Present the current frame; keep zoom logic intact
        if not hasattr(self, '_present_pending'):
            self._present_pending = False
        # --- FPS throttle ---
        try:
            from PySide6.QtCore import QTimer
            import time as _t
            if not hasattr(self, '_fps_target') or not self._fps_target:
                self._fps_target = 30  # default cap
            if not hasattr(self, '_last_present_ts'):
                self._last_present_ts = 0.0
            _now = _t.perf_counter()
            _interval = max(0.001, 1.0 / float(self._fps_target))
            _elapsed = _now - float(self._last_present_ts or 0.0)
            if _elapsed < _interval:
                _ms = int((_interval - _elapsed) * 1000)
                QTimer.singleShot(max(0, _ms), self._present_current_frame)
                return
            self._last_present_ts = _now
        except Exception:
            pass
        if not hasattr(self, '_present_busy'):
            self._present_busy = False
        self._present_pending = False
        if self._present_busy:
            return
        self._present_busy = True
        try:
            self._refresh_label_pixmap()
        except Exception:
            pass
        finally:
            self._present_busy = False
    # --- Background logo helpers ---
    def _clear_video_sink(self):
        try:
            self.currentFrame = None
        except Exception:
            pass
        try:
            from PySide6.QtMultimedia import QVideoFrame
            self.sink.blockSignals(True)
            self.sink.setVideoFrame(QVideoFrame())
            self.sink.blockSignals(False)
        except Exception:
            pass

    def _render_background_logo(self):
        try:
            # only when no image/video is active
            # PATCH 2025-10-03: allow background to render whenever there is NO active image frame or video frame,
            # even if _mode was left as 'video' (matches behavior after pressing Stop).
            active_image = (getattr(self, '_mode', None) == 'image' and getattr(self, '_image_pm_orig', None) is not None)
            active_video = False
            try:
                cf = getattr(self, 'currentFrame', None)
                active_video = (cf is not None and not cf.isNull())
            except Exception:
                active_video = False
            if active_image or active_video:
                return
            p = getattr(self, '_bg_logo_path', None)
            if not p:
                # Text-only fallback so the player never goes blank.
                try:
                    self.label.setMovie(None)
                except Exception:
                    pass
                try:
                    # Setting text clears any pixmap (if present).
                    self.label.setText("Drop a media file here")
                except Exception:
                    pass
                return
            pm = load_pixmap(p)
            if not pm or pm.isNull():
                return
            # stretch to fill label completely
            target = self.label.contentsRect().size()
            from PySide6.QtCore import QSize
            spm = pm.scaled(QSize(max(1,target.width()), max(1,target.height())),
                            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            # apply 50% opacity
            from PySide6.QtGui import QPixmap as _QPixmap, QPainter as _QPainter
            final = _QPixmap(spm.size())
            final.fill(Qt.transparent)
            _p = _QPainter(final)
            try:
                _p.setOpacity(0.85)
                _p.drawPixmap(0,0,spm)
            finally:
                _p.end()
            self.label.setPixmap(final)
        except Exception:
            pass

    def _show_empty_background(self):
        try:
            self._mode = 'empty'
            # clear any movie overlay
            try:
                self.label.setMovie(None)
            except Exception:
                pass
            self._render_background_logo()
        except Exception:
            pass

    def _handle_stop(self):
        # Guard against EndOfMedia callbacks triggering Repeat while we are intentionally stopping/unloading.
        try:
            self._fv_stop_intent = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(3500, lambda: setattr(self, "_fv_stop_intent", False))
        except Exception:
            pass
        try:
            self.player.stop()
        except Exception:
            pass
        # stop compare-right player too
        try:
            if getattr(self, "_compare_active", False) and getattr(self, "_compare_kind", None) == "video":
                rp = getattr(self, "_compare_right_player", None)
                if rp is not None:
                    rp.stop()
        except Exception:
            pass
        self._clear_video_sink()
        self._show_empty_background()
        # Keep Repeat button in sync with the currently loaded media (don't hide it on Stop).
        try:
            main = self.window()
            cp = getattr(main, "current_path", None)
            is_vid = False
            if cp:
                try:
                    from pathlib import Path as _P
                    ext = _P(str(cp)).suffix.lower()
                    try:
                        is_vid = ext in set([str(x).lower() for x in list(VIDEO_EXTS)])
                    except Exception:
                        is_vid = ext in {".mp4",".mov",".mkv",".avi",".webm",".m4v"}
                except Exception:
                    is_vid = False
            self._set_repeat_available(bool(is_vid))
        except Exception:
            pass

    def _on_media_status(self, status):
        try:
            from PySide6.QtMultimedia import QMediaPlayer as _QMediaPlayer
            if status == _QMediaPlayer.EndOfMedia:
                try:
                    if bool(getattr(self, "_fv_stop_intent", False)):
                        return
                except Exception:
                    pass
                try:
                    if bool(getattr(self, "_repeat_enabled", False)) and getattr(self, "_mode", None) == "video":
                        self.player.setPosition(0)
                        self.player.play()
                        return
                except Exception:
                    pass
                self._clear_video_sink()
                self._show_empty_background()
        except Exception:
            # be permissive
            pass


    def mousePressEvent(self, ev):
        # Disable click-to-toggle playback; allow normal event propagation
        try:
            QWidget.mousePressEvent(self, ev)
        except Exception:
            pass


    def on_pos(self,pos): self.slider.setValue(pos); self._update_time_label()
    def on_dur(self,dur): self.slider.setRange(0,dur if dur>0 else 0); self._update_time_label()
    def seek(self, pos):
        try:
            self.player.setPosition(pos)
        except Exception:
            return
        # keep compare-right video in sync
        try:
            if getattr(self, "_compare_active", False) and getattr(self, "_compare_kind", None) == "video":
                rp = getattr(self, "_compare_right_player", None)
                if rp is not None:
                    rp.setPosition(pos)
                    try:
                        import time as _t
                        self._compare_sync_soft_until = float(_t.perf_counter()) + 1.2
                    except Exception:
                        pass
        except Exception:
            pass

    def _fmt_time(self, ms):
        try:
            if ms is None:
                return '0:00'
            total = int(ms // 1000)
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            if h:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m}:{s:02d}"
        except Exception:
            return '0:00'

    def _update_ratio_button(self):
        try:
            # icon + tooltip per mode
            m = int(self.ratio_mode)
            mapping = {
                0: ("◻", "Center"),
                1: ("▣", "Fit"),
                2: ("◼", "Fill"),
                3: ("⛶", "Full"),
            }
            txt, tip = mapping.get(m, ("◻", "Center"))
            # ratio button removed
            try:
                pass
            except Exception:
                pass
        except Exception:
            pass

    def _cycle_ratio_mode(self):
        try:
            self.ratio_mode = (int(self.ratio_mode) + 1) % 4
        except Exception:
            self.ratio_mode = 1  # force FIT
        # _update_ratio_button removed

    
    def _choose_scaled(self, pm: QPixmap, target_size):
        try:
            if pm is None or pm.isNull():
                # draw background logo if empty
                try:
                    self._render_background_logo()
                except Exception:
                    pass
                return pm
            return pm.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            return pm
            tw, th = int(target_size.width()), int(target_size.height())
            if tw <= 0 or th <= 0:
                return pm
            mode = int(getattr(self, 'ratio_mode', 0))
            w, h = int(pm.width()), int(pm.height())

            if mode == 3:  # Full (stretch)
                return pm.scaled(target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            if mode == 2:  # Fill (cover)
                sp = pm.scaled(target_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                x = max(0, (sp.width() - tw) // 2)
                y = max(0, (sp.height() - th) // 2)
                x = min(x, max(0, sp.width() - 1))
                y = min(y, max(0, sp.height() - 1))
                cw = min(tw, sp.width())
                ch = min(th, sp.height())
                return sp.copy(x, y, cw, ch)

            if mode == 1:  # Fit (contain)
                return pm.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # mode == 0: Center (downscale-to-fit, no upscaling)
            if w > tw or h > th:
                return pm.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                return pm
        except Exception:
            return pm

        
    def _refresh_label_pixmap(self):
        try:
            pm_left = None
            if getattr(self, '_mode', 'video') == 'image':
                pm_left = getattr(self, '_image_pm_orig', None)
            else:
                if getattr(self, 'currentFrame', None) is not None and not self.currentFrame.isNull():
                    pm_left = QPixmap.fromImage(self.currentFrame)

            if pm_left is None or pm_left.isNull():
                try:
                    self._render_background_logo()
                except Exception:
                    pass
                return

            target = self.label.contentsRect().size()
            tw = max(1, int(target.width()))
            th = max(1, int(target.height()))
            mode = int(getattr(self, 'ratio_mode', 0))
            z = float(getattr(self, '_zoom', 1.0) or 1.0)

            if getattr(self, "_compare_active", False):
                pm_right = None
                kind = getattr(self, "_compare_kind", None)
                try:
                    if kind == "image":
                        pm_right = getattr(self, "_compare_right_image_pm_orig", None)
                    elif kind == "video":
                        rf = getattr(self, "_compare_right_frame", None)
                        if rf is not None and (not rf.isNull()):
                            pm_right = QPixmap.fromImage(rf)
                except Exception:
                    pm_right = None

                if pm_right is not None and (not pm_right.isNull()):
                    def _render(pm):
                        try:
                            if pm is None or pm.isNull():
                                return None
                            if mode == 0:
                                return self._center_zoom(pm, target)
                            cropped = self._apply_zoom_and_pan(pm)
                            return self._choose_scaled(cropped, target)
                        except Exception:
                            return pm

                    sp_left = _render(pm_left)
                    sp_right = _render(pm_right)
                    if sp_left is not None and (not sp_left.isNull()) and sp_right is not None and (not sp_right.isNull()):
                        out = QPixmap(tw, th)
                        try:
                            out.fill(Qt.transparent)
                        except Exception:
                            pass

                        from PySide6.QtGui import QPainter, QColor, QPen
                        from PySide6.QtCore import QRect

                        p = QPainter(out)

                        oxL = int((tw - sp_left.width()) / 2)
                        oyL = int((th - sp_left.height()) / 2)
                        oxR = int((tw - sp_right.width()) / 2)
                        oyR = int((th - sp_right.height()) / 2)

                        # Base: RIGHT on the canvas
                        p.drawPixmap(oxR, oyR, sp_right)

                        # Reveal: LEFT from the left edge (so Left stays on the left, Right on the right)
                        wipe = int(getattr(self, "_compare_wipe", 500) or 500)
                        wipe = max(0, min(1000, wipe))
                        reveal = int(round(sp_left.width() * (wipe / 1000.0)))

                        p.save()
                        p.setClipRect(QRect(oxL, oyL, reveal, sp_left.height()))
                        p.drawPixmap(oxL, oyL, sp_left)
                        p.restore()

                        x = oxL + reveal
                        # Cache divider geometry for hover/drag hit-testing
                        try:
                            self._compare_divider_x = int(x)
                            # Reuse existing variable names; these now represent the LEFT (revealed) pane.
                            self._compare_right_ox = int(oxL)
                            self._compare_right_oy = int(oyL)
                            self._compare_right_w = int(max(1, sp_left.width()))
                            self._compare_right_h = int(max(1, sp_left.height()))
                        except Exception:
                            pass
                        try:
                            pen = QPen(QColor(255, 255, 255, 200))
                            pen.setWidth(2)
                            p.setPen(pen)
                            p.drawLine(int(x), 0, int(x), th)
                        except Exception:
                            pass

                        p.end()

                        if z <= 1.0:
                            self._pan_cx = 0.5; self._pan_cy = 0.5

                        self.label.setPixmap(out)
                        return

            if mode == 0:
                spm = self._center_zoom(pm_left, target)
            else:
                cropped = self._apply_zoom_and_pan(pm_left)
                spm = self._choose_scaled(cropped, target)

            if z <= 1.0:
                self._pan_cx = 0.5; self._pan_cy = 0.5

            self.label.setPixmap(spm)
        except Exception:
            pass

            target = self.label.contentsRect().size()
            mode = int(getattr(self, 'ratio_mode', 0))
            z = float(getattr(self, '_zoom', 1.0) or 1.0)

            if mode == 0:  # Center
                if z < 1.0:
                    # Scale down only, keep centered
                    from PySide6.QtCore import QUrl, QSize
                    sw = max(1, int(round(pm.width() * z)))
                    sh = max(1, int(round(pm.height() * z)))
                    spm = pm.scaled(QSize(sw, sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                elif z == 1.0:
                    spm = pm
                else:
                    # Scale up then crop to the label viewport based on pan
                    from PySide6.QtCore import QUrl, QSize
                    sw = max(1, int(round(pm.width() * z)))
                    sh = max(1, int(round(pm.height() * z)))
                    spm_big = pm.scaled(QSize(sw, sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    tw = max(1, int(self.label.contentsRect().width()))
                    th = max(1, int(self.label.contentsRect().height()))
                    self._clamp_pan()
                    cx = float(getattr(self, '_pan_cx', 0.5)) * spm_big.width()
                    cy = float(getattr(self, '_pan_cy', 0.5)) * spm_big.height()
                    left = int(round(cx - tw / 2.0)); top = int(round(cy - th / 2.0))
                    left = max(0, min(spm_big.width() - tw, left))
                    top = max(0, min(spm_big.height() - th, top))
                    if spm_big.width() <= tw and spm_big.height() <= th:
                        spm = spm_big
                    else:
                        spm = spm_big.copy(left, top, min(tw, spm_big.width()), min(th, spm_big.height()))
            else:
                # Fit/Fill/Full use crop-first then scale
                cropped = self._apply_zoom_and_pan(pm)
                spm = self._choose_scaled(cropped, target)

            # When z < 1, recenter pan and ignore drag (no crop path)
            if z <= 1.0:
                self._pan_cx = 0.5; self._pan_cy = 0.5

            self.label.setPixmap(spm)
        except Exception:
            pass



    def set_info_text(self, text):
        try:
            s = str(text)
            try:
                mode = getattr(self, "_mode", None)
            except Exception:
                mode = None
            if mode == 'image':
                import re as _re
                s = _re.sub(r"\s*•\s*\d+(?:\.\d+)?\s*fps", "", s, flags=_re.IGNORECASE)
                s = _re.sub(r"\s*•\s*\d+(?:\.\d+)?\s*s\b", "", s, flags=_re.IGNORECASE)
                s = _re.sub(r"\s*•\s*•\s*", " • ", s)
                s = s.strip().strip('•').strip()
            self.info_label.setText(s)
        except Exception:
            pass

    def _update_time_label(self):
        try:
            # Hide when showing a still image
            try:
                if getattr(self, '_mode', None) == 'image':
                    try:
                        self.time_label.setText("")
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            pos = self.player.position() or 0
            dur = self.player.duration() or 0
            left = max(0, dur - pos)
            pct = int(round((pos / dur) * 100)) if dur else 0
            text = f"{self._fmt_time(pos)} / {self._fmt_time(left)} left" + (f" • {pct}%" if dur else "")
            try:
                self.time_label.setText(text)
            except Exception:
                pass
            try:
                if getattr(self, '_fs_time_lbl', None) is not None:
                    self._fs_time_lbl.setText(text)
                if getattr(self, '_fs_slider', None) is not None:
                    self._fs_slider.setRange(0, dur)
                    if not self._fs_slider.isSliderDown():
                        self._fs_slider.setValue(pos)
                if getattr(self, '_fs_btn_pp', None) is not None:
                    self._fs_btn_pp.setText("⏸" if self.player.playbackState()==QMediaPlayer.PlayingState else "▶")
            except Exception:
                pass
        except Exception:
            pass


    def _open_via_dialog(self):
            """Open File dialog via MainWindow (robust to layout changes)."""
            # Prefer the top-level window (MainWindow)
            try:
                w = self.window()
                if w is not None and hasattr(w, "open_file"):
                    return w.open_file()
            except Exception:
                pass

            # Fall back to stored main reference (some tools attach self.main)
            try:
                m = getattr(self, "main", None)
                if m is not None and hasattr(m, "open_file"):
                    return m.open_file()
            except Exception:
                pass

            # Walk parent chain (handles reparented layouts/splitters)
            try:
                p = self.parent()
                hops = 0
                while p is not None and hops < 16:
                    if hasattr(p, "open_file"):
                        try:
                            return p.open_file()
                        except Exception:
                            pass
                    p = p.parent()
                    hops += 1
            except Exception:
                pass

            # Last resort: open directly and route into this VideoPane
            try:
                d = config.get("last_open_dir", str(ROOT))
                flt = ("Media files (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.bmp *.webp "
                       "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff);;"
                       "Video files (*.mp4 *.mov *.mkv *.avi *.webm);;"
                       "Images (*.png *.jpg *.jpeg *.bmp *.webp);;"
                       "Audio files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff);;"
                       "All files (*.*)")
                fn, _ = QFileDialog.getOpenFileName(self, "Open media", d, flt)
                if fn:
                    p = Path(fn)
                    config["last_open_dir"] = str(p.parent)
                    try:
                        save_config()
                    except Exception:
                        pass
                    try:
                        w = self.window()
                        if w is not None and hasattr(w, "current_path"):
                            w.current_path = p
                    except Exception:
                        pass
                    try:
                        self.open(p)
                    except Exception:
                        pass
            except Exception:
                pass
    
    def open(self, path: Path):
        # Unified open: images -> QPixmap (GIF -> QMovie), audio -> QMediaPlayer, video -> QMediaPlayer
        try:
            p = Path(path)
        except Exception:
            from pathlib import Path as _P
            p = _P(str(path))
        ext = p.suffix.lower()
        # If compare mode is active, opening a new file should exit compare.
        # IMPORTANT: stop/clear first, then defer the open to the next tick. This avoids
        # QtMultimedia backend stalls/crashes when swapping sources while the compare
        # right-player is still winding down.
        try:
            if getattr(self, "_compare_active", False) and not getattr(self, "_compare_opening", False):
                try:
                    # Matches the manual workaround ("Stop" first), but automatic.
                    self._handle_stop()
                except Exception:
                    try:
                        self.player.stop()
                    except Exception:
                        pass
                try:
                    self.close_compare()
                except Exception:
                    try:
                        self._compare_active = False
                    except Exception:
                        pass
                try:
                    from PySide6.QtCore import QTimer
                    _pp = str(path)
                    QTimer.singleShot(0, lambda _p=_pp: self.open(_p))
                    return
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self._set_repeat_available(False)
        except Exception:
            pass

        try:
            _w = self.window() if hasattr(self, 'window') else None
            if _w is not None and hasattr(_w, 'current_path'):
                _w.current_path = p
                if hasattr(_w, 'hud') and hasattr(_w.hud, 'set_info'):
                    try: _w.hud.set_info(p)
                    except Exception: pass
                try:
                    if 'compose_video_info_text' in globals():
                        self.set_info_text(compose_video_info_text(p))
                    elif 'compose_image_info_text' in globals():
                        self.set_info_text(compose_image_info_text(p))
                    else:
                        self.set_info_text(str(p))
                except Exception: pass
        except Exception: pass

        # Animated GIF -> QMovie on label
        if ext == '.gif':
            try: self._autoplay_request = False; self._mode = 'image'; self.currentFrame = None
            except Exception: pass
            try: self.player.stop()
            except Exception: pass
            # --- HARD REBUILD to fully release multimedia threads before switching to stills ---
            try:
                self._rebuild_player()
            except Exception:
                pass
            # Clear any lingering GIF/movie overlay
            try:
                self.label.setMovie(None)
            except Exception:
                pass
            try:
                if hasattr(self, "_gif_movie") and self._gif_movie:
                    self._gif_movie.stop()
                    self._gif_movie = None
            except Exception:
                pass

            try: self.player.setSource(QUrl())
            except Exception: pass
            # Clear video sink
            try:
                self.sink.blockSignals(True)
                try:
                    from PySide6.QtMultimedia import QVideoFrame
                    self.sink.setVideoFrame(QVideoFrame())
                except Exception: pass
            except Exception: pass
            try:
                from PySide6.QtGui import QMovie
                self._gif_movie = QMovie(str(p))
                try: self._gif_movie.setCacheMode(QMovie.CacheAll)
                except Exception: pass
                try: self.label.setScaledContents(False)
                except Exception: pass
                self.label.setMovie(self._gif_movie)
                self._gif_movie.start()
            except Exception:
                # Fallback: first frame as still image
                pm = load_pixmap(p)
                if pm and not pm.isNull():
                    self.label.setMovie(None)
                    self.label.setPixmap(pm)
                    try: self._refresh_label_pixmap()
                    except Exception: pass
                else:
                    self.label.setText(f"Cannot display: {p.name}")
            try: self.slider.setEnabled(False)
            except Exception: pass
            try: self.sink.blockSignals(False)
            except Exception: pass
            return

        # Still images (non-animated)
        if ext in IMAGE_EXTS:
            try: self._autoplay_request = False; self._mode = 'image'; self.currentFrame = None
            except Exception: pass
            try: self.player.stop()
            except Exception: pass
            # --- HARD REBUILD to fully release multimedia threads before switching to stills ---
            try:
                self._rebuild_player()
            except Exception:
                pass
            # Clear any lingering GIF/movie overlay
            try:
                self.label.setMovie(None)
            except Exception:
                pass
            try:
                if hasattr(self, "_gif_movie") and self._gif_movie:
                    self._gif_movie.stop()
                    self._gif_movie = None
            except Exception:
                pass

            try: self.player.setSource(QUrl())
            except Exception: pass
            try:
                self.sink.blockSignals(True)
                try:
                    from PySide6.QtMultimedia import QVideoFrame
                    self.sink.setVideoFrame(QVideoFrame())
                except Exception: pass
            except Exception: pass
            pm = load_pixmap(p)
            if pm and not pm.isNull():
                self._image_pm_orig = pm
                try: self.label.setScaledContents(False)
                except Exception: pass
                try: self.label.setAlignment(Qt.AlignCenter)
                except Exception: pass
                self.label.setPixmap(pm)
                try: self._refresh_label_pixmap()
                except Exception: pass
            else:
                self.label.setText(f"Cannot display: {p.name}")
            try: self.slider.setEnabled(False)
            except Exception: pass
            try: self.sink.blockSignals(False)
            except Exception: pass
            return

        # Audio
        if ext in AUDIO_EXTS:
            try: self.slider.setEnabled(True)
            except Exception: pass
            self._mode = 'video'
            self.label.setMovie(None)
            self._rebuild_player()
            self.player.setSource(QUrl.fromLocalFile(str(p)))
            try:
                self.player.play()
            except Exception:
                pass
            try: self.label.setText(p.name)
            except Exception: pass
            return

        # Video
        try: self.slider.setEnabled(True)
        except Exception: pass
        try:
            self._set_repeat_available(True)
        except Exception:
            pass
        try: self._image_pm_orig = None
        except Exception: pass
        
        try: self._mode = 'video'
        except Exception: pass
        # --- Playback FPS cap (max 30fps) ---
        try:
            info = probe_media(p)
            _src = info.get('fps', None)
            try:
                self._src_fps = float(_src) if _src is not None else None
            except Exception:
                self._src_fps = None
            cap = float(getattr(self, '_fps_cap', 30.0) or 30.0)
            tgt = cap
            try:
                if self._src_fps is not None and float(self._src_fps) > 0:
                    tgt = min(cap, float(self._src_fps))
            except Exception:
                tgt = cap
            try:
                self._fps_target = float(tgt) if tgt else cap
            except Exception:
                self._fps_target = cap
            try:
                self._target_fps = int(round(float(self._fps_target)))
            except Exception:
                self._target_fps = 30
            # Reset throttle timestamps so the new cap applies immediately.
            try: self._last_frame_accept_ts = 0.0
            except Exception: pass
            try: self._last_present_ts = 0.0
            except Exception: pass
            try: self._compare_last_accept_ts = 0.0
            except Exception: pass
            try: self._compare_last_present_ts = 0.0
            except Exception: pass
        except Exception:
            try:
                self._fps_target = float(getattr(self, '_fps_cap', 30.0) or 30.0)
            except Exception:
                self._fps_target = 30.0
        try:
            # Ensure label is not stuck on a previous still/GIF overlay
            self.label.setMovie(None)
        except Exception: pass
        try:
            from PySide6.QtGui import QPixmap as _QPM
            # Clear old still/GIF frame and show a background while the first video frame arrives.
            self.label.setPixmap(_QPM())
            try:
                self._render_background_logo()
            except Exception:
                try:
                    self.label.setText("Loading…")
                except Exception:
                    pass
        except Exception:
            pass
        # Stop any active GIF/QMovie to prevent an overlay from sticking
        try:
            if hasattr(self, "_gif_movie") and self._gif_movie:
                self._gif_movie.stop()
                self._gif_movie = None
        except Exception:
            pass

        self._rebuild_player()
        self.player.setSource(QUrl.fromLocalFile(str(p)))
        try: self.player.play()
        except Exception: pass
    def play(self):
        """Play the main media; if compare-video is active, keep the right player in lockstep."""
        try:
            self.player.play()
        except Exception:
            pass
        try:
            if getattr(self, "_compare_active", False) and getattr(self, "_compare_kind", None) == "video":
                rp = getattr(self, "_compare_right_player", None)
                if rp is not None:
                    try:
                        rp.setPlaybackRate(self.player.playbackRate())
                    except Exception:
                        pass
                    try:
                        rp.setPosition(self.player.position())
                    except Exception:
                        pass
                    try:
                        rp.play()
                    except Exception:
                        pass
                    try:
                        from PySide6.QtCore import QTimer
                        import time as _t
                        # Tighten sync for the first moments after (re)start; QMediaPlayer often starts a few frames late.
                        self._compare_sync_soft_until = float(_t.perf_counter()) + 1.6
                        def _nudge_sync():
                            try:
                                if getattr(self, "_compare_active", False) and getattr(self, "_compare_kind", None) == "video":
                                    _rp2 = getattr(self, "_compare_right_player", None)
                                    if _rp2 is not None:
                                        _rp2.setPosition(int(self.player.position() or 0))
                            except Exception:
                                pass
                        QTimer.singleShot(60, _nudge_sync)
                        QTimer.singleShot(140, _nudge_sync)
                        QTimer.singleShot(260, _nudge_sync)
                    except Exception:
                        pass

                try:
                    self._compare_begin_video_sync()
                except Exception:
                    pass
        except Exception:
            pass


    def pause(self):
        self.player.pause()
        # pause compare-right video too (muted)
        try:
            if getattr(self, "_compare_active", False) and getattr(self, "_compare_kind", None) == "video":
                rp = getattr(self, "_compare_right_player", None)
                if rp is not None:
                    rp.pause()
        except Exception:
            pass
        if self.currentFrame is not None: self.frameCaptured.emit(self.currentFrame)

    
    def _on_playback_state(self, state):
        """Ensure video frames show after resume; switch mode from background to video."""
        try:
            from PySide6.QtMultimedia import QMediaPlayer as _QMP
            if state == _QMP.PlayingState:
                try: self._mode = 'video'
                except Exception: pass
                # Make sure no GIF/movie or background pixmap is pinning the label
                try: self.label.setMovie(None)
                except Exception: pass
                try:
                    from PySide6.QtGui import QPixmap as _QPM
                    if getattr(self, 'currentFrame', None) is not None and not self.currentFrame.isNull():
                        self.label.setPixmap(_QPM.fromImage(self.currentFrame))
                except Exception: pass
                try: self._refresh_label_pixmap()
                except Exception: pass
        except Exception:
            pass
    
    def _sync_play_button(self):
        try:
            icon_play = "▶"
            icon_pause = "▮▮"
            if self.player.playbackState() == QMediaPlayer.PlayingState:
                self.btn_play.setText(icon_pause)
            else:
                self.btn_play.setText(icon_play)
        except Exception:
            pass

    def _toggle_play_pause(self):
        try:
            if self.player.playbackState() == QMediaPlayer.PlayingState:
                self.pause()
            else:
                self.play()
            try: self._mode = 'video'
            except Exception: pass
            try: self.label.setMovie(None)
            except Exception: pass
            self._sync_play_button()
        except Exception:
            pass



    def _show_info_popup(self):
        main = self.window()
        p = getattr(main, "current_path", None)
        if not p:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Info", "No media loaded.")
            except Exception:
                pass
            return
        try:
            data = probe_media_all(Path(str(p)))
            show_info_popup(self, data)
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Info", str(e))
            except Exception:
                pass

    def quick_upscale(self):
        """Queue a quick 4x upscale of the current media using RealESRGAN-general-x4v3."""
        try:
            main = self.window()
            p = getattr(main, "current_path", None)
            if not p:
                try:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Upscale", "No media loaded.")
                except Exception:
                    pass
                return
            from pathlib import Path as _P
            p = _P(str(p))
            try:
                try:
                    from helpers.queue_adapter import enqueue, default_outdir
                except Exception:
                    from queue_adapter import enqueue, default_outdir
                outdir = default_outdir(True, 'upscale')
                enqueue('upscale_video', str(p), outdir, 4, 'RealESRGAN-general-x4v3')
                # Notify
                try:
                    try:
                        from helpers.toast import Toast
                    except Exception:
                        Toast = None
                    if Toast:
                        Toast("Queued for upscale", 2000).show_at(main)
                    else:
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.information(self, "Upscale", "Queued for upscale.")
                except Exception:
                    pass
            except Exception as e:
                try:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Upscale error", str(e))
                except Exception:
                    pass
        except Exception:
            pass


    def toggle_fullscreen(self):
        """Fullscreen ONLY the video label with an overlay control bar (slider + time played/left). Press Esc to exit."""
        try:
            fsw = getattr(self, '_fs_win', None)
            if fsw is not None:
                # Restore label to original layout
                try:
                    lp = getattr(self, '_label_parent', None)
                    ll = getattr(self, '_label_layout', None)
                    li = getattr(self, '_label_index', None)
                    if ll is None and lp is not None:
                        ll = lp.layout()
                    if self.label.parent() is fsw:
                        self.label.setParent(None)
                    if ll is not None:
                        try:
                            if isinstance(li, int) and 0 <= li <= ll.count():
                                idx = li
                            else:
                                idx = 0
                            ll.insertWidget(idx, self.label)
                            try:
                                ll.setStretch(idx, 1)
                            except Exception:
                                pass
                        except Exception:
                            try:
                                ll.addWidget(self.label)
                                try:
                                    ll.setStretch(0, 1)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                finally:
                    # Close fs window and clear refs
                    try: fsw.close()
                    except Exception: pass
                    self._fs_win = None
                    self._fs_slider = None
                    self._fs_time_lbl = None
                    self._fs_btn_pp = None
                    self.is_fullscreen = False
                # ----- Re-center/rescale after exiting fullscreen -----
                try:
                    # Reapply current ratio mode after the label is restored
                    if hasattr(self, 'ratio_mode') and int(self.ratio_mode) == 0:
                        self.label.setScaledContents(False)
                        self.label.setAlignment(Qt.AlignCenter)
                    # Defer refresh until layout has settled
                    QTimer.singleShot(0, self._refresh_label_pixmap)
                except Exception:
                    pass
                return

            # Enter fullscreen
            lp = self.label.parentWidget()
            ll = lp.layout() if lp else None
            li = None
            if ll is not None:
                try:
                    for idx in range(ll.count()):
                        it = ll.itemAt(idx)
                        if it and it.widget() is self.label:
                            li = idx
                            break
                except Exception:
                    pass
            self._label_parent = lp; self._label_layout = ll; self._label_index = li

            fsw = QWidget(None, Qt.Window | Qt.FramelessWindowHint)
            fsw.setObjectName('FrameVisionVideoFullscreen')
            try: fsw.setStyleSheet('#FrameVisionVideoFullscreen { background: black; }')
            except Exception: pass
            vbox = QVBoxLayout(fsw); vbox.setContentsMargins(0,0,0,0); vbox.setSpacing(0)
            # Video area
            self.label.setParent(fsw)
            vbox.addWidget(self.label, 1)
            # Overlay bar
            bar = QWidget(fsw)
            bar.setObjectName('FrameVisionFSBar')
            try: bar.setStyleSheet('#FrameVisionFSBar { background: rgba(0,0,0,140); } QLabel { color: white; }')
            except Exception: pass
            h = QHBoxLayout(bar); h.setContentsMargins(12,8,12,8); h.setSpacing(8)
            sl = QSlider(Qt.Horizontal, bar)
            sl.setRange(0, self.player.duration() or 0)
            lbl = QLabel("0:00 / 0:00 left", bar)
            # --- FS play/pause button ---
            try:
                from PySide6.QtWidgets import QPushButton
                pp = QPushButton("⏸" if self.player.playbackState()==QMediaPlayer.PlayingState else "▶", bar)
                pp.setToolTip("")
                h.insertWidget(0, pp, 0)
                self._fs_btn_pp = pp
                def _fs_sync_pp():
                    try:
                        if getattr(self, '_fs_btn_pp', None) is not None:
                            self._fs_btn_pp.setText("⏸" if self.player.playbackState()==QMediaPlayer.PlayingState else "▶")
                    except Exception:
                        pass
                def _fs_toggle():
                    _fs_bump_activity()
                    try:
                        if self.player.playbackState()==QMediaPlayer.PlayingState:
                            self.pause()
                        else:
                            self.play()
                        _fs_sync_pp()
                    except Exception:
                        pass
                try:
                    pp.clicked.connect(_fs_toggle)
                except Exception:
                    pass
                try:
                    self.player.playbackStateChanged.connect(lambda *_: _fs_sync_pp())
                except Exception:
                    pass
            except Exception:
                try:
                    self._fs_btn_pp = None
                except Exception:
                    pass
            h.addWidget(sl, 1); h.addWidget(lbl, 0)
            vbox.addWidget(bar, 0)
            self._fs_slider = sl; self._fs_time_lbl = lbl

            # --- FS overlay auto-hide on inactivity (6s) with fade (no layout flicker) ---
            try:
                INACT_MS = 6000
                self._fs_bar = bar
                from PySide6.QtCore import QTimer
                self._fs_inactivity_timer = QTimer(fsw)
                self._fs_inactivity_timer.setSingleShot(True)
                def _fs_set_visible(_show: bool):
                    try:
                        # Keep bar in layout; fade via opacity and pass-through mouse when hidden
                        try:
                            from PySide6.QtWidgets import QGraphicsOpacityEffect
                            from PySide6.QtCore import QPropertyAnimation
                        except Exception:
                            QGraphicsOpacityEffect = None
                            QPropertyAnimation = None
                        try:
                            bar.setVisible(True)  # never hide to avoid overlay flashing
                        except Exception:
                            pass
                        eff = getattr(self, "_fs_opacity_eff", None)
                        if eff is None and QGraphicsOpacityEffect is not None:
                            try:
                                eff = QGraphicsOpacityEffect(bar)
                                bar.setGraphicsEffect(eff)
                                self._fs_opacity_eff = eff
                            except Exception:
                                eff = None
                        anim = getattr(self, "_fs_opacity_anim", None)
                        if anim is None and QPropertyAnimation is not None and eff is not None:
                            try:
                                anim = QPropertyAnimation(eff, b"opacity", fsw)
                                anim.setDuration(180)  # quick fade
                                self._fs_opacity_anim = anim
                            except Exception:
                                anim = None
                        # pass-through mouse when hidden
                        try:
                            bar.setAttribute(Qt.WA_TransparentForMouseEvents, not _show)
                        except Exception:
                            pass
                        # cursor
                        try:
                            (fsw.unsetCursor() if _show else fsw.setCursor(Qt.BlankCursor))
                        except Exception:
                            pass
                        # animate or set opacity immediately
                        try:
                            if anim is not None and eff is not None:
                                try: anim.stop()
                                except Exception: pass
                                try:
                                    start = eff.opacity()
                                except Exception:
                                    start = 1.0
                                end = 1.0 if _show else 0.0
                                try: anim.setStartValue(start)
                                except Exception: pass
                                try: anim.setEndValue(end)
                                except Exception: pass
                                try: anim.start()
                                except Exception: pass
                            elif eff is not None:
                                eff.setOpacity(1.0 if _show else 0.0)
                        except Exception:
                            pass
                    except Exception:
                        pass
                def _fs_bump_activity(*_a, **_k):
                    try:
                        _fs_set_visible(True)
                        self._fs_inactivity_timer.start(INACT_MS)
                    except Exception:
                        pass
                try:
                    self._fs_inactivity_timer.timeout.connect(lambda: _fs_set_visible(False))
                except Exception:
                    pass
            except Exception:
                pass

            # Wire slider to seek

            # Also bump activity on slider and play/pause interactions
            try:
                sl.sliderPressed.connect(_fs_bump_activity)
                sl.sliderMoved.connect(lambda *_: _fs_bump_activity())
                sl.sliderReleased.connect(_fs_bump_activity)
            except Exception:
                pass
            try:
                if getattr(self, "_fs_btn_pp", None) is not None:
                    self._fs_btn_pp.clicked.connect(_fs_bump_activity)
            except Exception:
                pass

            try: sl.sliderMoved.connect(self.seek)
            except Exception: pass

            # Key handling: ONLY Esc exits
            def _keyPress(e):
                try:
                    if e.key() == Qt.Key_Escape:
                        self.toggle_fullscreen()
                        return
                    if e.key() == Qt.Key_Space:
                        if self.player.playbackState()==QMediaPlayer.PlayingState:
                            self.pause()
                        else:
                            self.play()
                        return
                except Exception:
                    pass
                QWidget.keyPressEvent(fsw, e)
            fsw.keyPressEvent = _keyPress
            # Do NOT exit on mouse clicks anymore
            fsw.mousePressEvent = lambda e: QWidget.mousePressEvent(fsw, e)

            # --- Wrap mouse events to track activity for auto-hide ---
            def _fs_mousePress(e):
                try: _fs_bump_activity()
                except Exception: pass
                QWidget.mousePressEvent(fsw, e)
            def _fs_mouseMove(e):
                try: _fs_bump_activity()
                except Exception: pass
                QWidget.mouseMoveEvent(fsw, e)
            try:
                fsw.setMouseTracking(True)
                self.label.setMouseTracking(True)
            except Exception:
                pass
            fsw.mouseMoveEvent = _fs_mouseMove
            fsw.mousePressEvent = _fs_mousePress

            fsw.closeEvent = lambda e: (self.toggle_fullscreen(), e.accept()) if getattr(self, '_fs_win', None) is not None else e.accept()

            fsw.showFullScreen()
            # Start inactivity countdown now that FS overlay is visible
            try:
                _fs_bump_activity()
            except Exception:
                pass

            try:
                from PySide6.QtCore import QUrl, QTimer
                QTimer.singleShot(0, self._refresh_label_pixmap)
            except Exception:
                pass
            self._fs_win = fsw
            self.is_fullscreen = True
            # Sync initial state
            try:
                self._update_time_label()
                if self._fs_slider is not None:
                    self._fs_slider.setRange(0, self.player.duration() or 0)
                    self._fs_slider.setValue(self.player.position() or 0)
            except Exception:
                pass
        except Exception:
            pass
    def eventFilter(self, obj, ev):
        # wheel zoom / right-click ratio / drag pan
        try:
            if obj is self.label:
                t = ev.type()
                # --- Compare divider hover + drag (move the white line) ---
                try:
                    if obj is self.label and getattr(self, "_compare_active", False):
                        divx = getattr(self, "_compare_divider_x", None)
                        ox = getattr(self, "_compare_right_ox", None)
                        rw = getattr(self, "_compare_right_w", None)
                        if divx is not None and ox is not None and rw is not None:
                            # Mouse position in label coords
                            mpos = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                            thresh = 10
                            near = abs(int(mpos.x()) - int(divx)) <= thresh
                            # Drag divider
                            if t == QEvent.MouseMove and getattr(self, "_compare_dragging", False):
                                x = int(mpos.x())
                                x = max(int(ox), min(int(ox) + int(rw), x))
                                wipe = int(round(((x - int(ox)) / max(1, int(rw))) * 1000))
                                wipe = max(0, min(1000, wipe))
                                self._compare_wipe = wipe
                                try:
                                    self.compare_slider.blockSignals(True)
                                    self.compare_slider.setValue(wipe)
                                    self.compare_slider.blockSignals(False)
                                except Exception:
                                    pass
                                try:
                                    self._refresh_label_pixmap()
                                except Exception:
                                    pass
                                return True

                            # Start divider drag on click near the line
                            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton and near:
                                self._compare_dragging = True
                                self.label.setCursor(Qt.SplitHCursor)
                                x = int(mpos.x())
                                x = max(int(ox), min(int(ox) + int(rw), x))
                                wipe = int(round(((x - int(ox)) / max(1, int(rw))) * 1000))
                                wipe = max(0, min(1000, wipe))
                                self._compare_wipe = wipe
                                try:
                                    self.compare_slider.blockSignals(True)
                                    self.compare_slider.setValue(wipe)
                                    self.compare_slider.blockSignals(False)
                                except Exception:
                                    pass
                                try:
                                    self._refresh_label_pixmap()
                                except Exception:
                                    pass
                                return True

                            # End divider drag
                            if t == QEvent.MouseButtonRelease and getattr(self, "_compare_dragging", False):
                                self._compare_dragging = False
                                self.label.setCursor(Qt.ArrowCursor)
                                return True

                            # Hover cursor near divider (only when not panning)
                            if t == QEvent.MouseMove and not getattr(self, "_dragging", False) and not getattr(self, "_compare_dragging", False):
                                try:
                                    self.label.setCursor(Qt.SplitHCursor if near else Qt.ArrowCursor)
                                except Exception:
                                    pass
                except Exception:
                    pass
                if t == QEvent.Wheel:
                    try:
                        dy = ev.angleDelta().y() if hasattr(ev, 'angleDelta') else 0
                        if dy:
                            step = self._zoom_step if dy > 0 else -self._zoom_step
                            self._set_zoom(getattr(self, '_zoom', 1.0) + step)
                        # Show/reset overlay near the cursor
                        try:
                            mpos = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                            self._show_zoom_overlay(mpos)
                        except Exception:
                            self._show_zoom_overlay()
                            return True
                    except Exception:
                        pass
                if t == QEvent.MouseButtonPress:
                    try:
                        if ev.button() == Qt.RightButton:
                            return False  # ratio cycling disabled
                        if ev.button() == Qt.LeftButton and getattr(self, '_zoom', 1.0) > 1.0:
                            self._dragging = True
                            self._last_pos = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                            self.label.setCursor(Qt.ClosedHandCursor)
                            return True
                    except Exception:
                        pass
                if t == QEvent.MouseMove and getattr(self, '_dragging', False):
                    try:
                        cur = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                        last = getattr(self, '_last_pos', None)
                        if last is not None:
                            dx = cur.x() - last.x(); dy = cur.y() - last.y()
                            z = max( 0.5, float(getattr(self, '_zoom', 1.0)))
                            lw = max(1, self.label.width()); lh = max(1, self.label.height())
                            self._pan_cx -= (dx / lw) / z
                            self._pan_cy -= (dy / lh) / z
                            self._clamp_pan()
                            self._last_pos = cur
                            self._refresh_label_pixmap()
                            return True
                    except Exception:
                        pass
                if t == QEvent.MouseButtonRelease and getattr(self, '_dragging', False):
                    try:
                        self._dragging = False
                        self._last_pos = None
                        self.label.setCursor(Qt.ArrowCursor)
                        return True
                    except Exception:
                        pass
            if obj is self.label and ev.type() == QEvent.Resize:
                try:
                    QTimer.singleShot(0, self._refresh_label_pixmap)
                except Exception:
                    QTimer.singleShot(0, self._refresh_label_pixmap)
        except Exception:
            pass
        try:
            return QWidget.eventFilter(self, obj, ev)
        except Exception:
            return False


    def resizeEvent(self, ev):
        try:
            self._refresh_label_pixmap()
        except Exception:
            pass
        try:
            self._update_compact_button_labels()
        except Exception:
            pass
        try:
            # Defer a second pass so layout-adjusted button widths are measured after the resize settles.
            from PySide6.QtCore import QTimer as _QTimer
            if not getattr(self, "_fv_compact_defer_pending", False):
                self._fv_compact_defer_pending = True
                def _fv_defer_compact_labels():
                    try:
                        self._fv_compact_defer_pending = False
                        self._update_compact_button_labels()
                    except Exception:
                        try:
                            self._fv_compact_defer_pending = False
                        except Exception:
                            pass
                _QTimer.singleShot(0, _fv_defer_compact_labels)
        except Exception:
            pass
        # PATCH 2025-10-03: also refresh the idle background logo on resize
        try:
            self._render_background_logo()
        except Exception:
            pass
        try:
            return QWidget.resizeEvent(self, ev)
        except Exception:
            pass


    # --- Drag & Drop on the entire pane ---
    def dragEnterEvent(self, e):
        try:
            md = e.mimeData()
            if md and (md.hasUrls() or md.hasText()):
                e.acceptProposedAction()
            else:
                e.ignore()
        except Exception:
            e.ignore()

    def dragMoveEvent(self, e):
        try:
            md = e.mimeData()
            if md and (md.hasUrls() or md.hasText()):
                e.acceptProposedAction()
            else:
                e.ignore()
        except Exception:
            e.ignore()

    def dropEvent(self, e):
        try:
            md = e.mimeData()
            paths = []
            if md and md.hasUrls():
                for u in md.urls():
                    if u.isLocalFile():
                        paths.append(u.toLocalFile())
            if not paths and md and md.hasText():
                for ln in md.text().splitlines():
                    ln = ln.strip().strip('"')
                    if ln:
                        paths.append(ln)
            if paths:
                from pathlib import Path as _P
                p = _P(paths[0])
                try:
                    self.open(p)
                    w = self.window()
                    if w is not None:
                        setattr(w, 'current_path', p)
                except Exception:
                    pass
                e.acceptProposedAction()
            else:
                e.ignore()
        except Exception:
            e.ignore()

    # -----------------------------
    # Compare tool (in-app)
    # -----------------------------


    def _compare_media_text(self, pth: str) -> str:
        """Return 'filename • WxH' for compare panes (works for images/videos)."""
        try:
            p = Path(str(pth))
            name = p.name
        except Exception:
            return "— • ?x?"
        w = h = None
        try:
            info = probe_media(p)
            w = info.get("width", None)
            h = info.get("height", None)
        except Exception:
            pass
        # Fallback: if probe failed and it's an image, try pixmap dimensions
        try:
            if (w is None or h is None) and p.suffix.lower() in IMAGE_EXTS:
                pm = QPixmap(str(p))
                if not pm.isNull():
                    w, h = pm.width(), pm.height()
        except Exception:
            pass
        try:
            if w is None or h is None:
                return f"{name} • ?x?"
            return f"{name} • {int(w)}x{int(h)}"
        except Exception:
            return f"{name} • ?x?"

    def _update_compare_info_labels(self):
        """Show/hide the compare info label (left/right file + resolution)."""
        try:
            lbl = getattr(self, "compare_info_label", None)
            if lbl is None:
                return

            # Hide the normal single-line info while Compare is active (avoids duplicate info).
            try:
                info_lbl = getattr(self, "info_label", None)
                if info_lbl is not None:
                    if getattr(self, "_compare_active", False):
                        info_lbl.hide()
                    else:
                        info_lbl.show()
            except Exception:
                pass

            if not getattr(self, "_compare_active", False):
                try:
                    lbl.hide()
                    lbl.setText("")
                except Exception:
                    pass
                return

            l = str(getattr(self, "_compare_left_path", "") or "")
            r = str(getattr(self, "_compare_right_path", "") or "")
            lt = self._compare_media_text(l) if l else "— • ?x?"
            rt = self._compare_media_text(r) if r else "— • ?x?"
            try:
                lbl.setText(f"Left: {lt}\nRight: {rt}")
                lbl.show()
            except Exception:
                pass
        except Exception:
            pass
    def _open_compare_dialog(self):
        try:
            from PySide6.QtWidgets import QDialog
            from helpers.compare_dialog import ComparePickDialog
            start_dir = ""
            try:
                start_dir = str(config.get("last_open_dir", ""))
            except Exception:
                start_dir = ""
            dlg = ComparePickDialog(self, start_dir=start_dir)
            if dlg.exec() != QDialog.Accepted:
                return
            left, right, kind, scale_mode = dlg.get_selection()
            if not left or not right or not kind:
                return
            try:
                self._compare_scale_mode_default = str(scale_mode or "fill")
            except Exception:
                self._compare_scale_mode_default = "fill"
            self.open_compare(left, right, kind, scale_mode=scale_mode)
        except Exception:
            pass

    def _on_compare_wipe_moved(self, v):
        try:
            self._compare_wipe = int(v)
        except Exception:
            self._compare_wipe = 500
        try:
            self._refresh_label_pixmap()
        except Exception:
            pass

    def _on_compare_wipe_changed(self, v):
        try:
            self._compare_wipe = int(v)
        except Exception:
            self._compare_wipe = 500
        try:
            self._refresh_label_pixmap()
        except Exception:
            pass

    def _on_compare_right_frame(self, frame):

        # Early FPS cap: drop compare frames BEFORE converting to QImage.
        try:
            import time as _t
            _tgt = float(getattr(self, '_fps_target', 30.0) or 30.0)
            if _tgt > 0:
                if not hasattr(self, '_compare_last_accept_ts'):
                    self._compare_last_accept_ts = 0.0
                _now = _t.perf_counter()
                _interval = max(0.001, 1.0 / float(_tgt))
                if (_now - float(self._compare_last_accept_ts or 0.0)) < _interval:
                    return
                self._compare_last_accept_ts = _now
        except Exception:
            pass
        try:
            img = frame.toImage()
            if img and not img.isNull():
                self._compare_right_frame = img
        except Exception:
            return
        if not hasattr(self, "_compare_present_pending"):
            self._compare_present_pending = False
        if self._compare_present_pending:
            return
        self._compare_present_pending = True
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._present_compare_frame)
        except Exception:
            try:
                self._present_compare_frame()
            except Exception:
                pass

    def _present_compare_frame(self):
        # Present compare frame; throttle to the same FPS cap as the main presenter.
        try:
            self._compare_present_pending = False
        except Exception:
            pass

        # --- FPS throttle ---
        try:
            from PySide6.QtCore import QTimer
            import time as _t
            if not hasattr(self, '_fps_target') or not self._fps_target:
                self._fps_target = 30  # default cap
            if not hasattr(self, '_compare_last_present_ts'):
                self._compare_last_present_ts = 0.0
            _now = _t.perf_counter()
            _interval = max(0.001, 1.0 / float(self._fps_target))
            _elapsed = _now - float(self._compare_last_present_ts or 0.0)
            if _elapsed < _interval:
                _ms = int((_interval - _elapsed) * 1000)
                if _ms > 0:
                    # Coalesce: schedule a later present and don't spam the UI thread.
                    if not getattr(self, "_compare_present_pending", False):
                        self._compare_present_pending = True
                        QTimer.singleShot(max(0, _ms), self._present_compare_frame)
                    return
            self._compare_last_present_ts = _now
        except Exception:
            pass

        try:
            self._refresh_label_pixmap()
        except Exception:
            pass

    def _compare_begin_video_sync(self):
        """Start a lightweight timer that keeps compare-right video synced to the main player."""
        try:
            if not getattr(self, "_compare_active", False) or getattr(self, "_compare_kind", None) != "video":
                return
            from PySide6.QtCore import QTimer
            try:
                import time as _t
                # After opening/resuming compare-video, be more aggressive about drift correction for a short window.
                self._compare_sync_soft_until = float(_t.perf_counter()) + 1.6
            except Exception:
                pass
            t = getattr(self, "_compare_sync_timer", None)
            if t is None:
                t = QTimer(self)
                try:
                    t.setInterval(80)
                except Exception:
                    pass
                try:
                    t.timeout.connect(self._compare_sync_tick)
                except Exception:
                    pass
                self._compare_sync_timer = t
            if not t.isActive():
                t.start()
        except Exception:
            pass

    def _compare_end_video_sync(self):
        try:
            t = getattr(self, "_compare_sync_timer", None)
            if t is not None and t.isActive():
                t.stop()
        except Exception:
            pass

    def _compare_sync_tick(self):
        try:
            if not getattr(self, "_compare_active", False) or getattr(self, "_compare_kind", None) != "video":
                try:
                    self._compare_end_video_sync()
                except Exception:
                    pass
                return

            rp = getattr(self, "_compare_right_player", None)
            if rp is None:
                return

            # Keep playback rate in sync.
            try:
                rp.setPlaybackRate(self.player.playbackRate())
            except Exception:
                pass

            # Mirror play/pause state.
            try:
                from PySide6.QtMultimedia import QMediaPlayer as _QMP
                lp_state = self.player.playbackState()
                rp_state = rp.playbackState()
                if lp_state == _QMP.PlayingState and rp_state != _QMP.PlayingState:
                    rp.play()
                    try:
                        import time as _t
                        self._compare_sync_soft_until = float(_t.perf_counter()) + 1.6
                    except Exception:
                        pass
                elif lp_state != _QMP.PlayingState and rp_state == _QMP.PlayingState:
                    rp.pause()
            except Exception:
                pass

            # Drift correction (skip while the user is scrubbing).
            try:
                if hasattr(self, "slider") and self.slider.isSliderDown():
                    return
            except Exception:
                pass
            try:
                import time as _t
                lp = int(self.player.position() or 0)
                rp_pos = int(rp.position() or 0)
                delta = rp_pos - lp
                now = float(_t.perf_counter())
                soft_until = float(getattr(self, "_compare_sync_soft_until", 0.0) or 0.0)
                thr = 12 if now <= soft_until else 25
                if abs(delta) >= thr:
                    rp.setPosition(lp)
                    # Some backends can "stick" after a position snap; re-issue play if needed.
                    try:
                        from PySide6.QtMultimedia import QMediaPlayer as _QMP
                        if self.player.playbackState() == _QMP.PlayingState:
                            rp.play()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass


    def open_compare(self, left_path: str, right_path: str, kind: str, scale_mode: str = None):
        try:
            try:
                self.close_compare()
            except Exception:
                pass

            self._compare_opening = True
            try:
                self.open(Path(left_path))
            finally:
                self._compare_opening = False

            self._compare_active = True
            self._compare_kind = str(kind).strip().lower()
            self._compare_left_path = str(left_path)
            self._compare_right_path = str(right_path)
            self._compare_wipe = 500

            # Compare scaling mode (normalizes mismatched resolutions/aspect ratios into the same canvas)
            try:
                sm = str(scale_mode or "").strip().lower()
            except Exception:
                sm = ""
            if sm not in ("fill","fit","stretch"):
                try:
                    sm = str(config.get("compare_scale_mode", "fill") or "fill").strip().lower()
                except Exception:
                    sm = "fill"
            if sm not in ("fill","fit","stretch"):
                sm = "fill"
            self._compare_scale_mode = sm
            try:
                config["compare_scale_mode"] = sm
                save_config()
            except Exception:
                pass


            try:
                self.compare_slider.setValue(500)
                self.compare_slider.show()
                self.compare_slider.setEnabled(True)
            except Exception:
                pass

            try:
                self._update_compare_info_labels()
            except Exception:
                pass

            if self._compare_kind == "image":
                try:
                    self._compare_right_image_pm_orig = QPixmap(str(right_path))
                except Exception:
                    self._compare_right_image_pm_orig = None
                try:
                    self.slider.hide()
                except Exception:
                    pass
                try:
                    self._mode = 'image'
                except Exception:
                    pass
            else:
                try:
                    self.slider.show()
                except Exception:
                    pass

                self._compare_right_frame = None
                try:
                    self._compare_right_player = QMediaPlayer(self)
                    self._compare_right_audio = None  # compare-right is silent (prevents audio clock drift)

                    self._compare_right_sink = QVideoSink(self)
                    self._compare_right_player.setVideoSink(self._compare_right_sink)
                    try:
                        self._compare_right_sink.videoFrameChanged.connect(self._on_compare_right_frame)
                    except Exception:
                        pass

                    rp = Path(str(right_path))
                    self._compare_right_player.setSource(QUrl.fromLocalFile(str(rp)))
                    # Ensure initial alignment happens after the backend has actually loaded the media.
                    try:
                        def _rp_loaded(st):
                            try:
                                from PySide6.QtMultimedia import QMediaPlayer as _QMP
                                if st == _QMP.LoadedMedia:
                                    try:
                                        self._compare_right_player.setPosition(int(self.player.position() or 0))
                                    except Exception:
                                        pass
                                    try:
                                        self._compare_right_player.setPlaybackRate(self.player.playbackRate())
                                    except Exception:
                                        pass
                                    try:
                                        if self.player.playbackState() == _QMP.PlayingState:
                                            self._compare_right_player.play()
                                    except Exception:
                                        pass
                                    try:
                                        self._compare_right_player.mediaStatusChanged.disconnect(_rp_loaded)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        self._compare_right_player.mediaStatusChanged.connect(_rp_loaded)
                    except Exception:
                        pass

                    try:
                        self._compare_right_player.setPosition(self.player.position())
                    except Exception:
                        pass
                    try:
                        import time as _t
                        self._compare_sync_soft_until = float(_t.perf_counter()) + 1.6
                    except Exception:
                        pass
                    try:
                        self._compare_right_player.setPlaybackRate(self.player.playbackRate())
                    except Exception:
                        pass
                    try:
                        if self.player.playbackState() == QMediaPlayer.PlayingState:
                            self._compare_right_player.play()
                    except Exception:
                        pass
                except Exception:
                    self._compare_right_player = None
                    self._compare_right_audio = None
                    self._compare_right_sink = None

            try:
                self._refresh_label_pixmap()
            except Exception:
                pass

            try:
                if getattr(self, "_compare_kind", None) == "video":
                    self._compare_begin_video_sync()
            except Exception:
                pass
        except Exception:
            pass

    def close_compare(self):
        try:
            self._compare_active = False
            self._compare_kind = None
        except Exception:
            pass


        try:
            self._update_compare_info_labels()
        except Exception:
            pass


        try:
            self._compare_end_video_sync()
        except Exception:
            pass

        try:
            rp = getattr(self, "_compare_right_player", None)
            if rp is not None:
                try:
                    rp.stop()
                except Exception:
                    pass
                try:
                    rp.setSource(QUrl())
                except Exception:
                    pass
        except Exception:
            pass

        try:
            rs = getattr(self, "_compare_right_sink", None)
            if rs is not None:
                try:
                    rs.videoFrameChanged.disconnect(self._on_compare_right_frame)
                except Exception:
                    pass
        except Exception:
            pass


        # Fully dispose compare-right multimedia objects (avoids leaks / backend freezes).
        try:
            rp = getattr(self, "_compare_right_player", None)
            rs = getattr(self, "_compare_right_sink", None)
            ra = getattr(self, "_compare_right_audio", None)
            if rp is not None:
                try:
                    rp.setVideoSink(None)
                except Exception:
                    pass
                try:
                    rp.setAudioOutput(None)
                except Exception:
                    pass
            for obj in (rs, ra, rp):
                try:
                    if obj is not None:
                        obj.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass


        for k in ["_compare_right_player","_compare_right_audio","_compare_right_sink",
                  "_compare_right_frame","_compare_right_image_pm_orig",
                  "_compare_left_path","_compare_right_path"]:
            try:
                setattr(self, k, None)
            except Exception:
                pass

        try:
            self.compare_slider.hide()
        except Exception:
            pass
        try:
            self.slider.show()
        except Exception:
            pass

        try:
            self._refresh_label_pixmap()
        except Exception:
            pass


class HUD(QWidget):
    def _show_info_popup(self):
        main = self.window()
        p = getattr(main, "current_path", None)
        if not p:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Info", "No media loaded.")
            except Exception:
                pass
            return
        try:
            data = probe_media_all(Path(str(p)))
            show_info_popup(self, data)
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Info", str(e))
            except Exception:
                pass

    def __init__(self, parent=None):
        super().__init__(parent)
        self.info = QLabel(f"{APP_NAME}"); self.hud = QLabel("—")
        v = QHBoxLayout(self); v.addWidget(self.info); v.addStretch(1); v.addWidget(self.hud)
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.setInterval(3000)
        self.path = None
        # Net I/O baseline for DL/UL speed in HUD
        self._net_prev = None  # (bytes_recv, bytes_sent, t)
    def set_info(self, path: Path):
        self.path = path
        # Keep fixed brand on the left; avoid file detail noise
        self.info.setText(f"{APP_NAME}")
        return

    def _spawn_worker(self):
        try:
            # Start helpers/worker.py detached using the same Python executable
            exe = sys.executable
            script = str(ROOT / "helpers" / "worker.py")
            # Avoid spamming: if heartbeat updated in the last 5s, skip
            import time
            try:
                if HEARTBEAT_PATH.exists() and (time.time() - HEARTBEAT_PATH.stat().st_mtime) < 5.0:
                    self.lbl_worker.setText("Worker: already running")
                    return
            except Exception:
                pass
            QProcess.startDetached(exe, [script])
            self.lbl_worker.setText("Worker: launching…")
        except Exception:
            pass

    def refresh(self):
        # Performance safe mode
        safe = False
        try:
            import os
            from PySide6.QtCore import QUrl, QSettings
            env = os.environ.get('FRAMEVISION_SAFE','0').lower() in ('1','true','on','yes')
            ini = str(QSettings('FrameVision','FrameVision').value('perf_safe','0')).lower() in ('1','true','on','yes')
            safe = bool(env or ini)
        except Exception:
            safe = False
        try:
            # CPU and RAM (DDR)
            cpu = int(round(psutil.cpu_percent()))
            mem = psutil.virtual_memory()
            ddr_used_gb = mem.used / (1024**3)
            ddr_total_gb = mem.total / (1024**3)
            ddr_pct = int(round(mem.percent))

            # GPU via nvidia-smi (first GPU)
            gpu_str = "GPU : —"
            try:
                if safe:
                    raise RuntimeError('safe-mode: skip nvidia-smi')
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.STDOUT, text=True, timeout=0.25)
                line = out.splitlines()[0].strip() if out else ""
                if line:
                    total_mib, used_mib, temp_c, util = [x.strip() for x in line.split(",")]
                    total_mib = float(total_mib or 0); used_mib = float(used_mib or 0)
                    temp_c = int(float(temp_c or 0)); util = int(float(util or 0))
                    used_gb = used_mib / 1024.0; total_gb = total_mib / 1024.0
                    pct = util  # show actual GPU load % (utilization), not VRAM usage
                    # Example: GPU : 16.5/24 52% 50C  (52% = GPU load)
                    gpu_str = f"GPU : {used_gb:.1f}/{total_gb:.0f} {pct}% {_format_temp_units(temp_c)}"
            except Exception:
                pass

            ddr_str = f"DDR {ddr_used_gb:.1f}/{ddr_total_gb:.0f} {ddr_pct}%"
            cpu_str = f"CPU {cpu}%"

            # Net download/upload speed (only show if >= 50 KB/s)
            net_part = ""
            try:
                import time as _time
                if psutil:
                    _io = psutil.net_io_counters()
                    _nowt = _time.time()
                    _prev = getattr(self, "_net_prev", None)
                    if not _prev:
                        self._net_prev = (_io.bytes_recv, _io.bytes_sent, _nowt)
                    else:
                        _pr, _ps, _pt = _prev
                        _dt = max(0.5, float(_nowt - float(_pt or _nowt)))
                        _dl_kbs = max(0.0, float(_io.bytes_recv - int(_pr or 0)) / 1024.0 / _dt)
                        _ul_kbs = max(0.0, float(_io.bytes_sent - int(_ps or 0)) / 1024.0 / _dt)
                        self._net_prev = (_io.bytes_recv, _io.bytes_sent, _nowt)

                        def _fmt_rate(_kbs: float) -> str:
                            """Show MB/s once >= 1 MB/s; otherwise show KB/s."""
                            try:
                                _k = float(_kbs)
                                if _k >= 1024.0:
                                    _mbs = _k / 1024.0
                                    _s = f"{_mbs:.1f}"
                                    if _s.endswith(".0"):
                                        _s = _s[:-2]
                                    return f"{_s} MB/s"
                            except Exception:
                                pass
                            try:
                                return f"{int(round(float(_kbs)))} KB/s"
                            except Exception:
                                return f"{_kbs} KB/s"

                        _parts = []
                        if _dl_kbs >= 50.0:
                            _parts.append(f"DL {_fmt_rate(_dl_kbs)}")
                        if _ul_kbs >= 50.0:
                            _parts.append(f"UL {_fmt_rate(_ul_kbs)}")
                        if _parts:
                            net_part = "  " + "  ".join(_parts)
            except Exception:
                pass
            now = datetime.now()
            time_str = f"{now.strftime('%a')}. {now.strftime('%d %b %H:%M')}"

            hud = f"{gpu_str}  {ddr_str}  {cpu_str}{net_part}  {time_str}"
            self.hud.setText(hud)
        except Exception:
            self.hud.setText("HUD unavailable")

# --- Presets / Instant Tools / Describe panes (short versions for size)
    def showEvent(self, e):
        try:
            super().showEvent(e)
        except Exception:
            pass
        # Defer HUD timer start ~1.5s after show
        try:
            from PySide6.QtCore import QUrl, QTimer
            QTimer.singleShot(1500, lambda: self.timer.start(self.timer.interval()))
        except Exception:
            pass

class PresetsPane(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent); self.main = main
        v = QVBoxLayout(self)
        # Resize
        v.addWidget(QLabel("Resize presets"))
        self.r_name = QLineEdit(); self.r_w = QSpinBox(); self.r_h = QSpinBox()
        self.r_w.setRange(16,8192); self.r_h.setRange(16,8192)
        rowr = QHBoxLayout(); [rowr.addWidget(x) for x in (QLabel("Name"), self.r_name, QLabel("W"), self.r_w, QLabel("H"), self.r_h)]
        br = QHBoxLayout(); self.btn_r_save=QPushButton("Save/Update"); self.btn_r_del=QPushButton("Delete"); br.addWidget(self.btn_r_save); br.addWidget(self.btn_r_del)
        v.addLayout(rowr); v.addLayout(br); self.lst_r=QListWidget(); v.addWidget(self.lst_r)
        v.addWidget(QLabel("Export presets (GIF)"))
        self.e_name = QLineEdit(); self.e_gif = QSpinBox(); self.e_gif.setRange(1,60)
        rowe = QHBoxLayout(); [rowe.addWidget(x) for x in (QLabel("Name"), self.e_name, QLabel("GIF fps"), self.e_gif)]
        be = QHBoxLayout(); self.btn_e_save=QPushButton("Save/Update"); self.btn_e_del=QPushButton("Delete"); be.addWidget(self.btn_e_save); be.addWidget(self.btn_e_del)
        v.addLayout(rowe); v.addLayout(be); self.lst_e=QListWidget(); v.addWidget(self.lst_e)
        # Crop
        v.addWidget(QLabel("Crop presets"))
        self.c_name=QLineEdit(); self.c_w=QSpinBox(); self.c_h=QSpinBox(); self.c_x=QSpinBox(); self.c_y=QSpinBox()
        for s in (self.c_w,self.c_h): s.setRange(16,8192)
        for s in (self.c_x,self.c_y): s.setRange(0,10000)
        rowc = QHBoxLayout(); [rowc.addWidget(x) for x in (QLabel("Name"), self.c_name, QLabel("W"), self.c_w, QLabel("H"), self.c_h, QLabel("X"), self.c_x, QLabel("Y"), self.c_y)]
        bc = QHBoxLayout(); self.btn_c_save=QPushButton("Save/Update"); self.btn_c_del=QPushButton("Delete"); bc.addWidget(self.btn_c_save); bc.addWidget(self.btn_c_del)
        v.addLayout(rowc); v.addLayout(bc); self.lst_c=QListWidget(); v.addWidget(self.lst_c)
        # wire
        self.refresh()
        self.btn_r_save.clicked.connect(lambda: self.save_p("resize"))
        self.btn_e_save.clicked.connect(lambda: self.save_p("export"))
        self.btn_c_save.clicked.connect(lambda: self.save_p("crop"))
        self.btn_r_del.clicked.connect(lambda: self.del_p("resize", self.lst_r))
        self.btn_e_del.clicked.connect(lambda: self.del_p("export", self.lst_e))
        self.btn_c_del.clicked.connect(lambda: self.del_p("crop", self.lst_c))
        self.lst_r.itemClicked.connect(lambda i: self.load_into("resize", i))
        self.lst_e.itemClicked.connect(lambda i: self.load_into("export", i))
        self.lst_c.itemClicked.connect(lambda i: self.load_into("crop", i))
    def refresh(self):
        self.lst_r.clear(); self.lst_e.clear(); self.lst_c.clear()
        for k in sorted(presets.get("resize",{}).keys()): self.lst_r.addItem(QListWidgetItem(k))
        for k in sorted(presets.get("export",{}).keys()): self.lst_e.addItem(QListWidgetItem(k))
        for k in sorted(presets.get("crop",{}).keys()):   self.lst_c.addItem(QListWidgetItem(k))
    def save_p(self, key):
        if key=="resize":
            name=self.r_name.text().strip();
            if not name: return
            presets["resize"][name]={"w":int(self.r_w.value()),"h":int(self.r_h.value())}
        elif key=="export":
            name=self.e_name.text().strip();
            if not name: return
            presets["export"][name]={"gif_fps":int(self.e_gif.value())}
        else:
            name=self.c_name.text().strip();
            if not name: return
            presets["crop"][name]={"w":int(self.c_w.value()),"h":int(self.c_h.value()),"x":int(self.c_x.value()),"y":int(self.c_y.value())}
        save_presets(); self.refresh()
    def del_p(self, key, lst):
        it=lst.currentItem()
        if not it: return
        presets[key].pop(it.text(), None); save_presets(); self.refresh()
    def load_into(self, key, it):
        name=it.text(); data=presets.get(key,{}).get(name,{})
        if key=="resize":
            self.r_name.setText(name); self.r_w.setValue(int(data.get("w",1280))); self.r_h.setValue(int(data.get("h",720)))
        elif key=="export":
            self.e_name.setText(name); self.e_gif.setValue(int(data.get("gif_fps",12)))
        else:
            self.c_name.setText(name); self.c_w.setValue(int(data.get("w",1920))); self.c_h.setValue(int(data.get("h",1080))); self.c_x.setValue(int(data.get('x',0))); self.c_y.setValue(int(data.get('y',0)))

# (dedup) CollapsibleSection second definition removed — using the earlier one.

class DescribePane(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent); self.main=main
        v = QVBoxLayout(self)
        sec_model = CollapsibleSection('Model & Options', expanded=True)
        sec_out = CollapsibleSection('Output', expanded=True)
        row = QHBoxLayout(); self.cmb=QComboBox(); self.cmb.addItems(["Model 1","Model 2"])
        self.chk=QCheckBox("Describe on Pause"); self.chk.setChecked(bool(config.get("auto_on_pause", True)))
        btn=QPushButton("Describe current frame")
        row.addWidget(QLabel("Caption model:")); row.addWidget(self.cmb); row.addStretch(1); row.addWidget(self.chk); row.addWidget(btn); lay_model = QVBoxLayout(); lay_model.addLayout(row); w1=QWidget(); w1.setLayout(lay_model); sec_model.setContentLayout(lay_model); v.addWidget(sec_model)
        self.txt=QTextEdit(); lay_out = QVBoxLayout(); lay_out.addWidget(self.txt)
        row2 = QHBoxLayout(); b1=QPushButton("Copy"); b2=QPushButton("Save"); row2.addWidget(b1); row2.addWidget(b2); row2.addStretch(1); lay_out.addLayout(row2); w2=QWidget(); w2.setLayout(lay_out); sec_out.setContentLayout(lay_out); v.addWidget(sec_out)
        btn.clicked.connect(self.describe_now, Qt.ConnectionType.UniqueConnection); b1.clicked.connect(lambda: QApplication.clipboard().setText(self.txt.toPlainText())); b2.clicked.connect(self.save_text, Qt.ConnectionType.UniqueConnection)
    def on_pause_capture(self, qimg: QImage):
        if self.chk.isChecked(): self.describe_now()
    def describe_now(self):
        if self.main.video.currentFrame is None: QMessageBox.information(self,"No frame","Play/pause to capture a frame."); return
        # Save current frame locally (no dependency on VideoPane.screenshot)
        out = OUT_SHOTS / f"{now_stamp()}_shot.png"; out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.main.video.currentFrame.save(str(out))
        except Exception:
            out = None
        base=re.sub(r"[_\-]+"," ", Path(out).stem) if out else Path(getattr(self.main,'current_path','frame')).stem
        ts=datetime.now().strftime("%H:%M")
        if self.cmb.currentText()=="Model 1":
            txt=f"A frame captured at {ts} showing {base}. Clean edges and balanced colors. Details are crisp."
        else:
            txt=(f"A frame captured at {ts} showing {base}. Clean edges and balanced colors. Emphasize motion and light direction. Foreground/background separation seems moderate; textures appear detailed.")
        self.txt.setPlainText(txt)
    def save_text(self):
        text=self.txt.toPlainText().strip();
        if not text: return
        first=re.sub(r"[^a-zA-Z0-9]+","", text.split()[0].lower()) if text.split() else "desc"
        fname=OUT_DESCR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{first}.txt"; fname.write_text(text, encoding="utf-8")
        QMessageBox.information(self,"Saved", str(fname))

# --- Queue Pane (moved to helpers/queue_pane.py)


class SettingsPane(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent); self.main=main
        v=QVBoxLayout(self); row=QHBoxLayout()
        self.cmb=QComboBox(); self.cmb.addItems(["Auto","Day","Evening","Night"]);
        idx=self.cmb.findText(config.get("theme","Auto"));
        if idx>=0: self.cmb.setCurrentIndex(idx)
        btn=QPushButton("Apply theme"); row.addWidget(QLabel("Theme:")); row.addWidget(self.cmb); row.addWidget(btn); row.addStretch(1); v.addLayout(row)

        # Real progress toggle row
        row2=QHBoxLayout()
        btn.clicked.connect(lambda: (config.update({"theme":self.cmb.currentText()}), save_config(), apply_theme(QApplication.instance(), config["theme"])))
        # About area
        about = QLabel(f"<h2>{APP_NAME}</h2><p>{TAGLINE}</p><p>© {datetime.now().year}</p><p>Placeholder — Settings_tab.py did not load or import.</p>")
        v.addWidget(about)

# --- Main Window

# --- Music wiring (audio visuals + playlist) ---
from helpers.music import wire_to_videopane as _fv_wire_music
from tools.diag_probe import wire_to_videopane as _fv_wire_diag
from tools.media_handoff_fix import wire as _fv_wire_handoff
_fv_wire_handoff(VideoPane)
_fv_wire_diag(VideoPane)
_fv_wire_music(VideoPane)

# --- Background Pane (new tab) ------------------------------------------------
class BackgroundPane(QWidget):
    """Wrapper tab that reuses helpers.background.install_background_tool,
    inside a vertical QScrollArea so the UI isn't cramped.
    """
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._container = QWidget(self._scroll)
        self._scroll.setWidget(self._container)

        outer.addWidget(self._scroll)

        try:
            install_background_tool(self, self)
        except Exception as _e:
            try:
                lbl = QLabel(f"Background tool failed to load: {_e}")
                outer.addWidget(lbl)
            except Exception:
                pass

    def setContentLayout(self, layout):
        old = self._container.layout()
        if old is not None:
            QWidget().setLayout(old)
        self._container.setLayout(layout)


class MainWindow(QMainWindow):
    def _install_tab_nav_arrows(self):
        """
        Place left/right tab selectors before the first tab and keep them theme-aware.
        The arrows should only be visible if the tab bar overflows (not all tabs fit).
        """
        try:
            bar = self.tabs.tabBar()
        except Exception:
            return
        try:
            # Disable the built-in scroll arrows so we can provide our own
            bar.setUsesScrollButtons(False)
        except Exception:
            pass
        try:
            # Build a compact container with two animated buttons
            cont = QWidget(self.tabs)
            lay = QHBoxLayout(cont)
            lay.setContentsMargins(6, 0, 6, 0)
            lay.setSpacing(4)
            btn_prev = _TabNavButton("‹", cont)  # U+2039
            btn_next = _TabNavButton("›", cont)  # U+203A
            # Accessible names for testing/automation
            btn_prev.setObjectName("tab_nav_prev")
            btn_next.setObjectName("tab_nav_next")
            lay.addWidget(btn_prev)
            lay.addWidget(btn_next)
            cont.setLayout(lay)
    
            def _go_prev():
                try:
                    i = self.tabs.currentIndex()
                    n = self.tabs.count()
                    if n <= 0:
                        return
                    self.tabs.setCurrentIndex(max(0, i - 1))
                except Exception:
                    pass
    
            def _go_next():
                try:
                    i = self.tabs.currentIndex()
                    n = self.tabs.count()
                    if n <= 0:
                        return
                    self.tabs.setCurrentIndex(min(n - 1, i + 1))
                except Exception:
                    pass
    
            btn_prev.clicked.connect(_go_prev, Qt.ConnectionType.UniqueConnection)
            btn_next.clicked.connect(_go_next, Qt.ConnectionType.UniqueConnection)
    
            # Put them at the very front of the bar (north position assumed)
            try:
                self.tabs.setCornerWidget(cont, Qt.TopLeftCorner)
            except Exception:
                # Fallback for unusual positions
                self.tabs.setCornerWidget(cont)
    
            # Keep references so we can update them later
            self._tabnav_container = cont
            self._tabnav_prev = btn_prev
            self._tabnav_next = btn_next
            self._tabbar_ref = bar
    
            def _tabs_overflow():
                """Return True if total tab widths exceed available bar width."""
                try:
                    total = 0
                    for _i in range(bar.count()):
                        total += bar.tabRect(_i).width()
                    avail = max(1, bar.width())
                    return total > avail
                except Exception:
                    return False
    
            def _sync_state(_=None):
                # Enable/disable arrows based on current tab index
                try:
                    n = self.tabs.count()
                    i = self.tabs.currentIndex()
                    btn_prev.setEnabled(bool(n > 0 and i > 0))
                    btn_next.setEnabled(bool(n > 0 and i < n - 1))
                except Exception:
                    pass
                # Show/hide entire arrow container based on overflow
                try:
                    cont.setVisible(bool(_tabs_overflow()))
                except Exception:
                    pass
    
            # Expose for eventFilter usage
            self._sync_tab_nav_state = _sync_state
    
            _sync_state()
            try:
                bar.tabMoved.connect(_sync_state)
            except Exception:
                pass
            try:
                self.tabs.currentChanged.connect(_sync_state)
            except Exception:
                pass
            # Watch for resize / relayout so visibility reacts to window size
            try:
                bar.installEventFilter(self)
            except Exception:
                pass
        except Exception:
            pass
    

    # ---- Tab reordering (drag tabs + persist order) ------------------------------
    def set_tabs_reorder_enabled(self, enabled: bool):
        """Enable/disable draggable tab reordering."""
        try:
            self.tabs.tabBar().setMovable(bool(enabled))
        except Exception:
            pass

    def _slug_tab_name(self, name: str) -> str:
        name = (name or "").strip().lower()
        if not name:
            return ""
        out = []
        last_us = False
        for ch in name:
            if ch.isalnum():
                out.append(ch)
                last_us = False
            else:
                if not last_us:
                    out.append("_")
                    last_us = True
        slug = "".join(out).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug

    def _ensure_main_tab_object_names(self):
        """Assign stable objectNames to tab pages if missing (used for persistence)."""
        try:
            for i in range(self.tabs.count()):
                w = self.tabs.widget(i)
                if w is None:
                    continue
                try:
                    if w.objectName():
                        continue
                except Exception:
                    pass
                try:
                    t = self.tabs.tabText(i)
                except Exception:
                    t = f"tab_{i}"
                slug = self._slug_tab_name(t)
                if not slug:
                    slug = f"tab_{i}"
                try:
                    w.setObjectName(f"tab_{slug}")
                except Exception:
                    pass
        except Exception:
            pass

    def _tab_id_for_index(self, i: int) -> str:
        try:
            w = self.tabs.widget(i)
        except Exception:
            w = None
        if w is not None:
            try:
                oid = w.objectName()
                if oid:
                    return str(oid)
            except Exception:
                pass
            try:
                return f"tab_{w.__class__.__name__}"
            except Exception:
                pass
        return f"tab_{i}"

    def _current_tab_order_ids(self) -> list[str]:
        ids: list[str] = []
        try:
            for i in range(self.tabs.count()):
                ids.append(self._tab_id_for_index(i))
        except Exception:
            pass
        return ids

    def _find_tab_index_by_id(self, tab_id: str) -> Optional[int]:
        try:
            for i in range(self.tabs.count()):
                w = self.tabs.widget(i)
                if w is None:
                    continue
                try:
                    if str(w.objectName()) == str(tab_id):
                        return i
                except Exception:
                    pass
                # fallback to derived id
                if self._tab_id_for_index(i) == tab_id:
                    return i
        except Exception:
            pass
        return None

    def _apply_tab_order_ids(self, desired_ids: list[str]):
        """Reorder tabs to match desired_ids (unknown ids ignored; missing tabs stay at end)."""
        try:
            bar = self.tabs.tabBar()
        except Exception:
            return
        self._tab_order_applying = True
        try:
            for target, tid in enumerate(desired_ids):
                idx = self._find_tab_index_by_id(tid)
                if idx is None:
                    continue
                if idx != target:
                    try:
                        bar.moveTab(idx, target)
                    except Exception:
                        pass
        finally:
            self._tab_order_applying = False

    def _save_tab_order_ids(self):
        if getattr(self, "_tab_order_applying", False):
            return
        try:
            s = QSettings("FrameVision", "FrameVision")
            s.setValue("main_tabs_order", self._current_tab_order_ids())
        except Exception:
            pass

    def _restore_saved_tab_order_ids(self):
        try:
            s = QSettings("FrameVision", "FrameVision")
            val = s.value("main_tabs_order", None)
            if not val:
                return
            if isinstance(val, (list, tuple)):
                desired = [str(x) for x in val]
            else:
                # Some backends may return a single string; ignore in that case
                return
            self._apply_tab_order_ids(desired)
        except Exception:
            pass

    def _post_init_tab_order_setup(self):
        # Ensure stable ids and capture default order before applying any saved order
        try:
            self._ensure_main_tab_object_names()
            if not hasattr(self, "_default_tab_order_ids"):
                self._default_tab_order_ids = self._current_tab_order_ids()
        except Exception:
            pass
        # Apply saved order (always)
        try:
            self._restore_saved_tab_order_ids()
        except Exception:
            pass
        # Save once so new tabs get persisted even if user never drags
        try:
            self._save_tab_order_ids()
        except Exception:
            pass

    def reset_tab_order_to_default(self):
        """Reset current tabs to the default order from this run, and persist it."""
        try:
            if not hasattr(self, "_default_tab_order_ids") or not self._default_tab_order_ids:
                self._ensure_main_tab_object_names()
                self._default_tab_order_ids = self._current_tab_order_ids()
            self._apply_tab_order_ids(list(self._default_tab_order_ids))
            self._save_tab_order_ids()
        except Exception:
            pass



    def _on_global_video_playback_state(self, state):
        """Global playback hook (owned by MainWindow). Keeps QueuePane passive."""
        try:
            from PySide6.QtMultimedia import QMediaPlayer as _QMP
            playing = (state == _QMP.PlayingState)
        except Exception:
            playing = False
        try:
            q = getattr(self, "queue", None)
            if q is not None and hasattr(q, "set_playback_active"):
                q.set_playback_active(bool(playing))
        except Exception:
            pass

    def _hook_video_player(self, player):
        """(Re)bind MainWindow playback hooks to the current QMediaPlayer instance."""
        # Disconnect previous binding if any
        try:
            old = getattr(self, "_bound_video_player", None)
            if old is not None and hasattr(old, "playbackStateChanged"):
                try:
                    old.playbackStateChanged.disconnect(self._on_global_video_playback_state)
                except Exception:
                    pass
        except Exception:
            pass

        self._bound_video_player = player
        try:
            if player is not None and hasattr(player, "playbackStateChanged"):
                try:
                    player.playbackStateChanged.connect(self._on_global_video_playback_state)
                except Exception:
                    pass
                try:
                    self._on_global_video_playback_state(player.playbackState())
                except Exception:
                    pass
        except Exception:
            pass


    def _install_tab_reorder_behavior(self):
        """Wire up movable tabs + persistence."""
        try:
            s = QSettings("FrameVision", "FrameVision")
            en = s.value("tabs_reorder_enabled", False, type=bool)
        except Exception:
            en = False
        try:
            self.set_tabs_reorder_enabled(bool(en))
        except Exception:
            pass
        try:
            self._tab_order_applying = False
        except Exception:
            pass
        try:
            self.tabs.tabBar().tabMoved.connect(lambda *_: self._save_tab_order_ids())
        except Exception:
            pass
        # Defer restore until all optional tabs are created/inserted
        try:
            QTimer.singleShot(0, self._post_init_tab_order_setup)
        except Exception:
            pass

    def eventFilter(self, obj, ev):
        """Watch the tab bar for resize/layout changes so we can hide/show
        the left/right arrow container dynamically."""
        try:
            if getattr(self, "_tabbar_ref", None) is obj:
                t = ev.type()
                if t == QEvent.Resize or t == QEvent.LayoutRequest:
                    try:
                        if hasattr(self, "_sync_tab_nav_state"):
                            self._sync_tab_nav_state()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            from PySide6.QtWidgets import QMainWindow as _QMain
            return _QMain.eventFilter(self, obj, ev)
        except Exception:
            # fall back
            try:
                return super().eventFilter(obj, ev)
            except Exception:
                return False
    
    def _restore_active_tab_by_name(self):
    
                try:
                    ss = config.get("session_restore", {})
                except Exception:
                    ss = {}
                try:
                    name_saved = str(ss.get("active_tab_name", "")).strip()
                except Exception:
                    name_saved = ""
                try:
                    idx_saved = int(ss.get("active_tab", 0) or 0)
                except Exception:
                    idx_saved = 0
                # Prefer name match (case-insensitive, normalized spaces)
                def _norm(x: str) -> str:
                    try:
                        return " ".join(str(x).split()).strip().lower()
                    except Exception:
                        return str(x).strip().lower()
                if name_saved:
                    target = _norm(name_saved)
                    found = -1
                    try:
                        for _i in range(self.tabs.count()):
                            if _norm(self.tabs.tabText(_i)) == target:
                                found = _i
                                break
                    except Exception:
                        found = -1
                    if found >= 0:
                        try:
                            self.tabs.setCurrentIndex(found)
                            return
                        except Exception:
                            pass
                # Fallback to clamped saved index
                try:
                    if idx_saved < 0 or idx_saved >= self.tabs.count():
                        idx_saved = max(0, min(self.tabs.count()-1, idx_saved))
                    self.tabs.setCurrentIndex(idx_saved)
                except Exception:
                    pass
            
    
    def _ensure_scrollbar_on_tabs(self, names):
        from PySide6.QtWidgets import QScrollArea, QFrame
        from PySide6.QtCore import QUrl, Qt
        # wrap each matching tab in a vertical-only QScrollArea (Describe-style)
        current = self.tabs.currentIndex()
        for i in range(self.tabs.count()):
            text = self.tabs.tabText(i)
            if text not in names:
                continue
            w = self.tabs.widget(i)
            if isinstance(w, QScrollArea):
                continue
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(w)
            try:
                from PySide6.QtWidgets import QSizePolicy
                w.setMinimumWidth(0)
                w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            except Exception:
                pass
            icon = self.tabs.tabIcon(i)
            self.tabs.removeTab(i)
            self.tabs.insertTab(i, sa, icon, text)
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
    # >>> FRAMEVISION_MEDIA_EXPLORER_HARDFAIL_BEGIN
    def _init_media_explorer_hardfail(self):
        """Initialize Media Explorer tab.

        NOTE: Despite the historical name, this is now **non-fatal**:
        if Media Explorer can't be imported/constructed, we return a placeholder widget
        and keep the app running.
        """
        import traceback
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit

        def _make_placeholder(err_text: str) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            title = QLabel("Media Explorer (not available)")
            title.setStyleSheet("font-weight: 600;")
            title.setWordWrap(True)
            msg = QLabel(
                "Media Explorer failed to load. FrameVision will keep running. "
                "You can reinstall/update the app or restore the missing file."
            )
            msg.setWordWrap(True)

            details = QTextEdit()
            details.setReadOnly(True)
            details.setLineWrapMode(QTextEdit.NoWrap)
            details.setPlainText(err_text)

            lay.addWidget(title)
            lay.addWidget(msg)
            lay.addWidget(details)
            lay.addStretch(1)
            return w

        # Try importing from helpers/ first, then fallback to project root.
        mod = None
        last_exc = None
        for mod_name in ("helpers.media_explorer_tab", "media_explorer_tab"):
            try:
                mod = __import__(mod_name, fromlist=["*"])
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                mod = None

        if mod is None:
            return _make_placeholder(traceback.format_exc())

        # Try common class factory names.
        for cls_name in ("MediaExplorerTab", "MediaExplorerPane", "MediaExplorerWidget"):
            try:
                if hasattr(mod, cls_name):
                    cls = getattr(mod, cls_name)
                    try:
                        return cls(self)
                    except TypeError:
                        return cls()
            except Exception:
                return _make_placeholder(traceback.format_exc())

        # Try factory functions if present.
        for fn_name in ("create_tab", "build_tab", "make_tab"):
            try:
                if hasattr(mod, fn_name):
                    fn = getattr(mod, fn_name)
                    try:
                        return fn(self)
                    except TypeError:
                        return fn()
            except Exception:
                return _make_placeholder(traceback.format_exc())

        # Nothing usable found in module
        err = (
            "Imported media_explorer_tab but couldn't find a compatible widget class.\n\n"
            "Expected one of: MediaExplorerTab / MediaExplorerPane / MediaExplorerWidget, or "
            "a factory: create_tab / build_tab / make_tab.\n\n"
            + traceback.format_exc()
        )
        return _make_placeholder(err)
    # <<< FRAMEVISION_MEDIA_EXPLORER_HARDFAIL_END


    def open_media_explorer_folder(self, folder: str, *, preset: Optional[str] = None, include_subfolders: Optional[bool] = None, activate: bool = True, rescan: bool = True, clear_first: bool = True) -> None:
        """Open the Media Explorer tab, point it to a folder, and optionally trigger a scan.

        This is intended as a single reusable entry-point that *any* tab can call.

        Args:
            folder: Target folder to browse.
            preset: Optional filter preset: "images", "videos", "audio", or "all".
            include_subfolders: If provided, sets the 'include subfolders' checkbox.
            activate: If True, switches to the Media Explorer tab.
            rescan: If True, triggers a rescan after setting the folder.
            clear_first: If True, stops any running scan and clears current results before rescanning.
        """
        try:
            from pathlib import Path
            p = Path(str(folder)).expanduser()
        except Exception:
            p = None

        if p is None or not p.exists() or not p.is_dir():
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Media Explorer", "Folder does not exist.")
            except Exception:
                pass
            return

        # Locate the Media Explorer tab index WITHOUT relying on the visible tab text.
        idx = -1
        tab_ref = None
        try:
            tab_ref = getattr(self, "media_explorer", None)
        except Exception:
            tab_ref = None

        # Preferred: direct widget reference.
        try:
            if tab_ref is not None:
                idx = int(self.tabs.indexOf(tab_ref))
        except Exception:
            idx = -1

        # Fallback: stable objectName.
        if idx < 0:
            try:
                found = self._find_tab_index_by_id("tab_media_explorer")
                if found is not None:
                    idx = int(found)
            except Exception:
                pass

        # Last-resort: normalize tabText and match a stable slug.
        if idx < 0:
            try:
                target = "media_explorer"
                for i in range(self.tabs.count()):
                    if self._slug_tab_name(self.tabs.tabText(i)) == target:
                        idx = i
                        break
            except Exception:
                pass

        if activate and idx >= 0:
            try:
                self.tabs.setCurrentIndex(idx)
            except Exception:
                pass

        # Get the actual tab widget (handle potential QScrollArea wrappers).
        tab = None
        try:
            tab = getattr(self, "media_explorer", None)
        except Exception:
            tab = None
        if tab is None and idx >= 0:
            try:
                tab = self.tabs.widget(idx)
            except Exception:
                tab = None
        try:
            from PySide6.QtWidgets import QScrollArea
            if tab is not None and isinstance(tab, QScrollArea):
                try:
                    tab = tab.widget()
                except Exception:
                    pass
        except Exception:
            pass

        if tab is None:
            return

        # Apply optional UI presets (best-effort; ignore if attributes are missing).
        try:
            if include_subfolders is not None and hasattr(tab, "cb_subfolders"):
                tab.cb_subfolders.setChecked(bool(include_subfolders))
        except Exception:
            pass

        try:
            if preset and isinstance(preset, str):
                pr = preset.strip().lower()
                if pr in ("images", "image"):
                    if hasattr(tab, "cb_images"): tab.cb_images.setChecked(True)
                    if hasattr(tab, "cb_videos"): tab.cb_videos.setChecked(False)
                    if hasattr(tab, "cb_audio"): tab.cb_audio.setChecked(False)
                elif pr in ("videos", "video"):
                    if hasattr(tab, "cb_images"): tab.cb_images.setChecked(False)
                    if hasattr(tab, "cb_videos"): tab.cb_videos.setChecked(True)
                    if hasattr(tab, "cb_audio"): tab.cb_audio.setChecked(False)
                elif pr in ("audio", "music"):
                    if hasattr(tab, "cb_images"): tab.cb_images.setChecked(False)
                    if hasattr(tab, "cb_videos"): tab.cb_videos.setChecked(False)
                    if hasattr(tab, "cb_audio"): tab.cb_audio.setChecked(True)
                elif pr in ("all", "any"):
                    if hasattr(tab, "cb_images"): tab.cb_images.setChecked(True)
                    if hasattr(tab, "cb_videos"): tab.cb_videos.setChecked(True)
                    if hasattr(tab, "cb_audio"): tab.cb_audio.setChecked(True)
        except Exception:
            pass

        # Point to folder + scan
        try:
            if hasattr(tab, "set_root_folder"):
                tab.set_root_folder(str(p))
        except Exception:
            pass

        if not rescan:
            return

        try:
            if clear_first and hasattr(tab, "clear"):
                tab.clear()
        except Exception:
            pass

        try:
            if hasattr(tab, "rescan"):
                tab.rescan()
        except Exception:
            pass



    # --- Left pane override: allow tools to temporarily occupy the big left media area ---
    def set_left_override_widget(self, w: QWidget, owner: str = "tool") -> None:
        """Show a tool-provided widget in the left pane (temporarily replacing the VideoPane visually)."""
        try:
            # If Compare overlay is active, don't let tools steal the left pane (prevents hiding compare).
            if w is not None:
                vp = getattr(self, 'video', None)
                if vp is not None and bool(getattr(vp, '_compare_active', False)):
                    ow = str(owner or '').lower()
                    if ow not in ('compare', 'autocompare'):
                        return
        except Exception:
            pass
        try:
            if w is None:
                self.clear_left_override_widget(owner=owner)
                return
            stack = getattr(self, "left_stack", None)
            if stack is None:
                return
            try:
                self._left_override_owner = owner or "tool"
                self._left_override_widget = w
            except Exception:
                pass
            try:
                idx = stack.indexOf(w)
            except Exception:
                idx = -1
            if idx is None or idx < 0:
                try:
                    stack.addWidget(w)
                except Exception:
                    pass
            try:
                stack.setCurrentWidget(w)
            except Exception:
                pass
        except Exception:
            pass

    def clear_left_override_widget(self, owner=None) -> None:
        """Return left pane to the normal VideoPane. If owner is set, only clears if it matches."""
        try:
            if owner:
                cur_owner = getattr(self, "_left_override_owner", None)
                if cur_owner and str(cur_owner) != str(owner):
                    return
        except Exception:
            pass
        try:
            stack = getattr(self, "left_stack", None)
            if stack is None:
                return
            try:
                stack.setCurrentWidget(self.video)
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._left_override_owner = None
            self._left_override_widget = None
        except Exception:
            pass


    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME + " V2.3 " + TAGLINE)
        self.resize(1280, 800)
        self.setMinimumSize(700, 500)
        self.current_path = None

        # left: video
        self.video = VideoPane(self)
        # right: hud + tabs
        self.hud = HUD(self)
        self.tabs = QTabWidget(self)
        try:
            self.tabs.setObjectName('main_tabs')
        except Exception:
            pass
        try:
            self._install_tab_nav_arrows()
        except Exception:
            pass

        try:
            self._install_tab_reorder_behavior()
        except Exception:
            pass

        # --- Optional hide state (from helpers/remove_hide.py JSON) ---
        try:
            self._optional_hidden_ids = set()
            try:
                try:
                    from helpers import remove_hide as _rh
                except Exception:
                    import remove_hide as _rh
                try:
                    _st = _rh.load_state(_rh.get_state_path())
                except Exception:
                    _st = {}
                _hid = _st.get('hidden_ids', []) if isinstance(_st, dict) else []
                if isinstance(_hid, list):
                    self._optional_hidden_ids = set(str(x) for x in _hid if isinstance(x, str))
            except Exception:
                self._optional_hidden_ids = set()
        except Exception:
            self._optional_hidden_ids = set()



        # >>> FRAMEVISION_TXT2IMG_BEGIN
        # Insert txt2img tab as the leftmost tab; label exactly "TXT to IMG".
        try:
            _hid = getattr(self, '_optional_hidden_ids', set()) or set()
            _txt2img_all_hidden = set(['sdxl_txt2img', 'zimage_turbo_fp16', 'zimage_turbo_gguf', 'qwen_image_2512']).issubset(_hid)
        except Exception:
            _txt2img_all_hidden = False

        if not _txt2img_all_hidden:
            try:
                if 'Txt2ImgPane' in globals() and Txt2ImgPane is not None:
                    self._txt2img_qwen = Txt2ImgPane(self)
                    self.tabs.insertTab(0, self._txt2img_qwen, "TXT to IMG")
                    # Wire to player: open resulting image in the left player
                    try:
                        self._txt2img_qwen.fileReady.connect(self.video.open)
                    except Exception:
                        pass
            except Exception as _e:
                print("[framevision] txt2img tab insert failed:", _e)
        else:
            try:
                self._txt2img_qwen = None
            except Exception:
                pass
        # <<< FRAMEVISION_TXT2IMG_END
        # The analyzer integration was disabled to avoid runtime errors.
        # (Previously initialized QuickActionDriver and installed it here.)
        self.edit = InterpPane(self, {
            'ROOT': ROOT,
            'BASE': BASE,
            'OUT_VIDEOS': OUT_VIDEOS,
            'JOBS_DIRS': JOBS_DIRS,
            'MANIFEST_PATH': MANIFEST_PATH,
            'config': config
        })
        self.tools = InstantToolsPane(self)
        self.describe = DescribePane(self)
        # >>> FRAMEVISION_PLANNER_TAB_INIT_BEGIN
        try:
            if 'PlannerPane' in globals() and PlannerPane is not None:
                self.planner = PlannerPane(self)
                try:
                    self.planner.setObjectName("tab_planner")
                except Exception:
                    pass
            else:
                self.planner = QWidget()
                _lay = QVBoxLayout(self.planner)
                _lay.setContentsMargins(12, 12, 12, 12)
                _lay.addWidget(QLabel("Planner tab failed to load."))
                _lay.addStretch(1)
        except Exception as _e:
            print("[framevision] planner init failed:", _e)
            self.planner = QWidget()
            _lay = QVBoxLayout(self.planner)
            _lay.setContentsMargins(12, 12, 12, 12)
            _lay.addWidget(QLabel("Planner tab failed to load."))
            _lay.addStretch(1)
        # <<< FRAMEVISION_PLANNER_TAB_INIT_END
        self.background = BackgroundPane(self)
        self.models = ModelsPane(self, {
            'MODELS_DIR': MODELS_DIR,
            'MANIFEST_PATH': MANIFEST_PATH,
            'config': config
        })
        self.queue = QueuePane(self, {'BASE': BASE, 'JOBS_DIRS': JOBS_DIRS, 'config': config, 'save_config': save_config})
        
        # Expose shared config/paths on MainWindow (helps keep panes decoupled)
        try:
            self.config = config
            self.BASE = BASE
            self.JOBS_DIRS = JOBS_DIRS
            self.save_config = save_config
        except Exception:
            pass

        # Hook playback state so the queue can pause its own refresh timers without touching the player.
        try:
            self._hook_video_player(getattr(self.video, 'player', None))
        except Exception:
            pass

        self.presets_tab = PresetsPane(self)
        self.settings = SettingsPane(self)

        # >>> FRAMEVISION_MUSICCLIP_INIT_BEGIN
        # Create Music Clip Creator tab instance if available.
        try:
            if 'MusicClipCreatorTab' in globals() and MusicClipCreatorTab is not None:
                self.music_clip_creator = MusicClipCreatorTab(main=self, parent=self)
                try:
                    self.music_clip_creator.setObjectName("music_clip_creator_tab")
                except Exception:
                    pass
            else:
                self.music_clip_creator = None
        except Exception as _e:
            print("[framevision] Music Clip Creator init failed:", _e)
            self.music_clip_creator = None
        # <<< FRAMEVISION_MUSICCLIP_INIT_END

# >>> FRAMEVISION_EDITOR_INIT_BEGIN
        # Create Editor tab instance if available
#        try:
 #           if 'EditorPane' in globals() and EditorPane is not None:
  #              self.editor = EditorPane(self)
   #         else:
    #            self.editor = None
     #   except Exception as _e:
      #      print("[framevision] editor init failed:", _e)
       #     self.editor = None
# <<< FRAMEVISION_EDITOR_INIT_END

        self.video.frameCaptured.connect(self.describe.on_pause_capture)
        # >>> FRAMEVISION_MEDIA_EXPLORER_INIT_BEGIN
        # Media Explorer is optional: if it fails to load we show a placeholder tab and keep running.
        self.media_explorer = self._init_media_explorer_hardfail()
        try:
            # Stable internal id (do NOT depend on tab label text, which may be emoji-only).
            if getattr(self, "media_explorer", None) is not None:
                try:
                    if not str(self.media_explorer.objectName() or ""):
                        self.media_explorer.setObjectName("tab_media_explorer")
                except Exception:
                    pass
        except Exception:
            pass
        # <<< FRAMEVISION_MEDIA_EXPLORER_INIT_END

        _main_tabs = [("Edit", self.edit),("Background", self.background),("Media Explorer", self.media_explorer),("Tools", self.tools),("Describe", self.describe),("Planner", self.planner),("Queue", self.queue),("Models", self.models),("Presets", self.presets_tab),("Settings", self.settings)]
        try:
            if getattr(self, "music_clip_creator", None) is not None:
                # Insert right after Media Explorer so it sits near other creation tools.
                _main_tabs.insert(3, ("Music Clip Creator", self.music_clip_creator))
        except Exception:
            pass

        for name, w in _main_tabs:
            self.tabs.addTab(w, name)

        # >>> FRAMEVISION_MEDIA_EXPLORER_POSTCHECK_BEGIN
        # Crash loudly if the tab still isn't present (so we don't silently fail).
        # IMPORTANT: Do NOT depend on the visible label text (it may include emojis or be empty).
        try:
            ok = False
            try:
                if getattr(self, "media_explorer", None) is not None:
                    ok = (self.tabs.indexOf(self.media_explorer) >= 0)
            except Exception:
                ok = False
            if not ok:
                try:
                    ok = (self._find_tab_index_by_id("tab_media_explorer") is not None)
                except Exception:
                    ok = False
            if not ok:
                try:
                    for _i in range(self.tabs.count()):
                        if self._slug_tab_name(self.tabs.tabText(_i)) == "media_explorer":
                            ok = True
                            break
                except Exception:
                    pass
            if not ok:
                raise RuntimeError("Media Explorer tab was not added to the main tabs.")
        except Exception as _e:
            import traceback
            from PySide6.QtWidgets import QMessageBox
            details = "Media Explorer post-check failed.\n\n" + traceback.format_exc()
            try:
                QMessageBox.critical(self, "Media Explorer not present", str(_e), details=details)
            except Exception:
                print(details)
            raise
        # <<< FRAMEVISION_MEDIA_EXPLORER_POSTCHECK_END


        # Create WAN 2.2 tab (always show; model loaders handle missing environments)

        # >>> FRAMEVISION_WAN22_INIT_BEGIN
        try:
            _hid = getattr(self, '_optional_hidden_ids', set()) or set()
            _wan_all_hidden = set(['wan22', 'hunyuan15']).issubset(_hid)
        except Exception:
            _wan_all_hidden = False

        if _wan_all_hidden:
            try:
                self.wan22 = None
            except Exception:
                pass
        else:
            try:
                if 'Wan22Pane' in globals() and Wan22Pane is not None:
                    try:
                        self.wan22 = Wan22Pane(self)
                    except Exception as _e:
                        # If the pane itself raises, show a friendly error tab instead of hiding it.
                        print("[framevision] WAN22 init failed in pane constructor:", _e)
                        err = QWidget(self)
                        lay = QVBoxLayout(err)
                        lab = QLabel(f"WAN 2.2 failed to load:\n{_e}", err)
                        lab.setWordWrap(True)
                        lay.addWidget(lab)
                        self.wan22 = err
                else:
                    # Module imported but no usable pane class, or import failed completely.
                    err = QWidget(self)
                    lay = QVBoxLayout(err)
                    txt = (
                        "WAN 2.2 module not found or no Wan22Pane class defined.\n"
                        "Make sure helpers/wan22.py defines a QWidget subclass named 'Wan22Pane'."
                    )
                    lab = QLabel(txt, err)
                    lab.setWordWrap(True)
                    lay.addWidget(lab)
                    self.wan22 = err
                # Insert WAN 2.2 tab just after TXT to IMG if present; otherwise append.
                try:
                    idx_txt = -1
                    try:
                        idx_txt = self.tabs.indexOf(getattr(self, "_txt2img_qwen", None))
                    except Exception:
                        idx_txt = -1
                    if idx_txt is not None and idx_txt >= 0:
                        self.tabs.insertTab(idx_txt + 1, self.wan22, "TXT/IMG/VID to Video")
                    else:
                        self.tabs.addTab(self.wan22, "TXT/IMG/VID to Video")
                except Exception as _attach_e:
                    try:
                        self.tabs.addTab(self.wan22, "TXT/IMG/VID to Video")
                    except Exception as _e2:
                        print("[framevision] WAN22 tab attach failed:", _e2)
            except Exception as _e:
                print("[framevision] WAN22 init failed:", _e)
        # <<< FRAMEVISION_WAN22_INIT_END

        # >>> FRAMEVISION_QWEN2511_INIT_BEGIN
        try:
            _hid = getattr(self, '_optional_hidden_ids', set()) or set()
            _qwen_hidden = ('qwen_edit_2511' in _hid)
        except Exception:
            _qwen_hidden = False

        if _qwen_hidden:
            try:
                self.qwen2511 = None
            except Exception:
                pass
        else:
            try:
                if 'Qwen2511Pane' in globals() and Qwen2511Pane is not None:
                    try:
                        self.qwen2511 = Qwen2511Pane(self)
                    except Exception as _e:
                        # If the pane itself raises, show a friendly error tab instead of hiding it.
                        print("[framevision] Qwen2511 init failed in pane constructor:", _e)
                        err = QWidget(self)
                        lay = QVBoxLayout(err)
                        lab = QLabel(f"Qwen2511 failed to load:\n{_e}", err)
                        lab.setWordWrap(True)
                        lay.addWidget(lab)
                        self.qwen2511 = err
                else:
                    # Module imported but no usable pane class, or import failed completely.
                    err = QWidget(self)
                    lay = QVBoxLayout(err)
                    txt = (
                        "Qwen2511 module not found or no Qwen2511Pane class defined.\n"
                        "Make sure helpers/qwen2511.py defines a QWidget subclass named 'Qwen2511Pane'."
                    )
                    lab = QLabel(txt, err)
                    lab.setWordWrap(True)
                    lay.addWidget(lab)
                    self.qwen2511 = err

                # Give the pane a reference to the main window (enables Media Explorer + internal player helpers).
                try:
                    setattr(self.qwen2511, 'main', self)
                except Exception:
                    pass

                # Stable internal id (avoid depending on tab label text).
                try:
                    if getattr(self, 'qwen2511', None) is not None:
                        if not str(getattr(self.qwen2511, 'objectName', lambda: '')() or ''):
                            self.qwen2511.setObjectName('tab_qwen2511')
                except Exception:
                    pass

                # Insert Qwen2511 tab just after WAN 2.2 if present; otherwise after TXT to IMG; otherwise append.
                try:
                    idx_wan = -1
                    try:
                        idx_wan = self.tabs.indexOf(getattr(self, 'wan22', None))
                    except Exception:
                        idx_wan = -1

                    if idx_wan is not None and idx_wan >= 0:
                        self.tabs.insertTab(idx_wan + 1, self.qwen2511, 'Qwen Edit  2511')
                    else:
                        idx_txt = -1
                        try:
                            idx_txt = self.tabs.indexOf(getattr(self, '_txt2img_qwen', None))
                        except Exception:
                            idx_txt = -1
                        if idx_txt is not None and idx_txt >= 0:
                            self.tabs.insertTab(idx_txt + 1, self.qwen2511, 'Qwen Edit  2511')
                        else:
                            self.tabs.addTab(self.qwen2511, 'Qwen Edit  2511')
                except Exception as _attach_e:
                    try:
                        self.tabs.addTab(self.qwen2511, 'Qwen Edit  2511')
                    except Exception as _e2:
                        print('[framevision] Qwen2511 tab attach failed:', _e2)
            except Exception as _e:
                print('[framevision] Qwen2511 init failed:', _e)
        # <<< FRAMEVISION_QWEN2511_INIT_END


                # >>> FRAMEVISION_ace_INIT_BEGIN
        # Create ace tab if available and the AceMusic environment is present
#        try:
 #           try:
  #              ace_enabled = ACE_ENV_DIR.exists()
   #         except Exception:
    #            from pathlib import Path as _Path
     #           ace_enabled = _Path(".").resolve().joinpath(".ace_env").exists()
      #      if ace_enabled and 'acePane' in globals() and acePane is not None:
       #         self.ace = acePane(self)
        #        try:
         #           idx_tools = self.tabs.indexOf(self.tools)
          #      except Exception:
 #                   idx_tools = -1
  #              if idx_tools is not None and idx_tools >= 0:
   #                 self.tabs.insertTab(idx_tools, self.ace, 'AceMusic')
    #            else:
     #               self.tabs.addTab(self.ace, 'ace')
      #      else:
       #         # AceMusic extra not installed or pane not available; don't create a tab.
        #        self.ace = None
   #     except Exception as _e:
    #        print('[framevision] ace init failed:', _e)
        # <<< FRAMEVISION_ace_INIT_END

        # >>> FRAMEVISION_ACE_STEP_15_INIT_BEGIN
        # Attach Ace-Step 1.5 as a main tab if its pane is available.
        try:
            _hid = getattr(self, '_optional_hidden_ids', set()) or set()
            if 'ace_step_15' in _hid:
                self.ace_step_15 = None
            elif 'AceStep15Pane' in globals() and AceStep15Pane is not None:
                try:
                    # Support either class (QWidget) or factory function returning QWidget
                    self.ace_step_15 = AceStep15Pane(self) if isinstance(AceStep15Pane, type) else AceStep15Pane(self)
                except Exception:
                    self.ace_step_15 = None

                if self.ace_step_15 is not None:
                    # Place it near other creation tools if possible (before Tools).
                    try:
                        idx_tools = self.tabs.indexOf(self.tools)
                    except Exception:
                        idx_tools = -1

                    if idx_tools is not None and idx_tools >= 0:
                        self.tabs.insertTab(idx_tools, self.ace_step_15, 'Ace-Step 1.5')
                    else:
                        self.tabs.addTab(self.ace_step_15, 'Ace-Step 1.5')
            else:
                self.ace_step_15 = None
        except Exception as _e:
            print('[framevision] Ace-Step 1.5 init failed:', _e)
            self.ace_step_15 = None
        # <<< FRAMEVISION_ACE_STEP_15_INIT_END







        # --- Hide legacy Upscale / RIFE tabs (now embedded in Tools tab) ---
        try:
            _hide = {"upscale", "rife fps", "rife_fps", "rife", "edit", "background", "bg", "background remover", "background/inpainter", "inpainter", "inpaint"}
            for _i in range(self.tabs.count() - 1, -1, -1):
                try:
                    _t = (self.tabs.tabText(_i) or "").strip().lower()
                except Exception:
                    _t = ""
                if _t in _hide:
                    try:
                        self.tabs.removeTab(_i)
                    except Exception:
                        pass
        except Exception:
            pass

        splitter = QSplitter(Qt.Horizontal, self)
        try:
            splitter.setObjectName('main_splitter')
        except Exception:
            pass
        self.left_stack = QStackedWidget()
        try:
            self.left_stack.setObjectName('left_stack')
        except Exception:
            pass
        try:
            self.left_stack.addWidget(self.video)
            self.left_stack.setCurrentWidget(self.video)
        except Exception:
            pass
        self._left_override_widget = None
        self._left_override_owner = None

        left = QWidget(); lv = QVBoxLayout(left)
        try:
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setSpacing(0)
        except Exception:
            pass
        lv.addWidget(self.left_stack)
        right = QWidget(); rv = QVBoxLayout(right); rv.addWidget(self.hud); rv.addWidget(self.tabs)
        left.setMinimumSize(320, 240); right.setMinimumSize(360, 240)
        self.tabs.setMinimumWidth(360)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1)
        splitter.setMinimumSize(680, 480)

        # Default ratio favoring the right pane (tabs) unless a saved state exists
        try:
            ss = config.get("session_restore", {})
        except Exception:
            ss = {}
        if not ss.get("splitter_state_b64"):
            from PySide6.QtCore import QUrl, QTimer
            def _apply_default_sizes():
                try:
                    tot = max(1000, self.width())
                    splitter.setSizes([int(tot*0.50), int(tot*0.50)])
                except Exception:
                    pass
            QTimer.singleShot(0, _apply_default_sizes)


        # Auto-refresh Queue list when the tab becomes active
        try:
            self.tabs.currentChanged.connect(self._on_tab_changed)
        except Exception:
            pass
        self.setCentralWidget(splitter)

        # --- Shortcut: ESC exits fullscreen (video overlay or window) ---
        try:
            esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
            def _exit_fullscreen():
                try:
                    # If video overlay fullscreen is active, exit that
                    if getattr(self.video, 'is_fullscreen', False) or getattr(self.video, '_fs_win', None) is not None:
                        self.video.toggle_fullscreen()
                        return
                    # Else if the main window itself is fullscreen, exit to normal
                    if self.isFullScreen():
                        self.showNormal()
                except Exception:
                    pass
            esc_shortcut.activated.connect(_exit_fullscreen)
        except Exception:
            pass
        # Restore saved UI state (tabs, splitters, geometry, etc.)
        try:
            state_persist.restore_all(self)
        except Exception:
            pass
        # Finalize tab selection after all inserts using a queued call
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._restore_active_tab_by_name)
        except Exception:
            pass
        # Ensure correct active tab after potential tab removals
        try:
            ss = config.get("session_restore", {})
            name_saved = str(ss.get("active_tab_name", "")).strip()
            idx = int(ss.get("active_tab", 0) or 0)
            if name_saved:
                for _i in range(self.tabs.count()):
                    if str(self.tabs.tabText(_i)).strip() == name_saved:
                        idx = _i
                        break
            if idx < 0 or idx >= self.tabs.count():
                # Default to Planner on fresh installs / missing saved tab
                try:
                    _p = None
                    try:
                        _p = self._find_tab_index_by_id("tab_planner")
                    except Exception:
                        _p = None
                    if _p is None:
                        try:
                            _w = getattr(self, "planner", None)
                            _p = self.tabs.indexOf(_w) if _w is not None else -1
                        except Exception:
                            _p = -1
                    if _p is not None and _p >= 0:
                        idx = _p
                    else:
                        idx = 0
                except Exception:
                    idx = 0
            self.tabs.setCurrentIndex(idx)
        except Exception:
            pass
        # Enforce a real, visible vertical scrollbar on Tools
        try:
            self._ensure_scrollbar_on_tabs({"Tools","Upscaler","Upscale","Models","Queue"})
            try:
                self._ensure_scrollbar_on_tabs({"Editor"})
            except Exception:
                pass

        except Exception:
            pass
        # Remove legacy 'Presets' tab if present
        for _i in range(self.tabs.count()-1, -1, -1):
            if self.tabs.tabText(_i).lower().startswith('preset'):
                self.tabs.removeTab(_i)


        # Hide any legacy 'Fix Layout' button
        for btn in self.findChildren(QPushButton):
            t = (btn.text() or "").strip().lower()
            if t == "fix layout":
                btn.hide(); btn.setEnabled(False)

        # Auto Compare (Before / After): poll finished jobs and open Compare when requested.
        try:
            self._auto_compare_seen = set()
            self._auto_compare_timer = QTimer(self)
            try:
                import time as _t
                self._auto_compare_start_time = float(_t.time())
            except Exception:
                self._auto_compare_start_time = 0.0
            try:
                self._auto_compare_pending = {}
            except Exception:
                pass
            self._auto_compare_timer.setInterval(750)
            self._auto_compare_timer.timeout.connect(self._auto_compare_poll_done_jobs)
            self._auto_compare_timer.start()
        except Exception:
            pass



        # Menu
        openAct = QAction("&Open", self); openAct.setShortcut(QKeySequence.Open); openAct.triggered.connect(self.open_file)
        try:
            self._install_optional_downloads_menu()
        except Exception:
            pass

    def _auto_compare_poll_done_jobs(self):
        """Job-finished hook: open Compare when a finished job requests it.

        Goals:
        - Works for queued jobs (including Qwen) without depending on the "Show last result" behavior.
        - Does NOT resurrect old compare overlays after restarting the app.

        Rules:
        - Only react to jobs whose own finished/started/created timestamp is AFTER this app session started.
        - After Compare successfully opens for a job, stamp the DONE json with auto_compare_consumed=True.
        """
        try:
            done_dir = JOBS_DIRS.get("done", None)
            if not done_dir:
                return
            done_dir = Path(done_dir)
            if not done_dir.exists():
                return
        except Exception:
            return

        # Only scan a small window of the most recent job manifests.
        try:
            files = list(done_dir.glob("*.json"))
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            files = files[:120]
        except Exception:
            return

        try:
            start_ts = float(getattr(self, "_auto_compare_start_time", 0.0) or 0.0)
        except Exception:
            start_ts = 0.0

        try:
            seen = getattr(self, "_auto_compare_seen", set())
            if not isinstance(seen, set):
                seen = set(seen)
        except Exception:
            seen = set()

        try:
            pending = getattr(self, "_auto_compare_pending", {})
            if not isinstance(pending, dict):
                pending = {}
        except Exception:
            pending = {}

        def _parse_job_ts(job: dict, jf: Path) -> float:
            """Return a best-effort timestamp for *when this job finished*.

            Important: do NOT rely on file mtime alone (moves between folders can preserve mtime).
            """
            try:
                import time as _t
                from datetime import datetime as _dt

                for k in ("finished_at", "finished", "ended_at", "ended", "started_at", "created_at"):
                    v = job.get(k)
                    if not v:
                        continue
                    # Numeric timestamp?
                    try:
                        if isinstance(v, (int, float)):
                            return float(v)
                    except Exception:
                        pass
                    # "YYYY-mm-dd HH:MM:SS" (worker format)
                    try:
                        return _t.mktime(_t.strptime(str(v), "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        pass
                    # ISO fallback
                    try:
                        return float(_dt.fromisoformat(str(v)).timestamp())
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                return float(jf.stat().st_mtime)
            except Exception:
                return 0.0

        def _bump_pending(jid: str) -> int:
            try:
                n = int(pending.get(jid, 0) or 0) + 1
            except Exception:
                n = 1
            pending[jid] = n
            try:
                self._auto_compare_pending = pending
            except Exception:
                pass
            return n

        def _mark_consumed(jf: Path, job: dict) -> None:
            """Persistently mark a DONE manifest so it won't re-open Compare after restart."""
            try:
                import time as _t
                job["auto_compare_consumed"] = True
                job["auto_compare_consumed_at"] = _t.strftime("%Y-%m-%d %H:%M:%S")
                tmp = jf.with_suffix(jf.suffix + ".tmp")
                tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
                try:
                    tmp.replace(jf)
                except Exception:
                    jf.write_text(json.dumps(job, indent=2), encoding="utf-8")
            except Exception:
                pass

        for jf in files:
            try:
                jid = jf.stem
            except Exception:
                continue

            try:
                if jid in seen:
                    continue
            except Exception:
                pass

            try:
                job = load_json(str(jf), default={})
                if not isinstance(job, dict) or not job:
                    continue
            except Exception:
                continue

            # Never re-open consumed jobs (including across app restarts).
            try:
                if bool(job.get("auto_compare_consumed", False)):
                    try:
                        seen.add(jid)
                    except Exception:
                        pass
                    continue
            except Exception:
                pass

            # Session fence: ignore DONE jobs from before this launch.
            try:
                if start_ts:
                    jts = _parse_job_ts(job, jf)
                    # Allow a small margin to avoid edge cases right at startup.
                    if jts and (jts < (start_ts - 1.0)):
                        try:
                            seen.add(jid)
                        except Exception:
                            pass
                        continue
            except Exception:
                pass

            # Eligibility: explicit request OR Qwen auto-compare toggle for Qwen jobs.
            try:
                is_generic = bool(job.get("auto_compare", False))
            except Exception:
                is_generic = False

            try:
                is_qwen = (not is_generic) and bool(self._qwen2511_job_wants_autocompare(job))
            except Exception:
                is_qwen = False

            if not is_generic and not is_qwen:
                continue

            # Extract before/after.
            try:
                if is_generic:
                    left, right = self._auto_compare_extract_paths(job)
                else:
                    left, right = self._qwen2511_extract_before_after(job)
            except Exception:
                left, right = "", ""

            # If paths aren't ready yet (common for just-finished queue jobs), retry a few times.
            if not left or not right:
                tries = _bump_pending(jid)
                if tries >= 16:
                    try:
                        seen.add(jid)
                    except Exception:
                        pass
                    try:
                        pending.pop(jid, None)
                    except Exception:
                        pass
                continue

            # Mark handled in-memory so we don't keep reopening during this session.
            try:
                seen.add(jid)
            except Exception:
                pass
            try:
                pending.pop(jid, None)
            except Exception:
                pass

            def _do_open(_jf=jf, _job=job, _l=left, _r=right):
                try:
                    # Ensure VideoPane is visible.
                    try:
                        self.clear_left_override_widget(owner=None)
                    except Exception:
                        pass

                    from helpers.compare_dialog import open_with_files
                    _ll, _rr, kind = open_with_files(self, _l, _r)
                    if kind is not None:
                        _mark_consumed(_jf, _job)
                except Exception:
                    pass

            # Delay slightly: queue may auto-open the result first (which would close compare).
            try:
                QTimer.singleShot(650, _do_open)
            except Exception:
                _do_open()

        # Persist back (best-effort)
        try:
            self._auto_compare_seen = seen
        except Exception:
            pass
        try:
            self._auto_compare_pending = pending
        except Exception:
            pass


    def _auto_compare_extract_paths(self, job: dict):
        """Best-effort extraction of left/right paths from a finished job manifest.

        Important: many tools store their real output under job['args'] (e.g. args['outfile']).
        """
        def _s(x):
            try:
                return str(x or "").strip()
            except Exception:
                return ""

        left = _s(job.get("auto_compare_left"))
        right = _s(job.get("auto_compare_right"))

        # Common field names (top-level)
        if not left:
            for k in ("input", "input_path", "infile", "src", "source", "init_img", "init_image", "image"):
                left = _s(job.get(k))
                if left:
                    break
        if not right:
            for k in ("output", "output_path", "outfile", "out", "result", "result_path", "final", "final_path", "produced"):
                right = _s(job.get(k))
                if right:
                    break

        # Many jobs store paths under args.
        try:
            args = job.get("args") or {}
        except Exception:
            args = {}

        if isinstance(args, dict):
            if not left:
                try:
                    v = args.get("ref_images")
                    if isinstance(v, (list, tuple)) and v:
                        left = _s(v[0])
                except Exception:
                    pass
            if not left:
                for k in ("init_img", "init_image", "image", "input", "input_path", "scene_image", "ref_image"):
                    left = _s(args.get(k))
                    if left:
                        break

            if not right:
                for k in ("outfile", "out_file", "output", "output_path", "save_path", "result", "result_path", "final", "final_path", "produced"):
                    right = _s(args.get(k))
                    if right:
                        break

        # Parse command list
        cmd = job.get("cmd")
        if isinstance(cmd, list):
            try:
                if not left:
                    for flag in ("-i", "--input", "--in", "--source"):
                        if flag in cmd:
                            i = cmd.index(flag)
                            if i + 1 < len(cmd):
                                left = _s(cmd[i + 1])
                                break
                if not right:
                    for flag in ("-o", "--output", "--out"):
                        if flag in cmd:
                            i = cmd.index(flag)
                            if i + 1 < len(cmd):
                                right = _s(cmd[i + 1])
                                break
            except Exception:
                pass

        # If output is a directory, pick the newest compatible file.
        try:
            if right and os.path.isdir(right):
                exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")
                cand = []
                for p in Path(right).glob("*"):
                    try:
                        if p.is_file() and p.suffix.lower() in exts:
                            cand.append(p)
                    except Exception:
                        pass
                if cand:
                    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    right = str(cand[0])
        except Exception:
            pass

        # Validate existence
        try:
            if left and not os.path.isfile(left):
                left = ""
            if right and not os.path.isfile(right):
                right = ""
        except Exception:
            pass

        return left, right



    # --- Qwen2511 queue auto-compare helpers -------------------------------------
    def _qwen2511_autocompare_enabled(self) -> bool:
        """Read the existing Qwen2511 auto-compare toggle (best-effort)."""
        try:
            pane = getattr(self, 'qwen2511', None)
            if pane is None:
                return False
            from PySide6.QtWidgets import QCheckBox
            for cb in pane.findChildren(QCheckBox):
                try:
                    t = (cb.text() or '').strip().lower()
                    n = (cb.objectName() or '').strip().lower()
                except Exception:
                    t = ''
                    n = ''
                if ('auto' in t and 'compare' in t) or ('auto' in n and 'compare' in n):
                    try:
                        return bool(cb.isChecked())
                    except Exception:
                        return False
        except Exception:
            pass
        return False

    def _qwen2511_job_wants_autocompare(self, job: dict) -> bool:
        try:
            if not self._qwen2511_autocompare_enabled():
                return False
        except Exception:
            return False
        try:
            return bool(self._is_qwen2511_job(job))
        except Exception:
            return False

    def _is_qwen2511_job(self, job: dict) -> bool:
        """Detect Qwen2511 jobs from queue manifests (best-effort)."""
        try:
            tool = str(job.get('tool') or job.get('name') or job.get('type') or '').strip().lower()
        except Exception:
            tool = ''
        if tool:
            if 'qwen' in tool and ('2511' in tool or 'edit' in tool):
                return True

        # Check cmd list/string
        try:
            cmd = job.get('cmd')
            if isinstance(cmd, list):
                s = ' '.join([str(x) for x in cmd]).lower()
            else:
                s = str(cmd or '').lower()
            if 'qwen' in s and '2511' in s:
                return True
        except Exception:
            pass

        # Heuristic: args contains ref_images and the job references qwen in any text field.
        args = self._coerce_job_args(job)
        try:
            has_refs = bool(args.get('ref_images'))
        except Exception:
            has_refs = False
        if has_refs:
            blob = ''
            try:
                blob = (str(job.get('title') or '') + ' ' + str(job.get('model') or '') + ' ' + str(job.get('engine') or '')).lower()
            except Exception:
                blob = ''
            if 'qwen' in blob:
                return True

        return False

    def _coerce_job_args(self, job: dict) -> dict:
        try:
            a = job.get('args', {})
        except Exception:
            a = {}
        if isinstance(a, dict):
            return a
        if isinstance(a, str):
            try:
                import json
                aa = json.loads(a)
                if isinstance(aa, dict):
                    return aa
            except Exception:
                return {}
        return {}

    def _qwen2511_extract_before_after(self, job: dict):
        """Extract BEFORE/AFTER for Qwen2511 queue jobs.

        BEFORE = first ref image (args['ref_images'][0]); fallback to a scene/source image only
                 if no ref_images exist.
        AFTER  = output file.
        """
        import os
        from pathlib import Path

        args = self._coerce_job_args(job)

        def _s(x):
            try:
                return str(x or '').strip()
            except Exception:
                return ''

        # BEFORE
        before = ''
        refs = args.get('ref_images', None)
        if isinstance(refs, list) and refs:
            r0 = refs[0]
            if isinstance(r0, dict):
                for k in ('path', 'file', 'image', 'img', 'value', 'src'):
                    if k in r0:
                        before = _s(r0.get(k))
                        break
            else:
                before = _s(r0)
        elif isinstance(refs, str):
            before = _s(refs)

        # Fallback: only if no ref images
        if not before:
            for k in ('scene_image', 'scene', 'image', 'input', 'input_path', 'init_image', 'init_img', 'src', 'source'):
                v = args.get(k)
                if not v:
                    v = job.get(k)
                if isinstance(v, str) and v.strip():
                    before = _s(v)
                    break

        # AFTER
        after = ''
        for src in (job, args):
            if not isinstance(src, dict):
                continue
            for k in ('output', 'output_path', 'out', 'outfile', 'result', 'result_path', 'final', 'final_path', 'save_path'):
                v = src.get(k)
                if isinstance(v, str) and v.strip():
                    after = _s(v)
                    break
            if after:
                break

        if not after:
            for k in ('outputs', 'result_files', 'files', 'out_files'):
                v = job.get(k) or args.get(k)
                if isinstance(v, list) and v:
                    after = _s(v[-1])
                    break

        # If output is a directory, pick newest media file.
        try:
            if after and os.path.isdir(after):
                exts = ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff','.gif','.mp4','.mkv','.mov','.webm','.avi','.m4v')
                cand = []
                for p in Path(after).glob('*'):
                    try:
                        if p.is_file() and p.suffix.lower() in exts:
                            cand.append(p)
                    except Exception:
                        pass
                if cand:
                    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    after = str(cand[0])
        except Exception:
            pass

        # Validate existence
        try:
            if before and not os.path.isfile(before):
                before = ''
            if after and not os.path.isfile(after):
                after = ''
        except Exception:
            pass

        return before, after

    # --- Startup layout settle (auto-resize/relayout) ----------------------------
    def showEvent(self, ev):
        # Some complex tabs (nested scroll areas / splitters) may not finalize their
        # geometry until the user manually resizes the window. We do a small, safe
        # one-time relayout after first show to make the UI correct immediately.
        try:
            super().showEvent(ev)
        except Exception:
            try:
                QMainWindow.showEvent(self, ev)
            except Exception:
                pass
        try:
            if getattr(self, "_did_startup_layout_fix", False):
                return
            self._did_startup_layout_fix = True
        except Exception:
            return

        # Multiple passes to catch late-created widgets / style polish.
        try:
            QTimer.singleShot(0,  lambda: self._startup_layout_fix_pass(0))
            QTimer.singleShot(50, lambda: self._startup_layout_fix_pass(1))
            QTimer.singleShot(200, lambda: self._startup_layout_fix_pass(2))
        except Exception:
            pass

    def _startup_layout_fix_pass(self, pass_no: int = 0):
        try:
            if not self.isVisible():
                return
        except Exception:
            pass

        try:
            st = self.windowState()
            is_max = bool(st & Qt.WindowMaximized) or bool(getattr(self, "isMaximized", lambda: False)())
            is_fs  = bool(st & Qt.WindowFullScreen) or bool(getattr(self, "isFullScreen", lambda: False)())
        except Exception:
            is_max = False
            is_fs = False

        # 1) Force polish + layout activation
        try:
            self.ensurePolished()
        except Exception:
            pass
        try:
            cw = self.centralWidget()
            if cw is not None:
                try:
                    cw.ensurePolished()
                except Exception:
                    pass
                try:
                    cw.updateGeometry()
                except Exception:
                    pass
                try:
                    lay = cw.layout()
                    if lay is not None:
                        lay.invalidate()
                        lay.activate()
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Current tab may contain scroll areas with late size hints
        try:
            if hasattr(self, "tabs") and self.tabs is not None:
                try:
                    self.tabs.updateGeometry()
                except Exception:
                    pass
                try:
                    cur = self.tabs.currentWidget()
                except Exception:
                    cur = None
                if cur is not None:
                    try:
                        cur.ensurePolished()
                    except Exception:
                        pass
                    try:
                        cur.updateGeometry()
                    except Exception:
                        pass
                    try:
                        lay2 = cur.layout()
                        if lay2 is not None:
                            lay2.invalidate()
                            lay2.activate()
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) Give Qt one chance to process queued layout events
        try:
            QApplication.processEvents()
        except Exception:
            pass

        # 4) Pixel-nudge (only when not maximized/fullscreen) to trigger a real resize event
        #    This mimics the user "making it bigger then smaller" without noticeable change.
        if (pass_no >= 1) and (not is_max) and (not is_fs):
            try:
                w0 = int(self.width())
                h0 = int(self.height())
                self.resize(w0 + 1, h0 + 1)
                self.resize(w0, h0)
            except Exception:
                pass

        try:
            self.updateGeometry()
        except Exception:
            pass
        try:
            self.repaint()
        except Exception:
            pass


    def open_optional_downloads(self):
        """Open the Optional installs UI (helpers.opt_installs)."""
        try:
            from helpers.opt_installs import show_optional_installs
        except Exception as _e:
            try:
                QMessageBox.warning(self, "Optional downloads", f"Optional installs UI not found: {_e}")
            except Exception:
                pass
            return
        try:
            show_optional_installs(parent=self, root_dir=str(ROOT))
        except Exception as _e:
            try:
                QMessageBox.warning(self, "Optional downloads", f"Failed to open Optional downloads: {_e}")
            except Exception:
                pass


    def open_remove_optional_installs(self):
        """Open the Remove/Hide manager UI (helpers.remove_hide)."""
        try:
            from helpers.remove_hide import RemoveHidePane, get_state_path
        except Exception as _e:
            try:
                QMessageBox.warning(self, "Optional downloads", f"Remove/Hide tool not found: {_e}")
            except Exception:
                pass
            return

        try:
            import os as _os
            app_root = str(ROOT) if 'ROOT' in globals() else _os.getcwd()
        except Exception:
            app_root = os.getcwd() if 'os' in globals() else '.'

        try:
            state_path = get_state_path("FrameVision")
        except Exception:
            state_path = None

        try:
            win = getattr(self, "_remove_hide_win", None)
        except Exception:
            win = None

        try:
            if win is None:
                win = QMainWindow(self)
                try:
                    win.setWindowTitle("Remove / Hide Optional Installs")
                except Exception:
                    pass
                try:
                    win.resize(1000, 750)
                except Exception:
                    pass
                try:
                    pane = RemoveHidePane(app_root=app_root, state_path=state_path)
                    win.setCentralWidget(pane)
                except Exception as _e:
                    try:
                        QMessageBox.warning(self, "Optional downloads", f"Failed to build Remove/Hide UI: {_e}")
                    except Exception:
                        pass
                    return
                try:
                    self._remove_hide_win = win
                except Exception:
                    pass

            try:
                win.show()
                win.raise_()
                win.activateWindow()
            except Exception:
                pass
        except Exception as _e:
            try:
                QMessageBox.warning(self, "Optional downloads", f"Failed to open Remove optional installs: {_e}")
            except Exception:
                pass


    def _install_optional_downloads_menu(self):
        """Add/ensure the 'Optional downloads' menu exists in the menubar."""
        try:
            mb = self.menuBar()
        except Exception:
            return

        def _norm(t):
            return str(t).replace("&", "").strip().lower()

        try:
            opt_menu = None
            for act in mb.actions():
                try:
                    if _norm(act.text()) == _norm("Optional downloads"):
                        opt_menu = act.menu()
                        if opt_menu is not None:
                            break
                except Exception:
                    continue

            if opt_menu is None:
                opt_menu = mb.addMenu("Optional downloads")

            # Avoid duplicate entries (allow adding new items over time)
            has_open = False
            has_remove = False
            for a in opt_menu.actions():
                try:
                    t = _norm(a.text())
                except Exception:
                    continue
                if t == _norm("Open optional installs..."):
                    has_open = True
                if t == _norm("Remove optional installs") or t == _norm("Remove optional installs..."):
                    has_remove = True

            if not has_open:
                a_open = QAction("Open optional installs...", self)
                a_open.triggered.connect(self.open_optional_downloads)
                opt_menu.addAction(a_open)

            if not has_remove:
                a_rm = QAction("Remove / Hide optional installs", self)
                a_rm.triggered.connect(self.open_remove_optional_installs)
                opt_menu.addAction(a_rm)

            # Ensure top-level menu order: File | Info | Optional downloads
            try:
                self._start_optional_downloads_menu_reorder()
            except Exception:
                pass
        except Exception:
            pass

    def _start_optional_downloads_menu_reorder(self):
        """Retry menubar reordering briefly so we don't miss startup timing."""
        try:
            from PySide6.QtCore import QTimer
        except Exception:
            return

        # If already running, don't start a second timer.
        try:
            if getattr(self, "_mb_reorder_timer", None) is not None:
                return
        except Exception:
            pass

        attempts = {"n": 0}
        timer = QTimer(self)
        try:
            timer.setInterval(120)
            timer.setSingleShot(False)
        except Exception:
            return

        def _tick():
            attempts["n"] += 1
            ok = False
            try:
                ok = bool(self._reorder_file_optional_info_menus())
            except Exception:
                ok = False

            if ok or attempts["n"] >= 60:
                try:
                    timer.stop()
                except Exception:
                    pass
                try:
                    self._mb_reorder_timer = None
                except Exception:
                    pass

        try:
            timer.timeout.connect(_tick)
        except Exception:
            return

        try:
            self._mb_reorder_timer = timer
        except Exception:
            pass

        try:
            QTimer.singleShot(0, _tick)
        except Exception:
            pass

        try:
            timer.start()
        except Exception:
            pass

    def _reorder_file_optional_info_menus(self) -> bool:
        """Force top-level order to: File | Info | Optional downloads. Returns True when done."""
        try:
            mb = self.menuBar()
        except Exception:
            return False

        def _norm(t: str) -> str:
            try:
                return str(t or "").replace("&", "").strip().lower()
            except Exception:
                return ""

        try:
            acts = list(mb.actions() or [])
        except Exception:
            return False

        file_act = None
        info_act = None
        opt_act = None

        for a in acts:
            try:
                t = _norm(a.text())
            except Exception:
                continue
            if t == "file":
                file_act = a
            elif t == "info":
                info_act = a
            elif t == "optional downloads":
                opt_act = a

        if not (file_act and info_act and opt_act):
            return False

        # 1) Ensure File comes before Info
        try:
            acts_now = list(mb.actions() or [])
            if acts_now.index(file_act) > acts_now.index(info_act):
                try:
                    mb.removeAction(file_act)
                except Exception:
                    pass
                try:
                    mb.insertAction(info_act, file_act)
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Ensure Optional downloads comes after Info
        try:
            acts_now = list(mb.actions() or [])
            if acts_now.index(opt_act) <= acts_now.index(info_act):
                try:
                    mb.removeAction(opt_act)
                except Exception:
                    pass

                acts2 = list(mb.actions() or [])
                try:
                    idx_info = acts2.index(info_act)
                except Exception:
                    idx_info = -1

                if idx_info >= 0 and (idx_info + 1) < len(acts2):
                    try:
                        mb.insertAction(acts2[idx_info + 1], opt_act)
                    except Exception:
                        try:
                            mb.addAction(opt_act)
                        except Exception:
                            pass
                else:
                    try:
                        mb.addAction(opt_act)
                    except Exception:
                        pass
        except Exception:
            pass

        # Verify
        try:
            acts_now = list(mb.actions() or [])
            return (acts_now.index(file_act) < acts_now.index(info_act) < acts_now.index(opt_act))
        except Exception:
            return False

    def _on_tab_changed(self, idx):
        try:
            name = str(self.tabs.tabText(idx)).strip().lower()
            if name == "queue":
                self.queue.start_auto()
            else:
                self.queue.stop_auto()
            # Auto-sync meme preview when switching to Tools tab
            try:
                if name == "tools" and hasattr(self, 'tools') and hasattr(self.tools, 'maybe_auto_sync_meme_preview'):
                    self.tools.maybe_auto_sync_meme_preview()
            except Exception:
                pass
        except Exception:
            pass
    def restore_session(self):
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtCore import QUrl, QRect
        def _clamp_visible(win):
            st = win.windowState()
            if st & (Qt.WindowFullScreen | Qt.WindowMaximized):
                return
            geo = win.frameGeometry()
            if geo.isNull():
                return
            union = None
            for scr in QGuiApplication.screens():
                r = scr.availableGeometry()
                union = r if union is None else union.united(r)
            if union is None:
                return
            inter = geo.intersected(union)
            if inter.isEmpty() or inter.width() < geo.width()*0.4 or inter.height() < geo.height()*0.4:
                new_w = min(geo.width(), union.width())
                new_h = min(geo.height(), union.height())
                new_x = max(union.x(), min(geo.x(), union.right() - new_w + 1))
                new_y = max(union.y(), min(geo.y(), union.bottom() - new_h + 1))
                win.setGeometry(new_x, new_y, new_w, new_h)

        ss = config.get("session_restore", {})
        geo = ss.get("win_geometry_b64","");
        if geo:
            try: self.restoreGeometry(QByteArray.fromBase64(geo.encode("ascii")))
            except Exception: pass
        try:
            _clamp_visible(self)
        except Exception:
            pass
        sp = ss.get("splitter_state_b64","");
        if sp:
            try: self.centralWidget().findChild(QSplitter).restoreState(QByteArray.fromBase64(sp.encode("ascii")))
            except Exception: pass
        try: self.tabs.setCurrentIndex(int(ss.get("active_tab",1)))
        except Exception: pass
        last = ss.get("last_file",""); pos = int(ss.get("last_position_ms",0) or 0)
        try:
            if bool(ss.get("is_fullscreen", False)):
                self.showFullScreen()
            elif bool(ss.get("is_maximized", False)):
                self.showMaximized()
        except Exception:
            pass

        try:
            flags = int(ss.get("win_state_flags", 0))
            if flags & Qt.WindowMaximized:
                self.showMaximized()
        except Exception:
            pass


    def save_session(self):
        ss = config.setdefault("session_restore", {})
        try:
            if not (self.windowState() & (Qt.WindowMaximized | Qt.WindowFullScreen)):
                ss["win_geometry_b64"] = bytes(self.saveGeometry().toBase64()).decode("ascii")
        except Exception:
            pass
        try:
            ss["splitter_state_b64"] = bytes(self.centralWidget().findChild(QSplitter).saveState().toBase64()).decode("ascii")
        except Exception:
            pass
        ss["is_maximized"] = bool(self.isMaximized())
        ss["is_fullscreen"] = bool(self.isFullScreen())
        ss["active_tab"] = int(self.tabs.currentIndex())
        
        ss["active_tab_name"] = str(self.tabs.tabText(self.tabs.currentIndex()))
        ss["last_file"] = str(self.current_path) if self.current_path else ""
        try: ss["last_position_ms"] = int(self.video.position_ms())
        except Exception: ss["last_position_ms"] = 0
        save_config()

    def closeEvent(self, ev):
        try: self.save_session()
        finally: super().closeEvent(ev)

    def open_file(self):
        d = config.get("last_open_dir", str(ROOT))
        fn, _ = QFileDialog.getOpenFileName(self, "Open media", d, "Media files (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.bmp *.webp *.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff);;Video files (*.mp4 *.mov *.mkv *.avi *.webm);;Images (*.png *.jpg *.jpeg *.bmp *.webp);;Audio files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff);;All files (*.*)")
        if fn:
            p = Path(fn)
            config["last_open_dir"] = str(p.parent); save_config()
            self.current_path = p
            self.video.open(p); self.hud.set_info(p);
            self.video.set_info_text(compose_video_info_text(p))
            try:
                # If user is on Tools tab, mirror media into Meme preview automatically.
                cur = str(self.tabs.tabText(self.tabs.currentIndex())).strip().lower()
                if cur == 'tools' and hasattr(self, 'tools') and hasattr(self.tools, 'maybe_auto_sync_meme_preview'):
                    self.tools.maybe_auto_sync_meme_preview()
            except Exception:
                pass
# --- Helpers presence (runtime import checks)
from helpers import job_helper  # noqa

def main():
    app = QApplication(sys.argv)
    from PySide6.QtGui import QIcon
    try:
        icon_path = str((ROOT / 'assets' / 'icons' / 'fv.png')) if 'ROOT' in globals() else 'assets/icons/fv.png'
        app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass

    apply_theme(app, config.get("theme","Auto"))
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
            
