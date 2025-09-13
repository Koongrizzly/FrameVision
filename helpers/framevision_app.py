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




# FrameVision V1.0 — Full-classic UI + NCNN upscalers + branding
# Classic layout (big player, control bar, seek slider, fullscreen) + click-to-play/pause
# Auto Theme (Day/Evening/Night), Session Restore, Instant Tools, Presets, Describe-on-Pause,
# Queue + Worker, Models Manager, Upscale Video/Photo buttons (queue) wired to NCNN CLIs.
# Branding: Title bar, About text, splash placeholder, default output under FrameVision/
# Backward-compat: if FrameLab/output/* exists, we still read/create there too.

import os, sys, json, subprocess, re, hashlib
from helpers.collapsible_compat import CollapsibleSection
from helpers import state_persist
from helpers.img_fallback import load_pixmap
from helpers.worker_led import WorkerStatusWidget
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
from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QRect, QEasingCurve, QPropertyAnimation, QByteArray, QEvent
from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QPixmap, QImage, QKeySequence, QColor, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFileDialog, QTabWidget, QSplitter, QListWidget, QListWidgetItem, QLineEdit, QFormLayout, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QStyle, QSlider, QToolButton, QSizePolicy, QScrollArea, QFrame, QGroupBox, QScrollArea, QFrame)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from helpers.tools_tab import InstantToolsPane
from helpers.themes import QSS_DAY, QSS_EVENING, QSS_NIGHT
from helpers.mediainfo import AUDIO_EXTS, probe_media_all, show_info_popup


# >>> FRAMEVISION_QWEN_BEGIN
# Safe import of the Qwen TXT->IMG pane; never crash app on failure.
try:
    from helpers.txt2img_qwen import Txt2ImgPane
except Exception as _e:
    print("[framevision] Qwen tab import failed:", _e)
    Txt2ImgPane = None
# <<< FRAMEVISION_QWEN_END

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
TAGLINE  = "All-in-one Video & Photo Upscaler/Editor"
ROOT = Path(".").resolve()


# --- Grabbable Splitter with themed hover/arrow & cursor ----------------------
from PySide6.QtWidgets import QSplitter, QSplitterHandle
from PySide6.QtGui import QPainter, QPen, QBrush, QPixmap, QCursor
from PySide6.QtCore import QPoint

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

def now_stamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def compose_video_info_text(path: Path) -> str:
    try:
        inf = probe_media(path)
        return f"{path.name} • {inf.get('width','?')}x{inf.get('height','?')} • {inf.get('fps','?')} fps • {inf.get('duration','?')} s • {human_mb(inf.get('size',0))} MB"
    except Exception:
        return path.name

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
    frameCaptured = Signal(QImage)
    # --- zoom/pan helpers ---
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
                return pm
            z = float(getattr(self, '_zoom', 1.0) or 1.0)
            from PySide6.QtCore import QSize
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

    def __init__(self,parent=None):
        super().__init__(parent)
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
        try:
            self.label.mouseDoubleClickEvent = lambda e: (self.toggle_fullscreen())
        except Exception:
            pass

        self.label.installEventFilter(self)
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,0); self.slider.sliderMoved.connect(self.seek)
        try:
            self.slider.setStyleSheet('QSlider::groove:horizontal{height:12px;} QSlider::sub-page:horizontal{height:12px;} QSlider::handle:horizontal{width:18px;height:18px;margin:-7px 0;}')
        except Exception:
            pass
        self.player.positionChanged.connect(self.on_pos); self.player.durationChanged.connect(self.on_dur)
        bar = QHBoxLayout()
        self.btn_open=QPushButton("📂"); self.btn_open.setToolTip("Open"); self.btn_play=QPushButton("▶"); self.btn_play.setToolTip("Play"); self.btn_pause=QPushButton("⏸"); self.btn_pause.setToolTip("Pause")
        self.btn_stop=QPushButton("⏹"); self.btn_stop.setToolTip("Stop"); self.btn_info=QPushButton("ℹ"); self.btn_info.setToolTip("Info"); self.btn_fs=QPushButton("⛶"); self.btn_fs.setToolTip("Fullscreen")
        self.btn_ratio=QPushButton("◻"); self.btn_ratio.setToolTip("Aspect: Center (click to cycle)")
        for b in [self.btn_open,self.btn_play,self.btn_pause,self.btn_stop,self.btn_info,self.btn_ratio,self.btn_fs]: bar.addWidget(b)
        self.btn_compare = QPushButton("▷│◁"); self.btn_compare.setToolTip("Compare view"); bar.addWidget(self.btn_compare)
        # Quick Upscale button
        self.btn_upscale = QPushButton("Upscale"); self.btn_upscale.setToolTip("Upscale")
        self.btn_upscale.setObjectName("btn_upscale_quick")
        try:
            self.btn_upscale.setStyleSheet(
                "QPushButton#btn_upscale_quick { padding:4px 14px; background:#3b82f6; color:white; border-radius:8px; font-weight:600; }"
                "QPushButton#btn_upscale_quick:hover { background:#2563eb; }"
            )
        except Exception:
            pass
        bar.addWidget(self.btn_upscale)
        # --- Uniform control sizing: keep all buttons the same height ---
        try:
            _all_ctrls = [self.btn_open, self.btn_play, self.btn_pause, self.btn_stop, self.btn_info, self.btn_ratio, self.btn_fs, self.btn_compare, self.btn_upscale]
            for _b in _all_ctrls:
                _b.setFixedHeight(48)
            # Keep play glyph larger but do NOT change height
            self.btn_play.setStyleSheet("font-size: 30px; padding: 0;")
            # Match Upscale to the raster/row height
            self.btn_upscale.setMinimumHeight(48)
            self.btn_upscale.setMaximumHeight(48)
        except Exception:
            pass


        bar.addStretch(1)
        lay = QVBoxLayout(self); lay.addWidget(self.label,1); self.info_label=QLabel("—"); self.info_label.setObjectName("videoInfo"); f=self.info_label.font(); f.setPointSize(max(10, f.pointSize())); self.info_label.setFont(f); lay.addWidget(self.info_label); lay.addWidget(self.slider); lay.addLayout(bar)
        self.btn_open.clicked.connect(self._open_via_dialog, Qt.ConnectionType.UniqueConnection); self.btn_play.clicked.connect(self.player.play, Qt.ConnectionType.UniqueConnection)
        self.btn_pause.clicked.connect(self.pause, Qt.ConnectionType.UniqueConnection); self.btn_stop.clicked.connect(self.player.stop, Qt.ConnectionType.UniqueConnection)
        self.btn_info.clicked.connect(self._show_info_popup, Qt.ConnectionType.UniqueConnection)
        self.btn_ratio.clicked.connect(self._cycle_ratio_mode, Qt.ConnectionType.UniqueConnection)
        self.btn_compare.clicked.connect(_open_compare_page, Qt.ConnectionType.UniqueConnection)
        self.btn_upscale.clicked.connect(self.quick_upscale, Qt.ConnectionType.UniqueConnection); self.btn_fs.clicked.connect(self.toggle_fullscreen, Qt.ConnectionType.UniqueConnection)
        self.is_fullscreen=False
        # Ratio modes: 0=Center, 1=Fit, 2=Fill, 3=Full(stretch)
        self.ratio_mode = 0
        self._update_ratio_button()
        
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
        self._max_zoom = 10.0
        self._zoom_step = 0.50
        self._pan_cx = 0.5
        self._pan_cy = 0.5
        self._dragging = False
        self._last_pos = None
        self._mode = 'video'
    
    def _on_frame(self, frame):
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
            from PySide6.QtCore import QTimer
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

    def mousePressEvent(self, ev):
        if self.player.playbackState()==QMediaPlayer.PlayingState:
            self.pause()
        else:
            self.player.play()

    def on_pos(self,pos): self.slider.setValue(pos); self._update_time_label()
    def on_dur(self,dur): self.slider.setRange(0,dur if dur>0 else 0); self._update_time_label()
    def seek(self,pos): self.player.setPosition(pos)

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
            self.btn_ratio.setText(txt)
            try:
                self.btn_ratio.setToolTip(f"Aspect: {tip} (click to cycle)")
            except Exception:
                pass
        except Exception:
            pass

    def _cycle_ratio_mode(self):
        try:
            self.ratio_mode = (int(self.ratio_mode) + 1) % 4
        except Exception:
            self.ratio_mode = 0
        self._update_ratio_button()
        # --- zoom/pan state + mode flag ---
        self._zoom = 1.0
        self._max_zoom = 10.0
        self._zoom_step = 0.50
        self._pan_cx = 0.5
        self._pan_cy = 0.5
        self._dragging = False
        self._last_pos = None
        self._mode = 'video'
        self._refresh_label_pixmap()

    
    def _choose_scaled(self, pm: QPixmap, target_size):
        try:
            if pm is None or pm.isNull():
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
            # Source image: original pixmap for images; latest frame for video
            pm = None
            if getattr(self, '_mode', 'video') == 'image':
                pm = getattr(self, '_image_pm_orig', None)
            else:
                if getattr(self, 'currentFrame', None) is not None and not self.currentFrame.isNull():
                    pm = QPixmap.fromImage(self.currentFrame)
            if pm is None or pm.isNull():
                return

            target = self.label.contentsRect().size()
            mode = int(getattr(self, 'ratio_mode', 0))
            z = float(getattr(self, '_zoom', 1.0) or 1.0)

            if mode == 0:  # Center
                if z < 1.0:
                    # Scale down only, keep centered
                    from PySide6.QtCore import QSize
                    sw = max(1, int(round(pm.width() * z)))
                    sh = max(1, int(round(pm.height() * z)))
                    spm = pm.scaled(QSize(sw, sh), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                elif z == 1.0:
                    spm = pm
                else:
                    # Scale up then crop to the label viewport based on pan
                    from PySide6.QtCore import QSize
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
            self.info_label.setText(str(text))
        except Exception:
            pass

    def _update_time_label(self):
        try:
            pos = self.player.position() or 0
            dur = self.player.duration() or 0
            left = max(0, dur - pos)
            pct = int(round((pos / dur) * 100)) if dur else 0
            text = f"{self._fmt_time(pos)} / {self._fmt_time(left)} left" + (f" • {pct}%" if dur else "")
            try:
                self.info_label.setText(text)
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


    def _open_via_dialog(self): self.parent().parent().parent().open_file()
    
    def open(self, path: Path):
        # Unified open: images -> QPixmap (GIF -> QMovie), audio -> QMediaPlayer, video -> QMediaPlayer
        try:
            p = Path(path)
        except Exception:
            from pathlib import Path as _P
            p = _P(str(path))
        ext = p.suffix.lower()

        # Animated GIF -> QMovie on label
        if ext == '.gif':
            try: self._autoplay_request = False; self._mode = 'image'; self.currentFrame = None
            except Exception: pass
            try: self.player.stop()
            except Exception: pass
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
            self.player.setSource(QUrl.fromLocalFile(str(p)))
            try: self.player.play()
            except Exception: pass
            try: self.label.setText(p.name)
            except Exception: pass
            return

        # Video
        try: self.slider.setEnabled(True)
        except Exception: pass
        try: self._image_pm_orig = None
        except Exception: pass
        self.player.setSource(QUrl.fromLocalFile(str(p)))
        try: self.player.play()
        except Exception: pass

    def pause(self):
        self.player.pause()
        if self.currentFrame is not None: self.frameCaptured.emit(self.currentFrame)


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
                pp.setToolTip("Play/Pause")
                h.insertWidget(0, pp, 0)
                self._fs_btn_pp = pp
                def _fs_sync_pp():
                    try:
                        if getattr(self, '_fs_btn_pp', None) is not None:
                            self._fs_btn_pp.setText("⏸" if self.player.playbackState()==QMediaPlayer.PlayingState else "▶")
                    except Exception:
                        pass
                def _fs_toggle():
                    try:
                        if self.player.playbackState()==QMediaPlayer.PlayingState:
                            self.pause()
                        else:
                            self.player.play()
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
            # Wire slider to seek
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
                            self.player.play()
                        return
                except Exception:
                    pass
                QWidget.keyPressEvent(fsw, e)
            fsw.keyPressEvent = _keyPress
            # Do NOT exit on mouse clicks anymore
            fsw.mousePressEvent = lambda e: QWidget.mousePressEvent(fsw, e)
            fsw.closeEvent = lambda e: (self.toggle_fullscreen(), e.accept()) if getattr(self, '_fs_win', None) is not None else e.accept()

            fsw.showFullScreen()
            try:
                from PySide6.QtCore import QTimer
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
                            self._cycle_ratio_mode()
                            return True
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
        try: self._refresh_label_pixmap()
        except Exception: pass
        try: return QWidget.resizeEvent(self, ev)
        except Exception: pass

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
            from PySide6.QtCore import QSettings
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
            now = datetime.now()
            time_str = f"{now.strftime('%a')}. {now.strftime('%d %b %H:%M.%S')}"

            hud = f"{gpu_str}  {ddr_str}  {cpu_str}  {time_str}"
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
            from PySide6.QtCore import QTimer
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

# --- Edit Pane: Upscale UI

# moved to helpers/interp.py as InterpPane

# --- Queue Pane


class QueuePane(QWidget):
    """
    Queue tab with vertical-only scrolling, responsive header (3 rows), worker LED, and live counters.
    """
    def __init__(self, main, parent=None):
        super().__init__(parent); self.main = main
        from PySide6.QtCore import QFileSystemWatcher

        # Timers (keep internal refresh cadence intact)
        self.auto_timer = QTimer(self); self.auto_timer.setInterval(2000); self.auto_timer.timeout.connect(self.refresh)
        self.watch_timer = QTimer(self); self.watch_timer.setInterval(1200); self.watch_timer.timeout.connect(self._watch_tick)
        self.worker_timer = QTimer(self); self.worker_timer.setInterval(1000); self.worker_timer.timeout.connect(self._update_worker_led)

        # Queue system and paths
        try:
            from helpers.queue_system import QueueSystem
        except Exception:
            from queue_system import QueueSystem
        self.qs = QueueSystem(BASE)

        # Root layout and scroll container
        root = QVBoxLayout(self); root.setContentsMargins(6,6,6,6); root.setSpacing(8)
        topw = QWidget(); grid = QGridLayout(topw); grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(6)

        # Row 1: Refresh · Clear finished/failed · Worker LED+label (right)
        self.btn_refresh = QPushButton("Refresh")
        self.worker_status = WorkerStatusWidget(); self.lbl_worker = self.worker_status.label
        self.btn_remove_done = QPushButton("Clear Finished")
        self.btn_remove_failed = QPushButton("Clear Failed")
        clearw = QWidget(); cl = QHBoxLayout(clearw); cl.setContentsMargins(0,0,0,0); cl.setSpacing(6)
        cl.addWidget(self.btn_remove_done); cl.addWidget(self.btn_remove_failed)
        grid.addWidget(self.btn_refresh, 0, 0); grid.addWidget(clearw, 0, 1); grid.addWidget(self.worker_status, 0, 2, 1, 1, Qt.AlignRight); grid.setColumnStretch(2, 1)

        # Row 2: Move up · Move down · Delete selected
        self.btn_move_up = QPushButton("Move Up"); self.btn_move_down = QPushButton("Move Down"); self.btn_delete_sel = QPushButton("Delete Selected")
        grid.addWidget(self.btn_move_up, 1, 0); grid.addWidget(self.btn_move_down, 1, 1); grid.addWidget(self.btn_delete_sel, 1, 2)

        # Row 3: Mark running → failed · Clear finished + failed · Recover running → pending
        self.btn_mark_running_failed = QPushButton("Mark Running → Failed")
        self.btn_clear_all = QPushButton("Clear Finished + Failed")
        self.btn_reset_running = QPushButton("Recover Running → Pending")
        grid.addWidget(self.btn_mark_running_failed, 2, 0); grid.addWidget(self.btn_clear_all, 2, 1); grid.addWidget(self.btn_reset_running, 2, 2)

        # Counters row
        self.counts = QLabel("Running 0 | Pending 0 | Done 0 | Failed 0")
        self.last_updated = QLabel("--:--:--")
        grid.addWidget(self.counts, 3, 0, 1, 2, Qt.AlignLeft)
        grid.addWidget(self.last_updated, 3, 2, 1, 1, Qt.AlignRight)
        root.addWidget(topw)

        # Scroll area with sections
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget(); v = QVBoxLayout(content); v.setContentsMargins(0,0,0,0); v.setSpacing(8)

        # Lists (5-row viewport, vertical only)
        self.lst_running = QListWidget(); self._apply_policies(self.lst_running)
        self.lst_pending = QListWidget(); self._apply_policies(self.lst_pending)
        self.lst_done = QListWidget(); self._apply_policies(self.lst_done)
        self.lst_failed = QListWidget(); self._apply_policies(self.lst_failed)

        sec_running = ToolsCollapsibleSection("Running", expanded=False)
        lay_sec_running = QVBoxLayout(); lay_sec_running.setContentsMargins(0,0,0,0); lay_sec_running.setSpacing(6)
        lay_sec_running.addWidget(self.lst_running)
        try:
            sec_running.setContentLayout(lay_sec_running)
        except Exception:
            # Fallback if API differs
            sec_running.content = QWidget(); sec_running.content.setLayout(lay_sec_running)
        v.addWidget(sec_running)
        sec_pending = ToolsCollapsibleSection("Pending", expanded=False)
        lay_sec_pending = QVBoxLayout(); lay_sec_pending.setContentsMargins(0,0,0,0); lay_sec_pending.setSpacing(6)
        lay_sec_pending.addWidget(self.lst_pending)
        try:
            sec_pending.setContentLayout(lay_sec_pending)
        except Exception:
            # Fallback if API differs
            sec_pending.content = QWidget(); sec_pending.content.setLayout(lay_sec_pending)
        v.addWidget(sec_pending)
        sec_done = ToolsCollapsibleSection("Finished", expanded=False)
        lay_sec_done = QVBoxLayout(); lay_sec_done.setContentsMargins(0,0,0,0); lay_sec_done.setSpacing(6)
        lay_sec_done.addWidget(self.lst_done)
        try:
            sec_done.setContentLayout(lay_sec_done)
        except Exception:
            # Fallback if API differs
            sec_done.content = QWidget(); sec_done.content.setLayout(lay_sec_done)
        v.addWidget(sec_done)
        sec_failed = ToolsCollapsibleSection("Failed", expanded=False)
        lay_sec_failed = QVBoxLayout(); lay_sec_failed.setContentsMargins(0,0,0,0); lay_sec_failed.setSpacing(6)
        lay_sec_failed.addWidget(self.lst_failed)
        try:
            sec_failed.setContentLayout(lay_sec_failed)
        except Exception:
            # Fallback if API differs
            sec_failed.content = QWidget(); sec_failed.content.setLayout(lay_sec_failed)
        v.addWidget(sec_failed)
        sc.setWidget(content); root.addWidget(sc)

        # File-system watcher for live counters + inserts (debounced <=5 Hz)
        self._fsw = QFileSystemWatcher(self)
        for k in ("pending","running","done","failed"):
            self._fsw.addPath(str(JOBS_DIRS[k]))
        self._debounce = QTimer(self); self._debounce.setInterval(180); self._debounce.setSingleShot(True)
        self._fsw.directoryChanged.connect(lambda _: self._debounce.start())
        self._debounce.timeout.connect(self._on_queue_changed)

        # Wire actions
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_remove_done.clicked.connect(self.clear_done)
        self.btn_remove_failed.clicked.connect(self.clear_failed)
        self.btn_clear_all.clicked.connect(self.clear_done_failed)
        self.btn_move_up.clicked.connect(self.move_up)
        self.btn_move_down.clicked.connect(self.move_down)
        self.btn_delete_sel.clicked.connect(self.delete_selected)
        self.btn_reset_running.clicked.connect(self.recover_running_to_pending)
        self.btn_mark_running_failed.clicked.connect(self.mark_running_failed)

        # First refresh and timers
        self.refresh()
        self.auto_timer.start(); self.worker_timer.start()

    def _apply_policies(self, w: QListWidget):
        try:
            w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            w.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            min_h = 56 * 5 + 8  # ~5 rows
            w.setMinimumHeight(min_h)
            w.setMaximumHeight(16777215)
        except Exception:
            pass


    def _is_main_job_json(self, p: Path) -> bool:
        name = p.name
        if not name.endswith(".json"):
            return False
        if name.endswith(".progress.json") or name.endswith(".json.progress") or name.endswith(".meta.json") or name.startswith("_"):
            return False
        return True


    def _populate(self, folder: Path, widget: QListWidget, status: str):
        from helpers.queue_widgets import JobRowWidget
        widget.clear()
        files = []
        try:
            for p in folder.glob('*.json'):
                name = p.name
                if (name.endswith('.progress.json') or name.endswith('.json.progress') or name.endswith('.progress')
                    or name.endswith('.meta.json') or name.startswith('_')):
                    continue
                try:
                    d = json.loads(p.read_text(encoding='utf-8') or '{}')
                    if not isinstance(d, dict) or (not d.get('type')) or (not d.get('input') and not d.get('frames')):
                        continue
                except Exception:
                    continue
                sort_ts = p.stat().st_mtime
                try:
                    from datetime import datetime
                    def _parse(s):
                        if not s: return None
                        for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M:%S.%f"):
                            try:
                                return datetime.strptime(s, fmt).timestamp()
                            except Exception:
                                pass
                        return None
                    if status == "pending":
                        sort_ts = _parse(d.get("enqueued_at") or d.get("created_at") or d.get("added_at")) or sort_ts
                    elif status == "running":
                        sort_ts = _parse(d.get("started_at")) or sort_ts
                    elif status in ("done","failed"):
                        sort_ts = _parse(d.get("finished_at")) or sort_ts
                except Exception:
                    pass
                files.append((sort_ts, p))
        except Exception:
            files = []
        for _ts, p in sorted(files, key=lambda t_p: t_p[0], reverse=True):
            it = QListWidgetItem("")
            w = JobRowWidget(str(p), status)
            try:
                hint = w.sizeHint()
                if hint.height() < 56:
                    from PySide6.QtCore import QSize
                    hint.setHeight(56)
                it.setSizeHint(hint)
            except Exception:
                it.setSizeHint(w.sizeHint())
            widget.addItem(it)
            widget.setItemWidget(it, w)

    def refresh(self):
        self._populate(JOBS_DIRS['running'], self.lst_running, 'running')
        self._populate(JOBS_DIRS['pending'], self.lst_pending, 'pending')
        self._populate(JOBS_DIRS['done'], self.lst_done, 'done')
        self._populate(JOBS_DIRS['failed'], self.lst_failed, 'failed')
        self._update_counts_label()
        self._update_worker_led()

    def _update_counts_label(self):
        try:
            r = sum(1 for pth in JOBS_DIRS["running"].glob("*.json") if self._is_main_job_json(pth))
            p = sum(1 for pth in JOBS_DIRS["pending"].glob("*.json") if self._is_main_job_json(pth))
            d = sum(1 for pth in JOBS_DIRS["done"].glob("*.json") if self._is_main_job_json(pth))
            f = sum(1 for pth in JOBS_DIRS["failed"].glob("*.json") if self._is_main_job_json(pth))
            self.counts.setText(f"Running {r} | Pending {p} | Done {d} | Failed {f}")
        except Exception:
            self.counts.setText("Running ? | Pending ? | Done ? | Failed ?")

    def _watch_tick(self):
        try:
            self.refresh()
        except Exception:
            pass

    def _on_queue_changed(self):
        self.refresh()

    # --- Queue actions (filesystem-based; minimal and safe) ---
    def _selected_job_path(self):
        # Return (bucket, Path) for the currently selected row; None if nothing selected
        try_lists = [
            ("running", self.lst_running),
            ("pending", self.lst_pending),
            ("done", self.lst_done),
            ("failed", self.lst_failed),
        ]
        for bucket, lst in try_lists:
            it = lst.currentItem()
            if it is None:
                continue
            try:
                w = lst.itemWidget(it)
                p = Path(getattr(w, "path", "") or getattr(w, "job_path", ""))
                if p and p.exists():
                    return bucket, p
            except Exception:
                pass
        return None, None

    def clear_done(self):
        try:
            for p in list(JOBS_DIRS["done"].glob("*.json")):
                try: p.unlink()
                except Exception: pass
        except Exception:
            pass
        self.refresh()

    def clear_failed(self):
        try:
            try:
                self.qs.remove_failed()
            except Exception:
                for p in list(JOBS_DIRS["failed"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
        except Exception:
            pass
        self.refresh()

    def clear_done_failed(self):
        try:
            try:
                self.qs.clear_finished_failed()
            except Exception:
                for p in list(JOBS_DIRS["done"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
                for p in list(JOBS_DIRS["failed"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
        except Exception:
            pass
        self.refresh()

    def delete_selected(self):
        bucket, p = self._selected_job_path()
        if not p:
            return
        try:
            p.unlink()
        except Exception:
            pass
        self.refresh()

    def move_up(self):
        bucket, p = self._selected_job_path()
        if bucket != "pending" or not p:
            return
        try:
            now = time.time()
            os.utime(p, (now, now + 60))
        except Exception:
            pass
        self.refresh()

    def move_down(self):
        bucket, p = self._selected_job_path()
        if bucket != "pending" or not p:
            return
        try:
            now = time.time()
            os.utime(p, (now, now - 60))
        except Exception:
            pass
        self.refresh()

    def recover_running_to_pending(self):
        bucket, p = self._selected_job_path()
        if bucket != "running" or not p:
            return
        try:
            dest = JOBS_DIRS["pending"] / p.name
            try:
                d = json.loads(p.read_text(encoding="utf-8") or "{}")
                for k in ("started_at","finished_at","ended_at","duration_sec","error"):
                    if k in d: d.pop(k, None)
                p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            except Exception:
                pass
            import shutil as _shutil
            _shutil.move(str(p), str(dest))
            try:
                self.qs.nudge_pending()
            except Exception:
                pass
        except Exception:
            pass
        self.refresh()

    def mark_running_failed(self):
        bucket, p = self._selected_job_path()
        if bucket != "running" or not p:
            return
        try:
            d = {}
            try:
                d = json.loads(p.read_text(encoding="utf-8") or "{}")
            except Exception:
                d = {}
            try:
                from datetime import datetime
                d["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            d["error"] = d.get("error") or "Manually marked as failed"
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            import shutil as _shutil
            dest = JOBS_DIRS["failed"] / p.name
            _shutil.move(str(p), str(dest))
        except Exception:
            pass
        self.refresh()

    def _update_worker_led(self):
        try:
            try:
                running_count = sum(1 for pth in JOBS_DIRS["running"].glob("*.json")
                                    if getattr(self, "_is_main_job_json", lambda p: True)(pth))
            except Exception:
                running_count = 0

            import time
            hb = globals().get('HEARTBEAT_PATH', BASE / 'logs' / 'worker_heartbeat.txt')
            age = None
            if hb.exists():
                try:
                    age = time.time() - hb.stat().st_mtime
                except Exception:
                    age = None

            if running_count > 0:
                self.worker_status.set_state("running", f"{running_count} active job(s)")
            elif age is not None and age < 12.0:
                self.worker_status.set_state("idle", f"Heartbeat {int(age)}s ago")
            elif age is not None:
                self.worker_status.set_state("stopped", f"No heartbeat for {int(age)}s")
            else:
                self.worker_status.set_state("stopped", "No heartbeat file")
        except Exception as e:
            try:
                self.worker_status.set_state("error", str(e))
            except Exception:
                pass


    def stop_auto(self):
        try:
            self.auto_timer.stop()
        except Exception:
            pass

    def closeEvent(self, e):
        try:
            state_persist.save_all(self)
        except Exception:
            pass
        try:
            self.stop_auto()
        except Exception:
            pass
        try:
            super().closeEvent(e)
        except Exception:
            pass

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
        about = QLabel(f"<h2>{APP_NAME}</h2><p>{TAGLINE}</p><p>© {datetime.now().year}</p><p>Placeholder splash — your logo can go here.</p>")
        v.addWidget(about)

# --- Main Window
class MainWindow(QMainWindow):

    def _ensure_scrollbar_on_tabs(self, names):
        from PySide6.QtWidgets import QScrollArea, QFrame
        from PySide6.QtCore import Qt
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME + " V1.0 " + TAGLINE)
        self.resize(1280, 800)
        self.setMinimumSize(900, 580)
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


        # >>> FRAMEVISION_QWEN_BEGIN
        # Insert Qwen tab as the leftmost tab; label exactly "TXT to IMG".
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
            print("[framevision] Qwen tab insert failed:", _e)
        # <<< FRAMEVISION_QWEN_END
        # GAB (QuickActionDriver) removed cleanly
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
        self.models = ModelsPane(self, {
            'MODELS_DIR': MODELS_DIR,
            'MANIFEST_PATH': MANIFEST_PATH,
            'config': config
        })
        self.queue = QueuePane(self)
        self.presets_tab = PresetsPane(self)
        self.settings = SettingsPane(self)

        self.video.frameCaptured.connect(self.describe.on_pause_capture)

        for name, w in [("Edit", self.edit),("Tools", self.tools),("Describe", self.describe),("Queue", self.queue),("Models", self.models),("Presets", self.presets_tab),("Settings", self.settings)]:
            self.tabs.addTab(w, name)

        splitter = QSplitter(Qt.Horizontal, self)
        try:
            splitter.setObjectName('main_splitter')
        except Exception:
            pass
        left = QWidget(); lv = QVBoxLayout(left); lv.addWidget(self.video)
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
            from PySide6.QtCore import QTimer
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
        # Restore saved UI state (tabs, splitters, geometry, etc.)
        try:
            state_persist.restore_all(self)
        except Exception:
            pass
        # Enforce a real, visible vertical scrollbar on Tools
        try:
            self._ensure_scrollbar_on_tabs({"Tools","Upscaler","Upscale","Models","Queue"})
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


        # Menu
        openAct = QAction("&Open", self); openAct.setShortcut(QKeySequence.Open); openAct.triggered.connect(self.open_file)

        # (Legacy File menu removed; see old/helpers/legacy_menu_snapshot.py)

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
        from PySide6.QtCore import QRect
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
    apply_theme(app, config.get("theme","Auto"))
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
            