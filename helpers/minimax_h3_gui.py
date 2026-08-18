from __future__ import annotations
import json, os, sys, subprocess, time, socket, urllib.request, re, uuid, html, shutil, hashlib, tempfile, threading, zipfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QEvent, QUrl, QSizeF, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor, QPainter, QColor, QBrush, QPixmap, QIcon

# FrameVision imports this module only after its QApplication is already running.
# Qt WebEngine/Multimedia can initialize native subsystems during import, so
# embedded mode deliberately skips them and uses the existing fallbacks.
_FRAMEVISION_EMBEDDED_IMPORT = os.environ.get("FRAMEVISION_MINIMAX_EMBEDDED_IMPORT", "") == "1"
if not _FRAMEVISION_EMBEDDED_IMPORT:
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWebEngineCore import QWebEnginePage
    except Exception:
        QWebEngineView = None
        QWebEnginePage = None
else:
    QWebEngineView = None
    QWebEnginePage = None
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QPlainTextEdit, QLineEdit, QFileDialog, QMessageBox, QTabWidget, QScrollArea,
    QCheckBox, QListWidget, QListWidgetItem, QFrame, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMenu, QSplitter, QSlider, QProgressBar, QGraphicsView, QGraphicsScene,
    QDialog, QDialogButtonBox, QAbstractItemView, QLayout, QSizePolicy
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments" / ".minimax_h3_int4" / "python.exe"
PRESET_DIR = ROOT / "presets" / "setsave"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_LORA_DIR = ROOT / "models" / "minimax_h3" / "loras"
LOG_DIR = ROOT / "logs"

APP_UPDATE_REPO = "Koongrizzly/MiniMax_H3_Standalone_app"
APP_UPDATE_PAGE = f"https://github.com/{APP_UPDATE_REPO}"
APP_UPDATE_ZIP = f"https://api.github.com/repos/{APP_UPDATE_REPO}/zipball"
APP_UPDATE_STATE = PRESET_DIR / "minimax_h3_update_state.json"
APP_UPDATE_EXCLUDED_TOP = {"environments", "models", "output", "logs", "jobs", ".git"}
APP_UPDATE_EXCLUDED_PREFIXES = {"presets/setsave", "h3_prompt_builder/.runtime"}


if not _FRAMEVISION_EMBEDDED_IMPORT:
    try:
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
    except Exception:
        QMediaPlayer = QAudioOutput = QGraphicsVideoItem = None
else:
    QMediaPlayer = QAudioOutput = QGraphicsVideoItem = None

try:
    import psutil
except Exception:
    psutil = None


if QWebEnginePage is not None:
    class PromptBuilderPage(QWebEnginePage):
        """Keep the local Prompt Builder in-app and send external links to Windows.

        QWebEngine treats links with target="_blank" as new-window requests rather
        than ordinary navigation.  The old page handler only caught normal clicks,
        so those links silently disappeared in an embedded QWebEngineView.
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            # Qt 6 exposes target=_blank/window.open through newWindowRequested.
            # Connect when available while keeping compatibility with older builds.
            try:
                self.newWindowRequested.connect(self._open_new_window_request)
            except Exception:
                pass

        @staticmethod
        def _is_external(url):
            try:
                host = (url.host() or "").lower()
                return url.scheme().lower() in ("http", "https") and host not in ("127.0.0.1", "localhost")
            except Exception:
                return False

        def _open_new_window_request(self, request):
            try:
                url = request.requestedUrl()
                if self._is_external(url):
                    QDesktopServices.openUrl(url)
                    return
            except Exception:
                pass

        def acceptNavigationRequest(self, url, nav_type, is_main_frame):
            try:
                if self._is_external(url):
                    # External pages never replace the embedded local builder.
                    QDesktopServices.openUrl(url)
                    return False
            except Exception:
                pass
            return super().acceptNavigationRequest(url, nav_type, is_main_frame)

        def createWindow(self, window_type):
            # Fallback for Qt builds where target=_blank reaches createWindow()
            # instead of newWindowRequested.  A short-lived page receives the URL
            # and forwards it to the user's default Windows browser.
            popup = QWebEnginePage(self)
            def forward(url):
                try:
                    if self._is_external(url):
                        QDesktopServices.openUrl(url)
                finally:
                    popup.deleteLater()
            popup.urlChanged.connect(forward)
            return popup
else:
    PromptBuilderPage = None

try:
    import pynvml
except Exception:
    pynvml = None

QUEUE_FILE = PRESET_DIR / "minimax_h3_queue.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from runtime.ffmpeg_tools import BIN_DIR as FFMPEG_BIN_DIR, tool_path as ffmpeg_tool_path, tools_ready as ffmpeg_tools_ready
except Exception:
    # Fresh FrameVision bootstrap: the MiniMax repository/runtime may not exist yet.
    # Keep the embedded GUI importable so the one-click installer can be shown.
    FFMPEG_BIN_DIR = ROOT / "presets" / "bin"
    def ffmpeg_tool_path(name):
        return FFMPEG_BIN_DIR / str(name)
    def ffmpeg_tools_ready():
        return all((FFMPEG_BIN_DIR / name).is_file() for name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"))


def _hud_color(value, yellow_at=None, orange_at=None, red_at=None):
    try:
        v = float(value)
    except Exception:
        return None
    if red_at is not None and v >= red_at:
        return "#d70022"
    if orange_at is not None and v >= orange_at:
        return "#d98a00"
    if yellow_at is not None and v >= yellow_at:
        return "#e0c600"
    return None

def _hud_span(text, color=None):
    safe = html.escape(str(text))
    if color:
        return f"<span style='color:{color};font-weight:600'>{safe}</span>"
    return safe

def _hud_rate(bytes_per_sec):
    v = float(max(0.0, bytes_per_sec))
    if v >= 1024**2:
        return f"{v/(1024**2):.1f} MB/s", v/(1024**2)
    return f"{v/1024:.0f} KB/s", v/(1024**2)


class SystemHud(QLabel):
    """Compact always-visible system HUD inspired by the user's FrameVision monitor."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimax_owner = parent
        self.setObjectName("systemHud")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self.setMinimumHeight(36)
        # The HUD is intentionally a single rich-text line. During generation the
        # JOB phase can become much longer than the idle text (for example VAE /
        # continuation loading messages). QLabel's normal horizontal size hint then
        # tries to make the *entire main layout* wide enough for that text, which can
        # push the right edge of the Generation tab outside a maximized Windows
        # client area. A manual resize appeared to fix it only because Qt renegotiated
        # the layout at that point. The HUD must consume whatever width is available;
        # it must never dictate the application's minimum/preferred width.
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setToolTip("Live GPU VRAM/load/temperature, DDR RAM, CPU load, network traffic above 100 KB/s, and local date/time.")
        self._last_net = None
        self._last_net_t = time.monotonic()
        self._nvml_handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount() > 0:
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
                self._last_net = psutil.net_io_counters()
            except Exception:
                pass
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _gpu(self):
        if self._nvml_handle is not None and pynvml is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                return mem.used/(1024**3), mem.total/(1024**3), int(util), int(temp)
            except Exception:
                pass
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL, text=True, timeout=0.8).strip().splitlines()[0]
            vals=[x.strip() for x in out.split(',')]
            return float(vals[0])/1024.0, float(vals[1])/1024.0, int(float(vals[2])), int(float(vals[3]))
        except Exception:
            return None

    def refresh(self):
        parts=[]
        gpu=self._gpu()
        if gpu:
            used,total,util,temp=gpu
            fill=(used/total*100.0) if total else 0
            memtxt=f"{used:.1f}/{total:.0f}"
            parts.append("GPU : " + _hud_span(memtxt, _hud_color(fill,85,90,95)))
            parts.append(_hud_span(f"{util}%", _hud_color(util,None,85,95)))
            parts.append(_hud_span(f"{temp}°C", _hud_color(temp,60,65,70)))
        else:
            parts.append("GPU : N/A")

        if psutil is not None:
            try:
                vm=psutil.virtual_memory(); used=(vm.total-vm.available)/(1024**3); total=vm.total/(1024**3)
                parts.append(f"DDR {used:.1f}/{total:.0f} {vm.percent:.0f}%")
            except Exception:
                parts.append("DDR N/A")
            try:
                cpu=float(psutil.cpu_percent(interval=None))
                parts.append("CPU " + _hud_span(f"{cpu:.0f}%", _hud_color(cpu,None,85,95)))
            except Exception:
                parts.append("CPU N/A")
            try:
                now_t=time.monotonic(); cur=psutil.net_io_counters()
                if self._last_net is not None:
                    dt=max(0.1, now_t-self._last_net_t)
                    dl=max(0,cur.bytes_recv-self._last_net.bytes_recv)/dt
                    ul=max(0,cur.bytes_sent-self._last_net.bytes_sent)/dt
                    # Only display a direction when it exceeds 100 KB/s.
                    if dl > 100*1024:
                        txt,mbs=_hud_rate(dl); parts.append("DL " + _hud_span(txt, _hud_color(mbs,10,50,95)))
                    if ul > 100*1024:
                        txt,mbs=_hud_rate(ul); parts.append("UL " + _hud_span(txt, _hud_color(mbs,10,50,95)))
                self._last_net=cur; self._last_net_t=now_t
            except Exception:
                pass
        else:
            parts.extend(["DDR N/A","CPU N/A"])

        # Keep the active generation visible in the HUD on every tab.
        try:
            # SystemHud is placed inside an intermediate central widget, so parent()
            # is not necessarily the MiniMax main window. window() reliably returns
            # the top-level application window that owns the live queue state.
            window = self._minimax_owner or self.window()
            job = None
            current_id = getattr(window, "current_job_id", None)
            if current_id and hasattr(window, "_job_by_id"):
                job = window._job_by_id(current_id)
            if job and job.get("state") == "running":
                phase = str(job.get("phase") or "Working").strip()
                progress = job.get("progress")
                step_now = job.get("step_now")
                step_total = job.get("step_total")
                elapsed = max(0, int(time.time() - float(job.get("started_at") or time.time())))
                elapsed_txt = f"{elapsed//60:02d}:{elapsed%60:02d}"
                detail = phase
                # Step counters belong only to the sampling stage. Once sampling
                # finishes and the job moves on to VAE/audio decode, muxing, etc.,
                # do not keep displaying the stale final "4/4" value.
                if phase.lower() == "sampling" and step_now is not None and step_total:
                    detail += f" {step_now}/{step_total}"
                if progress is not None:
                    detail += f" {int(progress)}%"
                parts.insert(0, "JOB " + _hud_span(f"{detail} · {elapsed_txt}", "#82e7ff"))
            else:
                parts.insert(0, "JOB " + _hud_span("Idle", "#82e7ff"))
        except Exception:
            pass

        parts.append(datetime.now().strftime("%a. %d %b %H:%M"))
        self.setText("&nbsp;&nbsp;".join(parts))

class ZoomVideoView(QGraphicsView):
    """Video preview with wheel zoom and hand-drag panning."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._zoom = 0
    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:
            return super().wheelEvent(event)
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        new_zoom = self._zoom + (1 if factor > 1 else -1)
        if -4 <= new_zoom <= 12:
            self.scale(factor, factor); self._zoom = new_zoom
        event.accept()
    def reset_view(self):
        self.resetTransform(); self._zoom = 0
        if self.scene() and not self.scene().itemsBoundingRect().isNull():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        return super().mousePressEvent(event)


class ImagePreviewDialog(QDialog):
    def __init__(self, title="Image preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 720)
        self._fullscreen = False
        self._last_normal_geometry = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        info = QLabel("Mouse wheel = zoom. Drag = pan. Double-click image to toggle fullscreen.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.scene = QGraphicsScene(self)
        self.view = ZoomVideoView(self)
        self.view.setScene(self.scene)
        self.view.setMinimumSize(320, 220)
        lay.addWidget(self.view, 1)
        self.pix_item = None
        btns = QHBoxLayout()
        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        btns.addWidget(self.path_label, 1)
        self.reset_btn = QPushButton("Reset zoom")
        self.full_btn = QPushButton("Fullscreen")
        self.close_btn = QPushButton("Close")
        self.reset_btn.clicked.connect(self.view.reset_view)
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        self.close_btn.clicked.connect(self.close)
        btns.addWidget(self.reset_btn)
        btns.addWidget(self.full_btn)
        btns.addWidget(self.close_btn)
        lay.addLayout(btns)

    def set_image(self, path):
        self.path_label.setText(str(path or ""))
        self.scene.clear()
        self.pix_item = None
        if not path:
            self.scene.addText("No image selected.")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.scene.addText("Unable to load preview for this image.")
            return
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self.view.reset_view()

    def toggle_fullscreen(self):
        if not self._fullscreen:
            self._last_normal_geometry = self.geometry()
            self.showFullScreen()
            self._fullscreen = True
            self.full_btn.setText("Exit fullscreen")
        else:
            self.showNormal()
            if self._last_normal_geometry is not None:
                self.setGeometry(self._last_normal_geometry)
            self._fullscreen = False
            self.full_btn.setText("Fullscreen")
            QTimer.singleShot(0, self.view.reset_view)

    def mouseDoubleClickEvent(self, event):
        self.toggle_fullscreen()
        event.accept()


# MiniMax H3 fixed resolution presets. Most combo labels are the exact generation
# dimensions. The familiar 1280 x 720 label is intentionally mapped to MiniMax's
# valid 704p tensor size (1280 x 704); 720 is not divisible by 32.
# Portrait presets transpose the actual generation dimensions and square presets
# use the matching short-edge generation size.
RESOLUTION_PRESETS = {
    "576 × 320":  {"16:9": (576, 320),  "9:16": (320, 576),  "1:1": (320, 320)},
    "736 × 384":  {"16:9": (736, 384),  "9:16": (384, 736),  "1:1": (384, 384)},
    "832 × 480":  {"16:9": (832, 480),  "9:16": (480, 832),  "1:1": (480, 480)},
    "960 × 544":  {"16:9": (960, 544),  "9:16": (544, 960),  "1:1": (544, 544)},
    "1280 × 720": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (704, 704)},
    "1344 × 768": {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (768, 768)},
    "1920 × 1088":{"16:9": (1920, 1088),"9:16": (1088, 1920),"1:1": (1088, 1088)},
}
DEFAULT_RESOLUTION = "832 × 480"
NORMAL_FRAME_MAX = 719   # last H3 native-grid value below 30 seconds (29.958 s at 24 FPS)
EXPERIMENTAL_FRAME_MAX = 2385  # last H3 native-grid value at/below 100 seconds (99.375 s at 24 FPS)
FRAME_PRESETS = sorted(set(list(range(124, NORMAL_FRAME_MAX + 1, 17)) + [480]))
EXPERIMENTAL_FRAME_PRESETS = sorted(set(list(range(124, EXPERIMENTAL_FRAME_MAX + 1, 17)) + [480]))
SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "dpmpp_2m",
    "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "uni_pc", "uni_pc_bh2"
]
SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]
MODEL_FILTER = "SafeTensors (*.safetensors);;All files (*)"


class NoWheelFilter(QWidget):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, (QComboBox, QSpinBox, QDoubleSpinBox)):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class FileRow(QWidget):
    def __init__(self, placeholder, filt, parent=None):
        super().__init__(parent)
        self.filter = filt
        self.preview_dialog = None
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.thumb = ClickableLabel()
        self.thumb.setFixedSize(84, 84)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setObjectName("imageThumb")
        self.thumb.setToolTip("Click to preview. Double-click the preview image to toggle fullscreen.")
        self.thumb.clicked.connect(self.open_preview)
        lay.addWidget(self.thumb, 0)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self._update_thumbnail)
        buttons = QHBoxLayout()
        b = QPushButton("Browse…"); b.clicked.connect(self.browse)
        p = QPushButton("Preview"); p.clicked.connect(self.open_preview)
        f = QPushButton("Fullscreen"); f.clicked.connect(self.open_fullscreen_preview)
        c = QPushButton("Clear"); c.clicked.connect(self.edit.clear)
        buttons.addWidget(b); buttons.addWidget(p); buttons.addWidget(f); buttons.addWidget(c)
        right.addWidget(self.edit)
        right.addLayout(buttons)
        lay.addLayout(right, 1)
        self._update_thumbnail()

    def _set_thumb_placeholder(self, text="No image"):
        self.thumb.setPixmap(QPixmap())
        self.thumb.setText(text)

    def _update_thumbnail(self):
        path = self.path()
        if not path:
            self._set_thumb_placeholder()
            self.thumb.setToolTip("No image selected.")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self._set_thumb_placeholder("Preview\nunavailable")
            self.thumb.setToolTip(path)
            return
        scaled = pix.scaled(self.thumb.size() - QSizeF(8, 8).toSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.thumb.setPixmap(scaled)
        self.thumb.setText("")
        self.thumb.setToolTip(path)

    def browse(self):
        start = self.path() or str(ROOT)
        p, _ = QFileDialog.getOpenFileName(self, "Select file", start, self.filter)
        if p: self.edit.setText(p)

    def _ensure_preview_dialog(self):
        if self.preview_dialog is None:
            self.preview_dialog = ImagePreviewDialog("Image preview", self)
        return self.preview_dialog

    def open_preview(self):
        path = self.path()
        if not path:
            QMessageBox.information(self, "Preview", "Choose an image first.")
            return
        dlg = self._ensure_preview_dialog()
        dlg.setWindowTitle(f"Preview - {Path(path).name}")
        dlg.set_image(path)
        dlg.showNormal()
        dlg.raise_()
        dlg.activateWindow()
        dlg.show()

    def open_fullscreen_preview(self):
        self.open_preview()
        if self.preview_dialog is not None and not self.preview_dialog._fullscreen:
            self.preview_dialog.toggle_fullscreen()

    def path(self): return self.edit.text().strip()


class VideoPathRow(QWidget):
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        b = QPushButton("Browse…"); b.clicked.connect(self.browse)
        c = QPushButton("Clear"); c.clicked.connect(self.edit.clear)
        lay.addWidget(self.edit, 1); lay.addWidget(b); lay.addWidget(c)
    def browse(self):
        start = self.edit.text().strip() or str(ROOT)
        if Path(start).is_file(): start = str(Path(start).parent)
        p, _ = QFileDialog.getOpenFileName(self, "Select source video to continue", start, "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if p: self.edit.setText(p)
    def path(self): return self.edit.text().strip()


class LoraPathRow(QWidget):
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        b = QPushButton("Browse…"); b.clicked.connect(self.browse)
        c = QPushButton("Clear"); c.clicked.connect(self.edit.clear)
        lay.addWidget(self.edit, 1); lay.addWidget(b); lay.addWidget(c)
    def browse(self):
        DEFAULT_LORA_DIR.mkdir(parents=True, exist_ok=True)
        start = self.edit.text().strip()
        if start and Path(start).is_file(): start = str(Path(start).parent)
        elif not start: start = str(DEFAULT_LORA_DIR)
        p, _ = QFileDialog.getOpenFileName(self, "Select MiniMax H3 LoRA", start, MODEL_FILTER)
        if p: self.edit.setText(p)
    def path(self): return self.edit.text().strip()


class ModelPathRow(QWidget):
    """Optional model override: one safetensors file OR a folder containing it."""
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        bf = QPushButton("File…"); bf.clicked.connect(self.browse_file)
        bd = QPushButton("Folder…"); bd.clicked.connect(self.browse_folder)
        bc = QPushButton("Clear"); bc.clicked.connect(self.edit.clear)
        lay.addWidget(self.edit, 1); lay.addWidget(bf); lay.addWidget(bd); lay.addWidget(bc)
    def browse_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", str(ROOT), MODEL_FILTER)
        if p: self.edit.setText(p)
    def browse_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select model folder", str(ROOT))
        if p: self.edit.setText(p)
    def path(self): return self.edit.text().strip()


class FolderRow(QWidget):
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        b = QPushButton("Browse…"); b.clicked.connect(self.browse)
        c = QPushButton("Clear"); c.clicked.connect(self.edit.clear)
        lay.addWidget(self.edit, 1); lay.addWidget(b); lay.addWidget(c)
    def browse(self):
        p = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text().strip() or str(ROOT))
        if p: self.edit.setText(p)
    def path(self): return self.edit.text().strip()


class RefList(QWidget):
    def __init__(self, title, filt, max_items, parent=None):
        super().__init__(parent); self.filt = filt; self.max_items = max_items
        self.preview_dialog = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout(); top.addWidget(QLabel(title)); top.addStretch()
        add = QPushButton("Add…"); rem = QPushButton("Remove"); prev = QPushButton("Preview"); full = QPushButton("Fullscreen")
        add.clicked.connect(self.add); rem.clicked.connect(self.remove); prev.clicked.connect(self.preview_selected); full.clicked.connect(self.preview_selected_fullscreen)
        top.addWidget(add); top.addWidget(rem); top.addWidget(prev); top.addWidget(full); lay.addLayout(top)
        self.list = QListWidget()
        self.list.setMinimumHeight(118)
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setSpacing(8)
        self.list.setIconSize(QSizeF(92, 92).toSize())
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemDoubleClicked.connect(lambda _it: self.preview_selected())
        lay.addWidget(self.list)

    def _make_item(self, path):
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        pix = QPixmap(str(path))
        if not pix.isNull():
            icon_pix = pix.scaled(self.list.iconSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QIcon(icon_pix))
        return item

    def _selected_path(self):
        item = self.list.currentItem() or (self.list.selectedItems()[0] if self.list.selectedItems() else None)
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def add(self):
        if self.list.count() >= self.max_items:
            QMessageBox.information(self, "Reference limit", f"Maximum {self.max_items} items in this group."); return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add references", str(ROOT), self.filt)
        existing = {self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())}
        for p in paths:
            if self.list.count() >= self.max_items: break
            if p not in existing:
                self.list.addItem(self._make_item(p))
                existing.add(p)

    def remove(self):
        for it in self.list.selectedItems(): self.list.takeItem(self.list.row(it))

    def _ensure_preview_dialog(self):
        if self.preview_dialog is None:
            self.preview_dialog = ImagePreviewDialog("Reference preview", self)
        return self.preview_dialog

    def preview_selected(self):
        path = self._selected_path()
        if not path:
            QMessageBox.information(self, "Preview", "Select a reference image first.")
            return
        dlg = self._ensure_preview_dialog()
        dlg.setWindowTitle(f"Reference preview - {Path(path).name}")
        dlg.set_image(path)
        dlg.showNormal()
        dlg.raise_()
        dlg.activateWindow()
        dlg.show()

    def preview_selected_fullscreen(self):
        self.preview_selected()
        if self.preview_dialog is not None and not self.preview_dialog._fullscreen:
            self.preview_dialog.toggle_fullscreen()

    def paths(self): return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def set_paths(self, vals):
        self.list.clear()
        for p in (vals or [])[:self.max_items]: self.list.addItem(self._make_item(str(p)))


class MainWindow(QMainWindow):
    update_check_finished = Signal(object)
    update_check_failed = Signal(object)
    bootstrap_repo_finished = Signal(object)
    bootstrap_failed = Signal(str)

    def __init__(self, parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)
        if self._embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        else:
            self.setWindowTitle("MiniMax H3 INT4 Standalone")
            # 1180x900 is only the fallback geometry used after the user explicitly
            # restores the window. Normal application startup is maximized.
            self.resize(1180, 900)
        self._user_window_size_override = False
        self._window_state_guard_pending = False
        self._layout_refresh_pending = False
        self._main_scroll_pages = []
        self._font_size_pt = 10
        self._font_apply_timer = QTimer(self)
        self._font_apply_timer.setSingleShot(True)
        self._font_apply_timer.setInterval(1800)
        self._font_apply_timer.timeout.connect(self._apply_pending_font_size)
        self.proc = None
        self.builder_process = None
        self.builder_port = 0
        self.builder_url = ""
        self.builder_loaded_url = ""
        self.builder_timer = None
        self.builder_template_default_applied = False
        self.prompt_webview = None
        self.queue_jobs = []
        self._next_job_number_value = 1
        self.current_job_id = None
        self._termination_action = None
        self._proc_buffer = ""
        self.preview_path = ""
        self._spinner_index = 0
        self._closing = False
        self._ffmpeg_setup_proc = None
        self._ffmpeg_setup_popup = None
        self._ffmpeg_setup_output = ""
        self._ffmpeg_setup_status = ""
        self._ffmpeg_setup_failed = False
        self._update_check_running = False
        self._update_payload = None
        self._bootstrap_running = False
        self._bootstrap_proc = None
        self._bootstrap_stage = ""
        self.update_check_finished.connect(self._handle_update_check_finished)
        self.update_check_failed.connect(self._handle_update_check_failed)
        self.bootstrap_repo_finished.connect(self._bootstrap_repo_ready)
        self.bootstrap_failed.connect(self._bootstrap_install_failed)
        self.wheel_filter = NoWheelFilter(self)
        self._build()
        self._apply_style()
        self.load_last()
        self._load_queue_state()
        self._sync_resolution()
        self._sync_mode()
        QTimer.singleShot(300, self.validate_install)
        QTimer.singleShot(350, self._refresh_bootstrap_button)
        QTimer.singleShot(700, self._recover_interrupted_job)
        QTimer.singleShot(1000, self._ensure_ffmpeg_async)
        QTimer.singleShot(10000, self._startup_update_check)

    def _scroll_page(self, content: QWidget) -> QScrollArea:
        # These pages are vertical-scroll-only.  Do NOT apply SetMinimumSize to
        # the whole page: on Windows that can preserve a pre-maximize width from
        # a child/layout sizeHint, while the horizontal scrollbar is deliberately
        # disabled.  The result is a clipped right edge until the user restores
        # and maximizes the window again.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        try:
            lay = content.layout()
            if lay is not None:
                lay.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        except Exception:
            pass
        # Ignore horizontal size hints so long labels/path controls cannot make
        # the hidden horizontal dimension wider than the visible viewport.
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        scroll.setWidget(content)
        self._main_scroll_pages.append(scroll)
        return scroll

    def _schedule_layout_refresh(self):
        if self._layout_refresh_pending or self._closing:
            return
        self._layout_refresh_pending = True
        QTimer.singleShot(0, self._refresh_main_layout)

    def _refresh_main_layout(self):
        """Recalculate width-sensitive tab contents after a native resize/state change."""
        self._layout_refresh_pending = False
        if self._closing:
            return
        try:
            central = self.centralWidget()
            if central is not None and central.layout() is not None:
                central.layout().invalidate()
                central.layout().activate()
            self.tabs.updateGeometry()
            for scroll in self._main_scroll_pages:
                content = scroll.widget()
                if content is None:
                    continue
                content.setMinimumWidth(0)
                content.updateGeometry()
                lay = content.layout()
                if lay is not None:
                    lay.invalidate()
                    lay.activate()
                scroll.updateGeometry()
                scroll.viewport().update()
        except RuntimeError:
            # A late queued refresh can race with application shutdown.
            pass

    def _build(self):
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)
        hdr = QHBoxLayout(); title = QLabel("MiniMax H3 INT4"); title.setObjectName("title")
        self.status = QLabel("Checking install…"); self.status.setObjectName("status")
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(self.status); outer.addLayout(hdr)

        # FrameVision already has its own system monitor. Avoid starting MiniMax HUD
        # polling/NVML when embedded; retain a harmless target for Settings code.
        if self._embedded:
            self.system_hud = QLabel("")
            self.system_hud.setVisible(False)
        else:
            self.system_hud = SystemHud(self)
            outer.addWidget(self.system_hud, 0)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)  # tab strip itself never scrolls
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)
        self._build_generation_tab()
        self._build_prompt_builder_tab()
        self._build_queue_tab()
        # FrameVision has its own Music Clip Creator location. Keep the MiniMax
        # Music Clip Creator available only in the standalone application.
        if not self._embedded:
            self._build_music_clip_tab()
        else:
            self.music_clip_tab_index = -1
        self._build_settings_tab()

        # Fixed bottom bar: remains visible on every tab and while tab contents scroll.
        bar = QWidget(); bar.setObjectName("bottomBar")
        controls = QHBoxLayout(bar); controls.setContentsMargins(8, 8, 8, 8)
        self.gen = QPushButton("Generate"); self.gen.setObjectName("primary"); self.gen.clicked.connect(self._main_generate_action)
        self.cancel = QPushButton("Cancel"); self.cancel.clicked.connect(self.cancel_job); self.cancel.setEnabled(False)
        val = QPushButton("Validate install"); val.clicked.connect(self.validate_install)
        self.install_minimax = QPushButton("Install MiniMax H3 model & repo")
        self.install_minimax.clicked.connect(self._start_minimax_bootstrap)
        self.install_minimax_progress = QProgressBar()
        self.install_minimax_progress.setRange(0, 0)
        self.install_minimax_progress.setTextVisible(False)
        self.install_minimax_progress.setFixedWidth(150)
        self.install_minimax_progress.setVisible(False)
        self.openout = QPushButton("Open output folder"); self.openout.clicked.connect(self.open_output_folder)
        controls.addWidget(self.gen); controls.addWidget(self.cancel); controls.addStretch(); controls.addWidget(self.install_minimax_progress); controls.addWidget(self.install_minimax); controls.addWidget(val); controls.addWidget(self.openout)
        outer.addWidget(bar, 0)
        self.setCentralWidget(root)
        self._add_tooltips()
        self.tabs.currentChanged.connect(self._sync_main_generate_button)
        self._sync_main_generate_button(self.tabs.currentIndex())

        for cls in (QComboBox, QSpinBox, QDoubleSpinBox):
            for w in self.findChildren(cls):
                w.installEventFilter(self.wheel_filter)
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _build_generation_tab(self):
        body = QWidget(); v = QVBoxLayout(body); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(10)
        basic = QGroupBox("Generation"); form = QFormLayout(basic)
        self.mode = QComboBox(); self.mode.addItems(["Text to video (T2VA)", "Image / Continue Video (FL2VA)", "Reference to video (Ref2VA)"]); self.mode.currentIndexChanged.connect(self._sync_mode)
        self.aspect = QComboBox(); self.aspect.addItems(["16:9", "9:16", "1:1"]); self.aspect.currentTextChanged.connect(self._sync_resolution)
        self.res_class = QComboBox(); self.res_class.addItems(list(RESOLUTION_PRESETS)); self.res_class.setCurrentText(DEFAULT_RESOLUTION); self.res_class.currentTextChanged.connect(self._sync_resolution)
        self.resolved = QLabel()
        self.lanczos_scale_2x = QCheckBox("Use Lanczos scaling")
        self.lanczos_scale_2x.setChecked(False)
        self.lanczos_scale_2x.setToolTip(
            "When enabled, the saved video is resized to 2× its generated width and height using FFmpeg's Lanczos scaler. "
            "Example: 1280×720 is saved as 2560×1440. This increases output resolution and file size; it does not generate additional model detail."
        )
        self.frames = QComboBox()
        for x in FRAME_PRESETS:
            self.frames.addItem(f"{x} frames — {x / 24.0:.2f} s", x)
        self._set_frame_count(362)
        self.experimental_long_duration = QCheckBox("Experimental long duration")
        self.experimental_long_duration.setChecked(False)
        self.experimental_long_duration.toggled.connect(self._sync_long_duration_mode)
        self.steps = QSpinBox(); self.steps.setRange(1, 100); self.steps.setValue(15)
        self.seed = QSpinBox(); self.seed.setRange(-1, 98_999_999); self.seed.setValue(-1); self.seed.setSpecialValueText("-1 (random)")
        form.addRow("Mode", self.mode)
        rr = QHBoxLayout(); rr.addWidget(self.res_class); rr.addWidget(self.aspect); rr.addWidget(self.resolved); rr.addWidget(self.lanczos_scale_2x); rr.addStretch(); form.addRow("Resolution", rr)
        form.addRow("Frames", self.frames); form.addRow("", self.experimental_long_duration); form.addRow("Steps", self.steps); form.addRow("Seed", self.seed)
        v.addWidget(basic)

        pg = QGroupBox("Prompt"); pgl = QVBoxLayout(pg)
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText("Describe the full MiniMax H3 video, including sound/dialogue/music directions when wanted."); self.prompt.setMinimumHeight(150)
        pgl.addWidget(self.prompt); v.addWidget(pg)

        self.fl_group = QGroupBox("FL2VA visual conditioning"); fl = QFormLayout(self.fl_group)
        self.first = FileRow("First frame image", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        self.last = FileRow("Last frame image", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        self.continue_video = VideoPathRow("optional source video to continue")
        self.continue_context = QComboBox()
        for n in (22, 39, 56, 73, 90, 107): self.continue_context.addItem(f"{n} history frames ({n/24:.2f} s)", n)
        self.continue_context.setCurrentIndex(1)
        self.glue_results = QCheckBox("Glue results")
        self.glue_results.setChecked(False)
        self.continue_last_result = QCheckBox("Continue last result")
        self.continue_last_result.setChecked(False)
        self.continue_last_result.toggled.connect(self._sync_continue_video_options)
        self.continue_audio_memory_row = QWidget()
        caml = QHBoxLayout(self.continue_audio_memory_row); caml.setContentsMargins(0, 0, 0, 0)
        self.continue_audio_memory = QCheckBox("Use sound in memory for new clip")
        self.continue_audio_memory.setChecked(False)
        self.continue_audio_memory.setToolTip("Carries the final 1.00 s of source audio as H3 history. The 24-frame / 40-step window is end-aligned to the new clip on H3's native 40 Hz audio timeline.")
        self.continue_audio_memory_warning = QLabel("1.00 s timeline-aligned test")
        self.continue_audio_memory_warning.setStyleSheet("color: #d58a00;")
        self.continue_audio_memory_warning.setToolTip("Test patch: full audio history ends exactly at target time zero; no guessed boundary-latent deletion is used.")
        caml.addWidget(self.continue_audio_memory); caml.addWidget(self.continue_audio_memory_warning); caml.addStretch()
        fl.addRow("First frame", self.first); fl.addRow("Last frame", self.last)
        fl.addRow("Continue video", self.continue_video); fl.addRow("Motion context", self.continue_context)
        fl.addRow("", self.glue_results); fl.addRow("", self.continue_last_result); fl.addRow("", self.continue_audio_memory_row); v.addWidget(self.fl_group)

        self.ref_group = QGroupBox("Ref2VA references"); rfl = QVBoxLayout(self.ref_group)
        note = QLabel("Prompt tags follow the native order: <Picture 1..9>, <Audio n> paired before <Video n>, then standalone <Audio n>."); note.setWordWrap(True); rfl.addWidget(note)
        self.ref_size = QComboBox(); self.ref_size.addItems(["match", "max"])
        rs = QFormLayout(); rs.addRow("Reference image size", self.ref_size); rfl.addLayout(rs)
        self.ref_images = RefList("Reference images (max 9)", "Images (*.png *.jpg *.jpeg *.webp *.bmp)", 9)
        self.ref_videos = RefList("Reference videos (max 3; soundtrack extracted when present)", "Video (*.mp4 *.mov *.mkv *.webm *.avi)", 3)
        self.ref_audios = RefList("Standalone reference audio (max 3)", "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg)", 3)
        rfl.addWidget(self.ref_images); rfl.addWidget(self.ref_videos); rfl.addWidget(self.ref_audios); v.addWidget(self.ref_group)

        loras = QGroupBox("LoRA adapters (up to 3)"); lf = QFormLayout(loras)
        lnote = QLabel(f"Optional MiniMax H3 diffusion-model LoRAs. Browse starts in {DEFAULT_LORA_DIR}. Files from any other folder can also be selected. Strength 1.0 = normal; 0 disables that slot.")
        lnote.setWordWrap(True)
        lnote.setMinimumWidth(0)
        lf.addRow(lnote)
        self.lora_rows = []
        for i in range(3):
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
            path = LoraPathRow(f"LoRA {i+1} (.safetensors)")
            strength = QDoubleSpinBox(); strength.setRange(-10.0, 10.0); strength.setDecimals(2); strength.setSingleStep(0.05); strength.setValue(1.0); strength.setMinimumWidth(90)
            strength.setToolTip("LoRA model strength. 1.0 = trained strength; 0 disables this slot; negative values are allowed for compatible LoRAs.")
            rl.addWidget(path, 1); rl.addWidget(QLabel("Strength")); rl.addWidget(strength)
            lf.addRow(f"LoRA {i+1}", row); self.lora_rows.append((path, strength))
        v.addWidget(loras)

        adv = QGroupBox("Advanced sampler"); af = QFormLayout(adv)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(0, 30); self.cfg.setDecimals(2); self.cfg.setSingleStep(.1); self.cfg.setValue(1.0)
        self.shift = QDoubleSpinBox(); self.shift.setRange(.01, 100); self.shift.setDecimals(2); self.shift.setValue(12.0)
        self.audio_shift = QDoubleSpinBox(); self.audio_shift.setRange(.01, 100); self.audio_shift.setDecimals(2); self.audio_shift.setValue(3.0)
        self.sampler = QComboBox(); self.sampler.addItems(SAMPLERS); self.sampler.setCurrentText("euler")
        self.scheduler = QComboBox(); self.scheduler.addItems(SCHEDULERS); self.scheduler.setCurrentText("simple")
        af.addRow("CFG", self.cfg); af.addRow("Video shift", self.shift); af.addRow("Audio shift", self.audio_shift); af.addRow("Sampler", self.sampler); af.addRow("Scheduler", self.scheduler)
        v.addWidget(adv)

        preset = QGroupBox("Presets"); pf = QFormLayout(preset)
        prow = QHBoxLayout(); self.preset_name = QLineEdit(); self.preset_name.setPlaceholderText("preset name")
        ps = QPushButton("Save preset"); pl = QPushButton("Load preset…"); safe = QPushButton("Safe BAT preset")
        ps.clicked.connect(self.save_named); pl.clicked.connect(self.load_named); safe.clicked.connect(self.safe_preset)
        prow.addWidget(self.preset_name, 1); prow.addWidget(ps); prow.addWidget(pl); prow.addWidget(safe); pf.addRow("Preset", prow); v.addWidget(preset)
        v.addStretch(1)
        self.tabs.addTab(self._scroll_page(body), "Generation")

    def _builder_root(self) -> Path:
        return ROOT / "h3_prompt_builder"

    def _build_prompt_builder_tab(self):
        # Deliberately a normal tab, not a QScrollArea. The embedded web app owns
        # its own internal scrolling, while the application's tab bar stays fixed.
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top = QHBoxLayout()
        self.builder_status = QLabel("Prompt Builder: waiting to start")
        self.builder_status.setObjectName("status")
        self.builder_start_btn = QPushButton("Start / restart")
        self.builder_start_btn.clicked.connect(self._start_prompt_builder)
        self.builder_browser_btn = QPushButton("Open in browser")
        self.builder_browser_btn.clicked.connect(self._open_prompt_builder_browser)
        self.builder_transfer_btn = QPushButton("Use prompt in Generation")
        self.builder_transfer_btn.setObjectName("primary")
        self.builder_transfer_btn.clicked.connect(self._transfer_prompt_from_builder)
        top.addWidget(self.builder_status, 1)
        top.addWidget(self.builder_start_btn)
        top.addWidget(self.builder_browser_btn)
        top.addWidget(self.builder_transfer_btn)
        layout.addLayout(top)

        if QWebEngineView is not None:
            self.prompt_webview = QWebEngineView(page)
            if PromptBuilderPage is not None:
                self.prompt_webview.setPage(PromptBuilderPage(self.prompt_webview))
            layout.addWidget(self.prompt_webview, 1)
        else:
            fallback = QLabel(
                "Qt WebEngine is not available in this environment. The Prompt Builder can still run locally; "
                "use 'Open in browser' and paste the finished prompt into Generation manually."
            )
            fallback.setWordWrap(True)
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback, 1)

        self.tabs.addTab(page, "Prompt Builder")
        self.builder_timer = QTimer(self)
        self.builder_timer.setInterval(1500)
        self.builder_timer.timeout.connect(self._poll_prompt_builder_status)
        self.builder_timer.start()
        self._prompt_builder_tab_index = self.tabs.indexOf(page)
        if self._embedded:
            self.tabs.currentChanged.connect(self._on_embedded_tab_changed)
        else:
            QTimer.singleShot(500, self._start_prompt_builder)

    def _on_embedded_tab_changed(self, index: int):
        if not self._embedded or index != getattr(self, "_prompt_builder_tab_index", -1):
            return
        if self.builder_port or (self.builder_process is not None and self.builder_process.poll() is None):
            return
        QTimer.singleShot(0, self._start_prompt_builder)

    def _port_is_free(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                return sock.connect_ex(("127.0.0.1", int(port))) != 0
        except Exception:
            return False

    def _builder_status_on_port(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/api/status", timeout=0.7) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
            return bool(payload.get("ok")) and payload.get("app") == "hailuo-h3-prompt-builder"
        except Exception:
            return False

    def _choose_prompt_builder_port(self, preferred: int = 8785) -> int:
        for port in (preferred, 8786, 8787, 8788, 8789, 8790):
            if self._builder_status_on_port(port) or self._port_is_free(port):
                return int(port)
        return int(preferred)


    def _set_prompt_builder_template_default(self):
        """Force Prompt Builder back to its built-in/template backend for each app session.

        This deliberately does not erase the saved GGUF/Ollama paths or model choices;
        it only changes the active backend to the built-in prompt engine so reopening
        the standalone app cannot start a local LLM unexpectedly.
        """
        if self.builder_template_default_applied or not self.builder_port:
            return
        base = f"http://127.0.0.1:{int(self.builder_port)}"
        try:
            body = json.dumps({"backend": "builtin"}).encode("utf-8")
            req = urllib.request.Request(
                base + "/api/settings/llm", data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=2.0).read()
            self.builder_template_default_applied = True
        except Exception:
            # The poller will retry once the local Prompt Builder server is ready.
            return

    def _start_prompt_builder(self):
        self.builder_template_default_applied = False
        builder_root = self._builder_root()
        server_py = builder_root / "server.py"
        if not server_py.is_file():
            self.builder_status.setText(f"Prompt Builder files missing: {builder_root}")
            return
        if self.builder_process is not None and self.builder_process.poll() is None:
            self.builder_status.setText(f"Prompt Builder running — {self.builder_url}")
            return

        self.builder_port = self._choose_prompt_builder_port(8785)
        self.builder_url = f"http://127.0.0.1:{self.builder_port}"
        if self._builder_status_on_port(self.builder_port):
            self.builder_status.setText(f"Prompt Builder ready — {self.builder_url}")
            self._set_prompt_builder_template_default()
            self._load_prompt_builder_view()
            return

        try:
            python_exe = str(PYTHON if PYTHON.is_file() else Path(sys.executable))
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.builder_process = subprocess.Popen(
                [python_exe, str(server_py), "--port", str(self.builder_port)],
                cwd=str(builder_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            self.builder_status.setText(f"Starting Prompt Builder — {self.builder_url}")
        except Exception as exc:
            self.builder_status.setText(f"Prompt Builder failed to start: {exc}")

    def _poll_prompt_builder_status(self):
        if not self.builder_port:
            return
        if self._builder_status_on_port(self.builder_port):
            self.builder_status.setText(f"Prompt Builder ready — {self.builder_url}")
            self._set_prompt_builder_template_default()
            self._load_prompt_builder_view()
        elif self.builder_process is not None and self.builder_process.poll() is not None:
            self.builder_status.setText("Prompt Builder stopped")

    def _load_prompt_builder_view(self):
        if self.prompt_webview is None or not self.builder_url or self.builder_loaded_url == self.builder_url:
            return
        self.prompt_webview.setUrl(QUrl(self.builder_url))
        self.builder_loaded_url = self.builder_url
        def force_template_backend(ok=True):
            if not ok or self.prompt_webview is None:
                return
            js = r"""(() => {
                try {
                    const key = 'h3-simple-settings';
                    const saved = JSON.parse(localStorage.getItem(key) || '{}');
                    saved.backend = 'builtin';
                    localStorage.setItem(key, JSON.stringify(saved));
                    const sel = document.getElementById('llmBackend');
                    if (sel) {
                        sel.value = 'builtin';
                        sel.dispatchEvent(new Event('change', {bubbles:true}));
                    }
                } catch (_) {}
            })();"""
            self.prompt_webview.page().runJavaScript(js)
        def sync_generation_state(ok=True):
            if ok:
                self._sync_prompt_builder_job_state()
        try:
            self.prompt_webview.loadFinished.connect(force_template_backend, Qt.ConnectionType.SingleShotConnection)
            self.prompt_webview.loadFinished.connect(sync_generation_state, Qt.ConnectionType.SingleShotConnection)
        except Exception:
            pass

    def _open_prompt_builder_browser(self):
        if not self.builder_url or not self._builder_status_on_port(self.builder_port):
            self._start_prompt_builder()
        if self.builder_url:
            QDesktopServices.openUrl(QUrl(self.builder_url))

    def _transfer_prompt_from_builder(self):
        if self.prompt_webview is None:
            QMessageBox.information(
                self, "Prompt Builder",
                "Embedded web view is unavailable. Open the Prompt Builder in your browser and paste the finished prompt manually."
            )
            return
        js = """(() => {
            const out = document.getElementById('promptOutput');
            const dur = document.getElementById('duration');
            const active = document.querySelector('#ratioButtons button.active');
            return JSON.stringify({
                prompt: out ? (out.value || '') : '',
                duration: dur ? (dur.value || '') : '',
                ratio: active ? (active.dataset.value || active.textContent || '') : ''
            });
        })();"""
        self.prompt_webview.page().runJavaScript(js, self._apply_prompt_builder_payload)


    def _frame_count(self) -> int:
        data = self.frames.currentData()
        if data is not None:
            try:
                return int(data)
            except Exception:
                pass
        # Backward compatibility if an older/custom combo item contains only a number.
        raw = self.frames.currentText().strip().split()[0]
        return int(raw)

    def _set_frame_count(self, value) -> None:
        try:
            wanted = int(value)
        except Exception:
            wanted = 362
        idx = self.frames.findData(wanted)
        if idx < 0:
            wanted = self._nearest_frame_preset(wanted)
            idx = self.frames.findData(wanted)
        if idx >= 0:
            self.frames.setCurrentIndex(idx)

    def _nearest_frame_preset(self, wanted: int) -> int:
        presets = EXPERIMENTAL_FRAME_PRESETS if getattr(self, "experimental_long_duration", None) and self.experimental_long_duration.isChecked() else FRAME_PRESETS
        return min(presets, key=lambda n: (abs(n - wanted), n))

    def _sync_long_duration_mode(self, enabled):
        """Expose H3's native-grid long-duration research values without changing normal mode."""
        current = self._frame_count()
        presets = EXPERIMENTAL_FRAME_PRESETS if bool(enabled) else FRAME_PRESETS
        self.frames.blockSignals(True)
        self.frames.clear()
        for x in presets:
            self.frames.addItem(f"{x} frames — {x / 24.0:.2f} s", x)
        if not enabled and current > NORMAL_FRAME_MAX:
            current = NORMAL_FRAME_MAX
        nearest = min(presets, key=lambda n: (abs(n - current), n))
        idx = self.frames.findData(nearest)
        if idx >= 0:
            self.frames.setCurrentIndex(idx)
        self.frames.blockSignals(False)
        if enabled:
            self.frames.setToolTip(
                "MiniMax H3 long-duration research mode. Values follow the model's native 17k+5 frame grid at 24 FPS, "
                "through 2385 frames = 99.375 seconds. Values above 30 seconds require this experimental mode. "
                "The maximum frame count that will actually run depends on the available system RAM and GPU VRAM."
            )
        else:
            self.frames.setToolTip(
                "Fixed MiniMax H3 frame count at 24 FPS. Normal mode stops at 719 frames = 29.96 seconds. "
                "Enable Experimental long duration for values above 30 seconds, up to 2385 frames = 99.38 seconds. "
                "The maximum frame count that will actually run depends on the available system RAM and GPU VRAM. Default: 362 frames."
            )

    def _apply_prompt_builder_payload(self, payload):
        try:
            data = json.loads(payload or "{}") if isinstance(payload, str) else (payload or {})
        except Exception:
            data = {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            QMessageBox.information(self, "Prompt Builder", "There is no generated prompt to transfer yet.")
            return
        self.prompt.setPlainText(prompt)

        ratio = str(data.get("ratio", "")).strip()
        if ratio in ("16:9", "9:16", "1:1"):
            self.aspect.setCurrentText(ratio)

        # Builder duration is expressed in seconds. This standalone intentionally
        # exposes only its approved fixed frame values, so select the nearest one
        # rather than creating an arbitrary frame count.
        try:
            seconds = float(str(data.get("duration", "")).strip() or "0")
        except Exception:
            seconds = 0.0
        if seconds > 0:
            wanted = min(NORMAL_FRAME_MAX, max(FRAME_PRESETS[0], int(round(seconds * 24))))
            chosen = self._nearest_frame_preset(wanted)
            self._set_frame_count(chosen)

        self.tabs.setCurrentIndex(0)
        self.status.setText("Prompt imported from Prompt Builder")

    def _stop_prompt_builder(self):
        if self.builder_timer is not None:
            self.builder_timer.stop()
        if self.builder_port and self._builder_status_on_port(self.builder_port):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{int(self.builder_port)}/api/shutdown",
                    data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
                )
                urllib.request.urlopen(req, timeout=1.0).read()
            except Exception:
                pass
        if self.builder_process is not None:
            try:
                self.builder_process.terminate()
                self.builder_process.wait(timeout=2.0)
            except Exception:
                try: self.builder_process.kill()
                except Exception: pass
        self.builder_process = None

    def _build_queue_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8,8,8,8)
        layout.setSpacing(8)

        # Queue workspace: fixed media/preview controls on the left, independently
        # scrolling queue lists on the right.  The left side never scrolls.
        splitter = QSplitter(Qt.Orientation.Horizontal, page)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        # ---- LEFT: fixed preview/player -----------------------------------------
        left = QWidget(splitter)
        left.setObjectName("QueuePreviewPane")
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(8)

        prev = QGroupBox("Preview")
        pv = QVBoxLayout(prev)
        pv.setContentsMargins(8,8,8,8)
        pv.setSpacing(8)
        self.preview_label = QLabel("Double-click a finished job to preview it. Mouse wheel = zoom; drag = pan.")
        self.preview_label.setWordWrap(True)
        pv.addWidget(self.preview_label)

        if QMediaPlayer is not None and QGraphicsVideoItem is not None:
            self.preview_scene = QGraphicsScene(self)
            self.preview_view = ZoomVideoView()
            self.preview_view.setScene(self.preview_scene)
            self.preview_view.setMinimumHeight(320)
            self.preview_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.video_item = QGraphicsVideoItem()
            self.video_item.setSize(QSizeF(960,540))
            self.preview_scene.addItem(self.video_item)
            self.media_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_item)
            self.media_player.positionChanged.connect(self._preview_position_changed)
            self.media_player.durationChanged.connect(self._preview_duration_changed)
            self.media_player.mediaStatusChanged.connect(self._preview_media_status)
            pv.addWidget(self.preview_view, 1)

            # Keep transport controls with the player.  The seek bar gets its own
            # row so it remains usable even when the window is not maximized.
            transport = QHBoxLayout()
            self.preview_play = QPushButton("Play / Pause")
            self.preview_stop = QPushButton("Stop")
            self.preview_repeat = QCheckBox("Repeat")
            self.preview_reset = QPushButton("Reset zoom")
            self.preview_play.clicked.connect(self._preview_toggle)
            self.preview_stop.clicked.connect(self.media_player.stop)
            self.preview_reset.clicked.connect(self.preview_view.reset_view)
            transport.addWidget(self.preview_play)
            transport.addWidget(self.preview_stop)
            transport.addWidget(self.preview_repeat)
            transport.addStretch(1)
            transport.addWidget(self.preview_reset)
            pv.addLayout(transport)

            seekrow = QHBoxLayout()
            self.preview_slider = QSlider(Qt.Orientation.Horizontal)
            self.preview_slider.setRange(0,0)
            self.preview_slider.sliderMoved.connect(self.media_player.setPosition)
            self.preview_time = QLabel("00:00 / 00:00")
            self.preview_time.setMinimumWidth(105)
            seekrow.addWidget(self.preview_slider, 1)
            seekrow.addWidget(self.preview_time)
            pv.addLayout(seekrow)
        else:
            self.preview_view = None
            self.media_player = None
            fallback = QLabel("Qt Multimedia is unavailable in this environment, so embedded playback is disabled. Double-click still opens the generated clip with the system player.")
            fallback.setWordWrap(True)
            pv.addWidget(fallback)

        left_layout.addWidget(prev, 1)
        left_layout.addStretch(0)

        # ---- RIGHT: queue lists, this side alone scrolls -------------------------
        right_scroll = QScrollArea(splitter)
        right_scroll.setObjectName("QueueJobsScrollArea")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setMinimumWidth(520)

        right_content = QWidget()
        right_content.setObjectName("QueueJobsScrollContent")
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0,0,4,0)
        right_layout.setSpacing(8)
        right_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        right_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.queue_summary = QLabel("Queue: 0 running • 0 pending • 0 finished/failed")
        self.queue_summary.setObjectName("queueSummary")
        self.queue_summary.setToolTip("Live queue counts. Jobs are persisted in presets\\setsave\\minimax_h3_queue.json.")
        right_layout.addWidget(self.queue_summary)

        self.running_tree = self._make_queue_tree(["Job", "Started", "Elapsed", "Progress", "Resolution", "Seed", "Dependency", "Source resolved", "Output", "Model"])
        self.pending_tree = self._make_queue_tree(["Job", "Queued", "Status", "Resolution", "Seed", "Dependency", "Source resolved", "Output", "Model"])
        self.finished_tree = self._make_queue_tree(["Job", "Done at", "Status", "Took", "Resolution", "Duration", "Seed", "Dependency", "Source resolved", "Output", "Model"])

        # Enough room to inspect several jobs at once.  Each tree keeps its own
        # scrollbar for longer lists, while the right column itself can scroll.
        self.running_tree.setMinimumHeight(130)
        self.pending_tree.setMinimumHeight(190)
        self.finished_tree.setMinimumHeight(190)

        self.running_group = QGroupBox("Running jobs (0)")
        rg = QVBoxLayout(self.running_group); rg.addWidget(self.running_tree)
        right_layout.addWidget(self.running_group)
        self.pending_group = QGroupBox("Pending jobs (0)")
        pg = QVBoxLayout(self.pending_group); pg.addWidget(self.pending_tree)
        right_layout.addWidget(self.pending_group)
        self.finished_group = QGroupBox("Finished / failed (0)")
        fg = QVBoxLayout(self.finished_group); fg.addWidget(self.finished_tree)
        right_layout.addWidget(self.finished_group)

        self.running_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.running_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.running_tree,pos,"running"))
        self.pending_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pending_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.pending_tree,pos,"pending"))
        self.finished_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.finished_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.finished_tree,pos,"finished"))
        self.finished_tree.itemDoubleClicked.connect(lambda item,col:self._play_job_item(item))

        clearrow = QHBoxLayout()
        clearrow.addStretch(1)
        self.clear_cancelled_btn = QPushButton("Clear cancelled")
        self.clear_cancelled_btn.setToolTip("Remove cancelled jobs from this queue history only. Files on disk are not deleted.")
        self.clear_cancelled_btn.clicked.connect(self._clear_cancelled_jobs)
        clearrow.addWidget(self.clear_cancelled_btn)
        self.clear_failed_btn = QPushButton("Clear failed")
        self.clear_failed_btn.setToolTip("Remove failed jobs from this queue history only. Files on disk are not deleted.")
        self.clear_failed_btn.clicked.connect(self._clear_failed_jobs)
        clearrow.addWidget(self.clear_failed_btn)
        self.clear_finished_btn = QPushButton("Clear finished / failed jobs")
        self.clear_finished_btn.setToolTip("Remove finished and failed jobs from this queue history only. Output files on disk are not deleted.")
        self.clear_finished_btn.clicked.connect(self._clear_finished_jobs)
        clearrow.addWidget(self.clear_finished_btn)
        right_layout.addLayout(clearrow)
        right_layout.addStretch(1)

        right_scroll.setWidget(right_content)
        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([520, 650])
        layout.addWidget(splitter, 1)

        self.tabs.addTab(page,"Queue")
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(500)
        self.queue_timer.timeout.connect(self._queue_tick)
        self.queue_timer.start()
        self._refresh_queue_views()

    def _make_queue_tree(self, headers):
        t=QTreeWidget()
        t.setColumnCount(len(headers)); t.setHeaderLabels(headers); t.setRootIsDecorated(False)
        t.setAlternatingRowColors(True); t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setMinimumHeight(92)
        t.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents); t.header().setStretchLastSection(True)
        # Do not rely on the native Windows palette here.  Some dark-theme combinations
        # produced a white viewport/header with near-white text, making real queue rows look empty.
        t.setStyleSheet("""
            QTreeWidget { background-color:#0b0f14; alternate-background-color:#101821; color:#e8eef6; border:1px solid #334556; }
            QTreeWidget::item { color:#e8eef6; min-height:24px; padding:2px; }
            QTreeWidget::item:selected { background:#176c89; color:#ffffff; }
            QHeaderView::section { background:#18232e; color:#dce8f3; border:0; border-right:1px solid #334556; border-bottom:1px solid #334556; padding:5px; }
        """)
        return t

    def _job_by_id(self, jid):
        return next((j for j in self.queue_jobs if j.get("id")==jid),None)

    def _ensure_queue_job_numbers(self):
        """Give every persisted queue entry a stable human-readable Job #."""
        used=set()
        for job in self.queue_jobs:
            try:
                n=int(job.get("job_number") or 0)
            except Exception:
                n=0
            if n > 0 and n not in used:
                job["job_number"]=n
                used.add(n)
            else:
                job.pop("job_number", None)

        # Old queue files had UUIDs only. Assign their numbers chronologically,
        # without changing the queue's actual list/order semantics.
        missing=[(i,j) for i,j in enumerate(self.queue_jobs) if not j.get("job_number")]
        missing.sort(key=lambda pair:(float(pair[1].get("created_at") or 0), pair[0]))
        next_number=max(used, default=0)+1
        for _, job in missing:
            while next_number in used:
                next_number += 1
            job["job_number"]=next_number
            used.add(next_number)
            next_number += 1

        # Cache the dependency's display number in the child too, so a historical
        # source can still be identified after its queue-history row is removed.
        by_id={j.get("id"):j for j in self.queue_jobs if j.get("id")}
        for job in self.queue_jobs:
            dep_id=job.get("continue_from_job_id")
            if dep_id and not job.get("continue_from_job_number"):
                dep=by_id.get(dep_id)
                if dep and dep.get("job_number"):
                    job["continue_from_job_number"]=int(dep["job_number"])

        self._next_job_number_value=max(int(getattr(self,"_next_job_number_value",1) or 1), max(used, default=0)+1)

    def _take_next_job_number(self):
        self._ensure_queue_job_numbers()
        number=int(self._next_job_number_value)
        self._next_job_number_value=number+1
        return number

    def _job_number_text(self, job):
        try:
            return f"Job #{int(job.get('job_number'))}"
        except Exception:
            return "Job #?"

    def _dependency_display(self, job):
        if not job.get("continue_last_result"):
            return "—"
        dep=self._job_by_id(job.get("continue_from_job_id"))
        dep_number=job.get("continue_from_job_number")
        if dep_number is None and dep is not None:
            dep_number=dep.get("job_number")
        try:
            label=f"Job #{int(dep_number)}"
        except Exception:
            label="previous job"

        if dep is None:
            return f"Missing: {label}"
        if job.get("state")=="pending" and dep.get("state") != "finished":
            return f"Waiting for: {label}"
        if job.get("state")=="failed" and dep.get("state") != "finished":
            return f"Blocked by: {label}"
        if job.get("state")=="failed" and not Path(dep.get("output", "")).is_file():
            return f"Blocked by: {label}"
        return f"Continues: {label}"

    def _resolved_source_display(self, job):
        source=str(job.get("resolved_continue_source") or "")
        if not source and not job.get("continue_last_result"):
            source=str(job.get("manual_continue_video") or "")
        if not source:
            return "—"
        return Path(source).name or source

    def _apply_queue_dependency_tooltips(self, item, job, dependency_col, source_col):
        dep_id=job.get("continue_from_job_id")
        if dep_id:
            dep=self._job_by_id(dep_id)
            dep_number=job.get("continue_from_job_number") or (dep.get("job_number") if dep else None)
            detail=f"Dependency: Job #{dep_number}" if dep_number else "Dependency: previous queue job"
            if dep:
                detail += f"\nState: {dep.get('state','unknown')}\nExpected output: {dep.get('output','—')}"
            else:
                detail += "\nThis dependency is no longer present in queue history."
            item.setToolTip(dependency_col, detail)
        source=str(job.get("resolved_continue_source") or job.get("manual_continue_video") or "")
        if source:
            item.setToolTip(source_col, source)

    def _fmt_clock(self, epoch):
        if not epoch: return "—"
        return time.strftime("%H:%M:%S", time.localtime(float(epoch)))

    def _fmt_elapsed(self, seconds):
        seconds=max(0,int(seconds or 0)); h,rem=divmod(seconds,3600); m,s=divmod(rem,60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _short_model(self, job):
        return job.get("model_label") or ("Ref2VA INT4" if job.get("mode")==2 else "FL2VA INT4")

    def _probe_clip_duration(self, output_path, frames=None):
        """Return finished clip duration in seconds. Prefer media metadata, fall back to frames/24."""
        path = Path(output_path or "")
        if path.is_file():
            # Never use PATH or another bundled copy: this standalone app owns one toolset in presets\bin.
            exe = ffmpeg_tool_path("ffprobe.exe")
            if exe.is_file():
                try:
                    out = subprocess.check_output(
                        [str(exe), "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                        text=True, stderr=subprocess.DEVNULL, timeout=4.0
                    ).strip()
                    value = float(out)
                    if value > 0:
                        return value
                except Exception:
                    pass
        try:
            f = int(frames or 0)
            if f > 0:
                return f / 24.0
        except Exception:
            pass
        return None

    def _fmt_clip_duration(self, seconds):
        try:
            value = float(seconds)
        except Exception:
            return "—"
        if value < 0:
            return "—"
        mins = int(value // 60)
        secs = value - mins * 60
        if mins:
            return f"{mins:02d}:{secs:04.1f}"
        return f"{secs:.1f}s"

    def _refresh_queue_views(self):
        if not hasattr(self,"running_tree"): return
        self._ensure_queue_job_numbers()
        self.running_tree.setUpdatesEnabled(False); self.pending_tree.setUpdatesEnabled(False); self.finished_tree.setUpdatesEnabled(False)
        try:
            self.running_tree.clear(); self.pending_tree.clear(); self.finished_tree.clear()
            now=time.time(); spin=("◐","◓","◑","◒")[self._spinner_index%4]
            counts={"running":0,"pending":0,"finished":0}
            normal_brush=QBrush(QColor("#e8eef6")); failed_brush=QBrush(QColor("#ff8f8f")); done_brush=QBrush(QColor("#9be7b0"))
            for j in self.queue_jobs:
                state=j.get("state")
                job_label=self._job_number_text(j)
                dependency=self._dependency_display(j)
                source=self._resolved_source_display(j)
                seed=str(j.get("actual_seed") if j.get("actual_seed") is not None else j.get("seed","—"))
                res=j.get("resolution","—"); out=Path(j.get("output","")).name or "—"; model=self._short_model(j)
                item=None
                if state=="running":
                    counts["running"]+=1
                    elapsed=now-float(j.get("started_at") or now)
                    vals=[job_label,self._fmt_clock(j.get("started_at")),self._fmt_elapsed(elapsed),"",res,seed,dependency,source,out,model]
                    item=QTreeWidgetItem(vals); item.setData(0,Qt.ItemDataRole.UserRole,j["id"]); self.running_tree.addTopLevelItem(item)
                    pb=QProgressBar(); p=j.get("progress")
                    if p is None:
                        pb.setRange(0,0); pb.setFormat(f"{spin} {j.get('phase','Working…')}")
                    else:
                        pb.setRange(0,100); pb.setValue(int(p)); pb.setFormat(f"Sampling %p%")
                    pb.setStyleSheet("QProgressBar{background:#111820;color:#e8eef6;border:1px solid #334556;text-align:center;} QProgressBar::chunk{background:#126680;}")
                    self.running_tree.setItemWidget(item,3,pb)
                    self._apply_queue_dependency_tooltips(item,j,6,7)
                elif state=="pending":
                    counts["pending"]+=1
                    pending_status = j.get("phase") or "Waiting"
                    if self._ffmpeg_setup_proc and self._ffmpeg_setup_proc.state() != QProcess.ProcessState.NotRunning:
                        pending_status = "Waiting for FFmpeg setup…"
                    elif self._ffmpeg_setup_failed and not ffmpeg_tools_ready():
                        pending_status = "FFmpeg setup failed"
                    item=QTreeWidgetItem([job_label,self._fmt_clock(j.get("created_at")),pending_status,res,seed,dependency,source,out,model]); item.setData(0,Qt.ItemDataRole.UserRole,j["id"]); self.pending_tree.addTopLevelItem(item)
                    self._apply_queue_dependency_tooltips(item,j,5,6)
                elif state in ("finished","failed","cancelled"):
                    counts["finished"]+=1
                    status="Finished" if state=="finished" else ("Cancelled" if state=="cancelled" else "Failed")
                    clip_duration = j.get("clip_duration")
                    if clip_duration is None and state == "finished":
                        clip_duration = self._probe_clip_duration(j.get("output"), j.get("frames"))
                        if clip_duration is not None:
                            j["clip_duration"] = clip_duration
                    item=QTreeWidgetItem([job_label,self._fmt_clock(j.get("finished_at")),status,self._fmt_elapsed(j.get("elapsed",0)),res,self._fmt_clip_duration(clip_duration),seed,dependency,source,out,model]); item.setData(0,Qt.ItemDataRole.UserRole,j["id"]); self.finished_tree.addTopLevelItem(item)
                    self._apply_queue_dependency_tooltips(item,j,7,8)
                    brush=done_brush if state=="finished" else failed_brush
                    for c in range(item.columnCount()): item.setForeground(c,brush)
                elif state=="interrupted":
                    counts["pending"]+=1
                    item=QTreeWidgetItem([job_label,self._fmt_clock(j.get("created_at")),"Recovery pending",res,seed,dependency,source,out,model]); item.setData(0,Qt.ItemDataRole.UserRole,j["id"]); self.pending_tree.addTopLevelItem(item)
                    item.setToolTip(0,"Interrupted by the previous application shutdown; recovery decision pending.")
                    self._apply_queue_dependency_tooltips(item,j,5,6)
                if item is not None and state not in ("finished","failed","cancelled"):
                    for c in range(item.columnCount()): item.setForeground(c,normal_brush)
            if hasattr(self,"running_group"): self.running_group.setTitle(f"Running jobs ({counts['running']})")
            if hasattr(self,"pending_group"): self.pending_group.setTitle(f"Pending jobs ({counts['pending']})")
            if hasattr(self,"finished_group"): self.finished_group.setTitle(f"Finished / failed ({counts['finished']})")
            if hasattr(self,"queue_summary"):
                base = f"Queue: {counts['running']} running • {counts['pending']} pending • {counts['finished']} finished/failed"
                if self._ffmpeg_setup_proc and self._ffmpeg_setup_proc.state() != QProcess.ProcessState.NotRunning:
                    detail = self._ffmpeg_setup_status or "Preparing local FFmpeg tools…"
                    base += "  •  " + detail
                elif self._ffmpeg_setup_failed and not ffmpeg_tools_ready():
                    base += "  •  FFmpeg setup failed"
                self.queue_summary.setText(base)
        finally:
            self.running_tree.setUpdatesEnabled(True); self.pending_tree.setUpdatesEnabled(True); self.finished_tree.setUpdatesEnabled(True)
            self.running_tree.viewport().update(); self.pending_tree.viewport().update(); self.finished_tree.viewport().update()

    def _queue_tick(self):
        # The 500 ms timer must never rebuild the queue trees. Clearing/recreating
        # the Finished tree between mouse presses destroys Qt's double-click target
        # and also resets selection/scroll position. Structural queue changes call
        # _refresh_queue_views() explicitly; the timer only updates live Running data.
        self._spinner_index += 1
        self._update_running_queue_view()
        self._sync_prompt_builder_job_state()

    def _sync_prompt_builder_job_state(self):
        """Tell the embedded Prompt Builder whether MiniMax currently owns the GPU."""
        if self.prompt_webview is None:
            return
        running = bool(
            self.current_job_id
            and self.proc
            and self.proc.state() != QProcess.ProcessState.NotRunning
        )
        try:
            self.prompt_webview.page().runJavaScript(
                f"window.__minimaxGenerationRunning = {'true' if running else 'false'};"
            )
        except Exception:
            pass

    def _update_running_queue_view(self):
        if not hasattr(self, "running_tree"):
            return
        now = time.time()
        spin = ("◐", "◓", "◑", "◒")[self._spinner_index % 4]
        jobs = {j.get("id"): j for j in self.queue_jobs if j.get("state") == "running"}

        # A start/finish/requeue operation changes the row structure and already
        # requests a full refresh. If something got out of sync, repair it once.
        if self.running_tree.topLevelItemCount() != len(jobs):
            self._refresh_queue_views()
            return

        self.running_tree.setUpdatesEnabled(False)
        try:
            for row in range(self.running_tree.topLevelItemCount()):
                item = self.running_tree.topLevelItem(row)
                job = jobs.get(item.data(0, Qt.ItemDataRole.UserRole))
                if job is None:
                    self._refresh_queue_views()
                    return

                elapsed = now - float(job.get("started_at") or now)
                item.setText(2, self._fmt_elapsed(elapsed))

                pb = self.running_tree.itemWidget(item, 3)
                if pb is None:
                    continue
                progress = job.get("progress")
                if progress is None:
                    if pb.minimum() != 0 or pb.maximum() != 0:
                        pb.setRange(0, 0)
                    pb.setFormat(f"{spin} {job.get('phase', 'Working…')}")
                else:
                    if pb.minimum() != 0 or pb.maximum() != 100:
                        pb.setRange(0, 100)
                    pb.setValue(int(progress))
                    pb.setFormat("Sampling %p%")
        finally:
            self.running_tree.setUpdatesEnabled(True)
            self.running_tree.viewport().update()

    def _queue_context_menu(self, tree, pos, section):
        item=tree.itemAt(pos)
        if not item: return
        job=self._job_by_id(item.data(0,Qt.ItemDataRole.UserRole))
        if not job: return
        menu=QMenu(self); info=menu.addAction("Show more info")
        if section=="running":
            requeue=menu.addAction("Cancel and move to bottom of Pending"); cancel=menu.addAction("Cancel job completely")
        elif section=="pending":
            delete=menu.addAction("Delete pending job")
        else:
            if job.get("state")=="finished" and Path(job.get("output","")).is_file():
                play=menu.addAction("Play clip")
                loaded=(self.preview_path and Path(self.preview_path)==Path(job.get("output","")))
                delete_disk=menu.addAction("Delete from disk + queue" + (" (currently loaded)" if loaded else ""))
            if job.get("state") in ("failed","cancelled"):
                reason=menu.addAction("Show why it failed / stopped")
            menu.addSeparator()
            remove_history=menu.addAction("Remove from list (keep file)")
        act=menu.exec(tree.viewport().mapToGlobal(pos))
        if not act: return
        if act==info: self._show_job_info(job)
        elif section=="running" and act==requeue: self._stop_running_job("requeue")
        elif section=="running" and act==cancel: self._stop_running_job("cancel")
        elif section=="pending" and act==delete:
            self.queue_jobs=[j for j in self.queue_jobs if j.get("id")!=job.get("id")]; self._save_queue_state(); self._refresh_queue_views()
        elif section=="finished" and 'play' in locals() and act==play: self._load_preview(job,autoplay=True)
        elif section=="finished" and 'delete_disk' in locals() and act==delete_disk: self._delete_job_output(job)
        elif section=="finished" and 'reason' in locals() and act==reason: QMessageBox.information(self,"Job failure / stop reason",job.get("error") or job.get("cancel_reason") or "No detailed reason was recorded.")
        elif section=="finished" and 'remove_history' in locals() and act==remove_history:
            self.queue_jobs=[j for j in self.queue_jobs if j.get("id")!=job.get("id")]
            self._save_queue_state(); self._refresh_queue_views()

    def _show_job_info(self, job):
        dlg=QDialog(self); dlg.setWindowTitle("Queue job details"); dlg.resize(760,560); lay=QVBoxLayout(dlg)
        text=QPlainTextEdit(); text.setReadOnly(True)
        data={k:v for k,v in job.items() if k not in ("args","log_tail")}; data["command_arguments"]=job.get("args",[]); data["recent_log"]=job.get("log_tail","")
        text.setPlainText(json.dumps(data,indent=2,ensure_ascii=False)); lay.addWidget(text,1)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept); lay.addWidget(bb); dlg.exec()

    def _play_job_item(self,item):
        job=self._job_by_id(item.data(0,Qt.ItemDataRole.UserRole))
        if not job or job.get("state")!="finished": return
        self._load_preview(job,autoplay=True)

    def _load_preview(self,job,autoplay=False):
        path=Path(job.get("output",""))
        if not path.is_file(): QMessageBox.warning(self,"Preview","Output file is no longer on disk."); return
        self.preview_path=str(path)
        if self.media_player is None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))); return
        self.media_player.setSource(QUrl.fromLocalFile(str(path.resolve()))); self.preview_label.setText(path.name); self.preview_view.reset_view()
        if autoplay: self.media_player.play()

    def _preview_toggle(self):
        if not self.media_player: return
        if self.media_player.playbackState()==QMediaPlayer.PlaybackState.PlayingState: self.media_player.pause()
        else: self.media_player.play()
    def _preview_duration_changed(self,d): self.preview_slider.setRange(0,max(0,int(d))); self._preview_position_changed(self.media_player.position() if self.media_player else 0)
    def _preview_position_changed(self,p):
        if not self.media_player: return
        if not self.preview_slider.isSliderDown(): self.preview_slider.setValue(int(p))
        def fmt(ms):
            sec=max(0,int(ms)//1000); return f"{sec//60:02d}:{sec%60:02d}"
        self.preview_time.setText(f"{fmt(p)} / {fmt(self.media_player.duration())}")
    def _preview_media_status(self,status):
        if self.media_player and status==QMediaPlayer.MediaStatus.EndOfMedia and self.preview_repeat.isChecked(): self.media_player.setPosition(0); self.media_player.play()

    def _delete_job_output(self,job):
        path=Path(job.get("output",""))
        if self.preview_path and Path(self.preview_path)==path and self.media_player:
            self.media_player.stop(); self.media_player.setSource(QUrl()); self.preview_path=""; self.preview_label.setText("Preview unloaded.")
        try:
            if path.is_file():
                path.unlink()
            # A deleted result must disappear from queue history as well.  Leaving
            # a finished record with a missing output makes "Continue last result"
            # select a source that no longer exists and blocks the next job.
            deleted_id=job.get("id")
            self.queue_jobs=[j for j in self.queue_jobs if j.get("id") != deleted_id]
            self._save_queue_state()
            self._refresh_queue_views()
        except Exception as exc:
            QMessageBox.critical(self,"Delete failed",str(exc))

    def _clear_finished_jobs(self):
        # Queue-history cleanup only; generated files remain untouched.
        self.queue_jobs=[j for j in self.queue_jobs if j.get("state") not in ("finished","failed")]
        self._save_queue_state(); self._refresh_queue_views()

    def _clear_failed_jobs(self):
        # Separate cleanup for failed jobs only; generated files remain untouched.
        self.queue_jobs=[j for j in self.queue_jobs if j.get("state") != "failed"]
        self._save_queue_state(); self._refresh_queue_views()

    def _clear_cancelled_jobs(self):
        # Separate cleanup requested so cancelled tests can be removed without
        # also clearing successful/failed history. Never deletes output files.
        self.queue_jobs=[j for j in self.queue_jobs if j.get("state") != "cancelled"]
        self._save_queue_state(); self._refresh_queue_views()

    def _save_queue_state(self):
        PRESET_DIR.mkdir(parents=True,exist_ok=True)
        try:
            self._ensure_queue_job_numbers()
            payload=json.dumps({"version":3,"saved_at":time.time(),"next_job_number":self._next_job_number_value,"jobs":self.queue_jobs},indent=2,ensure_ascii=False)
            tmp=QUEUE_FILE.with_suffix(QUEUE_FILE.suffix+".tmp")
            tmp.write_text(payload,encoding="utf-8")
            os.replace(str(tmp),str(QUEUE_FILE))
        except Exception as exc:
            if hasattr(self,"log"): self.append_log(f"Queue save warning: {exc}\n")

    def _load_queue_state(self):
        if not QUEUE_FILE.is_file(): self._refresh_queue_views(); return
        try:
            data=json.loads(QUEUE_FILE.read_text(encoding="utf-8")); self.queue_jobs=list(data.get("jobs",[])); self._next_job_number_value=max(1,int(data.get("next_job_number") or 1))
            # Migrate queue entries created before the inference launchers moved into helpers/.
            for j in self.queue_jobs:
                args=j.get("args")
                if isinstance(args,list) and args:
                    if args[0] == "generate.py": args[0] = "helpers/generate.py"
                    elif args[0] == "generate_ref.py": args[0] = "helpers/generate_ref.py"
            # A process cannot still belong to this freshly-started GUI. Preserve it as interrupted until user decides.
            for j in self.queue_jobs:
                if j.get("state")=="running": j["state"]="interrupted"; j["cancel_reason"]="Application closed while this job was running."
            self._ensure_queue_job_numbers()
        except Exception as exc:
            self.queue_jobs=[]
            if hasattr(self,"log"): self.append_log(f"Queue load warning: {exc}\n")
        self._refresh_queue_views()
        self._save_queue_state()

    def _recover_interrupted_job(self):
        interrupted=[j for j in self.queue_jobs if j.get("state")=="interrupted"]
        if interrupted:
            job=interrupted[-1]
            ans=QMessageBox.question(self,"Interrupted queue job","The application was closed while a job was running. Restart that last job now?\n\nYes = restart it first.\nNo = mark it cancelled and continue with pending jobs.",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.Yes)
            if ans==QMessageBox.StandardButton.Yes:
                job["state"]="pending"; job["started_at"]=None; job["finished_at"]=None; job["progress"]=None; job["phase"]="Waiting"; self.queue_jobs.remove(job); self.queue_jobs.insert(0,job)
            else:
                job["state"]="cancelled"; job["finished_at"]=time.time(); job["elapsed"]=0; job["error"]="Cancelled after application restart; previous run was interrupted by closing the app."
            for other in interrupted[:-1]:
                other["state"]="cancelled"; other["finished_at"]=time.time(); other["error"]="Interrupted by application shutdown."
            self._save_queue_state(); self._refresh_queue_views()
        self._start_next_pending()

    def _ensure_ffmpeg_async(self):
        """Install/verify the app-local FFmpeg toolset without blocking the GUI."""
        if ffmpeg_tools_ready():
            return True
        if self._ffmpeg_setup_proc and self._ffmpeg_setup_proc.state() != QProcess.ProcessState.NotRunning:
            return False
        if not PYTHON.is_file():
            return False

        FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_setup_output = ""
        self._ffmpeg_setup_status = "Preparing local FFmpeg tools…"
        self._ffmpeg_setup_failed = False
        self._refresh_queue_views()

        popup = QMessageBox(self)
        popup.setWindowTitle("Preparing FFmpeg tools")
        popup.setIcon(QMessageBox.Icon.Information)
        popup.setText("MiniMax H3 needs its local FFmpeg tools.")
        popup.setInformativeText(
            "Downloading and preparing ffmpeg.exe, ffprobe.exe and ffplay.exe in:\n"
            f"{FFMPEG_BIN_DIR}\n\nYou can close this message; setup will continue in the background."
        )
        popup.setStandardButtons(QMessageBox.StandardButton.Close)
        popup.setModal(False)
        popup.show()
        self._ffmpeg_setup_popup = popup

        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        proc.setProgram(str(PYTHON))
        proc.setArguments(["-m", "runtime.ensure_ffmpeg"])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._ffmpeg_setup_proc = proc

        def read_setup_output():
            text = bytes(proc.readAllStandardOutput()).decode(errors="replace")
            if not text:
                return
            self._ffmpeg_setup_output = (self._ffmpeg_setup_output + text)[-12000:]
            self.append_log(text)
            lines = [x.strip() for x in self._ffmpeg_setup_output.replace("\r", "\n").splitlines() if x.strip()]
            if lines:
                last = lines[-1]
                if last.startswith("Downloading FFmpeg:"):
                    self._ffmpeg_setup_status = last
                elif "Extracting" in last:
                    self._ffmpeg_setup_status = "Extracting FFmpeg tools…"
                elif "verified" in last.lower() or "ready" in last.lower():
                    self._ffmpeg_setup_status = "FFmpeg tools verified"
                else:
                    self._ffmpeg_setup_status = last[:120]
                self._refresh_queue_views()
            if lines and self._ffmpeg_setup_popup is popup and popup.isVisible():
                popup.setInformativeText(
                    "Installing the local FFmpeg tools in presets\bin. You can close this message; setup continues.\n\n"
                    + lines[-1]
                )

        def setup_done(code, status):
            read_setup_output()
            ready = ffmpeg_tools_ready()
            if ready:
                self._ffmpeg_setup_failed = False
                self._ffmpeg_setup_status = "FFmpeg tools ready"
                self.status.setText("FFmpeg tools ready")
                if self._ffmpeg_setup_popup is popup and popup.isVisible():
                    popup.setText("FFmpeg tools are ready.")
                    popup.setInformativeText(
                        f"Verified ffmpeg.exe, ffprobe.exe and ffplay.exe in:\n{FFMPEG_BIN_DIR}"
                    )
            else:
                self._ffmpeg_setup_failed = True
                self._ffmpeg_setup_status = "FFmpeg setup failed"
                self.status.setText("FFmpeg setup failed")
                detail = self._ffmpeg_setup_output.strip().splitlines()[-1] if self._ffmpeg_setup_output.strip() else f"Setup exited with code {code}."
                QMessageBox.critical(
                    self, "FFmpeg setup failed",
                    "The local FFmpeg toolset could not be prepared.\n\n" + detail +
                    f"\n\nExpected files:\n{FFMPEG_BIN_DIR / 'ffmpeg.exe'}\n{FFMPEG_BIN_DIR / 'ffprobe.exe'}\n{FFMPEG_BIN_DIR / 'ffplay.exe'}"
                )
            self._ffmpeg_setup_proc = None
            self._refresh_queue_views()
            if ready:
                QTimer.singleShot(0, self._start_next_pending)

        proc.readyReadStandardOutput.connect(read_setup_output)
        proc.finished.connect(setup_done)
        self.status.setText("Preparing FFmpeg tools…")
        proc.start()
        return False

    def _latest_non_cancelled_queue_job(self):
        # "Continue last result" should ignore cancelled tests.  Everything else
        # still counts as the latest prior queue job, including failed jobs, so
        # a failure in the chain stops continuation instead of silently skipping it.
        candidates=[j for j in self.queue_jobs if j.get("state") != "cancelled"]
        if not candidates:
            return None
        return max(candidates, key=lambda j: float(j.get("created_at") or 0))

    def _start_next_pending(self):
        if self.proc and self.proc.state()!=QProcess.ProcessState.NotRunning: return
        if self._framevision_queue_mode():
            self.current_job_id=None; self.gen.setEnabled(True); self.cancel.setEnabled(False)
            return
        if not ffmpeg_tools_ready():
            self._ensure_ffmpeg_async()
            return
        # Chained jobs are dependency-driven, not folder-driven. A pending job that
        # references another queued job is runnable only after that exact job finishes.
        # This also survives a requeue operation that temporarily changes list order.
        job=None; queue_changed=False
        for candidate in self.queue_jobs:
            if candidate.get("state") != "pending":
                continue
            dep_id = candidate.get("continue_from_job_id") if candidate.get("continue_last_result") else None
            if dep_id:
                dep = self._job_by_id(dep_id)
                if dep is None:
                    candidate["state"]="failed"; candidate["phase"]="Failed"; candidate["finished_at"]=time.time(); candidate["error"]=f"Continue dependency Job #{candidate.get('continue_from_job_number') or '?'} no longer exists in the queue history."; queue_changed=True; continue
                if dep.get("state") in ("failed","cancelled"):
                    candidate["state"]="failed"; candidate["phase"]="Failed"; candidate["finished_at"]=time.time(); candidate["error"]=f"Dependency {self._job_number_text(dep)} did not finish successfully ({dep.get('state')})."; queue_changed=True; continue
                if dep.get("state") != "finished":
                    continue
                if not Path(dep.get("output", "")).is_file():
                    candidate["state"]="failed"; candidate["phase"]="Failed"; candidate["finished_at"]=time.time(); candidate["error"]=f"Dependency {self._job_number_text(dep)} finished but its output file is missing."; queue_changed=True; continue
            job=candidate; break
        if queue_changed:
            self._save_queue_state(); self._refresh_queue_views()
        if not job:
            self.current_job_id=None; self.gen.setEnabled(True); self.cancel.setEnabled(False); return
        run_args=list(job.get("args",[]))
        continue_source=""
        if job.get("continue_last_result"):
            dep=self._job_by_id(job.get("continue_from_job_id"))
            continue_source=str(dep.get("output","")) if dep else ""
        else:
            continue_source=str(job.get("manual_continue_video") or "")
        if continue_source:
            run_args += ["--continue-video", continue_source, "--continue-context-frames", str(int(job.get("continue_context_frames") or 39))]
            if job.get("continue_last_result") and job.get("continue_audio_memory"):
                run_args += ["--continue-audio-memory"]
            if job.get("glue_results"):
                run_args += ["--glue-source", continue_source]
            job["resolved_continue_source"] = continue_source
        job["state"]="running"; job["started_at"]=time.time(); job["finished_at"]=None; job["progress"]=None; job["phase"]="Starting"; job["step_now"]=None; job["step_total"]=job.get("steps"); job["log_tail"]=""; self.current_job_id=job["id"]; self._termination_action=None; self._proc_buffer=""
        # The queue timer has started, so update the always-visible HUD immediately
        # instead of waiting for its next periodic refresh.
        if hasattr(self, "system_hud") and isinstance(self.system_hud, SystemHud):
            self.system_hud.refresh()
        self.proc=QProcess(self); self.proc.setWorkingDirectory(str(ROOT)); self.proc.setProgram(str(PYTHON)); self.proc.setArguments(run_args); self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._process_output); self.proc.finished.connect(self._finished)
        self._start_job_log_file(job)
        self.cancel.setEnabled(True); self.gen.setEnabled(True); self.status.setText("Queue: generating…"); self.append_log(f"\n=== QUEUE START ===\nOutput: {job.get('output')}\n"); self._save_queue_state(); self._refresh_queue_views(); self.proc.start()

    def _stop_running_job(self, action):
        if not self.proc or self.proc.state()==QProcess.ProcessState.NotRunning: return
        self._termination_action=action
        job=self._job_by_id(self.current_job_id)
        if job: job["cancel_reason"]="Cancelled by user and returned to pending." if action=="requeue" else "Cancelled by user."
        self.append_log("Cancelling current queue job…\n"); pid=int(self.proc.processId())
        if os.name=='nt' and pid: subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else: self.proc.kill()

    def _extract_failure_reason(self,job,code):
        txt=job.get("log_tail","")
        lines=[x.strip() for x in txt.replace("\r","\n").splitlines() if x.strip()]
        preferred=[x for x in lines if any(k in x.lower() for k in ("error","failed","exception","traceback","not found","missing"))]
        return (preferred[-1] if preferred else (lines[-1] if lines else f"Process exited with code {code}"))[:1500]

    def _framevision_queue_mode(self):
        return bool(self._embedded and hasattr(self, "use_framevision_queue") and self.use_framevision_queue.isChecked())

    def _sync_queue_preview_setting_visibility(self):
        show = bool(self.play_result_finished.isChecked()) and not self._framevision_queue_mode()
        self.play_result_queue_player.setVisible(show)
        if self._framevision_queue_mode():
            self.play_result_queue_player.setChecked(False)

    def _set_framevision_queue_mode(self, enabled):
        enabled = bool(enabled and self._embedded)
        # Keep the newer Music Clip Creator tab. Only the MiniMax internal Queue
        # tab is hidden while FrameVision owns generation scheduling.
        try:
            self.tabs.setTabVisible(2, not enabled)
        except Exception:
            pass
        if enabled:
            self.play_result_queue_player.setChecked(False)
            self.status.setText("FrameVision queue enabled — MiniMax pending jobs paused")
        else:
            self.status.setText("MiniMax internal queue enabled")
            QTimer.singleShot(0, self._start_next_pending)
        self._sync_queue_preview_setting_visibility()
        try:
            self.save_last()
        except Exception:
            pass

    def _framevision_queue_job_records(self):
        if not self._embedded:
            return []
        records = []
        jobs_root = ROOT / "jobs"
        state_map = {"pending":"pending", "running":"running", "done":"finished", "failed":"failed"}
        for folder, state in state_map.items():
            d = jobs_root / folder
            if not d.is_dir():
                continue
            for jp in d.glob("*.json"):
                if jp.name.endswith(".progress.json") or jp.name.startswith("_"):
                    continue
                try:
                    data = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(data.get("type") or "").lower() not in ("minimax_h3_generate", "minimax_h3"):
                    continue
                data = dict(data)
                data["state"] = state
                data["_queue_file"] = str(jp)
                try:
                    data["_queue_mtime"] = jp.stat().st_mtime
                except Exception:
                    data["_queue_mtime"] = 0.0
                records.append(data)
        return records

    def _latest_framevision_minimax_job(self):
        records = [j for j in self._framevision_queue_job_records() if str(j.get("status") or "").lower() not in ("cancelled", "canceled")]
        if not records:
            return None
        def key(j):
            created = j.get("created_at")
            try:
                if isinstance(created, (int, float)):
                    return float(created)
            except Exception:
                pass
            return float(j.get("_queue_mtime") or 0.0)
        return max(records, key=key)

    def _framevision_reserved_outputs(self):
        out = set()
        for j in self._framevision_queue_job_records():
            a = j.get("args") or {}
            p = j.get("produced") or j.get("output") or a.get("outfile") or a.get("out_file") or a.get("output")
            if p:
                try:
                    out.add(str(Path(str(p)).resolve()).lower())
                except Exception:
                    pass
        return out

    def _enqueue_framevision_minimax_job(self, job):
        try:
            from helpers.queue_adapter import enqueue_minimax_h3_job
        except Exception:
            try:
                import queue_adapter as _qa
                enqueue_minimax_h3_job = _qa.enqueue_minimax_h3_job
            except Exception as exc:
                raise RuntimeError(f"FrameVision queue adapter is unavailable: {exc}")
        qid = enqueue_minimax_h3_job(job)
        if not qid:
            raise RuntimeError("FrameVision queue did not accept the MiniMax job.")
        return qid

    def _build_music_clip_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            from minimax_music_clip import MiniMaxMusicClipWidget
            self.music_clip_widget = MiniMaxMusicClipWidget(page, queue_adapter=self._enqueue_music_clip_job)
            lay.addWidget(self.music_clip_widget, 1)
            self.music_clip_tab_index = self.tabs.addTab(page, "Music Clip Creator")
        except Exception as exc:
            self.music_clip_widget = None
            msg = QLabel(
                "Music Clip Creator could not be loaded.\n\n" + str(exc), page
            )
            msg.setWordWrap(True)
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(msg)
            lay.addStretch(1)
            self.music_clip_tab_index = self.tabs.addTab(page, "Music Clip Creator")

    def _sync_main_generate_button(self, index=None):
        is_music = (
            getattr(self, "music_clip_widget", None) is not None
            and getattr(self, "music_clip_tab_index", -1) == self.tabs.currentIndex()
        )
        if hasattr(self, "gen"):
            self.gen.setText("Create video clip" if is_music else "Generate")
            self.gen.setToolTip(
                "Analyze the selected song, use Whisper lyric timing when enabled, build the H3 shot list, queue every missing clip, then queue final trim/assembly."
                if is_music else "Add the current MiniMax generation job to the queue."
            )

    def _flash_generate_click(self):
        """Give immediate visual confirmation that the fixed Generate button was clicked."""
        if not hasattr(self, "gen"):
            return
        is_music = (
            getattr(self, "music_clip_widget", None) is not None
            and getattr(self, "music_clip_tab_index", -1) == self.tabs.currentIndex()
        )
        self.gen.setText("Create video clip ✓" if is_music else "Generate ✓")
        # Restore the normal context-sensitive label shortly after the click.
        QTimer.singleShot(700, self._sync_main_generate_button)

    def _main_generate_action(self):
        self._flash_generate_click()
        if (
            getattr(self, "music_clip_widget", None) is not None
            and getattr(self, "music_clip_tab_index", -1) == self.tabs.currentIndex()
        ):
            try:
                self.music_clip_widget.create_video_clip()
            except Exception as exc:
                QMessageBox.critical(self, "Create video clip failed", str(exc))
            return
        self.generate()

    def _enqueue_music_clip_job(self, spec):
        """Add one Music Clip Creator unit to the standalone MiniMax queue.

        One generated shot is one normal queue job.  Final assembly is also a queue
        job, so Generate all missing shots can safely enqueue a batch and the user can
        cancel/requeue individual clips from the existing Queue tab.
        """
        if not isinstance(spec, dict):
            raise ValueError("Invalid Music Clip queue job specification.")
        args = list(spec.get("args") or [])
        output = str(spec.get("output") or "").strip()
        if not args or not output:
            raise ValueError("Music Clip queue job is missing command arguments or output path.")
        label = str(spec.get("label") or "Music Clip job")
        frames = int(spec.get("frames") or 0)
        steps = int(spec.get("steps") or 0)
        seed = int(spec.get("seed") if spec.get("seed") is not None else -1)
        job = {
            "id": uuid.uuid4().hex,
            "job_number": self._take_next_job_number(),
            "state": "pending",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed": 0,
            "mode": 3,
            "mode_name": "Music Clip Creator",
            "model_label": "MiniMax H3 Ref2VA" if not spec.get("music_assembly") else "Music Clip assembly",
            "output": output,
            "seed": seed,
            "actual_seed": None,
            "resolution": str(spec.get("resolution") or "Music Clip"),
            "frames": frames,
            "steps": steps,
            "prompt": str(spec.get("prompt") or label),
            "args": args,
            "progress": None,
            "phase": "Waiting",
            "error": "",
            "cancel_reason": "",
            "settings": {},
            "log_tail": "",
            "continue_last_result": False,
            "continue_from_job_id": None,
            "continue_from_job_number": None,
            "manual_continue_video": "",
            "continue_context_frames": None,
            "glue_results": False,
            "continue_audio_memory": False,
            "music_clip_job": True,
            "music_clip_label": label,
        }
        for key in ("music_shot_index", "music_project_output", "music_assembly"):
            if key in spec:
                job[key] = spec[key]
        if self._framevision_queue_mode():
            qid = self._enqueue_framevision_minimax_job(job)
            self.status.setText(f"Added to FrameVision queue: {label}")
            return qid
        self.queue_jobs.append(job)
        self._save_queue_state()
        self._refresh_queue_views()
        self.status.setText(f"Added to queue: {label}")
        self._start_next_pending()
        return job.get("job_number")

    def _build_settings_tab(self):
        body = QWidget(); v = QVBoxLayout(body); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(10)

        self.system_hud_toggle = QCheckBox("System HUD")
        self.system_hud_toggle.setChecked(True)
        self.system_hud_toggle.setToolTip(
            "Show the compact live system HUD above the tabs. It displays GPU VRAM/load/temperature, "
            "DDR RAM, CPU load, network DL/UL only above 100 KB/s, and local date/time. Default: On."
        )
        self.system_hud_toggle.toggled.connect(self._set_system_hud_visible)
        if self._embedded:
            self.system_hud_toggle.setChecked(False)
            self.system_hud_toggle.setVisible(False)
        v.addWidget(self.system_hud_toggle)

        self.use_framevision_queue = QCheckBox("Use FrameVision queue")
        self.use_framevision_queue.setChecked(False)
        self.use_framevision_queue.setToolTip(
            "Send newly created MiniMax jobs to FrameVision's shared queue. The MiniMax internal queue is hidden and its pending jobs are paused while this is enabled."
        )
        self.use_framevision_queue.setVisible(self._embedded)
        self.use_framevision_queue.toggled.connect(self._set_framevision_queue_mode)
        v.addWidget(self.use_framevision_queue)

        self.sage_attention_enabled = QCheckBox("Enable SageAttention")
        self.sage_attention_enabled.setChecked(False)
        self.sage_attention_enabled.setToolTip("Transformer attention acceleration during sampling. Can be used together with Comfy Kitchen. Default: Off.")
        v.addWidget(self.sage_attention_enabled)

        self.comfy_kitchen_enabled = QCheckBox("Enable Comfy Kitchen W4A8 acceleration")
        self.comfy_kitchen_enabled.setChecked(True)
        self.comfy_kitchen_enabled.setToolTip("Uses Comfy Kitchen CUDA kernels for supported quantized W4A8 / ConvRot operations. This is separate from SageAttention and both may be enabled together. Default: On.")
        v.addWidget(self.comfy_kitchen_enabled)

        self.spectrum_enabled = QCheckBox("Enable Spectrum feature forecasting")
        self.spectrum_enabled.setChecked(False)
        self.spectrum_enabled.setToolTip("Uses the standalone MiniMax H3 Spectrum forecaster. It learns from real H3 target audio/video hidden states and skips selected later transformer evaluations. Conservative scheduling allows a 4-step Euler/Turbo run to forecast one middle step while keeping native refresh and final steps.")
        v.addWidget(self.spectrum_enabled)

        self.play_result_finished = QCheckBox("Play result when finished")
        self.play_result_finished.setChecked(False)
        v.addWidget(self.play_result_finished)

        self.play_result_queue_player = QCheckBox("Use player in Queue (Off = Windows default player)")
        self.play_result_queue_player.setChecked(False)
        self.play_result_queue_player.setVisible(False)
        self.play_result_finished.toggled.connect(self._sync_queue_preview_setting_visibility)
        v.addWidget(self.play_result_queue_player)

        self.auto_update_enabled = QCheckBox("Auto update app")
        self.auto_update_enabled.setChecked(True)
        self.auto_update_enabled.setToolTip(
            "When enabled, checks for application updates 10 seconds after startup."
        )
        v.addWidget(self.auto_update_enabled)

        fontg = QGroupBox("Interface font size")
        fontv = QVBoxLayout(fontg)
        fontrow = QHBoxLayout()
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal, fontg)
        self.font_size_slider.setRange(5, 15)
        self.font_size_slider.setValue(10)
        self.font_size_slider.setSingleStep(1)
        self.font_size_slider.setPageStep(1)
        self.font_size_label = QLabel("10 pt (default)", fontg)
        self.font_size_label.setMinimumWidth(105)
        self.font_size_slider.setToolTip("Changes the font size across the whole application. Range is 5 points smaller to 5 points larger than the 10 pt default. The GUI updates about two seconds after you stop moving the slider.")
        fontrow.addWidget(QLabel("Font size:", fontg))
        fontrow.addWidget(self.font_size_slider, 1)
        fontrow.addWidget(self.font_size_label)
        fontv.addLayout(fontrow)
        v.addWidget(fontg)
        self.font_size_slider.valueChanged.connect(self._font_size_slider_changed)

        outg = QGroupBox("Output"); of = QFormLayout(outg)
        self.output_folder = FolderRow(f"Blank = default: {DEFAULT_OUTPUT_DIR}")
        self.output_name = QLineEdit(); self.output_name.setPlaceholderText("Blank = automatic MiniMax H3 filename")
        of.addRow("Output folder", self.output_folder); of.addRow("Output name", self.output_name)
        v.addWidget(outg)

        models = QGroupBox("Model overrides"); mf = QFormLayout(models)
        mnote = QLabel("Leave a field empty for automatic model discovery. The app scans the matching MiniMax model folder and selects a compatible checkpoint; an override can be a .safetensors file or a folder to scan.")
        mnote.setWordWrap(True); mf.addRow(mnote)
        self.use_hybrid_model = QCheckBox("Use hybrid model")
        self.use_hybrid_model.setChecked(False)
        self.use_hybrid_model.setToolTip(
            "When enabled, the app ignores the separate FL2VA and Ref2VA checkpoint overrides and scans the MiniMax H3 model folders recursively for a .safetensors file with 'hybrid' in its filename. The same hybrid checkpoint is then used for T2VA/FL2VA and Ref2VA jobs."
        )
        mf.addRow(self.use_hybrid_model)
        self.fl2va_model = ModelPathRow("Blank = auto-scan diffusion_models for FL2VA")
        self.ref2va_model = ModelPathRow("Blank = auto-scan diffusion_models for Ref2VA")
        self.text_encoder_model = ModelPathRow("Blank = auto-scan text_encoders")
        self.video_vae_model = ModelPathRow("Blank = auto-scan models\\minimax_h3\\video_vae")
        self.audio_vae_model = ModelPathRow("Blank = auto-scan audio_vae")
        mf.addRow("FL2VA checkpoint", self.fl2va_model)
        mf.addRow("Ref2VA checkpoint", self.ref2va_model)
        mf.addRow("Text encoder", self.text_encoder_model)
        mf.addRow("Video VAE", self.video_vae_model)
        mf.addRow("Audio VAE", self.audio_vae_model)
        self.use_hybrid_model.toggled.connect(self._sync_hybrid_model_controls)
        self._sync_hybrid_model_controls(self.use_hybrid_model.isChecked())
        v.addWidget(models)

        vg = QGroupBox("VRAM Lab / Manager"); vf = QFormLayout(vg)
        vnote = QLabel("Automatic mode uses VRAM management when needed,leave all settings on default unless when testing or when offloading doesn't work correct with your gpu.")
        vnote.setWordWrap(True); vf.addRow(vnote)
        self.vram_manager_enabled = QCheckBox("Enable VRAM Manager protection")
        self.vram_manager_enabled.setChecked(True)
        self.vram_manager_enabled.setToolTip("Master switch for MiniMax-aware VRAM protection. With Automatic bypass enabled below, the manager is only injected when the current job is estimated not to fit the detected GPU's available dedicated VRAM.")
        vf.addRow(self.vram_manager_enabled)

        self.vram_manager_auto_bypass = QCheckBox("Automatic bypass when job fits")
        self.vram_manager_auto_bypass.setChecked(True)
        self.vram_manager_auto_bypass.setToolTip("Recommended. At the start of each job, detect GPU total/free dedicated VRAM and estimate native MiniMax H3 sampling demand from resolution and frame count. If the job fits with safety headroom, VRAM Lab is completely bypassed: no sampling hooks, residency manager, allocator guard, or manager-specific Comfy arguments. Uncheck to force VRAM Manager on for every job while the master switch is enabled.")
        vf.addRow(self.vram_manager_auto_bypass)

        self.vram_residency_engine = QComboBox()
        self.vram_residency_engine.addItem("Static partial (recommended)", "static")
        self.vram_residency_engine.addItem("DynamicVRAM / VBAR", "dynamic")
        self.vram_residency_engine.setCurrentIndex(0)
        self.vram_residency_engine.setToolTip("Static partial disables (Comfy) DynamicVRAM and uses the ModelPatcher partial-residency budget. Test both settings to see whatworks best for you")
        vf.addRow("Residency engine", self.vram_residency_engine)

        self.vram_runtime_free = QDoubleSpinBox(); self.vram_runtime_free.setRange(0.10, 8.0); self.vram_runtime_free.setDecimals(2); self.vram_runtime_free.setSingleStep(0.10); self.vram_runtime_free.setValue(0.50); self.vram_runtime_free.setSuffix(" GB")
        self.vram_runtime_free.setToolTip("Hard safety floor. If CUDA free memory drops below this, the manager asks Comfy to evict model weights. 0.50 GB is the aggressive 24 GB starting point.")
        vf.addRow("Runtime minimum free", self.vram_runtime_free)

        self.vram_text_headroom = QDoubleSpinBox(); self.vram_text_headroom.setRange(0.10, 16.0); self.vram_text_headroom.setDecimals(2); self.vram_text_headroom.setSingleStep(0.25); self.vram_text_headroom.setValue(1.0); self.vram_text_headroom.setSuffix(" GB")
        self.vram_text_headroom.setToolTip("Free VRAM target while Qwen/text-encoder weights are first admitted. Increase this first for 12-16 GB cards if prompt encoding spills.")
        vf.addRow("Text encoder load headroom", self.vram_text_headroom)

        self.vram_diffusion_headroom = QDoubleSpinBox(); self.vram_diffusion_headroom.setRange(0.10, 16.0); self.vram_diffusion_headroom.setDecimals(2); self.vram_diffusion_headroom.setSingleStep(0.25); self.vram_diffusion_headroom.setValue(1.0); self.vram_diffusion_headroom.setSuffix(" GB")
        self.vram_diffusion_headroom.setToolTip("Free VRAM target when MiniMax diffusion weights are first loaded. This room is for the first-step activations. Once sampling runs, Runtime minimum free becomes the floor.")
        vf.addRow("Diffusion load headroom", self.vram_diffusion_headroom)

        self.vram_offload_chunk = QSpinBox(); self.vram_offload_chunk.setRange(64, 4096); self.vram_offload_chunk.setSingleStep(64); self.vram_offload_chunk.setValue(512); self.vram_offload_chunk.setSuffix(" MB")
        self.vram_offload_chunk.setToolTip("Closest MiniMax equivalent to an offload block-size tuning knob. When headroom is breached, it will evict evict at least this much eligible model weight. Smaller = finer control; larger = fewer transfers but more VRAM released at once.")
        vf.addRow("Offload chunk", self.vram_offload_chunk)


        self.vram_residency_fill = QCheckBox("Fill unused dedicated VRAM with model weights")
        self.vram_residency_fill.setChecked(False)
        self.vram_residency_fill.setToolTip("Optional post-warm-up fill. With Static partial, Comfy already loads a larger initial resident weight set; this can use any remaining safe headroom. With DynamicVRAM this is retained mainly for comparison.")
        vf.addRow(self.vram_residency_fill)

        self.vram_residency_target_free = QDoubleSpinBox(); self.vram_residency_target_free.setRange(0.10, 8.0); self.vram_residency_target_free.setDecimals(2); self.vram_residency_target_free.setSingleStep(0.10); self.vram_residency_target_free.setValue(0.50); self.vram_residency_target_free.setSuffix(" GB")
        self.vram_residency_target_free.setToolTip("How much dedicated CUDA VRAM V2 tries to leave free while pulling offloaded diffusion weights back onto the GPU. Increase this if a resolution becomes unstable; lower it to search the 3090 limit.")
        vf.addRow("Residency target free", self.vram_residency_target_free)

        self.vram_residency_warmup = QSpinBox(); self.vram_residency_warmup.setRange(0, 50); self.vram_residency_warmup.setValue(2)
        self.vram_residency_warmup.setToolTip("Number of full DiT blocks allowed to run before V2 starts filling spare VRAM. This avoids filling the card before the first 1080p activation/workspace footprint is known.")
        vf.addRow("Residency warm-up blocks", self.vram_residency_warmup)

        self.vram_residency_refill_interval = QSpinBox(); self.vram_residency_refill_interval.setRange(1, 50); self.vram_residency_refill_interval.setValue(1)
        self.vram_residency_refill_interval.setToolTip("After warm-up, retry residency fill every N DiT blocks. 1 reacts fastest; larger values reduce residency adjustments.")
        vf.addRow("Residency refill every N blocks", self.vram_residency_refill_interval)

        self.vram_max_weights = QDoubleSpinBox(); self.vram_max_weights.setRange(0.0, 48.0); self.vram_max_weights.setDecimals(2); self.vram_max_weights.setSingleStep(0.25); self.vram_max_weights.setValue(0.0); self.vram_max_weights.setSuffix(" GB")
        self.vram_max_weights.setSpecialValueText("Auto")
        self.vram_max_weights.setToolTip("0/Auto = no fixed cap; headroom controls residency. Set a value to cap resident model-weight memory, useful when developing 12 GB profiles.")
        vf.addRow("Max resident model weights", self.vram_max_weights)

        self.vram_block_interval = QSpinBox(); self.vram_block_interval.setRange(1, 32); self.vram_block_interval.setValue(1)
        self.vram_block_interval.setToolTip("Check VRAM every N Qwen/DiT blocks. 1 reacts fastest. Larger values reduce tiny monitoring overhead but allow larger short-lived peaks.")
        vf.addRow("Check every N blocks", self.vram_block_interval)

        self.vram_async_streams = QSpinBox(); self.vram_async_streams.setRange(0, 8); self.vram_async_streams.setValue(2)
        self.vram_async_streams.setToolTip("Comfy async weight-offload streams. 2 is its normal NVIDIA default. 0 disables async offload; more is not always faster.")
        vf.addRow("Async offload streams", self.vram_async_streams)

        self.vram_video_vae_reserve = QDoubleSpinBox(); self.vram_video_vae_reserve.setRange(0.10, 16.0); self.vram_video_vae_reserve.setDecimals(2); self.vram_video_vae_reserve.setSingleStep(0.25); self.vram_video_vae_reserve.setValue(2.0); self.vram_video_vae_reserve.setSuffix(" GB")
        self.vram_video_vae_reserve.setToolTip("Reserve passed to the isolated video-VAE worker. Fresh-install default is 2 GB.")
        vf.addRow("Video VAE reserve", self.vram_video_vae_reserve)

        self.vram_video_vae_tile_size = QSpinBox(); self.vram_video_vae_tile_size.setRange(128, 1024); self.vram_video_vae_tile_size.setSingleStep(32); self.vram_video_vae_tile_size.setValue(256); self.vram_video_vae_tile_size.setSuffix(" px")
        self.vram_video_vae_tile_size.setToolTip("MiniMax video-VAE spatial tile size. Safe/current default is 256 px. Larger tiles may decode faster but use more VRAM. Test carefully for OOMs and visible tile seams.")
        vf.addRow("Video VAE tile size", self.vram_video_vae_tile_size)

        self.vram_video_vae_tile_overlap = QSpinBox(); self.vram_video_vae_tile_overlap.setRange(0, 512); self.vram_video_vae_tile_overlap.setSingleStep(32); self.vram_video_vae_tile_overlap.setValue(128); self.vram_video_vae_tile_overlap.setSuffix(" px")
        self.vram_video_vae_tile_overlap.setToolTip("Overlap between MiniMax video-VAE spatial tiles. Safe/current default is 128 px. 64 px is a speed test candidate, but lower overlap can make tile boundaries visible in the final MP4. Overlap must stay smaller than tile size.")
        vf.addRow("Video VAE tile overlap", self.vram_video_vae_tile_overlap)

        self.vram_audio_vae_reserve = QDoubleSpinBox(); self.vram_audio_vae_reserve.setRange(0.10, 16.0); self.vram_audio_vae_reserve.setDecimals(2); self.vram_audio_vae_reserve.setSingleStep(0.25); self.vram_audio_vae_reserve.setValue(1.0); self.vram_audio_vae_reserve.setSuffix(" GB")
        self.vram_audio_vae_reserve.setToolTip("Reserve passed to the isolated audio-VAE worker. Fresh-install default is 1 GB.")
        vf.addRow("Audio VAE reserve", self.vram_audio_vae_reserve)

        v.addWidget(vg)

        lg = QGroupBox("Logging"); lv = QVBoxLayout(lg)
        top = QHBoxLayout(); self.extended_logging = QCheckBox("Extended logging"); self.extended_logging.setChecked(False)
        self.extended_logging.setToolTip("Off = normal generation log (loads, settings, sampling/steps, decode, save). On = full diagnostic output including offload/VRAM details. Tile extraction is controlled separately.")
        self.tile_debugging = QCheckBox("Tile debugging"); self.tile_debugging.setChecked(False)
        self.tile_debugging.setToolTip("Off by default. When enabled, extracts a decoded frame plus tile plan/grid/overlay information for debugging video-VAE tile offloading. Leave this OFF unless testing tile offloading with the video VAE.")
        clear = QPushButton("Clear log"); clear.clicked.connect(lambda: self.log.clear())
        top.addWidget(self.extended_logging); top.addWidget(self.tile_debugging); top.addStretch(); top.addWidget(clear); lv.addLayout(top)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(300); lv.addWidget(self.log)
        v.addWidget(lg, 1)

        credits = QGroupBox("Credits / open-source components")
        cv = QVBoxLayout(credits)
        self.comfy_credit = QLabel(
            "PySide6 GUI and standalone installer by Contrinsan.\n\n"
            "This standalone uses components from ComfyUI (Comfy-Org), such as model-loading / "
            "MiniMax H3 nodes in comfy_extras and MiniMax VAE support to make this work. "
            "ComfyUI is licensed under GPL-3.0.\n\n"
            "INT4 model and text-encoder files used by this install are sourced from Winnougan / "
            "MiniMax-H3-INT4_Convrot_ComfyUI on Hugging Face.\n\n"
            "Spectrum Feature Forecasting is a standalone MiniMax H3 implementation based on the published Adaptive Spectral Feature Forecasting method by Han et al.\n\n"
            "The integrated Hailuo H3 Prompt Builder is an unofficial community tool created by Bob Doyle Media; "
            "its local server/UI was adapted here with standalone local-LLM and GGUF support."
        )
        self.comfy_credit.setWordWrap(True)
        crow = QHBoxLayout()
        self.comfy_repo = QPushButton("Open ComfyUI repository")
        self.comfy_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Comfy-Org/ComfyUI")))
        self.weights_repo = QPushButton("Open INT4 model repository")
        self.weights_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI")))
        self.prompt_builder_repo = QPushButton("Open Prompt Builder creator")
        self.prompt_builder_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://www.bobdoylemedia.com")))
        crow.addWidget(self.comfy_repo)
        crow.addWidget(self.weights_repo)
        crow.addWidget(self.prompt_builder_repo)
        crow.addStretch(1)
        cv.addWidget(self.comfy_credit)
        cv.addLayout(crow)
        v.addWidget(credits)
        v.addStretch(1)
        self.tabs.addTab(self._scroll_page(body), "Settings")

    def _set_system_hud_visible(self, enabled):
        if self._embedded:
            if hasattr(self, "system_hud"):
                self.system_hud.setVisible(False)
        elif hasattr(self, "system_hud") and isinstance(self.system_hud, SystemHud):
            enabled = bool(enabled)
            self.system_hud.setVisible(enabled)
            if enabled:
                self.system_hud.refresh()
                if not self.system_hud.timer.isActive(): self.system_hud.timer.start()
            else:
                self.system_hud.timer.stop()
        # Persist immediately as well as through the normal last-settings save.
        try:
            PRESET_DIR.mkdir(parents=True, exist_ok=True)
            p = PRESET_DIR / "minimax_h3_gui_last.json"
            d = {}
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
            d["system_hud"] = bool(enabled)
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _add_tooltips(self):
        """Short, useful help text. Defaults are stated where they matter."""
        self.mode.setToolTip(
            "Generation pipeline. T2VA = prompt only; FL2VA = first/last image conditioning; "
            "Ref2VA = reference images/video/audio. Default: Text to video (T2VA)."
        )
        self.aspect.setToolTip(
            "Output orientation. Changing this only swaps/selects the matching fixed MiniMax resolution; "
            "it does not change sampler settings. Default: 16:9."
        )
        self.res_class.setToolTip(
            "Fixed MiniMax H3 resolution preset. No free-form width/height values are used. "
            "The 1280 × 720 display preset generates at MiniMax-valid 1280 × 704 (704p). "
            "Default: 832 × 480 for 16:9. Higher resolutions need substantially more VRAM/RAM and time."
        )
        self.resolved.setToolTip("Exact width × height that will be sent to the backend for the selected aspect ratio.")
        self.frames.setToolTip(
            "Fixed MiniMax H3 frame count at 24 FPS. Normal mode stops at 719 frames = 29.96 seconds. "
            "Enable Experimental long duration for values above 30 seconds, up to 2385 frames = 99.38 seconds. "
            "The maximum frame count that will actually run depends on the available system RAM and GPU VRAM. Default: 362 frames."
        )
        self.experimental_long_duration.setToolTip(
            "Unlock experimental H3 durations above the normal 30-second range. Values follow the native 17k+5 frame grid "
            "up to 2385 frames = 99.38 seconds. The maximum duration that will actually run depends on available system RAM "
            "and GPU VRAM; this switch exposes research values and does not guarantee that every resolution or hardware setup can reach 100 seconds."
        )
        self.steps.setToolTip("Number of diffusion/sampling steps. More steps take longer. Default: 15.")
        self.seed.setToolTip("Random seed. Use -1 for a new random seed each generation. Default: -1 (random).")
        self.prompt.setToolTip(
            "Full video instruction sent to MiniMax H3. You can include action, camera direction, dialogue, "
            "sound effects and music instructions here."
        )
        self.first.setToolTip("Optional FL2VA first-frame image. Do not combine with Continue video, because the source video's final frame becomes the boundary anchor automatically.")
        self.last.setToolTip("Optional FL2VA last-frame/end image. It can also be used as a destination for a continued clip.")
        self.continue_video.setToolTip("Native H3 FL2VA continuation. The model receives a VAE-encoded block of preceding motion plus the source video's final frame as the exact boundary anchor; this is not last-frame-only I2V.")
        self.glue_results.setToolTip("When enabled, keep the complete source video first and append the newly generated continuation after it. No continuation overlap frames are trimmed from either clip during the glue step.")
        self.continue_last_result.setToolTip("Ignore the manual Continue video field and use the exact previous queue job as this job's continuation source. Pending chained jobs wait for that specific job to finish; the output folder is never scanned for the newest file.")
        self.continue_context.setToolTip("How many source-motion history frames H3 receives before the separate final-frame boundary anchor. Values use H3's native 17k+5 temporal grid so source motion and generated motion stay on the same 24 FPS model clock. 39 history frames (~1.63 s) is the default.")
        self.ref_size.setToolTip(
            "How Ref2VA prepares reference images. 'match' follows the generation/reference sizing behavior; "
            "'max' uses the maximum reference sizing path. Default: match."
        )
        self.ref_images.setToolTip("Ref2VA reference images. MiniMax H3 supports up to 9 images.")
        self.ref_videos.setToolTip("Ref2VA reference videos. MiniMax H3 supports up to 3 videos.")
        self.ref_audios.setToolTip("Ref2VA standalone audio references. MiniMax H3 supports up to 3 audio files.")
        self.cfg.setToolTip("Classifier-free guidance strength used by the sampler. Default: 1.0.")
        self.shift.setToolTip("Video timestep/sigma shift. Validated starting value for this install: 12.")
        self.audio_shift.setToolTip("Audio timestep/sigma shift. Validated starting value for this install: 3.")
        self.sampler.setToolTip("Diffusion sampler algorithm. Default: Euler. Change only when intentionally testing sampler behavior.")
        self.scheduler.setToolTip("Sigma/timestep schedule used by the sampler. Default: simple.")
        self.preset_name.setToolTip("Name used when saving the current GUI configuration as a JSON preset.")

        self.output_folder.setToolTip(f"Folder for generated MP4 files. Leave empty to use: {DEFAULT_OUTPUT_DIR}")
        self.output_name.setToolTip("Optional MP4 filename. Leave empty to create a timestamped filename automatically.")
        self.fl2va_model.setToolTip("Optional FL2VA INT4 checkpoint override. Select a .safetensors file or its folder. Leave empty for the install default.")
        self.ref2va_model.setToolTip("Optional Ref2VA INT4 checkpoint override. Select a .safetensors file or its folder. Leave empty for the install default.")
        self.text_encoder_model.setToolTip("Optional Qwen3-VL text-encoder override. Select a .safetensors file or its folder. Leave empty for the install default.")
        self.video_vae_model.setToolTip("Visual/video VAE home is models\\minimax_h3\\video_vae. Leave blank to scan that folder recursively for a compatible single-file video VAE; or select a specific file/folder.")
        self.audio_vae_model.setToolTip("Leave blank to scan models\\minimax_h3\\audio_vae recursively for a compatible audio VAE; or select a specific file/folder.")
        self.extended_logging.setToolTip(
            "Off (default): normal useful generation logging (model loads, settings, sampling/step progress, decode and save). On: full backend trace including offload/VRAM details. "
            "Tile extraction is controlled separately by Tile debugging."
        )
        self.tile_debugging.setToolTip(
            "Off by default. When enabled, extracts a decoded frame and tile plan/grid/overlay information for debugging video-VAE tile offloading. "
            "Leave this OFF unless testing tile offloading with the video VAE."
        )
        self.log.setToolTip("Generation log. Normal mode shows model loading, settings, sampler progress, decoding and saved output. Enable Extended logging only for low-level diagnostics.")
        self.comfy_credit.setToolTip("Credits the ComfyUI code used by this standalone backend. ComfyUI is an open-source project maintained by Comfy-Org.")
        self.comfy_repo.setToolTip("Open the official ComfyUI GitHub repository in your default web browser.")
        self.weights_repo.setToolTip("Open the Hugging Face repository that supplied the INT4 MiniMax H3 model and text-encoder files used by this install.")
        self.prompt_builder_repo.setToolTip("Open Bob Doyle Media, creator of the integrated Hailuo H3 Prompt Builder community tool.")
        self.builder_start_btn.setToolTip("Start or restart the local Hailuo H3 Prompt Builder server bundled with this standalone app.")
        self.builder_browser_btn.setToolTip("Open the same local Prompt Builder in your default web browser.")
        self.builder_transfer_btn.setToolTip("Copy the finished H3 prompt into the Generation tab. Supported aspect ratio and nearest approved frame preset are transferred too.")
        self.gen.setToolTip("Add the current generation settings as a queue job. If nothing is running it starts immediately; otherwise it waits in Pending. This button remains available on every tab.")
        self.cancel.setToolTip("Cancel the currently running queue job completely. Use the Queue tab context menu to cancel and move it back to Pending instead.")
        self.openout.setToolTip("Open the configured output folder in Windows Explorer.")

    def _update_font_size_label(self, value):
        if not hasattr(self, "font_size_label"):
            return
        value = int(value)
        suffix = " (default)" if value == 10 else (f" (+{value - 10})" if value > 10 else f" ({value - 10})")
        self.font_size_label.setText(f"{value} pt{suffix}")

    def _font_size_slider_changed(self, value):
        self._update_font_size_label(value)
        self.status.setText(f"Font size: {int(value)} pt — applying when slider movement stops…")
        self._font_apply_timer.start()

    def _apply_pending_font_size(self):
        if not hasattr(self, "font_size_slider"):
            return
        self._font_size_pt = int(self.font_size_slider.value())
        self._apply_style()
        self._schedule_layout_refresh()
        try:
            self.save_last()
        except Exception:
            pass
        self.status.setText(f"Font size applied globally: {self._font_size_pt} pt")

    def _apply_style(self):
        base = max(5, min(15, int(getattr(self, "_font_size_pt", 10))))
        title = base + 7
        hud = base + 5
        style = """
        QMainWindow,QWidget { background:#11151b; color:#e8eef6; font-size:__BASE__pt; }
        QTabWidget::pane { border:1px solid #263545; border-radius:7px; }
        QTabBar::tab { background:#171f28; border:1px solid #2e4051; padding:8px 18px; min-width:110px; }
        QTabBar::tab:selected { background:#1d3040; color:#82e7ff; }
        QGroupBox { border:1px solid #2a3948; border-radius:8px; margin-top:10px; padding:10px; font-weight:600; }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#75d8ff; }
        QLineEdit,QPlainTextEdit,QComboBox,QSpinBox,QDoubleSpinBox,QListWidget,QTreeWidget,QTableWidget { background:#0b0f14; color:#e8eef6; border:1px solid #334556; border-radius:5px; padding:5px; selection-background-color:#176c89; selection-color:#ffffff; }
        QScrollArea, QAbstractScrollArea, QAbstractScrollArea::viewport { background:#11151b; color:#e8eef6; border-color:#263545; }
        QHeaderView::section { background:#18232e; color:#dce8f3; border:0; border-right:1px solid #334556; border-bottom:1px solid #334556; padding:5px; }
        QTableCornerButton::section { background:#18232e; border:1px solid #334556; }
        QProgressBar { background:#0b0f14; color:#e8eef6; border:1px solid #334556; border-radius:4px; text-align:center; min-height:18px; }
        QProgressBar::chunk { background:#126680; border-radius:3px; }
        QComboBox QAbstractItemView { background:#0b0f14; color:#e8eef6; selection-background-color:#176c89; selection-color:#ffffff; outline:0; }
        QToolTip { background:#18232e; color:#e8eef6; border:1px solid #3c5267; padding:4px; }
        QPushButton { background:#1b2732; border:1px solid #3c5267; border-radius:5px; padding:6px 11px; }
        QPushButton:hover { background:#243544; } QPushButton:disabled { color:#66717a; background:#15191e; }
        QPushButton#primary { background:#126680; border-color:#28b6df; font-weight:700; padding:9px 28px; }
        QWidget#bottomBar { border:1px solid #2e4051; border-radius:7px; background:#0d1319; }
        QLabel#title { font-size:__TITLE__pt; font-weight:700; color:#7de3ff; } QLabel#status { color:#a8bac9; }
        QLabel#systemHud { background:#0d1319; border:1px solid #263545; border-radius:7px; padding:6px 10px; font-size:__HUD__pt; }
        QLabel#queueSummary { color:#82e7ff; font-weight:600; padding:3px 6px; }
        QLabel#imageThumb { background:#0b0f14; border:1px solid #334556; border-radius:8px; color:#9ab8d8; padding:4px; }
        QScrollBar:vertical { background:#0c1116; width:14px; margin:0; } QScrollBar::handle:vertical { background:#395166; min-height:30px; border-radius:6px; }
        """
        style = style.replace("__BASE__", str(base)).replace("__TITLE__", str(title)).replace("__HUD__", str(hud))
        self.setStyleSheet(style)

    def _sync_resolution(self):
        label = self.res_class.currentText()
        w, h = RESOLUTION_PRESETS[label][self.aspect.currentText()]
        if label == "1280 × 720":
            self.resolved.setText(f"Generation: {w} × {h} (704p)")
        else:
            self.resolved.setText(f"{w} × {h}")

    def _sync_mode(self):
        i = self.mode.currentIndex(); self.fl_group.setVisible(i == 1); self.ref_group.setVisible(i == 2)
        self._sync_continue_video_options()

    def _sync_continue_video_options(self):
        chain = bool(getattr(self, "continue_last_result", None) and self.continue_last_result.isChecked())
        if hasattr(self, "continue_video"):
            self.continue_video.setEnabled(not chain)
        # A queued-result continuation supplies its own first-frame boundary just like
        # a manually selected Continue Video source. Keep Last frame available as a destination.
        if hasattr(self, "first"):
            self.first.setEnabled(not chain)
        if hasattr(self, "continue_audio_memory_row"):
            self.continue_audio_memory_row.setVisible(chain)
        if hasattr(self, "continue_audio_memory"):
            self.continue_audio_memory.setEnabled(chain)

    def current_output_dir(self) -> Path:
        p = self.output_folder.path()
        return Path(p).expanduser() if p else DEFAULT_OUTPUT_DIR

    def open_output_folder(self):
        p = self.current_output_dir(); p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def make_output_path(self, mode: int) -> Path:
        folder = self.current_output_dir(); folder.mkdir(parents=True, exist_ok=True)
        name = self.output_name.text().strip()
        if not name:
            prefix = "minimax_h3_ref2va_int4" if mode == 2 else "minimax_h3_int4"
            name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        if not name.lower().endswith(".mp4"): name += ".mp4"
        return folder / Path(name).name

    def _start_job_log_file(self, job):
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp=time.strftime('%Y%m%d_%H%M%S')
            path=LOG_DIR / f"minimax_h3_{stamp}_{str(job.get('id',''))[:8]}.log"
            job['log_file']=str(path)
            self._active_log_file=path
            path.write_text(f"=== QUEUE START ===\nOutput: {job.get('output')}\n", encoding='utf-8')
            return path
        except Exception as exc:
            self._active_log_file=None
            self.append_log(f"Log file warning: {exc}\n")
            return None

    def _write_job_log_file(self, text):
        path=getattr(self, '_active_log_file', None)
        if not path or not text:
            return
        try:
            with Path(path).open('a', encoding='utf-8', errors='replace') as f:
                f.write(text)
        except Exception:
            pass

    def append_log(self, text):
        self.log.moveCursor(QTextCursor.MoveOperation.End); self.log.insertPlainText(text); self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _process_output(self):
        if not self.proc: return
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        if not text: return
        self._write_job_log_file(text)
        job = self._job_by_id(self.current_job_id) if self.current_job_id else None
        if job:
            job["log_tail"] = (job.get("log_tail", "") + text)[-24000:]
            for raw in text.replace("\r", "\n").splitlines():
                line = raw.strip()
                if not line: continue
                m = re.search(r"(?:^|\s)(\d{1,3})%\|", line)
                if m:
                    job["progress"] = max(0,min(100,int(m.group(1)))); job["phase"] = "Sampling"
                    sm = re.search(r"(?:^|\s)(\d+)\s*/\s*(\d+)(?:\s|\[|$)", line)
                    if sm:
                        job["step_now"] = int(sm.group(1)); job["step_total"] = int(sm.group(2))
                elif "Sampling " in line and "steps" in line:
                    job["progress"] = 0; job["phase"] = "Sampling"
                elif "seed=" in line:
                    sm = re.search(r"seed=(\d+)", line)
                    if sm: job["actual_seed"] = int(sm.group(1))
                if line.startswith("Using checkpoint:"):
                    job["model_label"] = line.split(":",1)[1].strip()
                elif any(k in line for k in ("Loading ","Encoding ","Starting VAE","Decoding video","Decoding audio","Muxing ","Sampling process exited")):
                    job["progress"] = None
                    job["phase"] = line[:80]
                    job["step_now"] = None
                    job["step_total"] = None
            self._save_queue_state()
            if hasattr(self, "system_hud"):
                self.system_hud.refresh()
        if self.extended_logging.isChecked():
            self.append_log(text)
            return
        # Normal mode is intentionally useful, not silent.  Keep user-facing
        # milestones and sampler progress while hiding low-level model-management,
        # offload/VRAM chatter, tensor diagnostics and per-block VAE traces.
        keep = []
        tokens = (
            "MiniMax-H3", "Generation settings", "Using checkpoint", "Using text encoder",
            "Using video VAE", "Using audio VAE", "Loading ", "Encoding ", "Sampling ",
            "Sampling stage complete", "Ref2VA sampling stage complete", "Sampling process",
            "Starting VAE", "Decoding video", "Video decode stage complete",
            "Decoding audio", "Audio decode stage complete", "Muxing ",
            "Saved ", "Glue results", "video-only fallback", "LoRA ", "VRAM Manager", "[VRAM-MGR]", "ERROR", "FAILED", "WARNING",
            "Traceback", "Exception", "VALIDATION",
        )
        for raw in text.replace("\r", "\n").splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            # tqdm/Comfy sampler progress, e.g. 40%|...| 6/15 [..]
            progress = ("%|" in line and ("/" in line or "it/s" in low or "s/it" in low))
            if progress or any(t.lower() in low for t in tokens):
                keep.append(line)
        if keep: self.append_log("\n".join(keep) + "\n")

    def settings_dict(self):
        return {
            "mode": self.mode.currentIndex(), "aspect": self.aspect.currentText(), "resolution": self.res_class.currentText(), "lanczos_scale_2x": self.lanczos_scale_2x.isChecked(), "frames": self._frame_count(), "experimental_long_duration": self.experimental_long_duration.isChecked(),
            "steps": self.steps.value(), "seed": self.seed.value(), "prompt": self.prompt.toPlainText(), "first": self.first.path(), "last": self.last.path(),
            "continue_video": self.continue_video.path(), "continue_context_frames": int(self.continue_context.currentData() or 39),
            "glue_results": self.glue_results.isChecked(), "continue_last_result": self.continue_last_result.isChecked(),
            "continue_audio_memory": self.continue_audio_memory.isChecked(),
            "ref_size": self.ref_size.currentText(), "ref_images": self.ref_images.paths(), "ref_videos": self.ref_videos.paths(), "ref_audios": self.ref_audios.paths(),
            "cfg": self.cfg.value(), "shift": self.shift.value(), "audio_shift": self.audio_shift.value(), "sampler": self.sampler.currentText(), "scheduler": self.scheduler.currentText(),
            "output_folder": self.output_folder.path(), "output_name": self.output_name.text().strip(), "extended_logging": self.extended_logging.isChecked(), "tile_debugging": self.tile_debugging.isChecked(),
            "system_hud": False if self._embedded else self.system_hud_toggle.isChecked(),
            "use_framevision_queue": self._framevision_queue_mode(),
            "auto_update_enabled": self.auto_update_enabled.isChecked(),
            "font_size_pt": int(self.font_size_slider.value()) if hasattr(self, "font_size_slider") else int(self._font_size_pt),
            "play_result_finished": self.play_result_finished.isChecked(),
            "play_result_queue_player": self.play_result_queue_player.isChecked(),
            "spectrum_enabled": self.spectrum_enabled.isChecked(),
            "sage_attention_enabled": self.sage_attention_enabled.isChecked(),
            "comfy_kitchen_enabled": self.comfy_kitchen_enabled.isChecked(),
            "use_hybrid_model": self.use_hybrid_model.isChecked(),
            "vram_manager_enabled": self.vram_manager_enabled.isChecked(), "vram_manager_auto_bypass": self.vram_manager_auto_bypass.isChecked(), "vram_residency_engine": self.vram_residency_engine.currentData(), "vram_runtime_free_gb": self.vram_runtime_free.value(),
            "vram_text_headroom_gb": self.vram_text_headroom.value(), "vram_diffusion_headroom_gb": self.vram_diffusion_headroom.value(),
            "vram_offload_chunk_mb": self.vram_offload_chunk.value(), "vram_max_resident_weights_gb": self.vram_max_weights.value(),
            "vram_residency_fill": self.vram_residency_fill.isChecked(), "vram_residency_target_free_gb": self.vram_residency_target_free.value(),
            "vram_residency_warmup_blocks": self.vram_residency_warmup.value(), "vram_residency_refill_interval": self.vram_residency_refill_interval.value(),
            "vram_block_check_interval": self.vram_block_interval.value(), "vram_async_streams": self.vram_async_streams.value(),
            "vram_video_vae_reserve_gb": self.vram_video_vae_reserve.value(), "vram_audio_vae_reserve_gb": self.vram_audio_vae_reserve.value(),
            "vram_video_vae_tile_size": self.vram_video_vae_tile_size.value(), "vram_video_vae_tile_overlap": self.vram_video_vae_tile_overlap.value(),
            "fl2va_model": self.fl2va_model.path(), "ref2va_model": self.ref2va_model.path(), "text_encoder_model": self.text_encoder_model.path(),
            "video_vae_model": self.video_vae_model.path(), "audio_vae_model": self.audio_vae_model.path(),
            "loras": [{"path": row.path(), "strength": strength.value()} for row, strength in self.lora_rows],
        }

    def apply_settings(self, d):
        try:
            self.mode.setCurrentIndex(int(d.get("mode", 0))); self.aspect.setCurrentText(d.get("aspect", "16:9"))
            saved_res = str(d.get("resolution", DEFAULT_RESOLUTION))
            saved_res = {"Low / test": "576 × 320", "480p": "832 × 480", "768p": "1344 × 768", "1080p": "1920 × 1088"}.get(saved_res, saved_res)
            if saved_res in RESOLUTION_PRESETS: self.res_class.setCurrentText(saved_res)
            else: self.res_class.setCurrentText(DEFAULT_RESOLUTION)
            self.lanczos_scale_2x.setChecked(bool(d.get("lanczos_scale_2x", False)))
            self.experimental_long_duration.setChecked(bool(d.get("experimental_long_duration", False)))
            self._sync_long_duration_mode(self.experimental_long_duration.isChecked())
            self._set_frame_count(d.get("frames", 362))
            self.steps.setValue(int(d.get("steps", 15))); self.seed.setValue(int(d.get("seed", -1))); self.prompt.setPlainText(d.get("prompt", "")); self.first.edit.setText(d.get("first", "")); self.last.edit.setText(d.get("last", "")); self.continue_video.edit.setText(d.get("continue_video", ""))
            ctx=int(d.get("continue_context_frames",39)); idx=self.continue_context.findData(ctx); self.continue_context.setCurrentIndex(idx if idx >= 0 else 1)
            self.glue_results.setChecked(bool(d.get("glue_results", False))); self.continue_last_result.setChecked(bool(d.get("continue_last_result", False))); self.continue_audio_memory.setChecked(bool(d.get("continue_audio_memory", False))); self._sync_continue_video_options()
            self.ref_size.setCurrentText(d.get("ref_size", "match")); self.ref_images.set_paths(d.get("ref_images", [])); self.ref_videos.set_paths(d.get("ref_videos", [])); self.ref_audios.set_paths(d.get("ref_audios", []))
            self.cfg.setValue(float(d.get("cfg", 1.0))); self.shift.setValue(float(d.get("shift", 12))); self.audio_shift.setValue(float(d.get("audio_shift", 3))); self.sampler.setCurrentText(d.get("sampler", "euler")); self.scheduler.setCurrentText(d.get("scheduler", "simple"))
            # Backward compatibility with the first GUI patch's single output field.
            old_output = d.get("output", "")
            if old_output and not d.get("output_folder") and not d.get("output_name"):
                op = Path(old_output); self.output_folder.edit.setText(str(op.parent)); self.output_name.setText(op.name)
            else:
                self.output_folder.edit.setText(d.get("output_folder", "")); self.output_name.setText(d.get("output_name", ""))
            self.extended_logging.setChecked(bool(d.get("extended_logging", False)))
            self.tile_debugging.setChecked(bool(d.get("tile_debugging", False)))
            self.system_hud_toggle.setChecked(False if self._embedded else bool(d.get("system_hud", True)))
            if hasattr(self, "use_framevision_queue"):
                self.use_framevision_queue.setChecked(bool(d.get("use_framevision_queue", False)) if self._embedded else False)
            self.auto_update_enabled.setChecked(bool(d.get("auto_update_enabled", True)))
            saved_font = max(5, min(15, int(d.get("font_size_pt", 10))))
            self.font_size_slider.blockSignals(True)
            self.font_size_slider.setValue(saved_font)
            self.font_size_slider.blockSignals(False)
            self._font_size_pt = saved_font
            self._update_font_size_label(saved_font)
            self._apply_style()
            self.play_result_finished.setChecked(bool(d.get("play_result_finished", False)))
            self.play_result_queue_player.setChecked(False if self._framevision_queue_mode() else bool(d.get("play_result_queue_player", False)))
            self._sync_queue_preview_setting_visibility()
            self.spectrum_enabled.setChecked(bool(d.get("spectrum_enabled", False)))
            self.sage_attention_enabled.setChecked(bool(d.get("sage_attention_enabled", False)))
            self.comfy_kitchen_enabled.setChecked(bool(d.get("comfy_kitchen_enabled", True)))
            self.use_hybrid_model.setChecked(bool(d.get("use_hybrid_model", False)))
            self._sync_hybrid_model_controls(self.use_hybrid_model.isChecked())
            self._set_system_hud_visible(False if self._embedded else self.system_hud_toggle.isChecked())
            self._set_framevision_queue_mode(self._framevision_queue_mode())
            self.vram_manager_enabled.setChecked(bool(d.get("vram_manager_enabled", True)))
            self.vram_manager_auto_bypass.setChecked(bool(d.get("vram_manager_auto_bypass", True)))
            engine = str(d.get("vram_residency_engine", "static")).lower()
            idx = self.vram_residency_engine.findData(engine)
            self.vram_residency_engine.setCurrentIndex(idx if idx >= 0 else 0)
            self.vram_runtime_free.setValue(float(d.get("vram_runtime_free_gb", 0.50)))
            self.vram_text_headroom.setValue(float(d.get("vram_text_headroom_gb", 1.0)))
            self.vram_diffusion_headroom.setValue(float(d.get("vram_diffusion_headroom_gb", 1.0)))
            self.vram_offload_chunk.setValue(int(d.get("vram_offload_chunk_mb", 512)))
            self.vram_max_weights.setValue(float(d.get("vram_max_resident_weights_gb", 0.0)))
            self.vram_residency_fill.setChecked(bool(d.get("vram_residency_fill", False)))
            self.vram_residency_target_free.setValue(float(d.get("vram_residency_target_free_gb", 0.50)))
            self.vram_residency_warmup.setValue(int(d.get("vram_residency_warmup_blocks", 2)))
            self.vram_residency_refill_interval.setValue(int(d.get("vram_residency_refill_interval", 1)))
            self.vram_block_interval.setValue(int(d.get("vram_block_check_interval", 1)))
            self.vram_async_streams.setValue(int(d.get("vram_async_streams", 2)))
            self.vram_video_vae_reserve.setValue(float(d.get("vram_video_vae_reserve_gb", 2.0)))
            self.vram_audio_vae_reserve.setValue(float(d.get("vram_audio_vae_reserve_gb", 1.0)))
            self.vram_video_vae_tile_size.setValue(int(d.get("vram_video_vae_tile_size", 256)))
            self.vram_video_vae_tile_overlap.setValue(int(d.get("vram_video_vae_tile_overlap", 128)))
            self.fl2va_model.edit.setText(d.get("fl2va_model", "")); self.ref2va_model.edit.setText(d.get("ref2va_model", "")); self.text_encoder_model.edit.setText(d.get("text_encoder_model", "")); self.video_vae_model.edit.setText(d.get("video_vae_model", "")); self.audio_vae_model.edit.setText(d.get("audio_vae_model", ""))
            saved_loras = d.get("loras", []) or []
            for i, (row, strength) in enumerate(self.lora_rows):
                item = saved_loras[i] if i < len(saved_loras) and isinstance(saved_loras[i], dict) else {}
                row.edit.setText(str(item.get("path", ""))); strength.setValue(float(item.get("strength", 1.0)))
        except Exception as e:
            self.append_log(f"Preset warning: {e}\n")
        self._sync_resolution(); self._sync_mode()

    def save_last(self):
        PRESET_DIR.mkdir(parents=True, exist_ok=True); (PRESET_DIR / "minimax_h3_gui_last.json").write_text(json.dumps(self.settings_dict(), indent=2), encoding="utf-8")
    def load_last(self):
        p = PRESET_DIR / "minimax_h3_gui_last.json"
        if p.is_file():
            try: self.apply_settings(json.loads(p.read_text(encoding="utf-8")))
            except Exception: pass
    def save_named(self):
        name = self.preset_name.text().strip()
        if not name: QMessageBox.information(self, "Preset", "Enter a preset name first."); return
        safe = ''.join(c if c.isalnum() or c in '-_ .' else '_' for c in name).strip().replace(' ', '_')
        PRESET_DIR.mkdir(parents=True, exist_ok=True); p = PRESET_DIR / f"minimax_h3_{safe}.json"; p.write_text(json.dumps(self.settings_dict(), indent=2), encoding="utf-8"); self.status.setText(f"Saved {p.name}")
    def load_named(self):
        PRESET_DIR.mkdir(parents=True, exist_ok=True); p, _ = QFileDialog.getOpenFileName(self, "Load preset", str(PRESET_DIR), "JSON (*.json)")
        if p: self.apply_settings(json.loads(Path(p).read_text(encoding="utf-8")))
    def safe_preset(self):
        self.mode.setCurrentIndex(0); self.res_class.setCurrentText("576 × 320"); self.aspect.setCurrentText("16:9"); self._set_frame_count(124); self.steps.setValue(10); self.cfg.setValue(1.0); self.shift.setValue(12); self.audio_shift.setValue(3); self.sampler.setCurrentText("euler"); self.scheduler.setCurrentText("simple")

    def lora_args(self):
        args = []
        for row, strength in self.lora_rows:
            path = row.path()
            value = float(strength.value())
            if path and value != 0.0:
                args += ["--lora", path, "--lora-strength", str(value)]
        return args

    def _sync_hybrid_model_controls(self, enabled):
        # Keep the saved FL2VA/Ref2VA locations visible, but make it clear they are ignored
        # while the single hybrid diffusion checkpoint is active.
        self.fl2va_model.setEnabled(not bool(enabled))
        self.ref2va_model.setEnabled(not bool(enabled))

    def _find_hybrid_checkpoint(self):
        """Return the preferred MiniMax H3 hybrid .safetensors checkpoint, or an empty string."""
        roots = []

        # Custom diffusion locations already entered by the user are valid search roots.
        for raw in (self.fl2va_model.path(), self.ref2va_model.path()):
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if candidate.is_file():
                if candidate.suffix.lower() == ".safetensors" and "hybrid" in candidate.name.lower():
                    return str(candidate.resolve())
                candidate = candidate.parent
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)

        default_root = ROOT / "models" / "minimax_h3"
        diffusion_root = default_root / "diffusion_models"
        for candidate in (diffusion_root, default_root):
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)

        matches = []
        seen = set()
        for root in roots:
            try:
                for path in root.rglob("*.safetensors"):
                    if "hybrid" not in path.name.lower():
                        continue
                    key = str(path.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        matches.append(path)
            except (OSError, PermissionError):
                continue

        if not matches:
            return ""

        # Prefer diffusion_models and shallow paths, then use alphabetical order so selection
        # stays deterministic if the user has more than one hybrid checkpoint installed.
        def sort_key(path):
            try:
                rel = path.resolve().relative_to(diffusion_root.resolve())
                return (0, len(rel.parts), path.name.lower(), str(path).lower())
            except Exception:
                try:
                    rel = path.resolve().relative_to(default_root.resolve())
                    return (1, len(rel.parts), path.name.lower(), str(path).lower())
                except Exception:
                    return (2, len(path.parts), path.name.lower(), str(path).lower())

        matches.sort(key=sort_key)
        return str(matches[0].resolve())

    def model_override_args(self):
        hybrid = self._find_hybrid_checkpoint() if self.use_hybrid_model.isChecked() else ""
        pairs = [
            ("--fl2va-checkpoint", hybrid if hybrid else self.fl2va_model.path()),
            ("--ref2va-checkpoint", hybrid if hybrid else self.ref2va_model.path()),
            ("--text-encoder", self.text_encoder_model.path()), ("--video-vae", self.video_vae_model.path()), ("--audio-vae", self.audio_vae_model.path()),
        ]
        args = []
        for flag, value in pairs:
            if value: args += [flag, value]
        return args

    def _minimax_bootstrap_ready(self):
        # Require the repo runtime/vendor plus the MiniMax model folder.  This
        # covers both fresh installs and damaged/partially deleted installations.
        return (
            (ROOT / "runtime").is_dir()
            and (ROOT / "vendor").is_dir()
            and (ROOT / "models" / "minimax_h3").is_dir()
        )

    def _refresh_bootstrap_button(self):
        if not hasattr(self, "install_minimax"):
            return
        if self._bootstrap_running:
            self.install_minimax.setVisible(True)
            self.install_minimax.setEnabled(False)
            self.install_minimax.setText("Please wait while installing")
            self.install_minimax_progress.setVisible(True)
            return
        ready = self._minimax_bootstrap_ready()
        self.install_minimax.setVisible(not ready)
        self.install_minimax.setEnabled(not ready)
        self.install_minimax.setText("Install MiniMax H3 model & repo")
        self.install_minimax_progress.setVisible(False)

    def _set_bootstrap_stage(self, text):
        self._bootstrap_stage = str(text or "")
        if self._bootstrap_stage:
            self.status.setText(self._bootstrap_stage)

    def _start_minimax_bootstrap(self):
        if self._bootstrap_running:
            return
        installer = ROOT / "presets" / "extra_env" / "install.bat"
        if not installer.is_file():
            QMessageBox.critical(
                self,
                "MiniMax H3 installer",
                f"The FrameVision MiniMax installer was not found:\n\n{installer}",
            )
            return
        self._bootstrap_running = True
        self._refresh_bootstrap_button()
        self._set_bootstrap_stage("Downloading MiniMax H3 repository…")
        threading.Thread(target=self._bootstrap_repo_worker, daemon=True).start()

    @staticmethod
    def _bootstrap_repo_rel_allowed(rel: str) -> bool:
        rel = str(rel).replace("\\", "/").lstrip("/")
        if not rel or rel.endswith("/"):
            return False
        low = rel.lower()
        # Never touch FrameVision's imported MiniMax helper/queue integration.
        if low == "helpers" or low.startswith("helpers/"):
            return False
        # Standalone launch/document/install files do not belong in FrameVision root.
        if "/" not in rel and low in {"readme.md", "start.bat", "install.bat"}:
            return False
        # Keep FrameVision's relocated/non-interactive wrapper installer intact.
        if low == "presets/extra_env/install.bat":
            return False
        if "__pycache__" in {part.lower() for part in rel.split("/")} or low.endswith((".pyc", ".pyo")):
            return False
        return True

    def _bootstrap_repo_worker(self):
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="framevision_minimax_h3_"))
            zip_path = temp_dir / "repo.zip"
            req = urllib.request.Request(
                APP_UPDATE_ZIP,
                headers={
                    "User-Agent": "FrameVision-MiniMax-H3-Installer/1.0",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as response, zip_path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)

            extract_dir = temp_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            roots = [x for x in extract_dir.iterdir() if x.is_dir()]
            if len(roots) != 1:
                raise RuntimeError("Unexpected GitHub repository ZIP layout.")
            source_root = roots[0]

            copied = 0
            for src in source_root.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(source_root).as_posix()
                if not self._bootstrap_repo_rel_allowed(rel):
                    continue
                dst = ROOT / Path(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".minimax_install_tmp")
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                copied += 1
            self.bootstrap_repo_finished.emit({"copied": copied})
        except Exception as exc:
            self.bootstrap_failed.emit(f"Repository setup failed: {type(exc).__name__}: {exc}")
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _bootstrap_repo_ready(self, payload):
        if not self._bootstrap_running:
            return
        self._set_bootstrap_stage("Installing MiniMax H3 environment…")
        installer = ROOT / "presets" / "extra_env" / "install.bat"
        proc = QProcess(self)
        self._bootstrap_proc = proc
        self._bootstrap_stage = "environment"
        proc.setWorkingDirectory(str(ROOT))
        proc.setProgram("cmd.exe")
        proc.setArguments(["/d", "/c", str(installer)])
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MINIMAX_H3_GUI_INSTALL", "1")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._bootstrap_process_output)
        proc.finished.connect(self._bootstrap_environment_finished)
        proc.start()
        if not proc.waitForStarted(5000):
            self._bootstrap_install_failed("Could not start presets\\extra_env\\install_minimax_h3.bat")

    def _bootstrap_process_output(self):
        proc = self._bootstrap_proc
        if proc is None:
            return
        try:
            raw = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            return
        # Keep the GUI responsive and useful without dumping an installer console
        # into the footer.  The last meaningful line becomes the status text.
        lines = [x.strip() for x in raw.replace("\r", "\n").split("\n") if x.strip()]
        if lines:
            self.status.setText(lines[-1][:180])

    def _bootstrap_environment_finished(self, exit_code, exit_status):
        if not self._bootstrap_running or self._bootstrap_stage != "environment":
            return
        self._bootstrap_proc = None
        if int(exit_code) != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self._bootstrap_install_failed(f"Environment installer failed with exit code {int(exit_code)}.")
            return
        if not PYTHON.is_file():
            self._bootstrap_install_failed(f"Environment installer finished, but Python was not found at:\n{PYTHON}")
            return
        self._start_bootstrap_model_download()

    def _start_bootstrap_model_download(self):
        self._set_bootstrap_stage("Downloading both MiniMax H3 models and all available LoRAs…")
        code = r'''from runtime import download_models as d
files=d.repo_files()
need=[d.REMOTE_FL2VA,d.REMOTE_REF2VA,d.REMOTE_TE,d.REMOTE_VVAE,d.REMOTE_AVAE]
missing=[x for x in need if x not in files]
if missing: raise RuntimeError("Required model file(s) missing from repository: " + ", ".join(missing))
loras=d.live_lora_files()
d.download(d.REMOTE_FL2VA,d.DIFF,d.LOCAL_FL2VA)
d.download(d.REMOTE_REF2VA,d.DIFF,d.LOCAL_REF2VA)
d.download(d.REMOTE_TE,d.TE,d.LOCAL_TE)
d.download(d.REMOTE_VVAE,d.VVAE,d.LOCAL_VVAE)
d.download(d.REMOTE_AVAE,d.AVAE,d.LOCAL_AVAE)
for item in loras: d.download(item["path"],d.LORAS,d.Path(item["path"]).name)
print("FRAMEVISION_MINIMAX_ALL_DOWNLOADS_COMPLETE", flush=True)
'''
        proc = QProcess(self)
        self._bootstrap_proc = proc
        self._bootstrap_stage = "models"
        proc.setWorkingDirectory(str(ROOT))
        proc.setProgram(str(PYTHON))
        proc.setArguments(["-c", code])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._bootstrap_process_output)
        proc.finished.connect(self._bootstrap_models_finished)
        proc.start()
        if not proc.waitForStarted(5000):
            self._bootstrap_install_failed("The MiniMax H3 model downloader could not be started.")

    def _bootstrap_models_finished(self, exit_code, exit_status):
        if not self._bootstrap_running or self._bootstrap_stage != "models":
            return
        self._bootstrap_proc = None
        if int(exit_code) != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self._bootstrap_install_failed(f"Model/LoRA download failed with exit code {int(exit_code)}.")
            return
        self._bootstrap_running = False
        self._bootstrap_stage = ""
        self._set_bootstrap_stage("MiniMax H3 installation complete")
        self._refresh_bootstrap_button()
        self.validate_install()

    def _bootstrap_install_failed(self, message):
        proc = self._bootstrap_proc
        self._bootstrap_proc = None
        self._bootstrap_running = False
        self._bootstrap_stage = ""
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            proc.kill()
        self.status.setText("MiniMax H3 installation failed")
        # Keep the install button visible even when a failed attempt happened to
        # create one or more of the marker directories.
        if hasattr(self, "install_minimax"):
            self.install_minimax.setVisible(True)
            self.install_minimax.setEnabled(True)
            self.install_minimax.setText("Install MiniMax H3 model & repo")
            self.install_minimax_progress.setVisible(False)
        QMessageBox.critical(self, "MiniMax H3 installer", str(message))

    @staticmethod
    def _update_rel_allowed(rel: str) -> bool:
        rel = str(rel).replace("\\", "/").lstrip("/")
        if not rel or rel.endswith("/"):
            return False
        parts = rel.split("/")
        if not parts or parts[0].lower() in {x.lower() for x in APP_UPDATE_EXCLUDED_TOP}:
            return False
        low = rel.lower()
        # FrameVision owns these two GUI integration files. Never replace them
        # with standalone-repository copies during an application update.
        if low in {"helpers/minimax_h3_gui.py", "helpers/minimax_music_clip.py"}:
            return False
        if any(low == p.lower() or low.startswith(p.lower() + "/") for p in APP_UPDATE_EXCLUDED_PREFIXES):
            return False
        if "__pycache__" in {x.lower() for x in parts} or low.endswith((".pyc", ".pyo")):
            return False
        return True

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _startup_update_check(self):
        if getattr(self, "_closing", False):
            return
        if hasattr(self, "auto_update_enabled") and self.auto_update_enabled.isChecked():
            self._check_for_updates(manual=False)

    def _check_for_updates(self, manual=False):
        if self._update_check_running:
            if manual:
                QMessageBox.information(self, "Application update", "An update check is already running.")
            return
        if not manual and (not hasattr(self, "auto_update_enabled") or not self.auto_update_enabled.isChecked()):
            return
        self._update_check_running = True
        if hasattr(self, "check_update_now"):
            self.check_update_now.setEnabled(False)
        if manual:
            self.status.setText("Checking GitHub for app updates…")
        threading.Thread(target=self._update_check_worker, args=(bool(manual),), daemon=True).start()

    def _update_check_worker(self, manual: bool):
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="minimax_h3_update_"))
            zip_path = temp_dir / "update.zip"
            req = urllib.request.Request(
                APP_UPDATE_ZIP,
                headers={
                    "User-Agent": "MiniMax-H3-Standalone-Updater/1.0",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response, zip_path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
                final_url = response.geturl()

            extract_dir = temp_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            if len(roots) != 1:
                raise RuntimeError("Unexpected GitHub update archive layout.")
            source_root = roots[0]
            changed = []
            upstream_manifest = []
            for src in source_root.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(source_root).as_posix()
                if not self._update_rel_allowed(rel):
                    continue
                upstream_manifest.append(rel)
                dst = ROOT / Path(rel)
                if not dst.is_file() or src.stat().st_size != dst.stat().st_size or self._sha256_file(src) != self._sha256_file(dst):
                    changed.append(rel)

            commit = ""
            m = re.search(r"/legacy\.zip/[^/]+/([0-9a-fA-F]{7,40})", final_url or "")
            if m:
                commit = m.group(1)
            payload = {
                "manual": manual,
                "temp_dir": str(temp_dir),
                "source_root": str(source_root),
                "changed": changed,
                "manifest": upstream_manifest,
                "commit": commit,
            }
            temp_dir = None  # GUI handler owns cleanup now.
            self.update_check_finished.emit(payload)
        except Exception as exc:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self.update_check_failed.emit({"manual": manual, "message": f"{type(exc).__name__}: {exc}"})

    def _cleanup_update_payload(self, payload=None):
        payload = payload or self._update_payload
        if payload and payload.get("temp_dir"):
            shutil.rmtree(payload["temp_dir"], ignore_errors=True)
        if payload is self._update_payload:
            self._update_payload = None

    def _handle_update_check_failed(self, payload):
        self._update_check_running = False
        if hasattr(self, "check_update_now"):
            self.check_update_now.setEnabled(True)
        manual = bool((payload or {}).get("manual")) if isinstance(payload, dict) else False
        message = (payload or {}).get("message", "Unknown update-check error") if isinstance(payload, dict) else str(payload)
        # Automatic checks stay quiet on transient network errors; manual checks explain them.
        if manual:
            self.status.setText("Update check failed")
            QMessageBox.warning(self, "Application update", f"Could not check GitHub for updates.\n\n{message}")

    def _handle_update_check_finished(self, payload):
        self._update_check_running = False
        if hasattr(self, "check_update_now"):
            self.check_update_now.setEnabled(True)
        self._update_payload = payload
        changed = payload.get("changed", [])
        manual = bool(payload.get("manual"))
        if not changed:
            self._cleanup_update_payload(payload)
            if manual:
                self.status.setText("Application is up to date")
                QMessageBox.information(self, "Application update", "No changed application files were found. You are up to date.")
            return

        preview = "\n".join(f"• {x}" for x in changed[:12])
        if len(changed) > 12:
            preview += f"\n• … and {len(changed) - 12} more file(s)"
        answer = QMessageBox.question(
            self,
            "Application update available",
            f"A MiniMax H3 Standalone update is available.\n\n"
            f"{len(changed)} application file(s) differ from the GitHub version:\n\n{preview}\n\n"
            "Install the update now?\n\n"
            "Your models, Conda environment, outputs, logs and saved settings will not be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._cleanup_update_payload(payload)
            self.status.setText("Update available - not installed")
            return
        self._install_update(payload)

    def _install_update(self, payload):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "Application update",
                "A generation is currently running. Finish or cancel the running job before installing the application update."
            )
            self._cleanup_update_payload(payload)
            return
        source_root = Path(payload["source_root"])
        changed = payload.get("changed", [])
        try:
            for rel in changed:
                if not self._update_rel_allowed(rel):
                    continue
                src = source_root / Path(rel)
                dst = ROOT / Path(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".update_tmp")
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)

            PRESET_DIR.mkdir(parents=True, exist_ok=True)
            state = {
                "repository": APP_UPDATE_REPO,
                "commit": payload.get("commit", ""),
                "installed_at": datetime.now().isoformat(timespec="seconds"),
                "files": payload.get("manifest", []),
            }
            APP_UPDATE_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
            self.status.setText(f"Application update installed ({len(changed)} files)")
        except Exception as exc:
            QMessageBox.critical(self, "Application update", f"The update could not be installed.\n\n{type(exc).__name__}: {exc}")
            self._cleanup_update_payload(payload)
            return
        finally:
            self._cleanup_update_payload(payload)

        restart = QMessageBox.question(
            self,
            "Application update installed",
            "The update was installed successfully and the temporary download folder was cleaned.\n\nRestart MiniMax H3 Standalone now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if restart == QMessageBox.StandardButton.Yes:
            self.save_last()
            start_bat = ROOT / "start.bat"
            try:
                if start_bat.is_file():
                    if os.name == "nt" and hasattr(os, "startfile"):
                        os.startfile(str(start_bat))
                    else:
                        subprocess.Popen([str(start_bat)], cwd=str(ROOT))
                else:
                    subprocess.Popen([str(PYTHON), str(Path(__file__).resolve())], cwd=str(ROOT))
                QApplication.instance().quit()
            except Exception as exc:
                QMessageBox.warning(self, "Restart", f"The update is installed, but the app could not restart automatically.\n\n{exc}\n\nPlease restart it manually.")

    def validate_install(self):
        if not PYTHON.is_file(): self.status.setText("Environment missing"); return
        self.status.setText("Validating…")
        args = ["-m", "runtime.validate_models"] + self.model_override_args()
        mode = self.mode.currentIndex()
        args += ["--mode", "ref2va" if mode == 2 else "fl2va"]
        p = QProcess(self); p.setWorkingDirectory(str(ROOT)); p.setProgram(str(PYTHON)); p.setArguments(args); p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        buf = []
        p.readyReadStandardOutput.connect(lambda: buf.append(bytes(p.readAllStandardOutput()).decode(errors='replace')))
        def done(code, status):
            text = ''.join(buf)
            if self.extended_logging.isChecked(): self.append_log(text + "\n")
            elif code != 0: self.append_log(text + "\n")
            else: self.append_log("Validation OK\n")
            self.status.setText("Install OK" if code == 0 else "Validation failed")
        p.finished.connect(done); p.start(); self._validator = p

    def _unload_prompt_builder_llm_for_generation(self):
        """Release any Prompt Builder LLM before a MiniMax job is queued/started.

        This deliberately mirrors the Prompt Builder's own Unload button so a local
        GGUF/llama-server or Ollama model cannot quietly keep several GiB of VRAM
        occupied when video generation begins. Failures are non-fatal but are logged.
        """
        if not self.builder_port or not self._builder_status_on_port(self.builder_port):
            return
        base = f"http://127.0.0.1:{int(self.builder_port)}"
        try:
            with urllib.request.urlopen(base + "/api/settings/llm", timeout=1.5) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
            settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
            backend = str(settings.get("backend", "gguf") or "gguf").strip().lower()

            if backend == "gguf":
                req = urllib.request.Request(
                    base + "/api/gguf/unload", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST"
                )
                urllib.request.urlopen(req, timeout=5.0).read()
                self.append_log("[LLM] Prompt Builder GGUF model unloaded before generation.\n")
            elif backend == "ollama":
                model = str(settings.get("ollama_model", "") or "").strip()
                url = str(settings.get("ollama_url", "http://localhost:11434") or "http://localhost:11434").strip()
                if model:
                    body = json.dumps({"url": url, "model": model, "action": "unload"}).encode("utf-8")
                    req = urllib.request.Request(
                        base + "/api/ollama/model", data=body,
                        headers={"Content-Type": "application/json"}, method="POST"
                    )
                    urllib.request.urlopen(req, timeout=35.0).read()
                    self.append_log(f"[LLM] Prompt Builder Ollama model unloaded before generation: {model}\n")
            # Built-in/browser-only backends do not hold a local model in this app.
        except Exception as exc:
            self.append_log(f"[LLM] Automatic Prompt Builder model unload warning: {exc}\n")

    def generate(self):
        prompt=self.prompt.toPlainText().strip()
        if not prompt: QMessageBox.warning(self,"Prompt required","Enter a prompt before adding the job to the queue."); return
        if not PYTHON.is_file(): QMessageBox.critical(self,"Environment missing",f"Missing {PYTHON}"); return
        mode=self.mode.currentIndex()
        hybrid_checkpoint = ""
        if self.use_hybrid_model.isChecked():
            hybrid_checkpoint = self._find_hybrid_checkpoint()
            if not hybrid_checkpoint:
                QMessageBox.critical(
                    self,
                    "Hybrid model not found",
                    "Use hybrid model is enabled, but no .safetensors checkpoint with 'hybrid' in its filename was found in the MiniMax H3 model folders.\n\nPlace the hybrid checkpoint under models\\minimax_h3 (normally diffusion_models), or point one of the FL2VA/Ref2VA checkpoint fields at the folder containing it before enabling hybrid mode."
                )
                return
        w,h=RESOLUTION_PRESETS[self.res_class.currentText()][self.aspect.currentText()]; frames=self._frame_count()
        long_mode = self.experimental_long_duration.isChecked()
        max_frames = EXPERIMENTAL_FRAME_MAX if long_mode else NORMAL_FRAME_MAX
        if frames > max_frames:
            QMessageBox.critical(self, "Invalid frames", f"Maximum frame count for the selected mode is {max_frames} ({max_frames / 24.0:.2f} seconds at 24 FPS)."); return
        # Generation always gets first claim on VRAM. If the integrated Prompt Builder
        # currently has a local LLM loaded, unload it exactly like its Settings button.
        self._unload_prompt_builder_llm_for_generation()
        script="helpers/generate.py" if mode<2 else "helpers/generate_ref.py"
        for row, strength in self.lora_rows:
            lp = row.path()
            if lp and float(strength.value()) != 0.0 and not Path(lp).is_file():
                QMessageBox.critical(self, "LoRA missing", f"Selected LoRA file was not found:\n{lp}"); return
        args=[script,"--width",str(w),"--height",str(h),"--frames",str(frames),"--steps",str(self.steps.value()),"--cfg",str(self.cfg.value()),"--shift",str(self.shift.value()),"--audio-shift",str(self.audio_shift.value()),"--seed",str(self.seed.value()),"--sampler",self.sampler.currentText(),"--scheduler",self.scheduler.currentText(),"--prompt",prompt]
        if long_mode:
            args += ["--experimental-long-duration"]
        continue_last=False; continue_from_job_id=None; continue_from_job_number=None; manual_continue_video=""; glue_results=False; continue_audio_memory=False
        if mode==1:
            continue_last=self.continue_last_result.isChecked()
            glue_results=self.glue_results.isChecked()
            continue_audio_memory=bool(continue_last and self.continue_audio_memory.isChecked())
            manual_continue_video="" if continue_last else self.continue_video.path()
            if (manual_continue_video or continue_last) and self.first.path():
                QMessageBox.warning(self,"Conflicting FL2VA inputs","Continue Video already supplies the first-frame boundary. Clear the separate First frame."); return
            if self.first.path(): args += ["--first-frame",self.first.path()]
            if self.last.path(): args += ["--last-frame",self.last.path()]
            if manual_continue_video and not Path(manual_continue_video).is_file():
                QMessageBox.warning(self,"Video missing","The selected Continue video file does not exist."); return
            if continue_last:
                previous=self._latest_framevision_minimax_job() if self._framevision_queue_mode() else self._latest_non_cancelled_queue_job()
                if previous is None:
                    QMessageBox.warning(self,"No previous queue job","Continue last result needs an earlier non-cancelled MiniMax queue job to continue from. Add or finish a source job first."); return
                if previous.get("state")=="failed":
                    QMessageBox.warning(self,"Previous job failed","The latest non-cancelled MiniMax queue job failed and continuation stops there. Remove the failed job or run a new successful source job first."); return
                if previous.get("state")=="finished":
                    pa = previous.get("args") or {}
                    prev_output = previous.get("produced") or previous.get("output") or pa.get("outfile") or pa.get("out_file")
                    if not prev_output or not Path(str(prev_output)).is_file():
                        QMessageBox.warning(self,"Previous output missing","The latest non-cancelled MiniMax queue job is finished but its output file is missing."); return
                continue_from_job_id=previous.get("id")
                continue_from_job_number=previous.get("job_number") or previous.get("id")
            if glue_results and not (manual_continue_video or continue_last):
                QMessageBox.warning(self,"Glue source required","Glue results requires either a selected Continue video or Continue last result."); return
            if not self.first.path() and not self.last.path() and not manual_continue_video and not continue_last:
                QMessageBox.warning(self,"Visual input required","Choose a first frame, last frame, Continue video, or Continue last result for FL2VA."); return
        elif mode==2:
            refs=self.ref_images.paths()+self.ref_videos.paths()+self.ref_audios.paths()
            if not refs: QMessageBox.warning(self,"Reference required","Add at least one Ref2VA reference."); return
            args += ["--ref-image-size",self.ref_size.currentText()]
            for pth in self.ref_images.paths(): args += ["--ref-image",pth]
            for pth in self.ref_videos.paths(): args += ["--ref-video",pth]
            for pth in self.ref_audios.paths(): args += ["--ref-audio",pth]
        args += self.model_override_args()
        args += self.lora_args()
        if self.vram_manager_enabled.isChecked():
            args += ["--vram-manager-auto" if self.vram_manager_auto_bypass.isChecked() else "--vram-manager"]
            args += ["--vram-residency-engine", str(self.vram_residency_engine.currentData() or "static"), "--vram-runtime-free-gb", str(self.vram_runtime_free.value()), "--vram-text-headroom-gb", str(self.vram_text_headroom.value()), "--vram-diffusion-headroom-gb", str(self.vram_diffusion_headroom.value()), "--vram-offload-chunk-mb", str(self.vram_offload_chunk.value()), "--vram-max-resident-weights-gb", str(self.vram_max_weights.value()), "--vram-block-check-interval", str(self.vram_block_interval.value()), "--vram-async-streams", str(self.vram_async_streams.value()), "--vram-video-vae-reserve-gb", str(self.vram_video_vae_reserve.value()), "--vram-audio-vae-reserve-gb", str(self.vram_audio_vae_reserve.value()), "--vram-residency-target-free-gb", str(self.vram_residency_target_free.value()), "--vram-residency-warmup-blocks", str(self.vram_residency_warmup.value()), "--vram-residency-refill-interval", str(self.vram_residency_refill_interval.value())]
            args += ["--vram-residency-fill" if self.vram_residency_fill.isChecked() else "--no-vram-residency-fill"]
        if self.spectrum_enabled.isChecked(): args += ["--spectrum"]
        if self.sage_attention_enabled.isChecked(): args += ["--sage-attention"]
        if not self.comfy_kitchen_enabled.isChecked(): args += ["--disable-comfy-kitchen"]
        # Video-VAE tiling is independent from sampling-side VRAM Manager activation.
        # Keep the proven 256/128 defaults unless the user deliberately changes them for testing.
        tile_size = int(self.vram_video_vae_tile_size.value())
        tile_overlap = int(self.vram_video_vae_tile_overlap.value())
        if tile_overlap >= tile_size:
            QMessageBox.warning(self, "Invalid VAE tiling", "Video VAE tile overlap must be smaller than the tile size.")
            return
        args += ["--video-vae-tile-size", str(tile_size), "--video-vae-tile-overlap", str(tile_overlap)]
        if self.extended_logging.isChecked(): args += ["--extended-logging"]
        if self.tile_debugging.isChecked(): args += ["--tile-debugging"]
        out=self.make_output_path(mode)
        # Queue jobs must never silently overwrite one another (or an existing clip), even when
        # the user entered a fixed output name. Preserve the requested base and add _002, _003...
        used={str(Path(j.get("output","")).resolve()).lower() for j in self.queue_jobs if j.get("output")}
        if self._framevision_queue_mode():
            used.update(self._framevision_reserved_outputs())
        base=out; n=2
        while out.exists() or str(out.resolve()).lower() in used:
            out=base.with_name(f"{base.stem}_{n:03d}{base.suffix}"); n+=1
        args += ["--output",str(out)]
        if self.use_hybrid_model.isChecked():
            model_path = hybrid_checkpoint or self._find_hybrid_checkpoint()
            model_label = f"Hybrid: {Path(model_path).name}" if model_path else "Hybrid model"
        else:
            model_path=self.ref2va_model.path() if mode==2 else self.fl2va_model.path()
            model_label=Path(model_path).name if model_path else ("Ref2VA INT4 (default)" if mode==2 else "FL2VA INT4 (default)")
        job={"id":uuid.uuid4().hex,"job_number":self._take_next_job_number(),"state":"pending","created_at":time.time(),"started_at":None,"finished_at":None,"elapsed":0,"mode":mode,"mode_name":self.mode.currentText(),"model_label":model_label,"output":str(out),"seed":self.seed.value(),"actual_seed":None,"resolution":f"{w} × {h}","frames":frames,"steps":self.steps.value(),"prompt":prompt,"args":args,"progress":None,"phase":"Waiting","error":"","cancel_reason":"","settings":self.settings_dict(),"log_tail":"","continue_last_result":bool(continue_last),"continue_from_job_id":continue_from_job_id,"continue_from_job_number":continue_from_job_number,"manual_continue_video":manual_continue_video,"continue_context_frames":int(self.continue_context.currentData() or 39) if mode==1 else None,"glue_results":bool(glue_results),"continue_audio_memory":bool(continue_audio_memory),"lanczos_scale_2x":bool(self.lanczos_scale_2x.isChecked())}
        if self._framevision_queue_mode():
            try:
                qid = self._enqueue_framevision_minimax_job(job)
            except Exception as exc:
                QMessageBox.critical(self, "FrameVision queue", f"Could not add the MiniMax job to FrameVision queue:\n\n{exc}")
                return
            self.save_last()
            self.status.setText(f"MiniMax job added to FrameVision queue ({qid})")
            return
        self.queue_jobs.append(job); self.save_last(); self._save_queue_state(); self._refresh_queue_views(); self.status.setText("Job added to queue")
        self._start_next_pending()

    def _apply_lanczos_scale_2x(self, output_path):
        """Replace a completed MiniMax MP4 with a 2× Lanczos-scaled copy."""
        src = Path(output_path or "")
        if not src.is_file():
            return False, f"Lanczos scaling source is missing: {src}"
        ffmpeg = ffmpeg_tool_path("ffmpeg.exe")
        if not ffmpeg or not Path(ffmpeg).is_file():
            return False, "FFmpeg is not available for Lanczos scaling."
        tmp = src.with_name(f".{src.stem}.lanczos2x.{uuid.uuid4().hex[:8]}.mp4")
        cmd = [
            str(ffmpeg), "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=iw*2:ih*2:flags=lanczos",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "copy", "-movflags", "+faststart", str(tmp),
        ]
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", creationflags=flags)
            if cp.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
                try: tmp.unlink(missing_ok=True)
                except Exception: pass
                tail = (cp.stdout or "").strip().splitlines()[-8:]
                return False, "Lanczos scaling failed" + ((":\n" + "\n".join(tail)) if tail else ".")
            os.replace(str(tmp), str(src))
            return True, ""
        except Exception as exc:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return False, f"Lanczos scaling failed: {exc}"

    def _play_completed_result(self, job):
        if not job or job.get("state") != "finished":
            return
        path = Path(job.get("output", ""))
        if not path.is_file():
            return
        if self.play_result_queue_player.isChecked():
            self.tabs.setCurrentIndex(2)
            self._load_preview(job, autoplay=True)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _finished(self, code, status):
        self._process_output()
        if self._closing:
            self.proc=None; return
        job=self._job_by_id(self.current_job_id) if self.current_job_id else None; now=time.time()
        play_finished_job = None
        if job:
            job["finished_at"]=now; job["elapsed"]=max(0,now-float(job.get("started_at") or now)); job["progress"]=None
            if self._termination_action=="requeue":
                job["state"]="pending"; job["started_at"]=None; job["finished_at"]=None; job["elapsed"]=0; job["phase"]="Waiting"; self.queue_jobs.remove(job); self.queue_jobs.append(job)
            elif self._termination_action=="cancel":
                job["state"]="cancelled"; job["error"]=job.get("cancel_reason") or "Cancelled by user."
            elif code in (0,3) and Path(job.get("output","")).is_file():
                if bool(job.get("lanczos_scale_2x")):
                    self.status.setText("Applying 2× Lanczos scaling…")
                    self.append_log("Applying 2× Lanczos scaling to saved output…\n")
                    ok, scale_error = self._apply_lanczos_scale_2x(job.get("output"))
                    if not ok:
                        job["state"]="failed"; job["error"]=scale_error; job["phase"]="Lanczos scaling failed"
                    else:
                        self.append_log("Lanczos scaling complete.\n")
                        job["state"]="finished"; job["phase"]="Finished"
                else:
                    job["state"]="finished"; job["phase"]="Finished"
                if job.get("state") == "finished":
                    job["clip_duration"] = self._probe_clip_duration(job.get("output"), job.get("frames"))
                if job.get("state") == "finished" and self.play_result_finished.isChecked():
                    play_finished_job = job
            else:
                job["state"]="failed"; job["error"]=self._extract_failure_reason(job,code); job["phase"]="Failed"
        effective_code = code
        if job and job.get("state") == "failed" and code in (0,3):
            effective_code = 2
        finish_line=f"=== FINISHED: exit {effective_code} ===\n"
        self._write_job_log_file(finish_line)
        self.append_log(finish_line); self.proc=None; self.current_job_id=None; self._termination_action=None; self.cancel.setEnabled(False); self.gen.setEnabled(True)
        self._active_log_file=None
        self.status.setText("Queue ready" if effective_code in (0,3) else f"Job stopped/failed ({effective_code})"); self._save_queue_state(); self._refresh_queue_views()
        if job and job.get("music_clip_job") and getattr(self, "music_clip_widget", None) is not None:
            try:
                self.music_clip_widget.external_queue_updated(job)
            except Exception as exc:
                self.append_log(f"Music Clip review refresh warning: {exc}\n")
        if play_finished_job is not None:
            QTimer.singleShot(50, lambda j=play_finished_job: self._play_completed_result(j))
        QTimer.singleShot(150,self._start_next_pending)

    def cancel_job(self):
        self._stop_running_job("cancel")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Qt/Windows can finalize maximized client geometry one event after the
        # initial show.  Re-run width-sensitive layouts against the real viewport.
        self._schedule_layout_refresh()

    def _restore_expected_maximized_state(self):
        """Undo accidental/programmatic restores while respecting the user."""
        self._window_state_guard_pending = False
        if self._embedded or self._closing or self._user_window_size_override:
            return
        state = self.windowState()
        if state & Qt.WindowState.WindowMinimized:
            return
        if not (state & Qt.WindowState.WindowMaximized):
            self.showMaximized()

    def changeEvent(self, event):
        if (not self._embedded) and event.type() == QEvent.Type.WindowStateChange:
            self._schedule_layout_refresh()
            state = self.windowState()
            minimized = bool(state & Qt.WindowState.WindowMinimized)
            maximized = bool(state & Qt.WindowState.WindowMaximized)

            # Native title-bar actions, dragging a maximized window, Win+Arrow,
            # etc. arrive as spontaneous events.  Those are user decisions and
            # must never be fought by the application.  Minimizing is also always
            # allowed; a maximized window will naturally restore maximized.
            if event.spontaneous() and not minimized:
                self._user_window_size_override = not maximized
            elif (
                not minimized
                and not maximized
                and not self._user_window_size_override
                and not self._window_state_guard_pending
            ):
                # Layout refreshes / programmatic show-state changes should not
                # unexpectedly knock the main window out of maximized mode.
                self._window_state_guard_pending = True
                QTimer.singleShot(0, self._restore_expected_maximized_state)

        super().changeEvent(event)

    def closeEvent(self, e):
        self._closing = True
        self.save_last()
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            job=self._job_by_id(self.current_job_id)
            if job:
                job["state"]="running"; job["cancel_reason"]="Application closed while this job was running."
                self._save_queue_state()
            pid=int(self.proc.processId())
            if os.name=='nt' and pid: subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else: self.proc.kill()
        self._stop_prompt_builder()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MiniMax H3 INT4 Standalone")
    w = MainWindow()
    # Start in the state most users keep this control-heavy GUI in.  A later
    # manual restore/resize is respected by MainWindow.changeEvent().
    w.showMaximized()
    QTimer.singleShot(0, w._schedule_layout_refresh)
    return app.exec()
if __name__ == "__main__": raise SystemExit(main())
