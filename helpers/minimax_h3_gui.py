from __future__ import annotations
import json, os, sys, subprocess, time, socket, urllib.request, urllib.parse, re, uuid, html, shutil, hashlib, tempfile, threading, zipfile, copy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QEvent, QUrl, QSizeF, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor, QPainter, QColor, QBrush, QPixmap, QIcon

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
except Exception:
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

try:
    from .minimax_timeline import TimelineTab
except ImportError:
    from minimax_timeline import TimelineTab

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments" / ".minimax_h3_int4" / "python.exe"
PRESET_DIR = ROOT / "presets" / "setsave"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_LORA_DIR = ROOT / "models" / "minimax_h3" / "loras"
LOG_DIR = ROOT / "logs"
APP_ICON = ROOT / "assets" / "grizzlymax-app.ico"

APP_UPDATE_REPO = "Koongrizzly/MiniMax_H3_Standalone_app"
APP_UPDATE_PAGE = f"https://github.com/{APP_UPDATE_REPO}"
APP_UPDATE_ZIP = f"https://api.github.com/repos/{APP_UPDATE_REPO}/zipball"
APP_UPDATE_STATE = PRESET_DIR / "minimax_h3_update_state.json"
APP_UPDATE_EXCLUDED_TOP = {"environments", "models", "output", "logs", "jobs", ".git"}
APP_UPDATE_EXCLUDED_PREFIXES = {"presets/setsave", "h3_prompt_builder/.runtime"}
FILE_DIALOG_HISTORY = PRESET_DIR / "minimax_file_dialog_history.json"

SUPPORTED_CHECKPOINT_REPO = "koongrizzly/MiniMax_H3_int4_W4A8_ConvRot_Pruned"
SUPPORTED_CHECKPOINT_SUBDIR = "diffusion_models"
SUPPORTED_CHECKPOINT_API = (
    f"https://huggingface.co/api/models/{SUPPORTED_CHECKPOINT_REPO}/tree/main/{SUPPORTED_CHECKPOINT_SUBDIR}"
    "?recursive=false&expand=false"
)
SUPPORTED_CHECKPOINT_RESOLVE = f"https://huggingface.co/{SUPPORTED_CHECKPOINT_REPO}/resolve/main/{SUPPORTED_CHECKPOINT_SUBDIR}"
SUPPORTED_CHECKPOINT_DIR = ROOT / "models" / "minimax_h3" / "diffusion_models"
ARIA2C = ROOT / "presets" / "bin" / "aria2c.exe"
ARIA2_VERSION = "1.37.0"
ARIA2_ARCHIVE_NAME = f"aria2-{ARIA2_VERSION}-win-64bit-build1.zip"
ARIA2_DOWNLOAD_URL = (
    f"https://github.com/aria2/aria2/releases/download/release-{ARIA2_VERSION}/{ARIA2_ARCHIVE_NAME}"
)
# Published by the official aria2 release and mirrored by Microsoft's winget manifest.
ARIA2_ARCHIVE_SHA256 = "67d015301eef0b612191212d564c5bb0a14b5b9c4796b76454276a4d28d9b288"


def _dialog_start_dir(key: str, preferred: str | Path | None = None, fallback: str | Path | None = None) -> str:
    """Return a useful persistent start folder for QFileDialog."""
    for candidate in (preferred,):
        if candidate:
            try:
                q = Path(str(candidate)).expanduser()
                if q.is_file():
                    q = q.parent
                if q.is_dir():
                    return str(q)
            except Exception:
                pass
    try:
        if FILE_DIALOG_HISTORY.is_file():
            data = json.loads(FILE_DIALOG_HISTORY.read_text(encoding="utf-8"))
            folders = data.get("folders", {}) if isinstance(data, dict) else {}
            for candidate in (folders.get(key), data.get("last_folder")):
                if candidate and Path(candidate).is_dir():
                    return str(Path(candidate))
    except Exception:
        pass
    try:
        q = Path(str(fallback or ROOT)).expanduser()
        if q.is_file():
            q = q.parent
        if q.is_dir():
            return str(q)
    except Exception:
        pass
    return str(ROOT)


def _remember_dialog_folder(key: str, selected: str | Path) -> None:
    if not selected:
        return
    try:
        q = Path(str(selected)).expanduser()
        folder = q if q.is_dir() else q.parent
        if not folder.is_dir():
            return
        data = {}
        if FILE_DIALOG_HISTORY.is_file():
            try:
                data = json.loads(FILE_DIALOG_HISTORY.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        folders = data.get("folders")
        if not isinstance(folders, dict):
            folders = {}
        folders[key] = str(folder)
        data["folders"] = folders
        data["last_folder"] = str(folder)
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        tmp = FILE_DIALOG_HISTORY.with_suffix(FILE_DIALOG_HISTORY.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(FILE_DIALOG_HISTORY)
    except Exception:
        pass


try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
except Exception:
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
from runtime.ffmpeg_tools import BIN_DIR as FFMPEG_BIN_DIR, tool_path as ffmpeg_tool_path, tools_ready as ffmpeg_tools_ready


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
            window = self.window()
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


class VideoFullscreenDialog(QDialog):
    """Fullscreen view of the shared preview player with its own transport controls."""
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Fullscreen preview")
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.scene = QGraphicsScene(self)
        self.view = ZoomVideoView(self)
        self.view.setScene(self.scene)
        self.video_item = QGraphicsVideoItem()
        self.video_item.setSize(QSizeF(1920, 1080))
        self.scene.addItem(self.video_item)
        lay.addWidget(self.view, 1)

        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play / Pause")
        self.stop_btn = QPushButton("Stop")
        self.repeat_box = QCheckBox("Repeat")
        self.exit_btn = QPushButton("Exit fullscreen")
        self.play_btn.clicked.connect(owner._preview_toggle)
        self.stop_btn.clicked.connect(owner.media_player.stop)
        self.repeat_box.setChecked(owner.preview_repeat.isChecked())
        self.repeat_box.toggled.connect(owner.preview_repeat.setChecked)
        owner.preview_repeat.toggled.connect(self.repeat_box.setChecked)
        self.exit_btn.clicked.connect(self.close)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addWidget(self.repeat_box)
        transport.addStretch(1)
        transport.addWidget(self.exit_btn)
        lay.addLayout(transport)

        seekrow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, int(owner.media_player.duration())))
        self.slider.sliderMoved.connect(owner.media_player.setPosition)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(105)
        seekrow.addWidget(self.slider, 1)
        seekrow.addWidget(self.time_label)
        lay.addLayout(seekrow)

        owner.media_player.positionChanged.connect(self._position_changed)
        owner.media_player.durationChanged.connect(self._duration_changed)

    @staticmethod
    def _fmt(ms):
        sec = max(0, int(ms) // 1000)
        return f"{sec//60:02d}:{sec%60:02d}"

    def _duration_changed(self, duration):
        self.slider.setRange(0, max(0, int(duration)))
        self._position_changed(self.owner.media_player.position())

    def _position_changed(self, position):
        if not self.slider.isSliderDown():
            self.slider.setValue(int(position))
        self.time_label.setText(
            f"{self._fmt(position)} / {self._fmt(self.owner.media_player.duration())}"
        )

    def open_fullscreen(self):
        self.repeat_box.setChecked(self.owner.preview_repeat.isChecked())
        self._duration_changed(self.owner.media_player.duration())
        self.owner.media_player.setVideoOutput(self.video_item)
        self.showFullScreen()
        QTimer.singleShot(0, self.view.reset_view)

    def closeEvent(self, event):
        try:
            self.owner.media_player.setVideoOutput(self.owner.video_item)
            QTimer.singleShot(0, self.owner.preview_view.reset_view)
        except Exception:
            pass
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)


# MiniMax H3 fixed resolution presets. Most combo labels are the exact generation
# dimensions. The familiar 1280 x 720 label is intentionally mapped to MiniMax's
# valid 704p tensor size (1280 x 704); 720 is not divisible by 32.
# Portrait presets transpose the actual generation dimensions and square presets
# use the matching short-edge generation size.
RESOLUTION_PRESETS = {
    "576 × 320":  {"16:9": (576, 320),  "9:16": (320, 576),  "1:1": (320, 320)},
    "736 × 384":  {"16:9": (736, 384),  "9:16": (384, 736),  "1:1": (384, 384)},
    "832 × 448":  {"16:9": (832, 448),  "9:16": (448, 832),  "1:1": (448, 448)},
    "960 × 544":  {"16:9": (960, 544),  "9:16": (544, 960),  "1:1": (544, 544)},
    "1024 × 576": {"16:9": (1024, 576), "9:16": (576, 1024), "1:1": (576, 576)},
    "1152 × 640": {"16:9": (1152, 640), "9:16": (640, 1152), "1:1": (640, 640)},
    "1280 × 720": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (704, 704)},
    "1344 × 768": {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (768, 768)},
    "1920 × 1088":{"16:9": (1920, 1088),"9:16": (1088, 1920),"1:1": (1088, 1088)},
}
# Exact 21:9 (7:3) presets. All dimensions stay on MiniMax's 32-pixel grid.
WIDESCREEN_21_9_PRESETS = {
    "Low — 896 × 384": (896, 384),
    "Medium — 1344 × 576": (1344, 576),
    "High — 1792 × 768": (1792, 768),
}
DEFAULT_RESOLUTION = "832 × 448"
NORMAL_FRAME_MAX = 719   # last H3 native-grid value below 30 seconds (29.958 s at 24 FPS)
EXPERIMENTAL_FRAME_MAX = 2385  # last H3 native-grid value at/below 100 seconds (99.375 s at 24 FPS)
FRAME_PRESETS = list(range(124, NORMAL_FRAME_MAX + 1, 17))
EXPERIMENTAL_FRAME_PRESETS = list(range(124, EXPERIMENTAL_FRAME_MAX + 1, 17))
SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab",
    "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2"
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
        start = _dialog_start_dir("images", self.path(), ROOT)
        p, _ = QFileDialog.getOpenFileName(self, "Select file", start, self.filter)
        if p:
            _remember_dialog_folder("images", p)
            self.edit.setText(p)

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
        start = _dialog_start_dir("videos", self.edit.text().strip(), ROOT)
        p, _ = QFileDialog.getOpenFileName(self, "Select source video to continue", start, "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if p:
            _remember_dialog_folder("videos", p)
            self.edit.setText(p)
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
        start = _dialog_start_dir("loras", self.edit.text().strip(), DEFAULT_LORA_DIR)
        p, _ = QFileDialog.getOpenFileName(self, "Select MiniMax H3 LoRA", start, MODEL_FILTER)
        if p:
            _remember_dialog_folder("loras", p)
            self.edit.setText(p)
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
        p, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", _dialog_start_dir("models", self.edit.text().strip(), ROOT), MODEL_FILTER)
        if p:
            _remember_dialog_folder("models", p)
            self.edit.setText(p)
    def browse_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select model folder", _dialog_start_dir("models", self.edit.text().strip(), ROOT))
        if p:
            _remember_dialog_folder("models", p)
            self.edit.setText(p)
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
        p = QFileDialog.getExistingDirectory(self, "Select folder", _dialog_start_dir("folders", self.edit.text().strip(), ROOT))
        if p:
            _remember_dialog_folder("folders", p)
            self.edit.setText(p)
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
        low_filter = self.filt.lower()
        history_key = "ref_audio" if "audio" in low_filter else ("ref_videos" if "video" in low_filter else "ref_images")
        paths, _ = QFileDialog.getOpenFileNames(self, "Add references", _dialog_start_dir(history_key, fallback=ROOT), self.filt)
        if paths:
            _remember_dialog_folder(history_key, paths[0])
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


def _human_bytes(value):
    try:
        size = float(value or 0)
    except Exception:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024.0
    return "—"


class SupportedCheckpointDialog(QDialog):
    """Repository-backed checkpoint picker. Downloads continue after this dialog closes."""
    results_ready = Signal(object)
    results_failed = Signal(str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Supported MiniMax H3 checkpoints")
        self.resize(880, 430)
        self.setModal(False)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Available supported diffusion checkpoints from the MiniMax H3 repository. "
            "Select one or more files. Installed checkpoints are detected automatically and will not be downloaded again."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.state = QLabel("Checking repository…")
        layout.addWidget(self.state)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Checkpoint", "Size", "Status"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_repository)
        self.download_btn = QPushButton("Download selected")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._download_selected)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(self.refresh_btn)
        row.addStretch()
        row.addWidget(self.download_btn)
        row.addWidget(close_btn)
        layout.addLayout(row)

        self.results_ready.connect(self._populate)
        self.results_failed.connect(self._show_error)
        QTimer.singleShot(0, self.load_repository)

    def load_repository(self):
        self.refresh_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.state.setText("Checking repository…")
        self.tree.clear()

        def worker():
            try:
                req = urllib.request.Request(
                    SUPPORTED_CHECKPOINT_API,
                    headers={"User-Agent": "MiniMax-H3-Standalone/CheckpointBrowser"},
                )
                with urllib.request.urlopen(req, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise RuntimeError("Unexpected Hugging Face repository response.")
                files = []
                for entry in payload:
                    if not isinstance(entry, dict) or entry.get("type") != "file":
                        continue
                    remote_path = str(entry.get("path") or "")
                    name = Path(remote_path).name
                    if not name.lower().endswith(".safetensors"):
                        continue
                    files.append({
                        "name": name,
                        "path": remote_path,
                        "size": int(entry.get("size") or 0),
                    })
                files.sort(key=lambda item: item["name"].lower())
                if not files:
                    raise RuntimeError("No .safetensors checkpoints were found in the repository folder.")
                self.results_ready.emit(files)
            except Exception as exc:
                self.results_failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _installed_state(self, info):
        target = SUPPORTED_CHECKPOINT_DIR / info["name"]
        control = Path(str(target) + ".aria2")
        if not target.exists():
            return False, "Available"
        expected = int(info.get("size") or 0)
        try:
            actual = target.stat().st_size
        except OSError:
            actual = 0
        if control.exists() or (expected and actual < expected):
            return False, f"Partial ({_human_bytes(actual)}) — resumable"
        return True, "Already installed"

    def _populate(self, files):
        if not self.isVisible():
            return
        self.refresh_btn.setEnabled(True)
        SUPPORTED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        available = 0
        for info in files:
            installed, status = self._installed_state(info)
            item = QTreeWidgetItem([info["name"], _human_bytes(info.get("size")), status])
            item.setData(0, Qt.ItemDataRole.UserRole, info)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if installed:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(0, Qt.CheckState.Unchecked)
            else:
                item.setFlags(flags)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                available += 1
            self.tree.addTopLevelItem(item)
        self.state.setText(
            f"{self.tree.topLevelItemCount()} supported checkpoint(s) found · "
            f"download folder: {SUPPORTED_CHECKPOINT_DIR}"
        )
        self.download_btn.setEnabled(available > 0)

    def _show_error(self, message):
        if not self.isVisible():
            return
        self.refresh_btn.setEnabled(True)
        self.state.setText("Could not read the supported checkpoint list.")
        QMessageBox.warning(self, "Checkpoint list", f"Could not read the Hugging Face repository:\n\n{message}")

    def _download_selected(self):
        selected = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                info = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(info, dict):
                    selected.append(info)
        if not selected:
            QMessageBox.information(self, "Supported checkpoints", "Select at least one checkpoint to download.")
            return
        if self.owner.start_supported_checkpoint_downloads(selected):
            self.accept()


class MainWindow(QMainWindow):
    update_check_finished = Signal(object)
    update_check_failed = Signal(object)
    aria2_bootstrap_finished = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("minimax H3 Standalone")
        # 1180x900 is only the fallback geometry used after the user explicitly
        # restores the window.  Normal application startup is maximized.
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
        self._preview_trim_start_ms = 0
        self._preview_trim_end_ms = None
        self._spinner_index = 0
        self._closing = False
        self._ffmpeg_setup_proc = None
        self._ffmpeg_setup_popup = None
        self._ffmpeg_setup_output = ""
        self._timeline_assembly_proc = None
        self._timeline_assembly_output = ""
        self._timeline_assembly_concat = None
        self._ffmpeg_setup_status = ""
        self._ffmpeg_setup_failed = False
        self._update_check_running = False
        self._update_payload = None
        self._checkpoint_dialog = None
        self._aria2_bootstrap_running = False
        self._aria2_pending_checkpoint_request = None
        self._checkpoint_download_proc = None
        self._checkpoint_download_queue = []
        self._checkpoint_download_total_bytes = 0
        self._checkpoint_download_completed_bytes = 0
        self._checkpoint_download_current = None
        self._checkpoint_download_current_start_size = 0
        self._checkpoint_download_output = ""
        self._checkpoint_progress_timer = QTimer(self)
        self._checkpoint_progress_timer.setInterval(500)
        self._checkpoint_progress_timer.timeout.connect(self._refresh_checkpoint_download_progress)
        self.aria2_bootstrap_finished.connect(self._handle_aria2_bootstrap_finished)
        self.update_check_finished.connect(self._handle_update_check_finished)
        self.update_check_failed.connect(self._handle_update_check_failed)
        self.wheel_filter = NoWheelFilter(self)
        self._build()
        self._apply_style()
        self._apply_first_run_defaults()
        self.load_last()
        self._load_queue_state()
        self._sync_resolution()
        self._sync_mode()
        QTimer.singleShot(300, self.validate_install)
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
        # WebEngine/native layout updates can occur after the tab change and
        # splitter update.  Re-assert the Prompt Builder exception after each
        # queued layout refresh so it cannot regress a moment later.
        if hasattr(self, "tabs") and hasattr(self, "global_preview_host"):
            self._enforce_prompt_builder_full_width()

    def _build(self):
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)
        hdr = QHBoxLayout(); title = QLabel("GrizzlyMax (Minimax H3 standalone)"); title.setObjectName("title")
        self.status = QLabel("Checking install…"); self.status.setObjectName("status")
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(self.status); outer.addLayout(hdr)

        # Always-visible system HUD. It sits outside the tabs so changing tabs never hides it.
        self.system_hud = SystemHud(self)
        outer.addWidget(self.system_hud, 0)

        # Persistent checkpoint download strip. The browser popup may be closed,
        # but aria2c remains owned by MainWindow and progress stays visible here.
        self.checkpoint_download_strip = QFrame(self)
        self.checkpoint_download_strip.setObjectName("checkpointDownloadStrip")
        dlrow = QHBoxLayout(self.checkpoint_download_strip)
        dlrow.setContentsMargins(8, 4, 8, 4)
        dlrow.setSpacing(8)
        self.checkpoint_download_label = QLabel("Checkpoint download")
        self.checkpoint_download_progress = QProgressBar()
        self.checkpoint_download_progress.setRange(0, 1000)
        self.checkpoint_download_progress.setValue(0)
        self.checkpoint_download_progress.setTextVisible(True)
        dlrow.addWidget(self.checkpoint_download_label)
        dlrow.addWidget(self.checkpoint_download_progress, 1)
        self.checkpoint_download_strip.hide()
        outer.addWidget(self.checkpoint_download_strip, 0)

        # One shared preview/player pane for the whole application.  The actual
        # preview widget is created by the Queue builder below, then adopted into
        # this outer splitter after all tabs exist.  Prompt Builder is the sole
        # exception: it keeps its full-width embedded web workspace.
        self.global_preview_splitter = QSplitter(Qt.Orientation.Horizontal, root)
        self.global_preview_splitter.setChildrenCollapsible(False)
        self.global_preview_splitter.setHandleWidth(6)
        self.global_preview_host = QWidget(self.global_preview_splitter)
        self.global_preview_host.setObjectName("GlobalPreviewHost")
        self.global_preview_host.setMinimumWidth(360)
        self.global_preview_layout = QVBoxLayout(self.global_preview_host)
        self.global_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.global_preview_layout.setSpacing(0)
        self.global_preview_host.hide()

        self.tabs = QTabWidget()
        # Main tabs are user-reorderable. Stable string keys are stored on the
        # QTabBar so the saved order remains valid even when new tabs are added.
        self.tabs.setMovable(True)
        # When the global preview pane is widened the right-hand workspace can
        # become narrower than the complete tab strip.  Keep all tabs at their
        # normal readable width and let Qt expose its native left/right tab
        # navigation buttons only while there is an overflow.
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setDocumentMode(True)
        # Do not let the largest tab's sizeHint impose an artificial minimum on
        # the right side of the global splitter.  Several tabs contain wide
        # controls, but they are already scrollable; the preview must still be
        # able to grow to about half of the application window.
        self.tabs.setMinimumWidth(0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.global_preview_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.global_preview_splitter.addWidget(self.global_preview_host)
        self.global_preview_splitter.addWidget(self.tabs)
        self.global_preview_splitter.setStretchFactor(0, 1)
        self.global_preview_splitter.setStretchFactor(1, 1)
        self.global_preview_splitter.setSizes([0, 1200])
        self.global_preview_splitter.splitterMoved.connect(self._remember_global_preview_width)
        outer.addWidget(self.global_preview_splitter, 1)

        self._build_generation_tab()
        self._build_prompt_builder_tab()
        self._build_queue_tab()
        self._build_music_clip_tab()
        self._build_timeline_tab()
        self._build_settings_tab()
        if getattr(self, "timeline_widget", None) is not None:
            self.timeline_widget.initialize_from_current_settings()
        self._adopt_global_preview_pane()

        # Fixed bottom bar: remains visible on every tab and while tab contents scroll.
        bar = QWidget(); bar.setObjectName("bottomBar")
        controls = QHBoxLayout(bar); controls.setContentsMargins(8, 8, 8, 8)
        self.gen = QPushButton("Generate"); self.gen.setObjectName("primary"); self.gen.clicked.connect(self._main_generate_action)
        self.cancel = QPushButton("Cancel"); self.cancel.clicked.connect(self.cancel_job); self.cancel.setEnabled(False)
        self.info_btn = QPushButton("Info"); self.info_btn.clicked.connect(self.open_feature_list)
        val = QPushButton("Validate install"); val.clicked.connect(self.validate_install)
        self.openout = QPushButton("Open output folder"); self.openout.clicked.connect(self.open_output_folder)
        controls.addWidget(self.gen); controls.addWidget(self.cancel); controls.addStretch(); controls.addWidget(self.info_btn); controls.addWidget(val); controls.addWidget(self.openout)
        outer.addWidget(bar, 0)
        self.setCentralWidget(root)
        self._add_tooltips()
        self.tabs.currentChanged.connect(self._sync_main_generate_button)
        self.tabs.currentChanged.connect(self._sync_global_preview_for_tab)
        self.tabs.tabBar().tabMoved.connect(self._sync_cached_tab_indexes)
        self._sync_main_generate_button(self.tabs.currentIndex())
        self._sync_global_preview_for_tab(self.tabs.currentIndex())

        for cls in (QComboBox, QSpinBox, QDoubleSpinBox):
            for w in self.findChildren(cls):
                w.installEventFilter(self.wheel_filter)
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _build_generation_tab(self):
        body = QWidget(); v = QVBoxLayout(body); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(10)
        basic = QGroupBox("Generation"); form = QFormLayout(basic)
        self.mode = QComboBox(); self.mode.addItems(["Text to video (T2VA)", "Image / Continue Video (FL2VA)", "Reference to video (Ref2VA)"]); self.mode.currentIndexChanged.connect(self._sync_mode)
        self.aspect = QComboBox(); self.aspect.addItems(["16:9", "9:16", "1:1", "21:9"]); self.aspect.currentTextChanged.connect(self._sync_resolution)
        self.res_class = QComboBox(); self.res_class.addItems(list(RESOLUTION_PRESETS)); self.res_class.setCurrentText(DEFAULT_RESOLUTION); self.res_class.currentTextChanged.connect(self._sync_resolution)
        self.widescreen_quality = QComboBox(); self.widescreen_quality.addItems(list(WIDESCREEN_21_9_PRESETS)); self.widescreen_quality.setCurrentText("Medium — 1344 × 576"); self.widescreen_quality.currentTextChanged.connect(self._sync_resolution)
        self.widescreen_quality.setVisible(False)
        self.resolved = QLabel()
        self.frames = QComboBox()
        for x in FRAME_PRESETS:
            self.frames.addItem(f"{x} frames — {x / 24.0:.2f} s", x)
        self._set_frame_count(124)
        self.experimental_long_duration = QCheckBox("Experimental long duration")
        self.experimental_long_duration.setChecked(False)
        self.experimental_long_duration.toggled.connect(self._sync_long_duration_mode)
        self.steps = QSpinBox(); self.steps.setRange(1, 100); self.steps.setValue(15)
        self.seed = QSpinBox(); self.seed.setRange(-1, 98_999_999); self.seed.setValue(-1); self.seed.setSpecialValueText("-1 (random)")
        form.addRow("Mode", self.mode)
        rr = QHBoxLayout(); rr.addWidget(self.res_class); rr.addWidget(self.widescreen_quality); rr.addWidget(self.aspect); rr.addWidget(self.resolved); rr.addStretch(); form.addRow("Resolution", rr)
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
        self.continue_context.setCurrentIndex(4)
        self.continue_context.setCurrentIndex(1)
        self.latent_continuation = QCheckBox("Use latent continuation (experimental/beta test)")
        self.latent_continuation.setChecked(False)
        self.latent_continuation.setToolTip(
            "Use the previous MiniMax H3 internal video/audio latent as continuation memory instead of rebuilding the motion history from decoded video frames. "
            "This avoids the video -> VAE re-encode round trip and can reduce cumulative quality/character drift in long chains. "
            "The previous result must have a compatible saved H3 latent at the same resolution. If no compatible latent is found, the app automatically falls back to normal video-frame continuation."
        )
        self.latent_continuation.toggled.connect(self._sync_continue_video_options)
        self.combine_frames_latent = QCheckBox("Combine frames memory & latent continuation")
        self.combine_frames_latent.setChecked(False)
        self.combine_frames_latent.setToolTip(
            "When latent continuation is enabled, also VAE-encode the decoded source-video history frames and provide them together with the saved native H3 latent history and exact final-frame boundary. "
            "This uses both frame memory and latent memory for stronger continuity. It costs some extra VAE work and memory."
        )
        self.combine_frames_latent.setVisible(False)
        self.combine_frames_latent.toggled.connect(self._sync_continue_video_options)
        self.continue_context.setToolTip(
            "Continuation history window. The same native 17k+5-aligned duration is used for decoded-frame memory and saved latent memory. "
            "When Combine frames memory & latent continuation is enabled, both memories use this same window."
        )
        self.glue_results = QCheckBox("Glue results")
        self.glue_results.setChecked(False)
        self.continue_last_result = QCheckBox("Continue last result")
        self.continue_last_result.setChecked(False)
        self.continue_last_result.toggled.connect(self._sync_continue_video_options)
        # Manual Continue video is also a valid source for sound-memory continuation,
        # so update the sound-memory control as soon as that path changes.
        if getattr(self.continue_video, "edit", None) is not None:
            self.continue_video.edit.textChanged.connect(self._sync_continue_video_options)
        self.continue_audio_memory_row = QWidget()
        caml = QHBoxLayout(self.continue_audio_memory_row); caml.setContentsMargins(0, 0, 0, 0)
        self.continue_audio_memory = QCheckBox("Use sound in memory for new clip")
        self.continue_audio_memory.setChecked(False)
        self.continue_audio_memory.setToolTip("Carries the final 1.00 s of source audio as H3 history. The 24-frame / 40-step window is end-aligned to the new clip on H3's native 40 Hz audio timeline.")
        self.continue_audio_memory_warning = QLabel("1.00 s timeline-aligned")
        self.continue_audio_memory_warning.setStyleSheet("color: #d58a00;")
        self.continue_audio_memory_warning.setToolTip("Test patch: full audio history ends exactly at target time zero; no guessed boundary-latent deletion is used.")
        caml.addWidget(self.continue_audio_memory); caml.addWidget(self.continue_audio_memory_warning); caml.addStretch()
        fl.addRow("First frame", self.first); fl.addRow("Last frame", self.last)
        fl.addRow("Continue video", self.continue_video); fl.addRow("Motion context", self.continue_context)
        fl.addRow("", self.latent_continuation)
        fl.addRow("", self.combine_frames_latent)
        fl.addRow("", self.glue_results); fl.addRow("", self.continue_last_result); fl.addRow("", self.continue_audio_memory_row); v.addWidget(self.fl_group)

        self.ref_group = QGroupBox("Ref2VA references"); rfl = QVBoxLayout(self.ref_group)
        note = QLabel("Prompt tags follow the native order: <Picture 1..9>, <Audio n> paired before <Video n>, then standalone <Audio n>."); note.setWordWrap(True); rfl.addWidget(note)
        self.ref_size = QComboBox(); self.ref_size.addItems(["match", "max"])
        rs = QFormLayout(); rs.addRow("Reference image size", self.ref_size); rfl.addLayout(rs)
        self.ref_images = RefList("Reference images (max 9)", "Images (*.png *.jpg *.jpeg *.webp *.bmp)", 9)
        self.ref_videos = RefList("Reference videos (max 3; soundtrack extracted when present)", "Video (*.mp4 *.mov *.mkv *.webm *.avi)", 3)
        self.ref_audios = RefList("Standalone reference audio (max 3)", "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg)", 3)
        self.lock_source_audio = QCheckBox("Use Audio 1 as exact source / output")
        self.lock_source_audio.setChecked(False)
        self.lock_source_audio.setToolTip(
            "OFF: all standalone audio files are used as normal Ref2VA references so MiniMax can reinterpret or regenerate audio. "
            "ON: Audio 1 is encoded into H3's target audio latent and locked during denoising, so the video follows the exact real timing while the untouched Audio 1 file is used in the final output. "
            "Additional audio files, if any, stay normal references."
        )
        rfl.addWidget(self.ref_images); rfl.addWidget(self.ref_videos); rfl.addWidget(self.ref_audios); rfl.addWidget(self.lock_source_audio); v.addWidget(self.ref_group)

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

        # The Generation tab can optionally share the queue's existing preview
        # player.  Keep the controls in their normal scroll area on the right and
        # reserve a splitter pane on the left for the preview when enabled.
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)
        self.generation_splitter = QSplitter(Qt.Orientation.Horizontal, page)
        self.generation_splitter.setChildrenCollapsible(False)
        self.generation_splitter.setHandleWidth(6)
        self.generation_preview_host = QWidget(self.generation_splitter)
        self.generation_preview_host.setObjectName("GenerationPreviewHost")
        self.generation_preview_host.setMinimumWidth(360)
        self.generation_preview_layout = QVBoxLayout(self.generation_preview_host)
        self.generation_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.generation_preview_layout.setSpacing(0)
        self.generation_preview_host.hide()
        generation_scroll = self._scroll_page(body)
        self.generation_splitter.addWidget(self.generation_preview_host)
        self.generation_splitter.addWidget(generation_scroll)
        self.generation_splitter.setStretchFactor(0, 5)
        self.generation_splitter.setStretchFactor(1, 6)
        self.generation_splitter.setSizes([520, 650])
        page_layout.addWidget(self.generation_splitter, 1)
        index = self.tabs.addTab(page, "Generation")
        self.tabs.tabBar().setTabData(index, "generation")

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

        # Keep a direct reference to the Prompt Builder page.  Tab indexes can
        # become stale if tabs/layouts are rebuilt or restored later, while the
        # widget identity is stable for the lifetime of this window.
        self.prompt_builder_page = page
        self.prompt_builder_tab_index = self.tabs.addTab(page, "Prompt Builder")
        self.tabs.tabBar().setTabData(self.prompt_builder_tab_index, "prompt_builder")

        # The embedded page starts asynchronously and can perform a visible
        # reload shortly after the user opens this tab.  Re-assert the Prompt
        # Builder's full-width layout around those loads so the global preview
        # can never pop back in during WebEngine startup/reload.
        if self.prompt_webview is not None:
            try:
                self.prompt_webview.loadStarted.connect(self._enforce_prompt_builder_full_width)
                self.prompt_webview.loadFinished.connect(lambda _ok=True: self._enforce_prompt_builder_full_width())
            except Exception:
                pass

        self.builder_timer = QTimer(self)
        self.builder_timer.setInterval(1500)
        self.builder_timer.timeout.connect(self._poll_prompt_builder_status)
        self.builder_timer.start()
        QTimer.singleShot(500, self._start_prompt_builder)

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
                "The maximum frame count that will actually run depends on the available system RAM and GPU VRAM. Default: 124 frames."
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
        # The player itself is kept in a movable pane so Settings can relocate
        # this exact widget to the Generation tab without creating a second media
        # player or losing the currently loaded clip/playback state.
        self.queue_preview_host = QWidget(splitter)
        self.queue_preview_host.setObjectName("QueuePreviewHost")
        self.queue_preview_host.setMinimumWidth(360)
        self.queue_preview_layout = QVBoxLayout(self.queue_preview_host)
        self.queue_preview_layout.setContentsMargins(0,0,0,0)
        self.queue_preview_layout.setSpacing(0)

        self.preview_pane = QWidget()
        self.preview_pane.setObjectName("QueuePreviewPane")
        self.preview_pane.setMinimumWidth(360)
        left_layout = QVBoxLayout(self.preview_pane)
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
            self.preview_fullscreen = QPushButton("Fullscreen")
            self.preview_play.clicked.connect(self._preview_toggle)
            self.preview_stop.clicked.connect(self.media_player.stop)
            self.preview_reset.clicked.connect(self.preview_view.reset_view)
            self.preview_fullscreen.clicked.connect(self._open_preview_fullscreen)
            transport.addWidget(self.preview_play)
            transport.addWidget(self.preview_stop)
            transport.addWidget(self.preview_repeat)
            transport.addStretch(1)
            transport.addWidget(self.preview_reset)
            transport.addWidget(self.preview_fullscreen)
            pv.addLayout(transport)

            seekrow = QHBoxLayout()
            self.preview_slider = QSlider(Qt.Orientation.Horizontal)
            self.preview_slider.setRange(0,0)
            self.preview_slider.sliderMoved.connect(self._preview_seek_relative)
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
        self.queue_preview_layout.addWidget(self.preview_pane, 1)

        # ---- RIGHT: scrolling queue lists + fixed action footer ------------------
        # Keep the action buttons outside the scroll area so queue maintenance is
        # always reachable even when Finished contains a very long history.
        right_host = QWidget(splitter)
        right_host.setObjectName("QueueRightHost")
        right_host.setMinimumWidth(520)
        right_host_layout = QVBoxLayout(right_host)
        right_host_layout.setContentsMargins(0,0,0,0)
        right_host_layout.setSpacing(8)

        right_scroll = QScrollArea(right_host)
        right_scroll.setObjectName("QueueJobsScrollArea")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)

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

        # Keep the newest results where they are visible immediately when opening
        # Queue: Finished first, then currently Running, then Pending work.
        self.finished_group = QGroupBox("Finished / failed / Cancelled (0)")
        fg = QVBoxLayout(self.finished_group); fg.addWidget(self.finished_tree)
        right_layout.addWidget(self.finished_group)
        self.running_group = QGroupBox("Running jobs (0)")
        rg = QVBoxLayout(self.running_group); rg.addWidget(self.running_tree)
        right_layout.addWidget(self.running_group)
        self.pending_group = QGroupBox("Pending jobs (0)")
        pg = QVBoxLayout(self.pending_group); pg.addWidget(self.pending_tree)
        right_layout.addWidget(self.pending_group)

        self.running_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.running_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.running_tree,pos,"running"))
        self.pending_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pending_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.pending_tree,pos,"pending"))
        self.finished_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.finished_tree.customContextMenuRequested.connect(lambda pos:self._queue_context_menu(self.finished_tree,pos,"finished"))
        self.finished_tree.itemDoubleClicked.connect(lambda item,col:self._play_job_item(item))

        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)
        right_host_layout.addWidget(right_scroll, 1)

        # Fixed queue action footer. Keep every queue action on one horizontal
        # line so the controls stay compact and immediately visible.
        footer = QWidget(right_host)
        footer.setObjectName("QueueActionFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0,0,4,0)
        footer_layout.setSpacing(6)
        footer_layout.addStretch(1)

        self.cancel_all_btn = QPushButton("Cancel all")
        self.cancel_all_btn.setToolTip("Cancel the current run and remove all pending jobs from the queue.")
        self.cancel_all_btn.clicked.connect(self._cancel_all_jobs)
        footer_layout.addWidget(self.cancel_all_btn)

        self.reset_counter_btn = QPushButton("Reset counter")
        self.reset_counter_btn.setToolTip("Reset the internal queue job counter so the next queued job starts again at Job #1.")
        self.reset_counter_btn.clicked.connect(self._reset_queue_counter)
        footer_layout.addWidget(self.reset_counter_btn)

        self.clear_cancelled_btn = QPushButton("Clear cancelled")
        self.clear_cancelled_btn.setToolTip("Remove cancelled jobs from this queue history only. Files on disk are not deleted.")
        self.clear_cancelled_btn.clicked.connect(self._clear_cancelled_jobs)
        footer_layout.addWidget(self.clear_cancelled_btn)

        self.clear_failed_btn = QPushButton("Clear failed")
        self.clear_failed_btn.setToolTip("Remove failed jobs from this queue history only. Files on disk are not deleted.")
        self.clear_failed_btn.clicked.connect(self._clear_failed_jobs)
        footer_layout.addWidget(self.clear_failed_btn)

        self.clear_finished_btn = QPushButton("Clear finished / failed jobs")
        self.clear_finished_btn.setToolTip("Remove finished and failed jobs from this queue history only. Output files on disk are not deleted.")
        self.clear_finished_btn.clicked.connect(self._clear_finished_jobs)
        footer_layout.addWidget(self.clear_finished_btn)
        right_host_layout.addWidget(footer, 0)

        splitter.addWidget(self.queue_preview_host)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([520, 650])
        layout.addWidget(splitter, 1)
        self.queue_splitter = splitter
        self.queue_jobs_scroll = right_scroll

        index = self.tabs.addTab(page,"Queue")
        self.tabs.tabBar().setTabData(index, "queue")
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
        """Ensure persisted entries have a display number without defeating a manual counter reset."""
        missing=[]
        used=[]
        for i, job in enumerate(self.queue_jobs):
            try:
                n=int(job.get("job_number") or 0)
            except Exception:
                n=0
            if n > 0:
                job["job_number"]=n
                used.append(n)
            else:
                job.pop("job_number", None)
                missing.append((i, job))

        # Migration for genuinely old queue files that had UUIDs only. Existing
        # numbered rows are deliberately left untouched, including duplicates after
        # a user-requested counter reset. Dependencies are keyed by UUID, not number.
        if missing:
            missing.sort(key=lambda pair:(float(pair[1].get("created_at") or 0), pair[0]))
            next_number=max(used, default=0)+1
            for _, job in missing:
                job["job_number"]=next_number
                used.append(next_number)
                next_number += 1
            self._next_job_number_value=max(int(getattr(self,"_next_job_number_value",1) or 1), next_number)
        else:
            self._next_job_number_value=max(1, int(getattr(self,"_next_job_number_value",1) or 1))

        # Cache the dependency's display number in the child too, so a historical
        # source can still be identified after its queue-history row is removed.
        by_id={j.get("id"):j for j in self.queue_jobs if j.get("id")}
        for job in self.queue_jobs:
            dep_id=job.get("continue_from_job_id")
            if dep_id and not job.get("continue_from_job_number"):
                dep=by_id.get(dep_id)
                if dep and dep.get("job_number"):
                    job["continue_from_job_number"]=int(dep["job_number"])

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
        if job.get("job_type") == "timeline_assembly":
            return "Timeline assembly"
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
            # Render terminal jobs newest-first while preserving the existing
            # order semantics of Running/Pending jobs.  queue_jobs itself is not
            # reordered because dependency and pending scheduling logic use it.
            active_jobs = [j for j in self.queue_jobs if j.get("state") not in ("finished", "failed", "cancelled")]
            terminal_jobs = [j for j in self.queue_jobs if j.get("state") in ("finished", "failed", "cancelled")]
            terminal_jobs.sort(key=lambda j: float(j.get("finished_at") or j.get("created_at") or 0), reverse=True)
            for j in active_jobs + terminal_jobs:
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
            if getattr(self, "timeline_widget", None) is not None:
                try:
                    self.timeline_widget.sync_queue_jobs(self.queue_jobs)
                except Exception as exc:
                    self.append_log(f"Timeline status refresh warning: {exc}\n")
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
            if job.get("job_type") == "timeline_assembly":
                try:
                    concat = Path(str(job.get("timeline_assembly_concat") or ""))
                    if concat.is_file(): concat.unlink()
                except Exception:
                    pass
                if getattr(self, "timeline_widget", None) is not None:
                    self.timeline_widget.mark_assembly_failed("Removed from queue")
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

    def _load_preview(self,job,autoplay=False,trim_in=None,trim_out=None,volume=1.0,playback_rate=1.0):
        path=Path(job.get("output",""))
        if getattr(self, "audio_output", None) is not None:
            try:
                self.audio_output.setVolume(max(0.0, min(1.0, float(volume))))
            except Exception:
                pass
        if not path.is_file(): QMessageBox.warning(self,"Preview","Output file is no longer on disk."); return
        self.preview_path=str(path)
        try:
            start_ms=max(0,int(round(float(trim_in or 0.0)*1000.0)))
        except Exception:
            start_ms=0
        try:
            end_ms=int(round(float(trim_out)*1000.0)) if trim_out is not None else None
        except Exception:
            end_ms=None
        if end_ms is not None and end_ms <= start_ms:
            end_ms=None
        self._preview_trim_start_ms=start_ms
        self._preview_trim_end_ms=end_ms
        if self.media_player is not None:
            try:
                rate = max(0.5, min(2.0, float(playback_rate or 1.0)))
                self.media_player.setPlaybackRate(rate)
            except Exception:
                try:
                    self.media_player.setPlaybackRate(1.0)
                except Exception:
                    pass
        if self.media_player is None:
            # External playback cannot enforce an in/out range, so only use it for
            # ordinary untrimmed previews. The embedded preview pane is the normal
            # GrizzlyMax path and supports the exact trim window below.
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))); return
        self.media_player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.preview_label.setText(path.name)
        self.preview_view.reset_view()
        if start_ms > 0:
            self.media_player.setPosition(start_ms)
        if autoplay:
            self.media_player.play()

    def _preview_bounds(self):
        if not self.media_player:
            return 0, 0
        duration=max(0,int(self.media_player.duration()))
        start=max(0,min(int(getattr(self,"_preview_trim_start_ms",0) or 0),duration))
        raw_end=getattr(self,"_preview_trim_end_ms",None)
        end=duration if raw_end is None else max(start,min(int(raw_end),duration))
        return start,end

    def _preview_seek_relative(self,value):
        if not self.media_player:
            return
        start,end=self._preview_bounds()
        target=max(start,min(start+int(value),end))
        self.media_player.setPosition(target)

    def _open_preview_fullscreen(self):
        if not self.media_player or not self.preview_path:
            QMessageBox.information(self, "Preview", "Load a video in the preview first.")
            return
        if not hasattr(self, "_preview_fullscreen_dialog") or self._preview_fullscreen_dialog is None:
            self._preview_fullscreen_dialog = VideoFullscreenDialog(self)
        self._preview_fullscreen_dialog.open_fullscreen()

    def _preview_toggle(self):
        if not self.media_player: return
        if self.media_player.playbackState()==QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            start,end=self._preview_bounds()
            pos=int(self.media_player.position())
            if pos < start or (end > start and pos >= end):
                self.media_player.setPosition(start)
            self.media_player.play()
    def _preview_duration_changed(self,d):
        start,end=self._preview_bounds()
        self.preview_slider.setRange(0,max(0,end-start))
        if self.media_player and int(self.media_player.position()) < start:
            self.media_player.setPosition(start)
        self._preview_position_changed(self.media_player.position() if self.media_player else 0)
    def _preview_position_changed(self,p):
        if not self.media_player: return
        start,end=self._preview_bounds()
        p=int(p)
        # Timeline clip previews honor the non-destructive trim window.  Stop at
        # trim_out rather than continuing through the remainder of the source.
        if end > start and p >= end:
            if self.media_player.playbackState()==QMediaPlayer.PlaybackState.PlayingState:
                if self.preview_repeat.isChecked():
                    self.media_player.setPosition(start); return
                self.media_player.pause()
            if p != end:
                self.media_player.setPosition(end)
            p=end
        relative=max(0,min(p-start,max(0,end-start)))
        if not self.preview_slider.isSliderDown(): self.preview_slider.setValue(relative)
        def fmt(ms):
            sec=max(0,int(ms)//1000); return f"{sec//60:02d}:{sec%60:02d}"
        self.preview_time.setText(f"{fmt(relative)} / {fmt(max(0,end-start))}")
    def _preview_media_status(self,status):
        if self.media_player and status==QMediaPlayer.MediaStatus.EndOfMedia and self.preview_repeat.isChecked():
            start,_end=self._preview_bounds(); self.media_player.setPosition(start); self.media_player.play()

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

    def _cancel_all_jobs(self):
        ans = QMessageBox.question(
            self,
            "Cancel all",
            "This will cancel current run and delete all pending jobs.\nAre you sure ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        # Remove pending work first so the normal process-finished callback cannot
        # immediately advance to another queued job while the active process stops.
        pending_assemblies = [j for j in self.queue_jobs if j.get("state") == "pending" and j.get("job_type") == "timeline_assembly"]
        for assembly_job in pending_assemblies:
            try:
                concat = Path(str(assembly_job.get("timeline_assembly_concat") or ""))
                if concat.is_file(): concat.unlink()
            except Exception:
                pass
        if pending_assemblies and getattr(self, "timeline_widget", None) is not None:
            self.timeline_widget.mark_assembly_failed("Cancelled")
        self.queue_jobs = [j for j in self.queue_jobs if j.get("state") != "pending"]
        self._save_queue_state()
        self._refresh_queue_views()

        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self._stop_running_job("cancel")
        else:
            self.status.setText("Queue ready")

    def _reset_queue_counter(self):
        ans = QMessageBox.warning(
            self,
            "Reset queue counter",
            "This will reset the internal queue counter. Do not use this while a 'continue last result' job is running to avoid mismatch assembly with the previous result(s).",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if ans != QMessageBox.StandardButton.Ok:
            return
        # Counter value 1 means the next newly queued item is displayed as Job #1.
        # Existing history is intentionally kept as-is. Queue dependencies use UUIDs.
        self._next_job_number_value = 1
        self._save_queue_state()
        self.status.setText("Queue counter reset; next job will be Job #1")

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

            # Keep restart history compact: all active/pending work is retained, while
            # only the newest 20 terminal results are remembered. A terminal result
            # referenced by an active/pending job is also retained so dependency chains
            # cannot be broken merely by restarting the GUI.
            terminal_states = {"finished", "failed", "cancelled"}
            active_jobs = [j for j in self.queue_jobs if j.get("state") not in terminal_states]
            required_terminal_ids = set()
            active_ids = {j.get("id") for j in active_jobs if j.get("id")}
            # Walk dependency links transitively because a pending chain can point to a
            # completed job which itself points to an older completed job.
            by_id = {j.get("id"): j for j in self.queue_jobs if j.get("id")}
            frontier = list(active_ids)
            seen = set(frontier)
            while frontier:
                jid = frontier.pop()
                job = by_id.get(jid)
                if not job:
                    continue
                dep_id = job.get("continue_from_job_id")
                if dep_id and dep_id not in seen:
                    seen.add(dep_id)
                    dep = by_id.get(dep_id)
                    if dep:
                        if dep.get("state") in terminal_states:
                            required_terminal_ids.add(dep_id)
                        frontier.append(dep_id)

            terminal_jobs = [j for j in self.queue_jobs if j.get("state") in terminal_states]
            terminal_jobs.sort(key=lambda j: float(j.get("finished_at") or j.get("created_at") or 0), reverse=True)
            keep_terminal_ids = {j.get("id") for j in terminal_jobs[:20] if j.get("id")} | required_terminal_ids
            self.queue_jobs = [
                j for j in self.queue_jobs
                if j.get("state") not in terminal_states or j.get("id") in keep_terminal_ids
            ]
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
            run_args += ["--continue-video", continue_source, "--continue-context-frames", str(int(job.get("continue_context_frames") or 90))]
            if job.get("continue_last_result") and job.get("continue_audio_memory"):
                run_args += ["--continue-audio-memory"]
            if job.get("latent_continuation"):
                run_args += ["--latent-continuation"]
                if job.get("combine_frames_latent"):
                    run_args += ["--combine-frames-latent"]
            if job.get("glue_results"):
                run_args += ["--glue-source", continue_source]
            job["resolved_continue_source"] = continue_source
        job["state"]="running"; job["started_at"]=time.time(); job["finished_at"]=None; job["progress"]=None; job["phase"]="Starting"; job["step_now"]=None; job["step_total"]=job.get("steps"); job["log_tail"]=""; self.current_job_id=job["id"]; self._termination_action=None; self._proc_buffer=""
        # The queue timer has started, so update the always-visible HUD immediately
        # instead of waiting for its next periodic refresh.
        if hasattr(self, "system_hud"):
            self.system_hud.refresh()
        self.proc=QProcess(self); self.proc.setWorkingDirectory(str(ROOT))
        queue_program = str(job.get("queue_program") or "").strip()
        self.proc.setProgram(queue_program if queue_program else str(PYTHON))
        self.proc.setArguments(run_args); self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._process_output); self.proc.finished.connect(self._finished)
        self._start_job_log_file(job)
        self.cancel.setEnabled(True); self.gen.setEnabled(True)
        if job.get("job_type") == "timeline_assembly":
            job["phase"] = "Assembling timeline"
            if getattr(self, "timeline_widget", None) is not None:
                self.timeline_widget.mark_assembly_started(job.get("output"))
            self.status.setText("Queue: assembling timeline…")
            self.append_log(f"\n=== QUEUE START: TIMELINE ASSEMBLY ===\nOutput: {job.get('output')}\n")
        else:
            self.status.setText("Queue: generating…")
            self.append_log(f"\n=== QUEUE START ===\nOutput: {job.get('output')}\n")
        self._save_queue_state(); self._refresh_queue_views(); self.proc.start()

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

    def _music_clip_generation_settings(self):
        """Single source of truth for Music Clip Creator generation jobs.

        Return the live main-GUI configuration. The main GUI already persists this
        state on normal save/generate/close paths, while Music Clip jobs consume the
        exact same in-memory settings immediately.
        """
        return self.settings_dict()

    def _build_music_clip_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            from minimax_music_clip import MiniMaxMusicClipWidget
            self.music_clip_widget = MiniMaxMusicClipWidget(
                page,
                queue_adapter=self._enqueue_music_clip_job,
                settings_provider=self._music_clip_generation_settings,
            )
            lay.addWidget(self.music_clip_widget, 1)
            self.music_clip_tab_index = self.tabs.addTab(page, "Music Clip Creator")
            self.tabs.tabBar().setTabData(self.music_clip_tab_index, "music_clip_creator")
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
            self.tabs.tabBar().setTabData(self.music_clip_tab_index, "music_clip_creator")

    def _build_timeline_tab(self):
        # The timeline owns its project/chunk/CUT model in minimax_timeline.py.
        # It only calls back into this window at the final queue boundary so the
        # existing, proven MiniMax generation/continuation backend stays the one
        # source of truth.
        self.timeline_widget = TimelineTab(
            self,
            settings_provider=self.settings_dict,
            queue_timeline_callback=self._queue_timeline_jobs,
            assemble_timeline_callback=self._assemble_timeline_project,
            preview_result_callback=self._preview_timeline_result,
            open_output_callback=self._open_timeline_output_folder,
            frame_values=EXPERIMENTAL_FRAME_PRESETS,
        )
        self.timeline_tab_index = self.tabs.addTab(self.timeline_widget, "Timeline")
        self.tabs.tabBar().setTabData(self.timeline_tab_index, "timeline")

    def _extract_timeline_first_frame(self, video_path, clip_id, trim_in_seconds=0.0):
        video = Path(str(video_path or ""))
        if not video.is_file():
            return ""
        if not ffmpeg_tools_ready():
            self._ensure_ffmpeg_async()
            QMessageBox.information(
                self,
                "Timeline bridge",
                "FFmpeg is not ready yet. FrameVision has started preparing it; try Generate Selected again when setup finishes.",
            )
            return ""
        bridge_dir = ROOT / "jobs" / "timeline_bridge_frames"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        out = bridge_dir / f"{str(clip_id or uuid.uuid4().hex)}_next_first.png"
        try:
            if out.is_file():
                out.unlink()
        except Exception:
            pass
        try:
            trim_in = max(0.0, float(trim_in_seconds or 0.0))
        except Exception:
            trim_in = 0.0
        cmd = [
            str(ffmpeg_tool_path("ffmpeg.exe")), "-y", "-nostdin", "-loglevel", "error",
            "-i", str(video),
        ]
        # When the destination clip is trimmed, its visible start is no longer
        # source frame 0. Seek to the effective trim-in so bridge replacements
        # target the same first visible frame the timeline/assembler uses.
        if trim_in > 0.0005:
            cmd += ["-ss", f"{trim_in:.6f}"]
        cmd += ["-map", "0:v:0", "-frames:v", "1", str(out)]
        try:
            # This runs on the GUI thread because the queue bridge needs the frame
            # path before it can construct the job. Bound it so a damaged/odd video
            # cannot freeze the application indefinitely.
            cp = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, "Timeline bridge", "Timed out while extracting the next clip's first frame. The regeneration was not queued.")
            return ""
        except Exception as exc:
            QMessageBox.critical(self, "Timeline bridge", f"Could not extract the next clip's first frame:\n{exc}")
            return ""
        if cp.returncode != 0 or not out.is_file():
            detail = (cp.stderr or cp.stdout or "FFmpeg frame extraction failed.").strip().splitlines()
            QMessageBox.critical(self, "Timeline bridge", detail[-1] if detail else "FFmpeg frame extraction failed.")
            return ""
        return str(out)

    def _queue_timeline_jobs(self, specs):
        """Translate timeline generation chunks into normal standalone queue jobs.

        The timeline deliberately does not own a second H3 runner.  Each clip is
        converted back into the same Generation-tab settings that ``generate``
        already knows how to validate and queue.  Continue clips therefore use
        the exact existing Continue Last Result dependency path.
        """
        if not isinstance(specs, list) or not specs:
            QMessageBox.warning(self, "Timeline", "The timeline has no clips to generate.")
            return False

        original = copy.deepcopy(self.settings_dict())
        created = []
        try:
            for pos, spec in enumerate(specs):
                clip_id = str(spec.get("id") or "")
                clip_name = str(spec.get("name") or f"Clip {pos + 1}")
                generation_mode = str(spec.get("generation_mode") or "new")
                if generation_mode in {"source", "loaded"}:
                    # Loaded media clips are existing timeline video, not H3 jobs.
                    # Keep them on the timeline and let neighboring generated clips
                    # use their exact file as a continuation/anchor source when needed.
                    source_video = str(spec.get("start_source_video") or spec.get("output") or "")
                    if not source_video or not Path(source_video).is_file():
                        QMessageBox.warning(self, "Timeline", f"{clip_name} is a loaded video clip, but its video file is missing.")
                        return False
                    continue
                settings = copy.deepcopy(spec.get("settings") or original)
                frames = int(spec.get("frames") or settings.get("frames") or 243)
                settings["frames"] = frames
                settings["experimental_long_duration"] = frames > NORMAL_FRAME_MAX
                settings["prompt"] = str(spec.get("compiled_prompt") or "").strip()
                settings["seed"] = int((spec.get("settings") or {}).get("seed", settings.get("seed", -1)))
                settings["steps"] = int((spec.get("settings") or {}).get("steps", settings.get("steps", 15)))
                settings["scheduler"] = str((spec.get("settings") or {}).get("scheduler", settings.get("scheduler", "beta")))

                # Timeline clips must never inherit the standalone Generation-tab
                # "Glue results" flag.  The timeline owns final assembly itself;
                # carrying Glue into a timeline job makes generate() demand a manual
                # Continue-video/Continue-last source before the timeline bridge has
                # established its dependency chain.  Force it off for every timeline
                # job, including HQ restart and projects saved with an older setting.
                settings["glue_results"] = False

                # HQ restart is deliberately a narrow override: keep the complete
                # per-clip Timeline/Generation settings snapshot and replace only
                # the output resolution/orientation selected by the HQ popup.
                hq_override = spec.get("timeline_hq_override")
                if isinstance(hq_override, dict):
                    hq_aspect = str(hq_override.get("aspect") or "").strip()
                    if hq_aspect in {"16:9", "9:16", "1:1", "21:9"}:
                        settings["aspect"] = hq_aspect
                    hq_resolution = str(hq_override.get("resolution") or "").strip()
                    if hq_resolution in RESOLUTION_PRESETS:
                        settings["resolution"] = hq_resolution
                    hq_wide = str(hq_override.get("widescreen_quality") or "").strip()
                    if hq_wide in WIDESCREEN_21_9_PRESETS:
                        settings["widescreen_quality"] = hq_wide

                # Timeline reference images are per-clip Ref2VA inputs.  They are
                # deliberately separate from the Generation tab's global refs.
                timeline_refs = [
                    str(item.get("path") or "")
                    for item in (spec.get("timeline_reference_images") or spec.get("reference_images") or [])
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                ]
                use_timeline_refs = bool(spec.get("use_reference_images", False))
                if use_timeline_refs:
                    if not timeline_refs:
                        QMessageBox.warning(self, "Timeline Ref2VA", f"{clip_name} has reference mode enabled but no reference images.")
                        return False
                    if len(timeline_refs) > 9:
                        QMessageBox.warning(self, "Timeline Ref2VA", f"{clip_name} has more than 9 reference images.")
                        return False
                    missing_ref = next((x for x in timeline_refs if not Path(x).is_file()), "")
                    if missing_ref:
                        QMessageBox.warning(self, "Timeline Ref2VA", f"{clip_name} cannot find reference image:\n{missing_ref}")
                        return False
                    if str(spec.get("generation_mode") or "new") == "continue":
                        QMessageBox.warning(self, "Timeline Ref2VA", f"{clip_name} uses reference images and cannot continue from the previous block.")
                        return False
                    settings["mode"] = 2  # Ref2VA; hybrid routing is handled by the existing generator when enabled.
                    settings["ref_images"] = timeline_refs
                    settings["ref_videos"] = []
                    settings["ref_audios"] = []
                    settings["first"] = ""
                    settings["last"] = ""
                    settings["continue_video"] = ""
                    settings["continue_last_result"] = False
                    settings["continue_audio_memory"] = False
                    settings["latent_continuation"] = False
                    settings["glue_results"] = False
                else:
                    settings["ref_images"] = []
                    settings["ref_videos"] = []
                    settings["ref_audios"] = []

                is_continue = str(spec.get("generation_mode") or "new") == "continue"
                single_regen = bool(spec.get("timeline_single_regeneration", False))
                edit_mode = str(spec.get("edit_mode") or "") if single_regen else ""
                if single_regen and edit_mode in {"anchor_next", "standalone"}:
                    # These replacement modes explicitly break the dependency on
                    # the previous timeline block. Do not let captured continuation
                    # helpers silently reintroduce that dependency.
                    settings["continue_last_result"] = False
                    settings["glue_results"] = False
                    settings["continue_audio_memory"] = False
                    settings["latent_continuation"] = False
                if is_continue:
                    timeline_index = int(spec.get("timeline_index", pos) or 0)
                    if timeline_index == 0:
                        QMessageBox.warning(self, "Timeline", "The first timeline clip cannot Continue Previous Clip.")
                        return False
                    settings["mode"] = 1  # FL2VA
                    settings["first"] = ""

                    # Timeline continuity always uses the strongest native continuation
                    # path available. Keep the existing .h3latent.pt sidecar format:
                    # when the previous generated clip has one, generate.py feeds that
                    # original H3 latent history to the worker. The worker ALSO uses
                    # the previous MP4's exact final RGB frame as the boundary anchor.
                    # If an imported/old clip has no .h3latent.pt, the existing pixel
                    # history fallback is used automatically.
                    settings["latent_continuation"] = True

                    if single_regen:
                        # A middle-clip replacement must continue from the exact
                        # preserved previous timeline result, not whichever queue
                        # job happens to be newest globally.
                        prev_output = str(spec.get("timeline_previous_output") or "")
                        if not prev_output or not Path(prev_output).is_file():
                            QMessageBox.warning(self, "Timeline", f"{clip_name} has no valid finished previous timeline result to continue from.")
                            return False
                        settings["continue_last_result"] = False
                        settings["continue_video"] = prev_output
                        # latent_continuation is already forced for every Timeline
                        # continuation above. Audio memory remains independent; turning
                        # it on never disables the visual latent + final-frame path.
                    else:
                        # Batch position is NOT timeline position.  When generating
                        # only missing/selected/later clips, the first queued job can
                        # be Clip 3, Clip 8, etc.  In that case continue explicitly
                        # from the already-rendered predecessor instead of treating
                        # this job as "the first timeline item".
                        prev_output = str(spec.get("timeline_previous_output") or "")
                        prev_exists = bool(prev_output and Path(prev_output).is_file())
                        previous_timeline_index = timeline_index - 1
                        batch_timeline_indices = {
                            int(x.get("timeline_index", n) or 0)
                            for n, x in enumerate(specs)
                            if str(x.get("generation_mode") or "new") not in {"source", "loaded"}
                        }
                        predecessor_is_in_this_batch = previous_timeline_index in batch_timeline_indices

                        if predecessor_is_in_this_batch and pos > 0:
                            # The predecessor was queued immediately/earlier in this
                            # same Timeline request, so the normal queue dependency
                            # chain is the strongest continuation source.
                            settings["continue_last_result"] = True
                            settings["continue_video"] = ""
                        elif prev_exists:
                            # Partial batch: predecessor already exists on disk.
                            # Pin continuation to that exact timeline result instead
                            # of whichever unrelated queue job happens to be newest.
                            settings["continue_last_result"] = False
                            settings["continue_video"] = prev_output
                        else:
                            # _generation_specs_for_indices normally converts this
                            # case to a detached "new" spec before we get here.
                            # Keep this final fallback permissive rather than aborting
                            # a whole batch because an old project still says Continue.
                            settings["mode"] = 0
                            settings["continue_last_result"] = False
                            settings["continue_video"] = ""
                            settings["latent_continuation"] = False
                    settings["last"] = ""
                else:
                    # Start a fresh dependency chain. A captured manual source
                    # remains valid, but it must never silently target the last
                    # queue result from an earlier timeline chain.
                    settings["continue_last_result"] = False
                    if bool(spec.get("timeline_single_regeneration", False)) and str(spec.get("edit_mode") or "") == "anchor_next":
                        if use_timeline_refs:
                            QMessageBox.warning(
                                self, "Timeline Ref2VA",
                                f"{clip_name} uses Ref2VA references. The current H3 runner cannot combine Ref2VA with a fixed FL2VA next-frame anchor in one replacement job. Choose Standalone clip for this referenced replacement."
                            )
                            return False
                        # Edit mode: do not use the previous timeline block, but
                        # still constrain the replacement to land on the next
                        # block's first frame. H3's last-frame boundary lives in
                        # FL2VA, so use that path without a continuation video.
                        settings["mode"] = 1
                        settings["continue_video"] = ""
                        settings["continue_last_result"] = False

                if bool(spec.get("match_next_first_frame", False)):
                    self.status.setText(f"Timeline: preparing bridge for {clip_name}…")
                    self.append_log(f"[TIMELINE] Preparing bridge destination frame for {clip_name}.\n")
                    QApplication.processEvents()
                    if not single_regen:
                        QMessageBox.warning(
                            self, "Timeline bridge",
                            f"{clip_name} is set to use the next video's first frame. Use Generate Selected for bridge replacement mode."
                        )
                        return False
                    next_output = str(spec.get("timeline_next_output") or "")
                    if not next_output or not Path(next_output).is_file():
                        QMessageBox.warning(self, "Timeline bridge", f"{clip_name} has no valid next timeline result to use as its end anchor.")
                        return False
                    anchor = self._extract_timeline_first_frame(
                        next_output,
                        clip_id,
                        spec.get("timeline_next_trim_in") or 0.0,
                    )
                    if not anchor:
                        return False
                    settings["last"] = anchor
                    self.append_log(f"[TIMELINE] Bridge destination frame ready: {anchor}\n")

                # Keep long timeline outputs readable without replacing a user's
                # chosen output directory. Existing make_output_path() still
                # guarantees uniqueness when names collide.
                timeline_number = int(spec.get("timeline_index", pos) or 0) + 1
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", clip_name).strip("_") or f"clip_{timeline_number:03d}"
                settings["output_name"] = f"timeline_{timeline_number:03d}_{safe_name}.mp4"
                project_folder = str(
                    getattr(self.timeline_widget, "project", {}).get("project_folder") or ""
                ).strip()
                if project_folder:
                    settings["output_folder"] = project_folder

                self.apply_settings(settings)
                before = len(self.queue_jobs)
                self.generate()
                if len(self.queue_jobs) <= before:
                    # generate() already showed the specific validation dialog.
                    # Stop here so a later Continue clip can never attach to the
                    # wrong queue job after an earlier clip failed to enqueue.
                    return False
                job = self.queue_jobs[-1]
                job["timeline_project_id"] = str(getattr(self.timeline_widget, "project", {}).get("project_id", ""))
                job["timeline_clip_id"] = clip_id
                job["timeline_clip_index"] = pos
                job["timeline_clip_name"] = clip_name
                job["timeline_generation_mode"] = generation_mode
                job["timeline_edit_mode"] = str(spec.get("edit_mode") or "")
                job["timeline_use_reference_images"] = bool(use_timeline_refs)
                job["compiled_prompt"] = str(settings.get("prompt") or "")
                job["timeline_reference_images"] = copy.deepcopy(spec.get("timeline_reference_images") or spec.get("reference_images") or [])
                job["timeline_settings_snapshot"] = copy.deepcopy(settings)
                if bool(spec.get("timeline_alternate_candidate", False)):
                    job["timeline_alternate_candidate"] = True
                    job["timeline_alternate_candidate_id"] = str(spec.get("timeline_alternate_candidate_id") or "")
                created.append((clip_id, job.get("id"), job.get("output", ""), job))

            # Add timeline metadata to persisted queue state after every generated
            # job has been tagged.
            self._save_queue_state()
            self._refresh_queue_views()
            for clip_id, job_id, output, job_info in created:
                if getattr(self, "timeline_widget", None) is not None:
                    self.timeline_widget.mark_queued(clip_id, job_id, output, job_info)
            media_count = sum(1 for spec in specs if str(spec.get("generation_mode") or "") in {"source", "loaded"})
            suffix = f" + {media_count} loaded video clip" if media_count == 1 else (f" + {media_count} loaded video clips" if media_count else "")
            self.status.setText(f"Timeline queued: {len(created)} H3 clip(s){suffix}")
            return True
        finally:
            # Timeline queuing must not leave the ordinary Generation tab changed
            # to whatever happened to be the last clip in the project.
            self.apply_settings(original)
            self.save_last()

    def _timeline_volume_preview_path(self, path: Path, volume_percent: int) -> Path:
        """Return a preview-safe media path for >100% timeline gain.

        QAudioOutput can attenuate normal playback but does not provide reliable
        gain above 100%, so boosted timeline previews get a tiny cached proxy
        that stream-copies video and re-encodes only audio with FFmpeg.
        """
        try:
            volume_percent = max(0, min(150, int(round(float(volume_percent)))))
        except Exception:
            volume_percent = 100
        if volume_percent <= 100:
            return path
        if not ffmpeg_tools_ready():
            self._ensure_ffmpeg_async()
            return path
        try:
            stat = path.stat()
            key = hashlib.sha1(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{volume_percent}".encode("utf-8", "ignore")).hexdigest()[:16]
            cache_dir = ROOT / "jobs" / "timeline_volume_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            proxy = cache_dir / f"{path.stem}_vol{volume_percent}_{key}.mp4"
            if proxy.is_file() and proxy.stat().st_size > 0:
                return proxy
            cmd = [
                str(ffmpeg_tool_path("ffmpeg.exe")), "-y", "-nostdin", "-loglevel", "error",
                "-i", str(path), "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "copy", "-af", f"volume={volume_percent / 100.0:.3f}",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(proxy),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags, timeout=120)
            if completed.returncode == 0 and proxy.is_file() and proxy.stat().st_size > 0:
                return proxy
        except Exception:
            pass
        return path

    def _preview_timeline_result(self, output, job_id=None, trim_in=None, trim_out=None, volume_percent=100, speed_multiplier=1.0):
        path = Path(str(output or ""))
        if not path.is_file():
            QMessageBox.warning(self, "Timeline preview", "The selected timeline output file is no longer on disk.")
            return False
        try:
            volume_percent = max(0, min(150, int(round(float(volume_percent or 100)))))
        except Exception:
            volume_percent = 100
        preview_path = self._timeline_volume_preview_path(path, volume_percent)
        job = self._job_by_id(job_id) if job_id else None
        preview_job = copy.deepcopy(job) if isinstance(job, dict) else {}
        preview_job["output"] = str(preview_path)
        self._load_preview(
            preview_job,
            autoplay=True,
            trim_in=trim_in,
            trim_out=trim_out,
            volume=((volume_percent / 100.0) if volume_percent <= 100 else 1.0),
            playback_rate=max(0.5, min(2.0, float(speed_multiplier or 1.0))),
        )
        return True

    def _open_timeline_output_folder(self, output):
        path = Path(str(output or ""))
        folder = path.parent if path.suffix else path
        if not folder.is_dir():
            QMessageBox.warning(self, "Timeline output", "The output folder is no longer on disk.")
            return False
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))
        return True

    @staticmethod
    def _timeline_safe_name(text):
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("_.")
        return value or "minimax_timeline"

    def _assemble_timeline_project(self, project):
        """Join finished timeline clip outputs into one final MP4 without re-encoding."""
        clips = list((project or {}).get("clips") or [])
        if not clips:
            QMessageBox.warning(self, "Timeline assembly", "The timeline has no clips to assemble.")
            return False
        outputs = []
        resolutions = set()
        jobs_by_id = {str(j.get("id")): j for j in self.queue_jobs if j.get("id")}
        for index, clip in enumerate(clips, 1):
            output = Path(str(clip.get("output") or ""))
            # Assembly means assembly: dependency/status/stale flags belong to
            # generation planning, not file concatenation. If a block has a real
            # video file, use it exactly as it appears in Timeline order.
            if not output.is_file():
                QMessageBox.warning(
                    self,
                    "Timeline assembly",
                    f"Clip {index} has no usable output file:\n\n{output}",
                )
                return False
            outputs.append(output)
            job = jobs_by_id.get(str(clip.get("queue_job_id") or ""))
            if job and job.get("resolution"):
                resolutions.add(str(job.get("resolution")))
            elif clip.get("loaded_resolution"):
                resolutions.add(str(clip.get("loaded_resolution")))
        normalize_mixed_resolution = False
        target_resolution = None
        if len(resolutions) > 1:
            # Mixed resolutions are a warning, never a hard blocker.  The user
            # remains in control: if they choose Assemble anyway, FFmpeg scales
            # the lower-resolution clips to the largest resolution already in
            # this timeline while leaving every source MP4 untouched.
            parsed_resolutions = []
            for value in sorted(resolutions):
                match = re.search(r"(\d+)\s*[x×]\s*(\d+)", str(value), re.IGNORECASE)
                if match:
                    w, h = int(match.group(1)), int(match.group(2))
                    if w > 0 and h > 0:
                        parsed_resolutions.append((w * h, w, h, str(value)))
            if parsed_resolutions:
                _, target_w, target_h, _ = max(parsed_resolutions, key=lambda item: (item[0], item[1], item[2]))
                target_resolution = (target_w, target_h)

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Timeline assembly")
            box.setText("The timeline contains clips with different output resolutions:")
            target_text = (
                f"\n\nThe assembled video will use {target_resolution[0]} × {target_resolution[1]}; "
                "lower-resolution clips will be scaled for the final video."
                if target_resolution else
                "\n\nThe clips will be normalized during assembly."
            )
            box.setInformativeText(
                "\n".join(sorted(resolutions))
                + target_text
                + "\n\nThe original clip files will not be changed."
            )
            assemble_anyway = box.addButton("Assemble anyway", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(assemble_anyway)
            box.exec()
            if box.clickedButton() is not assemble_anyway:
                return False
            normalize_mixed_resolution = True

        if not ffmpeg_tools_ready():
            self._ensure_ffmpeg_async()
            QMessageBox.information(
                self,
                "Timeline assembly",
                "FFmpeg is not ready yet. FrameVision has started preparing it; press Assemble Video again when setup finishes.",
            )
            return False

        # Keep the final assembled video with the Timeline project. Projects made
        # before project-folder support fall back to the normal standalone output.
        project_folder = str((project or {}).get("project_folder") or "").strip()
        out_dir = Path(project_folder) if project_folder else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        base = self._timeline_safe_name((project or {}).get("name") or "MiniMax Timeline") + "_assembled"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = out_dir / f"{base}_{stamp}.mp4"
        suffix = 2
        while final_path.exists():
            final_path = out_dir / f"{base}_{stamp}_{suffix}.mp4"
            suffix += 1

        work_dir = ROOT / "jobs" / "timeline_assembly"
        work_dir.mkdir(parents=True, exist_ok=True)
        concat_path = work_dir / f"concat_{uuid.uuid4().hex}.txt"

        def ffconcat_path(path):
            return str(path.resolve()).replace("\\", "/").replace("'", "\\'")

        soundtrack_mode = str((project or {}).get("soundtrack_mode") or "mix")
        if soundtrack_mode not in {"mix", "clips_only", "soundtrack_only"}:
            soundtrack_mode = "mix"

        soundtrack_specs = []
        soundtrack_items = [] if soundtrack_mode == "clips_only" else list((project or {}).get("soundtrack_clips") or [])
        for audio_index, item in enumerate(soundtrack_items, 1):
            if not isinstance(item, dict):
                continue
            audio_path = Path(str(item.get("path") or ""))
            if not audio_path.is_file():
                QMessageBox.warning(
                    self, "Timeline assembly",
                    f"Soundtrack audio {audio_index} has no usable file:\n\n{audio_path}",
                )
                return False
            try:
                source_duration = max(0.0, float(item.get("source_duration") or 0.0))
            except Exception:
                source_duration = 0.0
            if source_duration <= 0.0:
                source_duration = float(self._probe_clip_duration(str(audio_path), None) or 0.0)
            try:
                trim_in = max(0.0, float(item.get("trim_in") or 0.0))
            except Exception:
                trim_in = 0.0
            raw_out = item.get("trim_out")
            try:
                trim_out = float(raw_out) if raw_out is not None else source_duration
            except Exception:
                trim_out = source_duration
            if source_duration > 0.0:
                trim_in = min(trim_in, source_duration)
                trim_out = min(max(trim_out, trim_in), source_duration)
            try:
                timeline_start = max(0.0, float(item.get("timeline_start") or 0.0))
            except Exception:
                timeline_start = 0.0
            used = max(0.0, trim_out - trim_in)
            raw_end = item.get("timeline_end")
            if raw_end is not None:
                try:
                    used = min(used, max(0.0, float(raw_end) - timeline_start))
                except Exception:
                    pass
            if used <= 0.0005:
                continue
            try:
                volume_percent = max(0, min(150, int(round(float(item.get("volume_percent", 100) or 100)))))
            except Exception:
                volume_percent = 100
            try:
                fade_in_seconds = max(0.0, min(used, float(item.get("fade_in_seconds") or 0.0)))
            except Exception:
                fade_in_seconds = 0.0
            try:
                fade_out_seconds = max(0.0, min(used, float(item.get("fade_out_seconds") or 0.0)))
            except Exception:
                fade_out_seconds = 0.0
            soundtrack_specs.append((item, audio_path, trim_in, used, timeline_start, volume_percent, fade_in_seconds, fade_out_seconds))
        has_soundtrack = bool(soundtrack_specs)
        if soundtrack_mode == "soundtrack_only" and not has_soundtrack:
            QMessageBox.warning(
                self,
                "Timeline assembly",
                "Soundtrack only is selected, but there is no usable soundtrack audio on the timeline."
            )
            return False

        # The timeline trimmer is non-destructive: in/out points live only in
        # project JSON.  Do not rely on concat-demuxer inpoint/outpoint here:
        # those directives are timestamp/keyframe sensitive and can leak source
        # preroll into the assembled result.  When a trim is present we instead
        # open every source as its own input with an exact seek/duration and then
        # concatenate reset-timestamp streams in a filter graph.
        trim_specs = []
        has_trim = False
        has_volume_adjustment = False
        has_audio_fade = False
        has_video_edge_fade = False
        has_speed_adjustment = False
        has_transition = False
        has_loaded_media = any(str(c.get("generation_mode") or "") == "loaded" for c in clips)
        for clip_index, (clip, path) in enumerate(zip(clips, outputs)):
            try:
                user_trim_in = max(0.0, float(clip.get("trim_in") or 0.0))
            except Exception:
                user_trim_in = 0.0

            # MiniMax continuation renders deliberately reuse the previous clip's
            # ending as their first decoded video frame.  A timestamp seek to
            # 1/24 s is normally close, but timestamp rounding around an xfade can
            # still expose that duplicate for a frame.  When the user has NOT
            # already trimmed farther than one frame, keep the input at time zero
            # and remove decoded frame 0 explicitly in the filter graph instead.
            # Audio is trimmed by the matching 1/24 s there as well, preserving A/V
            # sync. A larger manual trim always wins and needs no extra frame drop.
            is_continuation = str(clip.get("generation_mode") or "new") == "continue"
            exact_drop_first_frame = bool(is_continuation and user_trim_in < (1.0 / 24.0 - 0.0005))
            trim_in = 0.0 if exact_drop_first_frame else user_trim_in
            effective_start = (1.0 / 24.0) if exact_drop_first_frame else trim_in

            raw_out = clip.get("trim_out")
            try:
                trim_out = float(raw_out) if raw_out is not None else None
            except Exception:
                trim_out = None
            source_duration = float(self._probe_clip_duration(str(path), clip.get("frames")) or 0.0)
            if source_duration > 0:
                effective_start = min(effective_start, max(0.0, source_duration - 0.001))
                if not exact_drop_first_frame:
                    trim_in = effective_start
                if trim_out is None:
                    trim_out = source_duration
                else:
                    trim_out = min(max(trim_out, effective_start + 0.001), source_duration)
            elif trim_out is None:
                trim_out = effective_start + max(0.001, float(clip.get('frames') or 24) / 24.0)
            effective_duration = max(0.001, float(trim_out) - effective_start)
            trimmed = effective_start > 0.001 or (source_duration > 0 and trim_out < source_duration - 0.02)
            has_trim = has_trim or trimmed
            try:
                volume_percent = max(0, min(150, int(round(float(clip.get("volume_percent", 100) or 100)))))
            except Exception:
                volume_percent = 100
            has_volume_adjustment = has_volume_adjustment or volume_percent != 100
            try:
                speed_multiplier = max(0.5, min(2.0, round(float(clip.get("speed_multiplier", 1.0) or 1.0), 1)))
            except Exception:
                speed_multiplier = 1.0
            final_duration = max(0.001, float(effective_duration) / float(speed_multiplier))
            try:
                audio_fade_in_seconds = max(0.0, min(final_duration, float(clip.get("audio_fade_in_seconds") or 0.0)))
            except Exception:
                audio_fade_in_seconds = 0.0
            try:
                audio_fade_out_seconds = max(0.0, min(final_duration, float(clip.get("audio_fade_out_seconds") or 0.0)))
            except Exception:
                audio_fade_out_seconds = 0.0
            has_audio_fade = has_audio_fade or audio_fade_in_seconds > 0.0005 or audio_fade_out_seconds > 0.0005
            try:
                video_fade_in_seconds = max(0.0, min(final_duration, float(clip.get("video_fade_in_seconds") or 0.0))) if clip_index == 0 else 0.0
            except Exception:
                video_fade_in_seconds = 0.0
            try:
                video_fade_out_seconds = max(0.0, min(final_duration, float(clip.get("video_fade_out_seconds") or 0.0))) if clip_index == len(clips) - 1 else 0.0
            except Exception:
                video_fade_out_seconds = 0.0
            has_video_edge_fade = has_video_edge_fade or video_fade_in_seconds > 0.0005 or video_fade_out_seconds > 0.0005
            has_speed_adjustment = has_speed_adjustment or abs(speed_multiplier - 1.0) > 0.0001
            transition_name = str(clip.get("transition_to_next") or "none").strip().lower()
            has_transition = has_transition or transition_name not in {"", "none"}
            trim_specs.append((clip, path, trim_in, trim_out, effective_duration, volume_percent, speed_multiplier, exact_drop_first_frame, audio_fade_in_seconds, audio_fade_out_seconds, video_fade_in_seconds, video_fade_out_seconds))

        # The concat recipe remains useful for the unchanged fast stream-copy path.
        concat_lines = [f"file '{ffconcat_path(path)}'\n" for path in outputs]
        concat_path.write_text("".join(concat_lines), encoding="utf-8")

        if has_trim or normalize_mixed_resolution or has_volume_adjustment or has_audio_fade or has_video_edge_fade or has_speed_adjustment or has_transition or has_soundtrack or has_loaded_media:
            # Re-encode when Timeline editing requires it, or whenever a ready-made
            # external clip is present. Imported media may use different codecs,
            # frame rates or audio formats, so normalize it to the same 24-fps /
            # stereo assembly pipeline instead of trusting concat stream-copy.
            # Each clip
            # is an independent input so trim-in is exact and cannot be defeated by
            # concat-demuxer timestamps/keyframes.  Sources remain untouched.
            args = ["-y"]
            input_meta = []

            def _input_has_audio(path):
                try:
                    exe = ffmpeg_tool_path("ffprobe.exe")
                    out = subprocess.check_output(
                        [str(exe), "-v", "error", "-select_streams", "a:0",
                         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                        text=True, stderr=subprocess.DEVNULL, timeout=4.0,
                    ).strip()
                    return bool(out)
                except Exception:
                    return False

            for _clip, path, trim_in, trim_out, effective_duration, volume_percent, speed_multiplier, exact_drop_first_frame, audio_fade_in_seconds, audio_fade_out_seconds, video_fade_in_seconds, video_fade_out_seconds in trim_specs:
                # Input-side seek is used for ordinary user trims. For the special
                # continuation seam fix we intentionally start at source time zero
                # and decode one extra frame, because frame 0 itself is discarded
                # exactly by trim=start_frame=1 in the filter graph below.
                if trim_in > 0.001:
                    args += ["-ss", f"{trim_in:.6f}"]
                read_duration = effective_duration + ((1.0 / 24.0) if exact_drop_first_frame else 0.0)
                args += ["-t", f"{read_duration:.6f}", "-i", str(path)]
                input_meta.append((len(input_meta), path, effective_duration, _input_has_audio(path), volume_percent, speed_multiplier, exact_drop_first_frame, audio_fade_in_seconds, audio_fade_out_seconds, video_fade_in_seconds, video_fade_out_seconds))

            filter_parts = []
            synthetic_audio_inputs = []
            video_labels = []
            audio_labels = []
            final_durations = []

            # Normalize each clip into an independent, zero-based 24-fps video
            # stream and 48-kHz stereo audio stream. xfade is strict about frame
            # rate/timebase/geometry compatibility, so normalizing here keeps the
            # transition path deterministic while leaving source MP4s untouched.
            for i, path, effective_duration, has_audio, volume_percent, speed_multiplier, exact_drop_first_frame, audio_fade_in_seconds, audio_fade_out_seconds, video_fade_in_seconds, video_fade_out_seconds in input_meta:
                final_duration = max(0.001, float(effective_duration) / float(speed_multiplier))
                final_durations.append(final_duration)
                vlabel = f"v{i}"
                # For an untrimmed continuation, remove decoded frame 0 itself,
                # rather than approximating it with a timestamp seek. This happens
                # before speed normalization and before xfade sees the clip.
                vprefix = f"[{i}:v:0]"
                if exact_drop_first_frame:
                    vprefix += "trim=start_frame=1,"
                vchain = vprefix + f"setpts=(PTS-STARTPTS)/{speed_multiplier:.3f},fps=24,settb=AVTB"
                if normalize_mixed_resolution and target_resolution:
                    tw, th = target_resolution
                    vchain += (
                        f",scale={tw}:{th}:force_original_aspect_ratio=decrease"
                        f",pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
                    )
                vchain += ",format=yuv420p"
                if video_fade_in_seconds > 0.0005:
                    vchain += f",fade=t=in:st=0:d={video_fade_in_seconds:.6f}:color=black"
                if video_fade_out_seconds > 0.0005:
                    fade_out_start = max(0.0, final_duration - video_fade_out_seconds)
                    vchain += f",fade=t=out:st={fade_out_start:.6f}:d={video_fade_out_seconds:.6f}:color=black"
                filter_parts.append(vchain + f"[{vlabel}]")
                video_labels.append(vlabel)

                if has_audio:
                    alabel = f"a{i}"
                    gain = max(0.0, min(1.5, float(volume_percent) / 100.0))
                    aprefix = f"[{i}:a:0]"
                    if exact_drop_first_frame:
                        aprefix += f"atrim=start={1.0 / 24.0:.9f},"
                    audio_filters = [
                        f"asetpts=PTS-STARTPTS",
                        f"atempo={speed_multiplier:.3f}",
                        f"volume={gain:.3f}",
                    ]
                    # Fade is intentionally applied AFTER the clip gain. So a clip
                    # set to 75% fades from silence up to exactly 75%, never back
                    # to 100%.
                    if audio_fade_in_seconds > 0.0005:
                        audio_filters.append(f"afade=t=in:st=0:d={audio_fade_in_seconds:.6f}")
                    if audio_fade_out_seconds > 0.0005:
                        fade_out_start = max(0.0, final_duration - audio_fade_out_seconds)
                        audio_filters.append(f"afade=t=out:st={fade_out_start:.6f}:d={audio_fade_out_seconds:.6f}")
                    audio_filters.extend([
                        "aresample=48000",
                        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
                    ])
                    filter_parts.append(aprefix + ",".join(audio_filters) + f"[{alabel}]")
                    audio_labels.append(alabel)
                else:
                    synthetic_audio_inputs.append((i, final_duration))
                    audio_labels.append(None)

            # Append silent audio inputs only for source clips without an audio
            # stream. This keeps video/audio chain lengths identical.
            next_input = len(input_meta)
            for clip_i, duration in synthetic_audio_inputs:
                args += ["-f", "lavfi", "-t", f"{duration:.6f}", "-i", "anullsrc=r=48000:cl=stereo"]
                alabel = f"a{clip_i}"
                filter_parts.append(
                    f"[{next_input}:a:0]asetpts=PTS-STARTPTS,"
                    f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[{alabel}]"
                )
                audio_labels[clip_i] = alabel
                next_input += 1

            # Soundtrack inputs are independent audio-only sources. They are
            # trimmed non-destructively, normalized to stereo/48 kHz, delayed to
            # their exact timeline start position, then mixed over the assembled
            # clip audio after all clip-to-clip transitions are resolved.
            soundtrack_labels = []
            for sidx, (_item, audio_path, trim_in, used_duration, timeline_start, volume_percent, fade_in_seconds, fade_out_seconds) in enumerate(soundtrack_specs):
                if trim_in > 0.001:
                    args += ["-ss", f"{trim_in:.6f}"]
                args += ["-t", f"{used_duration:.6f}", "-i", str(audio_path)]
                input_index = next_input
                next_input += 1
                label = f"st{sidx}"
                gain = max(0.0, min(1.5, float(volume_percent) / 100.0))
                delay_ms = max(0, int(round(float(timeline_start) * 1000.0)))
                audio_filters = [
                    "asetpts=PTS-STARTPTS",
                    f"volume={gain:.3f}",
                    "aresample=48000",
                    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
                ]
                if fade_in_seconds > 0.0005:
                    audio_filters.append(f"afade=t=in:st=0:d={fade_in_seconds:.6f}")
                if fade_out_seconds > 0.0005:
                    fade_out_start = max(0.0, used_duration - fade_out_seconds)
                    audio_filters.append(f"afade=t=out:st={fade_out_start:.6f}:d={fade_out_seconds:.6f}")
                audio_filters.append(f"adelay={delay_ms}|{delay_ms}")
                filter_parts.append(f"[{input_index}:a:0]" + ",".join(audio_filters) + f"[{label}]")
                soundtrack_labels.append(label)

            allowed_transitions = {
                "fade", "dissolve", "wipeleft", "wiperight", "wipeup", "wipedown",
                "slideleft", "slideright", "slideup", "slidedown",
                "smoothleft", "smoothright", "smoothup", "smoothdown",
                "circleopen", "circleclose", "pixelize", "radial", "zoomin",
            }

            # Build the timeline from left to right. A transition is stored on the
            # outgoing edge of clip i, so it overlaps clip i with clip i+1 and
            # shortens total timeline duration by exactly that overlap.
            current_v = video_labels[0]
            current_a = audio_labels[0]
            current_duration = final_durations[0]
            for i in range(1, len(input_meta)):
                edge_clip = trim_specs[i - 1][0]
                transition = str(edge_clip.get("transition_to_next") or "none").strip().lower()
                requested = float(edge_clip.get("transition_duration", 0.5) or 0.5)
                requested = max(0.1, min(3.0, round(requested, 1)))
                # xfade must leave at least one frame of valid material on both
                # sides. Clamp pathological saved values instead of failing the
                # entire assembly job.
                max_overlap = max(0.0, min(current_duration, final_durations[i]) - (1.0 / 24.0))
                duration = min(requested, max_overlap) if transition in allowed_transitions else 0.0

                next_v = video_labels[i]
                next_a = audio_labels[i]
                if duration > 0.0005:
                    offset = max(0.0, current_duration - duration)
                    out_v = f"vx{i}"
                    filter_parts.append(
                        f"[{current_v}][{next_v}]xfade=transition={transition}:"
                        f"duration={duration:.6f}:offset={offset:.6f}[{out_v}]"
                    )

                    audio_mode = str(edge_clip.get("transition_audio_mode") or "crossfade").strip().lower()
                    out_a = f"ax{i}"
                    if audio_mode == "hard_cut":
                        # Visual overlap still reduces timeline length. A hard audio
                        # cut therefore drops the overlapped tail of the outgoing
                        # audio and starts the next clip's audio at the cut point.
                        cut_a = f"acut{i}"
                        filter_parts.append(
                            f"[{current_a}]atrim=start=0:end={offset:.6f},asetpts=PTS-STARTPTS[{cut_a}]"
                        )
                        filter_parts.append(f"[{cut_a}][{next_a}]concat=n=2:v=0:a=1[{out_a}]")
                    else:
                        # 'fade_out_in' deliberately uses a steeper exponential
                        # curve, producing a deeper dip around the seam than the
                        # normal linear crossfade while preserving sync/duration.
                        curve = "exp" if audio_mode == "fade_out_in" else "tri"
                        filter_parts.append(
                            f"[{current_a}][{next_a}]acrossfade=d={duration:.6f}:c1={curve}:c2={curve}[{out_a}]"
                        )
                    current_v, current_a = out_v, out_a
                    current_duration = current_duration + final_durations[i] - duration
                else:
                    out_v = f"vc{i}"
                    out_a = f"ac{i}"
                    filter_parts.append(f"[{current_v}][{next_v}]concat=n=2:v=1:a=0[{out_v}]")
                    filter_parts.append(f"[{current_a}][{next_a}]concat=n=2:v=0:a=1[{out_a}]")
                    current_v, current_a = out_v, out_a
                    current_duration += final_durations[i]

            if soundtrack_mode == "mix":
                for sidx, soundtrack_label in enumerate(soundtrack_labels):
                    mixed = f"amix{sidx}"
                    filter_parts.append(
                        f"[{current_a}][{soundtrack_label}]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[{mixed}]"
                    )
                    current_a = mixed
            elif soundtrack_mode == "soundtrack_only":
                if len(soundtrack_labels) == 1:
                    soundtrack_mix = soundtrack_labels[0]
                else:
                    soundtrack_mix = "soundtrack_mix"
                    joined = "".join(f"[{label}]" for label in soundtrack_labels)
                    filter_parts.append(
                        f"{joined}amix=inputs={len(soundtrack_labels)}:duration=longest:dropout_transition=0:normalize=0[{soundtrack_mix}]"
                    )
                soundtrack_only = "soundtrack_only_out"
                # Keep final audio exactly as long as the assembled video, including
                # silence before/after positioned soundtrack clips.
                filter_parts.append(
                    f"[{soundtrack_mix}]apad=pad_dur={current_duration:.6f},atrim=duration={current_duration:.6f},"
                    f"asetpts=PTS-STARTPTS[{soundtrack_only}]"
                )
                current_a = soundtrack_only

            filter_parts.append(f"[{current_v}]setpts=PTS-STARTPTS[vout]")
            filter_parts.append(f"[{current_a}]asetpts=PTS-STARTPTS[aout]")
            args += [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(final_path),
            ]
        else:
            args = [
                "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
                "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                str(final_path),
            ]

        # Assembly is a first-class queue job.  This gives it the same persistent
        # Pending/Running/Finished lifecycle, cancellation controls and finished-
        # result autoplay behavior as normal MiniMax renders.
        active_assembly = next((
            j for j in self.queue_jobs
            if j.get("job_type") == "timeline_assembly"
            and j.get("state") in ("pending", "running")
            and str(j.get("timeline_project_id") or "") == str((project or {}).get("project_id") or (project or {}).get("id") or (project or {}).get("project_folder") or "")
        ), None)
        if active_assembly is not None:
            try:
                if concat_path.is_file():
                    concat_path.unlink()
            except Exception:
                pass
            QMessageBox.information(self, "Timeline assembly", "This timeline already has an assembly job in the queue.")
            return False

        project_key = str((project or {}).get("project_id") or (project or {}).get("id") or (project or {}).get("project_folder") or "")
        if target_resolution:
            queue_resolution = f"{target_resolution[0]} × {target_resolution[1]}"
        elif len(resolutions) == 1:
            queue_resolution = next(iter(resolutions))
        else:
            queue_resolution = "Timeline"
        job = {
            "id": uuid.uuid4().hex,
            "job_number": self._take_next_job_number(),
            "job_type": "timeline_assembly",
            "timeline_project_id": project_key,
            "state": "pending",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed": 0,
            "mode": -1,
            "mode_name": "Timeline assembly",
            "model_label": "Timeline assembly",
            "output": str(final_path),
            "seed": "—",
            "actual_seed": None,
            "resolution": queue_resolution,
            "frames": None,
            "steps": None,
            "prompt": f"Assemble {len(outputs)} timeline clips",
            "args": list(args),
            "queue_program": str(ffmpeg_tool_path("ffmpeg.exe")),
            "progress": None,
            "phase": "Waiting to assemble",
            "error": "",
            "cancel_reason": "",
            "settings": {},
            "log_tail": "",
            "continue_last_result": False,
            "continue_from_job_id": None,
            "continue_from_job_number": None,
            "manual_continue_video": "",
            "glue_results": False,
            "timeline_assembly_concat": str(concat_path),
            "timeline_soundtrack_mode": soundtrack_mode,
            "timeline_assembly_clip_count": len(outputs),
            "timeline_assembly_soundtrack_count": len(soundtrack_specs),
            "timeline_assembly_soundtrack": copy.deepcopy(list((project or {}).get("soundtrack_clips") or [])),
        }
        self.queue_jobs.append(job)
        self._save_queue_state()
        self._refresh_queue_views()
        if getattr(self, "timeline_widget", None) is not None:
            self.timeline_widget.mark_assembly_queued(str(final_path), job.get("id"))
        self.status.setText("Timeline assembly added to queue")
        self.append_log(
            f"\n=== TIMELINE ASSEMBLY QUEUED ===\n"
            f"{self._job_number_text(job)} • Clips: {len(outputs)}\nOutput: {final_path}\n"
        )
        self._start_next_pending()
        return True

    def _current_tab_order(self):
        bar = self.tabs.tabBar()
        order = []
        for index in range(self.tabs.count()):
            key = bar.tabData(index)
            if key:
                order.append(str(key))
        return order

    def _restore_tab_order(self, order):
        if not isinstance(order, (list, tuple)):
            return
        requested = [str(key) for key in order if key]
        if not requested:
            return

        bar = self.tabs.tabBar()
        current_key = bar.tabData(self.tabs.currentIndex())

        target = 0
        for key in requested:
            source = -1
            for index in range(self.tabs.count()):
                if str(bar.tabData(index) or "") == key:
                    source = index
                    break
            if source < 0:
                continue
            if source != target:
                bar.moveTab(source, target)
            target += 1

        # Keep whichever page was selected before restoration selected afterwards.
        if current_key:
            for index in range(self.tabs.count()):
                if str(bar.tabData(index) or "") == str(current_key):
                    self.tabs.setCurrentIndex(index)
                    break

        # Numeric tab indexes are not stable once users reorder the bar; refresh
        # the two legacy cached indexes still used by the main GUI.
        for index in range(self.tabs.count()):
            key = str(bar.tabData(index) or "")
            if key == "prompt_builder":
                self.prompt_builder_tab_index = index
            elif key == "music_clip_creator":
                self.music_clip_tab_index = index
            elif key == "timeline":
                self.timeline_tab_index = index

    def _sync_cached_tab_indexes(self, *args):
        bar = self.tabs.tabBar()
        for index in range(self.tabs.count()):
            key = str(bar.tabData(index) or "")
            if key == "prompt_builder":
                self.prompt_builder_tab_index = index
            elif key == "music_clip_creator":
                self.music_clip_tab_index = index
            elif key == "timeline":
                self.timeline_tab_index = index
        # A move can change the selected page's numeric index without changing
        # its identity. Refresh index-based behavior immediately.
        self._sync_main_generate_button(self.tabs.currentIndex())
        self._sync_global_preview_for_tab(self.tabs.currentIndex())

    def _sync_main_generate_button(self, index=None):
        is_music = (
            getattr(self, "music_clip_widget", None) is not None
            and getattr(self, "music_clip_tab_index", -1) == self.tabs.currentIndex()
        )
        is_timeline = getattr(self, "timeline_tab_index", -1) == self.tabs.currentIndex()
        if hasattr(self, "gen"):
            self.gen.setEnabled(True)
            if is_music:
                self.gen.setText("Create video clip")
                self.gen.setToolTip(
                    "Analyze the selected song, use Whisper lyric timing when enabled, build the H3 shot list, queue every missing clip, then queue final trim/assembly."
                )
            elif is_timeline:
                self.gen.setText("Generate Timeline")
                self.gen.setToolTip("Validate the visible timeline, queue every generation clip in order, and use existing Continue Last Result dependencies for continuation blocks.")
            else:
                self.gen.setText("Generate")
                self.gen.setToolTip("Add the current MiniMax generation job to the queue.")

    def _main_generate_action(self):
        if getattr(self, "timeline_tab_index", -1) == self.tabs.currentIndex():
            if getattr(self, "timeline_widget", None) is not None:
                self.timeline_widget.generate_timeline()
            return
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
        v.addWidget(self.system_hud_toggle)

        # Preview pane is now a permanent application-wide feature.  Keep the
        # compatibility checkbox object because older settings/utility code still
        # references it, but do not expose an option that can turn the pane off.
        self.preview_in_main_toggle = QCheckBox("Show preview pane globally in the app")
        self.preview_in_main_toggle.setChecked(True)
        self.preview_in_main_toggle.setVisible(False)

        self.sage_attention_enabled = QCheckBox("Enable SageAttention")
        self.sage_attention_enabled.setChecked(False)
        self.sage_attention_enabled.setToolTip("This affects transformer attention during sampling only; isolated video/audio VAE workers keep their normal attention path. Default: Off.")
        v.addWidget(self.sage_attention_enabled)

        self.sol_attention_enabled = QCheckBox("Enable Sol Attention")
        self.sol_attention_enabled.setChecked(False)
        self.sol_attention_enabled.setToolTip("Enable SOL Attention in the MiniMax H3 generation backend. Default: Off. If SLA is also enabled, SLA takes priority for that job.")
        v.addWidget(self.sol_attention_enabled)

        self.sla_attention_enabled = QCheckBox("Enable H3 SLA Attention")
        self.sla_attention_enabled.setChecked(False)
        self.sla_attention_enabled.setToolTip(
            "Enable MiniMax H3 block-sparse SLA attention using the tested 0.85 preset and Comfy Kitchen sol_attn kernel. "
            "SLA takes priority over Sol Attention. On a 24 GB GPU, keep enough runtime-free VRAM for the sparse kernel workspace (about 1.5 GB worked in testing)."
        )
        v.addWidget(self.sla_attention_enabled)

        self.spectrum_enabled = QCheckBox("Enable Spectrum feature forecasting")
        self.spectrum_enabled.setChecked(False)
        self.spectrum_enabled.setToolTip("Uses the standalone MiniMax H3 Spectrum forecaster. It learns from real H3 target audio/video hidden states and skips selected later transformer evaluations. Conservative scheduling allows a 4-step Euler/Turbo run to forecast one middle step while keeping native refresh and final steps.")
        v.addWidget(self.spectrum_enabled)

        self.play_result_finished = QCheckBox("Play result when finished")
        self.play_result_finished.setChecked(True)
        v.addWidget(self.play_result_finished)

        # Finished results always use the embedded preview pane.  Retain the old
        # checkbox as an internal compatibility flag, permanently enabled/hidden.
        self.play_result_queue_player = QCheckBox("Use player in Queue (Off = Windows default player)")
        self.play_result_queue_player.setChecked(True)
        self.play_result_queue_player.setVisible(False)

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
        mnote = QLabel("Leave a field empty for automatic model discovery. Explicit overrides can use supported INT4/W4A8 or INT8 ConvRot .safetensors checkpoints. Hybrid mode can use a manually selected hybrid checkpoint or auto-discover one.")
        mnote.setWordWrap(True); mf.addRow(mnote)
        self.check_supported_checkpoints = QPushButton("Check supported checkpoints")
        self.check_supported_checkpoints.setToolTip(
            "Show supported MiniMax H3 diffusion checkpoints from Hugging Face and download one or more with aria2c into models\\minimax_h3\\diffusion_models."
        )
        self.check_supported_checkpoints.clicked.connect(self.open_supported_checkpoint_browser)
        mf.addRow(self.check_supported_checkpoints)
        self.use_hybrid_model = QCheckBox("Use hybrid model")
        self.use_hybrid_model.setChecked(False)
        self.use_hybrid_model.setToolTip("When enabled, use one hybrid MiniMax H3 checkpoint for T2VA/FL2VA and Ref2VA. A selected file/folder is preferred; when blank or stale, the MiniMax model folders are scanned automatically for a compatible hybrid .safetensors file. This setting is remembered after restart.")
        self.hybrid_model = ModelPathRow("Select hybrid MiniMax H3 .safetensors checkpoint")
        mf.addRow(self.use_hybrid_model)
        mf.addRow("Hybrid checkpoint", self.hybrid_model)
        self.fl2va_model = ModelPathRow("Blank = auto-scan FL2VA; supports INT4/W4A8 or INT8 override")
        self.ref2va_model = ModelPathRow("Blank = auto-scan Ref2VA; supports INT4/W4A8 or INT8 override")
        self.text_encoder_model = ModelPathRow("Blank = auto-scan text_encoders")
        self.video_vae_model = ModelPathRow("Blank = auto-scan models\\minimax_h3\\video_vae")
        self.audio_vae_model = ModelPathRow("Blank = auto-scan audio_vae")
        mf.addRow("FL2VA checkpoint", self.fl2va_model)
        mf.addRow("Ref2VA checkpoint", self.ref2va_model)
        mf.addRow("Text encoder", self.text_encoder_model)
        mf.addRow("Video VAE", self.video_vae_model)
        mf.addRow("Audio VAE", self.audio_vae_model)
        self.use_hybrid_model.toggled.connect(self._sync_hybrid_model_ui)
        self._sync_hybrid_model_ui(self.use_hybrid_model.isChecked())
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
        self.vram_runtime_free.setToolTip("Hard safety floor. If CUDA free memory drops below this, the manager asks Comfy to evict model weights. 0.50 GB is the aggressive stock-model starting point; Hybrid mode automatically raises values below 1.50 GB to 1.50 GB.")
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

        self.vram_video_vae_tile_overlap = QSpinBox(); self.vram_video_vae_tile_overlap.setRange(0, 512); self.vram_video_vae_tile_overlap.setSingleStep(32); self.vram_video_vae_tile_overlap.setValue(64); self.vram_video_vae_tile_overlap.setSuffix(" px")
        self.vram_video_vae_tile_overlap.setToolTip("Overlap between MiniMax video-VAE spatial tiles. Default: 64 px. Increase it only if visible tile boundaries appear in the final MP4. Overlap must stay smaller than tile size.")
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
            "PySide6 standalone app by Contrinsan.\n\n"
            "This standalone uses components from ComfyUI (Comfy-Org), such as model-loading / "
            "MiniMax H3 nodes in comfy_extras and MiniMax VAE support to make this work. "
            "ComfyUI is licensed under GPL-3.0.\n\n"
            "INT4 model and text-encoder files used by this install are sourced from Winnougan / "
            "MiniMax-H3-INT4_Convrot_ComfyUI on Hugging Face.\n\n"
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
        index = self.tabs.addTab(self._scroll_page(body), "Settings")
        self.tabs.tabBar().setTabData(index, "settings")

    def _adopt_global_preview_pane(self):
        """Move the one real preview/player into the application-wide splitter."""
        if not all(hasattr(self, name) for name in (
            "preview_pane", "global_preview_host", "global_preview_layout",
            "queue_preview_host", "generation_preview_host",
            "queue_splitter", "generation_splitter"
        )):
            return
        try:
            self.queue_preview_layout.removeWidget(self.preview_pane)
            self.generation_preview_layout.removeWidget(self.preview_pane)
            self.preview_pane.setParent(self.global_preview_host)
            self.global_preview_layout.addWidget(self.preview_pane, 1)

            # The old per-tab preview hosts are retained only as harmless
            # compatibility shells.  Collapse them completely so each tab uses
            # its full right-hand workspace inside the shared outer splitter.
            self.queue_preview_host.hide()
            self.generation_preview_host.hide()
            self.queue_splitter.setSizes([0, max(900, self.queue_splitter.width())])
            self.generation_splitter.setSizes([0, max(900, self.generation_splitter.width())])
            self.preview_pane.show()
        except RuntimeError:
            return

    def _remember_global_preview_width(self, pos, _index):
        """Remember the user's shared splitter position across tab changes."""
        try:
            if self.global_preview_host.isVisible() and int(pos) >= 360:
                self._global_preview_width = int(pos)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _prompt_builder_is_active(self):
        """Return True only when the actual Prompt Builder page is selected."""
        try:
            page = getattr(self, "prompt_builder_page", None)
            if page is not None:
                return self.tabs.currentWidget() is page
            # Compatibility fallback for builds/settings created before the
            # direct page reference existed.
            return self.tabs.currentIndex() == int(getattr(self, "prompt_builder_tab_index", -1))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def _enforce_prompt_builder_full_width(self, *_args):
        """Keep Prompt Builder full width through delayed WebEngine reloads."""
        if not self._prompt_builder_is_active():
            return
        try:
            self.global_preview_host.hide()
            total = max(1000, self.global_preview_splitter.width())
            self.global_preview_splitter.setSizes([0, total])
            self.global_preview_splitter.updateGeometry()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return

    def _sync_global_preview_for_tab(self, index=None):
        """Show the shared preview globally except on the full-width Prompt Builder tab."""
        if not all(hasattr(self, name) for name in (
            "global_preview_splitter", "global_preview_host", "preview_pane", "tabs"
        )):
            return
        try:
            # Keep the old setting permanently enabled for every normal app tab.
            if hasattr(self, "preview_in_main_toggle"):
                self.preview_in_main_toggle.blockSignals(True)
                self.preview_in_main_toggle.setChecked(True)
                self.preview_in_main_toggle.blockSignals(False)

            # Use the selected page itself rather than relying on a cached tab
            # index. This remains correct through delayed builder startup/reload.
            if self._prompt_builder_is_active():
                self._enforce_prompt_builder_full_width()
                self._schedule_layout_refresh()
                return

            self.global_preview_host.show()
            self.preview_pane.show()
            total = max(1000, self.global_preview_splitter.width())
            remembered = int(getattr(self, "_global_preview_width", 520) or 520)
            # The shared preview is intentionally allowed to occupy up to half
            # of the available width.  The tab side can shrink below its size
            # hint because its actual content is scrollable where necessary.
            max_preview = max(360, total // 2)
            preview_w = max(360, min(remembered, max_preview))
            self.global_preview_splitter.setSizes([preview_w, max(1, total - preview_w)])
            self.global_preview_splitter.updateGeometry()
            self._schedule_layout_refresh()
        except (RuntimeError, TypeError, ValueError):
            return

    def _set_preview_in_main_tab(self, enabled, persist=True):
        """Backward-compatible setter for the now global preview preference."""
        enabled = bool(enabled)
        if hasattr(self, "preview_in_main_toggle") and self.preview_in_main_toggle.isChecked() != enabled:
            self.preview_in_main_toggle.blockSignals(True)
            self.preview_in_main_toggle.setChecked(enabled)
            self.preview_in_main_toggle.blockSignals(False)
        self._sync_global_preview_for_tab()

        # Save the new global key while also mirroring the legacy key so an older
        # build can still understand the preference if the user rolls back.
        if persist:
            try:
                PRESET_DIR.mkdir(parents=True, exist_ok=True)
                p = PRESET_DIR / "minimax_h3_gui_last.json"
                d = {}
                if p.is_file():
                    d = json.loads(p.read_text(encoding="utf-8"))
                d["preview_global"] = enabled
                d["preview_global_width"] = int(getattr(self, "_global_preview_width", 520) or 520)
                d["preview_in_main_tab"] = enabled
                p.write_text(json.dumps(d, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _set_system_hud_visible(self, enabled):
        if hasattr(self, "system_hud"):
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
            "Default: 832 × 448 for 16:9. Higher resolutions need substantially more VRAM/RAM and time."
        )
        self.resolved.setToolTip("Exact width × height that will be sent to the backend for the selected aspect ratio.")
        self.frames.setToolTip(
            "Fixed MiniMax H3 frame count at 24 FPS. Normal mode stops at 719 frames = 29.96 seconds. "
            "Enable Experimental long duration for values above 30 seconds, up to 2385 frames = 99.38 seconds. "
            "The maximum frame count that will actually run depends on the available system RAM and GPU VRAM. Default: 124 frames."
        )
        self.experimental_long_duration.setToolTip(
            "Unlock experimental H3 durations above the normal 30-second range. Values follow the native 17k+5 frame grid "
            "up to 2385 frames = 99.38 seconds. The maximum duration that will actually run depends on available system RAM "
            "and GPU VRAM; this switch exposes research values and does not guarantee that every resolution or hardware setup can reach 100 seconds."
        )
        self.steps.setToolTip("Number of diffusion/sampling steps. More steps take longer. Default good quality without a speed LoRA: 20 steps. First-run setting is 15 steps, or 4 when a detected EMA speed LoRA is loaded automatically.")
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
        self.continue_context.setToolTip("How much motion history H3 receives before the separate final-frame boundary anchor. With Use latent continuation OFF this history is rebuilt from decoded source-video frames. With it ON the same history duration is taken directly from the saved H3 latent when available. Values use H3's native 17k+5 temporal grid. 90 history frames (~3.75 s) is the default.")
        self.ref_size.setToolTip(
            "How Ref2VA prepares reference images. 'match' follows the generation/reference sizing behavior; "
            "'max' uses the maximum reference sizing path. Default: match."
        )
        self.ref_images.setToolTip("Ref2VA reference images. MiniMax H3 supports up to 9 images.")
        self.ref_videos.setToolTip("Ref2VA reference videos. MiniMax H3 supports up to 3 videos.")
        self.ref_audios.setToolTip("Ref2VA standalone audio references. MiniMax H3 supports up to 3 audio files.")
        self.lock_source_audio.setToolTip(
            "OFF: all standalone audio files are used as normal Ref2VA references. "
            "ON: Audio slot 1 becomes the exact source/output track while still helping drive the video timing."
        )
        self.cfg.setToolTip("Classifier-free guidance strength used by the sampler. Default: 1.0.")
        self.shift.setToolTip("Video timestep/sigma shift. Validated starting value for this install: 12.")
        self.audio_shift.setToolTip("Audio timestep/sigma shift. Validated starting value for this install: 3.")
        self.sampler.setToolTip("Diffusion sampler algorithm. Default: Euler. Change only when intentionally testing sampler behavior.")
        self.scheduler.setToolTip("Sigma/timestep schedule used by the sampler. First-run default is simple without a speed LoRA; when an EMA speed LoRA is auto-detected the app selects beta.")
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
        self.info_btn.setToolTip("Open the GrizzlyMax feature list in your default web browser.")
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

    def _current_resolution(self):
        aspect = self.aspect.currentText()
        if aspect == "21:9":
            return WIDESCREEN_21_9_PRESETS[self.widescreen_quality.currentText()]
        return RESOLUTION_PRESETS[self.res_class.currentText()][aspect]

    def _sync_resolution(self):
        aspect = self.aspect.currentText()
        is_widescreen = aspect == "21:9"
        self.res_class.setVisible(not is_widescreen)
        self.widescreen_quality.setVisible(is_widescreen)
        w, h = self._current_resolution()
        if not is_widescreen and self.res_class.currentText() == "1280 × 720":
            self.resolved.setText(f"Generation: {w} × {h} (704p)")
        else:
            self.resolved.setText(f"{w} × {h}")

    def _sync_mode(self):
        i = self.mode.currentIndex(); self.fl_group.setVisible(i == 1); self.ref_group.setVisible(i == 2)
        self._sync_continue_video_options()

    def _sync_continue_video_options(self):
        chain = bool(getattr(self, "continue_last_result", None) and self.continue_last_result.isChecked())
        latent = bool(getattr(self, "latent_continuation", None) and self.latent_continuation.isChecked())
        if hasattr(self, "combine_frames_latent"):
            self.combine_frames_latent.setVisible(latent)
            self.combine_frames_latent.setEnabled(latent)
        if hasattr(self, "continue_context"):
            # The context length controls both native latent history and decoded
            # frame history, so keep it available in every continuation mode.
            self.continue_context.setEnabled(True)
        if hasattr(self, "continue_video"):
            self.continue_video.setEnabled(not chain)
            # When Continue last result is active, the queued dependency is the only
            # valid source. Clear any manually selected start video immediately so it
            # cannot linger in the disabled field, be mistaken as still active, or be
            # persisted back into saved settings.
            if chain and getattr(self.continue_video, "edit", None) is not None and self.continue_video.edit.text().strip():
                self.continue_video.edit.clear()
        # A queued-result continuation supplies its own first-frame boundary just like
        # a manually selected Continue Video source. Keep Last frame available as a destination.
        if hasattr(self, "first"):
            self.first.setEnabled(not chain)
        # Sound memory is useful for either continuation source: the latest queued
        # result or a manually selected Continue video. Keep the control visible at
        # all times so its stored ON/OFF state is obvious, and only grey it out when
        # there is currently no continuation source to take audio history from.
        manual_source = bool(
            hasattr(self, "continue_video")
            and getattr(self.continue_video, "edit", None) is not None
            and self.continue_video.edit.text().strip()
        )
        has_audio_memory_source = bool(chain or manual_source)
        if hasattr(self, "continue_audio_memory_row"):
            self.continue_audio_memory_row.setVisible(True)
            self.continue_audio_memory_row.setEnabled(has_audio_memory_source)
        if hasattr(self, "continue_audio_memory"):
            self.continue_audio_memory.setEnabled(has_audio_memory_source)

    def current_output_dir(self) -> Path:
        p = self.output_folder.path()
        return Path(p).expanduser() if p else DEFAULT_OUTPUT_DIR

    def open_feature_list(self):
        path = ROOT / "assets" / "feature_list.html"
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Info",
                f"Feature list not found:\n{path}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

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

    def _queue_job_debug_summary(self, job):
        """Return a verbose human-readable snapshot of the actual queued job state."""
        job = job or {}
        lines = []
        lines.append("=== QUEUE JOB DEBUG SNAPSHOT ===")
        lines.append(f"Job #{job.get('job_number') or '?'} | state={job.get('state') or 'pending'} | mode={job.get('mode_name') or job.get('mode') or '?'}")
        lines.append(f"Output: {job.get('output') or ''}")
        if job.get('timeline_clip_name'):
            lines.append(f"Timeline clip: {job.get('timeline_clip_name')} (index {int(job.get('timeline_clip_index') or 0) + 1})")
        if job.get('timeline_generation_mode'):
            lines.append(f"Timeline generation mode: {job.get('timeline_generation_mode')}")
        if job.get('timeline_edit_mode'):
            lines.append(f"Timeline edit mode: {job.get('timeline_edit_mode')}")
        if job.get('model_label'):
            lines.append(f"Model label: {job.get('model_label')}")
        lines.append("")
        prompt = str(job.get('compiled_prompt') or job.get('prompt') or '').strip()
        lines.append("--- Prompt sent to MiniMax ---")
        lines.append(prompt if prompt else "(empty)")
        lines.append("")
        settings = job.get('timeline_settings_snapshot') or job.get('settings') or {}
        if not isinstance(settings, dict):
            settings = {}
        keys = [
            'mode', 'frames', 'steps', 'cfg', 'shift', 'audio_shift', 'sampler', 'scheduler',
            'seed', 'aspect', 'resolution', 'widescreen_quality', 'output_folder', 'output_name',
            'checkpoint', 'model', 'text_encoder', 'vae', 'audio_vae', 'use_hybrid_ref2va',
        ]
        lines.append("--- Key settings snapshot ---")
        for key in keys:
            if key in settings and settings.get(key) not in (None, '', [], {}):
                lines.append(f"{key}: {settings.get(key)}")
        lines.append(f"continue_last_result: {bool(job.get('continue_last_result'))}")
        lines.append(f"continue_audio_memory: {bool(job.get('continue_audio_memory'))}")
        lines.append(f"latent_continuation: {bool(job.get('latent_continuation'))}")
        lines.append(f"combine_frames_latent: {bool(job.get('combine_frames_latent'))}")
        if job.get('manual_continue_video'):
            lines.append(f"manual_continue_video: {job.get('manual_continue_video')}")
        if job.get('resolved_continue_source'):
            lines.append(f"resolved_continue_source: {job.get('resolved_continue_source')}")
        if job.get('continue_from_job_number'):
            lines.append(f"continue_from_job_number: {job.get('continue_from_job_number')}")
        if job.get('continue_context_frames') not in (None, ''):
            lines.append(f"continue_context_frames: {job.get('continue_context_frames')}")
        lines.append("")
        ref_meta = job.get('timeline_reference_images') or []
        ref_paths = settings.get('ref_images') or []
        if ref_meta or ref_paths:
            lines.append("--- Ref images ---")
            if ref_meta and isinstance(ref_meta, list):
                for idx, item in enumerate(ref_meta, 1):
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"Ref {idx} path: {item.get('path') or ''}")
                    pic = str(item.get('picture_description') or '').strip()
                    subj = str(item.get('subject_description') or '').strip()
                    if pic:
                        lines.append(f"Ref {idx} picture_description: {pic}")
                    if subj:
                        lines.append(f"Ref {idx} subject_description: {subj}")
            else:
                for idx, path in enumerate(ref_paths, 1):
                    lines.append(f"Ref {idx} path: {path}")
            lines.append("")
        lines.append("--- Full settings JSON snapshot ---")
        try:
            lines.append(json.dumps(settings, indent=2, ensure_ascii=False, sort_keys=True))
        except Exception:
            lines.append(str(settings))
        lines.append("")
        return "\n".join(lines)

    def _start_job_log_file(self, job):
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp=time.strftime('%Y%m%d_%H%M%S')
            path=LOG_DIR / f"minimax_h3_{stamp}_{str(job.get('id',''))[:8]}.log"
            job['log_file']=str(path)
            self._active_log_file=path
            header = f"=== QUEUE START ===\nOutput: {job.get('output')}\n\n{self._queue_job_debug_summary(job)}\n\n"
            path.write_text(header, encoding='utf-8')
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

    def _aria2c_ready(self):
        try:
            return ARIA2C.is_file() and ARIA2C.stat().st_size > 0
        except OSError:
            return False

    def _ensure_aria2c_async(self, pending_request=None):
        """Silently provision aria2c.exe from the official Windows x64 release when missing."""
        if self._aria2c_ready():
            return True
        if pending_request is not None:
            self._aria2_pending_checkpoint_request = list(pending_request)
        if self._aria2_bootstrap_running:
            return False

        self._aria2_bootstrap_running = True
        self.append_log(f"[CHECKPOINT] aria2c.exe missing; preparing aria2 {ARIA2_VERSION} downloader.\n")

        def worker():
            archive_path = ARIA2C.parent / ARIA2_ARCHIVE_NAME
            temp_exe = ARIA2C.with_suffix(".exe.tmp")
            try:
                ARIA2C.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(
                    ARIA2_DOWNLOAD_URL,
                    headers={"User-Agent": "MiniMax-H3-Standalone/aria2-bootstrap"},
                )
                digest = hashlib.sha256()
                with urllib.request.urlopen(req, timeout=45) as response, archive_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                actual_hash = digest.hexdigest().lower()
                if actual_hash != ARIA2_ARCHIVE_SHA256.lower():
                    raise RuntimeError(
                        f"aria2 archive checksum mismatch (got {actual_hash}, expected {ARIA2_ARCHIVE_SHA256})"
                    )

                with zipfile.ZipFile(archive_path, "r") as zf:
                    candidates = [name for name in zf.namelist() if Path(name).name.lower() == "aria2c.exe"]
                    if not candidates:
                        raise RuntimeError("aria2c.exe was not found inside the downloaded aria2 archive")
                    with zf.open(candidates[0], "r") as src, temp_exe.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                if not temp_exe.is_file() or temp_exe.stat().st_size <= 0:
                    raise RuntimeError("Extracted aria2c.exe is empty")
                temp_exe.replace(ARIA2C)
                self.aria2_bootstrap_finished.emit(True, str(ARIA2C))
            except Exception as exc:
                try:
                    if temp_exe.exists():
                        temp_exe.unlink()
                except OSError:
                    pass
                self.aria2_bootstrap_finished.emit(False, str(exc))
            finally:
                try:
                    if archive_path.exists():
                        archive_path.unlink()
                except OSError:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        return False

    def _handle_aria2_bootstrap_finished(self, ok, message):
        self._aria2_bootstrap_running = False
        pending = self._aria2_pending_checkpoint_request
        self._aria2_pending_checkpoint_request = None
        if ok:
            self.append_log(f"[CHECKPOINT] aria2c ready: {ARIA2C}\n")
            if pending:
                QTimer.singleShot(0, lambda req=pending: self.start_supported_checkpoint_downloads(req))
            return

        self.append_log(f"[CHECKPOINT] Could not prepare aria2c: {message}\n")
        if pending:
            QMessageBox.warning(
                self,
                "Checkpoint downloader",
                "Could not automatically prepare aria2c.exe.\n\n" + message,
            )

    def open_supported_checkpoint_browser(self):
        # Start provisioning immediately on popup open. Usually this finishes while
        # the Hugging Face checkpoint list is loading, so the downloader is ready
        # before the user can make a selection.
        self._ensure_aria2c_async()
        if self._checkpoint_dialog is not None:
            try:
                self._checkpoint_dialog.close()
            except RuntimeError:
                pass
        self._checkpoint_dialog = SupportedCheckpointDialog(self)
        self._checkpoint_dialog.finished.connect(lambda *_: setattr(self, "_checkpoint_dialog", None))
        self._checkpoint_dialog.show()
        self._checkpoint_dialog.raise_()
        self._checkpoint_dialog.activateWindow()

    def start_supported_checkpoint_downloads(self, requested):
        if self._checkpoint_download_proc is not None:
            QMessageBox.information(
                self,
                "Checkpoint download",
                "A checkpoint download is already running. Let it finish before starting another batch.",
            )
            return False
        if not self._aria2c_ready():
            # The popup-open path normally has this ready already. If the user is
            # unusually fast or the bootstrap is still downloading, keep their
            # selection and start the checkpoint batch automatically when ready.
            self._ensure_aria2c_async(requested)
            return True

        SUPPORTED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        queue = []
        skipped = []
        for info in requested:
            if not isinstance(info, dict):
                continue
            name = Path(str(info.get("name") or "")).name
            if not name.lower().endswith(".safetensors"):
                continue
            target = SUPPORTED_CHECKPOINT_DIR / name
            control = Path(str(target) + ".aria2")
            expected = int(info.get("size") or 0)
            installed = False
            if target.exists() and not control.exists():
                try:
                    installed = not expected or target.stat().st_size >= expected
                except OSError:
                    installed = False
            if installed:
                skipped.append(name)
                continue
            entry = dict(info)
            entry["name"] = name
            entry["target"] = str(target)
            queue.append(entry)

        if not queue:
            QMessageBox.information(
                self,
                "Supported checkpoints",
                "The selected checkpoint(s) are already installed. Nothing needs to be downloaded.",
            )
            return False

        self._checkpoint_download_queue = queue
        self._checkpoint_download_total_bytes = sum(max(0, int(x.get("size") or 0)) for x in queue)
        self._checkpoint_download_completed_bytes = 0
        self._checkpoint_download_current = None
        self._checkpoint_download_current_start_size = 0
        self._checkpoint_download_output = ""
        self.checkpoint_download_progress.setValue(0)
        self.checkpoint_download_strip.show()
        skipped_text = f" · {len(skipped)} already installed" if skipped else ""
        self.checkpoint_download_label.setText(f"Checkpoint downloads queued: {len(queue)}{skipped_text}")
        self.append_log(f"[CHECKPOINT] Queued {len(queue)} supported checkpoint download(s) with aria2c.{skipped_text}\n")
        self._checkpoint_progress_timer.start()
        self._start_next_supported_checkpoint_download()
        return True

    def _start_next_supported_checkpoint_download(self):
        if self._closing:
            return
        if not self._checkpoint_download_queue:
            self._checkpoint_download_proc = None
            self._checkpoint_download_current = None
            self._checkpoint_progress_timer.stop()
            self.checkpoint_download_progress.setValue(1000)
            self.checkpoint_download_progress.setFormat("100%")
            self.checkpoint_download_label.setText("Supported checkpoint download complete")
            self.status.setText("Checkpoint download complete")
            self.append_log(f"[CHECKPOINT] Downloads complete: {SUPPORTED_CHECKPOINT_DIR}\n")
            QTimer.singleShot(8000, self._hide_completed_checkpoint_download_strip)
            return

        info = self._checkpoint_download_queue.pop(0)
        target = Path(info["target"])
        expected = max(0, int(info.get("size") or 0))
        try:
            existing = target.stat().st_size if target.exists() else 0
        except OSError:
            existing = 0
        self._checkpoint_download_current = info
        self._checkpoint_download_current_start_size = existing
        self._checkpoint_download_output = ""

        url = f"{SUPPORTED_CHECKPOINT_RESOLVE}/{urllib.parse.quote(info['name'])}?download=true"
        args = [
            "--continue=true",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=16M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--console-log-level=notice",
            "--summary-interval=1",
            f"--dir={SUPPORTED_CHECKPOINT_DIR}",
            f"--out={info['name']}",
            url,
        ]
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._checkpoint_download_read_output)
        proc.finished.connect(self._checkpoint_download_finished)
        proc.errorOccurred.connect(self._checkpoint_download_process_error)
        self._checkpoint_download_proc = proc
        self.checkpoint_download_label.setText(f"Downloading {info['name']}")
        self.append_log(
            f"[CHECKPOINT] aria2c: {info['name']} ({_human_bytes(expected)}) -> {SUPPORTED_CHECKPOINT_DIR}\n"
        )
        proc.start(str(ARIA2C), args)

    def _checkpoint_download_read_output(self):
        proc = self._checkpoint_download_proc
        if proc is None:
            return
        try:
            chunk = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            return
        self._checkpoint_download_output = (self._checkpoint_download_output + chunk)[-16000:]

    def _refresh_checkpoint_download_progress(self):
        info = self._checkpoint_download_current
        if not info:
            return
        target = Path(info["target"])
        expected = max(0, int(info.get("size") or 0))
        try:
            current_size = target.stat().st_size if target.exists() else 0
        except OSError:
            current_size = 0
        current_size = min(current_size, expected) if expected else current_size

        if self._checkpoint_download_total_bytes > 0:
            done = self._checkpoint_download_completed_bytes + current_size
            ratio = max(0.0, min(1.0, done / self._checkpoint_download_total_bytes))
            self.checkpoint_download_progress.setValue(int(ratio * 1000))
            self.checkpoint_download_progress.setFormat(f"{ratio * 100:.1f}%")
        elif expected > 0:
            ratio = max(0.0, min(1.0, current_size / expected))
            self.checkpoint_download_progress.setValue(int(ratio * 1000))
            self.checkpoint_download_progress.setFormat(f"{ratio * 100:.1f}%")
        else:
            self.checkpoint_download_progress.setRange(0, 0)
            self.checkpoint_download_progress.setFormat("Downloading…")

        count_done = self._checkpoint_download_completed_bytes
        count_total = self._checkpoint_download_total_bytes
        suffix = f" · {_human_bytes(current_size)} / {_human_bytes(expected)}" if expected else f" · {_human_bytes(current_size)}"
        if count_total:
            suffix += f" · batch {_human_bytes(min(count_done + current_size, count_total))} / {_human_bytes(count_total)}"
        self.checkpoint_download_label.setText(f"Downloading {info['name']}{suffix}")

    def _checkpoint_download_finished(self, exit_code, exit_status):
        proc = self._checkpoint_download_proc
        if proc is not None:
            self._checkpoint_download_read_output()
        info = self._checkpoint_download_current
        self._checkpoint_download_proc = None
        if not info:
            return

        target = Path(info["target"])
        expected = max(0, int(info.get("size") or 0))
        try:
            actual = target.stat().st_size if target.exists() else 0
        except OSError:
            actual = 0
        control = Path(str(target) + ".aria2")
        complete = exit_code == 0 and target.exists() and not control.exists() and (not expected or actual >= expected)
        if complete:
            self._checkpoint_download_completed_bytes += expected if expected else actual
            self.append_log(f"[CHECKPOINT] Installed: {target.name}\n")
            self._checkpoint_download_current = None
            QTimer.singleShot(0, self._start_next_supported_checkpoint_download)
            return

        self._checkpoint_progress_timer.stop()
        self._checkpoint_download_current = None
        self._checkpoint_download_queue = []
        self.checkpoint_download_progress.setRange(0, 1000)
        self.checkpoint_download_progress.setFormat("Failed")
        self.checkpoint_download_label.setText(f"Checkpoint download failed: {info['name']}")
        tail = self._checkpoint_download_output.strip()[-4000:] or f"aria2c exit code {exit_code}"
        self.append_log(f"[CHECKPOINT] Download failed: {info['name']} (exit {exit_code})\n{tail}\n")
        QMessageBox.warning(
            self,
            "Checkpoint download failed",
            f"aria2c could not finish {info['name']}.\n\nThe partial download is kept so a later retry can resume it.\n\n{tail}",
        )

    def _checkpoint_download_process_error(self, error):
        if self._closing:
            return
        self.append_log(f"[CHECKPOINT] aria2c process error: {error}\n")
        if error != QProcess.ProcessError.FailedToStart:
            return
        info = self._checkpoint_download_current or {}
        self._checkpoint_download_proc = None
        self._checkpoint_download_current = None
        self._checkpoint_download_queue = []
        self._checkpoint_progress_timer.stop()
        self.checkpoint_download_progress.setRange(0, 1000)
        self.checkpoint_download_progress.setFormat("Failed")
        name = str(info.get("name") or "checkpoint")
        self.checkpoint_download_label.setText(f"Checkpoint download failed to start: {name}")
        QMessageBox.critical(
            self,
            "aria2c failed to start",
            f"Could not start aria2c.exe from:\n\n{ARIA2C}",
        )

    def _hide_completed_checkpoint_download_strip(self):
        if self._checkpoint_download_proc is None and not self._checkpoint_download_queue and self._checkpoint_download_current is None:
            self.checkpoint_download_strip.hide()
            self.checkpoint_download_progress.setRange(0, 1000)
            self.checkpoint_download_progress.setValue(0)
            self.checkpoint_download_progress.setFormat("%p%")

    def _sync_hybrid_model_ui(self, enabled):
        enabled = bool(enabled)
        self.hybrid_model.setEnabled(enabled)
        self.fl2va_model.setEnabled(not enabled)
        self.ref2va_model.setEnabled(not enabled)

        # Hybrid checkpoints can need a little more transient CUDA workspace than
        # the stock W4A8 models.  Do not let the aggressive 0.50 GB default carry
        # over when Hybrid mode is enabled.  This only raises low values; a user
        # who already selected a larger safety floor keeps that larger value.
        if enabled and hasattr(self, "vram_runtime_free"):
            if float(self.vram_runtime_free.value()) < 1.50:
                self.vram_runtime_free.setValue(1.50)

    def settings_dict(self):
        return {
            "mode": self.mode.currentIndex(), "aspect": self.aspect.currentText(), "resolution": self.res_class.currentText(), "widescreen_quality": self.widescreen_quality.currentText(), "frames": self._frame_count(), "experimental_long_duration": self.experimental_long_duration.isChecked(),
            "steps": self.steps.value(), "seed": self.seed.value(), "prompt": self.prompt.toPlainText(), "first": self.first.path(), "last": self.last.path(),
            "continue_video": "" if self.continue_last_result.isChecked() else self.continue_video.path(), "continue_context_frames": int(self.continue_context.currentData() or 90),
            "glue_results": self.glue_results.isChecked(), "continue_last_result": self.continue_last_result.isChecked(),
            "continue_audio_memory": self.continue_audio_memory.isChecked(), "latent_continuation": self.latent_continuation.isChecked(),
            "combine_frames_latent": self.combine_frames_latent.isChecked(),
            "ref_size": self.ref_size.currentText(), "ref_images": self.ref_images.paths(), "ref_videos": self.ref_videos.paths(), "ref_audios": self.ref_audios.paths(), "lock_source_audio": self.lock_source_audio.isChecked(),
            "cfg": self.cfg.value(), "shift": self.shift.value(), "audio_shift": self.audio_shift.value(), "sampler": self.sampler.currentText(), "scheduler": self.scheduler.currentText(),
            "output_folder": self.output_folder.path(), "output_name": self.output_name.text().strip(), "extended_logging": self.extended_logging.isChecked(), "tile_debugging": self.tile_debugging.isChecked(),
            "system_hud": self.system_hud_toggle.isChecked(),
            "preview_global": True,
            "preview_global_width": int(getattr(self, "_global_preview_width", 520) or 520),
            "preview_in_main_tab": True,
            "auto_update_enabled": self.auto_update_enabled.isChecked(),
            "font_size_pt": int(self.font_size_slider.value()) if hasattr(self, "font_size_slider") else int(self._font_size_pt),
            "play_result_finished": self.play_result_finished.isChecked(),
            "play_result_queue_player": True,
            "spectrum_enabled": self.spectrum_enabled.isChecked(),
            "sage_attention_enabled": self.sage_attention_enabled.isChecked(),
            "sol_attention_enabled": self.sol_attention_enabled.isChecked(),
            "sla_attention_enabled": self.sla_attention_enabled.isChecked(),
            "vram_manager_enabled": self.vram_manager_enabled.isChecked(), "vram_manager_auto_bypass": self.vram_manager_auto_bypass.isChecked(), "vram_residency_engine": self.vram_residency_engine.currentData(), "vram_runtime_free_gb": self.vram_runtime_free.value(),
            "vram_text_headroom_gb": self.vram_text_headroom.value(), "vram_diffusion_headroom_gb": self.vram_diffusion_headroom.value(),
            "vram_offload_chunk_mb": self.vram_offload_chunk.value(), "vram_max_resident_weights_gb": self.vram_max_weights.value(),
            "vram_residency_fill": self.vram_residency_fill.isChecked(), "vram_residency_target_free_gb": self.vram_residency_target_free.value(),
            "vram_residency_warmup_blocks": self.vram_residency_warmup.value(), "vram_residency_refill_interval": self.vram_residency_refill_interval.value(),
            "vram_block_check_interval": self.vram_block_interval.value(), "vram_async_streams": self.vram_async_streams.value(),
            "vram_video_vae_reserve_gb": self.vram_video_vae_reserve.value(), "vram_audio_vae_reserve_gb": self.vram_audio_vae_reserve.value(),
            "vram_video_vae_tile_size": self.vram_video_vae_tile_size.value(), "vram_video_vae_tile_overlap": self.vram_video_vae_tile_overlap.value(),
            "use_hybrid_model": self.use_hybrid_model.isChecked(), "hybrid_model": self.hybrid_model.path(),
            "fl2va_model": self.fl2va_model.path(), "ref2va_model": self.ref2va_model.path(), "text_encoder_model": self.text_encoder_model.path(),
            "video_vae_model": self.video_vae_model.path(), "audio_vae_model": self.audio_vae_model.path(),
            "loras": [{"path": row.path(), "strength": strength.value()} for row, strength in self.lora_rows],
        }

    def apply_settings(self, d):
        try:
            self.mode.setCurrentIndex(int(d.get("mode", 0))); self.aspect.setCurrentText(d.get("aspect", "16:9"))
            saved_res = str(d.get("resolution", DEFAULT_RESOLUTION))
            saved_res = {"Low / test": "576 × 320", "480p": "832 × 448", "832 × 480": "832 × 448", "768p": "1344 × 768", "1080p": "1920 × 1088"}.get(saved_res, saved_res)
            if saved_res in RESOLUTION_PRESETS: self.res_class.setCurrentText(saved_res)
            else: self.res_class.setCurrentText(DEFAULT_RESOLUTION)
            saved_wide = str(d.get("widescreen_quality", "Medium — 1344 × 576"))
            if saved_wide in WIDESCREEN_21_9_PRESETS: self.widescreen_quality.setCurrentText(saved_wide)
            else: self.widescreen_quality.setCurrentText("Medium — 1344 × 576")
            self.experimental_long_duration.setChecked(bool(d.get("experimental_long_duration", False)))
            self._sync_long_duration_mode(self.experimental_long_duration.isChecked())
            self._set_frame_count(d.get("frames", 362))
            self.steps.setValue(int(d.get("steps", 15))); self.seed.setValue(int(d.get("seed", -1))); self.prompt.setPlainText(d.get("prompt", "")); self.first.edit.setText(d.get("first", "")); self.last.edit.setText(d.get("last", ""))
            continue_last_setting = bool(d.get("continue_last_result", False))
            # Set the dependency toggle BEFORE restoring a manual Continue-video path.
            # If the GUI was previously left on "Continue last result", its change
            # handler deliberately clears the manual path. Restoring the path first
            # therefore made Timeline regenerate jobs lose their previous-clip source
            # and fail FL2VA validation with "Visual input required". Establish the
            # toggle state first, then restore the mutually exclusive manual source.
            self.continue_last_result.setChecked(continue_last_setting)
            self.continue_video.edit.setText("" if continue_last_setting else d.get("continue_video", ""))
            ctx=int(d.get("continue_context_frames",90)); idx=self.continue_context.findData(ctx); self.continue_context.setCurrentIndex(idx if idx >= 0 else 4)
            self.glue_results.setChecked(bool(d.get("glue_results", False))); self.continue_audio_memory.setChecked(bool(d.get("continue_audio_memory", False))); self.latent_continuation.setChecked(bool(d.get("latent_continuation", False))); self.combine_frames_latent.setChecked(bool(d.get("combine_frames_latent", False))); self._sync_continue_video_options()
            self.ref_size.setCurrentText(d.get("ref_size", "match")); self.ref_images.set_paths(d.get("ref_images", [])); self.ref_videos.set_paths(d.get("ref_videos", [])); self.ref_audios.set_paths(d.get("ref_audios", [])); self.lock_source_audio.setChecked(bool(d.get("lock_source_audio", False)))
            self.cfg.setValue(float(d.get("cfg", 1.0))); self.shift.setValue(float(d.get("shift", 12))); self.audio_shift.setValue(float(d.get("audio_shift", 3))); self.sampler.setCurrentText(d.get("sampler", "euler")); self.scheduler.setCurrentText(d.get("scheduler", "simple"))
            # Backward compatibility with the first GUI patch's single output field.
            old_output = d.get("output", "")
            if old_output and not d.get("output_folder") and not d.get("output_name"):
                op = Path(old_output); self.output_folder.edit.setText(str(op.parent)); self.output_name.setText(op.name)
            else:
                self.output_folder.edit.setText(d.get("output_folder", "")); self.output_name.setText(d.get("output_name", ""))
            self.extended_logging.setChecked(bool(d.get("extended_logging", False)))
            self.tile_debugging.setChecked(bool(d.get("tile_debugging", False)))
            self.system_hud_toggle.setChecked(bool(d.get("system_hud", True)))
            self._global_preview_width = max(360, int(d.get("preview_global_width", 520) or 520))
            self.preview_in_main_toggle.setChecked(True)
            self.auto_update_enabled.setChecked(bool(d.get("auto_update_enabled", True)))
            saved_font = max(5, min(15, int(d.get("font_size_pt", 10))))
            self.font_size_slider.blockSignals(True)
            self.font_size_slider.setValue(saved_font)
            self.font_size_slider.blockSignals(False)
            self._font_size_pt = saved_font
            self._update_font_size_label(saved_font)
            self._apply_style()
            self._restore_tab_order(d.get("tab_order", []))
            self.play_result_finished.setChecked(bool(d.get("play_result_finished", False)))
            self.play_result_queue_player.setChecked(True)
            self.play_result_queue_player.setVisible(False)
            self.spectrum_enabled.setChecked(bool(d.get("spectrum_enabled", False)))
            self.sage_attention_enabled.setChecked(bool(d.get("sage_attention_enabled", False)))
            self.sol_attention_enabled.setChecked(bool(d.get("sol_attention_enabled", False)))
            self.sla_attention_enabled.setChecked(bool(d.get("sla_attention_enabled", False)))
            self._set_system_hud_visible(self.system_hud_toggle.isChecked())
            self._set_preview_in_main_tab(True)
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
            self.vram_video_vae_tile_overlap.setValue(int(d.get("vram_video_vae_tile_overlap", 64)))
            self.use_hybrid_model.setChecked(bool(d.get("use_hybrid_model", False))); self.hybrid_model.edit.setText(d.get("hybrid_model", ""))
            self.fl2va_model.edit.setText(d.get("fl2va_model", "")); self.ref2va_model.edit.setText(d.get("ref2va_model", "")); self.text_encoder_model.edit.setText(d.get("text_encoder_model", "")); self.video_vae_model.edit.setText(d.get("video_vae_model", "")); self.audio_vae_model.edit.setText(d.get("audio_vae_model", ""))
            saved_loras = d.get("loras", []) or []
            for i, (row, strength) in enumerate(self.lora_rows):
                item = saved_loras[i] if i < len(saved_loras) and isinstance(saved_loras[i], dict) else {}
                row.edit.setText(str(item.get("path", ""))); strength.setValue(float(item.get("strength", 1.0)))
        except Exception as e:
            self.append_log(f"Preset warning: {e}\n")
        self._sync_hybrid_model_ui(self.use_hybrid_model.isChecked())
        self._sync_resolution(); self._sync_mode()

    @staticmethod
    def _looks_like_speed_ema_lora(path: Path) -> bool:
        name = path.name.lower()
        if path.suffix.lower() != ".safetensors" or "ema" not in name:
            return False
        speed_markers = ("turbo", "4step", "4_step", "4-step", "speed", "fast", "distill", "acceler")
        return any(marker in name for marker in speed_markers)

    def _find_first_speed_ema_lora(self):
        if not DEFAULT_LORA_DIR.is_dir():
            return None
        try:
            files = sorted(DEFAULT_LORA_DIR.rglob("*.safetensors"), key=lambda x: str(x).lower())
        except OSError:
            return None
        return next((p for p in files if self._looks_like_speed_ema_lora(p)), None)

    def _apply_first_run_defaults(self):
        """Apply defaults only when this install has no saved GUI state yet."""
        if (PRESET_DIR / "minimax_h3_gui_last.json").is_file():
            return
        self.res_class.setCurrentText(DEFAULT_RESOLUTION)
        self.aspect.setCurrentText("16:9")
        self._set_frame_count(124)
        self.vram_video_vae_tile_overlap.setValue(64)
        self.preview_in_main_toggle.setChecked(True)
        self._set_preview_in_main_tab(True, persist=False)
        self.play_result_finished.setChecked(True)
        self.play_result_queue_player.setChecked(True)
        self.play_result_queue_player.setVisible(False)
        self.steps.setValue(15)
        self.scheduler.setCurrentText("simple")

        speed_lora = self._find_first_speed_ema_lora()
        if speed_lora is not None and self.lora_rows:
            row, strength = self.lora_rows[0]
            row.edit.setText(str(speed_lora))
            strength.setValue(1.0)
            self.steps.setValue(4)
            self.scheduler.setCurrentText("beta")

    def _resolve_hybrid_model_path(self, populate=False):
        """Resolve a hybrid checkpoint from an explicit file/folder or auto-discover it."""
        raw = self.hybrid_model.path().strip()
        roots = []
        if raw:
            requested = Path(raw).expanduser()
            if requested.is_file() and requested.suffix.lower() == ".safetensors":
                return requested.resolve()
            if requested.is_dir():
                roots.append(requested)

        roots.extend([
            ROOT / "models" / "minimax_h3" / "diffusion_models",
            ROOT / "models" / "diffusion_models",
            ROOT / "models" / "minimax_h3",
            ROOT / "models",
        ])
        seen = set()
        candidates = []
        for root in roots:
            try:
                key = str(root.resolve()).lower()
            except OSError:
                key = str(root).lower()
            if key in seen or not root.is_dir():
                continue
            seen.add(key)
            try:
                for candidate in root.rglob("*.safetensors"):
                    name = candidate.name.lower()
                    if "hybrid" in name and ("minimax" in name or "h3" in name or "ref2va" in name):
                        candidates.append(candidate)
            except OSError:
                continue

        if not candidates:
            return None

        def score(path: Path):
            name = path.name.lower()
            return (
                0 if "ref2va_hybrid" in name else 1,
                0 if "b20-49" in name or "b20_49" in name else 1,
                0 if "prun" in name else 1,
                len(str(path)),
                str(path).lower(),
            )

        found = sorted(set(candidates), key=score)[0].resolve()
        if populate:
            self.hybrid_model.edit.setText(str(found))
        return found

    def save_last(self):
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        data = self.settings_dict()
        data["tab_order"] = self._current_tab_order()
        (PRESET_DIR / "minimax_h3_gui_last.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
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
        PRESET_DIR.mkdir(parents=True, exist_ok=True); p, _ = QFileDialog.getOpenFileName(self, "Load preset", _dialog_start_dir("presets", fallback=PRESET_DIR), "JSON (*.json)")
        if p:
            _remember_dialog_folder("presets", p)
            self.apply_settings(json.loads(Path(p).read_text(encoding="utf-8")))
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

    def model_override_args(self, mode=None):
        args = []
        hybrid_path = self._resolve_hybrid_model_path(populate=True) if self.use_hybrid_model.isChecked() else None
        hybrid = str(hybrid_path) if hybrid_path else ""
        if hybrid:
            if mode == 2:
                args += ["--ref2va-checkpoint", hybrid]
            elif mode in (0, 1):
                args += ["--fl2va-checkpoint", hybrid]
            else:
                args += ["--fl2va-checkpoint", hybrid, "--ref2va-checkpoint", hybrid]
        else:
            for flag, value in (("--fl2va-checkpoint", self.fl2va_model.path()), ("--ref2va-checkpoint", self.ref2va_model.path())):
                if value: args += [flag, value]
        for flag, value in (("--text-encoder", self.text_encoder_model.path()), ("--video-vae", self.video_vae_model.path()), ("--audio-vae", self.audio_vae_model.path())):
            if value: args += [flag, value]
        return args

    @staticmethod
    def _update_rel_allowed(rel: str) -> bool:
        rel = str(rel).replace("\\", "/").lstrip("/")
        if not rel or rel.endswith("/"):
            return False
        parts = rel.split("/")
        if not parts or parts[0].lower() in {x.lower() for x in APP_UPDATE_EXCLUDED_TOP}:
            return False
        low = rel.lower()
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

    def _write_external_update_launcher(self, payload):
        """Stage a detached Windows updater that replaces files only after this GUI exits."""
        source_root = Path(payload.get("source_root") or "")
        temp_dir = Path(payload.get("temp_dir") or "")
        changed = [str(rel).replace("\\", "/") for rel in (payload.get("changed") or []) if self._update_rel_allowed(rel)]
        if not source_root.is_dir() or not temp_dir.is_dir():
            raise RuntimeError("The downloaded update staging folder is no longer available.")
        if not changed:
            raise RuntimeError("The update contains no eligible changed application files.")

        launcher_dir = Path(tempfile.mkdtemp(prefix="grizzlymax_updater_"))
        ps1_path = launcher_dir / "grizzlymax_update.ps1"
        bat_path = launcher_dir / "update.bat"
        state_stage = temp_dir / "installed_update_state.json"
        state = {
            "repository": APP_UPDATE_REPO,
            "commit": payload.get("commit", ""),
            "installed_at": datetime.now().isoformat(timespec="seconds"),
            "files": payload.get("manifest", []),
        }
        state_stage.write_text(json.dumps(state, indent=2), encoding="utf-8")

        updater_payload = launcher_dir / "update_payload.json"
        updater_payload.write_text(json.dumps({
            "parent_pid": int(os.getpid()),
            "app_root": str(ROOT.resolve()),
            "source_root": str(source_root.resolve()),
            "download_temp": str(temp_dir.resolve()),
            "state_stage": str(state_stage.resolve()),
            "state_destination": str(APP_UPDATE_STATE.resolve()),
            "start_bat": str((ROOT / "start.bat").resolve()),
            "python_exe": str(PYTHON.resolve()),
            "gui_script": str(Path(__file__).resolve()),
            "changed": changed,
        }, indent=2), encoding="utf-8")

        ps1 = r'''param([Parameter(Mandatory=$true)][string]$PayloadPath)
$ErrorActionPreference = 'Stop'
$data = Get-Content -LiteralPath $PayloadPath -Raw | ConvertFrom-Json
$parentPid = [int]$data.parent_pid
$appRoot = [string]$data.app_root
$sourceRoot = [string]$data.source_root
$downloadTemp = [string]$data.download_temp
$stateStage = [string]$data.state_stage
$stateDestination = [string]$data.state_destination
$startBat = [string]$data.start_bat
$pythonExe = [string]$data.python_exe
$guiScript = [string]$data.gui_script
$logDir = Join-Path $appRoot 'logs'
$logPath = Join-Path $logDir 'app_update_last.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"[$(Get-Date -Format s)] External updater started. Waiting for GUI PID $parentPid." | Set-Content -LiteralPath $logPath -Encoding UTF8
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    $proc = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
    if ($null -eq $proc) { break }
    Start-Sleep -Milliseconds 250
}
if (Get-Process -Id $parentPid -ErrorAction SilentlyContinue) {
    throw "MiniMax H3 Standalone did not close within 120 seconds."
}
Start-Sleep -Milliseconds 500
$count = 0
foreach ($relValue in $data.changed) {
    $rel = [string]$relValue
    $nativeRel = $rel.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $src = Join-Path $sourceRoot $nativeRel
    $dst = Join-Path $appRoot $nativeRel
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw "Update source file is missing: $rel" }
    $dstDir = Split-Path -Parent $dst
    if ($dstDir) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
    $copied = $false
    $lastError = $null
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $tmp = "$dst.update_tmp"
            Copy-Item -LiteralPath $src -Destination $tmp -Force
            Move-Item -LiteralPath $tmp -Destination $dst -Force
            $copied = $true
            break
        } catch {
            $lastError = $_.Exception.Message
            Remove-Item -LiteralPath "$dst.update_tmp" -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $copied) { throw "Could not replace $rel after 30 attempts. $lastError" }
    $count++
    "[$(Get-Date -Format s)] Updated: $rel" | Add-Content -LiteralPath $logPath -Encoding UTF8
}
$stateDir = Split-Path -Parent $stateDestination
if ($stateDir) { New-Item -ItemType Directory -Force -Path $stateDir | Out-Null }
Copy-Item -LiteralPath $stateStage -Destination $stateDestination -Force
"[$(Get-Date -Format s)] Update complete: $count file(s)." | Add-Content -LiteralPath $logPath -Encoding UTF8
Remove-Item -LiteralPath $downloadTemp -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $startBat -PathType Leaf) {
    Start-Process -FilePath $startBat -WorkingDirectory $appRoot
} elseif (Test-Path -LiteralPath $pythonExe -PathType Leaf) {
    Start-Process -FilePath $pythonExe -ArgumentList @($guiScript) -WorkingDirectory $appRoot
} else {
    throw "Update installed, but neither start.bat nor the bundled Python executable could be found for restart."
}
"[$(Get-Date -Format s)] Restart launched." | Add-Content -LiteralPath $logPath -Encoding UTF8
$launcherDir = Split-Path -Parent $PayloadPath
$cleanupCommand = 'ping 127.0.0.1 -n 3 >nul & rmdir /s /q "' + $launcherDir + '"'
Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -ArgumentList @('/c', $cleanupCommand)
'''
        ps1_path.write_text(ps1, encoding="utf-8-sig")

        bat = r'''@echo off
setlocal
title GrizzlyMax MiniMax H3 Updater
echo.
echo GrizzlyMax updater is waiting for MiniMax H3 Standalone to close...
echo Do not close this window while files are being updated.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0grizzlymax_update.ps1" -PayloadPath "%~dp0update_payload.json"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo ================================================================
  echo UPDATE FAILED. No automatic restart was attempted.
  echo Check logs\app_update_last.log in the application folder.
  echo ================================================================
  echo.
  pause
)
exit /b %RC%
'''
        bat_path.write_text(bat, encoding="utf-8")
        return bat_path

    def _install_update(self, payload):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "Application update",
                "A generation is currently running. Finish or cancel the running job before installing the application update."
            )
            self._cleanup_update_payload(payload)
            return

        if os.name == "nt":
            try:
                updater_bat = self._write_external_update_launcher(payload)
                self.save_last()
                self.status.setText("Update prepared — closing app so the external updater can install it…")
                flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                subprocess.Popen(
                    ["cmd.exe", "/c", str(updater_bat)],
                    cwd=str(updater_bat.parent),
                    creationflags=flags,
                )
                if payload is self._update_payload:
                    self._update_payload = None
                QApplication.instance().quit()
                return
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Application update",
                    f"The external updater could not be started.\n\n{type(exc).__name__}: {exc}"
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
            "The update was installed successfully.\n\nRestart MiniMax H3 Standalone now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if restart == QMessageBox.StandardButton.Yes:
            self.save_last()
            start_bat = ROOT / "start.bat"
            try:
                if start_bat.is_file():
                    subprocess.Popen([str(start_bat)], cwd=str(ROOT))
                else:
                    subprocess.Popen([str(PYTHON), str(Path(__file__).resolve())], cwd=str(ROOT))
                QApplication.instance().quit()
            except Exception as exc:
                QMessageBox.warning(self, "Restart", f"The update is installed, but the app could not restart automatically.\n\n{exc}\n\nPlease restart it manually.")

    def validate_install(self):
        if not PYTHON.is_file(): self.status.setText("Environment missing"); return
        mode = self.mode.currentIndex()
        if self.use_hybrid_model.isChecked():
            hybrid = self._resolve_hybrid_model_path(populate=True)
            if hybrid is None:
                QMessageBox.warning(self, "Hybrid model missing", "Use hybrid model is enabled, but no compatible hybrid .safetensors checkpoint was found in the selected path or MiniMax model folders.")
                return
        self.status.setText("Validating…")
        args = ["-m", "runtime.validate_models"] + self.model_override_args(mode)
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
        mode=self.mode.currentIndex(); w,h=self._current_resolution(); frames=self._frame_count()
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
        continue_last=False; continue_from_job_id=None; continue_from_job_number=None; manual_continue_video=""; glue_results=False; continue_audio_memory=False; latent_continuation=False; combine_frames_latent=False
        if mode==1:
            continue_last=self.continue_last_result.isChecked()
            glue_results=self.glue_results.isChecked()
            continue_audio_memory=bool(continue_last and self.continue_audio_memory.isChecked())
            latent_continuation=bool(self.latent_continuation.isChecked())
            combine_frames_latent=bool(latent_continuation and self.combine_frames_latent.isChecked())
            manual_continue_video="" if continue_last else self.continue_video.path()
            if (manual_continue_video or continue_last) and self.first.path():
                QMessageBox.warning(self,"Conflicting FL2VA inputs","Continue Video already supplies the first-frame boundary. Clear the separate First frame."); return
            if self.first.path(): args += ["--first-frame",self.first.path()]
            if self.last.path(): args += ["--last-frame",self.last.path()]
            if manual_continue_video and not Path(manual_continue_video).is_file():
                QMessageBox.warning(self,"Video missing","The selected Continue video file does not exist."); return
            if continue_last:
                previous=self._latest_non_cancelled_queue_job()
                if previous is None:
                    QMessageBox.warning(self,"No previous queue job","Continue last result needs an earlier non-cancelled queue job to continue from. Add or finish a source job first."); return
                if previous.get("state")=="failed":
                    QMessageBox.warning(self,"Previous job failed","The latest non-cancelled queue job failed and continuation stops there. Remove the failed job or run a new successful source job first."); return
                if previous.get("state")=="finished" and not Path(previous.get("output","")).is_file():
                    QMessageBox.warning(self,"Previous output missing","The latest non-cancelled queue job is finished but its output file is missing."); return
                continue_from_job_id=previous.get("id")
                continue_from_job_number=previous.get("job_number")
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
            ref_audios = self.ref_audios.paths()
            for pth in ref_audios: args += ["--ref-audio",pth]
            if self.lock_source_audio.isChecked():
                if not ref_audios:
                    QMessageBox.warning(self, "Source audio required", "Use Audio 1 as exact source / output needs at least one standalone audio file in Audio slot 1.")
                    return
                args += ["--lock-source-audio-index", "1"]
        args += self.model_override_args(mode)
        args += self.lora_args()
        if self.vram_manager_enabled.isChecked():
            args += ["--vram-manager-auto" if self.vram_manager_auto_bypass.isChecked() else "--vram-manager"]
            args += ["--vram-residency-engine", str(self.vram_residency_engine.currentData() or "static"), "--vram-runtime-free-gb", str(self.vram_runtime_free.value()), "--vram-text-headroom-gb", str(self.vram_text_headroom.value()), "--vram-diffusion-headroom-gb", str(self.vram_diffusion_headroom.value()), "--vram-offload-chunk-mb", str(self.vram_offload_chunk.value()), "--vram-max-resident-weights-gb", str(self.vram_max_weights.value()), "--vram-block-check-interval", str(self.vram_block_interval.value()), "--vram-async-streams", str(self.vram_async_streams.value()), "--vram-video-vae-reserve-gb", str(self.vram_video_vae_reserve.value()), "--vram-audio-vae-reserve-gb", str(self.vram_audio_vae_reserve.value()), "--vram-residency-target-free-gb", str(self.vram_residency_target_free.value()), "--vram-residency-warmup-blocks", str(self.vram_residency_warmup.value()), "--vram-residency-refill-interval", str(self.vram_residency_refill_interval.value())]
            args += ["--vram-residency-fill" if self.vram_residency_fill.isChecked() else "--no-vram-residency-fill"]
        if self.spectrum_enabled.isChecked(): args += ["--spectrum"]
        if self.sage_attention_enabled.isChecked(): args += ["--sage-attention"]
        if self.sol_attention_enabled.isChecked(): args += ["--sol-attention"]
        if self.sla_attention_enabled.isChecked(): args += ["--sla-attention"]
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
        base=out; n=2
        while out.exists() or str(out.resolve()).lower() in used:
            out=base.with_name(f"{base.stem}_{n:03d}{base.suffix}"); n+=1
        args += ["--output",str(out)]
        if self.use_hybrid_model.isChecked():
            resolved_hybrid = self._resolve_hybrid_model_path(populate=True)
            if resolved_hybrid is None:
                QMessageBox.critical(self, "Hybrid model missing", "Use hybrid model is enabled, but no compatible hybrid .safetensors checkpoint was found in the selected path or MiniMax model folders."); return
            model_path = str(resolved_hybrid)
            model_label = f"Hybrid: {resolved_hybrid.name}"
        else:
            model_path=self.ref2va_model.path() if mode==2 else self.fl2va_model.path()
            model_label=Path(model_path).name if model_path else ("Ref2VA default" if mode==2 else "FL2VA default")
        job={"id":uuid.uuid4().hex,"job_number":self._take_next_job_number(),"state":"pending","created_at":time.time(),"started_at":None,"finished_at":None,"elapsed":0,"mode":mode,"mode_name":self.mode.currentText(),"model_label":model_label,"output":str(out),"seed":self.seed.value(),"actual_seed":None,"resolution":f"{w} × {h}","frames":frames,"steps":self.steps.value(),"prompt":prompt,"args":args,"progress":None,"phase":"Waiting","error":"","cancel_reason":"","settings":self.settings_dict(),"log_tail":"","continue_last_result":bool(continue_last),"continue_from_job_id":continue_from_job_id,"continue_from_job_number":continue_from_job_number,"manual_continue_video":manual_continue_video,"continue_context_frames":int(self.continue_context.currentData() or 90) if mode==1 else None,"glue_results":bool(glue_results),"continue_audio_memory":bool(continue_audio_memory),"latent_continuation":bool(latent_continuation),"combine_frames_latent":bool(combine_frames_latent)}
        self.queue_jobs.append(job); self.save_last(); self._save_queue_state(); self._refresh_queue_views(); self.status.setText("Job added to queue")
        self._start_next_pending()

    def _play_completed_result(self, job):
        if not job or job.get("state") != "finished":
            return
        path = Path(job.get("output", ""))
        if not path.is_file():
            return
        # Results now always use the permanent global preview pane.  The separate
        # Play result when finished toggle still controls whether this happens
        # automatically; it no longer switches between embedded/system players.
        self._load_preview(job, autoplay=True)

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
                job["state"]="finished"; job["phase"]="Finished"
                job["clip_duration"] = self._probe_clip_duration(job.get("output"), job.get("frames"))
                if job.get("job_type") == "timeline_assembly":
                    if getattr(self, "timeline_widget", None) is not None:
                        self.timeline_widget.mark_assembly_finished(job.get("output"))
                    self.status.setText(f"Timeline assembly finished: {Path(job.get('output','')).name}")
                    self.append_log(f"=== TIMELINE ASSEMBLY FINISHED ===\n{job.get('output')}\n")
                if self.play_result_finished.isChecked():
                    play_finished_job = job
            else:
                job["state"]="failed"; job["error"]=self._extract_failure_reason(job,code); job["phase"]="Failed"
                if job.get("job_type") == "timeline_assembly" and getattr(self, "timeline_widget", None) is not None:
                    self.timeline_widget.mark_assembly_failed(job.get("error") or "Assembly failed")
        if job and job.get("job_type") == "timeline_assembly":
            # Keep the concat recipe when a running assembly is explicitly requeued;
            # it is needed for the next attempt. Terminal states can clean it up.
            if job.get("state") in ("finished", "failed", "cancelled"):
                concat = Path(str(job.get("timeline_assembly_concat") or ""))
                try:
                    if concat.is_file():
                        concat.unlink()
                except Exception:
                    pass
                if job.get("state") != "finished":
                    try:
                        partial = Path(str(job.get("output") or ""))
                        if partial.is_file():
                            partial.unlink()
                    except Exception:
                        pass
            elif job.get("state") == "pending" and getattr(self, "timeline_widget", None) is not None:
                self.timeline_widget.mark_assembly_queued(job.get("output"), job.get("id"))
            if job.get("state") == "cancelled" and getattr(self, "timeline_widget", None) is not None:
                self.timeline_widget.mark_assembly_failed("Cancelled")
        finish_line=f"=== FINISHED: exit {code} ===\n"
        self._write_job_log_file(finish_line)
        self.append_log(finish_line); self.proc=None; self.current_job_id=None; self._termination_action=None; self.cancel.setEnabled(False); self.gen.setEnabled(True)
        self._active_log_file=None
        self.status.setText("Queue ready" if code in (0,3) else f"Job stopped/failed ({code})"); self._save_queue_state(); self._refresh_queue_views()
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
        if self._closing or self._user_window_size_override:
            return
        state = self.windowState()
        if state & Qt.WindowState.WindowMinimized:
            return
        if not (state & Qt.WindowState.WindowMaximized):
            self.showMaximized()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
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
        if self._checkpoint_download_proc is not None and self._checkpoint_download_proc.state() != QProcess.ProcessState.NotRunning:
            self._checkpoint_download_proc.kill()
            self._checkpoint_download_proc.waitForFinished(1500)
        self._stop_prompt_builder()
        super().closeEvent(e)


def main():
    # Give the standalone GUI its own Windows taskbar identity instead of being
    # grouped under python.exe / the generic Qt application icon.
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "GetGoingFast.GrizzlyMax.MiniMaxH3"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("minimax H3 Standalone")

    app_icon = QIcon(str(APP_ICON)) if APP_ICON.is_file() else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    w = MainWindow()
    if not app_icon.isNull():
        w.setWindowIcon(app_icon)

    # Start in the state most users keep this control-heavy GUI in.  A later
    # manual restore/resize is respected by MainWindow.changeEvent().
    w.showMaximized()
    QTimer.singleShot(0, w._schedule_layout_refresh)
    return app.exec()
if __name__ == "__main__": raise SystemExit(main())
