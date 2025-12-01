
# helpers/upsc.py — Upscale tab with video pipeline and model-dir fixes
from __future__ import annotations
import os
import shutil
import subprocess
import json, tempfile, time
from pathlib import Path
from typing import List, Tuple, Optional
from collections import OrderedDict
from helpers.mediainfo import refresh_info_now

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QTextCursor, QPixmap, QIcon, QPainter, QPainterPath

try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS, OUT_SHOTS
except Exception:
    ROOT = Path(__file__).resolve().parent.parent
    OUT_VIDEOS = ROOT / "output" / "video"
    OUT_SHOTS = ROOT / "output" / "photo"



# --------- flow layout (wrap) ---------
from PySide6.QtWidgets import QLayout, QSizePolicy
from PySide6.QtCore import QPoint, QRect, QSize, Qt

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, hSpacing=8, vSpacing=8):
        super().__init__(parent)
        self._itemList = []
        self._hSpace = hSpacing
        self._vSpace = vSpacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self._itemList.append(item)

    def count(self):
        return len(self._itemList)

    def itemAt(self, index):
        if 0 <= index < len(self._itemList):
            return self._itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._itemList):
            return self._itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._itemList:
            size = size.expandedTo(item.minimumSize())
        m_left, m_top, m_right, m_bottom = self.getContentsMargins()
        size += QSize(m_left + m_right, m_top + m_bottom)
        return size

    def horizontalSpacing(self):
        return self._hSpace

    def verticalSpacing(self):
        return self._vSpace

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spaceX = self.horizontalSpacing()
        spaceY = self.verticalSpacing()

        for item in self._itemList:
            wid = item.widget()
            if wid and not wid.isVisible():
                continue
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

# --- helper: rounded pixmap for nicer thumbnails (shared with recents) ---
def _rounded_pixmap(pm, radius: int = 10):
    try:
        if pm is None:
            return pm
        if isinstance(pm, QPixmap) and not pm.isNull():
            w, h = pm.width(), pm.height()
            if w <= 0 or h <= 0:
                return pm
            r = max(0, int(radius))
            out = QPixmap(w, h)
            out.fill(Qt.transparent)
            p = QPainter(out)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            path = QPainterPath()
            from PySide6.QtCore import QRectF as _QRectF
            path.addRoundedRect(_QRectF(0, 0, w, h), r, r)
            p.setClipPath(path)
            p.drawPixmap(0, 0, pm)
            p.end()
            return out
        return pm
    except Exception:
        return pm
# --- end helper ---


class _Disclosure(QtWidgets.QWidget):
    """Minimal collapsible section used for the Upscale recents box."""
    toggled = Signal(bool)

    def __init__(self, title: str, content: QtWidgets.QWidget, start_open: bool = False, parent=None):
        super().__init__(parent)
        self._btn = QtWidgets.QToolButton(self)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.DownArrow if start_open else Qt.RightArrow)
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(start_open)
        self._btn.toggled.connect(self._on_clicked)
        self._body = content
        self._body.setVisible(start_open)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._btn)
        lay.addWidget(self._body)

    def _on_clicked(self, checked: bool):
        try:
            self._body.setVisible(checked)
        except Exception:
            pass
        try:
            self._btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        except Exception:
            pass
        try:
            self.toggled.emit(checked)
        except Exception:
            pass


PRESETS_BIN = ROOT / "presets" / "bin"
BIN_DIR = ROOT / "bin"
MODELS_DIR = ROOT / "models"
REALSR_DIR = MODELS_DIR / "realesrgan"
WAIFU2X_DIR = MODELS_DIR / "waifu2x"
SRMD_DIR = MODELS_DIR / "srmd-ncnn-vulkan-master"
REALSR_NCNN_DIR = MODELS_DIR / "realsr-ncnn-vulkan-20220728-windows"

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".gif"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _exists(p: str | Path) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def _which_ffmpeg(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (PRESETS_BIN, BIN_DIR, ROOT):
        cand = d / exe
        if cand.exists():
            return str(cand)
    return name


FFMPEG = _which_ffmpeg("ffmpeg")
FFPROBE = _which_ffmpeg("ffprobe")

# UI switches
SHOW_UPSCALE_LOG = False  # hide the big log window; logging remains internal


def detect_engines() -> List[Tuple[str, str]]:
    engines: List[Tuple[str, str]] = []
    realsr = REALSR_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")
    if realsr.exists():
        engines.append(("Real-ESRGAN (ncnn)", str(realsr)))
    engines.append(("UltraSharp (ncnn)", str(realsr)))
    engines.append(("SRMD (ncnn via RealESRGAN)", str(realsr)))
    waifu = WAIFU2X_DIR / ("waifu2x-ncnn-vulkan.exe" if os.name == "nt" else "waifu2x-ncnn-vulkan")
    if waifu.exists():
        engines.append(("Waifu2x (ncnn)", str(waifu)))
    return engines



def scan_realsr_models() -> List[str]:
    names: set[str] = set()
    if REALSR_DIR.exists():
        for ext in ("*.bin", "*.param"):
            for p in sorted(REALSR_DIR.glob(ext)):
                stem = p.stem
                low = stem.lower()
                if low.startswith("srmd") or low.startswith("srmdnf") or low.startswith("srdmnf") or low.startswith("4x-ultrasharp"):
                    continue
                names.add(stem)
    if names:
        return sorted(names)
    # Fallback suggestions if no models detected (first-run scenarios)
    return [
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3-x4",
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
    ]




def scan_ultrasharp_models() -> List[str]:
    names: set[str] = set()
    try:
        if REALSR_DIR.exists():
            for ext in ("*.bin","*.param"):
                for p in sorted(REALSR_DIR.glob("4x-UltraSharp-*"+ext.split("*")[-1])):
                    names.add(p.stem)
    except Exception:
        pass
    return sorted(names) or ["4x-UltraSharp-fp16","4x-UltraSharp-fp32"]

def scan_srmd_realesrgan_models() -> List[str]:
    names: set[str] = set()
    try:
        if REALSR_DIR.exists():
            for ext in ("*.bin","*.param"):
                for p in sorted(REALSR_DIR.glob(ext)):
                    s = p.stem.lower()
                    if s.startswith("srmd") or s.startswith("srmdnf") or s.startswith("srdmnf"):
                        names.add(p.stem)
    except Exception:
        pass
    return sorted(names) or ["srmd_x2","srmd_x3","srmd_x4","srmdnf_x2","srmdnf_x3","srmdnf_x4"]

def scan_waifu2x_models() -> List[str]:
    names: List[str] = []
    if WAIFU2X_DIR.exists():
        for p in sorted(WAIFU2X_DIR.glob("models-*")):
            names.append(p.name)
    return names or ["models-cunet", "models-upconv_7_photo", "models-upconv_7_anime_style_art_rgb"]
def scan_srmd_models() -> List[str]:
    names: List[str] = []
    try:
        if SRMD_DIR.exists():
            for p in sorted(SRMD_DIR.glob("models-*")):
                if p.is_dir():
                    names.append(p.name)
    except Exception:
        pass
    return names or ["models-DF2K", "models-DF2K_JPEG"]

def scan_realsr_ncnn_models() -> List[str]:
    names: List[str] = []
    try:
        if REALSR_NCNN_DIR.exists():
            for p in sorted(REALSR_NCNN_DIR.glob("*.param")):
                names.append(p.stem)
    except Exception:
        pass
    return names or ["realsr-x2", "realsr-x4"]
    names: List[str] = []
    if WAIFU2X_DIR.exists():
        for p in sorted(WAIFU2X_DIR.glob("models-*")):
            names.append(p.name)
    return names or ["models-cunet", "models-upconv_7_photo", "models-upconv_7_anime_style_art_rgb"]


def _parse_fps(src: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "0", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            cwd=str(ROOT), universal_newlines=True
        )
        rat = out.strip() or "0/0"
        if "/" in rat:
            a, b = rat.split("/", 1)
            a = float(a)
            b = float(b) if float(b) != 0 else 1.0
            fps = a / b
        else:
            fps = float(rat)
        if fps <= 0:
            return None
        return f"{fps:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return None


class _RunThread(QtCore.QThread):
    progress = Signal(str)
    done = Signal(int, str)

    def __init__(self, cmds: List[List[str]], cwd: Optional[Path] = None, parent=None):
        super().__init__(parent)
        if cmds and isinstance(cmds[0], str):
            cmds = [cmds]
        self.cmds = cmds
        self.cwd = cwd

    def run(self):
        last_cmd = ""
        try:
            total = len(self.cmds)
            for i, cmd in enumerate(self.cmds, 1):
                last_cmd = " ".join(cmd)
                self.progress.emit(f"[{i}/{total}] {last_cmd}")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.cwd) if self.cwd else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                for line in proc.stdout:
                    self.progress.emit(line.rstrip())
                ret = proc.wait()
                if ret is not None and ret > 0x7FFFFFFF:
                    ret -= 0x100000000  # fix Windows -1 wrap
                if ret != 0:
                    self.done.emit(int(ret), last_cmd)
                    return
            self.done.emit(0, last_cmd)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.done.emit(-1, last_cmd)


class UpscPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._engines = detect_engines()
        self._realsr_models = scan_realsr_models()
        self._waifu_models = scan_waifu2x_models()
        self._srmd_models = scan_srmd_models()
        self._realsr_ncnn_models = scan_realsr_ncnn_models()
        self._ultrasharp_models = scan_ultrasharp_models()
        self._srmd_realsr_models = scan_srmd_realesrgan_models()
        self._last_outfile: Optional[Path] = None
        # LRU cache for thumbnails (path+size+radius)
        self._pm_cache = OrderedDict()
        self._pm_cache_cap = 128
        self._build_ui()

    def set_main(self, main):  # optional hook
        self._main = main

    def _build_ui(self):
        v_main = QtWidgets.QVBoxLayout(self)

        # Fancy green banner at the top
        self.banner = QtWidgets.QLabel("Upscaling", self)
        self.banner.setObjectName("upscBanner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.banner.setFixedHeight(45)
        self.banner.setStyleSheet(
            "#upscBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: white;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #1565c0,"      # deep blue
            "   stop:0.5 #1e88e5,"    # medium blue
            "   stop:1 #42a5f5"       # bright blue
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        v_main.addWidget(self.banner)
        v_main.addSpacing(4)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        _content = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(_content)
        scroll.setWidget(_content)
        v_main.addWidget(scroll, 1)

        
        # Preferences (compact grid: avoids cramped header and wraps nicely)
        prefs_grid = QtWidgets.QGridLayout()
        prefs_grid.setHorizontalSpacing(12)
        prefs_grid.setVerticalSpacing(6)

        self.chk_remember = QtWidgets.QCheckBox('Remember settings', self)
        self.chk_remember.setChecked(True)
        self.chk_remember.hide()  # hidden; always on
        self.chk_video_thumbs = QtWidgets.QCheckBox("Video thumbnails (heavier)", self)
        self.chk_video_thumbs.setChecked(True)
        self.chk_video_thumbs.hide()
        self.chk_video_thumbs.setToolTip("Generate first-frame thumbnails for videos. Uses more CPU/RAM. Disabled while jobs run.")
        prefs_grid.addWidget(self.chk_video_thumbs, 0, 2, 1, 2)

        self.chk_streaming_lowmem = getattr(self, "chk_streaming_lowmem", None) or QtWidgets.QCheckBox("Streaming (low memory)", self)
        self.chk_streaming_lowmem.setChecked(True)
        self.chk_streaming_lowmem.setToolTip("Process in a streaming, low-memory mode using smaller buffers. Good default for stability.")
        prefs_grid.addWidget(self.chk_streaming_lowmem, 1, 2, 1, 2)

        prefs_grid.setColumnStretch(3, 1)
        v.addLayout(prefs_grid)

        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        self.edit_input = QtWidgets.QLineEdit(self)
        self.edit_input.setObjectName("edit_input")
        self.btn_browse = QtWidgets.QPushButton("Browse…", self)
        self.btn_upscale = QtWidgets.QPushButton("Upscale", self)
        self.btn_batch = QtWidgets.QPushButton("Batch…", self)
        row.addWidget(self.edit_input, 1)
        row.addWidget(self.btn_browse)
        
        self.btn_info = QtWidgets.QPushButton("Info", self)
        v.addLayout(row)

        scale_lay = QtWidgets.QHBoxLayout()
        scale_lay.addWidget(QtWidgets.QLabel("Scale:", self))
        self.spin_scale = QtWidgets.QDoubleSpinBox(self)
        self.spin_scale.setRange(1.0, 8.0)
        self.spin_scale.setSingleStep(0.5)
        self.spin_scale.setValue(2.0)
        self.spin_scale.setDecimals(1)
        self.slider_scale = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider_scale.setMinimum(10)
        self.slider_scale.setMaximum(80)
        self.slider_scale.setSingleStep(5)
        self.slider_scale.setPageStep(5)
        self.slider_scale.setValue(20)
        scale_lay.addWidget(self.spin_scale)
        scale_lay.addWidget(self.slider_scale, 1)
        v.addLayout(scale_lay)

        from helpers.collapsible_compat import CollapsibleSection
        # Models (inline, no collapsible)
        self.box_models = QtWidgets.QWidget(self)
        inner = QtWidgets.QVBoxLayout(self.box_models)
        inner.setContentsMargins(6, 2, 6, 6)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel("Engine:", self))
        self.combo_engine = QtWidgets.QComboBox(self)
        for label, exe in self._engines:
            self.combo_engine.addItem(label, exe)
        if self.combo_engine.count() == 0:
            self.combo_engine.addItem("Real-ESRGAN (expected at models/realesrgan)",
                                      str(REALSR_DIR / "realesrgan-ncnn-vulkan.exe"))
        hl.addWidget(self.combo_engine, 1)
        inner.addLayout(hl)

        self.stk_models = QtWidgets.QStackedWidget(self)
        # Real-ESRGAN
        pg_r = QtWidgets.QWidget()
        lay_r = QtWidgets.QHBoxLayout(pg_r)
        lay_r.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_realsr = QtWidgets.QComboBox(self)
        for m in self._realsr_models:
            self.combo_model_realsr.addItem(m)
        lay_r.addWidget(self.combo_model_realsr, 1)
        # Badge for model category
        self.lbl_model_badge = QtWidgets.QLabel("—", self)
        self.lbl_model_badge.setStyleSheet("QLabel{padding:2px 6px;border-radius:8px;background:#394b70;color:#cfe3ff;font-size:12px;font-weight:600;}")
        
        self.stk_models.addWidget(pg_r)

        # Waifu2x
        pg_w = QtWidgets.QWidget()
        lay_w = QtWidgets.QHBoxLayout(pg_w)
        lay_w.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_w2x = QtWidgets.QComboBox(self)
        for m in self._waifu_models:
            self.combo_model_w2x.addItem(m)
        lay_w.addWidget(self.combo_model_w2x, 1)
        self.stk_models.addWidget(pg_w)
        # SRMD (ncnn)
        pg_srmd = QtWidgets.QWidget()
        lay_s = QtWidgets.QHBoxLayout(pg_srmd)
        lay_s.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_srmd = QtWidgets.QComboBox(self)
        for m in self._srmd_models:
            self.combo_model_srmd.addItem(m)
        lay_s.addWidget(self.combo_model_srmd, 1)
        self.stk_models.addWidget(pg_srmd)
        # RealSR (ncnn)
        pg_rs = QtWidgets.QWidget()
        lay_rs = QtWidgets.QHBoxLayout(pg_rs)
        lay_rs.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_realsr_ncnn = QtWidgets.QComboBox(self)
        for m in self._realsr_ncnn_models:
            self.combo_model_realsr_ncnn.addItem(m)
        lay_rs.addWidget(self.combo_model_realsr_ncnn, 1)
        self.stk_models.addWidget(pg_rs)
        # UltraSharp (via Real-ESRGAN backend)
        pg_ul = QtWidgets.QWidget()
        lay_ul = QtWidgets.QHBoxLayout(pg_ul)
        lay_ul.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_ultrasharp = QtWidgets.QComboBox(self)
        for m in self._ultrasharp_models:
            self.combo_model_ultrasharp.addItem(m)
        lay_ul.addWidget(self.combo_model_ultrasharp, 1)
        self.stk_models.addWidget(pg_ul)

        # SRMD (via Real-ESRGAN backend)
        pg_srmd_rs = QtWidgets.QWidget()
        lay_srmd_rs = QtWidgets.QHBoxLayout(pg_srmd_rs)
        lay_srmd_rs.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_srmd_realsr = QtWidgets.QComboBox(self)
        for m in self._srmd_realsr_models:
            self.combo_model_srmd_realsr.addItem(m)
        lay_srmd_rs.addWidget(self.combo_model_srmd_realsr, 1)
        self.stk_models.addWidget(pg_srmd_rs)


        inner.addWidget(self.stk_models)

        # Model hint row (hint + badge on the same line)
        self.lbl_model_hint = QtWidgets.QLabel("", self)
        self.lbl_model_hint.setWordWrap(True)
        self.lbl_model_hint.setStyleSheet("color:#9fb3c8;font-size:12px;")
        hl_hint = QtWidgets.QHBoxLayout()
        hl_hint.addWidget(self.lbl_model_hint, 1)
        hl_hint.addWidget(self.lbl_model_badge)
        inner.addLayout(hl_hint)

        hl_out = QtWidgets.QHBoxLayout()
        hl_out.addWidget(QtWidgets.QLabel("Output:", self))
        self.edit_outdir = QtWidgets.QLineEdit(str(OUT_VIDEOS))
        btn_out = QtWidgets.QPushButton("…", self)
        hl_out.addWidget(self.edit_outdir, 1)
        hl_out.addWidget(btn_out)
        inner.addLayout(hl_out)
        v.addWidget(self.box_models)

        # Encoder (inline)
        self.box_encoder = QtWidgets.QWidget(self)
        lay_enc = QtWidgets.QGridLayout(self.box_encoder); lay_enc.setContentsMargins(6,2,6,6)
        # Video codec
        lay_enc.addWidget(QtWidgets.QLabel("Video codec:", self), 0, 0)
        self.combo_vcodec = QtWidgets.QComboBox(self)
        for c in ("libx264","libx265","libsvtav1","libaom-av1"):
            self.combo_vcodec.addItem(c)
        self.combo_vcodec.setCurrentText("libx264")
        lay_enc.addWidget(self.combo_vcodec, 0, 1)
        # Rate control
        self.rad_crf = QtWidgets.QRadioButton("CRF", self); self.rad_crf.setChecked(True)
        self.spin_crf = QtWidgets.QSpinBox(self); self.spin_crf.setRange(0, 51); self.spin_crf.setValue(18)
        lay_enc.addWidget(self.rad_crf, 1, 0); lay_enc.addWidget(self.spin_crf, 1, 1)
        self.rad_bitrate = QtWidgets.QRadioButton("Bitrate (kbps)", self)
        self.spin_bitrate = QtWidgets.QSpinBox(self); self.spin_bitrate.setRange(100, 200000); self.spin_bitrate.setValue(8000)
        lay_enc.addWidget(self.rad_bitrate, 2, 0); lay_enc.addWidget(self.spin_bitrate, 2, 1)

        # Grey out the non-selected rate control
        try:
            def _update_rate_controls():
                use_crf = bool(self.rad_crf.isChecked())
                try:
                    self.spin_crf.setEnabled(use_crf)
                except Exception:
                    pass
                try:
                    self.spin_bitrate.setEnabled(bool(self.rad_bitrate.isChecked()))
                except Exception:
                    pass
            try:
                self.rad_crf.toggled.connect(_update_rate_controls)
                self.rad_bitrate.toggled.connect(_update_rate_controls)
            except Exception:
                pass
            _update_rate_controls()
        except Exception:
            pass

        # Preset + Keyint
        lay_enc.addWidget(QtWidgets.QLabel("Preset:", self), 3, 0)
        self.combo_preset = QtWidgets.QComboBox(self)
        for p in ("ultrafast","fast","medium","slow"):
            self.combo_preset.addItem(p)
        self.combo_preset.setCurrentText("fast")
        lay_enc.addWidget(self.combo_preset, 3, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Keyint (GOP):", self), 4, 0)
        self.spin_keyint = QtWidgets.QSpinBox(self); self.spin_keyint.setRange(0, 1000); self.spin_keyint.setValue(0)
        self.spin_keyint.setToolTip("0 = let encoder decide; otherwise sets -g keyint")
        lay_enc.addWidget(self.spin_keyint, 4, 1)

        # Hide Keyint (auto mode). Force to 0 under the hood.
        try:
            self._lbl_keyint = lay_enc.itemAtPosition(4, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None
            if self._lbl_keyint is not None:
                self._lbl_keyint.hide()
            self.spin_keyint.setValue(0)
            self.spin_keyint.hide()
            try:
                # ensure any saved setting won't override this
                self.spin_keyint.valueChanged.connect(lambda _v: self.spin_keyint.setValue(0))
            except Exception:
                pass
        except Exception:
            pass

        # Audio
        lay_enc.addWidget(QtWidgets.QLabel("Audio:", self), 5, 0)
        self.radio_a_copy = QtWidgets.QRadioButton("Keep", self); self.radio_a_copy.setChecked(True)
        self.radio_a_encode = QtWidgets.QRadioButton("Encode", self)
        self.radio_a_mute = QtWidgets.QRadioButton("Mute", self)
        arow = QtWidgets.QHBoxLayout(); arow.addWidget(self.radio_a_copy); arow.addWidget(self.radio_a_encode); arow.addWidget(self.radio_a_mute); arow.addStretch(1)
        lay_enc.addLayout(arow, 5, 1)

        # --- Ensure radio groups are independent ---
        try:
            self.grp_rate = QtWidgets.QButtonGroup(self.box_encoder)
            self.grp_rate.setExclusive(True)
            self.grp_rate.addButton(self.rad_crf)
            self.grp_rate.addButton(self.rad_bitrate)

            self.grp_audio = QtWidgets.QButtonGroup(self.box_encoder)
            self.grp_audio.setExclusive(True)
            self.grp_audio.addButton(self.radio_a_copy)
            self.grp_audio.addButton(self.radio_a_encode)
            self.grp_audio.addButton(self.radio_a_mute)
        except Exception:
            pass
        # Audio codec/bitrate
        lay_enc.addWidget(QtWidgets.QLabel("Audio codec:", self), 6, 0)
        self.combo_acodec = QtWidgets.QComboBox(self)
        for ac in ("aac","libopus","libvorbis"):
            self.combo_acodec.addItem(ac)
        lay_enc.addWidget(self.combo_acodec, 6, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Audio bitrate (kbps):", self), 7, 0)
        self.spin_abitrate = QtWidgets.QSpinBox(self); self.spin_abitrate.setRange(32, 1024); self.spin_abitrate.setValue(192)
        lay_enc.addWidget(self.spin_abitrate, 7, 1)

        # Replace spinner with dropdown for audio bitrate
        try:
            self.combo_abitrate = QtWidgets.QComboBox(self)
            _abrs = [24, 36, 48, 64, 112, 128, 160, 192, 256, 320]
            for _br in _abrs:
                self.combo_abitrate.addItem(str(_br))
            try:
                self.combo_abitrate.setCurrentText("192")
            except Exception:
                pass
            lay_enc.addWidget(self.combo_abitrate, 7, 1)
            try:
                self.spin_abitrate.hide()
            except Exception:
                pass
        except Exception:
            pass


        # --- Audio controls visibility & bitrate discretization ---
        try:
            # Cached label widgets from the grid layout by row/column
            self.lbl_acodec = lay_enc.itemAtPosition(6, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None
            self.lbl_abitrate = lay_enc.itemAtPosition(7, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None

            # Allowed audio bitrates
            self._allowed_abitrates = [24, 36, 48, 64, 112, 128, 160, 192, 256, 320]

            def _snap_abitrate(val: int):
                try:
                    target = min(self._allowed_abitrates, key=lambda x: (abs(x - int(val)), x))
                    if target != self.spin_abitrate.value():
                        old = self.spin_abitrate.blockSignals(True)
                        self.spin_abitrate.setValue(target)
                        self.spin_abitrate.blockSignals(old)
                except Exception:
                    pass

            try:
                self.spin_abitrate.setRange(min(self._allowed_abitrates), max(self._allowed_abitrates))
                self.spin_abitrate.setSingleStep(1)
                _snap_abitrate(self.spin_abitrate.value())
                self.spin_abitrate.valueChanged.connect(_snap_abitrate)
                if hasattr(self.spin_abitrate, "editingFinished"):
                    self.spin_abitrate.editingFinished.connect(lambda: _snap_abitrate(self.spin_abitrate.value()))
            except Exception:
                pass

            def _update_audio_controls():
                enc = False
                try:
                    enc = bool(self.radio_a_encode.isChecked())
                except Exception:
                    pass
                for w in (self.lbl_acodec, self.combo_acodec, self.lbl_abitrate, self.combo_abitrate):
                    try:
                        if w is not None:
                            w.setVisible(enc)
                    except Exception:
                        pass

            try:
                self.radio_a_copy.toggled.connect(_update_audio_controls)
                self.radio_a_encode.toggled.connect(_update_audio_controls)
                self.radio_a_mute.toggled.connect(_update_audio_controls)
            except Exception:
                pass

            _update_audio_controls()
        except Exception:
            pass
        # Encoder inline: layout already attached to self.box_encoder
        # (removed collapsible content setters)
        v.addWidget(self.box_encoder)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        try:
            fm = self.log.fontMetrics()
            self.log.setFixedHeight(int(fm.height() * 6.5))
        except Exception:
            self.log.setFixedHeight(100)
        self.log.setMaximumBlockCount(1000)
        if SHOW_UPSCALE_LOG:
            v.addWidget(self.log)
        else:
            self.log.hide()

        # Recent results (sticky, above Upscale / Batch / Info; collapsed by default)
        try:
            rec_body = QtWidgets.QWidget(self)
            rec_wrap = QtWidgets.QVBoxLayout(rec_body)
            rec_wrap.setContentsMargins(6, 2, 6, 6)
            rec_wrap.setSpacing(6)

            # Size + sort row
            size_row = QtWidgets.QHBoxLayout()

            # Sort dropdown (recent results)
            try:
                lbl_sort = QtWidgets.QLabel("Sort:", self)
            except Exception:
                lbl_sort = None
            try:
                self.combo_recent_sort = QtWidgets.QComboBox(self)
                self.combo_recent_sort.addItem("Newest first", "newest")
                self.combo_recent_sort.addItem("Oldest first", "oldest")
                self.combo_recent_sort.addItem("Alphabetical (A-Z)", "az")
                self.combo_recent_sort.addItem("Alphabetical (Z-A)", "za")
                self.combo_recent_sort.addItem("Size (smallest first)", "size_small")
                self.combo_recent_sort.addItem("Size (largest first)", "size_large")
            except Exception:
                self.combo_recent_sort = None

            try:
                if lbl_sort is not None:
                    size_row.addWidget(lbl_sort)
            except Exception:
                pass
            try:
                if self.combo_recent_sort is not None:
                    size_row.addWidget(self.combo_recent_sort)
            except Exception:
                pass

            size_row.addSpacing(12)
            size_row.addWidget(QtWidgets.QLabel("Thumb size:", self))
            self.sld_recent_size = QtWidgets.QSlider(Qt.Horizontal, self)
            self.sld_recent_size.setMinimum(50)
            self.sld_recent_size.setMaximum(180)
            self.sld_recent_size.setSingleStep(8)
            self.sld_recent_size.setPageStep(30)
            self.sld_recent_size.setValue(100)
            size_row.addWidget(self.sld_recent_size, 1)
            self.lbl_recent_size = QtWidgets.QLabel("100 px", self)
            size_row.addWidget(self.lbl_recent_size)
            rec_wrap.addLayout(size_row)

            # Scroll area with a wrapped grid of thumbnails
            self.recents_scroll = QtWidgets.QScrollArea(self)
            self.recents_scroll.setWidgetResizable(True)
            self.recents_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.recents_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            try:
                from PySide6.QtWidgets import QFrame as _QFrame
                self.recents_scroll.setFrameShape(_QFrame.NoFrame)
            except Exception:
                pass

            try:
                # Watch width changes so we can re-wrap items correctly
                self.recents_scroll.viewport().installEventFilter(self)
            except Exception:
                pass

            self._recents_inner = QtWidgets.QWidget(self)
            from PySide6.QtWidgets import QGridLayout as _QGridLayout
            self._recents_row = _QGridLayout(self._recents_inner)
            self._recents_row.setContentsMargins(0, 0, 0, 0)
            self._recents_row.setSpacing(8)
            self.recents_scroll.setWidget(self._recents_inner)
            rec_wrap.addWidget(self.recents_scroll)

            # Collapsible wrapper, closed by default (and stays closed after restart)
            self.recents_box = _Disclosure("Recent results", rec_body, start_open=False, parent=self)
            v_main.addWidget(self.recents_box)

            # Initial build + poller
            try:
                QTimer.singleShot(0, self._rebuild_recents)
            except Exception:
                pass
            try:
                self._install_recents_poller()
            except Exception:
                pass

            # Wire slider to resize thumbnails (resize in-place without rebuilding the list)
            def _on_recent_size(val):
                try:
                    self.lbl_recent_size.setText(f"{val} px")
                except Exception:
                    pass

                # Clamp and normalize size
                try:
                    size = int(val)
                except Exception:
                    size = 100
                if size < 40:
                    size = 40
                if size > 200:
                    size = 200

                # Resize existing buttons
                try:
                    layout = getattr(self, "_recents_row", None)
                    inner = getattr(self, "_recents_inner", None)
                    scroll = getattr(self, "recents_scroll", None)
                except Exception:
                    layout = inner = scroll = None

                if layout is not None:
                    try:
                        from PySide6 import QtWidgets as _QtW
                        from PySide6.QtCore import QSize as _QSize
                        for i in range(layout.count()):
                            item = layout.itemAt(i)
                            btn = item.widget() if item is not None else None
                            if isinstance(btn, _QtW.QToolButton):
                                try:
                                    btn.setIconSize(_QSize(int(size), int(size)))
                                    btn.setFixedSize(int(size * 1.25), int(size * 1.25) + 28)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Update scroll area height to account for new size
                try:
                    if layout is not None and inner is not None and scroll is not None:
                        spacing = getattr(layout, "spacing", lambda: 8)()
                        item_w = int(size * 1.25)
                        item_h = int(size * 1.25) + 28
                        try:
                            vpw = scroll.viewport().width()
                        except Exception:
                            try:
                                vpw = inner.width()
                            except Exception:
                                vpw = 0
                        if not vpw or vpw <= 1:
                            try:
                                vpw = max(scroll.width(), inner.width(), 600)
                            except Exception:
                                vpw = 600
                        cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                        total = layout.count()
                        rows = max(1, (total + cols - 1) // cols)
                        min_h = rows * item_h + max(0, rows - 1) * spacing + 12
                        try:
                            inner.setMinimumHeight(min_h)
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                self.sld_recent_size.valueChanged.connect(_on_recent_size)
            except Exception:
                pass

            # Wire sort dropdown to rebuild thumbnails
            try:
                if getattr(self, "combo_recent_sort", None) is not None:
                    def _on_recent_sort(_index):
                        try:
                            self._rebuild_recents()
                        except Exception:
                            pass
                    try:
                        self.combo_recent_sort.currentIndexChanged.connect(_on_recent_sort)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

        # ----- Fixed bottom action bar (does not scroll) -----
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.btn_upscale)
        bottom.addWidget(self.btn_batch)
        bottom.addWidget(self.btn_info)
        bottom.addStretch(1)
        v_main.addLayout(bottom)

        # Tooltips
        try:
            self.btn_upscale.setToolTip("Load any video or image in the media player, then select an engine and model and press 'Upscale'.")
            self.btn_batch.setToolTip("Add multiple files or a full folder to the queue for upscaling with the current settings.")
            self.btn_info.setToolTip("Shows detailed info (only works when you load an image or video directly in the Upscale tab).")
        except Exception:
            pass

        # -- Style tweaks: bigger bottom button fonts + Upscale hover blue
        try:
            # Make the bottom action buttons' font match the Describer actions
            for _b in (self.btn_upscale, self.btn_batch, self.btn_info):
                try:
                    _b.setMinimumHeight(32)
                except Exception:
                    pass
                _f = _b.font()
                try:
                    _sz = _f.pointSize()
                    if _sz <= 0:
                        _sz = 10
                except Exception:
                    _sz = 10
                # About +3pt for a clearly larger label, like in Describer
                _f.setPointSize(_sz + 3)
                _b.setFont(_f)
                try:
                    from PySide6.QtWidgets import QSizePolicy as _QSP
                    _b.setSizePolicy(_QSP.Preferred, _QSP.Fixed)
                except Exception:
                    pass
            # Hover background for Upscale button only
            self.btn_upscale.setObjectName("btn_upscale_main")
            self.btn_upscale.setStyleSheet((self.btn_upscale.styleSheet() or "") + "QPushButton:hover{background-color:#0d6efd;}")
        except Exception:
            pass


        # wiring
        self.btn_browse.clicked.connect(self._pick_single)
        self.btn_info.clicked.connect(self._show_media_info)
        self.btn_batch.clicked.connect(self._pick_batch)
        self.btn_upscale.clicked.connect(self._do_single)
        btn_out.clicked.connect(self._pick_outdir)
        self.spin_scale.valueChanged.connect(self._sync_scale_from_spin)
        self.slider_scale.valueChanged.connect(self._sync_scale_from_slider)
        self.combo_engine.currentIndexChanged.connect(self._update_engine_ui)
        self._update_engine_ui()
        try:
            self.combo_model_realsr.currentTextChanged.connect(self._update_engine_ui)
        except Exception:
            pass


    
        try:
            self.combo_model_realsr.currentTextChanged.connect(self._update_model_hint)
        except Exception:
            pass
        try:
            self.btn_auto_tile.clicked.connect(self._auto_tile_size)
        except Exception:
            pass
        self._update_model_hint()

        # Logger buffering to avoid UI stalls when many lines arrive quickly
        try:
            self._log_buf = []
            self._log_flush_timer = QTimer(self)
            self._log_flush_timer.setSingleShot(True)
            self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        except Exception:
            pass


        # Load persisted settings (if any)
        try:
            self._load_settings()
        except Exception:
            pass

        # Ensure batch limit matches current scale
        try:
            self._update_batch_limit()
        except Exception:
            pass

        # Wire quick buttons
        try:
            self.btn_scale_2x.clicked.connect(lambda: self._set_scale_quick(2.0))
            self.btn_scale_3x.clicked.connect(lambda: self._set_scale_quick(3.0))
            self.btn_scale_4x.clicked.connect(lambda: self._set_scale_quick(4.0))
        except Exception:
            pass

        
        # Save on application exit too (best-effort)
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                app.aboutToQuit.connect(self._auto_save_on_exit)
        except Exception:
            pass

        # Wire auto-save on change
        try:
            self._connect_auto_save()
        except Exception:
            pass

# UI helpers
    def _set_section_state(self, box, open_: bool):
        """Reliably set a CollapsibleSection open/closed, even if setChecked() is ignored.
        Uses toggle.click() when the current state differs, which triggers the same logic as a user click.
        """
        try:
            tb = getattr(box, "toggle", None)
            if tb is None:
                return
            want = bool(open_)
            cur = bool(tb.isChecked())
            if cur != want:
                tb.click()
        except Exception:
            pass


    def _set_scale_quick(self, val: float):
        try:
            self.spin_scale.setValue(float(val))
            self._sync_scale_from_spin(float(val))
        except Exception:
            pass

    def _settings_path(self) -> Path:
        try:
            return (ROOT / "presets" / "setsave" / "upsc_settings.json")
        except Exception:
            return Path("presets/setsave/upsc_settings.json").resolve()
    def eventFilter(self, obj, ev):
        try:
            return super().eventFilter(obj, ev)
        except Exception:
            return False



    def _connect_auto_save(self):
        """Connect change signals to auto-save, safely and individually, so a missing widget won't break others."""
        def _connect(obj, sig_name):
            try:
                if obj is None:
                    return False
                sig = getattr(obj, sig_name, None)
                if sig is None:
                    return False
                sig.connect(self._auto_save_if_enabled)
                return True
            except Exception:
                return False

        # Standard widgets
        pairs = [
            ("spin_scale", "valueChanged"),
            ("slider_scale", "valueChanged"),
            ("combo_engine", "currentIndexChanged"),
            ("combo_model_realsr", "currentTextChanged"),
            ("combo_model_w2x", "currentTextChanged"),
            ("edit_outdir", "textChanged"),
            ("chk_play_internal", "toggled"),
            ("chk_replace_in_player", "toggled"),
            ("combo_vcodec", "currentTextChanged"),
            ("rad_crf", "toggled"),
            ("spin_crf", "valueChanged"),
            ("rad_bitrate", "toggled"),
            ("spin_bitrate", "valueChanged"),
            ("combo_preset", "currentTextChanged"),
            ("spin_keyint", "valueChanged"),
            ("radio_a_copy", "toggled"),
            ("radio_a_encode", "toggled"),
            ("radio_a_mute", "toggled"),
            ("combo_acodec", "currentTextChanged"),
            ("combo_abitrate", "currentTextChanged"),
            ("spin_tile", "valueChanged"),
            ("spin_overlap", "valueChanged"),
            ("chk_deinterlace", "toggled"),
            ("combo_range", "currentTextChanged"),
            ("chk_deblock", "toggled"),
            ("chk_denoise", "toggled"),
            ("chk_deband", "toggled"),
            ("sld_sharpen", "valueChanged"),
            ("sld_recent_size", "valueChanged"),
            ("combo_recent_sort", "currentIndexChanged"),
                    ]

        connected = 0
        for name, sig_name in pairs:
            obj = getattr(self, name, None)
            if _connect(obj, sig_name):
                connected += 1

        # Collapsible sections (their .toggle has a .toggled signal)
        for box_name in ("box_models", "box_encoder", "box_advanced"):
            box = getattr(self, box_name, None)
            toggle = getattr(box, "toggle", None) if box is not None else None
            if _connect(toggle, "toggled"):
                connected += 1

        try:
            self._append_log(f"[settings] Auto-save wired to {connected} UI change signal(s).")
        except Exception:
            pass


# UI helpers

    def _auto_save_if_enabled(self, *args, **kwargs):
        try:
            self._save_settings()
        except Exception:
            pass


    def _gather_settings(self) -> dict:
        d = {}
        # Core toggles / basics
        try: d['remember'] = True
        except Exception: pass
        try: d["scale"] = float(self.spin_scale.value())
        except Exception: pass
        try: d["engine_index"] = int(self.combo_engine.currentIndex())
        except Exception: pass
        try: d["model_realsr"] = self.combo_model_realsr.currentText()
        except Exception: pass
        try: d["model_w2x"] = self.combo_model_w2x.currentText()
        except Exception: pass
        try: d["outdir"] = self.edit_outdir.text().strip()
        except Exception: pass
        try:
            d["play_internal"] = bool(self.chk_play_internal.isChecked())
            d["replace_in_player"] = bool(self.chk_replace_in_player.isChecked())
        except Exception: pass
        # Video encode settings
        try: d["vcodec"] = self.combo_vcodec.currentText()
        except Exception: pass
        try:
            d["use_crf"] = bool(self.rad_crf.isChecked())
            d["crf"] = int(self.spin_crf.value())
            d["use_bitrate"] = bool(self.rad_bitrate.isChecked())
            d["bitrate"] = int(self.spin_bitrate.value())
        except Exception: pass
        try: d["preset"] = self.combo_preset.currentText()
        except Exception: pass
        try: d["keyint"] = 0
        except Exception: pass
        # Audio
        try:
            if self.radio_a_mute.isChecked():
                d["audio"] = "mute"
            elif self.radio_a_encode.isChecked():
                d["audio"] = "encode"
            else:
                d["audio"] = "copy"
        except Exception: pass
        try: d["acodec"] = self.combo_acodec.currentText()
        except Exception: pass
        try: d["abitrate"] = int(self.combo_abitrate.currentText())
        except Exception: pass
        # Advanced / tiles
        try:
            d["tile"] = int(self.spin_tile.value())
            d["overlap"] = int(self.spin_overlap.value())
        except Exception: pass
        try: d["deinterlace"] = bool(self.chk_deinterlace.isChecked())
        except Exception: pass
        try: d["range"] = self.combo_range.currentText()
        except Exception: pass
        try:
            d["deblock"] = bool(self.chk_deblock.isChecked())
            d["denoise"] = bool(self.chk_denoise.isChecked())
            d["deband"] = bool(self.chk_deband.isChecked())
        except Exception: pass
        try: d["sharpen"] = int(self.sld_sharpen.value())
        except Exception: pass
        # Sections collapsed state
        d["sections"] = {}
        try:
            d["sections"]["models"] = bool(self.box_models.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["models"] = bool(getattr(self.box_models, "isExpanded", lambda: True)())
            except Exception:
                d["sections"]["models"] = True
        try:
            d["sections"]["encoder"] = bool(self.box_encoder.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["encoder"] = bool(getattr(self.box_encoder, "isExpanded", lambda: True)())
            except Exception:
                d["sections"]["encoder"] = True
        try:
            d["sections"]["advanced"] = bool(self.box_advanced.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["advanced"] = bool(getattr(self.box_advanced, "isExpanded", lambda: False)())
            except Exception:
                d["sections"]["advanced"] = False
        # Batch / gallery options
        try: d["video_thumbs"] = bool(self.chk_video_thumbs.isChecked())
        except Exception: d["video_thumbs"] = False
        try: d["streaming_lowmem"] = bool(self.chk_streaming_lowmem.isChecked())
        except Exception: d["streaming_lowmem"] = True

        # Recent results options
        try:
            d["recents_thumb_size"] = int(self.sld_recent_size.value())
        except Exception:
            pass
        try:
            sort = None
            try:
                sort = self.combo_recent_sort.currentData()
            except Exception:
                sort = None
            if sort is None:
                try:
                    if getattr(self, "combo_recent_sort", None) is not None:
                        sort = self.combo_recent_sort.currentText()
                except Exception:
                    sort = None
            if sort:
                d["recents_sort"] = str(sort)
        except Exception:
            pass
        return d

    def _apply_settings(self, d: dict):
        try:
            # remember toggle removed from UI; always on
            self.chk_remember.setChecked(True)
        except Exception:
            pass
        try:
            self.spin_scale.setValue(float(d.get("scale", 2.0)))
        except Exception:
            pass
        try:
            idx = int(d.get("engine_index", 0))
            if 0 <= idx < self.combo_engine.count():
                self.combo_engine.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            m = d.get("model_realsr")
            if m:
                i = self.combo_model_realsr.findText(m)
                if i >= 0: self.combo_model_realsr.setCurrentIndex(i)
        except Exception:
            pass
        try:
            m = d.get("model_w2x")
            if m:
                i = self.combo_model_w2x.findText(m)
                if i >= 0: self.combo_model_w2x.setCurrentIndex(i)
        except Exception:
            pass
        try:
            out = d.get("outdir")
            if out: self.edit_outdir.setText(out)
        except Exception:
            pass
        try:
            self.chk_play_internal.setChecked(bool(d.get("play_internal", True)))
            self.chk_replace_in_player.setChecked(bool(d.get("replace_in_player", True)))
        except Exception:
            pass
        try:
            vc = d.get("vcodec"); 
            if vc:
                i = self.combo_vcodec.findText(vc)
                if i >= 0: self.combo_vcodec.setCurrentIndex(i)
        except Exception:
            pass
        try:
            if d.get("use_crf", True):
                self.rad_crf.setChecked(True)
            if d.get("use_bitrate", False):
                self.rad_bitrate.setChecked(True)
            self.spin_crf.setValue(int(d.get("crf", 18)))
            self.spin_bitrate.setValue(int(d.get("bitrate", 8000)))
        except Exception:
            pass
        try:
            pr = d.get("preset")
            if pr:
                i = self.combo_preset.findText(pr)
                if i >= 0: self.combo_preset.setCurrentIndex(i)
        except Exception:
            pass
        try:
            self.spin_keyint.setValue(0)
        except Exception:
            pass
        try:
            a = d.get("audio", "copy")
            if a == "mute": self.radio_a_mute.setChecked(True)
            elif a == "encode": self.radio_a_encode.setChecked(True)
            else: self.radio_a_copy.setChecked(True)
        except Exception:
            pass
        try:
            ac = d.get("acodec")
            if ac:
                i = self.combo_acodec.findText(ac)
                if i >= 0: self.combo_acodec.setCurrentIndex(i)
            (_i := self.combo_abitrate.findText(str(d.get("abitrate", 192)))); self.combo_abitrate.setCurrentIndex(_i if _i >= 0 else self.combo_abitrate.findText("192"))
        except Exception:
            pass
        try:
            self.spin_tile.setValue(int(d.get("tile", 0)))
            self.spin_overlap.setValue(int(d.get("overlap", 0)))
        except Exception:
            pass
        try:
            self.chk_deinterlace.setChecked(bool(d.get("deinterlace", False)))
            rng = d.get("range")
            if rng:
                i = self.combo_range.findText(rng)
                if i >= 0: self.combo_range.setCurrentIndex(i)
            self.chk_deblock.setChecked(bool(d.get("deblock", False)))
            self.chk_denoise.setChecked(bool(d.get("denoise", False)))
            self.chk_deband.setChecked(bool(d.get("deband", False)))
            self.sld_sharpen.setValue(int(d.get("sharpen", 0)))
        except Exception:
            pass
        try:
            secs = d.get("sections", {})
            try:
                self._append_log(f"[settings] Loaded section states: {secs}")
            except Exception:
                pass
            self._set_section_state(self.box_models, bool(secs.get("models", True)))
            self._set_section_state(self.box_encoder, bool(secs.get("encoder", True)))
            self._set_section_state(self.box_advanced, bool(secs.get("advanced", False)))
        except Exception:
            pass
        try:
            self.chk_video_thumbs.setChecked(bool(d.get("video_thumbs", False)))
        except Exception:
            pass
        # Recent results options
        try:
            sz = int(d.get("recents_thumb_size", 100))
            if hasattr(self, "sld_recent_size"):
                try:
                    self.sld_recent_size.setValue(sz)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            sort = d.get("recents_sort")
            cb = getattr(self, "combo_recent_sort", None)
            if cb is not None and sort is not None:
                idx = -1
                try:
                    for i in range(cb.count()):
                        v = cb.itemData(i)
                        if v == sort:
                            idx = i
                            break
                except Exception:
                    idx = -1
                if idx < 0:
                    try:
                        idx = cb.findText(str(sort))
                    except Exception:
                        idx = -1
                if idx >= 0:
                    try:
                        cb.setCurrentIndex(idx)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.chk_streaming_lowmem.setChecked(bool(d.get("streaming_lowmem", True)))
        except Exception:
            pass
        except Exception:
            pass

    def _save_settings(self):
        try:
            d = self._gather_settings()
            p = self._settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
            try:
                self._append_log(f"[settings] Saved to: {p}")
            except Exception:
                pass
        except Exception as e:
            try:
                self._append_log(f"[settings] Save failed: {e}")
            except Exception:
                pass

    def _load_settings(self):
        try:
            p = self._settings_path()
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                try:
                    self._append_log(f"[settings] Loaded from: {p}")
                except Exception:
                    pass
                # Apply settings, then ensure remember is True (hidden)
                self._apply_settings(d)
                try:
                    self.chk_remember.setChecked(True)
                except Exception:
                    pass
            else:
                try:
                    self._append_log(f"[settings] No file yet at: {p}")
                except Exception:
                    pass
                self._recents = []
        except Exception as e:
            self._recents = []
            try:
                self._append_log(f"[settings] Load failed: {e}")
            except Exception:
                pass

    def _update_batch_limit(self):
        """Batch limit control removed; keeping method as no-op for compatibility."""
        return

    def _recents_dir(self):
        """Return Path to the Upscale recent-thumbnail folder."""
        from pathlib import Path as _Path
        try:
            try:
                base = ROOT  # type: ignore[name-defined]
            except Exception:
                base = _Path(__file__).resolve().parent.parent
            d = base / "output" / "last results" / "upsc"
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return d
        except Exception:
            # Fallback to relative path
            return _Path("output") / "last results" / "upsc"

    def _thumb_path_for_media(self, media_path, max_side: int = 120):
        """Return a Path under _recents_dir used to store a thumbnail for *media_path*."""
        from pathlib import Path as _P
        import hashlib

        d = self._recents_dir()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            stem = _P(media_path).stem
        except Exception:
            stem = "media"

        try:
            key = str(media_path)
            h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        except Exception:
            h = "thumb"

        return d / f"{stem}_{h}_{max_side}.jpg"

    def _ensure_recent_thumb_for_media(self, media_path, max_side: int = 120):
        """Create (or reuse) a thumbnail in _recents_dir for the given media file."""
        from pathlib import Path as _P

        media_path = _P(media_path)
        try:
            if not (media_path.exists() and media_path.is_file()):
                return None
        except Exception:
            return None

        thumb = self._thumb_path_for_media(media_path, max_side=max_side)
        try:
            if thumb.exists() and thumb.stat().st_mtime >= media_path.stat().st_mtime:
                return thumb
        except Exception:
            # If we cannot stat, fall through and try to rebuild
            pass

        ext = media_path.suffix.lower()
        img = None

        # Treat anything that is NOT a known video as an image first
        if ext not in _VIDEO_EXTS:
            try:
                from PySide6.QtGui import QImageReader
                from PySide6.QtCore import QSize as _QSize
                reader = QImageReader(str(media_path))
                reader.setAutoTransform(True)
                sz = reader.size()
                if sz.isValid():
                    w, h = sz.width(), sz.height()
                    if w > 0 and h > 0:
                        scale = max(w, h) / float(max_side or 1)
                        if scale > 1.0:
                            w = int(w / scale)
                            h = int(h / scale)
                            reader.setScaledSize(_QSize(max(16, w), max(16, h)))
                img = reader.read()
                if img.isNull():
                    img = None
            except Exception:
                img = None

        # Videos, or image-reader failure: fall back to _make_thumb
        if img is None and ext in _VIDEO_EXTS:
            try:
                pm = self._make_thumb(media_path)
            except Exception:
                pm = QPixmap()
            try:
                if pm and not pm.isNull():
                    pm2 = pm.scaled(int(max_side), int(max_side), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    pm2 = None
            except Exception:
                pm2 = None
            try:
                if pm2 is not None:
                    img = pm2.toImage()
            except Exception:
                img = None

        if img is None:
            return None

        try:
            thumb.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            img.save(str(thumb), "JPG", 88)
            return thumb
        except Exception:
            return None

    def _resolve_media_for_thumb(self, thumb_path):
        """Best-effort: given a thumbnail path, return the original media path."""
        from pathlib import Path as _P

        try:
            t = _P(str(thumb_path))
        except Exception:
            return thumb_path

        # 1) In-memory mapping (new thumbnails in this session)
        try:
            mapping = getattr(self, "_recents_thumb_map", {}) or {}
            orig = mapping.get(str(t))
            if orig:
                p = _P(str(orig))
                if p.exists() and p.is_file():
                    return p
        except Exception:
            pass

        # 2) Parse the thumbnail filename: stem_hash_size.jpg -> stem
        try:
            name_stem = t.stem
            parts = name_stem.rsplit("_", 2)
            if len(parts) == 3 and parts[1] and parts[2]:
                base_stem = parts[0]
            else:
                base_stem = name_stem
        except Exception:
            base_stem = None

        if not base_stem:
            return thumb_path

        media_exts = tuple(_IMAGE_EXTS | _VIDEO_EXTS)

        # Build a list of candidate folders to search
        dirs = []
        try:
            edit = getattr(self, "edit_outdir", None)
            out_dir = Path(edit.text().strip()) if (edit is not None and edit.text().strip()) else None
        except Exception:
            out_dir = None
        if out_dir and out_dir.exists():
            dirs.append(out_dir)
        try:
            if OUT_VIDEOS not in dirs:
                dirs.append(OUT_VIDEOS)
        except Exception:
            pass
        try:
            if OUT_SHOTS not in dirs:
                dirs.append(OUT_SHOTS)
        except Exception:
            pass

        for d in dirs:
            try:
                if not d.exists():
                    continue
                for p in d.iterdir():
                    try:
                        if not p.is_file():
                            continue
                        if p.suffix.lower() not in media_exts:
                            continue
                        if p.stem == base_stem:
                            return p
                    except Exception:
                        continue
            except Exception:
                continue

        return thumb_path

    def _list_recent_files(self):
        """List recent result thumbnail files for Upscale (most recent first).

        This merges three sources:
        - Existing thumbnails under output/last results/upsc
        - New images/videos from the usual Upscale output folders
        - Finished Upscale queue jobs (from jobs/finished or jobs/done)
        """
        from pathlib import Path as _Path
        exts = set(_IMAGE_EXTS | _VIDEO_EXTS)
        candidates = []

        # 1) Existing thumbs under the recents dir
        try:
            thumbs_dir = self._recents_dir()
            if thumbs_dir and thumbs_dir.exists():
                for p in thumbs_dir.iterdir():
                    try:
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
                            candidates.append(p)
                    except Exception:
                        continue
        except Exception:
            pass

        # 2) Direct Upscale output folders (non-queued runs)
        try:
            outs = []
            try:
                edit = getattr(self, "edit_outdir", None)
                if edit is not None:
                    text = (edit.text() or "").strip()
                    if text:
                        p = _Path(text).expanduser()
                        if p.exists() and p.is_dir():
                            outs.append(p)
            except Exception:
                pass
            try:
                if OUT_VIDEOS not in outs:
                    outs.append(OUT_VIDEOS)
            except Exception:
                pass
            try:
                if OUT_SHOTS not in outs:
                    outs.append(OUT_SHOTS)
            except Exception:
                pass
            # Deduplicate
            unique_outs = []
            seen = set()
            for d in outs:
                try:
                    key = str(_Path(d).resolve())
                except Exception:
                    key = str(d)
                if key in seen:
                    continue
                seen.add(key)
                unique_outs.append(_Path(d))

            for out_dir in unique_outs:
                try:
                    if not out_dir.exists():
                        continue
                    medias = [
                        p for p in out_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in exts
                    ]
                    medias.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    for media in medias[:40]:
                        tp = self._ensure_recent_thumb_for_media(media, max_side=120)
                        if tp:
                            candidates.append(tp)
                            try:
                                m = getattr(self, "_recents_thumb_map", {}) or {}
                                m[str(tp)] = str(media)
                                self._recents_thumb_map = m
                            except Exception:
                                pass
                except Exception:
                    continue
        except Exception:
            pass

        # 3) Finished queue jobs (Upscale jobs only)
        try:
            jobs = self._list_recent_upscale_jobs()
            for job_json in jobs:
                media, _j = self._resolve_output_from_upscale_job(job_json)
                if not media:
                    continue
                tp = self._ensure_recent_thumb_for_media(media, max_side=120)
                if tp:
                    candidates.append(tp)
                    try:
                        m = getattr(self, "_recents_thumb_map", {}) or {}
                        m[str(tp)] = str(media)
                        self._recents_thumb_map = m
                    except Exception:
                        pass
        except Exception:
            pass

        # Final sort on the thumbnail files themselves
        try:
            candidates = [p for p in candidates if p is not None]
            # dedupe by path string
            tmp = {}
            for p in candidates:
                try:
                    key = str(_Path(p))
                except Exception:
                    key = str(p)
                tmp[key] = p
            candidates = list(tmp.values())
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[:48]
        except Exception:
            return []

    def _jobs_done_dirs(self):
        """Return a list of job result folders to scan for finished Upscale queue jobs."""
        from pathlib import Path as _Path
        try:
            try:
                base = ROOT  # type: ignore[name-defined]
            except Exception:
                base = _Path(__file__).resolve().parent.parent
            roots = []
            for name in ("finished", "done"):
                try:
                    d = base / "jobs" / name
                    if d.exists() and d.is_dir():
                        roots.append(d)
                except Exception:
                    continue
            return roots
        except Exception:
            return []

    def _list_recent_upscale_jobs(self):
        """List finished Upscale job JSON files (newest first)."""
        from pathlib import Path as _Path
        jobs = []
        try:
            for d in self._jobs_done_dirs():
                try:
                    for p in d.iterdir():
                        try:
                            if p.is_file() and p.suffix.lower() == ".json" and "upscale" in p.name.lower():
                                jobs.append(p)
                        except Exception:
                            continue
                except Exception:
                    continue
            jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return jobs[:64]
        except Exception:
            return jobs

    def _resolve_output_from_upscale_job(self, job_json):
        """Given an Upscale job JSON file, return (media_path, job_dict)."""
        from pathlib import Path as _Path
        import json as _json
        try:
            jp = _Path(job_json)
            with jp.open("r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            return None, None
        try:
            if data.get("category") not in ("upscale", "Upscale", "UPSC", "upsc") and "upsc" not in str(data.get("name", "")).lower():
                # Not an Upscale job; ignore
                return None, data
        except Exception:
            pass
        media = None
        try:
            out = data.get("output") or data.get("outfile") or data.get("out") or data.get("result")
            if out:
                media = _Path(out)
        except Exception:
            media = None
        return media, data


    def _install_recents_poller(self):
        """Poll the Upscale recents folder every few seconds; rebuild UI on change."""
        try:
            if getattr(self, "_recents_poller", None):
                return

            def _sig():
                try:
                    files = self._list_recent_files()
                    return tuple((p.name, int(p.stat().st_mtime)) for p in files)
                except Exception:
                    return tuple()

            def _tick():
                try:
                    cur = _sig()
                    if cur != getattr(self, "_recents_sig", None):
                        self._recents_sig = cur
                        self._rebuild_recents()
                except Exception:
                    pass

            try:
                self._recents_sig = None
                _tick()
            except Exception:
                pass

            t = QTimer(self)
            t.setInterval(5000)
            t.timeout.connect(_tick)
            t.start()
            self._recents_poller = t
        except Exception:
            pass

    def _add_recent(self, media):
        """Record a freshly produced output in recents (thumbnail + in-memory map)."""
        from pathlib import Path as _P
        try:
            if not media:
                return
            p = _P(str(media))
            if not (p.exists() and p.is_file()):
                return
        except Exception:
            return

        try:
            size_slider = getattr(self, "sld_recent_size", None)
            size = int(size_slider.value()) if size_slider is not None else 100
        except Exception:
            size = 100

        thumb = self._ensure_recent_thumb_for_media(p, max_side=size)
        if not thumb:
            return
        try:
            mapping = getattr(self, "_recents_thumb_map", {}) or {}
            mapping[str(thumb)] = str(p)
            self._recents_thumb_map = mapping
        except Exception:
            pass

    def _rebuild_recents(self):
        """Rebuild the Recent results grid from thumbnail files."""
        try:
            layout = getattr(self, "_recents_row", None)
            inner = getattr(self, "_recents_inner", None)
            scroll = getattr(self, "recents_scroll", None)
            if layout is None or inner is None or scroll is None:
                return

            # Clear existing widgets
            try:
                while layout.count():
                    item = layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.setParent(None)
            except Exception:
                pass

            # Thumb size from slider (clamped)
            try:
                size_slider = getattr(self, "sld_recent_size", None)
                size = int(size_slider.value()) if size_slider is not None else 100
            except Exception:
                size = 100
            if size < 40:
                size = 40
            if size > 200:
                size = 200

            files = self._list_recent_files()
            if not files:
                lab = QtWidgets.QLabel("No results yet.", self)
                try:
                    lab.setStyleSheet("color:#9fb3c8;")
                except Exception:
                    pass
                layout.addWidget(lab, 0, 0)
                try:
                    inner.setMinimumHeight(lab.sizeHint().height() + 8)
                except Exception:
                    pass
                return

            # Helper functions for sorting by underlying media
            def _media_for_sort(thumb_path):
                try:
                    return self._resolve_media_for_thumb(thumb_path)
                except Exception:
                    return thumb_path

            def _mtime_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    if mp.exists():
                        return mp.stat().st_mtime
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).stat().st_mtime
                except Exception:
                    return 0

            def _name_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    return mp.name.lower()
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).name.lower()
                except Exception:
                    return str(thumb_path)

            def _size_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    if mp.exists():
                        return mp.stat().st_size
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).stat().st_size
                except Exception:
                    return 0

            # Determine sort mode from combo box
            try:
                mode = None
                cb = getattr(self, "combo_recent_sort", None)
                if cb is not None:
                    mode = cb.currentData()
                    if not mode:
                        mode = cb.currentText()
                if not mode:
                    mode = "newest"
            except Exception:
                mode = "newest"

            # Apply sorting
            try:
                if mode in ("newest", "oldest"):
                    files.sort(key=_mtime_for, reverse=(mode == "newest"))
                elif mode in ("az", "za"):
                    files.sort(key=_name_for, reverse=(mode == "za"))
                elif mode in ("size_small", "size_large"):
                    files.sort(key=_size_for, reverse=(mode == "size_large"))
            except Exception:
                # Fallback: newest first by thumbnail mtime
                try:
                    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                except Exception:
                    pass

            setattr(self, "_recents_idx", 0)
            for p in files:
                btn = QtWidgets.QToolButton(self)
                btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                try:
                    btn.setText(p.name)
                except Exception:
                    pass
                btn.setCursor(Qt.PointingHandCursor)
                btn.setAutoRaise(True)
                try:
                    btn.setStyleSheet(
                        "QToolButton { border-radius: 10px; padding: 4px 2px; }"
                        "QToolButton:hover { background: rgba(255,255,255,0.06); }"
                    )
                except Exception:
                    pass

                # Thumbnail icon
                try:
                    pm = QPixmap(str(p))
                except Exception:
                    pm = QPixmap()
                if pm and not pm.isNull():
                    try:
                        pm2 = pm.scaled(int(size), int(size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    except Exception:
                        pm2 = pm
                    try:
                        pm2 = _rounded_pixmap(pm2, 10)
                    except Exception:
                        pass
                    try:
                        btn.setIcon(QIcon(pm2))
                    except Exception:
                        pass
                try:
                    btn.setIconSize(QSize(int(size), int(size)))
                    btn.setFixedSize(int(size * 1.25), int(size * 1.25) + 28)
                except Exception:
                    pass

                # Tooltip with underlying media path (best effort)
                try:
                    media_tp = _media_for_sort(p)
                    from pathlib import Path as _P
                    btn.setToolTip(str(_P(str(media_tp))))
                except Exception:
                    pass

                # Left-click: open result
                def _mk_open(thumb_path):
                    def _open():
                        try:
                            media = self._resolve_media_for_thumb(thumb_path)
                        except Exception:
                            media = thumb_path
                        try:
                            if not self._play_in_player(media):
                                self._open_file(media)
                        except Exception:
                            try:
                                self._open_file(media)
                            except Exception:
                                pass
                    return _open
                try:
                    btn.clicked.connect(_mk_open(p))
                except Exception:
                    pass

                # Right-click: context menu with "Delete from disk"
                def _mk_ctx(thumb_path, button):
                    def _on_menu(pos):
                        from pathlib import Path as _P
                        import os as _os
                        try:
                            menu = QtWidgets.QMenu(button)
                        except Exception:
                            return

                        # Pre-resolve the underlying media, so all actions share it
                        try:
                            media = self._resolve_media_for_thumb(thumb_path)
                        except Exception:
                            media = thumb_path
                        try:
                            mp = _P(str(media))
                        except Exception:
                            mp = None

                        try:
                            act_info = menu.addAction("Info")
                        except Exception:
                            act_info = None
                        try:
                            act_rename = menu.addAction("Rename")
                        except Exception:
                            act_rename = None
                        try:
                            act_open = menu.addAction("Open folder")
                        except Exception:
                            act_open = None
                        try:
                            menu.addSeparator()
                        except Exception:
                            pass
                        try:
                            act_del = menu.addAction("Delete")
                        except Exception:
                            act_del = None

                        try:
                            global_pos = button.mapToGlobal(pos)
                        except Exception:
                            try:
                                global_pos = None
                            except Exception:
                                global_pos = None
                        try:
                            chosen = menu.exec(global_pos) if global_pos is not None else menu.exec()
                        except Exception:
                            chosen = None
                        if not chosen:
                            return

                        # Info: show ffprobe JSON for this media
                        if chosen is act_info and act_info is not None:
                            try:
                                if mp is not None and mp.exists():
                                    self._show_media_info_for(mp)
                                else:
                                    QtWidgets.QMessageBox.information(
                                        self,
                                        "Info",
                                        "File no longer exists on disk.",
                                    )
                            except Exception:
                                pass
                            return

                        # Rename: change the underlying media file name and keep the thumbnail in sync
                        if chosen is act_rename and act_rename is not None:
                            try:
                                if mp is None or not mp.exists():
                                    QtWidgets.QMessageBox.warning(
                                        self,
                                        "Rename",
                                        "File no longer exists on disk.",
                                    )
                                    return
                                cur_stem = mp.stem
                                new_name, ok = QtWidgets.QInputDialog.getText(
                                    self,
                                    "Rename file",
                                    "New name (without extension):",
                                    text=cur_stem,
                                )
                                if not ok:
                                    return
                                new_name = (new_name or "").strip()
                                if not new_name or new_name == cur_stem:
                                    return
                                new_path = mp.with_name(new_name + mp.suffix)
                                if new_path.exists():
                                    QtWidgets.QMessageBox.warning(
                                        self,
                                        "Rename",
                                        "A file with that name already exists.",
                                    )
                                    return
                                try:
                                    mp.rename(new_path)
                                    mp = new_path
                                    media = new_path
                                except Exception:
                                    QtWidgets.QMessageBox.warning(
                                        self,
                                        "Rename failed",
                                        "Could not rename the file on disk.",
                                    )
                                    return
                                # Try to rename the thumbnail file so the label matches
                                try:
                                    tp = _P(str(thumb_path))
                                except Exception:
                                    tp = None
                                new_tp = None
                                if tp is not None and tp.exists():
                                    try:
                                        stem = tp.stem
                                        parts = stem.rsplit("_", 2)
                                        if len(parts) == 3:
                                            # stem_hash_size -> preserve hash+size
                                            _, hash_part, size_part = parts
                                            new_stem = new_path.stem
                                            new_base = f"{new_stem}_{hash_part}_{size_part}"
                                        else:
                                            new_base = new_path.stem
                                        new_tp = tp.with_name(new_base + tp.suffix)
                                        if new_tp != tp:
                                            try:
                                                tp.rename(new_tp)
                                            except Exception:
                                                new_tp = tp
                                    except Exception:
                                        new_tp = tp
                                else:
                                    new_tp = tp
                                # Update in-memory mapping
                                try:
                                    mapping = getattr(self, "_recents_thumb_map", {}) or {}
                                    old_key = str(thumb_path)
                                    new_key = str(new_tp) if new_tp is not None else old_key
                                    val = mapping.pop(old_key, None)
                                    if val is None:
                                        val = str(new_path)
                                    mapping[new_key] = str(new_path)
                                    self._recents_thumb_map = mapping
                                except Exception:
                                    pass
                                try:
                                    self._rebuild_recents()
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            return

                        # Open folder: reveal the file in its folder
                        if chosen is act_open and act_open is not None:
                            try:
                                if mp is None or not mp.exists():
                                    QtWidgets.QMessageBox.warning(
                                        self,
                                        "Open folder",
                                        "File no longer exists on disk.",
                                    )
                                    return
                                folder = mp.parent
                                try:
                                    if _os.name == "nt":
                                        # On Windows, try to select the file in Explorer
                                        try:
                                            import subprocess as _sub
                                            _sub.Popen(["explorer", "/select,", str(mp)])
                                        except Exception:
                                            _os.startfile(str(folder))  # nosec - user initiated
                                    else:
                                        import subprocess as _sub
                                        _sub.Popen(["xdg-open", str(folder)])
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            return

                        # Delete: unload then remove media+thumbnail from disk
                        if chosen is not act_del or act_del is None:
                            return

                        # Confirm deletion
                        try:
                            fname = mp.name if mp is not None else str(media)
                        except Exception:
                            fname = str(media)
                        try:
                            res = QtWidgets.QMessageBox.question(
                                self,
                                "Delete upscaled file?",
                                f"Delete this upscaled file from disk?\n\n{fname}",
                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                QtWidgets.QMessageBox.No,
                            )
                        except Exception:
                            res = QtWidgets.QMessageBox.Yes
                        if res != QtWidgets.QMessageBox.Yes:
                            return

                        # Best-effort: unload from player/viewer before deleting
                        try:
                            if mp is not None:
                                self._try_unload_media_before_delete(mp)
                        except Exception:
                            pass

                        # Delete media file
                        delete_ok = True
                        try:
                            if mp is not None and mp.exists():
                                mp.unlink()
                        except Exception:
                            delete_ok = False
                            try:
                                QtWidgets.QMessageBox.warning(
                                    self,
                                    "Delete failed",
                                    "Could not delete file. Please load another recent result, then try again.",
                                )
                            except Exception:
                                pass

                        if not delete_ok:
                            return

                        # Delete thumbnail file
                        try:
                            tp = _P(str(thumb_path))
                            if tp.exists():
                                tp.unlink()
                        except Exception:
                            pass

                        # Remove from in-memory mapping
                        try:
                            mapping = getattr(self, "_recents_thumb_map", {}) or {}
                            mapping.pop(str(thumb_path), None)
                            self._recents_thumb_map = mapping
                        except Exception:
                            pass

                        # Rebuild grid
                        try:
                            self._rebuild_recents()
                        except Exception:
                            pass
                    return _on_menu

                try:
                    btn.setContextMenuPolicy(Qt.CustomContextMenu)
                    btn.customContextMenuRequested.connect(_mk_ctx(p, btn))
                except Exception:
                    pass

                # Grid placement with wrapping
                try:
                    try:
                        vpw = scroll.viewport().width()
                    except Exception:
                        vpw = inner.width()
                    if not vpw or vpw <= 1:
                        vpw = max(scroll.width(), self.width(), 600)
                except Exception:
                    vpw = 600
                try:
                    spacing = getattr(layout, "spacing", lambda: 8)()
                except Exception:
                    spacing = 8
                item_w = int(size * 1.25)
                if vpw <= item_w + spacing and len(files) > 1:
                    cols = min(len(files), 4)
                else:
                    cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                idx = getattr(self, "_recents_idx", 0)
                row = idx // cols
                col = idx % cols
                setattr(self, "_recents_idx", idx + 1)
                try:
                    layout.addWidget(btn, row, col)
                except Exception:
                    pass

            # Ensure the scroll area can expand vertically if needed
            try:
                spacing = getattr(layout, "spacing", lambda: 8)()
                item_w = int(size * 1.25)
                item_h = int(size * 1.25) + 28
                try:
                    vpw = scroll.viewport().width()
                except Exception:
                    vpw = inner.width()
                if not vpw or vpw <= 1:
                    vpw = max(scroll.width(), self.width(), 600)
                cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                total = layout.count()
                rows = max(1, (total + cols - 1) // cols)
                min_h = rows * item_h + max(0, rows - 1) * spacing + 12
                inner.setMinimumHeight(min_h)
            except Exception:
                pass
        except Exception as e:
            try:
                print("[upsc] recents rebuild error:", e)
            except Exception:
                pass

    def eventFilter(self, obj, ev):
        """Forward event filtering to base class, but watch recents viewport width."""
        try:
            from PySide6.QtCore import QEvent
            if hasattr(self, "recents_scroll") and self.recents_scroll is not None:
                if obj is self.recents_scroll.viewport():
                    if ev.type() == QEvent.Resize:
                        try:
                            w = ev.size().width()
                        except Exception:
                            w = 0
                        if w and w != getattr(self, "_recents_last_w", 0):
                            self._recents_last_w = w
                            try:
                                self._rebuild_recents()
                            except Exception:
                                pass
        except Exception:
            pass
        try:
            return super().eventFilter(obj, ev)
        except Exception:
            return False

    def _apply_streaming_lowmem(self, cmd: list) -> list:
        """Inject low-memory streaming flags into FFmpeg commands when the toggle is ON."""
        try:
            if getattr(self, "chk_streaming_lowmem", None) and self.chk_streaming_lowmem.isChecked():
                if cmd and isinstance(cmd, list) and len(cmd) > 1 and cmd[0] == FFMPEG:
                    inject = ["-fflags", "nobuffer", "-probesize", "64k", "-analyzeduration", "0"]
                    # add -threads 1 only if not already specified
                    if "-threads" not in cmd:
                        inject += ["-threads", "1"]
                    # rebuild: FFMPEG + inject + rest (excluding the first FFMPEG element)
                    cmd = [cmd[0]] + inject + cmd[1:]
        except Exception:
            pass
        return cmd




    def _make_thumb(self, p: "Path") -> QPixmap:
        try:
            ext = p.suffix.lower()
            # Skip work if a job is running
            if getattr(self, "_job_running", False):
                return QPixmap()
            # Images: load via QImageReader (scaled) to avoid huge allocations
            if ext in _IMAGE_EXTS:
                try:
                    from PySide6.QtGui import QImageReader, QPixmap
                    from PySide6.QtCore import QSize
                    reader = QImageReader(str(p))
                    reader.setAutoTransform(True)
                    # Decode to a thumbnail-sized image to keep memory low
                    try:
                        sz = reader.size()
                        max_dim = 512
                        if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                            w, h = sz.width(), sz.height()
                            if w >= h:
                                scaled = QSize(max_dim, max(1, int(h * max_dim / max(1, w))))
                            else:
                                scaled = QSize(max(1, int(w * max_dim / max(1, h))), max_dim)
                        else:
                            scaled = QSize(512, 512)
                        reader.setScaledSize(scaled)
                    except Exception:
                        reader.setScaledSize(QSize(512, 512))
                    img = reader.read()
                    if img and not img.isNull():
                        pm = QPixmap.fromImage(img)
                        if pm and not pm.isNull():
                            return pm
                except Exception:
                    pass
                return QPixmap()
            # Videos: optional, heavier
            if ext in _VIDEO_EXTS:
                try:
                    if not getattr(self, "chk_video_thumbs", None) or not self.chk_video_thumbs.isChecked():
                        return QPixmap()
                except Exception:
                    return QPixmap()
                try:
                    from pathlib import Path as _P
                    import time, subprocess, os
                    td = _P(".").resolve() / "output" / "_thumbs"
                    td.mkdir(parents=True, exist_ok=True)
                    outp = td / (p.stem + "_thumb.jpg")
                    if not outp.exists() or (time.time() - outp.stat().st_mtime > 24*3600):
                        cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-hwaccel", "none", "-threads", "1", "-v", "error",
                               "-i", str(p), "-vf", "thumbnail,scale=320:-1", "-frames:v", "1", str(outp)]
                        try:
                            cmd = self._apply_streaming_lowmem(cmd)
                        except Exception:
                            pass
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    pm = QPixmap(str(outp))
                    if pm and not pm.isNull():
                        return pm
                except Exception:
                    pass
            return QPixmap()
        except Exception:
            return QPixmap()

    def _update_engine_ui(self):
        # Keep model stack in sync with engine
        eng_txt = (self.combo_engine.currentText() or '').lower()
        page = 0
        if 'waifu2x' in eng_txt:
            page = 1
        elif 'srmd (ncnn via realesrgan' in eng_txt:
            page = 5
        elif 'ultrasharp' in eng_txt:
            page = 4
        elif 'srmd' in eng_txt:
            page = 2
        elif 'realsr' in eng_txt:
            page = 3
        else:
            page = 0
        self.stk_models.setCurrentIndex(page)
        # Auto-guide scale to the model's native when using Real-ESRGAN-compatible backends
        try:
            eng = (self.combo_engine.currentText() or '').lower()
        except Exception:
            eng = ''
        if ('realesrgan' in eng) or ('real-esrgan' in eng) or ('ultrasharp' in eng) or ('srmd (ncnn via realesrgan' in eng):
            try:
                model = (self.combo_model_w2x.currentText() if "Waifu2x" in engine_label else (
                    getattr(self, "combo_model_ultrasharp", self.combo_model_realsr).currentText() if "UltraSharp" in engine_label else (
                    getattr(self, "combo_model_srmd_realsr", self.combo_model_realsr).currentText() if "SRMD (ncnn via RealESRGAN)" in engine_label else self.combo_model_realsr.currentText()))) if page==0 else (
                        self.combo_model_ultrasharp.currentText() if page==4 else (
                        self.combo_model_srmd_realsr.currentText() if page==5 else self.combo_model_realsr.currentText()))
                t = (model or '').lower()
                native = 4 if ('-x4' in t or 'x4' in t) else (3 if ('-x3' in t or 'x3' in t) else (2 if ('-x2' in t or 'x2' in t) else 2))
                self.spin_scale.blockSignals(True); self.spin_scale.setValue(float(native)); self.spin_scale.blockSignals(False)
                self.slider_scale.blockSignals(True); self.slider_scale.setValue(int(native*10)); self.slider_scale.blockSignals(False)
                for name, want in (('rad_x2', 2), ('rad_x3', 3), ('rad_x4', 4)):
                    try:
                        w = getattr(self, name, None)
                        if w is not None:
                            w.blockSignals(True); w.setChecked(native == want); w.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass
        # Re-apply page
        self.stk_models.setCurrentIndex(page)
    def _sync_scale_from_spin(self, v: float):
        self.slider_scale.blockSignals(True)
        self.slider_scale.setValue(int(round(v * 10)))
        self.slider_scale.blockSignals(False)
        try:
            self._update_batch_limit()
        except Exception:
            pass

    def _sync_scale_from_slider(self, v: int):
        self.spin_scale.blockSignals(True)
        self.spin_scale.setValue(v / 10.0)
        self.spin_scale.blockSignals(False)
        try:
            self._update_batch_limit()
        except Exception:
            pass


    def _pick_single(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image or video", str(ROOT))
        if path:
            self.edit_input.setText(path)
            self._append_log(f"Selected input: {path}")

    def _pick_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(self.edit_outdir.text() or OUT_VIDEOS)
        )
        if d:
            self.edit_outdir.setText(d)
            self._append_log(f"Output set to: {d}")
    def _pick_batch(self):
        # Use the shared BatchSelectDialog (helpers/batch.py); fallback to legacy prompt if missing.
        try:
            from helpers.batch import BatchSelectDialog as _BatchDialog
        except Exception:
            try:
                from helpers.vatch import BatchSelectDialog as _BatchDialog  # optional alias
            except Exception:
                _BatchDialog = None

        if _BatchDialog is not None:
            try:
                # Allow both images and videos
                _exts = sorted(set(getattr(_BatchDialog, "VIDEO_EXTS", set())) | set(getattr(_BatchDialog, "IMAGE_EXTS", set())))
                files, conflict = _BatchDialog.pick(self, title="Add Batch", exts=_exts)
            except Exception:
                files, conflict = None, None

            if files is None:
                # cancelled
                return
            if not files:
                QtWidgets.QMessageBox.information(self, "No media", "No files selected.")
                return

            # map dialog result to internal duplicate mode
            if conflict in ("version", "autorename", "auto", "ver"):
                dup_mode = "version"
            elif conflict == "overwrite":
                dup_mode = "overwrite"
            else:
                dup_mode = "skip"

            # store for _build_outfile behavior (restored at end)
            prev_dup = getattr(self, "_dup_mode", None)
            try:
                self._dup_mode = dup_mode
            except Exception:
                pass

            # If skip: pre-filter files whose default outfile already exists
            to_run = []
            skipped = 0
            try:
                sc = int(round(float(self.spin_scale.value())))
            except Exception:
                sc = 2
            for f in files:
                p = Path(f)
                try:
                    is_video = p.suffix.lower() in _VIDEO_EXTS
                except Exception:
                    try:
                        from pathlib import Path as _P
                        is_video = _P(f).suffix.lower() in _VIDEO_EXTS
                    except Exception:
                        is_video = False
                try:
                    outd = Path(self.edit_outdir.text().strip()) if getattr(self, "edit_outdir", None) and (self.edit_outdir.text().strip()) else (OUT_VIDEOS if is_video else OUT_SHOTS)
                except Exception:
                    outd = OUT_VIDEOS if is_video else OUT_SHOTS
                # Compute the *default* name (no versioning) for skip check
                default_out = outd / f"{p.stem}_x{sc}{p.suffix}"
                if dup_mode == "skip" and default_out.exists():
                    skipped += 1
                    continue
                to_run.append(p)

            if not to_run:
                QtWidgets.QMessageBox.information(self, "Nothing to do", "All selected outputs already exist.")
                # restore dup mode
                if prev_dup is not None:
                    self._dup_mode = prev_dup
                else:
                    try: delattr(self, "_dup_mode")
                    except Exception: pass
                return

            self._run_batch_files(to_run)

            # restore dup mode after queuing
            if prev_dup is not None:
                self._dup_mode = prev_dup
            else:
                try: delattr(self, "_dup_mode")
                except Exception: pass
            if skipped:
                try:
                    self._append_log(f"[batch] Skipped {skipped} existing output(s).")
                except Exception:
                    pass
            return
        # ----- Legacy fallback (original UI) -----
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Batch input")
        dlg.setText("Pick a folder or choose files.")
        btn_folder = dlg.addButton("Folder…", QtWidgets.QMessageBox.AcceptRole)
        btn_files = dlg.addButton("Files…", QtWidgets.QMessageBox.ActionRole)
        btn_cancel = dlg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        dlg.setDefaultButton(btn_files)
        dlg.exec()
        if dlg.clickedButton() is btn_cancel:
            self._append_log("Batch canceled.")
            return
        if dlg.clickedButton() is btn_folder:
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
            if not d:
                self._append_log("Batch canceled.")
                return
            self._run_batch_from_folder(Path(d))
            return
        filt = "Media files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts *.m2ts *.wmv *.flv *.mpg *.mpeg *.3gp *.3g2 *.ogv *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif);;All files (*)"
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select files", "", filt)
        if files:
            self._run_batch_files([Path(f) for f in files])


    def _run_batch_from_folder(self, folder: Path):
        exts = _IMAGE_EXTS | _VIDEO_EXTS
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        if not files:
            QtWidgets.QMessageBox.information(self, "No media", "That folder has no supported images or videos.")
            return
        # Confirmation dialog before queuing
        count = len(files)
        resp = QtWidgets.QMessageBox.question(
            self,
            "Confirm queue",
            f"Queue {count} files?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        if resp != QtWidgets.QMessageBox.Yes:
            self._append_log("Batch canceled.")
            return
        for p in files:
            self._run_one(p)

    def _run_batch_files(self, files: List[Path]):
        # Confirmation dialog before queuing
        count = len(files)
        if count <= 0:
            QtWidgets.QMessageBox.information(self, "No media", "No files selected.")
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Confirm queue",
            f"Queue {count} files?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        if resp != QtWidgets.QMessageBox.Yes:
            self._append_log("Batch canceled.")
            return
        for p in files:
            self._run_one(p)
    def _build_outfile(self, src: Path, outd: Path, scale: int) -> Path:
        outd.mkdir(parents=True, exist_ok=True)
        base = outd / f"{src.stem}_x{scale}{src.suffix}"
        mode = getattr(self, "_dup_mode", "overwrite")
        if mode == "version":
            i = 1
            cand = base
            while cand.exists():
                cand = outd / f"{src.stem}_x{scale} ({i}){src.suffix}"
                i += 1
            return cand
        return base
    def _realsr_cmd_dir(self, exe: str, indir: Path, outdir: Path, model: str, scale: int) -> List[str]:
        n = model[:-3] if model.endswith(("-x2", "-x3", "-x4")) else model
        cmd = [exe, "-i", str(indir), "-o", str(outdir), "-n", n, "-s", str(scale), "-m", str(REALSR_DIR), "-f", "png"]
        try:
            t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
            if t > 0:
                cmd += ["-t", str(t)]
        except Exception:
            pass
        return cmd

    def _realsr_cmd_file(self, exe: str, infile: Path, outfile: Path, model: str, scale: int) -> List[str]:
        n = model[:-3] if model.endswith(("-x2", "-x3", "-x4")) else model
        cmd = [exe, "-i", str(infile), "-o", str(outfile), "-n", n, "-s", str(scale), "-m", str(REALSR_DIR)]
        try:
            t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
            if t > 0:
                cmd += ["-t", str(t)]
        except Exception:
            pass
        return cmd

    def _waifu_cmd_file(self, exe: str, infile: Path, outfile: Path, model_dirname: str, scale: int) -> List[str]:
        return [exe, "-i", str(infile), "-o", str(outfile), "-m", str(WAIFU2X_DIR / model_dirname), "-s", str(scale)]
    def _do_single(self):
        raw = (self.edit_input.text() or '').strip()
        src = Path(raw)
        if raw in ('.','./','..',''):
            # Try to fall back to player path if available
            try:
                p = _fv_get_input(self)
                if p:
                    src = Path(p)
            except Exception:
                pass
        if not src.exists():
            QtWidgets.QMessageBox.warning(self, "No input", "Please choose an image or video.")
            return
        if src.is_dir():
            self._run_batch_from_folder(src)
            return
        self._run_one(src)


    def _run_one(self, src: Path):
        # Make current file visible to the queue shim during batch
        try:
            self._last_infile = src
        except Exception:
            pass
        try:
            if hasattr(self, 'edit_input') and hasattr(self.edit_input, 'setText'):
                self.edit_input.setText(str(src))
        except Exception:
            pass
        self._job_running = True
        try:
            ext = src.suffix.lower()
            is_video = ext in _VIDEO_EXTS
            is_image = ext in _IMAGE_EXTS
    
            engine_label = self.combo_engine.currentText()
            engine_exe = self.combo_engine.currentData()
            if not engine_exe or not _exists(engine_exe):
                QtWidgets.QMessageBox.critical(
                    self, "Engine not found",
                    f"The selected engine executable was not found:\n{engine_exe}\n\nCheck your models folder paths."
                )
                self._append_log(f"✖ Engine missing: {engine_exe}")
                return
    
            scale = int(round(float(self.spin_scale.value())))
            scale = max(1, min(8, scale))
            outd = Path(self.edit_outdir.text().strip()) if self.edit_outdir.text().strip() else (OUT_VIDEOS if is_video else OUT_SHOTS)
            outfile = self._build_outfile(src, outd, scale)
            self._last_outfile = outfile
    
            if is_video:
                if "Waifu2x" in engine_label:
                    QtWidgets.QMessageBox.information(self, "Not supported", "Waifu2x (ncnn) handles images only for now. Please select Real-ESRGAN for videos.")
                    self._append_log("Waifu2x selected for a video — blocked (images only).")
                    return
    
                model = self.combo_model_realsr.currentText()
                fps = "30"
                work = outd / f"{src.stem}_x{scale}_work"
                in_dir = work / "in"
                out_dir = work / "out"
                in_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                seq_in = in_dir / "f_%08d.png"
                seq_out = out_dir / "f_%08d.png"
    
                pre = self._build_pre_filters()
                cmd_extract = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src), "-map", "0:v:0"]
                if pre: cmd_extract += ["-vf", pre]
                cmd_extract += ["-fps_mode", "vfr", str(seq_in)]
                cmd_upscale = self._realsr_cmd_dir(engine_exe, in_dir, out_dir, model, scale)
                post = self._build_post_filters()
                cmd_encode = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-framerate", fps, "-i", str(seq_out), "-i", str(src), "-map", "0:v:0"]
                if self.radio_a_mute.isChecked():
                    pass
                else:
                    cmd_encode += ["-map", "1:a?"]
                vcodec = self.combo_vcodec.currentText()
                cmd_encode += ["-c:v", vcodec, "-pix_fmt", "yuv420p"]
                cmd_encode += ["-vsync", "cfr"]
                if self.rad_crf.isChecked():
                    cmd_encode += ["-crf", str(self.spin_crf.value())]
                else:
                    cmd_encode += ["-b:v", f"{self.spin_bitrate.value()}k"]
                preset = self.combo_preset.currentText()
                if preset:
                    cmd_encode += ["-preset", preset]
                if int(self.spin_keyint.value() or 0) > 0:
                    cmd_encode += ["-g", str(int(self.spin_keyint.value()))]
                if post:
                    cmd_encode += ["-vf", f"fps={fps}," + post]
                else:
                    cmd_encode += ["-vf", f"fps={fps}"]
                if self.radio_a_mute.isChecked():
                    cmd_encode += ["-an"]
                elif self.radio_a_copy.isChecked():
                    cmd_encode += ["-c:a", "copy"]
                else:
                    cmd_encode += ["-c:a", self.combo_acodec.currentText(), "-b:a", f"{self.combo_abitrate.currentText()}k"]
                cmd_encode += ["-r", fps, "-shortest", str(outfile)]
    
                self._append_log(f"Engine: {engine_label}")
                self._append_log(f"Executable: {engine_exe}")
                self._append_log(f"Model: {model}")
                self._append_log(f"Model dir: {REALSR_DIR}")
                self._append_log(f"Scale: x{scale}")
                self._append_log(f"FPS: {fps}")
                self._append_log(f"Work dir: {work}")
                self._append_log(f"Output: {outfile}")
    
                self._run_cmd([cmd_extract, cmd_upscale, cmd_encode], open_on_success=True, cleanup_dirs=[work])
                return
    
            # Image path
            if "Waifu2x" in engine_label:
                model = self.combo_model_w2x.currentText()
                cmd = self._waifu_cmd_file(engine_exe, src, outfile, model, scale)
            elif "UltraSharp" in engine_label:
                model = getattr(self, "combo_model_ultrasharp", self.combo_model_realsr).currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
            elif "SRMD (ncnn via RealESRGAN)" in engine_label:
                model = getattr(self, "combo_model_srmd_realsr", self.combo_model_realsr).currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
            else:
                model = self.combo_model_realsr.currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
    
            self._append_log(f"Engine: {engine_label}")
            self._append_log(f"Executable: {engine_exe}")
            self._append_log(f"Model: {model}")
            if "Waifu2x" not in engine_label:
                self._append_log(f"Model dir: {REALSR_DIR}")
            self._append_log(f"Scale: x{scale}")
            self._append_log(f"Input: {src}")
            self._append_log(f"Output: {outfile}")
            self._run_cmd([cmd], open_on_success=True)
        finally:
            self._job_running = False
            try:
                self._rebuild_recents()
            except Exception:
                pass

    
    def _run_cmd(self, cmds: List[List[str]], open_on_success: bool = False, cleanup_dirs: Optional[List[Path]] = None):
            self.btn_upscale.setEnabled(False)
            self.btn_batch.setEnabled(False)
            self._thr = _RunThread(cmds, cwd=ROOT, parent=self)
            self._thr.progress.connect(self._append_log)

            def on_done(code: int, last: str):
                self.btn_upscale.setEnabled(True)
                self.btn_batch.setEnabled(True)
                if code == 0:
                    self._append_log("✔ Done.")
                    if self._last_outfile and self._last_outfile.exists():
                        self._append_log(f"Saved: {self._last_outfile}")
                        try:
                            self._add_recent(self._last_outfile)
                        except Exception:
                            pass
                        finally:
                            self._job_running = False
                            try:
                                self._rebuild_recents()
                            except Exception:
                                pass
                    if cleanup_dirs:
                        for d in cleanup_dirs:
                            try:
                                shutil.rmtree(d, ignore_errors=True)
                            except Exception:
                                pass
                else:
                    self._append_log(f"✖ Finished with code {code}")

            self._thr.done.connect(on_done)
            self._thr.start()

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        try:
            # Fast append at end; avoids O(n) inserts at start
            self.log.appendPlainText(line)
            self.log.ensureCursorVisible()
        except Exception:
            try:
                cur = self.log.textCursor()
                cur.movePosition(cur.End)
                cur.insertText(line + "\n")
                self.log.setTextCursor(cur)
                self.log.ensureCursorVisible()
            except Exception:
                # Last-resort fallback (avoid expensive full setPlainText on large logs)
                pass
# Build pre-extract filters (applied on input video before frame extraction)
    def _build_pre_filters(self) -> str:
        fs = []
        try:
            if self.chk_deinterlace.isChecked():
                fs.append("yadif=mode=1:parity=-1:deint=0")
            if self.chk_deblock.isChecked():
                fs.append("pp=deblock")
            if self.chk_denoise.isChecked():
                fs.append("hqdn3d=3:3:6:6")
            if self.chk_deband.isChecked():
                fs.append("gradfun=thr=0.6")
        except Exception:
            pass
        return ",".join(fs)

    # Build post-encode filters (after upscale, on image sequence)
    def _build_post_filters(self) -> str:
        fs = []
        try:
            rng = self.combo_range.currentText()
            if rng == "Full→Limited":
                fs.append("scale=in_range=pc:out_range=tv")
            elif rng == "Limited→Full":
                fs.append("scale=in_range=tv:out_range=pc")
            val = int(self.sld_sharpen.value()) if hasattr(self, "sld_sharpen") else 0
            if val > 0:
                amt = round(1.5 * (val / 100.0), 3)
                fs.append(f"unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount={amt}")
        except Exception:
            pass
        return ",".join(fs)


    def _try_unload_media_before_delete(self, media_path: Path):
        """Best-effort: unload *media_path* from the main player/viewer before deleting.

        This tries to clear any player in the main window that currently has this path
        open, so Windows can release file locks. Failures are silently ignored; the
        caller still needs to handle delete errors.
        """
        try:
            if media_path is None:
                return
            m = getattr(self, "_main", None) or getattr(self, "main", None)
            if not m:
                return
            try:
                media_resolved = media_path.resolve()
            except Exception:
                media_resolved = media_path
            targets = []
            for attr in ("video", "player", "viewer"):
                try:
                    obj = getattr(m, attr, None)
                except Exception:
                    obj = None
                if obj is not None:
                    targets.append(obj)
            for obj in targets:
                try:
                    cur = getattr(obj, "current_path", None)
                except Exception:
                    cur = None
                if not cur:
                    try:
                        cur = getattr(obj, "source", None)
                    except Exception:
                        cur = None
                if not cur:
                    continue
                try:
                    from pathlib import Path as _P
                    cur_p = _P(str(cur))
                    cur_resolved = cur_p.resolve()
                except Exception:
                    cur_resolved = cur
                try:
                    same = (cur_resolved == media_resolved)
                except Exception:
                    same = False
                if not same:
                    continue
                # Try a few common ways to unload
                for meth in ("clear", "close", "stop"):
                    fn = getattr(obj, meth, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                fn = getattr(obj, "open", None)
                if callable(fn):
                    for arg in (None, "", " "):
                        try:
                            fn(arg)
                            break
                        except Exception:
                            continue
            # Clear main-window current_path-style attributes if they point here
            for attr in ("current_path", "current_media", "current_file"):
                try:
                    cur = getattr(m, attr, None)
                except Exception:
                    cur = None
                if not cur:
                    continue
                try:
                    from pathlib import Path as _P
                    cur_p = _P(str(cur))
                    cur_resolved = cur_p.resolve()
                except Exception:
                    cur_resolved = cur
                try:
                    same = (cur_resolved == media_resolved)
                except Exception:
                    same = False
                if same:
                    try:
                        setattr(m, attr, None)
                    except Exception:
                        pass
        except Exception:
            # Best-effort only
            pass

    def _play_in_player(self, p: Path):

        m = getattr(self, "_main", None) or getattr(self, "main", None)
        try:
            if m is not None and hasattr(m, "video") and hasattr(m.video, "open"):
                m.video.open(p)
                try:
                    # Keep MainWindow's notion of current media in sync
                    m.current_path = p
                except Exception:
                    pass
                try:
                    # Force-refresh the Info dialog (if open)
                    refresh_info_now(p)
                except Exception:
                    pass
                return True
        except Exception:
            pass
        try:
            self._open_file(p)
            return True
        except Exception:
            return False
    def _show_media_info(self):
        src_path = self.edit_input.text().strip()
        if not src_path:
            QtWidgets.QMessageBox.information(self, "Info", "No input selected.")
            return
        self._show_media_info_for(src_path)

    def _show_media_info_for(self, src_path):
        src_path = (str(src_path) or "").strip()
        if not src_path:
            QtWidgets.QMessageBox.information(self, "Info", "No media selected.")
            return
        try:
            cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", src_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            info = out.decode("utf-8", errors="ignore") if out else "{}"
        except Exception as e:
            info = json.dumps({"error": str(e)}, indent=2)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Media info")
        lay = QtWidgets.QVBoxLayout(dlg)
        te = QtWidgets.QPlainTextEdit(dlg); te.setReadOnly(True); te.setPlainText(info)
        lay.addWidget(te)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dlg)
        btns.rejected.connect(dlg.reject); btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.resize(700, 600)
        dlg.exec()

    def _auto_tile_size(self):
        exe = self.combo_engine.currentData()
        if not exe or "realesrgan" not in str(exe).lower():
            QtWidgets.QMessageBox.information(self, "Auto tile", "Auto tile size works with Real-ESRGAN (ncnn) engine.")
            return
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="tileprobe_"))
        try:
            test_in = tmp / "in.png"; test_out = tmp / "out.png"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-f", "lavfi", "-i", "color=c=gray:s=256x256", str(test_in)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = self.combo_model_realsr.currentText(); scale = int(round(self.spin_scale.value()))
            candidates = [800, 600, 400, 300, 200, 100]
            ok = None
            for t in candidates:
                cmd = [exe, "-i", str(test_in), "-o", str(test_out), "-n", (model[:-3] if model.endswith(('-x2','-x3','-x4')) else model), "-s", str(scale), "-m", str(REALSR_DIR), "-t", str(t)]
                try:
                    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                    if r.returncode == 0 and test_out.exists():
                        ok = t; break
                except Exception:
                    pass
            if ok is None:
                self.lbl_tile_result.setText("Probe failed; using engine default.")
                self.spin_tile.setValue(0)
            else:
                self.spin_tile.setValue(ok)
                self.lbl_tile_result.setText(f"Chosen tile: {ok}")
        finally:
            try:
                for p in tmp.glob("*"): p.unlink()
                tmp.rmdir()
            except Exception:
                pass

    def _update_model_hint(self):
        try:
            name = self.combo_model_realsr.currentText().lower()
        except Exception:
            return
        cat, hint = "Other", ""
        if "anime" in name:
            cat, hint = "Anime", "Optimized for line art / anime; preserves edges with minimal ringing."
        elif "ultrasharp" in name or "ui" in name or "text" in name:
            cat, hint = "UI/Text", "Very crisp edges; can halo UI/text. Good for logos and screen captures."
        elif "general" in name or "x4plus" in name:
            cat, hint = "Photo", "Balanced detail for photographs; natural textures."
        try:
            self.lbl_model_badge.setText(cat)
        except Exception:
            pass
        try:
            self.lbl_model_hint.setText(hint)
        except Exception:
            pass

def _open_file(self, p: Path):
        try:
            if os.name == "nt":
                os.startfile(str(p))  # nosec - user initiated
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception:
            pass


# === FrameVision patch: queue + player wiring (adaptive queue args) ===
from pathlib import Path as _FVPath
import inspect as _FVinspect
try:
    from PySide6 import QtWidgets as _FVQtW  # type: ignore
except Exception:  # pragma: no cover
    _FVQtW = None  # type: ignore

def _fv_is_valid_file(p):
    try:
        return p and _FVPath(p).exists() and _FVPath(p).is_file()
    except Exception:
        return False

def _fv_guess_player_path(owner):
    m = getattr(owner, "_main", None) or getattr(owner, "main", None)
    candidates = []
    if m:
        candidates += [
            getattr(m, "current_path", None),
            getattr(m, "current_file", None),
            getattr(m, "current_media", None),
            getattr(getattr(m, "player", None), "current_path", None),
            getattr(getattr(m, "player", None), "source", None),
            getattr(getattr(m, "viewer", None), "current_path", None),
        ]
    seen = set(); cand = []
    for c in candidates:
        s = str(c) if c is not None else None
        if s and s not in seen:
            seen.add(s); cand.append(s)
    for c in cand:
        if _fv_is_valid_file(c):
            return c
    return None

def _fv_get_input(self):
    # try text fields first
    for attr in ("edit_input","line_input","edit_in","input_line","editPath","linePath"):
        w = getattr(self, attr, None)
        try:
            if w and hasattr(w, "text"):
                t = (w.text() or '').strip()
                if t in ('.','./','..',''):
                    pass
                elif _fv_is_valid_file(t):
                    return t
        except Exception:
            pass
    # then stored infile
    try:
        p = getattr(self, "_last_infile", None)
        if p and _fv_is_valid_file(str(p)):
            return str(p)
    except Exception:
        pass
    # finally ask player
    return _fv_guess_player_path(self)

def _fv_push_input_to_tab(self, path_str):
    pushed = False
    for attr in ("edit_input","line_input","edit_in","input_line","editPath","linePath"):
        w = getattr(self, attr, None)
        try:
            if w and hasattr(w, "setText"):
                w.setText(path_str); pushed = True
        except Exception:
            pass
    for meth in ("set_input_path","load_single_input","set_source","set_path"):
        fn = getattr(self, meth, None)
        try:
            if callable(fn):
                fn(path_str); pushed = True
        except Exception:
            pass
    try:
        self._last_infile = _FVPath(path_str)
    except Exception:
        pass
    return pushed

def _fv_find_enqueue(self):
    """Return a callable that accepts (job_dict) OR (input_path, out_dir, factor, model), plus a label."""
    m = getattr(self, "_main", None) or getattr(self, "main", None)

    candidates = []
    # direct
    for name in ("enqueue", "enqueue_job", "enqueue_external", "enqueue_single_action", "queue_add", "add_job"):
        fn = getattr(m, name, None) if m is not None else None
        if callable(fn): candidates.append((fn, f"main.{name}"))
    # queue_adapter on main
    qa = getattr(m, "queue_adapter", None) if m is not None else None
    for name in ("enqueue", "add", "put"):
        fn = getattr(qa, name, None) if qa is not None else None
        if callable(fn): candidates.append((fn, f"main.queue_adapter.{name}"))
    # nested queue
    q = getattr(m, "queue", None) if m is not None else None
    for name in ("enqueue", "add", "put", "add_job"):
        fn = getattr(q, name, None) if q is not None else None
        if callable(fn): candidates.append((fn, f"main.queue.{name}"))
    # module-level
    try:
        import helpers.queue_adapter as _qa  # type: ignore
        for name in ("enqueue", "add", "put", "add_job"):
            fn = getattr(_qa, name, None)
            if callable(fn): candidates.append((fn, f"helpers.queue_adapter.{name}"))
    except Exception:
        pass

    # pick best (prefer queue_adapter.enqueue if it exists)
    lbl_order = ["helpers.queue_adapter.enqueue", "main.queue_adapter.enqueue", "main.enqueue", "main.enqueue_job"]
    for prefer in lbl_order:
        for fn, label in candidates:
            if label.endswith(prefer):
                return fn, label
    return candidates[0] if candidates else (None, "")

def _fv_call_enqueue(self, enq, where_label, cmds, open_on_success):
    """Try to call enqueue either with a job dict, or with signature (input_path, out_dir, factor, model)."""
    sig = None
    try:
        sig = _FVinspect.signature(enq)
    except Exception:
        pass

    # Collect context
    input_path = _fv_get_input(self)
    out_dir = ""
    try:
        out_dir = (getattr(self, 'edit_outdir', None).text() if getattr(self, 'edit_outdir', None) else '') or (getattr(self, 'edit_output', None).text() if getattr(self, 'edit_output', None) else '')
    except Exception:
        pass
    # factor: try to read from UI or model name
    factor = 2
    try:
        factor = int(round(float(getattr(self, 'spin_scale', None).value())))
    except Exception:
        try:
            txt = getattr(self, 'combo_model', None).currentText()
            if isinstance(txt, str) and 'x4' in txt: factor = 4
            elif isinstance(txt, str) and 'x3' in txt: factor = 3
            elif isinstance(txt, str) and 'x2' in txt: factor = 2
        except Exception:
            pass
    model_name = ''
    try:
        eng_label = getattr(self, 'combo_engine', None).currentText()
    except Exception:
        eng_label = ''
    try:
        if isinstance(eng_label, str) and 'Waifu2x' in eng_label:
            cmw = getattr(self, 'combo_model_w2x', None)
            if cmw and hasattr(cmw, 'currentText'):
                model_name = cmw.currentText()
        else:
            cmr = getattr(self, 'combo_model_realsr', None)
            if cmr and hasattr(cmr, 'currentText'):
                model_name = cmr.currentText()
    except Exception:
        try:
            model_name = getattr(self, 'combo_model', None).currentText()
        except Exception:
            pass

    # If signature wants plain args, call with those
    if sig:
        params = list(sig.parameters.keys())
        if params[:4] == ["input_path", "out_dir", "factor", "model"] or set(("input_path","out_dir","factor","model")).issubset(set(params)):
            try:
                enq(job_type=('upscale_photo' if str(Path(input_path)).lower().endswith(tuple(_IMAGE_EXTS)) else 'upscale_video'), input_path=input_path, out_dir=out_dir, factor=factor, model=model_name)
                try:
                    self._append_log(f"Queued via {where_label} (kwargs).")
                except Exception: pass
                return True
            except Exception as e:
                try: self._append_log(f"Queue error via {where_label}: {e}")
                except Exception: pass
                return False

    # Else, fallback to job dicts (one per command)
    for i, c in enumerate(cmds, 1):
        job = {
            "name": "Upscale" if len(cmds) == 1 else f"Upscale ({i}/{len(cmds)})",
            "category": "upscale",
            "cmd": c,
            "cwd": str(globals().get("ROOT", ".")),
            "open_on_success": bool(open_on_success and i == len(cmds)),
            "output": str(getattr(self, "_last_outfile", "")),
        }
        try:
            enq(job)
        except Exception as e:
            try: self._append_log(f"Queue error via {where_label}: {e}")
            except Exception: pass
            return False
    try:
        self._append_log(f"Queued {len(cmds)} job(s) via {where_label}")
    except Exception:
        pass
    return True

try:
    _UpscClass = None
    for _n, _obj in list(globals().items()):
        if isinstance(_obj, type) and hasattr(_obj, "_run_cmd"):
            _UpscClass = _obj
            break

    if _UpscClass is not None:
        _orig_run_cmd = getattr(_UpscClass, "_run_cmd", None)
        def _patched_run_cmd(self, cmds, open_on_success: bool = False, cleanup_dirs=None):
            enq, where = _fv_find_enqueue(self)
            if callable(enq) and cmds:
                if _fv_call_enqueue(self, enq, where, cmds, open_on_success):
                    return
            # Fallback to original implementation
            if callable(_orig_run_cmd):
                return _orig_run_cmd(self, cmds, open_on_success=open_on_success, cleanup_dirs=cleanup_dirs)

        setattr(_UpscClass, "_run_cmd", _patched_run_cmd)

        _orig_set_main = getattr(_UpscClass, "set_main", None)
        def _patched_set_main(self, main):
            if callable(_orig_set_main):
                try:
                    _orig_set_main(self, main)
                except TypeError:
                    try: self._main = main
                    except Exception: pass
            else:
                try: self._main = main
                except Exception: pass
            if _FVQtW is not None and main is not None:
                try:
                    for _btn in main.findChildren(_FVQtW.QPushButton):
                        if _btn.text().strip().lower() == "upscale":
                            try: _btn.clicked.disconnect()
                            except Exception: pass
                            def _call():
                                p = _fv_guess_player_path(self) or _fv_get_input(self)
                                if not _fv_is_valid_file(p):
                                    try: self._append_log("Player Upscale: no valid file path available.")
                                    except Exception: pass
                                    return
                                _fv_push_input_to_tab(self, p)
                                try:
                                    if hasattr(self, "btn_upscale"):
                                        self.btn_upscale.click()
                                    else:
                                        self._do_single()
                                except Exception:
                                    try: self._do_single()
                                    except Exception: pass
                            _btn.clicked.connect(_call)
                            break
                except Exception:
                    pass

        setattr(_UpscClass, "set_main", _patched_set_main)

except Exception as _patch_exc:
    try:
        print(f"[upsc patch] non-fatal: {_patch_exc}")
    except Exception:
        pass



# --- FrameVision silent integration r11 ---
from typing import List, Tuple
from pathlib import Path as _P
import os as _O

def _fv_r11_find(root: _P, names) -> _P | None:
    try:
        if not root.exists():
            return None
        for nm in names:
            for q in root.rglob(nm):
                if q.is_file():
                    return q
    except Exception:
        return None
    return None

def _fv_r11_scan_pairs(root: _P):
    out = []
    try:
        if not root.exists():
            return out
        for prm in root.rglob("*.param"):
            base = prm.stem
            binf = prm.with_suffix(".bin")
            if not binf.exists():
                continue
            sc = 4
            low = base.lower()
            for k in ("-8x","-6x","-4x","-3x","-2x"):
                if k in low:
                    try:
                        sc = int(k[1])
                    except Exception:
                        sc = 4
                    break
            out.append((base, str(prm.parent), sc))
    except Exception:
        pass
    return out

def _fv_r11_install_upsc_models(self):
    try:
        root = ROOT  # provided by original module
    except Exception:
        root = _P(__file__).resolve().parents[1]
    cand_dirs = [root / "models" / "upscayl", root / "models" / "upscayl" / "resources" / "models"]
    seen = set()
    for d in cand_dirs:
        for base, dstr, sc in _fv_r11_scan_pairs(d):
            key = (base, dstr)
            if key in seen:
                continue
            seen.add(key)
            label = "Upscayl - " + base
            try:
                self.combo_model_realsr.addItem(label, {"fv":"upsc-model","base":base,"dir":dstr,"scale":sc})
            except Exception:
                self.combo_model_realsr.addItem(label)

def _fv_r11_wrap_file(orig):
    def _wrapped(self, engine_exe, src, outfile, model, scale):
        from pathlib import Path as __P
        idx = getattr(self.combo_model_realsr, "currentIndex", lambda: -1)()
        data = None
        try:
            data = self.combo_model_realsr.itemData(idx)
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("fv") == "upsc-model":
            base = data["base"]; dstr = data["dir"]; sc = int(data["scale"])
            in_p = __P(src); out_p = __P(outfile)
            if in_p.exists() and in_p.is_dir():
                out_p.mkdir(parents=True, exist_ok=True)
                return [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
            if (not out_p.suffix) or (out_p.exists() and out_p.is_dir()):
                out_p.mkdir(parents=True, exist_ok=True)
                out_p = out_p / (in_p.stem + ".png")
            return [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
        return orig(self, engine_exe, src, outfile, model, scale)
    return _wrapped

def _fv_r11_wrap_dir(orig):
    def _wrapped(self, engine_exe, in_dir, out_dir, model, scale):
        from pathlib import Path as __P
        idx = getattr(self.combo_model_realsr, "currentIndex", lambda: -1)()
        data = None
        try:
            data = self.combo_model_realsr.itemData(idx)
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("fv") == "upsc-model":
            base = data["base"]; dstr = data["dir"]; sc = int(data["scale"])
            in_p = __P(in_dir); out_p = __P(out_dir); out_p.mkdir(parents=True, exist_ok=True)
            return [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
        return orig(self, engine_exe, in_dir, out_dir, model, scale)
    return _wrapped

def _fv_r11_install_hooks():
    try:
        _Pane = UpscPane  # UI class from original module
    except Exception:
        return
    try:
        _old_init = _Pane.__init__
        def _new_init(self, *a, **k):
            _old_init(self, *a, **k)
            try:
                _fv_r11_install_upsc_models(self)
            except Exception:
                pass
        _Pane.__init__ = _new_init
        if hasattr(_Pane, "_realsr_cmd_file"):
            _Pane._realsr_cmd_file = _fv_r11_wrap_file(_Pane._realsr_cmd_file)
        if hasattr(_Pane, "_realsr_cmd_dir"):
            _Pane._realsr_cmd_dir = _fv_r11_wrap_dir(_Pane._realsr_cmd_dir)
    except Exception:
        pass

def _fv_r11_detect_engines() -> List[Tuple[str, str]]:
    engines: List[Tuple[str, str]] = []
    try:
        root = ROOT
    except Exception:
        root = _P(__file__).resolve().parents[1]
    models = root / "models"
    externals = root / "externals"
    def _find(root: _P, names):
        return _fv_r11_find(root, names)
    realsr = _find(models / "realesrgan", ["realesrgan-ncnn-vulkan.exe","realesrgan-ncnn-vulkan"])
    if realsr: engines.append(("Real-ESRGAN (ncnn)", str(realsr)))
    if realsr: engines.append(("UltraSharp (ncnn)", str(realsr)))
    if realsr: engines.append(("SRMD (ncnn via RealESRGAN)", str(realsr)))
    waifu = _find(models / "waifu2x", ["waifu2x-ncnn-vulkan.exe","waifu2x-ncnn-vulkan"])
    if waifu: engines.append(("Waifu2x (ncnn)", str(waifu)))
    srmd = _find(models / "srmd-ncnn-vulkan-master", ["srmd-ncnn-vulkan.exe","srmd-ncnn-vulkan"])
    if srmd: engines.append(("SRMD (ncnn)", str(srmd)))
    realsr_ncnn = _find(models / "realsr-ncnn-vulkan-20220728-windows", ["realsr-ncnn-vulkan.exe","realsr-ncnn-vulkan"])
    if realsr_ncnn: engines.append(("RealSR (ncnn)", str(realsr_ncnn)))
    swinir = _find(models / "swinir", ["swinir-ncnn-vulkan.exe","swinir-ncnn-vulkan"])
    if swinir: engines.append(("SwinIR (ncnn)", str(swinir)))
    lapsrn = _find(models / "lapsrn", ["lapsrn-ncnn-vulkan.exe","lapsrn-ncnn-vulkan"])
    if lapsrn: engines.append(("LapSRN (ncnn)", str(lapsrn)))
    # Upscayl: only show if CLI exists
    up_cli = None
    for base in (externals / "upscayl", models / "upscayl"):
        up_cli = _find(base, ["upscayl-cli.exe","upscayl-cli"])
        if up_cli:
            engines.append(("Upscayl (CLI)", str(up_cli)))
            break
    return engines

# Attach hooks and quiet detector
try:
    _fv_r11_install_hooks()
    detect_engines = _fv_r11_detect_engines
except Exception:
    pass
# --- end r11 ---


# --- FRAMEVISION_INPUT_SYNC_AND_SCALE_TOGGLE_PATCH ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_patched_input_and_scale"):
        _Pane._fv_patched_input_and_scale = True

        # ---------- Wrap _do_single to sync input from player ----------
        if not hasattr(_Pane, "_fv_orig_do_single"):
            _Pane._fv_orig_do_single = _Pane._do_single
            def _fv_wrapped_do_single(self, *a, **k):
                try:
                    p = _fv_guess_player_path(self)  # type: ignore  # noqa: F821
                    if p and _fv_is_valid_file(p):   # type: ignore  # noqa: F821
                        try:
                            cur = (self.edit_input.text() or '').strip()
                        except Exception:
                            cur = ''
                        if str(p) != cur:
                            try:
                                _fv_push_input_to_tab(self, str(p))  # type: ignore  # noqa: F821
                            except Exception:
                                try:
                                    self.edit_input.setText(str(p))
                                except Exception:
                                    pass
                except Exception:
                    pass
                return _Pane._fv_orig_do_single(self, *a, **k)
            _Pane._do_single = _fv_wrapped_do_single

        # ---------- Helpers for scale UI toggle ----------
        from PySide6 import QtWidgets  # type: ignore

        def _fv_is_realsr_engine(self) -> bool:
            try:
                t = (self.combo_engine.currentText() or '').lower()
                # Treat ALL engines as fixed-scale except Waifu2x.
                # When this returns True, the UI hides the Scale label/slider/spin.
                return ('waifu2x' not in t)
            except Exception:
                return False
        def _fv_native_scale(self) -> int:
            try:
                t = (self.combo_model_realsr.currentText() or '').lower()
                if ('-x4' in t) or ('x4' in t): return 4
                if ('-x3' in t) or ('x3' in t): return 3
                return 2
            except Exception:
                return 2

        def _fv_find_scale_label(self):
            try:
                for lab in self.findChildren(QtWidgets.QLabel):
                    if (lab.text() or '').strip().lower() == 'scale:':
                        return lab
            except Exception:
                pass
            return None

        def _fv_apply_scale_ui(self):
            # Show/hide and sync scale value depending on engine
            isr = _fv_is_realsr_engine(self)
            lbl = getattr(self, '_fv_lbl_scale', None)
            if lbl is None:
                lbl = _fv_find_scale_label(self)
                if lbl is not None:
                    self._fv_lbl_scale = lbl
            try:
                if isr:
                    native = _fv_native_scale(self)
                    try:
                        self.spin_scale.blockSignals(True)
                        self.spin_scale.setValue(float(native))
                        self.spin_scale.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        self.slider_scale.blockSignals(True)
                        self.slider_scale.setValue(int(native * 10))
                        self.slider_scale.blockSignals(False)
                    except Exception:
                        pass
                # Visibility
                vis = not isr
                try:
                    self.spin_scale.setVisible(vis)
                except Exception:
                    pass
                try:
                    self.slider_scale.setVisible(vis)
                except Exception:
                    pass
                try:
                    if lbl is not None:
                        lbl.setVisible(vis)
                except Exception:
                    pass
            except Exception:
                pass

        # ---------- Wrap __init__ to connect signals and apply on startup ----------
        if not hasattr(_Pane, "_fv_orig_init"):
            _Pane._fv_orig_init = _Pane.__init__
            def _fv_init(self, *a, **k):
                _Pane._fv_orig_init(self, *a, **k)
                try:
                    # Cache label and connect to updates
                    self._fv_lbl_scale = _fv_find_scale_label(self)
                    try:
                        self.combo_engine.currentIndexChanged.connect(lambda *_: _fv_apply_scale_ui(self))
                    except Exception:
                        pass
                    try:
                        self.combo_model_realsr.currentIndexChanged.connect(lambda *_: _fv_apply_scale_ui(self))
                    except Exception:
                        pass
                    _fv_apply_scale_ui(self)
                except Exception:
                    pass
            _Pane.__init__ = _fv_init
except Exception:
    pass


# --- FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH (2025-10-20) ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_patched_tooltips_modelinfo_20251020"):
        _Pane._fv_patched_tooltips_modelinfo_20251020 = True

        from PySide6 import QtWidgets as _FVQtW2  # type: ignore

        # Rich, sourced model info (short, single-line style)
        _FV_MODEL_HINTS = {
            # Real-ESRGAN family (primary)
            "realesrgan-x4plus": ("Photo", "General 4× model for real‑world photo/video; Can take a while to finish on longer videos."),
            "realesrnet-x4plus": ("Photo", "4× RealESRNet (no-GAN) — fewer hallucinated textures; stable on photos."),
            "realesrgan-x4plus-anime": ("Anime", "4× model tuned for anime/illustrations; preserves clean lines & flats."),
            "realesr-animevideov3-x4": ("Anime/Video", "4× anime‑video model (v3); reduces temporal artifacts on frames."),
            "realesr-general-x4v3": ("Photo (Tiny)", "4× general v3 (tiny) for blind SR; robust across varied content."),
            "realesr-general-wdn-x4v3": ("Photo+Denoise", "4× general v3 with built‑in denoise (WDN) to suppress compression noise."),
            # Popular community model
            "4x-ultrasharp": ("UI/Text", "4× UltraSharp: crisp, edge‑preserving detail; great for UI/screens; can look oversharp."),
            # Waifu2x model dirs
            "models-cunet": ("Anime", "Waifu2x CUNet model for anime‑style art with denoise levels."),
            "models-upconv_7_anime_style_art_rgb": ("Anime", "UpConv7 (anime RGB) — fast; keeps flat colors and linework."),
            "models-upconv_7_photo": ("Photo", "UpConv7 (photo) — better for natural images & gradients."),
            # SRMD (classical)
            "srmd": ("Classical SR", "SRMD classical SR; use SRMDNF for noise‑free variant; good on older photos."),
            "srmdnf": ("Classical SR", "SRMDNF (noise‑free) — removes noise prior to upscaling; stable results."),
            # RealSR (ncnn)
            "realsr-x2": ("Photo", "RealSR (x2) real‑world photo SR with degradation modeling; natural detail."),
            "realsr-x4": ("Photo", "RealSR (x4) real‑world photo SR; fewer artifacts on camera images."),
        }

        def _fv_lookup_hint(name: str, engine_label: str) -> tuple[str, str]:
            low = (name or "").lower().strip()
            elow = (engine_label or "").lower()
            # Direct keys or substring matches for known models
            for key, (cat, txt) in _FV_MODEL_HINTS.items():
                if key in low:
                    return cat, txt
            # Upscayl packaged models (label usually "Upscayl - <base>")
            if "upscayl" in low:
                if "anime" in low:
                    return "Anime", "Upscayl anime model based on ESRGAN family; strong line preservation."
                if "digital" in low or "art" in low:
                    return "Digital Art", "Upscayl digital‑art model; enhances CG and illustrations with sharp edges."
                return "Photo", "Upscayl general‑photo model (ESRGAN‑based) for everyday pictures."
            # Engine‑level fallbacks
            if "swinir" in elow:
                return "Transformer", "SwinIR (Transformer‑based SR) — strong fidelity on natural images."
            if "lapsrn" in elow:
                return "Pyramid CNN", "LapSRN (Laplacian pyramid CNN) — lightweight & fast SR."
            if "waifu2x" in elow:
                return "Anime", "Waifu2x family — best for anime/line art; choose CUNet/UpConv7 variants."
            if "realsr (" in elow:
                return "Photo", "RealSR (ncnn) for real‑world photos; good at reducing artifacts."
            if "srmd" in elow:
                return "Classical SR", "SRMD classical SR; SRMDNF variant includes denoise prior."
            # Generic
            if "ultrasharp" in low or "ui" in low or "text" in low:
                return "UI/Text", "Edge‑focused model for UI, text, and crisp graphics."
            if "anime" in low:
                return "Anime", "Optimized for anime/illustrations; preserves lines & flats."
            if "general" in low or "x4" in low or "x3" in low or "x2" in low:
                return "Photo", "Balanced detail for photographs across varied content."
            return "Other", ""

        def _fv_current_model_text(self) -> str:
            try:
                engine_label = self.combo_engine.currentText()
            except Exception:
                engine_label = ""
            try:
                page = int(self.stk_models.currentIndex())
            except Exception:
                page = 0
            # Choose the right combo for current engine/page
            if page == 0:
                if "Waifu2x" in engine_label and hasattr(self, "combo_model_w2x"):
                    return self.combo_model_w2x.currentText()
                if "UltraSharp" in engine_label and hasattr(self, "combo_model_ultrasharp"):
                    return self.combo_model_ultrasharp.currentText()
                if "SRMD (ncnn via RealESRGAN)" in engine_label and hasattr(self, "combo_model_srmd_realsr"):
                    return self.combo_model_srmd_realsr.currentText()
                return self.combo_model_realsr.currentText()
            # Other stacked pages (best‑effort)
            for nm in ("combo_model_ultrasharp","combo_model_srmd_realsr","combo_model_w2x","combo_model_srmd","combo_model_realsr_ncnn","combo_model_realsr"):
                w = getattr(self, nm, None)
                if w is not None and hasattr(w, "currentText"):
                    try:
                        return w.currentText()
                    except Exception:
                        pass
            try:
                return self.combo_model_realsr.currentText()
            except Exception:
                return ""

        # Replace the simple hint updater with the richer one
        def _fv_update_model_hint(self):
            try:
                engine_label = self.combo_engine.currentText()
            except Exception:
                engine_label = ""
            name = ""
            try:
                name = _fv_current_model_text(self)
            except Exception:
                pass
            cat, hint = _fv_lookup_hint(name, engine_label)
            try:
                self.lbl_model_badge.setText(cat)
            except Exception:
                pass
            try:
                self.lbl_model_hint.setText(hint)
            except Exception:
                pass

        # Install encoder/tool tooltips in a dedicated helper
        def _fv_install_tooltips(self):
            def tip(w, text):
                try:
                    if w is not None and hasattr(w, "setToolTip"):
                        w.setToolTip(text)
                except Exception:
                    pass

            # Video codec & rate control
            tip(getattr(self, "combo_vcodec", None),
                "Video codec: H.264 (libx264) is most compatible; H.265/HEVC (libx265) smaller files; AV1 (SVT‑AV1/libaom) best compression but slower/less supported.")
            tip(getattr(self, "rad_crf", None),
                "CRF = Constant Rate Factor (quality‑based). Lower CRF → higher quality & larger file. Typical: x264 ≈ 18–23, x265 ≈ 20–28. Ignored if Bitrate is selected.")
            tip(getattr(self, "spin_crf", None),
                "Set CRF quality level for quality‑based encoding. Lower is better quality/larger size.")
            tip(getattr(self, "rad_bitrate", None),
                "Bitrate mode targets an average video bitrate (kbps). Use when you need a fixed size/bitrate. Disables CRF.")
            tip(getattr(self, "spin_bitrate", None),
                "Target video bitrate in kbps (e.g., 8000 = 8 Mbps). Rough guide: 1080p 6–12 Mbps; 4K 12–35 Mbps (content dependent).")
            tip(getattr(self, "combo_preset", None),
                "Encoder speed/quality trade‑off. Slower presets compress better (smaller files for same quality).")
            tip(getattr(self, "spin_keyint", None),
                "Keyframe interval (GOP). 0 lets the encoder choose. Otherwise sets ffmpeg -g <N>.")

            # Audio controls
            tip(getattr(self, "radio_a_copy", None), "Copy audio stream without re‑encoding (fast; keeps original quality/codec).")
            tip(getattr(self, "radio_a_encode", None), "Encode audio with chosen codec/bitrate (use if changing container/codecs).")
            tip(getattr(self, "radio_a_mute", None), "Remove/mute audio from the output.")
            tip(getattr(self, "combo_acodec", None), "Audio codec: AAC (very compatible), Opus (efficient esp. speech), Vorbis (legacy/OGG).")
            tip(getattr(self, "spin_abitrate", None), "Audio bitrate (kbps). Common: AAC 128–192, Opus 96–160.")

            # Model badge/hint hover
            tip(getattr(self, "lbl_model_badge", None), "Category of the selected model.")
            tip(getattr(self, "lbl_model_hint", None), "selected model.")

        # Wrap __init__ to attach tooltips and hook our rich hint updater
        if not hasattr(_Pane, "_fv_orig_init_tooltips"):
            _Pane._fv_orig_init_tooltips = _Pane.__init__
            def _fv_init_plus(self, *a, **k):
                _Pane._fv_orig_init_tooltips(self, *a, **k)
                try:
                    # Install tooltips
                    _fv_install_tooltips(self)
                except Exception:
                    pass
                try:
                    # Rewire model-hint updater
                    self._update_model_hint = _fv_update_model_hint.__get__(self, _Pane)  # bind
                    # Run once to refresh
                    self._update_model_hint()
                    # Connect changes
                    for nm in ("combo_model_realsr", "combo_model_w2x",
                               "combo_model_ultrasharp", "combo_model_srmd_realsr",
                               "combo_model_srmd", "combo_model_realsr_ncnn", "combo_engine"):
                        w = getattr(self, nm, None)
                        try:
                            if hasattr(w, "currentTextChanged"):
                                w.currentTextChanged.connect(self._update_model_hint)
                        except Exception:
                            pass
                        try:
                            if hasattr(w, "currentIndexChanged"):
                                w.currentIndexChanged.connect(self._update_model_hint)
                        except Exception:
                            pass
                    try:
                        if hasattr(self, "stk_models"):
                            self.stk_models.currentChanged.connect(lambda *_: self._update_model_hint())
                    except Exception:
                        pass
                except Exception:
                    pass
            _Pane.__init__ = _fv_init_plus
except Exception as _fv_patch_exc:
    try:
        print("[upsc tooltip/modelinfo patch] non-fatal:", _fv_patch_exc)
    except Exception:
        pass
# --- END FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH ---



# --- FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH v2 (visible-combo fix) ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_visible_combo_fix_20251020b"):
        _Pane._fv_visible_combo_fix_20251020b = True

        def _fv_current_model_text(self):
            # Prefer the model combo that is actually visible in the stacked widget.
            combo_names = (
                "combo_model_realsr",
                "combo_model_w2x",
                "combo_model_ultrasharp",
                "combo_model_srmd_realsr",
                "combo_model_srmd",
                "combo_model_realsr_ncnn",
            )
            visible = []
            for nm in combo_names:
                w = getattr(self, nm, None)
                if w is None or not hasattr(w, "currentText"):
                    continue
                try:
                    if hasattr(w, "isVisible") and w.isVisible():
                        visible.append(w)
                except Exception:
                    # If visibility check fails, keep going
                    pass
            try:
                if visible:
                    return visible[0].currentText()
            except Exception:
                pass

            # If none are visible (rare), pick by engine label mapping.
            try:
                engine_label = self.combo_engine.currentText().lower()
            except Exception:
                engine_label = ""

            def pick(*names):
                for n in names:
                    w = getattr(self, n, None)
                    if w is not None and hasattr(w, "currentText"):
                        try:
                            return w.currentText()
                        except Exception:
                            pass
                return ""

            if "srmd" in engine_label:
                return pick("combo_model_srmd_realsr", "combo_model_srmd")
            if "waifu2x" in engine_label:
                return pick("combo_model_w2x")
            if "ultrasharp" in engine_label:
                return pick("combo_model_ultrasharp")
            if "realsr" in engine_label or "ncnn" in engine_label:
                return pick("combo_model_realsr", "combo_model_realsr_ncnn")

            # Fallback: first available
            return pick(*combo_names)

except Exception as _fv_fix_exc:
    try:
        print("[upsc tooltip/modelinfo patch v2] non-fatal:", _fv_fix_exc)
    except Exception:
        pass
# --- END v2 (visible-combo fix) ---


# --- FRAMEVISION_CLEARREALITY_V1_ENGINE_PATCH (2025-10-24) ---
# Adds a dedicated "ClearReality V1" engine option (re-using the Real-ESRGAN backend)
# and filters the model list to only ClearReality models when selected. Also augments
# model tooltips.
try:
    from PySide6 import QtCore as _FVQtCore2, QtWidgets as _FVQtW3  # type: ignore
except Exception:
    _FVQtCore2 = None
    _FVQtW3 = None

def _fv_scan_clearreality_models() -> list[str]:
    try:
        base = REALSR_DIR  # type: ignore  # provided by module
    except Exception:
        try:
            base = Path(__file__).resolve().parents[1] / "models" / "realesrgan"  # type: ignore
        except Exception:
            return []
    names: set[str] = set()
    try:
        if base.exists():
            for ext in ("*.bin", "*.param"):
                for p in sorted(base.glob(ext)):
                    s = p.stem.lower()
                    if "clearrealityv1" in s:
                        names.add(p.stem)
    except Exception:
        pass
    # Reasonable fallbacks if nothing found
    if not names:
        return ["4x-ClearRealityV1-fp16", "4x-ClearRealityV1-fp32",
                "4x-ClearRealityV1_Soft-fp16", "4x-ClearRealityV1_Soft-fp32"]
    return sorted(names)

# 1) Extend engine detection to include "ClearReality V1" (uses realesrgan-ncnn-vulkan)
try:
    _fv_orig_detect_engines = detect_engines  # type: ignore
except Exception:
    _fv_orig_detect_engines = None

def _fv_detect_engines_plus_clearreality():
    out = []
    try:
        if callable(_fv_orig_detect_engines):
            out = list(_fv_orig_detect_engines())  # type: ignore
    except Exception:
        out = []
    # Try to find the realesrgan executable used by other engines
    realsr_exe = None
    for label, exe in out:
        l = (label or "").lower()
        if ("real-esrgan" in l) or ("realesrgan" in l) or ("ultrasharp" in l) or ("srmd (ncnn via realesrgan" in l):
            realsr_exe = exe
            break
    if realsr_exe is None:
        try:
            cand = REALSR_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")  # type: ignore
            if cand.exists():
                realsr_exe = str(cand)
        except Exception:
            pass
    if realsr_exe and not any((lbl or "").lower().startswith("clearreality v1") for (lbl, _x) in out):
        # Place it near the top for convenience (after Real-ESRGAN if present)
        idx = 1 if out and "real-esrgan" in (out[0][0] or "").lower() else len(out)
        out.insert(idx, ("ClearReality V1", realsr_exe))
    return out

try:
    detect_engines = _fv_detect_engines_plus_clearreality  # type: ignore
except Exception:
    pass

# 2) Filter the Real-ESRGAN model combo when the "ClearReality V1" engine is selected.
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_clearreality_engine_patch_20251024"):
        _Pane._fv_clearreality_engine_patch_20251024 = True

        def _fv_restore_or_filter_models(self):
            try:
                eng = (self.combo_engine.currentText() or "").lower()
            except Exception:
                eng = ""
            combo = getattr(self, "combo_model_realsr", None)
            if combo is None or not hasattr(combo, "count"):
                return
            # Cache the full model list once
            full_key = "_fv_realsr_all_items"
            if not hasattr(self, full_key):
                try:
                    all_items = []
                    for i in range(combo.count()):
                        all_items.append(combo.itemText(i))
                    setattr(self, full_key, all_items)
                except Exception:
                    setattr(self, full_key, [])
            all_items = list(getattr(self, full_key, []))

            def _repopulate(items: list[str]):
                try:
                    combo.blockSignals(True)
                    combo.clear()
                    for it in items:
                        combo.addItem(it)
                finally:
                    try:
                        combo.blockSignals(False)
                    except Exception:
                        pass
                # refresh hint, if available
                try:
                    if hasattr(self, "_update_model_hint"):
                        self._update_model_hint()
                except Exception:
                    pass

            if "clearreality v1" in eng:
                # Only show ClearReality models (scan directly from disk)
                try:
                    items = list(_fv_scan_clearreality_models())
                except Exception:
                    items = []
 #           elif "bstexty for text upscaling" in eng:
  #              # Only show BStexty models
   #             items = [s for s in all_items if "bstexty" in (s or "").lower()]
 #           elif "fatality noisetoner for sharping/denoising" in eng:
    #            # Only show Fatality NoiseToner models
     #           items = [s for s in all_items if ("fatality" in (s or "").lower()) or ("noisetoner" in (s or "").lower())]
            elif "real-esrgan" in eng:
                # Real-ESRGAN engine: hide BStexty/Fatality models (they have their own engines)
                items = [s for s in all_items if ("bstexty" not in (s or "").lower()) and ("fatality" not in (s or "").lower()) and ("noisetoner" not in (s or "").lower())]
            else:
                items = None

            if items is not None:
                # If nothing matched, fall back to the full list so the combo never ends up empty.
                if not items:
                    items = list(all_items)
                _repopulate(items)
            else:
                # Restore full list if it looks filtered
                if combo.count() != len(all_items):
                    _repopulate(all_items)

        # Hook into __init__ to wire signal + run once
        _orig_init = _Pane.__init__
        def _init_and_wire(self, *a, **k):
            _orig_init(self, *a, **k)
            try:
                if hasattr(self.combo_engine, "currentTextChanged"):
                    self.combo_engine.currentTextChanged.connect(lambda *_: _fv_restore_or_filter_models(self))
            except Exception:
                pass
            try:
                _fv_restore_or_filter_models(self)
            except Exception:
                pass
        _Pane.__init__ = _init_and_wire

except Exception as _fv_cex:
    try:
        print("[ClearReality engine patch] non-fatal:", _fv_cex)
    except Exception:
        pass

# 3) Remove ClearReality entries from the generic Real-ESRGAN scan (to avoid duplicates)
try:
    _fv_orig_scan_realsr = scan_realsr_models  # type: ignore
    def scan_realsr_models():  # type: ignore
        try:
            items = list(_fv_orig_scan_realsr())
        except Exception:
            items = []
        # filter out ClearReality models; they will live under the dedicated engine
        return [i for i in items if "clearreality" not in (i or "").lower()]
except Exception:
    pass

# 4) Extend tooltip map for ClearReality variants (short, single-line style)
try:
    _cr_hints = {
        "4x-clearrealityv1-fp16": ("Photo/Sharp", "ClearReality V1 (fp16) — crisp detail and punchy edges; fast."),
        "4x-clearrealityv1-fp32": ("Photo/Sharp", "ClearReality V1 (fp32) — higher precision; slightly slower; crisp detail."),
        "4x-clearrealityv1_soft-fp16": ("Photo/Soft", "ClearReality V1 Soft (fp16) — gentler sharpening; fewer halos."),
        "4x-clearrealityv1_soft-fp32": ("Photo/Soft", "ClearReality V1 Soft (fp32) — soft variant with fp32 precision."),
        "clearrealityv1": ("Photo", "ClearReality V1 family — Real-ESRGAN fine‑tune with clean detail. Use Soft to reduce halos."),
    }
    try:
        _FV_MODEL_HINTS.update(_cr_hints)  # type: ignore
    except Exception:
        # If the dict wasn't defined yet, create a minimal overlay and a wrapper resolver
        _FV_MODEL_HINTS = dict(_cr_hints)  # type: ignore
except Exception:
    pass
# --- END FRAMEVISION_CLEARREALITY_V1_ENGINE_PATCH ---



# --- FRAMEVISION_HIDE_ENCODER_STYLE_IN_REALESRGAN (2025-10-24) ---
# Hides 'encoder' and 'style' entries from the Real-ESRGAN model dropdown, globally.
try:
    _fv_prev_scan_realsr_2 = scan_realsr_models  # type: ignore
    def scan_realsr_models():  # type: ignore
        try:
            items = list(_fv_prev_scan_realsr_2())
        except Exception:
            items = []
        blocked = {"encoder", "style"}
        out = []
        for i in items:
            s = (i or "").strip()
            low = s.lower()
            if low in blocked:
                continue
            # Also maintain prior behavior of removing ClearReality from the generic list
            if "clearreality" in low:
                continue
            out.append(s)
        return out
except Exception:
    pass

# Also prune any pre-populated combos on construction, just in case.
try:
    _Pane2 = UpscPane  # type: ignore
    if not hasattr(_Pane2, "_fv_hide_encoder_style_runtime_20251024"):
        _orig_init_hide = _Pane2.__init__
        def _init_hide(self, *a, **k):
            _orig_init_hide(self, *a, **k)
            try:
                combo = getattr(self, "combo_model_realsr", None)
                if combo is not None and hasattr(combo, "count"):
                    blocked = {"encoder", "style"}
                    for i in reversed(range(combo.count())):
                        t = (combo.itemText(i) or "").strip().lower()
                        if t in blocked:
                            combo.removeItem(i)
            except Exception:
                pass
        _Pane2.__init__ = _init_hide
        _Pane2._fv_hide_encoder_style_runtime_20251024 = True
except Exception:
    pass
# --- END FRAMEVISION_HIDE_ENCODER_STYLE_IN_REALESRGAN ---

