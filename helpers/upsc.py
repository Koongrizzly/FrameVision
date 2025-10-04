
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
from PySide6.QtGui import QTextCursor, QPixmap, QIcon

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

PRESETS_BIN = ROOT / "presets" / "bin"
BIN_DIR = ROOT / "bin"
MODELS_DIR = ROOT / "models"
REALSR_DIR = MODELS_DIR / "realesrgan"
WAIFU2X_DIR = MODELS_DIR / "waifu2x"

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
    waifu = WAIFU2X_DIR / ("waifu2x-ncnn-vulkan.exe" if os.name == "nt" else "waifu2x-ncnn-vulkan")
    if waifu.exists():
        engines.append(("Waifu2x (ncnn)", str(waifu)))
    return engines


def scan_realsr_models() -> List[str]:
    names: set[str] = set()
    if REALSR_DIR.exists():
        for ext in ("*.bin", "*.param"):
            for p in sorted(REALSR_DIR.glob(ext)):
                names.add(p.stem)
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



def scan_waifu2x_models() -> List[str]:
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
        self._last_outfile: Optional[Path] = None
        # LRU cache for thumbnails (path+size+radius)
        self._pm_cache = OrderedDict()
        self._pm_cache_cap = 128
        self._build_ui()

    def set_main(self, main):  # optional hook
        self._main = main

    def _build_ui(self):
        v_main = QtWidgets.QVBoxLayout(self)
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
        self.spin_scale.setRange(1.0, 4.0)
        self.spin_scale.setSingleStep(0.5)
        self.spin_scale.setValue(2.0)
        self.spin_scale.setDecimals(1)
        self.slider_scale = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider_scale.setMinimum(10)
        self.slider_scale.setMaximum(40)
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

        # --- Recent results (rebuilt) ---
        rec_box = CollapsibleSection("Recent results", expanded=True, parent=self)
        self.recents_box = rec_box

        rec_wrap = QtWidgets.QVBoxLayout(); rec_wrap.setContentsMargins(6,2,6,6); rec_wrap.setSpacing(6)

        # Size slider
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Thumb size:", self))
        self.sld_recent_size = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sld_recent_size.setMinimum(32)
        self.sld_recent_size.setMaximum(120)
        self.sld_recent_size.setSingleStep(4)
        self.sld_recent_size.setPageStep(8)
        self.sld_recent_size.setValue(75)
        size_row.addWidget(self.sld_recent_size, 1)
        self.lbl_recent_size = QtWidgets.QLabel("75 px", self)
        size_row.addWidget(self.lbl_recent_size)
        rec_wrap.addLayout(size_row)

        # Horizontal scroller with items in a row
        self.recents_scroll = QtWidgets.QScrollArea(self)
        self.recents_scroll.setWidgetResizable(True)
        self.recents_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recents_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        
        try:
            # Fix '1-per-row' by triggering rebuilds on width changes
            self.recents_scroll.viewport().installEventFilter(self)
        except Exception:
            pass

        self._recents_inner = QtWidgets.QWidget(self)
        self._recents_row = QtWidgets.QGridLayout(self._recents_inner)  # wrapped grid
        self._recents_row.setContentsMargins(0,0,0,0)
        self._recents_row.setSpacing(8)
        self.recents_scroll.setWidget(self._recents_inner)
        rec_wrap.addWidget(self.recents_scroll)

        try:
            self.recents_box.setContentLayout(rec_wrap)
        except Exception:
            _rw = QtWidgets.QWidget(); _rw.setLayout(rec_wrap); self.recents_box.addWidget(_rw)
        v.addWidget(rec_box)
        
        try:
            QTimer.singleShot(0, self._rebuild_recents)
        except Exception:
            pass

# start recents poller
        try:
            self._install_recents_poller()
        except Exception:
            pass

        # Wires for size slider
        def _on_recent_size(val):
            try:
                self.lbl_recent_size.setText(f"{val} px")
            except Exception:
                pass
            try:
                self._rebuild_recents()
            except Exception:
                pass

        try:
            self.sld_recent_size.valueChanged.connect(_on_recent_size)
        except Exception:
            pass

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
        # Preset + Keyint
        lay_enc.addWidget(QtWidgets.QLabel("Preset:", self), 3, 0)
        self.combo_preset = QtWidgets.QComboBox(self)
        for p in ("ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"):
            self.combo_preset.addItem(p)
        self.combo_preset.setCurrentText("veryfast")
        lay_enc.addWidget(self.combo_preset, 3, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Keyint (GOP):", self), 4, 0)
        self.spin_keyint = QtWidgets.QSpinBox(self); self.spin_keyint.setRange(0, 1000); self.spin_keyint.setValue(0)
        self.spin_keyint.setToolTip("0 = let encoder decide; otherwise sets -g keyint")
        lay_enc.addWidget(self.spin_keyint, 4, 1)
        # Audio
        lay_enc.addWidget(QtWidgets.QLabel("Audio:", self), 5, 0)
        self.radio_a_copy = QtWidgets.QRadioButton("Copy", self); self.radio_a_copy.setChecked(True)
        self.radio_a_encode = QtWidgets.QRadioButton("Encode", self)
        self.radio_a_mute = QtWidgets.QRadioButton("Mute", self)
        arow = QtWidgets.QHBoxLayout(); arow.addWidget(self.radio_a_copy); arow.addWidget(self.radio_a_encode); arow.addWidget(self.radio_a_mute); arow.addStretch(1)
        lay_enc.addLayout(arow, 5, 1)
        # Audio codec/bitrate
        lay_enc.addWidget(QtWidgets.QLabel("Audio codec:", self), 6, 0)
        self.combo_acodec = QtWidgets.QComboBox(self)
        for ac in ("aac","libopus","libvorbis"):
            self.combo_acodec.addItem(ac)
        lay_enc.addWidget(self.combo_acodec, 6, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Audio bitrate (kbps):", self), 7, 0)
        self.spin_abitrate = QtWidgets.QSpinBox(self); self.spin_abitrate.setRange(32, 1024); self.spin_abitrate.setValue(192)
        lay_enc.addWidget(self.spin_abitrate, 7, 1)
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

        # ----- Fixed bottom action bar (does not scroll) -----
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.btn_upscale)
        bottom.addWidget(self.btn_batch)
        bottom.addWidget(self.btn_info)
        bottom.addStretch(1)
        v_main.addLayout(bottom)

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


        # --- Settings + Recents wiring ---
        try:
            self.list_recents.itemClicked.connect(self._open_recent_item)
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
            from PySide6.QtCore import QEvent
            if hasattr(self, "recents_scroll") and obj is self.recents_scroll.viewport():
                if ev.type() == QEvent.Resize:
                    w = ev.size().width()
                    if w != getattr(self, "_recents_last_w", 0):
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
            ("spin_abitrate", "valueChanged"),
            ("spin_tile", "valueChanged"),
            ("spin_overlap", "valueChanged"),
            ("chk_deinterlace", "toggled"),
            ("combo_range", "currentTextChanged"),
            ("chk_deblock", "toggled"),
            ("chk_denoise", "toggled"),
            ("chk_deband", "toggled"),
            ("sld_sharpen", "valueChanged"),
                    ]

        connected = 0
        for name, sig_name in pairs:
            obj = getattr(self, name, None)
            if _connect(obj, sig_name):
                connected += 1

        # Collapsible sections (their .toggle has a .toggled signal)
        for box_name in ("box_models", "box_encoder", "box_advanced", "recents_box"):
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
        try: d["keyint"] = int(self.spin_keyint.value())
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
        try: d["abitrate"] = int(self.spin_abitrate.value())
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
        try:
            if hasattr(self, "recents_box"):
                d["sections"]["recents"] = bool(self.recents_box.toggle.isChecked())
            else:
                d["sections"]["recents"] = True
        except Exception:
            d["sections"]["recents"] = True
        # Batch / gallery options
        try: d["video_thumbs"] = bool(self.chk_video_thumbs.isChecked())
        except Exception: d["video_thumbs"] = False
        try: d["streaming_lowmem"] = bool(self.chk_streaming_lowmem.isChecked())
        except Exception: d["streaming_lowmem"] = True
        # Recents list
        try: d["recents"] = [str(p) for p in getattr(self, "_recents", [])]
        except Exception: pass
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
            self.spin_keyint.setValue(int(d.get("keyint", 0)))
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
            self.spin_abitrate.setValue(int(d.get("abitrate", 192)))
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
            if hasattr(self, "recents_box"):
                self._set_section_state(self.recents_box, bool(secs.get("recents", True)))
        except Exception:
            pass
        try:
            self.chk_video_thumbs.setChecked(bool(d.get("video_thumbs", False)))
        except Exception:
            pass
        try:
            self.chk_streaming_lowmem.setChecked(bool(d.get("streaming_lowmem", True)))
        except Exception:
            pass
        except Exception:
            pass
        # Recents
        try:
            recs = []
            for p in d.get("recents", []) or []:
                from pathlib import Path as _P
                pp = _P(p)
                if pp.exists():
                    recs.append(pp)
            self._recents = recs[:15]
            self._rebuild_recents()
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

    def _open_recent_item(self, item):
        try:
            from pathlib import Path as _P
            p = _P(item.data(Qt.UserRole))
            if p.exists():
                if not self._play_in_player(p):
                    self._open_file(p)
        except Exception:
            pass

    def _add_recent(self, p: Path):
        try:
            if not hasattr(self, "_recents") or self._recents is None:
                self._recents = []
            from pathlib import Path as _P
            p = _P(p)
            if not p.exists():
                return
            self._recents = [x for x in self._recents if _P(x) != p]
            self._recents.insert(0, p)
            self._recents = self._recents[:10]
            self._rebuild_recents()
            self._save_settings()
        except Exception:
            pass

    def _refresh_recents_gallery(self):
        try:
            self.list_recents.clear()
            for p in getattr(self, "_recents", [])[:10]:
                it = QtWidgets.QListWidgetItem()
                it.setText(p.name)
                try:
                    pm = self._make_thumb(p)
                    if pm and not pm.isNull():
                        it.setIcon(QIcon(pm))
                except Exception:
                    pass
                it.setData(Qt.UserRole, str(p))
                self.list_recents.addItem(it)
        except Exception:
            pass


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

            files = self._list_last_results()
            if not files:
                lab = QtWidgets.QLabel("No results yet.", self)
                lab.setStyleSheet("color:#9fb3c8;")
                layout.addWidget(lab)
                return

            for p in files:
                # ensure tiny thumb exists
                tp = self._thumb_for(p, 120)
                btn = QtWidgets.QToolButton(self)
                btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                btn.setText(p.name)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setAutoRaise(True)
                if tp:
                    btn.setIcon(QIcon(str(tp)))
                btn.setIconSize(QSize(size, size))
                btn.setFixedSize(int(size*1.25), int(size*1.25)+28)

                def _mk_open(path: Path):
                    def _open():
                        try:
                            if not self._play_in_player(path):
                                _open_file(self, path)
                        except Exception:
                            _open_file(self, path)
                    return _open
                btn.clicked.connect(_mk_open(p))
                layout.addWidget(btn)

            # ensure the scroll area can expand vertically if needed
            try:
                self._recents_inner.setMinimumHeight(int(size*1.25)+36)
                try:
                    vpw = self.recents_scroll.viewport().width()
                except Exception:
                    vpw = self._recents_inner.width()
                spacing = getattr(layout, "spacing", lambda: 8)()
                item_w = int(size*1.25)
                cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                rows = max(1, (getattr(self, "_recents_idx", 0)+cols-1)//cols)
                self._recents_inner.setMinimumHeight(rows * (int(size*1.25)+28) + 12)
            except Exception:
                pass

        except Exception as e:
            try:
                self._append_log(f"[recents] rebuild error: {e}")
            except Exception:
                pass

    def _update_engine_ui(self):
        # Keep model stack in sync with engine
        self.stk_models.setCurrentIndex(0 if self.combo_engine.currentIndex() == 0 else 1)
        # Auto-guide scale to the model's native when using Real-ESRGAN
        try:
            eng = (self.combo_engine.currentText() or '').lower()
        except Exception:
            eng = ''
        if ('realesrgan' in eng) or ('real-esrgan' in eng):
            try:
                model = self.combo_model_realsr.currentText()
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

        self.stk_models.setCurrentIndex(0 if self.combo_engine.currentIndex() == 0 else 1)

    def _sync_scale_from_spin(self, v: float):
        self.slider_scale.blockSignals(True)
        self.slider_scale.setValue(int(round(v * 10)))
        self.slider_scale.blockSignals(False)
        try:
            self._update_batch_limit()
        except Exception:
            pass

    # ===== Recents (interp) — driven by finished jobs =====
    def _jobs_done_dir(self) -> Path:
        try:
            base = ROOT
        except Exception:
            base = Path(__file__).resolve().parent.parent
        return base / "jobs" / "done"

    def _recents_dir(self) -> Path:
        try:
            base = ROOT
        except Exception:
            base = Path(__file__).resolve().parent.parent
        # Thumbs only live here
        return base / "output" / "last results" / "interp"

    def _list_recent_jobs(self) -> list[Path]:
        d = self._jobs_done_dir()
        try:
            if not d.exists():
                return []
            items = [p for p in d.iterdir() if p.suffix.lower()==".json" and p.is_file()]
            items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return items[:15]
        except Exception:
            return []

    def _resolve_output_from_job(self, job_json: Path) -> tuple[Path|None, dict]:
        """Return (media_path, job_data) for a finished job JSON."""
        try:
            j = json.loads(job_json.read_text(encoding="utf-8"))
        except Exception:
            j = {}
        def _as_path(val):
            if not val: return None
            try:
                p = Path(str(val)).expanduser()
                if not p.is_absolute():
                    out_dir = j.get("out_dir") or (j.get("args") or {}).get("out_dir")
                    if out_dir:
                        p = Path(out_dir).expanduser() / p
                return p
            except Exception:
                return None
        # Priority fields
        for k in ("produced","outfile","output","result","file","path"):
            v = j.get(k) or (j.get("args") or {}).get(k)
            p = _as_path(v)
            if p and p.exists() and p.is_file():
                return p, j
        # List fields
        for k in ("outputs","produced_files","results","files","artifacts","saved"):
            seq = j.get(k) or (j.get("args") or {}).get(k)
            if isinstance(seq, (list, tuple)):
                for v in seq:
                    p = _as_path(v)
                    if p and p.exists() and p.is_file():
                        return p, j

        # ## PATCH derive expected media from input/factor/format
        try:
            args = (j.get("args") or {})
            inp = (j.get("input") or args.get("input") or "").strip()
            fac = int(args.get("factor") or 0)
            fmt = (args.get("format") or "png").lower()
            out_dir_val = j.get("out_dir") or args.get("out_dir")
            from pathlib import Path as _P
            if inp and fac and out_dir_val:
                cand = _P(str(out_dir_val)).expanduser() / f"{_P(inp).stem}_x{fac}.{fmt}"
                if cand.exists() and cand.is_file():
                    return cand, j
        except Exception:
            pass

        # Fallback: scan out_dir for newest media
        media_exts = {'.mp4','.mov','.mkv','.avi','.webm','.gif','.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
        out_dir = _as_path(j.get("out_dir") or (j.get("args") or {}).get("out_dir"))
        try:
            if out_dir and out_dir.exists():
                cand = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in media_exts]
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if cand:
                    return cand[0], j
        except Exception:
            pass
        return None, j

    def _thumb_path_for_job(self, job_json: Path, media_path: Path, max_side: int) -> Path:
        d = self._recents_dir(); d.mkdir(parents=True, exist_ok=True)
        # ## PATCH unique thumb per job

        name = f"{job_json.stem}__{media_path.stem}.__thumb.jpg"
        return d / name

    def _thumb_from_media(self, media_path: Path, target_jpg: Path, max_side: int) -> None:
        try:
            if max_side < 32: max_side = 32
            if max_side > 120: max_side = 120
            # if media is video -> ffmpeg first frame; else use Qt decode
            vext = {'.mp4','.mov','.mkv','.avi','.webm'}
            if media_path.suffix.lower() in vext:
                cmd = [FFMPEG, "-hide_banner", "-loglevel", "quiet", "-y",
                       "-fflags", "nobuffer", "-probesize", "64k", "-analyzeduration", "0",
                       "-i", str(media_path),
                       "-vf", f"thumbnail,scale={int(max_side)}:-1",
                       "-frames:v", "1", str(target_jpg)]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            else:
                from PySide6.QtGui import QImageReader
                reader = QImageReader(str(media_path))
                reader.setAutoTransform(True)
                sz = reader.size()
                if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                    w, h = sz.width(), sz.height()
                    if w >= h:
                        reader.setScaledSize(QSize(int(max_side), max(1, int(h * max_side / max(1, w)))))
                    else:
                        reader.setScaledSize(QSize(max(1, int(w * max_side / max(1, h))), int(max_side)))
                img = reader.read()
                if not img.isNull():
                    img.save(str(target_jpg), "JPG", 70)
        except Exception:
            pass

    def _ensure_recent_thumb(self, job_json: Path, media: Path, max_side: int) -> Path|None:
        try:
            t = self._thumb_path_for_job(job_json, media, max_side)
            if (not t.exists()) or (t.stat().st_mtime < media.stat().st_mtime):
                self._thumb_from_media(media, t, max_side)
            return t if t.exists() else None
        except Exception:
            return None


    def _load_pixmap_cached(self, path: "Path", size: int, rounded: bool = False, radius: int | None = None):
        """Load and scale a QPixmap with LRU(128) caching; optionally return a rounded pixmap."""
        try:
            key = f"{str(path)}|{int(size)}|r{int(radius) if (rounded and radius is not None) else 0}"
            if hasattr(self, "_pm_cache") and key in self._pm_cache:
                pm = self._pm_cache.pop(key)
                self._pm_cache[key] = pm  # refresh LRU
                return pm
            from PySide6.QtGui import QPixmap, QPainter, QPainterPath
            from PySide6.QtCore import QRectF, Qt
            pm = QPixmap(str(path))
            if pm and not pm.isNull():
                pm2 = pm.scaled(int(size), int(size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if rounded:
                    try:
                        R = int(radius) if radius is not None else max(6, int(size*0.18))
                        out = QPixmap(int(size), int(size))
                        out.fill(Qt.transparent)
                        painter = QPainter(out)
                        painter.setRenderHint(QPainter.Antialiasing, True)
                        x = max(0, (int(size) - pm2.width()) // 2)
                        y = max(0, (int(size) - pm2.height()) // 2)
                        path = QPainterPath()
                        path.addRoundedRect(QRectF(x, y, pm2.width(), pm2.height()), R, R)
                        painter.setClipPath(path)
                        painter.drawPixmap(x, y, pm2)
                        painter.end()
                        pm2 = out
                    except Exception:
                        pass
                try:
                    self._pm_cache[key] = pm2
                    while len(self._pm_cache) > getattr(self, "_pm_cache_cap", 128):
                        self._pm_cache.popitem(last=False)
                except Exception:
                    pass
                return pm2
        except Exception:
            pass
        try:
            from PySide6.QtGui import QPixmap as _QPM
            return _QPM()
        except Exception:
            return None

    def _install_recents_poller(self):
        # Lightweight: poll jobs/done timestamps every second and rebuild only on change
        try:
            if getattr(self, "_recents_poller", None):
                return
            from PySide6.QtCore import QTimer
            self._recents_seen_ts = 0.0
            def _tick():
                try:
                    d = self._jobs_done_dir()
                    if not d.exists():
                        return
                    latest = 0.0
                    for p in d.glob("*.json"):
                        ts = p.stat().st_mtime
                        if ts > latest:
                            latest = ts
                    if latest > getattr(self, "_recents_seen_ts", 0.0):
                        self._recents_seen_ts = latest
                        self._rebuild_recents()
                except Exception:
                    pass
            t = QTimer(self)
            t.setInterval(1000)
            t.timeout.connect(_tick)
            t.start()
            self._recents_poller = t
        except Exception:
            pass

    def _rebuild_recents(self):
        """Rebuild the horizontal recents row from jobs/done JSONs. Low-resource, no heavy loading."""
        try:
            layout = getattr(self, "_recents_row", None)
            if layout is None:
                return
            self._recents_idx = 0
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            size = 96
            try: size = int(self.sld_recent_size.value())
            except Exception: pass

            jobs = self._list_recent_jobs()
            if not jobs:
                lab = QtWidgets.QLabel("No results yet.", self)
                lab.setStyleSheet("color:#9fb3c8;")
                layout.addWidget(lab)
                return

            for jpath in jobs:
                media, j = self._resolve_output_from_job(jpath)
                if not (media and media.exists() and media.is_file()):
                    continue
                tp = self._ensure_recent_thumb(jpath, media, 120)

                btn = QtWidgets.QToolButton(self)
                btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                btn.setText(Path(media).name)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setAutoRaise(True)
                try:
                    btn.setStyleSheet("QToolButton{border:1px solid rgba(255,255,255,0.12);border-radius:12px;padding:6px 6px 10px 6px;background:rgba(255,255,255,0.03);}QToolButton:hover{background:rgba(255,255,255,0.06);}")
                except Exception:
                    pass
                if tp:
                    try:
                        pm = self._load_pixmap_cached(tp, size, rounded=True, radius=12)
                        if pm:
                            btn.setIcon(QIcon(pm))
                        else:
                            btn.setIcon(QIcon(str(tp)))
                    except Exception:
                        btn.setIcon(QIcon(str(tp)))
                btn.setIconSize(QSize(size, size))
                btn.setFixedSize(int(size*1.25), int(size*1.25)+28)

                def _mk_open(path: Path):
                    def _open():
                        try:
                            if not self._play_in_player(path):
                                _open_file(self, path)
                        except Exception:
                            _open_file(self, path)
                    return _open
                btn.clicked.connect(_mk_open(media))
                # grid placement with wrapping
                try:
                    vpw = self.recents_scroll.viewport().width()
                except Exception:
                    vpw = self._recents_inner.width()
                try:
                    if not vpw or vpw <= 1:
                        vpw = max(self.recents_scroll.width(), self.width(), 600)
                except Exception:
                    vpw = 600
                spacing = getattr(layout, "spacing", lambda: 8)()
                item_w = int(size*1.25)
                if vpw <= item_w + spacing and len(jobs) > 1:
                    cols = min(len(jobs), 4)
                else:
                    cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                idx = getattr(self, "_recents_idx", 0)
                row = idx // cols
                col = idx % cols
                setattr(self, "_recents_idx", idx+1)
                layout.addWidget(btn, row, col)

            try:
                self._recents_inner.setMinimumHeight(int(size*1.25)+36)
            except Exception:
                pass

        except Exception as e:
            try: self._append_log(f"[recents] rebuild error: {e}")
            except Exception: pass


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
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Batch input")
        dlg.setText("Pick a folder or choose files.")
        btn_folder = dlg.addButton("Folder…", QtWidgets.QMessageBox.AcceptRole)
        btn_files = dlg.addButton("Files…", QtWidgets.QMessageBox.ActionRole)
        dlg.addButton(QtWidgets.QMessageBox.Cancel)
        dlg.exec()
        clicked = dlg.clickedButton()
        if clicked is btn_folder:
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder", str(ROOT))
            if d:
                self._run_batch_from_folder(Path(d))
        elif clicked is btn_files:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select files", str(ROOT))
            if files:
                self._run_batch_files([Path(f) for f in files])

    # Core
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
        return outd / f"{src.stem}_x{scale}{src.suffix}"

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
            scale = max(1, min(4, scale))
            outd = Path(self.edit_outdir.text().strip()) if self.edit_outdir.text().strip() else (OUT_VIDEOS if is_video else OUT_SHOTS)
            outfile = self._build_outfile(src, outd, scale)
            self._last_outfile = outfile
    
            if is_video:
                if "Waifu2x" in engine_label:
                    QtWidgets.QMessageBox.information(self, "Not supported", "Waifu2x (ncnn) handles images only for now. Please select Real-ESRGAN for videos.")
                    self._append_log("Waifu2x selected for a video — blocked (images only).")
                    return
    
                model = self.combo_model_realsr.currentText()
                fps = _parse_fps(src) or "30"
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
                    cmd_encode += ["-vf", post]
                if self.radio_a_mute.isChecked():
                    cmd_encode += ["-an"]
                elif self.radio_a_copy.isChecked():
                    cmd_encode += ["-c:a", "copy"]
                else:
                    cmd_encode += ["-c:a", self.combo_acodec.currentText(), "-b:a", f"{self.spin_abitrate.value()}k"]
                cmd_encode += ["-shortest", str(outfile)]
    
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
    waifu = _find(models / "waifu2x", ["waifu2x-ncnn-vulkan.exe","waifu2x-ncnn-vulkan"])
    if waifu: engines.append(("Waifu2x (ncnn)", str(waifu)))
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
                return ('realesrgan' in t) or ('real-esrgan' in t)
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