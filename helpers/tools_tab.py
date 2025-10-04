from PySide6.QtWidgets import QMessageBox
import subprocess
# helpers/tools_tab.py — extracted Tools pane (modular)
import os, re, subprocess
from pathlib import Path
from helpers.meme_tool import MemeToolPane
from helpers.trim_tool import install_trim_tool
from helpers.audiotool import install_audio_tool
from helpers.prompt import install_prompt_tool

from helpers.renam import RenamPane

import re

def _slug_title(t:str)->str:
    try:
        return re.sub(r'[^a-z0-9]+','_', (t or '').lower()).strip('_')
    except Exception:
        return 'section'
# Ensure widget classes are imported for use below
from PySide6.QtWidgets import QSpinBox, QSlider, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSettings, QTimer
from PySide6.QtCore import QDate, QTime, QDateTime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFormLayout, QSizePolicy, QToolButton, QGroupBox, QLineEdit, QComboBox, QCheckBox, QMessageBox, QFileDialog)

from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QRadioButton, QDateEdit, QTimeEdit, QDateTimeEdit, QMessageBox
from helpers import gif as gif_backend

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS, OUT_TRIMS, OUT_SHOTS, OUT_TEMP
except Exception:
    from pathlib import Path as _Path
    ROOT = _Path('.').resolve()
    BASE = ROOT
    OUT_VIDEOS = BASE/'output'/'video'
    OUT_TRIMS  = BASE/'output'/'trims'
    OUT_SHOTS  = BASE/'output'/'screenshots'
    OUT_TEMP   = BASE/'output'/'_temp'

def ffmpeg_path():
    candidates = [ROOT/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), "ffmpeg"]
    for c in candidates:
        try: subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT); return str(c)
        except Exception: continue
    return "ffmpeg"


def ffprobe_path():
    candidates = [Path('.').resolve()/'bin'/('ffprobe.exe' if os.name=='nt' else 'ffprobe'), 'ffprobe']
    for c in candidates:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT); return str(c)
        except Exception: continue
    return 'ffprobe'

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
QSS_DAY = """
QWidget { background:#f3f7fb; color:#111; }
QPushButton { background:#e8f0ff; border:1px solid #a8c4ff; border-radius:8px; padding:6px 10px; }
QPushButton:hover { background:#dbe7ff; }
QPushButton:pressed { background:#cbd9fb; }
QMenuBar, QMenu { background:#f3f7fb; color:#111; }
QMenu::item:selected { background:#e0e8ff; }
QGroupBox { background:transparent; border:1px solid #cfd8ea; border-radius:8px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#f3f7fb; }
QTabWidget::pane { background:#f3f7fb; border:1px solid #cfd8ea; border-radius:10px; }
QTabBar::tab { background:#e9effa; color:#111; padding:6px 12px; border:1px solid #cfd8ea; border-bottom: none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#ffffff; }
QTabBar::tab:hover { background:#f6fbff; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#ffffff; color:#111; border:1px solid #cfd8ea; border-radius:8px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#ffffff; color:#111; border:1px solid #cfd8ea; border-radius:8px; }
QSlider::groove:horizontal { height:6px; background:#c9cfd9; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#6a8cff; border:1px solid #3d5bd6; margin:-5px 0; border-radius:7px; }
QToolTip { background:#ffffff; color:#111; border:1px solid #a8c4ff; }
QMenuBar::item { background:#f3f7fb; color:#111; padding:4px 8px; }
QMenuBar::item:selected { background:#e0e8ff; }
QMenuBar::item:pressed { background:#e0e8ff; }

"""
QSS_EVENING = """


QWidget { background:#808080; color:#e7eefc; }              /* dark gray */
QPushButton { background:#2b3540; color:#e7eefc; border:1px solid #3a4a58; border-radius:12px; padding:8px 12px; }
QPushButton:hover { background:#34404c; }
QPushButton:pressed { background:#2a333d; }
QMenuBar, QMenu { background:#808080; color:#e7eefc; }
QMenu::item:selected { background:#2b3540; }
QGroupBox { background:transparent; border:1px solid #2f3a45; border-radius:10px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#808080; }
QTabWidget::pane { background:#808080; border:1px solid #2f3a45; border-radius:10px; }
QTabBar::tab { background:#222a33; color:#e7eefc; padding:6px 12px; border:1px solid #2f3a45; border-bottom:none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#2b3540; }
QTabBar::tab:hover { background:#2a333d; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#141920; color:#e7eefc; border:1px solid #2f3a45; border-radius:10px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#141920; color:#e7eefc; border:1px solid #2f3a45; border-radius:10px; }
QSlider::groove:horizontal { height:6px; background:#26303a; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#49617a; border:1px solid #394d62; margin:-5px 0; border-radius:7px; }
QToolTip { background:#222a33; color:#e7eefc; border:1px solid #3a4a58; }
QMenuBar::item { background:#1a1f26; color:#e7eefc; padding:4px 8px; }
QMenuBar::item:selected { background:#2b3540; }
QMenuBar::item:pressed { background:#2b3540; }



"""
QSS_NIGHT = """
QWidget { background:#0b1118; color:#d8e6ff; }
QPushButton { background:#0f1b2a; color:#d8e6ff; border:1px solid #1f2e45; border-radius:12px; padding:8px 12px; }
QPushButton:hover { background:#122136; }
QPushButton:pressed { background:#0e1a2a; }
QMenuBar, QMenu { background:#0b1118; color:#d8e6ff; }
QMenu::item:selected { background:#142238; }
QGroupBox { background:transparent; border:1px solid #1a2a40; border-radius:10px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#0b1118; }
QTabWidget::pane { background:#0b1118; border:1px solid #1a2a40; border-radius:10px; }
QTabBar::tab { background:#0f1926; color:#d8e6ff; padding:6px 12px; border:1px solid #1a2a40; border-bottom:none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#142238; }
QTabBar::tab:hover { background:#122136; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#0b141f; color:#d8e6ff; border:1px solid #1a2a40; border-radius:10px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#0b141f; color:#d8e6ff; border:1px solid #1a2a40; border-radius:10px; }
QSlider::groove:horizontal { height:6px; background:#12213a; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#1f3963; border:1px solid #12213a; margin:-5px 0; border-radius:7px; }
QToolTip { background:#0f1b2a; color:#d8e6ff; border:1px solid #1f2e45; }
QMenuBar::item { background:#0b1118; color:#d8e6ff; padding:4px 8px; }
QMenuBar::item:selected { background:#142238; }
QMenuBar::item:pressed { background:#142238; }

"""


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None, expanded=False):
        super().__init__(parent)
        try:
            self.setObjectName('sec_' + _slug_title(title))
        except Exception:
            pass
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
        lay.setSpacing(6)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

        def _on_toggled(on):
            self._expanded = on
            self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            self.content.setVisible(True)  # visible during animation
            start = self.content.maximumHeight()
            end = self.content.sizeHint().height() if on else 0
            self.anim.stop(); self.anim.setStartValue(start); self.anim.setEndValue(end); self.anim.start()
        self.toggle.toggled.connect(_on_toggled)

    def _on_anim_finished(self):
        if self._expanded:
            self.content.setMaximumHeight(16777215)  # give natural height so scrollbars can appear
        else:
            self.content.setMaximumHeight(0)
            self.content.setVisible(False)

    def setContentLayout(self, layout):
        QWidget().setLayout(self.content.layout())
        self.content.setLayout(layout)
        if self._expanded:
            self.content.setMaximumHeight(16777215)
        else:
            self.content.setMaximumHeight(0)


# --- Add safe helpers to expose expanded state for saving/restoring
def _cs_isChecked(self):
    try:
        return bool(self._expanded)
    except Exception:
        return False

def _cs_setChecked(self, on: bool):
    try:
        on = bool(on)
        # Drive via the toggle so animation/state stays consistent
        if getattr(self, '_expanded', None) is None:
            self._expanded = on
        if getattr(self, 'toggle', None) is not None:
            if self._expanded != on:
                self.toggle.setChecked(on)
            else:
                # ensure visuals are correct
                self.content.setVisible(on)
                self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        else:
            # Fallback: directly set
            self._expanded = on
    except Exception:
        pass

try:
    CollapsibleSection.isChecked = _cs_isChecked
    CollapsibleSection.setChecked = _cs_setChecked
except Exception:
    pass


class InstantToolsPane(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(12)
        root.setSpacing(10)
        # --- Global remember toggle for Tools tab ---
        try:
            self._qs = QSettings("FrameVision","FrameVision")
        except Exception:
            from PySide6.QtCore import QSettings as _QS
            self._qs = _QS("FrameVision","FrameVision")
        try:
            remember_enabled = bool(self._qs.value("ToolsPane/remember_enabled", True, type=bool))
        except Exception:
            remember_enabled = True
        topbar = QHBoxLayout(); topbar.setContentsMargins(0,0,0,0); topbar.addStretch(1)
        try:
            self.cb_remember = QCheckBox("Remember settings"); self.cb_remember.setChecked(remember_enabled)
            self.btn_remember_menu = QPushButton("Remember…")
            self.remember_menu = QMenu(self.btn_remember_menu)
            self.btn_remember_menu.setMenu(self.remember_menu)
            topbar.addWidget(self.cb_remember); topbar.addWidget(self.btn_remember_menu)
            root.addLayout(topbar)
            self.cb_remember.toggled.connect(lambda v: self._qs.setValue("ToolsPane/remember_enabled", bool(v)))
        except Exception:
            pass
    

        # Sections
        sec_speed = CollapsibleSection("Speed", expanded=False)
        sec_resize = CollapsibleSection("Resize", expanded=False)
        sec_gif = CollapsibleSection("Export GIF", expanded=False)
        sec_extract = CollapsibleSection("Extract frames", expanded=False)
        sec_trim = CollapsibleSection("Trim", expanded=False)
        sec_crop = CollapsibleSection("Crop", expanded=False)

        sec_quality = CollapsibleSection("Quality / Size Video", expanded=False)

        # --- Quality/Size controls ---
        self.q_mode = QComboBox(); self.q_mode.addItems(["CRF (quality)", "Bitrate (kbps)"])
        self.q_preset = QComboBox(); self.q_preset.addItems(["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"])
        self.q_codec = QComboBox(); self.q_codec.addItems([
            "H.264 (x264)", "H.265 (x265)", "AV1 (SVT-AV1)",
            "H.264 (NVENC)", "HEVC (NVENC)", "AV1 (NVENC)",
        ])
        self.q_crf = QSpinBox(); self.q_crf.setRange(0, 51); self.q_crf.setValue(23)
        self.q_bitrate = QSpinBox(); self.q_bitrate.setRange(100, 50000); self.q_bitrate.setSingleStep(100); self.q_bitrate.setValue(3500)
        self.q_audio = QComboBox(); self.q_audio.addItems(["copy", "aac 128k", "aac 192k", "opus 96k", "opus 128k"])
        self.q_format = QComboBox(); self.q_format.addItems(["mp4","mkv","mov"])
        self.btn_quality = QPushButton("Change quality"); self.btn_quality_batch = QPushButton("Batch…")
        self.btn_quality.setToolTip("Re-encode/transcode with selected codec and quality mode.")

        lay_q = QFormLayout()
        lay_q.addRow("Quality mode", self.q_mode)
        lay_q.addRow("Video codec", self.q_codec)
                # sliders paired with spinboxes
        self.q_crf_slider = QSlider(Qt.Horizontal); self.q_crf_slider.setRange(0,51); self.q_crf_slider.setValue(self.q_crf.value())
        self.q_bitrate_slider = QSlider(Qt.Horizontal); self.q_bitrate_slider.setRange(100,50000); self.q_bitrate_slider.setSingleStep(100); self.q_bitrate_slider.setValue(self.q_bitrate.value())
        # link slider<->spin
        self.q_crf_slider.valueChanged.connect(self.q_crf.setValue); self.q_crf.valueChanged.connect(self.q_crf_slider.setValue)
        self.q_bitrate_slider.valueChanged.connect(self.q_bitrate.setValue); self.q_bitrate.valueChanged.connect(self.q_bitrate_slider.setValue)
        _w_crf = QWidget(); _h1 = QHBoxLayout(_w_crf); _h1.setContentsMargins(0,0,0,0); _h1.addWidget(self.q_crf_slider); _h1.addWidget(self.q_crf)
        _w_br = QWidget(); _h2 = QHBoxLayout(_w_br); _h2.setContentsMargins(0,0,0,0); _h2.addWidget(self.q_bitrate_slider); _h2.addWidget(self.q_bitrate)
        lay_q.addRow("CRF (0-51)", _w_crf)
        lay_q.addRow("Bitrate (kbps)", _w_br)
        lay_q.addRow("Encoder preset", self.q_preset)
        lay_q.addRow("Audio", self.q_audio)
        lay_q.addRow("Format", self.q_format)
        row_q = QHBoxLayout(); row_q.addWidget(self.btn_quality); row_q.addWidget(self.btn_quality_batch); lay_q.addRow(row_q)
        sec_quality.setContentLayout(lay_q)

        # ---- Image Quality / Convert ----
        sec_img = CollapsibleSection("Image Quality / Convert", expanded=False)

        self.img_format = QComboBox(); self.img_format.addItems(["jpg","webp","png"])
        self.img_quality = QSpinBox(); self.img_quality.setRange(1,100); self.img_quality.setValue(85)
        self.img_w = QSpinBox(); self.img_w.setRange(0,16384); self.img_w.setValue(0)
        self.img_h = QSpinBox(); self.img_h.setRange(0,16384); self.img_h.setValue(0)
        self.img_keep_meta = QCheckBox("Keep metadata (EXIF)"); self.img_keep_meta.setChecked(False)

        # Sliders paired with quality and width/height
        self.img_q_slider = QSlider(Qt.Horizontal); self.img_q_slider.setRange(1,100); self.img_q_slider.setValue(self.img_quality.value())
        self.img_q_slider.valueChanged.connect(self.img_quality.setValue); self.img_quality.valueChanged.connect(self.img_q_slider.setValue)

        self.btn_img_convert = QPushButton("Convert image"); self.btn_img_batch = QPushButton("Batch…")

        
        # Mode: Quality or Target size
        self.img_mode = QComboBox(); self.img_mode.addItems(["Quality (%)","Target size"])
        self.img_target = QSpinBox(); self.img_target.setRange(10, 500000); self.img_target.setValue(400)  # KB
        self.img_target_unit = QComboBox(); self.img_target_unit.addItems(["KB","MB"])
        lay_i = QFormLayout()
        lay_i.addRow("Mode", self.img_mode)
        _it = QWidget(); _hti = QHBoxLayout(_it); _hti.setContentsMargins(0,0,0,0); _hti.addWidget(self.img_target); _hti.addWidget(self.img_target_unit); lay_i.addRow("Target size", _it)
        _iq = QWidget(); _hqi = QHBoxLayout(_iq); _hqi.setContentsMargins(0,0,0,0); _hqi.addWidget(self.img_q_slider); _hqi.addWidget(self.img_quality)
        lay_i.addRow("Output format", self.img_format)
        lay_i.addRow("Quality (1-100)", _iq)
        lay_i.addRow("Max width (0 = keep)", self.img_w)
        lay_i.addRow("Max height (0 = keep)", self.img_h)
        lay_i.addRow(self.img_keep_meta)
        row_i = QHBoxLayout(); row_i.addWidget(self.btn_img_convert); row_i.addWidget(self.btn_img_batch); lay_i.addRow(row_i)
        sec_img.setContentLayout(lay_i)
        # ---- Meme / Caption ----
        sec_meme = CollapsibleSection("Thumbnail / Meme Creator", expanded=True)
        _meme_wrap = QWidget(); _memel = QVBoxLayout(_meme_wrap); _memel.setContentsMargins(0,0,0,0)
        try:
            _meme = MemeToolPane(self.main, self)
        except Exception:
            _meme = MemeToolPane(None, self)
        _memel.addWidget(_meme)
        sec_meme.setContentLayout(_memel)
        
        
        # ---- Prompt Generator ----
        sec_prompt = CollapsibleSection("Prompt Generator", expanded=False)
        try:
            install_prompt_tool(self, sec_prompt)
        except Exception:
            try:
                _p = QWidget(); _pl = QVBoxLayout(_p); _pl.setContentsMargins(0,0,0,0); _pl.addWidget(QLabel("Prompt tool failed to load."))
                sec_prompt.setContentLayout(_pl)
            except Exception:
                pass
# ---- Multi Rename (moved to helpers/renam.py) ----
        sec_rename = CollapsibleSection("Multi Rename", expanded=False)
        _rn_wrap = QWidget(); _rn_layout = QVBoxLayout(_rn_wrap); _rn_layout.setContentsMargins(0,0,0,0)
        try:
            _rn_widget = RenamPane(self.main, self)
        except Exception:
            _rn_widget = RenamPane(None, self)
        _rn_layout.addWidget(_rn_widget)
        sec_rename.setContentLayout(_rn_layout)


        def _img_mode_update():
            tgt = (self.img_mode.currentIndex()==1)
            qual_ok = (self.img_format.currentText().lower() != "png")
            self.img_q_slider.setEnabled(not tgt)
            self.img_quality.setEnabled(not tgt)
            self.img_target.setEnabled(tgt and qual_ok)
            self.img_target_unit.setEnabled(tgt and qual_ok)
        self.img_mode.currentIndexChanged.connect(lambda _=0: _img_mode_update())
        self.img_format.currentIndexChanged.connect(lambda _=0: _img_mode_update())
        _img_mode_update()

        self._probe_codecs_and_disable_unavailable()

        # Toggle CRF/Bitrate widgets visibility based on mode
        def _upd_q_mode(idx):
            use_crf = idx == 0
            self.q_crf.setEnabled(use_crf); self.q_bitrate.setEnabled(not use_crf)
            try:
                self.q_crf_slider.setEnabled(use_crf); self.q_bitrate_slider.setEnabled(not use_crf)
            except Exception:
                pass
        _upd_q_mode(self.q_mode.currentIndex())
        self.q_mode.currentIndexChanged.connect(_upd_q_mode)

        # Speed
        self.speed = QSlider(Qt.Horizontal); self.speed.setRange(25, 250); self.speed.setValue(150)
        self.lbl_speed = QLabel("1.50x"); self.speed.valueChanged.connect(lambda v: self.lbl_speed.setText(f"{v/100:.2f}x"))
        from PySide6.QtWidgets import QDoubleSpinBox
        self.spin_speed = QDoubleSpinBox(); self.spin_speed.setRange(0.25, 2.50); self.spin_speed.setDecimals(2); self.spin_speed.setSingleStep(0.01); self.spin_speed.setValue(1.50)
        # sync both ways
        self.speed.valueChanged.connect(lambda v: self.spin_speed.setValue(round(v/100.0,2)))
        self.spin_speed.valueChanged.connect(lambda val: self.speed.setValue(int(round(val*100))))
        self.btn_speed = QPushButton("Change Speed"); self.btn_speed.setToolTip("Change playback speed. If Sound sync is on, audio pitch is preserved."); self.btn_speed.setToolTip("Change playback speed. 1.00x keeps pitch if Sound sync is on."); self.btn_speed_batch = QPushButton("Batch…"); self.btn_speed_batch.setToolTip("Batch with current Speed settings."); self.btn_speed_batch.setToolTip("Select multiple videos or a folder; one job per file using current Speed settings.")
        lay_speed = QFormLayout();
        row_speed = QHBoxLayout(); row_speed.addWidget(self.speed); row_speed.addWidget(self.spin_speed); lay_speed.addRow("Speed factor", row_speed); lay_speed.addRow("", self.lbl_speed); row_b = QHBoxLayout(); row_b.addWidget(self.btn_speed); row_b.addWidget(self.btn_speed_batch); lay_speed.addRow(row_b)

        # Audio options
        try:
            self.cb_speed_sync = QCheckBox("Sound sync (adjust audio tempo)"); self.cb_speed_sync.setToolTip("Adjust audio tempo to match new speed."); self.cb_speed_sync.setToolTip("When enabled, audio tempo is adjusted to match the new speed.")
            self.cb_speed_sync.setChecked(True)
        except Exception:
            pass
        try:
            self.cb_speed_mute = QCheckBox("Mute audio"); self.cb_speed_mute.setToolTip("Output video without audio."); self.cb_speed_mute.setToolTip("Mute the audio track in the output.")
            self.cb_speed_mute.setChecked(False)
        except Exception:
            pass
        try:
            lay_speed.addRow(self.cb_speed_sync)
            lay_speed.addRow(self.cb_speed_mute)
        except Exception:
            pass
        
        try:
            lay_speed.addRow("Factor (type)", self.spin_speed)
        except Exception:
            pass
        sec_speed.setContentLayout(lay_speed)
        # Presets buttons
        row = QHBoxLayout(); btn_ss = QPushButton("Save preset"); btn_ls = QPushButton("Load preset"); row.addWidget(btn_ss); row.addWidget(btn_ls)
        lay_speed.addRow(row)
        btn_ss.clicked.connect(lambda: self._save_preset_speed())
        btn_ls.clicked.connect(lambda: self._load_preset_speed())

        
        # Resize (moved to helpers/resize.py)

        try:

            from helpers.resize import install_resize_tool

            install_resize_tool(self, sec_resize)

        except Exception as e:

            try:

                QMessageBox.critical(self, 'Resize init failed', str(e))

            except Exception:

                pass
# GIF
        self.gif_fps = QSlider(Qt.Horizontal); self.gif_fps.setRange(5,30); self.gif_fps.setValue(12)
        self.lbl_gif_fps = QLabel("12"); self.gif_fps.valueChanged.connect(lambda v: self.lbl_gif_fps.setText(str(v)))
        self.spin_gif_fps = QSpinBox(); self.spin_gif_fps.setRange(5,30); self.spin_gif_fps.setValue(12)
        self.gif_fps.valueChanged.connect(lambda v: self.spin_gif_fps.setValue(int(v)))
        self.spin_gif_fps.valueChanged.connect(lambda val: self.gif_fps.setValue(int(val)))
        self.gif_same = QCheckBox("Same as video"); self.gif_same.setToolTip("Use the source video FPS for the GIF."); self.gif_same.toggled.connect(lambda on: (self.gif_fps.setEnabled(not on), self.spin_gif_fps.setEnabled(not on)))
        self.btn_gif = QPushButton("Export GIF"); self.btn_gif.setToolTip("Queue a GIF export using the settings above."); self.btn_gif_batch = QPushButton("Batch…"); self.btn_gif_batch.setToolTip("Batch with current GIF settings."); self.btn_gif_batch.setToolTip("Batch GIF export with current FPS setting.")
        self.btn_gif.setToolTip("Exports a GIF from the video using the chosen FPS. Lower FPS = smaller file.")
        lay_gif = QFormLayout();
        row_gfps = QHBoxLayout(); row_gfps.addWidget(self.gif_fps); row_gfps.addWidget(self.spin_gif_fps); lay_gif.addRow("GIF fps", row_gfps); lay_gif.addRow("", self.gif_same); row_gb = QHBoxLayout(); row_gb.addWidget(self.btn_gif); row_gb.addWidget(self.btn_gif_batch); lay_gif.addRow(row_gb)
        
        try:
            lay_gif.addRow("GIF fps (type)", self.spin_gif_fps)
        except Exception:
            pass
                # --- Advanced GIF options ---
        gif_backend.install_ui(self, lay_gif, sec_gif)
        sec_gif.setContentLayout(lay_gif)
        # Audio (moved to helpers/audiotool.py)
        sec_audio = CollapsibleSection("Audio", expanded=False)
        try:
            install_audio_tool(self, sec_audio)
        except Exception:
            pass


        # moved below to reorder; see tuple later
        # root.addWidget(sec_audio)
        row = QHBoxLayout(); btn_sg = QPushButton("Save preset"); btn_lg = QPushButton("Load preset"); row.addWidget(btn_sg); row.addWidget(btn_lg)
        lay_gif.addRow(row)
        btn_sg.clicked.connect(lambda: self._save_preset_gif())
        btn_lg.clicked.connect(lambda: self._load_preset_gif())

        # Extract
        self.btn_last = QPushButton("Extract Last Frame"); self.btn_all = QPushButton("Extract All Frames"); self.btn_all.setToolTip("Export every frame to images. Large output!")
        lay_ext = QVBoxLayout(); lay_ext.addWidget(self.btn_last); lay_ext.addWidget(self.btn_all)
        sec_extract.setContentLayout(lay_ext)
        row = QHBoxLayout(); btn_se = QPushButton("Save preset"); btn_le = QPushButton("Load preset"); row.addWidget(btn_se); row.addWidget(btn_le)
        lay_ext.addLayout(row)
        btn_se.clicked.connect(lambda: self._save_preset_extract())
        btn_le.clicked.connect(lambda: self._load_preset_extract())

        # Trim (moved to helpers/trim_tool.py)
        install_trim_tool(self, sec_trim)



        self.crop_y = QSlider(Qt.Horizontal); self.crop_y.setRange(0,8192); self.lbl_crop_y = QLabel("0")
        self.crop_h = QSlider(Qt.Horizontal); self.crop_h.setRange(16,8192); self.lbl_crop_h = QLabel("16")
        self.spin_crop_y = QSpinBox(); self.spin_crop_y.setRange(0,8192); self.spin_crop_y.setValue(0)
        self.spin_crop_h = QSpinBox(); self.spin_crop_h.setRange(16,8192); self.spin_crop_h.setValue(16)
        self.crop_y.valueChanged.connect(lambda v: (self.lbl_crop_y.setText(str(v)), self.spin_crop_y.setValue(int(v))))
        self.crop_h.valueChanged.connect(lambda v: (self.lbl_crop_h.setText(str(v)), self.spin_crop_h.setValue(int(v))))
        self.spin_crop_y.valueChanged.connect(lambda val: self.crop_y.setValue(int(val)))
        self.spin_crop_h.valueChanged.connect(lambda val: self.crop_h.setValue(int(val)))
        # Tooltips
        self.crop_y.setToolTip("Top offset in pixels for vertical crop.")
        self.spin_crop_y.setToolTip("Type exact top offset (px).")
        self.crop_h.setToolTip("Crop height in pixels (vertical crop).")
        self.spin_crop_h.setToolTip("Type exact crop height (px).")

        self.crop_y.valueChanged.connect(lambda v: self.lbl_crop_y.setText(str(v)))
        self.crop_h.valueChanged.connect(lambda v: self.lbl_crop_h.setText(str(v)))
        self.btn_crop = QPushButton("Crop"); self.btn_crop_batch = QPushButton("Batch…"); self.btn_crop_batch.setToolTip("Batch with current Crop settings."); self.btn_crop_batch.setToolTip("Batch Crop using current Y/H values.")
        lay_crop = QFormLayout(); lay_crop.addRow("Crop Y", self.crop_y); lay_crop.addRow("", self.lbl_crop_y); lay_crop.addRow("Crop H", self.crop_h); lay_crop.addRow("", self.lbl_crop_h); row_cb = QHBoxLayout(); row_cb.addWidget(self.btn_crop); row_cb.addWidget(self.btn_crop_batch); lay_crop.addRow(row_cb)
        
        try:
            lay_crop.addRow("Crop Y (type)", self.spin_crop_y)
            lay_crop.addRow("Crop H (type)", self.spin_crop_h)
        except Exception:
            pass
        sec_crop.setContentLayout(lay_crop)
        row = QHBoxLayout(); btn_sc = QPushButton("Save preset"); btn_lc = QPushButton("Load preset"); row.addWidget(btn_sc); row.addWidget(btn_lc)
        lay_crop.addRow(row)
        btn_sc.clicked.connect(lambda: self._save_preset_crop())
        btn_lc.clicked.connect(lambda: self._load_preset_crop())

        for sec in (sec_meme, sec_prompt, sec_audio, sec_speed, sec_resize, sec_gif, sec_extract, sec_trim, sec_crop, sec_quality, sec_img, sec_rename):
            root.addWidget(sec)
        root.addStretch(1)
        # --- Remember settings (per-tool + global) ---
        def _sec_name_map():
            return {
                "Audio": sec_audio,
                "Speed": sec_speed,
                "Resize": sec_resize,
                "Export GIF": sec_gif,
                "Extract frames": sec_extract,
                "Trim": sec_trim,
                "Crop": sec_crop,
                "Quality / Size Video": sec_quality,
                "Image Quality / Convert": sec_img,
                "Thumbnail / Meme Creator": sec_meme,
                "Prompt Generator": sec_prompt,
                "Multi Rename": sec_rename
            }
        self._sections_map = _sec_name_map()
        # Build default whitelist (all except Trim and Audio)
        default_whitelist = [k for k in self._sections_map.keys() if k not in ("Trim","Audio")]
        try:
            import json as _json
            wl_txt = self._qs.value("ToolsPane/remember_whitelist_json", "", type=str) or ""
            self._remember_whitelist = set(_json.loads(wl_txt)) if wl_txt else set(default_whitelist)
        except Exception:
            self._remember_whitelist = set(default_whitelist)
        # populate menu
        try:
            if self.remember_menu:
                self.remember_menu.clear()
                for name in self._sections_map.keys():
                    if name in ("Trim","Audio"): 
                        continue
                    act = self.remember_menu.addAction(name)
                    act.setCheckable(True); act.setChecked(name in self._remember_whitelist)
                    act.toggled.connect(lambda on, nm=name: self._toggle_remember_tool(nm, on))
        except Exception:
            pass

        def _widget_snapshot(w):
            from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                                           QSpinBox, QDoubleSpinBox, QComboBox,
                                           QCheckBox, QRadioButton, QSlider,
                                           QDateEdit, QTimeEdit, QDateTimeEdit)
            from PySide6.QtCore import Qt
            classes = (QLineEdit, QPlainTextEdit, QTextEdit,
                       QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
                       QCheckBox, QRadioButton, QDateEdit, QTimeEdit, QDateTimeEdit)
            snap = []
            try:
                for cls in classes:
                    children = w.findChildren(cls)
                    for idx, child in enumerate(children):
                        name = child.objectName() or ""
                        try:
                            if isinstance(child, QLineEdit):
                                val = child.text()
                            elif isinstance(child, (QPlainTextEdit, QTextEdit)):
                                val = child.toPlainText()
                            elif isinstance(child, (QSpinBox, QSlider)):
                                val = int(child.value())
                            elif isinstance(child, QDoubleSpinBox):
                                val = float(child.value())
                            elif isinstance(child, QComboBox):
                                val = int(child.currentIndex())
                            elif isinstance(child, (QCheckBox, QRadioButton)):
                                val = bool(child.isChecked())
                            elif isinstance(child, QDateEdit):
                                val = child.date().toString(Qt.ISODate)
                            elif isinstance(child, QTimeEdit):
                                val = child.time().toString(Qt.ISODate)
                            elif isinstance(child, QDateTimeEdit):
                                val = child.dateTime().toString(Qt.ISODate)
                            else:
                                continue
                            snap.append((cls.__name__, idx, name, val))
                        except Exception:
                            continue
            except Exception:
                pass
            return snap

        def _widget_restore(w, snap):
            try:
                from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                                               QSpinBox, QDoubleSpinBox, QComboBox,
                                               QCheckBox, QRadioButton, QSlider,
                                               QDateEdit, QTimeEdit, QDateTimeEdit)
                from PySide6.QtCore import Qt, QDate, QTime, QDateTime
            except Exception:
                return
            cls_map = {"QLineEdit":QLineEdit, "QPlainTextEdit":QPlainTextEdit, "QTextEdit":QTextEdit,
                       "QSpinBox":QSpinBox, "QDoubleSpinBox":QDoubleSpinBox, "QSlider":QSlider,
                       "QComboBox":QComboBox, "QCheckBox":QCheckBox, "QRadioButton":QRadioButton,
                       "QDateEdit":QDateEdit, "QTimeEdit":QTimeEdit, "QDateTimeEdit":QDateTimeEdit}
            for item in (snap or []):
                try:
                    cls_name, idx, name, val = item
                except Exception:
                    continue
                cls = cls_map.get(str(cls_name))
                if not cls:
                    continue
                target = None
                if name:
                    target = w.findChild(cls, name)
                if target is None:
                    children = w.findChildren(cls)
                    try:
                        idx_i = int(idx)
                    except Exception:
                        idx_i = -1
                    if 0 <= idx_i < len(children):
                        target = children[idx_i]
                if target is None:
                    continue
                try:
                    if isinstance(target, QLineEdit):
                        target.setText(str(val))
                    elif isinstance(target, (QPlainTextEdit, QTextEdit)):
                        target.setPlainText(str(val))
                    elif isinstance(target, (QSpinBox, QSlider)):
                        target.setValue(int(val))
                    elif isinstance(target, QDoubleSpinBox):
                        target.setValue(float(val))
                    elif isinstance(target, QComboBox):
                        target.setCurrentIndex(int(val))
                    elif isinstance(target, (QCheckBox, QRadioButton)):
                        target.setChecked(bool(val))
                    elif isinstance(target, QDateEdit):
                        d = QDate.fromString(str(val), Qt.ISODate)
                        if d.isValid(): target.setDate(d)
                    elif isinstance(target, QTimeEdit):
                        t = QTime.fromString(str(val), Qt.ISODate)
                        if t.isValid(): target.setTime(t)
                    elif isinstance(target, QDateTimeEdit):
                        dt = QDateTime.fromString(str(val), Qt.ISODate)
                        if dt.isValid(): target.setDateTime(dt)
                except Exception:
                    continue

        def _save_all_tools():
            try:
                import json as _json
                data = {"sections": {}}
                for name, sec in self._sections_map.items():
                    try:
                        expanded = bool(getattr(sec, "isChecked", lambda: True)())
                    except Exception:
                        expanded = True
                    content = getattr(sec, "content", sec)
                    data["sections"][name] = {"expanded": bool(expanded), "snap": _widget_snapshot(content)}
                self._qs.setValue("ToolsPane/saved_json", _json.dumps(data))
            except Exception:
                pass

        def _apply_saved():
            try:
                import json as _json
                txt = self._qs.value("ToolsPane/saved_json", "", type=str) or ""
                if not txt:
                    return
                data = _json.loads(txt)
                for name, sec in self._sections_map.items():
                    entry = (data.get("sections") or {}).get(name)
                    if not entry:
                        continue
                    snap = entry.get("snap")
                    if snap:
                        content = getattr(sec, "content", sec)
                        _widget_restore(content, snap)
                    if "expanded" in entry:
                        try:
                            sec.setChecked(bool(entry["expanded"]))
                        except Exception:
                            pass
            except Exception:
                pass

        def _autosave():
            """Debounced autosave.
            Creates a single QTimer once and restarts it on every call.
            When the timer fires, it performs one _save_all_tools() and stops.
            This avoids spawning/connecting new timers repeatedly.
            """
            try:
                if getattr(self, '_autosave_timer', None) is None:
                    self._autosave_timer = QTimer(self)
                    try:
                        # Use a coarse timer on Windows to reduce WndProc load
                        self._autosave_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
                    except Exception:
                        pass
                    self._autosave_timer.setSingleShot(True)
                    self._autosave_timer.setInterval(1500)
                    self._autosave_timer.timeout.connect(lambda: _save_all_tools())
                # debounce restart
                self._autosave_timer.start()
            except Exception:
                # Fallback: save immediately if timer can't start
                try:
                    _save_all_tools()
                except Exception:
                    pass

        def _wire_watchers():
            try:
                from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton,
                    QSlider, QDateEdit, QTimeEdit, QDateTimeEdit)
            except Exception:
                return
            def _wire(root):
                try:
                    for w in root.findChildren(QLineEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QPlainTextEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QTextEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QSpinBox): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDoubleSpinBox): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QComboBox): w.currentIndexChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QCheckBox): w.toggled.connect(lambda *_: _autosave())
                    for w in root.findChildren(QRadioButton): w.toggled.connect(lambda *_: _autosave())
                    for w in root.findChildren(QSlider): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDateEdit): w.dateChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QTimeEdit): w.timeChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDateTimeEdit): w.dateTimeChanged.connect(lambda *_: _autosave())
                except Exception:
                    pass
            try:
                for _name, _sec in self._sections_map.items():
                    _root = getattr(_sec, 'content', _sec)
                    _wire(_root)
            except Exception:
                pass
        def _toggle_remember_tool(name, on):
            try:
                if on: self._remember_whitelist.add(name)
                else: self._remember_whitelist.discard(name)
                import json as _json
                self._qs.setValue("ToolsPane/remember_whitelist_json", _json.dumps(sorted(self._remember_whitelist)))
                _save_all_tools()
            except Exception:
                pass
        self._toggle_remember_tool = _toggle_remember_tool

        # Apply saved settings on init
        try:
            _apply_saved()
        except Exception:
            pass

        try:
            _wire_watchers()
            _save_all_tools()
        except Exception:
            pass

    

        
        self.btn_quality.clicked.connect(self.run_quality)
        self.btn_quality_batch.clicked.connect(self.run_quality_batch_popup)
        # removed: batch_folder button
        self.btn_img_convert.clicked.connect(self.run_img_convert)
        self.btn_img_batch.clicked.connect(self.run_img_batch_popup)
        # removed: batch_folder button
# Wire
        self.btn_speed.clicked.connect(self.run_speed)
        self.btn_speed_batch.clicked.connect(self.run_speed_batch)
        self.btn_gif.clicked.connect(self.run_gif)
        self.btn_gif_batch.clicked.connect(self.run_gif_batch)
        self.btn_last.clicked.connect(self.run_last)
        self.btn_all.clicked.connect(self.run_all)

        self.btn_trim.clicked.connect(self.run_trim)
        # Let the Trim tool own the batch popup; wire here only if provided.
        try:
            if hasattr(self, 'run_trim_batch'):
                self.btn_trim_batch.clicked.connect(self.run_trim_batch)
        except Exception:
            pass
        self.btn_crop.clicked.connect(self.run_crop)
        self.btn_crop_batch.clicked.connect(self.run_crop_batch)

    # Helpers and actions stay the same as your previous version
    
    def _pick_batch_files(self, for_videos=True):
        try:
            from PySide6.QtWidgets import QFileDialog
            import os
            files, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Media files (*.*)")
            paths = [p for p in files or [] if os.path.isfile(p)]
            if not paths:
                folder = QFileDialog.getExistingDirectory(self, "Or select folder")
                if folder:
                    exts_vid = {".mp4",".mov",".mkv",".webm",".avi"}
                    exts_img = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
                    for name in os.listdir(folder):
                        p=os.path.join(folder,name); ext=os.path.splitext(name)[1].lower()
                        if os.path.isfile(p) and ((for_videos and ext in exts_vid) or ((not for_videos) and ext in exts_img)):
                            paths.append(p)
            return paths
        except Exception:
            return []

    def _ensure_input(self, allow_images: bool=False):
        if not self.main.current_path:
            QMessageBox.warning(self, "No file", "Open a video first.");
            return None
        return self.main.current_path

    def _run(self, cmd_list, outfile):
        # Enqueue instead of executing immediately
        inp = self._ensure_input()
        if not inp:
            return
        from pathlib import Path as _P
        out_path = outfile if isinstance(outfile, _P) else _P(str(outfile))
        try:
            try:
                from helpers.queue_adapter import enqueue_tool_job as enq
            except Exception:
                from queue_adapter import enqueue_tool_job as enq
            jid = enq("tools_ffmpeg", str(inp), str(out_path.parent), {"ffmpeg_cmd": cmd_list, "outfile": str(out_path)}, priority=600)
            try:
                QMessageBox.information(self, "Queued", f"Job queued to jobs/pending.\n\n{jid}")
            except Exception:
                pass
            # Switch to Queue tab
            try:
                win = self.window()
                tabs = win.findChild(QTabWidget)
                if tabs:
                    for i in range(tabs.count()):
                        if tabs.tabText(i).strip().lower() == "queue":
                            tabs.setCurrentIndex(i); break
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.warning(self, "Queue error", str(e))
            except Exception:
                pass
        return



    def run_speed(self):
        inp = self._ensure_input()
        if not inp:
            return
        factor = self.speed.value() / 100.0
        try:
            out_dir = OUT_VIDEOS
        except Exception:
            out_dir = Path('.')
        out = out_dir / f"{inp.stem}_spd_{factor:.2f}x.mp4"
        setpts = 1.0 / float(factor if factor != 0 else 1.0)
        mute = False
        sync = True
        try:
            mute = bool(self.cb_speed_mute.isChecked())
        except Exception:
            pass
        try:
            sync = bool(self.cb_speed_sync.isChecked())
        except Exception:
            pass
        cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", f"setpts={setpts:.6f}*PTS"]
        if mute:
            cmd += ["-an"]
        else:
            if sync:
                atempos = []
                f = float(factor)
                while f > 2.0:
                    atempos.append("2.0"); f /= 2.0
                while f < 0.5:
                    atempos.append("0.5"); f *= 2.0
                atempos.append(f"{f:.3f}")
                cmd += ["-filter:a", "atempo=" + ",".join(atempos)]
            else:
                cmd += ["-c:a", "copy", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(out)]
        self._run(cmd, out)

    
    def run_gif(self):
        inp = self._ensure_input()
        if not inp:
            return
        fmt = self.gif_fmt.currentText().strip().lower() if hasattr(self, "gif_fmt") else "gif"
        opts = gif_backend.options_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self, "gif_fps") else 0, batch=False)
        out = OUT_VIDEOS / gif_backend.output_name_for(inp.stem, fmt, batch=False)
        # Progress + ETA like interp.py
        try:
            gif_backend.encode_with_progress(self, inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
        except Exception:
            gif_backend.encode(inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
        # Remember output to enable "Play last" button in the GIF UI
        try:
            self._gif_last_out = out
        except Exception:
            pass
    def run_last(self):
        inp = self._ensure_input();
        if not inp: return
        out=OUT_VIDEOS / f"{inp.stem}_lastframe.png"
        cmd=[ffmpeg_path(),"-y","-sseof","-1","-i",str(inp),"-update","1","-frames:v","1",str(out)]
        self._run(cmd,out)

    def run_all(self):
        inp = self._ensure_input();
        if not inp: return
        outdir=OUT_VIDEOS / f"{inp.stem}_frames"; outdir.mkdir(parents=True, exist_ok=True); out=outdir / "frame_%06d.png"
        cmd=[ffmpeg_path(),"-y","-i",str(inp),str(out)]
        self._run(cmd,outdir)

    def run_trim(self):
        inp = self._ensure_input();
        if not inp: return
        start=self.trim_start.text().strip(); end=self.trim_end.text().strip()
        out=OUT_TRIMS / f"{inp.stem}_trim_{start.replace(':','-').replace('.','_')}_{(end or 'end').replace(':','-').replace('.','_')}.mp4"
        if self.trim_mode.currentIndex()==0:
            cmd=[ffmpeg_path(),"-y"]+ (["-ss",start] if start else []) + (["-to",end] if end else []) + ["-i",str(inp),"-c","copy","-movflags","+faststart",str(out)]
        else:
            cmd=[ffmpeg_path(),"-y"] + (["-ss",start] if start else []) + ["-i",str(inp)] + (["-to",end] if end else []) + ["-c:v","libx264","-preset","veryfast","-c:a","aac","-movflags","+faststart",str(out)]
        self._run(cmd,out)

    def run_crop(self):
        inp = self._ensure_input();
        if not inp: return
        try:
            inf = probe_media(inp); ih = int(inf.get("height", 0))
        except Exception:
            ih = 0
        y = int(self.crop_y.value()); h = int(self.crop_h.value())
        if ih:
            y = max(0, min(y, max(0, ih-1)))
            h = max(16, min(h, max(16, ih - y)))
        out = OUT_VIDEOS / f"{inp.stem}_crop_y{y}_h{h}.mp4"
        filter_str = f"crop=iw:{h}:0:{y}"
        cmd=[ffmpeg_path(),"-y","-i",str(inp),"-vf",filter_str,"-c:v","libx264","-preset","veryfast","-movflags","+faststart", str(out)]
        self._run(cmd,out)


    # ---- Presets helpers ----
    def _preset_dir(self):
        try:
            base = Path(config.get("last_preset_dir", str(ROOT / "presets" / "Tools")))
        except Exception:
            base = ROOT / "presets" / "Tools"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _choose_save_path(self, suggested_name):
        d = self._preset_dir()
        path = d / suggested_name
        fn, _ = QFileDialog.getSaveFileName(self, "Save preset", str(path), "FrameVision Preset (*.json)")
        if not fn:
            return None
        p = Path(fn)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        # overwrite confirmation
        if p.exists():
            res = QMessageBox.question(self, "Overwrite?", f"{p.name} already exists. Overwrite?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if res != QMessageBox.Yes:
                return None
        # remember folder
        try:
            config["last_preset_dir"] = str(p.parent)
            save_config()
        except Exception:
            pass
        return p

    def _choose_open_path(self):
        d = self._preset_dir()
        fn, _ = QFileDialog.getOpenFileName(self, "Load preset", str(d), "FrameVision Preset (*.json)")
        if not fn:
            return None
        p = Path(fn)
        try:
            config["last_preset_dir"] = str(p.parent)
            save_config()
        except Exception:
            pass
        return p

    # ---- Speed ----
    def _save_preset_speed(self):
        factor = self.speed.value()/100.0
        name = f"speed_{int(round(factor*100))}pct_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data = {"tool":"speed","factor":factor}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_speed(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="speed": raise ValueError("Wrong preset type")
            factor = float(data.get("factor", 1.0))
            self.speed.setValue(int(round(factor*100)))
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Resize ----
    def _save_preset_gif(self):
        inp = self._ensure_input(silent=True)
        if not inp:
            # we can still save UI-only presets without an input
            inp = Path('dummy.mp4')
        name = 'gif_preset.json'
        p = self._choose_save_path(name)
        if not p:
            return
        data = gif_backend.preset_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self,'gif_fps') else 0)
        p.write_text(json.dumps(data, indent=2), encoding='utf-8')
        try:
            QMessageBox.information(self, 'Preset saved', str(p))
        except Exception:
            pass

    def _load_preset_gif(self):
        p = self._choose_open_path()
        if not p:
            return
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            gif_backend.apply_preset(self, data)
            try:
                QMessageBox.information(self, 'Preset loaded', str(p))
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.critical(self, 'Preset error', str(e))
            except Exception:
                pass

    def _save_preset_extract(self):
        name = "extract_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data={"tool":"extract"}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_extract(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="extract": raise ValueError("Wrong preset type")
            QMessageBox.information(self, "Preset", "Loaded extract preset (no parameters).")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Trim ----
    def _save_preset_trim(self):
        start=self.trim_start.text().strip(); end=self.trim_end.text().strip()
        mode="copy" if self.trim_mode.currentIndex()==0 else "encode"
        def safe(s):
            return (s or "none").replace(":", "-").replace(".", "_").replace(" ", "")
        name = f"trim_{safe(start)}_{safe(end)}_{mode}_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data={"tool":"trim","mode":mode,"start":start,"end":end}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_trim(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="trim": raise ValueError("Wrong preset type")
            mode=data.get("mode","copy"); self.trim_mode.setCurrentIndex(0 if mode=="copy" else 1)
            self.trim_start.setText(data.get("start",""))
            self.trim_end.setText(data.get("end",""))
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Crop ----
    def _save_preset_crop(self):
        y = int(self.crop_y.value()); h = int(self.crop_h.value())
        name = f"crop_y{y}_h{h}_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data={"tool":"crop","y":y,"h":h}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_crop(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="crop": raise ValueError("Wrong preset type")
            self.crop_y.setValue(int(data.get("y", self.crop_y.value())))
            self.crop_h.setValue(int(data.get("h", self.crop_h.value())))
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))


    def _map_preset(self, enc: str, preset_txt: str) -> list[str]:
        # Map generic preset to encoder-specific flags
        p = preset_txt.lower()
        if enc in ("libx264","libx265"):
            return ["-preset", p]
        if enc.endswith("_nvenc"):
            # map to p1..p7; medium->p5
            m = {"ultrafast":"p1","superfast":"p2","veryfast":"p3","faster":"p4","fast":"p4","medium":"p5","slow":"p6","slower":"p6","veryslow":"p7"}
            return ["-preset", m.get(p, "p5")]
        if enc == "libsvtav1":
            # 0 best quality/slowest .. 13 fastest; map around medium=6
            m = {"ultrafast":"12","superfast":"11","veryfast":"10","faster":"9","fast":"8","medium":"6","slow":"4","slower":"3","veryslow":"2"}
            return ["-preset", m.get(p, "6")]
        return []

    def _audio_flags(self):
        a = (self.q_audio.currentText() if hasattr(self, 'q_audio') else "copy").lower()
        if a.startswith("copy"):
            return ["-c:a","copy"]
        if "opus" in a:
            kb = re.findall(r"(\d+)", a)
            return ["-c:a","libopus","-b:a", f"{kb[0]}k" if kb else "128k"]
        # default aac
        kb = re.findall(r"(\d+)", a)
        return ["-c:a","aac","-b:a", f"{kb[0]}k" if kb else "128k"]

    def _build_quality_cmd_for(self, inp_path):
        from pathlib import Path as _P
        enc, short = self._codec_selected()
        fmt = (self.q_format.currentText() if hasattr(self,'q_format') else "mp4").lower()
        mode = 0 if (hasattr(self,'q_mode') and self.q_mode.currentIndex()==0) else 1
        crf = int(self.q_crf.value()) if hasattr(self,'q_crf') else 23
        br = int(self.q_bitrate.value()) if hasattr(self,'q_bitrate') else 3500
        preset = self._map_preset(enc, self.q_preset.currentText() if hasattr(self,'q_preset') else "medium")

        inp = _P(inp_path)
        suf = (f"{short}_crf{crf}" if mode==0 else f"{short}_b{br}k")
        out = (OUT_VIDEOS if 'OUT_VIDEOS' in globals() else _P('.')) / f"{inp.stem}_{suf}.{fmt}"
        cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-c:v", enc]

        # presets
        cmd += preset

        if mode == 0:
            if enc.endswith("_nvenc"):
                # CQ mode for NVENC
                cmd += ["-rc","vbr","-cq", str(crf), "-b:v","0"]
            elif enc == "libsvtav1":
                cmd += ["-crf", str(crf)]
            else:
                cmd += ["-crf", str(crf)]
        else:
            cmd += ["-b:v", f"{br}k", "-maxrate", f"{br}k", "-bufsize", f"{br*2}k"]

        # audio
        cmd += self._audio_flags()

        # container tweaks
        if fmt == "mp4":
            cmd += ["-movflags","+faststart"]

        return cmd, out

    def run_quality(self):
        inp = self._ensure_input()
        if not inp: return
        try:
            cmd, out = self._build_quality_cmd_for(inp)
            self._run(cmd, out)
        except Exception as e:
            try: QMessageBox.warning(self, "Quality tool", str(e))
            except Exception: pass
    def run_quality_batch_popup(self):
        choice = self._ask_files_or_folder("Batch videos")
        if choice == "files":
            return self.run_quality_batch()
        elif choice == "folder":
            return self.run_quality_batch_folder()
        return None



    def run_quality_batch(self):
        paths = self._batch_paths_prompt(True, "Batch")
        if not paths: return
        ok = 0
        for p in paths:
            try:
                cmd, out = self._build_quality_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch Quality", f"Queued {ok} item(s).")
        except Exception: pass
    def run_quality_batch_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick a folder with videos", "")
        if not d: 
            return
        import os
        # typical video extensions
        exts = {".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts"}
        paths = []
        for name in os.listdir(d):
            p = Path(d) / name
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
        if not paths:
            try: QMessageBox.information(self, "Batch folder", "No video files were found in that folder.")
            except Exception: pass
            return
        ok = 0
        for p in paths:
            try:
                cmd, out = self._build_quality_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch folder", f"Queued {ok} file(s).")
        except Exception: pass

    def _probe_codecs_and_disable_unavailable(self):
        """Probe ffmpeg encoders once and grey-out unsupported codec choices in the Quality/Size tool."""
        try:
            ff = ffmpeg_path()
            out = subprocess.check_output([ff, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True, creationflags=0 if os.name!="nt" else 0x08000000)
        except Exception:
            out = ""
        encs = set()
        for ln in out.splitlines():
            m = re.match(r"\s*[A-Z\.]{6}\s+(\S+)", ln)
            if m:
                encs.add(m.group(1).strip().lower())
        # Map combobox display -> encoder id
        map_disp = {
            "H.264 (x264)": "libx264",
            "H.265 (x265)": "libx265",
            "AV1 (SVT-AV1)": "libsvtav1",
            "H.264 (NVENC)": "h264_nvenc",
            "HEVC (NVENC)": "hevc_nvenc",
            "AV1 (NVENC)": "av1_nvenc",
        }
        for i in range(self.q_codec.count()):
            disp = self.q_codec.itemText(i)
            enc = map_disp.get(disp, "").lower()
            available = (enc in encs) if enc else True
            try:
                # Grey-out
                itm = self.q_codec.model().item(i)
                if itm is not None:
                    itm.setEnabled(bool(available))
                    if not available:
                        itm.setToolTip("Not available in this ffmpeg build")
                else:
                    # Fallback method
                    from PySide6.QtCore import Qt
                    self.q_codec.setItemData(i, bool(available), Qt.ItemDataRole.EnabledRole)
                    if not available:
                        self.q_codec.setItemData(i, "Not available in this ffmpeg build", Qt.ItemDataRole.ToolTipRole)
            except Exception:
                pass
    

    # ---- Image conversion helpers ----

    def _batch_paths_prompt(self, for_videos: bool|str, title: str):
        """Return a list of paths chosen via a Files/Folder dialog.
        for_videos: True -> videos only, False -> images only, "both" -> videos+images."""
        choice = self._ask_files_or_folder(title)
        allow_both = (for_videos == "both")
        if choice == "files":
            if allow_both:
                exts = "Videos & Images (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.mpg *.mpeg *.ts *.m2ts *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif)"
                files, _ = QFileDialog.getOpenFileNames(self, "Pick files", "", exts)
                return [str(p) for p in files]
            return self._pick_video_files() if for_videos else self._pick_image_files()
        elif choice == "folder":
            title_txt = "Pick a folder with videos" if (for_videos is True) else ("Pick a folder with images" if (for_videos is False) else "Pick a folder with media")
            d = QFileDialog.getExistingDirectory(self, title_txt, "")
            if not d:
                return []
            from pathlib import Path as _P
            if allow_both:
                exts = {".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts",".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
            elif for_videos:
                exts = {".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts"}
            else:
                exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
            paths = []
            try:
                for name in os.listdir(d):
                    pth = _P(d) / name
                    if pth.is_file() and pth.suffix.lower() in exts:
                        paths.append(str(pth))
            except Exception:
                pass
            return paths
        else:
            return []

    def _ask_files_or_folder(self, title: str = "Batch") -> str | None:
        """Ask user to choose 'files' or 'folder'. Returns 'files', 'folder', or None."""
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle(title)
            msg.setText("Choose what to batch")
            btn_files = msg.addButton("Files…", QMessageBox.AcceptRole)
            btn_folder = msg.addButton("Folder…", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_files:
                return "files"
            if clicked == btn_folder:
                return "folder"
            return None
        except Exception:
            return None

    def _pick_video_files(self):

        exts = "Videos (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.mpg *.mpeg *.ts *.m2ts)"
        files, _ = QFileDialog.getOpenFileNames(self, "Pick videos", "", exts)
        return [str(p) for p in files]
    def _pick_image_files(self):
        exts = "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif)"
        files, _ = QFileDialog.getOpenFileNames(self, "Pick images", "", exts)
        return [str(p) for p in files]

    
    def _estimate_quality_for_target_image(self, inp_path: str, fmt: str, target_bytes: int, mw: int, mh: int) -> int:
        """Estimate quality (1..100) for a target size after optional resize using heuristic bits-per-pixel curves."""
        import subprocess, json, os
        # Probe size
        w = h = None
        try:
            ff = ffmpeg_path()
            probe = ff.replace("ffmpeg", "ffprobe")
            out = subprocess.check_output([probe, "-v", "error", "-select_streams", "v:0",
                                           "-show_entries", "stream=width,height",
                                           "-of", "json", inp_path], text=True, creationflags=0x08000000 if os.name=="nt" else 0)
            data = json.loads(out)
            st = (data.get("streams") or [{}])[0]
            w = int(st.get("width") or 0) or None
            h = int(st.get("height") or 0) or None
        except Exception:
            pass
        if not w or not h:
            w, h = 1920, 1080
        # Apply resize box conservatively (decrease)
        if mw or mh:
            sx = (mw / w) if mw else 1.0
            sy = (mh / h) if mh else 1.0
            s = min(sx if mw else 1.0, sy if mh else 1.0)
            if s > 0: w, h = max(1, int(w*s)), max(1, int(h*s))
        pixels = max(1, w*h)
        bpp = (target_bytes * 8.0) / pixels

        def jpg_q(b):
            pts = [(0.15,50),(0.25,60),(0.35,70),(0.55,80),(0.9,90),(1.3,95),(2.0,98)]
            if b <= pts[0][0]: return 45
            for i in range(len(pts)-1):
                x1,y1=pts[i]; x2,y2=pts[i+1]
                if b<=x2: return int(round(y1+(y2-y1)*(b-x1)/(x2-x1)))
            return 98
        def webp_q(b):
            pts = [(0.10,45),(0.20,60),(0.30,70),(0.45,80),(0.65,88),(0.85,93),(1.1,97)]
            if b <= pts[0][0]: return 40
            for i in range(len(pts)-1):
                x1,y1=pts[i]; x2,y2=pts[i+1]
                if b<=x2: return int(round(y1+(y2-y1)*(b-x1)/(x2-x1)))
            return 97

        if (fmt or "jpg").lower() == "webp":
            return max(1, min(100, webp_q(bpp)))
        return max(1, min(100, jpg_q(bpp)))
    def _build_image_cmd_for(self, inp_path):
            from pathlib import Path as _P
            fmt = (self.img_format.currentText() if hasattr(self,'img_format') else "jpg")
            q = int(self.img_quality.value()) if hasattr(self,'img_quality') else 85
            mw = int(self.img_w.value()) if hasattr(self,'img_w') else 0
            mh = int(self.img_h.value()) if hasattr(self,'img_h') else 0
            keep_meta = bool(self.img_keep_meta.isChecked()) if hasattr(self,'img_keep_meta') else False
            use_target = bool(getattr(self,'img_mode',None) and self.img_mode.currentIndex()==1 and (self.img_format.currentText().lower()!="png"))
            if use_target:
                # Convert target to bytes
                unit = (self.img_target_unit.currentText() if hasattr(self,'img_target_unit') else 'KB')
                mult = 1024 if unit=='KB' else 1024*1024
                target_bytes = int(self.img_target.value()) * mult if hasattr(self,'img_target') else 400*1024
                q = self._estimate_quality_for_target_image(str(inp_path), fmt, target_bytes, mw, mh)
    
            inp = _P(inp_path)
            suf = (f"{fmt}_t{int(self.img_target.value())}{self.img_target_unit.currentText().lower()}" if (use_target and hasattr(self,'img_target')) else f"{fmt}_q{q}") + (f"_w{mw}" if mw else "") + (f"_h{mh}" if mh else "")
            out = (OUT_IMAGES if 'OUT_IMAGES' in globals() else _P('.')) / f"{inp.stem}_{suf}.{fmt}"
    
            cmd = [ffmpeg_path(), "-y", "-i", str(inp)]
    
            # Resize if requested
            if mw or mh:
                if mw == 0: mw = -1
                if mh == 0: mh = -1
                filt = f"scale=w={mw if mw>0 else 'iw'}:h={mh if mh>0 else 'ih'}:force_original_aspect_ratio=decrease"
                cmd += ["-vf", filt]
    
            # Map format & quality
            if fmt == "jpg":
                qscale = max(2, min(31, int(round(31 - (q/100.0)*29))))
                cmd += ["-q:v", str(qscale)]
            elif fmt == "webp":
                cmd += ["-c:v", "libwebp", "-q:v", str(q), "-compression_level", "4"]
            elif fmt == "png":
                clevel = max(0, min(9, int(round((100-q)/100.0 * 9))))
                cmd += ["-compression_level", str(clevel)]
    
            if keep_meta:
                cmd += ["-map_metadata", "0"]
            else:
                cmd += ["-map_metadata", "-1"]
    
            return cmd, out

    def run_img_convert(self):
        inp = self._ensure_input(allow_images=True)
        if not inp:
            files = self._pick_image_files()
            if not files: return
            inp = files[0]
        cmd, out = self._build_image_cmd_for(inp)
        self._run(cmd, out)
    def run_img_batch_popup(self):
        choice = self._ask_files_or_folder("Batch images")
        if choice == "files":
            return self.run_img_batch()
        elif choice == "folder":
            return self.run_img_batch_folder()
        return None
    def run_img_batch_popup(self):
        choice = self._ask_files_or_folder("Batch images")
        if choice == "files":
            return self.run_img_batch()
        elif choice == "folder":
            return self.run_img_batch_folder()
        return None





    def run_img_batch(self):
        files = self._pick_image_files()
        if not files: return
        ok = 0
        for p in files:
            try:
                cmd, out = self._build_image_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch images", f"Queued {ok} image(s).")
        except Exception: pass
    def run_img_batch_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick a folder with images", "")
        if not d: 
            return
        import os
        img_exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
        paths = []
        for name in os.listdir(d):
            p = Path(d) / name
            if p.is_file() and p.suffix.lower() in img_exts:
                paths.append(str(p))
        if not paths:
            try: QMessageBox.information(self, "Batch folder", "No images found in that folder.")
            except Exception: pass
            return
        ok = 0
        for p in paths:
            try:
                cmd, out = self._build_image_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch folder", f"Queued {ok} image(s).")
        except Exception: pass
    def _enqueue_cmd_for_input(self, inp_path, cmd_list, outfile):
                try:
                    from pathlib import Path as _P
                    out_path = outfile if isinstance(outfile, _P) else _P(str(outfile))
                    try:
                        from helpers.queue_adapter import enqueue_tool_job as enq
                    except Exception:
                        from queue_adapter import enqueue_tool_job as enq
                    enq("tools_ffmpeg", str(inp_path), str(out_path.parent), {"ffmpeg_cmd": cmd_list, "outfile": str(out_path)}, priority=600)
                    return True
                except Exception:
                    return False

    def run_speed_batch(self):
        paths = self._batch_paths_prompt(True, "Batch")
        if not paths:
            return
        try:
            if QMessageBox.question(self, "Batch Speed", f"Add {len(paths)} file(s) with current Speed settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        # Build per-file
        for p in paths:
            try:
                from pathlib import Path as _P
                inp = _P(p)
                factor = self.speed.value() / 100.0
                out = (OUT_VIDEOS if 'OUT_VIDEOS' in globals() else _P('.')) / f"{inp.stem}_spd_{factor:.2f}x.mp4"
                setpts = 1.0 / float(factor if factor != 0 else 1.0)
                mute = bool(self.cb_speed_mute.isChecked()) if hasattr(self,'cb_speed_mute') else False
                sync = bool(self.cb_speed_sync.isChecked()) if hasattr(self,'cb_speed_sync') else True
                if mute:
                    cmd=[ffmpeg_path(),"-y","-i",str(inp),"-an","-vf",f"setpts={setpts}*PTS",str(out)]
                else:
                    if sync:
                        atempo = factor if factor>=0.5 else 0.5
                        cmd=[ffmpeg_path(),"-y","-i",str(inp),"-vf",f"setpts={setpts}*PTS","-filter:a",f"atempo={atempo}",str(out)]
                    else:
                        cmd=[ffmpeg_path(),"-y","-i",str(inp),"-vf",f"setpts={setpts}*PTS",str(out)]
                self._enqueue_cmd_for_input(inp, cmd, out)
            except Exception:
                continue
        try:
            QMessageBox.information(self, "Batch Speed", f"Queued {len(paths)} item(s).")
        except Exception:
            pass

    
    def run_gif_batch(self):
        paths = self._pick_batch_files(for_videos=True)
        if not paths:
            return
        fmt = self.gif_fmt.currentText().strip().lower() if hasattr(self, "gif_fmt") else "gif"
        for p in paths:
            inp = Path(p)
            opts = gif_backend.options_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self, "gif_fps") else 0, batch=True)
            out = OUT_VIDEOS / gif_backend.output_name_for(inp.stem, fmt, batch=True)
            cmds = gif_backend.build_commands(inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
            # enqueue only the first command per spec
            self._enqueue_cmd_for_input(inp, cmds[0], out)
    def run_crop_batch(self):
        paths = self._batch_paths_prompt(True, "Batch")
        if not paths:
            return
        try:
            if QMessageBox.question(self, "Batch Crop", f"Add {len(paths)} file(s) with current Crop settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        for p in paths:
            try:
                from pathlib import Path as _P
                inp=_P(p)
                y=int(self.crop_y.value()); h=int(self.crop_h.value())
                out=OUT_VIDEOS / f"{inp.stem}_crop_y{y}_h{h}.mp4"
                filter_str = f"crop=iw:{h}:0:{y}"
                cmd=[ffmpeg_path(),"-y","-i",str(inp),"-vf",filter_str,"-c:v","libx264","-preset","veryfast","-movflags","+faststart",str(out)]
                self._enqueue_cmd_for_input(inp, cmd, out)
            except Exception:
                continue
        try:
            QMessageBox.information(self, "Batch Crop", f"Queued {len(paths)} item(s).")
        except Exception:
            pass