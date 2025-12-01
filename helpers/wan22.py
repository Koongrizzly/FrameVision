from __future__ import annotations

import os
import sys
import json
import random
import subprocess
import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, Signal, QUrl, QTimer, QSize, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QSizePolicy,
    QLabel, QLineEdit, QTextEdit, QSpinBox, QComboBox, QSlider,
    QPushButton, QFileDialog, QPlainTextEdit, QCheckBox,
    QApplication, QMessageBox, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup, QMenu, QInputDialog,
)
from PySide6.QtWidgets import QScrollArea
from PySide6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, QDesktopServices, QIcon
from helpers.mediainfo import refresh_info_now

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
SETTINGS_FILE = APP_ROOT / "presets" / "setsave" / "wan22.json"

TEMP_DIR = APP_ROOT / "temp"
try:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

EXTEND_FRAMES_DIR = APP_ROOT / "output" / "frames" / "extend"
try:
    EXTEND_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

USE_NVENC_DEFAULT = False  # Experimental: try to re-encode final video with FFmpeg NVENC


def _ffmpeg_exe() -> str | None:
    """
    Locate the bundled ffmpeg executable, if present.
    On Windows we expect: <app-root>/presets/bin/ffmpeg.exe
    On other platforms we try <app-root>/presets/bin/ffmpeg then plain 'ffmpeg'.
    """
    if os.name == "nt":
        cand = APP_ROOT / "presets" / "bin" / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
        return None
    else:
        cand = APP_ROOT / "presets" / "bin" / "ffmpeg"
        if cand.exists():
            return str(cand)
        # Fallback to PATH
        return "ffmpeg"




def _ensure_settings_dir():
    """Ensure the settings directory exists"""
    settings_dir = APP_ROOT / "presets" / "setsave"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def _wan_python_exe() -> str:
    """
    Prefer the dedicated .wan_venv Python if it exists, otherwise fall back to
    the current interpreter. This lets FrameVision run Wan 2.2 in its own venv.
    """
    if os.name == "nt":
        cand = APP_ROOT / ".wan_venv" / "Scripts" / "python.exe"
    else:
        cand = APP_ROOT / ".wan_venv" / "bin" / "python"
    if cand.exists():
        return str(cand)
    return sys.executable


def _wan_model_root() -> Path:
    """
    Location of the Wan2.2 repo / weights. By default we expect:
        <app-root>/models/wan22
    """
    return APP_ROOT / "models" / "wan22"




class FlowLayout(QHBoxLayout):
    """Wrap layout for thumbnails that automatically reflows on resize."""

    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)

        self._rows: list[list[QWidget]] = []
        self._relayouting: bool = False

        self.setSpacing(spacing)
        self.setContentsMargins(8, 8, 8, 8)

        # Internal container so we can listen for resize events and
        # keep the layout self‑contained inside scroll areas.
        self._container = QWidget()
        self._container.installEventFilter(self)

        self._v = QVBoxLayout(self._container)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(spacing)

        super().addWidget(self._container)

    # --- Helpers ---------------------------------------------------------

    def _iter_widgets(self) -> list[QWidget]:
        """Collect all widgets currently managed by this flow layout."""
        widgets: list[QWidget] = []
        for i in range(self._v.count()):
            item = self._v.itemAt(i)
            lay = item.layout()
            if not lay:
                continue
            for j in range(lay.count()):
                it2 = lay.itemAt(j)
                w = it2.widget()
                if w is not None:
                    widgets.append(w)
        return widgets

    def _compute_max_per_row(self, widgets: list[QWidget]) -> int:
        """Decide how many thumbnails fit per row for the current width."""
        default = 6
        if not widgets:
            return default

        try:
            margins = self.contentsMargins()
            available = max(
                0,
                self._container.width() - (margins.left() + margins.right()),
            )
        except Exception:
            available = self._container.width()

        if available <= 0:
            return default

        # Use the first widget as a representative for thumbnail width
        rep = widgets[0]
        try:
            w_hint = rep.sizeHint().width()
        except Exception:
            w_hint = 0
        w_current = getattr(rep, "width", lambda: 0)() or 0
        item_w = max(w_hint, w_current, 1)

        total = item_w + self.spacing()
        if total <= 0:
            return default

        per_row = max(1, int(available / total))
        # Avoid silly extremes
        if per_row > 12:
            per_row = 12
        return per_row or default

    def _build_rows(self, widgets: list[QWidget]):
        """Internal helper to (re)build row layouts from a flat widget list."""
        # Remove any existing row layouts
        while self._v.count():
            item = self._v.takeAt(0)
            lay = item.layout()
            if lay is not None:
                # Let Qt own and delete the layout; widgets stay alive
                del lay

        self._rows.clear()
        if not widgets:
            return

        max_per_row = self._compute_max_per_row(widgets)
        if max_per_row < 1:
            max_per_row = len(widgets)

        row_layout: Optional[QHBoxLayout] = None
        count_in_row = 0

        for w in widgets:
            if row_layout is None or count_in_row >= max_per_row:
                row_layout = QHBoxLayout()
                row_layout.setSpacing(self.spacing())
                row_layout.setContentsMargins(0, 0, 0, 0)
                self._v.addLayout(row_layout)
                self._rows.append([])  # track widgets for this row
                count_in_row = 0

            row_layout.addWidget(w)
            self._rows[-1].append(w)
            count_in_row += 1

    # --- Public API ------------------------------------------------------

    def addWidget(self, w: QWidget):
        """Add a widget and place it into the appropriate row."""
        widgets = self._iter_widgets()
        widgets.append(w)
        self._build_rows(widgets)

    def relayout(self):
        """Re‑wrap existing widgets based on the current container width."""
        if self._relayouting:
            return
        self._relayouting = True
        try:
            widgets = self._iter_widgets()
            self._build_rows(widgets)
        finally:
            self._relayouting = False

    def clear(self):
        """Remove all widgets and reset internal state."""
        while self._v.count():
            item = self._v.takeAt(0)
            lay = item.layout()
            if lay:
                while lay.count():
                    it2 = lay.takeAt(0)
                    w = it2.widget()
                    if w:
                        w.setParent(None)
            elif item.widget():
                item.widget().setParent(None)
        self._rows.clear()

    # --- Qt events -------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._container and event.type() == QEvent.Resize:
            try:
                self.relayout()
            except Exception:
                pass
        return super().eventFilter(obj, event)

class Collapsible(QFrame):
    toggled = Signal(bool)
    def __init__(self, title: str, start_open: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        self.header = QPushButton(title)
        self.header.setCheckable(True)
        self.header.setChecked(start_open)
        self.header.setStyleSheet("text-align:left; padding:6px; font-weight:600;")
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 8, 8, 8)
        self.body_layout.setSpacing(6)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.header.toggled.connect(self.body.setVisible)
        self.header.toggled.connect(self.toggled.emit)
        self.body.setVisible(start_open)
class Wan22Pane(QWidget):
    """
    Very small GUI pane that shells out to Wan2.2's official generate.py CLI.

    It assumes the Wan2.2 repository (with generate.py, wan/, assets/, etc.)
    lives under models/wan22 and that the model weights are in the same folder.
    """
    fileReady = Signal(object)  # emits Path of the produced file (if found)

    def __init__(self, main=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main = main

        # Thumbnail size state for Recent results thumbnails
        self._thumb_base_size = 180
        self._thumb_display_size = self._thumb_base_size
        self._recent_thumb_buttons = []
        # Extend-chain state (text2video → chained image2video segments)
        self._extend_active = False
        self._extend_remaining = 0
        self._extend_segments: list[Path] = []
        self._extend_frame_index = 0
        self._extend_auto_merge = False
        self._extend_include_source = False
        self._extend_pending_output: Optional[Path] = None
        # Remember the exact output path used for the last direct run
        # so extend-chains can always find the correct clip instead of
        # guessing from the recent-results folder.
        self._last_run_out_path: Optional[Path] = None

        # Video-to-video: selected source video (if any)
        self._video2video_path: Optional[Path] = None

        # Settings persistence
        self._load_settings()

        layout = QVBoxLayout(self)

        # Fancy yellow banner at the top
        self.banner = QLabel("Video Creation with Wan 2.2 5B")
        self.banner.setObjectName("wanBanner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.banner.setFixedHeight(45)
        self.banner.setStyleSheet(
            "#wanBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: #332b00;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #fff176,"
            "   stop:0.5 #ffeb3b,"
            "   stop:1 #ffc107"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        layout.addWidget(self.banner)
        layout.addSpacing(4)


        # --- Top: mode selector -------------------------------------------------
        top = QHBoxLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["text2video", "image2video"])
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.cmb_mode)
        top.addWidget(QLabel("  (WAN 2.2 TI2V-5B)"))
        top.addStretch(1)
        layout.addLayout(top)

        # --- Form with settings -------------------------------------------------
        form = QFormLayout()

        # Prompt
        self.ed_prompt = QTextEdit()
        self.ed_prompt.setPlaceholderText("Describe the video you want to generate…")
        self.ed_prompt.setFixedHeight(80)
        self.ed_prompt.setToolTip("Describe the video you want to generate. Be specific about what you want to see.")
        form.addRow("Prompt:", self.ed_prompt)

        # Negative prompt
        self.ed_negative = QTextEdit()
        self.ed_negative.setPlaceholderText("Things you do NOT want to see in the video (e.g. low quality, distortion, text artifacts)")
        self.ed_negative.setFixedHeight(60)
        self.ed_negative.setToolTip("Negative prompt: describe what should be avoided in the video.")
        form.addRow("Negative:", self.ed_negative)

        # Start image (for image2video)
        img_row = QHBoxLayout()
        self.ed_image = QLineEdit()
        self.ed_image.setToolTip("Path to the starting image for image-to-video generation")
        btn_img = QPushButton("Browse")

        def _pick_image():
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Choose start image",
                "",
                "Images (*.png *.jpg *.jpeg *.webp);;All files (*.*)",
            )
            if fn:
                self.ed_image.setText(fn)

        btn_img.clicked.connect(_pick_image)
        img_row.addWidget(self.ed_image)
        img_row.addWidget(btn_img)
        img_widget = QWidget()
        img_widget.setLayout(img_row)
        form.addRow("Start image:", img_widget)

        # Video to video (use last frame of an existing video as the start image)
        video2_row = QHBoxLayout()
        self.chk_video2video = QCheckBox("Video to video")
        self.chk_video2video.setToolTip(
            "Use the last frame of a selected video as the start image for a new generation."
        )
        self.btn_video2_browse = QPushButton("Browse")
        self.btn_video2_browse.setToolTip("Pick a source video to start from.")
        self.lbl_video2_info = QLabel("No video selected")
        self.lbl_video2_info.setStyleSheet("color:#888;")

        # Hidden by default until the toggle is enabled
        self.btn_video2_browse.setVisible(False)
        self.lbl_video2_info.setVisible(False)

        self.chk_video2video.toggled.connect(self._on_video2video_toggled)
        self.btn_video2_browse.clicked.connect(self._on_video2video_browse)

        video2_row.addWidget(self.chk_video2video)
        video2_row.addWidget(self.btn_video2_browse)
        video2_row.addWidget(self.lbl_video2_info, 1)
        video2_widget = QWidget()
        video2_widget.setLayout(video2_row)
        form.addRow("", video2_widget)

        # Size & Seed on same line
        size_seed_row = QHBoxLayout()
        
        # Size preset (only working resolutions)
        self.cmb_size = QComboBox()
        self.cmb_size.addItems([
            "1280*704",    # Landscape 704p (primary)
            "704*1280"     # Portrait 704p  
        ])
        self.cmb_size.setToolTip("Video resolution. Only these two seem to work reliably with Wan2.2")
        
        # Seed controls
        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(0, 2147483647)
        self.spn_seed.setValue(42)
        self.spn_seed.setToolTip("Random seed for reproducible results. Same seed = same video")
        
        self.chk_random_seed = QCheckBox("Random")
        self.chk_random_seed.setToolTip("When checked: generates a new random seed for each video generation. When unchecked: use the manual seed value above")
        self.chk_random_seed.toggled.connect(self._on_random_seed_toggled)
        
        # Sync random seed with seed spinbox changes
        self.spn_seed.valueChanged.connect(self._on_seed_changed)
        
        size_seed_row.addWidget(QLabel("Size:"))
        size_seed_row.addWidget(self.cmb_size)
        size_seed_row.addSpacing(20)
        size_seed_row.addWidget(QLabel("Seed:"))
        size_seed_row.addWidget(self.spn_seed)
        size_seed_row.addWidget(self.chk_random_seed)
        size_seed_row.addStretch(1)
        size_seed_widget = QWidget()
        size_seed_widget.setLayout(size_seed_row)
        form.addRow("", size_seed_widget)

        # Steps & Guidance on same line  
        steps_guidance_row = QHBoxLayout()
        
        # Steps control (spinner and slider)
        self.spn_steps = QSpinBox()
        self.spn_steps.setRange(1, 100)
        self.spn_steps.setValue(30)
        self.spn_steps.setToolTip("Number of sampling steps. Higher = better quality but slower")
        
        self.slider_steps = QSlider(Qt.Horizontal)
        self.slider_steps.setRange(1, 100)
        self.slider_steps.setValue(30)
        self.slider_steps.setToolTip("Number of sampling steps. Higher = better quality but slower")
        
        # Sync spinner and slider
        self.spn_steps.valueChanged.connect(self.slider_steps.setValue)
        self.slider_steps.valueChanged.connect(self.spn_steps.setValue)
        
        # Guidance scale control (spinner and slider)
        self.spn_guidance = QSpinBox()
        self.spn_guidance.setRange(1, 20)
        self.spn_guidance.setValue(7)
        self.spn_guidance.setToolTip("How closely to follow the prompt. Higher = more literal")
        
        self.slider_guidance = QSlider(Qt.Horizontal)
        self.slider_guidance.setRange(1, 20)
        self.slider_guidance.setValue(7)
        self.slider_guidance.setToolTip("How closely to follow the prompt. Higher = more literal")
        
        # Sync spinner and slider for guidance
        self.spn_guidance.valueChanged.connect(self.slider_guidance.setValue)
        self.slider_guidance.valueChanged.connect(self.spn_guidance.setValue)
        
        steps_guidance_row.addWidget(QLabel("Steps:"))
        steps_guidance_row.addWidget(self.spn_steps)
        steps_guidance_row.addWidget(self.slider_steps)
        steps_guidance_row.addSpacing(20)
        steps_guidance_row.addWidget(QLabel("Guidance:"))
        steps_guidance_row.addWidget(self.spn_guidance)
        steps_guidance_row.addWidget(self.slider_guidance)
        steps_guidance_row.addStretch(1)
        steps_guidance_widget = QWidget()
        steps_guidance_widget.setLayout(steps_guidance_row)
        form.addRow("", steps_guidance_widget)

        # Frames / FPS controls
        frames_row = QHBoxLayout()
        self.spn_frames = QSpinBox()
        self.spn_frames.setRange(24, 240)
        self.spn_frames.setValue(121)
        self.spn_frames.setToolTip("Number of frames in the generated video")

        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 60)
        self.spn_fps.setValue(24)
        self.spn_fps.setToolTip("Frames per second. Affects video smoothness and duration")

        # Batch button (image2video only)
        self.btn_img_batch = QPushButton("Batch")
        self.btn_img_batch.setToolTip(
            "Batch image2video jobs: multiple images or repeat the current image."
        )
        self.btn_img_batch.clicked.connect(self._show_image_batch_dialog)

        # Batch count (text2video only)
        self.spn_batch = QSpinBox()
        self.spn_batch.setRange(1, 99)
        self.spn_batch.setValue(1)
        self.spn_batch.setToolTip(
            "Number of videos to generate in a batch (text2video only).\n"
            "When greater than 1, jobs will always go through the queue."
        )

        # Extend chain controls (text2video only)
        self.lbl_extend = QLabel("Extend:")
        self.spn_extend = QSpinBox()
        self.spn_extend.setRange(0, 99)
        self.spn_extend.setValue(0)
        self.spn_extend.setToolTip(
            "Extend video by chaining additional segments.\n"
            "0 = off. When greater than 0, Wan will repeatedly take the last "
            "frame of the previous segment and use it as the start image for "
            "the next segment (direct runs only)."
        )
        self.chk_extend_merge = QCheckBox("Auto-merge")
        self.chk_extend_merge.setToolTip(
            "When enabled and Extend > 0, automatically merge all chained "
            "segments into one final video after the last segment finishes."
        )
        self.chk_extend_include_source = QCheckBox("Include source in merge")
        self.chk_extend_include_source.setToolTip(
            "Video to video only: when enabled, Auto-merge will prepend the "
            "source video before the extended segments in the merged output."
        )
        self.chk_extend_include_source.setVisible(False)

        # First row: frames + img-batch + text2video batch controls
        frames_row.addWidget(QLabel("Frames:"))
        frames_row.addWidget(self.spn_frames)
        frames_row.addWidget(self.btn_img_batch)
        frames_row.addSpacing(12)
        self.lbl_batch = QLabel("Batch:")
        frames_row.addWidget(self.lbl_batch)
        frames_row.addWidget(self.spn_batch)
        frames_row.addStretch(1)

        # Second row: extend controls (always visible)
        extend_row = QHBoxLayout()
        extend_row.addWidget(self.lbl_extend)
        extend_row.addWidget(self.spn_extend)
        extend_row.addWidget(self.chk_extend_merge)
        extend_row.addWidget(self.chk_extend_include_source)
        extend_row.addStretch(1)
        extend_widget = QWidget()
        extend_widget.setLayout(extend_row)
        form.addRow("", extend_widget)

        frames_widget = QWidget()
        frames_widget.setLayout(frames_row)
        form.addRow("", frames_widget)

        # Output path (optional – Wan2.2 can also auto-name)
        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Optional: Specify exact output file path. Leave empty for auto-naming")
        btn_out = QPushButton("Browse")

        def _pick_out():
            fn, _ = QFileDialog.getSaveFileName(
                self,
                "Choose output MP4 (optional)",
                "",
                "Video (*.mp4);;All files (*.*)",
            )
            if fn:
                if not fn.lower().endswith(".mp4"):
                    fn += ".mp4"
                self.ed_out.setText(fn)

        btn_out.clicked.connect(_pick_out)
        out_row.addWidget(self.ed_out)
        out_row.addWidget(btn_out)
        out_widget = QWidget()
        out_widget.setLayout(out_row)
        self.lbl_out = QLabel("Output path:")
        self.out_widget = out_widget
        form.addRow(self.lbl_out, self.out_widget)

        # Wrap settings + LoRA controls in a scroll area so the bottom buttons stay visible
        scroll_inner = QWidget()
        scroll_layout = QVBoxLayout(scroll_inner)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(6)
        scroll_layout.addLayout(form)

        scroll = QScrollArea()
        scroll.setWidget(scroll_inner)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        
        # --- Recent results (collapsible) --------------------------------------
        recent_box = Collapsible("Recent results", start_open=False)
        recent_header = QHBoxLayout()

        # Sort dropdown for recent results (before Refresh)
        lbl_recent_sort = QLabel("Sort:")
        self.combo_recent_sort = QComboBox()
        self.combo_recent_sort.addItem("Newest first", "newest")
        self.combo_recent_sort.addItem("Oldest first", "oldest")
        self.combo_recent_sort.addItem("Alphabetical (A-Z)", "az")
        self.combo_recent_sort.addItem("Alphabetical (Z-A)", "za")
        self.combo_recent_sort.addItem("Size (smallest first)", "size_small")
        self.combo_recent_sort.addItem("Size (largest first)", "size_large")

        recent_header.addWidget(lbl_recent_sort)
        recent_header.addWidget(self.combo_recent_sort)

        btn_recent_refresh = QPushButton("Refresh")
        btn_recent_open = QPushButton("Open folder")
        recent_header.addWidget(btn_recent_refresh)
        recent_header.addWidget(btn_recent_open)
        recent_header.addStretch(1)
        recent_box.body_layout.addLayout(recent_header)

        # Thumbnail size slider
        thumb_row = QHBoxLayout()
        lbl_thumb = QLabel("Thumbnail size:")
        self.sld_thumb_size = QSlider(Qt.Horizontal)
        self.sld_thumb_size.setRange(20, 100)
        self.sld_thumb_size.setValue(100)
        self.sld_thumb_size.setFixedWidth(160)
        self.lbl_thumb_size_value = QLabel("100%")
        thumb_row.addWidget(lbl_thumb)
        thumb_row.addWidget(self.sld_thumb_size)
        thumb_row.addWidget(self.lbl_thumb_size_value)
        thumb_row.addStretch(1)
        recent_box.body_layout.addLayout(thumb_row)

        # Recent thumbnails scroll area
        recent_scroll = QScrollArea()
        recent_scroll.setWidgetResizable(True)
        # Make the box a bit taller when opened so a full thumbnail is visible
        recent_scroll.setMinimumHeight(self._thumb_base_size + 40)
        recent_container = QWidget()
        recent_layout = FlowLayout()
        recent_container.setLayout(recent_layout)
        recent_scroll.setWidget(recent_container)
        recent_box.body_layout.addWidget(recent_scroll)
        layout.addWidget(recent_box)

        # keep refs
        self._recent_box = recent_box
        self._recent_layout = recent_layout
        self._recent_container = recent_container
        self._recent_scroll = recent_scroll
        self._btn_recent_refresh = btn_recent_refresh
        self._btn_recent_open = btn_recent_open

# --- LoRA management ----------------------------------------------------
        lora_form = QFormLayout()
        
        # LoRA 1 configuration
        lora1_row = QHBoxLayout()
        self.ed_lora1_path = QLineEdit()
        self.ed_lora1_path.setPlaceholderText("Path to LoRA 1 file (.safetensors/.ckpt)")
        self.btn_lora1_browse = QPushButton("Browse")
        self.btn_lora1_browse.setToolTip("Browse for LoRA 1 file")

        def _pick_lora1():
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Choose LoRA 1 file",
                "",
                "LoRA files (*.safetensors *.ckpt);;All files (*.*)",
            )
            if fn:
                self.ed_lora1_path.setText(fn)

        self.btn_lora1_browse.clicked.connect(_pick_lora1)
        
        # LoRA 1 weight
        self.lbl_lora1_weight = QLabel("Weight:")
        self.slider_lora1_weight = QSlider(Qt.Horizontal)
        self.slider_lora1_weight.setRange(0, 100)  # 0.0 to 1.0
        self.slider_lora1_weight.setValue(70)  # Default: 0.7
        self.slider_lora1_weight.setToolTip("LoRA 1 weight (0.0 = no effect, 1.0 = full effect)")
        
        self.lbl_lora1_value = QLabel("0.70")
        self.lbl_lora1_value.setMinimumWidth(40)
        
        # Update LoRA 1 weight label when slider moves
        self.slider_lora1_weight.valueChanged.connect(
            lambda v: self.lbl_lora1_value.setText(f"{v/100:.2f}")
        )
        
        lora1_file_layout = QHBoxLayout()
        lora1_file_layout.addWidget(self.ed_lora1_path, stretch=1)
        lora1_file_layout.addWidget(self.btn_lora1_browse)
        
        lora1_layout = QVBoxLayout()
        lora1_layout.addLayout(lora1_file_layout)
        
        lora1_weight_layout = QHBoxLayout()
        lora1_weight_layout.addWidget(self.lbl_lora1_weight)
        lora1_weight_layout.addWidget(self.slider_lora1_weight)
        lora1_weight_layout.addWidget(self.lbl_lora1_value)
        lora1_layout.addLayout(lora1_weight_layout)
        
        # LoRA 2 configuration
        lora2_row = QHBoxLayout()
        self.ed_lora2_path = QLineEdit()
        self.ed_lora2_path.setPlaceholderText("Path to LoRA 2 file (.safetensors/.ckpt)")
        self.btn_lora2_browse = QPushButton("Browse")
        self.btn_lora2_browse.setToolTip("Browse for LoRA 2 file")

        def _pick_lora2():
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Choose LoRA 2 file",
                "",
                "LoRA files (*.safetensors *.ckpt);;All files (*.*)",
            )
            if fn:
                self.ed_lora2_path.setText(fn)

        self.btn_lora2_browse.clicked.connect(_pick_lora2)
        
        # LoRA 2 weight
        self.lbl_lora2_weight = QLabel("Weight:")
        self.slider_lora2_weight = QSlider(Qt.Horizontal)
        self.slider_lora2_weight.setRange(0, 100)  # 0.0 to 1.0
        self.slider_lora2_weight.setValue(50)  # Default: 0.5
        self.slider_lora2_weight.setToolTip("LoRA 2 weight (0.0 = no effect, 1.0 = full effect)")
        
        self.lbl_lora2_value = QLabel("0.50")
        self.lbl_lora2_value.setMinimumWidth(40)
        
        # Update LoRA 2 weight label when slider moves
        self.slider_lora2_weight.valueChanged.connect(
            lambda v: self.lbl_lora2_value.setText(f"{v/100:.2f}")
        )
        
        lora2_file_layout = QHBoxLayout()
        lora2_file_layout.addWidget(self.ed_lora2_path, stretch=1)
        lora2_file_layout.addWidget(self.btn_lora2_browse)
        
        lora2_layout = QVBoxLayout()
        lora2_layout.addLayout(lora2_file_layout)
        
        lora2_weight_layout = QHBoxLayout()
        lora2_weight_layout.addWidget(self.lbl_lora2_weight)
        lora2_weight_layout.addWidget(self.slider_lora2_weight)
        lora2_weight_layout.addWidget(self.lbl_lora2_value)
        lora2_layout.addLayout(lora2_weight_layout)
        
        # Relighting LoRA (existing feature)
        self.chk_relighting_lora = QCheckBox("Use relighting LoRA")
        self.chk_relighting_lora.setChecked(False)
        self.chk_relighting_lora.setToolTip("Apply relighting LoRA for enhanced lighting effects")
        
        # Add LoRA controls to form
        lora_form.addRow("LoRA 1:", lora1_layout)
        lora_form.addRow("LoRA 2:", lora2_layout)
        lora_form.addRow("", self.chk_relighting_lora)
        
        lora_widget = QWidget()
        lora_widget.setLayout(lora_form)
        scroll_layout.addWidget(lora_widget)

        # --- Log output ---------------------------------------------------------
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setToolTip("Generation log and progress information")
        self.log.setFixedHeight(52)   # tweak this number to taste
        layout.addWidget(self.log, stretch=1)

        # --- Buttons ------------------------------------------------------------
        btn_row = QHBoxLayout()

        # Queue toggle: run via worker queue vs direct QProcess
        self.chk_use_queue = QCheckBox("Use queue")
        self.chk_use_queue.setToolTip("Tip : Always use queue if you want to do more then one job in the app")

        # Use Current button: grab current frame from Media Player
        self.btn_use_current = QPushButton("Use Current")
        self.btn_use_current.setToolTip(
            "Switch to image2video and use the current Media Player frame as the start image."
        )

        # Probe button (kept for possible future use, but hidden from the layout)
        self.btn_probe = QPushButton("Probe")
        self.btn_probe.setToolTip("Check if Wan2.2 is properly installed and configured")
        self.btn_probe.setVisible(False)

        # Main RUN button: more prominent + hover color
        self.btn_run = QPushButton("Generate Video")
        self.btn_run.setToolTip("Start video generation")
        self.btn_run.setObjectName("wanRunButton")
        self.btn_run.setStyleSheet(
            "QPushButton#wanRunButton {"
            "  font-size: 16px;"
            "  font-weight: 600;"
            "  padding: 6px 22px;"
            "  border-radius: 6px;"
            "}"
            "QPushButton#wanRunButton:hover {"
            "  background-color: #ffca28;"
            "  color: #201a00;"
            "}"
        )

        # Match font size/weight of the RUN button for the rest of the row
        shared_btn_font = "font-size: 16px; font-weight: 600;"
        self.chk_use_queue.setStyleSheet(shared_btn_font)
        self.btn_use_current.setStyleSheet(shared_btn_font)

        # "Play last" button: opens the most recently generated video
        self.btn_play_last = QPushButton("Play last")
        self.btn_play_last.setToolTip("Open the last generated Wan2.2 video")
        self.btn_play_last.setStyleSheet(shared_btn_font)

        # Order: RUN button first, then Play last, then queue toggle, then Use Current
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_play_last)
        btn_row.addWidget(self.chk_use_queue)
        btn_row.addWidget(self.btn_use_current)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # --- Process handling ---------------------------------------------------
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_proc)
        # With MergedChannels, all output arrives on standard output; we don't need
        # to read standard error separately (doing so prints Qt warnings).
        self.proc.finished.connect(self._on_finished)

        # Force UTF-8 in child process so Chinese negative prompts don't break logging
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self.proc.setProcessEnvironment(env)

        # Mode toggling
        self.cmb_mode.currentTextChanged.connect(self._update_mode)
        self._update_mode()

        # Button actions
        # Probe button is hidden in the UI but kept for possible future use
        # self.btn_probe.clicked.connect(self._do_probe)
        self.btn_run.clicked.connect(self._launch)
        self.btn_play_last.clicked.connect(self._on_play_last)
        self.btn_use_current.clicked.connect(self._on_use_current)
        
        # Connect value change signals to save settings
        self._connect_signals_for_saving()
        
        # Apply loaded settings after UI is fully initialized
        self._apply_loaded_settings()

        # Recent wiring
        try:
            self._btn_recent_refresh.clicked.connect(self._refresh_recent)
            self._btn_recent_open.clicked.connect(self._on_recent_open)
            if hasattr(self, "sld_thumb_size"):
                self.sld_thumb_size.valueChanged.connect(self._on_thumb_size_changed)
            if getattr(self, "combo_recent_sort", None):
                self.combo_recent_sort.currentIndexChanged.connect(lambda _idx: self._refresh_recent())
            QTimer.singleShot(0, self._refresh_recent)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Settings Persistence
    # ---------------------------------------------------------------------
    def _connect_signals_for_saving(self):
        """Connect all control signals to save settings when values change"""
        self.cmb_mode.currentTextChanged.connect(self._save_settings)
        self.ed_prompt.textChanged.connect(self._save_settings)
        if getattr(self, "ed_negative", None):
            self.ed_negative.textChanged.connect(self._save_settings)
        self.ed_image.textChanged.connect(self._save_settings)
        self.cmb_size.currentTextChanged.connect(self._save_settings)
        self.spn_seed.valueChanged.connect(self._save_settings)
        self.chk_random_seed.toggled.connect(self._save_settings)
        self.spn_steps.valueChanged.connect(self._save_settings)
        self.spn_guidance.valueChanged.connect(self._save_settings)
        self.spn_frames.valueChanged.connect(self._save_settings)
        self.spn_fps.valueChanged.connect(self._save_settings)
        if getattr(self, "spn_extend", None):
            self.spn_extend.valueChanged.connect(self._save_settings)
        if getattr(self, "chk_extend_merge", None):
            self.chk_extend_merge.toggled.connect(self._save_settings)
        if getattr(self, "chk_extend_include_source", None):
            self.chk_extend_include_source.toggled.connect(self._save_settings)
        if getattr(self, "spn_batch", None):
            self.spn_batch.valueChanged.connect(self._save_settings)
        self.ed_out.textChanged.connect(self._save_settings)
        if getattr(self, "chk_use_queue", None):
            self.chk_use_queue.toggled.connect(self._save_settings)
        if getattr(self, "chk_use_nvenc", None):
            self.chk_use_nvenc.toggled.connect(self._save_settings)
        # Recent results controls
        if getattr(self, "sld_thumb_size", None):
            self.sld_thumb_size.valueChanged.connect(self._save_settings)
        if getattr(self, "combo_recent_sort", None):
            self.combo_recent_sort.currentIndexChanged.connect(self._save_settings)

        # LoRA-related widgets
        if getattr(self, "ed_lora1_path", None):
            self.ed_lora1_path.textChanged.connect(self._save_settings)
        if getattr(self, "ed_lora2_path", None):
            self.ed_lora2_path.textChanged.connect(self._save_settings)
        if getattr(self, "slider_lora1_weight", None):
            self.slider_lora1_weight.valueChanged.connect(self._save_settings)
        if getattr(self, "slider_lora2_weight", None):
            self.slider_lora2_weight.valueChanged.connect(self._save_settings)
        if getattr(self, "chk_relighting_lora", None):
            self.chk_relighting_lora.toggled.connect(self._save_settings)



    def _save_settings(self):
        """Save current settings to JSON file"""
        try:
            _ensure_settings_dir()
            settings = {
                "mode": self.cmb_mode.currentText(),
                "prompt": self.ed_prompt.toPlainText(),
                "negative_prompt": self.ed_negative.toPlainText() if getattr(self, "ed_negative", None) else "",
                "image_path": self.ed_image.text(),
                "size": self.cmb_size.currentText(),
                "seed": self.spn_seed.value(),
                "random_seed": self.chk_random_seed.isChecked(),
                "steps": self.spn_steps.value(),
                "guidance_scale": self.spn_guidance.value(),
                "frames": self.spn_frames.value(),
                "fps": self.spn_fps.value(),
                "extend_auto_merge": bool(getattr(self, "chk_extend_merge", None) and self.chk_extend_merge.isChecked()),
                "extend_include_source_in_merge": bool(getattr(self, "chk_extend_include_source", None) and self.chk_extend_include_source.isChecked()),
                "batch_count": int(self.spn_batch.value()) if getattr(self, "spn_batch", None) else 1,
                "output_path": self.ed_out.text(),
                "use_queue": bool(getattr(self, "chk_use_queue", None) and self.chk_use_queue.isChecked()),
                "use_nvenc": bool(getattr(self, "chk_use_nvenc", None) and self.chk_use_nvenc.isChecked()),
                "thumb_size": int(self.sld_thumb_size.value()) if getattr(self, "sld_thumb_size", None) else 100,
                "recent_sort": (
                    self.combo_recent_sort.currentData()
                    if getattr(self, "combo_recent_sort", None) and self.combo_recent_sort.currentData()
                    else (
                        self.combo_recent_sort.currentText()
                        if getattr(self, "combo_recent_sort", None)
                        else "newest"
                    )
                ),
            }
            # Optional LoRA-related settings
            if getattr(self, "ed_lora1_path", None):
                settings["lora1_path"] = self.ed_lora1_path.text()
            if getattr(self, "slider_lora1_weight", None):
                settings["lora1_weight"] = float(self.slider_lora1_weight.value()) / 100.0
            if getattr(self, "ed_lora2_path", None):
                settings["lora2_path"] = self.ed_lora2_path.text()
            if getattr(self, "slider_lora2_weight", None):
                settings["lora2_weight"] = float(self.slider_lora2_weight.value()) / 100.0
            if getattr(self, "chk_relighting_lora", None):
                settings["use_relighting_lora"] = bool(self.chk_relighting_lora.isChecked())

            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")



    def _load_settings(self):
        """Load settings from JSON file"""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self._loaded_settings = settings
            else:
                self._loaded_settings = {}
        except Exception as e:
            print(f"Failed to load settings: {e}")
            self._loaded_settings = {}

    def _apply_loaded_settings(self):
        """Apply loaded settings to controls"""
        if not hasattr(self, "_loaded_settings"):
            return

        settings = self._loaded_settings

        # Apply basic settings if they exist
        mode = settings.get("mode")
        if mode is not None:
            self.cmb_mode.setCurrentText(mode)

        prompt = settings.get("prompt")
        if prompt is not None:
            self.ed_prompt.setPlainText(prompt)

        neg = settings.get("negative_prompt")
        if neg is not None and getattr(self, "ed_negative", None):
            self.ed_negative.setPlainText(neg)

        image_path = settings.get("image_path")
        if image_path is not None:
            self.ed_image.setText(image_path)

        size = settings.get("size")
        if size is not None:
            self.cmb_size.setCurrentText(size)

        if "seed" in settings:
            self.spn_seed.setValue(settings["seed"])
        if "random_seed" in settings:
            self.chk_random_seed.setChecked(bool(settings["random_seed"]))

        if "steps" in settings:
            self.spn_steps.setValue(settings["steps"])
            self.slider_steps.setValue(settings["steps"])

        if "guidance_scale" in settings:
            self.spn_guidance.setValue(settings["guidance_scale"])
            self.slider_guidance.setValue(settings["guidance_scale"])

        if "frames" in settings:
            self.spn_frames.setValue(settings["frames"])

        if "fps" in settings:
            self.spn_fps.setValue(settings["fps"])

        # Extend spinner intentionally does NOT persist across sessions; it always
        # starts at 0 after app restart so chains must be opted into each time.
        if "extend_auto_merge" in settings and getattr(self, "chk_extend_merge", None):
            try:
                self.chk_extend_merge.setChecked(bool(settings["extend_auto_merge"]))
            except Exception:
                pass

        if "extend_include_source_in_merge" in settings and getattr(self, "chk_extend_include_source", None):
            try:
                self.chk_extend_include_source.setChecked(bool(settings["extend_include_source_in_merge"]))
            except Exception:
                pass

        if "batch_count" in settings and getattr(self, "spn_batch", None):
            try:
                v = int(settings["batch_count"])
                if v < 1:
                    v = 1
                if v > 99:
                    v = 99
                self.spn_batch.setValue(v)
            except Exception:
                pass

        if "output_path" in settings:
            self.ed_out.setText(settings["output_path"])

        if "use_queue" in settings and getattr(self, "chk_use_queue", None):
            self.chk_use_queue.setChecked(bool(settings["use_queue"]))
        if "use_nvenc" in settings and getattr(self, "chk_use_nvenc", None):
            self.chk_use_nvenc.setChecked(bool(settings["use_nvenc"]))

        # LoRA-related settings
        if "lora1_path" in settings and getattr(self, "ed_lora1_path", None):
            self.ed_lora1_path.setText(settings.get("lora1_path") or "")
        if "lora1_weight" in settings and getattr(self, "slider_lora1_weight", None):
            try:
                w1 = float(settings["lora1_weight"])
                self.slider_lora1_weight.setValue(max(0, min(100, int(round(w1 * 100)))))
            except Exception:
                pass

        if "lora2_path" in settings and getattr(self, "ed_lora2_path", None):
            self.ed_lora2_path.setText(settings.get("lora2_path") or "")
        if "lora2_weight" in settings and getattr(self, "slider_lora2_weight", None):
            try:
                w2 = float(settings["lora2_weight"])
                self.slider_lora2_weight.setValue(max(0, min(100, int(round(w2 * 100)))))
            except Exception:
                pass

        if "use_relighting_lora" in settings and getattr(self, "chk_relighting_lora", None):
            self.chk_relighting_lora.setChecked(bool(settings["use_relighting_lora"]))


        # Recent results UI state
        try:
            if "thumb_size" in settings and getattr(self, "sld_thumb_size", None):
                val = int(settings.get("thumb_size", 100))
                if val < 20:
                    val = 20
                if val > 100:
                    val = 100
                self.sld_thumb_size.setValue(val)
        except Exception:
            pass

        try:
            sort = settings.get("recent_sort")
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
                    cb.setCurrentIndex(idx)
        except Exception:
            pass


    def _on_random_seed_toggled(self, checked: bool):
        """Handle random seed toggle"""
        self.spn_seed.setEnabled(not checked)
        if checked:
            # Enable automatic random seed generation - disable manual editing
            self.spn_seed.setStyleSheet("color: gray;")
        else:
            # Disable automatic random seed generation - enable manual editing
            self.spn_seed.setStyleSheet("color: black;")
        self._save_settings()

    def _on_seed_changed(self, value: int):
        """Handle seed value change"""
        if self.chk_random_seed.isChecked():
            # If random is checked, uncheck it when user manually changes seed
            self.chk_random_seed.setChecked(False)
            self.spn_seed.setEnabled(True)
        self._save_settings()

    def _on_video2video_toggled(self, checked: bool):
        """Show/hide video-to-video controls when the toggle changes."""
        # Show or hide the browse button and info label
        if getattr(self, "btn_video2_browse", None):
            self.btn_video2_browse.setVisible(bool(checked))
        if getattr(self, "lbl_video2_info", None):
            self.lbl_video2_info.setVisible(bool(checked))
        if getattr(self, "chk_extend_include_source", None):
            self.chk_extend_include_source.setVisible(bool(checked))

        if checked:
            # Video-to-video always runs via image2video, so switch mode for the user.
            try:
                if self.cmb_mode.currentText() != "image2video":
                    self.cmb_mode.setCurrentText("image2video")
            except Exception:
                pass
        else:
            # Clear selection when disabling video-to-video
            self._video2video_path = None
            if getattr(self, "lbl_video2_info", None):
                self.lbl_video2_info.setText("No video selected")

    def _on_video2video_browse(self):
        """Let the user pick a source video for video-to-video."""
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Choose source video",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*.*)",
        )
        if not fn:
            return
        self._video2video_path = Path(fn)

        # Update label with some info about the video
        try:
            size_mb = self._video2video_path.stat().st_size / (1024 * 1024)
            info = f"{self._video2video_path.name}  ({size_mb:.1f} MB)"
        except Exception:
            info = self._video2video_path.name
        if getattr(self, "lbl_video2_info", None):
            self.lbl_video2_info.setText(info)

        # Ensure we're in image2video mode
        try:
            if self.cmb_mode.currentText() != "image2video":
                self.cmb_mode.setCurrentText("image2video")
        except Exception:
            pass

        # Use the existing extend-chain helper to grab the last frame as start image
        try:
            frame_path = self._next_extend_frame_path()
            if not self._extract_last_frame(self._video2video_path, frame_path):
                self._append_log(f"Video2Video: failed to extract last frame from {self._video2video_path.name}.")
                return
            # Point the Start image field at the extracted frame so the normal image2video
            # pipeline (including extend/merge) continues to work unchanged.
            self.ed_image.setText(str(frame_path))
            self._append_log(f"Video2Video: using last frame from {self._video2video_path.name} as start image.")
        except Exception as e:
            self._append_log(f"Video2Video: error while preparing start image: {e}")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    # ---- Recent results helpers ----
    def _wan_outputs_dir(self) -> Path:
        """Return the folder where Wan2.2 renders its final videos.

        This is intentionally *not* the thumbnail/"last results" folder.
        Some older queue configurations pointed the generic default_outdir()
        for "wan22" at the thumbnail tree (output/last results/wan22),
        which meant the Recent Results browser was scanning the thumbs
        directory instead of the real renders directory.

        Here we correct that by:
        - asking default_outdir() for the wan22 folder (when available)
        - if that path lives under a "last results" folder, we instead
          use a sibling output/wan22 directory for the real videos
        - making sure the chosen directory exists
        """
        # Start with a sensible default under APP_ROOT/output/video/wan22
        wan_dir = APP_ROOT / "output" / "video" / "wan22"
        try:
            wan_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # If the shared queue_adapter has a custom location, prefer it
        try:
            from helpers.queue_adapter import default_outdir as _default_outdir
        except Exception:
            _default_outdir = None

        if _default_outdir is not None:
            try:
                d = Path(_default_outdir(True, "wan22"))
                # If default_outdir accidentally points into a "last results"
                # tree, hop back to the main output root instead.
                parts_lower = {p.lower() for p in d.parts}
                if "models" in parts_lower:
                    # Avoid using any path under a models directory; stick to output/video/wan22
                    pass
                elif "last results" in parts_lower or "last_results" in parts_lower:
                    # .../output/last results/wan22 -> .../output/wan22
                    try:
                        output_root = d.parent.parent  # go up from wan22/ to output/
                        candidate = output_root / "video" / "wan22"
                        candidate.mkdir(parents=True, exist_ok=True)
                        wan_dir = candidate
                    except Exception:
                        # If anything goes wrong, keep the previous wan_dir
                        pass
                else:
                    # Normal case: use the queue adapter's folder directly
                    if d.exists() or d.parent.exists():
                        wan_dir = d
            except Exception:
                # On any error, fall back to APP_ROOT/output/wan22
                pass

        return wan_dir
    def _wan_recents_dir(self) -> Path:
        """Return Path to the Wan2.2 recent-thumbnail folder, similar to txt2img recents."""
        from pathlib import Path as _Path
        try:
            try:
                base = APP_ROOT
            except Exception:
                base = _Path(__file__).resolve().parent.parent
            d = base / "output" / "last results" / "wan22"
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return d
        except Exception:
            # Fallback to a simple relative path
            return _Path("output") / "last results" / "wan22"

    def _jobs_done_dirs(self) -> list[Path]:
        """Return list of job result folders to scan for finished Wan2.2 queue jobs."""
        try:
            base = APP_ROOT
        except Exception:
            from pathlib import Path as _Path
            base = _Path(__file__).resolve().parent.parent
        dirs: list[Path] = []
        for name in ("finished", "done"):
            try:
                d = base / "jobs" / name
                if d.exists() and d.is_dir():
                    dirs.append(d)
            except Exception:
                continue
        return dirs




    def _jobs_done_dir(self) -> Path:
        """Return the jobs/done directory used by the queue system."""
        try:
            base = APP_ROOT
        except Exception:
            base = Path(__file__).resolve().parent.parent
        return base / "jobs" / "done"

    
    def _list_recent_jobs(self) -> list[Path]:
        """Return a list of recent finished Wan2.2 job JSON files (newest first)."""
        jobs: list[Path] = []
        try:
            for d in self._jobs_done_dirs():
                try:
                    for p in d.iterdir():
                        try:
                            if p.is_file() and p.suffix.lower() == ".json":
                                jobs.append(p)
                        except Exception:
                            continue
                except Exception:
                    continue
            jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return jobs[:40]
        except Exception:
            return []


    def _resolve_output_from_job(self, job_json: Path) -> tuple[Optional[Path], dict]:
        """Resolve the primary media file for a finished job JSON."""
        try:
            j = json.loads(job_json.read_text(encoding="utf-8"))
        except Exception:
            j = {}
        def _as_path(val):
            if not val:
                return None
            try:
                p = Path(str(val)).expanduser()
                if not p.is_absolute():
                    out_dir = j.get("out_dir") or (j.get("args") or {}).get("out_dir")
                    if out_dir:
                        p = Path(out_dir).expanduser() / p
                return p
            except Exception:
                return None

        # Single-path fields
        for k in ("produced", "outfile", "output", "result", "file", "path"):
            try:
                v = j.get(k) or (j.get("args") or {}).get(k)
            except Exception:
                v = None
            p = _as_path(v)
            if p and p.exists() and p.is_file():
                return p, j

        # List fields
        for k in ("outputs", "produced_files", "results", "files", "artifacts", "saved"):
            try:
                seq = j.get(k) or (j.get("args") or {}).get(k)
            except Exception:
                seq = None
            if isinstance(seq, (list, tuple)):
                for v in seq:
                    p = _as_path(v)
                    if p and p.exists() and p.is_file():
                        return p, j

        # Fallback: newest media in out_dir
        media_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        out_dir = None
        try:
            out_dir = j.get("out_dir") or (j.get("args") or {}).get("out_dir")
        except Exception:
            out_dir = None
        out_dir_path = _as_path(out_dir)
        try:
            if out_dir_path and out_dir_path.exists():
                cand = [p for p in out_dir_path.iterdir() if p.is_file() and p.suffix.lower() in media_exts]
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if cand:
                    return cand[0], j
        except Exception:
            pass

        return None, j

    
    def _iter_recent_videos(self, limit: int = 40) -> list[Path]:
        """Collect recent Wan2.2 video outputs from finished queue jobs and wan22 output folder.

        This mirrors the txt2img 'last results' behavior but for Wan2.2 videos:
        - Any finished job whose result looks like a video under the Wan2.2 output tree
        - Any video file found directly under the Wan2.2 output dir (and subfolders)
        """
        from pathlib import Path as _Path

        videos: list[Path] = []
        seen: set[str] = set()

        # Determine wan22 base output dir
        try:
            wan_dir = self._wan_outputs_dir().resolve()
        except Exception:
            wan_dir = None

        media_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

        # 1) From finished jobs
        try:
            jobs = self._list_recent_jobs()
        except Exception:
            jobs = []
        for job_json in jobs:
            try:
                media_path, j = self._resolve_output_from_job(job_json)
            except Exception:
                media_path, j = None, {}
            if not media_path:
                continue
            try:
                media_path = _Path(str(media_path))
            except Exception:
                continue
            try:
                if not (media_path.exists() and media_path.is_file()):
                    continue
            except Exception:
                continue
            ext = media_path.suffix.lower()
            if ext not in media_exts:
                continue

            # Only accept jobs that look like Wan2.2 runs (category/tool) or live under wan_dir
            accept = False
            try:
                cat = str(
                    (j.get("category") or j.get("tool") or j.get("name") or "")
                ).lower()
            except Exception:
                cat = ""
            if "wan" in cat or "wan22" in cat:
                accept = True
            elif wan_dir is not None:
                try:
                    parent = media_path.parent.resolve()
                    if parent == wan_dir or wan_dir in parent.parents:
                        accept = True
                except Exception:
                    pass
            if not accept:
                continue

            key = str(media_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            videos.append(media_path)
            if len(videos) >= limit:
                break

        # 2) Fallback: scan wan22 output directory directly (non-queued runs)
        try:
            if wan_dir is not None and wan_dir.exists():
                # Walk at shallow depth first (top-level) then subfolders
                for root, dirs, files in __import__("os").walk(wan_dir):
                    try:
                        rpath = _Path(root)
                    except Exception:
                        continue
                    for name in files:
                        try:
                            p = rpath / name
                            if not p.is_file():
                                continue
                            ext = p.suffix.lower()
                            if ext not in media_exts:
                                continue
                            key = str(p.resolve())
                            if key in seen:
                                continue
                            seen.add(key)
                            videos.append(p)
                            if len(videos) >= limit:
                                break
                        except Exception:
                            continue
                    if len(videos) >= limit:
                        break
        except Exception:
            pass

        # Final sort by mtime (newest first)
        try:
            videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass

        if limit is not None and limit > 0:
            return videos[:limit]
        return videos



    def _rounded_with_border(self, pix: QPixmap, radius: int = 9) -> QPixmap:
        if pix.isNull(): return pix
        w,h = pix.width(), pix.height()
        out = QPixmap(w,h); out.fill(Qt.transparent)
        p = QPainter(out); p.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath(); path.addRoundedRect(0,0,w,h, radius, radius)
        p.setClipPath(path); p.drawPixmap(0,0,pix); p.setClipping(False)
        pen = QPen(QColor(0,0,0,60)); pen.setWidth(1); p.setPen(pen)
        p.drawRoundedRect(0.5,0.5,w-1,h-1, radius, radius); p.end()
        return out

    def _extract_first_frame(self, video: Path, dest: Path) -> bool:
        ff = _ffmpeg_exe()
        if not ff: return False
        try:
            import subprocess
            cmd = [ff, "-y", "-hide_banner", "-i", str(video), "-frames:v", "1", str(dest)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return dest.exists()
        except Exception:
            return False

    
    def _extract_last_frame(self, video: Path, dest: Path) -> bool:
        """Extract the last frame of a video to ``dest`` using ffmpeg.
        This is used for the extend-chain feature.
        """
        ff = _ffmpeg_exe()
        if not ff:
            return False
        try:
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-sseof",
                "-0.05",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(dest),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return dest.exists()
        except Exception:
            return False

    def _next_extend_frame_path(self) -> Path:
        """Return a unique Path under EXTEND_FRAMES_DIR for the next last-frame snapshot.
        The filename encodes a small prompt snippet plus date/time and an index, so that
        frames from previous jobs are never accidentally reused.
        """
        try:
            EXTEND_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Build a short, filesystem-safe prefix from the current prompt (a few words).
        try:
            raw_prompt = self.ed_prompt.toPlainText()
        except Exception:
            raw_prompt = ""
        raw_prompt = (raw_prompt or "").strip()
        words = raw_prompt.split()
        safe_words = []
        for w in words[:4]:
            # Keep only alphanumeric characters, replace others with nothing
            clean = "".join(ch for ch in w if ch.isalnum())
            if clean:
                safe_words.append(clean)
        if not safe_words:
            safe_words.append("wan22")
        prefix = "_".join(safe_words)

        # Timestamp component to make this job unique.
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts = "time"

        # Per-session index so multiple segments in one extend-chain remain ordered.
        try:
            self._extend_frame_index += 1
        except Exception:
            self._extend_frame_index = 1
        idx = self._extend_frame_index

        name = f"{prefix}_{ts}_{idx:02d}.png"
        return EXTEND_FRAMES_DIR / name

    def _get_thumb_pixmap(self, video: Path, size: int = 180) -> QPixmap:
        """Return a cached QPixmap thumbnail for a Wan2.2 video.

        Thumbnails are stored under output/last results/wan22 so they persist across sessions,
        similar to txt2img 'last results' thumbnails.
        """
        try:
            from pathlib import Path as _Path
            import hashlib

            # In-memory cache
            thumbs = self._thumb_cache if hasattr(self, "_thumb_cache") else {}
            self._thumb_cache = thumbs
            key = f"{str(video)}::{size}"
            if key in thumbs:
                return thumbs[key]

            try:
                vpath = _Path(str(video))
            except Exception:
                vpath = None

            thumb_path = None
            if vpath is not None:
                try:
                    # Stable name from full path + size
                    h = hashlib.sha1(str(vpath).encode("utf-8")).hexdigest()[:8]
                    rec_dir = self._wan_recents_dir()
                    try:
                        rec_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    thumb_path = rec_dir / f"{vpath.stem}_{h}_{size}.jpg"
                except Exception:
                    thumb_path = None

            if thumb_path is not None and vpath is not None:
                # Re-generate thumbnail if missing or older than the video
                try:
                    need_regen = True
                    if thumb_path.exists() and vpath.exists():
                        try:
                            if thumb_path.stat().st_mtime >= vpath.stat().st_mtime:
                                need_regen = False
                        except Exception:
                            pass
                    if need_regen and vpath.exists():
                        thumb_path.parent.mkdir(parents=True, exist_ok=True)
                        ok = self._extract_first_frame(vpath, thumb_path)
                        if not ok:
                            thumb_path = None
                except Exception:
                    thumb_path = None

            pm = QPixmap(str(thumb_path)) if (thumb_path is not None and thumb_path.exists()) else QPixmap()
            if not pm.isNull():
                # aspect crop to square then scale
                w, h = pm.width(), pm.height()
                if w != h:
                    side = min(w, h)
                    x = (w - side) // 2
                    y = (h - side) // 2
                    pm = pm.copy(x, y, side, side)
                pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = self._rounded_with_border(pm)
            thumbs[key] = pm
            return pm
        except Exception:
            return QPixmap()


    def _open_in_player(self, p: Path):
        """Try to open the Wan2.2 result in the app's internal media player first.

        Falls back to the older open_video hook and finally the system player
        if the internal player is not available.
        """
        # 1) Preferred: internal media player (same as Interp/RIFE tabs)
        try:
            player = getattr(self.main, "video", None)
            if player and hasattr(player, "open"):
                try:
                    player.open(p)
                except TypeError:
                    # some builds expect a string path
                    player.open(str(p))
                try:
                    # keep main.current_path in sync with the media player
                    self.main.current_path = Path(str(p))
                except Exception:
                    pass
                try:
                    # refresh sidebar/media info if available
                    refresh_info_now(p)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # 2) Legacy hook for older builds
        try:
            if hasattr(self.main, "open_video"):
                self.main.open_video(str(p))
                return
        except Exception:
            pass

        # 3) Final fallback: system default player
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception:
            pass

    def _on_recent_open(self):
        folder = None
        try:
            videos = self._iter_recent_videos(limit=1)
        except Exception:
            videos = []
        if videos:
            try:
                folder = videos[0].parent
            except Exception:
                folder = None
        if folder is None:
            try:
                folder = self._wan_outputs_dir()
            except Exception:
                folder = None
        if folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


    def _on_play_last(self):
        """Play the last generated Wan2.2 video in the main player (if available) or the system default player."""
        try:
            videos = self._iter_recent_videos(limit=1)
        except Exception:
            videos = []
        if not videos:
            QMessageBox.information(self, "Wan 2.2", "No generated videos found yet.")
            return
        p = videos[0]
        try:
            if not p.exists():
                raise FileNotFoundError
        except Exception:
            QMessageBox.warning(self, "Wan 2.2", f"Last video not found:\n{p}")
            return
        self._open_in_player(p)

    def _on_thumb_size_changed(self, value: int):
        """Adjust recent-results thumbnail display size without regenerating thumbnails."""
        try:
            # Slider gives 20–100, treat as percentage of base size
            value = max(20, min(100, int(value)))
            base = getattr(self, "_thumb_base_size", 180)
            display_size = max(20, int(base * (value / 100.0)))
            self._thumb_display_size = display_size
            if hasattr(self, "lbl_thumb_size_value"):
                self.lbl_thumb_size_value.setText(f"{value}%")
            buttons = getattr(self, "_recent_thumb_buttons", [])
            if not buttons:
                return
            btn_side = max(display_size + 16, 40)
            for btn in buttons:
                try:
                    btn.setFixedSize(btn_side, btn_side)
                    btn.setIconSize(QSize(display_size, display_size))
                except Exception:
                    continue

            # Ask the recent-results layout (FlowLayout) to re-wrap thumbnails
            lay = getattr(self, "_recent_layout", None)
            if hasattr(lay, "relayout"):
                try:
                    lay.relayout()
                except Exception:
                    pass
        except Exception:
            pass

    def _refresh_recent(self):
        lay = getattr(self, "_recent_layout", None)
        if lay is None:
            return
        lay.clear()
        base_size = getattr(self, "_thumb_base_size", 180)
        display_size = getattr(self, "_thumb_display_size", base_size)
        btn_side = max(display_size + 16, 40)
        self._recent_thumb_buttons = []
        try:
            videos = self._iter_recent_videos(limit=40)
        except Exception:
            videos = []

        # Apply sort mode from combo_recent_sort (UI-only; Play last uses newest-first from the backend)
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

        try:
            from pathlib import Path as _P

            def _mtime_for(p):
                try:
                    return _P(str(p)).stat().st_mtime
                except Exception:
                    return 0

            def _name_for(p):
                try:
                    return _P(str(p)).name.lower()
                except Exception:
                    return str(p)

            def _size_for(p):
                try:
                    return _P(str(p)).stat().st_size
                except Exception:
                    return 0

            if mode in ("newest", "oldest"):
                videos.sort(key=_mtime_for, reverse=(mode == "newest"))
            elif mode in ("az", "za"):
                videos.sort(key=_name_for, reverse=(mode == "za"))
            elif mode in ("size_small", "size_large"):
                videos.sort(key=_size_for, reverse=(mode == "size_large"))
        except Exception:
            # If anything goes wrong, keep the original newest-first order from _iter_recent_videos
            pass
        if not videos:
            lbl = QLabel("No results yet. Run Wan2.2 (or queue jobs) to see your recent videos here.")
            lbl.setStyleSheet("color:#888;")
            lay.addWidget(lbl)
            return

        for p in videos:
            btn = QPushButton()
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setFixedSize(btn_side, btn_side)
            pm = self._get_thumb_pixmap(p, size=base_size)
            if not pm.isNull():
                btn.setIcon(QIcon(pm))
                btn.setIconSize(QSize(display_size, display_size))
            btn.setToolTip(p.name)

            # Left click: open in player
            btn.clicked.connect(lambda _=False, path=p: self._open_in_player(path))

            # Right click: context menu for recent result
            try:
                btn.setContextMenuPolicy(Qt.CustomContextMenu)
                btn.customContextMenuRequested.connect(
                    lambda pos, path=p, button=btn: self._show_recent_context_menu(button, path, pos)
                )
            except Exception:
                pass

            lay.addWidget(btn)
            self._recent_thumb_buttons.append(btn)


    def _show_recent_context_menu(self, button, path: Path, pos):
        """Show right-click menu for a recent Wan2.2 result."""
        try:
            menu = QMenu(self)
        except Exception:
            return
        act_source = menu.addAction("Use as source for video to video")
        act_info = menu.addAction("Info")
        act_open = menu.addAction("Open folder")
        act_rename = menu.addAction("Rename")
        menu.addSeparator()
        act_delete = menu.addAction("Delete from disk & recents")

        global_pos = button.mapToGlobal(pos)
        action = menu.exec(global_pos)
        if not action:
            return
        if action == act_source:
            self._use_recent_as_v2v_source(path)
        elif action == act_info:
            self._show_recent_info(path)
        elif action == act_open:
            self._open_recent_folder(path)
        elif action == act_rename:
            self._rename_recent_on_disk(path)
        elif action == act_delete:
            self._delete_recent_from_disk(path)

    def _use_recent_as_v2v_source(self, path: Path):
        """Treat the chosen recent video as the source for video-to-video."""
        try:
            p = Path(path)
        except Exception:
            return
        if not p.exists():
            try:
                QMessageBox.warning(self, "Wan 2.2", "File no longer exists on disk.")
            except Exception:
                pass
            try:
                self._refresh_recent()
            except Exception:
                pass
            return

        # Enable video-to-video mode so the UI reflects the choice
        try:
            if getattr(self, "chk_video2video", None) and not self.chk_video2video.isChecked():
                self.chk_video2video.setChecked(True)
        except Exception:
            pass

        # Remember the source video
        self._video2video_path = p

        # Update label with some info about the video
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            info = f"{p.name}  ({size_mb:.1f} MB)"
        except Exception:
            info = p.name
        if getattr(self, "lbl_video2_info", None):
            self.lbl_video2_info.setText(info)

        # Ensure we're in image2video mode
        try:
            if self.cmb_mode.currentText() != "image2video":
                self.cmb_mode.setCurrentText("image2video")
        except Exception:
            pass

        # Use the existing extend helpers to grab the last frame as the start image
        try:
            frame_path = self._next_extend_frame_path()
            if not self._extract_last_frame(p, frame_path):
                self._append_log(f"Video2Video: failed to extract last frame from {p.name}.")
                return
            self.ed_image.setText(str(frame_path))
            self._append_log(f"Video2Video: using last frame from {p.name} as start image.")
        except Exception as e:
            self._append_log(f"Video2Video: error while preparing start image from recent: {e}")

    def _show_recent_info(self, path: Path):
        """Show basic file info for a recent Wan2.2 result."""
        try:
            p = Path(path)
        except Exception:
            return
        if not p.exists():
            try:
                QMessageBox.information(self, "Wan 2.2", "File no longer exists on disk.")
            except Exception:
                pass
            try:
                self._refresh_recent()
            except Exception:
                pass
            return

        # Update sidebar/media info if available
        try:
            refresh_info_now(p)
        except Exception:
            pass

        # Build a small info message
        lines = [str(p)]
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            lines.append(f"Size: {size_mb:.1f} MB")
        except Exception:
            pass
        try:
            mtime = p.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            lines.append(f"Modified: {dt}")
        except Exception:
            pass
        msg = "\n".join(lines)
        try:
            QMessageBox.information(self, "Wan 2.2 – Video info", msg)
        except Exception:
            pass


    def _open_recent_folder(self, path: Path):
        """Open the folder containing a recent Wan2.2 result in the system file browser."""
        try:
            p = Path(path)
        except Exception:
            return
        if not p.exists():
            try:
                QMessageBox.information(self, "Wan 2.2", "File no longer exists on disk.")
            except Exception:
                pass
            try:
                self._refresh_recent()
            except Exception:
                pass
            return
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
        except Exception as e:
            try:
                self._append_log(f"Open folder failed for {p}: {e}")
            except Exception:
                pass

    def _rename_recent_on_disk(self, path: Path):
        """Rename a recent Wan2.2 result on disk and refresh recents."""
        try:
            p = Path(path)
        except Exception:
            return
        if not p.exists():
            try:
                QMessageBox.information(self, "Wan 2.2", "File no longer exists on disk.")
            except Exception:
                pass
            try:
                self._refresh_recent()
            except Exception:
                pass
            return

        # Ask the user for a new file name
        try:
            new_name, ok = QInputDialog.getText(
                self,
                "Rename video",
                "New file name:",
                QLineEdit.Normal,
                p.name,
            )
        except Exception:
            ok = False
            new_name = ""
        if not ok:
            return
        new_name = str(new_name).strip()
        if not new_name:
            return

        # Preserve extension if user omitted it
        try:
            stem, suffix = p.stem, p.suffix
        except Exception:
            stem, suffix = p.name, ""
        if not os.path.splitext(new_name)[1] and suffix:
            new_name = new_name + suffix

        new_path = p.with_name(new_name)

        # Avoid overwriting existing files
        if new_path.exists():
            try:
                QMessageBox.warning(
                    self,
                    "Rename failed",
                    f"Cannot rename to {new_name}: file already exists.",
                )
            except Exception:
                pass
            return

        # Attempt the rename
        try:
            p.rename(new_path)
            self._append_log(f"Renamed video: {p} -> {new_path}")
        except Exception as e:
            try:
                QMessageBox.warning(
                    self,
                    "Rename failed",
                    f"Could not rename file:\n{p}\n\nto:\n{new_path}\n\n{e}",
                )
            except Exception:
                pass
            return

        # If the main player was pointing at this path, update it
        try:
            if getattr(self, "main", None) is not None and hasattr(self.main, "current_path"):
                cur = self.main.current_path
                if cur:
                    try:
                        cur_path = Path(str(cur)).resolve()
                        if cur_path == p.resolve():
                            self.main.current_path = str(new_path)
                    except Exception:
                        pass
        except Exception:
            pass

        # Refresh recent list after rename
        try:
            self._refresh_recent()
        except Exception:
            pass

    def _delete_recent_from_disk(self, path: Path):
        """Delete a recent Wan2.2 result from disk and refresh recents.

        If the video is currently playing in the internal player, attempt to
        stop it first so the file handle is released before deletion.
        """
        try:
            p = Path(path)
        except Exception:
            return
        if not p.exists():
            try:
                self._refresh_recent()
            except Exception:
                pass
            return

        # Confirm deletion with the user
        try:
            resp = QMessageBox.question(
                self,
                "Delete video",
                f"Delete this video from disk and Recent results?\n\n{p}",
            )
        except Exception:
            resp = QMessageBox.Yes
        if resp != QMessageBox.Yes:
            return

        # Try to stop playback if the internal player is using this file
        player = getattr(self.main, "video", None) if getattr(self, "main", None) is not None else None
        if player is not None:
            try:
                # Best-effort: stop or pause playback, but do NOT close the player widget
                # Closing the widget would remove the media player area from the layout
                # until the app is restarted.
                if hasattr(player, "stop"):
                    player.stop()
                elif hasattr(player, "pause"):
                    player.pause()
            except Exception:
                pass
        # Clear current_path if it matches this file
        try:
            cur = getattr(self.main, "current_path", None)
            if cur is not None:
                try:
                    cur_path = Path(str(cur)).resolve()
                    if cur_path == p.resolve():
                        self.main.current_path = None
                except Exception:
                    pass
        except Exception:
            pass

        # Delete file from disk
        try:
            p.unlink()
            self._append_log(f"Deleted video: {p}")
        except Exception as e:
            try:
                QMessageBox.warning(
                    self,
                    "Delete failed",
                    f"Could not delete file:\n{p}\n\n{e}",
                )
            except Exception:
                pass
            return

        # Refresh recent list after deletion
        try:
            self._refresh_recent()
        except Exception:
            pass

    def _update_mode(self):
        is_img = self.cmb_mode.currentText() == "image2video"
        self.ed_image.setEnabled(is_img)

        # Batch is only used in text2video mode
        show_batch = not is_img
        if hasattr(self, "lbl_batch"):
            self.lbl_batch.setVisible(show_batch)
        if hasattr(self, "spn_batch"):
            self.spn_batch.setVisible(show_batch)

        # Extend controls work in both modes, so they stay visible
        show_extend = True
        if hasattr(self, "lbl_extend"):
            self.lbl_extend.setVisible(show_extend)
        if hasattr(self, "spn_extend"):
            self.spn_extend.setVisible(show_extend)
        if hasattr(self, "chk_extend_merge"):
            self.chk_extend_merge.setVisible(show_extend)

        # Image batch button is only visible in image2video mode
        if hasattr(self, "btn_img_batch"):
            self.btn_img_batch.setVisible(is_img)

        # When running image2video (used by txt2img pipeline), hide manual output path
        show_output = not is_img
        if hasattr(self, "lbl_out"):
            self.lbl_out.setVisible(show_output)
        if hasattr(self, "out_widget"):
            self.out_widget.setVisible(show_output)
        if is_img:
            # Clear any previous manual output so Wan can auto-name the file
            self.ed_out.clear()

    def _append_log(self, text: str):
        self.log.appendPlainText(text)

    def _python_exe(self) -> str:
        return _wan_python_exe()

    def _model_root(self) -> Path:
        return _wan_model_root()

    def _generate_script(self) -> Path:
        return self._model_root() / "generate.py"

    def _detect_wan_lora_capabilities(self, py_exe: str, gen_path: Path, model_root: Path):
        """
        Inspect `generate.py --help` once and cache whether it knows about any
        LoRA-related command-line switches. This prevents us from blindly
        passing unsupported flags like --lora_dir and breaking runs on
        vanilla Wan2.2 installs.
        """
        if hasattr(self, "_wan_lora_caps"):
            return self._wan_lora_caps

        caps = {
            "has_lora_dir": False,
            "has_lora_path": False,
            "has_lora_scale": False,
            "has_relighting_flag": False,
        }
        try:
            proc = subprocess.run(
                [py_exe, str(gen_path), "--help"],
                cwd=str(model_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            caps["has_lora_dir"] = "--lora_dir" in help_text
            caps["has_lora_path"] = "--lora_path" in help_text
            caps["has_lora_scale"] = "--lora_scale" in help_text
            caps["has_relighting_flag"] = "--use_relighting_lora" in help_text
        except Exception as e:
            # Don't crash if detection fails; just assume no LoRA support.
            print(f"Wan2.2 LoRA CLI detection failed: {e}")

        self._wan_lora_caps = caps
        return caps



    def _detect_wan_negative_flag(self, py_exe: str, gen_path: Path, model_root: Path):
        """
        Inspect `generate.py --help` once and cache a supported negative prompt flag,
        if any (for example --negative_prompt).
        """
        if hasattr(self, "_wan_negative_flag"):
            return self._wan_negative_flag

        neg_flag = None
        try:
            proc = subprocess.run(
                [py_exe, str(gen_path), "--help"],
                cwd=str(model_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for flag in ("--negative_prompt", "--negative_prompt_text", "--negative"):
                if flag in help_text:
                    neg_flag = flag
                    break
        except Exception as e:
            # Don't crash if detection fails; just assume no negative-prompt support.
            print(f"Wan2.2 negative prompt CLI detection failed: {e}")

        self._wan_negative_flag = neg_flag
        return neg_flag



    def _build_command(self):
        """
        Build the python + args + working directory tuple for QProcess.

        We deliberately do NOT pass --t5_cpu or --offload_model so that WAN
        can fully use the GPU by default. If VRAM OOMs, we can revisit this
        and add switches in the UI.
        """
        py = self._python_exe()
        gen = self._generate_script()
        model_root = self._model_root()

        if not gen.exists():
            raise RuntimeError(
                f"generate.py not found at {gen}.\n"
                f"Make sure the Wan2.2 repo is unpacked into {model_root}."
            )

        # Use the selected size format directly
        size_str = self.cmb_size.currentText()

        mode = self.cmb_mode.currentText()
        prompt = self.ed_prompt.toPlainText().strip()
        negative = self.ed_negative.toPlainText().strip() if getattr(self, "ed_negative", None) else ""
        image = self.ed_image.text().strip()

        # LoRA / relighting options from the UI (optional)
        lora1_path = ""
        lora2_path = ""
        use_relighting = False
        if getattr(self, "ed_lora1_path", None):
            lora1_path = self.ed_lora1_path.text().strip()
        if getattr(self, "ed_lora2_path", None):
            lora2_path = self.ed_lora2_path.text().strip()
        if getattr(self, "chk_relighting_lora", None):
            use_relighting = self.chk_relighting_lora.isChecked()

        # Prepare LoRA capability detection only if needed
        lora_caps = None
        if lora1_path or lora2_path or use_relighting:
            lora_caps = self._detect_wan_lora_capabilities(py, gen, model_root)

        # Get seed (use random if random_seed is enabled)
        seed_value = self.spn_seed.value()
        if self.chk_random_seed.isChecked():
            seed_value = random.randint(0, 2147483647)

        args = [
            str(gen),
            "--task", "ti2v-5B",
            "--size", size_str,
            "--sample_steps", str(self.spn_steps.value()),
            "--sample_guide_scale", str(self.spn_guidance.value()),
            "--base_seed", str(seed_value),
            "--frame_num", str(self.spn_frames.value()),
            "--ckpt_dir", str(model_root),
            "--convert_model_dtype",
        ]

        if prompt:
            args += ["--prompt", prompt]

        # Negative prompt (best-effort; only if generate.py advertises a matching flag)
        if negative:
            neg_flag = self._detect_wan_negative_flag(py, gen, model_root)
            if neg_flag:
                args += [neg_flag, negative]
            else:
                self._append_log(
                    "Negative prompt is set, but your Wan2.2 generate.py does not "
                    "advertise a known negative-prompt flag in --help; skipping."
                )

        if mode == "image2video":
            if not image:
                raise RuntimeError("Start image is required for image2video mode.")
            args += ["--image", image]

        # LoRA CLI integration (best-effort; only if generate.py supports it)
        if lora1_path or lora2_path or use_relighting:
            caps = lora_caps or {}
            has_any_lora_flag = any(
                caps.get(k, False)
                for k in ("has_lora_dir", "has_lora_path", "has_relighting_flag")
            )

            if not has_any_lora_flag:
                self._append_log(
                    "LoRA options are set, but your Wan2.2 generate.py does not "
                    "advertise any LoRA-related command-line flags. Running without LoRA."
                )
            else:
                # Only LoRA 1 is wired through to the CLI for now
                if lora1_path:
                    if caps.get("has_lora_dir"):
                        lora_dir = lora1_path
                        # If a file was selected, pass its folder as lora_dir
                        if os.path.isfile(lora_dir):
                            lora_dir = os.path.dirname(lora_dir)
                        args += ["--lora_dir", lora_dir]
                        self._append_log(f"Using Wan2.2 LoRA directory: {lora_dir}")
                    elif caps.get("has_lora_path"):
                        args += ["--lora_path", lora1_path]
                        if caps.get("has_lora_scale") and getattr(self, "slider_lora1_weight", None):
                            scale = max(
                                0.0,
                                min(1.0, float(self.slider_lora1_weight.value()) / 100.0),
                            )
                            args += ["--lora_scale", f"{scale:.3f}"]
                        self._append_log(f"Using Wan2.2 LoRA file: {lora1_path}")

                if lora2_path:
                    self._append_log(
                        "Note: LoRA 2 is configured in the UI, but Wan2.2's CLI "
                        "currently only accepts a single LoRA. Only LoRA 1 is "
                        "passed through for now."
                    )

                if use_relighting:
                    if caps.get("has_relighting_flag"):
                        args.append("--use_relighting_lora")
                        self._append_log("Enabling Wan-Animate relighting LoRA.")
                    else:
                        self._append_log(
                            "Relighting LoRA requested, but --use_relighting_lora is "
                            "not supported by your generate.py; skipping."
                        )

        # Determine final output file path. Always resolve to either the user-selected
        # file or a sensible default under the Wan2.2 output directory so that
        # videos never end up in the models/wan22 folder.
        wan_dir = self._wan_outputs_dir()
        out_hint = self.ed_out.text().strip()

        if out_hint:
            # Respect the user-selected filename, but ensure it is an absolute path
            # and lives in a real filesystem folder rather than under models/wan22
            out_path = Path(out_hint).expanduser()
            if not out_path.is_absolute():
                out_path = wan_dir / out_path
            if not str(out_path).lower().endswith(".mp4"):
                out_path = out_path.with_suffix(".mp4")
        else:
            # Auto-name: always place auto-named results under the Wan2.2 output folder
            # (APP_ROOT/output/video/wan22 or the custom default_outdir() location).
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception:
                timestamp = "auto"
            out_path = wan_dir / f"wan22_{timestamp}_{seed_value}.mp4"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Remember this absolute output path so that extend-chains and
        # other helpers can reliably use the exact video that was just
        # rendered, even when the output field is hidden in the UI.
        try:
            self._last_run_out_path = out_path
        except Exception:
            self._last_run_out_path = None

        args += ["--save_file", str(out_path)]

        return py, args, str(model_root)



    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------
    
    def _find_main_with_video(self):
        """Best-effort search for the main window that owns the .video player."""
        try:
            p = self.parent()
            while p is not None:
                if hasattr(p, "video"):
                    return p
                try:
                    p = p.parent()
                except Exception:
                    break
        except Exception:
            pass
        try:
            for w in QApplication.topLevelWidgets():
                if hasattr(w, "video"):
                    return w
        except Exception:
            pass
        return None

    def _grab_current_qimage(self):
        """Grab a QImage for the currently visible frame/image, mirroring Ask popup logic."""
        try:
            main = self._find_main_with_video()
            if main is None:
                return None
            video = getattr(main, "video", None)
            if video is None:
                return None

            # 1) Prefer a direct currentFrame QImage
            img = getattr(video, "currentFrame", None)
            if isinstance(img, QImage) and not img.isNull():
                return img

            # 2) Try video.label.pixmap()
            try:
                label = getattr(video, "label", None)
                if label is not None and hasattr(label, "pixmap"):
                    pm = label.pixmap()
                    if pm is not None and not pm.isNull():
                        return pm.toImage()
            except Exception:
                pass

            # 3) Fallback: any reasonably sized QLabel pixmap in the main window
            try:
                from PySide6.QtWidgets import QLabel as _QLabel
                labels = main.findChildren(_QLabel)
                for lb in reversed(labels):
                    if hasattr(lb, "pixmap"):
                        pm = lb.pixmap()
                        if pm is not None and not pm.isNull() and pm.width() > 32 and pm.height() > 32:
                            return pm.toImage()
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _on_use_current(self):
        """Switch to image2video and pull the current Media Player frame in as the start image."""
        qimg = self._grab_current_qimage()
        if qimg is None or qimg.isNull():
            QMessageBox.warning(
                self,
                "No current frame",
                "No current frame or image was found.\n\n"
                "Load an image or pause a video in the Media Player first."
            )
            return

        # Save the frame to a temporary PNG and wire it into image2video.
        try:
            import time as _time
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp_inp = TEMP_DIR / f"wan22_current_{int(_time.time())}.png"
            if not qimg.save(str(tmp_inp), "PNG"):
                raise RuntimeError("Failed to save temporary frame image.")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Temporary file error",
                f"Could not create a temporary image file:\n{e}",
            )
            return

        # Switch mode and set the start image path.
        self.cmb_mode.setCurrentText("image2video")
        self.ed_image.setText(str(tmp_inp))
        self._append_log(f"Using Media Player current frame as start image:\n{tmp_inp}")

    def _show_image_batch_dialog(self):
        """Popup for image2video batch operations (files/folder or repeat current image)."""
        if self.cmb_mode.currentText() != "image2video":
            QMessageBox.warning(self, "Wan 2.2", "Image batch is only available in image2video mode.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Image2Video batch")

        layout = QVBoxLayout(dlg)

        # Main options
        opt_multi = QRadioButton("Batch multiple images")
        opt_repeat = QRadioButton("Repeat current image")
        opt_multi.setChecked(True)

        main_group = QButtonGroup(dlg)
        main_group.addButton(opt_multi, 1)
        main_group.addButton(opt_repeat, 2)

        layout.addWidget(opt_multi)

        # Sub-options for "Batch multiple images"
        sub_row = QHBoxLayout()
        sub_row.addSpacing(24)
        rb_files = QRadioButton("Files")
        rb_folder = QRadioButton("Folder")
        rb_files.setChecked(True)

        sub_group = QButtonGroup(dlg)
        sub_group.addButton(rb_files, 1)
        sub_group.addButton(rb_folder, 2)

        sub_row.addWidget(rb_files)
        sub_row.addWidget(rb_folder)
        sub_row.addStretch(1)
        layout.addLayout(sub_row)

        # Option 2: repeat current image N times
        repeat_row = QHBoxLayout()
        repeat_row.addWidget(opt_repeat)
        repeat_row.addSpacing(24)
        lbl_repeat = QLabel("Copies:")
        spn_repeat = QSpinBox()
        spn_repeat.setRange(2, 99)
        spn_repeat.setValue(4)
        spn_repeat.setToolTip("How many videos to enqueue for the current image.")
        repeat_row.addWidget(lbl_repeat)
        repeat_row.addWidget(spn_repeat)
        repeat_row.addStretch(1)
        layout.addLayout(repeat_row)

        def _update_enabled():
            multi = opt_multi.isChecked()
            rb_files.setEnabled(multi)
            rb_folder.setEnabled(multi)
            lbl_repeat.setEnabled(opt_repeat.isChecked())
            spn_repeat.setEnabled(opt_repeat.isChecked())

        opt_multi.toggled.connect(_update_enabled)
        opt_repeat.toggled.connect(_update_enabled)
        _update_enabled()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return

        if opt_multi.isChecked():
            mode = "files" if rb_files.isChecked() else "folder"
            self._run_image_batch_multi(mode)
        else:
            count = spn_repeat.value()
            self._run_image_batch_repeat(count)

    def _run_image_batch_multi(self, mode: str):
        """Handle 'Batch multiple images' for image2video mode."""
        image_paths = []

        if mode == "files":
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select images for batch",
                "",
                "Images (*.png *.jpg *.jpeg *.webp);;All files (*.*)",
            )
            image_paths = [f for f in files if os.path.isfile(f)]
        else:  # folder
            folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
            if folder:
                exts = {".png", ".jpg", ".jpeg", ".webp"}
                for name in sorted(os.listdir(folder)):
                    p = os.path.join(folder, name)
                    if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
                        image_paths.append(p)

        if not image_paths:
            QMessageBox.information(self, "Wan 2.2", "No images selected for batch.")
            return

        self._enqueue_image_batch_jobs(image_paths)

    def _run_image_batch_repeat(self, count: int):
        """Handle 'repeat current image N times' for image2video mode."""
        image_path = self.ed_image.text().strip()
        if not image_path:
            QMessageBox.warning(
                self,
                "Wan 2.2",
                "No start image selected. Set 'Start image' first, then run batch.",
            )
            return

        image_paths = [image_path] * max(2, int(count))
        self._enqueue_image_batch_jobs(image_paths)

    def _enqueue_image_batch_jobs(self, image_paths):
        """Enqueue one image2video Wan2.2 job per image path, always using the queue."""
        if not image_paths:
            return

        self.log.clear()
        self._append_log(
            f"Image2video batch: queuing {len(image_paths)} job(s) in background worker…"
        )
        self._append_log("Note: Batch always uses the queue, even if 'Use queue' is turned off.")

        try:
            try:
                from helpers.queue_adapter import enqueue_wan22_from_widget
            except Exception:
                try:
                    import queue_adapter as _qa  # type: ignore
                    enqueue_wan22_from_widget = _qa.enqueue_wan22_from_widget  # type: ignore
                except Exception as e2:
                    raise RuntimeError(f"Queue adapter not available: {e2}")

            original_mode = self.cmb_mode.currentText()
            original_image = self.ed_image.text()

            ok = True
            # Force image2video mode while enqueuing jobs; restore afterwards.
            self.cmb_mode.setCurrentText("image2video")

            for img in image_paths:
                self.ed_image.setText(str(img))
                if not enqueue_wan22_from_widget(self):
                    ok = False
                    break

        except Exception as e:
            self._append_log(f"Image2video batch enqueue failed: {e}")
            ok = False
        finally:
            try:
                self.cmb_mode.setCurrentText(original_mode)
            except Exception:
                pass
            try:
                self.ed_image.setText(original_image)
            except Exception:
                pass

        if ok:
            try:
                self._save_settings()
            except Exception:
                pass
            self._append_log(
                f"{len(image_paths)} Wan2.2 image2video job(s) enqueued. "
                "Monitor progress in the Queue tab."
            )
        else:
            self._append_log(
                "One or more batch jobs could not be enqueued. "
                "Image2video batch does not fall back to direct runs."
            )


    def _launch(self):
        """Run Wan2.2 either directly or via the background queue."""
        # Determine batch count (only meaningful for text2video mode)
        batch_count = 1
        try:
            if getattr(self, "spn_batch", None) and self.cmb_mode.currentText() == "text2video":
                batch_val = int(self.spn_batch.value())
                if batch_val > 1:
                    batch_count = min(max(batch_val, 1), 99)
        except Exception:
            batch_count = 1

        # Decide whether to use the queue:
        # - Always use queue when batch_count > 1
        # - Otherwise follow the "Use queue" toggle
        use_queue = False
        if batch_count > 1:
            use_queue = True
        elif getattr(self, "chk_use_queue", None) and self.chk_use_queue.isChecked():
            use_queue = True

        # Initialise extend-chain state for this run (direct runs only; queue not supported).
        self._extend_active = False
        self._extend_remaining = 0
        self._extend_segments = []
        self._extend_pending_output = None
        self._extend_auto_merge = bool(getattr(self, "chk_extend_merge", None) and self.chk_extend_merge.isChecked())
        self._extend_include_source = False
        is_v2v = bool(getattr(self, "chk_video2video", None) and self.chk_video2video.isChecked())
        if getattr(self, "spn_extend", None):
            try:
                ext_val = int(self.spn_extend.value())
            except Exception:
                ext_val = 0
            # Extend only makes sense when we are not using the queue and effectively
            # running a single job at a time.
            if ext_val > 0 and not use_queue and batch_count == 1:
                self._extend_active = True
                self._extend_segments = []
                self._extend_frame_index = 0

                if is_v2v:
                    # In video-to-video mode, treat the source video as the initial
                    # clip. Extend=N means "generate N extra segments after source".
                    # Internally we still run a first generation, but we only need
                    # (N-1) chained segments after that first run.
                    self._extend_remaining = max(ext_val - 1, 0)
                    self._append_log(
                        f"Extend: video-to-video → will generate {ext_val} segment(s) chained after the source video."
                    )
                else:
                    # Default text/image-to-video behaviour: extend=N means N extra
                    # chained segments after the initial run.
                    self._extend_remaining = ext_val
                    self._append_log(
                        f"Extend: will generate {ext_val} extra segment(s) by chaining the last frame of each clip."
                    )

                # Include-source behaviour only applies when auto-merge is on,
                # video-to-video is enabled, and we're running a direct job.
                self._extend_include_source = (
                    is_v2v
                    and self._extend_auto_merge
                    and bool(getattr(self, "chk_extend_include_source", None) and self.chk_extend_include_source.isChecked())
                    and not use_queue
                )

        if use_queue:
            self.log.clear()
            if batch_count > 1:
                self._append_log(f"Queuing {batch_count} Wan2.2 jobs in background worker…")
            else:
                self._append_log("Queuing Wan2.2 job in background worker…")
            try:
                try:
                    from helpers.queue_adapter import enqueue_wan22_from_widget
                except Exception:
                    # Fallback: local import when running helpers as a loose script
                    try:
                        import queue_adapter as _qa  # type: ignore
                        enqueue_wan22_from_widget = _qa.enqueue_wan22_from_widget  # type: ignore
                    except Exception as e2:
                        raise RuntimeError(f"Queue adapter not available: {e2}")
                ok = True
                for _ in range(batch_count):
                    if not enqueue_wan22_from_widget(self):
                        ok = False
                        break
            except Exception as e:
                self._append_log(f"Queue enqueue failed; falling back to direct run: {e}")
                ok = False

            if ok:
                # Settings are valid; remember them across sessions
                try:
                    self._save_settings()
                except Exception:
                    pass
                if batch_count > 1:
                    self._append_log(f"{batch_count} Wan2.2 jobs enqueued. Monitor progress in the Queue tab.")
                else:
                    self._append_log("Wan2.2 job enqueued. Monitor progress in the Queue tab.")
                return
            else:
                self._append_log("Falling back to direct run in this tab…")


        # Direct QProcess execution (previous behaviour)
        if self.proc.state() != QProcess.NotRunning:
            self._append_log("A Wan2.2 process is already running.")
            return

        try:
            py, args, cwd = self._build_command()
        except Exception as e:
            self._append_log(f"Error: {e}")
            return

        self.log.clear()
        self._append_log(f"Python: {py}")
        self._append_log(f"Working dir: {cwd}")

        # Show seed information
        if self.chk_random_seed.isChecked():
            self._append_log("Using RANDOM seed (new seed generated for each generation)")
        else:
            self._append_log(f"Using manual seed: {self.spn_seed.value()}")

        self._append_log("Launching Wan2.2 generate.py…")
        self._append_log(
            "Note: After finishing steps it starts part 2 of the process which may take about same time like the steps to finish."
        )

        self.proc.setProgram(py)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(cwd)
        self.proc.start()

    def _do_probe(self):
        self.log.clear()
        py = self._python_exe()
        model_root = self._model_root()
        gen = self._generate_script()

        self._append_log("WAN 2.2 probe:")
        self._append_log(f"Python: {py}")
        self._append_log(f"Model root: {model_root}")
        self._append_log(f"generate.py exists: {gen.exists()}")
        self._append_log(
            "If generate.py fails, try running it once in a terminal inside "
            + str(model_root)
            + " using the same Python interpreter."
        )

    def _read_proc(self):
        data = bytes(self.proc.readAllStandardOutput())
        if data:
            try:
                text = data.decode(errors="ignore")
            except Exception:
                text = repr(data)
            self.log.appendPlainText(text.rstrip("\n"))

    def _try_nvenc_reencode(self, out_path: Path):
        """Optionally re-encode the final MP4 using FFmpeg NVENC.

        This is experimental and only runs when the NVENC checkbox is enabled
        and ffmpeg is available. On failure it leaves the original file intact.
        """
        if not getattr(self, "chk_use_nvenc", None) or not self.chk_use_nvenc.isChecked():
            return

        ffmpeg = _ffmpeg_exe()
        if not ffmpeg:
            self._append_log("NVENC: Bundled ffmpeg not found, skipping GPU encode.")
            return

        if not out_path.is_file():
            self._append_log(f"NVENC: Output file not found at {out_path}, skipping.")
            return

        tmp_out = out_path.with_suffix(out_path.suffix + ".nvenc_tmp")
        fps = self.spn_fps.value() if getattr(self, "spn_fps", None) else 24

        self._append_log(f"NVENC: Re-encoding video with FFmpeg NVENC → {out_path.name}")
        try:
            import subprocess

            cmd = [
                ffmpeg,
                "-y",
                "-hwaccel", "cuda",
                "-i", str(out_path),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-b:v", "10M",
                "-pix_fmt", "yuv420p",
            ]
            if fps > 0:
                cmd.extend(["-r", str(fps)])
            cmd.append(str(tmp_out))

            self._append_log("NVENC cmd: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            subprocess.run(cmd, check=True)

            try:
                out_path.unlink()
            except Exception:
                pass
            tmp_out.rename(out_path)
            self._append_log("NVENC: Re-encode completed successfully.")
        except Exception as e:
            self._append_log(f"NVENC: Failed to re-encode with ffmpeg NVENC: {e}")
            try:
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass


    def _auto_merge_extend_segments(self):
        """Auto-merge all extend-chain segments into a single video (ffmpeg concat)."""
        segments: list[Path] = []
        try:
            # Optionally prepend the original video2video source when requested.
            if getattr(self, "_extend_include_source", False):
                try:
                    src = getattr(self, "_video2video_path", None)
                except Exception:
                    src = None
                if isinstance(src, Path) and src.is_file():
                    segments.append(src)

            for p in getattr(self, "_extend_segments", []):
                if isinstance(p, Path) and p.is_file():
                    segments.append(p)
        except Exception:
            segments = []
        if len(segments) < 2:
            return
        ff = _ffmpeg_exe()
        if not ff:
            self._append_log("Extend: ffmpeg not found; cannot auto-merge segments.")
            return
        try:
            base_out = self.ed_out.text().strip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if base_out:
                base = Path(base_out)
                stem = base.stem or "wan22_extend"
                suffix = base.suffix or ".mp4"
                merged = base.with_name(f"{stem}_merged_{timestamp}{suffix}")
            else:
                # Always use APP_ROOT/output/video/wan22 for extend auto-merge output,
                # so results stay with the normal Wan2.2 videos.
                out_dir = APP_ROOT / "output" / "video" / "wan22"
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                merged = out_dir / f"wan22_extend_merged_{timestamp}.mp4"

            concat_file = TEMP_DIR / "wan22_extend_concat.txt"
            lines = []
            for p in segments:
                try:
                    lines.append(f"file '{p.as_posix()}'\n")
                except Exception:
                    continue
            if not lines:
                self._append_log("Extend: no valid segments to merge.")
                return
            with open(concat_file, "w", encoding="utf-8") as f:
                f.writelines(lines)

            self._append_log(f"Extend: merging {len(segments)} segments → {merged.name}")
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(merged),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if merged.is_file():
                self._append_log("Extend: auto-merge completed.")
                try:
                    self.fileReady.emit(merged)
                except Exception:
                    pass
        except Exception as e:
            self._append_log(f"Extend: failed to auto-merge segments: {e}")

    def _start_extend_segment(self, last_segment: Path):
        """Start the next extend-chain segment using the last frame of ``last_segment``."""
        if self.proc.state() != QProcess.NotRunning:
            self._append_log("Extend: a Wan2.2 process is already running, cannot continue chain.")
            self._extend_active = False
            self._extend_remaining = 0
            return
        frame_path = self._next_extend_frame_path()
        if not self._extract_last_frame(last_segment, frame_path):
            self._append_log(f"Extend: failed to extract last frame from {last_segment.name}; stopping chain.")
            self._extend_active = False
            self._extend_remaining = 0
            return
        base_out = self.ed_out.text().strip()
        if base_out:
            base = Path(base_out)
            stem = base.stem or "wan22_extend"
            suffix = base.suffix or ".mp4"
            idx = len(self._extend_segments)
            next_path = base.with_name(f"{stem}_ext{idx:02d}{suffix}")
        else:
            # Always use APP_ROOT/output/video/wan22 for extend segments so they
            # live alongside the main Wan2.2 video outputs.
            out_dir = APP_ROOT / "output" / "video" / "wan22"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            idx = len(self._extend_segments)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            next_path = out_dir / f"wan22_extend_{timestamp}_{idx:02d}.mp4"

        self._append_log(
            f"Extend: starting chained segment {len(self._extend_segments)} "
            f"using {frame_path.name} → {next_path.name}"
        )

        prev_mode = self.cmb_mode.currentText()
        prev_image = self.ed_image.text()
        prev_out = self.ed_out.text()
        try:
            self.cmb_mode.blockSignals(True)
            self.cmb_mode.setCurrentText("image2video")
        finally:
            self.cmb_mode.blockSignals(False)

        self.ed_image.setText(str(frame_path))
        self.ed_out.setText(str(next_path))
        self._extend_pending_output = next_path

        try:
            py, args, cwd = self._build_command()
        except Exception as e:
            self._append_log(f"Extend: failed to build command for chained segment: {e}")
            self.ed_image.setText(prev_image)
            self.ed_out.setText(prev_out)
            try:
                self.cmb_mode.blockSignals(True)
                self.cmb_mode.setCurrentText(prev_mode)
            finally:
                self.cmb_mode.blockSignals(False)
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
            return

        # Restore UI state for the user while the chained process runs
        self.ed_image.setText(prev_image)
        self.ed_out.setText(prev_out)
        try:
            self.cmb_mode.blockSignals(True)
            self.cmb_mode.setCurrentText(prev_mode)
        finally:
            self.cmb_mode.blockSignals(False)

        if self._extend_remaining > 0:
            self._extend_remaining -= 1

        self.log.clear()
        self._append_log(f"Python: {py}")
        self._append_log(f"Working dir: {cwd}")
        if self.chk_random_seed.isChecked():
            self._append_log("Using RANDOM seed (new seed generated for each generation)")
        else:
            self._append_log(f"Using manual seed: {self.spn_seed.value()}")
        self._append_log("Launching Wan2.2 generate.py…")
        self._append_log(
            "Note: After finishing steps it starts part 2 of the...rocess which may take about same time like the steps to finish."
        )
        self.proc.setProgram(py)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(cwd)
        self.proc.start()

    def _handle_extend_after_finished(self, code: int, segment_path: Optional[Path]):
        """Handle extend-chain bookkeeping after each QProcess finishes."""
        if not getattr(self, "_extend_active", False):
            return
        if code != 0 or segment_path is None or not segment_path.is_file():
            self._append_log("Extend: last run failed or output missing; stopping extend chain.")
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
            return
        try:
            self._extend_segments.append(segment_path)
        except Exception:
            pass
        if self._extend_remaining > 0:
            # There are still chained segments pending; start the next one.
            self._start_extend_segment(segment_path)
        else:
            # Chain is done; optionally merge.
            if getattr(self, "_extend_auto_merge", False) and self._extend_segments:
                self._auto_merge_extend_segments()
            self._extend_active = False
            self._extend_segments = []
            self._extend_pending_output = None

    def _on_finished(self, code: int, status):
        self._append_log(f"Process finished with code {code}.")
        segment_path: Optional[Path] = None
        out_text = self.ed_out.text().strip()
        # Normal explicit-output handling (for UI + NVENC)
        if out_text:
            out_path = Path(out_text)
            if code == 0 and out_path.exists():
                self._try_nvenc_reencode(out_path)
            if out_path.exists():
                try:
                    self.fileReady.emit(out_path)
                except Exception:
                    pass
                # For non-extend runs or the initial manual segment, this is a good
                # candidate for the segment path.
                if not getattr(self, "_extend_active", False) or self._extend_pending_output is None:
                    segment_path = out_path

        # For extend-chained segments, prefer the pending output path we set
        # just before launching the chained QProcess.
        if getattr(self, "_extend_active", False) and getattr(self, "_extend_pending_output", None) is not None:
            try:
                p = self._extend_pending_output
            except Exception:
                p = None
            if isinstance(p, Path) and p.is_file():
                segment_path = p
            self._extend_pending_output = None

        # If we are in extend mode and we still don't know which video was
        # just rendered, first try the exact path recorded in _build_command.
        if segment_path is None and getattr(self, "_extend_active", False) and code == 0:
            try:
                p = getattr(self, "_last_run_out_path", None)
            except Exception:
                p = None
            if isinstance(p, Path) and p.is_file():
                segment_path = p

        # As a final fallback, guess from the most recent Wan2.2 video in
        # the outputs tree (legacy behaviour for robustness).
        if segment_path is None and getattr(self, "_extend_active", False) and code == 0:
            try:
                vids = self._iter_recent_videos(limit=1)
            except Exception:
                vids = []
            if vids:
                segment_path = vids[0]

        # Handle extend-chain logic (if enabled)
        try:
            self._handle_extend_after_finished(code, segment_path)
        except Exception as e:
            self._append_log(f"Extend: internal error while handling chain: {e}")

        # Save settings after generation completes
        self._save_settings()

        try:
            QTimer.singleShot(0, self._refresh_recent)
        except Exception:
            pass

