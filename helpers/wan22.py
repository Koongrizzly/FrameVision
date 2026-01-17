from __future__ import annotations

import os
import sys
import json
import random
import subprocess
import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, Signal, QUrl, QTimer, QSize, QSignalBlocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy,
    QLabel, QLineEdit, QTextEdit, QSpinBox, QComboBox, QSlider,
    QPushButton, QFileDialog, QPlainTextEdit, QCheckBox,
    QApplication, QMessageBox, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup,
    QStackedWidget,
)
from PySide6.QtWidgets import QScrollArea
from PySide6.QtGui import QImage, QDesktopServices
from helpers.mediainfo import refresh_info_now

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
SETTINGS_FILE = APP_ROOT / "presets" / "setsave" / "wan22.json"


# Shared Prompt Tool settings (used by the Qwen prompt enhancer helper)
PROMPT_SETTINGS_FILE = APP_ROOT / "presets" / "setsave" / "prompt.json"
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

        # Extend-chain state (text2video → chained image2video segments)
        self._extend_active = False
        self._extend_remaining = 0
        self._extend_segments: list[Path] = []
        self._extend_frame_index = 0
        self._extend_auto_merge = False
        self._extend_include_source = False
        self._extend_pending_output: Optional[Path] = None
        # Multi-prompt for extend-chain
        self._extend_multiprompt_enabled = False
        self._extend_multiprompt_prompts: list[str] = []
        # Remember the exact output path used for the last direct run
        # so extend-chains can always find the correct clip instead of
        # guessing from the recent-results folder.
        self._last_run_out_path: Optional[Path] = None

        # Sidecar JSON metadata (written next to finished outputs)
        self._last_run_meta: dict = {}
        self._sidecar_sync_timer = QTimer(self)
        self._sidecar_sync_timer.setInterval(5000)
        self._sidecar_sync_timer.timeout.connect(self._sync_sidecar_from_queue_jobs)
        try:
            self._sidecar_sync_timer.start()
        except Exception:
            pass

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
        self.banner.setFixedHeight(48)
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

        # --- Engine selector -----------------------------------------------------
        engine_row = QHBoxLayout()
        self.cmb_engine = QComboBox()
        self.cmb_engine.addItem("WAN 2.2", "wan22")
        self.cmb_engine.addItem("HunyuanVideo 1.5", "hunyuan15")
        engine_row.addWidget(QLabel("Engine:"))
        engine_row.addWidget(self.cmb_engine)
        engine_row.addStretch(1)
        layout.addLayout(engine_row)

        # --- Engine pages --------------------------------------------------------
        self._engine_stack = QStackedWidget()
        layout.addWidget(self._engine_stack, stretch=1)

        self._wan_page = QWidget()
        wan_layout = QVBoxLayout(self._wan_page)
        wan_layout.setContentsMargins(0, 0, 0, 0)
        wan_layout.setSpacing(6)



        # --- Top: mode selector -------------------------------------------------
        top = QHBoxLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["text2video", "image2video"])
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.cmb_mode)
        top.addWidget(QLabel("  (WAN 2.2 TI2V-5B)"))
        top.addStretch(1)
        wan_layout.addLayout(top)

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
        # Prompt helper row (Enhance + Clear) between prompt and negatives
        prompt_btn_row = QHBoxLayout()
        self.btn_prompt_enhance = QPushButton("Enhance prompt (Qwen)")
        try:
            self.btn_prompt_enhance.setToolTip(
                "Expand this prompt with the Qwen3-VL prompt helper (running in its own .venv). "
                "Great for adding detail and variety to Wan 2.2 prompts."
            )
        except Exception:
            pass
        try:
            self.btn_prompt_enhance.clicked.connect(self._on_enhance_prompt_clicked)
        except Exception:
            pass

        self.btn_prompt_clear = QPushButton("Clear")
        try:
            self.btn_prompt_clear.setToolTip("Clear the main prompt box so you can start over.")
        except Exception:
            pass
        try:
            self.btn_prompt_clear.clicked.connect(self._on_clear_prompt_clicked)
        except Exception:
            pass

        try:
            prompt_btn_row.addWidget(self.btn_prompt_enhance)
            prompt_btn_row.addWidget(self.btn_prompt_clear)
            prompt_btn_row.addStretch(1)
            prompt_btn_wrap = QWidget(self)
            prompt_btn_wrap.setLayout(prompt_btn_row)
            form.addRow("", prompt_btn_wrap)
        except Exception:
            pass

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

       #     "640*384",
       #     "736*432",
       #     "832*480, not working well",
       #     "480*832, not working well",
            "896*512, gives many artifacts",
            "512*896",
            "960*544, still not good quality",
            "1024*704",            
            "1280*544", 
            "1280*704",    # Landscape 704p (primary)
            "704*1280",    # Portrait 704p              
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
        self.spn_frames.setRange(16, 300)
        self.spn_frames.setValue(121)
        self.spn_frames.setToolTip("Number of frames in the generated video")

        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(16, 30)
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
        self.spn_extend.valueChanged.connect(self._on_extend_value_changed)
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
        frames_row.addSpacing(8)
        frames_row.addWidget(QLabel("FPS:"))
        frames_row.addWidget(self.spn_fps)
        frames_row.addSpacing(8)
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

        self.btn_multi_prompt = QPushButton("Multi prompt")
        self.btn_multi_prompt.setToolTip(
            "Assign a new prompt per Extend segment.\n"
            "Only applies to direct Extend chains (not queue)."
        )
        self.btn_multi_prompt.setVisible(False)
        self.btn_multi_prompt.clicked.connect(self._on_multi_prompt_clicked)

        extend_row.addWidget(self.btn_multi_prompt)
        extend_row.addStretch(1)
        extend_widget = QWidget()
        extend_widget.setLayout(extend_row)
        form.addRow("", extend_widget)

        frames_widget = QWidget()
        frames_widget.setLayout(frames_row)
        form.addRow("", frames_widget)
        # T5 text encoder CPU offload + model offload controls (below frames/extend)
        t5_offload_row = QHBoxLayout()
        self.chk_t5_cpu = QCheckBox("T5 on CPU")
        self.chk_t5_cpu.setToolTip(
            "Offload the WAN 2.2 text encoder (T5) to CPU to reduce VRAM usage. Helpful on 8–12GB GPUs. Slower when enabled."
        )
        self.chk_offload_model = QCheckBox("Offload model")
        self.chk_offload_model.setToolTip(
            "When enabled, Wan 2.2 will offload the main model weights to CPU during sampling (if supported by generate.py)."
        )
        t5_offload_row.addWidget(self.chk_t5_cpu)
        t5_offload_row.addWidget(self.chk_offload_model)
        t5_offload_row.addStretch(1)
        t5_offload_widget = QWidget()
        t5_offload_widget.setLayout(t5_offload_row)
        form.addRow("", t5_offload_widget)

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
        wan_layout.addWidget(scroll, stretch=1)

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
        wan_layout.addWidget(self.log, stretch=1)

        # --- Buttons ------------------------------------------------------------
        btn_row = QHBoxLayout()

        # Queue toggle: run via worker queue vs direct QProcess
        self.chk_use_queue = QCheckBox("Use queue")
        self.chk_use_queue.setToolTip("Tip : DISABLE queue if you want to EXTEND a video,Do not start other vram heavy jobs while queue is off")

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
        btn_row.addStretch(1)
        wan_layout.addLayout(btn_row)
        # Second row: tool helpers (right-aligned)
        btn_row2 = QHBoxLayout()

        self.btn_view_results = QPushButton("View results")
        self.btn_view_results.setToolTip("Open Media Explorer and scan this tool's output folder.")
        self.btn_view_results.setStyleSheet(shared_btn_font)

        btn_row2.addStretch(1)
        btn_row2.addWidget(self.btn_use_current)
        btn_row2.addWidget(self.btn_view_results)
        wan_layout.addLayout(btn_row2)


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


        # Finalize engine pages (WAN)
        try:
            self._engine_stack.addWidget(self._wan_page)
        except Exception:
            pass

        # HunyuanVideo 1.5 page (optional)
        self._huny_page = QWidget()
        huny_layout = QVBoxLayout(self._huny_page)
        huny_layout.setContentsMargins(0, 0, 0, 0)
        huny_layout.setSpacing(0)
        self._hunyuan_widget = None
        try:
            try:
                from helpers.hunyuan15 import Hunyuan15ToolWidget  # type: ignore
            except Exception:
                from hunyuan15 import Hunyuan15ToolWidget  # type: ignore
            self._hunyuan_widget = Hunyuan15ToolWidget(parent=self._huny_page, standalone=False)
            huny_layout.addWidget(self._hunyuan_widget)
        except Exception:
            lbl = QLabel("HunyuanVideo 1.5 UI not available (missing hunyuan15.py).")
            lbl.setWordWrap(True)
            huny_layout.addWidget(lbl)

        try:
            self._engine_stack.addWidget(self._huny_page)
        except Exception:
            pass

        # Mode toggling
        self.cmb_mode.currentTextChanged.connect(self._update_mode)
        self._update_mode()

        # Button actions
        # Probe button is hidden in the UI but kept for possible future use
        # self.btn_probe.clicked.connect(self._do_probe)
        self.btn_run.clicked.connect(self._launch)
        self.btn_play_last.clicked.connect(self._on_play_last)
        self.btn_use_current.clicked.connect(self._on_use_current)
        self.btn_view_results.clicked.connect(self._on_view_results)
        
        # Queue/T5 CPU safety:
        # When running direct (Use queue unchecked), T5-on-CPU can stall on some rigs.
        # We therefore disable and force-off the T5 toggle unless queue is effectively used.
        try:
            if getattr(self, "chk_use_queue", None):
                self.chk_use_queue.toggled.connect(self._on_use_queue_toggled)
        except Exception:
            pass
        try:
            if getattr(self, "spn_batch", None):
                self.spn_batch.valueChanged.connect(lambda _v: self._sync_t5_cpu_availability())
        except Exception:
            pass
        try:
            self.cmb_mode.currentTextChanged.connect(lambda _t: self._sync_t5_cpu_availability())
        except Exception:
            pass

        # Connect value change signals to save settings
        self._connect_signals_for_saving()
        
        # Apply loaded settings after UI is fully initialized
        self._apply_loaded_settings()

        # Ensure the correct engine page is visible on startup
        try:
            self._sync_engine_view(save=False)
        except Exception:
            pass

        # Now that the UI is set up, persist engine changes immediately
        try:
            self.cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        except Exception:
            pass

    def _on_enhance_prompt_clicked(self):
        """Enhance the prompt using the shared Qwen helper without freezing the UI."""
        # Guard against re-entry
        try:
            if getattr(self, "_qwen_proc", None) is not None and self._qwen_proc.state() != QProcess.NotRunning:
                try:
                    QMessageBox.information(self, "Prompt enhancer", "Qwen is already enhancing a prompt.")
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Collect current prompt/negative
        try:
            base_prompt = (self.ed_prompt.toPlainText() or "").strip()
        except Exception:
            base_prompt = ""
        if not base_prompt:
            try:
                QMessageBox.warning(self, "Prompt enhancer", "Please enter a base prompt first.")
            except Exception:
                pass
            return

        try:
            neg = (self.ed_negative.toPlainText() or "").strip() if getattr(self, "ed_negative", None) else ""
        except Exception:
            neg = ""

        # Locate app root / helpers
        try:
            from pathlib import Path as _P
            here = _P(__file__).resolve()
            helpers_dir = here.parent
            app_root = helpers_dir.parent
        except Exception:
            try:
                from pathlib import Path as _P
                app_root = _P.cwd()
                helpers_dir = app_root / "helpers"
            except Exception:
                app_root = Path.cwd()
                helpers_dir = app_root

        # Locate dedicated .venv Python (Qwen environment)
        py_path = None
        try:
            venv = app_root / ".venv"
            win_py = venv / "Scripts" / "python.exe"
            nix_py = venv / "bin" / "python"
            if os.name == "nt" and win_py.exists():
                py_path = win_py
            elif nix_py.exists():
                py_path = nix_py
        except Exception:
            py_path = None

        if py_path is None:
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "Could not find a dedicated .venv Python.\nExpected .venv/Scripts/python.exe or .venv/bin/python next to the app folder."
                )
            except Exception:
                pass
            return

        # Use shared CLI helper
        cli_path = helpers_dir / "prompt_enhancer_cli.py"
        if not cli_path.exists():
            try:
                alt = app_root / "helpers" / "prompt_enhancer_cli.py"
                if alt.exists():
                    cli_path = alt
            except Exception:
                pass

        if not cli_path.exists():
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "helpers/prompt_enhancer_cli.py is missing.\nThis button reuses the Txt2Img Qwen helper."
                )
            except Exception:
                pass
            return

        # Force the shared prompt-helper settings to VIDEO for WAN 2.2
        try:
            self._prompttool_force_target("video")
        except Exception:
            pass

        # Build command
        cmd = [str(py_path), str(cli_path), "--seed", base_prompt]
        if neg:
            cmd += ["--neg", neg]

        # Ensure the QProcess exists
        self._ensure_qwen_process()

        # Reset buffers + start
        try:
            self._qwen_stdout_buf = bytearray()
            self._qwen_stderr_buf = bytearray()
        except Exception:
            self._qwen_stdout_buf = bytearray()
            self._qwen_stderr_buf = bytearray()

        # UI busy state
        self._set_qwen_busy(True)

        # Environment
        try:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUTF8", "1")
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert("PYTHONUNBUFFERED", "1")
            self._qwen_proc.setProcessEnvironment(env)
        except Exception:
            pass

        try:
            self._qwen_proc.setWorkingDirectory(str(app_root))
        except Exception:
            pass

        try:
            self._append_log("Enhancing prompt with Qwen3-VL…, You can change detailed settings in Tools/prompt enhancement for more variation")
        except Exception:
            pass

        try:
            self._qwen_proc.start(cmd[0], cmd[1:])
        except Exception as e:
            self._set_qwen_busy(False)
            try:
                QMessageBox.critical(self, "Prompt enhancer", f"Failed to start Qwen helper: {e}")
            except Exception:
                pass

    def _ensure_qwen_process(self):
        """Lazy-init the Qwen QProcess and its signal wiring."""
        if getattr(self, "_qwen_proc", None) is not None:
            return
        self._qwen_proc = QProcess(self)
        # Keep stdout/stderr separate so JSON parsing of stdout stays clean.
        try:
            self._qwen_proc.setProcessChannelMode(QProcess.SeparateChannels)
        except Exception:
            pass
        self._qwen_proc.readyReadStandardOutput.connect(self._read_qwen_stdout)
        self._qwen_proc.readyReadStandardError.connect(self._read_qwen_stderr)
        self._qwen_proc.finished.connect(self._on_qwen_finished)
        self._qwen_stdout_buf = bytearray()
        self._qwen_stderr_buf = bytearray()

    def _set_qwen_busy(self, busy: bool):
        """Disable the enhance button while Qwen runs (prevents double-click freezes)."""
        btn = getattr(self, "btn_prompt_enhance", None)
        if btn is None:
            return
        try:
            if not hasattr(self, "_qwen_btn_text"):
                self._qwen_btn_text = btn.text()
        except Exception:
            self._qwen_btn_text = "Enhance prompt (Qwen)"

        try:
            btn.setEnabled(not busy)
        except Exception:
            pass
        try:
            btn.setText("Enhancing…" if busy else str(getattr(self, "_qwen_btn_text", "Enhance prompt (Qwen)")))
        except Exception:
            pass

    def _read_qwen_stdout(self):
        try:
            self._qwen_stdout_buf += bytes(self._qwen_proc.readAllStandardOutput())
        except Exception:
            pass

    def _read_qwen_stderr(self):
        try:
            self._qwen_stderr_buf += bytes(self._qwen_proc.readAllStandardError())
        except Exception:
            pass

    def _prompttool_force_target(self, target: str = "video"):
        """Temporarily force PromptTool's 'target' (image/video) for the shared Qwen helper.

        The shared CLI helper used by WAN reads PromptTool's saved state from
        presets/setsave/prompt.json. WAN should always request VIDEO prompts.
        """
        try:
            p = PROMPT_SETTINGS_FILE
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            had_file = bool(p.exists())
            data = {}
            had_target_key = False
            prev_target = None

            if had_file:
                try:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    obj = json.loads(raw) if raw.strip() else {}
                    if isinstance(obj, dict):
                        data = obj
                except Exception:
                    data = {}

            try:
                had_target_key = "target" in data
                prev_target = data.get("target")
            except Exception:
                had_target_key = False
                prev_target = None

            # If already correct, don't set restore info.
            try:
                if isinstance(prev_target, str) and prev_target.lower() == str(target).lower():
                    self._prompttool_restore_info = None
                    return
            except Exception:
                pass

            data["target"] = str(target)

            try:
                p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                # If write fails, don't attempt restore later.
                self._prompttool_restore_info = None
                return

            # Save restore info for when Qwen finishes.
            self._prompttool_restore_info = {
                "path": str(p),
                "had_file": had_file,
                "had_target_key": had_target_key,
                "prev_target": prev_target,
            }
        except Exception:
            self._prompttool_restore_info = None

    def _prompttool_restore_target(self):
        """Restore PromptTool 'target' after a WAN enhancement run."""
        info = getattr(self, "_prompttool_restore_info", None)
        if not info:
            return
        try:
            p = Path(info.get("path", ""))
        except Exception:
            self._prompttool_restore_info = None
            return

        try:
            had_file = bool(info.get("had_file", False))
            had_target_key = bool(info.get("had_target_key", False))
            prev_target = info.get("prev_target", None)

            if not had_file:
                # We created the file only for WAN; remove it to avoid altering PromptTool defaults.
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            else:
                # Restore previous target or remove the key if it didn't exist before.
                data = {}
                try:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    obj = json.loads(raw) if raw.strip() else {}
                    if isinstance(obj, dict):
                        data = obj
                except Exception:
                    data = {}

                try:
                    if had_target_key:
                        data["target"] = prev_target
                    else:
                        data.pop("target", None)
                    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
        finally:
            self._prompttool_restore_info = None

    def _on_qwen_finished(self, exit_code: int, exit_status):
        self._set_qwen_busy(False)

        # Restore PromptTool target (image/video) if we temporarily forced it for WAN
        try:
            self._prompttool_restore_target()
        except Exception:
            pass

        try:
            out_txt = bytes(getattr(self, "_qwen_stdout_buf", b"")).decode("utf-8", "ignore").strip()
        except Exception:
            out_txt = ""
        try:
            err_txt = bytes(getattr(self, "_qwen_stderr_buf", b"")).decode("utf-8", "ignore").strip()
        except Exception:
            err_txt = ""

        if exit_code != 0:
            msg = err_txt or out_txt or f"Exit code {exit_code}"
            if len(msg) > 2000:
                msg = msg[:2000] + "..."
            try:
                QMessageBox.critical(self, "Prompt enhancer", "Qwen prompt helper failed:\n\n" + msg)
            except Exception:
                pass
            return

        # Parse JSON payload from stdout
        data = None
        try:
            data = json.loads(out_txt)
        except Exception:
            data = None

        if not isinstance(data, dict) or not data.get("ok"):
            msg = out_txt or err_txt or "Unexpected response from helper."
            if len(msg) > 2000:
                msg = msg[:2000] + "..."
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "Qwen prompt helper returned an unexpected payload:\n\n" + msg
                )
            except Exception:
                pass
            return

        new_prompt = data.get("prompt") or ""
        new_neg = data.get("negatives") or ""

        if new_prompt:
            try:
                self.ed_prompt.setPlainText(new_prompt)
            except Exception:
                pass
        if new_neg and getattr(self, "ed_negative", None):
            try:
                self.ed_negative.setPlainText(new_neg)
            except Exception:
                pass

        try:
            self._append_log("Prompt enhanced with Qwen3-VL")
        except Exception:
            pass
    def _on_clear_prompt_clicked(self):

        """Clear the main positive prompt box."""
        try:
            self.ed_prompt.clear()
        except Exception:
            try:
                self.ed_prompt.setPlainText("")
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
        if getattr(self, "chk_t5_cpu", None):
            self.chk_t5_cpu.toggled.connect(self._save_settings)
        if getattr(self, "chk_offload_model", None):
            self.chk_offload_model.toggled.connect(self._save_settings)

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
                "engine": (self.cmb_engine.currentData() if getattr(self, "cmb_engine", None) and self.cmb_engine.currentData() else (self.cmb_engine.currentText() if getattr(self, "cmb_engine", None) else "wan22")),
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
                "t5_cpu": bool(getattr(self, "chk_t5_cpu", None) and self.chk_t5_cpu.isChecked()),
                "offload_model": bool(getattr(self, "chk_offload_model", None) and self.chk_offload_model.isChecked()),
                "use_nvenc": bool(getattr(self, "chk_use_nvenc", None) and self.chk_use_nvenc.isChecked()),
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

        engine = settings.get("engine")
        if engine is not None and getattr(self, "cmb_engine", None):
            try:
                idx = self.cmb_engine.findData(engine)
                if idx < 0:
                    idx = self.cmb_engine.findText(str(engine))
                if idx >= 0:
                    self.cmb_engine.setCurrentIndex(idx)
            except Exception:
                pass


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
        if "t5_cpu" in settings and getattr(self, "chk_t5_cpu", None):
            self.chk_t5_cpu.setChecked(bool(settings["t5_cpu"]))
            try:
                self._t5_cpu_restore = bool(settings["t5_cpu"])
            except Exception:
                self._t5_cpu_restore = bool(self.chk_t5_cpu.isChecked())
        if "offload_model" in settings and getattr(self, "chk_offload_model", None):
            self.chk_offload_model.setChecked(bool(settings["offload_model"]))
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

    # ---------------------------------------------------------------------
    # Queue-aware T5 CPU UI guard
    # ---------------------------------------------------------------------
    def _effective_queue_for_ui(self) -> bool:
        """Return True when the UI should behave as if the queue will be used.

        Rules:
        - If 'Use queue' is checked => True
        - If text2video batch > 1 => True (because _launch forces the queue)
        """
        try:
            if getattr(self, "chk_use_queue", None) and self.chk_use_queue.isChecked():
                return True
        except Exception:
            pass
        try:
            if getattr(self, "spn_batch", None) and self.cmb_mode.currentText() == "text2video":
                if int(self.spn_batch.value()) > 1:
                    return True
        except Exception:
            pass
        return False

    def _sync_t5_cpu_availability(self):
        """Disable 'T5 on CPU' for direct runs to avoid stalls.

        When queue is not effectively used, the checkbox is forced OFF and disabled.
        We remember the previous user choice and restore it when queue-use is enabled again.
        """
        if not getattr(self, "chk_t5_cpu", None):
            return

        effective_queue = self._effective_queue_for_ui()

        if effective_queue:
            try:
                self.chk_t5_cpu.setEnabled(True)
            except Exception:
                pass

            try:
                restore = bool(getattr(self, "_t5_cpu_restore", False))
            except Exception:
                restore = False

            try:
                if restore and not self.chk_t5_cpu.isChecked():
                    blocker = QSignalBlocker(self.chk_t5_cpu)
                    self.chk_t5_cpu.setChecked(True)
                    del blocker
            except Exception:
                pass
        else:
            try:
                self._t5_cpu_restore = bool(self.chk_t5_cpu.isChecked())
            except Exception:
                self._t5_cpu_restore = False

            try:
                blocker = QSignalBlocker(self.chk_t5_cpu)
                self.chk_t5_cpu.setChecked(False)
                del blocker
            except Exception:
                try:
                    self.chk_t5_cpu.setChecked(False)
                except Exception:
                    pass

            try:
                self.chk_t5_cpu.setEnabled(False)
            except Exception:
                pass


    def _sync_engine_view(self, save: bool = False):
        """Sync the visible engine page + banner text with the engine selector."""
        key = "wan22"
        try:
            if getattr(self, "cmb_engine", None) is not None:
                data = self.cmb_engine.currentData()
                key = str(data) if data else str(self.cmb_engine.currentText())
        except Exception:
            key = "wan22"

        try:
            if getattr(self, "_engine_stack", None) is not None:
                self._engine_stack.setCurrentIndex(0 if key == "wan22" else 1)
        except Exception:
            pass

        try:
            if key == "wan22":
                self.banner.setText("Video Creation with Wan 2.2 5B")
            else:
                self.banner.setText("Video Creation with HunyuanVideo 1.5")
        except Exception:
            pass

        if save:
            try:
                self._save_settings()
            except Exception:
                pass

    def _on_engine_changed(self, *_):
        self._sync_engine_view(save=True)

    def _on_use_queue_toggled(self, checked: bool):
        """Handle 'Use queue' toggle for T5 CPU safety UI."""
        try:
            self._sync_t5_cpu_availability()
        except Exception:
            pass

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
    # ---- Output path helpers ----
    def _wan_outputs_dir(self) -> Path:
        """Return the folder where Wan2.2 renders its final videos.

        This avoids accidentally using any thumbnail/"last results" folders.

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

    def _on_extend_value_changed(self, v: int):
        try:
            v = int(v)
        except Exception:
            v = 0

        if getattr(self, "btn_multi_prompt", None):
            self.btn_multi_prompt.setVisible(v > 0)

        if v <= 0:
            # Clear multiprompt state when extend is disabled
            self._extend_multiprompt_enabled = False
            self._extend_multiprompt_prompts = []

    def _on_multi_prompt_clicked(self):
        try:
            ext_val = int(self.spn_extend.value()) if getattr(self, "spn_extend", None) else 0
        except Exception:
            ext_val = 0

        if ext_val <= 0:
            try:
                QMessageBox.information(self, "Multi prompt", "Set Extend to 1 or higher first.")
            except Exception:
                pass
            return

        # Base prompt for convenience
        try:
            base_prompt = (self.ed_prompt.toPlainText() or "").strip()
        except Exception:
            base_prompt = ""

        try:
            from helpers.wan22_multiprompt_dialog import MultiPromptExtendDialog
        except Exception:
            try:
                from wan22_multiprompt_dialog import MultiPromptExtendDialog  # type: ignore
            except Exception as e:
                QMessageBox.critical(self, "Multi prompt", f"Dialog file missing:\n{e}")
                return

        dlg = MultiPromptExtendDialog(ext_val, base_prompt=base_prompt, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        prompts = dlg.prompts()

        # Store config for the upcoming extend chain
        self._extend_multiprompt_enabled = True
        self._extend_multiprompt_prompts = prompts

        try:
            filled = sum(1 for p in prompts if p.strip())
            self._append_log(f"Multi prompt configured for {ext_val} extend segment(s). "
                             f"{filled} prompt(s) filled.")
        except Exception:
            pass

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


    def _on_view_results(self):
        """Jump to Media Explorer and scan this tool's output folder."""
        # Pick a sensible folder based on the selected engine.
        try:
            key = str(self.cmb_engine.currentData() or self.cmb_engine.currentText() or "").lower()
        except Exception:
            key = "wan22"

        folder = None
        try:
            if "huny" in key:
                cand = APP_ROOT / "output" / "video" / "hunyuan15"
                if not cand.exists():
                    cand = APP_ROOT / "output" / "video"
                folder = cand
            else:
                folder = self._wan_outputs_dir()
        except Exception:
            folder = APP_ROOT / "output" / "video"

        folder_str = str(folder)
        main = getattr(self, "main", None)

        # Preferred: go through the app router if present.
        try:
            if main is not None and hasattr(main, "open_media_explorer_folder"):
                main.open_media_explorer_folder(
                    folder_str,
                    want_images=False,
                    want_videos=True,
                    want_audio=False,
                    include_subfolders=False,
                )
                return
        except Exception:
            pass

        # Fallback: direct access to the Media Explorer tab (if available).
        try:
            if main is not None and hasattr(main, "media_explorer") and hasattr(main, "tabs"):
                tab = main.media_explorer
                # Set filter toggles if present (keep it safe on older builds).
                try:
                    if hasattr(tab, "cb_videos"):
                        tab.cb_videos.setChecked(True)
                    if hasattr(tab, "cb_images"):
                        tab.cb_images.setChecked(False)
                    if hasattr(tab, "cb_audio"):
                        tab.cb_audio.setChecked(False)
                    if hasattr(tab, "cb_subfolders"):
                        tab.cb_subfolders.setChecked(False)
                except Exception:
                    pass

                try:
                    if hasattr(tab, "set_root_folder"):
                        tab.set_root_folder(folder_str)
                    elif hasattr(tab, "ed_folder"):
                        tab.ed_folder.setText(folder_str)
                except Exception:
                    pass

                # Switch to the Explorer tab.
                try:
                    main.tabs.setCurrentWidget(tab)
                except Exception:
                    try:
                        for i in range(main.tabs.count()):
                            if main.tabs.widget(i) is tab:
                                main.tabs.setCurrentIndex(i)
                                break
                    except Exception:
                        pass

                # Kick a scan.
                try:
                    if hasattr(tab, "rescan"):
                        tab.rescan()
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Last resort: open folder in OS file browser.
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_str))
        except Exception:
            try:
                if os.name == "nt":
                    os.startfile(folder_str)  # type: ignore[attr-defined]
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


    # ---------------------------------------------------------------------
    # Install safety (model presence)
    # ---------------------------------------------------------------------
    def _wan22_models_installed(self) -> bool:
        """Return True when the WAN 2.2 model folder exists and contains at least one file."""
        try:
            model_root = self._model_root()
        except Exception:
            model_root = None

        try:
            if model_root is None:
                return False
            if (not model_root.exists()) or (not model_root.is_dir()):
                return False
        except Exception:
            return False

        # Look for any file inside (recursive). This is intentionally simple:
        # optional downloads may place different file sets depending on version.
        try:
            for p in model_root.rglob("*"):
                try:
                    if p.is_file():
                        return True
                except Exception:
                    continue
        except Exception:
            return False

        return False

    def _open_optional_installs_py(self):
        """Open helpers/opt_installs.py in the OS default editor (best-effort)."""
        try:
            # wan22.py lives in helpers/, so APP_ROOT/helpers is the right target.
            p = APP_ROOT / "helpers" / "opt_installs.py"
            if not p.exists():
                # Fallback: same folder as this file (should also be helpers/)
                try:
                    p2 = Path(__file__).resolve().parent / "opt_installs.py"
                    if p2.exists():
                        p = p2
                except Exception:
                    pass

            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            else:
                try:
                    QMessageBox.information(
                        self,
                        "Optional downloads",
                        "Could not find helpers/opt_installs.py."
                    )
                except Exception:
                    pass
        except Exception:
            pass

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





    def _detect_wan_t5_cpu_flag(self, py_exe: str, gen_path: Path, model_root: Path):
        """
        Inspect `generate.py --help` once and cache whether WAN supports the
        --t5_cpu flag (text encoder CPU offload).
        """
        if hasattr(self, "_wan_t5_cpu_flag"):
            return self._wan_t5_cpu_flag

        has_flag = False
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
            if "--t5_cpu" in help_text:
                has_flag = True
        except Exception as e:
            try:
                print(f"Wan2.2 T5 CPU flag detection failed: {e}")
            except Exception:
                pass

        self._wan_t5_cpu_flag = has_flag
        return has_flag

    def _build_command(self):
        """
        Build the python + args + working directory tuple for QProcess.

        We avoid full-model CPU offload because it can make long clips extremely slow.
        The optional "T5 on CPU" toggle lets users offload only the text encoder
        to reduce VRAM usage with a smaller speed penalty.
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
            "--fps", str(self.spn_fps.value() if getattr(self, "spn_fps", None) else 24),
            "--ckpt_dir", str(model_root),
            "--convert_model_dtype",
        ]

        # Optional model offload override: when unchecked, do not pass a flag
        # so generate.py can apply its own default behavior (matches queue).
        try:
            if getattr(self, "chk_offload_model", None) and self.chk_offload_model.isChecked():
                args += ["--offload_model", "True"]
        except Exception:
            pass

        if prompt:
            args += ["--prompt", prompt]

        # Optional text-encoder (T5) CPU offload: only when user requests it.
        try:
            if getattr(self, "chk_t5_cpu", None) and self.chk_t5_cpu.isChecked():
                has_t5_cpu = self._detect_wan_t5_cpu_flag(py, gen, model_root)
                if has_t5_cpu:
                    args.append("--t5_cpu")
                else:
                    try:
                        self._append_log(
                            "T5 on CPU requested, but --t5_cpu is not supported by your generate.py; skipping."
                        )
                    except Exception:
                        pass
        except Exception:
            pass

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

        # Capture run metadata for sidecar JSON
        try:
            l1w = None
            l2w = None
            if getattr(self, "slider_lora1_weight", None) is not None:
                try:
                    l1w = float(self.slider_lora1_weight.value()) / 100.0
                except Exception:
                    l1w = None
            if getattr(self, "slider_lora2_weight", None) is not None:
                try:
                    l2w = float(self.slider_lora2_weight.value()) / 100.0
                except Exception:
                    l2w = None

            v2v_enabled = bool(getattr(self, "chk_video2video", None) and self.chk_video2video.isChecked())
            v2v_src = None
            try:
                if v2v_enabled and isinstance(getattr(self, "_video2video_path", None), Path):
                    v2v_src = str(self._video2video_path)
            except Exception:
                v2v_src = None

            self._last_run_meta = {
                "engine": "wan22",
                "mode": mode,
                "prompt": prompt,
                "negative": negative,
                "size": size_str,
                "steps": int(self.spn_steps.value()),
                "guidance": float(self.spn_guidance.value()),
                "seed": int(seed_value),
                "frames": int(self.spn_frames.value()),
                "fps": int(self.spn_fps.value() if getattr(self, "spn_fps", None) else 24),
                "t5_cpu": bool(getattr(self, "chk_t5_cpu", None) and self.chk_t5_cpu.isChecked()),
                "offload_model": bool(getattr(self, "chk_offload_model", None) and self.chk_offload_model.isChecked()),
                "lora1_path": lora1_path,
                "lora1_weight": l1w,
                "lora2_path": lora2_path,
                "lora2_weight": l2w,
                "relighting_lora": bool(use_relighting),
                "video2video": bool(v2v_enabled),
                "video2video_source": v2v_src,
                "extend": int(self.spn_extend.value()) if getattr(self, "spn_extend", None) else 0,
                "batch": int(self.spn_batch.value()) if getattr(self, "spn_batch", None) else 1,
                "out_path": str(out_path),
            }
        except Exception:
            try:
                self._last_run_meta = {"engine": "wan22", "out_path": str(out_path)}
            except Exception:
                self._last_run_meta = {}

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

    def _current_media_path(self) -> Optional[Path]:
        """Best-effort: return the currently loaded media file path from the main player."""
        main = None
        try:
            main = self._find_main_with_video()
        except Exception:
            main = None
        if main is None:
            return None

        # Prefer the app's tracked current_path (kept in sync by _open_in_player and other tabs)
        try:
            cur = getattr(main, "current_path", None)
            if cur:
                p = Path(str(cur)).expanduser()
                if p.exists():
                    return p
        except Exception:
            pass

        # Fallback: ask the video widget for a path / url
        try:
            video = getattr(main, "video", None)
        except Exception:
            video = None
        if video is None:
            return None

        # Common attribute names used across different player implementations
        for attr in ("current_path", "path", "file_path", "filepath", "source", "filename", "file"):
            try:
                val = getattr(video, attr, None)
            except Exception:
                val = None
            if not val:
                continue
            try:
                # QUrl -> local file
                if hasattr(val, "toLocalFile"):
                    val = val.toLocalFile()
            except Exception:
                pass
            try:
                p = Path(str(val)).expanduser()
                if p.exists():
                    return p
            except Exception:
                continue

        return None

    def _player_position_seconds(self, video_obj) -> Optional[float]:
        """Best-effort: extract current playback position as seconds (float)."""
        try:
            objs = [video_obj]
            for a in ("player", "mediaPlayer", "mp", "qplayer"):
                try:
                    o = getattr(video_obj, a, None)
                except Exception:
                    o = None
                if o is not None:
                    objs.append(o)

            for o in objs:
                if o is None:
                    continue
                for name in ("position", "currentPosition", "pos", "position_ms", "current_ms", "currentTime"):
                    if not hasattr(o, name):
                        continue
                    try:
                        v = getattr(o, name)
                    except Exception:
                        continue
                    try:
                        v = v() if callable(v) else v
                    except Exception:
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    v = float(v)
                    # Heuristics: QMediaPlayer returns ms; some custom players may return seconds.
                    if "ms" in name or v > 10000.0:
                        return max(0.0, v / 1000.0)
                    return max(0.0, v)
        except Exception:
            pass
        return None

    def _save_qimage_jpg95(self, qimg: QImage, out_path: Path) -> bool:
        """Save QImage as JPEG (quality 95) while preserving pixel dimensions."""
        try:
            img = qimg
            try:
                if img.hasAlphaChannel():
                    # JPEG has no alpha; drop it safely.
                    img = img.convertToFormat(QImage.Format_RGB32)
            except Exception:
                pass
            return bool(img.save(str(out_path), "JPG", 95))
        except Exception:
            return False

    def _export_current_media_to_temp_jpg(self) -> Optional[Path]:
        """Export the current player frame/image to a temp JPG at original resolution."""
        try:
            import time as _time
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            out_jpg = TEMP_DIR / f"wan22_current_{int(_time.time())}.jpg"
        except Exception:
            return None

        main = None
        try:
            main = self._find_main_with_video()
        except Exception:
            main = None

        video = None
        try:
            video = getattr(main, "video", None) if main is not None else None
        except Exception:
            video = None

        src = self._current_media_path()

        # 1) If the player has a real source path, prefer exporting from that (full-res).
        if src is not None:
            ext = (src.suffix or "").lower()
            img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            vid_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
            try:
                if ext in img_exts:
                    q = QImage(str(src))
                    if not q.isNull() and self._save_qimage_jpg95(q, out_jpg):
                        return out_jpg
                elif ext in vid_exts:
                    ff = _ffmpeg_exe()
                    if ff:
                        sec = self._player_position_seconds(video) if video is not None else None
                        # Extract as PNG first (lossless), then re-save as JPG 95%.
                        tmp_png = out_jpg.with_suffix(".png")
                        cmd = [ff, "-y", "-hide_banner"]
                        if sec is not None:
                            cmd += ["-ss", f"{sec:.3f}"]
                        cmd += ["-i", str(src), "-frames:v", "1", str(tmp_png)]
                        try:
                            import subprocess
                            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                            q = QImage(str(tmp_png))
                            if not q.isNull() and self._save_qimage_jpg95(q, out_jpg):
                                try:
                                    tmp_png.unlink(missing_ok=True)  # py3.8+ on win uses exists check
                                except Exception:
                                    try:
                                        if tmp_png.exists():
                                            tmp_png.unlink()
                                    except Exception:
                                        pass
                                return out_jpg
                        except Exception:
                            # If ffmpeg extraction failed, fall back below
                            pass
            except Exception:
                pass

        # 2) Fallback: whatever is currently visible (may be scaled in some player builds).
        qimg = self._grab_current_qimage()
        if qimg is None or qimg.isNull():
            return None
        if self._save_qimage_jpg95(qimg, out_jpg):
            return out_jpg
        return None

    def _on_use_current(self):
        """Switch to image2video and pull the current Media Player frame in as the start image."""
        tmp_inp = self._export_current_media_to_temp_jpg()
        if tmp_inp is None or not Path(str(tmp_inp)).exists():
            QMessageBox.warning(
                self,
                "No current frame",
                "No current frame or image was found.\n\n"
                "Load an image or pause a video in the Media Player first."
            )
            return

        # Switch mode and set the start image path.
        self.cmb_mode.setCurrentText("image2video")
        self.ed_image.setText(str(tmp_inp))
        try:
            q = QImage(str(tmp_inp))
            if not q.isNull():
                self._append_log(
                    f"Using Media Player current frame as start image (JPG 95%, {q.width()}x{q.height()}):\n{tmp_inp}"
                )
            else:
                self._append_log(f"Using Media Player current frame as start image (JPG 95%):\n{tmp_inp}")
        except Exception:
            self._append_log(f"Using Media Player current frame as start image (JPG 95%):\n{tmp_inp}")

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

        try:
            msg = f"{len(image_paths)} images loaded.\n\nContinue and add them to the Queue?"
            res = QMessageBox.question(self, "Wan 2.2", msg, QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return
        except Exception:
            # If QMessageBox fails for any reason, proceed silently.
            pass

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

        n = max(2, int(count))
        try:
            msg = f"Repeat the current image {n} times.\n\nContinue and add them to the Queue?"
            res = QMessageBox.question(self, "Wan 2.2", msg, QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return
        except Exception:
            pass

        image_paths = [image_path] * n
        self._enqueue_image_batch_jobs(image_paths)

    def _enqueue_image_batch_jobs(self, image_paths):
        """Enqueue one image2video Wan2.2 job per image path, always using the queue.

        Notes:
        - We force unique output paths per job so the queue doesn't reject duplicates.
        - If Random seed is OFF, we also increment the seed to create unique variations.
        """
        if not image_paths:
            return

        self.log.clear()
        self._append_log(
            f"Image2video batch: queuing {len(image_paths)} job(s) in background worker…"
        )
        self._append_log("Note: Batch always uses the queue, even if 'Use queue' is turned off.")

        def _safe_stub(s: str) -> str:
            try:
                import re as _re
                s = (s or "").strip()
                s = _re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
                return (s or "img")[:40]
            except Exception:
                return "img"

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
            original_out = self.ed_out.text()
            try:
                original_seed = int(self.spn_seed.value())
            except Exception:
                original_seed = 0

            use_random = bool(getattr(self, "chk_random_seed", None) and self.chk_random_seed.isChecked())

            ok = True
            # Force image2video mode while enqueuing jobs; restore afterwards.
            try:
                self.cmb_mode.blockSignals(True)
                self.cmb_mode.setCurrentText("image2video")
            finally:
                try:
                    self.cmb_mode.blockSignals(False)
                except Exception:
                    pass

            wan_dir = self._wan_outputs_dir()
            ts_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for i, img in enumerate(image_paths):
                imgp = str(img)
                self.ed_image.setText(imgp)

                # Seed: increment deterministic seeds; keep Random behavior untouched.
                if not use_random and getattr(self, "spn_seed", None) is not None:
                    try:
                        self.spn_seed.blockSignals(True)
                        self.spn_seed.setValue(int(original_seed) + int(i))
                    finally:
                        try:
                            self.spn_seed.blockSignals(False)
                        except Exception:
                            pass

                # Output: always unique per job.
                out_hint = (original_out or "").strip()
                img_stub = "img"
                try:
                    img_stub = _safe_stub(Path(imgp).stem)
                except Exception:
                    img_stub = "img"

                outp = ""
                try:
                    if out_hint:
                        base = Path(out_hint).expanduser()
                        if not base.is_absolute():
                            base = wan_dir / base
                        if not str(base).lower().endswith(".mp4"):
                            base = base.with_suffix(".mp4")
                        outp = str(base.with_name(f"{base.stem}_{img_stub}_b{i+1:02d}{base.suffix}"))
                    else:
                        outp = str(wan_dir / f"wan22_i2v_{img_stub}_b{i+1:02d}_{ts_base}.mp4")
                except Exception:
                    outp = ""

                if outp:
                    try:
                        self.ed_out.blockSignals(True)
                        self.ed_out.setText(outp)
                    finally:
                        try:
                            self.ed_out.blockSignals(False)
                        except Exception:
                            pass

                if not enqueue_wan22_from_widget(self):
                    ok = False
                    break

        except Exception as e:
            self._append_log(f"Image2video batch enqueue failed: {e}")
            ok = False
        finally:
            # Restore UI state
            try:
                self.cmb_mode.setCurrentText(original_mode)
            except Exception:
                pass
            try:
                self.ed_image.setText(original_image)
            except Exception:
                pass
            try:
                self.ed_out.setText(original_out)
            except Exception:
                pass
            try:
                if getattr(self, "spn_seed", None) is not None:
                    self.spn_seed.setValue(int(original_seed))
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
        # Safety: ensure WAN 2.2 model files exist before launching.
        if not self._wan22_models_installed():
            try:
                QMessageBox.information(
                    self,
                    "Wan 2.2",
                    "Models are not installed yet, please select 'wan 2.2 5B' from the 'optional downloads' menu to download the correct model for this tool"
                )
            except Exception:
                pass

            # After OK, open the optional installs list for the user.
            try:
                self._open_optional_installs_py()
            except Exception:
                pass
            return

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

                # Multi-prompt sanity check (direct extend only)
                if getattr(self, "_extend_multiprompt_enabled", False):
                    try:
                        expected = int(self._extend_remaining)
                    except Exception:
                        expected = 0
                    prompts = getattr(self, "_extend_multiprompt_prompts", []) or []
                    if expected > 0 and len(prompts) < expected:
                        try:
                            self._append_log(
                                "Multi prompt: fewer prompts than required for this Extend chain. "
                                "Missing segments will fall back to the main prompt."
                            )
                        except Exception:
                            pass

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

                # Ensure each queued job has a unique output path (and seed when Random is off),
                # otherwise the queue can reject duplicates (same output filename).
                orig_out = ""
                try:
                    orig_out = self.ed_out.text()
                except Exception:
                    orig_out = ""
                try:
                    orig_seed = int(self.spn_seed.value())
                except Exception:
                    orig_seed = 0
                use_random = bool(getattr(self, "chk_random_seed", None) and self.chk_random_seed.isChecked())

                try:
                    try:
                        prompt_txt = (self.ed_prompt.toPlainText() or "").strip()
                    except Exception:
                        prompt_txt = ""
                    import re as _re
                    words = [w for w in _re.split(r"\s+", (prompt_txt or "").strip()) if w][:3]
                    if not words:
                        words = ["wan22"]
                    _parts = []
                    for w in words:
                        w2 = _re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
                        if w2:
                            _parts.append(w2)
                    slug = ("_".join(_parts) if _parts else "wan22")[:32]
                except Exception:
                    slug = "wan22"

                try:
                    wan_dir = self._wan_outputs_dir()
                except Exception:
                    wan_dir = None

                try:
                    ts_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                except Exception:
                    ts_base = "auto"

                try:
                    for i in range(int(batch_count)):
                        # Seed: increment deterministic seeds; keep Random behavior untouched.
                        if not use_random and getattr(self, "spn_seed", None) is not None:
                            try:
                                self.spn_seed.blockSignals(True)
                                self.spn_seed.setValue(int(orig_seed) + int(i))
                            finally:
                                try:
                                    self.spn_seed.blockSignals(False)
                                except Exception:
                                    pass

                        # Output path: if user supplied a filename, suffix it; else auto-generate in output dir.
                        outp = ""
                        try:
                            out_hint = (orig_out or "").strip()
                            if out_hint:
                                base = Path(out_hint).expanduser()
                                if wan_dir is not None and not base.is_absolute():
                                    base = Path(wan_dir) / base
                                if not str(base).lower().endswith(".mp4"):
                                    base = base.with_suffix(".mp4")
                                outp = str(base.with_name(f"{base.stem}_b{i+1:02d}{base.suffix}"))
                            else:
                                if wan_dir is not None:
                                    outp = str(Path(wan_dir) / f"wan22_{slug}_b{i+1:02d}_{ts_base}.mp4")
                        except Exception:
                            outp = ""

                        if outp:
                            try:
                                self.ed_out.blockSignals(True)
                                self.ed_out.setText(outp)
                            finally:
                                try:
                                    self.ed_out.blockSignals(False)
                                except Exception:
                                    pass

                        if not enqueue_wan22_from_widget(self):
                            ok = False
                            break
                finally:
                    # Restore UI fields so the user doesn't see temporary batch names.
                    try:
                        self.ed_out.setText(orig_out)
                    except Exception:
                        pass
                    try:
                        if getattr(self, "spn_seed", None) is not None:
                            self.spn_seed.setValue(int(orig_seed))
                    except Exception:
                        pass

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
                    self._append_log("Wan2.2 job enqueued. Monitor progress in the Queue tab or follow steps in the worker.")
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


    # ---------------------------------------------------------------------
    # Sidecar JSON (same-name .json next to finished outputs)
    # ---------------------------------------------------------------------
    def _write_sidecar_json(self, media_path: Path, payload: dict):
        """Write <output>.json next to the produced media file (best-effort).

        The file is only created if it does not already exist.
        """
        try:
            if media_path is None:
                return
            p = Path(str(media_path))
            if not p.is_file():
                return
            sidecar = p.with_suffix(".json")
            if sidecar.exists():
                return
            try:
                sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                # Fallback: try ASCII-safe if encoding chokes (should be rare)
                sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as e:
            try:
                self._append_log(f"Sidecar JSON: failed to write for {media_path}: {e}")
            except Exception:
                pass

    def _sidecar_payload_direct(self, media_path: Path) -> dict:
        """Build a consistent sidecar payload for direct runs."""
        try:
            run_meta = dict(getattr(self, "_last_run_meta", {}) or {})
        except Exception:
            run_meta = {}
        try:
            now_iso = datetime.datetime.now().isoformat()
        except Exception:
            now_iso = ""
        return {
            "sidecar_version": 1,
            "tool": "wan22",
            "source": "direct",
            "created_at": now_iso,
            "output_file": str(media_path),
            "run": run_meta,
        }

    def _sidecar_payload_queue(self, media_path: Path, job: dict, job_json: Optional[Path] = None) -> dict:
        """Build a consistent sidecar payload for queue jobs."""
        try:
            now_iso = datetime.datetime.now().isoformat()
        except Exception:
            now_iso = ""
        payload = {
            "sidecar_version": 1,
            "tool": "wan22",
            "source": "queue",
            "created_at": now_iso,
            "output_file": str(media_path),
            "job": job or {},
        }
        if job_json is not None:
            try:
                payload["job_json"] = str(job_json)
            except Exception:
                pass
        return payload

    def _sync_sidecar_from_queue_jobs(self):
        """Best-effort: mirror finished queue job JSON next to the produced media file."""
        try:
            jobs = self._list_recent_jobs()
        except Exception:
            jobs = []
        if not jobs:
            return
        for job_json in jobs:
            try:
                media_path, j = self._resolve_output_from_job(job_json)
            except Exception:
                media_path, j = None, {}
            if not media_path:
                continue
            try:
                mp = Path(str(media_path))
            except Exception:
                continue
            try:
                if not (mp.exists() and mp.is_file()):
                    continue
            except Exception:
                continue
            sidecar = mp.with_suffix(".json")
            try:
                if sidecar.exists():
                    continue
            except Exception:
                continue
            self._write_sidecar_json(mp, self._sidecar_payload_queue(mp, j, job_json))

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
                try:
                    payload = self._sidecar_payload_direct(merged)
                    try:
                        payload["merge"] = {"merged_from": [str(p) for p in segments]}
                    except Exception:
                        pass
                    self._write_sidecar_json(merged, payload)
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
        prev_prompt = ""
        prev_negative = ""
        try:
            prev_prompt = self.ed_prompt.toPlainText()
        except Exception:
            prev_prompt = ""
        try:
            prev_negative = self.ed_negative.toPlainText() if getattr(self, "ed_negative", None) else ""
        except Exception:
            prev_negative = ""

        # Determine which extend prompt to use:
        # _extend_segments already includes the initial clip,
        # so chain_index 0 = first extra segment.
        chain_index = 0
        try:
            chain_index = max(len(self._extend_segments) - 1, 0)
        except Exception:
            chain_index = 0

        mp_enabled = bool(getattr(self, "_extend_multiprompt_enabled", False))
        mp_prompts = getattr(self, "_extend_multiprompt_prompts", []) or []
        new_prompt = ""
        if mp_enabled and chain_index < len(mp_prompts):
            try:
                new_prompt = (mp_prompts[chain_index] or "").strip()
            except Exception:
                new_prompt = ""

        if new_prompt:
            try:
                self.ed_prompt.blockSignals(True)
                self.ed_prompt.setPlainText(new_prompt)
            finally:
                try:
                    self.ed_prompt.blockSignals(False)
                except Exception:
                    pass

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
            self.ed_prompt.blockSignals(True)
            self.ed_prompt.setPlainText(prev_prompt)
        finally:
            try:
                self.ed_prompt.blockSignals(False)
            except Exception:
                pass

        if getattr(self, "ed_negative", None):
            try:
                self.ed_negative.blockSignals(True)
                self.ed_negative.setPlainText(prev_negative)
            finally:
                try:
                    self.ed_negative.blockSignals(False)
                except Exception:
                    pass
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
                if code == 0:
                    try:
                        self._write_sidecar_json(out_path, self._sidecar_payload_direct(out_path))
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

        # Write sidecar JSON for the finished segment (best-effort)
        if code == 0 and segment_path is not None:
            try:
                if isinstance(segment_path, Path) and segment_path.is_file():
                    self._write_sidecar_json(segment_path, self._sidecar_payload_direct(segment_path))
            except Exception:
                pass

        # Handle extend-chain logic (if enabled)
        try:
            self._handle_extend_after_finished(code, segment_path)
        except Exception as e:
            self._append_log(f"Extend: internal error while handling chain: {e}")

        # Save settings after generation completes
        self._save_settings()
