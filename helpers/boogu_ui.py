
#!/usr/bin/env python3
# helpers/boogu_ui.py
#
# FrameVision Boogu Image helper UI for stable-diffusion.cpp / sd-cli.
# - Normal tab: text-to-image with the Boogu Turbo model.
# - Edit tab: instruction-driven image editing with reference thumbnails.
# - Settings tab: model paths, runtime toggles, output folder names, and logs.
#
# Settings are stored in:
#   <FrameVision root>/presets/setsave/boogu.json
#
# This file is intentionally self-contained so it can be imported as a tab/helper
# or launched directly for testing.

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PySide6.QtCore import Qt, QSize, QProcess, QTimer, Signal, QPropertyAnimation, QEasingCurve
    from PySide6.QtGui import QPixmap, QIcon, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGraphicsOpacityEffect,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - only visible when PySide6 is missing
    raise RuntimeError("boogu_ui.py requires PySide6") from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_EXTS = {".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".bin"}
MODEL_FILE_FILTER = "Model files (*.gguf *.safetensors *.sft *.ckpt *.pt *.bin);;GGUF files (*.gguf);;Safetensors (*.safetensors *.sft);;All files (*.*)"
IMAGE_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)"
ALL_FILE_FILTER = "All files (*.*)"
SAMPLERS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep", "res_2s",
    "er_sde", "euler_cfg_pp", "euler_a_cfg_pp",
]
SCHEDULERS = [
    "model default", "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
    "sgm_uniform", "simple", "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal",
]
PREVIEW_METHODS = ["none", "proj", "tae", "vae"]
ASPECT_MODES = ["1:1", "16:9", "9:16", "custom"]


def find_app_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "helpers":
        return here.parent.parent
    # When launched outside FrameVision during testing, prefer cwd if it looks like root.
    cwd = Path.cwd().resolve()
    if (cwd / "presets").exists() or (cwd / "models").exists():
        return cwd
    return here.parent.parent


APP_ROOT = find_app_root()
CONFIG_PATH = APP_ROOT / "presets" / "setsave" / "boogu.json"


def rel_or_abs(path_text: str) -> Path:
    p = Path(path_text.strip().strip('"'))
    if not p.is_absolute():
        p = APP_ROOT / p
    return p


def nice_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_ROOT.resolve()))
    except Exception:
        return str(path)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class NoWheelSpinBox(QSpinBox):
    """Prevents accidental value changes while scrolling the form."""
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Prevents accidental value changes while scrolling the form."""
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class NoWheelComboBox(QComboBox):
    """Prevents accidental combo changes while scrolling the form."""
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class StickyScrollTab(QWidget):
    """Reusable layout: scrollable content + sticky bottom bar."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(8, 8, 8, 8)
        self.outer.setSpacing(8)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 4, 8, 4)
        self.content_layout.setSpacing(10)
        self.scroll.setWidget(self.content)
        self.outer.addWidget(self.scroll, 1)

        self.bottom = QFrame(self)
        self.bottom.setFrameShape(QFrame.StyledPanel)
        self.bottom_layout = QHBoxLayout(self.bottom)
        self.bottom_layout.setContentsMargins(8, 6, 8, 6)
        self.bottom_layout.setSpacing(8)
        self.outer.addWidget(self.bottom, 0)


class BooguUI(QWidget):
    generated = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("BooguUI")
        self.setWindowTitle("Boogu Image")
        self.process: Optional[QProcess] = None
        self.reference_images: List[str] = []
        self._loading = False
        self._aspect_sync_active = False
        self._toast_label: Optional[QLabel] = None
        self._toast_animation: Optional[QPropertyAnimation] = None
        self.config = self.default_config()
        self.load_config()
        self.build_ui()
        self.apply_config_to_ui()
        self.connect_auto_save()
        self.log("Boogu UI ready.")

    def show_toast(self, message: str, duration_ms: int = 2200) -> None:
        """Show a small non-blocking notification bubble that fades away."""
        if self._toast_animation is not None:
            self._toast_animation.stop()
            self._toast_animation = None
        if self._toast_label is not None:
            self._toast_label.deleteLater()

        toast = QLabel(message, self)
        toast.setObjectName("BooguToast")
        toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        toast.setAlignment(Qt.AlignCenter)
        toast.setStyleSheet(
            "QLabel#BooguToast {"
            " background: rgba(22, 26, 36, 235);"
            " color: white;"
            " border: 1px solid rgba(120, 220, 255, 180);"
            " border-radius: 10px;"
            " padding: 10px 16px;"
            " font-weight: 600;"
            "}"
        )
        toast.adjustSize()
        margin = 18
        x = max(margin, self.width() - toast.width() - margin)
        y = max(margin, self.height() - toast.height() - margin)
        toast.move(x, y)
        toast.raise_()

        opacity = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(opacity)
        opacity.setOpacity(1.0)
        toast.show()

        self._toast_label = toast

        def fade_out() -> None:
            if self._toast_label is not toast:
                return
            animation = QPropertyAnimation(opacity, b"opacity", toast)
            animation.setDuration(500)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.InOutQuad)

            def cleanup() -> None:
                if self._toast_label is toast:
                    self._toast_label = None
                if self._toast_animation is animation:
                    self._toast_animation = None
                toast.deleteLater()

            animation.finished.connect(cleanup)
            self._toast_animation = animation
            animation.start()

        QTimer.singleShot(max(0, duration_ms), fade_out)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def default_config(self) -> Dict[str, Any]:
        return {
            "paths": {
                "sd_cli": "presets/bin/sd-cli.exe",
                # Diffusion model can be safetensors/sft or experimental GGUF.
                # Boogu GGUF support depends on the sd-cli build/model tensor mapping.
                "turbo_model": "models/boogu_image/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
                "edit_model": "models/boogu_image/diffusion_models/boogu_image_edit_fp8_scaled.safetensors",
                "vae": "models/boogu_image/vae/ae.safetensors",
                "llm": "models/boogu_image/llm/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                "llm_vision": "models/boogu_image/llm/mmproj-BF16.gguf",
                "normal_output_subfolder": "output/images/boogu",
                "edit_output_subfolder": "output/edits/boogu",
            },
            "normal": {
                "prompt": "a detailed cinematic photo of a futuristic glass house in a pine forest, warm sunset light",
                "negative_prompt": "",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "cfg_scale": 1.0,
                "guidance": 3.5,
                "seed": -1,
                "batch_count": 1,
                "sampler": "euler",
                "scheduler": "model default",
                "preview": "none",
                "preview_interval": 1,
                "aspect_mode": "custom",
            },
            "edit": {
                "prompt": "change the background to a rainy neon city street while keeping the main subject recognizable",
                "negative_prompt": "",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "cfg_scale": 1.0,
                "img_cfg_scale": 1.0,
                "guidance": 3.5,
                "strength": 0.75,
                "seed": -1,
                "batch_count": 1,
                "sampler": "euler",
                "scheduler": "model default",
                "preview": "none",
                "preview_interval": 1,
                "increase_ref_index": True,
                "disable_auto_resize_ref_image": False,
                "reference_images": [],
            },
            "runtime": {
                "diffusion_flash_attention_only": True,
                "offload_to_cpu": True,
                "mmap": False,
                "eager_load": False,
                "vae_tiling": True,
                "disable_metadata": False,
                "verbose": True,
                "framevision_queue": True,
                "threads": -1,
                "rng": "cuda",
                "extra_args": "",
            },
        }

    def load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.config = self.deep_merge(self.config, data)
        except Exception as exc:
            print(f"Could not read {CONFIG_PATH}: {exc}")

    def save_config(self) -> None:
        if self._loading:
            return
        try:
            self.pull_ui_to_config()
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        except Exception as exc:
            self.log(f"Could not save settings: {exc}")

    @staticmethod
    def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = BooguUI.deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        self.normal_tab = StickyScrollTab(self)
        self.edit_tab = StickyScrollTab(self)
        self.settings_tab = StickyScrollTab(self)

        self.tabs.addTab(self.normal_tab, "Create")
        self.tabs.addTab(self.edit_tab, "Edit")
        self.tabs.addTab(self.settings_tab, "Settings")

        self.build_normal_tab()
        self.build_edit_tab()
        self.build_settings_tab()

    def make_form_group(self, title: str, tab: StickyScrollTab) -> QFormLayout:
        group = QGroupBox(title)
        layout = QFormLayout(group)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        tab.content_layout.addWidget(group)
        return layout

    def build_normal_tab(self) -> None:
        form = self.make_form_group("Text to image", self.normal_tab)
        self.n_prompt = QTextEdit()
        self.n_prompt.setMinimumHeight(100)
        self.n_prompt.setToolTip("Main prompt passed to sd-cli with --prompt. Boogu Turbo is best with clear, direct descriptions.")
        form.addRow("Prompt", self.n_prompt)

        self.n_negative = QLineEdit()
        self.n_negative.setToolTip("Optional --negative-prompt. Leave empty unless you need to push specific artifacts away.")
        form.addRow("Negative", self.n_negative)

        gen_group = QGroupBox("Generation settings")
        gen_grid = QGridLayout(gen_group)
        gen_grid.setContentsMargins(12, 10, 12, 10)
        gen_grid.setHorizontalSpacing(10)
        gen_grid.setVerticalSpacing(8)
        gen_grid.setColumnMinimumWidth(0, 90)
        gen_grid.setColumnMinimumWidth(3, 90)
        gen_grid.setColumnStretch(2, 1)
        gen_grid.setColumnStretch(5, 1)
        self.normal_tab.content_layout.addWidget(gen_group)

        self.n_width = self.spin(256, 4096, 16, "Width passed with --width. Aspect presets keep width and height synced; Custom leaves them free.")
        self.n_height = self.spin(256, 4096, 16, "Height passed with --height. Aspect presets keep width and height synced; Custom leaves them free.")
        self.n_steps = self.spin(1, 100, 1, "Boogu Turbo is designed for roughly 3-4 sampling steps; 4 is the practical default.")
        self.n_batch = self.spin(1, 64, 1, "Batch count passed with --batch-count. This creates multiple outputs from one command.")
        self.n_seed = self.spin(-1, 2_147_483_647, 1, "Seed passed with --seed. Use -1 for random.")
        self.n_cfg = self.dspin(0.0, 30.0, 0.1, "--cfg-scale. Boogu/sd.cpp guidance usually works better low; 1.0 is a safe starting point.")
        self.n_guidance = self.dspin(0.0, 30.0, 0.1, "--guidance distilled guidance. sd-cli default is 3.5, and that is a good first test.")
        self.n_sampler = self.combo(SAMPLERS, "--sampling-method. sd-cli defaults to euler for Flux-like models; keep euler unless testing.")
        self.n_scheduler = self.combo(SCHEDULERS, "--scheduler. 'model default' omits the argument and lets sd-cli choose.")
        self.n_preview = self.combo(PREVIEW_METHODS, "Preview method. 'none' is fastest and safest; 'vae' writes preview files but costs time/VRAM.")
        self.n_preview_interval = self.spin(1, 100, 1, "Preview update interval in denoising steps.")
        self.n_aspect_mode = self.combo(ASPECT_MODES, "Choose 1:1, 16:9, 9:16 or Custom. Presets lock the ratio and cap the maximum size; Custom leaves width and height free.")

        self.add_grid_pair(gen_grid, 0, "Width", self.n_width, "Height", self.n_height)
        self.add_grid_pair(gen_grid, 1, "Steps", self.n_steps, "Batch", self.n_batch)
        self.add_grid_pair(gen_grid, 2, "Seed", self.n_seed, "Format", self.n_aspect_mode)
        self.add_grid_pair(gen_grid, 3, "CFG", self.n_cfg, "Guidance", self.n_guidance)
        self.add_grid_pair(gen_grid, 4, "Sampler", self.n_sampler, "Scheduler", self.n_scheduler)
        self.add_grid_pair(gen_grid, 5, "Preview", self.n_preview, "Every", self.n_preview_interval)

        self.n_aspect_mode.currentIndexChanged.connect(self.on_normal_aspect_mode_changed)
        self.n_width.valueChanged.connect(self.on_normal_width_changed)
        self.n_height.valueChanged.connect(self.on_normal_height_changed)

        self.normal_tab.content_layout.addStretch(1)
        self.n_cmd_preview = QLineEdit()
        self.n_cmd_preview.setReadOnly(True)
        self.n_cmd_preview.setToolTip("Last command preview. It updates before generation.")
        self.normal_tab.bottom_layout.addWidget(self.n_cmd_preview, 1)
        self.n_generate = QPushButton("Generate")
        self.n_generate.setToolTip("Run sd-cli with the Create tab settings. Button stays visible while the form scrolls.")
        self.n_generate.clicked.connect(lambda: self.generate("normal"))
        self.normal_tab.bottom_layout.addWidget(self.n_generate, 0)

    def build_edit_tab(self) -> None:
        form = self.make_form_group("Instruction edit", self.edit_tab)
        self.e_prompt = QTextEdit()
        self.e_prompt.setMinimumHeight(100)
        self.e_prompt.setToolTip("Edit instruction passed to --prompt. Keep it direct: say what should change and what should stay.")
        form.addRow("Instruction", self.e_prompt)

        self.e_negative = QLineEdit()
        self.e_negative.setToolTip("Optional --negative-prompt for the edit run.")
        form.addRow("Negative", self.e_negative)

        refs_group = QGroupBox("Reference images")
        refs_l = QVBoxLayout(refs_group)
        self.ref_list = QListWidget()
        self.ref_list.setViewMode(QListWidget.IconMode)
        self.ref_list.setIconSize(QSize(128, 128))
        self.ref_list.setResizeMode(QListWidget.Adjust)
        self.ref_list.setMovement(QListWidget.Static)
        self.ref_list.setMinimumHeight(170)
        self.ref_list.setToolTip("Images passed with repeated --ref-image entries. Add one or more source images for edit mode.")
        refs_l.addWidget(self.ref_list)
        btns = QHBoxLayout()
        add_btn = QPushButton("Add image")
        add_btn.setToolTip("Add reference image(s) for --ref-image.")
        add_btn.clicked.connect(self.add_reference_images)
        rem_btn = QPushButton("Remove selected")
        rem_btn.clicked.connect(self.remove_selected_reference)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_references)
        btns.addWidget(add_btn); btns.addWidget(rem_btn); btns.addWidget(clear_btn); btns.addStretch(1)
        refs_l.addLayout(btns)
        self.edit_tab.content_layout.addWidget(refs_group)

        gen_group = QGroupBox("Edit generation")
        gen_grid = QGridLayout(gen_group)
        gen_grid.setContentsMargins(12, 10, 12, 10)
        gen_grid.setHorizontalSpacing(10)
        gen_grid.setVerticalSpacing(8)
        gen_grid.setColumnMinimumWidth(0, 90)
        gen_grid.setColumnMinimumWidth(3, 90)
        gen_grid.setColumnStretch(2, 1)
        gen_grid.setColumnStretch(5, 1)
        self.edit_tab.content_layout.addWidget(gen_group)

        self.e_width = self.spin(256, 4096, 64, "Width passed with --width. Match the source image aspect ratio when possible.")
        self.e_height = self.spin(256, 4096, 64, "Height passed with --height. Match the source image aspect ratio when possible.")
        self.e_steps = self.spin(1, 100, 1, "Edit is not the 4-step Turbo workflow. 20 is a conservative sd-cli-style starting point.")
        self.e_batch = self.spin(1, 64, 1, "Batch count passed with --batch-count.")
        self.e_seed = self.spin(-1, 2_147_483_647, 1, "Seed passed with --seed. Use -1 for random.")
        self.e_cfg = self.dspin(0.0, 30.0, 0.1, "--cfg-scale. Low values are a safer starting point for Qwen/Boogu style pipelines.")
        self.e_img_cfg = self.dspin(0.0, 30.0, 0.1, "--img-cfg-scale. Image guidance for image edit models; defaults to CFG if omitted, but explicit 1.0 is predictable.")
        self.e_guidance = self.dspin(0.0, 30.0, 0.1, "--guidance distilled guidance. sd-cli default is 3.5.")
        self.e_strength = self.dspin(0.0, 1.0, 0.01, "--strength. 0.75 is sd-cli's default; lower keeps more of the input, higher changes more.")
        self.e_sampler = self.combo(SAMPLERS, "--sampling-method. euler is the safe default for Flux-like models in sd-cli.")
        self.e_scheduler = self.combo(SCHEDULERS, "--scheduler. 'model default' omits the argument and lets sd-cli choose.")
        self.e_preview = self.combo(PREVIEW_METHODS, "Preview method. 'none' is fastest and safest.")
        self.e_preview_interval = self.spin(1, 100, 1, "Preview update interval in denoising steps.")

        self.add_grid_pair(gen_grid, 0, "Width", self.e_width, "Height", self.e_height)
        self.add_grid_pair(gen_grid, 1, "Steps", self.e_steps, "Batch", self.e_batch)
        self.add_grid_pair(gen_grid, 2, "Seed", self.e_seed, "", None)
        self.add_grid_pair(gen_grid, 3, "CFG", self.e_cfg, "Img CFG", self.e_img_cfg)
        self.add_grid_pair(gen_grid, 4, "Guidance", self.e_guidance, "Strength", self.e_strength)
        self.add_grid_pair(gen_grid, 5, "Sampler", self.e_sampler, "Scheduler", self.e_scheduler)
        self.add_grid_pair(gen_grid, 6, "Preview", self.e_preview, "Every", self.e_preview_interval)

        ref_row = QWidget()
        ref_l = QHBoxLayout(ref_row)
        ref_l.setContentsMargins(0, 0, 0, 0)
        self.e_increase_ref_index = QCheckBox("Increase reference index")
        self.e_increase_ref_index.setToolTip("Adds --increase-ref-index. Useful when sending multiple reference images in order.")
        self.e_disable_resize = QCheckBox("Disable auto resize ref image")
        self.e_disable_resize.setToolTip("Adds --disable-auto-resize-ref-image. Usually leave off unless you need exact source sizing.")
        ref_l.addWidget(self.e_increase_ref_index)
        ref_l.addWidget(self.e_disable_resize)
        ref_l.addStretch(1)
        label = self.grid_label("Reference options")
        gen_grid.addWidget(label, 7, 0, Qt.AlignRight | Qt.AlignVCenter)
        gen_grid.addWidget(ref_row, 7, 1, 1, 5)

        self.edit_tab.content_layout.addStretch(1)
        self.e_cmd_preview = QLineEdit()
        self.e_cmd_preview.setReadOnly(True)
        self.e_cmd_preview.setToolTip("Last command preview. It updates before generation.")
        self.edit_tab.bottom_layout.addWidget(self.e_cmd_preview, 1)
        self.e_generate = QPushButton("Generate edit")
        self.e_generate.setToolTip("Run sd-cli with the Edit tab settings. Button stays visible while the form scrolls.")
        self.e_generate.clicked.connect(lambda: self.generate("edit"))
        self.edit_tab.bottom_layout.addWidget(self.e_generate, 0)

    def build_settings_tab(self) -> None:
        paths_form = self.make_form_group("Paths", self.settings_tab)
        self.path_sd_cli = self.path_row(paths_form, "sd-cli", "Path to sd-cli.exe. Default is presets/bin/sd-cli.exe.", file_mode=True)
        self.path_turbo_model = self.path_row(paths_form, "Turbo model", "Boogu Turbo diffusion model used by Create tab. Supports safetensors/sft and experimental GGUF files if your sd-cli build can load them.", file_mode=True, file_filter=MODEL_FILE_FILTER)
        self.path_edit_model = self.path_row(paths_form, "Edit model", "Boogu Edit diffusion model used by Edit tab. Supports safetensors/sft and experimental GGUF files if your sd-cli build can load them.", file_mode=True, file_filter=MODEL_FILE_FILTER)
        self.path_vae = self.path_row(paths_form, "VAE", "FLUX VAE passed with --vae.", file_mode=True, file_filter="VAE files (*.safetensors *.sft *.gguf);;All files (*.*)")
        self.path_llm = self.path_row(paths_form, "Qwen LLM", "Qwen3-VL GGUF text encoder passed with --llm.", file_mode=True, file_filter="GGUF files (*.gguf);;All files (*.*)")
        self.path_llm_vision = self.path_row(paths_form, "Vision mmproj", "Matching mmproj passed with --llm_vision for edit mode.", file_mode=True, file_filter="GGUF files (*.gguf);;All files (*.*)")
        self.path_normal_out = self.path_row(paths_form, "Create output", "Folder used by Create tab outputs.", file_mode=False)
        self.path_edit_out = self.path_row(paths_form, "Edit output", "Folder used by Edit tab outputs.", file_mode=False)

        runtime_form = self.make_form_group("Runtime", self.settings_tab)
        self.rt_diffusion_fa = QCheckBox("Diffusion flash attention only")
        self.rt_diffusion_fa.setToolTip("Adds --diffusion-fa. Used in the sd.cpp Boogu examples.")
        self.rt_offload = QCheckBox("Offload to CPU")
        self.rt_offload.setToolTip("Adds --offload-to-cpu. Used in the sd.cpp Boogu examples and helps fit large models.")
        self.rt_mmap = QCheckBox("Memory map")
        self.rt_mmap.setToolTip("Adds --mmap. Can reduce loading overhead depending on storage and build.")
        self.rt_eager = QCheckBox("Eager load")
        self.rt_eager.setToolTip("Adds --eager-load. Loads parameters at startup instead of lazily. Usually leave off.")
        self.rt_vae_tiling = QCheckBox("VAE tiling")
        self.rt_vae_tiling.setToolTip("Adds --vae-tiling. Useful for larger images and lower VRAM.")
        self.rt_disable_metadata = QCheckBox("Disable metadata")
        self.rt_disable_metadata.setToolTip("Adds --disable-image-metadata.")
        self.rt_verbose = QCheckBox("Verbose")
        self.rt_verbose.setToolTip("Adds -v for more sd-cli output in the log.")
        self.rt_framevision_queue = QCheckBox("Use FrameVision queue")
        self.rt_framevision_queue.setToolTip("When enabled, Generate adds the Boogu job to the FrameVision queue instead of running sd-cli directly in this tab.")
        self.rt_framevision_queue.toggled.connect(self.update_queue_button_text)
        checks = QWidget(); checks_l = QGridLayout(checks); checks_l.setContentsMargins(0,0,0,0)
        for i, cb in enumerate([self.rt_diffusion_fa, self.rt_offload, self.rt_mmap, self.rt_eager, self.rt_vae_tiling, self.rt_disable_metadata, self.rt_verbose, self.rt_framevision_queue]):
            checks_l.addWidget(cb, i // 2, i % 2)
        runtime_form.addRow("Toggles", checks)

        self.rt_threads = self.spin(-1, 128, 1, "--threads. -1 lets sd-cli use physical CPU cores.")
        self.rt_rng = self.combo(["cuda", "cpu", "std_default"], "--rng. sd-cli help says default is cuda(sd-webui), cpu(comfyui). CUDA is a good default for this helper.")
        pair = QWidget(); pair_l = QHBoxLayout(pair); pair_l.setContentsMargins(0,0,0,0)
        pair_l.addWidget(QLabel("Threads")); pair_l.addWidget(self.rt_threads)
        pair_l.addSpacing(12); pair_l.addWidget(QLabel("RNG")); pair_l.addWidget(self.rt_rng); pair_l.addStretch(1)
        runtime_form.addRow("CPU / RNG", pair)

        self.rt_extra_args = QLineEdit()
        self.rt_extra_args.setToolTip("Extra sd-cli arguments appended last. Advanced use only. add --max-vram 20 --stream-layers for higher resolutions on low vram cards")
        runtime_form.addRow("Extra args", self.rt_extra_args)

        log_group = QGroupBox("Logs")
        log_l = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        self.log_box.setToolTip("sd-cli output and helper messages.")
        log_l.addWidget(self.log_box)
        log_btns = QHBoxLayout()
        clear_log = QPushButton("Clear log")
        clear_log.clicked.connect(self.log_box.clear)
        open_cfg = QPushButton("Open settings folder")
        open_cfg.clicked.connect(lambda: self.open_folder(CONFIG_PATH.parent))
        log_btns.addWidget(clear_log); log_btns.addWidget(open_cfg); log_btns.addStretch(1)
        log_l.addLayout(log_btns)
        self.settings_tab.content_layout.addWidget(log_group)
        self.settings_tab.content_layout.addStretch(1)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the current sd-cli process if one is running.")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("Save settings")
        self.save_btn.setToolTip("Write current settings to presets/setsave/boogu.json.")
        self.save_btn.clicked.connect(self.save_config)
        self.settings_tab.bottom_layout.addStretch(1)
        self.settings_tab.bottom_layout.addWidget(self.stop_btn)
        self.settings_tab.bottom_layout.addWidget(self.save_btn)

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------
    def grid_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setMinimumWidth(90)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return label

    def add_grid_pair(self, grid: QGridLayout, row: int, label_a: str, widget_a: Optional[QWidget], label_b: str, widget_b: Optional[QWidget]) -> None:
        if label_a:
            grid.addWidget(self.grid_label(label_a), row, 0, Qt.AlignRight | Qt.AlignVCenter)
        if widget_a is not None:
            grid.addWidget(widget_a, row, 1)
        if label_b:
            grid.addWidget(self.grid_label(label_b), row, 3, Qt.AlignRight | Qt.AlignVCenter)
        if widget_b is not None:
            grid.addWidget(widget_b, row, 4)

    def spin(self, lo: int, hi: int, step: int, tooltip: str) -> NoWheelSpinBox:
        s = NoWheelSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setToolTip(tooltip)
        s.setKeyboardTracking(False)
        return s

    def dspin(self, lo: float, hi: float, step: float, tooltip: str) -> NoWheelDoubleSpinBox:
        s = NoWheelDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(3 if step < 0.01 else 2)
        s.setToolTip(tooltip)
        s.setKeyboardTracking(False)
        return s

    def combo(self, items: List[str], tooltip: str) -> NoWheelComboBox:
        c = NoWheelComboBox()
        c.addItems(items)
        c.setToolTip(tooltip)
        return c

    def path_row(self, form: QFormLayout, label: str, tooltip: str, file_mode: bool, file_filter: str = ALL_FILE_FILTER) -> QLineEdit:
        edit = QLineEdit()
        edit.setToolTip(tooltip)
        btn = QPushButton("Browse")
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda: self.browse_path(edit, file_mode=file_mode, file_filter=file_filter))
        row = QWidget(); row_l = QHBoxLayout(row); row_l.setContentsMargins(0,0,0,0)
        row_l.addWidget(edit, 1); row_l.addWidget(btn, 0)
        form.addRow(label, row)
        return edit

    # ------------------------------------------------------------------
    # Config apply/pull/connect
    # ------------------------------------------------------------------
    def apply_config_to_ui(self) -> None:
        self._loading = True
        p = self.config["paths"]
        self.path_sd_cli.setText(p.get("sd_cli", ""))
        self.path_turbo_model.setText(p.get("turbo_model", ""))
        self.path_edit_model.setText(p.get("edit_model", ""))
        self.path_vae.setText(p.get("vae", ""))
        self.path_llm.setText(p.get("llm", ""))
        self.path_llm_vision.setText(p.get("llm_vision", ""))
        self.path_normal_out.setText(p.get("normal_output_subfolder", ""))
        self.path_edit_out.setText(p.get("edit_output_subfolder", ""))

        n = self.config["normal"]
        self.n_prompt.setPlainText(n.get("prompt", ""))
        self.n_negative.setText(n.get("negative_prompt", ""))
        self.n_width.setValue(int(n.get("width", 1024)))
        self.n_height.setValue(int(n.get("height", 1024)))
        self.set_combo(self.n_aspect_mode, n.get("aspect_mode", "custom"))
        self.n_steps.setValue(int(n.get("steps", 4)))
        self.n_batch.setValue(int(n.get("batch_count", 1)))
        self.n_seed.setValue(int(n.get("seed", -1)))
        self.n_cfg.setValue(float(n.get("cfg_scale", 1.0)))
        self.n_guidance.setValue(float(n.get("guidance", 3.5)))
        self.set_combo(self.n_sampler, n.get("sampler", "euler"))
        self.set_combo(self.n_scheduler, n.get("scheduler", "model default"))
        self.set_combo(self.n_preview, n.get("preview", "none"))
        self.n_preview_interval.setValue(int(n.get("preview_interval", 1)))

        e = self.config["edit"]
        self.e_prompt.setPlainText(e.get("prompt", ""))
        self.e_negative.setText(e.get("negative_prompt", ""))
        self.e_width.setValue(int(e.get("width", 1024)))
        self.e_height.setValue(int(e.get("height", 1024)))
        self.e_steps.setValue(int(e.get("steps", 20)))
        self.e_batch.setValue(int(e.get("batch_count", 1)))
        self.e_seed.setValue(int(e.get("seed", -1)))
        self.e_cfg.setValue(float(e.get("cfg_scale", 1.0)))
        self.e_img_cfg.setValue(float(e.get("img_cfg_scale", 1.0)))
        self.e_guidance.setValue(float(e.get("guidance", 3.5)))
        self.e_strength.setValue(float(e.get("strength", 0.75)))
        self.set_combo(self.e_sampler, e.get("sampler", "euler"))
        self.set_combo(self.e_scheduler, e.get("scheduler", "model default"))
        self.set_combo(self.e_preview, e.get("preview", "none"))
        self.e_preview_interval.setValue(int(e.get("preview_interval", 1)))
        self.e_increase_ref_index.setChecked(bool(e.get("increase_ref_index", True)))
        self.e_disable_resize.setChecked(bool(e.get("disable_auto_resize_ref_image", False)))
        self.reference_images = list(e.get("reference_images", []))
        self.refresh_reference_list()

        rt = self.config["runtime"]
        self.rt_diffusion_fa.setChecked(bool(rt.get("diffusion_flash_attention_only", True)))
        self.rt_offload.setChecked(bool(rt.get("offload_to_cpu", True)))
        self.rt_mmap.setChecked(bool(rt.get("mmap", False)))
        self.rt_eager.setChecked(bool(rt.get("eager_load", False)))
        self.rt_vae_tiling.setChecked(bool(rt.get("vae_tiling", True)))
        self.rt_disable_metadata.setChecked(bool(rt.get("disable_metadata", False)))
        self.rt_verbose.setChecked(bool(rt.get("verbose", True)))
        self.rt_framevision_queue.setChecked(bool(rt.get("framevision_queue", True)))
        self.update_queue_button_text()
        self.rt_threads.setValue(int(rt.get("threads", -1)))
        self.set_combo(self.rt_rng, rt.get("rng", "cuda"))
        self.rt_extra_args.setText(rt.get("extra_args", ""))
        self._loading = False
        self.apply_normal_aspect_mode()

    def pull_ui_to_config(self) -> None:
        self.config["paths"] = {
            "sd_cli": self.path_sd_cli.text().strip(),
            "turbo_model": self.path_turbo_model.text().strip(),
            "edit_model": self.path_edit_model.text().strip(),
            "vae": self.path_vae.text().strip(),
            "llm": self.path_llm.text().strip(),
            "llm_vision": self.path_llm_vision.text().strip(),
            "normal_output_subfolder": self.path_normal_out.text().strip(),
            "edit_output_subfolder": self.path_edit_out.text().strip(),
        }
        self.config["normal"] = {
            "prompt": self.n_prompt.toPlainText().strip(),
            "negative_prompt": self.n_negative.text().strip(),
            "width": self.n_width.value(),
            "height": self.n_height.value(),
            "steps": self.n_steps.value(),
            "cfg_scale": self.n_cfg.value(),
            "guidance": self.n_guidance.value(),
            "seed": self.n_seed.value(),
            "batch_count": self.n_batch.value(),
            "sampler": self.n_sampler.currentText(),
            "scheduler": self.n_scheduler.currentText(),
            "preview": self.n_preview.currentText(),
            "preview_interval": self.n_preview_interval.value(),
            "aspect_mode": self.n_aspect_mode.currentText(),
        }
        self.config["edit"] = {
            "prompt": self.e_prompt.toPlainText().strip(),
            "negative_prompt": self.e_negative.text().strip(),
            "width": self.e_width.value(),
            "height": self.e_height.value(),
            "steps": self.e_steps.value(),
            "cfg_scale": self.e_cfg.value(),
            "img_cfg_scale": self.e_img_cfg.value(),
            "guidance": self.e_guidance.value(),
            "strength": self.e_strength.value(),
            "seed": self.e_seed.value(),
            "batch_count": self.e_batch.value(),
            "sampler": self.e_sampler.currentText(),
            "scheduler": self.e_scheduler.currentText(),
            "preview": self.e_preview.currentText(),
            "preview_interval": self.e_preview_interval.value(),
            "increase_ref_index": self.e_increase_ref_index.isChecked(),
            "disable_auto_resize_ref_image": self.e_disable_resize.isChecked(),
            "reference_images": self.reference_images,
        }
        self.config["runtime"] = {
            "diffusion_flash_attention_only": self.rt_diffusion_fa.isChecked(),
            "offload_to_cpu": self.rt_offload.isChecked(),
            "mmap": self.rt_mmap.isChecked(),
            "eager_load": self.rt_eager.isChecked(),
            "vae_tiling": self.rt_vae_tiling.isChecked(),
            "disable_metadata": self.rt_disable_metadata.isChecked(),
            "verbose": self.rt_verbose.isChecked(),
            "framevision_queue": self.rt_framevision_queue.isChecked(),
            "threads": self.rt_threads.value(),
            "rng": self.rt_rng.currentText(),
            "extra_args": self.rt_extra_args.text().strip(),
        }

    @staticmethod
    def round_to_step(value: int, step: int = 16) -> int:
        return max(step, int(round(value / step) * step))

    def apply_normal_aspect_mode(self, driver: str = "width") -> None:
        if self._loading or self._aspect_sync_active:
            return
        mode = self.n_aspect_mode.currentText()
        self._aspect_sync_active = True
        try:
            self.n_width.setRange(256, 4096)
            self.n_height.setRange(256, 4096)
            if mode == "custom":
                return

            if mode == "1:1":
                limit = 1760
                base = self.n_height.value() if driver == "height" else self.n_width.value()
                value = min(max(256, self.round_to_step(base, 16)), limit)
                self.n_width.setMaximum(limit)
                self.n_height.setMaximum(limit)
                self.n_width.setValue(value)
                self.n_height.setValue(value)
                return

            if mode == "16:9":
                rw, rh, max_w, max_h = 16, 9, 2304, 1296
            else:
                rw, rh, max_w, max_h = 9, 16, 1296, 2304

            self.n_width.setMaximum(max_w)
            self.n_height.setMaximum(max_h)

            if driver == "height":
                height = min(max(256, self.round_to_step(self.n_height.value(), 16)), max_h)
                width = self.round_to_step(int(round(height * rw / rh)), 16)
                if width > max_w:
                    width = max_w
                    height = self.round_to_step(int(round(width * rh / rw)), 16)
            else:
                width = min(max(256, self.round_to_step(self.n_width.value(), 16)), max_w)
                height = self.round_to_step(int(round(width * rh / rw)), 16)
                if height > max_h:
                    height = max_h
                    width = self.round_to_step(int(round(height * rw / rh)), 16)

            width = max(256, min(width, max_w))
            height = max(256, min(height, max_h))
            self.n_width.setValue(width)
            self.n_height.setValue(height)
        finally:
            self._aspect_sync_active = False

    def on_normal_aspect_mode_changed(self, *args) -> None:
        self.apply_normal_aspect_mode("width")

    def on_normal_width_changed(self, *args) -> None:
        self.apply_normal_aspect_mode("width")

    def on_normal_height_changed(self, *args) -> None:
        self.apply_normal_aspect_mode("height")

    def connect_auto_save(self) -> None:
        widgets = [
            self.n_prompt, self.e_prompt, self.n_negative, self.e_negative,
            self.path_sd_cli, self.path_turbo_model, self.path_edit_model, self.path_vae,
            self.path_llm, self.path_llm_vision, self.path_normal_out, self.path_edit_out,
            self.rt_extra_args,
        ]
        for w in widgets:
            if isinstance(w, QTextEdit):
                w.textChanged.connect(self.debounced_save)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self.debounced_save)
        for w in [
            self.n_width, self.n_height, self.n_steps, self.n_batch, self.n_seed, self.n_cfg, self.n_guidance,
            self.n_preview_interval, self.e_width, self.e_height, self.e_steps, self.e_batch, self.e_seed,
            self.e_cfg, self.e_img_cfg, self.e_guidance, self.e_strength, self.e_preview_interval, self.rt_threads,
        ]:
            w.valueChanged.connect(self.debounced_save)
        for w in [self.n_sampler, self.n_scheduler, self.n_preview, self.n_aspect_mode, self.e_sampler, self.e_scheduler, self.e_preview, self.rt_rng]:
            w.currentIndexChanged.connect(self.debounced_save)
        for w in [self.e_increase_ref_index, self.e_disable_resize, self.rt_diffusion_fa, self.rt_offload, self.rt_mmap, self.rt_eager, self.rt_vae_tiling, self.rt_disable_metadata, self.rt_verbose, self.rt_framevision_queue]:
            w.toggled.connect(self.debounced_save)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_config)

    def debounced_save(self, *args) -> None:
        if not self._loading:
            self._save_timer.start(350)

    @staticmethod
    def set_combo(combo: QComboBox, value: str) -> None:
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------
    def add_reference_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add reference image",
            str(APP_ROOT / "output"),
            IMAGE_FILE_FILTER,
        )
        for file in files:
            if file not in self.reference_images:
                self.reference_images.append(file)
        self.refresh_reference_list()
        self.save_config()

    def remove_selected_reference(self) -> None:
        rows = sorted([idx.row() for idx in self.ref_list.selectedIndexes()], reverse=True)
        for row in rows:
            if 0 <= row < len(self.reference_images):
                self.reference_images.pop(row)
        self.refresh_reference_list()
        self.save_config()

    def clear_references(self) -> None:
        self.reference_images.clear()
        self.refresh_reference_list()
        self.save_config()

    def refresh_reference_list(self) -> None:
        self.ref_list.clear()
        for path_text in self.reference_images:
            p = Path(path_text)
            item = QListWidgetItem(p.name)
            item.setToolTip(path_text)
            if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                pix = QPixmap(str(p))
                if not pix.isNull():
                    item.setIcon(QIcon(pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.ref_list.addItem(item)

    # ------------------------------------------------------------------
    # Command build / generation
    # ------------------------------------------------------------------
    def update_queue_button_text(self, *args) -> None:
        try:
            use_queue = bool(self.rt_framevision_queue.isChecked())
        except Exception:
            use_queue = False
        try:
            self.n_generate.setText("Add to queue" if use_queue else "Generate")
        except Exception:
            pass
        try:
            self.e_generate.setText("Add edit to queue" if use_queue else "Generate edit")
        except Exception:
            pass

    def build_command(self, mode: str) -> List[str]:
        self.pull_ui_to_config()
        p = self.config["paths"]
        rt = self.config["runtime"]
        section = self.config["normal"] if mode == "normal" else self.config["edit"]

        sd_cli = rel_or_abs(p["sd_cli"])
        model = rel_or_abs(p["turbo_model"] if mode == "normal" else p["edit_model"])
        vae = rel_or_abs(p["vae"])
        llm = rel_or_abs(p["llm"])
        out_dir = rel_or_abs(p["normal_output_subfolder"] if mode == "normal" else p["edit_output_subfolder"])
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"boogu_{mode}_{now_stamp()}_%03d.png"
        preview = out_dir / f"boogu_{mode}_{now_stamp()}_preview.png"

        cmd = [
            str(sd_cli),
            "--diffusion-model", str(model),
            "--llm", str(llm),
            "--vae", str(vae),
            "--vae-format", "flux",
            "--prompt", section["prompt"],
            "--output", str(output),
            "--width", str(section["width"]),
            "--height", str(section["height"]),
            "--steps", str(section["steps"]),
            "--cfg-scale", str(section["cfg_scale"]),
            "--guidance", str(section["guidance"]),
            "--seed", str(section["seed"]),
            "--batch-count", str(section["batch_count"]),
            "--sampling-method", section["sampler"],
            "--rng", rt["rng"],
        ]

        if section.get("negative_prompt"):
            cmd += ["--negative-prompt", section["negative_prompt"]]
        if section.get("scheduler") and section["scheduler"] != "model default":
            cmd += ["--scheduler", section["scheduler"]]
        if section.get("preview") and section["preview"] != "none":
            cmd += ["--preview", section["preview"], "--preview-path", str(preview), "--preview-interval", str(section.get("preview_interval", 1))]

        if mode == "edit":
            llm_vision = rel_or_abs(p["llm_vision"])
            cmd += ["--llm_vision", str(llm_vision)]
            cmd += ["--img-cfg-scale", str(section["img_cfg_scale"]), "--strength", str(section["strength"])]
            for ref in self.reference_images:
                cmd += ["--ref-image", str(Path(ref))]
            if section.get("increase_ref_index"):
                cmd.append("--increase-ref-index")
            if section.get("disable_auto_resize_ref_image"):
                cmd.append("--disable-auto-resize-ref-image")

        # Runtime toggles. No --max-vram is exposed or generated here.
        if rt.get("diffusion_flash_attention_only"):
            cmd.append("--diffusion-fa")
        if rt.get("offload_to_cpu"):
            cmd.append("--offload-to-cpu")
        if rt.get("mmap"):
            cmd.append("--mmap")
        if rt.get("eager_load"):
            cmd.append("--eager-load")
        if rt.get("vae_tiling"):
            cmd.append("--vae-tiling")
        if rt.get("disable_metadata"):
            cmd.append("--disable-image-metadata")
        if rt.get("verbose"):
            cmd.append("-v")
        if int(rt.get("threads", -1)) != -1:
            cmd += ["--threads", str(rt["threads"])]

        extra = rt.get("extra_args", "").strip()
        if extra:
            try:
                cmd += shlex.split(extra, posix=False if os.name == "nt" else True)
            except ValueError:
                cmd += extra.split()

        return cmd

    def validate_command(self, mode: str, cmd: List[str]) -> bool:
        missing = []
        for i, token in enumerate(cmd):
            if token in {"--diffusion-model", "--llm", "--vae", "--llm_vision"} and i + 1 < len(cmd):
                p = Path(cmd[i + 1])
                if not p.exists():
                    missing.append(f"{token}: {p}")
        sd = Path(cmd[0])
        if not sd.exists():
            missing.insert(0, f"sd-cli: {sd}")
        if mode == "edit" and not self.reference_images:
            missing.append("reference image: add at least one image in the Edit tab")
        if missing:
            QMessageBox.warning(self, "Boogu Image", "Missing required files:\n\n" + "\n".join(missing))
            self.log("Generation blocked. Missing required files.")
            for item in missing:
                self.log(f"  {item}")
            return False
        return True

    def generate(self, mode: str) -> None:
        if self.process is not None:
            QMessageBox.information(self, "Boogu Image", "A generation is already running.")
            return
        self.save_config()
        cmd = self.build_command(mode)
        preview = self.format_cmd(cmd)
        if mode == "normal":
            self.n_cmd_preview.setText(preview)
        else:
            self.e_cmd_preview.setText(preview)
        if not self.validate_command(mode, cmd):
            return

        model_path = self.get_arg_value(cmd, "--diffusion-model")
        if model_path and Path(model_path).suffix.lower() == ".gguf":
            self.log("Using GGUF diffusion model. This is experimental for Boogu and depends on the sd-cli build supporting this GGUF tensor layout.")

        use_queue = False
        try:
            use_queue = bool(self.config.get("runtime", {}).get("framevision_queue", True))
        except Exception:
            use_queue = False

        if use_queue:
            try:
                p = self.config.get("paths", {})
                out_key = "normal_output_subfolder" if mode == "normal" else "edit_output_subfolder"
                out_dir = rel_or_abs(p.get(out_key, "output/images/boogu" if mode == "normal" else "output/edits/boogu"))
                prompt_text = (self.config.get("normal", {}) if mode == "normal" else self.config.get("edit", {})).get("prompt", "")
                try:
                    from helpers import queue_adapter as _qa  # type: ignore
                except Exception:
                    import queue_adapter as _qa  # type: ignore
                _qa.enqueue_boogu_generate({
                    "mode": mode,
                    "cmd": cmd,
                    "ffmpeg_cmd": cmd,
                    "cwd": str(APP_ROOT),
                    "output_dir": str(out_dir),
                    "scan_dir": str(out_dir),
                    "scan_ext": ".png",
                    "prompt": str(prompt_text or ""),
                    "width": (self.config.get("normal", {}) if mode == "normal" else self.config.get("edit", {})).get("width", 1024),
                    "height": (self.config.get("normal", {}) if mode == "normal" else self.config.get("edit", {})).get("height", 1024),
                    "steps": (self.config.get("normal", {}) if mode == "normal" else self.config.get("edit", {})).get("steps", 4),
                    "seed": (self.config.get("normal", {}) if mode == "normal" else self.config.get("edit", {})).get("seed", -1),
                    "label": "Boogu Image " + ("edit" if mode == "edit" else "create"),
                })
                self.log("Added Boogu Image job to FrameVision queue.")
                self.show_toast("Added to FrameVision queue")
                self.generated.emit(str(APP_ROOT))
                return
            except Exception as exc:
                self.log(f"Could not add Boogu job to queue: {exc}")
                try:
                    QMessageBox.warning(self, "Boogu Image", f"Could not add job to FrameVision queue:\n{exc}")
                except Exception:
                    pass
                return

        self.log("Starting sd-cli")
        self.log(preview)
        self.set_running(True)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(APP_ROOT))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self.on_process_output(proc))
        proc.finished.connect(lambda exit_code, exit_status: self.on_process_finished(exit_code, exit_status))
        proc.errorOccurred.connect(lambda err: self.on_process_error(proc, err))
        self.process = proc
        proc.start()

    def on_process_output(self, proc: QProcess) -> None:
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            for line in data.rstrip().splitlines():
                self.log(line)

    def on_process_finished(self, exit_code: int, exit_status) -> None:
        self.log(f"sd-cli finished with exit code {exit_code}.")
        self.process = None
        self.set_running(False)
        self.generated.emit(str(APP_ROOT))

    def on_process_error(self, proc: QProcess, err) -> None:
        self.log(f"sd-cli process error: {err}")
        self.process = None
        self.set_running(False)

    def stop_process(self) -> None:
        if self.process is None:
            return
        self.log("Stopping sd-cli")
        self.process.kill()

    def set_running(self, running: bool) -> None:
        self.n_generate.setEnabled(not running)
        self.e_generate.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    @staticmethod
    def get_arg_value(cmd: List[str], key: str) -> str:
        try:
            idx = cmd.index(key)
        except ValueError:
            return ""
        if idx + 1 >= len(cmd):
            return ""
        return cmd[idx + 1]

    @staticmethod
    def format_cmd(cmd: List[str]) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline(cmd)
        return " ".join(shlex.quote(x) for x in cmd)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def browse_path(self, edit: QLineEdit, file_mode: bool, file_filter: str = ALL_FILE_FILTER) -> None:
        start = rel_or_abs(edit.text()) if edit.text().strip() else APP_ROOT
        if file_mode:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", str(start if start.parent.exists() else APP_ROOT), file_filter)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", str(start if start.exists() else APP_ROOT))
        if path:
            edit.setText(nice_path(Path(path)))
            self.save_config()

    def open_folder(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def log(self, message: str) -> None:
        text = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        if hasattr(self, "log_box"):
            self.log_box.appendPlainText(text)
            self.log_box.moveCursor(QTextCursor.End)
        else:
            print(text)


def create_tab(parent: Optional[QWidget] = None) -> BooguUI:
    """Small helper for FrameVision-style dynamic tab loading."""
    return BooguUI(parent)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = BooguUI()
    w.resize(1100, 760)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
