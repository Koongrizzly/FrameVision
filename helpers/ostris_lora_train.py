from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image

from PySide6.QtCore import QProcess, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QDoubleSpinBox, QSplitter, QTabWidget, QTextEdit,
    QToolButton, QVBoxLayout, QWidget
)


def find_app_root() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parents[1]
    if (candidate / "presets").exists():
        return candidate
    return Path.cwd().resolve()


APP_ROOT = find_app_root()
REPO_ROOT = APP_ROOT / "models" / "ostris" / "ai-toolkit"
ENV_ROOT = APP_ROOT / "environments" / ".ostris"
SETTINGS_PATH = APP_ROOT / "presets" / "setsave" / "ostris.json"
LOG_ROOT = APP_ROOT / "logs"
OUTPUT_ROOT = APP_ROOT / "output" / "ostris_lora"
DATASET_ROOT = OUTPUT_ROOT / "_datasets"
CONFIG_ROOT = OUTPUT_ROOT / "_configs"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class ModelPreset:
    label: str
    family: str
    media: str
    arch: str
    default_model: str
    resolution: int
    rank: int
    steps: int
    lr: float
    frames: int = 1
    fps: int = 24
    quantize: bool = True
    qtype: str = "qfloat8"
    notes: str = ""


PRESETS: list[ModelPreset] = [
    ModelPreset(
        "Krea 2 Raw — Exact Identity", "Krea 2", "image", "krea2",
        "krea/Krea-2-Raw", 1024, 64, 2500, 0.0001,
        notes="Train on Krea-2-Raw. The resulting LoRA can be tested on Krea-2-Turbo."
    ),
    ModelPreset(
        "Krea 2 Raw — Style", "Krea 2", "image", "krea2",
        "krea/Krea-2-Raw", 1024, 32, 2000, 0.0001,
        notes="Style preset with lower adapter capacity than Exact Identity."
    ),
    ModelPreset(
        "FLUX.1 Dev — Character", "FLUX", "image", "flux",
        "black-forest-labs/FLUX.1-dev", 1024, 32, 2000, 0.0001,
        notes="Requires access to the gated FLUX.1-dev checkpoint when using Hugging Face."
    ),
    ModelPreset(
        "SDXL — Character or Style", "SDXL", "image", "sdxl",
        "stabilityai/stable-diffusion-xl-base-1.0", 1024, 32, 2000, 0.0001,
    ),
    ModelPreset(
        "LTX 2.3 — Image or Video LoRA", "LTX", "video", "ltx2.3",
        "Lightricks/LTX-2.3", 512, 32, 3000, 0.0001, frames=121, fps=24,
        notes="Choose Images for character or appearance training, or Videos for motion training. Video training is substantially heavier."
    ),
    ModelPreset(
        "Wan 2.2 I2V A14B — Video LoRA", "Wan", "video", "wan22",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 512, 32, 3000, 0.0001, frames=81, fps=16,
    ),
    ModelPreset(
        "ACE-Step 1.5 — Audio LoRA", "ACE-Step", "audio", "acestep1.5",
        "ACE-Step/Ace-Step1.5", 512, 32, 2000, 0.0001,
        notes="Audio datasets should include a matching text caption for each audio file."
    ),
    ModelPreset(
        "Custom AI Toolkit Model", "Custom", "image", "",
        "", 1024, 32, 2000, 0.0001,
        notes="Enter the model path, architecture, and settings manually. Review YAML before training."
    ),
]


def env_python() -> Path:
    return ENV_ROOT / "python.exe" if os.name == "nt" else ENV_ROOT / "bin" / "python"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-") or "lora_job"


def sidecar_path(media: Path) -> Path:
    return media.with_suffix(".txt")


def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if target.resolve() == src.resolve():
        return target
    stem, suffix = src.stem, src.suffix
    index = 2
    while target.exists():
        try:
            if target.stat().st_size == src.stat().st_size:
                return target
        except OSError:
            pass
        target = dst_dir / f"{stem}_{index}{suffix}"
        index += 1
    shutil.copy2(src, target)
    return target


class MediaItemWidget(QWidget):
    removed = Signal(str)

    def __init__(self, path: Path, media_type: str):
        super().__init__()
        self.path = path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        preview = QLabel()
        preview.setFixedSize(88, 66)
        preview.setAlignment(Qt.AlignCenter)
        if media_type == "image":
            pix = QPixmap(str(path))
            if not pix.isNull():
                preview.setPixmap(pix.scaled(88, 66, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                preview.setText("IMAGE")
        else:
            preview.setText(media_type.upper())
        layout.addWidget(preview)

        text = QVBoxLayout()
        name = QLabel(path.name)
        name.setToolTip(str(path))
        text.addWidget(name)
        meta = QLabel(self._metadata(path, media_type))
        meta.setStyleSheet("color: #888;")
        text.addWidget(meta)
        layout.addLayout(text, 1)

        remove = QToolButton()
        remove.setText("Remove")
        remove.clicked.connect(lambda: self.removed.emit(str(self.path)))
        layout.addWidget(remove)

    @staticmethod
    def _metadata(path: Path, media_type: str) -> str:
        try:
            size = path.stat().st_size / (1024 * 1024)
            if media_type == "image":
                with Image.open(path) as im:
                    return f"{im.width} × {im.height}  •  {size:.1f} MB"
            return f"{size:.1f} MB"
        except Exception:
            return ""


class FrameVisionLoraTrainer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("  LoRA Trainer")
        self.resize(1280, 850)
        self.process: Optional[QProcess] = None
        self.media_paths: list[Path] = []
        self.current_dataset_dir: Optional[Path] = None
        self.log_file = LOG_ROOT / f"ostris_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self._build_ui()
        self._load_settings()
        self._preset_changed()
        self._check_backend()

    def _build_ui(self) -> None:
        central = QWidget()
        main = QVBoxLayout(central)
        main.setContentsMargins(14, 14, 14, 14)

        head = QHBoxLayout()
        title = QLabel("  LoRA Trainer")
        f = QFont()
        f.setPointSize(17)
        f.setBold(True)
        title.setFont(f)
        head.addWidget(title)
        head.addStretch(1)

        credit = QLabel("Shell for Ostris AI Toolkit")
        credit.setToolTip(
            "  provides this interface, presets, dataset preparation and job control. "
            "Ostris AI Toolkit performs the underlying model training."
        )
        head.addWidget(credit)
        self.open_repo_btn = QPushButton("Ostris AI Toolkit")
        self.open_repo_btn.clicked.connect(
            lambda: webbrowser.open("https://github.com/ostris/ai-toolkit")
        )
        head.addWidget(self.open_repo_btn)
        main.addLayout(head)

        self.backend_status = QLabel()
        self.backend_status.setWordWrap(True)
        main.addWidget(self.backend_status)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_setup_tab(), "1. Setup")
        self.tabs.addTab(self._build_dataset_tab(), "2. Dataset")
        self.tabs.addTab(self._build_training_tab(), "3. Training")
        self.tabs.addTab(self._build_yaml_tab(), "4. YAML / Advanced")
        self.tabs.addTab(self._build_results_tab(), "5. Results")
        main.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self._save_settings)
        footer.addWidget(self.save_settings_btn)

        self.prepare_btn = QPushButton("Prepare Job")
        self.prepare_btn.clicked.connect(self.prepare_job)
        footer.addWidget(self.prepare_btn)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        footer.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        footer.addWidget(self.stop_btn)

        footer.addStretch(1)
        self.open_output_btn = QPushButton("Open Output")
        self.open_output_btn.clicked.connect(lambda: self.open_path(OUTPUT_ROOT))
        footer.addWidget(self.open_output_btn)
        main.addLayout(footer)

        self.setCentralWidget(central)

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Model and job")
        form = QFormLayout(group)

        self.preset_combo = QComboBox()
        for preset in PRESETS:
            self.preset_combo.addItem(preset.label)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.preset_combo.setToolTip(
            "Select a model-aware starting configuration. All generated settings remain editable."
        )
        form.addRow("Training preset:", self.preset_combo)

        self.job_name = QLineEdit("my_lora")
        self.job_name.setToolTip(
            "Used for the output folder, generated configuration and checkpoint names."
        )
        form.addRow("LoRA name:", self.job_name)

        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText("Example: fvperson")
        self.trigger.setToolTip(
            "Unique token representing the person, style, object or concept. "
            "It is inserted into new captions when requested."
        )
        form.addRow("Trigger word:", self.trigger)

        self.model_path = QLineEdit()
        self.model_path.setToolTip(
            "A Hugging Face model ID or a local model/checkpoint path supported by AI Toolkit."
        )
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path, 1)
        browse_model = QPushButton("Browse")
        browse_model.clicked.connect(self.browse_model)
        model_row.addWidget(browse_model)
        form.addRow("Base model:", model_row)

        self.arch = QLineEdit()
        self.arch.setToolTip(
            "AI Toolkit architecture identifier. Presets fill this automatically."
        )
        form.addRow("Architecture:", self.arch)

        self.notes = QLabel()
        self.notes.setWordWrap(True)
        form.addRow("Preset notes:", self.notes)
        layout.addWidget(group)

        paths = QGroupBox("Standalone paths")
        pform = QFormLayout(paths)
        for label, value in [
            ("Backend repository:", REPO_ROOT),
            ("Conda environment:", ENV_ROOT),
            ("Settings:", SETTINGS_PATH),
            ("Logs:", LOG_ROOT),
            ("LoRA output:", OUTPUT_ROOT),
        ]:
            line = QLineEdit(str(value))
            line.setReadOnly(True)
            pform.addRow(label, line)
        layout.addWidget(paths)
        layout.addStretch(1)
        return page

    def _build_dataset_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        info = QLabel(
            "Add source files for the selected model family. Imported files are copied into a "
            "job-specific dataset folder when the job is prepared. Matching .txt captions are "
            "copied or created beside the media."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset type:"))
        self.dataset_type = QComboBox()
        self.dataset_type.addItems(["Images", "Videos"])
        self.dataset_type.currentIndexChanged.connect(self._dataset_type_changed)
        controls.addWidget(self.dataset_type)
        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        controls.addWidget(self.add_files_btn)
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        controls.addWidget(self.add_folder_btn)
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self.remove_selected)
        controls.addWidget(self.remove_selected_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_media)
        controls.addWidget(self.clear_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter()
        self.media_list = QListWidget()
        self.media_list.currentRowChanged.connect(self._selection_changed)
        splitter.addWidget(self.media_list)

        right = QWidget()
        rlayout = QVBoxLayout(right)
        self.selected_file = QLabel("No file selected")
        self.selected_file.setWordWrap(True)
        rlayout.addWidget(self.selected_file)

        cap_label = QLabel("Caption")
        cap_label.setToolTip(
            "Describe changeable details such as clothing, pose, framing, setting and lighting. "
            "Use the trigger word for the trained identity or concept."
        )
        rlayout.addWidget(cap_label)
        self.caption_edit = QTextEdit()
        self.caption_edit.setPlaceholderText("Caption for the selected file")
        rlayout.addWidget(self.caption_edit, 1)

        cap_buttons = QHBoxLayout()
        save_cap = QPushButton("Save Caption")
        save_cap.clicked.connect(self.save_caption)
        cap_buttons.addWidget(save_cap)
        prepend_trigger = QPushButton("Add Trigger")
        prepend_trigger.clicked.connect(self.add_trigger_to_caption)
        cap_buttons.addWidget(prepend_trigger)
        rlayout.addLayout(cap_buttons)

        self.dataset_summary = QLabel()
        self.dataset_summary.setWordWrap(True)
        rlayout.addWidget(self.dataset_summary)
        splitter.addWidget(right)
        splitter.setSizes([700, 420])
        layout.addWidget(splitter, 1)
        return page

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        settings = QGroupBox("Training settings")
        form = QFormLayout(settings)

        self.resolution = QSpinBox()
        self.resolution.setRange(256, 2048)
        self.resolution.setSingleStep(64)
        self.resolution.setToolTip(
            "Dataset bucket target. Higher values preserve more detail but increase VRAM and training time."
        )
        form.addRow("Resolution:", self.resolution)

        self.rank = QSpinBox()
        self.rank.setRange(4, 256)
        self.rank.setSingleStep(4)
        self.rank.setToolTip(
            "LoRA capacity. Higher rank can encode more identity detail but costs memory and can overfit."
        )
        form.addRow("Rank / alpha:", self.rank)

        self.steps = QSpinBox()
        self.steps.setRange(50, 100000)
        self.steps.setSingleStep(250)
        self.steps.setToolTip(
            "Total optimization steps. Periodic checkpoints allow selecting the strongest result before overtraining."
        )
        form.addRow("Training steps:", self.steps)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(7)
        self.learning_rate.setRange(0.0000001, 0.1)
        self.learning_rate.setSingleStep(0.00001)
        self.learning_rate.setToolTip("Optimizer learning rate.")
        form.addRow("Learning rate:", self.learning_rate)

        self.save_every = QSpinBox()
        self.save_every.setRange(25, 10000)
        self.save_every.setValue(250)
        self.save_every.setToolTip("Creates a checkpoint at this interval.")
        form.addRow("Save every:", self.save_every)

        self.keep_checkpoints = QSpinBox()
        self.keep_checkpoints.setRange(1, 100)
        self.keep_checkpoints.setValue(12)
        self.keep_checkpoints.setToolTip("Maximum periodic checkpoints retained by AI Toolkit.")
        form.addRow("Keep checkpoints:", self.keep_checkpoints)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(1)
        form.addRow("Batch size:", self.batch_size)

        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 32)
        self.grad_accum.setValue(1)
        self.grad_accum.setToolTip("Accumulates gradients without increasing instantaneous VRAM use.")
        form.addRow("Gradient accumulation:", self.grad_accum)

        self.caption_dropout = QDoubleSpinBox()
        self.caption_dropout.setDecimals(2)
        self.caption_dropout.setRange(0.0, 1.0)
        self.caption_dropout.setSingleStep(0.01)
        self.caption_dropout.setValue(0.05)
        form.addRow("Caption dropout:", self.caption_dropout)

        self.frames = QSpinBox()
        self.frames.setRange(1, 721)
        self.frames.setSingleStep(8)
        self.frames.setValue(1)
        self.frames.setToolTip("Video frame count used by compatible video trainers.")
        form.addRow("Video frames:", self.frames)

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(24)
        form.addRow("Video FPS:", self.fps)

        self.quantize = QCheckBox("Quantize base model")
        self.quantize.setChecked(True)
        self.quantize.setToolTip("Reduces VRAM use while loading and training supported models.")
        form.addRow("", self.quantize)

        self.low_vram = QCheckBox("Use low-VRAM behavior")
        self.low_vram.setChecked(True)
        self.low_vram.setToolTip("Recommended when the RTX 3090 also drives the desktop.")
        form.addRow("", self.low_vram)

        self.cache_latents = QCheckBox("Cache latents to disk")
        self.cache_latents.setChecked(True)
        form.addRow("", self.cache_latents)

        self.cache_text = QCheckBox("Cache text embeddings")
        self.cache_text.setChecked(True)
        form.addRow("", self.cache_text)

        self.disable_samples = QCheckBox("Disable sample generation during training")
        self.disable_samples.setChecked(False)
        self.disable_samples.setToolTip(
            "Disabling samples saves time and VRAM. Checkpoints are still saved normally."
        )
        form.addRow("", self.disable_samples)

        layout.addWidget(settings)

        sample_group = QGroupBox("Checkpoint sample prompts")
        sform = QFormLayout(sample_group)
        self.sample_prompts = QPlainTextEdit()
        self.sample_prompts.setPlaceholderText(
            "One prompt per line. Include the trigger word.\n"
            "Example: fvperson, close-up portrait, neutral studio lighting"
        )
        sform.addRow("Prompts:", self.sample_prompts)
        self.sample_every = QSpinBox()
        self.sample_every.setRange(25, 10000)
        self.sample_every.setValue(250)
        sform.addRow("Sample every:", self.sample_every)
        layout.addWidget(sample_group)
        layout.addStretch(1)
        return page

    def _build_yaml_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        info = QLabel(
            "Prepare Job generates the exact YAML passed to AI Toolkit. You may edit it here. "
            "Starting training uses the current YAML text, not hidden values."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.yaml_edit = QPlainTextEdit()
        self.yaml_edit.setPlaceholderText("Generated AI Toolkit YAML")
        layout.addWidget(self.yaml_edit, 1)
        buttons = QHBoxLayout()
        regenerate = QPushButton("Regenerate YAML")
        regenerate.clicked.connect(self.prepare_job)
        buttons.addWidget(regenerate)
        validate = QPushButton("Validate YAML")
        validate.clicked.connect(self.validate_yaml)
        buttons.addWidget(validate)
        export = QPushButton("Export YAML")
        export.clicked.connect(self.export_yaml)
        buttons.addWidget(export)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        refresh = QPushButton("Refresh Results")
        refresh.clicked.connect(self.refresh_results)
        top.addWidget(refresh)
        open_selected = QPushButton("Open Selected Folder")
        open_selected.clicked.connect(self.open_selected_result)
        top.addWidget(open_selected)
        top.addStretch(1)
        layout.addLayout(top)
        self.results_list = QListWidget()
        layout.addWidget(self.results_list, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Training output")
        layout.addWidget(self.log, 2)
        return page

    def current_preset(self) -> ModelPreset:
        return PRESETS[self.preset_combo.currentIndex()]

    def _preset_changed(self) -> None:
        p = self.current_preset()
        self.model_path.setText(p.default_model)
        self.arch.setText(p.arch)
        self.resolution.setValue(p.resolution)
        self.rank.setValue(p.rank)
        self.steps.setValue(p.steps)
        self.learning_rate.setValue(p.lr)
        self.frames.setValue(p.frames)
        self.fps.setValue(p.fps)
        self.quantize.setChecked(p.quantize)
        self.notes.setText(p.notes or "Review generated YAML before training.")
        if p.family == "LTX":
            self.dataset_type.setEnabled(True)
            self.dataset_type.setCurrentIndex(0)
        else:
            self.dataset_type.setEnabled(False)
            self.dataset_type.setCurrentIndex(1 if p.media == "video" else 0)
        video = self.effective_media_type() == "video"
        self.frames.setEnabled(video)
        self.fps.setEnabled(video)
        self.clear_media()
        self._update_dataset_summary()

    def _check_backend(self) -> None:
        repo_ok = (REPO_ROOT / "run.py").exists()
        py_ok = env_python().exists()
        if repo_ok and py_ok:
            self.backend_status.setText(
                f"Backend ready: {REPO_ROOT}  •  Environment: {ENV_ROOT}"
            )
            self.backend_status.setStyleSheet("color: #3a8f55;")
            self.start_btn.setEnabled(True)
        else:
            self.backend_status.setText(
                "Backend is not installed. Run presets/extra_env/ostris_install.py first."
            )
            self.backend_status.setStyleSheet("color: #a25b2a;")
            self.start_btn.setEnabled(False)

    def browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select local model folder", str(APP_ROOT / "models"))
        if path:
            self.model_path.setText(path)

    def effective_media_type(self) -> str:
        p = self.current_preset()
        if p.family == "LTX":
            return "image" if self.dataset_type.currentIndex() == 0 else "video"
        return p.media

    def _dataset_type_changed(self) -> None:
        if not hasattr(self, "dataset_type"):
            return
        p = self.current_preset()
        is_ltx = p.family == "LTX"
        self.dataset_type.setEnabled(is_ltx)
        video = self.effective_media_type() == "video"
        self.frames.setEnabled(video)
        self.fps.setEnabled(video)
        self.clear_media()
        self._update_dataset_summary()

    def accepted_extensions(self) -> set[str]:
        media = self.effective_media_type()
        return IMAGE_EXTS if media == "image" else VIDEO_EXTS if media == "video" else AUDIO_EXTS

    def file_filter(self) -> str:
        media = self.effective_media_type()
        if media == "image":
            return "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        if media == "video":
            return "Videos (*.mp4 *.mov *.mkv *.webm *.avi)"
        return "Audio (*.wav *.flac *.mp3 *.ogg *.m4a)"

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add dataset files", str(APP_ROOT), self.file_filter())
        self._add_paths([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add dataset folder", str(APP_ROOT))
        if not folder:
            return
        exts = self.accepted_extensions()
        self._add_paths(sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts))

    def _add_paths(self, paths: list[Path]) -> None:
        exts = self.accepted_extensions()
        known = {p.resolve() for p in self.media_paths if p.exists()}
        for path in paths:
            if path.suffix.lower() not in exts or not path.exists():
                continue
            if path.resolve() not in known:
                self.media_paths.append(path)
                known.add(path.resolve())
        self._refresh_media_list()

    def _refresh_media_list(self) -> None:
        current = self.media_list.currentRow()
        self.media_list.clear()
        media_type = self.effective_media_type()
        for path in self.media_paths:
            item = QListWidgetItem()
            item.setSizeHint(QSize(400, 78))
            self.media_list.addItem(item)
            widget = MediaItemWidget(path, media_type)
            widget.removed.connect(self.remove_path)
            self.media_list.setItemWidget(item, widget)
        if self.media_paths:
            self.media_list.setCurrentRow(min(max(current, 0), len(self.media_paths) - 1))
        self._update_dataset_summary()

    def remove_path(self, path_str: str) -> None:
        self.media_paths = [p for p in self.media_paths if str(p) != path_str]
        self._refresh_media_list()

    def remove_selected(self) -> None:
        row = self.media_list.currentRow()
        if 0 <= row < len(self.media_paths):
            self.media_paths.pop(row)
            self._refresh_media_list()

    def clear_media(self) -> None:
        self.media_paths.clear()
        if hasattr(self, "media_list"):
            self.media_list.clear()
        if hasattr(self, "caption_edit"):
            self.caption_edit.clear()
        if hasattr(self, "selected_file"):
            self.selected_file.setText("No file selected")
        self._update_dataset_summary()

    def _selection_changed(self, row: int) -> None:
        if not (0 <= row < len(self.media_paths)):
            return
        path = self.media_paths[row]
        self.selected_file.setText(str(path))
        cap = sidecar_path(path)
        self.caption_edit.setPlainText(cap.read_text(encoding="utf-8") if cap.exists() else "")

    def save_caption(self) -> None:
        row = self.media_list.currentRow()
        if not (0 <= row < len(self.media_paths)):
            return
        sidecar_path(self.media_paths[row]).write_text(
            self.caption_edit.toPlainText().strip(), encoding="utf-8"
        )
        self._update_dataset_summary()

    def add_trigger_to_caption(self) -> None:
        trigger = self.trigger.text().strip()
        if not trigger:
            QMessageBox.warning(self, "Trigger word missing", "Enter a trigger word first.")
            return
        current = self.caption_edit.toPlainText().strip()
        if trigger.lower() not in current.lower():
            self.caption_edit.setPlainText(f"{trigger}, {current}".rstrip(", "))
        self.save_caption()

    def _update_dataset_summary(self) -> None:
        if not hasattr(self, "dataset_summary"):
            return
        captions = sum(sidecar_path(p).exists() and bool(sidecar_path(p).read_text(
            encoding="utf-8", errors="ignore").strip()) for p in self.media_paths)
        self.dataset_summary.setText(
            f"{len(self.media_paths)} source files  •  {captions} captions present  •  "
            f"Expected media: {self.effective_media_type()}"
        )

    def _settings_dict(self) -> dict[str, Any]:
        return {
            "preset_index": self.preset_combo.currentIndex(),
            "job_name": self.job_name.text(),
            "trigger": self.trigger.text(),
            "model_path": self.model_path.text(),
            "arch": self.arch.text(),
            "resolution": self.resolution.value(),
            "rank": self.rank.value(),
            "steps": self.steps.value(),
            "learning_rate": self.learning_rate.value(),
            "save_every": self.save_every.value(),
            "keep_checkpoints": self.keep_checkpoints.value(),
            "batch_size": self.batch_size.value(),
            "grad_accum": self.grad_accum.value(),
            "caption_dropout": self.caption_dropout.value(),
            "frames": self.frames.value(),
            "fps": self.fps.value(),
            "quantize": self.quantize.isChecked(),
            "low_vram": self.low_vram.isChecked(),
            "cache_latents": self.cache_latents.isChecked(),
            "cache_text": self.cache_text.isChecked(),
            "disable_samples": self.disable_samples.isChecked(),
            "sample_prompts": self.sample_prompts.toPlainText(),
            "sample_every": self.sample_every.value(),
            "dataset_type": self.dataset_type.currentText(),
            "media_paths": [str(p) for p in self.media_paths],
        }

    def _save_settings(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self._settings_dict(), indent=2), encoding="utf-8")
        self.append_log(f"Settings saved to {SETTINGS_PATH}")

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        index = int(data.get("preset_index", 0))
        self.preset_combo.setCurrentIndex(max(0, min(index, len(PRESETS) - 1)))
        fields = [
            ("job_name", self.job_name, "setText"),
            ("trigger", self.trigger, "setText"),
            ("model_path", self.model_path, "setText"),
            ("arch", self.arch, "setText"),
            ("resolution", self.resolution, "setValue"),
            ("rank", self.rank, "setValue"),
            ("steps", self.steps, "setValue"),
            ("learning_rate", self.learning_rate, "setValue"),
            ("save_every", self.save_every, "setValue"),
            ("keep_checkpoints", self.keep_checkpoints, "setValue"),
            ("batch_size", self.batch_size, "setValue"),
            ("grad_accum", self.grad_accum, "setValue"),
            ("caption_dropout", self.caption_dropout, "setValue"),
            ("frames", self.frames, "setValue"),
            ("fps", self.fps, "setValue"),
            ("sample_every", self.sample_every, "setValue"),
        ]
        for key, widget, method in fields:
            if key in data:
                getattr(widget, method)(data[key])
        for key, widget in [
            ("quantize", self.quantize), ("low_vram", self.low_vram),
            ("cache_latents", self.cache_latents), ("cache_text", self.cache_text),
            ("disable_samples", self.disable_samples),
        ]:
            if key in data:
                widget.setChecked(bool(data[key]))
        self.sample_prompts.setPlainText(data.get("sample_prompts", ""))
        if self.current_preset().family == "LTX":
            self.dataset_type.setCurrentText(data.get("dataset_type", "Images"))
        self.media_paths = [Path(p) for p in data.get("media_paths", []) if Path(p).exists()]
        self._refresh_media_list()

    def prepare_dataset(self) -> Path:
        job = slugify(self.job_name.text())
        dataset_dir = DATASET_ROOT / job
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not self.media_paths:
            raise ValueError("Add at least one dataset file.")
        trigger = self.trigger.text().strip()
        for source in self.media_paths:
            copied = safe_copy(source, dataset_dir)
            source_cap = sidecar_path(source)
            target_cap = sidecar_path(copied)
            if source_cap.exists():
                shutil.copy2(source_cap, target_cap)
            elif trigger:
                target_cap.write_text(trigger, encoding="utf-8")
            else:
                target_cap.touch()
        self.current_dataset_dir = dataset_dir
        return dataset_dir

    def build_config(self, dataset_dir: Path) -> dict[str, Any]:
        p = self.current_preset()
        dataset_media = self.effective_media_type()
        job = slugify(self.job_name.text())
        output_dir = OUTPUT_ROOT / p.family.replace(" ", "_").lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_lines = [line.strip() for line in self.sample_prompts.toPlainText().splitlines() if line.strip()]
        samples = [{"prompt": line} for line in prompt_lines]
        process: dict[str, Any] = {
            "type": "diffusion_trainer",
            "training_folder": str(output_dir),
            "sqlite_db_path": str(OUTPUT_ROOT / "aitk_db.db"),
            "device": "cuda",
            "trigger_word": self.trigger.text().strip() or None,
            "network": {
                "type": "lora",
                "linear": self.rank.value(),
                "linear_alpha": self.rank.value(),
                "network_kwargs": {"ignore_if_contains": []},
            },
            "save": {
                "dtype": "bf16",
                "save_every": self.save_every.value(),
                "max_step_saves_to_keep": self.keep_checkpoints.value(),
                "save_format": "diffusers",
                "push_to_hub": False,
            },
            "datasets": [{
                "folder_path": str(dataset_dir),
                "caption_ext": "txt",
                "caption_dropout_rate": self.caption_dropout.value(),
                "cache_latents_to_disk": self.cache_latents.isChecked(),
                "is_reg": False,
                "network_weight": 1,
                "resolution": [self.resolution.value()],
                "num_repeats": 1,
                "controls": [],
            }],
            "train": {
                "batch_size": self.batch_size.value(),
                "steps": self.steps.value(),
                "gradient_accumulation": self.grad_accum.value(),
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch",
                "optimizer": "adamw8bit",
                "timestep_type": "weighted",
                "content_or_style": "balanced",
                "optimizer_params": {"weight_decay": 0.0001},
                "cache_text_embeddings": self.cache_text.isChecked(),
                "lr": self.learning_rate.value(),
                "skip_first_sample": False,
                "disable_sampling": self.disable_samples.isChecked(),
                "dtype": "bf16",
                "loss_type": "mse",
            },
            "model": {
                "name_or_path": self.model_path.text().strip(),
                "quantize": self.quantize.isChecked(),
                "qtype": p.qtype,
                "quantize_te": self.quantize.isChecked(),
                "qtype_te": "qfloat8",
                "arch": self.arch.text().strip() or None,
                "low_vram": self.low_vram.isChecked(),
                "model_kwargs": {},
                "layer_offloading": self.low_vram.isChecked(),
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": self.sample_every.value(),
                "width": self.resolution.value(),
                "height": self.resolution.value(),
                "samples": samples,
                "neg": "",
                "seed": 42,
                "walk_seed": True,
                "guidance_scale": 3.5,
                "sample_steps": 28,
                "num_frames": self.frames.value() if p.media == "video" else 1,
                "fps": self.fps.value(),
            },
            "logging": {"log_every": 1, "use_ui_logger": True},
        }

        if p.media == "video":
            ds = process["datasets"][0]
            if dataset_media == "image":
                ds.update({
                    "shrink_video_to_frames": True,
                    "num_frames": 1,
                    "auto_frame_count": False,
                    "do_i2v": False,
                })
            else:
                ds.update({
                    "shrink_video_to_frames": True,
                    "num_frames": self.frames.value(),
                    "auto_frame_count": False,
                    "do_i2v": "I2V" in p.label,
                    "fps": self.fps.value(),
                })
            if os.name == "nt":
                process["train"]["num_dataloader_workers"] = 0
        elif p.media == "audio":
            process["datasets"][0]["dataset_type"] = "audio"

        # Remove null arch for custom configs if user intentionally leaves it blank.
        if process["model"]["arch"] is None:
            process["model"].pop("arch")

        return {
            "job": "extension",
            "config": {
                "name": job,
                "process": [process],
            },
            "meta": {
                "name": "[name]",
                "version": "1.0",
                "frontend": "  LoRA Trainer",
                "backend_credit": "Ostris AI Toolkit",
            },
        }

    def prepare_job(self) -> None:
        try:
            dataset = self.prepare_dataset()
            config = self.build_config(dataset)
            rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            self.yaml_edit.setPlainText(rendered)
            CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            path = CONFIG_ROOT / f"{slugify(self.job_name.text())}.yaml"
            path.write_text(rendered, encoding="utf-8")
            self._save_settings()
            self.append_log(f"Prepared dataset: {dataset}")
            self.append_log(f"Prepared config: {path}")
            self.tabs.setCurrentIndex(3)
        except Exception as exc:
            QMessageBox.critical(self, "Could not prepare job", str(exc))

    def validate_yaml(self) -> bool:
        try:
            data = yaml.safe_load(self.yaml_edit.toPlainText())
            if not isinstance(data, dict) or "config" not in data:
                raise ValueError("YAML does not contain an AI Toolkit config section.")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid YAML", str(exc))
            return False
        QMessageBox.information(self, "Valid YAML", "The YAML is syntactically valid.")
        return True

    def export_yaml(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export AI Toolkit YAML",
            str(CONFIG_ROOT / f"{slugify(self.job_name.text())}.yaml"),
            "YAML (*.yaml *.yml)"
        )
        if filename:
            Path(filename).write_text(self.yaml_edit.toPlainText(), encoding="utf-8")

    def start_training(self) -> None:
        if self.process:
            return
        if not (REPO_ROOT / "run.py").exists() or not env_python().exists():
            QMessageBox.critical(
                self, "Backend missing",
                "Install AI Toolkit with presets/extra_env/ostris_install.py first."
            )
            return
        if not self.yaml_edit.toPlainText().strip():
            self.prepare_job()
            if not self.yaml_edit.toPlainText().strip():
                return
        try:
            yaml.safe_load(self.yaml_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid YAML", str(exc))
            return

        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_ROOT / f"{slugify(self.job_name.text())}.yaml"
        config_path.write_text(self.yaml_edit.toPlainText(), encoding="utf-8")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        env = self.process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("HF_HUB_DISABLE_TELEMETRY", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_training_output)
        self.process.finished.connect(self._training_finished)
        self.progress.setRange(0, 0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setCurrentIndex(4)
        self.append_log(f"Starting: {env_python()} run.py {config_path}")
        self.process.start(str(env_python()), ["run.py", str(config_path)])

    def _read_training_output(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(text)
        # Extract common step formats for a determinate progress bar.
        match = re.findall(r"(?:step|steps?)\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)", text, re.I)
        if match:
            current, total = map(int, match[-1])
            self.progress.setRange(0, total)
            self.progress.setValue(current)

    def _training_finished(self, code: int, _status) -> None:
        self._read_training_output()
        self.process = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if code == 0 else 0)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(f"Training process exited with code {code}.")
        self.refresh_results()
        if code == 0:
            QMessageBox.information(self, "Training completed", "AI Toolkit finished the job.")
        else:
            QMessageBox.critical(
                self, "Training stopped or failed",
                f"The training process exited with code {code}. Check the log for the underlying error."
            )

    def stop_training(self) -> None:
        if not self.process:
            return
        answer = QMessageBox.question(
            self, "Stop training",
            "Stop the training process?\n\nDo not stop while AI Toolkit is writing a checkpoint, "
            "because that checkpoint may be incomplete."
        )
        if answer != QMessageBox.Yes:
            return
        self.append_log("Stop requested.")
        self.process.terminate()
        QTimer.singleShot(8000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self.append_log("Training did not exit after terminate; killing process.")
            self.process.kill()

    def append_log(self, text: str) -> None:
        text = text.rstrip()
        if not text:
            return
        self.log.appendPlainText(text)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text + "\n")

    def refresh_results(self) -> None:
        self.results_list.clear()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        dirs = [p for p in OUTPUT_ROOT.rglob("*") if p.is_dir() and not p.name.startswith("_")]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in dirs[:100]:
            # Only show folders that contain likely model/checkpoint outputs.
            files = list(path.glob("*.safetensors")) + list(path.glob("*.pt")) + list(path.glob("*.bin"))
            if files or path.parent == OUTPUT_ROOT:
                item = QListWidgetItem(str(path.relative_to(OUTPUT_ROOT)))
                item.setData(Qt.UserRole, str(path))
                self.results_list.addItem(item)

    def open_selected_result(self) -> None:
        item = self.results_list.currentItem()
        if item:
            self.open_path(Path(item.data(Qt.UserRole)))

    @staticmethod
    def open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(path.as_uri())


if __name__ == "__main__":
    for required in [LOG_ROOT, OUTPUT_ROOT, DATASET_ROOT, CONFIG_ROOT, SETTINGS_PATH.parent]:
        required.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    window = FrameVisionLoraTrainer()
    window.show()
    sys.exit(app.exec())
