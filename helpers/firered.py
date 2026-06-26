from __future__ import annotations

import json
import random
import os
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QProcess, Qt, Signal, QSize
from PySide6.QtGui import QTextCursor
from PySide6.QtGui import QAction, QIntValidator, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _norm(path: str) -> str:
    if not path:
        return ""
    return os.path.normpath(os.path.expandvars(os.path.expanduser(path.strip().strip('"'))))


def _shared_model_path(root: Path, filename: str) -> Path:
    return root / "models" / "shared" / filename

def _resolve_shared_model_path(root: Path, value: str) -> str:
    value = _norm(value)
    if value and os.path.isfile(value):
        return value
    name = os.path.basename(value)
    if name == "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf":
        shared = _shared_model_path(root, name)
        if shared.exists():
            return str(shared)
    return value

def _quote(path: str) -> str:
    if not path:
        return '""'
    return f'"{path}"' if (" " in path or "(" in path or ")" in path) else path



def _iter_candidates(root: Path, patterns: List[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    return found



def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _make_thumb(path: str, size: int = 56) -> Optional[QIcon]:
    try:
        pix = QPixmap(path)
        if pix.isNull():
            return None
        pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(pix)
    except Exception:
        return None


@dataclass
class FireRedCommand:
    exe: str
    args: List[str]
    output_file: str

    def as_shell(self) -> str:
        return " ".join([_quote(self.exe)] + [_quote(a) for a in self.args])


# -----------------------------------------------------------------------------
# Collapsible container
# -----------------------------------------------------------------------------


class CollapsibleBox(QWidget):
    def __init__(self, title: str, expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggled)

        self._header_line = QFrame()
        self._header_line.setFrameShape(QFrame.HLine)
        self._header_line.setFrameShadow(QFrame.Sunken)

        self._content = QWidget()
        self._content.setVisible(expanded)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 6, 0, 0)
        self._content_layout.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._toggle)
        top.addWidget(self._header_line)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(top)
        root.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


# -----------------------------------------------------------------------------
# Main pane
# -----------------------------------------------------------------------------


class FireRedPane(QWidget):
    """
    Standalone PySide6 pane for FireRed-Image-Edit-1.1 GGUF via stable-diffusion.cpp's sd-cli.exe.

    Assumptions:
    - This file lives at <framevision_root>/helpers/firered.py
    - The FireRed GGUF lives under <framevision_root>/models/FireRed-Image-Edit-1.1/
    - sd-cli.exe may live in the FrameVision root, /bin, /bin/Release, or /presets/bin
    - A matching Qwen-image VAE + LLM/text-encoder must also be supplied.
    """

    result_ready = Signal(str)
    run_started = Signal(str)
    run_finished = Signal(int, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.framevision_root = Path(__file__).resolve().parent.parent
        self.state_path = self.framevision_root / "presets" / "setsave" / "firered.json"
        self.process: Optional[QProcess] = None
        self._loading_state = True
        self._build_ui()
        self._wire_signals()
        self._apply_defaults()
        self._load_state()
        self._auto_detect_paths(log_it=False)
        self._finalize_startup_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(scroll, 1)

        self.bottom_bar = QWidget()
        self.bottom_bar_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_bar_layout.setSpacing(8)
        self.bottom_bar_layout.addStretch(1)
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setMinimumHeight(40)
        self.bottom_bar_layout.addWidget(self.btn_generate)
        root.addWidget(self.bottom_bar, 0)

        page = QWidget()
        scroll.setWidget(page)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        title = QLabel("FireRed Image Edit 1.1 GGUF")
        title.setObjectName("fireredTitle")
        f = title.font()
        f.setPointSize(max(f.pointSize(), 12))
        f.setBold(True)
        title.setFont(f)
        page_layout.addWidget(title)

 #       subtitle = QLabel(
  #          "Instruction-based image editing panel for FireRed 1.1 GGUF. "
   #         "Supports one or more reference images, prompt-driven edits, command preview, and external sd-cli execution."
    #    )
     #   subtitle.setWordWrap(True)
      #  page_layout.addWidget(subtitle)

        main_split = QSplitter(Qt.Vertical)
        main_split.setChildrenCollapsible(False)
        page_layout.addWidget(main_split, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        main_split.addWidget(top)

        # Input images
        images_box = QGroupBox("Input images")
        images_layout = QVBoxLayout(images_box)
        images_layout.setSpacing(8)

        images_tip = QLabel(
            "Use dedicated slots: image 1 is the base image that gets edited, image 2 is the donor face/style/object reference, and image 3 can be used for an extra object/background helper. Only slots 2 and 3 are sent as auxiliary references."
        )
        images_tip.setWordWrap(True)
        images_layout.addWidget(images_tip)

        self.main_image_path = ""
        self.ref_image2_path = ""
        self.ref_image3_path = ""

        self.main_image_slot = self._make_image_slot(
            "1) Base image",
            "This is the image FireRed should directly edit. It will be passed as the init image.",
            "main",
        )
        self.ref_image2_slot = self._make_image_slot(
            "2) Reference image 1",
            "Use this for a face swap, identity hint, style donor, or another strong secondary reference.",
            "ref2",
        )
        self.ref_image3_slot = self._make_image_slot(
            "3) Reference image 2",
            "Optional third image for objects, clothing, props, environment, or background guidance.",
            "ref3",
        )
        images_layout.addWidget(self.main_image_slot)
        images_layout.addWidget(self.ref_image2_slot)
        images_layout.addWidget(self.ref_image3_slot)
        top_layout.addWidget(images_box)

        # Prompt + settings
        edit_box = QGroupBox("Edit settings")
        edit_layout = QVBoxLayout(edit_box)
        edit_layout.setSpacing(8)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the edit you want. Example: change the outfit to a futuristic silver jacket, keep the face and pose consistent.")
        self.prompt_edit.setFixedHeight(100)
        edit_layout.addWidget(QLabel("Prompt"))
        edit_layout.addWidget(self.prompt_edit)

        self.negative_edit = QPlainTextEdit()
        self.negative_edit.setPlaceholderText("Optional negatives, one line or comma-separated. Leave empty if you do not want negatives.")
        self.negative_edit.setFixedHeight(52)
        edit_layout.addWidget(QLabel("Negative prompt (optional)"))
        edit_layout.addWidget(self.negative_edit)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 4096)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(1024)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 4096)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1024)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(8)

        self.cfg_spin = QSpinBox()
        self.cfg_spin.setRange(1, 100)
        self.cfg_spin.setValue(4)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setValue(0.75)
        self.strength_spin.setToolTip("How strongly the edit should change the source image(s). Lower keeps more of the original; higher changes more.")

        self.seed_edit = QLineEdit("0")
        self.seed_edit.setValidator(QIntValidator(0, 2147483647, self))
        self.chk_random_seed = QCheckBox("Random seed")
        self.chk_random_seed.setChecked(True)
        self.chk_random_seed.setToolTip("When enabled, a new real seed is generated at the start of each run or queue job and written into the seed box.")

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems([
            "euler",
            "euler_a",
            "heun",
            "dpm2",
            "dpm++2m",
            "dpm++2mv2",
            "lcm",
        ])
        self.sampler_combo.setCurrentText("euler")

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "webp"])

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(1)

        self.lora_combo = QComboBox()
        self.lora_combo.setEditable(True)
        self.lora_combo.addItem("")
        self.lora_combo.setToolTip("Optional LoRA file. Auto-detected from the LoRA folder, but you can override it.")
        self.selected_lora_value = QLabel("[none]")
        self.selected_lora_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.selected_lora_value.setWordWrap(True)
        self.selected_lora_value.setToolTip("Currently selected LoRA file name.")

        self.lora_strength_spin = QDoubleSpinBox()
        self.lora_strength_spin.setRange(0.0, 2.0)
        self.lora_strength_spin.setDecimals(2)
        self.lora_strength_spin.setSingleStep(0.05)
        self.lora_strength_spin.setValue(1.00)
        self.lora_strength_spin.setToolTip("Weight used in the injected <lora:name:weight> tag.")

        self.prefix_edit = QLineEdit("firered")
        self.output_dir_edit = QLineEdit()
        self.btn_browse_output = QPushButton("Browse")

        self.chk_offload_cpu = QCheckBox("Offload to CPU")
        self.chk_offload_cpu.setChecked(True)
        self.chk_flash_attn = QCheckBox("Diffusion FA")
        self.chk_flash_attn.setChecked(True)
        self.chk_vae_tiling = QCheckBox("VAE tiling")
        self.chk_verbose = QCheckBox("Verbose logs")
        self.chk_verbose.setChecked(True)
        self.chk_keep_aspect = QCheckBox("Auto-fit to first input image")
        self.chk_keep_aspect.setChecked(True)
        self.chk_reuse_last = QCheckBox("Reuse last output as first input")

        row = 0
        grid.addWidget(QLabel("Width"), row, 0)
        grid.addWidget(self.width_spin, row, 1)
        grid.addWidget(QLabel("Height"), row, 2)
        grid.addWidget(self.height_spin, row, 3)
        row += 1
        grid.addWidget(QLabel("Steps"), row, 0)
        grid.addWidget(self.steps_spin, row, 1)
        grid.addWidget(QLabel("CFG"), row, 2)
        grid.addWidget(self.cfg_spin, row, 3)
        row += 1
        grid.addWidget(QLabel("Strength"), row, 0)
        grid.addWidget(self.strength_spin, row, 1)
        grid.addWidget(QLabel("Seed"), row, 2)
        seed_wrap = QWidget()
        seed_lay = QHBoxLayout(seed_wrap)
        seed_lay.setContentsMargins(0, 0, 0, 0)
        seed_lay.setSpacing(6)
        seed_lay.addWidget(self.seed_edit, 1)
        seed_lay.addWidget(self.chk_random_seed)
        grid.addWidget(seed_wrap, row, 3)
        row += 1
        grid.addWidget(QLabel("Sampler"), row, 0)
        grid.addWidget(self.sampler_combo, row, 1)
        row += 1
        grid.addWidget(QLabel("Batch"), row, 0)
        grid.addWidget(self.batch_spin, row, 1)
        grid.addWidget(QLabel("Format"), row, 2)
        grid.addWidget(self.format_combo, row, 3)
        row += 1
        grid.addWidget(QLabel("LoRA weight"), row, 0)
        grid.addWidget(self.lora_strength_spin, row, 1)
        grid.addWidget(QLabel("Selected LoRA"), row, 2)
        grid.addWidget(self.selected_lora_value, row, 3)
        row += 1
        grid.addWidget(QLabel("File prefix"), row, 0)
        grid.addWidget(self.prefix_edit, row, 1)
        grid.addWidget(QLabel("Output folder"), row, 2)
        out_wrap = QWidget()
        out_lay = QHBoxLayout(out_wrap)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(6)
        out_lay.addWidget(self.output_dir_edit, 1)
        out_lay.addWidget(self.btn_browse_output)
        grid.addWidget(out_wrap, row, 3)
        row += 1

        flags_wrap = QWidget()
        flags_lay = QGridLayout(flags_wrap)
        flags_lay.setContentsMargins(0, 0, 0, 0)
        flags_lay.addWidget(self.chk_offload_cpu, 0, 0)
        flags_lay.addWidget(self.chk_flash_attn, 0, 1)
        flags_lay.addWidget(self.chk_vae_tiling, 0, 2)
        flags_lay.addWidget(self.chk_verbose, 1, 0)
        flags_lay.addWidget(self.chk_keep_aspect, 1, 1)
        flags_lay.addWidget(self.chk_reuse_last, 1, 2)
        edit_layout.addLayout(grid)
        edit_layout.addWidget(flags_wrap)
        top_layout.addWidget(edit_box)

        # Run buttons
        run_bar = QHBoxLayout()
        self.btn_autodetect = QPushButton("Auto-detect paths")
        self.btn_preview = QPushButton("Preview command")
        self.btn_run = QPushButton("Run FireRed")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        run_bar.addStretch(1)
        run_bar.addWidget(self.btn_run)
        run_bar.addWidget(self.btn_stop)
        top_layout.addLayout(run_bar)

        # Bottom area
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        main_split.addWidget(bottom)
        main_split.setStretchFactor(0, 5)
        main_split.setStretchFactor(1, 3)

        self.paths_box = CollapsibleBox("Paths, executables and folders", expanded=False)
        self._build_paths_box()
        bottom_layout.addWidget(self.paths_box)

        self.logs_box = CollapsibleBox("Logs", expanded=False)
        self._build_logs_box()
        bottom_layout.addWidget(self.logs_box)

    def _build_paths_box(self) -> None:
        lay = self.paths_box.content_layout()

#•        info = QLabel(
  #          "Use it when auto-detection misses something or when you want to override the default executable, model, VAE, LLM, LoRA, FFmpeg or output paths."
  #      )
   #     info.setWordWrap(True)
    #    lay.addWidget(info)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.sdcli_edit = QLineEdit()
        self.ffmpeg_edit = QLineEdit()
        self.ffprobe_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.vae_edit = QLineEdit()
        self.llm_edit = QLineEdit()
        self.lora_dir_edit = QLineEdit(str(self.framevision_root / "models" / "lora"))
        self.root_edit = QLineEdit(str(self.framevision_root))
        self.models_root_edit = QLineEdit(str(self.framevision_root / "models"))

        form.addRow("FrameVision root", self._path_row(self.root_edit, dir_mode=True))
        form.addRow("Models folder", self._path_row(self.models_root_edit, dir_mode=True))
        form.addRow("LoRA folder", self._path_row(self.lora_dir_edit, dir_mode=True))
        form.addRow("sd-cli.exe", self._path_row(self.sdcli_edit, file_mode=True, name_filters="Executables (*.exe);;All files (*)"))
        form.addRow("ffmpeg.exe", self._path_row(self.ffmpeg_edit, file_mode=True, name_filters="Executables (*.exe);;All files (*)"))
        form.addRow("ffprobe.exe", self._path_row(self.ffprobe_edit, file_mode=True, name_filters="Executables (*.exe);;All files (*)"))
        form.addRow("FireRed GGUF", self._path_row(self.model_edit, file_mode=True, name_filters="GGUF (*.gguf);;All files (*)"))
        form.addRow("Qwen-image VAE", self._path_row(self.vae_edit, file_mode=True, name_filters="Model files (*.safetensors *.gguf *.bin);;All files (*)"))
        form.addRow("Qwen-image LLM", self._path_row(self.llm_edit, file_mode=True, name_filters="Model files (*.gguf *.safetensors *.bin);;All files (*)"))
        form.addRow("LoRA file", self._lora_path_row())
        lay.addLayout(form)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_autodetect)
        actions.addStretch(1)
        lay.addLayout(actions)

        note = QLabel(
            "FireRed GGUF still needs a compatible VAE and Qwen-image text/vision encoder path. LoRA auto-detection scans your LoRA folder and fills the dropdown, but manual override stays available. "
            "The panel tries to find these automatically, but manual override stays available here."
        )
        note.setWordWrap(True)
        lay.addWidget(note)

    def _build_logs_box(self) -> None:
        lay = self.logs_box.content_layout()

        tools = QHBoxLayout()
        self.btn_clear_logs = QPushButton("Clear logs")
        self.btn_copy_logs = QPushButton("Copy logs")
        tools.addWidget(self.btn_clear_logs)
        tools.addWidget(self.btn_copy_logs)
        tools.addStretch(1)
        tools.addWidget(self.btn_preview)
        lay.addLayout(tools)

        cmd_box = QGroupBox("Command preview")
        cmd_layout = QVBoxLayout(cmd_box)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(120)
        cmd_layout.addWidget(self.command_preview)
        lay.addWidget(cmd_box)

        self.logs_edit = QPlainTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setMinimumHeight(220)
        lay.addWidget(self.logs_edit)

    def _path_row(
        self,
        line_edit: QLineEdit,
        *,
        file_mode: bool = False,
        dir_mode: bool = False,
        name_filters: str = "All files (*)",
    ) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line_edit, file_mode=file_mode, dir_mode=dir_mode, name_filters=name_filters))
        h.addWidget(line_edit, 1)
        h.addWidget(btn)
        return w

    def _lora_path_row(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        btn_browse = QPushButton("Browse")
        btn_refresh = QPushButton("Refresh")
        btn_browse.clicked.connect(self._browse_lora)
        btn_refresh.clicked.connect(self._refresh_lora_choices)
        h.addWidget(self.lora_combo, 1)
        h.addWidget(btn_browse)
        h.addWidget(btn_refresh)
        return w

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        self.btn_main_image_browse.clicked.connect(lambda: self._choose_image_for_slot("main"))
        self.btn_main_image_preview.clicked.connect(lambda: self._preview_slot_image("main"))
        self.btn_main_image_clear.clicked.connect(lambda: self._clear_image_slot("main"))
        self.btn_ref2_image_browse.clicked.connect(lambda: self._choose_image_for_slot("ref2"))
        self.btn_ref2_image_preview.clicked.connect(lambda: self._preview_slot_image("ref2"))
        self.btn_ref2_image_clear.clicked.connect(lambda: self._clear_image_slot("ref2"))
        self.btn_ref3_image_browse.clicked.connect(lambda: self._choose_image_for_slot("ref3"))
        self.btn_ref3_image_preview.clicked.connect(lambda: self._preview_slot_image("ref3"))
        self.btn_ref3_image_clear.clicked.connect(lambda: self._clear_image_slot("ref3"))
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_autodetect.clicked.connect(lambda: self._auto_detect_paths(log_it=True))
        self.btn_preview.clicked.connect(self._refresh_preview)
        self.btn_run.clicked.connect(self._run)
        self.btn_generate.clicked.connect(self._generate_queue)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_clear_logs.clicked.connect(self.logs_edit.clear)
        self.btn_copy_logs.clicked.connect(self._copy_logs)
        self.chk_random_seed.toggled.connect(self._on_random_seed_toggled)
        self.lora_combo.currentTextChanged.connect(self._update_selected_lora_label)
        try:
            self.lora_combo.lineEdit().textChanged.connect(self._update_selected_lora_label)
        except Exception:
            pass

        for w in [
            self.prompt_edit,
            self.negative_edit,
            self.width_spin,
            self.height_spin,
            self.steps_spin,
            self.cfg_spin,
            self.strength_spin,
            self.seed_edit,
            self.chk_random_seed,
            self.sampler_combo,
            self.batch_spin,
            self.output_dir_edit,
            self.prefix_edit,
            self.sdcli_edit,
            self.model_edit,
            self.vae_edit,
            self.llm_edit,
            self.root_edit,
            self.models_root_edit,
            self.lora_dir_edit,
            self.lora_combo,
            self.lora_strength_spin,
        ]:
            try:
                w.textChanged.connect(self._refresh_preview)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                w.textChanged.connect(self._save_state)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                w.valueChanged.connect(self._refresh_preview)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                w.valueChanged.connect(self._save_state)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                w.currentTextChanged.connect(self._refresh_preview)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                w.currentTextChanged.connect(self._save_state)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.chk_offload_cpu.toggled.connect(self._refresh_preview)
        self.chk_offload_cpu.toggled.connect(self._save_state)
        self.chk_flash_attn.toggled.connect(self._refresh_preview)
        self.chk_flash_attn.toggled.connect(self._save_state)
        self.chk_vae_tiling.toggled.connect(self._refresh_preview)
        self.chk_vae_tiling.toggled.connect(self._save_state)
        self.chk_verbose.toggled.connect(self._refresh_preview)
        self.chk_verbose.toggled.connect(self._save_state)
        self.chk_keep_aspect.toggled.connect(self._maybe_fit_resolution_to_first_input)
        self.chk_keep_aspect.toggled.connect(self._refresh_preview)
        self.chk_keep_aspect.toggled.connect(self._save_state)
        self.chk_reuse_last.toggled.connect(self._refresh_preview)
        self.chk_reuse_last.toggled.connect(self._save_state)

        for box in [self.paths_box, self.logs_box]:
            try:
                box._toggle.toggled.connect(self._save_state)
            except Exception:
                pass

        self.run_finished.connect(lambda *_: self._save_state())

    def closeEvent(self, event) -> None:
        try:
            self._save_state()
        finally:
            super().closeEvent(event)

    # ------------------------------------------------------------------
    # Defaults and autodetect
    # ------------------------------------------------------------------

    def _on_random_seed_toggled(self, checked: bool) -> None:
        self.seed_edit.setEnabled(not checked)
        if checked:
            self.seed_edit.setPlaceholderText("A new seed will be generated on each run")
        else:
            if not (self.seed_edit.text() or "").strip():
                self.seed_edit.setText("0")
            self.seed_edit.setPlaceholderText("")
        self._refresh_preview()

    def _apply_defaults(self) -> None:
        self.output_dir_edit.setText(str(self.framevision_root / "output" / "firered"))
        self._on_random_seed_toggled(self.chk_random_seed.isChecked())
        self._refresh_preview()

    def _auto_detect_paths(self, log_it: bool = False) -> None:
        root = Path(_norm(self.root_edit.text()) or str(self.framevision_root))
        models_root = Path(_norm(self.models_root_edit.text()) or str(root / "models"))

        # sd-cli
        sd_candidates = [
            root / "sd-cli.exe",
            root / "bin" / "sd-cli.exe",
            root / "bin" / "Release" / "sd-cli.exe",
            root / "presets" / "bin" / "sd-cli.exe",
        ]
        sd_found = _pick_first_existing(sd_candidates)
        if sd_found and not _norm(self.sdcli_edit.text()):
            self.sdcli_edit.setText(str(sd_found))

        # ffmpeg / ffprobe
        ffmpeg_candidates = [
            root / "ffmpeg.exe",
            root / "bin" / "ffmpeg.exe",
            root / "presets" / "bin" / "ffmpeg.exe",
        ]
        ffprobe_candidates = [
            root / "ffprobe.exe",
            root / "bin" / "ffprobe.exe",
            root / "presets" / "bin" / "ffprobe.exe",
        ]
        ffmpeg_found = _pick_first_existing(ffmpeg_candidates)
        ffprobe_found = _pick_first_existing(ffprobe_candidates)
        if ffmpeg_found and not _norm(self.ffmpeg_edit.text()):
            self.ffmpeg_edit.setText(str(ffmpeg_found))
        if ffprobe_found and not _norm(self.ffprobe_edit.text()):
            self.ffprobe_edit.setText(str(ffprobe_found))

        # FireRed GGUF
        firered_dir = models_root / "FireRed-Image-Edit-1.1"
        model_found = None
        if firered_dir.exists():
            ggufs = sorted(firered_dir.rglob("*.gguf"))
            if ggufs:
                model_found = ggufs[0]
        if model_found and not _norm(self.model_edit.text()):
            self.model_edit.setText(str(model_found))

        # VAE + LLM search
        vae_candidates = _iter_candidates(models_root, [
            "*qwen*vae*.safetensors",
            "*qwen*image*vae*.safetensors",
            "*vae*.safetensors",
            "*ae*.safetensors",
        ])
        shared_llm = _shared_model_path(root, "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf")
        llm_candidates = ([shared_llm] if shared_llm.exists() else []) + _iter_candidates(models_root, [
            "*Qwen*VL*.gguf",
            "*Qwen2.5-VL*.gguf",
            "*Qwen3*.gguf",
            "*qwen*text*encoder*.gguf",
            "*qwen*.gguf",
        ])

        # try smarter ranking
        vae_candidates = sorted(
            vae_candidates,
            key=lambda p: (
                0 if "qwen" in p.name.lower() else 1,
                0 if "vae" in p.name.lower() or "ae" in p.name.lower() else 1,
                len(str(p)),
            ),
        )
        llm_candidates = sorted(
            llm_candidates,
            key=lambda p: (
                0 if "vl" in p.name.lower() else 1,
                0 if "qwen" in p.name.lower() else 1,
                len(str(p)),
            ),
        )

        if vae_candidates and not _norm(self.vae_edit.text()):
            self.vae_edit.setText(str(vae_candidates[0]))
        if llm_candidates and not _norm(self.llm_edit.text()):
            self.llm_edit.setText(str(llm_candidates[0]))

        # LoRA folder and list
        default_lora_dir = models_root / "lora"
        if default_lora_dir.exists() and not _norm(self.lora_dir_edit.text()):
            self.lora_dir_edit.setText(str(default_lora_dir))
        self._refresh_lora_choices()

        if log_it:
            self._log("Auto-detect completed.")
            self._log(f"Root: {root}")
            self._log(f"sd-cli: {self.sdcli_edit.text() or '[not found]'}")
            self._log(f"FireRed GGUF: {self.model_edit.text() or '[not found]'}")
            self._log(f"VAE: {self.vae_edit.text() or '[not found]'}")
            self._log(f"LLM: {self.llm_edit.text() or '[not found]'}")
            self._log(f"LoRA folder: {self.lora_dir_edit.text() or '[not found]'}")
            self._log(f"Selected LoRA: {self.lora_combo.currentText() or '[none]'}")
        self._refresh_preview()

    def _browse_lora(self) -> None:
        start = _norm(self.lora_combo.currentText()) or _norm(self.lora_dir_edit.text()) or str(self.framevision_root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LoRA file",
            start,
            "LoRA files (*.safetensors *.ckpt *.bin *.pt);;All files (*)",
        )
        if path:
            if self.lora_combo.findText(path) < 0:
                self.lora_combo.addItem(path)
            self.lora_combo.setCurrentText(path)
            self._update_selected_lora_label()
            self._refresh_preview()
            self._save_state()

    def _refresh_lora_choices(self) -> None:
        current = _norm(self.lora_combo.currentText())
        lora_dir = Path(_norm(self.lora_dir_edit.text()) or str(self.framevision_root / "models" / "lora"))
        found = []
        if lora_dir.exists():
            for pattern in ("*.safetensors", "*.ckpt", "*.bin", "*.pt"):
                found.extend(lora_dir.rglob(pattern))
        found = sorted({str(p.resolve()) for p in found}, key=lambda s: (Path(s).name.lower(), len(s)))

        self.lora_combo.blockSignals(True)
        self.lora_combo.clear()
        self.lora_combo.addItem("")
        for p in found:
            self.lora_combo.addItem(p)
        if current and current not in found:
            self.lora_combo.addItem(current)
        if current:
            self.lora_combo.setCurrentText(current)
        elif len(found) == 1:
            self.lora_combo.setCurrentText(found[0])
        self.lora_combo.blockSignals(False)
        self._update_selected_lora_label()

    def _update_selected_lora_label(self) -> None:
        text = _norm(self.lora_combo.currentText())
        if text:
            self.selected_lora_value.setText(Path(text).name)
            self.selected_lora_value.setToolTip(text)
        else:
            self.selected_lora_value.setText("[none]")
            self.selected_lora_value.setToolTip("No LoRA selected.")

    def _effective_prompt(self) -> str:
        prompt = self.prompt_edit.toPlainText().strip()
        lora_path = _norm(self.lora_combo.currentText())
        if lora_path and os.path.isfile(lora_path):
            name = Path(lora_path).stem
            tag = f"<lora:{name}:{self.lora_strength_spin.value():.2f}>"
            if f"<lora:{name}:" not in prompt and f"<lora:{name}>" not in prompt:
                prompt = f"{prompt} {tag}".strip()
        return prompt

    def _collect_state(self) -> dict:
        return {
            "prompt": self.prompt_edit.toPlainText(),
            "negative": self.negative_edit.toPlainText(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "cfg": self.cfg_spin.value(),
            "strength": self.strength_spin.value(),
            "seed": self.seed_edit.text(),
            "random_seed": self.chk_random_seed.isChecked(),
            "sampler": self.sampler_combo.currentText(),
            "format": self.format_combo.currentText(),
            "batch": self.batch_spin.value(),
            "prefix": self.prefix_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "root": self.root_edit.text(),
            "models_root": self.models_root_edit.text(),
            "lora_dir": self.lora_dir_edit.text(),
            "sdcli": self.sdcli_edit.text(),
            "ffmpeg": self.ffmpeg_edit.text(),
            "ffprobe": self.ffprobe_edit.text(),
            "model": self.model_edit.text(),
            "vae": self.vae_edit.text(),
            "llm": self.llm_edit.text(),
            "lora": self.lora_combo.currentText(),
            "lora_strength": self.lora_strength_spin.value(),
            "offload_cpu": self.chk_offload_cpu.isChecked(),
            "flash_attn": self.chk_flash_attn.isChecked(),
            "vae_tiling": self.chk_vae_tiling.isChecked(),
            "verbose": self.chk_verbose.isChecked(),
            "keep_aspect": self.chk_keep_aspect.isChecked(),
            "reuse_last": self.chk_reuse_last.isChecked(),
            "main_image": self.main_image_path,
            "ref_image2": self.ref_image2_path,
            "ref_image3": self.ref_image3_path,
            "paths_box_open": self.paths_box._toggle.isChecked(),
            "logs_box_open": self.logs_box._toggle.isChecked(),
            "last_output_file": getattr(self, "_last_output_file", ""),
        }

    def _save_state(self) -> None:
        if getattr(self, "_loading_state", False):
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._collect_state(), indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"Could not save FireRed state: {exc}")

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not read FireRed state JSON: {exc}")
            return

        self._loading_state = True
        try:
            self.prompt_edit.setPlainText(data.get("prompt", self.prompt_edit.toPlainText()))
            self.negative_edit.setPlainText(data.get("negative", self.negative_edit.toPlainText()))
            self.width_spin.setValue(int(data.get("width", self.width_spin.value())))
            self.height_spin.setValue(int(data.get("height", self.height_spin.value())))
            self.steps_spin.setValue(int(data.get("steps", self.steps_spin.value())))
            self.cfg_spin.setValue(int(data.get("cfg", self.cfg_spin.value())))
            self.strength_spin.setValue(float(data.get("strength", self.strength_spin.value())))
            saved_seed = str(data.get("seed", self.seed_edit.text()))
            saved_random_seed = data.get("random_seed", None)
            if saved_random_seed is None:
                saved_random_seed = str(saved_seed).strip() == "-1"
                if saved_random_seed:
                    saved_seed = "0"
            self.chk_random_seed.setChecked(bool(saved_random_seed))
            self.seed_edit.setText(str(saved_seed if str(saved_seed).strip() != "-1" else "0"))
            self.sampler_combo.setCurrentText(str(data.get("sampler", self.sampler_combo.currentText())))
            self.format_combo.setCurrentText(str(data.get("format", self.format_combo.currentText())))
            self.batch_spin.setValue(int(data.get("batch", self.batch_spin.value())))
            self.prefix_edit.setText(str(data.get("prefix", self.prefix_edit.text())))
            self.output_dir_edit.setText(str(data.get("output_dir", self.output_dir_edit.text())))
            self.root_edit.setText(str(data.get("root", self.root_edit.text())))
            self.models_root_edit.setText(str(data.get("models_root", self.models_root_edit.text())))
            self.lora_dir_edit.setText(str(data.get("lora_dir", self.lora_dir_edit.text())))
            self.sdcli_edit.setText(str(data.get("sdcli", self.sdcli_edit.text())))
            self.ffmpeg_edit.setText(str(data.get("ffmpeg", self.ffmpeg_edit.text())))
            self.ffprobe_edit.setText(str(data.get("ffprobe", self.ffprobe_edit.text())))
            self.model_edit.setText(str(data.get("model", self.model_edit.text())))
            self.vae_edit.setText(str(data.get("vae", self.vae_edit.text())))
            root_for_shared = Path(_norm(self.root_edit.text()) or str(self.framevision_root))
            self.llm_edit.setText(_resolve_shared_model_path(root_for_shared, str(data.get("llm", self.llm_edit.text()))))
            self._refresh_lora_choices()
            self.lora_combo.setCurrentText(str(data.get("lora", self.lora_combo.currentText())))
            self._update_selected_lora_label()
            self.lora_strength_spin.setValue(float(data.get("lora_strength", self.lora_strength_spin.value())))
            self.chk_offload_cpu.setChecked(bool(data.get("offload_cpu", self.chk_offload_cpu.isChecked())))
            self.chk_flash_attn.setChecked(bool(data.get("flash_attn", self.chk_flash_attn.isChecked())))
            self.chk_vae_tiling.setChecked(bool(data.get("vae_tiling", self.chk_vae_tiling.isChecked())))
            self.chk_verbose.setChecked(bool(data.get("verbose", self.chk_verbose.isChecked())))
            self.chk_keep_aspect.setChecked(bool(data.get("keep_aspect", self.chk_keep_aspect.isChecked())))
            self.chk_reuse_last.setChecked(bool(data.get("reuse_last", self.chk_reuse_last.isChecked())))
            legacy_images = [str(img) for img in data.get("images", []) if img]
            self._set_image_slot("main", str(data.get("main_image", legacy_images[0] if len(legacy_images) > 0 else "")), refresh=False, save=False)
            self._set_image_slot("ref2", str(data.get("ref_image2", legacy_images[1] if len(legacy_images) > 1 else "")), refresh=False, save=False)
            self._set_image_slot("ref3", str(data.get("ref_image3", legacy_images[2] if len(legacy_images) > 2 else "")), refresh=False, save=False)
            self._last_output_file = str(data.get("last_output_file", ""))
            self.paths_box._toggle.setChecked(bool(data.get("paths_box_open", False)))
            self.paths_box._on_toggled(self.paths_box._toggle.isChecked())
            self.logs_box._toggle.setChecked(bool(data.get("logs_box_open", False)))
            self.logs_box._on_toggled(self.logs_box._toggle.isChecked())
        finally:
            self._loading_state = False

    def _finalize_startup_state(self) -> None:
        self._loading_state = False
        self._refresh_lora_choices()
        self._update_selected_lora_label()
        self._refresh_preview()
        self._save_state()

    # ------------------------------------------------------------------
    # Image slot handling
    # ------------------------------------------------------------------

    def _make_image_slot(self, title: str, subtitle: str, slot_key: str) -> QWidget:
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        box.setObjectName(f"firered_{slot_key}_slot")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(title_lbl)

        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setWordWrap(True)
        lay.addWidget(subtitle_lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        btn_col = QVBoxLayout()
        btn_browse = QPushButton("Browse")
        btn_preview = QPushButton("Preview")
        btn_clear = QPushButton("Clear")
        btn_col.addWidget(btn_browse)
        btn_col.addWidget(btn_preview)
        btn_col.addWidget(btn_clear)
        btn_col.addStretch(1)
        row.addLayout(btn_col)

        thumb_lbl = QLabel("No image")
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setMinimumSize(72, 72)
        thumb_lbl.setMaximumSize(72, 72)
        thumb_lbl.setFrameShape(QFrame.Box)
        row.addWidget(thumb_lbl)

        text_col = QVBoxLayout()
        path_lbl = QLabel("Empty")
        path_lbl.setWordWrap(True)
        text_col.addWidget(path_lbl)
        text_col.addStretch(1)
        row.addLayout(text_col, 1)

        lay.addLayout(row)

        setattr(self, f"{slot_key}_image_thumb", thumb_lbl)
        setattr(self, f"{slot_key}_image_label", path_lbl)
        setattr(self, f"btn_{slot_key}_image_browse", btn_browse)
        setattr(self, f"btn_{slot_key}_image_preview", btn_preview)
        setattr(self, f"btn_{slot_key}_image_clear", btn_clear)
        return box

    def _slot_path_attr(self, slot_key: str) -> str:
        return {"main": "main_image_path", "ref2": "ref_image2_path", "ref3": "ref_image3_path"}[slot_key]

    def _slot_path(self, slot_key: str) -> str:
        return _norm(str(getattr(self, self._slot_path_attr(slot_key), "")))

    def _slot_label_widget(self, slot_key: str) -> QLabel:
        return getattr(self, f"{slot_key}_image_label")

    def _slot_thumb_widget(self, slot_key: str) -> QLabel:
        return getattr(self, f"{slot_key}_image_thumb")

    def _update_image_slot_widget(self, slot_key: str) -> None:
        path = self._slot_path(slot_key)
        label = self._slot_label_widget(slot_key)
        thumb = self._slot_thumb_widget(slot_key)
        if path and os.path.isfile(path):
            label.setText(f"{Path(path).name}\n{path}")
            pix = QPixmap(path)
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                thumb.setText("")
            else:
                thumb.clear()
                thumb.setText("Image")
        else:
            label.setText("Empty")
            thumb.clear()
            thumb.setText("No image")

    def _set_image_slot(self, slot_key: str, path: str, *, refresh: bool = True, save: bool = True) -> None:
        norm = _norm(path)
        setattr(self, self._slot_path_attr(slot_key), norm)
        self._update_image_slot_widget(slot_key)
        if slot_key == "main":
            self._maybe_fit_resolution_to_first_input()
        if refresh:
            self._refresh_preview()
        if save:
            self._save_state()

    def _clear_image_slot(self, slot_key: str) -> None:
        self._set_image_slot(slot_key, "")

    def _choose_image_for_slot(self, slot_key: str) -> None:
        current = self._slot_path(slot_key) or str(self.framevision_root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FireRed image",
            current,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)",
        )
        if path:
            self._set_image_slot(slot_key, path)

    def _preview_slot_image(self, slot_key: str) -> None:
        path = self._slot_path(slot_key)
        self._preview_image_path(path)

    def _preview_image_path(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Preview", "Image file not found.")
            return

        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Preview", "Could not load image preview.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(Path(path).name)
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(8, 8, 8, 8)
        info = QLabel(path)
        info.setWordWrap(True)
        lay.addWidget(info)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scaled = pix.scaled(860, 620, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(scaled)
        lay.addWidget(lbl, 1)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btns.addWidget(btn_close)
        lay.addLayout(btns)
        dlg.exec()

    def _reference_images(self) -> List[str]:
        # Only donor/helper slots are passed as auxiliary references.
        # The main image is sent via -i and should not be duplicated as -r,
        # otherwise FireRed tends to cling to the base image and ignore the donor.
        result: List[str] = []
        for slot_key in ["ref2", "ref3"]:
            path = self._slot_path(slot_key)
            if path:
                result.append(path)
        out: List[str] = []
        seen = set()
        for p in result:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _first_input(self) -> Optional[str]:
        if self.chk_reuse_last.isChecked() and hasattr(self, "_last_output_file") and self._last_output_file:
            if os.path.isfile(self._last_output_file):
                return self._last_output_file
        main_image = self._slot_path("main")
        if main_image:
            return main_image
        return None

    # ------------------------------------------------------------------
    # File dialogs / output
    # ------------------------------------------------------------------

    def _browse_path(self, line_edit: QLineEdit, *, file_mode: bool, dir_mode: bool, name_filters: str) -> None:
        start = _norm(line_edit.text()) or str(self.framevision_root)
        path = ""
        if file_mode:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start, name_filters)
        elif dir_mode:
            path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if path:
            line_edit.setText(path)
            if line_edit is self.lora_dir_edit:
                self._refresh_lora_choices()
            self._refresh_preview()
            self._save_state()

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_dir_edit.text() or str(self.framevision_root))
        if folder:
            self.output_dir_edit.setText(folder)
            self._refresh_preview()
            self._save_state()

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.logs_edit.toPlainText())
        self._log("Logs copied to clipboard.")

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _collect_images(self) -> List[str]:
        result: List[str] = []
        init_image = self._first_input()
        if init_image:
            result.append(init_image)
        for ref in self._reference_images():
            if ref:
                result.append(ref)
        out: List[str] = []
        seen = set()
        for p in result:
            n = _norm(p)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _resolve_runtime_seed(self, *, apply_to_ui: bool = False) -> int:
        use_random = self.chk_random_seed.isChecked()
        raw = (self.seed_edit.text() or "").strip()
        try:
            seed = int(raw)
        except Exception:
            seed = 0

        if use_random:
            seed = random.SystemRandom().randint(0, 2147483647)
        elif seed < 0:
            seed = 0

        if apply_to_ui:
            self.seed_edit.setText(str(seed))
            self._log(f"Resolved seed: {seed} {'(random)' if use_random else '(manual)'}")
            if not getattr(self, "_loading_state", False):
                self._save_state()

        return seed

    def _build_command(self, *, seed_override: Optional[int] = None) -> FireRedCommand:
        exe = _norm(self.sdcli_edit.text())
        model = _norm(self.model_edit.text())
        vae = _norm(self.vae_edit.text())
        root_for_shared = Path(_norm(self.root_edit.text()) or str(self.framevision_root))
        llm = _resolve_shared_model_path(root_for_shared, self.llm_edit.text())
        out_dir = _norm(self.output_dir_edit.text())
        prefix = self.prefix_edit.text().strip() or "firered"
        seed = str(seed_override) if seed_override is not None else (self.seed_edit.text().strip() or "0")
        prompt = self._effective_prompt()
        negative = self.negative_edit.toPlainText().strip()
        init_image = _norm(self._first_input() or "")
        ref_images = [_norm(p) for p in self._reference_images() if _norm(p)]
        lora_path = _norm(self.lora_combo.currentText())

        # Make the slot mapping explicit and stable:
        # - slot 1 / reuse-last output -> main init image (-i)
        # - slot 2 and slot 3         -> auxiliary references (-r)
        # Never echo the init image back into the reference list.
        ref_images = [p for p in ref_images if p and p != init_image]

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = self.format_combo.currentText().strip().lower()
        output_file = os.path.join(out_dir, f"{prefix}_{stamp}.{ext}")

        args: List[str] = [
            "--diffusion-model", model,
            "--vae", vae,
            "--llm", llm,
            "-p", prompt,
            "-W", str(self.width_spin.value()),
            "-H", str(self.height_spin.value()),
            "--steps", str(self.steps_spin.value()),
            "--cfg-scale", str(self.cfg_spin.value()),
            "--strength", str(self.strength_spin.value()),
            "--sampling-method", self.sampler_combo.currentText(),
            "-s", seed,
            "-o", output_file,
        ]

        if init_image:
            args.extend(["-i", init_image])

        for img in ref_images:
            args.extend(["-r", img])

        if lora_path and os.path.isfile(lora_path):
            args.extend(["--lora-model-dir", str(Path(lora_path).parent)])

        if negative:
            args.extend(["-n", negative])

        if self.chk_offload_cpu.isChecked():
            args.append("--offload-to-cpu")
        if self.chk_flash_attn.isChecked():
            args.append("--diffusion-fa")
        if self.chk_vae_tiling.isChecked():
            args.append("--vae-tiling")
        if self.chk_verbose.isChecked():
            args.append("-v")

        batch = self.batch_spin.value()
        if batch > 1:
            args.extend(["-b", str(batch)])

        return FireRedCommand(exe=exe, args=args, output_file=output_file)

    def _validate_command(self, cmd: FireRedCommand) -> List[str]:
        errors: List[str] = []
        if not cmd.exe:
            errors.append("sd-cli.exe path is empty.")
        elif not os.path.isfile(cmd.exe):
            errors.append(f"sd-cli.exe not found: {cmd.exe}")

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            errors.append("Prompt is empty.")

        images = self._collect_images()
        if not images:
            errors.append("Add at least one input image. FireRed is an image edit model.")
        if not self._slot_path("main") and not (self.chk_reuse_last.isChecked() and getattr(self, "_last_output_file", "")):
            errors.append("Main image slot is empty. Put the image you want to edit in slot 1.")

        if not self.model_edit.text().strip() or not os.path.isfile(_norm(self.model_edit.text())):
            errors.append("FireRed GGUF model file is missing or invalid.")
        if not self.vae_edit.text().strip() or not os.path.isfile(_norm(self.vae_edit.text())):
            errors.append("VAE path is missing or invalid.")
        root_for_shared = Path(_norm(self.root_edit.text()) or str(self.framevision_root))
        if not self.llm_edit.text().strip() or not os.path.isfile(_resolve_shared_model_path(root_for_shared, self.llm_edit.text())):
            errors.append("LLM/text-encoder path is missing or invalid.")

        for img in images:
            if not os.path.isfile(img):
                errors.append(f"Input image not found: {img}")

        lora_path = _norm(self.lora_combo.currentText())
        if lora_path and not os.path.isfile(lora_path):
            errors.append(f"Selected LoRA file not found: {lora_path}")

        return errors

    def _refresh_preview(self) -> None:
        try:
            cmd = self._build_command()
            self.command_preview.setPlainText(cmd.as_shell())
        except Exception as exc:
            self.command_preview.setPlainText(f"Could not build command: {exc}")
        if not getattr(self, "_loading_state", False):
            self._save_state()

    def _maybe_fit_resolution_to_first_input(self) -> None:
        if not self.chk_keep_aspect.isChecked():
            return
        first = self._first_input()
        if not first or not os.path.isfile(first):
            return
        try:
            from PIL import Image
            with Image.open(first) as im:
                w, h = im.size
            if w < 64 or h < 64:
                return
            max_side = 1536
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                w = int(round(w * scale))
                h = int(round(h * scale))
            # keep multiples of 16/64 friendly
            w = max(256, int(round(w / 16.0) * 16))
            h = max(256, int(round(h / 16.0) * 16))
            self.width_spin.blockSignals(True)
            self.height_spin.blockSignals(True)
            self.width_spin.setValue(w)
            self.height_spin.setValue(h)
            self.width_spin.blockSignals(False)
            self.height_spin.blockSignals(False)
            self._refresh_preview()
        except Exception as exc:
            self._log(f"Could not auto-fit resolution from first image: {exc}")

    # ------------------------------------------------------------------
    # Process run
    # ------------------------------------------------------------------

    def _run(self) -> None:
        resolved_seed = self._resolve_runtime_seed(apply_to_ui=True)
        cmd = self._build_command(seed_override=resolved_seed)
        errors = self._validate_command(cmd)
        if errors:
            QMessageBox.warning(self, "FireRed", "\n".join(errors))
            self._log("Run blocked because validation failed:")
            for e in errors:
                self._log(f"  - {e}")
            return

        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "FireRed", "A FireRed process is already running.")
            return

        self.process = QProcess(self)
        self.process.setProgram(cmd.exe)
        self.process.setArguments(cmd.args)
        self.process.setWorkingDirectory(str(Path(cmd.exe).parent))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_process_output)
        self.process.readyReadStandardError.connect(self._on_process_output)
        self.process.started.connect(self._on_process_started)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self._pending_output_file = cmd.output_file
        self._log("=" * 80)
        self._log("Starting FireRed process")
        self._log(f"Working dir: {Path(cmd.exe).parent}")
        self._log(f"Main init image (-i): {self._first_input() or '[none]'}")
        refs = self._reference_images()
        self._log(f"Reference images (-r): {', '.join(refs) if refs else '[none]'}")
        self._log(f"Seed used: {resolved_seed}")
        self._log(f"Command: {cmd.as_shell()}")
        self.run_started.emit(cmd.as_shell())
        self.process.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self) -> None:
        if not self.process:
            return
        if self.process.state() == QProcess.NotRunning:
            return
        self._log("Stop requested. Terminating FireRed process...")
        self.process.terminate()
        if not self.process.waitForFinished(2500):
            self._log("Process did not exit cleanly, killing it.")
            self.process.kill()

    def _on_process_started(self) -> None:
        self._log("FireRed started.")

    def _on_process_output(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.logs_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.logs_edit.insertPlainText(data)
            self.logs_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        status_text = "normal" if exit_status == QProcess.NormalExit else "crashed"
        self._log(f"FireRed finished with exit code {exit_code} ({status_text}).")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        output_file = getattr(self, "_pending_output_file", "")
        if exit_code == 0 and output_file and os.path.isfile(output_file):
            self._last_output_file = output_file
            self._log(f"Output created: {output_file}")
            self.result_ready.emit(output_file)
        elif exit_code == 0:
            self._log("Process ended successfully, but the expected output file was not found.")
        self.run_finished.emit(exit_code, output_file)
        self._save_state()

    def _on_process_error(self, err) -> None:
        self._log(f"QProcess error: {err}")

    # ------------------------------------------------------------------
    # Queue run
    # ------------------------------------------------------------------

    def _generate_queue(self) -> None:
        resolved_seed = self._resolve_runtime_seed(apply_to_ui=True)
        try:
            cmd = self._build_command(seed_override=resolved_seed)
        except Exception as exc:
            QMessageBox.warning(self, "FireRed", f"Could not build queue command: {exc}")
            self._log(f"Queue blocked: could not build command: {exc}")
            return

        errors = self._validate_command(cmd)
        if errors:
            QMessageBox.warning(self, "FireRed", "\n".join(errors))
            self._log("Queue blocked because validation failed:")
            for e in errors:
                self._log(f"  - {e}")
            return

        try:
            try:
                from helpers.queue_adapter import enqueue_firered_from_widget
            except Exception:
                from queue_adapter import enqueue_firered_from_widget
            self._log(f"[queue] seed used: {resolved_seed}")
            ok = bool(enqueue_firered_from_widget(self))
            if ok:
                self._log("[queue] queued FireRed job")
            else:
                self._log("[queue] failed to queue FireRed job")
                QMessageBox.warning(self, "FireRed", "Could not add FireRed job to the queue.")
        except Exception as exc:
            self._log(f"[queue] enqueue failed: {exc}")
            QMessageBox.warning(self, "FireRed", f"Could not add FireRed job to the queue:\n{exc}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("[%H:%M:%S]")
        self.logs_edit.appendPlainText(f"{stamp} {text}")


# -----------------------------------------------------------------------------
# Standalone test harness
# -----------------------------------------------------------------------------


class FireRedWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FireRed Image Edit 1.1 GGUF")
        self.resize(1220, 900)
        pane = FireRedPane(self)
        self.setCentralWidget(pane)
        pane.result_ready.connect(self._on_result_ready)

    def _on_result_ready(self, path: str) -> None:
        self.statusBar().showMessage(f"Result: {path}", 10000)


if __name__ == "__main__":
    app = QApplication([])
    win = FireRedWindow()
    win.show()
    app.exec()
