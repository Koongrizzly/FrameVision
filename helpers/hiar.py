from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
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
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_btn.clicked.connect(self._on_toggled)

        self._content.setVisible(expanded)
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self._content)

        layout.addWidget(self.toggle_btn)
        layout.addWidget(frame)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)



class HiARPane(QWidget):
    """
    FrameVision-style helper for the HiAR repo.

    This wrapper does not modify the HiAR repo. It only launches:
        <repo_root>/inference.py

    Expected layout under the FrameVision root:
      /models/hiar/HiAR/
      /models/hiar/HiAR/ckpts/hiar.pt
      /models/hiar/HiAR/configs/hiar.yaml
      /environments/.hiar/
      /presets/setsave/hiar.json
      /logs/hiar/
    """

    SETTINGS_REL = Path("presets/setsave/hiar.json")
    LOG_DIR_REL = Path("logs/hiar")
    DEFAULT_OUTPUT_REL = Path("output/hiar")

    def _guess_framevision_root(self) -> Path:
        env_root = os.environ.get("FRAMEVISION_ROOT", "").strip()
        if env_root:
            p = Path(env_root).expanduser()
            if p.exists():
                return p

        # Running inside FrameVision as a helper: helpers/hiar.py -> parent.parent
        module_path = Path(__file__).resolve()
        if module_path.parent.name.lower() == "helpers":
            candidate = module_path.parent.parent
            if candidate.exists():
                return candidate

        # Frozen app support
        exe_path = Path(sys.executable).resolve()
        exe_parent = exe_path.parent
        for candidate in (exe_parent, exe_parent.parent):
            if (candidate / "helpers").exists() or (candidate / "models").exists() or (candidate / "presets").exists():
                return candidate

        # Current working directory fallback
        cwd = Path.cwd().resolve()
        for candidate in (cwd, cwd.parent):
            if (candidate / "helpers").exists() or (candidate / "models").exists() or (candidate / "presets").exists():
                return candidate

        # Last resort: keep the old helpers/.. assumption
        return module_path.parent.parent

    def __init__(self, parent: Optional[QWidget] = None, framevision_root: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setObjectName("HiARPane")

        self._proc: Optional[QProcess] = None
        self._loading_settings = False
        self._temp_prompt_file: Optional[str] = None
        self._temp_config_file: Optional[str] = None
        self._temp_runner_file: Optional[str] = None
        self._run_output_dir: Optional[Path] = None
        self._run_temp_output_dir: Optional[Path] = None
        self._run_existing_mp4s: set[str] = set()
        self._last_preview = ""
        self._run_started_at: Optional[float] = None
        self._run_started_wallclock: Optional[datetime] = None

        self.framevision_root = self._resolve_framevision_root(framevision_root)
        self.settings_path = self.framevision_root / self.SETTINGS_REL
        self.logs_dir = self.framevision_root / self.LOG_DIR_REL
        self.default_output_dir = self.framevision_root / self.DEFAULT_OUTPUT_REL

        self._build_ui()
        self._apply_base_defaults()
        self._load_settings_protected()
        self._autodetect_missing_only()
        self._finish_startup_save_once()

    # -------------------------
    # Startup / persistence
    # -------------------------
    def _resolve_framevision_root(self, explicit_root: Optional[str]) -> Path:
        if explicit_root:
            return Path(explicit_root).resolve()

        here = Path(__file__).resolve()
        # helpers/hiar.py -> FrameVision root
        if here.parent.name.lower() == "helpers":
            return here.parent.parent.resolve()
        return Path.cwd().resolve()

    def _label_with_tip(self, text: str, tip: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setToolTip(tip)
        return lbl

    def _apply_base_defaults(self) -> None:
        self.repo_root_edit.setText(str(self.framevision_root / "models/hiar/HiAR"))
        self.python_edit.setText(str(self.framevision_root / "environments/.hiar/Scripts/python.exe"))
        self.config_edit.setText(str(self.framevision_root / "models/hiar/HiAR/configs/hiar.yaml"))
        self.checkpoint_edit.setText(str(self.framevision_root / "models/hiar/HiAR/ckpts/hiar.pt"))
        self.output_edit.setText(str(self.default_output_dir))
        self.extended_prompt_edit.setText("")
        self.prompt_file_edit.setText("")
        self.seed_spin.setValue(0)
        self.frames_spin.setValue(21)
        self.guidance_spin.setValue(3.0)
        self.negative_prompt_box.setPlainText("")
        self.samples_spin.setValue(1)
        self.inference_method_combo.setCurrentText("timestep_first")
        self.frame_first_blocks_spin.setValue(1)
        self.use_ema_check.setChecked(True)
        self.save_with_index_check.setChecked(True)
        self.auto_open_output_check.setChecked(False)
        self.prompt_text.setPlainText("")
        self.preview_box.setPlainText("")
        self.logs_box.setPlainText("")
        try:
            self.use_queue_check.setChecked(True)
        except Exception:
            pass
        self._update_hybrid_visibility()

    def _autodetect_missing_only(self) -> None:
        repo_root = Path(self.repo_root_edit.text().strip())
        python_path = Path(self.python_edit.text().strip())
        config_path = Path(self.config_edit.text().strip())
        checkpoint_path = Path(self.checkpoint_edit.text().strip())

        if not self.repo_root_edit.text().strip() or not repo_root.exists():
            candidate = self.framevision_root / "models/hiar/HiAR"
            if candidate.exists():
                self.repo_root_edit.setText(str(candidate))
                repo_root = candidate

        if (not self.python_edit.text().strip()) or (not python_path.exists()):
            candidate = self.framevision_root / "environments/.hiar/Scripts/python.exe"
            if candidate.exists():
                self.python_edit.setText(str(candidate))

        if (not self.config_edit.text().strip()) or (not config_path.exists()):
            candidate = repo_root / "configs/hiar.yaml"
            if candidate.exists():
                self.config_edit.setText(str(candidate))

        if (not self.checkpoint_edit.text().strip()) or (not checkpoint_path.exists()):
            candidate = repo_root / "ckpts/hiar.pt"
            if candidate.exists():
                self.checkpoint_edit.setText(str(candidate))

        if not self.output_edit.text().strip():
            self.output_edit.setText(str(self.default_output_dir))

    def _finish_startup_save_once(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        self._save_settings()

    def _load_settings_protected(self) -> None:
        self._loading_settings = True
        try:
            if not self.settings_path.exists():
                return
            with self.settings_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self.repo_root_edit.setText(data.get("repo_root", self.repo_root_edit.text()))
            self.python_edit.setText(data.get("python_path", self.python_edit.text()))
            self.config_edit.setText(data.get("config_path", self.config_edit.text()))
            self.checkpoint_edit.setText(data.get("checkpoint_path", self.checkpoint_edit.text()))
            self.prompt_file_edit.setText(data.get("prompt_file", ""))
            self.extended_prompt_edit.setText(data.get("extended_prompt_path", ""))
            self.output_edit.setText(data.get("output_folder", self.output_edit.text()))
            self.prompt_text.setPlainText(data.get("prompt_text", ""))
            self.seed_spin.setValue(int(data.get("seed", self.seed_spin.value())))
            self.frames_spin.setValue(int(data.get("num_output_frames", self.frames_spin.value())))
            self.guidance_spin.setValue(float(data.get("guidance_scale", self.guidance_spin.value())))
            self.negative_prompt_box.setPlainText(data.get("negative_prompt", self.negative_prompt_box.toPlainText()))
            self.samples_spin.setValue(int(data.get("num_samples", self.samples_spin.value())))
            saved_method = data.get("inference_method", self.inference_method_combo.currentText())
            if saved_method == "frame_first":
                saved_method = "timestep_first"
            self.inference_method_combo.setCurrentText(saved_method)
            self.frame_first_blocks_spin.setValue(int(data.get("num_frame_first_blocks", self.frame_first_blocks_spin.value())))
            self.use_ema_check.setChecked(True)
            self.save_with_index_check.setChecked(bool(data.get("save_with_index", self.save_with_index_check.isChecked())))
            self.auto_open_output_check.setChecked(bool(data.get("auto_open_output", self.auto_open_output_check.isChecked())))
            try:
                self.use_queue_check.setChecked(bool(data.get("use_queue", self.use_queue_check.isChecked())))
            except Exception:
                pass
            self._update_hybrid_visibility()
        except Exception as exc:
            self._append_log(f"[settings] failed to load settings: {exc}")
        finally:
            self._loading_settings = False

    def _save_settings(self) -> None:
        if self._loading_settings:
            return
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "repo_root": self.repo_root_edit.text().strip(),
                "python_path": self.python_edit.text().strip(),
                "config_path": self.config_edit.text().strip(),
                "checkpoint_path": self.checkpoint_edit.text().strip(),
                "prompt_file": self.prompt_file_edit.text().strip(),
                "extended_prompt_path": self.extended_prompt_edit.text().strip(),
                "output_folder": self.output_edit.text().strip(),
                "prompt_text": self.prompt_text.toPlainText(),
                "seed": self.seed_spin.value(),
                "num_output_frames": self.frames_spin.value(),
                "guidance_scale": self.guidance_spin.value(),
                "negative_prompt": self.negative_prompt_box.toPlainText(),
                "num_samples": self.samples_spin.value(),
                "inference_method": self.inference_method_combo.currentText(),
                "num_frame_first_blocks": self.frame_first_blocks_spin.value(),
                "use_ema": True,
                "save_with_index": self.save_with_index_check.isChecked(),
                "auto_open_output": self.auto_open_output_check.isChecked(),
                "use_queue": self.use_queue_check.isChecked(),
            }
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            self._append_log(f"[settings] failed to save settings: {exc}")

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        self._create_action_buttons()
        self.paths_section = _CollapsibleSection("Folders and paths", self._build_paths_group(), expanded=False)
        self.preview_logs_section = _CollapsibleSection("Command preview and logs", self._build_preview_logs_group(), expanded=False)

        main_layout.addWidget(self._build_generation_group())
        main_layout.addWidget(self._build_prompt_group())
        main_layout.addWidget(self.paths_section)
        main_layout.addWidget(self.preview_logs_section, 1)
        main_layout.addStretch(1)

    def _make_path_row(self, label_text: str, browse_mode: str = "file"):
        label = QLabel(label_text)
        edit = QLineEdit()
        browse = QPushButton("Browse")
        if browse_mode == "dir":
            browse.clicked.connect(lambda: self._browse_dir(edit))
        else:
            browse.clicked.connect(lambda: self._browse_file(edit))
        return label, edit, browse

    def _build_paths_group(self) -> QWidget:
        container = QWidget()
        layout = QGridLayout(container)

        _, self.repo_root_edit, repo_btn = self._make_path_row("Repo root", "dir")
        _, self.python_edit, python_btn = self._make_path_row("Python exe", "file")
        _, self.config_edit, config_btn = self._make_path_row("Config", "file")
        _, self.checkpoint_edit, ckpt_btn = self._make_path_row("Checkpoint", "file")
        _, self.prompt_file_edit, prompt_file_btn = self._make_path_row("Prompt file", "file")
        _, self.extended_prompt_edit, ext_btn = self._make_path_row("Extended prompt file", "file")
        _, self.output_edit, out_btn = self._make_path_row("Output folder", "dir")

        rows = [
            ("Repo root", self.repo_root_edit, repo_btn),
            ("Python exe", self.python_edit, python_btn),
            ("Config", self.config_edit, config_btn),
            ("Checkpoint", self.checkpoint_edit, ckpt_btn),
            ("Prompt file", self.prompt_file_edit, prompt_file_btn),
            ("Extended prompt file", self.extended_prompt_edit, ext_btn),
            ("Output folder", self.output_edit, out_btn),
        ]
        for row_idx, (label_text, edit, btn) in enumerate(rows):
            layout.addWidget(QLabel(label_text), row_idx, 0)
            layout.addWidget(edit, row_idx, 1)
            layout.addWidget(btn, row_idx, 2)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        self.auto_detect_btn = QPushButton("Auto detect paths")
        button_layout.addWidget(self.auto_detect_btn)
        button_layout.addWidget(self.import_check_btn)
        button_layout.addStretch(1)
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        layout.addWidget(button_row, len(rows), 0, 1, 3)
        return container

    def _build_generation_group(self) -> QGroupBox:
        group = QGroupBox("Generation")
        form = QFormLayout(group)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 528)
        self.frames_spin.setSingleStep(3)
        self.frames_spin.setValue(66)
        self.frames_spin.setToolTip("Start frames. Fixed to multiples of 3 because HiAR uses KV inference with 3 frames per block.")

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 30.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setValue(3.0)
        self.guidance_spin.setToolTip("Guidance / CFG scale. Previous fixed default from the config was 3.0.")

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 64)
        self.samples_spin.setValue(1)

        self.inference_method_combo = QComboBox()
        self.inference_method_combo.addItems(["timestep_first", "hybrid_block0"])
        self.inference_method_combo.currentTextChanged.connect(self._update_hybrid_visibility)

        self.frame_first_blocks_spin = QSpinBox()
        self.frame_first_blocks_spin.setRange(1, 64)
        self.frame_first_blocks_spin.setValue(1)

        self.use_ema_check = QCheckBox("Use EMA")
        self.use_ema_check.setChecked(True)
        self.save_with_index_check = QCheckBox("Save with index")
        self.auto_open_output_check = QCheckBox("Open output folder when done")

        self.samples_spin.setToolTip(
            "How many videos to generate for each prompt line. Higher values increase VRAM use and runtime very quickly. "
            "On a 24 GB card, keep this at 1 unless you really need multiple variations."
        )
        self.inference_method_combo.setToolTip(
            "Selects how HiAR handles long-video inference. frame_first is hidden because it improved movement but degraded quality too quickly in testing."
        )
        self.frame_first_blocks_spin.setToolTip(
            "Only used by hybrid methods. Controls how many early blocks use the frame-first strategy before switching."
        )
        self.save_with_index_check.setToolTip(
            "Adds an index to output filenames so repeated runs do not overwrite earlier results."
        )
        self.auto_open_output_check.setToolTip(
            "Automatically opens the output folder after the run finishes."
        )

        seed_frames_row = QWidget()
        seed_frames_layout = QHBoxLayout(seed_frames_row)
        seed_frames_layout.setContentsMargins(0, 0, 0, 0)
        seed_frames_layout.setSpacing(8)
        seed_frames_layout.addWidget(self._label_with_tip("Seed", "Random seed. Previous default was 0."))
        seed_frames_layout.addWidget(self.seed_spin, 1)
        seed_frames_layout.addSpacing(8)
        seed_frames_layout.addWidget(self._label_with_tip("Frames", "Number of output frames. Previous default was 21 in this helper."))
        seed_frames_layout.addWidget(self.frames_spin, 1)

        guidance_row = QWidget()
        guidance_row_layout = QHBoxLayout(guidance_row)
        guidance_row_layout.setContentsMargins(0, 0, 0, 0)
        guidance_row_layout.setSpacing(8)
        guidance_row_layout.addWidget(self._label_with_tip("Guidance", "Guidance / CFG scale. Previous fixed default from the config was 3.0."))
        guidance_row_layout.addWidget(self.guidance_spin, 1)
        guidance_row_layout.addStretch(1)

        self.frames_info_label = QLabel()
        self.frames_info_label.setWordWrap(True)
        self.frames_info_label.setToolTip(
            "HiAR is locked to 832 × 480 @ 16 fps in this helper. The finished video becomes longer than the selected start frames because HiAR uses KV inference with 3 frames per block. Observed example: 33 start frames becomes about 8 seconds final output."
        )
        self.frames_warning_label = QLabel()
        self.frames_warning_label.setWordWrap(True)
        self.frames_warning_label.setStyleSheet("color: #c97a00; font-weight: 600;")

        self.frames_spin.valueChanged.connect(self._update_frames_ui)

        form.addRow(seed_frames_row)
        form.addRow(guidance_row)
        form.addRow(self.frames_info_label)
        form.addRow(self.frames_warning_label)
        form.addRow(self._build_advanced_generation_section())
        self._update_frames_ui()
        return group

    def _build_advanced_generation_section(self) -> QWidget:
        content = QWidget()
        content_form = QFormLayout(content)
        content_form.setContentsMargins(0, 0, 0, 0)
        content_form.setSpacing(8)
        content_form.addRow(self._label_with_tip("Samples per prompt", "How many videos to generate for each prompt line. Higher values increase VRAM use and runtime very quickly. On a 24 GB card, keep this at 1 unless you really need multiple variations."), self.samples_spin)
        content_form.addRow(self._label_with_tip("Inference method", "Selects how HiAR handles long-video inference. Different methods trade speed, memory use, and temporal behavior."), self.inference_method_combo)
        content_form.addRow(self._label_with_tip("Hybrid frame-first blocks", "Only used by hybrid methods. Controls how many early blocks use the frame-first strategy before switching."), self.frame_first_blocks_spin)
        content_form.addRow(self.save_with_index_check)
        content_form.addRow(self.auto_open_output_check)
        return _CollapsibleSection("Advanced inference and output", content, expanded=False)

    def _build_prompt_group(self) -> QGroupBox:
        group = QGroupBox("Prompt")
        layout = QVBoxLayout(group)
        info = QLabel(
            "Paste one or more prompts here. If this box is not empty, the helper writes a temporary prompt file and uses that. "
            "If this box is empty, it uses the Prompt file path above."
        )
        info.setWordWrap(True)
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText("One prompt per line")
        self.negative_prompt_box = QPlainTextEdit()
        self.negative_prompt_box.setPlaceholderText("Optional negative prompt override")
        self.negative_prompt_box.setFixedHeight(72)
        self.negative_prompt_box.setToolTip("Optional negative prompt override. Leave empty to keep the repo config default negative prompt.")
        layout.addWidget(info)
        layout.addWidget(self.prompt_text)
        layout.addWidget(self._label_with_tip("Negative prompt", "Optional negative prompt override. Leave empty to keep the repo config default negative prompt."))
        layout.addWidget(self.negative_prompt_box)
        return group

    def _create_action_buttons(self) -> None:
        self.preview_btn = QPushButton("Preview command")
        self.import_check_btn = QPushButton("Import check")
        self.generate_btn = QPushButton("Generate")
        self.stop_btn = QPushButton("Stop")
        self.use_queue_check = QCheckBox("Use queue")
        self.use_queue_check.setChecked(True)
        self.use_queue_check.setToolTip("When enabled, Generate adds the HiAR video job to the FrameVision queue instead of launching it directly in this window.")
        self.open_output_btn = QPushButton("Open output")
        self.clear_logs_btn = QPushButton("Clear logs")

        self.preview_btn.clicked.connect(self._on_preview)
        self.import_check_btn.clicked.connect(self._on_import_check)
        self.generate_btn.clicked.connect(self._on_generate)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_output_btn.clicked.connect(self._open_output_folder)
        self.clear_logs_btn.clicked.connect(self._on_clear_logs)


    def _on_clear_logs(self) -> None:
        if hasattr(self, "logs_box") and self.logs_box is not None:
            self.logs_box.clear()

    def build_footer_bar(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch(1)
        layout.addWidget(self.open_output_btn)
        return footer

    def _build_preview_logs_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        preview_row = QWidget()
        preview_row_layout = QHBoxLayout(preview_row)
        preview_row_layout.setContentsMargins(0, 0, 0, 0)
        preview_row_layout.setSpacing(6)
        preview_row_layout.addWidget(QLabel("Command preview"))
        preview_row_layout.addStretch(1)
        preview_row_layout.addWidget(self.preview_btn)

        self.preview_box = QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(120)

        logs_row = QWidget()
        logs_row_layout = QHBoxLayout(logs_row)
        logs_row_layout.setContentsMargins(0, 0, 0, 0)
        logs_row_layout.setSpacing(6)
        logs_row_layout.addWidget(QLabel("Logs"))
        logs_row_layout.addStretch(1)
        logs_row_layout.addWidget(self.use_queue_check)
        logs_row_layout.addWidget(self.clear_logs_btn)

        self.logs_box = QPlainTextEdit()
        self.logs_box.setReadOnly(True)
        self.logs_box.setMinimumHeight(180)

        layout.addWidget(preview_row)
        layout.addWidget(self.preview_box)
        layout.addWidget(logs_row)
        layout.addWidget(self.logs_box)
        return container

    # -------------------------
    # UI actions
    # -------------------------
    def _browse_file(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text().strip() or str(self.framevision_root))
        if path:
            edit.setText(path)
            self._save_settings()

    def _browse_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", edit.text().strip() or str(self.framevision_root))
        if path:
            edit.setText(path)
            self._save_settings()

    def _on_auto_detect(self) -> None:
        repo_root = self.framevision_root / "models/hiar/HiAR"
        py_exe = self.framevision_root / "environments/.hiar/Scripts/python.exe"
        config = repo_root / "configs/hiar.yaml"
        ckpt = repo_root / "ckpts/hiar.pt"
        out = self.framevision_root / self.DEFAULT_OUTPUT_REL

        if repo_root.exists():
            self.repo_root_edit.setText(str(repo_root))
        if py_exe.exists():
            self.python_edit.setText(str(py_exe))
        if config.exists():
            self.config_edit.setText(str(config))
        if ckpt.exists():
            self.checkpoint_edit.setText(str(ckpt))
        out.mkdir(parents=True, exist_ok=True)
        self.output_edit.setText(str(out))
        self._save_settings()
        self._append_log(f"[ui] auto detect finished :: root={self.framevision_root}")

    def _update_hybrid_visibility(self) -> None:
        is_hybrid = self.inference_method_combo.currentText() == "hybrid_block0"
        self.frame_first_blocks_spin.setVisible(is_hybrid)
        label = self._label_for_widget(self.frame_first_blocks_spin)
        if label is not None:
            label.setVisible(is_hybrid)

    def _label_for_widget(self, widget: QWidget) -> Optional[QLabel]:
        parent = widget.parentWidget()
        if not isinstance(parent, QGroupBox):
            return None
        form = parent.layout()
        if not isinstance(form, QFormLayout):
            return None
        for i in range(form.rowCount()):
            role_item = form.itemAt(i, QFormLayout.FieldRole)
            if role_item and role_item.widget() is widget:
                label_item = form.itemAt(i, QFormLayout.LabelRole)
                if label_item:
                    return label_item.widget()
        return None

    def _on_import_check(self) -> None:
        try:
            repo_root = Path(self.repo_root_edit.text().strip())
            py_exe = Path(self.python_edit.text().strip())
            cmd = [
                str(py_exe), "-u", "-c",
                "import sys, os, importlib, traceback; print(sys.executable); print(os.getcwd()); import torch; print('torch ok'); import imageio, omegaconf, einops, lmdb, av, sentencepiece; print('deps ok'); mods=['pipeline','utils.wan_wrapper','wan','wan.modules','wan.modules.t5']; ok=True;\nfor m in mods:\n    try:\n        print('trying', m)\n        importlib.import_module(m)\n        print('ok', m)\n    except Exception:\n        print('FAILED', m)\n        traceback.print_exc()\n        ok=False\n        break\nprint('safe import-check ok')\nraise SystemExit(0 if ok else 1)"
            ]
            proc = QProcess(self)
            proc.setWorkingDirectory(str(repo_root))
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONPATH", self._join_env_path(str(repo_root), os.environ.get("PYTHONPATH", "")))
            env.insert("PYTHONUNBUFFERED", "1")
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert("USERNAME", os.environ.get("USERNAME", "FrameVision"))
            env.insert("USER", os.environ.get("USER", os.environ.get("USERNAME", "FrameVision")))
            env.insert("USERPROFILE", os.environ.get("USERPROFILE", str(self.framevision_root)))
            env.insert("HOME", os.environ.get("HOME", os.environ.get("USERPROFILE", str(self.framevision_root))))
            env.insert("TEMP", os.environ.get("TEMP", str(self.framevision_root / "temp")))
            env.insert("TMP", os.environ.get("TMP", os.environ.get("TEMP", str(self.framevision_root / "temp"))))
            proc.setProcessEnvironment(env)
            proc.readyReadStandardOutput.connect(lambda: self._append_log(proc.readAllStandardOutput().data().decode("utf-8", errors="replace").rstrip()))
            proc.finished.connect(lambda code, status: self._append_log(f"[import-check] finished with exit code {code} (status={status})"))
            self._append_log("[import-check] starting")
            self._append_log(self._quote_cmd(cmd))
            proc.start()
            self._import_proc = proc
        except Exception as exc:
            self._append_log(f"[import-check] failed: {exc}")

    def _on_preview(self) -> None:
        try:
            cmd = self._build_command(preview_only=True)
            self._last_preview = self._quote_cmd(cmd)
            self.preview_box.setPlainText(self._last_preview)
            self._append_log("[preview] command updated")
        except Exception as exc:
            self._append_log(f"[preview] failed: {exc}")
            QMessageBox.warning(self, "HiAR", str(exc))

    def _on_generate(self) -> None:
        if self.use_queue_check.isChecked():
            try:
                from helpers.queue_adapter import enqueue_hiar_from_widget
            except Exception:
                try:
                    from queue_adapter import enqueue_hiar_from_widget
                except Exception as exc:
                    self._append_log(f"[queue] failed to import queue adapter: {exc}")
                    QMessageBox.warning(self, "HiAR", "Queue adapter import failed:\n" + str(exc))
                    return
            try:
                cmd = self._build_command(preview_only=True)
                self._last_preview = self._quote_cmd(cmd)
                self.preview_box.setPlainText(self._last_preview)
            except Exception as exc:
                self._append_log(f"[queue] preview build failed: {exc}")
            try:
                self._save_settings()
                ok = bool(enqueue_hiar_from_widget(self))
                if ok:
                    self._append_log("[queue] queued HiAR job")
                else:
                    self._append_log("[queue] failed to queue HiAR job")
                    QMessageBox.warning(self, "HiAR", "Failed to queue HiAR job. Check logs for details.")
            except Exception as exc:
                self._append_log(f"[queue] failed: {exc}")
                QMessageBox.warning(self, "HiAR", str(exc))
            return

        if self._proc and self._proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "HiAR", "A HiAR process is already running.")
            return

        repo_root = Path(self.repo_root_edit.text().strip())
        output_folder = Path(self.output_edit.text().strip())
        output_folder.mkdir(parents=True, exist_ok=True)
        self._run_output_dir = output_folder
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._run_temp_output_dir = output_folder / f"_hiar_run_{run_stamp}"
        self._run_temp_output_dir.mkdir(parents=True, exist_ok=True)
        self._run_existing_mp4s = {str(x.resolve()) for x in output_folder.glob("*.mp4")}

        try:
            cmd = self._build_command(preview_only=False)
        except Exception as exc:
            self._append_log(f"[run] invalid setup: {exc}")
            temp_dir = self._run_temp_output_dir
            self._run_output_dir = None
            self._run_temp_output_dir = None
            self._run_existing_mp4s = set()
            if temp_dir:
                try:
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except Exception:
                    pass
            QMessageBox.warning(self, "HiAR", str(exc))
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._save_settings()

        self._last_preview = self._quote_cmd(cmd)
        self.preview_box.setPlainText(self._last_preview)
        self._run_started_at = time.monotonic()
        self._run_started_wallclock = datetime.now()
        self._append_log(f"[run] starting HiAR process :: {self._run_started_wallclock.strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(self._last_preview)

        self._append_log(f"[diag] repo_root exists: {repo_root.exists()} :: {repo_root}")
        self._append_log(f"[diag] inference.py exists: {(repo_root / 'inference.py').exists()} :: {repo_root / 'inference.py'}")
        self._append_log(f"[diag] config exists: {Path(self.config_edit.text().strip()).exists()} :: {self.config_edit.text().strip()}")
        self._append_log(f"[diag] checkpoint exists: {Path(self.checkpoint_edit.text().strip()).exists()} :: {self.checkpoint_edit.text().strip()}")
        self._append_log(f"[diag] python exists: {Path(self.python_edit.text().strip()).exists()} :: {self.python_edit.text().strip()}")
        self._append_log(f"[diag] wan model dir exists: {(repo_root / 'wan_models' / 'Wan2.1-T2V-1.3B').exists()} :: {repo_root / 'wan_models' / 'Wan2.1-T2V-1.3B'}")
        self._append_log(f"[diag] VAE exists: {(repo_root / 'wan_models' / 'Wan2.1-T2V-1.3B' / 'Wan2.1_VAE.pth').exists()}")
        self._append_log(f"[diag] T5 exists: {(repo_root / 'wan_models' / 'Wan2.1-T2V-1.3B' / 'models_t5_umt5-xxl-enc-bf16.pth').exists()}")
        self._append_log(f"[diag] tokenizer dir exists: {(repo_root / 'wan_models' / 'Wan2.1-T2V-1.3B' / 'google' / 'umt5-xxl').exists()}")

        proc = QProcess(self)
        proc.setWorkingDirectory(str(repo_root))
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()

        repo_root_str = str(repo_root)
        fv_root_str = str(self.framevision_root)
        # Keep caches inside FrameVision / repo tree.
        env.insert("PYTHONPATH", self._join_env_path(repo_root_str, os.environ.get("PYTHONPATH", "")))
        env.insert("HF_HOME", str(self.framevision_root / "models/hiar/hf_cache"))
        env.insert("HUGGINGFACE_HUB_CACHE", str(self.framevision_root / "models/hiar/hf_cache/hub"))
        env.insert("TRANSFORMERS_CACHE", str(self.framevision_root / "models/hiar/hf_cache/transformers"))
        env.insert("TORCH_HOME", str(self.framevision_root / "models/hiar/torch_cache"))
        env.insert("XDG_CACHE_HOME", str(self.framevision_root / "models/hiar/cache"))
        env.insert("FRAMEVISION_ROOT", fv_root_str)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("USERNAME", os.environ.get("USERNAME", "FrameVision"))
        env.insert("USER", os.environ.get("USER", os.environ.get("USERNAME", "FrameVision")))
        env.insert("USERPROFILE", os.environ.get("USERPROFILE", fv_root_str))
        env.insert("HOME", os.environ.get("HOME", os.environ.get("USERPROFILE", fv_root_str)))
        env.insert("TEMP", os.environ.get("TEMP", str(self.framevision_root / "temp")))
        env.insert("TMP", os.environ.get("TMP", os.environ.get("TEMP", str(self.framevision_root / "temp"))))
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._read_proc_output)
        proc.readyReadStandardError.connect(self._read_proc_output)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)
        self._proc = proc
        proc.start()

    def _on_stop(self) -> None:
        if not self._proc or self._proc.state() == QProcess.NotRunning:
            self._append_log("[run] no running HiAR process to stop")
            return
        self._append_log("[run] stopping HiAR process")
        self._proc.kill()

    def _open_output_folder(self) -> None:
        path = Path(self.output_edit.text().strip())
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            self._append_log(f"[ui] failed to open output folder: {exc}")

    def _on_proc_error(self, err) -> None:
        self._append_log(f"[run] process error: {err}")

    def _on_proc_finished(self, exit_code: int, exit_status) -> None:
        self._read_proc_output()
        elapsed_text = self._format_elapsed_runtime()
        self._append_log(f"[run] finished with exit code {exit_code} (status={exit_status}) :: elapsed {elapsed_text}")
        if exit_code == 0:
            self._rename_new_outputs()
        if self.auto_open_output_check.isChecked() and exit_code == 0:
            QTimer.singleShot(150, self._open_output_folder)
        self._run_started_at = None
        self._run_started_wallclock = None
        self._cleanup_temp_prompt_file()
        self._cleanup_temp_config_file()
        self._cleanup_temp_runner_file()


    def _format_elapsed_runtime(self) -> str:
        if self._run_started_at is None:
            return "unknown"
        elapsed_seconds = max(0.0, time.monotonic() - self._run_started_at)
        total_seconds = int(round(elapsed_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

    def _prompt_words_for_filename(self, max_words: int = 4) -> str:
        prompt_bases = self._prompt_bases_for_batch(max_words=max_words)
        if prompt_bases:
            return prompt_bases[0]
        return "output"

    def _load_prompt_lines_for_batch(self) -> List[str]:
        prompt_text = self.prompt_text.toPlainText().strip()
        if prompt_text:
            raw_text = prompt_text
        else:
            prompt_file = self.prompt_file_edit.text().strip()
            if not prompt_file or not Path(prompt_file).exists():
                return []
            try:
                raw_text = Path(prompt_file).read_text(encoding="utf-8")
            except Exception:
                return []

        lines: List[str] = []
        normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized_text.split("\n"):
            cleaned = line.strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _sanitize_prompt_for_filename(self, source: str, max_words: int = 4) -> str:
        source = source.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip().lower()
        cleaned: List[str] = []
        for part in source.split():
            word = "".join(ch for ch in part if ch.isalnum())
            if word:
                cleaned.append(word)
            if len(cleaned) >= max_words:
                break
        return "_".join(cleaned) if cleaned else "output"

    def _prompt_bases_for_batch(self, max_words: int = 4) -> List[str]:
        lines = self._load_prompt_lines_for_batch()
        if not lines:
            return []
        return [self._sanitize_prompt_for_filename(line, max_words=max_words) for line in lines]

    def _rename_new_outputs(self) -> None:
        try:
            if not self._run_output_dir:
                return
            output_dir = self._run_output_dir
            source_dir = self._run_temp_output_dir or output_dir
            existing = self._run_existing_mp4s or set()
            if source_dir == output_dir:
                new_files = [x for x in output_dir.glob("*.mp4") if str(x.resolve()) not in existing]
            else:
                new_files = list(source_dir.glob("*.mp4"))
            if not new_files:
                return

            prompt_parts = self._prompt_bases_for_batch()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_count = max(1, int(self.samples_spin.value()))

            new_files = sorted(new_files, key=lambda x: x.stat().st_mtime)
            total_files = len(new_files)
            total_prompts = len(prompt_parts)
            mismatch_note_logged = False

            for idx, src_path in enumerate(new_files, 1):
                prompt_part = None
                sample_suffix = ""

                if total_prompts == total_files and total_prompts > 0:
                    prompt_part = prompt_parts[idx - 1]
                elif total_prompts > 0 and total_files == total_prompts * sample_count:
                    prompt_index = (idx - 1) // sample_count
                    sample_index = ((idx - 1) % sample_count) + 1
                    prompt_part = prompt_parts[prompt_index]
                    if sample_count > 1:
                        sample_suffix = f"_{sample_index:02d}"
                elif total_prompts == 1:
                    prompt_part = prompt_parts[0]
                elif total_prompts > 0:
                    fallback_index = min(max(idx - 1, 0), total_prompts - 1)
                    prompt_part = prompt_parts[fallback_index]
                    if not mismatch_note_logged:
                        self._append_log(
                            f"[run] rename note: prompt/output count mismatch ({total_prompts} prompts, {total_files} videos); using sequential prompt fallback"
                        )
                        mismatch_note_logged = True

                if not prompt_part:
                    prompt_part = self._prompt_words_for_filename()

                prefix = f"hiar_{prompt_part}_{stamp}"
                if total_files == 1 and not sample_suffix:
                    target_name = f"{prefix}.mp4"
                else:
                    target_name = f"{prefix}{sample_suffix}.mp4"
                target_path = output_dir / target_name
                n = 2
                while target_path.exists():
                    if total_files == 1 and not sample_suffix:
                        target_path = output_dir / f"{prefix}_{n:02d}.mp4"
                    else:
                        target_path = output_dir / f"{prefix}{sample_suffix}_{n:02d}.mp4"
                    n += 1
                src_path.replace(target_path)
                self._append_log(f"[run] renamed output -> {target_path.name}")
        except Exception as exc:
            self._append_log(f"[run] output rename failed: {exc}")
        finally:
            temp_dir = self._run_temp_output_dir
            self._run_existing_mp4s = set()
            self._run_output_dir = None
            self._run_temp_output_dir = None
            if temp_dir:
                try:
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except Exception:
                    pass

    def _read_proc_output(self) -> None:
        if not self._proc:
            return
        data = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.logs_box.appendPlainText(data.rstrip("\n"))
            self._write_run_log(data)

    # -------------------------
    # Command building
    # -------------------------
    def _build_command(self, preview_only: bool) -> List[str]:
        repo_root = Path(self.repo_root_edit.text().strip())
        py_exe = Path(self.python_edit.text().strip())
        config_path = Path(self.config_edit.text().strip())
        checkpoint_path = Path(self.checkpoint_edit.text().strip())
        output_folder = self._run_temp_output_dir or Path(self.output_edit.text().strip())

        if not repo_root.exists():
            raise FileNotFoundError(f"Repo root not found: {repo_root}")
        if not (repo_root / "inference.py").exists():
            raise FileNotFoundError(f"inference.py not found in repo root: {repo_root}")
        if not py_exe.exists():
            raise FileNotFoundError(f"Python exe not found: {py_exe}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config_path = Path(self._resolve_config_for_run(config_path, preview_only=preview_only))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data_path = self._resolve_prompt_input(preview_only=preview_only)
        if not data_path:
            raise ValueError("Provide prompt text or select a prompt file.")

        extended_prompt_path = self.extended_prompt_edit.text().strip()

        inference_script = self._resolve_inference_script_for_run(repo_root, config_path, preview_only=preview_only)

        cmd = [
            str(py_exe),
            "-u",
            str(inference_script),
            "--config_path", str(config_path),
            "--checkpoint_path", str(checkpoint_path),
            "--data_path", data_path,
            "--output_folder", str(output_folder),
            "--num_output_frames", str(self.frames_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--num_samples", str(self.samples_spin.value()),
            "--inference_method", self.inference_method_combo.currentText(),
        ]

        if extended_prompt_path:
            cmd.extend(["--extended_prompt_path", extended_prompt_path])
        cmd.append("--use_ema")
        if self.save_with_index_check.isChecked():
            cmd.append("--save_with_index")
        if self.inference_method_combo.currentText() == "hybrid_block0":
            cmd.extend(["--num_frame_first_blocks", str(self.frame_first_blocks_spin.value())])

        return cmd


    def _resolve_inference_script_for_run(self, repo_root: Path, config_path: Path, preview_only: bool) -> str:
        return str(repo_root / "inference.py")

    def _cleanup_temp_runner_file(self) -> None:
        if not self._temp_runner_file:
            return
        try:
            if os.path.exists(self._temp_runner_file):
                os.remove(self._temp_runner_file)
        except Exception:
            pass
        self._temp_runner_file = None

    def _resolve_config_for_run(self, config_path: Path, preview_only: bool) -> str:
        guidance = self.guidance_spin.value()
        negative_prompt = self.negative_prompt_box.toPlainText().strip()

        custom_needed = (abs(guidance - 3.0) > 1e-9) or bool(negative_prompt)
        if not custom_needed:
            return str(config_path)
        if preview_only:
            return "<temp_config_generated_at_run_time>"

        self._cleanup_temp_config_file()
        temp_dir = self.framevision_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix="hiar_config_", suffix=".yaml", dir=str(temp_dir))
        os.close(fd)

        original = config_path.read_text(encoding="utf-8")
        updated = original
        updated = self._replace_yaml_scalar(updated, "guidance_scale", self._format_yaml_float(guidance))
        if negative_prompt:
            updated = self._replace_yaml_scalar(updated, "negative_prompt", json.dumps(negative_prompt, ensure_ascii=False))

        with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        self._temp_config_file = temp_path
        return temp_path

    def _replace_yaml_scalar(self, text: str, key: str, value_literal: str) -> str:
        pattern = rf'^(?P<indent>[ 	]*){re.escape(key)}\s*:\s*(?P<value>.*)$'
        replacement = rf'\g<indent>{key}: {value_literal}'
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            new_text = text.rstrip() + f"\n{key}: {value_literal}\n"
        return new_text

    def _format_yaml_float(self, value: float) -> str:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return text

    def _resolve_prompt_input(self, preview_only: bool) -> str:
        prompt_text = self.prompt_text.toPlainText().strip()
        prompt_file = self.prompt_file_edit.text().strip()

        if prompt_text:
            if preview_only:
                return "<temp_prompt_file_generated_at_run_time>"
            self._cleanup_temp_prompt_file()
            temp_dir = self.framevision_root / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix="hiar_prompts_", suffix=".txt", dir=str(temp_dir))
            os.close(fd)
            with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
                text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
            self._temp_prompt_file = temp_path
            return temp_path

        if prompt_file:
            if not Path(prompt_file).exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            return prompt_file

        return ""

    def _cleanup_temp_prompt_file(self) -> None:
        if self._temp_prompt_file:
            try:
                if os.path.exists(self._temp_prompt_file):
                    os.remove(self._temp_prompt_file)
            except Exception:
                pass
            self._temp_prompt_file = None

    def _cleanup_temp_config_file(self) -> None:
        if self._temp_config_file:
            try:
                if os.path.exists(self._temp_config_file):
                    os.remove(self._temp_config_file)
            except Exception:
                pass
            self._temp_config_file = None

    def _estimated_final_seconds(self, start_frames: int) -> float:
        return start_frames * (8.0 / 33.0)

    def _estimated_final_frames(self, start_frames: int) -> int:
        return max(1, int(round(self._estimated_final_seconds(start_frames) * 16.0)))

    def _update_frames_ui(self) -> None:
        start_frames = self.frames_spin.value()
        est_seconds = self._estimated_final_seconds(start_frames)
        est_final_frames = self._estimated_final_frames(start_frames)
        self.frames_info_label.setText(
            f"Fixed output: 832 × 480 @ 16 fps. Expected final length: about {est_final_frames} frames / {est_seconds:.1f} sec."
        )
        if start_frames >= 165:
            self.frames_warning_label.setText(
                "Warning: 165+ start frames may need 24 GB VRAM or can take a long time to finish."
            )
            self.frames_warning_label.show()
        else:
            self.frames_warning_label.clear()
            self.frames_warning_label.hide()

    # -------------------------
    # Logging helpers
    # -------------------------
    def _append_log(self, text: str) -> None:
        self.logs_box.appendPlainText(text)
        self._write_run_log(text + "\n")

    def _write_run_log(self, text: str) -> None:
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / "hiar_last_run.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def _quote_cmd(self, cmd: List[str]) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline(cmd)
        return " ".join(shlex.quote(p) for p in cmd)

    def _join_env_path(self, first: str, second: str) -> str:
        if not second:
            return first
        sep = ";" if os.name == "nt" else ":"
        return first + sep + second


class _HiARWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HiAR helper")
        self.resize(1100, 900)

        pane = HiARPane(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(pane)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll, 1)
        layout.addWidget(pane.build_footer_bar(), 0)
        self.setCentralWidget(central)


def main() -> int:
    app = QApplication(sys.argv)
    win = _HiARWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
