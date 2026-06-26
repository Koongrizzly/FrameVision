# -*- coding: utf-8 -*-
"""
FrameVision helper: Krea 2 GGUF

Runs Krea 2 through stable-diffusion.cpp / sd-cli.exe.
No Python ML environment is required. Models are expected in /models/krea2/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find_app_root(start: Optional[Path] = None) -> Path:
    """Find FrameVision/root app folder from this file or cwd."""
    candidates: List[Path] = []
    if start:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except Exception:
        pass
    candidates.append(Path.cwd().resolve())

    for base in candidates:
        p = base if base.is_dir() else base.parent
        for parent in [p, *p.parents]:
            if (parent / "presets").exists() or (parent / "models").exists() or (parent / "helpers").exists():
                return parent
    return Path.cwd().resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _safe_name(text: str, max_len: int = 44) -> str:
    text = re.sub(r"[^a-zA-Z0-9_ -]+", "", text).strip().replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return (text[:max_len] or "krea2")


def _quote_cmd(args: List[str]) -> str:
    def q(a: str) -> str:
        if not a:
            return '""'
        if any(c in a for c in ' &()[]{}^=;!\'",`'):
            return '"' + a.replace('"', '\\"') + '"'
        return a
    return " ".join(q(str(a)) for a in args)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class SdCliWorker(QThread):
    line = Signal(str)
    done = Signal(int, str)

    def __init__(self, args: List[str], cwd: Path, output_path: Path, parent=None):
        super().__init__(parent)
        self.args = args
        self.cwd = cwd
        self.output_path = output_path
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def run(self):  # noqa: D401
        try:
            self.line.emit(_quote_cmd(self.args))
            self.line.emit("")
            self._proc = subprocess.Popen(
                self.args,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
            )
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                self.line.emit(raw.rstrip("\n"))
                if self._stop_requested:
                    break
            rc = self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rc = -9
            self.line.emit("Process did not stop cleanly; killing it.")
            try:
                self._proc.kill()  # type: ignore[union-attr]
            except Exception:
                pass
        except Exception as exc:
            rc = -1
            self.line.emit(f"ERROR: {exc}")
        self.done.emit(rc, str(self.output_path))


class Krea2Widget(QWidget):
    def __init__(self, app_root: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.app_root = app_root or _find_app_root()
        self.settings_path = self.app_root / "presets" / "setsave" / "krea2.json"
        self.models_dir = self.app_root / "models" / "krea2"
        self.output_dir = self.app_root / "output" / "images" / "krea2"
        self.worker: Optional[SdCliWorker] = None
        self._last_output: Optional[Path] = None
        self._settings: Dict[str, object] = {}
        self._loading_settings = False
        self._autosave_ready = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._auto_save_settings)

        self.setWindowTitle("Krea 2")
        self._build_ui()
        self._load_settings()
        self.refresh_models()
        self._apply_loaded_settings()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._wrap_scroll(self._main_page()), "Krea 2")
        self.tabs.addTab(self._wrap_scroll(self._settings_page()), "Settings")
        root.addWidget(self.tabs, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_output_btn = QPushButton("Open output")
        self.open_output_btn.setToolTip("Open the configured output folder.")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the running sd-cli process.")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(34)
        self.generate_btn.setToolTip("Run Krea 2 with the current settings.")
        self.generate_btn.clicked.connect(self.generate)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.open_output_btn)
        bottom.addWidget(self.stop_btn)
        bottom.addWidget(self.generate_btn)
        root.addLayout(bottom)

    def _wrap_scroll(self, page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(page)
        return area

    def _main_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        split = QSplitter(Qt.Vertical)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        prompt_box = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_box)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the image")
        self.prompt_edit.setToolTip("Main text prompt sent to sd-cli.")
        self.prompt_edit.setMinimumHeight(120)
        prompt_layout.addWidget(self.prompt_edit)
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText("Negative prompt")
        self.negative_edit.setToolTip("Optional things to avoid. Has more effect when CFG is above 0.")
        prompt_layout.addWidget(self.negative_edit)
        controls_layout.addWidget(prompt_box)

        size_box = QGroupBox("Image")
        grid = QGridLayout(size_box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(10, 1)

        self.size_combo = NoWheelComboBox()
        self.size_combo.setFixedWidth(120)
        self.size_combo.setToolTip("Aspect mode. Fixed ratios keep width and height linked; Custom leaves both free.")
        for label in ["1:1", "16:9", "9:16", "Custom"]:
            self.size_combo.addItem(label)
        self.size_combo.currentIndexChanged.connect(self._aspect_mode_changed)

        self._syncing_size = False
        self.width_spin = NoWheelSpinBox()
        self.width_spin.setFixedWidth(120)
        self.width_spin.setRange(256, 4096)
        self.width_spin.setSingleStep(1)
        self.width_spin.setValue(1024)
        self.width_spin.setToolTip("Output width. Fixed aspect modes adjust height automatically.")
        self.height_spin = NoWheelSpinBox()
        self.height_spin.setFixedWidth(120)
        self.height_spin.setRange(256, 4096)
        self.height_spin.setSingleStep(1)
        self.height_spin.setValue(1024)
        self.height_spin.setToolTip("Output height. Fixed aspect modes adjust width automatically.")
        self.width_spin.valueChanged.connect(lambda _v: self._sync_aspect(changed='width'))
        self.height_spin.valueChanged.connect(lambda _v: self._sync_aspect(changed='height'))

        grid.addWidget(QLabel("Aspect"), 0, 0)
        grid.addWidget(self.size_combo, 0, 1)
        grid.addWidget(QLabel("W"), 0, 2)
        grid.addWidget(self.width_spin, 0, 3)
        grid.addWidget(QLabel("H"), 0, 4)
        grid.addWidget(self.height_spin, 0, 5)

        self.steps_spin = NoWheelSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(8)
        self.steps_spin.setToolTip("Denoising steps. Defaults switch with Base/Turbo model selection.")
        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("-1 uses a random seed. Use a fixed number to repeat a result.")
        grid.addWidget(QLabel("Steps"), 0, 6)
        grid.addWidget(self.steps_spin, 0, 7)
        grid.addWidget(QLabel("Seed"), 0, 8)
        grid.addWidget(self.seed_spin, 0, 9)

        self.cfg_spin = NoWheelDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 24.0)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setSingleStep(0.25)
        self.cfg_spin.setValue(1.0)
        self.cfg_spin.setToolTip("Prompt-following strength. Defaults: Turbo 1.0, Base/Raw 3.0.")
        self.guidance_spin = NoWheelDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.25)
        self.guidance_spin.setValue(3.5)
        self.guidance_spin.setToolTip("Distilled guidance value used by supported models. 3.5 is a good default.")
        grid.addWidget(QLabel("CFG"), 1, 0)
        grid.addWidget(self.cfg_spin, 1, 1)
        grid.addWidget(QLabel("Guidance"), 1, 2)
        grid.addWidget(self.guidance_spin, 1, 3)

        self.flow_shift_spin = NoWheelDoubleSpinBox()
        self.flow_shift_spin.setRange(-1.0, 10.0)
        self.flow_shift_spin.setDecimals(2)
        self.flow_shift_spin.setSingleStep(0.05)
        self.flow_shift_spin.setValue(1.15)
        self.flow_shift_spin.setSpecialValueText("auto")
        self.flow_shift_spin.setToolTip("Flow-model shift. 1.15 is a good starting point; auto lets sd-cli choose.")
        self.batch_spin = NoWheelSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(1)
        self.batch_spin.setToolTip("Number of images to create in one run. Keep 1 for first tests.")
        grid.addWidget(QLabel("Shift"), 1, 4)
        grid.addWidget(self.flow_shift_spin, 1, 5)
        grid.addWidget(QLabel("Batch"), 1, 6)
        grid.addWidget(self.batch_spin, 1, 7)

        self.sampler_combo = NoWheelComboBox()
        for x in ["euler", "auto", "euler_a", "heun", "dpm++2m", "dpm++2mv2", "ddim_trailing", "lcm", "res_multistep", "res_2s"]:
            self.sampler_combo.addItem(x)
        self.sampler_combo.setToolTip("Sampling method. Default: euler. Auto lets sd-cli choose.")
        self.scheduler_combo = NoWheelComboBox()
        for x in ["auto", "simple", "sgm_uniform", "karras", "exponential", "lcm", "ltx2", "logit_normal"]:
            self.scheduler_combo.addItem(x)
        self.scheduler_combo.setToolTip("Denoiser scheduler. Auto lets sd-cli choose.")
        for _w in (
            self.steps_spin, self.seed_spin, self.cfg_spin, self.guidance_spin,
            self.flow_shift_spin, self.batch_spin, self.sampler_combo, self.scheduler_combo
        ):
            _w.setFixedWidth(120)
        grid.addWidget(QLabel("Sampler"), 2, 0)
        grid.addWidget(self.sampler_combo, 2, 1, 1, 3)
        grid.addWidget(QLabel("Scheduler"), 2, 4)
        grid.addWidget(self.scheduler_combo, 2, 5, 1, 3)

        controls_layout.addWidget(size_box)
        controls_layout.addStretch(1)

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setStyleSheet("border: 1px solid rgba(128,128,128,80); border-radius: 6px;")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        split.addWidget(controls)
        split.addWidget(self.preview_label)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)
        return page

    def _settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        paths_box = QGroupBox("Paths")
        grid = QGridLayout(paths_box)
        grid.setColumnStretch(1, 1)

        self.sdcli_edit = QLineEdit()
        self.sdcli_edit.setToolTip("Path to sd-cli.exe. Default: /presets/bin/sd-cli.exe")
        grid.addWidget(QLabel("sd-cli"), 0, 0)
        grid.addWidget(self.sdcli_edit, 0, 1)
        b = QPushButton("Browse")
        b.clicked.connect(lambda: self._browse_file(self.sdcli_edit, "sd-cli.exe", (".exe",)))
        grid.addWidget(b, 0, 2)

        self.model_combo = NoWheelComboBox()
        self.model_combo.setToolTip("Krea 2 GGUF from /models/krea2/. Switching Base/Turbo updates steps and CFG defaults.")
        grid.addWidget(QLabel("Model"), 1, 0)
        grid.addWidget(self.model_combo, 1, 1)
        model_browse = QPushButton("Browse")
        model_browse.clicked.connect(lambda: self._browse_into_combo(self.model_combo, "GGUF", (".gguf",)))
        grid.addWidget(model_browse, 1, 2)

        self.llm_combo = NoWheelComboBox()
        self.llm_combo.setToolTip("Qwen3VL 4B GGUF text encoder from /models/krea2/.")
        grid.addWidget(QLabel("Text encoder"), 2, 0)
        grid.addWidget(self.llm_combo, 2, 1)
        llm_browse = QPushButton("Browse")
        llm_browse.clicked.connect(lambda: self._browse_into_combo(self.llm_combo, "GGUF", (".gguf",)))
        grid.addWidget(llm_browse, 2, 2)

        self.vae_edit = QLineEdit()
        self.vae_edit.setToolTip("Path to the Wan 2.1 VAE used by Krea 2.")
        grid.addWidget(QLabel("VAE"), 3, 0)
        grid.addWidget(self.vae_edit, 3, 1)
        vae_b = QPushButton("Browse")
        vae_b.clicked.connect(lambda: self._browse_file(self.vae_edit, "VAE", (".safetensors", ".gguf")))
        grid.addWidget(vae_b, 3, 2)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setToolTip("Folder where generated images are saved.")
        grid.addWidget(QLabel("Output"), 4, 0)
        grid.addWidget(self.output_dir_edit, 4, 1)
        out_b = QPushButton("Browse")
        out_b.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        grid.addWidget(out_b, 4, 2)

        refresh = QPushButton("Refresh")
        refresh.setToolTip("Rescan /models/krea2/ for GGUF files.")
        refresh.clicked.connect(self.refresh_models)
        grid.addWidget(refresh, 5, 2)
        layout.addWidget(paths_box)

        run_box = QGroupBox("Run")
        run = QGridLayout(run_box)
        run.setColumnStretch(1, 1)
        self.offload_chk = QCheckBox("Offload")
        self.offload_chk.setChecked(False)
        self.offload_chk.setToolTip("Use system RAM as a safety net for larger files or resolutions. Usually slower; leave off when the model fits in VRAM.")
        self.diff_fa_chk = QCheckBox("Diffusion FA")
        self.diff_fa_chk.setChecked(True)
        self.diff_fa_chk.setToolTip("Use flash attention in the diffusion model when supported. Usually faster and lighter on VRAM.")
        self.vae_tiling_chk = QCheckBox("VAE tiling")
        self.vae_tiling_chk.setToolTip("Decode the final image in smaller VAE tiles. Helps when decode runs out of VRAM, but can be slower.")
        self.verbose_chk = QCheckBox("Verbose")
        self.verbose_chk.setChecked(True)
        self.verbose_chk.setToolTip("Show detailed sd-cli logs for loading, VRAM use, and generation progress.")
        run.addWidget(self.offload_chk, 0, 0)
        run.addWidget(self.diff_fa_chk, 0, 1)
        run.addWidget(self.vae_tiling_chk, 1, 0)
        run.addWidget(self.verbose_chk, 1, 1)

        self.backend_edit = QLineEdit()
        self.backend_edit.setPlaceholderText("")
        self.backend_edit.setToolTip("Optional backend assignment. Example: vae=cpu or diffusion=cuda0,vae=cpu.")
        self.params_backend_edit = QLineEdit()
        self.params_backend_edit.setPlaceholderText("")
        self.params_backend_edit.setToolTip("Optional parameter backend assignment. Example: disk, cpu, diffusion=disk.")
        run.addWidget(QLabel("Backend"), 2, 0)
        run.addWidget(self.backend_edit, 2, 1)
        run.addWidget(QLabel("Params"), 3, 0)
        run.addWidget(self.params_backend_edit, 3, 1)

        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.setToolTip("Extra sd-cli arguments appended at the end. Use only for testing flags not exposed in the UI.")
        run.addWidget(QLabel("Extra"), 4, 0)
        run.addWidget(self.extra_args_edit, 4, 1)
        layout.addWidget(run_box)

        optional_box = QGroupBox("Optional")
        opt = QGridLayout(optional_box)
        opt.setColumnStretch(1, 1)
        self.init_img_edit = QLineEdit()
        self.init_img_edit.setToolTip("Optional init image for img2img-style testing. Uses Strength below.")
        init_btn = QPushButton("Browse")
        init_btn.clicked.connect(lambda: self._browse_file(self.init_img_edit, "Image", IMAGE_EXTS))
        clear_init = QPushButton("Clear")
        clear_init.clicked.connect(lambda: self.init_img_edit.clear())
        opt.addWidget(QLabel("Init"), 0, 0)
        opt.addWidget(self.init_img_edit, 0, 1)
        opt.addWidget(init_btn, 0, 2)
        opt.addWidget(clear_init, 0, 3)

        self.strength_spin = NoWheelDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setValue(0.75)
        self.strength_spin.setToolTip("How strongly the init image is changed. Only used when Init is set.")
        self.disable_metadata_chk = QCheckBox("No metadata")
        self.disable_metadata_chk.setToolTip("Do not write generation settings into the saved image metadata.")
        opt.addWidget(QLabel("Strength"), 1, 0)
        opt.addWidget(self.strength_spin, 1, 1)
        opt.addWidget(self.disable_metadata_chk, 1, 2, 1, 2)
        layout.addWidget(optional_box)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(180)
        self.log_edit.setToolTip("sd-cli command output and progress messages.")
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_box)

        cmd_box = QGroupBox("Command")
        cmd_layout = QVBoxLayout(cmd_box)
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(140)
        self.command_preview.setToolTip("Exact command that will be run.")
        cmd_layout.addWidget(self.command_preview)
        copy_btn = QPushButton("Copy command")
        copy_btn.clicked.connect(self.copy_command)
        cmd_layout.addWidget(copy_btn, 0, Qt.AlignRight)
        layout.addWidget(cmd_box)

        save_row = QHBoxLayout()
        save_btn = QPushButton("Save settings")
        save_btn.clicked.connect(self.save_settings)
        defaults_btn = QPushButton("Defaults")
        defaults_btn.clicked.connect(self.reset_defaults)
        save_row.addStretch(1)
        save_row.addWidget(defaults_btn)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)
        layout.addStretch(1)
        return page

    # ---------- settings / discovery ----------
    def _load_settings(self):
        self._settings = {}
        if self.settings_path.exists():
            try:
                self._settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            except Exception:
                self._settings = {}

    def _apply_loaded_settings(self):
        self._loading_settings = True
        s = self._settings
        self.sdcli_edit.setText(str(s.get("sdcli", _rel(self.app_root / "presets" / "bin" / "sd-cli.exe", self.app_root))))
        self.vae_edit.setText(str(s.get("vae", _rel(self.models_dir / "wan_2.1_vae.safetensors", self.app_root))))
        self.output_dir_edit.setText(str(s.get("output_dir", _rel(self.output_dir, self.app_root))))
        self.prompt_edit.setPlainText(str(s.get("prompt", "")))
        self.negative_edit.setText(str(s.get("negative", "")))
        self._set_combo_text(self.size_combo, str(s.get("aspect", "1:1")))
        self.width_spin.setValue(int(s.get("width", 1024)))
        self.height_spin.setValue(int(s.get("height", 1024)))
        self._sync_aspect(changed="width")
        self.steps_spin.setValue(int(s.get("steps", 8)))
        self.seed_spin.setValue(int(s.get("seed", -1)))
        self.cfg_spin.setValue(float(s.get("cfg", 1.0)))
        self.guidance_spin.setValue(float(s.get("guidance", 3.5)))
        self.flow_shift_spin.setValue(float(s.get("flow_shift", 1.15)))
        self.batch_spin.setValue(int(s.get("batch", 1)))
        self.strength_spin.setValue(float(s.get("strength", 0.75)))
        self.offload_chk.setChecked(bool(s.get("offload", False)))
        self.diff_fa_chk.setChecked(bool(s.get("diffusion_fa", True)))
        self.vae_tiling_chk.setChecked(bool(s.get("vae_tiling", False)))
        self.verbose_chk.setChecked(bool(s.get("verbose", True)))
        self.disable_metadata_chk.setChecked(bool(s.get("disable_metadata", False)))
        self.backend_edit.setText(str(s.get("backend", "")))
        self.params_backend_edit.setText(str(s.get("params_backend", "")))
        self.extra_args_edit.setText(str(s.get("extra_args", "")))
        self._set_combo_text(self.model_combo, str(s.get("model", "")))
        self._set_combo_text(self.llm_combo, str(s.get("llm", "")))
        self._set_combo_text(self.sampler_combo, str(s.get("sampler", "euler")))
        self._set_combo_text(self.scheduler_combo, str(s.get("scheduler", "auto")))
        self.update_command_preview()

        if not getattr(self, "_signals_connected", False):
            for w in [
                self.prompt_edit, self.negative_edit, self.size_combo, self.width_spin, self.height_spin, self.steps_spin,
                self.seed_spin, self.cfg_spin, self.guidance_spin, self.flow_shift_spin, self.batch_spin,
                self.init_img_edit, self.strength_spin, self.sdcli_edit, self.vae_edit, self.output_dir_edit,
                self.backend_edit, self.params_backend_edit, self.extra_args_edit,
                self.model_combo, self.llm_combo, self.sampler_combo, self.scheduler_combo,
                self.offload_chk, self.diff_fa_chk, self.vae_tiling_chk, self.verbose_chk, self.disable_metadata_chk,
            ]:
                try:
                    if hasattr(w, "textChanged"):
                        w.textChanged.connect(self.update_command_preview)
                        w.textChanged.connect(self._schedule_save_settings)
                    if hasattr(w, "valueChanged"):
                        w.valueChanged.connect(self.update_command_preview)
                        w.valueChanged.connect(self._schedule_save_settings)
                    if hasattr(w, "currentIndexChanged"):
                        w.currentIndexChanged.connect(self.update_command_preview)
                        w.currentIndexChanged.connect(self._schedule_save_settings)
                    if hasattr(w, "toggled"):
                        w.toggled.connect(self.update_command_preview)
                        w.toggled.connect(self._schedule_save_settings)
                except Exception:
                    pass
            self.model_combo.currentIndexChanged.connect(self._model_changed)
            self._signals_connected = True
        self._loading_settings = False
        self._autosave_ready = True

    def _schedule_save_settings(self, *args):
        if self._loading_settings or not self._autosave_ready:
            return
        self._save_timer.start()

    def _auto_save_settings(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(self.current_settings(), indent=2), encoding="utf-8")
        except Exception as exc:
            self.status_label.setText(f"Autosave failed: {exc}")

    def current_settings(self) -> Dict[str, object]:
        return {
            "sdcli": self.sdcli_edit.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "llm": self.llm_combo.currentText().strip(),
            "vae": self.vae_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "prompt": self.prompt_edit.toPlainText(),
            "negative": self.negative_edit.text(),
            "aspect": self.size_combo.currentText(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "seed": self.seed_spin.value(),
            "cfg": self.cfg_spin.value(),
            "guidance": self.guidance_spin.value(),
            "flow_shift": self.flow_shift_spin.value(),
            "batch": self.batch_spin.value(),
            "strength": self.strength_spin.value(),
            "offload": self.offload_chk.isChecked(),
            "diffusion_fa": self.diff_fa_chk.isChecked(),
            "vae_tiling": self.vae_tiling_chk.isChecked(),
            "verbose": self.verbose_chk.isChecked(),
            "disable_metadata": self.disable_metadata_chk.isChecked(),
            "backend": self.backend_edit.text().strip(),
            "params_backend": self.params_backend_edit.text().strip(),
            "sampler": self.sampler_combo.currentText(),
            "scheduler": self.scheduler_combo.currentText(),
            "extra_args": self.extra_args_edit.text().strip(),
        }

    def save_settings(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(self.current_settings(), indent=2), encoding="utf-8")
            self.status_label.setText("Settings saved")
        except Exception as exc:
            QMessageBox.warning(self, "Krea 2", f"Could not save settings:\n{exc}")

    def reset_defaults(self):
        self._settings = {}
        self._apply_loaded_settings()
        self.save_settings()
        self.status_label.setText("Defaults restored")

    def refresh_models(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        old_model = self.model_combo.currentText().strip() if hasattr(self, "model_combo") else ""
        old_llm = self.llm_combo.currentText().strip() if hasattr(self, "llm_combo") else ""
        if not hasattr(self, "model_combo"):
            return
        self.model_combo.blockSignals(True)
        self.llm_combo.blockSignals(True)
        self.model_combo.clear()
        self.llm_combo.clear()

        ggufs = sorted(self.models_dir.glob("*.gguf"), key=lambda p: ("turbo" not in p.name.lower(), p.name.lower()))
        models = [p for p in ggufs if "krea" in p.name.lower()]
        llms = [p for p in ggufs if "qwen" in p.name.lower() and ("vl" in p.name.lower() or "vision" in p.name.lower())]
        for p in models:
            self.model_combo.addItem(_rel(p, self.app_root))
        for p in llms:
            self.llm_combo.addItem(_rel(p, self.app_root))
        if not models:
            self.model_combo.addItem("")
        if not llms:
            self.llm_combo.addItem("")
        self.model_combo.blockSignals(False)
        self.llm_combo.blockSignals(False)
        if old_model:
            self._set_combo_text(self.model_combo, old_model)
        if old_llm:
            self._set_combo_text(self.llm_combo, old_llm)
        self._apply_model_defaults()
        self.update_command_preview()

    def _model_kind(self) -> str:
        name = self.model_combo.currentText().lower()
        if "turbo" in name:
            return "turbo"
        if "krea" in name:
            return "base"
        return ""

    def _apply_model_defaults(self):
        kind = self._model_kind()
        if kind == "turbo":
            self.steps_spin.setValue(8)
            self.cfg_spin.setValue(1.0)
        elif kind == "base":
            self.steps_spin.setValue(30)
            self.cfg_spin.setValue(3.0)

    def _model_changed(self, *_args):
        self._apply_model_defaults()
        self.update_command_preview()

    def _aspect_mode_changed(self, *_args):
        self._sync_aspect(changed="width")
        self.update_command_preview()

    def _sync_aspect(self, changed: str = "width"):
        if getattr(self, "_syncing_size", False):
            return
        mode = self.size_combo.currentText()
        if mode == "Custom":
            return
        self._syncing_size = True
        try:
            if mode == "1:1":
                if changed == "width":
                    self.height_spin.setValue(self.width_spin.value())
                else:
                    self.width_spin.setValue(self.height_spin.value())
            elif mode == "16:9":
                if changed == "width":
                    self.height_spin.setValue(max(1, round(self.width_spin.value() * 9 / 16)))
                else:
                    self.width_spin.setValue(max(1, round(self.height_spin.value() * 16 / 9)))
            elif mode == "9:16":
                if changed == "width":
                    self.height_spin.setValue(max(1, round(self.width_spin.value() * 16 / 9)))
                else:
                    self.width_spin.setValue(max(1, round(self.height_spin.value() * 9 / 16)))
        finally:
            self._syncing_size = False

    # ---------- actions ----------
    def _path_from_text(self, text: str) -> Path:
        p = Path(text.strip().strip('"'))
        if not p.is_absolute():
            p = self.app_root / p
        return p

    def _selected_model_path(self) -> Path:
        return self._path_from_text(self.model_combo.currentText())

    def _selected_llm_path(self) -> Path:
        return self._path_from_text(self.llm_combo.currentText())

    def build_command(self) -> Tuple[List[str], Path]:
        sdcli = self._path_from_text(self.sdcli_edit.text())
        model = self._selected_model_path()
        llm = self._selected_llm_path()
        vae = self._path_from_text(self.vae_edit.text())
        out_dir = self._path_from_text(self.output_dir_edit.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = self.prompt_edit.toPlainText().strip()
        name = _safe_name(prompt.splitlines()[0] if prompt else "krea2")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.batch_spin.value() > 1:
            output_path = out_dir / f"{stamp}_{name}_%03d.png"
        else:
            output_path = out_dir / f"{stamp}_{name}.png"

        args: List[str] = [
            str(sdcli),
            "--diffusion-model", str(model),
            "--llm", str(llm),
            "--vae", str(vae),
            "-p", prompt,
            "--steps", str(self.steps_spin.value()),
            "--cfg-scale", str(self.cfg_spin.value()),
            "--guidance", str(self.guidance_spin.value()),
            "--width", str(self.width_spin.value()),
            "--height", str(self.height_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--batch-count", str(self.batch_spin.value()),
            "--output", str(output_path),
        ]
        neg = self.negative_edit.text().strip()
        if neg:
            args += ["--negative-prompt", neg]
        # SpecialValueText displays auto at the minimum value. Omit flag when auto.
        if self.flow_shift_spin.value() >= 0:
            args += ["--flow-shift", str(self.flow_shift_spin.value())]
        init_img = self.init_img_edit.text().strip()
        if init_img:
            args += ["--init-img", str(self._path_from_text(init_img)), "--strength", str(self.strength_spin.value())]
        if self.diff_fa_chk.isChecked():
            args.append("--diffusion-fa")
        if self.offload_chk.isChecked():
            args.append("--offload-to-cpu")
        if self.vae_tiling_chk.isChecked():
            args.append("--vae-tiling")
        if self.disable_metadata_chk.isChecked():
            args.append("--disable-image-metadata")
        if self.verbose_chk.isChecked():
            args.append("-v")
        backend = self.backend_edit.text().strip()
        if backend:
            args += ["--backend", backend]
        params_backend = self.params_backend_edit.text().strip()
        if params_backend:
            args += ["--params-backend", params_backend]
        sampler = self.sampler_combo.currentText().strip()
        if sampler and sampler != "auto":
            args += ["--sampling-method", sampler]
        scheduler = self.scheduler_combo.currentText().strip()
        if scheduler and scheduler != "auto":
            args += ["--scheduler", scheduler]
        extra = self.extra_args_edit.text().strip()
        if extra:
            # Simple split is good enough for internal testing flags. Keep paths in normal fields above.
            args += extra.split()
        return args, output_path

    def validate(self) -> bool:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Krea 2", "Prompt is empty.")
            return False
        checks = [
            ("sd-cli", self._path_from_text(self.sdcli_edit.text())),
            ("Krea 2 GGUF", self._selected_model_path()),
            ("Qwen3VL text encoder", self._selected_llm_path()),
            ("VAE", self._path_from_text(self.vae_edit.text())),
        ]
        missing = [f"{name}: {path}" for name, path in checks if not path.exists()]
        if missing:
            QMessageBox.warning(self, "Krea 2", "Missing file(s):\n\n" + "\n".join(missing))
            self.tabs.setCurrentIndex(1)
            return False
        return True

    def generate(self):
        if self.worker and self.worker.isRunning():
            return
        if not self.validate():
            return
        self.save_settings()
        args, output_path = self.build_command()
        self._last_output = output_path
        self.log_edit.clear()
        self.status_label.setText("Running")
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker = SdCliWorker(args=args, cwd=self.app_root, output_path=output_path, parent=self)
        self.worker.line.connect(self._append_log)
        self.worker.done.connect(self._generation_done)
        self.worker.start()

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping")

    def _generation_done(self, rc: int, out: str):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Done" if rc == 0 else f"Failed ({rc})")
        self._load_preview(Path(out))

    def _append_log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.moveCursor(QTextCursor.End)

    def _load_preview(self, output_path: Path):
        candidates: List[Path] = []
        if "%" in str(output_path):
            parent = output_path.parent
            stem = output_path.name.split("%")[0]
            candidates = sorted(parent.glob(stem + "*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            candidates = [output_path]
        for path in candidates:
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.preview_label.setToolTip(str(path))
                    return

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._last_output:
            self._load_preview(self._last_output)

    def update_command_preview(self, *args):
        if not hasattr(self, "command_preview"):
            return
        try:
            cmd, _ = self.build_command()
            self.command_preview.setPlainText(_quote_cmd(cmd))
        except Exception as exc:
            self.command_preview.setPlainText(str(exc))

    def copy_command(self):
        QApplication.clipboard().setText(self.command_preview.toPlainText())
        self.status_label.setText("Command copied")

    def open_output_folder(self):
        out_dir = self._path_from_text(self.output_dir_edit.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(out_dir)])
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])

    def closeEvent(self, event):  # noqa: N802 - Qt override
        try:
            self._save_timer.stop()
            self._auto_save_settings()
        except Exception:
            pass
        super().closeEvent(event)

    # ---------- small helpers ----------
    def _browse_file(self, edit: QLineEdit, title: str, exts: Tuple[str, ...]):
        filt = "Files (" + " ".join("*" + e for e in exts) + ");;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, title, str(self.app_root), filt)
        if path:
            edit.setText(_rel(Path(path), self.app_root))

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Folder", str(self.app_root))
        if path:
            edit.setText(_rel(Path(path), self.app_root))

    def _browse_into_combo(self, combo: QComboBox, title: str, exts: Tuple[str, ...]):
        filt = "Files (" + " ".join("*" + e for e in exts) + ");;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, title, str(self.models_dir), filt)
        if path:
            text = _rel(Path(path), self.app_root)
            self._set_combo_text(combo, text)

    def _set_combo_text(self, combo: QComboBox, text: str):
        if not text:
            return
        idx = combo.findText(text)
        if idx < 0:
            combo.addItem(text)
            idx = combo.findText(text)
        combo.setCurrentIndex(idx)



# Common FrameVision-friendly factory names.
def create_widget(parent=None):
    return Krea2Widget(parent=parent)


def create_tab(parent=None):
    return Krea2Widget(parent=parent)


def get_widget(parent=None):
    return Krea2Widget(parent=parent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Krea2Widget()
    w.resize(QSize(1100, 820))
    w.show()
    sys.exit(app.exec())
