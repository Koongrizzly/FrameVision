from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
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
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Wan 2.2 Distill INT8 (4-step)"
MODEL_NAME = "wan22_int8"
SETTINGS_REL = Path("presets") / "setsave" / f"{MODEL_NAME}.json"
LOGS_REL = Path("logs")
DEFAULT_ENV_NAME = ".wan22_i2v"
DEFAULT_OUT_SUBDIR = Path("output") / "wan22_int8"

HIGH_NOISE_FILE = "wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors"
LOW_NOISE_FILE = "wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors"


@dataclass
class UiState:
    env_python: str = ""
    lightx2v_repo: str = ""
    model_root: str = ""
    high_noise_ckpt: str = ""
    low_noise_ckpt: str = ""
    t5_path: str = ""
    clip_path: str = ""
    vae_path: str = ""
    xlm_roberta_dir: str = ""
    google_dir: str = ""
    input_image: str = ""
    output_dir: str = ""
    output_name: str = "wan22_int8_output"
    prompt: str = "A cinematic fantasy shot of a bear DJ performing in a neon nightclub while the crowd dances, detailed lighting, vivid motion, smooth camera feel."
    negative_prompt: str = "low quality, blurry, overexposed, static image, watermark, text, subtitles, ugly hands, deformed face, duplicate limbs"
    seed: int = -1
    width: int = 832
    height: int = 480
    num_frames: int = 81
    infer_steps: int = 4
    sample_shift: float = 5.0
    cfg_high: float = 1.0
    cfg_low: float = 1.0
    attn_mode: str = "sage_attn2"
    cpu_offload: bool = True
    text_encoder_offload: bool = True
    image_encoder_offload: bool = False
    vae_offload: bool = False
    offload_granularity: str = "block"
    save_last_preview: bool = True


class ProcessWorker(QObject):
    log = Signal(str)
    finished = Signal(int, str)
    started = Signal()

    def __init__(self, command: list[str], workdir: str, env: dict[str, str]):
        super().__init__()
        self.command = command
        self.workdir = workdir
        self.env = env
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stop_requested = False

    def run(self) -> None:
        self.started.emit()
        try:
            self.log.emit("[wan22-int8] starting process...\n")
            self.log.emit(f"[wan22-int8] cwd={self.workdir}\n")
            self.log.emit(f"[wan22-int8] cmd={format_command(self.command)}\n\n")
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.workdir,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.log.emit(line)
                if self._stop_requested:
                    break
            rc = self._proc.wait()
            self.finished.emit(rc, "stopped" if self._stop_requested else "finished")
        except Exception as exc:
            self.log.emit(f"\n[wan22-int8] ERROR: {exc}\n")
            self.finished.emit(1, "error")

    def stop(self) -> None:
        self._stop_requested = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


class ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("No image loaded")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(240)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QLabel { background: #111; color: #bbb; }")
        self.setScaledContents(False)
        self._image_path: Optional[str] = None
        self._orig_pixmap: Optional[QPixmap] = None

    def clear_image(self, message: str = "No image loaded") -> None:
        self._image_path = None
        self._orig_pixmap = None
        self.clear()
        self.setText(message)

    def load_image(self, path: str) -> bool:
        try:
            path = str(Path(path))
            if not path or not Path(path).is_file():
                self.clear_image("No image loaded")
                return False

            pix = QPixmap(path)
            if pix.isNull():
                self.clear_image("Could not load image")
                return False

            self._image_path = path
            self._orig_pixmap = pix
            self._update_scaled_pixmap()
            return True
        except Exception:
            self.clear_image("Could not load image")
            return False

    def _update_scaled_pixmap(self) -> None:
        if not self._orig_pixmap or self._orig_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 1 or target.height() <= 1:
            return
        scaled = self._orig_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_pixmap()


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, *, expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._content = content
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

        top = QVBoxLayout(self)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(self._toggle)

        self._content_frame = QFrame()
        frame_lay = QVBoxLayout(self._content_frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        frame_lay.addWidget(content)
        self._content_frame.setVisible(expanded)
        top.addWidget(self._content_frame)

        self._toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)

class Wan22Int8Pane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.root_dir = self._guess_root_dir()
        self.settings_path = self.root_dir / SETTINGS_REL
        self.logs_dir = self.root_dir / LOGS_REL
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

        self._loading_settings = True
        self._worker: Optional[ProcessWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()
        self._apply_base_defaults()
        self._load_settings_protected()
        self._auto_detect_missing_only()
        self._loading_settings = False
        self._refresh_preview()
        QTimer.singleShot(0, self._save_settings_safe)

    # ---------- paths ----------
    def _guess_root_dir(self) -> Path:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "helpers":
            return here.parent.parent
        return here.parent

    def _default_env_python(self) -> Path:
        candidates = [
            self.root_dir / "environments" / ".wan22_i2v" / "Scripts" / "python.exe",
            self.root_dir / "environments" / DEFAULT_ENV_NAME / "Scripts" / "python.exe",
            self.root_dir / "environments" / ".wan22_int8" / "Scripts" / "python.exe",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return candidates[0]

    def _default_repo(self) -> Path:
        return self.root_dir / "models" / "wan22_i2v_lightx2v" / "LightX2V"

    def _default_model_root(self) -> Path:
        return self.root_dir / "models" / "wan22_i2v_lightx2v" / "wan2.2_i2v_int8_4step"

    def _default_output_dir(self) -> Path:
        return self.root_dir / DEFAULT_OUT_SUBDIR

    # ---------- ui ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        header = QLabel(APP_TITLE)
        header.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(header)

        info = QLabel(
            "Uses LightX2V with Wan 2.2 distilled 4-step INT8 image-to-video. "
            "This packaged INT8 model mostly wants its own model_root layout left alone: high_noise_model, low_noise_model, text_encoder, vae and google/umt5-xxl under model_root."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(self._build_generation_group())
        layout.addWidget(self._build_offload_group())
        layout.addWidget(self._build_preview_group())
        layout.addWidget(self._build_paths_section())
        layout.addWidget(self._build_logs_section())
        layout.addStretch(1)

        self._wire_change_saves()

    def _build_paths_section(self) -> QWidget:
        return CollapsibleSection("Paths and files", self._build_paths_group(), expanded=False)

    def _build_logs_section(self) -> QWidget:
        return CollapsibleSection("Logs", self._build_logs_group(), expanded=False)

    def _build_paths_group(self) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)
        r = 0

        self.env_python_edit = self._path_row(grid, r, "Env python", file_mode=True); r += 1
        self.repo_edit = self._path_row(grid, r, "LightX2V repo", dir_mode=True); r += 1
        self.model_root_edit = self._path_row(grid, r, "Model root", dir_mode=True); r += 1
        self.high_noise_edit = self._path_row(grid, r, "High-noise INT8 model file", file_mode=True); r += 1
        self.low_noise_edit = self._path_row(grid, r, "Low-noise INT8 model file", file_mode=True); r += 1
        self.t5_edit = self._path_row(grid, r, "Text encoder folder", dir_mode=True); r += 1
        self.clip_edit = self._path_row(grid, r, "Legacy CLIP path (ignored for this INT8 pack)", dir_mode=True); r += 1
        self.vae_edit = self._path_row(grid, r, "VAE folder", dir_mode=True); r += 1
        self.xlm_edit = self._path_row(grid, r, "Legacy tokenizer/cache path (ignored)", dir_mode=True); r += 1
        self.google_edit = self._path_row(grid, r, "Legacy google path (ignored; comes from model_root)", dir_mode=True); r += 1
        self.input_image_edit = self._path_row(grid, r, "Input image", file_mode=True, image_pick=True); r += 1
        self.output_dir_edit = self._path_row(grid, r, "Output dir", dir_mode=True); r += 1

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("wan22_int8_output")
        grid.addWidget(QLabel("Output name"), r, 0)
        grid.addWidget(self.output_name_edit, r, 1, 1, 2)
        r += 1

        btns = QHBoxLayout()
        self.auto_detect_btn = QPushButton("Auto detect paths")
        self.validate_btn = QPushButton("Validate setup")
        self.open_output_btn = QPushButton("Open output folder")
        btns.addWidget(self.auto_detect_btn)
        btns.addWidget(self.validate_btn)
        btns.addWidget(self.open_output_btn)
        btns.addStretch(1)
        grid.addLayout(btns, r, 0, 1, 3)

        self.auto_detect_btn.clicked.connect(self._auto_detect_all)
        self.validate_btn.clicked.connect(self._validate_and_show)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        return box

    def _build_generation_group(self) -> QWidget:
        box = QGroupBox("Generation")
        form = QFormLayout(box)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(110)
        self.negative_edit = QTextEdit()
        self.negative_edit.setMinimumHeight(90)

        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 2_147_483_647); self.seed_spin.setValue(-1)
        self.width_spin = QSpinBox(); self.width_spin.setRange(128, 4096); self.width_spin.setSingleStep(32)
        self.height_spin = QSpinBox(); self.height_spin.setRange(128, 4096); self.height_spin.setSingleStep(32)
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(1, 1000)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 100)
        self.sample_shift_spin = QDoubleSpinBox(); self.sample_shift_spin.setRange(0.0, 50.0); self.sample_shift_spin.setSingleStep(0.1); self.sample_shift_spin.setDecimals(2)
        self.cfg_high_spin = QDoubleSpinBox(); self.cfg_high_spin.setRange(0.0, 20.0); self.cfg_high_spin.setSingleStep(0.1); self.cfg_high_spin.setDecimals(2)
        self.cfg_low_spin = QDoubleSpinBox(); self.cfg_low_spin.setRange(0.0, 20.0); self.cfg_low_spin.setSingleStep(0.1); self.cfg_low_spin.setDecimals(2)

        self.attn_combo = QComboBox()
        self.attn_combo.addItems(["sage_attn2", "flash", "torch", "sdpa"])

        dims = QWidget(); dims_l = QHBoxLayout(dims); dims_l.setContentsMargins(0, 0, 0, 0)
        dims_l.addWidget(self.width_spin); dims_l.addWidget(QLabel("x")); dims_l.addWidget(self.height_spin); dims_l.addStretch(1)

        cfgs = QWidget(); cfg_l = QHBoxLayout(cfgs); cfg_l.setContentsMargins(0, 0, 0, 0)
        cfg_l.addWidget(QLabel("high")); cfg_l.addWidget(self.cfg_high_spin)
        cfg_l.addWidget(QLabel("low")); cfg_l.addWidget(self.cfg_low_spin)
        cfg_l.addStretch(1)

        form.addRow("Prompt", self.prompt_edit)
        form.addRow("Negative prompt", self.negative_edit)
        form.addRow("Seed (-1 = random)", self.seed_spin)
        form.addRow("Resolution", dims)
        form.addRow("Frames", self.frames_spin)
        form.addRow("Steps", self.steps_spin)
        form.addRow("Sample shift", self.sample_shift_spin)
        form.addRow("Guidance scale", cfgs)
        form.addRow("Attention mode", self.attn_combo)

        row = QHBoxLayout()
        self.preview_cmd_btn = QPushButton("Preview command")
        self.run_btn = QPushButton("Generate")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.preview_cmd_btn)
        row.addStretch(1)
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        form.addRow(row)

        self.preview_cmd_btn.clicked.connect(self._refresh_preview)
        self.run_btn.clicked.connect(self._start_run)
        self.stop_btn.clicked.connect(self._stop_run)
        return box

    def _build_offload_group(self) -> QWidget:
        box = QGroupBox("Offload / memory")
        form = QFormLayout(box)
        self.cpu_offload_chk = QCheckBox("Enable CPU offload")
        self.text_offload_chk = QCheckBox("Offload text encoder")
        self.image_offload_chk = QCheckBox("Offload image encoder")
        self.vae_offload_chk = QCheckBox("Offload VAE")
        self.save_preview_chk = QCheckBox("Save last frame preview JPG")

        self.offload_combo = QComboBox()
        self.offload_combo.addItems(["block", "phase"])

        form.addRow(self.cpu_offload_chk)
        form.addRow(self.text_offload_chk)
        form.addRow(self.image_offload_chk)
        form.addRow(self.vae_offload_chk)
        form.addRow("Granularity", self.offload_combo)
        form.addRow(self.save_preview_chk)
        return box

    def _build_preview_group(self) -> QWidget:
        box = QGroupBox("Command preview and image")
        lay = QVBoxLayout(box)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(140)
        self.preview_image = ImageLabel()
        lay.addWidget(self.command_preview)
        lay.addWidget(self.preview_image)
        return box

    def _build_logs_group(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.logs_edit = QPlainTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setMinimumHeight(260)
        lay.addWidget(self.logs_edit)
        return box

    def _path_row(self, grid: QGridLayout, row: int, label: str, *, file_mode: bool = False, dir_mode: bool = False, image_pick: bool = False) -> QLineEdit:
        edit = QLineEdit()
        browse = QPushButton("Browse")
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(edit, row, 1)
        grid.addWidget(browse, row, 2)

        def _pick() -> None:
            current = edit.text().strip() or str(self.root_dir)
            if file_mode:
                filt = "All files (*.*)"
                if image_pick:
                    filt = "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
                path, _ = QFileDialog.getOpenFileName(self, label, current, filt)
            elif dir_mode:
                path = QFileDialog.getExistingDirectory(self, label, current)
            else:
                path = ""
            if path:
                edit.setText(path)
                self._save_settings_safe()
                self._refresh_preview()

        browse.clicked.connect(_pick)
        return edit

    # ---------- defaults / settings ----------
    def _apply_base_defaults(self) -> None:
        self.env_python_edit.setText(str(self._default_env_python()))
        self.repo_edit.setText(str(self._default_repo()))
        self.model_root_edit.setText(str(self._default_model_root()))
        self.output_dir_edit.setText(str(self._default_output_dir()))
        self.output_name_edit.setText("wan22_int8_output")
        self.prompt_edit.setPlainText(UiState.prompt)
        self.negative_edit.setPlainText(UiState.negative_prompt)
        self.seed_spin.setValue(-1)
        self.width_spin.setValue(832)
        self.height_spin.setValue(480)
        self.frames_spin.setValue(81)
        self.steps_spin.setValue(4)
        self.sample_shift_spin.setValue(5.0)
        self.cfg_high_spin.setValue(1.0)
        self.cfg_low_spin.setValue(1.0)
        self.attn_combo.setCurrentText("sage_attn2")
        self.cpu_offload_chk.setChecked(True)
        self.text_offload_chk.setChecked(True)
        self.image_offload_chk.setChecked(False)
        self.vae_offload_chk.setChecked(False)
        self.offload_combo.setCurrentText("block")
        self.save_preview_chk.setChecked(True)

    def _load_settings_protected(self) -> None:
        prev_loading = self._loading_settings
        self._loading_settings = True
        try:
            if not self.settings_path.is_file():
                return
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            self._apply_state(UiState(**data))
            saved_env = self.env_python_edit.text().strip()
            default_env = str(self._default_env_python())
            if (not saved_env) or ('.ace_15' in saved_env.lower()) or (not Path(saved_env).is_file() and Path(default_env).is_file()):
                self.env_python_edit.setText(default_env)
        except Exception as exc:
            self._append_log(f"[wan22-int8] warning: could not load settings: {exc}\n")
        finally:
            self._loading_settings = prev_loading

    def _save_settings_safe(self) -> None:
        if self._loading_settings:
            return
        try:
            state = self._collect_state()
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[wan22-int8] warning: could not save settings: {exc}\n")

    def _wire_change_saves(self) -> None:
        widgets = [
            self.env_python_edit, self.repo_edit, self.model_root_edit,
            self.high_noise_edit, self.low_noise_edit, self.t5_edit,
            self.clip_edit, self.vae_edit, self.xlm_edit, self.google_edit,
            self.input_image_edit, self.output_dir_edit, self.output_name_edit,
        ]
        for w in widgets:
            w.textChanged.connect(self._save_settings_safe)
            w.textChanged.connect(self._refresh_preview)
        for w in [self.prompt_edit, self.negative_edit]:
            w.textChanged.connect(self._save_settings_safe)
            w.textChanged.connect(self._refresh_preview)
        for w in [self.seed_spin, self.width_spin, self.height_spin, self.frames_spin, self.steps_spin]:
            w.valueChanged.connect(self._save_settings_safe)
            w.valueChanged.connect(self._refresh_preview)
        for w in [self.sample_shift_spin, self.cfg_high_spin, self.cfg_low_spin]:
            w.valueChanged.connect(self._save_settings_safe)
            w.valueChanged.connect(self._refresh_preview)
        for w in [self.attn_combo, self.offload_combo]:
            w.currentTextChanged.connect(self._save_settings_safe)
            w.currentTextChanged.connect(self._refresh_preview)
        for w in [self.cpu_offload_chk, self.text_offload_chk, self.image_offload_chk, self.vae_offload_chk, self.save_preview_chk]:
            w.toggled.connect(self._save_settings_safe)
            w.toggled.connect(self._refresh_preview)

    def _collect_state(self) -> UiState:
        return UiState(
            env_python=self.env_python_edit.text().strip(),
            lightx2v_repo=self.repo_edit.text().strip(),
            model_root=self.model_root_edit.text().strip(),
            high_noise_ckpt=self.high_noise_edit.text().strip(),
            low_noise_ckpt=self.low_noise_edit.text().strip(),
            t5_path=self.t5_edit.text().strip(),
            clip_path=self.clip_edit.text().strip(),
            vae_path=self.vae_edit.text().strip(),
            xlm_roberta_dir=self.xlm_edit.text().strip(),
            google_dir=self.google_edit.text().strip(),
            input_image=self.input_image_edit.text().strip(),
            output_dir=self.output_dir_edit.text().strip(),
            output_name=self.output_name_edit.text().strip() or "wan22_int8_output",
            prompt=self.prompt_edit.toPlainText().strip(),
            negative_prompt=self.negative_edit.toPlainText().strip(),
            seed=int(self.seed_spin.value()),
            width=int(self.width_spin.value()),
            height=int(self.height_spin.value()),
            num_frames=int(self.frames_spin.value()),
            infer_steps=int(self.steps_spin.value()),
            sample_shift=float(self.sample_shift_spin.value()),
            cfg_high=float(self.cfg_high_spin.value()),
            cfg_low=float(self.cfg_low_spin.value()),
            attn_mode=self.attn_combo.currentText(),
            cpu_offload=self.cpu_offload_chk.isChecked(),
            text_encoder_offload=self.text_offload_chk.isChecked(),
            image_encoder_offload=self.image_offload_chk.isChecked(),
            vae_offload=self.vae_offload_chk.isChecked(),
            offload_granularity=self.offload_combo.currentText(),
            save_last_preview=self.save_preview_chk.isChecked(),
        )

    def _apply_state(self, state: UiState) -> None:
        self.env_python_edit.setText(state.env_python)
        self.repo_edit.setText(state.lightx2v_repo)
        self.model_root_edit.setText(state.model_root)
        self.high_noise_edit.setText(state.high_noise_ckpt)
        self.low_noise_edit.setText(state.low_noise_ckpt)
        self.t5_edit.setText(state.t5_path)
        self.clip_edit.setText(state.clip_path)
        self.vae_edit.setText(state.vae_path)
        self.xlm_edit.setText(state.xlm_roberta_dir)
        self.google_edit.setText(state.google_dir)
        self.input_image_edit.setText(state.input_image)
        self.output_dir_edit.setText(state.output_dir)
        self.output_name_edit.setText(state.output_name)
        self.prompt_edit.setPlainText(state.prompt)
        self.negative_edit.setPlainText(state.negative_prompt)
        self.seed_spin.setValue(state.seed)
        self.width_spin.setValue(state.width)
        self.height_spin.setValue(state.height)
        self.frames_spin.setValue(state.num_frames)
        self.steps_spin.setValue(state.infer_steps)
        self.sample_shift_spin.setValue(state.sample_shift)
        self.cfg_high_spin.setValue(state.cfg_high)
        self.cfg_low_spin.setValue(state.cfg_low)
        self.attn_combo.setCurrentText(state.attn_mode)
        self.cpu_offload_chk.setChecked(state.cpu_offload)
        self.text_offload_chk.setChecked(state.text_encoder_offload)
        self.image_offload_chk.setChecked(state.image_encoder_offload)
        self.vae_offload_chk.setChecked(state.vae_offload)
        self.offload_combo.setCurrentText(state.offload_granularity)
        self.save_preview_chk.setChecked(state.save_last_preview)
        if state.input_image and Path(state.input_image).is_file():
            self.preview_image.load_image(state.input_image)

    # ---------- autodetect ----------
    def _auto_detect_missing_only(self) -> None:
        self._auto_detect(fill_only_missing=True)

    def _auto_detect_all(self) -> None:
        self._auto_detect(fill_only_missing=False)

    def _auto_detect(self, *, fill_only_missing: bool) -> None:
        prev_loading = self._loading_settings
        if fill_only_missing:
            self._loading_settings = True

        def set_if(edit: QLineEdit, value: Path | str) -> None:
            value = str(value)
            if fill_only_missing and edit.text().strip():
                return
            if value:
                edit.setText(value)

        def first_existing(*paths: Path) -> Path:
            for path in paths:
                if path.exists():
                    return path
            return paths[0]

        try:
            model_root = Path(self.model_root_edit.text().strip() or self._default_model_root())
            repo_root = Path(self.repo_edit.text().strip() or self._default_repo())

            set_if(self.env_python_edit, self._default_env_python())
            set_if(self.repo_edit, self._default_repo())
            set_if(self.model_root_edit, self._default_model_root())
            set_if(self.output_dir_edit, self._default_output_dir())

            high_noise = first_existing(
                model_root / "high_noise_model" / HIGH_NOISE_FILE,
                model_root / "high_noise_model" / "diffusion_pytorch_model.safetensors",
                model_root / HIGH_NOISE_FILE,
            )
            low_noise = first_existing(
                model_root / "low_noise_model" / LOW_NOISE_FILE,
                model_root / "low_noise_model" / "diffusion_pytorch_model.safetensors",
                model_root / LOW_NOISE_FILE,
            )
            t5_path = first_existing(
                model_root / "google" / "umt5-xxl",
                model_root / "text_encoder",
                model_root / "umt5_xxl",
            )
            vae_path = first_existing(
                model_root / "vae",
                model_root,
            )

            candidates = {
                self.high_noise_edit: high_noise,
                self.low_noise_edit: low_noise,
                self.t5_edit: t5_path,
                self.vae_edit: vae_path,
            }
            for edit, path in candidates.items():
                if path.exists() or not fill_only_missing:
                    set_if(edit, path)

            if not fill_only_missing:
                self.clip_edit.clear()
                self.xlm_edit.clear()
                self.google_edit.clear()
        finally:
            self._loading_settings = prev_loading

        self._refresh_preview()
        self._save_settings_safe()

    # ---------- validation ----------
    def _validate_paths(self) -> tuple[list[str], list[str]]:
        state = self._collect_state()
        required = [
            ("Env python", state.env_python, True),
            ("LightX2V repo", state.lightx2v_repo, False),
            ("Model root", state.model_root, False),
            ("High-noise INT8 model file", state.high_noise_ckpt, True),
            ("Low-noise INT8 model file", state.low_noise_ckpt, True),
            ("Text encoder folder", state.t5_path, False),
            ("VAE folder", state.vae_path, False),
            ("Input image", state.input_image, True),
            ("Output dir", state.output_dir, False),
        ]
        errors: list[str] = []
        warnings: list[str] = []
        for label, path_str, is_file in required:
            if not path_str:
                errors.append(f"Missing: {label}")
                continue
            p = Path(path_str)
            ok = p.is_file() if is_file else p.is_dir()
            if not ok:
                errors.append(f"Not found: {label} -> {path_str}")

        if state.infer_steps != 4:
            warnings.append("The distilled model is meant for 4-step inference. You changed it away from 4.")
        if state.high_noise_ckpt and not state.high_noise_ckpt.endswith("_int8_lightx2v_4step.safetensors"):
            warnings.append("High-noise model filename does not look like the LightX2V INT8 4-step file.")
        if state.low_noise_ckpt and not state.low_noise_ckpt.endswith("_int8_lightx2v_4step.safetensors"):
            warnings.append("Low-noise model filename does not look like the LightX2V INT8 4-step file.")
        if state.t5_path:
            t5_dir = Path(state.t5_path)
            has_t5_markers = (
                t5_dir.joinpath("config.json").exists()
                or t5_dir.joinpath("tokenizer_config.json").exists()
                or t5_dir.joinpath("spiece.model").exists()
                or any(t5_dir.glob("*.model"))
                or any(t5_dir.glob("*.json"))
            )
            if not has_t5_markers:
                warnings.append("Text encoder folder does not look like a valid T5/tokenizer folder.")
        if state.t5_path and Path(state.t5_path).name == "text_encoder":
            warnings.append("Text encoder folder points to model_root/text_encoder. For this Wan 2.2 pack, model_root/google/umt5-xxl is the safer T5 target.")
        if state.vae_path and not Path(state.vae_path).joinpath("config.json").exists():
            warnings.append("VAE folder does not contain config.json.")
        google_under_model_root = Path(state.model_root).joinpath("google", "umt5-xxl") if state.model_root else None
        if google_under_model_root is not None and not google_under_model_root.is_dir():
            warnings.append("model_root/google/umt5-xxl was not found. Wan 2.2 text encoding usually expects that folder under model_root.")
        if state.clip_path:
            warnings.append("Legacy CLIP path is ignored for this packaged Wan INT8 model.")
        if state.xlm_roberta_dir:
            warnings.append("Legacy tokenizer/cache path is ignored for this packaged Wan INT8 model.")
        if state.google_dir:
            warnings.append("Legacy google path is ignored. Wan 2.2 reads google/umt5-xxl from inside model_root.")
        if state.width % 32 != 0 or state.height % 32 != 0:
            warnings.append("Width/height are usually safer as multiples of 32.")
        return errors, warnings

    def _validate_and_show(self) -> None:
        errors, warnings = self._validate_paths()
        msg = []
        if errors:
            msg.append("Errors:\n- " + "\n- ".join(errors))
        if warnings:
            msg.append("Warnings:\n- " + "\n- ".join(warnings))
        if not msg:
            msg = ["Setup looks valid."]
        QMessageBox.information(self, APP_TITLE, "\n\n".join(msg))

    # ---------- run ----------
    def _build_output_path(self) -> Path:
        out_dir = Path(self.output_dir_edit.text().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (self.output_name_edit.text().strip() or "wan22_int8_output").strip()
        return out_dir / f"{name}.mp4"

    def _build_preview_jpg_path(self) -> Path:
        out_dir = Path(self.output_dir_edit.text().strip())
        name = (self.output_name_edit.text().strip() or "wan22_int8_output").strip()
        return out_dir / f"{name}_lastframe.jpg"

    def _build_runner_code(self, state: UiState, output_mp4: str, preview_jpg: str) -> str:
        payload = {
            "repo": state.lightx2v_repo,
            "model_root": state.model_root,
            "high": state.high_noise_ckpt,
            "low": state.low_noise_ckpt,
            "t5": state.t5_path,
            "vae": state.vae_path,
            "image": state.input_image,
            "prompt": state.prompt,
            "negative": state.negative_prompt,
            "seed": state.seed,
            "width": state.width,
            "height": state.height,
            "frames": state.num_frames,
            "steps": state.infer_steps,
            "sample_shift": state.sample_shift,
            "cfg": [state.cfg_high, state.cfg_low],
            "attn_mode": state.attn_mode,
            "cpu_offload": state.cpu_offload,
            "text_encoder_offload": state.text_encoder_offload,
            "image_encoder_offload": state.image_encoder_offload,
            "vae_offload": state.vae_offload,
            "offload_granularity": state.offload_granularity,
            "output_mp4": output_mp4,
            "preview_jpg": preview_jpg,
            "save_preview": state.save_last_preview,
        }
        dumped = json.dumps(payload, ensure_ascii=False)
        return textwrap.dedent(
            f"""
            import json, os, sys
            from pathlib import Path
            from types import SimpleNamespace

            payload = json.loads({dumped!r})
            repo = Path(payload["repo"]).resolve()
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

            # Prevent Python from executing lightx2v/__init__.py, which force-imports
            # the giant pipeline and unrelated runners. Expose only package shells with paths.
            import types
            import importlib.machinery

            def _stub_pkg(name, path):
                if name in sys.modules:
                    return
                mod = types.ModuleType(name)
                setattr(mod, "__file__", str(Path(path) / "__init__.py"))
                setattr(mod, "__package__", name)
                setattr(mod, "__path__", [str(path)])
                setattr(mod, "__spec__", importlib.machinery.ModuleSpec(name, loader=None, is_package=True))
                sys.modules[name] = mod

            lx_root = repo / "lightx2v"
            _stub_pkg("lightx2v", lx_root)
            _stub_pkg("lightx2v.utils", lx_root / "utils")
            _stub_pkg("lightx2v.common", lx_root / "common")
            _stub_pkg("lightx2v.models", lx_root / "models")
            _stub_pkg("lightx2v.models.runners", lx_root / "models" / "runners")
            _stub_pkg("lightx2v.models.runners.wan", lx_root / "models" / "runners" / "wan")
            _stub_pkg("lightx2v.models.networks", lx_root / "models" / "networks")
            _stub_pkg("lightx2v.models.networks.wan", lx_root / "models" / "networks" / "wan")
            _stub_pkg("lightx2v.models.schedulers", lx_root / "models" / "schedulers")
            _stub_pkg("lightx2v.models.schedulers.wan", lx_root / "models" / "schedulers" / "wan")
            _stub_pkg("lightx2v.models.schedulers.wan.step_distill", lx_root / "models" / "schedulers" / "wan" / "step_distill")

            # Keep the downloaded repo read-only. Some shared LightX2V base modules import
            # GGUF helpers unconditionally even for Wan INT8 safetensors paths. Provide a small
            # stub so those imports succeed without pulling optional GGUF support into this run.
            if "gguf" not in sys.modules:
                gguf_stub = types.ModuleType("gguf")
                setattr(gguf_stub, "__file__", "<framevision-gguf-stub>")
                setattr(gguf_stub, "__package__", "gguf")
                setattr(gguf_stub, "__spec__", importlib.machinery.ModuleSpec("gguf", loader=None, is_package=False))

                class _GGMLQuantizationType:
                    F32 = "F32"
                    F16 = "F16"
                    BF16 = "BF16"
                    Q4_0 = "Q4_0"
                    Q4_1 = "Q4_1"
                    Q5_0 = "Q5_0"
                    Q5_1 = "Q5_1"
                    Q8_0 = "Q8_0"
                    Q8_1 = "Q8_1"
                    Q2_K = "Q2_K"
                    Q3_K = "Q3_K"
                    Q4_K = "Q4_K"
                    Q5_K = "Q5_K"
                    Q6_K = "Q6_K"
                    Q8_K = "Q8_K"

                class _GGUFValueType:
                    ARRAY = "ARRAY"
                    INT32 = "INT32"
                    STRING = "STRING"

                class _GGUFReader:
                    def __init__(self, *args, **kwargs):
                        self.tensors = []
                        self.fields = {{}}

                class _GGUFQuants:
                    @staticmethod
                    def dequantize(*args, **kwargs):
                        raise RuntimeError("GGUF dequantize should not be called for Wan INT8 safetensors runs")

                def _quant_shape_from_byte_shape(shape, qtype):
                    return shape

                setattr(gguf_stub, "GGMLQuantizationType", _GGMLQuantizationType)
                setattr(gguf_stub, "GGUFValueType", _GGUFValueType)
                setattr(gguf_stub, "GGUFReader", _GGUFReader)
                setattr(gguf_stub, "GGML_QUANT_SIZES", {{}})
                setattr(gguf_stub, "quant_shape_from_byte_shape", _quant_shape_from_byte_shape)
                setattr(gguf_stub, "quants", _GGUFQuants())
                sys.modules["gguf"] = gguf_stub

            import lightx2v_platform.set_ai_device  # noqa: F401
            from lightx2v.utils.set_config import set_config
            from lightx2v.utils.utils import validate_config_paths
            from lightx2v.utils.input_info import I2VInputInfo
            from lightx2v.models.runners.wan.wan_distill_runner import Wan22MoeDistillRunner
            from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
            import lightx2v.models.runners.wan.wan_runner as wan_runner_mod
            import torch

            def _patch_hf_text_encoder_loader():
                print("[runner] HF text encoder fallback disabled for Wan INT8 pack")
                return

            _patch_hf_text_encoder_loader()

            try:
                import lightx2v.common.ops.attn.torch_sdpa  # noqa: F401
            except Exception as exc:
                print(f"[runner] torch_sdpa import warning: {{exc}}")
            try:
                import lightx2v.common.ops.attn.flash_attn  # noqa: F401
            except Exception as exc:
                print(f"[runner] flash_attn import warning: {{exc}}")
            try:
                import lightx2v.common.ops.attn.sage_attn  # noqa: F401
            except Exception as exc:
                print(f"[runner] sage_attn import warning: {{exc}}")

            model_root = Path(payload["model_root"]).resolve()
            image_path = Path(payload["image"]).resolve()
            output_mp4 = Path(payload["output_mp4"]).resolve()
            preview_jpg = Path(payload["preview_jpg"]).resolve()
            output_mp4.parent.mkdir(parents=True, exist_ok=True)

            print("[runner] repo=", repo)
            print("[runner] model_root=", model_root)
            print("[runner] output=", output_mp4)
            print("[runner] using direct Wan22MoeDistillRunner import with lightx2v package stub")

            cfg_value = payload["cfg"]
            enable_cfg = not (isinstance(cfg_value, list) and len(cfg_value) == 2 and cfg_value[0] == 1 and cfg_value[1] == 1)

            requested_attn = str(payload.get("attn_mode") or "sage_attn2").strip()
            attn_alias_map = dict(
                sdpa="torch_sdpa",
                torch="torch_sdpa",
                torch_sdpa="torch_sdpa",
                flash="flash_attn2",
                flash_attn2="flash_attn2",
                sage_attn2="sage_attn2",
            )
            resolved_attn = attn_alias_map.get(requested_attn, requested_attn)
            available_attn = set(getattr(ATTN_WEIGHT_REGISTER, "_dict", {{}}).keys())
            if resolved_attn not in available_attn:
                fallback_attn = "torch_sdpa" if "torch_sdpa" in available_attn else next(iter(available_attn), resolved_attn)
                print(f"[runner] requested attn '{{requested_attn}}' resolved to '{{resolved_attn}}' but not registered; falling back to '{{fallback_attn}}'")
                resolved_attn = fallback_attn
            else:
                print(f"[runner] requested attn '{{requested_attn}}' -> '{{resolved_attn}}'")

            # For this packaged Wan 2.2 INT8 layout, tokenizer assets are resolved from
            # model_root/google/umt5-xxl by the LightX2V Wan runner.
            t5_override = payload.get("t5")
            if t5_override:
                t5_override = str(Path(t5_override).resolve())
                if Path(t5_override).name == "text_encoder":
                    google_t5 = model_root / "google" / "umt5-xxl"
                    if google_t5.is_dir():
                        print(f"[runner] remapping T5 from '{{t5_override}}' to '{{google_t5}}'")
                        t5_override = str(google_t5.resolve())
            args = SimpleNamespace(
                task="i2v",
                model_path=str(model_root),
                model_cls="wan2.2_moe_distill",
                sf_model_path=None,
                config_json=None,
                dit_original_ckpt=None,
                low_noise_original_ckpt=None,
                high_noise_original_ckpt=None,
                transformer_model_name=None,

                infer_steps=payload["steps"],
                target_video_length=payload["frames"],
                target_height=payload["height"],
                target_width=payload["width"],
                sample_guide_scale=cfg_value,
                sample_shift=payload["sample_shift"],
                fps=16,
                aspect_ratio="custom",
                boundary=0.9,
                boundary_step_index=2,
                denoising_step_list=[1000, 750, 500, 250],
                rope_type="torch",
                double_precision_rope=True,
                norm_modulate_backend="torch",
                self_attn_1_type=resolved_attn,
                cross_attn_1_type=resolved_attn,
                cross_attn_2_type=resolved_attn,

                cpu_offload=payload["cpu_offload"],
                offload_granularity=payload["offload_granularity"],
                t5_cpu_offload=payload["text_encoder_offload"],
                clip_cpu_offload=payload["image_encoder_offload"],
                vae_cpu_offload=payload["vae_offload"],
                use_prompt_enhancer=False,
                parallel=False,
                seq_parallel=False,
                cfg_parallel=False,
                enable_cfg=enable_cfg,
                use_image_encoder=False,

                dit_quantized=True,
                dit_quant_scheme="int8-vllm",
                dit_quantized_ckpt=None,
                high_noise_quantized_ckpt=str(Path(payload["high"]).resolve()),
                low_noise_quantized_ckpt=str(Path(payload["low"]).resolve()),

                t5_original_ckpt=t5_override if t5_override else None,
                vae_path=str(Path(payload["vae"]).resolve()) if payload.get("vae") else None,
            )

            config = set_config(args)
            validate_config_paths(config)

            print("[runner] config model_cls=", config["model_cls"])
            print("[runner] config task=", config["task"])
            print("[runner] config use_image_encoder=", config.get("use_image_encoder"))
            print("[runner] config t5_original_ckpt=", config.get("t5_original_ckpt"))
            print("[runner] config vae_path=", config.get("vae_path"))

            runner = Wan22MoeDistillRunner(config)
            runner.init_modules()

            input_info = I2VInputInfo(
                seed=payload["seed"],
                prompt=payload["prompt"],
                negative_prompt=payload["negative"],
                image_path=str(image_path),
                save_result_path=str(output_mp4),
                return_result_tensor=False,
            )

            result = runner.run_pipeline(input_info)
            print("[runner] generation return type:", type(result).__name__)

            if payload.get("save_preview") and output_mp4.is_file():
                try:
                    import imageio.v3 as iio
                    frame = iio.imread(str(output_mp4), index=-1)
                    iio.imwrite(str(preview_jpg), frame)
                    print("[runner] wrote preview:", preview_jpg)
                except Exception as exc:
                    print(f"[runner] preview warning: {{exc}}")

            print("[runner] done")
            """
        ).strip()

    def _build_command(self) -> tuple[list[str], str, dict[str, str], str]:

        state = self._collect_state()
        output_mp4 = str(self._build_output_path())
        preview_jpg = str(self._build_preview_jpg_path())
        runner_code = self._build_runner_code(state, output_mp4, preview_jpg)
        env_python = state.env_python
        default_env_python = str(self._default_env_python())
        if (not env_python) or ('.ace_15' in env_python.lower()) or (not Path(env_python).is_file() and Path(default_env_python).is_file()):
            env_python = default_env_python
        command = [env_python, "-u", "-c", runner_code]
        env = os.environ.copy()
        repo = state.lightx2v_repo.strip()
        if repo:
            env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("SKIP_PLATFORM_CHECK", "1")
        workdir = repo or str(self.root_dir)
        return command, workdir, env, output_mp4

    def _refresh_preview(self) -> None:
        try:
            state = self._collect_state()
            errors, warnings = self._validate_paths()
            preview = [
                "Validation:",
                *([f"ERROR: {e}" for e in errors] if errors else ["No blocking errors found."]),
                *([f"WARN: {w}" for w in warnings] if warnings else []),
                "",
                "Command preview:",
            ]
            command, _, _, _ = self._build_command()
            preview.append(format_command(command))
            self.command_preview.setPlainText("\n".join(preview))
        except Exception as exc:
            self.command_preview.setPlainText(f"Could not build command preview:\n{exc}")

        img = self.input_image_edit.text().strip()
        if img and Path(img).is_file():
            self.preview_image.load_image(img)
        else:
            self.preview_image.clear_image()

    def _ensure_runtime_dependencies(self, env_python: str) -> None:
        py = Path(env_python)
        if not py.is_file():
            raise FileNotFoundError(f"Python executable not found: {env_python}")

        checks = [
            ("imageio", "imageio"),
            ("imageio_ffmpeg", "imageio-ffmpeg"),
            ("ftfy", "ftfy"),
            ("prometheus_client", "prometheus-client"),
            ("pydantic", "pydantic"),
        ]
        missing: list[str] = []

        for import_name, pip_name in checks:
            probe = subprocess.run(
                [str(py), "-c", f"import {import_name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if probe.returncode != 0:
                missing.append(pip_name)
                stderr = (probe.stderr or "").strip()
                if stderr:
                    self._append_log(f"[wan22-int8] missing dependency probe for {import_name}: {stderr}\n")

        if not missing:
            self._append_log("[wan22-int8] runtime dependency check passed\n")
            return

        unique_missing = []
        for name in missing:
            if name not in unique_missing:
                unique_missing.append(name)

        self._append_log(
            "[wan22-int8] repairing missing runtime dependencies in selected env: "
            + ", ".join(unique_missing)
            + "\n"
        )
        install = subprocess.run(
            [str(py), "-m", "pip", "install", *unique_missing],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if install.stdout:
            self._append_log(install.stdout.rstrip() + "\n")
        if install.returncode != 0:
            raise RuntimeError(
                "Could not install required runtime dependencies into the selected env: "
                + ", ".join(unique_missing)
            )

        self._append_log("[wan22-int8] runtime dependency repair finished\n")

    def _start_run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, APP_TITLE, "A run is already active.")
            return

        try:
            errors, warnings = self._validate_paths()
            if errors:
                msg = "\n".join(errors)
                self._append_log("[wan22-int8] validation failed before run:\n")
                for e in errors:
                    self._append_log(f"  - {e}\n")
                self._append_log("\n")
                QMessageBox.warning(self, APP_TITLE, msg)
                return

            self.logs_edit.clear()
            self._append_log("[wan22-int8] Generate pressed\n")
            env_python = self.env_python_edit.text().strip()
            self._ensure_runtime_dependencies(env_python)
            if warnings:
                self._append_log("[wan22-int8] warnings before run:\n")
                for w in warnings:
                    self._append_log(f"  - {w}\n")
                self._append_log("\n")

            command, workdir, env, output_mp4 = self._build_command()
            self._append_log(f"[wan22-int8] workdir: {workdir}\n")
            self._append_log(f"[wan22-int8] python: {command[0]}\n")
            self._append_log(f"[wan22-int8] target output: {output_mp4}\n")
            self._append_log(f"[wan22-int8] launching...\n\n")

            if not Path(command[0]).is_file():
                raise FileNotFoundError(f"Python executable not found: {command[0]}")
            if not Path(workdir).exists():
                raise FileNotFoundError(f"Working directory not found: {workdir}")

            self._worker = ProcessWorker(command, workdir, env)
            self._thread = QThread(self)
            self._worker.moveToThread(self._thread)

            self._thread.started.connect(lambda: self._append_log("[wan22-int8] worker thread started\n"))
            self._thread.started.connect(self._worker.run)
            self._worker.log.connect(self._append_log)
            self._worker.started.connect(lambda: self._append_log("[wan22-int8] subprocess starting\n"))
            self._worker.finished.connect(self._on_finished)
            self._worker.finished.connect(self._thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.finished.connect(self._clear_thread)

            self._set_running(True)
            self._thread.start()

        except Exception as exc:
            self._set_running(False)
            self._append_log(f"[wan22-int8] startup error: {exc}\n")
            QMessageBox.warning(self, APP_TITLE, f"Could not start generation:\n{exc}")

    def _stop_run(self) -> None:
        if self._worker:
            self._append_log("\n[wan22-int8] stop requested...\n")
            self._worker.stop()

    def _on_finished(self, rc: int, status: str) -> None:
        self._append_log(f"\n[wan22-int8] process {status} with code {rc}\n")
        try:
            preview_jpg = self._build_preview_jpg_path()
            if preview_jpg.is_file():
                self.preview_image.load_image(str(preview_jpg))
            else:
                img = self.input_image_edit.text().strip()
                if img and Path(img).is_file():
                    self.preview_image.load_image(img)
                else:
                    self.preview_image.clear_image()
        finally:
            self._set_running(False)

    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.preview_cmd_btn.setEnabled(not running)

    # ---------- helpers ----------
    def _append_log(self, text: str) -> None:
        self.logs_edit.moveCursor(QTextCursor.End)
        self.logs_edit.insertPlainText(text)
        self.logs_edit.moveCursor(QTextCursor.End)

    def _open_output_dir(self) -> None:
        out_dir = Path(self.output_dir_edit.text().strip() or self._default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(out_dir)])
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1100, 980)
        self.setCentralWidget(Wan22Int8Pane(self))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec())
