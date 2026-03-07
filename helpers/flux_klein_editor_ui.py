#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrameVision – FLUX.2 Klein 4B (GGUF) Editor UI
Backend: stable-diffusion.cpp `sd-cli.exe` (no torch/diffusers)

Uses the official FLUX.2 Klein CLI wiring (Flux2 docs):
- --diffusion-model <flux2-klein-4b*.gguf>
- --vae <flux2_ae.safetensors>
- --llm <qwen3-4b*.gguf>
- -p/--prompt, -n/--negative-prompt
- --steps, --cfg-scale, --sampling-method
- -r/--ref-image can be used multiple times for Flux Kontext-style reference editing
- -o/--output for output image
Docs: https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/flux2.md

UI features:
- Prompt + Negative prompt
- Size, steps, CFG, seed/random
- Reference images list (add/remove/reorder) -> emits repeated -r arguments
- Model selectors: Flux GGUF, Qwen3 4B GGUF, VAE safetensors
- Performance toggles: --diffusion-fa, --offload-to-cpu, --vae-tiling
- Command preview + live log
- Output preview (zoom/pan), open output folder, save copy

Paths (auto-detected; overrideable):
- FrameVision root is detected by finding: models/ and presets/bin/
- sd-cli: <root>/presets/bin/sd-cli.exe
- default model dir: <root>/models/klein4b_gguf/
- outputs: <root>/output/klein4b_gguf/

Requires (UI python):
- PySide6
- Pillow
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Root/path discovery
# -----------------------------

def _find_framevision_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "models").is_dir() and (cur / "presets" / "bin").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def _detect_framevision_root() -> Path:
    """Try multiple anchors to robustly locate the FrameVision root."""
    anchors: List[Path] = []
    try:
        anchors.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    try:
        if sys.argv and sys.argv[0]:
            anchors.append(Path(sys.argv[0]).resolve().parent)
    except Exception:
        pass
    try:
        anchors.append(Path.cwd())
    except Exception:
        pass

    for a in anchors:
        root = _find_framevision_root(a)
        if (root / "models").is_dir() and (root / "presets" / "bin").is_dir():
            return root
    return anchors[0] if anchors else Path.cwd()


def _first_existing(candidates: List[Path]) -> Path:
    """Return the first *file* that exists.

    Important: On Windows, a path like '.' exists but is a directory.
    Trying to execute it causes: WinError 5 (Access is denied).
    """
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    # Fallback: keep behavior stable (but never return a directory)
    for p in candidates:
        try:
            if p.exists() and not p.is_dir():
                return p
        except Exception:
            continue
    return candidates[0]


def _scan(folder: Path, exts: tuple[str, ...]) -> List[Path]:
    if not folder.exists():
        return []
    items = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            items.append(p)
    return sorted(items)


# -----------------------------
# Data
# -----------------------------

@dataclass
class Paths:
    root: str
    sd_cli: str
    model_dir: str
    out_dir: str

    @staticmethod
    def from_root(root: Path) -> "Paths":
        sd = root / "presets" / "bin" / ("sd-cli.exe" if os.name == "nt" else "sd-cli")
        model_dir = root / "models" / "klein4b_gguf"
        out_dir = root / "output" / "edits" / "flux_klein"
        out_dir.mkdir(parents=True, exist_ok=True)
        return Paths(root=str(root), sd_cli=str(sd), model_dir=str(model_dir), out_dir=str(out_dir))


@dataclass
class GGUFModels:
    diffusion_model: str = ""   # flux gguf
    llm_model: str = ""         # qwen3 4b gguf
    vae_file: str = ""          # flux2_ae.safetensors


@dataclass
class RunConfig:
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 4
    cfg_scale: float = 1.0
    seed: int = 0
    random_seed: bool = True
    sampling_method: str = "euler"
    diffusion_fa: bool = True
    offload_to_cpu: bool = False
    vae_tiling: bool = False
    out_name: str = ""
    # Flux Kontext-style ref images (can be used multiple times)
    ref_images: List[str] = None

    def __post_init__(self):
        if self.ref_images is None:
            self.ref_images = []


# -----------------------------
# Image viewer widget
# -----------------------------

class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pix_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pix_item)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
            | QtGui.QPainter.RenderHint.Antialiasing
        )

    def set_image(self, img: Optional[Image.Image]) -> None:
        if img is None:
            self._pix_item.setPixmap(QtGui.QPixmap())
            self.scene().setSceneRect(QtCore.QRectF())
            return
        qimg = ImageQt.ImageQt(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        self._pix_item.setPixmap(pix)
        self.scene().setSceneRect(QtCore.QRectF(pix.rect()))
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pix_item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()


# -----------------------------
# Runner thread
# -----------------------------

class CliRunner(QtCore.QObject):
    logLine = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # ok, output_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        try:
            if self._proc and self.is_running():
                self._proc.terminate()
        except Exception:
            pass

    def run(self, cmd: List[str], cwd: Path, expected_out: Path):
        if self.is_running():
            self.logLine.emit("[ui] A job is already running.")
            return

        def _worker():
            try:
                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

                self.logLine.emit("[cmd] " + _pretty_cmd(cmd))
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=creationflags,
                )
                assert self._proc.stdout
                for line in self._proc.stdout:
                    self.logLine.emit(line.rstrip("\n"))
                rc = self._proc.wait()
                out_path = expected_out  # avoid Python's local-scope reassignment trap
                ok = (rc == 0) and out_path.exists()
                if not out_path.exists():
                    # Some builds may still write to default ./output.png if -o failed.
                    fallback = cwd / "output.png"
                    if fallback.exists():
                        out_path = fallback
                        ok = (rc == 0)
                self.finished.emit(ok, str(out_path) if out_path.exists() else "")
            except Exception as e:
                self.logLine.emit(f"[ui] Runner error: {e}")
                self.finished.emit(False, "")
            finally:
                self._proc = None

        threading.Thread(target=_worker, daemon=True).start()


def _pretty_cmd(cmd: List[str]) -> str:
    if os.name == "nt":
        return " ".join(shlex.quote(c) for c in cmd)
    return " ".join(shlex.quote(c) for c in cmd)


# -----------------------------
# Main window
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLUX.2 Klein 4B — GGUF Editor (sd-cli)")
        self.resize(1350, 820)

        root = _detect_framevision_root()
        self.paths = Paths.from_root(root)

        self.models = GGUFModels()
        self.cfg = RunConfig()

        self.current_output: Optional[Path] = None

        self.runner = CliRunner()
        self.runner.logLine.connect(self._append_log)
        self.runner.finished.connect(self._on_finished)

        # During UI construction / state restore we must NOT let change-signals
        # immediately overwrite the settings file with widget defaults.
        # (Qt will emit a lot of change signals while we populate controls.)
        self._suspend_autosave = 1

        self._build_ui()
        self._load_settings()
        # ensure sd-cli + dropdowns are populated
        self._auto_fill_models()
        self._refresh_ui()

        # Now that the UI reflects the loaded (or first-run default) state,
        # enable autosave and persist once.
        self._suspend_autosave = 0
        self._update_cmd_preview()
        self._save_settings()

    # -------- settings --------

    def _settings_path(self) -> Path:
        cfg_dir = Path(self.paths.root) / "presets" / "setsave"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "klein_settings.json"

    def _legacy_settings_path(self) -> Path:
        cfg_dir = Path(self.paths.root) / "presets" / "setsave"
        return cfg_dir / "klain_settings.json"

    def _load_settings(self):
        p = self._settings_path()
        if not p.exists():
            legacy = self._legacy_settings_path()
            if legacy.exists():
                p = legacy
            else:
                return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.paths = Paths(**data.get("paths", asdict(self.paths)))
            self.models = GGUFModels(**data.get("models", asdict(self.models)))
            self.cfg = RunConfig(**data.get("run", asdict(self.cfg)))

            if p.name.lower() == "klain_settings.json":
                try:
                    self._save_settings()
                except Exception:
                    pass
        except Exception as e:
            self._append_log(f"[settings] failed to load: {e}")

    def _save_settings(self):
        p = self._settings_path()
        data = {
            "paths": asdict(self.paths),
            "models": asdict(self.models),
            "run": asdict(self.cfg),
        }
        try:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self._append_log(f"[settings] failed to save: {e}")

    # -------- UI --------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        # Use a vertical shell so we can keep a sticky bottom bar
        # (Generate/Stop) that does not scroll with the left panel.
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Left panel inside scroll area
        left_inner = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_inner)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_inner)

        # Prompt
        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Prompt / edit instruction.")
        self.prompt_edit.setMinimumHeight(120)

        self.neg_edit = QtWidgets.QPlainTextEdit()
        self.neg_edit.setPlaceholderText("Negative prompt (optional).")
        self.neg_edit.setMinimumHeight(70)

        # Reference images
        ref_group = QtWidgets.QGroupBox("Reference images (Flux Kontext style: repeated -r / --ref-image)")
        ref_layout = QtWidgets.QVBoxLayout(ref_group)
        self.ref_list = QtWidgets.QListWidget()
        self.ref_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        ref_layout.addWidget(self.ref_list)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_ref = QtWidgets.QPushButton("Add…")
        self.btn_remove_ref = QtWidgets.QPushButton("Remove")
        self.btn_up = QtWidgets.QPushButton("Up")
        self.btn_down = QtWidgets.QPushButton("Down")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        for b in (self.btn_add_ref, self.btn_remove_ref, self.btn_up, self.btn_down, self.btn_clear):
            btn_row.addWidget(b)
        ref_layout.addLayout(btn_row)

        # Generation settings
        gen_group = QtWidgets.QGroupBox("Generation settings")
        gen_form = QtWidgets.QFormLayout(gen_group)
        gen_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.width_spin = QtWidgets.QSpinBox(); self.width_spin.setRange(64, 4096); self.width_spin.setSingleStep(8)
        self.height_spin = QtWidgets.QSpinBox(); self.height_spin.setRange(64, 4096); self.height_spin.setSingleStep(8)
        self.btn_match_first = QtWidgets.QPushButton("Match 1st ref")
        self.btn_1024 = QtWidgets.QPushButton("1024×1024")
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QtWidgets.QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_row.addWidget(self.btn_match_first)
        size_row.addWidget(self.btn_1024)
        size_row.addStretch(1)

        self.steps_spin = QtWidgets.QSpinBox(); self.steps_spin.setRange(1, 150)
        self.cfg_spin = QtWidgets.QDoubleSpinBox(); self.cfg_spin.setRange(0.0, 20.0); self.cfg_spin.setDecimals(2); self.cfg_spin.setSingleStep(0.25)
        self.seed_spin = QtWidgets.QSpinBox(); self.seed_spin.setRange(-1, 2_147_483_647)
        self.chk_rand_seed = QtWidgets.QCheckBox("Random seed")
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed_spin)
        seed_row.addWidget(self.chk_rand_seed)
        seed_row.addStretch(1)

        self.sampling_combo = QtWidgets.QComboBox()
        self.sampling_combo.addItems(["euler","euler_a","dpm++2s_a"])

        self.chk_diffusion_fa = QtWidgets.QCheckBox("Use --diffusion-fa (faster attention)")
        self.chk_offload_cpu = QtWidgets.QCheckBox("Use --offload-to-cpu (low VRAM)")
        self.chk_vae_tiling = QtWidgets.QCheckBox("Use --vae-tiling (reduce VRAM)")

        self.out_name_edit = QtWidgets.QLineEdit()
        self.out_name_edit.setPlaceholderText("Optional output filename (e.g. my_klein.png). Empty = auto timestamp.")

        gen_form.addRow("Size:", size_row)
        gen_form.addRow("Steps:", self.steps_spin)
        gen_form.addRow("CFG scale:", self.cfg_spin)
        gen_form.addRow("Seed:", seed_row)
        gen_form.addRow("Sampling:", self.sampling_combo)
        gen_form.addRow("", self.chk_diffusion_fa)
        gen_form.addRow("", self.chk_offload_cpu)
        gen_form.addRow("", self.chk_vae_tiling)
        gen_form.addRow("Output name:", self.out_name_edit)

        # Model paths
        model_group = QtWidgets.QGroupBox("Model files (required)")
        model_form = QtWidgets.QFormLayout(model_group)
        model_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.sdcli_edit = QtWidgets.QLineEdit()
        self.sdcli_edit.setReadOnly(True)
        self.btn_sdcli = QtWidgets.QPushButton("Browse…")

        # Dropdown selectors (populated from Model folder)
        self.flux_combo = QtWidgets.QComboBox()
        self.btn_flux = QtWidgets.QPushButton("Browse…")
        self.vae_combo = QtWidgets.QComboBox()
        self.btn_vae = QtWidgets.QPushButton("Browse…")
        self.llm_combo = QtWidgets.QComboBox()
        self.btn_llm = QtWidgets.QPushButton("Browse…")

        # Model folder picker (affects dropdown contents)
        self.modeldir_edit = QtWidgets.QLineEdit()
        self.btn_modeldir = QtWidgets.QPushButton("Browse…")

        def row(edit, btn):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0,0,0,0)
            h.addWidget(edit, 1)
            h.addWidget(btn, 0)
            return w

        model_form.addRow("sd-cli.exe:", row(self.sdcli_edit, self.btn_sdcli))
        model_form.addRow("Model folder:", row(self.modeldir_edit, self.btn_modeldir))
        model_form.addRow("Flux GGUF:", row(self.flux_combo, self.btn_flux))
        model_form.addRow("VAE (flux2_ae.safetensors):", row(self.vae_combo, self.btn_vae))
        model_form.addRow("Text encoder (Qwen3 4B):", row(self.llm_combo, self.btn_llm))

        # Command preview
        self.cmd_preview = QtWidgets.QPlainTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumBlockCount(2000)
        self.cmd_preview.setMinimumHeight(90)

        cmd_group = QtWidgets.QGroupBox("Command preview")
        cmd_layout = QtWidgets.QVBoxLayout(cmd_group)
        cmd_layout.addWidget(self.cmd_preview)

        # Run controls (moved to sticky bottom bar)
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_run.setToolTip("Generate (Ctrl+Enter)")
        self.btn_run.setMinimumHeight(42)
        self.btn_stop = QtWidgets.QPushButton("Stop")

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.log.setMinimumHeight(180)

        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.addWidget(self.log)

        # Output helpers
        out_row = QtWidgets.QHBoxLayout()
        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_save_copy = QtWidgets.QPushButton("Save copy…")
        self.btn_copy_path = QtWidgets.QPushButton("Copy output path")
        out_row.addWidget(self.btn_open_out)
        out_row.addWidget(self.btn_save_copy)
        out_row.addWidget(self.btn_copy_path)
        out_row.addStretch(1)

        # Left assembly
        left_layout.addWidget(QtWidgets.QLabel("Prompt"))
        left_layout.addWidget(self.prompt_edit)
        left_layout.addWidget(QtWidgets.QLabel("Negative prompt"))
        left_layout.addWidget(self.neg_edit)
        left_layout.addWidget(ref_group)
        left_layout.addWidget(gen_group)
        left_layout.addWidget(model_group)
        left_layout.addWidget(cmd_group)
        left_layout.addLayout(out_row)
        left_layout.addWidget(log_group)
        left_layout.addStretch(1)

        # Right preview
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0,0,0,0)
        right_layout.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Idle")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        top.addWidget(self.lbl_status)
        top.addStretch(1)
        top.addWidget(self.btn_fit)

        self.preview = ImageView()

        right_layout.addLayout(top)
        right_layout.addWidget(self.preview, 1)

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 670])

        # Sticky bottom bar (does not scroll)
        bottom = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_layout.addWidget(self.btn_run, 3)
        bottom_layout.addWidget(self.btn_stop, 1)
        outer.addWidget(bottom, 0)

        # Signals
        self.btn_add_ref.clicked.connect(self._add_refs)
        self.btn_remove_ref.clicked.connect(self._remove_selected_refs)
        self.btn_clear.clicked.connect(self._clear_refs)
        self.btn_up.clicked.connect(lambda: self._move_ref(-1))
        self.btn_down.clicked.connect(lambda: self._move_ref(1))

        self.btn_match_first.clicked.connect(self._match_first_ref)
        self.btn_1024.clicked.connect(lambda: self._set_size(1024,1024))

        self.btn_sdcli.clicked.connect(self._browse_sdcli)
        self.btn_modeldir.clicked.connect(self._browse_model_dir)
        self.btn_flux.clicked.connect(lambda: self._browse_and_set_combo(self.flux_combo, "Flux GGUF", "GGUF (*.gguf);;All files (*.*)"))
        self.btn_llm.clicked.connect(lambda: self._browse_and_set_combo(self.llm_combo, "Qwen3 GGUF", "GGUF (*.gguf);;All files (*.*)"))
        self.btn_vae.clicked.connect(lambda: self._browse_and_set_combo(self.vae_combo, "VAE safetensors", "SafeTensors (*.safetensors);;All files (*.*)"))

        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_open_out.clicked.connect(self._open_output_folder)
        self.btn_save_copy.clicked.connect(self._save_copy)
        self.btn_copy_path.clicked.connect(self._copy_output_path)
        self.btn_fit.clicked.connect(lambda: self.preview.fitInView(self.preview.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        # Update cmd preview on edits
        for w in (self.prompt_edit, self.neg_edit, self.width_spin, self.height_spin, self.steps_spin, self.cfg_spin,
                  self.seed_spin, self.chk_rand_seed, self.sampling_combo,
                  self.chk_diffusion_fa, self.chk_offload_cpu, self.chk_vae_tiling,
                  self.out_name_edit, self.sdcli_edit, self.modeldir_edit):
            if isinstance(w, QtWidgets.QPlainTextEdit):
                w.textChanged.connect(self._update_cmd_preview)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentTextChanged.connect(lambda _=None: self._update_cmd_preview())
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(lambda _=None: self._update_cmd_preview())
            else:
                w.valueChanged.connect(lambda _=None: self._update_cmd_preview()) if hasattr(w, "valueChanged") else w.textChanged.connect(self._update_cmd_preview)

        self.flux_combo.currentIndexChanged.connect(lambda _=None: self._update_cmd_preview())
        self.vae_combo.currentIndexChanged.connect(lambda _=None: self._update_cmd_preview())
        self.llm_combo.currentIndexChanged.connect(lambda _=None: self._update_cmd_preview())

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self._run)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self._run)

    # -------- helpers --------

    def _append_log(self, s: str):
        self.log.appendPlainText(s)

    def _browse_file(self, edit: QtWidgets.QLineEdit, title: str, filt: str):
        cur = edit.text().strip()
        if cur:
            try:
                cp = Path(cur)
                start = str(cp.parent)
            except Exception:
                start = self.paths.root
        else:
            start = self.paths.root
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, filt)
        if p:
            edit.setText(p)

    def _browse_sdcli(self):

        # Prefer presets/bin
        prefer = Path(self.paths.root) / "presets" / "bin"
        start = str(prefer) if prefer.exists() else self.paths.root
        cur = self.sdcli_edit.text().strip()
        if cur:
            try:
                cp = Path(cur)
                if cp.exists():
                    start = str(cp.parent)
            except Exception:
                pass
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "sd-cli.exe", start, "Executable (*.exe);;All files (*.*)")
        if p:
            self.sdcli_edit.setText(p)
            self.paths.sd_cli = p
            self._update_cmd_preview()

    def _browse_and_set_combo(self, combo: QtWidgets.QComboBox, title: str, filt: str):
        cur = str(combo.currentData() or "").strip()
        if cur:
            try:
                cp = Path(cur)
                start = str(cp.parent)
            except Exception:
                start = self.paths.root
        else:
            start = self.paths.root
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, filt)
        if not p:
            return
        self._set_combo_to_path(combo, p)

    def _browse_model_dir(self):
        start = self.modeldir_edit.text().strip() or self.paths.model_dir or self.paths.root
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select model folder", start)
        if not d:
            return
        self.paths.model_dir = d
        self.modeldir_edit.setText(d)
        self._populate_model_dropdowns()
        self._update_cmd_preview()
        self._save_settings()

    def _set_combo_to_path(self, combo: QtWidgets.QComboBox, path: str):
        # Try find existing item by stored data
        for i in range(combo.count()):
            if combo.itemData(i) == path:
                combo.setCurrentIndex(i)
                return
        # Add custom entry
        label = Path(path).name
        combo.addItem(label, path)
        combo.setCurrentIndex(combo.count() - 1)

    def _populate_model_dropdowns(self):
        """Scan model folder recursively and fill dropdowns."""
        base = Path(self.paths.model_dir)
        ggufs = []
        safes = []
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    suf = p.suffix.lower()
                    if suf == ".gguf":
                        ggufs.append(p)
                    elif suf == ".safetensors":
                        safes.append(p)

        # Build lists
        flux = [p for p in ggufs if "flux" in p.name.lower() and "klein" in p.name.lower()]
        llm = [p for p in ggufs if "qwen" in p.name.lower() and "4b" in p.name.lower()]
        vae = [p for p in safes if "vae" in p.name.lower() and ("flux2" in p.name.lower() or "flux" in p.name.lower())]

        def refill(combo: QtWidgets.QComboBox, items: List[Path], preferred: str, keep_path: str):
            combo.blockSignals(True)
            combo.clear()
            # preferred first
            items_sorted = sorted(items, key=lambda x: (0 if preferred in x.name.lower() else 1, x.name.lower()))
            for pth in items_sorted:
                rel = str(pth.relative_to(base)) if base.exists() else pth.name
                combo.addItem(rel, str(pth))
            combo.blockSignals(False)
            # restore selection
            if keep_path:
                self._set_combo_to_path(combo, keep_path)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

        refill(self.flux_combo, flux, "q4_k_m", getattr(self.models, "diffusion_model", ""))
        refill(self.llm_combo, llm, "q4_k_m", getattr(self.models, "llm_model", ""))
        # prefer flux2_ae
        preferred_vae = "flux2_ae"
        refill(self.vae_combo, vae, preferred_vae, getattr(self.models, "vae_file", ""))


    def _set_size(self, w: int, h: int):
        self.width_spin.setValue(w)
        self.height_spin.setValue(h)

    def _sync_ref_list(self):
        self.ref_list.clear()
        for p in self.cfg.ref_images:
            self.ref_list.addItem(p)
        self._update_cmd_preview()
        self._save_settings()

    def _add_refs(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add reference images", self.paths.root,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)"
        )
        if not files:
            return
        self.cfg.ref_images.extend([f for f in files if Path(f).exists()])
        self._sync_ref_list()

    def _remove_selected_refs(self):
        rows = sorted({i.row() for i in self.ref_list.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.cfg.ref_images):
                self.cfg.ref_images.pop(r)
        self._sync_ref_list()

    def _clear_refs(self):
        self.cfg.ref_images = []
        self._sync_ref_list()

    def _move_ref(self, delta: int):
        row = self.ref_list.currentRow()
        if row < 0 or row >= len(self.cfg.ref_images):
            return
        nr = row + delta
        if nr < 0 or nr >= len(self.cfg.ref_images):
            return
        self.cfg.ref_images[row], self.cfg.ref_images[nr] = self.cfg.ref_images[nr], self.cfg.ref_images[row]
        self._sync_ref_list()
        self.ref_list.setCurrentRow(nr)

    def _match_first_ref(self):
        if not self.cfg.ref_images:
            QtWidgets.QMessageBox.information(self, "No reference image", "Add at least one reference image first.")
            return
        try:
            img = Image.open(self.cfg.ref_images[0]).convert("RGB")
            w, h = img.size
            w = max(64, int(round(w / 8) * 8))
            h = max(64, int(round(h / 8) * 8))
            self._set_size(w, h)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Failed", str(e))

    def _open_output_folder(self):
        out_dir = Path(self.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])

    def _save_copy(self):
        if not self.current_output or not self.current_output.exists():
            QtWidgets.QMessageBox.information(self, "No output", "Generate an image first.")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save copy", str(self.current_output),
            "Images (*.png *.jpg *.jpeg *.webp);;All files (*.*)"
        )
        if not p:
            return
        try:
            img = Image.open(self.current_output).convert("RGB")
            ext = Path(p).suffix.lower()
            if ext in (".jpg",".jpeg"):
                img.save(p, quality=95, subsampling=0)
            else:
                img.save(p)
            self._append_log(f"[ui] saved copy -> {p}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _copy_output_path(self):
        if not self.current_output:
            return
        QtWidgets.QApplication.clipboard().setText(str(self.current_output))
        self._append_log("[ui] copied output path")

    # -------- auto detect --------

    def _auto_fill_models(self):
        # sd-cli
        root = Path(self.paths.root)
        sd_candidates = [
            Path(self.paths.sd_cli),
            root / "presets" / "bin" / "sd-cli.exe",
            root / "presets" / "bin" / "sd-cli",
        ]
        self.paths.sd_cli = str(_first_existing(sd_candidates))

        # model dir default
        if not self.paths.model_dir:
            self.paths.model_dir = str(root / "models" / "klein4b_gguf")
        # ensure UI has it
        if hasattr(self, "modeldir_edit"):
            self.modeldir_edit.setText(self.paths.model_dir)

        # populate dropdowns
        self._populate_model_dropdowns()


    def _refresh_ui(self):
        # paths
        self.sdcli_edit.setText(self.paths.sd_cli)
        self.modeldir_edit.setText(self.paths.model_dir)
        self._populate_model_dropdowns()

        # run
        self.prompt_edit.setPlainText(self.cfg.prompt)
        self.neg_edit.setPlainText(self.cfg.negative)
        self.width_spin.setValue(int(self.cfg.width))
        self.height_spin.setValue(int(self.cfg.height))
        self.steps_spin.setValue(int(self.cfg.steps))
        self.cfg_spin.setValue(float(self.cfg.cfg_scale))
        self.seed_spin.setValue(int(self.cfg.seed))
        self.chk_rand_seed.setChecked(bool(self.cfg.random_seed))
        self.sampling_combo.setCurrentText(self.cfg.sampling_method or "euler")
        self.chk_diffusion_fa.setChecked(bool(self.cfg.diffusion_fa))
        self.chk_offload_cpu.setChecked(bool(self.cfg.offload_to_cpu))
        self.chk_vae_tiling.setChecked(bool(self.cfg.vae_tiling))
        self.out_name_edit.setText(self.cfg.out_name or "")

        self._sync_ref_list()

    # -------- cmd build --------

    def _collect_state(self):
        self.paths.sd_cli = self.sdcli_edit.text().strip()
        self.paths.model_dir = self.modeldir_edit.text().strip() or self.paths.model_dir
        self.models.diffusion_model = str(self.flux_combo.currentData() or "").strip()
        self.models.vae_file = str(self.vae_combo.currentData() or "").strip()
        self.models.llm_model = str(self.llm_combo.currentData() or "").strip()

        self.cfg.prompt = self.prompt_edit.toPlainText().strip()
        self.cfg.negative = self.neg_edit.toPlainText().strip()
        self.cfg.width = int(self.width_spin.value())
        self.cfg.height = int(self.height_spin.value())
        self.cfg.steps = int(self.steps_spin.value())
        self.cfg.cfg_scale = float(self.cfg_spin.value())
        self.cfg.sampling_method = self.sampling_combo.currentText().strip()
        self.cfg.random_seed = bool(self.chk_rand_seed.isChecked())
        self.cfg.seed = int(self.seed_spin.value())
        self.cfg.diffusion_fa = bool(self.chk_diffusion_fa.isChecked())
        self.cfg.offload_to_cpu = bool(self.chk_offload_cpu.isChecked())
        self.cfg.vae_tiling = bool(self.chk_vae_tiling.isChecked())
        self.cfg.out_name = self.out_name_edit.text().strip()

    def _build_cmd(self, out_path: Path) -> List[str]:
        self._collect_state()

        cmd = [self.paths.sd_cli]

        cmd += ["--diffusion-model", self.models.diffusion_model]
        cmd += ["--vae", self.models.vae_file]
        cmd += ["--llm", self.models.llm_model]

        # prompts
        cmd += ["-p", self.cfg.prompt]
        if self.cfg.negative:
            cmd += ["-n", self.cfg.negative]

        # refs
        for r in self.cfg.ref_images:
            cmd += ["-r", r]

        # gen
        cmd += ["-W", str(self.cfg.width), "-H", str(self.cfg.height)]
        cmd += ["--steps", str(self.cfg.steps)]
        cmd += ["--cfg-scale", str(self.cfg.cfg_scale)]
        cmd += ["--sampling-method", self.cfg.sampling_method or "euler"]

        # seed: stable-diffusion.cpp uses random seed for < 0  (see help)
        seed = -1 if self.cfg.random_seed else int(self.cfg.seed)
        cmd += ["-s", str(seed)]

        # perf toggles
        if self.cfg.diffusion_fa:
            cmd += ["--diffusion-fa"]
        if self.cfg.offload_to_cpu:
            cmd += ["--offload-to-cpu"]
        if self.cfg.vae_tiling:
            cmd += ["--vae-tiling"]

        # output
        cmd += ["-o", str(out_path)]

        # verbose so users can see what's going on
        cmd += ["-v"]

        return cmd

    def _update_cmd_preview(self):
        # While we are restoring UI state, Qt emits many change signals.
        # If we collect+save during that phase, we'll overwrite the user's
        # JSON with widget defaults (64x64, steps=1, cfg=0, etc.).
        if getattr(self, "_suspend_autosave", 0) > 0:
            return
        try:
            self._collect_state()
            out_dir = Path(self.paths.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            preview_out = out_dir / "preview_output.png"
            cmd = self._build_cmd(preview_out)
            self.cmd_preview.setPlainText(_pretty_cmd(cmd))
            # autosave on change
            self._save_settings()
        except Exception as e:
            self.cmd_preview.setPlainText(f"(command preview error) {e}")

    # -------- run --------

    def _validate(self) -> Optional[str]:
        self._collect_state()

        if not Path(self.paths.sd_cli).exists():
            return f"sd-cli not found:\n{self.paths.sd_cli}"
        if not Path(self.models.diffusion_model).exists():
            return f"Flux GGUF not found:\n{self.models.diffusion_model}"
        if not Path(self.models.llm_model).exists():
            return f"Qwen3 4B GGUF not found:\n{self.models.llm_model}"
        if not Path(self.models.vae_file).exists():
            return f"VAE not found:\n{self.models.vae_file}"
        if not self.cfg.prompt:
            return "Prompt is empty."
        return None

    def _run(self):
        err = self._validate()
        if err:
            QtWidgets.QMessageBox.warning(self, "Missing / invalid settings", err)
            return

        # output path
        out_dir = Path(self.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.out_name:
            out_path = out_dir / self.cfg.out_name
        else:
            out_path = out_dir / f"klein_{time.strftime('%Y%m%d_%H%M%S')}.png"

        cmd = self._build_cmd(out_path)
        self._save_settings()

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("color: #b60; font-weight: 600;")
        self.runner.run(cmd, cwd=out_dir, expected_out=out_path)

    def _stop(self):
        self.runner.stop()
        self._append_log("[ui] stop requested")

    def _on_finished(self, ok: bool, out_path: str):
        self.btn_run.setEnabled(True)
        if ok and out_path:
            p = Path(out_path)
            self.current_output = p
            self.lbl_status.setText(f"Done: {p.name}")
            self.lbl_status.setStyleSheet("color: #090; font-weight: 600;")
            try:
                img = Image.open(p).convert("RGB")
                self.preview.set_image(img)
            except Exception as e:
                self._append_log(f"[ui] preview failed: {e}")
        else:
            self.lbl_status.setText("Failed")
            self.lbl_status.setStyleSheet("color: #c00; font-weight: 600;")

    # -------- events --------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self.runner.stop()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
