#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrameVision – FLUX.2 Klein 4B & 9B (GGUF loader) Editor UI
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


# --- FrameVision media-explorer results opener ------------------------------
def _fv_open_results_in_media_explorer(widget, folder, preset="images") -> bool:
    """Open/scan a results folder in FrameVision Media Explorer when embedded.

    Falls back to the operating-system file explorer when the main FrameVision
    helper is not available (for standalone tool runs).
    """
    try:
        from pathlib import Path as _Path
        _folder = _Path(folder).expanduser()
        try:
            _folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _folder_s = str(_folder)
    except Exception:
        return False

    def _try_main(_mw) -> bool:
        try:
            if _mw is not None and hasattr(_mw, "open_media_explorer_folder"):
                try:
                    _mw.open_media_explorer_folder(_folder_s, preset=preset, include_subfolders=False)
                    return True
                except TypeError:
                    kwargs = {"include_subfolders": False}
                    if preset == "images":
                        kwargs.update({"want_images": True, "want_videos": False, "want_audio": False})
                    elif preset == "videos":
                        kwargs.update({"want_images": False, "want_videos": True, "want_audio": False})
                    elif preset == "audio":
                        kwargs.update({"want_images": False, "want_videos": False, "want_audio": True})
                    _mw.open_media_explorer_folder(_folder_s, **kwargs)
                    return True
        except Exception:
            pass
        return False

    try:
        _w = widget
        while _w is not None:
            if _try_main(_w):
                return True
            try:
                _w = _w.parent()
            except Exception:
                break
    except Exception:
        pass

    try:
        from PySide6.QtWidgets import QApplication as _QApplication
        _app = _QApplication.instance()
        if _app is not None:
            for _w in _app.topLevelWidgets():
                if _try_main(_w):
                    return True
    except Exception:
        pass

    try:
        from PySide6.QtGui import QDesktopServices as _QDesktopServices
        from PySide6.QtCore import QUrl as _QUrl
        _QDesktopServices.openUrl(_QUrl.fromLocalFile(_folder_s))
        return True
    except Exception:
        pass

    try:
        import os as _os, sys as _sys, subprocess as _subprocess
        if _os.name == "nt":
            _os.startfile(_folder_s)  # type: ignore[attr-defined]
        elif _sys.platform == "darwin":
            _subprocess.Popen(["open", _folder_s])
        else:
            _subprocess.Popen(["xdg-open", _folder_s])
        return True
    except Exception:
        return False
# ---------------------------------------------------------------------------

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
import re

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
    lora_file: str = ""         # optional LoRA adapter (.safetensors/.ckpt/.pt)


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
    lora_strength: float = 1.0
    # Flux Kontext-style ref images (can be used multiple times)
    ref_images: List[str] = None

    def __post_init__(self):
        if self.ref_images is None:
            self.ref_images = []


def _rel_lora_name(lora_path: str, lora_model_dir: str) -> str:
    """Return sd.cpp/webui style LoRA identifier relative to --lora-model-dir, without extension."""
    if not lora_path:
        return ""
    lp = Path(lora_path)
    try:
        base = Path(lora_model_dir) if lora_model_dir else lp.parent
        rel = lp.relative_to(base)
    except Exception:
        rel = Path(lp.name)
    rel_no_ext = rel.with_suffix("")
    return rel_no_ext.as_posix().strip()


def _append_lora_tag(prompt: str, lora_name: str, strength: float) -> str:
    if not lora_name:
        return prompt
    tag = f"<lora:{lora_name}:{strength:g}>"
    if tag in prompt:
        return prompt
    # If the user already typed a LoRA tag manually, leave their prompt alone.
    if re.search(r"<\s*lora\s*:[^>]+>", prompt, flags=re.IGNORECASE):
        return prompt
    prompt = (prompt or "").rstrip()
    return f"{prompt} {tag}".strip()


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


class RefImagePreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Reference preview – {Path(image_path).name}")
        self.resize(1000, 760)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.viewer = ImageView(self)
        layout.addWidget(self.viewer, 1)

        self.path_label = QtWidgets.QLabel(image_path)
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_actual = QtWidgets.QPushButton("100%")
        self.btn_open = QtWidgets.QPushButton("Open folder")
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.btn_fit)
        btn_row.addWidget(self.btn_actual)
        btn_row.addWidget(self.btn_open)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        try:
            img = Image.open(image_path).convert("RGBA")
            self.viewer.set_image(img)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preview failed", str(e))

        self.btn_fit.clicked.connect(lambda: self.viewer.fitInView(self.viewer.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.btn_actual.clicked.connect(self._reset_zoom)
        self.btn_open.clicked.connect(lambda: self._open_folder(image_path))
        self.btn_close.clicked.connect(self.accept)

    def _reset_zoom(self):
        self.viewer.resetTransform()

    def _open_folder(self, image_path: str):
        p = Path(image_path)
        folder = p.parent if p.exists() else p
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass


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
        self._apply_run_mode_ui()

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

            # Migrate older default output folder from /output/klein4b_gguf/
            # to the newer /output/edits/flux_klein/ location, while still
            # respecting any custom folder the user may have chosen.
            try:
                root = Path(self.paths.root)
                old_default = root / "output" / "klein4b_gguf"
                new_default = root / "output" / "edits" / "flux_klein"
                current_out = Path(self.paths.out_dir)
                if current_out == old_default:
                    self.paths.out_dir = str(new_default)
            except Exception:
                pass

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
        self.ref_list.setIconSize(QtCore.QSize(110, 110))
        self.ref_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.ref_list.setWordWrap(True)
        self.ref_list.setSpacing(8)
        self.ref_list.setMinimumHeight(180)
        self.ref_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
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
        self.lora_combo = QtWidgets.QComboBox()
        self.btn_lora = QtWidgets.QPushButton("Browse…")
        self.lora_strength_spin = QtWidgets.QDoubleSpinBox()
        self.lora_strength_spin.setRange(-4.0, 4.0)
        self.lora_strength_spin.setDecimals(2)
        self.lora_strength_spin.setSingleStep(0.05)
        self.lora_strength_spin.setValue(1.0)

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
        model_form.addRow("LoRA (optional):", row(self.lora_combo, self.btn_lora))
        model_form.addRow("LoRA strength:", self.lora_strength_spin)

        # Command preview
        self.cmd_preview = QtWidgets.QPlainTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumBlockCount(2000)
        self.cmd_preview.setMinimumHeight(90)

        self.cmd_group = QtWidgets.QGroupBox("Command preview")
        cmd_layout = QtWidgets.QVBoxLayout(self.cmd_group)
        cmd_layout.addWidget(self.cmd_preview)

        # Run controls (moved to sticky bottom bar)
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_run.setToolTip("Generate image (Ctrl+Enter)")
        self.btn_run.setMinimumHeight(42)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.log.setMinimumHeight(180)

        self.log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(self.log_group)
        log_layout.addWidget(self.log)

        # Output helpers
        out_row = QtWidgets.QHBoxLayout()
        self.btn_open_out = QtWidgets.QPushButton("View results")
        self.btn_save_copy = QtWidgets.QPushButton("Save copy…")
        self.btn_copy_path = QtWidgets.QPushButton("Copy output path")
        self.chk_use_queue = QtWidgets.QCheckBox("Use queue")
        self.chk_use_queue.setChecked(True)
        # Keep View results in the sticky footer with Generate.
        # Save/copy helpers stay in the scrolled output-helper row.
        out_row.addWidget(self.btn_save_copy)
        out_row.addWidget(self.btn_copy_path)
        out_row.addStretch(1)
        out_row.addWidget(self.chk_use_queue)

        # Left assembly
        left_layout.addWidget(QtWidgets.QLabel("Prompt"))
        left_layout.addWidget(self.prompt_edit)
        left_layout.addWidget(QtWidgets.QLabel("Negative prompt"))
        left_layout.addWidget(self.neg_edit)
        left_layout.addWidget(ref_group)
        left_layout.addWidget(gen_group)
        left_layout.addWidget(model_group)
        left_layout.addWidget(self.cmd_group)
        left_layout.addLayout(out_row)
        left_layout.addWidget(self.log_group)
        left_layout.addStretch(1)

        # Right preview
        self.right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        right_layout.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Idle / not queued")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        top.addWidget(self.lbl_status)
        top.addStretch(1)
        top.addWidget(self.btn_fit)

        self.preview_stack = QtWidgets.QStackedWidget()
        self.preview = ImageView()
        self.preview_placeholder = QtWidgets.QLabel(
            "Preview follows direct run only.\n\nTurn off 'Use queue' to see live logs and the finished image here."
        )
        self.preview_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setWordWrap(True)
        self.preview_placeholder.setStyleSheet("padding: 18px; color: #888;")
        self.preview_stack.addWidget(self.preview)
        self.preview_stack.addWidget(self.preview_placeholder)

        right_layout.addLayout(top)
        right_layout.addWidget(self.preview_stack, 1)

        splitter.addWidget(scroll)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 670])

        # Sticky bottom bar (does not scroll)
        bottom = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        self.btn_open_out.setMinimumHeight(42)
        try:
            self.btn_open_out.setMinimumWidth(160)
        except Exception:
            pass
        bottom_layout.addWidget(self.btn_run, 1)
        bottom_layout.addWidget(self.btn_open_out, 0)
        outer.addWidget(bottom, 0)

        # Signals
        self.btn_add_ref.clicked.connect(self._add_refs)
        self.btn_remove_ref.clicked.connect(self._remove_selected_refs)
        self.btn_clear.clicked.connect(self._clear_refs)
        self.btn_up.clicked.connect(lambda: self._move_ref(-1))
        self.btn_down.clicked.connect(lambda: self._move_ref(1))
        self.ref_list.itemDoubleClicked.connect(self._preview_ref_item)
        self.ref_list.customContextMenuRequested.connect(self._show_ref_context_menu)

        self.btn_match_first.clicked.connect(self._match_first_ref)
        self.btn_1024.clicked.connect(lambda: self._set_size(1024,1024))

        self.btn_sdcli.clicked.connect(self._browse_sdcli)
        self.btn_modeldir.clicked.connect(self._browse_model_dir)
        self.btn_flux.clicked.connect(lambda: self._browse_and_set_combo(self.flux_combo, "Flux GGUF", "GGUF (*.gguf);;All files (*.*)"))
        self.btn_llm.clicked.connect(lambda: self._browse_and_set_combo(self.llm_combo, "Qwen3 GGUF", "GGUF (*.gguf);;All files (*.*)"))
        self.btn_vae.clicked.connect(lambda: self._browse_and_set_combo(self.vae_combo, "VAE safetensors", "SafeTensors (*.safetensors);;All files (*.*)"))
        self.btn_lora.clicked.connect(lambda: self._browse_and_set_combo(self.lora_combo, "LoRA adapter", "LoRA (*.safetensors *.ckpt *.pt);;All files (*.*)"))

        self.btn_run.clicked.connect(self._run)
        self.btn_open_out.clicked.connect(self._view_results)
        self.btn_save_copy.clicked.connect(self._save_copy)
        self.btn_copy_path.clicked.connect(self._copy_output_path)
        self.btn_fit.clicked.connect(lambda: self.preview.fitInView(self.preview.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.chk_use_queue.toggled.connect(lambda _=None: self._apply_run_mode_ui())

        # Update cmd preview on edits
        for w in (self.prompt_edit, self.neg_edit, self.width_spin, self.height_spin, self.steps_spin, self.cfg_spin,
                  self.seed_spin, self.chk_rand_seed, self.sampling_combo, self.lora_strength_spin,
                  self.chk_diffusion_fa, self.chk_offload_cpu, self.chk_vae_tiling,
                  self.out_name_edit, self.sdcli_edit, self.modeldir_edit, self.chk_use_queue):
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
        self.lora_combo.currentIndexChanged.connect(lambda _=None: self._update_cmd_preview())

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self._run)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self._run)

    # -------- helpers --------

    def _append_log(self, s: str):
        self.log.appendPlainText(s)

    def _apply_run_mode_ui(self):
        use_queue = bool(self.chk_use_queue.isChecked())
        self.log_group.setVisible(not use_queue)
        self.right_panel.setVisible(not use_queue)
        self.preview_stack.setCurrentWidget(self.preview_placeholder if use_queue else self.preview)
        self.btn_fit.setEnabled(not use_queue)
        self.btn_save_copy.setEnabled(not use_queue and self.current_output is not None and self.current_output.exists())
        self.btn_copy_path.setEnabled(not use_queue and self.current_output is not None)
        if use_queue:
            self.lbl_status.setText("Queued")
            self.lbl_status.setStyleSheet("color: #064; font-weight: 600;")
        else:
            if self.current_output and self.current_output.exists():
                self.lbl_status.setText(f"Ready / last: {self.current_output.name}")
            else:
                self.lbl_status.setText("Idle / direct run")
            self.lbl_status.setStyleSheet("font-weight: 600;")

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
        ggufs: List[Path] = []
        safes: List[Path] = []
        loras_strict: List[Path] = []
        loras_fallback: List[Path] = []
        if base.exists():
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                suf = p.suffix.lower()
                name_l = p.name.lower()
                full_l = p.as_posix().lower()
                if suf == ".gguf":
                    ggufs.append(p)
                elif suf == ".safetensors":
                    safes.append(p)

                if suf in (".safetensors", ".ckpt", ".pt"):
                    if "lora" in full_l or "loras" in full_l or "lora" in name_l:
                        loras_strict.append(p)
                    elif "flux" in name_l and "vae" not in name_l:
                        loras_fallback.append(p)

        # Build lists
        flux = [p for p in ggufs if "flux" in p.name.lower() and "klein" in p.name.lower()]
        llm = [p for p in ggufs if "qwen" in p.name.lower() and "4b" in p.name.lower()]
        vae = [p for p in safes if "vae" in p.name.lower() and ("flux2" in p.name.lower() or "flux" in p.name.lower())]

        loras = loras_strict if loras_strict else loras_fallback

        # Prefer likely LoRA folders / names, but keep it broad enough for custom layouts.
        def lora_sort_key(x: Path):
            full_l = x.as_posix().lower()
            name_l = x.name.lower()
            pri = 0 if ("/loras/" in full_l or full_l.endswith('/loras') or "\\loras\\" in str(x).lower()) else 1
            return (pri, name_l)

        def refill(combo: QtWidgets.QComboBox, items: List[Path], preferred: str, keep_path: str, none_label: str | None = None):
            combo.blockSignals(True)
            combo.clear()
            if none_label is not None:
                combo.addItem(none_label, "")
            # preferred first
            items_sorted = sorted(items, key=lambda x: (0 if preferred and preferred in x.name.lower() else 1, x.name.lower()))
            for pth in items_sorted:
                rel = str(pth.relative_to(base)) if base.exists() else pth.name
                combo.addItem(rel, str(pth))
            combo.blockSignals(False)
            # restore selection
            if keep_path:
                self._set_combo_to_path(combo, keep_path)
            elif none_label is not None:
                combo.setCurrentIndex(0)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

        refill(self.flux_combo, flux, "q4_k_m", getattr(self.models, "diffusion_model", ""))
        refill(self.llm_combo, llm, "q4_k_m", getattr(self.models, "llm_model", ""))
        # prefer flux2_ae
        preferred_vae = "flux2_ae"
        refill(self.vae_combo, vae, preferred_vae, getattr(self.models, "vae_file", ""))
        refill(self.lora_combo, sorted(set(loras), key=lora_sort_key), "", getattr(self.models, "lora_file", ""), none_label="(None)")


    def _set_size(self, w: int, h: int):
        self.width_spin.setValue(w)
        self.height_spin.setValue(h)

    def _make_ref_icon(self, image_path: str) -> QtGui.QIcon:
        pm = QtGui.QPixmap(110, 110)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGBA")
                qimg = ImageQt.ImageQt(img)
                src = QtGui.QPixmap.fromImage(qimg)
            scaled = src.scaled(106, 106, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            painter = QtGui.QPainter(pm)
            painter.fillRect(pm.rect(), QtGui.QColor(24, 24, 24))
            x = int((pm.width() - scaled.width()) / 2)
            y = int((pm.height() - scaled.height()) / 2)
            painter.drawPixmap(x, y, scaled)
            painter.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90)))
            painter.drawRect(pm.rect().adjusted(0, 0, -1, -1))
            painter.end()
        except Exception:
            painter = QtGui.QPainter(pm)
            painter.fillRect(pm.rect(), QtGui.QColor(24, 24, 24))
            painter.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150)))
            painter.drawRect(pm.rect().adjusted(0, 0, -1, -1))
            painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No preview")
            painter.end()
        return QtGui.QIcon(pm)

    def _sync_ref_list(self):
        self.ref_list.clear()
        for p in self.cfg.ref_images:
            item = QtWidgets.QListWidgetItem(self._make_ref_icon(p), f"{Path(p).name}\n{p}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            item.setSizeHint(QtCore.QSize(0, 122))
            self.ref_list.addItem(item)
        self._update_cmd_preview()
        self._save_settings()

    def _preview_ref_item(self, item: QtWidgets.QListWidgetItem | None = None):
        if item is None:
            item = self.ref_list.currentItem()
        if item is None:
            return
        image_path = item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
        if not image_path:
            return
        dlg = RefImagePreviewDialog(str(image_path), self)
        dlg.exec()

    def _show_ref_context_menu(self, pos: QtCore.QPoint):
        item = self.ref_list.itemAt(pos)
        menu = QtWidgets.QMenu(self)

        if item is not None:
            self.ref_list.setCurrentItem(item)
            image_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")

            act_preview = menu.addAction("Preview")
            act_open_folder = menu.addAction("Open folder")
            act_open_with = menu.addAction("Open with…")
            menu.addSeparator()
            act_remove = menu.addAction("Remove")
            act_remove_selected = menu.addAction("Remove selected")

            chosen = menu.exec(self.ref_list.viewport().mapToGlobal(pos))
            if chosen == act_preview:
                self._preview_ref_item(item)
            elif chosen == act_open_folder:
                self._open_ref_folder(image_path)
            elif chosen == act_open_with:
                self._open_ref_with(image_path)
            elif chosen == act_remove:
                row = self.ref_list.row(item)
                if 0 <= row < len(self.cfg.ref_images):
                    self.cfg.ref_images.pop(row)
                    self._sync_ref_list()
            elif chosen == act_remove_selected:
                self._remove_selected_refs()
            return

        act_add = menu.addAction("Add…")
        if self.ref_list.count() > 0:
            act_clear = menu.addAction("Clear all")
        else:
            act_clear = None

        chosen = menu.exec(self.ref_list.viewport().mapToGlobal(pos))
        if chosen == act_add:
            self._add_refs()
        elif act_clear is not None and chosen == act_clear:
            self._clear_refs()

    def _open_ref_folder(self, image_path: str):
        p = Path(image_path)
        folder = p.parent if p.exists() else p
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open folder failed", str(e))

    def _open_ref_with(self, image_path: str):
        p = Path(image_path)
        if not p.exists():
            QtWidgets.QMessageBox.information(self, "Missing file", f"File not found:\n{image_path}")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open with failed", str(e))

    def _add_refs(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add reference images", self.paths.root,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)"
        )
        if not files:
            return
        self.cfg.ref_images.extend([f for f in files if Path(f).exists()])
        self._sync_ref_list()
        self._apply_run_mode_ui()

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

    def _view_results(self):
        _fv_open_results_in_media_explorer(self, Path(self.paths.out_dir), preset="images")

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
        self.lora_strength_spin.setValue(float(getattr(self.cfg, "lora_strength", 1.0)))
        self.chk_use_queue.setChecked(bool(getattr(self.cfg, "use_queue", True)))

        self._sync_ref_list()

    # -------- cmd build --------

    def _collect_state(self):
        self.paths.sd_cli = self.sdcli_edit.text().strip()
        self.paths.model_dir = self.modeldir_edit.text().strip() or self.paths.model_dir
        self.models.diffusion_model = str(self.flux_combo.currentData() or "").strip()
        self.models.vae_file = str(self.vae_combo.currentData() or "").strip()
        self.models.llm_model = str(self.llm_combo.currentData() or "").strip()
        self.models.lora_file = str(self.lora_combo.currentData() or "").strip()

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
        self.cfg.lora_strength = float(self.lora_strength_spin.value())
        self.cfg.use_queue = bool(self.chk_use_queue.isChecked())

    def _build_cmd(self, out_path: Path) -> List[str]:
        self._collect_state()

        cmd = [self.paths.sd_cli]

        cmd += ["--diffusion-model", self.models.diffusion_model]
        cmd += ["--vae", self.models.vae_file]
        cmd += ["--llm", self.models.llm_model]

        prompt_for_cmd = self.cfg.prompt
        if self.models.lora_file:
            lora_dir = str(Path(self.models.lora_file).parent)
            lora_name = _rel_lora_name(self.models.lora_file, lora_dir)
            prompt_for_cmd = _append_lora_tag(prompt_for_cmd, lora_name, float(getattr(self.cfg, "lora_strength", 1.0)))
            cmd += ["--lora-model-dir", lora_dir]

        # prompts
        cmd += ["-p", prompt_for_cmd]
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
            mode = "queue" if bool(getattr(self.cfg, "use_queue", True)) else "direct run"
            self.cmd_preview.setPlainText(f"[{mode}]\n" + _pretty_cmd(cmd))
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
        if self.models.lora_file and not Path(self.models.lora_file).exists():
            return f"LoRA not found:\n{self.models.lora_file}"
        if not self.cfg.prompt:
            return "Prompt is empty."
        return None

    def _open_queue_tab(self):
        try:
            win = self.window()
            tabs = win.findChild(QtWidgets.QTabWidget) if win is not None else None
            if tabs:
                for i in range(tabs.count()):
                    if tabs.tabText(i).strip().lower() == "queue":
                        tabs.setCurrentIndex(i)
                        return
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, "Queue", "This editor now adds jobs to the FrameVision queue. Open the Queue tab in the main app to review, run, or cancel jobs.")

    def _run(self):
        err = self._validate()
        if err:
            QtWidgets.QMessageBox.warning(self, "Missing / invalid settings", err)
            return

        self._collect_state()
        self._save_settings()

        use_queue = bool(getattr(self.cfg, "use_queue", True))
        if use_queue:
            try:
                try:
                    from helpers.queue_adapter import enqueue_flux_klein_from_widget as _enqueue_flux_klein
                except Exception:
                    from queue_adapter import enqueue_flux_klein_from_widget as _enqueue_flux_klein

                jid = _enqueue_flux_klein(self)

                out_file = None
                try:
                    out_file = str(Path(self.paths.out_dir) / self.cfg.out_name) if self.cfg.out_name else None
                except Exception:
                    out_file = None

                self.lbl_status.setText("Queued")
                self.lbl_status.setStyleSheet("color: #064; font-weight: 600;")
                self._append_log(f"[queue] queued Flux Klein job -> {jid}")
                self._apply_run_mode_ui()
                if out_file:
                    self._append_log(f"[queue] target output -> {out_file}")
            except Exception as e:
                self.lbl_status.setText("Queue failed")
                self.lbl_status.setStyleSheet("color: #c00; font-weight: 600;")
                self.preview_stack.setCurrentWidget(self.preview_placeholder)
                QtWidgets.QMessageBox.warning(self, "Queue error", str(e))
            return

        out_dir = Path(self.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.out_name:
            out_path = out_dir / self.cfg.out_name
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"flux_klein_{stamp}.png"

        cmd = self._build_cmd(out_path)
        self.current_output = out_path
        self.preview_stack.setCurrentWidget(self.preview)
        self.lbl_status.setText("Running")
        self.lbl_status.setStyleSheet("color: #06c; font-weight: 600;")
        self.btn_run.setEnabled(False)
        self.runner.run(cmd, Path(self.paths.root), out_path)

    def _stop(self):
        self._open_queue_tab()

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
                self.preview_stack.setCurrentWidget(self.preview)
            except Exception as e:
                self._append_log(f"[ui] preview failed: {e}")
        else:
            self.lbl_status.setText("Failed")
            self.lbl_status.setStyleSheet("color: #c00; font-weight: 600;")
        self._apply_run_mode_ui()

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
