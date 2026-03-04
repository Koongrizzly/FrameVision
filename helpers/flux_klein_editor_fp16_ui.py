#!/usr/bin/env python3
"""
Flux Klein 4B (FLUX.2-klein-4B) — PySide6 local UI editor (text-to-image + multi-reference image editing)

How it works:
- The UI (this file) runs with your normal Python.
- Inference runs in a separate persistent worker process using:
    <FrameVisionRoot>/environments/.klein4B/python.exe
  so you can keep all model deps isolated.
- The worker loads the model from:
    <FrameVisionRoot>/models/klein4b/
  (point it at your local diffusers repo folder that contains model_index.json, transformer/, vae/, etc.)
- Optional: ffmpeg/ffprobe are expected at:
    <FrameVisionRoot>/presets/bin/

Model notes (from official sources):
- The Distilled model is commonly run with num_inference_steps=4 and guidance_scale=1.0.
- Editing is done by providing `image=` (a single PIL image or list of images) in the pipeline call.

This UI exposes:
- Prompt, seed/random seed, width/height (with "match first image" helper), steps, guidance scale
- Mode presets (Distilled 4-step vs Base 50-step) with default steps/CFG
- Reference image manager (add/remove/reorder; 0 images => pure text-to-image, >=1 => editing/multi-ref)
- Offload/slicing/tiling toggles (worker-side)
- History list with quick reload and "use as base image"
- Export helpers: save as PNG/JPG/WebP, and optional ffmpeg comparison video/gif

Requirements (UI side):
- PySide6
- pillow (PIL)

Requirements (worker side, in your .klein4B env):
- torch
- diffusers (a version that includes Flux2KleinPipeline; the model card recommends installing diffusers from GitHub)
- transformers, accelerate, safetensors, etc. per diffusers/flux requirements
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageQt

from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Root/path discovery
# -----------------------------

def _find_framevision_root(start: Path) -> Path:
    """
    Try to find the FrameVision root by walking up from `start` until we find:
      - environments/
      - models/
      - presets/bin/
    Falls back to the script directory if not found.
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "environments").is_dir() and (cur / "models").is_dir() and (cur / "presets" / "bin").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


@dataclass
class AppPaths:
    root: str
    python_exe: str
    model_dir: str
    presets_bin: str

    @staticmethod
    def from_root(root: Path) -> "AppPaths":
        env_root = root / "environments" / ".klein4B"
        if os.name == "nt":
            py_candidates = [
                env_root / "Scripts" / "python.exe",
                env_root / "scripts" / "python.exe",
                env_root / "python.exe",
            ]
        else:
            py_candidates = [env_root / "bin" / "python", env_root / "python"]
        py = next((p for p in py_candidates if p.exists()), py_candidates[0])
        model_dir = root / "models" / "klein4b"
        presets_bin = root / "presets" / "bin"
        return AppPaths(
            root=str(root),
            python_exe=str(py),
            model_dir=str(model_dir),
            presets_bin=str(presets_bin),
        )


# -----------------------------
# Worker protocol
# -----------------------------

_WORKER_CODE = r"""
import json
import os
import sys
import time
from pathlib import Path

def _log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _safe_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1","true","yes","y","on")
    return bool(x)

def main():
    # Read init message first
    init_line = sys.stdin.readline()
    if not init_line:
        return
    init = json.loads(init_line)

    model_dir = Path(init["model_dir"]).resolve()
    out_dir = Path(init["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional toggles
    device = init.get("device", "cuda")
    dtype_name = init.get("dtype", "bfloat16")  # bfloat16/float16/float32
    enable_cpu_offload = _safe_bool(init.get("cpu_offload", False))
    enable_attention_slicing = _safe_bool(init.get("attention_slicing", False))
    enable_vae_slicing = _safe_bool(init.get("vae_slicing", False))
    enable_vae_tiling = _safe_bool(init.get("vae_tiling", False))

    # Lazy imports inside worker env
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

    _log(f"[klein-worker] loading model from: {model_dir}")
    _log(f"[klein-worker] device={device} dtype={dtype_name}")
    pipe = Flux2KleinPipeline.from_pretrained(str(model_dir), torch_dtype=torch_dtype)

    if device == "cuda" and torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    # Optional memory helpers
    try:
        if enable_attention_slicing:
            pipe.enable_attention_slicing()
    except Exception as e:
        _log(f"[klein-worker] attention_slicing not available: {e}")
    try:
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
    except Exception as e:
        _log(f"[klein-worker] cpu offload not available: {e}")
    try:
        if enable_vae_slicing:
            pipe.vae.enable_slicing()
    except Exception as e:
        _log(f"[klein-worker] vae slicing not available: {e}")
    try:
        if enable_vae_tiling:
            pipe.vae.enable_tiling()
    except Exception as e:
        _log(f"[klein-worker] vae tiling not available: {e}")

    pipe.set_progress_bar_config(disable=True)

    _send({"type":"ready"})

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line)
        mtype = msg.get("type")

        if mtype == "shutdown":
            _send({"type":"bye"})
            break

        if mtype != "run":
            _send({"type":"error", "error": f"Unknown message type: {mtype}"})
            continue

        req_id = msg.get("id", "")
        try:
            prompt = msg["prompt"]
            width = int(msg["width"])
            height = int(msg["height"])
            steps = int(msg["steps"])
            guidance = float(msg["guidance"])
            seed = msg.get("seed", None)

            # Reference images
            image_paths = msg.get("images", []) or []
            pil_list = []
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                pil_list.append(img)

            generator = None
            if seed is not None:
                # Generator must be on correct device
                gen_device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(int(seed))

            kwargs = dict(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            if generator is not None:
                kwargs["generator"] = generator
            if pil_list:
                kwargs["image"] = pil_list if len(pil_list) > 1 else pil_list[0]

            t0 = time.time()
            out = pipe(**kwargs).images[0]
            dt = time.time() - t0

            # Save
            name = msg.get("save_name") or f"klein_{int(time.time())}_{req_id[:8]}.png"
            out_path = out_dir / name
            out.save(out_path)

            _send({"type":"result", "id": req_id, "path": str(out_path), "seconds": dt})

        except Exception as e:
            _send({"type":"error", "id": req_id, "error": repr(e)})

if __name__ == "__main__":
    main()
"""


class KleinWorker(QtCore.QObject):
    """
    Persistent subprocess worker that runs inference in the dedicated env.
    Communicates via newline-delimited JSON over stdin/stdout.
    """
    readyChanged = QtCore.Signal(bool)
    logLine = QtCore.Signal(str)
    resultReady = QtCore.Signal(str, str, float)  # (req_id, path, seconds)
    errorRaised = QtCore.Signal(str, str)         # (req_id, error)

    def __init__(self, paths: AppPaths, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.paths = paths
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = False
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="klein_ui_"))
        self._worker_py = self._tmp_dir / "klein_worker.py"
        self._worker_py.write_text(_WORKER_CODE, encoding="utf-8")

    def is_ready(self) -> bool:
        return self._ready

    def start(self, init_overrides: Optional[Dict[str, Any]] = None) -> None:
        if self._proc and self._proc.poll() is None:
            return

        py_exe = Path(self.paths.python_exe)
        if not py_exe.exists():
            raise FileNotFoundError(f"python.exe not found: {py_exe}")

        model_dir = Path(self.paths.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        out_dir = Path(self.paths.root) / "output" / "klein4b"
        out_dir.mkdir(parents=True, exist_ok=True)

        init_msg = {
            "model_dir": str(model_dir),
            "out_dir": str(out_dir),
            "device": "cuda",
            "dtype": "bfloat16",
            "cpu_offload": False,
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
        }
        if init_overrides:
            init_msg.update(init_overrides)

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        self._proc = subprocess.Popen(
            [str(py_exe), "-u", str(self._worker_py)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        assert self._proc.stdin and self._proc.stdout and self._proc.stderr

        # Send init
        self._proc.stdin.write(json.dumps(init_msg) + "\n")
        self._proc.stdin.flush()

        # Start readers
        self._ready = False
        self.readyChanged.emit(False)

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        threading.Thread(target=self._stderr_loop, daemon=True).start()

    def shutdown(self) -> None:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return
            try:
                assert self._proc.stdin
                self._proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self, payload: Dict[str, Any]) -> str:
        """
        Send a 'run' request. Returns request id.
        """
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("Worker is not running.")
        if not self._ready:
            raise RuntimeError("Worker is not ready yet (model still loading).")

        req_id = payload.get("id") or str(uuid.uuid4())
        msg = dict(payload)
        msg["type"] = "run"
        msg["id"] = req_id

        with self._lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        return req_id

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                self.logLine.emit(f"[worker:stdout] {line}")
                continue

            mtype = msg.get("type")
            if mtype == "ready":
                self._ready = True
                self.readyChanged.emit(True)
            elif mtype == "result":
                self.resultReady.emit(msg.get("id", ""), msg.get("path", ""), float(msg.get("seconds", 0.0)))
            elif mtype == "error":
                self.errorRaised.emit(msg.get("id", ""), msg.get("error", "Unknown error"))
            elif mtype == "bye":
                self.logLine.emit("[worker] shutdown complete")
                self._ready = False
                self.readyChanged.emit(False)
                break
            else:
                self.logLine.emit(f"[worker] {msg}")

        self._ready = False
        self.readyChanged.emit(False)

    def _stderr_loop(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self.logLine.emit(line.rstrip("\n"))


# -----------------------------
# Image viewer widget (zoom/pan)
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
        self._zoom = 0

    def set_image(self, img: Optional[Image.Image]) -> None:
        if img is None:
            self._pix_item.setPixmap(QtGui.QPixmap())
            self.scene().setSceneRect(QtCore.QRectF())
            return
        qimg = ImageQt.ImageQt(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        self._pix_item.setPixmap(pix)
        self.scene().setSceneRect(QtCore.QRectF(pix.rect()))
        self._zoom = 0
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pix_item.pixmap().isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_F:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            event.accept()
            return
        super().keyPressEvent(event)


# -----------------------------
# UI models
# -----------------------------

@dataclass
class RunConfig:
    prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 1.0
    seed: Optional[int] = 0
    random_seed: bool = False
    save_name: str = ""
    # memory toggles
    dtype: str = "bfloat16"     # bfloat16/float16/float32
    cpu_offload: bool = False
    attention_slicing: bool = False
    vae_slicing: bool = False
    vae_tiling: bool = False


@dataclass
class HistoryItem:
    when: float
    prompt: str
    images: List[str]
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int]
    output_path: str
    seconds: float


# -----------------------------
# Main window
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FLUX.2 Klein 4B — Local Editor (PySide6)")
        self.resize(1280, 780)

        # Discover paths
        root = _find_framevision_root(Path(__file__).resolve().parent)
        self.paths = AppPaths.from_root(root)

        # State
        self.run_cfg = RunConfig()
        self.ref_images: List[Path] = []
        self.base_image_path: Optional[Path] = None
        self.current_output: Optional[Path] = None
        self.history: List[HistoryItem] = []

        # Worker
        self.worker = KleinWorker(self.paths)
        self.worker.readyChanged.connect(self._on_worker_ready)
        self.worker.logLine.connect(self._append_log)
        self.worker.resultReady.connect(self._on_result)
        self.worker.errorRaised.connect(self._on_error)

        self._pending_req: Optional[str] = None
        self._pending_payload: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._load_settings()
        self._refresh_paths_ui()
        self._ensure_worker_started()

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(self.splitter)

        # Left: controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Prompt
        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Prompt / edit instruction.\n"
                                            "Tip: With reference images, be explicit about what stays the same.")
        self.prompt_edit.setMinimumHeight(120)

        # Reference images manager
        ref_group = QtWidgets.QGroupBox("Input image(s) / references (0 = text-to-image; 1+ = edit / multi-ref)")
        ref_layout = QtWidgets.QVBoxLayout(ref_group)
        self.ref_list = QtWidgets.QListWidget()
        self.ref_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        ref_layout.addWidget(self.ref_list)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_ref = QtWidgets.QPushButton("Add…")
        self.btn_remove_ref = QtWidgets.QPushButton("Remove")
        self.btn_up_ref = QtWidgets.QPushButton("Up")
        self.btn_down_ref = QtWidgets.QPushButton("Down")
        self.btn_clear_ref = QtWidgets.QPushButton("Clear")
        for b in (self.btn_add_ref, self.btn_remove_ref, self.btn_up_ref, self.btn_down_ref, self.btn_clear_ref):
            btn_row.addWidget(b)
        ref_layout.addLayout(btn_row)

        # Size + steps + cfg
        gen_group = QtWidgets.QGroupBox("Generation / Editing settings")
        form = QtWidgets.QFormLayout(gen_group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Distilled (4 steps)", "Base (50 steps)"])

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setSingleStep(8)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setSingleStep(8)

        self.btn_match_first = QtWidgets.QPushButton("Match 1st image")
        self.btn_square_1024 = QtWidgets.QPushButton("1024×1024")
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QtWidgets.QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_row.addWidget(self.btn_match_first)
        size_row.addWidget(self.btn_square_1024)
        size_row.addStretch(1)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.cfg_spin = QtWidgets.QDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 20.0)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setSingleStep(0.25)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setSingleStep(1)
        self.chk_random_seed = QtWidgets.QCheckBox("Random seed")

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed_spin)
        seed_row.addWidget(self.chk_random_seed)
        seed_row.addStretch(1)

        self.save_name_edit = QtWidgets.QLineEdit()
        self.save_name_edit.setPlaceholderText("Optional output filename (e.g. my_edit.png). Leave empty for auto.")

        form.addRow("Mode preset:", self.mode_combo)
        form.addRow("Size:", size_row)
        form.addRow("Steps:", self.steps_spin)
        form.addRow("Guidance (CFG):", self.cfg_spin)
        form.addRow("Seed:", seed_row)
        form.addRow("Save name:", self.save_name_edit)

        # Memory & performance toggles
        perf_group = QtWidgets.QGroupBox("Performance / VRAM helpers (worker-side)")
        perf_form = QtWidgets.QFormLayout(perf_group)

        self.dtype_combo = QtWidgets.QComboBox()
        self.dtype_combo.addItems(["bfloat16", "float16", "float32"])

        self.chk_cpu_offload = QtWidgets.QCheckBox("Enable model CPU offload")
        self.chk_attention_slicing = QtWidgets.QCheckBox("Enable attention slicing")
        self.chk_vae_slicing = QtWidgets.QCheckBox("Enable VAE slicing")
        self.chk_vae_tiling = QtWidgets.QCheckBox("Enable VAE tiling")

        perf_form.addRow("dtype:", self.dtype_combo)
        perf_form.addRow("", self.chk_cpu_offload)
        perf_form.addRow("", self.chk_attention_slicing)
        perf_form.addRow("", self.chk_vae_slicing)
        perf_form.addRow("", self.chk_vae_tiling)

        # Run controls
        self.btn_run = QtWidgets.QPushButton("Run (Ctrl+Enter)")
        self.btn_run.setMinimumHeight(42)
        self.btn_stop = QtWidgets.QPushButton("Stop worker")
        self.btn_restart = QtWidgets.QPushButton("Restart worker")

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.btn_run, 2)
        run_row.addWidget(self.btn_restart, 1)
        run_row.addWidget(self.btn_stop, 1)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setMinimumHeight(160)

        # Paths/settings
        paths_group = QtWidgets.QGroupBox("Paths (auto-detected; override if needed)")
        paths_form = QtWidgets.QFormLayout(paths_group)
        self.root_edit = QtWidgets.QLineEdit()
        self.python_edit = QtWidgets.QLineEdit()
        self.model_edit = QtWidgets.QLineEdit()
        self.bin_edit = QtWidgets.QLineEdit()
        self.btn_browse_root = QtWidgets.QPushButton("Browse…")
        self.btn_browse_python = QtWidgets.QPushButton("Browse…")
        self.btn_browse_model = QtWidgets.QPushButton("Browse…")
        self.btn_browse_bin = QtWidgets.QPushButton("Browse…")

        def row_with_btn(edit: QtWidgets.QLineEdit, btn: QtWidgets.QPushButton) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(edit, 1)
            l.addWidget(btn, 0)
            return w

        paths_form.addRow("FrameVision root:", row_with_btn(self.root_edit, self.btn_browse_root))
        paths_form.addRow("Worker python.exe:", row_with_btn(self.python_edit, self.btn_browse_python))
        paths_form.addRow("Model dir:", row_with_btn(self.model_edit, self.btn_browse_model))
        paths_form.addRow("presets/bin:", row_with_btn(self.bin_edit, self.btn_browse_bin))

        self.btn_apply_paths = QtWidgets.QPushButton("Apply paths + restart worker")

        # History
        hist_group = QtWidgets.QGroupBox("History")
        hist_layout = QtWidgets.QVBoxLayout(hist_group)
        self.hist_list = QtWidgets.QListWidget()
        self.hist_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        hist_btns = QtWidgets.QHBoxLayout()
        self.btn_use_output_as_base = QtWidgets.QPushButton("Use output as base image")
        self.btn_open_output = QtWidgets.QPushButton("Open output folder")
        self.btn_save_copy = QtWidgets.QPushButton("Save copy…")
        self.btn_export_compare = QtWidgets.QPushButton("Export compare (mp4/gif)…")
        for b in (self.btn_use_output_as_base, self.btn_open_output, self.btn_save_copy, self.btn_export_compare):
            hist_btns.addWidget(b)
        hist_layout.addWidget(self.hist_list)
        hist_layout.addLayout(hist_btns)

        # Assemble left side
        left_layout.addWidget(QtWidgets.QLabel("Prompt"))
        left_layout.addWidget(self.prompt_edit)
        left_layout.addWidget(ref_group)
        left_layout.addWidget(gen_group)
        left_layout.addWidget(perf_group)
        left_layout.addLayout(run_row)
        left_layout.addWidget(hist_group)
        left_layout.addWidget(paths_group)
        left_layout.addWidget(self.btn_apply_paths)
        left_layout.addWidget(QtWidgets.QLabel("Log"))
        left_layout.addWidget(self.log, 1)

        # Make the left panel scrollable (so the Paths section is always reachable)
        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setWidget(left)

        # Right: image preview
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.preview = ImageView()
        self.preview.setMinimumWidth(520)

        info_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Worker: starting…")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        self.lbl_out = QtWidgets.QLabel("")
        self.btn_fit = QtWidgets.QPushButton("Fit (F)")
        self.btn_copy_path = QtWidgets.QPushButton("Copy output path")
        info_bar.addWidget(self.lbl_status, 0)
        info_bar.addStretch(1)
        info_bar.addWidget(self.lbl_out, 0)
        info_bar.addStretch(1)
        info_bar.addWidget(self.btn_fit, 0)
        info_bar.addWidget(self.btn_copy_path, 0)

        right_layout.addLayout(info_bar)
        right_layout.addWidget(self.preview, 1)

        # Splitter
        self.splitter.addWidget(self.left_scroll)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([640, 640])

        # Signals
        self.btn_add_ref.clicked.connect(self._add_ref_images)
        self.btn_remove_ref.clicked.connect(self._remove_selected_refs)
        self.btn_clear_ref.clicked.connect(self._clear_refs)
        self.btn_up_ref.clicked.connect(lambda: self._move_ref(-1))
        self.btn_down_ref.clicked.connect(lambda: self._move_ref(1))

        self.btn_match_first.clicked.connect(self._match_first_image)
        self.btn_square_1024.clicked.connect(lambda: self._set_size(1024, 1024))
        self.mode_combo.currentTextChanged.connect(self._apply_mode_preset)

        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop_worker)
        self.btn_restart.clicked.connect(self._restart_worker)
        self.btn_fit.clicked.connect(lambda: self.preview.fitInView(self.preview.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.btn_copy_path.clicked.connect(self._copy_output_path)

        self.btn_open_output.clicked.connect(self._open_output_folder)
        self.btn_save_copy.clicked.connect(self._save_copy_as)
        self.btn_export_compare.clicked.connect(self._export_compare)

        self.btn_apply_paths.clicked.connect(self._apply_paths_and_restart)

        self.hist_list.currentRowChanged.connect(self._on_history_selected)

        for btn, edit, kind in (
            (self.btn_browse_root, self.root_edit, "dir"),
            (self.btn_browse_python, self.python_edit, "file"),
            (self.btn_browse_model, self.model_edit, "dir"),
            (self.btn_browse_bin, self.bin_edit, "dir"),
        ):
            btn.clicked.connect(lambda _=False, e=edit, k=kind: self._browse_path(e, k))

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self._run)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self._run)

    # ---------------- settings ----------------

    def _settings_path(self) -> Path:
        cfg_dir = Path(self.paths.root) / "presets" / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "flux_klein_ui.json"

    def _load_settings(self) -> None:
        p = self._settings_path()
        if not p.exists():
            # defaults
            self._apply_mode_preset(self.mode_combo.currentText())
            self.width_spin.setValue(1024)
            self.height_spin.setValue(1024)
            self.seed_spin.setValue(0)
            self.chk_random_seed.setChecked(False)
            self.dtype_combo.setCurrentText("bfloat16")
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # paths
            self.paths = AppPaths(**data.get("paths", asdict(self.paths)))
            # auto-fix stale paths from older runs (e.g., python.exe moved under Scripts/)
            try:
                if not Path(self.paths.python_exe).exists():
                    fresh = AppPaths.from_root(Path(self.paths.root))
                    self.paths.python_exe = fresh.python_exe
                if not Path(self.paths.model_dir).exists():
                    fresh = AppPaths.from_root(Path(self.paths.root))
                    self.paths.model_dir = fresh.model_dir
                if not Path(self.paths.presets_bin).exists():
                    fresh = AppPaths.from_root(Path(self.paths.root))
                    self.paths.presets_bin = fresh.presets_bin
            except Exception:
                pass

            # run cfg
            rc = data.get("run_cfg", {})
            self.mode_combo.setCurrentText(rc.get("mode", "Distilled (4 steps)"))
            self.width_spin.setValue(int(rc.get("width", 1024)))
            self.height_spin.setValue(int(rc.get("height", 1024)))
            self.steps_spin.setValue(int(rc.get("steps", 4)))
            self.cfg_spin.setValue(float(rc.get("guidance", 1.0)))
            self.seed_spin.setValue(int(rc.get("seed", 0)))
            self.chk_random_seed.setChecked(bool(rc.get("random_seed", False)))
            self.dtype_combo.setCurrentText(rc.get("dtype", "bfloat16"))
            self.chk_cpu_offload.setChecked(bool(rc.get("cpu_offload", False)))
            self.chk_attention_slicing.setChecked(bool(rc.get("attention_slicing", False)))
            self.chk_vae_slicing.setChecked(bool(rc.get("vae_slicing", False)))
            self.chk_vae_tiling.setChecked(bool(rc.get("vae_tiling", False)))
        except Exception as e:
            self._append_log(f"[settings] failed to load settings: {e}")

    def _save_settings(self) -> None:
        p = self._settings_path()
        data = {
            "paths": asdict(self.paths),
            "run_cfg": {
                "mode": self.mode_combo.currentText(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "steps": self.steps_spin.value(),
                "guidance": self.cfg_spin.value(),
                "seed": self.seed_spin.value(),
                "random_seed": self.chk_random_seed.isChecked(),
                "dtype": self.dtype_combo.currentText(),
                "cpu_offload": self.chk_cpu_offload.isChecked(),
                "attention_slicing": self.chk_attention_slicing.isChecked(),
                "vae_slicing": self.chk_vae_slicing.isChecked(),
                "vae_tiling": self.chk_vae_tiling.isChecked(),
            }
        }
        try:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self._append_log(f"[settings] failed to save settings: {e}")

    # ---------------- paths ----------------

    def _refresh_paths_ui(self) -> None:
        self.root_edit.setText(self.paths.root)
        self.python_edit.setText(self.paths.python_exe)
        self.model_edit.setText(self.paths.model_dir)
        self.bin_edit.setText(self.paths.presets_bin)

    def _browse_path(self, edit: QtWidgets.QLineEdit, kind: str) -> None:
        start = edit.text().strip() or self.paths.root
        if kind == "dir":
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
            if p:
                edit.setText(p)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start)
            if p:
                edit.setText(p)

    def _apply_paths_and_restart(self) -> None:
        self.paths = AppPaths(
            root=self.root_edit.text().strip(),
            python_exe=self.python_edit.text().strip(),
            model_dir=self.model_edit.text().strip(),
            presets_bin=self.bin_edit.text().strip(),
        )
        self._save_settings()
        self._restart_worker()

    # ---------------- worker lifecycle ----------------

    def _ensure_worker_started(self) -> None:
        try:
            init_overrides = {
                "dtype": self.dtype_combo.currentText(),
                "cpu_offload": self.chk_cpu_offload.isChecked(),
                "attention_slicing": self.chk_attention_slicing.isChecked(),
                "vae_slicing": self.chk_vae_slicing.isChecked(),
                "vae_tiling": self.chk_vae_tiling.isChecked(),
            }
            self.worker.paths = self.paths
            self.worker.start(init_overrides=init_overrides)
        except Exception as e:
            self._append_log(f"[worker] failed to start: {e}")
            self.lbl_status.setText("Worker: failed to start (check paths)")
            self.lbl_status.setStyleSheet("color: #c00; font-weight: 600;")

    def _stop_worker(self) -> None:
        self.worker.shutdown()

    def _restart_worker(self) -> None:
        self.worker.shutdown()
        # small delay so process releases GPU cleanly
        QtCore.QTimer.singleShot(200, self._ensure_worker_started)

    def _on_worker_ready(self, ready: bool) -> None:
        if ready:
            self.lbl_status.setText("Worker: ready")
            self.lbl_status.setStyleSheet("color: #090; font-weight: 600;")
        else:
            self.lbl_status.setText("Worker: loading / offline")
            self.lbl_status.setStyleSheet("color: #555; font-weight: 600;")

    # ---------------- ref images ----------------

    def _add_ref_images(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add reference images",
            self.paths.root,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
        )
        if not files:
            return
        for f in files:
            p = Path(f)
            if p.exists():
                self.ref_images.append(p)
        self._sync_ref_list()

    def _remove_selected_refs(self) -> None:
        rows = sorted({i.row() for i in self.ref_list.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.ref_images):
                self.ref_images.pop(r)
        self._sync_ref_list()

    def _clear_refs(self) -> None:
        self.ref_images.clear()
        self._sync_ref_list()

    def _move_ref(self, delta: int) -> None:
        row = self.ref_list.currentRow()
        if row < 0 or row >= len(self.ref_images):
            return
        new_row = row + delta
        if new_row < 0 or new_row >= len(self.ref_images):
            return
        self.ref_images[row], self.ref_images[new_row] = self.ref_images[new_row], self.ref_images[row]
        self._sync_ref_list()
        self.ref_list.setCurrentRow(new_row)

    def _sync_ref_list(self) -> None:
        self.ref_list.clear()
        for p in self.ref_images:
            self.ref_list.addItem(str(p))

    # ---------------- helpers ----------------

    def _apply_mode_preset(self, mode: str) -> None:
        # Defaults as used in BFL's official Space app
        if "Base" in mode:
            self.steps_spin.setValue(50)
            self.cfg_spin.setValue(4.0)
        else:
            self.steps_spin.setValue(4)
            self.cfg_spin.setValue(1.0)

    def _match_first_image(self) -> None:
        if not self.ref_images:
            QtWidgets.QMessageBox.information(self, "No input images", "Add at least one input image first.")
            return
        try:
            img = Image.open(self.ref_images[0]).convert("RGB")
            w, h = img.size
            # Flux models typically like multiples of 8/16. Keep it at multiple of 8.
            w = max(64, int(round(w / 8) * 8))
            h = max(64, int(round(h / 8) * 8))
            self._set_size(w, h)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Failed", f"Couldn't read first image size:\n{e}")

    def _set_size(self, w: int, h: int) -> None:
        self.width_spin.setValue(int(w))
        self.height_spin.setValue(int(h))

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    # ---------------- run ----------------

    def _run(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QtWidgets.QMessageBox.information(self, "Missing prompt", "Please enter a prompt / edit instruction.")
            return

        # Update cfg + persist
        self._save_settings()

        seed: Optional[int]
        if self.chk_random_seed.isChecked():
            seed = None
        else:
            seed = int(self.seed_spin.value())

        payload = {
            "prompt": prompt,
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "steps": int(self.steps_spin.value()),
            "guidance": float(self.cfg_spin.value()),
            "seed": seed,
            "images": [str(p) for p in self.ref_images],
            "save_name": self.save_name_edit.text().strip(),
        }

        try:
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Running…")
            self.lbl_out.setText("")

            self._pending_payload = payload
            req_id = self.worker.run(payload)
            self._pending_req = req_id
            self._append_log(f"[ui] submitted job {req_id}")
        except Exception as e:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("Run (Ctrl+Enter)")
            QtWidgets.QMessageBox.warning(self, "Run failed", str(e))

    def _on_result(self, req_id: str, path: str, seconds: float) -> None:
        if self._pending_req and req_id != self._pending_req:
            # Another run finished; still record it.
            self._append_log(f"[ui] result (non-active) {req_id} -> {path}")
        self._pending_req = None

        out_path = Path(path)
        self.current_output = out_path
        self.lbl_out.setText(f"Saved: {out_path.name}  ({seconds:.2f}s)")
        self._append_log(f"[ui] done in {seconds:.2f}s -> {path}")

        # Preview
        try:
            img = Image.open(out_path).convert("RGB")
            self.preview.set_image(img)
        except Exception as e:
            self._append_log(f"[ui] failed to preview image: {e}")

        # History item
        payload = self._pending_payload or {}
        self._pending_payload = None
        hi = HistoryItem(
            when=time.time(),
            prompt=payload.get("prompt", ""),
            images=list(payload.get("images", [])),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            steps=int(payload.get("steps", 0)),
            guidance=float(payload.get("guidance", 0.0)),
            seed=payload.get("seed", None),
            output_path=str(out_path),
            seconds=float(seconds),
        )
        self.history.insert(0, hi)
        self._rebuild_history_list()

        # Reset run button
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run (Ctrl+Enter)")

    def _on_error(self, req_id: str, error: str) -> None:
        if self._pending_req and req_id == self._pending_req:
            self._pending_req = None
        self._append_log(f"[ui] ERROR: {error}")

        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run (Ctrl+Enter)")
        QtWidgets.QMessageBox.critical(self, "Generation error", error)

    # ---------------- history ----------------

    def _rebuild_history_list(self) -> None:
        self.hist_list.clear()
        for i, item in enumerate(self.history):
            ts = time.strftime("%H:%M:%S", time.localtime(item.when))
            base = "edit" if item.images else "t2i"
            name = Path(item.output_path).name
            self.hist_list.addItem(f"[{ts}] {base} • {item.width}×{item.height} • {item.steps} steps • cfg {item.guidance:g} • {name}")
        if self.history:
            self.hist_list.setCurrentRow(0)

    def _on_history_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.history):
            return
        item = self.history[row]
        # load preview
        p = Path(item.output_path)
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                self.preview.set_image(img)
                self.current_output = p
                self.lbl_out.setText(f"Saved: {p.name}  ({item.seconds:.2f}s)")
            except Exception as e:
                self._append_log(f"[ui] failed to preview history image: {e}")

    def _copy_output_path(self) -> None:
        if not self.current_output:
            return
        QtWidgets.QApplication.clipboard().setText(str(self.current_output))
        self._append_log("[ui] copied output path to clipboard")

    def _open_output_folder(self) -> None:
        out_dir = Path(self.paths.root) / "output" / "klein4b"
        out_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])

    def _save_copy_as(self) -> None:
        if not self.current_output or not self.current_output.exists():
            QtWidgets.QMessageBox.information(self, "No output", "Generate something first.")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save a copy",
            str(self.current_output),
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not p:
            return
        try:
            img = Image.open(self.current_output).convert("RGB")
            ext = Path(p).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img.save(p, quality=95, subsampling=0)
            else:
                img.save(p)
            self._append_log(f"[ui] saved copy -> {p}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _export_compare(self) -> None:
        """
        Create a simple side-by-side compare (before/after) MP4 or GIF using ffmpeg.
        - Uses first reference image as "before" if present; otherwise uses previous history item.
        """
        if not self.current_output or not self.current_output.exists():
            QtWidgets.QMessageBox.information(self, "No output", "Generate something first.")
            return

        ffmpeg = Path(self.paths.presets_bin) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not ffmpeg.exists():
            QtWidgets.QMessageBox.warning(self, "ffmpeg missing", f"ffmpeg not found at:\n{ffmpeg}")
            return

        before: Optional[Path] = None
        if self.ref_images:
            before = self.ref_images[0]
        elif len(self.history) >= 2:
            before = Path(self.history[1].output_path)

        if not before or not before.exists():
            QtWidgets.QMessageBox.information(self, "No 'before' image", "Add a reference image first (or run twice).")
            return

        out_dir = Path(self.paths.root) / "output" / "klein4b"
        out_dir.mkdir(parents=True, exist_ok=True)

        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export compare (MP4/GIF)",
            str(out_dir / f"compare_{int(time.time())}.mp4"),
            "Video (*.mp4);;GIF (*.gif)"
        )
        if not p:
            return

        # Build: side-by-side, then hold 2 seconds
        # We create a 2s 30fps video from single frames.
        try:
            ext = Path(p).suffix.lower()
            if ext == ".gif":
                # GIF: 512 wide each, 2 seconds
                cmd = [
                    str(ffmpeg), "-y",
                    "-loop", "1", "-t", "2", "-i", str(before),
                    "-loop", "1", "-t", "2", "-i", str(self.current_output),
                    "-filter_complex",
                    " [0:v]scale=512:-1,setsar=1[left];"
                    " [1:v]scale=512:-1,setsar=1[right];"
                    " [left][right]hstack=inputs=2,split[s0][s1];"
                    " [s0]palettegen[p];[s1][p]paletteuse ",
                    "-r", "30",
                    str(p),
                ]
            else:
                cmd = [
                    str(ffmpeg), "-y",
                    "-loop", "1", "-t", "2", "-i", str(before),
                    "-loop", "1", "-t", "2", "-i", str(self.current_output),
                    "-filter_complex",
                    " [0:v]scale=640:-1,setsar=1[left];"
                    " [1:v]scale=640:-1,setsar=1[right];"
                    " [left][right]hstack=inputs=2 ",
                    "-r", "30",
                    "-pix_fmt", "yuv420p",
                    str(p),
                ]
            self._append_log("[ui] ffmpeg: " + " ".join(cmd))
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            subprocess.check_call(cmd, creationflags=creationflags)
            self._append_log(f"[ui] exported compare -> {p}")
        except subprocess.CalledProcessError as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", f"ffmpeg failed:\n{e}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    # ---------------- events ----------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        try:
            self.worker.shutdown()
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
