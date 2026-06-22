"""
FrameVision SPARK.Chroma helper / simple test UI.

Run UI from FrameVision's normal Python:
    python helpers/chroma.py

Generate from the Chroma image-model environment:
    conda layout:  environments/.images_models/python.exe helpers/chroma.py --generate --prompt "..."
    venv layout:   environments/.images_models/Scripts/python.exe helpers/chroma.py --generate --prompt "..."

Download/repair model:
    environments/.images_models/python.exe helpers/chroma.py --download-only
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

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

MODEL_REPO = "SG161222/SPARK.Chroma_v1"
MODEL_SUBDIR = Path("models") / "chroma" / "SPARK.Chroma_v1"
OUTPUT_SUBDIR = Path("output") / "images" / "chroma"
SHARED_ENV_SUBDIR = Path("environments") / ".images_models"
DEDICATED_ENV_SUBDIR = Path("environments") / ".images_models"
SETTINGS_SUBDIR = Path("presets") / "setsave" / "chroma.json"

ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

REQUIRED_FILES = [
    Path("model_index.json"),
    Path("scheduler") / "scheduler_config.json",
    Path("text_encoder") / "model.safetensors.index.json",
    Path("tokenizer") / "tokenizer_config.json",
    Path("transformer") / "diffusion_pytorch_model.safetensors.index.json",
    Path("vae") / "diffusion_pytorch_model.safetensors",
]

DEFAULT_NEGATIVE = (
    "low quality, ugly, unfinished, out of focus, deformed, disfigured, "
    "blurry, smudged, flat colors, text, watermark"
)

PIPELINE = None
PIPELINE_OFFLOAD_CPU = None
PIPELINE_LOCK = Lock()


def framevision_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_dir() -> Path:
    return framevision_root() / MODEL_SUBDIR


def output_dir() -> Path:
    return framevision_root() / OUTPUT_SUBDIR


def settings_path() -> Path:
    return framevision_root() / SETTINGS_SUBDIR


def python_candidates_for_env(env_dir: Path) -> list[Path]:
    return [
        env_dir / "python.exe",              # Windows conda prefix
        env_dir / "Scripts" / "python.exe",  # Windows venv
        env_dir / "bin" / "python",          # Linux/macOS conda or venv
    ]


def shared_env_dir() -> Path:
    return framevision_root() / SHARED_ENV_SUBDIR


def dedicated_env_dir() -> Path:
    return framevision_root() / DEDICATED_ENV_SUBDIR


def first_existing_python(env_dir: Path) -> Path | None:
    for py in python_candidates_for_env(env_dir):
        if py.exists():
            return py
    return None


def env_python() -> Path:
    # Chroma must stay isolated from FrameVision's main environment.
    # Prefer and report only the /environments/.images_models/ Python.
    py = first_existing_python(dedicated_env_dir())
    if py is not None:
        return py
    fallback = python_candidates_for_env(dedicated_env_dir())
    return fallback[0] if os.name == "nt" else fallback[-1]


def env_name() -> str:
    return ".images_models"


def model_ready() -> bool:
    base = model_dir()
    return all((base / rel).exists() for rel in REQUIRED_FILES)


def download_model() -> str:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    base = model_dir()
    base.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(base),
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )

    if not model_ready():
        missing = [str(rel) for rel in REQUIRED_FILES if not (base / rel).exists()]
        raise RuntimeError("Download finished, but required files are missing: " + ", ".join(missing))

    return f"Model ready at {base}"


def load_pipeline(offload_cpu: bool = True):
    global PIPELINE, PIPELINE_OFFLOAD_CPU

    if PIPELINE is not None and PIPELINE_OFFLOAD_CPU == bool(offload_cpu):
        return PIPELINE

    if PIPELINE is not None:
        unload_pipeline()

    if not model_ready():
        raise RuntimeError(
            "SPARK.Chroma model files are missing. Run the Chroma optional installer first, "
            "or run this helper with --download-only from the image environment."
        )

    import torch
    from diffusers import ChromaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Stopping instead of running SPARK.Chroma on CPU.")

    dtype = torch.bfloat16
    pipe = ChromaPipeline.from_pretrained(str(model_dir()), torch_dtype=dtype)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    try:
        if bool(offload_cpu):
            pipe.enable_model_cpu_offload()
            print("[chroma] enabled model CPU offload", flush=True)
        else:
            pipe = pipe.to("cuda")
            print("[chroma] CPU offload disabled; pipeline kept on CUDA", flush=True)
    except Exception as exc:
        print(f"[chroma] CPU offload setup failed, moving pipeline to CUDA: {exc}", flush=True)
        pipe = pipe.to("cuda")

    PIPELINE = pipe
    PIPELINE_OFFLOAD_CPU = bool(offload_cpu)
    return PIPELINE


def unload_pipeline() -> str:
    global PIPELINE, PIPELINE_OFFLOAD_CPU

    with PIPELINE_LOCK:
        if PIPELINE is None:
            return "Model is not loaded."
        del PIPELINE
        PIPELINE = None
        PIPELINE_OFFLOAD_CPU = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return "Model unloaded."


def safe_filename(text: str, limit: int = 80) -> str:
    keep = []
    for ch in text.lower().strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " _-":
            keep.append("_")
    name = "".join(keep).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return (name[:limit].strip("_") or "chroma")


def generate_image(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    max_sequence_length: int,
    output: str | None = None,
    offload_cpu: bool = True,
) -> Path:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if seed < 0:
        seed = random.randint(0, 2_147_483_647)

    started = time.time()
    print(f"[chroma] prompt={prompt}", flush=True)
    print(f"[chroma] size={width}x{height} steps={steps} guidance={guidance_scale} seed={seed}", flush=True)

    with PIPELINE_LOCK:
        import torch
        pipe = load_pipeline(offload_cpu=offload_cpu)
        generator = torch.Generator("cpu").manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=(negative_prompt or "").strip() or None,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            max_sequence_length=int(max_sequence_length),
            num_images_per_prompt=1,
        )
        image = result.images[0]

    if output:
        out_path = Path(output)
        if not out_path.is_absolute():
            out_path = framevision_root() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"chroma_{stamp}_seed{seed}_{safe_filename(prompt, 54)}.png"

    image.save(out_path)
    elapsed = time.time() - started
    print(f"[chroma] saved={out_path}", flush=True)
    print(f"[chroma] elapsed={elapsed:.2f}s", flush=True)
    print("CHROMA_RESULT_JSON=" + json.dumps({"ok": True, "output": str(out_path), "seed": seed, "elapsed": elapsed}), flush=True)
    return out_path


def save_settings(data: dict) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_settings() -> dict:
    path = settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FrameVision SPARK.Chroma helper")
    parser.add_argument("--download-only", action="store_true", help="Download/repair the model and exit")
    parser.add_argument("--generate", action="store_true", help="Generate one image and exit")
    parser.add_argument("--prompt", default="", help="Prompt")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE, help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", default="", help="Optional output file path")
    parser.add_argument("--offload-cpu", dest="offload_cpu", action="store_true", default=None)
    parser.add_argument("--no-offload-cpu", dest="offload_cpu", action="store_false")
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> int:
    if args.download_only:
        print(download_model(), flush=True)
        return 0

    if args.generate:
        generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            max_sequence_length=args.max_sequence_length,
            output=args.output or None,
            offload_cpu=(True if args.offload_cpu is None else bool(args.offload_cpu)),
        )
        return 0

    return run_ui()


def run_ui() -> int:
    try:
        from PySide6.QtCore import Qt, QThread, Signal
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        print("PySide6 is required for the test UI. Run --generate from the image environment for CLI mode.", flush=True)
        print(str(exc), flush=True)
        return 1

    class ProcessThread(QThread):
        line = Signal(str)
        finished_ok = Signal(str)
        failed = Signal(str)

        def __init__(self, cmd: list[str], parent=None):
            super().__init__(parent)
            self.cmd = cmd
            self.output_path = ""

        def run(self):
            try:
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.Popen(
                    self.cmd,
                    cwd=str(framevision_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip()
                    self.line.emit(line)
                    if line.startswith("CHROMA_RESULT_JSON="):
                        try:
                            payload = json.loads(line.split("=", 1)[1])
                            self.output_path = payload.get("output", "")
                        except Exception:
                            pass
                code = proc.wait()
                if code == 0:
                    self.finished_ok.emit(self.output_path)
                else:
                    self.failed.emit(f"Process failed with exit code {code}")
            except Exception as exc:
                self.failed.emit(str(exc))

    class ChromaWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.worker = None
            self.setWindowTitle("FrameVision SPARK.Chroma test helper")
            self.resize(980, 760)
            self.settings = load_settings()

            self.prompt = QTextEdit()
            self.prompt.setPlaceholderText("Prompt")
            self.prompt.setPlainText(self.settings.get("prompt", "a cinematic neon robot walking through a rainy alley, reflections, detailed, 35mm"))

            self.negative = QTextEdit()
            self.negative.setPlaceholderText("Negative prompt")
            self.negative.setPlainText(self.settings.get("negative", DEFAULT_NEGATIVE))
            self.negative.setFixedHeight(80)

            self.width = QSpinBox(); self.width.setRange(512, 1536); self.width.setSingleStep(64); self.width.setValue(int(self.settings.get("width", 1024)))
            self.height = QSpinBox(); self.height.setRange(512, 1536); self.height.setSingleStep(64); self.height.setValue(int(self.settings.get("height", 1024)))
            self.steps = QSpinBox(); self.steps.setRange(8, 60); self.steps.setValue(int(self.settings.get("steps", 35)))
            self.guidance = QDoubleSpinBox(); self.guidance.setRange(1.0, 8.0); self.guidance.setSingleStep(0.1); self.guidance.setValue(float(self.settings.get("guidance", 3.0)))
            self.seed = QSpinBox(); self.seed.setRange(-1, 2_147_483_647); self.seed.setValue(int(self.settings.get("seed", -1)))
            self.max_seq = QSpinBox(); self.max_seq.setRange(128, 512); self.max_seq.setSingleStep(64); self.max_seq.setValue(int(self.settings.get("max_sequence_length", 512)))
            self.offload_cpu = QCheckBox("Enable CPU offload")
            self.offload_cpu.setChecked(bool(self.settings.get("offload_cpu", True)))

            self.status = QLabel(self.status_text())
            self.status.setWordWrap(True)
            self.output_path = QLineEdit(); self.output_path.setReadOnly(True)
            self.preview = QLabel("Preview appears here after generation")
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setMinimumHeight(260)
            self.preview.setStyleSheet("border: 1px solid #555; padding: 8px;")
            self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)

            self.generate_btn = QPushButton("Generate test image")
            self.download_btn = QPushButton("Download / repair model")
            self.open_btn = QPushButton("View results")

            form = QFormLayout()
            form.addRow("Width", self.width)
            form.addRow("Height", self.height)
            form.addRow("Steps", self.steps)
            form.addRow("Guidance", self.guidance)
            form.addRow("Seed (-1 random)", self.seed)
            form.addRow("Max sequence", self.max_seq)
            form.addRow("Memory", self.offload_cpu)

            buttons = QHBoxLayout()
            buttons.addWidget(self.generate_btn)
            buttons.addWidget(self.download_btn)
            buttons.addWidget(self.open_btn)

            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Prompt"))
            layout.addWidget(self.prompt)
            layout.addWidget(QLabel("Negative prompt"))
            layout.addWidget(self.negative)
            layout.addLayout(form)
            layout.addWidget(self.status)
            layout.addLayout(buttons)
            layout.addWidget(QLabel("Output"))
            layout.addWidget(self.output_path)
            layout.addWidget(self.preview)
            layout.addWidget(QLabel("Log"))
            layout.addWidget(self.log)

            self.generate_btn.clicked.connect(self.generate)
            self.download_btn.clicked.connect(self.download)
            self.open_btn.clicked.connect(self.view_results)

        def status_text(self) -> str:
            py = env_python()
            return (
                f"Environment: {'ready' if py.exists() else 'missing'} ({env_name()} -> {py})\n"
                f"Model: {'ready' if model_ready() else 'missing'} ({model_dir()})\n"
                f"Output: {output_dir()}"
            )

        def collect_settings(self) -> dict:
            return {
                "prompt": self.prompt.toPlainText(),
                "negative": self.negative.toPlainText(),
                "width": self.width.value(),
                "height": self.height.value(),
                "steps": self.steps.value(),
                "guidance": self.guidance.value(),
                "seed": self.seed.value(),
                "max_sequence_length": self.max_seq.value(),
                "offload_cpu": bool(self.offload_cpu.isChecked()),
            }

        def set_busy(self, busy: bool) -> None:
            self.generate_btn.setEnabled(not busy)
            self.download_btn.setEnabled(not busy)

        def append_log(self, text: str) -> None:
            self.log.appendPlainText(text)

        def run_process(self, cmd: list[str]) -> None:
            self.set_busy(True)
            self.log.clear()
            self.append_log("Running: " + " ".join(cmd))
            self.worker = ProcessThread(cmd, self)
            self.worker.line.connect(self.append_log)
            self.worker.finished_ok.connect(self.process_finished)
            self.worker.failed.connect(self.process_failed)
            self.worker.start()

        def download(self) -> None:
            py = env_python()
            if not py.exists():
                QMessageBox.warning(self, "Missing environment", "Run install_chroma_framevision.bat first.")
                return
            self.run_process([str(py), str(Path(__file__).resolve()), "--download-only"])

        def generate(self) -> None:
            py = env_python()
            if not py.exists():
                QMessageBox.warning(self, "Missing environment", "Run install_chroma_framevision.bat first.")
                return
            data = self.collect_settings()
            save_settings(data)
            cmd = [
                str(py), str(Path(__file__).resolve()), "--generate",
                "--prompt", data["prompt"],
                "--negative", data["negative"],
                "--width", str(data["width"]),
                "--height", str(data["height"]),
                "--steps", str(data["steps"]),
                "--guidance", str(data["guidance"]),
                "--seed", str(data["seed"]),
                "--max-sequence-length", str(data["max_sequence_length"]),
                "--offload-cpu" if bool(data.get("offload_cpu", True)) else "--no-offload-cpu",
            ]
            self.run_process(cmd)

        def process_finished(self, path: str) -> None:
            self.set_busy(False)
            self.status.setText(self.status_text())
            if path:
                self.output_path.setText(path)
                pix = QPixmap(path)
                if not pix.isNull():
                    self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.append_log("Done.")

        def process_failed(self, msg: str) -> None:
            self.set_busy(False)
            self.status.setText(self.status_text())
            self.append_log("ERROR: " + msg)
            QMessageBox.critical(self, "Chroma failed", msg)

        def view_results(self) -> None:
            _fv_open_results_in_media_explorer(self, output_dir(), preset="images")

        def open_output_folder(self) -> None:
            out = output_dir()
            out.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(out))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            else:
                subprocess.Popen(["xdg-open", str(out)])

    app = QApplication.instance() or QApplication(sys.argv)
    win = ChromaWindow()
    win.show()
    return app.exec()




# Module-level embeddable Chroma UI for FrameVision txt2img.
# The older standalone run_ui() keeps its local window, but txt2img imports this class
# directly so Chroma can live inside the txt2img page like Lens Turbo U4.
try:
    from PySide6.QtCore import Qt as _Qt, QThread as _QThread, Signal as _Signal
    from PySide6.QtGui import QPixmap as _QPixmap
    from PySide6.QtWidgets import (
        QApplication as _QApplication,
        QCheckBox as _QCheckBox,
        QDoubleSpinBox as _QDoubleSpinBox,
        QFormLayout as _QFormLayout,
        QHBoxLayout as _QHBoxLayout,
        QLabel as _QLabel,
        QLineEdit as _QLineEdit,
        QMessageBox as _QMessageBox,
        QPushButton as _QPushButton,
        QPlainTextEdit as _QPlainTextEdit,
        QScrollArea as _QScrollArea,
        QSpinBox as _QSpinBox,
        QTextEdit as _QTextEdit,
        QVBoxLayout as _QVBoxLayout,
        QWidget as _QWidget,
    )

    class ChromaProcessThread(_QThread):
        line = _Signal(str)
        finished_ok = _Signal(str)
        failed = _Signal(str)

        def __init__(self, cmd: list[str], parent=None):
            super().__init__(parent)
            self.cmd = cmd
            self.output_path = ""

        def run(self):
            try:
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.Popen(
                    self.cmd,
                    cwd=str(framevision_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip()
                    self.line.emit(line)
                    if line.startswith("CHROMA_RESULT_JSON="):
                        try:
                            payload = json.loads(line.split("=", 1)[1])
                            self.output_path = payload.get("output", "")
                        except Exception:
                            pass
                code = proc.wait()
                if code == 0:
                    self.finished_ok.emit(self.output_path)
                else:
                    self.failed.emit(f"Process failed with exit code {code}")
            except Exception as exc:
                self.failed.emit(str(exc))

    class ChromaWindow(_QWidget):
        def __init__(self, embedded: bool = False):
            super().__init__()
            self._embedded_in_framevision = bool(embedded)
            self.worker = None
            self.setWindowTitle("FrameVision SPARK.Chroma")
            self.resize(980, 760)
            self.settings = load_settings()

            self.prompt = _QTextEdit()
            self.prompt.setPlaceholderText("Prompt")
            self.prompt.setPlainText(self.settings.get("prompt", "a cinematic neon robot walking through a rainy alley, reflections, detailed, 35mm"))

            self.negative = _QTextEdit()
            self.negative.setPlaceholderText("Negative prompt")
            self.negative.setPlainText(self.settings.get("negative", DEFAULT_NEGATIVE))
            self.negative.setFixedHeight(80)

            self.width = _QSpinBox(); self.width.setRange(512, 1536); self.width.setSingleStep(64); self.width.setValue(int(self.settings.get("width", 1024)))
            self.height = _QSpinBox(); self.height.setRange(512, 1536); self.height.setSingleStep(64); self.height.setValue(int(self.settings.get("height", 1024)))
            self.steps = _QSpinBox(); self.steps.setRange(8, 60); self.steps.setValue(int(self.settings.get("steps", 35)))
            self.guidance = _QDoubleSpinBox(); self.guidance.setRange(1.0, 8.0); self.guidance.setSingleStep(0.1); self.guidance.setValue(float(self.settings.get("guidance", 3.0)))
            self.seed = _QSpinBox(); self.seed.setRange(-1, 2_147_483_647); self.seed.setValue(int(self.settings.get("seed", -1)))
            self.max_seq = _QSpinBox(); self.max_seq.setRange(128, 512); self.max_seq.setSingleStep(64); self.max_seq.setValue(int(self.settings.get("max_sequence_length", 512)))
            self.offload_cpu = _QCheckBox("Enable CPU offload")
            self.offload_cpu.setChecked(bool(self.settings.get("offload_cpu", True)))
            self.use_queue = _QCheckBox("Use FrameVision queue")
            self.use_queue.setChecked(bool(self.settings.get("use_queue", False)))

            self.status = _QLabel(self.status_text())
            self.status.setWordWrap(True)
            self.output_path = _QLineEdit(); self.output_path.setReadOnly(True)
            self.preview = _QLabel("Preview appears here after generation")
            self.preview.setAlignment(_Qt.AlignCenter)
            self.preview.setMinimumHeight(260)
            self.preview.setStyleSheet("border: 1px solid #555; padding: 8px;")
            self.log = _QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)

            self.generate_btn = _QPushButton("Generate Chroma image")
            self.download_btn = _QPushButton("Download / repair model")
            self.open_btn = _QPushButton("View results")

            # Tooltips for the controls that remain visible while queue mode is enabled.
            self.prompt.setToolTip("Describe the image Chroma should create. This prompt is saved with the queued job.")
            self.negative.setToolTip("Optional negative prompt. Use this to tell Chroma what to avoid.")
            self.width.setToolTip("Output image width. Larger sizes use more VRAM and take longer.")
            self.height.setToolTip("Output image height. Larger sizes use more VRAM and take longer.")
            self.steps.setToolTip("Inference steps. Higher values can improve detail but increase render time.")
            self.guidance.setToolTip("Prompt guidance strength. Chroma usually works well around the default value.")
            self.seed.setToolTip("Use -1 for a random seed, or set a fixed seed to repeat a result.")
            self.max_seq.setToolTip("Maximum text sequence length for long prompts.")
            self.offload_cpu.setToolTip("Keeps Chroma safer on VRAM by allowing model CPU offload.")
            self.use_queue.setToolTip("When enabled, Chroma jobs are sent to the main FrameVision queue and preview/logs are hidden here.")
            self.generate_btn.setToolTip("Generate now, or add the current Chroma settings to the FrameVision queue when Use FrameVision queue is enabled.")
            self.download_btn.setToolTip("Download or repair the Chroma model files using the isolated image-model environment.")
            self.open_btn.setToolTip("Open the Chroma output folder in FrameVision Media Explorer.")
            self.output_path.setToolTip("Shows the latest direct output path, or the queued job file when using the queue.")
            self.status.setToolTip("Shows whether the Chroma environment and model files are detected.")

            form = _QFormLayout()
            form.addRow("Width", self.width)
            form.addRow("Height", self.height)
            form.addRow("Steps", self.steps)
            form.addRow("Guidance", self.guidance)
            form.addRow("Seed (-1 random)", self.seed)
            form.addRow("Max sequence", self.max_seq)
            form.addRow("Memory", self.offload_cpu)

            buttons = _QHBoxLayout()
            buttons.addWidget(self.use_queue)
            buttons.addStretch(1)
            buttons.addWidget(self.generate_btn)
            buttons.addWidget(self.download_btn)
            buttons.addWidget(self.open_btn)

            self.bottom_bar = _QWidget(self)
            self.bottom_bar.setLayout(buttons)

            layout = _QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Chroma may be embedded inside TXT2IMG as a full engine page.
            # Give Chroma its own scroll area so parent pages do not need to scroll
            # or clip the Chroma controls/log/preview.
            scroll = _QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
            content = _QWidget()
            content_layout = _QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(6)
            scroll.setWidget(content)
            layout.addWidget(scroll, 1)

            self.preview_label = _QLabel("Preview")
            self.log_label = _QLabel("Log")

            content_layout.addWidget(_QLabel("Prompt"))
            content_layout.addWidget(self.prompt)
            content_layout.addWidget(_QLabel("Negative prompt"))
            content_layout.addWidget(self.negative)
            content_layout.addLayout(form)
            content_layout.addWidget(self.status)
            content_layout.addWidget(_QLabel("Output"))
            content_layout.addWidget(self.output_path)
            content_layout.addWidget(self.preview_label)
            content_layout.addWidget(self.preview)
            content_layout.addWidget(self.log_label)
            content_layout.addWidget(self.log)
            layout.addWidget(self.bottom_bar, 0)

            self.generate_btn.clicked.connect(self.generate)
            self.download_btn.clicked.connect(self.download)
            self.open_btn.clicked.connect(self.view_results)
            self.use_queue.toggled.connect(self._queue_mode_changed)
            self._queue_mode_changed(self.use_queue.isChecked())

        def status_text(self) -> str:
            py = env_python()
            return (
                f"Environment: {'ready' if py.exists() else 'missing'} ({env_name()} -> {py})\n"
                f"Model: {'ready' if model_ready() else 'missing'} ({model_dir()})\n"
                f"Output: {output_dir()}"
            )

        def collect_settings(self) -> dict:
            return {
                "prompt": self.prompt.toPlainText(),
                "negative": self.negative.toPlainText(),
                "width": self.width.value(),
                "height": self.height.value(),
                "steps": self.steps.value(),
                "guidance": self.guidance.value(),
                "seed": self.seed.value(),
                "max_sequence_length": self.max_seq.value(),
                "offload_cpu": bool(self.offload_cpu.isChecked()),
                "use_queue": bool(self.use_queue.isChecked()),
            }

        def _queue_mode_changed(self, checked: bool) -> None:
            checked = bool(checked)
            self.generate_btn.setText("Add to queue" if checked else "Generate Chroma image")
            for widget in (self.preview_label, self.preview, self.log_label, self.log):
                try:
                    widget.setVisible(not checked)
                except Exception:
                    pass
            try:
                data = self.collect_settings()
                save_settings(data)
            except Exception:
                pass

        def set_busy(self, busy: bool) -> None:
            self.generate_btn.setEnabled(not busy)
            self.download_btn.setEnabled(not busy)

        def append_log(self, text: str) -> None:
            self.log.appendPlainText(text)

        def run_process(self, cmd: list[str]) -> None:
            self.set_busy(True)
            self.log.clear()
            self.append_log("Running: " + " ".join(cmd))
            self.worker = ChromaProcessThread(cmd, self)
            self.worker.line.connect(self.append_log)
            self.worker.finished_ok.connect(self.process_finished)
            self.worker.failed.connect(self.process_failed)
            self.worker.start()

        def download(self) -> None:
            py = env_python()
            if not py.exists():
                _QMessageBox.warning(self, "Missing environment", "Run the Chroma optional install first.")
                return
            self.run_process([str(py), str(Path(__file__).resolve()), "--download-only"])

        def generate(self) -> None:
            py = env_python()
            if not py.exists():
                _QMessageBox.warning(self, "Missing environment", "Run the Chroma optional install first.")
                return
            data = self.collect_settings()
            save_settings(data)

            if bool(data.get("use_queue", False)):
                try:
                    from helpers.queue_adapter import enqueue_chroma_generate
                    job_path = enqueue_chroma_generate(data)
                    self.output_path.setText(str(job_path))
                    self.status.setText(self.status_text() + "\nQueued Chroma job: " + str(job_path))
                    _QMessageBox.information(self, "Chroma queued", "Chroma image job added to the FrameVision queue.")
                except Exception as exc:
                    _QMessageBox.critical(self, "Queue failed", "Could not add Chroma to the FrameVision queue:\n" + str(exc))
                return

            cmd = [
                str(py), str(Path(__file__).resolve()), "--generate",
                "--prompt", data["prompt"],
                "--negative", data["negative"],
                "--width", str(data["width"]),
                "--height", str(data["height"]),
                "--steps", str(data["steps"]),
                "--guidance", str(data["guidance"]),
                "--seed", str(data["seed"]),
                "--max-sequence-length", str(data["max_sequence_length"]),
                "--offload-cpu" if bool(data.get("offload_cpu", True)) else "--no-offload-cpu",
            ]
            self.run_process(cmd)

        def process_finished(self, path: str) -> None:
            self.set_busy(False)
            self.status.setText(self.status_text())
            if path:
                self.output_path.setText(path)
                pix = _QPixmap(path)
                if not pix.isNull():
                    self.preview.setPixmap(pix.scaled(self.preview.size(), _Qt.KeepAspectRatio, _Qt.SmoothTransformation))
            self.append_log("Done.")

        def process_failed(self, msg: str) -> None:
            self.set_busy(False)
            self.status.setText(self.status_text())
            self.append_log("ERROR: " + msg)
            _QMessageBox.critical(self, "Chroma failed", msg)

        def view_results(self) -> None:
            _fv_open_results_in_media_explorer(self, output_dir(), preset="images")

        def open_output_folder(self) -> None:
            out = output_dir()
            out.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(out))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            else:
                subprocess.Popen(["xdg-open", str(out)])

except Exception:
    ChromaWindow = None  # type: ignore

def main() -> int:
    args = parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
