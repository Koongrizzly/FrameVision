#!/usr/bin/env python3
"""
FrameVision Mage-Flow-Edit helper UI and persistent backend.

Expected location
-----------------
    <FrameVision root>/helpers/mage_edit.py

Installed runtime
-----------------
    <FrameVision root>/environments/.mage_edit/
    <FrameVision root>/models/mage_edit/Mage-Flow-Edit-Turbo/

Persistent settings
-------------------
    <FrameVision root>/presets/setsave/mage_edit.json

Design
------
- The same file is started with ``--server`` inside the dedicated Mage
  environment. That backend keeps the model loaded between generations.
- PySide6 is imported only by the FrameVision/UI process.
- The tabs and global footer never scroll.
- Each tab gets its own vertical scroll area only when needed.
- Spin boxes and combo boxes ignore the mouse wheel so scrolling a form cannot
  accidentally change a value.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

APP_NAME = "Mage Edit"
SETTINGS_SCHEMA = 1
EVENT_PREFIX = "FVMAGE_EVENT "
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL_DIRNAME = "Mage-Flow-Edit-Turbo"
DEFAULT_NAME_TEMPLATE = "mage_edit_{timestamp}_seed{seed}"


# =============================================================================
# Shared paths and JSON helpers
# =============================================================================

def detect_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    here = Path(__file__).resolve()
    if here.parent.name.lower() == "helpers":
        return here.parent.parent

    cwd = Path.cwd().resolve()
    if (cwd / "presets").exists() and (cwd / "models").exists():
        return cwd

    for parent in (here.parent, *here.parents):
        if (parent / "presets").exists() and (parent / "models").exists():
            return parent
    return here.parent.parent


def environment_python(root: Path) -> Path:
    env = root / "environments" / ".mage_edit"
    if os.name == "nt":
        direct = env / "python.exe"
        scripts = env / "Scripts" / "python.exe"
        return direct if direct.exists() or not scripts.exists() else scripts
    return env / "bin" / "python"


def settings_path(root: Path) -> Path:
    return root / "presets" / "setsave" / "mage_edit.json"


def default_model_path(root: Path) -> Path:
    return root / "models" / "mage_edit" / DEFAULT_MODEL_DIRNAME


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def portable_path(root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def slug(text: str, fallback: str = "edit", max_length: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or fallback)[:max_length]


def prompt_slug(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return slug("_".join(words[:5]), "edit", 64).lower()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 100000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output name for: {path}")


def scan_model_directories(root: Path) -> list[Path]:
    model_root = root / "models" / "mage_edit"
    candidates: list[Path] = []
    if not model_root.exists():
        return candidates

    direct = default_model_path(root)
    if (direct / "model_index.json").exists():
        candidates.append(direct.resolve())

    for path in model_root.iterdir():
        if not path.is_dir() or path.name.lower() in {"repo", "cache"}:
            continue
        if (path / "model_index.json").exists():
            resolved = path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)

    return candidates


# =============================================================================
# Dedicated backend process
# =============================================================================

def worker_event(event: str, **payload: Any) -> None:
    print(
        EVENT_PREFIX + json.dumps({"event": event, **payload}, ensure_ascii=False),
        flush=True,
    )


def worker_log(message: str, level: str = "info") -> None:
    worker_event("log", level=level, message=str(message))


def configure_flash_attention() -> None:
    os.environ["VF_HF_ATTN_IMPL"] = "flash_attention_2"
    from mage_flow.models import mage_flow as mage_model
    from mage_flow.models.modules._attn_backend import set_attn_backend

    field = mage_model.ModelConfig.model_fields.get("attn_type")
    if field is not None:
        field.default = "flash2"
    set_attn_backend("flash2")


def gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


class MagePipelineServer:
    """Persistent Mage pipeline with explicit between-job residency modes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pipeline: Any = None
        self.model_path: Optional[Path] = None
        self.on_cpu = False

    def unload(self, announce: bool = True) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.model.to("cpu")
            except Exception:
                pass
        self.pipeline = None
        self.model_path = None
        self.on_cpu = False
        gpu_cleanup()
        if announce:
            worker_event("unloaded")

    def move_to_cpu(self) -> None:
        if self.pipeline is None or self.on_cpu:
            return
        worker_log("Moving the loaded Mage model to system RAM between jobs.")
        self.pipeline.model.to("cpu")
        self.on_cpu = True
        gpu_cleanup()
        worker_event("model_residency", location="cpu")

    def move_to_gpu(self) -> None:
        if self.pipeline is None or not self.on_cpu:
            return
        worker_event("status", message="Moving Mage model back to the GPU…", phase="loading")
        self.pipeline.model.to("cuda")
        self.on_cpu = False
        worker_event("model_residency", location="cuda")

    def ensure_pipeline(self, requested_model: str) -> Any:
        model_path = Path(requested_model).expanduser().resolve()
        if not (model_path / "model_index.json").exists():
            raise FileNotFoundError(
                f"Selected Mage checkpoint is not a complete local model: {model_path}"
            )

        if self.pipeline is not None and self.model_path == model_path:
            self.move_to_gpu()
            worker_log("Reusing the loaded Mage model.")
            return self.pipeline

        if self.pipeline is not None:
            worker_log("The selected model changed; unloading the previous Mage model.")
            self.unload(announce=False)

        configure_flash_attention()
        worker_event(
            "status",
            message=f"Loading {model_path.name}…",
            phase="loading",
        )
        worker_log(f"Model path: {model_path}")

        from mage_flow import MageFlowPipeline

        started = time.time()
        pipeline = MageFlowPipeline.from_pretrained(str(model_path), device="cuda")
        self.pipeline = pipeline
        self.model_path = model_path
        self.on_cpu = False

        worker_event(
            "model_loaded",
            model=str(model_path),
            seconds=round(time.time() - started, 2),
        )
        return pipeline

    @staticmethod
    def _actual_seeds(seed: int, count: int) -> list[int]:
        if seed < 0:
            rng = random.SystemRandom()
            return [rng.randrange(0, 2**32) for _ in range(count)]
        return [int(seed) + index for index in range(count)]

    @staticmethod
    def _extension(output_format: str) -> str:
        normalized = output_format.lower()
        if normalized in {"jpg", "jpeg"}:
            return ".jpg"
        if normalized == "webp":
            return ".webp"
        return ".png"

    @staticmethod
    def _render_name(template: str, prompt: str, seed: int, index: int) -> str:
        now = datetime.now()
        tokens = {
            "timestamp": now.strftime("%Y%m%d_%H%M%S"),
            "date": now.strftime("%Y%m%d"),
            "time": now.strftime("%H%M%S"),
            "seed": seed,
            "index": index,
            "prompt": prompt_slug(prompt),
        }
        try:
            rendered = template.format(**tokens)
        except Exception as exc:
            raise ValueError(
                "Invalid output-name template. Supported tokens are "
                "{timestamp}, {date}, {time}, {seed}, {index}, and {prompt}."
            ) from exc
        return slug(rendered, f"mage_edit_{tokens['timestamp']}_seed{seed}", 180)

    def generate(self, request: dict[str, Any]) -> None:
        from PIL import PngImagePlugin

        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("The edit instruction is empty.")

        references = [
            Path(str(value)).expanduser().resolve()
            for value in request.get("references", [])
        ]
        if not references:
            raise ValueError("Add at least one reference image.")
        for reference in references:
            if not reference.exists():
                raise FileNotFoundError(f"Reference image not found: {reference}")
            if reference.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported reference image: {reference}")

        count = max(1, int(request.get("count", 1)))
        if len(references) > 3:
            worker_log(
                "More than three references were supplied. Mage accepts them, but the "
                "official model was trained with up to three references.",
                "warning",
            )

        seed_value = int(request.get("seed", -1))
        seeds = self._actual_seeds(seed_value, count)
        prompts = [prompt] * count
        negative_prompt = str(request.get("negative_prompt", " ")) or " "
        negative_prompts = [negative_prompt] * count
        references_per_sample = [[str(path) for path in references] for _ in range(count)]

        steps = max(1, int(request.get("steps", 4)))
        cfg = float(request.get("cfg", 1.0))
        static_shift_value = request.get("static_shift", 6.0)
        static_shift = (
            None if static_shift_value in (None, "", 0, 0.0)
            else float(static_shift_value)
        )

        kwargs: dict[str, Any] = {
            "neg_prompts": negative_prompts,
            "seeds": seeds,
            "steps": steps,
            "cfg": cfg,
            "prompt_template": str(
                request.get("prompt_template", "mage-flow-edit")
            ).strip() or "mage-flow-edit",
            "static_shift": static_shift,
            "vl_cond_long_edge": int(request.get("vl_cond_long_edge", 384)),
            "renormalization": bool(request.get("renormalization", False)),
            "batch_cfg": bool(request.get("batch_cfg", True)),
        }

        size_mode = str(request.get("size_mode", "max_side"))
        if size_mode == "explicit":
            width = max(16, int(request.get("width", 1024)))
            height = max(16, int(request.get("height", 1024)))
            width -= width % 16
            height -= height % 16
            kwargs["widths"] = [width] * count
            kwargs["heights"] = [height] * count
        elif size_mode == "max_side":
            max_size = max(16, int(request.get("max_size", 1024)))
            max_size -= max_size % 16
            kwargs["max_size"] = max_size
        elif size_mode == "source":
            pass
        else:
            raise ValueError(f"Unknown output-size mode: {size_mode}")

        pipeline = self.ensure_pipeline(str(request["model_path"]))
        worker_event(
            "status",
            message=(
                f"Editing {count} image{'s' if count != 1 else ''} "
                f"at {steps} step{'s' if steps != 1 else ''}…"
            ),
            phase="generating",
        )
        worker_event("progress", current=0, total=0, indeterminate=True)

        started = time.time()
        results = pipeline.edit(prompts, references_per_sample, **kwargs)
        generation_seconds = time.time() - started

        output = dict(request.get("output", {}))
        output_folder = Path(
            str(output.get("folder", self.root / "output" / "edits"))
        ).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)

        output_format = str(output.get("format", "png")).lower()
        extension = self._extension(output_format)
        name_template = str(
            output.get("name_template", DEFAULT_NAME_TEMPLATE)
        ).strip() or DEFAULT_NAME_TEMPLATE
        jpeg_quality = max(1, min(100, int(output.get("jpeg_quality", 95))))
        webp_quality = max(1, min(100, int(output.get("webp_quality", 95))))
        embed_metadata = bool(output.get("embed_metadata", True))
        write_sidecar = bool(output.get("write_sidecar", True))

        saved: list[str] = []
        item_metadata: list[dict[str, Any]] = []

        worker_event("status", message="Saving output files…", phase="saving")
        for index, (image, actual_seed) in enumerate(zip(results, seeds), start=1):
            stem = self._render_name(name_template, prompt, actual_seed, index)
            if count > 1 and "{index}" not in name_template and "{seed}" not in name_template:
                stem += f"_{index:02d}"
            output_path = unique_path(output_folder / f"{stem}{extension}")

            metadata = {
                "application": "FrameVision Mage Edit",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "references": [str(path) for path in references],
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "size_mode": size_mode,
                "max_size": request.get("max_size"),
                "width": request.get("width"),
                "height": request.get("height"),
                "vl_cond_long_edge": kwargs["vl_cond_long_edge"],
                "static_shift": static_shift,
                "renormalization": kwargs["renormalization"],
                "batch_cfg": kwargs["batch_cfg"],
                "prompt_template": kwargs["prompt_template"],
                "model_path": str(self.model_path),
            }

            if extension == ".png":
                png_info = None
                if embed_metadata:
                    png_info = PngImagePlugin.PngInfo()
                    png_info.add_text(
                        "FrameVision Mage Edit",
                        json.dumps(metadata, ensure_ascii=False),
                    )
                image.save(output_path, pnginfo=png_info, compress_level=4)
            elif extension == ".jpg":
                save_kwargs: dict[str, Any] = {
                    "quality": jpeg_quality,
                    "optimize": True,
                    "subsampling": 0,
                }
                if embed_metadata:
                    save_kwargs["comment"] = json.dumps(
                        metadata, ensure_ascii=False
                    ).encode("utf-8")[:65000]
                image.convert("RGB").save(output_path, **save_kwargs)
            else:
                image.save(
                    output_path,
                    format="WEBP",
                    quality=webp_quality,
                    method=6,
                )

            if write_sidecar:
                atomic_write_json(output_path.with_suffix(output_path.suffix + ".json"), metadata)

            saved.append(str(output_path))
            item_metadata.append(metadata)
            worker_event(
                "image_saved",
                path=str(output_path),
                seed=actual_seed,
                index=index,
                total=count,
            )

        report_path: Optional[Path] = None
        if bool(request.get("write_generation_report", True)):
            log_folder = Path(
                str(request.get("log_folder", self.root / "logs" / "mage_edit"))
            ).expanduser().resolve()
            log_folder.mkdir(parents=True, exist_ok=True)
            report_path = unique_path(
                log_folder / f"mage_edit_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            atomic_write_json(
                report_path,
                {
                    "ok": True,
                    "generation_seconds": round(generation_seconds, 3),
                    "outputs": saved,
                    "seeds": seeds,
                    "request": request,
                    "metadata": item_metadata,
                },
            )

        residency = str(request.get("residency", "gpu"))
        if residency == "cpu":
            self.move_to_cpu()
        elif residency == "unload":
            worker_log("Unloading Mage after generation.")
            self.unload(announce=False)
            worker_event("model_residency", location="unloaded")
        elif residency != "gpu":
            raise ValueError(f"Unknown residency mode: {residency}")

        worker_event(
            "result",
            outputs=saved,
            seeds=seeds,
            seconds=round(generation_seconds, 2),
            report=str(report_path) if report_path else "",
        )


def server_main(root: Path) -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["VF_HF_ATTN_IMPL"] = "flash_attention_2"

    server = MagePipelineServer(root)
    worker_event("ready", root=str(root), pid=os.getpid())

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "generate":
                server.generate(request)
            elif command == "unload":
                server.unload()
            elif command == "ping":
                worker_event("pong")
            elif command == "quit":
                server.unload(announce=False)
                worker_event("bye")
                return 0
            else:
                raise ValueError(f"Unknown backend command: {command}")
        except Exception as exc:
            worker_event(
                "error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )
    return 0


def parse_early_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--root")
    parsed, _ = parser.parse_known_args()
    return parsed


_EARLY_ARGS = parse_early_args()
if _EARLY_ARGS.server:
    raise SystemExit(server_main(detect_root(_EARLY_ARGS.root)))


# =============================================================================
# PySide6 UI
# =============================================================================

try:
    from PySide6.QtCore import (
        QByteArray,
        QEvent,
        QProcess,
        QProcessEnvironment,
        QSize,
        Qt,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import (
        QAction,
        QCloseEvent,
        QDesktopServices,
        QDragEnterEvent,
        QDropEvent,
        QIcon,
        QKeySequence,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("helpers/mage_edit.py requires PySide6 in the main FrameVision environment") from exc


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ScrollTab(QWidget):
    """A tab whose content scrolls while the tab bar and global footer stay fixed."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(4, 4, 10, 8)
        self.layout.setSpacing(10)
        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll, 1)


class ImagePreviewDialog(QDialog):
    def __init__(self, path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path
        self.setWindowTitle(path.name)
        self.resize(1000, 760)

        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        open_button = buttons.addButton("Open externally", QDialogButtonBox.ActionRole)
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_label.setText(f"Could not preview:\n{path}")
        else:
            self.image_label.setPixmap(pixmap)


class ThumbnailList(QListWidget):
    filesDropped = Signal(list)

    def __init__(self, accept_drops: bool, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setWrapping(True)
        self.setSpacing(8)
        self.setIconSize(QSize(150, 110))
        self.setGridSize(QSize(180, 145))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMinimumHeight(175)
        self.setAcceptDrops(accept_drops)
        self.setDragDropMode(
            QAbstractItemView.DropOnly if accept_drops else QAbstractItemView.NoDragDrop
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        accepted = [
            path for path in paths
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if accepted:
            self.filesDropped.emit(accepted)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class MageEditUI(QWidget):
    generated = Signal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        root: Optional[str | Path] = None,
    ) -> None:
        super().__init__(parent)
        self.root = detect_root(str(root) if root else None)
        self.setObjectName("MageEditUI")
        self.setWindowTitle(APP_NAME)

        self.process: Optional[QProcess] = None
        self.backend_ready = False
        self.pending_request: Optional[dict[str, Any]] = None
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.reference_paths: list[str] = []
        self.output_paths: list[str] = []
        self.busy = False
        self.batch_active = False
        self.batch_total = 0
        self.batch_completed = 0
        self.batch_request: Optional[dict[str, Any]] = None
        self._loading_settings = False
        self._session_log_path: Optional[Path] = None

        self.settings = self.default_settings()
        loaded = read_json(settings_path(self.root), {})
        if isinstance(loaded, dict):
            self.settings = deep_merge(self.settings, loaded)
            self.migrate_legacy_default_paths()

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(350)
        self.save_timer.timeout.connect(self.save_settings)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(180)
        self.preview_timer.timeout.connect(self.update_command_preview)

        self.build_ui()
        self.apply_settings()
        self.connect_persistence()
        self.update_size_controls()
        self.update_command_preview()
        self.append_log("Mage Edit UI ready.")

    # ------------------------------------------------------------------
    # Defaults and persistence
    # ------------------------------------------------------------------
    def default_settings(self) -> dict[str, Any]:
        return {
            "schema": SETTINGS_SCHEMA,
            "generation": {
                "prompt": "Replace the background while preserving the main subject and fine details.",
                "negative_prompt": " ",
                "references": [],
                "steps": 4,
                "cfg": 1.0,
                "seed": -1,
                "count": 1,
                "size_mode": "max_side",
                "max_size": 1024,
                "width": 1024,
                "height": 1024,
                "vl_cond_long_edge": 384,
                "static_shift": 6.0,
                "renormalization": False,
                "batch_cfg": True,
                "prompt_template": "mage-flow-edit",
            },
            "paths": {
                "environment_python": portable_path(
                    self.root, environment_python(self.root)
                ),
                "model_path": portable_path(
                    self.root, default_model_path(self.root)
                ),
                "output_folder": "output/edits",
                "log_folder": "logs/mage_edit",
            },
            "runtime": {
                "residency": "gpu",
                "write_generation_report": True,
                "save_session_log": True,
                "auto_scroll_log": True,
            },
            "output": {
                "name_template": DEFAULT_NAME_TEMPLATE,
                "format": "png",
                "jpeg_quality": 95,
                "webp_quality": 95,
                "embed_metadata": True,
                "write_sidecar": True,
            },
            "ui": {
                "current_tab": 0,
            },
        }

    def migrate_legacy_default_paths(self) -> None:
        """Move only the helper's old defaults; preserve user-chosen custom paths."""
        paths = self.settings.setdefault("paths", {})
        output_value = str(paths.get("output_folder", "")).replace("\\", "/").rstrip("/")
        log_value = str(paths.get("log_folder", "")).replace("\\", "/").rstrip("/")

        if output_value.lower() == "output/edits/mage_edit":
            paths["output_folder"] = "output/edits"
        if log_value.lower() == "output/logs/mage_edit":
            paths["log_folder"] = "logs/mage_edit"

    def schedule_save(self, *_args: Any) -> None:
        if not self._loading_settings:
            self.save_timer.start()
            self.preview_timer.start()

    def save_settings(self) -> None:
        if self._loading_settings:
            return
        try:
            payload = self.collect_settings()
            atomic_write_json(settings_path(self.root), payload)
            self.settings = payload
        except Exception as exc:
            self.append_log(f"Could not save settings: {exc}", "error")

    def collect_settings(self) -> dict[str, Any]:
        return {
            "schema": SETTINGS_SCHEMA,
            "generation": {
                "prompt": self.prompt_edit.toPlainText(),
                "negative_prompt": self.negative_edit.toPlainText(),
                "references": [
                    portable_path(self.root, path)
                    for path in self.reference_paths
                ],
                "steps": self.steps_spin.value(),
                "cfg": self.cfg_spin.value(),
                "seed": self.seed_spin.value(),
                "count": self.count_spin.value(),
                "size_mode": self.size_mode_combo.currentData(),
                "max_size": self.max_size_spin.value(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "vl_cond_long_edge": self.vl_edge_spin.value(),
                "static_shift": self.static_shift_spin.value(),
                "renormalization": self.renormalize_check.isChecked(),
                "batch_cfg": self.batch_cfg_check.isChecked(),
                "prompt_template": self.prompt_template_edit.text().strip()
                or "mage-flow-edit",
            },
            "paths": {
                "environment_python": portable_path(
                    self.root, self.environment_edit.text()
                ),
                "model_path": portable_path(
                    self.root, self.current_model_path()
                ),
                "output_folder": portable_path(
                    self.root, self.output_folder_edit.text()
                ),
                "log_folder": portable_path(
                    self.root, self.log_folder_edit.text()
                ),
            },
            "runtime": {
                "residency": self.residency_combo.currentData(),
                "write_generation_report": self.report_check.isChecked(),
                "save_session_log": self.session_log_check.isChecked(),
                "auto_scroll_log": self.auto_scroll_check.isChecked(),
            },
            "output": {
                "name_template": self.name_template_edit.text().strip()
                or DEFAULT_NAME_TEMPLATE,
                "format": self.format_combo.currentData(),
                "jpeg_quality": self.jpeg_quality_spin.value(),
                "webp_quality": self.webp_quality_spin.value(),
                "embed_metadata": self.metadata_check.isChecked(),
                "write_sidecar": self.sidecar_check.isChecked(),
            },
            "ui": {
                "current_tab": self.tabs.currentIndex(),
            },
        }

    def apply_settings(self) -> None:
        self._loading_settings = True
        try:
            generation = self.settings["generation"]
            paths = self.settings["paths"]
            runtime = self.settings["runtime"]
            output = self.settings["output"]

            self.prompt_edit.setPlainText(str(generation["prompt"]))
            self.negative_edit.setPlainText(str(generation["negative_prompt"]))
            self.steps_spin.setValue(int(generation["steps"]))
            self.cfg_spin.setValue(float(generation["cfg"]))
            self.seed_spin.setValue(int(generation["seed"]))
            self.count_spin.setValue(int(generation["count"]))
            self.select_combo_data(self.size_mode_combo, generation["size_mode"])
            self.max_size_spin.setValue(int(generation["max_size"]))
            self.width_spin.setValue(int(generation["width"]))
            self.height_spin.setValue(int(generation["height"]))
            self.vl_edge_spin.setValue(int(generation["vl_cond_long_edge"]))
            self.static_shift_spin.setValue(float(generation["static_shift"]))
            self.renormalize_check.setChecked(bool(generation["renormalization"]))
            self.batch_cfg_check.setChecked(bool(generation["batch_cfg"]))
            self.prompt_template_edit.setText(str(generation["prompt_template"]))

            references: list[str] = []
            for value in generation.get("references", []):
                path = resolve_path(self.root, value)
                if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    references.append(str(path))
            self.set_references(references)

            self.environment_edit.setText(
                str(resolve_path(self.root, paths["environment_python"]))
            )
            self.refresh_models(preferred=str(resolve_path(self.root, paths["model_path"])))
            self.output_folder_edit.setText(
                str(resolve_path(self.root, paths["output_folder"]))
            )
            self.log_folder_edit.setText(
                str(resolve_path(self.root, paths["log_folder"]))
            )

            self.select_combo_data(self.residency_combo, runtime["residency"])
            self.report_check.setChecked(bool(runtime["write_generation_report"]))
            self.session_log_check.setChecked(bool(runtime["save_session_log"]))
            self.auto_scroll_check.setChecked(bool(runtime["auto_scroll_log"]))

            self.name_template_edit.setText(str(output["name_template"]))
            self.select_combo_data(self.format_combo, output["format"])
            self.jpeg_quality_spin.setValue(int(output["jpeg_quality"]))
            self.webp_quality_spin.setValue(int(output["webp_quality"]))
            self.metadata_check.setChecked(bool(output["embed_metadata"]))
            self.sidecar_check.setChecked(bool(output["write_sidecar"]))

            self.tabs.setCurrentIndex(
                max(0, min(1, int(self.settings.get("ui", {}).get("current_tab", 0))))
            )
        finally:
            self._loading_settings = False

    @staticmethod
    def select_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.generation_tab = ScrollTab(self)
        self.settings_tab = ScrollTab(self)
        self.tabs.addTab(self.generation_tab, "Edit")
        self.tabs.addTab(self.settings_tab, "Settings")
        outer.addWidget(self.tabs, 1)

        self.build_generation_tab()
        self.build_settings_tab()

        self.footer = QFrame(self)
        self.footer.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(8, 6, 8, 6)
        footer_layout.setSpacing(8)

        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(180)
        footer_layout.addWidget(self.status_label, 0)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        footer_layout.addWidget(self.progress, 1)

        self.batch_count_spin = NoWheelSpinBox()
        self.batch_count_spin.setRange(2, 999)
        self.batch_count_spin.setValue(4)
        self.batch_count_spin.setMinimumWidth(64)
        self.batch_count_spin.setToolTip(
            "Number of separate generations. Every batch item forces a fresh random "
            "seed and one output, while the loaded Mage model stays warm."
        )
        footer_layout.addWidget(self.batch_count_spin)

        self.batch_button = QPushButton("Batch")
        self.batch_button.setToolTip(
            "Run the selected number of separate generations. Each run uses seed -1, "
            "so every image gets a different random seed."
        )
        self.batch_button.clicked.connect(self.start_batch)
        footer_layout.addWidget(self.batch_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        footer_layout.addWidget(self.cancel_button)

        self.generate_button = QPushButton("Generate")
        self.generate_button.setDefault(True)
        self.generate_button.clicked.connect(self.generate)
        footer_layout.addWidget(self.generate_button)

        outer.addWidget(self.footer, 0)

    def build_generation_tab(self) -> None:
        instruction_group = QGroupBox("Edit instruction")
        instruction_layout = QVBoxLayout(instruction_group)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(105)
        self.prompt_edit.setPlaceholderText(
            "Describe the required change clearly and directly."
        )
        self.prompt_edit.setToolTip(
            "Instruction sent to Mage-Flow-Edit. Refer to multiple inputs as "
            "image 1, image 2, and image 3."
        )
        instruction_layout.addWidget(self.prompt_edit)

        self.negative_edit = QTextEdit()
        self.negative_edit.setMaximumHeight(64)
        self.negative_edit.setPlaceholderText(
            "Negative prompt; mainly useful when CFG is above 1."
        )
        instruction_layout.addWidget(self.negative_edit)
        self.generation_tab.layout.addWidget(instruction_group)

        references_group = QGroupBox("Reference images")
        references_layout = QVBoxLayout(references_group)

        self.reference_list = ThumbnailList(True)
        self.reference_list.filesDropped.connect(self.add_reference_paths)
        self.reference_list.itemDoubleClicked.connect(self.preview_item)
        self.reference_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.reference_list.customContextMenuRequested.connect(
            self.reference_context_menu
        )
        references_layout.addWidget(self.reference_list)

        reference_buttons = QHBoxLayout()
        self.add_reference_button = QPushButton("Add")
        self.remove_reference_button = QPushButton("Remove")
        self.clear_reference_button = QPushButton("Clear")
        self.move_left_button = QPushButton("Move left")
        self.move_right_button = QPushButton("Move right")
        self.preview_reference_button = QPushButton("Preview")

        self.add_reference_button.clicked.connect(self.choose_references)
        self.remove_reference_button.clicked.connect(self.remove_selected_references)
        self.clear_reference_button.clicked.connect(self.clear_references)
        self.move_left_button.clicked.connect(lambda: self.move_reference(-1))
        self.move_right_button.clicked.connect(lambda: self.move_reference(1))
        self.preview_reference_button.clicked.connect(self.preview_selected_reference)

        for button in (
            self.add_reference_button,
            self.remove_reference_button,
            self.clear_reference_button,
            self.move_left_button,
            self.move_right_button,
            self.preview_reference_button,
        ):
            reference_buttons.addWidget(button)
        reference_buttons.addStretch(1)
        references_layout.addLayout(reference_buttons)

        reference_hint = QLabel(
            "The first image controls the source aspect ratio. Mage was trained "
            "with up to three references, although the runtime accepts more."
        )
        reference_hint.setWordWrap(True)
        references_layout.addWidget(reference_hint)
        self.generation_tab.layout.addWidget(references_group)

        generation_group = QGroupBox("Generation")
        grid = QGridLayout(generation_group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.steps_spin = self.spin(1, 100, 1)
        self.steps_spin.setToolTip("Turbo is designed for 4 steps.")
        self.cfg_spin = self.dspin(0.0, 20.0, 0.1, 2)
        self.cfg_spin.setToolTip("Turbo is designed for CFG 1.0.")
        self.seed_spin = self.spin(-1, 2_147_483_647, 1)
        self.seed_spin.setToolTip("-1 chooses random seeds. Multiple outputs use consecutive seeds.")
        self.count_spin = self.spin(1, 999, 1)
        self.count_spin.setToolTip(
            "Number of edits. Mage packs multiple outputs into one forward per step; "
            "large counts can require substantial VRAM."
        )

        self.add_grid_pair(grid, 0, "Steps", self.steps_spin, "CFG", self.cfg_spin)
        self.add_grid_pair(grid, 1, "Seed", self.seed_spin, "Outputs", self.count_spin)

        self.size_mode_combo = NoWheelComboBox()
        self.size_mode_combo.addItem("Match source aspect · max side", "max_side")
        self.size_mode_combo.addItem("Keep source resolution", "source")
        self.size_mode_combo.addItem("Explicit width and height", "explicit")
        self.size_mode_combo.currentIndexChanged.connect(self.update_size_controls)

        self.max_size_spin = self.spin(512, 2048, 16)
        self.max_size_spin.setToolTip(
            "Longest output edge. The shorter edge follows the first reference."
        )
        self.width_spin = self.spin(512, 2048, 16)
        self.height_spin = self.spin(512, 2048, 16)

        grid.addWidget(QLabel("Output size"), 2, 0)
        grid.addWidget(self.size_mode_combo, 2, 1, 1, 3)
        self.add_grid_pair(grid, 3, "Max side", self.max_size_spin, "", None)
        self.add_grid_pair(grid, 4, "Width", self.width_spin, "Height", self.height_spin)

        self.generation_tab.layout.addWidget(generation_group)

        advanced_group = QGroupBox("Advanced generation")
        advanced = QFormLayout(advanced_group)
        advanced.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.vl_edge_spin = self.spin(0, 4096, 16)
        self.vl_edge_spin.setToolTip(
            "Long-edge cap for the reference image sent to the vision-language "
            "conditioner. 384 matches training; 0 disables the cap."
        )
        advanced.addRow("VL condition edge", self.vl_edge_spin)

        self.static_shift_spin = self.dspin(0.0, 30.0, 0.1, 2)
        self.static_shift_spin.setToolTip(
            "Flow-matching sigma shift. The official default is 6.0. "
            "Set 0 to use the checkpoint scheduler value."
        )
        advanced.addRow("Static shift", self.static_shift_spin)

        self.prompt_template_edit = QLineEdit("mage-flow-edit")
        self.prompt_template_edit.setToolTip(
            "Official edit prompt template. Normally leave this at mage-flow-edit."
        )
        advanced.addRow("Prompt template", self.prompt_template_edit)

        self.batch_cfg_check = QCheckBox(
            "Fuse positive and negative CFG branches into one packed forward"
        )
        self.batch_cfg_check.setToolTip(
            "Official batch_cfg option. It matters only when CFG is above 1."
        )
        advanced.addRow("", self.batch_cfg_check)

        self.renormalize_check = QCheckBox(
            "CFG velocity renormalization"
        )
        self.renormalize_check.setToolTip(
            "Can reduce over-saturation at high CFG values."
        )
        advanced.addRow("", self.renormalize_check)
        self.generation_tab.layout.addWidget(advanced_group)

        results_group = QGroupBox("Outputs")
        results_layout = QVBoxLayout(results_group)
        self.output_list = ThumbnailList(False)
        self.output_list.itemDoubleClicked.connect(self.preview_item)
        results_layout.addWidget(self.output_list)

        output_buttons = QHBoxLayout()
        preview_output_button = QPushButton("Preview")
        preview_output_button.clicked.connect(self.preview_selected_output)
        open_output_button = QPushButton("Open file")
        open_output_button.clicked.connect(self.open_selected_output)
        open_folder_button = QPushButton("Open output folder")
        open_folder_button.clicked.connect(self.open_output_folder)
        clear_outputs_button = QPushButton("Clear thumbnails")
        clear_outputs_button.clicked.connect(self.clear_output_thumbnails)
        for button in (
            preview_output_button,
            open_output_button,
            open_folder_button,
            clear_outputs_button,
        ):
            output_buttons.addWidget(button)
        output_buttons.addStretch(1)
        results_layout.addLayout(output_buttons)
        self.generation_tab.layout.addWidget(results_group)

        self.generation_tab.layout.addStretch(1)

    def build_settings_tab(self) -> None:
        paths_group = QGroupBox("Model and folders")
        paths_form = QFormLayout(paths_group)
        paths_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.model_combo = NoWheelComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        refresh_models_button = QPushButton("Refresh")
        refresh_models_button.clicked.connect(lambda: self.refresh_models())
        browse_model_button = QPushButton("Browse")
        browse_model_button.clicked.connect(self.browse_model)
        model_row_layout.addWidget(self.model_combo, 1)
        model_row_layout.addWidget(refresh_models_button)
        model_row_layout.addWidget(browse_model_button)
        paths_form.addRow("Model", model_row)

        self.environment_edit = QLineEdit()
        environment_browse = QPushButton("Browse")
        environment_row = self.path_row(self.environment_edit, environment_browse)
        environment_browse.clicked.connect(self.browse_environment)
        paths_form.addRow("Environment Python", environment_row)

        self.output_folder_edit = QLineEdit()
        output_browse = QPushButton("Browse")
        output_row = self.path_row(self.output_folder_edit, output_browse)
        output_browse.clicked.connect(
            lambda: self.browse_folder(self.output_folder_edit, "Select output folder")
        )
        paths_form.addRow("Output folder", output_row)

        self.log_folder_edit = QLineEdit()
        log_browse = QPushButton("Browse")
        log_row = self.path_row(self.log_folder_edit, log_browse)
        log_browse.clicked.connect(
            lambda: self.browse_folder(self.log_folder_edit, "Select log folder")
        )
        paths_form.addRow("Log folder", log_row)
        self.settings_tab.layout.addWidget(paths_group)

        runtime_group = QGroupBox("Runtime")
        runtime_form = QFormLayout(runtime_group)
        runtime_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.residency_combo = NoWheelComboBox()
        self.residency_combo.addItem("Keep model on GPU between jobs", "gpu")
        self.residency_combo.addItem("Move model to system RAM between jobs", "cpu")
        self.residency_combo.addItem("Unload model after every job", "unload")
        self.residency_combo.setToolTip(
            "Mage does not expose layer-by-layer Diffusers offload. These are real "
            "between-job residency choices: GPU, system RAM, or complete unload."
        )
        runtime_form.addRow("Model residency", self.residency_combo)

        self.report_check = QCheckBox("Write a JSON generation report")
        runtime_form.addRow("", self.report_check)

        self.session_log_check = QCheckBox("Save UI/backend logs to disk")
        runtime_form.addRow("", self.session_log_check)

        self.auto_scroll_check = QCheckBox("Auto-scroll the log")
        runtime_form.addRow("", self.auto_scroll_check)

        unload_button = QPushButton("Unload model now")
        unload_button.clicked.connect(self.request_unload)
        runtime_form.addRow("", unload_button)
        self.settings_tab.layout.addWidget(runtime_group)

        output_group = QGroupBox("Output files")
        output_form = QFormLayout(output_group)
        output_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.name_template_edit = QLineEdit(DEFAULT_NAME_TEMPLATE)
        self.name_template_edit.setToolTip(
            "Supported tokens: {timestamp}, {date}, {time}, {seed}, {index}, {prompt}."
        )
        output_form.addRow("Name template", self.name_template_edit)

        self.format_combo = NoWheelComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("JPEG", "jpg")
        self.format_combo.addItem("WebP", "webp")
        output_form.addRow("Format", self.format_combo)

        self.jpeg_quality_spin = self.spin(1, 100, 1)
        output_form.addRow("JPEG quality", self.jpeg_quality_spin)
        self.webp_quality_spin = self.spin(1, 100, 1)
        output_form.addRow("WebP quality", self.webp_quality_spin)

        self.metadata_check = QCheckBox("Embed generation metadata when supported")
        output_form.addRow("", self.metadata_check)
        self.sidecar_check = QCheckBox("Write a metadata JSON beside every image")
        output_form.addRow("", self.sidecar_check)
        self.settings_tab.layout.addWidget(output_group)

        command_group = QGroupBox("Command preview")
        command_layout = QVBoxLayout(command_group)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(190)
        self.command_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        command_layout.addWidget(self.command_preview)
        copy_command_button = QPushButton("Copy preview")
        copy_command_button.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self.command_preview.toPlainText()
            )
        )
        command_layout.addWidget(copy_command_button, 0, Qt.AlignRight)
        self.settings_tab.layout.addWidget(command_group)

        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(260)
        self.log_edit.setMaximumBlockCount(10000)
        logs_layout.addWidget(self.log_edit)

        log_buttons = QHBoxLayout()
        clear_log_button = QPushButton("Clear")
        clear_log_button.clicked.connect(self.log_edit.clear)
        copy_log_button = QPushButton("Copy")
        copy_log_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.log_edit.toPlainText())
        )
        open_log_folder_button = QPushButton("Open log folder")
        open_log_folder_button.clicked.connect(self.open_log_folder)
        log_buttons.addWidget(clear_log_button)
        log_buttons.addWidget(copy_log_button)
        log_buttons.addWidget(open_log_folder_button)
        log_buttons.addStretch(1)
        logs_layout.addLayout(log_buttons)
        self.settings_tab.layout.addWidget(logs_group)

        self.settings_tab.layout.addStretch(1)

    @staticmethod
    def path_row(edit: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return container

    @staticmethod
    def add_grid_pair(
        grid: QGridLayout,
        row: int,
        label_a: str,
        widget_a: Optional[QWidget],
        label_b: str,
        widget_b: Optional[QWidget],
    ) -> None:
        if label_a:
            grid.addWidget(QLabel(label_a), row, 0)
        if widget_a is not None:
            grid.addWidget(widget_a, row, 1)
        if label_b:
            grid.addWidget(QLabel(label_b), row, 2)
        if widget_b is not None:
            grid.addWidget(widget_b, row, 3)

    @staticmethod
    def spin(minimum: int, maximum: int, step: int) -> NoWheelSpinBox:
        widget = NoWheelSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setAccelerated(True)
        return widget

    @staticmethod
    def dspin(
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
    ) -> NoWheelDoubleSpinBox:
        widget = NoWheelDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setAccelerated(True)
        return widget

    # ------------------------------------------------------------------
    # Persistence connections
    # ------------------------------------------------------------------
    def connect_persistence(self) -> None:
        text_widgets: Iterable[Any] = (
            self.prompt_edit,
            self.negative_edit,
        )
        for widget in text_widgets:
            widget.textChanged.connect(self.schedule_save)

        line_edits = (
            self.prompt_template_edit,
            self.environment_edit,
            self.output_folder_edit,
            self.log_folder_edit,
            self.name_template_edit,
        )
        for widget in line_edits:
            widget.textChanged.connect(self.schedule_save)

        spin_widgets = (
            self.steps_spin,
            self.cfg_spin,
            self.seed_spin,
            self.count_spin,
            self.max_size_spin,
            self.width_spin,
            self.height_spin,
            self.vl_edge_spin,
            self.static_shift_spin,
            self.jpeg_quality_spin,
            self.webp_quality_spin,
        )
        for widget in spin_widgets:
            widget.valueChanged.connect(self.schedule_save)

        combo_widgets = (
            self.size_mode_combo,
            self.model_combo,
            self.residency_combo,
            self.format_combo,
        )
        for widget in combo_widgets:
            widget.currentIndexChanged.connect(self.schedule_save)
            if widget.isEditable():
                widget.editTextChanged.connect(self.schedule_save)

        checks = (
            self.renormalize_check,
            self.batch_cfg_check,
            self.report_check,
            self.session_log_check,
            self.auto_scroll_check,
            self.metadata_check,
            self.sidecar_check,
        )
        for widget in checks:
            widget.toggled.connect(self.schedule_save)

        self.tabs.currentChanged.connect(self.schedule_save)

    # ------------------------------------------------------------------
    # Reference and output thumbnails
    # ------------------------------------------------------------------
    def make_thumbnail_item(self, path: str) -> QListWidgetItem:
        file_path = Path(path)
        pixmap = QPixmap(str(file_path))
        icon = QIcon()
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                150,
                110,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            icon = QIcon(scaled)
        item = QListWidgetItem(icon, file_path.name)
        item.setData(Qt.UserRole, str(file_path))
        item.setToolTip(str(file_path))
        item.setSizeHint(QSize(180, 145))
        return item

    def set_references(self, paths: Iterable[str]) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for value in paths:
            path = Path(value).expanduser().resolve()
            key = os.path.normcase(str(path))
            if (
                path.exists()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and key not in seen
            ):
                unique.append(str(path))
                seen.add(key)

        self.reference_paths = unique
        self.reference_list.clear()
        for path in self.reference_paths:
            self.reference_list.addItem(self.make_thumbnail_item(path))
        self.schedule_save()

    def add_reference_paths(self, paths: Iterable[str]) -> None:
        combined = list(self.reference_paths)
        combined.extend(str(Path(path).expanduser().resolve()) for path in paths)
        self.set_references(combined)
        self.update_command_preview()

    def choose_references(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add reference images",
            str(self.root),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if files:
            self.add_reference_paths(files)

    def remove_selected_references(self) -> None:
        selected_rows = sorted(
            {self.reference_list.row(item) for item in self.reference_list.selectedItems()},
            reverse=True,
        )
        paths = list(self.reference_paths)
        for row in selected_rows:
            if 0 <= row < len(paths):
                del paths[row]
        self.set_references(paths)

    def clear_references(self) -> None:
        self.set_references([])

    def move_reference(self, delta: int) -> None:
        row = self.reference_list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= len(self.reference_paths):
            return
        paths = list(self.reference_paths)
        paths[row], paths[target] = paths[target], paths[row]
        self.set_references(paths)
        self.reference_list.setCurrentRow(target)

    def reference_context_menu(self, position) -> None:
        item = self.reference_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        preview_action = menu.addAction("Preview")
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(self.reference_list.mapToGlobal(position))
        if chosen == preview_action:
            self.preview_path(Path(str(item.data(Qt.UserRole))))
        elif chosen == remove_action:
            self.reference_list.setCurrentItem(item)
            self.remove_selected_references()

    def preview_item(self, item: QListWidgetItem) -> None:
        self.preview_path(Path(str(item.data(Qt.UserRole))))

    def preview_path(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, APP_NAME, f"File no longer exists:\n{path}")
            return
        dialog = ImagePreviewDialog(path, self)
        dialog.exec()

    def preview_selected_reference(self) -> None:
        item = self.reference_list.currentItem()
        if item:
            self.preview_item(item)

    def preview_selected_output(self) -> None:
        item = self.output_list.currentItem()
        if item:
            self.preview_item(item)

    def open_selected_output(self) -> None:
        item = self.output_list.currentItem()
        if item:
            path = Path(str(item.data(Qt.UserRole)))
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def clear_output_thumbnails(self) -> None:
        self.output_paths.clear()
        self.output_list.clear()

    def add_output_thumbnail(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        self.output_paths.append(resolved)
        item = self.make_thumbnail_item(resolved)
        self.output_list.addItem(item)
        self.output_list.setCurrentItem(item)
        self.generated.emit(resolved)

    # ------------------------------------------------------------------
    # Settings paths and model selection
    # ------------------------------------------------------------------
    def refresh_models(self, preferred: Optional[str] = None) -> None:
        current = preferred or (
            self.current_model_path() if hasattr(self, "model_combo") else ""
        )
        found = scan_model_directories(self.root)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for path in found:
            self.model_combo.addItem(path.name, str(path))

        selected = False
        if current:
            current_normalized = os.path.normcase(str(Path(current).resolve()))
            for index in range(self.model_combo.count()):
                value = self.model_combo.itemData(index)
                if value and os.path.normcase(str(Path(value).resolve())) == current_normalized:
                    self.model_combo.setCurrentIndex(index)
                    selected = True
                    break

        if not selected:
            fallback = current or str(default_model_path(self.root))
            self.model_combo.setEditText(str(fallback))
        self.model_combo.blockSignals(False)
        self.schedule_save()

    def current_model_path(self) -> str:
        index = self.model_combo.currentIndex()
        text = self.model_combo.currentText().strip()
        data = self.model_combo.itemData(index) if index >= 0 else None
        if data and text == self.model_combo.itemText(index):
            return str(resolve_path(self.root, data))
        return str(resolve_path(self.root, text))

    def browse_model(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Mage checkpoint folder",
            self.current_model_path(),
        )
        if folder:
            self.model_combo.setEditText(folder)
            self.schedule_save()

    def browse_environment(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mage environment Python",
            self.environment_edit.text() or str(self.root),
            "Python executable (python.exe python);;All files (*.*)",
        )
        if file_path:
            self.environment_edit.setText(file_path)

    def browse_folder(self, edit: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            title,
            edit.text() or str(self.root),
        )
        if folder:
            edit.setText(folder)

    def open_output_folder(self) -> None:
        folder = resolve_path(self.root, self.output_folder_edit.text())
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_log_folder(self) -> None:
        folder = resolve_path(self.root, self.log_folder_edit.text())
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------
    # Backend command and generation request
    # ------------------------------------------------------------------
    def build_request(self) -> dict[str, Any]:
        size_mode = self.size_mode_combo.currentData()
        return {
            "command": "generate",
            "prompt": self.prompt_edit.toPlainText().strip(),
            "negative_prompt": self.negative_edit.toPlainText(),
            "references": list(self.reference_paths),
            "steps": self.steps_spin.value(),
            "cfg": self.cfg_spin.value(),
            "seed": self.seed_spin.value(),
            "count": self.count_spin.value(),
            "size_mode": size_mode,
            "max_size": self.max_size_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "vl_cond_long_edge": self.vl_edge_spin.value(),
            "static_shift": self.static_shift_spin.value(),
            "renormalization": self.renormalize_check.isChecked(),
            "batch_cfg": self.batch_cfg_check.isChecked(),
            "prompt_template": self.prompt_template_edit.text().strip()
            or "mage-flow-edit",
            "model_path": self.current_model_path(),
            "residency": self.residency_combo.currentData(),
            "log_folder": str(resolve_path(self.root, self.log_folder_edit.text())),
            "write_generation_report": self.report_check.isChecked(),
            "output": {
                "folder": str(resolve_path(self.root, self.output_folder_edit.text())),
                "name_template": self.name_template_edit.text().strip()
                or DEFAULT_NAME_TEMPLATE,
                "format": self.format_combo.currentData(),
                "jpeg_quality": self.jpeg_quality_spin.value(),
                "webp_quality": self.webp_quality_spin.value(),
                "embed_metadata": self.metadata_check.isChecked(),
                "write_sidecar": self.sidecar_check.isChecked(),
            },
        }

    def backend_command(self) -> tuple[str, list[str]]:
        python_path = str(resolve_path(self.root, self.environment_edit.text()))
        arguments = [
            "-u",
            str(Path(__file__).resolve()),
            "--server",
            "--root",
            str(self.root),
        ]
        return python_path, arguments

    def update_command_preview(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        try:
            program, arguments = self.backend_command()
            command_parts = [program, *arguments]
            quoted = " ".join(
                f'"{part}"' if any(char in part for char in " \t&()") else part
                for part in command_parts
            )
            request = self.build_request()
            preview = (
                "Persistent backend:\n"
                f"{quoted}\n\n"
                "JSON request:\n"
                f"{json.dumps(request, indent=2, ensure_ascii=False)}"
            )
            self.command_preview.setPlainText(preview)
        except Exception as exc:
            self.command_preview.setPlainText(f"Preview unavailable: {exc}")

    def validate_request(self, request: dict[str, Any]) -> Optional[str]:
        if not request["prompt"]:
            return "Enter an edit instruction."
        if not request["references"]:
            return "Add at least one reference image."

        python_path = Path(self.backend_command()[0])
        if not python_path.exists():
            return f"Mage environment Python was not found:\n{python_path}"

        model_path = Path(request["model_path"])
        if not (model_path / "model_index.json").exists():
            return (
                "The selected model folder is incomplete or incorrect:\n"
                f"{model_path}"
            )

        for value in request["references"]:
            if not Path(value).exists():
                return f"Reference image was not found:\n{value}"

        if request["count"] > 8:
            answer = QMessageBox.question(
                self,
                APP_NAME,
                f"You selected {request['count']} packed outputs. This can require "
                "a large amount of VRAM.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return "Generation cancelled."
        return None

    def generate(self) -> None:
        if self.busy:
            return

        request = self.build_request()
        validation_error = self.validate_request(request)
        if validation_error:
            if validation_error != "Generation cancelled.":
                QMessageBox.warning(self, APP_NAME, validation_error)
            return

        self.save_settings()
        self.pending_request = request
        self.set_busy(True, "Starting Mage backend…")
        self.ensure_backend()

    def start_batch(self) -> None:
        if self.busy:
            return

        request = self.build_request()
        # Batch means separate jobs, not one packed multi-output request.
        request["count"] = 1
        request["seed"] = -1
        validation_error = self.validate_request(request)
        if validation_error:
            if validation_error != "Generation cancelled.":
                QMessageBox.warning(self, APP_NAME, validation_error)
            return

        self.save_settings()
        self.batch_active = True
        self.batch_total = self.batch_count_spin.value()
        self.batch_completed = 0
        self.batch_request = request
        self.append_log(
            f"Starting batch of {self.batch_total} separate generations with random seeds."
        )
        self.set_busy(True, f"Starting batch 1/{self.batch_total}…")
        self.queue_next_batch_item()

    def queue_next_batch_item(self) -> None:
        if not self.batch_active or self.batch_request is None:
            return
        if self.batch_completed >= self.batch_total:
            self.finish_batch()
            return

        request = dict(self.batch_request)
        request["seed"] = -1
        request["count"] = 1
        self.pending_request = request
        item = self.batch_completed + 1
        self.status_label.setText(f"Batch {item}/{self.batch_total}…")
        self.ensure_backend()

    def finish_batch(self) -> None:
        total = self.batch_total
        self.batch_active = False
        self.batch_request = None
        self.batch_total = 0
        self.batch_completed = 0
        self.set_busy(False, f"Batch finished · {total} images")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.append_log(f"Batch finished: {total} separate images generated.")

    def ensure_backend(self) -> None:
        if (
            self.process is not None
            and self.process.state() != QProcess.NotRunning
            and self.backend_ready
        ):
            self.send_pending_request()
            return

        if self.process is not None:
            self.process.deleteLater()

        self.backend_ready = False
        self.stdout_buffer = ""
        self.stderr_buffer = ""

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.finished.connect(self.process_finished)
        process.errorOccurred.connect(self.process_error)

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("VF_HF_ATTN_IMPL", "flash_attention_2")
        environment.insert(
            "HF_HOME",
            str(self.root / "temp" / "mage_edit" / "cache" / "huggingface"),
        )
        environment.insert(
            "TORCH_HOME",
            str(self.root / "temp" / "mage_edit" / "cache" / "torch"),
        )
        environment.insert(
            "TEMP",
            str(self.root / "temp" / "mage_edit"),
        )
        environment.insert(
            "TMP",
            str(self.root / "temp" / "mage_edit"),
        )
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(self.root))

        program, arguments = self.backend_command()
        self.append_log(f"Starting backend: {program} {' '.join(arguments)}")
        process.start(program, arguments)
        self.process = process

        if not process.waitForStarted(5000):
            self.set_busy(False, "Backend failed to start")
            QMessageBox.critical(
                self,
                APP_NAME,
                f"Could not start the Mage backend:\n{process.errorString()}",
            )

    def send_pending_request(self) -> None:
        if self.pending_request is None or self.process is None:
            return
        payload = json.dumps(self.pending_request, ensure_ascii=False) + "\n"
        self.process.write(QByteArray(payload.encode("utf-8")))
        self.process.waitForBytesWritten(1000)
        self.append_log("Generation request sent.")
        self.pending_request = None
        self.status_label.setText("Generating…")
        self.progress.setRange(0, 0)

    def request_unload(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.append_log("Mage backend is not running.")
            return
        payload = json.dumps({"command": "unload"}) + "\n"
        self.process.write(QByteArray(payload.encode("utf-8")))
        self.process.waitForBytesWritten(1000)

    def cancel_generation(self) -> None:
        if self.process is None:
            return
        self.append_log(
            "Cancelling by terminating the dedicated backend. The model will need "
            "to reload for the next edit.",
            "warning",
        )
        self.pending_request = None
        self.batch_active = False
        self.batch_request = None
        self.batch_total = 0
        self.batch_completed = 0
        self.process.kill()
        self.process.waitForFinished(5000)
        self.set_busy(False, "Cancelled")

    # ------------------------------------------------------------------
    # Backend output handling
    # ------------------------------------------------------------------
    def read_stdout(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self.stdout_buffer += data
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            self.handle_backend_line(line.rstrip("\r"), is_error=False)

    def read_stderr(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        self.stderr_buffer += data
        while "\n" in self.stderr_buffer:
            line, self.stderr_buffer = self.stderr_buffer.split("\n", 1)
            self.handle_backend_line(line.rstrip("\r"), is_error=True)

    def handle_backend_line(self, line: str, is_error: bool) -> None:
        if not line:
            return
        if line.startswith(EVENT_PREFIX):
            try:
                event = json.loads(line[len(EVENT_PREFIX):])
                self.handle_event(event)
            except Exception:
                self.append_log(line, "error")
            return
        self.append_log(line, "error" if is_error else "backend")

    def handle_event(self, event: dict[str, Any]) -> None:
        name = event.get("event")
        if name == "ready":
            self.backend_ready = True
            self.append_log(f"Backend ready (PID {event.get('pid')}).")
            self.send_pending_request()
        elif name == "log":
            self.append_log(
                str(event.get("message", "")),
                str(event.get("level", "info")),
            )
        elif name == "status":
            message = str(event.get("message", "Working…"))
            self.status_label.setText(message)
            if event.get("phase") in {"loading", "generating"}:
                self.progress.setRange(0, 0)
        elif name == "progress":
            if event.get("indeterminate"):
                self.progress.setRange(0, 0)
            else:
                total = max(1, int(event.get("total", 1)))
                current = max(0, int(event.get("current", 0)))
                self.progress.setRange(0, total)
                self.progress.setValue(current)
        elif name == "model_loaded":
            self.append_log(
                f"Model loaded in {event.get('seconds')} seconds."
            )
        elif name == "model_residency":
            self.append_log(
                f"Model residency after the job: {event.get('location')}."
            )
        elif name == "image_saved":
            path = str(event.get("path", ""))
            if path:
                self.add_output_thumbnail(path)
                self.append_log(
                    f"Saved {Path(path).name} · seed {event.get('seed')}"
                )
        elif name == "result":
            seconds = event.get("seconds")
            outputs = event.get("outputs", [])
            self.append_log(
                f"Finished {len(outputs)} output(s) in {seconds} seconds."
            )
            report = event.get("report")
            if report:
                self.append_log(f"Generation report: {report}")
            if self.batch_active:
                self.batch_completed += 1
                self.progress.setRange(0, self.batch_total)
                self.progress.setValue(self.batch_completed)
                if self.batch_completed < self.batch_total:
                    next_item = self.batch_completed + 1
                    self.status_label.setText(
                        f"Batch {next_item}/{self.batch_total}…"
                    )
                    QTimer.singleShot(0, self.queue_next_batch_item)
                else:
                    self.finish_batch()
            else:
                self.set_busy(False, "Finished")
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
        elif name == "error":
            message = str(event.get("message", "Unknown backend error"))
            trace = str(event.get("traceback", ""))
            self.append_log(message, "error")
            if trace:
                self.append_log(trace, "error")
            self.batch_active = False
            self.batch_request = None
            self.batch_total = 0
            self.batch_completed = 0
            self.set_busy(False, "Failed")
            QMessageBox.critical(self, APP_NAME, message)
        elif name == "unloaded":
            self.append_log("Mage model unloaded.")
        elif name == "bye":
            self.backend_ready = False

    def process_finished(self, exit_code: int, _status) -> None:
        self.backend_ready = False
        if self.stdout_buffer.strip():
            self.handle_backend_line(self.stdout_buffer.strip(), False)
        if self.stderr_buffer.strip():
            self.handle_backend_line(self.stderr_buffer.strip(), True)
        self.stdout_buffer = ""
        self.stderr_buffer = ""

        if self.busy:
            self.batch_active = False
            self.batch_request = None
            self.batch_total = 0
            self.batch_completed = 0
            self.set_busy(False, "Backend stopped")
            if exit_code != 0:
                self.append_log(
                    f"Mage backend exited with code {exit_code}.",
                    "error",
                )

    def process_error(self, _error) -> None:
        if self.process is not None:
            self.append_log(
                f"Backend process error: {self.process.errorString()}",
                "error",
            )

    def set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.generate_button.setEnabled(not busy)
        self.batch_button.setEnabled(not busy)
        self.batch_count_spin.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.status_label.setText(status)
        if not busy and status not in {"Finished"}:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    # ------------------------------------------------------------------
    # Log handling and small UI helpers
    # ------------------------------------------------------------------
    def append_log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = level.upper() if level not in {"info", "backend"} else (
            "BACKEND" if level == "backend" else "INFO"
        )
        line = f"[{timestamp}] [{prefix}] {message}"
        if hasattr(self, "log_edit"):
            self.log_edit.appendPlainText(line)
            if self.auto_scroll_check.isChecked():
                bar = self.log_edit.verticalScrollBar()
                bar.setValue(bar.maximum())

        if hasattr(self, "session_log_check") and self.session_log_check.isChecked():
            try:
                folder = resolve_path(self.root, self.log_folder_edit.text())
                folder.mkdir(parents=True, exist_ok=True)
                if self._session_log_path is None:
                    self._session_log_path = (
                        folder / f"mage_edit_ui_{datetime.now():%Y%m%d_%H%M%S}.log"
                    )
                with self._session_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:
                pass

    def update_size_controls(self, *_args: Any) -> None:
        mode = self.size_mode_combo.currentData()
        self.max_size_spin.setEnabled(mode == "max_side")
        self.width_spin.setEnabled(mode == "explicit")
        self.height_spin.setEnabled(mode == "explicit")
        self.schedule_save()

    # ------------------------------------------------------------------
    # Close / standalone
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.save_settings()
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            try:
                payload = json.dumps({"command": "quit"}) + "\n"
                self.process.write(QByteArray(payload.encode("utf-8")))
                self.process.waitForBytesWritten(500)
                if not self.process.waitForFinished(1500):
                    self.process.kill()
            except Exception:
                self.process.kill()
        super().closeEvent(event)


# Common names used by different FrameVision tab loaders.
MageEditWidget = MageEditUI
MageEditTab = MageEditUI


def create_widget(
    parent: Optional[QWidget] = None,
    root: Optional[str | Path] = None,
) -> MageEditUI:
    return MageEditUI(parent=parent, root=root)


def create_tab(
    parent: Optional[QWidget] = None,
    root: Optional[str | Path] = None,
) -> MageEditUI:
    return MageEditUI(parent=parent, root=root)


def build_widget(
    parent: Optional[QWidget] = None,
    root: Optional[str | Path] = None,
) -> MageEditUI:
    return MageEditUI(parent=parent, root=root)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    widget = MageEditUI()
    widget.resize(1080, 850)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
