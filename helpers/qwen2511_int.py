#!/usr/bin/env python3
"""
FrameVision helper UI and persistent backend for Qwen-Image-Edit-2511 INT4.

Expected location:
    <FrameVision root>/helpers/qwen2511_int.py

Installed runtime expected at:
    <root>/environments/.qwen2511_int
    <root>/models/qwen2511_int

Settings:
    <root>/presets/setsave/qwen2511_int.json

The same file is launched with ``--server`` by the UI inside the dedicated
Qwen environment.  The server keeps the pipeline loaded between generations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

APP_NAME = "Qwen 2511 INT4 Image Edit"
SETTINGS_SCHEMA = 1
EVENT_PREFIX = "FVQWEN_EVENT "
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_LORAS = {".safetensors", ".bin", ".pt"}

LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

MODEL_LABELS = {
    "recommended": "Recommended · quality-r64 · 4 steps",
    "fastest": "Fastest / lowest footprint · balanced-r32 · 4 steps",
    "best-low-step": "Best low-step candidate · quality-r128-b15 · 4 steps",
    "fidelity": "Best measured fidelity · mid-r128 · 8 steps",
}


def detect_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve()
    # Intended location is <root>/helpers/qwen2511_int.py.
    if here.parent.name.lower() == "helpers":
        return here.parent.parent
    for parent in (here.parent, *here.parents):
        if (parent / "presets").exists() and (parent / "models").exists():
            return parent
    return here.parent.parent


def environment_python(root: Path) -> Path:
    env = root / "environments" / ".qwen2511_int"
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def model_root(root: Path) -> Path:
    return root / "models" / "qwen2511_int"


def settings_path(root: Path) -> Path:
    return root / "presets" / "setsave" / "qwen2511_int.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def safe_slug_words(prompt: str, count: int = 3) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt)
    selected = words[:count] or ["edit"]
    return "_".join(word.lower()[:32] for word in selected)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique output path for {path}")


def worker_event(event: str, **payload: Any) -> None:
    message = {"event": event, **payload}
    print(EVENT_PREFIX + json.dumps(message, ensure_ascii=False), flush=True)


def worker_log(message: str, level: str = "info") -> None:
    worker_event("log", level=level, message=str(message))


def _load_profiles(root: Path) -> dict[str, Any]:
    path = model_root(root) / "model_profiles.json"
    profiles = read_json(path, {})
    if not isinstance(profiles, dict) or "models" not in profiles:
        raise FileNotFoundError(
            f"Model profile file is missing or invalid: {path}. Run Qwen2511_INT4_install.py first."
        )
    return profiles


def _resolve_model(root: Path, key: str) -> tuple[dict[str, Any], Path]:
    profiles = _load_profiles(root)
    models = profiles.get("models", {})
    if key not in models:
        raise KeyError(f"Unknown model profile: {key}")
    profile = dict(models[key])
    checkpoint = model_root(root) / profile["relative_path"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Selected INT4 checkpoint is not installed: {checkpoint}")
    return profile, checkpoint


def _gpu_memory_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def _exclude_transformer_from_diffusers_offload(pipeline: Any) -> None:
    current = getattr(pipeline, "_exclude_from_cpu_offload", None)
    if current is None:
        pipeline._exclude_from_cpu_offload = ["transformer"]
    elif "transformer" not in current:
        current.append("transformer")


def _apply_offload(
    pipeline: Any,
    transformer: Any,
    mode: str,
    blocks_on_gpu: int,
    pin_memory: bool,
) -> str:
    normalized = str(mode or "auto").lower().replace("_", "-")
    if normalized == "auto":
        normalized = "model-cpu" if _gpu_memory_gib() > 18.0 else "low-vram"

    if normalized in {"model-cpu", "balanced"}:
        pipeline.enable_model_cpu_offload()
        return "model-cpu"

    if normalized in {"low", "low-vram", "sequential"}:
        transformer.set_offload(
            True,
            use_pin_memory=bool(pin_memory),
            num_blocks_on_gpu=max(1, int(blocks_on_gpu)),
        )
        _exclude_transformer_from_diffusers_offload(pipeline)
        pipeline.enable_sequential_cpu_offload()
        return "low-vram"

    if normalized in {"full", "full-gpu", "cuda"}:
        pipeline.to("cuda")
        return "full-gpu"

    raise ValueError(f"Unknown offload mode: {mode}")


def _install_qwen_rope_compatibility(transformer: Any) -> None:
    """Bridge Nunchaku 1.2.x to the newer Diffusers Qwen RoPE API.

    Newer Qwen pipelines no longer pass ``txt_seq_lens`` into Nunchaku's
    transformer. Nunchaku still forwards that value to ``QwenEmbedRope``.
    Recover the exact encoded text length from ``encoder_hidden_states`` and
    translate the legacy positional RoPE call to ``max_txt_seq_len``.
    """
    import inspect
    import types

    if getattr(transformer, "_framevision_rope_compat", False):
        return

    original_transformer_forward = transformer.forward
    transformer_signature = inspect.signature(original_transformer_forward)

    def compatible_transformer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = transformer_signature.bind_partial(*args, **kwargs)
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states")
        txt_seq_lens = bound.arguments.get("txt_seq_lens")

        if txt_seq_lens is None and encoder_hidden_states is not None:
            text_length = int(encoder_hidden_states.shape[1])
            batch_size = int(encoder_hidden_states.shape[0])
            # Nunchaku accepts a per-sample list. For a padded batch the RoPE
            # table must cover the complete encoded sequence length.
            bound.arguments["txt_seq_lens"] = [text_length] * batch_size

        return original_transformer_forward(*bound.args, **bound.kwargs)

    transformer.forward = types.MethodType(compatible_transformer_forward, transformer)
    transformer._framevision_rope_compat = True

    pos_embed = getattr(transformer, "pos_embed", None)
    if pos_embed is None or getattr(pos_embed, "_framevision_rope_compat", False):
        worker_log("Installed transformer text-length compatibility bridge.")
        return

    original_rope_forward = pos_embed.forward
    rope_signature = inspect.signature(original_rope_forward)
    rope_parameters = rope_signature.parameters

    # An older Diffusers build still accepts txt_seq_lens directly, so only the
    # transformer-side recovery above is required.
    if "txt_seq_lens" in rope_parameters:
        pos_embed._framevision_rope_compat = True
        worker_log(
            f"Installed Nunchaku text-length bridge; legacy RoPE API: {rope_signature}"
        )
        return

    def _max_text_length(value: Any) -> int:
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            values = []
            for item in value:
                if isinstance(item, (list, tuple)):
                    values.extend(item)
                else:
                    values.append(item)
            return max(int(item) for item in values) if values else 0
        return int(value)

    def compatible_rope_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Nunchaku calls: pos_embed(img_shapes, txt_seq_lens, device=...).
        if len(args) >= 2:
            img_shapes = args[0]
            txt_seq_lens = args[1]
            remaining = args[2:]
            call_kwargs = dict(kwargs)
            max_len = _max_text_length(txt_seq_lens)
            if max_len <= 0:
                raise ValueError(
                    "Unable to determine the encoded text sequence length for Qwen RoPE."
                )
            call_kwargs["max_txt_seq_len"] = max_len
            return original_rope_forward(img_shapes, *remaining, **call_kwargs)
        return original_rope_forward(*args, **kwargs)

    pos_embed.forward = types.MethodType(compatible_rope_forward, pos_embed)
    pos_embed._framevision_rope_compat = True
    worker_log(
        f"Installed Nunchaku/Diffusers Qwen RoPE bridge: {rope_signature}"
    )

def _normalise_lora_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not row.get("enabled", True):
            continue
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"LoRA file not found: {path}")
        strength = float(row.get("strength", 1.0))
        result.append(
            {
                "path": str(path),
                "strength": strength,
                "adapter": f"fv_lora_{index}",
                "mtime": path.stat().st_mtime_ns,
            }
        )
    return result


def _load_loras(pipeline: Any, transformer: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    worker_log(
        "Loading custom LoRAs. They must use a Qwen-Image Diffusers-compatible key layout; "
        "the selected Lightning acceleration is already baked into the INT4 checkpoint."
    )

    # Current Nunchaku Qwen transformers do not expose the native FLUX LoRA API.
    # Diffusers' Qwen LoRA loader is therefore the primary integration path.
    if hasattr(pipeline, "load_lora_weights"):
        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        for row in rows:
            path = Path(row["path"])
            adapter = row["adapter"]
            worker_log(f"Loading LoRA: {path.name} at strength {row['strength']:.3f}")
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter,
            )
            adapter_names.append(adapter)
            adapter_weights.append(float(row["strength"]))

        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        elif len(adapter_names) == 1 and hasattr(transformer, "set_lora_strength"):
            transformer.set_lora_strength(adapter_weights[0])
        elif any(abs(weight - 1.0) > 1e-6 for weight in adapter_weights):
            raise RuntimeError(
                "The installed Diffusers/Nunchaku combination loaded the LoRA but cannot set adapter strengths."
            )
        return

    # Future Nunchaku Qwen releases may add the same single-LoRA methods used by FLUX.
    if len(rows) == 1 and hasattr(transformer, "update_lora_params"):
        transformer.update_lora_params(rows[0]["path"])
        if hasattr(transformer, "set_lora_strength"):
            transformer.set_lora_strength(float(rows[0]["strength"]))
        return

    raise RuntimeError(
        "Custom LoRAs are not supported by this installed Nunchaku Qwen build. "
        "Update Nunchaku/Diffusers or disable the LoRA rows."
    )


class PipelineServer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pipeline: Any = None
        self.transformer: Any = None
        self.cache_key: Optional[str] = None
        self.runtime_info: dict[str, Any] = {}

    def unload(self) -> None:
        if self.pipeline is not None:
            worker_log("Unloading Qwen pipeline from memory.")
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
        self.pipeline = None
        self.transformer = None
        self.cache_key = None
        self.runtime_info = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        worker_event("unloaded")

    def _make_cache_key(self, request: dict[str, Any], loras: list[dict[str, Any]]) -> str:
        payload = {
            "model": request["model"],
            "offload": request["offload"],
            "blocks_on_gpu": int(request.get("blocks_on_gpu", 1)),
            "pin_memory": bool(request.get("pin_memory", False)),
            "loras": loras,
        }
        return json.dumps(payload, sort_keys=True)

    def ensure_pipeline(self, request: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel

        loras = _normalise_lora_rows(request.get("loras", []))
        cache_key = self._make_cache_key(request, loras)
        if self.pipeline is not None and cache_key == self.cache_key:
            worker_log("Reusing the loaded model.")
            return self.pipeline, dict(self.runtime_info)

        if self.pipeline is not None:
            self.unload()

        profile, checkpoint = _resolve_model(self.root, request["model"])
        base_dir = model_root(self.root) / "base" / "Qwen-Image-Edit-2511"
        if not (base_dir / "model_index.json").exists():
            raise FileNotFoundError(f"Qwen shared pipeline assets are missing: {base_dir}")

        worker_event("status", message=f"Loading {profile.get('label', request['model'])}")
        worker_log(f"Checkpoint: {checkpoint}")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16
        )
        _install_qwen_rope_compatibility(transformer)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER_CONFIG)
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(base_dir),
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipeline.set_progress_bar_config(disable=True)

        _load_loras(pipeline, transformer, loras)
        applied_offload = _apply_offload(
            pipeline,
            transformer,
            request.get("offload", "auto"),
            int(request.get("blocks_on_gpu", 1)),
            bool(request.get("pin_memory", False)),
        )

        self.pipeline = pipeline
        self.transformer = transformer
        self.cache_key = cache_key
        self.runtime_info = {
            "model": request["model"],
            "model_label": profile.get("label", request["model"]),
            "checkpoint": str(checkpoint),
            "profile_steps": int(profile.get("steps", 4)),
            "offload": applied_offload,
            "lora_count": len(loras),
        }
        worker_event("model_loaded", **self.runtime_info)
        return pipeline, dict(self.runtime_info)

    def generate(self, request: dict[str, Any]) -> None:
        import torch
        from PIL import Image, PngImagePlugin

        references = [Path(value).expanduser().resolve() for value in request.get("references", [])]
        if not references:
            raise ValueError("At least one reference image is required.")
        for path in references:
            if not path.exists():
                raise FileNotFoundError(f"Reference image not found: {path}")

        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("The edit prompt is empty.")

        pipeline, runtime = self.ensure_pipeline(request)
        images = []
        for path in references:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))

        width = request.get("width")
        height = request.get("height")
        steps = int(request.get("steps") or runtime["profile_steps"])
        count = max(1, min(8, int(request.get("count", 1))))
        seed_value = int(request.get("seed", -1))
        if seed_value < 0:
            seed_value = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)

        output = request.get("output", {})
        output_folder = Path(str(output.get("folder", self.root / "output" / "edits"))).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        output_format = str(output.get("format", "png")).lower()
        extension = ".jpg" if output_format in {"jpg", "jpeg"} else ".png"
        auto_name = bool(output.get("auto_name", True))
        manual_name = str(output.get("manual_name", "")).strip()
        jpeg_quality = max(70, min(100, int(output.get("jpeg_quality", 95))))

        true_cfg = float(request.get("true_cfg_scale", 1.0))
        negative_prompt = str(request.get("negative_prompt", ""))
        max_sequence_length = max(128, min(1024, int(request.get("max_sequence_length", 512))))
        cleanup_between = bool(request.get("cleanup_between", True))

        outputs: list[str] = []
        seeds: list[int] = []
        batch_started = time.time()

        for image_index in range(count):
            actual_seed = seed_value + image_index
            seeds.append(actual_seed)
            worker_event(
                "status",
                message=f"Generating image {image_index + 1}/{count} · seed {actual_seed}",
            )

            def callback(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]):
                worker_event(
                    "progress",
                    current=step_index + 1,
                    total=steps,
                    image=image_index + 1,
                    image_total=count,
                )
                return callback_kwargs

            kwargs: dict[str, Any] = {
                "image": images if len(images) > 1 else images[0],
                "prompt": prompt,
                "num_inference_steps": steps,
                "true_cfg_scale": true_cfg,
                "generator": torch.Generator(device="cpu").manual_seed(actual_seed),
                "max_sequence_length": max_sequence_length,
                "callback_on_step_end": callback,
            }
            if true_cfg > 1.0:
                kwargs["negative_prompt"] = negative_prompt or " "
            if width:
                kwargs["width"] = int(width)
            if height:
                kwargs["height"] = int(height)

            started = time.time()
            result = pipeline(**kwargs)
            generated = result.images[0]

            now = datetime.now()
            if auto_name or not manual_name:
                stem = (
                    f"qwen2511_{safe_slug_words(prompt)}_{actual_seed}_"
                    f"{now:%Y%m%d}_{now:%H%M%S}"
                )
            else:
                stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(manual_name).stem).strip("._")
                stem = stem or "qwen2511_edit"
                if count > 1:
                    stem += f"_{image_index + 1:02d}"

            output_path = unique_path(output_folder / f"{stem}{extension}")
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": actual_seed,
                "steps": steps,
                "true_cfg_scale": true_cfg,
                "model": runtime["model"],
                "checkpoint": runtime["checkpoint"],
                "offload": runtime["offload"],
                "references": [str(path) for path in references],
                "loras": request.get("loras", []),
            }
            if extension == ".png":
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text("FrameVision Qwen2511", json.dumps(metadata, ensure_ascii=False))
                generated.save(output_path, pnginfo=png_info, compress_level=4)
            else:
                generated.convert("RGB").save(
                    output_path,
                    quality=jpeg_quality,
                    optimize=True,
                    subsampling=0,
                    comment=json.dumps(metadata, ensure_ascii=False).encode("utf-8")[:65000],
                )

            elapsed = time.time() - started
            outputs.append(str(output_path))
            worker_event(
                "image_saved",
                path=str(output_path),
                seed=actual_seed,
                seconds=round(elapsed, 2),
            )

            if cleanup_between and torch.cuda.is_available():
                torch.cuda.empty_cache()

        worker_event(
            "result",
            outputs=outputs,
            seeds=seeds,
            seconds=round(time.time() - batch_started, 2),
            runtime=runtime,
        )


def server_main(root: Path) -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    server = PipelineServer(root)
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
                server.unload()
                worker_event("bye")
                return 0
            else:
                raise ValueError(f"Unknown server command: {command}")
        except Exception as exc:
            worker_event(
                "error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )
    return 0


def queue_job_main(root: Path, request_path: str) -> int:
    """Run one queued edit without importing the PySide6 UI.

    FrameVision's queue worker launches this mode inside the dedicated
    ``.qwen2511_int`` environment.  The request is a normal ``build_request``
    payload, so queued and direct generations use exactly the same backend.
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    server = PipelineServer(root)
    try:
        path = Path(request_path).expanduser().resolve()
        request = read_json(path, None)
        if not isinstance(request, dict):
            raise ValueError(f"Queued request is missing or invalid: {path}")
        request["command"] = "generate"
        worker_event("ready", root=str(root), pid=os.getpid(), queue_job=True)
        server.generate(request)
        return 0
    except Exception as exc:
        worker_event("error", message=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        try:
            server.unload()
        except Exception:
            pass


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--queue-job")
    parser.add_argument("--root")
    args, _ = parser.parse_known_args()
    return args


_WORKER_ARGS = parse_worker_args()
if _WORKER_ARGS.server:
    raise SystemExit(server_main(detect_root(_WORKER_ARGS.root)))
if _WORKER_ARGS.queue_job:
    raise SystemExit(queue_job_main(detect_root(_WORKER_ARGS.root), _WORKER_ARGS.queue_job))


# UI imports are intentionally below the worker dispatch.  The dedicated model
# environment does not need PySide6 to run the backend.
from PySide6.QtCore import (  # noqa: E402
    QByteArray,
    QEvent,
    QObject,
    QProcess,
    QProcessEnvironment,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (  # noqa: E402
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class DropListWidget(QListWidget):
    filesDropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self.filesDropped.emit(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)


class ImagePreviewDialog(QDialog):
    def __init__(self, path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(path.name)
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(path))
        label.setPixmap(pixmap)
        label.resize(pixmap.size())
        scroll.setWidget(label)
        layout.addWidget(scroll)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


class Qwen2511IntWidget(QWidget):
    generationRequested = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, root: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.root = (root or detect_root()).resolve()
        self.settings_file = settings_path(self.root)
        self.model_dir = model_root(self.root)
        self.env_python = environment_python(self.root)
        self.process: Optional[QProcess] = None
        self.server_ready = False
        self.pending_request: Optional[dict[str, Any]] = None
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.running = False
        self.last_output: Optional[Path] = None
        self.reference_paths: list[str] = []
        self.lora_rows: list[dict[str, Any]] = []
        self._loading_settings = True

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(350)
        self.save_timer.timeout.connect(self.save_settings)

        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_backend)
        self._apply_style()
        self.refresh_models()
        self.load_settings()
        self._connect_autosave()
        self._loading_settings = False
        self.update_install_status()
        self.update_steps_from_model()
        self.update_filename_preview()
        self.update_generate_enabled()

    # ------------------------------ UI construction ------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("PageTitle")
        subtitle = QLabel("Fast multi-reference editing with the installed Nunchaku INT4 profiles")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.install_badge = QLabel("Checking installation…")
        self.install_badge.setObjectName("StatusBadge")
        header.addWidget(self.install_badge)
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_generation_tab(), "Generation")
        self.tabs.addTab(self._build_settings_tab(), "Model, Output & Logs")
        outer.addWidget(self.tabs, 1)

        # Sticky bottom action bar. It remains visible while either tab scrolls.
        action_frame = QFrame()
        action_frame.setObjectName("ActionBar")
        action = QHBoxLayout(action_frame)
        action.setContentsMargins(10, 8, 10, 8)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(220)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")
        self.progress.setMinimumWidth(260)
        self.open_output_button = QPushButton("Open last output")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_last_output)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.generate_button = QPushButton("Generate edit")
        self.generate_button.setObjectName("GenerateButton")
        self.generate_button.clicked.connect(self.generate)
        action.addWidget(self.status_label, 1)
        action.addWidget(self.progress, 2)
        action.addWidget(self.open_output_button)
        action.addWidget(self.cancel_button)
        action.addWidget(self.generate_button)
        outer.addWidget(action_frame)

    def _scroll_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _build_generation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 16)
        layout.setSpacing(12)

        prompt_group = QGroupBox("Edit instruction")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Describe the change. With multiple references, refer to them as image 1, image 2, and so on."
        )
        self.prompt_edit.setMinimumHeight(120)
        self.prompt_edit.setToolTip(
            "Qwen 2511 follows direct editing instructions well. State what should change and what must remain unchanged."
        )
        prompt_layout.addWidget(self.prompt_edit)
        negative_row = QHBoxLayout()
        negative_label = QLabel("Negative prompt")
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText("Optional; only used when True CFG is above 1.0")
        self.negative_edit.setToolTip(
            "Lightning checkpoints normally use True CFG 1.0, where a negative prompt is ignored. "
            "Raise True CFG only when a specific edit needs stronger prompt adherence."
        )
        negative_row.addWidget(negative_label)
        negative_row.addWidget(self.negative_edit, 1)
        prompt_layout.addLayout(negative_row)
        layout.addWidget(prompt_group)

        references_group = QGroupBox("Reference images")
        references_layout = QVBoxLayout(references_group)
        hint = QLabel(
            "Order matters: the prompt can identify these as image 1, image 2, etc. Drag thumbnails to reorder them."
        )
        hint.setObjectName("Hint")
        references_layout.addWidget(hint)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_list = DropListWidget()
        self.reference_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.reference_list.setIconSize(QSize(150, 110))
        self.reference_list.setGridSize(QSize(174, 152))
        self.reference_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.reference_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.reference_list.setMinimumHeight(250)
        self.reference_list.filesDropped.connect(self.add_reference_paths)
        self.reference_list.currentItemChanged.connect(self.show_reference_preview)
        self.reference_list.itemDoubleClicked.connect(self.open_reference_dialog)
        self.reference_list.model().rowsMoved.connect(lambda *_: self.references_reordered())
        left_layout.addWidget(self.reference_list)
        ref_buttons = QHBoxLayout()
        add_ref = QPushButton("Add images")
        add_ref.clicked.connect(self.browse_references)
        paste_ref = QPushButton("Paste")
        paste_ref.setToolTip("Save the clipboard image under FrameVision's temp folder and add it as a reference.")
        paste_ref.clicked.connect(self.paste_reference)
        remove_ref = QPushButton("Remove")
        remove_ref.clicked.connect(self.remove_reference)
        clear_ref = QPushButton("Clear")
        clear_ref.clicked.connect(self.clear_references)
        ref_buttons.addWidget(add_ref)
        ref_buttons.addWidget(paste_ref)
        ref_buttons.addStretch(1)
        ref_buttons.addWidget(remove_ref)
        ref_buttons.addWidget(clear_ref)
        left_layout.addLayout(ref_buttons)

        right = QFrame()
        right.setObjectName("PreviewFrame")
        right_layout = QVBoxLayout(right)
        self.reference_preview = QLabel("Select a reference image")
        self.reference_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_preview.setMinimumSize(310, 250)
        self.reference_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.reference_preview.setObjectName("ImagePreview")
        self.reference_info = QLabel("")
        self.reference_info.setWordWrap(True)
        self.reference_info.setObjectName("Hint")
        right_layout.addWidget(self.reference_preview, 1)
        right_layout.addWidget(self.reference_info)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([620, 380])
        references_layout.addWidget(splitter)
        layout.addWidget(references_group)

        lora_group = QGroupBox("Custom LoRAs")
        lora_layout = QVBoxLayout(lora_group)
        lora_note = QLabel(
            "Optional. Use LoRAs trained for Qwen-Image/Edit in Diffusers format. The 4/8-step Lightning LoRA is already baked into these INT4 checkpoints."
        )
        lora_note.setWordWrap(True)
        lora_note.setObjectName("Hint")
        lora_note.setToolTip(
            "Nunchaku's current Qwen transformer does not expose the native FLUX LoRA API. "
            "This helper uses Diffusers' Qwen LoRA loader and reports a clear error when a LoRA key layout is incompatible."
        )
        lora_layout.addWidget(lora_note)
        self.lora_table = QTableWidget(0, 4)
        self.lora_table.setHorizontalHeaderLabels(["Use", "LoRA file", "Strength", "Path"])
        self.lora_table.verticalHeader().setVisible(False)
        self.lora_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.lora_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lora_table.setMinimumHeight(170)
        self.lora_table.horizontalHeader().setStretchLastSection(True)
        self.lora_table.itemChanged.connect(self.schedule_save)
        self.lora_table.setColumnWidth(0, 52)
        self.lora_table.setColumnWidth(1, 230)
        self.lora_table.setColumnWidth(2, 100)
        lora_layout.addWidget(self.lora_table)
        lora_buttons = QHBoxLayout()
        add_lora = QPushButton("Add LoRA")
        add_lora.clicked.connect(self.browse_loras)
        remove_lora = QPushButton("Remove")
        remove_lora.clicked.connect(self.remove_lora)
        clear_lora = QPushButton("Clear")
        clear_lora.clicked.connect(self.clear_loras)
        lora_buttons.addWidget(add_lora)
        lora_buttons.addWidget(remove_lora)
        lora_buttons.addWidget(clear_lora)
        lora_buttons.addStretch(1)
        lora_layout.addLayout(lora_buttons)
        layout.addWidget(lora_group)

        image_group = QGroupBox("Generation controls")
        grid = QGridLayout(image_group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.resolution_combo = NoWheelComboBox()
        self.resolution_combo.addItem("Match first reference · max side 1024", "source")
        self.resolution_combo.addItem("Square · 1024 × 1024", "1024x1024")
        self.resolution_combo.addItem("Landscape · 1344 × 768", "1344x768")
        self.resolution_combo.addItem("Portrait · 768 × 1344", "768x1344")
        self.resolution_combo.addItem("HD landscape · 1280 × 720", "1280x720")
        self.resolution_combo.addItem("HD portrait · 720 × 1280", "720x1280")
        self.resolution_combo.addItem("Custom", "custom")
        self.resolution_combo.setToolTip(
            "Matching the first reference preserves its composition. Dimensions are rounded to a multiple of 16."
        )
        self.resolution_combo.currentIndexChanged.connect(self.resolution_changed)

        self.width_spin = NoWheelSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setSingleStep(16)
        self.width_spin.setValue(1024)
        self.height_spin = NoWheelSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setSingleStep(16)
        self.height_spin.setValue(1024)

        self.source_max_spin = NoWheelSpinBox()
        self.source_max_spin.setRange(512, 2048)
        self.source_max_spin.setSingleStep(64)
        self.source_max_spin.setValue(1024)
        self.source_max_spin.setToolTip(
            "The first reference is resized proportionally so its longest side is this value. "
            "Larger images need substantially more VRAM and time."
        )

        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("Use -1 for a new random seed. A fixed seed helps compare models and settings.")
        random_seed = QPushButton("Random")
        random_seed.clicked.connect(lambda: self.seed_spin.setValue(-1))

        self.count_spin = NoWheelSpinBox()
        self.count_spin.setRange(1, 8)
        self.count_spin.setValue(1)
        self.count_spin.setToolTip(
            "Outputs are generated sequentially to avoid multiplying VRAM use. Seeds increase by one for each result."
        )

        self.cfg_spin = NoWheelDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 8.0)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setSingleStep(0.25)
        self.cfg_spin.setValue(1.0)
        self.cfg_spin.setToolTip(
            "The baked Lightning checkpoints are designed for True CFG 1.0. Higher values can improve instruction adherence, "
            "but enable an extra guidance pass and can reduce quality or speed."
        )

        self.steps_override = QCheckBox("Override profile steps")
        self.steps_override.setToolTip(
            "Normally keep this off: each downloaded Lightning checkpoint is trained for exactly 4 or 8 steps."
        )
        self.steps_override.toggled.connect(self.steps_override_changed)
        self.steps_spin = NoWheelSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(4)
        self.steps_spin.setEnabled(False)

        self.max_sequence_spin = NoWheelSpinBox()
        self.max_sequence_spin.setRange(128, 1024)
        self.max_sequence_spin.setSingleStep(64)
        self.max_sequence_spin.setValue(512)
        self.max_sequence_spin.setToolTip(
            "Maximum text-token length used by the Qwen processor. 512 is enough for normal edit instructions; "
            "raising it mainly helps unusually long prompts and consumes more memory."
        )

        grid.addWidget(QLabel("Resolution"), 0, 0)
        grid.addWidget(self.resolution_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("Width"), 1, 0)
        grid.addWidget(self.width_spin, 1, 1)
        grid.addWidget(QLabel("Height"), 1, 2)
        grid.addWidget(self.height_spin, 1, 3)
        grid.addWidget(QLabel("Source max side"), 2, 0)
        grid.addWidget(self.source_max_spin, 2, 1)
        grid.addWidget(QLabel("Seed"), 2, 2)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_spin, 1)
        seed_row.addWidget(random_seed)
        grid.addLayout(seed_row, 2, 3)
        grid.addWidget(QLabel("Number of outputs"), 3, 0)
        grid.addWidget(self.count_spin, 3, 1)
        grid.addWidget(QLabel("True CFG"), 3, 2)
        grid.addWidget(self.cfg_spin, 3, 3)
        grid.addWidget(self.steps_override, 4, 0, 1, 2)
        grid.addWidget(self.steps_spin, 4, 2)
        grid.addWidget(QLabel("Profile default shown"), 4, 3)
        grid.addWidget(QLabel("Maximum prompt tokens"), 5, 0)
        grid.addWidget(self.max_sequence_spin, 5, 1)
        layout.addWidget(image_group)

        result_group = QGroupBox("Latest result")
        result_layout = QVBoxLayout(result_group)
        self.result_preview = QLabel("Generated output will appear here")
        self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview.setMinimumHeight(320)
        self.result_preview.setObjectName("ImagePreview")
        self.result_preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.result_preview.customContextMenuRequested.connect(self.result_menu)
        result_layout.addWidget(self.result_preview)
        self.result_info = QLabel("")
        self.result_info.setObjectName("Hint")
        self.result_info.setWordWrap(True)
        result_layout.addWidget(self.result_info)
        layout.addWidget(result_group)
        layout.addStretch(1)

        return self._scroll_page(page)

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 16)
        layout.setSpacing(12)

        install_group = QGroupBox("Installation and model")
        install_layout = QGridLayout(install_group)
        install_layout.setColumnStretch(1, 1)
        self.install_details = QLabel("")
        self.install_details.setWordWrap(True)
        self.install_details.setObjectName("Hint")
        self.model_combo = NoWheelComboBox()
        self.model_combo.setToolTip(
            "Only downloaded checkpoints are selectable. Switching model profiles reloads the persistent backend."
        )
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        refresh_models = QPushButton("Refresh")
        refresh_models.clicked.connect(self.refresh_models)
        open_models = QPushButton("Open model folder")
        open_models.clicked.connect(lambda: self.open_path(self.model_dir))
        install_layout.addWidget(QLabel("Installed model"), 0, 0)
        install_layout.addWidget(self.model_combo, 0, 1)
        install_layout.addWidget(refresh_models, 0, 2)
        install_layout.addWidget(open_models, 0, 3)
        install_layout.addWidget(self.install_details, 1, 0, 1, 4)
        layout.addWidget(install_group)

        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)
        output_layout.setColumnStretch(1, 1)
        self.output_folder_edit = QLineEdit(str(self.root / "output" / "edits"))
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.browse_output_folder)
        open_output = QPushButton("Open")
        open_output.clicked.connect(lambda: self.open_path(Path(self.output_folder_edit.text())))
        self.format_combo = NoWheelComboBox()
        self.format_combo.addItem("PNG · lossless", "png")
        self.format_combo.addItem("JPG · smaller files", "jpg")
        self.format_combo.currentIndexChanged.connect(self.output_format_changed)
        self.jpeg_quality_spin = NoWheelSpinBox()
        self.jpeg_quality_spin.setRange(70, 100)
        self.jpeg_quality_spin.setValue(95)
        self.jpeg_quality_spin.setToolTip(
            "95 keeps fine edit detail while reducing file size. PNG is preferable for repeated editing or text-heavy images."
        )
        self.auto_name_check = QCheckBox("Automatic filename")
        self.auto_name_check.setChecked(True)
        self.auto_name_check.setToolTip(
            "Creates qwen2511_<first three prompt words>_<seed>_<date>_<time> and prevents overwriting existing files."
        )
        self.auto_name_check.toggled.connect(self.auto_name_changed)
        self.manual_name_edit = QLineEdit("qwen2511_edit")
        self.manual_name_edit.setEnabled(False)
        self.filename_preview = QLabel("")
        self.filename_preview.setObjectName("Hint")
        output_layout.addWidget(QLabel("Folder"), 0, 0)
        output_layout.addWidget(self.output_folder_edit, 0, 1)
        output_layout.addWidget(browse_output, 0, 2)
        output_layout.addWidget(open_output, 0, 3)
        output_layout.addWidget(QLabel("Format"), 1, 0)
        output_layout.addWidget(self.format_combo, 1, 1)
        output_layout.addWidget(QLabel("JPG quality"), 1, 2)
        output_layout.addWidget(self.jpeg_quality_spin, 1, 3)
        output_layout.addWidget(self.auto_name_check, 2, 0, 1, 2)
        output_layout.addWidget(self.manual_name_edit, 2, 2, 1, 2)
        output_layout.addWidget(QLabel("Example"), 3, 0)
        output_layout.addWidget(self.filename_preview, 3, 1, 1, 3)
        layout.addWidget(output_group)

        queue_group = QGroupBox("FrameVision queue")
        queue_layout = QVBoxLayout(queue_group)
        self.queue_check = QCheckBox("Use FrameVision queue")
        self.queue_check.setChecked(True)
        self.queue_check.setToolTip(
            "When enabled, the footer adds this edit to jobs/pending instead of running it immediately. "
            "The Queue worker launches the same Qwen 2511 INT4 backend and settings."
        )
        self.queue_check.toggled.connect(self.queue_mode_changed)
        queue_hint = QLabel(
            "Queued jobs run sequentially with FrameVision's other jobs. Disable this to use the persistent direct-run backend."
        )
        queue_hint.setWordWrap(True)
        queue_hint.setObjectName("Hint")
        queue_layout.addWidget(self.queue_check)
        queue_layout.addWidget(queue_hint)
        layout.addWidget(queue_group)

        memory_group = QGroupBox("Memory and backend")
        memory_layout = QGridLayout(memory_group)
        memory_layout.setColumnStretch(1, 1)
        self.offload_combo = NoWheelComboBox()
        self.offload_combo.addItem("Auto · model CPU offload above 18 GB", "auto")
        self.offload_combo.addItem("Balanced · model CPU offload", "model-cpu")
        self.offload_combo.addItem("Lowest VRAM · Nunchaku blocks + sequential offload", "low-vram")
        self.offload_combo.addItem("Full GPU · fastest when it fits", "full-gpu")
        self.offload_combo.setToolTip(
            "Auto is the safe default. Lowest VRAM can run on smaller cards but is slower. Full GPU may exceed 24 GB once the text encoder and VAE are included."
        )
        self.offload_combo.currentIndexChanged.connect(self.offload_changed)
        self.blocks_spin = NoWheelSpinBox()
        self.blocks_spin.setRange(1, 60)
        self.blocks_spin.setValue(1)
        self.blocks_spin.setToolTip(
            "Only used by Lowest VRAM mode. More blocks on the GPU improve speed but raise peak VRAM. Start at 1, then increase cautiously."
        )
        self.pin_memory_check = QCheckBox("Use pinned CPU memory")
        self.pin_memory_check.setToolTip(
            "Can improve transfer speed on some systems but reserves page-locked RAM. Leave off unless low-VRAM offload is stable and system RAM is plentiful."
        )
        self.keep_loaded_check = QCheckBox("Keep model loaded between generations")
        self.keep_loaded_check.setChecked(True)
        self.keep_loaded_check.setToolTip(
            "Avoids reloading the transformer, text encoder and VAE after every edit. The backend reloads automatically when model, LoRAs or offload settings change."
        )
        self.cleanup_check = QCheckBox("Empty unused CUDA cache between multiple outputs")
        self.cleanup_check.setChecked(True)
        self.cleanup_check.setToolTip(
            "Useful for long batches and tight VRAM. It does not unload the model and normally has little speed impact."
        )
        unload_button = QPushButton("Unload model now")
        unload_button.clicked.connect(self.unload_backend)
        restart_button = QPushButton("Restart backend")
        restart_button.clicked.connect(self.restart_backend)
        memory_layout.addWidget(QLabel("Offload profile"), 0, 0)
        memory_layout.addWidget(self.offload_combo, 0, 1, 1, 3)
        memory_layout.addWidget(QLabel("Transformer blocks on GPU"), 1, 0)
        memory_layout.addWidget(self.blocks_spin, 1, 1)
        memory_layout.addWidget(self.pin_memory_check, 1, 2, 1, 2)
        memory_layout.addWidget(self.keep_loaded_check, 2, 0, 1, 2)
        memory_layout.addWidget(self.cleanup_check, 2, 2, 1, 2)
        memory_layout.addWidget(unload_button, 3, 2)
        memory_layout.addWidget(restart_button, 3, 3)
        layout.addWidget(memory_group)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        self.log_edit.setMinimumHeight(360)
        log_layout.addWidget(self.log_edit)
        log_buttons = QHBoxLayout()
        clear_log = QPushButton("Clear")
        clear_log.clicked.connect(self.log_edit.clear)
        copy_log = QPushButton("Copy")
        copy_log.clicked.connect(lambda: QApplication.clipboard().setText(self.log_edit.toPlainText()))
        save_log = QPushButton("Save log")
        save_log.clicked.connect(self.save_log)
        log_buttons.addWidget(clear_log)
        log_buttons.addWidget(copy_log)
        log_buttons.addWidget(save_log)
        log_buttons.addStretch(1)
        log_layout.addLayout(log_buttons)
        layout.addWidget(log_group)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-size: 10pt; }
            QLabel#PageTitle { font-size: 20pt; font-weight: 700; }
            QLabel#Subtitle, QLabel#Hint { color: #7e8a99; }
            QLabel#StatusBadge { padding: 6px 10px; border: 1px solid #4f6f8f; border-radius: 10px; }
            QGroupBox { font-weight: 600; margin-top: 9px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QFrame#ActionBar { border-top: 1px solid #54606d; }
            QFrame#PreviewFrame, QLabel#ImagePreview { border: 1px solid #52606d; border-radius: 6px; }
            QPushButton#GenerateButton { font-size: 11pt; font-weight: 700; padding: 9px 18px; }
            QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; }
            """
        )

    # ------------------------------- model/settings ------------------------------
    def installed_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = read_json(self.model_dir / "model_profiles.json", {})
        result: dict[str, dict[str, Any]] = {}
        for key, profile in profiles.get("models", {}).items() if isinstance(profiles, dict) else []:
            checkpoint = self.model_dir / str(profile.get("relative_path", ""))
            if checkpoint.exists():
                item = dict(profile)
                item["path"] = str(checkpoint)
                result[key] = item
        return result

    def refresh_models(self) -> None:
        previous = self.model_combo.currentData() if self.model_combo.count() else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        profiles = self.installed_profiles()
        for key, profile in profiles.items():
            label = MODEL_LABELS.get(key) or profile.get("label") or key
            size = profile.get("size_gib")
            suffix = f" · {size} GiB" if size else ""
            self.model_combo.addItem(label + suffix, key)
        if previous:
            index = self.model_combo.findData(previous)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self.update_install_status()
        self.update_steps_from_model()
        self.update_generate_enabled()

    def update_install_status(self) -> None:
        missing = []
        if not self.env_python.exists():
            missing.append("environment")
        if not (self.model_dir / "base" / "Qwen-Image-Edit-2511" / "model_index.json").exists():
            missing.append("shared Qwen assets")
        profiles = self.installed_profiles()
        if not profiles:
            missing.append("INT4 checkpoint")

        if missing:
            self.install_badge.setText("Missing: " + ", ".join(missing))
            self.install_badge.setProperty("ok", False)
        else:
            self.install_badge.setText(f"Ready · {len(profiles)} model{'s' if len(profiles) != 1 else ''}")
            self.install_badge.setProperty("ok", True)

        self.install_details.setText(
            f"Environment: {self.env_python}\n"
            f"Model root: {self.model_dir}\n"
            f"Installed profiles: {', '.join(profiles) if profiles else 'none'}"
        )

    def load_settings(self) -> None:
        data = read_json(self.settings_file, {})
        if not isinstance(data, dict):
            data = {}

        self.prompt_edit.setPlainText(str(data.get("prompt", "")))
        self.negative_edit.setText(str(data.get("negative_prompt", "")))

        model_key = data.get("model", "recommended")
        index = self.model_combo.findData(model_key)
        if index < 0 and self.model_combo.count():
            index = 0
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        self._set_combo_data(self.resolution_combo, data.get("resolution", "source"))
        self.width_spin.setValue(int(data.get("width", 1024)))
        self.height_spin.setValue(int(data.get("height", 1024)))
        self.source_max_spin.setValue(int(data.get("source_max_side", 1024)))
        self.seed_spin.setValue(int(data.get("seed", -1)))
        self.count_spin.setValue(int(data.get("count", 1)))
        self.cfg_spin.setValue(float(data.get("true_cfg_scale", 1.0)))
        self.steps_override.setChecked(bool(data.get("steps_override", False)))
        self.steps_spin.setValue(int(data.get("steps", 4)))
        self.max_sequence_spin.setValue(int(data.get("max_sequence_length", 512)))

        self.output_folder_edit.setText(str(data.get("output_folder", self.root / "output" / "edits")))
        self._set_combo_data(self.format_combo, data.get("format", "png"))
        self.jpeg_quality_spin.setValue(int(data.get("jpeg_quality", 95)))
        self.auto_name_check.setChecked(bool(data.get("auto_name", True)))
        self.manual_name_edit.setText(str(data.get("manual_name", "qwen2511_edit")))

        self._set_combo_data(self.offload_combo, data.get("offload", "auto"))
        self.blocks_spin.setValue(int(data.get("blocks_on_gpu", 1)))
        self.pin_memory_check.setChecked(bool(data.get("pin_memory", False)))
        self.keep_loaded_check.setChecked(bool(data.get("keep_loaded", True)))
        self.cleanup_check.setChecked(bool(data.get("cleanup_between", True)))
        self.queue_check.setChecked(bool(data.get("use_framevision_queue", True)))

        self.reference_paths = []
        self.reference_list.clear()
        self.add_reference_paths([path for path in data.get("references", []) if Path(path).exists()])
        self.lora_rows = []
        self.lora_table.setRowCount(0)
        for row in data.get("loras", []):
            if isinstance(row, dict):
                self.add_lora_path(str(row.get("path", "")), float(row.get("strength", 1.0)), bool(row.get("enabled", True)))

        self.resolution_changed()
        self.steps_override_changed(self.steps_override.isChecked())
        self.auto_name_changed(self.auto_name_check.isChecked())
        self.output_format_changed()
        self.offload_changed()
        self.queue_mode_changed(self.queue_check.isChecked())

    def _set_combo_data(self, combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def settings_payload(self) -> dict[str, Any]:
        self.sync_reference_paths()
        self.sync_lora_rows()
        return {
            "schema_version": SETTINGS_SCHEMA,
            "prompt": self.prompt_edit.toPlainText(),
            "negative_prompt": self.negative_edit.text(),
            "references": self.reference_paths,
            "loras": self.lora_rows,
            "model": self.model_combo.currentData(),
            "resolution": self.resolution_combo.currentData(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "source_max_side": self.source_max_spin.value(),
            "seed": self.seed_spin.value(),
            "count": self.count_spin.value(),
            "true_cfg_scale": self.cfg_spin.value(),
            "steps_override": self.steps_override.isChecked(),
            "steps": self.steps_spin.value(),
            "max_sequence_length": self.max_sequence_spin.value(),
            "output_folder": self.output_folder_edit.text(),
            "format": self.format_combo.currentData(),
            "jpeg_quality": self.jpeg_quality_spin.value(),
            "auto_name": self.auto_name_check.isChecked(),
            "manual_name": self.manual_name_edit.text(),
            "offload": self.offload_combo.currentData(),
            "blocks_on_gpu": self.blocks_spin.value(),
            "pin_memory": self.pin_memory_check.isChecked(),
            "keep_loaded": self.keep_loaded_check.isChecked(),
            "cleanup_between": self.cleanup_check.isChecked(),
            "use_framevision_queue": self.queue_check.isChecked(),
        }

    def save_settings(self) -> None:
        if self._loading_settings:
            return
        try:
            atomic_write_json(self.settings_file, self.settings_payload())
        except Exception as exc:
            self.append_log(f"Could not save settings: {exc}", "error")

    def schedule_save(self, *_args) -> None:
        if not self._loading_settings:
            self.save_timer.start()
            self.update_filename_preview()

    def _connect_autosave(self) -> None:
        widgets = [
            self.prompt_edit,
            self.negative_edit,
            self.model_combo,
            self.resolution_combo,
            self.width_spin,
            self.height_spin,
            self.source_max_spin,
            self.seed_spin,
            self.count_spin,
            self.cfg_spin,
            self.steps_override,
            self.steps_spin,
            self.max_sequence_spin,
            self.output_folder_edit,
            self.format_combo,
            self.jpeg_quality_spin,
            self.auto_name_check,
            self.manual_name_edit,
            self.offload_combo,
            self.blocks_spin,
            self.pin_memory_check,
            self.keep_loaded_check,
            self.cleanup_check,
            self.queue_check,
        ]
        for widget in widgets:
            if isinstance(widget, QTextEdit):
                widget.textChanged.connect(self.schedule_save)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.schedule_save)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_save)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.schedule_save)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.schedule_save)

    # ----------------------------- reference images -----------------------------
    def browse_references(self) -> None:
        start = str(Path(self.reference_paths[-1]).parent) if self.reference_paths else str(self.root)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Qwen reference images",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        self.add_reference_paths(files)

    def add_reference_paths(self, paths: list[str]) -> None:
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() not in SUPPORTED_IMAGES or not path.exists():
                continue
            if str(path) in self.reference_paths:
                continue
            if len(self.reference_paths) >= 8:
                QMessageBox.information(self, APP_NAME, "A maximum of eight reference images is supported by this helper.")
                break
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            self.reference_paths.append(str(path))
            thumb = pixmap.scaled(
                150,
                110,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem(QIcon(thumb), f"Image {len(self.reference_paths)}\n{path.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            item.setSizeHint(QSize(170, 145))
            self.reference_list.addItem(item)
        if self.reference_list.count() and self.reference_list.currentRow() < 0:
            self.reference_list.setCurrentRow(0)
        self.renumber_references()
        self.schedule_save()
        self.update_generate_enabled()
        self.resolution_changed()

    def paste_reference(self) -> None:
        image = QApplication.clipboard().image()
        if image.isNull():
            QMessageBox.information(self, APP_NAME, "The clipboard does not contain an image.")
            return
        temp = self.root / "temp" / "qwen2511_int4"
        temp.mkdir(parents=True, exist_ok=True)
        path = temp / f"clipboard_ref_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        if not image.save(str(path), "PNG"):
            QMessageBox.warning(self, APP_NAME, f"Could not save the clipboard image to {path}")
            return
        self.add_reference_paths([str(path)])

    def remove_reference(self) -> None:
        row = self.reference_list.currentRow()
        if row < 0:
            return
        self.reference_list.takeItem(row)
        self.sync_reference_paths()
        self.renumber_references()
        self.show_reference_preview(self.reference_list.currentItem(), None)
        self.schedule_save()
        self.update_generate_enabled()

    def clear_references(self) -> None:
        self.reference_list.clear()
        self.reference_paths = []
        self.reference_preview.clear()
        self.reference_preview.setText("Select a reference image")
        self.reference_info.clear()
        self.schedule_save()
        self.update_generate_enabled()

    def sync_reference_paths(self) -> None:
        self.reference_paths = [
            str(self.reference_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.reference_list.count())
        ]

    def references_reordered(self) -> None:
        self.sync_reference_paths()
        self.renumber_references()
        self.schedule_save()

    def renumber_references(self) -> None:
        for row in range(self.reference_list.count()):
            item = self.reference_list.item(row)
            path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
            item.setText(f"Image {row + 1}\n{path.name}")

    def show_reference_preview(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        path = Path(str(current.data(Qt.ItemDataRole.UserRole)))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.reference_preview.setText("Preview unavailable")
            return
        target = self.reference_preview.size() - QSize(16, 16)
        self.reference_preview.setPixmap(
            pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self.reference_info.setText(f"{path.name} · {pixmap.width()} × {pixmap.height()}\n{path}")

    def open_reference_dialog(self, item: QListWidgetItem) -> None:
        path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        if path.exists():
            ImagePreviewDialog(path, self).exec()

    # ----------------------------------- LoRA -----------------------------------
    def browse_loras(self) -> None:
        default = self.model_dir / "loras"
        default.mkdir(parents=True, exist_ok=True)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Qwen LoRAs",
            str(default),
            "LoRA weights (*.safetensors *.bin *.pt)",
        )
        for path in files:
            self.add_lora_path(path)
        self.schedule_save()

    def add_lora_path(self, value: str, strength: float = 1.0, enabled: bool = True) -> None:
        path = Path(value).expanduser().resolve()
        if not path.exists() or path.suffix.lower() not in SUPPORTED_LORAS:
            return
        for row in range(self.lora_table.rowCount()):
            if self.lora_table.item(row, 3) and self.lora_table.item(row, 3).text() == str(path):
                return

        row = self.lora_table.rowCount()
        self.lora_table.insertRow(row)
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(enabled_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        enabled_item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
        enabled_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        name_item = QTableWidgetItem(path.name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        strength_widget = NoWheelDoubleSpinBox()
        strength_widget.setRange(-2.0, 3.0)
        strength_widget.setDecimals(3)
        strength_widget.setSingleStep(0.05)
        strength_widget.setValue(strength)
        strength_widget.setToolTip(
            "1.0 applies the LoRA at its trained strength. Lower values are subtler; negative strengths are only useful for LoRAs designed for inversion."
        )
        strength_widget.valueChanged.connect(self.schedule_save)
        path_item = QTableWidgetItem(str(path))
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.lora_table.setItem(row, 0, enabled_item)
        self.lora_table.setItem(row, 1, name_item)
        self.lora_table.setCellWidget(row, 2, strength_widget)
        self.lora_table.setItem(row, 3, path_item)
        self.sync_lora_rows()

    def remove_lora(self) -> None:
        row = self.lora_table.currentRow()
        if row >= 0:
            self.lora_table.removeRow(row)
            self.sync_lora_rows()
            self.schedule_save()

    def clear_loras(self) -> None:
        self.lora_table.setRowCount(0)
        self.lora_rows = []
        self.schedule_save()

    def sync_lora_rows(self) -> None:
        rows = []
        for row in range(self.lora_table.rowCount()):
            enabled_item = self.lora_table.item(row, 0)
            path_item = self.lora_table.item(row, 3)
            strength = self.lora_table.cellWidget(row, 2)
            if path_item is None or not isinstance(strength, QDoubleSpinBox):
                continue
            rows.append(
                {
                    "enabled": enabled_item.checkState() == Qt.CheckState.Checked if enabled_item else True,
                    "path": path_item.text(),
                    "strength": strength.value(),
                }
            )
        self.lora_rows = rows

    # ------------------------------- UI reactions -------------------------------
    def model_changed(self) -> None:
        self.update_steps_from_model()
        self.schedule_save()

    def update_steps_from_model(self) -> None:
        key = self.model_combo.currentData()
        profile = self.installed_profiles().get(str(key), {})
        default_steps = int(profile.get("steps", 4))
        if not self.steps_override.isChecked():
            self.steps_spin.setValue(default_steps)
        self.steps_spin.setSuffix(f"  (profile: {default_steps})")

    def steps_override_changed(self, enabled: bool) -> None:
        self.steps_spin.setEnabled(enabled)
        if not enabled:
            self.update_steps_from_model()

    def resolution_changed(self) -> None:
        mode = self.resolution_combo.currentData()
        self.source_max_spin.setEnabled(mode == "source")
        self.width_spin.setEnabled(mode == "custom")
        self.height_spin.setEnabled(mode == "custom")
        if isinstance(mode, str) and "x" in mode and mode != "custom":
            width, height = mode.split("x", 1)
            self.width_spin.setValue(int(width))
            self.height_spin.setValue(int(height))
        elif mode == "source":
            size = self.source_resolution()
            if size:
                self.width_spin.setValue(size[0])
                self.height_spin.setValue(size[1])

    def source_resolution(self) -> Optional[tuple[int, int]]:
        if not self.reference_paths:
            return None
        image = QImage(self.reference_paths[0])
        if image.isNull():
            return None
        width, height = image.width(), image.height()
        maximum = self.source_max_spin.value()
        scale = min(1.0, maximum / max(width, height))
        width = max(256, int(round((width * scale) / 16) * 16))
        height = max(256, int(round((height * scale) / 16) * 16))
        return min(width, 2048), min(height, 2048)

    def effective_resolution(self) -> tuple[int, int]:
        if self.resolution_combo.currentData() == "source":
            return self.source_resolution() or (1024, 1024)
        return self.width_spin.value(), self.height_spin.value()

    def output_format_changed(self) -> None:
        self.jpeg_quality_spin.setEnabled(self.format_combo.currentData() == "jpg")
        self.update_filename_preview()

    def auto_name_changed(self, checked: bool) -> None:
        self.manual_name_edit.setEnabled(not checked)
        self.update_filename_preview()

    def offload_changed(self) -> None:
        low = self.offload_combo.currentData() == "low-vram"
        self.blocks_spin.setEnabled(low)
        self.pin_memory_check.setEnabled(low)

    def queue_mode_changed(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.generate_button.setText("Add to queue" if enabled else "Generate edit")
        # A model kept by the direct-run backend would compete with the Queue
        # worker for VRAM. Release it as soon as queue mode is selected.
        if enabled and not self.running:
            self.stop_backend()
        self.cancel_button.setVisible((not enabled) or self.running)
        if not self.running:
            self.status_label.setText("Queue mode" if enabled else "Ready")
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("Ready to queue" if enabled else "Idle")
        self.update_generate_enabled()

    def update_filename_preview(self) -> None:
        extension = self.format_combo.currentData() or "png"
        seed = self.seed_spin.value()
        seed_text = "random-seed" if seed < 0 else str(seed)
        if self.auto_name_check.isChecked():
            preview = f"qwen2511_{safe_slug_words(self.prompt_edit.toPlainText())}_{seed_text}_{datetime.now():%Y%m%d_%H%M%S}.{extension}"
        else:
            stem = Path(self.manual_name_edit.text() or "qwen2511_edit").stem
            preview = f"{stem}.{extension}"
        self.filename_preview.setText(preview)

    def update_generate_enabled(self) -> None:
        ready = (
            not self.running
            and self.env_python.exists()
            and self.model_combo.count() > 0
            and self.reference_list.count() > 0
        )
        self.generate_button.setEnabled(ready)
        try:
            self.queue_check.setEnabled(not self.running)
            self.generate_button.setText("Add to queue" if self.queue_check.isChecked() else "Generate edit")
        except Exception:
            pass

    # -------------------------------- generation --------------------------------
    def build_request(self) -> dict[str, Any]:
        self.sync_reference_paths()
        self.sync_lora_rows()
        width, height = self.effective_resolution()
        return {
            "command": "generate",
            "prompt": self.prompt_edit.toPlainText().strip(),
            "negative_prompt": self.negative_edit.text(),
            "references": self.reference_paths,
            "loras": self.lora_rows,
            "model": self.model_combo.currentData(),
            "width": width,
            "height": height,
            "seed": self.seed_spin.value(),
            "count": self.count_spin.value(),
            "true_cfg_scale": self.cfg_spin.value(),
            "steps": self.steps_spin.value(),
            "max_sequence_length": self.max_sequence_spin.value(),
            "offload": self.offload_combo.currentData(),
            "blocks_on_gpu": self.blocks_spin.value(),
            "pin_memory": self.pin_memory_check.isChecked(),
            "cleanup_between": self.cleanup_check.isChecked(),
            "output": {
                "folder": self.output_folder_edit.text(),
                "format": self.format_combo.currentData(),
                "jpeg_quality": self.jpeg_quality_spin.value(),
                "auto_name": self.auto_name_check.isChecked(),
                "manual_name": self.manual_name_edit.text(),
            },
        }

    def validate_request(self, request: dict[str, Any]) -> Optional[str]:
        if not request["prompt"]:
            return "Enter an edit instruction."
        if not request["references"]:
            return "Add at least one reference image."
        if not request["model"]:
            return "No installed INT4 model is selected."
        if not self.env_python.exists():
            return f"Dedicated environment Python was not found: {self.env_python}"
        output = Path(request["output"]["folder"]).expanduser()
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"The output folder cannot be created: {exc}"
        return None

    def generate(self) -> None:
        request = self.build_request()
        error = self.validate_request(request)
        if error:
            QMessageBox.warning(self, APP_NAME, error)
            return
        self.save_settings()
        if self.queue_check.isChecked():
            self.enqueue_generation(request)
            return
        self.running = True
        self.update_generate_enabled()
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, max(1, request["steps"] * request["count"]))
        self.progress.setValue(0)
        self.progress.setFormat("Starting backend…")
        self.status_label.setText("Starting")
        self.pending_request = request
        self.generationRequested.emit(request)
        self.ensure_backend()

    def enqueue_generation(self, request: dict[str, Any]) -> None:
        try:
            try:
                from helpers.queue_adapter import enqueue_qwen2511_int4_request
            except Exception:
                from queue_adapter import enqueue_qwen2511_int4_request
            job_id = enqueue_qwen2511_int4_request(request, root=self.root)
            if not job_id:
                raise RuntimeError("The queue adapter did not create a job file.")
            self.stop_backend()
            self.status_label.setText("Added to queue")
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat("Queued")
            self.append_log(f"Added Qwen 2511 INT4 edit to FrameVision queue: {job_id}")
            QMessageBox.information(self, APP_NAME, "The edit was added to the FrameVision queue.")
        except Exception as exc:
            self.status_label.setText("Queue failed")
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("Queue failed")
            self.append_log(f"Could not add edit to queue: {exc}", "error")
            QMessageBox.critical(self, APP_NAME, f"Could not add the edit to the FrameVision queue:\n\n{exc}")
        finally:
            self.update_generate_enabled()

    def ensure_backend(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            if self.server_ready and self.pending_request:
                self.send_request(self.pending_request)
                self.pending_request = None
            return

        self.server_ready = False
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        env.insert("HF_HUB_OFFLINE", "1")
        env.insert("TRANSFORMERS_OFFLINE", "1")
        env.insert("FRAMEVISION_ROOT", str(self.root))
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_backend_stdout)
        self.process.readyReadStandardError.connect(self.read_backend_stderr)
        self.process.finished.connect(self.backend_finished)
        self.process.errorOccurred.connect(self.backend_error)
        arguments = ["-u", str(Path(__file__).resolve()), "--server", "--root", str(self.root)]
        self.append_log(f"Starting backend: {self.env_python} {' '.join(arguments)}")
        self.process.start(str(self.env_python), arguments)

    def send_request(self, request: dict[str, Any]) -> None:
        if not self.process or self.process.state() == QProcess.ProcessState.NotRunning:
            raise RuntimeError("Backend is not running")
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        self.process.write(payload.encode("utf-8"))
        self.process.waitForBytesWritten(1000)
        self.append_log(f"Sent generation request using model '{request['model']}'.")

    def read_backend_stdout(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.stdout_buffer += text
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            self.handle_backend_line(line.rstrip("\r"))

    def read_backend_stderr(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.stderr_buffer += text
        while "\n" in self.stderr_buffer:
            line, self.stderr_buffer = self.stderr_buffer.split("\n", 1)
            if line.strip():
                self.append_log(line.rstrip("\r"), "stderr")

    def handle_backend_line(self, line: str) -> None:
        if not line:
            return
        if not line.startswith(EVENT_PREFIX):
            self.append_log(line)
            return
        try:
            event = json.loads(line[len(EVENT_PREFIX) :])
        except ValueError:
            self.append_log(line, "error")
            return
        event_name = event.get("event")
        if event_name == "ready":
            self.server_ready = True
            self.append_log(f"Backend ready (PID {event.get('pid')}).")
            if self.pending_request:
                self.send_request(self.pending_request)
                self.pending_request = None
        elif event_name == "log":
            self.append_log(str(event.get("message", "")), str(event.get("level", "info")))
        elif event_name == "status":
            message = str(event.get("message", "Working"))
            self.status_label.setText(message)
            self.progress.setFormat(message)
            self.append_log(message)
        elif event_name == "model_loaded":
            self.append_log(
                f"Loaded {event.get('model_label')} · offload={event.get('offload')} · LoRAs={event.get('lora_count')}"
            )
        elif event_name == "progress":
            current = int(event.get("current", 0))
            total = int(event.get("total", 1))
            image = int(event.get("image", 1))
            image_total = int(event.get("image_total", 1))
            overall = (image - 1) * total + current
            self.progress.setRange(0, total * image_total)
            self.progress.setValue(overall)
            self.progress.setFormat(f"Image {image}/{image_total} · step {current}/{total}")
        elif event_name == "image_saved":
            self.append_log(
                f"Saved {event.get('path')} · seed {event.get('seed')} · {event.get('seconds')} seconds"
            )
        elif event_name == "result":
            self.generation_finished(event)
        elif event_name == "error":
            self.generation_failed(str(event.get("message", "Unknown backend error")), str(event.get("traceback", "")))
        elif event_name == "unloaded":
            self.append_log("Backend model unloaded.")
        elif event_name == "bye":
            self.append_log("Backend closed.")
        else:
            self.append_log(json.dumps(event, ensure_ascii=False))

    def generation_finished(self, event: dict[str, Any]) -> None:
        outputs = [Path(path) for path in event.get("outputs", [])]
        if outputs:
            self.last_output = outputs[-1]
            self.show_result(self.last_output)
            self.open_output_button.setEnabled(True)
        self.running = False
        self.cancel_button.setEnabled(False)
        self.update_generate_enabled()
        self.status_label.setText("Finished")
        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat(f"Finished · {event.get('seconds', 0)} seconds")
        self.result_info.setText(
            f"{len(outputs)} output{'s' if len(outputs) != 1 else ''} · seeds {event.get('seeds')} · total {event.get('seconds')} seconds"
        )
        self.append_log(f"Generation complete in {event.get('seconds')} seconds.")
        if not self.keep_loaded_check.isChecked():
            self.unload_backend()

    def generation_failed(self, message: str, details: str) -> None:
        self.running = False
        self.cancel_button.setEnabled(False)
        self.update_generate_enabled()
        self.status_label.setText("Failed")
        self.progress.setFormat("Failed")
        self.append_log(message, "error")
        if details:
            self.append_log(details, "error")
        QMessageBox.critical(self, APP_NAME, message)

    def cancel_generation(self) -> None:
        if not self.process or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.append_log("Cancelling generation by stopping the backend process.", "warning")
        self.process.kill()
        self.process.waitForFinished(3000)
        self.process = None
        self.server_ready = False
        self.pending_request = None
        self.running = False
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelled")
        self.progress.setFormat("Cancelled")
        self.update_generate_enabled()

    def unload_backend(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning and self.server_ready:
            self.send_request({"command": "unload"})

    def restart_backend(self) -> None:
        if self.running:
            QMessageBox.information(self, APP_NAME, "Cancel the current generation before restarting the backend.")
            return
        self.stop_backend()
        self.append_log("Backend will restart on the next generation.")

    def stop_backend(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            try:
                if self.server_ready:
                    self.send_request({"command": "quit"})
                    if not self.process.waitForFinished(2500):
                        self.process.kill()
                else:
                    self.process.kill()
            except Exception:
                self.process.kill()
        self.process = None
        self.server_ready = False

    def backend_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self.stdout_buffer.strip():
            self.handle_backend_line(self.stdout_buffer.strip())
        if self.stderr_buffer.strip():
            self.append_log(self.stderr_buffer.strip(), "stderr")
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        was_running = self.running
        self.process = None
        self.server_ready = False
        if was_running:
            self.generation_failed(f"Backend exited unexpectedly with code {exit_code}.", "")
        else:
            self.append_log(f"Backend exited with code {exit_code}.")

    def backend_error(self, error: QProcess.ProcessError) -> None:
        message = f"Backend process error: {error}"
        self.append_log(message, "error")
        if self.running:
            self.generation_failed(message, "")

    # --------------------------------- previews ---------------------------------
    def show_result(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.result_preview.setText(f"Saved: {path}")
            return
        target = QSize(max(300, self.result_preview.width() - 20), 600)
        self.result_preview.setPixmap(
            pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self.result_preview.setToolTip(str(path))

    def result_menu(self, pos) -> None:
        if not self.last_output:
            return
        menu = QMenu(self)
        open_file = menu.addAction("Open image")
        open_folder = menu.addAction("Open containing folder")
        copy_path = menu.addAction("Copy path")
        selected = menu.exec(self.result_preview.mapToGlobal(pos))
        if selected == open_file:
            self.open_path(self.last_output)
        elif selected == open_folder:
            self.open_path(self.last_output.parent)
        elif selected == copy_path:
            QApplication.clipboard().setText(str(self.last_output))

    def open_last_output(self) -> None:
        if self.last_output:
            self.open_path(self.last_output)

    def open_path(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if path.suffix and not path.exists():
            path = path.parent
        path.mkdir(parents=True, exist_ok=True) if not path.suffix else None
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ---------------------------------- logs ------------------------------------
    def append_log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = level.upper() if level not in {"info", "stdout"} else "INFO"
        self.log_edit.appendPlainText(f"[{timestamp}] [{prefix}] {message}")
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def save_log(self) -> None:
        default = self.root / "output" / "edits" / f"qwen2511_{datetime.now():%Y%m%d_%H%M%S}.log"
        path, _ = QFileDialog.getSaveFileName(self, "Save Qwen log", str(default), "Log files (*.log);;Text files (*.txt)")
        if path:
            Path(path).write_text(self.log_edit.toPlainText(), encoding="utf-8")

    def browse_output_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_folder_edit.text())
        if selected:
            self.output_folder_edit.setText(selected)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.save_settings()
        self.stop_backend()
        super().closeEvent(event)


class Qwen2511IntWindow(QMainWindow):
    def __init__(self, root: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 900)
        self.setMinimumSize(900, 700)
        self.widget = Qwen2511IntWidget(self, root=root)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.widget.save_settings()
        self.widget.stop_backend()
        super().closeEvent(event)


# Common names/factory methods make embedding into FrameVision easier.
Qwen2511IntUI = Qwen2511IntWidget
Qwen2511INTWidget = Qwen2511IntWidget


def create_widget(parent: Optional[QWidget] = None, root: Optional[Path] = None) -> Qwen2511IntWidget:
    return Qwen2511IntWidget(parent=parent, root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--root", help="FrameVision root folder")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    window = Qwen2511IntWindow(root=detect_root(args.root))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
