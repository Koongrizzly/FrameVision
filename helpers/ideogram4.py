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
import inspect
import contextlib
import re
import importlib.util
import json
import os
import queue
import random
import secrets
import sys
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_PIPELINE_CACHE: dict[tuple[str, bool], Any] = {}


def framevision_root() -> Path:
    return Path(__file__).resolve().parents[1]


def settings_path() -> Path:
    out = framevision_root() / "presets" / "setsave" / "ideogram.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def layout_templates_dir() -> Path:
    out = framevision_root() / "presets" / "setsave" / "templates"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_template_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_. -]+", "_", str(name or "template")).strip(" ._")
    safe = re.sub(r"\s+", "_", safe)
    return safe or "template"


def load_saved_settings() -> dict[str, Any]:
    path = settings_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_saved_settings(data: dict[str, Any]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def looks_like_sdnq_model_dir(path: Path) -> bool:
    try:
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return False
        # Avoid accepting accidental folders such as /helpers just because they exist.
        return (path / "model_index.json").exists() and (path / "ideogram4_sdnq_pipeline.py").exists()
    except Exception:
        return False


def default_model_dir() -> Path:
    data = load_saved_settings()
    p = Path(str(data.get("model_dir", "") or ""))
    if str(p) and looks_like_sdnq_model_dir(p):
        return p

    legacy = framevision_root() / "presets" / "setsave" / "ideogram4_sdnq_location.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            p = Path(data.get("model_dir", ""))
            if looks_like_sdnq_model_dir(p):
                return p
        except Exception:
            pass
    return framevision_root() / "models" / "ideogram4" / "sdnq_uint4"


def default_output_dir() -> Path:
    out = framevision_root() / "output" / "image" / "ideogram4"
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_gguf_dir() -> Path:
    data = load_saved_settings()
    p = Path(str(data.get("gguf_dir", "") or ""))
    if str(p) and p.exists():
        return p
    return framevision_root() / "models" / "ideogram4_gguf"


def default_sd_cli_path() -> Path:
    data = load_saved_settings()
    p = Path(str(data.get("sd_cli_path", "") or ""))
    if str(p) and p.exists():
        return p
    return framevision_root() / "presets" / "bin" / "sd-cli.exe"


# --- FrameVision queue fallback for Ideogram 4 GGUF -------------------------
def _enqueue_ideogram4_generate_local(settings: dict, priority: int = 610):
    """Queue Ideogram 4 directly when helpers.queue_adapter is older/broken.

    Some queue_adapter builds can throw NameError("inner") from unrelated
    widget-based enqueue helpers. Ideogram already has all required settings,
    so this fallback writes the normal FrameVision pending job directly.
    """
    try:
        from helpers.job_helper import make_job_json
    except Exception:
        from job_helper import make_job_json

    root = framevision_root()
    pending_dir = root / "jobs" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    data = dict(settings or {})
    out_dir = Path(str(data.get("output_dir") or default_output_dir())).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _text(key: str, default: str = "") -> str:
        try:
            return str(data.get(key, default) or "")
        except Exception:
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except Exception:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return int(default)

    def _float(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except Exception:
            return float(default)

    prompt = _text("prompt").strip()
    if not prompt:
        raise RuntimeError("Prompt is empty.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"
    out_file = str(out_dir / f"ideogram4_gguf_{stamp}.png")
    preview = prompt.replace("\n", " ").strip()[:80] or "Ideogram 4 GGUF"

    args = {
        "label": "Ideogram 4 GGUF: " + preview,
        "engine": "ideogram4_gguf",
        "prompt": prompt,
        "negative": _text("negative"),
        "width": _int("width", 1024),
        "height": _int("height", 1024),
        "steps": _int("steps", 20),
        "guidance": _float("guidance", 3.5),
        "preset": _text("preset", "Custom"),
        "seed": _int("seed", -1),
        "raw_prompt": bool(data.get("raw_prompt", False)),
        "gguf_stream_layers": bool(data.get("gguf_stream_layers", False)),
        "gguf_dir": _text("gguf_dir"),
        "gguf_diffusion_file": _text("gguf_diffusion_file"),
        "gguf_unconditional_file": _text("gguf_unconditional_file"),
        "gguf_llm_file": _text("gguf_llm_file"),
        "gguf_vae_file": _text("gguf_vae_file"),
        "sd_cli_path": _text("sd_cli_path"),
        "lora_enabled": bool(data.get("lora_enabled", False)),
        "lora_path": _text("lora_path"),
        "lora_scale": _float("lora_scale", 1.0),
        "lora_apply_mode": _lora_apply_mode_value(data.get("lora_apply_mode", "auto")),
        "output_dir": str(out_dir),
        "out_file": out_file,
        "outfile": out_file,
    }
    return make_job_json("ideogram4_generate", "", str(out_dir), args, str(pending_dir), priority=int(priority))
# ---------------------------------------------------------------------------


def looks_like_gguf_dir(path: Path) -> bool:
    try:
        path = Path(path)
        return path.exists() and path.is_dir()
    except Exception:
        return False


def looks_like_sd_cli_path(path: Path) -> bool:
    try:
        path = Path(path)
        return path.exists() and path.is_file() and path.name.lower() == "sd-cli.exe"
    except Exception:
        return False


def _gguf_name_role(name: str) -> str:
    low = str(name or "").strip().lower()
    if not low:
        return ""
    if low.endswith((".safetensors", ".sft")):
        return "vae"
    if any(token in low for token in ("qwen", "llm", "text-encoder", "text_encoder", "textencoder")):
        return "llm"
    if any(token in low for token in ("uncond", "unconditional")):
        return "unconditional"
    if low.endswith('.gguf'):
        return "diffusion"
    return ""


DEFAULT_GGUF_FILES = {
    "diffusion": "ideogram4-Q4_0.gguf",
    "unconditional": "ideogram4_uncond-Q4_0.gguf",
    "llm": "qwen3-vl-8b-instruct-q4_k_m.gguf",
    "vae": "flux2-vae.safetensors",
}


def default_gguf_file_name(key: str) -> str:
    data = load_saved_settings()
    saved_key = f"gguf_{key}_file"
    value = str(data.get(saved_key, "") or "").strip()
    return value or DEFAULT_GGUF_FILES.get(key, "")


def required_gguf_files(
    gguf_dir: Path,
    diffusion_file: str | None = None,
    unconditional_file: str | None = None,
    llm_file: str | None = None,
    vae_file: str | None = None,
) -> dict[str, Path]:
    return {
        "diffusion": gguf_dir / (diffusion_file or DEFAULT_GGUF_FILES["diffusion"]),
        "unconditional": gguf_dir / (unconditional_file or DEFAULT_GGUF_FILES["unconditional"]),
        "llm": gguf_dir / (llm_file or DEFAULT_GGUF_FILES["llm"]),
        "vae": gguf_dir / (vae_file or DEFAULT_GGUF_FILES["vae"]),
    }


def check_gguf_complete(
    gguf_dir: Path,
    sd_cli_path: Path,
    diffusion_file: str | None = None,
    unconditional_file: str | None = None,
    llm_file: str | None = None,
    vae_file: str | None = None,
) -> None:
    missing: list[str] = []
    if not sd_cli_path.exists():
        missing.append(str(sd_cli_path))
    for p in required_gguf_files(gguf_dir, diffusion_file, unconditional_file, llm_file, vae_file).values():
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit(
            "GGUF backend is incomplete. Missing required file(s):\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun presets/extra_env/ideogram4_gguf_install.py from the FrameVision root, or select the correct GGUF folder/files."
        )


def logs_dir() -> Path:
    out = framevision_root() / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def new_session_log_path(prefix: str = "ideogram4_sdnq_gui") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir() / f"{prefix}_{stamp}.log"


def _preview_text(value: str, max_len: int = 1200) -> str:
    value = str(value or "")
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"\n... [truncated {len(value) - max_len} chars]"


def output_path(output_dir: Path | None = None, prefix: str = "ideogram4") -> Path:
    out_dir = output_dir if output_dir is not None else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "ideogram4")).strip("._") or "ideogram4"
    return out_dir / f"{safe_prefix}_{stamp}.png"


def looks_like_json_prompt(text: str) -> bool:
    text = (text or "").strip()
    if not text.startswith("{"):
        return False
    try:
        value = json.loads(text)
        return isinstance(value, dict)
    except Exception:
        return False


def _norm_bbox_1000(box: dict[str, Any]) -> list[int]:
    # KJ Ideogram node format: normalized {x,y,w,h} fractions ->
    # [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    def c(v):
        try:
            return max(0, min(1000, round(float(v) * 1000)))
        except Exception:
            return 0
    x = float(box.get("x", 0.0) or 0.0)
    y = float(box.get("y", 0.0) or 0.0)
    w = float(box.get("w", 0.0) or 0.0)
    h = float(box.get("h", 0.0) or 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _palette_upper(colors) -> list[str]:
    if isinstance(colors, dict):
        colors = colors.values()
    out = []
    for color in colors or []:
        color = str(color or "").strip()
        if color:
            out.append(color.upper())
    return out


def _kj_dumps(v, lvl: int = 0) -> str:
    # Same style as the KJ node: pretty JSON, scalar arrays kept on one line.
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_kj_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _kj_dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _kj_dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def build_structured_caption(
    prompt: str,
    width: int,
    height: int,
    *,
    regions: list[dict[str, Any]] | None = None,
    background: str = "",
    aesthetics: str = "",
    lighting: str = "",
    photo: str = "",
    art_style: str = "",
    medium: str = "",
    style: str = "photo",
) -> str:
    prompt = (prompt or "").strip()

    # Keep the original plain-prompt path untouched for normal generation.
    # The KJ-exact schema below is used only when the layout builder passes a
    # regions list.
    if regions is None:
        orientation = "square"
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        caption = {
            "high_level_description": prompt,
            "style_description": {
                "aesthetics": "high quality, detailed, clean composition, visually striking",
                "lighting": "balanced cinematic lighting with clear subject separation",
                "photo": f"sharp focus, professional {orientation} composition",
                "medium": "digital image",
                "color_palette": ["#111827", "#2563EB", "#FACC15", "#FFFFFF", "#22C55E"],
            },
            "compositional_deconstruction": {
                "background": "A coherent background that supports the main subject without distracting from it.",
                "elements": [
                    {
                        "type": "obj",
                        "bbox": [80, 80, 920, 920],
                        "desc": prompt,
                        "color_palette": ["#111827", "#2563EB", "#FACC15", "#FFFFFF"],
                    }
                ],
            },
        }
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

    kind = (style or "none").strip()
    if kind == "art style":
        kind = "art_style"

    caption: dict[str, Any] = {}
    if prompt:
        caption["high_level_description"] = prompt

    if kind != "none":
        # Key order mirrors Ideogram4PromptBuilderKJ. For a chosen style, all
        # required style keys are present, even when their value is blank.
        sd: dict[str, Any] = {"aesthetics": aesthetics, "lighting": lighting}
        if kind == "photo":
            sd["photo"] = photo
            sd["medium"] = medium
        else:
            sd["medium"] = medium
            sd["art_style"] = art_style or photo
        caption["style_description"] = sd

    elements: list[dict[str, Any]] = []
    for box in regions or []:
        if not isinstance(box, dict):
            continue
        etype = "text" if box.get("type") == "text" else "obj"
        elem: dict[str, Any] = {"type": etype}
        if not box.get("nobbox"):
            elem["bbox"] = _norm_bbox_1000(box)
        if etype == "text":
            elem["text"] = str(box.get("text", "") or "")
        elem["desc"] = str(box.get("desc", "") or "")
        palette = _palette_upper(box.get("palette", []))
        if palette:
            elem["color_palette"] = palette[:5]
        elements.append(elem)

    caption["compositional_deconstruction"] = {
        "background": background,
        "elements": elements,
    }
    return _kj_dumps(caption)


PRESET_KEYS = {
    "Turbo 12": "V4_TURBO_12",
    "Default 20": "V4_DEFAULT_20",
    "Quality 48": "V4_QUALITY_48",
}


def choose_ideogram_preset(step_count: int, preset_name: str = "Default 20"):
    try:
        from ideogram4 import PRESETS
        key = PRESET_KEYS.get(preset_name or "", "")
        if key and key in PRESETS:
            return PRESETS[key], key
        if key == "CUSTOM":
            return None, "Custom"
        if step_count <= 12 and "V4_TURBO_12" in PRESETS:
            return PRESETS["V4_TURBO_12"], "V4_TURBO_12"
        if step_count >= 48 and "V4_QUALITY_48" in PRESETS:
            return PRESETS["V4_QUALITY_48"], "V4_QUALITY_48"
        if "V4_DEFAULT_20" in PRESETS:
            return PRESETS["V4_DEFAULT_20"], "V4_DEFAULT_20"
    except Exception:
        return None, "unavailable"
    return None, "unavailable"


def required_model_files(model_dir: Path) -> list[Path]:
    return [
        model_dir / "model_index.json",
        model_dir / "ideogram4_sdnq_pipeline.py",
        model_dir / "quantization_manifest.json",
        model_dir / "scheduler" / "scheduler_config.json",
        model_dir / "tokenizer" / "tokenizer.json",
        model_dir / "tokenizer" / "tokenizer_config.json",
        model_dir / "text_encoder" / "config.json",
        model_dir / "text_encoder" / "quantization_config.json",
        model_dir / "text_encoder" / "model.safetensors.index.json",
        model_dir / "text_encoder" / "model-00001-of-00002.safetensors",
        model_dir / "text_encoder" / "model-00002-of-00002.safetensors",
        model_dir / "transformer" / "config.json",
        model_dir / "transformer" / "quantization_config.json",
        model_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        model_dir / "unconditional_transformer" / "config.json",
        model_dir / "unconditional_transformer" / "quantization_config.json",
        model_dir / "unconditional_transformer" / "diffusion_pytorch_model.safetensors",
        model_dir / "vae" / "config.json",
        model_dir / "vae" / "quantization_config.json",
        model_dir / "vae" / "diffusion_pytorch_model.safetensors",
    ]


def check_model_complete(model_dir: Path) -> None:
    missing = [p for p in required_model_files(model_dir) if not p.exists()]
    if missing:
        rel = []
        for p in missing:
            try:
                rel.append(str(p.relative_to(model_dir)))
            except Exception:
                rel.append(str(p))
        installer = framevision_root() / "install_ideogram4_sdnq_uint4.bat"
        raise SystemExit(
            "Model folder is incomplete. Missing required file(s):\n  - "
            + "\n  - ".join(rel[:12])
            + ("\n  - ..." if len(rel) > 12 else "")
            + "\n\nRun the installer again from the FrameVision root so it can complete the model download:\n"
            + str(installer)
        )


def ensure_framevision_env() -> None:
    root = framevision_root()
    env_python = root / "environments" / ".images_models" / "python.exe"
    try:
        current = Path(sys.executable).resolve()
        target = env_python.resolve()
    except Exception:
        current = Path(sys.executable)
        target = env_python

    if current != target:
        if env_python.exists():
            print(f"[Ideogram4 SDNQ] Relaunching with FrameVision env Python: {env_python}")
            os.execv(str(env_python), [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(
            "This helper must run with FrameVision's environments/.images_models Python.\n"
            f"Expected: {env_python}\n"
            f"Current:  {sys.executable}\n"
            "Run the installer first, or use the helper BAT file."
        )


def load_ideogram4_sdnq_pipeline_class(model_dir: Path):
    pipeline_file = model_dir / "ideogram4_sdnq_pipeline.py"
    if not pipeline_file.exists():
        raise RuntimeError("Missing custom pipeline file in model folder: " + str(pipeline_file))

    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    spec = importlib.util.spec_from_file_location("framevision_ideogram4_sdnq_pipeline", str(pipeline_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for: " + str(pipeline_file))

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    pipeline_cls = getattr(module, "Ideogram4SDNQPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError("Custom pipeline file did not define Ideogram4SDNQPipeline: " + str(pipeline_file))
    return pipeline_cls


@dataclass
class GenerateConfig:
    prompt: str
    negative: str = ""
    model_dir: Path = default_model_dir()
    output: Path | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 4.0
    seed: int = -1
    compile_sdnq: bool = False
    raw_prompt: bool = False
    preset: str = "Default 20"
    text_encoder_cpu_offload: bool = False
    backend: str = "sdnq"
    gguf_dir: Path = default_gguf_dir()
    gguf_diffusion_file: str = DEFAULT_GGUF_FILES["diffusion"]
    gguf_unconditional_file: str = DEFAULT_GGUF_FILES["unconditional"]
    gguf_llm_file: str = DEFAULT_GGUF_FILES["llm"]
    gguf_vae_file: str = DEFAULT_GGUF_FILES["vae"]
    sd_cli_path: Path = default_sd_cli_path()
    gguf_max_vram: float = 0.0
    gguf_stream_layers: bool = False
    lora_enabled: bool = False
    lora_path: str = ""
    lora_scale: float = 1.0
    lora_apply_mode: str = "auto"


def _first_image(result):
    if hasattr(result, "images"):
        images = result.images
    else:
        images = result
    if isinstance(images, (list, tuple)):
        if not images:
            raise RuntimeError("Pipeline returned an empty image list.")
        return images[0]
    if hasattr(images, "save"):
        return images
    raise TypeError(f"Unexpected Ideogram pipeline result type: {type(result)!r}")


def _make_generator(torch_mod, seed: int):
    # Keep UI seed=-1 as the user-facing random mode, but still create an
    # actual CUDA generator with a real random seed. Use SystemRandom/secrets
    # instead of the normal random module so repeat sessions do not accidentally
    # reuse the same pseudo-random sequence.
    if seed is None or int(seed) < 0:
        used_seed = secrets.randbelow(2_147_483_647)
    else:
        used_seed = int(seed)

    gen = None
    try:
        gen = torch_mod.Generator(device="cuda").manual_seed(used_seed)
    except Exception:
        try:
            gen = torch_mod.Generator().manual_seed(used_seed)
        except Exception:
            gen = None
    return gen, used_seed


def _seed_all_torch_rngs(torch_mod, used_seed: int, log=None) -> None:
    # Ideogram's pipeline may ignore the generator kwarg and use global torch RNG
    # internally. Seed both CPU and CUDA global RNGs right before the pipeline call
    # so seed=-1 really changes the output every job.
    try:
        torch_mod.manual_seed(int(used_seed))
    except Exception as exc:
        if log:
            log(f"Could not seed torch CPU RNG: {exc}")
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.manual_seed(int(used_seed))
            torch_mod.cuda.manual_seed_all(int(used_seed))
    except Exception as exc:
        if log:
            log(f"Could not seed torch CUDA RNG: {exc}")


_PROGRESS_STEP_RE = re.compile(r"(?<!\d)(\d{1,5})/(\d{1,5})(?!\d)")
_QUEUE_STEP_LOG_RE = re.compile(r"^Step\s+(\d{1,5})/(\d{1,5})(?:\s*\|\s*(\d+(?:\.\d+)?)s/it)?$", re.IGNORECASE)
_QUEUE_ANY_STEP_RE = re.compile(r"(?<!\d)(\d{1,5})/(\d{1,5})(?!\d)")
_QUEUE_STEP_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s/it", re.IGNORECASE)
_PROGRESS_LOG_GUARD = threading.local()


def _progress_guard_active() -> bool:
    return bool(getattr(_PROGRESS_LOG_GUARD, "active", False))


class _ProgressLogStream:
    def __init__(self, base_stream, log):
        self.base_stream = base_stream
        self.log = log
        self._buffer = ""
        self._last_step: tuple[int, int] | None = None

    def write(self, data):
        text = str(data or "")
        if self.base_stream is not None:
            try:
                self.base_stream.write(text)
            except Exception:
                pass
        if not text:
            return 0
        self._buffer += text
        while True:
            split_idx = None
            for sep in ("\r", "\n"):
                idx = self._buffer.find(sep)
                if idx != -1 and (split_idx is None or idx < split_idx):
                    split_idx = idx
            if split_idx is None:
                break
            line = self._buffer[:split_idx]
            self._buffer = self._buffer[split_idx + 1:]
            self._process_line(line)
        return len(text)

    def flush(self):
        if self.base_stream is not None:
            try:
                self.base_stream.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return bool(self.base_stream.isatty())
        except Exception:
            return False

    def fileno(self):
        try:
            return self.base_stream.fileno()
        except Exception:
            raise OSError("No file descriptor available")

    @property
    def encoding(self):
        return getattr(self.base_stream, "encoding", "utf-8")

    def _process_line(self, line: str) -> None:
        if not self.log or _progress_guard_active():
            return
        text = str(line or "").strip()
        if not text:
            return
        if "%|" not in text and "it/s" not in text and "s/it" not in text:
            return
        matches = list(_PROGRESS_STEP_RE.finditer(text))
        if not matches:
            return
        current = total = None
        for match in reversed(matches):
            try:
                cur = int(match.group(1))
                tot = int(match.group(2))
            except Exception:
                continue
            if tot > 0 and 0 <= cur <= tot:
                current, total = cur, tot
                break
        if current is None or total is None:
            return
        step = (current, total)
        if step == self._last_step:
            return
        self._last_step = step
        speed_match = re.search(r"(\d+(?:\.\d+)?\s*(?:s/it|it/s))", text)
        extra = f" | {speed_match.group(1).replace(' ', '')}" if speed_match else ""
        _PROGRESS_LOG_GUARD.active = True
        try:
            self.log(f"Step {current}/{total}{extra}")
        finally:
            _PROGRESS_LOG_GUARD.active = False


@contextlib.contextmanager
def _capture_step_progress(log=None):
    if not callable(log):
        yield
        return
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_proxy = _ProgressLogStream(old_stdout, log)
    stderr_proxy = _ProgressLogStream(old_stderr, log)
    sys.stdout = stdout_proxy
    sys.stderr = stderr_proxy
    try:
        yield
    finally:
        try:
            stdout_proxy.flush()
        except Exception:
            pass
        try:
            stderr_proxy.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _pipe_device(pipe) -> str:
    try:
        device = getattr(pipe, "device", None)
        if device is not None:
            return str(device)
    except Exception:
        pass
    return "cuda"


def _text_encoder_module(pipe):
    try:
        text_encoder = getattr(pipe, "text_encoder", None)
        language_model = getattr(text_encoder, "language_model", None)
        if language_model is not None and hasattr(language_model, "to"):
            return language_model
        if text_encoder is not None and hasattr(text_encoder, "to"):
            return text_encoder
    except Exception:
        pass
    return None


def _safe_cuda_cleanup() -> None:
    try:
        gc.collect()
    except Exception:
        pass
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


def _lora_path_text(value: Any) -> str:
    return str(value or "").strip()


def _lora_is_enabled(cfg: GenerateConfig) -> bool:
    return bool(getattr(cfg, "lora_enabled", False)) and bool(_lora_path_text(getattr(cfg, "lora_path", "")))


def _lora_scale_value(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 1.0


def _lora_apply_mode_value(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    return text if text in {"auto", "immediately", "at_runtime"} else "auto"


def _lora_adapter_name(path_text: str) -> str:
    text = _lora_path_text(path_text)
    if not text:
        return "framevision_lora"
    try:
        p = Path(text)
        if p.suffix:
            name = p.stem
        else:
            name = p.name
    except Exception:
        name = text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "framevision_lora"


def _configure_diffusers_lora(pipe, enabled: bool, lora_path: str, lora_scale: float = 1.0, log=None) -> None:
    """Load/unload one LoRA adapter on Diffusers-like Ideogram pipelines.

    The official TurboTime LoRA card uses:
        pipe.load_lora_weights("ostris/ideogram_4_turbotime_lora")

    This wrapper keeps cached pipelines predictable by unloading the previous
    FrameVision LoRA before loading a new one.
    """
    enabled = bool(enabled) and bool(_lora_path_text(lora_path))
    current_key = getattr(pipe, "_framevision_lora_key", "")
    next_key = f"{_lora_path_text(lora_path)}|{_lora_scale_value(lora_scale):.6g}" if enabled else ""

    if current_key == next_key:
        if enabled and log:
            log(f"LoRA already active: {_lora_path_text(lora_path)} @ {_lora_scale_value(lora_scale):.3g}")
        return

    if current_key:
        unloaded = False
        for method_name in ("unload_lora_weights", "delete_adapters", "disable_lora"):
            method = getattr(pipe, method_name, None)
            if not callable(method):
                continue
            try:
                if method_name == "delete_adapters":
                    method(["framevision_lora"])
                else:
                    method()
                unloaded = True
                if log:
                    log("Previous LoRA unloaded from cached pipeline.")
                break
            except TypeError:
                try:
                    method("framevision_lora")
                    unloaded = True
                    if log:
                        log("Previous LoRA unloaded from cached pipeline.")
                    break
                except Exception:
                    pass
            except Exception as exc:
                if log:
                    log(f"Could not unload previous LoRA with {method_name}: {exc}")
        if not unloaded and log:
            log("Previous LoRA could not be unloaded cleanly; continuing with best effort.")

    setattr(pipe, "_framevision_lora_key", "")

    if not enabled:
        return

    loader = getattr(pipe, "load_lora_weights", None)
    if not callable(loader):
        if log:
            log("LoRA requested, but this pipeline has no load_lora_weights() method. LoRA ignored.")
        return

    path_text = _lora_path_text(lora_path)
    adapter_name = "framevision_lora"
    try:
        p = Path(path_text)
        if p.exists() and p.is_file():
            loader(str(p.parent), weight_name=p.name, adapter_name=adapter_name)
        elif p.exists() and p.is_dir():
            loader(str(p), adapter_name=adapter_name)
        else:
            # Hugging Face repo id, e.g. ostris/ideogram_4_turbotime_lora
            loader(path_text, adapter_name=adapter_name)
    except TypeError:
        try:
            loader(path_text)
        except Exception as exc:
            if log:
                log(f"LoRA load failed: {exc}")
            return
    except Exception as exc:
        if log:
            log(f"LoRA load failed: {exc}")
        return

    scale = _lora_scale_value(lora_scale)
    try:
        set_adapters = getattr(pipe, "set_adapters", None)
        if callable(set_adapters):
            set_adapters([adapter_name], adapter_weights=[scale])
    except Exception as exc:
        if log:
            log(f"LoRA loaded, but adapter strength could not be set: {exc}")

    setattr(pipe, "_framevision_lora_key", next_key)
    if log:
        log(f"LoRA loaded: {path_text} @ {scale:.3g}")


def _sd_cli_lora_dir_flag(help_text: str) -> str:
    """Pick the installed sd-cli LoRA folder flag from --help output.

    Different stable-diffusion.cpp/sd-cli builds have used slightly different
    LoRA option names. Do not guess blindly: inspect the real binary help text
    at runtime and only emit a flag that exists in that build.
    """
    help_text = str(help_text or "")
    candidates = ("--lora-model-dir", "--lora-dir", "--lora-models-dir", "--lora-path")
    for flag in candidates:
        if flag in help_text:
            return flag
    return ""


def _sd_cli_lora_help_lines(help_text: str) -> list[str]:
    lines = []
    for line in str(help_text or "").splitlines():
        if "lora" in line.lower():
            line = line.strip()
            if line:
                lines.append(line)
    return lines[:20]


def _gguf_lora_command_parts(lora_path: str, lora_scale: float = 1.0, help_text: str = "") -> tuple[str, str, str, str] | None:
    """Return (dir_flag, model_dir, lora_name, prompt_tag) for sd-cli LoRA.

    stable-diffusion.cpp-style LoRA is normally selected with a local LoRA
    folder flag plus a prompt tag like <lora:name:strength>. This helper first
    checks the installed sd-cli --help output so FrameVision does not pass a
    LoRA flag unsupported by the user's binary.
    """
    dir_flag = _sd_cli_lora_dir_flag(help_text)
    if not dir_flag:
        return None
    path_text = _lora_path_text(lora_path)
    if not path_text:
        return None
    p = Path(path_text)
    if p.exists() and p.is_file():
        model_dir = str(p.parent)
        name = p.stem
    elif p.exists() and p.is_dir():
        model_dir = str(p)
        safes = sorted(list(p.glob("*.safetensors")) + list(p.glob("*.ckpt")), key=lambda x: x.name.lower())
        name = safes[0].stem if safes else p.name
    else:
        return None
    name = _lora_adapter_name(name)
    scale = _lora_scale_value(lora_scale)
    return dir_flag, model_dir, name, f"<lora:{name}:{scale:.4g}>"


def _inject_lora_tag_into_prompt(prompt_text: str, tag: str) -> str:
    text = str(prompt_text or "")
    tag = str(tag or "").strip()
    if not tag:
        return text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            hld = str(data.get("high_level_description", "") or "")
            data["high_level_description"] = (hld + " " + tag).strip() if hld else tag
            return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        pass
    return (text + " " + tag).strip()


def _module_device(module) -> str:
    try:
        for p in module.parameters(recurse=True):
            return str(p.device)
    except Exception:
        pass
    try:
        for b in module.buffers(recurse=True):
            return str(b.device)
    except Exception:
        pass
    return "unknown"


def _move_text_encoder(pipe, device: str, log=None, reason: str = "") -> bool:
    module = _text_encoder_module(pipe)
    if module is None:
        if log:
            log("Text encoder CPU offload: no text encoder module found to move.")
        return False
    try:
        before = _module_device(module)
        target = str(device)
        # Avoid repeated .to(cpu)/.to(cuda) calls. Repeated copies were the likely
        # source of the DDR/VRAM creep seen after several queued images.
        if before != "unknown" and before.lower().startswith(target.lower().split(":", 1)[0]):
            if log:
                suffix = f" ({reason})" if reason else ""
                log(f"Text encoder already on {before}{suffix}.")
            return True
        module.to(device)
        after = _module_device(module)
        if str(device).lower().startswith("cpu"):
            _safe_cuda_cleanup()
        if log:
            suffix = f" ({reason})" if reason else ""
            log(f"Text encoder moved {before} -> {after}{suffix}.")
        return True
    except Exception as exc:
        if log:
            log(f"Text encoder move to {device} failed: {exc}")
        return False


def _configure_text_encoder_cpu_offload(pipe, enabled: bool, log=None) -> None:
    setattr(pipe, "_framevision_text_encoder_cpu_offload_enabled", bool(enabled))
    module = _text_encoder_module(pipe)
    if module is None:
        if enabled and log:
            log("Text encoder CPU offload: no compatible text encoder module found.")
        return

    if not enabled:
        # If the user turns the option off after a previous run, make sure the
        # cached pipeline is back on CUDA before the next generation.
        _move_text_encoder(pipe, _pipe_device(pipe), log=log, reason="offload disabled")
        return

    if getattr(pipe, "_framevision_text_encoder_cpu_offload_hooked", False):
        if log:
            log("Text encoder CPU offload: safe hook already installed.")
        return

    # Only wrap known prompt-encoding methods. Do NOT wrap language_model.forward.
    # Forward can be called by lower-level model code and moving the full Qwen module
    # back and forth there caused memory to creep upward over repeated jobs.
    method_names = (
        "encode_prompt",
        "_encode_prompt",
        "encode_prompts",
        "_encode_prompts",
        "get_text_embeddings",
        "_get_text_embeddings",
        "prepare_prompt",
        "_prepare_prompt",
    )
    for name in method_names:
        original = getattr(pipe, name, None)
        if callable(original):
            def wrapped_encode(*args, __original=original, __name=name, **kwargs):
                if getattr(pipe, "_framevision_text_encoder_cpu_offload_enabled", False):
                    _move_text_encoder(pipe, _pipe_device(pipe), log=log, reason=f"before {__name}")
                result = __original(*args, **kwargs)
                if getattr(pipe, "_framevision_text_encoder_cpu_offload_enabled", False):
                    _move_text_encoder(pipe, "cpu", log=log, reason=f"after {__name}")
                return result
            setattr(pipe, name, wrapped_encode)
            setattr(pipe, "_framevision_text_encoder_cpu_offload_hooked", True)
            setattr(pipe, "_framevision_text_encoder_cpu_offload_safe", True)
            if log:
                log(f"Text encoder CPU offload: wrapped safe pipeline method {name}.")
            return

    setattr(pipe, "_framevision_text_encoder_cpu_offload_hooked", True)
    setattr(pipe, "_framevision_text_encoder_cpu_offload_safe", False)
    if log:
        log("Text encoder CPU offload: no safe prompt-encode method found; forward fallback disabled to avoid memory creep. Using post-job cleanup only.")


def get_pipeline(model_dir: Path, compile_sdnq: bool = False, text_encoder_cpu_offload: bool = False, log=None):
    import torch
    import sdnq  # noqa: F401

    if not torch.cuda.is_available():
        raise SystemExit("CPU fallback detected. Aborting.")

    model_dir = model_dir.resolve()
    key = (str(model_dir), bool(compile_sdnq))
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is not None:
        if log:
            log("Pipeline cache hit. Reusing already-loaded pipeline.")
        _configure_text_encoder_cpu_offload(pipe, bool(text_encoder_cpu_offload), log=log)
        return pipe

    if log:
        log(f"Checking model folder: {model_dir}")
    check_model_complete(model_dir)
    if log:
        log("Model folder looks complete.")
        log("Importing bundled custom pipeline class...")
    pipeline_cls = load_ideogram4_sdnq_pipeline_class(model_dir)
    if log:
        log(f"Pipeline class: {pipeline_cls.__module__}.{pipeline_cls.__name__}")
        log("Loading pipeline with device=cuda, dtype=torch.bfloat16, dequantize_fp32=False")
        log(f"Compile SDNQ / quantized matmul: {bool(compile_sdnq)}")
    load_t0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(
        str(model_dir),
        device="cuda",
        dtype=torch.bfloat16,
        use_quantized_matmul=bool(compile_sdnq),
        dequantize_fp32=False,
    )
    load_dt = time.perf_counter() - load_t0
    if log:
        log(f"Pipeline load finished in {load_dt:.2f}s")
    _configure_text_encoder_cpu_offload(pipe, bool(text_encoder_cpu_offload), log=log)
    _PIPELINE_CACHE[key] = pipe
    return pipe



def _backend_name(value: str) -> str:
    value = str(value or "sdnq").strip().lower()
    if value in {"gguf", "sd-cli", "sdcli", "sdcpp"}:
        return "gguf"
    return "sdnq"


def _quote_cmd_part(value: str) -> str:
    value = str(value)
    if not value:
        return '""'
    if re.search(r'\s|["^&|<>]', value):
        return '"' + value.replace('"', r'\"') + '"'
    return value


def _format_cmd_for_log(cmd: list[str]) -> str:
    return " ".join(_quote_cmd_part(c) for c in cmd)


def _check_sd_cli_supports_ideogram(sd_cli_path: Path) -> str:
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            [str(sd_cli_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            startupinfo=startupinfo,
            errors="replace",
        )
    except TypeError:
        result = subprocess.run(
            [str(sd_cli_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            startupinfo=startupinfo,
        )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if "--uncond-diffusion-model" not in help_text:
        raise SystemExit(
            "sd-cli.exe was found, but it is too old for Ideogram 4 GGUF. "
            "It must support --uncond-diffusion-model. Run the GGUF installer again."
        )
    return help_text


def print_gguf_self_test(gguf_dir: Path | None = None, sd_cli_path: Path | None = None) -> int:
    gguf_dir = Path(gguf_dir or default_gguf_dir()).resolve()
    sd_cli_path = Path(sd_cli_path or default_sd_cli_path()).resolve()
    check_gguf_complete(gguf_dir, sd_cli_path)
    _check_sd_cli_supports_ideogram(sd_cli_path)
    print(f"[Ideogram4 GGUF] Self-test OK.")
    print(f"[Ideogram4 GGUF] sd-cli: {sd_cli_path}")
    print(f"[Ideogram4 GGUF] model folder: {gguf_dir}")
    return 0


def generate_once_gguf(cfg: GenerateConfig, log=None) -> tuple[Path, int, str]:
    started = time.perf_counter()
    gguf_dir = Path(cfg.gguf_dir or default_gguf_dir()).resolve()
    sd_cli_path = Path(cfg.sd_cli_path or default_sd_cli_path()).resolve()
    output_target = Path(cfg.output).resolve() if cfg.output else output_path(prefix="ideogram4_gguf")

    check_gguf_complete(
        gguf_dir,
        sd_cli_path,
        cfg.gguf_diffusion_file,
        cfg.gguf_unconditional_file,
        cfg.gguf_llm_file,
        cfg.gguf_vae_file,
    )
    sd_cli_help_text = _check_sd_cli_supports_ideogram(sd_cli_path)
    files = required_gguf_files(
        gguf_dir,
        cfg.gguf_diffusion_file,
        cfg.gguf_unconditional_file,
        cfg.gguf_llm_file,
        cfg.gguf_vae_file,
    )

    final_prompt = cfg.prompt if (cfg.raw_prompt or looks_like_json_prompt(cfg.prompt)) else build_structured_caption(cfg.prompt, cfg.width, cfg.height)
    prompt_note = "Wrapped plain prompt into Ideogram JSON caption." if final_prompt != cfg.prompt else "Using raw/JSON prompt as provided."

    gguf_lora_parts = None
    if _lora_is_enabled(cfg):
        gguf_lora_parts = _gguf_lora_command_parts(cfg.lora_path, cfg.lora_scale, sd_cli_help_text)
        if gguf_lora_parts is not None:
            _lora_flag, _lora_dir, _lora_name, _lora_tag = gguf_lora_parts
            final_prompt = _inject_lora_tag_into_prompt(final_prompt, _lora_tag)
            prompt_note += " LoRA enabled."
        else:
            if not _sd_cli_lora_dir_flag(sd_cli_help_text):
                prompt_note += " LoRA was requested but could not be enabled."
            else:
                prompt_note += " LoRA was requested but no valid local LoRA file/folder was selected."

    output_target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(sd_cli_path),
        "--diffusion-model", str(files["diffusion"]),
        "--uncond-diffusion-model", str(files["unconditional"]),
        "--llm", str(files["llm"]),
        "--vae", str(files["vae"]),
        "--vae-format", "flux2",
        "--width", str(int(cfg.width)),
        "--height", str(int(cfg.height)),
        "--steps", str(int(cfg.steps)),
        "--cfg-scale", str(float(cfg.guidance)),
        "--seed", str(int(cfg.seed)),
        "-p", final_prompt,
        "-o", str(output_target),
    ]
    if cfg.negative.strip():
        cmd.extend(["--negative-prompt", cfg.negative.strip()])
    try:
        max_vram = float(cfg.gguf_max_vram or 0.0)
    except Exception:
        max_vram = 0.0
    if max_vram != 0.0:
        cmd.extend(["--max-vram", str(max_vram)])
    if bool(cfg.gguf_stream_layers):
        cmd.append("--stream-layers")
    if gguf_lora_parts is not None:
        cmd.extend([gguf_lora_parts[0], gguf_lora_parts[1]])
        cmd.extend(["--lora-apply-mode", _lora_apply_mode_value(getattr(cfg, "lora_apply_mode", "auto"))])

    if log:
        log("Backend: GGUF / sd-cli")
        log(f"sd-cli: {sd_cli_path}")
        log(f"GGUF dir: {gguf_dir}")
        log(f"Diffusion GGUF: {files['diffusion'].name}")
        log(f"Unconditional GGUF: {files['unconditional'].name}")
        log(f"LLM GGUF: {files['llm'].name}")
        log(f"VAE: {files['vae'].name}")
        log(f"Output target: {output_target}")
        log(f"Requested size: {cfg.width}x{cfg.height}")
        log(f"Requested steps: {cfg.steps}")
        log(f"Requested cfg/guidance: {cfg.guidance}")
        log(f"Requested seed: {cfg.seed}")
        log(f"Negative prompt present: {'yes' if cfg.negative.strip() else 'no'}")
        log(f"Raw prompt mode: {cfg.raw_prompt}")
        log(f"Max VRAM: {max_vram if max_vram else 'off'}")
        log(f"Stream layers: {bool(cfg.gguf_stream_layers)}")
        if _lora_is_enabled(cfg):
            log(f"LoRA: {cfg.lora_path} @ {_lora_scale_value(cfg.lora_scale):.3g}")
            if gguf_lora_parts is None:
                log("LoRA could not be enabled for this backend/path.")
        log(prompt_note)
        log("Final prompt payload sent to sd-cli:")
        log(_preview_text(final_prompt, 4000))
        log("Command:")
        log(_format_cmd_for_log(cmd))

    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(framevision_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not start sd-cli.exe: {exc}") from exc

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line and log:
            log("[sd-cli] " + line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"sd-cli.exe failed with exit code {rc}")

    if not output_target.exists():
        raise RuntimeError(f"sd-cli.exe finished, but output file was not created: {output_target}")

    total_dt = time.perf_counter() - started
    if log:
        log(f"Image saved: {output_target}")
        log(f"Total run time: {total_dt:.2f}s")

    used_seed = int(cfg.seed)
    notes = [prompt_note, "Backend: GGUF / sd-cli", f"Total run time: {total_dt:.2f}s"]
    return output_target, used_seed, "\n".join(notes)


def generate_once(cfg: GenerateConfig, log=None) -> tuple[Path, int, str]:
    if _backend_name(getattr(cfg, "backend", "sdnq")) == "gguf":
        return generate_once_gguf(cfg, log=log)

    import torch
    import sdnq  # noqa: F401

    if not torch.cuda.is_available():
        raise SystemExit("CPU fallback detected. Aborting.")

    started = time.perf_counter()
    model_dir = Path(cfg.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model folder not found: {model_dir}")

    output_target = Path(cfg.output).resolve() if cfg.output else output_path(prefix="ideogram4_sdnq")

    if log:
        log(f"Torch: {torch.__version__} / CUDA build: {torch.version.cuda}")
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"Model dir: {model_dir}")
        log(f"Output target: {output_target}")
        log(f"Requested preset: {cfg.preset}")
        log(f"Requested size: {cfg.width}x{cfg.height}")
        log(f"Requested steps: {cfg.steps}")
        log(f"Requested guidance: {cfg.guidance}")
        log(f"Requested seed: {cfg.seed}")
        log(f"Negative prompt present: {'yes' if cfg.negative.strip() else 'no'}")
        log(f"Raw prompt mode: {cfg.raw_prompt}")
        log(f"Text encoder CPU offload: {bool(cfg.text_encoder_cpu_offload)}")
        log(f"LoRA enabled: {bool(_lora_is_enabled(cfg))}")
        if _lora_is_enabled(cfg):
            log(f"LoRA path/repo: {cfg.lora_path}")
            log(f"LoRA strength: {_lora_scale_value(cfg.lora_scale):.3g}")

    pipe = get_pipeline(model_dir, compile_sdnq=cfg.compile_sdnq, text_encoder_cpu_offload=cfg.text_encoder_cpu_offload, log=log)
    _configure_diffusers_lora(pipe, _lora_is_enabled(cfg), cfg.lora_path, cfg.lora_scale, log=log)

    generator, used_seed = _make_generator(torch, cfg.seed)
    final_prompt = cfg.prompt if (cfg.raw_prompt or looks_like_json_prompt(cfg.prompt)) else build_structured_caption(cfg.prompt, cfg.width, cfg.height)
    prompt_note = "Wrapped plain prompt into Ideogram JSON caption." if final_prompt != cfg.prompt else "Using raw/JSON prompt as provided."

    if log:
        log(prompt_note)
        log("Original prompt:")
        log(_preview_text(cfg.prompt, 1500))
        if cfg.negative.strip():
            log("Negative prompt:")
            log(_preview_text(cfg.negative.strip(), 800))
        log("Final prompt payload sent to model:")
        log(_preview_text(final_prompt, 4000))
        log(f"Seed used for this run: {used_seed}")

    preset, preset_key = choose_ideogram_preset(cfg.steps, cfg.preset)
    requested_kwargs = {
        "width": int(cfg.width),
        "height": int(cfg.height),
        "num_steps": int(cfg.steps),
        "guidance_scale": float(cfg.guidance),
        "raise_on_caption_issues": False,
    }
    if preset is not None:
        for name in ("guidance_schedule", "mu", "std"):
            if hasattr(preset, name):
                requested_kwargs[name] = getattr(preset, name)
        if hasattr(preset, "num_steps"):
            requested_kwargs["num_steps"] = int(getattr(preset, "num_steps"))

    if log:
        log(f"Resolved preset: {preset_key}")
        if preset is not None:
            details = []
            for name in ("num_steps", "guidance_schedule", "mu", "std"):
                if hasattr(preset, name):
                    details.append(f"{name}={getattr(preset, name)!r}")
            if details:
                log("Preset details: " + ", ".join(details))

    # Some Ideogram builds accept a generator, some accept a seed, and some ignore
    # both and rely on global torch RNG. Provide both safe options; unsupported
    # options are filtered out below.
    if generator is not None:
        requested_kwargs["generator"] = generator
    requested_kwargs["seed"] = int(used_seed)
    if cfg.negative.strip():
        requested_kwargs["negative_prompt"] = cfg.negative.strip()

    try:
        sig = inspect.signature(pipe.__call__)
        params = sig.parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        call_kwargs = dict(requested_kwargs) if accepts_var_kwargs else {k: v for k, v in requested_kwargs.items() if k in params}
        if "guidance" in params and "guidance_scale" not in params:
            call_kwargs["guidance"] = float(cfg.guidance)
        dropped = [k for k in requested_kwargs if k not in call_kwargs]
        dropped_note = ""
        if dropped:
            dropped_note = "Ignoring unsupported call option(s): " + ", ".join(dropped)
    except Exception:
        call_kwargs = dict(requested_kwargs)
        dropped_note = ""

    if log:
        try:
            sig = inspect.signature(pipe.__call__)
            log(f"Pipeline __call__ signature: {sig}")
        except Exception as exc:
            log(f"Could not inspect pipeline signature: {exc}")
        safe_kwargs = {k: ("<torch.Generator>" if k == "generator" else v) for k, v in call_kwargs.items()}
        log(f"Effective call kwargs: {_preview_text(repr(safe_kwargs), 2000)}")
        if dropped_note:
            log(dropped_note)

    with torch.inference_mode():
        run_t0 = time.perf_counter()
        _seed_all_torch_rngs(torch, used_seed, log=log)
        if log:
            log(f"Torch RNGs seeded for this run: {used_seed}")
        if log:
            log(f"Step logging active for {int(call_kwargs.get('num_steps', cfg.steps))} steps when the pipeline reports tqdm progress.")
        try:
            with _capture_step_progress(log=log):
                result = pipe(final_prompt, **call_kwargs)
        except TypeError as exc:
            minimal = {"width": int(cfg.width), "height": int(cfg.height)}
            try:
                sig = inspect.signature(pipe.__call__)
                minimal = {k: v for k, v in minimal.items() if k in sig.parameters}
            except Exception:
                pass
            if log:
                log(f"Pipeline rejected one or more options; retrying minimal call. ({exc})")
                log(f"Minimal retry kwargs: {minimal}")
            _seed_all_torch_rngs(torch, used_seed, log=log)
            with _capture_step_progress(log=log):
                result = pipe(final_prompt, **minimal)
            dropped_note = (dropped_note + "\n" if dropped_note else "") + f"Pipeline rejected one or more options; retried minimal call. ({exc})"

        run_dt = time.perf_counter() - run_t0
        if log:
            log(f"Generation call finished in {run_dt:.2f}s")
            log(f"Raw pipeline result type: {type(result)!r}")

        image = _first_image(result)
        if log:
            log(f"First image object type: {type(image)!r}")

    out = output_target
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)

    # Drop temporary references before returning to the queue/GUI. This does not
    # change peak VRAM during the run, but it prevents queued generations from
    # accumulating stale tensors/cached blocks between jobs.
    try:
        del result
        del image
    except Exception:
        pass
    _safe_cuda_cleanup()

    total_dt = time.perf_counter() - started
    if log:
        log(f"Image saved: {out}")
        log(f"Total run time: {total_dt:.2f}s")

    notes = [prompt_note, f"Preset: {preset_key}", f"Total run time: {total_dt:.2f}s"]
    if _lora_is_enabled(cfg):
        notes.append(f"LoRA: {cfg.lora_path} @ {_lora_scale_value(cfg.lora_scale):.3g}")
    if dropped_note:
        notes.append(dropped_note)
    return out, used_seed, "\n".join(notes)

def print_self_test() -> int:
    import torch
    import sdnq  # noqa: F401

    if not torch.cuda.is_available():
        raise SystemExit("CPU fallback detected. Aborting.")

    print(f"[Ideogram4 SDNQ] Helper self-test OK. Torch: {torch.__version__} / CUDA build: {torch.version.cuda}")
    print(f"[Ideogram4 SDNQ] GPU: {torch.cuda.get_device_name(0)}")
    return 0


def cli_main(args: argparse.Namespace) -> int:
    ensure_framevision_env()

    if args.self_test:
        if _backend_name(args.backend) == "gguf":
            return print_gguf_self_test(Path(args.gguf_dir), Path(args.sd_cli_path))
        return print_self_test()

    if not args.prompt.strip():
        raise SystemExit("Missing --prompt. Launch without arguments to open the simple helper window.")

    if _backend_name(args.backend) == "gguf":
        print(f"[Ideogram4 GGUF] Loading sd-cli backend: {Path(args.gguf_dir).resolve()}")
    else:
        import torch
        print(f"[Ideogram4 SDNQ] Torch: {torch.__version__} / CUDA build: {torch.version.cuda}")
        print(f"[Ideogram4 SDNQ] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Ideogram4 SDNQ] Loading: {Path(args.model_dir).resolve()}")
        print("[Ideogram4 SDNQ] Using bundled custom pipeline: ideogram4_sdnq_pipeline.py")

    cfg = GenerateConfig(
        prompt=args.prompt,
        negative=args.negative,
        model_dir=Path(args.model_dir),
        output=Path(args.output) if args.output else None,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        preset=args.preset,
        seed=args.seed,
        compile_sdnq=args.compile_sdnq,
        raw_prompt=args.raw_prompt,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        backend=_backend_name(args.backend),
        gguf_dir=Path(args.gguf_dir),
        gguf_diffusion_file=str(args.gguf_diffusion_file),
        gguf_unconditional_file=str(args.gguf_unconditional_file),
        gguf_llm_file=str(args.gguf_llm_file),
        gguf_vae_file=str(args.gguf_vae_file),
        sd_cli_path=Path(args.sd_cli_path),
        gguf_max_vram=float(args.gguf_max_vram),
        gguf_stream_layers=bool(args.gguf_stream_layers),
        lora_enabled=bool(args.lora_enabled),
        lora_path=str(args.lora_path),
        lora_scale=float(args.lora_scale),
        lora_apply_mode=_lora_apply_mode_value(args.lora_apply_mode),
    )
    log_prefix = "[Ideogram4 GGUF]" if _backend_name(args.backend) == "gguf" else "[Ideogram4 SDNQ]"
    out, used_seed, notes = generate_once(cfg, log=lambda m: print(f"{log_prefix} {m}"))
    if notes:
        for line in notes.splitlines():
            if line.strip():
                print(f"{log_prefix} {line}")
    print(f"{log_prefix} Seed used: {used_seed}")
    print(f"{log_prefix} Saved: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple helper for WaveCut Ideogram 4 SDNQ UInt4")
    parser.add_argument("--gui", action="store_true", help="Open the simple helper window")
    parser.add_argument("--prompt", default="", help="Generation prompt")
    parser.add_argument("--raw-prompt", action="store_true", help="Send prompt exactly as typed instead of wrapping plain text into Ideogram JSON")
    parser.add_argument("--self-test", action="store_true", help="Import/CUDA/model-location test only; no generation")
    parser.add_argument("--negative", default="")
    parser.add_argument("--model-dir", default=str(default_model_dir()))
    parser.add_argument("--output", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--preset", default="Default 20", choices=["Turbo 12", "Default 20", "Quality 48", "Custom"], help="Ideogram preset exposed by ideogram4.PRESETS")
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--compile-sdnq", action="store_true", help="Try SDNQ quantized matmul / compile acceleration")
    parser.add_argument("--text-encoder-cpu-offload", action="store_true", help="Experimental: move the text encoder to CPU after prompt encoding to reduce denoise VRAM")
    parser.add_argument("--backend", default="sdnq", choices=["sdnq", "gguf"], help="Generation backend: sdnq quants or gguf sd-cli")
    parser.add_argument("--gguf-dir", default=str(default_gguf_dir()), help="Folder containing Ideogram 4 GGUF files")
    parser.add_argument("--gguf-diffusion-file", default=DEFAULT_GGUF_FILES["diffusion"], help="Main Ideogram diffusion GGUF filename")
    parser.add_argument("--gguf-unconditional-file", default=DEFAULT_GGUF_FILES["unconditional"], help="Unconditional Ideogram diffusion GGUF filename")
    parser.add_argument("--gguf-llm-file", default=DEFAULT_GGUF_FILES["llm"], help="Qwen LLM GGUF filename")
    parser.add_argument("--gguf-vae-file", default=DEFAULT_GGUF_FILES["vae"], help="Flux2 VAE filename")
    parser.add_argument("--sd-cli-path", default=str(default_sd_cli_path()), help="Path to sd-cli.exe")
    parser.add_argument("--gguf-max-vram", type=float, default=0.0, help="Optional sd-cli --max-vram value. 0 disables it.")
    parser.add_argument("--gguf-stream-layers", action="store_true", help="Use sd-cli --stream-layers with --max-vram")
    parser.add_argument("--lora-enabled", action="store_true", help="Load one LoRA adapter. Official TurboTime path is Diffusers/SDNQ; GGUF is best-effort when sd-cli supports LoRA tags.")
    parser.add_argument("--lora-path", default="", help="LoRA repo id, folder, or .safetensors/.ckpt file. Example: ostris/ideogram_4_turbotime_lora")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA adapter strength. 1.0 is the normal default.")
    parser.add_argument("--lora-apply-mode", default="auto", choices=["auto", "immediately", "at_runtime"], help="sd-cli LoRA apply mode. Auto uses runtime mode for quantized models.")
    return parser


THEME_LABELS: dict[str, str] = {
    "off": "Themes off",
    "dark": "Dark",
    "light": "Light",
    "purple_nebula": "Purple Delight",
    "graphite_midnight": "Graphite Midnight",
    "dracula": "Violet Dusk",
}
THEME_LABEL_TO_KEY: dict[str, str] = {label: key for key, label in THEME_LABELS.items()}
THEME_CHOICES: list[str] = list(THEME_LABELS.values())


def normalize_theme_key(value: Any) -> str:
    text = str(value or "dark").strip()
    if text in THEME_LABELS:
        return text
    if text in THEME_LABEL_TO_KEY:
        return THEME_LABEL_TO_KEY[text]
    lowered = text.lower().replace(" ", "_").replace("-", "_")
    return lowered if lowered in THEME_LABELS else "dark"


GUI_DEFAULTS: dict[str, Any] = {
    "width": "1024",
    "height": "1024",
    "steps": "20",
    "guidance": "4.0",
    "preset": "Default 20",
    "seed": "-1",
    "backend": "gguf",
    "model_dir": str(default_model_dir()),
    "gguf_dir": str(default_gguf_dir()),
    "gguf_diffusion_file": DEFAULT_GGUF_FILES["diffusion"],
    "gguf_unconditional_file": DEFAULT_GGUF_FILES["unconditional"],
    "gguf_llm_file": DEFAULT_GGUF_FILES["llm"],
    "gguf_vae_file": DEFAULT_GGUF_FILES["vae"],
    "sd_cli_path": str(default_sd_cli_path()),
    "gguf_max_vram": "0",
    "gguf_stream_layers": False,
    "lora_enabled": False,
    "lora_path": "",
    "lora_scale": "1.0",
    "lora_apply_mode": "auto",
    "output_dir": str(default_output_dir()),
    "compile_sdnq": False,
    "text_encoder_cpu_offload": False,
    "raw_prompt": False,
    "layout_prompt_enabled": False,
    "layout_background": "",
    "layout_style": "photo",
    "layout_style_photo": "",
    "layout_aesthetics": "",
    "layout_lighting": "",
    "layout_medium": "",
    "layout_regions": [],
    "layout_background_preview_path": "",
    "layout_background_preview_live": False,
    "layout_background_preview_opacity": 35,
    "theme": "dark",
    "generate_split": "0.60",
}


def load_gui_settings() -> dict[str, Any]:
    data = dict(GUI_DEFAULTS)
    saved = load_saved_settings()
    for key in GUI_DEFAULTS:
        if key in saved:
            data[key] = saved[key]
    return data


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tkm = __import__("tkinter")
        tw = self.tipwindow = tkm.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = __import__('tkinter').Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            background="#ffffe0",
            foreground="#000000",
            wraplength=360,
        )
        label.pack()

    def _hide(self, event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None




def queue_state_path() -> Path:
    out = framevision_root() / "presets" / "setsave" / "ideogram_queue.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_queue_state() -> dict[str, Any]:
    path = queue_state_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                jobs = data.get("jobs", [])
                if isinstance(jobs, list):
                    data["jobs"] = [j for j in jobs if isinstance(j, dict)]
                    return data
        except Exception:
            pass
    return {"version": 1, "jobs": []}


def save_queue_state(data: dict[str, Any]) -> None:
    path = queue_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data or {})
    data["version"] = 1
    data.setdefault("jobs", [])
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _format_dt(value: str | None) -> str:
    dt = _parse_iso(value)
    if not dt:
        return ""
    return dt.strftime("%H:%M:%S")


def _format_duration(seconds: float | int | None) -> str:
    try:
        seconds = max(0, int(seconds or 0))
    except Exception:
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _short_prompt(prompt: str, max_len: int = 90) -> str:
    text = " ".join(str(prompt or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"




def _list_gguf_files_for_role(folder: Path, role: str) -> list[str]:
    """Return GGUF/VAE filenames for the small embedded PySide UI."""
    try:
        folder = Path(folder)
        if not folder.exists():
            return []
        names: list[str] = []
        for f in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if not f.is_file():
                continue
            r = _gguf_name_role(f.name)
            if role == "vae":
                if r == "vae":
                    names.append(f.name)
            elif role == r:
                names.append(f.name)
        return names
    except Exception:
        return []


try:
    from PySide6 import QtCore as _FVQtCore, QtGui as _FVQtGui, QtWidgets as _FVQtWidgets
except Exception:
    _FVQtCore = None  # type: ignore
    _FVQtGui = None  # type: ignore
    _FVQtWidgets = None  # type: ignore


if _FVQtWidgets is not None:
    class RegionPromptCanvas(_FVQtWidgets.QWidget):
        selectionChanged = _FVQtCore.Signal(int)
        regionsChanged = _FVQtCore.Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self._regions: list[dict[str, Any]] = []
            self._selected_index = -1
            self._canvas_width = 1024
            self._canvas_height = 1024
            self._drag_mode = ""
            self._drag_anchor = None
            self._drag_region = None
            self._background_pixmap = _FVQtGui.QPixmap()
            self._background_path = ""
            self._background_opacity = 0.35
            self.setMinimumHeight(260)
            self.setMinimumWidth(260)
            self.setMouseTracking(True)

        def set_canvas_size(self, width: int, height: int) -> None:
            self._canvas_width = max(1, int(width or 1))
            self._canvas_height = max(1, int(height or 1))
            self.update()

        def set_regions(self, regions: list[dict[str, Any]]) -> None:
            self._regions = regions
            if self._selected_index >= len(self._regions):
                self._selected_index = len(self._regions) - 1
            self.update()

        def set_background_image(self, path: str) -> bool:
            path = str(path or "").strip()
            if not path:
                return False
            pix = _FVQtGui.QPixmap(path)
            if pix.isNull():
                return False
            self._background_pixmap = pix
            self._background_path = path
            self.update()
            return True

        def clear_background_image(self) -> None:
            self._background_pixmap = _FVQtGui.QPixmap()
            self._background_path = ""
            self.update()

        def background_path(self) -> str:
            return str(self._background_path or "")

        def set_background_opacity(self, value: float | int) -> None:
            try:
                value = float(value)
            except Exception:
                value = 35.0
            if value > 1.0:
                value = value / 100.0
            self._background_opacity = max(0.0, min(1.0, value))
            self.update()

        def selected_index(self) -> int:
            return self._selected_index

        def set_selected_index(self, idx: int) -> None:
            idx = int(idx)
            if idx < -1:
                idx = -1
            if idx >= len(self._regions):
                idx = len(self._regions) - 1
            if idx != self._selected_index:
                self._selected_index = idx
                self.selectionChanged.emit(self._selected_index)
            self.update()

        def _inner_rect(self):
            margin = 12
            area = self.rect().adjusted(margin, margin, -margin, -margin)
            if area.width() <= 0 or area.height() <= 0:
                return _FVQtCore.QRectF()
            aspect = float(self._canvas_width) / float(max(1, self._canvas_height))
            w = float(area.width())
            h = float(area.height())
            if w / max(1.0, h) > aspect:
                draw_h = h
                draw_w = h * aspect
            else:
                draw_w = w
                draw_h = w / max(aspect, 1e-6)
            x = area.x() + (w - draw_w) / 2.0
            y = area.y() + (h - draw_h) / 2.0
            return _FVQtCore.QRectF(x, y, draw_w, draw_h)

        def _region_rect(self, reg: dict[str, Any]):
            inner = self._inner_rect()
            x = inner.x() + float(reg.get('x', 0.0)) * inner.width()
            y = inner.y() + float(reg.get('y', 0.0)) * inner.height()
            w = float(reg.get('w', 0.1)) * inner.width()
            h = float(reg.get('h', 0.1)) * inner.height()
            return _FVQtCore.QRectF(x, y, w, h)

        def paintEvent(self, _event):
            p = _FVQtGui.QPainter(self)
            p.setRenderHint(_FVQtGui.QPainter.Antialiasing, True)
            p.fillRect(self.rect(), _FVQtGui.QColor(24, 28, 34))
            inner = self._inner_rect()
            p.fillRect(inner, _FVQtGui.QColor(16, 20, 25))
            try:
                if not self._background_pixmap.isNull() and inner.width() > 1 and inner.height() > 1:
                    target_size = _FVQtCore.QSize(max(1, int(inner.width())), max(1, int(inner.height())))
                    scaled = self._background_pixmap.scaled(target_size, _FVQtCore.Qt.KeepAspectRatio, _FVQtCore.Qt.SmoothTransformation)
                    x = inner.x() + (inner.width() - scaled.width()) / 2.0
                    y = inner.y() + (inner.height() - scaled.height()) / 2.0
                    p.save()
                    p.setClipRect(inner)
                    p.setOpacity(float(self._background_opacity))
                    p.drawPixmap(_FVQtCore.QPointF(x, y), scaled)
                    p.restore()
            except Exception:
                pass
            p.setPen(_FVQtGui.QPen(_FVQtGui.QColor(88, 102, 120), 1))
            p.drawRoundedRect(inner, 6, 6)
            p.setPen(_FVQtGui.QPen(_FVQtGui.QColor(50, 62, 78), 1, _FVQtCore.Qt.DashLine))
            p.drawLine(inner.center().x(), inner.top(), inner.center().x(), inner.bottom())
            p.drawLine(inner.left(), inner.center().y(), inner.right(), inner.center().y())
            for i, reg in enumerate(self._regions):
                rr = self._region_rect(reg)
                selected = (i == self._selected_index)
                pen = _FVQtGui.QPen(_FVQtGui.QColor(70, 180, 255) if selected else _FVQtGui.QColor(190, 190, 190), 2 if selected else 1)
                p.setPen(pen)
                fill = _FVQtGui.QColor(35, 130, 210, 48) if selected else _FVQtGui.QColor(255, 255, 255, 18)
                p.fillRect(rr, fill)
                p.drawRect(rr)
                tag_rect = _FVQtCore.QRectF(rr.x(), rr.y(), 28, 20)
                p.fillRect(tag_rect, _FVQtGui.QColor(230, 230, 230, 210))
                p.setPen(_FVQtGui.QPen(_FVQtGui.QColor(40, 40, 40), 1))
                p.drawText(tag_rect, _FVQtCore.Qt.AlignCenter, f"{i+1:02d}")
                p.setPen(_FVQtGui.QPen(_FVQtGui.QColor(225, 225, 225), 1))
                desc = str(reg.get('desc') or reg.get('text') or '').strip()
                if desc:
                    txt = desc[:64] + ('…' if len(desc) > 64 else '')
                    text_rect = rr.adjusted(6, 24, -6, -6)
                    p.drawText(text_rect, _FVQtCore.Qt.TextWordWrap | _FVQtCore.Qt.AlignLeft | _FVQtCore.Qt.AlignTop, txt)
                if selected:
                    handle = _FVQtCore.QRectF(rr.right() - 8, rr.bottom() - 8, 10, 10)
                    p.fillRect(handle, _FVQtGui.QColor(70, 180, 255))
            p.end()

        def _hit_test(self, pos):
            for i in range(len(self._regions) - 1, -1, -1):
                rr = self._region_rect(self._regions[i])
                handle = _FVQtCore.QRectF(rr.right() - 10, rr.bottom() - 10, 14, 14)
                if handle.contains(pos):
                    return i, 'resize'
                if rr.contains(pos):
                    return i, 'move'
            return -1, ''

        def mousePressEvent(self, event):
            pos = event.position() if hasattr(event, 'position') else event.posF()
            idx, mode = self._hit_test(pos)
            self.set_selected_index(idx)
            if idx >= 0 and mode:
                self._drag_mode = mode
                self._drag_anchor = (float(pos.x()), float(pos.y()))
                self._drag_region = dict(self._regions[idx])
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            pos = event.position() if hasattr(event, 'position') else event.posF()
            idx, mode = self._hit_test(pos)
            if not self._drag_mode:
                if mode == 'resize':
                    self.setCursor(_FVQtCore.Qt.SizeFDiagCursor)
                elif mode == 'move':
                    self.setCursor(_FVQtCore.Qt.SizeAllCursor)
                else:
                    self.setCursor(_FVQtCore.Qt.ArrowCursor)
            if self._drag_mode and 0 <= self._selected_index < len(self._regions):
                inner = self._inner_rect()
                if inner.width() > 1 and inner.height() > 1 and self._drag_anchor and self._drag_region is not None:
                    dx = (float(pos.x()) - self._drag_anchor[0]) / inner.width()
                    dy = (float(pos.y()) - self._drag_anchor[1]) / inner.height()
                    reg = dict(self._drag_region)
                    if self._drag_mode == 'move':
                        reg['x'] = min(max(0.0, float(reg.get('x', 0.0)) + dx), 1.0 - float(reg.get('w', 0.1)))
                        reg['y'] = min(max(0.0, float(reg.get('y', 0.0)) + dy), 1.0 - float(reg.get('h', 0.1)))
                    elif self._drag_mode == 'resize':
                        reg['w'] = min(max(0.04, float(reg.get('w', 0.1)) + dx), 1.0 - float(reg.get('x', 0.0)))
                        reg['h'] = min(max(0.04, float(reg.get('h', 0.1)) + dy), 1.0 - float(reg.get('y', 0.0)))
                    self._regions[self._selected_index].update(reg)
                    self.regionsChanged.emit()
                    self.update()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            self._drag_mode = ''
            self._drag_anchor = None
            self._drag_region = None
            self.setCursor(_FVQtCore.Qt.ArrowCursor)
            super().mouseReleaseEvent(event)

    class Ideogram4Widget(_FVQtWidgets.QWidget):
        """Embedded PySide Ideogram 4 GGUF UI for FrameVision txt2img.

        This is a real embedded page with its own Generate, Queue and Settings tabs.
        It does not launch the old Tkinter UI and it does not mix controls into the
        normal txt2img page.
        """

        _log_signal = _FVQtCore.Signal(str)
        _done_signal = _FVQtCore.Signal(str, str)
        _error_signal = _FVQtCore.Signal(str)
        _queue_changed_signal = _FVQtCore.Signal()

        def __init__(self, parent=None, embedded: bool = True):
            super().__init__(parent)
            self.embedded = bool(embedded)
            self._worker_thread = None
            self._queue_worker_thread = None
            self._last_output = ""
            self.queue_data = load_queue_state()
            self.jobs = self.queue_data.setdefault("jobs", [])
            self._loading_settings = True
            self._build_ui()
            self._connect_signals()
            self._load_settings_into_ui()
            self.refresh_files()
            self._refresh_queue_table()
            self._loading_settings = False

        def _build_ui(self) -> None:
            Qt = _FVQtCore.Qt
            self.setObjectName("FrameVisionIdeogram4Embedded")
            root = _FVQtWidgets.QVBoxLayout(self)
            root.setContentsMargins(10, 10, 10, 10)
            root.setSpacing(8)

            # FrameVision already shows the engine banner/title. Keep this helper
            # title only when the widget is used outside the embedded FrameVision page.
            if not self.embedded:
                title = _FVQtWidgets.QLabel("Ideogram 4 GGUF")
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("font-size: 20px; font-weight: 700; padding: 6px;")
                root.addWidget(title)

            self.tabs = _FVQtWidgets.QTabWidget()
            self.tabs.setDocumentMode(True)
            root.addWidget(self.tabs, 1)

            self.generate_tab = _FVQtWidgets.QWidget()
            self.queue_tab = _FVQtWidgets.QWidget()
            self.settings_tab = _FVQtWidgets.QWidget()
            self.tabs.addTab(self.generate_tab, "Generate")
            self.tabs.addTab(self.queue_tab, "Queue")
            self.tabs.addTab(self.settings_tab, "Settings")

            self._build_generate_tab()
            self._build_queue_tab()
            self._build_settings_tab()

        def _build_generate_tab(self) -> None:
            Qt = _FVQtCore.Qt
            layout = _FVQtWidgets.QVBoxLayout(self.generate_tab)
            layout.setContentsMargins(0, 8, 0, 0)
            layout.setSpacing(8)

            self.generate_splitter = _FVQtWidgets.QSplitter(Qt.Horizontal)
            layout.addWidget(self.generate_splitter, 1)

            left_scroll = _FVQtWidgets.QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setFrameShape(_FVQtWidgets.QFrame.NoFrame)
            self.generate_splitter.addWidget(left_scroll)

            left = _FVQtWidgets.QWidget()
            left_scroll.setWidget(left)
            left_layout = _FVQtWidgets.QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 6, 0)
            left_layout.setSpacing(8)

            self.prompt = _FVQtWidgets.QTextEdit()
            self.prompt.setPlaceholderText("Describe the image you want...")
            self.prompt.setMinimumHeight(140)
            left_layout.addWidget(_FVQtWidgets.QLabel("Prompt"))
            left_layout.addWidget(self.prompt)

            self.negative = _FVQtWidgets.QTextEdit()
            self.negative.setPlaceholderText("Negative prompt (optional)")
            self.negative.setMaximumHeight(110)
            left_layout.addWidget(_FVQtWidgets.QLabel("Negative"))
            left_layout.addWidget(self.negative)

            row = _FVQtWidgets.QHBoxLayout()
            self.preset_combo = _FVQtWidgets.QComboBox()
            self.preset_combo.addItems(["Turbo 12", "Default 20", "Quality 48", "Custom"])
            self.preset_combo.setCurrentText("Default 20")
            self.preset_combo.setToolTip("Ideogram preset. Turbo 12 is fastest, Default 20 is balanced, Quality 48 is slowest/highest quality.")
            self.width = _FVQtWidgets.QSpinBox(); self.width.setRange(256, 4096); self.width.setSingleStep(32); self.width.setValue(1024)
            self.height = _FVQtWidgets.QSpinBox(); self.height.setRange(256, 4096); self.height.setSingleStep(32); self.height.setValue(1024)
            self.steps = _FVQtWidgets.QSpinBox(); self.steps.setRange(1, 200); self.steps.setValue(20)
            self.guidance = _FVQtWidgets.QDoubleSpinBox(); self.guidance.setRange(0.0, 20.0); self.guidance.setDecimals(2); self.guidance.setSingleStep(0.1); self.guidance.setValue(4.0); self.guidance.setToolTip("Default: 4.0")
            self.seed = _FVQtWidgets.QSpinBox(); self.seed.setRange(-1, 2147483647); self.seed.setValue(-1)
            for label, widget in (("Preset", self.preset_combo), ("W", self.width), ("H", self.height), ("Steps", self.steps), ("CFG", self.guidance), ("Seed", self.seed)):
                col = _FVQtWidgets.QVBoxLayout()
                col.addWidget(_FVQtWidgets.QLabel(label))
                col.addWidget(widget)
                row.addLayout(col)
            row.addStretch(1)
            left_layout.addLayout(row)

            options_row = _FVQtWidgets.QHBoxLayout()
            self.raw_prompt = _FVQtWidgets.QCheckBox("Raw prompt")
            self.raw_prompt.setToolTip("Send the prompt exactly as typed. Off wraps a normal prompt into Ideogram's JSON-style caption payload.")
            self.stream_layers = _FVQtWidgets.QCheckBox("Stream layers")
            self.stream_layers.setToolTip("Optional sd-cli streaming mode. Leave off unless testing low-VRAM behavior.")
            options_row.addWidget(self.raw_prompt)
            options_row.addWidget(self.stream_layers)
            options_row.addStretch(1)
            left_layout.addLayout(options_row)

            tip = _FVQtWidgets.QLabel("Runtime paths and GGUF file selection are in the Settings tab.")
            try:
                tip.setWordWrap(True)
                tip.setStyleSheet("opacity: 0.8;")
            except Exception:
                pass
            left_layout.addWidget(tip)

            self.layout_group = _FVQtWidgets.QGroupBox("Layout / Region Prompt Builder")
            layout_box = _FVQtWidgets.QVBoxLayout(self.layout_group)
            layout_box.setContentsMargins(10, 10, 10, 10)
            layout_box.setSpacing(8)

            top_row = _FVQtWidgets.QHBoxLayout()
            self.enable_layout_prompt = _FVQtWidgets.QCheckBox("Enable layout prompt builder")
            self.enable_layout_prompt.setToolTip("Use draggable regions to build the Ideogram JSON prompt used by the Comfy/KJ workflow style.")
            self.layout_preview_json_btn = _FVQtWidgets.QPushButton("Preview built prompt")
            self.layout_add_region_btn = _FVQtWidgets.QPushButton("Add region")
            self.layout_remove_region_btn = _FVQtWidgets.QPushButton("Remove selected")
            self.layout_clear_regions_btn = _FVQtWidgets.QPushButton("Clear all")
            top_row.addWidget(self.enable_layout_prompt)
            top_row.addStretch(1)
            top_row.addWidget(self.layout_preview_json_btn)
            top_row.addWidget(self.layout_add_region_btn)
            top_row.addWidget(self.layout_remove_region_btn)
            top_row.addWidget(self.layout_clear_regions_btn)
            layout_box.addLayout(top_row)

            template_row = _FVQtWidgets.QHBoxLayout()
            self.layout_save_template_btn = _FVQtWidgets.QPushButton("Save template")
            self.layout_template_include_text = _FVQtWidgets.QCheckBox("Include text")
            self.layout_template_include_text.setChecked(True)
            self.layout_template_include_text.setToolTip("When off, saves the same boxes/regions but clears prompt and region text so the layout can be reused as a blank structure.")
            self.layout_load_template_btn = _FVQtWidgets.QPushButton("Load template")
            self.layout_load_template_btn.setToolTip("Open the template folder and load a saved Ideogram layout template JSON.")
            template_row.addStretch(1)
            template_row.addWidget(self.layout_save_template_btn)
            template_row.addWidget(self.layout_template_include_text)
            template_row.addWidget(self.layout_load_template_btn)
            layout_box.addLayout(template_row)

            credit_label = _FVQtWidgets.QLabel("Inspired by the ComfyUI KJ Ideogram prompt node. Check 3rd party license section for more info.")
            credit_label.setWordWrap(True)
            try:
                credit_label.setStyleSheet("color: #9ca3af; font-size: 11px;")
            except Exception:
                pass
            layout_box.addWidget(credit_label)

            bg_row = _FVQtWidgets.QHBoxLayout()
            self.layout_grab_background_btn = _FVQtWidgets.QPushButton("Grab background")
            self.layout_grab_background_btn.setToolTip("Use the latest generated image as a faint guide behind the layout boxes. If there is no latest image, pick one from the output folder.")
            self.layout_clear_background_btn = _FVQtWidgets.QPushButton("Clear background")
            self.layout_live_background = _FVQtWidgets.QCheckBox("Live background")
            self.layout_live_background.setToolTip("Automatically show each newly generated Ideogram image behind the layout boxes.")
            self.layout_background_opacity = _FVQtWidgets.QSlider(Qt.Horizontal)
            self.layout_background_opacity.setRange(0, 100)
            self.layout_background_opacity.setValue(35)
            self.layout_background_opacity.setToolTip("Background guide opacity / transparency behind the region boxes.")
            self.layout_background_opacity_label = _FVQtWidgets.QLabel("BG opacity: 35%")
            bg_row.addWidget(self.layout_grab_background_btn)
            bg_row.addWidget(self.layout_clear_background_btn)
            bg_row.addWidget(self.layout_live_background)
            bg_row.addWidget(self.layout_background_opacity_label)
            bg_row.addWidget(self.layout_background_opacity, 1)
            layout_box.addLayout(bg_row)

            layout_note = _FVQtWidgets.QLabel("Use the normal Prompt box above as the high-level description. Add boxes below to describe what should appear in specific parts of the image. Text regions support exact text and optional typography notes.")
            layout_note.setWordWrap(True)
            layout_box.addWidget(layout_note)

            style_grid = _FVQtWidgets.QGridLayout()
            style_grid.addWidget(_FVQtWidgets.QLabel("Background"), 0, 0)
            self.layout_background = _FVQtWidgets.QLineEdit()
            self.layout_background.setPlaceholderText("Background / scene notes")
            style_grid.addWidget(self.layout_background, 0, 1)
            style_grid.addWidget(_FVQtWidgets.QLabel("Style"), 0, 2)
            self.layout_style = _FVQtWidgets.QComboBox()
            self.layout_style.addItems(["none", "photo", "art_style"])
            style_grid.addWidget(self.layout_style, 0, 3)
            style_grid.addWidget(_FVQtWidgets.QLabel("Style.photo"), 1, 0)
            self.layout_style_photo = _FVQtWidgets.QLineEdit()
            self.layout_style_photo.setPlaceholderText("Example: realistic, amateur, casual")
            style_grid.addWidget(self.layout_style_photo, 1, 1)
            style_grid.addWidget(_FVQtWidgets.QLabel("Aesthetics"), 1, 2)
            self.layout_aesthetics = _FVQtWidgets.QLineEdit()
            self.layout_aesthetics.setPlaceholderText("Example: minimalist, cinematic")
            style_grid.addWidget(self.layout_aesthetics, 1, 3)
            style_grid.addWidget(_FVQtWidgets.QLabel("Lighting"), 2, 0)
            self.layout_lighting = _FVQtWidgets.QLineEdit()
            self.layout_lighting.setPlaceholderText("Example: overcast, calm")
            style_grid.addWidget(self.layout_lighting, 2, 1)
            style_grid.addWidget(_FVQtWidgets.QLabel("Medium"), 2, 2)
            self.layout_medium = _FVQtWidgets.QLineEdit()
            self.layout_medium.setPlaceholderText("Example: photography, poster, watercolor")
            style_grid.addWidget(self.layout_medium, 2, 3)
            layout_box.addLayout(style_grid)

            mid = _FVQtWidgets.QHBoxLayout()
            self.layout_canvas = RegionPromptCanvas()
            mid.addWidget(self.layout_canvas, 2)
            side = _FVQtWidgets.QVBoxLayout()
            self.layout_region_list = _FVQtWidgets.QListWidget()
            self.layout_region_list.setMinimumWidth(230)
            side.addWidget(self.layout_region_list, 2)
            self.layout_region_desc = _FVQtWidgets.QTextEdit()
            self.layout_region_desc.setPlaceholderText("Describe what should be in the selected region")
            self.layout_region_desc.setMinimumHeight(80)
            side.addWidget(self.layout_region_desc)
            detail_row = _FVQtWidgets.QHBoxLayout()
            self.layout_region_type = _FVQtWidgets.QComboBox()
            self.layout_region_type.addItems(["obj", "text"])
            detail_row.addWidget(self.layout_region_type)
            self.layout_region_text = _FVQtWidgets.QLineEdit()
            self.layout_region_text.setPlaceholderText("Exact text for text regions")
            detail_row.addWidget(self.layout_region_text, 1)
            side.addLayout(detail_row)
            self.layout_region_palette = _FVQtWidgets.QLineEdit()
            self.layout_region_palette.setPlaceholderText("Optional palette: #ff6600, #ffffff")
            side.addWidget(self.layout_region_palette)
            self.layout_json_preview = _FVQtWidgets.QPlainTextEdit()
            self.layout_json_preview.setReadOnly(True)
            self.layout_json_preview.setMinimumHeight(120)
            self.layout_json_preview.setPlaceholderText("Built layout prompt preview")
            side.addWidget(self.layout_json_preview, 1)
            mid.addLayout(side, 1)
            layout_box.addLayout(mid)
            left_layout.addWidget(self.layout_group)
            self.layout_regions = []

            # Keep the bottom action row outside the scroll area.
            # When the layout builder grows taller than the tab, only the form
            # should scroll; Generate/Add/View/Open must remain reachable.
            left_layout.addStretch(1)

            self.generate_right_panel = _FVQtWidgets.QWidget()
            right_layout = _FVQtWidgets.QVBoxLayout(self.generate_right_panel)
            right_layout.setContentsMargins(6, 0, 0, 0)
            self.generate_splitter.addWidget(self.generate_right_panel)
            self.preview = _FVQtWidgets.QLabel("Preview will appear here")
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setMinimumHeight(280)
            self.preview.setStyleSheet("border: 1px solid rgba(130, 150, 170, 90); border-radius: 8px; padding: 8px;")
            right_layout.addWidget(self.preview, 3)
            self.log_box = _FVQtWidgets.QPlainTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setMinimumHeight(180)
            right_layout.addWidget(self.log_box, 2)

            self.generate_button_bar = _FVQtWidgets.QWidget()
            self.generate_button_bar.setObjectName("IdeogramGenerateStickyButtonBar")
            btns = _FVQtWidgets.QHBoxLayout(self.generate_button_bar)
            btns.setContentsMargins(0, 0, 0, 0)
            btns.setSpacing(8)
            self.generate_btn = _FVQtWidgets.QPushButton("Generate Ideogram image")
            self.add_queue_btn = _FVQtWidgets.QPushButton("Add to Ideogram queue")
            self.framevision_queue_btn = _FVQtWidgets.QPushButton("Add to queue")
            self.framevision_queue_btn.setVisible(False)
            self.open_output_btn = _FVQtWidgets.QPushButton("View results")
            self.open_last_btn = _FVQtWidgets.QPushButton("Open last image")
            btns.addWidget(self.generate_btn, 2)
            btns.addWidget(self.add_queue_btn, 2)
            btns.addWidget(self.framevision_queue_btn, 2)
            btns.addWidget(self.open_output_btn)
            btns.addWidget(self.open_last_btn)
            layout.addWidget(self.generate_button_bar, 0)

            try:
                self.generate_splitter.setSizes([680, 540])
            except Exception:
                pass

        def _build_settings_tab(self) -> None:
            Qt = _FVQtCore.Qt
            outer = _FVQtWidgets.QVBoxLayout(self.settings_tab)
            outer.setContentsMargins(0, 8, 0, 0)
            outer.setSpacing(8)

            scroll = _FVQtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            outer.addWidget(scroll, 1)
            body = _FVQtWidgets.QWidget()
            scroll.setWidget(body)
            layout = _FVQtWidgets.QVBoxLayout(body)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            queue_box = _FVQtWidgets.QGroupBox("FrameVision queue")
            queue_layout = _FVQtWidgets.QVBoxLayout(queue_box)
            queue_layout.setContentsMargins(10, 10, 10, 10)
            self.use_framevision_queue = _FVQtWidgets.QCheckBox("Use FrameVision queue")
            self.use_framevision_queue.setToolTip("When enabled, Ideogram uses the main FrameVision queue instead of the built-in Ideogram queue. This hides the Ideogram queue tab, preview pane and logs, and the Generate page shows one Add to queue button.")
            try:
                self.use_framevision_queue.setStyleSheet("font-weight: 700; padding: 4px;")
            except Exception:
                pass
            queue_layout.addWidget(self.use_framevision_queue)
            queue_note = _FVQtWidgets.QLabel("When enabled, the Generate tab uses one Add to queue button and hides Ideogram's own preview, logs and queue tab.")
            queue_note.setWordWrap(True)
            try:
                queue_note.setStyleSheet("opacity: 0.85;")
            except Exception:
                pass
            queue_layout.addWidget(queue_note)
            layout.addWidget(queue_box)

            runtime_box = _FVQtWidgets.QGroupBox("Runtime and paths")
            form = _FVQtWidgets.QFormLayout(runtime_box)
            self.gguf_dir = _FVQtWidgets.QLineEdit(str(default_gguf_dir()))
            self.sd_cli_path = _FVQtWidgets.QLineEdit(str(default_sd_cli_path()))
            self.output_dir = _FVQtWidgets.QLineEdit(str(default_output_dir()))
            self.btn_browse_gguf = _FVQtWidgets.QPushButton("Browse")
            self.btn_browse_sdcli = _FVQtWidgets.QPushButton("Browse")
            self.btn_browse_output = _FVQtWidgets.QPushButton("Browse")
            self.btn_refresh = _FVQtWidgets.QPushButton("Refresh GGUF file lists")

            def line_with_button(edit, btn):
                w = _FVQtWidgets.QWidget()
                l = _FVQtWidgets.QHBoxLayout(w)
                l.setContentsMargins(0, 0, 0, 0)
                l.addWidget(edit, 1)
                l.addWidget(btn)
                return w

            form.addRow("GGUF folder", line_with_button(self.gguf_dir, self.btn_browse_gguf))
            form.addRow("sd-cli.exe", line_with_button(self.sd_cli_path, self.btn_browse_sdcli))
            form.addRow("Output folder", line_with_button(self.output_dir, self.btn_browse_output))
            layout.addWidget(runtime_box)

            files_box = _FVQtWidgets.QGroupBox("GGUF files")
            files_form = _FVQtWidgets.QFormLayout(files_box)
            self.diffusion_file = _FVQtWidgets.QComboBox()
            self.uncond_file = _FVQtWidgets.QComboBox()
            self.llm_file = _FVQtWidgets.QComboBox()
            self.vae_file = _FVQtWidgets.QComboBox()
            files_form.addRow("Conditional GGUF", self.diffusion_file)
            files_form.addRow("Unconditional GGUF", self.uncond_file)
            files_form.addRow("Qwen/VL GGUF", self.llm_file)
            files_form.addRow("Flux2 VAE", self.vae_file)
            files_form.addRow("", self.btn_refresh)
            layout.addWidget(files_box)

            lora_box = _FVQtWidgets.QGroupBox("LoRA")
            lora_form = _FVQtWidgets.QFormLayout(lora_box)
            self.lora_path = _FVQtWidgets.QLineEdit("")
            self.btn_browse_lora = _FVQtWidgets.QPushButton("Load")
            self.lora_scale = _FVQtWidgets.QSlider(Qt.Horizontal)
            self.lora_scale.setRange(0, 200)
            self.lora_scale.setSingleStep(5)
            self.lora_scale.setPageStep(10)
            try:
                self.lora_scale.setTickPosition(_FVQtWidgets.QSlider.TickPosition.TicksBelow)
            except Exception:
                try:
                    self.lora_scale.setTickPosition(_FVQtWidgets.QSlider.TicksBelow)
                except Exception:
                    pass
            self.lora_scale.setTickInterval(50)
            self.lora_scale.setValue(100)
            self.lora_scale_value = _FVQtWidgets.QLabel("1.00")
            self.lora_scale_value.setMinimumWidth(48)
            self.lora_scale_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lora_strength_row = _FVQtWidgets.QWidget()
            lora_strength_layout = _FVQtWidgets.QHBoxLayout(lora_strength_row)
            lora_strength_layout.setContentsMargins(0, 0, 0, 0)
            lora_strength_layout.setSpacing(8)
            lora_strength_layout.addWidget(self.lora_scale, 1)
            lora_strength_layout.addWidget(self.lora_scale_value)
            self.lora_enabled = _FVQtWidgets.QCheckBox()
            self.lora_apply_mode = _FVQtWidgets.QComboBox()
            self.lora_apply_mode.addItems(["auto", "at_runtime", "immediately"])
            lora_form.addRow("LoRA", line_with_button(self.lora_path, self.btn_browse_lora))
            lora_form.addRow("Strength", lora_strength_row)
            layout.addWidget(lora_box)

            layout.addStretch(1)

        def _build_queue_tab(self) -> None:
            layout = _FVQtWidgets.QVBoxLayout(self.queue_tab)
            layout.setContentsMargins(0, 8, 0, 0)
            layout.setSpacing(8)

            info = _FVQtWidgets.QLabel(f"Ideogram queue JSON: {queue_state_path()}")
            info.setWordWrap(True)
            layout.addWidget(info)

            self.queue_table = _FVQtWidgets.QTableWidget(0, 7)
            self.queue_table.setHorizontalHeaderLabels(["Status", "Steps", "Started", "Duration", "Finished", "Resolution", "Prompt"])
            try:
                self.queue_table.horizontalHeader().setStretchLastSection(True)
                self.queue_table.setSelectionBehavior(_FVQtWidgets.QAbstractItemView.SelectRows)
                self.queue_table.setEditTriggers(_FVQtWidgets.QAbstractItemView.NoEditTriggers)
            except Exception:
                pass
            layout.addWidget(self.queue_table, 1)

            btns = _FVQtWidgets.QHBoxLayout()
            self.queue_add_btn = _FVQtWidgets.QPushButton("Add current prompt")
            self.queue_start_btn = _FVQtWidgets.QPushButton("Start queue")
            self.queue_stop_after_btn = _FVQtWidgets.QPushButton("Stop after current")
            self.queue_clear_done_btn = _FVQtWidgets.QPushButton("Clear done/failed")
            self.queue_open_output_btn = _FVQtWidgets.QPushButton("View results")
            btns.addWidget(self.queue_add_btn)
            btns.addWidget(self.queue_start_btn)
            btns.addWidget(self.queue_stop_after_btn)
            btns.addWidget(self.queue_clear_done_btn)
            btns.addWidget(self.queue_open_output_btn)
            layout.addLayout(btns)
            self._queue_stop_requested = False

        def _default_region(self) -> dict[str, Any]:
            count = len(getattr(self, 'layout_regions', []) or [])
            offset = min(0.06 * count, 0.42)
            return {
                'x': max(0.0, min(0.72, 0.06 + offset)),
                'y': max(0.0, min(0.72, 0.06 + offset)),
                'w': 0.28,
                'h': 0.22,
                'type': 'obj',
                'text': '',
                'desc': f'region {count + 1}',
                'palette': [],
            }

        def _collect_layout_regions(self) -> list[dict[str, Any]]:
            items = []
            for reg in getattr(self, 'layout_regions', []) or []:
                try:
                    item = {
                        'x': float(reg.get('x', 0.0)),
                        'y': float(reg.get('y', 0.0)),
                        'w': float(reg.get('w', 0.2)),
                        'h': float(reg.get('h', 0.2)),
                        'type': str(reg.get('type') or 'obj'),
                        'text': str(reg.get('text') or ''),
                        'desc': str(reg.get('desc') or ''),
                        'palette': list(reg.get('palette') or []),
                    }
                except Exception:
                    continue
                items.append(item)
            return items

        def _layout_prompt_enabled(self) -> bool:
            try:
                return bool(self.enable_layout_prompt.isChecked())
            except Exception:
                return False

        def _build_layout_prompt_json(self) -> str:
            return build_structured_caption(
                self.prompt.toPlainText().strip(),
                int(self.width.value()),
                int(self.height.value()),
                regions=self._collect_layout_regions(),
                background=self.layout_background.text().strip(),
                aesthetics=self.layout_aesthetics.text().strip(),
                lighting=self.layout_lighting.text().strip(),
                photo=self.layout_style_photo.text().strip(),
                art_style=self.layout_style_photo.text().strip(),
                medium=self.layout_medium.text().strip(),
                style=self.layout_style.currentText().strip() or 'photo',
            )

        def _layout_template_payload(self, include_text: bool) -> dict[str, Any]:
            regions: list[dict[str, Any]] = []
            for reg in self._collect_layout_regions():
                item = dict(reg)
                if not include_text:
                    item["text"] = ""
                    item["desc"] = ""
                regions.append(item)

            return {
                "layout_prompt_enabled": bool(self.enable_layout_prompt.isChecked()),
                "last_prompt": self.prompt.toPlainText() if include_text else "",
                "layout_background": self.layout_background.text().strip() if include_text else "",
                "layout_style": self.layout_style.currentText().strip(),
                "layout_style_photo": self.layout_style_photo.text().strip() if include_text else "",
                "layout_aesthetics": self.layout_aesthetics.text().strip() if include_text else "",
                "layout_lighting": self.layout_lighting.text().strip() if include_text else "",
                "layout_medium": self.layout_medium.text().strip() if include_text else "",
                "layout_regions": regions,
            }

        def _apply_layout_template_payload(self, payload: dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                raise ValueError("Template payload is not a JSON object.")

            def _text(key: str, default: str = "") -> str:
                return str(payload.get(key, default) or default)

            self.enable_layout_prompt.setChecked(bool(payload.get("layout_prompt_enabled", True)))
            self.prompt.setPlainText(_text("last_prompt", ""))
            self.layout_background.setText(_text("layout_background", ""))
            self.layout_style_photo.setText(_text("layout_style_photo", ""))
            self.layout_aesthetics.setText(_text("layout_aesthetics", ""))
            self.layout_lighting.setText(_text("layout_lighting", ""))
            self.layout_medium.setText(_text("layout_medium", ""))
            style_value = _text("layout_style", "photo")
            if style_value == "art style":
                style_value = "art_style"
            idx = self.layout_style.findText(style_value)
            if idx >= 0:
                self.layout_style.setCurrentIndex(idx)

            regions = payload.get("layout_regions", [])
            self.layout_regions = list(regions) if isinstance(regions, list) else []
            try:
                self.layout_canvas.set_selected_index(0 if self.layout_regions else -1)
            except Exception:
                pass
            self._refresh_layout_ui(keep_preview=True)
            self._save_from_ui()

        def _save_layout_template(self) -> None:
            name, ok = _FVQtWidgets.QInputDialog.getText(self, "Save Ideogram layout template", "Template name:")
            if not ok:
                return
            name = str(name or "").strip()
            if not name:
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", "Template name is empty.")
                return
            include_text = bool(self.layout_template_include_text.isChecked())
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{_safe_template_filename(name)}_{stamp}.json"
            path = layout_templates_dir() / filename
            data = {
                "version": 1,
                "kind": "ideogram4_layout_template",
                "name": name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "include_text": include_text,
                "data": self._layout_template_payload(include_text),
            }
            try:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", f"Could not save template.\n\n{exc}")
                return
            self.log(f"[Ideogram4] Saved layout template: {path}")
            _FVQtWidgets.QMessageBox.information(self, "Ideogram 4 GGUF", f"Template saved:\n{path}")

        def _load_layout_template(self) -> None:
            folder = layout_templates_dir()
            path, _ = _FVQtWidgets.QFileDialog.getOpenFileName(self, "Load Ideogram layout template", str(folder), "JSON templates (*.json);;All files (*)")
            if not path:
                return
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                payload = data.get("data", data) if isinstance(data, dict) else {}
                self._apply_layout_template_payload(payload)
            except Exception as exc:
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", f"Could not load template.\n\n{exc}")
                return
            self.log(f"[Ideogram4] Loaded layout template: {path}")

        def _set_layout_background_preview(self, path: str, *, show_warning: bool = False) -> bool:
            try:
                ok = self.layout_canvas.set_background_image(str(path or ""))
                if ok:
                    self.log(f"[Ideogram4] Layout background preview: {path}")
                    self._save_from_ui()
                    return True
            except Exception:
                ok = False
            if show_warning:
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", "Could not load background preview image.")
            return False

        def _grab_layout_background(self) -> None:
            path = str(getattr(self, "_last_output", "") or "").strip()
            if path and Path(path).exists():
                self._set_layout_background_preview(path, show_warning=True)
                return
            folder = Path(self.output_dir.text().strip() or str(default_output_dir()))
            path, _ = _FVQtWidgets.QFileDialog.getOpenFileName(self, "Select layout background preview", str(folder), "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
            if path:
                self._set_layout_background_preview(path, show_warning=True)

        def _clear_layout_background(self) -> None:
            try:
                self.layout_canvas.clear_background_image()
                self._save_from_ui()
            except Exception:
                pass

        def _on_layout_background_opacity_changed(self, value: int) -> None:
            try:
                value = int(value)
            except Exception:
                value = 35
            value = max(0, min(100, value))
            try:
                self.layout_background_opacity_label.setText(f"BG opacity: {value}%")
            except Exception:
                pass
            try:
                self.layout_canvas.set_background_opacity(value)
            except Exception:
                pass
            self._save_from_ui()

        def _refresh_layout_ui(self, keep_preview: bool = True) -> None:
            if not hasattr(self, 'layout_region_list'):
                return
            selected = -1
            try:
                selected = self.layout_canvas.selected_index()
            except Exception:
                pass
            self.layout_region_list.blockSignals(True)
            self.layout_region_list.clear()
            for i, reg in enumerate(getattr(self, 'layout_regions', []) or []):
                desc = str(reg.get('desc') or reg.get('text') or f'region {i+1}').strip()
                self.layout_region_list.addItem(_FVQtWidgets.QListWidgetItem(f"{i+1:02d}  {desc[:80]}"))
            self.layout_region_list.blockSignals(False)
            try:
                self.layout_canvas.set_canvas_size(int(self.width.value()), int(self.height.value()))
                self.layout_canvas.set_regions(self.layout_regions)
            except Exception:
                pass
            if 0 <= selected < self.layout_region_list.count():
                self.layout_region_list.setCurrentRow(selected)
                self.layout_canvas.set_selected_index(selected)
            elif self.layout_region_list.count() and self.layout_canvas.selected_index() < 0:
                self.layout_region_list.setCurrentRow(0)
                self.layout_canvas.set_selected_index(0)
            else:
                self._sync_region_editor_from_selection()
            if keep_preview:
                self._update_layout_preview()

        def _sync_region_editor_from_selection(self) -> None:
            idx = -1
            try:
                idx = self.layout_canvas.selected_index()
            except Exception:
                pass
            enabled = 0 <= idx < len(getattr(self, 'layout_regions', []) or [])
            for w in (self.layout_region_desc, self.layout_region_type, self.layout_region_text, self.layout_region_palette, self.layout_remove_region_btn):
                try:
                    w.setEnabled(enabled)
                except Exception:
                    pass
            if not enabled:
                try:
                    self._layout_loading_editor = True
                    self.layout_region_desc.blockSignals(True); self.layout_region_desc.setPlainText(''); self.layout_region_desc.blockSignals(False)
                    self.layout_region_text.blockSignals(True); self.layout_region_text.setText(''); self.layout_region_text.blockSignals(False)
                    self.layout_region_palette.blockSignals(True); self.layout_region_palette.setText(''); self.layout_region_palette.blockSignals(False)
                except Exception:
                    pass
                finally:
                    self._layout_loading_editor = False
                return
            reg = self.layout_regions[idx]
            try:
                self._layout_loading_editor = True
                self.layout_region_desc.blockSignals(True); self.layout_region_desc.setPlainText(str(reg.get('desc') or '')); self.layout_region_desc.blockSignals(False)
                self.layout_region_text.blockSignals(True); self.layout_region_text.setText(str(reg.get('text') or '')); self.layout_region_text.blockSignals(False)
                self.layout_region_type.blockSignals(True); self.layout_region_type.setCurrentText(str(reg.get('type') or 'obj')); self.layout_region_type.blockSignals(False)
                palette = ', '.join(list(reg.get('palette') or []))
                self.layout_region_palette.blockSignals(True); self.layout_region_palette.setText(palette); self.layout_region_palette.blockSignals(False)
            except Exception:
                pass
            finally:
                self._layout_loading_editor = False

        def _sync_region_selection_from_list(self) -> None:
            self.layout_canvas.set_selected_index(self.layout_region_list.currentRow())
            self._sync_region_editor_from_selection()

        def _on_layout_canvas_selection_changed(self, idx: int) -> None:
            try:
                self.layout_region_list.blockSignals(True)
                self.layout_region_list.setCurrentRow(int(idx))
                self.layout_region_list.blockSignals(False)
            except Exception:
                pass
            self._sync_region_editor_from_selection()

        def _on_layout_region_editor_changed(self) -> None:
            if bool(getattr(self, '_layout_loading_editor', False)):
                return
            idx = self.layout_canvas.selected_index()
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                return
            reg = self.layout_regions[idx]
            reg['desc'] = self.layout_region_desc.toPlainText().strip()
            reg['type'] = self.layout_region_type.currentText().strip() or 'obj'
            reg['text'] = self.layout_region_text.text().strip()
            reg['palette'] = [p.strip() for p in self.layout_region_palette.text().split(',') if p.strip()]
            try:
                desc = str(reg.get('desc') or reg.get('text') or f'region {idx+1}').strip()
                item = self.layout_region_list.item(idx)
                if item is not None:
                    item.setText(f"{idx+1:02d}  {desc[:80]}")
            except Exception:
                pass
            try:
                self.layout_canvas.update()
            except Exception:
                pass
            self._update_layout_preview()
            self._save_from_ui()

        def _on_layout_regions_changed(self) -> None:
            self._refresh_layout_ui(keep_preview=True)
            self._save_from_ui()

        def _add_layout_region(self) -> None:
            self.layout_regions.append(self._default_region())
            self._refresh_layout_ui(keep_preview=True)
            last = len(self.layout_regions) - 1
            try:
                self.layout_region_list.setCurrentRow(last)
                self.layout_canvas.set_selected_index(last)
            except Exception:
                pass
            self._save_from_ui()

        def _remove_layout_region(self) -> None:
            idx = self.layout_canvas.selected_index()
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                return
            self.layout_regions.pop(idx)
            idx = min(idx, len(self.layout_regions) - 1) if self.layout_regions else -1
            try:
                self.layout_canvas.set_selected_index(idx)
            except Exception:
                pass
            self._refresh_layout_ui(keep_preview=True)
            self._save_from_ui()

        def _clear_layout_regions(self) -> None:
            self.layout_regions = []
            try:
                self.layout_canvas.set_selected_index(-1)
            except Exception:
                pass
            self._refresh_layout_ui(keep_preview=True)
            self._save_from_ui()

        def _update_layout_preview(self) -> None:
            if not hasattr(self, 'layout_json_preview'):
                return
            if self._layout_prompt_enabled():
                try:
                    self.layout_json_preview.setPlainText(self._build_layout_prompt_json())
                except Exception as exc:
                    self.layout_json_preview.setPlainText(f'Could not build prompt yet: {exc}')
            else:
                self.layout_json_preview.setPlainText('Enable the layout prompt builder to preview the generated Ideogram JSON prompt.')

        def _on_preview_layout_prompt(self) -> None:
            self._update_layout_preview()
            try:
                self.layout_json_preview.setFocus()
            except Exception:
                pass

        def _on_size_changed(self, *_args) -> None:
            try:
                self.layout_canvas.set_canvas_size(int(self.width.value()), int(self.height.value()))
            except Exception:
                pass
            self._update_layout_preview()
            self._save_from_ui()

        def _on_pyside_preset_changed(self, value: str) -> None:
            try:
                preset = str(value or '')
                if preset == 'Turbo 12':
                    self.steps.setValue(12)
                elif preset == 'Default 20':
                    self.steps.setValue(20)
                elif preset == 'Quality 48':
                    self.steps.setValue(48)
            except Exception:
                pass

        def _connect_signals(self) -> None:
            self._log_signal.connect(self._append_log)
            self._done_signal.connect(self._on_done)
            self._error_signal.connect(self._on_error)
            self._queue_changed_signal.connect(self._refresh_queue_table)
            self.btn_refresh.clicked.connect(self.refresh_files)
            self.generate_btn.clicked.connect(self.generate_clicked)
            self.add_queue_btn.clicked.connect(self.add_current_to_queue)
            self.framevision_queue_btn.clicked.connect(self.add_current_to_framevision_queue)
            self.queue_add_btn.clicked.connect(self.add_current_to_queue)
            self.queue_start_btn.clicked.connect(self.start_queue)
            self.queue_stop_after_btn.clicked.connect(self.stop_queue_after_current)
            self.queue_clear_done_btn.clicked.connect(self.clear_finished_queue_jobs)
            self.queue_open_output_btn.clicked.connect(self.view_results)
            self.open_output_btn.clicked.connect(self.view_results)
            self.open_last_btn.clicked.connect(self.open_last_image)
            self.btn_browse_gguf.clicked.connect(lambda: self._browse_dir(self.gguf_dir, refresh=True))
            self.btn_browse_output.clicked.connect(lambda: self._browse_dir(self.output_dir, refresh=False))
            self.btn_browse_sdcli.clicked.connect(self._browse_sdcli)
            self.btn_browse_lora.clicked.connect(self._browse_lora)
            for w in (self.gguf_dir, self.sd_cli_path, self.output_dir, self.lora_path):
                w.editingFinished.connect(self._save_from_ui)
            for w in (self.diffusion_file, self.uncond_file, self.llm_file, self.vae_file):
                w.currentTextChanged.connect(self._save_from_ui)
            self.preset_combo.currentTextChanged.connect(self._on_pyside_preset_changed)
            self.preset_combo.currentTextChanged.connect(self._save_from_ui)
            for w in (self.steps, self.guidance, self.seed):
                w.valueChanged.connect(self._save_from_ui)
            self.width.valueChanged.connect(self._on_size_changed)
            self.height.valueChanged.connect(self._on_size_changed)
            self.raw_prompt.toggled.connect(self._save_from_ui)
            self.stream_layers.toggled.connect(self._save_from_ui)
            self.lora_scale.valueChanged.connect(self._on_lora_scale_changed)
            self.use_framevision_queue.toggled.connect(self._on_framevision_queue_toggled)
            self.prompt.textChanged.connect(self._save_from_ui)
            self.prompt.textChanged.connect(self._update_layout_preview)
            self.negative.textChanged.connect(self._save_from_ui)
            self.enable_layout_prompt.toggled.connect(self._save_from_ui)
            self.enable_layout_prompt.toggled.connect(self._update_layout_preview)
            self.layout_preview_json_btn.clicked.connect(self._on_preview_layout_prompt)
            self.layout_save_template_btn.clicked.connect(self._save_layout_template)
            self.layout_load_template_btn.clicked.connect(self._load_layout_template)
            self.layout_grab_background_btn.clicked.connect(self._grab_layout_background)
            self.layout_clear_background_btn.clicked.connect(self._clear_layout_background)
            self.layout_live_background.toggled.connect(self._save_from_ui)
            self.layout_background_opacity.valueChanged.connect(self._on_layout_background_opacity_changed)
            self.layout_add_region_btn.clicked.connect(self._add_layout_region)
            self.layout_remove_region_btn.clicked.connect(self._remove_layout_region)
            self.layout_clear_regions_btn.clicked.connect(self._clear_layout_regions)
            self.layout_region_list.currentRowChanged.connect(lambda _row: self._sync_region_selection_from_list())
            self.layout_canvas.selectionChanged.connect(self._on_layout_canvas_selection_changed)
            self.layout_canvas.regionsChanged.connect(self._on_layout_regions_changed)
            self.layout_region_desc.textChanged.connect(self._on_layout_region_editor_changed)
            self.layout_region_type.currentTextChanged.connect(self._on_layout_region_editor_changed)
            self.layout_region_text.textChanged.connect(self._on_layout_region_editor_changed)
            self.layout_region_palette.textChanged.connect(self._on_layout_region_editor_changed)
            for w in (self.layout_background, self.layout_style_photo, self.layout_aesthetics, self.layout_lighting, self.layout_medium):
                w.textChanged.connect(self._save_from_ui)
                w.textChanged.connect(self._update_layout_preview)
            self.layout_style.currentTextChanged.connect(self._save_from_ui)
            self.layout_style.currentTextChanged.connect(self._update_layout_preview)

        def _append_log(self, text: str) -> None:
            try:
                self.log_box.appendPlainText(str(text))
                sb = self.log_box.verticalScrollBar()
                sb.setValue(sb.maximum())
            except Exception:
                pass

        def log(self, text: str) -> None:
            self._log_signal.emit(str(text))

        def _browse_dir(self, edit, refresh: bool = False) -> None:
            start = edit.text().strip() or str(framevision_root())
            folder = _FVQtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
            if folder:
                edit.setText(folder)
                self._save_from_ui()
                if refresh:
                    self.refresh_files()

        def _browse_sdcli(self) -> None:
            start = str(Path(self.sd_cli_path.text().strip() or str(default_sd_cli_path())).parent)
            path, _ = _FVQtWidgets.QFileDialog.getOpenFileName(self, "Select sd-cli.exe", start, "sd-cli.exe (sd-cli.exe);;Executables (*.exe);;All files (*)")
            if path:
                self.sd_cli_path.setText(path)
                self._save_from_ui()

        def _browse_lora(self) -> None:
            current = self.lora_path.text().strip()
            start = str(Path(current).parent) if current else str(framevision_root() / "models")
            path, _ = _FVQtWidgets.QFileDialog.getOpenFileName(self, "Load LoRA", start, "LoRA files (*.safetensors *.ckpt);;All files (*)")
            if path:
                self.lora_path.setText(path)
                self.lora_enabled.setChecked(True)
                self._save_from_ui()

        def _on_lora_scale_changed(self, value: int) -> None:
            try:
                self.lora_scale_value.setText(f"{float(value) / 100.0:.2f}")
            except Exception:
                pass
            self._save_from_ui()

        def _choose_combo_text(self, combo, preferred: str) -> None:
            try:
                idx = combo.findText(str(preferred))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            except Exception:
                pass

        def refresh_files(self) -> None:
            folder = Path(self.gguf_dir.text().strip() or str(default_gguf_dir()))
            saved = load_saved_settings()
            pairs = [
                (self.diffusion_file, "diffusion", str(saved.get("gguf_diffusion_file") or "")),
                (self.uncond_file, "unconditional", str(saved.get("gguf_unconditional_file") or "")),
                (self.llm_file, "llm", str(saved.get("gguf_llm_file") or "")),
                (self.vae_file, "vae", str(saved.get("gguf_vae_file") or "")),
            ]
            changed_selection = False
            for combo, role, preferred in pairs:
                combo.blockSignals(True)
                try:
                    combo.clear()
                    names = _list_gguf_files_for_role(folder, role)
                    for name in names:
                        if name:
                            combo.addItem(name)
                    if preferred and preferred in names:
                        idx = combo.findText(preferred)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    elif names:
                        combo.setCurrentIndex(0)
                        changed_selection = True
                    else:
                        try:
                            combo.setEditText("")
                        except Exception:
                            pass
                        changed_selection = True
                finally:
                    combo.blockSignals(False)
            self.log(f"[Ideogram4] Scanned GGUF folder: {folder}")
            if changed_selection and not getattr(self, "_loading_settings", False):
                self._save_from_ui()

        def _load_settings_into_ui(self) -> None:
            s = load_saved_settings()
            def set_text(edit, key, default):
                try:
                    edit.setText(str(s.get(key) or default))
                except Exception:
                    pass
            set_text(self.gguf_dir, "gguf_dir", default_gguf_dir())
            set_text(self.sd_cli_path, "sd_cli_path", default_sd_cli_path())
            set_text(self.output_dir, "output_dir", default_output_dir())
            try: self.width.setValue(int(s.get("width", 1024)))
            except Exception: pass
            try: self.height.setValue(int(s.get("height", 1024)))
            except Exception: pass
            try:
                preset_value = str(s.get("preset", "Default 20") or "Default 20")
                idx = self.preset_combo.findText(preset_value)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)
            except Exception: pass
            try: self.steps.setValue(int(s.get("steps", 20)))
            except Exception: pass
            try: self.guidance.setValue(float(s.get("guidance", 4.0)))
            except Exception: pass
            try: self.seed.setValue(int(s.get("seed", -1)))
            except Exception: pass
            try: self.raw_prompt.setChecked(bool(s.get("raw_prompt", False)))
            except Exception: pass
            try: self.stream_layers.setChecked(bool(s.get("gguf_stream_layers", False)))
            except Exception: pass
            set_text(self.lora_path, "lora_path", "")
            try: self.lora_enabled.setChecked(bool(self.lora_path.text().strip()))
            except Exception: pass
            try: self.lora_scale.setValue(int(round(float(s.get("lora_scale", 1.0)) * 100.0)))
            except Exception: pass
            try: self.lora_scale_value.setText(f"{float(self.lora_scale.value()) / 100.0:.2f}")
            except Exception: pass
            try:
                idx = self.lora_apply_mode.findText("auto")
                if idx >= 0:
                    self.lora_apply_mode.setCurrentIndex(idx)
            except Exception: pass
            try: self.use_framevision_queue.setChecked(bool(s.get("use_framevision_queue", False)))
            except Exception: pass
            try:
                self.prompt.setPlainText(str(s.get("last_prompt", "") or ""))
                self.negative.setPlainText(str(s.get("negative", "") or ""))
            except Exception:
                pass
            try: self.enable_layout_prompt.setChecked(bool(s.get("layout_prompt_enabled", False)))
            except Exception: pass
            set_text(self.layout_background, "layout_background", "")
            set_text(self.layout_style_photo, "layout_style_photo", "")
            set_text(self.layout_aesthetics, "layout_aesthetics", "")
            set_text(self.layout_lighting, "layout_lighting", "")
            set_text(self.layout_medium, "layout_medium", "")
            try:
                style_value = str(s.get("layout_style", "photo") or "photo")
                if style_value == "art style":
                    style_value = "art_style"
                idx = self.layout_style.findText(style_value)
                if idx >= 0:
                    self.layout_style.setCurrentIndex(idx)
            except Exception:
                pass
            try:
                regions = s.get("layout_regions", [])
                self.layout_regions = list(regions) if isinstance(regions, list) else []
            except Exception:
                self.layout_regions = []
            try:
                opacity = int(float(s.get("layout_background_preview_opacity", 35) or 35))
                opacity = max(0, min(100, opacity))
                self.layout_background_opacity.setValue(opacity)
                self.layout_canvas.set_background_opacity(opacity)
                self.layout_background_opacity_label.setText(f"BG opacity: {opacity}%")
            except Exception:
                pass
            try:
                self.layout_live_background.setChecked(bool(s.get("layout_background_preview_live", False)))
            except Exception:
                pass
            try:
                bg_path = str(s.get("layout_background_preview_path", "") or "").strip()
                if bg_path and Path(bg_path).exists():
                    self.layout_canvas.set_background_image(bg_path)
            except Exception:
                pass
            self._refresh_layout_ui(keep_preview=True)
            self._apply_framevision_queue_mode()

        def _save_from_ui(self) -> None:
            if getattr(self, "_loading_settings", False):
                return
            try:
                s = load_saved_settings()
                s.update({
                    "backend": "gguf",
                    "gguf_dir": self.gguf_dir.text().strip(),
                    "sd_cli_path": self.sd_cli_path.text().strip(),
                    "output_dir": self.output_dir.text().strip(),
                    "gguf_diffusion_file": self.diffusion_file.currentText().strip(),
                    "gguf_unconditional_file": self.uncond_file.currentText().strip(),
                    "gguf_llm_file": self.llm_file.currentText().strip(),
                    "gguf_vae_file": self.vae_file.currentText().strip(),
                    "width": int(self.width.value()),
                    "height": int(self.height.value()),
                    "preset": self.preset_combo.currentText().strip(),
                    "steps": int(self.steps.value()),
                    "guidance": float(self.guidance.value()),
                    "seed": int(self.seed.value()),
                    "gguf_max_vram": 0.0,
                    "gguf_stream_layers": bool(self.stream_layers.isChecked()),
                    "lora_enabled": bool(self.lora_path.text().strip()),
                    "lora_path": self.lora_path.text().strip(),
                    "lora_scale": float(self.lora_scale.value()) / 100.0,
                    "lora_apply_mode": "auto",
                    "raw_prompt": bool(self.raw_prompt.isChecked()),
                    "use_framevision_queue": bool(self.use_framevision_queue.isChecked()),
                    "last_prompt": self.prompt.toPlainText(),
                    "negative": self.negative.toPlainText(),
                    "layout_prompt_enabled": bool(self.enable_layout_prompt.isChecked()),
                    "layout_background": self.layout_background.text().strip(),
                    "layout_style": self.layout_style.currentText().strip(),
                    "layout_style_photo": self.layout_style_photo.text().strip(),
                    "layout_aesthetics": self.layout_aesthetics.text().strip(),
                    "layout_lighting": self.layout_lighting.text().strip(),
                    "layout_medium": self.layout_medium.text().strip(),
                    "layout_regions": self._collect_layout_regions(),
                    "layout_background_preview_path": self.layout_canvas.background_path(),
                    "layout_background_preview_live": bool(self.layout_live_background.isChecked()),
                    "layout_background_preview_opacity": int(self.layout_background_opacity.value()),
                })
                save_saved_settings(s)
            except Exception:
                pass

        def _set_queue_tab_visible(self, visible: bool) -> None:
            idx = self.tabs.indexOf(self.queue_tab)
            try:
                bar = self.tabs.tabBar()
            except Exception:
                bar = None
            if idx >= 0:
                try:
                    if bar is not None and hasattr(bar, "setTabVisible"):
                        bar.setTabVisible(idx, bool(visible))
                        return
                except Exception:
                    pass
                if not visible:
                    try:
                        self._queue_tab_hidden_index = idx
                        self.tabs.removeTab(idx)
                    except Exception:
                        pass
                return
            if visible:
                try:
                    idx = int(getattr(self, "_queue_tab_hidden_index", 1))
                except Exception:
                    idx = 1
                try:
                    self.tabs.insertTab(min(max(0, idx), self.tabs.count()), self.queue_tab, "Queue")
                except Exception:
                    try:
                        self.tabs.addTab(self.queue_tab, "Queue")
                    except Exception:
                        pass

        def _apply_framevision_queue_mode(self) -> None:
            enabled = bool(getattr(self, 'use_framevision_queue', None) and self.use_framevision_queue.isChecked())
            try:
                self.generate_btn.setVisible(not enabled)
                self.add_queue_btn.setVisible(not enabled)
                self.framevision_queue_btn.setVisible(enabled)
            except Exception:
                pass
            try:
                self.generate_right_panel.setVisible(not enabled)
            except Exception:
                pass
            try:
                if enabled and self.tabs.currentWidget() == self.queue_tab:
                    self.tabs.setCurrentWidget(self.generate_tab)
            except Exception:
                pass
            self._set_queue_tab_visible(not enabled)
            try:
                if enabled:
                    self.generate_splitter.setSizes([1000, 0])
                else:
                    self.generate_splitter.setSizes([680, 540])
            except Exception:
                pass

        def _on_framevision_queue_toggled(self, *_args) -> None:
            self._apply_framevision_queue_mode()
            self._save_from_ui()

        def _switch_to_main_queue_tab(self) -> None:
            try:
                win = self.window()
                for tabs in win.findChildren(_FVQtWidgets.QTabWidget):
                    try:
                        for i in range(tabs.count()):
                            if tabs.tabText(i).strip().lower() == 'queue':
                                tabs.setCurrentIndex(i)
                                return
                    except Exception:
                        pass
            except Exception:
                pass

        def add_current_to_framevision_queue(self) -> None:
            if not self.prompt.toPlainText().strip():
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", "Add a prompt first.")
                return
            self._save_from_ui()
            cfg = self._build_config()
            payload = {
                'prompt': cfg.prompt,
                'negative': cfg.negative,
                'width': int(cfg.width),
                'height': int(cfg.height),
                'steps': int(cfg.steps),
                'guidance': float(cfg.guidance),
                'preset': str(cfg.preset or 'Custom'),
                'seed': int(cfg.seed),
                'raw_prompt': bool(cfg.raw_prompt),
                'gguf_dir': str(cfg.gguf_dir),
                'gguf_diffusion_file': str(cfg.gguf_diffusion_file or ''),
                'gguf_unconditional_file': str(cfg.gguf_unconditional_file or ''),
                'gguf_llm_file': str(cfg.gguf_llm_file or ''),
                'gguf_vae_file': str(cfg.gguf_vae_file or ''),
                'sd_cli_path': str(cfg.sd_cli_path),
                'gguf_stream_layers': bool(cfg.gguf_stream_layers),
                'lora_enabled': bool(cfg.lora_enabled),
                'lora_path': str(cfg.lora_path or ''),
                'lora_scale': float(cfg.lora_scale),
                'lora_apply_mode': _lora_apply_mode_value(getattr(cfg, 'lora_apply_mode', 'auto')),
                'output_dir': str(Path(self.output_dir.text().strip() or str(default_output_dir()))),
            }
            try:
                try:
                    from helpers.queue_adapter import enqueue_ideogram4_generate as _enq
                except Exception:
                    from queue_adapter import enqueue_ideogram4_generate as _enq
                try:
                    jid = _enq(payload, priority=610)
                    self.log(f"[FrameVision queue] queued: {jid}")
                except Exception as exc:
                    # Some older/broken queue_adapter builds can throw
                    # NameError("inner") from unrelated widget enqueue helpers.
                    # Do not block Ideogram for that; write the normal pending
                    # job directly from the settings we already collected here.
                    if isinstance(exc, NameError) and "inner" in str(exc):
                        self.log(f"[FrameVision queue] queue_adapter failed ({exc}); using Ideogram local queue fallback.")
                        jid = _enqueue_ideogram4_generate_local(payload, priority=610)
                        self.log(f"[FrameVision queue] queued with fallback: {jid}")
                    else:
                        raise
                try:
                    self._switch_to_main_queue_tab()
                except Exception:
                    pass
            except Exception as exc:
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", f"Could not add job to FrameVision queue.\n\n{exc}")

        def _build_config(self) -> GenerateConfig:
            seed = int(self.seed.value())
            if seed < 0:
                seed = secrets.randbelow(2147483646) + 1
            out_dir = Path(self.output_dir.text().strip() or str(default_output_dir()))
            out = output_path(out_dir, "ideogram4_gguf")
            prompt_text = self.prompt.toPlainText().strip()
            layout_enabled = self._layout_prompt_enabled()
            if layout_enabled:
                if not self._collect_layout_regions():
                    raise ValueError("Add at least one layout region before generating with the layout prompt builder.")
                prompt_text = self._build_layout_prompt_json()
            return GenerateConfig(
                prompt=prompt_text,
                negative=self.negative.toPlainText().strip(),
                model_dir=default_model_dir(),
                output=out,
                width=int(self.width.value()),
                height=int(self.height.value()),
                steps=int(self.steps.value()),
                guidance=float(self.guidance.value()),
                preset=self.preset_combo.currentText().strip() or "Custom",
                seed=seed,
                compile_sdnq=False,
                raw_prompt=bool(True if layout_enabled else self.raw_prompt.isChecked()),
                text_encoder_cpu_offload=False,
                backend="gguf",
                gguf_dir=Path(self.gguf_dir.text().strip() or str(default_gguf_dir())),
                gguf_diffusion_file=self.diffusion_file.currentText().strip(),
                gguf_unconditional_file=self.uncond_file.currentText().strip(),
                gguf_llm_file=self.llm_file.currentText().strip(),
                gguf_vae_file=self.vae_file.currentText().strip(),
                sd_cli_path=Path(self.sd_cli_path.text().strip() or str(default_sd_cli_path())),
                gguf_max_vram=0.0,
                gguf_stream_layers=bool(self.stream_layers.isChecked()),
                lora_enabled=bool(self.lora_path.text().strip()),
                lora_path=self.lora_path.text().strip(),
                lora_scale=float(self.lora_scale.value()) / 100.0,
                lora_apply_mode="auto",
            )

        def _config_to_job(self, cfg: GenerateConfig) -> dict[str, Any]:
            return {
                "id": f"ig4_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                "status": "pending",
                "progress": "",
                "prompt": cfg.prompt,
                "negative": cfg.negative,
                "width": cfg.width,
                "height": cfg.height,
                "steps": cfg.steps,
                "guidance": cfg.guidance,
                "preset": cfg.preset,
                "seed": cfg.seed,
                "raw_prompt": cfg.raw_prompt,
                "gguf_dir": str(cfg.gguf_dir),
                "gguf_diffusion_file": cfg.gguf_diffusion_file,
                "gguf_unconditional_file": cfg.gguf_unconditional_file,
                "gguf_llm_file": cfg.gguf_llm_file,
                "gguf_vae_file": cfg.gguf_vae_file,
                "sd_cli_path": str(cfg.sd_cli_path),
                "gguf_max_vram": cfg.gguf_max_vram,
                "gguf_stream_layers": cfg.gguf_stream_layers,
                "lora_enabled": bool(cfg.lora_enabled),
                "lora_path": str(cfg.lora_path or ""),
                "lora_scale": float(cfg.lora_scale),
                "lora_apply_mode": _lora_apply_mode_value(getattr(cfg, "lora_apply_mode", "auto")),
                "output_dir": str(Path(cfg.output).parent),
                "output": str(cfg.output),
                "created": datetime.now().strftime("%H:%M:%S"),
                "started": "",
                "finished": "",
                "duration": "",
                "error": "",
            }

        def _job_to_config(self, job: dict[str, Any]) -> GenerateConfig:
            out_dir = Path(job.get("output_dir") or self.output_dir.text().strip() or str(default_output_dir()))
            output = Path(job.get("output") or output_path(out_dir, "ideogram4_gguf"))
            return GenerateConfig(
                prompt=str(job.get("prompt") or ""),
                negative=str(job.get("negative") or ""),
                model_dir=default_model_dir(),
                output=output,
                width=int(job.get("width") or 1024),
                height=int(job.get("height") or 1024),
                steps=int(job.get("steps") or 20),
                guidance=float(job.get("guidance") or 4.0),
                preset=str(job.get("preset") or "Custom"),
                seed=int(job.get("seed") or -1),
                compile_sdnq=False,
                raw_prompt=bool(job.get("raw_prompt", False)),
                text_encoder_cpu_offload=False,
                backend="gguf",
                gguf_dir=Path(job.get("gguf_dir") or self.gguf_dir.text().strip() or str(default_gguf_dir())),
                gguf_diffusion_file=str(job.get("gguf_diffusion_file") or self.diffusion_file.currentText().strip()),
                gguf_unconditional_file=str(job.get("gguf_unconditional_file") or self.uncond_file.currentText().strip()),
                gguf_llm_file=str(job.get("gguf_llm_file") or self.llm_file.currentText().strip()),
                gguf_vae_file=str(job.get("gguf_vae_file") or self.vae_file.currentText().strip()),
                sd_cli_path=Path(job.get("sd_cli_path") or self.sd_cli_path.text().strip() or str(default_sd_cli_path())),
                gguf_max_vram=float(job.get("gguf_max_vram") or 0.0),
                gguf_stream_layers=bool(job.get("gguf_stream_layers", False)),
                lora_enabled=bool(job.get("lora_enabled", False)),
                lora_path=str(job.get("lora_path") or ""),
                lora_scale=float(job.get("lora_scale") or 1.0),
                lora_apply_mode=_lora_apply_mode_value(job.get("lora_apply_mode", "auto")),
            )

        def generate_clicked(self) -> None:
            if self._worker_thread and self._worker_thread.is_alive():
                _FVQtWidgets.QMessageBox.information(self, "Ideogram 4 GGUF", "A generation is already running.")
                return
            if not self.prompt.toPlainText().strip():
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", "Add a prompt first.")
                return
            self._save_from_ui()
            cfg = self._build_config()
            self.generate_btn.setEnabled(False)
            self.log("=" * 60)
            self.log("[Ideogram4] Starting direct generation inside FrameVision...")

            def worker():
                try:
                    out, used_seed, notes = generate_once(cfg, log=self.log)
                    self._done_signal.emit(str(out), f"Seed used: {used_seed}\n{notes}")
                except Exception as exc:
                    self._error_signal.emit(str(exc))

            self._worker_thread = threading.Thread(target=worker, daemon=True)
            self._worker_thread.start()

        def add_current_to_queue(self) -> None:
            if not self.prompt.toPlainText().strip():
                _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", "Add a prompt first.")
                return
            self._save_from_ui()
            job = self._config_to_job(self._build_config())
            self.jobs.append(job)
            self._save_queue_state()
            self._refresh_queue_table()
            self.tabs.setCurrentWidget(self.queue_tab)
            self.log(f"[queue] Added job: {job.get('id')}")

        def _save_queue_state(self) -> None:
            try:
                self.queue_data["jobs"] = self.jobs
                save_queue_state(self.queue_data)
            except Exception as exc:
                self.log(f"[queue] Could not save queue: {exc}")

        def _refresh_queue_table(self) -> None:
            try:
                self.queue_table.setRowCount(len(self.jobs))
                for row, job in enumerate(self.jobs):
                    values = [
                        job.get("status", ""),
                        job.get("progress", ""),
                        job.get("started", ""),
                        job.get("duration", ""),
                        job.get("finished", ""),
                        f"{job.get('width', '')}x{job.get('height', '')}",
                        str(job.get("prompt", ""))[:180],
                    ]
                    for col, value in enumerate(values):
                        self.queue_table.setItem(row, col, _FVQtWidgets.QTableWidgetItem(str(value)))
                try:
                    self.queue_table.resizeColumnsToContents()
                except Exception:
                    pass
            except Exception:
                pass

        def start_queue(self) -> None:
            if self._queue_worker_thread and self._queue_worker_thread.is_alive():
                _FVQtWidgets.QMessageBox.information(self, "Ideogram 4 GGUF", "Queue is already running.")
                return
            self._queue_stop_requested = False
            self.tabs.setCurrentWidget(self.queue_tab)
            self.log("[queue] Starting Ideogram queue...")

            def worker():
                for job in self.jobs:
                    if self._queue_stop_requested:
                        break
                    if str(job.get("status")) not in ("pending", "failed"):
                        continue
                    started = time.time()
                    job["status"] = "running"
                    job["started"] = datetime.now().strftime("%H:%M:%S")
                    job["finished"] = ""
                    job["duration"] = ""
                    job["error"] = ""
                    self._save_queue_state()
                    self._queue_changed_signal.emit()
                    cfg = self._job_to_config(job)
                    try:
                        out, used_seed, notes = generate_once(cfg, log=self.log)
                        job["status"] = "done"
                        job["output"] = str(out)
                        job["seed"] = int(used_seed)
                        job["progress"] = f"{job.get('steps', cfg.steps)}/{job.get('steps', cfg.steps)}"
                        self._last_output = str(out)
                        self._done_signal.emit(str(out), f"Queue job done: {job.get('id')}\nSeed used: {used_seed}\n{notes}")
                    except Exception as exc:
                        job["status"] = "failed"
                        job["error"] = str(exc)
                        self.log(f"[queue:error] {job.get('id')}: {exc}")
                    finally:
                        job["finished"] = datetime.now().strftime("%H:%M:%S")
                        job["duration"] = self._format_duration(time.time() - started)
                        self._save_queue_state()
                        self._queue_changed_signal.emit()
                self.log("[queue] Queue stopped/finished.")

            self._queue_worker_thread = threading.Thread(target=worker, daemon=True)
            self._queue_worker_thread.start()

        def stop_queue_after_current(self) -> None:
            self._queue_stop_requested = True
            self.log("[queue] Stop requested. Current running job will finish first.")

        def clear_finished_queue_jobs(self) -> None:
            self.jobs[:] = [j for j in self.jobs if str(j.get("status")) not in ("done", "failed", "cancelled")]
            self._save_queue_state()
            self._refresh_queue_table()

        def _format_duration(self, seconds: float) -> str:
            seconds = max(0, int(seconds))
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m}:{s:02d}"

        def _on_done(self, out: str, notes: str) -> None:
            self.generate_btn.setEnabled(True)
            self._last_output = out
            self.log(f"[done] {out}")
            if notes:
                self.log(notes)
            self._load_preview(out)
            try:
                if self.layout_live_background.isChecked():
                    self._set_layout_background_preview(out)
            except Exception:
                pass

        def _on_error(self, msg: str) -> None:
            self.generate_btn.setEnabled(True)
            self.log(f"[error] {msg}")
            _FVQtWidgets.QMessageBox.warning(self, "Ideogram 4 GGUF", msg)

        def _load_preview(self, path: str) -> None:
            try:
                pix = _FVQtGui.QPixmap(path)
                if pix.isNull():
                    self.preview.setText(f"Saved, but preview could not load:\n{path}")
                    return
                scaled = pix.scaled(self.preview.size(), _FVQtCore.Qt.KeepAspectRatio, _FVQtCore.Qt.SmoothTransformation)
                self.preview.setPixmap(scaled)
                self.preview.setToolTip(path)
            except Exception:
                self.preview.setText(str(path))

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._last_output:
                self._load_preview(self._last_output)

        def view_results(self) -> None:
            path = Path(self.output_dir.text().strip() or str(default_output_dir()))
            _fv_open_results_in_media_explorer(self, path, preset="images")

        def open_output_folder(self) -> None:
            path = Path(self.output_dir.text().strip() or str(default_output_dir()))
            path.mkdir(parents=True, exist_ok=True)
            try:
                _FVQtGui.QDesktopServices.openUrl(_FVQtCore.QUrl.fromLocalFile(str(path)))
            except Exception:
                pass

        def open_last_image(self) -> None:
            if not self._last_output:
                self.open_output_folder()
                return
            try:
                _FVQtGui.QDesktopServices.openUrl(_FVQtCore.QUrl.fromLocalFile(str(Path(self._last_output))))
            except Exception:
                pass

        def closeEvent(self, event):
            self._save_from_ui()
            self._save_queue_state()
            super().closeEvent(event)
else:
    class Ideogram4Widget:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PySide6 is required for embedded Ideogram4Widget")

def launch_gui() -> int:
    ensure_framevision_env()
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("Ideogram 4 SDNQ Helper")
            self.root.geometry("1220x860")
            self.root.minsize(1000, 760)
            self.msg_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
            self.queue_worker: threading.Thread | None = None
            self.preview_ref = None
            self.preview_image_original = None
            self.preview_image_path: Path | None = None
            self.preview_zoom = 1.0
            self.preview_canvas_image_id = None
            self.preview_message_id = None
            self.last_output: Path | None = None
            self.session_log_path = new_session_log_path()
            self.jobs_lock = threading.RLock()
            self.queue_data = load_queue_state()
            self.jobs: list[dict[str, Any]] = self.queue_data.setdefault("jobs", [])
            self.active_job_id: str | None = None
            self.shutdown_requested = False

            # Any job that was still running when the helper last closed is no longer active.
            changed = False
            for job in self.jobs:
                if job.get("status") in {"running", "cancelling"}:
                    job["status"] = "failed"
                    job["finished_at"] = job.get("finished_at") or _now_iso()
                    job["error"] = job.get("error") or "Interrupted when the helper closed."
                    changed = True
            if changed:
                self._save_queue_state_silent()

            self.prompt_var = tk.StringVar()
            self.negative_var = tk.StringVar()
            self.width_var = tk.StringVar(value="1024")
            self.height_var = tk.StringVar(value="1024")
            self.steps_var = tk.StringVar(value="20")
            self.guidance_var = tk.StringVar(value="4.0")
            self.preset_var = tk.StringVar(value="Default 20")
            self.seed_var = tk.StringVar(value="-1")
            self.backend_var = tk.StringVar(value="sdnq")
            self.model_dir_var = tk.StringVar(value=str(default_model_dir()))
            self.gguf_dir_var = tk.StringVar(value=str(default_gguf_dir()))
            self.gguf_diffusion_var = tk.StringVar(value=DEFAULT_GGUF_FILES["diffusion"])
            self.gguf_unconditional_var = tk.StringVar(value=DEFAULT_GGUF_FILES["unconditional"])
            self.gguf_llm_var = tk.StringVar(value=DEFAULT_GGUF_FILES["llm"])
            self.gguf_vae_var = tk.StringVar(value=DEFAULT_GGUF_FILES["vae"])
            self.sd_cli_path_var = tk.StringVar(value=str(default_sd_cli_path()))
            self.gguf_max_vram_var = tk.StringVar(value="0")
            self.gguf_stream_layers_var = tk.BooleanVar(value=False)
            self.output_dir_var = tk.StringVar(value=str(default_output_dir()))
            self.compile_var = tk.BooleanVar(value=False)
            self.text_encoder_offload_var = tk.BooleanVar(value=False)
            self.raw_prompt_var = tk.BooleanVar(value=False)
            self.layout_prompt_var = tk.BooleanVar(value=False)
            self.layout_background_var = tk.StringVar(value="")
            self.layout_style_var = tk.StringVar(value="photo")
            self.layout_style_photo_var = tk.StringVar(value="")
            self.layout_aesthetics_var = tk.StringVar(value="")
            self.layout_lighting_var = tk.StringVar(value="")
            self.layout_medium_var = tk.StringVar(value="")
            self.layout_region_type_var = tk.StringVar(value="obj")
            self.layout_region_text_var = tk.StringVar(value="")
            self.layout_region_palette_var = tk.StringVar(value="")
            self.layout_regions: list[dict[str, Any]] = []
            self.layout_selected_index = -1
            self.layout_drag_mode = ""
            self.layout_drag_start: tuple[float, float] | None = None
            self.layout_drag_region: dict[str, Any] | None = None
            self.theme_var = tk.StringVar(value=THEME_LABELS["dark"])
            self.hud_var = tk.StringVar(value="DDR: -- / --   VRAM: -- / --   CPU: --%")
            self.status_var = tk.StringVar(value="Ready")
            self._loading_settings = True

            self._build_ui()
            self._load_settings()
            self.root.after(250, self._restore_generate_splitter)
            self._autofill_gguf_install_paths()
            self._refresh_gguf_file_dropdowns(save_fallback=False)
            self._update_backend_visibility()
            self._apply_theme()
            self._loading_settings = False
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.after(200, self._poll_queue)
            self.root.after(1000, self._tick_queue_times)
            self.root.after(1000, self._update_hud)
            self._refresh_queue_tree()
            self._maybe_start_queue_worker()
            self._write_log("Ready. Enter a prompt and click Generate to add it to the queue.")
            self._write_log(f"Queue state: {queue_state_path()}")
            self._write_log(f"Detailed session log: {self.session_log_path}")

        def _build_ui(self):
            topbar = ttk.Frame(self.root, padding=(10, 8, 10, 4))
            topbar.pack(fill="x")
            topbar.columnconfigure(0, weight=1)
            self.hud_label = ttk.Label(topbar, textvariable=self.hud_var, anchor="w")
            self.hud_label.grid(row=0, column=0, sticky="ew")
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill="both", expand=True)
            self.generate_tab = ttk.Frame(notebook)
            self.queue_tab = ttk.Frame(notebook)
            self.settings_tab = ttk.Frame(notebook)
            notebook.add(self.generate_tab, text="Generate")
            notebook.add(self.queue_tab, text="Queue")
            notebook.add(self.settings_tab, text="Settings")

            self.generate_tab.columnconfigure(0, weight=1)
            self.generate_tab.rowconfigure(0, weight=1)
            self.generate_tab.rowconfigure(1, weight=0)

            self.generate_canvas = tk.Canvas(self.generate_tab, highlightthickness=0)
            self.generate_canvas.grid(row=0, column=0, sticky="nsew")
            self.generate_vscroll = ttk.Scrollbar(self.generate_tab, orient="vertical", command=self.generate_canvas.yview)
            self.generate_vscroll.grid(row=0, column=1, sticky="ns")
            self.generate_canvas.configure(yscrollcommand=self._generate_yscroll_changed)

            outer = ttk.Frame(self.generate_canvas, padding=10)
            self.generate_canvas_window = self.generate_canvas.create_window((0, 0), window=outer, anchor="nw")
            outer.bind("<Configure>", self._generate_scroll_region_changed)
            self.generate_canvas.bind("<Configure>", self._generate_canvas_configured)
            self.generate_canvas.bind("<Enter>", self._bind_generate_mousewheel)
            self.generate_canvas.bind("<Leave>", self._unbind_generate_mousewheel)
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)

            self.generate_paned = ttk.Panedwindow(outer, orient="horizontal")
            self.generate_paned.grid(row=0, column=0, sticky="nsew")
            ToolTip(self.generate_paned, "Drag the divider to resize the generation controls and preview pane.")

            left = ttk.Frame(self.generate_paned)
            left.columnconfigure(0, weight=1)

            right = ttk.Frame(self.generate_paned)
            right.columnconfigure(0, weight=1)
            right.rowconfigure(0, weight=1)

            try:
                self.generate_paned.add(left, weight=3)
                self.generate_paned.add(right, weight=2)
            except Exception:
                self.generate_paned.add(left)
                self.generate_paned.add(right)
            self.generate_paned.bind("<ButtonRelease-1>", lambda event: self._save_settings(), add="+")

            prompt_box = ttk.LabelFrame(left, text="Prompt", padding=10)
            prompt_box.grid(row=0, column=0, sticky="nsew")
            prompt_box.columnconfigure(0, weight=1)
            self.prompt_text = tk.Text(prompt_box, height=10, wrap="word")
            self.prompt_text.grid(row=0, column=0, sticky="nsew")
            self._install_edit_menu(self.prompt_text)
            ToolTip(self.prompt_text, "Main prompt. Plain text is automatically wrapped into the Ideogram JSON caption format unless Raw prompt / JSON is enabled.")

            negative_box = ttk.LabelFrame(left, text="Negative prompt (optional)", padding=10)
            negative_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            negative_box.columnconfigure(0, weight=1)
            self.negative_entry = ttk.Entry(negative_box, textvariable=self.negative_var)
            self.negative_entry.grid(row=0, column=0, sticky="ew")
            self._install_edit_menu(self.negative_entry)
            ToolTip(self.negative_entry, "Optional negative prompt. Leave empty if not needed.")

            settings = ttk.LabelFrame(left, text="Generation", padding=10)
            settings.grid(row=2, column=0, sticky="ew", pady=(10, 0))
            for col in range(4):
                settings.columnconfigure(col, weight=1)

            self.backend_label = ttk.Label(settings, text="Backend")
            self.backend_label.grid(row=0, column=0, sticky="w")
            self.backend_combo = ttk.Combobox(settings, textvariable=self.backend_var, values=["sdnq", "gguf"], state="readonly")
            self.backend_combo.grid(row=1, column=0, sticky="ew", padx=(0, 6))
            self.backend_combo.bind("<<ComboboxSelected>>", self._backend_changed)
            ToolTip(self.backend_label, "Choose quants / SDNQ or GGUF / sd-cli. Default: sdnq.")
            ToolTip(self.backend_combo, "GGUF uses presets/bin/sd-cli.exe and models/ideogram4_gguf.")

            self.preset_label = ttk.Label(settings, text="Preset")
            self.preset_label.grid(row=0, column=1, sticky="w")
            self.preset_combo = ttk.Combobox(settings, textvariable=self.preset_var, values=["Turbo 12", "Default 20", "Quality 48", "Custom"], state="readonly")
            self.preset_combo.grid(row=1, column=1, sticky="ew", padx=(0, 6))
            self.preset_combo.bind("<<ComboboxSelected>>", self._preset_changed)
            ToolTip(self.preset_label, "Quick quality preset. Defaults: Turbo 12 = fast, Default 20 = balanced, Quality 48 = best quality but slowest.")
            ToolTip(self.preset_combo, "Choose a preset. Default: Default 20.")

            self.width_label = ttk.Label(settings, text="Width")
            self.width_label.grid(row=0, column=2, sticky="w")
            self.height_label = ttk.Label(settings, text="Height")
            self.height_label.grid(row=0, column=3, sticky="w")
            self.steps_label = ttk.Label(settings, text="Steps")
            self.steps_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
            self.width_entry = ttk.Entry(settings, textvariable=self.width_var)
            self.width_entry.grid(row=1, column=2, sticky="ew", padx=(0, 6))
            self._install_edit_menu(self.width_entry)
            self.height_entry = ttk.Entry(settings, textvariable=self.height_var)
            self.height_entry.grid(row=1, column=3, sticky="ew")
            self._install_edit_menu(self.height_entry)
            self.steps_entry = ttk.Entry(settings, textvariable=self.steps_var)
            self.steps_entry.grid(row=3, column=0, sticky="ew", padx=(0, 6))
            self._install_edit_menu(self.steps_entry)
            ToolTip(self.width_label, "Image width in pixels. Default: 1024.")
            ToolTip(self.width_entry, "Image width in pixels. Default: 1024.")
            ToolTip(self.height_label, "Image height in pixels. Default: 1024.")
            ToolTip(self.height_entry, "Image height in pixels. Default: 1024.")
            ToolTip(self.steps_label, "Sampling steps. Defaults: 12 Turbo, 20 Default, 48 Quality.")
            ToolTip(self.steps_entry, "Sampling steps. Higher usually means slower but can improve quality.")

            self.guidance_label = ttk.Label(settings, text="Guidance")
            self.guidance_label.grid(row=2, column=1, sticky="w", pady=(10, 0))
            self.seed_label = ttk.Label(settings, text="Seed (-1 = random)")
            self.seed_label.grid(row=2, column=2, sticky="w", pady=(10, 0))
            self.guidance_entry = ttk.Entry(settings, textvariable=self.guidance_var)
            self.guidance_entry.grid(row=3, column=1, sticky="ew", padx=(0, 6))
            self._install_edit_menu(self.guidance_entry)
            self.seed_entry = ttk.Entry(settings, textvariable=self.seed_var)
            self.seed_entry.grid(row=3, column=2, sticky="ew", padx=(0, 6))
            self._install_edit_menu(self.seed_entry)
            self.raw_prompt_check = ttk.Checkbutton(settings, text="Raw prompt / JSON", variable=self.raw_prompt_var)
            self.raw_prompt_check.grid(row=3, column=3, sticky="w")
            self.compile_check = ttk.Checkbutton(settings, text="Compile SDNQ", variable=self.compile_var)
            self.compile_check.grid(row=4, column=0, sticky="w", pady=(8, 0))
            self.text_encoder_offload_check = ttk.Checkbutton(settings, text="Text encoder CPU offload", variable=self.text_encoder_offload_var)
            self.text_encoder_offload_check.grid(row=4, column=1, columnspan=2, sticky="w", pady=(8, 0))
            ToolTip(self.guidance_label, "Default: 4.0")
            ToolTip(self.guidance_entry, "Default: 4.0")
            ToolTip(self.seed_label, "Seed. Default: -1 which means random seed each run.")
            ToolTip(self.seed_entry, "Seed. Use -1 for random, or enter a fixed number for repeatability.")
            ToolTip(self.raw_prompt_check, "If enabled, the prompt is sent exactly as typed. If disabled, plain text is wrapped into Ideogram JSON automatically.")
            ToolTip(self.compile_check, "Try SDNQ compile / quantized matmul acceleration. Default: off. Can help on some systems, but keep off if unsure.")
            ToolTip(self.text_encoder_offload_check, "Experimental low-VRAM mode. Moves the Qwen text encoder back to CPU after prompt encoding so denoising has more VRAM. Slower, but may reduce shared-memory spill.")

            layout_box = ttk.LabelFrame(left, text="Layout / region prompt builder", padding=10)
            layout_box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
            layout_box.columnconfigure(0, weight=1)
            layout_box.columnconfigure(1, weight=1)
            layout_box.rowconfigure(4, weight=1)

            layout_top = ttk.Frame(layout_box)
            layout_top.grid(row=0, column=0, columnspan=2, sticky="ew")
            layout_top.columnconfigure(1, weight=1)
            self.layout_prompt_check = ttk.Checkbutton(layout_top, text="Enable layout prompt builder", variable=self.layout_prompt_var, command=self._layout_settings_changed)
            self.layout_prompt_check.grid(row=0, column=0, sticky="w")
            ttk.Label(layout_top, text="Drag boxes to place objects/text like the Ideogram KJ prompt-builder workflow.").grid(row=0, column=1, sticky="w", padx=(12, 0))

            layout_style = ttk.Frame(layout_box)
            layout_style.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            for col in (1, 3):
                layout_style.columnconfigure(col, weight=1)
            ttk.Label(layout_style, text="Background").grid(row=0, column=0, sticky="w")
            self.layout_background_entry = ttk.Entry(layout_style, textvariable=self.layout_background_var)
            self.layout_background_entry.grid(row=0, column=1, sticky="ew", padx=(4, 10))
            ttk.Label(layout_style, text="Style").grid(row=0, column=2, sticky="w")
            self.layout_style_combo = ttk.Combobox(layout_style, textvariable=self.layout_style_var, values=["none", "photo", "art_style"], state="readonly")
            self.layout_style_combo.grid(row=0, column=3, sticky="ew")
            ttk.Label(layout_style, text="Style.photo").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self.layout_style_photo_entry = ttk.Entry(layout_style, textvariable=self.layout_style_photo_var)
            self.layout_style_photo_entry.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(6, 0))
            ttk.Label(layout_style, text="Aesthetics").grid(row=1, column=2, sticky="w", pady=(6, 0))
            self.layout_aesthetics_entry = ttk.Entry(layout_style, textvariable=self.layout_aesthetics_var)
            self.layout_aesthetics_entry.grid(row=1, column=3, sticky="ew", pady=(6, 0))
            ttk.Label(layout_style, text="Lighting").grid(row=2, column=0, sticky="w", pady=(6, 0))
            self.layout_lighting_entry = ttk.Entry(layout_style, textvariable=self.layout_lighting_var)
            self.layout_lighting_entry.grid(row=2, column=1, sticky="ew", padx=(4, 10), pady=(6, 0))
            ttk.Label(layout_style, text="Medium").grid(row=2, column=2, sticky="w", pady=(6, 0))
            self.layout_medium_entry = ttk.Entry(layout_style, textvariable=self.layout_medium_var)
            self.layout_medium_entry.grid(row=2, column=3, sticky="ew", pady=(6, 0))

            layout_buttons = ttk.Frame(layout_box)
            layout_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            self.layout_add_btn = ttk.Button(layout_buttons, text="Add region", command=self._layout_add_region)
            self.layout_add_btn.grid(row=0, column=0, sticky="w", padx=(0, 6))
            self.layout_remove_btn = ttk.Button(layout_buttons, text="Remove selected", command=self._layout_remove_region)
            self.layout_remove_btn.grid(row=0, column=1, sticky="w", padx=(0, 6))
            self.layout_clear_btn = ttk.Button(layout_buttons, text="Clear all", command=self._layout_clear_regions)
            self.layout_clear_btn.grid(row=0, column=2, sticky="w", padx=(0, 6))
            self.layout_preview_prompt_btn = ttk.Button(layout_buttons, text="Preview built prompt", command=self._layout_preview_prompt)
            self.layout_preview_prompt_btn.grid(row=0, column=3, sticky="w")

            self.layout_canvas = tk.Canvas(layout_box, height=260, background="#1f242b", highlightthickness=1, highlightbackground="#7b8794", cursor="crosshair")
            self.layout_canvas.grid(row=4, column=0, sticky="nsew", pady=(8, 0), padx=(0, 8))
            self.layout_canvas.bind("<ButtonPress-1>", self._layout_canvas_press)
            self.layout_canvas.bind("<B1-Motion>", self._layout_canvas_drag)
            self.layout_canvas.bind("<ButtonRelease-1>", self._layout_canvas_release)
            self.layout_canvas.bind("<Configure>", lambda event: self._layout_redraw())

            layout_side = ttk.Frame(layout_box)
            layout_side.grid(row=4, column=1, sticky="nsew", pady=(8, 0))
            layout_side.columnconfigure(0, weight=1)
            layout_side.rowconfigure(1, weight=1)
            ttk.Label(layout_side, text="Regions").grid(row=0, column=0, sticky="w")
            self.layout_region_list = tk.Listbox(layout_side, height=6, exportselection=False)
            self.layout_region_list.grid(row=1, column=0, sticky="nsew")
            self.layout_region_list.bind("<<ListboxSelect>>", self._layout_list_selected)
            ttk.Label(layout_side, text="Selected region description").grid(row=2, column=0, sticky="w", pady=(8, 0))
            self.layout_desc_text = tk.Text(layout_side, height=4, wrap="word")
            self.layout_desc_text.grid(row=3, column=0, sticky="ew")
            self.layout_desc_text.bind("<KeyRelease>", lambda event: self._layout_editor_changed())
            self._install_edit_menu(self.layout_desc_text)
            detail_row = ttk.Frame(layout_side)
            detail_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
            detail_row.columnconfigure(3, weight=1)
            ttk.Label(detail_row, text="Type").grid(row=0, column=0, sticky="w")
            self.layout_type_combo = ttk.Combobox(detail_row, textvariable=self.layout_region_type_var, values=["obj", "text"], state="readonly", width=8)
            self.layout_type_combo.grid(row=0, column=1, sticky="w", padx=(4, 10))
            self.layout_type_combo.bind("<<ComboboxSelected>>", lambda event: self._layout_editor_changed())
            ttk.Label(detail_row, text="Text").grid(row=0, column=2, sticky="w")
            self.layout_text_entry = ttk.Entry(detail_row, textvariable=self.layout_region_text_var)
            self.layout_text_entry.grid(row=0, column=3, sticky="ew", padx=(4, 0))
            ttk.Label(layout_side, text="Optional palette (#hex, comma separated)").grid(row=5, column=0, sticky="w", pady=(8, 0))
            self.layout_palette_entry = ttk.Entry(layout_side, textvariable=self.layout_region_palette_var)
            self.layout_palette_entry.grid(row=6, column=0, sticky="ew")
            self.layout_region_text_var.trace_add("write", lambda *_: self._layout_editor_changed())
            self.layout_region_palette_var.trace_add("write", lambda *_: self._layout_editor_changed())
            for _w in (self.layout_background_entry, self.layout_style_photo_entry, self.layout_aesthetics_entry, self.layout_lighting_entry, self.layout_medium_entry):
                _w.bind("<KeyRelease>", lambda event: self._layout_settings_changed(), add="+")
                _w.bind("<FocusOut>", lambda event: self._layout_settings_changed(), add="+")
            self.layout_style_combo.bind("<<ComboboxSelected>>", lambda event: self._layout_settings_changed())
            self.prompt_text.bind("<KeyRelease>", lambda event: self._layout_update_preview_text(), add="+")
            self.width_entry.bind("<KeyRelease>", lambda event: (self._layout_redraw(), self._layout_update_preview_text()), add="+")
            self.height_entry.bind("<KeyRelease>", lambda event: (self._layout_redraw(), self._layout_update_preview_text()), add="+")
            self.layout_json_preview = tk.Text(layout_side, height=5, wrap="word", state="disabled")
            self.layout_json_preview.grid(row=7, column=0, sticky="ew", pady=(8, 0))
            self._install_edit_menu(self.layout_json_preview, readonly=True)

            ToolTip(self.layout_prompt_check, "Turn this on to send the generated region JSON prompt instead of the plain prompt.")
            ToolTip(self.layout_canvas, "Click a box to select it, drag to move it, drag the lower-right corner to resize it.")

            preview_frame = ttk.LabelFrame(right, text="Preview", padding=10)
            preview_frame.grid(row=0, column=0, sticky="nsew")
            preview_frame.columnconfigure(0, weight=1)
            preview_frame.rowconfigure(1, weight=1)

            preview_tools = ttk.Frame(preview_frame)
            preview_tools.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            preview_tools.columnconfigure(4, weight=1)
            self.preview_zoom_out_btn = ttk.Button(preview_tools, text="-", width=3, command=lambda: self._zoom_preview(0.8))
            self.preview_zoom_out_btn.grid(row=0, column=0, sticky="w", padx=(0, 4))
            self.preview_zoom_in_btn = ttk.Button(preview_tools, text="+", width=3, command=lambda: self._zoom_preview(1.25))
            self.preview_zoom_in_btn.grid(row=0, column=1, sticky="w", padx=(0, 8))
            self.preview_fit_btn = ttk.Button(preview_tools, text="Fit", command=self._fit_preview_to_canvas)
            self.preview_fit_btn.grid(row=0, column=2, sticky="w", padx=(0, 4))
            self.preview_reset_btn = ttk.Button(preview_tools, text="100%", command=self._reset_preview_zoom)
            self.preview_reset_btn.grid(row=0, column=3, sticky="w", padx=(0, 8))
            self.preview_zoom_var = tk.StringVar(value="No image")
            ttk.Label(preview_tools, textvariable=self.preview_zoom_var).grid(row=0, column=4, sticky="w")

            self.preview_canvas = tk.Canvas(preview_frame, highlightthickness=0, background="#202020", cursor="fleur")
            self.preview_canvas.grid(row=1, column=0, sticky="nsew")
            self.preview_vscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
            self.preview_vscroll.grid(row=1, column=1, sticky="ns")
            self.preview_hscroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_canvas.xview)
            self.preview_hscroll.grid(row=2, column=0, sticky="ew")
            self.preview_canvas.configure(xscrollcommand=self._preview_xscroll_changed, yscrollcommand=self._preview_yscroll_changed)
            self.preview_canvas.bind("<ButtonPress-1>", self._preview_pan_start)
            self.preview_canvas.bind("<B1-Motion>", self._preview_pan_move)
            self.preview_canvas.bind("<MouseWheel>", self._preview_mousewheel_zoom)
            self.preview_canvas.bind("<Button-4>", self._preview_mousewheel_zoom)
            self.preview_canvas.bind("<Button-5>", self._preview_mousewheel_zoom)
            self.preview_canvas.bind("<Double-Button-1>", lambda event: self._fit_preview_to_canvas())
            self.preview_canvas.bind("<Configure>", self._preview_canvas_configured)
            self._set_preview_message("No image yet")
            ToolTip(self.preview_canvas, "Mouse wheel zooms. Drag with left mouse button to pan. Double-click fits the image.")

            actions = ttk.Frame(self.generate_tab, padding=(10, 8, 10, 10))
            actions.grid(row=1, column=0, columnspan=2, sticky="ew")
            for col in range(4):
                actions.columnconfigure(col, weight=1)
            self.generate_btn = ttk.Button(actions, text="Generate", command=self._start_generate)
            self.generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            self.open_output_btn = ttk.Button(actions, text="View results", command=self._open_output_folder)
            self.open_output_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))
            self.open_logs_btn = ttk.Button(actions, text="Open logs folder", command=self._open_logs_folder)
            self.open_logs_btn.grid(row=0, column=2, sticky="ew", padx=(0, 6))
            self.self_test_btn = ttk.Button(actions, text="Self-test", command=self._self_test)
            self.self_test_btn.grid(row=0, column=3, sticky="ew")
            ttk.Label(actions, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
            ToolTip(self.generate_btn, "Add image generation to the queue with the current settings.")
            ToolTip(self.open_output_btn, "Open the output folder where generated images are saved.")
            ToolTip(self.open_logs_btn, "Open the log folder with the detailed GUI session log.")
            ToolTip(self.self_test_btn, "Check whether CUDA, Torch and the Ideogram runtime can be imported.")

            self._build_queue_tab()
            self._build_settings_tab()

        def _current_generate_split(self) -> str:
            try:
                paned = getattr(self, "generate_paned", None)
                if paned is None:
                    return GUI_DEFAULTS["generate_split"]
                total = max(1, int(paned.winfo_width()))
                pos = int(paned.sashpos(0))
                # Clamp so a saved value never hides either pane after resizing.
                frac = min(0.85, max(0.25, pos / float(total)))
                return f"{frac:.4f}"
            except Exception:
                return GUI_DEFAULTS["generate_split"]

        def _restore_generate_splitter(self):
            try:
                paned = getattr(self, "generate_paned", None)
                if paned is None:
                    return
                raw = load_saved_settings().get("generate_split", GUI_DEFAULTS["generate_split"])
                frac = float(raw)
                frac = min(0.85, max(0.25, frac))
                total = max(1, int(paned.winfo_width()))
                paned.sashpos(0, int(total * frac))
            except Exception:
                pass

        def _install_edit_menu(self, widget, readonly: bool = False):
            """Add a simple right-click edit menu to Tk/ttk text inputs."""
            try:
                def has_selection() -> bool:
                    try:
                        if isinstance(widget, tk.Text):
                            return bool(widget.tag_ranges("sel"))
                        return bool(widget.selection_present())
                    except Exception:
                        return False

                def can_edit() -> bool:
                    if readonly:
                        return False
                    try:
                        state = str(widget.cget("state")).lower()
                        return state not in {"disabled", "readonly"}
                    except Exception:
                        return True

                def do_select_all():
                    try:
                        widget.focus_set()
                        if isinstance(widget, tk.Text):
                            widget.tag_add("sel", "1.0", "end-1c")
                            widget.mark_set("insert", "1.0")
                        else:
                            widget.selection_range(0, "end")
                            widget.icursor("end")
                    except Exception:
                        pass
                    return "break"

                def do_delete():
                    if not can_edit() or not has_selection():
                        return "break"
                    try:
                        if isinstance(widget, tk.Text):
                            widget.delete("sel.first", "sel.last")
                        else:
                            widget.delete("sel.first", "sel.last")
                    except Exception:
                        pass
                    return "break"

                def popup(event):
                    try:
                        widget.focus_set()
                    except Exception:
                        pass
                    menu = tk.Menu(widget, tearoff=0)
                    selected = has_selection()
                    editable = can_edit()
                    menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"), state=("normal" if selected and editable else "disabled"))
                    menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"), state=("normal" if selected else "disabled"))
                    menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"), state=("normal" if editable else "disabled"))
                    menu.add_command(label="Delete", command=do_delete, state=("normal" if selected and editable else "disabled"))
                    menu.add_separator()
                    menu.add_command(label="Select all", command=do_select_all)
                    try:
                        menu.tk_popup(event.x_root, event.y_root)
                    finally:
                        try:
                            menu.grab_release()
                        except Exception:
                            pass
                    return "break"

                widget.bind("<Button-3>", popup, add="+")
                widget.bind("<Control-Button-1>", popup, add="+")
            except Exception:
                pass

        def _canvas_has_vertical_overflow(self, canvas) -> bool:
            try:
                bbox = canvas.bbox("all")
                if not bbox:
                    return False
                content_h = max(0, int(bbox[3]) - int(bbox[1]))
                view_h = max(1, int(canvas.winfo_height()))
                return content_h > view_h + 2
            except Exception:
                return False

        def _canvas_has_horizontal_overflow(self, canvas) -> bool:
            try:
                bbox = canvas.bbox("all")
                if not bbox:
                    return False
                content_w = max(0, int(bbox[2]) - int(bbox[0]))
                view_w = max(1, int(canvas.winfo_width()))
                return content_w > view_w + 2
            except Exception:
                return False

        def _set_scrollbar_visible(self, scrollbar, visible: bool, row: int, column: int, sticky: str):
            try:
                if visible:
                    if not scrollbar.winfo_ismapped():
                        scrollbar.grid(row=row, column=column, sticky=sticky)
                else:
                    if scrollbar.winfo_ismapped():
                        scrollbar.grid_remove()
            except Exception:
                pass

        def _generate_yscroll_changed(self, first, last):
            try:
                self.generate_vscroll.set(first, last)
                show = float(first) > 0.0 or float(last) < 1.0 or self._canvas_has_vertical_overflow(self.generate_canvas)
                self._set_scrollbar_visible(self.generate_vscroll, show, 0, 1, "ns")
            except Exception:
                pass

        def _settings_yscroll_changed(self, first, last):
            try:
                self.settings_vscroll.set(first, last)
                show = float(first) > 0.0 or float(last) < 1.0 or self._canvas_has_vertical_overflow(self.settings_canvas)
                self._set_scrollbar_visible(self.settings_vscroll, show, 0, 1, "ns")
            except Exception:
                pass

        def _preview_yscroll_changed(self, first, last):
            try:
                self.preview_vscroll.set(first, last)
                show = float(first) > 0.0 or float(last) < 1.0 or self._canvas_has_vertical_overflow(self.preview_canvas)
                self._set_scrollbar_visible(self.preview_vscroll, show, 1, 1, "ns")
            except Exception:
                pass

        def _preview_xscroll_changed(self, first, last):
            try:
                self.preview_hscroll.set(first, last)
                show = float(first) > 0.0 or float(last) < 1.0 or self._canvas_has_horizontal_overflow(self.preview_canvas)
                self._set_scrollbar_visible(self.preview_hscroll, show, 2, 0, "ew")
            except Exception:
                pass

        def _refresh_canvas_scrollbars(self):
            try:
                if hasattr(self, "generate_canvas"):
                    self._generate_yscroll_changed(*self.generate_canvas.yview())
                if hasattr(self, "settings_canvas"):
                    self._settings_yscroll_changed(*self.settings_canvas.yview())
                if hasattr(self, "preview_canvas"):
                    self._preview_yscroll_changed(*self.preview_canvas.yview())
                    self._preview_xscroll_changed(*self.preview_canvas.xview())
            except Exception:
                pass

        def _generate_scroll_region_changed(self, event=None):
            try:
                self.generate_canvas.configure(scrollregion=self.generate_canvas.bbox("all"))
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _generate_canvas_configured(self, event=None):
            try:
                self.generate_canvas.itemconfigure(self.generate_canvas_window, width=max(1, int(event.width)))
                self.generate_canvas.configure(scrollregion=self.generate_canvas.bbox("all"))
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _bind_generate_mousewheel(self, event=None):
            try:
                self.root.bind_all("<MouseWheel>", self._generate_mousewheel, add="+")
                self.root.bind_all("<Button-4>", self._generate_mousewheel, add="+")
                self.root.bind_all("<Button-5>", self._generate_mousewheel, add="+")
            except Exception:
                pass

        def _unbind_generate_mousewheel(self, event=None):
            try:
                self.root.unbind_all("<MouseWheel>")
                self.root.unbind_all("<Button-4>")
                self.root.unbind_all("<Button-5>")
            except Exception:
                pass

        def _generate_mousewheel(self, event):
            try:
                if getattr(event, "num", None) == 4:
                    units = -3
                elif getattr(event, "num", None) == 5:
                    units = 3
                else:
                    units = -1 * int(event.delta / 120)
                if self._canvas_has_vertical_overflow(self.generate_canvas):
                    self.generate_canvas.yview_scroll(units, "units")
                    return "break"
                return None
            except Exception:
                return None

        def _build_settings_tab(self):
            self.settings_tab.columnconfigure(0, weight=1)
            self.settings_tab.rowconfigure(0, weight=1)

            self.settings_canvas = tk.Canvas(self.settings_tab, highlightthickness=0)
            self.settings_canvas.grid(row=0, column=0, sticky="nsew")
            self.settings_vscroll = ttk.Scrollbar(self.settings_tab, orient="vertical", command=self.settings_canvas.yview)
            self.settings_vscroll.grid(row=0, column=1, sticky="ns")
            self.settings_canvas.configure(yscrollcommand=self._settings_yscroll_changed)

            settings_outer = ttk.Frame(self.settings_canvas, padding=10)
            self.settings_canvas_window = self.settings_canvas.create_window((0, 0), window=settings_outer, anchor="nw")
            settings_outer.bind("<Configure>", self._settings_scroll_region_changed)
            self.settings_canvas.bind("<Configure>", self._settings_canvas_configured)
            self.settings_canvas.bind("<Enter>", self._bind_settings_mousewheel)
            self.settings_canvas.bind("<Leave>", self._unbind_settings_mousewheel)
            settings_outer.columnconfigure(0, weight=1)

            appearance = ttk.LabelFrame(settings_outer, text="Appearance", padding=10)
            appearance.grid(row=0, column=0, sticky="ew")
            appearance.columnconfigure(1, weight=1)
            self.theme_label = ttk.Label(appearance, text="Theme")
            self.theme_label.grid(row=0, column=0, sticky="w")
            self.theme_combo = ttk.Combobox(appearance, textvariable=self.theme_var, values=THEME_CHOICES, state="readonly")
            self.theme_combo.grid(row=0, column=1, sticky="ew", padx=6)
            self.theme_combo.bind("<<ComboboxSelected>>", self._theme_changed)
            ToolTip(self.theme_label, "Choose the helper window theme.")
            ToolTip(self.theme_combo, "Themes: Dark, Light, Purple Delight, Graphite Midnight and Violet Dusk.")

            memory = ttk.LabelFrame(settings_outer, text="GGUF advanced", padding=10)
            self.gguf_memory_frame = memory
            memory.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            memory.columnconfigure(0, weight=1)
            self.gguf_stream_check = ttk.Checkbutton(memory, text="Stream layers", variable=self.gguf_stream_layers_var, command=self._save_settings)
            self.gguf_stream_check.grid(row=0, column=0, sticky="w")
            ToolTip(self.gguf_stream_check, "Experimental sd-cli option. Leave off unless you are testing it for a specific GGUF model.")

            paths = ttk.LabelFrame(settings_outer, text="Paths", padding=10)
            paths.grid(row=2, column=0, sticky="ew", pady=(10, 0))
            paths.columnconfigure(1, weight=1)

            self.model_dir_label = ttk.Label(paths, text="SDNQ model folder")
            self.model_dir_label.grid(row=0, column=0, sticky="w")
            self.model_dir_entry = ttk.Entry(paths, textvariable=self.model_dir_var)
            self.model_dir_entry.grid(row=0, column=1, sticky="ew", padx=6)
            self._install_edit_menu(self.model_dir_entry)
            self.model_dir_browse = ttk.Button(paths, text="Browse", command=self._browse_model)
            self.model_dir_browse.grid(row=0, column=2)

            self.gguf_dir_label = ttk.Label(paths, text="GGUF folder")
            self.gguf_dir_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
            self.gguf_dir_entry = ttk.Entry(paths, textvariable=self.gguf_dir_var)
            self.gguf_dir_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
            self._install_edit_menu(self.gguf_dir_entry)
            self.gguf_dir_entry.bind("<Return>", lambda event: (self._refresh_gguf_file_dropdowns(save_fallback=True), self._save_settings()))
            self.gguf_dir_entry.bind("<FocusOut>", lambda event: (self._refresh_gguf_file_dropdowns(save_fallback=True), self._save_settings()))
            self.gguf_dir_browse = ttk.Button(paths, text="Browse", command=self._browse_gguf_model)
            self.gguf_dir_browse.grid(row=1, column=2, pady=(8, 0))

            self.gguf_diffusion_label = ttk.Label(paths, text="Main GGUF")
            self.gguf_diffusion_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
            self.gguf_diffusion_combo = ttk.Combobox(paths, textvariable=self.gguf_diffusion_var, state="readonly")
            self.gguf_diffusion_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=(8, 0))
            self.gguf_diffusion_combo.bind("<<ComboboxSelected>>", lambda event: self._save_settings())

            self.gguf_unconditional_label = ttk.Label(paths, text="Uncond GGUF")
            self.gguf_unconditional_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
            self.gguf_unconditional_combo = ttk.Combobox(paths, textvariable=self.gguf_unconditional_var, state="readonly")
            self.gguf_unconditional_combo.grid(row=3, column=1, sticky="ew", padx=6, pady=(8, 0))
            self.gguf_unconditional_combo.bind("<<ComboboxSelected>>", lambda event: self._save_settings())

            self.gguf_llm_label = ttk.Label(paths, text="Qwen / LLM GGUF")
            self.gguf_llm_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
            self.gguf_llm_combo = ttk.Combobox(paths, textvariable=self.gguf_llm_var, state="readonly")
            self.gguf_llm_combo.grid(row=4, column=1, sticky="ew", padx=6, pady=(8, 0))
            self.gguf_llm_combo.bind("<<ComboboxSelected>>", lambda event: self._save_settings())

            self.gguf_vae_label = ttk.Label(paths, text="VAE")
            self.gguf_vae_label.grid(row=5, column=0, sticky="w", pady=(8, 0))
            self.gguf_vae_combo = ttk.Combobox(paths, textvariable=self.gguf_vae_var, state="readonly")
            self.gguf_vae_combo.grid(row=5, column=1, sticky="ew", padx=6, pady=(8, 0))
            self.gguf_vae_combo.bind("<<ComboboxSelected>>", lambda event: self._save_settings())

            self.sd_cli_label = ttk.Label(paths, text="sd-cli.exe")
            self.sd_cli_label.grid(row=6, column=0, sticky="w", pady=(8, 0))
            self.sd_cli_entry = ttk.Entry(paths, textvariable=self.sd_cli_path_var)
            self.sd_cli_entry.grid(row=6, column=1, sticky="ew", padx=6, pady=(8, 0))
            self._install_edit_menu(self.sd_cli_entry)
            self.sd_cli_browse = ttk.Button(paths, text="Browse", command=self._browse_sd_cli)
            self.sd_cli_browse.grid(row=6, column=2, pady=(8, 0))

            self.output_dir_label = ttk.Label(paths, text="Output folder")
            self.output_dir_label.grid(row=7, column=0, sticky="w", pady=(8, 0))
            self.output_dir_entry = ttk.Entry(paths, textvariable=self.output_dir_var)
            self.output_dir_entry.grid(row=7, column=1, sticky="ew", padx=6, pady=(8, 0))
            self._install_edit_menu(self.output_dir_entry)
            self.output_dir_browse = ttk.Button(paths, text="Browse", command=self._browse_output)
            self.output_dir_browse.grid(row=7, column=2, pady=(8, 0))

            ToolTip(self.model_dir_label, f"Folder that contains the Ideogram SDNQ model. Default: {default_model_dir()}")
            ToolTip(self.model_dir_entry, f"Folder that contains the Ideogram SDNQ model. Default: {default_model_dir()}")
            ToolTip(self.model_dir_browse, "Browse to the SDNQ model folder.")
            ToolTip(self.gguf_dir_label, f"Folder that contains the Ideogram GGUF files. Default: {default_gguf_dir()}")
            ToolTip(self.gguf_dir_entry, f"Choose a folder, then select the GGUF files from the dropdowns below. Default: {default_gguf_dir()}")
            ToolTip(self.gguf_dir_browse, "Browse to the GGUF model folder and refresh the dropdown file list.")
            ToolTip(self.gguf_diffusion_label, "Main Ideogram diffusion GGUF. Default: ideogram4-Q4_0.gguf.")
            ToolTip(self.gguf_diffusion_combo, "Main Ideogram diffusion GGUF found in the selected folder.")
            ToolTip(self.gguf_unconditional_label, "Unconditional Ideogram GGUF. Default: ideogram4_unconditional-Q4_0.gguf.")
            ToolTip(self.gguf_unconditional_combo, "Unconditional Ideogram GGUF found in the selected folder.")
            ToolTip(self.gguf_llm_label, "Qwen / LLM text encoder GGUF. Default: qwen3-vl-8b-instruct-q4_k_m.gguf.")
            ToolTip(self.gguf_llm_combo, "Qwen / LLM GGUF found in the selected folder.")
            ToolTip(self.gguf_vae_label, "Flux2 VAE safetensors file. Default: flux2-vae.safetensors.")
            ToolTip(self.gguf_vae_combo, "VAE safetensors file found in the selected folder.")
            ToolTip(self.sd_cli_label, f"Path to stable-diffusion.cpp sd-cli.exe. Auto-fills from {default_sd_cli_path()} when found.")
            ToolTip(self.sd_cli_entry, f"Path to sd-cli.exe. Default: {default_sd_cli_path()}")
            ToolTip(self.sd_cli_browse, "Browse to a different sd-cli.exe if needed.")
            ToolTip(self.output_dir_label, f"Folder where images are saved. Default: {default_output_dir()}")
            ToolTip(self.output_dir_entry, f"Folder where images are saved. Default: {default_output_dir()}")
            ToolTip(self.output_dir_browse, "Browse to the output folder.")

            self.logs_expanded_var = tk.BooleanVar(value=False)
            logs_shell = ttk.LabelFrame(settings_outer, text="Logs", padding=10)
            logs_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
            logs_shell.columnconfigure(0, weight=1)
            self.logs_toggle = ttk.Checkbutton(logs_shell, text="Show detailed logs", variable=self.logs_expanded_var, command=self._toggle_settings_logs)
            self.logs_toggle.grid(row=0, column=0, sticky="w")
            ToolTip(self.logs_toggle, "Show or hide the live GUI log. The full log is still saved in the logs folder.")
            self.logs_body = ttk.Frame(logs_shell)
            self.logs_body.columnconfigure(0, weight=1)
            self.logs_body.rowconfigure(0, weight=1)
            self.log_text = tk.Text(self.logs_body, height=14, wrap="word", state="disabled")
            self.log_text.grid(row=0, column=0, sticky="nsew", pady=(8, 0))
            self._install_edit_menu(self.log_text, readonly=True)
            self._toggle_settings_logs()

        def _toggle_settings_logs(self):
            try:
                if self.logs_expanded_var.get():
                    self.logs_body.grid(row=1, column=0, sticky="nsew")
                else:
                    self.logs_body.grid_remove()
                self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _settings_scroll_region_changed(self, event=None):
            try:
                self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _settings_canvas_configured(self, event=None):
            try:
                self.settings_canvas.itemconfigure(self.settings_canvas_window, width=max(1, int(event.width)))
                self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _bind_settings_mousewheel(self, event=None):
            try:
                self.root.bind_all("<MouseWheel>", self._settings_mousewheel, add="+")
                self.root.bind_all("<Button-4>", self._settings_mousewheel, add="+")
                self.root.bind_all("<Button-5>", self._settings_mousewheel, add="+")
            except Exception:
                pass

        def _unbind_settings_mousewheel(self, event=None):
            try:
                self.root.unbind_all("<MouseWheel>")
                self.root.unbind_all("<Button-4>")
                self.root.unbind_all("<Button-5>")
            except Exception:
                pass

        def _settings_mousewheel(self, event):
            try:
                if getattr(event, "num", None) == 4:
                    units = -3
                elif getattr(event, "num", None) == 5:
                    units = 3
                else:
                    units = -1 * int(event.delta / 120)
                if self._canvas_has_vertical_overflow(self.settings_canvas):
                    self.settings_canvas.yview_scroll(units, "units")
                    return "break"
                return None
            except Exception:
                return None

        def _build_queue_tab(self):
            self.queue_tab.columnconfigure(0, weight=1)
            self.queue_tab.rowconfigure(0, weight=1)
            self.queue_tab.rowconfigure(1, weight=0)

            queue_paned = ttk.Panedwindow(self.queue_tab, orient="vertical")
            queue_paned.grid(row=0, column=0, sticky="nsew")

            top = ttk.Frame(queue_paned, padding=(10, 10, 10, 0))
            top.columnconfigure(0, weight=1)
            top.rowconfigure(1, weight=1)

            bottom = ttk.Frame(queue_paned, padding=(10, 8, 10, 0))
            bottom.columnconfigure(0, weight=1)
            bottom.columnconfigure(1, weight=1)
            bottom.rowconfigure(0, weight=1)

            try:
                queue_paned.add(top, weight=1)
                queue_paned.add(bottom, weight=1)
            except Exception:
                queue_paned.add(top)
                queue_paned.add(bottom)

            header = ttk.Frame(top)
            header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text=f"Queue JSON: {queue_state_path()}").grid(row=0, column=0, sticky="w")

            columns = ("status", "progress", "eta", "started", "duration", "finished", "resolution", "prompt")
            self.queue_tree = ttk.Treeview(top, columns=columns, show="headings", selectmode="browse")
            self.queue_tree.heading("status", text="Status")
            self.queue_tree.heading("progress", text="Steps")
            self.queue_tree.heading("eta", text="ETA")
            self.queue_tree.heading("started", text="Started")
            self.queue_tree.heading("duration", text="Duration")
            self.queue_tree.heading("finished", text="Finished")
            self.queue_tree.heading("resolution", text="Resolution")
            self.queue_tree.heading("prompt", text="Prompt")
            self.queue_tree.column("status", width=95, anchor="w", stretch=False)
            self.queue_tree.column("progress", width=92, anchor="w", stretch=False)
            self.queue_tree.column("eta", width=88, anchor="w", stretch=False)
            self.queue_tree.column("started", width=90, anchor="w", stretch=False)
            self.queue_tree.column("duration", width=90, anchor="w", stretch=False)
            self.queue_tree.column("finished", width=90, anchor="w", stretch=False)
            self.queue_tree.column("resolution", width=115, anchor="w", stretch=False)
            self.queue_tree.column("prompt", width=540, anchor="w", stretch=True)
            self.queue_tree.grid(row=1, column=0, sticky="nsew")
            self.queue_tree.bind("<Button-3>", self._queue_right_click)
            self.queue_tree.bind("<Double-1>", lambda event: self._show_selected_queue_job())
            self.queue_tree.bind("<<TreeviewSelect>>", lambda event: self._update_queue_preview_panel())

            scroll = ttk.Scrollbar(top, orient="vertical", command=self.queue_tree.yview)
            scroll.grid(row=1, column=1, sticky="ns")
            self.queue_tree.configure(yscrollcommand=scroll.set)

            queue_preview_frame = ttk.LabelFrame(bottom, text="Selected result preview", padding=10)
            queue_preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            queue_preview_frame.columnconfigure(0, weight=1)
            queue_preview_frame.rowconfigure(0, weight=1)
            self.queue_preview_canvas = tk.Canvas(queue_preview_frame, highlightthickness=0, background="#202020")
            self.queue_preview_canvas.grid(row=0, column=0, sticky="nsew")
            self.queue_preview_ref = None
            self.queue_preview_message_id = None
            self.queue_preview_canvas.bind("<Configure>", lambda event: self._update_queue_preview_panel())
            ToolTip(self.queue_preview_canvas, "Shows the output image for the selected queue item when available.")

            info_frame = ttk.LabelFrame(bottom, text="Selected job info", padding=10)
            info_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            info_frame.columnconfigure(0, weight=1)
            info_frame.rowconfigure(0, weight=1)
            self.queue_info_text = tk.Text(info_frame, height=10, wrap="word", state="disabled")
            self.queue_info_text.grid(row=0, column=0, sticky="nsew")
            self.queue_info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.queue_info_text.yview)
            self.queue_info_scroll.grid(row=0, column=1, sticky="ns")
            self.queue_info_text.configure(yscrollcommand=self.queue_info_scroll.set)
            self._install_edit_menu(self.queue_info_text)
            ToolTip(self.queue_info_text, "Shows details for the selected queue item.")

            footer = ttk.Frame(self.queue_tab, padding=(10, 8, 10, 10))
            footer.grid(row=1, column=0, sticky="ew")
            footer.columnconfigure(0, weight=1)
            self.queue_summary_var = tk.StringVar(value="No jobs yet")
            ttk.Label(footer, textvariable=self.queue_summary_var).grid(row=0, column=0, sticky="w")
            self.clear_finished_btn = ttk.Button(footer, text="Clear finished / failed results", command=self._clear_finished_jobs)
            self.clear_finished_btn.grid(row=0, column=1, sticky="e", padx=(8, 6))
            self.queue_refresh_btn = ttk.Button(footer, text="Refresh", command=self._refresh_queue_tree)
            self.queue_refresh_btn.grid(row=0, column=2, sticky="e")

        def _theme_key(self) -> str:
            return normalize_theme_key(self.theme_var.get())

        def _theme_palette(self) -> dict[str, str]:
            key = self._theme_key()
            palettes = {
                "light": {
                    "bg": "#f3f4f6",
                    "panel": "#ffffff",
                    "panel2": "#f9fafb",
                    "fg": "#111827",
                    "muted": "#4b5563",
                    "entry": "#ffffff",
                    "select": "#2563eb",
                    "button": "#e5e7eb",
                    "border": "#d1d5db",
                    "canvas": "#e5e7eb",
                },
                "purple_nebula": {
                    "bg": "#12071f",
                    "panel": "#1d1033",
                    "panel2": "#321657",
                    "fg": "#f3e8ff",
                    "muted": "#c4b5fd",
                    "entry": "#0f0820",
                    "select": "#8b5cf6",
                    "button": "#4c1d95",
                    "border": "#6d28d9",
                    "canvas": "#05010d",
                },
                "graphite_midnight": {
                    "bg": "#0b0f14",
                    "panel": "#111827",
                    "panel2": "#1f2937",
                    "fg": "#e5e7eb",
                    "muted": "#9ca3af",
                    "entry": "#05080d",
                    "select": "#38bdf8",
                    "button": "#263241",
                    "border": "#334155",
                    "canvas": "#020617",
                },
                "dracula": {
                    "bg": "#282a36",
                    "panel": "#343746",
                    "panel2": "#44475a",
                    "fg": "#f8f8f2",
                    "muted": "#bd93f9",
                    "entry": "#1e1f29",
                    "select": "#bd93f9",
                    "button": "#44475a",
                    "border": "#6272a4",
                    "canvas": "#191a21",
                },
                "dark": {
                    "bg": "#0f172a",
                    "panel": "#111827",
                    "panel2": "#1f2937",
                    "fg": "#e5e7eb",
                    "muted": "#9ca3af",
                    "entry": "#0b1220",
                    "select": "#2563eb",
                    "button": "#374151",
                    "border": "#334155",
                    "canvas": "#020617",
                },
            }
            return palettes.get(key, palettes["dark"])

        def _theme_changed(self, event=None):
            self._apply_theme()
            self._save_settings()

        def _toggle_theme(self):
            self._theme_changed()

        def _apply_theme(self):
            if self._theme_key() == "off":
                try:
                    self.theme_var.set(THEME_LABELS["off"])
                except Exception:
                    pass
                return
            pal = self._theme_palette()
            try:
                self.root.configure(bg=pal["bg"])
            except Exception:
                pass
            try:
                style = ttk.Style(self.root)
                try:
                    style.theme_use("clam")
                except Exception:
                    pass
                style.configure(".", background=pal["bg"], foreground=pal["fg"], fieldbackground=pal["entry"], bordercolor=pal["border"], lightcolor=pal["border"], darkcolor=pal["border"])
                style.configure("TFrame", background=pal["bg"])
                style.configure("TLabel", background=pal["bg"], foreground=pal["fg"])
                style.configure("TLabelFrame", background=pal["bg"], foreground=pal["fg"], bordercolor=pal["border"])
                style.configure("TLabelFrame.Label", background=pal["bg"], foreground=pal["fg"])
                style.configure("TButton", background=pal["button"], foreground=pal["fg"], bordercolor=pal["border"], focusthickness=1, focuscolor=pal["select"])
                style.map("TButton", background=[("active", pal["panel2"]), ("pressed", pal["select"])] )
                style.configure("TCheckbutton", background=pal["bg"], foreground=pal["fg"])
                style.map("TCheckbutton", background=[("active", pal["bg"])] , foreground=[("active", pal["fg"])] )
                style.configure("TEntry", fieldbackground=pal["entry"], foreground=pal["fg"], insertcolor=pal["fg"], bordercolor=pal["border"])
                style.configure("TCombobox", fieldbackground=pal["entry"], background=pal["entry"], foreground=pal["fg"], arrowcolor=pal["fg"], bordercolor=pal["border"])
                style.map("TCombobox", fieldbackground=[("readonly", pal["entry"])], foreground=[("readonly", pal["fg"])])
                style.configure("TNotebook", background=pal["bg"], bordercolor=pal["border"])
                style.configure("TNotebook.Tab", background=pal["panel2"], foreground=pal["fg"], padding=(12, 6))
                style.map("TNotebook.Tab", background=[("selected", pal["panel"])], foreground=[("selected", pal["fg"])])
                style.configure("Treeview", background=pal["panel"], fieldbackground=pal["panel"], foreground=pal["fg"], bordercolor=pal["border"], rowheight=26)
                style.configure("Treeview.Heading", background=pal["panel2"], foreground=pal["fg"], bordercolor=pal["border"])
                style.map("Treeview", background=[("selected", pal["select"])], foreground=[("selected", "#ffffff")])
                style.configure("Vertical.TScrollbar", background=pal["button"], troughcolor=pal["bg"], bordercolor=pal["border"], arrowcolor=pal["fg"])
                style.configure("Horizontal.TScrollbar", background=pal["button"], troughcolor=pal["bg"], bordercolor=pal["border"], arrowcolor=pal["fg"])
            except Exception:
                pass

            text_widgets = [getattr(self, "prompt_text", None), getattr(self, "log_text", None), getattr(self, "queue_info_text", None)]
            for widget in text_widgets:
                if widget is None:
                    continue
                try:
                    widget.configure(background=pal["entry"], foreground=pal["fg"], insertbackground=pal["fg"], selectbackground=pal["select"], selectforeground="#ffffff", highlightbackground=pal["border"], highlightcolor=pal["select"])
                except Exception:
                    pass
            try:
                self.preview_canvas.configure(background=pal["canvas"], highlightbackground=pal["border"])
            except Exception:
                pass
            try:
                self.queue_preview_canvas.configure(background=pal["canvas"], highlightbackground=pal["border"])
            except Exception:
                pass
            try:
                self.generate_canvas.configure(background=pal["bg"], highlightbackground=pal["border"])
            except Exception:
                pass
            try:
                self.settings_canvas.configure(background=pal["bg"], highlightbackground=pal["border"])
            except Exception:
                pass
            try:
                key = self._theme_key()
                self.theme_var.set(THEME_LABELS.get(key, THEME_LABELS["dark"]))
            except Exception:
                pass
            try:
                self._redraw_preview()
            except Exception:
                pass

        def _format_bytes_gb(self, used: float, total: float) -> str:
            gb = 1024 ** 3
            return f"{used / gb:.1f}/{total / gb:.1f} GB"

        def _read_ram_cpu(self) -> tuple[str, str]:
            try:
                import psutil  # type: ignore
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)
                return self._format_bytes_gb(float(mem.used), float(mem.total)), f"{cpu:.0f}%"
            except Exception:
                pass
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    used = float(stat.ullTotalPhys - stat.ullAvailPhys)
                    return self._format_bytes_gb(used, float(stat.ullTotalPhys)), "--%"
            except Exception:
                pass
            return "--/--", "--%"

        def _read_vram(self) -> str:
            try:
                cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
                startupinfo = None
                if os.name == "nt":
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2, startupinfo=startupinfo)
                if result.returncode == 0 and result.stdout.strip():
                    first = result.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in first.split(",")]
                    if len(parts) >= 2:
                        used_mb = float(parts[0])
                        total_mb = float(parts[1])
                        return f"{used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB"
            except Exception:
                pass
            return "--/--"

        def _update_hud(self):
            ram, cpu = self._read_ram_cpu()
            vram = self._read_vram()
            self.hud_var.set(f"DDR: {ram}   VRAM: {vram}   CPU: {cpu}")
            self.root.after(2000, self._update_hud)

        def _clear_finished_jobs(self):
            with self.jobs_lock:
                before = len(self.jobs)
                self.jobs[:] = [j for j in self.jobs if str(j.get("status", "")) not in {"finished", "failed"}]
                removed = before - len(self.jobs)
                self._save_queue_state()
            self._write_log(f"Cleared {removed} finished/failed queue result(s).")
            self._refresh_queue_tree()
            self._update_queue_preview_panel()

        def _write_log(self, text: str):
            stamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {text.rstrip()}"
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            try:
                self.session_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.session_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                latest = logs_dir() / "ideogram4_sdnq_gui_latest.log"
                latest.write_text(self.session_log_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

        def _thread_log(self, text: str):
            raw_text = str(text)
            prefix = ""
            if self.active_job_id:
                prefix = f"Job {self.active_job_id}: "
            self.msg_queue.put(("log", {"text": prefix + raw_text}))
            if self.active_job_id:
                current = total = None
                step_seconds = None

                # SDNQ logs are normalized as "Step 5/20 | 25.28s/it".
                # GGUF / sd-cli logs usually arrive as "[sd-cli] | 5/20 - 25.28s/it...".
                # Parse both so the Queue tab can update while the job is running.
                m = _QUEUE_STEP_LOG_RE.match(raw_text.strip())
                if m:
                    current = int(m.group(1))
                    total = int(m.group(2))
                    step_seconds = float(m.group(3)) if m.group(3) else None
                else:
                    matches = list(_QUEUE_ANY_STEP_RE.finditer(raw_text))
                    for match in reversed(matches):
                        try:
                            cur = int(match.group(1))
                            tot = int(match.group(2))
                        except Exception:
                            continue
                        if tot > 0 and 0 <= cur <= tot:
                            current, total = cur, tot
                            break
                    speed_match = _QUEUE_STEP_SPEED_RE.search(raw_text)
                    if speed_match:
                        try:
                            step_seconds = float(speed_match.group(1))
                        except Exception:
                            step_seconds = None

                if current is not None and total is not None:
                    self.msg_queue.put(("queue_progress", {
                        "job_id": str(self.active_job_id),
                        "current": int(current),
                        "total": int(total),
                        "step_seconds": step_seconds,
                    }))

        def _layout_canvas_rect(self):
            try:
                cw = max(1, int(self.layout_canvas.winfo_width()))
                ch = max(1, int(self.layout_canvas.winfo_height()))
                margin = 10
                aw = max(1, cw - margin * 2)
                ah = max(1, ch - margin * 2)
                try:
                    w = max(1, int(float(self.width_var.get().strip())))
                    h = max(1, int(float(self.height_var.get().strip())))
                except Exception:
                    w, h = 1024, 1024
                aspect = w / float(max(1, h))
                if aw / float(max(1, ah)) > aspect:
                    draw_h = ah
                    draw_w = int(draw_h * aspect)
                else:
                    draw_w = aw
                    draw_h = int(draw_w / max(aspect, 1e-6))
                x0 = margin + (aw - draw_w) // 2
                y0 = margin + (ah - draw_h) // 2
                return x0, y0, max(1, draw_w), max(1, draw_h)
            except Exception:
                return 10, 10, 300, 200

        def _layout_region_to_canvas(self, reg: dict[str, Any]):
            x0, y0, cw, ch = self._layout_canvas_rect()
            x = x0 + float(reg.get('x', 0.0)) * cw
            y = y0 + float(reg.get('y', 0.0)) * ch
            w = float(reg.get('w', 0.2)) * cw
            h = float(reg.get('h', 0.2)) * ch
            return x, y, x + w, y + h

        def _layout_collect_regions(self) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for reg in getattr(self, 'layout_regions', []) or []:
                try:
                    out.append({
                        'x': float(reg.get('x', 0.0)),
                        'y': float(reg.get('y', 0.0)),
                        'w': float(reg.get('w', 0.2)),
                        'h': float(reg.get('h', 0.2)),
                        'type': str(reg.get('type') or 'obj'),
                        'text': str(reg.get('text') or ''),
                        'desc': str(reg.get('desc') or ''),
                        'palette': list(reg.get('palette') or []),
                    })
                except Exception:
                    pass
            return out

        def _layout_redraw(self):
            if not hasattr(self, 'layout_canvas'):
                return
            c = self.layout_canvas
            c.delete('all')
            x0, y0, cw, ch = self._layout_canvas_rect()
            c.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill='#111820', outline='#7b8794')
            c.create_line(x0 + cw / 2, y0, x0 + cw / 2, y0 + ch, fill='#394555', dash=(3, 3))
            c.create_line(x0, y0 + ch / 2, x0 + cw, y0 + ch / 2, fill='#394555', dash=(3, 3))
            try:
                w = int(float(self.width_var.get().strip())); h = int(float(self.height_var.get().strip()))
                c.create_text(x0 + 8, y0 + 8, anchor='nw', fill='#b9c5d6', text=f"{w} x {h}")
            except Exception:
                pass
            for i, reg in enumerate(getattr(self, 'layout_regions', []) or []):
                x1, y1, x2, y2 = self._layout_region_to_canvas(reg)
                selected = (i == int(getattr(self, 'layout_selected_index', -1)))
                outline = '#35bdf6' if selected else '#c0c0c0'
                fill = '#265d7a' if selected else '#3c424b'
                stipple = 'gray25' if selected else 'gray50'
                c.create_rectangle(x1, y1, x2, y2, outline=outline, width=2 if selected else 1, fill=fill, stipple=stipple)
                c.create_rectangle(x1, y1, x1 + 28, y1 + 20, fill='#eeeeee', outline='')
                c.create_text(x1 + 14, y1 + 10, text=f"{i+1:02d}", fill='#222222')
                desc = str(reg.get('desc') or reg.get('text') or '').strip()
                if desc:
                    txt = desc[:72] + ('…' if len(desc) > 72 else '')
                    c.create_text(x1 + 6, y1 + 26, anchor='nw', fill='#f0f0f0', text=txt, width=max(20, int(x2 - x1 - 12)))
                if selected:
                    c.create_rectangle(x2 - 9, y2 - 9, x2 + 2, y2 + 2, fill='#35bdf6', outline='')

        def _layout_hit_test(self, x: float, y: float):
            for i in range(len(getattr(self, 'layout_regions', []) or []) - 1, -1, -1):
                x1, y1, x2, y2 = self._layout_region_to_canvas(self.layout_regions[i])
                if x2 - 12 <= x <= x2 + 4 and y2 - 12 <= y <= y2 + 4:
                    return i, 'resize'
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return i, 'move'
            return -1, ''

        def _layout_select(self, idx: int):
            try: idx = int(idx)
            except Exception: idx = -1
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                idx = -1
            self.layout_selected_index = idx
            if hasattr(self, 'layout_region_list'):
                try:
                    self.layout_region_list.selection_clear(0, 'end')
                    if idx >= 0:
                        self.layout_region_list.selection_set(idx)
                        self.layout_region_list.see(idx)
                except Exception:
                    pass
            self._layout_load_selected_editor()
            self._layout_redraw()

        def _layout_load_selected_editor(self):
            idx = int(getattr(self, 'layout_selected_index', -1))
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                try:
                    self._layout_loading_editor = True
                    self.layout_desc_text.delete('1.0', 'end')
                    self.layout_region_text_var.set('')
                    self.layout_region_type_var.set('obj')
                    self.layout_region_palette_var.set('')
                except Exception:
                    pass
                finally:
                    self._layout_loading_editor = False
                return
            reg = self.layout_regions[idx]
            try:
                self._layout_loading_editor = True
                self.layout_desc_text.delete('1.0', 'end')
                self.layout_desc_text.insert('1.0', str(reg.get('desc') or ''))
                self.layout_region_text_var.set(str(reg.get('text') or ''))
                self.layout_region_type_var.set(str(reg.get('type') or 'obj'))
                self.layout_region_palette_var.set(', '.join(list(reg.get('palette') or [])))
            except Exception:
                pass
            finally:
                self._layout_loading_editor = False

        def _layout_refresh_list(self):
            if not hasattr(self, 'layout_region_list'):
                return
            try:
                self.layout_region_list.delete(0, 'end')
                for i, reg in enumerate(getattr(self, 'layout_regions', []) or []):
                    desc = str(reg.get('desc') or reg.get('text') or f'region {i+1}').strip()
                    self.layout_region_list.insert('end', f"{i+1:02d}  {desc[:80]}")
            except Exception:
                pass
            self._layout_select(getattr(self, 'layout_selected_index', -1))
            self._layout_update_preview_text()

        def _layout_add_region(self):
            count = len(getattr(self, 'layout_regions', []) or [])
            offset = min(0.06 * count, 0.42)
            self.layout_regions.append({
                'x': min(0.72, 0.06 + offset),
                'y': min(0.72, 0.06 + offset),
                'w': 0.28,
                'h': 0.22,
                'type': 'obj',
                'text': '',
                'desc': f'region {count + 1}',
                'palette': [],
            })
            self.layout_selected_index = len(self.layout_regions) - 1
            self._layout_refresh_list()
            self._save_settings()

        def _layout_remove_region(self):
            idx = int(getattr(self, 'layout_selected_index', -1))
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                return
            self.layout_regions.pop(idx)
            self.layout_selected_index = min(idx, len(self.layout_regions) - 1)
            self._layout_refresh_list()
            self._save_settings()

        def _layout_clear_regions(self):
            self.layout_regions = []
            self.layout_selected_index = -1
            self._layout_refresh_list()
            self._save_settings()

        def _layout_list_selected(self, _event=None):
            if bool(getattr(self, '_layout_list_updating', False)):
                return
            try:
                sel = self.layout_region_list.curselection()
                self._layout_select(int(sel[0]) if sel else -1)
            except Exception:
                self._layout_select(-1)

        def _layout_canvas_press(self, event):
            idx, mode = self._layout_hit_test(event.x, event.y)
            self._layout_select(idx)
            if idx >= 0:
                self.layout_drag_mode = mode
                self.layout_drag_start = (float(event.x), float(event.y))
                self.layout_drag_region = dict(self.layout_regions[idx])

        def _layout_canvas_drag(self, event):
            idx = int(getattr(self, 'layout_selected_index', -1))
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                return
            if not self.layout_drag_mode or self.layout_drag_start is None or self.layout_drag_region is None:
                return
            _x0, _y0, cw, ch = self._layout_canvas_rect()
            dx = (float(event.x) - self.layout_drag_start[0]) / max(1.0, float(cw))
            dy = (float(event.y) - self.layout_drag_start[1]) / max(1.0, float(ch))
            reg = dict(self.layout_drag_region)
            if self.layout_drag_mode == 'move':
                rw = float(reg.get('w', 0.2)); rh = float(reg.get('h', 0.2))
                reg['x'] = min(max(0.0, float(reg.get('x', 0.0)) + dx), max(0.0, 1.0 - rw))
                reg['y'] = min(max(0.0, float(reg.get('y', 0.0)) + dy), max(0.0, 1.0 - rh))
            elif self.layout_drag_mode == 'resize':
                reg['w'] = min(max(0.04, float(reg.get('w', 0.2)) + dx), max(0.04, 1.0 - float(reg.get('x', 0.0))))
                reg['h'] = min(max(0.04, float(reg.get('h', 0.2)) + dy), max(0.04, 1.0 - float(reg.get('y', 0.0))))
            self.layout_regions[idx].update(reg)
            self._layout_redraw()
            self._layout_update_preview_text()

        def _layout_canvas_release(self, _event=None):
            self.layout_drag_mode = ''
            self.layout_drag_start = None
            self.layout_drag_region = None
            self._layout_refresh_list()
            self._save_settings()

        def _layout_editor_changed(self):
            if bool(getattr(self, '_layout_loading_editor', False)):
                return
            idx = int(getattr(self, 'layout_selected_index', -1))
            if idx < 0 or idx >= len(getattr(self, 'layout_regions', []) or []):
                return
            try:
                self.layout_regions[idx]['desc'] = self.layout_desc_text.get('1.0', 'end').strip()
                self.layout_regions[idx]['type'] = self.layout_region_type_var.get().strip() or 'obj'
                self.layout_regions[idx]['text'] = self.layout_region_text_var.get().strip()
                self.layout_regions[idx]['palette'] = [p.strip() for p in self.layout_region_palette_var.get().split(',') if p.strip()]
            except Exception:
                pass
            self._layout_redraw()
            try:
                self._layout_list_updating = True
                self.layout_region_list.delete(idx)
                desc = str(self.layout_regions[idx].get('desc') or self.layout_regions[idx].get('text') or f'region {idx+1}').strip()
                self.layout_region_list.insert(idx, f"{idx+1:02d}  {desc[:80]}")
                self.layout_region_list.selection_set(idx)
            except Exception:
                pass
            finally:
                self._layout_list_updating = False
            self._layout_update_preview_text()
            self._save_settings()

        def _layout_build_prompt_json(self) -> str:
            return build_structured_caption(
                self.prompt_text.get('1.0', 'end').strip(),
                int(float(self.width_var.get().strip() or 1024)),
                int(float(self.height_var.get().strip() or 1024)),
                regions=self._layout_collect_regions(),
                background=self.layout_background_var.get().strip(),
                aesthetics=self.layout_aesthetics_var.get().strip(),
                lighting=self.layout_lighting_var.get().strip(),
                photo=self.layout_style_photo_var.get().strip(),
                art_style=self.layout_style_photo_var.get().strip(),
                medium=self.layout_medium_var.get().strip(),
                style=self.layout_style_var.get().strip() or 'photo',
            )

        def _layout_update_preview_text(self):
            if not hasattr(self, 'layout_json_preview'):
                return
            txt = 'Enable the layout prompt builder to preview the Ideogram JSON prompt.'
            if bool(self.layout_prompt_var.get()):
                try:
                    txt = self._layout_build_prompt_json()
                except Exception as exc:
                    txt = f'Could not build prompt yet: {exc}'
            try:
                self.layout_json_preview.configure(state='normal')
                self.layout_json_preview.delete('1.0', 'end')
                self.layout_json_preview.insert('1.0', txt)
                self.layout_json_preview.configure(state='disabled')
            except Exception:
                pass

        def _layout_preview_prompt(self):
            self._layout_update_preview_text()
            try:
                preview = self._layout_build_prompt_json()
                messagebox.showinfo('Built Ideogram JSON prompt', preview[:3500] + ('\n\n...' if len(preview) > 3500 else ''))
            except Exception as exc:
                messagebox.showerror('Layout prompt', str(exc))

        def _layout_settings_changed(self):
            self._layout_redraw()
            self._layout_update_preview_text()
            self._save_settings()

        def _load_settings(self):
            data = load_gui_settings()
            self.width_var.set(str(data.get("width", GUI_DEFAULTS["width"])))
            self.height_var.set(str(data.get("height", GUI_DEFAULTS["height"])))
            self.steps_var.set(str(data.get("steps", GUI_DEFAULTS["steps"])))
            self.guidance_var.set(str(data.get("guidance", GUI_DEFAULTS["guidance"])))
            self.preset_var.set(str(data.get("preset", GUI_DEFAULTS["preset"])))
            self.seed_var.set(str(data.get("seed", GUI_DEFAULTS["seed"])))
            self.backend_var.set(_backend_name(data.get("backend", GUI_DEFAULTS["backend"])))
            model_dir_value = Path(str(data.get("model_dir", GUI_DEFAULTS["model_dir"]) or ""))
            if not looks_like_sdnq_model_dir(model_dir_value):
                model_dir_value = Path(GUI_DEFAULTS["model_dir"])
            self.model_dir_var.set(str(model_dir_value))
            gguf_dir_value = Path(str(data.get("gguf_dir", GUI_DEFAULTS["gguf_dir"]) or ""))
            if not looks_like_gguf_dir(gguf_dir_value):
                gguf_dir_value = Path(GUI_DEFAULTS["gguf_dir"])
            self.gguf_dir_var.set(str(gguf_dir_value))
            self.gguf_diffusion_var.set(str(data.get("gguf_diffusion_file", "") or ""))
            self.gguf_unconditional_var.set(str(data.get("gguf_unconditional_file", "") or ""))
            self.gguf_llm_var.set(str(data.get("gguf_llm_file", "") or ""))
            self.gguf_vae_var.set(str(data.get("gguf_vae_file", "") or ""))
            sd_cli_value = Path(str(data.get("sd_cli_path", GUI_DEFAULTS["sd_cli_path"]) or ""))
            if not looks_like_sd_cli_path(sd_cli_value):
                sd_cli_value = Path(GUI_DEFAULTS["sd_cli_path"])
            self.sd_cli_path_var.set(str(sd_cli_value))
            self.gguf_max_vram_var.set("0")
            self.gguf_stream_layers_var.set(bool(data.get("gguf_stream_layers", GUI_DEFAULTS["gguf_stream_layers"])))
            self.output_dir_var.set(str(data.get("output_dir", GUI_DEFAULTS["output_dir"])))
            self.compile_var.set(bool(data.get("compile_sdnq", GUI_DEFAULTS["compile_sdnq"])))
            self.text_encoder_offload_var.set(bool(data.get("text_encoder_cpu_offload", GUI_DEFAULTS["text_encoder_cpu_offload"])))
            self.raw_prompt_var.set(bool(data.get("raw_prompt", GUI_DEFAULTS["raw_prompt"])))
            self.layout_prompt_var.set(bool(data.get("layout_prompt_enabled", GUI_DEFAULTS["layout_prompt_enabled"])))
            self.layout_background_var.set(str(data.get("layout_background", GUI_DEFAULTS["layout_background"])))
            style_value = str(data.get("layout_style", GUI_DEFAULTS["layout_style"]))
            if style_value == "art style":
                style_value = "art_style"
            self.layout_style_var.set(style_value)
            self.layout_style_photo_var.set(str(data.get("layout_style_photo", GUI_DEFAULTS["layout_style_photo"])))
            self.layout_aesthetics_var.set(str(data.get("layout_aesthetics", GUI_DEFAULTS["layout_aesthetics"])))
            self.layout_lighting_var.set(str(data.get("layout_lighting", GUI_DEFAULTS["layout_lighting"])))
            self.layout_medium_var.set(str(data.get("layout_medium", GUI_DEFAULTS["layout_medium"])))
            regions = data.get("layout_regions", GUI_DEFAULTS["layout_regions"])
            self.layout_regions = list(regions) if isinstance(regions, list) else []
            theme = normalize_theme_key(data.get("theme", GUI_DEFAULTS.get("theme", "dark")))
            self.theme_var.set(THEME_LABELS.get(theme, THEME_LABELS["dark"]))
            self.layout_selected_index = 0 if self.layout_regions else -1
            self._layout_refresh_list()
            self._layout_redraw()
            self._layout_update_preview_text()

        def _gather_settings(self) -> dict[str, Any]:
            return {
                "width": self.width_var.get().strip() or GUI_DEFAULTS["width"],
                "height": self.height_var.get().strip() or GUI_DEFAULTS["height"],
                "steps": self.steps_var.get().strip() or GUI_DEFAULTS["steps"],
                "guidance": self.guidance_var.get().strip() or GUI_DEFAULTS["guidance"],
                "preset": self.preset_var.get().strip() or GUI_DEFAULTS["preset"],
                "seed": self.seed_var.get().strip() or GUI_DEFAULTS["seed"],
                "backend": _backend_name(self.backend_var.get()),
                "model_dir": self._clean_model_dir_for_save(),
                "gguf_dir": self.gguf_dir_var.get().strip() or GUI_DEFAULTS["gguf_dir"],
                "gguf_diffusion_file": self.gguf_diffusion_var.get().strip(),
                "gguf_unconditional_file": self.gguf_unconditional_var.get().strip(),
                "gguf_llm_file": self.gguf_llm_var.get().strip(),
                "gguf_vae_file": self.gguf_vae_var.get().strip(),
                "sd_cli_path": self.sd_cli_path_var.get().strip() or GUI_DEFAULTS["sd_cli_path"],
                "gguf_max_vram": "0",
                "gguf_stream_layers": bool(self.gguf_stream_layers_var.get()),
                "output_dir": self.output_dir_var.get().strip() or GUI_DEFAULTS["output_dir"],
                "compile_sdnq": bool(self.compile_var.get()),
                "text_encoder_cpu_offload": bool(self.text_encoder_offload_var.get()),
                "raw_prompt": bool(self.raw_prompt_var.get()),
                "layout_prompt_enabled": bool(self.layout_prompt_var.get()),
                "layout_background": self.layout_background_var.get().strip(),
                "layout_style": self.layout_style_var.get().strip(),
                "layout_style_photo": self.layout_style_photo_var.get().strip(),
                "layout_aesthetics": self.layout_aesthetics_var.get().strip(),
                "layout_lighting": self.layout_lighting_var.get().strip(),
                "layout_medium": self.layout_medium_var.get().strip(),
                "layout_regions": self._layout_collect_regions(),
                "theme": self._theme_key(),
                "generate_split": self._current_generate_split(),
            }

        def _clean_model_dir_for_save(self) -> str:
            value = self.model_dir_var.get().strip()
            if not value:
                return GUI_DEFAULTS["model_dir"]
            path = Path(value)
            # Do not persist accidental app/helper folders as the SDNQ model folder.
            if path.exists() and not looks_like_sdnq_model_dir(path):
                try:
                    root = framevision_root().resolve()
                    bad_folders = {root, (root / "helpers").resolve(), (root / "presets").resolve()}
                    if path.resolve() in bad_folders:
                        return GUI_DEFAULTS["model_dir"]
                except Exception:
                    pass
            return value


        def _save_settings(self):
            if getattr(self, "_loading_settings", False):
                return
            try:
                data = load_saved_settings()
                data.update(self._gather_settings())
                save_saved_settings(data)
            except Exception as exc:
                self._write_log(f"Settings save failed: {exc}")

        def _save_queue_state_silent(self):
            try:
                self.queue_data["jobs"] = self.jobs
                self.queue_data["updated_at"] = _now_iso()
                save_queue_state(self.queue_data)
            except Exception:
                pass

        def _save_queue_state(self):
            try:
                self.queue_data["jobs"] = self.jobs
                self.queue_data["updated_at"] = _now_iso()
                save_queue_state(self.queue_data)
            except Exception as exc:
                self._write_log(f"Queue save failed: {exc}")

        def _on_close(self):
            self.shutdown_requested = True
            self._save_settings()
            self._save_queue_state()
            self.root.destroy()

        def _preset_changed(self, event=None):
            preset = self.preset_var.get()
            if preset == "Turbo 12":
                self.steps_var.set("12")
            elif preset == "Default 20":
                self.steps_var.set("20")
            elif preset == "Quality 48":
                self.steps_var.set("48")
            self._save_settings()


        def _backend_changed(self, event=None):
            self._autofill_gguf_install_paths()
            self._refresh_gguf_file_dropdowns(save_fallback=True)
            self._update_backend_visibility()
            self._save_settings()
            self.status_var.set(f"Backend: {_backend_name(self.backend_var.get())}")

        def _autofill_gguf_install_paths(self):
            default_dir = default_gguf_dir()
            current_dir_text = self.gguf_dir_var.get().strip()
            current_dir = Path(current_dir_text) if current_dir_text else None
            try:
                if looks_like_gguf_dir(default_dir) and (not current_dir_text or current_dir is None or not looks_like_gguf_dir(current_dir)):
                    self.gguf_dir_var.set(str(default_dir))
            except Exception:
                if looks_like_gguf_dir(default_dir):
                    self.gguf_dir_var.set(str(default_dir))

            default_path = default_sd_cli_path()
            current_text = self.sd_cli_path_var.get().strip()
            current = Path(current_text) if current_text else None
            try:
                if looks_like_sd_cli_path(default_path) and (not current_text or current is None or not looks_like_sd_cli_path(current)):
                    self.sd_cli_path_var.set(str(default_path))
            except Exception:
                if looks_like_sd_cli_path(default_path):
                    self.sd_cli_path_var.set(str(default_path))

        def _path_widgets(self, *names):
            for name in names:
                widget = getattr(self, name, None)
                if widget is not None:
                    yield widget

        def _update_backend_visibility(self):
            backend = _backend_name(self.backend_var.get())
            sdnq_widgets = list(self._path_widgets(
                "model_dir_label", "model_dir_entry", "model_dir_browse",
                "compile_check", "text_encoder_offload_check",
            ))
            gguf_widgets = list(self._path_widgets(
                "gguf_dir_label", "gguf_dir_entry", "gguf_dir_browse",
                "gguf_diffusion_label", "gguf_diffusion_combo",
                "gguf_unconditional_label", "gguf_unconditional_combo",
                "gguf_llm_label", "gguf_llm_combo",
                "gguf_vae_label", "gguf_vae_combo",
                "sd_cli_label", "sd_cli_entry", "sd_cli_browse",
                "gguf_stream_check",
            ))
            memory_frame = getattr(self, "gguf_memory_frame", None)
            if backend == "gguf":
                if memory_frame is not None:
                    memory_frame.grid()
                for widget in sdnq_widgets:
                    widget.grid_remove()
                for widget in gguf_widgets:
                    widget.grid()
                layout = [
                    ("gguf_dir_label", 0, 0), ("gguf_dir_entry", 0, 1), ("gguf_dir_browse", 0, 2),
                    ("gguf_diffusion_label", 1, 0), ("gguf_diffusion_combo", 1, 1),
                    ("gguf_unconditional_label", 2, 0), ("gguf_unconditional_combo", 2, 1),
                    ("gguf_llm_label", 3, 0), ("gguf_llm_combo", 3, 1),
                    ("gguf_vae_label", 4, 0), ("gguf_vae_combo", 4, 1),
                    ("sd_cli_label", 5, 0), ("sd_cli_entry", 5, 1), ("sd_cli_browse", 5, 2),
                    ("output_dir_label", 6, 0), ("output_dir_entry", 6, 1), ("output_dir_browse", 6, 2),
                ]
                for name, row, col in layout:
                    widget = getattr(self, name, None)
                    if widget is not None:
                        widget.grid_configure(row=row, column=col)
                try:
                    self.gguf_stream_check.grid_configure(row=0, column=0)
                except Exception:
                    pass
            else:
                if memory_frame is not None:
                    memory_frame.grid_remove()
                for widget in sdnq_widgets:
                    widget.grid()
                for widget in gguf_widgets:
                    widget.grid_remove()
                for name, row, col in (
                    ("model_dir_label", 0, 0), ("model_dir_entry", 0, 1), ("model_dir_browse", 0, 2),
                    ("output_dir_label", 1, 0), ("output_dir_entry", 1, 1), ("output_dir_browse", 1, 2),
                ):
                    widget = getattr(self, name, None)
                    if widget is not None:
                        widget.grid_configure(row=row, column=col)
            try:
                self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
                self.settings_canvas.yview_moveto(0)
                self._refresh_canvas_scrollbars()
            except Exception:
                pass

        def _file_names_in_gguf_dir(self, suffixes: tuple[str, ...]) -> list[str]:
            folder = Path(self.gguf_dir_var.get().strip() or str(default_gguf_dir()))
            try:
                if not folder.exists():
                    return []
                names = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
                return sorted(names, key=lambda x: x.lower())
            except Exception:
                return []

        def _pick_default_file(self, names: list[str], current: str, key: str) -> str:
            current = str(current or "").strip()
            if current in names:
                return current
            return names[0] if names else ""

        def _refresh_gguf_file_dropdowns(self, save_fallback: bool = False):
            gguf_names = self._file_names_in_gguf_dir((".gguf",))
            vae_names = self._file_names_in_gguf_dir((".safetensors", ".sft"))
            diffusion_names = [name for name in gguf_names if _gguf_name_role(name) == "diffusion"]
            unconditional_names = [name for name in gguf_names if _gguf_name_role(name) == "unconditional"]
            llm_names = [name for name in gguf_names if _gguf_name_role(name) == "llm"]

            if getattr(self, "gguf_diffusion_combo", None) is not None:
                self.gguf_diffusion_combo.configure(values=diffusion_names)
            if getattr(self, "gguf_unconditional_combo", None) is not None:
                self.gguf_unconditional_combo.configure(values=unconditional_names)
            if getattr(self, "gguf_llm_combo", None) is not None:
                self.gguf_llm_combo.configure(values=llm_names)
            if getattr(self, "gguf_vae_combo", None) is not None:
                self.gguf_vae_combo.configure(values=vae_names)

            before = (
                self.gguf_diffusion_var.get().strip(),
                self.gguf_unconditional_var.get().strip(),
                self.gguf_llm_var.get().strip(),
                self.gguf_vae_var.get().strip(),
            )
            self.gguf_diffusion_var.set(self._pick_default_file(diffusion_names, before[0], "diffusion"))
            self.gguf_unconditional_var.set(self._pick_default_file(unconditional_names, before[1], "unconditional"))
            self.gguf_llm_var.set(self._pick_default_file(llm_names, before[2], "llm"))
            self.gguf_vae_var.set(self._pick_default_file(vae_names, before[3], "vae"))
            after = (
                self.gguf_diffusion_var.get().strip(),
                self.gguf_unconditional_var.get().strip(),
                self.gguf_llm_var.get().strip(),
                self.gguf_vae_var.get().strip(),
            )
            if save_fallback and after != before and not getattr(self, "_loading_settings", False):
                self._save_settings()

        def _browse_model(self):
            folder = filedialog.askdirectory(initialdir=self.model_dir_var.get() or str(default_model_dir()))
            if folder:
                self.model_dir_var.set(folder)
                self._save_settings()

        def _browse_gguf_model(self):
            folder = filedialog.askdirectory(initialdir=self.gguf_dir_var.get() or str(default_gguf_dir()))
            if folder:
                self.gguf_dir_var.set(folder)
                self._refresh_gguf_file_dropdowns(save_fallback=True)
                self._save_settings()

        def _browse_sd_cli(self):
            path = filedialog.askopenfilename(
                initialdir=str(Path(self.sd_cli_path_var.get() or str(default_sd_cli_path())).parent),
                title="Select sd-cli.exe",
                filetypes=[("sd-cli.exe", "sd-cli.exe"), ("Executable files", "*.exe"), ("All files", "*.*")]
            )
            if path:
                self.sd_cli_path_var.set(path)
                self._save_settings()

        def _browse_output(self):
            folder = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(default_output_dir()))
            if folder:
                self.output_dir_var.set(folder)
                self._save_settings()

        def _open_output_folder(self):
            folder = Path(self.output_dir_var.get().strip() or str(default_output_dir()))
            folder.mkdir(parents=True, exist_ok=True)
            try:
                os.startfile(str(folder))
            except Exception as exc:
                messagebox.showerror("Open folder failed", str(exc))

        def _open_logs_folder(self):
            folder = logs_dir()
            folder.mkdir(parents=True, exist_ok=True)
            try:
                os.startfile(str(folder))
            except Exception as exc:
                messagebox.showerror("Open logs folder failed", str(exc))

        def _self_test(self):
            try:
                if _backend_name(self.backend_var.get()) == "gguf":
                    gguf_dir = Path(self.gguf_dir_var.get().strip() or str(default_gguf_dir()))
                    sd_cli = Path(self.sd_cli_path_var.get().strip() or str(default_sd_cli_path()))
                    check_gguf_complete(
                        gguf_dir,
                        sd_cli,
                        self.gguf_diffusion_var.get().strip(),
                        self.gguf_unconditional_var.get().strip(),
                        self.gguf_llm_var.get().strip(),
                        self.gguf_vae_var.get().strip(),
                    )
                    _check_sd_cli_supports_ideogram(sd_cli)
                    msg = f"Backend: GGUF / sd-cli\nsd-cli: {sd_cli}\nModel folder: {gguf_dir}"
                    self._write_log("GGUF self-test OK. " + msg.replace("\n", " | "))
                    messagebox.showinfo("Self-test OK", msg)
                    return

                import torch
                import sdnq  # noqa: F401
                if not torch.cuda.is_available():
                    raise RuntimeError("CPU fallback detected. CUDA is not available.")
                gpu = torch.cuda.get_device_name(0)
                msg = f"Backend: SDNQ / quants\nTorch: {torch.__version__} / CUDA build: {torch.version.cuda}\nGPU: {gpu}"
                self._write_log("Self-test OK. " + msg.replace("\n", " | "))
                messagebox.showinfo("Self-test OK", msg)
            except Exception as exc:
                self._write_log("Self-test failed: " + str(exc))
                messagebox.showerror("Self-test failed", str(exc))

        def _collect_config(self) -> GenerateConfig:
            prompt = self.prompt_text.get("1.0", "end").strip()
            if not prompt:
                raise ValueError("Prompt is empty.")
            width = int(self.width_var.get().strip())
            height = int(self.height_var.get().strip())
            steps = int(self.steps_var.get().strip())
            guidance = float(self.guidance_var.get().strip())
            seed = int(self.seed_var.get().strip())
            backend = _backend_name(self.backend_var.get())
            model_dir_text = self.model_dir_var.get().strip() or GUI_DEFAULTS["model_dir"]
            model_dir = Path(model_dir_text)
            if _backend_name(self.backend_var.get()) == "sdnq" and model_dir.exists() and not looks_like_sdnq_model_dir(model_dir):
                model_dir = Path(GUI_DEFAULTS["model_dir"])
            gguf_dir = Path(self.gguf_dir_var.get().strip() or str(default_gguf_dir()))
            sd_cli_path = Path(self.sd_cli_path_var.get().strip() or str(default_sd_cli_path()))
            gguf_max_vram = float(self.gguf_max_vram_var.get().strip() or "0")
            out_dir = Path(self.output_dir_var.get().strip() or str(default_output_dir()))
            layout_enabled = bool(self.layout_prompt_var.get())
            if layout_enabled:
                if not self._layout_collect_regions():
                    raise ValueError("Add at least one layout region before generating with the layout prompt builder.")
                prompt = self._layout_build_prompt_json()
            return GenerateConfig(
                prompt=prompt,
                negative=self.negative_var.get().strip(),
                model_dir=model_dir,
                output=output_path(out_dir, prefix="ideogram4_gguf" if backend == "gguf" else "ideogram4_sdnq"),
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                preset=self.preset_var.get(),
                compile_sdnq=self.compile_var.get(),
                raw_prompt=(True if layout_enabled else self.raw_prompt_var.get()),
                text_encoder_cpu_offload=self.text_encoder_offload_var.get(),
                backend=backend,
                gguf_dir=gguf_dir,
                gguf_diffusion_file=self.gguf_diffusion_var.get().strip() or DEFAULT_GGUF_FILES["diffusion"],
                gguf_unconditional_file=self.gguf_unconditional_var.get().strip() or DEFAULT_GGUF_FILES["unconditional"],
                gguf_llm_file=self.gguf_llm_var.get().strip() or DEFAULT_GGUF_FILES["llm"],
                gguf_vae_file=self.gguf_vae_var.get().strip() or DEFAULT_GGUF_FILES["vae"],
                sd_cli_path=sd_cli_path,
                gguf_max_vram=gguf_max_vram,
                gguf_stream_layers=self.gguf_stream_layers_var.get(),
            )

        def _config_to_job(self, cfg: GenerateConfig) -> dict[str, Any]:
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            suffix = random.randint(1000, 9999)
            return {
                "id": f"ideo_{stamp}_{suffix}",
                "status": "pending",
                "created_at": _now_iso(),
                "started_at": "",
                "finished_at": "",
                "duration_seconds": 0,
                "prompt": cfg.prompt,
                "prompt_preview": _short_prompt(cfg.prompt),
                "negative": cfg.negative,
                "backend": _backend_name(cfg.backend),
                "model_dir": str(cfg.model_dir),
                "gguf_dir": str(cfg.gguf_dir),
                "gguf_diffusion_file": str(cfg.gguf_diffusion_file),
                "gguf_unconditional_file": str(cfg.gguf_unconditional_file),
                "gguf_llm_file": str(cfg.gguf_llm_file),
                "gguf_vae_file": str(cfg.gguf_vae_file),
                "sd_cli_path": str(cfg.sd_cli_path),
                "gguf_max_vram": float(cfg.gguf_max_vram),
                "gguf_stream_layers": bool(cfg.gguf_stream_layers),
                "output": str(cfg.output) if cfg.output else "",
                "width": int(cfg.width),
                "height": int(cfg.height),
                "steps": int(cfg.steps),
                "guidance": float(cfg.guidance),
                "seed": int(cfg.seed),
                "used_seed": None,
                "preset": cfg.preset,
                "compile_sdnq": bool(cfg.compile_sdnq),
                "text_encoder_cpu_offload": bool(cfg.text_encoder_cpu_offload),
                "raw_prompt": bool(cfg.raw_prompt),
                "notes": "",
                "error": "",
                "cancel_requested": False,
                "progress_current": 0,
                "progress_total": int(cfg.steps),
                "progress_step_seconds": 0.0,
                "eta_seconds": None,
            }

        def _job_to_config(self, job: dict[str, Any]) -> GenerateConfig:
            return GenerateConfig(
                prompt=str(job.get("prompt", "")),
                negative=str(job.get("negative", "")),
                model_dir=Path(str(job.get("model_dir", default_model_dir()))),
                output=Path(str(job.get("output"))) if str(job.get("output", "")).strip() else None,
                width=int(job.get("width", 1024)),
                height=int(job.get("height", 1024)),
                steps=int(job.get("steps", 20)),
                guidance=float(job.get("guidance", 4.0)),
                seed=int(job.get("seed", -1)),
                preset=str(job.get("preset", "Default 20")),
                compile_sdnq=bool(job.get("compile_sdnq", False)),
                raw_prompt=bool(job.get("raw_prompt", False)),
                text_encoder_cpu_offload=bool(job.get("text_encoder_cpu_offload", False)),
                backend=_backend_name(job.get("backend", "sdnq")),
                gguf_dir=Path(str(job.get("gguf_dir", default_gguf_dir()))),
                gguf_diffusion_file=str(job.get("gguf_diffusion_file", DEFAULT_GGUF_FILES["diffusion"])),
                gguf_unconditional_file=str(job.get("gguf_unconditional_file", DEFAULT_GGUF_FILES["unconditional"])),
                gguf_llm_file=str(job.get("gguf_llm_file", DEFAULT_GGUF_FILES["llm"])),
                gguf_vae_file=str(job.get("gguf_vae_file", DEFAULT_GGUF_FILES["vae"])),
                sd_cli_path=Path(str(job.get("sd_cli_path", default_sd_cli_path()))),
                gguf_max_vram=float(job.get("gguf_max_vram", 0.0) or 0.0),
                gguf_stream_layers=bool(job.get("gguf_stream_layers", False)),
            )

        def _start_generate(self):
            try:
                cfg = self._collect_config()
            except Exception as exc:
                messagebox.showerror("Invalid settings", str(exc))
                return

            self._save_settings()
            job = self._config_to_job(cfg)
            with self.jobs_lock:
                self.jobs.append(job)
                self._save_queue_state()
            self.status_var.set(f"Queued: {job['id']}")
            self._write_log(f"Queued job {job['id']}: {_backend_name(cfg.backend)} | {cfg.width}x{cfg.height} | {cfg.preset} | Steps {cfg.steps}")
            self._write_log(f"Output file: {cfg.output}")
            self._refresh_queue_tree()
            self._maybe_start_queue_worker()

        def _maybe_start_queue_worker(self):
            if self.queue_worker and self.queue_worker.is_alive():
                return
            with self.jobs_lock:
                has_pending = any(job.get("status") == "pending" for job in self.jobs)
            if has_pending:
                self.queue_worker = threading.Thread(target=self._queue_worker_loop, daemon=True)
                self.queue_worker.start()

        def _next_pending_job(self) -> dict[str, Any] | None:
            with self.jobs_lock:
                for job in self.jobs:
                    if job.get("status") == "pending":
                        return job
            return None

        def _queue_worker_loop(self):
            while not self.shutdown_requested:
                job = self._next_pending_job()
                if job is None:
                    self.msg_queue.put(("queue_idle", {}))
                    return

                with self.jobs_lock:
                    job["status"] = "running"
                    job["started_at"] = _now_iso()
                    job["finished_at"] = ""
                    job["duration_seconds"] = 0
                    job["error"] = ""
                    job["cancel_requested"] = False
                    job["progress_current"] = 0
                    job["progress_total"] = int(job.get("steps", 0) or 0)
                    job["progress_step_seconds"] = 0.0
                    job["eta_seconds"] = None
                    self.active_job_id = str(job.get("id"))
                    self._save_queue_state_silent()

                self.msg_queue.put(("queue_started", {"job_id": job.get("id")}))
                started = time.perf_counter()
                try:
                    cfg = self._job_to_config(job)
                    out, used_seed, notes = generate_once(cfg, log=self._thread_log)
                    duration = time.perf_counter() - started
                    with self.jobs_lock:
                        if job.get("cancel_requested"):
                            job["status"] = "failed"
                            job["error"] = "Cancelled by user after the active generation finished."
                        else:
                            job["status"] = "finished"
                            job["error"] = ""
                        job["finished_at"] = _now_iso()
                        job["duration_seconds"] = int(duration)
                        job["output"] = str(out)
                        job["used_seed"] = int(used_seed)
                        job["notes"] = str(notes or "")
                        job["progress_current"] = int(job.get("progress_total") or job.get("steps") or 0)
                        job["eta_seconds"] = 0
                        self._save_queue_state_silent()
                    self.msg_queue.put(("queue_done", {"job_id": job.get("id"), "out": str(out), "used_seed": int(used_seed), "notes": notes, "cancelled": bool(job.get("cancel_requested"))}))
                except Exception as exc:
                    duration = time.perf_counter() - started
                    with self.jobs_lock:
                        job["status"] = "failed"
                        job["finished_at"] = _now_iso()
                        job["duration_seconds"] = int(duration)
                        job["error"] = str(exc)
                        job["trace"] = traceback.format_exc()
                        job["eta_seconds"] = None
                        self._save_queue_state_silent()
                    self.msg_queue.put(("queue_failed", {"job_id": job.get("id"), "message": str(exc), "trace": traceback.format_exc()}))
                finally:
                    _safe_cuda_cleanup()
                    self.msg_queue.put(("log", {"text": "Post-job CUDA/Python cleanup finished."}))
                    with self.jobs_lock:
                        self.active_job_id = None

        def _find_job(self, job_id: str | None) -> dict[str, Any] | None:
            if not job_id:
                return None
            with self.jobs_lock:
                for job in self.jobs:
                    if str(job.get("id")) == str(job_id):
                        return job
            return None

        def _selected_queue_job(self) -> dict[str, Any] | None:
            if not hasattr(self, "queue_tree"):
                return None
            sel = self.queue_tree.selection()
            if not sel:
                return None
            return self._find_job(sel[0])

        def _refresh_queue_tree(self):
            if not hasattr(self, "queue_tree"):
                return
            selected = self.queue_tree.selection()
            selected_id = selected[0] if selected else ""
            current_ids = set(self.queue_tree.get_children(""))
            with self.jobs_lock:
                ordered = list(self.jobs)
            wanted_ids = {str(job.get("id")) for job in ordered}
            for item in current_ids - wanted_ids:
                self.queue_tree.delete(item)
            now = datetime.now()
            counts: dict[str, int] = {"running": 0, "pending": 0, "finished": 0, "failed": 0}
            for job in ordered:
                job_id = str(job.get("id"))
                status = str(job.get("status", "pending"))
                if status == "cancelling":
                    display_status = "running"
                else:
                    display_status = status
                if display_status in counts:
                    counts[display_status] += 1
                started_at = str(job.get("started_at", ""))
                finished_at = str(job.get("finished_at", ""))
                started_dt = _parse_iso(started_at)
                finished_dt = _parse_iso(finished_at)
                if status in {"running", "cancelling"} and started_dt:
                    duration = _format_duration((now - started_dt).total_seconds())
                elif finished_dt and started_dt:
                    duration = _format_duration((finished_dt - started_dt).total_seconds())
                else:
                    duration = _format_duration(job.get("duration_seconds", 0)) if job.get("duration_seconds") else ""
                progress_current = int(job.get("progress_current", 0) or 0)
                progress_total = int(job.get("progress_total", 0) or 0)
                progress = f"{progress_current}/{progress_total}" if progress_total > 0 else ""
                eta_value = job.get("eta_seconds", None)
                eta = _format_duration(eta_value) if eta_value not in (None, "") else ""
                if status == "finished":
                    eta = "0:00"
                elif status not in {"running", "cancelling"}:
                    eta = eta if status == "failed" and eta else ""
                resolution = f"{_backend_name(job.get('backend', 'sdnq'))} {job.get('width', '')}x{job.get('height', '')}"
                prompt = str(job.get("prompt_preview") or _short_prompt(str(job.get("prompt", ""))))
                values = (display_status, progress, eta, _format_dt(started_at), duration, _format_dt(finished_at), resolution, prompt)
                if job_id in current_ids:
                    self.queue_tree.item(job_id, values=values)
                else:
                    self.queue_tree.insert("", "end", iid=job_id, values=values)
            if selected_id and selected_id in wanted_ids:
                self.queue_tree.selection_set(selected_id)
            summary = f"Running: {counts['running']}   Pending: {counts['pending']}   Finished: {counts['finished']}   Failed: {counts['failed']}"
            self.queue_summary_var.set(summary)
            self._update_queue_preview_panel()

        def _tick_queue_times(self):
            self._refresh_queue_tree()
            self.root.after(1000, self._tick_queue_times)

        def _queue_right_click(self, event):
            row = self.queue_tree.identify_row(event.y)
            if not row:
                return
            self.queue_tree.selection_set(row)
            job = self._find_job(row)
            if not job:
                return
            status = str(job.get("status", "pending"))
            menu = tk.Menu(self.root, tearoff=0)
            if status in {"running", "cancelling"}:
                menu.add_command(label="Cancel", command=lambda: self._cancel_job(row))
                menu.add_command(label="Show job info", command=lambda: self._show_job_info(row))
            elif status == "pending":
                menu.add_command(label="Remove", command=lambda: self._remove_job(row))
                menu.add_command(label="Show job info", command=lambda: self._show_job_info(row))
            else:
                menu.add_command(label="Remove", command=lambda: self._remove_job(row))
                menu.add_command(label="Show", command=lambda: self._show_queue_job(row))
            menu.tk_popup(event.x_root, event.y_root)

        def _cancel_job(self, job_id: str):
            job = self._find_job(job_id)
            if not job:
                return
            with self.jobs_lock:
                if job.get("status") == "pending":
                    job["status"] = "failed"
                    job["finished_at"] = _now_iso()
                    job["error"] = "Cancelled before it started."
                elif job.get("status") in {"running", "cancelling"}:
                    job["status"] = "cancelling"
                    job["cancel_requested"] = True
                    job["error"] = "Cancel requested. Active CUDA generation will finish before the job is marked cancelled."
                self._save_queue_state()
            self._write_log(f"Cancel requested for job {job_id}.")
            self._refresh_queue_tree()

        def _remove_job(self, job_id: str):
            job = self._find_job(job_id)
            if not job:
                return
            if job.get("status") in {"running", "cancelling"}:
                messagebox.showinfo("Job is running", "Cancel the running job first. It can be removed after it is no longer active.")
                return
            with self.jobs_lock:
                self.jobs[:] = [j for j in self.jobs if str(j.get("id")) != str(job_id)]
                self._save_queue_state()
            self._write_log(f"Removed job {job_id} from the queue list.")
            self._refresh_queue_tree()


        def _job_info_text(self, job: dict[str, Any]) -> str:
            if not job:
                return "No queue item selected."
            lines = [
                f"Job: {job.get('id')}",
                f"Status: {job.get('status')}",
                f"Created: {job.get('created_at', '')}",
                f"Started: {job.get('started_at', '')}",
                f"Finished: {job.get('finished_at', '')}",
                f"Duration: {_format_duration(job.get('duration_seconds', 0))}",
                f"Backend: {_backend_name(job.get('backend', 'sdnq'))}",
                f"Resolution: {job.get('width')}x{job.get('height')}",
                f"Preset: {job.get('preset')} | Steps: {job.get('steps')} | Guidance: {job.get('guidance')}",
                f"Progress: {job.get('progress_current', 0)}/{job.get('progress_total', job.get('steps', 0))}",
                f"ETA: {_format_duration(job.get('eta_seconds')) if job.get('eta_seconds') not in (None, '') else ''}",
                f"Seed: {job.get('seed')} | Used seed: {job.get('used_seed')}",
                f"Output: {job.get('output', '')}",
                "",
                "Prompt:",
                str(job.get("prompt", "")),
            ]
            if job.get("negative"):
                lines.extend(["", "Negative:", str(job.get("negative", ""))])
            if job.get("notes"):
                lines.extend(["", "Notes:", str(job.get("notes", ""))])
            if job.get("error"):
                lines.extend(["", "Error:", str(job.get("error", ""))])
            return "\n".join(lines)

        def _set_queue_info_text(self, text: str):
            widget = getattr(self, "queue_info_text", None)
            if widget is None:
                return
            try:
                widget.configure(state="normal")
                widget.delete("1.0", "end")
                widget.insert("1.0", text)
                widget.configure(state="disabled")
            except Exception:
                pass

        def _set_queue_preview_message(self, text: str):
            canvas = getattr(self, "queue_preview_canvas", None)
            if canvas is None:
                return
            try:
                canvas.delete("all")
                self.queue_preview_ref = None
                w = max(1, canvas.winfo_width())
                h = max(1, canvas.winfo_height())
                self.queue_preview_message_id = canvas.create_text(w // 2, h // 2, text=text, fill="#ffffff", anchor="center")
            except Exception:
                pass

        def _show_queue_preview_image(self, image_path: Path):
            canvas = getattr(self, "queue_preview_canvas", None)
            if canvas is None:
                return
            try:
                img = Image.open(image_path).convert("RGB")
                canvas.update_idletasks()
                cw = max(80, canvas.winfo_width())
                ch = max(80, canvas.winfo_height())
                scale = min(cw / max(1, img.width), ch / max(1, img.height), 1.0)
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                resample = getattr(Image, "Resampling", Image).LANCZOS
                img = img.resize(new_size, resample)
                photo = ImageTk.PhotoImage(img)
                self.queue_preview_ref = photo
                canvas.delete("all")
                x = max(0, (cw - new_size[0]) // 2)
                y = max(0, (ch - new_size[1]) // 2)
                canvas.create_image(x, y, image=photo, anchor="nw")
            except Exception as exc:
                self._set_queue_preview_message(f"Preview failed: {exc}")

        def _update_queue_preview_panel(self):
            if not hasattr(self, "queue_tree"):
                return
            job = self._selected_queue_job()
            if not job:
                self._set_queue_info_text("No queue item selected.")
                self._set_queue_preview_message("Select a queue item")
                return
            self._set_queue_info_text(self._job_info_text(job))
            output = str(job.get("output", "") or "").strip()
            if output and Path(output).exists():
                self._show_queue_preview_image(Path(output))
            elif str(job.get("status", "")) in {"pending", "running", "cancelling"}:
                self._set_queue_preview_message("Output not ready yet")
            else:
                self._set_queue_preview_message("No output image found")

        def _show_selected_queue_job(self):
            job = self._selected_queue_job()
            if not job:
                return
            if job.get("status") in {"finished", "failed"} and job.get("output"):
                self._show_queue_job(str(job.get("id")))
            else:
                self._show_job_info(str(job.get("id")))

        def _show_queue_job(self, job_id: str):
            job = self._find_job(job_id)
            if not job:
                return
            out = Path(str(job.get("output", "")))
            if out.exists():
                self.last_output = out
                self._show_preview(out)
                try:
                    os.startfile(str(out))
                except Exception:
                    pass
            else:
                self._show_job_info(job_id)
                if str(job.get("status")) == "finished":
                    messagebox.showwarning("Output missing", "The output file could not be found.")

        def _show_job_info(self, job_id: str):
            job = self._find_job(job_id)
            if not job:
                return
            messagebox.showinfo("Job info", self._job_info_text(job))

        def _set_preview_message(self, text: str):
            try:
                self.preview_canvas.delete("all")
                self.preview_canvas_image_id = None
                w = max(1, self.preview_canvas.winfo_width())
                h = max(1, self.preview_canvas.winfo_height())
                self.preview_message_id = self.preview_canvas.create_text(w // 2, h // 2, text=text, fill="#ffffff", anchor="center")
                self.preview_canvas.configure(scrollregion=(0, 0, w, h))
                self._refresh_canvas_scrollbars()
                self.preview_zoom_var.set(text)
            except Exception:
                pass

        def _preview_canvas_configured(self, event=None):
            if self.preview_message_id is not None and self.preview_canvas_image_id is None:
                try:
                    self.preview_canvas.coords(self.preview_message_id, max(1, self.preview_canvas.winfo_width()) // 2, max(1, self.preview_canvas.winfo_height()) // 2)
                    w = max(1, self.preview_canvas.winfo_width())
                    h = max(1, self.preview_canvas.winfo_height())
                    self.preview_canvas.configure(scrollregion=(0, 0, w, h))
                    self._refresh_canvas_scrollbars()
                except Exception:
                    pass

        def _preview_pan_start(self, event):
            self.preview_canvas.scan_mark(event.x, event.y)

        def _preview_pan_move(self, event):
            self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

        def _preview_mousewheel_zoom(self, event):
            if self.preview_image_original is None:
                return "break"
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                factor = 1.15
            else:
                factor = 1 / 1.15
            self._zoom_preview(factor, event=event)
            return "break"

        def _zoom_preview(self, factor: float, event=None):
            if self.preview_image_original is None:
                return
            old_zoom = float(self.preview_zoom or 1.0)
            new_zoom = max(0.05, min(8.0, old_zoom * float(factor)))
            if abs(new_zoom - old_zoom) < 0.001:
                return

            canvas = self.preview_canvas
            if event is not None:
                cx = canvas.canvasx(event.x)
                cy = canvas.canvasy(event.y)
                x_fraction = cx / max(1, int(self.preview_image_original.width * old_zoom))
                y_fraction = cy / max(1, int(self.preview_image_original.height * old_zoom))
            else:
                x_fraction = (canvas.canvasx(canvas.winfo_width() // 2)) / max(1, int(self.preview_image_original.width * old_zoom))
                y_fraction = (canvas.canvasy(canvas.winfo_height() // 2)) / max(1, int(self.preview_image_original.height * old_zoom))

            self.preview_zoom = new_zoom
            self._render_preview_image()

            try:
                new_w = max(1, int(self.preview_image_original.width * self.preview_zoom))
                new_h = max(1, int(self.preview_image_original.height * self.preview_zoom))
                target_x = x_fraction * new_w
                target_y = y_fraction * new_h
                if event is not None:
                    left = max(0, target_x - event.x)
                    top = max(0, target_y - event.y)
                else:
                    left = max(0, target_x - canvas.winfo_width() // 2)
                    top = max(0, target_y - canvas.winfo_height() // 2)
                canvas.xview_moveto(left / max(1, new_w))
                canvas.yview_moveto(top / max(1, new_h))
            except Exception:
                pass

        def _reset_preview_zoom(self):
            if self.preview_image_original is None:
                return
            self.preview_zoom = 1.0
            self._render_preview_image()

        def _fit_preview_to_canvas(self):
            if self.preview_image_original is None:
                return
            self.preview_canvas.update_idletasks()
            cw = max(80, self.preview_canvas.winfo_width())
            ch = max(80, self.preview_canvas.winfo_height())
            iw = max(1, self.preview_image_original.width)
            ih = max(1, self.preview_image_original.height)
            self.preview_zoom = max(0.05, min(8.0, min(cw / iw, ch / ih)))
            self._render_preview_image()
            try:
                self.preview_canvas.xview_moveto(0)
                self.preview_canvas.yview_moveto(0)
            except Exception:
                pass

        def _render_preview_image(self):
            if self.preview_image_original is None:
                return
            try:
                zoom = float(self.preview_zoom or 1.0)
                iw = max(1, int(self.preview_image_original.width * zoom))
                ih = max(1, int(self.preview_image_original.height * zoom))
                resample = getattr(Image, "Resampling", Image).LANCZOS
                img = self.preview_image_original.resize((iw, ih), resample)
                photo = ImageTk.PhotoImage(img)
                self.preview_ref = photo
                self.preview_canvas.delete("all")
                self.preview_message_id = None
                self.preview_canvas_image_id = self.preview_canvas.create_image(0, 0, image=photo, anchor="nw")
                self.preview_canvas.configure(scrollregion=(0, 0, iw, ih))
                self._refresh_canvas_scrollbars()
                if self.preview_image_path is not None:
                    self.preview_zoom_var.set(f"{int(round(zoom * 100))}%  •  {self.preview_image_original.width}x{self.preview_image_original.height}  •  {self.preview_image_path.name}")
                else:
                    self.preview_zoom_var.set(f"{int(round(zoom * 100))}%")
            except Exception as exc:
                self._set_preview_message(f"Preview failed: {exc}")

        def _show_preview(self, image_path: Path):
            try:
                img = Image.open(image_path).convert("RGB")
                self.preview_image_original = img.copy()
                self.preview_image_path = Path(image_path)
                self.root.after(50, self._fit_preview_to_canvas)
            except Exception as exc:
                self.preview_image_original = None
                self.preview_image_path = None
                self._set_preview_message(f"Preview failed: {exc}")

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.msg_queue.get_nowait()
                    if kind == "log":
                        self._write_log(payload.get("text", ""))
                    elif kind == "queue_started":
                        job_id = str(payload.get("job_id", ""))
                        self.status_var.set(f"Running: {job_id}")
                        self._write_log(f"Starting queued job {job_id}.")
                        self._refresh_queue_tree()
                    elif kind == "queue_progress":
                        job_id = str(payload.get("job_id", ""))
                        current = int(payload.get("current", 0) or 0)
                        total = int(payload.get("total", 0) or 0)
                        step_seconds = payload.get("step_seconds", None)
                        if step_seconds is not None:
                            try:
                                step_seconds = float(step_seconds)
                            except Exception:
                                step_seconds = None
                        job = self._find_job(job_id)
                        if job is not None:
                            with self.jobs_lock:
                                job["progress_current"] = current
                                if total > 0:
                                    job["progress_total"] = total
                                started_dt = _parse_iso(str(job.get("started_at", "")))
                                if step_seconds is None and started_dt and current > 0:
                                    elapsed = max(0.0, (datetime.now() - started_dt).total_seconds())
                                    step_seconds = elapsed / max(1, current)
                                if step_seconds is not None:
                                    job["progress_step_seconds"] = float(step_seconds)
                                    remaining = max(0, int(job.get("progress_total", total) or total) - current)
                                    job["eta_seconds"] = max(0, int(round(remaining * float(step_seconds))))
                                self._save_queue_state_silent()
                            self.status_var.set(f"Running: {job_id} ({current}/{total})")
                            self._refresh_queue_tree()
                    elif kind == "queue_done":
                        job_id = str(payload.get("job_id", ""))
                        out = Path(str(payload.get("out", "")))
                        used_seed = int(payload.get("used_seed", -1))
                        notes = str(payload.get("notes", "")).strip()
                        cancelled = bool(payload.get("cancelled"))
                        self.last_output = out
                        # Do not overwrite the seed input with the resolved random seed.
                        # When the box says -1, the next queued job must stay random.
                        if cancelled:
                            self.status_var.set(f"Cancelled: {job_id}")
                            self._write_log(f"Cancelled after active run finished: {job_id}")
                        else:
                            self.status_var.set(f"Done: {out.name}")
                            self._write_log(f"Finished job {job_id}. Saved: {out}")
                            self._write_log(f"Seed used: {used_seed}")
                            if notes:
                                for line in notes.splitlines():
                                    if line.strip():
                                        self._write_log(line)
                            self._show_preview(out)
                        self._save_settings()
                        self._refresh_queue_tree()
                    elif kind == "queue_failed":
                        job_id = str(payload.get("job_id", ""))
                        self.status_var.set(f"Failed: {job_id}")
                        self._write_log(f"Job failed: {job_id}")
                        self._write_log(payload.get("message", "Unknown error"))
                        trace = payload.get("trace", "")
                        if trace:
                            self._write_log(trace)
                        self._refresh_queue_tree()
                    elif kind == "queue_idle":
                        self.status_var.set("Queue idle")
                        self._refresh_queue_tree()
            except queue.Empty:
                pass
            self.root.after(200, self._poll_queue)

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # If launched without arguments, open the simple GUI helper.
    if len(sys.argv) == 1 or args.gui:
        return launch_gui()
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
