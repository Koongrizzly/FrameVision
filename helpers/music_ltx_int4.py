from __future__ import annotations

"""Music Clip Creator adapter for the isolated LTX 2.3 INT4 runner.

This file deliberately does not contain INT4 inference or VRAM policy logic.
It reuses the existing ``clip2ltx_cli.py`` music-clip bridge and replaces only
its final command builder so generated shots are sent to ``ltx_int4_cli.py``.
The isolated INT4 CLI remains the single owner of automatic card/profile and
per-stage VRAM planning.
"""

import importlib.util
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_INT4_BACKEND_NAMES = {
    "int4",
    "ltx-int4",
    "ltx23-int4",
    "ltx-23-int4",
    "sdnq-int4",
    "quant-int4",
}
_REQUIRED_BRIDGE_EXPORTS = (
    "export_musicclip_scene_plan",
    "create_prompt_plan",
    "create_ltx_shot_plan",
    "create_ltx_director_plan",
)
_REQUIRED_MODEL_DIRS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "connectors",
    "scheduler",
    "vae",
    "audio_vae",
    "vocoder",
)


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()


def _safe_text(value: Any) -> str:
    try:
        return str(value if value is not None else "").strip()
    except Exception:
        return ""


def _resolve_path(root: Path, value: Any, fallback: Path) -> Path:
    text = _safe_text(value)
    path = Path(text).expanduser() if text else Path(fallback)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _int4_model_root(root: Optional[Path] = None) -> Path:
    base = Path(root or _project_root()).resolve()
    env_path = _safe_text(os.environ.get("FRAMEVISION_LTX23_INT4_MODEL_ROOT"))
    return _resolve_path(base, env_path, base / "models" / "ltx23_int4")


def _int4_cli_path(root: Optional[Path] = None) -> Path:
    base = Path(root or _project_root()).resolve()
    env_path = _safe_text(os.environ.get("FRAMEVISION_LTX23_INT4_CLI"))
    return _resolve_path(base, env_path, base / "helpers" / "ltx_int4_cli.py")


def _base_bridge_path(root: Optional[Path] = None) -> Path:
    base = Path(root or _project_root()).resolve()
    return base / "helpers" / "clip2ltx_cli.py"


def _int4_python(root: Optional[Path] = None) -> str:
    base = Path(root or _project_root()).resolve()
    env_python = _safe_text(os.environ.get("FRAMEVISION_LTX23_PYTHON"))
    candidates = []
    if env_python:
        candidates.append(Path(env_python).expanduser())
    candidates.extend(
        [
            base / "environments" / ".ltx23" / "python.exe",
            base / "environments" / ".ltx23" / "Scripts" / "python.exe",
            base / "environments" / ".ltx23" / "bin" / "python",
            base / "environments" / ".ltx23_native" / "Scripts" / "python.exe",
            base / "environments" / ".ltx23_native" / "bin" / "python",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except Exception:
            continue
    return ""


def _validate_int4_model_root(model_root: Path) -> Tuple[bool, str]:
    root = Path(model_root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        root = root.absolute()
    if not root.is_dir():
        return False, f"INT4 model folder is missing: {root}"

    missing: List[str] = []
    if not (root / "model_index.json").is_file():
        missing.append("model_index.json")
    for name in _REQUIRED_MODEL_DIRS:
        if not (root / name).is_dir():
            missing.append(name + "/")
    for relative in ("transformer/config.json", "text_encoder/config.json"):
        if not (root / relative).is_file():
            missing.append(relative)
    if missing:
        return False, "INT4 model folder is incomplete; missing " + ", ".join(missing)

    try:
        model_index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Could not read INT4 model_index.json: {type(exc).__name__}: {exc}"
    if _safe_text(model_index.get("_class_name")) != "LTX2Pipeline":
        return False, "INT4 model_index.json is not an LTX2Pipeline folder"

    try:
        transformer_config = json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Could not read INT4 transformer/config.json: {type(exc).__name__}: {exc}"
    if not isinstance(transformer_config.get("quantization_config"), dict):
        return False, "INT4 transformer/config.json has no quantization_config"
    return True, f"INT4 model ready: {root}"


def int4_install_status(root: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(root or _project_root()).resolve()
    bridge_path = _base_bridge_path(base)
    cli_path = _int4_cli_path(base)
    model_root = _int4_model_root(base)
    python_exe = _int4_python(base)
    model_ok, model_note = _validate_int4_model_root(model_root)

    problems: List[str] = []
    if not bridge_path.is_file():
        problems.append(f"missing helpers/{bridge_path.name}")
    if not cli_path.is_file():
        problems.append(f"missing helpers/{cli_path.name}")
    if not python_exe:
        problems.append("missing environments/.ltx23 Python")
    if not model_ok:
        problems.append(model_note)

    return {
        "ok": not problems,
        "root": str(base),
        "bridge_path": str(bridge_path),
        "cli_path": str(cli_path),
        "model_root": str(model_root),
        "python_exe": str(python_exe),
        "message": "LTX INT4 is ready" if not problems else "; ".join(problems),
    }


def _load_base_bridge() -> Any:
    path = _base_bridge_path()
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_framevision_musicclip_int4_base", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        for name in _REQUIRED_BRIDGE_EXPORTS:
            if not callable(getattr(module, name, None)):
                return None
        return module
    except BaseException:
        return None


def _parse_resolution(value: Any, default: str = "1280x704") -> Tuple[int, int]:
    text = _safe_text(value) or default
    match = re.search(r"(\d{2,5})\s*[xX]\s*(\d{2,5})", text)
    if not match:
        match = re.search(r"(\d{2,5})\D+(\d{2,5})", default)
    width = int(match.group(1)) if match else 1280
    height = int(match.group(2)) if match else 704
    # The two-stage LTX upsampler requires final dimensions divisible by 64.
    width = max(64, int(round(width / 64.0)) * 64)
    height = max(64, int(round(height / 64.0)) * 64)
    return width, height


def _seed_for_cli(seed: Optional[int]) -> int:
    try:
        value = int(seed) if seed is not None else -1
    except Exception:
        value = -1
    if value < 0:
        return secrets.randbelow(2_147_483_647)
    return value


_BASE = _load_base_bridge()
_ORIGINAL_NORMALIZE = getattr(_BASE, "_normalize_ltx_generation_backend", None) if _BASE is not None else None
_ORIGINAL_BUILD = getattr(_BASE, "_ltx23_build_vramlab_direct_args", None) if _BASE is not None else None
_ORIGINAL_CLI_LOOKUP = getattr(_BASE, "_ltx23_musicclip_vramlab_cli", None) if _BASE is not None else None


def _normalize_backend_for_base(value: Any, root: Optional[Path] = None) -> str:
    text = _safe_text(value).lower().replace("_", "-")
    if text in _INT4_BACKEND_NAMES:
        # The reused bridge has two internal command branches. Map INT4 onto its
        # own-workflow branch; the command builder below then selects INT4.
        return "vramlab"
    if callable(_ORIGINAL_NORMALIZE):
        return str(_ORIGINAL_NORMALIZE(value, root))
    return "vramlab"


def _int4_cli_lookup(root: Path) -> str:
    path = _int4_cli_path(Path(root).resolve())
    return str(path) if path.is_file() else ""


def _build_int4_args(
    *,
    root: Path,
    prompt: str,
    start_image_path: Path,
    out_path: Path,
    fps: int,
    frame_count: int,
    steps: int,
    resolution: str,
    audio_path: str,
    seed: Optional[int],
    lora_file: str = "",
    **_extra: Any,
) -> List[str]:
    base = Path(root).resolve()

    # The isolated INT4 runner currently has no LoRA interface. Preserve the
    # requested feature by using the original FP16/FP8 command builder instead
    # of silently dropping the LoRA.
    if _safe_text(lora_file):
        if not callable(_ORIGINAL_BUILD):
            raise RuntimeError("LTX LoRA was requested, but the FP16/FP8 fallback bridge is unavailable.")
        return list(
            _ORIGINAL_BUILD(
                root=base,
                prompt=prompt,
                start_image_path=start_image_path,
                out_path=out_path,
                fps=fps,
                frame_count=frame_count,
                steps=steps,
                resolution=resolution,
                audio_path=audio_path,
                seed=seed,
                lora_file=lora_file,
            )
        )

    status = int4_install_status(base)
    if not bool(status.get("ok")):
        raise RuntimeError(_safe_text(status.get("message")) or "LTX INT4 is not installed correctly.")

    width, height = _parse_resolution(resolution)
    frame_rate = max(1.0, float(fps or 24))
    frames = max(1, int(frame_count or 1))
    duration = max(0.05, frames / frame_rate)
    command: List[str] = [
        _safe_text(status.get("python_exe")),
        _safe_text(status.get("cli_path")),
        "--pipeline",
        "two_stages",
        "--model-root",
        _safe_text(status.get("model_root")),
        "--vram-profile",
        "auto",
        "--int4-auto-vram",
        "--prompt",
        _safe_text(prompt),
        "--output-path",
        str(Path(out_path).resolve()),
        "--height",
        str(int(height)),
        "--width",
        str(int(width)),
        "--num-frames",
        str(frames),
        "--frame-rate",
        str(frame_rate),
        "--num-inference-steps",
        str(max(1, int(steps or 8))),
        "--seed",
        str(_seed_for_cli(seed)),
        "--shift",
        "5.0",
        "--ltx-root",
        str(base),
    ]

    image = Path(start_image_path).expanduser()
    if image.is_file():
        command.extend(["--i2v-image", str(image.resolve())])
    audio = Path(_safe_text(audio_path)).expanduser() if _safe_text(audio_path) else None
    if audio is not None and audio.is_file():
        command.extend(
            [
                "--audio-path",
                str(audio.resolve()),
                "--audio-start-time",
                "0.0",
                "--audio-max-duration",
                f"{duration:.6f}",
            ]
        )
    return command


if _BASE is not None:
    # Patch only the private routing points used by the copied bridge. All shot
    # planning, image generation, review, duration correction and assembly stay
    # in the existing bridge implementation.
    setattr(_BASE, "_normalize_ltx_generation_backend", _normalize_backend_for_base)
    setattr(_BASE, "_ltx23_musicclip_vramlab_cli", _int4_cli_lookup)
    setattr(_BASE, "_ltx23_build_vramlab_direct_args", _build_int4_args)

    # Re-export the bridge's public API. Functions keep their original module
    # globals, so they see the three patched routing points above.
    for _name in dir(_BASE):
        if _name.startswith("_") or _name == "is_available":
            continue
        globals()[_name] = getattr(_BASE, _name)


def _unavailable_result(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    status = int4_install_status()
    return {"ok": False, "message": _safe_text(status.get("message")) or "LTX INT4 is unavailable."}


# Keep the required bridge surface callable even when installation validation
# fails. auto_music_sync.py will then hide this backend because is_available()
# returns False, while the normal FP16/FP8 bridge remains untouched.
for _required_name in _REQUIRED_BRIDGE_EXPORTS:
    if not callable(globals().get(_required_name)):
        globals()[_required_name] = _unavailable_result


def is_available() -> bool:
    if _BASE is None:
        return False
    return bool(int4_install_status().get("ok"))
