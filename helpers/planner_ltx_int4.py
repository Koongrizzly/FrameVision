from __future__ import annotations

"""Planner-side INT4 command adapter for the repaired LTX 2.3 CLI.

This helper does not run its own SDNQ/Diffusers workflow. It only detects the
INT4 install and translates Planner clip settings into the same public command
surface used by the working LTX UI and helpers/ltx_int4_cli.py.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_REQUIRED_DIRS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "connectors",
    "scheduler",
    "vae",
    "audio_vae",
    "vocoder",
)


def _framevision_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    try:
        here = Path(__file__).resolve()
        return here.parent.parent if here.parent.name.lower() == "helpers" else here.parent
    except Exception:
        return Path.cwd().resolve()


def _find_ltx_python(root: Path) -> str:
    env_value = str(
        os.environ.get("FRAMEVISION_LTX23_PYTHON", "")
        or os.environ.get("FRAMEVISION_LTX_PYTHON", "")
        or ""
    ).strip()
    candidates: List[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            root / "environments" / ".ltx23" / "python.exe",
            root / "environments" / ".ltx23" / "Scripts" / "python.exe",
            root / "environments" / ".ltx23" / "bin" / "python",
            root / "environments" / ".ltx23_native" / "python.exe",
            root / "environments" / ".ltx23_native" / "Scripts" / "python.exe",
            root / "environments" / ".ltx23_native" / "bin" / "python",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except Exception:
            continue
    return ""


def validate_sdnq_model_root(model_root: Path) -> Dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    missing: List[str] = []
    if not root.is_dir():
        missing.append(str(root))
    if not (root / "model_index.json").is_file():
        missing.append("model_index.json")
    for name in _REQUIRED_DIRS:
        if not (root / name).exists():
            missing.append(name)
    return {
        "ok": not missing,
        "root": str(root),
        "missing": missing,
        "message": (
            "LTX 2.3 INT4 model folder is complete"
            if not missing
            else "Missing INT4 model items: " + ", ".join(missing)
        ),
    }


def int4_install_status(root: Optional[Path] = None) -> Dict[str, Any]:
    base = _framevision_root(root)
    model_value = str(os.environ.get("FRAMEVISION_LTX23_INT4_MODEL_ROOT", "") or "").strip()
    model_root = Path(model_value).expanduser() if model_value else base / "models" / "ltx23_int4"
    if not model_root.is_absolute():
        model_root = base / model_root

    # Important: Planner must call the repaired shared INT4 CLI, not itself.
    helper_path = base / "helpers" / "ltx_int4_cli.py"
    python_exe = _find_ltx_python(base)

    problems: List[str] = []
    check = validate_sdnq_model_root(model_root)
    if not bool(check.get("ok")):
        problems.append(str(check.get("message") or "INT4 model folder incomplete"))
    if not helper_path.is_file():
        problems.append(f"Missing INT4 CLI: {helper_path}")
    if not python_exe:
        problems.append("Missing environments/.ltx23 Python")

    return {
        "ok": not problems,
        "root": str(base),
        "model_root": str(model_root.resolve() if model_root.exists() else model_root),
        "helper_path": str(helper_path),
        "python_exe": python_exe,
        "message": "LTX 2.3 INT4 ready" if not problems else "; ".join(problems),
    }


def _timestamped_report(root: Path, prefix: str = "ltx_int4_planner_report") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(root / "tools" / "vram_lab" / f"{prefix}_{stamp}.txt")


def build_auto_planner_command(
    *,
    app_root: Path,
    native_command: List[str],
    prompt: str,
    negative_prompt: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    shift: float = 5.0,
    start_image: str = "",
    end_image: str = "",
    image_conditions: Optional[List[Tuple[str, int, float]]] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
    audio_path: str = "",
    audio_start_time: float = 0.0,
    audio_max_duration: float = 0.0,
    spatial_upsampler_path: str = "",
    extra_args: Optional[List[str]] = None,
    preferred_backend: str = "auto",
    report_path: str = "",
    deep_log_path: str = "",
) -> Dict[str, Any]:
    """Choose INT4 when available, otherwise keep the untouched native command."""
    root = _framevision_root(app_root)
    preference = str(preferred_backend or "auto").strip().lower().replace("-", "_")
    status = int4_install_status(root)

    native_requested = preference in {"native", "fp16", "fp8", "fp16_fp8"}
    use_int4 = not native_requested and bool(status.get("ok"))
    if not use_int4:
        if native_requested:
            reason = "native explicitly requested"
        elif preference == "int4":
            reason = (
                "recorded INT4 backend unavailable; automatic native fallback: "
                + str(status.get("message") or "INT4 unavailable")
            )
        else:
            reason = str(status.get("message") or "INT4 unavailable")
        return {
            "backend": "native",
            "command": list(native_command),
            "status": status,
            "reason": reason,
        }

    frames = max(1, int(num_frames))
    fps = max(1.0, float(frame_rate))
    audio_text = str(audio_path or "").strip()
    pipeline_name = "a2vid_two_stage" if audio_text else "two_stages"

    command: List[str] = [
        str(status["python_exe"]),
        str(status["helper_path"]),
        "--pipeline", pipeline_name,
        "--model-root", str(status["model_root"]),
        "--vram-profile", "auto",
        "--int4-auto-vram",
        "--prompt", str(prompt or ""),
        "--output-path", str(output_path),
        "--height", str(int(height)),
        "--width", str(int(width)),
        "--num-frames", str(frames),
        "--frame-rate", f"{fps:g}",
        "--num-inference-steps", str(max(1, int(num_inference_steps))),
        "--seed", str(int(seed)),
        "--shift", f"{float(shift):g}",
        "--ltx-root", str(root),
        "--report-path", str(report_path or _timestamped_report(root)),
        "--attention-backend", "auto",
        "--no-boundary-echo",
    ]

    if str(negative_prompt or "").strip():
        command += ["--negative-prompt", str(negative_prompt).strip()]
    if str(spatial_upsampler_path or "").strip():
        command += ["--spatial-upsampler-path", str(spatial_upsampler_path).strip()]
    if str(deep_log_path or "").strip():
        command += ["--deep-log-path", str(deep_log_path).strip(), "--deep-lifecycle-log"]

    start_text = str(start_image or "").strip()
    end_text = str(end_image or "").strip()
    extra_conditions: List[Tuple[str, int, float]] = []
    for item in image_conditions or []:
        try:
            p = str(item[0] or "").strip()
            if p:
                extra_conditions.append((p, int(item[1]), float(item[2])))
        except Exception:
            continue

    # Match the working LTX UI exactly:
    # - start image only -> dedicated I2V pipeline route
    # - start + end / positioned images -> repeated --image conditions
    simple_start_i2v = bool(start_text and not end_text and not extra_conditions)
    if simple_start_i2v:
        command += [
            "--i2v-image", start_text,
            "--i2v-image-frame", "0",
            "--i2v-image-strength", "1",
            "--i2v-image-crf", "0",
        ]
    else:
        conditions: List[Tuple[str, int, float]] = []
        if start_text:
            conditions.append((start_text, 0, 1.0))
        conditions.extend(extra_conditions)
        if end_text:
            conditions.append((end_text, max(0, frames - 1), 1.0))

        seen_conditions = set()
        for path_text, frame, strength in conditions:
            key = (str(Path(path_text).expanduser()), int(frame), float(strength))
            if key in seen_conditions:
                continue
            seen_conditions.add(key)
            command += [
                "--image", path_text,
                str(int(frame)),
                f"{float(strength):g}",
                "0",
            ]

    seen_loras = set()
    for lora_path, multiplier in loras or []:
        path_text = str(lora_path or "").strip()
        if not path_text:
            continue
        key = (str(Path(path_text).expanduser()), float(multiplier))
        if key in seen_loras:
            continue
        seen_loras.add(key)
        command += ["--lora", path_text, f"{float(multiplier):g}"]

    if audio_text:
        command += [
            "--audio-path", audio_text,
            "--audio-start-time", f"{float(audio_start_time):g}",
        ]
        if float(audio_max_duration or 0.0) > 0.0:
            command += ["--audio-max-duration", f"{float(audio_max_duration):.6f}"]

    forwarded_extra = [str(x) for x in list(extra_args or []) if str(x).strip()]

    # Planner uses reference audio only when its Lip Sync path is active.  The
    # ordinary Planner defaults are deliberately gentle and are not strong
    # enough to make the generated face follow speech reliably.  Replace only
    # the audio/video guidance entries for audio-conditioned runs, while
    # leaving all non-lipsync clips untouched.
    if audio_text:
        strong_lipsync = {
            "--video-cfg-guidance-scale": "3",
            "--video-stg-guidance-scale": "1",
            "--video-rescale-scale": "0.7",
            "--audio-cfg-guidance-scale": "7",
            "--audio-stg-guidance-scale": "1",
            "--audio-rescale-scale": "0.7",
            "--a2v-guidance-scale": "3",
            "--v2a-guidance-scale": "3",
        }
        cleaned: List[str] = []
        index = 0
        while index < len(forwarded_extra):
            token = forwarded_extra[index]
            if token in strong_lipsync:
                index += 2 if index + 1 < len(forwarded_extra) else 1
                continue
            cleaned.append(token)
            index += 1
        forwarded_extra = cleaned
        for flag, value in strong_lipsync.items():
            forwarded_extra.extend([flag, value])

    if forwarded_extra:
        command.append("--extra")
        command.extend(forwarded_extra)

    return {
        "backend": "int4",
        "command": command,
        "status": status,
        "reason": (
            f"complete INT4 install detected; pipeline={pipeline_name}; "
            f"condition_route={'dedicated_i2v' if simple_start_i2v else 'positioned_images'}; "
            f"lipsync_profile={'strong' if audio_text else 'normal'}"
        ),
    }
