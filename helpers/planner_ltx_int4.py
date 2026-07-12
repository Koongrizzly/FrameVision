from __future__ import annotations

"""FrameVision Planner/UI CLI backend for the split-folder LTX 2.3 INT4 model.

This module deliberately keeps the SDNQ/Diffusers path isolated from the
existing official ltx_core runner.  Importing ``SDNQConfig`` registers the
pre-quantized loader with Transformers/Diffusers; the complete local model
folder is then loaded without merging its transformer/text-encoder shards into
one giant BF16 checkpoint.

Current scope:
- LTX 2.3 Distilled 1.1 SDNQ INT4 folder in Diffusers layout.
- INT8 is intentionally not connected in this restart patch.
- one-stage and Euler two-stage text-to-video / multi-image conditioned video.
- first, middle, final and reference image conditions through repeated --image arguments.
- one or more Diffusers/PEFT LTX-2 LoRAs through repeated --lora arguments.
- synchronized generated audio, plus uploaded-audio A2V through normal two-stage with frozen audio latents.
- workload-aware model/group/sequential CPU offload selected from the FrameVision VRAM profile.
- latent-first Stage 2 teardown so VAE/vocoder decode never overlaps the quant transformer.
- spill-free temporal VAE window decoding with immediate CPU handoff for long clips.

The two-stage SDNQ route loads the existing raw LTX 2.3 x2 spatial-upscaler
checkpoint directly into Diffusers and keeps Stage 1/upsample/Stage 2 as
separate memory phases.
"""

import argparse
import contextlib
import gc
import importlib
import importlib.metadata
import json
import os
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def int4_install_status(root: Optional[Path] = None) -> Dict[str, Any]:
    """Return a strict, import-safe install check used by Planner auto routing."""
    base = _framevision_root(root)
    model_value = str(os.environ.get("FRAMEVISION_LTX23_INT4_MODEL_ROOT", "") or "").strip()
    model_root = Path(model_value).expanduser() if model_value else base / "models" / "ltx23_int4"
    if not model_root.is_absolute():
        model_root = base / model_root
    helper_path = base / "helpers" / "planner_ltx_int4.py"
    python_exe = _find_ltx_python(base)
    problems: List[str] = []
    try:
        validate_sdnq_model_root(model_root)
    except Exception as exc:
        problems.append(str(exc))
    if not helper_path.is_file():
        problems.append(f"Missing Planner INT4 helper: {helper_path}")
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
    """Choose INT4 when available and otherwise return the untouched native command.

    ``preferred_backend`` may be ``auto``, ``int4`` or ``native``. ``auto``
    falls back to the untouched native command when INT4 is unavailable. An
    explicit ``int4`` assignment is strict so resume/recreate cannot silently
    mix a native FP16/FP8 clip into an INT4 Planner job.
    """
    root = _framevision_root(app_root)
    preference = str(preferred_backend or "auto").strip().lower().replace("-", "_")
    status = int4_install_status(root)
    native_requested = preference in {"native", "fp16", "fp8", "fp16_fp8"}
    if preference == "int4" and not bool(status.get("ok")):
        raise RuntimeError(
            "This Planner job is assigned to LTX INT4, but the INT4 installation is not ready: "
            + str(status.get("message") or "unknown INT4 installation error")
        )
    use_int4 = not native_requested and bool(status.get("ok"))
    if not use_int4:
        return {
            "backend": "native",
            "command": list(native_command),
            "status": status,
            "reason": "native explicitly requested" if native_requested else str(status.get("message") or "INT4 unavailable"),
        }

    frames = max(1, int(num_frames))
    fps = max(1.0, float(frame_rate))
    command: List[str] = [
        str(status["python_exe"]),
        str(status["helper_path"]),
        "--pipeline", "two_stages",
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

    conditions: List[Tuple[str, int, float]] = []
    if str(start_image or "").strip():
        conditions.append((str(start_image).strip(), 0, 1.0))
    for item in image_conditions or []:
        try:
            conditions.append((str(item[0]).strip(), int(item[1]), float(item[2])))
        except Exception:
            continue
    if str(end_image or "").strip():
        conditions.append((str(end_image).strip(), max(0, frames - 1), 1.0))
    seen_conditions = set()
    for path_text, frame, strength in conditions:
        if not path_text:
            continue
        key = (str(Path(path_text).expanduser()), int(frame), float(strength))
        if key in seen_conditions:
            continue
        seen_conditions.add(key)
        command += ["--image", path_text, str(int(frame)), f"{float(strength):g}"]

    for lora_path, multiplier in loras or []:
        path_text = str(lora_path or "").strip()
        if path_text:
            command += ["--lora", path_text, f"{float(multiplier):g}"]

    audio_text = str(audio_path or "").strip()
    if audio_text:
        command += ["--audio-path", audio_text, "--audio-start-time", f"{float(audio_start_time):g}"]
        if float(audio_max_duration or 0.0) > 0.0:
            command += ["--audio-max-duration", f"{float(audio_max_duration):.6f}"]
    forwarded_extra = list(extra_args or [])
    if forwarded_extra:
        command.append("--extra")
        command.extend(str(x) for x in forwarded_extra)
    return {
        "backend": "int4",
        "command": command,
        "status": status,
        "reason": "complete INT4 install detected",
    }


def validate_sdnq_model_root(model_root: Path) -> Dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    missing = []
    if not root.is_dir():
        missing.append(str(root))
    if not (root / "model_index.json").is_file():
        missing.append("model_index.json")
    for name in _REQUIRED_DIRS:
        if not (root / name).is_dir():
            missing.append(name + "/")
    for rel in ("transformer/config.json", "text_encoder/config.json"):
        if not (root / rel).is_file():
            missing.append(rel)
    if missing:
        raise FileNotFoundError(
            "SDNQ model folder is incomplete. Missing: " + ", ".join(missing)
        )

    try:
        model_index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read SDNQ model_index.json: {type(exc).__name__}: {exc}") from exc
    if str(model_index.get("_class_name", "")) != "LTX2Pipeline":
        raise RuntimeError(
            f"Unsupported SDNQ pipeline class: {model_index.get('_class_name')!r}; expected 'LTX2Pipeline'."
        )

    try:
        transformer_config = json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read transformer/config.json: {type(exc).__name__}: {exc}") from exc
    quant = transformer_config.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError("transformer/config.json has no SDNQ quantization_config.")

    dtype_map = quant.get("modules_dtype_dict") or {}
    dtype_counts: Dict[str, int] = {}
    if isinstance(dtype_map, dict):
        for key, value in dtype_map.items():
            try:
                dtype_counts[str(key)] = len(value) if isinstance(value, list) else 0
            except Exception:
                dtype_counts[str(key)] = 0

    shard_files = sorted((root / "transformer").glob("*.safetensors"))
    text_encoder_files = sorted((root / "text_encoder").glob("*.safetensors"))
    missing_shards = []
    if not shard_files:
        missing_shards.append("transformer/*.safetensors")
    if not text_encoder_files:
        missing_shards.append("text_encoder/*.safetensors")
    if missing_shards:
        raise FileNotFoundError(
            "SDNQ model folder has configs but no model weights. Missing: "
            + ", ".join(missing_shards)
        )
    return {
        "root": root,
        "model_index": model_index,
        "transformer_config": transformer_config,
        "quantization_config": quant,
        "dtype_counts": dtype_counts,
        "transformer_shards": shard_files,
        "text_encoder_shards": text_encoder_files,
        "transformer_bytes": sum(p.stat().st_size for p in shard_files if p.is_file()),
        "text_encoder_bytes": sum(p.stat().st_size for p in text_encoder_files if p.is_file()),
    }


def _extra_option(extra: Iterable[str], names: Tuple[str, ...], default: Any = None) -> Any:
    tokens = list(extra or [])
    for index, token in enumerate(tokens):
        if token in names and index + 1 < len(tokens):
            return tokens[index + 1]
        for name in names:
            prefix = name + "="
            if str(token).startswith(prefix):
                return str(token)[len(prefix) :]
    return default


def _float_option(extra: Iterable[str], names: Tuple[str, ...], default: float) -> float:
    try:
        return float(_extra_option(extra, names, default))
    except Exception:
        return float(default)


def _int_option(extra: Iterable[str], names: Tuple[str, ...], default: int) -> int:
    try:
        return int(str(_extra_option(extra, names, default)).strip())
    except Exception:
        return int(default)


_SAFE_GROUP_BLOCKS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24, 32, 48)
_BASELINE_WORK_UNITS = float(832 * 448 * 241)


def _snap_group_blocks(requested: int, block_count: int) -> int:
    requested = max(1, min(int(requested), int(block_count)))
    choices = [value for value in _SAFE_GROUP_BLOCKS if value <= requested and value <= block_count]
    return max(choices) if choices else 1


def _select_stepped_group(work_factor: float, steps: Tuple[Tuple[float, int], ...], fallback: int) -> int:
    """Return the first group size whose workload ceiling contains the job."""
    for ceiling, group_blocks in steps:
        if float(work_factor) <= float(ceiling):
            return int(group_blocks)
    return int(fallback)


def _select_sdnq_offload_policy(
    *,
    weight_dtype: str,
    profile_gb: int,
    width: int,
    height: int,
    num_frames: int,
    block_count: int,
    override_group_blocks: int = 0,
    override_mode: str = "auto",
    workflow: str = "one_stage",
    stage: str = "stage1",
    input_mode: str = "text-to-video",
    automation_enabled: bool = True,
    int4_24_stage2_full_max_work: float = 4.40,
    int4_24_i2v_stage1_full_max_work: float = 0.45,
    int4_24_i2v_stage2_full_max_work: float = 1.25,
) -> Dict[str, Any]:
    """Choose the residency policy for one specific denoise stage.

    The workload unit is resolution × frames relative to the original
    832×448×241 calibration point.  This makes every boundary automatically
    move with resolution instead of treating frame count as the only source of
    pressure.

    The 24 GB two-stage I2V path preserves the measured fast 121-frame route,
    then uses progressively smaller Stage 2 groups as full-resolution temporal
    pressure rises.  Music-synchronised 704p jobs around 225-241 aligned frames
    use a 4-block Stage 2 group so refinement steps 2 and 3 retain real driver
    headroom instead of touching the WDDM shared-memory floor.

    The 16 GB, 12 GB, and 8 GB Stage 2 plans step down even earlier.  Stage 1
    remains on the already-proven faster policies; this calibration is focused
    specifically on full-resolution Stage 2 residency.

    Manual mode/group overrides still win.  Disabling the automatic planner
    selects a fixed per-card fallback instead of applying workload thresholds.
    """

    dtype_text = str(weight_dtype or "unknown").strip().lower()
    if "4" in dtype_text:
        variant = "int4"
    elif "8" in dtype_text:
        variant = "int8"
    else:
        variant = dtype_text or "unknown"

    try:
        profile = int(profile_gb)
    except Exception:
        profile = 24
    if profile >= 20:
        profile_bucket = 24
    elif profile >= 14:
        profile_bucket = 16
    elif profile >= 10:
        profile_bucket = 12
    else:
        profile_bucket = 8

    blocks = max(1, int(block_count or 48))
    work_factor = max(
        0.05,
        float(max(1, width) * max(1, height) * max(1, num_frames))
        / _BASELINE_WORK_UNITS,
    )
    equivalent_704_frames = (
        float(max(1, width) * max(1, height) * max(1, num_frames))
        / float(1280 * 704)
    )
    workflow = str(workflow or "one_stage").strip()
    stage = str(stage or "stage1").strip().lower()
    input_mode = str(input_mode or "text-to-video").strip().lower()
    is_i2v = input_mode in {
        "i2v", "image", "image-to-video", "image_to_video",
        "image-conditioned-video", "image_conditioned_video",
    }

    requested_mode = str(override_mode or "auto").strip().lower().replace("-", "_")
    if requested_mode not in {
        "auto",
        "group",
        "group_offload",
        "model",
        "model_offload",
        "sequential",
        "sequential_offload",
    }:
        requested_mode = "auto"

    # Explicit user overrides always win over the under-the-hood planner.
    if requested_mode in {"model", "model_offload"}:
        desired = blocks
        mode = "model_cpu_offload"
        use_stream = False
        reason = "manual model offload override"
    elif requested_mode in {"sequential", "sequential_offload"}:
        desired = 1
        mode = "sequential_cpu_offload"
        use_stream = False
        reason = "manual sequential offload override"
    elif requested_mode in {"group", "group_offload"}:
        if int(override_group_blocks or 0) > 0:
            desired = int(override_group_blocks)
        else:
            desired = {8: 2, 12: 4, 16: 8, 24: 12}[profile_bucket]
        mode = "group_cpu_offload"
        use_stream = False
        reason = "manual group offload override; prefetch disabled for predictable VRAM"
    elif variant == "int4" and not bool(automation_enabled):
        # VRAM Lab OFF in the UI: keep a deterministic fixed policy.  Stage 1
        # retains its established profile while Stage 2 still receives a safe
        # per-card residency ceiling; disabling automation must not turn a
        # 16/24 GB selection into an accidental full-resolution spill path.
        if profile_bucket == 24:
            desired = blocks if stage == "stage1" else 4
        elif profile_bucket == 16:
            desired = 16 if stage == "stage1" else 3
        elif profile_bucket == 12:
            desired = 8 if stage == "stage1" else 2
        else:
            desired = 4 if stage == "stage1" else 1
        if stage == "stage1" and profile_bucket == 24:
            mode = "model_cpu_offload"
        else:
            mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
        use_stream = False
        reason = (
            f"INT4 automatic VRAM planner disabled; fixed conservative "
            f"{profile_bucket}GB {stage} policy"
        )
    elif (
        variant == "int4"
        and profile_bucket == 24
        and workflow == "two_stages"
        and stage == "stage1"
    ):
        # Preserve the proven 121-frame/704p fast path. Larger jobs switch to
        # synchronous groups before Windows shared-memory paging begins.
        full_limit = float(int4_24_i2v_stage1_full_max_work) if is_i2v else 1.50
        if work_factor <= full_limit:
            desired = blocks
            mode = "model_cpu_offload"
            use_stream = False
            reason = (
                f"INT4 24GB {'I2V' if is_i2v else 'T2V'} Stage 1 fast path "
                f"(work={work_factor:.3f} <= {full_limit:.3f})"
            )
        else:
            first_group_limit = 0.80 if is_i2v else 2.00
            second_group_limit = 1.20 if is_i2v else 2.75
            if work_factor <= first_group_limit:
                # The music route normally runs Stage 1 without CFG and the
                # measured 24-block path used only ~11.7 GB at 241 model
                # frames.  A 32-block group uses the available 24 GB card more
                # productively while retaining a large activation reserve.
                desired = 32
            elif work_factor <= second_group_limit:
                desired = 24
            else:
                desired = 16
            mode = "group_cpu_offload"
            use_stream = False
            reason = (
                f"INT4 24GB {'I2V' if is_i2v else 'T2V'} Stage 1 guard; "
                f"synchronous groups without CUDA prefetch (work={work_factor:.3f})"
            )
    elif (
        variant == "int4"
        and profile_bucket == 24
        and workflow == "two_stages"
        and stage == "stage2"
    ):
        full_limit = (
            float(int4_24_i2v_stage2_full_max_work)
            if is_i2v
            else float(int4_24_stage2_full_max_work)
        )
        if work_factor <= full_limit:
            desired = blocks
            mode = "model_cpu_offload"
            use_stream = False
            reason = (
                f"INT4 24GB {'I2V' if is_i2v else 'T2V'} Stage 2 fast path "
                f"(work={work_factor:.3f} <= {full_limit:.3f})"
            )
        else:
            if is_i2v:
                desired = _select_stepped_group(
                    work_factor,
                    (
                        (1.55, 24),
                        (1.85, 14),
                        # Full-resolution refinement grows substantially after
                        # the first step.  Reduce residency before the 201-217f
                        # band, then use four blocks for the common 225-241f
                        # Music Creator route.  The measured 8-block route hit
                        # 24.59 GB reserved / 0.00 GB driver-free on step 2/3.
                        (2.05, 8),
                        (2.25, 6),
                        (2.80, 4),
                        (3.40, 3),
                        (4.25, 2),
                    ),
                    1,
                )
            else:
                desired = _select_stepped_group(
                    work_factor,
                    ((5.50, 12), (6.50, 8)),
                    6,
                )
            mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
            use_stream = False
            reason = (
                f"INT4 24GB {'I2V' if is_i2v else 'T2V'} Stage 2 boundary guard; "
                f"synchronous groups without CUDA prefetch (work={work_factor:.3f})"
            )
    elif variant == "int4" and profile_bucket == 24:
        # One-stage and any future non-standard workflow still receive a real
        # 24 GB plan. Two-stage Stage 1/Stage 2 were handled by the calibrated
        # branches above.
        if work_factor <= 1.25:
            desired = blocks
            mode = "model_cpu_offload"
        else:
            desired = _select_stepped_group(
                work_factor,
                ((2.75, 24), (4.00, 12)),
                8,
            )
            mode = "group_cpu_offload"
        use_stream = False
        reason = (
            f"INT4 automatic 24GB {stage} general planner; "
            f"group={desired} (work={work_factor:.3f})"
        )
    elif variant == "int4" and profile_bucket == 16:
        # A 16 GB card cannot safely hold the complete 10.74 GB transformer plus
        # 704p activations. Stage 2 therefore begins with 16 blocks. Near the
        # user's expected ~180-frame/704p boundary (work≈1.80), it steps to 12.
        if stage == "stage1":
            desired = _select_stepped_group(
                work_factor,
                # The measured 24-block Stage 1 route stayed around 11.7 GB
                # for 241 model frames at 704p, so it is also the useful fast
                # tier for a real 16 GB card.  Step down only as the quarter-
                # resolution activation load grows beyond that range.
                ((0.75, 24), (1.00, 16), (1.30, 12)),
                8,
            )
        else:
            desired = _select_stepped_group(
                work_factor,
                # Stage 2 is activation-heavy and cannot reuse the Stage 1
                # residency assumptions.  A 241f/704p job now resolves to
                # three blocks instead of eight/twelve.
                ((1.10, 8), (1.55, 6), (1.95, 4), (2.75, 3), (3.40, 2)),
                1,
            )
        mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
        use_stream = False
        reason = (
            f"INT4 automatic 16GB {stage} planner; resolution/frame-aware "
            f"group={desired} (work={work_factor:.3f})"
        )
    elif variant == "int4" and profile_bucket == 12:
        # On 12 GB, the same activation workload reaches the boundary roughly
        # twice as early. Stage 1 begins stepping down around 121 frames at
        # 704p; Stage 2 uses a 4-block group by that workload.
        if stage == "stage1":
            desired = _select_stepped_group(
                work_factor,
                ((0.25, 16), (0.45, 12), (0.65, 8), (0.90, 6)),
                4,
            )
        else:
            desired = _select_stepped_group(
                work_factor,
                ((0.65, 6), (1.10, 4), (1.80, 3), (2.75, 2)),
                1,
            )
        mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
        use_stream = False
        reason = (
            f"INT4 automatic 12GB {stage} planner; resolution/frame-aware "
            f"group={desired} (work={work_factor:.3f})"
        )
    elif variant == "int4" and profile_bucket == 8:
        # Emergency 8 GB profile. This is intentionally conservative and may be
        # slow, but it avoids pretending a 12 GB plan can fit.
        if stage == "stage1":
            desired = _select_stepped_group(
                work_factor,
                ((0.25, 8), (0.50, 6), (0.80, 4)),
                2,
            )
        else:
            desired = _select_stepped_group(
                work_factor,
                ((0.55, 3), (1.10, 2)),
                1,
            )
        mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
        use_stream = False
        reason = (
            f"INT4 automatic 8GB {stage} emergency planner; "
            f"group={desired} (work={work_factor:.3f})"
        )
    else:
        # INT8 is not connected to the UI, but retain the prior conservative
        # fallback for manual commands.
        if profile_bucket == 24:
            desired = 24
            if stage == "stage2" and work_factor > 1.25:
                desired = 16
            if work_factor > 2.75:
                desired = 12
            mode = "group_cpu_offload"
        elif profile_bucket == 16:
            desired = 12 if stage == "stage1" else 8
            if work_factor > 1.5:
                desired = 6
            if work_factor > 3.0:
                desired = 4
            mode = "group_cpu_offload"
        elif profile_bucket == 12:
            desired = 8 if stage == "stage1" else 4
            if work_factor > 1.5:
                desired = 3
            if work_factor > 3.0:
                desired = 2
            mode = "group_cpu_offload"
        else:
            desired = 4 if stage == "stage1" else 2
            if work_factor > 1.5:
                desired = 1
            mode = "group_cpu_offload" if desired > 1 else "sequential_cpu_offload"
        use_stream = False
        reason = (
            f"SDNQ {variant.upper()} {profile_bucket}GB {stage} profile; "
            "synchronous residency prevents shared-memory prefetch spill"
        )

    if int(override_group_blocks or 0) > 0 and mode == "group_cpu_offload":
        desired = int(override_group_blocks)

    group_blocks = _snap_group_blocks(desired, blocks)
    if mode == "model_cpu_offload":
        group_blocks = blocks

    return {
        "variant": variant,
        "profile_bucket": profile_bucket,
        "work_factor": work_factor,
        "equivalent_704_frames": equivalent_704_frames,
        "block_count": blocks,
        "group_blocks": group_blocks,
        "mode": mode,
        "override_mode": requested_mode,
        "automation_enabled": bool(automation_enabled),
        "use_stream": bool(use_stream),
        "reason": reason,
        "stage": stage,
        "input_mode": "image-to-video" if is_i2v else "text-to-video",
        "policy_tier": (
            "fast" if mode == "model_cpu_offload"
            else "guarded" if mode == "group_cpu_offload"
            else "heavy"
        ),
    }


def _configure_attention_backend(pipe: Any, requested: str, ctx: Dict[str, Any]) -> str:
    """Select a bounded-memory Diffusers attention backend for LTX INT4.

    Native SDPA can choose a math fallback for some long cross-attention
    shapes.  That fallback materialises much larger temporary tensors and is a
    poor match for the 704p Stage 2 sequence.  ``_native_efficient`` is part of
    PyTorch/Diffusers and does not require another downloaded package.
    """
    requested_text = str(requested or "auto").strip().lower().replace("-", "_")
    backend_map = {
        "auto": "_native_efficient",
        "efficient": "_native_efficient",
        "native_efficient": "_native_efficient",
        "native": "native",
        "flash": "_native_flash",
        "native_flash": "_native_flash",
        "sage": "sage",
    }
    selected = backend_map.get(requested_text, "_native_efficient")
    transformer = getattr(pipe, "transformer", None)
    setter = getattr(transformer, "set_attention_backend", None)
    if not callable(setter):
        ctx["ltx_sdnq_attention_backend"] = (
            "default: transformer does not expose set_attention_backend"
        )
        return "default"
    try:
        setter(selected)
        ctx["ltx_sdnq_attention_backend"] = (
            f"{selected}: explicit bounded-memory attention dispatcher"
        )
        print(
            f"[ltx-status] SDNQ attention backend: {selected}",
            flush=True,
        )
        return selected
    except Exception as exc:
        # The PyTorch native default remains a correct fallback.  Do not abort
        # model loading merely because an optional dispatcher backend is absent.
        try:
            resetter = getattr(transformer, "reset_attention_backend", None)
            if callable(resetter):
                resetter()
        except Exception:
            pass
        ctx["ltx_sdnq_attention_backend"] = (
            f"default fallback; {selected} unavailable: {type(exc).__name__}: {exc}"
        )
        print(
            "[ltx-warning] Bounded-memory attention backend unavailable; "
            f"using Diffusers default: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return "default"


def _resolve_prompt_sequence_length(
    tokenizer: Any,
    prompt: str,
    negative_prompt: Optional[str],
    do_cfg: bool,
    *,
    hard_max: int = 1024,
) -> Tuple[int, int]:
    """Return a connector-compatible padded length for the prompt batch.

    Diffusers defaults LTX-2 to 1024 text tokens even for a short prompt.  The
    resulting all-padding cross-attention mask scales with the video token
    count and becomes expensive around 9-10 seconds at 704p.  LTX-2's video
    connector groups text states around 128 learnable registers, so the cached
    sequence length must remain divisible by 128.  Probe token lengths on CPU
    and round up to the next 128-token block without removing any real token.
    """
    maximum = max(128, int(hard_max))
    values = [str(prompt or "")]
    if bool(do_cfg):
        values.append(str(negative_prompt or ""))
    try:
        encoded = tokenizer(
            values,
            padding=False,
            truncation=True,
            max_length=maximum,
            add_special_tokens=True,
        )
        ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        lengths = []
        if ids is not None:
            if ids and isinstance(ids[0], int):
                lengths = [len(ids)]
            else:
                lengths = [len(row) for row in ids]
        actual = max(lengths) if lengths else maximum
    except Exception:
        actual = maximum
    register_block = 128
    effective = min(
        maximum,
        max(register_block, ((int(actual) + register_block - 1) // register_block) * register_block),
    )
    return int(effective), int(actual)


def _driver_free_bytes(torch_module: Any) -> Optional[int]:
    """Return current driver-visible free CUDA memory, or None if unavailable."""
    try:
        if not torch_module.cuda.is_available():
            return None
        return int(torch_module.cuda.mem_get_info()[0])
    except Exception:
        return None


def _cuda_text(torch_module: Any) -> str:
    try:
        if not torch_module.cuda.is_available():
            return "CUDA unavailable"
        allocated = float(torch_module.cuda.memory_allocated()) / (1024 ** 3)
        reserved = float(torch_module.cuda.memory_reserved()) / (1024 ** 3)
        free_b, total_b = torch_module.cuda.mem_get_info()
        return (
            f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, "
            f"driver_free={float(free_b)/(1024**3):.2f} GB/{float(total_b)/(1024**3):.2f} GB"
        )
    except Exception as exc:
        return f"CUDA snapshot failed: {type(exc).__name__}: {exc}"


def _version_of(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _prepare_sdnq_cache_env(args: Any, model_root: Path) -> Path:
    """Use persistent portable caches so Triton/Inductor compilation survives app restarts."""

    root_text = str(getattr(args, "ltx_root", "") or "").strip()
    if root_text:
        framevision_root = Path(root_text).expanduser().resolve()
    else:
        # models/ltx23_int8 -> FrameVision root
        framevision_root = model_root.parent.parent if model_root.parent.name.lower() == "models" else model_root.parent

    cache_root = framevision_root / "cache" / "ltx23_sdnq"
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", "0")
    return cache_root


def _probe_triton_inductor(torch_module: Any) -> Dict[str, Any]:
    """Run a real CUDA compile probe instead of trusting that ``import triton`` works."""

    result: Dict[str, Any] = {
        "ok": False,
        "triton_version": "unknown",
        "triton_distribution": "unknown",
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "compile_time_s": 0.0,
        "detail": "not attempted",
    }
    try:
        import triton  # type: ignore

        result["triton_version"] = str(getattr(triton, "__version__", "unknown"))
        try:
            result["triton_distribution"] = importlib.metadata.version("triton-windows")
        except Exception:
            try:
                result["triton_distribution"] = importlib.metadata.version("triton")
            except Exception:
                result["triton_distribution"] = "unknown"

        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        # Torch 2.8 requires Triton 3.4. A newer Triton can import successfully
        # while still being incompatible with Inductor, which was the exact
        # failure mode that silently pushed SDNQ into PyTorch Eager mode.
        torch_mm = ".".join(str(torch_module.__version__).split("+")[0].split(".")[:2])
        triton_mm = ".".join(str(result["triton_version"]).split("+")[0].split(".")[:2])
        expected = {"2.7": "3.3", "2.8": "3.4", "2.9": "3.5", "2.10": "3.6"}.get(torch_mm)
        if expected is not None and triton_mm != expected:
            raise RuntimeError(
                f"Torch {torch_mm} requires Triton {expected}.x for this FrameVision environment, "
                f"but Triton {result['triton_version']} is installed"
            )

        def _probe_fn(a, b):
            return torch_module.nn.functional.silu(a @ b)

        t0 = time.perf_counter()
        compiled = torch_module.compile(_probe_fn, fullgraph=True, dynamic=False)
        a = torch_module.randn((32, 32), device="cuda", dtype=torch_module.float16)
        b = torch_module.randn((32, 32), device="cuda", dtype=torch_module.float16)
        out = compiled(a, b)
        torch_module.cuda.synchronize()
        expected_out = torch_module.nn.functional.silu(a @ b)
        if not torch_module.allclose(out, expected_out, rtol=1e-3, atol=1e-3):
            raise RuntimeError("compiled CUDA probe returned an incorrect result")
        result["compile_time_s"] = time.perf_counter() - t0
        result["ok"] = True
        result["detail"] = "Triton/Inductor CUDA compile and execution passed"
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


def _cpu_tensor(value: Any) -> Any:
    if value is None:
        return None
    try:
        return value.detach().to(device="cpu")
    except Exception:
        return value


def _set_vae_spatial_tiling(
    vae: Any,
    *,
    enabled: bool,
    tile_height: int = 512,
    tile_width: int = 512,
    stride_height: int = 448,
    stride_width: int = 448,
) -> str:
    """Configure only spatial VAE tiling.

    LTX 2.3 has a separate temporal/framewise decode path.  Spatial tiling by
    itself does not solve long-video decode pressure because every temporal
    activation is still present at once.  The final decode therefore uses the
    streaming temporal helper below and enables spatial tiling only for very
    small VRAM profiles or unusually large output resolutions.
    """

    details = []
    if enabled:
        enable = getattr(vae, "enable_tiling", None)
        if callable(enable):
            try:
                enable(
                    tile_sample_min_height=int(tile_height),
                    tile_sample_min_width=int(tile_width),
                    tile_sample_stride_height=int(stride_height),
                    tile_sample_stride_width=int(stride_width),
                )
            except TypeError:
                enable()
            details.append("enable_tiling()")
        if hasattr(vae, "use_tiling"):
            setattr(vae, "use_tiling", True)
            details.append("use_tiling=True")
        if not bool(getattr(vae, "use_tiling", False)):
            raise RuntimeError("The video VAE did not enable spatial tiling")
    else:
        disable = getattr(vae, "disable_tiling", None)
        if callable(disable):
            disable()
            details.append("disable_tiling()")
        if hasattr(vae, "use_tiling"):
            setattr(vae, "use_tiling", False)
            details.append("use_tiling=False")
        if bool(getattr(vae, "use_tiling", False)):
            raise RuntimeError("The video VAE still reports spatial tiling enabled")
    return ", ".join(details) or ("spatial tiling enabled" if enabled else "spatial tiling disabled")


def _select_video_decode_plan(
    *,
    profile_gb: int,
    width: int,
    height: int,
    num_frames: int,
    extra: Iterable[str],
) -> Dict[str, Any]:
    """Choose a spill-free temporal VAE decode window.

    A full 241-frame 1280x704 decode retained roughly 13.6 GB of live tensors
    and made the CUDA allocator reserve 35 GB, which forced Windows to back
    about 12.7 GB with shared GPU memory.  Decoding overlapping temporal
    windows keeps the same official causal/blended behavior while ensuring
    only one window resides on CUDA at a time.
    """

    profile = int(profile_gb)
    pixels = max(1, int(width) * int(height))
    if profile >= 24:
        if pixels <= 1280 * 720:
            window = 48
        elif pixels <= 1920 * 1088:
            window = 32
        else:
            window = 24
    elif profile >= 16:
        window = 32 if pixels <= 1280 * 720 else 24
    elif profile >= 12:
        window = 24 if pixels <= 1280 * 720 else 16
    else:
        window = 16

    window = _int_option(
        extra,
        ("--sdnq-decode-window-frames", "--sdnq_decode_window_frames"),
        window,
    )
    overlap = _int_option(
        extra,
        ("--sdnq-decode-overlap-frames", "--sdnq_decode_overlap_frames"),
        8,
    )
    spatial_override = str(
        _extra_option(
            extra,
            ("--sdnq-decode-spatial-tiling", "--sdnq_decode_spatial_tiling"),
            "auto",
        )
        or "auto"
    ).strip().lower()

    # LTX video latents use 8x temporal compression.  Keep both values aligned
    # so every emitted section starts on a real latent boundary.
    temporal_ratio = 8
    window = max(16, (int(window) // temporal_ratio) * temporal_ratio)
    overlap = max(temporal_ratio, (int(overlap) // temporal_ratio) * temporal_ratio)
    if overlap >= window:
        overlap = temporal_ratio
    stride = window - overlap

    if spatial_override in {"1", "true", "yes", "on", "enabled"}:
        spatial_tiling = True
    elif spatial_override in {"0", "false", "no", "off", "disabled"}:
        spatial_tiling = False
    else:
        spatial_tiling = bool(profile <= 12 or pixels > 1920 * 1088)

    return {
        "window_frames": int(window),
        "stride_frames": int(stride),
        "overlap_frames": int(overlap),
        "spatial_tiling": bool(spatial_tiling),
        "temporal_streaming": bool(int(num_frames) > int(window)),
    }


def _streaming_temporal_vae_decode(
    *,
    vae: Any,
    latents: Any,
    torch_module: Any,
    target_num_frames: int,
    window_frames: int,
    stride_frames: int,
    spatial_tiling: bool,
    sample_memory: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Decode LTX video latents in overlapping temporal windows.

    This mirrors Diffusers' official ``_temporal_tiled_decode`` overlap and
    blending rules, but moves each completed window to system RAM immediately.
    The stock helper keeps every decoded window on CUDA until the final concat,
    which is unnecessary for an inference-only FrameVision export and can still
    build a large reservation on Windows.
    """

    if latents is None or not hasattr(latents, "shape") or len(latents.shape) != 5:
        raise RuntimeError("Expected 5D LTX video latents for temporal VAE decode")

    temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 8) or 8)
    if temporal_ratio <= 0:
        temporal_ratio = 8
    latent_window = max(1, int(window_frames) // temporal_ratio)
    latent_stride = max(1, int(stride_frames) // temporal_ratio)
    effective_window = latent_window * temporal_ratio
    effective_stride = latent_stride * temporal_ratio
    overlap_frames = max(0, effective_window - effective_stride)
    latent_frames = int(latents.shape[2])
    expected_frames = min(
        int(target_num_frames),
        (latent_frames - 1) * temporal_ratio + 1,
    )

    original = {
        "use_tiling": getattr(vae, "use_tiling", None),
        "use_framewise_decoding": getattr(vae, "use_framewise_decoding", None),
        "tile_sample_min_height": getattr(vae, "tile_sample_min_height", None),
        "tile_sample_min_width": getattr(vae, "tile_sample_min_width", None),
        "tile_sample_stride_height": getattr(vae, "tile_sample_stride_height", None),
        "tile_sample_stride_width": getattr(vae, "tile_sample_stride_width", None),
    }

    # We own temporal chunking here.  Leave Diffusers' own framewise route off
    # to avoid nested temporal splitting; spatial tiling remains independently
    # available for lower-memory cards.
    if hasattr(vae, "use_framewise_decoding"):
        setattr(vae, "use_framewise_decoding", False)
    spatial_status = _set_vae_spatial_tiling(vae, enabled=bool(spatial_tiling))

    decode_device = getattr(vae, "device", None)
    if decode_device is None:
        try:
            decode_device = next(vae.parameters()).device
        except Exception:
            decode_device = torch_module.device("cuda")

    output_chunks = []
    previous_cpu = None
    emitted_frames = 0
    decoded_chunks = 0
    starts = list(range(0, latent_frames, latent_stride))
    try:
        for chunk_index, latent_start in enumerate(starts, start=1):
            if emitted_frames >= expected_frames:
                break
            latent_tile_cpu = latents[
                :, :, latent_start : latent_start + latent_window + 1, :, :
            ]
            # A final single latent after the first window produces one sample
            # frame which the official algorithm discards for non-first tiles.
            if latent_start > 0 and int(latent_tile_cpu.shape[2]) <= 1:
                break

            latent_tile_cuda = latent_tile_cpu.to(
                device=decode_device, dtype=getattr(vae, "dtype", torch_module.bfloat16)
            )
            if bool(getattr(vae.config, "timestep_conditioning", False)):
                decode_timestep = torch_module.zeros(
                    (latent_tile_cuda.shape[0],),
                    device=latent_tile_cuda.device,
                    dtype=latent_tile_cuda.dtype,
                )
            else:
                decode_timestep = None

            with torch_module.inference_mode():
                decoded_cuda = vae.decode(
                    latent_tile_cuda, decode_timestep, return_dict=False
                )[0]
            if latent_start > 0:
                decoded_cuda = decoded_cuda[:, :, :-1, :, :]
            # Convert during the device transfer.  ``decoded.float().cpu()``
            # first creates a second full-size FP32 tensor on CUDA.  Clone on
            # CPU outside inference_mode so overlap blending may update it.
            current_cpu = decoded_cuda.to(
                device="cpu", dtype=torch_module.float32
            ).clone()

            if previous_cpu is not None and overlap_frames > 0:
                blend_extent = min(
                    int(previous_cpu.shape[2]),
                    int(current_cpu.shape[2]),
                    int(overlap_frames),
                )
                if blend_extent > 0:
                    for frame_index in range(blend_extent):
                        alpha = float(frame_index) / float(blend_extent)
                        current_cpu[:, :, frame_index, :, :] = (
                            previous_cpu[:, :, -blend_extent + frame_index, :, :]
                            * (1.0 - alpha)
                            + current_cpu[:, :, frame_index, :, :] * alpha
                        )

            take = effective_stride + (1 if previous_cpu is None else 0)
            take = min(take, int(current_cpu.shape[2]), expected_frames - emitted_frames)
            if take > 0:
                output_chunks.append(current_cpu[:, :, :take, :, :].contiguous())
                emitted_frames += int(take)
            previous_cpu = current_cpu
            decoded_chunks += 1

            del latent_tile_cuda, decoded_cuda, decode_timestep
            if callable(sample_memory):
                sample_memory()
            print(
                f"[ltx-status] Final VAE decode window {decoded_chunks}: "
                f"{emitted_frames}/{expected_frames} frames",
                flush=True,
            )

        if emitted_frames < expected_frames:
            raise RuntimeError(
                "Temporal VAE decode ended early: "
                f"decoded {emitted_frames}/{expected_frames} frames"
            )
        decoded = torch_module.cat(output_chunks, dim=2)[:, :, :expected_frames]
    finally:
        for name, value in original.items():
            if value is not None and hasattr(vae, name):
                try:
                    setattr(vae, name, value)
                except Exception:
                    pass

    return decoded, {
        "chunks": int(decoded_chunks),
        "window_frames": int(effective_window),
        "stride_frames": int(effective_stride),
        "overlap_frames": int(overlap_frames),
        "spatial_tiling": bool(spatial_tiling),
        "spatial_tiling_status": spatial_status,
        "decoded_frames": int(expected_frames),
    }


def _remove_diffusers_hooks_recursive(module: Any) -> Tuple[int, Tuple[str, ...]]:
    """Remove Diffusers HookRegistry hooks from a module and all children.

    ``DiffusionPipeline.remove_all_hooks()`` primarily handles pipeline-level
    and Accelerate hooks. Group offloading registers HookRegistry entries on
    leaf modules, so those must be detached explicitly before a normal ``.to``
    and full VAE decode.
    """

    if module is None or not hasattr(module, "named_modules"):
        return 0, ()

    removed = []
    # Snapshot the module list first because removing a hook rewrites forward
    # methods and may mutate registry internals.
    for module_name, submodule in list(module.named_modules()):
        registry = getattr(submodule, "_diffusers_hook", None)
        if registry is None:
            continue
        hook_order = list(
            getattr(registry, "_hook_order", list(getattr(registry, "hooks", {}).keys()))
        )
        for hook_name in reversed(hook_order):
            registry.remove_hook(hook_name, recurse=False)
            removed.append(f"{module_name or '<root>'}:{hook_name}")
        if not getattr(registry, "hooks", {}):
            try:
                delattr(submodule, "_diffusers_hook")
            except Exception:
                pass
    return len(removed), tuple(removed)


def _remove_pipeline_diffusers_hooks(pipe: Any) -> Tuple[int, Tuple[str, ...]]:
    total_removed = 0
    samples = []
    sample_limit = 32
    for component_name in (
        "transformer",
        "text_encoder",
        "connectors",
        "vae",
        "audio_vae",
        "vocoder",
    ):
        component = getattr(pipe, component_name, None)
        count, names = _remove_diffusers_hooks_recursive(component)
        total_removed += count
        remaining = sample_limit - len(samples)
        if remaining > 0:
            samples.extend(f"{component_name}.{name}" for name in names[:remaining])
    if total_removed > len(samples):
        samples.append(f"... +{total_removed - len(samples)} more")
    return total_removed, tuple(samples)


def _load_ltx23_spatial_upsampler(
    checkpoint_path: Path,
    *,
    model_cls: Any,
    torch_module: Any,
) -> Tuple[Any, float, int]:
    """Load the official raw LTX 2.3 x2 spatial upsampler as a Diffusers model."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LTX 2.3 spatial upsampler not found: {path}")

    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"safetensors is required to load the LTX 2.3 spatial upsampler: {type(exc).__name__}: {exc}"
        ) from exc

    config = {
        "in_channels": 128,
        "mid_channels": 1024,
        "num_blocks_per_stage": 4,
        "dims": 3,
        "spatial_upsample": True,
        "temporal_upsample": False,
        "rational_spatial_scale": 2.0,
        "use_rational_resampler": False,
    }

    load_t0 = time.perf_counter()
    state_dict = load_file(str(path), device="cpu")
    try:
        try:
            from accelerate import init_empty_weights  # type: ignore

            with init_empty_weights():
                model = model_cls(**config)
            model.load_state_dict(state_dict, strict=True, assign=True)
        except ImportError:
            model = model_cls(**config)
            model.load_state_dict(state_dict, strict=True)
        except TypeError:
            # Older torch builds may not expose ``assign``. This is less RAM
            # efficient, but keeps the loader usable instead of silently
            # replacing or ignoring weights.
            model = model_cls(**config)
            model.load_state_dict(state_dict, strict=True)
    finally:
        del state_dict

    model.to(device="cpu", dtype=torch_module.bfloat16)
    model.eval()
    return model, time.perf_counter() - load_t0, int(path.stat().st_size)



def _install_i2v_conditioning_mask_guard(pipe: Any, ctx: Dict[str, Any]) -> None:
    """Keep the Diffusers 0.39 two-stage I2V conditioning mask with its latents.

    When Stage 2 receives CPU latents, Diffusers creates the conditioning mask on
    CPU and only moves the latent tensor to CUDA. The first denoise operation then
    mixes a CUDA timestep with a CPU mask. The guard is instance-local and exists
    only in this isolated INT4 CLI; no installed package or native LTX file is
    edited.
    """
    if pipe.__class__.__name__ not in {"LTX2ImageToVideoPipeline", "LTX2ConditionPipeline"}:
        ctx["ltx_int4_conditioning_mask_guard"] = "not needed: text-to-video"
        ctx["ltx_int4_conditioning_mask_rewrites"] = "0"
        return

    original = getattr(pipe, "prepare_latents", None)
    if not callable(original):
        ctx["ltx_int4_conditioning_mask_guard"] = "unavailable: prepare_latents not found"
        ctx["ltx_int4_conditioning_mask_rewrites"] = "0"
        return

    state = {"rewrites": 0, "last": "none"}

    def guarded(_self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            return result
        latents, conditioning_mask, *rest = result
        try:
            latent_device = getattr(latents, "device", None)
            mask_device = getattr(conditioning_mask, "device", None)
            if latent_device is not None and mask_device is not None and latent_device != mask_device:
                before = str(mask_device)
                conditioning_mask = conditioning_mask.to(device=latent_device)
                state["rewrites"] += 1
                state["last"] = f"{before} -> {latent_device}; dtype={getattr(conditioning_mask, 'dtype', 'unknown')}"
                ctx["ltx_int4_conditioning_mask_rewrites"] = str(state["rewrites"])
                ctx["ltx_int4_conditioning_mask_last_move"] = state["last"]
                print(
                    "[ltx-status] Aligned INT4 I2V conditioning mask with video latents: "
                    + state["last"],
                    flush=True,
                )
        except Exception as exc:
            ctx["ltx_int4_conditioning_mask_guard_error"] = f"{type(exc).__name__}: {exc}"
            raise
        return (latents, conditioning_mask, *rest)

    pipe.prepare_latents = types.MethodType(guarded, pipe)
    ctx["ltx_int4_conditioning_mask_guard"] = "installed: instance-local Diffusers I2V alignment"
    ctx["ltx_int4_conditioning_mask_rewrites"] = "0"
    ctx["ltx_int4_conditioning_mask_last_move"] = "none"


def _add_ltx_source_paths(ltx_root: Path, ctx: Dict[str, Any]) -> None:
    """Expose the installed official LTX audio preprocessing helpers.

    This only adds import paths for the isolated INT4 process.  It does not
    import, call, or modify FrameVision's native VRAM Lab CLI.
    """
    root = Path(ltx_root).expanduser().resolve()
    repo = root / "models" / "ltx23" / "repos" / "LTX-2" / "packages"
    candidates = (
        repo / "ltx-core" / "src",
        repo / "ltx-pipelines" / "src",
    )
    missing = [str(path) for path in candidates if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "The isolated INT4 audio route needs the installed official LTX source helpers. "
            "Missing: " + ", ".join(missing)
        )
    added = []
    for path in candidates:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            added.append(value)
    ctx["ltx_int4_audio_helper_paths"] = " | ".join(added or (str(path) for path in candidates))


def _extract_audio_latents(
    *,
    args: Any,
    pipe: Any,
    torch_module: Any,
    ctx: Dict[str, Any],
    num_frames: int,
    frame_rate: float,
    output_num_frames: Optional[int] = None,
) -> Tuple[Any, Any, Optional[int]]:
    """Decode the selected soundtrack and encode it with the split model audio VAE.

    Returns unnormalised Diffusers audio latents plus the original decoded
    waveform.  The waveform is kept for the final mux, while the latent is
    frozen in both denoise stages so audio-to-video cross attention can drive
    motion without rewriting the supplied soundtrack.
    """
    audio_text = str(getattr(args, "audio_path", "") or "").strip()
    if not audio_text:
        ctx["ltx_int4_reference_audio"] = "disabled"
        return None, None, None

    audio_path = Path(audio_text).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"INT4 reference audio not found: {audio_path}")
    if str(getattr(args, "pipeline", "")) != "two_stages":
        raise RuntimeError("INT4 reference-audio generation requires the normal two_stages workflow.")

    _add_ltx_source_paths(Path(str(getattr(args, "ltx_root", Path.cwd()))), ctx)
    try:
        from ltx_core.model.audio_vae import AudioProcessor  # type: ignore
        from ltx_core.types import Audio  # type: ignore
        from ltx_pipelines.utils.media_io import decode_audio_from_file  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official LTX audio decoder/processor from the installed LTX-2 repo: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Keep the user-visible waveform at the exact requested output duration,
    # while the encoded/padded conditioning latent below follows the aligned
    # 8n+1 model length used by the transformer.
    waveform_frames = int(output_num_frames) if output_num_frames is not None else int(num_frames)
    duration = float(waveform_frames) / float(frame_rate)
    requested_max = getattr(args, "audio_max_duration", None)
    if requested_max is not None and float(requested_max) > 0:
        duration = min(duration, float(requested_max))
    start_time = max(0.0, float(getattr(args, "audio_start_time", 0.0) or 0.0))

    device = torch_module.device("cuda")
    print(
        f"[ltx-status] Decoding INT4 reference audio: {audio_path} "
        f"(start={start_time:.3f}s, duration={duration:.3f}s)",
        flush=True,
    )
    decoded_audio = decode_audio_from_file(
        str(audio_path),
        device,
        start_time,
        duration,
    )
    if decoded_audio is None:
        raise RuntimeError(f"Failed to decode INT4 reference audio: {audio_path}")

    waveform = decoded_audio.waveform
    while getattr(waveform, "ndim", 0) > 2 and int(waveform.shape[0]) == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if int(waveform.shape[0]) == 1:
        waveform = waveform.repeat(2, 1)
    elif int(waveform.shape[0]) > 2:
        waveform = waveform[:2]
    sample_rate = int(decoded_audio.sampling_rate)
    target_samples = max(1, int(round(duration * sample_rate)))
    if int(waveform.shape[-1]) > target_samples:
        waveform = waveform[..., :target_samples]
    elif int(waveform.shape[-1]) < target_samples:
        waveform = torch_module.nn.functional.pad(
            waveform,
            (0, target_samples - int(waveform.shape[-1])),
        )
    waveform = waveform.contiguous().clamp(-1.0, 1.0)

    audio_vae = getattr(pipe, "audio_vae", None)
    if audio_vae is None:
        raise RuntimeError("The INT4 split pipeline has no audio_vae component.")
    config = audio_vae.config
    processor = AudioProcessor(
        target_sample_rate=int(getattr(config, "sample_rate", sample_rate)),
        mel_bins=int(getattr(config, "mel_bins", 64)),
        mel_hop_length=int(getattr(config, "mel_hop_length", 160)),
        n_fft=int(getattr(config, "n_fft", 1024)),
    ).to(device)

    encode_t0 = time.perf_counter()
    try:
        # Official LTX uses float32 for audio VAE encoding quality.
        audio_vae.to(device=device, dtype=torch_module.float32)
        audio_for_model = Audio(
            waveform=waveform.unsqueeze(0).to(device=device, dtype=torch_module.float32),
            sampling_rate=sample_rate,
        )
        mel = processor.waveform_to_mel(audio_for_model).to(
            device=device,
            dtype=torch_module.float32,
        )
        with torch_module.inference_mode():
            encoded = audio_vae.encode(mel, return_dict=True)
        posterior = getattr(encoded, "latent_dist", None)
        if posterior is not None and callable(getattr(posterior, "mode", None)):
            audio_latents = posterior.mode()
        elif posterior is not None and callable(getattr(posterior, "sample", None)):
            audio_latents = posterior.sample()
        elif isinstance(encoded, (tuple, list)) and encoded:
            audio_latents = encoded[0]
        else:
            raise RuntimeError("Diffusers audio_vae.encode returned no usable latent tensor.")

        expected_frames = round(
            (float(num_frames) / float(frame_rate))
            * (
                float(getattr(pipe, "audio_sampling_rate", sample_rate))
                / float(getattr(pipe, "audio_hop_length", 160))
                / float(getattr(pipe, "audio_vae_temporal_compression_ratio", 4))
            )
        )
        expected_frames = max(1, int(expected_frames))
        current_frames = int(audio_latents.shape[2])
        if current_frames > expected_frames:
            audio_latents = audio_latents[:, :, :expected_frames, :]
        elif current_frames < expected_frames:
            audio_latents = torch_module.nn.functional.pad(
                audio_latents,
                (0, 0, 0, expected_frames - current_frames),
            )
        audio_latents = audio_latents.detach().to(device="cpu", dtype=torch_module.float32).contiguous()
    finally:
        try:
            audio_vae.to(device="cpu", dtype=torch_module.bfloat16)
        except Exception:
            try:
                audio_vae.to("cpu")
            except Exception:
                pass
        try:
            processor.to("cpu")
        except Exception:
            pass
        gc.collect()
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass

    ctx["ltx_int4_reference_audio"] = str(audio_path)
    ctx["ltx_int4_reference_audio_start_s"] = f"{start_time:.3f}"
    ctx["ltx_int4_reference_audio_duration_s"] = f"{duration:.3f}"
    ctx["ltx_int4_reference_audio_sample_rate"] = str(sample_rate)
    ctx["ltx_int4_reference_audio_waveform_shape"] = str(tuple(int(v) for v in waveform.shape))
    ctx["ltx_int4_reference_audio_latent_shape"] = str(tuple(int(v) for v in audio_latents.shape))
    ctx["ltx_int4_reference_audio_encode_time_s"] = f"{time.perf_counter() - encode_t0:.3f}"
    ctx["ltx_int4_reference_audio_mode"] = (
        "normal two_stages; audio VAE latent frozen in Stage 1 and Stage 2; original waveform muxed"
    )
    print(
        "[ltx-status] INT4 reference audio encoded and prepared as a frozen two-stage condition: "
        f"latent={tuple(audio_latents.shape)}",
        flush=True,
    )
    return audio_latents, waveform.detach().to("cpu", dtype=torch_module.float32), sample_rate


@contextlib.contextmanager
def _freeze_diffusers_audio_scheduler(pipe: Any, enabled: bool, ctx: Dict[str, Any]):
    """Freeze only the supplied reference-audio latent inside Diffusers.

    Diffusers uses one ``noise_scale`` argument for both the video and audio
    latent inputs, then creates a private scheduler copy for the audio branch.
    Stage 2 needs non-zero video refinement noise, while official A2V keeps the
    uploaded audio latent clean and frozen.  This instance-local guard therefore
    forces zero noise only in ``prepare_audio_latents`` and makes only the copied
    audio scheduler return its input unchanged.  The video scheduler and video
    refinement noise remain untouched.
    """
    if not enabled:
        yield
        return

    module = importlib.import_module(pipe.__class__.__module__)
    copy_module = getattr(module, "copy", None)
    scheduler_cls = type(pipe.scheduler)
    original_deepcopy = getattr(copy_module, "deepcopy", None)
    original_step = getattr(scheduler_cls, "step", None)
    original_prepare_audio = getattr(pipe, "prepare_audio_latents", None)
    if (
        not callable(original_deepcopy)
        or not callable(original_step)
        or not callable(original_prepare_audio)
    ):
        raise RuntimeError("Could not install the isolated INT4 frozen-audio guards.")

    state = {"copies": 0, "steps": 0, "prepare_calls": 0, "zero_noise_rewrites": 0}

    def marked_deepcopy(obj: Any, *deep_args: Any, **deep_kwargs: Any) -> Any:
        result = original_deepcopy(obj, *deep_args, **deep_kwargs)
        if obj is pipe.scheduler:
            setattr(result, "_framevision_frozen_audio_scheduler", True)
            state["copies"] += 1
        return result

    def guarded_step(self: Any, model_output: Any, timestep: Any, sample: Any, *step_args: Any, **step_kwargs: Any) -> Any:
        if bool(getattr(self, "_framevision_frozen_audio_scheduler", False)):
            state["steps"] += 1
            if step_kwargs.get("return_dict", True):
                return types.SimpleNamespace(prev_sample=sample)
            return (sample,)
        return original_step(self, model_output, timestep, sample, *step_args, **step_kwargs)

    def guarded_prepare_audio(*prepare_args: Any, **prepare_kwargs: Any) -> Any:
        state["prepare_calls"] += 1
        supplied_latents = prepare_kwargs.get("latents")
        if supplied_latents is not None:
            previous_noise = prepare_kwargs.get("noise_scale", 0.0)
            try:
                nonzero = float(previous_noise) != 0.0
            except Exception:
                nonzero = True
            prepare_kwargs["noise_scale"] = 0.0
            if nonzero:
                state["zero_noise_rewrites"] += 1
        return original_prepare_audio(*prepare_args, **prepare_kwargs)

    copy_module.deepcopy = marked_deepcopy
    scheduler_cls.step = guarded_step
    pipe.prepare_audio_latents = guarded_prepare_audio
    try:
        yield
    finally:
        copy_module.deepcopy = original_deepcopy
        scheduler_cls.step = original_step
        pipe.prepare_audio_latents = original_prepare_audio
        ctx["ltx_int4_frozen_audio_scheduler_copies"] = str(
            int(ctx.get("ltx_int4_frozen_audio_scheduler_copies", "0")) + state["copies"]
        )
        ctx["ltx_int4_frozen_audio_scheduler_steps"] = str(
            int(ctx.get("ltx_int4_frozen_audio_scheduler_steps", "0")) + state["steps"]
        )
        ctx["ltx_int4_frozen_audio_prepare_calls"] = str(
            int(ctx.get("ltx_int4_frozen_audio_prepare_calls", "0")) + state["prepare_calls"]
        )
        ctx["ltx_int4_frozen_audio_zero_noise_rewrites"] = str(
            int(ctx.get("ltx_int4_frozen_audio_zero_noise_rewrites", "0"))
            + state["zero_noise_rewrites"]
        )
        ctx["ltx_int4_frozen_audio_scheduler_guard"] = (
            "installed and restored per pipeline call; audio noise forced to 0; "
            "video scheduler/noise unchanged"
        )


def _call_ltx2_pipe(pipe: Any, kwargs: Dict[str, Any], *, freeze_audio: bool, ctx: Dict[str, Any]) -> Any:
    with _freeze_diffusers_audio_scheduler(pipe, freeze_audio, ctx):
        return pipe(**kwargs)


def _lora_requests(args: Any) -> List[Tuple[Path, float]]:
    requests: List[Tuple[Path, float]] = []
    for raw in list(getattr(args, "loras", []) or []):
        try:
            path_text, strength_text = raw
            path = Path(str(path_text)).expanduser().resolve()
            strength = float(strength_text)
        except Exception as exc:
            raise RuntimeError(f"Invalid --lora entry {raw!r}: {type(exc).__name__}: {exc}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"LTX INT4 LoRA not found: {path}")
        if not (-4.0 <= strength <= 4.0):
            raise RuntimeError(f"LTX INT4 LoRA strength must be between -4 and 4: {strength}")
        requests.append((path, strength))
    return requests


def _load_requested_loras(pipe: Any, args: Any, ctx: Dict[str, Any]) -> None:
    requests = _lora_requests(args)
    if not requests:
        ctx["ltx_int4_loras"] = "none"
        return
    loader = getattr(pipe, "load_lora_weights", None)
    setter = getattr(pipe, "set_adapters", None)
    if not callable(loader) or not callable(setter):
        raise RuntimeError(
            f"{pipe.__class__.__name__} does not expose Diffusers LoRA loading. "
            "Repair the .ltx23 Diffusers/PEFT installation."
        )
    names: List[str] = []
    weights: List[float] = []
    loaded: List[str] = []
    for index, (path, strength) in enumerate(requests, start=1):
        adapter_name = f"framevision_ltx23_{index}_{path.stem[:32]}"
        try:
            loader(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter_name,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            # Keep compatibility with Diffusers builds that do not forward one
            # of the optional loader flags. The local directory + weight name
            # still prevents any network access and supports single-file LoRAs.
            loader(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter_name,
            )
        names.append(adapter_name)
        weights.append(float(strength))
        loaded.append(f"{path.name}@{strength:g}")
        print(f"[ltx-status] Loaded INT4 LoRA: {path.name} @ {strength:g}", flush=True)
    setter(names, adapter_weights=weights)
    ctx["ltx_int4_loras"] = "; ".join(loaded)
    ctx["ltx_int4_lora_adapter_names"] = "; ".join(names)


def _frame_to_latent_index(frame: int, num_frames: int, temporal_ratio: int) -> int:
    frame = int(frame)
    if frame < 0 or frame >= max(1, int(num_frames)) - 1:
        return -1
    if frame <= 0:
        return 0
    return ((frame - 1) // max(1, int(temporal_ratio))) + 1


def run_sdnq_diffusers(args: Any, ctx: Dict[str, Any], torch_module: Any) -> Dict[str, Any]:
    info = validate_sdnq_model_root(Path(str(args.sdnq_model_root)))
    root: Path = info["root"]
    workflow = str(getattr(args, "pipeline", "one_stage") or "one_stage").strip()

    if workflow == "two_stages_hq":
        raise RuntimeError(
            "SDNQ two_stages_hq is not connected yet. That workflow uses the official res_2s sampler, "
            "while the current SDNQ bridge implements the proven Euler two_stages path. Select two_stages."
        )
    if workflow not in {"one_stage", "two_stages"}:
        raise RuntimeError(f"Unsupported SDNQ workflow: {workflow!r}")

    target_width = int(getattr(args, "width", 768))
    target_height = int(getattr(args, "height", 512))
    requested_num_frames = max(1, int(getattr(args, "num_frames", 121)))
    frame_rate = float(getattr(args, "frame_rate", 24))

    # LTX-2's video VAE represents F output frames as
    # ``(latent_frames - 1) * 8 + 1``. Passing an arbitrary count such as 228
    # directly therefore creates only 225 decodable frames while other parts
    # of the pipeline still receive 228. Besides shortening the clip, those
    # mismatched temporal shapes produced a repeatable Stage 2 allocator spike
    # on the Music Creator route. Generate the next valid 8n+1 model length and
    # trim the streamed VAE output back to the exact requested frame count.
    vae_scale_factors = info.get("transformer_config", {}).get("vae_scale_factors", (8, 32, 32))
    try:
        temporal_ratio = max(1, int(vae_scale_factors[0]))
    except Exception:
        temporal_ratio = 8
    model_num_frames = (
        ((requested_num_frames - 1 + temporal_ratio - 1) // temporal_ratio)
        * temporal_ratio
        + 1
    )
    num_frames = int(model_num_frames)
    ctx["ltx_sdnq_requested_num_frames"] = str(requested_num_frames)
    ctx["ltx_sdnq_model_num_frames"] = str(model_num_frames)
    if model_num_frames == requested_num_frames:
        ctx["ltx_sdnq_frame_alignment"] = "already valid 8n+1 model length"
    else:
        ctx["ltx_sdnq_frame_alignment"] = (
            f"model generation padded {requested_num_frames}->{model_num_frames} frames; "
            f"final decode trimmed back to {requested_num_frames}"
        )
        print(
            "[ltx-status] Aligning requested frame count for LTX temporal VAE: "
            f"{requested_num_frames} -> {model_num_frames}; output will be trimmed back",
            flush=True,
        )

    if workflow == "two_stages":
        if target_width % 64 != 0 or target_height % 64 != 0:
            raise RuntimeError(
                "SDNQ two_stages needs final width and height divisible by 64 so Stage 1 remains divisible "
                f"by 32 after the x2 split. Received {target_width}x{target_height}."
            )
        stage1_width = target_width // 2
        stage1_height = target_height // 2
    else:
        stage1_width = target_width
        stage1_height = target_height

    ctx["ltx_sdnq_requested"] = "YES"
    ctx["ltx_sdnq_workflow"] = workflow
    ctx["ltx_sdnq_model_root"] = str(root)
    ctx["ltx_sdnq_transformer_shards"] = str(len(info["transformer_shards"]))
    ctx["ltx_sdnq_transformer_size_gb"] = f"{info['transformer_bytes'] / (1024**3):.2f}"
    ctx["ltx_sdnq_text_encoder_shards"] = str(len(info["text_encoder_shards"]))
    ctx["ltx_sdnq_text_encoder_size_gb"] = f"{info['text_encoder_bytes'] / (1024**3):.2f}"
    ctx["ltx_sdnq_dtype_summary"] = ", ".join(
        f"{key}:{value}" for key, value in sorted(info["dtype_counts"].items())
    ) or "quantization_config present"
    ctx["ltx_sdnq_load_status"] = "valid split-folder model; runtime import pending"
    ctx["ltx_sdnq_stage1_resolution"] = f"{stage1_width}x{stage1_height}"
    ctx["ltx_sdnq_stage1_steps"] = str(int(getattr(args, "num_inference_steps", 8) or 8))
    ctx["ltx_sdnq_stage2_resolution"] = (
        f"{target_width}x{target_height}" if workflow == "two_stages" else "n/a"
    )
    ctx["ltx_sdnq_stage2_steps"] = "3" if workflow == "two_stages" else "n/a"
    ctx["ltx_sdnq_stage2_generator_device"] = "cpu" if workflow == "two_stages" else "n/a"

    cache_root = _prepare_sdnq_cache_env(args, root)
    ctx["ltx_sdnq_compile_cache"] = str(cache_root)
    print(f"[ltx-status] Verifying Triton/Inductor acceleration", flush=True)
    triton_probe = _probe_triton_inductor(torch_module)
    ctx["ltx_sdnq_triton_version"] = str(triton_probe["triton_version"])
    ctx["ltx_sdnq_triton_distribution"] = str(triton_probe["triton_distribution"])
    ctx["ltx_sdnq_triton_probe"] = str(triton_probe["detail"])
    ctx["ltx_sdnq_triton_probe_time_s"] = f"{float(triton_probe['compile_time_s']):.3f}"
    allow_eager = str(os.environ.get("FRAMEVISION_LTX_SDNQ_ALLOW_EAGER", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if not bool(triton_probe["ok"]) and not allow_eager:
        raise RuntimeError(
            "SDNQ acceleration is unavailable, so FrameVision stopped before loading the model instead of silently "
            "running the very slow PyTorch Eager fallback. Repair the LTX 2.3 optional install so Torch 2.8 uses "
            "triton-windows==3.4.0.post21. Probe result: " + str(triton_probe["detail"])
        )
    if bool(triton_probe["ok"]):
        os.environ["SDNQ_USE_TORCH_COMPILE"] = "1"

    print(f"[ltx-status] Loading SDNQ split model from {root}", flush=True)
    try:
        import sdnq  # type: ignore
        from sdnq import SDNQConfig  # noqa: F401  # type: ignore
        from sdnq.common import use_torch_compile as sdnq_compile_active  # type: ignore
        from sdnq.loader import apply_sdnq_options_to_model  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "SDNQ is not installed in the selected .ltx23 environment. Run:\n"
            f'  "{getattr(args, "_python_executable", "python")}" -m pip install -U sdnq\n'
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        import diffusers  # type: ignore
        from diffusers import (  # type: ignore
            FlowMatchEulerDiscreteScheduler,
            LTX2ConditionPipeline,
            LTX2ImageToVideoPipeline,
            LTX2Pipeline,
        )
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline  # type: ignore
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition  # type: ignore
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel  # type: ignore
        from diffusers.pipelines.ltx2.utils import (  # type: ignore
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )
        from diffusers.utils import load_image  # type: ignore
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video  # type: ignore
        except Exception:
            from diffusers.utils import encode_video  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The selected .ltx23 environment needs Diffusers with the LTX 2.3 base, condition, and "
            "latent-upsample pipelines. Repair the LTX 2.3 optional install before testing SDNQ generation. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    ctx["ltx_sdnq_package_version"] = _version_of(sdnq)
    ctx["ltx_sdnq_diffusers_version"] = _version_of(diffusers)
    ctx["ltx_sdnq_compiled_backend"] = "ACTIVE" if bool(sdnq_compile_active) else "EAGER"
    if not bool(sdnq_compile_active) and not allow_eager:
        raise RuntimeError(
            "Triton compiled successfully, but SDNQ still selected PyTorch Eager mode. "
            "Re-run the corrected LTX installer before generating."
        )
    ctx["ltx_sdnq_cuda_before_load"] = _cuda_text(torch_module)

    condition_specs = list(getattr(args, "_planner_conditions", []) or [])
    has_conditions = bool(condition_specs)
    # Keep the proven base T2V pipeline when there is no visual input. Any
    # visual input uses Diffusers' general LTX2ConditionPipeline, which supports
    # start, middle and final frame conditions in one route.
    pipeline_cls = LTX2ConditionPipeline if has_conditions else LTX2Pipeline
    ctx["ltx_sdnq_pipeline_class"] = pipeline_cls.__name__

    load_t0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(
        str(root),
        torch_dtype=torch_module.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    load_s = time.perf_counter() - load_t0
    ctx["ltx_sdnq_load_time_s"] = f"{load_s:.3f}"
    ctx["ltx_sdnq_load_status"] = f"loaded split pipeline successfully in {load_s:.3f}s"
    ctx["ltx_sdnq_cuda_after_load"] = _cuda_text(torch_module)
    print(f"[ltx-status] SDNQ model loaded in {load_s:.2f}s", flush=True)
    _install_i2v_conditioning_mask_guard(pipe, ctx)
    _load_requested_loras(pipe, args, ctx)

    quantized_matmul_status = []
    for component_name in ("transformer", "text_encoder"):
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        try:
            apply_sdnq_options_to_model(component, use_quantized_matmul=True)
            quantized_matmul_status.append(f"{component_name}=enabled")
        except Exception as exc:
            quantized_matmul_status.append(f"{component_name}=unchanged ({type(exc).__name__}: {exc})")
    ctx["ltx_sdnq_quantized_matmul"] = "; ".join(quantized_matmul_status) or "not found"
    print(f"[ltx-status] SDNQ quantized matmul: {ctx['ltx_sdnq_quantized_matmul']}", flush=True)

    _configure_attention_backend(
        pipe,
        str(getattr(args, "attention_backend", "auto") or "auto"),
        ctx,
    )

    try:
        pipe.vae.enable_tiling()
        ctx["ltx_sdnq_vae_tiling"] = "enabled"
    except Exception as exc:
        ctx["ltx_sdnq_vae_tiling"] = f"unavailable: {type(exc).__name__}: {exc}"

    reference_audio_latents, reference_audio_waveform, reference_audio_rate = _extract_audio_latents(
        args=args,
        pipe=pipe,
        torch_module=torch_module,
        ctx=ctx,
        num_frames=num_frames,
        frame_rate=frame_rate,
        output_num_frames=requested_num_frames,
    )
    freeze_reference_audio = reference_audio_latents is not None

    extra = list(getattr(args, "extra", []) or [])
    raw_weight_dtype = str(
        info["quantization_config"].get("weights_dtype")
        or info["quantization_config"].get("weight_dtype")
        or "unknown"
    )
    dtype_probe = " ".join([raw_weight_dtype, *[str(key) for key in info.get("dtype_counts", {}).keys()]]).lower()
    if "int4" not in dtype_probe and "uint4" not in dtype_probe:
        raise RuntimeError(
            "ltx_int4_cli.py only accepts the INT4 split model. "
            f"Detected quantization description: {dtype_probe or 'unknown'}"
        )
    ctx["ltx_int4_model_validation"] = "PASS: INT4 split model detected"
    block_count = int(info["transformer_config"].get("num_layers", 48) or 48)
    group_override = _int_option(
        extra,
        ("--sdnq-group-blocks", "--sdnq_group_blocks"),
        int(os.environ.get("FRAMEVISION_LTX_SDNQ_GROUP_BLOCKS", "0") or 0),
    )
    stage2_group_override = _int_option(
        extra,
        ("--sdnq-stage2-group-blocks", "--sdnq_stage2_group_blocks"),
        int(os.environ.get("FRAMEVISION_LTX_SDNQ_STAGE2_GROUP_BLOCKS", "0") or 0),
    )
    offload_override = str(
        _extra_option(
            extra,
            ("--sdnq-offload-mode", "--sdnq_offload_mode"),
            os.environ.get("FRAMEVISION_LTX_SDNQ_OFFLOAD_MODE", "auto"),
        )
        or "auto"
    )
    int4_24_stage2_full_max_work = _float_option(
        extra,
        ("--sdnq-int4-stage2-full-max-work", "--sdnq_int4_stage2_full_max_work"),
        float(os.environ.get("FRAMEVISION_LTX_SDNQ_INT4_STAGE2_FULL_MAX_WORK", "4.40") or 4.40),
    )
    int4_24_i2v_stage1_full_max_work = _float_option(
        extra,
        (
            "--sdnq-int4-i2v-stage1-full-max-work",
            "--sdnq_int4_i2v_stage1_full_max_work",
        ),
        float(
            os.environ.get(
                "FRAMEVISION_LTX_SDNQ_INT4_I2V_STAGE1_FULL_MAX_WORK", "0.45"
            )
            or 0.45
        ),
    )
    int4_24_i2v_stage2_full_max_work = _float_option(
        extra,
        (
            "--sdnq-int4-i2v-stage2-full-max-work",
            "--sdnq_int4_i2v_stage2_full_max_work",
        ),
        float(
            os.environ.get(
                "FRAMEVISION_LTX_SDNQ_INT4_I2V_STAGE2_FULL_MAX_WORK", "1.25"
            )
            or 1.25
        ),
    )
    profile_gb = int(str(getattr(args, "vram_profile", "24") or "24"))
    auto_vram_enabled = bool(getattr(args, "int4_auto_vram", True))
    ctx["ltx_int4_vram_automation"] = (
        f"enabled: detected/selected {profile_gb}GB profile with resolution/frame-aware stage planning"
        if auto_vram_enabled
        else f"disabled: fixed {profile_gb}GB per-card policy"
    )

    stage1_policy = _select_sdnq_offload_policy(
        weight_dtype=raw_weight_dtype,
        profile_gb=profile_gb,
        width=stage1_width,
        height=stage1_height,
        num_frames=num_frames,
        block_count=block_count,
        override_group_blocks=group_override,
        override_mode=offload_override,
        workflow=workflow,
        stage="stage1",
        input_mode="image-conditioned-video" if has_conditions else "text-to-video",
        automation_enabled=auto_vram_enabled,
        int4_24_stage2_full_max_work=int4_24_stage2_full_max_work,
        int4_24_i2v_stage1_full_max_work=int4_24_i2v_stage1_full_max_work,
        int4_24_i2v_stage2_full_max_work=int4_24_i2v_stage2_full_max_work,
    )
    stage2_policy = _select_sdnq_offload_policy(
        weight_dtype=raw_weight_dtype,
        profile_gb=profile_gb,
        width=target_width,
        height=target_height,
        num_frames=num_frames,
        block_count=block_count,
        override_group_blocks=stage2_group_override or group_override,
        override_mode=offload_override,
        workflow=workflow,
        stage="stage2",
        input_mode="image-conditioned-video" if has_conditions else "text-to-video",
        automation_enabled=auto_vram_enabled,
        int4_24_stage2_full_max_work=int4_24_stage2_full_max_work,
        int4_24_i2v_stage1_full_max_work=int4_24_i2v_stage1_full_max_work,
        int4_24_i2v_stage2_full_max_work=int4_24_i2v_stage2_full_max_work,
    )
    if workflow != "two_stages":
        stage2_policy = dict(stage1_policy)

    # Every grouped Stage 2 route can accumulate inactive CUDA allocator
    # blocks between its three refinement steps.  The residency planner above
    # now leaves additional live-tensor headroom; this cache guard complements
    # it by returning stale allocator blocks after steps 1 and 2.  Live tensors
    # are untouched and the proven short full-transformer fast path is unchanged.
    stage2_allocator_trim = bool(
        workflow == "two_stages"
        and str(stage2_policy.get("mode", ""))
        in {"group_cpu_offload", "sequential_cpu_offload"}
    )
    ctx["ltx_sdnq_stage2_allocator_guard"] = (
        "enabled: collect Python garbage and release inactive CUDA allocator "
        "cache after grouped refinement steps 1-2"
        if stage2_allocator_trim
        else "not needed: Stage 2 uses the full-transformer fast path"
    )
    ctx["ltx_sdnq_stage2_conservative_policy"] = (
        f"profile={stage2_policy['profile_bucket']}GB; "
        f"group={stage2_policy['group_blocks']}/{stage2_policy['block_count']}; "
        f"mode={stage2_policy['mode']}; work={stage2_policy['work_factor']:.3f}"
    )
    ctx["ltx_sdnq_stage2_allocator_trim_count"] = "0"

    def _estimated_group_gb(selected_policy: Dict[str, Any]) -> float:
        return (
            float(info["transformer_bytes"])
            * float(min(selected_policy["group_blocks"], selected_policy["block_count"]))
            / float(max(1, selected_policy["block_count"]))
            / float(1024 ** 3)
        )

    stage1_group_gb = _estimated_group_gb(stage1_policy)
    stage2_group_gb = _estimated_group_gb(stage2_policy)
    ctx["ltx_sdnq_weight_dtype"] = stage1_policy["variant"]
    ctx["ltx_sdnq_transformer_blocks"] = str(stage1_policy["block_count"])
    ctx["ltx_sdnq_group_blocks"] = str(stage1_policy["group_blocks"])
    ctx["ltx_sdnq_group_estimated_gb"] = f"{stage1_group_gb:.2f}"
    ctx["ltx_sdnq_work_factor"] = f"{stage2_policy['work_factor']:.3f}"
    ctx["ltx_sdnq_stage2_group_blocks"] = str(stage2_policy["group_blocks"])
    ctx["ltx_sdnq_stage2_group_estimated_gb"] = f"{stage2_group_gb:.2f}"
    ctx["ltx_sdnq_memory_policy_input_mode"] = (
        "image-conditioned-video" if has_conditions else "text-to-video"
    )
    ctx["ltx_sdnq_stage1_policy_tier"] = str(stage1_policy["policy_tier"])
    ctx["ltx_sdnq_stage2_policy_tier"] = str(stage2_policy["policy_tier"])
    ctx["ltx_sdnq_stage1_equivalent_704_frames"] = f"{stage1_policy['equivalent_704_frames']:.1f}"
    ctx["ltx_sdnq_stage2_equivalent_704_frames"] = f"{stage2_policy['equivalent_704_frames']:.1f}"
    ctx["ltx_sdnq_i2v_fast_path_limits"] = (
        f"24GB Stage1<={int4_24_i2v_stage1_full_max_work:.3f}; "
        f"Stage2<={int4_24_i2v_stage2_full_max_work:.3f}"
    )
    ctx["ltx_sdnq_stage1_offload_policy"] = (
        f"{stage1_policy['mode']}; group={stage1_policy['group_blocks']}/"
        f"{stage1_policy['block_count']}; work={stage1_policy['work_factor']:.3f}; "
        f"{stage1_policy['reason']}"
    )
    ctx["ltx_sdnq_stage2_offload_policy"] = (
        f"{stage2_policy['mode']}; group={stage2_policy['group_blocks']}/"
        f"{stage2_policy['block_count']}; work={stage2_policy['work_factor']:.3f}; "
        f"{stage2_policy['reason']}"
    )
    ctx["ltx_sdnq_offload_policy"] = (
        f"Stage1={ctx['ltx_sdnq_stage1_offload_policy']} | "
        f"Stage2={ctx['ltx_sdnq_stage2_offload_policy']}"
    )
    ctx["ltx_sdnq_offload_fallback"] = "none"
    ctx["ltx_sdnq_stage2_policy_switch"] = "not needed"

    from diffusers.hooks import apply_group_offloading  # type: ignore

    def _apply_component_group_offload(
        component_name: str,
        *,
        offload_type: str,
        num_blocks_per_group: Optional[int] = None,
        use_stream: bool = False,
    ) -> None:
        component = getattr(pipe, component_name, None)
        if component is None:
            return
        kwargs: Dict[str, Any] = {
            "onload_device": torch_module.device("cuda"),
            "offload_device": torch_module.device("cpu"),
            "offload_type": offload_type,
            "non_blocking": bool(use_stream),
            "use_stream": bool(use_stream),
            "record_stream": False,
            # Streamed INT8 uses on-demand pinning to avoid retaining a second
            # pinned transformer copy. Boundary INT4 Stage 2 never uses a
            # prefetch stream.
            "low_cpu_mem_usage": bool(use_stream),
        }
        if offload_type == "block_level":
            kwargs["num_blocks_per_group"] = int(num_blocks_per_group or 1)

        enable_component_group = getattr(component, "enable_group_offload", None)
        if callable(enable_component_group):
            enable_component_group(**kwargs)
        else:
            apply_group_offloading(module=component, **kwargs)

    def _apply_policy(
        selected_policy: Dict[str, Any],
        *,
        include_text_encoder: bool,
        phase_name: str,
    ) -> str:
        requested_mode = str(selected_policy["mode"])
        if requested_mode == "model_cpu_offload":
            pipe.enable_model_cpu_offload(device="cuda")
            ctx[f"ltx_sdnq_{phase_name}_stream_prefetch"] = "disabled: model-level offload"
            return "model_cpu_offload"
        if requested_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload(device="cuda")
            ctx[f"ltx_sdnq_{phase_name}_stream_prefetch"] = "disabled: sequential offload"
            return "sequential_cpu_offload"

        transformer_group = int(selected_policy["group_blocks"])
        use_stream = bool(
            selected_policy.get("use_stream", False) and transformer_group == 1
        )
        _apply_component_group_offload(
            "transformer",
            offload_type="block_level",
            num_blocks_per_group=transformer_group,
            use_stream=use_stream,
        )
        if include_text_encoder:
            # Prompt encoding is configured separately from transformer
            # residency.  The generic whole-model block hook did not lower the
            # Gemma peak because LTX2 calls Gemma3ForConditionalGeneration,
            # which also executes the large language-model head.  The dedicated
            # 8/12 GB route below bypasses that head and applies offload directly
            # to the Gemma text backbone.
            if profile_gb <= 12:
                ctx["ltx_sdnq_text_encoder_offload"] = (
                    f"planned: dedicated {profile_gb}GB low-VRAM Gemma path"
                )
            else:
                ctx["ltx_sdnq_text_encoder_offload"] = (
                    f"disabled: {profile_gb}GB profile allows full text encoder"
                )
        for component_name in ("connectors", "vae", "audio_vae", "vocoder"):
            _apply_component_group_offload(
                component_name,
                offload_type="leaf_level",
                use_stream=False,
            )

        stream_suffix = "+cuda_prefetch" if use_stream else "+no_prefetch"
        ctx[f"ltx_sdnq_{phase_name}_stream_prefetch"] = (
            "enabled: transformer groups use a separate CUDA prefetch stream"
            if use_stream
            else "disabled: synchronous groups prevent allocator/prefetch spill"
        )
        return (
            f"component_group_offload_{transformer_group}_of_"
            f"{selected_policy['block_count']}_blocks{stream_suffix}+leaf_aux_components"
        )

    try:
        stage1_offload_mode = _apply_policy(
            stage1_policy,
            include_text_encoder=True,
            phase_name="stage1",
        )
    except Exception as exc:
        if str(stage1_policy["mode"]) == "group_cpu_offload":
            fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                "[ltx-warning] SDNQ Stage 1 group offload failed; "
                f"falling back to sequential CPU offload: {fallback_reason}",
                flush=True,
            )
            pipe.enable_sequential_cpu_offload(device="cuda")
            stage1_offload_mode = "sequential_cpu_offload_fallback"
            ctx["ltx_sdnq_offload_fallback"] = fallback_reason
        else:
            raise RuntimeError(
                f"Could not enable SDNQ Stage 1 offload: {type(exc).__name__}: {exc}"
            ) from exc

    stage2_offload_mode = stage1_offload_mode
    offload_mode = stage1_offload_mode
    ctx["ltx_sdnq_stage1_offload_mode"] = stage1_offload_mode
    ctx["ltx_sdnq_stage2_offload_mode"] = (
        "pending transition" if workflow == "two_stages" else stage1_offload_mode
    )
    ctx["ltx_sdnq_stream_prefetch"] = ctx.get(
        "ltx_sdnq_stage1_stream_prefetch", "not attempted"
    )
    ctx["ltx_sdnq_decode_offload_guard"] = (
        "stage-aware: Stage 1 and Stage 2 policies are independent; "
        "boundary INT4 Stage 2 disables CUDA prefetch; final decode removes "
        "leaf HookRegistry entries and streams overlapping temporal VAE windows to CPU"
    )
    ctx["ltx_sdnq_offload_mode"] = (
        f"Stage1={stage1_offload_mode}; Stage2="
        f"{'pending' if workflow == 'two_stages' else stage1_offload_mode}"
    )
    print(
        f"[ltx-status] SDNQ Stage 1 offload: {stage1_offload_mode} "
        f"(work={stage1_policy['work_factor']:.2f}x, "
        f"estimated transformer residency={stage1_group_gb:.2f} GB)",
        flush=True,
    )
    if workflow == "two_stages":
        print(
            f"[ltx-status] SDNQ Stage 2 planned offload: {stage2_policy['mode']} "
            f"group={stage2_policy['group_blocks']}/{stage2_policy['block_count']} "
            f"prefetch={'on' if stage2_policy.get('use_stream') else 'off'} "
            f"(work={stage2_policy['work_factor']:.2f}x)",
            flush=True,
        )

    def _prepare_low_vram_text_encoder() -> bool:
        """Configure the dedicated 8/12 GB Gemma prompt path.

        LTX2's stock prompt helper calls Gemma3ForConditionalGeneration and
        computes the vocabulary logits even though only hidden states are used.
        It also applies a generic hook to the outer multimodal model.  On the
        quantized checkpoint that still produced a ~13.5 GB CUDA peak.  The
        low-VRAM path keeps the unused vision tower and LM head on CPU, offloads
        the text backbone itself, and later calls ``text_encoder.model``
        directly so no logits are materialized.
        """
        if profile_gb > 12:
            # The transformer group-offload policy leaves unrelated pipeline
            # components on CPU.  Diffusers still creates prompt token IDs on
            # pipe._execution_device (CUDA), so the stock Gemma path must move
            # the complete text encoder to CUDA explicitly before encode_prompt.
            # Without this, the embedding weight remains on CPU while input_ids
            # are CUDA tensors, raising a CPU/CUDA index_select device mismatch.
            text_encoder = getattr(pipe, "text_encoder", None)
            if text_encoder is None:
                raise RuntimeError("The SDNQ pipeline has no text encoder.")
            text_encoder.to(device=torch_module.device("cuda"))
            ctx["ltx_sdnq_text_encoder_offload"] = (
                f"disabled: {profile_gb}GB profile explicitly loads full text encoder on CUDA"
            )
            ctx["ltx_sdnq_prompt_encoder_path"] = (
                "stock full Gemma path; explicit CUDA placement before encode_prompt"
            )
            ctx["ltx_sdnq_prompt_sequence_length"] = "1024"
            return False

        text_encoder = getattr(pipe, "text_encoder", None)
        text_backbone = getattr(text_encoder, "model", None)
        if text_encoder is None or text_backbone is None:
            raise RuntimeError(
                "The selected 8/12 GB SDNQ profile requires the Gemma3 text "
                "backbone at pipe.text_encoder.model, but that structure was not found."
            )

        try:
            text_encoder.to("cpu")
        except Exception:
            pass
        gc.collect()
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass

        if profile_gb <= 8:
            # Leaf-level offload is intentionally used only for the once-per-run
            # prompt pass.  It is slower than block groups but reliably keeps
            # the 7.76 GB quantized Gemma plus temporary matmul buffers below an
            # actual 8 GB card.
            apply_group_offloading(
                module=text_backbone,
                onload_device=torch_module.device("cuda"),
                offload_device=torch_module.device("cpu"),
                offload_type="leaf_level",
                non_blocking=False,
                use_stream=False,
                record_stream=False,
                low_cpu_mem_usage=False,
            )
            ctx["ltx_sdnq_text_encoder_offload"] = (
                "enabled: 8GB dedicated Gemma backbone leaf offload; "
                "vision tower and LM head remain on CPU"
            )
            ctx["ltx_sdnq_prompt_encoder_path"] = (
                "low-vram text_encoder.model without LM-head logits; leaf offload"
            )
        else:
            language_model = getattr(text_backbone, "language_model", None)
            if language_model is None:
                raise RuntimeError(
                    "The 12 GB SDNQ text-encoder path could not find "
                    "pipe.text_encoder.model.language_model."
                )
            apply_group_offloading(
                module=language_model,
                onload_device=torch_module.device("cuda"),
                offload_device=torch_module.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=2,
                non_blocking=False,
                use_stream=False,
                record_stream=False,
                low_cpu_mem_usage=False,
            )
            ctx["ltx_sdnq_text_encoder_offload"] = (
                "enabled: 12GB dedicated Gemma language-model 2-block offload; "
                "vision tower and LM head remain on CPU"
            )
            ctx["ltx_sdnq_prompt_encoder_path"] = (
                "low-vram text_encoder.model without LM-head logits; 2-block offload"
            )
        ctx["ltx_sdnq_prompt_sequence_length"] = "1024"
        return True

    def _encode_gemma_hidden_states_low_vram(
        prompt_value: Any,
        *,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ) -> Tuple[Any, Any]:
        text_encoder = getattr(pipe, "text_encoder", None)
        text_backbone = getattr(text_encoder, "model", None)
        tokenizer = getattr(pipe, "tokenizer", None)
        if text_backbone is None or tokenizer is None:
            raise RuntimeError("Low-VRAM Gemma prompt encoder is incomplete.")

        prompts = [prompt_value] if isinstance(prompt_value, str) else list(prompt_value)
        prompts = [str(value).strip() for value in prompts]
        batch_size = len(prompts)
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=int(max_sequence_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        device = torch_module.device("cuda")
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with torch_module.inference_mode():
            outputs = text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "Gemma low-VRAM forward returned no hidden states; refusing to "
                "fall back to the 13 GB full-encoder path."
            )

        # LTX2 needs every Gemma layer.  Copy each layer to system RAM first and
        # stack there instead of creating a second full hidden-state stack on
        # CUDA.  This preserves the exact LTX2 conditioning tensor while cutting
        # the prompt-phase CUDA peak.
        hidden_states_cpu = [
            state.detach().to(device="cpu", dtype=torch_module.bfloat16)
            for state in hidden_states
        ]
        attention_mask_cpu = attention_mask.detach().to("cpu")
        del outputs, hidden_states, input_ids, attention_mask
        gc.collect()
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass

        prompt_embeds = torch_module.stack(hidden_states_cpu, dim=-1).flatten(2, 3)
        del hidden_states_cpu
        _, sequence_length, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, int(num_videos_per_prompt), 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * int(num_videos_per_prompt), sequence_length, -1
        )
        attention_mask_cpu = attention_mask_cpu.view(batch_size, -1)
        attention_mask_cpu = attention_mask_cpu.repeat(int(num_videos_per_prompt), 1)
        return prompt_embeds, attention_mask_cpu

    def _encode_prompt_low_vram(
        prompt_value: Any,
        negative_prompt_value: Any,
        *,
        do_classifier_free_guidance: bool,
        max_sequence_length: int = 1024,
    ) -> Tuple[Any, Any, Any, Any]:
        prompts = [prompt_value] if isinstance(prompt_value, str) else list(prompt_value)
        prompt_embeds, prompt_mask = _encode_gemma_hidden_states_low_vram(
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        negative_embeds = None
        negative_mask = None
        if do_classifier_free_guidance:
            negative_value = negative_prompt_value or ""
            negatives = (
                len(prompts) * [negative_value]
                if isinstance(negative_value, str)
                else list(negative_value)
            )
            if len(negatives) != len(prompts):
                raise ValueError(
                    "Negative prompt batch size must match the prompt batch size."
                )
            negative_embeds, negative_mask = _encode_gemma_hidden_states_low_vram(
                negatives,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )
        return prompt_embeds, prompt_mask, negative_embeds, negative_mask

    low_vram_prompt_path = _prepare_low_vram_text_encoder()

    negative_prompt = str(
        getattr(args, "negative_prompt", "")
        or _extra_option(extra, ("--negative-prompt", "--negative_prompt", "--neg-prompt"), "")
        or ""
    ).strip() or None
    guidance_scale = _float_option(extra, ("--video-cfg-guidance-scale",), 1.0)
    stg_scale = _float_option(extra, ("--video-stg-guidance-scale",), 0.0)
    guidance_rescale = _float_option(extra, ("--video-rescale-scale",), 0.7)
    modality_scale = _float_option(extra, ("--a2v-guidance-scale",), 1.0)
    audio_guidance_scale = _float_option(extra, ("--audio-cfg-guidance-scale",), 1.0)
    audio_stg_scale = _float_option(extra, ("--audio-stg-guidance-scale",), 0.0)
    audio_guidance_rescale = _float_option(extra, ("--audio-rescale-scale",), 0.0)
    audio_modality_scale = _float_option(extra, ("--v2a-guidance-scale",), 1.0)

    prompt = str(getattr(args, "prompt", ""))
    do_cfg = bool(guidance_scale > 1.0 or audio_guidance_scale > 1.0)
    prompt_sequence_length, prompt_token_count = _resolve_prompt_sequence_length(
        getattr(pipe, "tokenizer", None),
        prompt,
        negative_prompt,
        do_cfg,
        hard_max=1024,
    )
    ctx["ltx_sdnq_prompt_sequence_length_requested"] = "1024"
    ctx["ltx_sdnq_prompt_token_count"] = str(prompt_token_count)
    ctx["ltx_sdnq_prompt_sequence_length"] = str(prompt_sequence_length)
    ctx["ltx_sdnq_prompt_padding_tokens_removed"] = str(
        max(0, 1024 - int(prompt_sequence_length))
    )
    print(
        "[ltx-status] Encoding SDNQ prompt once for all stages "
        f"({prompt_sequence_length} tokens instead of fixed 1024 padding)",
        flush=True,
    )
    try:
        torch_module.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    prompt_driver_free_before = _driver_free_bytes(torch_module)
    prompt_t0 = time.perf_counter()
    if low_vram_prompt_path:
        (
            cached_prompt_embeds,
            cached_prompt_attention_mask,
            cached_negative_prompt_embeds,
            cached_negative_prompt_attention_mask,
        ) = _encode_prompt_low_vram(
            prompt,
            negative_prompt,
            do_classifier_free_guidance=do_cfg,
            max_sequence_length=prompt_sequence_length,
        )
    else:
        (
            cached_prompt_embeds,
            cached_prompt_attention_mask,
            cached_negative_prompt_embeds,
            cached_negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=prompt_sequence_length,
            device=pipe._execution_device,
            dtype=torch_module.bfloat16,
        )
    cached_prompt_embeds = _cpu_tensor(cached_prompt_embeds)
    cached_prompt_attention_mask = _cpu_tensor(cached_prompt_attention_mask)
    cached_negative_prompt_embeds = _cpu_tensor(cached_negative_prompt_embeds)
    cached_negative_prompt_attention_mask = _cpu_tensor(cached_negative_prompt_attention_mask)
    prompt_encode_s = time.perf_counter() - prompt_t0
    try:
        prompt_peak_allocated = float(torch_module.cuda.max_memory_allocated())
        prompt_peak_reserved = float(torch_module.cuda.max_memory_reserved())
    except Exception:
        prompt_peak_allocated = 0.0
        prompt_peak_reserved = 0.0
    prompt_driver_free_after = _driver_free_bytes(torch_module)
    ctx["ltx_sdnq_prompt_cache"] = "enabled: reused by Stage 1 and Stage 2"
    ctx["ltx_sdnq_prompt_encode_time_s"] = f"{prompt_encode_s:.3f}"
    ctx["ltx_sdnq_prompt_peak_allocated_gb"] = f"{prompt_peak_allocated / (1024**3):.2f}"
    ctx["ltx_sdnq_prompt_peak_reserved_gb"] = f"{prompt_peak_reserved / (1024**3):.2f}"
    prompt_min_free = min(
        value for value in (prompt_driver_free_before, prompt_driver_free_after)
        if value is not None
    ) if any(value is not None for value in (prompt_driver_free_before, prompt_driver_free_after)) else None
    ctx["ltx_sdnq_prompt_min_driver_free_gb"] = (
        f"{prompt_min_free / (1024**3):.2f}" if prompt_min_free is not None else "n/a"
    )

    # Prompt embeddings are cached on CPU, so the text encoder is no longer
    # needed for either denoise stage.  Releasing it here prevents the 8/12 GB
    # profiles from carrying prompt-encoder hooks and allocator reservations
    # into Stage 1.
    encoder_cleanup = "not needed"
    try:
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            try:
                text_encoder.to("cpu")
            except Exception:
                pass
            pipe.text_encoder = None
            del text_encoder
            gc.collect()
            torch_module.cuda.empty_cache()
            encoder_cleanup = _cuda_text(torch_module)
    except Exception as exc:
        encoder_cleanup = f"warning: {type(exc).__name__}: {exc}"
    ctx["ltx_sdnq_text_encoder_cleanup"] = encoder_cleanup

    seed = int(getattr(args, "seed", 0) or 0)
    generator = torch_module.Generator(device="cuda").manual_seed(seed)
    # Stage 2 receives the Stage 1 video/audio latents from system RAM.
    # Diffusers adds the refinement noise *before* moving supplied latents to
    # the pipeline execution device, so the random generator must live on the
    # same device as those CPU latents.  Reusing the CUDA generator here raises:
    #   Cannot generate a cpu tensor from a generator of type cuda.
    # Keep a separate generator with the same seed for the CPU latent handoff.
    stage2_cpu_generator = torch_module.Generator(device="cpu").manual_seed(seed)
    total_steps = int(getattr(args, "num_inference_steps", 8) or 8)
    step_events = []
    phase_min_driver_free: Dict[str, Optional[int]] = {}
    phase_peak_allocated: Dict[str, float] = {}
    phase_peak_reserved: Dict[str, float] = {}
    current_phase = "idle"

    def _sync_cuda() -> None:
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()
        except Exception:
            pass

    def _sample_driver_free(phase_name: Optional[str] = None) -> None:
        name = str(phase_name or current_phase)
        try:
            free_bytes = int(torch_module.cuda.mem_get_info()[0])
            previous = phase_min_driver_free.get(name)
            phase_min_driver_free[name] = (
                free_bytes if previous is None else min(previous, free_bytes)
            )
        except Exception:
            pass

    def _begin_memory_phase(phase_name: str) -> None:
        nonlocal current_phase
        current_phase = str(phase_name)
        _sync_cuda()
        try:
            torch_module.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        phase_min_driver_free[current_phase] = None
        _sample_driver_free(current_phase)

    def _finish_memory_phase(phase_name: str) -> None:
        _sync_cuda()
        _sample_driver_free(phase_name)
        try:
            phase_peak_allocated[phase_name] = float(
                torch_module.cuda.max_memory_allocated()
            ) / (1024 ** 3)
            phase_peak_reserved[phase_name] = float(
                torch_module.cuda.max_memory_reserved()
            ) / (1024 ** 3)
        except Exception:
            phase_peak_allocated[phase_name] = 0.0
            phase_peak_reserved[phase_name] = 0.0
        minimum = phase_min_driver_free.get(phase_name)
        ctx[f"ltx_sdnq_{phase_name}_peak_allocated_gb"] = (
            f"{phase_peak_allocated[phase_name]:.2f}"
        )
        ctx[f"ltx_sdnq_{phase_name}_peak_reserved_gb"] = (
            f"{phase_peak_reserved[phase_name]:.2f}"
        )
        ctx[f"ltx_sdnq_{phase_name}_min_driver_free_gb"] = (
            f"{float(minimum)/(1024**3):.2f}" if minimum is not None else "unavailable"
        )

    def _make_callback(
        label: str,
        expected_steps: int,
        phase_name: str,
        *,
        trim_cuda_cache: bool = False,
    ):
        previous = time.perf_counter()
        trim_count = 0
        memory_events = []

        def _callback(
            _pipe: Any,
            step: int,
            _timestep: int,
            callback_kwargs: Dict[str, Any],
        ) -> Dict[str, Any]:
            nonlocal previous, trim_count
            now = time.perf_counter()
            real = now - previous
            event = f"{label} step {int(step)+1}/{expected_steps} real={real:.3f}s"
            step_events.append(event)
            _sample_driver_free(phase_name)
            print(f"[ltx-step-time] {event}", flush=True)

            if str(phase_name) == "stage2":
                try:
                    allocated = float(torch_module.cuda.memory_allocated()) / (1024 ** 3)
                    reserved = float(torch_module.cuda.memory_reserved()) / (1024 ** 3)
                    driver_free = float(torch_module.cuda.mem_get_info()[0]) / (1024 ** 3)
                    memory_events.append(
                        f"step {int(step)+1}/{expected_steps}: allocated={allocated:.2f} GB, "
                        f"reserved={reserved:.2f} GB, driver_free={driver_free:.2f} GB"
                    )
                    ctx["ltx_sdnq_stage2_step_memory"] = " | ".join(memory_events)
                except Exception:
                    pass

            # Do this only between Stage 2 refinement steps. ``empty_cache``
            # never frees live tensors; it returns unused allocator blocks to
            # the CUDA driver so WDDM does not keep paging a stale 20+ GB cache
            # through shared GPU memory for the next step.
            if bool(trim_cuda_cache) and int(step) + 1 < int(expected_steps):
                try:
                    _sync_cuda()
                    before_reserved = float(torch_module.cuda.memory_reserved()) / (1024 ** 3)
                    before_free = float(torch_module.cuda.mem_get_info()[0]) / (1024 ** 3)
                    gc.collect()
                    torch_module.cuda.empty_cache()
                    _sync_cuda()
                    after_reserved = float(torch_module.cuda.memory_reserved()) / (1024 ** 3)
                    after_free = float(torch_module.cuda.mem_get_info()[0]) / (1024 ** 3)
                    trim_count += 1
                    ctx["ltx_sdnq_stage2_allocator_trim_count"] = str(trim_count)
                    ctx["ltx_sdnq_stage2_allocator_trim_last"] = (
                        f"after step {int(step)+1}/{expected_steps}: "
                        f"reserved {before_reserved:.2f}->{after_reserved:.2f} GB; "
                        f"driver_free {before_free:.2f}->{after_free:.2f} GB"
                    )
                    print(
                        "[ltx-status] Stage 2 allocator trim: "
                        f"reserved {before_reserved:.2f}->{after_reserved:.2f} GB; "
                        f"driver_free {before_free:.2f}->{after_free:.2f} GB",
                        flush=True,
                    )
                    _sample_driver_free(phase_name)
                except Exception as exc:
                    ctx["ltx_sdnq_stage2_allocator_trim_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            previous = time.perf_counter()
            return callback_kwargs

        return _callback

    loaded_conditions: List[Any] = []
    if has_conditions:
        rendered: List[str] = []
        for spec in condition_specs:
            path_text = str((spec or {}).get("path") or "").strip()
            raw_frame = int((spec or {}).get("frame") or 0)
            strength = float((spec or {}).get("strength") if (spec or {}).get("strength") is not None else 1.0)
            image_file = Path(path_text).expanduser()
            if not image_file.is_file():
                raise FileNotFoundError(f"SDNQ visual condition not found: {image_file}")
            latent_index = _frame_to_latent_index(raw_frame, requested_num_frames, temporal_ratio)
            loaded_conditions.append(
                LTX2VideoCondition(
                    frames=load_image(str(image_file)),
                    index=latent_index,
                    strength=strength,
                )
            )
            rendered.append(f"{image_file.name}:frame={raw_frame}:latent={latent_index}:strength={strength:g}")
        ctx["ltx_sdnq_input_mode"] = "image-conditioned-video"
        ctx["ltx_int4_visual_conditions"] = "; ".join(rendered)
        ctx["ltx_int4_visual_condition_count"] = str(len(loaded_conditions))
    else:
        ctx["ltx_sdnq_input_mode"] = "text-to-video"
        ctx["ltx_int4_visual_condition_count"] = "0"

    common_kwargs: Dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": cached_prompt_embeds,
        "prompt_attention_mask": cached_prompt_attention_mask,
        "negative_prompt_embeds": cached_negative_prompt_embeds,
        "negative_prompt_attention_mask": cached_negative_prompt_attention_mask,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "stg_scale": stg_scale,
        "modality_scale": modality_scale,
        "guidance_rescale": guidance_rescale,
        "audio_guidance_scale": audio_guidance_scale,
        "audio_stg_scale": audio_stg_scale,
        "audio_modality_scale": audio_modality_scale,
        "audio_guidance_rescale": audio_guidance_rescale,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "max_sequence_length": prompt_sequence_length,
    }
    if stg_scale > 0.0 or audio_stg_scale > 0.0:
        common_kwargs["spatio_temporal_guidance_blocks"] = [28]
    if loaded_conditions:
        common_kwargs["conditions"] = loaded_conditions
    if reference_audio_latents is not None:
        # Stage 1 uses a CUDA generator, so supplied latents must enter on CUDA.
        # They are small and frozen; the video transformer can still attend to them.
        common_kwargs["audio_latents"] = reference_audio_latents.to(
            device="cuda", dtype=torch_module.float32
        )
        common_kwargs["noise_scale"] = 0.0

    try:
        distilled_sigmas = list(DISTILLED_SIGMA_VALUES)
    except Exception as exc:
        distilled_sigmas = []
        ctx["ltx_sdnq_sigma_schedule"] = (
            f"scheduler default; constant unavailable: {type(exc).__name__}: {exc}"
        )

    output_path = Path(str(getattr(args, "output_path"))).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workflow == "one_stage":
        call_kwargs = dict(common_kwargs)
        call_kwargs.update(
            {
                "width": target_width,
                "height": target_height,
                "num_inference_steps": total_steps,
                "output_type": "np",
                "return_dict": False,
                "callback_on_step_end": _make_callback("sdnq", total_steps, "stage1"),
            }
        )
        if total_steps in {len(distilled_sigmas), max(1, len(distilled_sigmas) - 1)} and distilled_sigmas:
            call_kwargs["sigmas"] = distilled_sigmas
            ctx["ltx_sdnq_sigma_schedule"] = f"DISTILLED_SIGMA_VALUES ({len(distilled_sigmas)} values)"
        elif "ltx_sdnq_sigma_schedule" not in ctx:
            ctx["ltx_sdnq_sigma_schedule"] = f"scheduler default for {total_steps} steps"

        print("[ltx-status] Running SDNQ denoising", flush=True)
        _begin_memory_phase("stage1")
        run_t0 = time.perf_counter()
        video, audio = _call_ltx2_pipe(
            pipe, call_kwargs, freeze_audio=freeze_reference_audio, ctx=ctx
        )
        run_s = time.perf_counter() - run_t0
        _finish_memory_phase("stage1")
        ctx["ltx_sdnq_stage1_time_s"] = f"{run_s:.3f}"
        ctx["ltx_sdnq_stage2_time_s"] = "n/a"
        ctx["ltx_sdnq_upsample_time_s"] = "n/a"
        ctx["ltx_sdnq_generation_time_s"] = f"{run_s:.3f}"
        ctx["ltx_sdnq_cuda_after_stage1"] = _cuda_text(torch_module)
        ctx["ltx_sdnq_cuda_after_generation"] = ctx["ltx_sdnq_cuda_after_stage1"]
    else:
        stage1_kwargs = dict(common_kwargs)
        stage1_kwargs.update(
            {
                "width": stage1_width,
                "height": stage1_height,
                "num_inference_steps": total_steps,
                "output_type": "latent",
                "return_dict": False,
                "callback_on_step_end": _make_callback("sdnq stage1", total_steps, "stage1"),
            }
        )
        if total_steps in {len(distilled_sigmas), max(1, len(distilled_sigmas) - 1)} and distilled_sigmas:
            stage1_kwargs["sigmas"] = distilled_sigmas
            ctx["ltx_sdnq_sigma_schedule"] = f"DISTILLED_SIGMA_VALUES ({len(distilled_sigmas)} values)"
        elif "ltx_sdnq_sigma_schedule" not in ctx:
            ctx["ltx_sdnq_sigma_schedule"] = f"scheduler default for {total_steps} steps"

        print(
            f"[ltx-status] SDNQ Stage 1: {stage1_width}x{stage1_height}, {total_steps} steps",
            flush=True,
        )
        _begin_memory_phase("stage1")
        stage1_t0 = time.perf_counter()
        video_latent, stage1_audio_latent = _call_ltx2_pipe(
            pipe, stage1_kwargs, freeze_audio=freeze_reference_audio, ctx=ctx
        )
        # The official A2V route feeds the same encoded audio latent into both
        # stages.  Reuse the original reference tensor explicitly rather than
        # depending on a normalize/unpack round-trip from the Stage 1 output.
        if reference_audio_latents is not None:
            audio_latent = reference_audio_latents
            del stage1_audio_latent
            ctx["ltx_int4_stage2_audio_source"] = "original encoded reference audio latent"
        else:
            audio_latent = stage1_audio_latent
            ctx["ltx_int4_stage2_audio_source"] = "Stage 1 generated audio latent"
        stage1_s = time.perf_counter() - stage1_t0
        _finish_memory_phase("stage1")
        ctx["ltx_sdnq_stage1_time_s"] = f"{stage1_s:.3f}"
        ctx["ltx_sdnq_cuda_after_stage1"] = _cuda_text(torch_module)

        # Keep the latent handoff in system RAM. The pipeline has already run
        # maybe_free_model_hooks(), so this clears activation memory before the
        # standalone spatial upsampler enters CUDA.
        video_latent = video_latent.to(device="cpu")
        audio_latent = audio_latent.to(device="cpu")
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
        ctx["ltx_sdnq_cuda_before_upsample"] = _cuda_text(torch_module)

        upsampler_text = str(getattr(args, "spatial_upsampler_path", "") or "").strip()
        if not upsampler_text:
            raise RuntimeError("SDNQ two_stages requires --spatial-upsampler-path.")
        upsampler_path = Path(upsampler_text).expanduser().resolve()
        ctx["ltx_sdnq_upsampler_path"] = str(upsampler_path)
        print(f"[ltx-status] Loading LTX 2.3 spatial upsampler: {upsampler_path}", flush=True)
        latent_upsampler, upsampler_load_s, upsampler_bytes = _load_ltx23_spatial_upsampler(
            upsampler_path,
            model_cls=LTX2LatentUpsamplerModel,
            torch_module=torch_module,
        )
        ctx["ltx_sdnq_upsampler_load_time_s"] = f"{upsampler_load_s:.3f}"
        ctx["ltx_sdnq_upsampler_size_mb"] = f"{upsampler_bytes/(1024**2):.2f}"

        upsample_t0 = time.perf_counter()
        try:
            latent_upsampler.to(device="cuda", dtype=torch_module.bfloat16)
            upsample_pipe = LTX2LatentUpsamplePipeline(
                vae=pipe.vae,
                latent_upsampler=latent_upsampler,
            )
            upscaled_video_latent = upsample_pipe(
                latents=video_latent,
                latents_normalized=False,
                height=stage1_height,
                width=stage1_width,
                num_frames=num_frames,
                generator=generator,
                output_type="latent",
                return_dict=False,
            )[0]
            upscaled_video_latent = upscaled_video_latent.to(device="cpu")
        finally:
            try:
                latent_upsampler.to(device="cpu")
            except Exception:
                pass
            try:
                del upsample_pipe
            except Exception:
                pass
            try:
                del latent_upsampler
            except Exception:
                pass
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass
        upsample_s = time.perf_counter() - upsample_t0
        ctx["ltx_sdnq_upsample_time_s"] = f"{upsample_s:.3f}"
        ctx["ltx_sdnq_cuda_after_upsample"] = _cuda_text(torch_module)
        del video_latent

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )
        stage2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)
        ctx["ltx_sdnq_stage2_sigma_schedule"] = (
            f"STAGE_2_DISTILLED_SIGMA_VALUES ({len(stage2_sigmas)} values)"
        )
        stage2_steps = 3
        stage2_kwargs: Dict[str, Any] = {
            "latents": upscaled_video_latent,
            "audio_latents": audio_latent,
            "prompt": None,
            "negative_prompt": None,
            "prompt_embeds": cached_prompt_embeds,
            "prompt_attention_mask": cached_prompt_attention_mask,
            "negative_prompt_embeds": cached_negative_prompt_embeds,
            "negative_prompt_attention_mask": cached_negative_prompt_attention_mask,
            "width": target_width,
            "height": target_height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": stage2_steps,
            "noise_scale": float(stage2_sigmas[0]),
            "sigmas": stage2_sigmas,
            "guidance_scale": 1.0,
            "stg_scale": 0.0,
            "modality_scale": 1.0,
            "guidance_rescale": 0.0,
            "audio_guidance_scale": 1.0,
            "audio_stg_scale": 0.0,
            "audio_modality_scale": 1.0,
            "audio_guidance_rescale": 0.0,
            "generator": stage2_cpu_generator,
            # Return denoised latents first. Decoding inside the group-offloaded
            # Stage 2 pipeline keeps transformer hooks and CUDA reservations
            # alive while the VAE/vocoder enter CUDA, which caused sustained
            # shared-memory spill after step 3/3.
            "output_type": "latent",
            "return_dict": False,
            "callback_on_step_end": _make_callback(
                "sdnq stage2",
                stage2_steps,
                "stage2",
                trim_cuda_cache=stage2_allocator_trim,
            ),
            "callback_on_step_end_tensor_inputs": ["latents"],
            "max_sequence_length": prompt_sequence_length,
        }
        # Visual conditions are applied in Stage 1 and are already embedded in
        # the upsampled video latents. Diffusers' official two-stage condition
        # example does not re-apply them during the three-step refinement pass.

        policy_changed = bool(
            stage2_policy["mode"] != stage1_policy["mode"]
            or int(stage2_policy["group_blocks"]) != int(stage1_policy["group_blocks"])
            or bool(stage2_policy.get("use_stream", False))
            != bool(stage1_policy.get("use_stream", False))
        )
        if policy_changed:
            print(
                "[ltx-status] Rebuilding SDNQ residency for Stage 2 "
                f"({stage1_policy['mode']} -> {stage2_policy['mode']})",
                flush=True,
            )
            switch_before = _cuda_text(torch_module)
            try:
                pipe.remove_all_hooks()
            except Exception as exc:
                raise RuntimeError(
                    "Could not remove Stage 1 Accelerate hooks before the "
                    f"Stage 2 residency switch: {type(exc).__name__}: {exc}"
                ) from exc
            switch_hook_count, switch_hook_names = _remove_pipeline_diffusers_hooks(pipe)
            ctx["ltx_sdnq_stage2_switch_diffusers_hooks_removed"] = str(switch_hook_count)
            if switch_hook_names:
                ctx["ltx_sdnq_stage2_switch_diffusers_hook_names"] = " | ".join(switch_hook_names)
            try:
                pipe.to("cpu", silence_dtype_warnings=True)
            except TypeError:
                pipe.to("cpu")
            _sync_cuda()
            gc.collect()
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass
            _sync_cuda()
            switch_clean = _cuda_text(torch_module)
            try:
                stage2_offload_mode = _apply_policy(
                    stage2_policy,
                    include_text_encoder=False,
                    phase_name="stage2",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not apply the Stage 2 SDNQ residency policy "
                    f"for the selected VRAM profile: {type(exc).__name__}: {exc}"
                ) from exc
            ctx["ltx_sdnq_stage2_policy_switch"] = (
                f"applied: before={switch_before}; after_cleanup={switch_clean}; "
                f"mode={stage2_offload_mode}"
            )
        else:
            stage2_offload_mode = stage1_offload_mode
            ctx["ltx_sdnq_stage2_stream_prefetch"] = ctx.get(
                "ltx_sdnq_stage1_stream_prefetch", "disabled"
            )
            ctx["ltx_sdnq_stage2_policy_switch"] = "not needed: same policy for both stages"

        ctx["ltx_sdnq_stage2_offload_mode"] = stage2_offload_mode
        offload_mode = f"Stage1={stage1_offload_mode}; Stage2={stage2_offload_mode}"
        ctx["ltx_sdnq_offload_mode"] = offload_mode
        ctx["ltx_sdnq_stream_prefetch"] = (
            f"Stage1={ctx.get('ltx_sdnq_stage1_stream_prefetch', 'n/a')}; "
            f"Stage2={ctx.get('ltx_sdnq_stage2_stream_prefetch', 'n/a')}"
        )

        ctx["ltx_sdnq_cuda_before_stage2"] = _cuda_text(torch_module)
        print(
            f"[ltx-status] SDNQ Stage 2: {target_width}x{target_height}, {stage2_steps} refinement steps",
            flush=True,
        )
        _begin_memory_phase("stage2")
        stage2_t0 = time.perf_counter()
        video_latents_final, audio_latents_final = _call_ltx2_pipe(
            pipe, stage2_kwargs, freeze_audio=freeze_reference_audio, ctx=ctx
        )
        stage2_s = time.perf_counter() - stage2_t0
        _finish_memory_phase("stage2")
        ctx["ltx_sdnq_stage2_time_s"] = f"{stage2_s:.3f}"
        ctx["ltx_sdnq_stage2_output_type"] = "latent"
        ctx["ltx_sdnq_cuda_after_stage2_latents"] = _cuda_text(torch_module)

        # Preserve only CPU latents before tearing down Stage 2. The latent
        # tensors are already denormalized by Diffusers when output_type is
        # ``latent``. This boundary is intentionally before VAE/audio decode.
        video_latents_final = _cpu_tensor(video_latents_final)
        audio_latents_final = _cpu_tensor(audio_latents_final)
        del upscaled_video_latent, audio_latent

        teardown_before = _cuda_text(torch_module)
        teardown_t0 = time.perf_counter()
        try:
            pipe.remove_all_hooks()
        except Exception as exc:
            raise RuntimeError(
                "Could not remove Stage 2 group-offload hooks before latent "
                f"decode: {type(exc).__name__}: {exc}"
            ) from exc
        decode_hook_count, decode_hook_names = _remove_pipeline_diffusers_hooks(pipe)
        ctx["ltx_sdnq_stage2_diffusers_hooks_removed"] = str(decode_hook_count)
        if decode_hook_names:
            ctx["ltx_sdnq_stage2_diffusers_hook_names"] = " | ".join(decode_hook_names)
        try:
            pipe.to("cpu", silence_dtype_warnings=True)
        except TypeError:
            pipe.to("cpu")
        _sync_cuda()
        gc.collect()
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
        _sync_cuda()
        teardown_after = _cuda_text(torch_module)
        ctx["ltx_sdnq_stage2_decode_teardown"] = (
            f"pipeline/Accelerate hooks removed; {decode_hook_count} nested Diffusers hooks removed; "
            f"pipeline moved to CPU; before={teardown_before}; after={teardown_after}"
        )
        ctx["ltx_sdnq_stage2_decode_teardown_time_s"] = (
            f"{time.perf_counter() - teardown_t0:.3f}"
        )
        ctx["ltx_sdnq_cuda_after_stage2"] = teardown_after
        ctx["ltx_sdnq_cuda_after_generation"] = teardown_after

        print(
            "[ltx-status] Stage 2 complete; transformer hooks released before decode",
            flush=True,
        )
        _begin_memory_phase("decode")
        decode_t0 = time.perf_counter()

        # Decode video with only the split VAE on CUDA.  Spatial tiling alone
        # cannot solve a long-video decode because all 241 temporal activations
        # still coexist.  Stream overlapping temporal windows instead and move
        # every completed section directly to CPU before decoding the next one.
        video_vae = pipe.vae
        video_decode_t0 = time.perf_counter()
        decode_plan = _select_video_decode_plan(
            profile_gb=profile_gb,
            width=target_width,
            height=target_height,
            num_frames=num_frames,
            extra=extra,
        )
        ctx["ltx_sdnq_video_decode_plan"] = (
            f"temporal window={decode_plan['window_frames']}f, "
            f"stride={decode_plan['stride_frames']}f, "
            f"overlap={decode_plan['overlap_frames']}f, "
            f"spatial_tiling={'on' if decode_plan['spatial_tiling'] else 'off'}"
        )
        ctx["ltx_sdnq_video_decode_mode"] = (
            "streaming temporal VAE decode; one overlapping window on CUDA at a time; "
            "completed windows transferred directly to CPU; nested group-offload hooks removed"
        )
        print(
            "[ltx-status] Final video decode: spill-free temporal windows "
            f"{decode_plan['window_frames']}f/{decode_plan['stride_frames']}f "
            f"(spatial tiling {'on' if decode_plan['spatial_tiling'] else 'off'})",
            flush=True,
        )
        decoded_video = None
        try:
            video_vae.to("cuda")
            decoded_video, decode_stats = _streaming_temporal_vae_decode(
                vae=video_vae,
                latents=video_latents_final,
                torch_module=torch_module,
                target_num_frames=requested_num_frames,
                window_frames=int(decode_plan["window_frames"]),
                stride_frames=int(decode_plan["stride_frames"]),
                spatial_tiling=bool(decode_plan["spatial_tiling"]),
                sample_memory=lambda: _sample_driver_free("decode"),
            )
            ctx["ltx_sdnq_video_decode_chunks"] = str(decode_stats["chunks"])
            ctx["ltx_sdnq_video_decode_window_frames"] = str(decode_stats["window_frames"])
            ctx["ltx_sdnq_video_decode_stride_frames"] = str(decode_stats["stride_frames"])
            ctx["ltx_sdnq_video_decode_overlap_frames"] = str(decode_stats["overlap_frames"])
            ctx["ltx_sdnq_video_decode_output_frames"] = str(decode_stats["decoded_frames"])
            ctx["ltx_sdnq_video_decode_spatial_tiling"] = (
                "enabled" if decode_stats["spatial_tiling"] else "disabled"
            )
            ctx["ltx_sdnq_vae_tiling"] = (
                "pipeline preparation used spatial tiling for I2V encoding; final decode used "
                f"temporal CPU-streaming with {decode_stats['spatial_tiling_status']}"
            )
            video = pipe.video_processor.postprocess_video(
                decoded_video, output_type="np"
            )
        finally:
            try:
                video_vae.to("cpu")
            except Exception:
                pass
            try:
                del decoded_video
            except Exception:
                pass
            gc.collect()
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass
            _sync_cuda()
        ctx["ltx_sdnq_video_decode_time_s"] = (
            f"{time.perf_counter() - video_decode_t0:.3f}"
        )
        ctx["ltx_sdnq_cuda_after_video_decode"] = _cuda_text(torch_module)

        # Uploaded reference audio is muxed directly, so the generated audio
        # latent does not need the audio VAE or vocoder at all. For normal T2V
        # audio, load and release those two components one at a time.
        if reference_audio_waveform is not None:
            audio = None
            ctx["ltx_sdnq_audio_decode_time_s"] = "skipped: reference waveform muxed"
            ctx["ltx_sdnq_audio_decode_mode"] = "reference waveform; no audio VAE/vocoder CUDA load"
        else:
            audio_decode_t0 = time.perf_counter()
            audio_vae = pipe.audio_vae
            vocoder = pipe.vocoder
            generated_mel = None
            try:
                audio_vae.to("cuda")
                decode_audio_latents = audio_latents_final.to(
                    device="cuda", dtype=audio_vae.dtype
                )
                with torch_module.inference_mode():
                    generated_mel = audio_vae.decode(
                        decode_audio_latents, return_dict=False
                    )[0].float().cpu()
            finally:
                try:
                    audio_vae.to("cpu")
                except Exception:
                    pass
                try:
                    del decode_audio_latents
                except Exception:
                    pass
                gc.collect()
                try:
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
                _sync_cuda()
            try:
                vocoder.to("cuda")
                vocoder_dtype = getattr(vocoder, "dtype", torch_module.float32)
                with torch_module.inference_mode():
                    audio = vocoder(generated_mel.to(device="cuda", dtype=vocoder_dtype))
                    audio = audio.float().cpu()
            finally:
                try:
                    vocoder.to("cpu")
                except Exception:
                    pass
                try:
                    del generated_mel
                except Exception:
                    pass
                gc.collect()
                try:
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
                _sync_cuda()
            ctx["ltx_sdnq_audio_decode_time_s"] = (
                f"{time.perf_counter() - audio_decode_t0:.3f}"
            )
            ctx["ltx_sdnq_audio_decode_mode"] = "separate audio VAE then vocoder"

        del video_latents_final, audio_latents_final
        decode_s = time.perf_counter() - decode_t0
        _finish_memory_phase("decode")
        ctx["ltx_sdnq_decode_time_s"] = f"{decode_s:.3f}"
        ctx["ltx_sdnq_cuda_after_decode"] = _cuda_text(torch_module)

        run_s = stage1_s + upsample_s + stage2_s
        ctx["ltx_sdnq_generation_time_s"] = f"{run_s:.3f}"

    ctx["ltx_realtime_step_timer_installed"] = "YES: SDNQ callback"
    ctx["ltx_realtime_step_event_count"] = str(len(step_events))
    ctx["ltx_realtime_step_events"] = step_events
    ctx["ltx_realtime_step_summary"] = step_events[-1] if step_events else "no callback events"

    overall_peak_allocated = max(phase_peak_allocated.values(), default=0.0)
    overall_peak_reserved = max(phase_peak_reserved.values(), default=0.0)
    valid_phase_mins = [
        value for value in phase_min_driver_free.values() if value is not None
    ]
    overall_min_driver_free = min(valid_phase_mins) if valid_phase_mins else None
    ctx["ltx_sdnq_peak_allocated_gb"] = f"{overall_peak_allocated:.2f}"
    ctx["ltx_sdnq_peak_reserved_gb"] = f"{overall_peak_reserved:.2f}"
    ctx["ltx_sdnq_min_driver_free_gb"] = (
        f"{float(overall_min_driver_free)/(1024**3):.2f}"
        if overall_min_driver_free is not None
        else "unavailable"
    )
    if overall_min_driver_free is None:
        ctx["ltx_sdnq_vram_safety_result"] = "unavailable"
    else:
        free_gb = float(overall_min_driver_free) / (1024 ** 3)
        if free_gb >= 1.00:
            safety = "SAFE: at least 1.00 GB driver-free"
        elif free_gb >= 0.50:
            safety = "TIGHT: below 1.00 GB driver-free"
        elif free_gb > 0.05:
            safety = "EDGE: below 0.50 GB driver-free"
        else:
            safety = "UNSAFE/SPILL LIKELY: driver-free VRAM reached the floor"
        ctx["ltx_sdnq_vram_safety_result"] = f"{safety} ({free_gb:.2f} GB)"
    if workflow == "one_stage":
        ctx["ltx_sdnq_stage2_peak_allocated_gb"] = "n/a"
        ctx["ltx_sdnq_stage2_peak_reserved_gb"] = "n/a"
        ctx["ltx_sdnq_stage2_min_driver_free_gb"] = "n/a"
        ctx["ltx_sdnq_stage2_offload_mode"] = "n/a"
        ctx["ltx_sdnq_stage2_offload_policy"] = "n/a"
        ctx["ltx_sdnq_stage2_policy_switch"] = "n/a"
        ctx["ltx_sdnq_stage2_stream_prefetch"] = "n/a"

    print("[ltx-status] Encoding SDNQ output", flush=True)
    if reference_audio_waveform is not None and reference_audio_rate is not None:
        output_audio = reference_audio_waveform.float().cpu()
        audio_sample_rate = int(reference_audio_rate)
        ctx["ltx_int4_output_audio_source"] = "original uploaded reference audio"
    else:
        if audio is None:
            raise RuntimeError("Generated LTX audio was not decoded.")
        output_audio = audio[0].float().cpu()
        audio_sample_rate = int(getattr(pipe.vocoder.config, "output_sampling_rate", 24000))
        ctx["ltx_int4_output_audio_source"] = "generated LTX audio"
    encode_video(
        video[0],
        fps=frame_rate,
        audio=output_audio,
        audio_sample_rate=audio_sample_rate,
        output_path=str(output_path),
    )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"SDNQ encode_video did not create a valid output: {output_path}")

    ctx["ltx_sdnq_output_audio_rate"] = str(audio_sample_rate)
    ctx["ltx_sdnq_output_size_mb"] = f"{output_path.stat().st_size/(1024**2):.2f}"
    ctx["ltx_sdnq_cuda_after_encode"] = _cuda_text(torch_module)
    print(f"[ltx-status] SDNQ output saved: {output_path}", flush=True)

    try:
        del video, audio, pipe
    except Exception:
        pass
    return {
        "output_path": output_path,
        "workflow": workflow,
        "load_time_s": load_s,
        "prompt_encode_time_s": prompt_encode_s,
        "generation_time_s": run_s,
        "offload_mode": offload_mode,
        "group_blocks": int(stage2_policy["group_blocks"]),
        "weight_dtype": str(stage1_policy["variant"]),
    }
# ---------------------------------------------------------------------------
# Standalone isolated CLI wrapper
# ---------------------------------------------------------------------------

def _detect_vram_total_gb(torch_module: Any) -> float:
    try:
        return float(torch_module.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
    except Exception:
        return 24.0


def _detect_vram_profile(torch_module: Any) -> int:
    total_gb = _detect_vram_total_gb(torch_module)
    # Consumer cards often report slightly below their marketed capacity in GiB.
    if total_gb >= 23.0:
        return 24
    if total_gb >= 15.0:
        return 16
    if total_gb >= 11.0:
        return 12
    return 8


def _normalize_input_image(args: Any, ctx: Dict[str, Any]) -> None:
    """Validate and normalize all Planner visual conditions.

    The legacy ``--i2v-image`` flags remain accepted for the Music Creator and
    older callers.  New Planner/UI callers use repeated ``--image PATH FRAME
    STRENGTH`` entries so start, reference, middle and final images all travel
    through the same condition pipeline.
    """
    if int(getattr(args, "i2v_image_crf", 0) or 0) != 0:
        raise RuntimeError("INT4 image CRF is not implemented in the Planner runner; keep it at 0.")

    raw_conditions: List[Tuple[str, int, float]] = []
    legacy_path = str(getattr(args, "i2v_image", "") or "").strip()
    if legacy_path:
        raw_conditions.append(
            (
                legacy_path,
                int(getattr(args, "i2v_image_frame", 0) or 0),
                float(getattr(args, "i2v_image_strength", 1.0) or 1.0),
            )
        )

    for raw in list(getattr(args, "image_conditions", []) or []):
        try:
            path_text, frame_text, strength_text = raw
            raw_conditions.append((str(path_text).strip(), int(frame_text), float(strength_text)))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid --image entry {raw!r}; expected PATH FRAME STRENGTH: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if not raw_conditions:
        args._planner_conditions = []
        ctx["ltx_int4_input_mode_requested"] = "text-to-video"
        ctx["ltx_int4_visual_condition_count"] = "0"
        return

    root = Path(
        str(getattr(args, "ltx_root", "") or Path(__file__).resolve().parent.parent)
    ).expanduser().resolve()
    normalize = bool(getattr(args, "normalize_input_image", True))
    temp_dir = root / "temp" / "ltx_int4"
    if normalize:
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                f"Pillow is required to normalize INT4 condition images: {type(exc).__name__}: {exc}"
            ) from exc

    normalized: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for order, (path_text, frame, strength) in enumerate(raw_conditions, start=1):
        if not path_text:
            raise RuntimeError(f"INT4 visual condition {order} has an empty path.")
        if not (0.0 <= float(strength) <= 4.0):
            raise RuntimeError(
                f"INT4 visual condition strength must be between 0 and 4: {strength}"
            )
        source = Path(path_text).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"INT4 visual condition not found: {source}")

        final_path = source
        original_mode = "unchanged"
        original_size = "unknown"
        if normalize:
            digest_source = f"{source}|{source.stat().st_mtime_ns}|{source.stat().st_size}"
            digest = __import__("hashlib").sha1(
                digest_source.encode("utf-8", "ignore")
            ).hexdigest()[:12]
            target = temp_dir / f"clean_{source.stem}_{digest}.png"
            with Image.open(source) as image:
                original_mode = str(image.mode)
                original_size = f"{image.size[0]}x{image.size[1]}"
                clean = image.convert("RGB")
                clean.save(target, format="PNG", optimize=False)
            final_path = target

        normalized.append(
            {
                "path": str(final_path),
                "frame": int(frame),
                "strength": float(strength),
            }
        )
        rendered.append(
            f"{source.name}:frame={int(frame)}:strength={float(strength):g}:"
            f"mode={original_mode}:size={original_size}"
        )

    args._planner_conditions = normalized
    # Preserve a useful legacy value for diagnostics and any harmless caller
    # inspection, while generation itself reads _planner_conditions.
    args.i2v_image = str(normalized[0]["path"])
    ctx["ltx_int4_input_mode_requested"] = "image-conditioned-video"
    ctx["ltx_int4_visual_condition_count"] = str(len(normalized))
    ctx["ltx_int4_image_normalization"] = "enabled" if normalize else "disabled"
    ctx["ltx_int4_visual_condition_inputs"] = "; ".join(rendered)


class _CudaMonitor:
    def __init__(self, torch_module: Any, path: Path, interval: float, max_events: int) -> None:
        self.torch = torch_module
        self.path = path
        self.interval = max(0.1, float(interval))
        self.max_events = max(1, int(max_events))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.started = time.perf_counter()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("FrameVision isolated INT4 CUDA monitor\n", encoding="utf-8")
        self.thread = threading.Thread(target=self._run, name="ltx-int4-monitor", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        count = 0
        while not self.stop_event.is_set() and count < self.max_events:
            elapsed = time.perf_counter() - self.started
            line = f"{elapsed:10.3f}s | {_cuda_text(self.torch)}"
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:
                pass
            count += 1
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval * 2.0))


def _write_report(path: Path, ctx: Dict[str, Any], exception_text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 78,
        "FrameVision isolated LTX INT4 report",
        "=" * 78,
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {sys.executable}",
        "Native ltx23_vram_lab_cli.py used: NO",
        "INT8 enabled: NO",
        "",
        "INT4 run",
        "-" * 78,
    ]
    preferred = [
        "generation_status", "generation_completed", "total_runtime_s",
        "ltx_int4_model_validation", "ltx_int4_input_mode_requested",
        "requested_vram_profile", "detected_cuda_vram_gb", "resolved_vram_profile_gb",
        "ltx_int4_vram_lab_toggle", "ltx_int4_vram_automation",
        "ltx_sdnq_input_mode", "ltx_sdnq_pipeline_class", "ltx_sdnq_model_root",
        "ltx_sdnq_workflow", "ltx_sdnq_weight_dtype", "ltx_sdnq_offload_policy",
        "ltx_sdnq_memory_policy_input_mode", "ltx_sdnq_stage1_policy_tier",
        "ltx_sdnq_stage2_policy_tier", "ltx_sdnq_requested_num_frames",
        "ltx_sdnq_model_num_frames", "ltx_sdnq_frame_alignment",
        "ltx_sdnq_stage1_equivalent_704_frames",
        "ltx_sdnq_stage2_equivalent_704_frames", "ltx_sdnq_i2v_fast_path_limits",
        "ltx_sdnq_stage2_allocator_guard", "ltx_sdnq_stage2_conservative_policy",
        "ltx_sdnq_stage2_allocator_trim_count", "ltx_sdnq_stage2_allocator_trim_last",
        "ltx_sdnq_stage2_allocator_trim_error", "ltx_sdnq_stage2_step_memory",
        "ltx_sdnq_attention_backend",
        "ltx_sdnq_stage1_offload_policy", "ltx_sdnq_stage2_offload_policy",
        "ltx_sdnq_prompt_sequence_length_requested", "ltx_sdnq_prompt_token_count",
        "ltx_sdnq_prompt_sequence_length", "ltx_sdnq_prompt_padding_tokens_removed",
        "ltx_sdnq_prompt_peak_allocated_gb", "ltx_sdnq_prompt_peak_reserved_gb",
        "ltx_sdnq_stage1_peak_allocated_gb", "ltx_sdnq_stage1_peak_reserved_gb",
        "ltx_sdnq_stage1_min_driver_free_gb", "ltx_sdnq_stage2_peak_allocated_gb",
        "ltx_sdnq_stage2_peak_reserved_gb", "ltx_sdnq_stage2_min_driver_free_gb",
        "ltx_sdnq_peak_allocated_gb", "ltx_sdnq_peak_reserved_gb", "ltx_sdnq_min_driver_free_gb",
        "ltx_sdnq_vram_safety_result", "ltx_sdnq_video_decode_plan",
        "ltx_sdnq_video_decode_chunks", "ltx_sdnq_video_decode_window_frames",
        "ltx_sdnq_video_decode_stride_frames", "ltx_sdnq_video_decode_overlap_frames",
        "ltx_sdnq_video_decode_output_frames", "ltx_sdnq_video_decode_spatial_tiling",
        "ltx_sdnq_video_decode_time_s",
        "ltx_int4_conditioning_mask_guard", "ltx_int4_conditioning_mask_rewrites",
        "ltx_int4_conditioning_mask_last_move", "output_path",
    ]
    emitted = set()
    for key in preferred:
        if key in ctx:
            lines.append(f"{key}: {ctx[key]}")
            emitted.add(key)
    for key in sorted(ctx):
        if key not in emitted and key != "ltx_realtime_step_events":
            lines.append(f"{key}: {ctx[key]}")
    events = ctx.get("ltx_realtime_step_events") or []
    if events:
        lines.extend(["", "Step timings", "-" * 78])
        lines.extend(str(item) for item in events)
    if exception_text:
        lines.extend(["", "Exception", "-" * 78, exception_text.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated FrameVision LTX 2.3 INT4 runner. It does not import or modify the native VRAM Lab CLI."
    )
    parser.add_argument("--pipeline", choices=["one_stage", "two_stages"], default="two_stages")
    parser.add_argument("--model-root", "--sdnq-model-root", dest="sdnq_model_root", required=True)
    parser.add_argument("--vram-profile", choices=["auto", "24", "16", "12", "8"], default="auto")
    planner = parser.add_mutually_exclusive_group()
    planner.add_argument(
        "--int4-auto-vram",
        dest="int4_auto_vram",
        action="store_true",
        help="Enable automatic per-stage INT4 residency planning from detected VRAM, resolution and frames.",
    )
    planner.add_argument(
        "--no-int4-auto-vram",
        dest="int4_auto_vram",
        action="store_false",
        help="Disable workload scaling and use a fixed per-card INT4 residency policy.",
    )
    parser.set_defaults(int4_auto_vram=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--i2v-image", default="")
    parser.add_argument("--i2v-image-frame", type=int, default=0)
    parser.add_argument("--i2v-image-strength", type=float, default=1.0)
    parser.add_argument("--i2v-image-crf", type=int, default=0)
    parser.add_argument(
        "--image",
        dest="image_conditions",
        action="append",
        nargs=3,
        metavar=("PATH", "FRAME", "STRENGTH"),
        default=[],
        help="Add a visual condition at an output frame. Repeat for start/end/reference images.",
    )
    parser.add_argument(
        "--lora",
        dest="loras",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="Load a local LTX LoRA adapter. Repeat to combine adapters.",
    )
    parser.add_argument("--normalize-input-image", dest="normalize_input_image", action="store_true", default=True)
    parser.add_argument("--no-normalize-input-image", dest="normalize_input_image", action="store_false")
    parser.add_argument("--spatial-upsampler-path", default="")
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--audio-start-time", type=float, default=0.0)
    parser.add_argument("--audio-max-duration", type=float, default=None)
    parser.add_argument("--ltx-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--report-path", default="")
    parser.add_argument("--deep-log-path", default="")
    parser.add_argument("--deep-log-interval", type=float, default=1.0)
    parser.add_argument("--deep-log-max-events", type=int, default=4000)
    parser.add_argument("--deep-lifecycle-log", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "native", "efficient", "flash", "sage"],
        default="auto",
    )
    parser.add_argument("--no-boundary-echo", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args._python_executable = sys.executable

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    report_path = Path(args.report_path).expanduser() if args.report_path else (
        Path(args.ltx_root).expanduser() / "tools" / "vram_lab" / "ltx_int4_report.txt"
    )
    deep_path = Path(args.deep_log_path).expanduser() if args.deep_log_path else (
        Path(args.ltx_root).expanduser() / "tools" / "vram_lab" / "ltx_int4_cuda_monitor.txt"
    )
    ctx: Dict[str, Any] = {
        "command": " ".join(__import__("subprocess").list2cmdline([part]) for part in sys.argv),
        "model_root": str(args.sdnq_model_root),
        "requested_pipeline": str(args.pipeline),
        "requested_resolution": f"{args.width}x{args.height}",
        "requested_frames": str(args.num_frames),
        "requested_fps": str(args.frame_rate),
        "output_path": str(args.output_path),
    }
    started = time.perf_counter()
    monitor = None
    exception_text = ""

    try:
        if abs(float(args.shift) - 5.0) > 1e-6:
            raise RuntimeError("The isolated INT4 runner currently uses the distilled sigma schedule; keep Shift at 5.0.")
        if str(args.audio_path or "").strip() and args.pipeline != "two_stages":
            raise RuntimeError("INT4 uploaded reference audio must use the normal two_stages workflow.")
        if args.pipeline == "two_stages" and not str(args.spatial_upsampler_path or "").strip():
            default_up = Path(args.ltx_root).expanduser() / "models" / "ltx23" / "spatial_upsampler" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            args.spatial_upsampler_path = str(default_up)

        _normalize_input_image(args, ctx)
        print("[ltx-status] Loading isolated INT4 runtime", flush=True)
        import torch
        ctx["torch_version"] = str(torch.__version__)
        ctx["cuda_available"] = str(bool(torch.cuda.is_available()))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected .ltx23 environment.")

        requested_profile = str(args.vram_profile)
        detected_total_gb = _detect_vram_total_gb(torch)
        if requested_profile.lower() == "auto":
            args.vram_profile = str(_detect_vram_profile(torch))
        ctx["requested_vram_profile"] = requested_profile
        ctx["detected_cuda_vram_gb"] = f"{detected_total_gb:.2f}"
        ctx["resolved_vram_profile_gb"] = str(args.vram_profile)
        ctx["ltx_int4_vram_lab_toggle"] = "ON" if bool(args.int4_auto_vram) else "OFF"

        if args.deep_lifecycle_log:
            monitor = _CudaMonitor(torch, deep_path, args.deep_log_interval, args.deep_log_max_events)
            monitor.start()
            ctx["cuda_monitor_path"] = str(deep_path)

        result = run_sdnq_diffusers(args, ctx, torch)
        ctx["result"] = json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False)
        ctx["generation_status"] = "completed"
        ctx["generation_completed"] = "YES"
        return_code = 0
    except Exception as exc:
        exception_text = traceback.format_exc()
        ctx["generation_status"] = f"failed: {type(exc).__name__}: {exc}"
        ctx["generation_completed"] = "NO"
        print(exception_text, file=sys.stderr, flush=True)
        return_code = 1
    finally:
        if monitor is not None:
            monitor.stop()
        ctx["total_runtime_s"] = f"{time.perf_counter() - started:.3f}"
        try:
            _write_report(report_path, ctx, exception_text)
            print(f"[ltx-status] INT4 report saved: {report_path}", flush=True)
        except Exception as report_exc:
            print(f"[ltx-warning] Could not write INT4 report: {type(report_exc).__name__}: {report_exc}", file=sys.stderr, flush=True)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
