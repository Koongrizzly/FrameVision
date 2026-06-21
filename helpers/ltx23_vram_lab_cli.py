from __future__ import annotations

"""FrameVision VRAM Lab wrapper for native LTX 2.3.

Thin integration wrapper only. It keeps the LTX repo untouched, runs the selected
official ``ltx_pipelines`` module in-process, and records VRAM Lab
boundary telemetry/reporting. The default remains the original one-stage path.
"""

import argparse
import math
import gc
import importlib
import importlib.util
import hashlib
import json
import os
import re
import runpy
import shutil
import sys
import time
import threading
from datetime import datetime
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE = Path(__file__).resolve()
HELPERS_DIR = THIS_FILE.parent
APP_ROOT = HELPERS_DIR.parent
VRAM_LAB_DIR = APP_ROOT / "tools" / "vram_lab"
REPORT_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_PATH = VRAM_LAB_DIR / f"ltx_vram_lab_integration_report_{REPORT_STAMP}.txt"
DEEP_LIVE_LOG_PATH = VRAM_LAB_DIR / f"ltx_deep_lifecycle_{REPORT_STAMP}.txt"

_TIMESTAMP_RE = re.compile(r"(?:^|[_-])\d{8}[_-]\d{6}(?:$|[_-])")


def _path_has_timestamp(path: Path) -> bool:
    try:
        return bool(_TIMESTAMP_RE.search(path.stem))
    except Exception:
        return False


def _dated_output_path(path: Path, *, stamp: str | None = None, force: bool = False) -> Path:
    """Return a dated output path so LTX reports/logs no longer overwrite old runs."""
    stamp = stamp or REPORT_STAMP
    try:
        path = Path(path)
        if force or not _path_has_timestamp(path):
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix or '.txt'}")
        # If two jobs start in the same second, keep both reports instead of overwriting.
        candidate = path
        idx = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix or '.txt'}")
            idx += 1
        return candidate
    except Exception:
        return path

DEFAULT_LTX_ROOT = Path(r"C:\ltx23")
DEFAULT_PYTHON = DEFAULT_LTX_ROOT / "environments" / ".ltx23" / "python.exe"
DEFAULT_CHECKPOINT = DEFAULT_LTX_ROOT / "models" / "ltx23" / "distilled-1.1" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_GEMMA_ROOT = DEFAULT_LTX_ROOT / "models" / "ltx23" / "text_encoder" / "lightricks_gemma_original"
DEFAULT_OUTPUT = DEFAULT_LTX_ROOT / "output" / "ltx_test" / "ltx_t2v_25f_vram_lab.mp4"
DEFAULT_PROMPT = "A tiny silver robot walks through a neon spaceship corridor, cinematic lighting, smooth motion, highly detailed."

# FrameVision LTX VRAM profiles.
# 24 GB uses the current proven native LTX 2.3 / VRAM Lab CFG 2 profile.
# Lower VRAM profiles reduce both the main/distilled denoiser hot-window
# and the Gemma/text-encoder streaming VRAM target. Gemma must not keep
# using the 24 GB target on 12/16 GB cards.
LTX_VRAM_PROFILE_BASE_GB = 24
LTX_VRAM_PROFILE_MAIN_HOT_WINDOW_24GB = 16.5
LTX_VRAM_PROFILE_MAIN_I2V_SAFETY_DERATE_GB = 3.0
LTX_VRAM_PROFILE_GEMMA_TARGET_24GB = 20.2
LTX_VRAM_PROFILE_GEMMA_TARGET_16GB = 5.5
LTX_VRAM_PROFILE_GEMMA_TARGET_12GB = 1.5
LTX_VRAM_PROFILE_HOT_PIN_24GB = 6.0
LTX_VRAM_PROFILE_HOT_PIN_16GB = 5.5
LTX_VRAM_PROFILE_HOT_PIN_12GB = 1.5
LTX_VRAM_PROFILE_DISK_SLOTS_24GB = 6
LTX_VRAM_PROFILE_DISK_SLOTS_16GB = 6
LTX_VRAM_PROFILE_DISK_SLOTS_12GB = 6
LTX_VRAM_PROFILES = {
    24: {
        "main_hot_window_gb": 16.5,
        "stage2_hot_window_gb": 12.5,
        "emergency_free_vram_floor_gb": 0.5,
        "stage1_stable_hotset_fraction": 1.15,
        "stage2_stable_hotset_fraction": 0.9,
        "gemma_target_vram_gb": LTX_VRAM_PROFILE_GEMMA_TARGET_24GB,
        "gemma_hot_pin_gb": LTX_VRAM_PROFILE_HOT_PIN_24GB,
        "gemma_disk_slots": LTX_VRAM_PROFILE_DISK_SLOTS_24GB,
        "delta_from_24gb": 0.0,
        "main_safety_derate_gb": LTX_VRAM_PROFILE_MAIN_I2V_SAFETY_DERATE_GB,
        "note": "24 GB profile: main/distilled denoiser hot-window 16.5 GB; Stage 2/refine 12.5 GB; Gemma target 20.2 GB",
    },
    16: {
        "main_hot_window_gb": 8.0,
        "stage2_hot_window_gb": 4.0,
        "emergency_free_vram_floor_gb": 0.5,
        "gemma_target_vram_gb": LTX_VRAM_PROFILE_GEMMA_TARGET_16GB,
        "gemma_hot_pin_gb": LTX_VRAM_PROFILE_HOT_PIN_16GB,
        "gemma_disk_slots": LTX_VRAM_PROFILE_DISK_SLOTS_16GB,
        "delta_from_24gb": 8.0,
        "main_safety_derate_gb": LTX_VRAM_PROFILE_MAIN_I2V_SAFETY_DERATE_GB,
        "note": "16 GB profile: main/distilled denoiser hot-window 8.0 GB; Gemma target 5.5 GB",
    },
    12: {
        "main_hot_window_gb": 4.4,
        "stage2_hot_window_gb": 2.0,
        "emergency_free_vram_floor_gb": 0.5,
        "gemma_target_vram_gb": LTX_VRAM_PROFILE_GEMMA_TARGET_12GB,
        "gemma_hot_pin_gb": LTX_VRAM_PROFILE_HOT_PIN_12GB,
        "gemma_disk_slots": LTX_VRAM_PROFILE_DISK_SLOTS_12GB,
        "delta_from_24gb": 12.0,
        "main_safety_derate_gb": LTX_VRAM_PROFILE_MAIN_I2V_SAFETY_DERATE_GB,
        "note": "12 GB profile: main/distilled denoiser hot-window 4.4 GB; Gemma target 1.5 GB",
    },
}


# CLI-side Auto VRAM workflow automation.
#
# The UI has had tuned Auto values for resolution/frame-count dependent LTX runs.
# Other FrameVision tools can call this CLI directly and may only pass
# "--vram-profile auto".  Keep the same workflow automation here so Auto means:
#   profile selection + main hot-window + Stage 2 block limit + staged hotset defaults.
# This prevents callers outside ltx23_ui.py from getting only the profile number
# while silently falling back to older static block-size defaults.
#
# Important: one-stage LTX behaves closer to the two-stage Stage-2/refine
# pressure profile because it effectively combines the first/refine workload.
# Therefore one-stage Auto intentionally uses the Stage-2 anchor value for
# both main hotset and Stage-1 block size.  Do not reuse the two-stage Stage-1
# value for one-stage workflows.
LTX_AUTO_STAGE2_SURVIVAL_FLOOR_GB = 1.5
LTX_AUTO_HOTSET_SURVIVAL_FLOOR_GB = 1.5
LTX_AUTO_RESOLUTION_ANCHORS = {
    "480p": {
        "label": "480p",
        "size": "832x512",
        # Per-profile Auto ceilings for 480p.
        "max_frames_24gb": 1201,
        "max_frames_16gb": 901,
        "max_frames_12gb": 601,
        "max_frames_default": 481,
        "anchors": [
            (121, 21.0, 21.0, 18.5),
            (241, 19.5, 19.5, 17.0),
            (481, 16.5, 16.5, 14.5),
            (601, 15.5, 15.5, 12.5),
            (721, 14.5, 14.5, 11.0),
            (901, 13.0, 13.0, 9.5),
            (1140, 11.0, 11.0, 7.0),
            (1201, 11.0, 11.0, 1.5),
        ],
    },
    "704p": {
        "label": "704p",
        "size": "1280x704",
        "max_frames_24gb": 577,
        "max_frames_16gb": 433,
        "max_frames_12gb": 289,
        "anchors": [
            (24, 20.3, 20.3, 19.0),
            (73, 19.0, 19.0, 17.5),
            (121, 18.0, 18.0, 16.0),
            (169, 18.0, 18.0, 15.0),
            (217, 17.5, 17.5, 14.0),
            (241, 17.5, 17.5, 13.0),
            (265, 17.0, 17.0, 13.0),
            (289, 17.0, 17.0, 12.5),
            (313, 17.0, 17.0, 12.5),
            (361, 16.0, 16.0, 10.8),
            (409, 15.0, 15.0, 9.7),
            (433, 14.5, 14.5, 9.5),
            (481, 14.0, 14.0, 6.5),
            (521, 13.5, 13.5, 5.5),
            (577, 13.0, 13.0, 3.0),
        ],
    },
    "1088p": {
        "label": "1088p",
        "size": "1920x1088",
        "max_frames_24gb": 265,
        "max_frames_16gb": 199,
        "max_frames_12gb": 133,
        "anchors": [
            (73, 18.0, 18.0, 15.5),
            (121, 17.0, 17.0, 13.0),
            (133, 16.8, 16.8, 12.5),
            (185, 15.3, 15.3, 10.5),
            (199, 15.0, 15.0, 9.5),
            (265, 13.2, 13.2, 1.5),
        ],
    },
}


def _is_auto_vram_profile_value(value: Any) -> bool:
    raw = str(value or os.environ.get("FRAMEVISION_LTX_VRAM_PROFILE_GB", "") or "").strip().lower()
    return raw in {"auto", "detect", "gpu"}


def _ltx_auto_resolution_key(width: int, height: int) -> str:
    try:
        short_side = min(int(width), int(height))
    except Exception:
        short_side = 704
    if short_side <= 576:
        return "480p"
    if short_side <= 800:
        return "704p"
    return "1088p"


def _interpolate_ltx_auto_anchor_value(anchors: List[Tuple[int, float, float, float]], frame_count: int) -> Tuple[Tuple[int, float, float, float], Tuple[int, float, float, float], Tuple[float, float, float]]:
    frames = int(frame_count)
    ordered = sorted(anchors, key=lambda item: int(item[0]))
    if not ordered:
        raise ValueError("No Auto VRAM anchors configured.")
    if frames <= int(ordered[0][0]):
        a = ordered[0]
        return a, a, (float(a[1]), float(a[2]), float(a[3]))
    for a, b in zip(ordered, ordered[1:]):
        fa, fb = int(a[0]), int(b[0])
        if frames == fa:
            return a, a, (float(a[1]), float(a[2]), float(a[3]))
        if frames == fb:
            return b, b, (float(b[1]), float(b[2]), float(b[3]))
        if fa < frames < fb:
            ratio = (frames - fa) / float(fb - fa)
            return a, b, (
                float(a[1]) + (float(b[1]) - float(a[1])) * ratio,
                float(a[2]) + (float(b[2]) - float(a[2])) * ratio,
                float(a[3]) + (float(b[3]) - float(a[3])) * ratio,
            )
    a = ordered[-1]
    return a, a, (float(a[1]), float(a[2]), float(a[3]))


def _ltx_auto_profile_subtract_gb(vram_profile_gb: int) -> float:
    if int(vram_profile_gb) >= 24:
        return 0.0
    if int(vram_profile_gb) >= 16:
        return 8.0
    return 12.0


def _ltx_auto_scaled_values_at_frame(anchors: List[Tuple[int, float, float, float]], vram_profile_gb: int, frame_count: int, workflow_group: str = "two_stage") -> Tuple[float, float, float]:
    _lower, _upper, values = _interpolate_ltx_auto_anchor_value(anchors, int(frame_count))
    subtract = _ltx_auto_profile_subtract_gb(int(vram_profile_gb))
    two_stage_hotset = max(0.0, float(values[0]) - subtract)
    two_stage_stage1 = two_stage_hotset
    two_stage_stage2 = max(0.0, float(values[2]) - subtract)
    if str(workflow_group).strip().lower() == "one_stage":
        # One-stage combines pressure that behaves closer to the two-stage
        # refine/Stage-2 limit.  Use the Stage-2 value as the whole one-stage
        # budget so external callers using --vram-profile auto get the same
        # stable workflow as the UI.
        return two_stage_stage2, two_stage_stage2, two_stage_stage2
    return two_stage_hotset, two_stage_stage1, two_stage_stage2


def _ltx_auto_survival_value_at_frame(anchors: List[Tuple[int, float, float, float]], vram_profile_gb: int, frame_count: int, workflow_group: str) -> float:
    hotset, _stage1, stage2 = _ltx_auto_scaled_values_at_frame(anchors, int(vram_profile_gb), int(frame_count), workflow_group)
    return hotset if str(workflow_group).strip().lower() == "one_stage" else stage2


def _ltx_auto_survival_floor_for_workflow(workflow_group: str, vram_profile_gb: int | None = None) -> float:
    workflow = str(workflow_group).strip().lower()
    if workflow == "one_stage":
        return LTX_AUTO_HOTSET_SURVIVAL_FLOOR_GB
    try:
        profile = int(float(vram_profile_gb or 24))
    except Exception:
        profile = 24
    if profile < 16:
        return 0.5
    return LTX_AUTO_STAGE2_SURVIVAL_FLOOR_GB


def _ltx_auto_hard_max_frames_for_profile(table: Dict[str, Any], anchors: List[Tuple[int, float, float, float]], vram_profile_gb: int) -> int:
    if int(vram_profile_gb) >= 24:
        return int(table.get("max_frames_24gb", anchors[-1][0]))
    if int(vram_profile_gb) >= 16:
        return int(table.get("max_frames_16gb", table.get("max_frames_default", min(table.get("max_frames_24gb", anchors[-1][0]), 481))))
    return int(table.get("max_frames_12gb", table.get("max_frames_default", min(table.get("max_frames_24gb", anchors[-1][0]), 481))))


def _ltx_auto_max_frames_for_profile(anchors: List[Tuple[int, float, float, float]], vram_profile_gb: int, hard_max_frames: int, workflow_group: str = "two_stage") -> int:
    ordered = sorted(anchors, key=lambda item: int(item[0]))
    if not ordered:
        return 0
    floor = _ltx_auto_survival_floor_for_workflow(workflow_group, vram_profile_gb)
    first_frame = int(ordered[0][0])
    first_value = _ltx_auto_survival_value_at_frame(ordered, int(vram_profile_gb), first_frame, workflow_group)
    if first_value < floor:
        return 0
    max_frames = min(int(hard_max_frames), int(ordered[-1][0]))
    best = min(first_frame, max_frames)
    for frame in range(1, max_frames + 1):
        if _ltx_auto_survival_value_at_frame(ordered, int(vram_profile_gb), frame, workflow_group) >= floor:
            best = frame
        elif frame >= first_frame:
            break
    return int(best)


def _calculate_ltx_auto_vram_settings(vram_profile_gb: int, width: int, height: int, frame_count: int, workflow_group: str = "two_stage") -> Dict[str, Any]:
    try:
        profile = int(float(vram_profile_gb))
    except Exception:
        profile = 24
    profile = 24 if profile >= 24 else 16 if profile >= 16 else 12
    workflow = "one_stage" if str(workflow_group).strip().lower() == "one_stage" else "two_stage"
    workflow_label = "one-stage" if workflow == "one_stage" else "two-stage"
    key = _ltx_auto_resolution_key(int(width), int(height))
    table = LTX_AUTO_RESOLUTION_ANCHORS.get(key) or LTX_AUTO_RESOLUTION_ANCHORS["704p"]
    anchors = list(table["anchors"])
    hard_max = _ltx_auto_hard_max_frames_for_profile(table, anchors, profile)
    max_frames = _ltx_auto_max_frames_for_profile(anchors, profile, hard_max, workflow)
    # Per-profile frame ceilings are user-facing limits, not only survival
    # suggestions from the anchor scan. Keep the full hard max so the tuned
    # 12/16/24 GB caps remain reachable without the old generic 481-frame gate
    # blocking the run.
    max_frames = max(int(max_frames), int(hard_max))
    frames = int(frame_count)
    if max_frames <= 0:
        return {
            "supported": False,
            "reason": f"Auto {profile} GB / {table['label']} / {workflow_label}: this resolution is unsupported for this VRAM profile. Use a lower resolution or larger VRAM profile.",
            "hotset_gb": 0.0,
            "stage1_gb": 0.0,
            "stage2_gb": 0.0,
            "max_frames_for_profile_resolution": 0,
            "source_anchors_used": [],
            "resolution_label": table["label"],
            "vram_profile_gb": profile,
            "workflow_group": workflow,
        }
    if frames > max_frames:
        return {
            "supported": False,
            "reason": f"Auto {profile} GB / {table['label']} / {workflow_label}: selected frame count is too high. Lower frames to {max_frames} or less, lower resolution, or use a larger VRAM profile.",
            "hotset_gb": 0.0,
            "stage1_gb": 0.0,
            "stage2_gb": 0.0,
            "max_frames_for_profile_resolution": max_frames,
            "source_anchors_used": [],
            "resolution_label": table["label"],
            "vram_profile_gb": profile,
            "workflow_group": workflow,
        }
    lower, upper, _base_values = _interpolate_ltx_auto_anchor_value(anchors, frames)
    hotset, stage1, stage2 = _ltx_auto_scaled_values_at_frame(anchors, profile, frames, workflow)
    survival_value = hotset if workflow == "one_stage" else stage2
    floor = _ltx_auto_survival_floor_for_workflow(workflow, profile)
    if survival_value < floor:
        return {
            "supported": False,
            "reason": f"Auto {profile} GB / {table['label']} / {workflow_label}: selected frame count is too high. Lower frames, lower resolution, or use a larger VRAM profile.",
            "hotset_gb": round(hotset, 1),
            "stage1_gb": round(stage1, 1),
            "stage2_gb": round(stage2, 1),
            "max_frames_for_profile_resolution": max_frames,
            "source_anchors_used": [lower, upper],
            "resolution_label": table["label"],
            "vram_profile_gb": profile,
            "workflow_group": workflow,
        }
    return {
        "supported": True,
        "reason": "",
        "hotset_gb": round(hotset, 1),
        "stage1_gb": round(stage1, 1),
        "stage2_gb": round(stage2, 1),
        "max_frames_for_profile_resolution": max_frames,
        "source_anchors_used": [lower, upper],
        "resolution_label": table["label"],
        "vram_profile_gb": profile,
        "workflow_group": workflow,
    }


def _apply_ltx_cli_auto_workflow_profile(
    args: argparse.Namespace,
    profile_gb: int,
    profile: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Apply full CLI-side Auto settings when --vram-profile auto is used.

    This is intentionally in the CLI, not only the UI, so Planner/Music Clip
    Creator/other helpers can request Auto and get the same working workflow.
    Explicit caller overrides still win:
      - --main-hot-window-gb overrides the Auto main hot-window
      - --stage2-block-size-limit-gb overrides the Auto Stage 2 limit
      - staged hotset fraction flags override the profile defaults
    """
    if not _is_auto_vram_profile_value(getattr(args, "vram_profile", None)):
        return None
    try:
        width = int(getattr(args, "width", 1280) or 1280)
        height = int(getattr(args, "height", 704) or 704)
        frames = int(getattr(args, "num_frames", 241) or 241)
    except Exception:
        width, height, frames = 1280, 704, 241
    pipeline = str(getattr(args, "pipeline", "") or "").strip().lower()
    workflow = "one_stage" if pipeline == "one_stage" else "two_stage"
    auto = _calculate_ltx_auto_vram_settings(int(profile_gb), width, height, frames, workflow)
    profile["auto_workflow_settings"] = auto
    profile["auto_workflow_enabled"] = True
    if not bool(auto.get("supported", False)):
        profile["note"] = str(profile.get("note", "selected LTX VRAM profile")) + f"; Auto workflow fallback: {auto.get('reason', 'unsupported settings')}"
        return auto

    hotset = float(auto.get("hotset_gb", 0.0) or 0.0)
    stage1 = float(auto.get("stage1_gb", hotset) or hotset)
    stage2 = float(auto.get("stage2_gb", 0.0) or 0.0)

    # Auto owns the workflow block sizes unless the caller explicitly passed
    # the corresponding override. Other apps often only send --vram-profile auto.
    if getattr(args, "main_hot_window_gb", None) is None and hotset > 0.0:
        profile["main_hot_window_gb"] = hotset
        profile["main_hot_window_override_gb"] = hotset
    if getattr(args, "stage2_block_size_limit_gb", None) is None and stage2 > 0.0:
        profile["stage2_hot_window_gb"] = stage2
    profile.setdefault("stage1_stable_hotset_fraction", 1.15)
    profile.setdefault("stage2_stable_hotset_fraction", 0.9)
    profile["note"] = (
        str(profile.get("note", "selected LTX VRAM profile"))
        + f"; Auto workflow {auto.get('resolution_label')} {frames}f {workflow}: "
          f"main/hotset {hotset:.1f} GB, Stage 1 {stage1:.1f} GB, Stage 2 {stage2:.1f} GB"
    )
    return auto



PIPELINE_MODULES = {
    "one_stage": "ltx_pipelines.ti2vid_one_stage",
    "two_stages": "ltx_pipelines.ti2vid_two_stages",
    "two_stages_hq": "ltx_pipelines.ti2vid_two_stages_hq",
    "a2vid_two_stage": "ltx_pipelines.a2vid_two_stage",
}



def _install_ltx_runtime_warning_filters(ctx: Dict[str, Any] | None = None) -> None:
    """Silence known harmless LTX runtime warnings without hiding real errors.

    Keep this wrapper-side only. These warnings are either caused by our
    in-process runpy wrapper, by Transformers advisory logging, or by a CUDA
    allocator option that Windows/PyTorch reports as unsupported.
    """
    applied: list[str] = []
    try:
        # Transformers emits this as an advisory log/warning. It does not change
        # generation unless the repo explicitly asks for use_fast=True/False.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        applied.append("TRANSFORMERS_NO_ADVISORY_WARNINGS=1")
    except Exception:
        pass
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*found in sys\.modules after import of package 'ltx_pipelines'.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Using a slow image processor as `use_fast` is unset.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*expandable_segments not supported on this platform.*",
            category=UserWarning,
        )
        applied.append("python warnings filters")
    except Exception:
        pass
    if ctx is not None:
        ctx["ltx_warning_filters"] = ", ".join(applied) if applied else "none"


def _apply_allocator_config_for_platform(ctx: Dict[str, Any]) -> None:
    """Request CUDA allocator options only when they are useful/supported.

    expandable_segments throws a repeated warning on Windows in this Torch build,
    so don't request it there. This is better than only hiding the warning.
    """
    try:
        platform_name = str(sys.platform or "").lower()
        existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if platform_name.startswith("win"):
            if "expandable_segments" in existing:
                kept = [part for part in existing.split(",") if "expandable_segments" not in part]
                if kept:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(kept)
                    ctx["allocator_config_requested"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
                    ctx["allocator_config_note"] = "removed unsupported expandable_segments on Windows; kept remaining allocator config"
                else:
                    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                    ctx["allocator_config_requested"] = "not set"
                    ctx["allocator_config_note"] = "skipped unsupported expandable_segments on Windows"
            else:
                ctx["allocator_config_requested"] = existing or "not set"
                ctx["allocator_config_note"] = "Windows: expandable_segments not requested"
        else:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            ctx["allocator_config_requested"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "n/a")
            ctx["allocator_config_note"] = "non-Windows: expandable_segments requested"
    except Exception as exc:
        ctx["allocator_config_requested"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "n/a")
        ctx["allocator_config_note"] = f"allocator config setup failed: {type(exc).__name__}: {exc}"

# Wrapper-side image inputs that should be normalized before LTX sees them.
# Keep this outside the LTX repo: unusual EXIF/modes/alpha/metadata should not
# be allowed to crash the native process before the wrapper can explain what happened.
LTX_SINGLE_IMAGE_PATH_OPTIONS = {
    "--image",
    "--image-path",
    "--image_path",
    "--input-image",
    "--input_image",
    "--input-media-path",
    "--input_media_path",
    "--start-image",
    "--start_image",
    "--end-media-path",
    "--end_media_path",
    "--image-end-path",
    "--image_end_path",
    "--last-frame-path",
    "--last_frame_path",
}
LTX_MULTI_IMAGE_PATH_OPTIONS = {
    "--conditioning-media-paths",
    "--conditioning_media_paths",
    "--reference-images",
    "--reference_images",
}


def _fmt_bytes(n: int | None) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if abs(v) < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}" if u != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{n} B"


def _format_bytes(n: int | None) -> str:
    return _fmt_bytes(n)


def _cuda_snapshot(torch_module: Any | None = None) -> str:
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            alloc = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            free = total = 0
            try:
                free, total = torch.cuda.mem_get_info()
            except Exception:
                pass
            return (
                f"allocated={_fmt_bytes(alloc)}, reserved={_fmt_bytes(reserved)}, "
                f"driver_free={_fmt_bytes(int(free))}, driver_total={_fmt_bytes(int(total))}"
            )
    except Exception as exc:
        return f"n/a ({type(exc).__name__}: {exc})"
    return "n/a"



def _latent_preview_sidecar_write(ctx: Dict[str, Any] | None, event: Dict[str, Any]) -> None:
    """Append a tiny JSONL event for FrameVision's LTX Latent Preview strip.

    This is intentionally wrapper-side. The native LTX pipeline is still left
    untouched; these events let the UI update during direct runs and queued runs
    where stdout belongs to the global worker instead of the LTX widget.
    """
    try:
        if ctx is None:
            return
        if str(ctx.get("latent_preview_enabled", "NO")).upper() != "YES":
            return
        path_text = str(ctx.get("latent_preview_sidecar", "") or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event or {})
        payload.setdefault("time", time.time())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# LTX 2.x latent RGB preview factors from the Comfy/KJ LTX preview node.
# Shape: 128 latent channels -> RGB. Used only for rough UI previews; final output still uses the official decoder.
_FRAMEVISION_LTX2_LATENT_RGB_FACTORS = [[0.002269406570121646, -0.02110900916159153, -0.009850316680967808], [-0.016038373112678528, -0.012462412007153034, -0.01112896017730236], [0.025274179875850677, 0.011209743097424507, 0.025426799431443214], [0.04690725728869438, 0.041542328894138336, 0.03568895906209946], [-0.02388044260442257, -0.0018645941745489836, 0.01858334057033062], [0.03720448538661003, 0.0220357533544302, 0.027937663719058037], [-0.07273884862661362, -0.09326262027025223, -0.11579664051532745], [-0.063837431371212, 0.00026216846890747547, 0.03042735904455185], [0.02903873845934868, 0.042082373052835464, 0.030649805441498756], [0.03777873516082764, 0.0322984978556633, -0.005671461578458548], [-0.0075670829974114895, -0.012113905511796474, -0.01638956367969513], [0.026524530723690987, 0.060518112033605576, 0.059549521654844284], [0.10093028098344803, 0.10073262453079224, 0.0505094900727272], [0.03725508227944374, 0.015382086858153343, 0.005786076188087463], [-0.03139607608318329, -0.01690264232456684, -0.0013519978383556008], [-0.027200624346733093, -0.02517341822385788, -0.008874989114701748], [0.024963486939668655, 0.04293748363852501, 0.05582639202475548], [-0.0364827960729599, -0.026975594460964203, -0.021950015798211098], [0.027655167505145073, 0.025136707350611687, 0.043967027217149734], [0.035822272300720215, 0.013104500249028206, 0.01113432738929987], [0.05353763327002525, 0.013606574386358261, -0.018720127642154694], [-0.013587888330221176, -0.01689346879720688, -0.027842802926898003], [0.059415675699710846, 0.03734271228313446, 0.04562298208475113], [-0.02946414425969124, -0.038338612765073776, 0.001805233070626855], [0.03921474143862724, 0.0651894062757492, 0.10681862384080887], [-0.00744189927354455, 0.007951526902616024, 0.020728807896375656], [-0.04038553684949875, -0.05215264856815338, -0.07213657349348068], [-0.004655141849070787, 0.01305423304438591, 0.026104029268026352], [0.03434251993894577, 0.018448110669851303, 0.013096392154693604], [0.0022075253073126078, -0.0011812079465016723, 0.0002940484555438161], [-0.00043441299931146204, 0.02366728149354458, 0.035889431834220886], [-0.030657343566417694, -0.024926183745265007, -0.012355240061879158], [-0.018955843523144722, -0.017360301688313484, -0.008214764297008514], [-0.01113052573055029, -0.01201171800494194, -0.002986249281093478], [0.018902746960520744, 0.01758778840303421, 0.026414571329951286], [-0.019977254793047905, -0.01605399139225483, -0.019136475399136543], [-0.00300968368537724, -0.017609693109989166, -0.013655650429427624], [0.0022096361499279737, 0.017998533323407173, 0.01815750263631344], [0.05186990648508072, 0.03285299986600876, 0.016072165220975876], [0.012626334093511105, 0.0013884707586839795, -0.012077193707227707], [-0.0037861645687371492, -0.013902144506573677, -0.01911942847073078], [-0.014163163490593433, -0.00513274222612381, -0.014303527772426605], [-0.010461323894560337, 0.009658926166594028, 0.01644069515168667], [-0.008665377274155617, 0.002501955023035407, -0.009703717194497585], [-0.03404829278588295, -0.02546044997870922, -0.014914450235664845], [0.04997691139578819, 0.06592527031898499, 0.073111392557621], [0.027394814416766167, 0.024555068463087082, 0.019957970827817917], [-0.027501430362462997, -0.01673700101673603, -0.03089248389005661], [-0.018696032464504242, -0.0020940247923135757, 0.015244065783917904], [-0.0062704551964998245, -0.0067006442695856094, -0.007532030809670687], [0.014871004968881607, 0.009914354421198368, 0.020960720255970955], [0.03662937879562378, 0.04413224756717682, 0.04220828413963318], [-0.011242181062698364, -0.013539309613406658, -0.016438307240605354], [-0.014854325912892818, 0.0038217694964259863, -0.002461288822814822], [-0.014826249331235886, 0.0009719038498587906, -0.012078499421477318], [-0.029396841302514076, -0.01432017982006073, 0.013018904253840446], [0.02755064144730568, 0.028369395062327385, 0.01640605367720127], [0.12049165368080139, 0.1395745575428009, 0.14566579461097717], [0.019721267744898796, 0.009739740751683712, 0.0023876908235251904], [-0.007320966571569443, 0.0065013207495212555, 0.01603059470653534], [0.007391378283500671, -0.0073603675700724125, -0.01770283281803131], [0.02984853833913803, 0.012391146272420883, 0.010563627816736698], [-0.013479884713888168, -0.008637298829853535, -0.013457189314067364], [0.04127075523138046, 0.03032625839114189, 0.024770958349108696], [-0.06524652987718582, -0.012209279462695122, 0.02087211236357689], [-0.1179763451218605, -0.060323599725961685, -0.07592175155878067], [-0.07122819870710373, -0.04385707899928093, -0.022124603390693665], [-0.04682473465800285, -0.022610662505030632, -0.010107148438692093], [-0.0054328180849552155, -0.010368981398642063, -0.008167334832251072], [0.029181398451328278, 0.030588403344154358, 0.028090540319681168], [0.016619984060525894, 0.004931286443024874, -0.006450849585235119], [0.01035264041274786, 0.002237115055322647, 0.0013903985964134336], [-0.04313831403851509, -0.061772625893354416, -0.08946335315704346], [0.0150345079600811, 0.007781678810715675, 0.0011013159528374672], [-0.013585779815912247, 0.008117705583572388, 0.020367907360196114], [-0.172962948679924, -0.16406646370887756, -0.1668281853199005], [0.0083833709359169, 0.0015236001927405596, -0.01731627807021141], [0.021939430385828018, 0.018004458397626877, 0.014768349006772041], [0.008083095774054527, -0.013463049195706844, -0.022061636671423912], [0.024328550323843956, 0.0128010343760252, 0.014966367743909359], [0.05850301682949066, 0.027980001643300056, 0.02225641906261444], [0.09690416604280472, 0.06929530203342438, 0.03253814950585365], [0.048208240419626236, 0.025294817984104156, 0.023508133366703987], [-0.026432134211063385, -0.040383171290159225, -0.03950457274913788], [-0.021598653867840767, -0.017070941627025604, -0.010933087207376957], [0.011645167134702206, 0.002806191798299551, 0.003779367310926318], [0.10478592664003372, 0.08954174816608429, 0.06555330753326416], [0.015151776373386383, -0.016160616651177406, -0.024905217811465263], [0.019659176468849182, 0.008487952873110771, 0.002426224760711193], [-0.05173315480351448, -0.026337839663028717, -0.02127116546034813], [0.016987523064017296, 0.006270893849432468, 0.0015798212261870503], [0.007938026450574398, -0.005250005517154932, -0.020408453419804573], [0.013017759658396244, 0.01654384844005108, 0.04163840040564537], [-0.009886542335152626, -0.026848411187529564, -0.03070281818509102], [0.01108171883970499, 0.01827266439795494, -0.007332107983529568], [-0.0285995751619339, -0.031727731227874756, -0.03370537981390953], [0.005299570970237255, 0.05678633600473404, 0.02825017087161541], [-0.055322226136922836, -0.09084303677082062, -0.12999044358730316], [0.01844066195189953, 0.031044499948620796, 0.021148500964045525], [-0.004471115302294493, 0.005830412730574608, 0.00911418441683054], [-0.04053843766450882, -0.016424428671598434, -0.0010634599020704627], [0.03858831524848938, 0.007309338077902794, -0.005618985276669264], [0.01423253770917654, -0.0055681923404335976, 3.394074519746937e-05], [0.11455483734607697, 0.14653916656970978, 0.1488018035888672], [-0.005231931805610657, -0.0033921014983206987, -0.000995257287286222], [0.01449565589427948, 0.019586293026804924, 0.04565812274813652], [-0.005179048050194979, -0.011201606132090092, -0.0008710073889233172], [-0.015361929312348366, 0.00778581015765667, -0.008238887414336205], [-0.1147838830947876, -0.09109023958444595, -0.050579313188791275], [0.09037500619888306, 0.09597006440162659, 0.10811734944581985], [0.001873677596449852, -0.01772197335958481, -0.07681205868721008], [-0.020383257418870926, -0.016072455793619156, -0.01077069528400898], [-0.060444317758083344, -0.05499502643942833, -0.06153025105595589], [-0.016717270016670227, 0.026493264362215996, 0.021835654973983765], [0.008203534409403801, 0.00418612826615572, 0.013867748901247978], [0.0789225772023201, 0.05467747151851654, 0.016568133607506752], [-0.15149451792240143, -0.1526806503534317, -0.14325062930583954], [0.00538366474211216, 0.010192245244979858, -0.00449327751994133], [-0.004906965419650078, -0.005569908302277327, -0.02096559666097164], [0.024530155584216118, 0.010962833650410175, 0.0034586559049785137], [0.03551010414958, 0.017310436815023422, 0.007064413744956255], [0.11111932247877121, 0.09825586527585983, 0.08827318251132965], [-0.051722846925258636, -0.047595202922821045, -0.03763044252991676], [-0.02975175902247429, -0.02153967320919037, -0.021425534039735794], [-0.03462936729192734, -0.025198571383953094, -0.017322326079010963], [-0.016921017318964005, -0.012419789098203182, -0.0154880927875638], [-0.08035065978765488, -0.08451078832149506, -0.09623870998620987], [-0.03870908170938492, -0.04211008921265602, -0.04383759945631027]]
_FRAMEVISION_LTX2_LATENT_RGB_BIAS = [-0.6957847476005554, -0.7276281118392944, -0.7405748963356018]

def _latent_preview_output_dir(ctx: Dict[str, Any] | None) -> Path | None:
    try:
        if ctx is None or str(ctx.get("latent_preview_enabled", "NO")).upper() != "YES":
            return None
        sidecar = str(ctx.get("latent_preview_sidecar", "") or "").strip()
        if sidecar:
            root = Path(sidecar).with_suffix("")
        else:
            root = APP_ROOT / "temp" / "ltx23_latent_preview" / REPORT_STAMP
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        return None

def _latent_preview_should_emit(ctx: Dict[str, Any] | None, stage_key: str, step: int, total: int) -> bool:
    try:
        if ctx is None or str(ctx.get("latent_preview_enabled", "NO")).upper() != "YES":
            return False
        key = f"_latent_preview_last_emit_{stage_key}"
        now = time.time()
        last = float(ctx.get(key, 0.0) or 0.0)
        try:
            rate = max(1, min(30, int(ctx.get("latent_preview_rate", "8") or 8)))
        except Exception:
            rate = 8
        # Always emit first and final step. Otherwise obey the requested rate.
        if int(step) not in {1, int(total)} and (now - last) < (1.0 / float(rate)):
            return False
        ctx[key] = now
        return True
    except Exception:
        return False

def _latent_preview_infer_video_grid(ctx: Dict[str, Any] | None, tensor: Any) -> tuple[int, int, int] | None:
    try:
        import torch  # type: ignore
        if not isinstance(tensor, torch.Tensor):
            return None
        shape = tuple(int(x) for x in tensor.shape)
        if len(shape) == 5:
            return shape[2], shape[3], shape[4]
        if len(shape) != 3:
            return None
        tokens = int(shape[1])
        frames = int(ctx.get("latent_preview_num_frames", 0) or ctx.get("num_frames", 0) or 0) if ctx else 0
        width = int(ctx.get("latent_preview_width", 0) or ctx.get("width", 0) or 0) if ctx else 0
        height = int(ctx.get("latent_preview_height", 0) or ctx.get("height", 0) or 0) if ctx else 0
        latent_f = ((frames - 1) // 8 + 1) if frames > 0 else 0
        candidates: list[tuple[int, int, int, str]] = []
        if latent_f > 0 and width > 0 and height > 0:
            candidates.append((latent_f, max(1, (height // 2) // 32), max(1, (width // 2) // 32), "stage1"))
            candidates.append((latent_f, max(1, height // 32), max(1, width // 32), "stage2"))
        for f, h, w, _name in candidates:
            if f * h * w == tokens:
                return f, h, w
        # Fallback: factor tokens into a plausible video grid. Prefer known frame count.
        if latent_f > 0 and tokens % latent_f == 0:
            hw = tokens // latent_f
            # Use requested aspect ratio to choose h/w.
            aspect = (float(width) / float(height)) if width > 0 and height > 0 else 16.0 / 9.0
            h = max(1, int(round((hw / aspect) ** 0.5)))
            while h > 1 and hw % h != 0:
                h -= 1
            w = max(1, hw // h)
            return latent_f, h, w
    except Exception:
        return None
    return None

def _latent_preview_to_5d(ctx: Dict[str, Any] | None, tensor: Any) -> Any | None:
    try:
        import torch  # type: ignore
        if not isinstance(tensor, torch.Tensor):
            return None
        x = tensor.detach()
        if x.ndim == 5:
            return x
        if x.ndim == 3:
            grid = _latent_preview_infer_video_grid(ctx, x)
            if grid is None:
                return None
            f, h, w = grid
            if int(x.shape[1]) < f * h * w:
                return None
            x = x[:, : f * h * w, :]
            # b (f h w) c -> b c f h w, patch size is 1 in LTX 2.x pipelines.
            return x.reshape(x.shape[0], f, h, w, x.shape[2]).permute(0, 4, 1, 2, 3).contiguous()
    except Exception:
        return None
    return None

def _latent_preview_make_contact_sheet(ctx: Dict[str, Any] | None, denoised_video: Any, step: int, total: int, stage_key: str = "stage") -> Path | None:
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore
        x5 = _latent_preview_to_5d(ctx, denoised_video)
        if x5 is None or x5.ndim != 5 or int(x5.shape[1]) != 128:
            if ctx is not None:
                ctx["latent_preview_last_skip"] = f"unsupported tensor shape: {tuple(getattr(denoised_video, 'shape', ()))}"
            return None
        out_dir = _latent_preview_output_dir(ctx)
        if out_dir is None:
            return None
        # Pick a few temporal samples so the strip gives a sense of the clip, not only frame zero.
        frame_count = int(x5.shape[2])
        if frame_count <= 1:
            frame_ids = [0]
        else:
            frame_ids = sorted(set([0, frame_count // 3, (2 * frame_count) // 3, frame_count - 1]))
        sample = x5[:1, :, frame_ids, :, :].detach().to(device="cpu", dtype=torch.float32)
        # n c h w -> n h w c
        sample = sample[0].permute(1, 2, 3, 0).contiguous()
        factors = torch.tensor(_FRAMEVISION_LTX2_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(_FRAMEVISION_LTX2_LATENT_RGB_BIAS, dtype=sample.dtype)
        img = F.linear(sample, factors.t(), bias=bias).sigmoid().clamp(0, 1)
        # Enlarge tiny latent grids for readable UI thumbnails. Optional upscale gets a little more size.
        scale = 8 if str(ctx.get("latent_preview_upscale", "NO")).upper() == "YES" else 6
        img = img.permute(0, 3, 1, 2)
        img = F.interpolate(img, scale_factor=scale, mode="nearest").permute(0, 2, 3, 1)
        arr = (img.mul(255).byte().numpy())
        pil_frames = [Image.fromarray(a, mode="RGB") for a in arr]
        gap = 4
        label_h = 18
        w = sum(p.width for p in pil_frames) + gap * max(0, len(pil_frames) - 1)
        h = max(p.height for p in pil_frames) + label_h
        sheet = Image.new("RGB", (w, h), (24, 28, 36))
        draw = ImageDraw.Draw(sheet)
        xoff = 0
        for idx, frame_img in zip(frame_ids, pil_frames):
            sheet.paste(frame_img, (xoff, 0))
            draw.text((xoff + 3, frame_img.height + 2), f"f{idx}", fill=(230, 235, 245))
            xoff += frame_img.width + gap
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_key or "stage"))[:40]
        path = out_dir / f"latent_preview_{safe_stage}_step{int(step):03d}_of_{int(total):03d}.jpg"
        sheet.save(path, format="JPEG", quality=90, optimize=False)
        return path
    except Exception as exc:
        if ctx is not None:
            ctx["latent_preview_last_error"] = f"{type(exc).__name__}: {exc}"
        return None

def _latent_preview_emit_image(ctx: Dict[str, Any] | None, denoised_video: Any, step: int, total: int, stage_key: str = "stage") -> None:
    try:
        if not _latent_preview_should_emit(ctx, stage_key, int(step), int(total)):
            return
        path = _latent_preview_make_contact_sheet(ctx, denoised_video, int(step), int(total), stage_key)
        if path is None:
            return
        event = {
            "kind": "image",
            "step": int(step),
            "total": int(total),
            "stage": str(stage_key),
            "path": str(path),
            "message": f"{stage_key} step {int(step)}/{int(total)}",
        }
        _latent_preview_sidecar_write(ctx, event)
        print(f"[framevision-latent-preview] stage={stage_key} step={int(step)}/{int(total)} path={path}", flush=True)
    except Exception as exc:
        if ctx is not None:
            ctx["latent_preview_emit_error"] = f"{type(exc).__name__}: {exc}"

def _install_native_ltx_latent_preview_hook(ctx: Dict[str, Any]) -> None:
    """Patch LTX's Euler denoising loop so FrameVision can preview real x0/denoised video latents.

    This does not edit LTX repo files. It replaces the imported loop before
    runpy imports the selected pipeline, captures the denoised video latent each
    sampling step, converts it to a tiny RGB contact sheet, and writes JSONL
    events for the UI preview strip.
    """
    try:
        if str(ctx.get("latent_preview_enabled", "NO")).upper() != "YES":
            ctx["latent_preview_hook_installed"] = "NO: disabled"
            return
        import importlib as _importlib
        from dataclasses import replace as _replace
        samplers = _importlib.import_module("ltx_pipelines.utils.samplers")
        blocks = _importlib.import_module("ltx_pipelines.utils.blocks")
        if getattr(samplers, "_framevision_latent_preview_hooked", False):
            ctx["latent_preview_hook_installed"] = "YES: already installed"
            return
        original_euler = getattr(samplers, "euler_denoising_loop", None)
        original_step_state = getattr(samplers, "_step_state", None)
        original_post = getattr(samplers, "post_process_latent", None)
        original_tqdm = getattr(samplers, "tqdm", None)
        if original_euler is None or original_step_state is None or original_tqdm is None:
            raise AttributeError("required sampler symbols missing")

        def _fv_euler_denoising_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
            total = max(0, int(len(sigmas) - 1))
            stage_index = int(ctx.get("_latent_preview_stage_index", 0) or 0) + 1
            ctx["_latent_preview_stage_index"] = stage_index
            stage_key = "stage1" if stage_index == 1 else "stage2" if stage_index == 2 else f"stage{stage_index}"
            for step_idx, _ in enumerate(original_tqdm(sigmas[:-1])):
                video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
                denoised_video = video_result.denoised if video_result is not None else None
                denoised_audio = audio_result.denoised if audio_result is not None else None
                preview_video = denoised_video
                try:
                    if video_state is not None and denoised_video is not None and original_post is not None:
                        preview_video = original_post(denoised_video, video_state.denoise_mask, video_state.clean_latent)
                except Exception:
                    preview_video = denoised_video
                _latent_preview_emit_image(ctx, preview_video, int(step_idx) + 1, total, stage_key)
                video_state = original_step_state(video_state, denoised_video, stepper, sigmas, step_idx)
                audio_state = original_step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)
            return (video_state, audio_state)

        setattr(_fv_euler_denoising_loop, "_framevision_latent_preview_loop", True)
        setattr(samplers, "euler_denoising_loop", _fv_euler_denoising_loop)
        # DiffusionStage.run resolves this global name from blocks.py when loop=None.
        setattr(blocks, "euler_denoising_loop", _fv_euler_denoising_loop)
        setattr(samplers, "_framevision_latent_preview_hooked", True)
        ctx["latent_preview_hook_installed"] = "YES: patched ltx_pipelines.utils.samplers.euler_denoising_loop"
        ctx["latent_preview_runtime"] = "native x0/denoised latent RGB preview hook active"
        _latent_preview_sidecar_write(ctx, {"kind": "status", "message": "Native latent RGB preview hook active."})
    except Exception as exc:
        ctx["latent_preview_hook_installed"] = f"NO: {type(exc).__name__}: {exc}"
        _latent_preview_sidecar_write(ctx, {"kind": "status", "message": f"Latent preview hook failed: {type(exc).__name__}: {exc}"})

def _ltx_quiet_status(ctx: Dict[str, Any] | None, message: str, enabled: bool | None = None) -> None:
    """Print one tiny progress breadcrumb for No Boundary Echo mode.

    This is separate from the verbose VRAM boundary trace. It gives the UI/user
    a useful phase label during long LTX load/encode steps without bringing back
    the full boundary log.
    """
    try:
        msg = str(message or "").strip()
        if not msg:
            return
        if ctx is None:
            should_print = True if enabled is None else bool(enabled)
            last = ""
        else:
            should_print = bool(ctx.get("ltx_quiet_status_enabled", False)) if enabled is None else bool(enabled)
            ctx["ltx_quiet_status"] = msg
            ctx.setdefault("ltx_quiet_status_events", []).append(msg)
            last = str(ctx.get("_ltx_quiet_status_last", "") or "")
            ctx["_ltx_quiet_status_last"] = msg
        _latent_preview_sidecar_write(ctx, {"kind": "status", "message": msg})
        if not should_print or msg == last:
            return
        print(f"[ltx-status] {msg}", flush=True)
    except Exception:
        pass




def _install_ltx_realtime_step_timer(ctx: Dict[str, Any], expected_steps: int | None = None, *, echo: bool = True) -> None:
    """Log real wall-clock time for completed denoise steps.

    LTX/tqdm averages can make the first displayed step look huge because setup
    time before the first progress update is folded into the average. This hook
    measures the time spent between yielding one tqdm item and receiving control
    back after that item finishes, so the printed value is the real completed-step
    duration and does not include model loading before step 1.
    """
    try:
        ctx["ltx_realtime_step_timer_installed"] = "NO"
        ctx["ltx_realtime_step_timer_expected_steps"] = str(expected_steps or "")
        ctx["ltx_realtime_step_event_count"] = "0"
        ctx["ltx_realtime_step_summary"] = "not installed"
        ctx.setdefault("ltx_realtime_step_events", [])

        try:
            expected = int(expected_steps or 0)
        except Exception:
            expected = 0

        import importlib as _importlib
        tqdm_mod = _importlib.import_module("tqdm")
        tqdm_cls = getattr(tqdm_mod, "tqdm", None)
        if tqdm_cls is None:
            ctx["ltx_realtime_step_summary"] = "not installed: tqdm.tqdm missing"
            return

        if getattr(tqdm_cls, "_framevision_ltx_realtime_step_timer_patched", False):
            ctx["ltx_realtime_step_timer_installed"] = "YES"
            ctx["ltx_realtime_step_summary"] = "already installed"
            return

        original_iter = getattr(tqdm_cls, "__iter__")
        original_update = getattr(tqdm_cls, "update")

        state = {"bar_index": 0}

        def _track_bar(bar: Any) -> bool:
            try:
                total = getattr(bar, "total", None)
                if total is None:
                    return False
                total_i = int(total)
                if total_i <= 0:
                    return False
                # Main LTX denoise bars use num_inference_steps. This avoids
                # logging unrelated long file/load/frame progress bars.
                if expected > 0:
                    return total_i == expected
                return total_i <= 100
            except Exception:
                return False

        def _bar_name(bar: Any, bar_id: int) -> str:
            try:
                desc = str(getattr(bar, "desc", "") or "").strip()
            except Exception:
                desc = ""
            return desc if desc else f"progress-{bar_id}"

        def _record(bar: Any, bar_id: int, completed: int, total: int, dt: float, source: str) -> None:
            try:
                if dt < 0:
                    return
                msg = f"{_bar_name(bar, bar_id)} step {completed}/{total} real={dt:.3f}s"
                events = ctx.setdefault("ltx_realtime_step_events", [])
                if isinstance(events, list) and len(events) < 500:
                    events.append(msg)
                ctx["ltx_realtime_step_event_count"] = str(len(events) if isinstance(events, list) else 0)
                ctx["ltx_realtime_step_summary"] = msg
                if echo:
                    print(f"[ltx-step-time] {msg}", flush=True)
            except Exception:
                pass

        def _wrapped_iter(self: Any):
            tracked = _track_bar(self)
            bar_id = 0
            total = 0
            if tracked:
                state["bar_index"] += 1
                bar_id = int(state["bar_index"])
                try:
                    total = int(getattr(self, "total", 0) or 0)
                except Exception:
                    total = expected or 0
                try:
                    setattr(self, "_framevision_ltx_realtime_iter_active", True)
                except Exception:
                    pass
                try:
                    start_msg = f"{_bar_name(self, bar_id)} timer armed for {total} steps"
                    ctx["ltx_realtime_step_summary"] = start_msg
                    if echo:
                        print(f"[ltx-step-time] {start_msg}", flush=True)
                except Exception:
                    pass

            completed = 0
            step_start = None
            try:
                for obj in original_iter(self):
                    if tracked:
                        step_start = time.perf_counter()
                    yield obj
                    if tracked and step_start is not None:
                        completed += 1
                        _record(self, bar_id, completed, total or completed, time.perf_counter() - step_start, "iter")
                        step_start = None
            finally:
                if tracked:
                    try:
                        setattr(self, "_framevision_ltx_realtime_iter_active", False)
                    except Exception:
                        pass

        def _wrapped_update(self: Any, n: int = 1):
            before_n = 0
            tracked = False
            try:
                tracked = _track_bar(self) and not bool(getattr(self, "_framevision_ltx_realtime_iter_active", False))
                before_n = int(getattr(self, "n", 0) or 0)
            except Exception:
                tracked = False
            result = original_update(self, n)
            if tracked:
                try:
                    total = int(getattr(self, "total", 0) or 0)
                    now = time.perf_counter()
                    last = getattr(self, "_framevision_ltx_realtime_last_update", None)
                    after_n = int(getattr(self, "n", before_n) or before_n)
                    if last is None:
                        setattr(self, "_framevision_ltx_realtime_last_update", now)
                        state["bar_index"] += 1
                        setattr(self, "_framevision_ltx_realtime_bar_id", int(state["bar_index"]))
                        if echo:
                            print(f"[ltx-step-time] {_bar_name(self, int(state['bar_index']))} first progress update seen; timing following completed steps", flush=True)
                    else:
                        bar_id = int(getattr(self, "_framevision_ltx_realtime_bar_id", 0) or 0)
                        if bar_id <= 0:
                            state["bar_index"] += 1
                            bar_id = int(state["bar_index"])
                            setattr(self, "_framevision_ltx_realtime_bar_id", bar_id)
                        completed = after_n if after_n > 0 else before_n + int(n or 1)
                        _record(self, bar_id, completed, total or completed, now - float(last), "update")
                        setattr(self, "_framevision_ltx_realtime_last_update", now)
                except Exception:
                    pass
            return result

        setattr(tqdm_cls, "__iter__", _wrapped_iter)
        setattr(tqdm_cls, "update", _wrapped_update)
        setattr(tqdm_cls, "_framevision_ltx_realtime_step_timer_patched", True)
        ctx["ltx_realtime_step_timer_installed"] = "YES"
        ctx["ltx_realtime_step_summary"] = "installed"
    except Exception as exc:
        try:
            ctx["ltx_realtime_step_timer_installed"] = "NO"
            ctx["ltx_realtime_step_summary"] = f"install failed: {type(exc).__name__}: {exc}"
        except Exception:
            pass



def _system_memory_snapshot() -> str:
    """Best-effort process/RAM/pagefile snapshot for LTX load probes."""
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return (
            f"RSS={_fmt_bytes(int(getattr(mi, 'rss', 0)))}, "
            f"VMS={_fmt_bytes(int(getattr(mi, 'vms', 0)))}, "
            f"RAM used={_fmt_bytes(int(getattr(vm, 'used', 0)))}/{_fmt_bytes(int(getattr(vm, 'total', 0)))}, "
            f"swap/pagefile used={_fmt_bytes(int(getattr(swap, 'used', 0)))}/{_fmt_bytes(int(getattr(swap, 'total', 0)))}"
        )
    except Exception as exc:
        return f"RSS/VMS/RAM/swap unavailable ({type(exc).__name__}: {exc})"


def _full_memory_snapshot(torch_module: Any | None = None) -> str:
    return f"{_cuda_snapshot(torch_module)}; {_system_memory_snapshot()}"

def _cleanup_cuda(torch_module: Any | None = None) -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _ltx_prepare_for_video_decoder_finalize_guard(
    ctx: Dict[str, Any],
    hooks_mod: Any | None,
    torch_module: Any | None,
    reason: str = "before VideoDecoderConfigurator",
    echo: bool = True,
) -> None:
    """One-shot cleanup boundary before LTX loads the final VideoDecoder.

    The one-stage crash happened after all denoise steps completed, right when
    the official pipeline started loading VideoDecoder weights. This guard stays
    outside the LTX repo: detach VRAM Lab's denoise block hooks, park any hooked
    blocks on CPU, clear CUDA caches, then let the normal LTX VideoDecoder load
    continue. It deliberately runs only once per job and only at VideoDecoder.
    """
    try:
        if str(ctx.get("_ltx_video_decoder_finalize_guard_done", "NO")).upper().startswith("YES"):
            return
        ctx["_ltx_video_decoder_finalize_guard_done"] = "YES"
        ctx["ltx_video_decoder_finalize_guard"] = "STARTED"
        ctx["ltx_video_decoder_finalize_guard_reason"] = str(reason)
        before = _full_memory_snapshot(torch_module)
        ctx["ltx_video_decoder_finalize_guard_before"] = before
        runtimes = ctx.get("_ltx_early_residency_runtimes") or []
        if not isinstance(runtimes, list):
            runtimes = []
        guard = None
        try:
            if hooks_mod is not None and hasattr(hooks_mod, "make_finalize_guard"):
                guard = hooks_mod.make_finalize_guard(ctx=ctx, label="ltx_video_decoder", torch_module=torch_module)
        except Exception as exc:
            ctx["ltx_video_decoder_finalize_guard_errors"] = f"make_finalize_guard failed: {type(exc).__name__}: {exc}"
            guard = None
        if guard is not None:
            try:
                guard.prepare_for_decode(
                    runtimes=list(runtimes),
                    components=[],
                    label="before_video_decoder_load",
                    vae=None,
                )
            except Exception as exc:
                ctx["ltx_video_decoder_finalize_guard_errors"] = f"prepare_for_decode failed: {type(exc).__name__}: {exc}"
                try:
                    _update_ltx_early_residency_runtimes(ctx, detach=True)
                except Exception:
                    pass
                _cleanup_cuda(torch_module)
        else:
            try:
                _update_ltx_early_residency_runtimes(ctx, detach=True)
            except Exception as exc:
                ctx["ltx_video_decoder_finalize_guard_errors"] = f"runtime detach failed: {type(exc).__name__}: {exc}"
            for _ in range(2):
                _cleanup_cuda(torch_module)
        after = _full_memory_snapshot(torch_module)
        ctx["ltx_video_decoder_finalize_guard_after"] = after
        ctx["ltx_video_decoder_finalize_guard"] = "YES: detached denoise hooks and cleaned CUDA before VideoDecoder load"
        ctx.setdefault("notes", []).append(
            "Installed LTX VideoDecoder finalize guard: before final VideoDecoder load, detach VRAM Lab denoise hooks and clean CUDA to reduce post-step native crashes."
        )
        if echo:
            try:
                print(
                    f"[vram-lab-ltx-finalize] before VideoDecoder load cleanup | before: {before} | after: {after}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception as exc:
        ctx["ltx_video_decoder_finalize_guard"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_video_decoder_finalize_guard_errors"] = f"{type(exc).__name__}: {exc}"


def _ltx_prepare_for_pre_upscaler_video_encoder_guard(
    ctx: Dict[str, Any],
    torch_module: Any | None,
    reason: str = "before post-stage1 VideoEncoderConfigurator",
    echo: bool = True,
) -> None:
    """One-shot pressure drop before the between-stage VideoEncoder load.

    Real 481-frame tests show a native Windows crash can happen after Stage 1
    reaches 8/8 but before Stage 2 begins, when the official two-stage pipeline
    loads VideoEncoder while the retained denoiser still owns a large hotset.
    This guard keeps the phase-retention bridge alive, keeps hooks installed,
    but trims the retained denoiser runtime down to the Stage-2/encoder-safe
    budget before VideoEncoder loads. It is deliberately one-shot and only runs
    after Stage-1 forward has completed.
    """
    try:
        if str(ctx.get("_ltx_pre_upscaler_video_encoder_guard_done", "NO")).upper().startswith("YES"):
            return
        if str(ctx.get("_ltx_low_profile_after_stage1_ready", "NO")).upper() != "YES":
            ctx["ltx_pre_upscaler_video_encoder_guard"] = "SKIPPED: Stage 1 not marked complete"
            return

        ctx["_ltx_pre_upscaler_video_encoder_guard_done"] = "YES"
        ctx["ltx_pre_upscaler_video_encoder_guard"] = "STARTED"
        ctx["ltx_pre_upscaler_video_encoder_guard_reason"] = str(reason)
        before = _full_memory_snapshot(torch_module)
        ctx["ltx_pre_upscaler_video_encoder_guard_before"] = before

        runtime = ctx.get("_ltx_phase_retention_runtime")
        if runtime is None:
            runtimes = ctx.get("_ltx_early_residency_runtimes") or []
            if isinstance(runtimes, list) and runtimes:
                runtime = runtimes[0]
        ctx["ltx_pre_upscaler_video_encoder_guard_runtime"] = "found" if runtime is not None else "missing"

        gb, source = _block_limit_for_role(ctx, "stage2_refine_denoise")
        if gb <= 0.0:
            gb = max(2.0, min(6.0, _ctx_float(ctx, "stage1_block_size_limit_gb", 12.0) * 0.35))
            source = "derived safe fallback from Stage 1 limit"
        gb = max(2.0, min(8.0, float(gb)))
        bytes_value = int(float(gb) * 1024 ** 3)
        ctx["ltx_pre_upscaler_video_encoder_guard_target_gb"] = f"{gb:.1f} GB ({source})"

        if runtime is not None:
            try:
                setattr(runtime, "hot_block_budget_bytes", bytes_value)
                setattr(runtime, "safe_hot_window_gb", float(gb))
                setattr(runtime, "balanced_hot_window_gb", float(gb))
                # Keep the trim conservative: planned hotset can be rebuilt later,
                # but the between-stage encoder load needs immediate headroom.
                setattr(runtime, "stable_hotset_budget_bytes", 0)
                policy = getattr(runtime, "policy", None)
                if isinstance(policy, dict):
                    policy["hot_block_budget_bytes"] = bytes_value
                    policy["safe_hot_window_gb"] = float(gb)
                    policy["safe_hot_window_bytes"] = bytes_value
                    policy["balanced_hot_window_gb"] = float(gb)
                    policy["balanced_hot_window_bytes"] = bytes_value
                    policy["stable_hotset_budget_bytes"] = 0
                if hasattr(runtime, "_plan_stable_hotset"):
                    runtime._plan_stable_hotset()
                if hasattr(runtime, "_trim_hot_blocks"):
                    runtime._trim_hot_blocks(keep_name="")
                if hasattr(runtime, "update_context"):
                    runtime.update_context(ctx)
                    _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="pre-upscaler VideoEncoder guard trimmed retained Stage-1 hotset")
                ctx["ltx_pre_upscaler_video_encoder_guard_trim"] = "YES: retained denoiser hotset trimmed before VideoEncoder load"
            except Exception as exc:
                ctx["ltx_pre_upscaler_video_encoder_guard_trim"] = f"FAILED: {type(exc).__name__}: {exc}"
        else:
            ctx["ltx_pre_upscaler_video_encoder_guard_trim"] = "SKIPPED: runtime unavailable"

        try:
            gc.collect()
            if torch_module is not None and hasattr(torch_module, "cuda"):
                try:
                    torch_module.cuda.synchronize()
                except Exception:
                    pass
                if hasattr(torch_module.cuda, "empty_cache"):
                    torch_module.cuda.empty_cache()
                if hasattr(torch_module.cuda, "ipc_collect"):
                    try:
                        torch_module.cuda.ipc_collect()
                    except Exception:
                        pass
            ctx["ltx_pre_upscaler_video_encoder_guard_cleanup"] = "YES: gc + cuda empty_cache/ipc_collect once"
        except Exception as exc:
            ctx["ltx_pre_upscaler_video_encoder_guard_cleanup"] = f"FAILED: {type(exc).__name__}: {exc}"

        after = _full_memory_snapshot(torch_module)
        ctx["ltx_pre_upscaler_video_encoder_guard_after"] = after
        ctx["ltx_pre_upscaler_video_encoder_guard"] = "YES: pressure dropped before post-stage1 VideoEncoder load"
        ctx.setdefault("notes", []).append(
            "Installed LTX pre-upscaler VideoEncoder guard: after Stage 1 completes, trim the retained denoiser hotset to the Stage-2/safe budget before the official VideoEncoder load."
        )
        if echo:
            try:
                print(
                    f"[vram-lab-ltx-pre-upscaler] before VideoEncoder load guard | target={gb:.1f}GB | before: {before} | after: {after}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception as exc:
        ctx["ltx_pre_upscaler_video_encoder_guard"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_pre_upscaler_video_encoder_guard_errors"] = f"{type(exc).__name__}: {exc}"



def _maybe_ltx_low_profile_transition_cleanup(
    ctx: Dict[str, Any],
    torch_module: Any | None,
    stage_key: str,
    reason: str,
    echo: bool = True,
) -> None:
    """Run a narrow cleanup before low-profile post-denoise component rebuilds.

    12/16 GB profiles can finish denoise but crash while the official pipeline
    rebuilds VideoEncoder/Decoder/Audio/Vocoder components. This does not change
    LTX output logic; it only gives Python/CUDA a chance to release stale cache
    and collects a breadcrumb right before the risky rebuild/finalization zone.
    """
    # Disabled: repeated transition cleanup made 12/24 GB runs unstable in real tests.
    # Keep the function as a no-op so old call sites/report fields remain compatible.
    try:
        ctx[f"ltx_low_profile_{stage_key}_cleanup"] = "NO: disabled after unstable repeated-cleanup tests"
        ctx[f"ltx_low_profile_{stage_key}_cleanup_reason"] = "disabled"
    except Exception:
        pass
    return

    try:
        profile = int(float(str(ctx.get("vram_profile_gb", "24") or "24")))
    except Exception:
        profile = 24
    if profile >= 24:
        return
    if str(ctx.get(f"_ltx_low_profile_{stage_key}_ready", "NO")) != "YES":
        return
    if str(ctx.get(f"_ltx_low_profile_{stage_key}_cleanup_done", "NO")) == "YES":
        return

    ctx[f"_ltx_low_profile_{stage_key}_cleanup_done"] = "YES"
    before = _full_memory_snapshot(torch_module)
    ctx[f"ltx_low_profile_{stage_key}_cleanup_before"] = before
    ctx[f"ltx_low_profile_{stage_key}_cleanup_reason"] = str(reason)
    try:
        for _ in range(2):
            _cleanup_cuda(torch_module)
            time.sleep(0.05)
    except Exception as exc:
        ctx[f"ltx_low_profile_{stage_key}_cleanup_errors"] = f"{type(exc).__name__}: {exc}"
    after = _full_memory_snapshot(torch_module)
    ctx[f"ltx_low_profile_{stage_key}_cleanup_after"] = after
    ctx[f"ltx_low_profile_{stage_key}_cleanup"] = f"YES: before {reason}"
    if echo:
        try:
            print(
                f"[vram-lab-ltx-low-profile-cleanup] {stage_key}: before {reason} | before: {before} | after: {after}",
                flush=True,
            )
        except Exception:
            pass



def _detect_primary_cuda_vram_gb() -> float:
    """Best-effort CUDA VRAM detection for --vram-profile auto."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(getattr(props, "total_memory", 0) or 0) / (1024.0 ** 3)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        first = str(out).splitlines()[0].strip()
        if first:
            return float(first) / 1024.0
    except Exception:
        pass
    return 0.0


def _auto_ltx_vram_profile_gb() -> int:
    total = _detect_primary_cuda_vram_gb()
    # Same policy as the Planner/WAN auto profile, with a safer RTX 3090 edge:
    # below 16GB -> 12GB, 16GB up to 22.99GB -> 16GB, 23GB+ -> 24GB.
    # Some 24GB cards report around 23.x/23.9GB usable, so do not require 23.9+.
    if total >= 23.0:
        return 24
    if total >= 16.0:
        return 16
    return 12


def _resolve_ltx_vram_profile(profile_value: Any) -> tuple[int, Dict[str, Any]]:
    """Return the selected clean-room LTX VRAM profile.

    Supports explicit 12/16/24GB profiles and auto detection for Planner use.
    """
    raw = str(profile_value or os.environ.get("FRAMEVISION_LTX_VRAM_PROFILE_GB", "24") or "24").strip().lower()
    if raw in {"auto", "detect", "gpu"}:
        profile_gb = _auto_ltx_vram_profile_gb()
    else:
        if raw.endswith("gb"):
            raw = raw[:-2].strip()
        try:
            profile_gb = int(float(raw))
        except Exception:
            profile_gb = 24
    if profile_gb not in LTX_VRAM_PROFILES:
        profile_gb = 24
    profile = dict(LTX_VRAM_PROFILES[profile_gb])
    if str(profile_value or "").strip().lower() in {"auto", "detect", "gpu"}:
        profile["note"] = str(profile.get("note", "")) + "; selected automatically from detected GPU VRAM"
        profile["auto_selected"] = True
    return profile_gb, profile


def _apply_ltx_profile_env_defaults(ctx: Dict[str, Any], profile_gb: int, profile: Dict[str, Any]) -> None:
    """Apply Gemma/profile defaults without overriding explicit user env vars."""
    gemma_target = float(profile.get("gemma_target_vram_gb", LTX_VRAM_PROFILE_GEMMA_TARGET_24GB))
    gemma_hot_pin = float(profile.get("gemma_hot_pin_gb", LTX_VRAM_PROFILE_HOT_PIN_24GB))
    gemma_disk_slots = int(profile.get("gemma_disk_slots", LTX_VRAM_PROFILE_DISK_SLOTS_24GB))
    # Profile selection must control Gemma too. Do not use setdefault here:
    # stale 24GB values from the parent environment would make 12/16GB profiles
    # still try to give Gemma far too much VRAM.
    os.environ["FRAMEVISION_LTX_GEMMA_TARGET_VRAM_GB"] = f"{gemma_target:.1f}"
    os.environ["FRAMEVISION_LTX_GEMMA_HOT_PIN_GB"] = f"{gemma_hot_pin:.1f}"
    os.environ["FRAMEVISION_LTX_GEMMA_DISK_SLOTS"] = str(gemma_disk_slots)
    ctx["vram_profile_gb"] = str(profile_gb)
    ctx["vram_profile_source"] = "--vram-profile / FRAMEVISION_LTX_VRAM_PROFILE_GB"
    ctx["ltx_main_profile_hot_window_gb"] = f"{float(profile.get('main_hot_window_gb', 0.0)):.1f}"
    ctx["ltx_gemma_profile_target_vram_gb"] = f"{gemma_target:.1f}"
    ctx["ltx_gemma_profile_hot_pin_gb"] = f"{gemma_hot_pin:.1f}"
    ctx["ltx_gemma_profile_disk_slots"] = str(gemma_disk_slots)


def _load_vram_lab_hooks_module(ctx: Dict[str, Any]) -> Any | None:
    """Load VRAM Lab 0.7.4 hook helpers from either supported filename.

    Older/working local tests may store the 0.7.4 file as either:
      - tools/vram_lab/vram_forward_hooks.py
      - tools/vram_lab/vram-lab-074.py

    The second name cannot be imported with a normal Python import because of
    the hyphens, so load it explicitly by file path. This only restores the
    existing 0.7.4 hook module; it does not change hook behavior.
    """
    errors: List[str] = []

    try:
        mod = importlib.import_module("vram_forward_hooks")
        ctx["vram_lab_hooks_module"] = "vram_forward_hooks"
        return mod
    except Exception as exc:
        errors.append(f"vram_forward_hooks import failed: {type(exc).__name__}: {exc}")

    candidates = [
        VRAM_LAB_DIR / "vram-lab-074.py",
        VRAM_LAB_DIR / "vram_lab_074.py",
        HELPERS_DIR / "vram-lab-074.py",
        HELPERS_DIR / "vram_lab_074.py",
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("vram_lab_074_runtime", str(path))
            if spec is None or spec.loader is None:
                errors.append(f"{path}: could not create import spec")
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("vram_lab_074_runtime", mod)
            spec.loader.exec_module(mod)
            ctx["vram_lab_hooks_module"] = str(path)
            ctx.setdefault("notes", []).append(f"Loaded VRAM Lab hooks from {path.name}; normal vram_forward_hooks.py was not found.")
            return mod
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    ctx["vram_lab_hooks_module"] = "unavailable"
    ctx["vram_lab_hooks_module_errors"] = " | ".join(errors) if errors else "no candidate file found"
    return None


def _apply_ltx_vram_hot_window_override(ctx: Dict[str, Any], hooks_mod: Any, hot_window_gb: float = 16.5, profile_gb: int = 24, profile: Dict[str, Any] | None = None, emergency_floor_gb: float = 1.5) -> None:
    """Apply the selected LTX VRAM profile denoiser hot-window from the wrapper.

    This keeps the current clean-room LTX wrapper as the source of truth and
    avoids editing the reusable VRAM Lab helper file just to test one residency
    tune. The hook module still owns the runtime logic; this only updates its
    exported active-profile constants before BatchSplitAdapter hooks attach.
    """
    try:
        gb = float(hot_window_gb)
    except Exception:
        gb = 16.5
    gb = max(0.0, gb)
    try:
        floor_gb = max(0.25, min(3.0, float(emergency_floor_gb)))
    except Exception:
        floor_gb = 1.5
    hot_bytes = int(gb * 1024 ** 3)
    selected_profile = dict(profile or {})
    profile_gb = int(profile_gb or 24)
    delta = float(selected_profile.get("delta_from_24gb", max(0, 24 - profile_gb)))
    main_safety_derate = float(selected_profile.get("main_safety_derate_gb", 0.0))
    gemma_target = float(selected_profile.get("gemma_target_vram_gb", LTX_VRAM_PROFILE_GEMMA_TARGET_24GB))
    note = (
        f"{profile_gb} GB LTX VRAM profile: main/distilled denoiser hot-window {gb:.1f} GB; "
        f"Gemma target {gemma_target:.1f} GB"
        if profile_gb != 24 else
        f"24 GB LTX VRAM profile: main/distilled denoiser hot-window {gb:.1f} GB; "
        f"Gemma target {gemma_target:.1f} GB"
    )

    ctx["vram_profile_gb"] = str(profile_gb)
    ctx["vram_safe_hot_window_gb"] = f"{gb:.1f}"
    ctx["vram_balanced_hot_window_gb"] = f"{gb:.1f}"
    ctx["vram_profile_note"] = f"{note}; emergency trim below {floor_gb:.2f} GB driver-free"
    ctx["vram_emergency_driver_free_floor"] = f"{floor_gb:.2f} GB"

    try:
        setattr(hooks_mod, "ACTIVE_VRAM_LAB_PROFILE_GB", profile_gb)
        setattr(hooks_mod, "SAFE_HOT_WINDOW_GB", gb)
        setattr(hooks_mod, "SAFE_HOT_WINDOW_BYTES", hot_bytes)
        setattr(hooks_mod, "BALANCED_HOT_WINDOW_GB", gb)
        setattr(hooks_mod, "BALANCED_HOT_WINDOW_BYTES", hot_bytes)
        profiles = getattr(hooks_mod, "VRAM_LAB_RESIDENCY_PROFILES", None)
        if isinstance(profiles, dict):
            # Replace the old 12/16 GB placeholders with real runtime profiles.
            for pgb, pval in LTX_VRAM_PROFILES.items():
                pnote = str(pval.get("note", f"{pgb} GB LTX VRAM profile"))
                profiles.setdefault(pgb, {})["safe"] = {
                    "hot_window_gb": float(pval.get("main_hot_window_gb", 0.0)),
                    "driver_free_floor_gb": floor_gb,
                    "note": pnote,
                }
            active_safe = profiles.setdefault(profile_gb, {}).setdefault("safe", {})
            if isinstance(active_safe, dict):
                active_safe["hot_window_gb"] = gb
                active_safe["driver_free_floor_gb"] = floor_gb
                active_safe["note"] = note
        ctx["vram_hot_window_override"] = f"YES: {profile_gb}GB profile safe hot-window set to {gb:.1f} GB before hook attachment"
        ctx.setdefault("notes", []).append(
            f"Applied LTX VRAM profile {profile_gb}GB: main/distilled denoiser residency {gb:.1f} GB; Gemma target {gemma_target:.1f} GB."
        )
    except Exception as exc:
        ctx["vram_hot_window_override"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx.setdefault("notes", []).append(f"VRAM hot-window override failed: {type(exc).__name__}: {exc}")



def _install_ltx_block_churn_policy_override(
    ctx: Dict[str, Any],
    hooks_mod: Any,
    policy_name: str = "default",
    emergency_floor_gb: float = 2.0,
) -> None:
    """Install a narrow runtime policy override for denoiser block churn tests.

    Default VRAM Lab already keeps blocks resident until the rolling hot-window is
    full. Long LTX one-stage runs showed hundreds of block load/unload cycles,
    so this wrapper adds one focused experiment: keep hot blocks sticky and only
    trim when the real driver-free floor is crossed. This does not edit
    vram_forward_hooks.py and does not change the LTX repo.
    """
    selected = str(policy_name or "default").strip().lower()
    if selected in {"", "default", "off"}:
        ctx["ltx_block_churn_policy"] = "default"
        ctx["ltx_block_churn_override_installed"] = "NO"
        ctx["ltx_block_churn_hot_window_mode"] = "default rolling hot-window"
        ctx["ltx_block_churn_emergency_floor_gb"] = "default"
        ctx["ltx_block_churn_errors"] = "none"
        return

    ctx["ltx_block_churn_policy"] = selected
    ctx["ltx_block_churn_override_installed"] = "NO"
    ctx["ltx_block_churn_hot_window_mode"] = "n/a"
    ctx["ltx_block_churn_emergency_floor_gb"] = f"{float(emergency_floor_gb or 0.0):.2f}"
    ctx["ltx_block_churn_errors"] = "none"

    if selected != "sticky_floor":
        ctx["ltx_block_churn_errors"] = f"unsupported policy: {selected}"
        return

    try:
        floor_gb = max(0.25, float(emergency_floor_gb or 2.0))
    except Exception:
        floor_gb = 2.0
    floor_bytes = int(floor_gb * 1024 ** 3)

    try:
        original = getattr(hooks_mod, "apply_vram_lab_profile_defaults")
    except Exception as exc:
        ctx["ltx_block_churn_errors"] = f"apply_vram_lab_profile_defaults unavailable: {type(exc).__name__}: {exc}"
        return

    if getattr(original, "_framevision_ltx_block_churn_policy", False):
        ctx["ltx_block_churn_override_installed"] = "YES: already installed"
        return

    def wrapped_apply_vram_lab_profile_defaults(policy: Any = None, mode: Any = None) -> Dict[str, Any]:
        out = original(policy, mode)
        try:
            out["release_after_forward"] = False
            out["unload_other_blocks_before_load"] = False
            out["synchronize_after_unload"] = False
            out["empty_cache_after_unload"] = False
            out["empty_cache_every"] = 0
            # 0 disables the normal rolling hot-window trim in vram_forward_hooks.
            # Emergency trim remains active through emergency_driver_free_floor_bytes.
            out["hot_block_budget_bytes"] = 0
            out["safe_hot_window_gb"] = 0.0
            out["safe_hot_window_bytes"] = 0
            out["balanced_hot_window_gb"] = 0.0
            out["balanced_hot_window_bytes"] = 0
            out["emergency_driver_free_floor_bytes"] = floor_bytes
            old_note = str(out.get("profile_note", ""))
            out["profile_note"] = (old_note + " | " if old_note else "") + (
                f"FrameVision sticky_floor churn test: no normal hot-window trim; "
                f"trim only if driver free falls below {floor_gb:.2f} GB."
            )
        except Exception:
            pass
        return out

    try:
        setattr(wrapped_apply_vram_lab_profile_defaults, "_framevision_ltx_block_churn_policy", True)
        setattr(wrapped_apply_vram_lab_profile_defaults, "_framevision_ltx_block_churn_original", original)
    except Exception:
        pass

    try:
        setattr(hooks_mod, "apply_vram_lab_profile_defaults", wrapped_apply_vram_lab_profile_defaults)
        ctx["ltx_block_churn_override_installed"] = "YES"
        ctx["ltx_block_churn_hot_window_mode"] = "sticky_floor: normal hot-window trim disabled; emergency trim only"
        ctx["ltx_block_churn_emergency_floor_gb"] = f"{floor_gb:.2f}"
        ctx.setdefault("notes", []).append(
            f"Installed LTX block-churn sticky_floor policy: keep denoiser blocks resident until driver-free VRAM falls below {floor_gb:.2f} GB."
        )
    except Exception as exc:
        ctx["ltx_block_churn_errors"] = f"install failed: {type(exc).__name__}: {exc}"

def _install_ltx_block_streaming_pinned_layout_guard(ctx: Dict[str, Any]) -> None:
    """Disable LTX full-layout pinned RAM allocation from the FrameVision wrapper.

    This is intentionally narrow. It does not edit the LTX repo, does not patch
    torch.Tensor.to, and does not change VRAM Lab 0.7.4 hook/runtime behavior.

    Why this exists:
    The failed LTX CPU/block-streaming run died before hook attachment at:
        ltx_core.block_streaming.builder._build_pinned_source
        -> allocate_layout_views(blocks_layout, pin_memory=True)
    This guard keeps the same CPU/RAM-backed layout path, but prevents the whole
    layout from being allocated as pinned/page-locked memory upfront. That matches
    the VRAM Lab target better: VRAM-first residency with RAM backing, not a full
    pinned model layout before the denoiser hooks can attach.
    """
    stats = {
        "installed": False,
        "targets": [],
        "calls": 0,
        "rewrites": 0,
        "errors": [],
    }
    ctx["ltx_pinned_layout_guard_installed"] = "NO"
    ctx["ltx_pinned_layout_guard_targets"] = "none"
    ctx["ltx_pinned_layout_guard_calls"] = "0"
    ctx["ltx_pinned_layout_guard_rewrites"] = "0"

    def _make_guard(original: Any, target_name: str) -> Any:
        if getattr(original, "_framevision_vram_lab_pinned_guard", False):
            return original

        def guarded_allocate_layout_views(*args: Any, **kwargs: Any) -> Any:
            stats["calls"] = int(stats.get("calls", 0)) + 1
            changed = False
            new_args = args
            new_kwargs = kwargs

            # Common LTX call: allocate_layout_views(blocks_layout, pin_memory=True)
            if new_kwargs.get("pin_memory") is True:
                new_kwargs = dict(new_kwargs)
                new_kwargs["pin_memory"] = False
                changed = True
            # Be defensive if LTX passes pin_memory positionally later.
            elif len(new_args) >= 3 and new_args[2] is True:
                tmp = list(new_args)
                tmp[2] = False
                new_args = tuple(tmp)
                changed = True

            if changed:
                stats["rewrites"] = int(stats.get("rewrites", 0)) + 1
                ctx["ltx_pinned_layout_guard_calls"] = str(stats["calls"])
                ctx["ltx_pinned_layout_guard_rewrites"] = str(stats["rewrites"])
            return original(*new_args, **new_kwargs)

        try:
            guarded_allocate_layout_views.__name__ = getattr(original, "__name__", "allocate_layout_views")
            guarded_allocate_layout_views.__doc__ = getattr(original, "__doc__", None)
            setattr(guarded_allocate_layout_views, "_framevision_vram_lab_pinned_guard", True)
            setattr(guarded_allocate_layout_views, "_framevision_vram_lab_original", original)
            setattr(guarded_allocate_layout_views, "_framevision_vram_lab_target", target_name)
        except Exception:
            pass
        return guarded_allocate_layout_views

    try:
        utils_mod = importlib.import_module("ltx_core.block_streaming.utils")
        builder_mod = importlib.import_module("ltx_core.block_streaming.builder")
    except Exception as exc:
        msg = f"LTX pinned-layout guard not installed: {type(exc).__name__}: {exc}"
        stats["errors"].append(msg)
        ctx.setdefault("notes", []).append(msg)
        ctx["ltx_pinned_layout_guard_errors"] = " | ".join(stats["errors"])
        return

    for mod, attr, label in (
        (utils_mod, "allocate_layout_views", "ltx_core.block_streaming.utils.allocate_layout_views"),
        # builder.py imports allocate_layout_views directly, so patch that module
        # name too; patching utils alone may not affect the already-bound symbol.
        (builder_mod, "allocate_layout_views", "ltx_core.block_streaming.builder.allocate_layout_views"),
    ):
        try:
            original = getattr(mod, attr, None)
            if original is None:
                stats["errors"].append(f"missing {label}")
                continue
            guarded = _make_guard(original, label)
            setattr(mod, attr, guarded)
            stats["targets"].append(label)
        except Exception as exc:
            stats["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    stats["installed"] = bool(stats["targets"])
    ctx["ltx_pinned_layout_guard_installed"] = "YES" if stats["installed"] else "NO"
    ctx["ltx_pinned_layout_guard_targets"] = " | ".join(stats["targets"]) if stats["targets"] else "none"
    ctx["ltx_pinned_layout_guard_calls"] = str(stats["calls"])
    ctx["ltx_pinned_layout_guard_rewrites"] = str(stats["rewrites"])
    if stats["errors"]:
        ctx["ltx_pinned_layout_guard_errors"] = " | ".join(stats["errors"])
    if stats["installed"]:
        ctx.setdefault("notes", []).append(
            "Installed LTX block-streaming pinned-layout guard: full-layout pin_memory=True allocations are downgraded to pin_memory=False from the FrameVision wrapper."
        )




def _image_preprocess_log(ctx: Dict[str, Any], message: str) -> None:
    ctx.setdefault("image_preprocess_events", []).append(message)
    ctx["image_preprocess_last_event"] = message
    print(f"[vram-lab-ltx] {message}", flush=True)


def _looks_like_image_path(text: Any) -> bool:
    value = str(text or "").strip().strip('"')
    if not value or value.startswith("-"):
        return False
    suffix = Path(value).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jfif", ".gif"}


def _image_size_text(size: Any) -> str:
    try:
        w, h = size
        return f"{int(w)}x{int(h)}"
    except Exception:
        return "unknown"


def _clean_ltx_image_path(original_path: str, temp_dir: Path, ctx: Dict[str, Any]) -> str:
    """Open a user image safely and hand LTX a clean RGB PNG copy.

    The original image is never overwritten and dimensions/aspect ratio are not
    changed here. Existing LTX resize/crop behavior remains responsible for any
    later model-size adaptation.
    """
    original = Path(str(original_path).strip().strip('"')).expanduser()
    _image_preprocess_log(ctx, f"image_preprocess:start path={original}")
    ctx["image_preprocess_last_stage"] = "during image preprocess"
    if not original.exists():
        raise FileNotFoundError(f"input image not found: {original}")

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow/PIL is required to normalize input image before LTX: {type(exc).__name__}: {exc}") from exc

    try:
        with Image.open(original) as img:
            original_format = str(img.format or "unknown")
            original_mode = str(img.mode or "unknown")
            original_size = _image_size_text(img.size)
            _image_preprocess_log(
                ctx,
                f"image_preprocess:opened mode={original_mode} size={original_size} format={original_format}",
            )
            transposed = ImageOps.exif_transpose(img)
            if transposed.mode != "RGB":
                # Convert all normal real-world modes (RGBA/LA/P/CMYK/L/I;16/etc.)
                # into a simple RGB surface. Alpha is flattened by PIL conversion;
                # LTX only receives a clean 3-channel PNG path.
                converted = transposed.convert("RGB")
            else:
                converted = transposed.copy()
            clean_mode = str(converted.mode or "RGB")
            clean_size = _image_size_text(converted.size)
            _image_preprocess_log(ctx, f"image_preprocess:converted mode={clean_mode} size={clean_size}")

            temp_dir.mkdir(parents=True, exist_ok=True)
            stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in original.stem)[:48] or "image"
            digest = hashlib.sha1((str(original.resolve()) + f"|{original.stat().st_mtime_ns}|{time.time_ns()}").encode("utf-8", "ignore")).hexdigest()[:12]
            cleaned = temp_dir / f"ltx_clean_{stem}_{digest}.png"
            converted.save(cleaned, format="PNG")
            _image_preprocess_log(ctx, f"image_preprocess:saved_clean_copy path={cleaned}")

            record = {
                "original_path": str(original),
                "original_mode": original_mode,
                "original_size": original_size,
                "original_format": original_format,
                "cleaned_path": str(cleaned),
                "cleaned_mode": clean_mode,
                "cleaned_size": clean_size,
                "cleaned_format": "PNG",
                "error": "none",
            }
            ctx.setdefault("image_preprocess_records", []).append(record)
            return str(cleaned)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _image_preprocess_log(ctx, f"image_preprocess:failed reason={reason}")
        ctx["image_preprocess_error"] = reason
        ctx.setdefault("image_preprocess_records", []).append({
            "original_path": str(original),
            "original_mode": "unknown",
            "original_size": "unknown",
            "original_format": "unknown",
            "cleaned_path": "none",
            "cleaned_mode": "none",
            "cleaned_size": "none",
            "cleaned_format": "none",
            "error": reason,
        })
        raise RuntimeError(f"image_preprocess failed for {original}: {reason}") from exc


def _ltx_image_temp_dir(args: argparse.Namespace) -> Path:
    root_text = str(getattr(args, "ltx_root", "") or "").strip()
    if root_text:
        return Path(root_text).expanduser() / "temp" / "ltx_image_preprocess"
    return APP_ROOT / "temp" / "ltx_image_preprocess"


def _normalize_ltx_input_images(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Mutate wrapper args so LTX receives clean RGB PNG copies for image inputs."""
    normalize = bool(getattr(args, "ltx_normalize_input_image", True))
    ctx["image_preprocess_enabled"] = "YES" if normalize else "NO"
    ctx["image_preprocess_error"] = "none"
    ctx["image_preprocess_last_stage"] = "before image preprocess"
    ctx["image_preprocess_records"] = []
    ctx["image_preprocess_events"] = []
    if not normalize:
        _image_preprocess_log(ctx, "image_preprocess:disabled")
        return

    temp_dir = _ltx_image_temp_dir(args)
    changed = 0

    if getattr(args, "i2v_image", None):
        ctx["image_preprocess_last_stage"] = "during image preprocess"
        args.i2v_image = _clean_ltx_image_path(str(args.i2v_image), temp_dir, ctx)
        _image_preprocess_log(ctx, f"ltx_image_arg:using_clean_copy path={args.i2v_image}")
        changed += 1

    extra = list(getattr(args, "extra", None) or [])
    out: List[str] = []
    idx = 0
    while idx < len(extra):
        item = str(extra[idx])
        option_name = item.split("=", 1)[0] if item.startswith("--") else item

        if option_name in LTX_SINGLE_IMAGE_PATH_OPTIONS and "=" in item:
            opt, value = item.split("=", 1)
            if _looks_like_image_path(value):
                clean = _clean_ltx_image_path(value, temp_dir, ctx)
                _image_preprocess_log(ctx, f"ltx_image_arg:using_clean_copy path={clean}")
                out.append(f"{opt}={clean}")
                changed += 1
            else:
                out.append(item)
            idx += 1
            continue

        if item in LTX_SINGLE_IMAGE_PATH_OPTIONS and idx + 1 < len(extra):
            out.append(item)
            value = str(extra[idx + 1])
            if _looks_like_image_path(value):
                clean = _clean_ltx_image_path(value, temp_dir, ctx)
                _image_preprocess_log(ctx, f"ltx_image_arg:using_clean_copy path={clean}")
                out.append(clean)
                changed += 1
            else:
                out.append(value)
            idx += 2
            continue

        if item in LTX_MULTI_IMAGE_PATH_OPTIONS:
            out.append(item)
            idx += 1
            while idx < len(extra) and not str(extra[idx]).startswith("--"):
                value = str(extra[idx])
                if _looks_like_image_path(value):
                    clean = _clean_ltx_image_path(value, temp_dir, ctx)
                    _image_preprocess_log(ctx, f"ltx_image_arg:using_clean_copy path={clean}")
                    out.append(clean)
                    changed += 1
                else:
                    out.append(value)
                idx += 1
            continue

        out.append(item)
        idx += 1

    args.extra = out
    ctx["image_preprocess_cleaned_count"] = str(changed)
    if changed <= 0:
        ctx["image_preprocess_enabled"] = "NO: no image input detected"
        _image_preprocess_log(ctx, "image_preprocess:no_image_input_detected")
    else:
        ctx["image_preprocess_last_stage"] = "after image preprocess before LTX run"
        _image_preprocess_log(ctx, f"image_preprocess:complete cleaned_count={changed}")


def _collapse_image_records(ctx: Dict[str, Any], key: str) -> str:
    records = ctx.get("image_preprocess_records") or []
    if not records:
        return "none"
    values = []
    for rec in records:
        if isinstance(rec, dict):
            values.append(str(rec.get(key, "none")))
    return " | ".join(values) if values else "none"

def _has_cli_option(items: List[str], option: str) -> bool:
    """Return True if an argparse-style option is present in a remainder list."""
    prefix = option + "="
    for item in list(items or []):
        text = str(item)
        if text == option or text.startswith(prefix):
            return True
    return False


def _estimate_tensor_layout_bytes(layout: Dict[str, Any]) -> int:
    total = 0
    for _name, spec in dict(layout or {}).items():
        try:
            shape, dtype = spec
            numel = 1
            for dim in tuple(shape):
                numel *= int(dim)
            # torch dtypes all expose element_size via an empty tensor; keep a
            # tiny fallback for odd/meta dtypes.
            try:
                import torch  # type: ignore
                itemsize = int(torch.empty((), dtype=dtype).element_size())
            except Exception:
                itemsize = 2
            total += int(numel) * int(itemsize)
        except Exception:
            continue
    return int(total)


def _install_ltx_gemma_partial_streaming_gate(ctx: Dict[str, Any], allow_main_transformer_streaming: bool = False) -> None:
    """Install a narrow pre-CUDA ownership gate for official LTX Gemma loading.

    The previous full ``--offload cpu`` attempt proved that official LTX can be
    controlled before Gemma fills VRAM, but it pushed the main transformer into
    the official full CPU-pinned streaming path. On a 64GB machine that can fill
    DDR and grind the drive.

    This gate keeps the useful early owner point but changes the shape:
      * PromptEncoder/Gemma may use official streaming before CUDA fill.
      * DiffusionStage/main transformer is forced back to normal GPU path, so it
        does not enter official full CPU offload.
      * Gemma's streaming builder gets a small hybrid source: optional hot
        blocks plus disk-backed cold blocks, with CPU staging unpinned by
        default. That prevents the startup shared-memory spike while avoiding
        the old full-Gemma CUDA slam.

    It does not patch Tensor.to, does not patch Module.to, does not edit LTX repo
    files, and does not modify VRAM Lab 0.7.4 denoise hook logic.
    """
    ctx["ltx_gemma_partial_stream_gate_installed"] = "NO"
    ctx["ltx_gemma_partial_stream_gate_calls"] = "0"
    ctx["ltx_gemma_partial_stream_gate_hot_blocks"] = "0"
    ctx["ltx_gemma_partial_stream_gate_disk_slots"] = "0"
    ctx["ltx_gemma_partial_stream_gate_pinned_budget_gb"] = "0"
    ctx["ltx_gemma_partial_stream_gate_main_transformer"] = "not patched"
    ctx["ltx_gemma_partial_stream_gate_errors"] = "none"
    ctx["ltx_gemma_checkpoint_prefix_selected"] = "n/a"
    ctx["ltx_gemma_checkpoint_prefix_candidates"] = "n/a"
    ctx["ltx_gemma_checkpoint_block_indices"] = "0"
    ctx["ltx_gemma_checkpoint_non_block_keys"] = "0"
    ctx["ltx_gemma_checkpoint_prefix_note"] = "n/a"

    try:
        import torch  # type: ignore
        import ltx_core.block_streaming.builder as builder_mod  # type: ignore
        import ltx_pipelines.utils.blocks as pipeline_blocks_mod  # type: ignore
        from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource  # type: ignore
        from ltx_core.block_streaming.pool import WeightPool  # type: ignore
        from ltx_core.block_streaming.provider import WeightsProvider  # type: ignore
        from ltx_core.block_streaming.wrapper import BlockStreamingWrapper  # type: ignore
        from ltx_core.block_streaming.utils import allocate_layout_views, derive_layout, make_block_key, resolve_attr  # type: ignore
        from ltx_core.loader.helpers import create_meta_model, read_model_config  # type: ignore
        from ltx_pipelines.utils.types import OffloadMode  # type: ignore
    except Exception as exc:
        ctx["ltx_gemma_partial_stream_gate_errors"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    # Startup VRAM guard:
    # The previous shared-memory guard stopped pinned/shared staging but still
    # allowed Gemma to preallocate a very large CUDA slot pool. That showed up as
    # the 9s->12s jump from ~2.7GB to ~20GB allocated before the text encoder
    # flushed. Keep Gemma streaming, keep CPU staging unpinned, and lower the
    # default startup CUDA target so the builder cannot shove ~20GB into VRAM
    # before the wrapper is ready. Advanced testers can override with
    # FRAMEVISION_LTX_GEMMA_TARGET_VRAM_GB.
    try:
        hot_budget_gb = float(os.environ.get("FRAMEVISION_LTX_GEMMA_HOT_PIN_GB", "0.0") or "0.0")
    except Exception:
        hot_budget_gb = 0.0
    try:
        gemma_target_vram_gb = float(os.environ.get("FRAMEVISION_LTX_GEMMA_TARGET_VRAM_GB", "12.0") or "12.0")
    except Exception:
        gemma_target_vram_gb = 12.0
    try:
        pin_gemma_env = os.environ.get("FRAMEVISION_LTX_GEMMA_PIN_STAGING", "0").strip().lower()
        pin_gemma_staging = pin_gemma_env in {"1", "true", "yes", "on"}
    except Exception:
        pin_gemma_staging = False
    try:
        disk_slots_count = int(os.environ.get("FRAMEVISION_LTX_GEMMA_DISK_SLOTS", "6") or "6")
    except Exception:
        disk_slots_count = 6
    try:
        _gpu_slots_env = os.environ.get("FRAMEVISION_LTX_GEMMA_GPU_SLOTS")
        gpu_slots_count_default = int(_gpu_slots_env) if _gpu_slots_env not in (None, "") else None
    except Exception:
        gpu_slots_count_default = None

    hot_budget_bytes = max(0, int(hot_budget_gb * 1024**3))
    gemma_target_vram_bytes = max(0, int(gemma_target_vram_gb * 1024**3))
    disk_slots_count = max(1, disk_slots_count)
    if gpu_slots_count_default is not None:
        gpu_slots_count_default = max(2, gpu_slots_count_default)

    class _FrameVisionPartialPinnedDiskSource:
        def __init__(self, pinned: Dict[int, Dict[str, Any]], disk_source: Any, block_layout: Dict[str, Any]) -> None:
            self._pinned = pinned
            self._disk_source = disk_source
            self._block_layout = block_layout

        @property
        def block_layout(self) -> Dict[str, Any]:
            return self._block_layout

        def get(self, idx: int) -> Dict[str, Any]:
            if idx in self._pinned:
                return self._pinned[idx]
            return self._disk_source.get(idx)

        def release(self, idx: int, event: Any) -> None:
            if idx in self._pinned:
                return
            self._disk_source.release(idx, event)

        def cleanup(self) -> None:
            self._pinned.clear()
            self._disk_source.cleanup()

        def __len__(self) -> int:
            try:
                return len(self._pinned) + len(self._disk_source)
            except Exception:
                return len(self._pinned)

    original_build = getattr(builder_mod.StreamingModelBuilder, "build", None)
    if original_build is None:
        ctx["ltx_gemma_partial_stream_gate_errors"] = "StreamingModelBuilder.build not found"
        return
    if getattr(original_build, "_framevision_ltx_gemma_partial_stream_gate", False):
        ctx["ltx_gemma_partial_stream_gate_installed"] = "YES: already installed"
    else:
        def guarded_build(self: Any, target_device: Any, dtype: Any, cpu_slots_count: int | None = None, gpu_slots_count: int | None = None, **kwargs: Any) -> Any:
            blocks_attr = str(getattr(self, "blocks_attr", "") or "")
            blocks_prefix = str(getattr(self, "blocks_prefix", "") or "")
            is_gemma_language_layers = (
                blocks_attr == "model.model.language_model.layers"
                or blocks_prefix == "model.model.language_model.layers"
            )
            # Only replace official CPU offload's full-pinned Gemma path. Disk
            # mode and all transformer/non-Gemma streaming behavior stay with
            # official LTX unless explicitly changed elsewhere.
            if not is_gemma_language_layers or cpu_slots_count is not None:
                return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

            ctx["ltx_gemma_partial_stream_gate_calls"] = str(int(str(ctx.get("ltx_gemma_partial_stream_gate_calls", "0") or "0")) + 1)
            checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
            config = read_model_config(self.model_path, self.model_loader)
            meta_model = create_meta_model(self.model_class_configurator, config, self.module_ops)
            if self.model_wrapper is not None:
                meta_model = self.model_wrapper(meta_model)
            meta_model.eval()

            blocks = resolve_attr(meta_model, self.blocks_attr)
            def _candidate_prefixes(prefix: str) -> List[str]:
                base = str(prefix or "")
                candidates: List[str] = []
                def add(value: str) -> None:
                    value = str(value or "").strip(".")
                    if value and value not in candidates:
                        candidates.append(value)
                add(base)
                # Gemma checkpoints found in the wild differ in how much model
                # nesting is present after SDOps. If the prefix is wrong, all
                # decoder layers are treated as non-block weights and the text
                # encoder gets shoved into CUDA at startup (~20GB). Try the
                # common variants and pick the one that actually discovers the
                # language-layer block weights.
                if base.startswith("model.model."):
                    add(base[len("model."):])
                    add(base[len("model.model."):])
                if base.startswith("model."):
                    add(base[len("model."):])
                    add("model." + base)
                add("model.model.language_model.layers")
                add("model.language_model.layers")
                add("language_model.layers")
                return candidates

            prefix_results: List[tuple[str, Dict[int, List[tuple[str, str]]], List[tuple[str, str]]]] = []
            for _prefix in _candidate_prefixes(self.blocks_prefix):
                try:
                    _bmap, _non = builder_mod._scan_checkpoint_keys(checkpoint_paths, self.model_sd_ops, _prefix)
                    prefix_results.append((_prefix, _bmap, _non))
                except Exception:
                    continue
            if prefix_results:
                chosen_prefix, block_key_map, non_block_keys = max(
                    prefix_results,
                    key=lambda item: (len(item[1]), sum(len(v) for v in item[1].values()), -len(item[2])),
                )
            else:
                chosen_prefix = self.blocks_prefix
                block_key_map, non_block_keys = builder_mod._scan_checkpoint_keys(checkpoint_paths, self.model_sd_ops, self.blocks_prefix)
            if not block_key_map:
                # Hard fallback to the official builder rather than repeating
                # the broken behavior where every decoder layer becomes a
                # non-block CUDA load. This makes a missed prefix obvious and
                # prevents the guard from pretending it worked.
                ctx["ltx_gemma_checkpoint_prefix_note"] = "no block keys found; falling back to official builder"
                return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)
            ctx["ltx_gemma_checkpoint_prefix_selected"] = str(chosen_prefix)
            ctx["ltx_gemma_checkpoint_prefix_candidates"] = " | ".join(
                f"{p}:blocks={len(b)}:entries={sum(len(v) for v in b.values())}:non={len(n)}"
                for p, b, n in prefix_results
            ) or "none"
            ctx["ltx_gemma_checkpoint_block_indices"] = str(len(block_key_map))
            ctx["ltx_gemma_checkpoint_non_block_keys"] = str(len(non_block_keys))
            ctx["ltx_gemma_checkpoint_prefix_note"] = (
                "selected checkpoint prefix by scanning safetensors keys before non-block load; "
                "this prevents Gemma language blocks from being misclassified as startup CUDA non-block weights"
            )
            reader = DiskTensorReader(checkpoint_paths)
            lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]

            # Non-block Gemma weights still need to live on the target device for
            # compute, but this is much smaller than the full language stack and
            # avoids the old 22.8GB first-seen slam.
            self._load_non_block_weights(
                reader,
                non_block_keys,
                meta_model,
                target_device,
                dtype,
                sd_ops=self.model_sd_ops,
                key_prefix=self.state_dict_prefix,
                lora_sources=lora_sources,
            )

            base_layout = derive_layout(dict(blocks[0].named_parameters()), dtype)
            per_block_bytes = max(1, _estimate_tensor_layout_bytes(base_layout))
            available_indices = sorted(int(i) for i in block_key_map.keys())
            max_hot_by_budget = int(hot_budget_bytes // per_block_bytes) if hot_budget_bytes > 0 else 0
            hot_count = min(len(available_indices), max(0, max_hot_by_budget))
            hot_indices = set(available_indices[:hot_count])

            pinned: Dict[int, Dict[str, Any]] = {}
            if hot_indices:
                hot_memory_layout = {
                    f"{block_idx}/{name}": spec
                    for block_idx in sorted(hot_indices)
                    for name, spec in base_layout.items()
                }
                hot_views = allocate_layout_views(hot_memory_layout, device=torch.device("cpu"), pin_memory=pin_gemma_staging)
                block_reader = DiskBlockReader(
                    reader=reader,
                    block_key_map=block_key_map,
                    sd_ops=self.model_sd_ops,
                    blocks_prefix=chosen_prefix,
                )
                for block_idx in sorted(hot_indices):
                    target = {name: hot_views[f"{block_idx}/{name}"] for name in base_layout}
                    block_reader.read_into(target, block_idx)
                    pinned[block_idx] = target

            # Cold blocks use a small pinned CPU LRU pool. The reader is shared;
            # it remains open until wrapper teardown.
            disk_pool = WeightPool(
                base_layout,
                min(max(1, disk_slots_count), max(1, len(available_indices))),
                torch.device("cpu"),
                reuse_barrier=lambda event: event.synchronize(),
                pin_memory=pin_gemma_staging,
            )
            disk_reader = DiskBlockReader(
                reader=reader,
                block_key_map=block_key_map,
                sd_ops=self.model_sd_ops,
                blocks_prefix=chosen_prefix,
            )
            disk_source = builder_mod.DiskWeightSource(disk_pool, disk_reader)
            source = _FrameVisionPartialPinnedDiskSource(pinned, disk_source, base_layout)

            try:
                alloc_before_gpu_pool = int(torch.cuda.memory_allocated(target_device))
            except Exception:
                alloc_before_gpu_pool = 0
            if gpu_slots_count is not None:
                resolved_gpu_slots = int(gpu_slots_count)
            elif gpu_slots_count_default is not None:
                resolved_gpu_slots = int(gpu_slots_count_default)
            else:
                # Target total Gemma-stage allocated VRAM, not just block slots.
                # This keeps the default near 16GB without hardcoding a slot count
                # that may change if the model/block layout changes.
                remaining_for_slots = max(per_block_bytes * 2, gemma_target_vram_bytes - alloc_before_gpu_pool)
                resolved_gpu_slots = int(remaining_for_slots // per_block_bytes)
            resolved_gpu_slots = max(2, min(max(2, len(available_indices)), resolved_gpu_slots))

            copy_stream = torch.cuda.Stream(device=target_device)
            gpu_pool = WeightPool(
                source.block_layout,
                resolved_gpu_slots,
                target_device,
                reuse_barrier=lambda event: copy_stream.wait_event(event),
            )
            provider = WeightsProvider(gpu_pool, copy_stream, target_device, source, lora_sources, chosen_prefix)
            ctx["ltx_gemma_partial_stream_gate_hot_blocks"] = str(len(pinned))
            ctx["ltx_gemma_partial_stream_gate_disk_slots"] = str(disk_slots_count)
            ctx["ltx_gemma_partial_stream_gate_gpu_slots"] = str(resolved_gpu_slots)
            ctx["ltx_gemma_partial_stream_gate_target_vram_gb"] = f"{gemma_target_vram_gb:.2f}"
            ctx["ltx_gemma_partial_stream_gate_alloc_before_gpu_pool"] = _fmt_bytes(alloc_before_gpu_pool)
            ctx["ltx_gemma_partial_stream_gate_pinned_budget_gb"] = f"{hot_budget_gb:.2f}"
            ctx["ltx_gemma_partial_stream_gate_per_block"] = _fmt_bytes(per_block_bytes)
            ctx["ltx_gemma_startup_shared_guard"] = "ON"
            ctx["ltx_gemma_startup_pin_staging"] = "ON" if pin_gemma_staging else "OFF"
            ctx["ltx_gemma_startup_shared_guard_note"] = (
                "Gemma startup CUDA target capped lower to avoid the old ~20GB pre-wrapper shove; "
                "CPU staging remains unpinned by default; set FRAMEVISION_LTX_GEMMA_TARGET_VRAM_GB / "
                "FRAMEVISION_LTX_GEMMA_PIN_STAGING to override for tests."
            )
            ctx["ltx_gemma_partial_stream_gate_status"] = (
                f"Gemma startup shared-memory guard: target about {gemma_target_vram_gb:.1f}GB VRAM with {resolved_gpu_slots} GPU slots; "
                f"CPU hot cache {len(pinned)} block(s) from {hot_budget_gb:.1f}GB budget; "
                f"CPU staging pin_memory={'ON' if pin_gemma_staging else 'OFF'}; "
                f"{disk_slots_count} disk-backed cold slot(s); main transformer streaming probe={'ON' if allow_main_transformer_streaming else 'OFF'}."
            )
            return BlockStreamingWrapper(
                model=meta_model,
                blocks=blocks,
                provider=provider,
                target_device=target_device,
            )

        try:
            guarded_build.__name__ = getattr(original_build, "__name__", "build")
            guarded_build.__doc__ = getattr(original_build, "__doc__", None)
            setattr(guarded_build, "_framevision_ltx_gemma_partial_stream_gate", True)
            setattr(guarded_build, "_framevision_ltx_gemma_partial_stream_original", original_build)
            setattr(builder_mod.StreamingModelBuilder, "build", guarded_build)
            ctx["ltx_gemma_partial_stream_gate_installed"] = "YES"
        except Exception as exc:
            ctx["ltx_gemma_partial_stream_gate_errors"] = f"StreamingModelBuilder.build patch failed: {type(exc).__name__}: {exc}"
            return

    # Critical part: when the wrapper injects --offload cpu, official LTX would
    # normally stream Gemma AND the main transformer. Default remains the proven
    # Gemma-only path. The experimental main-transformer probe deliberately skips
    # this forced-NONE guard so the main DiffusionStage StreamingModelBuilder can
    # be inspected/wrapped before full CPU materialization.
    if allow_main_transformer_streaming:
        ctx["ltx_gemma_partial_stream_gate_main_transformer"] = "probe mode: DiffusionStage offload_mode is NOT forced to NONE"
        ctx.setdefault("notes", []).append(
            "Installed LTX Gemma partial streaming gate with main-transformer probe enabled: DiffusionStage offload is allowed so the main StreamingModelBuilder path can be inspected."
        )
        return
    try:
        original_diffusion_init = getattr(pipeline_blocks_mod.DiffusionStage, "__init__")
        if not getattr(original_diffusion_init, "_framevision_ltx_main_transformer_gpu_gate", False):
            def guarded_diffusion_init(self: Any, *args: Any, **kwargs: Any) -> None:
                if kwargs.get("offload_mode", OffloadMode.NONE) != OffloadMode.NONE:
                    kwargs = dict(kwargs)
                    kwargs["offload_mode"] = OffloadMode.NONE
                    ctx["ltx_gemma_partial_stream_gate_main_transformer"] = "forced DiffusionStage offload_mode=NONE; Gemma-only streaming gate active"
                return original_diffusion_init(self, *args, **kwargs)

            guarded_diffusion_init.__name__ = getattr(original_diffusion_init, "__name__", "__init__")
            guarded_diffusion_init.__doc__ = getattr(original_diffusion_init, "__doc__", None)
            setattr(guarded_diffusion_init, "_framevision_ltx_main_transformer_gpu_gate", True)
            setattr(guarded_diffusion_init, "_framevision_ltx_main_transformer_gpu_original", original_diffusion_init)
            setattr(pipeline_blocks_mod.DiffusionStage, "__init__", guarded_diffusion_init)
        ctx.setdefault("notes", []).append(
            "Installed LTX Gemma-only partial streaming gate: wrapper may request official --offload cpu, but DiffusionStage/main transformer is forced back to normal GPU path; Gemma VRAM and CPU hot-pin budgets now follow the selected 12/16/24GB profile."
        )
    except Exception as exc:
        ctx["ltx_gemma_partial_stream_gate_errors"] = f"DiffusionStage guard failed: {type(exc).__name__}: {exc}"




def _install_ltx_main_transformer_streaming_probe(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Experimental main LTX transformer StreamingModelBuilder probe.

    This is intentionally opt-in. It lets the official main transformer CPU
    offload/streaming builder be reached, logs what it exposes, and tries a
    Gemma-style partial streaming wrapper for recognized transformer block
    containers. It falls back to the current builder if recognition/wrapping is
    not safe.
    """
    ctx["main_transformer_streaming_probe"] = "ON"
    ctx["main_transformer_partial_streaming_installed"] = "NO"
    ctx["main_transformer_partial_streaming_reason"] = "waiting for StreamingModelBuilder.build"
    ctx["main_transformer_streaming_builder_reached"] = "NO"
    ctx["main_transformer_streaming_builder_calls"] = "0"
    ctx["main_transformer_detected_block_path"] = "none"
    ctx["main_transformer_block_count"] = "0"
    ctx["main_transformer_per_block_size"] = "n/a"
    ctx["main_transformer_hot_cpu_budget"] = "bounded by staging guard"
    ctx["main_transformer_hot_cpu_slots"] = "0"
    ctx["main_transformer_disk_slots"] = "1"
    ctx["main_transformer_streaming_staging_guard"] = "ON"
    ctx["main_transformer_streaming_cpu_pin_memory"] = "OFF"
    ctx["main_transformer_streaming_disk_pin_memory"] = "OFF"
    ctx["main_transformer_streaming_estimated_pinned_bytes"] = "0 B"
    ctx["main_transformer_streaming_staging_note"] = "main streaming keeps fast GPU transfer but avoids large pinned/mapped CPU staging pools"
    ctx["main_transformer_gpu_slots"] = "0"
    ctx["main_transformer_gpu_target"] = str(ctx.get("ltx_main_profile_hot_window_gb", "profile"))
    ctx["main_transformer_full_cpu_pinned_layout_avoided"] = "unknown"
    ctx["main_transformer_streaming_probe_errors"] = "none"
    ctx.setdefault("main_transformer_streaming_probe_events", [])

    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        import ltx_core.block_streaming.builder as builder_mod  # type: ignore
        from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource  # type: ignore
        from ltx_core.block_streaming.pool import WeightPool  # type: ignore
        from ltx_core.block_streaming.provider import WeightsProvider  # type: ignore
        from ltx_core.block_streaming.wrapper import BlockStreamingWrapper  # type: ignore
        from ltx_core.block_streaming.utils import allocate_layout_views, derive_layout, resolve_attr  # type: ignore
        from ltx_core.loader.helpers import create_meta_model, read_model_config  # type: ignore
    except Exception as exc:
        ctx["main_transformer_partial_streaming_reason"] = f"import failed: {type(exc).__name__}: {exc}"
        ctx["main_transformer_streaming_probe_errors"] = ctx["main_transformer_partial_streaming_reason"]
        return

    def _probe_event(label: str, detail: str) -> None:
        line = f"{label}: {detail} | {_full_memory_snapshot(torch)}"
        try:
            events = ctx.setdefault("main_transformer_streaming_probe_events", [])
            if isinstance(events, list) and len(events) < 300:
                events.append(line)
            ctx["main_transformer_streaming_probe_event_count"] = str(len(events) if isinstance(events, list) else 0)
        except Exception:
            pass
        if echo:
            try:
                print(f"[vram-lab-main-stream-probe] {line}", flush=True)
            except Exception:
                pass

    def _builder_text(self: Any) -> str:
        parts: List[str] = []
        for attr in ("blocks_attr", "blocks_prefix", "state_dict_prefix"):
            try:
                parts.append(f"{attr}={getattr(self, attr, None)}")
            except Exception:
                pass
        try:
            cfg = getattr(getattr(self, "model_class_configurator", None), "__name__", str(getattr(self, "model_class_configurator", None)))
            parts.append(f"configurator={cfg}")
        except Exception:
            pass
        try:
            paths = _normalise_model_paths(getattr(self, "model_path", None))
            parts.append(_file_size_summary(paths))
            parts.append("paths=" + " | ".join(paths[:3]))
        except Exception:
            pass
        return "; ".join(str(x) for x in parts if x is not None)

    def _is_main_candidate(self: Any) -> bool:
        blocks_attr = str(getattr(self, "blocks_attr", "") or "")
        blocks_prefix = str(getattr(self, "blocks_prefix", "") or "")
        kind = _classify_ltx_model_builder(self)
        text = f"{kind} {blocks_attr} {blocks_prefix} {_builder_text(self)}".lower()
        if blocks_attr == "model.model.language_model.layers" or blocks_prefix == "model.model.language_model.layers":
            return False
        if any(x in text for x in ("transformer_blocks", "diffusion_model", "velocity_model", "ltxmodel", "main_transformer", "ltx-2.3", "distilled-1.1")):
            return True
        return kind == "main_transformer"

    original_build = getattr(builder_mod.StreamingModelBuilder, "build", None)
    if original_build is None:
        ctx["main_transformer_partial_streaming_reason"] = "StreamingModelBuilder.build not found"
        return
    if getattr(original_build, "_framevision_ltx_main_transformer_stream_probe", False):
        ctx["main_transformer_partial_streaming_reason"] = "already installed"
        return

    class _FrameVisionMainPartialPinnedDiskSource:
        def __init__(self, pinned: Dict[int, Dict[str, Any]], disk_source: Any, block_layout: Dict[str, Any]) -> None:
            self._pinned = pinned
            self._disk_source = disk_source
            self._block_layout = block_layout

        @property
        def block_layout(self) -> Dict[str, Any]:
            return self._block_layout

        def get(self, idx: int) -> Dict[str, Any]:
            if idx in self._pinned:
                return self._pinned[idx]
            return self._disk_source.get(idx)

        def release(self, idx: int, event: Any) -> None:
            if idx in self._pinned:
                return
            return self._disk_source.release(idx, event)

        def cleanup(self) -> None:
            try:
                self._pinned.clear()
            except Exception:
                pass
            try:
                self._disk_source.cleanup()
            except Exception:
                pass

        def __len__(self) -> int:
            try:
                return len(self._pinned) + len(self._disk_source)
            except Exception:
                return len(self._pinned)

    def guarded_build(self: Any, target_device: Any, dtype: Any, cpu_slots_count: int | None = None, gpu_slots_count: int | None = None, **kwargs: Any) -> Any:
        blocks_attr = str(getattr(self, "blocks_attr", "") or "")
        blocks_prefix = str(getattr(self, "blocks_prefix", "") or "")
        kind = _classify_ltx_model_builder(self)
        ctx["main_transformer_streaming_builder_calls"] = str(int(str(ctx.get("main_transformer_streaming_builder_calls", "0") or "0")) + 1)
        detail = (
            f"kind={kind}; blocks_attr={blocks_attr}; blocks_prefix={blocks_prefix}; target={target_device}; dtype={dtype}; "
            f"cpu_slots={cpu_slots_count}; gpu_slots={gpu_slots_count}; {_builder_text(self)}"
        )
        _probe_event("StreamingModelBuilder.build:start", detail)

        if not _is_main_candidate(self):
            return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

        ctx["main_transformer_streaming_builder_reached"] = "YES"
        try:
            main_build_count = int(ctx.get("_main_transformer_streaming_main_build_count", 0) or 0) + 1
        except Exception:
            main_build_count = 1
        ctx["_main_transformer_streaming_main_build_count"] = main_build_count
        main_build_role = "stage1_initial_denoise" if main_build_count == 1 else "stage2_refine_denoise" if main_build_count == 2 else f"main_transformer_build_{main_build_count}"
        try:
            main_build_limit_gb, main_build_limit_source = _block_limit_for_role(ctx, main_build_role)
        except Exception:
            main_build_limit_gb, main_build_limit_source = (0.0, "profile fallback")
        ctx[f"main_transformer_streaming_build_{main_build_count}_role"] = main_build_role
        ctx[f"main_transformer_streaming_build_{main_build_count}_limit"] = f"{float(main_build_limit_gb):.1f} GB ({main_build_limit_source})"
        ctx["main_transformer_detected_block_path"] = blocks_attr or blocks_prefix or "unknown"
        try:
            checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
            ctx["main_transformer_checkpoint_summary"] = _file_size_summary([str(x) for x in checkpoint_paths])
            config = read_model_config(self.model_path, self.model_loader)
            meta_model = create_meta_model(self.model_class_configurator, config, self.module_ops)
            if self.model_wrapper is not None:
                meta_model = self.model_wrapper(meta_model)
            try:
                meta_model.eval()
            except Exception:
                pass

            candidate_paths = []
            for attr in (blocks_attr, "velocity_model.transformer_blocks", "transformer_blocks", "model.diffusion_model.transformer_blocks", "diffusion_model.transformer_blocks"):
                if attr and attr not in candidate_paths:
                    candidate_paths.append(attr)
            blocks = None
            found_path = "none"
            last_err = ""
            for path in candidate_paths:
                try:
                    blocks = resolve_attr(meta_model, path)
                    found_path = path
                    break
                except Exception as exc:
                    last_err = f"{path}: {type(exc).__name__}: {exc}"
            if blocks is None:
                ctx["main_transformer_partial_streaming_reason"] = f"no recognized block path; last={last_err or 'none'}"
                ctx["main_transformer_partial_streaming_installed"] = "NO"
                _probe_event("main_builder:no_block_path", ctx["main_transformer_partial_streaming_reason"])
                return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

            try:
                block_count = len(blocks)
            except Exception:
                block_count = 0
            ctx["main_transformer_detected_block_path"] = found_path
            ctx["main_transformer_block_count"] = str(block_count)
            if not block_count:
                ctx["main_transformer_partial_streaming_reason"] = "detected block path has no measurable blocks"
                return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

            _probe_event("before main builder/build", f"path={found_path}; blocks={block_count}")
            block_key_map, non_block_keys = builder_mod._scan_checkpoint_keys(checkpoint_paths, self.model_sd_ops, self.blocks_prefix)
            reader = DiskTensorReader(checkpoint_paths)
            lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]
            self._load_non_block_weights(
                reader,
                non_block_keys,
                meta_model,
                target_device,
                dtype,
                sd_ops=self.model_sd_ops,
                key_prefix=self.state_dict_prefix,
                lora_sources=lora_sources,
            )
            _probe_event("after main load_state_dict", f"non_block_keys={len(non_block_keys) if hasattr(non_block_keys, '__len__') else 'n/a'}")

            base_layout = derive_layout(dict(blocks[0].named_parameters()), dtype)
            per_block_bytes = max(1, _estimate_tensor_layout_bytes(base_layout))
            ctx["main_transformer_per_block_size"] = _fmt_bytes(per_block_bytes)
            available_indices = sorted(int(i) for i in block_key_map.keys())
            if not available_indices:
                ctx["main_transformer_partial_streaming_reason"] = "checkpoint scan returned no block indices"
                _probe_event("main_builder:no_block_indices", ctx["main_transformer_partial_streaming_reason"])
                return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

            # Main-transformer streaming staging guard.
            #
            # The first probe proved that the useful fast DDR -> VRAM transfer path can also
            # make Windows/NVIDIA report a large "Shared GPU memory" jump.  That does not
            # behave like true active VRAM spill, but a large pinned/mapped staging side pool
            # still slows later steps and can erase the gains from streaming.  Keep the
            # streaming wrapper, but stop the main transformer from creating multi-GB pinned
            # CPU staging pools.  Gemma keeps its own separate tuned path above.
            try:
                env_hot = os.environ.get("FRAMEVISION_LTX_MAIN_STREAM_HOT_CPU_GB", "").strip()
                hot_budget_gb = float(env_hot) if env_hot else max(0.0, min(0.85, float(per_block_bytes) / float(1024**3)))
            except Exception:
                hot_budget_gb = max(0.0, min(0.85, float(per_block_bytes) / float(1024**3)))
            try:
                disk_slots_count = int(os.environ.get("FRAMEVISION_LTX_MAIN_STREAM_DISK_SLOTS", "1") or "1")
            except Exception:
                disk_slots_count = 1
            disk_slots_count = max(1, min(2, disk_slots_count))
            pin_env = os.environ.get("FRAMEVISION_LTX_MAIN_STREAM_PIN_STAGING", "0").strip().lower()
            pin_main_staging = pin_env in {"1", "true", "yes", "on"}
            hot_budget_bytes = int(max(0.0, hot_budget_gb) * 1024**3)
            hot_count = min(len(available_indices), int(hot_budget_bytes // per_block_bytes)) if hot_budget_bytes > 0 else 0
            # If the per-block estimate is larger than the budget, keep one tiny hot cache
            # only when explicit pinning is requested.  Default is disk-backed cold source
            # plus GPU pool, with no long-lived pinned CPU hot block.
            if hot_count <= 0 and pin_main_staging:
                hot_count = 1
            hot_indices = set(available_indices[:max(0, hot_count)])
            pinned: Dict[int, Dict[str, Any]] = {}
            estimated_pinned_bytes = 0
            if hot_indices:
                hot_memory_layout = {f"{block_idx}/{name}": spec for block_idx in sorted(hot_indices) for name, spec in base_layout.items()}
                hot_views = allocate_layout_views(hot_memory_layout, device=torch.device("cpu"), pin_memory=pin_main_staging)
                estimated_pinned_bytes += (per_block_bytes * len(hot_indices)) if pin_main_staging else 0
                block_reader = DiskBlockReader(reader=reader, block_key_map=block_key_map, sd_ops=self.model_sd_ops, blocks_prefix=self.blocks_prefix)
                for block_idx in sorted(hot_indices):
                    target = {name: hot_views[f"{block_idx}/{name}"] for name in base_layout}
                    block_reader.read_into(target, block_idx)
                    pinned[block_idx] = target

            disk_pool_slots = min(max(1, disk_slots_count), max(1, len(available_indices)))
            estimated_pinned_bytes += (per_block_bytes * disk_pool_slots) if pin_main_staging else 0
            disk_pool = WeightPool(base_layout, disk_pool_slots, torch.device("cpu"), reuse_barrier=lambda event: event.synchronize(), pin_memory=pin_main_staging)
            disk_reader = DiskBlockReader(reader=reader, block_key_map=block_key_map, sd_ops=self.model_sd_ops, blocks_prefix=self.blocks_prefix)
            disk_source = builder_mod.DiskWeightSource(disk_pool, disk_reader)
            source = _FrameVisionMainPartialPinnedDiskSource(pinned, disk_source, base_layout)

            try:
                profile_target_gb = float(main_build_limit_gb or ctx.get("ltx_main_profile_hot_window_gb", 0.0) or 0.0)
            except Exception:
                profile_target_gb = 0.0
            try:
                alloc_before = int(torch.cuda.memory_allocated(target_device))
            except Exception:
                alloc_before = 0
            if gpu_slots_count is not None:
                resolved_gpu_slots = int(gpu_slots_count)
            else:
                target_bytes = int(max(0.0, profile_target_gb) * 1024**3)
                remaining = max(per_block_bytes * 2, target_bytes - alloc_before) if target_bytes else per_block_bytes * 2
                resolved_gpu_slots = int(remaining // per_block_bytes)
            resolved_gpu_slots = max(1, min(max(1, len(available_indices)), resolved_gpu_slots))
            copy_stream = torch.cuda.Stream(device=target_device)
            gpu_pool = WeightPool(source.block_layout, resolved_gpu_slots, target_device, reuse_barrier=lambda event: copy_stream.wait_event(event))
            provider = WeightsProvider(gpu_pool, copy_stream, target_device, source, lora_sources, self.blocks_prefix)

            ctx["main_transformer_partial_streaming_installed"] = "YES"
            ctx["main_transformer_partial_streaming_reason"] = "installed experimental partial pinned/disk/GPU provider"
            ctx["main_transformer_hot_cpu_budget"] = f"{hot_budget_gb:.2f} GB"
            ctx["main_transformer_hot_cpu_slots"] = str(len(pinned))
            ctx["main_transformer_disk_slots"] = str(disk_slots_count)
            ctx["main_transformer_streaming_staging_guard"] = "ON"
            ctx["main_transformer_streaming_cpu_pin_memory"] = "ON" if pin_main_staging else "OFF"
            ctx["main_transformer_streaming_disk_pin_memory"] = "ON" if pin_main_staging else "OFF"
            ctx["main_transformer_streaming_estimated_pinned_bytes"] = _fmt_bytes(estimated_pinned_bytes)
            ctx["main_transformer_streaming_staging_note"] = (
                "bounded main streaming staging: GPU hot-window is unchanged, but long-lived CPU hot/disk staging "
                "uses unpinned RAM by default to avoid Task-Manager shared-GPU-memory pollution"
            )
            ctx["main_transformer_gpu_slots"] = str(resolved_gpu_slots)
            ctx["main_transformer_gpu_target"] = f"{main_build_role} hot-window {profile_target_gb:.2f} GB ({main_build_limit_source}); alloc_before_gpu_pool={_fmt_bytes(alloc_before)}"
            ctx["main_transformer_full_cpu_pinned_layout_avoided"] = "YES: full pinned CPU layout avoided; main streaming staging guard also avoids long-lived pinned disk/hot pools by default"
            _probe_event("after main builder/build", f"role={main_build_role}; active_limit={profile_target_gb:.1f}GB ({main_build_limit_source}); installed path={found_path}; hot_cpu={len(pinned)}; disk_slots={disk_slots_count}; gpu_slots={resolved_gpu_slots}; pin_staging={'ON' if pin_main_staging else 'OFF'}; est_pinned={_fmt_bytes(estimated_pinned_bytes)}")
            try:
                gc.collect()
            except Exception:
                pass
            _probe_event("after state_dict cleanup/del/gc", "partial streaming provider ready")
            _probe_event("before first denoise step", "returning BlockStreamingWrapper for main transformer")
            return BlockStreamingWrapper(model=meta_model, blocks=blocks, provider=provider, target_device=target_device)
        except Exception as exc:
            ctx["main_transformer_partial_streaming_installed"] = "NO"
            ctx["main_transformer_partial_streaming_reason"] = f"failed to create wrapper; fallback to official builder: {type(exc).__name__}: {exc}"
            ctx["main_transformer_streaming_probe_errors"] = ctx["main_transformer_partial_streaming_reason"]
            _probe_event("main_builder:fallback", ctx["main_transformer_partial_streaming_reason"])
            return original_build(self, target_device=target_device, dtype=dtype, cpu_slots_count=cpu_slots_count, gpu_slots_count=gpu_slots_count, **kwargs)

    try:
        guarded_build.__name__ = getattr(original_build, "__name__", "build")
        guarded_build.__doc__ = getattr(original_build, "__doc__", None)
        setattr(guarded_build, "_framevision_ltx_main_transformer_stream_probe", True)
        setattr(guarded_build, "_framevision_ltx_main_transformer_stream_probe_original", original_build)
        setattr(builder_mod.StreamingModelBuilder, "build", guarded_build)
        ctx["main_transformer_partial_streaming_reason"] = "probe installed; waiting for main StreamingModelBuilder"
        ctx.setdefault("notes", []).append("Installed experimental main transformer streaming probe with bounded staging guard: keeps the streaming path but avoids long-lived pinned CPU hot/disk pools by default.")
    except Exception as exc:
        ctx["main_transformer_partial_streaming_reason"] = f"install failed: {type(exc).__name__}: {exc}"
        ctx["main_transformer_streaming_probe_errors"] = ctx["main_transformer_partial_streaming_reason"]


def _normalise_model_paths(value: Any) -> List[str]:
    try:
        if value is None:
            return []
        if isinstance(value, (str, os.PathLike)):
            return [str(value)]
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]
    except Exception:
        pass
    return [str(value)]


def _file_size_summary(paths: List[str]) -> str:
    total = 0
    existing = 0
    largest = (0, "")
    for item in list(paths or []):
        try:
            path = Path(item)
            if path.exists() and path.is_file():
                size = int(path.stat().st_size)
                total += size
                existing += 1
                if size > largest[0]:
                    largest = (size, path.name)
        except Exception:
            continue
    if not paths:
        return "no paths"
    if existing <= 0:
        return f"{len(paths)} path(s), file sizes unavailable"
    return f"{existing}/{len(paths)} file(s), total={_fmt_bytes(total)}, largest={largest[1]} {_fmt_bytes(largest[0])}"



def _ltx_component_name_from_configurator(configurator: Any) -> str | None:
    """Return a small LTX component name from an official configurator string.

    Keep this separate from _classify_ltx_model_builder because file names such
    as ltx-2.3-22b_vocoder.safetensors contain "ltx-2" and would otherwise be
    misclassified as the main transformer.
    """
    try:
        text = str(configurator or "")
        name = getattr(configurator, "__name__", "") or ""
        text = f"{name} {text}".lower()
    except Exception:
        text = ""
    if "vocoderconfigurator" in text or "vocoder" in text:
        return "Vocoder"
    if "audiodecoderconfigurator" in text or "audiodecoder" in text:
        return "AudioDecoder"
    if "audioencoderconfigurator" in text or "audioencoder" in text:
        return "AudioEncoder"
    if "videodecoderconfigurator" in text or "videodecoder" in text:
        return "VideoDecoder"
    if "videoencoderconfigurator" in text or "videoencoder" in text:
        return "VideoEncoder"
    return None


def _ltx_find_component_file_override(ctx: Dict[str, Any], component: str) -> Path | None:
    """Find FrameVision's optional small split file for final component loads.

    Keep this intentionally narrow. Vocoder was already routed through the split
    file in this base; this patch only adds VideoDecoder and AudioDecoder.
    VideoEncoder, text encoder, EmbeddingsProcessor, main transformer, and LoRA
    paths are deliberately not redirected here.
    """
    component = str(component or "").strip()
    try:
        root = Path(str(ctx.get("ltx_root") or APP_ROOT))
    except Exception:
        root = APP_ROOT
    base = root / "models" / "ltx23"

    if component == "Vocoder":
        names = [
            # Current FrameVision installer/default split name.
            "vocoder.safetensors",
            # Older/manual extracted names kept for backwards compatibility.
            "ltx2322bvocoder.safetensors",
            "ltx-2.3-22b_vocoder.safetensors",
            "ltx_2_3_22b_vocoder.safetensors",
        ]
        subdirs = ["split", "", "vocoder", "support/vocoder", "audio_vae", "support/audio_vae"]
    elif component == "AudioDecoder":
        names = [
            "audio_vae.safetensors",
            "ltx2322baudiovae.safetensors",
            "ltx-2.3-22b_audio_vae.safetensors",
            "ltx_2_3_22b_audio_vae.safetensors",
        ]
        subdirs = ["split", "audio_vae", "support/audio_vae", ""]
    elif component == "VideoDecoder":
        names = [
            "vae.safetensors",
            "ltx2322bvae.safetensors",
            "ltx-2.3-22b_vae.safetensors",
            "ltx_2_3_22b_vae.safetensors",
        ]
        subdirs = ["split", "vae", "video_vae", "support/vae", "support/video_vae", ""]
    else:
        return None

    for sub in subdirs:
        folder = base / sub if sub else base
        for name in names:
            path = folder / name
            try:
                if path.exists() and path.is_file():
                    return path
            except Exception:
                continue
    return None


def _ltx_component_name_from_model_instance(model: Any) -> str | None:
    """Classify the actual meta model instance for final-load redirection.

    This stays stricter than configurator/file-name matching so we only redirect
    the exact late components that have matching split files.
    """
    try:
        name = f"{type(model).__module__}.{type(model).__name__}"
    except Exception:
        name = ""
    low = name.lower()
    if "audio_vae.vocoder" in low and "vocoderwithbwe" in low:
        return "Vocoder"
    if "audio_vae.audio_vae" in low and "audiodecoder" in low:
        return "AudioDecoder"
    if "video_vae.video_vae" in low and "videodecoder" in low:
        return "VideoDecoder"
    return None


def _ltx_apply_component_file_override(builder: Any, ctx: Dict[str, Any], torch_module: Any | None, echo: bool) -> str | None:
    """Record vocoder override availability at builder level.

    The official SingleGPUModelBuilder is frozen in this environment, so the
    actual safe redirect happens later at _load_model_weights, where the
    model_path argument can be replaced. This function only logs availability
    and never redirects video/audio VAE components.
    """
    try:
        cfg = getattr(builder, "model_class_configurator", None)
        component = _ltx_component_name_from_configurator(cfg)
    except Exception:
        component = None
    if component != "Vocoder":
        return None

    override = _ltx_find_component_file_override(ctx, component)
    original_paths = _normalise_model_paths(getattr(builder, "model_path", None))
    original_summary = _file_size_summary(original_paths)
    ctx.setdefault("ltx_component_file_events", [])
    if override is None:
        status = f"Vocoder: no small override file found; using original path(s): {original_summary}"
        ctx["ltx_component_file_vocoder_status"] = "fallback_original"
        ctx["ltx_component_file_vocoder_original"] = original_summary
        try:
            events = ctx.setdefault("ltx_component_file_events", [])
            if isinstance(events, list) and len(events) < 100:
                events.append(status)
        except Exception:
            pass
        if echo:
            try:
                print(f"[ltx-files] WARNING: {status}", flush=True)
            except Exception:
                pass
        return status

    status = f"Vocoder: small component file ready for final-load redirect {override} ({_file_size_summary([str(override)])}); original builder path(s): {original_summary}"
    ctx["ltx_component_file_vocoder_status"] = "pending_final_load_redirect"
    ctx["ltx_component_file_vocoder_path"] = str(override)
    ctx["ltx_component_file_vocoder_original"] = original_summary
    try:
        events = ctx.setdefault("ltx_component_file_events", [])
        if isinstance(events, list) and len(events) < 100:
            events.append(status)
    except Exception:
        pass
    if echo:
        try:
            print(f"[ltx-files] {status}", flush=True)
        except Exception:
            pass
    try:
        _record_ltx_main_load_trace(ctx, "FrameVisionVocoderFileReady", status, torch_module, echo)
    except Exception:
        pass
    return status

def _classify_ltx_model_builder(builder: Any) -> str:
    try:
        cfg = getattr(builder, "model_class_configurator", None)
        cfg_name = getattr(cfg, "__name__", str(cfg or ""))
    except Exception:
        cfg_name = ""
    try:
        paths = " | ".join(_normalise_model_paths(getattr(builder, "model_path", None))).lower()
    except Exception:
        paths = ""
    text = f"{cfg_name} {paths}".lower()
    # Prefer configurator/component names before generic file-name matches like
    # "ltx-2" so small split files such as ltx-2.3-22b_vocoder.safetensors
    # are not misclassified as the main transformer.
    if "videoencoder" in text:
        return "video_encoder"
    if "videodecoder" in text:
        return "video_decoder"
    if "audiodecoder" in text:
        return "audio_decoder"
    if "audioencoder" in text:
        return "audio_encoder"
    if "vocoder" in text:
        return "audio_vocoder"
    if "gemmatextencoder" in text or "gemma" in text:
        return "gemma_text_encoder"
    if "embeddingsprocessor" in text:
        return "gemma_embeddings_processor"
    if "ltxmodelconfigurator" in text or "ltx-2" in text or "distilled" in text:
        return "main_transformer"
    return cfg_name or "unknown"


def _record_ltx_main_load_trace(
    ctx: Dict[str, Any],
    label: str,
    detail: str,
    torch_module: Any | None = None,
    echo: bool = True,
) -> None:
    try:
        t0 = float(ctx.setdefault("_ltx_main_load_trace_t0", time.perf_counter()))
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = 0.0
    try:
        snapshot = _full_memory_snapshot(torch_module)
    except Exception:
        snapshot = "n/a"
    line = f"{elapsed:9.3f}s | {label}: {detail} | {snapshot}"
    try:
        events = ctx.setdefault("ltx_main_load_trace_events", [])
        if isinstance(events, list) and len(events) < 600:
            events.append(line)
        ctx["ltx_main_load_trace_event_count"] = str(len(events) if isinstance(events, list) else 0)
    except Exception:
        pass
    if echo:
        try:
            print(f"[vram-lab-ltx-load] {line}", flush=True)
        except Exception:
            pass




def _install_ltx_main_cpu_first_build_gate(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Build the main LTX transformer CPU-first, then let VRAM Lab own blocks.

    This is a focused alternative to LTX's official StreamingModelBuilder path.
    It avoids the traced full ``LTXModel load_state_dict(device=cuda:0)`` spike
    without switching the whole run to the very slow official block-streaming
    backend. The wrapper keeps the normal DiffusionStage flow, but for the main
    transformer only it asks the existing SingleGPUModelBuilder to materialize
    the LTXModel on CPU, moves the small non-block parts to CUDA, and leaves
    ``velocity_model.transformer_blocks`` on CPU so the existing BatchSplitAdapter
    early hook can load the hot blocks during denoise.
    """
    ctx["ltx_main_cpu_first_gate_installed"] = "NO"
    ctx["ltx_main_cpu_first_gate_calls"] = "0"
    ctx["ltx_main_cpu_first_gate_built"] = "0"
    ctx["ltx_main_cpu_first_gate_non_block_cuda_moves"] = "0"
    ctx["ltx_main_cpu_first_gate_left_cpu_blocks"] = "0"
    ctx["ltx_main_cpu_first_gate_status"] = "not attempted"
    ctx["ltx_main_cpu_first_gate_errors"] = "none"

    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        blocks_mod = importlib.import_module("ltx_pipelines.utils.blocks")
        ds_cls = getattr(blocks_mod, "DiffusionStage")
        X0Model = getattr(blocks_mod, "X0Model")
        modify_sd_ops_for_compilation = getattr(blocks_mod, "modify_sd_ops_for_compilation")
        _chain_quantization = getattr(blocks_mod, "_chain_quantization")
        LoraPathStrengthAndSDOps = getattr(blocks_mod, "LoraPathStrengthAndSDOps")
    except Exception as exc:
        ctx["ltx_main_cpu_first_gate_errors"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    try:
        original_bt = getattr(ds_cls, "_build_transformer")
        if getattr(original_bt, "_framevision_ltx_main_cpu_first_gate", False):
            ctx["ltx_main_cpu_first_gate_installed"] = "YES: already installed"
            return
    except Exception as exc:
        ctx["ltx_main_cpu_first_gate_errors"] = f"DiffusionStage._build_transformer unavailable: {type(exc).__name__}: {exc}"
        return

    def _move_direct_params_buffers(module: Any, device: Any) -> int:
        moved = 0
        try:
            import torch as _torch  # type: ignore
            with _torch.no_grad():
                for name, param in list(module.named_parameters(recurse=False)):
                    try:
                        if getattr(param, "is_meta", False):
                            continue
                        if str(getattr(param, "device", "")) == str(device):
                            continue
                        new_param = _torch.nn.Parameter(param.to(device=device, non_blocking=True), requires_grad=param.requires_grad)
                        setattr(module, name, new_param)
                        moved += 1
                    except Exception:
                        # Keep the gate best-effort; a later forward will surface
                        # a real device mismatch if this was required.
                        pass
                for name, buf in list(module.named_buffers(recurse=False)):
                    try:
                        if buf is None or getattr(buf, "is_meta", False):
                            continue
                        if str(getattr(buf, "device", "")) == str(device):
                            continue
                        module._buffers[name] = buf.to(device=device, non_blocking=True)
                        moved += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return moved

    def _move_ltx_non_block_parts_to_device(x0_model: Any, target: Any) -> tuple[int, int]:
        """Move everything except velocity_model.transformer_blocks to target."""
        moved = 0
        left_blocks = 0
        velocity = getattr(x0_model, "velocity_model", None)
        if velocity is None:
            return moved, left_blocks
        try:
            blocks = getattr(velocity, "transformer_blocks", None)
            if blocks is not None:
                try:
                    left_blocks = len(blocks)
                except Exception:
                    left_blocks = 1
        except Exception:
            left_blocks = 0
        try:
            moved += _move_direct_params_buffers(velocity, target)
            for child_name, child in list(velocity.named_children()):
                if child_name == "transformer_blocks":
                    continue
                try:
                    child.to(target)
                    moved += 1
                except Exception:
                    pass
        except Exception:
            pass
        try:
            moved += _move_direct_params_buffers(x0_model, target)
        except Exception:
            pass
        return moved, left_blocks

    def guarded_build_transformer(self: Any, *args: Any, **kwargs: Any) -> Any:
        builder = getattr(self, "_transformer_builder", None)
        kind = _classify_ltx_model_builder(builder)
        target = kwargs.get("device", None)
        if target is None:
            target = getattr(self, "_device", None)
        try:
            target = torch.device(target or "cuda")
        except Exception:
            target = torch.device("cuda")

        # Only touch the exact main LTXModel path. Anything exotic falls back to
        # official LTX behavior so this patch remains narrow.
        if kind != "main_transformer" or str(target).startswith("cpu"):
            return original_bt(self, *args, **kwargs)

        ctx["ltx_main_cpu_first_gate_calls"] = str(int(str(ctx.get("ltx_main_cpu_first_gate_calls", "0") or "0")) + 1)
        if echo:
            try:
                print(f"[vram-lab-ltx-main-cpu] CPU-first main transformer build starting; target={target} | {_cuda_snapshot(torch)}", flush=True)
            except Exception:
                pass

        try:
            # Match official DiffusionStage._build_transformer logic, but build
            # the LTXModel on CPU first and do not call X0Model(...).to(cuda).
            sd_ops = getattr(builder, "model_sd_ops", None)
            module_ops = getattr(builder, "module_ops", ())
            loras = getattr(builder, "loras", ())

            if bool(getattr(self, "_torch_compile", False)):
                # Keep compile path official. It rewrites block names/SD ops and
                # is not part of the current LTX 2.3 VRAM Lab test path.
                ctx["ltx_main_cpu_first_gate_status"] = "skipped torch_compile path; used official builder"
                return original_bt(self, *args, **kwargs)

            quantization = getattr(self, "_quantization", None)
            if quantization is not None:
                sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, quantization)

            builder2 = builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)

            cpu_kwargs = dict(kwargs)
            cpu_kwargs.pop("device", None)
            cpu_model = builder2.build(device=torch.device("cpu"), **cpu_kwargs)
            x0_model = X0Model(cpu_model)
            moved, left_blocks = _move_ltx_non_block_parts_to_device(x0_model, target)
            try:
                x0_model.eval()
            except Exception:
                pass

            ctx["ltx_main_cpu_first_gate_built"] = str(int(str(ctx.get("ltx_main_cpu_first_gate_built", "0") or "0")) + 1)
            ctx["ltx_main_cpu_first_gate_non_block_cuda_moves"] = str(moved)
            ctx["ltx_main_cpu_first_gate_left_cpu_blocks"] = str(left_blocks)
            ctx["ltx_main_cpu_first_gate_status"] = (
                f"main LTXModel loaded on CPU first; moved {moved} non-block module/parameter groups to {target}; "
                f"left {left_blocks} transformer blocks for BatchSplitAdapter/VRAM Lab residency"
            )
            if echo:
                try:
                    print(f"[vram-lab-ltx-main-cpu] CPU-first main transformer build ready; blocks_left_cpu={left_blocks}; moved={moved} | {_cuda_snapshot(torch)}", flush=True)
                except Exception:
                    pass
            return x0_model
        except Exception as exc:
            ctx["ltx_main_cpu_first_gate_errors"] = f"{type(exc).__name__}: {exc}"
            ctx["ltx_main_cpu_first_gate_status"] = "exception; falling back is disabled for safety in this run"
            if echo:
                try:
                    print(f"[vram-lab-ltx-main-cpu] ERROR: {type(exc).__name__}: {exc}", flush=True)
                except Exception:
                    pass
            raise

    try:
        guarded_build_transformer.__name__ = getattr(original_bt, "__name__", "_build_transformer")
        guarded_build_transformer.__doc__ = getattr(original_bt, "__doc__", None)
        setattr(guarded_build_transformer, "_framevision_ltx_main_cpu_first_gate", True)
        setattr(guarded_build_transformer, "_framevision_ltx_main_cpu_first_original", original_bt)
        setattr(ds_cls, "_build_transformer", guarded_build_transformer)
        ctx["ltx_main_cpu_first_gate_installed"] = "YES"
        ctx["ltx_main_cpu_first_gate_status"] = "installed; waiting for main transformer build"
        ctx.setdefault("notes", []).append(
            "Installed LTX main CPU-first transformer gate: main LTXModel loads on CPU first, non-block parts move to CUDA, transformer_blocks are left for BatchSplitAdapter/VRAM Lab."
        )
    except Exception as exc:
        ctx["ltx_main_cpu_first_gate_errors"] = f"install failed: {type(exc).__name__}: {exc}"


def _install_ltx_main_direct_load_state_gate(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Force the traced main LTXModel safetensor load to CPU and block full CUDA .to().

    The previous DiffusionStage-level CPU-first gate was too indirect: official
    LTX could still reach ``load_state_dict(..., device=cuda:0)`` for the main
    42.98GB LTXModel. This gate patches the exact loader boundary that the trace
    found, plus only the LTXModel/X0Model class-specific ``.to(cuda)`` calls that
    would otherwise materialize the whole transformer after CPU load.

    It is intentionally not a global Tensor.to / Module.to monkeypatch.
    """
    ctx["ltx_main_direct_load_gate_installed"] = "NO"
    ctx["ltx_main_direct_load_gate_targets"] = "none"
    ctx["ltx_main_direct_load_gate_calls"] = "0"
    ctx["ltx_main_direct_load_gate_forced_cpu_loads"] = "0"
    ctx["ltx_main_direct_load_gate_skipped_non_main_loads"] = "0"
    ctx["ltx_main_direct_load_gate_active_weight_model"] = "none"
    ctx["ltx_main_direct_load_gate_checkpoint_independent"] = "YES: main LTXModel CPU gate uses model class/context, not checkpoint folder/name"
    ctx["ltx_main_direct_load_gate_last_reason"] = "none"
    ctx["ltx_main_direct_load_gate_x0_to_intercepts"] = "0"
    ctx["ltx_main_direct_load_gate_ltx_to_intercepts"] = "0"
    ctx["ltx_main_direct_load_gate_left_cpu_blocks"] = "0"
    ctx["ltx_main_direct_load_gate_non_block_cuda_moves"] = "0"
    ctx["ltx_main_direct_load_gate_status"] = "not attempted"
    ctx["ltx_main_direct_load_gate_errors"] = "none"

    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        sgmb = importlib.import_module("ltx_core.loader.single_gpu_model_builder")
        blocks_mod = importlib.import_module("ltx_pipelines.utils.blocks")
        model_mod = importlib.import_module("ltx_core.model.transformer.model")
        X0Model = getattr(blocks_mod, "X0Model")
        LTXModel = getattr(model_mod, "LTXModel")
    except Exception as exc:
        ctx["ltx_main_direct_load_gate_errors"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    active_weight_model_stack: list[str] = []

    def _model_path_total_bytes(model_path: Any) -> int:
        total = 0
        try:
            for item in _normalise_model_paths(model_path):
                try:
                    path = Path(str(item))
                    if path.exists() and path.is_file():
                        total += int(path.stat().st_size)
                except Exception:
                    continue
        except Exception:
            pass
        return int(total)

    def _ddr_cpu_load_preflight(component_name: str, model_path: Any) -> tuple[bool, str]:
        """Protect Windows commit/pagefile before using CPU as backup VRAM.

        CPU load_state_dict for the 42.98GB LTX safetensor can mmap/commit a huge
        virtual range. On 32/64GB systems this may push Windows into code 1455 or,
        worse, MEMORY_MANAGEMENT. When the projected VMS/commit is unsafe, skip the
        CPU guard and leave the official CUDA load route alone instead of forcing a
        dangerous CPU mapping.
        """
        try:
            import psutil  # type: ignore
            proc = psutil.Process(os.getpid())
            mi = proc.memory_info()
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            rss = int(getattr(mi, "rss", 0) or 0)
            vms = int(getattr(mi, "vms", 0) or 0)
            ram_total = int(getattr(vm, "total", 0) or 0)
            ram_available = int(getattr(vm, "available", 0) or 0)
            ram_used = int(getattr(vm, "used", 0) or 0)
            swap_total = int(getattr(sm, "total", 0) or 0)
            swap_used = int(getattr(sm, "used", 0) or 0)
            model_bytes = max(_model_path_total_bytes(model_path), 0)
            gb = 1024 ** 3

            if ram_total <= 40 * gb:
                profile = "32GB"
                ram_reserve = 8 * gb
                commit_fraction = 0.72
            elif ram_total <= 96 * gb:
                profile = "64GB"
                ram_reserve = 12 * gb
                commit_fraction = 0.82
            else:
                profile = "128GB+"
                ram_reserve = 16 * gb
                commit_fraction = 0.88

            # psutil.swap_memory().total maps to the Windows paging-file/commit pool in this environment.
            commit_total = swap_total if swap_total > 0 else max(ram_total * 2, vms + model_bytes)
            projected_vms = vms + model_bytes
            commit_limit = int(float(commit_total) * commit_fraction)
            commit_available = max(commit_total - swap_used, 0)

            reasons = [
                f"profile={profile}",
                f"component={component_name}",
                f"model={_fmt_bytes(model_bytes)}",
                f"rss={_fmt_bytes(rss)}",
                f"vms={_fmt_bytes(vms)}",
                f"projected_vms={_fmt_bytes(projected_vms)}",
                f"commit_limit={_fmt_bytes(commit_limit)}",
                f"commit_available={_fmt_bytes(commit_available)}",
                f"ram_available={_fmt_bytes(ram_available)}",
                f"ram_reserve={_fmt_bytes(ram_reserve)}",
                f"ram_used={_fmt_bytes(ram_used)}/{_fmt_bytes(ram_total)}",
                f"pagefile_used={_fmt_bytes(swap_used)}/{_fmt_bytes(commit_total)}",
            ]

            if component_name == "AudioDecoder":
                # AudioDecoder is a small component slice inside the shared 42.98 GB
                # LTX safetensor. Real crash logs showed the direct CUDA route can
                # be unstable here, while the CPU state-dict route survived. Keep
                # this exception for AudioDecoder only.
                #
                # Do NOT apply this to VocoderWithBWE. A later crash log showed the
                # Vocoder CPU state-dict route can hard-crash immediately after
                # load_state_dict:start, even when the same preflight passes.
                audio_min_available = max(4 * gb, int(float(ram_total) * 0.10))
                if ram_available >= audio_min_available and commit_available >= audio_min_available:
                    return True, "AudioDecoder CPU load allowed for small component slice; ignoring full-file projected VMS for safetensor mmap; " + "; ".join(reasons)

            if ram_available < ram_reserve:
                return False, "RAM reserve would be crossed; " + "; ".join(reasons)

            # VAE Encoder/Decoder weights are small slices inside the large 42.98GB
            # shared LTX safetensor. The previous guard treated the whole file as
            # committed RAM/VMS and then fell back to direct CUDA loading. Real logs
            # showed that fallback is the path that often dies with 3221225477.
            # Keep the DDR reserve check, but do not block VAE CPU loading only
            # because the backing safetensor mmap makes projected VMS look huge.
            if component_name in ("VideoEncoder", "VideoDecoder") and commit_available >= ram_reserve:
                return True, "VAE CPU load allowed by physical RAM reserve; ignoring full-file projected VMS for safetensor mmap; " + "; ".join(reasons)

            if model_bytes > 0 and projected_vms > commit_limit:
                return False, "projected VMS/commit too high for CPU safetensor load; " + "; ".join(reasons)
            if model_bytes > 0 and commit_available < (model_bytes + ram_reserve):
                return False, "not enough commit headroom for CPU safetensor load; " + "; ".join(reasons)
            return True, "DDR/commit preflight passed; " + "; ".join(reasons)
        except Exception as exc:
            return True, f"DDR/commit preflight unavailable ({type(exc).__name__}: {exc}); allowing existing CPU guard"

    def _summarise_model_paths(model_path: Any) -> str:
        try:
            return _file_size_summary(_normalise_model_paths(model_path))
        except Exception:
            try:
                return str(model_path)
            except Exception:
                return "unknown"

    def _active_weight_model_name() -> str:
        try:
            if active_weight_model_stack:
                return str(active_weight_model_stack[-1])
        except Exception:
            pass
        return str(ctx.get("ltx_main_direct_load_gate_active_weight_model", "") or "")

    def _active_weight_model_is_main_ltx() -> bool:
        name = _active_weight_model_name()
        return (
            name == "ltx_core.model.transformer.model.LTXModel"
            or name.endswith(".LTXModel")
        )

    def _is_cuda_device(device: Any) -> bool:
        try:
            d = torch.device(device)
            return str(d).startswith("cuda")
        except Exception:
            return str(device).lower().startswith("cuda")

    def _extract_to_device(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any | None:
        if "device" in kwargs and kwargs.get("device") is not None:
            return kwargs.get("device")
        if args:
            first = args[0]
            # nn.Module.to can also receive dtype as first positional arg. Only
            # treat device-like values as devices.
            try:
                if isinstance(first, (str, torch.device)):
                    return first
            except Exception:
                pass
            try:
                if hasattr(first, "type") and str(getattr(first, "type", "")) in ("cpu", "cuda", "meta"):
                    return first
            except Exception:
                pass
        return None

    def _normalise_to_device(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        dev = _extract_to_device(args, kwargs)
        try:
            return torch.device(dev or "cuda")
        except Exception:
            return torch.device("cuda")

    def _path_is_main_transformer(model_path: Any) -> bool:
        try:
            paths = " | ".join(_normalise_model_paths(model_path)).lower()
        except Exception:
            paths = str(model_path).lower()
        if not paths:
            return False
        if "gemma" in paths or "text_encoder" in paths:
            return False
        return (
            "ltx-2" in paths
            or "22b" in paths
            or "distilled" in paths
            or "transformer" in paths and paths.endswith(".safetensors")
        )

    def _looks_like_main_ltx_model(module: Any) -> bool:
        try:
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is not None:
                try:
                    return len(blocks) >= 8
                except Exception:
                    return True
        except Exception:
            pass
        try:
            velocity = getattr(module, "velocity_model", None)
            blocks = getattr(velocity, "transformer_blocks", None)
            if blocks is not None:
                try:
                    return len(blocks) >= 8
                except Exception:
                    return True
        except Exception:
            pass
        return False

    def _move_direct_params_buffers(module: Any, device: Any) -> int:
        moved = 0
        try:
            with torch.no_grad():
                for name, param in list(module.named_parameters(recurse=False)):
                    try:
                        if param is None or getattr(param, "is_meta", False):
                            continue
                        if str(getattr(param, "device", "")) == str(device):
                            continue
                        setattr(module, name, torch.nn.Parameter(param.to(device=device, non_blocking=True), requires_grad=param.requires_grad))
                        moved += 1
                    except Exception:
                        pass
                for name, buf in list(module.named_buffers(recurse=False)):
                    try:
                        if buf is None or getattr(buf, "is_meta", False):
                            continue
                        if str(getattr(buf, "device", "")) == str(device):
                            continue
                        module._buffers[name] = buf.to(device=device, non_blocking=True)
                        moved += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return moved

    def _move_ltx_model_non_blocks(model: Any, device: Any) -> tuple[int, int]:
        moved = 0
        left_blocks = 0
        try:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                try:
                    left_blocks = len(blocks)
                except Exception:
                    left_blocks = 1
        except Exception:
            pass
        try:
            moved += _move_direct_params_buffers(model, device)
            for child_name, child in list(model.named_children()):
                if child_name == "transformer_blocks":
                    continue
                try:
                    child.to(device)
                    moved += 1
                except Exception:
                    pass
        except Exception:
            pass
        return moved, left_blocks

    def _move_x0_non_blocks(x0_model: Any, device: Any) -> tuple[int, int]:
        moved = 0
        left_blocks = 0
        try:
            moved += _move_direct_params_buffers(x0_model, device)
        except Exception:
            pass
        velocity = getattr(x0_model, "velocity_model", None)
        if velocity is not None:
            m, b = _move_ltx_model_non_blocks(velocity, device)
            moved += m
            left_blocks = max(left_blocks, b)
        return moved, left_blocks

    installed: list[str] = []
    errors: list[str] = []

    try:
        original_lmw_for_gate = getattr(sgmb, "_load_model_weights")
        if not getattr(original_lmw_for_gate, "_framevision_ltx_main_direct_model_context", False):
            def guarded_load_model_weights_for_context(*args: Any, **kwargs: Any) -> Any:
                meta_model = kwargs.get("meta_model", args[0] if len(args) > 0 else None)
                model_name = f"{type(meta_model).__module__}.{type(meta_model).__name__}" if meta_model is not None else "unknown"
                active_weight_model_stack.append(model_name)
                ctx["ltx_main_direct_load_gate_active_weight_model"] = model_name
                try:
                    return original_lmw_for_gate(*args, **kwargs)
                finally:
                    try:
                        active_weight_model_stack.pop()
                    except Exception:
                        pass
                    ctx["ltx_main_direct_load_gate_active_weight_model"] = str(active_weight_model_stack[-1]) if active_weight_model_stack else "none"

            guarded_load_model_weights_for_context.__name__ = getattr(original_lmw_for_gate, "__name__", "_load_model_weights")
            guarded_load_model_weights_for_context.__doc__ = getattr(original_lmw_for_gate, "__doc__", None)
            setattr(guarded_load_model_weights_for_context, "_framevision_ltx_main_direct_model_context", True)
            setattr(guarded_load_model_weights_for_context, "_framevision_ltx_main_direct_model_context_original", original_lmw_for_gate)
            setattr(sgmb, "_load_model_weights", guarded_load_model_weights_for_context)
            installed.append("single_gpu_model_builder._load_model_weights context")
        else:
            installed.append("single_gpu_model_builder._load_model_weights context already wrapped")
    except Exception as exc:
        errors.append(f"_load_model_weights context: {type(exc).__name__}: {exc}")

    try:
        original_load_state_dict = getattr(sgmb, "load_state_dict")
        if not getattr(original_load_state_dict, "_framevision_ltx_main_direct_load_gate", False):
            def guarded_load_state_dict(model_path: Any, loader: Any, registry: Any, device: Any, model_sd_ops: Any = None) -> Any:
                is_main_path = _path_is_main_transformer(model_path)
                is_cuda = _is_cuda_device(device)
                active_model = _active_weight_model_name()
                if is_cuda and _active_weight_model_is_main_ltx():
                    # Checkpoint-location-independent safety gate.  User models such as
                    # Sulphur/Eros/dev safetensors must get the same CPU-first path even
                    # when they live outside models/ltx23/distilled-1.1/.  Do not depend
                    # on file name/folder here; the active builder/model class tells us
                    # this is the main transformer.
                    reason = "model class/context match; checkpoint path independent"
                    ctx["ltx_main_direct_load_gate_calls"] = str(int(str(ctx.get("ltx_main_direct_load_gate_calls", "0") or "0")) + 1)
                    ctx["ltx_main_direct_load_gate_forced_cpu_loads"] = str(int(str(ctx.get("ltx_main_direct_load_gate_forced_cpu_loads", "0") or "0")) + 1)
                    ctx["ltx_main_direct_load_gate_checkpoint_independent"] = "YES"
                    ctx["ltx_main_direct_load_gate_last_reason"] = reason
                    ctx["ltx_main_direct_load_gate_status"] = f"forcing main LTXModel load_state_dict device from CUDA to CPU; active_model={active_model}; reason={reason}"
                    ctx["ltx_main_direct_load_gate_model_file_summary"] = _summarise_model_paths(model_path)
                    _ltx_quiet_status(ctx, "Loading main/distilled safetensor")
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-main-direct] load_state_dict main LTXModel: {device} -> cpu | active_model={active_model} | reason={reason} | {_summarise_model_paths(model_path)} | {_cuda_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    _record_ltx_main_load_trace(ctx, "FrameVisionMainDirectLoadGate:load_state_dict", f"forced device {device} -> cpu; active_model={active_model}; reason={reason}; {_summarise_model_paths(model_path)}", torch, echo)
                    return original_load_state_dict(model_path, loader, registry, torch.device("cpu"), model_sd_ops)

                # Stability guard for VAE component reloads around the Stage-1/Stage-2/finalize path.
                # Real crash logs showed direct CUDA state-dict loads for VideoEncoder and later
                # VideoDecoder hard-crashing the process with Windows code 3221225477 before Python
                # could write a normal failure report. Load these VAE state dicts on CPU, then let
                # the official builder move the finished module to CUDA normally.
                active_is_video_encoder = (
                    active_model == "ltx_core.model.video_vae.video_vae.VideoEncoder"
                    or active_model.endswith(".VideoEncoder")
                    or "video_vae.video_vae.VideoEncoder" in active_model
                )
                active_is_video_decoder = (
                    active_model == "ltx_core.model.video_vae.video_vae.VideoDecoder"
                    or active_model.endswith(".VideoDecoder")
                    or "video_vae.video_vae.VideoDecoder" in active_model
                )
                active_is_audio_decoder = (
                    active_model == "ltx_core.model.audio_vae.audio_vae.AudioDecoder"
                    or active_model.endswith(".AudioDecoder")
                    or "audio_vae.audio_vae.AudioDecoder" in active_model
                )
                active_is_vocoder = (
                    active_model == "ltx_core.model.audio_vae.vocoder.VocoderWithBWE"
                    or active_model.endswith(".VocoderWithBWE")
                    or "audio_vae.vocoder.VocoderWithBWE" in active_model
                )
                stage1_done = str(ctx.get("_ltx_low_profile_after_stage1_ready", "NO")).upper().startswith("YES")
                safe_audio_load = bool(ctx.get("ltx_safe_audio_load_requested", "NO") == "YES")
                if is_main_path and is_cuda and stage1_done and safe_audio_load and active_is_vocoder:
                    # Important: do not CPU-route VocoderWithBWE. Crash logs showed
                    # the Vocoder CPU load_state_dict path can die immediately with
                    # Windows code 3221225477. Keep sound enabled, but leave this
                    # component on the original CUDA route.
                    component_name = "Vocoder"
                    ctx["ltx_safe_audio_load_status"] = "BYPASS: Vocoder CPU route disabled; keeping CUDA route"
                    ctx["ltx_safe_audio_load_last_component"] = component_name
                    ctx["ltx_safe_audio_vocoder_cuda_bypass_calls"] = str(int(str(ctx.get("ltx_safe_audio_vocoder_cuda_bypass_calls", "0") or "0")) + 1)
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-audio-safe] Vocoder CPU route disabled after crash evidence; keeping original device {device} | active_model={active_model} | {_summarise_model_paths(model_path)} | {_full_memory_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    _record_ltx_main_load_trace(ctx, "FrameVisionVocoderAudioSafe:cuda_bypass", f"Vocoder CPU route disabled; kept device {device}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                    return original_load_state_dict(model_path, loader, registry, device, model_sd_ops)

                if is_main_path and is_cuda and stage1_done and safe_audio_load and active_is_audio_decoder:
                    component_name = "AudioDecoder"
                    ctx["ltx_safe_audio_load_status"] = f"YES: {component_name} load_state_dict routed through CPU when DDR preflight allows"
                    ctx["ltx_safe_audio_load_last_component"] = component_name
                    ctx["ltx_safe_audio_load_calls"] = str(int(str(ctx.get("ltx_safe_audio_load_calls", "0") or "0")) + 1)
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-audio-safe] load_state_dict {component_name}: {device} -> cpu check | active_model={active_model} | {_summarise_model_paths(model_path)} | {_full_memory_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    try:
                        # Drop stale CUDA cache right before the risky late audio decoder load.
                        _cleanup_cuda(torch)
                    except Exception:
                        pass
                    ddr_ok, ddr_reason = _ddr_cpu_load_preflight(component_name, model_path)
                    ctx["ltx_safe_audio_load_ddr_reason"] = ddr_reason
                    if not ddr_ok:
                        ctx["ltx_safe_audio_load_status"] = f"SKIPPED: {component_name} CPU route blocked by DDR/commit guard"
                        if echo:
                            try:
                                print(f"[vram-lab-ltx-audio-safe] {component_name} CPU route blocked; keeping original device {device} | {ddr_reason} | {_full_memory_snapshot(torch)}", flush=True)
                            except Exception:
                                pass
                        _record_ltx_main_load_trace(ctx, f"FrameVision{component_name}AudioSafe:skip_cpu_load", f"CPU route blocked; kept device {device}; {ddr_reason}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                        return original_load_state_dict(model_path, loader, registry, device, model_sd_ops)
                    _record_ltx_main_load_trace(ctx, f"FrameVision{component_name}AudioSafe:load_state_dict", f"forced device {device} -> cpu after audio DDR preflight; {ddr_reason}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                    return original_load_state_dict(model_path, loader, registry, torch.device("cpu"), model_sd_ops)

                if is_main_path and is_cuda and stage1_done and (active_is_video_encoder or active_is_video_decoder):
                    component_name = "VideoDecoder" if active_is_video_decoder else "VideoEncoder"
                    if active_is_video_decoder:
                        ctx["ltx_video_decoder_cpu_load_guard"] = "YES: final VideoDecoder load_state_dict forced CUDA -> CPU"
                        ctx["ltx_video_decoder_cpu_load_guard_calls"] = str(int(str(ctx.get("ltx_video_decoder_cpu_load_guard_calls", "0") or "0")) + 1)
                        ctx["ltx_video_decoder_cpu_load_guard_active_model"] = active_model
                        ctx["ltx_video_decoder_cpu_load_guard_model_file_summary"] = _summarise_model_paths(model_path)
                    else:
                        ctx["ltx_video_encoder_cpu_load_guard"] = "YES: post-Stage-1 VideoEncoder load_state_dict forced CUDA -> CPU"
                        ctx["ltx_video_encoder_cpu_load_guard_calls"] = str(int(str(ctx.get("ltx_video_encoder_cpu_load_guard_calls", "0") or "0")) + 1)
                        ctx["ltx_video_encoder_cpu_load_guard_active_model"] = active_model
                        ctx["ltx_video_encoder_cpu_load_guard_model_file_summary"] = _summarise_model_paths(model_path)
                    ctx["ltx_vae_cpu_load_guard"] = f"YES: {component_name} load_state_dict forced CUDA -> CPU"
                    ctx["ltx_vae_cpu_load_guard_calls"] = str(int(str(ctx.get("ltx_vae_cpu_load_guard_calls", "0") or "0")) + 1)
                    ctx["ltx_vae_cpu_load_guard_last_component"] = component_name
                    ctx["ltx_main_direct_load_gate_status"] = f"forcing {component_name} load_state_dict device from CUDA to CPU; active_model={active_model}"
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-vae-cpu-load] load_state_dict {component_name}: {device} -> cpu | active_model={active_model} | {_summarise_model_paths(model_path)} | {_cuda_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    ddr_ok, ddr_reason = _ddr_cpu_load_preflight(component_name, model_path)
                    ctx["ltx_ddr_profile_guard"] = "SAFE_FOR_CPU_LOAD" if ddr_ok else "CPU_LOAD_BLOCKED"
                    ctx["ltx_ddr_profile_guard_last_component"] = component_name
                    ctx["ltx_ddr_profile_guard_reason"] = ddr_reason
                    ctx["ltx_vae_cpu_load_guard_cache"] = "DISABLED: CPU state-dict cache removed after Windows MEMORY_MANAGEMENT crash"
                    if not ddr_ok:
                        ctx["ltx_vae_cpu_load_guard"] = f"SKIPPED: {component_name} CPU load blocked by DDR/commit guard"
                        ctx["ltx_vae_cpu_load_guard_skipped_by_ddr"] = str(int(str(ctx.get("ltx_vae_cpu_load_guard_skipped_by_ddr", "0") or "0")) + 1)
                        if active_is_video_encoder:
                            ctx["ltx_video_encoder_cpu_load_guard_skipped_by_ddr"] = str(int(str(ctx.get("ltx_video_encoder_cpu_load_guard_skipped_by_ddr", "0") or "0")) + 1)
                        if active_is_video_decoder:
                            ctx["ltx_video_decoder_cpu_load_guard_skipped_by_ddr"] = str(int(str(ctx.get("ltx_video_decoder_cpu_load_guard_skipped_by_ddr", "0") or "0")) + 1)
                        if echo:
                            try:
                                print(f"[vram-lab-ltx-ddr-guard] {component_name} CPU load blocked; keeping original device {device} | {ddr_reason} | {_full_memory_snapshot(torch)}", flush=True)
                            except Exception:
                                pass
                        _record_ltx_main_load_trace(ctx, f"FrameVision{component_name}DDRGuard:skip_cpu_load", f"CPU load blocked; kept device {device}; {ddr_reason}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                        return original_load_state_dict(model_path, loader, registry, device, model_sd_ops)

                    _record_ltx_main_load_trace(ctx, f"FrameVision{component_name}CPULoadGuard:load_state_dict", f"forced device {device} -> cpu after DDR preflight; {ddr_reason}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                    return original_load_state_dict(model_path, loader, registry, torch.device("cpu"), model_sd_ops)

                if is_main_path and is_cuda:
                    ctx["ltx_main_direct_load_gate_skipped_non_main_loads"] = str(int(str(ctx.get("ltx_main_direct_load_gate_skipped_non_main_loads", "0") or "0")) + 1)
                    ctx["ltx_main_direct_load_gate_status"] = f"skipped non-LTXModel load_state_dict; active_model={active_model}"
                    _record_ltx_main_load_trace(ctx, "FrameVisionMainDirectLoadGate:load_state_dict_skip", f"kept device {device}; active_model={active_model}; {_summarise_model_paths(model_path)}", torch, echo)
                return original_load_state_dict(model_path, loader, registry, device, model_sd_ops)

            guarded_load_state_dict.__name__ = getattr(original_load_state_dict, "__name__", "load_state_dict")
            guarded_load_state_dict.__doc__ = getattr(original_load_state_dict, "__doc__", None)
            setattr(guarded_load_state_dict, "_framevision_ltx_main_direct_load_gate", True)
            setattr(guarded_load_state_dict, "_framevision_ltx_main_direct_load_original", original_load_state_dict)
            setattr(sgmb, "load_state_dict", guarded_load_state_dict)
            installed.append("single_gpu_model_builder.load_state_dict")
        else:
            installed.append("single_gpu_model_builder.load_state_dict already wrapped")
    except Exception as exc:
        errors.append(f"load_state_dict: {type(exc).__name__}: {exc}")

    try:
        original_ltx_to = getattr(LTXModel, "to")
        if not getattr(original_ltx_to, "_framevision_ltx_main_direct_load_gate", False):
            def guarded_ltx_to(self: Any, *args: Any, **kwargs: Any) -> Any:
                target = _normalise_to_device(args, kwargs)
                if _is_cuda_device(target) and _looks_like_main_ltx_model(self):
                    if int(str(ctx.get("ltx_main_direct_load_gate_forced_cpu_loads", "0") or "0")) <= 0:
                        ctx["ltx_main_direct_load_gate_status"] = "blocked LTXModel.to(cuda) before direct CPU load gate was hit"
                        raise RuntimeError("FrameVision main direct load gate blocked full LTXModel.to(cuda) before CPU load_state_dict gate was hit")
                    ctx["ltx_main_direct_load_gate_ltx_to_intercepts"] = str(int(str(ctx.get("ltx_main_direct_load_gate_ltx_to_intercepts", "0") or "0")) + 1)
                    moved, left_blocks = _move_ltx_model_non_blocks(self, target)
                    ctx["ltx_main_direct_load_gate_non_block_cuda_moves"] = str(int(str(ctx.get("ltx_main_direct_load_gate_non_block_cuda_moves", "0") or "0")) + moved)
                    ctx["ltx_main_direct_load_gate_left_cpu_blocks"] = str(left_blocks)
                    ctx["ltx_main_direct_load_gate_status"] = f"intercepted LTXModel.to({target}); moved non-blocks only; left {left_blocks} blocks on CPU"
                    _ltx_quiet_status(ctx, "Moving model parts to GPU")
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-main-direct] intercepted LTXModel.to({target}); left_blocks={left_blocks}; moved={moved} | {_cuda_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    return self
                return original_ltx_to(self, *args, **kwargs)

            guarded_ltx_to.__name__ = getattr(original_ltx_to, "__name__", "to")
            guarded_ltx_to.__doc__ = getattr(original_ltx_to, "__doc__", None)
            setattr(guarded_ltx_to, "_framevision_ltx_main_direct_load_gate", True)
            setattr(guarded_ltx_to, "_framevision_ltx_main_direct_load_original", original_ltx_to)
            setattr(LTXModel, "to", guarded_ltx_to)
            installed.append("LTXModel.to")
        else:
            installed.append("LTXModel.to already wrapped")
    except Exception as exc:
        errors.append(f"LTXModel.to: {type(exc).__name__}: {exc}")

    try:
        original_x0_to = getattr(X0Model, "to")
        if not getattr(original_x0_to, "_framevision_ltx_main_direct_load_gate", False):
            def guarded_x0_to(self: Any, *args: Any, **kwargs: Any) -> Any:
                target = _normalise_to_device(args, kwargs)
                if _is_cuda_device(target) and _looks_like_main_ltx_model(self):
                    if int(str(ctx.get("ltx_main_direct_load_gate_forced_cpu_loads", "0") or "0")) <= 0:
                        ctx["ltx_main_direct_load_gate_status"] = "blocked X0Model.to(cuda) before direct CPU load gate was hit"
                        raise RuntimeError("FrameVision main direct load gate blocked full X0Model.to(cuda) before CPU load_state_dict gate was hit")
                    ctx["ltx_main_direct_load_gate_x0_to_intercepts"] = str(int(str(ctx.get("ltx_main_direct_load_gate_x0_to_intercepts", "0") or "0")) + 1)
                    moved, left_blocks = _move_x0_non_blocks(self, target)
                    ctx["ltx_main_direct_load_gate_non_block_cuda_moves"] = str(int(str(ctx.get("ltx_main_direct_load_gate_non_block_cuda_moves", "0") or "0")) + moved)
                    ctx["ltx_main_direct_load_gate_left_cpu_blocks"] = str(left_blocks)
                    ctx["ltx_main_direct_load_gate_status"] = f"intercepted X0Model.to({target}); moved non-blocks only; left {left_blocks} blocks on CPU"
                    if echo:
                        try:
                            print(f"[vram-lab-ltx-main-direct] intercepted X0Model.to({target}); left_blocks={left_blocks}; moved={moved} | {_cuda_snapshot(torch)}", flush=True)
                        except Exception:
                            pass
                    return self
                return original_x0_to(self, *args, **kwargs)

            guarded_x0_to.__name__ = getattr(original_x0_to, "__name__", "to")
            guarded_x0_to.__doc__ = getattr(original_x0_to, "__doc__", None)
            setattr(guarded_x0_to, "_framevision_ltx_main_direct_load_gate", True)
            setattr(guarded_x0_to, "_framevision_ltx_main_direct_load_original", original_x0_to)
            setattr(X0Model, "to", guarded_x0_to)
            installed.append("X0Model.to")
        else:
            installed.append("X0Model.to already wrapped")
    except Exception as exc:
        errors.append(f"X0Model.to: {type(exc).__name__}: {exc}")

    ctx["ltx_main_direct_load_gate_installed"] = "YES" if installed else "NO"
    ctx["ltx_main_direct_load_gate_targets"] = " | ".join(installed) if installed else "none"
    ctx["ltx_main_direct_load_gate_errors"] = "none" if not errors else " | ".join(errors[-8:])
    if installed:
        ctx["ltx_main_direct_load_gate_status"] = "installed; waiting for main LTXModel load_state_dict(device=cuda)"
        ctx.setdefault("notes", []).append(
            "Installed LTX main direct load gate: main transformer load_state_dict(cuda) is forced to CPU, and class-specific LTXModel/X0Model.to(cuda) moves only non-block parts while leaving transformer_blocks for VRAM Lab."
        )


def _install_ltx_main_load_start_trace(
    ctx: Dict[str, Any],
    torch_module: Any | None = None,
    echo: bool = True,
    hooks_mod: Any | None = None,
) -> None:
    """Trace the exact official LTX start of large model loading.

    This is a narrow diagnostic patch for the 40GB distilled transformer path.
    It does not change residency, does not patch Tensor.to/Module.to, and does
    not change VRAM Lab denoise hooks. It marks the official owner functions
    before SingleGPUModelBuilder starts loading safetensors so we can identify
    the first safe place to replace full-load-then-park with VRAM-Lab-owned
    loading.
    """
    ctx["ltx_main_load_trace_installed"] = "NO"
    ctx["ltx_main_load_trace_targets"] = "none"
    ctx["ltx_main_load_trace_event_count"] = "0"
    ctx["ltx_main_load_trace_first_main_transformer_event"] = "none"
    ctx["ltx_main_load_trace_first_main_transformer_load_state_dict"] = "none"
    ctx["ltx_main_load_trace_main_transformer_file_summary"] = "none"
    ctx["ltx_main_load_trace_errors"] = "none"
    ctx.setdefault("ltx_main_load_trace_events", [])
    ctx["ltx_component_file_override"] = "enabled: vocoder-only final-load resolver"
    ctx.setdefault("ltx_component_file_events", [])

    installed: List[str] = []
    errors: List[str] = []

    def remember_first(key: str, line: str) -> None:
        try:
            if str(ctx.get(key, "none")) in ("", "none", "n/a"):
                ctx[key] = line
        except Exception:
            pass

    def _vae_component_name_from_configurator(cfg: Any) -> str | None:
        component = _ltx_component_name_from_configurator(cfg)
        if component in ("VideoEncoder", "VideoDecoder"):
            return component
        return None

    def _module_has_meta_tensors(module: Any) -> bool:
        try:
            for param in module.parameters(recurse=True):
                if bool(getattr(param, "is_meta", False)):
                    return True
            for buf in module.buffers(recurse=True):
                if bool(getattr(buf, "is_meta", False)):
                    return True
        except Exception:
            return True
        return False

    def _module_device_summary(module: Any) -> str:
        counts: Dict[str, int] = {}
        try:
            for param in module.parameters(recurse=True):
                key = str(getattr(param, "device", "unknown"))
                counts[key] = counts.get(key, 0) + 1
            for buf in module.buffers(recurse=True):
                key = str(getattr(buf, "device", "unknown"))
                counts[key] = counts.get(key, 0) + 1
        except Exception as exc:
            return f"device summary unavailable: {type(exc).__name__}: {exc}"
        return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no tensors"

    def _install_vae_keepalive_to(module: Any, component: str) -> None:
        try:
            current_to = getattr(module, "to")
            if getattr(current_to, "_framevision_ltx_vae_keepalive", False):
                return
            def guarded_vae_to(*args: Any, **kwargs: Any) -> Any:
                target = None
                try:
                    if "device" in kwargs and kwargs.get("device") is not None:
                        target = kwargs.get("device")
                    elif args:
                        target = args[0]
                    t = str(target).lower() if target is not None else ""
                except Exception:
                    t = ""
                if t.startswith("meta"):
                    ctx["ltx_vae_reuse_cache"] = f"KEEPALIVE: {component}.to(meta) parked on CPU instead"
                    ctx[f"ltx_{component.lower()}_reuse_cache"] = "KEEPALIVE_CPU"
                    try:
                        return current_to("cpu")
                    except Exception as exc:
                        ctx[f"ltx_{component.lower()}_reuse_cache_error"] = f"CPU park failed: {type(exc).__name__}: {exc}"
                        return module
                return current_to(*args, **kwargs)
            try:
                guarded_vae_to.__name__ = getattr(current_to, "__name__", "to")
                guarded_vae_to.__doc__ = getattr(current_to, "__doc__", None)
                setattr(guarded_vae_to, "_framevision_ltx_vae_keepalive", True)
            except Exception:
                pass
            setattr(module, "to", guarded_vae_to)
        except Exception as exc:
            ctx[f"ltx_{component.lower()}_reuse_cache_error"] = f"keepalive install failed: {type(exc).__name__}: {exc}"

    try:
        if not isinstance(ctx.get("_ltx_vae_component_cache"), dict):
            ctx["_ltx_vae_component_cache"] = {}
    except Exception:
        pass

    try:
        sgmb = importlib.import_module("ltx_core.loader.single_gpu_model_builder")
        helpers_mod = importlib.import_module("ltx_core.loader.helpers")
        blocks_mod = importlib.import_module("ltx_pipelines.utils.blocks")
    except Exception as exc:
        ctx["ltx_main_load_trace_errors"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    try:
        cls = getattr(sgmb, "SingleGPUModelBuilder")
        original_build = getattr(cls, "build")
        if not getattr(original_build, "_framevision_ltx_main_load_trace", False):
            def traced_build(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    cfg_obj = getattr(self, "model_class_configurator", None)
                    cfg = getattr(cfg_obj, "__name__", str(cfg_obj or ""))
                except Exception:
                    cfg_obj = None
                    cfg = "unknown"
                _ltx_apply_component_file_override(self, ctx, torch_module, echo)
                kind = _classify_ltx_model_builder(self)
                paths = _normalise_model_paths(getattr(self, "model_path", None))
                size_summary = _file_size_summary(paths)
                device = kwargs.get("device", None)
                if device is None and args:
                    device = args[0]
                dtype = kwargs.get("dtype", None)
                detail = f"kind={kind}; configurator={cfg}; device={device}; dtype={dtype}; {size_summary}"
                if str(cfg) == "VideoDecoderConfigurator" or "VideoDecoderConfigurator" in str(cfg):
                    _ltx_prepare_for_video_decoder_finalize_guard(ctx, hooks_mod, torch_module, detail, echo)
                if str(cfg) == "VideoEncoderConfigurator" or "VideoEncoderConfigurator" in str(cfg):
                    _ltx_prepare_for_pre_upscaler_video_encoder_guard(ctx, torch_module, detail, echo)
                _maybe_ltx_low_profile_transition_cleanup(ctx, torch_module, "after_stage1", detail, echo)
                _maybe_ltx_low_profile_transition_cleanup(ctx, torch_module, "after_stage2", detail, echo)
                _record_ltx_main_load_trace(ctx, "SingleGPUModelBuilder.build:start", detail, torch_module, echo)
                if kind == "main_transformer":
                    remember_first("ltx_main_load_trace_first_main_transformer_event", detail + " | " + _cuda_snapshot(torch_module))
                    ctx["ltx_main_load_trace_main_transformer_file_summary"] = size_summary
                component = _vae_component_name_from_configurator(cfg)
                if component is not None:
                    try:
                        cache = ctx.get("_ltx_vae_component_cache")
                        cached = cache.get(component) if isinstance(cache, dict) else None
                        if cached is not None and not _module_has_meta_tensors(cached):
                            ctx["ltx_vae_reuse_cache"] = f"REUSED: {component} returned from keepalive cache"
                            ctx[f"ltx_{component.lower()}_reuse_cache"] = f"REUSED: {_module_device_summary(cached)}"
                            _record_ltx_main_load_trace(ctx, f"FrameVision{component}ReuseCache:hit", f"reused cached {component}; requested device={device}; {_module_device_summary(cached)}", torch_module, echo)
                            try:
                                if device is not None:
                                    cached.to(device)
                            except Exception as exc:
                                ctx[f"ltx_{component.lower()}_reuse_cache_error"] = f"move to requested device failed: {type(exc).__name__}: {exc}"
                            return cached
                        if cached is not None:
                            ctx[f"ltx_{component.lower()}_reuse_cache"] = "MISS: cached module had meta tensors"
                    except Exception as exc:
                        ctx[f"ltx_{component.lower()}_reuse_cache_error"] = f"cache lookup failed: {type(exc).__name__}: {exc}"
                try:
                    result = original_build(self, *args, **kwargs)
                    if component is not None and result is not None:
                        try:
                            _install_vae_keepalive_to(result, component)
                            cache = ctx.get("_ltx_vae_component_cache")
                            if isinstance(cache, dict):
                                cache[component] = result
                            ctx["ltx_vae_reuse_cache"] = f"CACHED: {component} with CPU keepalive"
                            ctx[f"ltx_{component.lower()}_reuse_cache"] = f"CACHED: {_module_device_summary(result)}"
                            _record_ltx_main_load_trace(ctx, f"FrameVision{component}ReuseCache:store", f"cached {component}; {_module_device_summary(result)}", torch_module, echo)
                        except Exception as exc:
                            ctx[f"ltx_{component.lower()}_reuse_cache_error"] = f"cache store failed: {type(exc).__name__}: {exc}"
                    _record_ltx_main_load_trace(ctx, "SingleGPUModelBuilder.build:end", f"kind={kind}; result={type(result).__module__}.{type(result).__name__}", torch_module, echo)
                    return result
                except Exception as exc:
                    _record_ltx_main_load_trace(ctx, "SingleGPUModelBuilder.build:exception", f"kind={kind}; {type(exc).__name__}: {exc}", torch_module, echo)
                    raise
            try:
                traced_build.__name__ = getattr(original_build, "__name__", "build")
                traced_build.__doc__ = getattr(original_build, "__doc__", None)
                setattr(traced_build, "_framevision_ltx_main_load_trace", True)
                setattr(traced_build, "_framevision_ltx_main_load_trace_original", original_build)
            except Exception:
                pass
            setattr(cls, "build", traced_build)
            installed.append("SingleGPUModelBuilder.build")
    except Exception as exc:
        errors.append(f"SingleGPUModelBuilder.build: {type(exc).__name__}: {exc}")

    try:
        original_lmw = getattr(sgmb, "_load_model_weights")
        if not getattr(original_lmw, "_framevision_ltx_main_load_trace", False):
            def traced_load_model_weights(*args: Any, **kwargs: Any) -> Any:
                meta_model = kwargs.get("meta_model", args[0] if len(args) > 0 else None)
                model_path = kwargs.get("model_path", args[1] if len(args) > 1 else None)
                device = kwargs.get("device", args[5] if len(args) > 5 else None)
                dtype = kwargs.get("dtype", args[6] if len(args) > 6 else None)
                model_name = f"{type(meta_model).__module__}.{type(meta_model).__name__}" if meta_model is not None else "unknown"

                # Final split-file redirect. Vocoder already used this route in the
                # base file; this patch only extends the same single hook to the
                # late VideoDecoder and AudioDecoder split files. VideoEncoder and
                # text/embedding components are deliberately left untouched.
                component = _ltx_component_name_from_model_instance(meta_model)
                if component in ("Vocoder", "VideoDecoder", "AudioDecoder"):
                    override = _ltx_find_component_file_override(ctx, component)
                    component_key = str(component).lower()
                    if override is not None:
                        original_summary = _file_size_summary(_normalise_model_paths(model_path))
                        replacement = str(override)
                        if "model_path" in kwargs or len(args) <= 1:
                            kwargs = dict(kwargs)
                            kwargs["model_path"] = replacement
                        else:
                            arg_list = list(args)
                            arg_list[1] = replacement
                            args = tuple(arg_list)
                        model_path = replacement
                        status = f"{component}: final load redirected to split component file {override} ({_file_size_summary([replacement])}); original was {original_summary}"
                        ctx[f"ltx_component_file_{component_key}_status"] = "redirected_final_load"
                        ctx[f"ltx_component_file_{component_key}_path"] = replacement
                        ctx[f"ltx_component_file_{component_key}_original"] = original_summary
                        try:
                            events = ctx.setdefault("ltx_component_file_events", [])
                            if isinstance(events, list) and len(events) < 100:
                                events.append(status)
                        except Exception:
                            pass
                        if echo:
                            try:
                                print(f"[ltx-files] {status}", flush=True)
                            except Exception:
                                pass
                    else:
                        status = f"{component}: no split component file found at final load; using original path(s): {_file_size_summary(_normalise_model_paths(model_path))}"
                        ctx[f"ltx_component_file_{component_key}_status"] = "fallback_original_final_load"
                        try:
                            events = ctx.setdefault("ltx_component_file_events", [])
                            if isinstance(events, list) and len(events) < 100:
                                events.append(status)
                        except Exception:
                            pass
                        if echo:
                            try:
                                print(f"[ltx-files] WARNING: {status}", flush=True)
                            except Exception:
                                pass

                paths = _normalise_model_paths(model_path)
                size_summary = _file_size_summary(paths)
                detail = f"model={model_name}; device={device}; dtype={dtype}; {size_summary}"
                _record_ltx_main_load_trace(ctx, "_load_model_weights:start", detail, torch_module, echo)
                if "PromptEncoder" in model_name or "Gemma" in model_name or "TextEncoder" in model_name:
                    _ltx_quiet_status(ctx, "Loading text encoder")
                elif "ltx_core.model.transformer" in model_name or "LTXModel" in model_name:
                    _ltx_quiet_status(ctx, "Loading main/distilled safetensor")
                if "ltx_core.model.transformer" in model_name or "LTXModel" in model_name:
                    remember_first("ltx_main_load_trace_first_main_transformer_load_state_dict", detail + " | " + _cuda_snapshot(torch_module))
                    ctx["ltx_main_load_trace_main_transformer_file_summary"] = size_summary
                try:
                    result = original_lmw(*args, **kwargs)
                    _record_ltx_main_load_trace(ctx, "_load_model_weights:end", f"model={model_name}", torch_module, echo)
                    return result
                except Exception as exc:
                    _record_ltx_main_load_trace(ctx, "_load_model_weights:exception", f"model={model_name}; {type(exc).__name__}: {exc}", torch_module, echo)
                    raise
            try:
                traced_load_model_weights.__name__ = getattr(original_lmw, "__name__", "_load_model_weights")
                traced_load_model_weights.__doc__ = getattr(original_lmw, "__doc__", None)
                setattr(traced_load_model_weights, "_framevision_ltx_main_load_trace", True)
                setattr(traced_load_model_weights, "_framevision_ltx_main_load_trace_original", original_lmw)
            except Exception:
                pass
            setattr(sgmb, "_load_model_weights", traced_load_model_weights)
            installed.append("single_gpu_model_builder._load_model_weights")
    except Exception as exc:
        errors.append(f"_load_model_weights: {type(exc).__name__}: {exc}")

    try:
        original_sg_load_sd = getattr(sgmb, "load_state_dict")
        if not getattr(original_sg_load_sd, "_framevision_ltx_main_load_trace", False):
            def traced_sg_load_state_dict(model_path: Any, loader: Any, registry: Any, device: Any, model_sd_ops: Any = None) -> Any:
                paths = _normalise_model_paths(model_path)
                size_summary = _file_size_summary(paths)
                loader_name = f"{type(loader).__module__}.{type(loader).__name__}"
                _record_ltx_main_load_trace(ctx, "load_state_dict:start", f"device={device}; loader={loader_name}; {size_summary}", torch_module, echo)
                try:
                    result = original_sg_load_sd(model_path, loader, registry, device, model_sd_ops)
                    try:
                        sd_obj = getattr(result, "sd", None)
                        count = len(sd_obj) if hasattr(sd_obj, "__len__") else "n/a"
                    except Exception:
                        count = "n/a"
                    _record_ltx_main_load_trace(ctx, "load_state_dict:end", f"device={device}; tensors={count}; {size_summary}", torch_module, echo)
                    return result
                except Exception as exc:
                    _record_ltx_main_load_trace(ctx, "load_state_dict:exception", f"device={device}; {type(exc).__name__}: {exc}; {size_summary}", torch_module, echo)
                    raise
            try:
                traced_sg_load_state_dict.__name__ = getattr(original_sg_load_sd, "__name__", "load_state_dict")
                traced_sg_load_state_dict.__doc__ = getattr(original_sg_load_sd, "__doc__", None)
                setattr(traced_sg_load_state_dict, "_framevision_ltx_main_load_trace", True)
                setattr(traced_sg_load_state_dict, "_framevision_ltx_main_load_trace_original", original_sg_load_sd)
            except Exception:
                pass
            setattr(sgmb, "load_state_dict", traced_sg_load_state_dict)
            installed.append("single_gpu_model_builder.load_state_dict")
    except Exception as exc:
        errors.append(f"load_state_dict: {type(exc).__name__}: {exc}")

    try:
        ds_cls = getattr(blocks_mod, "DiffusionStage")
        original_bt = getattr(ds_cls, "_build_transformer")
        if not getattr(original_bt, "_framevision_ltx_main_load_trace", False):
            def traced_build_transformer(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    builder = getattr(self, "_transformer_builder", None)
                    kind = _classify_ltx_model_builder(builder)
                    paths = _normalise_model_paths(getattr(builder, "model_path", None))
                    size_summary = _file_size_summary(paths)
                    offload = getattr(self, "_offload_mode", "unknown")
                    device = kwargs.get("device", None) or getattr(self, "_device", None)
                    detail = f"kind={kind}; offload_mode={offload}; target={device}; {size_summary}"
                except Exception as exc:
                    detail = f"detail failed: {type(exc).__name__}: {exc}"
                _record_ltx_main_load_trace(ctx, "DiffusionStage._build_transformer:start", detail, torch_module, echo)
                _ltx_quiet_status(ctx, "Building video transformer")
                remember_first("ltx_main_load_trace_first_main_transformer_event", detail + " | " + _cuda_snapshot(torch_module))
                try:
                    result = original_bt(self, *args, **kwargs)
                    _record_ltx_main_load_trace(ctx, "DiffusionStage._build_transformer:end", f"result={type(result).__module__}.{type(result).__name__}", torch_module, echo)
                    return result
                except Exception as exc:
                    _record_ltx_main_load_trace(ctx, "DiffusionStage._build_transformer:exception", f"{type(exc).__name__}: {exc}", torch_module, echo)
                    raise
            try:
                traced_build_transformer.__name__ = getattr(original_bt, "__name__", "_build_transformer")
                traced_build_transformer.__doc__ = getattr(original_bt, "__doc__", None)
                setattr(traced_build_transformer, "_framevision_ltx_main_load_trace", True)
                setattr(traced_build_transformer, "_framevision_ltx_main_load_trace_original", original_bt)
            except Exception:
                pass
            setattr(ds_cls, "_build_transformer", traced_build_transformer)
            installed.append("DiffusionStage._build_transformer")
    except Exception as exc:
        errors.append(f"DiffusionStage._build_transformer: {type(exc).__name__}: {exc}")

    try:
        pe_cls = getattr(blocks_mod, "PromptEncoder")
        original_bte = getattr(pe_cls, "_build_text_encoder")
        if not getattr(original_bte, "_framevision_ltx_main_load_trace", False):
            def traced_build_text_encoder(self: Any, *args: Any, **kwargs: Any) -> Any:
                _record_ltx_main_load_trace(ctx, "PromptEncoder._build_text_encoder:start", "Gemma non-streaming owner path", torch_module, echo)
                _ltx_quiet_status(ctx, "Loading text encoder")
                try:
                    result = original_bte(self, *args, **kwargs)
                    _record_ltx_main_load_trace(ctx, "PromptEncoder._build_text_encoder:end", f"result={type(result).__module__}.{type(result).__name__}", torch_module, echo)
                    _ltx_quiet_status(ctx, "Text encoder ready")
                    return result
                except Exception as exc:
                    _record_ltx_main_load_trace(ctx, "PromptEncoder._build_text_encoder:exception", f"{type(exc).__name__}: {exc}", torch_module, echo)
                    raise
            try:
                setattr(traced_build_text_encoder, "_framevision_ltx_main_load_trace", True)
                setattr(traced_build_text_encoder, "_framevision_ltx_main_load_trace_original", original_bte)
            except Exception:
                pass
            setattr(pe_cls, "_build_text_encoder", traced_build_text_encoder)
            installed.append("PromptEncoder._build_text_encoder")

        # Best-effort prompt/text encoding labels. Different LTX versions expose
        # this phase through different PromptEncoder methods, so wrap whichever
        # candidates exist without changing their arguments or return values.
        for method_name in ("encode", "encode_prompt", "forward", "__call__"):
            try:
                original_method = getattr(pe_cls, method_name, None)
                if original_method is None or not callable(original_method):
                    continue
                if getattr(original_method, "_framevision_ltx_prompt_encode_status", False):
                    continue

                def make_prompt_encode_wrapper(original: Any, label: str) -> Any:
                    def wrapped_prompt_encode(self: Any, *args: Any, **kwargs: Any) -> Any:
                        _ltx_quiet_status(ctx, "Encoding text")
                        try:
                            return original(self, *args, **kwargs)
                        finally:
                            _ltx_quiet_status(ctx, "Text encoded")
                    try:
                        wrapped_prompt_encode.__name__ = getattr(original, "__name__", label)
                        wrapped_prompt_encode.__doc__ = getattr(original, "__doc__", None)
                        setattr(wrapped_prompt_encode, "_framevision_ltx_prompt_encode_status", True)
                        setattr(wrapped_prompt_encode, "_framevision_ltx_prompt_encode_original", original)
                    except Exception:
                        pass
                    return wrapped_prompt_encode

                setattr(pe_cls, method_name, make_prompt_encode_wrapper(original_method, method_name))
                installed.append(f"PromptEncoder.{method_name} status")
            except Exception as exc:
                errors.append(f"PromptEncoder.{method_name} status: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"PromptEncoder._build_text_encoder/text encoding status: {type(exc).__name__}: {exc}")

    ctx["ltx_main_load_trace_installed"] = "YES" if installed else "NO"
    ctx["ltx_main_load_trace_targets"] = " | ".join(installed) if installed else "none"
    ctx["ltx_main_load_trace_errors"] = "none" if not errors else " | ".join(errors[-8:])
    if installed:
        ctx.setdefault("notes", []).append(
            "Installed LTX main-load start trace: records DiffusionStage, SingleGPUModelBuilder, _load_model_weights, and load_state_dict boundaries before the large transformer safetensor can become a full CUDA model. Diagnostic only; no residency behavior changes."
        )
        _record_ltx_main_load_trace(ctx, "trace_installed", ctx["ltx_main_load_trace_targets"], torch_module, echo)


def _install_ltx_batch_split_early_residency_guard(
    ctx: Dict[str, Any],
    hooks_mod: Any,
    torch_module: Any,
    mode: str,
    echo: bool = False,
) -> None:
    """Attach VRAM Lab to LTX BatchSplitAdapter as soon as it is constructed.

    This is the first FrameVision-side load-boundary step that is earlier than
    the current X0Model first-call discovery. The latest report showed the huge
    pre-hook spike at ltx_core.batch_split.BatchSplitAdapter before useful
    denoise hooks could manage residency. BatchSplitAdapter owns/wraps the LTX
    model object and exposes the same transformer block path reported by the
    tracer (_model.velocity_model.transformer_blocks.*), so attaching here lets
    VRAM Lab park those blocks to CPU/RAM before the first BatchSplitAdapter
    forward instead of waiting for X0Model's first inner call.

    This does not edit the LTX repo, does not patch Tensor.to, does not change
    the 0.7.4 denoise hook runtime, and does not add a profile/knob.
    """
    stats = {
        "installed": False,
        "attach_attempts": 0,
        "attached": 0,
        "blocks": 0,
        "errors": [],
    }
    ctx["ltx_batch_split_early_guard_installed"] = "NO"
    ctx["ltx_batch_split_early_guard_attach_attempts"] = "0"
    ctx["ltx_batch_split_early_guard_attached"] = "0"
    ctx["ltx_batch_split_early_guard_blocks"] = "0"
    ctx["ltx_batch_split_early_guard_errors"] = "none"

    try:
        attach_vram_hooks = getattr(hooks_mod, "attach_vram_hooks")
    except Exception as exc:
        ctx["ltx_batch_split_early_guard_errors"] = f"attach_vram_hooks unavailable: {type(exc).__name__}: {exc}"
        return

    try:
        batch_mod = importlib.import_module("ltx_core.batch_split")
        cls = getattr(batch_mod, "BatchSplitAdapter", None)
        if cls is None:
            raise AttributeError("ltx_core.batch_split.BatchSplitAdapter not found")
    except Exception as exc:
        ctx["ltx_batch_split_early_guard_errors"] = f"BatchSplitAdapter import failed: {type(exc).__name__}: {exc}"
        return

    try:
        original_init = getattr(cls, "__init__")
        if getattr(original_init, "_framevision_vram_lab_batch_split_guard", False):
            ctx["ltx_batch_split_early_guard_installed"] = "YES: already installed"
            return
    except Exception as exc:
        ctx["ltx_batch_split_early_guard_errors"] = f"BatchSplitAdapter.__init__ unavailable: {type(exc).__name__}: {exc}"
        return

    hook_pattern = (
        r"(^|\.)(transformer_blocks|blocks|layers|double_blocks|single_blocks|temporal_blocks|spatial_blocks)\.\d+$"
        r"|(^|\.)(temporal|spatial).*blocks?\.\d+$"
    )

    def _safe_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return float(default)
            text = str(value).strip()
            if not text or text.lower() in {"n/a", "none", "unknown"}:
                return float(default)
            return float(text.split()[0])
        except Exception:
            return float(default)

    # Keep the BatchSplitAdapter hook wrapper self-contained. These values are
    # calculated in main(), stored in ctx for the report, and read back here so
    # the patched __init__ closure does not depend on main() local variables.
    # This fixes the split-stage hotset NameError and keeps the actual VRAM Lab
    # core generic; only the caller supplies stage-specific policy values.
    residency_strategy_value = str(
        ctx.get("vram_residency_strategy_requested")
        or ctx.get("vram_residency_strategy")
        or "planned_hotset"
    ).strip().lower()
    if residency_strategy_value not in {"rolling", "planned_hotset"}:
        residency_strategy_value = "planned_hotset"

    shared_hotset_fraction = max(
        0.10,
        min(2.00, _safe_float(ctx.get("vram_stable_hotset_fraction_requested"), 0.95)),
    )
    stage1_hotset_fraction = max(
        0.10,
        min(2.00, _safe_float(ctx.get("vram_stage1_stable_hotset_fraction_requested"), shared_hotset_fraction)),
    )
    stage2_hotset_fraction = max(
        0.70,
        min(1.10, _safe_float(ctx.get("vram_stage2_stable_hotset_fraction_requested"), min(1.10, shared_hotset_fraction))),
    )
    stable_hotset_budget_gb_value = max(
        0.0,
        _safe_float(ctx.get("vram_stable_hotset_budget_gb_requested"), 0.0),
    )
    emergency_floor_gb_value = max(
        0.25,
        min(3.00, _safe_float(ctx.get("vram_emergency_driver_free_floor_requested_gb"), _safe_float(ctx.get("vram_emergency_driver_free_floor"), 1.5))),
    )
    emergency_floor_bytes_value = int(float(emergency_floor_gb_value) * 1024 ** 3)
    ctx["vram_emergency_driver_free_floor"] = f"{float(emergency_floor_gb_value):.2f} GB"

    def _record_runtime(runtime: Any) -> None:
        try:
            runtimes = ctx.setdefault("_ltx_early_residency_runtimes", [])
            if isinstance(runtimes, list):
                runtimes.append(runtime)
                if ctx.get("_ltx_phase_retention_runtime") is None:
                    ctx["_ltx_phase_retention_runtime"] = runtime
        except Exception:
            pass
        try:
            runtime.update_context(ctx)
            _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="attached early to BatchSplitAdapter before first forward")
        except Exception as exc:
            stats["errors"].append(f"early runtime context update failed: {type(exc).__name__}: {exc}")

    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "_framevision_vram_lab_batch_split_runtime", None) is not None:
            return
        stats["attach_attempts"] = int(stats.get("attach_attempts", 0)) + 1
        ctx["ltx_batch_split_early_guard_attach_attempts"] = str(stats["attach_attempts"])
        try:
            attach_index = int(stats.get("attach_attempts", 1) or 1)
            stage_role = "stage1_initial_denoise" if attach_index == 1 else "stage2_refine_denoise" if attach_index == 2 else f"adapter_{attach_index}"
            active_limit_gb, active_limit_source = _block_limit_for_role(ctx, stage_role)
            if stage_role == "stage2_refine_denoise":
                role_hotset_fraction = stage2_hotset_fraction
            elif stage_role == "stage1_initial_denoise":
                role_hotset_fraction = stage1_hotset_fraction
            else:
                role_hotset_fraction = shared_hotset_fraction
            # If the wrapper retained the Stage-1 transformer for Stage 2, the
            # original block hooks are already installed on this same model.
            # Skipping a second attach avoids an initial CPU park/reload at the
            # phase boundary while keeping VRAM Lab core generic.
            try:
                adapter_model = getattr(self, "_model", None)
                if stage_role == "stage2_refine_denoise" and getattr(adapter_model, "_framevision_ltx_retained_for_stage2", False):
                    runtime = ctx.get("_ltx_phase_retention_runtime")
                    if runtime is None:
                        runtimes = ctx.get("_ltx_early_residency_runtimes") or []
                        if isinstance(runtimes, list) and runtimes:
                            runtime = runtimes[0]
                    setattr(self, "_framevision_vram_lab_batch_split_runtime", runtime)
                    ctx["ltx_phase_retention_stage2_attach"] = "SKIPPED: reused Stage-1 hooks/runtime"
                    ctx["hook_attachment_status"] = "stage2 reused Stage-1 hooks/runtime"
                    if runtime is not None:
                        try:
                            _install_ltx_stage2_step_profiler_on_adapter(ctx, self, runtime, torch_module, echo=echo)
                        except Exception as exc:
                            ctx["ltx_phase_retention_stage2_attach_profiler_error"] = f"{type(exc).__name__}: {exc}"
                    return
            except Exception as exc:
                ctx["ltx_phase_retention_stage2_attach"] = f"skip check failed: {type(exc).__name__}: {exc}"
            policy_payload = {
                "mode": str(mode or "safe"),
                "device": "cuda",
                "hook_name_regex": hook_pattern,
                "max_blocks": 256,
                "residency_strategy": residency_strategy_value,
                "stable_hotset_fraction": float(role_hotset_fraction),
                "stable_hotset_budget_bytes": int(float(stable_hotset_budget_gb_value) * 1024 ** 3) if float(stable_hotset_budget_gb_value or 0.0) > 0 else 0,
                "emergency_driver_free_floor_bytes": emergency_floor_bytes_value,
                "driver_free_floor_gb": float(emergency_floor_gb_value),
            }
            ctx[f"vram_stable_hotset_fraction_for_{stage_role}"] = f"{float(role_hotset_fraction):.2f}"
            if active_limit_gb > 0:
                active_limit_bytes = int(float(active_limit_gb) * 1024 ** 3)
                # Important: pass the stage-specific budget before hooks attach.
                # Updating after attach was too late for the Stage-2 constructor spike.
                policy_payload.update({
                    "hot_block_budget_bytes": active_limit_bytes,
                    "safe_hot_window_gb": float(active_limit_gb),
                    "safe_hot_window_bytes": active_limit_bytes,
                    "balanced_hot_window_gb": float(active_limit_gb),
                    "balanced_hot_window_bytes": active_limit_bytes,
                })
                ctx[f"pre_attach_block_limit_for_{stage_role}"] = f"{float(active_limit_gb):.1f} GB ({active_limit_source})"
            runtime = attach_vram_hooks(
                {"BatchSplitAdapterEarly": self},
                policy=policy_payload,
                torch_module=torch_module,
            )
            block_count = int(getattr(runtime, "blocks", []) and len(runtime.blocks) or 0)
            if block_count > 0:
                setattr(self, "_framevision_vram_lab_batch_split_runtime", runtime)
                stats["attached"] = int(stats.get("attached", 0)) + 1
                stats["blocks"] = max(int(stats.get("blocks", 0)), block_count)
                ctx["ltx_batch_split_early_guard_attached"] = str(stats["attached"])
                ctx["ltx_batch_split_early_guard_blocks"] = str(stats["blocks"])
                ctx["hook_attachment_attempted"] = "YES"
                ctx["hook_attachment_status"] = "attached early to BatchSplitAdapter before first forward"
                ctx["hooked_component_names"] = "BatchSplitAdapterEarly"
                ctx["hooked_block_count"] = str(block_count)
                _record_runtime(runtime)
                _repair_ltx_runtime_planned_hotset(ctx, runtime, stage_role, torch_module=torch_module, echo=echo, reason="early-attach")
                try:
                    ctx[f"vram_residency_strategy_for_{stage_role}"] = str(ctx.get("vram_residency_strategy", "n/a"))
                    ctx[f"vram_stable_hotset_count_for_{stage_role}"] = str(ctx.get("vram_stable_hotset_count", "n/a"))
                    ctx[f"vram_stable_hotset_bytes_for_{stage_role}"] = str(ctx.get("vram_stable_hotset_bytes", "n/a"))
                    ctx[f"vram_stable_hotset_fraction_effective_for_{stage_role}"] = str(ctx.get("vram_stable_hotset_fraction", "n/a"))
                except Exception:
                    pass
                # The stage-2 step profiler is useful for two-stage pipelines, but
                # keep it out of one-stage completely. One-stage should stay on the
                # proven early-residency path only, without the extra forward wrapper
                # or stage labels that are specific to two-stage refinement.
                selected_pipeline = str(ctx.get("selected_pipeline", "one_stage"))
                if selected_pipeline in TWO_STAGE_PIPELINE_SET:
                    try:
                        _install_ltx_stage2_step_profiler_on_adapter(ctx, self, runtime, torch_module, echo=echo)
                    except Exception as exc:
                        ctx["ltx_stage2_step_profiler_errors"] = f"attach failed: {type(exc).__name__}: {exc}"
                else:
                    ctx["ltx_stage2_step_profiler_installed"] = "NO"
                    ctx["ltx_stage2_step_profiler_targets"] = "disabled for one-stage"
                    ctx["ltx_stage2_step_profiler_errors"] = "none"
                    ctx["ltx_stage2_step_summary"] = "disabled: selected pipeline is one_stage"
                    ctx.setdefault("notes", []).append(
                        "Skipped stage-2 step profiler for one-stage; early residency hooks remain active."
                    )
                if echo:
                    print(f"[vram-lab-ltx-early] BatchSplitAdapter early residency attached: {block_count} blocks", flush=True)
            else:
                try:
                    runtime.detach_vram_hooks()
                except Exception:
                    pass
                stats["errors"].append("BatchSplitAdapter early attach found 0 matching blocks")
        except Exception as exc:
            stats["errors"].append(f"BatchSplitAdapter early attach failed: {type(exc).__name__}: {exc}")
        finally:
            ctx["ltx_batch_split_early_guard_attach_attempts"] = str(stats["attach_attempts"])
            ctx["ltx_batch_split_early_guard_attached"] = str(stats["attached"])
            ctx["ltx_batch_split_early_guard_blocks"] = str(stats["blocks"])
            ctx["ltx_batch_split_early_guard_errors"] = "none" if not stats["errors"] else " | ".join(str(x) for x in stats["errors"][-8:])

    try:
        guarded_init.__name__ = getattr(original_init, "__name__", "__init__")
        guarded_init.__doc__ = getattr(original_init, "__doc__", None)
        setattr(guarded_init, "_framevision_vram_lab_batch_split_guard", True)
        setattr(guarded_init, "_framevision_vram_lab_original", original_init)
        setattr(cls, "__init__", guarded_init)
        stats["installed"] = True
        ctx["ltx_batch_split_early_guard_installed"] = "YES"
        ctx.setdefault("notes", []).append(
            "Installed LTX BatchSplitAdapter early residency guard: transformer blocks are hooked/parked when the adapter is constructed, before first forward."
        )
    except Exception as exc:
        stats["errors"].append(f"BatchSplitAdapter.__init__ patch failed: {type(exc).__name__}: {exc}")
        ctx["ltx_batch_split_early_guard_errors"] = " | ".join(str(x) for x in stats["errors"][-8:])


def _copy_vram_runtime_fields(ctx: Dict[str, Any], attached_name: str = "BatchSplitAdapterEarly", attached_status: str = "attached early") -> None:
    """Copy shared VRAM runtime fields into the LTX wrapper's report keys."""
    ctx["hook_attachment_attempted"] = "YES"
    ctx["hook_attachment_status"] = attached_status
    ctx["hooked_component_names"] = ctx.get("vram_hooked_component_names", attached_name)
    ctx["hooked_block_count"] = ctx.get("vram_hooked_block_count", ctx.get("hooked_block_count", "0"))
    ctx["pre_forward_hook_calls"] = ctx.get("vram_pre_forward_calls", ctx.get("pre_forward_hook_calls", "0"))
    ctx["post_forward_hook_calls"] = ctx.get("vram_post_forward_calls", ctx.get("post_forward_hook_calls", "0"))
    ctx["block_load_count"] = ctx.get("vram_block_load_count", ctx.get("block_load_count", "0"))
    ctx["block_unload_count"] = ctx.get("vram_block_unload_count", ctx.get("block_unload_count", "0"))
    ctx["peak_cuda_during_hooked_execution"] = ctx.get("vram_peak_cuda_during_hooked_execution", ctx.get("peak_cuda_during_hooked_execution", "n/a"))



def _stage2_limit_enabled_for_ctx(ctx: Dict[str, Any]) -> bool:
    try:
        return str(ctx.get("separate_stage2_block_limit", "OFF")).upper().startswith("ON")
    except Exception:
        return False


def _ctx_float(ctx: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(str(ctx.get(key, default)).strip().split()[0])
    except Exception:
        return float(default)


def _block_limit_for_role(ctx: Dict[str, Any], role: str) -> tuple[float, str]:
    """Return active LTX block/hot-window limit for a stage role.

    Stage 2 is opt-in. If unset/disabled, it deliberately falls back to the
    existing Stage 1/general limit so one-stage and old two-stage commands stay
    unchanged.
    """
    role_l = str(role or "").lower()
    stage1_gb = _ctx_float(ctx, "stage1_block_size_limit_gb", _ctx_float(ctx, "ltx_main_profile_hot_window_gb", 0.0))
    stage2_gb = _ctx_float(ctx, "stage2_block_size_limit_gb", 0.0)
    if _stage2_limit_enabled_for_ctx(ctx) and stage2_gb > 0.0 and any(x in role_l for x in ("stage2", "refine", "second_stage", "second pass")):
        return stage2_gb, str(ctx.get("stage2_limit_source", "CLI/UI"))
    return stage1_gb, "Stage 1/general fallback"



def _repair_ltx_runtime_planned_hotset(
    ctx: Dict[str, Any],
    runtime: Any,
    role: str,
    torch_module: Any | None = None,
    echo: bool = False,
    reason: str = "attach",
) -> None:
    """Keep LTX planned_hotset from silently falling back to full 48/48 churn.

    The VRAM Lab core can only build a planned hotset when it can estimate block
    sizes. After a fresh install or after some LTX handoffs, the size estimate can
    be missing/zero at attach time. In that case the runtime still says
    planned_hotset, but every step loads/unloads every block again. This helper
    re-runs the normal planner and, only if it still produced an empty hotset,
    installs a small count-based prefix fallback.
    """
    try:
        if runtime is None:
            return
        role_s = str(role or "")
        strategy = str(getattr(runtime, "residency_strategy", "") or "").strip().lower()
        if strategy not in {"planned_hotset", "planned", "stable", "stable_prefix", "planned_stable"}:
            return
        if strategy != "planned_hotset":
            try:
                setattr(runtime, "residency_strategy", "planned_hotset")
            except Exception:
                pass
        gb, source = _block_limit_for_role(ctx, role_s)
        if gb > 0.0:
            bytes_value = int(float(gb) * 1024 ** 3)
            try:
                setattr(runtime, "hot_block_budget_bytes", bytes_value)
                setattr(runtime, "safe_hot_window_gb", float(gb))
                setattr(runtime, "balanced_hot_window_gb", float(gb))
                policy = getattr(runtime, "policy", None)
                if isinstance(policy, dict):
                    policy["hot_block_budget_bytes"] = bytes_value
                    policy["safe_hot_window_gb"] = float(gb)
                    policy["safe_hot_window_bytes"] = bytes_value
                    policy["balanced_hot_window_gb"] = float(gb)
                    policy["balanced_hot_window_bytes"] = bytes_value
            except Exception:
                pass
        try:
            if "stage2" in role_s.lower() or "refine" in role_s.lower():
                fraction = _ctx_float(ctx, "vram_stage2_stable_hotset_fraction_requested", _ctx_float(ctx, "vram_stage2_stable_hotset_fraction", 0.90))
            elif "stage1" in role_s.lower():
                fraction = _ctx_float(ctx, "vram_stage1_stable_hotset_fraction_requested", _ctx_float(ctx, "vram_stage1_stable_hotset_fraction", 1.15))
            else:
                fraction = _ctx_float(ctx, "vram_stable_hotset_fraction_requested", _ctx_float(ctx, "vram_stable_hotset_fraction", 0.95))
            fraction = max(0.10, min(2.00, float(fraction)))
            setattr(runtime, "stable_hotset_fraction", float(fraction))
            policy = getattr(runtime, "policy", None)
            if isinstance(policy, dict):
                policy["stable_hotset_fraction"] = float(fraction)
        except Exception:
            fraction = 0.0
        before_count = 0
        try:
            before_count = len(getattr(runtime, "_stable_hotset_names", set()) or set())
        except Exception:
            before_count = 0
        try:
            if hasattr(runtime, "_plan_stable_hotset"):
                runtime._plan_stable_hotset()
        except Exception as exc:
            ctx["ltx_planned_hotset_repair_errors"] = f"normal plan failed for {role_s}: {type(exc).__name__}: {exc}"
        try:
            names = getattr(runtime, "_stable_hotset_names", set()) or set()
            order = getattr(runtime, "_stable_hotset_order", []) or []
            after_count = len(names)
        except Exception:
            names, order, after_count = set(), [], 0
        fallback_used = False
        # Last-resort stability fix: if the normal planner produced no stable
        # blocks, keep a small prefix by count. This is safer than allowing the
        # pathological 48-load/48-unload mode and keeps Stage 2 conservative.
        if after_count <= 0:
            try:
                blocks = list(getattr(runtime, "blocks", []) or [])
                block_count = len(blocks)
                if block_count > 1:
                    if "stage2" in role_s.lower() or "refine" in role_s.lower():
                        target = max(2, min(block_count - 1, int(round(block_count * 0.13))))
                    elif "stage1" in role_s.lower():
                        target = max(8, min(block_count - 1, int(round(block_count * 0.44))))
                    else:
                        target = max(4, min(block_count - 1, int(round(block_count * 0.25))))
                    picked = []
                    total = 0
                    for block in blocks[:target]:
                        name = getattr(block, "name", None)
                        if not name:
                            continue
                        picked.append(str(name))
                        try:
                            total += int(getattr(block, "bytes", 0) or 0)
                        except Exception:
                            pass
                    if picked:
                        setattr(runtime, "_stable_hotset_names", set(picked))
                        setattr(runtime, "_stable_hotset_order", list(picked))
                        setattr(runtime, "_stable_hotset_bytes", int(total))
                        after_count = len(picked)
                        fallback_used = True
            except Exception as exc:
                ctx["ltx_planned_hotset_repair_errors"] = f"fallback failed for {role_s}: {type(exc).__name__}: {exc}"
        try:
            if hasattr(runtime, "update_context"):
                runtime.update_context(ctx)
                _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status=f"planned_hotset repair for {role_s}")
        except Exception:
            pass
        try:
            ctx[f"ltx_planned_hotset_repair_for_{role_s}"] = (
                f"reason={reason}; before={before_count}; after={after_count}; "
                f"fallback={'YES' if fallback_used else 'NO'}; fraction={float(fraction):.2f}; "
                f"limit={float(gb):.1f} GB ({source})"
            )
            if echo:
                print(
                    f"[vram-lab-ltx-hotset] {role_s}: planned hotset repair "
                    f"reason={reason}; before={before_count}; after={after_count}; "
                    f"fallback={'YES' if fallback_used else 'NO'}; limit={float(gb):.1f}GB ({source}) | {_cuda_snapshot(torch_module)}",
                    flush=True,
                )
        except Exception:
            pass
    except Exception as exc:
        try:
            ctx["ltx_planned_hotset_repair_errors"] = f"repair failed for {role}: {type(exc).__name__}: {exc}"
        except Exception:
            pass

def _apply_runtime_block_limit_for_role(
    ctx: Dict[str, Any],
    runtime: Any,
    role: str,
    torch_module: Any | None = None,
    echo: bool = False,
) -> None:
    """Narrowly retarget the existing VRAM Lab runtime hot-window for Stage 2.

    This does not rewrite VRAM Lab. It updates the already-created runtime's
    existing hot_block_budget fields before that stage's first forward.
    """
    try:
        gb, source = _block_limit_for_role(ctx, role)
        if gb <= 0.0 or runtime is None:
            return
        bytes_value = int(float(gb) * 1024 ** 3)
        try:
            setattr(runtime, "hot_block_budget_bytes", bytes_value)
            setattr(runtime, "safe_hot_window_gb", float(gb))
            setattr(runtime, "balanced_hot_window_gb", float(gb))
            policy = getattr(runtime, "policy", None)
            if isinstance(policy, dict):
                policy["hot_block_budget_bytes"] = bytes_value
                policy["safe_hot_window_gb"] = float(gb)
                policy["safe_hot_window_bytes"] = bytes_value
                policy["balanced_hot_window_gb"] = float(gb)
                policy["balanced_hot_window_bytes"] = bytes_value
                floor_gb = max(0.25, min(3.00, _ctx_float(ctx, "vram_emergency_driver_free_floor_requested_gb", _ctx_float(ctx, "vram_emergency_driver_free_floor", 1.5))))
                policy["emergency_driver_free_floor_bytes"] = int(float(floor_gb) * 1024 ** 3)
                policy["driver_free_floor_gb"] = float(floor_gb)
                ctx["vram_emergency_driver_free_floor"] = f"{float(floor_gb):.2f} GB"
        except Exception as exc:
            ctx["stage2_block_limit_runtime_errors"] = f"runtime update failed for {role}: {type(exc).__name__}: {exc}"
            return
        key = f"active_block_limit_for_{role}"
        ctx[key] = f"{gb:.1f} GB"
        if "stage1" in str(role).lower():
            ctx["active_block_limit_for_stage1_initial_denoise"] = f"{gb:.1f} GB"
        if "stage2" in str(role).lower() or "refine" in str(role).lower():
            ctx["active_block_limit_for_stage2_refine_denoise"] = f"{gb:.1f} GB"
        ctx["last_active_stage_block_limit"] = f"{role}: {gb:.1f} GB ({source})"
        ctx["stage2_block_limit_runtime_errors"] = str(ctx.get("stage2_block_limit_runtime_errors", "none") or "none")
        _repair_ltx_runtime_planned_hotset(ctx, runtime, role, torch_module=torch_module, echo=echo, reason="block-limit-apply")
        try:
            runtime.update_context(ctx)
            _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="attached early to BatchSplitAdapter before first forward")
        except Exception:
            pass
        if echo:
            try:
                print(f"[vram-lab-ltx-stage-limit] {role}: active block limit {gb:.1f} GB ({source}) | {_cuda_snapshot(torch_module)}", flush=True)
            except Exception:
                pass
    except Exception as exc:
        ctx["stage2_block_limit_runtime_errors"] = f"apply failed for {role}: {type(exc).__name__}: {exc}"


def _ltx_runtime_hotset_watchdog_check(
    ctx: Dict[str, Any],
    runtime: Any,
    role: str,
    torch_module: Any | None = None,
    echo: bool = False,
    low_duration_hint_s: float = 0.0,
    reason: str = "after-forward",
) -> None:
    """Lower the active LTX hotset/block budget for this run only under sustained VRAM pressure.

    This is intentionally runtime-only: it does not save JSON and does not create
    per-checkpoint model rules.  It reacts to real driver-free VRAM observed
    during the current run and lowers the current role's hot-window by 0.5 GB
    when pressure stays too high long enough.  The goal is to continue the run
    instead of letting a near-full 24 GB card drift into shared memory or a
    native/device mismatch crash.
    """
    try:
        if runtime is None or str(ctx.get("ltx_runtime_hotset_watchdog_enabled", "YES")).upper().startswith("NO"):
            return
        tm = torch_module
        if tm is None:
            try:
                import torch as tm  # type: ignore
            except Exception:
                tm = None
        if tm is None or not hasattr(tm, "cuda") or not tm.cuda.is_available():
            return
        try:
            free_b, total_b = tm.cuda.mem_get_info()
        except Exception:
            return
        now = time.monotonic()
        role_s = str(role or "stage1_initial_denoise")
        try:
            floor_gb = _ctx_float(ctx, "vram_emergency_driver_free_floor_requested_gb", _ctx_float(ctx, "vram_emergency_driver_free_floor", 1.0))
        except Exception:
            floor_gb = 1.0
        # Use the configured emergency/free floor as the maximum allowed pressure.
        # Keep a sane minimum so custom low floors still get protection.
        threshold_gb = max(0.75, min(3.0, _ctx_float(ctx, "ltx_runtime_hotset_watchdog_threshold_gb", floor_gb)))
        hold_s = max(1.0, min(10.0, _ctx_float(ctx, "ltx_runtime_hotset_watchdog_hold_s", 3.0)))
        step_gb = max(0.1, min(2.0, _ctx_float(ctx, "ltx_runtime_hotset_watchdog_step_gb", 0.5)))
        min_gb = max(1.5, min(18.0, _ctx_float(ctx, "ltx_runtime_hotset_watchdog_min_gb", 1.5)))
        max_retunes = max(1, min(20, int(_ctx_float(ctx, "ltx_runtime_hotset_watchdog_max_retunes", 8))))
        threshold_b = int(float(threshold_gb) * 1024 ** 3)
        # Hard pressure means Task Manager can already show shared-memory spill
        # or driver_free can be zero during a long Stage-2 forward. Do not wait
        # for the soft hold timer in that case; lower the active budget now.
        hard_threshold_gb = max(0.50, min(1.00, _ctx_float(ctx, "ltx_runtime_hotset_watchdog_hard_threshold_gb", 0.75)))
        hard_threshold_b = int(float(hard_threshold_gb) * 1024 ** 3)
        hard_pressure = int(free_b) <= hard_threshold_b
        if hard_pressure:
            step_gb = max(float(step_gb), _ctx_float(ctx, "ltx_runtime_hotset_watchdog_hard_step_gb", 0.5))
        ctx["ltx_runtime_hotset_watchdog_last_driver_free"] = _fmt_bytes(int(free_b))
        ctx["ltx_runtime_hotset_watchdog_last_role"] = role_s
        if int(free_b) >= threshold_b:
            ctx["_ltx_runtime_hotset_watchdog_low_since"] = None
            return
        low_since = ctx.get("_ltx_runtime_hotset_watchdog_low_since")
        if not isinstance(low_since, (int, float)):
            low_since = now
            ctx["_ltx_runtime_hotset_watchdog_low_since"] = low_since
        observed_s = max(float(now - float(low_since)), float(low_duration_hint_s or 0.0))
        if hard_pressure:
            observed_s = max(observed_s, hold_s)
        pressure_kind = "hard" if hard_pressure else "soft"
        ctx["ltx_runtime_hotset_watchdog_pressure"] = (
            f"{pressure_kind} pressure: driver_free={_fmt_bytes(int(free_b))} below {float(threshold_gb):.2f} GB "
            f"for about {observed_s:.2f}s; hard threshold={float(hard_threshold_gb):.2f} GB"
        )
        if observed_s < hold_s:
            return
        retune_count = int(_ctx_float(ctx, "ltx_runtime_hotset_watchdog_retune_count", 0))
        if retune_count >= max_retunes:
            ctx["ltx_runtime_hotset_watchdog_status"] = f"limit reached: {retune_count}/{max_retunes} retunes"
            return
        current_gb, source = _block_limit_for_role(ctx, role_s)
        if current_gb <= 0.0:
            return
        new_gb = max(float(min_gb), round((float(current_gb) - float(step_gb)) * 10.0) / 10.0)
        if new_gb >= float(current_gb) - 1e-6:
            ctx["ltx_runtime_hotset_watchdog_status"] = f"at minimum {float(min_gb):.1f} GB; cannot lower {role_s} further"
            return
        before = _cuda_snapshot(tm)
        bytes_value = int(float(new_gb) * 1024 ** 3)
        try:
            setattr(runtime, "hot_block_budget_bytes", bytes_value)
            setattr(runtime, "safe_hot_window_gb", float(new_gb))
            setattr(runtime, "balanced_hot_window_gb", float(new_gb))
            policy = getattr(runtime, "policy", None)
            if isinstance(policy, dict):
                policy["hot_block_budget_bytes"] = bytes_value
                policy["safe_hot_window_gb"] = float(new_gb)
                policy["safe_hot_window_bytes"] = bytes_value
                policy["balanced_hot_window_gb"] = float(new_gb)
                policy["balanced_hot_window_bytes"] = bytes_value
        except Exception as exc:
            ctx["ltx_runtime_hotset_watchdog_errors"] = f"runtime budget update failed: {type(exc).__name__}: {exc}"
            return
        role_l = role_s.lower()
        if "stage2" in role_l or "refine" in role_l:
            ctx["stage2_block_size_limit_gb"] = f"{float(new_gb):.1f}"
            ctx["active_block_limit_for_stage2_refine_denoise"] = f"{float(new_gb):.1f} GB"
            ctx["stage2_limit_source"] = f"runtime watchdog lowered from {float(current_gb):.1f} GB ({source})"
        else:
            ctx["stage1_block_size_limit_gb"] = f"{float(new_gb):.1f}"
            ctx["active_block_limit_for_stage1_initial_denoise"] = f"{float(new_gb):.1f} GB"
            ctx["ltx_main_profile_hot_window_gb"] = f"{float(new_gb):.1f}"
            ctx["ltx_main_hot_window_override_gb"] = f"{float(new_gb):.1f}"
        ctx["last_active_stage_block_limit"] = f"{role_s}: {float(new_gb):.1f} GB (runtime watchdog lowered from {float(current_gb):.1f} GB)"
        trim_status = "not attempted"
        try:
            if hasattr(runtime, "_trim_hot_blocks"):
                runtime._trim_hot_blocks(keep_name="")
                trim_status = "trimmed hot blocks"
            else:
                trim_status = "runtime has no _trim_hot_blocks"
            if hard_pressure:
                try:
                    if hasattr(tm, "cuda"):
                        tm.cuda.empty_cache()
                        if hasattr(tm.cuda, "ipc_collect"):
                            tm.cuda.ipc_collect()
                    trim_status += " + hard-pressure cuda cleanup"
                except Exception as cleanup_exc:
                    trim_status += f" + cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}"
        except Exception as exc:
            trim_status = f"trim failed: {type(exc).__name__}: {exc}"
        try:
            if hasattr(runtime, "update_context"):
                runtime.update_context(ctx)
                _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="runtime hotset watchdog lowered active budget")
        except Exception:
            pass
        after = _cuda_snapshot(tm)
        retune_count += 1
        ctx["ltx_runtime_hotset_watchdog_retune_count"] = str(retune_count)
        ctx["ltx_runtime_hotset_watchdog_last_retune"] = (
            f"{role_s}: {float(current_gb):.1f}->{float(new_gb):.1f} GB; "
            f"reason={reason}; free={_fmt_bytes(int(free_b))}; observed={observed_s:.2f}s; {trim_status}"
        )
        hist = ctx.setdefault("ltx_runtime_hotset_watchdog_history", [])
        if isinstance(hist, list):
            hist.append(
                f"{time.strftime('%H:%M:%S')} | {role_s}: {float(current_gb):.1f}->{float(new_gb):.1f} GB | "
                f"free={_fmt_bytes(int(free_b))} below {float(threshold_gb):.2f}GB for {observed_s:.2f}s | "
                f"reason={reason} | before={before} | after={after} | {trim_status}"
            )
            del hist[40:]
        ctx["ltx_runtime_hotset_watchdog_status"] = "retuned active hotset for this run only"
        ctx["_ltx_runtime_hotset_watchdog_low_since"] = now
        if echo:
            try:
                print(
                    f"[vram-lab-ltx-watchdog] {role_s}: lowered hotset {float(current_gb):.1f}->{float(new_gb):.1f}GB "
                    f"because driver_free={_fmt_bytes(int(free_b))} stayed below {float(threshold_gb):.2f}GB for {observed_s:.2f}s; {trim_status} | {after}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception as exc:
        try:
            ctx["ltx_runtime_hotset_watchdog_errors"] = f"{type(exc).__name__}: {exc}"
        except Exception:
            pass


def _start_ltx_runtime_hotset_watchdog_poller(
    ctx: Dict[str, Any],
    torch_module: Any | None,
    *,
    echo: bool = False,
    interval_s: float = 0.75,
) -> Any:
    """Poll VRAM during long denoise forwards so Stage 2 can retune mid-step."""
    stop_event = threading.Event()
    try:
        interval = max(0.25, min(2.0, float(interval_s)))
    except Exception:
        interval = 0.75
    ctx["ltx_runtime_hotset_watchdog_poller"] = "ON"
    ctx["ltx_runtime_hotset_watchdog_poller_interval_s"] = f"{interval:.2f}"
    ctx["ltx_runtime_hotset_watchdog_poller_checks"] = "0"

    def _current_runtime() -> Any:
        try:
            rt = ctx.get("_ltx_phase_retention_runtime")
            if rt is not None:
                return rt
            runtimes = ctx.get("_ltx_early_residency_runtimes")
            if isinstance(runtimes, list) and runtimes:
                return runtimes[-1]
        except Exception:
            return None
        return None

    def _current_role() -> str:
        try:
            text = str(ctx.get("last_active_stage_block_limit", "") or "").lower()
            if "stage2" in text or "refine" in text:
                return "stage2_refine_denoise"
            step_events = ctx.get("ltx_stage2_step_events")
            if isinstance(step_events, list) and any("stage2_refine_denoise" in str(x) for x in step_events[-4:]):
                return "stage2_refine_denoise"
        except Exception:
            pass
        return "stage1_initial_denoise"

    def _loop() -> None:
        checks = 0
        while not stop_event.wait(interval):
            runtime = _current_runtime()
            if runtime is None:
                continue
            checks += 1
            ctx["ltx_runtime_hotset_watchdog_poller_checks"] = str(checks)
            try:
                _ltx_runtime_hotset_watchdog_check(
                    ctx,
                    runtime,
                    _current_role(),
                    torch_module=torch_module,
                    echo=echo,
                    low_duration_hint_s=interval,
                    reason="in-step poller",
                )
            except Exception as exc:
                ctx["ltx_runtime_hotset_watchdog_errors"] = f"poller failed: {type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_loop, name="FrameVisionLTXHotsetWatchdog", daemon=True)
    thread.start()

    def _stop() -> None:
        try:
            stop_event.set()
            thread.join(timeout=2.0)
        except Exception:
            pass
        ctx["ltx_runtime_hotset_watchdog_poller"] = "STOPPED"

    return _stop


def _update_ltx_early_residency_runtimes(ctx: Dict[str, Any], detach: bool = False) -> None:
    runtimes = ctx.get("_ltx_early_residency_runtimes") or []
    if not isinstance(runtimes, list):
        return
    for runtime in list(runtimes):
        if runtime is None:
            continue
        try:
            if hasattr(runtime, "update_context"):
                runtime.update_context(ctx)
                _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="attached early to BatchSplitAdapter before first forward")
        except Exception as exc:
            ctx.setdefault("notes", []).append(f"early residency runtime update failed: {type(exc).__name__}: {exc}")
        if detach:
            try:
                if hasattr(runtime, "detach_vram_hooks"):
                    runtime.detach_vram_hooks()
            except Exception as exc:
                ctx.setdefault("notes", []).append(f"early residency runtime detach failed: {type(exc).__name__}: {exc}")



def _retune_ltx_retained_runtime_for_stage2(
    ctx: Dict[str, Any],
    runtime: Any,
    torch_module: Any | None = None,
    echo: bool = False,
) -> None:
    """Retarget the already-attached denoise residency runtime for Stage 2.

    Handoff-safe behavior: do not rebuild/trim the stable hotset at the exact
    Stage-1 -> Stage-2 boundary. The retained transformer can be in a fragile
    half-reused state there. This only applies the Stage-2 budget/fraction,
    performs one simple allocator cleanup, and defers hotset surgery until the
    first Stage-2 forward has completed successfully.
    """
    try:
        if runtime is None:
            ctx["ltx_phase_retention_stage2_runtime_retune"] = "NO: runtime unavailable"
            return
        gb, source = _block_limit_for_role(ctx, "stage2_refine_denoise")
        if gb <= 0.0:
            ctx["ltx_phase_retention_stage2_runtime_retune"] = "NO: no stage2 block limit"
            return
        bytes_value = int(float(gb) * 1024 ** 3)
        try:
            fraction = _ctx_float(ctx, "vram_stage2_stable_hotset_fraction_requested", _ctx_float(ctx, "vram_stage2_stable_hotset_fraction", 0.90))
        except Exception:
            fraction = 0.90
        fraction = max(0.10, min(2.00, float(fraction)))
        before = _cuda_snapshot(torch_module)
        ctx["ltx_phase_retention_handoff_safe_mode"] = "ON"
        ctx["ltx_phase_retention_stage2_deferred_trim"] = "YES: stable hotset plan/trim deferred until after first Stage-2 forward"
        ctx["ltx_phase_retention_stage2_deferred_trim_pending"] = "YES"
        try:
            setattr(runtime, "hot_block_budget_bytes", bytes_value)
            setattr(runtime, "safe_hot_window_gb", float(gb))
            setattr(runtime, "balanced_hot_window_gb", float(gb))
            setattr(runtime, "stable_hotset_fraction", float(fraction))
            setattr(runtime, "stable_hotset_budget_bytes", 0)
            policy = getattr(runtime, "policy", None)
            if isinstance(policy, dict):
                policy["hot_block_budget_bytes"] = bytes_value
                policy["safe_hot_window_gb"] = float(gb)
                policy["safe_hot_window_bytes"] = bytes_value
                policy["balanced_hot_window_gb"] = float(gb)
                policy["balanced_hot_window_bytes"] = bytes_value
                policy["stable_hotset_fraction"] = float(fraction)
                policy["stable_hotset_budget_bytes"] = 0
            ctx["ltx_phase_retention_stage2_handoff_cleanup_before"] = _cuda_snapshot(torch_module)
            try:
                gc.collect()
                if torch_module is not None and hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
                    torch_module.cuda.empty_cache()
                ctx["ltx_phase_retention_stage2_handoff_cleanup"] = "YES: gc.collect + torch.cuda.empty_cache once"
            except Exception as cleanup_exc:
                ctx["ltx_phase_retention_stage2_handoff_cleanup"] = f"FAILED: {type(cleanup_exc).__name__}: {cleanup_exc}"
            ctx["ltx_phase_retention_stage2_handoff_cleanup_after"] = _cuda_snapshot(torch_module)
            if hasattr(runtime, "update_context"):
                runtime.update_context(ctx)
                _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="stage1 hooks reused across two-stage boundary; stage2 trim deferred")
        except Exception as exc:
            ctx["ltx_phase_retention_stage2_runtime_retune"] = f"FAILED: {type(exc).__name__}: {exc}"
            return
        after = _cuda_snapshot(torch_module)
        ctx["ltx_phase_retention_stage2_runtime_retune"] = f"YES-SAFE: {gb:.1f} GB ({source}), fraction {fraction:.2f}; hotset plan/trim deferred"
        ctx["ltx_phase_retention_stage2_retune_before"] = before
        ctx["ltx_phase_retention_stage2_retune_after"] = after
        ctx["active_block_limit_for_stage2_refine_denoise"] = f"{gb:.1f} GB"
        ctx["last_active_stage_block_limit"] = f"stage2_refine_denoise: {gb:.1f} GB ({source})"
        if echo:
            try:
                print(f"[vram-lab-ltx-retain] stage2 safe-retuned to {gb:.1f} GB ({source}), fraction {fraction:.2f}; hotset trim deferred | before: {before} | after: {after}", flush=True)
            except Exception:
                pass
    except Exception as exc:
        ctx["ltx_phase_retention_stage2_runtime_retune"] = f"FAILED: {type(exc).__name__}: {exc}"


def _finish_deferred_ltx_stage2_hotset_trim(
    ctx: Dict[str, Any],
    runtime: Any,
    torch_module: Any | None = None,
    echo: bool = False,
) -> None:
    """Run the deferred Stage-2 stable-hotset plan/trim after Stage 2 has proven it can enter."""
    try:
        if str(ctx.get("ltx_phase_retention_stage2_deferred_trim_pending", "NO")).upper() != "YES":
            return
        if runtime is None:
            ctx["ltx_phase_retention_stage2_deferred_trim"] = "SKIPPED: runtime unavailable"
            ctx["ltx_phase_retention_stage2_deferred_trim_pending"] = "NO"
            return
        before = _cuda_snapshot(torch_module)
        ctx["ltx_phase_retention_stage2_deferred_trim_before"] = before
        try:
            if hasattr(runtime, "_plan_stable_hotset"):
                runtime._plan_stable_hotset()
            if hasattr(runtime, "_trim_hot_blocks"):
                runtime._trim_hot_blocks(keep_name="")
            if hasattr(runtime, "update_context"):
                runtime.update_context(ctx)
                _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="stage2 deferred trim completed after first forward")
            ctx["ltx_phase_retention_stage2_deferred_trim"] = "DONE: after first successful Stage-2 forward"
        except Exception as exc:
            ctx["ltx_phase_retention_stage2_deferred_trim"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_phase_retention_stage2_deferred_trim_after"] = _cuda_snapshot(torch_module)
        ctx["ltx_phase_retention_stage2_deferred_trim_pending"] = "NO"
        if echo:
            try:
                print(f"[vram-lab-ltx-retain] stage2 deferred hotset trim {ctx.get('ltx_phase_retention_stage2_deferred_trim')} | before: {before} | after: {ctx.get('ltx_phase_retention_stage2_deferred_trim_after')}", flush=True)
            except Exception:
                pass
    except Exception as exc:
        ctx["ltx_phase_retention_stage2_deferred_trim"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_phase_retention_stage2_deferred_trim_pending"] = "NO"


def _install_ltx_two_stage_phase_retention_bridge(
    ctx: Dict[str, Any],
    torch_module: Any | None = None,
    echo: bool = False,
) -> None:
    """Keep the first two-stage transformer context alive through Stage 2.

    This is a wrapper-side lifecycle bridge. It does not edit the LTX repo and
    does not alter the reusable VRAM Lab core. The official DiffusionStage
    normally enters a model context for each stage and releases it when that
    call returns. For the full distilled FrameVision path, that means Stage 1's
    working hotset is thrown away before Stage 2. This bridge holds the first
    transformer context open, reuses that transformer for the second diffusion
    call, then releases it normally after the second call.
    """
    ctx["ltx_phase_retention_bridge_installed"] = "NO"
    ctx["ltx_phase_retention_bridge_calls"] = "0"
    ctx["ltx_phase_retention_bridge_reused_stage2"] = "NO"
    ctx["ltx_phase_retention_bridge_errors"] = "none"
    ctx["ltx_phase_retention_stage2_runtime_retune"] = "not attempted"
    ctx["ltx_phase_retention_handoff_safe_mode"] = "not attempted"
    ctx["ltx_phase_retention_stage2_deferred_trim"] = "not attempted"
    ctx["ltx_phase_retention_stage2_deferred_trim_pending"] = "NO"
    ctx["ltx_phase_retention_stage2_handoff_cleanup"] = "not attempted"
    try:
        pipeline_name = str(ctx.get("selected_pipeline", "") or "").strip()
        if pipeline_name not in {"two_stages", "two_stages_hq", "a2vid_two_stage"}:
            ctx["ltx_phase_retention_bridge_installed"] = "NO: selected pipeline is not two-stage"
            return
        blocks_mod = importlib.import_module("ltx_pipelines.utils.blocks")
        cls = getattr(blocks_mod, "DiffusionStage", None)
        if cls is None:
            raise AttributeError("ltx_pipelines.utils.blocks.DiffusionStage not found")
        original_call = getattr(cls, "__call__")
        if getattr(original_call, "_framevision_ltx_phase_retention_bridge", False):
            ctx["ltx_phase_retention_bridge_installed"] = "YES: already installed"
            return
        import inspect
        signature = inspect.signature(original_call)
        state: Dict[str, Any] = {
            "call_count": 0,
            "ctx_manager": None,
            "transformer": None,
            "released": False,
            "stage1_runtime": None,
        }

        def _build_video_tools(bound: Any) -> Any:
            try:
                video = bound.arguments.get("video")
                if video is None:
                    return None
                width = int(bound.arguments.get("width"))
                height = int(bound.arguments.get("height"))
                frames = int(bound.arguments.get("frames"))
                fps = float(bound.arguments.get("fps"))
                pixel_shape = blocks_mod.VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
                v_shape = blocks_mod.VideoLatentShape.from_pixel_shape(pixel_shape)
                return blocks_mod.VideoLatentTools(blocks_mod.VideoLatentPatchifier(patch_size=1), v_shape, fps)
            except Exception:
                return None

        def _run_with_bound(stage_self: Any, transformer: Any, bound: Any) -> Any:
            return stage_self.run(
                transformer,
                bound.arguments.get("denoiser"),
                bound.arguments.get("sigmas"),
                bound.arguments.get("noiser"),
                bound.arguments.get("width"),
                bound.arguments.get("height"),
                bound.arguments.get("frames"),
                bound.arguments.get("fps"),
                bound.arguments.get("video"),
                bound.arguments.get("audio"),
                bound.arguments.get("stepper"),
                bound.arguments.get("loop"),
                bound.arguments.get("max_batch_size"),
            )

        def _release_retained(label: str) -> None:
            manager = state.get("ctx_manager")
            if manager is None or state.get("released"):
                return
            before = _cuda_snapshot(torch_module)
            try:
                manager.__exit__(None, None, None)
                state["released"] = True
                ctx["ltx_phase_retention_release"] = f"YES: {label}"
                ctx["ltx_phase_retention_release_before"] = before
                ctx["ltx_phase_retention_release_after"] = _cuda_snapshot(torch_module)
            except Exception as exc:
                ctx["ltx_phase_retention_bridge_errors"] = f"release failed: {type(exc).__name__}: {exc}"

        def retained_call(stage_self: Any, *args: Any, **kwargs: Any) -> Any:
            call_index = int(state.get("call_count", 0) or 0) + 1
            state["call_count"] = call_index
            ctx["ltx_phase_retention_bridge_calls"] = str(call_index)
            bound = signature.bind(stage_self, *args, **kwargs)
            bound.apply_defaults()
            if call_index == 1:
                video_tools = _build_video_tools(bound)
                manager = stage_self._transformer_ctx(video_tools=video_tools)
                state["ctx_manager"] = manager
                before = _cuda_snapshot(torch_module)
                transformer = manager.__enter__()
                state["transformer"] = transformer
                try:
                    setattr(transformer, "_framevision_ltx_retained_for_stage2", True)
                except Exception:
                    pass
                ctx["ltx_phase_retention_stage1_context"] = "OPEN"
                ctx["ltx_phase_retention_stage1_enter_before"] = before
                ctx["ltx_phase_retention_stage1_enter_after"] = _cuda_snapshot(torch_module)
                try:
                    return _run_with_bound(stage_self, transformer, bound)
                except Exception:
                    _release_retained("stage1 exception")
                    raise
            if call_index == 2 and state.get("transformer") is not None and not state.get("released"):
                transformer = state.get("transformer")
                ctx["ltx_phase_retention_bridge_reused_stage2"] = "YES"
                ctx["ltx_phase_retention_stage2_reuse_before"] = _cuda_snapshot(torch_module)
                runtime = state.get("stage1_runtime") or ctx.get("_ltx_phase_retention_runtime")
                if runtime is None:
                    runtimes = ctx.get("_ltx_early_residency_runtimes") or []
                    if isinstance(runtimes, list) and runtimes:
                        runtime = runtimes[0]
                state["stage1_runtime"] = runtime
                _retune_ltx_retained_runtime_for_stage2(ctx, runtime, torch_module=torch_module, echo=echo)
                try:
                    result = _run_with_bound(stage_self, transformer, bound)
                    ctx["ltx_phase_retention_stage2_reuse_after"] = _cuda_snapshot(torch_module)
                    return result
                finally:
                    _release_retained("after stage2")
            return original_call(stage_self, *args, **kwargs)

        setattr(retained_call, "_framevision_ltx_phase_retention_bridge", True)
        setattr(retained_call, "_framevision_ltx_phase_retention_original", original_call)
        setattr(cls, "__call__", retained_call)
        ctx["ltx_phase_retention_bridge_installed"] = "YES"
        ctx.setdefault("notes", []).append(
            "Installed LTX two-stage phase-retention bridge: keeps the first transformer context alive through the second diffusion phase, uses handoff-safe Stage-2 retune with deferred hotset trim, then releases it after Stage 2."
        )
    except Exception as exc:
        ctx["ltx_phase_retention_bridge_installed"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_phase_retention_bridge_errors"] = f"{type(exc).__name__}: {exc}"

def _safe_int_ctx(ctx: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(str(ctx.get(key, default)).strip())
    except Exception:
        return int(default)


def _summarize_value_shape(value: Any, prefix: str = "arg", limit: int = 10, depth: int = 0) -> List[str]:
    """Return compact tensor/array shape summaries without importing heavy libs.

    Kept intentionally defensive: this runs inside the generation process and
    must never break official LTX execution. It captures enough to answer the
    important question for the two-stage pass: what shape/size did the second
    denoise/refinement step actually receive?
    """
    out: List[str] = []
    if len(out) >= limit:
        return out
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                shape_s = "x".join(str(int(x)) for x in list(shape))
            except Exception:
                shape_s = str(shape)
            dtype = getattr(value, "dtype", None)
            device = getattr(value, "device", None)
            out.append(f"{prefix}:shape={shape_s};dtype={dtype};device={device}")
            return out
    except Exception:
        pass
    if depth >= 2:
        return out
    try:
        if isinstance(value, dict):
            for k, v in list(value.items())[:8]:
                out.extend(_summarize_value_shape(v, f"{prefix}.{k}", limit, depth + 1))
                if len(out) >= limit:
                    break
            return out[:limit]
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(list(value)[:8]):
                out.extend(_summarize_value_shape(v, f"{prefix}[{i}]", limit, depth + 1))
                if len(out) >= limit:
                    break
            return out[:limit]
    except Exception:
        return out[:limit]
    return out[:limit]


def _ltx_stage2_step_event(
    ctx: Dict[str, Any],
    label: str,
    detail: str = "",
    torch_module: Any | None = None,
    echo: bool = False,
) -> None:
    try:
        t0 = float(ctx.setdefault("_ltx_stage2_step_t0", time.perf_counter()))
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = 0.0
    try:
        cuda = _cuda_snapshot(torch_module)
    except Exception:
        cuda = "n/a"
    line = f"{elapsed:9.3f}s | {label}: {detail} | {cuda}"
    try:
        events = ctx.setdefault("ltx_stage2_step_events", [])
        if isinstance(events, list) and len(events) < 500:
            events.append(line)
        ctx["ltx_stage2_step_event_count"] = str(len(events) if isinstance(events, list) else 0)
    except Exception:
        pass
    if echo:
        try:
            print(f"[vram-lab-ltx-stage2] {line}", flush=True)
        except Exception:
            pass


def _ltx_stage2_step_summarize(ctx: Dict[str, Any]) -> None:
    try:
        adapters = ctx.get("_ltx_stage2_step_adapters") or {}
        if not isinstance(adapters, dict):
            return
        parts: List[str] = []
        stage2_s = 0.0
        stage2_calls = 0
        for key in sorted(adapters.keys(), key=lambda x: int(x) if str(x).isdigit() else 9999):
            item = adapters.get(key) or {}
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", f"adapter_{key}"))
            calls = int(item.get("calls", 0) or 0)
            total = float(item.get("total_s", 0.0) or 0.0)
            avg = (total / calls) if calls else 0.0
            load_delta = int(item.get("block_load_delta", 0) or 0)
            unload_delta = int(item.get("block_unload_delta", 0) or 0)
            flash_ok = int(item.get("flash_success_delta", 0) or 0)
            flash_fb = int(item.get("flash_fallback_delta", 0) or 0)
            last_shape = str(item.get("last_shapes", "n/a"))
            parts.append(
                f"{role}: calls={calls}, total={total:.3f}s, avg={avg:.3f}s, "
                f"block_load_delta={load_delta}, block_unload_delta={unload_delta}, "
                f"flash_ok_delta={flash_ok}, flash_fallback_delta={flash_fb}, last_shapes={last_shape}"
            )
            if int(key) == 2:
                stage2_s = total
                stage2_calls = calls
        ctx["ltx_stage2_step_summary"] = " | ".join(parts) if parts else "none"
        ctx["ltx_stage2_refine_total_s"] = f"{stage2_s:.3f}"
        ctx["ltx_stage2_refine_step_count"] = str(stage2_calls)
        ctx["ltx_stage2_refine_avg_step_s"] = f"{(stage2_s / stage2_calls):.3f}" if stage2_calls else "0.000"
    except Exception as exc:
        ctx["ltx_stage2_step_profiler_errors"] = f"summary failed: {type(exc).__name__}: {exc}"



def _ltx_stage1_churn_block_snapshot(runtime: Any) -> List[Dict[str, Any]]:
    """Best-effort, wrapper-local snapshot of VRAM Lab block residency.

    Diagnostic-only. Uses reflection so the reusable VRAM Lab core stays clean.
    """
    out: List[Dict[str, Any]] = []

    def _get_attr(obj: Any, names: List[str], default: Any = None) -> Any:
        for name in names:
            try:
                if isinstance(obj, dict) and name in obj:
                    return obj.get(name)
                if hasattr(obj, name):
                    return getattr(obj, name)
            except Exception:
                pass
        return default

    def _module_of(obj: Any) -> Any:
        mod = _get_attr(obj, ["module", "block", "model", "target", "target_module", "mod"], None)
        if mod is not None:
            return mod
        try:
            if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")):
                return obj
        except Exception:
            pass
        return None

    def _device_of(obj: Any) -> str:
        raw = _get_attr(obj, ["device", "current_device", "resident_device", "last_device", "placement"], None)
        if raw is not None:
            text = str(raw)
            if text and text.lower() not in {"none", "unknown"}:
                return text
        mod = _module_of(obj)
        if mod is not None:
            try:
                for param in mod.parameters(recurse=True):
                    return str(getattr(param, "device", "unknown"))
            except Exception:
                pass
            try:
                for buf in mod.buffers(recurse=True):
                    return str(getattr(buf, "device", "unknown"))
            except Exception:
                pass
        return "unknown"

    def _bytes_of(obj: Any) -> int:
        for name in ["num_bytes", "nbytes", "size_bytes", "bytes", "param_bytes", "weight_bytes", "estimated_bytes"]:
            try:
                val = _get_attr(obj, [name], None)
                if val is not None:
                    return int(float(val))
            except Exception:
                pass
        mod = _module_of(obj)
        total = 0
        if mod is not None:
            try:
                for param in mod.parameters(recurse=True):
                    total += int(param.numel()) * int(param.element_size())
            except Exception:
                pass
        return int(total)

    def _stable_set() -> set:
        vals: List[Any] = []
        for name in ["stable_hotset", "stable_hotset_blocks", "hotset", "hot_blocks", "resident_hotset", "planned_hotset", "stable_block_indices", "hot_block_indices"]:
            try:
                val = getattr(runtime, name, None)
            except Exception:
                val = None
            if val is None:
                continue
            try:
                if isinstance(val, dict):
                    vals.extend(list(val.keys()))
                elif isinstance(val, (set, list, tuple)):
                    vals.extend(list(val))
            except Exception:
                pass
        return {str(v) for v in vals}

    try:
        blocks = getattr(runtime, "blocks", []) or []
    except Exception:
        blocks = []
    hotset = _stable_set()
    try:
        iterable = list(blocks)
    except Exception:
        iterable = []
    for idx, block in enumerate(iterable):
        name = _get_attr(block, ["name", "path", "module_name", "qualified_name", "fqn", "key"], None)
        if not name:
            name = f"block_{idx:03d}"
        dev = _device_of(block)
        name_s = str(name)
        idx_s = str(idx)
        stable = (name_s in hotset) or (idx_s in hotset)
        if not stable:
            try:
                stable = bool(_get_attr(block, ["stable", "is_stable", "is_hot", "in_hotset"], False))
            except Exception:
                stable = False
        out.append({
            "idx": idx,
            "name": name_s,
            "device": dev,
            "is_cuda": str(dev).lower().startswith("cuda"),
            "is_meta": str(dev).lower().startswith("meta"),
            "stable": bool(stable),
            "bytes": _bytes_of(block),
        })
    return out


def _ltx_stage1_churn_record(ctx: Dict[str, Any], runtime: Any, role: str, call_no: int, when: str, snap: List[Dict[str, Any]]) -> None:
    if str(role) != "stage1_initial_denoise":
        return
    try:
        state = ctx.setdefault("_ltx_stage1_churn_state", {})
        if not isinstance(state, dict):
            return
        call_key = str(int(call_no))
        call_state = state.setdefault(call_key, {})
        call_state[str(when)] = snap
        ctx["ltx_stage1_churn_profiler_installed"] = "YES"
        ctx["ltx_stage1_churn_profiler_targets"] = "BatchSplitAdapter.forward / runtime.blocks snapshot"
        ctx["ltx_stage1_churn_profiler_errors"] = "none"
    except Exception as exc:
        ctx["ltx_stage1_churn_profiler_errors"] = f"record failed: {type(exc).__name__}: {exc}"


def _ltx_stage1_churn_summarize(ctx: Dict[str, Any]) -> None:
    try:
        state = ctx.get("_ltx_stage1_churn_state") or {}
        if not isinstance(state, dict) or not state:
            ctx.setdefault("ltx_stage1_churn_summary", "none")
            return
        parts: List[str] = []
        hot_after_counts: Dict[str, int] = {}
        stable_names: Dict[str, int] = {}
        bytes_by_name: Dict[str, int] = {}
        for key in sorted(state.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            item = state.get(key) or {}
            before = item.get("before") or []
            after = item.get("after") or []
            bmap = {str(x.get("name")): x for x in before if isinstance(x, dict)}
            amap = {str(x.get("name")): x for x in after if isinstance(x, dict)}
            before_cuda = sum(1 for x in bmap.values() if x.get("is_cuda"))
            after_cuda = sum(1 for x in amap.values() if x.get("is_cuda"))
            before_stable_cuda = sum(1 for x in bmap.values() if x.get("stable") and x.get("is_cuda"))
            after_stable_cuda = sum(1 for x in amap.values() if x.get("stable") and x.get("is_cuda"))
            visible_loaded = [name for name, aval in amap.items() if aval.get("is_cuda") and not bool((bmap.get(name) or {}).get("is_cuda"))]
            visible_unloaded = [name for name, bval in bmap.items() if bval.get("is_cuda") and not bool((amap.get(name) or {}).get("is_cuda"))]
            for name, aval in amap.items():
                try:
                    bytes_by_name[name] = max(int(bytes_by_name.get(name, 0)), int(aval.get("bytes", 0) or 0))
                except Exception:
                    pass
                if aval.get("is_cuda"):
                    hot_after_counts[name] = int(hot_after_counts.get(name, 0)) + 1
                if aval.get("stable"):
                    stable_names[name] = int(stable_names.get(name, 0)) + 1
            parts.append(
                f"call {key}: before_cuda={before_cuda}, after_cuda={after_cuda}, "
                f"before_stable_cuda={before_stable_cuda}, after_stable_cuda={after_stable_cuda}, "
                f"visible_loaded={len(visible_loaded)}, visible_unloaded={len(visible_unloaded)}"
            )
        ctx["ltx_stage1_churn_summary"] = " | ".join(parts)
        total_calls = len(state)
        always_cuda = [name for name, n in hot_after_counts.items() if n >= total_calls]
        often_cuda = sorted(hot_after_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
        stable_list = sorted(stable_names.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
        ctx["ltx_stage1_churn_always_cuda_after_blocks"] = ", ".join(always_cuda[:24]) if always_cuda else "none"
        ctx["ltx_stage1_churn_top_cuda_after_blocks"] = ", ".join([f"{name}({count}/{total_calls})" for name, count in often_cuda]) if often_cuda else "none"
        ctx["ltx_stage1_churn_stable_candidate_blocks"] = ", ".join([f"{name}({count}/{total_calls})" for name, count in stable_list]) if stable_list else "none"
        try:
            total_cuda_bytes = sum(int(bytes_by_name.get(name, 0) or 0) for name in always_cuda)
            ctx["ltx_stage1_churn_always_cuda_after_bytes"] = _fmt_bytes(total_cuda_bytes)
        except Exception:
            ctx["ltx_stage1_churn_always_cuda_after_bytes"] = "n/a"
    except Exception as exc:
        ctx["ltx_stage1_churn_profiler_errors"] = f"summary failed: {type(exc).__name__}: {exc}"

def _ltx_stage1_move_profiler_start(ctx: Dict[str, Any], torch_module: Any | None, role: str, call_no: int) -> Any:
    """Temporarily trace real transformer block .to(cuda/cpu/meta) moves during Stage 1.

    The first Stage-1 churn probe tried to read runtime.blocks directly, but the
    active runtime does not expose that list in this wrapper path. This tracer is
    still wrapper-local and generic-clean: it observes only official Module.to()
    calls during the Stage-1 BatchSplitAdapter.forward call and records movement
    for transformer BasicAVTransformerBlock objects.
    """
    if str(role) != "stage1_initial_denoise":
        return None
    try:
        if torch_module is None or not hasattr(torch_module, "nn"):
            return None
        module_cls = getattr(getattr(torch_module, "nn", None), "Module", None)
        if module_cls is None:
            return None
        original_to = getattr(module_cls, "to", None)
        if original_to is None or getattr(original_to, "_framevision_stage1_move_profiler", False):
            return None

        ctx["ltx_stage1_move_profiler_installed"] = "YES"
        ctx["ltx_stage1_move_profiler_targets"] = "torch.nn.Module.to during Stage-1 BatchSplitAdapter.forward"
        ctx["ltx_stage1_move_profiler_errors"] = "none"

        # Diagnostic only: do not retain or block any movement here.
        # The previous one-block tail-retain test was unsafe because it
        # intercepted Module.to(cpu) after the runtime had already decided to
        # unload. This probe only records the callsite that asks blocks to move,
        # so the real unload decision can be found without changing behavior.
        ctx["ltx_stage1_tail_retain_installed"] = "NO: disabled after native crash"
        ctx["ltx_stage1_tail_retain_blocks"] = "none"
        ctx["ltx_stage1_tail_retain_release"] = "n/a"
        ctx["ltx_stage1_tail_retain_skipped_cpu_moves"] = "0"

        label_by_id = ctx.setdefault("_ltx_stage1_move_label_by_id", {})
        calls = ctx.setdefault("_ltx_stage1_move_calls", {})
        if not isinstance(label_by_id, dict) or not isinstance(calls, dict):
            return None
        call_key = str(int(call_no))
        call_state = calls.setdefault(call_key, {"cuda": {}, "cpu": {}, "meta": {}, "other": {}, "classes": {}})

        def _target_from_args(args: Any, kwargs: Any) -> str:
            try:
                if isinstance(kwargs, dict):
                    for k in ("device",):
                        if k in kwargs and kwargs.get(k) is not None:
                            return str(kwargs.get(k))
                if args:
                    first = args[0]
                    # dtype-only calls should not look like device movement.
                    if str(first).startswith("torch.") and "dtype" in str(first).lower():
                        return "dtype"
                    return str(first)
            except Exception:
                pass
            return "unknown"

        def _is_block(mod: Any) -> bool:
            try:
                cls = f"{type(mod).__module__}.{type(mod).__name__}"
                return cls.endswith("BasicAVTransformerBlock") or cls.endswith(".BasicAVTransformerBlock")
            except Exception:
                return False

        def _label(mod: Any) -> str:
            mid = str(id(mod))
            val = label_by_id.get(mid)
            if val:
                return str(val)
            try:
                idx = len(label_by_id)
            except Exception:
                idx = 0
            val = f"block_{idx:02d}"
            label_by_id[mid] = val
            return val

        def _bump(bucket: Dict[str, Any], key: str) -> None:
            try:
                bucket[key] = int(bucket.get(key, 0) or 0) + 1
            except Exception:
                bucket[key] = 1

        def _caller_key() -> str:
            try:
                frames = traceback.extract_stack(limit=16)[:-2]
                # Prefer the first useful non-wrapper frame near the callsite.
                for fr in reversed(frames):
                    filename = str(fr.filename).replace("\\", "/")
                    func = str(fr.name)
                    if "ltx23_vram_lab_cli.py" in filename:
                        continue
                    if "torch/nn/modules/module.py" in filename and func in {"to", "_apply"}:
                        continue
                    return f"{filename.rsplit('/', 1)[-1]}:{func}:{fr.lineno}"
                if frames:
                    fr = frames[-1]
                    filename = str(fr.filename).replace("\\", "/")
                    return f"{filename.rsplit('/', 1)[-1]}:{fr.name}:{fr.lineno}"
            except Exception:
                pass
            return "unknown"

        def wrapped_to(self: Any, *args: Any, **kwargs: Any) -> Any:
            target = _target_from_args(args, kwargs)
            if _is_block(self):
                try:
                    name = _label(self)
                    cls_name = f"{type(self).__module__}.{type(self).__name__}"
                    if isinstance(call_state.get("classes"), dict):
                        call_state["classes"][name] = cls_name
                    low = str(target).lower()
                    if low.startswith("cuda"):
                        _bump(call_state.setdefault("cuda", {}), name)
                        _bump(call_state.setdefault("cuda_callsite", {}), _caller_key())
                    elif low.startswith("cpu"):
                        _bump(call_state.setdefault("cpu", {}), name)
                        _bump(call_state.setdefault("cpu_callsite", {}), _caller_key())
                    elif low.startswith("meta"):
                        _bump(call_state.setdefault("meta", {}), name)
                        _bump(call_state.setdefault("meta_callsite", {}), _caller_key())
                    else:
                        _bump(call_state.setdefault("other", {}), name + ":" + str(target))
                except Exception as exc:
                    ctx["ltx_stage1_move_profiler_errors"] = f"record failed: {type(exc).__name__}: {exc}"
            return original_to(self, *args, **kwargs)

        try:
            setattr(wrapped_to, "_framevision_stage1_move_profiler", True)
            setattr(wrapped_to, "_framevision_stage1_move_profiler_original", original_to)
        except Exception:
            pass
        setattr(module_cls, "to", wrapped_to)
        return (module_cls, original_to)
    except Exception as exc:
        ctx["ltx_stage1_move_profiler_errors"] = f"install failed: {type(exc).__name__}: {exc}"
        return None


def _ltx_stage1_move_profiler_stop(ctx: Dict[str, Any], token: Any) -> None:
    try:
        if not token:
            return
        module_cls, original_to = token
        if module_cls is not None and original_to is not None:
            setattr(module_cls, "to", original_to)
    except Exception as exc:
        ctx["ltx_stage1_move_profiler_errors"] = f"restore failed: {type(exc).__name__}: {exc}"


def _ltx_stage1_move_profiler_summarize(ctx: Dict[str, Any]) -> None:
    try:
        calls = ctx.get("_ltx_stage1_move_calls") or {}
        if not isinstance(calls, dict) or not calls:
            ctx.setdefault("ltx_stage1_move_summary", "none")
            return
        parts: List[str] = []
        cuda_seen: Dict[str, int] = {}
        cpu_seen: Dict[str, int] = {}
        meta_seen: Dict[str, int] = {}
        total_cuda: Dict[str, int] = {}
        total_cpu: Dict[str, int] = {}
        call_count = 0
        for key in sorted(calls.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            item = calls.get(key) or {}
            if not isinstance(item, dict):
                continue
            call_count += 1
            cuda = item.get("cuda") or {}
            cpu = item.get("cpu") or {}
            meta = item.get("meta") or {}
            cuda_callsite = item.get("cuda_callsite") or {}
            cpu_callsite = item.get("cpu_callsite") or {}
            meta_callsite = item.get("meta_callsite") or {}
            if not isinstance(cuda, dict):
                cuda = {}
            if not isinstance(cpu, dict):
                cpu = {}
            if not isinstance(meta, dict):
                meta = {}
            if not isinstance(cuda_callsite, dict):
                cuda_callsite = {}
            if not isinstance(cpu_callsite, dict):
                cpu_callsite = {}
            if not isinstance(meta_callsite, dict):
                meta_callsite = {}
            for name, cnt in cuda.items():
                cuda_seen[str(name)] = cuda_seen.get(str(name), 0) + 1
                total_cuda[str(name)] = total_cuda.get(str(name), 0) + int(cnt or 0)
            for name, cnt in cpu.items():
                cpu_seen[str(name)] = cpu_seen.get(str(name), 0) + 1
                total_cpu[str(name)] = total_cpu.get(str(name), 0) + int(cnt or 0)
            for key, cnt in cuda_callsite.items():
                total_cuda_callsite[str(key)] = total_cuda_callsite.get(str(key), 0) + int(cnt or 0)
            for key, cnt in cpu_callsite.items():
                total_cpu_callsite[str(key)] = total_cpu_callsite.get(str(key), 0) + int(cnt or 0)
            for key, cnt in meta_callsite.items():
                total_meta_callsite[str(key)] = total_meta_callsite.get(str(key), 0) + int(cnt or 0)
            for name in meta.keys():
                meta_seen[str(name)] = meta_seen.get(str(name), 0) + 1
            parts.append(
                f"call {key}: cuda_unique={len(cuda)}, cpu_unique={len(cpu)}, meta_unique={len(meta)}, "
                f"cuda_moves={sum(int(v or 0) for v in cuda.values())}, cpu_moves={sum(int(v or 0) for v in cpu.values())}"
            )
        all_cuda = [name for name, n in cuda_seen.items() if n == call_count]
        all_cpu = [name for name, n in cpu_seen.items() if n == call_count]
        churn = sorted(set(all_cuda) & set(all_cpu), key=lambda x: int(x.split("_")[-1]) if x.rsplit("_", 1)[-1].isdigit() else 999)
        top_cuda = sorted(total_cuda.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
        top_cpu = sorted(total_cpu.items(), key=lambda kv: (-kv[1], kv[0]))[:24]
        ctx["ltx_stage1_move_summary"] = " | ".join(parts) if parts else "none"
        ctx["ltx_stage1_move_always_cuda_blocks"] = ", ".join(all_cuda[:80]) if all_cuda else "none"
        ctx["ltx_stage1_move_always_cpu_blocks"] = ", ".join(all_cpu[:80]) if all_cpu else "none"
        ctx["ltx_stage1_move_always_churn_blocks"] = ", ".join(churn[:80]) if churn else "none"
        ctx["ltx_stage1_move_top_cuda_blocks"] = ", ".join(f"{k}={v}" for k, v in top_cuda) if top_cuda else "none"
        ctx["ltx_stage1_move_top_cpu_blocks"] = ", ".join(f"{k}={v}" for k, v in top_cpu) if top_cpu else "none"
        top_cuda_sites = sorted(total_cuda_callsite.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        top_cpu_sites = sorted(total_cpu_callsite.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        top_meta_sites = sorted(total_meta_callsite.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        ctx["ltx_stage1_move_cuda_call_sites"] = " | ".join(f"{k}={v}" for k, v in top_cuda_sites) if top_cuda_sites else "none"
        ctx["ltx_stage1_move_cpu_call_sites"] = " | ".join(f"{k}={v}" for k, v in top_cpu_sites) if top_cpu_sites else "none"
        ctx["ltx_stage1_move_meta_call_sites"] = " | ".join(f"{k}={v}" for k, v in top_meta_sites) if top_meta_sites else "none"
    except Exception as exc:
        ctx["ltx_stage1_move_profiler_errors"] = f"summary failed: {type(exc).__name__}: {exc}"


def _install_ltx_stage2_step_profiler_on_adapter(
    ctx: Dict[str, Any],
    adapter: Any,
    runtime: Any,
    torch_module: Any | None,
    echo: bool = False,
) -> None:
    """Wrap a BatchSplitAdapter instance to time stage-1 and stage-2 denoise calls.

    The earlier LoRA profiler proved the cache removes the long LoRA-fusion gap.
    The remaining unexplained zone is between the cached stage-2 transformer being
    ready and the final decoder starting. In two-stage runs this should be the
    extra/refinement denoise pass. BatchSplitAdapter is the stable boundary we
    already hook for VRAM Lab, so timing its forward calls gives per-pass step
    timing, shape summaries, block churn deltas, and Flash2 deltas without
    touching official LTX code or UI.
    """
    try:
        if getattr(adapter, "_framevision_ltx_stage2_step_profiler", False):
            return
        original_forward = getattr(adapter, "forward", None)
        if original_forward is None:
            raise AttributeError("BatchSplitAdapter.forward not found")
        count = int(ctx.get("_ltx_stage2_step_adapter_count", 0) or 0) + 1
        ctx["_ltx_stage2_step_adapter_count"] = count
        role = "stage1_initial_denoise" if count == 1 else "stage2_refine_denoise" if count == 2 else f"adapter_{count}"
        adapters = ctx.setdefault("_ltx_stage2_step_adapters", {})
        if isinstance(adapters, dict):
            adapters[str(count)] = {
                "role": role,
                "calls": 0,
                "total_s": 0.0,
                "block_load_delta": 0,
                "block_unload_delta": 0,
                "flash_success_delta": 0,
                "flash_fallback_delta": 0,
                "last_shapes": "n/a",
            }
        ctx["ltx_stage2_step_profiler_installed"] = "YES"
        ctx["ltx_stage2_step_profiler_targets"] = "BatchSplitAdapter.forward"
        # Stability baseline: the old Stage-1 churn/move probes are deliberately
        # not installed here. They wrapped/sampled Stage-1 movement and made the
        # Stage-1 -> Stage-2 handoff harder to reason about. Keep only the
        # stage timing/block-delta profiler below.
        ctx.setdefault("ltx_stage1_churn_profiler_installed", "NO: stability baseline")
        ctx.setdefault("ltx_stage1_churn_profiler_targets", "disabled")
        ctx.setdefault("ltx_stage1_churn_profiler_errors", "none")
        ctx.setdefault("ltx_stage1_move_profiler_installed", "NO: stability baseline")
        ctx.setdefault("ltx_stage1_move_profiler_targets", "disabled")
        ctx.setdefault("ltx_stage1_move_profiler_errors", "none")
        ctx.setdefault("ltx_stage1_tail_retain_installed", "NO: disabled after native crash")
        ctx.setdefault("notes", []).append(
            "Installed stage-2 step profiler: times BatchSplitAdapter forward calls, records tensor shapes, block churn deltas, and Flash2 deltas. Diagnostic only; no UI or generation changes."
        )
        _apply_runtime_block_limit_for_role(ctx, runtime, role, torch_module, echo)
        active_limit, active_source = _block_limit_for_role(ctx, role)
        _ltx_stage2_step_event(ctx, "adapter_attached", f"adapter_index={count}; role={role}; active_limit={active_limit:.1f}GB; limit_source={active_source}; class={type(adapter).__module__}.{type(adapter).__name__}", torch_module, echo)

        def _fv_stage2_forward_wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
            item = None
            try:
                adapters2 = ctx.get("_ltx_stage2_step_adapters") or {}
                if isinstance(adapters2, dict):
                    item = adapters2.get(str(count))
            except Exception:
                item = None
            if not isinstance(item, dict):
                item = {"role": role, "calls": 0, "total_s": 0.0}
            call_no = int(item.get("calls", 0) or 0) + 1
            shapes: List[str] = []
            try:
                for i, val in enumerate(list(f_args)[:6]):
                    shapes.extend(_summarize_value_shape(val, f"arg{i}", limit=12))
                    if len(shapes) >= 12:
                        break
                for k, val in list(f_kwargs.items())[:8]:
                    shapes.extend(_summarize_value_shape(val, f"kw.{k}", limit=12))
                    if len(shapes) >= 12:
                        break
            except Exception as exc:
                shapes = [f"shape_summary_failed={type(exc).__name__}: {exc}"]
            shape_s = " | ".join(shapes[:12]) if shapes else "no tensor/array shapes found"
            # Stage-1 churn snapshot probe disabled for clean stability baseline.
            before_load = _safe_int_ctx(ctx, "vram_block_load_count", _safe_int_ctx(ctx, "block_load_count", 0))
            before_unload = _safe_int_ctx(ctx, "vram_block_unload_count", _safe_int_ctx(ctx, "block_unload_count", 0))
            before_flash_ok = _safe_int_ctx(ctx, "ltx_flash2_success_calls", 0)
            before_flash_fb = _safe_int_ctx(ctx, "ltx_flash2_fallback_calls", 0)
            _ltx_stage2_step_event(ctx, "forward:start", f"adapter={count}; role={role}; call={call_no}; shapes={shape_s}", torch_module, echo)
            t0 = time.perf_counter()
            failed = None
            try:
                return original_forward(*f_args, **f_kwargs)
            except Exception as exc:
                failed = exc
                raise
            finally:
                dt = time.perf_counter() - t0
                try:
                    if failed is None and str(role) == "stage2_refine_denoise" and call_no == 1:
                        _finish_deferred_ltx_stage2_hotset_trim(ctx, runtime, torch_module=torch_module, echo=echo)
                except Exception as exc:
                    ctx.setdefault("notes", []).append(f"stage2 deferred trim hook failed: {type(exc).__name__}: {exc}")
                try:
                    if runtime is not None and hasattr(runtime, "update_context"):
                        runtime.update_context(ctx)
                        _copy_vram_runtime_fields(ctx, attached_name="BatchSplitAdapterEarly", attached_status="attached early to BatchSplitAdapter before first forward")
                except Exception as exc:
                    ctx.setdefault("notes", []).append(f"stage2 profiler runtime update failed: {type(exc).__name__}: {exc}")
                if failed is None:
                    try:
                        _ltx_runtime_hotset_watchdog_check(
                            ctx,
                            runtime,
                            role,
                            torch_module=torch_module,
                            echo=echo,
                            low_duration_hint_s=dt,
                            reason=f"after {role} call {call_no}",
                        )
                    except Exception as exc:
                        ctx["ltx_runtime_hotset_watchdog_errors"] = f"check failed: {type(exc).__name__}: {exc}"
                # Stage-1 churn/move summaries disabled for clean stability baseline.
                after_load = _safe_int_ctx(ctx, "vram_block_load_count", _safe_int_ctx(ctx, "block_load_count", before_load))
                after_unload = _safe_int_ctx(ctx, "vram_block_unload_count", _safe_int_ctx(ctx, "block_unload_count", before_unload))
                after_flash_ok = _safe_int_ctx(ctx, "ltx_flash2_success_calls", before_flash_ok)
                after_flash_fb = _safe_int_ctx(ctx, "ltx_flash2_fallback_calls", before_flash_fb)
                load_delta = max(0, after_load - before_load)
                unload_delta = max(0, after_unload - before_unload)
                flash_ok_delta = max(0, after_flash_ok - before_flash_ok)
                flash_fb_delta = max(0, after_flash_fb - before_flash_fb)
                try:
                    block_count_for_repair = int(ctx.get("vram_hooked_block_count_int") or ctx.get("vram_hooked_block_count") or 0)
                except Exception:
                    block_count_for_repair = 0
                if failed is None and str(role) == "stage1_initial_denoise" and block_count_for_repair > 0:
                    if load_delta >= max(1, block_count_for_repair - 1) and unload_delta >= max(1, block_count_for_repair - 1):
                        ctx["ltx_stage1_full_churn_detected"] = f"call={call_no}; load_delta={load_delta}; unload_delta={unload_delta}; blocks={block_count_for_repair}"
                        _repair_ltx_runtime_planned_hotset(ctx, runtime, role, torch_module=torch_module, echo=echo, reason=f"stage1-full-churn-call-{call_no}")
                try:
                    item["calls"] = call_no
                    item["total_s"] = float(item.get("total_s", 0.0) or 0.0) + dt
                    item["block_load_delta"] = int(item.get("block_load_delta", 0) or 0) + load_delta
                    item["block_unload_delta"] = int(item.get("block_unload_delta", 0) or 0) + unload_delta
                    item["flash_success_delta"] = int(item.get("flash_success_delta", 0) or 0) + flash_ok_delta
                    item["flash_fallback_delta"] = int(item.get("flash_fallback_delta", 0) or 0) + flash_fb_delta
                    item["last_shapes"] = shape_s
                    adapters3 = ctx.setdefault("_ltx_stage2_step_adapters", {})
                    if isinstance(adapters3, dict):
                        adapters3[str(count)] = item
                except Exception:
                    pass
                status = "exception" if failed is not None else "end"
                if failed is None:
                    try:
                        if str(role) == "stage1_initial_denoise":
                            ctx["_ltx_low_profile_after_stage1_ready"] = "YES"
                            ctx["ltx_low_profile_after_stage1_cleanup_trigger"] = f"after {role} call {call_no}"
                        elif str(role) == "stage2_refine_denoise":
                            ctx["_ltx_low_profile_after_stage2_ready"] = "YES"
                            ctx["ltx_low_profile_after_stage2_cleanup_trigger"] = f"after {role} call {call_no}"
                    except Exception:
                        pass
                err = f"; error={type(failed).__name__}: {failed}" if failed is not None else ""
                _ltx_stage2_step_event(
                    ctx,
                    f"forward:{status}",
                    f"adapter={count}; role={role}; call={call_no}; duration={dt:.3f}s; "
                    f"block_load_delta={load_delta}; block_unload_delta={unload_delta}; "
                    f"flash_ok_delta={flash_ok_delta}; flash_fallback_delta={flash_fb_delta}{err}",
                    torch_module,
                    echo,
                )
                try:
                    ctx["ltx_stage2_step_profiler_forward_calls"] = str(_safe_int_ctx(ctx, "ltx_stage2_step_profiler_forward_calls", 0) + 1)
                    _ltx_stage2_step_summarize(ctx)
                except Exception:
                    pass

        try:
            setattr(_fv_stage2_forward_wrapper, "_framevision_ltx_stage2_step_profiler", True)
            setattr(_fv_stage2_forward_wrapper, "_framevision_ltx_stage2_step_original", original_forward)
        except Exception:
            pass
        setattr(adapter, "forward", _fv_stage2_forward_wrapper)
        setattr(adapter, "_framevision_ltx_stage2_step_profiler", True)
    except Exception as exc:
        ctx["ltx_stage2_step_profiler_errors"] = f"install failed: {type(exc).__name__}: {exc}"


def _find_repo_root_from_python() -> Path:
    """Infer the LTX root from the active interpreter when possible."""
    exe = Path(sys.executable).resolve()
    # Expected default: C:\ltx23\environments\.ltx23\python.exe
    parts = list(exe.parts)
    try:
        idx = [p.lower() for p in parts].index("environments")
        if idx > 0:
            return Path(*parts[:idx])
    except Exception:
        pass
    return DEFAULT_LTX_ROOT




TWO_STAGE_PIPELINE_SET = {"two_stages", "two_stages_hq", "a2vid_two_stage"}


def _looks_like_internal_distilled_lora_token(value: Any) -> bool:
    """Return True for the official LTX 2.3 distilled two-stage LoRA path/name."""
    text = str(value or "").replace("\\", "/").lower()
    name = text.rsplit("/", 1)[-1]
    return (
        "ltx-2.3-22b-distilled-lora" in name
        or ("distilled" in name and "lora" in name and name.endswith(('.safetensors', '.pt', '.pth')))
    )


def _sanitize_one_stage_lora_args(args: argparse.Namespace) -> argparse.Namespace:
    """Hard safety guard: one-stage must not receive the internal two-stage LoRA.

    The UI normally only passes --distilled-lora for two-stage pipelines, but
    stale saved settings or manual extra args can still sneak a LoRA through to
    official one_stage, where it causes the same huge CPU/RAM fuse_lora_weights
    path. Keep user-supplied normal LoRAs alone, but strip the official
    distilled two-stage LoRA and spatial upsampler from one-stage runs.
    """
    pipeline = str(getattr(args, "pipeline", "one_stage"))
    removed: List[str] = []
    if pipeline in TWO_STAGE_PIPELINE_SET:
        setattr(args, "_one_stage_lora_guard_status", f"not needed: selected pipeline is {pipeline}")
        setattr(args, "_one_stage_lora_guard_removed", "none")
        return args

    if getattr(args, "distilled_lora", None):
        removed.append(f"--distilled-lora x{len(args.distilled_lora)}")
        args.distilled_lora = []
    if getattr(args, "spatial_upsampler_path", None):
        removed.append("--spatial-upsampler-path")
        args.spatial_upsampler_path = None

    extra = list(getattr(args, "extra", None) or [])
    cleaned: List[str] = []
    i = 0
    while i < len(extra):
        tok = str(extra[i])
        low = tok.lower()
        if low in {"--distilled-lora", "--spatial-upsampler-path"}:
            removed.append(tok)
            i += 1
            while i < len(extra) and not str(extra[i]).startswith("--"):
                i += 1
            continue
        if low == "--lora":
            group = [tok]
            i += 1
            while i < len(extra) and not str(extra[i]).startswith("--"):
                group.append(str(extra[i]))
                i += 1
            if any(_looks_like_internal_distilled_lora_token(x) for x in group[1:]):
                removed.append("--lora internal-distilled-two-stage-lora")
                continue
            cleaned.extend(group)
            continue
        cleaned.append(tok)
        i += 1

    args.extra = cleaned
    if removed:
        setattr(args, "_one_stage_lora_guard_status", "removed internal two-stage LoRA/spatial-upscaler args from one-stage")
        setattr(args, "_one_stage_lora_guard_removed", "; ".join(removed))
    else:
        setattr(args, "_one_stage_lora_guard_status", "OK: no internal two-stage LoRA args found for one-stage")
        setattr(args, "_one_stage_lora_guard_removed", "none")
    return args



def _sanitize_disabled_distilled_lora_args(args: argparse.Namespace) -> argparse.Namespace:
    """Permanent FrameVision no-LoRA route for two-stage LTX.

    The Quality-tab LoRA path/strength are now placeholders.  When the
    FrameVision wrapper runs a two-stage pipeline, strip any official
    --distilled-lora supplied directly or through --extra and force the old
    fused-cache machinery off.  Spatial upsampler is still required.
    """
    if str(getattr(args, "pipeline", "one_stage")) not in TWO_STAGE_PIPELINE_SET:
        return args
    args.disable_distilled_lora = True
    args.distilled_lora = []
    args.lora_fusion_cache = "off"
    extra = list(getattr(args, "extra", None) or [])
    cleaned: List[str] = []
    removed: List[str] = []
    i = 0
    while i < len(extra):
        tok = str(extra[i])
        low = tok.lower()
        if low == "--distilled-lora":
            removed.append("--distilled-lora")
            i += 1
            while i < len(extra) and not str(extra[i]).startswith("--"):
                i += 1
            continue
        if low == "--lora":
            group = [tok]
            i += 1
            while i < len(extra) and not str(extra[i]).startswith("--"):
                group.append(str(extra[i]))
                i += 1
            if any(_looks_like_internal_distilled_lora_token(x) for x in group[1:]):
                removed.append("--lora internal-distilled-two-stage-lora")
                continue
            cleaned.extend(group)
            continue
        cleaned.append(tok)
        i += 1
    args.extra = cleaned
    setattr(args, "_distilled_lora_permanent_skip", "YES: official distilled LoRA path disabled by FrameVision wrapper")
    if removed:
        setattr(args, "_distilled_lora_permanent_skip_removed", "; ".join(removed))
    else:
        setattr(args, "_distilled_lora_permanent_skip_removed", "none")
    return args

def _install_ltx_disable_distilled_lora_parser_override(ctx: Dict[str, Any]) -> None:
    """Allow official LTX two-stage modules to run without --distilled-lora.

    The upstream two-stage parsers mark --distilled-lora as required. For the
    FrameVision test toggle we want a no-repo-edit proof run where stage 2 uses
    only normal user LoRAs (if any) and the spatial upsampler, with no official
    distilled LoRA load/fuse/cache path.
    """
    try:
        import importlib
        import argparse as _argparse

        args_mod = importlib.import_module("ltx_pipelines.utils.args")
        original = getattr(args_mod, "_framevision_original_default_2_stage_arg_parser", None)
        if original is None:
            original = getattr(args_mod, "default_2_stage_arg_parser")
            setattr(args_mod, "_framevision_original_default_2_stage_arg_parser", original)

        def _framevision_default_2_stage_arg_parser_no_required_lora(*call_args: Any, **call_kwargs: Any):
            parser = original(*call_args, **call_kwargs)
            try:
                for action in list(getattr(parser, "_actions", []) or []):
                    opts = set(getattr(action, "option_strings", []) or [])
                    if "--distilled-lora" in opts:
                        action.required = False
                        action.default = []
                        try:
                            action.help = str(getattr(action, "help", "") or "") + " [FrameVision override: optional because the official distilled LoRA is skipped.]"
                        except Exception:
                            pass
                        break
            except Exception:
                pass
            return parser

        setattr(args_mod, "default_2_stage_arg_parser", _framevision_default_2_stage_arg_parser_no_required_lora)

        # If a pipeline module was imported earlier in this process, it may hold a
        # direct reference imported via `from ltx_pipelines.utils.args import ...`.
        # Patch those references too without editing the repo files.
        import sys as _sys
        patched_modules = []
        for mod_name in ("ltx_pipelines.ti2vid_two_stages", "ltx_pipelines.a2vid_two_stage"):
            mod = _sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "default_2_stage_arg_parser"):
                try:
                    setattr(mod, "default_2_stage_arg_parser", _framevision_default_2_stage_arg_parser_no_required_lora)
                    patched_modules.append(mod_name)
                except Exception:
                    pass

        ctx["ltx_disable_distilled_lora_parser_override"] = "YES: --distilled-lora made optional for upstream two-stage parser"
        if patched_modules:
            ctx["ltx_disable_distilled_lora_parser_override_modules"] = ", ".join(patched_modules)
        else:
            ctx["ltx_disable_distilled_lora_parser_override_modules"] = "args module only; pipeline module will import patched parser"
        ctx.setdefault("notes", []).append(
            "Applied no-repo-edit parser override: upstream two-stage --distilled-lora is optional because FrameVision skips the official distilled LoRA path."
        )
    except Exception as exc:
        ctx["ltx_disable_distilled_lora_parser_override"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx.setdefault("notes", []).append(f"Disable distilled LoRA parser override failed: {type(exc).__name__}: {exc}")

def _install_ltx_scheduler_shift_override(ctx: Dict[str, Any], shift: float | None) -> None:
    """Override LTX2Scheduler sigma shift for this wrapper run.

    The official LTX pipeline does not expose a top-level CLI shift setting.
    This wrapper applies the value at the scheduler boundary before the
    selected ltx_pipelines module is executed. It forces both base_shift and
    max_shift to the selected value, producing a fixed sigma-shift schedule.
    """
    if shift is None:
        ctx["ltx_scheduler_shift_override"] = "disabled"
        return
    try:
        shift_value = max(0.0, min(10.0, float(shift)))
    except Exception as exc:
        ctx["ltx_scheduler_shift_override"] = f"FAILED: invalid value {shift!r}: {type(exc).__name__}: {exc}"
        return
    try:
        import ltx_core.components.schedulers as schedulers  # type: ignore

        scheduler_cls = getattr(schedulers, "LTX2Scheduler")
        original_execute = getattr(scheduler_cls, "_framevision_original_execute", None)
        if original_execute is None:
            original_execute = scheduler_cls.execute
            setattr(scheduler_cls, "_framevision_original_execute", original_execute)

        def _framevision_execute_with_shift(self, *call_args: Any, **call_kwargs: Any):
            args_list = list(call_args)
            # Signature after self: steps, latent=None, max_shift=2.05, base_shift=0.95, ...
            if len(args_list) >= 3:
                args_list[2] = shift_value
            else:
                call_kwargs["max_shift"] = shift_value
            if len(args_list) >= 4:
                args_list[3] = shift_value
            else:
                call_kwargs["base_shift"] = shift_value
            return original_execute(self, *args_list, **call_kwargs)

        scheduler_cls.execute = _framevision_execute_with_shift
        beta_cls = getattr(schedulers, "BetaScheduler", None)
        if beta_cls is not None:
            setattr(beta_cls, "shift", shift_value)
        ctx["ltx_scheduler_shift_override"] = f"enabled: {shift_value:g}"
        ctx.setdefault("notes", []).append(
            f"Applied LTX scheduler shift override: base_shift=max_shift={shift_value:g}; BetaScheduler.shift={shift_value:g}."
        )
    except Exception as exc:
        ctx["ltx_scheduler_shift_override"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx.setdefault("notes", []).append(f"LTX scheduler shift override failed: {type(exc).__name__}: {exc}")


def _parse_args() -> argparse.Namespace:
    detected_root = _find_repo_root_from_python()
    parser = argparse.ArgumentParser(description="Run LTX 2.3 through FrameVision VRAM Lab wrapper.")
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_MODULES), default="one_stage", help="Official LTX pipeline module to run. one_stage/two_stages are prompt-driven; a2vid_two_stage uses prompt + audio conditioning.")
    parser.add_argument("--fast-iclora-route", choices=["on", "off"], default=os.environ.get("FRAMEVISION_LTX_FAST_ICLORA_ROUTE", "on"), help="For regular two_stages runs, use the native ltx_pipelines.ic_lora route with a tiny neutral conditioning clip to chase the faster ~10s late Phase-1 steps. Default: on.")
    parser.add_argument("--vram-lab", choices=["off", "safe", "edge", "balanced", "aggressive"], default="safe", help="VRAM Lab mode: safe/off. Legacy names (balanced, edge, aggressive) are accepted but map to the selected safe profile.")
    parser.add_argument("--vram-profile", choices=["auto", "24", "16", "12"], default=os.environ.get("FRAMEVISION_LTX_VRAM_PROFILE_GB", "24"), help="LTX VRAM profile in GB. Use auto to select from detected GPU VRAM: <16GB -> 12, 16-22.99GB -> 16, 23GB+ -> 24. All main profiles include a 3GB I2V safety derate.")
    parser.add_argument("--main-hot-window-gb", type=float, default=None, help="Optional override for the main/distilled 41GB safetensor VRAM load/hot-window budget. Leave unset to use the selected 12/16/24GB profile default. Reduce for longer videos to avoid shared-memory spill. Gemma follows the selected VRAM profile.")
    parser.add_argument("--stage2-block-size-limit-gb", type=float, default=None, help="Optional two-stage-only override for the Stage 2/refine denoise block size / hot-window limit. If omitted or <=0, Stage 2 falls back to the main Stage 1/general limit.")
    parser.add_argument("--vram-residency-strategy", choices=["rolling", "planned_hotset"], default=os.environ.get("FRAMEVISION_VRAM_RESIDENCY_STRATEGY", "planned_hotset"), help="Generic VRAM Lab residency strategy. rolling keeps the old rolling hot-window behavior; planned_hotset keeps a stable hot block subset across repeated forwards.")
    parser.add_argument("--allow-low-profile-planned-hotset", action="store_true", help="Unsafe test override: allow planned_hotset on 12/16 GB profiles. By default lower profiles are forced to rolling because planned_hotset crashed before report writing in low-profile tests.")
    parser.add_argument("--stable-hotset-fraction", type=float, default=float(os.environ.get("FRAMEVISION_VRAM_STABLE_HOTSET_FRACTION", "0.95")), help="Legacy/shared planned_hotset fraction fallback. Stage-specific values override this when supplied.")
    parser.add_argument("--stage1-stable-hotset-fraction", type=float, default=None, help="Planned_hotset fraction for Stage 1 / first denoise. Test range from UI: 0.10 to 2.00.")
    parser.add_argument("--stage2-stable-hotset-fraction", type=float, default=None, help="Planned_hotset fraction for Stage 2 / refine denoise. Test range from UI: 0.70 to 1.10.")
    parser.add_argument("--stable-hotset-budget-gb", type=float, default=float(os.environ.get("FRAMEVISION_VRAM_STABLE_HOTSET_BUDGET_GB", "0.0")), help="Optional fixed planned_hotset stable budget in GB. 0.0 means auto from hot-window * fraction.")
    parser.add_argument("--emergency-free-vram-floor-gb", type=float, default=None, help="Driver-free VRAM floor for emergency trimming/correction. Test range from UI: 0.25 to 3.00 GB. Leave unset to use the selected VRAM profile default; current default is 0.5 GB.")
    parser.add_argument("--block-churn-policy", choices=["default", "sticky_floor"], default="default", help="Compatibility only. sticky_floor is accepted but ignored because it caused emergency trims/shared-memory spill in real tests; default rolling hot-window is always used.")
    parser.add_argument("--block-churn-floor-gb", type=float, default=float(os.environ.get("FRAMEVISION_LTX_BLOCK_CHURN_FLOOR_GB", "2.0")), help="Driver-free VRAM floor for --block-churn-policy sticky_floor. When free VRAM drops below this, oldest resident blocks are trimmed. Default: 2.0 GB.")
    parser.add_argument("--attention-backend", choices=["auto", "sdpa", "pytorch", "flash2", "sage"], default=os.environ.get("FRAMEVISION_LTX_ATTENTION_BACKEND", "auto"), help="Experimental LTX attention backend override. auto tries SageAttention first, then FlashAttention2, then PyTorch SDPA; sdpa/pytorch forces official PyTorch SDPA path; flash2 or sage try the optional backend with safe fallback to PyTorch per call.")
    parser.add_argument("--lora-fusion-cache", choices=["auto", "off", "read", "rebuild"], default="off", help="Legacy/disabled. FrameVision now skips the official distilled LoRA path by default, so fused LoRA cache is forced off.")
    parser.add_argument("--lora-fusion-cache-max", type=int, default=int(os.environ.get("FRAMEVISION_LTX_LORA_FUSION_CACHE_MAX", "2")), help="Maximum number of large fused LoRA cache files to keep. Cleanup runs only after a new cache is saved. Default: 2.")
    parser.add_argument("--lora-fusion-cache-shard-gb", type=float, default=float(os.environ.get("FRAMEVISION_LTX_LORA_FUSION_CACHE_SHARD_GB", "4")), help="Shard size for large fused LoRA cache saves. Large caches are saved as several safetensors shards instead of one huge file. Default: 4 GB.")
    parser.add_argument("--lora-fusion-cache-shard-threshold-gb", type=float, default=float(os.environ.get("FRAMEVISION_LTX_LORA_FUSION_CACHE_SHARD_THRESHOLD_GB", "8")), help="Use sharded fused LoRA cache when the estimated cache is at least this large. Default: 8 GB.")
    parser.add_argument("--lora-fusion-cache-miss-inplace", choices=["auto", "off"], default=os.environ.get("FRAMEVISION_LTX_LORA_FUSION_CACHE_MISS_INPLACE", "auto"), help="On fused LoRA cache miss, try to fuse into the existing base state dict instead of filling a duplicate destination state dict. This reduces first-build RAM/pagefile pressure. Default: auto.")
    parser.add_argument("--lora-fusion-cache-preclean", choices=["auto", "off"], default=os.environ.get("FRAMEVISION_LTX_LORA_FUSION_CACHE_PRECLEAN", "auto"), help="Before first-time fused LoRA cache creation, run a cleanup boundary and log RAM/pagefile. Keeps cache creation from stacking stale tensors on top of the build path. Default: auto.")
    parser.add_argument("--main-transformer-stream-probe", action="store_true", help="Experimental: tries to let the main 42GB LTX transformer use a Gemma-style partial streaming path before full CPU materialization. Memory/probe tests only; default OFF.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--gemma-root", default=str(DEFAULT_GEMMA_ROOT))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--frame-rate", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shift", type=float, default=None, help="Optional scheduler/sigma shift override. Range is clamped to 0..10. When set, the wrapper applies it to LTX2Scheduler before the official pipeline runs; it is not forwarded as an official ltx_pipelines CLI argument.")
    parser.add_argument("--i2v-image", default=None, help="Convenience wrapper for official LTX --image PATH FRAME STRENGTH CRF. Use this instead of putting --image in --extra for cleaner I2V tests.")
    parser.add_argument("--i2v-image-frame", type=int, default=0, help="Frame index for --i2v-image. Default: 0.")
    parser.add_argument("--i2v-image-strength", type=float, default=1.0, help="Official LTX image conditioning strength for --i2v-image. 1.0 preserves the source image strongest. Default: 1.0.")
    parser.add_argument("--i2v-image-crf", type=int, default=0, help="Optional CRF value for official LTX --image conditioning. 0 keeps the conditioning image lossless. Default: 0.")
    image_norm_group = parser.add_mutually_exclusive_group()
    image_norm_group.add_argument("--ltx-normalize-input-image", dest="ltx_normalize_input_image", action="store_true", default=True, help="Normalize image inputs to clean RGB PNG temp copies before handing them to LTX. Default: ON.")
    image_norm_group.add_argument("--no-ltx-normalize-input-image", dest="ltx_normalize_input_image", action="store_false", help="Disable wrapper-side image normalization and pass original image paths to LTX.")
    parser.add_argument("--ltx-video-cfg", type=float, default=None, help="Optional passthrough for official --video-cfg-guidance-scale. Leave unset to use official pipeline default.")
    parser.add_argument("--ltx-video-stg", type=float, default=None, help="Optional passthrough for official --video-stg-guidance-scale. For VRAM Lab two-stage tests, 0.0 avoids the expensive STG extra passes.")
    parser.add_argument("--ltx-video-rescale", type=float, default=None, help="Optional passthrough for official --video-rescale-scale. Leave unset to use official pipeline default.")
    parser.add_argument("--ltx-video-stg-blocks", default=None, help="Optional passthrough for official --video-stg-blocks, for example 28. Leave unset to avoid adding the argument.")
    parser.add_argument("--audio-path", default=None, help="Required for --pipeline a2vid_two_stage. This audio file conditions the generated video and is baked into the output by LTX.")
    parser.add_argument("--audio-start-time", type=float, default=0.0, help="For --pipeline a2vid_two_stage: start time in seconds to read from the audio file.")
    parser.add_argument("--audio-max-duration", type=float, default=None, help="For --pipeline a2vid_two_stage: maximum audio duration in seconds. Leave unset to use video duration.")
    audio_safe_group = parser.add_mutually_exclusive_group()
    audio_safe_group.add_argument("--safe-ltx-audio-load", dest="safe_ltx_audio_load", action="store_true", default=True, help="Default ON: preserve LTX sound, CPU-route late AudioDecoder when safe, and keep Vocoder on CUDA because the Vocoder CPU route can hard-crash on some runs.")
    audio_safe_group.add_argument("--no-safe-ltx-audio-load", dest="safe_ltx_audio_load", action="store_false", help="Advanced comparison only: disable the safer late AudioDecoder load route.")
    parser.add_argument("--disable-distilled-lora", action="store_true", default=True, help="Default ON: for two-stage runs, do not pass the official distilled LoRA to LTX. Spatial upsampler is still required. This flag is kept for command compatibility.")
    parser.add_argument("--distilled-lora", action="append", nargs="+", default=[], metavar=("PATH", "STRENGTH"), help="Required by official two-stage LTX modules unless --disable-distilled-lora is used. Pass PATH and optional STRENGTH, for example: --distilled-lora path/to/distilled_lora.safetensors 0.8")
    parser.add_argument("--lora", action="append", nargs="+", default=[], metavar=("PATH", "STRENGTH"), help="Optional user LoRA passthrough. Accepted here to avoid argparse ambiguity with --lora-fusion-cache options, then forwarded to the selected official LTX pipeline as --lora PATH [STRENGTH].")
    parser.add_argument("--lora-budget-reserve-ui-applied", action="store_true", default=False, help="Internal FrameVision UI flag: visible Stage 1/hotset/Stage 2 values were already reduced for user LoRA size; CLI reports but does not subtract again.")
    parser.add_argument("--lora-budget-ui-total-gb", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lora-budget-ui-stage1-deduction-gb", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lora-budget-ui-stage2-deduction-gb", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--spatial-upsampler-path", default=None, help="Required by official two-stage LTX modules: path to the spatial upsampler .safetensors file.")
    parser.add_argument("--ltx-root", default=str(detected_root if detected_root.exists() else DEFAULT_LTX_ROOT))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--latent-preview", action="store_true", help="Experimental FrameVision latent preview toggle. Enables wrapper-side preview reporting and future sampling callback wiring.")
    parser.add_argument("--latent-preview-mode", choices=["fast_rgb", "tae"], default="fast_rgb", help="Latent preview mode. fast_rgb uses LTX latent RGB factors; tae requests a TAE-style preview path when available.")
    parser.add_argument("--latent-preview-rate", type=int, default=8, help="Requested latent preview FPS/rate. Clamped to 1..30 by the wrapper.")
    parser.add_argument("--latent-preview-sidecar", default="", help="Optional JSONL sidecar path for FrameVision UI latent-preview/step events.")
    parser.add_argument("--latent-preview-upscale", action="store_true", help="Request optional latent-upscale preview handling when available.")
    parser.add_argument("--latent-preview-tae-decode", action="store_true", help="Request TAE-style VAE preview handling when available.")
    parser.add_argument("--no-boundary-echo", action="store_true", help="Do not print [vram-lab-boundary] lines live.")
    parser.add_argument("--deep-lifecycle-log", action="store_true", help="Opt-in noisy logger for Module.to/_apply and CUDA memory polling during LTX run.")
    parser.add_argument("--deep-log-interval", type=float, default=1.0, help="Seconds between deep lifecycle CUDA memory samples.")
    parser.add_argument("--deep-log-max-events", type=int, default=4000, help="Maximum deep lifecycle events kept in the report.")
    parser.add_argument("--deep-log-path", default=str(DEEP_LIVE_LOG_PATH), help="Live deep lifecycle txt log path. Written/flushed after every event.")
    parser.add_argument("--msr-enabled", action="store_true", help="Enable FrameVision MSR multi-reference preparation for LTX 2.3.")
    parser.add_argument("--msr-ref-1", default="", help="MSR subject/reference image 1.")
    parser.add_argument("--msr-ref-2", default="", help="MSR subject/reference image 2.")
    parser.add_argument("--msr-ref-3", default="", help="MSR subject/reference image 3.")
    parser.add_argument("--msr-ref-4", default="", help="MSR subject/reference image 4.")
    parser.add_argument("--msr-background", default="", help="Optional MSR background/world reference image. Leave empty to let LTX create the background from the prompt.")
    parser.add_argument("--msr-ref-1-description", default="")
    parser.add_argument("--msr-ref-2-description", default="")
    parser.add_argument("--msr-ref-3-description", default="")
    parser.add_argument("--msr-ref-4-description", default="")
    parser.add_argument("--msr-background-description", default="")
    parser.add_argument("--msr-frame-count", type=int, choices=[17, 25, 33, 41], default=17)
    parser.add_argument("--msr-fps", type=int, default=50)
    parser.add_argument("--msr-resize-mode", choices=["stretch", "fit", "fill"], default="stretch")
    parser.add_argument("--msr-output-dir", default="")
    parser.add_argument("--msr-save-video", action="store_true")
    parser.add_argument("--msr-prompt-block", default="", help="Optional global reference prompt block to prepend for this run.")
    parser.add_argument("--msr-transport", choices=["auto", "native-iclora"], default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--msr-strength", type=float, default=1.0, help="MSR / IC-LoRA video-conditioning strength.")
    parser.add_argument("--fast-iclora-neutral-frames", type=int, default=int(os.environ.get("FRAMEVISION_LTX_FAST_ICLORA_NEUTRAL_FRAMES", "17")), help=argparse.SUPPRESS)
    parser.add_argument("--fast-iclora-neutral-max-dim", type=int, default=int(os.environ.get("FRAMEVISION_LTX_FAST_ICLORA_NEUTRAL_MAX_DIM", "320")), help=argparse.SUPPRESS)
    parser.add_argument("--fast-iclora-neutral-fps", type=int, default=int(os.environ.get("FRAMEVISION_LTX_FAST_ICLORA_NEUTRAL_FPS", "8")), help=argparse.SUPPRESS)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments passed to the selected official ltx_pipelines module")
    args = parser.parse_args()
    args = _sanitize_one_stage_lora_args(args)
    args = _sanitize_disabled_distilled_lora_args(args)

    # Normal user LoRA 1-4 must NOT go through SingleGPUModelBuilder's full
    # apply_loras/fuse_lora_weights route. Earlier tests also proved that simply
    # forcing --main-transformer-stream-probe is the wrong fix: this repo only
    # reached the Gemma streaming builder, not the main transformer LoRA source,
    # and it made denoise very slow. Keep LoRA as a FrameVision-owned runtime
    # hook instead: upstream can parse --lora, but DiffusionStage is patched
    # before construction so those LoRAs are not passed into the official fuse path.
    if _args_or_extra_contain_user_lora(args):
        setattr(args, "_user_lora_runtime_forward_hooks", True)
        setattr(args, "_user_lora_fast_stream_forced", False)
    else:
        setattr(args, "_user_lora_runtime_forward_hooks", False)
        setattr(args, "_user_lora_fast_stream_forced", False)
    if args.pipeline in ("two_stages", "two_stages_hq", "a2vid_two_stage"):
        extra = list(args.extra or [])
        args.distilled_lora = []
        args.lora_fusion_cache = "off"
        args.disable_distilled_lora = True
        if "--spatial-upsampler-path" not in extra and not args.spatial_upsampler_path:
            parser.error(f"--pipeline {args.pipeline} requires --spatial-upsampler-path PATH based on the official LTX two-stage/a2vid parser")
    if args.pipeline == "a2vid_two_stage":
        extra = list(args.extra or [])
        has_extra_audio = "--audio-path" in extra
        if not has_extra_audio and not args.audio_path:
            parser.error("--pipeline a2vid_two_stage requires --audio-path AUDIO_FILE")
    return args


def _configure_latent_preview_runtime(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Record and export FrameVision latent preview settings.

    The Comfy node reference wraps the sampling callback and converts x0 latents
    to JPEG previews using LTX latent RGB factors. Native FrameVision LTX runs
    do not use Comfy's PromptServer, so this wrapper keeps the controls and
    command path isolated here and exposes stable environment/settings for the
    later native callback bridge without forwarding unknown args to LTX.
    """
    enabled = bool(getattr(args, "latent_preview", False))
    mode = str(getattr(args, "latent_preview_mode", "fast_rgb") or "fast_rgb").strip().lower()
    if mode not in {"fast_rgb", "tae"}:
        mode = "fast_rgb"
    try:
        rate = max(1, min(30, int(getattr(args, "latent_preview_rate", 8) or 8)))
    except Exception:
        rate = 8
    upscale = bool(getattr(args, "latent_preview_upscale", False))
    tae_decode = bool(getattr(args, "latent_preview_tae_decode", False) or mode == "tae")

    ctx["latent_preview_enabled"] = "YES" if enabled else "NO"
    ctx["latent_preview_mode"] = mode
    ctx["latent_preview_rate"] = str(rate)
    ctx["latent_preview_upscale"] = "YES" if upscale else "NO"
    ctx["latent_preview_tae_decode"] = "YES" if tae_decode else "NO"
    ctx["latent_preview_sidecar"] = str(getattr(args, "latent_preview_sidecar", "") or "").strip()
    ctx["latent_preview_width"] = str(getattr(args, "width", "") or "")
    ctx["latent_preview_height"] = str(getattr(args, "height", "") or "")
    ctx["latent_preview_num_frames"] = str(getattr(args, "num_frames", "") or "")
    ctx["latent_preview_runtime"] = "native x0/denoised latent RGB preview requested" if enabled else "disabled"
    if enabled:
        try:
            sidecar = ctx.get("latent_preview_sidecar", "")
            if sidecar:
                sp = Path(str(sidecar))
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text("", encoding="utf-8")
        except Exception:
            pass
        _latent_preview_sidecar_write(ctx, {"kind": "status", "message": f"Latent preview enabled ({mode}, {rate} fps). Waiting for sampling."})

    os.environ["FRAMEVISION_LTX_LATENT_PREVIEW"] = "1" if enabled else "0"
    os.environ["FRAMEVISION_LTX_LATENT_PREVIEW_MODE"] = mode
    os.environ["FRAMEVISION_LTX_LATENT_PREVIEW_RATE"] = str(rate)
    os.environ["FRAMEVISION_LTX_LATENT_PREVIEW_UPSCALE"] = "1" if upscale else "0"
    os.environ["FRAMEVISION_LTX_LATENT_PREVIEW_TAE"] = "1" if tae_decode else "0"
    if enabled:
        ctx.setdefault("notes", []).append(
            f"Latent Preview requested: mode={mode}, rate={rate} fps, upscale={'yes' if upscale else 'no'}, tae={'yes' if tae_decode else 'no'}."
        )


def _selected_module(args: argparse.Namespace) -> str:
    # MSR always uses the native IC-LoRA route. For normal two_stages runs,
    # the fast IC-LoRA route can be toggled on/off without keeping the MSR UI.
    if bool(getattr(args, "msr_enabled", False)):
        return "ltx_pipelines.ic_lora"
    if str(getattr(args, "pipeline", "")) == "two_stages" and str(getattr(args, "fast_iclora_route", "on")).lower().strip() == "on":
        return "ltx_pipelines.ic_lora"
    return PIPELINE_MODULES[str(args.pipeline)]


def _ltx_module_source_contains(module_name: str, option: str) -> bool:
    """Best-effort check whether the selected native LTX module knows an option."""
    try:
        spec = importlib.util.find_spec(str(module_name))
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            return False
        path = Path(str(origin))
        if not path.exists() or not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
        return str(option) in text
    except Exception:
        return False


def _msr_reference_args_for_ltx(args: argparse.Namespace, selected_module: str) -> List[str]:
    """Compatibility stub.

    Real MSR routing is owned by helpers/ltx23_msr_iclora_workflow.py and is
    selected at the start of _ltx_argv(). Do not add normal --image fallback
    transport here.
    """
    return []


def _uses_native_ic_lora_fast_route(args: argparse.Namespace, selected_module: str) -> bool:
    return str(selected_module).strip() == "ltx_pipelines.ic_lora"


def _ensure_neutral_ic_lora_conditioning_video(args: argparse.Namespace) -> str:
    """Create/reuse a tiny neutral black conditioning MP4 for the fast IC-LoRA route.

    The native ic_lora parser requires --video-conditioning, but using a full-size
    full-length dummy clip caused heavy shared-memory spill. For the speed path we
    intentionally keep the placeholder tiny: a small black clip with a short fixed
    frame count and low FPS. It is only meant to unlock the faster parser/module
    path, not to act as a real visual reference.
    """
    out_w = max(64, int(getattr(args, "width", 1280) or 1280))
    out_h = max(64, int(getattr(args, "height", 704) or 704))
    max_dim = max(96, int(getattr(args, "fast_iclora_neutral_max_dim", 320) or 320))
    frames = max(1, min(33, int(getattr(args, "fast_iclora_neutral_frames", 17) or 17)))
    fps_safe = max(1, min(30, int(getattr(args, "fast_iclora_neutral_fps", 8) or 8)))

    def _round32_down(v: int) -> int:
        return max(64, int(v) // 32 * 32)

    scale = min(float(max_dim) / float(out_w), float(max_dim) / float(out_h), 1.0)
    width = _round32_down(int(round(out_w * scale)))
    height = _round32_down(int(round(out_h * scale)))
    width = min(width, max_dim)
    height = min(height, max_dim)

    try:
        root = Path(str(APP_ROOT))
    except Exception:
        root = Path.cwd()
    out_dir = root / "temp" / "video_refs" / "fast_ic_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"neutral_black_small_{width}x{height}_{frames}f_{fps_safe}fps.mp4"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return str(out_path)

    try:
        import cv2
        import numpy as np
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_safe), (int(width), int(height)))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        for _ in range(int(frames)):
            writer.write(frame)
        writer.release()
        if out_path.is_file() and out_path.stat().st_size > 0:
            return str(out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to create fast IC-LoRA neutral conditioning video: {type(exc).__name__}: {exc}") from exc

    raise RuntimeError("Failed to create fast IC-LoRA neutral conditioning video: output file missing")


def _ltx_argv(args: argparse.Namespace) -> List[str]:
    selected_module = _selected_module(args)
    if bool(getattr(args, "msr_enabled", False)):
        try:
            from ltx23_msr_iclora_workflow import prepare_msr_iclora_plan
        except Exception:
            from helpers.ltx23_msr_iclora_workflow import prepare_msr_iclora_plan
        plan = prepare_msr_iclora_plan(args, app_root=APP_ROOT, report_stamp=REPORT_STAMP)
        return list(plan.argv)
    uses_ic_lora_fast_route = _uses_native_ic_lora_fast_route(args, selected_module)
    if uses_ic_lora_fast_route:
        argv = [
            selected_module,
            "--distilled-checkpoint-path", str(args.checkpoint_path),
            "--gemma-root", str(args.gemma_root),
            "--prompt", str(args.prompt),
            "--output-path", str(args.output_path),
            "--height", str(int(args.height)),
            "--width", str(int(args.width)),
            "--num-frames", str(int(args.num_frames)),
            "--frame-rate", str(float(args.frame_rate)),
            "--seed", str(int(args.seed)),
        ]
    else:
        argv = [
            selected_module,
            "--checkpoint-path", str(args.checkpoint_path),
            "--gemma-root", str(args.gemma_root),
            "--prompt", str(args.prompt),
            "--output-path", str(args.output_path),
            "--height", str(int(args.height)),
            "--width", str(int(args.width)),
            "--num-frames", str(int(args.num_frames)),
            "--frame-rate", str(int(args.frame_rate)),
            "--num-inference-steps", str(int(args.num_inference_steps)),
            "--seed", str(int(args.seed)),
        ]
    if str(args.pipeline) in ("two_stages", "two_stages_hq", "a2vid_two_stage"):
        if not uses_ic_lora_fast_route and not bool(getattr(args, "disable_distilled_lora", False)):
            for lora_args in list(args.distilled_lora or []):
                argv.append("--distilled-lora")
                argv.extend(str(x) for x in lora_args)
        if args.spatial_upsampler_path:
            argv.extend(["--spatial-upsampler-path", str(args.spatial_upsampler_path)])
    if str(args.pipeline) == "a2vid_two_stage" and getattr(args, "audio_path", None):
        argv.extend(["--audio-path", str(args.audio_path)])
        argv.extend(["--audio-start-time", str(float(getattr(args, "audio_start_time", 0.0)))])
        audio_max_duration = getattr(args, "audio_max_duration", None)
        if audio_max_duration is not None and float(audio_max_duration) > 0:
            argv.extend(["--audio-max-duration", str(float(audio_max_duration))])
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    if getattr(args, "i2v_image", None) and not _has_cli_option(extra, "--image"):
        argv.extend([
            "--image",
            str(args.i2v_image),
            str(int(getattr(args, "i2v_image_frame", 0))),
            str(float(getattr(args, "i2v_image_strength", 1.0))),
            str(int(getattr(args, "i2v_image_crf", 0))),
        ])

    if not uses_ic_lora_fast_route:
        if getattr(args, "ltx_video_cfg", None) is not None and not _has_cli_option(extra, "--video-cfg-guidance-scale"):
            argv.extend(["--video-cfg-guidance-scale", str(float(args.ltx_video_cfg))])
        if getattr(args, "ltx_video_stg", None) is not None and not _has_cli_option(extra, "--video-stg-guidance-scale"):
            argv.extend(["--video-stg-guidance-scale", str(float(args.ltx_video_stg))])
        if getattr(args, "ltx_video_rescale", None) is not None and not _has_cli_option(extra, "--video-rescale-scale"):
            argv.extend(["--video-rescale-scale", str(float(args.ltx_video_rescale))])
        stg_blocks = getattr(args, "ltx_video_stg_blocks", None)
        if stg_blocks not in (None, "") and not _has_cli_option(extra, "--video-stg-blocks"):
            argv.extend(["--video-stg-blocks", str(stg_blocks)])

    for user_lora_args in list(getattr(args, "lora", None) or []):
        clean_lora_args = [str(x) for x in (user_lora_args or []) if str(x).strip()]
        if clean_lora_args:
            argv.append("--lora")
            argv.extend(clean_lora_args)

    if uses_ic_lora_fast_route and not _has_cli_option(extra, "--video-conditioning"):
        neutral_conditioning_video = _ensure_neutral_ic_lora_conditioning_video(args)
        argv.extend(["--video-conditioning", neutral_conditioning_video, "0"])

    if str(getattr(args, "vram_lab", "off")).lower().strip() != "off" and not _has_cli_option(extra, "--offload"):
        # Request the official pre-CUDA streaming owner point, then the
        # FrameVision Gemma-only partial streaming gate below prevents the old
        # full-transformer CPU-offload/DDR-fill behavior.
        argv.extend(["--offload", "cpu"])
    if extra:
        if uses_ic_lora_fast_route:
            try:
                from ltx23_msr_iclora_workflow import _filter_ic_lora_extra_args
            except Exception:
                try:
                    from helpers.ltx23_msr_iclora_workflow import _filter_ic_lora_extra_args
                except Exception:
                    _filter_ic_lora_extra_args = None
            if _filter_ic_lora_extra_args is not None:
                extra = _filter_ic_lora_extra_args(extra)
        argv.extend(extra)
    return argv



# ---- User LoRA visibility / diagnostics ------------------------------------
# These helpers are intentionally logs-only. They do not change LoRA behavior,
# LoRA strength, cache mode, or the official LTX argument flow.

def _safe_file_size_text(path_text: Any) -> tuple[bool, str, str]:
    try:
        path = Path(str(path_text)).expanduser()
        exists = path.is_file()
        if exists:
            return True, _format_bytes(int(path.stat().st_size)), str(path)
        return False, "missing", str(path)
    except Exception as exc:
        return False, f"error: {type(exc).__name__}: {exc}", str(path_text)


def _normalise_user_lora_groups(raw_groups: Any) -> List[List[str]]:
    out: List[List[str]] = []
    try:
        for group in list(raw_groups or []):
            clean = [str(x).strip() for x in list(group or []) if str(x).strip()]
            if clean:
                out.append(clean)
    except Exception:
        pass
    return out


def _scan_argv_lora_groups(argv: List[str]) -> List[List[str]]:
    groups: List[List[str]] = []
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if token == "--lora":
            group: List[str] = []
            j = i + 1
            while j < len(argv) and not str(argv[j]).startswith("--"):
                group.append(str(argv[j]))
                j += 1
            groups.append(group)
            i = j
            continue
        i += 1
    return groups


def _args_or_extra_contain_user_lora(args: argparse.Namespace) -> bool:
    """Return True when normal user LoRA args are present.

    FrameVision UI/queue can send LoRA either through wrapper --lora or
    through --extra --lora PATH STRENGTH.  This helper is intentionally tiny
    and side-effect free so _parse_args can decide whether to force the safe
    streaming route before the upstream LTX parser sees the arguments.
    """
    try:
        if _normalise_user_lora_groups(getattr(args, "lora", None)):
            return True
    except Exception:
        pass
    try:
        if _scan_argv_lora_groups(list(getattr(args, "extra", []) or [])):
            return True
    except Exception:
        pass
    return False


def _user_lora_group_summary(groups: List[List[str]]) -> str:
    if not groups:
        return "none"
    parts: List[str] = []
    for idx, group in enumerate(groups, 1):
        path = group[0] if group else ""
        strength = group[1] if len(group) > 1 else "default"
        exists, size, norm_path = _safe_file_size_text(path)
        parts.append(f"slot {idx}: exists={'YES' if exists else 'NO'}; size={size}; strength={strength}; path={norm_path}")
    return " | ".join(parts)


def _collect_user_lora_groups_from_args(args: argparse.Namespace) -> List[List[str]]:
    """Collect normal user LoRA groups from wrapper args and --extra.

    This intentionally ignores the official distilled LoRA path.  It is used
    for runtime LoRA hooks and for the temporary VRAM budget reservation.
    """
    groups: List[List[str]] = []
    try:
        groups.extend(_normalise_user_lora_groups(getattr(args, "lora", None)))
    except Exception:
        pass
    try:
        groups.extend(_scan_argv_lora_groups(list(getattr(args, "extra", []) or [])))
    except Exception:
        pass
    return groups


def _user_lora_total_size_bytes(groups: List[List[str]]) -> int:
    """Return total on-disk size for existing user LoRA files.

    The current runtime-forward implementation keeps LoRA tensors resident on
    CUDA, so this size is a useful conservative reservation input.  Missing
    paths contribute 0 so bad UI paths do not break argparse/setup.
    """
    total = 0
    seen: set[str] = set()
    for group in groups or []:
        try:
            path_text = str(group[0]).strip() if group else ""
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            if path.is_file():
                total += int(path.stat().st_size)
        except Exception:
            continue
    return int(total)


def _apply_user_lora_budget_reservation(
    args: argparse.Namespace,
    profile: Dict[str, Any],
    stage2_limit_gb: float,
    stage2_enabled: bool,
    stable_hotset_budget_gb: float,
) -> tuple[float, bool, float, Dict[str, str]]:
    """Temporarily reserve VRAM for CUDA-resident user LoRAs.

    This is a simple safety workaround until VRAM Lab treats runtime LoRA
    tensors as first-class permanent residents.  Stage 1/main hot-window and
    the stable hotset budget lose roughly LoRA size + headroom.  Stage 2 loses
    about 2x LoRA size + headroom because tests showed refine needs much more
    breathing room to avoid spill.
    """
    info: Dict[str, str] = {
        "ltx_user_lora_budget_adjust": "OFF: no user LoRA",
        "ltx_user_lora_budget_total_bytes": "0 B",
        "ltx_user_lora_budget_total_gb": "0.00",
        "ltx_user_lora_budget_stage1_deduction_gb": "0.00",
        "ltx_user_lora_budget_stage2_deduction_gb": "0.00",
        "ltx_user_lora_budget_original_stage1_gb": f"{float(profile.get('main_hot_window_gb', 0.0) or 0.0):.2f}",
        "ltx_user_lora_budget_effective_stage1_gb": f"{float(profile.get('main_hot_window_gb', 0.0) or 0.0):.2f}",
        "ltx_user_lora_budget_original_stage2_gb": f"{float(stage2_limit_gb or 0.0):.2f}",
        "ltx_user_lora_budget_effective_stage2_gb": f"{float(stage2_limit_gb or 0.0):.2f}",
        "ltx_user_lora_budget_original_hotset_budget_gb": f"{float(stable_hotset_budget_gb or 0.0):.2f}",
        "ltx_user_lora_budget_effective_hotset_budget_gb": f"{float(stable_hotset_budget_gb or 0.0):.2f}",
        "ltx_user_lora_budget_note": "not active",
    }
    groups = _collect_user_lora_groups_from_args(args)
    if not groups:
        return float(stage2_limit_gb or 0.0), bool(stage2_enabled), float(stable_hotset_budget_gb or 0.0), info

    total_bytes = _user_lora_total_size_bytes(groups)
    total_gb = float(total_bytes) / float(1024 ** 3) if total_bytes > 0 else 0.0

    if bool(getattr(args, "lora_budget_reserve_ui_applied", False)):
        try:
            ui_total_gb = float(getattr(args, "lora_budget_ui_total_gb", 0.0) or 0.0)
        except Exception:
            ui_total_gb = total_gb
        try:
            ui_stage1_deduct = float(getattr(args, "lora_budget_ui_stage1_deduction_gb", 0.0) or 0.0)
        except Exception:
            ui_stage1_deduct = 0.0
        try:
            ui_stage2_deduct = float(getattr(args, "lora_budget_ui_stage2_deduction_gb", 0.0) or 0.0)
        except Exception:
            ui_stage2_deduct = 0.0
        info.update({
            "ltx_user_lora_budget_adjust": "ON: already applied by UI command builder",
            "ltx_user_lora_budget_total_bytes": _fmt_bytes(total_bytes),
            "ltx_user_lora_budget_total_gb": f"{(ui_total_gb or total_gb):.2f}",
            "ltx_user_lora_budget_stage1_deduction_gb": f"{ui_stage1_deduct:.2f}",
            "ltx_user_lora_budget_stage2_deduction_gb": f"{ui_stage2_deduct:.2f}",
            "ltx_user_lora_budget_original_stage1_gb": "UI already reduced",
            "ltx_user_lora_budget_effective_stage1_gb": f"{float(profile.get('main_hot_window_gb', 0.0) or 0.0):.2f}",
            "ltx_user_lora_budget_original_stage2_gb": "UI already reduced",
            "ltx_user_lora_budget_effective_stage2_gb": f"{float(stage2_limit_gb or 0.0):.2f}",
            "ltx_user_lora_budget_original_hotset_budget_gb": "UI already reduced",
            "ltx_user_lora_budget_effective_hotset_budget_gb": f"{float(stable_hotset_budget_gb or 0.0):.2f}",
            "ltx_user_lora_budget_note": "UI command builder already subtracted LoRA reserve from Stage 1/main, stable hotset, and Stage 2; CLI skipped second subtraction.",
        })
        return float(stage2_limit_gb or 0.0), bool(stage2_enabled), float(stable_hotset_budget_gb or 0.0), info

    # Environment overrides are intentionally simple for edge testing.
    try:
        headroom_gb = max(0.0, float(os.environ.get("FRAMEVISION_LTX_LORA_BUDGET_HEADROOM_GB", "0.5") or 0.5))
    except Exception:
        headroom_gb = 0.5
    try:
        stage2_mult = max(1.0, float(os.environ.get("FRAMEVISION_LTX_LORA_STAGE2_MULTIPLIER", "2.0") or 2.0))
    except Exception:
        stage2_mult = 2.0
    try:
        min_limit_gb = max(1.5, float(os.environ.get("FRAMEVISION_LTX_LORA_MIN_BLOCK_LIMIT_GB", "1.5") or 1.5))
    except Exception:
        min_limit_gb = 1.5

    stage1_deduct = total_gb + headroom_gb
    stage2_deduct = (total_gb * stage2_mult) + headroom_gb

    original_stage1 = float(profile.get("main_hot_window_gb", 0.0) or 0.0)
    original_stage2 = float(stage2_limit_gb or 0.0)
    original_hotset = float(stable_hotset_budget_gb or 0.0)

    effective_stage1 = max(min_limit_gb, original_stage1 - stage1_deduct) if original_stage1 > 0 else original_stage1
    effective_stage2 = max(min_limit_gb, original_stage2 - stage2_deduct) if original_stage2 > 0 else original_stage2
    effective_hotset = max(min_limit_gb, original_hotset - stage1_deduct) if original_hotset > 0 else original_hotset

    # Round down a little so we do not keep planning exactly on the pressure edge.
    if effective_stage1 > 0:
        effective_stage1 = math.floor(effective_stage1 * 10.0) / 10.0
    if effective_stage2 > 0:
        effective_stage2 = math.floor(effective_stage2 * 10.0) / 10.0
    if effective_hotset > 0:
        effective_hotset = math.floor(effective_hotset * 10.0) / 10.0

    if original_stage1 > 0:
        profile["main_hot_window_gb"] = float(effective_stage1)
        profile["main_hot_window_lora_original_gb"] = float(original_stage1)
    if original_hotset > 0:
        stable_hotset_budget_gb = float(effective_hotset)
    if original_stage2 > 0:
        stage2_limit_gb = float(effective_stage2)
        stage2_enabled = bool(stage2_limit_gb > 0.0 and str(getattr(args, "pipeline", "")).strip() in TWO_STAGE_PIPELINE_SET)

    info.update({
        "ltx_user_lora_budget_adjust": "ON",
        "ltx_user_lora_budget_total_bytes": _fmt_bytes(total_bytes),
        "ltx_user_lora_budget_total_gb": f"{total_gb:.2f}",
        "ltx_user_lora_budget_stage1_deduction_gb": f"{stage1_deduct:.2f}",
        "ltx_user_lora_budget_stage2_deduction_gb": f"{stage2_deduct:.2f}",
        "ltx_user_lora_budget_original_stage1_gb": f"{original_stage1:.2f}",
        "ltx_user_lora_budget_effective_stage1_gb": f"{float(profile.get('main_hot_window_gb', 0.0) or 0.0):.2f}",
        "ltx_user_lora_budget_original_stage2_gb": f"{original_stage2:.2f}",
        "ltx_user_lora_budget_effective_stage2_gb": f"{float(stage2_limit_gb or 0.0):.2f}",
        "ltx_user_lora_budget_original_hotset_budget_gb": f"{original_hotset:.2f}",
        "ltx_user_lora_budget_effective_hotset_budget_gb": f"{float(stable_hotset_budget_gb or 0.0):.2f}",
        "ltx_user_lora_budget_note": (
            f"temporary reservation: Stage1/main and stable hotset subtract LoRA size + {headroom_gb:.2f}GB; "
            f"Stage2 subtracts LoRA size x{stage2_mult:.2f} + {headroom_gb:.2f}GB; min limit {min_limit_gb:.1f}GB"
        ),
    })
    return float(stage2_limit_gb or 0.0), bool(stage2_enabled), float(stable_hotset_budget_gb or 0.0), info


def _record_user_lora_event(ctx: Dict[str, Any], label: str, detail: str, torch_module: Any | None = None, echo: bool = True) -> None:
    try:
        t0 = float(ctx.setdefault("_ltx_user_lora_t0", time.perf_counter()))
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = 0.0
    try:
        cuda = _cuda_snapshot(torch_module)
    except Exception:
        cuda = "n/a"
    try:
        ram = _ram_snapshot()
    except Exception:
        ram = "n/a"
    line = f"{elapsed:9.3f}s | {label}: {detail} | {cuda} | {ram}"
    try:
        events = ctx.setdefault("ltx_user_lora_events", [])
        if isinstance(events, list) and len(events) < 300:
            events.append(line)
        ctx["ltx_user_lora_event_count"] = str(len(events) if isinstance(events, list) else 0)
    except Exception:
        pass
    if echo:
        try:
            print(f"[ltx-lora] {line}", flush=True)
        except Exception:
            pass


def _init_user_lora_report(ctx: Dict[str, Any], args: argparse.Namespace) -> None:
    groups = _normalise_user_lora_groups(getattr(args, "lora", None))
    ctx["ltx_user_lora_cli_count"] = str(len(groups))
    ctx["ltx_user_lora_cli_summary"] = _user_lora_group_summary(groups)
    ctx["ltx_user_lora_argv_count"] = "not built yet"
    ctx["ltx_user_lora_argv_summary"] = "not built yet"
    ctx["ltx_user_lora_forwarded"] = "not checked"
    ctx["ltx_user_lora_runtime_logger"] = "not installed"
    ctx["ltx_user_lora_runtime_status"] = "not observed"
    ctx["ltx_user_lora_load_model_weights_calls"] = "0"
    ctx["ltx_user_lora_apply_loras_calls"] = "0"
    ctx["ltx_user_lora_fuse_lora_weights_calls"] = "0"
    ctx["ltx_user_lora_events"] = []
    ctx["ltx_user_lora_event_count"] = "0"


def _record_user_lora_argv(ctx: Dict[str, Any], argv: List[str], torch_module: Any | None = None, echo: bool = True) -> None:
    groups = _scan_argv_lora_groups(argv)
    ctx["ltx_user_lora_argv_count"] = str(len(groups))
    ctx["ltx_user_lora_argv_summary"] = _user_lora_group_summary(groups)
    ctx["ltx_user_lora_forwarded"] = "YES" if groups else "NO"
    ctx["ltx_final_argv_contains_lora"] = "YES" if groups else "NO"
    try:
        ctx["ltx_final_argv_lora_pairs"] = " | ".join(" ".join(g) for g in groups) if groups else "none"
    except Exception:
        ctx["ltx_final_argv_lora_pairs"] = "unavailable"
    _record_user_lora_event(ctx, "argv", f"final argv contains --lora={'YES' if groups else 'NO'}; {ctx.get('ltx_user_lora_argv_summary', 'none')}", torch_module, echo)
    if not groups and str(ctx.get("ltx_user_lora_cli_count", "0")) not in ("0", "none", "not built yet"):
        ctx.setdefault("notes", []).append("WARNING: User LoRA was present in wrapper args but no --lora entry was found in final LTX argv.")
        _record_user_lora_event(ctx, "warning", "wrapper args contained LoRA groups but final argv did not", torch_module, echo)



def _install_ltx_user_lora_runtime_source_patch(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Replace LTX block-streaming LoraSource with a FrameVision runtime source.

    The upstream LoraSource always builds a pinned-memory CPU cache.  On Windows
    that shows up as Shared GPU memory and can make LoRA 1-4 slow even when the
    full apply_loras/fuse_lora_weights path is blocked.  This wrapper-side
    source keeps normal user LoRAs out of the SingleGPU fuse route, then keeps
    the LoRA A/B tensors resident on CUDA when there is enough dedicated VRAM.
    It falls back to ordinary non-pinned CPU tensors instead of pinned/pageable
    staging, so it should not create the old NVMe/DDR/shared-memory storm.
    """
    ctx["ltx_user_lora_runtime_source_patch"] = "NO"
    ctx.setdefault("ltx_user_lora_runtime_source_residency", "n/a")
    ctx.setdefault("ltx_user_lora_runtime_source_bytes", "0 B")
    ctx.setdefault("ltx_user_lora_runtime_source_pairs", "0")
    ctx.setdefault("ltx_user_lora_runtime_source_errors", "none")
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        import safetensors  # type: ignore
        disk_mod = importlib.import_module("ltx_core.block_streaming.disk")
        builder_mod = importlib.import_module("ltx_core.block_streaming.builder")
        fuse_mod = importlib.import_module("ltx_core.loader.fuse_loras")
    except Exception as exc:
        ctx["ltx_user_lora_runtime_source_patch"] = f"FAILED import: {type(exc).__name__}: {exc}"
        ctx["ltx_user_lora_runtime_source_errors"] = ctx["ltx_user_lora_runtime_source_patch"]
        return

    try:
        original_source = getattr(disk_mod, "_framevision_original_lora_source", None)
        if original_source is None and hasattr(disk_mod, "LoraSource"):
            original_source = getattr(disk_mod, "LoraSource")
            setattr(disk_mod, "_framevision_original_lora_source", original_source)

        LoraProduct = getattr(fuse_mod, "LoraProduct")

        def _tensor_bytes(t: Any) -> int:
            try:
                if torch.is_tensor(t):
                    return int(t.numel()) * int(t.element_size())
            except Exception:
                pass
            return 0

        def _fmt_local(n: int) -> str:
            try:
                return _fmt_bytes(int(n))
            except Exception:
                return f"{n} B"

        class FrameVisionRuntimeLoraSource:
            def __init__(self, path: str, sd_ops: Any, strength: float) -> None:
                self.path = str(path)
                self.strength = float(strength)
                self._pinned_ab: Dict[str, tuple[Any, Any]] = {}
                self._residency = "uninitialized"
                self._bytes = 0
                self._pairs = 0
                self._errors = "none"

                a_keys: Dict[str, str] = {}
                b_keys: Dict[str, str] = {}
                try:
                    with safetensors.safe_open(self.path, framework="pt", device="cpu") as handle:
                        for sft_key in handle.keys():
                            model_key = sd_ops.apply_to_key(sft_key) if sd_ops is not None else sft_key
                            if model_key is None:
                                continue
                            if str(model_key).endswith(".lora_A.weight"):
                                a_keys[str(model_key)[: -len(".lora_A.weight")]] = sft_key
                            elif str(model_key).endswith(".lora_B.weight"):
                                b_keys[str(model_key)[: -len(".lora_B.weight")]] = sft_key

                        matched_prefixes = list(a_keys.keys() & b_keys.keys())
                        cpu_pairs: Dict[str, tuple[Any, Any]] = {}
                        total_bytes = 0
                        for prefix in matched_prefixes:
                            a = handle.get_tensor(a_keys[prefix])
                            b = handle.get_tensor(b_keys[prefix])
                            total_bytes += _tensor_bytes(a) + _tensor_bytes(b)
                            cpu_pairs[prefix] = (a, b)

                    prefer_cuda = os.environ.get("FRAMEVISION_LTX_USER_LORA_RESIDENCY", "cuda").strip().lower()
                    cuda_ok = False
                    if prefer_cuda not in {"cpu", "normal_cpu", "no_cuda"}:
                        try:
                            cuda_ok = bool(torch.cuda.is_available())
                        except Exception:
                            cuda_ok = False

                    # Keep some real headroom.  This is LoRA-only residency, not a
                    # model hotset target.  If the LoRA cannot fit cleanly, use
                    # ordinary CPU tensors instead of pinned shared-memory tensors.
                    headroom_bytes = int(float(os.environ.get("FRAMEVISION_LTX_USER_LORA_CUDA_HEADROOM_GB", "0.50") or "0.50") * (1024**3))
                    if cuda_ok:
                        try:
                            free_bytes, _total_bytes = torch.cuda.mem_get_info()
                            cuda_ok = int(free_bytes) > int(total_bytes) + headroom_bytes
                        except Exception:
                            cuda_ok = False

                    if cuda_ok:
                        device = torch.device("cuda:0")
                        resident: Dict[str, tuple[Any, Any]] = {}
                        for prefix, (a, b) in cpu_pairs.items():
                            resident[prefix] = (a.to(device=device, non_blocking=False), b.to(device=device, non_blocking=False))
                        self._pinned_ab = resident
                        self._residency = "cuda"
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    else:
                        self._pinned_ab = cpu_pairs
                        self._residency = "cpu_unpinned"

                    self._bytes = int(total_bytes)
                    self._pairs = len(self._pinned_ab)
                    try:
                        ctx["ltx_user_lora_runtime_source_residency"] = self._residency
                        ctx["ltx_user_lora_runtime_source_bytes"] = _fmt_local(self._bytes)
                        ctx["ltx_user_lora_runtime_source_pairs"] = str(self._pairs)
                        ctx["ltx_user_lora_runtime_source_path"] = self.path
                        _record_user_lora_event(
                            ctx,
                            "runtime_source:init",
                            f"residency={self._residency}; pairs={self._pairs}; bytes={_fmt_local(self._bytes)}; path={self.path}",
                            torch,
                            echo,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    self._errors = f"{type(exc).__name__}: {exc}"
                    ctx["ltx_user_lora_runtime_source_errors"] = self._errors
                    _record_user_lora_event(ctx, "runtime_source:error", self._errors, torch, echo)
                    raise

            def get_ab(self, param_prefix: str, device: Any | None = None, dtype: Any | None = None) -> Any | None:
                pair = self._pinned_ab.get(param_prefix)
                if pair is None:
                    return None
                a, b = pair

                # If CUDA resident, keep it resident and only cast once.  If CPU
                # fallback, return transfer/cast tensors without creating pinned
                # shared staging memory.
                try:
                    target_device = device if device is not None else None
                    if target_device is not None and getattr(target_device, "type", str(target_device)) == "cuda":
                        if getattr(a, "device", None) != target_device:
                            a = a.to(device=target_device, non_blocking=False)
                            b = b.to(device=target_device, non_blocking=False)
                    if dtype is not None and getattr(a, "dtype", None) != dtype:
                        a = a.to(dtype=dtype)
                        b = b.to(dtype=dtype)
                    # Cache the converted pair only for CUDA-resident mode.  This
                    # avoids repeated per-block casts without turning CPU fallback
                    # into a hidden permanent CUDA allocation when VRAM is tight.
                    if getattr(a, "is_cuda", False) and self._residency == "cuda":
                        self._pinned_ab[param_prefix] = (a, b)
                except Exception:
                    # Let the original caller surface real dtype/device failures.
                    raise
                return LoraProduct(a, b, self.strength)

            def cleanup(self) -> None:
                try:
                    self._pinned_ab.clear()
                except Exception:
                    pass

        setattr(disk_mod, "LoraSource", FrameVisionRuntimeLoraSource)
        try:
            setattr(builder_mod, "LoraSource", FrameVisionRuntimeLoraSource)
        except Exception:
            pass

        ctx["ltx_user_lora_runtime_source_patch"] = "YES: FrameVision CUDA-resident/non-pinned runtime LoraSource"
        ctx.setdefault("notes", []).append(
            "Installed FrameVision runtime LoRA source: user LoRA A/B tensors prefer CUDA dedicated VRAM; fallback is ordinary CPU, not pinned shared-memory staging."
        )
        _record_user_lora_event(ctx, "runtime_source", ctx["ltx_user_lora_runtime_source_patch"], torch, echo)
    except Exception as exc:
        ctx["ltx_user_lora_runtime_source_patch"] = f"FAILED patch: {type(exc).__name__}: {exc}"
        ctx["ltx_user_lora_runtime_source_errors"] = ctx["ltx_user_lora_runtime_source_patch"]
        _record_user_lora_event(ctx, "runtime_source", ctx["ltx_user_lora_runtime_source_patch"], torch_module, echo)



def _install_ltx_user_lora_forward_hook_patch(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Apply user LoRA 1-4 as runtime Linear hooks instead of official fusion.

    This is the FrameVision path for normal user LoRAs.  The local LTX 2.3 repo's
    normal ``--lora`` path reaches SingleGPUModelBuilder.apply_loras /
    fuse_lora_weights, which materializes hundreds of full transformer deltas and
    hammers NVMe/RAM.  The repo's LoraSource path is only reached when the main
    transformer uses the official StreamingModelBuilder; our VRAM Lab route uses
    the direct-load + BatchSplitAdapter path, so LoraSource was never instantiated.

    This patch therefore intercepts DiffusionStage before it gives LoRAs to the
    official builders, stores them on the stage, and attaches temporary forward
    hooks to matching Linear modules for the duration of each denoise call.
    """
    ctx["ltx_user_lora_runtime_forward_patch"] = "NO"
    ctx["ltx_user_lora_runtime_forward_status"] = "not installed"
    ctx["ltx_user_lora_runtime_forward_loras"] = "0"
    ctx["ltx_user_lora_runtime_forward_targets"] = "0"
    ctx["ltx_user_lora_runtime_forward_pairs"] = "0"
    ctx["ltx_user_lora_runtime_forward_bytes"] = "0 B"
    ctx["ltx_user_lora_runtime_forward_residency"] = "n/a"
    ctx["ltx_user_lora_runtime_forward_errors"] = "none"
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        import safetensors  # type: ignore
        import types as _types
        blocks_mod = importlib.import_module("ltx_pipelines.utils.blocks")
        DiffusionStage = getattr(blocks_mod, "DiffusionStage")
    except Exception as exc:
        ctx["ltx_user_lora_runtime_forward_patch"] = f"FAILED import: {type(exc).__name__}: {exc}"
        ctx["ltx_user_lora_runtime_forward_errors"] = ctx["ltx_user_lora_runtime_forward_patch"]
        _record_user_lora_event(ctx, "runtime_forward", ctx["ltx_user_lora_runtime_forward_patch"], torch_module, echo)
        return

    def _tensor_bytes(t: Any) -> int:
        try:
            if torch.is_tensor(t):
                return int(t.numel()) * int(t.element_size())
        except Exception:
            pass
        return 0

    def _strip_known_prefixes(name: str) -> list[str]:
        raw = str(name or "")
        names = {raw}
        for prefix in (
            "_model.",
            "model.",
            "_model.velocity_model.",
            "model.velocity_model.",
            "velocity_model.",
            "_orig_mod.",
            "module.",
        ):
            if raw.startswith(prefix):
                names.add(raw[len(prefix):])
        # Also make suffix candidates from the first transformer_blocks occurrence.
        marker = "transformer_blocks."
        if marker in raw:
            names.add(raw[raw.index(marker):])
        return [n for n in names if n]

    def _module_candidates(name: str) -> set[str]:
        cands = set(_strip_known_prefixes(name))
        more = set()
        for n in list(cands):
            more.add(n.replace(".0.", "."))
        cands |= more
        return cands

    def _load_runtime_lora_pairs(loras: Any, device: Any, dtype: Any) -> tuple[dict[str, list[tuple[Any, Any, float]]], int, int, str]:
        by_prefix: dict[str, list[tuple[Any, Any, float]]] = {}
        total_bytes = 0
        pair_count = 0
        residency = "cuda"
        lora_items = list(loras or [])
        for item in lora_items:
            path = str(getattr(item, "path", item))
            strength = float(getattr(item, "strength", 1.0) or 1.0)
            sd_ops = getattr(item, "sd_ops", None)
            a_keys: dict[str, str] = {}
            b_keys: dict[str, str] = {}
            with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                for sft_key in handle.keys():
                    model_key = sd_ops.apply_to_key(sft_key) if sd_ops is not None else sft_key
                    if model_key is None:
                        continue
                    model_key = str(model_key)
                    if model_key.endswith(".lora_A.weight"):
                        a_keys[model_key[:-len(".lora_A.weight")]] = sft_key
                    elif model_key.endswith(".lora_B.weight"):
                        b_keys[model_key[:-len(".lora_B.weight")]] = sft_key
                prefixes = sorted(a_keys.keys() & b_keys.keys())
                # Header estimate first, so we can decide whether CUDA residency is safe.
                estimated = 0
                for prefix in prefixes:
                    try:
                        a_slice = handle.get_slice(a_keys[prefix])
                        b_slice = handle.get_slice(b_keys[prefix])
                        # LoRAs are normally fp16/bf16/fp32.  Use 2 bytes unless dtype says otherwise.
                        def _slice_bytes(s: Any) -> int:
                            shape = s.get_shape()
                            dt = str(s.get_dtype()).upper()
                            item = 4 if dt in {"F32", "I32"} else 8 if dt in {"F64", "I64"} else 2
                            n = 1
                            for dim in shape:
                                n *= int(dim)
                            return int(n) * item
                        estimated += _slice_bytes(a_slice) + _slice_bytes(b_slice)
                    except Exception:
                        pass
                use_cuda = False
                try:
                    if torch.cuda.is_available() and device is not None and str(device).startswith("cuda"):
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
                        headroom = int(float(os.environ.get("FRAMEVISION_LTX_USER_LORA_RUNTIME_HEADROOM_GB", "0.75") or "0.75") * (1024**3))
                        use_cuda = int(free_bytes) > int(estimated) + headroom
                except Exception:
                    use_cuda = False
                if not use_cuda:
                    residency = "cpu_unpinned"
                for prefix in prefixes:
                    a = handle.get_tensor(a_keys[prefix])
                    b = handle.get_tensor(b_keys[prefix])
                    total_bytes += _tensor_bytes(a) + _tensor_bytes(b)
                    if use_cuda:
                        a = a.to(device=device, dtype=dtype, non_blocking=False)
                        b = b.to(device=device, dtype=dtype, non_blocking=False)
                    else:
                        # CPU fallback is intentionally normal pageable CPU, not pinned.
                        if dtype is not None:
                            a = a.to(dtype=dtype)
                            b = b.to(dtype=dtype)
                    by_prefix.setdefault(prefix, []).append((a, b, strength))
                    pair_count += 1
        if residency == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        return by_prefix, pair_count, total_bytes, residency

    def _attach_runtime_lora_hooks(model: Any, loras: Any, device: Any, dtype: Any) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        lora_items = tuple(loras or ())
        if not lora_items:
            return [], {"targets": 0, "pairs": 0, "bytes": 0, "residency": "none"}
        by_prefix, pair_count, total_bytes, residency = _load_runtime_lora_pairs(lora_items, device, dtype)
        if not by_prefix:
            return [], {"targets": 0, "pairs": 0, "bytes": total_bytes, "residency": residency}

        handles: list[tuple[Any, Any]] = []
        targets = 0
        import torch.nn.functional as F  # type: ignore
        for module_name, module in model.named_modules():
            try:
                if not isinstance(module, torch.nn.Linear):
                    continue
            except Exception:
                continue
            candidates = _module_candidates(module_name)
            matched_prefix = None
            for cand in candidates:
                if cand in by_prefix:
                    matched_prefix = cand
                    break
            if matched_prefix is None:
                # Last resort suffix match for wrappers with unknown prefixes.
                for prefix in by_prefix:
                    if any(cand.endswith("." + prefix) for cand in candidates):
                        matched_prefix = prefix
                        break
            if matched_prefix is None:
                continue

            lora_pairs = list(by_prefix.get(matched_prefix, []))
            if not lora_pairs:
                continue
            original_forward = module.forward

            def _make_forward(_orig: Any, _pairs: list[tuple[Any, Any, float]]) -> Any:
                def _fv_lora_forward(self: Any, input: Any) -> Any:
                    out = _orig(input)
                    try:
                        delta = None
                        in_device = getattr(input, "device", None)
                        in_dtype = getattr(input, "dtype", None)
                        for a, b, strength in _pairs:
                            aa = a
                            bb = b
                            if in_device is not None and getattr(aa, "device", None) != in_device:
                                aa = aa.to(device=in_device, non_blocking=False)
                                bb = bb.to(device=in_device, non_blocking=False)
                            if in_dtype is not None and getattr(aa, "dtype", None) != in_dtype:
                                aa = aa.to(dtype=in_dtype)
                                bb = bb.to(dtype=in_dtype)
                            part = F.linear(F.linear(input, aa), bb) * float(strength)
                            delta = part if delta is None else delta + part
                        if delta is not None:
                            out = out + delta
                    except Exception as exc:
                        # Fail loudly: silent no-op LoRA is worse than a clean abort.
                        raise RuntimeError(f"FrameVision runtime LoRA forward failed in {type(self).__name__}: {type(exc).__name__}: {exc}") from exc
                    return out
                return _fv_lora_forward

            try:
                module.forward = _types.MethodType(_make_forward(original_forward, lora_pairs), module)
                handles.append((module, original_forward))
                targets += 1
            except Exception:
                continue

        return handles, {"targets": targets, "pairs": pair_count, "bytes": total_bytes, "residency": residency}

    try:
        original_init = getattr(DiffusionStage, "_framevision_runtime_lora_original_init", None)
        if original_init is None:
            original_init = getattr(DiffusionStage, "__init__")
            setattr(DiffusionStage, "_framevision_runtime_lora_original_init", original_init)

            def _fv_diffusion_stage_init(self: Any, checkpoint_path: str, dtype: Any, device: Any, loras: Any = (), quantization: Any = None, registry: Any = None, torch_compile: bool = False, offload_mode: Any = None, transformer_builder: Any = None) -> None:
                user_loras = tuple(loras or ())
                ctx["ltx_user_lora_runtime_forward_loras"] = str(int(ctx.get("ltx_user_lora_runtime_forward_loras", "0") or "0") + len(user_loras))
                if user_loras:
                    _record_user_lora_event(ctx, "runtime_forward:capture", f"captured {len(user_loras)} LoRA(s) for DiffusionStage; official builder loras stripped", torch, echo)
                original_init(
                    self,
                    checkpoint_path=checkpoint_path,
                    dtype=dtype,
                    device=device,
                    loras=(),
                    quantization=quantization,
                    registry=registry,
                    torch_compile=torch_compile,
                    offload_mode=offload_mode,
                    transformer_builder=transformer_builder,
                )
                self._framevision_runtime_loras = user_loras

            setattr(DiffusionStage, "__init__", _fv_diffusion_stage_init)

        original_run = getattr(DiffusionStage, "_framevision_runtime_lora_original_run", None)
        if original_run is None:
            original_run = getattr(DiffusionStage, "run")
            setattr(DiffusionStage, "_framevision_runtime_lora_original_run", original_run)

            def _fv_diffusion_stage_run(self: Any, transformer: Any, *run_args: Any, **run_kwargs: Any) -> Any:
                loras = tuple(getattr(self, "_framevision_runtime_loras", ()) or ())
                handles: list[tuple[Any, Any]] = []
                if loras:
                    try:
                        device = getattr(self, "_device", None)
                        dtype = getattr(self, "_dtype", None)
                        handles, stats = _attach_runtime_lora_hooks(transformer, loras, device, dtype)
                        ctx["ltx_user_lora_runtime_forward_targets"] = str(stats.get("targets", 0))
                        ctx["ltx_user_lora_runtime_forward_pairs"] = str(stats.get("pairs", 0))
                        ctx["ltx_user_lora_runtime_forward_bytes"] = _fmt_bytes(int(stats.get("bytes", 0) or 0))
                        ctx["ltx_user_lora_runtime_forward_residency"] = str(stats.get("residency", "unknown"))
                        ctx["ltx_user_lora_runtime_forward_status"] = f"attached {stats.get('targets', 0)} Linear runtime LoRA target(s)"
                        _record_user_lora_event(
                            ctx,
                            "runtime_forward:attach",
                            f"targets={stats.get('targets', 0)}; pairs={stats.get('pairs', 0)}; bytes={_fmt_bytes(int(stats.get('bytes', 0) or 0))}; residency={stats.get('residency', 'unknown')}",
                            torch,
                            echo,
                        )
                        if int(stats.get("targets", 0) or 0) <= 0:
                            raise RuntimeError("FrameVision runtime LoRA found no matching Linear modules. Refusing to continue with silent no-op LoRA.")
                    except Exception as exc:
                        ctx["ltx_user_lora_runtime_forward_errors"] = f"{type(exc).__name__}: {exc}"
                        _record_user_lora_event(ctx, "runtime_forward:error", ctx["ltx_user_lora_runtime_forward_errors"], torch, echo)
                        raise
                try:
                    return original_run(self, transformer, *run_args, **run_kwargs)
                finally:
                    for module, original_forward in handles:
                        try:
                            module.forward = original_forward
                        except Exception:
                            pass
                    if handles:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            setattr(DiffusionStage, "run", _fv_diffusion_stage_run)

        ctx["ltx_user_lora_runtime_forward_patch"] = "YES: DiffusionStage runtime Linear forward hooks; official builder LoRAs stripped"
        ctx["ltx_user_lora_runtime_forward_status"] = "installed"
        _record_user_lora_event(ctx, "runtime_forward", ctx["ltx_user_lora_runtime_forward_patch"], torch, echo)
    except Exception as exc:
        ctx["ltx_user_lora_runtime_forward_patch"] = f"FAILED patch: {type(exc).__name__}: {exc}"
        ctx["ltx_user_lora_runtime_forward_errors"] = ctx["ltx_user_lora_runtime_forward_patch"]
        _record_user_lora_event(ctx, "runtime_forward", ctx["ltx_user_lora_runtime_forward_patch"], torch, echo)


def _install_ltx_user_lora_runtime_logger(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Logs-only tracer for real user LoRA loading/application.

    This is separate from the old distilled-LoRA/fusion-cache machinery. It is
    installed even when the official distilled LoRA path is disabled, so a normal
    user --lora can be proven visible or invisible in the report.
    """
    ctx["ltx_user_lora_runtime_logger"] = "NO"
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        sgb = importlib.import_module("ltx_core.loader.single_gpu_model_builder")
        try:
            fuse_mod = importlib.import_module("ltx_core.loader.fuse_loras")
        except Exception:
            fuse_mod = None
    except Exception as exc:
        ctx["ltx_user_lora_runtime_logger"] = f"FAILED import: {type(exc).__name__}: {exc}"
        _record_user_lora_event(ctx, "runtime_logger", ctx["ltx_user_lora_runtime_logger"], torch_module, echo)
        return

    counters = ctx.setdefault("_ltx_user_lora_counters", {})
    if not isinstance(counters, dict):
        counters = {}
        ctx["_ltx_user_lora_counters"] = counters

    try:
        original_lmw = getattr(sgb, "_framevision_user_lora_original_load_model_weights", None)
        if original_lmw is None:
            original_lmw = getattr(sgb, "_load_model_weights")
            setattr(sgb, "_framevision_user_lora_original_load_model_weights", original_lmw)

            def _fv_user_lora_load_model_weights(meta_model: Any, model_path: Any, loras: Any, loader: Any, registry: Any, device: Any, dtype: Any, model_sd_ops: Any, lora_load_device: Any = None) -> Any:
                lora_tuple = tuple(loras or ())
                if lora_tuple:
                    counters["load_model_weights_calls"] = int(counters.get("load_model_weights_calls", 0)) + 1
                    ctx["ltx_user_lora_load_model_weights_calls"] = str(counters["load_model_weights_calls"])
                    summaries: List[str] = []
                    for idx, item in enumerate(lora_tuple, 1):
                        path = getattr(item, "path", item)
                        strength = getattr(item, "strength", "n/a")
                        exists, size, norm_path = _safe_file_size_text(path)
                        summaries.append(f"slot {idx}: exists={'YES' if exists else 'NO'}; size={size}; strength={strength}; path={norm_path}")
                    detail = f"model={type(meta_model).__module__}.{type(meta_model).__name__}; device={device}; dtype={dtype}; loras={len(lora_tuple)}; " + " | ".join(summaries)
                    ctx["ltx_user_lora_runtime_status"] = "load_model_weights saw user LoRA"
                    _record_user_lora_event(ctx, "load_model_weights:start", detail, torch, echo)

                    # Safety brake: if user LoRA reaches SingleGPUModelBuilder for the
                    # main LTX transformer while the fast-stream path was requested,
                    # something failed before the streaming builder took over.  Do NOT
                    # continue into apply_loras/fuse_lora_weights, because that is the
                    # exact NVMe/DDR storm path seen in the failed run.
                    try:
                        if os.environ.get("FRAMEVISION_LTX_USER_LORA_FAST_STREAM_REQUIRED", "0") == "1":
                            model_name = f"{type(meta_model).__module__}.{type(meta_model).__name__}"
                            if "ltx_core.model.transformer" in model_name or model_name.endswith(".LTXModel"):
                                ctx["ltx_user_lora_runtime_status"] = "ABORTED: user LoRA reached SingleGPU full-fuse path"
                                _record_user_lora_event(ctx, "safety_abort", "user LoRA reached SingleGPUModelBuilder for main transformer; refused apply_loras/fuse_lora_weights to protect NVMe/RAM", torch, echo)
                                raise RuntimeError("FrameVision user LoRA fast path failed before streaming builder. Refusing official SingleGPU apply_loras/fuse_lora_weights path to avoid huge NVMe/RAM work. Check main_transformer_streaming_probe/report and upload logs.")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass

                    t0 = time.perf_counter()
                    try:
                        return original_lmw(meta_model, model_path, loras, loader, registry, device, dtype, model_sd_ops, lora_load_device)
                    finally:
                        _record_user_lora_event(ctx, "load_model_weights:end", f"duration={time.perf_counter() - t0:.3f}s; {detail}", torch, echo)
                return original_lmw(meta_model, model_path, loras, loader, registry, device, dtype, model_sd_ops, lora_load_device)

            setattr(sgb, "_load_model_weights", _fv_user_lora_load_model_weights)
    except Exception as exc:
        ctx["ltx_user_lora_runtime_logger"] = f"FAILED _load_model_weights patch: {type(exc).__name__}: {exc}"
        _record_user_lora_event(ctx, "runtime_logger", ctx["ltx_user_lora_runtime_logger"], torch_module, echo)
        return

    if fuse_mod is not None:
        try:
            original_apply = getattr(fuse_mod, "_framevision_user_lora_original_apply_loras", None)
            if original_apply is None and hasattr(fuse_mod, "apply_loras"):
                original_apply = getattr(fuse_mod, "apply_loras")
                setattr(fuse_mod, "_framevision_user_lora_original_apply_loras", original_apply)

                def _fv_user_lora_apply_loras(*call_args: Any, **call_kwargs: Any) -> Any:
                    counters["apply_loras_calls"] = int(counters.get("apply_loras_calls", 0)) + 1
                    ctx["ltx_user_lora_apply_loras_calls"] = str(counters["apply_loras_calls"])
                    try:
                        lora_sd_and_strengths = call_kwargs.get("lora_sd_and_strengths", call_args[1] if len(call_args) > 1 else None)
                        lora_len = len(lora_sd_and_strengths or [])
                    except Exception:
                        lora_len = -1
                    ctx["ltx_user_lora_runtime_status"] = "apply_loras called"
                    _record_user_lora_event(ctx, "apply_loras:start", f"lora_state_dicts={lora_len}", torch, echo)
                    t0 = time.perf_counter()
                    try:
                        return original_apply(*call_args, **call_kwargs)
                    finally:
                        _record_user_lora_event(ctx, "apply_loras:end", f"duration={time.perf_counter() - t0:.3f}s; lora_state_dicts={lora_len}", torch, echo)

                setattr(fuse_mod, "apply_loras", _fv_user_lora_apply_loras)
                try:
                    setattr(sgb, "apply_loras", _fv_user_lora_apply_loras)
                except Exception:
                    pass
        except Exception as exc:
            ctx.setdefault("notes", []).append(f"User LoRA apply_loras logger failed: {type(exc).__name__}: {exc}")

        try:
            original_fuse = getattr(fuse_mod, "_framevision_user_lora_original_fuse_lora_weights", None)
            if original_fuse is None and hasattr(fuse_mod, "fuse_lora_weights"):
                original_fuse = getattr(fuse_mod, "fuse_lora_weights")
                setattr(fuse_mod, "_framevision_user_lora_original_fuse_lora_weights", original_fuse)

                def _fv_user_lora_fuse_lora_weights(*call_args: Any, **call_kwargs: Any):
                    counters["fuse_lora_weights_calls"] = int(counters.get("fuse_lora_weights_calls", 0)) + 1
                    ctx["ltx_user_lora_fuse_lora_weights_calls"] = str(counters["fuse_lora_weights_calls"])
                    _record_user_lora_event(ctx, "fuse_lora_weights:start", "iterating modified LoRA weights", torch, echo)
                    count = 0
                    t0 = time.perf_counter()
                    try:
                        for item in original_fuse(*call_args, **call_kwargs):
                            count += 1
                            if count <= 3 or count % 50 == 0:
                                try:
                                    key = str(item[0])
                                except Exception:
                                    key = "?"
                                _record_user_lora_event(ctx, "fuse_lora_weights:progress", f"modified_weights={count}; current_key={key}", torch, echo)
                            yield item
                    finally:
                        _record_user_lora_event(ctx, "fuse_lora_weights:end", f"duration={time.perf_counter() - t0:.3f}s; modified_weights={count}", torch, echo)

                setattr(fuse_mod, "fuse_lora_weights", _fv_user_lora_fuse_lora_weights)
        except Exception as exc:
            ctx.setdefault("notes", []).append(f"User LoRA fuse_lora_weights logger failed: {type(exc).__name__}: {exc}")

    ctx["ltx_user_lora_runtime_logger"] = "YES"
    _record_user_lora_event(ctx, "runtime_logger", "installed logs-only user LoRA tracer", torch_module, echo)


def _safe_module_import_status(module_name: str) -> str:
    """Return a compact import/version status for optional kernel packages.

    Diagnostic only: importing a package here does not mean LTX will use it.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return f"FAILED find_spec: {type(exc).__name__}: {exc}"
    if spec is None:
        return "MISSING"
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None) or "unknown origin"
        return f"OK" + (f" version={version}" if version else "") + f" origin={origin}"
    except Exception as exc:
        origin = getattr(spec, "origin", None) or "unknown origin"
        return f"FOUND but import failed: {type(exc).__name__}: {exc} origin={origin}"


def _torch_sdp_status(torch_module: Any) -> str:
    try:
        cuda_backends = getattr(getattr(torch_module, "backends", None), "cuda", None)
        if cuda_backends is None:
            return "torch.backends.cuda unavailable"
        parts = []
        for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
            fn = getattr(cuda_backends, name, None)
            if callable(fn):
                try:
                    parts.append(f"{name}={fn()}")
                except Exception as exc:
                    parts.append(f"{name}=FAILED({type(exc).__name__}: {exc})")
        return ", ".join(parts) if parts else "no known SDP status helpers found"
    except Exception as exc:
        return f"FAILED: {type(exc).__name__}: {exc}"


def _probe_ltx_attention_module(ctx: Dict[str, Any], torch_module: Any) -> None:
    """Inspect optional kernels and official LTX attention backend state.

    This is intentionally read-only. It does not select/force any attention backend.
    """
    ctx["kernel_probe_flash_attn"] = _safe_module_import_status("flash_attn")
    ctx["kernel_probe_flash_attn_interface"] = _safe_module_import_status("flash_attn_interface")
    ctx["kernel_probe_triton"] = _safe_module_import_status("triton")
    ctx["kernel_probe_xformers"] = _safe_module_import_status("xformers")
    ctx["kernel_probe_xformers_ops"] = _safe_module_import_status("xformers.ops")
    ctx["kernel_probe_sageattention"] = _safe_module_import_status("sageattention")
    ctx["kernel_probe_torch_sdp"] = _torch_sdp_status(torch_module)
    try:
        ctx["kernel_probe_torch_version"] = f"{getattr(torch_module, '__version__', '?')} cuda={getattr(getattr(torch_module, 'version', None), 'cuda', '?')}"
        if getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
            ctx["kernel_probe_gpu"] = f"{torch_module.cuda.get_device_name(0)} capability={torch_module.cuda.get_device_capability(0)}"
        else:
            ctx["kernel_probe_gpu"] = "torch.cuda.is_available() is False"
    except Exception as exc:
        ctx["kernel_probe_gpu"] = f"FAILED: {type(exc).__name__}: {exc}"

    try:
        attn_mod = importlib.import_module("ltx_core.model.transformer.attention")
        mem_eff = getattr(attn_mod, "memory_efficient_attention", None)
        flash3 = getattr(attn_mod, "flash_attn_interface", None)
        default_callable = "unknown"
        try:
            default_callable = type(attn_mod.AttentionFunction.DEFAULT.to_callable()).__name__
        except Exception as exc:
            default_callable = f"FAILED resolving default: {type(exc).__name__}: {exc}"
        ctx["ltx_attention_module_file"] = str(getattr(attn_mod, "__file__", "unknown"))
        ctx["ltx_attention_xformers_available"] = "YES" if mem_eff is not None else "NO"
        ctx["ltx_attention_flash3_interface_available"] = "YES" if flash3 is not None else "NO"
        ctx["ltx_attention_default_callable"] = str(default_callable)
        ctx["ltx_attention_flash_attn_v2_note"] = (
            "normal flash_attn package is not used by official LTX attention.py; "
            "official FlashAttention path expects flash_attn_interface / FlashAttention3"
        )
    except Exception as exc:
        ctx["ltx_attention_module_file"] = f"FAILED: {type(exc).__name__}: {exc}"
        ctx["ltx_attention_xformers_available"] = "unknown"
        ctx["ltx_attention_flash3_interface_available"] = "unknown"
        ctx["ltx_attention_default_callable"] = "unknown"
        ctx["ltx_attention_flash_attn_v2_note"] = "unknown"



def _install_ltx_attention_backend_override(ctx: Dict[str, Any], requested_backend: str) -> None:
    """Experimental runtime attention backend override for official LTX.

    Wrapper-side only; does not edit the official repo.
    auto tries SageAttention first, then FlashAttention2, then PyTorch SDPA.
    sdpa/pytorch forces PyTorch SDPA.
    flash2/sage install safe per-call wrappers that fall back to PyTorch SDPA.
    """
    backend = str(requested_backend or "auto").strip().lower()
    if backend == "pytorch":
        backend = "sdpa"
    ctx["ltx_attention_backend_requested"] = backend
    ctx["ltx_attention_backend_override_status"] = "not attempted"
    ctx["ltx_attention_backend_selected"] = "official default"

    ctx["ltx_flash2_import_status"] = "not attempted"
    ctx["ltx_flash2_success_calls"] = "0"
    ctx["ltx_flash2_fallback_calls"] = "0"
    ctx["ltx_flash2_first_fallback_error"] = "none"
    ctx["ltx_flash2_last_fallback_reason"] = "none"

    ctx["ltx_sage_import_status"] = "not attempted"
    ctx["ltx_sage_success_calls"] = "0"
    ctx["ltx_sage_fallback_calls"] = "0"
    ctx["ltx_sage_first_fallback_error"] = "none"
    ctx["ltx_sage_last_fallback_reason"] = "none"

    try:
        attn_mod = importlib.import_module("ltx_core.model.transformer.attention")
        Attention = getattr(attn_mod, "Attention", None)
        AttentionFunction = getattr(attn_mod, "AttentionFunction", None)
        PytorchAttention = getattr(attn_mod, "PytorchAttention", None)
        if Attention is None or AttentionFunction is None or PytorchAttention is None:
            ctx["ltx_attention_backend_override_status"] = "FAILED: official Attention/PytorchAttention classes not found"
            return

        sageattn = None
        flash_attn_func = None

        if backend in ("auto", "sage"):
            try:
                from sageattention import sageattn as _sageattn  # type: ignore
                sageattn = _sageattn
                ctx["ltx_sage_import_status"] = "OK: from sageattention import sageattn"
            except Exception as exc:
                ctx["ltx_sage_import_status"] = f"FAILED: {type(exc).__name__}: {exc}"

        if backend in ("auto", "flash2") and sageattn is None:
            try:
                from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
                flash_attn_func = _flash_attn_func
                ctx["ltx_flash2_import_status"] = "OK: from flash_attn import flash_attn_func"
            except Exception as exc:
                ctx["ltx_flash2_import_status"] = f"FAILED: {type(exc).__name__}: {exc}"
        elif backend == "flash2":
            try:
                from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
                flash_attn_func = _flash_attn_func
                ctx["ltx_flash2_import_status"] = "OK: from flash_attn import flash_attn_func"
            except Exception as exc:
                ctx["ltx_flash2_import_status"] = f"FAILED: {type(exc).__name__}: {exc}"

        selected_backend = backend
        if backend == "auto":
            if sageattn is not None:
                selected_backend = "sage"
            elif flash_attn_func is not None:
                selected_backend = "flash2"
            else:
                selected_backend = "sdpa"
        elif backend == "sage" and sageattn is None:
            ctx["ltx_attention_backend_override_status"] = "sage requested but import failed; falling back to PyTorch SDPA"
            selected_backend = "sdpa"
        elif backend == "flash2" and flash_attn_func is None:
            ctx["ltx_attention_backend_override_status"] = "flash2 requested but import failed; falling back to PyTorch SDPA"
            selected_backend = "sdpa"

        if selected_backend == "sdpa":
            original_init = getattr(Attention, "_framevision_attention_backend_original_init", None)
            if original_init is None:
                original_init = Attention.__init__

                def _fv_pytorch_attention_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    kwargs["attention_function"] = AttentionFunction.PYTORCH
                    original_init(self, *args, **kwargs)

                setattr(Attention, "_framevision_attention_backend_original_init", original_init)
                Attention.__init__ = _fv_pytorch_attention_init
            ctx["ltx_attention_backend_selected"] = "sdpa"
            if ctx.get("ltx_attention_backend_override_status") in ("not attempted", ""):
                ctx["ltx_attention_backend_override_status"] = "installed: Attention.__init__ forces AttentionFunction.PYTORCH"
            ctx.setdefault("notes", []).append("Installed LTX attention backend override: sdpa (official PyTorch SDPA path).")
            return

        class FrameVisionSageAttention:
            def __init__(self) -> None:
                self.fallback = PytorchAttention()

            def __call__(self, q, k, v, heads: int, mask=None):  # type: ignore[no-untyped-def]
                def _fallback(reason: str):
                    try:
                        ctx["ltx_sage_fallback_calls"] = str(int(ctx.get("ltx_sage_fallback_calls", "0")) + 1)
                    except Exception:
                        ctx["ltx_sage_fallback_calls"] = "1"
                    ctx["ltx_sage_last_fallback_reason"] = str(reason)
                    if ctx.get("ltx_sage_first_fallback_error", "none") == "none":
                        ctx["ltx_sage_first_fallback_error"] = str(reason)
                    return self.fallback(q, k, v, heads, mask)

                try:
                    if sageattn is None:
                        return _fallback("sageattn callable is not available")
                    if mask is not None:
                        return _fallback("mask is not supported by this first SageAttention wrapper")
                    if not getattr(q, "is_cuda", False) or not getattr(k, "is_cuda", False) or not getattr(v, "is_cuda", False):
                        return _fallback("q/k/v are not all CUDA tensors")
                    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
                        return _fallback(f"expected rank-3 q/k/v, got {getattr(q, 'shape', '?')} {getattr(k, 'shape', '?')} {getattr(v, 'shape', '?')}")
                    b, _, inner = q.shape
                    if int(heads) <= 0 or int(inner) % int(heads) != 0:
                        return _fallback(f"inner dim {inner} not divisible by heads {heads}")
                    dim_head = int(inner) // int(heads)
                    q4 = q.view(b, -1, int(heads), dim_head)
                    k4 = k.view(b, -1, int(heads), dim_head)
                    v4 = v.view(b, -1, int(heads), dim_head)
                    # SageAttention expects either NHD [B, S, H, D] or HND [B, H, S, D].
                    # Use NHD to match the Flash2 wrapper layout. Fall back instead of forcing quality-risky conversions.
                    try:
                        out = sageattn(q4.to(v.dtype), k4.to(v.dtype), v4, tensor_layout="NHD", is_causal=False)
                    except TypeError:
                        out = sageattn(q4.to(v.dtype), k4.to(v.dtype), v4, is_causal=False)
                    out = out.reshape(b, -1, int(heads) * dim_head)
                    try:
                        ctx["ltx_sage_success_calls"] = str(int(ctx.get("ltx_sage_success_calls", "0")) + 1)
                    except Exception:
                        ctx["ltx_sage_success_calls"] = "1"
                    return out
                except Exception as exc:
                    return _fallback(f"{type(exc).__name__}: {exc}")

        class FrameVisionFlashAttention2:
            def __init__(self) -> None:
                self.fallback = PytorchAttention()

            def __call__(self, q, k, v, heads: int, mask=None):  # type: ignore[no-untyped-def]
                def _fallback(reason: str):
                    try:
                        ctx["ltx_flash2_fallback_calls"] = str(int(ctx.get("ltx_flash2_fallback_calls", "0")) + 1)
                    except Exception:
                        ctx["ltx_flash2_fallback_calls"] = "1"
                    ctx["ltx_flash2_last_fallback_reason"] = str(reason)
                    if ctx.get("ltx_flash2_first_fallback_error", "none") == "none":
                        ctx["ltx_flash2_first_fallback_error"] = str(reason)
                    return self.fallback(q, k, v, heads, mask)

                try:
                    if flash_attn_func is None:
                        return _fallback("flash_attn_func callable is not available")
                    if mask is not None:
                        return _fallback("mask is not supported by this first FlashAttention2 wrapper")
                    if not getattr(q, "is_cuda", False) or not getattr(k, "is_cuda", False) or not getattr(v, "is_cuda", False):
                        return _fallback("q/k/v are not all CUDA tensors")
                    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
                        return _fallback(f"expected rank-3 q/k/v, got {getattr(q, 'shape', '?')} {getattr(k, 'shape', '?')} {getattr(v, 'shape', '?')}")
                    b, _, inner = q.shape
                    if int(heads) <= 0 or int(inner) % int(heads) != 0:
                        return _fallback(f"inner dim {inner} not divisible by heads {heads}")
                    dim_head = int(inner) // int(heads)
                    q4 = q.view(b, -1, int(heads), dim_head)
                    k4 = k.view(b, -1, int(heads), dim_head)
                    v4 = v.view(b, -1, int(heads), dim_head)
                    out = flash_attn_func(q4.to(v.dtype), k4.to(v.dtype), v4, dropout_p=0.0, causal=False)
                    out = out.reshape(b, -1, int(heads) * dim_head)
                    try:
                        ctx["ltx_flash2_success_calls"] = str(int(ctx.get("ltx_flash2_success_calls", "0")) + 1)
                    except Exception:
                        ctx["ltx_flash2_success_calls"] = "1"
                    return out
                except Exception as exc:
                    return _fallback(f"{type(exc).__name__}: {exc}")

        original_init = getattr(Attention, "_framevision_attention_backend_original_init", None)
        if original_init is None:
            original_init = Attention.__init__

            def _fv_attention_backend_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_init(self, *args, **kwargs)
                try:
                    if selected_backend == "sage":
                        self.attention_function = FrameVisionSageAttention()
                    elif selected_backend == "flash2":
                        self.attention_function = FrameVisionFlashAttention2()
                except Exception as exc:
                    ctx["ltx_attention_backend_override_status"] = f"FAILED replacing attention_function: {type(exc).__name__}: {exc}"

            setattr(Attention, "_framevision_attention_backend_original_init", original_init)
            Attention.__init__ = _fv_attention_backend_init
            ctx["ltx_attention_backend_override_status"] = f"installed: Attention.__init__ replaces callable with FrameVision{selected_backend} safe fallback"
        else:
            ctx["ltx_attention_backend_override_status"] = "already installed or another wrapper was present; did not replace existing wrapper"

        ctx["ltx_attention_backend_selected"] = selected_backend
        if selected_backend == "sage":
            ctx.setdefault("notes", []).append(
                "Installed experimental LTX SageAttention wrapper: uses installed sageattention when possible, falls back to PytorchAttention per call."
            )
        elif selected_backend == "flash2":
            ctx.setdefault("notes", []).append(
                "Installed experimental LTX FlashAttention2 wrapper: uses installed flash_attn v2 when possible, falls back to PytorchAttention per call."
            )
    except Exception as exc:
        ctx["ltx_attention_backend_override_status"] = f"FAILED: {type(exc).__name__}: {exc}"

def _install_ltx_attention_init_probe(ctx: Dict[str, Any]) -> None:
    """Patch LTX Attention.__init__ only to record the actual callable class.

    This is a diagnostic shim. It does not change the selected backend.
    """
    try:
        attn_mod = importlib.import_module("ltx_core.model.transformer.attention")
        Attention = getattr(attn_mod, "Attention", None)
        if Attention is None:
            ctx["ltx_attention_init_probe_status"] = "FAILED: Attention class not found"
            return
        original = getattr(Attention, "_framevision_attention_probe_original_init", None)
        if original is not None:
            ctx["ltx_attention_init_probe_status"] = "already installed"
            return
        original = Attention.__init__
        counts: Dict[str, int] = {}
        examples: List[str] = []

        def _fv_attention_probe_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            original(self, *args, **kwargs)
            try:
                fn = getattr(self, "attention_function", None)
                cls_name = type(fn).__name__ if fn is not None else "None"
                counts[cls_name] = int(counts.get(cls_name, 0)) + 1
                if len(examples) < 12:
                    heads = getattr(self, "heads", "?")
                    dim_head = getattr(self, "dim_head", "?")
                    examples.append(f"{cls_name}(heads={heads}, dim_head={dim_head})")
                ctx["ltx_attention_runtime_callable_counts"] = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
                ctx["ltx_attention_runtime_callable_examples"] = " | ".join(examples) if examples else "none"
            except Exception as exc:
                ctx["ltx_attention_init_probe_runtime_error"] = f"{type(exc).__name__}: {exc}"

        setattr(Attention, "_framevision_attention_probe_original_init", original)
        Attention.__init__ = _fv_attention_probe_init
        ctx["ltx_attention_init_probe_status"] = "installed: Attention.__init__ records callable class only; no backend override"
        ctx["ltx_attention_runtime_callable_counts"] = "none yet"
        ctx["ltx_attention_runtime_callable_examples"] = "none yet"
        ctx.setdefault("notes", []).append(
            "Installed LTX attention init probe: records actual Attention attention_function callable class; does not force Flash/Sage/xFormers."
        )
    except Exception as exc:
        ctx["ltx_attention_init_probe_status"] = f"FAILED: {type(exc).__name__}: {exc}"



def _ram_snapshot() -> str:
    """Best-effort process + system RAM snapshot without requiring psutil."""
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        parts = [
            f"rss={_fmt_bytes(int(getattr(mi, 'rss', 0)))}",
            f"vms={_fmt_bytes(int(getattr(mi, 'vms', 0)))}",
            f"ram_used={_fmt_bytes(int(vm.used))}/{_fmt_bytes(int(vm.total))} ({float(vm.percent):.1f}%)",
        ]
        try:
            parts.append(f"swap_used={_fmt_bytes(int(sm.used))}/{_fmt_bytes(int(sm.total))} ({float(sm.percent):.1f}%)")
        except Exception:
            pass
        return ", ".join(parts)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            phys_used = int(mem.ullTotalPhys - mem.ullAvailPhys)
            page_used = int(mem.ullTotalPageFile - mem.ullAvailPageFile)
            return (
                f"ram_used={_fmt_bytes(phys_used)}/{_fmt_bytes(int(mem.ullTotalPhys))} ({int(mem.dwMemoryLoad)}%), "
                f"commit/pagefile_used={_fmt_bytes(page_used)}/{_fmt_bytes(int(mem.ullTotalPageFile))}"
            )
        except Exception as exc:
            return f"ram snapshot unavailable: {type(exc).__name__}: {exc}"

    try:
        import resource  # type: ignore
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"maxrss={_fmt_bytes(int(rss_kb) * 1024)}"
    except Exception as exc:
        return f"ram snapshot unavailable: {type(exc).__name__}: {exc}"


def _two_stage_gap_event(
    ctx: Dict[str, Any],
    label: str,
    detail: str = "",
    torch_module: Any | None = None,
    echo: bool = True,
) -> None:
    try:
        t0 = float(ctx.setdefault("_ltx_two_stage_gap_t0", time.perf_counter()))
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = 0.0
    try:
        cuda = _cuda_snapshot(torch_module)
    except Exception:
        cuda = "n/a"
    try:
        ram = _ram_snapshot()
    except Exception:
        ram = "n/a"
    line = f"{elapsed:9.3f}s | {label}: {detail} | {cuda} | {ram}"
    try:
        events = ctx.setdefault("ltx_two_stage_gap_events", [])
        if isinstance(events, list) and len(events) < 800:
            events.append(line)
        ctx["ltx_two_stage_gap_event_count"] = str(len(events) if isinstance(events, list) else 0)
    except Exception:
        pass
    if echo:
        try:
            print(f"[vram-lab-ltx-gap] {line}", flush=True)
        except Exception:
            pass


def _install_ltx_two_stage_gap_profiler(ctx: Dict[str, Any], torch_module: Any | None = None, echo: bool = True) -> None:
    """Instrument the expensive two-stage transition gap without changing behavior.

    Focus area observed in reports:
    LoRA safetensors load finishes quickly, then the process spends ~160s before
    the stage-2 transformer is ready. This scanner times LoRA fusion, per-weight
    fusion progress, and the final Module.load_state_dict(assign=True) step while
    also sampling RAM/commit and CUDA state.
    """
    ctx["ltx_two_stage_gap_profiler_installed"] = "NO"
    ctx["ltx_two_stage_gap_profiler_errors"] = "none"
    ctx["ltx_two_stage_gap_event_count"] = "0"
    ctx["ltx_two_stage_gap_summary"] = "not attempted"
    ctx.setdefault("ltx_two_stage_gap_events", [])

    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
        sgb = importlib.import_module("ltx_core.loader.single_gpu_model_builder")
        fuse_mod = importlib.import_module("ltx_core.loader.fuse_loras")
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    counters = ctx.setdefault("_ltx_two_stage_gap_counters", {})
    if not isinstance(counters, dict):
        counters = {}
        ctx["_ltx_two_stage_gap_counters"] = counters
    ctx.setdefault("ltx_lora_fusion_cache_early_fast_path", "NO")
    ctx.setdefault("ltx_lora_fusion_cache_early_hit", "NO")
    ctx.setdefault("ltx_lora_fusion_cache_early_skipped_original_lora_load", "NO")
    ctx.setdefault("ltx_lora_fusion_cache_early_skipped_official_apply_loras", "NO")
    ctx.setdefault("ltx_lora_fusion_cache_early_loaded_tensors", "0")
    ctx.setdefault("ltx_lora_fusion_cache_early_load_time_s", "0.000")
    ctx.setdefault("ltx_lora_fusion_cache_early_apply_time_s", "0.000")
    ctx.setdefault("ltx_lora_fusion_cache_early_errors", "none")
    try:
        _initial_cache_mode = _lora_cache_mode()
        if _initial_cache_mode == "off":
            ctx["ltx_lora_fusion_cache_status"] = "OFF: official LTX LoRA path only; no fused-cache lookup and no fused-cache creation"
            ctx["ltx_lora_fusion_cache_save"] = "disabled: cache mode off"
        elif _initial_cache_mode == "read":
            ctx["ltx_lora_fusion_cache_status"] = "READ ONLY: will use a matching fused cache if present; will not create a new cache"
    except Exception:
        pass

    def _summarize() -> None:
        try:
            parts = []
            for key in [
                "load_model_weights_calls",
                "lora_load_model_weights_calls",
                "apply_loras_calls",
                "fuse_lora_weights_calls",
                "fused_weight_count",
                "aggregate_calls",
                "module_load_state_dict_calls",
            ]:
                parts.append(f"{key}={counters.get(key, 0)}")
            for key in [
                "apply_loras_total_s",
                "fuse_lora_weights_total_s",
                "aggregate_total_s",
                "module_load_state_dict_total_s",
                "lora_gap_candidate_total_s",
            ]:
                try:
                    parts.append(f"{key}={float(counters.get(key, 0.0)):.3f}")
                except Exception:
                    parts.append(f"{key}={counters.get(key, 0)}")
            ctx["ltx_two_stage_gap_summary"] = ", ".join(parts)
        except Exception:
            pass

    def _lora_cache_mode() -> str:
        return str(ctx.get("ltx_lora_fusion_cache_mode", "auto")).lower().strip()

    def _lora_cache_enabled() -> bool:
        return _lora_cache_mode() in {"auto", "read", "rebuild"}

    def _lora_cache_miss_inplace_enabled() -> bool:
        return str(ctx.get("ltx_lora_fusion_cache_miss_inplace", "auto")).lower().strip() != "off"

    def _lora_cache_preclean_enabled() -> bool:
        return str(ctx.get("ltx_lora_fusion_cache_preclean", "auto")).lower().strip() != "off"

    def _lora_cache_preclean(label: str) -> None:
        """Small cleanup boundary before expensive first-time cache build.

        This does not delete live model_sd / lora tensors. It only drops stale Python/CUDA
        allocations before official LoRA fusion starts, and records a clear report marker.
        """
        if not _lora_cache_preclean_enabled():
            ctx["ltx_lora_fusion_cache_preclean_status"] = "OFF"
            return
        try:
            before_alloc = _format_bytes(int(torch.cuda.memory_allocated())) if torch.cuda.is_available() else "n/a"
            before_reserved = _format_bytes(int(torch.cuda.memory_reserved())) if torch.cuda.is_available() else "n/a"
        except Exception:
            before_alloc = before_reserved = "n/a"
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            after_alloc = _format_bytes(int(torch.cuda.memory_allocated())) if torch.cuda.is_available() else "n/a"
            after_reserved = _format_bytes(int(torch.cuda.memory_reserved())) if torch.cuda.is_available() else "n/a"
        except Exception:
            after_alloc = after_reserved = "n/a"
        msg = f"{label}; cuda allocated {before_alloc}->{after_alloc}; reserved {before_reserved}->{after_reserved}"
        ctx["ltx_lora_fusion_cache_preclean_status"] = msg
        _two_stage_gap_event(ctx, "lora_fusion_cache:preclean", msg, torch, echo)

    def _lora_cache_apply_cached_stage_limits(label: str) -> None:
        """Make cached/fused LoRA loads inherit the same Stage-2 block limit.

        Cached LoRA is not allowed to become a separate memory route.  When a
        two-stage LoRA model/cache path is being prepared, retarget the active
        VRAM Lab hook-module hot-window to the Stage-2 limit before any cached
        fused tensors or fused state_dict are loaded.  The later BatchSplitAdapter
        hook still applies the same limit again at attach time; this guards the
        earlier cached/LoRA model-load boundary.
        """
        try:
            gb, source = _block_limit_for_role(ctx, "stage2_refine_denoise")
            if float(gb) <= 0.0:
                ctx["cached_lora_stage_limits"] = "NO: no positive Stage-2 block limit"
                return
            hooks_mod = ctx.get("_ltx_vram_hooks_module")
            hot_bytes = int(float(gb) * 1024 ** 3)
            ctx["cached_lora_stage_limits"] = "YES"
            ctx["cached_lora_uses_stage_limits"] = "YES"
            ctx["cached_lora_stage1_block_limit"] = f"{_ctx_float(ctx, 'stage1_block_size_limit_gb', _ctx_float(ctx, 'ltx_main_profile_hot_window_gb', 0.0)):.1f} GB"
            ctx["cached_lora_stage2_block_limit"] = f"{float(gb):.1f} GB ({source})"
            ctx["cached_lora_limit_last_boundary"] = str(label)
            ctx["cached_lora_cache_bypassed_hooks"] = "NO: cached LoRA path retargeted to Stage-2 block limit before model/cache load"
            ctx["cached_fused_model_memory_route"] = "guarded: cached/fused LoRA inherits Stage-1/Stage-2 block limits"
            ctx["cached_path_uses_stage_limits"] = "YES"
            if hooks_mod is not None:
                for attr, value in (
                    ("SAFE_HOT_WINDOW_GB", float(gb)),
                    ("BALANCED_HOT_WINDOW_GB", float(gb)),
                    ("SAFE_HOT_WINDOW_BYTES", hot_bytes),
                    ("BALANCED_HOT_WINDOW_BYTES", hot_bytes),
                ):
                    try:
                        setattr(hooks_mod, attr, value)
                    except Exception:
                        pass
                try:
                    profiles = getattr(hooks_mod, "VRAM_LAB_RESIDENCY_PROFILES", None)
                    profile_gb = int(_ctx_float(ctx, "vram_profile_gb", 24))
                    if isinstance(profiles, dict):
                        active_safe = profiles.setdefault(profile_gb, {}).setdefault("safe", {})
                        if isinstance(active_safe, dict):
                            active_safe["hot_window_gb"] = float(gb)
                            floor_gb = max(0.25, min(3.00, _ctx_float(ctx, "vram_emergency_driver_free_floor_requested_gb", _ctx_float(ctx, "vram_emergency_driver_free_floor", 1.5))))
                            active_safe["driver_free_floor_gb"] = floor_gb
                            active_safe["note"] = f"Stage-2 cached/fused LoRA guarded hot-window {float(gb):.1f} GB; emergency trim below {floor_gb:.2f} GB driver-free"
                except Exception:
                    pass
            _two_stage_gap_event(ctx, "lora_fusion_cache:stage_limit_guard", f"boundary={label}; stage2_limit={float(gb):.1f}GB; source={source}", torch, echo)
        except Exception as exc:
            ctx["cached_lora_stage_limits"] = f"FAILED: {type(exc).__name__}: {exc}"
            ctx["cached_lora_uses_stage_limits"] = "NO"

    def _normalise_file_for_key(path_value: Any) -> Dict[str, Any]:
        try:
            pp = Path(str(path_value))
            st = pp.stat()
            return {
                "path": str(pp.resolve()),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        except Exception:
            return {"path": str(path_value), "size": -1, "mtime_ns": -1}

    def _lora_cache_base_dir() -> Path:
        try:
            root = Path(str(ctx.get("ltx_root") or APP_ROOT))
        except Exception:
            root = APP_ROOT
        return root / "models" / "ltx23" / "fused_lora_cache"

    def _lora_cache_make_key(model_paths: Any, loras_tuple: Any, dtype: Any, device: Any) -> tuple[str, Path, Path, Dict[str, Any]]:
        model_path_list = _normalise_model_paths(model_paths)
        lora_items: List[Dict[str, Any]] = []
        try:
            for lora in tuple(loras_tuple or ()):
                lora_items.append({
                    "file": _normalise_file_for_key(getattr(lora, "path", lora)),
                    "strength": float(getattr(lora, "strength", 0.0)),
                    "sd_ops": str(type(getattr(lora, "sd_ops", None)).__module__) + "." + str(type(getattr(lora, "sd_ops", None)).__name__),
                })
        except Exception:
            pass
        payload = {
            "format": "framevision_ltx_lora_fusion_cache_v1",
            "model_files": [_normalise_file_for_key(x) for x in model_path_list],
            "loras": lora_items,
            "dtype": str(dtype),
            "device": str(device),
            "note": "Cache stores only fused tensors modified by LoRA, not the full base checkpoint.",
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        key = hashlib.sha256(raw).hexdigest()[:24]
        base = _lora_cache_base_dir()
        return key, base / f"{key}.safetensors", base / f"{key}.json", payload

    def _lora_cache_modified_keys(model_sd: Any, lora_sd_and_strengths: Any) -> set[str]:
        keys: set[str] = set()
        try:
            base_sd = getattr(model_sd, "sd", {}) or {}
            for item in list(lora_sd_and_strengths or []):
                try:
                    lora_sd = getattr(item, "state_dict", None)
                    lora_dict = getattr(lora_sd, "sd", {}) or {}
                except Exception:
                    lora_dict = {}
                for lk in lora_dict.keys():
                    if not isinstance(lk, str) or not lk.endswith(".lora_A.weight"):
                        continue
                    prefix = lk[: -len(".lora_A.weight")]
                    weight_key = f"{prefix}.weight"
                    if weight_key in base_sd:
                        keys.add(weight_key)
                        scale_key = f"{prefix}.weight_scale"
                        if scale_key in base_sd:
                            keys.add(scale_key)
        except Exception:
            pass
        return keys

    def _lora_cache_shard_dir(cache_path: Path) -> Path:
        return cache_path.with_name(f"{cache_path.stem}.shards")

    def _lora_cache_shard_manifest(cache_path: Path) -> Path:
        return _lora_cache_shard_dir(cache_path) / "manifest.json"

    def _lora_cache_entry_exists(cache_path: Path) -> bool:
        if cache_path.exists() and cache_path.is_file():
            return True
        manifest = _lora_cache_shard_manifest(cache_path)
        return manifest.exists() and manifest.is_file()

    def _tensor_nbytes(value: Any) -> int:
        try:
            nb = getattr(value, "nbytes", None)
            if nb is not None:
                return int(nb)
        except Exception:
            pass
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return 0

    def _lora_cache_shard_size_bytes() -> int:
        try:
            gb = float(ctx.get("ltx_lora_fusion_cache_shard_gb", 4.0))
        except Exception:
            gb = 4.0
        gb = max(0.25, gb)
        return int(gb * 1024 * 1024 * 1024)

    def _lora_cache_shard_threshold_bytes() -> int:
        try:
            gb = float(ctx.get("ltx_lora_fusion_cache_shard_threshold_gb", 8.0))
        except Exception:
            gb = 8.0
        gb = max(0.25, gb)
        return int(gb * 1024 * 1024 * 1024)

    def _lora_cache_try_load(model_sd: Any, cache_path: Path, expected_keys: set[str]) -> bool:
        try:
            from safetensors import safe_open  # type: ignore
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_status"] = f"disabled: safetensors import failed: {type(exc).__name__}: {exc}"
            return False
        if _lora_cache_mode() == "rebuild":
            return False

        def _cache_breadcrumb(label: str, detail: str = "") -> None:
            """Tiny always-flushed breadcrumb for native crashes in the cache-hit handoff."""
            try:
                msg = f"{label}: {detail}" if detail else str(label)
                ctx["ltx_lora_fusion_cache_last_breadcrumb"] = msg
                _two_stage_gap_event(ctx, f"lora_fusion_cache:{label}", detail, torch, echo)
                try:
                    root = Path(str(ctx.get("ltx_root") or APP_ROOT))
                    log_dir = root / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    snap = _cuda_snapshot(torch)
                    ram = _ram_snapshot()
                    (log_dir / "ltx_lora_cache_breadcrumb_latest.txt").write_text(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"breadcrumb: {msg}\n"
                        f"cache_path: {cache_path}\n"
                        f"cache_mode: {_lora_cache_mode()}\n"
                        f"attention_backend: {ctx.get('ltx_attention_backend_selected', ctx.get('ltx_attention_backend_requested', 'n/a'))}\n"
                        f"stage2_limit: {ctx.get('ltx_stage2_block_size_limit_gb', ctx.get('stage2_block_size_limit_gb', 'n/a'))}\n"
                        f"loaded_tensors_so_far: {ctx.get('ltx_lora_fusion_cache_loaded_tensors', '0')}\n"
                        f"last_key: {ctx.get('ltx_lora_fusion_cache_last_key', 'n/a')}\n"
                        f"cuda: {snap}\n"
                        f"ram: {ram}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            except Exception:
                pass

        def _assign_cached_tensor(file_obj: Any, key: str) -> None:
            """Load one cached tensor with an owned CPU copy before replacing the base tensor.

            The previous cache-hit path assigned tensors returned directly by safe_open into
            the destination state dict. That was fast, but native crashes were observed during
            the cache-load handoff when the default/PyTorch attention path was selected. This
            safer handoff copies each tensor into owned CPU storage before the old base tensor
            is released, and leaves breadcrumbs before native safetensors calls.
            """
            ctx["ltx_lora_fusion_cache_last_key"] = str(key)
            new_tensor = file_obj.get_tensor(key)
            try:
                # Keep cached tensors independent from the safe_open backing/mmap lifetime.
                if str(getattr(new_tensor, "device", "cpu")) == "cpu":
                    new_tensor = new_tensor.detach().clone(memory_format=torch.preserve_format)
            except Exception:
                try:
                    new_tensor = new_tensor.clone()
                except Exception:
                    pass
            old = model_sd.sd.get(key)
            model_sd.sd[key] = new_tensor
            del old
            del new_tensor

        def _lora_cache_should_use_bulk_single_loader() -> bool:
            """Use a safer whole-file CPU loader for the fragile non-Flash cache handoff.

            The official-default attention run crashed inside safe_open().get_tensor() on
            the very first fused-cache tensor. Flash2 runs keep the old faster safe_open
            path because that path is already proven stable there.
            """
            try:
                selected = str(ctx.get("ltx_attention_backend_selected", "") or "").strip().lower()
                requested = str(ctx.get("ltx_attention_backend_requested", "") or "").strip().lower()
                backend = selected or requested
                if backend == "flash2":
                    return False
                if "flash" in backend:
                    return False
                return backend in {"", "auto", "official default", "official", "default", "pytorch", "none", "not attempted"} or "official" in backend or "pytorch" in backend
            except Exception:
                return True

        def _lora_cache_load_single_bulk(model_sd: Any, cache_path: Path, expected_keys: set[str]) -> bool:
            """Load the single fused cache through safetensors.torch.load_file on CPU.

            This avoids the observed native access violation on the first
            safe_open().get_tensor() call when Flash2 is OFF/default attention is used.
            It is intentionally used only for the single-file cache path and only for
            non-Flash attention backends.
            """
            nonlocal loaded
            try:
                from safetensors.torch import load_file as _safe_load_file  # type: ignore
            except Exception as exc:
                ctx["ltx_lora_fusion_cache_status"] = f"bulk cache loader unavailable: {type(exc).__name__}: {exc}"
                _cache_breadcrumb("bulk_import_failed", f"{type(exc).__name__}: {exc}")
                return False

            _cache_breadcrumb("bulk_load_file_start", f"path={cache_path}; expected_keys={len(expected_keys or [])}; device=cpu")
            try:
                cached_sd = _safe_load_file(str(cache_path), device="cpu")
            except Exception as exc:
                ctx["ltx_lora_fusion_cache_status"] = f"bulk cache load failed: {type(exc).__name__}: {exc}"
                _cache_breadcrumb("bulk_load_file_failed", f"{type(exc).__name__}: {exc}")
                return False

            try:
                file_keys = set(str(k) for k in cached_sd.keys())
                _cache_breadcrumb("bulk_loaded", f"file_keys={len(file_keys)}; expected_keys={len(expected_keys or [])}")
                if expected_keys and not expected_keys.issubset(file_keys):
                    missing = len(expected_keys - file_keys)
                    ctx["ltx_lora_fusion_cache_status"] = f"bulk cache present but incomplete: missing {missing} expected tensor(s)"
                    _cache_breadcrumb("bulk_missing_keys", f"missing={missing}")
                    return False

                keys_to_load = sorted(expected_keys or file_keys)
                for key_index, key in enumerate(keys_to_load, start=1):
                    if key not in cached_sd:
                        continue
                    try:
                        if key_index == 1 or loaded % 25 == 0:
                            _cache_breadcrumb("bulk_assign_tensor", f"index={key_index}/{len(keys_to_load)}; total_loaded={loaded}; key={key}")
                        ctx["ltx_lora_fusion_cache_last_key"] = str(key)
                        new_tensor = cached_sd.pop(key)
                        try:
                            if str(getattr(new_tensor, "device", "cpu")) != "cpu":
                                new_tensor = new_tensor.detach().cpu()
                            else:
                                new_tensor = new_tensor.detach()
                        except Exception:
                            pass
                        old = model_sd.sd.get(key)
                        model_sd.sd[key] = new_tensor
                        del old
                        del new_tensor
                        loaded += 1
                        ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                        if loaded % 100 == 0:
                            gc.collect()
                    except Exception as exc:
                        ctx["ltx_lora_fusion_cache_status"] = f"bulk cache assign failed at {key}: {type(exc).__name__}: {exc}"
                        _cache_breadcrumb("bulk_assign_failed", f"key={key}; error={type(exc).__name__}: {exc}")
                        return False
                try:
                    cached_sd.clear()
                except Exception:
                    pass
                del cached_sd
                gc.collect()
                return True
            except Exception as exc:
                ctx["ltx_lora_fusion_cache_status"] = f"bulk cache load failed after read: {type(exc).__name__}: {exc}"
                _cache_breadcrumb("bulk_failed", f"{type(exc).__name__}: {exc}")
                return False

        t0 = time.perf_counter()
        loaded = 0
        try:
            # Cache-hit replacement is a CPU state-dict operation. Keep it CPU-only even if
            # the wrapped model later moves non-block pieces to CUDA. This avoids accidental
            # CUDA materialization inside the cache read path.
            target_device = "cpu"
            ctx["ltx_lora_fusion_cache_load_device"] = target_device
            manifest_path = _lora_cache_shard_manifest(cache_path)
            _cache_breadcrumb("load_start", f"path={cache_path}; expected_keys={len(expected_keys or [])}; device={target_device}")

            # New sharded cache path. This keeps the useful fused-cache speedup but avoids one huge 37 GB read/write event.
            if manifest_path.exists() and manifest_path.is_file():
                _cache_breadcrumb("open_manifest", str(manifest_path))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("format") != "framevision_ltx_lora_fusion_cache_sharded_v1":
                    ctx["ltx_lora_fusion_cache_status"] = "sharded cache manifest invalid"
                    _cache_breadcrumb("manifest_invalid", str(manifest_path))
                    return False
                shards = list(manifest.get("shards") or [])
                all_keys: set[str] = set()
                for shard in shards:
                    for key in list((shard or {}).get("keys") or []):
                        all_keys.add(str(key))
                _cache_breadcrumb("manifest_ok", f"shards={len(shards)}; keys={len(all_keys)}")
                if expected_keys and not expected_keys.issubset(all_keys):
                    missing = len(expected_keys - all_keys)
                    ctx["ltx_lora_fusion_cache_status"] = f"sharded cache present but incomplete: missing {missing} expected tensor(s)"
                    _cache_breadcrumb("manifest_missing_keys", f"missing={missing}")
                    return False
                shard_dir = manifest_path.parent
                for shard_index, shard in enumerate(shards, start=1):
                    shard_file = shard_dir / str(shard.get("file", ""))
                    if not shard_file.exists() or not shard_file.is_file():
                        ctx["ltx_lora_fusion_cache_status"] = f"sharded cache missing shard: {shard_file}"
                        _cache_breadcrumb("missing_shard", str(shard_file))
                        return False
                    _cache_breadcrumb("open_shard", f"shard={shard_index}/{len(shards)}; file={shard_file}")
                    with safe_open(str(shard_file), framework="pt", device=target_device) as f:
                        file_keys = set(str(k) for k in f.keys())
                        keys_to_load = sorted((expected_keys & file_keys) if expected_keys else file_keys)
                        _cache_breadcrumb("shard_opened", f"shard={shard_index}/{len(shards)}; file_keys={len(file_keys)}; load_keys={len(keys_to_load)}")
                        for key_index, key in enumerate(keys_to_load, start=1):
                            try:
                                if key_index == 1 or loaded % 25 == 0:
                                    _cache_breadcrumb("load_tensor", f"shard={shard_index}/{len(shards)}; index={key_index}/{len(keys_to_load)}; total_loaded={loaded}; key={key}")
                                _assign_cached_tensor(f, key)
                                loaded += 1
                                ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                                if loaded % 100 == 0:
                                    gc.collect()
                            except Exception as exc:
                                ctx["ltx_lora_fusion_cache_status"] = f"sharded cache load failed at {key}: {type(exc).__name__}: {exc}"
                                _cache_breadcrumb("load_failed", f"key={key}; error={type(exc).__name__}: {exc}")
                                return False
                    gc.collect()
                    _two_stage_gap_event(ctx, "lora_fusion_cache:load_shard", f"shard={shard_index}/{len(shards)}; loaded_tensors={loaded}; file={shard_file.name}", torch, echo)
                dt = time.perf_counter() - t0
                ctx["ltx_lora_fusion_cache_hit"] = "YES"
                ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                ctx["ltx_lora_fusion_cache_load_time_s"] = f"{dt:.3f}"
                ctx["ltx_lora_fusion_cache_status"] = f"HIT: loaded {loaded} fused LoRA tensor(s) from sharded cache"
                _cache_breadcrumb("hit", f"loaded_tensors={loaded}; shards={len(shards)}; duration={dt:.3f}s; path={manifest_path}")
                _two_stage_gap_event(ctx, "lora_fusion_cache:hit", f"loaded_tensors={loaded}; shards={len(shards)}; duration={dt:.3f}s; path={manifest_path}", torch, echo)
                return True

            # Existing single-file cache path stays supported.
            if not cache_path.exists() or not cache_path.is_file():
                _cache_breadcrumb("miss", f"single cache file not found: {cache_path}")
                return False

            # The official-default/PyTorch attention path crashed inside the first
            # safe_open().get_tensor() call for this cache file. For that backend,
            # use safetensors.torch.load_file on CPU, then assign tensors from the
            # resulting owned dict. Flash2 keeps the old proven fast path.
            if _lora_cache_should_use_bulk_single_loader():
                ctx["ltx_lora_fusion_cache_load_strategy"] = "single_bulk_load_file_cpu"
                if _lora_cache_load_single_bulk(model_sd, cache_path, expected_keys):
                    dt = time.perf_counter() - t0
                    ctx["ltx_lora_fusion_cache_hit"] = "YES"
                    ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                    ctx["ltx_lora_fusion_cache_load_time_s"] = f"{dt:.3f}"
                    ctx["ltx_lora_fusion_cache_status"] = f"HIT: loaded {loaded} fused LoRA tensor(s) from cache via bulk CPU loader"
                    _cache_breadcrumb("hit", f"loaded_tensors={loaded}; duration={dt:.3f}s; path={cache_path}; strategy=bulk_cpu")
                    _two_stage_gap_event(ctx, "lora_fusion_cache:hit", f"loaded_tensors={loaded}; duration={dt:.3f}s; path={cache_path}; strategy=bulk_cpu", torch, echo)
                    return True
                return False

            ctx["ltx_lora_fusion_cache_load_strategy"] = "single_safe_open_get_tensor"
            _cache_breadcrumb("open_single", str(cache_path))
            with safe_open(str(cache_path), framework="pt", device=target_device) as f:
                file_keys = set(str(k) for k in f.keys())
                _cache_breadcrumb("single_opened", f"file_keys={len(file_keys)}; expected_keys={len(expected_keys or [])}")
                if expected_keys and not expected_keys.issubset(file_keys):
                    missing = len(expected_keys - file_keys)
                    ctx["ltx_lora_fusion_cache_status"] = f"cache present but incomplete: missing {missing} expected tensor(s)"
                    _cache_breadcrumb("single_missing_keys", f"missing={missing}")
                    return False
                keys_to_load = sorted(expected_keys or file_keys)
                for key_index, key in enumerate(keys_to_load, start=1):
                    if key not in file_keys:
                        continue
                    try:
                        if key_index == 1 or loaded % 25 == 0:
                            _cache_breadcrumb("load_tensor", f"index={key_index}/{len(keys_to_load)}; total_loaded={loaded}; key={key}")
                        _assign_cached_tensor(f, key)
                        loaded += 1
                        ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                        if loaded % 100 == 0:
                            gc.collect()
                    except Exception as exc:
                        ctx["ltx_lora_fusion_cache_status"] = f"cache load failed at {key}: {type(exc).__name__}: {exc}"
                        _cache_breadcrumb("load_failed", f"key={key}; error={type(exc).__name__}: {exc}")
                        return False
            dt = time.perf_counter() - t0
            ctx["ltx_lora_fusion_cache_hit"] = "YES"
            ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
            ctx["ltx_lora_fusion_cache_load_time_s"] = f"{dt:.3f}"
            ctx["ltx_lora_fusion_cache_status"] = f"HIT: loaded {loaded} fused LoRA tensor(s) from cache"
            _cache_breadcrumb("hit", f"loaded_tensors={loaded}; duration={dt:.3f}s; path={cache_path}")
            _two_stage_gap_event(ctx, "lora_fusion_cache:hit", f"loaded_tensors={loaded}; duration={dt:.3f}s; path={cache_path}", torch, echo)
            return True
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_status"] = f"cache load failed: {type(exc).__name__}: {exc}"
            _cache_breadcrumb("load_failed", f"error={type(exc).__name__}: {exc}")
            _two_stage_gap_event(ctx, "lora_fusion_cache:load_failed", str(exc), torch, echo)
            return False

    def _lora_cache_meta_is_valid(meta_path: Path, cache_meta: Dict[str, Any]) -> bool:
        """Best-effort validation for existing fused-cache metadata.

        Keep compatibility with the existing cache format. Missing/corrupt metadata
        is treated as a miss so the old creation path can rebuild it.
        """
        try:
            if not meta_path.exists() or not meta_path.is_file():
                ctx["ltx_lora_fusion_cache_early_errors"] = f"meta missing: {meta_path}"
                return False
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                ctx["ltx_lora_fusion_cache_early_errors"] = "meta is not a JSON object"
                return False
            # The saved meta is the same payload with extra fields. Validate only
            # stable identity fields so old cache files remain compatible.
            for key in ["format", "model_files", "loras"]:
                if key in cache_meta and meta.get(key) != cache_meta.get(key):
                    ctx["ltx_lora_fusion_cache_early_errors"] = f"meta mismatch: {key}"
                    return False
            return True
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_early_errors"] = f"meta validation failed: {type(exc).__name__}: {exc}"
            return False

    def _module_lookup_child(obj: Any, name: str) -> Any:
        try:
            modules = getattr(obj, "_modules", None)
            if isinstance(modules, dict) and name in modules:
                return modules[name]
        except Exception:
            pass
        try:
            if name.isdigit() and hasattr(obj, "__getitem__"):
                return obj[int(name)]
        except Exception:
            pass
        return getattr(obj, name)

    def _module_set_tensor_by_key(module: Any, key: str, tensor: Any) -> bool:
        """Replace a parameter/buffer by state_dict key without building a full state dict."""
        try:
            import torch as _torch  # type: ignore
            parts = str(key).split(".")
            if not parts:
                return False
            obj = module
            for part in parts[:-1]:
                obj = _module_lookup_child(obj, part)
            leaf = parts[-1]
            params = getattr(obj, "_parameters", None)
            if isinstance(params, dict) and leaf in params:
                old = params.get(leaf)
                req = bool(getattr(old, "requires_grad", False))
                if not isinstance(tensor, _torch.nn.Parameter):
                    tensor = _torch.nn.Parameter(tensor, requires_grad=req)
                else:
                    try:
                        tensor.requires_grad_(req)
                    except Exception:
                        pass
                params[leaf] = tensor
                return True
            bufs = getattr(obj, "_buffers", None)
            if isinstance(bufs, dict) and leaf in bufs:
                bufs[leaf] = tensor
                return True
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_early_errors"] = f"set tensor failed for {key}: {type(exc).__name__}: {exc}"
        return False

    def _lora_cache_apply_to_module_streaming(module: Any, cache_path: Path, meta_path: Path, cache_meta: Dict[str, Any]) -> tuple[bool, int, float, float, str]:
        """Stream cached fused tensors directly into a freshly built base module.

        This is the early cache-hit fast path: it avoids loading the original LoRA
        safetensors and avoids official apply_loras on cache hit.
        """
        if not cache_path.exists() or not cache_path.is_file():
            return False, 0, 0.0, 0.0, f"cache missing: {cache_path}"
        if not _lora_cache_meta_is_valid(meta_path, cache_meta):
            return False, 0, 0.0, 0.0, str(ctx.get("ltx_lora_fusion_cache_early_errors", "meta invalid"))
        try:
            from safetensors import safe_open  # type: ignore
        except Exception as exc:
            return False, 0, 0.0, 0.0, f"safetensors import failed: {type(exc).__name__}: {exc}"

        load_t0 = time.perf_counter()
        loaded = 0
        try:
            if module is None or not hasattr(module, "state_dict"):
                return False, 0, 0.0, 0.0, f"module unavailable for cache apply: {type(module).__name__}"
            module_sd = module.state_dict()
            module_keys = set(str(k) for k in module_sd.keys())
            load_dt = 0.0
            apply_t0 = time.perf_counter()
            with safe_open(str(cache_path), framework="pt", device="cpu") as f:
                cache_keys = [str(k) for k in f.keys()]
                if not cache_keys:
                    return False, 0, 0.0, 0.0, "cache contains no tensors"
                missing = [k for k in cache_keys if k not in module_keys]
                if missing:
                    return False, 0, 0.0, 0.0, f"cache key missing in module: {missing[0]} (+{len(missing)-1} more)"
                load_dt = time.perf_counter() - load_t0
                for key in cache_keys:
                    try:
                        old = module_sd.get(key)
                        tensor = f.get_tensor(key)
                        if old is not None and tuple(getattr(tensor, "shape", ())) != tuple(getattr(old, "shape", ())):
                            return False, loaded, load_dt, time.perf_counter() - apply_t0, f"shape mismatch for {key}: cache={tuple(getattr(tensor, 'shape', ()))} module={tuple(getattr(old, 'shape', ()))}"
                        if not _module_set_tensor_by_key(module, key, tensor):
                            return False, loaded, load_dt, time.perf_counter() - apply_t0, f"could not assign {key}"
                        del old
                        loaded += 1
                        if loaded % 100 == 0:
                            gc.collect()
                    except Exception as exc:
                        return False, loaded, load_dt, time.perf_counter() - apply_t0, f"failed at {key}: {type(exc).__name__}: {exc}"
            apply_dt = time.perf_counter() - apply_t0
            return True, loaded, load_dt, apply_dt, "none"
        except Exception as exc:
            return False, loaded, 0.0, 0.0, f"{type(exc).__name__}: {exc}"


    def _lora_cache_max_files() -> int:
        try:
            value = int(ctx.get("ltx_lora_fusion_cache_max_files", 2))
        except Exception:
            value = 2
        return max(1, value)

    def _lora_cache_cleanup_after_save(current_cache_path: Path) -> None:
        """Keep only the newest N fused LoRA cache entries.

        Supports both old single-file entries and the new sharded-cache directories.
        """
        max_files = _lora_cache_max_files()
        ctx["ltx_lora_fusion_cache_max_files"] = str(max_files)
        try:
            cache_dir = current_cache_path.parent
            current_shard_dir = _lora_cache_shard_dir(current_cache_path)
            entries: List[tuple[float, str, Path]] = []
            seen: set[str] = set()
            for fp in cache_dir.glob("*.safetensors"):
                if fp.is_file():
                    try:
                        key = str(fp.resolve())
                    except Exception:
                        key = str(fp)
                    if key not in seen:
                        seen.add(key)
                        entries.append((float(fp.stat().st_mtime_ns), "file", fp))
            for dp in cache_dir.glob("*.shards"):
                if dp.is_dir() and (dp / "manifest.json").exists():
                    try:
                        key = str(dp.resolve())
                    except Exception:
                        key = str(dp)
                    if key not in seen:
                        seen.add(key)
                        entries.append((float((dp / "manifest.json").stat().st_mtime_ns), "dir", dp))
            entries.sort(key=lambda x: (x[0], x[2].name), reverse=True)
            deleted_files: List[str] = []
            deleted_bytes = 0
            for _mtime, kind, path in entries[max_files:]:
                try:
                    if kind == "file" and path.resolve() == current_cache_path.resolve():
                        continue
                    if kind == "dir" and path.resolve() == current_shard_dir.resolve():
                        continue
                except Exception:
                    pass
                try:
                    if kind == "file":
                        for victim in [path, path.with_suffix(".json")]:
                            if victim.exists() and victim.is_file():
                                try:
                                    deleted_bytes += int(victim.stat().st_size)
                                except Exception:
                                    pass
                                victim.unlink()
                                deleted_files.append(str(victim))
                    else:
                        for victim in path.rglob("*"):
                            try:
                                if victim.is_file():
                                    deleted_bytes += int(victim.stat().st_size)
                            except Exception:
                                pass
                        shutil.rmtree(path, ignore_errors=True)
                        deleted_files.append(str(path))
                except Exception as exc:
                    ctx["ltx_lora_fusion_cache_cleanup_errors"] = f"failed deleting {path}: {type(exc).__name__}: {exc}"
            remaining_files = [p for p in cache_dir.glob("*.safetensors") if p.is_file()]
            remaining_dirs = [p for p in cache_dir.glob("*.shards") if p.is_dir() and (p / "manifest.json").exists()]
            remaining_size = 0
            for fp in remaining_files:
                try:
                    remaining_size += int(fp.stat().st_size)
                except Exception:
                    pass
            for dp in remaining_dirs:
                for fp in dp.rglob("*"):
                    try:
                        if fp.is_file():
                            remaining_size += int(fp.stat().st_size)
                    except Exception:
                        pass
            ctx["ltx_lora_fusion_cache_cleanup_deleted"] = str(len(deleted_files))
            ctx["ltx_lora_fusion_cache_cleanup_deleted_bytes"] = _format_bytes(deleted_bytes)
            ctx["ltx_lora_fusion_cache_remaining_files"] = str(len(remaining_files) + len(remaining_dirs))
            ctx["ltx_lora_fusion_cache_remaining_size"] = _format_bytes(remaining_size)
            if deleted_files:
                msg = f"kept newest {max_files}; deleted {len(deleted_files)} cache entrie(s), freed {_format_bytes(deleted_bytes)}"
            else:
                msg = f"kept newest {max_files}; no old cache entries deleted"
            ctx["ltx_lora_fusion_cache_cleanup"] = msg
            _two_stage_gap_event(ctx, "lora_fusion_cache:cleanup", msg, torch, echo)
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_cleanup_errors"] = f"{type(exc).__name__}: {exc}"
            _two_stage_gap_event(ctx, "lora_fusion_cache:cleanup_failed", f"{type(exc).__name__}: {exc}", torch, echo)

    def _lora_cache_try_save(final_sd: Any, cache_path: Path, meta_path: Path, meta: Dict[str, Any], modified_keys: set[str]) -> None:
        if not modified_keys:
            ctx["ltx_lora_fusion_cache_save"] = "skipped: no modified keys detected"
            return
        try:
            from safetensors.torch import save_file  # type: ignore
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_save"] = f"failed: safetensors.torch import failed: {type(exc).__name__}: {exc}"
            return
        t0 = time.perf_counter()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            sd = getattr(final_sd, "sd", {}) or {}
            items: List[tuple[str, Any, int]] = []
            total_bytes = 0
            for key in sorted(modified_keys):
                val = sd.get(key)
                if val is None:
                    continue
                try:
                    tensor = val.detach()
                except Exception:
                    tensor = val
                nb = _tensor_nbytes(tensor)
                items.append((key, tensor, nb))
                total_bytes += int(nb)
            ctx["ltx_lora_fusion_cache_estimated_size"] = _format_bytes(total_bytes)
            shard_size = _lora_cache_shard_size_bytes()
            shard_threshold = _lora_cache_shard_threshold_bytes()
            ctx["ltx_lora_fusion_cache_shard_gb"] = f"{shard_size / (1024 ** 3):.2f}"
            ctx["ltx_lora_fusion_cache_shard_threshold_gb"] = f"{shard_threshold / (1024 ** 3):.2f}"

            if total_bytes >= shard_threshold:
                shard_dir = _lora_cache_shard_dir(cache_path)
                tmp_dir = cache_path.with_name(f"{cache_path.stem}.shards.tmp_{os.getpid()}_{int(time.time())}")
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                manifest_shards: List[Dict[str, Any]] = []
                shard_tensors: Dict[str, Any] = {}
                shard_bytes = 0
                shard_index = 0
                saved_tensors = 0

                def _flush_shard() -> None:
                    nonlocal shard_tensors, shard_bytes, shard_index, saved_tensors
                    if not shard_tensors:
                        return
                    shard_index += 1
                    shard_name = f"shard_{shard_index:05d}.safetensors"
                    shard_path = tmp_dir / shard_name
                    tmp_shard_path = tmp_dir / f"{shard_name}.tmp"
                    save_file(shard_tensors, str(tmp_shard_path))
                    os.replace(str(tmp_shard_path), str(shard_path))
                    keys = sorted(str(k) for k in shard_tensors.keys())
                    manifest_shards.append({
                        "file": shard_name,
                        "tensor_count": len(keys),
                        "bytes": int(shard_bytes),
                        "keys": keys,
                    })
                    saved_tensors += len(keys)
                    _two_stage_gap_event(ctx, "lora_fusion_cache:save_shard", f"shard={shard_index}; tensors={len(keys)}; bytes={_format_bytes(int(shard_bytes))}; file={shard_name}", torch, echo)
                    shard_tensors = {}
                    shard_bytes = 0
                    gc.collect()

                for key, tensor, nb in items:
                    if shard_tensors and shard_bytes + int(nb) > shard_size:
                        _flush_shard()
                    shard_tensors[key] = tensor
                    shard_bytes += int(nb)
                _flush_shard()

                manifest = dict(meta)
                manifest.update({
                    "format": "framevision_ltx_lora_fusion_cache_sharded_v1",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tensor_count": saved_tensors,
                    "total_bytes": int(total_bytes),
                    "shard_size_bytes": int(shard_size),
                    "cache_dir": str(shard_dir),
                    "shards": manifest_shards,
                })
                (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                if shard_dir.exists():
                    shutil.rmtree(shard_dir, ignore_errors=True)
                os.replace(str(tmp_dir), str(shard_dir))
                # Remove stale old single-file cache for the same key only after the sharded cache is complete.
                try:
                    if cache_path.exists() and cache_path.is_file():
                        cache_path.unlink()
                except Exception:
                    pass
                try:
                    if meta_path.exists() and meta_path.is_file():
                        meta_path.unlink()
                except Exception:
                    pass
                dt = time.perf_counter() - t0
                ctx["ltx_lora_fusion_cache_save"] = f"saved {saved_tensors} tensor(s) as {len(manifest_shards)} shard(s)"
                ctx["ltx_lora_fusion_cache_saved_tensors"] = str(saved_tensors)
                ctx["ltx_lora_fusion_cache_save_time_s"] = f"{dt:.3f}"
                ctx["ltx_lora_fusion_cache_path"] = str(shard_dir)
                _two_stage_gap_event(ctx, "lora_fusion_cache:save", f"saved_tensors={saved_tensors}; shards={len(manifest_shards)}; total={_format_bytes(total_bytes)}; duration={dt:.3f}s; path={shard_dir}", torch, echo)
                _lora_cache_cleanup_after_save(cache_path)
                return

            # Small caches keep the old simple single-file format.
            tensors: Dict[str, Any] = {key: tensor for key, tensor, _nb in items}
            tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            save_file(tensors, str(tmp_cache_path))
            os.replace(str(tmp_cache_path), str(cache_path))
            meta = dict(meta)
            meta.update({
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tensor_count": len(tensors),
                "total_bytes": int(total_bytes),
                "cache_file": str(cache_path),
            })
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            dt = time.perf_counter() - t0
            ctx["ltx_lora_fusion_cache_save"] = f"saved {len(tensors)} tensor(s)"
            ctx["ltx_lora_fusion_cache_saved_tensors"] = str(len(tensors))
            ctx["ltx_lora_fusion_cache_save_time_s"] = f"{dt:.3f}"
            _two_stage_gap_event(ctx, "lora_fusion_cache:save", f"saved_tensors={len(tensors)}; total={_format_bytes(total_bytes)}; duration={dt:.3f}s; path={cache_path}", torch, echo)
            _lora_cache_cleanup_after_save(cache_path)
        except Exception as exc:
            ctx["ltx_lora_fusion_cache_save"] = f"failed: {type(exc).__name__}: {exc}"
            _two_stage_gap_event(ctx, "lora_fusion_cache:save_failed", f"{type(exc).__name__}: {exc}", torch, echo)

    # Wrap _load_model_weights so stage-2 model rebuilds and LoRA presence are explicit.
    try:
        original_lmw = getattr(sgb, "_framevision_gap_original_load_model_weights", None)
        if original_lmw is None:
            original_lmw = getattr(sgb, "_load_model_weights")
            setattr(sgb, "_framevision_gap_original_load_model_weights", original_lmw)

            def _fv_gap_load_model_weights(meta_model: Any, model_path: Any, loras: Any, loader: Any, registry: Any, device: Any, dtype: Any, model_sd_ops: Any = None, lora_load_device: Any = None) -> Any:
                loras_tuple = tuple(loras or ())
                counters["load_model_weights_calls"] = int(counters.get("load_model_weights_calls", 0)) + 1
                if loras_tuple:
                    counters["lora_load_model_weights_calls"] = int(counters.get("lora_load_model_weights_calls", 0)) + 1
                model_name = f"{type(meta_model).__module__}.{type(meta_model).__name__}"
                model_paths = _normalise_model_paths(model_path)
                lora_paths = []
                try:
                    for lora in loras_tuple:
                        lora_paths.append(str(getattr(lora, "path", lora)))
                except Exception:
                    pass
                detail = (
                    f"model={model_name}; device={device}; dtype={dtype}; "
                    f"model_files={_file_size_summary(model_paths)}; "
                    f"loras={len(loras_tuple)}; lora_files={_file_size_summary(lora_paths)}; "
                    f"lora_load_device={lora_load_device}"
                )

                cache_info = None
                if loras_tuple and _lora_cache_enabled():
                    try:
                        cache_key, cache_path, meta_path, cache_meta = _lora_cache_make_key(model_path, loras_tuple, dtype, device)
                        cache_info = {
                            "key": cache_key,
                            "path": str(cache_path),
                            "meta_path": str(meta_path),
                            "meta": cache_meta,
                        }
                        ctx["_ltx_lora_cache_current"] = cache_info
                        ctx["ltx_lora_fusion_cache_key"] = cache_key
                        ctx["ltx_lora_fusion_cache_path"] = str(cache_path)
                        ctx["ltx_lora_fusion_cache_meta_path"] = str(meta_path)
                        ctx["ltx_lora_fusion_cache_status"] = "prepared; early direct-cache apply disabled for stability; using normal LoRA/cache path"
                        ctx["ltx_lora_fusion_cache_early_fast_path"] = "NO: disabled for stability"
                        ctx["ltx_lora_fusion_cache_early_errors"] = "early direct-cache apply disabled after stage-2 hard crashes"
                    except Exception as exc:
                        ctx["ltx_lora_fusion_cache_status"] = f"prepare failed: {type(exc).__name__}: {exc}"
                        ctx["ltx_lora_fusion_cache_early_errors"] = f"prepare failed: {type(exc).__name__}: {exc}"
                        ctx["_ltx_lora_cache_current"] = None

                if loras_tuple:
                    _lora_cache_apply_cached_stage_limits("_load_model_weights:lora_model:start")
                    _two_stage_gap_event(ctx, "_load_model_weights:lora_model:start", detail, torch, echo)

                    # Hard read-only guard.  "Read existing only" must not start the
                    # expensive official LoRA/cache-miss build path.  If there is no
                    # matching fused cache entry, stop before the 42GB base/LoRA state
                    # dicts are materialized and before any new cache/temp output can
                    # be prepared.  Users can choose Off for the normal official LoRA
                    # path or Create/Rebuild when they intentionally want a cache build.
                    if _lora_cache_mode() == "read" and isinstance(cache_info, dict):
                        try:
                            cache_path = Path(str(cache_info.get("path", "")))
                            if not _lora_cache_entry_exists(cache_path):
                                msg = f"READ ONLY MISS: no matching fused LoRA cache found at {cache_path}; refusing cache-miss LoRA build/fuse"
                                ctx["ltx_lora_fusion_cache_status"] = msg
                                ctx["ltx_lora_fusion_cache_hit"] = "NO"
                                ctx["ltx_lora_fusion_cache_save"] = "skipped: read existing only mode"
                                ctx["ltx_lora_fusion_cache_early_errors"] = "read-only cache miss before official LoRA load"
                                _two_stage_gap_event(ctx, "lora_fusion_cache:read_only_miss_abort", msg, torch, echo)
                                raise RuntimeError(
                                    "FrameVision LTX LoRA cache is set to Read existing only, but no matching fused cache was found. "
                                    "Use Create/use or Rebuild to make a cache, or Off for the official non-cache LoRA path."
                                )
                        except RuntimeError:
                            raise
                        except Exception as exc:
                            msg = f"READ ONLY CHECK FAILED: {type(exc).__name__}: {exc}; refusing cache-miss LoRA build/fuse"
                            ctx["ltx_lora_fusion_cache_status"] = msg
                            ctx["ltx_lora_fusion_cache_save"] = "skipped: read existing only mode"
                            _two_stage_gap_event(ctx, "lora_fusion_cache:read_only_check_failed", msg, torch, echo)
                            raise RuntimeError(msg)
                else:
                    # Keep non-LoRA model rebuilds visible but less noisy.
                    _two_stage_gap_event(ctx, "_load_model_weights:start", detail, torch, echo=False)

                t0 = time.perf_counter()
                early_attempted = False
                try:
                    # Early fused-cache direct-apply path disabled for stability.
                    # It caused stage-2 hard crashes after the base 42.98 GB model build.
                    # Keep the normal official apply_loras/cache wrapper below instead.
                    if False and loras_tuple and _lora_cache_enabled() and isinstance(cache_info, dict):
                        early_attempted = True
                        cache_path = Path(str(cache_info.get("path", "")))
                        meta_path = Path(str(cache_info.get("meta_path", "")))
                        cache_meta = dict(cache_info.get("meta") or {})
                        ctx["ltx_lora_fusion_cache_early_fast_path"] = "YES"
                        ctx["ltx_lora_fusion_cache_early_hit"] = "NO"
                        ctx["ltx_lora_fusion_cache_early_skipped_original_lora_load"] = "NO"
                        ctx["ltx_lora_fusion_cache_early_skipped_official_apply_loras"] = "NO"
                        ctx["ltx_lora_fusion_cache_early_loaded_tensors"] = "0"
                        ctx["ltx_lora_fusion_cache_early_load_time_s"] = "0.000"
                        ctx["ltx_lora_fusion_cache_early_apply_time_s"] = "0.000"
                        ctx.setdefault("ltx_lora_fusion_cache_early_errors", "none")

                        if _lora_cache_entry_exists(cache_path) and _lora_cache_mode() != "rebuild":
                            _two_stage_gap_event(ctx, "_load_model_weights:lora_model:early_cache_hit_check", f"path={cache_path}", torch, echo)
                            base_model_result = original_lmw(meta_model, model_path, (), loader, registry, device, dtype, model_sd_ops, lora_load_device)
                            # single_gpu_model_builder._load_model_weights mutates meta_model in-place and normally returns None.
                            # Use the live meta_model for early fused-cache streaming instead of treating None as the module.
                            base_model = base_model_result if base_model_result is not None else meta_model
                            ok, loaded, load_dt, apply_dt, err = _lora_cache_apply_to_module_streaming(base_model, cache_path, meta_path, cache_meta)
                            ctx["ltx_lora_fusion_cache_early_loaded_tensors"] = str(loaded)
                            ctx["ltx_lora_fusion_cache_early_load_time_s"] = f"{load_dt:.3f}"
                            ctx["ltx_lora_fusion_cache_early_apply_time_s"] = f"{apply_dt:.3f}"
                            if ok:
                                ctx["ltx_lora_fusion_cache_early_hit"] = "YES"
                                ctx["ltx_lora_fusion_cache_early_skipped_original_lora_load"] = "YES"
                                ctx["ltx_lora_fusion_cache_early_skipped_official_apply_loras"] = "YES"
                                ctx["ltx_lora_fusion_cache_early_errors"] = "none"
                                ctx["ltx_lora_fusion_cache_hit"] = "YES"
                                ctx["ltx_lora_fusion_cache_loaded_tensors"] = str(loaded)
                                ctx["ltx_lora_fusion_cache_load_time_s"] = f"{load_dt + apply_dt:.3f}"
                                ctx["ltx_lora_fusion_cache_status"] = f"EARLY HIT: streamed {loaded} fused LoRA tensor(s) from cache before original LoRA load"
                                _two_stage_gap_event(ctx, "lora_fusion_cache:early_hit", f"loaded_tensors={loaded}; load={load_dt:.3f}s; apply={apply_dt:.3f}s; skipped_original_lora=YES; skipped_apply_loras=YES; path={cache_path}", torch, echo)
                                return base_model

                            # Critical safety: after base_model was built, do not fall back into
                            # the official LoRA path. That would reload the same 42.98 GB base and
                            # can reproduce the double-load crash. Exit cleanly instead.
                            ctx["ltx_lora_fusion_cache_early_errors"] = err or "unknown early-cache failure"
                            ctx["ltx_lora_fusion_cache_status"] = f"early cache failed after base build: {ctx['ltx_lora_fusion_cache_early_errors']}"
                            _two_stage_gap_event(ctx, "lora_fusion_cache:early_abort", f"{ctx['ltx_lora_fusion_cache_early_errors']}; refusing unsafe official fallback after base build", torch, echo)
                            raise RuntimeError(f"FrameVision LTX early fused LoRA cache failed after base build; refusing unsafe official fallback: {ctx['ltx_lora_fusion_cache_early_errors']}")
                        else:
                            ctx["ltx_lora_fusion_cache_early_errors"] = "cache miss or rebuild requested"
                            ctx["ltx_lora_fusion_cache_status"] = "early cache miss; using existing official path so cache can be created"

                    return original_lmw(meta_model, model_path, loras, loader, registry, device, dtype, model_sd_ops, lora_load_device)
                finally:
                    dt = time.perf_counter() - t0
                    if loras_tuple:
                        counters["lora_gap_candidate_total_s"] = float(counters.get("lora_gap_candidate_total_s", 0.0)) + dt
                        suffix = "; early_fast_path_attempted=YES" if early_attempted else ""
                        _two_stage_gap_event(ctx, "_load_model_weights:lora_model:end", f"duration={dt:.3f}s; {detail}{suffix}", torch, echo)
                    _summarize()

            setattr(sgb, "_load_model_weights", _fv_gap_load_model_weights)
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"_load_model_weights patch failed: {type(exc).__name__}: {exc}"

    # Wrap LoRA application. This should bracket the suspicious post-LoRA-load gap.
    try:
        original_apply = getattr(fuse_mod, "_framevision_gap_original_apply_loras", None)
        if original_apply is None:
            original_apply = getattr(fuse_mod, "apply_loras")
            setattr(fuse_mod, "_framevision_gap_original_apply_loras", original_apply)

            def _fv_gap_apply_loras(*args: Any, **kwargs: Any) -> Any:
                counters["apply_loras_calls"] = int(counters.get("apply_loras_calls", 0)) + 1
                model_sd = None
                lora_sd_and_strengths = None
                dtype = None
                destination_sd = None
                modified_keys: set[str] = set()
                cache_info = ctx.get("_ltx_lora_cache_current")
                try:
                    model_sd = kwargs.get("model_sd", args[0] if len(args) > 0 else None)
                    lora_sd_and_strengths = kwargs.get("lora_sd_and_strengths", args[1] if len(args) > 1 else None)
                    dtype = kwargs.get("dtype", args[2] if len(args) > 2 else None)
                    destination_sd = kwargs.get("destination_sd", None)
                    sd_len = len(getattr(model_sd, "sd", {}) or {})
                    lora_len = len(lora_sd_and_strengths or [])
                    modified_keys = _lora_cache_modified_keys(model_sd, lora_sd_and_strengths)
                    detail = (
                        f"base_tensors={sd_len}; lora_state_dicts={lora_len}; dtype={dtype}; "
                        f"destination_sd={'YES' if destination_sd is not None else 'NO'}; "
                        f"modified_keys={len(modified_keys)}; "
                        f"base_size={_fmt_bytes(int(getattr(model_sd, 'size', 0) or 0))}; "
                        f"base_device={getattr(model_sd, 'device', 'n/a')}"
                    )
                except Exception as exc:
                    detail = f"detail unavailable: {type(exc).__name__}: {exc}"
                _two_stage_gap_event(ctx, "apply_loras:start", detail, torch, echo)
                t0 = time.perf_counter()
                cache_loaded = False
                try:
                    # Fast path: if a previous run already built the same base+LoRA+strength cache,
                    # stream the fused tensors into the current state dict and skip official CPU fusion.
                    if (
                        _lora_cache_enabled()
                        and isinstance(cache_info, dict)
                        and model_sd is not None
                        and destination_sd is not None
                    ):
                        try:
                            cache_path = Path(str(cache_info.get("path", "")))
                            ctx["ltx_lora_fusion_cache_path"] = str(cache_path)
                            ctx["ltx_lora_fusion_cache_meta_path"] = str(cache_info.get("meta_path", ""))
                            ctx["ltx_lora_fusion_cache_key"] = str(cache_info.get("key", "n/a"))
                            _lora_cache_apply_cached_stage_limits("apply_loras:cache_load")
                            cache_loaded = _lora_cache_try_load(destination_sd, cache_path, modified_keys)
                            if cache_loaded:
                                ctx["ltx_lora_fusion_cache_status"] = str(ctx.get("ltx_lora_fusion_cache_status", "cache hit")) + "; guarded by Stage-2 block limit"

                                return destination_sd
                            if _lora_cache_mode() == "read":
                                msg = (
                                    "READ ONLY MISS: cache file was missing, incomplete, or failed validation; "
                                    "refusing official LoRA fusion/cache rebuild path"
                                )
                                ctx["ltx_lora_fusion_cache_status"] = msg
                                ctx["ltx_lora_fusion_cache_hit"] = "NO"
                                ctx["ltx_lora_fusion_cache_save"] = "skipped: read existing only mode"
                                _two_stage_gap_event(ctx, "lora_fusion_cache:read_only_miss_abort", msg, torch, echo)
                                raise RuntimeError(
                                    "FrameVision LTX LoRA cache is set to Read existing only, but the matching cache could not be loaded. "
                                    "Use Create/use or Rebuild to make a cache, or Off for the official non-cache LoRA path."
                                )
                        except RuntimeError:
                            raise
                        except Exception as exc:
                            ctx["ltx_lora_fusion_cache_status"] = f"cache fast path failed: {type(exc).__name__}: {exc}"
                            _two_stage_gap_event(ctx, "lora_fusion_cache:fast_path_failed", str(exc), torch, echo)
                            if _lora_cache_mode() == "read":
                                raise

                    # Cache-miss first build: official LTX normally receives a separate destination_sd and
                    # fills it with ~37GB of fused tensors while the base state dict is still alive. On 64GB
                    # systems that pushes Windows into pagefile crawl before saving even starts. When possible,
                    # fuse into the existing base state dict instead. This keeps the useful first cache build but
                    # avoids the extra full-size destination state dict.
                    call_kwargs = kwargs
                    inplace_used = False
                    if (
                        _lora_cache_enabled()
                        and _lora_cache_miss_inplace_enabled()
                        and not cache_loaded
                        and model_sd is not None
                        and destination_sd is not None
                        and "destination_sd" in kwargs
                    ):
                        try:
                            call_kwargs = dict(kwargs)
                            call_kwargs["destination_sd"] = model_sd
                            destination_sd = model_sd
                            inplace_used = True
                            ctx["ltx_lora_fusion_cache_miss_inplace_used"] = "YES"
                            _two_stage_gap_event(ctx, "lora_fusion_cache:miss_inplace", "cache miss: using base state dict as destination_sd to reduce first-build RAM/pagefile pressure", torch, echo)
                        except Exception as exc:
                            ctx["ltx_lora_fusion_cache_miss_inplace_used"] = f"NO: {type(exc).__name__}: {exc}"
                    if not inplace_used:
                        ctx.setdefault("ltx_lora_fusion_cache_miss_inplace_used", "NO")
                    if _lora_cache_enabled() and not cache_loaded and _lora_cache_mode() != "read":
                        _lora_cache_preclean("before official LoRA cache-miss fusion/build")
                    final_sd = original_apply(*args, **call_kwargs)

                    # Slow path completed. Save the fused modified tensors so future runs can skip this gap.
                    if (
                        _lora_cache_enabled()
                        and isinstance(cache_info, dict)
                        and final_sd is not None
                        and not cache_loaded
                    ):
                        try:
                            cache_path = Path(str(cache_info.get("path", "")))
                            meta_path = Path(str(cache_info.get("meta_path", "")))
                            meta = dict(cache_info.get("meta") or {})
                            cache_mode = _lora_cache_mode()
                            if cache_mode == "read":
                                ctx["ltx_lora_fusion_cache_save"] = "skipped: read existing only mode"
                            elif cache_mode == "rebuild" or not _lora_cache_entry_exists(cache_path):
                                _lora_cache_try_save(final_sd, cache_path, meta_path, meta, modified_keys)
                            else:
                                ctx["ltx_lora_fusion_cache_save"] = "skipped: cache already exists"
                        except Exception as exc:
                            ctx["ltx_lora_fusion_cache_save"] = f"failed: {type(exc).__name__}: {exc}"
                    return final_sd
                finally:
                    dt = time.perf_counter() - t0
                    counters["apply_loras_total_s"] = float(counters.get("apply_loras_total_s", 0.0)) + dt
                    _two_stage_gap_event(ctx, "apply_loras:end", f"duration={dt:.3f}s; cache_loaded={cache_loaded}; {detail}", torch, echo)
                    _summarize()

            setattr(fuse_mod, "apply_loras", _fv_gap_apply_loras)
            # single_gpu_model_builder imported apply_loras directly, so update its module global too.
            try:
                setattr(sgb, "apply_loras", _fv_gap_apply_loras)
            except Exception:
                pass
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"apply_loras patch failed: {type(exc).__name__}: {exc}"

    # Wrap fuse_lora_weights to see progress and count modified tensors.
    try:
        original_fuse_weights = getattr(fuse_mod, "_framevision_gap_original_fuse_lora_weights", None)
        if original_fuse_weights is None:
            original_fuse_weights = getattr(fuse_mod, "fuse_lora_weights")
            setattr(fuse_mod, "_framevision_gap_original_fuse_lora_weights", original_fuse_weights)

            def _fv_gap_fuse_lora_weights(*args: Any, **kwargs: Any):
                counters["fuse_lora_weights_calls"] = int(counters.get("fuse_lora_weights_calls", 0)) + 1
                _two_stage_gap_event(ctx, "fuse_lora_weights:start", "iterating modified base weights", torch, echo)
                t0 = time.perf_counter()
                count = 0
                last_log = t0
                try:
                    for item in original_fuse_weights(*args, **kwargs):
                        count += 1
                        counters["fused_weight_count"] = int(counters.get("fused_weight_count", 0)) + 1
                        now = time.perf_counter()
                        if count <= 3 or count % 25 == 0 or (now - last_log) > 20.0:
                            try:
                                key = str(item[0])
                            except Exception:
                                key = "?"
                            _two_stage_gap_event(ctx, "fuse_lora_weights:progress", f"modified_weights={count}; current_key={key}", torch, echo)
                            last_log = now
                        yield item
                finally:
                    dt = time.perf_counter() - t0
                    counters["fuse_lora_weights_total_s"] = float(counters.get("fuse_lora_weights_total_s", 0.0)) + dt
                    _two_stage_gap_event(ctx, "fuse_lora_weights:end", f"duration={dt:.3f}s; modified_weights={count}", torch, echo)
                    _summarize()

            setattr(fuse_mod, "fuse_lora_weights", _fv_gap_fuse_lora_weights)
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"fuse_lora_weights patch failed: {type(exc).__name__}: {exc}"

    # Wrap aggregate_lora_products to separate matmul/addmm time from outer Python overhead.
    try:
        original_aggregate = getattr(fuse_mod, "_framevision_gap_original_aggregate_lora_products", None)
        if original_aggregate is None:
            original_aggregate = getattr(fuse_mod, "aggregate_lora_products")
            setattr(fuse_mod, "_framevision_gap_original_aggregate_lora_products", original_aggregate)

            def _fv_gap_aggregate_lora_products(*args: Any, **kwargs: Any) -> Any:
                counters["aggregate_calls"] = int(counters.get("aggregate_calls", 0)) + 1
                t0 = time.perf_counter()
                try:
                    return original_aggregate(*args, **kwargs)
                finally:
                    dt = time.perf_counter() - t0
                    counters["aggregate_total_s"] = float(counters.get("aggregate_total_s", 0.0)) + dt
                    # Avoid logging every single call; summary is enough.
                    _summarize()

            setattr(fuse_mod, "aggregate_lora_products", _fv_gap_aggregate_lora_products)
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"aggregate_lora_products patch failed: {type(exc).__name__}: {exc}"

    # Wrap Module.load_state_dict only to time the large assign=True LTXModel assignment after fusion.
    try:
        nn_mod = getattr(torch, "nn", None)
        Module = getattr(nn_mod, "Module", None)
        if Module is not None and getattr(Module, "_framevision_gap_original_load_state_dict", None) is None:
            original_module_lsd = Module.load_state_dict
            setattr(Module, "_framevision_gap_original_load_state_dict", original_module_lsd)

            def _fv_gap_module_load_state_dict(self: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
                should_log = False
                detail = ""
                try:
                    cls_name = f"{type(self).__module__}.{type(self).__name__}"
                    sd_len = len(state_dict or {})
                    assign = bool(kwargs.get("assign", False))
                    should_log = ("ltx_core.model.transformer.model" in cls_name and sd_len > 1000) or (assign and sd_len > 3000)
                    if should_log:
                        detail = f"module={cls_name}; state_dict_tensors={sd_len}; assign={assign}; strict={kwargs.get('strict', 'default')}"
                        _two_stage_gap_event(ctx, "module.load_state_dict:start", detail, torch, echo)
                except Exception:
                    should_log = False
                t0 = time.perf_counter()
                try:
                    return original_module_lsd(self, state_dict, *args, **kwargs)
                finally:
                    if should_log:
                        dt = time.perf_counter() - t0
                        counters["module_load_state_dict_calls"] = int(counters.get("module_load_state_dict_calls", 0)) + 1
                        counters["module_load_state_dict_total_s"] = float(counters.get("module_load_state_dict_total_s", 0.0)) + dt
                        _two_stage_gap_event(ctx, "module.load_state_dict:end", f"duration={dt:.3f}s; {detail}", torch, echo)
                        _summarize()

            Module.load_state_dict = _fv_gap_module_load_state_dict
    except Exception as exc:
        ctx["ltx_two_stage_gap_profiler_errors"] = f"Module.load_state_dict patch failed: {type(exc).__name__}: {exc}"

    ctx["ltx_two_stage_gap_profiler_installed"] = "YES"
    ctx["ltx_two_stage_gap_profiler_errors"] = str(ctx.get("ltx_two_stage_gap_profiler_errors", "none") or "none")
    ctx["ltx_two_stage_gap_summary"] = "installed; waiting for two-stage LoRA transition"
    ctx.setdefault("notes", []).append(
        "Installed two-stage transition profiler: times LoRA load/apply/fusion, large Module.load_state_dict(assign=True), RAM/commit, and CUDA snapshots around the long pre-stage-2 gap."
    )

def _write_report(ctx: Dict[str, Any], report_path: Path, decision: str, next_step: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("==============================================================================")
    lines.append("FrameVision LTX 2.3 → VRAM Lab Integration")
    lines.append("==============================================================================")
    lines.append("Scope: LTX owns generation; VRAM Lab owns boundary telemetry/hooks when available.")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Python executable used: {ctx.get('python_executable', sys.executable)}")
    lines.append(f"Report path: {ctx.get('report_path', str(report_path))}")
    if ctx.get("deep_lifecycle_live_log_path"):
        lines.append(f"Deep lifecycle live log path: {ctx.get('deep_lifecycle_live_log_path')}")
    lines.append("")
    lines.append("LTX")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "selected_pipeline",
        "selected_module",
        "ltx_user_lora_fast_stream_forced",
        "ltx_user_lora_fast_stream_route",
        "ltx_user_lora_budget_adjust",
        "ltx_user_lora_budget_total_bytes",
        "ltx_user_lora_budget_total_gb",
        "ltx_user_lora_budget_stage1_deduction_gb",
        "ltx_user_lora_budget_stage2_deduction_gb",
        "ltx_user_lora_budget_original_stage1_gb",
        "ltx_user_lora_budget_effective_stage1_gb",
        "ltx_user_lora_budget_original_stage2_gb",
        "ltx_user_lora_budget_effective_stage2_gb",
        "ltx_user_lora_budget_original_hotset_budget_gb",
        "ltx_user_lora_budget_effective_hotset_budget_gb",
        "ltx_user_lora_budget_note",
        "ltx_user_lora_runtime_source_patch",
        "ltx_user_lora_runtime_source_residency",
        "ltx_user_lora_runtime_source_bytes",
        "ltx_user_lora_runtime_source_pairs",
        "ltx_user_lora_runtime_source_path",
        "ltx_user_lora_runtime_source_errors",
        "one_stage_lora_guard_status",
        "one_stage_lora_guard_removed",
        "selected_vram_lab_mode",
        "vram_profile_gb",
        "vram_profile_source",
        "ltx_auto_workflow_profile_enabled",
        "ltx_auto_workflow_profile_supported",
        "ltx_auto_workflow_profile_reason",
        "ltx_auto_workflow_profile_resolution",
        "ltx_auto_workflow_profile_max_frames",
        "ltx_auto_workflow_profile_hotset_gb",
        "ltx_auto_workflow_profile_stage1_gb",
        "ltx_auto_workflow_profile_stage2_gb",
        "ltx_main_profile_hot_window_gb",
        "ltx_main_hot_window_override_gb",
        "separate_stage2_block_limit",
        "stage1_block_size_limit_gb",
        "stage2_block_size_limit_gb",
        "stage2_limit_source",
        "active_block_limit_for_stage1_initial_denoise",
        "active_block_limit_for_stage2_refine_denoise",
        "last_active_stage_block_limit",
        "stage2_block_limit_runtime_errors",
        "ltx_disable_distilled_lora_parser_override",
        "ltx_disable_distilled_lora_parser_override_modules",
        "ltx_gemma_profile_target_vram_gb",
        "ltx_gemma_profile_hot_pin_gb",
        "ltx_gemma_profile_disk_slots",
        "ltx_root",
        "checkpoint_path",
        "gemma_root",
        "output_path",
        "output_path_checked",
        "output_exists",
        "image_preprocess_enabled",
        "image_preprocess_last_stage",
        "image_preprocess_cleaned_count",
        "original_image_path",
        "original_image_mode",
        "original_image_size",
        "cleaned_image_path",
        "cleaned_image_mode",
        "cleaned_image_size",
        "image_preprocess_error",
        "ltx_call_phase",
        "resolution",
        "frame_count",
        "frame_rate",
        "inference_steps",
        "seed",
        "generation_status",
        "generation_completed",
        "total_runtime",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    lines.append("")
    lines.append("VRAM Lab")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "allocator_config_requested",
        "allocator_config_note",
        "ltx_warning_filters",
        "vram_lab_hooks_module",
        "vram_lab_hooks_module_errors",
        "ltx_pinned_layout_guard_installed",
        "ltx_pinned_layout_guard_targets",
        "ltx_pinned_layout_guard_calls",
        "ltx_pinned_layout_guard_rewrites",
        "ltx_pinned_layout_guard_errors",
        "main_transformer_streaming_probe",
        "main_transformer_streaming_builder_reached",
        "main_transformer_streaming_builder_calls",
        "main_transformer_partial_streaming_installed",
        "main_transformer_partial_streaming_reason",
        "main_transformer_detected_block_path",
        "main_transformer_block_count",
        "main_transformer_per_block_size",
        "main_transformer_hot_cpu_budget",
        "main_transformer_hot_cpu_slots",
        "main_transformer_disk_slots",
        "main_transformer_streaming_staging_guard",
        "main_transformer_streaming_cpu_pin_memory",
        "main_transformer_streaming_disk_pin_memory",
        "main_transformer_streaming_estimated_pinned_bytes",
        "main_transformer_streaming_staging_note",
        "main_transformer_gpu_slots",
        "main_transformer_gpu_target",
        "main_transformer_full_cpu_pinned_layout_avoided",
        "main_transformer_streaming_probe_event_count",
        "main_transformer_streaming_probe_errors",
        "ltx_gemma_partial_stream_gate_installed",
        "ltx_gemma_partial_stream_gate_calls",
        "ltx_gemma_partial_stream_gate_hot_blocks",
        "ltx_gemma_partial_stream_gate_disk_slots",
        "ltx_gemma_partial_stream_gate_gpu_slots",
        "ltx_gemma_partial_stream_gate_target_vram_gb",
        "ltx_gemma_partial_stream_gate_alloc_before_gpu_pool",
        "ltx_gemma_partial_stream_gate_pinned_budget_gb",
        "ltx_gemma_partial_stream_gate_per_block",
        "ltx_gemma_partial_stream_gate_status",
        "ltx_gemma_startup_shared_guard",
        "ltx_gemma_startup_pin_staging",
        "ltx_gemma_startup_shared_guard_note",
        "ltx_gemma_partial_stream_gate_main_transformer",
        "ltx_gemma_partial_stream_gate_errors",
        "ltx_gemma_checkpoint_prefix_selected",
        "ltx_gemma_checkpoint_prefix_candidates",
        "ltx_gemma_checkpoint_block_indices",
        "ltx_gemma_checkpoint_non_block_keys",
        "ltx_gemma_checkpoint_prefix_note",
        "ltx_main_cpu_first_gate_installed",
        "ltx_main_cpu_first_gate_calls",
        "ltx_main_cpu_first_gate_built",
        "ltx_main_cpu_first_gate_non_block_cuda_moves",
        "ltx_main_cpu_first_gate_left_cpu_blocks",
        "ltx_main_cpu_first_gate_status",
        "ltx_main_cpu_first_gate_errors",
        "ltx_main_direct_load_gate_installed",
        "ltx_main_direct_load_gate_targets",
        "ltx_main_direct_load_gate_calls",
        "ltx_main_direct_load_gate_forced_cpu_loads",
        "ltx_main_direct_load_gate_skipped_non_main_loads",
        "ltx_main_direct_load_gate_active_weight_model",
        "ltx_main_direct_load_gate_checkpoint_independent",
        "ltx_main_direct_load_gate_last_reason",
        "ltx_main_direct_load_gate_x0_to_intercepts",
        "ltx_main_direct_load_gate_ltx_to_intercepts",
        "ltx_main_direct_load_gate_left_cpu_blocks",
        "ltx_main_direct_load_gate_non_block_cuda_moves",
        "ltx_main_direct_load_gate_status",
        "ltx_main_direct_load_gate_errors",
        "ltx_main_load_trace_installed",
        "ltx_main_load_trace_targets",
        "ltx_main_load_trace_event_count",
        "ltx_main_load_trace_first_main_transformer_event",
        "ltx_main_load_trace_first_main_transformer_load_state_dict",
        "ltx_main_load_trace_main_transformer_file_summary",
        "ltx_main_load_trace_errors",
        "ltx_default_device_guard",
        "ltx_batch_split_early_guard_installed",
        "ltx_batch_split_early_guard_attach_attempts",
        "ltx_batch_split_early_guard_attached",
        "ltx_batch_split_early_guard_blocks",
        "ltx_batch_split_early_guard_errors",
        "ltx_phase_retention_bridge_installed",
        "ltx_phase_retention_bridge_calls",
        "ltx_phase_retention_bridge_reused_stage2",
        "ltx_phase_retention_stage1_context",
        "ltx_phase_retention_stage1_enter_before",
        "ltx_phase_retention_stage1_enter_after",
        "ltx_phase_retention_stage2_reuse_before",
        "ltx_phase_retention_handoff_safe_mode",
        "ltx_phase_retention_stage2_runtime_retune",
        "ltx_phase_retention_stage2_retune_before",
        "ltx_phase_retention_stage2_retune_after",
        "ltx_phase_retention_stage2_handoff_cleanup",
        "ltx_phase_retention_stage2_handoff_cleanup_before",
        "ltx_phase_retention_stage2_handoff_cleanup_after",
        "ltx_phase_retention_stage2_deferred_trim",
        "ltx_phase_retention_stage2_deferred_trim_before",
        "ltx_phase_retention_stage2_deferred_trim_after",
        "ltx_phase_retention_stage2_deferred_trim_pending",
        "ltx_phase_retention_stage2_attach",
        "ltx_phase_retention_stage2_reuse_after",
        "ltx_phase_retention_release",
        "ltx_phase_retention_release_before",
        "ltx_phase_retention_release_after",
        "ltx_phase_retention_bridge_errors",
        "vram_profile_note",
        "vram_hot_window_override",
        "vram_residency_strategy_requested",
        "vram_stable_hotset_fraction_requested",
        "vram_stage1_stable_hotset_fraction_requested",
        "vram_stage2_stable_hotset_fraction_requested",
        "vram_stable_hotset_budget_gb_requested",
        "vram_residency_strategy",
        "vram_residency_active",
        "vram_stable_hotset_fraction",
        "vram_stage1_stable_hotset_fraction",
        "vram_stage2_stable_hotset_fraction",
        "vram_stable_hotset_budget_gb",
        "vram_stable_hotset_count",
        "vram_stable_hotset_bytes",
        "vram_stable_hotset_fraction_actual",
        "vram_stable_hotset_fraction_for_stage1_initial_denoise",
        "vram_stable_hotset_fraction_effective_for_stage1_initial_denoise",
        "vram_stable_hotset_count_for_stage1_initial_denoise",
        "vram_stable_hotset_bytes_for_stage1_initial_denoise",
        "vram_stable_hotset_fraction_for_stage2_refine_denoise",
        "vram_stable_hotset_fraction_effective_for_stage2_refine_denoise",
        "vram_stable_hotset_count_for_stage2_refine_denoise",
        "vram_stable_hotset_bytes_for_stage2_refine_denoise",
        "vram_transient_unload_count",
        "ltx_block_churn_policy",
        "ltx_block_churn_override_installed",
        "ltx_block_churn_hot_window_mode",
        "ltx_block_churn_emergency_floor_gb",
        "ltx_block_churn_errors",
        "vram_safe_hot_window_gb",
        "vram_edge_hot_window_gb",
        "aggressive_extra_gb",
        "vram_balanced_hot_window_gb",
        "vram_hot_block_budget",
        "vram_emergency_driver_free_floor",
        "vram_lowest_driver_free_during_hooks",
        "vram_highest_driver_used_during_hooks",
        "vram_hot_block_trim_count",
        "vram_emergency_trim_count",
        "vram_policy_warning",
        "ltx_runtime_hotset_watchdog_enabled",
        "ltx_runtime_hotset_watchdog_threshold_gb",
        "ltx_runtime_hotset_watchdog_hold_s",
        "ltx_runtime_hotset_watchdog_step_gb",
        "ltx_runtime_hotset_watchdog_hard_threshold_gb",
        "ltx_runtime_hotset_watchdog_hard_step_gb",
        "ltx_runtime_hotset_watchdog_min_gb",
        "ltx_runtime_hotset_watchdog_retune_count",
        "ltx_runtime_hotset_watchdog_status",
        "ltx_runtime_hotset_watchdog_pressure",
        "ltx_runtime_hotset_watchdog_last_driver_free",
        "ltx_runtime_hotset_watchdog_last_role",
        "ltx_runtime_hotset_watchdog_last_retune",
        "ltx_runtime_hotset_watchdog_errors",
        "boundary_finder_enabled",
        "hook_discovery_ran",
        "hook_attachment_attempted",
        "hook_attachment_status",
        "hooked_component_names",
        "hooked_block_count",
        "pre_forward_hook_calls",
        "post_forward_hook_calls",
        "block_load_count",
        "block_unload_count",
        "peak_cuda_during_hooked_execution",
        "candidate_module_classes_found",
        "candidate_module_paths_names",
        "hook_attachment_attempts",
        "module_call_tracer_enabled",
        "module_call_tracer_runtime_state",
        "module_call_total_calls",
        "module_call_unique_classes",
        "module_call_candidate_class_count",
        "module_call_likely_hook_candidates",
        "ltx_component_first_seen_summary",
        "module_call_tracer_failures",
        "deep_lifecycle_logger_enabled",
        "deep_lifecycle_event_count",
        "deep_lifecycle_logger_failures",
        "deep_lifecycle_top_call_counts",
        "cuda_before_torch_import",
        "cuda_after_torch_import",
        "cuda_before_ltx_run",
        "cuda_after_ltx_run",
        "cuda_after_final_cleanup",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    hist_watchdog = ctx.get("ltx_runtime_hotset_watchdog_history")
    if isinstance(hist_watchdog, list) and hist_watchdog:
        lines.append("")
        lines.append("Runtime hotset watchdog retunes")
        lines.append("------------------------------------------------------------------------------")
        for item in hist_watchdog[-40:]:
            lines.append(str(item))
    lines.append("")
    lines.append("Two-stage transition / LoRA gap profiler")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "ltx_two_stage_gap_profiler_installed",
        "ltx_two_stage_gap_profiler_errors",
        "ltx_two_stage_gap_event_count",
        "ltx_two_stage_gap_summary",
        "ltx_user_lora_cli_count",
        "ltx_user_lora_cli_summary",
        "ltx_user_lora_argv_count",
        "ltx_user_lora_argv_summary",
        "ltx_user_lora_forwarded",
        "ltx_final_argv_contains_lora",
        "ltx_final_argv_lora_pairs",
        "ltx_user_lora_runtime_logger",
        "ltx_user_lora_runtime_status",
        "ltx_user_lora_load_model_weights_calls",
        "ltx_user_lora_apply_loras_calls",
        "ltx_user_lora_fuse_lora_weights_calls",
        "ltx_user_lora_event_count",
        "ltx_lora_fusion_cache_mode",
        "ltx_lora_fusion_cache_early_fast_path",
        "ltx_lora_fusion_cache_early_hit",
        "ltx_lora_fusion_cache_early_skipped_original_lora_load",
        "ltx_lora_fusion_cache_early_skipped_official_apply_loras",
        "ltx_lora_fusion_cache_early_loaded_tensors",
        "ltx_lora_fusion_cache_early_load_time_s",
        "ltx_lora_fusion_cache_early_apply_time_s",
        "ltx_lora_fusion_cache_early_errors",
        "ltx_lora_fusion_cache_status",
        "cached_lora_stage_limits",
        "cached_lora_uses_stage_limits",
        "cached_fused_model_memory_route",
        "cached_path_uses_stage_limits",
        "cached_lora_cache_bypassed_hooks",
        "cached_lora_stage1_block_limit",
        "cached_lora_stage2_block_limit",
        "cached_lora_limit_last_boundary",
        "ltx_lora_fusion_cache_key",
        "ltx_lora_fusion_cache_path",
        "ltx_lora_fusion_cache_meta_path",
        "ltx_lora_fusion_cache_hit",
        "ltx_lora_fusion_cache_save",
        "ltx_lora_fusion_cache_load_device",
        "ltx_lora_fusion_cache_load_strategy",
        "ltx_lora_fusion_cache_last_breadcrumb",
        "ltx_lora_fusion_cache_last_key",
        "ltx_lora_fusion_cache_load_time_s",
        "ltx_lora_fusion_cache_save_time_s",
        "ltx_lora_fusion_cache_loaded_tensors",
        "ltx_lora_fusion_cache_saved_tensors",
        "ltx_lora_fusion_cache_max_files",
        "ltx_lora_fusion_cache_shard_gb",
        "ltx_lora_fusion_cache_shard_threshold_gb",
        "ltx_lora_fusion_cache_miss_inplace",
        "ltx_lora_fusion_cache_miss_inplace_used",
        "ltx_lora_fusion_cache_preclean",
        "ltx_lora_fusion_cache_preclean_status",
        "ltx_lora_fusion_cache_estimated_size",
        "ltx_lora_fusion_cache_cleanup",
        "ltx_lora_fusion_cache_cleanup_deleted",
        "ltx_lora_fusion_cache_cleanup_deleted_bytes",
        "ltx_lora_fusion_cache_remaining_files",
        "ltx_lora_fusion_cache_remaining_size",
        "ltx_lora_fusion_cache_cleanup_errors",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    gap_events = ctx.get("ltx_two_stage_gap_events") or []
    if gap_events:
        lines.append("")
        lines.append("Two-stage transition / LoRA gap events")
        lines.append("------------------------------------------------------------------------------")
        for item in gap_events:
            lines.append(str(item))
    user_lora_events = ctx.get("ltx_user_lora_events") or []
    if user_lora_events:
        lines.append("")
        lines.append("User LoRA visibility events")
        lines.append("------------------------------------------------------------------------------")
        for item in user_lora_events:
            lines.append(str(item))
    lines.append("")
    lines.append("Stage-1 block churn profiler")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "ltx_stage1_churn_profiler_installed",
        "ltx_stage1_churn_profiler_targets",
        "ltx_stage1_churn_profiler_errors",
        "ltx_stage1_churn_summary",
        "ltx_stage1_churn_always_cuda_after_blocks",
        "ltx_stage1_churn_always_cuda_after_bytes",
        "ltx_stage1_churn_top_cuda_after_blocks",
        "ltx_stage1_churn_stable_candidate_blocks",
        "ltx_stage1_move_profiler_installed",
        "ltx_stage1_move_profiler_targets",
        "ltx_stage1_move_profiler_errors",
        "ltx_stage1_move_summary",
        "ltx_stage1_move_always_cuda_blocks",
        "ltx_stage1_move_always_cpu_blocks",
        "ltx_stage1_move_always_churn_blocks",
        "ltx_stage1_move_top_cuda_blocks",
        "ltx_stage1_move_top_cpu_blocks",
        "ltx_stage1_move_cuda_call_sites",
        "ltx_stage1_move_cpu_call_sites",
        "ltx_stage1_move_meta_call_sites",
        "ltx_stage1_tail_retain_installed",
        "ltx_stage1_tail_retain_blocks",
        "ltx_stage1_tail_retain_release",
        "ltx_stage1_tail_retain_skipped_cpu_moves",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")

    lines.append("")
    lines.append("Stage-2 step / refinement profiler")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "ltx_stage2_step_profiler_installed",
        "ltx_stage2_step_profiler_targets",
        "ltx_stage2_step_profiler_errors",
        "ltx_stage2_step_event_count",
        "ltx_stage2_step_profiler_forward_calls",
        "ltx_stage2_refine_step_count",
        "ltx_stage2_refine_total_s",
        "ltx_stage2_refine_avg_step_s",
        "ltx_low_profile_after_stage1_cleanup",
        "ltx_low_profile_after_stage1_cleanup_trigger",
        "ltx_low_profile_after_stage1_cleanup_reason",
        "ltx_low_profile_after_stage1_cleanup_before",
        "ltx_low_profile_after_stage1_cleanup_after",
        "ltx_low_profile_after_stage1_cleanup_errors",
        "ltx_low_profile_after_stage2_cleanup",
        "ltx_low_profile_after_stage2_cleanup_trigger",
        "ltx_low_profile_after_stage2_cleanup_reason",
        "ltx_low_profile_after_stage2_cleanup_before",
        "ltx_low_profile_after_stage2_cleanup_after",
        "ltx_low_profile_after_stage2_cleanup_errors",
        "ltx_pre_upscaler_video_encoder_guard",
        "ltx_pre_upscaler_video_encoder_guard_reason",
        "ltx_pre_upscaler_video_encoder_guard_runtime",
        "ltx_pre_upscaler_video_encoder_guard_target_gb",
        "ltx_pre_upscaler_video_encoder_guard_before",
        "ltx_pre_upscaler_video_encoder_guard_trim",
        "ltx_pre_upscaler_video_encoder_guard_cleanup",
        "ltx_pre_upscaler_video_encoder_guard_after",
        "ltx_pre_upscaler_video_encoder_guard_errors",
        "ltx_stage2_step_summary",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    step_events = ctx.get("ltx_stage2_step_events") or []
    if step_events:
        lines.append("")
        lines.append("Stage-2 step / refinement events")
        lines.append("------------------------------------------------------------------------------")
        for item in step_events:
            lines.append(str(item))
    lines.append("")
    lines.append("Realtime denoise step timings")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "ltx_realtime_step_timer_installed",
        "ltx_realtime_step_timer_expected_steps",
        "ltx_realtime_step_event_count",
        "ltx_realtime_step_summary",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    realtime_step_events = ctx.get("ltx_realtime_step_events") or []
    if realtime_step_events:
        lines.append("")
        for item in realtime_step_events:
            lines.append(str(item))
    lines.append("")
    lines.append("Kernel / attention backend probe")
    lines.append("------------------------------------------------------------------------------")
    for key in [
        "kernel_probe_torch_version",
        "kernel_probe_gpu",
        "kernel_probe_torch_sdp",
        "kernel_probe_flash_attn",
        "kernel_probe_flash_attn_interface",
        "kernel_probe_triton",
        "kernel_probe_xformers",
        "kernel_probe_xformers_ops",
        "kernel_probe_sageattention",
        "ltx_attention_module_file",
        "ltx_attention_xformers_available",
        "ltx_attention_flash3_interface_available",
        "ltx_attention_default_callable",
        "ltx_attention_flash_attn_v2_note",
        "ltx_attention_backend_requested",
        "ltx_attention_backend_override_status",
        "ltx_attention_backend_selected",
        "ltx_flash2_import_status",
        "ltx_flash2_success_calls",
        "ltx_flash2_fallback_calls",
        "ltx_flash2_first_fallback_error",
        "ltx_flash2_last_fallback_reason",
        "ltx_sage_import_status",
        "ltx_sage_success_calls",
        "ltx_sage_fallback_calls",
        "ltx_sage_first_fallback_error",
        "ltx_sage_last_fallback_reason",
        "ltx_attention_init_probe_status",
        "ltx_attention_runtime_callable_counts",
        "ltx_attention_runtime_callable_examples",
        "ltx_attention_init_probe_runtime_error",
    ]:
        lines.append(f"{key.replace('_', ' ')}: {ctx.get(key, 'n/a')}")
    image_events = ctx.get("image_preprocess_events") or []
    if image_events:
        lines.append("")
        lines.append("Image preprocess breadcrumbs")
        lines.append("------------------------------------------------------------------------------")
        for item in image_events:
            lines.append(str(item))
    notes = ctx.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Notes")
        lines.append("------------------------------------------------------------------------------")
        for n in notes:
            lines.append(str(n))
    stages = ctx.get("boundary_trace_stages") or []
    if stages:
        lines.append("")
        lines.append("VRAM Lab boundary trace")
        lines.append("------------------------------------------------------------------------------")
        for s in stages:
            lines.append(str(s))
    trace_notes = ctx.get("boundary_trace_notes") or []
    if trace_notes:
        lines.append("")
        lines.append("Boundary notes")
        lines.append("------------------------------------------------------------------------------")
        for n in trace_notes:
            lines.append(str(n))
    first_seen = ctx.get("ltx_component_first_seen_stages") or []
    if first_seen:
        lines.append("")
        lines.append("LTX component first-seen CUDA snapshots")
        lines.append("------------------------------------------------------------------------------")
        for item in first_seen:
            lines.append(str(item))
    component_file_events = ctx.get("ltx_component_file_events") or []
    if component_file_events:
        lines.append("")
        lines.append("LTX component file resolver")
        lines.append("------------------------------------------------------------------------------")
        for item in component_file_events:
            lines.append(str(item))
        for key in [
            "ltx_component_file_override",
            "ltx_component_file_vocoder_status",
            "ltx_component_file_vocoder_path",
            "ltx_component_file_vocoder_original",
            "ltx_component_file_audiodecoder_status",
            "ltx_component_file_audiodecoder_path",
            "ltx_component_file_audiodecoder_original",
            "ltx_component_file_videoencoder_status",
            "ltx_component_file_videoencoder_path",
            "ltx_component_file_videoencoder_original",
            "ltx_component_file_videodecoder_status",
            "ltx_component_file_videodecoder_path",
            "ltx_component_file_videodecoder_original",
        ]:
            if key in ctx:
                lines.append(f"{key.replace('_', ' ')}: {ctx.get(key)}")
    load_trace_events = ctx.get("ltx_main_load_trace_events") or []
    if load_trace_events:
        lines.append("")
        lines.append("LTX main-load start trace")
        lines.append("------------------------------------------------------------------------------")
        for item in load_trace_events:
            lines.append(str(item))
    stream_probe_events = ctx.get("main_transformer_streaming_probe_events") or []
    if stream_probe_events:
        lines.append("")
        lines.append("Main transformer streaming probe events")
        lines.append("------------------------------------------------------------------------------")
        for item in stream_probe_events:
            lines.append(str(item))
    if ctx.get("module_call_top_classes"):
        lines.append("")
        lines.append("LTX module call discovery — top classes")
        lines.append("------------------------------------------------------------------------------")
        lines.append(str(ctx.get("module_call_top_classes")))
    if ctx.get("module_call_candidate_summary"):
        lines.append("")
        lines.append("LTX module call discovery — candidate classes")
        lines.append("------------------------------------------------------------------------------")
        lines.append(str(ctx.get("module_call_candidate_summary")))
    deep_events = ctx.get("deep_lifecycle_events") or []
    if deep_events:
        lines.append("")
        lines.append("Deep lifecycle logger — CUDA / Module transitions")
        lines.append("------------------------------------------------------------------------------")
        for item in deep_events:
            lines.append(str(item))
    if ctx.get("exception"):
        lines.append("")
        lines.append("Exception")
        lines.append("------------------------------------------------------------------------------")
        lines.append(str(ctx.get("exception")))
    lines.append("")
    lines.append(f"PASS/WARN/FAIL decision: {decision}")
    lines.append(f"Next recommended step: {next_step}")
    lines.append(f"LTX VRAM LAB INTEGRATION RESULT: {decision}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    selected_module = _selected_module(args)
    report_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = _dated_output_path(Path(args.report_path), stamp=report_stamp)
    # Deep lifecycle live logs are part of the report set too. Date them so a
    # crash in a later clip cannot erase the only useful evidence from an earlier clip.
    try:
        args.deep_log_path = str(_dated_output_path(Path(getattr(args, "deep_log_path", DEEP_LIVE_LOG_PATH)), stamp=report_stamp))
    except Exception:
        args.deep_log_path = str(DEEP_LIVE_LOG_PATH)
    profile_gb, ltx_vram_profile = _resolve_ltx_vram_profile(getattr(args, "vram_profile", "24"))
    ltx_auto_workflow_settings = _apply_ltx_cli_auto_workflow_profile(args, profile_gb, ltx_vram_profile)
    main_hot_window_override = getattr(args, "main_hot_window_gb", None)
    if main_hot_window_override is not None:
        try:
            override_gb = float(main_hot_window_override)
        except Exception:
            override_gb = 0.0
        if override_gb > 0.0:
            ltx_vram_profile["main_hot_window_gb"] = override_gb
            ltx_vram_profile["main_hot_window_override_gb"] = override_gb
            ltx_vram_profile["note"] = (
                str(ltx_vram_profile.get("note", "selected LTX VRAM profile"))
                + f"; UI/CLI override: main/distilled safetensor hot-window set to {override_gb:.1f} GB"
            )
    stage2_block_limit_raw = getattr(args, "stage2_block_size_limit_gb", None)
    try:
        if stage2_block_limit_raw is None:
            stage2_block_limit_gb = float(ltx_vram_profile.get("stage2_hot_window_gb", 0.0) or 0.0)
        else:
            stage2_block_limit_gb = float(stage2_block_limit_raw)
    except Exception:
        stage2_block_limit_gb = 0.0
    stage2_block_limit_enabled = (stage2_block_limit_gb > 0.0 and str(getattr(args, "pipeline", "")).strip() in TWO_STAGE_PIPELINE_SET)

    residency_strategy = str(getattr(args, "vram_residency_strategy", "planned_hotset") or "planned_hotset").strip().lower()
    if residency_strategy not in {"rolling", "planned_hotset"}:
        residency_strategy = "planned_hotset"
    low_profile_residency_override = "none"
    if (
        str(getattr(args, "vram_lab", "off")).lower().strip() != "off"
        and int(profile_gb) < 24
        and residency_strategy == "planned_hotset"
        and not bool(getattr(args, "allow_low_profile_planned_hotset", False))
    ):
        low_profile_residency_override = f"forced rolling for {profile_gb}GB profile; planned_hotset currently proven only on 24GB"
        residency_strategy = "rolling"
    try:
        stable_hotset_fraction = max(0.10, min(2.00, float(getattr(args, "stable_hotset_fraction", 0.95) or 0.95)))
    except Exception:
        stable_hotset_fraction = 0.95
    try:
        raw_stage1_fraction = getattr(args, "stage1_stable_hotset_fraction", None)
        if raw_stage1_fraction is None:
            # Profile defaults are the source of truth when the caller does not
            # explicitly pass a stage fraction. This prevents stale parent env
            # values from silently overriding the proven 24 GB CFG 2 default.
            raw_stage1_fraction = ltx_vram_profile.get(
                "stage1_stable_hotset_fraction",
                os.environ.get("FRAMEVISION_VRAM_STAGE1_STABLE_HOTSET_FRACTION", stable_hotset_fraction),
            )
        stage1_stable_hotset_fraction = max(0.10, min(2.00, float(raw_stage1_fraction)))
    except Exception:
        stage1_stable_hotset_fraction = stable_hotset_fraction
    try:
        raw_stage2_fraction = getattr(args, "stage2_stable_hotset_fraction", None)
        if raw_stage2_fraction is None:
            # Same rule as Stage 1: use the selected profile's real default
            # first, then fall back to env only for profiles without a staged
            # value. The 24 GB default is 0.9. Older 0.80 crashed; 0.90 is now the selected default.
            raw_stage2_fraction = ltx_vram_profile.get(
                "stage2_stable_hotset_fraction",
                os.environ.get("FRAMEVISION_VRAM_STAGE2_STABLE_HOTSET_FRACTION", min(1.10, stable_hotset_fraction)),
            )
        stage2_stable_hotset_fraction = max(0.70, min(1.10, float(raw_stage2_fraction)))
    except Exception:
        stage2_stable_hotset_fraction = min(1.10, stable_hotset_fraction)
    try:
        stable_hotset_budget_gb = max(0.0, float(getattr(args, "stable_hotset_budget_gb", 0.0) or 0.0))
    except Exception:
        stable_hotset_budget_gb = 0.0

    stage2_block_limit_gb, stage2_block_limit_enabled, stable_hotset_budget_gb, user_lora_budget_info = _apply_user_lora_budget_reservation(
        args,
        ltx_vram_profile,
        stage2_block_limit_gb,
        stage2_block_limit_enabled,
        stable_hotset_budget_gb,
    )

    # These are generic VRAM Lab controls, not LTX-specific behavior. The hook
    # runtime reads them when it creates its residency manager. Stage-specific
    # values are supplied by the current adapter before attaching the core hooks.
    os.environ["FRAMEVISION_VRAM_RESIDENCY_STRATEGY"] = residency_strategy
    os.environ["FRAMEVISION_VRAM_STABLE_HOTSET_FRACTION"] = f"{stable_hotset_fraction:g}"
    os.environ["FRAMEVISION_VRAM_STAGE1_STABLE_HOTSET_FRACTION"] = f"{stage1_stable_hotset_fraction:g}"
    os.environ["FRAMEVISION_VRAM_STAGE2_STABLE_HOTSET_FRACTION"] = f"{stage2_stable_hotset_fraction:g}"
    os.environ["FRAMEVISION_VRAM_STABLE_HOTSET_BUDGET_GB"] = f"{stable_hotset_budget_gb:g}"
    try:
        raw_emergency_floor = getattr(args, "emergency_free_vram_floor_gb", None)
        if raw_emergency_floor is None:
            # Prefer the selected profile default when no CLI arg is supplied.
            # Parent env vars should not drag the 24 GB fast profile back to an
            # older 1.5 GB emergency floor by accident.
            raw_emergency_floor = ltx_vram_profile.get(
                "emergency_free_vram_floor_gb",
                os.environ.get("FRAMEVISION_LTX_EMERGENCY_FREE_VRAM_FLOOR_GB", 0.5),
            )
        emergency_free_vram_floor_gb = max(0.25, min(3.00, float(raw_emergency_floor)))
    except Exception:
        emergency_free_vram_floor_gb = 0.5
    os.environ["FRAMEVISION_LTX_EMERGENCY_FREE_VRAM_FLOOR_GB"] = f"{emergency_free_vram_floor_gb:g}"
    os.environ["FRAMEVISION_VRAM_EMERGENCY_FREE_FLOOR_GB"] = f"{emergency_free_vram_floor_gb:g}"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx: Dict[str, Any] = {
        "selected_pipeline": str(args.pipeline),
        "selected_module": selected_module,
        "ltx_fast_ic_lora_route": (
            "YES"
            if (str(selected_module).strip() == "ltx_pipelines.ic_lora" and not bool(getattr(args, "msr_enabled", False)))
            else "NO"
        ),
        "ltx_fast_ic_lora_neutral_conditioning": (
            "YES: tiny neutral --video-conditioning strength 0 used for fast IC-LoRA route"
            if (str(selected_module).strip() == "ltx_pipelines.ic_lora" and not bool(getattr(args, "msr_enabled", False)))
            else "NO"
        ),
        "ltx_fast_ic_lora_neutral_frames": str(int(getattr(args, "fast_iclora_neutral_frames", 17) or 17)),
        "ltx_fast_ic_lora_neutral_max_dim": str(int(getattr(args, "fast_iclora_neutral_max_dim", 320) or 320)),
        "ltx_safe_audio_load_requested": "YES" if bool(getattr(args, "safe_ltx_audio_load", False)) else "NO",
        "ltx_safe_audio_load_status": "not triggered",
        "ltx_distilled_lora_permanent_skip": str(getattr(args, "_distilled_lora_permanent_skip", "YES: official distilled LoRA path disabled by FrameVision wrapper")),
        "ltx_distilled_lora_permanent_skip_removed": str(getattr(args, "_distilled_lora_permanent_skip_removed", "none")),
        "ltx_user_lora_fast_stream_forced": "YES" if bool(getattr(args, "_user_lora_fast_stream_forced", False)) else "NO",
        "ltx_user_lora_fast_stream_route": "forced main transformer block-streaming LoraSource path" if bool(getattr(args, "_user_lora_fast_stream_forced", False)) else "not needed",
        **user_lora_budget_info,
        "one_stage_lora_guard_status": str(getattr(args, "_one_stage_lora_guard_status", "not checked")),
        "one_stage_lora_guard_removed": str(getattr(args, "_one_stage_lora_guard_removed", "none")),
        "selected_vram_lab_mode": ("off" if str(args.vram_lab).lower().strip() == "off" else "safe"),
        "vram_profile_gb": str(profile_gb) if str(args.vram_lab).lower().strip() != "off" else "off",
        "vram_profile_source": "--vram-profile / FRAMEVISION_LTX_VRAM_PROFILE_GB",
        "ltx_main_profile_hot_window_gb": f"{float(ltx_vram_profile.get('main_hot_window_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "ltx_main_hot_window_override_gb": f"{float(ltx_vram_profile.get('main_hot_window_override_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "separate_stage2_block_limit": "ON" if stage2_block_limit_enabled else "OFF",
        "stage1_block_size_limit_gb": f"{float(ltx_vram_profile.get('main_hot_window_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "stage2_block_size_limit_gb": f"{stage2_block_limit_gb:.1f}" if stage2_block_limit_enabled else "fallback to Stage 1",
        "stage2_limit_source": "CLI/UI --stage2-block-size-limit-gb" if stage2_block_limit_enabled else "fallback",
        "_ltx_num_inference_steps": str(int(getattr(args, "num_inference_steps", 8) or 8)),
        "vram_residency_strategy_requested": str(getattr(args, "vram_residency_strategy", residency_strategy)),
        "vram_low_profile_residency_override": low_profile_residency_override,
        "vram_stable_hotset_fraction_requested": f"{stable_hotset_fraction:.2f}",
        "vram_stage1_stable_hotset_fraction_requested": f"{stage1_stable_hotset_fraction:.2f}",
        "vram_stage2_stable_hotset_fraction_requested": f"{stage2_stable_hotset_fraction:.2f}",
        "vram_stable_hotset_budget_gb_requested": f"{stable_hotset_budget_gb:.1f}",
        "vram_emergency_driver_free_floor_requested_gb": f"{emergency_free_vram_floor_gb:.2f}",
        "vram_emergency_driver_free_floor": f"{emergency_free_vram_floor_gb:.2f} GB",
        "vram_residency_strategy": residency_strategy,
        "vram_residency_active": "YES" if (str(args.vram_lab).lower().strip() != "off" and residency_strategy == "planned_hotset") else "NO",
        "vram_stable_hotset_fraction": f"{stable_hotset_fraction:.2f}",
        "vram_stage1_stable_hotset_fraction": f"{stage1_stable_hotset_fraction:.2f}",
        "vram_stage2_stable_hotset_fraction": f"{stage2_stable_hotset_fraction:.2f}",
        "vram_stable_hotset_budget_gb": f"{stable_hotset_budget_gb:.1f}",
        "active_block_limit_for_stage1_initial_denoise": f"{float(ltx_vram_profile.get('main_hot_window_gb', 0.0)):.1f} GB" if str(args.vram_lab).lower().strip() != "off" else "0.0 GB",
        "active_block_limit_for_stage2_refine_denoise": (f"{stage2_block_limit_gb:.1f} GB" if stage2_block_limit_enabled else "fallback to Stage 1"),
        "stage2_block_limit_runtime_errors": "none",
        "ltx_disable_distilled_lora_parser_override": "not requested",
        "ltx_disable_distilled_lora_parser_override_modules": "n/a",
        "ltx_gemma_profile_target_vram_gb": f"{float(ltx_vram_profile.get('gemma_target_vram_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "ltx_gemma_profile_hot_pin_gb": f"{float(ltx_vram_profile.get('gemma_hot_pin_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "ltx_gemma_profile_disk_slots": str(int(ltx_vram_profile.get('gemma_disk_slots', 0))) if str(args.vram_lab).lower().strip() != "off" else "0",
        "aggressive_extra_gb": "0.0",
        "vram_safe_hot_window_gb": f"{float(ltx_vram_profile.get('main_hot_window_gb', 0.0)):.1f}" if str(args.vram_lab).lower().strip() != "off" else "0.0",
        "vram_edge_hot_window_gb": "0.0",
        "vram_hot_window_override": ("not attempted" if str(args.vram_lab).lower().strip() != "off" else "off"),
        "ltx_block_churn_policy": "default",
        "ltx_block_churn_override_installed": "NO: sticky_floor disabled for stability",
        "ltx_block_churn_hot_window_mode": "default rolling hot-window",
        "ltx_block_churn_emergency_floor_gb": f"{emergency_free_vram_floor_gb:.2f}",
        "ltx_block_churn_errors": "none",
        "ltx_runtime_hotset_watchdog_enabled": "YES: runtime-only, no saved checkpoint profile",
        "ltx_runtime_hotset_watchdog_threshold_gb": f"{max(0.75, float(emergency_free_vram_floor_gb)):.2f}",
        "ltx_runtime_hotset_watchdog_hold_s": "3.0",
        "ltx_runtime_hotset_watchdog_step_gb": "0.5",
        "ltx_runtime_hotset_watchdog_min_gb": "1.5",
        "ltx_runtime_hotset_watchdog_retune_count": "0",
        "ltx_runtime_hotset_watchdog_status": "watching after denoise forwards",
        "ltx_runtime_hotset_watchdog_pressure": "none",
        "ltx_runtime_hotset_watchdog_last_driver_free": "n/a",
        "ltx_runtime_hotset_watchdog_last_role": "n/a",
        "ltx_runtime_hotset_watchdog_last_retune": "none",
        "ltx_runtime_hotset_watchdog_errors": "none",
        "ltx_runtime_hotset_watchdog_history": [],
        "vram_profile_note": (
            str(ltx_vram_profile.get("note", "selected LTX VRAM profile"))
            if str(args.vram_lab).lower().strip() != "off" else "off"
        ),
        "python_executable": sys.executable,
        "report_path": str(report_path),
        "deep_lifecycle_live_log_path": str(getattr(args, "deep_log_path", DEEP_LIVE_LOG_PATH)),
        "ltx_root": str(args.ltx_root),
        "checkpoint_path": str(args.checkpoint_path),
        "gemma_root": str(args.gemma_root),
        "output_path": str(output_path),
        "output_path_checked": "NO",
        "output_exists": "unknown",
        "image_preprocess_enabled": "NO",
        "image_preprocess_last_stage": "not started",
        "image_preprocess_cleaned_count": "0",
        "image_preprocess_error": "none",
        "original_image_path": "none",
        "original_image_mode": "none",
        "original_image_size": "none",
        "cleaned_image_path": "none",
        "cleaned_image_mode": "none",
        "cleaned_image_size": "none",
        "ltx_call_phase": "wrapper setup",
        "resolution": f"{int(args.width)}x{int(args.height)}",
        "frame_count": str(int(args.num_frames)),
        "frame_rate": str(int(args.frame_rate)),
        "inference_steps": str(int(args.num_inference_steps)),
        "seed": str(int(args.seed)),
        "vram_lab_hooks_module": "not attempted",
        "vram_lab_hooks_module_errors": "none",
        "ltx_pinned_layout_guard_installed": "not attempted",
        "ltx_pinned_layout_guard_targets": "none",
        "ltx_pinned_layout_guard_calls": "0",
        "ltx_pinned_layout_guard_rewrites": "0",
        "ltx_pinned_layout_guard_errors": "none",
        "ltx_gemma_partial_stream_gate_installed": "not attempted",
        "ltx_gemma_partial_stream_gate_calls": "0",
        "ltx_gemma_partial_stream_gate_hot_blocks": "0",
        "ltx_gemma_partial_stream_gate_disk_slots": "0",
        "ltx_gemma_partial_stream_gate_gpu_slots": "0",
        "ltx_gemma_partial_stream_gate_target_vram_gb": "0",
        "ltx_gemma_partial_stream_gate_alloc_before_gpu_pool": "n/a",
        "ltx_gemma_partial_stream_gate_pinned_budget_gb": "0",
        "ltx_gemma_partial_stream_gate_per_block": "n/a",
        "ltx_gemma_partial_stream_gate_status": "not attempted",
        "ltx_gemma_startup_shared_guard": "not attempted",
        "ltx_gemma_startup_pin_staging": "not attempted",
        "ltx_gemma_startup_shared_guard_note": "n/a",
        "ltx_gemma_partial_stream_gate_main_transformer": "not attempted",
        "ltx_gemma_partial_stream_gate_errors": "none",
        "ltx_gemma_checkpoint_prefix_selected": "n/a",
        "ltx_gemma_checkpoint_prefix_candidates": "n/a",
        "ltx_gemma_checkpoint_block_indices": "0",
        "ltx_gemma_checkpoint_non_block_keys": "0",
        "ltx_gemma_checkpoint_prefix_note": "n/a",
        "ltx_main_cpu_first_gate_installed": "not attempted",
        "ltx_main_cpu_first_gate_calls": "0",
        "ltx_main_cpu_first_gate_built": "0",
        "ltx_main_cpu_first_gate_non_block_cuda_moves": "0",
        "ltx_main_cpu_first_gate_left_cpu_blocks": "0",
        "ltx_main_cpu_first_gate_status": "not attempted",
        "ltx_main_cpu_first_gate_errors": "none",
        "ltx_main_direct_load_gate_installed": "not attempted",
        "ltx_main_direct_load_gate_targets": "none",
        "ltx_main_direct_load_gate_calls": "0",
        "ltx_main_direct_load_gate_forced_cpu_loads": "0",
        "ltx_main_direct_load_gate_checkpoint_independent": "not attempted",
        "ltx_main_direct_load_gate_last_reason": "none",
        "ltx_main_direct_load_gate_x0_to_intercepts": "0",
        "ltx_main_direct_load_gate_ltx_to_intercepts": "0",
        "ltx_main_direct_load_gate_left_cpu_blocks": "0",
        "ltx_main_direct_load_gate_non_block_cuda_moves": "0",
        "ltx_main_direct_load_gate_status": "not attempted",
        "ltx_main_direct_load_gate_errors": "none",
        "ltx_main_load_trace_installed": "not attempted",
        "ltx_main_load_trace_targets": "none",
        "ltx_main_load_trace_event_count": "0",
        "ltx_main_load_trace_first_main_transformer_event": "none",
        "ltx_main_load_trace_first_main_transformer_load_state_dict": "none",
        "ltx_main_load_trace_main_transformer_file_summary": "none",
        "ltx_main_load_trace_errors": "none",
        "ltx_main_load_trace_events": [],
        "ltx_default_device_guard": "not attempted",
        "ltx_batch_split_early_guard_installed": "not attempted",
        "ltx_batch_split_early_guard_attach_attempts": "0",
        "ltx_batch_split_early_guard_attached": "0",
        "ltx_batch_split_early_guard_blocks": "0",
        "ltx_batch_split_early_guard_errors": "none",
        "ltx_phase_retention_bridge_installed": "not attempted",
        "ltx_phase_retention_bridge_calls": "0",
        "ltx_phase_retention_bridge_reused_stage2": "NO",
        "ltx_phase_retention_stage1_context": "not attempted",
        "ltx_phase_retention_stage1_enter_before": "n/a",
        "ltx_phase_retention_stage1_enter_after": "n/a",
        "ltx_phase_retention_stage2_reuse_before": "n/a",
        "ltx_phase_retention_stage2_runtime_retune": "not attempted",
        "ltx_phase_retention_stage2_retune_before": "n/a",
        "ltx_phase_retention_stage2_retune_after": "n/a",
        "ltx_phase_retention_stage2_attach": "not attempted",
        "ltx_phase_retention_stage2_reuse_after": "n/a",
        "ltx_phase_retention_release": "not attempted",
        "ltx_phase_retention_release_before": "n/a",
        "ltx_phase_retention_release_after": "n/a",
        "ltx_phase_retention_bridge_errors": "none",
        "hook_discovery_ran": "NO",
        "hook_attachment_attempted": "DISCOVERY: module-call tracer will look for LTX hook candidates",
        "hook_attachment_status": "pending discovery",
        "hooked_component_names": "none",
        "hooked_block_count": "0",
        "pre_forward_hook_calls": "0",
        "post_forward_hook_calls": "0",
        "block_load_count": "0",
        "block_unload_count": "0",
        "peak_cuda_during_hooked_execution": "n/a",
        "candidate_module_classes_found": "none",
        "candidate_module_paths_names": "none",
        "hook_attachment_attempts": "none",
        "module_call_tracer_enabled": "not started",
        "module_call_tracer_runtime_state": "not started",
        "module_call_total_calls": "0",
        "module_call_unique_classes": "0",
        "module_call_candidate_class_count": "0",
        "module_call_likely_hook_candidates": "none",
        "ltx_component_first_seen_summary": "none",
        "module_call_tracer_failures": "n/a",
        "deep_lifecycle_logger_enabled": "not requested",
        "deep_lifecycle_event_count": "0",
        "deep_lifecycle_logger_failures": "none",
        "deep_lifecycle_top_call_counts": "none",
        "deep_lifecycle_events": [],
        "kernel_probe_torch_version": "not attempted",
        "kernel_probe_gpu": "not attempted",
        "kernel_probe_torch_sdp": "not attempted",
        "kernel_probe_flash_attn": "not attempted",
        "kernel_probe_flash_attn_interface": "not attempted",
        "kernel_probe_triton": "not attempted",
        "kernel_probe_xformers": "not attempted",
        "kernel_probe_xformers_ops": "not attempted",
        "kernel_probe_sageattention": "not attempted",
        "ltx_attention_module_file": "not attempted",
        "ltx_attention_xformers_available": "not attempted",
        "ltx_attention_flash3_interface_available": "not attempted",
        "ltx_attention_default_callable": "not attempted",
        "ltx_attention_flash_attn_v2_note": "not attempted",
        "ltx_attention_init_probe_status": "not attempted",
        "ltx_attention_runtime_callable_counts": "not attempted",
        "ltx_attention_runtime_callable_examples": "not attempted",
        "ltx_attention_init_probe_runtime_error": "none",
        "ltx_attention_backend_requested": str(getattr(args, "attention_backend", "auto")),
        "ltx_attention_backend_override_status": "not attempted",
        "ltx_attention_backend_selected": "not attempted",
        "ltx_flash2_import_status": "not attempted",
        "ltx_flash2_success_calls": "0",
        "ltx_flash2_fallback_calls": "0",
        "ltx_flash2_first_fallback_error": "none",
        "ltx_flash2_last_fallback_reason": "none",
        "ltx_two_stage_gap_profiler_installed": "not attempted",
        "ltx_two_stage_gap_profiler_errors": "none",
        "ltx_two_stage_gap_event_count": "0",
        "ltx_two_stage_gap_summary": "not attempted",
        "ltx_two_stage_gap_events": [],
        "ltx_distilled_lora_disabled": "YES" if bool(getattr(args, "disable_distilled_lora", False)) else "NO",
        "ltx_distilled_lora_runtime_args": "skipped permanently by FrameVision wrapper" if bool(getattr(args, "disable_distilled_lora", False)) else str(getattr(args, "distilled_lora", [])),
        "ltx_lora_fusion_cache_mode": str(getattr(args, "lora_fusion_cache", "auto")),
        "ltx_lora_fusion_cache_status": "disabled: official distilled LoRA permanently skipped" if bool(getattr(args, "disable_distilled_lora", False)) else "not attempted",
        "cached_lora_stage_limits": "not attempted",
        "cached_lora_uses_stage_limits": "NO",
        "cached_fused_model_memory_route": "n/a",
        "cached_path_uses_stage_limits": "NO",
        "cached_lora_cache_bypassed_hooks": "unknown",
        "cached_lora_stage1_block_limit": "n/a",
        "cached_lora_stage2_block_limit": "n/a",
        "cached_lora_limit_last_boundary": "n/a",
        "ltx_lora_fusion_cache_key": "n/a",
        "ltx_lora_fusion_cache_path": "n/a",
        "ltx_lora_fusion_cache_meta_path": "n/a",
        "ltx_lora_fusion_cache_hit": "NO",
        "ltx_lora_fusion_cache_save": "not attempted",
        "ltx_lora_fusion_cache_load_device": "n/a",
        "ltx_lora_fusion_cache_last_breadcrumb": "none",
        "ltx_lora_fusion_cache_last_key": "none",
        "ltx_lora_fusion_cache_load_time_s": "0.000",
        "ltx_lora_fusion_cache_save_time_s": "0.000",
        "ltx_lora_fusion_cache_loaded_tensors": "0",
        "ltx_lora_fusion_cache_saved_tensors": "0",
        "ltx_lora_fusion_cache_max_files": str(max(1, int(getattr(args, "lora_fusion_cache_max", 2) or 2))),
        "ltx_lora_fusion_cache_shard_gb": str(float(getattr(args, "lora_fusion_cache_shard_gb", 4.0) or 4.0)),
        "ltx_lora_fusion_cache_shard_threshold_gb": str(float(getattr(args, "lora_fusion_cache_shard_threshold_gb", 8.0) or 8.0)),
        "ltx_lora_fusion_cache_miss_inplace": str(getattr(args, "lora_fusion_cache_miss_inplace", "auto") or "auto"),
        "ltx_lora_fusion_cache_preclean": str(getattr(args, "lora_fusion_cache_preclean", "auto") or "auto"),
        "ltx_lora_fusion_cache_preclean_status": "not attempted",
        "ltx_lora_fusion_cache_estimated_size": "0 B",
        "ltx_lora_fusion_cache_cleanup": "not attempted",
        "ltx_lora_fusion_cache_cleanup_deleted": "0",
        "ltx_lora_fusion_cache_cleanup_deleted_bytes": "0 B",
        "ltx_lora_fusion_cache_remaining_files": "0",
        "ltx_lora_fusion_cache_remaining_size": "0 B",
        "ltx_lora_fusion_cache_cleanup_errors": "none",
        "ltx_stage2_step_profiler_installed": "NO",
        "ltx_stage2_step_profiler_targets": "none",
        "ltx_stage2_step_profiler_errors": "none",
        "ltx_stage2_step_event_count": "0",
        "ltx_stage2_step_profiler_forward_calls": "0",
        "ltx_stage2_refine_step_count": "0",
        "ltx_stage2_refine_total_s": "0.000",
        "ltx_stage2_refine_avg_step_s": "0.000",
        "ltx_low_profile_after_stage1_cleanup": "NO",
        "ltx_low_profile_after_stage1_cleanup_trigger": "none",
        "ltx_low_profile_after_stage1_cleanup_reason": "none",
        "ltx_low_profile_after_stage1_cleanup_before": "none",
        "ltx_low_profile_after_stage1_cleanup_after": "none",
        "ltx_low_profile_after_stage1_cleanup_errors": "none",
        "ltx_low_profile_after_stage2_cleanup": "NO",
        "ltx_low_profile_after_stage2_cleanup_trigger": "none",
        "ltx_low_profile_after_stage2_cleanup_reason": "none",
        "ltx_low_profile_after_stage2_cleanup_before": "none",
        "ltx_low_profile_after_stage2_cleanup_after": "none",
        "ltx_low_profile_after_stage2_cleanup_errors": "none",
        "ltx_pre_upscaler_video_encoder_guard": "NO",
        "ltx_pre_upscaler_video_encoder_guard_reason": "none",
        "ltx_pre_upscaler_video_encoder_guard_runtime": "none",
        "ltx_pre_upscaler_video_encoder_guard_target_gb": "none",
        "ltx_pre_upscaler_video_encoder_guard_before": "none",
        "ltx_pre_upscaler_video_encoder_guard_trim": "none",
        "ltx_pre_upscaler_video_encoder_guard_cleanup": "none",
        "ltx_pre_upscaler_video_encoder_guard_after": "none",
        "ltx_pre_upscaler_video_encoder_guard_errors": "none",
        "ltx_video_decoder_finalize_guard": "NO",
        "ltx_video_decoder_finalize_guard_reason": "none",
        "ltx_video_decoder_finalize_guard_before": "none",
        "ltx_video_decoder_finalize_guard_after": "none",
        "ltx_video_decoder_finalize_guard_errors": "none",
        "ltx_stage2_step_summary": "none",
        "main_transformer_streaming_probe": "ON" if bool(getattr(args, "main_transformer_stream_probe", False) or getattr(args, "_user_lora_fast_stream_forced", False)) else "OFF",
        "main_transformer_partial_streaming_installed": "NO",
        "main_transformer_partial_streaming_reason": "forced by user LoRA fast path" if bool(getattr(args, "_user_lora_fast_stream_forced", False)) else ("disabled" if not bool(getattr(args, "main_transformer_stream_probe", False)) else "not attempted yet"),
        "main_transformer_streaming_builder_reached": "NO",
        "main_transformer_streaming_builder_calls": "0",
        "main_transformer_detected_block_path": "none",
        "main_transformer_block_count": "0",
        "main_transformer_per_block_size": "n/a",
        "main_transformer_hot_cpu_budget": "0",
        "main_transformer_hot_cpu_slots": "0",
        "main_transformer_disk_slots": "0",
        "main_transformer_streaming_staging_guard": "OFF",
        "main_transformer_streaming_cpu_pin_memory": "n/a",
        "main_transformer_streaming_disk_pin_memory": "n/a",
        "main_transformer_streaming_estimated_pinned_bytes": "0 B",
        "main_transformer_streaming_staging_note": "n/a",
        "main_transformer_gpu_slots": "0",
        "main_transformer_gpu_target": "n/a",
        "main_transformer_full_cpu_pinned_layout_avoided": "unknown",
        "main_transformer_streaming_probe_event_count": "0",
        "main_transformer_streaming_probe_errors": "none",
        "ltx_quiet_status_enabled": "YES" if bool(args.no_boundary_echo) else "NO",
        "ltx_quiet_status": "not started",
        "ltx_quiet_status_events": [],
        "ltx_realtime_step_timer_installed": "NO",
        "ltx_realtime_step_timer_expected_steps": str(getattr(args, "num_inference_steps", "")),
        "ltx_realtime_step_event_count": "0",
        "ltx_realtime_step_summary": "not started",
        "ltx_realtime_step_events": [],
        "notes": [],
    }

    _configure_latent_preview_runtime(args, ctx)

    if ltx_auto_workflow_settings is not None:
        ctx["ltx_auto_workflow_profile_enabled"] = "YES"
        ctx["ltx_auto_workflow_profile_supported"] = "YES" if bool(ltx_auto_workflow_settings.get("supported", False)) else "NO"
        ctx["ltx_auto_workflow_profile_reason"] = str(ltx_auto_workflow_settings.get("reason", ""))
        ctx["ltx_auto_workflow_profile_resolution"] = str(ltx_auto_workflow_settings.get("resolution_label", ""))
        ctx["ltx_auto_workflow_profile_max_frames"] = str(ltx_auto_workflow_settings.get("max_frames_for_profile_resolution", ""))
        ctx["ltx_auto_workflow_profile_hotset_gb"] = f"{float(ltx_auto_workflow_settings.get('hotset_gb', 0.0) or 0.0):.1f}"
        ctx["ltx_auto_workflow_profile_stage1_gb"] = f"{float(ltx_auto_workflow_settings.get('stage1_gb', 0.0) or 0.0):.1f}"
        ctx["ltx_auto_workflow_profile_stage2_gb"] = f"{float(ltx_auto_workflow_settings.get('stage2_gb', 0.0) or 0.0):.1f}"
    else:
        ctx["ltx_auto_workflow_profile_enabled"] = "NO"

    ctx["ltx_quiet_status_enabled"] = bool(args.no_boundary_echo)
    _ltx_quiet_status(ctx, "Preparing LTX run")

    if low_profile_residency_override != "none":
        ctx.setdefault("notes", []).append("Low-profile safety override: " + low_profile_residency_override)

    _install_ltx_runtime_warning_filters(ctx)

    if args.vram_lab != "off":
        _apply_ltx_profile_env_defaults(ctx, profile_gb, ltx_vram_profile)
        _apply_allocator_config_for_platform(ctx)
    else:
        ctx["allocator_config_requested"] = "off mode; not changed"
        ctx["allocator_config_note"] = "off mode; not changed"

    _init_user_lora_report(ctx, args)
    _record_user_lora_event(ctx, "wrapper_args", f"user LoRA groups={ctx.get('ltx_user_lora_cli_count', '0')}; {ctx.get('ltx_user_lora_cli_summary', 'none')}", None, echo=True)

    start = time.perf_counter()
    tracer = None
    module_call_tracer = None
    deep_logger = None
    hooks_mod = None
    hotset_watchdog_stop = None
    torch = None
    decision = "FAIL"
    next_step = "Wrapper failed before a usable LTX boundary report could be produced."
    original_argv = list(sys.argv)
    original_cwd = os.getcwd()
    original_default_device = None
    default_device_was_set = False

    try:
        if str(VRAM_LAB_DIR) not in sys.path:
            sys.path.insert(0, str(VRAM_LAB_DIR))
        ltx_root = Path(args.ltx_root)
        if str(ltx_root) not in sys.path:
            sys.path.insert(0, str(ltx_root))

        # LTX packages are source-layout packages under the repo copy, not
        # installed into the Python environment. Add them explicitly from the
        # selected --ltx-root so C:\ltx23 works without relying on an older
        # F:\ltx23 PYTHONPATH/model-tree side effect. Keep this narrow: path
        # setup only, no VRAM/residency behavior changes.
        ltx_repo = ltx_root / "models" / "ltx23" / "repos" / "LTX-2"
        ltx_src_paths = [
            ltx_repo / "packages" / "ltx-core" / "src",
            ltx_repo / "packages" / "ltx-pipelines" / "src",
            ltx_repo / "packages" / "ltx-trainer" / "src",
        ]
        added_ltx_paths = []
        missing_ltx_paths = []
        for p in ltx_src_paths:
            try:
                if p.exists():
                    sp = str(p)
                    if sp not in sys.path:
                        sys.path.insert(0, sp)
                    added_ltx_paths.append(sp)
                else:
                    missing_ltx_paths.append(str(p))
            except Exception as exc:
                missing_ltx_paths.append(f"{p} ({type(exc).__name__}: {exc})")
        if added_ltx_paths:
            ctx.setdefault("notes", []).append("Added LTX source paths from selected ltx-root: " + " | ".join(added_ltx_paths))
        if missing_ltx_paths:
            ctx.setdefault("notes", []).append("Missing optional/expected LTX source paths: " + " | ".join(missing_ltx_paths))

        if bool(getattr(args, "disable_distilled_lora", False)) and str(getattr(args, "pipeline", "")).strip() in TWO_STAGE_PIPELINE_SET:
            _install_ltx_disable_distilled_lora_parser_override(ctx)

        # Work from the LTX root so relative repo paths behave like the manual command.
        if ltx_root.exists():
            os.chdir(str(ltx_root))

        ctx["cuda_before_torch_import"] = _cuda_snapshot(None)
        _ltx_quiet_status(ctx, "Loading PyTorch / CUDA")
        import torch as _torch  # type: ignore
        torch = _torch
        ctx["cuda_after_torch_import"] = _cuda_snapshot(torch)
        _ltx_quiet_status(ctx, "Preparing attention backend")
        _probe_ltx_attention_module(ctx, torch)
        _install_ltx_attention_backend_override(ctx, str(getattr(args, "attention_backend", "auto")))
        _install_ltx_attention_init_probe(ctx)
        _install_ltx_scheduler_shift_override(ctx, getattr(args, "shift", None))

        # Wan2GP/MMGP mechanism: force the default construction device to CPU
        # before model loading. Wan2GP does this immediately before its
        # model_type_handler.load_model(...) call so accidental/default tensor
        # construction does not land on CUDA before MMGP owns residency. This is
        # a narrow wrapper-side equivalent for the official LTX module run: it
        # does not change test size, timing, output, or patch LTX internals.
        try:
            try:
                original_default_device = torch.get_default_device()
            except Exception:
                original_default_device = None
            torch.set_default_device("cpu")
            default_device_was_set = True
            ctx["ltx_default_device_guard"] = "YES: torch default device forced to CPU before LTX run"
            ctx.setdefault("notes", []).append(
                "Applied Wan2GP/MMGP-style default-device guard: torch.set_default_device('cpu') before LTX model loading, then restore after run."
            )
        except Exception as exc:
            ctx["ltx_default_device_guard"] = f"NO: {type(exc).__name__}: {exc}"

        # Install the one narrow LTX load-boundary guard before the LTX module is
        # executed. This targets the exact OOM path found in the 0.7.4 report and
        # leaves VRAM Lab denoise hooks unchanged.
        if args.vram_lab != "off":
            _ltx_quiet_status(ctx, "Preparing VRAM Lab hooks")
            main_stream_probe_enabled = bool(getattr(args, "main_transformer_stream_probe", False) or getattr(args, "_user_lora_fast_stream_forced", False))
            if bool(getattr(args, "_user_lora_fast_stream_forced", False)):
                os.environ["FRAMEVISION_LTX_USER_LORA_FAST_STREAM_REQUIRED"] = "1"
                ctx["ltx_user_lora_fast_stream_route"] = "forced main transformer block-streaming LoraSource path; SingleGPU full-fuse guarded"
                ctx.setdefault("notes", []).append("User LoRA detected: forcing bounded main-transformer block streaming so LoRA deltas are applied via LoraSource per block instead of full apply_loras/fuse_lora_weights.")
            else:
                os.environ.pop("FRAMEVISION_LTX_USER_LORA_FAST_STREAM_REQUIRED", None)
            _install_ltx_block_streaming_pinned_layout_guard(ctx)
            _install_ltx_gemma_partial_streaming_gate(ctx, allow_main_transformer_streaming=main_stream_probe_enabled)
            ctx["ltx_main_cpu_first_gate_status"] = "not used in this build; replaced by direct load_state_dict gate" if not main_stream_probe_enabled else "not used: main transformer streaming probe enabled"
            _install_ltx_main_load_start_trace(ctx, torch_module=torch, echo=not bool(args.no_boundary_echo), hooks_mod=hooks_mod)
            if main_stream_probe_enabled:
                _install_ltx_main_transformer_streaming_probe(ctx, torch_module=torch, echo=not bool(args.no_boundary_echo))
                ctx["ltx_main_direct_load_gate_status"] = "not installed: main transformer streaming probe enabled"
                ctx["ltx_main_direct_load_gate_installed"] = "NO"
            else:
                _install_ltx_main_direct_load_state_gate(ctx, torch_module=torch, echo=not bool(args.no_boundary_echo))

            # The LoRA fusion cache is a two-stage-only optimization for the
            # official distilled LoRA + spatial-upsampler refinement path.
            # Keep one-stage completely out of this machinery so normal T2V/I2V
            # runs cannot accidentally prepare, load, or save a huge fused-cache
            # file just because LoRA helpers are present in the environment.
            if bool(getattr(args, "disable_distilled_lora", False)):
                ctx["ltx_two_stage_gap_profiler_installed"] = "NO"
                ctx["ltx_two_stage_gap_summary"] = "disabled: official distilled LoRA permanently skipped; no official LoRA/cache path expected"
                ctx["ltx_lora_fusion_cache_status"] = "disabled: official distilled LoRA permanently skipped"
                ctx["ltx_lora_fusion_cache_mode"] = "off-for-permanent-no-lora"
                ctx.setdefault("notes", []).append(
                    "Skipped LoRA fusion cache/profiler because the official distilled LoRA path is permanently disabled in FrameVision."
                )
            elif str(getattr(args, "pipeline", "one_stage")) in {"two_stages", "two_stages_hq", "a2vid_two_stage"}:
                _install_ltx_two_stage_gap_profiler(ctx, torch_module=torch, echo=not bool(args.no_boundary_echo))
            else:
                ctx["ltx_two_stage_gap_profiler_installed"] = "NO"
                ctx["ltx_two_stage_gap_summary"] = "disabled: selected pipeline is one_stage; LoRA fusion cache is two-stage only"
                ctx["ltx_lora_fusion_cache_status"] = "disabled: selected pipeline is one_stage"
                ctx["ltx_lora_fusion_cache_mode"] = "off-for-one-stage"
                ctx.setdefault("notes", []).append(
                    "Skipped two-stage LoRA fusion cache/profiler because selected pipeline is one_stage."
                )

        if args.vram_lab != "off":
            try:
                _ltx_quiet_status(ctx, "Loading VRAM Lab hook module")
                hooks_mod = _load_vram_lab_hooks_module(ctx)
                ctx["_ltx_vram_hooks_module"] = hooks_mod
                if hooks_mod is None:
                    raise ModuleNotFoundError(str(ctx.get("vram_lab_hooks_module_errors", "VRAM Lab hooks module unavailable")))
                _apply_ltx_vram_hot_window_override(
                    ctx,
                    hooks_mod,
                    hot_window_gb=float(ltx_vram_profile.get("main_hot_window_gb", LTX_VRAM_PROFILE_MAIN_HOT_WINDOW_24GB)),
                    profile_gb=profile_gb,
                    profile=ltx_vram_profile,
                    emergency_floor_gb=float(emergency_free_vram_floor_gb),
                )
                if bool(getattr(args, "deep_lifecycle_log", False)):
                    try:
                        make_deep_lifecycle_logger = getattr(hooks_mod, "make_deep_lifecycle_logger")
                        deep_logger = make_deep_lifecycle_logger(
                            ctx=ctx,
                            label="ltx_deep",
                            torch_module=torch,
                            echo=not bool(args.no_boundary_echo),
                            interval=float(getattr(args, "deep_log_interval", 1.0) or 1.0),
                            max_events=int(getattr(args, "deep_log_max_events", 4000) or 4000),
                            live_path=str(getattr(args, "deep_log_path", DEEP_LIVE_LOG_PATH) or DEEP_LIVE_LOG_PATH),
                        )
                        deep_logger.start()
                        deep_logger.mark("selected_pipeline", str(args.pipeline))
                        deep_logger.mark("selected_module", selected_module)
                        deep_logger.mark("after_torch_import", "before default-device guard / LTX run")
                    except Exception as exc:
                        ctx["deep_lifecycle_logger_enabled"] = f"FAILED: {type(exc).__name__}: {exc}"
                        ctx.setdefault("notes", []).append(f"deep lifecycle logger unavailable: {type(exc).__name__}: {exc}")
                # Sticky-floor block churn test disabled for stability: it caused emergency trims/spill.
                if str(getattr(args, "block_churn_policy", "default") or "default").lower().strip() != "default":
                    ctx.setdefault("notes", []).append("Ignored --block-churn-policy: sticky_floor disabled after emergency trim/spill failures.")
                _install_ltx_block_churn_policy_override(
                    ctx=ctx,
                    hooks_mod=hooks_mod,
                    policy_name="default",
                    emergency_floor_gb=float(emergency_free_vram_floor_gb),
                )
                _install_ltx_batch_split_early_residency_guard(
                    ctx=ctx,
                    hooks_mod=hooks_mod,
                    torch_module=torch,
                    mode=str(args.vram_lab),
                    echo=not bool(args.no_boundary_echo),
                )
                _install_ltx_two_stage_phase_retention_bridge(
                    ctx=ctx,
                    torch_module=torch,
                    echo=not bool(args.no_boundary_echo),
                )
                # In-step watchdog polling is disabled. Retuning while PyTorch is
                # inside an active inference forward can mutate tensors/blocks at
                # an unsafe time and caused "Inference tensors do not track version
                # counter" failures. Keep watchdog retunes at safe boundaries only.
                hotset_watchdog_stop = None
                ctx["ltx_runtime_hotset_watchdog_poller"] = "DISABLED: unsafe during active forward; boundary-only watchdog active"
                make_boundary_tracer = getattr(hooks_mod, "make_boundary_tracer")
                make_module_call_tracer = getattr(hooks_mod, "make_module_call_tracer")
                tracer = make_boundary_tracer(ctx=ctx, label="ltx", torch_module=torch, echo=not bool(args.no_boundary_echo))
                tracer.mark("ltx_after_torch_import", "torch imported; before ltx_pipelines run")
                module_call_tracer = make_module_call_tracer(
                    ctx=ctx,
                    label="ltx_module_call",
                    torch_module=torch,
                    echo=not bool(args.no_boundary_echo),
                    max_candidates=240,
                    sample_every=500,
                    mode=str(args.vram_lab),
                )
                module_call_tracer.start()
                ctx["hook_attachment_attempted"] = "DISCOVERY: torch.nn.Module call tracer active during LTX run"
                ctx["hook_attachment_status"] = "DISCOVERY: candidate module classes will be reported"
            except Exception as exc:
                ctx.setdefault("notes", []).append(f"boundary/module tracer unavailable: {type(exc).__name__}: {exc}")
                tracer = None
                module_call_tracer = None

        if tracer:
            tracer.mark("ltx_before_module_run", f"entering {selected_module}")
        else:
            ctx["cuda_before_ltx_run"] = _cuda_snapshot(torch)
        if deep_logger is not None:
            try:
                deep_logger.mark("before_ltx_module_run", f"entering {selected_module}")
            except Exception:
                pass

        ctx["ltx_call_phase"] = "before image preprocess"
        _ltx_quiet_status(ctx, "Preparing input images")
        try:
            _normalize_ltx_input_images(args, ctx)
            ctx["original_image_path"] = _collapse_image_records(ctx, "original_path")
            ctx["original_image_mode"] = _collapse_image_records(ctx, "original_mode")
            ctx["original_image_size"] = _collapse_image_records(ctx, "original_size")
            ctx["cleaned_image_path"] = _collapse_image_records(ctx, "cleaned_path")
            ctx["cleaned_image_mode"] = _collapse_image_records(ctx, "cleaned_mode")
            ctx["cleaned_image_size"] = _collapse_image_records(ctx, "cleaned_size")
        except Exception as exc:
            ctx["ltx_call_phase"] = "image preprocess failed before LTX run"
            ctx["generation_status"] = f"failed before LTX image/latent encoding: {type(exc).__name__}: {exc}"
            raise
        ctx["ltx_call_phase"] = "after image preprocess before LTX run"
        sys.argv = _ltx_argv(args)
        _record_user_lora_argv(ctx, list(sys.argv), torch_module=torch, echo=True)
        ctx["ltx_call_phase"] = "during LTX image/latent encoding or later"
        _ltx_quiet_status(ctx, "Starting LTX pipeline")
        try:
            _install_ltx_runtime_warning_filters(ctx)
            _install_native_ltx_latent_preview_hook(ctx)
            _install_ltx_realtime_step_timer(
                ctx,
                expected_steps=int(getattr(args, "num_inference_steps", 0) or 0),
                echo=True,
            )
            if bool(getattr(args, "_user_lora_runtime_forward_hooks", False)):
                _install_ltx_user_lora_forward_hook_patch(ctx, torch_module=torch, echo=True)
            if bool(getattr(args, "_user_lora_fast_stream_forced", False)):
                _install_ltx_user_lora_runtime_source_patch(ctx, torch_module=torch, echo=True)
            _install_ltx_user_lora_runtime_logger(ctx, torch_module=torch, echo=True)
            _record_user_lora_event(ctx, "module_run", f"before run_module {selected_module}", torch, True)
            _ltx_lora_module_t0 = time.perf_counter()
            try:
                runpy.run_module(selected_module, run_name="__main__")
            finally:
                _record_user_lora_event(ctx, "module_run", f"after run_module/finally {selected_module}; duration={time.perf_counter() - _ltx_lora_module_t0:.3f}s", torch, True)
        finally:
            if deep_logger is not None:
                try:
                    deep_logger.mark("ltx_module_run_finally", f"{selected_module} finished/raised; entering wrapper post-run cleanup")
                except Exception:
                    pass
            if hotset_watchdog_stop is not None:
                try:
                    hotset_watchdog_stop()
                except Exception:
                    pass
                hotset_watchdog_stop = None
            if module_call_tracer is not None:
                try:
                    module_call_tracer.stop()
                    module_call_tracer.update_context(ctx)
                    _update_ltx_early_residency_runtimes(ctx, detach=False)
                    try:
                        module_call_tracer.detach()
                    except Exception:
                        pass
                except Exception as exc:
                    ctx.setdefault("notes", []).append(f"module call tracer stop/update failed: {type(exc).__name__}: {exc}")

        if tracer:
            tracer.mark("ltx_after_module_run", f"{selected_module} returned")
        if deep_logger is not None:
            try:
                deep_logger.mark("after_ltx_module_run", f"{selected_module} returned")
            except Exception:
                pass
        ctx["cuda_after_ltx_run"] = _cuda_snapshot(torch)
        _ltx_quiet_status(ctx, "Finalizing output")
        ctx["ltx_call_phase"] = "LTX run returned; post-run output/finalization checks"
        ctx["generation_completed"] = "YES"
        ctx["generation_status"] = "completed"
        if deep_logger is not None:
            try:
                deep_logger.mark("after_ltx_module_run_output_check", f"checking output path: {output_path}")
            except Exception:
                pass
        ctx["output_path_checked"] = str(output_path)
        if output_path.exists():
            ctx["output_exists"] = "YES"
        else:
            ctx["output_exists"] = "unknown: output path not found after run"
            ctx.setdefault("notes", []).append("The wrapper finished but the expected output path was not found.")
        if deep_logger is not None:
            try:
                deep_logger.mark("before_wrapper_cleanup", "detaching VRAM Lab runtime hooks and cleaning CUDA")
            except Exception:
                pass
        _ltx_quiet_status(ctx, "Cleaning up VRAM")
        _update_ltx_early_residency_runtimes(ctx, detach=True)
        _cleanup_cuda(torch)
        ctx["cuda_after_final_cleanup"] = _cuda_snapshot(torch)
        if deep_logger is not None:
            try:
                deep_logger.mark("after_final_cleanup", "success path cleanup complete")
                deep_logger.stop()
                deep_logger.update_context(ctx)
            except Exception as exc:
                ctx.setdefault("notes", []).append(f"deep lifecycle logger stop failed: {type(exc).__name__}: {exc}")
        ctx["total_runtime"] = f"{time.perf_counter() - start:.3f}s"
        # PASS only when real shared VRAM hooks attached and fired. Otherwise
        # keep the successful LTX run as WARN with useful hook-target evidence.
        if args.vram_lab == "off":
            decision = "PASS"
            next_step = "Off-mode wrapper ran. Next run Safe/Balanced/Aggressive to collect VRAM Lab boundary telemetry."
        else:
            try:
                pre_calls = int(str(ctx.get("pre_forward_hook_calls", "0") or "0"))
                post_calls = int(str(ctx.get("post_forward_hook_calls", "0") or "0"))
                block_count = int(str(ctx.get("hooked_block_count", "0") or "0"))
            except Exception:
                pre_calls = post_calls = block_count = 0
            if block_count > 0 and pre_calls > 0 and post_calls > 0:
                decision = "PASS"
                next_step = "LTX completed and shared VRAM Lab hooks attached/fired. Confirm the hooked component is LTXModel/X0Model/BasicAVTransformerBlock, then test one slightly larger LTX run while watching load/decode/finalize memory."
            else:
                decision = "WARN"
                next_step = "LTX completed and hook discovery ran, but no safe hook point fired yet. Use the listed candidate module class/path with block samples as the next exact LTX hook target."
        return 0
    except BaseException as exc:
        if hotset_watchdog_stop is not None:
            try:
                hotset_watchdog_stop()
            except Exception:
                pass
            hotset_watchdog_stop = None
        if module_call_tracer is not None:
            try:
                module_call_tracer.stop()
                module_call_tracer.update_context(ctx)
                _update_ltx_early_residency_runtimes(ctx, detach=False)
                try:
                    module_call_tracer.detach()
                except Exception:
                    pass
            except Exception:
                pass
        if deep_logger is not None:
            try:
                deep_logger.mark("ltx_module_run_exception", f"{selected_module} failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass
            try:
                deep_logger.mark("exception_path", f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        ctx["generation_completed"] = "NO"
        if not str(ctx.get("generation_status", "")).startswith("failed before LTX image/latent encoding"):
            ctx["generation_status"] = f"failed: {type(exc).__name__}: {exc}"
        ctx["exception"] = traceback.format_exc()
        try:
            _update_ltx_early_residency_runtimes(ctx, detach=True)
            _cleanup_cuda(torch)
            ctx["cuda_after_final_cleanup"] = _cuda_snapshot(torch)
            if deep_logger is not None:
                try:
                    deep_logger.mark("after_final_cleanup", "failure path cleanup complete")
                    deep_logger.stop()
                    deep_logger.update_context(ctx)
                except Exception as log_exc:
                    ctx.setdefault("notes", []).append(f"deep lifecycle logger stop failed: {type(log_exc).__name__}: {log_exc}")
        except Exception:
            pass
        ctx["total_runtime"] = f"{time.perf_counter() - start:.3f}s"
        decision = "FAIL"
        if str(ctx.get("ltx_pinned_layout_guard_rewrites", "0")) not in ("0", "n/a", "none"):
            next_step = "Pinned-layout guard ran. If this still failed, inspect the new traceback for the next pre-hook load/finalize control point; do not rewrite 0.7.4 hooks."
        else:
            next_step = "Pinned-layout guard did not rewrite a pinned allocation before failure. Inspect the traceback for a changed LTX load target before attempting hooks."
        return 1
    finally:
        try:
            if default_device_was_set and torch is not None:
                try:
                    torch.set_default_device(original_default_device if original_default_device is not None else "cpu")
                except Exception:
                    torch.set_default_device("cpu")
        except Exception:
            pass
        try:
            sys.argv = original_argv
        except Exception:
            pass
        try:
            os.chdir(original_cwd)
        except Exception:
            pass
        try:
            if deep_logger is not None:
                try:
                    deep_logger.mark("wrapper_finally_before_report", "about to write integration report")
                    # If a rare path reached finally without stop(), force a live-file close marker.
                    deep_logger.update_context(ctx)
                except Exception:
                    pass
            _write_report(ctx, report_path, decision, next_step)
            print(f"[vram-lab-ltx] report: {report_path}", flush=True)
        except Exception as report_exc:
            print(f"[vram-lab-ltx] failed to write report: {type(report_exc).__name__}: {report_exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
