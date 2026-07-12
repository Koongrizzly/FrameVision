
from __future__ import annotations

"""PySide6 UI helper for running native LTX 2.3 through FrameVision's VRAM Lab CLI.

Drop this file in:
    FrameVision/helpers/ltx23_ui.py

Expected companion files:
    FrameVision/helpers/ltx23_vram_lab_cli.py  (FP16/FP8 only)
    FrameVision/helpers/planner_ltx_int4.py     (INT4 + conditions/LoRA)

The widget keeps the proven FP16/FP8 CLI untouched and routes INT4 to a
separate helper instead of mixing quant code into native VRAM Lab / LTX logic. It can be imported into FrameVision or launched directly
for testing:
    python helpers/ltx23_ui.py
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QTextCursor, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent if HERE.name.lower() == "helpers" else HERE
DEFAULT_CLI_PATH = APP_ROOT / "helpers" / "ltx23_vram_lab_cli.py"
INT4_CLI_PATH = APP_ROOT / "helpers" / "planner_ltx_int4.py"
INT4_MODEL_RELATIVE = Path("models") / "ltx23_int4"
DEFAULT_LTX_ROOT = APP_ROOT
OLD_DEFAULT_LTX_ROOTS = {str(Path(r"C:\ltx23")).lower(), str(Path(r"C:\ltx")).lower()}
DEFAULT_PYTHON = DEFAULT_LTX_ROOT / "environments" / ".ltx23" / "python.exe"
DEFAULT_CHECKPOINT = DEFAULT_LTX_ROOT / "models" / "ltx23" / "distilled-1.1" / "ltx-2.3-22b-distilled-1.1.safetensors"
FP16_CHECKPOINT_RELATIVE = Path("models") / "ltx23" / "distilled-1.1" / "ltx-2.3-22b-distilled-1.1.safetensors"
FP8_CHECKPOINT_RELATIVE = Path("models") / "ltx23" / "fp8" / "ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA_ROOT = DEFAULT_LTX_ROOT / "models" / "ltx23" / "text_encoder" / "lightricks_gemma_original"
DEFAULT_OUTPUT_DIR = DEFAULT_LTX_ROOT / "output" / "ltx_ui"
DEFAULT_REPORT_PATH = APP_ROOT / "tools" / "vram_lab" / "ltx_vram_lab_integration_report.txt"
DEFAULT_DEEP_LOG_PATH = APP_ROOT / "tools" / "vram_lab" / "ltx_deep_lifecycle_latest.txt"
DEFAULT_SETTINGS_PATH = APP_ROOT / "presets" / "setsave" / "ltx23_ui.json"
DEFAULT_FRAME_FFMPEG = APP_ROOT / "presets" / "bin" / "ffmpeg.exe"
DEFAULT_VIDEO_FRAME_DIR = APP_ROOT / "temp" / "ltx23_video_frames"
VIDEO_FILE_FILTER = "Videos (*.mp4 *.mov *.mkv *.webm *.avi *.m4v);;All files (*.*)"
QUANTIZATION_MODE_CHOICES = ["None", "Auto", "fp8-scaled-mm", "fp8-cast"]
QUANTIZATION_MODE_AUTO = "Auto"
QUANTIZATION_MODE_NONE = "None"
LTX_STANDARD_FRAME_LIMIT = 241
LTX_EXTENDED_FRAME_LIMIT = 1201


def fixed_settings_path() -> Path:
    """Return the only settings path this helper is allowed to write to.

    No QSettings, no registry, no AppData/temp folder fallback, and no custom
    settings path. State is always stored inside the FrameVision root.
    """
    return (APP_ROOT / "presets" / "setsave" / "ltx23_ui.json").resolve()

RESOLUTION_PRESETS = [
    # Keep the preset list focused on the tuned Auto VRAM profiles.
    # Custom resolutions can still be typed manually because the combo is editable.
    "832x512",
    "512x832",
    "1280x704",
    "704x1280",
    "1920x1088",
    "1088x1920",
]

BAD_RESOLUTION_REPLACEMENTS = {
    "416x256": "832x512",
    "448x256": "832x512",
    "640x384": "832x512",
    "704x416": "832x512",
    "704x448": "832x512",
    "768x448": "832x512",
    "832x480": "832x512",
    "1024x576": "1280x704",
    "1088x640": "1280x704",
    "1344x768": "1280x704",
    "1536x864": "1920x1088",
    "1536x896": "1920x1088",
}

# Must match the defaults in helpers/ltx23_vram_lab_cli.py.
# 0.0 in the UI means: use the selected profile default.
LTX_MAIN_HOT_WINDOW_DEFAULTS_GB = {
    "24": 16.5,
    "16": 8.0,
    "12": 4.4,
}

# one-phase uses a lighter default hot window than the two-phase workflows.
# User asked for each one-phase profile default to be 2 GB lower, while still
# preserving separate last-used values when switching workflows.
LTX_ONE_STAGE_MAIN_HOT_WINDOW_DEFAULTS_GB = {
    key: max(0.0, float(value) - 2.0)
    for key, value in LTX_MAIN_HOT_WINDOW_DEFAULTS_GB.items()
}

# Stage 2/refine uses its own per-profile block limit.
# Keep this visible as a real number in the UI instead of placeholder text.
LTX_STAGE2_HOT_WINDOW_DEFAULTS_GB = {
    "24": 12.5,
    "16": 4.0,
    "12": 2.0,
}

# Planned hotset is currently proven only on the 24 GB speed profile.
# Lower profiles use rolling by default for normal/saved settings, but the UI
# keeps the toggle available so test runs can explicitly send the CLI override.
LTX_PROFILE_RESIDENCY_DEFAULTS = {
    "24": "planned_hotset",
    "16": "rolling",
    "12": "rolling",
}


def _detect_ui_primary_cuda_vram_gb() -> float:
    """Best-effort GPU VRAM detection for the UI's auto profile preview."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        first = str(out).splitlines()[0].strip()
        if first:
            return float(first) / 1024.0
    except Exception:
        pass
    return 0.0


def _auto_ui_vram_profile_key() -> str:
    """Match the CLI auto policy: <16 -> 12, 16-22.99 -> 16, 23+ -> 24."""
    total = _detect_ui_primary_cuda_vram_gb()
    if total >= 23.0:
        return "24"
    if total >= 16.0:
        return "16"
    return "12"


# Migrate older saved profile defaults to the corrected defaults above.
# Real custom values are preserved unless they exactly match an obsolete default.
LTX_OLD_MAIN_HOT_WINDOW_DEFAULTS_GB = {
    "24": 12.0,
    "16": 6.0,
    "12": 4.0,
}
# Additional old values that appeared in saved JSON/default dataclass versions.
# Exact matches migrate to the current profile default; real custom values stay untouched.
LTX_OLD_MAIN_HOT_WINDOW_DEFAULT_VALUES_GB = {
    "24": {12.0, 16.0, 16.5, 17.0},
    "16": {6.0},
    "12": {4.0},
}


# Auto VRAM profile values for LTX 2.3.
# Values are tuned anchors from real testing. Do not apply extra hidden safety
# reductions here. two-phase uses Stage 1 for the first denoise and Stage 2
# for refinement. one-phase behaves closer to the two-phase Stage-2/refine
# pressure profile, so one-phase Auto uses the Stage-2 anchor value for both
# main hotset and Stage-1 block size.
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


def _ltx_auto_resolution_key(width: int, height: int) -> str:
    """Bucket current UI resolution into the tuned Auto profile table.

    Portrait presets use the same tuning bucket as their landscape partner, so
    classify by the short side rather than raw height.
    """
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
    """Return lower/upper anchors and interpolated 24 GB values."""
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
        # one-phase combines pressure that behaves closer to the two-phase
        # refine/Stage-2 limit.  Use the Stage-2 value as the whole one-phase
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
    """Find practical max frames where the active workflow's required budget stays alive."""
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


def calculate_ltx_auto_vram_settings(vram_profile_gb: int, resolution_key: str, frame_count: int, workflow_group: str = "two_stage") -> Dict[str, Any]:
    """Calculate UI/CLI Auto VRAM limits for LTX 2.3 profile automation."""
    try:
        profile = int(float(vram_profile_gb))
    except Exception:
        profile = 24
    profile = 24 if profile >= 24 else 16 if profile >= 16 else 12
    workflow = "one_stage" if str(workflow_group).strip().lower() == "one_stage" else "two_stage"
    workflow_label = "one-phase" if workflow == "one_stage" else "two-phase"
    key = str(resolution_key or "704p").strip().lower()
    if key not in LTX_AUTO_RESOLUTION_ANCHORS:
        key = "704p"
    table = LTX_AUTO_RESOLUTION_ANCHORS[key]
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

AUDIO_MODE_DISABLED = "Disabled"
AUDIO_MODE_REMUX = "Prompt only (add audio after generation)"
AUDIO_MODE_A2V = "Prompt + audio file (audio-conditioned LTX)"
AUDIO_MODE_CHOICES = [AUDIO_MODE_DISABLED, AUDIO_MODE_REMUX, AUDIO_MODE_A2V]
AUDIO_MODE_COMPAT_MAP = {
    "Disabled": AUDIO_MODE_DISABLED,
    "Add audio after generation": AUDIO_MODE_REMUX,
    "Remux after generation": AUDIO_MODE_REMUX,
    "Both pass and remux": AUDIO_MODE_REMUX,
    "Pass to LTX as extra": AUDIO_MODE_A2V,
    "Prompt only": AUDIO_MODE_REMUX,
    "Prompt only (add audio after generation)": AUDIO_MODE_REMUX,
    "Prompt + audio file": AUDIO_MODE_A2V,
    "Prompt + audio file (audio-conditioned LTX)": AUDIO_MODE_A2V,
}

LORA_CACHE_MODE_OFF = "Off (official LoRA, no cache)"
LORA_CACHE_MODE_READ = "Read existing only"
LORA_CACHE_MODE_SINGLE = "Create/use single fused cache"
LORA_CACHE_MODE_SHARDED = "Create/use sharded cache"
LORA_CACHE_MODE_REBUILD_SINGLE = "Rebuild single cache"
LORA_CACHE_MODE_REBUILD_SHARDED = "Rebuild sharded cache"
LORA_CACHE_MODE_CHOICES = [
    LORA_CACHE_MODE_READ,
    LORA_CACHE_MODE_SINGLE,
    LORA_CACHE_MODE_SHARDED,
    LORA_CACHE_MODE_OFF,
    LORA_CACHE_MODE_REBUILD_SINGLE,
    LORA_CACHE_MODE_REBUILD_SHARDED,
]
LORA_CACHE_MODE_COMPAT_MAP = {
    "": LORA_CACHE_MODE_READ,
    "auto": LORA_CACHE_MODE_SINGLE,
    "read": LORA_CACHE_MODE_READ,
    "readonly": LORA_CACHE_MODE_READ,
    "read existing": LORA_CACHE_MODE_READ,
    "read existing only": LORA_CACHE_MODE_READ,
    "off": LORA_CACHE_MODE_OFF,
    "off (official lora, no cache)": LORA_CACHE_MODE_OFF,
    "official lora": LORA_CACHE_MODE_OFF,
    "no cache": LORA_CACHE_MODE_OFF,
    "rebuild": LORA_CACHE_MODE_REBUILD_SINGLE,
    "single": LORA_CACHE_MODE_SINGLE,
    "create/use single fused cache": LORA_CACHE_MODE_SINGLE,
    "sharded": LORA_CACHE_MODE_SHARDED,
    "create/use sharded cache": LORA_CACHE_MODE_SHARDED,
    "rebuild single cache": LORA_CACHE_MODE_REBUILD_SINGLE,
    "rebuild sharded cache": LORA_CACHE_MODE_REBUILD_SHARDED,
}


# Official LTX 2.3 two-phase assets, from the bundled LTX-2 README.
# Keep downloads portable inside the selected LTX root; never rely on HF cache, AppData,
# temp folders, or hidden Windows user locations.
TWO_STAGE_ASSETS = {
    "distilled_lora": {
        "label": "Distilled LoRA",
        "filename": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "subdir": ("models", "ltx23", "loras"),
        # Prevent a stale/wrong saved path (for example the main 41 GB checkpoint)
        # from making the UI say "ready".
        "min_bytes": 1024 * 1024,
        "name_markers": ("distilled", "lora"),
    },
    "spatial_upsampler": {
        "label": "Spatial upsampler x2",
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "subdir": ("models", "ltx23", "spatial_upsampler"),
        "min_bytes": 1024 * 1024,
        "name_markers": ("spatial", "upscal"),
    },
}
TWO_STAGE_PIPELINES = {"two_stages", "two_stages_hq", "a2vid_two_stage"}

EXTRA_ARG_DISABLED = "Do not pass"
NEGATIVE_ARG_CHOICES = [EXTRA_ARG_DISABLED, "--negative-prompt", "--negative_prompt", "--neg-prompt"]
MEDIA_ARG_CHOICES = [
    EXTRA_ARG_DISABLED,
    "--input-media-path",
    "--input_media_path",
    "--image-path",
    "--image_path",
    "--conditioning-media-paths",
    "--conditioning_media_paths",
]
END_MEDIA_ARG_CHOICES = [
    EXTRA_ARG_DISABLED,
    "--end-media-path",
    "--end_media_path",
    "--image-end-path",
    "--image_end_path",
    "--last-frame-path",
    "--last_frame_path",
]
VIDEO_ARG_CHOICES = [
    EXTRA_ARG_DISABLED,
    "--video-path",
    "--video_path",
    "--input-video-path",
    "--input_video_path",
    "--control-video-path",
    "--control_video_path",
]
AUDIO_ARG_CHOICES = [
    EXTRA_ARG_DISABLED,
    "--audio-path",
    "--audio_path",
    "--audio-source",
    "--audio_source",
    "--audio-guide",
    "--audio_guide",
]
REFERENCE_ARG_CHOICES = [
    EXTRA_ARG_DISABLED,
    "--reference-image-paths",
    "--reference_image_paths",
    "--image-refs",
    "--image_refs",
]


def _wheel_focus_allows(widget: QWidget) -> bool:
    """Allow mouse-wheel changes only after the control was focused/clicked.

    This prevents accidental value changes while scrolling the page over a
    spinbox or combo box. The wheel still works normally after the user clicks
    into the control or tabs to it.
    """
    focus_widget = QApplication.focusWidget()
    if focus_widget is None:
        return False
    if focus_widget is widget:
        return True
    try:
        return widget.isAncestorOf(focus_widget)
    except Exception:
        return False


class WheelGuardComboBox(QComboBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not _wheel_focus_allows(self):
            event.ignore()
            return
        super().wheelEvent(event)


class WheelGuardSpinBox(QSpinBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not _wheel_focus_allows(self):
            event.ignore()
            return
        super().wheelEvent(event)


class WheelGuardDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not _wheel_focus_allows(self):
            event.ignore()
            return
        super().wheelEvent(event)


class WheelGuardSlider(QSlider):
    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not _wheel_focus_allows(self):
            event.ignore()
            return
        super().wheelEvent(event)


class CollapsibleSection(QWidget):
    """Small dependency-free collapsible section for PySide6 forms."""

    def __init__(self, title: str, parent: Optional[QWidget] = None, *, open_by_default: bool = False) -> None:
        super().__init__(parent)
        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(open_by_default)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if open_by_default else Qt.RightArrow)
        self.toggle.clicked.connect(self._on_toggled)

        self.content = QWidget(self)
        self.content.setVisible(open_by_default)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(18, 4, 0, 4)
        self.content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def addWidget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def addLayout(self, layout: QVBoxLayout) -> None:
        self.content_layout.addLayout(layout)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class PathRow(QWidget):
    changed = Signal(str)

    def __init__(
        self,
        label: str,
        default: str = "",
        parent: Optional[QWidget] = None,
        *,
        mode: str = "file",
        file_filter: str = "All files (*.*)",
        save_file: bool = False,
        open_action: str = "open",
    ) -> None:
        super().__init__(parent)
        self.label_text = label
        self.mode = mode
        self.file_filter = file_filter
        self.save_file = save_file

        self.label = QLabel(label)
        self.edit = QLineEdit(default)
        self._last_browse_dir = ""
        self.edit.textChanged.connect(self._on_text_changed)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse)
        self.open_action = str(open_action or "open").strip().lower()
        if self.open_action == "clear":
            self.open_btn = QPushButton("Clear")
            self.open_btn.setToolTip("Clear this path.")
            self.open_btn.clicked.connect(self.clear)
        else:
            self.open_btn = QPushButton("Open")
            self.open_btn.clicked.connect(self.open_location)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.label, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_btn, 0)
        layout.addWidget(self.open_btn, 0)

    def _on_text_changed(self, text: str) -> None:
        self._remember_folder_from_path(text)
        self.changed.emit(text)

    def _remember_folder_from_path(self, value: str) -> None:
        value = str(value or "").strip().strip('"')
        if not value:
            return
        try:
            p = Path(value).expanduser()
            folder = p if self.mode == "dir" else p.parent
            folder_text = str(folder)
            if folder_text and folder_text not in {".", ""}:
                self._last_browse_dir = folder_text
        except Exception:
            pass

    def set_last_used_folder(self, folder: str) -> None:
        folder = str(folder or "").strip()
        if folder:
            self._last_browse_dir = folder

    def last_used_folder(self) -> str:
        return self._last_browse_dir

    def _dialog_start_path(self) -> str:
        current = self.text()
        if current:
            if self.mode == "dir":
                return current
            try:
                return str(Path(current).expanduser().parent)
            except Exception:
                return current
        if self._last_browse_dir:
            return self._last_browse_dir
        return str(APP_ROOT)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value or "")

    def path(self) -> Path:
        return Path(self.text()).expanduser()

    def clear(self) -> None:
        self.edit.clear()


    def browse(self) -> None:
        start = self._dialog_start_path()
        selected = ""
        if self.mode == "dir":
            selected = QFileDialog.getExistingDirectory(self, self.label_text, start)
        elif self.save_file:
            selected, _ = QFileDialog.getSaveFileName(self, self.label_text, self.text() or start, self.file_filter)
        else:
            selected, _ = QFileDialog.getOpenFileName(self, self.label_text, self.text() or start, self.file_filter)
        if selected:
            self._remember_folder_from_path(selected)
            self.setText(selected)

    def open_location(self) -> None:
        p = self.path()
        target = p if p.is_dir() else p.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class MediaPathRow(PathRow):
    """PathRow variant for media inputs: same PathRow, plus a Use Current button."""

    def __init__(
        self,
        label: str,
        default: str = "",
        parent: Optional[QWidget] = None,
        *,
        mode: str = "file",
        file_filter: str = "All files (*.*)",
        save_file: bool = False,
        open_action: str = "open",
    ) -> None:
        super().__init__(label, default, parent, mode=mode, file_filter=file_filter, save_file=save_file, open_action=open_action)
        self.use_current_btn = QPushButton("Use Current")
        self.use_current_btn.setToolTip("Use the current image, frame, or video from the FrameVision Media Player.")
        try:
            self.edit.setMinimumWidth(260)
            self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass

        layout = self.layout()
        if isinstance(layout, QHBoxLayout):
            try:
                layout.setSpacing(6)
                layout.removeWidget(self.edit)
                layout.removeWidget(self.browse_btn)
                layout.removeWidget(self.open_btn)
            except Exception:
                pass
            # Only the labels are aligned later by the owner. Buttons keep their natural size.
            layout.addWidget(self.browse_btn, 0)
            layout.addWidget(self.use_current_btn, 0)
            layout.addWidget(self.open_btn, 0)
            layout.addWidget(self.edit, 1)

class TwoStageAssetDownloadWorker(QThread):
    progressChanged = Signal(str, int, str)
    logMessage = Signal(str)
    finishedWithResult = Signal(bool, str, dict)

    def __init__(self, assets: List[Dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.assets = assets

    def run(self) -> None:
        resolved: Dict[str, str] = {}
        try:
            for asset in self.assets:
                key = str(asset["key"])
                label = str(asset["label"])
                url = str(asset["url"])
                target = Path(str(asset["target"]))
                target.parent.mkdir(parents=True, exist_ok=True)

                if target.exists() and target.stat().st_size > 0:
                    resolved[key] = str(target)
                    self.progressChanged.emit(label, 100, f"Already present: {target.name}")
                    continue

                tmp = target.with_name(target.name + ".part")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

                self.logMessage.emit(f"Downloading {label}: {target.name}")
                self.logMessage.emit(f"Target: {target}")
                request = Request(url, headers={"User-Agent": "FrameVision-LTX23-UI"})
                downloaded = 0
                with urlopen(request, timeout=45) as response:
                    total_raw = response.headers.get("Content-Length") or "0"
                    try:
                        total = int(total_raw)
                    except Exception:
                        total = 0
                    self.progressChanged.emit(label, 0 if total else -1, f"Downloading {target.name}")
                    with tmp.open("wb") as out:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                percent = max(0, min(100, int(downloaded * 100 / total)))
                                msg = f"Downloading {target.name}: {downloaded / (1024**2):.1f} / {total / (1024**2):.1f} MB"
                                self.progressChanged.emit(label, percent, msg)
                            else:
                                msg = f"Downloading {target.name}: {downloaded / (1024**2):.1f} MB"
                                self.progressChanged.emit(label, -1, msg)

                if downloaded <= 0:
                    raise RuntimeError(f"Download produced an empty file: {target.name}")
                os.replace(str(tmp), str(target))
                resolved[key] = str(target)
                self.progressChanged.emit(label, 100, f"Finished {target.name}")

            self.finishedWithResult.emit(True, "two-phase assets are ready.", resolved)
        except Exception as exc:
            self.finishedWithResult.emit(False, f"{type(exc).__name__}: {exc}", resolved)


@dataclass
class LTX23UISettings:
    pipeline: str = "two_stages"
    vram_lab: str = "safe"
    fast_iclora_route: bool = True
    vram_profile: str = "24"
    main_hot_window_gb: float = 16.5
    stage2_block_size_limit_gb: float = 12.5
    vram_residency_strategy: str = "planned_hotset"
    stable_hotset_fraction: float = 1.15  # legacy/shared fallback; Stage 1 uses this when stage-specific value is missing
    stage1_stable_hotset_fraction: float = 1.15
    stage2_stable_hotset_fraction: float = 0.9
    stable_hotset_budget_gb: float = 0.0
    emergency_free_vram_floor_gb: float = 0.5
    vram_profile_block_size_limits: Dict[str, float] = field(default_factory=dict)
    vram_profile_block_size_limits_one_stage: Dict[str, float] = field(default_factory=dict)
    vram_profile_block_size_limits_two_stage: Dict[str, float] = field(default_factory=dict)
    prompt: str = "A tiny silver robot walks through a neon spaceship corridor, cinematic lighting, smooth motion, highly detailed."
    negative_prompt: str = ""
    latent_preview_enabled: bool = False
    latent_preview_mode: str = "Fast Latent RGB"
    latent_preview_rate: int = 8
    latent_preview_upscale: bool = False
    latent_preview_tae_decode: bool = False
    keep_latents: bool = False
    resolution: str = "1280x704"
    width: int = 1280
    height: int = 704
    frames: int = 121
    fps: int = 24
    steps: int = 8
    seed: int = 12345
    random_seed: bool = False
    snap_resolution: bool = True
    no_boundary_echo: bool = False
    deep_lifecycle_log: bool = False
    deep_log_interval: float = 1.0
    deep_log_max_events: int = 4000
    distilled_lora_path: str = ""
    distilled_lora_strength: float = 0.5
    user_lora_1_path: str = ""
    user_lora_1_strength: float = 1.0
    user_lora_2_path: str = ""
    user_lora_2_strength: float = 1.0
    user_lora_3_path: str = ""
    user_lora_3_strength: float = 1.0
    user_lora_4_path: str = ""
    user_lora_4_strength: float = 1.0
    disable_distilled_lora: bool = True
    disable_distilled_lora_extra_args: str = ""
    spatial_upsampler_path: str = ""
    auto_download_two_stage_assets: bool = True
    lora_fusion_cache_mode: str = LORA_CACHE_MODE_OFF
    lora_fusion_cache_shard_gb: float = 4.0
    lora_fusion_cache_shard_threshold_gb: float = 999999.0
    lora_fusion_cache_max_files: int = 2
    lora_fusion_cache_miss_inplace: bool = True
    main_transformer_stream_probe: bool = False
    start_media_path: str = ""
    start_video_path: str = ""
    start_image_frame: int = 0
    start_image_strength: float = 1.0
    end_media_path: str = ""
    end_video_path: str = ""
    glue_input_videos: bool = False
    end_image_strength: float = 1.0
    reference_images: str = ""
    reference_image_strength: float = 0.8
    ltx_normalize_input_image: bool = True
    source_video_path: str = ""
    audio_path: str = ""
    audio_mode: str = AUDIO_MODE_DISABLED
    audio_start_time: float = 0.0
    audio_max_duration: float = 0.0
    remux_audio_bitrate: str = "192k"
    remux_shortest: bool = True
    remux_replace_output: bool = False
    safe_ltx_audio_load: bool = True
    video_cfg_guidance_scale: float = 2.0
    scheduler_shift: float = 5.0
    video_stg_guidance_scale: float = 0.0
    video_rescale_scale: float = 0.7
    audio_cfg_guidance_scale: float = 1.0
    audio_stg_guidance_scale: float = 0.0
    audio_rescale_scale: float = 0.0
    a2v_guidance_scale: float = 1.0
    v2a_guidance_scale: float = 1.0
    video_skip_step: int = 0
    audio_skip_step: int = 0
    max_batch_size: int = 2
    enhance_prompt: bool = False
    enable_flash_attention: bool = False  # legacy compatibility; attention_backend is the real selector
    attention_backend: str = "auto"
    quantization_mode: str = QUANTIZATION_MODE_NONE
    custom_extra_args: str = ""
    ltx_root: str = str(DEFAULT_LTX_ROOT)
    python_exe: str = str(DEFAULT_PYTHON)
    cli_path: str = str(DEFAULT_CLI_PATH)
    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    gemma_root: str = str(DEFAULT_GEMMA_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    output_name: str = ""
    report_path: str = str(DEFAULT_REPORT_PATH)
    deep_log_path: str = str(DEFAULT_DEEP_LOG_PATH)
    ffmpeg_path: str = "ffmpeg"
    open_output_when_done: bool = False
    use_framevision_queue: bool = True
    last_used_folders: Dict[str, str] = field(default_factory=dict)

class LTX23RunnerWidget(QWidget):
    generationStarted = Signal(list)
    generationFinished = Signal(bool, str)
    outputReady = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, settings_path: Optional[Path | str] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LTX23RunnerWidget")
        # Deliberately ignore custom settings_path. This helper must only persist
        # inside FrameVision: /presets/setsave/ltx23_ui.json.
        self.settings_path = fixed_settings_path()
        self.process: Optional[QProcess] = None
        self.remux_process: Optional[QProcess] = None
        self.active_output_path: Optional[Path] = None
        self._loading = False
        self._last_command: List[str] = []
        self._last_success = False
        self._last_browse_dirs: Dict[str, str] = {}
        self._updating_vram_profile_controls = False
        self._vram_profile_block_limits: Dict[str, float] = {}
        self._vram_profile_block_limits_one_stage: Dict[str, float] = {}
        self._vram_profile_block_limits_two_stage: Dict[str, float] = {}
        self._active_vram_profile = "24"
        self._active_vram_limit_workflow_group = "one_stage"
        self._download_worker: Optional[TwoStageAssetDownloadWorker] = None
        self._pending_start_after_download = False
        self._prepared_start_frame_path = ""
        self._prepared_start_video_source = ""
        self._prepared_end_frame_path = ""
        self._prepared_end_video_source = ""
        self._fast_iclora_user_preference = True
        self._fast_iclora_auto_forced_off = False
        self._fast_iclora_base_tooltip = (
            "Uses a faster native ltx_pipeline route for regular two-phase runs by feeding a tiny dummy conditioning clip. "
            "This can speed up runs a lot, but if start images / image-to-video are not respected correctly, turn this OFF to fall back to the original route."
        )
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(650)
        self._save_timer.timeout.connect(lambda: self._save_settings(silent=True))
        self._vram_profile_autofill_timer = QTimer(self)
        self._vram_profile_autofill_timer.setSingleShot(True)
        self._vram_profile_autofill_timer.setInterval(650)
        self._vram_profile_autofill_timer.timeout.connect(lambda: self._apply_auto_vram_settings(update_hint=True, overwrite_manual=True))
        self._build_ui()
        self._load_settings()
        self._update_pipeline_dependent_controls()
        self._connect_auto_save()
        self._refresh_command_preview()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("FrameVision LTX 2.3 Runner")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("LTX 2.3 Runner")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)
        root.addLayout(header)

        # Keep the full form in one scroll area.
        # The first version used a vertical splitter for the main tabs vs. bottom
        # sections; when the bottom sections were opened/closed, Qt could preserve
        # an unlucky splitter size and leave the real controls hidden at height 0.
        # One scroll host is boring, predictable, and gives the user a normal
        # vertical scrollbar whenever the window is too small.
        self.content_scroll = QScrollArea(self)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.NoFrame)
        self.content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.content_scroll, 1)

        form_host = QWidget()
        form = QVBoxLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        form.addWidget(self.tabs)
        self._build_generation_tab()
        self._build_inputs_tab()
        self._build_quality_tab()
        self._build_audio_tab()
        self._build_advanced_tab()
        self._build_test_tab()

        self.locations_section = CollapsibleSection("Folder locations / executable paths", open_by_default=False)
        self._build_locations_section()
        form.addWidget(self.locations_section)

        self.log_section = CollapsibleSection("Log", open_by_default=False)
        self._build_log_section()
        form.addWidget(self.log_section)

        form.addStretch(1)
        self.content_scroll.setWidget(form_host)

        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run LTX")
        self.run_btn.clicked.connect(self.start_generation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        self.view_results_btn = QPushButton("View results")
        self.view_results_btn.clicked.connect(self.view_results)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        controls.addWidget(self.view_results_btn)
        root.addLayout(controls)

    def _build_generation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        basic = QGroupBox("Main generation")
        grid = QGridLayout(basic)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.pipeline_combo = WheelGuardComboBox()
        self.pipeline_combo.addItems(["one_stage", "two_stages", "two_stages_hq"])
        self.vram_lab_combo = WheelGuardComboBox()
        self.vram_lab_combo.addItems(["ON", "OFF"])
        self.vram_profile_combo = WheelGuardComboBox()
        self.vram_profile_combo.addItems(["auto", "24", "16", "12"])
        self.resolution_combo = WheelGuardComboBox()
        self.resolution_combo.addItems(RESOLUTION_PRESETS)
        self.resolution_combo.setEditable(True)
        self.width_spin = self._spin(64, 4096, 64, 1280)
        self.height_spin = self._spin(64, 4096, 64, 704)
        self.frames_spin = self._spin(1, LTX_EXTENDED_FRAME_LIMIT, 1, 121)
        self.fps_spin = self._spin(1, 120, 1, 24)
        self.steps_spin = self._spin(1, 200, 1, 8)
        self.seed_spin = self._spin(-1, 2_147_483_647, 1, 12345)
        self.random_seed_check = QCheckBox("Random seed on run")
        self.snap_resolution_check = QCheckBox("Snap width/height to multiples of 64")
        self.snap_resolution_check.setChecked(True)
        self._set_tooltip(
            self.vram_lab_combo,
            "Simple VRAM Lab toggle. ON uses the safe VRAM Lab mode. OFF disables VRAM Lab and runs without that protection layer.",
        )
        self.extended_frames_warning_label = QLabel(
            "Lower blocksize in VRAM Lab below default settings to avoid OOM."
        )
        self.extended_frames_warning_label.setWordWrap(True)
        self.extended_frames_warning_label.setVisible(False)
        self.auto_vram_advice_label = QLabel(
            "Tip: enable Auto in the VRAM Lab settings to calculate blocksize automatically."
        )
        self.auto_vram_advice_label.setWordWrap(True)
        self.auto_vram_advice_label.setVisible(False)

        rows = [
            ("Pipeline", self.pipeline_combo, "VRAM Lab", self.vram_lab_combo),
            ("Resolution preset", self.resolution_combo, "Width", self.width_spin),
            ("Height", self.height_spin, "Frames", self.frames_spin),
            ("FPS", self.fps_spin, "Steps", self.steps_spin),
            ("Seed", self.seed_spin, "", QWidget()),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            grid.addWidget(QLabel(l1), r, 0)
            grid.addWidget(w1, r, 1)
            grid.addWidget(QLabel(l2), r, 2)
            grid.addWidget(w2, r, 3)
        grid.addWidget(self.extended_frames_warning_label, 4, 2, 1, 2)
        grid.addWidget(self.auto_vram_advice_label, 5, 2, 1, 2)
        grid.addWidget(self.random_seed_check, len(rows) + 1, 1)
        grid.addWidget(self.snap_resolution_check, len(rows) + 1, 3)
        layout.addWidget(basic)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setPlaceholderText("Describe the video to generate...")
        self.prompt_edit.setMinimumHeight(130)
        self.negative_edit = QTextEdit()
        self.negative_edit.setAcceptRichText(False)
        self.negative_edit.setPlaceholderText("Optional negative prompt. Leave empty to use the LTX default.")
        self.negative_edit.setMinimumHeight(70)

        prompt_group = QGroupBox("Prompt")
        prompt_layout = QFormLayout(prompt_group)
        prompt_layout.addRow("Prompt", self.prompt_edit)
        prompt_layout.addRow("Negative", self.negative_edit)
        layout.addWidget(prompt_group)

        self.latent_preview_section = CollapsibleSection("Latent Preview", open_by_default=False)
        latent_preview_box = QWidget()
        latent_preview_form = QFormLayout(latent_preview_box)
        latent_preview_form.setContentsMargins(0, 0, 0, 0)

        self.latent_preview_enabled_check = QCheckBox("Enable latent preview")
        self.latent_preview_enabled_check.setChecked(False)
        self.latent_preview_mode_combo = WheelGuardComboBox()
        self.latent_preview_mode_combo.addItems(["Fast Latent RGB", "TAE Preview"])
        self.latent_preview_rate_spin = self._spin(1, 30, 1, 8)
        self.latent_preview_upscale_check = QCheckBox("Use latent upscale preview")
        self.latent_preview_tae_decode_check = QCheckBox("Use TAE-style VAE preview when available")
        self.keep_latents_check = QCheckBox("Keep latents")
        self.keep_latents_check.setChecked(False)
        self.latent_preview_hint = QLabel(
            "Experimental. Shows rough sampling previews from latents while LTX is running. "
            "Fast Latent RGB is the lightweight option; TAE preview is higher quality but heavier."
        )
        self.latent_preview_hint.setWordWrap(True)

        self.latent_preview_strip_scroll = QScrollArea()
        self.latent_preview_strip_scroll.setWidgetResizable(True)
        self.latent_preview_strip_scroll.setMinimumHeight(104)
        self.latent_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.latent_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.latent_preview_strip_scroll.setFrameShape(QFrame.StyledPanel)
        self.latent_preview_strip_host = QWidget()
        self.latent_preview_strip_layout = QHBoxLayout(self.latent_preview_strip_host)
        self.latent_preview_strip_layout.setContentsMargins(8, 8, 8, 8)
        self.latent_preview_strip_layout.setSpacing(8)
        self.latent_preview_strip_layout.addStretch(1)
        self.latent_preview_strip_scroll.setWidget(self.latent_preview_strip_host)
        self.latent_preview_status_label = QLabel("Latent preview is off.")
        self.latent_preview_status_label.setWordWrap(True)
        self._latent_preview_cards = []
        self._latent_preview_sidecar_path = ""
        self._latent_preview_dir_path = ""
        self._latent_preview_last_dir = ""
        self._latent_preview_sidecar_offset = 0
        self._latent_preview_poll_timer = QTimer(self)
        self._latent_preview_poll_timer.setInterval(750)
        self._latent_preview_poll_timer.timeout.connect(self._poll_latent_preview_sidecar)

        self._set_tooltip(self.latent_preview_enabled_check, "Turn latent previews on or off for LTX sampling. Default is off so existing runs are unchanged.")
        self._set_tooltip(self.latent_preview_mode_combo, "Fast Latent RGB uses LTX RGB factors for lightweight previews. TAE Preview is for a TAE-style preview path when available.")
        self._set_tooltip(self.latent_preview_rate_spin, "Maximum preview refresh rate in frames per second. Lower values reduce UI and CPU/GPU overhead.")
        self._set_tooltip(self.latent_preview_upscale_check, "Optional experimental latent-upscale preview. Leave off unless testing preview quality.")
        self._set_tooltip(self.latent_preview_tae_decode_check, "Use a TAE-style VAE preview path when the runtime has a compatible preview VAE available.")
        self._set_tooltip(self.keep_latents_check, "When off, the latent preview image folder from the previous run is deleted automatically when a new run starts. Default is off.")

        latent_preview_form.addRow(self.latent_preview_enabled_check)
        latent_preview_form.addRow("Preview type", self.latent_preview_mode_combo)
        latent_preview_form.addRow("Preview FPS", self.latent_preview_rate_spin)
        latent_preview_form.addRow(self.latent_preview_upscale_check)
        latent_preview_form.addRow(self.latent_preview_tae_decode_check)
        latent_preview_form.addRow(self.keep_latents_check)
        latent_preview_form.addRow(self.latent_preview_hint)
        latent_preview_form.addRow("Preview strip", self.latent_preview_strip_scroll)
        latent_preview_form.addRow("Status", self.latent_preview_status_label)
        self.latent_preview_section.addWidget(latent_preview_box)
        self._reset_latent_preview_strip("Latent preview is off.")
        layout.addWidget(self.latent_preview_section)

        self.tabs.addTab(tab, "Generation")

        self.resolution_combo.currentTextChanged.connect(self._resolution_text_changed)
        self.width_spin.valueChanged.connect(self._manual_resolution_changed)
        self.height_spin.valueChanged.connect(self._manual_resolution_changed)
        self.frames_spin.valueChanged.connect(self._frame_count_changed)
        self._update_frame_spin_limit(clamp=True)
        self._update_extended_frames_warning()


    def _align_path_row_labels(self, rows: Iterable[PathRow]) -> None:
        """Make compound PathRow labels end at the same X position without forcing button widths."""
        try:
            valid_rows = [row for row in rows if row is not None and getattr(row, "label", None) is not None]
            if not valid_rows:
                return
            width = max(row.label.sizeHint().width() for row in valid_rows) + 12
            for row in valid_rows:
                try:
                    row.label.setFixedWidth(width)
                    row.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                except Exception:
                    pass
        except Exception:
            pass

    def _build_inputs_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        media_group = QGroupBox("Media")
        form = QFormLayout(media_group)
        self.start_media_row = MediaPathRow(
            "Start image",
            mode="file",
            file_filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
            open_action="clear",
        )
        self.end_media_row = MediaPathRow(
            "End image",
            mode="file",
            file_filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
            open_action="clear",
        )
        self.start_video_row = MediaPathRow(
            "Continue video",
            mode="file",
            file_filter=VIDEO_FILE_FILTER,
            open_action="clear",
        )
        self.end_video_row = MediaPathRow(
            "End with video",
            mode="file",
            file_filter=VIDEO_FILE_FILTER,
            open_action="clear",
        )
        self.glue_input_videos_check = QCheckBox("Glue videos after generation")
        self.glue_input_videos_check.setChecked(False)
        self.reference_images_edit = QLineEdit()
        self.reference_images_edit.setPlaceholderText("Optional: extra conditioning images separated by ;")
        self.reference_browse_btn = QPushButton("Add images")
        self.reference_browse_btn.clicked.connect(self._browse_reference_images)
        self._set_tooltip(
            self.start_media_row,
            "Optional start image for image-to-video. Default frame = 0 and strength = 1.0. "
            "Normal use: pick an image and leave the advanced image details alone.",
        )
        self._set_tooltip(
            self.end_media_row,
            "Optional end image target. The UI places it at the final generated frame. "
            "Leave empty for normal start-image or text-to-video runs.",
        )
        self._set_tooltip(
            self.start_video_row,
            "Optional start video. On run, FrameVision extracts the first frame as a lossless PNG and uses it as the start image.",
        )
        self._set_tooltip(
            self.end_video_row,
            "Optional end video. On run, FrameVision extracts the last frame as a lossless PNG and uses it as the end image.",
        )
        self._set_tooltip(
            self.glue_input_videos_check,
            "When enabled, FrameVision combines the Continue video, the new LTX result, and the End with video in that order after generation. "
            "If only one input video is set, it glues that video with the new result.",
        )
        self._set_tooltip(
            self.reference_images_edit,
            "Optional extra conditioning images separated by semicolons. Use only when additional references are needed; "
            "default strength is 0.8.",
        )
        self._set_tooltip(self.reference_browse_btn, "Add one or more extra reference images to the semicolon-separated list.")
        refs_row = QWidget()
        refs_layout = QHBoxLayout(refs_row)
        refs_layout.setContentsMargins(0, 0, 0, 0)
        refs_layout.addWidget(self.reference_images_edit, 1)
        refs_layout.addWidget(self.reference_browse_btn)

        self.normalize_input_image_check = QCheckBox("Normalize input images before LTX")
        self.normalize_input_image_check.setChecked(True)
        self._set_tooltip(
            self.normalize_input_image_check,
            "Converts input images to clean RGB PNG copies before LTX to avoid native crashes from unusual image modes/metadata.",
        )

        try:
            self.start_media_row.use_current_btn.clicked.connect(lambda: self._use_current_for_image_row(self.start_media_row, "Start image"))
            self.end_media_row.use_current_btn.clicked.connect(lambda: self._use_current_for_image_row(self.end_media_row, "end image"))
            self.start_video_row.use_current_btn.clicked.connect(lambda: self._use_current_for_video_row(self.start_video_row, "Continue video"))
            self.end_video_row.use_current_btn.clicked.connect(lambda: self._use_current_for_video_row(self.end_video_row, "End with video"))
        except Exception:
            pass

        self._align_path_row_labels([self.start_media_row, self.end_media_row, self.start_video_row, self.end_video_row])

        form.addRow(self.start_media_row)
        form.addRow(self.end_media_row)
        form.addRow(self.start_video_row)
        form.addRow(self.end_video_row)
        form.addRow(self.glue_input_videos_check)
        form.addRow("Extra images", refs_row)
        form.addRow(self.normalize_input_image_check)
        layout.addWidget(media_group)

        details = CollapsibleSection("Image conditioning details", open_by_default=False)
        details_host = QWidget()
        details_form = QFormLayout(details_host)
        self.start_image_frame_spin = self._spin(0, 3000, 1, 0)
        self.start_image_strength_spin = self._double_spin(0.0, 2.0, 0.05, 1.0)
        self.end_image_strength_spin = self._double_spin(0.0, 2.0, 0.05, 1.0)
        self.reference_image_strength_spin = self._double_spin(0.0, 2.0, 0.05, 0.8)
        self._set_tooltip(
            self.start_image_frame_spin,
            "Frame index where the start image is applied. Default 0 = first frame. Usually leave this at 0.",
        )
        self._set_tooltip(
            self.start_image_strength_spin,
            "How strongly the start image guides the first frame. Default 1.0. Lower values loosen the image influence.",
        )
        self._set_tooltip(
            self.end_image_strength_spin,
            "How strongly the optional end image guides the final frame. Default 1.0. Only matters when an end image is selected.",
        )
        self._set_tooltip(
            self.reference_image_strength_spin,
            "Strength for optional extra reference images. Default 0.8. Use lower values if the reference pulls the result too hard.",
        )
        details_form.addRow("Start image frame", self.start_image_frame_spin)
        details_form.addRow("Start image strength", self.start_image_strength_spin)
        details_form.addRow("End image strength", self.end_image_strength_spin)
        details_form.addRow("Extra image strength", self.reference_image_strength_spin)
        details.addWidget(details_host)
        layout.addWidget(details)

        lora_group = QGroupBox("LoRAs")
        lora_form = QFormLayout(lora_group)
        self.user_lora_rows: List[PathRow] = []
        self.user_lora_strength_spins: List[WheelGuardDoubleSpinBox] = []
        self.user_lora_strength_sliders: List[WheelGuardSlider] = []
        for index in range(4):
            row = PathRow(f"LoRA {index + 1}", mode="file", file_filter="Safetensors (*.safetensors);;All files (*.*)", open_action="clear")
            spin = self._double_spin(0.0, 2.0, 0.05, 1.0)
            spin.setDecimals(2)
            slider = WheelGuardSlider(Qt.Horizontal)
            slider.setRange(0, 200)
            slider.setSingleStep(5)
            slider.setPageStep(10)
            slider.setValue(100)
            slider.setToolTip("LoRA strength. 1.0 is normal strength; 0 disables that LoRA.")
            spin.setToolTip(slider.toolTip())
            spin.valueChanged.connect(lambda value, s=slider: self._lora_strength_spin_changed(value, s))
            slider.valueChanged.connect(lambda value, sp=spin: self._lora_strength_slider_changed(value, sp))
            strength_layout = QHBoxLayout()
            strength_layout.setContentsMargins(0, 0, 0, 0)
            strength_layout.setSpacing(6)
            strength_layout.addWidget(spin, 0)
            strength_layout.addWidget(slider, 1)
            strength_host = QWidget()
            strength_host.setLayout(strength_layout)
            lora_form.addRow(row)
            lora_form.addRow("Strength", strength_host)
            self.user_lora_rows.append(row)
            self.user_lora_strength_spins.append(spin)
            self.user_lora_strength_sliders.append(slider)
        layout.addWidget(lora_group)

        layout.addStretch(1)
        self.tabs.addTab(tab, "Inputs")


    def _build_quality_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        guidance_group = QGroupBox("Video guidance")
        guidance_form = QFormLayout(guidance_group)
        self.video_cfg_spin = self._double_spin(0.0, 20.0, 0.1, 2.0)
        self.shift_slider = WheelGuardSlider(Qt.Horizontal)
        self.shift_slider.setRange(0, 1000)
        self.shift_slider.setSingleStep(5)
        self.shift_slider.setPageStep(50)
        self.shift_slider.setValue(500)
        self.shift_spin = self._double_spin(0.0, 10.0, 0.05, 5.0)
        self.shift_spin.setDecimals(2)
        self.shift_slider.valueChanged.connect(self._shift_slider_changed)
        self.shift_spin.valueChanged.connect(self._shift_spin_changed)
        self.video_stg_spin = self._double_spin(0.0, 20.0, 0.1, 0.0)
        self.video_rescale_spin = self._double_spin(0.0, 2.0, 0.05, 0.7)
        self.video_skip_step_spin = self._spin(0, 50, 1, 0)
        self.max_batch_size_combo = WheelGuardComboBox()
        self.max_batch_size_combo.addItems(["1", "2", "4"])
        self.max_batch_size_combo.setCurrentText("2")
        self.enhance_prompt_check = QCheckBox("Enhance prompt")
        self._set_tooltip(
            self.video_cfg_spin,
            "Video CFG guidance controls how strongly the prompt pulls the video. Default 1.0 for the distilled workflow. "
            "Higher values can follow the prompt harder but may look less natural and can slow the run.",
        )
        self._set_tooltip(
            self.shift_slider,
            "Scheduler/sigma shift override. Range 0.00 to 10.00. Default 5.00. "
            "This changes the sigma schedule used during denoising; adjust only when testing stability, structure, or motion differences.",
        )
        self._set_tooltip(self.shift_spin, self.shift_slider.toolTip())
        self._set_tooltip(
            self.video_stg_spin,
            "Video STG guidance. Default 0.0/off. Raise only when intentionally testing STG behavior.",
        )
        self._set_tooltip(
            self.video_rescale_spin,
            "Guidance rescale for video. Default 0.7. Helps keep guidance from becoming too harsh or overcooked.",
        )
        self._set_tooltip(
            self.video_skip_step_spin,
            "Skip early video guidance steps. Default 0. Leave at 0 unless testing official LTX skip-step behavior.",
        )
        self._set_tooltip(
            self.max_batch_size_combo,
            "Maximum internal batch size used by the LTX pipeline. Default 2. Lower to 1 if VRAM/shared-memory pressure appears; "
            "higher may be faster only when memory is safe.",
        )
        self._set_tooltip(
            self.enhance_prompt_check,
            "Ask LTX to enhance/expand the prompt before generation. Default off. Useful for short prompts, but it can change the prompt wording.",
        )
        guidance_form.addRow("CFG guidance", self.video_cfg_spin)
        shift_row = QWidget()
        shift_layout = QHBoxLayout(shift_row)
        shift_layout.setContentsMargins(0, 0, 0, 0)
        shift_layout.addWidget(self.shift_slider, 1)
        shift_layout.addWidget(self.shift_spin, 0)
        guidance_form.addRow("Shift", shift_row)
        guidance_form.addRow("STG guidance", self.video_stg_spin)
        guidance_form.addRow("Rescale", self.video_rescale_spin)
        guidance_form.addRow("Video skip step", self.video_skip_step_spin)
        self.max_batch_size_label = QLabel("Max batch size")
        self.max_batch_size_label.setToolTip(self.max_batch_size_combo.toolTip())
        guidance_form.addRow(self.max_batch_size_label, self.max_batch_size_combo)
        guidance_form.addRow(self.enhance_prompt_check)
        layout.addWidget(guidance_group)

        audio_guidance = CollapsibleSection("Audio guidance from LTX", open_by_default=False)
        audio_host = QWidget()
        audio_form = QFormLayout(audio_host)
        self.audio_cfg_spin = self._double_spin(0.0, 20.0, 0.1, 1.0)
        self.audio_stg_spin = self._double_spin(0.0, 20.0, 0.1, 0.0)
        self.audio_rescale_spin = self._double_spin(0.0, 2.0, 0.05, 0.0)
        self.a2v_guidance_spin = self._double_spin(0.0, 20.0, 0.1, 1.0)
        self.v2a_guidance_spin = self._double_spin(0.0, 20.0, 0.1, 1.0)
        self.audio_skip_step_spin = self._spin(0, 50, 1, 0)
        self._set_tooltip(self.audio_cfg_spin, "Audio CFG guidance for official LTX audio/video modes. Default 1.0. Usually leave this alone.")
        self._set_tooltip(self.audio_stg_spin, "Audio STG guidance. Default 0.0/off. Only change when testing audio STG behavior.")
        self._set_tooltip(self.audio_rescale_spin, "Audio guidance rescale. Default 0.0. Usually leave this alone.")
        self._set_tooltip(self.a2v_guidance_spin, "Audio-to-video influence. Default 1.0. In Prompt + audio mode, higher values may push motion closer to the audio.")
        self._set_tooltip(self.v2a_guidance_spin, "Video-to-audio influence. Default 1.0. Usually leave this at default unless testing baked audio/video behavior.")
        self._set_tooltip(self.audio_skip_step_spin, "Skip early audio guidance steps. Default 0. Leave at 0 unless testing official LTX skip-step behavior.")
        audio_form.addRow("Audio CFG", self.audio_cfg_spin)
        audio_form.addRow("Audio STG", self.audio_stg_spin)
        audio_form.addRow("Audio rescale", self.audio_rescale_spin)
        audio_form.addRow("Audio → Video guidance", self.a2v_guidance_spin)
        audio_form.addRow("Video → Audio guidance", self.v2a_guidance_spin)
        audio_form.addRow("Audio skip step", self.audio_skip_step_spin)
        audio_guidance.addWidget(audio_host)
        layout.addWidget(audio_guidance)

        two_stage_group = QGroupBox("two-phase / HQ requirements")
        form = QFormLayout(two_stage_group)
        self.distilled_lora_row = PathRow("Distilled LoRA", mode="file", file_filter="Safetensors (*.safetensors);;All files (*.*)")
        self.distilled_lora_row.setVisible(False)
        self.distilled_lora_strength_spin = self._double_spin(0.0, 10.0, 0.05, 0.5)
        self.distilled_lora_strength_spin.setVisible(False)
        self.spatial_upsampler_row = PathRow("Spatial upsampler", mode="file", file_filter="Safetensors (*.safetensors);;All files (*.*)")
        self._set_tooltip(
            self.spatial_upsampler_row,
            "Required upsampler for two-phase / HQ output. Auto-download stores it in models/ltx23/spatial_upsampler.",
        )
        self.auto_download_two_stage_check = QCheckBox("Auto-download missing two-phase files on Run")
        self.auto_download_two_stage_check.setChecked(True)
        self.auto_download_two_stage_check.setToolTip("Downloads/checks only the missing spatial upsampler.")
        self.two_stage_status_label = QLabel("two-phase asset status: not checked yet")
        self.two_stage_status_label.setWordWrap(True)
        self.two_stage_progress = QProgressBar()
        self.two_stage_progress.setRange(0, 100)
        self.two_stage_progress.setValue(0)
        self.two_stage_progress.setVisible(False)
        self.two_stage_download_btn = QPushButton("Find / download missing two-phase files")
        self.two_stage_download_btn.setToolTip(
            "Checks the saved paths, searches the LTX model folders, then downloads only missing two-phase assets. "
            "A progress bar is shown while downloading."
        )
        self.two_stage_status_label.setToolTip("Shows the spatial upsampler status.")
        self.two_stage_progress.setToolTip("Download progress for missing two-phase files.")
        self.two_stage_download_btn.clicked.connect(lambda: self._ensure_two_stage_assets(start_after_download=False, manual=True))
        form.addRow(self.spatial_upsampler_row)
        form.addRow(self.auto_download_two_stage_check)
        form.addRow("Status", self.two_stage_status_label)
        form.addRow(self.two_stage_progress)
        form.addRow(self.two_stage_download_btn)
        layout.addWidget(two_stage_group)

        layout.addStretch(1)
        self.tabs.addTab(tab, "Quality")


    def _build_audio_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        audio_group = QGroupBox("Audio / soundtrack")
        form = QFormLayout(audio_group)
        self.audio_row = PathRow("Audio file", mode="file", file_filter="Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All files (*.*)")
        self.audio_mode_combo = WheelGuardComboBox()
        self.audio_mode_combo.addItems(AUDIO_MODE_CHOICES)
        self._set_tooltip(
            self.audio_row,
            "Optional audio file. In Prompt only mode it is added after generation. In Prompt + audio mode, LTX uses it to create synced motion/audio.",
        )
        self._set_tooltip(
            self.audio_mode_combo,
            "Disabled = no audio. Prompt only = generate video first, then add soundtrack. "
            "Prompt + audio file = use LTX audio-conditioned generation around the sound file.",
        )

        self.audio_start_spin = self._double_spin(0.0, 36000.0, 0.1, 0.0)
        self.audio_start_spin.setSuffix(" s")
        self.audio_start_spin.setToolTip("For audio-conditioned LTX: start reading the audio file at this time.")
        self.audio_max_duration_spin = self._double_spin(0.0, 36000.0, 0.1, 0.0)
        self.audio_max_duration_spin.setSuffix(" s")
        self.audio_max_duration_spin.setSpecialValueText("Auto")
        self.audio_max_duration_spin.setToolTip("For audio-conditioned LTX: 0 = use video duration from frames / FPS.")

        self.remux_bitrate_combo = WheelGuardComboBox()
        self.remux_bitrate_combo.addItems(["128k", "160k", "192k", "256k", "320k"])
        self._set_tooltip(self.remux_bitrate_combo, "AAC bitrate used only for Prompt only / add-audio-after-generation mode. Default 192k.")
        self.remux_shortest_check = QCheckBox("End when the shortest stream ends")
        self.remux_shortest_check.setChecked(True)
        self._set_tooltip(
            self.remux_shortest_check,
            "Default on. FFmpeg stops at the shorter of the generated video or the audio file, avoiding black/silent tails.",
        )
        self.remux_replace_check = QCheckBox("Replace original output with audio version")
        self._set_tooltip(
            self.remux_replace_check,
            "Default off. When enabled, the remuxed audio version replaces the silent/original output file.",
        )
        form.addRow(self.audio_row)
        form.addRow("Mode", self.audio_mode_combo)
        form.addRow("Audio start", self.audio_start_spin)
        form.addRow("Audio max duration", self.audio_max_duration_spin)
        form.addRow("AAC bitrate", self.remux_bitrate_combo)
        form.addRow(self.remux_shortest_check)
        form.addRow(self.remux_replace_check)
        layout.addWidget(audio_group)

        self.audio_hint_label = QLabel()
        self.audio_hint_label.setWordWrap(True)
        layout.addWidget(self.audio_hint_label)
        layout.addStretch(1)
        self.audio_mode_combo.currentTextChanged.connect(self._audio_mode_changed)
        self._audio_mode_changed(self.audio_mode_combo.currentText())
        self.tabs.addTab(tab, "Audio")


    def _audio_mode_changed(self, mode: str) -> None:
        mode = AUDIO_MODE_COMPAT_MAP.get(str(mode or "").strip(), AUDIO_MODE_DISABLED)
        is_a2v = mode == AUDIO_MODE_A2V
        is_remux = mode == AUDIO_MODE_REMUX

        for widget in (self.audio_start_spin, self.audio_max_duration_spin):
            widget.setEnabled(is_a2v)
        for widget in (self.remux_bitrate_combo, self.remux_shortest_check, self.remux_replace_check):
            widget.setEnabled(is_remux)

        if hasattr(self, "audio_hint_label"):
            if is_a2v:
                is_int4 = False
                try:
                    is_int4 = self._selected_model_variant() == "INT4"
                except Exception:
                    pass
                if is_int4:
                    self.audio_hint_label.setText(
                        "INT4 uses the normal two-stage workflow. The selected audio is encoded once, kept frozen "
                        "through both stages to drive the video, and the original waveform is baked into the output. "
                        "This does not use two_stages_hq or the native VRAM Lab CLI."
                    )
                else:
                    self.audio_hint_label.setText(
                        "Prompt + audio file passes the audio into official LTX a2vid_two_stage using --audio-path. "
                        "LTX should build the video around that sound and output baked audio/video. "
                        "This mode needs the distilled LoRA and spatial upsampler paths in the Quality tab."
                    )
            elif is_remux:
                self.audio_hint_label.setText(
                    "Prompt only ignores the audio while LTX generates the picture, then FFmpeg copies the video stream "
                    "and adds the selected audio afterward. This is only a soundtrack/remux mode, not lipsync/dance-sync."
                )
            else:
                self.audio_hint_label.setText("Own audio is disabled.")

        if hasattr(self, "pipeline_combo"):
            self._update_pipeline_dependent_controls()

        if not self._loading and hasattr(self, "command_preview") and hasattr(self, "python_row"):
            self._refresh_command_preview()
            self._queue_settings_save()


    def _effective_pipeline(self) -> str:
        if AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED) == AUDIO_MODE_A2V:
            return "a2vid_two_stage"
        return self.pipeline_combo.currentText()


    def _build_advanced_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        runtime_group = QGroupBox("Runtime / diagnostics")
        form = QFormLayout(runtime_group)
        self.no_boundary_echo_check = QCheckBox("No boundary echo")
        self.deep_lifecycle_check = QCheckBox("Deep lifecycle log")
        self.deep_interval_spin = self._double_spin(0.1, 30.0, 0.1, 1.0)
        self.deep_max_events_spin = self._spin(100, 100000, 100, 4000)
        self._set_tooltip(self.no_boundary_echo_check, "Reduces noisy VRAM boundary echo. Brief [ltx-status] phase labels still show so long loads do not look stuck.")
        self._set_tooltip(self.deep_lifecycle_check, "Writes detailed lifecycle/VRAM timing logs. Default off. Turn on when debugging stalls or memory behavior.")
        self._set_tooltip(self.deep_interval_spin, "Seconds between deep lifecycle samples. Default 1.0. Lower = more detail but more log spam.")
        self._set_tooltip(self.deep_max_events_spin, "Maximum deep lifecycle log events to keep. Default 4000.")
        self.use_framevision_queue_check = QCheckBox("Use FrameVision queue")
        self.use_framevision_queue_check.setChecked(True)
        self.use_framevision_queue_check.setToolTip(
            "When enabled, Run LTX adds the job to the normal FrameVision Queue tab instead of launching it directly from this panel. "
            "Turn this off for quick command-window style tests or when debugging the LTX helper itself."
        )
        form.addRow(self.use_framevision_queue_check)
        form.addRow(self.no_boundary_echo_check)
        layout.addWidget(runtime_group)

        attention_group = QGroupBox("Attention backend")
        attention_form = QFormLayout(attention_group)
        self.attention_backend_combo = WheelGuardComboBox()
        self.attention_backend_combo.addItems(["auto", "sdpa", "flash2", "sage"])
        self.attention_backend_combo.setToolTip(
            "Select the LTX attention backend. auto tries SageAttention first, then FlashAttention2, then PyTorch SDPA. "
            "If an optional backend is missing or errors per call, the wrapper falls back safely instead of crashing."
        )
        self.flash_attention_check = QCheckBox("Enable FlashAttention")
        self.flash_attention_check.setVisible(False)
        self.flash_attention_check.setChecked(False)
        self.flash_attention_check.setToolTip("Legacy hidden option. Use the Attention backend selector instead.")
        self.flash_attention_status_label = QLabel("Checking attention backend availability...")
        self.flash_attention_status_label.setWordWrap(True)
        self.flash_attention_status_label.setToolTip(
            "SageAttention and FlashAttention are checked in the selected Python environment. The final LTX report proves what backend was actually used."
        )
        attention_form.addRow("Backend", self.attention_backend_combo)
        attention_form.addRow(self.flash_attention_status_label)
        layout.addWidget(attention_group)

        self.vram_group = QGroupBox("VRAM Lab settings")
        vram_group = self.vram_group
        vram_form = QFormLayout(vram_group)
        self.main_hot_window_spin = self._double_spin(0.0, 32.0, 0.1, LTX_MAIN_HOT_WINDOW_DEFAULTS_GB["24"])
        self.main_hot_window_spin.setDecimals(1)
        self.main_hot_window_spin.setSuffix(" GB")
        self.main_hot_window_spin.setSpecialValueText("Profile default")
        profile_defaults_tip = (
            "Default block limits by VRAM profile: "
            "12 GB = Stage 1 4.4 GB / Stage 2 2.0 GB; "
            "16 GB = Stage 1 8.0 GB / Stage 2 4.0 GB; "
            "24 GB = Stage 1 16.5 GB / Stage 2 12.5 GB. Auto uses 23+ GB as the 24 GB profile."
        )
        tip = (
            "Advanced VRAM Lab control for the main/distilled LTX safetensor block size. "
            "First selection uses the selected profile default. Each profile remembers its own edited value. Reduce this number when creating longer videos "
            "to avoid spilling into shared GPU memory. Gemma/text encoder settings are not changed. "
            + profile_defaults_tip
        )
        self.main_hot_window_spin.setToolTip(tip)
        self.vram_profile_combo.setToolTip(
            "Select 12/16/24 GB to test that VRAM profile with resolution/frame-aware hotset values, or auto to detect your GPU profile automatically. "
            + profile_defaults_tip
        )
        self.vram_profile_hint = QLabel()
        self.vram_profile_hint.setWordWrap(True)
        vram_form.addRow("VRAM profile", self.vram_profile_combo)
        vram_form.addRow("Block size limit", self.main_hot_window_spin)
        self.stage2_block_size_limit_spin = self._double_spin(0.0, 32.0, 0.1, LTX_STAGE2_HOT_WINDOW_DEFAULTS_GB["24"])
        self.stage2_block_size_limit_spin.setDecimals(1)
        self.stage2_block_size_limit_spin.setSuffix(" GB")
        self.stage2_block_size_limit_spin.setToolTip(
            "Stage 2/refine block size limit. Defaults by profile: 12 GB = 2.0 GB, 16 GB = 4.0 GB, 24 GB = 12.5 GB."
        )
        self.stage2_block_size_limit_help = QLabel(
            "Profile defaults for 241 frames at 704p : 12 GB = 4.4 / 2.0 GB, 16 GB = 8.0 / 4.0 GB, 24 GB = 16.5 / 12.5 GB (Stage 1 / Stage 2)."
        )
        self.stage2_block_size_limit_help.setWordWrap(True)
        self.stage2_block_size_limit_help.setToolTip(self.stage2_block_size_limit_spin.toolTip())
        vram_form.addRow("Stage 2 block size limit", self.stage2_block_size_limit_spin)
        vram_form.addRow(self.stage2_block_size_limit_help)
        self.emergency_free_vram_floor_spin = self._double_spin(0.25, 3.0, 0.25, 0.5)
        self.emergency_free_vram_floor_spin.setDecimals(2)
        self.emergency_free_vram_floor_spin.setSuffix(" GB")
        self.emergency_free_vram_floor_spin.setToolTip(
            "Emergency free VRAM floor for VRAM Lab trimming. Default is 0.50 GB for all profiles; lower values let the 24 GB profile run closer to full VRAM before correction. "
            "Range: 0.25 to 3.00 GB. Default: 0.50 GB."
        )
        vram_form.addRow("Emergency free VRAM floor", self.emergency_free_vram_floor_spin)
        vram_form.addRow(self.vram_profile_hint)
        self.main_transformer_stream_probe_check = QCheckBox("Main transformer streaming probe")
        self.main_transformer_stream_probe_check.setChecked(False)
        self.main_transformer_stream_probe_check.setVisible(False)
        self.main_transformer_stream_probe_check.setToolTip(
            "Hidden for now. Experimental probe path added time in real tests; leave OFF unless deliberately restoring that experiment."
        )
        # Hidden on purpose: this old probe branch is currently slower/noisier than useful.
        # Keep the setting forced OFF, but do not show the control in the normal test UI.
        # vram_form.addRow(self.main_transformer_stream_probe_check)
        # Moved to the top of the VRAM Lab tab.
        self.vram_profile_combo.currentTextChanged.connect(self._vram_profile_changed)
        self.main_hot_window_spin.valueChanged.connect(self._block_size_limit_changed)
        self.stage2_block_size_limit_spin.valueChanged.connect(self._update_vram_profile_hint)
        self._update_vram_profile_hint()


        # Hidden compatibility field: command/settings code still reads custom_extra_args,
        # but the unused Raw extra arguments UI has been removed from Settings.
        self.extra_args_edit = QTextEdit(tab)
        self.extra_args_edit.setAcceptRichText(False)
        self.extra_args_edit.setVisible(False)

        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setAcceptRichText(False)
        self.command_preview.setMinimumHeight(120)
        self.command_preview.setToolTip("Shows the exact command the UI will run. Useful for copying into a command prompt when debugging.")
        layout.addWidget(QLabel("Command preview"))
        layout.addWidget(self.command_preview)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Settings")


    def _build_test_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        if hasattr(self, "vram_group"):
            layout.addWidget(self.vram_group)

        vram_test_box = QGroupBox("VRAM Lab residency")
        vram_test_form = QFormLayout(vram_test_box)

        self.planned_hotset_check = QCheckBox("Use planned hotset")
        self.planned_hotset_check.setChecked(True)
        self.planned_hotset_check.setToolTip(
            "When enabled, VRAM Lab uses the generic planned_hotset residency strategy. "
            "When disabled, it uses the old rolling strategy."
        )

        self.fast_iclora_route_check = QCheckBox("Faster text to video steps for two-phase runs")
        self.fast_iclora_route_check.setChecked(True)
        self.fast_iclora_route_check.setToolTip(self._fast_iclora_base_tooltip)

        self.stage1_stable_hotset_fraction_spin = self._double_spin(0.10, 2.00, 0.01, 1.15)
        self.stage1_stable_hotset_fraction_spin.setDecimals(2)
        self.stage1_stable_hotset_fraction_spin.setToolTip(
            "Stage 1 / first denoise stable-hotset fraction. Higher values keep more blocks hot for repeated forwards. "
            "Testing showed Stage 1 benefits from higher values. Range: 0.10 to 2.00."
        )

        self.stage2_stable_hotset_fraction_spin = self._double_spin(0.70, 1.10, 0.01, 0.9)
        self.stage2_stable_hotset_fraction_spin.setDecimals(2)
        self.stage2_stable_hotset_fraction_spin.setToolTip(
            "Stage 2 / refine denoise stable-hotset fraction. Keep this more conservative than Stage 1. "
            "Range: 0.70 to 1.10."
        )
        # Backward-compatible alias used by older helper code paths.
        self.stable_hotset_fraction_spin = self.stage1_stable_hotset_fraction_spin

        self.stable_hotset_budget_gb_spin = self._double_spin(0.0, 32.0, 0.1, 0.0)
        self.stable_hotset_budget_gb_spin.setDecimals(1)
        self.stable_hotset_budget_gb_spin.setSuffix(" GB")
        self.stable_hotset_budget_gb_spin.setToolTip(
            "Optional fixed stable hotset budget. Leave at 0.0 GB to let VRAM Lab calculate it from the stage hot-window and fraction."
        )

        self.vram_residency_hint = QLabel(
            "Generic VRAM Lab setting. 24 GB default: Stage 1 1.15 / Stage 2 0.9. Stage 2 should stay conservative."
        )
        self.vram_residency_hint.setWordWrap(True)

        vram_test_form.addRow("", self.planned_hotset_check)
        vram_test_form.addRow("", self.fast_iclora_route_check)
        vram_test_form.addRow("Stage 1 hotset fraction", self.stage1_stable_hotset_fraction_spin)
        vram_test_form.addRow("Stage 2 hotset fraction", self.stage2_stable_hotset_fraction_spin)
        vram_test_form.addRow("Stable hotset budget", self.stable_hotset_budget_gb_spin)
        vram_test_form.addRow("", self.vram_residency_hint)
        layout.addWidget(vram_test_box)

        deep_log_box = QGroupBox("Deep lifecycle log")
        deep_log_form = QFormLayout(deep_log_box)
        deep_log_form.addRow(self.deep_lifecycle_check)
        deep_log_form.addRow("Deep log interval", self.deep_interval_spin)
        deep_log_form.addRow("Deep max events", self.deep_max_events_spin)
        layout.addWidget(deep_log_box)

        # Hidden compatibility widgets: old settings/save code still expects these
        # attributes, but the controls are intentionally no longer shown in the Test tab.
        self.disable_distilled_lora_check = QCheckBox()
        self.disable_distilled_lora_check.setChecked(True)
        self.disable_lora_extra_args_edit = QLineEdit()
        self.lora_cache_mode_combo = WheelGuardComboBox()
        self.lora_cache_mode_combo.addItems(LORA_CACHE_MODE_CHOICES)
        self._set_combo(self.lora_cache_mode_combo, LORA_CACHE_MODE_OFF)
        self.lora_cache_miss_inplace_check = QCheckBox()
        self.lora_cache_miss_inplace_check.setChecked(False)
        self.lora_cache_shard_gb_spin = self._double_spin(1.0, 64.0, 1.0, 4.0)
        self.lora_cache_shard_threshold_spin = self._double_spin(1.0, 999999.0, 1.0, 999999.0)
        self.lora_cache_max_spin = self._spin(0, 20, 1, 0)
        self.lora_cache_status_label = QLabel("LoRA cache UI removed")
        self.open_lora_cache_btn = QPushButton()
        self.delete_partial_lora_cache_btn = QPushButton()
        self.refresh_lora_cache_btn = QPushButton()

        layout.addStretch(1)
        self.vram_lab_tab = tab
        self.tabs.addTab(tab, "VRAM Lab")

    def _build_locations_section(self) -> None:
        host = QWidget()
        form = QFormLayout(host)
        self.ltx_root_row = PathRow("LTX root", str(DEFAULT_LTX_ROOT), mode="dir")
        self.python_row = PathRow("Python exe", str(DEFAULT_PYTHON), mode="file", file_filter="Python (python.exe python);;All files (*.*)")
        self.cli_row = PathRow("LTX CLI", str(DEFAULT_CLI_PATH), mode="file", file_filter="Python (*.py);;All files (*.*)")
        self.checkpoint_row = PathRow("Checkpoint", str(DEFAULT_CHECKPOINT), mode="file", file_filter="Safetensors (*.safetensors);;All files (*.*)")
        self.gemma_row = PathRow("Gemma root", str(DEFAULT_GEMMA_ROOT), mode="dir")
        self.output_dir_row = PathRow("Output folder", str(DEFAULT_OUTPUT_DIR), mode="dir")
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Blank = auto name")
        self.output_name_edit.setToolTip("Optional output filename. Leave blank to auto-create a timestamped name from resolution, frame count, seed and prompt.")
        self.report_row = PathRow("Report path", str(DEFAULT_REPORT_PATH), mode="file", file_filter="Text (*.txt);;All files (*.*)", save_file=True)
        self.deep_log_row = PathRow("Deep log path", str(DEFAULT_DEEP_LOG_PATH), mode="file", file_filter="Text (*.txt);;All files (*.*)", save_file=True)
        self.ffmpeg_row = PathRow("FFmpeg", "ffmpeg", mode="file", file_filter="FFmpeg (ffmpeg.exe ffmpeg);;All files (*.*)")
        self._align_path_row_labels([self.ltx_root_row, self.python_row, self.cli_row, self.checkpoint_row, self.gemma_row, self.output_dir_row, self.report_row, self.deep_log_row, self.ffmpeg_row])
        self.model_variant_combo = WheelGuardComboBox()
        self.model_variant_combo.addItems(["FP16", "FP8", "INT4"])
        self.auto_fill_model_paths_btn = QPushButton("Auto fill selected")
        self.auto_fill_model_paths_btn.clicked.connect(self._auto_fill_selected_model_paths)
        model_preset_row = QWidget()
        model_preset_layout = QHBoxLayout(model_preset_row)
        model_preset_layout.setContentsMargins(0, 0, 0, 0)
        model_preset_layout.setSpacing(6)
        model_preset_layout.addWidget(self.model_variant_combo)
        model_preset_layout.addWidget(self.auto_fill_model_paths_btn)
        model_preset_layout.addStretch(1)
        self.quantization_combo = WheelGuardComboBox()
        self.quantization_combo.addItems(QUANTIZATION_MODE_CHOICES)
        self._set_combo(self.quantization_combo, QUANTIZATION_MODE_NONE)
        self._set_tooltip(self.model_variant_combo, "Choose FP16/FP8 on the untouched native VRAM Lab CLI, or INT4 on the isolated planner_ltx_int4.py route.")
        self._set_tooltip(self.auto_fill_model_paths_btn, "Auto-fill LTX root, env, checkpoint, Gemma, output and tool paths for the selected model.")
        self._set_tooltip(self.quantization_combo, "Quantization is off by default. FP8 modes are experimental and must be selected manually.")
        self._set_tooltip(self.ltx_root_row, "Portable LTX root folder. Default is the FrameVision install root; models, outputs and portable settings stay under this root.")
        self._set_tooltip(self.python_row, "Python executable from the LTX environment, usually under the FrameVision install root: environments\\.ltx23\\python.exe.")
        self._set_tooltip(self.cli_row, "FP16/FP8 use helpers\\ltx23_vram_lab_cli.py. INT4 uses helpers\\planner_ltx_int4.py. The UI switches this path automatically.")
        self._set_tooltip(self.checkpoint_row, "Main LTX 2.3 checkpoint / safetensors file. This is the large model file.")
        self._set_tooltip(self.gemma_row, "Gemma text encoder folder. Usually leave this unchanged once it works.")
        self._set_tooltip(self.output_dir_row, "Folder where generated videos are saved. Default is under the LTX root output folder.")
        self._set_tooltip(self.report_row, "VRAM Lab summary report path. Keep this inside the standalone LTX tools/vram_lab folder.")
        self._set_tooltip(self.deep_log_row, "Deep lifecycle log path used when Deep lifecycle log is enabled. Keep this inside tools/vram_lab.")
        self._set_tooltip(self.ffmpeg_row, "FFmpeg executable. Leave as 'ffmpeg' if it is on PATH, or browse to presets/bin/ffmpeg.exe.")
        self.open_when_done_check = QCheckBox("Open output folder when done")
        self._set_tooltip(self.open_when_done_check, "Open the output folder automatically after generation finishes. Default off.")
        form.addRow("Model preset", model_preset_row)
        form.addRow("Quantization", self.quantization_combo)
        self._apply_quantization_availability("FP16")
        for row in [self.ltx_root_row, self.python_row, self.cli_row, self.checkpoint_row, self.gemma_row, self.output_dir_row, self.report_row, self.deep_log_row, self.ffmpeg_row]:
            form.addRow(row)
        form.addRow("Output filename", self.output_name_edit)
        form.addRow(self.open_when_done_check)
        self.locations_section.addWidget(host)

    def _build_log_section(self) -> None:
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setAcceptRichText(False)
        self.log_edit.setMinimumHeight(220)
        self.log_section.addWidget(self.log_edit)
        log_buttons = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear log")
        self.clear_log_btn.clicked.connect(self.log_edit.clear)
        self.save_log_btn = QPushButton("Save log")
        self.save_log_btn.clicked.connect(self.save_log)
        log_buttons.addStretch(1)
        log_buttons.addWidget(self.clear_log_btn)
        log_buttons.addWidget(self.save_log_btn)
        self.log_section.addLayout(log_buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_main_with_video(self):
        """Best-effort search for the main FrameVision window that owns the media player."""
        try:
            p = self.parent()
            while p is not None:
                if hasattr(p, "video"):
                    return p
                try:
                    p = p.parent()
                except Exception:
                    break
        except Exception:
            pass
        try:
            for w in QApplication.topLevelWidgets():
                if hasattr(w, "video"):
                    return w
        except Exception:
            pass
        return None

    def _current_media_path(self) -> Optional[Path]:
        main = self._find_main_with_video()
        if main is None:
            return None
        try:
            cur = getattr(main, "current_path", None)
            if cur:
                p = Path(str(cur)).expanduser()
                if p.exists():
                    return p
        except Exception:
            pass
        try:
            video = getattr(main, "video", None)
        except Exception:
            video = None
        if video is None:
            return None
        for attr in ("current_path", "path", "file_path", "filepath", "source", "filename", "file"):
            try:
                val = getattr(video, attr, None)
            except Exception:
                val = None
            if not val:
                continue
            try:
                if hasattr(val, "toLocalFile"):
                    val = val.toLocalFile()
            except Exception:
                pass
            try:
                p = Path(str(val)).expanduser()
                if p.exists():
                    return p
            except Exception:
                pass
        return None

    def _grab_current_qimage(self) -> Optional[QImage]:
        try:
            main = self._find_main_with_video()
            if main is None:
                return None
            video = getattr(main, "video", None)
            if video is None:
                return None
            img = getattr(video, "currentFrame", None)
            if isinstance(img, QImage) and not img.isNull():
                return img
            try:
                label = getattr(video, "label", None)
                if label is not None and hasattr(label, "pixmap"):
                    pm = label.pixmap()
                    if pm is not None and not pm.isNull():
                        return pm.toImage()
            except Exception:
                pass
            try:
                labels = main.findChildren(QLabel)
                for lb in reversed(labels):
                    if hasattr(lb, "pixmap"):
                        pm = lb.pixmap()
                        if pm is not None and not pm.isNull() and pm.width() > 32 and pm.height() > 32:
                            return pm.toImage()
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _player_position_seconds(self, video_obj) -> Optional[float]:
        try:
            objs = [video_obj]
            for a in ("player", "mediaPlayer", "mp", "qplayer"):
                try:
                    o = getattr(video_obj, a, None)
                except Exception:
                    o = None
                if o is not None:
                    objs.append(o)
            for o in objs:
                if o is None:
                    continue
                for name in ("position", "currentPosition", "pos", "position_ms", "current_ms", "currentTime"):
                    if not hasattr(o, name):
                        continue
                    try:
                        v = getattr(o, name)
                        v = v() if callable(v) else v
                    except Exception:
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    v = float(v)
                    if "ms" in name or v > 10000.0:
                        return max(0.0, v / 1000.0)
                    return max(0.0, v)
        except Exception:
            pass
        return None

    def _ffmpeg_exe_for_media(self) -> str:
        try:
            raw = self.ffmpeg_row.text().strip() if hasattr(self, "ffmpeg_row") else ""
        except Exception:
            raw = ""
        candidates = []
        if raw:
            candidates.append(raw)
        try:
            candidates.append(str(DEFAULT_FRAME_FFMPEG))
        except Exception:
            pass
        candidates.append("ffmpeg")
        for cand in candidates:
            if not cand:
                continue
            try:
                p = Path(str(cand)).expanduser()
                if p.exists():
                    return str(p)
            except Exception:
                pass
            if str(cand).lower() == "ffmpeg":
                return "ffmpeg"
        return "ffmpeg"

    def _save_qimage_png(self, qimg: QImage, out_path: Path) -> bool:
        try:
            img = qimg
            return bool(img.save(str(out_path), "PNG"))
        except Exception:
            return False

    def _export_current_media_to_temp_image(self) -> Optional[Path]:
        try:
            DEFAULT_VIDEO_FRAME_DIR.mkdir(parents=True, exist_ok=True)
            out_png = DEFAULT_VIDEO_FRAME_DIR / f"ltx23_current_{int(time.time())}.png"
        except Exception:
            return None
        main = self._find_main_with_video()
        try:
            video = getattr(main, "video", None) if main is not None else None
        except Exception:
            video = None
        src = self._current_media_path()
        img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        vid_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
        if src is not None:
            ext = (src.suffix or "").lower()
            try:
                if ext in img_exts:
                    return src
                if ext in vid_exts:
                    ff = self._ffmpeg_exe_for_media()
                    sec = self._player_position_seconds(video) if video is not None else None
                    cmd = [ff, "-y", "-hide_banner"]
                    if sec is not None:
                        cmd += ["-ss", f"{sec:.3f}"]
                    cmd += ["-i", str(src), "-frames:v", "1", str(out_png)]
                    try:
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        q = QImage(str(out_png))
                        if not q.isNull():
                            return out_png
                    except Exception:
                        pass
            except Exception:
                pass
        qimg = self._grab_current_qimage()
        if qimg is None or qimg.isNull():
            return None
        if self._save_qimage_png(qimg, out_png):
            return out_png
        return None

    def _use_current_for_image_row(self, row: PathRow, label: str) -> None:
        path = self._export_current_media_to_temp_image()
        if path is None or not Path(str(path)).exists():
            QMessageBox.warning(self, "No current image", "No current image or video frame was found.\n\nLoad an image or pause a video in the Media Player first.")
            return
        row.setText(str(path))
        try:
            self._append_log(f"Using Media Player current image/frame for {label}: {path}")
        except Exception:
            pass

    def _use_current_for_video_row(self, row: PathRow, label: str) -> None:
        src = self._current_media_path()
        vid_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
        if src is None or (src.suffix or "").lower() not in vid_exts:
            QMessageBox.warning(self, "No current video", "No current video was found.\n\nLoad a video in the Media Player first, then click Use Current again.")
            return
        row.setText(str(src))
        try:
            self._append_log(f"Using Media Player current video for {label}: {src}")
        except Exception:
            pass

    def view_results(self) -> None:
        target = self.active_output_path.parent if self.active_output_path else Path(self.output_dir_row.text() or str(DEFAULT_OUTPUT_DIR))
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        main = self._find_main_with_video()
        try:
            if main is not None and hasattr(main, "open_media_explorer_folder"):
                main.open_media_explorer_folder(str(target), preset="videos", include_subfolders=False)
                return
        except Exception:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _spin(self, minimum: int, maximum: int, step: int, value: int) -> QSpinBox:
        w = WheelGuardSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setValue(value)
        return w

    def _double_spin(self, minimum: float, maximum: float, step: float, value: float) -> QDoubleSpinBox:
        w = WheelGuardDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setDecimals(3)
        w.setValue(value)
        return w

    def _set_shift_value(self, value: float) -> None:
        value = max(0.0, min(10.0, float(value)))
        if hasattr(self, "shift_spin"):
            self.shift_spin.blockSignals(True)
            self.shift_spin.setValue(value)
            self.shift_spin.blockSignals(False)
        if hasattr(self, "shift_slider"):
            self.shift_slider.blockSignals(True)
            self.shift_slider.setValue(int(round(value * 100)))
            self.shift_slider.blockSignals(False)

    def _shift_slider_changed(self, raw_value: int) -> None:
        if hasattr(self, "shift_spin"):
            self.shift_spin.blockSignals(True)
            self.shift_spin.setValue(float(raw_value) / 100.0)
            self.shift_spin.blockSignals(False)
        self._settings_changed()

    def _shift_spin_changed(self, value: float) -> None:
        if hasattr(self, "shift_slider"):
            self.shift_slider.blockSignals(True)
            self.shift_slider.setValue(int(round(float(value) * 100)))
            self.shift_slider.blockSignals(False)
        self._settings_changed()

    def _set_tooltip(self, widget: QWidget, text: str) -> None:
        """Apply one tooltip to a widget and its visible child controls.

        PathRow is a small compound widget, so the tooltip needs to be applied
        to the label, edit box and buttons as well. This keeps the UI readable
        without adding more visible help text everywhere.
        """
        if widget is None or not text:
            return
        try:
            widget.setToolTip(text)
        except Exception:
            pass
        for attr in ("label", "edit", "browse_btn", "use_current_btn", "open_btn", "toggle", "content"):
            child = getattr(widget, attr, None)
            if child is not None:
                try:
                    child.setToolTip(text)
                except Exception:
                    pass

    def _is_auto_vram_profile_mode(self, profile: Optional[str] = None) -> bool:
        raw = str(profile if profile is not None else (self.vram_profile_combo.currentText() if hasattr(self, "vram_profile_combo") else "24") or "24").strip().lower()
        return raw in {"auto", "detect", "gpu"}

    def _current_auto_vram_resolution_key(self) -> str:
        width = self.width_spin.value() if hasattr(self, "width_spin") else 1280
        height = self.height_spin.value() if hasattr(self, "height_spin") else 704
        return _ltx_auto_resolution_key(width, height)

    def _current_frame_limit(self) -> int:
        """Return the active UI frame ceiling.

        Auto frame ceilings depend on both VRAM profile and resolution
        bucket: 12 GB = 480p/704p/1088p -> 601/289/133, 16 GB ->
        901/433/199, and 24 GB -> 1201/577/265.
        """
        try:
            profile_key = self._effective_vram_profile_key(str(self.vram_profile_combo.currentText() or "24"))
        except Exception:
            profile_key = "24"
        try:
            resolution_key = self._current_auto_vram_resolution_key()
        except Exception:
            resolution_key = "704p"
        anchors = LTX_AUTO_RESOLUTION_ANCHORS.get(resolution_key) or {}
        if profile_key == "24":
            return int(anchors.get("max_frames_24gb", 481) or 481)
        if profile_key == "16":
            return int(anchors.get("max_frames_16gb", anchors.get("max_frames_default", 481)) or 481)
        return int(anchors.get("max_frames_12gb", anchors.get("max_frames_default", 481)) or 481)

    def _update_frame_spin_limit(self, *, clamp: bool = True) -> None:
        if not hasattr(self, "frames_spin"):
            return
        try:
            limit = int(self._current_frame_limit())
            current = int(self.frames_spin.value())
            if int(self.frames_spin.maximum()) != limit:
                self.frames_spin.blockSignals(True)
                self.frames_spin.setMaximum(limit)
                if clamp and current > limit:
                    self.frames_spin.setValue(limit)
                self.frames_spin.blockSignals(False)
            elif clamp and current > limit:
                self.frames_spin.blockSignals(True)
                self.frames_spin.setValue(limit)
                self.frames_spin.blockSignals(False)
        except Exception:
            pass

    def _current_auto_vram_settings(self) -> Dict[str, Any]:
        profile_key = self._effective_vram_profile_key(str(self.vram_profile_combo.currentText() or "24"))
        return calculate_ltx_auto_vram_settings(
            int(profile_key),
            self._current_auto_vram_resolution_key(),
            int(self.frames_spin.value() if hasattr(self, "frames_spin") else 121),
            self._active_vram_limit_workflow_group_name(),
        )

    def _set_auto_vram_spin_state(self, auto_mode: bool) -> None:
        # Only the real "auto" profile locks/greys the budget fields. Manual
        # 24/16/12 profiles still auto-fill suggested values, but must remain
        # editable so lower-VRAM profiles and frame limits can be tested.
        for name in ("stable_hotset_budget_gb_spin", "main_hot_window_spin", "stage2_block_size_limit_spin"):
            spin = getattr(self, name, None)
            if spin is None:
                continue
            try:
                spin.setReadOnly(bool(auto_mode))
            except Exception:
                pass
            try:
                spin.setEnabled(True)
            except Exception:
                pass

    def _apply_auto_vram_settings(self, *, update_hint: bool = True, overwrite_manual: bool = False) -> Optional[Dict[str, Any]]:
        # The tuning table is useful for explicit 12/16/24 GB testing too.
        # "auto" decides the VRAM profile and locks the controls.
        # Manual 12/16/24 profiles get one suggested autofill after profile /
        # resolution / frame changes, but the user can freely edit afterwards.
        auto_profile_mode = self._is_auto_vram_profile_mode()
        self._set_auto_vram_spin_state(auto_profile_mode)
        result = self._current_auto_vram_settings()
        should_write_values = bool(auto_profile_mode or overwrite_manual)
        if should_write_values and result.get("supported"):
            hotset = float(result.get("hotset_gb", 0.0))
            stage1 = float(result.get("stage1_gb", hotset))
            stage2 = float(result.get("stage2_gb", 0.0))
            self._updating_vram_profile_controls = True
            try:
                for spin, value in (
                    (getattr(self, "stable_hotset_budget_gb_spin", None), hotset),
                    (getattr(self, "main_hot_window_spin", None), stage1),
                    (getattr(self, "stage2_block_size_limit_spin", None), stage2),
                ):
                    if spin is None:
                        continue
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)
            finally:
                self._updating_vram_profile_controls = False
        if update_hint:
            self._update_vram_profile_hint()
        return result

    def _schedule_vram_profile_autofill(self) -> None:
        # Debounce frame/resolution changes so the boxes update once after the
        # user stops changing frames. Manual edits after that are left alone
        # until frames/resolution/profile changes again.
        timer = getattr(self, "_vram_profile_autofill_timer", None)
        if timer is not None:
            timer.start(650)
        else:
            QTimer.singleShot(650, lambda: self._apply_auto_vram_settings(update_hint=True, overwrite_manual=True))

    def _ensure_auto_vram_settings_supported(self) -> bool:
        if self._is_auto_vram_profile_mode():
            result = self._apply_auto_vram_settings(update_hint=True, overwrite_manual=True) or {}
        else:
            result = self._current_auto_vram_settings()
            self._update_vram_profile_hint()
        if result.get("supported"):
            return True
        QMessageBox.warning(self, "LTX 2.3", str(result.get("reason") or "The selected VRAM profile is unsupported for the selected resolution/frame count."))
        return False

    def _frame_count_changed(self, *args: Any) -> None:
        self._update_frame_spin_limit(clamp=False)
        self._update_extended_frames_warning()
        self._schedule_vram_profile_autofill()

    def _effective_vram_profile_key(self, profile: Optional[str] = None) -> str:
        profile_key = str(profile or self.vram_profile_combo.currentText() or "24").strip().lower()
        if profile_key in {"auto", "detect", "gpu"}:
            return _auto_ui_vram_profile_key()
        if profile_key.endswith("gb"):
            profile_key = profile_key[:-2].strip()
        return profile_key if profile_key in {"12", "16", "24"} else "24"

    def _active_vram_limit_workflow_group_name(self, pipeline_name: Optional[str] = None) -> str:
        if pipeline_name is None:
            try:
                pipeline_name = self._effective_pipeline()
            except Exception:
                pipeline_name = self.pipeline_combo.currentText() if hasattr(self, "pipeline_combo") else "one_stage"
        return "two_stage" if str(pipeline_name) in TWO_STAGE_PIPELINES else "one_stage"

    def _profile_default_main_hot_window(self, profile: Optional[str] = None, workflow_group: Optional[str] = None) -> float:
        profile_key = self._effective_vram_profile_key(profile)
        group = str(workflow_group or self._active_vram_limit_workflow_group_name()).strip().lower()
        defaults = LTX_MAIN_HOT_WINDOW_DEFAULTS_GB if group == "two_stage" else LTX_ONE_STAGE_MAIN_HOT_WINDOW_DEFAULTS_GB
        return float(defaults.get(profile_key, defaults.get("24", LTX_MAIN_HOT_WINDOW_DEFAULTS_GB["24"])))

    def _profile_default_stage2_hot_window(self, profile: Optional[str] = None) -> float:
        profile_key = self._effective_vram_profile_key(profile)
        return float(LTX_STAGE2_HOT_WINDOW_DEFAULTS_GB.get(profile_key, LTX_STAGE2_HOT_WINDOW_DEFAULTS_GB["24"]))

    def _profile_default_residency_strategy(self, profile: Optional[str] = None) -> str:
        profile_key = self._effective_vram_profile_key(profile)
        return str(LTX_PROFILE_RESIDENCY_DEFAULTS.get(profile_key, "rolling"))

    def _apply_profile_residency_default(self, profile: Optional[str] = None) -> None:
        if not hasattr(self, "planned_hotset_check"):
            return
        raw_profile_key = str(profile or self.vram_profile_combo.currentText() or "24").strip().lower()
        profile_key = self._effective_vram_profile_key(raw_profile_key)
        strategy = self._profile_default_residency_strategy(raw_profile_key)
        is_low_profile = profile_key in {"12", "16"}
        self.planned_hotset_check.blockSignals(True)
        self.planned_hotset_check.setChecked(strategy == "planned_hotset")
        self.planned_hotset_check.setEnabled(True)
        self.planned_hotset_check.setToolTip(
            "24 GB uses planned_hotset by default for speed. "
            "12/16 GB use rolling by default for safety, but you can manually enable this toggle for testing; "
            "the CLI will receive the explicit low-profile override."
            if is_low_profile else
            "24 GB uses planned_hotset by default for speed. Turn this off to test rolling mode."
        )
        self.planned_hotset_check.blockSignals(False)

    def _workflow_vram_profile_limit_dict(self, workflow_group: Optional[str] = None) -> Dict[str, float]:
        group = str(workflow_group or self._active_vram_limit_workflow_group_name()).strip().lower()
        return self._vram_profile_block_limits_two_stage if group == "two_stage" else self._vram_profile_block_limits_one_stage

    def _repair_cross_workflow_limit_defaults(self) -> None:
        """Clean stale saved defaults that came from the other workflow.

        Earlier test builds only had one saved block-limit map, so a one-phase
        default such as 15.0 GB could accidentally become the saved two-phase
        value. Do not treat those exact opposite-workflow defaults as real user
        tuning; let the active workflow fall back to its own proper default.
        """
        try:
            profiles = set(LTX_MAIN_HOT_WINDOW_DEFAULTS_GB.keys())
            for profile_key in profiles:
                two_default = float(LTX_MAIN_HOT_WINDOW_DEFAULTS_GB.get(profile_key, 0.0))
                one_default = float(LTX_ONE_STAGE_MAIN_HOT_WINDOW_DEFAULTS_GB.get(profile_key, 0.0))
                if profile_key in self._vram_profile_block_limits_two_stage:
                    try:
                        value = float(self._vram_profile_block_limits_two_stage.get(profile_key, 0.0))
                        if abs(value - one_default) < 0.001:
                            self._vram_profile_block_limits_two_stage.pop(profile_key, None)
                    except Exception:
                        pass
                if profile_key in self._vram_profile_block_limits_one_stage:
                    try:
                        value = float(self._vram_profile_block_limits_one_stage.get(profile_key, 0.0))
                        if abs(value - two_default) < 0.001:
                            self._vram_profile_block_limits_one_stage.pop(profile_key, None)
                    except Exception:
                        pass
        except Exception:
            pass

    def _normalize_profile_block_limit(self, profile: str, value: Any, workflow_group: Optional[str] = None) -> float:
        default = self._profile_default_main_hot_window(profile, workflow_group)
        try:
            parsed = float(value)
        except Exception:
            return default
        if parsed < 0:
            return default
        old_values = LTX_OLD_MAIN_HOT_WINDOW_DEFAULT_VALUES_GB.get(self._effective_vram_profile_key(profile), set())
        if any(abs(parsed - float(old)) < 0.001 for old in old_values):
            return default
        return parsed

    def _current_profile_block_limit(self, profile: Optional[str] = None, workflow_group: Optional[str] = None) -> float:
        profile_key = str(profile or self.vram_profile_combo.currentText() or "24")
        limits = self._workflow_vram_profile_limit_dict(workflow_group)
        group = str(workflow_group or self._active_vram_limit_workflow_group_name()).strip().lower()
        if profile_key in limits:
            return self._normalize_profile_block_limit(profile_key, limits[profile_key], group)
        return self._profile_default_main_hot_window(profile_key, group)

    def _store_current_vram_profile_limit(self, workflow_group: Optional[str] = None) -> None:
        if not hasattr(self, "main_hot_window_spin") or not hasattr(self, "vram_profile_combo"):
            return
        profile_key = str(self._active_vram_profile or self.vram_profile_combo.currentText() or "24")
        if self._is_auto_vram_profile_mode(profile_key):
            return
        group = str(workflow_group or self._active_vram_limit_workflow_group_name()).strip().lower()
        self._workflow_vram_profile_limit_dict(group)[profile_key] = float(self.main_hot_window_spin.value())
        # Keep the legacy single-dict mirror in sync with the currently active workflow
        # so old save/load code paths still have a sane fallback value.
        self._vram_profile_block_limits = dict(self._workflow_vram_profile_limit_dict(group))

    def _set_block_limit_spin_for_profile(self, profile: str) -> None:
        if not hasattr(self, "main_hot_window_spin"):
            return
        workflow_group = self._active_vram_limit_workflow_group_name()
        self._active_vram_limit_workflow_group = workflow_group
        self._apply_profile_residency_default(profile)
        # Profile changes get one immediate suggested autofill. Manual profiles
        # remain editable after this until the next frame/resolution/profile
        # change.
        self._apply_auto_vram_settings(update_hint=True, overwrite_manual=True)

    def _vram_profile_changed(self, new_profile: str) -> None:
        if getattr(self, "_updating_vram_profile_controls", False):
            return
        old_profile = str(getattr(self, "_active_vram_profile", "") or "")
        if old_profile and old_profile != str(new_profile):
            self._store_current_vram_profile_limit()
        self._active_vram_profile = str(new_profile or "24")
        self._set_block_limit_spin_for_profile(self._active_vram_profile)
        self._update_frame_spin_limit(clamp=True)
        self._update_extended_frames_warning()

    def _block_size_limit_changed(self, *args: Any) -> None:
        if getattr(self, "_updating_vram_profile_controls", False):
            return
        profile_key = str(self.vram_profile_combo.currentText() or self._active_vram_profile or "24")
        if self._is_auto_vram_profile_mode(profile_key):
            self._apply_auto_vram_settings(update_hint=True)
            return
        workflow_group = self._active_vram_limit_workflow_group_name()
        self._workflow_vram_profile_limit_dict(workflow_group)[profile_key] = float(self.main_hot_window_spin.value())
        self._vram_profile_block_limits = dict(self._workflow_vram_profile_limit_dict(workflow_group))
        self._update_vram_profile_hint()

    def _update_vram_profile_hint(self) -> None:
        if not hasattr(self, "vram_profile_hint"):
            return
        raw_profile_key = str(self.vram_profile_combo.currentText() or "24").strip().lower()
        effective_profile_key = self._effective_vram_profile_key(raw_profile_key)
        workflow_group = self._active_vram_limit_workflow_group_name()
        if self._is_auto_vram_profile_mode(raw_profile_key):
            result = self._current_auto_vram_settings()
            resolution_label = str(result.get("resolution_label") or self._current_auto_vram_resolution_key())
            max_frames = int(result.get("max_frames_for_profile_resolution") or 0)
            workflow_label = "one-phase" if workflow_group == "one_stage" else "two-phase"
            if result.get("supported"):
                if workflow_group == "one_stage":
                    self.vram_profile_hint.setText(
                        f"Auto {effective_profile_key} GB / {resolution_label} / {workflow_label}: {self.frames_spin.value()} frames → "
                        f"hotset {float(result.get('hotset_gb', 0.0)):.1f} GB, "
                        f"one-phase block size {float(result.get('stage1_gb', 0.0)):.1f} GB. "
                        f"one-phase Auto follows the tuned Stage-2/refine budget. "
                        f"Max for this profile/resolution: {max_frames} frames."
                    )
                else:
                    self.vram_profile_hint.setText(
                        f"Auto {effective_profile_key} GB / {resolution_label} / {workflow_label}: {self.frames_spin.value()} frames → "
                        f"hotset {float(result.get('hotset_gb', 0.0)):.1f} GB, "
                        f"Stage 1 {float(result.get('stage1_gb', 0.0)):.1f} GB, "
                        f"Stage 2 {float(result.get('stage2_gb', 0.0)):.1f} GB. "
                        f"Max for this profile/resolution: {max_frames} frames."
                    )
            else:
                self.vram_profile_hint.setText(str(result.get("reason") or "Auto profile unsupported."))
            return
        default_gb = self._profile_default_main_hot_window(raw_profile_key, workflow_group)
        active_gb = float(self.main_hot_window_spin.value()) if hasattr(self, "main_hot_window_spin") else default_gb
        if active_gb <= 0:
            mode = "profile default"
            shown_gb = default_gb
        elif abs(active_gb - default_gb) < 0.001:
            mode = "profile default"
            shown_gb = active_gb
        else:
            mode = "saved for this profile"
            shown_gb = active_gb
        stage2_gb = float(self.stage2_block_size_limit_spin.value()) if hasattr(self, "stage2_block_size_limit_spin") else self._profile_default_stage2_hot_window(raw_profile_key)
        selected_text = f"Selected {effective_profile_key} GB profile"
        workflow_label = "one-phase" if workflow_group == "one_stage" else "two-phase"
        self.vram_profile_hint.setText(
            f"{selected_text}. {workflow_label} defaults active. "
            f"Block limits: Stage 1 {shown_gb:.1f} GB ({mode}) / Stage 2 {stage2_gb:.1f} GB."
        )

    def _update_extended_frames_warning(self, *args: Any) -> None:
        if not hasattr(self, "extended_frames_warning_label"):
            return
        is_extended = False
        try:
            is_extended = int(self.frames_spin.value()) > LTX_STANDARD_FRAME_LIMIT
        except Exception:
            is_extended = False
        auto_vram_enabled = self._is_auto_vram_profile_mode() if hasattr(self, "vram_profile_combo") else False
        if self._selected_model_variant() == "INT4" and hasattr(self, "vram_lab_combo"):
            auto_vram_enabled = self.vram_lab_combo.currentText().strip().upper() != "OFF"
        should_show_manual_warning = bool(is_extended and not auto_vram_enabled)
        self.extended_frames_warning_label.setVisible(should_show_manual_warning)
        if hasattr(self, "auto_vram_advice_label"):
            self.auto_vram_advice_label.setVisible(should_show_manual_warning)

    def _safe_resolution_pair(self, width: int, height: int) -> Tuple[int, int]:
        """Return a resolution that is safe for all LTX pipelines, including two-phase."""
        width = max(64, int(round(int(width) / 64) * 64))
        height = max(64, int(round(int(height) / 64) * 64))
        return width, height

    def _normalize_resolution_for_pipeline(self) -> None:
        """Keep UI resolution values on 64-pixel boundaries before building commands."""
        width, height = self._safe_resolution_pair(self.width_spin.value(), self.height_spin.value())
        changed = width != self.width_spin.value() or height != self.height_spin.value()
        if changed:
            self.width_spin.blockSignals(True)
            self.height_spin.blockSignals(True)
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
            self.width_spin.blockSignals(False)
            self.height_spin.blockSignals(False)
        text = f"{width}x{height}"
        if self.resolution_combo.currentText() != text:
            self.resolution_combo.blockSignals(True)
            self.resolution_combo.setEditText(text)
            self.resolution_combo.blockSignals(False)

    def _resolution_text_changed(self, text: str) -> None:
        match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", text or "", re.I)
        if not match:
            return
        raw_text = f"{int(match.group(1))}x{int(match.group(2))}"
        replacement = BAD_RESOLUTION_REPLACEMENTS.get(raw_text)
        if replacement:
            repl_match = re.match(r"^(\d+)x(\d+)$", replacement)
            width, height = int(repl_match.group(1)), int(repl_match.group(2))
        else:
            width, height = int(match.group(1)), int(match.group(2))
            if self.snap_resolution_check.isChecked():
                width, height = self._safe_resolution_pair(width, height)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        safe_text = f"{width}x{height}"
        if self.resolution_combo.currentText() != safe_text:
            self.resolution_combo.blockSignals(True)
            self.resolution_combo.setEditText(safe_text)
            self.resolution_combo.blockSignals(False)
        self._update_frame_spin_limit(clamp=True)
        self._schedule_vram_profile_autofill()
        self._refresh_command_preview()

    def _manual_resolution_changed(self) -> None:
        if self.snap_resolution_check.isChecked():
            sender = self.sender()
            if isinstance(sender, QSpinBox):
                value = sender.value()
                snapped = max(64, int(round(value / 64) * 64))
                if snapped != value:
                    sender.blockSignals(True)
                    sender.setValue(snapped)
                    sender.blockSignals(False)
        text = f"{self.width_spin.value()}x{self.height_spin.value()}"
        if self.resolution_combo.currentText() != text:
            self.resolution_combo.blockSignals(True)
            self.resolution_combo.setEditText(text)
            self.resolution_combo.blockSignals(False)
        self._update_frame_spin_limit(clamp=True)
        self._schedule_vram_profile_autofill()
        self._refresh_command_preview()

    def _browse_reference_images(self) -> None:
        start = self._last_browse_dirs.get("reference_images") or self._folder_from_paths_text(self.reference_images_edit.text()) or str(APP_ROOT)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Reference images",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )
        if not files:
            return
        self._remember_last_folder("reference_images", files[0])
        existing = [p for p in self._split_paths(self.reference_images_edit.text()) if p]
        merged = existing + [f for f in files if f not in existing]
        self.reference_images_edit.setText(";".join(merged))
        self._settings_changed()

    def _split_paths(self, text: str) -> List[str]:
        return [p.strip().strip('"') for p in re.split(r"[;\n]+", text or "") if p.strip()]

    def _quote_command(self, parts: Iterable[str]) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline([str(p) for p in parts])
        return " ".join(shlex.quote(str(p)) for p in parts)

    def _append_extra_pair(self, extra: List[str], arg_name: str, value: str) -> None:
        value = str(value or "").strip()
        if arg_name and arg_name != EXTRA_ARG_DISABLED and value:
            extra.extend([arg_name, value])

    def _append_extra_paths(self, extra: List[str], arg_name: str, paths_text: str) -> None:
        paths = self._split_paths(paths_text)
        if arg_name and arg_name != EXTRA_ARG_DISABLED and paths:
            extra.append(arg_name)
            extra.extend(paths)

    def _make_output_path(self) -> Path:
        out_dir = Path(self.output_dir_row.text() or str(DEFAULT_OUTPUT_DIR)).expanduser()
        name = self.output_name_edit.text().strip()
        if not name:
            seed = self.seed_spin.value()
            safe_prompt = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.prompt_edit.toPlainText().strip())[:48].strip("_") or "ltx"
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"ltx23_{stamp}_{self.width_spin.value()}x{self.height_spin.value()}_{self.frames_spin.value()}f_seed{seed}_{safe_prompt}.mp4"
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        return out_dir / name

    def collect_settings(self) -> LTX23UISettings:
        self._store_current_vram_profile_limit()
        return LTX23UISettings(
            pipeline=self.pipeline_combo.currentText(),
            vram_lab=self._current_vram_lab_cli_value(),
            fast_iclora_route=bool(getattr(self, "_fast_iclora_user_preference", True)),
            vram_profile=self.vram_profile_combo.currentText(),
            main_hot_window_gb=self.main_hot_window_spin.value(),
            stage2_block_size_limit_gb=self.stage2_block_size_limit_spin.value(),
            vram_residency_strategy=("planned_hotset" if self.planned_hotset_check.isChecked() else "rolling"),
            stable_hotset_fraction=self.stage1_stable_hotset_fraction_spin.value(),
            stage1_stable_hotset_fraction=self.stage1_stable_hotset_fraction_spin.value(),
            stage2_stable_hotset_fraction=self.stage2_stable_hotset_fraction_spin.value(),
            stable_hotset_budget_gb=self.stable_hotset_budget_gb_spin.value(),
            emergency_free_vram_floor_gb=self.emergency_free_vram_floor_spin.value(),
            vram_profile_block_size_limits=dict(self._workflow_vram_profile_limit_dict(self._active_vram_limit_workflow_group_name())),
            vram_profile_block_size_limits_one_stage=dict(self._vram_profile_block_limits_one_stage),
            vram_profile_block_size_limits_two_stage=dict(self._vram_profile_block_limits_two_stage),
            prompt=self.prompt_edit.toPlainText(),
            negative_prompt=self.negative_edit.toPlainText(),
            latent_preview_enabled=bool(self.latent_preview_enabled_check.isChecked()),
            latent_preview_mode=self.latent_preview_mode_combo.currentText(),
            latent_preview_rate=int(self.latent_preview_rate_spin.value()),
            latent_preview_upscale=bool(self.latent_preview_upscale_check.isChecked()),
            latent_preview_tae_decode=bool(self.latent_preview_tae_decode_check.isChecked()),
            keep_latents=bool(getattr(self, "keep_latents_check", None) and self.keep_latents_check.isChecked()),
            resolution=self.resolution_combo.currentText(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            frames=self.frames_spin.value(),
            fps=self.fps_spin.value(),
            steps=self.steps_spin.value(),
            seed=self.seed_spin.value(),
            random_seed=self.random_seed_check.isChecked(),
            snap_resolution=self.snap_resolution_check.isChecked(),
            no_boundary_echo=self.no_boundary_echo_check.isChecked(),
            deep_lifecycle_log=self.deep_lifecycle_check.isChecked(),
            deep_log_interval=self.deep_interval_spin.value(),
            deep_log_max_events=self.deep_max_events_spin.value(),
            distilled_lora_path=self.distilled_lora_row.text(),
            distilled_lora_strength=self.distilled_lora_strength_spin.value(),
            user_lora_1_path=self.user_lora_rows[0].text(),
            user_lora_1_strength=self.user_lora_strength_spins[0].value(),
            user_lora_2_path=self.user_lora_rows[1].text(),
            user_lora_2_strength=self.user_lora_strength_spins[1].value(),
            user_lora_3_path=self.user_lora_rows[2].text(),
            user_lora_3_strength=self.user_lora_strength_spins[2].value(),
            user_lora_4_path=self.user_lora_rows[3].text(),
            user_lora_4_strength=self.user_lora_strength_spins[3].value(),
            disable_distilled_lora=True,
            disable_distilled_lora_extra_args="",
            spatial_upsampler_path=self.spatial_upsampler_row.text(),
            auto_download_two_stage_assets=self.auto_download_two_stage_check.isChecked(),
            lora_fusion_cache_mode=LORA_CACHE_MODE_OFF,
            lora_fusion_cache_shard_gb=4.0,
            lora_fusion_cache_shard_threshold_gb=999999.0,
            lora_fusion_cache_max_files=0,
            lora_fusion_cache_miss_inplace=False,
            main_transformer_stream_probe=False,
            start_media_path=self.start_media_row.text(),
            start_video_path=self.start_video_row.text(),
            start_image_frame=self.start_image_frame_spin.value(),
            start_image_strength=self.start_image_strength_spin.value(),
            end_media_path=self.end_media_row.text(),
            end_video_path=self.end_video_row.text(),
            glue_input_videos=bool(getattr(self, "glue_input_videos_check", None) and self.glue_input_videos_check.isChecked()),
            end_image_strength=self.end_image_strength_spin.value(),
            reference_images=self.reference_images_edit.text(),
            reference_image_strength=self.reference_image_strength_spin.value(),
            ltx_normalize_input_image=self.normalize_input_image_check.isChecked(),
            source_video_path="",
            audio_path=self.audio_row.text(),
            audio_mode=AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED),
            audio_start_time=self.audio_start_spin.value(),
            audio_max_duration=self.audio_max_duration_spin.value(),
            remux_audio_bitrate=self.remux_bitrate_combo.currentText(),
            remux_shortest=self.remux_shortest_check.isChecked(),
            remux_replace_output=self.remux_replace_check.isChecked(),
            safe_ltx_audio_load=True,
            video_cfg_guidance_scale=self.video_cfg_spin.value(),
            scheduler_shift=self.shift_spin.value(),
            video_stg_guidance_scale=self.video_stg_spin.value(),
            video_rescale_scale=self.video_rescale_spin.value(),
            audio_cfg_guidance_scale=self.audio_cfg_spin.value(),
            audio_stg_guidance_scale=self.audio_stg_spin.value(),
            audio_rescale_scale=self.audio_rescale_spin.value(),
            a2v_guidance_scale=self.a2v_guidance_spin.value(),
            v2a_guidance_scale=self.v2a_guidance_spin.value(),
            video_skip_step=self.video_skip_step_spin.value(),
            audio_skip_step=self.audio_skip_step_spin.value(),
            max_batch_size=self._current_batch_size(),
            enhance_prompt=self.enhance_prompt_check.isChecked(),
            enable_flash_attention=(self.attention_backend_combo.currentText().strip().lower() == "flash2"),
            attention_backend=self.attention_backend_combo.currentText().strip().lower(),
            quantization_mode=self.quantization_combo.currentText() if hasattr(self, "quantization_combo") else QUANTIZATION_MODE_NONE,
            custom_extra_args=self.extra_args_edit.toPlainText(),
            ltx_root=self.ltx_root_row.text(),
            python_exe=self.python_row.text(),
            cli_path=self.cli_row.text(),
            checkpoint_path=self.checkpoint_row.text(),
            gemma_root=self.gemma_row.text(),
            output_dir=self.output_dir_row.text(),
            output_name=self.output_name_edit.text(),
            report_path=self.report_row.text(),
            deep_log_path=self.deep_log_row.text(),
            ffmpeg_path=self.ffmpeg_row.text(),
            open_output_when_done=self.open_when_done_check.isChecked(),
            use_framevision_queue=bool(getattr(self, "use_framevision_queue_check", None) and self.use_framevision_queue_check.isChecked()),
            last_used_folders=self._collect_last_used_folders(),
        )


    def apply_settings(self, data: Dict[str, Any]) -> None:
        self._loading = True
        raw_data = data or {}
        s = {**asdict(LTX23UISettings()), **raw_data}
        self._set_combo(self.pipeline_combo, s["pipeline"])
        self._set_combo(self.vram_lab_combo, self._display_vram_lab_value(s["vram_lab"]))
        self._fast_iclora_user_preference = bool(s.get("fast_iclora_route", True))
        if hasattr(self, "fast_iclora_route_check"):
            self.fast_iclora_route_check.setChecked(self._fast_iclora_user_preference)
        selected_profile = str(s.get("vram_profile", "24"))
        raw_profile_limits_legacy = s.get("vram_profile_block_size_limits", {})
        raw_profile_limits_one_stage = s.get("vram_profile_block_size_limits_one_stage", {})
        raw_profile_limits_two_stage = s.get("vram_profile_block_size_limits_two_stage", {})
        self._vram_profile_block_limits = {}
        self._vram_profile_block_limits_one_stage = {}
        self._vram_profile_block_limits_two_stage = {}

        def _load_profile_limit_map(raw_limits: Any, workflow_group: str) -> Dict[str, float]:
            loaded: Dict[str, float] = {}
            if isinstance(raw_limits, dict):
                for profile_key, profile_value in raw_limits.items():
                    profile_key = str(profile_key)
                    if profile_key in LTX_MAIN_HOT_WINDOW_DEFAULTS_GB or profile_key in {"auto", "detect", "gpu"}:
                        normalized = self._normalize_profile_block_limit(profile_key, profile_value, workflow_group)
                        old_default = LTX_OLD_MAIN_HOT_WINDOW_DEFAULTS_GB.get(profile_key)
                        if old_default is not None and abs(normalized - float(old_default)) < 0.001:
                            normalized = self._profile_default_main_hot_window(profile_key, workflow_group)
                        loaded[profile_key] = normalized
            return loaded

        self._vram_profile_block_limits_one_stage = _load_profile_limit_map(raw_profile_limits_one_stage, "one_stage")
        self._vram_profile_block_limits_two_stage = _load_profile_limit_map(raw_profile_limits_two_stage, "two_stage")

        # Backward compatibility: if the new workflow-specific maps are missing,
        # use the legacy single map as the two-phase source and derive one-phase
        # defaults separately.
        if not self._vram_profile_block_limits_one_stage and not self._vram_profile_block_limits_two_stage:
            self._vram_profile_block_limits_two_stage = _load_profile_limit_map(raw_profile_limits_legacy, "two_stage")

        legacy_limit = raw_data.get("main_hot_window_gb", None)
        if selected_profile not in self._vram_profile_block_limits_two_stage and legacy_limit is not None:
            normalized_legacy = self._normalize_profile_block_limit(selected_profile, legacy_limit, "two_stage")
            old_default = LTX_OLD_MAIN_HOT_WINDOW_DEFAULTS_GB.get(selected_profile)
            if old_default is not None and abs(normalized_legacy - float(old_default)) < 0.001:
                normalized_legacy = self._profile_default_main_hot_window(selected_profile, "two_stage")
            self._vram_profile_block_limits_two_stage[selected_profile] = normalized_legacy
        self._repair_cross_workflow_limit_defaults()
        self._active_vram_profile = selected_profile
        self._active_vram_limit_workflow_group = self._active_vram_limit_workflow_group_name(str(s.get("pipeline", "one_stage")))
        self._vram_profile_block_limits = dict(self._workflow_vram_profile_limit_dict(self._active_vram_limit_workflow_group))
        self._set_combo(self.vram_profile_combo, selected_profile)
        self._set_block_limit_spin_for_profile(selected_profile)
        # Stage 2 follows the selected VRAM profile default. Older saved JSONs can
        # contain a stale value from another profile, so do not let that override
        # the profile switch/default behavior.
        self.stage2_block_size_limit_spin.setValue(self._profile_default_stage2_hot_window(selected_profile))
        # Profile owns the safe default. Older saved JSON may still say planned_hotset
        # for 12/16 GB, so default lower effective profiles back to rolling, but keep
        # the toggle enabled for explicit manual testing.
        effective_selected_profile = self._effective_vram_profile_key(selected_profile)
        if effective_selected_profile == "24":
            self.planned_hotset_check.setChecked(str(s.get("vram_residency_strategy", "planned_hotset")).strip().lower() != "rolling")
        else:
            self.planned_hotset_check.setChecked(False)
        self.planned_hotset_check.setEnabled(True)
        # Migrate old saved/default hotset fractions for the 24 GB profile to the
        # current proven CFG 2 baseline. Exact old defaults are treated as stale
        # defaults, not as user tuning; non-default custom values are preserved.
        effective_selected_profile = self._effective_vram_profile_key(selected_profile)
        try:
            legacy_fraction = float(s.get("stable_hotset_fraction", 1.15))
        except Exception:
            legacy_fraction = 1.15
        try:
            stage1_fraction = float(s.get("stage1_stable_hotset_fraction", legacy_fraction))
        except Exception:
            stage1_fraction = legacy_fraction
        try:
            stage2_fraction = float(s.get("stage2_stable_hotset_fraction", min(1.10, legacy_fraction)))
        except Exception:
            stage2_fraction = min(1.10, legacy_fraction)
        if effective_selected_profile == "24":
            if abs(stage1_fraction - 0.95) < 0.001 or abs(stage1_fraction - 0.82) < 0.001:
                stage1_fraction = 1.15
            if abs(stage2_fraction - 0.95) < 0.001 or abs(stage2_fraction - 0.82) < 0.001 or abs(stage2_fraction - 0.85) < 0.001:
                stage2_fraction = 0.9
        self.stage1_stable_hotset_fraction_spin.setValue(stage1_fraction)
        self.stage2_stable_hotset_fraction_spin.setValue(stage2_fraction)
        try:
            self.stable_hotset_budget_gb_spin.setValue(float(s.get("stable_hotset_budget_gb", 0.0)))
        except Exception:
            self.stable_hotset_budget_gb_spin.setValue(0.0)
        try:
            self.emergency_free_vram_floor_spin.setValue(float(s.get("emergency_free_vram_floor_gb", 0.5)))
        except Exception:
            self.emergency_free_vram_floor_spin.setValue(0.5)
        self.prompt_edit.setPlainText(s["prompt"])
        self.negative_edit.setPlainText(s["negative_prompt"])
        self.latent_preview_enabled_check.setChecked(bool(s.get("latent_preview_enabled", False)))
        self._set_combo(self.latent_preview_mode_combo, str(s.get("latent_preview_mode", "Fast Latent RGB")))
        try:
            self.latent_preview_rate_spin.setValue(int(s.get("latent_preview_rate", 8)))
        except Exception:
            self.latent_preview_rate_spin.setValue(8)
        self.latent_preview_upscale_check.setChecked(bool(s.get("latent_preview_upscale", False)))
        self.latent_preview_tae_decode_check.setChecked(bool(s.get("latent_preview_tae_decode", False)))
        self.keep_latents_check.setChecked(bool(s.get("keep_latents", False)))
        saved_resolution = str(s.get("resolution", "1280x704"))
        saved_resolution = BAD_RESOLUTION_REPLACEMENTS.get(saved_resolution, saved_resolution)
        res_match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", saved_resolution, re.I)
        if res_match:
            safe_width, safe_height = self._safe_resolution_pair(int(res_match.group(1)), int(res_match.group(2)))
        else:
            safe_width, safe_height = self._safe_resolution_pair(int(s.get("width", 1280)), int(s.get("height", 704)))
        safe_resolution = f"{safe_width}x{safe_height}"
        self.resolution_combo.setEditText(safe_resolution)
        self.width_spin.setValue(safe_width)
        self.height_spin.setValue(safe_height)
        self.frames_spin.setValue(int(s["frames"]))
        self.fps_spin.setValue(int(s["fps"]))
        self.steps_spin.setValue(int(s["steps"]))
        self.seed_spin.setValue(int(s["seed"]))
        self.random_seed_check.setChecked(bool(s["random_seed"]))
        self.snap_resolution_check.setChecked(bool(s["snap_resolution"]))
        self.no_boundary_echo_check.setChecked(bool(s["no_boundary_echo"]))
        self.deep_lifecycle_check.setChecked(bool(s["deep_lifecycle_log"]))
        self.deep_interval_spin.setValue(float(s["deep_log_interval"]))
        self.deep_max_events_spin.setValue(int(s["deep_log_max_events"]))
        self.distilled_lora_row.setText(s["distilled_lora_path"])
        self.distilled_lora_strength_spin.setValue(float(s["distilled_lora_strength"]))
        for index, row in enumerate(getattr(self, "user_lora_rows", []), start=1):
            row.setText(str(s.get(f"user_lora_{index}_path", "") or ""))
            self._set_user_lora_strength(index - 1, s.get(f"user_lora_{index}_strength", 1.0))
        if hasattr(self, "disable_distilled_lora_check"):
            self.disable_distilled_lora_check.setChecked(True)
        if hasattr(self, "disable_lora_extra_args_edit"):
            self.disable_lora_extra_args_edit.setText("")
        self._update_test_lora_strength_ui()
        self.spatial_upsampler_row.setText(s["spatial_upsampler_path"])
        self.auto_download_two_stage_check.setChecked(bool(s.get("auto_download_two_stage_assets", True)))
        self._set_combo(self.lora_cache_mode_combo, LORA_CACHE_MODE_OFF)
        self.lora_cache_shard_gb_spin.setValue(4.0)
        self.lora_cache_shard_threshold_spin.setValue(999999.0)
        self.lora_cache_max_spin.setValue(0)
        self.lora_cache_miss_inplace_check.setChecked(False)
        if hasattr(self, "main_transformer_stream_probe_check"):
            self.main_transformer_stream_probe_check.setChecked(bool(s.get("main_transformer_stream_probe", False)))
        self._update_two_stage_asset_status()
        self._refresh_lora_cache_status()
        self.start_media_row.setText(s.get("start_media_path", ""))
        self.start_video_row.setText(s.get("start_video_path", ""))
        self.start_image_frame_spin.setValue(int(s.get("start_image_frame", 0)))
        self.start_image_strength_spin.setValue(float(s.get("start_image_strength", 1.0)))
        self.end_media_row.setText(s.get("end_media_path", ""))
        self.end_video_row.setText(s.get("end_video_path", ""))
        if hasattr(self, "glue_input_videos_check"):
            self.glue_input_videos_check.setChecked(bool(s.get("glue_input_videos", False)))
        self.end_image_strength_spin.setValue(float(s.get("end_image_strength", 1.0)))
        self.reference_images_edit.setText(s.get("reference_images", ""))
        self.reference_image_strength_spin.setValue(float(s.get("reference_image_strength", 0.8)))
        self.normalize_input_image_check.setChecked(bool(s.get("ltx_normalize_input_image", True)))
        self.audio_row.setText(s["audio_path"])
        audio_mode = AUDIO_MODE_COMPAT_MAP.get(str(s.get("audio_mode", AUDIO_MODE_REMUX)).strip(), AUDIO_MODE_DISABLED)
        self._set_combo(self.audio_mode_combo, audio_mode)
        self.audio_start_spin.setValue(float(s.get("audio_start_time", 0.0)))
        self.audio_max_duration_spin.setValue(float(s.get("audio_max_duration", 0.0)))
        self._set_combo(self.remux_bitrate_combo, s["remux_audio_bitrate"])
        self.remux_shortest_check.setChecked(bool(s["remux_shortest"]))
        self.remux_replace_check.setChecked(bool(s["remux_replace_output"]))
        self._audio_mode_changed(audio_mode)
        self.video_cfg_spin.setValue(float(s.get("video_cfg_guidance_scale", 2.0)))
        self._set_shift_value(float(s.get("scheduler_shift", 5.0)))
        self.video_stg_spin.setValue(float(s.get("video_stg_guidance_scale", 0.0)))
        self.video_rescale_spin.setValue(float(s.get("video_rescale_scale", 0.7)))
        self.audio_cfg_spin.setValue(float(s.get("audio_cfg_guidance_scale", 1.0)))
        self.audio_stg_spin.setValue(float(s.get("audio_stg_guidance_scale", 0.0)))
        self.audio_rescale_spin.setValue(float(s.get("audio_rescale_scale", 0.0)))
        self.a2v_guidance_spin.setValue(float(s.get("a2v_guidance_scale", 1.0)))
        self.v2a_guidance_spin.setValue(float(s.get("v2a_guidance_scale", 1.0)))
        self.video_skip_step_spin.setValue(int(s.get("video_skip_step", 0)))
        self.audio_skip_step_spin.setValue(int(s.get("audio_skip_step", 0)))
        self._set_combo(self.max_batch_size_combo, str(self._normalize_batch_size_value(s.get("max_batch_size", 2))))
        self.enhance_prompt_check.setChecked(bool(s.get("enhance_prompt", False)))
        attention_backend = str(s.get("attention_backend", "") or "").strip().lower()
        if not attention_backend:
            attention_backend = "flash2" if bool(s.get("enable_flash_attention", False)) else "auto"
        if attention_backend == "pytorch":
            attention_backend = "sdpa"
        if attention_backend not in ("auto", "sdpa", "flash2", "sage"):
            attention_backend = "auto"
        self._set_combo(self.attention_backend_combo, attention_backend)
        self.flash_attention_check.setChecked(attention_backend == "flash2")
        if hasattr(self, "quantization_combo"):
            quantization_mode = str(s.get("quantization_mode", QUANTIZATION_MODE_NONE) or QUANTIZATION_MODE_NONE).strip()
            if quantization_mode.lower() == "auto":
                quantization_mode = QUANTIZATION_MODE_AUTO
            elif quantization_mode.lower() == "none":
                quantization_mode = QUANTIZATION_MODE_NONE
            elif quantization_mode not in QUANTIZATION_MODE_CHOICES:
                quantization_mode = QUANTIZATION_MODE_NONE
            self._set_combo(self.quantization_combo, quantization_mode)
        self.main_transformer_stream_probe_check.setChecked(False)
        # Keep old JSON compatibility: removed experimental UI fields are ignored.
        self.extra_args_edit.setPlainText(s.get("custom_extra_args", ""))
        # First-run and stale-default repair: old builds pointed the LTX root at
        # C:\ltx23/C:\ltx. When there is no saved JSON, or when an old JSON still
        # contains those baked-in defaults, use the actual FrameVision install root
        # so every dependent folder is portable/offline by default.
        old_ltx_roots = {str(Path(r"C:\ltx23")).lower(), str(Path(r"C:\ltx")).lower()}
        ltx_root_text = str(s.get("ltx_root", "") or "").strip()
        if not ltx_root_text or str(Path(ltx_root_text)).lower() in old_ltx_roots:
            s["ltx_root"] = str(DEFAULT_LTX_ROOT)
            s["python_exe"] = str(DEFAULT_PYTHON)
            s["checkpoint_path"] = str(DEFAULT_CHECKPOINT)
            s["gemma_root"] = str(DEFAULT_GEMMA_ROOT)
            s["output_dir"] = str(DEFAULT_OUTPUT_DIR)
        self.ltx_root_row.setText(s["ltx_root"])
        self.python_row.setText(s["python_exe"])
        self.cli_row.setText(s["cli_path"])
        self.checkpoint_row.setText(s["checkpoint_path"])
        if hasattr(self, "model_variant_combo"):
            loaded_variant = self._infer_model_variant_from_checkpoint(s.get("checkpoint_path", ""))
            self._set_combo(self.model_variant_combo, loaded_variant)
            self._apply_model_variant_route(loaded_variant, update_checkpoint=False)
        if hasattr(self, "quantization_combo"):
            self._apply_quantization_availability()
        self.gemma_row.setText(s["gemma_root"])
        self.output_dir_row.setText(s["output_dir"])
        self.output_name_edit.setText(s["output_name"])
        self.report_row.setText(s["report_path"])
        self.deep_log_row.setText(s["deep_log_path"])
        self.ffmpeg_row.setText(s["ffmpeg_path"])
        self.open_when_done_check.setChecked(bool(s["open_output_when_done"]))
        if hasattr(self, "use_framevision_queue_check"):
            self.use_framevision_queue_check.setChecked(bool(s.get("use_framevision_queue", True)))
        self._last_browse_dirs = dict(s.get("last_used_folders") or {})
        self._apply_last_used_folders()
        self._loading = False
        self._update_frame_spin_limit(clamp=True)
        self._update_extended_frames_warning()
        if self._selected_model_variant() != "INT4":
            self._apply_auto_vram_settings(update_hint=True)
        self._update_fast_iclora_route_controls()
        self._update_flash_attention_status()
        self._refresh_command_preview()


    def _set_combo(self, combo: QComboBox, value: Any) -> None:
        text = str(value)
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable():
            combo.setEditText(text)


    def _normalize_lora_cache_mode(self, value: Any) -> str:
        text = str(value or "").strip()
        low = text.lower()
        return LORA_CACHE_MODE_COMPAT_MAP.get(low, text if text in LORA_CACHE_MODE_CHOICES else LORA_CACHE_MODE_READ)

    def _lora_cache_folder(self) -> Path:
        root_text = self.ltx_root_row.text().strip() if hasattr(self, "ltx_root_row") else ""
        root = Path(root_text or str(DEFAULT_LTX_ROOT)).expanduser()
        return root / "models" / "ltx23" / "fused_lora_cache"

    def _refresh_lora_cache_status(self) -> None:
        if not hasattr(self, "lora_cache_status_label"):
            return
        folder = self._lora_cache_folder()
        if not folder.exists():
            self.lora_cache_status_label.setText(f"Cache folder does not exist yet: {folder}")
            return
        try:
            singles = list(folder.glob("*.safetensors"))
            manifests = list(folder.glob("*/manifest.json"))
            partials = list(folder.glob("*.tmp")) + list(folder.glob("*.tmp_*")) + list(folder.glob("*.shards.tmp_*"))
            total = 0
            for path in list(singles) + list(manifests):
                try:
                    total += int(path.stat().st_size)
                except Exception:
                    pass
            self.lora_cache_status_label.setText(
                f"{folder} | single files: {len(singles)} | sharded caches: {len(manifests)} | partial/temp: {len(partials)}"
            )
        except Exception as exc:
            self.lora_cache_status_label.setText(f"Cache status failed: {type(exc).__name__}: {exc}")

    def _open_lora_cache_folder(self) -> None:
        folder = self._lora_cache_folder()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        self._refresh_lora_cache_status()

    def _delete_partial_lora_caches(self) -> None:
        folder = self._lora_cache_folder()
        if not folder.exists():
            self._refresh_lora_cache_status()
            return
        candidates: List[Path] = []
        for pattern in ("*.tmp", "*.tmp_*", "*.shards.tmp_*"):
            candidates.extend(folder.glob(pattern))
        # Also remove shard folders that never got a manifest, but leave completed caches alone.
        for child in folder.iterdir():
            if child.is_dir() and child.name.endswith(".shards") and not (child / "manifest.json").exists():
                candidates.append(child)
        unique: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(path)
        if not unique:
            QMessageBox.information(self, "LTX 2.3", "No partial LoRA cache files found.")
            self._refresh_lora_cache_status()
            return
        reply = QMessageBox.question(
            self,
            "Delete partial LoRA caches",
            f"Delete {len(unique)} partial/temp cache item(s)?\n\nCompleted single and manifest-based sharded caches are kept.",
        )
        if reply != QMessageBox.Yes:
            return
        deleted = 0
        errors: List[str] = []
        for path in unique:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted += 1
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        if errors:
            QMessageBox.warning(self, "LTX 2.3", f"Deleted {deleted} item(s), but some failed:\n" + "\n".join(errors[:6]))
        else:
            QMessageBox.information(self, "LTX 2.3", f"Deleted {deleted} partial/temp cache item(s).")
        self._refresh_lora_cache_status()

    def _lora_strength_spin_changed(self, value: float, slider: QSlider) -> None:
        target = int(round(max(0.0, min(2.0, float(value))) * 100.0))
        if slider.value() != target:
            slider.blockSignals(True)
            slider.setValue(target)
            slider.blockSignals(False)

    def _lora_strength_slider_changed(self, value: int, spin: QDoubleSpinBox) -> None:
        target = max(0.0, min(2.0, float(value) / 100.0))
        if abs(float(spin.value()) - target) > 0.0001:
            spin.blockSignals(True)
            spin.setValue(target)
            spin.blockSignals(False)

    def _set_user_lora_strength(self, index: int, value: Any) -> None:
        try:
            strength = max(0.0, min(2.0, float(value)))
        except Exception:
            strength = 1.0
        try:
            spin = self.user_lora_strength_spins[index]
            slider = self.user_lora_strength_sliders[index]
        except Exception:
            return
        spin.blockSignals(True)
        slider.blockSignals(True)
        spin.setValue(strength)
        slider.setValue(int(round(strength * 100.0)))
        slider.blockSignals(False)
        spin.blockSignals(False)

    def _user_lora_entries(self) -> List[Tuple[str, float]]:
        entries: List[Tuple[str, float]] = []
        rows = list(getattr(self, "user_lora_rows", []))
        spins = list(getattr(self, "user_lora_strength_spins", []))
        for row, spin in zip(rows, spins):
            path = row.text().strip()
            if not path:
                continue
            try:
                strength = float(spin.value())
            except Exception:
                strength = 1.0
            if strength <= 0.0:
                continue
            entries.append((path, strength))
        return entries

    def _user_lora_cli_args(self) -> List[str]:
        args: List[str] = []
        for path, strength in self._user_lora_entries():
            args.extend(["--lora", path, f"{strength:g}"])
        return args

    def _has_user_loras(self) -> bool:
        try:
            return any(bool(str(path).strip()) for path, _strength in self._user_lora_entries())
        except Exception:
            return False

    def _effective_distilled_lora_strength(self) -> float:
        """Return the normal approved LoRA strength from the Inputs tab."""
        try:
            return float(self.distilled_lora_strength_spin.value())
        except Exception:
            return 1.0

    def _distilled_lora_strength_arg(self) -> str:
        return f"{self._effective_distilled_lora_strength():g}"

    def _update_test_lora_strength_ui(self) -> None:
        if not hasattr(self, "test_lora_strength_status_label"):
            return
        mode = "n/a"
        try:
            mode = self._normalize_lora_cache_mode(self.lora_cache_mode_combo.currentText())
        except Exception:
            pass
        value = self._effective_distilled_lora_strength()
        disabled = bool(getattr(self, "disable_distilled_lora_check", None) and self.disable_distilled_lora_check.isChecked())
        if disabled:
            self.test_lora_strength_status_label.setText(
                f"No-LoRA test is ON: distilled LoRA and fused cache will be skipped. Normal Quality strength would be {value:g}."
            )
        else:
            self.test_lora_strength_status_label.setText(
                f"Normal LoRA mode: active strength {value:g} from the Inputs tab. Cache mode: {mode}."
            )

    def _lora_cache_cli_args(self) -> List[str]:
        if bool(getattr(self, "disable_distilled_lora_check", None) and self.disable_distilled_lora_check.isChecked()):
            return ["--lora-fusion-cache", "off"]
        mode = self._normalize_lora_cache_mode(self.lora_cache_mode_combo.currentText())
        shard_gb = max(1.0, float(self.lora_cache_shard_gb_spin.value()))
        threshold_gb = max(1.0, float(self.lora_cache_shard_threshold_spin.value()))
        args: List[str] = []
        if mode == LORA_CACHE_MODE_OFF:
            return ["--lora-fusion-cache", "off"]
        elif mode == LORA_CACHE_MODE_READ:
            args.extend(["--lora-fusion-cache", "read"])
        elif mode == LORA_CACHE_MODE_SHARDED:
            args.extend(["--lora-fusion-cache", "auto", "--lora-fusion-cache-shard-gb", f"{shard_gb:g}", "--lora-fusion-cache-shard-threshold-gb", f"{threshold_gb:g}"])
        elif mode == LORA_CACHE_MODE_REBUILD_SINGLE:
            args.extend(["--lora-fusion-cache", "rebuild", "--lora-fusion-cache-shard-threshold-gb", "999999"])
        elif mode == LORA_CACHE_MODE_REBUILD_SHARDED:
            args.extend(["--lora-fusion-cache", "rebuild", "--lora-fusion-cache-shard-gb", f"{shard_gb:g}", "--lora-fusion-cache-shard-threshold-gb", f"{threshold_gb:g}"])
        else:
            args.extend(["--lora-fusion-cache", "auto", "--lora-fusion-cache-shard-threshold-gb", "999999"])
        args.extend(["--lora-fusion-cache-max", str(self.lora_cache_max_spin.value())])
        args.extend(["--lora-fusion-cache-miss-inplace", "auto" if self.lora_cache_miss_inplace_check.isChecked() else "off"])
        # Cache creation should not stack stale CUDA/Python allocations on top of the build path.
        # Read-only/off modes do not create cache, but passing this is harmless and keeps command previews explicit.
        args.extend(["--lora-fusion-cache-preclean", "auto"])
        return args

    def _display_vram_lab_value(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        return "OFF" if text == "off" else "ON"

    def _current_vram_lab_cli_value(self) -> str:
        return "off" if self.vram_lab_combo.currentText().strip().upper() == "OFF" else "safe"

    def _normalize_batch_size_value(self, value: Any) -> int:
        try:
            ivalue = int(value)
        except Exception:
            return 2
        if ivalue <= 1:
            return 1
        if ivalue >= 4:
            return 4
        return 2

    def _current_batch_size(self) -> int:
        # one-phase should always run with batch size 1. The visible batch-size
        # control only applies to two-phase workflows.
        try:
            pipeline_name = self._effective_pipeline()
        except Exception:
            pipeline_name = self.pipeline_combo.currentText() if hasattr(self, "pipeline_combo") else "one_stage"
        if pipeline_name not in TWO_STAGE_PIPELINES:
            return 1
        return self._normalize_batch_size_value(self.max_batch_size_combo.currentText())


    def _looks_like_internal_two_stage_lora_token(self, value: str) -> bool:
        text = str(value or "").replace("\\", "/").lower()
        name = text.rsplit("/", 1)[-1]
        return (
            "ltx-2.3-22b-distilled-lora" in name
            or ("distilled" in name and "lora" in name and name.endswith((".safetensors", ".pt", ".pth")))
        )

    def _sanitize_extra_for_pipeline(self, pipeline_name: str, extra: List[str]) -> List[str]:
        """Keep one-phase from accidentally receiving the internal two-phase LoRA.

        Normal two-phase runs still pass the distilled LoRA through the dedicated
        CLI option. For one-phase, stale Custom extra args such as --lora pointing
        to the official distilled LoRA can trigger huge RAM-heavy fusion, so strip
        only those internal two-phase assets.
        """
        if pipeline_name in TWO_STAGE_PIPELINES:
            return extra
        cleaned: List[str] = []
        removed: List[str] = []
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
                if any(self._looks_like_internal_two_stage_lora_token(x) for x in group[1:]):
                    removed.append("--lora internal-distilled-two-phase-lora")
                    continue
                cleaned.extend(group)
                continue
            cleaned.append(tok)
            i += 1
        if removed:
            self._append_log("one-phase guard removed internal two-phase asset args from Custom extra: " + "; ".join(removed))
        return cleaned

    def _update_pipeline_dependent_controls(self) -> None:
        """Update controls that only apply to two-phase style workflows."""
        if not hasattr(self, "pipeline_combo"):
            return

        if hasattr(self, "audio_mode_combo"):
            pipeline_name = self._effective_pipeline()
        else:
            pipeline_name = self.pipeline_combo.currentText()
        is_two_stage = pipeline_name in TWO_STAGE_PIPELINES

        # Max batch size only matters for the two-phase workflows in this UI.
        # Keep one-phase simpler by hiding the setting entirely and forcing the
        # command value to 1 while one-phase is selected.
        previous_is_two_stage = getattr(self, "_last_is_two_stage_workflow", None)
        previous_group = "two_stage" if previous_is_two_stage else "one_stage" if previous_is_two_stage is not None else None
        current_group = "two_stage" if is_two_stage else "one_stage"
        if previous_group and previous_group != current_group:
            self._store_current_vram_profile_limit(previous_group)
            self._repair_cross_workflow_limit_defaults()
            self._active_vram_limit_workflow_group = current_group
            self._vram_profile_block_limits = dict(self._workflow_vram_profile_limit_dict(current_group))
            self._set_block_limit_spin_for_profile(str(self.vram_profile_combo.currentText() or self._active_vram_profile or "24"))
        else:
            self._repair_cross_workflow_limit_defaults()
            self._active_vram_limit_workflow_group = current_group
            self._vram_profile_block_limits = dict(self._workflow_vram_profile_limit_dict(current_group))
        if hasattr(self, "max_batch_size_combo"):
            self.max_batch_size_combo.blockSignals(True)
            if is_two_stage and previous_is_two_stage is False:
                self.max_batch_size_combo.setCurrentText("2")
            elif not is_two_stage:
                self.max_batch_size_combo.setCurrentText("1")
            self.max_batch_size_combo.blockSignals(False)
        self._last_is_two_stage_workflow = is_two_stage

        for widget_name in ("max_batch_size_label", "max_batch_size_combo"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(is_two_stage)

        for widget in list(getattr(self, "user_lora_rows", [])) + list(getattr(self, "user_lora_strength_spins", [])) + list(getattr(self, "user_lora_strength_sliders", [])):
            widget.setEnabled(True)

        for widget_name in (
            "spatial_upsampler_row",
            "auto_download_two_stage_check", "two_stage_download_btn",
            "stage2_block_size_limit_spin", "stage2_block_size_limit_help",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(is_two_stage)
        if self._is_auto_vram_profile_mode():
            self._apply_auto_vram_settings(update_hint=True)
        if hasattr(self, "two_stage_status_label"):
            self._update_two_stage_asset_status()

    def _frame_ffmpeg_path(self) -> str:
        bundled = DEFAULT_FRAME_FFMPEG
        if bundled.exists():
            return str(bundled)
        text = self.ffmpeg_row.text().strip() if hasattr(self, "ffmpeg_row") else ""
        return text or "ffmpeg"

    def _frame_ffprobe_path(self, ffmpeg_path: str) -> str:
        try:
            ffmpeg = Path(ffmpeg_path)
            if ffmpeg.name.lower() in {"ffmpeg.exe", "ffmpeg"}:
                probe_name = "ffprobe.exe" if ffmpeg.name.lower().endswith(".exe") else "ffprobe"
                probe = ffmpeg.with_name(probe_name)
                if probe.exists():
                    return str(probe)
        except Exception:
            pass
        return "ffprobe"

    def _video_frame_output_path(self, video_path: Path, boundary: str) -> Path:
        try:
            stat = video_path.stat()
            key = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{boundary}"
        except Exception:
            key = f"{video_path}|{boundary}|{time.time()}"
        digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem).strip("._") or "video"
        return DEFAULT_VIDEO_FRAME_DIR / f"{safe_stem}_{boundary}_{digest}.png"

    def _run_frame_command(self, args: List[str], label: str) -> None:
        completed = subprocess.run(
            args,
            cwd=str(APP_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 1200:
                detail = detail[-1200:]
            raise RuntimeError(f"{label} failed. {detail}")

    def _probe_video_frame_count(self, video_path: Path, ffmpeg_path: str) -> int:
        ffprobe = self._frame_ffprobe_path(ffmpeg_path)
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_frames",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "default=nokey=1:noprint_wrappers=1",
                    str(video_path),
                ],
                cwd=str(APP_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
            if completed.returncode == 0:
                for line in (completed.stdout or "").splitlines():
                    line = line.strip()
                    if line.isdigit():
                        return int(line)
        except Exception:
            pass
        return 0

    def _extract_video_boundary_frame(self, video_text: str, boundary: str) -> str:
        video_path = Path(video_text).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        DEFAULT_VIDEO_FRAME_DIR.mkdir(parents=True, exist_ok=True)
        output_path = self._video_frame_output_path(video_path, boundary)
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        ffmpeg = self._frame_ffmpeg_path()
        if boundary == "start":
            args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(video_path),
                "-map", "0:v:0",
                "-frames:v", "1",
                "-c:v", "png",
                "-compression_level", "0",
                str(output_path),
            ]
            self._run_frame_command(args, "Start video frame extraction")
            return str(output_path)

        frame_count = self._probe_video_frame_count(video_path, ffmpeg)
        if frame_count > 0:
            select_expr = f"select=eq(n\\,{frame_count - 1})"
            args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(video_path),
                "-vf", select_expr,
                "-vsync", "0",
                "-frames:v", "1",
                "-c:v", "png",
                "-compression_level", "0",
                str(output_path),
            ]
            self._run_frame_command(args, "End video frame extraction")
            return str(output_path)

        fallback_args = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-sseof", "-10",
            "-i", str(video_path),
            "-vf", "reverse",
            "-frames:v", "1",
            "-c:v", "png",
            "-compression_level", "0",
            str(output_path),
        ]
        self._run_frame_command(fallback_args, "End video frame extraction")
        return str(output_path)

    def _prepared_frame_for_video(self, boundary: str, video_text: str) -> str:
        video_text = str(video_text or "").strip()
        if not video_text:
            return ""
        if boundary == "start":
            prepared = self._prepared_start_frame_path
            source = self._prepared_start_video_source
        else:
            prepared = self._prepared_end_frame_path
            source = self._prepared_end_video_source
        if prepared and source == video_text and Path(prepared).exists():
            return prepared
        return ""

    def _cleanup_prepared_video_frames(self, *, success_only: bool = True) -> None:
        """Remove temporary PNG frames extracted from Continue/End videos after a finished run.

        The source videos and generated/glued outputs are never touched.  On failures
        we keep these frames so the failed command can still be inspected/retried.
        """
        if not success_only:
            return
        paths = []
        for attr in ("_prepared_start_frame_path", "_prepared_end_frame_path"):
            try:
                value = str(getattr(self, attr, "") or "").strip()
            except Exception:
                value = ""
            if value:
                paths.append(value)
        cleaned = 0
        for value in dict.fromkeys(paths):
            try:
                p = Path(value).expanduser()
                # Only clean files in FrameVision's temp video-frame folder.
                if p.exists() and p.is_file() and p.parent.resolve() == DEFAULT_VIDEO_FRAME_DIR.resolve():
                    p.unlink()
                    cleaned += 1
            except Exception:
                pass
        if cleaned:
            self._append_log(f"Cleaned {cleaned} temporary LTX video frame(s).")
        for attr in ("_prepared_start_frame_path", "_prepared_end_frame_path", "_prepared_start_video_source", "_prepared_end_video_source"):
            try:
                setattr(self, attr, "")
            except Exception:
                pass
        try:
            # Remove the temp frame folder only when it is empty.
            DEFAULT_VIDEO_FRAME_DIR.rmdir()
        except Exception:
            pass

    def _prepare_video_input_frames(self) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        start_video = self.start_video_row.text().strip() if hasattr(self, "start_video_row") else ""
        end_video = self.end_video_row.text().strip() if hasattr(self, "end_video_row") else ""

        if start_video:
            # "Continue video" must continue from the final decoded frame of the
            # selected start video. Using the first frame here creates a visible
            # story/time jump when the videos are glued together.
            frame = self._extract_video_boundary_frame(start_video, "end")
            self._prepared_start_frame_path = frame
            self._prepared_start_video_source = start_video
            resolved["start"] = frame
            self._append_log(f"Start/continue video LAST frame prepared: {frame}")
        if end_video:
            # "End with video" must guide the generated clip toward the first
            # decoded frame of the selected end video. Using the end video's last
            # frame makes the glued result jump when the end video begins.
            frame = self._extract_video_boundary_frame(end_video, "start")
            self._prepared_end_frame_path = frame
            self._prepared_end_video_source = end_video
            resolved["end"] = frame
            self._append_log(f"End-with video FIRST frame prepared: {frame}")
        return resolved

    # ------------------------------------------------------------------
    # Command building / validation
    # ------------------------------------------------------------------
    def _extra_has_option(self, tokens: List[str], option: str) -> bool:
        target = str(option).strip().lower()
        return any(str(tok).strip().lower() == target for tok in tokens)

    def _selected_quantization_cli_value(self) -> str:
        mode = self.quantization_combo.currentText().strip() if hasattr(self, "quantization_combo") else QUANTIZATION_MODE_NONE
        low = mode.lower()
        # Safety rule: never infer FP8 quantization from the checkpoint filename.
        # The FP8 checkpoint currently runs fast but can produce invalid green/tiny
        # outputs when the native path is not wired exactly right. Only pass an
        # official quantization arg when the user explicitly selects a concrete
        # FP8 test mode. Auto/None both mean: pass no --quantization arg.
        if low in {"fp8-cast", "fp8-scaled-mm"}:
            return low
        return ""

    def _has_image_or_video_conditioning_inputs(self) -> bool:
        inputs = [
            self.start_media_row.text().strip() if hasattr(self, "start_media_row") else "",
            self.end_media_row.text().strip() if hasattr(self, "end_media_row") else "",
            self.start_video_row.text().strip() if hasattr(self, "start_video_row") else "",
            self.end_video_row.text().strip() if hasattr(self, "end_video_row") else "",
            self.reference_images_edit.text().strip() if hasattr(self, "reference_images_edit") else "",
        ]
        return any(bool(value) for value in inputs)

    def _effective_fast_iclora_route_enabled(self) -> bool:
        return bool(getattr(self, "_fast_iclora_user_preference", True)) and not self._has_image_or_video_conditioning_inputs()

    def _remember_fast_iclora_route_preference(self, checked: bool) -> None:
        if getattr(self, "_loading", False):
            return
        if getattr(self, "_fast_iclora_auto_forced_off", False):
            return
        self._fast_iclora_user_preference = bool(checked)

    def _update_fast_iclora_route_controls(self) -> None:
        if not hasattr(self, "fast_iclora_route_check"):
            return
        forced_off = self._has_image_or_video_conditioning_inputs()
        self._fast_iclora_auto_forced_off = forced_off
        tooltip = str(getattr(self, "_fast_iclora_base_tooltip", "") or "")
        if forced_off:
            tooltip = (tooltip + "\n\nAutomatically OFF while any start / middle / end image or video conditioning input is loaded.").strip()
        self.fast_iclora_route_check.blockSignals(True)
        try:
            self.fast_iclora_route_check.setToolTip(tooltip)
            if forced_off:
                self.fast_iclora_route_check.setChecked(False)
                self.fast_iclora_route_check.setEnabled(False)
            else:
                self.fast_iclora_route_check.setEnabled(True)
                self.fast_iclora_route_check.setChecked(bool(getattr(self, "_fast_iclora_user_preference", True)))
        finally:
            self.fast_iclora_route_check.blockSignals(False)

    @staticmethod
    def _timestamped_diagnostic_path(path_text: str, stamp: str) -> str:
        """Return a per-run report/log path without changing the saved UI base path."""
        raw = str(path_text or "").strip()
        if not raw:
            return raw
        path = Path(raw)
        suffix = path.suffix or ".txt"
        stem = path.stem if path.suffix else path.name
        return str(path.with_name(f"{stem}_{stamp}{suffix}"))

    def _run_diagnostic_paths(self) -> Tuple[str, str]:
        """Create matching timestamped summary and deep-lifecycle filenames."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return (
            self._timestamped_diagnostic_path(self.report_row.text(), stamp),
            self._timestamped_diagnostic_path(self.deep_log_row.text(), stamp),
        )

    def _build_int4_command(self, *, prepare_video_inputs: bool = False) -> Tuple[str, List[str], Path, List[str]]:
        """Build the isolated INT4 command with under-the-hood VRAM automation."""
        python_exe = self.python_row.text() or str(DEFAULT_PYTHON)
        cli_path = str(INT4_CLI_PATH)
        self._normalize_resolution_for_pipeline()
        output_path = self._make_output_path()
        report_path, deep_log_path = self._run_diagnostic_paths()
        audio_mode = AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED)
        # Native FP16/FP8 keeps using a2vid_two_stage.  Isolated INT4 maps the
        # same UI mode onto its normal Euler two_stages backend.
        pipeline_name = "two_stages" if audio_mode == AUDIO_MODE_A2V else self._effective_pipeline()
        if pipeline_name not in {"one_stage", "two_stages"}:
            raise RuntimeError("INT4 currently supports one_stage and normal two_stages only. two_stages_hq remains on FP16/FP8.")
        if audio_mode == AUDIO_MODE_A2V:
            audio_path = self.audio_row.text().strip()
            if not audio_path:
                raise RuntimeError("Select an audio file for INT4 Prompt + audio file mode.")
            if not Path(audio_path).is_file():
                raise RuntimeError(f"INT4 reference audio file not found: {audio_path}")
        if getattr(self, "latent_preview_enabled_check", None) is not None and self.latent_preview_enabled_check.isChecked():
            raise RuntimeError("Latent Preview is not connected to the isolated INT4 CLI yet.")
        if self.enhance_prompt_check.isChecked():
            raise RuntimeError("Prompt enhancement is not connected to the isolated INT4 CLI yet.")

        prepared_video_frames = self._prepare_video_input_frames() if prepare_video_inputs else {}
        start_video_text = self.start_video_row.text().strip() if hasattr(self, "start_video_row") else ""
        start_image = (
            prepared_video_frames.get("start")
            or self._prepared_frame_for_video("start", start_video_text)
            or self.start_media_row.text().strip()
        )
        end_video_text = self.end_video_row.text().strip() if hasattr(self, "end_video_row") else ""
        end_image = (
            prepared_video_frames.get("end")
            or self._prepared_frame_for_video("end", end_video_text)
            or self.end_media_row.text().strip()
        )
        references = [path for path in self._split_paths(self.reference_images_edit.text()) if path]

        # INT4 keeps its memory planner entirely under the hood. The visible
        # VRAM Lab toggle only enables/disables that planner; the hidden native
        # VRAM Lab profile controls are never forwarded to the quant CLI.
        int4_auto_vram = self.vram_lab_combo.currentText().strip().upper() != "OFF"
        args = [
            cli_path,
            "--pipeline", pipeline_name,
            "--vram-profile", "auto",
            "--int4-auto-vram" if int4_auto_vram else "--no-int4-auto-vram",
            "--model-root", self.checkpoint_row.text().strip(),
            "--prompt", self.prompt_edit.toPlainText().strip(),
            "--output-path", str(output_path),
            "--height", str(self.height_spin.value()),
            "--width", str(self.width_spin.value()),
            "--num-frames", str(self.frames_spin.value()),
            "--frame-rate", str(self.fps_spin.value()),
            "--num-inference-steps", str(self.steps_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--shift", f"{self.shift_spin.value():g}",
            "--ltx-root", self.ltx_root_row.text(),
            "--report-path", report_path,
            "--attention-backend", "auto",
        ]
        negative_prompt = self.negative_edit.toPlainText().strip()
        if negative_prompt:
            args.extend(["--negative-prompt", negative_prompt])
        if start_image:
            args.extend([
                "--image", start_image,
                str(self.start_image_frame_spin.value()),
                f"{self.start_image_strength_spin.value():g}",
            ])
        if end_image:
            args.extend([
                "--image", end_image,
                str(max(0, self.frames_spin.value() - 1)),
                f"{self.end_image_strength_spin.value():g}",
            ])
        for ref_path in references:
            args.extend([
                "--image", ref_path,
                "0",
                f"{self.reference_image_strength_spin.value():g}",
            ])
        args.extend(self._user_lora_cli_args())
        if hasattr(self, "normalize_input_image_check") and not self.normalize_input_image_check.isChecked():
            args.append("--no-normalize-input-image")
        if pipeline_name == "two_stages" and self.spatial_upsampler_row.text().strip():
            args.extend(["--spatial-upsampler-path", self.spatial_upsampler_row.text().strip()])
        if audio_mode == AUDIO_MODE_A2V:
            args.extend([
                "--audio-path", self.audio_row.text().strip(),
                "--audio-start-time", f"{self.audio_start_spin.value():g}",
            ])
            if self.audio_max_duration_spin.value() > 0:
                args.extend(["--audio-max-duration", f"{self.audio_max_duration_spin.value():g}"])
        if self.no_boundary_echo_check.isChecked():
            args.append("--no-boundary-echo")
        if self.deep_lifecycle_check.isChecked():
            args.extend([
                "--deep-log-interval", str(self.deep_interval_spin.value()),
                "--deep-log-max-events", str(self.deep_max_events_spin.value()),
                "--deep-log-path", deep_log_path,
                "--deep-lifecycle-log",
            ])

        extra: List[str] = [
            "--video-cfg-guidance-scale", f"{self.video_cfg_spin.value():g}",
            "--video-stg-guidance-scale", f"{self.video_stg_spin.value():g}",
            "--video-rescale-scale", f"{self.video_rescale_spin.value():g}",
            "--audio-cfg-guidance-scale", f"{self.audio_cfg_spin.value():g}",
            "--audio-stg-guidance-scale", f"{self.audio_stg_spin.value():g}",
            "--audio-rescale-scale", f"{self.audio_rescale_spin.value():g}",
            "--a2v-guidance-scale", f"{self.a2v_guidance_spin.value():g}",
            "--v2a-guidance-scale", f"{self.v2a_guidance_spin.value():g}",
        ]
        custom_extra = self.extra_args_edit.toPlainText().strip()
        if custom_extra:
            extra.extend(shlex.split(custom_extra, posix=os.name != "nt"))
        if extra:
            args.append("--extra")
            args.extend(extra)
        return python_exe, args, output_path, extra

    def build_command(self, *, randomize_seed: bool = False, prepare_video_inputs: bool = False) -> Tuple[str, List[str], Path, List[str]]:
        if randomize_seed and self.random_seed_check.isChecked():
            self.seed_spin.setValue(int(time.time() * 1000) % 2_147_483_647)

        if self._selected_model_variant() == "INT4":
            return self._build_int4_command(prepare_video_inputs=prepare_video_inputs)

        if self._is_auto_vram_profile_mode():
            auto_result = self._apply_auto_vram_settings(update_hint=True) or {}
            if not auto_result.get("supported"):
                raise RuntimeError(str(auto_result.get("reason") or "Auto VRAM profile is unsupported for the selected resolution/frame count."))
        self._store_current_vram_profile_limit()
        python_exe = self.python_row.text() or str(DEFAULT_PYTHON)
        cli_path = self.cli_row.text() or str(DEFAULT_CLI_PATH)
        self._normalize_resolution_for_pipeline()
        output_path = self._make_output_path()
        report_path, deep_log_path = self._run_diagnostic_paths()
        pipeline_name = self._effective_pipeline()
        audio_mode = AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED)
        extra: List[str] = []
        prepared_video_frames = self._prepare_video_input_frames() if prepare_video_inputs else {}

        negative_prompt = self.negative_edit.toPlainText().strip()
        if negative_prompt:
            extra.extend(["--negative-prompt", negative_prompt])

        if getattr(self, "latent_preview_enabled_check", None) is not None and self.latent_preview_enabled_check.isChecked():
            args_preview_mode = str(self.latent_preview_mode_combo.currentText() or "Fast Latent RGB").strip().lower()
            if "tae" in args_preview_mode:
                args_preview_mode = "tae"
            else:
                args_preview_mode = "fast_rgb"
            extra_preview_rate = max(1, min(30, int(self.latent_preview_rate_spin.value())))
            # Wrapper-level arguments: the CLI consumes these before launching the official LTX module.
            sidecar_path = output_path.with_suffix(".latent_preview.jsonl")
            self._latent_preview_sidecar_path = str(sidecar_path)
            self._latent_preview_dir_path = str(output_path.with_suffix(".latent_preview"))
            extra_preview_args = [
                "--latent-preview",
                "--latent-preview-mode", args_preview_mode,
                "--latent-preview-rate", str(extra_preview_rate),
                "--latent-preview-sidecar", str(sidecar_path),
            ]
            if self.latent_preview_upscale_check.isChecked():
                extra_preview_args.append("--latent-preview-upscale")
            if self.latent_preview_tae_decode_check.isChecked():
                extra_preview_args.append("--latent-preview-tae-decode")
        else:
            self._latent_preview_sidecar_path = ""
            self._latent_preview_dir_path = ""
            extra_preview_args = []

        start_video_text = self.start_video_row.text().strip() if hasattr(self, "start_video_row") else ""
        start_image = (
            prepared_video_frames.get("start")
            or self._prepared_frame_for_video("start", start_video_text)
            or self.start_media_row.text().strip()
        )
        if start_image:
            extra.extend([
                "--image",
                start_image,
                str(self.start_image_frame_spin.value()),
                f"{self.start_image_strength_spin.value():g}",
            ])

        end_video_text = self.end_video_row.text().strip() if hasattr(self, "end_video_row") else ""
        end_image = (
            prepared_video_frames.get("end")
            or self._prepared_frame_for_video("end", end_video_text)
            or self.end_media_row.text().strip()
        )
        if end_image:
            end_frame = max(0, self.frames_spin.value() - 1)
            extra.extend([
                "--image",
                end_image,
                str(end_frame),
                f"{self.end_image_strength_spin.value():g}",
            ])

        for ref_path in self._split_paths(self.reference_images_edit.text()):
            if ref_path:
                extra.extend([
                    "--image",
                    ref_path,
                    "0",
                    f"{self.reference_image_strength_spin.value():g}",
                ])

        extra.extend(self._user_lora_cli_args())

        extra.extend([
            "--video-cfg-guidance-scale", f"{self.video_cfg_spin.value():g}",
            "--video-stg-guidance-scale", f"{self.video_stg_spin.value():g}",
            "--video-rescale-scale", f"{self.video_rescale_spin.value():g}",
            "--audio-cfg-guidance-scale", f"{self.audio_cfg_spin.value():g}",
            "--audio-stg-guidance-scale", f"{self.audio_stg_spin.value():g}",
            "--audio-rescale-scale", f"{self.audio_rescale_spin.value():g}",
            "--a2v-guidance-scale", f"{self.a2v_guidance_spin.value():g}",
            "--v2a-guidance-scale", f"{self.v2a_guidance_spin.value():g}",
            "--video-skip-step", str(self.video_skip_step_spin.value()),
            "--audio-skip-step", str(self.audio_skip_step_spin.value()),
            "--max-batch-size", str(self._current_batch_size()),
        ])
        if self.enhance_prompt_check.isChecked():
            extra.append("--enhance-prompt")

        custom_extra = self.extra_args_edit.toPlainText().strip()
        custom_extra_tokens = shlex.split(custom_extra, posix=os.name != "nt") if custom_extra else []
        quantization_value = self._selected_quantization_cli_value()
        if quantization_value and not self._extra_has_option(extra + custom_extra_tokens, "--quantization"):
            extra.extend(["--quantization", quantization_value])
        if custom_extra_tokens:
            extra.extend(custom_extra_tokens)
        extra = self._sanitize_extra_for_pipeline(pipeline_name, extra)

        args = [
            str(cli_path),
            "--pipeline", pipeline_name,
            "--vram-lab", self._current_vram_lab_cli_value(),
            "--fast-iclora-route", "on" if self._effective_fast_iclora_route_enabled() else "off",
            "--vram-profile", self._effective_vram_profile_key(str(self.vram_profile_combo.currentText() or "24")) if self._is_auto_vram_profile_mode() else self.vram_profile_combo.currentText(),
            "--checkpoint-path", self.checkpoint_row.text(),
            "--gemma-root", self.gemma_row.text(),
            "--prompt", self.prompt_edit.toPlainText().strip(),
            "--output-path", str(output_path),
            "--height", str(self.height_spin.value()),
            "--width", str(self.width_spin.value()),
            "--num-frames", str(self.frames_spin.value()),
            "--frame-rate", str(self.fps_spin.value()),
            "--num-inference-steps", str(self.steps_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--shift", f"{self.shift_spin.value():g}",
            "--ltx-root", self.ltx_root_row.text(),
            "--report-path", report_path,
        ]
        if extra_preview_args:
            args.extend(extra_preview_args)
        # Safe LTX AudioDecoder loading is now owned by the CLI and enabled by default.

        # Keep normal Direct Run on the same minimal modern path as the queue.
        # The CLI owns the no-LoRA path and the 24 GB fast defaults, so do not
        # pass redundant/debug/default override args unless the user explicitly
        # enables the matching debug/override controls below.
        # Do not auto-enable main-transformer-stream-probe for normal LoRA 1-4.
        # Tests showed this only reached the Gemma streaming builder in this repo
        # and made denoise slow.  Normal user LoRA is handled by the CLI-side
        # runtime forward-hook bridge.  Keep this flag manual for diagnostics only.
        if getattr(self, "main_transformer_stream_probe_check", None) is not None and self.main_transformer_stream_probe_check.isChecked():
            args.append("--main-transformer-stream-probe")
        if audio_mode == AUDIO_MODE_A2V:
            args.extend([
                "--audio-path", self.audio_row.text().strip(),
                "--audio-start-time", f"{self.audio_start_spin.value():g}",
            ])
            if self.audio_max_duration_spin.value() > 0:
                args.extend(["--audio-max-duration", f"{self.audio_max_duration_spin.value():g}"])
        profile_for_residency = self._effective_vram_profile_key(str(self.vram_profile_combo.currentText() or "24"))

        # Always pass the complete visible VRAM Lab state. Do not rely on CLI
        # defaults or hidden saved profile maps, because restart/queue tests need
        # the command JSON to be the single source of truth.
        main_hot_window = float(self.main_hot_window_spin.value())
        if main_hot_window > 0:
            args.extend(["--main-hot-window-gb", f"{main_hot_window:g}"])

        residency_strategy = "planned_hotset" if self.planned_hotset_check.isChecked() else "rolling"
        args.extend(["--vram-residency-strategy", residency_strategy])
        if residency_strategy == "planned_hotset" and profile_for_residency in {"12", "16"}:
            args.append("--allow-low-profile-planned-hotset")

        stage1_fraction = float(self.stage1_stable_hotset_fraction_spin.value())
        stage2_fraction = float(self.stage2_stable_hotset_fraction_spin.value())
        args.extend(["--stable-hotset-fraction", f"{stage1_fraction:g}"])  # legacy/shared fallback
        args.extend(["--stage1-stable-hotset-fraction", f"{stage1_fraction:g}"])
        args.extend(["--stage2-stable-hotset-fraction", f"{stage2_fraction:g}"])
        args.extend(["--stable-hotset-budget-gb", f"{self.stable_hotset_budget_gb_spin.value():g}"])

        emergency_floor = float(self.emergency_free_vram_floor_spin.value())
        args.extend(["--emergency-free-vram-floor-gb", f"{emergency_floor:g}"])

        stage2_block_limit = float(self.stage2_block_size_limit_spin.value())
        if pipeline_name in TWO_STAGE_PIPELINES and stage2_block_limit > 0:
            args.extend(["--stage2-block-size-limit-gb", f"{stage2_block_limit:g}"])
        attention_backend = self.attention_backend_combo.currentText().strip().lower() if hasattr(self, "attention_backend_combo") else "auto"
        if attention_backend == "pytorch":
            attention_backend = "sdpa"
        if attention_backend not in ("auto", "sdpa", "flash2", "sage"):
            attention_backend = "auto"
        args.extend(["--attention-backend", attention_backend])
        if hasattr(self, "normalize_input_image_check") and not self.normalize_input_image_check.isChecked():
            args.append("--no-ltx-normalize-input-image")
        if self.no_boundary_echo_check.isChecked():
            args.append("--no-boundary-echo")
        if self.deep_lifecycle_check.isChecked():
            args.extend([
                "--deep-log-interval", str(self.deep_interval_spin.value()),
                "--deep-log-max-events", str(self.deep_max_events_spin.value()),
                "--deep-log-path", deep_log_path,
                "--deep-lifecycle-log",
            ])
        if pipeline_name in ("two_stages", "two_stages_hq", "a2vid_two_stage"):
            # Distilled LoRA is intentionally not passed. The CLI makes the upstream
            # two-phase parser optional and runs Stage 2 without the official LoRA.
            if self.spatial_upsampler_row.text().strip():
                args.extend(["--spatial-upsampler-path", self.spatial_upsampler_row.text().strip()])
        if extra:
            args.append("--extra")
            args.extend(extra)

        return python_exe, args, output_path, extra


    def validate_before_run(self) -> bool:
        is_int4 = self._selected_model_variant() == "INT4"
        checks = [
            (self.python_row.text(), "Python executable is empty."),
            (str(INT4_CLI_PATH) if is_int4 else self.cli_row.text(), "LTX CLI path is empty."),
            (self.checkpoint_row.text(), "INT4 model folder is empty." if is_int4 else "Checkpoint path is empty."),
            ("not-needed" if is_int4 else self.gemma_row.text(), "Gemma root is empty."),
            (self.prompt_edit.toPlainText().strip(), "Prompt is empty."),
        ]
        for value, message in checks:
            if not str(value or "").strip():
                QMessageBox.warning(self, "LTX 2.3", message)
                return False
        cli_path = INT4_CLI_PATH if is_int4 else Path(self.cli_row.text())
        if not cli_path.exists():
            QMessageBox.warning(self, "LTX 2.3", f"CLI file not found:\n{cli_path}")
            return False
        if is_int4:
            model_root = Path(self.checkpoint_row.text().strip())
            required = [model_root / "model_index.json", model_root / "transformer" / "config.json", model_root / "text_encoder" / "config.json"]
            missing = [str(path) for path in required if not path.exists()]
            if not model_root.is_dir() or missing:
                detail = "\n".join(missing) if missing else str(model_root)
                QMessageBox.warning(self, "LTX 2.3 INT4", "INT4 model folder is incomplete:\n" + detail)
                return False

        audio_mode = AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED)
        audio_path = self.audio_row.text().strip()
        if audio_mode in (AUDIO_MODE_REMUX, AUDIO_MODE_A2V):
            if not audio_path:
                QMessageBox.warning(self, "LTX 2.3", "Select an audio file or set Audio mode to Disabled.")
                return False
            if not Path(audio_path).exists():
                QMessageBox.warning(self, "LTX 2.3", f"Audio file not found:\n{audio_path}")
                return False

        for video_path, label in (
            (getattr(self, "start_video_row", None).text().strip() if getattr(self, "start_video_row", None) else "", "Continue video"),
            (getattr(self, "end_video_row", None).text().strip() if getattr(self, "end_video_row", None) else "", "End with video"),
        ):
            if video_path and not Path(video_path).exists():
                QMessageBox.warning(self, "LTX 2.3", f"{label} file not found:\n{video_path}")
                return False
        if (getattr(self, "start_video_row", None) and self.start_video_row.text().strip()) or (getattr(self, "end_video_row", None) and self.end_video_row.text().strip()):
            ffmpeg = self._frame_ffmpeg_path()
            if ffmpeg != "ffmpeg" and not Path(ffmpeg).exists():
                QMessageBox.warning(self, "LTX 2.3", f"FFmpeg not found:\n{ffmpeg}")
                return False

        if not is_int4 and not self._ensure_auto_vram_settings_supported():
            return False

        pipeline_name = self._effective_pipeline()
        if pipeline_name in TWO_STAGE_PIPELINES:
            if not self._ensure_two_stage_assets(start_after_download=True, manual=False):
                return False
        return True


    def start_generation(self) -> None:
        if self.process is not None or self._download_worker is not None:
            return
        if not self.validate_before_run():
            return
        self._cleanup_old_latent_preview_folder_before_new_run()
        try:
            program, args, output_path, _extra = self.build_command(randomize_seed=True, prepare_video_inputs=True)
        except Exception as exc:
            QMessageBox.warning(self, "LTX 2.3", f"Could not prepare video input frame:\n{type(exc).__name__}: {exc}")
            return
        self.active_output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.report_row.text()).parent.mkdir(parents=True, exist_ok=True)
        Path(self.deep_log_row.text()).parent.mkdir(parents=True, exist_ok=True)

        self._save_settings(silent=True)
        self._last_command = [program] + args
        self._last_success = False

        if bool(getattr(self, "use_framevision_queue_check", None) and self.use_framevision_queue_check.isChecked()):
            if self._enqueue_framevision_queue_job():
                return
            self._append_log("FrameVision queue enqueue failed; falling back to direct LTX run.")

        self._append_log("\n" + "=" * 78)
        self._append_log("Starting LTX 2.3")
        self._append_log(self._quote_command(self._last_command))
        if bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
            self._reset_latent_preview_strip("Latent preview enabled. Waiting for sampling updates...")
            self._start_latent_preview_sidecar_polling(clear_existing=True)
        else:
            self._stop_latent_preview_sidecar_polling()
            self._reset_latent_preview_strip("Latent preview is off.")
        self.status_label.setText("Running LTX...")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setProgram(program)
        self.process.setArguments(args)
        self.process.setWorkingDirectory(self.ltx_root_row.text() or str(APP_ROOT))
        env = self.process.processEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        # Do not inject VRAM Lab defaults through environment variables here.
        # Direct Run should match the queue/minimal CLI path; explicit UI
        # overrides are passed as CLI args by _build_command().
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.generationStarted.emit(self._last_command)
        self.process.start()

    def _enqueue_framevision_queue_job(self) -> bool:
        """Add the current LTX command to the normal FrameVision Queue tab."""
        try:
            from helpers.queue_adapter import enqueue_ltx23_from_widget
        except Exception:
            try:
                from queue_adapter import enqueue_ltx23_from_widget  # type: ignore
            except Exception as exc:
                self._append_log(f"Could not import LTX queue adapter: {type(exc).__name__}: {exc}")
                return False
        try:
            ok = bool(enqueue_ltx23_from_widget(self))
        except Exception as exc:
            self._append_log(f"Could not enqueue LTX job: {type(exc).__name__}: {exc}")
            return False
        if ok:
            self._append_log("\n" + "=" * 78)
            self._append_log("LTX 2.3 job enqueued. Monitor progress in the Queue tab.")
            self._append_log(self._quote_command(self._last_command))
            self.status_label.setText("Queued in FrameVision")
            if bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
                self._reset_latent_preview_strip("Queued. Waiting for latent preview events...")
                self._start_latent_preview_sidecar_polling(clear_existing=True)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        return ok


    def stop_generation(self) -> None:
        for proc in (self.remux_process, self.process):
            if proc is not None:
                self._append_log("Stopping process...")
                proc.kill()
        self.status_label.setText("Stopping...")

    def _read_process_output(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip("\n"))
            self._update_status_from_output(data)
            self._update_latent_preview_from_output(data)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"QProcess error: {error}")


    def _ltx_glue_enabled(self) -> bool:
        try:
            if not (hasattr(self, "glue_input_videos_check") and self.glue_input_videos_check.isChecked()):
                return False
            start_video = self.start_video_row.text().strip() if hasattr(self, "start_video_row") else ""
            end_video = self.end_video_row.text().strip() if hasattr(self, "end_video_row") else ""
            return bool(start_video or end_video)
        except Exception:
            return False

    def _unique_glue_output_path(self, generated_path: Path) -> Path:
        base = generated_path.with_name(generated_path.stem + "_glued" + generated_path.suffix)
        if not base.exists():
            return base
        for index in range(1, 1000):
            candidate = generated_path.with_name(f"{generated_path.stem}_glued_{index:03d}{generated_path.suffix}")
            if not candidate.exists():
                return candidate
        return generated_path.with_name(f"{generated_path.stem}_glued_{int(time.time())}{generated_path.suffix}")

    def _probe_video_geometry_for_glue(self, video_path: Path) -> Tuple[int, int, str]:
        ffprobe = self._frame_ffprobe_path()
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                    "-of", "json",
                    str(video_path),
                ],
                cwd=str(APP_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
            if completed.returncode == 0:
                data = json.loads(completed.stdout or "{}")
                stream = (data.get("streams") or [{}])[0]
                width = int(stream.get("width") or self.width_spin.value())
                height = int(stream.get("height") or self.height_spin.value())
                fps = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "").strip()
                if not fps or fps in {"0/0", "0", "N/A"}:
                    fps = str(int(self.fps_spin.value()))
                return width, height, fps
        except Exception:
            pass
        return int(self.width_spin.value()), int(self.height_spin.value()), str(int(self.fps_spin.value()))

    def _write_concat_list_for_glue(self, videos: List[Path], list_path: Path) -> None:
        def _escape(path: Path) -> str:
            return str(path).replace("\\", "/").replace("'", "'\\''")
        list_path.write_text("".join(f"file '{_escape(video)}'\n" for video in videos), encoding="utf-8")

    def _run_glue_command(self, args: List[str], label: str) -> bool:
        self._append_log(label)
        self._append_log(self._quote_command(args))
        completed = subprocess.run(
            args,
            cwd=str(APP_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=None,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
        if completed.stdout:
            self._append_log(completed.stdout.rstrip())
        return int(completed.returncode) == 0

    def _glue_videos_after_generation(self, generated_path: Path) -> Path:
        if not self._ltx_glue_enabled():
            return generated_path
        start_video = Path(self.start_video_row.text().strip()).expanduser() if hasattr(self, "start_video_row") and self.start_video_row.text().strip() else None
        end_video = Path(self.end_video_row.text().strip()).expanduser() if hasattr(self, "end_video_row") and self.end_video_row.text().strip() else None
        videos: List[Path] = []
        for item in (start_video, generated_path, end_video):
            if item and item.exists():
                videos.append(item)
        if len(videos) < 2:
            return generated_path

        ffmpeg = self._frame_ffmpeg_path()
        final_path = self._unique_glue_output_path(generated_path)
        concat_list = final_path.with_suffix(".concat.txt")
        try:
            self._write_concat_list_for_glue(videos, concat_list)
            copy_args = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(final_path),
            ]
            if self._run_glue_command(copy_args, "Gluing videos with stream copy...") and final_path.exists() and final_path.stat().st_size > 4096:
                self._append_log(f"Glued video ready: {final_path}")
                return final_path

            try:
                final_path.unlink(missing_ok=True)
            except Exception:
                pass
            width, height, fps = self._probe_video_geometry_for_glue(generated_path)
            inputs: List[str] = []
            filters: List[str] = []
            concat_labels: List[str] = []
            for index, video in enumerate(videos):
                inputs.extend(["-i", str(video)])
                label = f"v{index}"
                filters.append(f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[{label}]")
                concat_labels.append(f"[{label}]")
            filter_complex = ";".join(filters) + ";" + "".join(concat_labels) + f"concat=n={len(videos)}:v=1:a=0[outv]"
            reencode_args = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(final_path),
            ]
            if self._run_glue_command(reencode_args, "Stream copy failed; gluing videos with safe re-encode...") and final_path.exists() and final_path.stat().st_size > 4096:
                self._append_log(f"Glued video ready: {final_path}")
                return final_path
            self._append_log("Video glue failed; keeping original LTX output.")
            return generated_path
        except Exception as exc:
            self._append_log(f"Video glue skipped: {type(exc).__name__}: {exc}")
            return generated_path
        finally:
            try:
                concat_list.unlink(missing_ok=True)
            except Exception:
                pass


    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        ok = exit_status == QProcess.NormalExit and exit_code == 0
        self._append_log(f"LTX process finished: exit_code={exit_code}, status={exit_status.name}")
        self.process = None
        self._last_success = ok
        if ok and self.active_output_path and self.active_output_path.exists():
            self.active_output_path = self._glue_videos_after_generation(self.active_output_path)
            self._cleanup_prepared_video_frames(success_only=True)
            self.outputReady.emit(str(self.active_output_path))
            if self._should_remux_audio():
                self._start_audio_remux(self.active_output_path)
                return
        self._finish_all(ok, str(self.active_output_path or ""))

    def _should_remux_audio(self) -> bool:
        mode = AUDIO_MODE_COMPAT_MAP.get(self.audio_mode_combo.currentText(), AUDIO_MODE_DISABLED)
        return mode == AUDIO_MODE_REMUX and bool(self.audio_row.text().strip())



    def _start_audio_remux(self, video_path: Path) -> None:
        audio_path = Path(self.audio_row.text())
        if not audio_path.exists():
            self._append_log(f"Audio remux skipped. Audio file not found: {audio_path}")
            self._finish_all(True, str(video_path))
            return
        ffmpeg = self.ffmpeg_row.text().strip() or "ffmpeg"
        if self.remux_replace_check.isChecked():
            remux_path = video_path.with_name(video_path.stem + "_with_audio_tmp" + video_path.suffix)
        else:
            remux_path = video_path.with_name(video_path.stem + "_with_audio" + video_path.suffix)
        args = [
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", self.remux_bitrate_combo.currentText(),
        ]
        if self.remux_shortest_check.isChecked():
            args.append("-shortest")
        args.append(str(remux_path))
        self._append_log("Starting audio remux")
        self._append_log(self._quote_command([ffmpeg] + args))
        self.status_label.setText("Adding audio...")
        self.remux_process = QProcess(self)
        self.remux_process.setProcessChannelMode(QProcess.MergedChannels)
        self.remux_process.setProgram(ffmpeg)
        self.remux_process.setArguments(args)
        self.remux_process.readyReadStandardOutput.connect(self._read_remux_output)
        self.remux_process.finished.connect(lambda code, status: self._remux_finished(code, status, video_path, remux_path))
        self.remux_process.start()

    def _read_remux_output(self) -> None:
        if not self.remux_process:
            return
        data = bytes(self.remux_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip("\n"))

    def _remux_finished(self, exit_code: int, exit_status: QProcess.ExitStatus, original: Path, remuxed: Path) -> None:
        ok = exit_status == QProcess.NormalExit and exit_code == 0 and remuxed.exists()
        self._append_log(f"Audio remux finished: exit_code={exit_code}, status={exit_status.name}")
        self.remux_process = None
        final_path = remuxed
        if ok and self.remux_replace_check.isChecked():
            try:
                backup = original.with_name(original.stem + "_video_only" + original.suffix)
                if backup.exists():
                    backup.unlink()
                original.rename(backup)
                remuxed.rename(original)
                final_path = original
                self._append_log(f"Replaced output. Video-only backup: {backup}")
            except Exception as exc:
                ok = False
                self._append_log(f"Could not replace original output: {type(exc).__name__}: {exc}")
        if ok:
            self.active_output_path = final_path
            self.outputReady.emit(str(final_path))
        self._finish_all(ok, str(final_path if ok else original))

    def _finish_all(self, ok: bool, output_path: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Done" if ok else "Failed")
        if ok:
            self._append_log(f"Output: {output_path}")
            if self.open_when_done_check.isChecked():
                self.open_output_folder()
        if ok or not bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
            self._stop_latent_preview_sidecar_polling()
        self.generationFinished.emit(ok, output_path)

    def _update_latent_preview_controls(self) -> None:
        enabled = bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked())
        for widget in (
            getattr(self, "latent_preview_mode_combo", None),
            getattr(self, "latent_preview_rate_spin", None),
            getattr(self, "latent_preview_upscale_check", None),
            getattr(self, "latent_preview_tae_decode_check", None),
            getattr(self, "keep_latents_check", None),
            getattr(self, "latent_preview_strip_scroll", None),
        ):
            if widget is not None:
                try:
                    widget.setEnabled(enabled)
                except Exception:
                    pass
        if not enabled:
            self._stop_latent_preview_sidecar_polling()
            self._reset_latent_preview_strip("Latent preview is off.")
        elif not getattr(self, "_latent_preview_cards", []):
            self._reset_latent_preview_strip("Latent preview enabled. Run LTX to fill this strip.")

    def _reset_latent_preview_strip(self, status: str = "") -> None:
        try:
            layout = getattr(self, "latent_preview_strip_layout", None)
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                placeholder = QLabel("No latent frames yet")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setMinimumSize(160, 72)
                placeholder.setStyleSheet("border: 1px dashed rgba(180, 190, 210, 120); border-radius: 8px; padding: 8px;")
                layout.addWidget(placeholder)
                layout.addStretch(1)
            self._latent_preview_cards = []
            self._latent_preview_sidecar_offset = 0
            if not str(getattr(self, "_latent_preview_sidecar_path", "") or "").strip():
                self._latent_preview_dir_path = ""
            if getattr(self, "latent_preview_status_label", None) is not None:
                self.latent_preview_status_label.setText(status or "Waiting for latent preview frames...")
        except Exception:
            pass

    def _clear_latent_preview_placeholder(self) -> None:
        try:
            layout = getattr(self, "latent_preview_strip_layout", None)
            if layout is None:
                return
            if getattr(self, "_latent_preview_cards", []):
                return
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        except Exception:
            pass

    def _add_latent_preview_card(self, title: str, subtitle: str = "", image_path: str = "") -> None:
        if not bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
            return
        try:
            self._clear_latent_preview_placeholder()
            card = QLabel()
            card.setAlignment(Qt.AlignCenter)
            card.setMinimumSize(120, 72)
            card.setMaximumWidth(180)
            card.setWordWrap(True)
            shown_image = False
            if image_path:
                pix = QPixmap(str(image_path))
                if not pix.isNull():
                    card.setPixmap(pix.scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    card.setToolTip(f"Click to open larger preview\n{title}\n{subtitle}\n{image_path}".strip())
                    card.setCursor(Qt.PointingHandCursor)
                    preview_path = str(image_path)
                    preview_title = str(title)
                    preview_subtitle = str(subtitle)
                    card.mousePressEvent = lambda event, p=preview_path, t=preview_title, st=preview_subtitle: self._open_latent_preview_image(p, t, st)
                    try:
                        self._latent_preview_last_dir = str(Path(preview_path).parent)
                    except Exception:
                        pass
                    shown_image = True
            if not shown_image:
                card.setText((title + ("\n" + subtitle if subtitle else "")).strip())
                card.setToolTip((title + ("\n" + subtitle if subtitle else "")).strip())
            card.setStyleSheet("border: 1px solid rgba(180, 190, 210, 120); border-radius: 8px; padding: 6px;")
            layout = self.latent_preview_strip_layout
            insert_at = max(0, layout.count() - 1)
            layout.insertWidget(insert_at, card)
            self._latent_preview_cards.append(card)
            while len(self._latent_preview_cards) > 12:
                old = self._latent_preview_cards.pop(0)
                old.deleteLater()
            if getattr(self, "latent_preview_status_label", None) is not None:
                self.latent_preview_status_label.setText(subtitle or title)
            try:
                self.latent_preview_strip_scroll.horizontalScrollBar().setValue(self.latent_preview_strip_scroll.horizontalScrollBar().maximum())
            except Exception:
                pass
        except Exception as exc:
            try:
                self.latent_preview_status_label.setText(f"Latent preview strip update failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _open_latent_preview_image(self, image_path: str, title: str = "Latent preview", subtitle: str = "") -> None:
        """Open a larger in-app view of a latent preview contact sheet."""
        try:
            path = Path(str(image_path or ""))
            if not path.exists():
                if getattr(self, "latent_preview_status_label", None) is not None:
                    self.latent_preview_status_label.setText(f"Latent preview file not found: {path}")
                return
            pix = QPixmap(str(path))
            if pix.isNull():
                if getattr(self, "latent_preview_status_label", None) is not None:
                    self.latent_preview_status_label.setText(f"Could not open latent preview: {path}")
                return

            dlg = QDialog(self)
            dlg.setWindowTitle(str(title or "Latent preview"))
            dlg.resize(980, 620)
            layout = QVBoxLayout(dlg)

            header = QLabel((str(title or "Latent preview") + (" — " + str(subtitle) if subtitle else "")).strip())
            header.setWordWrap(True)
            layout.addWidget(header)

            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setBackgroundRole(self.backgroundRole())
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.StyledPanel)
            scroll.setWidget(image_label)
            layout.addWidget(scroll, 1)

            footer = QHBoxLayout()
            info = QLabel(str(path))
            info.setWordWrap(True)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.close)
            footer.addWidget(info, 1)
            footer.addWidget(close_btn)
            layout.addLayout(footer)

            def _rescale() -> None:
                try:
                    area = scroll.viewport().size()
                    # Show a zoomed-in version, but keep it inside the available view.
                    target_w = max(640, area.width() - 24)
                    target_h = max(360, area.height() - 24)
                    image_label.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception:
                    image_label.setPixmap(pix)

            _rescale()
            dlg.show()
            QTimer.singleShot(100, _rescale)
        except Exception as exc:
            try:
                if getattr(self, "latent_preview_status_label", None) is not None:
                    self.latent_preview_status_label.setText(f"Open latent preview failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _current_latent_preview_folder_candidates(self) -> List[Path]:
        candidates: List[Path] = []
        for raw in (
            getattr(self, "_latent_preview_dir_path", ""),
            getattr(self, "_latent_preview_last_dir", ""),
        ):
            raw_text = str(raw or "").strip()
            if raw_text:
                try:
                    candidates.append(Path(raw_text))
                except Exception:
                    pass
        sidecar_text = str(getattr(self, "_latent_preview_sidecar_path", "") or "").strip()
        if sidecar_text:
            try:
                candidates.append(Path(sidecar_text).with_suffix(""))
            except Exception:
                pass
        unique: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _cleanup_old_latent_preview_folder_before_new_run(self) -> None:
        keep_latents = bool(getattr(self, "keep_latents_check", None) and self.keep_latents_check.isChecked())
        if keep_latents:
            return
        deleted_any = False
        for folder in self._current_latent_preview_folder_candidates():
            try:
                if folder.exists() and folder.is_dir():
                    shutil.rmtree(folder, ignore_errors=False)
                    deleted_any = True
            except Exception as exc:
                try:
                    self._append_log(f"Could not remove old latent preview folder: {folder} ({type(exc).__name__}: {exc})")
                except Exception:
                    pass
        self._latent_preview_last_dir = ""
        if deleted_any:
            try:
                self._append_log("Removed previous latent preview folder because Keep latents is off.")
            except Exception:
                pass

    def _start_latent_preview_sidecar_polling(self, clear_existing: bool = False) -> None:
        try:
            path = str(getattr(self, "_latent_preview_sidecar_path", "") or "").strip()
            if not path:
                return
            self._latent_preview_sidecar_offset = 0
            if clear_existing:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            timer = getattr(self, "_latent_preview_poll_timer", None)
            if timer is not None and not timer.isActive():
                timer.start()
        except Exception:
            pass

    def _stop_latent_preview_sidecar_polling(self) -> None:
        try:
            timer = getattr(self, "_latent_preview_poll_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
        except Exception:
            pass

    def _handle_latent_preview_event(self, event: Dict[str, Any]) -> None:
        try:
            kind = str(event.get("kind", "") or "").strip().lower()
            if kind == "step":
                # Step telemetry is handled by logs/progress bars. Latent Preview should only show real image previews.
                return
            elif kind == "image":
                step = str(event.get("step", ""))
                total = str(event.get("total", ""))
                stage = str(event.get("stage", "") or "").strip()
                frame = str(event.get("frame", event.get("index", "")))
                title = f"{stage} preview".strip() if stage else "Latent preview"
                if frame:
                    title += f" {frame}"
                subtitle = f"Step {step}/{total}" if step and total else (f"Step {step}" if step else str(event.get("message", "")))
                self._add_latent_preview_card(title, subtitle, str(event.get("path", "") or ""))
            elif kind == "status":
                msg = str(event.get("message", "") or "").strip()
                if msg and getattr(self, "latent_preview_status_label", None) is not None:
                    self.latent_preview_status_label.setText("Latent preview: " + msg)
            elif kind == "done":
                msg = str(event.get("message", "Done") or "Done")
                if getattr(self, "latent_preview_status_label", None) is not None:
                    self.latent_preview_status_label.setText("Latent preview: " + msg)
                self._stop_latent_preview_sidecar_polling()
        except Exception:
            pass

    def _poll_latent_preview_sidecar(self) -> None:
        if not bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
            self._stop_latent_preview_sidecar_polling()
            return
        try:
            path_text = str(getattr(self, "_latent_preview_sidecar_path", "") or "").strip()
            if not path_text:
                return
            path = Path(path_text)
            if not path.exists():
                return
            size = path.stat().st_size
            offset = int(getattr(self, "_latent_preview_sidecar_offset", 0) or 0)
            if size < offset:
                offset = 0
            if size == offset:
                return
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                self._latent_preview_sidecar_offset = f.tell()
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                self._handle_latent_preview_event(event)
        except Exception:
            pass

    def _update_latent_preview_from_output(self, text: str) -> None:
        if not bool(getattr(self, "latent_preview_enabled_check", None) and self.latent_preview_enabled_check.isChecked()):
            return
        try:
            for line in str(text or "").splitlines():
                if "[framevision-latent-preview]" in line:
                    payload = line.split("]", 1)[-1].strip()
                    path_match = re.search(r"(?:path|file)=([^|]+?)(?:\s+\w+=|$)", payload)
                    step_match = re.search(r"step\s*=\s*(\d+)", payload)
                    idx_match = re.search(r"(?:index|frame)\s*=\s*(\d+)", payload)
                    image_path = path_match.group(1).strip().strip('"') if path_match else ""
                    title = "Latent frame"
                    if idx_match:
                        title += f" {idx_match.group(1)}"
                    subtitle = f"Step {step_match.group(1)}" if step_match else payload
                    self._add_latent_preview_card(title, subtitle, image_path)
            status_matches = re.findall(r"^\[ltx-status\]\s*(.+?)\s*$", text, flags=re.MULTILINE)
            if status_matches and getattr(self, "latent_preview_status_label", None) is not None:
                self.latent_preview_status_label.setText("Latent preview: " + status_matches[-1].strip())
        except Exception:
            pass

    def _update_status_from_output(self, text: str) -> None:
        status_matches = re.findall(r"^\[ltx-status\]\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if status_matches:
            self.status_label.setText(status_matches[-1].strip())
            return
        if "error" in text.lower() or "traceback" in text.lower():
            self.status_label.setText("Running — check log")
            return
        percent = re.findall(r"(\d{1,3})%", text)
        if percent:
            self.status_label.setText(f"Running {percent[-1]}%")
        elif "step" in text.lower():
            self.status_label.setText("Running steps...")

    # ------------------------------------------------------------------
    # Settings / log / utility actions
    # ------------------------------------------------------------------
    def _connect_auto_save(self) -> None:
        widgets: List[QWidget] = [
            self.pipeline_combo, self.vram_lab_combo, self.vram_profile_combo, self.resolution_combo,
            self.main_hot_window_spin, self.stage2_block_size_limit_spin, self.width_spin, self.height_spin, self.frames_spin, self.fps_spin, self.steps_spin, self.seed_spin,
            self.random_seed_check, self.snap_resolution_check, self.no_boundary_echo_check,
            self.deep_lifecycle_check, self.deep_interval_spin, self.deep_max_events_spin,
            self.distilled_lora_strength_spin, self.auto_download_two_stage_check,
            *getattr(self, "user_lora_strength_spins", []), *getattr(self, "user_lora_strength_sliders", []),
            self.lora_cache_mode_combo, self.lora_cache_miss_inplace_check, self.lora_cache_shard_gb_spin,
            self.lora_cache_shard_threshold_spin, self.lora_cache_max_spin, self.audio_mode_combo,
            self.audio_start_spin, self.audio_max_duration_spin,
            self.remux_bitrate_combo, self.remux_shortest_check, self.remux_replace_check,
            self.open_when_done_check, self.start_image_frame_spin, self.start_image_strength_spin,
            self.end_image_strength_spin, self.reference_image_strength_spin, self.normalize_input_image_check, self.video_cfg_spin, self.shift_spin, self.shift_slider,
            self.video_stg_spin, self.video_rescale_spin, self.audio_cfg_spin, self.audio_stg_spin,
            self.audio_rescale_spin, self.a2v_guidance_spin, self.v2a_guidance_spin,
            self.video_skip_step_spin, self.audio_skip_step_spin, self.max_batch_size_combo,
            self.enhance_prompt_check, self.attention_backend_combo, self.flash_attention_check, self.quantization_combo, self.main_transformer_stream_probe_check,
            self.planned_hotset_check, self.fast_iclora_route_check, self.stage1_stable_hotset_fraction_spin, self.stage2_stable_hotset_fraction_spin, self.stable_hotset_budget_gb_spin, self.emergency_free_vram_floor_spin,
            self.disable_distilled_lora_check,
            self.use_framevision_queue_check,
            self.latent_preview_enabled_check, self.latent_preview_mode_combo, self.latent_preview_rate_spin,
            self.latent_preview_upscale_check, self.latent_preview_tae_decode_check, self.keep_latents_check,
            self.model_variant_combo,
        ]
        for w in widgets:
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._settings_changed)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox, QSlider)):
                w.valueChanged.connect(self._settings_changed)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(self._settings_changed)
        for edit in [
            self.prompt_edit, self.negative_edit, self.extra_args_edit,
        ]:
            edit.textChanged.connect(self._settings_changed)
        for line in [
            self.reference_images_edit, self.output_name_edit, self.disable_lora_extra_args_edit,
        ]:
            line.textChanged.connect(self._settings_changed)
        for row in self._path_rows().values():
            row.changed.connect(self._settings_changed)
        self.pipeline_combo.currentTextChanged.connect(lambda *_: self._update_pipeline_dependent_controls())
        self.ltx_root_row.changed.connect(lambda *_: self._update_two_stage_asset_status())
        self.ltx_root_row.changed.connect(lambda *_: self._refresh_lora_cache_status())
        self.distilled_lora_row.changed.connect(lambda *_: self._update_two_stage_asset_status())
        self.spatial_upsampler_row.changed.connect(lambda *_: self._update_two_stage_asset_status())
        self.python_row.changed.connect(lambda *_: self._update_flash_attention_status())
        self.checkpoint_row.changed.connect(lambda *_: self._sync_model_variant_from_checkpoint())
        self.model_variant_combo.currentTextChanged.connect(lambda *_: self._model_variant_changed())
        self.distilled_lora_strength_spin.valueChanged.connect(lambda *_: self._update_test_lora_strength_ui())
        self.lora_cache_mode_combo.currentTextChanged.connect(lambda *_: self._update_test_lora_strength_ui())
        if hasattr(self, "disable_distilled_lora_check"):
            self.disable_distilled_lora_check.toggled.connect(lambda *_: self._update_test_lora_strength_ui())
            self.disable_distilled_lora_check.toggled.connect(lambda *_: self._update_two_stage_asset_status())
        try:
            self.fast_iclora_route_check.toggled.connect(self._remember_fast_iclora_route_preference)
        except Exception:
            pass
        try:
            self.latent_preview_enabled_check.toggled.connect(lambda *_: self._update_latent_preview_controls())
        except Exception:
            pass


    def _probe_attention_backends_available(self) -> Tuple[bool, str]:
        """Check the selected LTX Python env for optional attention packages.

        The selector itself stays enabled even when optional packages are missing;
        the CLI falls back safely. This status text is only a convenience check.
        """
        try:
            candidates: List[Path] = []
            py_text = self.python_row.text().strip() if hasattr(self, "python_row") else ""
            if py_text:
                candidates.append(Path(py_text))

            ltx_root_text = self.ltx_root_row.text().strip() if hasattr(self, "ltx_root_row") else ""
            if ltx_root_text:
                root = Path(ltx_root_text)
                candidates.extend([
                    root / "environments" / ".ltx23" / "python.exe",
                    root / "environments" / ".ltx23" / "Scripts" / "python.exe",
                    root / ".ltx23" / "python.exe",
                    root / ".ltx23" / "Scripts" / "python.exe",
                ])

            candidates.extend([
                DEFAULT_PYTHON,
                DEFAULT_LTX_ROOT / "environments" / ".ltx23" / "Scripts" / "python.exe",
            ])

            seen: set[str] = set()
            unique_candidates: List[Path] = []
            for candidate in candidates:
                key = str(candidate).lower()
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(candidate)

            code = (
                "import importlib, sys, torch\n"
                "print('python=' + sys.executable)\n"
                "print('torch=' + str(getattr(torch, '__version__', '?')) + ' cuda=' + str(getattr(torch.version, 'cuda', '?')))\n"
                "for name in ('sageattention', 'flash_attn'):\n"
                "    try:\n"
                "        m = importlib.import_module(name)\n"
                "        print(name + '=OK ' + str(getattr(m, '__version__', 'version unavailable')))\n"
                "    except Exception as exc:\n"
                "        print(name + '=MISSING ' + type(exc).__name__ + ': ' + str(exc))\n"
            )

            missing: List[str] = []
            failures: List[str] = []
            probe_cwd = APP_ROOT
            if ltx_root_text:
                try:
                    probe_cwd = Path(ltx_root_text).resolve()
                except Exception:
                    probe_cwd = APP_ROOT

            for py in unique_candidates:
                if not py.exists():
                    missing.append(str(py))
                    continue

                env = dict(os.environ)
                env["PYTHONNOUSERSITE"] = "1"
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["PATH"] = str(py.parent) + os.pathsep + str(py.parent / "Scripts") + os.pathsep + env.get("PATH", "")

                try:
                    completed = subprocess.run(
                        [str(py), "-c", code],
                        cwd=str(probe_cwd),
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    failures.append(f"{py}: timed out")
                    continue

                output_lines = (completed.stdout or completed.stderr or "").strip().splitlines()
                msg = " | ".join(line.strip() for line in output_lines if line.strip()) or "no output"
                if completed.returncode == 0:
                    return True, msg
                failures.append(f"{py}: {msg}")

            if failures:
                return False, "Attention backend probe failed: " + " ; ".join(failures[-3:])
            return False, "No usable LTX Python found. Checked: " + " ; ".join(missing[:5])
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _probe_flash_attention_available(self) -> Tuple[bool, str]:
        # Compatibility wrapper for older internal calls.
        ok, detail = self._probe_attention_backends_available()
        return ok and "flash_attn=OK" in detail, detail

    def _update_flash_attention_status(self) -> None:
        if not hasattr(self, "attention_backend_combo") or not hasattr(self, "flash_attention_status_label"):
            return

        ok, detail = self._probe_attention_backends_available()
        selected = self.attention_backend_combo.currentText().strip().lower()

        # Keep the hidden legacy checkbox synchronized for old save/load code.
        if hasattr(self, "flash_attention_check"):
            self.flash_attention_check.blockSignals(True)
            try:
                self.flash_attention_check.setChecked(selected == "flash2")
                self.flash_attention_check.setEnabled(True)
            finally:
                self.flash_attention_check.blockSignals(False)

        if not ok:
            self.flash_attention_status_label.setText(
                "Attention backend probe could not run. The CLI still uses safe fallback; check the final report."
            )
            self.flash_attention_status_label.setToolTip(detail)
        else:
            sage_ok = "sageattention=OK" in detail
            flash_ok = "flash_attn=OK" in detail
            parts = []
            parts.append("SageAttention found" if sage_ok else "SageAttention not found")
            parts.append("FlashAttention found" if flash_ok else "FlashAttention not found")
            parts.append("Auto tries: sage → flash2 → sdpa.")
            self.flash_attention_status_label.setText(" | ".join(parts))
            self.flash_attention_status_label.setToolTip(
                detail + "\n\n"
                "Backend selector values:\n"
                "auto = try SageAttention, then FlashAttention2, then PyTorch SDPA\n"
                "sdpa = force PyTorch SDPA\n"
                "flash2 = try FlashAttention2 with per-call fallback\n"
                "sage = try SageAttention with per-call fallback\n"
                "The final LTX report confirms success/fallback counts."
            )

        self._refresh_command_preview()


    def _portable_ltx_root_for_autofill(self) -> Path:
        """Pick the most likely portable LTX root for the model auto-fill button."""
        candidates: List[Path] = []
        try:
            text = self.ltx_root_row.text().strip() if hasattr(self, "ltx_root_row") else ""
            if text:
                candidates.append(Path(text).expanduser())
        except Exception:
            pass
        candidates.append(APP_ROOT)
        candidates.append(DEFAULT_LTX_ROOT)

        for candidate in candidates:
            try:
                root = candidate.resolve()
            except Exception:
                root = candidate
            if (
                (root / "models" / "ltx23").exists()
                or (root / "environments" / ".ltx23").exists()
                or root == APP_ROOT
            ):
                return root
        return APP_ROOT

    def _checkpoint_for_variant(self, root: Path, variant: str) -> Path:
        token = str(variant or "FP16").strip().upper()
        if token == "INT4":
            return root / INT4_MODEL_RELATIVE
        rel = FP8_CHECKPOINT_RELATIVE if token == "FP8" else FP16_CHECKPOINT_RELATIVE
        return root / rel

    def _infer_model_variant_from_checkpoint(self, value: Any) -> str:
        text = str(value or "").strip().replace("\\", "/").rstrip("/").lower()
        folder_name = text.rsplit("/", 1)[-1] if text else ""
        # The installed split-folder model is models/ltx23_int4. Keep the old
        # ltx_int4 spelling recognized too so stale settings cannot accidentally
        # send a quant folder into the native FP16/FP8 CLI.
        if folder_name in {"ltx23_int4", "ltx_int4"} or folder_name.endswith("_int4"):
            return "INT4"
        return "FP8" if "fp8" in text else "FP16"

    def _selected_model_variant(self) -> str:
        # Safety first: a recognized INT4 split-model folder must always use the
        # isolated planner_ltx_int4.py, even when an older settings file restored the
        # combo as FP16. This prevents the native VRAM Lab CLI from ever receiving
        # the INT4 directory as --checkpoint-path.
        inferred = self._infer_model_variant_from_checkpoint(
            self.checkpoint_row.text() if hasattr(self, "checkpoint_row") else ""
        )
        if inferred == "INT4":
            return "INT4"
        if hasattr(self, "model_variant_combo"):
            token = self.model_variant_combo.currentText().strip().upper()
            if token in {"FP16", "FP8", "INT4"}:
                return token
        return inferred

    def _set_vram_lab_tab_visible(self, visible: bool) -> None:
        """Show native VRAM Lab tuning only for FP16/FP8 models."""
        if not hasattr(self, "tabs"):
            return
        target = getattr(self, "vram_lab_tab", None)
        index = self.tabs.indexOf(target) if target is not None else -1
        if index < 0:
            for candidate in range(self.tabs.count()):
                if self.tabs.tabText(candidate).strip().lower() == "vram lab":
                    index = candidate
                    break
        if index < 0:
            return
        try:
            self.tabs.setTabVisible(index, bool(visible))
        except AttributeError:
            # PySide6 builds used by FrameVision expose setTabVisible. Keep a
            # harmless fallback for older Qt bindings.
            self.tabs.setTabEnabled(index, bool(visible))
        if not visible and self.tabs.currentIndex() == index:
            self.tabs.setCurrentIndex(0)

    def _apply_model_variant_route(self, variant: str = "", *, update_checkpoint: bool = False) -> None:
        token = str(variant or self._selected_model_variant()).strip().upper() or "FP16"
        is_int4 = token == "INT4"
        if hasattr(self, "cli_row"):
            self.cli_row.setText(str(INT4_CLI_PATH if is_int4 else DEFAULT_CLI_PATH))
        if hasattr(self, "checkpoint_row"):
            self.checkpoint_row.mode = "dir" if is_int4 else "file"
            self.checkpoint_row.label.setText("INT4 model folder" if is_int4 else "Checkpoint")
            self.checkpoint_row.file_filter = "All files (*.*)" if is_int4 else "Safetensors (*.safetensors);;All files (*.*)"
            self._set_tooltip(
                self.checkpoint_row,
                "Local split-folder INT4 model, normally models\\ltx23_int4." if is_int4
                else "Main LTX 2.3 checkpoint / safetensors file. This is the large model file.",
            )
            if update_checkpoint:
                root = self._portable_ltx_root_for_autofill()
                self.checkpoint_row.setText(str(self._checkpoint_for_variant(root, token)))
        if hasattr(self, "vram_lab_combo"):
            # Keep the simple ON/OFF control available for every model. For
            # INT4, ON enables automatic card detection plus workload-aware
            # Stage 1/Stage 2 residency in planner_ltx_int4.py. Native VRAM Lab
            # tuning remains exclusive to FP16/FP8.
            self.vram_lab_combo.setEnabled(True)
            self._set_tooltip(
                self.vram_lab_combo,
                (
                    "INT4 automatic VRAM planner. ON detects the installed GPU as a 12/16/24 GB profile and automatically scales Stage 1, Stage 2 and final decode from resolution and frame count. OFF uses a fixed per-card INT4 policy."
                    if is_int4 else
                    "Simple VRAM Lab toggle. ON uses the safe VRAM Lab mode. OFF disables VRAM Lab and runs without that protection layer."
                ),
            )
        self._set_vram_lab_tab_visible(not is_int4)

    def _sync_model_variant_from_checkpoint(self) -> None:
        if self._loading or not hasattr(self, "model_variant_combo"):
            return
        variant = self._infer_model_variant_from_checkpoint(self.checkpoint_row.text())
        if self.model_variant_combo.currentText() != variant:
            old_loading = self._loading
            self._loading = True
            try:
                self._set_combo(self.model_variant_combo, variant)
            finally:
                self._loading = old_loading
        self._apply_model_variant_route(variant, update_checkpoint=False)
        self._apply_quantization_availability(variant)

    def _model_variant_changed(self) -> None:
        if self._loading or not hasattr(self, "model_variant_combo"):
            return
        variant = self.model_variant_combo.currentText().strip().upper() or "FP16"
        # Model routing is strict: native FP16/FP8 always use the untouched native
        # CLI; INT4 always uses the separate quant CLI and split model folder.
        try:
            self._apply_model_variant_route(variant, update_checkpoint=True)
        except Exception:
            pass
        if hasattr(self, "quantization_combo"):
            # Keep every variant safe by default. FP8 quant modes are manual tests,
            # not automatic defaults.
            self._set_combo(self.quantization_combo, QUANTIZATION_MODE_NONE)
            self._apply_quantization_availability(variant)
        self._refresh_command_preview()
        self._save_settings(silent=True)

    def _apply_quantization_availability(self, variant: str = "") -> None:
        if not hasattr(self, "quantization_combo"):
            return
        token = (variant or (self.model_variant_combo.currentText() if hasattr(self, "model_variant_combo") else "FP16")).strip().upper()
        is_fp8 = token == "FP8"
        is_int4 = token == "INT4"

        self.quantization_combo.blockSignals(True)
        try:
            if is_int4:
                self._set_combo(self.quantization_combo, QUANTIZATION_MODE_NONE)
                self.quantization_combo.setEnabled(False)
                self._set_tooltip(
                    self.quantization_combo,
                    "INT4 is already pre-quantized and runs only through the isolated planner_ltx_int4.py.",
                )
            elif not is_fp8:
                self._set_combo(self.quantization_combo, QUANTIZATION_MODE_NONE)
                self.quantization_combo.setEnabled(False)
                self._set_tooltip(
                    self.quantization_combo,
                    "Quantization is off for FP16 because the current FP16 FrameVision/LTX path is the proven working path.",
                )
            else:
                self.quantization_combo.setEnabled(True)
                self._set_tooltip(
                    self.quantization_combo,
                    "FP8-only experimental setting. None/Auto pass no quantization arg. fp8-cast/fp8-scaled-mm are manual test modes only.",
                )
        finally:
            self.quantization_combo.blockSignals(False)

    def _auto_fill_selected_model_paths(self) -> None:
        variant = self.model_variant_combo.currentText().strip().upper() if hasattr(self, "model_variant_combo") else "FP16"
        root = self._portable_ltx_root_for_autofill()
        python_candidates = [
            root / "environments" / ".ltx23" / "python.exe",
            root / "environments" / ".ltx23" / "Scripts" / "python.exe",
        ]
        python_path = next((p for p in python_candidates if p.exists()), python_candidates[0])
        ffmpeg_path = DEFAULT_FRAME_FFMPEG if DEFAULT_FRAME_FFMPEG.exists() else Path("ffmpeg")

        self.ltx_root_row.setText(str(root))
        self.python_row.setText(str(python_path))
        self.cli_row.setText(str(INT4_CLI_PATH if variant == "INT4" else DEFAULT_CLI_PATH))
        self.checkpoint_row.setText(str(self._checkpoint_for_variant(root, variant)))
        self._apply_model_variant_route(variant, update_checkpoint=False)
        if hasattr(self, "quantization_combo"):
            # Safe default for both variants. FP8 quantization modes are explicit
            # manual tests only; do not turn them on from Auto Fill.
            self._set_combo(self.quantization_combo, QUANTIZATION_MODE_NONE)
            self._apply_quantization_availability(variant)
        self.gemma_row.setText(str(root / "models" / "ltx23" / "text_encoder" / "lightricks_gemma_original"))
        self.output_dir_row.setText(str(root / "output" / "ltx_ui"))
        self.report_row.setText(str(DEFAULT_REPORT_PATH))
        self.deep_log_row.setText(str(DEFAULT_DEEP_LOG_PATH))
        self.ffmpeg_row.setText(str(ffmpeg_path))
        self._append_log(f"Auto-filled {variant} model paths.")
        self._refresh_command_preview()
        self._save_settings(silent=True)

    def _settings_changed(self, *args: Any) -> None:
        if self._loading:
            return
        self._update_extended_frames_warning()
        if self._selected_model_variant() != "INT4":
            self._apply_auto_vram_settings(update_hint=True)
        self._update_fast_iclora_route_controls()
        self._refresh_command_preview()
        self._queue_settings_save()

    def _queue_settings_save(self) -> None:
        if self._loading:
            return
        self._save_timer.start()

    def _path_rows(self) -> Dict[str, PathRow]:
        return {
            "start_media": self.start_media_row,
            "end_media": self.end_media_row,
            "start_video": self.start_video_row,
            "end_video": self.end_video_row,
            "audio": self.audio_row,
            "distilled_lora": self.distilled_lora_row,
            "user_lora_1": self.user_lora_rows[0],
            "user_lora_2": self.user_lora_rows[1],
            "user_lora_3": self.user_lora_rows[2],
            "user_lora_4": self.user_lora_rows[3],
            "spatial_upsampler": self.spatial_upsampler_row,
            "ltx_root": self.ltx_root_row,
            "python_exe": self.python_row,
            "cli_path": self.cli_row,
            "checkpoint": self.checkpoint_row,
            "gemma_root": self.gemma_row,
            "output_dir": self.output_dir_row,
            "report_path": self.report_row,
            "deep_log_path": self.deep_log_row,
            "ffmpeg": self.ffmpeg_row,
        }


    def _ltx_root_text_for_asset_checks(self) -> str:
        # During startup, audio/pipeline controls can refresh before the bottom
        # folder-location section has created self.ltx_root_row. Use the default
        # standalone LTX root until the row exists instead of crashing the UI.
        row = getattr(self, "ltx_root_row", None)
        if row is not None:
            try:
                value = row.text().strip()
                if value:
                    return value
            except Exception:
                pass
        return str(DEFAULT_LTX_ROOT)

    def _two_stage_default_path(self, key: str) -> Path:
        asset = TWO_STAGE_ASSETS[key]
        root = Path(self._ltx_root_text_for_asset_checks()).expanduser()
        return root.joinpath(*asset["subdir"], asset["filename"])

    def _two_stage_search_dirs(self, key: str) -> List[Path]:
        root = Path(self._ltx_root_text_for_asset_checks()).expanduser()
        asset = TWO_STAGE_ASSETS[key]
        dirs = [root.joinpath(*asset["subdir"]), root / "models" / "ltx23"]
        if key == "distilled_lora":
            dirs += [root / "models" / "ltx23" / "loras", root / "models" / "ltx23" / "lora"]
        else:
            dirs += [root / "models" / "ltx23" / "spatial_upsampler", root / "models" / "ltx23" / "upsamplers"]
        out: List[Path] = []
        seen = set()
        for d in dirs:
            try:
                r = d.resolve()
            except Exception:
                r = d
            if str(r).lower() not in seen:
                out.append(d)
                seen.add(str(r).lower())
        return out

    def _two_stage_asset_is_valid_path(self, key: str, path: Optional[Path]) -> Tuple[bool, str]:
        """Strictly validate a two-phase asset path.

        Older builds only checked "file exists and size > 0", which could make a
        stale saved path to the main checkpoint look like a valid LoRA/upsampler.
        This validation is intentionally boring and visible: correct-ish
        filename plus non-trivial size.
        """
        if path is None:
            return False, "missing"
        try:
            text = str(path).strip()
        except Exception:
            return False, "missing"
        if not text:
            return False, "missing"
        try:
            if not path.exists():
                return False, "file not found"
            if not path.is_file():
                return False, "not a file"
            size = path.stat().st_size
        except Exception as exc:
            return False, f"cannot read file: {type(exc).__name__}"

        asset = TWO_STAGE_ASSETS[key]
        min_bytes = int(asset.get("min_bytes", 1))
        if size < min_bytes:
            return False, f"too small ({size} bytes)"

        name = path.name.lower()
        expected = str(asset["filename"]).lower()
        if name == expected:
            return True, "found"

        if not name.endswith(".safetensors"):
            return False, "not a .safetensors file"

        markers = tuple(str(m).lower() for m in asset.get("name_markers", ()))
        if markers and all(marker in name for marker in markers):
            return True, "found compatible filename"

        if key == "spatial_upsampler" and "spatial" in name and ("upsampler" in name or "upscaler" in name):
            return True, "found compatible spatial upsampler"

        return False, f"wrong file for {asset['label']}"

    def _find_two_stage_asset(self, key: str) -> Optional[Path]:
        row = self.distilled_lora_row if key == "distilled_lora" else self.spatial_upsampler_row
        current_text = row.text().strip()
        if current_text:
            current = Path(current_text)
            ok, _reason = self._two_stage_asset_is_valid_path(key, current)
            if ok:
                return current

        filename = str(TWO_STAGE_ASSETS[key]["filename"])
        for folder in self._two_stage_search_dirs(key):
            candidate = folder / filename
            ok, _reason = self._two_stage_asset_is_valid_path(key, candidate)
            if ok:
                return candidate

        # Fallback: any close official filename in the expected folders, but
        # still require key-specific name markers so a random safetensors file is
        # never treated as ready.
        patterns = [filename]
        if key == "distilled_lora":
            patterns += ["*distilled*lora*.safetensors"]
        else:
            patterns += ["*spatial*upscaler*.safetensors", "*spatial*upsampler*.safetensors"]
        for folder in self._two_stage_search_dirs(key):
            for pat in patterns:
                try:
                    hits = sorted(folder.glob(pat), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                except Exception:
                    hits = []
                for hit in hits:
                    ok, _reason = self._two_stage_asset_is_valid_path(key, hit)
                    if ok:
                        return hit
        return None

    def _auto_fill_two_stage_asset_paths(self) -> None:
        upsampler = self._find_two_stage_asset("spatial_upsampler")
        if upsampler and self.spatial_upsampler_row.text().strip() != str(upsampler):
            self.spatial_upsampler_row.setText(str(upsampler))

    def _two_stage_asset_status_line(self, key: str) -> Tuple[bool, str]:
        asset = TWO_STAGE_ASSETS[key]
        row = self.distilled_lora_row if key == "distilled_lora" else self.spatial_upsampler_row
        label = str(asset["label"])
        path_text = row.text().strip()
        if path_text:
            path = Path(path_text)
            ok, reason = self._two_stage_asset_is_valid_path(key, path)
            if ok:
                return True, f"{label}: found - {path.name}"
            return False, f"{label}: not ready - {reason} ({path.name})"
        target = self._two_stage_default_path(key)
        return False, f"{label}: missing - will download to {target}"

    def _missing_two_stage_asset_keys(self) -> List[str]:
        self._auto_fill_two_stage_asset_paths()
        missing: List[str] = []
        keys = ["spatial_upsampler"]
        for key in keys:
            ok, _line = self._two_stage_asset_status_line(key)
            if not ok:
                missing.append(key)
        return missing

    def _update_two_stage_asset_status(self, message: str = "") -> None:
        if not hasattr(self, "two_stage_status_label"):
            return
        missing = self._missing_two_stage_asset_keys()
        lines = [self._two_stage_asset_status_line("spatial_upsampler")[1]]
        if message:
            lines.insert(0, message)
        elif not missing:
            lines.insert(0, "two-phase assets ready.")
        else:
            lines.insert(0, "two-phase assets not ready.")
        if self.pipeline_combo.currentText() not in TWO_STAGE_PIPELINES:
            lines.insert(0, "two-phase is not selected; these files will only be used for two-phase / HQ / audio-conditioned runs.")
        self.two_stage_status_label.setText("\n".join(lines))

    def _ensure_two_stage_assets(self, *, start_after_download: bool, manual: bool) -> bool:
        if self._download_worker is not None:
            return False
        self._update_two_stage_asset_status()
        missing = self._missing_two_stage_asset_keys()
        if not missing:
            self._update_two_stage_asset_status("two-phase assets ready.")
            self._save_settings(silent=True)
            return True
        if not self.auto_download_two_stage_check.isChecked():
            labels = "\n".join(f"- {TWO_STAGE_ASSETS[k]['label']}" for k in missing)
            QMessageBox.warning(self, "LTX 2.3", f"two-phase needs these missing files:\n{labels}\n\nEnable auto-download or select them in the Quality tab.")
            return False
        assets = []
        for key in missing:
            asset = dict(TWO_STAGE_ASSETS[key])
            asset["key"] = key
            asset["target"] = str(self._two_stage_default_path(key))
            assets.append(asset)
        self._start_two_stage_download(assets, start_after_download=start_after_download)
        return False

    def _start_two_stage_download(self, assets: List[Dict[str, Any]], *, start_after_download: bool) -> None:
        if self._download_worker is not None:
            return
        self._pending_start_after_download = bool(start_after_download)
        self.run_btn.setEnabled(False)
        self.two_stage_download_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.two_stage_progress.setVisible(True)
        self.two_stage_progress.setRange(0, 100)
        self.two_stage_progress.setValue(0)
        names = ", ".join(str(a["label"]) for a in assets)
        self.status_label.setText("Downloading two-phase files...")
        self.two_stage_status_label.setText(f"Downloading: {names}")
        self._append_log(f"Downloading missing two-phase assets: {names}")
        self._download_worker = TwoStageAssetDownloadWorker(assets, self)
        self._download_worker.progressChanged.connect(self._two_stage_download_progress)
        self._download_worker.logMessage.connect(self._append_log)
        self._download_worker.finishedWithResult.connect(self._two_stage_download_finished)
        self._download_worker.start()

    def _two_stage_download_progress(self, label: str, percent: int, text: str) -> None:
        if percent < 0:
            self.two_stage_progress.setRange(0, 0)
        else:
            self.two_stage_progress.setRange(0, 100)
            self.two_stage_progress.setValue(max(0, min(100, int(percent))))
        self.two_stage_status_label.setText(f"{label}: {text}")
        self.status_label.setText("Downloading two-phase files...")

    def _two_stage_download_finished(self, ok: bool, message: str, resolved: Dict[str, str]) -> None:
        self._append_log(message)
        self._download_worker = None
        self.two_stage_progress.setRange(0, 100)
        self.two_stage_progress.setValue(100 if ok else 0)
        self.two_stage_download_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        if resolved.get("distilled_lora"):
            self.distilled_lora_row.setText(resolved["distilled_lora"])
        if resolved.get("spatial_upsampler"):
            self.spatial_upsampler_row.setText(resolved["spatial_upsampler"])
        self._update_two_stage_asset_status(message if not ok else "two-phase assets ready.")
        self._save_settings(silent=True)
        start_after = self._pending_start_after_download
        self._pending_start_after_download = False
        if not ok:
            QMessageBox.warning(self, "LTX 2.3", f"Could not download two-phase files:\n{message}")
            self.status_label.setText("Download failed")
            return
        self.status_label.setText("Ready")
        if start_after:
            QTimer.singleShot(100, self.start_generation)


    def _folder_from_path(self, value: str, *, directory_value: bool = False) -> str:
        value = str(value or "").strip().strip('"')
        if not value:
            return ""
        try:
            p = Path(value).expanduser()
            folder = p if directory_value else p.parent
            folder_text = str(folder)
            return "" if folder_text in {".", ""} else folder_text
        except Exception:
            return ""

    def _folder_from_paths_text(self, value: str) -> str:
        paths = self._split_paths(value)
        return self._folder_from_path(paths[0]) if paths else ""

    def _remember_last_folder(self, key: str, value: str, *, directory_value: bool = False) -> None:
        folder = self._folder_from_path(value, directory_value=directory_value)
        if folder:
            self._last_browse_dirs[key] = folder

    def _collect_last_used_folders(self) -> Dict[str, str]:
        data = dict(self._last_browse_dirs)
        for key, row in self._path_rows().items():
            folder = row.last_used_folder()
            if folder:
                data[key] = folder
        output_dir = self.output_dir_row.text().strip()
        if output_dir:
            data["output_dir"] = output_dir
        return {str(k): str(v) for k, v in data.items() if str(v).strip()}

    def _apply_last_used_folders(self) -> None:
        for key, row in self._path_rows().items():
            if key in self._last_browse_dirs:
                row.set_last_used_folder(self._last_browse_dirs[key])

    def _refresh_command_preview(self) -> None:
        if self._loading or not hasattr(self, "command_preview") or not hasattr(self, "python_row"):
            return
        try:
            program, args, _output, _extra = self.build_command()
            self.command_preview.setPlainText(self._quote_command([program] + args))
        except Exception as exc:
            self.command_preview.setPlainText(f"Could not build command: {type(exc).__name__}: {exc}")

    def _save_settings(self, silent: bool = False) -> None:
        try:
            # Hard guard: settings can only be written to FrameVision/root/presets/setsave/ltx23_ui.json.
            self.settings_path = fixed_settings_path()
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            payload = asdict(self.collect_settings())
            payload["settings_path_locked_to"] = str(self.settings_path)

            # Avoid touching the JSON file when nothing actually changed.
            # This keeps timestamps stable and prevents silent rewrite churn from
            # refresh/status actions that do not alter user settings.
            if self.settings_path.exists():
                try:
                    with self.settings_path.open("r", encoding="utf-8") as f:
                        existing_payload = json.load(f)
                    if existing_payload == payload:
                        if not silent:
                            self._append_log(f"Settings unchanged; not saved: {self.settings_path}")
                        return
                except Exception:
                    # Broken/old JSON should be repaired by writing the current payload.
                    pass

            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            if not silent:
                self._append_log(f"Saved settings: {self.settings_path}")
        except Exception as exc:
            msg = f"Could not save settings: {type(exc).__name__}: {exc}"
            try:
                self._append_log(msg)
            except Exception:
                pass
            if not silent:
                QMessageBox.warning(self, "Save settings", str(exc))

    def _load_settings(self) -> None:
        self.settings_path = fixed_settings_path()
        if not self.settings_path.exists():
            self.apply_settings(asdict(LTX23UISettings()))
            return
        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                self.apply_settings(json.load(f))
        except Exception as exc:
            self.apply_settings(asdict(LTX23UISettings()))
            self._append_log(f"Could not load settings: {type(exc).__name__}: {exc}")

    def _load_settings_dialog(self) -> None:
        start = self._last_browse_dirs.get("preset_load") or str(self.settings_path.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Load LTX preset", start, "JSON (*.json);;All files (*.*)")
        if not path:
            return
        try:
            self._remember_last_folder("preset_load", path)
            with open(path, "r", encoding="utf-8") as f:
                self.apply_settings(json.load(f))
            self._append_log(f"Loaded preset into fixed settings file: {path}")
            self._save_settings(silent=True)
        except Exception as exc:
            QMessageBox.warning(self, "Load preset", str(exc))

    def _append_log(self, text: str) -> None:
        self.log_edit.moveCursor(QTextCursor.End)
        self.log_edit.insertPlainText(str(text) + "\n")
        self.log_edit.moveCursor(QTextCursor.End)

    def save_log(self) -> None:
        start_dir = self._last_browse_dirs.get("save_log") or str(APP_ROOT / "logs")
        path, _ = QFileDialog.getSaveFileName(self, "Save log", str(Path(start_dir) / "ltx23_ui_log.txt"), "Text (*.txt);;All files (*.*)")
        if not path:
            return
        self._remember_last_folder("save_log", path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.log_edit.toPlainText(), encoding="utf-8")
        self._queue_settings_save()

    def copy_command_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.command_preview.toPlainText())
        self._append_log("Command copied to clipboard.")

    def open_output_folder(self) -> None:
        target = self.active_output_path.parent if self.active_output_path else Path(self.output_dir_row.text() or str(DEFAULT_OUTPUT_DIR))
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def closeEvent(self, event: Any) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_settings(silent=True)
        super().closeEvent(event)


# Friendly aliases for importing from FrameVision.
def create_widget(parent: Optional[QWidget] = None) -> LTX23RunnerWidget:
    return LTX23RunnerWidget(parent)


def make_widget(parent: Optional[QWidget] = None) -> LTX23RunnerWidget:
    return create_widget(parent)


def main() -> int:
    app = QApplication(sys.argv)
    widget = LTX23RunnerWidget()
    widget.resize(1100, 850)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
