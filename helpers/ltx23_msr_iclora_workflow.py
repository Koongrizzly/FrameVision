# FrameVision LTX 2.3 MSR / IC-LoRA workflow bridge.
#
# This file owns the reusable MSR workflow routing. The VRAM Lab CLI remains a
# runner / memory wrapper and should only call into this helper. Planner and
# Music Clip Creator can later use the same helper without depending on the
# VRAM Lab CLI.
#
# Workflow basis:
# - LiconStudio / ComfyUI-Licon-MSR reference ordering and frame sequence idea
# - Lightricks LTX native ltx_pipelines.ic_lora video-conditioning entry point
# - Lightricks/ComfyUI-LTXVideo IC-LoRA guide behavior

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


@dataclass
class MSRICLoRAPlan:
    """Prepared native LTX IC-LoRA run plan."""

    module_name: str
    argv: List[str]
    reference_video_path: str
    frames_dir: str
    metadata_path: str
    prompt_block: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _has_cli_option(args: Sequence[str], option: str) -> bool:
    return any(str(x) == option for x in list(args or []))


def _append_lora_groups(argv: List[str], groups: Any) -> None:
    for group in list(groups or []):
        clean = [str(x).strip() for x in list(group or []) if str(x).strip()]
        if clean:
            argv.append("--lora")
            argv.extend(clean)


def _import_ref_builder():
    try:
        from video_ref_builder import build_msr_reference, build_reference_prompt_block, validate_is_msr_ready
        return build_msr_reference, build_reference_prompt_block, validate_is_msr_ready
    except Exception:
        from helpers.video_ref_builder import build_msr_reference, build_reference_prompt_block, validate_is_msr_ready
        return build_msr_reference, build_reference_prompt_block, validate_is_msr_ready



# Arguments accepted by the normal two-stage LTX module but not by
# ltx_pipelines.ic_lora. Strip only on the MSR/IC-LoRA route.
_ICLORA_UNSUPPORTED_VALUE_OPTIONS = {
    "--video-cfg-guidance-scale": 1,
    "--video-stg-guidance-scale": 1,
    "--video-rescale-scale": 1,
    "--audio-cfg-guidance-scale": 1,
    "--audio-stg-guidance-scale": 1,
    "--audio-rescale-scale": 1,
    "--a2v-guidance-scale": 1,
    "--v2a-guidance-scale": 1,
    "--video-skip-step": 1,
    "--audio-skip-step": 1,
}

_ICLORA_UNSUPPORTED_MULTI_OPTIONS = {
    "--video-stg-blocks",
    "--audio-stg-blocks",
}

_ICLORA_SUPPORTED_EXTRA_STARTS = {
    "--image",
    "--lora",
    "--compile",
    "--conditioning-attention-mask",
}

_ICLORA_SUPPORTED_VALUE_OPTIONS = {
    "--offload": 1,
    "--max-batch-size": 1,
    "--quantization": 1,
    "--conditioning-attention-mask": 2,
}

_ICLORA_SUPPORTED_FLAGS = {
    "--enhance-prompt",
    "--skip-stage-2",
    "--compile",
}


def _filter_ic_lora_extra_args(extra: Sequence[str]) -> List[str]:
    """Keep extras compatible with ltx_pipelines.ic_lora.

    The normal FrameVision LTX argv may contain guider options used by
    ti2vid_two_stages. The native IC-LoRA parser is distilled/simple and rejects
    those. This filter is intentionally scoped to the MSR route only.
    """
    items = [str(x) for x in list(extra or [])]
    out: List[str] = []
    i = 0
    while i < len(items):
        item = items[i]

        if item in _ICLORA_UNSUPPORTED_VALUE_OPTIONS:
            i += 1 + int(_ICLORA_UNSUPPORTED_VALUE_OPTIONS[item])
            continue

        if item in _ICLORA_UNSUPPORTED_MULTI_OPTIONS:
            i += 1
            while i < len(items) and not str(items[i]).startswith("--"):
                i += 1
            continue

        if item in _ICLORA_SUPPORTED_FLAGS:
            out.append(item)
            i += 1
            continue

        if item in _ICLORA_SUPPORTED_VALUE_OPTIONS:
            count = int(_ICLORA_SUPPORTED_VALUE_OPTIONS[item])
            out.extend(items[i:i + 1 + count])
            i += 1 + count
            continue

        if item in _ICLORA_SUPPORTED_EXTRA_STARTS:
            # Variable-length parser actions. Copy option plus values until the
            # next option token; argparse will validate exact arity.
            out.append(item)
            i += 1
            while i < len(items) and not str(items[i]).startswith("--"):
                out.append(items[i])
                i += 1
            continue

        # Unknown options are intentionally dropped on this route instead of
        # passing normal-pipeline-only args into ic_lora and failing before load.
        if item.startswith("--"):
            i += 1
            while i < len(items) and not str(items[i]).startswith("--"):
                i += 1
            continue

        # Positional leftovers should not exist here; drop them.
        i += 1

    return out


def _build_prompt(args: argparse.Namespace, generated_prompt_block: str) -> str:
    prompt = _clean_text(getattr(args, "prompt", ""))
    prompt_block = _clean_text(getattr(args, "msr_prompt_block", "")) or _clean_text(generated_prompt_block)
    if prompt_block and prompt_block not in prompt:
        return f"{prompt_block}\n\n{prompt}" if prompt else prompt_block
    return prompt


def prepare_msr_iclora_plan(
    args: argparse.Namespace,
    *,
    app_root: str | Path,
    report_stamp: str,
) -> MSRICLoRAPlan:
    """Build the MSR reference and return a native ltx_pipelines.ic_lora argv.

    The native LTX IC-LoRA pipeline owns reference-video conditioning through
    --video-conditioning. This helper prepares that route and deliberately does
    not create normal --image keyframe fallback args.
    """
    if not bool(getattr(args, "msr_enabled", False)):
        raise RuntimeError("prepare_msr_iclora_plan called while MSR is disabled")

    build_msr_reference, build_reference_prompt_block, validate_is_msr_ready = _import_ref_builder()

    subject_paths = [
        _clean_text(getattr(args, "msr_ref_1", "")),
        _clean_text(getattr(args, "msr_ref_2", "")),
        _clean_text(getattr(args, "msr_ref_3", "")),
        _clean_text(getattr(args, "msr_ref_4", "")),
    ]
    if not any(subject_paths):
        raise RuntimeError("--msr-enabled requires at least one --msr-ref-N image.")

    descriptions = [
        str(getattr(args, "msr_ref_1_description", "") or ""),
        str(getattr(args, "msr_ref_2_description", "") or ""),
        str(getattr(args, "msr_ref_3_description", "") or ""),
        str(getattr(args, "msr_ref_4_description", "") or ""),
    ]

    root = Path(app_root)
    out_dir = _clean_text(getattr(args, "msr_output_dir", ""))
    if not out_dir:
        out_dir = str(root / "temp" / "video_refs" / "cli_runs" / f"ltx_msr_{report_stamp}")

    result = build_msr_reference(
        subject_paths=subject_paths,
        background_path=_clean_text(getattr(args, "msr_background", "")),
        width=int(getattr(args, "width", 1280) or 1280),
        height=int(getattr(args, "height", 704) or 704),
        frame_count=int(getattr(args, "msr_frame_count", 17) or 17),
        output_dir=out_dir,
        fps=int(getattr(args, "msr_fps", 50) or 50),
        descriptions=descriptions,
        background_description=str(getattr(args, "msr_background_description", "") or ""),
        save_frames=True,
        save_video=True,
        prefix="msr_ref",
        resize_mode=_clean_text(getattr(args, "msr_resize_mode", "stretch")) or "stretch",
    )
    ok, message = validate_is_msr_ready(result)
    if not ok:
        raise RuntimeError(message)
    if not result.video_path:
        raise RuntimeError("MSR reference video was not created.")

    # Preserve these fields on args so the caller/report can expose them without
    # owning the workflow details.
    try:
        setattr(args, "_msr_last_frames_dir", str(result.frames_dir or ""))
        setattr(args, "_msr_last_video_path", str(result.video_path or ""))
        setattr(args, "_msr_last_metadata_path", str(result.metadata_path or ""))
    except Exception:
        pass

    prompt_block = build_reference_prompt_block(result.used_sources)
    prompt = _build_prompt(args, prompt_block)

    extra = list(getattr(args, "extra", None) or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    extra = _filter_ic_lora_extra_args(extra)

    argv: List[str] = [
        "ltx_pipelines.ic_lora",
        "--distilled-checkpoint-path", str(getattr(args, "checkpoint_path")),
        "--gemma-root", str(getattr(args, "gemma_root")),
        "--prompt", prompt,
        "--output-path", str(getattr(args, "output_path")),
        "--height", str(int(getattr(args, "height", 0))),
        "--width", str(int(getattr(args, "width", 0))),
        "--num-frames", str(int(getattr(args, "num_frames", 0))),
        "--frame-rate", str(float(getattr(args, "frame_rate", 0))),
        "--seed", str(int(getattr(args, "seed", 0))),
    ]

    spatial = _clean_text(getattr(args, "spatial_upsampler_path", ""))
    if spatial:
        argv.extend(["--spatial-upsampler-path", spatial])

    # Normal start image support remains native --image input, separate from MSR.
    if _clean_text(getattr(args, "i2v_image", "")) and not _has_cli_option(extra, "--image"):
        argv.extend([
            "--image",
            str(getattr(args, "i2v_image")),
            str(int(getattr(args, "i2v_image_frame", 0))),
            str(float(getattr(args, "i2v_image_strength", 1.0))),
            str(int(getattr(args, "i2v_image_crf", 0))),
        ])

    _append_lora_groups(argv, getattr(args, "lora", None))

    strength = float(getattr(args, "msr_strength", 1.0) or 1.0)
    argv.extend(["--video-conditioning", str(result.video_path), f"{strength:g}"])

    if str(getattr(args, "vram_lab", "off")).lower().strip() != "off" and not _has_cli_option(extra, "--offload"):
        argv.extend(["--offload", "cpu"])

    if extra:
        argv.extend(extra)

    print(f"[ltx-msr] Built MSR reference: {result.frame_count} frames, {result.width}x{result.height}, frames={result.frames_dir}", flush=True)
    print(f"[ltx-msr] MP4 reference: {result.video_path}", flush=True)
    print(f"[ltx-msr] Metadata: {result.metadata_path}", flush=True)
    print("[ltx-msr] Native IC-LoRA transport: ltx_pipelines.ic_lora --video-conditioning", flush=True)

    return MSRICLoRAPlan(
        module_name="ltx_pipelines.ic_lora",
        argv=argv,
        reference_video_path=str(result.video_path or ""),
        frames_dir=str(result.frames_dir or ""),
        metadata_path=str(result.metadata_path or ""),
        prompt_block=prompt_block,
    )
