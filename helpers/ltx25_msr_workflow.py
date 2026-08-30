"""FrameVision LTX 2.5 Licon MSR workflow bridge.

Separate from the older ltx23_msr_iclora_workflow helper.  This bridge does not
build a pseudo reference MP4 and does not route through ltx_pipelines.ic_lora.
It prepares arguments for the native FrameVision helper ``ltx25_msr_pipeline``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence


DEFAULT_MSR_RELATIVE = Path("models") / "ltx-2.5" / "msr" / "LTX-2.5-Licon-MSR-V1.safetensors"


@dataclass
class LTX25MSRPlan:
    module_name: str
    argv: List[str]
    msr_lora_path: str
    references: List[str]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _has_option(items: Sequence[str], option: str) -> bool:
    return any(str(item) == option for item in items)


def _append_lora_groups(argv: List[str], groups: Any) -> None:
    for group in list(groups or []):
        clean = [str(x).strip() for x in list(group or []) if str(x).strip()]
        if clean:
            argv.append("--lora")
            argv.extend(clean)


def _resolve_msr_model(args: argparse.Namespace, app_root: Path) -> Path:
    configured = _clean(getattr(args, "msr25_lora_path", "")) or _clean(getattr(args, "msr_lora_path", ""))
    path = Path(configured).expanduser() if configured else app_root / DEFAULT_MSR_RELATIVE
    if not path.is_absolute():
        path = app_root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(
            "LTX 2.5 Licon MSR model is missing: "
            f"{path}. Run presets/extra_env/install_ltx25_msr.bat first."
        )
    return path


def _collect_refs(args: argparse.Namespace) -> list[tuple[str, str]]:
    candidates = [
        ("--msr-ref-1", _clean(getattr(args, "msr_ref_1", ""))),
        ("--msr-ref-2", _clean(getattr(args, "msr_ref_2", ""))),
        ("--msr-ref-3", _clean(getattr(args, "msr_ref_3", ""))),
        ("--msr-ref-4", _clean(getattr(args, "msr_ref_4", ""))),
        ("--msr-background", _clean(getattr(args, "msr_background", ""))),
    ]
    result: list[tuple[str, str]] = []
    for option, value in candidates:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"LTX 2.5 MSR reference does not exist: {path}")
        result.append((option, str(path)))
    if not any(option == "--msr-ref-1" for option, _ in result):
        raise RuntimeError("LTX 2.5 MSR requires reference 1 (--msr-ref-1).")
    if len(result) > 5:
        raise RuntimeError("LTX 2.5 MSR supports at most 5 connected references.")
    return result


def prepare_ltx25_msr_plan(
    args: argparse.Namespace,
    *,
    app_root: str | Path,
) -> LTX25MSRPlan:
    """Build the argv for the isolated LTX 2.5 MSR pipeline.

    Expected FrameVision fields intentionally mirror the older MSR route where
    practical (msr_ref_1..4, msr_background, msr_strength), so the GUI can reuse
    its existing reference widgets while choosing a different backend.
    """

    root = Path(app_root).resolve()
    msr_model = _resolve_msr_model(args, root)
    refs = _collect_refs(args)

    audio_path = _clean(getattr(args, "audio_path", ""))
    if not audio_path or not Path(audio_path).expanduser().is_file():
        raise RuntimeError(
            "The first LTX 2.5 MSR FrameVision backend is the supplied-audio path and requires --audio-path."
        )

    module_name = "ltx25_msr_pipeline"
    argv: List[str] = [module_name]

    # LTX 2.5 normal model arguments.  FrameVision already owns these paths; this
    # helper only forwards them and never creates another model/environment tree.
    model_arg_pairs = (
        ("--checkpoint-path", "checkpoint_path"),
        ("--spatial-upsampler-path", "spatial_upsampler_path"),
        ("--gemma-root", "gemma_root"),
        ("--prompt-enhancer-gemma-root", "prompt_enhancer_gemma_root"),
    )
    for option, attr in model_arg_pairs:
        value = _clean(getattr(args, attr, ""))
        if value:
            argv.extend([option, value])

    argv.extend([
        "--prompt", _clean(getattr(args, "prompt", "")),
        "--negative-prompt", _clean(getattr(args, "negative_prompt", "")),
        "--output-path", _clean(getattr(args, "output_path", "")),
        "--height", str(int(getattr(args, "height", 0))),
        "--width", str(int(getattr(args, "width", 0))),
        "--num-frames", str(int(getattr(args, "num_frames", 0))),
        "--frame-rate", str(float(getattr(args, "frame_rate", 0))),
        "--seed", str(int(getattr(args, "seed", 0))),
        "--num-inference-steps", str(int(getattr(args, "num_inference_steps", 30) or 30)),
        "--msr-lora-path", str(msr_model),
        "--msr-strength-model", str(float(getattr(args, "msr25_model_strength", 1.0) or 1.0)),
        "--msr-reference-strength", str(float(getattr(args, "msr_strength", 1.0) or 1.0)),
        "--msr-reference-frames", str(int(getattr(args, "msr25_reference_frames", 33) or 33)),
        "--msr-tile-size", str(int(getattr(args, "msr25_tile_size", 256) or 256)),
        "--msr-tile-overlap", str(int(getattr(args, "msr25_tile_overlap", 64) or 64)),
        "--audio-path", str(Path(audio_path).expanduser().resolve()),
        "--audio-start-time", str(float(getattr(args, "audio_start_time", 0.0) or 0.0)),
    ])

    for option, value in refs:
        argv.extend([option, value])

    audio_max = getattr(args, "audio_max_duration", None)
    if audio_max is not None and float(audio_max) > 0:
        argv.extend(["--audio-max-duration", str(float(audio_max))])

    if bool(getattr(args, "msr25_use_tiled_encode", True)):
        argv.append("--msr-use-tiled-encode")
    else:
        argv.append("--no-msr-use-tiled-encode")

    _append_lora_groups(argv, getattr(args, "lora", None))
    distilled = getattr(args, "distilled_lora", None)
    for group in list(distilled or []):
        clean = [str(x).strip() for x in list(group or []) if str(x).strip()]
        if clean:
            argv.append("--distilled-lora")
            argv.extend(clean)

    # Pass through the normal LTX guider/offload/quantization options unchanged.
    extra = list(getattr(args, "extra", None) or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra:
        argv.extend(str(x) for x in extra)

    print(
        f"[ltx25-msr] Native Licon MSR | refs={len(refs)} | frames="
        f"{int(getattr(args, 'msr25_reference_frames', 33) or 33)} | model={msr_model}",
        flush=True,
    )
    print("[ltx25-msr] No pseudo-video / no LTX 2.3 IC-LoRA route", flush=True)
    print("[ltx25-msr] MSR references will be encoded independently in Stage 1 and Stage 2", flush=True)

    return LTX25MSRPlan(
        module_name=module_name,
        argv=argv,
        msr_lora_path=str(msr_model),
        references=[value for _, value in refs],
    )
