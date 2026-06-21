# -*- coding: utf-8 -*-
"""
Reusable video/reference-frame builder for FrameVision.

Initial target: LTX 2.3 LiconStudio MSR / Multiple-Subject-Reference workflows.

What this does:
- Loads up to 4 subject/reference images plus one required background image.
- Keeps the same source order used by ComfyUI-Licon-MSR:
      ref_1 -> ref_2 -> ref_3 -> ref_4 -> background
- Resizes every source to the target video size.
- Expands the sources into a short fixed-frame reference sequence.
- Can save that sequence as PNG frames, optional MP4, metadata JSON, and/or return a torch tensor.

This file intentionally contains no Qt/UI code and no Planner/Music Clip/LTX runner logic.
It should stay reusable from all three workflows.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps


PathLike = Union[str, os.PathLike]

MSR_ALLOWED_FRAME_COUNTS = (17, 25, 33, 41)
MSR_MAX_SUBJECTS = 4
DEFAULT_VIDEO_FPS = 24
DEFAULT_MSR_FPS = 50


@dataclass
class ReferenceSource:
    """One logical reference source after validation."""

    slot: str
    path: str
    role: str
    description: str = ""
    source_index: int = 0


@dataclass
class ReferenceBuildResult:
    """Result returned by build_msr_reference()."""

    ok: bool
    mode: str
    width: int
    height: int
    frame_count: int
    fps: int
    output_dir: str
    frames_dir: str = ""
    video_path: str = ""
    metadata_path: str = ""
    used_sources: List[Dict[str, Any]] = None
    frame_plan: List[Dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("used_sources") is None:
            data["used_sources"] = []
        if data.get("frame_plan") is None:
            data["frame_plan"] = []
        return data


class VideoReferenceError(RuntimeError):
    """Raised when reference building fails."""


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def build_msr_reference(
    subject_paths: Optional[Sequence[Optional[PathLike]]] = None,
    background_path: Optional[PathLike] = None,
    width: int = 1280,
    height: int = 704,
    frame_count: int = 17,
    output_dir: Optional[PathLike] = None,
    *,
    fps: int = DEFAULT_MSR_FPS,
    descriptions: Optional[Sequence[str]] = None,
    background_description: str = "",
    save_frames: bool = True,
    save_video: bool = False,
    save_metadata: bool = True,
    return_tensor: bool = False,
    clear_output_dir: bool = False,
    prefix: str = "msr_ref",
    resize_mode: str = "stretch",
    video_codec: str = "mp4v",
) -> Union[ReferenceBuildResult, Tuple[ReferenceBuildResult, Any]]:
    """
    Build an LTX/MSR-style multiple-reference frame sequence.

    Parameters
    ----------
    subject_paths:
        Up to 4 image paths. Empty/None items are skipped.
    background_path:
        Required background/environment image path. Always placed last.
    width, height:
        Target frame size. LTX sizes should normally be divisible by 32.
    frame_count:
        Official MSR choices are 17, 25, 33, 41.
    output_dir:
        Folder where frames/metadata/video are written.
    fps:
        Metadata/video FPS. MSR model notes often recommend 50 fps for motion.
    descriptions:
        Optional descriptions for subject slots. These are stored in metadata so
        UI/workflows can generate global prompt text later.
    background_description:
        Optional background description stored in metadata.
    save_frames:
        Save PNG frames in output_dir/frames.
    save_video:
        Also encode MP4 with OpenCV if available.
    save_metadata:
        Save JSON metadata next to the frame folder.
    return_tensor:
        If True, return (result, tensor) where tensor is float32 [F,H,W,C].
        Torch is imported lazily only when requested.
    clear_output_dir:
        Delete output_dir before writing. Use carefully.
    prefix:
        Prefix for output files/folders.
    resize_mode:
        "stretch" matches ComfyUI-Licon-MSR behavior.
        "fit" letterboxes/pads without cropping.
        "fill" center-crops after preserving aspect.
    video_codec:
        FourCC string used by OpenCV when save_video=True.

    Returns
    -------
    ReferenceBuildResult, or (ReferenceBuildResult, torch.Tensor) when return_tensor=True.
    """

    try:
        sources = collect_msr_sources(
            subject_paths=subject_paths,
            background_path=background_path,
            descriptions=descriptions,
            background_description=background_description,
        )
        validate_build_settings(width=width, height=height, frame_count=frame_count, fps=fps)

        out_dir = prepare_output_dir(output_dir, clear_output_dir=clear_output_dir, prefix=prefix)
        frames_dir = out_dir / "frames"
        if save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        prepared_images = [
            load_and_prepare_image(src.path, width=width, height=height, resize_mode=resize_mode)
            for src in sources
        ]
        frames, frame_plan = expand_reference_frames(prepared_images, sources, frame_count)

        if save_frames:
            write_png_frames(frames, frames_dir, prefix=prefix)

        video_path = ""
        if save_video:
            video_path = str(out_dir / f"{prefix}_{width}x{height}_{frame_count}f_{fps}fps.mp4")
            write_mp4(frames, video_path, fps=fps, codec=video_codec)

        metadata_path = ""
        if save_metadata:
            metadata_path = str(out_dir / f"{prefix}_metadata.json")
            write_metadata(
                metadata_path,
                mode="ltx_msr",
                width=width,
                height=height,
                frame_count=frame_count,
                fps=fps,
                resize_mode=resize_mode,
                output_dir=str(out_dir),
                frames_dir=str(frames_dir) if save_frames else "",
                video_path=video_path,
                sources=sources,
                frame_plan=frame_plan,
            )

        result = ReferenceBuildResult(
            ok=True,
            mode="ltx_msr",
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            output_dir=str(out_dir),
            frames_dir=str(frames_dir) if save_frames else "",
            video_path=video_path,
            metadata_path=metadata_path,
            used_sources=[asdict(src) for src in sources],
            frame_plan=frame_plan,
        )

        if return_tensor:
            return result, frames_to_torch_tensor(frames)
        return result

    except Exception as exc:
        result = ReferenceBuildResult(
            ok=False,
            mode="ltx_msr",
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            output_dir=str(output_dir or ""),
            error=str(exc),
            used_sources=[],
            frame_plan=[],
        )
        if return_tensor:
            return result, None
        return result


def build_reference_prompt_block(
    sources: Sequence[Union[ReferenceSource, Dict[str, Any]]],
    *,
    intro: str = "Use the following reference images as visual memory:",
) -> str:
    """
    Build a small global-prompt block describing reference slots.

    This does not try to write the shot action. It only explains what each
    reference slot represents, so Planner/Music Clip/Normal LTX can prepend it
    to a prompt relay/global prompt.
    """

    lines = [intro]
    for item in sources:
        if isinstance(item, ReferenceSource):
            slot = item.slot
            role = item.role
            desc = item.description
        else:
            slot = str(item.get("slot", "reference"))
            role = str(item.get("role", "reference"))
            desc = str(item.get("description", ""))

        label = slot.replace("ref_", "reference image ").replace("background", "background reference")
        if desc:
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {label}: {role}")
    lines.append("Preserve character identity, clothing, object details, environment style, and background consistency across the generated video.")
    return "\n".join(lines)


def validate_is_msr_ready(result: ReferenceBuildResult) -> Tuple[bool, str]:
    """Small convenience check for UI/workflow callers."""

    if not result.ok:
        return False, result.error or "MSR reference build failed."
    if not result.used_sources or len(result.used_sources) < 2:
        return False, "MSR needs at least one subject/reference image plus one background image."
    if result.frame_count not in MSR_ALLOWED_FRAME_COUNTS:
        return False, f"MSR frame count should be one of {MSR_ALLOWED_FRAME_COUNTS}."
    return True, "MSR reference is ready."


# -----------------------------------------------------------------------------
# Source collection / validation
# -----------------------------------------------------------------------------


def collect_msr_sources(
    subject_paths: Optional[Sequence[Optional[PathLike]]],
    background_path: Optional[PathLike],
    descriptions: Optional[Sequence[str]] = None,
    background_description: str = "",
) -> List[ReferenceSource]:
    """Collect sources in the same order as ComfyUI-Licon-MSR."""

    if background_path is None or not str(background_path).strip():
        raise VideoReferenceError("background_path is required for MSR reference building.")

    descriptions = list(descriptions or [])
    subject_paths = list(subject_paths or [])[:MSR_MAX_SUBJECTS]

    sources: List[ReferenceSource] = []
    for idx, raw_path in enumerate(subject_paths, start=1):
        if raw_path is None or not str(raw_path).strip():
            continue
        path = normalize_existing_path(raw_path)
        desc = descriptions[idx - 1].strip() if idx - 1 < len(descriptions) and descriptions[idx - 1] else ""
        sources.append(
            ReferenceSource(
                slot=f"ref_{idx}",
                path=str(path),
                role="subject/reference",
                description=desc,
                source_index=idx,
            )
        )

    if not sources:
        raise VideoReferenceError("At least one subject/reference image is required.")

    bg_path = normalize_existing_path(background_path)
    sources.append(
        ReferenceSource(
            slot="background",
            path=str(bg_path),
            role="background/environment",
            description=background_description.strip() if background_description else "",
            source_index=len(sources) + 1,
        )
    )
    return sources


def normalize_existing_path(path: PathLike) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise VideoReferenceError(f"File not found: {p}")
    if not p.is_file():
        raise VideoReferenceError(f"Not a file: {p}")
    return p


def validate_build_settings(width: int, height: int, frame_count: int, fps: int) -> None:
    if width <= 0 or height <= 0:
        raise VideoReferenceError("width and height must be positive.")
    if width % 8 != 0 or height % 8 != 0:
        raise VideoReferenceError("width and height should be divisible by 8 at minimum.")
    if frame_count not in MSR_ALLOWED_FRAME_COUNTS:
        raise VideoReferenceError(f"frame_count must be one of {MSR_ALLOWED_FRAME_COUNTS}.")
    if fps <= 0:
        raise VideoReferenceError("fps must be positive.")


# -----------------------------------------------------------------------------
# Image preparation / frame expansion
# -----------------------------------------------------------------------------


def load_and_prepare_image(path: PathLike, *, width: int, height: int, resize_mode: str = "stretch") -> np.ndarray:
    """Load an image as RGB uint8 HWC and resize it."""

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = resize_pil_image(im, width=width, height=height, resize_mode=resize_mode)
        arr = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(arr)


def resize_pil_image(image: Image.Image, *, width: int, height: int, resize_mode: str = "stretch") -> Image.Image:
    """
    Resize image.

    "stretch" intentionally matches the official ComfyUI-Licon-MSR behavior.
    "fit" and "fill" are included for FrameVision UI choices later.
    """

    resize_mode = (resize_mode or "stretch").lower().strip()
    target = (int(width), int(height))

    if resize_mode == "stretch":
        return image.resize(target, Image.Resampling.LANCZOS)

    if resize_mode == "fit":
        fitted = ImageOps.contain(image, target, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", target, (0, 0, 0))
        x = (target[0] - fitted.width) // 2
        y = (target[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        return canvas

    if resize_mode == "fill":
        return ImageOps.fit(image, target, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    raise VideoReferenceError(f"Unknown resize_mode: {resize_mode}. Use stretch, fit, or fill.")


def expand_reference_frames(
    images: Sequence[np.ndarray],
    sources: Sequence[ReferenceSource],
    frame_count: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Spread frame_count over connected images in order.

    This mirrors ComfyUI-Licon-MSR:
    base_count = frame_count // len(images)
    remainder = frame_count % len(images)
    earlier sources receive the remainder frames first.
    """

    if not images:
        raise VideoReferenceError("No prepared images were provided.")
    if len(images) != len(sources):
        raise VideoReferenceError("Internal mismatch between images and source metadata.")

    base_count = frame_count // len(images)
    remainder = frame_count % len(images)

    frames: List[np.ndarray] = []
    frame_plan: List[Dict[str, Any]] = []
    start = 0

    for index, (image, src) in enumerate(zip(images, sources)):
        repeats = base_count + (1 if index < remainder else 0)
        end = start + repeats - 1
        frame_plan.append(
            {
                "slot": src.slot,
                "role": src.role,
                "path": src.path,
                "description": src.description,
                "repeat_count": repeats,
                "frame_start": start,
                "frame_end": end,
            }
        )
        frames.extend([image.copy() for _ in range(repeats)])
        start += repeats

    if len(frames) != frame_count:
        raise VideoReferenceError(f"Frame expansion failed: expected {frame_count}, got {len(frames)}.")
    return frames, frame_plan


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def prepare_output_dir(output_dir: Optional[PathLike], *, clear_output_dir: bool = False, prefix: str = "msr_ref") -> Path:
    if output_dir is None or not str(output_dir).strip():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "temp" / "video_refs" / f"{prefix}_{stamp}"
    out_dir = Path(output_dir).expanduser().resolve()
    if clear_output_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_png_frames(frames: Sequence[np.ndarray], frames_dir: PathLike, *, prefix: str = "msr_ref") -> List[str]:
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    pad = max(4, int(math.log10(max(1, len(frames)))) + 1)
    for i, frame in enumerate(frames):
        path = frames_dir / f"{prefix}_{i:0{pad}d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        written.append(str(path))
    return written


def write_mp4(frames: Sequence[np.ndarray], video_path: PathLike, *, fps: int = DEFAULT_VIDEO_FPS, codec: str = "mp4v") -> str:
    """Write MP4 using OpenCV. Imported lazily to avoid forcing cv2 unless needed."""

    if not frames:
        raise VideoReferenceError("No frames to write.")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise VideoReferenceError("OpenCV is required for save_video=True. Install opencv-python or disable save_video.") from exc

    first = frames[0]
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise VideoReferenceError(f"Could not open video writer for: {video_path}")
    try:
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
    return str(video_path)


def write_metadata(
    metadata_path: PathLike,
    *,
    mode: str,
    width: int,
    height: int,
    frame_count: int,
    fps: int,
    resize_mode: str,
    output_dir: str,
    frames_dir: str,
    video_path: str,
    sources: Sequence[ReferenceSource],
    frame_plan: Sequence[Dict[str, Any]],
) -> str:
    metadata = {
        "ok": True,
        "builder": "video_ref_builder.py",
        "mode": mode,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "resize_mode": resize_mode,
        "output_dir": output_dir,
        "frames_dir": frames_dir,
        "video_path": video_path,
        "source_order": ["ref_1", "ref_2", "ref_3", "ref_4", "background"],
        "used_sources": [asdict(src) for src in sources],
        "frame_plan": list(frame_plan),
        "prompt_block": build_reference_prompt_block(sources),
        "notes": [
            "Initial behavior mirrors ComfyUI-Licon-MSR source ordering and frame distribution.",
            "Load the matching LTX 2.3 MSR LoRA in the LTX pipeline before using this reference sequence.",
            "For high-motion MSR tests, 50 fps is commonly recommended by the model notes.",
        ],
    }
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def frames_to_numpy(frames: Sequence[np.ndarray], *, normalize: bool = True) -> np.ndarray:
    arr = np.stack(frames, axis=0)
    if normalize:
        return arr.astype(np.float32) / 255.0
    return arr


def frames_to_torch_tensor(frames: Sequence[np.ndarray]):
    """Return torch float32 tensor [F,H,W,C], matching ComfyUI IMAGE convention."""

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise VideoReferenceError("Torch is required for return_tensor=True.") from exc
    return torch.from_numpy(frames_to_numpy(frames, normalize=True).astype(np.float32))


# -----------------------------------------------------------------------------
# Simple CLI for standalone testing
# -----------------------------------------------------------------------------


def _parse_cli(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Build an LTX/MSR-style multi-reference frame sequence.")
    parser.add_argument("--ref1", default="", help="Subject/reference image 1")
    parser.add_argument("--ref2", default="", help="Subject/reference image 2")
    parser.add_argument("--ref3", default="", help="Subject/reference image 3")
    parser.add_argument("--ref4", default="", help="Subject/reference image 4")
    parser.add_argument("--background", required=True, help="Required background/environment image")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--frame-count", type=int, default=17, choices=MSR_ALLOWED_FRAME_COUNTS)
    parser.add_argument("--fps", type=int, default=DEFAULT_MSR_FPS)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--resize-mode", default="stretch", choices=("stretch", "fit", "fill"))
    parser.add_argument("--desc1", default="")
    parser.add_argument("--desc2", default="")
    parser.add_argument("--desc3", default="")
    parser.add_argument("--desc4", default="")
    parser.add_argument("--background-desc", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_cli(argv)
    subjects = [args.ref1, args.ref2, args.ref3, args.ref4]
    subjects = [p for p in subjects if p]
    descriptions = [args.desc1, args.desc2, args.desc3, args.desc4]

    result = build_msr_reference(
        subject_paths=subjects,
        background_path=args.background,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
        fps=args.fps,
        output_dir=args.output_dir or None,
        descriptions=descriptions,
        background_description=args.background_desc,
        save_frames=True,
        save_video=args.save_video,
        save_metadata=True,
        return_tensor=False,
        resize_mode=args.resize_mode,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
