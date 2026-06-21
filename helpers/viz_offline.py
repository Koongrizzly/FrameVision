from __future__ import annotations

"""
Offline visual renderer for the Music Clip Creator.

This module reuses the music player's VisualEngine + visual presets and
renders an audio–reactive visual track to MP4 so it can be overlaid on top
of the generated video clip.

Design notes
------------
- Uses the same PreAnalyzer + VisualEngine as the live music player.
- Runs fully offline inside the existing render worker thread – no timers,
  no extra threads, no UI dependencies.
- Supports several strategies:
  * strategy=0: one random visual for the whole track
  * strategy=1: new random visual every segment boundary (segment_boundaries)
  * strategy=2: one random visual per section type (section_map)
"""

import os
import math
import random
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PySide6.QtCore import QObject, QSize
from PySide6.QtGui import QImage

# Import analyzer + visual engine from the music player helpers
from .music import PreAnalyzer, VisualEngine


def _run_ffmpeg(cmd: list[str]) -> tuple[int, str]:
    """Small helper so we do not depend on auto_music_sync internals."""
    import subprocess

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out, _ = proc.communicate()
        return proc.returncode or 0, out or ""
    except Exception as e:
        return -1, str(e)


def _analyze_audio(path: Path, bands: int = 48, hop_ms: int = 70) -> tuple[list[int], list[list[float]], list[float]]:
    """
    Run the same offline pre-analysis as the music player, but synchronously.

    We call PreAnalyzer.run() directly instead of starting a QThread. The
    Qt Signal still works and simply calls our slot in the same thread.
    """
    worker = PreAnalyzer(path, bands=bands, hop_ms=hop_ms, parent=None)

    result: dict[str, object] = {}

    def _take(times: list, bands_mat: list, rms_list: list) -> None:
        # This slot will be called multiple times; the final call contains
        # the full arrays, so we just overwrite each time.
        result["times"] = list(times)
        result["bands"] = list(bands_mat)
        result["rms"] = list(rms_list)

    try:
        worker.ready.connect(_take)  # type: ignore[arg-type]
    except Exception:
        # If something goes wrong, just fall back to empty results.
        return [], [], []

    worker.run()

    times = result.get("times") or []
    bands_mat = result.get("bands") or []
    rms_list = result.get("rms") or []

    # Basic type safety
    if not isinstance(times, list) or not isinstance(bands_mat, list) or not isinstance(rms_list, list):
        return [], [], []
    if not times or not bands_mat or not rms_list:
        return [], [], []
    return times, bands_mat, rms_list  # type: ignore[return-value]


def _list_visual_modes(engine: VisualEngine) -> list[str]:
    """Return a list of usable visual mode names."""
    try:
        modes = [m for m in engine.available_modes() if isinstance(m, str) and m.startswith("viz:")]
        if not modes:
            # Fallback: keep at least something available
            modes = ["spectrum"]
        return modes
    except Exception:
        return ["spectrum"]


def _build_schedule_strategy0(modes: list[str], total_ms: int) -> list[tuple[int, int, str]]:
    """Single random visual for the whole track."""
    mode = random.choice(modes)
    return [(0, total_ms, mode)]


def _build_schedule_strategy1(modes: list[str], total_ms: int, segment_boundaries: Optional[list[float]]) -> list[tuple[int, int, str]]:
    """
    New random visual every segment boundary.

    segment_boundaries is a list of start times in seconds.
    """
    if not segment_boundaries:
        return _build_schedule_strategy0(modes, total_ms)

    # Ensure sorted and unique
    starts_s = sorted(set(float(x) for x in segment_boundaries))
    if not starts_s or starts_s[0] > 0.01:
        starts_s.insert(0, 0.0)

    # Append an end sentinel based on total_ms
    end_s = total_ms / 1000.0
    if starts_s[-1] >= end_s:
        starts_s = [s for s in starts_s if s < end_s]
    starts_s.append(end_s)

    schedule: list[tuple[int, int, str]] = []
    for i in range(len(starts_s) - 1):
        start_ms = int(starts_s[i] * 1000.0)
        end_ms = int(starts_s[i + 1] * 1000.0)
        if end_ms <= start_ms:
            continue
        mode = random.choice(modes)
        schedule.append((start_ms, end_ms, mode))
    if not schedule:
        return _build_schedule_strategy0(modes, total_ms)
    return schedule



def _build_schedule_strategy2(
    modes: list[str],
    total_ms: int,
    section_map: Optional[list[tuple[float, float, str]]],
    section_visual_overrides: Optional[Dict[str, Optional[str]]] = None,
) -> list[tuple[int, int, str]]:
    """
    One visual per section type.

    section_map: list of (start_sec, end_sec, kind_str)
    section_visual_overrides: optional mapping from section kind (e.g. "intro")
        to a specific visual mode name or None to explicitly disable that kind.
    """
    if not section_map:
        return _build_schedule_strategy0(modes, total_ms)

    # Normalise overrides (keys = lowercased section kinds)
    norm_overrides: Dict[str, Optional[str]] = {}
    if section_visual_overrides:
        for k, v in section_visual_overrides.items():
            if not isinstance(k, str):
                continue
            key = k.lower().strip()
            if not key:
                continue
            if v in (None, ""):
                norm_overrides[key] = None
            elif isinstance(v, str):
                norm_overrides[key] = v

    # Group by kind
    by_kind: Dict[str, list[tuple[float, float]]] = {}
    for start_s, end_s, kind in section_map:
        try:
            start_s = float(start_s)
            end_s = float(end_s)
        except Exception:
            continue
        if end_s <= start_s:
            continue
        k = str(kind or "unknown").lower()
        by_kind.setdefault(k, []).append((start_s, end_s))

    if not by_kind:
        return _build_schedule_strategy0(modes, total_ms)

    schedule: list[tuple[int, int, str]] = []
    kinds = sorted(by_kind.keys())
    for k in kinds:
        ranges = by_kind[k]
        override = norm_overrides.get(k) if norm_overrides else None

        # If an override exists and is explicitly None, skip visuals for this kind.
        if k in norm_overrides and override is None:
            continue

        if isinstance(override, str) and override:
            mode = override
        else:
            mode = random.choice(modes)

        for start_s, end_s in ranges:
            start_ms = int(start_s * 1000.0)
            end_ms = int(end_s * 1000.0)
            if end_ms <= start_ms:
                continue
            schedule.append((start_ms, end_ms, mode))

    if not schedule:
        return _build_schedule_strategy0(modes, total_ms)
    return schedule

def _build_schedule(
    modes: list[str],
    total_ms: int,
    strategy: int,
    segment_boundaries: Optional[list[float]] = None,
    section_map: Optional[list[tuple[float, float, str]]] = None,
    section_visual_overrides: Optional[Dict[str, Optional[str]]] = None,
) -> list[tuple[int, int, str]]:
    """Build a list of (start_ms, end_ms, mode_name) intervals."""
    try:
        s = int(strategy)
    except Exception:
        s = 0

    if s == 1:
        return _build_schedule_strategy1(modes, total_ms, segment_boundaries)
    if s == 2:
        return _build_schedule_strategy2(modes, total_ms, section_map, section_visual_overrides)
    return _build_schedule_strategy0(modes, total_ms)


def _render_frames(
    engine: VisualEngine,
    times_ms: List[int],
    bands_mat: List[List[float]],
    rms_list: List[float],
    fps: int,
    size: Tuple[int, int],
    out_dir: Path,
    schedule: Optional[List[tuple[int, int, str]]] = None,
    default_mode: str = "spectrum",
) -> int:
    """
    Drive the VisualEngine manually and save frames as PNGs.

    schedule: optional list of (start_ms, end_ms, mode_name). If provided,
    we switch VisualEngine mode whenever we enter a new interval. If not
    provided, we keep default_mode for the whole track.

    Returns the number of frames written.
    """
    if not times_ms or not bands_mat or not rms_list:
        return 0

    w, h = size
    engine.set_target(QSize(max(64, int(w)), max(64, int(h))))
    # Disable warm gate so visuals are immediately visible in the render.
    engine.set_enabled(True, warm_gate=False)

    # Slot to grab rendered frames
    frame_index = {"i": 0}

    def _on_frame(img: QImage) -> None:
        i = frame_index["i"]
        frame_path = out_dir / f"frame_{i:06d}.png"
        try:
            img.save(str(frame_path), "PNG")
        except Exception:
            # Ignore failed frames; continue.
            pass
        frame_index["i"] = i + 1

    try:
        engine.frameReady.connect(_on_frame)  # type: ignore[arg-type]
    except Exception:
        return 0

    total_ms = int(times_ms[-1])
    total_frames = max(1, int(math.ceil(total_ms / 1000.0 * fps)))

    # Pre-sort schedule for quick lookup
    sched = schedule or []
    sched.sort(key=lambda t: t[0])
    current_mode = None

    def _mode_for_time(t_ms: int) -> str:
        if not sched:
            return default_mode
        for start_ms, end_ms, mode_name in sched:
            if start_ms <= t_ms < end_ms:
                return mode_name
        return default_mode

    idx = 0
    n = len(times_ms)

    import time as _time

    for fi in range(total_frames):
        t_ms = int(fi * 1000.0 / fps)
        # Advance index while we have timestamps <= current time
        while idx + 1 < n and times_ms[idx + 1] <= t_ms:
            idx += 1
        bands = bands_mat[idx]
        rms = rms_list[idx]

        # Switch visual mode if needed
        mode_name = _mode_for_time(t_ms)
        if mode_name != current_mode:
            try:
                engine.set_mode(mode_name)
                current_mode = mode_name
            except Exception:
                # If a particular mode fails, leave engine as-is.
                pass

        # Align VisualEngine's internal time with the audio timeline
        t_sec = t_ms / 1000.0
        try:
            engine.start_time = _time.time() - float(t_sec)
        except Exception:
            pass

        try:
            engine.inject_levels(bands)
            engine.inject_rms(float(rms))
        except Exception:
            # If injection fails, continue so at least some frames render.
            pass

        # Tick once – this will emit frameReady and our slot will save the frame.
        try:
            engine._tick()  # type: ignore[attr-defined]
        except Exception:
            break

    return frame_index["i"]


def render_visual_track(
    audio_path: str,
    out_video: str,
    ffmpeg_bin: Optional[str] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30,
    strategy: int = 0,
    segment_boundaries: Optional[list[float]] = None,
    section_map: Optional[list[tuple[float, float, str]]] = None,
    section_visual_overrides: Optional[Dict[str, Optional[str]]] = None,
) -> bool:
    """
    High-level helper used by the Music Clip Creator.

    Parameters
    ----------
    audio_path:
        Path to the music track used for analysis (same as the clip's audio).
    out_video:
        Destination MP4 path for the rendered visual track.
    ffmpeg_bin:
        ffmpeg executable path. If None, uses plain "ffmpeg".
    resolution:
        Target (width, height) of the visuals, typically the same as the
        final video resolution.
    fps:
        Frame rate for the generated visuals.
    strategy:
        0 = single visual for the whole track
        1 = new random visual every segment (segment_boundaries)
        2 = one visual per section type (section_map)
    segment_boundaries:
        List of segment start times in seconds (for strategy=1).
    section_map:
        List of (start_sec, end_sec, kind_str) entries (for strategy=2).

    Returns True on success, False on failure.
    """
    audio = Path(audio_path)
    if not audio.exists():
        return False

    ffmpeg = ffmpeg_bin or "ffmpeg"

    # Step 1: offline analysis
    times_ms, bands_mat, rms_list = _analyze_audio(audio)
    if not times_ms:
        return False

    total_ms = int(times_ms[-1])

    # Step 2: instantiate the visual engine and build schedule
    engine = VisualEngine(parent=None)
    modes = _list_visual_modes(engine)
    default_mode = modes[0] if modes else "spectrum"
    schedule = _build_schedule(
        modes=modes,
        total_ms=total_ms,
        strategy=strategy,
        segment_boundaries=segment_boundaries,
        section_map=section_map,
        section_visual_overrides=section_visual_overrides,
    )

    # Step 3: render visuals into a temporary frames directory
    work_dir = Path(tempfile.mkdtemp(prefix="fv_viz_render_"))
    frames_dir = work_dir / "frames"
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False

    try:
        frame_count = _render_frames(
            engine,
            times_ms,
            bands_mat,
            rms_list,
            fps=fps,
            size=resolution,
            out_dir=frames_dir,
            schedule=schedule,
            default_mode=default_mode,
        )
        if frame_count <= 0:
            return False

        # Step 4: encode frames into an MP4 using ffmpeg
        pattern = str(frames_dir / "frame_%06d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-r",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            out_video,
        ]
        code, _out = _run_ffmpeg(cmd)
        if code != 0 or (not os.path.exists(out_video)):
            return False
        return True
    finally:
        # Always try to clean up frames; temp root may be left on disk if removal fails.
        import shutil

        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
