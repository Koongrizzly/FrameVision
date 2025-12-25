
"""
Music Clip Creator / Auto Music Sync for FrameVision

Features
--------
- Audio-driven beat + energy analysis (lightweight RMS-based)
- Works with a single long video OR a folder of short clips
- FX Level: Minimal / Moderate / High
- Microclips:
    - Off
    - Only in chorus / drops
    - Whole track
- Transitions:
    - Soft fades
    - Hard cuts
    - Mixed (fades + flash cuts)
- Clip order modes for folders:
    - Random (avoids immediate repeats)
    - Sequential
    - Shuffle (no repeats until all clips used)
- Optional fixed random seed for repeatable edits
- Resolution strategies for mixed clip sets
- Timestamped output name to avoid overwrites:
  <audio-base-name>_clip_YYYYMMDD_HHMM.mp4

Integration from tools_tab.py:

    from helpers.auto_music_sync import install_auto_music_sync_tool
    sec_musicclip = CollapsibleSection("Music Clip Creator", expanded=False)
    install_auto_music_sync_tool(self, sec_musicclip)
"""

from __future__ import annotations

import os
import sys
import math
import json
import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QMessageBox,
    QSizePolicy,
    QGroupBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
)


# ---------------------------- small helpers --------------------------------


def _find_ffmpeg_from_env() -> str:
    env_ffmpeg = os.environ.get("FV_FFMPEG")
    if env_ffmpeg and os.path.exists(env_ffmpeg):
        return env_ffmpeg

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "presets", "bin", "ffmpeg.exe"),
        os.path.join(here, "..", "presets", "bin", "ffmpeg"),
        "ffmpeg.exe",
        "ffmpeg",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c

    return "ffmpeg"


def _find_ffprobe_from_env() -> str:
    env_ffprobe = os.environ.get("FV_FFPROBE")
    if env_ffprobe and os.path.exists(env_ffprobe):
        return env_ffprobe

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "presets", "bin", "ffprobe.exe"),
        os.path.join(here, "..", "presets", "bin", "ffprobe"),
        "ffprobe.exe",
        "ffprobe",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c

    return "ffprobe"


def _run_ffmpeg(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        out, _ = proc.communicate()
        return proc.returncode, out or ""
    except Exception as e:
        return 1, f"Failed to run ffmpeg: {e!r}"


def _ffprobe_duration(ffprobe: str, path: str) -> Optional[float]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(out.strip())
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------- data classes ----------------------------------


@dataclass
class Beat:
    time: float
    strength: float
    kind: str  # "major" / "minor"


@dataclass
class Section:
    start: float
    end: float
    kind: str  # "intro", "verse", "chorus", "drop", "break"


@dataclass
class ClipSource:
    path: str
    duration: float


@dataclass
class TimelineSegment:
    clip_path: str
    clip_start: float
    duration: float
    effect: str
    energy_class: str  # "low", "mid", "high"
    transition: str    # "none", "fade", "flashcut"


@dataclass
class MusicAnalysisConfig:
    sensitivity: int = 5


@dataclass
class MusicAnalysisResult:
    beats: List[Beat]
    sections: List[Section]
    duration: float


# --------------------------- music analysis --------------------------------


def analyze_music(audio_path: str, ffmpeg: str, config: Optional[MusicAnalysisConfig] = None) -> MusicAnalysisResult:
    """Very lightweight beat + energy analysis using ffmpeg + PCM RMS."""
    tmpdir = tempfile.mkdtemp(prefix="fv_mclip_an_")
    wav_path = os.path.join(tmpdir, "mono.wav")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-f",
        "wav",
        wav_path,
    ]
    code, out = _run_ffmpeg(cmd)
    if code != 0 or not os.path.exists(wav_path):
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("Failed to convert audio for analysis:\n" + out)

    import wave
    import struct

    with wave.open(wav_path, "rb") as wf:
        fr = wf.getframerate()
        total_frames = wf.getnframes()
        duration = total_frames / float(fr)

        win_size = int(fr * 0.05)  # 50ms windows
        values = []
        times = []
        read = 0
        while read < total_frames:
            n = min(win_size, total_frames - read)
            raw = wf.readframes(n)
            read += n
            if not raw:
                break
            count = len(raw) // 2  # 16-bit
            if count == 0:
                rms = 0.0
            else:
                fmt = "<" + "h" * count
                samples = struct.unpack(fmt, raw)
                acc = 0.0
                for s in samples:
                    acc += (s / 32768.0) ** 2
                rms = math.sqrt(acc / count)
            values.append(rms)
            times.append(len(values) * (n / float(fr)))

    shutil.rmtree(tmpdir, ignore_errors=True)

    if not values:
        raise RuntimeError("Audio analysis produced no samples.")

    max_v = max(values) or 1.0
    norm = [v / max_v for v in values]

    mean = sum(norm) / len(norm)
    var = sum((v - mean) ** 2 for v in norm) / len(norm)
    std = math.sqrt(var)

    # Apply beat sensitivity from config (1 = fewer beats, 10 = more beats)
    if config is not None:
        try:
            raw_val = float(config.sensitivity)
        except Exception:
            raw_val = 10.0
    else:
        raw_val = 10.0
    # Slider uses 2..20 -> map to 1.0..10.0 in 0.5 steps
    sens = max(1.0, min(10.0, raw_val / 2.0))
    # Map sensitivity linearly: 1 -> ~1.24x thresholds (fewer beats), 10 -> ~0.70x (more beats)
    scale = 1.0 - (sens - 5.0) * 0.06
    if scale < 0.6:
        scale = 0.6
    elif scale > 1.4:
        scale = 1.4
    beat_thr = mean + std * 0.7 * scale
    major_thr = mean + std * 1.4 * scale

    beats: List[Beat] = []
    last_peak = -9999
    min_dist = int(0.15 / 0.05)  # 150ms

    for i in range(1, len(norm) - 1):
        v = norm[i]
        if v < beat_thr:
            continue
        if v >= norm[i - 1] and v >= norm[i + 1]:
            if i - last_peak < min_dist:
                if beats and v > beats[-1].strength:
                    beats[-1] = Beat(
                        time=times[i],
                        strength=v,
                        kind="major" if v >= major_thr else "minor",
                    )
                    last_peak = i
                continue
            kind = "major" if v >= major_thr else "minor"
            beats.append(Beat(time=times[i], strength=v, kind=kind))
            last_peak = i

    # If we have beats but there's a long quiet tail with no peaks,
    # synthesize gentle extra beats in the tail so quiet / outro parts
    # still have material for the timeline.
    if beats:
        tail = duration - beats[-1].time
        if tail > 5.0:
            # Place minor beats at regular spacing in the tail.
            # Use a stable pattern so results remain predictable.
            num_virtual = max(2, min(16, int(tail / 1.0)))
            if num_virtual > 0:
                spacing = tail / (num_virtual + 1)
                t = beats[-1].time + spacing
                while t < duration - 0.25:
                    beats.append(
                        Beat(
                            time=t,
                            strength=mean,
                            kind="minor",
                        )
                    )
                    t += spacing

    # Ensure beats are sorted in time in case virtual beats were appended
    beats.sort(key=lambda b: b.time)

    # Energy profile in 1s windows
    win1 = int(1.0 / 0.05)
    e_vals = []
    e_times = []
    for i in range(0, len(norm), win1):
        chunk = norm[i : i + win1]
        if not chunk:
            continue
        e_vals.append(sum(chunk) / len(chunk))
        e_times.append(i * 0.05)

    if not e_vals:
        e_vals = [mean]
        e_times = [0.0]

    e_mean = sum(e_vals) / len(e_vals)
    e_var = sum((v - e_mean) ** 2 for v in e_vals) / len(e_vals)
    e_std = math.sqrt(e_var)

    low_thr = e_mean - 0.4 * e_std
    high_thr = e_mean + 0.4 * e_std

    sections: List[Section] = []
    cur_kind = None
    cur_start = 0.0

    def flush(end_t: float):
        nonlocal cur_kind, cur_start
        if cur_kind is None:
            return
        sections.append(Section(start=cur_start, end=end_t, kind=cur_kind))

    for i, v in enumerate(e_vals):
        t = e_times[i]
        if v < low_thr:
            kind = "intro_or_break"
        elif v > high_thr:
            kind = "chorus_or_drop"
        else:
            kind = "verse_or_mid"
        if cur_kind is None:
            cur_kind = kind
            cur_start = t
        elif kind != cur_kind:
            flush(t)
            cur_kind = kind
            cur_start = t
    flush(duration)

    labelled: List[Section] = []
    chorus_seen = False
    for idx, s in enumerate(sections):
        if idx == 0:
            k = "intro"
        elif s.kind == "chorus_or_drop":
            k = "chorus" if not chorus_seen else "drop"
            chorus_seen = True
        elif s.kind == "intro_or_break":
            k = "break"
        else:
            k = "verse"
        labelled.append(Section(start=s.start, end=s.end, kind=k))

    return MusicAnalysisResult(beats=beats, sections=labelled, duration=duration)


# --------------------------- timeline builder ------------------------------


def discover_video_sources(video_input: str, ffprobe: str) -> List[ClipSource]:
    """Return a list of ClipSource from a file, directory, or a '|' separated list of files."""
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
    sources: List[ClipSource] = []

    # Multiple explicit files
    if "|" in video_input:
        parts = [p.strip() for p in video_input.split("|") if p.strip()]
        for raw in parts:
            path = os.path.abspath(raw)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in exts:
                continue
            dur = _ffprobe_duration(ffprobe, path)
            if dur and dur > 0:
                sources.append(ClipSource(path=path, duration=dur))
        return sources

    # Single file or directory
    video_input = os.path.abspath(video_input)

    if os.path.isdir(video_input):
        for name in sorted(os.listdir(video_input)):
            path = os.path.join(video_input, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in exts:
                continue
            dur = _ffprobe_duration(ffprobe, path)
            if dur and dur > 0:
                sources.append(ClipSource(path=path, duration=dur))
    elif os.path.isfile(video_input):
        dur = _ffprobe_duration(ffprobe, video_input)
        if dur and dur > 0:
            sources.append(ClipSource(path=video_input, duration=dur))

    return sources


def build_timeline(
    analysis: MusicAnalysisResult,
    sources: List[ClipSource],
    fx_level: str,
    microclip_mode: int,
    beats_per_segment: int,
    transition_mode: int,
    clip_order_mode: int,
    force_full_length: bool,
    seed_enabled: bool,
    seed_value: int,
    transition_random: bool,
    transition_modes_enabled: Optional[List[int]],
) -> List[TimelineSegment]:
    """Build a musical timeline from beats, sections and sources.

    fx_level:
        "minimal", "moderate", "high"

    microclip_mode:
        0 = off
        1 = only in high-energy sections (chorus / drops)
        2 = whole track

    transition_mode:
        0 = soft fades
        1 = hard cuts
        2 = mixed (fades + flash cuts sometimes)

    clip_order_mode:
        0 = random (default)
        1 = sequential
        2 = shuffle (no repeats until all clips used)
    """
    if not sources:
        return []

    if seed_enabled:
        # Deterministic run: same seed + same inputs = same edit.
        random.seed(int(seed_value) & 0xFFFFFFFF)
    else:
        # Non-deterministic run: reseed with high-entropy so each render
        # starts from a fresh random sequence, even inside the same session.
        try:
            random.seed(os.urandom(16))
        except Exception:
            # Fallback: time-based seed
            from datetime import datetime as _dt
            random.seed(int(_dt.now().timestamp() * 1000) & 0xFFFFFFFF)

    # Determine which transition modes are allowed for randomization
    allowed_modes: List[int] = []
    if transition_modes_enabled:
        for m in transition_modes_enabled:
            if 0 <= m <= 7 and m not in allowed_modes:
                allowed_modes.append(m)
    if not allowed_modes:
        allowed_modes.append(transition_mode)

    beats = analysis.beats
    if len(beats) < 4:
        return [
            TimelineSegment(
                clip_path=sources[0].path,
                clip_start=0.0,
                duration=analysis.duration,
                effect="none",
                energy_class="mid",
                transition="none" if transition_mode == 1 else "fade",
            )
        ]

    def energy_class(time_t: float) -> str:
        for s in analysis.sections:
            if s.start <= time_t < s.end:
                if s.kind in ("chorus", "drop"):
                    return "high"
                if s.kind in ("intro", "break"):
                    return "low"
                return "mid"
        return "mid"

    # Beat intervals
    # We build intervals between consecutive beats (plus a tail segment up to the
    # end of the track), then group those intervals into base segments according
    # to beats_per_segment. For each grouped segment we also track how many beats
    # it spans so we can detect "calm" regions (large gaps between beats).
    intervals = []
    for i in range(len(beats) - 1):
        t0 = beats[i].time
        t1 = beats[i + 1].time
        if t1 > t0:
            intervals.append((t0, t1))
    if beats[-1].time < analysis.duration:
        intervals.append((beats[-1].time, analysis.duration))

    grouped = []  # list of (start_time, end_time, beat_count)
    cur_start = None
    cur_count = 0
    for (t0, t1) in intervals:
        if cur_start is None:
            cur_start = t0
            cur_count = 1
        else:
            cur_count += 1
        if cur_count >= max(1, beats_per_segment):
            grouped.append((cur_start, t1, cur_count))
            cur_start = None
            cur_count = 0
    if cur_start is not None:
        # Tail segment that didn't reach beats_per_segment; still keep its count.
        grouped.append((cur_start, analysis.duration, max(1, cur_count)))

    num_sources = len(sources)
    last_source_idx: Optional[int] = None
    shuffle_pool = list(range(num_sources))
    random.shuffle(shuffle_pool)
    shuffle_pos = 0
    clip_offsets = {i: 0.0 for i in range(num_sources)}

    def _pick_source_index() -> int:
        nonlocal last_source_idx, shuffle_pool, shuffle_pos
        if num_sources == 1:
            idx = 0
        elif clip_order_mode == 1:  # sequential
            if last_source_idx is None:
                idx = 0
            else:
                idx = (last_source_idx + 1) % num_sources
        elif clip_order_mode == 2:  # shuffle, no repeats until all used
            if shuffle_pos >= len(shuffle_pool):
                shuffle_pool = list(range(num_sources))
                random.shuffle(shuffle_pool)
                shuffle_pos = 0
            idx = shuffle_pool[shuffle_pos]
            shuffle_pos += 1
        else:  # 0 = random (no immediate repeat if possible)
            candidates = list(range(num_sources))
            if last_source_idx is not None and num_sources > 1:
                try:
                    candidates.remove(last_source_idx)
                except ValueError:
                    pass
            idx = random.choice(candidates)
        last_source_idx = idx
        return idx

    def pick_region(length: float) -> Tuple[str, float]:
        idx = _pick_source_index()
        src = sources[idx]
        offset = clip_offsets.get(idx, 0.0)
        if offset + length > src.duration:
            offset = 0.0
        start = offset
        clip_offsets[idx] = offset + length
        return src.path, max(0.0, start)

    segments: List[TimelineSegment] = []

    # Smart energy-aware pacing:
    # In calm regions (few / widely spaced beats or low-energy sections),
    # we prefer longer shots so the video can breathe instead of cutting
    # every beat. In active regions we keep the existing microclip logic.
    CALM_GAP_THRESHOLD = 1.4   # seconds between beats to consider section "calm"
    CALM_MIN_LEN = 3.0         # preferred minimum length for calm shots
    CALM_MAX_LEN = 7.0         # preferred maximum length for calm shots

    for (t0, t1, beat_count) in grouped:
        dur = max(0.35, t1 - t0)
        center = 0.5 * (t0 + t1)
        energy = energy_class(center)

        # Average spacing between beats in this segment (proxy for beat density)
        avg_gap = dur / max(1, beat_count)
        is_calm = (avg_gap >= CALM_GAP_THRESHOLD) or (energy == "low")

        # Base segment length
        if is_calm:
            # Calm / break-like section: use longer, more relaxed shots.
            if dur <= CALM_MIN_LEN:
                seg_len = dur
            else:
                max_len = min(CALM_MAX_LEN, dur)
                seg_len = random.uniform(CALM_MIN_LEN, max_len)
        else:
            # Original microclip logic for active sections.
            if microclip_mode == 0:
                seg_len = min(dur, 4.0)
            elif microclip_mode == 1:
                if energy == "high":
                    seg_len = max(0.3, min(0.8, dur, random.uniform(0.3, 0.8)))
                else:
                    seg_len = min(dur, 4.0)
            else:
                if energy == "high":
                    base_min, base_max = 0.25, 0.7
                elif energy == "mid":
                    base_min, base_max = 0.4, 1.0
                else:
                    base_min, base_max = 0.6, 1.3
                seg_len = max(
                    base_min,
                    min(base_max, dur, random.uniform(base_min, base_max)),
                )

        clip_path, clip_start = pick_region(seg_len)

        # FX selection by level and energy
        # Now supports a small library of segment FX:
        #   - zoom        : gentle punch-in
        #   - flash       : small brightness pop
        #   - rgb_split   : chromatic aberration
        #   - vhs         : VHS-style noise + scanlines
        #   - motion_blur : mild global blend
        effect = "none"
        r = random.random()

        if fx_level == "minimal":
            # Very subtle: only occasional zoom on strong peaks
            if energy == "high" and r < 0.18:
                effect = "zoom"

        elif fx_level == "moderate":
            if energy == "high":
                if r < 0.25:
                    effect = "zoom"
                elif r < 0.45:
                    effect = "flash"
                elif r < 0.65:
                    effect = "rgb_split"
                elif r < 0.78:
                    effect = "vhs"
                elif r < 0.90:
                    effect = "motion_blur"
            elif energy == "mid":
                if r < 0.20:
                    effect = random.choice(["zoom", "rgb_split"])

        else:  # high
            if energy == "high":
                if r < 0.25:
                    effect = "zoom"
                elif r < 0.45:
                    effect = "flash"
                elif r < 0.65:
                    effect = "rgb_split"
                elif r < 0.80:
                    effect = "vhs"
                else:
                    effect = "motion_blur"
            elif energy == "mid":
                if r < 0.30:
                    effect = random.choice(["zoom", "rgb_split", "flash"])
                elif r < 0.45:
                    effect = "vhs"
                elif r < 0.60:
                    effect = "motion_blur"
            else:
                if r < 0.22:
                    effect = random.choice(["zoom", "rgb_split"])

        # Transition choice (fade-safe, using a small transition effect library)
        if transition_random and allowed_modes:
            mode_for_segment = random.choice(allowed_modes)
        else:
            mode_for_segment = transition_mode

        # 0 = Slide   -> currently behaves like a gentle cut (no extra filter)
        # 1 = Hard cuts
        # 2 = Mixed   -> mix of white flash + cuts
        # 3 = Creative mix -> mix of color flashes + subtle whip pans + cuts
        # 4 = Zoom pulse / scale pulse (legacy)
        # 5 = RGB split / chromatic aberration (legacy)
        # 6 = VHS noise + scanlines (legacy)
        # 7 = Motion blur boost (legacy)
        if mode_for_segment == 1:  # Hard cuts
            transition = "none"
        elif mode_for_segment == 2:  # Mixed (white flash + cuts)
            if energy == "high" and random.random() < 0.5:
                transition = "flashcut"      # white flash
            else:
                transition = "none"
        elif mode_for_segment == 3:  # Creative mix (color flashes + subtle whip)
            if energy == "high":
                # On strong peaks, alternate between color flash and whip-like motion
                if random.random() < 0.6:
                    transition = "flashcolor"
                else:
                    transition = "whip"
            elif energy == "mid":
                r = random.random()
                if r < 0.3:
                    transition = "flashcolor"
                elif r < 0.5:
                    transition = "whip"
                else:
                    transition = "none"
            else:
                transition = "none"
        elif mode_for_segment == 4:  # Zoom pulse / scale pulse
            transition = "t_zoom_pulse"
        elif mode_for_segment == 5:  # RGB split / chromatic aberration
            transition = "t_rgb_split"
        elif mode_for_segment == 6:  # VHS noise + scanlines
            transition = "t_vhs"
        elif mode_for_segment == 7:  # Motion blur boost
            transition = "t_motion_blur"
        else:  # 0 = Slide (soft cut for now)
            transition = "none"
        segments.append(
            TimelineSegment(
                clip_path=clip_path,
                clip_start=clip_start,
                duration=seg_len,
                effect=effect,
                energy_class=energy,
                transition=transition,
            )
        )


    # Optionally extend timeline so total video duration matches full music length
    if force_full_length and segments:
        total_video = sum(seg.duration for seg in segments)
        target = analysis.duration
        # Only extend (never shrink); allow a small tolerance before extending.
        if total_video < target * 0.98 and target > 0:
            remaining = target - total_video
            # Use a conservative average length for filler segments
            avg_len = max(0.5, min(2.5, total_video / len(segments)))
            def _make_filler_segment(start_fraction: float, length: float) -> TimelineSegment:
                # Clamp logical time to the track duration
                t_center = target * min(max(start_fraction, 0.0), 1.0)
                energy = energy_class(t_center)
                clip_path, clip_start = pick_region(length)
                # Use similar transition logic for fillers, but a bit softer
                if transition_random and allowed_modes:
                    mode_for_segment = random.choice(allowed_modes)
                else:
                    mode_for_segment = transition_mode

                if mode_for_segment == 1:
                    transition = "none"
                elif mode_for_segment == 2:
                    if energy == "high" and random.random() < 0.4:
                        transition = "flashcut"
                    else:
                        transition = "none"
                elif mode_for_segment == 3:
                    if energy == "high" and random.random() < 0.4:
                        transition = "flashcolor"
                    elif energy == "mid" and random.random() < 0.2:
                        transition = random.choice(["flashcolor", "whip"])
                    else:
                        transition = "none"
                else:
                    transition = "none"
                return TimelineSegment(
                    clip_path=clip_path,
                    clip_start=clip_start,
                    duration=length,
                    effect="none",
                    energy_class=energy,
                    transition=transition,
                )

            # Add filler segments towards the end of the track until we cover the gap
            while remaining > 0.05:
                seg_len = min(avg_len, remaining)
                start_frac = (target - remaining + seg_len * 0.5) / target
                filler = _make_filler_segment(start_frac, seg_len)
                segments.append(filler)
                remaining -= seg_len

    return segments

# ---------------------------- render worker --------------------------------


class RenderWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        audio_path: str,
        output_dir: str,
        analysis: MusicAnalysisResult,
        segments: List[TimelineSegment],
        ffmpeg: str,
        ffprobe: str,
        target_resolution: Optional[Tuple[int, int]],
        transition_mode: int,
        intro_fade: bool,
        outro_fade: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.analysis = analysis
        self.segments = segments
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.target_resolution = target_resolution
        self.transition_mode = transition_mode
        self.intro_fade = intro_fade
        self.outro_fade = outro_fade


    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as e:
            self.failed.emit(str(e))

    def _run_impl(self) -> None:
        if not self.segments:
            raise RuntimeError("Timeline is empty.")

        _ensure_dir(self.output_dir)
        tmpdir = tempfile.mkdtemp(prefix="fv_mclip_render_")

        try:
            parts = []
            n = len(self.segments)
            for i, seg in enumerate(self.segments):
                pct = int(5 + 80 * (i / max(1, n)))
                self.progress.emit(pct, f"Rendering segment {i+1}/{n}...")
                out_part = os.path.join(tmpdir, f"part_{i:04d}.mp4")

                vf_parts: List[str] = []
                base_vf_parts: List[str] = []

                if self.target_resolution is not None:
                    w, h = self.target_resolution
                    base_vf_parts = [
                        f"scale={w}:{h}:force_original_aspect_ratio=decrease",
                        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
                    ]
                    vf_parts.extend(base_vf_parts)

                # Segment-level FX (per-segment visual styles)
                if seg.effect == "zoom":
                    # Gentle punch-in (about 8%)
                    vf_parts.append("scale=iw*1.08:ih*1.08,crop=iw/1.08:ih/1.08")
                elif seg.effect == "flash":
                    # Small global brightness lift for the whole segment
                    vf_parts.append("eq=brightness=0.18")
                elif seg.effect == "rgb_split":
                    # Chromatic aberration / RGB split
                    vf_parts.append("chromashift=cbh=-4:cbv=-4:crh=4:crv=4")
                elif seg.effect == "vhs":
                    # VHS noise + scanlines (downscale + upscale with neighbor sampling)
                    vf_parts.append("noise=alls=15:allf=t+u,scale=iw:ih/2:flags=neighbor,scale=iw:ih:flags=neighbor")
                elif seg.effect == "motion_blur":
                    # Mild motion blur via temporal blend
                    vf_parts.append("tblend=all_mode=average,framestep=1")

                # Transitions using a small effect library (all short and fade-safe)
                if seg.transition == "flashcut":
                    # White flash: short brightness pop at the start (~60–120ms).
                    flash_d = max(0.06, min(0.12, seg.duration / 6.0))
                    vf_parts.append(
                        f"eq=brightness=0.45:enable='between(t,0,{flash_d:.3f})'"
                    )
                elif seg.transition == "flashcolor":
                    # Colored flash: short hue-shifted flash at the start.
                    flash_d = max(0.06, min(0.12, seg.duration / 6.0))
                    # Rotate hue by a fixed offset per segment index for reproducible variety.
                    hue_shift = (i * 37) % 360  # pseudo "16-ish" cycle
                    vf_parts.append(
                        f"eq=brightness=0.35:enable='between(t,0,{flash_d:.3f})',"
                        f"hue=h={hue_shift}:enable='between(t,0,{flash_d:.3f})'"
                    )
                elif seg.transition == "whip":
                    # Zoom pulse / scale pulse: beat-style zoom using scale+crop.
                    # This is a visual effect only (no fades), safe across ffmpeg builds.
                    # Use a fixed gentle pulse period (~0.5s) since we don't have beat_period here.
                    base_period = 0.5
                    zoom_amount = 0.06
                    zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
                    vf_parts.append(
                        f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih"
                    )
                elif seg.transition == "t_zoom_pulse":
                    # Legacy zoom pulse / scale pulse from the old Auto Music Sync tool
                    base_period = 0.5
                    zoom_amount = 0.03
                    zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
                    vf_parts.append(
                        f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih"
                    )
                elif seg.transition == "t_rgb_split":
                    # Legacy RGB split / chromatic aberration transition
                    vf_parts.append("chromashift=cbh=-4:cbv=-4:crh=4:crv=4")
                elif seg.transition == "t_vhs":
                    # Legacy VHS noise + scanlines transition
                    vf_parts.append("noise=alls=15:allf=t+u,scale=iw:ih/2:flags=neighbor,scale=iw:ih:flags=neighbor")
                elif seg.transition == "t_motion_blur":
                    # Legacy motion blur boost transition
                    vf_parts.append("tblend=all_mode=average,framestep=1")
                else:
                    # "none" and anything else fall back to a hard cut (no extra filters here)
                    pass

                vf_arg = ",".join(vf_parts) if vf_parts else None
                safe_vf_arg = ",".join(base_vf_parts) if base_vf_parts else None

                # Base ffmpeg command for this segment (input, trim, no audio)
                base_cmd = [
                    self.ffmpeg,
                    "-y",
                    "-i",
                    seg.clip_path,
                    "-ss",
                    f"{seg.clip_start:.3f}",
                    "-t",
                    f"{seg.duration:.3f}",
                    "-an",
                ]

                # Common encoding settings
                encode_args = [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-r",
                    "30",
                    out_part,
                ]

                # First attempt: full filter chain (resolution + FX + transitions)
                cmd = list(base_cmd)
                if vf_arg:
                    cmd += ["-vf", vf_arg]
                cmd += encode_args
                code, out = _run_ffmpeg(cmd)

                # Second attempt: only resolution scaling/padding (if any)
                if (code != 0 or not os.path.exists(out_part)) and safe_vf_arg and safe_vf_arg != vf_arg:
                    try:
                        if os.path.exists(out_part):
                            os.remove(out_part)
                    except Exception:
                        pass
                    cmd = list(base_cmd)
                    cmd += ["-vf", safe_vf_arg]
                    cmd += encode_args
                    code, out = _run_ffmpeg(cmd)

                # Final attempt: no video filters at all
                if code != 0 or not os.path.exists(out_part):
                    try:
                        if os.path.exists(out_part):
                            os.remove(out_part)
                    except Exception:
                        pass
                    cmd = list(base_cmd)
                    cmd += encode_args
                    code, out = _run_ffmpeg(cmd)

                if code != 0 or not os.path.exists(out_part):
                    raise RuntimeError(f"ffmpeg failed for segment {i+1}:\n" + out)
                parts.append(out_part)
            self.progress.emit(90, "Concatenating segments...")
            concat_list = os.path.join(tmpdir, "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for p in parts:
                    safe = p.replace("\\", "/")
                    f.write(f"file '{safe}'\n")

            concat_video = os.path.join(tmpdir, "video_concat.mp4")
            # Re-encode concatenated video to ensure stable timestamps and avoid
            # black/freezing issues in some players. We only have video here;
            # audio is added in the final mux step.
            cmd = [
                self.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                concat_video,
            ]
            code, out = _run_ffmpeg(cmd)
            if code != 0 or not os.path.exists(concat_video):
                raise RuntimeError("Failed to concat segments:\n" + out)

            # Optional intro/outro styling (fade to/from black)
            video_for_mux = concat_video
            if self.intro_fade or self.outro_fade:
                self.progress.emit(93, "Applying intro/outro styling...")
                dur = _ffprobe_duration(self.ffprobe, concat_video)
                if dur and dur > 0.2:
                    styled_video = os.path.join(tmpdir, "video_styled.mp4")
                    vf_parts = []
                    # Fade-in duration: up to 15%% of track, clamped between 0.2s and 1.0s
                    if self.intro_fade:
                        fade_in_d = 0.8
                        fade_in_d = min(fade_in_d, max(0.2, dur * 0.15))
                        vf_parts.append(f"fade=t=in:st=0:d={fade_in_d:.3f}")
                    # Fade-out duration: up to 15%% of track, clamped between 0.2s and 1.2s
                    if self.outro_fade:
                        fade_out_d = 0.8
                        fade_out_d = min(fade_out_d, max(0.2, dur * 0.15))
                        start_out = max(0.0, dur - fade_out_d)
                        vf_parts.append(f"fade=t=out:st={start_out:.3f}:d={fade_out_d:.3f}")
                    if vf_parts:
                        vf_chain = ",".join(vf_parts)
                        cmd = [
                            self.ffmpeg,
                            "-y",
                            "-i",
                            concat_video,
                            "-vf",
                            vf_chain,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "18",
                            "-pix_fmt",
                            "yuv420p",
                            styled_video,
                        ]
                        code, out = _run_ffmpeg(cmd)
                        if code == 0 and os.path.exists(styled_video):
                            video_for_mux = styled_video

            # mux with audio
            self.progress.emit(95, "Merging with audio...")
            base = os.path.splitext(os.path.basename(self.audio_path))[0]
            ts = datetime.now().strftime("%d%m%H%M")  # ddmmhhmm to avoid overwrites
            safe_base = base.strip().replace(" ", "_")
            out_name = f"{safe_base}_clip_{ts}.mp4"
            out_final = os.path.join(self.output_dir, out_name)
            cmd = [
                self.ffmpeg,
                "-y",
                "-i",
                video_for_mux,
                "-i",
                self.audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                out_final,
            ]
            code, out = _run_ffmpeg(cmd)
            if code != 0 or not os.path.exists(out_final):
                raise RuntimeError("Failed to mux video and audio:\n" + out)

            self.progress.emit(100, "Done.")
            self.finished_ok.emit(out_final)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------- main widget ----------------------------------


class AutoMusicSyncWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ffmpeg = _find_ffmpeg_from_env()
        self._ffprobe = _find_ffprobe_from_env()
        self._analysis: Optional[MusicAnalysisResult] = None
        self._analysis_config = MusicAnalysisConfig()
        self._worker: Optional[RenderWorker] = None
        # Enabled transition styles for randomization (indices of combo_transitions)
        self._enabled_transition_modes = {0, 1, 2, 3, 4, 5, 6, 7}

        # Settings for remembering last paths & options
        self._settings = QSettings("FrameVision", "MusicClipCreator")

        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setSpacing(4)

        # audio
        self.edit_audio = QLineEdit(self)
        btn_a = QPushButton("Browse...", self)
        row_a = QHBoxLayout()
        row_a.addWidget(self.edit_audio, 1)
        row_a.addWidget(btn_a)
        form.addRow("Music / Audio:", row_a)

        # video (file or folder)
        self.edit_video = QLineEdit(self)
        btn_vf = QPushButton("Video file...", self)
        btn_vd = QPushButton("Clip folder...", self)
        btn_vmf = QPushButton("Clip files...", self)
        row_v = QHBoxLayout()
        row_v.addWidget(self.edit_video, 1)
        row_v.addWidget(btn_vf)
        row_v.addWidget(btn_vd)
        row_v.addWidget(btn_vmf)
        form.addRow("Video input:", row_v)

        # output
        self.edit_output = QLineEdit(self)
        btn_o = QPushButton("Browse...", self)
        row_o = QHBoxLayout()
        row_o.addWidget(self.edit_output, 1)
        row_o.addWidget(btn_o)
        form.addRow("Output folder:", row_o)

        root.addLayout(form)

        # options
        box_opts = QGroupBox("Options", self)
        opts = QVBoxLayout(box_opts)

        # FX level
        row_fx = QHBoxLayout()
        row_fx.addWidget(QLabel("FX Level:", self))
        self.combo_fx = QComboBox(self)
        self.combo_fx.addItems(["Minimal", "Moderate", "High"])
        self.combo_fx.setToolTip(
            "Choose how active the visual effects should be:\n"
            "- Minimal: clean cuts with subtle accents.\n"
            "- Moderate: more zoom/flash on peaks.\n"
            "- High: strongest visual activity, best with many short clips."
        )
        row_fx.addWidget(self.combo_fx, 1)
        row_fx.addStretch(1)
        opts.addLayout(row_fx)

        # microclip toggles
        row_micro = QHBoxLayout()
        self.check_micro_chorus = QCheckBox("Microclips in chorus/drops only", self)
        self.check_micro_chorus.setToolTip(
            "Short energetic microclips only during high-energy sections\n"
            "(chorus / drops). Verses and intros stay calmer."
        )
        row_micro.addWidget(self.check_micro_chorus)

        self.check_micro_all = QCheckBox("Microclips for the whole track", self)
        self.check_micro_all.setToolTip(
            "Microclips are used throughout the entire song. Great with many\n"
            "short clips; can feel hyperactive on a single long video."
        )
        row_micro.addWidget(self.check_micro_all)
        row_micro.addStretch(1)
        opts.addLayout(row_micro)

        # full-length mode (hidden, always enabled internally)
        self.check_full_length = QCheckBox("Always fill full music length", self)
        self.check_full_length.setToolTip(
            "When enabled, the generated video will be extended with extra segments\n"
            "so that its duration matches the full music track. Useful when microclips\n"
            "or sparse beats would otherwise create a shorter video."
        )
        self.check_full_length.setChecked(True)
        self.check_full_length.hide()

        # intro / outro fades
        row_fade = QHBoxLayout()
        self.check_intro_fade = QCheckBox("Fade in from black at start         ", self)
        self.check_intro_fade.setChecked(True)
        self.check_intro_fade.setToolTip(
            "Add a short fade-in from black at the very beginning of the music clip."
        )
        row_fade.addWidget(self.check_intro_fade)

        self.check_outro_fade = QCheckBox("Fade out to black at end", self)
        self.check_outro_fade.setChecked(True)
        self.check_outro_fade.setToolTip(
            "Add a short fade-out to black at the very end of the music clip."
        )
        row_fade.addWidget(self.check_outro_fade)
        row_fade.addStretch(1)
        opts.addLayout(row_fade)

# clip order
        row_order = QHBoxLayout()
        row_order.addWidget(QLabel("Clip order:", self))
        self.combo_clip_order = QComboBox(self)
        self.combo_clip_order.addItems(
            [
                "Random (default)",
                "Sequential",
                "Shuffle (no repeats)",
            ]
        )
        self.combo_clip_order.setToolTip(
            "How clips from a folder are picked:\n"
            "- Random: random clips, avoid using the same clip twice in a row.\n"
            "- Sequential: go through the folder in order, then loop.\n"
            "- Shuffle: each round uses all clips once in random order before repeating."
        )
        row_order.addWidget(self.combo_clip_order, 1)
        row_order.addStretch(1)
        root.addLayout(row_order)

        # minimum clip length filter
        row_minclip = QHBoxLayout()
        self.check_min_clip = QCheckBox("Ignore clips shorter than", self)
        self.check_min_clip.setChecked(False)
        self.spin_min_clip = QDoubleSpinBox(self)
        self.spin_min_clip.setDecimals(1)
        self.spin_min_clip.setSingleStep(0.5)
        self.spin_min_clip.setRange(0.5, 10.0)
        self.spin_min_clip.setValue(1.5)
        self.spin_min_clip.setSuffix(" s")
        self.check_min_clip.setToolTip(
            "When enabled, video clips shorter than this duration will be ignored\n"
            "during clip discovery. Useful to avoid ultra-short fragments that can\n"
            "cause jittery pacing or visual glitches."
        )
        row_minclip.addWidget(self.check_min_clip)
        row_minclip.addWidget(self.spin_min_clip)
        row_minclip.addStretch(1)
        root.addLayout(row_minclip)

        # transitions
        row_trans = QHBoxLayout()
        self.label_transitions = QLabel("Transitions:", self)
        row_trans.addWidget(self.label_transitions)
        self.combo_transitions = QComboBox(self)
        self.combo_transitions.addItems(
            [
                "Slide",
                "Hard cuts",
                "Zoom pulse / scale pulse",
                "RGB split / chromatic aberration",
                "VHS noise + scanlines",
                "Motion blur boost",
                "Mixed (fades + flash cuts)",
                "Creative mix (fades + dips + flashes)",
            ]
        )
        self.combo_transitions.setToolTip(
            "How clip-to-clip transitions look:\n"
            "- Slide: segments with gentle motion fades (soft feel).\n"
            "- Hard cuts: no fades, straight cuts.\n"
            "- Zoom pulse / scale pulse: rhythmic zoom-in/zoom-out on the frame.\n"
            "- RGB split / chromatic aberration: subtle color-channel offset.\n"
            "- VHS noise + scanlines: noisy, retro VHS-style texture.\n"
            "- Motion blur boost: strong motion blur accent on movement."
            "- Mixed: mostly fades with occasional flash cuts on peaks.\n"
            "- Creative mix: combination of fades, dips and flashes.\n"
        )
        row_trans.addWidget(self.combo_transitions, 1)
        row_trans.addStretch(1)
        opts.addLayout(row_trans)

        
        # random transitions toggle + manager
        row_trans_ctrl = QHBoxLayout()
        self.check_trans_random = QCheckBox("Random transitions", self)
        self.check_trans_random.setToolTip(
            "When enabled, each segment will pick a transition style at random\n"
            "from the enabled list below. When disabled, the selected style\n"
            "in the dropdown is used for all segments."
        )
        row_trans_ctrl.addWidget(self.check_trans_random)
        btn_manage_trans = QPushButton("Manage transitions...", self)
        btn_manage_trans.setToolTip(
            "Choose which transition styles are allowed when 'Random transitions'\n"
            "is enabled (Slide, Hard cuts, Zoom pulse, RGB split, VHS, Motion blur), Mixed, Creative mix."
        )
        row_trans_ctrl.addWidget(btn_manage_trans)
        row_trans_ctrl.addStretch(1)
        opts.addLayout(row_trans_ctrl)

# random seed
        row_seed = QHBoxLayout()
        row_seed.addWidget(QLabel("Random seed:", self))
        self.spin_seed = QSpinBox(self)
        self.spin_seed.setRange(0, 999999999)
        self.spin_seed.setValue(1337)
        self.spin_seed.setEnabled(False)
        self.spin_seed.setToolTip(
            "Seed value used when 'Use fixed seed' is enabled.\n"
            "Same seed + same inputs = repeatable clip order and FX.\n"
            "Disable to get a fresh random result each run."
        )
        row_seed.addWidget(self.spin_seed)
        self.check_use_seed = QCheckBox("Use fixed seed", self)
        self.check_use_seed.setToolTip(
            "When enabled, the random generator is seeded with the value above,\n"
            "so you can reproduce the same edit again. When disabled, each run\n"
            "will be different."
        )
        row_seed.addWidget(self.check_use_seed)
        row_seed.addStretch(1)
        root.addLayout(row_seed)

        # resolution
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Output resolution:", self))
        self.combo_res = QComboBox(self)
        self.combo_res.addItems(
            [
                "Auto (single video: keep source)",
                "Multi clips: highest clip resolution",
                "Multi clips: lowest clip resolution",
                "Fixed: 480p",
                "Fixed: 720p",
                "Fixed: 1080p",
            ]
        )
        self.combo_res.setToolTip(
            "Resolution strategy when working with multiple clips.\n"
            "- Auto: single video -> keep its resolution.\n"
            "- Highest/Lowest clip resolution: unify to that size.\n"
            "- Fixed: scale everything to 480p / 720p / 1080p."
        )
        row_res.addWidget(self.combo_res, 1)
        root.addLayout(row_res)

        root.addWidget(box_opts)

        # advanced
        box_adv = QGroupBox("Advanced", self)
        box_adv.setCheckable(True)
        box_adv.setChecked(False)
        adv = QFormLayout(box_adv)
        adv.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        adv.setSpacing(4)

        self.slider_sens = QSlider(Qt.Horizontal, self)
        self.slider_sens.setMinimum(10)   # represents 1.0
        self.slider_sens.setMaximum(200)  # represents 20.0 (i.e. 2.0 steps *10)
        self.slider_sens.setValue(50)     # default 5.0
        self.slider_sens.setSingleStep(5) # 0.5 represented as +5
        
        self.slider_sens.setMinimum(2)
        self.slider_sens.setMaximum(20)
        self.slider_sens.setValue(10)
        self.slider_sens.setToolTip(
            "Beat sensitivity for the detector (0.5–20.0 internal scale).\n"
            "Lower = fewer beats (stricter), higher = more beats (looser)."
        )
        adv.addRow("Beat sensitivity:", self.slider_sens)

        self.spin_beats_per_seg = QSpinBox(self)
        self.spin_beats_per_seg.setRange(2, 16)
        self.spin_beats_per_seg.setValue(2)
        self.spin_beats_per_seg.setToolTip(
            "Number of beats grouped into one base video segment.\n"
            "1 = very fast cuts, 2 = default, 4+ = slower cuts."
        )
        adv.addRow("Beats per base segment:", self.spin_beats_per_seg)

        root.addWidget(box_adv)

        # progress + buttons
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ready.")
        root.addWidget(self.progress)

        # compact analysis summary
        self.label_summary = QLabel("", self)
        self.label_summary.setWordWrap(True)
        root.addWidget(self.label_summary)

        row_btn = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze Music", self)
        self.btn_generate = QPushButton("Generate Music Clip", self)
        self.btn_cancel = QPushButton("Cancel", self)
        row_btn.addWidget(self.btn_analyze)
        row_btn.addWidget(self.btn_generate)
        row_btn.addWidget(self.btn_cancel)
        root.addLayout(row_btn)
        root.addStretch(1)

        # connections
        btn_a.clicked.connect(self._browse_audio)
        btn_vf.clicked.connect(self._browse_video_file)
        btn_vd.clicked.connect(self._browse_video_dir)
        btn_vmf.clicked.connect(self._browse_video_files)
        btn_o.clicked.connect(self._browse_output)
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_manage_trans.clicked.connect(self._on_manage_transitions)

        self.check_micro_chorus.stateChanged.connect(self._on_micro_mode_changed)
        self.check_micro_all.stateChanged.connect(self._on_micro_mode_changed)
        self.check_use_seed.stateChanged.connect(self._on_seed_toggle)
        self.check_trans_random.stateChanged.connect(self._on_trans_random_toggled)


    def _on_manage_transitions(self) -> None:
        """Open a dialog to choose which transition styles are allowed for random mode."""
        names = [
            "Slide",
            "Hard cuts",
            "Zoom pulse / scale pulse",
            "RGB split / chromatic aberration",
            "VHS noise + scanlines",
            "Motion blur boost",
            "Mixed (fades + flash cuts)",
            "Creative mix (fades + dips + flashes)",
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Manage transitions")
        layout = QVBoxLayout(dlg)
        info = QLabel(
            "Select which transition styles can be used when\n"
            "'Random transitions' is enabled."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        checkboxes = []
        for i, name in enumerate(names):
            cb = QCheckBox(name, dlg)
            cb.setChecked(i in self._enabled_transition_modes)
            layout.addWidget(cb)
            checkboxes.append(cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        selected = {i for i, cb in enumerate(checkboxes) if cb.isChecked()}
        if not selected:
            QMessageBox.warning(
                self,
                "No transitions selected",
                "At least one transition style must be enabled for random mode.",
                QMessageBox.Ok,
            )
            return
        self._enabled_transition_modes = selected



    def _update_transition_visibility(self) -> None:
        """Show or hide the transitions dropdown row based on random mode."""
        # If random transitions are enabled, hide the label + dropdown since
        # they are not used. When disabled, show them again.
        try:
            is_random = self.check_trans_random.isChecked()
        except Exception:
            return
        visible = not is_random
        try:
            self.label_transitions.setVisible(visible)
        except Exception:
            pass
        try:
            self.combo_transitions.setVisible(visible)
        except Exception:
            pass

    def _on_trans_random_toggled(self, state: int) -> None:
        """Keep the transitions row in sync with the 'Random transitions' toggle."""
        self._update_transition_visibility()

    def _on_cancel(self) -> None:
        """Cancel current render (if any), reset progress and clean temp folder."""
        # If a worker exists and is running, ask it to stop.
        if getattr(self, "_worker", None) is not None:
            try:
                self._worker.requestInterruption()
            except Exception:
                pass
        # Update UI
        self.progress.setValue(0)
        self.progress.setFormat("Cancelled.")
        self.label_summary.setText(self.label_summary.text() + "\nRender cancelled.")
        # Best-effort temp cleanup by rerunning cleanup helper.
        try:
            cleanup_temp_dir(self.edit_output.text().strip())
        except Exception:
            pass

    def _load_settings(self) -> None:
        """Load last-used paths and options from QSettings."""
        s = self._settings
        self.edit_audio.setText(s.value("audio_path", "", str))
        self.edit_video.setText(s.value("video_path", "", str))
        self.edit_output.setText(s.value("output_path", "", str))

        self.combo_fx.setCurrentIndex(int(s.value("fx_level", self.combo_fx.currentIndex())))
        self.check_micro_chorus.setChecked(bool(int(s.value("micro_chorus", int(self.check_micro_chorus.isChecked())))))
        self.check_micro_all.setChecked(bool(int(s.value("micro_all", int(self.check_micro_all.isChecked())))))
        self.check_full_length.setChecked(True)
        self.check_intro_fade.setChecked(bool(int(s.value("intro_fade", int(self.check_intro_fade.isChecked())))))
        self.check_outro_fade.setChecked(bool(int(s.value("outro_fade", int(self.check_outro_fade.isChecked())))))
        self.combo_clip_order.setCurrentIndex(int(s.value("clip_order", self.combo_clip_order.currentIndex())))
        self.combo_transitions.setCurrentIndex(int(s.value("transitions_mode", self.combo_transitions.currentIndex())))
        self.check_trans_random.setChecked(bool(int(s.value("transitions_random", int(self.check_trans_random.isChecked())))))
        self.spin_seed.setValue(int(s.value("seed_value", self.spin_seed.value())))
        self.check_use_seed.setChecked(bool(int(s.value("use_seed", int(self.check_use_seed.isChecked())))))
        self.combo_res.setCurrentIndex(int(s.value("res_mode", self.combo_res.currentIndex())))

        self.slider_sens.setValue(int(s.value("beat_sensitivity", self.slider_sens.value())))
        self.spin_beats_per_seg.setValue(int(s.value("beats_per_segment", self.spin_beats_per_seg.value())))

        self.check_min_clip.setChecked(bool(int(s.value("min_clip_enabled", int(self.check_min_clip.isChecked())))))
        self.spin_min_clip.setValue(float(s.value("min_clip_seconds", self.spin_min_clip.value())))

        # Ensure transitions dropdown visibility matches restored random toggle
        self._update_transition_visibility()

    def _save_settings(self) -> None:
        """Save last-used paths and options to QSettings."""
        s = self._settings
        s.setValue("audio_path", self.edit_audio.text().strip())
        s.setValue("video_path", self.edit_video.text().strip())
        s.setValue("output_path", self.edit_output.text().strip())

        s.setValue("fx_level", self.combo_fx.currentIndex())
        s.setValue("micro_chorus", int(self.check_micro_chorus.isChecked()))
        s.setValue("micro_all", int(self.check_micro_all.isChecked()))
        s.setValue("full_length", int(self.check_full_length.isChecked()))
        s.setValue("intro_fade", int(self.check_intro_fade.isChecked()))
        s.setValue("outro_fade", int(self.check_outro_fade.isChecked()))
        s.setValue("clip_order", self.combo_clip_order.currentIndex())
        s.setValue("transitions_mode", self.combo_transitions.currentIndex())
        s.setValue("transitions_random", int(self.check_trans_random.isChecked()))
        s.setValue("seed_value", self.spin_seed.value())
        s.setValue("use_seed", int(self.check_use_seed.isChecked()))
        s.setValue("res_mode", self.combo_res.currentIndex())

        s.setValue("beat_sensitivity", self.slider_sens.value())
        s.setValue("beats_per_segment", self.spin_beats_per_seg.value())

        s.setValue("min_clip_enabled", int(self.check_min_clip.isChecked()))
        s.setValue("min_clip_seconds", float(self.spin_min_clip.value()))
    # dialogs / helpers

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select music/audio file",
            "",
            "Audio files (*.mp3 *.wav *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if path:
            self.edit_audio.setText(path)

    def _browse_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg);;All files (*.*)",
        )
        if path:
            self.edit_video.setText(path)

    def _browse_video_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder with clips", "")
        if path:
            self.edit_video.setText(path)

    def _browse_video_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more video clips",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg);;All files (*.*)",
        )
        if paths:
            # Join multiple paths with '|' so discovery can treat them as a list
            self.edit_video.setText("|".join(paths))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if path:
            self.edit_output.setText(path)

    def _error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg, QMessageBox.Ok)

    def _info(self, title: str, msg: str) -> None:
        QMessageBox.information(self, title, msg, QMessageBox.Ok)

    def _resolve_paths(self) -> Optional[Tuple[str, str, str]]:
        audio = self.edit_audio.text().strip()
        video = self.edit_video.text().strip()
        out_dir = self.edit_output.text().strip()

        if not audio or not os.path.isfile(audio):
            self._error("Missing audio", "Please select a valid music/audio file.")
            return None

        # video can be:
        # - a single file
        # - a folder
        # - or a '|' separated list of files (from Clip files... picker)
        if not video:
            self._error(
                "Missing video input",
                "Please select a valid video file, a folder containing clips, "
                "or use the Clip files... button to pick multiple clips.",
            )
            return None

        if "|" in video:
            # basic validation: at least one existing file in the list
            parts = [p.strip() for p in video.split("|") if p.strip()]
            if not parts or not any(os.path.isfile(p) for p in parts):
                self._error(
                    "Missing video input",
                    "None of the selected clip files could be found. Please re-select your clips.",
                )
                return None
        elif not (os.path.isfile(video) or os.path.isdir(video)):
            self._error(
                "Missing video input",
                "Please select a valid video file or a folder containing clips.",
            )
            return None
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(audio), "output")
            self.edit_output.setText(out_dir)
        _ensure_dir(out_dir)
        return audio, video, out_dir

    # microclip mode mutual exclusivity

    def _on_micro_mode_changed(self, state: int) -> None:
        if not state:
            return
        sender = self.sender()
        if sender is self.check_micro_chorus:
            self.check_micro_all.blockSignals(True)
            self.check_micro_all.setChecked(False)
            self.check_micro_all.blockSignals(False)
        elif sender is self.check_micro_all:
            self.check_micro_chorus.blockSignals(True)
            self.check_micro_chorus.setChecked(False)
            self.check_micro_chorus.blockSignals(False)

    def _on_seed_toggle(self, state: int) -> None:
        enabled = state != 0
        self.spin_seed.setEnabled(enabled)

    # analysis

    def _on_analyze(self) -> None:
        resolved = self._resolve_paths()
        if not resolved:
            return
        audio, _, _ = resolved
        self.progress.setValue(0)
        self.progress.setFormat("Analyzing music...")
        # Sync analysis config with current UI
        self._analysis_config.sensitivity = self.slider_sens.value()
        try:
            self._analysis = analyze_music(audio, self._ffmpeg, self._analysis_config)
        except Exception as e:
            self._analysis = None
            self.progress.setValue(0)
            self.progress.setFormat("Ready.")
            self._error("Analysis failed", str(e))
            return

        beats = len(self._analysis.beats)
        secs = len(self._analysis.sections)
        self.progress.setValue(30)
        self.progress.setFormat(f"Analysis complete: {beats} beats, {secs} sections.")

        # Save current paths and analysis-related options
        self._save_settings()

        # Compact analysis summary under the progress bar
        try:
            total_dur = self._analysis.duration
        except Exception:
            total_dur = None

        lines = []
        if total_dur is not None:
            lines.append(f"Total duration: {total_dur:.1f}s")
        lines.append(f"Beats detected: {beats}")
        lines.append(f"Sections: {secs}")
        # Section breakdown like: Intro 0–18s
        try:
            for sec in self._analysis.sections:
                name = getattr(sec, "label", None) or getattr(sec, "kind", None) or "Section"
                start = getattr(sec, "start", None)
                end = getattr(sec, "end", None)
                if start is not None and end is not None:
                    lines.append(f"{name}: {start:.0f}s–{end:.0f}s")
        except Exception:
            pass

        self.label_summary.setText(" \u2022 ".join(lines))

    def _target_resolution(
        self, sources: List[ClipSource], video_input: str
    ) -> Optional[Tuple[int, int]]:
        mode = self.combo_res.currentIndex()

        # Mode 0: single video keep source
        if os.path.isfile(video_input) and mode == 0:
            return None

        if not sources:
            return None

        def get_res(path: str) -> Optional[Tuple[int, int]]:
            cmd = [
                self._ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                data = json.loads(out)
                streams = data.get("streams") or []
                if not streams:
                    return None
                s = streams[0]
                return int(s.get("width", 0)), int(s.get("height", 0))
            except Exception:
                return None

        res_list = []
        for s in sources:
            r = get_res(s.path)
            if r and r[0] > 0 and r[1] > 0:
                res_list.append(r)

        if not res_list:
            return None

        if mode == 1:  # highest
            w = max(r[0] for r in res_list)
            h = max(r[1] for r in res_list)
            return w, h
        if mode == 2:  # lowest
            w = min(r[0] for r in res_list)
            h = min(r[1] for r in res_list)
            return w, h
        if mode == 3:
            return 854, 480
        if mode == 4:
            return 1280, 720
        if mode == 5:
            return 1920, 1080
        return None

    def _on_generate(self) -> None:
        resolved = self._resolve_paths()
        if not resolved:
            return
        audio, video, out_dir = resolved

        if self._analysis is None:
            self._on_analyze()
            if self._analysis is None:
                return

        sources = discover_video_sources(video, self._ffprobe)
        if not sources:
            self._error("No video clips", "Could not find any usable video sources.")
            return

        # Optional minimum clip length filter
        if self.check_min_clip.isChecked():
            min_len = float(self.spin_min_clip.value())
            filtered = [s for s in sources if s.duration >= min_len]
            skipped = len(sources) - len(filtered)
            sources = filtered
            if not sources:
                self._error(
                    "All clips too short",
                    f"All discovered clips were shorter than {min_len:.1f}s and were ignored.",
                )
                return
            if skipped > 0:
                self._info(
                    "Short clips skipped",
                    f"Ignored {skipped} clip(s) shorter than {min_len:.1f}s.",
                )

        idx = self.combo_fx.currentIndex()
        if idx == 0:
            fx_level = "minimal"
        elif idx == 1:
            fx_level = "moderate"
        else:
            fx_level = "high"

        if self.check_micro_chorus.isChecked():
            micro_mode = 1
        elif self.check_micro_all.isChecked():
            micro_mode = 2
        else:
            micro_mode = 0

        beats_per = max(1, self.spin_beats_per_seg.value())

        trans_idx = self.combo_transitions.currentIndex()
        transition_mode = trans_idx  # 0,1,2

        clip_order_mode = self.combo_clip_order.currentIndex()  # 0,1,2

        force_full_length = self.check_full_length.isChecked()
        seed_enabled = self.check_use_seed.isChecked()
        seed_value = self.spin_seed.value()

        segments = build_timeline(
            self._analysis,
            sources,
            fx_level=fx_level,
            microclip_mode=micro_mode,
            beats_per_segment=beats_per,
            transition_mode=transition_mode,
            clip_order_mode=clip_order_mode,
            force_full_length=force_full_length,
            seed_enabled=seed_enabled,
            seed_value=seed_value,
            transition_random=self.check_trans_random.isChecked(),
            transition_modes_enabled=sorted(self._enabled_transition_modes),
        )
        if not segments:
            self._error("Timeline empty", "Failed to build a video timeline.")
            return

        target_res = self._target_resolution(sources, video)

        if self._worker is not None and self._worker.isRunning():
            self._error("Busy", "A render is already running.")
            return

        # Persist current options before kicking off the render
        self._save_settings()

        self.progress.setValue(0)
        self.progress.setFormat("Starting render...")

        self._worker = RenderWorker(
            audio_path=audio,
            output_dir=out_dir,
            analysis=self._analysis,
            segments=segments,
            ffmpeg=self._ffmpeg,
            ffprobe=self._ffprobe,
            target_resolution=target_res,
            transition_mode=transition_mode,
            intro_fade=self.check_intro_fade.isChecked(),
            outro_fade=self.check_outro_fade.isChecked(),
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    # worker callbacks

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(max(0, min(100, pct)))
        if msg:
            self.progress.setFormat(msg)

    def _on_worker_finished(self, out_path: str) -> None:
        self.progress.setValue(100)
        self.progress.setFormat("Done.")
        QMessageBox.information(
            self,
            "Music Clip created",
            f"Finished generating music clip:\n{out_path}",
            QMessageBox.Ok,
        )
        self._worker = None

    def _on_worker_failed(self, msg: str) -> None:
        self.progress.setValue(0)
        self.progress.setFormat("Failed.")
        self._error("Music Clip Creator failed", msg)
        self._worker = None


# ------------------------- integration entry point -------------------------


def install_auto_music_sync_tool(parent, section) -> AutoMusicSyncWidget:
    """Install the Music Clip Creator UI into a CollapsibleSection."""
    content = getattr(section, "content", None)
    parent_widget = content if content is not None else section

    widget = AutoMusicSyncWidget(parent=parent_widget)
    widget.setObjectName("MusicClipCreatorWidget")

    if content is not None:
        layout = content.layout()
        if layout is None:
            layout = QVBoxLayout(content)
    else:
        layout = section.layout()
        if layout is None:
            layout = QVBoxLayout(section)

    layout.addWidget(widget)
    return widget


def cleanup_temp_dir(output_path: str) -> None:
    """Best-effort removal of temp files for the music clip creator.

    This looks for a sibling 'temp' directory next to the chosen output file
    and removes any intermediate segment files if possible.
    """
    import shutil
    from pathlib import Path

    if not output_path:
        return
    p = Path(output_path)
    # If user picked a folder, we still look for a 'temp' subdir.
    parent = p if p.is_dir() else p.parent
    temp_dir = parent / "temp"
    if temp_dir.is_dir():
        try:
            for child in temp_dir.iterdir():
                if child.is_file() and child.suffix in {".mp4", ".mkv", ".mov", ".ts"}:
                    child.unlink(missing_ok=True)
        except Exception:
            # Don't crash the UI if cleanup fails.
            pass


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = AutoMusicSyncWidget()
    w.setWindowTitle("Music Clip Creator (Standalone Test)")
    w.resize(900, 620)
    w.show()
    sys.exit(app.exec())
