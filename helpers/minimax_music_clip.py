from __future__ import annotations

"""MiniMax H3 Music Clip Creator.

A MiniMax-native music-video helper for FrameVision / standalone PySide6 use.

Design rules:
- Ref2VA is the generation path because it can condition on the song audio slice.
- The master song timeline is authoritative; MiniMax frame counts are generation constraints.
- MiniMax native frame counts are selected from the 17-frame grid (124 + n*17).
- Generated clips may include a small head/tail margin and are trimmed during assembly.
- Lyric songs prefer Whisper phrase boundaries; instrumental songs use evenly distributed,
  beat-aware cuts without treating tiny timing mismatches as fatal.
- A visual change may be directed *inside* one MiniMax generation when a lyric/section
  boundary does not fit a useful physical clip boundary.
- The original full song is muxed back at final assembly; generated clip audio is not used
  as the final soundtrack.

This first helper intentionally does not import the old LTX Music Clip Creator. It keeps a
small project JSON and calls the existing helpers/generate_ref.py backend.
"""

import json
import hashlib
import concurrent.futures
import math
import os
import re
import random
import shutil
import struct
import subprocess
import threading
import sys
import tempfile
import urllib.request
import zipfile
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PySide6.QtCore import QThread, Qt, Signal, QUrl, QTimer, QEvent
    from PySide6.QtGui import QDesktopServices, QPixmap, QPalette
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSlider,
        QScrollArea,
        QFrame,
        QSizePolicy,
        QSpinBox,
        QDoubleSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        QAbstractItemView,
    )
except Exception as exc:  # pragma: no cover - useful error when launched from wrong env
    raise RuntimeError("MiniMax Music Clip Creator requires PySide6.") from exc


def _detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for candidate in candidates:
        if (candidate / "environments").is_dir() and (candidate / "helpers").is_dir():
            return candidate
    return here.parent


ROOT = _detect_project_root()
HELPERS_DIR = ROOT / "helpers"
OUTPUT_ROOT = ROOT / "output" / "minimax_music_clips"
SETTINGS_PATH = ROOT / "presets" / "minimax_music_clip_settings.json"
AUTOSAVE_PATH = ROOT / "presets" / "setsave" / "minimax_music_clip.json"
WHISPER_DIR = ROOT / "presets" / "bin" / "whisper"
WHISPER_MODEL = WHISPER_DIR / "ggml-small.bin"
WHISPER_RUNTIME_ZIP = WHISPER_DIR / "_whisper_runtime.zip"
WHISPER_RUNTIME_URL = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true"
MINIMAX_ENV = ROOT / "environments" / ".minimax_h3_int4"
MINIMAX_PY = MINIMAX_ENV / "python.exe"
if not MINIMAX_PY.is_file():
    MINIMAX_PY = MINIMAX_ENV / "Scripts" / "python.exe"
GENERATE_REF = HELPERS_DIR / "generate_ref.py"

NORMAL_FRAME_MAX = 719
MUSIC_FRAME_MIN = 124
MUSIC_FRAME_DEFAULT_MAX = 396
MUSIC_FRAME_GRID = tuple(range(MUSIC_FRAME_MIN, MUSIC_FRAME_DEFAULT_MAX + 1, 17))
FPS = 24.0


def _music_project_identity(title: str, audio_path: str) -> str:
    """Stable identity: same project title + same master track means same project folder."""
    title_key = re.sub(r"\s+", " ", (title or "").strip()).casefold()
    audio_key = ""
    if audio_path:
        try:
            audio_key = str(Path(audio_path).expanduser().resolve()).casefold()
        except Exception:
            audio_key = str(audio_path).strip().casefold()
    if not title_key and not audio_key:
        return ""
    return hashlib.sha1((title_key + "\n" + audio_key).encode("utf-8", "ignore")).hexdigest()


def _cleanup_music_clip_temp_artifacts() -> None:
    """Remove stale scratch data without touching rendered raw clips or final videos."""
    now = time.time()
    try:
        temp_root = Path(tempfile.gettempdir())
        for child in temp_root.glob("fv_minimax_music_an_*"):
            try:
                if child.is_dir() and now - child.stat().st_mtime > 3600:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass

    for folder, max_age in ((OUTPUT_ROOT / "_temp", 3600), (OUTPUT_ROOT / "_preview_cache", 7 * 86400)):
        try:
            if not folder.is_dir():
                continue
            for child in folder.iterdir():
                try:
                    if now - child.stat().st_mtime > max_age:
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                if not any(folder.iterdir()):
                    folder.rmdir()
            except Exception:
                pass
        except Exception:
            pass

    try:
        if OUTPUT_ROOT.is_dir():
            for folder in OUTPUT_ROOT.rglob("_assembly"):
                try:
                    if folder.is_dir() and now - folder.stat().st_mtime > 6 * 3600:
                        shutil.rmtree(folder, ignore_errors=True)
                except Exception:
                    pass
    except Exception:
        pass

RESOLUTION_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "576 × 320": {"16:9": (576, 320), "9:16": (320, 576), "1:1": (320, 320)},
    "736 × 384": {"16:9": (736, 384), "9:16": (384, 736), "1:1": (384, 384)},
    "832 × 480": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (480, 480)},
    "960 × 544": {"16:9": (960, 544), "9:16": (544, 960), "1:1": (544, 544)},
    "1280 × 720": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (704, 704)},
    "1344 × 768": {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (768, 768)},
    "1920 × 1088": {"16:9": (1920, 1088), "9:16": (1088, 1920), "1:1": (1088, 1088)},
}

REFERENCE_TYPES = ("Character", "Background / Location", "Object / Prop", "Style / Mood", "Picture / Composition anchor", "Other")
REFERENCE_TYPE_ALIASES = {
    "Location": "Background / Location",
    "Object": "Object / Prop",
    "Style": "Style / Mood",
}

def _normalise_reference_kind(kind: str) -> str:
    k = (kind or "").strip()
    k = REFERENCE_TYPE_ALIASES.get(k, k)
    return k if k in REFERENCE_TYPES else "Other"


def _safe_stem(value: str) -> str:
    stem = Path(value or "music_video").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or "music_video"


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _existing_executable(candidates: Iterable[Path | str]) -> str:
    for candidate in candidates:
        text = str(candidate)
        if not text:
            continue
        p = Path(text)
        if p.is_file():
            return str(p)
        found = shutil.which(text)
        if found:
            return found
    return ""


def ffmpeg_path() -> str:
    return _existing_executable(
        [
            ROOT / "presets" / "bin" / "ffmpeg.exe",
            ROOT / "presets" / "bin" / "ffmpeg",
            "ffmpeg.exe",
            "ffmpeg",
        ]
    )


def ffprobe_path() -> str:
    return _existing_executable(
        [
            ROOT / "presets" / "bin" / "ffprobe.exe",
            ROOT / "presets" / "bin" / "ffprobe",
            "ffprobe.exe",
            "ffprobe",
        ]
    )


def probe_duration(path: str) -> float:
    probe = ffprobe_path()
    if not probe:
        return 0.0
    cp = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        return max(0.0, float((cp.stdout or "").strip()))
    except Exception:
        return 0.0


def minimax_frame_grid(max_frames: int) -> List[int]:
    max_frames = max(MUSIC_FRAME_MIN, min(NORMAL_FRAME_MAX, int(max_frames)))
    values = list(range(MUSIC_FRAME_MIN, max_frames + 1, 17))
    # 480 is supported by the existing MiniMax GUI/backend even though it is off-grid.
    if MUSIC_FRAME_MIN <= 480 <= max_frames:
        values.append(480)
    return sorted(set(values))


def nearest_generation_frames(required_seconds: float, max_frames: int, prefer_longer: bool = True) -> int:
    """Choose a valid MiniMax frame count for a required amount of source time.

    Prefer a duration that is at least as long as requested. If the request exceeds
    the user's max, return the maximum and let the planner split the edit interval.
    """
    grid = minimax_frame_grid(max_frames)
    if not grid:
        return MUSIC_FRAME_MIN
    wanted = max(0.0, float(required_seconds)) * FPS
    if prefer_longer:
        for frames in grid:
            if frames + 1e-9 >= wanted:
                return frames
    return min(grid, key=lambda f: abs(float(f) - wanted))


def _clean_minimax_env() -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "_CE_CONDA",
        "_CE_M",
    ):
        env.pop(key, None)
    if MINIMAX_PY.is_file():
        prefix = MINIMAX_PY.parents[1]
        env["CONDA_PREFIX"] = str(prefix)
        preferred = [prefix, prefix / "Scripts", prefix / "Library" / "bin"]
        old_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        cleaned: List[str] = []
        for part in old_parts:
            low = os.path.normcase(os.path.abspath(part))
            if os.path.normcase(str(ROOT / "environments")) in low and os.path.normcase(str(prefix)) not in low:
                continue
            cleaned.append(part)
        env["PATH"] = os.pathsep.join([str(p) for p in preferred] + cleaned)
    env["PYTHONNOUSERSITE"] = "1"
    return env


@dataclass
class Beat:
    time: float
    strength: float
    kind: str


@dataclass
class Section:
    start: float
    end: float
    kind: str


@dataclass
class AnalysisResult:
    duration: float = 0.0
    beats: List[Beat] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)


@dataclass
class LyricSegment:
    start: float
    end: float
    text: str


@dataclass
class ReferenceAsset:
    name: str
    kind: str
    path: str
    description: str = ""
    enabled: bool = True


@dataclass
class MusicShot:
    index: int
    edit_start: float
    edit_end: float
    frames: int
    generation_start: float
    generation_end: float
    trim_in: float
    lyrics: str = ""
    section: str = ""
    prompt: str = ""
    reference_names: List[str] = field(default_factory=list)
    internal_cuts: List[float] = field(default_factory=list)  # song-absolute seconds
    output_path: str = ""
    status: str = "Planned"
    seed: int = -1

    @property
    def edit_duration(self) -> float:
        return max(0.0, self.edit_end - self.edit_start)

    @property
    def generation_duration(self) -> float:
        return self.frames / FPS


@dataclass
class MusicProject:
    version: int = 1
    audio_path: str = ""
    output_dir: str = ""
    output_identity: str = ""
    title: str = ""
    main_idea: str = ""
    style_theme: str = ""
    characters_subjects: str = ""
    locations_world: str = ""
    camera_choreography: str = ""
    resolution: str = "832 × 480"
    aspect: str = "16:9"
    max_frames: int = MUSIC_FRAME_DEFAULT_MAX
    head_padding: float = 0.35
    tail_padding: float = 0.45
    phrase_snap_tolerance: float = 1.25
    beat_sensitivity: int = 10
    whisper_timing_enabled: bool = True
    visible_lyric_subtitles: bool = False
    job_seed: int = -1
    steps: int = 15
    cfg: float = 1.0
    shift: float = 12.0
    audio_shift: float = 3.0
    ref_image_size: str = "match"
    sage_attention: bool = False
    spectrum: bool = False
    use_hybrid_model: bool = False
    hybrid_model_path: str = ""
    vram_manager_enabled: bool = True
    vram_auto_bypass: bool = True
    # Keep the MiniMax GUI fresh-install VRAM defaults here so Music Clip Creator
    # launches Ref2VA with the same protection policy instead of only passing the
    # abbreviated --vram-manager-auto flag.
    vram_residency_engine: str = "static"
    vram_runtime_free_gb: float = 0.50
    vram_text_headroom_gb: float = 1.0
    vram_diffusion_headroom_gb: float = 1.0
    vram_offload_chunk_mb: int = 512
    vram_max_resident_weights_gb: float = 0.0
    vram_block_check_interval: int = 1
    vram_async_streams: int = 2
    vram_video_vae_reserve_gb: float = 2.0
    vram_audio_vae_reserve_gb: float = 1.0
    vram_residency_fill: bool = False
    vram_residency_target_free_gb: float = 0.50
    vram_residency_warmup_blocks: int = 2
    vram_residency_refill_interval: int = 1
    turbo_lora_path: str = ""
    turbo_lora_strength: float = 1.0
    randomize_reference_characters: bool = False
    use_framevision_queue: bool = False
    reference_random_seed: int = -1
    references: List[ReferenceAsset] = field(default_factory=list)
    lyrics: List[LyricSegment] = field(default_factory=list)
    analysis: AnalysisResult = field(default_factory=AnalysisResult)
    shots: List[MusicShot] = field(default_factory=list)


# ------------------------- lightweight music analysis -------------------------


def analyze_music(audio_path: str, sensitivity: int = 10) -> AnalysisResult:
    """Port of the useful lightweight RMS beat/energy analysis from Music Clip Creator."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found in presets/bin or PATH.")
    tmpdir = Path(tempfile.mkdtemp(prefix="fv_minimax_music_an_"))
    wav_path = tmpdir / "mono.wav"
    try:
        cp = subprocess.run(
            [ffmpeg, "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", str(wav_path)],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if cp.returncode != 0 or not wav_path.is_file():
            raise RuntimeError("Failed to convert audio for analysis:\n" + (cp.stderr or cp.stdout or ""))

        with wave.open(str(wav_path), "rb") as wf:
            fr = wf.getframerate()
            total_frames = wf.getnframes()
            duration = total_frames / float(fr) if fr else 0.0
            win_size = max(1, int(fr * 0.05))
            values: List[float] = []
            times: List[float] = []
            read = 0
            while read < total_frames:
                n = min(win_size, total_frames - read)
                raw = wf.readframes(n)
                read += n
                if not raw:
                    break
                count = len(raw) // 2
                if not count:
                    rms = 0.0
                else:
                    samples = struct.unpack("<" + "h" * count, raw)
                    acc = sum((s / 32768.0) ** 2 for s in samples)
                    rms = math.sqrt(acc / count)
                values.append(rms)
                times.append(read / float(fr))

        if not values:
            raise RuntimeError("Audio analysis produced no samples.")
        max_v = max(values) or 1.0
        norm = [v / max_v for v in values]
        mean = sum(norm) / len(norm)
        std = math.sqrt(sum((v - mean) ** 2 for v in norm) / len(norm))
        sens = max(0.0, min(20.0, float(sensitivity)))
        if sens <= 10.0:
            frac = (10.0 - sens) / 10.0
            scale = 1.0 + (frac ** 1.75) * 2.60
        else:
            frac = (sens - 10.0) / 10.0
            scale = 1.0 - (frac ** 1.20) * 0.42
        scale = max(0.55, min(3.60, scale))
        beat_thr = mean + std * 0.7 * scale
        major_thr = mean + std * 1.4 * scale

        beats: List[Beat] = []
        last_peak = -9999
        min_dist = max(1, int(0.15 / 0.05))
        for i in range(1, len(norm) - 1):
            v = norm[i]
            if v < beat_thr or not (v >= norm[i - 1] and v >= norm[i + 1]):
                continue
            if i - last_peak < min_dist:
                if beats and v > beats[-1].strength:
                    beats[-1] = Beat(times[i], v, "major" if v >= major_thr else "minor")
                    last_peak = i
                continue
            beats.append(Beat(times[i], v, "major" if v >= major_thr else "minor"))
            last_peak = i
        beats.sort(key=lambda b: b.time)

        win1 = max(1, int(1.0 / 0.05))
        e_vals: List[float] = []
        e_times: List[float] = []
        for i in range(0, len(norm), win1):
            chunk = norm[i : i + win1]
            if chunk:
                e_vals.append(sum(chunk) / len(chunk))
                e_times.append(i * 0.05)
        if not e_vals:
            e_vals, e_times = [mean], [0.0]
        e_mean = sum(e_vals) / len(e_vals)
        e_std = math.sqrt(sum((v - e_mean) ** 2 for v in e_vals) / len(e_vals))
        low_thr, high_thr = e_mean - 0.4 * e_std, e_mean + 0.4 * e_std
        raw_sections: List[Section] = []
        cur_kind: Optional[str] = None
        cur_start = 0.0
        for t, value in zip(e_times, e_vals):
            kind = "intro_or_break" if value < low_thr else ("chorus_or_drop" if value > high_thr else "verse_or_mid")
            if cur_kind is None:
                cur_kind, cur_start = kind, t
            elif kind != cur_kind:
                raw_sections.append(Section(cur_start, t, cur_kind))
                cur_kind, cur_start = kind, t
        if cur_kind is not None:
            raw_sections.append(Section(cur_start, duration, cur_kind))
        labelled: List[Section] = []
        chorus_seen = False
        for idx, section in enumerate(raw_sections):
            if idx == 0:
                kind = "intro"
            elif section.kind == "chorus_or_drop":
                kind = "chorus" if not chorus_seen else "drop"
                chorus_seen = True
            elif section.kind == "intro_or_break":
                kind = "break"
            else:
                kind = "verse"
            labelled.append(Section(section.start, section.end, kind))
        if labelled:
            labelled[-1] = Section(labelled[-1].start, labelled[-1].end, "outro")
        return AnalysisResult(duration=duration, beats=beats, sections=labelled)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ------------------------------ shot planning -------------------------------


def _section_at(t: float, sections: Sequence[Section]) -> str:
    for section in sections:
        if section.start - 1e-6 <= t < section.end + 1e-6:
            return section.kind
    return "unknown"


def _lyrics_overlapping(start: float, end: float, lyrics: Sequence[LyricSegment]) -> str:
    parts: List[str] = []
    for seg in lyrics:
        if max(start, seg.start) < min(end, seg.end):
            text = re.sub(r"\s+", " ", seg.text or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _nearest_beat(target: float, beats: Sequence[Beat], tolerance: float = 0.75) -> float:
    candidates = [b.time for b in beats if abs(b.time - target) <= tolerance]
    if not candidates:
        return target
    return min(candidates, key=lambda x: abs(x - target))


def _choose_next_instrumental_boundary(
    cursor: float,
    hard_limit: float,
    sections: Sequence[Section],
    beats: Sequence[Beat],
    project: MusicProject,
    max_frames: int,
) -> float:
    """Choose an instrumental edit from the detected musical grid.

    Instrumental planning must not be duration packing.  The detector's beat grid
    is the timing skeleton: low-energy passages span more beats, high-energy
    passages span fewer, and strong/major accents plus analysed section boundaries
    are preferred as the actual cut.  Only *after* this returns does the normal
    MiniMax frame-grid code choose enough generation frames to cover the edit.
    """
    valid_lengths = _valid_edit_lengths(cursor, project, max_frames)
    if not valid_lengths:
        return hard_limit

    min_edit = max(1.5, valid_lengths[0][0])
    max_edit = max(min_edit, min(hard_limit - cursor, valid_lengths[-1][0]))
    if max_edit <= min_edit + 0.05:
        return min(hard_limit, cursor + max_edit)

    ordered = sorted(
        [b for b in beats if math.isfinite(float(getattr(b, "time", 0.0) or 0.0))],
        key=lambda b: float(getattr(b, "time", 0.0) or 0.0),
    )
    usable_beats = [b for b in ordered if cursor + 0.02 < float(b.time) <= hard_limit + 1e-6]

    # Weak/no beat detection: use a real analysed section boundary if available,
    # otherwise fall back to a medium clip rather than the previous maximum pack.
    if len(ordered) < 2 or not usable_beats:
        structural = sorted({
            float(t)
            for sec in sections
            for t in (sec.start, sec.end)
            if cursor + min_edit <= float(t) <= hard_limit + 1e-6
        })
        if structural:
            return structural[0]
        return min(hard_limit, cursor + min(max_edit, max(min_edit, 6.5)))

    # Estimate the useful musical pulse from detected beat spacing.  Ignore very
    # tiny duplicate peaks and huge gaps so a breakdown does not corrupt the BPM.
    intervals = []
    for a, b in zip(ordered, ordered[1:]):
        dt = float(b.time) - float(a.time)
        if 0.22 <= dt <= 1.50:
            intervals.append(dt)
    if intervals:
        intervals.sort()
        beat_interval = intervals[len(intervals) // 2]
    else:
        beat_interval = 0.50

    # Local energy comes from the strength of the *actual forthcoming beats*, not
    # from a fixed shot-duration ratio.  This makes a rise/drop shorten the next
    # shot while quieter stretches naturally breathe longer.
    strengths_all = [max(0.0, float(getattr(b, "strength", 0.0) or 0.0)) for b in ordered]
    peak_strength = max(strengths_all) if strengths_all else 1.0
    peak_strength = peak_strength or 1.0
    local = [
        max(0.0, float(getattr(b, "strength", 0.0) or 0.0)) / peak_strength
        for b in usable_beats
        if float(b.time) <= min(hard_limit, cursor + 8.0)
    ]
    local_energy = (sum(local) / len(local)) if local else 0.5

    section = _section_at(cursor + 0.05, sections)
    # Beat counts are deliberately different by musical role. At ~120 BPM these
    # correspond roughly to 5-10 second edits, comfortably inside H3's limits.
    base_beats = {
        "intro": 18,
        "break": 18,
        "verse": 15,
        "chorus": 12,
        "drop": 10,
        "outro": 17,
    }.get(section, 15)
    if local_energy >= 0.76:
        base_beats -= 2
    elif local_energy >= 0.62:
        base_beats -= 1
    elif local_energy <= 0.36:
        base_beats += 3
    elif local_energy <= 0.48:
        base_beats += 1
    target_beats = max(8, min(22, int(base_beats)))

    # Important section entries are first-class cuts.  Prefer one when it lands in
    # the useful musical window instead of marching through a fixed beat count.
    min_end = cursor + min_edit
    structural = []
    for sec in sections:
        for t in (sec.start, sec.end):
            t = float(t)
            if min_end <= t <= hard_limit + 1e-6:
                structural.append(t)
    structural = sorted(set(structural))

    # Enumerate actual beat candidates and count beats from this shot's start.
    candidates = []
    count = 0
    prev_strength = None
    for beat in usable_beats:
        t = float(beat.time)
        if t < min_end - 1e-6:
            continue
        count += 1
        duration = t - cursor
        if duration > max_edit + 0.05:
            break
        strength = max(0.0, float(getattr(beat, "strength", 0.0) or 0.0)) / peak_strength
        kind = str(getattr(beat, "kind", "") or "").lower()
        novelty = abs(strength - prev_strength) if prev_strength is not None else 0.0
        prev_strength = strength

        # Prefer the musically appropriate beat count, but allow a nearby strong
        # accent to win. This is what creates genuine variable shot lengths.
        beat_error = abs(count - target_beats)
        score = beat_error * 0.19
        score -= strength * 0.85
        score -= novelty * 0.45
        if kind == "major":
            score -= 1.15
        # Do not drift to the hard maximum just because it exists.
        if t >= hard_limit - 0.18:
            score += 1.10
        candidates.append((score, t, count, strength, kind))

    if structural:
        # Section boundaries get a large priority bonus, but only when the shot is
        # not absurdly short. This preserves drops/breaks even if they interrupt a
        # normal 12/16-beat grouping.
        for t in structural:
            duration = t - cursor
            approx_beats = max(1, int(round(duration / max(0.001, beat_interval))))
            if duration < min_edit - 1e-6 or duration > max_edit + 0.05:
                continue
            beat_error = abs(approx_beats - target_beats)
            candidates.append((beat_error * 0.13 - 1.65, t, approx_beats, 1.0, "section"))

    if candidates:
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    # Last resort: stay on a detected beat near a section/energy-derived beat count.
    wanted = cursor + target_beats * beat_interval
    in_range = [b for b in usable_beats if min_end <= float(b.time) <= hard_limit + 1e-6]
    if in_range:
        return min(in_range, key=lambda b: abs(float(b.time) - wanted)).time
    return min(hard_limit, cursor + min(max_edit, max(min_edit, 6.5)))

def _candidate_lyric_boundaries(lyrics: Sequence[LyricSegment], duration: float) -> List[float]:
    """Return lyric *line ends* as the primary physical edit anchors.

    Whisper segment starts are useful for prompt context, but a line ending is the
    natural place to hand the song to the next generated clip.  Using starts and
    ends equally was one reason the old planner could cut through phrases.
    """
    vals = {max(0.0, duration)}
    for seg in lyrics:
        if 0.0 < seg.end < duration:
            vals.add(round(seg.end, 3))
    return sorted(vals)


def _candidate_lyric_starts(lyrics: Sequence[LyricSegment], duration: float) -> List[float]:
    vals = set()
    for seg in lyrics:
        if 0.0 < seg.start < duration:
            vals.add(round(seg.start, 3))
    return sorted(vals)


def _valid_edit_lengths(cursor: float, project: MusicProject, max_frames: int) -> List[Tuple[float, int]]:
    """Map valid H3 frame counts to the edit length they can safely cover here."""
    generation_start = max(0.0, cursor - project.head_padding)
    trim_in = cursor - generation_start
    out: List[Tuple[float, int]] = []
    for frames in minimax_frame_grid(max_frames):
        edit_len = frames / FPS - trim_in - project.tail_padding
        if edit_len > 1.20:
            out.append((edit_len, frames))
    return out


def _choose_next_lyric_boundary(
    cursor: float,
    hard_limit: float,
    lyric_ends: Sequence[float],
    lyric_starts: Sequence[float],
    section_bounds: Sequence[float],
    beats: Sequence[Beat],
    project: MusicProject,
    max_frames: int,
) -> float:
    """Choose a physical cut with lyric line endings as the timing skeleton.

    Priority is deliberately different from the old duration-packing planner:
      1. lyric line end that is already close to a valid MiniMax frame duration
      2. other lyric line ends that can be covered by a valid duration and trimmed
      3. section boundaries / lyric starts
      4. beats only as a small tie-breaker

    The song edit boundary remains authoritative; MiniMax may generate a little past
    it and final assembly trims the excess.
    """
    min_end = cursor + 1.25
    valid_lengths = _valid_edit_lengths(cursor, project, max_frames)
    if not valid_lengths:
        return hard_limit

    def frame_fit_error(end: float) -> Tuple[float, int]:
        wanted = end - cursor
        # Prefer a frame count that covers the boundary; trimming is allowed.
        covering = [(length - wanted, frames) for length, frames in valid_lengths if length + 1e-6 >= wanted]
        if not covering:
            return (999.0, max_frames)
        return min(covering, key=lambda x: x[0])

    def beat_distance(t: float) -> float:
        if not beats:
            return 9.0
        return min(abs(b.time - t) for b in beats)

    lyric_candidates = [v for v in lyric_ends if min_end < v <= hard_limit + 1e-6]
    if lyric_candidates:
        scored = []
        for value in lyric_candidates:
            excess, frames = frame_fit_error(value)
            if excess >= 998.0:
                continue
            # Near-grid lyric endings are the jackpot.  Otherwise still prefer a
            # lyric ending and trim the generation overrun rather than crossing it.
            near_grid = 0 if excess <= 0.35 else 1
            # Prefer useful clip lengths over tiny fragments, then smaller trim.
            short_penalty = max(0.0, 4.0 - (value - cursor))
            scored.append((near_grid, short_penalty, excess, beat_distance(value), -value, value, frames))
        if scored:
            return min(scored)[5]

    # No usable lyric ending before the current H3 limit. Use meaningful structural
    # boundaries before falling back to pure duration packing.
    structural = [v for v in section_bounds if min_end < v <= hard_limit + 1e-6]
    structural += [v for v in lyric_starts if min_end < v <= hard_limit + 1e-6]
    if structural:
        candidates = []
        for value in sorted(set(structural)):
            excess, _frames = frame_fit_error(value)
            if excess < 998.0:
                candidates.append((excess, beat_distance(value), -value, value))
        if candidates:
            return min(candidates)[3]

    return hard_limit


def build_shot_plan(project: MusicProject) -> List[MusicShot]:
    duration = float(project.analysis.duration or probe_duration(project.audio_path))
    if duration <= 0.05:
        raise RuntimeError("Could not determine song duration.")

    max_frames = max(MUSIC_FRAME_MIN, int(project.max_frames))
    max_gen_seconds = max_frames / FPS
    usable_edit_max = max(1.5, max_gen_seconds - project.head_padding - project.tail_padding)
    target_edit = usable_edit_max * 0.90
    lyrics = list(project.lyrics)
    use_lyric_timing = bool(project.whisper_timing_enabled and lyrics)
    lyric_bounds = _candidate_lyric_boundaries(lyrics, duration)
    lyric_starts = _candidate_lyric_starts(lyrics, duration)
    section_bounds = sorted({round(s.start, 3) for s in project.analysis.sections} | {round(s.end, 3) for s in project.analysis.sections})

    edit_ranges: List[Tuple[float, float]] = []
    cursor = 0.0
    while cursor < duration - 0.04:
        hard_limit = min(duration, cursor + usable_edit_max)
        target = min(duration, cursor + target_edit)
        if use_lyric_timing:
            end = _choose_next_lyric_boundary(
                cursor, hard_limit, lyric_bounds, lyric_starts, section_bounds,
                project.analysis.beats, project, max_frames
            )
        else:
            # Instrumental/default mode: music chooses the physical cut first.
            # MiniMax frame fitting happens afterwards when the shot is constructed.
            end = _choose_next_instrumental_boundary(
                cursor, hard_limit, project.analysis.sections, project.analysis.beats,
                project, max_frames
            )
            if end <= cursor + 1.25 or end > hard_limit + 0.05:
                end = min(hard_limit, target)
        if duration - end < 1.0:
            end = duration
        if end <= cursor + 0.20:
            end = min(duration, cursor + usable_edit_max)
        edit_ranges.append((round(cursor, 3), round(end, 3)))
        cursor = end

    # Avoid a tiny final physical clip: fold it into the previous edit interval when
    # the previous generation can still cover it. Otherwise keep it; assembly is best effort.
    if len(edit_ranges) >= 2 and edit_ranges[-1][1] - edit_ranges[-1][0] < 1.25:
        prev = edit_ranges[-2]
        merged = (prev[0], edit_ranges[-1][1])
        needed = merged[1] - merged[0] + project.head_padding + project.tail_padding
        if needed <= max_gen_seconds + 0.05:
            edit_ranges[-2:] = [merged]

    shots: List[MusicShot] = []
    all_visual_boundaries = sorted(set(lyric_bounds + lyric_starts + section_bounds))
    for idx, (edit_start, edit_end) in enumerate(edit_ranges, start=1):
        generation_start = max(0.0, edit_start - project.head_padding)
        trim_in = edit_start - generation_start
        required = trim_in + (edit_end - edit_start) + project.tail_padding
        frames = nearest_generation_frames(required, max_frames, prefer_longer=True)
        generation_duration = frames / FPS
        generation_end = min(duration, generation_start + generation_duration)
        # If the requested range is too close to the song end, the audio slice may be
        # shorter than the MiniMax generation. This is allowed; final trim remains authoritative.
        internal = [
            b for b in all_visual_boundaries
            if edit_start + 0.30 < b < edit_end - 0.30
        ]
        lyric_text = _lyrics_overlapping(edit_start, edit_end, lyrics)
        section = _section_at((edit_start + edit_end) * 0.5, project.analysis.sections)
        shots.append(
            MusicShot(
                index=idx,
                edit_start=edit_start,
                edit_end=edit_end,
                frames=frames,
                generation_start=round(generation_start, 3),
                generation_end=round(generation_end, 3),
                trim_in=round(trim_in, 3),
                lyrics=lyric_text,
                section=section,
                internal_cuts=internal,
            )
        )
    return shots


def _selected_refs_for_shot(project: MusicProject, shot: MusicShot) -> List[ReferenceAsset]:
    by_name = {r.name: r for r in project.references if r.enabled and Path(r.path).is_file()}
    return [by_name[name] for name in shot.reference_names if name in by_name][:9]


def _h3_ref_subject_definitions(project: MusicProject, refs: Sequence[ReferenceAsset], *, has_lyrics: bool = True) -> List[str]:
    lines: List[str] = []
    subject_no = 0
    for picture_no, ref in enumerate(refs, 1):
        name = (ref.name or f"reference {picture_no}").strip()
        detail = re.sub(r"\s+", " ", (ref.description or "").strip())
        kind = _normalise_reference_kind(ref.kind)
        if kind == "Picture / Composition anchor":
            text = f"<Picture {picture_no}> is the composition/shot-planning reference named {name}. Use its framing, spatial arrangement and visual layout as a concrete shot anchor."
        else:
            subject_no += 1
            if kind == "Character":
                purpose = "reusable character/person"
            elif kind == "Background / Location":
                purpose = "reusable background/location environment"
            elif kind == "Object / Prop":
                purpose = "reusable object/prop"
            elif kind == "Style / Mood":
                purpose = "reusable visual style/mood reference"
            else:
                purpose = "reusable referenced subject"
            text = f"<Subject {subject_no}> is {name}, the {purpose} defined by <Picture {picture_no}>."
        if detail:
            text += f" Purpose/details: {detail}."
        lines.append(text)
    if project.characters_subjects.strip():
        lines.append(
            "Project reference-purpose details: "
            + re.sub(r"\s+", " ", project.characters_subjects.strip())
            + ". Apply these named roles, outfit/prop rules, environment rules and continuity details to the matching references above."
        )
    if has_lyrics:
        lines.append(
            "<Audio 1> is the supplied song segment for this shot. It is the authoritative music/performance source for vocals, lyrics, rhythm, beat, timing and continuity."
        )
    else:
        lines.append(
            "<Audio 1> is the supplied instrumental song segment for this shot. It is the authoritative audio source for rhythm, beat, timing, continuity and musical energy. It contains no requested vocal performance."
        )
    return lines


def _h3_reference_labels(refs: Sequence[ReferenceAsset]) -> Dict[str, List[Tuple[int, int, ReferenceAsset]]]:
    """Return H3 labels grouped by semantic role.

    Tuple format is ``(subject_no, picture_no, ref)``. Composition anchors have a
    subject number of 0 because they remain standalone <Picture N> references.
    """
    groups: Dict[str, List[Tuple[int, int, ReferenceAsset]]] = {
        "characters": [],
        "backgrounds": [],
        "objects": [],
        "styles": [],
        "others": [],
        "pictures": [],
    }
    subject_no = 0
    for picture_no, ref in enumerate(refs, 1):
        kind = _normalise_reference_kind(ref.kind)
        if kind == "Picture / Composition anchor":
            groups["pictures"].append((0, picture_no, ref))
            continue
        subject_no += 1
        item = (subject_no, picture_no, ref)
        if kind == "Character":
            groups["characters"].append(item)
        elif kind == "Background / Location":
            groups["backgrounds"].append(item)
        elif kind == "Object / Prop":
            groups["objects"].append(item)
        elif kind == "Style / Mood":
            groups["styles"].append(item)
        else:
            groups["others"].append(item)
    return groups


def _creative_pool_items(value: str, *, allow_commas: bool = False) -> List[str]:
    """Parse a Creative brief field as an option pool without breaking real sequences.

    New lines, semicolons and pipes are explicit option separators.  Location/camera
    fields also accept simple comma-separated lists, but connected choreography such as
    "from the basement through the hallway to the roof" or "whip pan then rack focus"
    stays one instruction.
    """
    raw = str(value or "").strip()
    if not raw:
        return []

    sequence_markers = (
        " then ", " followed by ", " before ", " after ", " while ",
        " from ", " through ", " into ", " to the ", " and then ", "->", "→",
    )
    normalized_raw = re.sub(r"\s+", " ", raw)
    low = f" {normalized_raw.lower()} "
    connected_sequence = any(marker in low for marker in sequence_markers)

    parts = re.split(r"[\r\n;|]+", raw)
    parts = [re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", p).strip() for p in parts]
    parts = [p for p in parts if p]

    if allow_commas and len(parts) == 1 and not connected_sequence and "," in parts[0]:
        comma_parts = [p.strip() for p in parts[0].split(",") if p.strip()]
        # Commas are treated as a list only when they look like compact alternatives,
        # not a long prose sentence containing incidental commas.
        if len(comma_parts) >= 2 and max(len(p) for p in comma_parts) <= 140:
            parts = comma_parts

    out: List[str] = []
    seen = set()
    for part in parts:
        clean = re.sub(r"\s+", " ", part).strip(" .")
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _creative_pool_choice(value: str, shot_index: int, *, allow_commas: bool = False) -> Tuple[str, int]:
    """Return this shot's option and the total pool size, rotating before reuse."""
    items = _creative_pool_items(value, allow_commas=allow_commas)
    if not items:
        return "", 0
    idx = max(0, int(shot_index or 1) - 1) % len(items)
    return items[idx], len(items)


def build_h3_reference_prompt(
    project: MusicProject,
    shot: MusicShot,
    selected_refs: Sequence[ReferenceAsset],
) -> str:
    """Build a MiniMax-H3 full-reference prompt using the official Ref2VA layout.

    Reference roles stay semantically separate: characters are performers, locations
    are environments, props are objects, style refs guide appearance, and composition
    pictures anchor framing. This prevents a background reference from being treated as
    another visible person/performer.
    """
    refs = list(selected_refs)[:9]
    lyric_text = _clean_whisper_lyric_text(shot.lyrics)
    has_lyrics = bool(lyric_text)

    # 1) subject_definitions
    sections: List[str] = ["subject_definitions:"]
    sections.extend(_h3_ref_subject_definitions(project, refs, has_lyrics=has_lyrics))

    # Keep role groups separate throughout the prompt. A background may legitimately be
    # a reusable <Subject N> in H3, but it must never be grouped with character performers.
    groups = _h3_reference_labels(refs)
    characters = groups["characters"]
    backgrounds = groups["backgrounds"]
    objects = groups["objects"]
    styles = groups["styles"]
    others = groups["others"]
    picture_anchors = groups["pictures"]

    def labels(items: Sequence[Tuple[int, int, ReferenceAsset]]) -> str:
        return ", ".join(f"<Subject {subject_no}>" for subject_no, _picture_no, _ref in items if subject_no)

    # Keep the visual-text rule close to the reference definitions so written lyrics are
    # not mistaken for a request to create karaoke/subtitle graphics later in the prompt.
    sections.append("visual_text_policy:")
    if not has_lyrics:
        sections.append(
            "This is an instrumental-only shot. There is no speech, no singing, no narration, no lip-sync, no spoken words, and no visible subtitles or lyric text. "
            "Only physical scene text explicitly requested by the user may appear. Do not invent dialogue, captions, karaoke text, lyric overlays, title cards, lower-thirds or floating text."
        )
    elif bool(getattr(project, "visible_lyric_subtitles", False)):
        sections.append(
            "Visible lyric subtitles are enabled. The current lyric phrase from <Audio 1> may appear as synchronized readable subtitle/lyric text. "
            "Do not invent unrelated captions, title cards, logos, lower-thirds or other text. Text explicitly requested as a physical part of the scene may also remain visible."
        )
    else:
        sections.append(
            "Lyrics and dialogue inside <d>...</d> are audio/performance instructions only. They are heard and lip-synced, never displayed as subtitles, captions, karaoke lyrics, lyric overlays, title cards, lower-thirds or floating text. "
            "Only text explicitly requested as a physical part of a scene, object, poster, sign, monitor or interface may be visible. Do not turn spoken or sung words into on-screen text."
        )

    # 2) summary. Creative-brief lists are option pools, not a checklist for one clip.
    story_focus, story_pool_count = _creative_pool_choice(project.main_idea, shot.index, allow_commas=False)
    story = re.sub(r"\s+", " ", story_focus.strip()) if story_focus.strip() else "the current music-video story"
    summary_bits = [f"[reference generation + audio reuse] Create a music-video shot for {story}."]
    if story_pool_count > 1:
        summary_bits.append("This is one selected story beat from the larger project idea; keep the other listed story beats for other clips instead of combining them here.")
    if characters:
        if has_lyrics:
            summary_bits.append(f"Use {labels(characters)} as the visible character/performer reference{'s' if len(characters) != 1 else ''}.")
        else:
            summary_bits.append(f"Use {labels(characters)} as the visible character reference{'s' if len(characters) != 1 else ''}; this interval is instrumental-only, so the character action is visual rather than vocal.")
    if backgrounds:
        summary_bits.append(f"Use {labels(backgrounds)} only as the scene environment/background reference{'s' if len(backgrounds) != 1 else ''}, not as additional people or performers.")
    if objects:
        summary_bits.append(f"Use {labels(objects)} only as referenced objects/props when they belong in the scene.")
    if styles:
        summary_bits.append(f"Use {labels(styles)} only for visual style/mood guidance.")
    if others:
        summary_bits.append(f"Use {labels(others)} according to their explicitly defined purposes.")
    if picture_anchors:
        picture_summary = ", ".join(f"<Picture {picture_no}>" for _subject_no, picture_no, _ref in picture_anchors)
        summary_bits.append(f"Use {picture_summary} as concrete composition/shot-planning anchors rather than as character identities.")
    if has_lyrics:
        summary_bits.append("Reuse <Audio 1> as the continuous authoritative song/performance source for this shot.")
    else:
        summary_bits.append("Reuse <Audio 1> as the continuous authoritative instrumental audio source for this shot; the visible character performs silent story action while the music continues unchanged.")
    sections.append("summary:")
    sections.append(" ".join(summary_bits))

    # 3) retention_analysis
    sections.append("retention_analysis:")
    for group_name in ("characters", "backgrounds", "objects", "styles", "others"):
        for i, _picture_no, ref in groups[group_name]:
            name = (ref.name or f"Subject {i}").strip()
            detail = re.sub(r"\s+", " ", (ref.description or "").strip())
            kind = _normalise_reference_kind(ref.kind)
            if kind == "Background / Location":
                rule = f"<Subject {i}>: fully_preserved - keep {name} strictly as the referenced environment/background, preserving its recognizable room design, lighting, layout and named recurring features; it is not a person or performer."
            elif kind == "Object / Prop":
                rule = f"<Subject {i}>: fully_preserved - keep {name} strictly as the referenced object/prop with its recognizable appearance and intended purpose; it is not an additional person or performer."
            elif kind == "Style / Mood":
                rule = f"<Subject {i}>: weak_reference - use {name} only for the intended visual style/mood without turning it into a physical character or object."
            elif kind == "Character":
                rule = f"<Subject {i}>: fully_preserved - keep {name}'s referenced identity, appearance and defined character role consistent."
            else:
                rule = f"<Subject {i}>: fully_preserved - keep {name}'s explicitly defined reference role consistent."
            if detail:
                rule += f" Preserve these reference details: {detail}."
            sections.append(rule)
    for _subject_no, picture_no, ref in picture_anchors:
        name = (ref.name or f"Picture {picture_no}").strip()
        detail = re.sub(r"\s+", " ", (ref.description or "").strip())
        rule = f"<Picture {picture_no}>: partially_preserved - use {name} as the requested framing/composition anchor while allowing scene motion and shot development."
        if detail:
            rule += f" Follow these anchor details: {detail}."
        sections.append(rule)
    if project.characters_subjects.strip() and refs:
        sections.append(
            "Named subject rules: fully_preserved - "
            + re.sub(r"\s+", " ", project.characters_subjects.strip())
            + "."
        )
    if has_lyrics:
        sections.append(
            "<Audio 1>: fully_copy - keep the supplied song segment continuous as the target performance track; preserve its existing vocals, lyrics, instrumental passages, rhythm and timing instead of inventing replacement vocals or music."
        )
    else:
        sections.append(
            "<Audio 1>: fully_copy - keep the supplied instrumental audio unchanged as the complete sound for this shot. Do not add any extra voice, speech, singing, humming, mumbling, narration or replacement music. Preserve its rhythm, beat, timing, continuity and musical energy."
        )

    # 4) detailed_description
    sections.append("detailed_description:")
    style = re.sub(r"\s+", " ", project.style_theme.strip()) if project.style_theme.strip() else "cinematic music-video"
    location, location_pool_count = _creative_pool_choice(project.locations_world, shot.index, allow_commas=True)
    location = re.sub(r"\s+", " ", location.strip())
    if location:
        # Avoid awkward output such as "set in in the basement" when the user already
        # typed a leading "in".
        location_text = re.sub(r"^(?:set\s+in\s+|in\s+)", "", location, flags=re.IGNORECASE).strip()
        sections.append(f"The target video uses a {style} style and takes place in {location_text}.")
        if location_pool_count > 1:
            sections.append("Use only this selected project location as the primary setting for this clip. The other locations listed in the Creative brief are alternatives reserved for other clips; do not visit or combine them in this shot unless the selected location itself explicitly describes a continuous transition.")
    else:
        sections.append(f"The target video uses a {style} style.")

    shot1: List[str] = ["[Shot 1]"]
    if characters:
        if len(characters) == 1:
            shot1.append(f"The visible character is {labels(characters)}; preserve this character's identity and role.")
        else:
            shot1.append(f"The visible characters are {labels(characters)}; preserve their identities and roles.")
    if backgrounds:
        shot1.append(f"Place the character action inside {labels(backgrounds)} as the referenced background/location; preserve the environment but do not create a second person from it.")
    if objects:
        shot1.append(f"Use {labels(objects)} only as props/objects where relevant to the action.")
    if styles:
        shot1.append(f"Apply {labels(styles)} only as visual style/mood guidance.")
    if picture_anchors:
        first_pictures = ", ".join(f"<Picture {picture_no}>" for _subject_no, picture_no, _ref in picture_anchors)
        shot1.append(f"Use {first_pictures} for the specified composition/framing purpose.")
    if story_focus.strip():
        shot1.append(re.sub(r"\s+", " ", story_focus.strip()) + ".")
    shot1.append(f"This is the {shot.section or 'current'} section of the music video.")
    camera_choice, camera_pool_count = _creative_pool_choice(project.camera_choreography, shot.index, allow_commas=True)
    if camera_choice.strip():
        shot1.append("Primary camera concept: " + re.sub(r"\s+", " ", camera_choice.strip()) + ".")
        if camera_pool_count > 1:
            shot1.append("Use this one selected camera concept as the dominant camera language for the clip. Do not stack the other camera moves from the Creative brief into this shot; they are reserved for later clips.")
    if has_lyrics:
        shot1.append("<Audio 1> begins immediately and remains continuous and synchronized with the visible vocal performance.")
    else:
        shot1.append(
            "<Audio 1> begins immediately as an instrumental-only passage and remains continuous. "
            "Every visible character remains completely silent. Use expressive body movement, arm movement, eye movement and silent facial acting to communicate the story and react to the beat. "
            "Mouth movement stays minimal, natural and non-speaking, with no lip-sync, no sung articulation and no speech-like mouth shapes."
        )

    if has_lyrics:
        lyric = lyric_text
        visible_subtitles = bool(getattr(project, "visible_lyric_subtitles", False))
        if len(characters) == 1:
            performer = f"<Subject {characters[0][0]}>"
            if visible_subtitles:
                shot1.append(
                    f"Lyric performance: <d>[Original language] {lyric}</d>. {performer} is the singer/performer and lip-syncs/sings these words in time with <Audio 1>. Visible lyric subtitles are enabled, so the current phrase may also appear as synchronized readable subtitle text."
                )
            else:
                shot1.append(
                    f"Audio-only lyric performance: <d>[Original language] {lyric}</d>. {performer} is the singer/performer and lip-syncs/sings these words in time with <Audio 1>. These lyric words are heard and lip-synced only; do not display them anywhere in the image."
                )
        elif len(characters) > 1:
            suffix = " Visible lyric subtitles are enabled for this phrase." if visible_subtitles else " The lyric words are audio-only and must not be displayed visually."
            shot1.append(
                f"Lyric performance: <d>[Original language] {lyric}</d>. Only the character whose user-defined purpose identifies them as the singer/lead vocalist performs the lyric; keep every other referenced character in their defined role." + suffix
            )
        else:
            if visible_subtitles:
                shot1.append(
                    f"Lyric performance: <d>[Original language] {lyric}</d>. Synchronize the visible performance and mouth movement to <Audio 1>. Visible lyric subtitles are enabled and may show the current phrase."
                )
            else:
                shot1.append(
                    f"Audio-only lyric performance: <d>[Original language] {lyric}</d>. Synchronize the visible performance and mouth movement to <Audio 1>. These words are heard only and must not appear visually."
                )
    else:
        shot1.append(
            "Instrumental interval: <Audio 1> contains the complete intended sound for this section. "
            "Translate every story idea into visible physical action rather than spoken or sung content. Characters may dance, gesture, react, work, move through the environment and interact with props, but they do not talk, sing, narrate, hum, mumble or mouth words. "
            "Do not interpret phrases such as asks, tells, explains, argues, jokes, calls, sings or says as permission to create audible speech; express the intended meaning silently through action and reaction instead."
        )
    sections.append(" ".join(shot1))

    for cut_no, absolute in enumerate(shot.internal_cuts[:4], start=2):
        rel = max(0.0, absolute - shot.generation_start)
        mm = int(rel // 60)
        ss = rel - mm * 60
        timestamp = f"{mm:02d}:{ss:06.3f}"
        if has_lyrics:
            continuity = "<Audio 1> continues seamlessly across the cut with unchanged musical timing and vocal continuity."
        else:
            continuity = "<Audio 1> continues seamlessly across the cut as the same instrumental passage; every character remains completely silent with minimal natural non-speaking mouth movement and communicates only through visible action and reaction."
        sections.append(
            f"[Shot {cut_no}] At {timestamp}, the camera cuts to a complementary new angle or visual beat within the same established scene. "
            "Keep the same character identities, performer roles, props and background/location roles. " + continuity
        )

    if bool(getattr(project, "visible_lyric_subtitles", False)):
        sections.append("Keep all visible character identities and all reference-purpose details stable across shots. Background/location references remain environments, and prop references remain objects. Start useful action immediately. Do not invent unrelated captions or title cards beyond the explicitly enabled lyric subtitles and explicitly requested physical scene text.")
    else:
        if has_lyrics:
            sections.append("Keep all visible character identities and all reference-purpose details stable across shots. Background/location references remain environments, and prop references remain objects. Start useful action immediately. No visible lyric subtitles, captions, karaoke text, lyric overlays or title cards; only explicitly requested physical scene text may appear.")
        else:
            sections.append("Keep all visible character identities and all reference-purpose details stable across shots. Background/location references remain environments, and prop references remain objects. Start useful action immediately. Do not create any extra person, assistant, mascot, floating icon or shoulder creature. No speech, singing, narration, lip-sync, humming, mumbling, visible lyric subtitles, captions, karaoke text, lyric overlays or title cards; only explicitly requested physical scene text may appear.")

    # 5/6) audio sections. The master/reference song is the only music layer we want.
    sections.append("overall_soundscape:")
    if has_lyrics:
        sections.append(
            "Natural physical ambience and incidental scene sounds may be subtle and secondary. Keep <Audio 1> dominant, continuous and clearly synchronized with the vocal performance."
        )
    else:
        sections.append(
            "<Audio 1> is the complete instrumental soundtrack for this interval. Natural physical ambience may be subtle and secondary, but must not replace, interrupt or compete with the supplied instrumental audio. Every visible character remains a silent visual participant rather than a speaker or singer."
        )
    sections.append("non_diegetic_music:")
    if has_lyrics:
        sections.append("<Audio 1> is directly reused as the complete music track for this shot. Do not add a separate score or replacement music.")
    else:
        sections.append("<Audio 1> is directly reused as the complete music track for this shot. Do not add a separate score, voice, vocalization or replacement music.")

    return "\n".join(x.strip() for x in sections if x and x.strip())

def build_default_prompt(project: MusicProject, shot: MusicShot) -> str:
    return build_h3_reference_prompt(project, shot, _selected_refs_for_shot(project, shot))


def build_generation_prompt(project: MusicProject, shot: MusicShot, selected_refs: List[ReferenceAsset]) -> str:
    """Return the exact Director prompt for generation when one exists.

    The Director prompt editor is authoritative.  Rebuilding the prompt here from the
    project Idea/Story box used to silently discard manual Director edits on retry and
    recreate, which could re-introduce story text the user had explicitly removed.
    Only legacy/restored shots with no saved Director prompt are rebuilt automatically.
    """
    saved_prompt = (shot.prompt or "").strip()
    if saved_prompt:
        return saved_prompt
    prompt = build_h3_reference_prompt(project, shot, selected_refs)
    shot.prompt = prompt
    return prompt



def _randomized_character_reference_names(project: MusicProject, shot: MusicShot, enabled: Sequence[ReferenceAsset]) -> List[str]:
    """Return a stable per-shot random character subset when the feature is enabled.

    Only enabled Character references participate. One or two characters are selected
    per shot (or fewer when fewer are available). The project-level seed is refreshed
    when the plan/prompts are rebuilt, then saved with the project so retries keep the
    same shot-to-reference mapping.
    """
    if not bool(getattr(project, "randomize_reference_characters", False)):
        return []
    chars = [r for r in enabled if _normalise_reference_kind(r.kind) == "Character" and r.name]
    if not chars:
        return []
    max_count = min(2, len(chars))
    try:
        base_seed = int(getattr(project, "reference_random_seed", -1))
    except Exception:
        base_seed = -1
    if base_seed < 0:
        base_seed = random.SystemRandom().randint(0, 2_147_483_647)
        try:
            project.reference_random_seed = int(base_seed)
        except Exception:
            pass
    rng = random.Random(f"{base_seed}:{int(getattr(shot, 'index', 0) or 0)}:{len(chars)}")
    count = rng.randint(1, max_count)
    names = [r.name for r in chars]
    if count >= len(names):
        rng.shuffle(names)
        return names
    return list(rng.sample(names, count))


def auto_assign_references(project: MusicProject, shot: MusicShot) -> List[str]:
    """Conservative first-pass router with optional per-shot random character refs.

    Users can edit the assignment in the shot table. This avoids sending all nine references
    blindly while still giving the director useful defaults. When random character refs are
    enabled, one or two Character references are chosen per shot while non-character
    context (background/style plus explicitly named props/anchors) stays stable.
    """
    enabled = [r for r in project.references if r.enabled and Path(r.path).is_file()]
    randomize_chars = bool(getattr(project, "randomize_reference_characters", False))
    blob = " ".join((shot.prompt, shot.lyrics, project.characters_subjects, project.main_idea)).lower()
    chosen: List[str] = []
    for ref in enabled:
        if ref.name and ref.name.lower() in blob:
            kind = _normalise_reference_kind(ref.kind)
            if randomize_chars and kind == "Character":
                continue
            chosen.append(ref.name)
    # Background/location and style refs are normally project-wide context, so include
    # them even when the user did not repeat their names in every shot prompt.
    for ref in enabled:
        kind = _normalise_reference_kind(ref.kind)
        if kind in ("Background / Location", "Style / Mood") and ref.name not in chosen:
            chosen.append(ref.name)
    if randomize_chars:
        for name in _randomized_character_reference_names(project, shot, enabled):
            if name not in chosen:
                chosen.append(name)
    elif not any(_normalise_reference_kind(r.kind) == "Character" and r.name in chosen for r in enabled):
        # Non-random mode must still use the whole enabled character pool over the
        # course of a music video.  The old fallback always selected chars[:2],
        # which meant references 3..N could never reach MiniMax unless their names
        # were explicitly written into a shot prompt.  Use a stable round-robin
        # pair instead: shot 1 -> refs 1+2, shot 2 -> refs 3+4, shot 3 -> refs 5+1,
        # etc.  This keeps retries/rebuilds deterministic while giving every enabled
        # character a turn.  A single available character is simply reused.
        chars = [r.name for r in enabled if _normalise_reference_kind(r.kind) == "Character" and r.name]
        if chars:
            try:
                shot_index = max(1, int(getattr(shot, "index", 1) or 1))
            except Exception:
                shot_index = 1
            pair_count = min(2, len(chars))
            start = ((shot_index - 1) * pair_count) % len(chars)
            for offset in range(pair_count):
                name = chars[(start + offset) % len(chars)]
                if name not in chosen:
                    chosen.append(name)
    # Objects/props and picture/composition anchors remain opt-in by name/purpose so an
    # unrelated prop or storyboard image is not injected into every shot automatically.
    return chosen[:9]


# ----------------------------- worker threads ------------------------------


class FunctionWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            result = self.fn(self.progress.emit, *self.args, **self.kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


def _analysis_task(progress, audio_path: str, sensitivity: int) -> AnalysisResult:
    progress("Analyzing music energy and beat candidates...")
    result = analyze_music(audio_path, sensitivity)
    progress(f"Analysis complete: {len(result.beats)} beat candidates, {len(result.sections)} sections.")
    return result


def _whisper_cli_path() -> Optional[Path]:
    direct = WHISPER_DIR / "whisper-cli.exe"
    if direct.is_file():
        return direct
    if WHISPER_DIR.is_dir():
        for candidate in WHISPER_DIR.rglob("whisper-cli.exe"):
            if candidate.is_file():
                return candidate
    return None


def _whisper_cpp_ready() -> bool:
    return _whisper_cli_path() is not None and WHISPER_MODEL.is_file() and WHISPER_MODEL.stat().st_size > 100_000_000


def _download_with_resume(url: str, destination: Path, label: str, progress) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            existing = part.stat().st_size if part.is_file() else 0
            headers = {"User-Agent": "MiniMax-H3-Music-Clip-Creator/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                code = getattr(response, "status", 200)
                append = existing > 0 and code == 206
                if not append and existing:
                    existing = 0
                total_header = response.headers.get("Content-Length")
                total = (int(total_header) + existing) if total_header and append else (int(total_header) if total_header else 0)
                mode = "ab" if append else "wb"
                done = existing
                last_report = -1
                with open(part, mode) as f:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = int(done * 100 / total)
                            if pct >= last_report + 5:
                                last_report = pct
                                progress(f"Downloading {label}: {pct}% ({done / (1024**2):.0f}/{total / (1024**2):.0f} MiB)")
                        elif done // (64 * 1024 * 1024) != (done - len(chunk)) // (64 * 1024 * 1024):
                            progress(f"Downloading {label}: {done / (1024**2):.0f} MiB")
            if not part.is_file() or part.stat().st_size < 1024:
                raise RuntimeError(f"Downloaded {label} is empty.")
            os.replace(str(part), str(destination))
            return destination
        except Exception as exc:
            if attempt >= 3:
                raise RuntimeError(f"Could not download {label}: {exc}") from exc
            progress(f"{label} download retry {attempt}/3...")
            time.sleep(1.0 * attempt)
    return destination


def _install_whisper_runtime(runtime_zip: Path) -> None:
    extract_dir = WHISPER_DIR / "_runtime_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(runtime_zip, "r") as zf:
        zf.extractall(extract_dir)
    exe = next((x for x in extract_dir.rglob("whisper-cli.exe") if x.is_file()), None)
    if exe is None:
        raise RuntimeError("Downloaded whisper.cpp runtime does not contain whisper-cli.exe")
    # Current Windows releases dynamically link the DLLs beside whisper-cli.exe.
    # Flatten that runtime directory into presets/bin/whisper so the folder is portable.
    for item in exe.parent.iterdir():
        if item.is_file():
            shutil.copy2(item, WHISPER_DIR / item.name)
    shutil.rmtree(extract_dir, ignore_errors=True)
    runtime_zip.unlink(missing_ok=True)


def _download_whisper_cpp_task(progress) -> str:
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    if _whisper_cpp_ready():
        return str(WHISPER_DIR)
    progress("Downloading standalone multilingual Whisper.cpp Small model and runtime...")
    # Runtime and model are independent, so download them simultaneously. The .part
    # files are retained on interruption and HTTP Range is used on the next attempt.
    jobs = []
    if _whisper_cli_path() is None:
        jobs.append((WHISPER_RUNTIME_URL, WHISPER_RUNTIME_ZIP, "Whisper.cpp runtime"))
    if not WHISPER_MODEL.is_file() or WHISPER_MODEL.stat().st_size < 100_000_000:
        jobs.append((WHISPER_MODEL_URL, WHISPER_MODEL, "Whisper Small multilingual model"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(_download_with_resume, url, dest, label, progress) for url, dest, label in jobs]
        for future in futures:
            future.result()
    if WHISPER_RUNTIME_ZIP.is_file():
        progress("Installing Whisper.cpp runtime...")
        _install_whisper_runtime(WHISPER_RUNTIME_ZIP)
    if not _whisper_cpp_ready():
        raise RuntimeError("Whisper.cpp installation validation failed. whisper-cli.exe or ggml-small.bin is missing.")
    progress("Whisper.cpp Small multilingual is ready.")
    return str(WHISPER_DIR)


def _parse_whisper_json_segments(data: Dict[str, Any]) -> List[LyricSegment]:
    lyrics: List[LyricSegment] = []
    for seg in data.get("transcription", []) if isinstance(data, dict) else []:
        if not isinstance(seg, dict):
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text") or "")).strip()
        offsets = seg.get("offsets") if isinstance(seg.get("offsets"), dict) else {}
        try:
            start = float(offsets.get("from", 0)) / 1000.0
            end = float(offsets.get("to", 0)) / 1000.0
        except Exception:
            continue
        if text and end > start:
            lyrics.append(LyricSegment(start, end, text))
    return lyrics


# Whisper Small is intentionally used to keep the portable standalone install light.
# Its transcription is good enough for lyric-aware planning, but its raw segment cuts
# can occur in the middle of a sentence/lyric phrase.  The planner only needs *safe*
# phrase ends, so clean the segmentation before exposing it as MusicProject.lyrics.
_WHISPER_CONTINUATION_WORDS = {
    "a", "an", "and", "as", "at", "because", "but", "by", "for", "from", "if",
    "in", "into", "is", "it", "of", "on", "or", "so", "than", "that", "the",
    "then", "through", "to", "when", "where", "while", "with", "without", "yet",
    "i", "i'm", "i've", "i'll", "you", "you're", "we", "we're", "they", "they're",
    "my", "your", "our", "their", "this", "these", "those", "one", "two", "every",
}


_WHISPER_NON_LYRIC_EVENTS = {
    "music", "instrumental", "instrumental music", "intro", "outro", "silence",
    "silent", "noise", "background music", "applause", "clapping", "cheering",
    "laughter", "laughing", "singing", "vocalizing", "humming", "hum",
}


def _clean_whisper_lyric_text(text: str) -> str:
    """Return actual lyric words only; Whisper event markers such as [Music] are not lyrics."""
    value = re.sub(r"\s+", " ", text or "").strip()
    if not value:
        return ""

    # Remove leading/trailing bracketed event tags only when the tag is a known
    # non-lexical audio event. Preserve ordinary brackets that are part of real text.
    event_pattern = re.compile(r"^\s*[\[(]([^\]\)]+)[\])]")
    while True:
        match = event_pattern.match(value)
        if not match:
            break
        event = re.sub(r"[^a-z ]+", " ", match.group(1).lower())
        event = re.sub(r"\s+", " ", event).strip()
        if event not in _WHISPER_NON_LYRIC_EVENTS:
            break
        value = value[match.end():].strip(" -–—:;,.\t")

    # A segment consisting solely of a non-lyrical marker is intentionally empty.
    bare = re.sub(r"^[\[(]|[\])]$", "", value.strip()).strip().lower()
    bare = re.sub(r"[^a-z ]+", " ", bare)
    bare = re.sub(r"\s+", " ", bare).strip()
    if bare in _WHISPER_NON_LYRIC_EVENTS:
        return ""
    return value


def _whisper_phrase_cleanup(raw: Sequence[LyricSegment]) -> List[LyricSegment]:
    """Merge obviously broken Whisper.cpp Small segments into safer lyric phrases.

    This deliberately errs toward a slightly longer phrase: for music-video planning a
    late cut is preferable to a cut halfway through a sung phrase.  Strong punctuation
    or a real timestamp gap remains a boundary.  Very long merged phrases are capped so
    the director still receives useful edit anchors.
    """
    cleaned_raw: List[LyricSegment] = []
    for seg in raw:
        text = _clean_whisper_lyric_text(seg.text)
        if text:
            cleaned_raw.append(LyricSegment(float(seg.start), float(seg.end), text))
    if not cleaned_raw:
        return []

    result: List[LyricSegment] = []
    cur_start = float(cleaned_raw[0].start)
    cur_end = float(cleaned_raw[0].end)
    cur_text = cleaned_raw[0].text

    def last_word(text: str) -> str:
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", text.lower())
        return words[-1] if words else ""

    def strong_end(text: str) -> bool:
        return bool(re.search(r"[.!?][\"')\]]*$", text.strip()))

    for nxt in cleaned_raw[1:]:
        nxt_text = nxt.text
        if not nxt_text:
            continue
        gap = max(0.0, float(nxt.start) - cur_end)
        merged_duration = float(nxt.end) - cur_start
        current_duration = cur_end - cur_start
        tail = last_word(cur_text)

        # A noticeable pause or strong sentence punctuation is a trustworthy phrase end.
        boundary_is_safe = gap >= 0.38 or strong_end(cur_text)

        # Small's most damaging cuts are contiguous fragments such as "down the" /
        # "page ...".  Always join those.  Also join short contiguous fragments because
        # Whisper's segment boundary is not a lyric boundary in that case.
        clearly_continues = (
            gap <= 0.20 and (
                tail in _WHISPER_CONTINUATION_WORDS
                or current_duration < 3.0
                or len(cur_text.split()) < 7
            )
        )

        # When punctuation is absent (common on sung material), allow contiguous chunks
        # to grow to a normal lyric-phrase size, but do not merge half a verse forever.
        phrase_needs_more_context = (
            gap <= 0.12
            and not boundary_is_safe
            and current_duration < 5.2
            and merged_duration <= 8.5
        )

        should_merge = clearly_continues or phrase_needs_more_context
        if should_merge and merged_duration <= 9.5:
            cur_text = (cur_text + " " + nxt_text).strip()
            cur_end = float(nxt.end)
            continue

        result.append(LyricSegment(round(cur_start, 3), round(cur_end, 3), cur_text))
        cur_start = float(nxt.start)
        cur_end = float(nxt.end)
        cur_text = nxt_text

    if cur_text:
        result.append(LyricSegment(round(cur_start, 3), round(cur_end, 3), cur_text))
    return result


def _whisper_task(progress, audio_path: str) -> List[LyricSegment]:
    cli = _whisper_cli_path()
    if cli is None or not WHISPER_MODEL.is_file():
        raise RuntimeError("Standalone Whisper.cpp is not installed. Use 'Download Whisper.cpp Small' on the Song Analysis tab first.")
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found; it is required to prepare audio for Whisper.cpp.")
    progress("Preparing 16 kHz mono audio for Whisper.cpp...")
    payload_dir = OUTPUT_ROOT / "_temp"
    payload_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    wav_path = payload_dir / f"whisper_input_{stamp}.wav"
    result_prefix = payload_dir / f"whisper_result_{stamp}"
    result_json = result_prefix.with_suffix(".json")
    try:
        cp = subprocess.run(
            [ffmpeg, "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if cp.returncode != 0 or not wav_path.is_file():
            raise RuntimeError("Could not prepare audio for Whisper.cpp:\n" + (cp.stderr or cp.stdout or ""))
        threads = max(2, min(12, (os.cpu_count() or 8) // 2))
        progress(f"Transcribing with standalone Whisper.cpp Small multilingual ({threads} CPU threads, language auto-detect)...")
        cmd = [
            str(cli), "-m", str(WHISPER_MODEL), "-f", str(wav_path),
            "-l", "auto", "-t", str(threads), "-oj", "-of", str(result_prefix), "-np",
        ]
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(cli.parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if cp.returncode != 0:
            raise RuntimeError("Whisper.cpp transcription failed:\n" + (cp.stderr or cp.stdout or ""))
        if not result_json.is_file():
            raise RuntimeError("Whisper.cpp completed but did not create its JSON result.")
        data = json.loads(result_json.read_text(encoding="utf-8-sig"))
        raw_lyrics = _parse_whisper_json_segments(data)
        lyrics = _whisper_phrase_cleanup(raw_lyrics)
        language = ""
        try:
            language = str(data.get("result", {}).get("language") or "")
        except Exception:
            pass
        suffix = f" ({language})" if language else ""
        if len(lyrics) != len(raw_lyrics):
            progress(f"Whisper.cpp complete: {len(raw_lyrics)} raw segments -> {len(lyrics)} cleaned lyric phrases{suffix}.")
        else:
            progress(f"Whisper.cpp complete: {len(lyrics)} cleaned lyric phrases{suffix}.")
        return lyrics
    finally:
        # whisper.cpp may create sidecar files next to the requested JSON prefix.
        for cleanup in [wav_path, *payload_dir.glob(result_prefix.name + "*")]:
            try:
                cleanup.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            if payload_dir.is_dir() and not any(payload_dir.iterdir()):
                payload_dir.rmdir()
        except Exception:
            pass

def _extract_audio_slice(audio: str, out_path: Path, start: float, duration: float) -> None:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-ss", f"{max(0.0, start):.6f}", "-i", audio,
        "-t", f"{max(0.10, duration):.6f}", "-vn", "-ac", "2", "-ar", "32000",
        "-c:a", "pcm_s16le", str(out_path),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if cp.returncode != 0 or not out_path.is_file():
        raise RuntimeError("Could not create Ref2VA audio slice:\n" + (cp.stderr or cp.stdout or ""))


_GENERATION_CANCEL = threading.Event()
_ACTIVE_GENERATION_PROCESS = None
_ACTIVE_GENERATION_LOCK = threading.Lock()


def _terminate_generation_process_tree() -> None:
    global _ACTIVE_GENERATION_PROCESS
    with _ACTIVE_GENERATION_LOCK:
        proc = _ACTIVE_GENERATION_PROCESS
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _recreate_output_path(raw_dir: Path, shot_index: int) -> Tuple[Path, bool]:
    """Return an output path that cannot collide with a clip already being reviewed.

    First generation keeps the stable shot_###.mp4 name. Any recreation/retry of an
    existing shot ALWAYS gets a new versioned filename. Do not probe by unlinking the
    existing clip: on Windows a player/review widget can acquire or reacquire a handle
    during the several-minute render, creating a race that only fails at final mux.
    """
    preferred = raw_dir / f"shot_{shot_index:03d}.mp4"
    if not preferred.exists():
        return preferred, False

    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = raw_dir / f"shot_{shot_index:03d}_retry_{stamp}.mp4"
    serial = 2
    while candidate.exists():
        candidate = raw_dir / f"shot_{shot_index:03d}_retry_{stamp}_{serial}.mp4"
        serial += 1
    return candidate, True


def _generation_task(progress, project: MusicProject, shot_indices: List[int]) -> List[Dict[str, Any]]:
    global _ACTIVE_GENERATION_PROCESS
    _GENERATION_CANCEL.clear()
    if not MINIMAX_PY.is_file():
        raise RuntimeError(f"MiniMax environment Python not found. Checked: {MINIMAX_ENV / 'python.exe'} and {MINIMAX_ENV / 'Scripts' / 'python.exe'}")
    if not GENERATE_REF.is_file():
        raise RuntimeError("MiniMax Ref2VA launcher not found: helpers/generate_ref.py")
    if not Path(project.audio_path).is_file():
        raise RuntimeError("The project song file is missing.")

    out_dir = Path(project.output_dir or OUTPUT_ROOT / _safe_stem(project.audio_path)).resolve()
    raw_dir = out_dir / "raw_clips"
    audio_dir = out_dir / "audio_chunks"
    logs_dir = ROOT / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    width, height = RESOLUTION_PRESETS[project.resolution][project.aspect]
    refs_by_name = {r.name: r for r in project.references if r.enabled and Path(r.path).is_file()}
    results: List[Dict[str, Any]] = []
    targets = [s for s in project.shots if s.index in shot_indices]
    for pos, shot in enumerate(targets, start=1):
        if _GENERATION_CANCEL.is_set():
            progress("Generation stopped. Completed clips were kept.")
            break
        progress(f"Generating shot {shot.index} ({pos}/{len(targets)})...")
        audio_chunk = audio_dir / f"shot_{shot.index:03d}.wav"
        _extract_audio_slice(project.audio_path, audio_chunk, shot.generation_start, shot.frames / FPS)
        out_path, used_retry_name = _recreate_output_path(raw_dir, shot.index)
        if used_retry_name:
            progress(f"Shot {shot.index}: recreation uses a new output file {out_path.name}; the existing clip is left untouched.")
        shot.output_path = str(out_path)
        selected_names = [n for n in shot.reference_names if n in refs_by_name]
        if not selected_names:
            # Older/restored projects can have an empty shot assignment even though valid
            # project references exist. Re-run the same conservative router at generation
            # time so a selected project character image is never silently dropped.
            selected_names = [n for n in auto_assign_references(project, shot) if n in refs_by_name]
        if not selected_names and refs_by_name:
            # Last-resort safety: Ref2VA music shots should not silently ignore all images.
            selected_names = [next(iter(refs_by_name))]
        selected = [refs_by_name[n] for n in selected_names[:9]]
        generation_prompt = build_generation_prompt(project, shot, selected)
        (raw_dir / f"shot_{shot.index:03d}_prompt.txt").write_text(generation_prompt, encoding="utf-8")
        cmd = [
            str(MINIMAX_PY), "-u", str(GENERATE_REF),
            "--prompt", generation_prompt,
            "--width", str(width), "--height", str(height),
            "--frames", str(shot.frames), "--steps", str(project.steps),
            "--cfg", str(project.cfg), "--seed", str(shot.seed),
            "--shift", str(project.shift), "--audio-shift", str(project.audio_shift),
            "--ref-image-size", project.ref_image_size,
            "--ref-audio", str(audio_chunk),
            "--output", str(out_path),
        ]
        if project.use_hybrid_model:
            hybrid = Path(str(project.hybrid_model_path or "").strip())
            if not hybrid.is_file():
                raise RuntimeError("Use hybrid model is enabled, but the selected hybrid .safetensors file was not found.")
            cmd += ["--ref2va-checkpoint", str(hybrid.resolve())]
        if project.vram_manager_enabled:
            cmd += ["--vram-manager-auto" if project.vram_auto_bypass else "--vram-manager"]
            cmd += [
                "--vram-residency-engine", str(project.vram_residency_engine or "static"),
                "--vram-runtime-free-gb", str(float(project.vram_runtime_free_gb)),
                "--vram-text-headroom-gb", str(float(project.vram_text_headroom_gb)),
                "--vram-diffusion-headroom-gb", str(float(project.vram_diffusion_headroom_gb)),
                "--vram-offload-chunk-mb", str(int(project.vram_offload_chunk_mb)),
                "--vram-max-resident-weights-gb", str(float(project.vram_max_resident_weights_gb)),
                "--vram-block-check-interval", str(int(project.vram_block_check_interval)),
                "--vram-async-streams", str(int(project.vram_async_streams)),
                "--vram-video-vae-reserve-gb", str(float(project.vram_video_vae_reserve_gb)),
                "--vram-audio-vae-reserve-gb", str(float(project.vram_audio_vae_reserve_gb)),
                "--vram-residency-target-free-gb", str(float(project.vram_residency_target_free_gb)),
                "--vram-residency-warmup-blocks", str(int(project.vram_residency_warmup_blocks)),
                "--vram-residency-refill-interval", str(int(project.vram_residency_refill_interval)),
            ]
            cmd += ["--vram-residency-fill" if project.vram_residency_fill else "--no-vram-residency-fill"]
        if project.sage_attention:
            cmd += ["--sage-attention"]
        if project.spectrum:
            cmd += ["--spectrum"]
        turbo_lora = str(project.turbo_lora_path or "").strip()
        if turbo_lora:
            turbo_path = Path(turbo_lora)
            if not turbo_path.is_file():
                raise RuntimeError(f"Turbo LoRA not found: {turbo_lora}")
            cmd += ["--lora", str(turbo_path.resolve()), "--lora-strength", str(float(project.turbo_lora_strength))]
        for ref in selected:
            cmd += ["--ref-image", ref.path]
        # Ref2VA only requires at least one reference. The song chunk already fills that requirement.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"minimax_music_clip_{stamp}_shot{shot.index:03d}.log"
        original_seed = int(shot.seed)
        attempt_seed = original_seed
        success = False
        last_rc = None
        last_tail: List[str] = []
        cancelled_attempt = False

        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("MiniMax Music Clip Creator - Ref2VA shot log\n")
            log.write(f"Shot: {shot.index}\n")
            log.write(f"MiniMax Python: {MINIMAX_PY}\n")
            log.write(f"Working directory: {ROOT}\n")
            log.write(f"Audio chunk: {audio_chunk}\n")
            log.write(f"Output: {out_path}\n")
            log.flush()

            for attempt in (1, 2):
                if attempt == 2:
                    if _GENERATION_CANCEL.is_set():
                        cancelled_attempt = True
                        break
                    attempt_seed = random.SystemRandom().randint(0, 2_147_483_647)
                    shot.seed = int(attempt_seed)
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    # Replace the existing CLI seed without rebuilding the complete command.
                    try:
                        seed_pos = cmd.index("--seed") + 1
                        cmd[seed_pos] = str(attempt_seed)
                    except Exception:
                        pass
                    progress(f"Shot {shot.index}: first attempt failed; retrying once with random seed {attempt_seed}...")
                    log.write(f"\n===== AUTOMATIC RETRY 1/1 | new seed {attempt_seed} =====\n")

                log.write(f"Attempt {attempt}/2 seed: {attempt_seed}\n")
                log.write("Command: " + subprocess.list2cmdline(cmd) + "\n\n")
                log.flush()
                cp = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                with _ACTIVE_GENERATION_LOCK:
                    _ACTIVE_GENERATION_PROCESS = cp
                tail: List[str] = []
                assert cp.stdout is not None
                for line in cp.stdout:
                    if _GENERATION_CANCEL.is_set() and cp.poll() is None:
                        _terminate_generation_process_tree()
                    log.write(line)
                    log.flush()
                    text = line.rstrip()
                    if text:
                        tail.append(text)
                        tail = tail[-40:]
                        if "step" in text.lower() or "saved" in text.lower() or "error" in text.lower():
                            progress(f"Shot {shot.index}: {text}")
                rc = cp.wait()
                last_rc = rc
                last_tail = list(tail)
                log.write(f"\n[Music Clip] attempt {attempt}/2 exited with code {rc}; output_exists={out_path.is_file()}\n")
                log.flush()
                with _ACTIVE_GENERATION_LOCK:
                    if _ACTIVE_GENERATION_PROCESS is cp:
                        _ACTIVE_GENERATION_PROCESS = None

                if _GENERATION_CANCEL.is_set():
                    cancelled_attempt = True
                    break
                if rc == 0 and out_path.is_file():
                    success = True
                    break

        if cancelled_attempt or _GENERATION_CANCEL.is_set():
            if out_path.is_file():
                results.append({"index": shot.index, "ok": True, "output_path": str(out_path), "log_path": str(log_path)})
                progress(f"__MINIMAX_SHOT_DONE__|{shot.index}|{out_path}")
            else:
                results.append({"index": shot.index, "ok": False, "cancelled": True, "message": "Cancelled", "log_path": str(log_path)})
            progress("Generation stopped. Completed clips were kept.")
            break

        if not success:
            results.append({
                "index": shot.index,
                "ok": False,
                "message": ("\n".join(last_tail) or f"Exit code {last_rc}") + f"\n\nAutomatic retry with a new random seed also failed.\nLog: {log_path}",
                "log_path": str(log_path),
            })
            progress(f"__MINIMAX_SHOT_FAILED__|{shot.index}|{log_path}")
            continue

        results.append({"index": shot.index, "ok": True, "output_path": str(out_path), "log_path": str(log_path), "seed": int(attempt_seed), "retried": attempt_seed != original_seed})
        progress(f"__MINIMAX_SHOT_DONE__|{shot.index}|{out_path}")
    return results


def _assembly_task(progress, project: MusicProject) -> str:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found.")
    if not project.shots:
        raise RuntimeError("No shot plan exists.")
    missing = [s.index for s in project.shots if not s.output_path or not Path(s.output_path).is_file()]
    if missing:
        raise RuntimeError("Missing generated clips for shots: " + ", ".join(map(str, missing)))

    out_dir = Path(project.output_dir or OUTPUT_ROOT / _safe_stem(project.audio_path)).resolve()
    temp_dir = out_dir / "_assembly"
    temp_dir.mkdir(parents=True, exist_ok=True)
    trimmed: List[Path] = []
    for i, shot in enumerate(project.shots, start=1):
        progress(f"Trimming shot {shot.index} ({i}/{len(project.shots)})...")
        target = temp_dir / f"trim_{shot.index:03d}.mp4"
        # tpad makes tiny source-duration mismatches non-fatal; trim then enforces the edit slot.
        vf = (
            f"tpad=stop_mode=clone:stop_duration=1.0,"
            f"trim=start={shot.trim_in:.6f}:duration={shot.edit_duration:.6f},"
            "setpts=PTS-STARTPTS"
        )
        cmd = [
            ffmpeg, "-y", "-i", shot.output_path, "-an", "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if cp.returncode != 0 or not target.is_file():
            raise RuntimeError(f"Could not trim shot {shot.index}:\n" + (cp.stderr or cp.stdout or ""))
        trimmed.append(target)

    concat_file = temp_dir / "concat.txt"
    concat_file.write_text("\n".join("file '" + str(p).replace("'", "'\\''") + "'" for p in trimmed), encoding="utf-8")
    video_only = temp_dir / "video_only.mp4"
    progress("Concatenating trimmed shots...")
    cp = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(video_only)],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if cp.returncode != 0:
        # Codec-copy concat can fail with odd source headers. Re-encode fallback.
        cp = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video_only)],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if cp.returncode != 0 or not video_only.is_file():
        raise RuntimeError("Could not concatenate clips:\n" + (cp.stderr or cp.stdout or ""))

    final = out_dir / f"{_safe_stem(project.title or project.audio_path)}_minimax_music_video.mp4"
    song_duration = project.analysis.duration or probe_duration(project.audio_path)
    progress("Muxing the original master song...")
    cmd = [
        ffmpeg, "-y", "-i", str(video_only), "-i", project.audio_path,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
    ]
    if song_duration > 0:
        cmd += ["-t", f"{song_duration:.6f}"]
    cmd += [str(final)]
    cp = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if cp.returncode != 0 or not final.is_file():
        raise RuntimeError("Final mux failed:\n" + (cp.stderr or cp.stdout or ""))
    progress(f"Saved final music video: {final}")
    # These are disposable project work folders. Keep raw_clips + final output,
    # but do not accumulate assembly/audio/queue scratch after a successful build.
    for disposable in (temp_dir, out_dir / "audio_chunks", out_dir / "_queue"):
        try:
            if disposable.is_dir():
                shutil.rmtree(disposable, ignore_errors=True)
        except Exception:
            pass
    return str(final)


# ---------------------------------- GUI ------------------------------------


class MiniMaxMusicClipWidget(QWidget):
    """Embeddable MiniMax Music Clip Creator widget."""

    def __init__(self, parent: Optional[QWidget] = None, queue_adapter=None):
        super().__init__(parent)
        self.queue_adapter = queue_adapter
        self.project = MusicProject(output_dir=str(OUTPUT_ROOT))
        self.project_path = ""
        self.worker: Optional[FunctionWorker] = None
        self._autosave_last_text = ""
        self._one_click_active = False
        _cleanup_music_clip_temp_artifacts()
        self._build_ui()
        # Restore the complete working session first. The older small settings file
        # remains only as a fallback for installs that do not have a session save yet.
        if not self._load_autosave():
            self._load_settings()
        self._sync_ui_from_project()
        self._start_autosave()

    # ---- UI construction ----
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

        self.page_project = QWidget(self)
        self.page_refs = QWidget(self)
        self.page_analysis = QWidget(self)
        self.page_director = QWidget(self)
        self.page_generate = QWidget(self)
        self.page_settings = QWidget(self)
        for page, title in (
            (self.page_project, "Project"),
            (self.page_refs, "References"),
            (self.page_analysis, "Song Analysis"),
            (self.page_director, "Director / Shot List"),
            (self.page_generate, "Generate & Review"),
            (self.page_settings, "Settings"),
        ):
            self.tabs.addTab(page, title)

        self._build_project_tab()
        self._build_refs_tab()
        self._build_analysis_tab()
        self._build_director_tab()
        self._build_generate_tab()
        self._build_settings_tab()
        self._build_global_action_bar(outer)

        # QTableWidget headers can fall back to the native Windows style even
        # when FrameVision has already themed the table viewport.  Apply a tiny
        # palette-derived fallback after Qt has polished the widgets so those
        # headers/corner cells track the host theme instead of turning white.
        QTimer.singleShot(0, self._apply_host_theme_fallbacks)

        footer = QHBoxLayout()
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status = QLabel("Ready.", self)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        footer.addWidget(self.progress, 1)
        footer.addWidget(self.status, 2)
        outer.addLayout(footer)

    @staticmethod
    def _css_color(color) -> str:
        """Return a Qt stylesheet-safe ARGB/RGB color string."""
        return color.name()

    def _apply_host_theme_fallbacks(self) -> None:
        """Keep native table chrome aligned with FrameVision's active theme.

        FrameVision styles the table viewport globally, but on some Windows Qt
        styles QHeaderView and the top-left corner button can still be painted by
        the native light theme.  We sample each table's *effective* palette after
        host styling and only style that native chrome.  No fixed MiniMax colors
        are used, so light/dark/custom FrameVision themes remain authoritative.
        """
        tables = [
            getattr(self, "refs_table", None),
            getattr(self, "lyrics_table", None),
            getattr(self, "shot_table", None),
            getattr(self, "review_table", None),
        ]
        for table in tables:
            if table is None:
                continue
            # The viewport is the best source when a host stylesheet targets
            # QAbstractItemView/QTableWidget rather than QApplication's palette.
            pal = table.viewport().palette()
            base = self._css_color(pal.color(QPalette.ColorRole.Base))
            text = self._css_color(pal.color(QPalette.ColorRole.Text))
            border = self._css_color(pal.color(QPalette.ColorRole.Mid))
            alt = self._css_color(pal.color(QPalette.ColorRole.AlternateBase))
            highlight = self._css_color(pal.color(QPalette.ColorRole.Highlight))
            highlighted_text = self._css_color(pal.color(QPalette.ColorRole.HighlightedText))
            qss = f"""
                QHeaderView {{ background: {base}; color: {text}; }}
                QHeaderView::section {{
                    background: {base}; color: {text}; border: 0px;
                    border-right: 1px solid {border};
                    border-bottom: 1px solid {border}; padding: 4px;
                }}
                QHeaderView::section:hover {{ background: {alt}; }}
                QHeaderView::section:checked {{ background: {highlight}; color: {highlighted_text}; }}
                QTableCornerButton::section {{
                    background: {base}; border: 0px;
                    border-right: 1px solid {border};
                    border-bottom: 1px solid {border};
                }}
            """
            table.horizontalHeader().setStyleSheet(qss)
            table.verticalHeader().setStyleSheet(qss)
            # Corner button belongs to QTableWidget rather than either header.
            # Append only this narrow fallback; existing host rules still style
            # the table body, selection, grid, scrollbars, etc.
            table.setStyleSheet(table.styleSheet() + f"""
                QTableWidget QTableCornerButton::section {{
                    background: {base}; border: 0px;
                    border-right: 1px solid {border};
                    border-bottom: 1px solid {border};
                }}
            """)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            # Theme switches can happen while FrameVision is running. Delay until
            # the host style has finished propagating to child viewports.
            QTimer.singleShot(0, self._apply_host_theme_fallbacks)

    def _build_global_action_bar(self, outer: QVBoxLayout) -> None:
        """Keep the primary music-clip actions visible on every tab."""
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_create_video_clip = QPushButton("Create video clip", self)
        self.btn_create_video_clip.setToolTip(
            "One-click workflow: analyze the selected song, use Whisper lyric timing when enabled, "
            "build/direct the complete shot list, add all missing MiniMax clips to the FrameVision queue, "
            "then queue final trim and assembly."
        )
        self.btn_generate_selected = QPushButton("Generate selected shot", self)
        self.btn_generate_all = QPushButton("Generate all missing shots", self)
        self.btn_stop_generation = QPushButton("Stop generation", self)
        self.btn_stop_generation.setEnabled(False)
        self.btn_stop_generation.setToolTip(
            "Stops the current direct MiniMax clip. When FrameVision queue mode is active, "
            "cancel or requeue running jobs from FrameVision's main Queue tab."
        )
        self.btn_assemble = QPushButton("Assemble final music video", self)
        self.btn_open_output = QPushButton("Open output folder", self)

        for button in (
            self.btn_create_video_clip,
            self.btn_generate_selected,
            self.btn_generate_all,
            self.btn_stop_generation,
            self.btn_assemble,
            self.btn_open_output,
        ):
            bar.addWidget(button)
        self.check_framevision_queue = QCheckBox("Use FrameVision queue", self)
        self.check_framevision_queue.setToolTip(
            "When enabled, MiniMax Music Clip shots and final assembly are submitted to FrameVision's shared queue/worker. "
            "When disabled, this helper generates clips directly as before. Failed queued shots retry once with a new random seed before being marked failed."
        )
        self.check_framevision_queue.toggled.connect(self._framevision_queue_toggled)
        bar.addWidget(self.check_framevision_queue)
        bar.addStretch(1)
        outer.addLayout(bar)

        self.btn_create_video_clip.clicked.connect(self.create_video_clip)
        self.btn_generate_selected.clicked.connect(self._generate_selected)
        self.btn_generate_all.clicked.connect(self._generate_all)
        self.btn_stop_generation.clicked.connect(self._stop_generation)
        self.btn_assemble.clicked.connect(self._assemble)
        self.btn_open_output.clicked.connect(self._open_output_folder)

    def _scrollable_tab_body(self, page: QWidget) -> tuple[QVBoxLayout, QWidget, QVBoxLayout]:
        """Create a vertically scrollable tab body plus a fixed bottom area.

        The returned outer layout belongs to the tab page. Add normal content to
        body_layout and add action rows directly to outer after the scroll area so
        buttons stay visible at the bottom while only the content scrolls.
        """
        outer = QVBoxLayout(page)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Do not paint a private dark theme here.  This widget is embedded in
        # FrameVision, so the host application's global stylesheet/palette must
        # remain authoritative (and standalone mode should inherit Qt/app style).
        body = QWidget(scroll)
        body.setMinimumWidth(0)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(2, 2, 2, 2)
        body_layout.setSpacing(6)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        return outer, body, body_layout

    def _build_project_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_project)
        audio_box = QGroupBox("Song", body)
        form = QFormLayout(audio_box)
        row = QHBoxLayout()
        self.edit_audio = QLineEdit(audio_box)
        self.btn_audio = QPushButton("Browse...", audio_box)
        row.addWidget(self.edit_audio, 1); row.addWidget(self.btn_audio)
        form.addRow("Master song:", row)
        self.label_duration = QLabel("Duration: not analyzed", audio_box)
        form.addRow("", self.label_duration)
        row2 = QHBoxLayout()
        self.edit_output = QLineEdit(audio_box)
        self.btn_output = QPushButton("Browse...", audio_box)
        row2.addWidget(self.edit_output, 1); row2.addWidget(self.btn_output)
        form.addRow("Project output:", row2)
        self.edit_title = QLineEdit(audio_box)
        form.addRow("Project title:", self.edit_title)
        lay.addWidget(audio_box)

        brief = QGroupBox("Creative brief", body)
        bf = QFormLayout(brief)
        self.edit_idea = QPlainTextEdit(brief); self.edit_idea.setMaximumHeight(90)
        self.edit_style = QLineEdit(brief)
        self.edit_subjects = QPlainTextEdit(brief); self.edit_subjects.setMaximumHeight(90)
        self.edit_world = QLineEdit(brief)
        self.edit_camera = QLineEdit(brief)
        self.edit_subjects.setToolTip("Optional continuity details for named reference images. Example: Erica is the blue-haired woman in her reference, keeps that outfit, and is always the drummer when the drum kit is present. Leave blank when MiniMax may invent role, outfit or props.")
        self.edit_idea.setToolTip("Write one continuous story, or put alternative story beats on separate lines / separated by semicolons. Alternative beats are distributed across clips instead of being crammed into every clip.")
        self.edit_world.setToolTip("Enter one location, or a list separated by commas, semicolons or new lines. One primary location is assigned per clip and the list rotates before reuse.")
        self.edit_camera.setToolTip("Enter one camera concept, or a list separated by commas, semicolons or new lines. One primary camera concept is assigned per clip and the list rotates before reuse. Connected instructions such as 'whip pan then rack focus' stay together.")
        bf.addRow("Main idea / story:", self.edit_idea)
        bf.addRow("Style / theme:", self.edit_style)
        bf.addRow("Ref image purpose details:", self.edit_subjects)
        bf.addRow("Locations / world:", self.edit_world)
        bf.addRow("Camera choreography:", self.edit_camera)
        lay.addWidget(brief)

        actions = QHBoxLayout()
        self.btn_new = QPushButton("New project", self.page_project)
        self.btn_open = QPushButton("Open project...", self.page_project)
        self.btn_save = QPushButton("Save project", self.page_project)
        self.btn_save_as = QPushButton("Save project as...", self.page_project)
        actions.addWidget(self.btn_new); actions.addWidget(self.btn_open); actions.addWidget(self.btn_save); actions.addWidget(self.btn_save_as); actions.addStretch(1)
        lay.addStretch(1)
        outer.addLayout(actions)

        self.btn_audio.clicked.connect(self._browse_audio)
        self.btn_output.clicked.connect(self._browse_output)
        self.btn_new.clicked.connect(self._new_project)
        self.btn_open.clicked.connect(self._open_project)
        self.btn_save.clicked.connect(self._save_project)
        self.btn_save_as.clicked.connect(lambda: self._save_project(force_as=True))

    def _build_refs_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_refs)
        info = QLabel(
            "Project references are sent directly to MiniMax Ref2VA. Choose what each image is for: Character, Background / Location, Object / Prop, Style / Mood, or Picture / Composition anchor. "
            "MiniMax then receives the correct <Subject N> or <Picture N> relationship automatically. For Character refs, add a short identity/appearance description so multi-reference shots can keep the characters distinct. "
            "Each shot can use a different subset; up to 9 images can be passed to one Ref2VA job.",
            body,
        )
        info.setWordWrap(True); lay.addWidget(info)
        self.check_randomize_ref_characters = QCheckBox("Randomize reference characters per clip", body)
        self.check_randomize_ref_characters.setToolTip(
            "When enabled, each shot automatically picks a random subset of enabled Character references. "
            "The shot uses either 1 or 2 character refs at a time (or fewer when fewer are available). "
            "When disabled, character references are still rotated in a stable round-robin order so later references are not ignored. "
            "Backgrounds, style refs and other non-character references keep their normal behavior. Rebuild prompts or create a new plan to refresh assignments."
        )
        lay.addWidget(self.check_randomize_ref_characters)
        self.refs_table = QTableWidget(0, 6, body)
        # Put the useful editable fields first. Long filenames/paths are supporting
        # metadata and must never consume the reference tab at the expense of role
        # and identity/purpose descriptions.
        self.refs_table.setHorizontalHeaderLabels(["Use", "Preview", "Reference role", "Character / image description", "Name", "Path"])
        header = self.refs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.resizeSection(1, 122)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.resizeSection(2, 190)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.resizeSection(4, 180)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        header.resizeSection(5, 150)
        self.refs_table.setWordWrap(True)
        self.refs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.refs_table.setMinimumHeight(320)
        lay.addWidget(self.refs_table, 1)
        row = QHBoxLayout()
        self.btn_add_ref = QPushButton("Add reference image...", self.page_refs)
        self.btn_remove_ref = QPushButton("Remove selected", self.page_refs)
        row.addWidget(self.btn_add_ref); row.addWidget(self.btn_remove_ref); row.addStretch(1)
        outer.addLayout(row)
        self.btn_add_ref.clicked.connect(self._add_reference)
        self.btn_remove_ref.clicked.connect(self._remove_reference)

    def _build_analysis_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_analysis)
        box = QGroupBox("Analysis", body)
        form = QFormLayout(box)
        self.spin_sensitivity = QSpinBox(box); self.spin_sensitivity.setRange(0, 20); self.spin_sensitivity.setValue(10)
        self.spin_sensitivity.setToolTip("Beat detector sensitivity. 0 = strict/fewer peaks, 10 = normal, 20 = loose/more peaks.")
        form.addRow("Beat sensitivity:", self.spin_sensitivity)
        self.check_whisper_timing = QCheckBox("Use Whisper lyrics for shot timing", box)
        self.check_whisper_timing.setChecked(True)
        self.check_whisper_timing.setToolTip(
            "On: Whisper phrase endings are the preferred shot boundaries when they can be covered by a valid MiniMax H3 frame count. "
            "Beat/energy analysis becomes supporting information. Off: use the normal beat/energy duration planner. If no lyrics are loaded, normal planning is used automatically."
        )
        form.addRow("Lyric-aware planning:", self.check_whisper_timing)
        self.check_visible_lyric_subtitles = QCheckBox("Generate visible lyric subtitles", box)
        self.check_visible_lyric_subtitles.setChecked(False)
        self.check_visible_lyric_subtitles.setToolTip(
            "Off (default): Whisper lyrics are audio/performance timing guidance only and must not appear as subtitles, captions, karaoke text or lyric overlays. "
            "On: MiniMax may render the current lyric phrase as visible subtitles. Text explicitly requested as a physical part of the scene remains independent of this option."
        )
        form.addRow("Visible lyrics:", self.check_visible_lyric_subtitles)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze track", self.page_analysis)
        self.btn_whisper = QPushButton("Transcribe lyrics with Whisper.cpp", self.page_analysis)
        self.btn_download_whisper = QPushButton("Download Whisper.cpp Small", self.page_analysis)
        self.btn_download_whisper.setToolTip(
            "Downloads the standalone multilingual Whisper.cpp Small model (~488 MiB) plus the small Windows runtime into presets/bin/whisper. "
            "It is independent of FrameVision's old Whisper environment and can be reused when this creator is added to the standalone MiniMax app."
        )
        self.btn_clear_lyrics = QPushButton("Treat as instrumental", self.page_analysis)
        row.addWidget(self.btn_analyze); row.addWidget(self.btn_whisper); row.addWidget(self.btn_download_whisper); row.addWidget(self.btn_clear_lyrics); row.addStretch(1)
        self.analysis_summary = QLabel("No analysis yet.", body); self.analysis_summary.setWordWrap(True); lay.addWidget(self.analysis_summary)
        self.lyrics_table = QTableWidget(0, 3, body)
        self.lyrics_table.setHorizontalHeaderLabels(["Start", "End", "Whisper phrase"])
        self.lyrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.lyrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.lyrics_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.lyrics_table.setMinimumHeight(360)
        lay.addWidget(self.lyrics_table, 1)
        outer.addLayout(row)
        self.btn_analyze.clicked.connect(self._start_analysis)
        self.btn_whisper.clicked.connect(self._start_whisper)
        self.btn_download_whisper.clicked.connect(self._download_whisper_cpp)
        self.btn_clear_lyrics.clicked.connect(self._clear_lyrics)
        self._update_whisper_buttons()

    def _build_director_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_director)
        top = QHBoxLayout()
        self.btn_plan = QPushButton("Create / rebuild shot plan", self.page_director)
        self.btn_refresh_prompts = QPushButton("Rebuild H3 Ref2VA prompts", self.page_director)
        top.addWidget(self.btn_plan); top.addWidget(self.btn_refresh_prompts); top.addStretch(1)
        note = QLabel(
            "Prompts are built in MiniMax-H3 full-reference format: subject definitions, reference retention, shot-by-shot description and explicit audio roles. "
            "Lyric/section boundaries that fall inside a generated clip become timestamped H3 shots while the supplied audio continues across the cut.",
            body,
        ); note.setWordWrap(True); lay.addWidget(note)
        self.shot_table = QTableWidget(0, 10, body)
        self.shot_table.setHorizontalHeaderLabels([
            "#", "Song range", "Edit", "MiniMax frames", "Gen", "Section", "Lyrics", "Internal cuts", "References", "Prompt"
        ])
        for col in (0, 2, 3, 4): self.shot_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        for col in (1, 5, 6, 7, 8, 9): self.shot_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        self.shot_table.setWordWrap(True)
        self.shot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.shot_table.verticalHeader().setDefaultSectionSize(72)
        self.shot_table.verticalHeader().setMinimumSectionSize(56)
        self.shot_table.setMinimumHeight(420)
        lay.addWidget(self.shot_table, 1)

        prompt_box = QGroupBox("Selected shot H3 prompt", body)
        prompt_lay = QVBoxLayout(prompt_box)
        self.shot_prompt_editor = QPlainTextEdit(prompt_box)
        self.shot_prompt_editor.setPlaceholderText("Select a shot above to view or edit its complete MiniMax-H3 Ref2VA prompt.")
        self.shot_prompt_editor.setMinimumHeight(260)
        self.shot_prompt_editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        prompt_lay.addWidget(self.shot_prompt_editor)
        lay.addWidget(prompt_box)

        outer.addLayout(top)
        self.btn_plan.clicked.connect(self._create_plan)
        self.btn_refresh_prompts.clicked.connect(self._rebuild_prompts)
        self.shot_table.itemChanged.connect(self._shot_table_item_changed)
        self.shot_table.itemSelectionChanged.connect(self._show_selected_shot_prompt)
        self.shot_prompt_editor.textChanged.connect(self._selected_shot_prompt_changed)

    def _build_generate_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_generate)
        self.label_job_seed = QLabel("Job seed: auto (first generation locks one random seed for the whole job). Edit a shot's Seed value below to override it for retries.", body)
        self.label_job_seed.setWordWrap(True)
        lay.addWidget(self.label_job_seed)
        self.review_table = QTableWidget(0, 6, body)
        self.review_table.setHorizontalHeaderLabels(["#", "Status", "Frames", "Edit range", "Output", "Seed"])
        self.review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.review_table.setMinimumHeight(320)
        lay.addWidget(self.review_table, 1)

        preview_box = QGroupBox("Selected clip preview", body)
        preview_lay = QVBoxLayout(preview_box)
        self.review_preview = QLabel("Select a generated shot to preview it.", preview_box)
        self.review_preview.setAlignment(Qt.AlignCenter)
        self.review_preview.setMinimumHeight(180)
        self.review_preview.setMaximumHeight(280)
        # Let the host theme draw this panel instead of forcing MiniMax-specific
        # dark colors that can clash with FrameVision themes.
        self.review_preview.setFrameShape(QFrame.Shape.StyledPanel)
        preview_lay.addWidget(self.review_preview, 1)
        preview_actions = QHBoxLayout()
        self.btn_play_clip = QPushButton("Play selected clip", preview_box)
        self.btn_open_clip_folder = QPushButton("Open clip folder", preview_box)
        preview_actions.addWidget(self.btn_play_clip); preview_actions.addWidget(self.btn_open_clip_folder); preview_actions.addStretch(1)
        preview_lay.addLayout(preview_actions)
        lay.addWidget(preview_box)
        self.btn_play_clip.clicked.connect(self._play_selected_clip)
        self.btn_open_clip_folder.clicked.connect(self._open_selected_clip_folder)
        self.review_table.itemSelectionChanged.connect(self._show_selected_clip_preview)
        self.review_table.itemChanged.connect(self._review_table_item_changed)
        self.review_table.cellDoubleClicked.connect(lambda _r, _c: self._play_selected_clip())

    def _build_settings_tab(self) -> None:
        outer, body, lay = self._scrollable_tab_body(self.page_settings)
        gen = QGroupBox("MiniMax generation", body)
        form = QFormLayout(gen)
        self.combo_resolution = QComboBox(gen); self.combo_resolution.addItems(list(RESOLUTION_PRESETS.keys())); self.combo_resolution.setCurrentText("832 × 480")
        self.combo_aspect = QComboBox(gen); self.combo_aspect.addItems(["16:9", "9:16", "1:1"])
        form.addRow("Resolution:", self.combo_resolution); form.addRow("Aspect ratio:", self.combo_aspect)

        max_row = QHBoxLayout()
        self.slider_frames = QSlider(Qt.Horizontal, gen); self.slider_frames.setRange(0, len(MUSIC_FRAME_GRID) - 1); self.slider_frames.setValue(len(MUSIC_FRAME_GRID) - 1)
        self.label_frames = QLabel(gen)
        max_row.addWidget(self.slider_frames, 1); max_row.addWidget(self.label_frames)
        form.addRow("Maximum generated shot length:", max_row)
        self.slider_frames.setToolTip(
            "Sets the maximum MiniMax Ref2VA generation length used by the planner. The director may use shorter valid frame counts. "
            "Generated clips can include extra material at the beginning/end and are trimmed during final assembly."
        )
        self.spin_head = QDoubleSpinBox(gen); self.spin_head.setRange(0.0, 2.0); self.spin_head.setSingleStep(0.05); self.spin_head.setValue(0.35); self.spin_head.setSuffix(" s")
        self.spin_tail = QDoubleSpinBox(gen); self.spin_tail.setRange(0.0, 2.0); self.spin_tail.setSingleStep(0.05); self.spin_tail.setValue(0.45); self.spin_tail.setSuffix(" s")
        self.spin_snap = QDoubleSpinBox(gen); self.spin_snap.setRange(0.0, 3.0); self.spin_snap.setSingleStep(0.05); self.spin_snap.setValue(1.25); self.spin_snap.setSuffix(" s")
        form.addRow("Extra context before edit:", self.spin_head)
        form.addRow("Extra context after edit:", self.spin_tail)
        form.addRow("Phrase-boundary snap tolerance:", self.spin_snap)
        self.spin_steps = QSpinBox(gen); self.spin_steps.setRange(1, 100); self.spin_steps.setValue(15)
        self.spin_cfg = QDoubleSpinBox(gen); self.spin_cfg.setRange(0.0, 20.0); self.spin_cfg.setValue(1.0); self.spin_cfg.setDecimals(2)
        self.spin_shift = QDoubleSpinBox(gen); self.spin_shift.setRange(0.0, 30.0); self.spin_shift.setValue(12.0); self.spin_shift.setDecimals(2)
        self.spin_audio_shift = QDoubleSpinBox(gen); self.spin_audio_shift.setRange(0.0, 20.0); self.spin_audio_shift.setValue(3.0); self.spin_audio_shift.setDecimals(2)
        form.addRow("Steps:", self.spin_steps); form.addRow("CFG:", self.spin_cfg); form.addRow("Shift:", self.spin_shift); form.addRow("Audio shift:", self.spin_audio_shift)
        self.combo_ref_size = QComboBox(gen); self.combo_ref_size.addItems(["match", "max"]); form.addRow("Reference image size:", self.combo_ref_size)
        lora_row = QHBoxLayout()
        self.edit_turbo_lora = QLineEdit(gen)
        self.edit_turbo_lora.setPlaceholderText("Optional speed / Turbo LoRA (.safetensors)")
        self.btn_turbo_lora = QPushButton("Browse...", gen)
        self.spin_turbo_lora = QDoubleSpinBox(gen); self.spin_turbo_lora.setRange(-4.0, 4.0); self.spin_turbo_lora.setSingleStep(0.05); self.spin_turbo_lora.setDecimals(2); self.spin_turbo_lora.setValue(1.0)
        self.spin_turbo_lora.setToolTip("Strength for the selected MiniMax LoRA. 1.0 = normal strength.")
        lora_row.addWidget(self.edit_turbo_lora, 1); lora_row.addWidget(self.btn_turbo_lora); lora_row.addWidget(QLabel("Strength:", gen)); lora_row.addWidget(self.spin_turbo_lora)
        form.addRow("Turbo / speed LoRA:", lora_row)
        self.btn_turbo_lora.clicked.connect(self._browse_turbo_lora)
        self.check_hybrid_model = QCheckBox("Use hybrid model", gen)
        self.check_hybrid_model.setToolTip("Use one hybrid MiniMax H3 diffusion checkpoint for these Ref2VA music clips instead of the normal Ref2VA checkpoint. This choice is remembered after restart.")
        hybrid_row = QHBoxLayout()
        self.edit_hybrid_model = QLineEdit(gen); self.edit_hybrid_model.setPlaceholderText("Hybrid MiniMax H3 checkpoint (.safetensors)")
        self.btn_hybrid_model = QPushButton("Browse...", gen)
        hybrid_row.addWidget(self.edit_hybrid_model, 1); hybrid_row.addWidget(self.btn_hybrid_model)
        form.addRow(self.check_hybrid_model)
        form.addRow("Hybrid checkpoint:", hybrid_row)
        self.btn_hybrid_model.clicked.connect(self._browse_hybrid_model)
        self.check_hybrid_model.toggled.connect(self._sync_hybrid_model_controls)
        self._sync_hybrid_model_controls(False)
        self.check_vram_manager = QCheckBox("Enable VRAM Manager protection", gen); self.check_vram_manager.setChecked(True)
        self.check_vram_manager.setToolTip("Master switch. Off = never use VRAM Manager. On = use the setting below to choose automatic bypass or always-on protection.")
        self.check_vram_auto_bypass = QCheckBox("Automatic bypass when job fits", gen); self.check_vram_auto_bypass.setChecked(True)
        self.check_vram_auto_bypass.setToolTip("On = MiniMax decides per stage/job whether native loading is safe. Off = VRAM Manager stays active for every job.")
        self.check_sage = QCheckBox("SageAttention", gen)
        self.check_spectrum = QCheckBox("Spectrum", gen)
        flags = QHBoxLayout(); flags.addWidget(self.check_vram_manager); flags.addWidget(self.check_vram_auto_bypass); flags.addWidget(self.check_sage); flags.addWidget(self.check_spectrum); flags.addStretch(1)
        form.addRow("Acceleration / VRAM:", flags)
        lay.addWidget(gen)
        explanation = QLabel(
            "Timing rule: the song timeline is authoritative. MiniMax valid frame counts determine how much source footage is generated, "
            "then FFmpeg trims each source clip to its exact edit slot. Small timing mismatches are repaired during assembly instead of aborting.",
            body,
        ); explanation.setWordWrap(True); lay.addWidget(explanation); lay.addStretch(1)
        self.slider_frames.valueChanged.connect(self._update_frame_label)
        self._update_frame_label()

    # ---- state conversion ----
    def _commit_selected_shot_prompt(self) -> None:
        """Commit the visible Director editor text into the selected shot immediately."""
        if not hasattr(self, "shot_prompt_editor") or not hasattr(self, "shot_table"):
            return
        shot = self._selected_director_shot()
        if shot is None:
            return
        text = self.shot_prompt_editor.toPlainText()
        if shot.prompt != text:
            shot.prompt = text
        row = next((r for r, s in enumerate(self.project.shots) if s.index == shot.index), -1)
        if row >= 0:
            item = self.shot_table.item(row, 9)
            if item is not None and item.text() != text:
                self.shot_table.blockSignals(True)
                item.setText(text)
                self.shot_table.blockSignals(False)

    def _pull_ui(self) -> None:
        # The Director prompt editor is authoritative. Commit it before ANY save,
        # queue, generation, restart autosave, or project-state snapshot.
        self._commit_selected_shot_prompt()
        self.project.audio_path = self.edit_audio.text().strip()
        self.project.output_dir = self.edit_output.text().strip() or str(OUTPUT_ROOT)
        self.project.title = self.edit_title.text().strip()
        self.project.main_idea = self.edit_idea.toPlainText().strip()
        self.project.style_theme = self.edit_style.text().strip()
        self.project.characters_subjects = self.edit_subjects.toPlainText().strip()
        self.project.locations_world = self.edit_world.text().strip()
        self.project.camera_choreography = self.edit_camera.text().strip()
        self.project.beat_sensitivity = self.spin_sensitivity.value()
        self.project.whisper_timing_enabled = self.check_whisper_timing.isChecked()
        self.project.visible_lyric_subtitles = self.check_visible_lyric_subtitles.isChecked()
        self.project.resolution = self.combo_resolution.currentText()
        self.project.aspect = self.combo_aspect.currentText()
        self.project.max_frames = MUSIC_FRAME_GRID[self.slider_frames.value()]
        self.project.head_padding = self.spin_head.value()
        self.project.tail_padding = self.spin_tail.value()
        self.project.phrase_snap_tolerance = self.spin_snap.value()
        self.project.steps = self.spin_steps.value()
        self.project.cfg = self.spin_cfg.value()
        self.project.shift = self.spin_shift.value()
        self.project.audio_shift = self.spin_audio_shift.value()
        self.project.ref_image_size = self.combo_ref_size.currentText()
        self.project.turbo_lora_path = self.edit_turbo_lora.text().strip()
        self.project.turbo_lora_strength = self.spin_turbo_lora.value()
        self.project.use_hybrid_model = self.check_hybrid_model.isChecked()
        self.project.hybrid_model_path = self.edit_hybrid_model.text().strip()
        self.project.vram_manager_enabled = self.check_vram_manager.isChecked()
        self.project.vram_auto_bypass = self.check_vram_auto_bypass.isChecked()
        self.project.sage_attention = self.check_sage.isChecked()
        self.project.spectrum = self.check_spectrum.isChecked()
        self.project.randomize_reference_characters = bool(self.check_randomize_ref_characters.isChecked())
        self.project.use_framevision_queue = bool(self.check_framevision_queue.isChecked())
        self.project.references = self._refs_from_table()

    def _sync_ui_from_project(self) -> None:
        p = self.project
        self.edit_audio.setText(p.audio_path); self.edit_output.setText(p.output_dir or str(OUTPUT_ROOT)); self.edit_title.setText(p.title)
        self.edit_idea.setPlainText(p.main_idea); self.edit_style.setText(p.style_theme); self.edit_subjects.setPlainText(p.characters_subjects); self.edit_world.setText(p.locations_world); self.edit_camera.setText(p.camera_choreography)
        self.spin_sensitivity.setValue(p.beat_sensitivity)
        self.check_whisper_timing.setChecked(bool(getattr(p, "whisper_timing_enabled", True)))
        self.check_visible_lyric_subtitles.setChecked(bool(getattr(p, "visible_lyric_subtitles", False)))
        self.combo_resolution.setCurrentText(p.resolution if p.resolution in RESOLUTION_PRESETS else "832 × 480")
        self.combo_aspect.setCurrentText(p.aspect if p.aspect in ("16:9", "9:16", "1:1") else "16:9")
        nearest_idx = min(range(len(MUSIC_FRAME_GRID)), key=lambda i: abs(MUSIC_FRAME_GRID[i] - int(p.max_frames or MUSIC_FRAME_DEFAULT_MAX)))
        self.slider_frames.setValue(nearest_idx)
        self.spin_head.setValue(p.head_padding); self.spin_tail.setValue(p.tail_padding); self.spin_snap.setValue(p.phrase_snap_tolerance)
        self.spin_steps.setValue(p.steps); self.spin_cfg.setValue(p.cfg); self.spin_shift.setValue(p.shift); self.spin_audio_shift.setValue(p.audio_shift)
        self.combo_ref_size.setCurrentText(p.ref_image_size if p.ref_image_size in ("match", "max") else "match")
        self.edit_turbo_lora.setText(p.turbo_lora_path or "")
        self.spin_turbo_lora.setValue(float(p.turbo_lora_strength or 1.0))
        self.check_hybrid_model.setChecked(bool(getattr(p, "use_hybrid_model", False)))
        self.edit_hybrid_model.setText(str(getattr(p, "hybrid_model_path", "") or ""))
        self.check_vram_manager.setChecked(bool(p.vram_manager_enabled)); self.check_vram_auto_bypass.setChecked(bool(p.vram_auto_bypass)); self.check_sage.setChecked(p.sage_attention); self.check_spectrum.setChecked(p.spectrum)
        self.check_randomize_ref_characters.setChecked(bool(getattr(p, "randomize_reference_characters", False)))
        self.check_framevision_queue.setChecked(bool(getattr(p, "use_framevision_queue", False)))
        self._populate_refs(); self._populate_analysis(); self._populate_shots(); self._populate_review(); self._update_frame_label()

    def _project_dict(self) -> Dict[str, Any]:
        self._pull_ui()
        return asdict(self.project)

    @staticmethod
    def _project_from_dict(data: Dict[str, Any]) -> MusicProject:
        p = MusicProject()
        simple_fields = [k for k in p.__dataclass_fields__ if k not in ("references", "lyrics", "analysis", "shots")]
        for key in simple_fields:
            if key in data:
                setattr(p, key, data[key])
        p.references = [ReferenceAsset(**x) for x in data.get("references", []) if isinstance(x, dict)]
        p.lyrics = [LyricSegment(**x) for x in data.get("lyrics", []) if isinstance(x, dict)]
        a = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
        p.analysis = AnalysisResult(
            duration=float(a.get("duration", 0.0) or 0.0),
            beats=[Beat(**x) for x in a.get("beats", []) if isinstance(x, dict)],
            sections=[Section(**x) for x in a.get("sections", []) if isinstance(x, dict)],
        )
        p.shots = [MusicShot(**x) for x in data.get("shots", []) if isinstance(x, dict)]
        return p

    def _ensure_project_output_folder(self, reset_generated_state: bool = True) -> Path:
        """Bind output to the current title+track pair so unrelated projects never share clips."""
        title = self.edit_title.text().strip() if hasattr(self, "edit_title") else self.project.title
        audio = self.edit_audio.text().strip() if hasattr(self, "edit_audio") else self.project.audio_path
        identity = _music_project_identity(title, audio)
        if not identity:
            return Path(self.project.output_dir or OUTPUT_ROOT)

        if self.project.output_identity == identity and self.project.output_dir:
            return Path(self.project.output_dir)

        label = _safe_stem(title or (Path(audio).stem if audio else "music_project")) or "music_project"
        track_label = _safe_stem(Path(audio).stem) if audio else ""
        if track_label and track_label.casefold() != label.casefold():
            label = f"{label}__{track_label}"

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        base = OUTPUT_ROOT / label
        candidate = base
        serial = 2
        marker_name = ".minimax_music_project.json"
        while candidate.exists():
            marker = candidate / marker_name
            try:
                if marker.is_file():
                    data = json.loads(marker.read_text(encoding="utf-8"))
                    if str(data.get("identity") or "") == identity:
                        break
                elif not any(candidate.iterdir()):
                    break
            except Exception:
                pass
            candidate = Path(str(base) + f"_{serial}")
            serial += 1

        old_identity = self.project.output_identity
        candidate.mkdir(parents=True, exist_ok=True)
        try:
            (candidate / marker_name).write_text(json.dumps({
                "identity": identity, "title": title, "audio_path": audio
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        self.project.output_identity = identity
        self.project.output_dir = str(candidate)
        if hasattr(self, "edit_output"):
            self.edit_output.setText(str(candidate))

        if reset_generated_state and old_identity and old_identity != identity:
            for shot in self.project.shots:
                shot.output_path = ""
                if shot.status == "Generated":
                    shot.status = "Planned"
        return candidate

    # ---- project file operations ----
    def _new_project(self) -> None:
        # Keep user generation preferences when starting a new project. Project-specific
        # material (song, creative brief, references, analysis and shots) is cleared.
        self._pull_ui()
        old = self.project
        self.project = MusicProject(output_dir=str(OUTPUT_ROOT))
        for name in (
            "resolution", "aspect", "max_frames", "head_padding", "tail_padding",
            "phrase_snap_tolerance", "steps", "cfg", "shift", "audio_shift",
            "ref_image_size", "turbo_lora_path", "turbo_lora_strength",
            "use_hybrid_model", "hybrid_model_path",
            "vram_manager_enabled", "vram_auto_bypass", "vram_residency_engine",
            "vram_runtime_free_gb", "vram_text_headroom_gb", "vram_diffusion_headroom_gb",
            "vram_offload_chunk_mb", "vram_max_resident_weights_gb", "vram_block_check_interval",
            "vram_async_streams", "vram_video_vae_reserve_gb", "vram_audio_vae_reserve_gb",
            "vram_residency_fill", "vram_residency_target_free_gb", "vram_residency_warmup_blocks",
            "vram_residency_refill_interval", "sage_attention", "spectrum", "beat_sensitivity", "whisper_timing_enabled", "visible_lyric_subtitles",
            "randomize_reference_characters",
        ):
            setattr(self.project, name, getattr(old, name))
        self.project_path = ""
        _cleanup_music_clip_temp_artifacts()
        self._sync_ui_from_project()
        self.status.setText("New project.")
        self._save_settings()
        self._write_autosave(force=True)

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select master song", "", "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All files (*.*)")
        if path:
            self.edit_audio.setText(path)
            if not self.edit_title.text().strip(): self.edit_title.setText(Path(path).stem)
            # A newly selected track invalidates the old project's output binding.
            # Do not create the replacement folder yet: the user may still rename the
            # project before Analyze/Create plan. That avoids leaving empty test folders.
            self.project.audio_path = path
            self.project.title = self.edit_title.text().strip()
            self.project.output_identity = ""
            self.project.output_dir = str(OUTPUT_ROOT)
            self.edit_output.setText(str(OUTPUT_ROOT))
            for shot in self.project.shots:
                shot.output_path = ""
                if shot.status == "Generated":
                    shot.status = "Planned"
            dur = probe_duration(path)
            self.label_duration.setText(f"Duration: {_fmt_time(dur)} ({dur:.2f} s)" if dur else "Duration: unknown")

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select project output folder", self.edit_output.text().strip() or str(OUTPUT_ROOT))
        if path: self.edit_output.setText(path)

    def _save_project(self, force_as: bool = False) -> None:
        if force_as or not self.project_path:
            default_dir = Path(self.edit_output.text().strip() or OUTPUT_ROOT); default_dir.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(self, "Save MiniMax music project", str(default_dir / "minimax_music_project.json"), "JSON (*.json)")
            if not path: return
            self.project_path = path
        Path(self.project_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.project_path).write_text(json.dumps(self._project_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.setText(f"Saved project: {self.project_path}")
        self._save_settings()
        self._write_autosave(force=True)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open MiniMax music project", "", "JSON (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8")); self.project = self._project_from_dict(data); self.project_path = path; self._sync_ui_from_project(); self.status.setText(f"Opened: {path}")
            self._write_autosave(force=True)
        except Exception as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))

    # ---- refs ----
    def _add_reference(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add MiniMax reference images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if not paths: return
        current = self._refs_from_table()
        for path in paths:
            if len(current) >= 9: break
            if any(os.path.normcase(r.path) == os.path.normcase(path) for r in current): continue
            current.append(ReferenceAsset(name=Path(path).stem, kind="Character", path=path))
        self.project.references = current; self._populate_refs()

    def _remove_reference(self) -> None:
        rows = sorted({i.row() for i in self.refs_table.selectedIndexes()}, reverse=True)
        for row in rows: self.refs_table.removeRow(row)
        self.project.references = self._refs_from_table()

    def _populate_refs(self) -> None:
        self.refs_table.blockSignals(True); self.refs_table.setRowCount(0)
        for ref in self.project.references[:9]:
            row = self.refs_table.rowCount(); self.refs_table.insertRow(row); self.refs_table.setRowHeight(row, 92)
            use = QTableWidgetItem(""); use.setFlags(use.flags() | Qt.ItemIsUserCheckable); use.setCheckState(Qt.Checked if ref.enabled else Qt.Unchecked); self.refs_table.setItem(row, 0, use)
            preview = QLabel(self.refs_table); preview.setAlignment(Qt.AlignCenter); preview.setMinimumSize(112, 78); preview.setMaximumSize(112, 78)
            pix = QPixmap(ref.path) if ref.path and Path(ref.path).is_file() else QPixmap()
            if not pix.isNull():
                preview.setPixmap(pix.scaled(108, 74, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                preview.setToolTip(ref.path)
            else:
                preview.setText("No preview")
            self.refs_table.setCellWidget(row, 1, preview)
            role_combo = QComboBox(self.refs_table)
            role_combo.addItems(list(REFERENCE_TYPES))
            role_combo.setCurrentText(_normalise_reference_kind(ref.kind))
            role_combo.setToolTip(
                "Character = reusable person/identity. Background / Location = recurring environment. Object / Prop = reusable item. "
                "Style / Mood = visual treatment. Picture / Composition anchor = use the image itself as framing/keyframe/shot-planning guidance."
            )
            self.refs_table.setCellWidget(row, 2, role_combo)
            desc_item = QTableWidgetItem(ref.description)
            desc_item.setToolTip(ref.description or "For characters: describe the individual appearance/identity traits that should stay distinct from other refs.")
            self.refs_table.setItem(row, 3, desc_item)
            name_item = QTableWidgetItem(ref.name); name_item.setToolTip(ref.name)
            path_item = QTableWidgetItem(ref.path); path_item.setToolTip(ref.path)
            self.refs_table.setItem(row, 4, name_item); self.refs_table.setItem(row, 5, path_item)
        self.refs_table.blockSignals(False)

    def _refs_from_table(self) -> List[ReferenceAsset]:
        refs: List[ReferenceAsset] = []
        for row in range(self.refs_table.rowCount()):
            def txt(col): return self.refs_table.item(row, col).text().strip() if self.refs_table.item(row, col) else ""
            role_widget = self.refs_table.cellWidget(row, 2)
            kind = _normalise_reference_kind(role_widget.currentText() if isinstance(role_widget, QComboBox) else txt(2))
            refs.append(ReferenceAsset(name=txt(4) or Path(txt(5)).stem, kind=kind, path=txt(5), description=txt(3), enabled=bool(self.refs_table.item(row,0) and self.refs_table.item(row,0).checkState() == Qt.Checked)))
        return refs[:9]

    # ---- one-click video clip workflow ----
    def create_video_clip(self) -> None:
        """Run the user-facing Music Clip workflow and enqueue the complete video.

        The standalone host routes its fixed bottom Generate button here while the
        Music Clip Creator tab is selected. Analysis/transcription/planning happen
        first; normal MiniMax queue jobs are then created for the physical shots,
        followed by one final trim/assembly queue job.
        """
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Music Clip Creator busy", "Wait for the current analysis or Whisper task to finish first.")
            return

        audio = self.edit_audio.text().strip()
        if not audio or not Path(audio).is_file():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select master song for the video clip", "",
                "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All files (*.*)"
            )
            if not path:
                return
            self.edit_audio.setText(path)
            if not self.edit_title.text().strip():
                self.edit_title.setText(Path(path).stem)
            audio = path

        self._pull_ui()
        self.project.audio_path = audio
        self._ensure_project_output_folder(reset_generated_state=True)
        self._one_click_active = True
        # Show the stage that is actually running; the user can still inspect/change tabs.
        self.tabs.setCurrentIndex(2)
        self._set_busy("Video clip: analyzing track structure...")
        self._run_worker(_analysis_task, self._one_click_analysis_done, audio, int(self.project.beat_sensitivity))

    def _one_click_start_worker_when_idle(self, fn, success, *args) -> None:
        # FunctionWorker emits its success signal immediately before QThread emits
        # finished. Wait for that tiny handoff window so chained one-click stages do
        # not trip the normal "A job is already running" protection.
        if self.worker is not None and self.worker.isRunning():
            QTimer.singleShot(60, lambda: self._one_click_start_worker_when_idle(fn, success, *args))
            return
        self._run_worker(fn, success, *args)

    def _one_click_analysis_done(self, result: AnalysisResult) -> None:
        self.project.analysis = result
        self._populate_analysis()
        self._write_autosave(force=True)

        if not bool(self.check_whisper_timing.isChecked()):
            self._one_click_build_plan_and_queue()
            return

        if _whisper_cpp_ready():
            self._set_busy("Video clip: transcribing lyrics with Whisper.cpp...")
            self._one_click_start_worker_when_idle(_whisper_task, self._one_click_whisper_done, self.project.audio_path)
            return

        answer = QMessageBox.question(
            self,
            "Whisper.cpp not installed",
            "Lyric-aware timing is enabled, but portable Whisper.cpp Small is not installed yet.\n\n"
            "Download it now to presets/bin/whisper?\n\n"
            "Choose No to continue this video with beat/energy timing only.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._set_busy("Video clip: downloading portable Whisper.cpp Small...")
            self._one_click_start_worker_when_idle(_download_whisper_cpp_task, self._one_click_whisper_downloaded)
        else:
            # Do not accidentally reuse lyrics from an older song/session.
            self.project.lyrics = []
            self._populate_analysis()
            self._one_click_build_plan_and_queue()

    def _one_click_whisper_downloaded(self, _path: str) -> None:
        self._update_whisper_buttons()
        self._set_busy("Video clip: transcribing lyrics with Whisper.cpp...")
        self._one_click_start_worker_when_idle(_whisper_task, self._one_click_whisper_done, self.project.audio_path)

    def _one_click_whisper_done(self, lyrics: List[LyricSegment]) -> None:
        self.project.lyrics = list(lyrics or [])
        self._populate_analysis()
        self._write_autosave(force=True)
        self._one_click_build_plan_and_queue()

    def _one_click_build_plan_and_queue(self) -> None:
        self.tabs.setCurrentIndex(3)
        self._pull_ui()
        self._ensure_project_output_folder(reset_generated_state=True)
        if self.project.analysis.duration <= 0:
            self.project.analysis.duration = probe_duration(self.project.audio_path)
        try:
            self.project.shots = build_shot_plan(self.project)
            self.project.reference_random_seed = self._next_job_seed() if self.project.randomize_reference_characters else -1
            for shot in self.project.shots:
                shot.reference_names = auto_assign_references(self.project, shot)
                shot.prompt = build_default_prompt(self.project, shot)
            self._populate_shots()
            self._populate_review()
            self._write_autosave(force=True)
        except Exception as exc:
            self._one_click_active = False
            self._set_ready("Video clip planning failed.")
            QMessageBox.critical(self, "Video clip planning failed", str(exc))
            return

        if not self.project.shots:
            self._one_click_active = False
            self._set_ready("No shots were created.")
            QMessageBox.warning(self, "No shots", "The Director did not create any shots for this track.")
            return

        self.tabs.setCurrentIndex(4)
        self._sync_existing_generated_outputs()
        missing = [
            shot.index for shot in self.project.shots
            if not shot.output_path or not Path(shot.output_path).is_file() or shot.status == "Failed"
        ]

        if self._queue_mode_active():
            if missing:
                self._queue_shots(missing)
            # Assembly is deliberately enqueued last. The standalone queue therefore
            # reaches it only after all physical shot jobs ahead of it have finished.
            self._queue_assembly()
            self._one_click_active = False
            self._set_ready(
                f"Video clip queued: {len(missing)} shot{'s' if len(missing) != 1 else ''} + final trim/assembly."
                if missing else "All shot files already exist; final trim/assembly queued."
            )
            return

        # Direct/standalone helper fallback: generate first; assembly remains the
        # explicit next action because there is no host queue dependency mechanism.
        self._one_click_active = False
        if missing:
            self._generate_indices(missing)
        else:
            self._assemble()

    # ---- analysis ----
    def _require_audio(self) -> Optional[str]:
        path = self.edit_audio.text().strip()
        if not path or not Path(path).is_file(): QMessageBox.warning(self, "Song missing", "Select a readable master song first."); return None
        return path

    def _set_busy(self, text: str) -> None:
        self.progress.setRange(0, 0); self.status.setText(text)

    def _set_ready(self, text: str = "Ready.") -> None:
        self.progress.setRange(0, 1); self.progress.setValue(0); self.status.setText(text)

    def _run_worker(self, fn, success, *args) -> None:
        if self.worker is not None and self.worker.isRunning(): QMessageBox.information(self, "Busy", "A job is already running."); return
        self.worker = FunctionWorker(fn, *args); self.worker.progress.connect(self._worker_progress); self.worker.succeeded.connect(success); self.worker.failed.connect(self._worker_failed); self.worker.finished.connect(lambda: self.progress.setRange(0,1)); self.worker.start()

    def _worker_progress(self, text: str) -> None:
        marker = "__MINIMAX_SHOT_DONE__|"
        fail_marker = "__MINIMAX_SHOT_FAILED__|"
        if text.startswith(marker):
            parts = text.split("|", 2)
            if len(parts) == 3:
                try:
                    index = int(parts[1]); output_path = parts[2]
                    for shot in self.project.shots:
                        if shot.index == index:
                            shot.output_path = output_path; shot.status = "Generated"; break
                    self._populate_review(select_index=index)
                    self.status.setText(f"Shot {index} finished. Continuing with the remaining shots...")
                    return
                except Exception:
                    pass
        elif text.startswith(fail_marker):
            parts = text.split("|", 2)
            if len(parts) >= 2:
                try:
                    index = int(parts[1])
                    for shot in self.project.shots:
                        if shot.index == index:
                            shot.status = "Failed"; break
                    self._populate_review(select_index=index)
                    self.status.setText(f"Shot {index} failed. Continuing with the remaining shots...")
                    return
                except Exception:
                    pass
        self.status.setText(text)

    def _worker_failed(self, message: str) -> None:
        if hasattr(self, "btn_stop_generation"):
            self.btn_stop_generation.setEnabled(False)
        self._set_ready("Failed."); QMessageBox.critical(self, "MiniMax Music Clip Creator", message)

    def _start_analysis(self) -> None:
        audio = self._require_audio()
        if not audio: return
        self._pull_ui(); self._ensure_project_output_folder(reset_generated_state=True); self._set_busy("Analyzing track..."); self._run_worker(_analysis_task, self._analysis_done, audio, self.spin_sensitivity.value())

    def _analysis_done(self, result: AnalysisResult) -> None:
        self.project.analysis = result; self._populate_analysis(); self._set_ready("Music analysis complete.")

    def _update_whisper_buttons(self) -> None:
        ready = _whisper_cpp_ready()
        if hasattr(self, "btn_whisper"):
            self.btn_whisper.setEnabled(ready)
            self.btn_whisper.setToolTip(
                "Transcribe the selected song with the portable multilingual Whisper.cpp Small model."
                if ready else "Download Whisper.cpp Small first."
            )
        if hasattr(self, "btn_download_whisper"):
            self.btn_download_whisper.setText("Whisper.cpp Small installed" if ready else "Download Whisper.cpp Small")
            self.btn_download_whisper.setEnabled(not ready)

    def _download_whisper_cpp(self) -> None:
        if _whisper_cpp_ready():
            self._update_whisper_buttons()
            return
        self._set_busy("Downloading Whisper.cpp Small multilingual...")
        self._run_worker(_download_whisper_cpp_task, self._whisper_download_done)

    def _whisper_download_done(self, _path: str) -> None:
        self._update_whisper_buttons()
        self._set_ready("Whisper.cpp Small multilingual installed in presets/bin/whisper.")

    def _start_whisper(self) -> None:
        if not _whisper_cpp_ready():
            QMessageBox.information(self, "Whisper.cpp not installed", "Use the Download Whisper.cpp Small button first. It will be stored in presets/bin/whisper.")
            return
        audio = self._require_audio()
        if not audio: return
        self._set_busy("Transcribing lyrics with standalone Whisper.cpp..."); self._run_worker(_whisper_task, self._whisper_done, audio)

    def _whisper_done(self, lyrics: List[LyricSegment]) -> None:
        self.project.lyrics = lyrics; self._populate_analysis(); self._set_ready(f"Whisper.cpp produced {len(lyrics)} cleaned lyric phrases for safe shot timing." if lyrics else "Whisper.cpp found no lyric phrases; instrumental planning will be used.")

    def _clear_lyrics(self) -> None:
        self.project.lyrics = []; self._populate_analysis(); self.status.setText("Lyrics cleared. Planner will use instrumental timing.")

    def _populate_analysis(self) -> None:
        a = self.project.analysis
        dur = a.duration or (probe_duration(self.project.audio_path) if self.project.audio_path else 0.0)
        self.label_duration.setText(f"Duration: {_fmt_time(dur)} ({dur:.2f} s)" if dur else "Duration: not analyzed")
        mode = "Whisper lyric timing" if (self.project.whisper_timing_enabled and self.project.lyrics) else ("Lyrics loaded / default timing" if self.project.lyrics else "Instrumental/default timing")
        self.analysis_summary.setText(f"{mode} • {len(a.beats)} beat candidates • {len(a.sections)} energy sections • {len(self.project.lyrics)} cleaned Whisper phrases")
        self.lyrics_table.setRowCount(0)
        for seg in self.project.lyrics:
            r = self.lyrics_table.rowCount(); self.lyrics_table.insertRow(r); self.lyrics_table.setItem(r,0,QTableWidgetItem(_fmt_time(seg.start))); self.lyrics_table.setItem(r,1,QTableWidgetItem(_fmt_time(seg.end))); self.lyrics_table.setItem(r,2,QTableWidgetItem(seg.text))

    # ---- director ----
    def _create_plan(self) -> None:
        audio = self._require_audio()
        if not audio: return
        self._pull_ui()
        self._ensure_project_output_folder(reset_generated_state=True)
        if self.project.analysis.duration <= 0:
            self.project.analysis.duration = probe_duration(audio)
        try:
            self.project.shots = build_shot_plan(self.project)
            for shot in self.project.shots:
                shot.reference_names = auto_assign_references(self.project, shot); shot.prompt = build_default_prompt(self.project, shot)
            self._populate_shots(); self._populate_review(); self.status.setText(f"Created {len(self.project.shots)} MiniMax shots.")
        except Exception as exc:
            QMessageBox.critical(self, "Planning failed", str(exc))

    def _rebuild_prompts(self) -> None:
        self._pull_ui()
        self.project.reference_random_seed = self._next_job_seed() if self.project.randomize_reference_characters else -1
        for shot in self.project.shots:
            shot.reference_names = auto_assign_references(self.project, shot); shot.prompt = build_default_prompt(self.project, shot)
        self._populate_shots(); self.status.setText("MiniMax-H3 Ref2VA prompts and reference suggestions rebuilt.")

    def _populate_shots(self) -> None:
        selected_index = None
        rows = sorted({i.row() for i in self.shot_table.selectedIndexes()})
        if rows and 0 <= rows[0] < len(self.project.shots):
            selected_index = self.project.shots[rows[0]].index
        self.shot_table.blockSignals(True); self.shot_table.setRowCount(0)
        select_row = -1
        for shot in self.project.shots:
            r = self.shot_table.rowCount(); self.shot_table.insertRow(r)
            values = [
                str(shot.index), f"{_fmt_time(shot.edit_start)} – {_fmt_time(shot.edit_end)}", f"{shot.edit_duration:.2f}s",
                str(shot.frames), f"{shot.generation_duration:.2f}s", shot.section, shot.lyrics,
                ", ".join(_fmt_time(x) for x in shot.internal_cuts), ", ".join(shot.reference_names), shot.prompt,
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if c not in (8, 9): item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.shot_table.setItem(r, c, item)
            self.shot_table.setRowHeight(r, 72)
            if selected_index == shot.index:
                select_row = r
        self.shot_table.blockSignals(False)
        if select_row >= 0:
            self.shot_table.selectRow(select_row)
        self._show_selected_shot_prompt()

    def _shot_table_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not (0 <= row < len(self.project.shots)): return
        shot = self.project.shots[row]
        if item.column() == 8:
            allowed = {r.name for r in self._refs_from_table()}
            shot.reference_names = [x.strip() for x in item.text().split(",") if x.strip() in allowed][:9]
        elif item.column() == 9:
            shot.prompt = item.text().strip()
            if row in {i.row() for i in self.shot_table.selectedIndexes()}:
                self.shot_prompt_editor.blockSignals(True)
                self.shot_prompt_editor.setPlainText(shot.prompt)
                self.shot_prompt_editor.blockSignals(False)

    def _selected_director_shot(self) -> Optional[MusicShot]:
        rows = sorted({i.row() for i in self.shot_table.selectedIndexes()})
        if not rows or not (0 <= rows[0] < len(self.project.shots)):
            return None
        return self.project.shots[rows[0]]

    def _show_selected_shot_prompt(self) -> None:
        shot = self._selected_director_shot()
        self.shot_prompt_editor.blockSignals(True)
        if shot is None:
            self.shot_prompt_editor.clear()
            self.shot_prompt_editor.setPlaceholderText("Select a shot above to view or edit its complete MiniMax-H3 Ref2VA prompt.")
        else:
            self.shot_prompt_editor.setPlainText(shot.prompt or "")
        self.shot_prompt_editor.blockSignals(False)

    def _selected_shot_prompt_changed(self) -> None:
        self._commit_selected_shot_prompt()

    def _next_job_seed(self) -> int:
        return random.SystemRandom().randint(0, 2_147_483_647)

    def _ensure_job_seed(self) -> int:
        try:
            current = int(self.project.job_seed)
        except Exception:
            current = -1
        if current < 0:
            current = self._next_job_seed()
            self.project.job_seed = current
        return current

    def _prepare_generation_seeds(self, shot_indices: Sequence[int]) -> int:
        base_seed = self._ensure_job_seed()
        target_set = {int(x) for x in shot_indices}
        for shot in self.project.shots:
            if shot.seed < 0:
                # Default behavior: one random seed is chosen once for the full job
                # and reused for every shot unless the user manually overrides a row.
                shot.seed = base_seed
            elif shot.index in target_set and shot.seed >= 0:
                # Explicit per-shot retry seed override; keep it as entered.
                pass
        return base_seed

    def _review_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 5:
            return
        row = item.row()
        if not (0 <= row < len(self.project.shots)):
            return
        shot = self.project.shots[row]
        text = (item.text() or "").strip()
        try:
            value = int(text) if text else -1
        except Exception:
            self.review_table.blockSignals(True)
            item.setText(str(shot.seed))
            self.review_table.blockSignals(False)
            QMessageBox.warning(self, "Invalid seed", "Seed must be an integer. Use -1 to follow the job seed.")
            return
        if value < -1:
            self.review_table.blockSignals(True)
            item.setText(str(shot.seed))
            self.review_table.blockSignals(False)
            QMessageBox.warning(self, "Invalid seed", "Seed must be -1 or a non-negative integer.")
            return
        shot.seed = value
        if value >= 0:
            self.status.setText(f"Shot {shot.index} seed override set to {value}.")
        else:
            self.status.setText(f"Shot {shot.index} will follow the job seed.")
        self._update_job_seed_label()
        self._write_autosave(force=True)

    def _update_job_seed_label(self) -> None:
        base = getattr(self.project, "job_seed", -1)
        if base is None or int(base) < 0:
            text = "Job seed: auto (first generation locks one random seed for the whole job). Edit a shot's Seed value below to override it for retries."
        else:
            overrides = sum(1 for s in self.project.shots if int(getattr(s, "seed", -1)) >= 0 and int(getattr(s, "seed", -1)) != int(base))
            if overrides:
                text = f"Job seed: {int(base)} ({overrides} shot override{'s' if overrides != 1 else ''}). Enter -1 in a shot row to follow the job seed again."
            else:
                text = f"Job seed: {int(base)}. All shots use this seed unless you enter a different value in the Seed column for a retry."
        self.label_job_seed.setText(text)

    # ---- generation / assembly ----
    def _sync_existing_generated_outputs(self) -> int:
        """Reconcile saved shot state with raw clips already present on disk."""
        if not self.project.shots:
            return 0
        out_dir = Path(self.project.output_dir or OUTPUT_ROOT / _safe_stem(self.project.audio_path)).resolve()
        raw_dir = out_dir / "raw_clips"
        found = 0
        for shot in self.project.shots:
            expected = raw_dir / f"shot_{shot.index:03d}.mp4"
            current = Path(shot.output_path) if shot.output_path else None
            if current is not None and current.is_file():
                if shot.status != "Generated":
                    shot.status = "Generated"
                found += 1
                continue
            if expected.is_file():
                shot.output_path = str(expected)
                shot.status = "Generated"
                found += 1
            elif shot.status == "Generated":
                shot.output_path = ""
                shot.status = "Planned"
        return found

    def _populate_review(self, select_index: Optional[int] = None) -> None:
        self._sync_existing_generated_outputs()
        current_index = select_index
        if current_index is None:
            rows = sorted({i.row() for i in self.review_table.selectedIndexes()})
            if rows and 0 <= rows[0] < len(self.project.shots): current_index = self.project.shots[rows[0]].index
        self.review_table.blockSignals(True); self.review_table.setRowCount(0)
        select_row = -1
        for shot in self.project.shots:
            r = self.review_table.rowCount(); self.review_table.insertRow(r)
            vals = [str(shot.index), shot.status, str(shot.frames), f"{_fmt_time(shot.edit_start)}–{_fmt_time(shot.edit_end)}", shot.output_path, str(shot.seed)]
            for c, value in enumerate(vals):
                item = QTableWidgetItem(value)
                if c != 5:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.review_table.setItem(r, c, item)
            if current_index == shot.index: select_row = r
        self.review_table.blockSignals(False)
        self._update_job_seed_label()
        if select_row >= 0:
            self.review_table.selectRow(select_row)
        self._show_selected_clip_preview()

    def _selected_review_shot(self) -> Optional[MusicShot]:
        rows = sorted({i.row() for i in self.review_table.selectedIndexes()})
        if not rows or not (0 <= rows[0] < len(self.project.shots)): return None
        return self.project.shots[rows[0]]

    def _video_thumbnail(self, video_path: str) -> Optional[QPixmap]:
        path = Path(video_path)
        if not path.is_file(): return None
        ffmpeg = ffmpeg_path()
        if not ffmpeg: return None
        preview_dir = OUTPUT_ROOT / "_preview_cache"; preview_dir.mkdir(parents=True, exist_ok=True)
        key = f"{path.stem}_{int(path.stat().st_mtime_ns)}.jpg"
        thumb = preview_dir / key
        if not thumb.is_file():
            subprocess.run([ffmpeg, "-y", "-ss", "0.5", "-i", str(path), "-frames:v", "1", "-vf", "scale=640:-2", str(thumb)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if not thumb.is_file(): return None
        pix = QPixmap(str(thumb))
        return None if pix.isNull() else pix

    def _show_selected_clip_preview(self) -> None:
        if not hasattr(self, "review_preview"): return
        shot = self._selected_review_shot()
        if not shot:
            self.review_preview.setPixmap(QPixmap()); self.review_preview.setText("Select a generated shot to preview it."); return
        if not shot.output_path or not Path(shot.output_path).is_file():
            self.review_preview.setPixmap(QPixmap()); self.review_preview.setText(f"Shot {shot.index}: {shot.status or 'Not generated yet'}"); return
        pix = self._video_thumbnail(shot.output_path)
        if pix is None:
            self.review_preview.setPixmap(QPixmap()); self.review_preview.setText(f"Shot {shot.index} generated. Use Play selected clip to review it."); return
        self.review_preview.setText("")
        self.review_preview.setPixmap(pix.scaled(640, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.review_preview.setToolTip(shot.output_path)

    def _play_selected_clip(self) -> None:
        shot = self._selected_review_shot()
        if not shot or not shot.output_path or not Path(shot.output_path).is_file():
            QMessageBox.information(self, "No clip", "Select a generated shot first."); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(shot.output_path).resolve())))

    def _open_selected_clip_folder(self) -> None:
        shot = self._selected_review_shot()
        if not shot or not shot.output_path or not Path(shot.output_path).is_file():
            QMessageBox.information(self, "No clip", "Select a generated shot first."); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(shot.output_path).resolve().parent)))

    def _framevision_queue_toggled(self, enabled: bool) -> None:
        self.project.use_framevision_queue = bool(enabled)
        if enabled:
            try:
                self._ensure_framevision_queue_adapter()
                self.status.setText("FrameVision queue enabled. New Music Clip jobs will be added to the shared queue.")
            except Exception as exc:
                self.check_framevision_queue.blockSignals(True)
                self.check_framevision_queue.setChecked(False)
                self.check_framevision_queue.blockSignals(False)
                self.project.use_framevision_queue = False
                QMessageBox.warning(self, "FrameVision queue unavailable", str(exc))
        else:
            self.status.setText("Direct generation enabled. Music Clip jobs will run immediately in this helper.")
        try:
            self._save_settings()
            self._write_autosave(force=True)
        except Exception:
            pass

    def _ensure_framevision_queue_adapter(self):
        """Resolve FrameVision's shared queue from inside this helper.

        Hosts may still inject an adapter, but the normal FrameVision helper does not
        need its importer/host patched just to use the queue.
        """
        if callable(self.queue_adapter):
            return self.queue_adapter
        try:
            from helpers.queue_adapter import enqueue_minimax_h3_job
        except Exception:
            try:
                from queue_adapter import enqueue_minimax_h3_job
            except Exception as exc:
                raise RuntimeError(f"FrameVision queue adapter is unavailable: {exc}") from exc

        def _submit(spec: Dict[str, Any]):
            qid = enqueue_minimax_h3_job(spec)
            if not qid:
                raise RuntimeError("FrameVision queue did not accept the MiniMax Music Clip job.")
            return qid

        self.queue_adapter = _submit
        return self.queue_adapter

    def _queue_mode_active(self) -> bool:
        if not bool(getattr(self, "check_framevision_queue", None) and self.check_framevision_queue.isChecked()):
            return False
        try:
            self._ensure_framevision_queue_adapter()
            return True
        except Exception:
            return False

    def _prepare_shot_queue_job(self, shot: MusicShot) -> Dict[str, Any]:
        # A retry/recreate must use what is visible in Director right now, not a
        # regenerated project-Idea prompt and not a stale autosave copy.
        self._commit_selected_shot_prompt()
        if not MINIMAX_PY.is_file():
            raise RuntimeError(f"MiniMax environment Python not found: {MINIMAX_PY}")
        if not GENERATE_REF.is_file():
            raise RuntimeError("MiniMax Ref2VA launcher not found: helpers/generate_ref.py")
        if not Path(self.project.audio_path).is_file():
            raise RuntimeError("The project song file is missing.")
        self._ensure_project_output_folder(reset_generated_state=True)
        out_dir = Path(self.project.output_dir or OUTPUT_ROOT / _safe_stem(self.project.audio_path)).resolve()
        raw_dir = out_dir / "raw_clips"
        audio_dir = out_dir / "audio_chunks"
        raw_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_chunk = audio_dir / f"shot_{shot.index:03d}.wav"
        _extract_audio_slice(self.project.audio_path, audio_chunk, shot.generation_start, shot.frames / FPS)
        out_path, used_retry_name = _recreate_output_path(raw_dir, shot.index)
        if used_retry_name:
            self.status.setText(
                f"Shot {shot.index}: recreation uses a new output file {out_path.name} so the existing clip can stay open."
            )
        refs_by_name = {r.name: r for r in self.project.references if r.enabled and Path(r.path).is_file()}
        selected_names = [n for n in shot.reference_names if n in refs_by_name]
        if not selected_names:
            selected_names = [n for n in auto_assign_references(self.project, shot) if n in refs_by_name]
        if not selected_names and refs_by_name:
            selected_names = [next(iter(refs_by_name))]
        selected = [refs_by_name[n] for n in selected_names[:9]]
        generation_prompt = build_generation_prompt(self.project, shot, selected)

        # Persist the exact text handed to MiniMax for audit/debugging. This makes it
        # impossible to confuse the Director editor, autosave state, and actual queue prompt.
        prompt_sidecar = raw_dir / f"shot_{shot.index:03d}_prompt.txt"
        prompt_sidecar.write_text(generation_prompt, encoding="utf-8")

        width, height = RESOLUTION_PRESETS[self.project.resolution][self.project.aspect]
        args = [
            "helpers/generate_ref.py",
            "--prompt", generation_prompt,
            "--width", str(width), "--height", str(height),
            "--frames", str(shot.frames), "--steps", str(self.project.steps),
            "--cfg", str(self.project.cfg), "--seed", str(shot.seed),
            "--shift", str(self.project.shift), "--audio-shift", str(self.project.audio_shift),
            "--ref-image-size", self.project.ref_image_size,
            "--ref-audio", str(audio_chunk),
            "--output", str(out_path),
        ]
        if self.project.use_hybrid_model:
            hybrid = Path(str(self.project.hybrid_model_path or "").strip())
            if not hybrid.is_file():
                raise RuntimeError("Use hybrid model is enabled, but the selected hybrid .safetensors file was not found.")
            args += ["--ref2va-checkpoint", str(hybrid.resolve())]
        if self.project.vram_manager_enabled:
            args += ["--vram-manager-auto" if self.project.vram_auto_bypass else "--vram-manager"]
            args += [
                "--vram-residency-engine", str(self.project.vram_residency_engine or "static"),
                "--vram-runtime-free-gb", str(float(self.project.vram_runtime_free_gb)),
                "--vram-text-headroom-gb", str(float(self.project.vram_text_headroom_gb)),
                "--vram-diffusion-headroom-gb", str(float(self.project.vram_diffusion_headroom_gb)),
                "--vram-offload-chunk-mb", str(int(self.project.vram_offload_chunk_mb)),
                "--vram-max-resident-weights-gb", str(float(self.project.vram_max_resident_weights_gb)),
                "--vram-block-check-interval", str(int(self.project.vram_block_check_interval)),
                "--vram-async-streams", str(int(self.project.vram_async_streams)),
                "--vram-video-vae-reserve-gb", str(float(self.project.vram_video_vae_reserve_gb)),
                "--vram-audio-vae-reserve-gb", str(float(self.project.vram_audio_vae_reserve_gb)),
                "--vram-residency-target-free-gb", str(float(self.project.vram_residency_target_free_gb)),
                "--vram-residency-warmup-blocks", str(int(self.project.vram_residency_warmup_blocks)),
                "--vram-residency-refill-interval", str(int(self.project.vram_residency_refill_interval)),
            ]
            args += ["--vram-residency-fill" if self.project.vram_residency_fill else "--no-vram-residency-fill"]
        if self.project.sage_attention:
            args += ["--sage-attention"]
        if self.project.spectrum:
            args += ["--spectrum"]
        turbo_lora = str(self.project.turbo_lora_path or "").strip()
        if turbo_lora:
            turbo_path = Path(turbo_lora)
            if not turbo_path.is_file():
                raise RuntimeError(f"Turbo LoRA not found: {turbo_lora}")
            args += ["--lora", str(turbo_path.resolve()), "--lora-strength", str(float(self.project.turbo_lora_strength))]
        for ref in selected:
            args += ["--ref-image", ref.path]
        shot.output_path = str(out_path)
        shot.status = "Queued"
        return {
            "args": args,
            "output": str(out_path),
            "label": f"Music Clip Shot {shot.index}: {(self.project.title or _safe_stem(self.project.audio_path))}",
            "frames": int(shot.frames),
            "steps": int(self.project.steps),
            "seed": int(shot.seed),
            "resolution": f"{width} × {height}",
            "prompt": generation_prompt,
            "music_prompt_file": str(prompt_sidecar),
            "music_clip_job": True,
            "music_shot_index": int(shot.index),
            # The worker owns one immediate retry for Music Clip shots. Keeping the
            # retry inside the same queue item preserves FIFO order: final assembly
            # cannot overtake a retry and start while a required clip is still missing.
            "music_retry_once": True,
            "music_project_output": str(out_dir),
            "music_recreated_to_new_name": bool(used_retry_name),
        }

    def _queue_shots(self, indices: Sequence[int]) -> None:
        if not self._queue_mode_active():
            return
        self._pull_ui()
        self._prepare_generation_seeds(list(indices))
        by_index = {s.index: s for s in self.project.shots}
        queued = 0
        try:
            for index in indices:
                shot = by_index.get(int(index))
                if shot is None:
                    continue
                spec = self._prepare_shot_queue_job(shot)
                self.queue_adapter(spec)
                queued += 1
            self._populate_review(select_index=(int(indices[0]) if len(indices) == 1 else None))
            self._write_autosave(force=True)
            self.status.setText(f"Added {queued} Music Clip shot{'s' if queued != 1 else ''} to the MiniMax queue.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not queue Music Clip shot", str(exc))

    def _queue_assembly(self) -> None:
        if not self._queue_mode_active():
            return
        self._pull_ui()
        self._sync_existing_generated_outputs()
        out_dir = Path(self.project.output_dir or OUTPUT_ROOT / _safe_stem(self.project.audio_path)).resolve()
        raw_dir = out_dir / "raw_clips"
        for shot in self.project.shots:
            expected = raw_dir / f"shot_{shot.index:03d}.mp4"
            if not shot.output_path:
                shot.output_path = str(expected)
        queue_dir = out_dir / "_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        snapshot = queue_dir / f"assembly_project_{int(time.time() * 1000)}.json"
        snapshot.write_text(json.dumps(asdict(self.project), ensure_ascii=False, indent=2), encoding="utf-8")
        final = out_dir / f"{_safe_stem(self.project.title or self.project.audio_path)}_minimax_music_video.mp4"
        spec = {
            "args": ["helpers/minimax_music_clip.py", "--queue-assemble", str(snapshot)],
            "output": str(final),
            "label": f"Assemble Music Video: {(self.project.title or _safe_stem(self.project.audio_path))}",
            "frames": 0,
            "steps": 0,
            "seed": -1,
            "resolution": "Music assembly",
            "prompt": "Assemble trimmed Music Clip shots and mux the original master song.",
            "music_assembly": True,
            "music_project_output": str(out_dir),
        }
        self.queue_adapter(spec)
        self.status.setText("Added final Music Clip assembly to the MiniMax queue.")

    def external_queue_updated(self, job: Optional[Dict[str, Any]] = None) -> None:
        """Called by the standalone host when its queue changes or a Music Clip job ends."""
        changed = self._sync_existing_generated_outputs()
        if changed or (job and (job.get("music_shot_index") or job.get("music_assembly"))):
            self._populate_review(select_index=(int(job.get("music_shot_index")) if job and job.get("music_shot_index") else None))
            self._write_autosave(force=True)

    def _generate_selected(self) -> None:
        rows = sorted({i.row() for i in self.review_table.selectedIndexes()})
        if not rows: QMessageBox.information(self, "Select shot", "Select a shot row first."); return
        indices = [self.project.shots[rows[0]].index]
        if self._queue_mode_active():
            self._queue_shots(indices)
            return
        self._generate_indices(indices)

    def _generate_all(self) -> None:
        self._sync_existing_generated_outputs()
        indices = [s.index for s in self.project.shots if not s.output_path or not Path(s.output_path).is_file() or s.status == "Failed"]
        if not indices: self.status.setText("All planned shots already have outputs."); return
        if self._queue_mode_active():
            self._queue_shots(indices)
            return
        self._generate_indices(indices)

    def _generate_indices(self, indices: List[int]) -> None:
        if not self.project.shots: QMessageBox.warning(self, "No plan", "Create a shot plan first."); return
        _GENERATION_CANCEL.clear()
        self.btn_stop_generation.setEnabled(True)
        self._pull_ui()
        self._ensure_project_output_folder(reset_generated_state=True)
        base_seed = self._prepare_generation_seeds(indices)
        selected_index = indices[0] if len(indices) == 1 else None
        self._populate_review(select_index=selected_index)
        self._set_busy(f"Starting MiniMax Ref2VA generation... job seed {base_seed}")
        self._write_autosave(force=True)
        self._run_worker(_generation_task, self._generation_done, self.project, indices)

    def _stop_generation(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self.btn_stop_generation.setEnabled(False)
            return
        _GENERATION_CANCEL.set()
        self.btn_stop_generation.setEnabled(False)
        self.status.setText("Stopping MiniMax generation...")
        _terminate_generation_process_tree()

    def _generation_done(self, results: List[Dict[str, Any]]) -> None:
        by_index = {s.index: s for s in self.project.shots}
        failures = []
        cancelled = False
        for result in results:
            shot = by_index.get(int(result.get("index", -1)))
            if not shot: continue
            if result.get("ok"):
                shot.output_path = str(result.get("output_path") or ""); shot.status = "Generated"
                if result.get("seed") is not None:
                    try:
                        shot.seed = int(result.get("seed"))
                    except Exception:
                        pass
            elif result.get("cancelled"):
                shot.status = "Planned"
                cancelled = True
            else:
                shot.status = "Failed"; failures.append(f"Shot {shot.index}: {result.get('message','failed')}")
        self.btn_stop_generation.setEnabled(False)
        self._populate_review()
        if cancelled or _GENERATION_CANCEL.is_set():
            self._set_ready("Generation stopped. Finished clips were kept; unfinished clips remain planned.")
        else:
            self._set_ready("Generation finished." if not failures else f"Generation finished with {len(failures)} failed shot(s).")
        if failures: QMessageBox.warning(self, "Some shots failed", "\n\n".join(failures[:5]))

    def _assemble(self) -> None:
        if not self.project.shots: QMessageBox.warning(self, "No plan", "Create and generate the shot plan first."); return
        if self._queue_mode_active():
            self._queue_assembly()
            return
        self._pull_ui(); self._set_busy("Assembling final music video..."); self._run_worker(_assembly_task, self._assembly_done, self.project)

    def _assembly_done(self, path: str) -> None:
        self._set_ready(f"Final video saved: {path}")
        QMessageBox.information(self, "Finished", f"Final music video saved:\n{path}")

    def _open_output_folder(self) -> None:
        self._pull_ui()
        path = self._ensure_project_output_folder(reset_generated_state=False)
        path.mkdir(parents=True, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _sync_hybrid_model_controls(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.edit_hybrid_model.setEnabled(enabled)
        self.btn_hybrid_model.setEnabled(enabled)

    def _browse_hybrid_model(self) -> None:
        start = self.edit_hybrid_model.text().strip() or str(ROOT / "models" / "minimax_h3")
        path, _ = QFileDialog.getOpenFileName(self, "Select hybrid MiniMax H3 checkpoint", start, "SafeTensors (*.safetensors);;All files (*.*)")
        if path:
            self.edit_hybrid_model.setText(path)
            self._pull_ui(); self._save_settings()

    def _browse_turbo_lora(self) -> None:
        start = self.edit_turbo_lora.text().strip()
        if start and Path(start).is_file():
            start = str(Path(start).parent)
        if not start:
            start = str(ROOT / "models" / "minimax_h3" / "loras")
        path, _ = QFileDialog.getOpenFileName(self, "Select MiniMax Turbo / speed LoRA", start, "LoRA files (*.safetensors *.pt *.bin);;All files (*.*)")
        if path:
            self.edit_turbo_lora.setText(path)

    # ---- working-session autosave ----
    def _autosave_payload(self, pull_ui: bool = True) -> Dict[str, Any]:
        """Return the complete recoverable working state.

        ``pull_ui`` must be False while *loading* a saved session. The widgets still
        contain their construction defaults at that point; pulling them would overwrite
        the just-restored project (idea text, references, LoRA, steps, etc.).
        """
        if pull_ui:
            self._pull_ui()
        return {
            "version": 2,
            "project_path": str(self.project_path or ""),
            "project": asdict(self.project),
        }

    def _load_autosave(self) -> bool:
        try:
            if not AUTOSAVE_PATH.is_file():
                return False
            data = json.loads(AUTOSAVE_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False
            # Current format wraps the project so we can also remember the explicit
            # project filename. Accept an old/direct project dict as a safe fallback.
            project_data = data.get("project") if isinstance(data.get("project"), dict) else data
            self.project = self._project_from_dict(project_data)
            self.project_path = str(data.get("project_path") or "") if isinstance(data.get("project"), dict) else ""
            # Critical: do NOT pull values from the not-yet-synchronised widgets here.
            # Doing that used to replace restored fields with UI defaults on every start.
            self._autosave_last_text = json.dumps(self._autosave_payload(pull_ui=False), ensure_ascii=False, sort_keys=True)
            return True
        except Exception:
            return False

    def _write_autosave(self, force: bool = False) -> None:
        try:
            self._commit_selected_shot_prompt()
            payload = self._autosave_payload()
            compare_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if not force and compare_text == self._autosave_last_text:
                return
            AUTOSAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = AUTOSAVE_PATH.with_suffix(AUTOSAVE_PATH.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(AUTOSAVE_PATH))
            self._autosave_last_text = compare_text
            # Generation preferences also survive deliberate New Project/session resets.
            self._save_settings()
        except Exception:
            # Autosave must never interrupt generation or normal GUI use.
            pass

    def _start_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.timeout.connect(self._write_autosave)
        self._autosave_timer.start()
        # Create presets/setsave/minimax_music_clip.json immediately, even before
        # the first edit, so the persistence location is predictable.
        self._write_autosave(force=True)
        app = QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(lambda: self._write_autosave(force=True))
            except Exception:
                pass

    # ---- settings ----
    def _update_frame_label(self) -> None:
        frames = MUSIC_FRAME_GRID[self.slider_frames.value()]
        self.label_frames.setText(f"{frames} frames (~{frames / FPS:.2f} s)")

    def _load_settings(self) -> None:
        try:
            if SETTINGS_PATH.is_file():
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                self.project.output_dir = str(data.get("output_dir") or self.project.output_dir)
                self.project.resolution = str(data.get("resolution") or self.project.resolution)
                self.project.aspect = str(data.get("aspect") or self.project.aspect)
                self.project.max_frames = int(data.get("max_frames") or self.project.max_frames)
                self.project.steps = int(data.get("steps") or self.project.steps)
                self.project.turbo_lora_path = str(data.get("turbo_lora_path") or self.project.turbo_lora_path)
                self.project.turbo_lora_strength = float(data.get("turbo_lora_strength", self.project.turbo_lora_strength))
                # Migration: older Music Clip Creator builds stored one vram_auto flag.
                if "vram_manager_enabled" in data:
                    self.project.vram_manager_enabled = bool(data.get("vram_manager_enabled"))
                    self.project.vram_auto_bypass = bool(data.get("vram_auto_bypass", True))
                elif "vram_auto" in data:
                    # Old checked state meant --vram-manager-auto; old unchecked state meant no manager.
                    self.project.vram_manager_enabled = bool(data.get("vram_auto"))
                    self.project.vram_auto_bypass = True
                for key in (
                    "vram_residency_engine", "vram_runtime_free_gb", "vram_text_headroom_gb",
                    "vram_diffusion_headroom_gb", "vram_offload_chunk_mb", "vram_max_resident_weights_gb",
                    "vram_block_check_interval", "vram_async_streams", "vram_video_vae_reserve_gb",
                    "vram_audio_vae_reserve_gb", "vram_residency_fill", "vram_residency_target_free_gb",
                    "vram_residency_warmup_blocks", "vram_residency_refill_interval",
                ):
                    if key in data:
                        setattr(self.project, key, data[key])
                self.project.use_hybrid_model = bool(data.get("use_hybrid_model", self.project.use_hybrid_model))
                self.project.hybrid_model_path = str(data.get("hybrid_model_path") or self.project.hybrid_model_path)
                self.project.sage_attention = bool(data.get("sage_attention", self.project.sage_attention))
                self.project.spectrum = bool(data.get("spectrum", self.project.spectrum))
                self.project.randomize_reference_characters = bool(data.get("randomize_reference_characters", self.project.randomize_reference_characters))
                self.project.use_framevision_queue = bool(data.get("use_framevision_queue", getattr(self.project, "use_framevision_queue", False)))
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "output_dir": self.project.output_dir, "resolution": self.project.resolution, "aspect": self.project.aspect,
                "max_frames": self.project.max_frames, "steps": self.project.steps,
                "vram_manager_enabled": self.project.vram_manager_enabled, "vram_auto_bypass": self.project.vram_auto_bypass,
                "vram_residency_engine": self.project.vram_residency_engine,
                "vram_runtime_free_gb": self.project.vram_runtime_free_gb,
                "vram_text_headroom_gb": self.project.vram_text_headroom_gb,
                "vram_diffusion_headroom_gb": self.project.vram_diffusion_headroom_gb,
                "vram_offload_chunk_mb": self.project.vram_offload_chunk_mb,
                "vram_max_resident_weights_gb": self.project.vram_max_resident_weights_gb,
                "vram_block_check_interval": self.project.vram_block_check_interval,
                "vram_async_streams": self.project.vram_async_streams,
                "vram_video_vae_reserve_gb": self.project.vram_video_vae_reserve_gb,
                "vram_audio_vae_reserve_gb": self.project.vram_audio_vae_reserve_gb,
                "vram_residency_fill": self.project.vram_residency_fill,
                "vram_residency_target_free_gb": self.project.vram_residency_target_free_gb,
                "vram_residency_warmup_blocks": self.project.vram_residency_warmup_blocks,
                "vram_residency_refill_interval": self.project.vram_residency_refill_interval,
                "turbo_lora_path": self.project.turbo_lora_path, "turbo_lora_strength": self.project.turbo_lora_strength,
                "use_hybrid_model": self.project.use_hybrid_model, "hybrid_model_path": self.project.hybrid_model_path,
                "sage_attention": self.project.sage_attention, "spectrum": self.project.spectrum,
                "randomize_reference_characters": self.project.randomize_reference_characters,
                "use_framevision_queue": bool(getattr(self.project, "use_framevision_queue", False)),
            }
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


class MiniMaxMusicClipWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniMax H3 Music Clip Creator")
        self.resize(1500, 900)
        self.setCentralWidget(MiniMaxMusicClipWidget(self))


def install_minimax_music_clip_tool(parent: QWidget, container: QWidget, queue_adapter=None) -> MiniMaxMusicClipWidget:
    """Small integration hook for FrameVision/standalone hosts."""
    widget = MiniMaxMusicClipWidget(container, queue_adapter=queue_adapter)
    layout = container.layout()
    if layout is None:
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    return widget


def _queue_cli_main(argv: Sequence[str]) -> Optional[int]:
    if "--queue-assemble" not in argv:
        return None
    try:
        pos = list(argv).index("--queue-assemble")
        project_path = Path(argv[pos + 1]).resolve()
        data = json.loads(project_path.read_text(encoding="utf-8"))
        project = MiniMaxMusicClipWidget._project_from_dict(data)
        result = _assembly_task(lambda text: print(text, flush=True), project)
        print(f"Saved final music video: {result}", flush=True)
        return 0
    except Exception as exc:
        print(f"Music Clip assembly failed: {exc}", file=sys.stderr, flush=True)
        return 2


def main() -> int:
    cli_rc = _queue_cli_main(sys.argv[1:])
    if cli_rc is not None:
        return cli_rc
    app = QApplication.instance() or QApplication(sys.argv)
    win = MiniMaxMusicClipWindow(); win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
