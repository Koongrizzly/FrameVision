
"""
Transition Video helper for FrameVision

New standalone helper:
- load a list or folder with video files
- thumbnail previews with double-click playback / right-click menu
- per-clip transition selection
- randomize transitions or use one transition for all
- transition duration control
- optional audio mixing, optional mute on save
- export controls for fps / resolution / bitrate
- settings JSON saved under /presets/setsave/ with protected startup loading
- ffmpeg / ffprobe expected under /presets/bin/

Integration example:

    from helpers.transition_video import install_transition_video_tool
    sec_transition_video = CollapsibleSection("Transition Video", expanded=False)
    install_transition_video_tool(self, sec_transition_video)
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, QTimer, QSize, Signal, QPoint, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QAbstractItemView,
    QGridLayout,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".m4v"}

# Transition list aligned to the names used by the uploaded auto_music_sync reference.
# Some of the more custom transitions are implemented as safe "closest ffmpeg equivalent"
# variants so the helper stays standalone and does not depend on the bigger music helper.
TRANSITIONS: List[Tuple[str, str]] = [
    ("hardcut", "Hard cut"),
    ("t_exposure_dissolve", "Exposure dissolve"),
    ("t_scale_punch", "Scale punch"),
    ("t_shimmer_blur", "Shimmer blur"),
    ("t_iris", "Iris reveal"),
    ("t_motion_blur", "Motion blur whip"),
    ("t_slitscan_push", "Slit-scan push"),
    ("t_radial_burst", "Radial burst reveal"),
    ("t_push", "Directional push"),
    ("t_wipe", "Wipe"),
    ("t_smooth_zoom", "Smooth zoom crossfade"),
    ("t_curtain_open", "Curtain open"),
    ("t_pixelize", "Pixelize"),
    ("t_distance", "Distance"),
    ("t_wind", "Wind smears"),
    ("t_circle_reveal", "Circle reveal"),
    ("t_rectangle_reveal", "Rectangle reveal"),
    ("t_zoom_out", "Zoom out"),
    ("t_zoom_in", "Zoom in"),
    ("t_mirror_slide", "Mirror slide"),
]
TRANSITION_LABELS = {k: v for k, v in TRANSITIONS}


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _app_root() -> str:
    here = _here()
    if os.path.basename(here).lower() == "helpers":
        return os.path.abspath(os.path.join(here, ".."))
    return here


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _thumbnail_cache_dir() -> str:
    path = os.path.join(_app_root(), "temp", "transition_video_thumbs")
    _ensure_dir(path)
    return path


def _settings_path() -> str:
    path = os.path.join(_app_root(), "presets", "setsave", "transition_video_settings.json")
    _ensure_dir(os.path.dirname(path))
    return path


def _output_dir() -> str:
    # Keep the user's requested folder spelling.
    path = os.path.join(_app_root(), "output", "transtion video")
    _ensure_dir(path)
    return path


def _ffmpeg_path() -> str:
    env = os.environ.get("FV_FFMPEG", "").strip()
    if env and os.path.exists(env):
        return env
    root = _app_root()
    candidates = [
        os.path.join(root, "presets", "bin", "ffmpeg.exe"),
        os.path.join(root, "presets", "bin", "ffmpeg"),
        "ffmpeg.exe",
        "ffmpeg",
    ]
    for c in candidates:
        if os.path.exists(c) or shutil.which(c):
            return c
    return "ffmpeg"


def _ffprobe_path() -> str:
    env = os.environ.get("FV_FFPROBE", "").strip()
    if env and os.path.exists(env):
        return env
    root = _app_root()
    candidates = [
        os.path.join(root, "presets", "bin", "ffprobe.exe"),
        os.path.join(root, "presets", "bin", "ffprobe"),
        "ffprobe.exe",
        "ffprobe",
    ]
    for c in candidates:
        if os.path.exists(c) or shutil.which(c):
            return c
    return "ffprobe"


def _sanitize_stem(value: str, fallback: str = "transition_video") -> str:
    try:
        s = str(value)
    except Exception:
        s = fallback
    s = s.strip().replace(" ", "_")
    s = "".join("_" if (ord(ch) < 32 or ch in '<>:"/\\|?*') else ch for ch in s)
    s = re.sub(r"_+", "_", s).strip("._ ")
    return s or fallback


def _atomic_write_json(path: str, data: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    bak = path + ".bak"
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    try:
        if os.path.exists(path):
            try:
                if os.path.exists(bak):
                    os.remove(bak)
            except Exception:
                pass
            try:
                os.replace(path, bak)
            except Exception:
                try:
                    shutil.copy2(path, bak)
                except Exception:
                    pass
    except Exception:
        pass
    os.replace(tmp, path)


def _read_json(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_process(cmd: List[str]) -> Tuple[int, str]:
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
        return 1, f"Failed to run process: {e!r}"


def _ffprobe_json(path: str) -> dict:
    cmd = [
        _ffprobe_path(),
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = json.loads(out or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "nan"):
            return default
        return float(s)
    except Exception:
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "nan"):
            return default
        return int(float(s))
    except Exception:
        return default


def _probe_media(path: str) -> dict:
    data = _ffprobe_json(path)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    v = None
    a = None
    for st in streams:
        ctype = str(st.get("codec_type") or "").lower()
        if ctype == "video" and v is None:
            v = st
        elif ctype == "audio" and a is None:
            a = st

    width = _to_int((v or {}).get("width"), 0)
    height = _to_int((v or {}).get("height"), 0)

    fps_expr = str((v or {}).get("avg_frame_rate") or (v or {}).get("r_frame_rate") or "").strip()
    fps = 0.0
    if fps_expr and fps_expr not in ("0/0", "0"):
        try:
            if "/" in fps_expr:
                n, d = fps_expr.split("/", 1)
                denom = float(d)
                if abs(denom) > 1e-9:
                    fps = float(n) / denom
            else:
                fps = float(fps_expr)
        except Exception:
            fps = 0.0

    duration = _to_float(fmt.get("duration"), 0.0)
    v_bitrate = _to_int((v or {}).get("bit_rate"), 0)
    a_bitrate = _to_int((a or {}).get("bit_rate"), 0)
    fmt_bitrate = _to_int(fmt.get("bit_rate"), 0)
    if v_bitrate <= 0 and fmt_bitrate > 0:
        if a_bitrate > 0 and fmt_bitrate > a_bitrate:
            v_bitrate = max(0, fmt_bitrate - a_bitrate)
        else:
            v_bitrate = fmt_bitrate

    has_audio = a is not None
    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "video_bitrate": v_bitrate,
        "audio_bitrate": a_bitrate,
        "has_audio": has_audio,
    }


def _display_name(path: str) -> str:
    return os.path.basename(path)


def _transition_label(transition_id: str) -> str:
    return TRANSITION_LABELS.get(transition_id, transition_id)


def _make_item_text(path: str, transition_id: str) -> str:
    return f"{_display_name(path)}\n→ {_transition_label(transition_id)}"


def _thumbnail_path_for(path: str) -> str:
    digest = hashlib.md5(path.encode("utf-8", errors="ignore")).hexdigest()
    return os.path.join(_thumbnail_cache_dir(), f"{digest}.jpg")


def _generate_thumbnail(path: str) -> Optional[str]:
    out_path = _thumbnail_path_for(path)
    if os.path.exists(out_path):
        return out_path

    meta = _probe_media(path)
    dur = max(0.0, float(meta.get("duration") or 0.0))
    t = 0.5
    if dur > 1.5:
        t = min(1.0, dur * 0.25)

    cmd = [
        _ffmpeg_path(),
        "-y",
        "-ss", f"{t:.3f}",
        "-i", path,
        "-frames:v", "1",
        "-vf",
        "scale=240:135:force_original_aspect_ratio=decrease,"
        "pad=240:135:(ow-iw)/2:(oh-ih)/2:color=black",
        out_path,
    ]
    code, _ = _run_process(cmd)
    if code == 0 and os.path.exists(out_path):
        return out_path
    return None


def _same_enough(values: List[float], tolerance_ratio: float = 0.10, abs_tol: float = 0.75) -> bool:
    vals = [float(v) for v in values if float(v) > 0.0]
    if not vals:
        return False
    base = vals[0]
    for v in vals[1:]:
        if abs(v - base) > max(abs_tol, abs(base) * tolerance_ratio):
            return False
    return True


def _pick_keep_source_fps_or_fallback(items: List[dict]) -> float:
    vals = [float(m.get("fps") or 0.0) for m in items]
    vals = [v for v in vals if v > 0.0]
    if not vals:
        return 30.0
    if _same_enough(vals, tolerance_ratio=0.03, abs_tol=0.75):
        return round(sum(vals) / len(vals), 3)
    return 30.0


def _pick_keep_source_bitrate_or_fallback(items: List[dict]) -> int:
    vals = [int(m.get("video_bitrate") or 0) for m in items]
    vals = [v for v in vals if v > 0]
    if not vals:
        return 3500
    kbps = [v / 1000.0 for v in vals]
    if _same_enough(kbps, tolerance_ratio=0.15, abs_tol=450.0):
        return int(round(sum(kbps) / len(kbps)))
    return 3500


def _pick_keep_source_resolution_or_fallback(items: List[dict]) -> Tuple[int, int]:
    sizes = []
    for m in items:
        w = int(m.get("width") or 0)
        h = int(m.get("height") or 0)
        if w > 0 and h > 0:
            sizes.append((w, h))
    if not sizes:
        return 1280, 720
    first = sizes[0]
    if all(s == first for s in sizes[1:]):
        return first
    return 1280, 720


def _target_resolution(height_name: str, items: Optional[List[dict]] = None) -> Tuple[int, int]:
    key = str(height_name).strip()
    if key == "keep":
        return _pick_keep_source_resolution_or_fallback(items or [])
    if key == "1080":
        return 1920, 1080
    if key == "720":
        return 1280, 720
    return 854, 480


@dataclass
class RenderConfig:
    items: List[dict]
    randomize_transitions: bool
    randomize_transition_ids: List[str]
    one_transition_for_all: bool
    global_transition_id: str
    transition_seconds: float
    mute_output: bool
    fps_mode: str
    resolution_mode: str
    quality_mode: str
    output_name: str
    output_folder: str


class TransitionVideoRenderWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    done = Signal(bool, str)

    def __init__(self, cfg: RenderConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cfg = cfg
        self.ffmpeg = _ffmpeg_path()
        self.ffprobe = _ffprobe_path()

    def _log(self, text: str) -> None:
        try:
            self.log.emit(str(text))
        except Exception:
            pass

    def _resolve_render_fps(self, metas: List[dict]) -> float:
        if self.cfg.fps_mode == "30":
            return 30.0
        return _pick_keep_source_fps_or_fallback(metas)

    def _resolve_render_bitrate_kbps(self, metas: List[dict]) -> int:
        if self.cfg.quality_mode == "5000":
            return 5000
        if self.cfg.quality_mode == "keep":
            return _pick_keep_source_bitrate_or_fallback(metas)
        return 3500

    def _norm_scale_filter(self, width: int, height: int) -> str:
        # Resize, not upscale.
        # Each clip is kept inside the target canvas. Smaller clips stay smaller and get padded.
        return (
            f"scale='min(iw,{width})':'min(ih,{height})':force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    def _prepare_clip(self, meta: dict, out_path: str, fps: float, width: int, height: int, bitrate_kbps: int) -> None:
        duration = max(0.1, float(meta.get("duration") or 0.0))
        src = meta["path"]
        vf = f"{self._norm_scale_filter(width, height)},fps={fps:.6f},format=yuv420p"

        cmd = [self.ffmpeg, "-y", "-i", src]
        has_audio = bool(meta.get("has_audio"))

        if self.cfg.mute_output:
            cmd += [
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-b:v", f"{int(bitrate_kbps)}k",
                "-pix_fmt", "yuv420p",
                "-an",
                out_path,
            ]
        else:
            if has_audio:
                cmd += [
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-b:v", f"{int(bitrate_kbps)}k",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "48000",
                    "-ac", "2",
                    out_path,
                ]
            else:
                # Add silent audio so every staged file has an audio stream and acrossfade stays simple.
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", src,
                    "-f", "lavfi",
                    "-t", f"{duration:.6f}",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-filter_complex",
                    f"[0:v]{vf}[v];[1:a]anull[a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-b:v", f"{int(bitrate_kbps)}k",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "48000",
                    "-ac", "2",
                    "-shortest",
                    out_path,
                ]

        code, out = _run_process(cmd)
        if code != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Failed to normalize clip:\n{src}\n\n{out}")

    def _resolve_transition_for_join(self, join_index: int) -> str:
        # join_index is the index of clip A in A->B
        transition_id = str(self.cfg.items[join_index].get("transition_id") or "hardcut")
        if self.cfg.one_transition_for_all:
            transition_id = self.cfg.global_transition_id
        elif self.cfg.randomize_transitions:
            allowed = [str(tid) for tid in (self.cfg.randomize_transition_ids or []) if str(tid)]
            if not allowed:
                allowed = [tid for tid, _ in TRANSITIONS]
            try:
                transition_id = random.choice(allowed)
            except Exception:
                transition_id = allowed[0] if allowed else "hardcut"
        return transition_id

    def _stitch_hardcut(self, a: str, b: str, out_path: str, with_audio: bool) -> None:
        if with_audio:
            fc = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                out_path,
            ]
        else:
            fc = "[0:v][1:v]concat=n=2:v=1:a=0[v]"
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-an",
                out_path,
            ]
        code, out = _run_process(cmd)
        if code != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Hard cut stitch failed.\n\n{out}")

    def _transition_mode_to_xfade(self, transition_id: str) -> str:
        mapping = {
            "t_exposure_dissolve": "fade",
            "t_shimmer_blur": "hblur",
            "t_iris": "circleopen",
            "t_motion_blur": "hblur",
            "t_slitscan_push": "slideleft",
            "t_radial_burst": "radial",
            "t_push": "slideleft",
            "t_wipe": "wipeleft",
            "t_curtain_open": "custom",
            "t_pixelize": "pixelize",
            "t_distance": "distance",
            "t_wind": "hblur",
            "t_circle_reveal": "circleopen",
            "t_rectangle_reveal": "rectcrop",
            "t_mirror_slide": "slideleft",
            "t_zoom_out": "fade",
            "t_zoom_in": "fade",
        }
        return mapping.get(transition_id, "fade")

    def _stitch_scale_like(self, a: str, b: str, out_path: str, width: int, height: int, duration: float, with_audio: bool, gentle: bool) -> None:
        dur_a = max(0.0, _probe_media(a).get("duration") or 0.0)
        dur_b = max(0.0, _probe_media(b).get("duration") or 0.0)
        d = max(0.12, min(float(duration), min(dur_a, dur_b) * 0.45 if dur_a and dur_b else float(duration)))
        offset = max(0.0, dur_a - d)
        t0 = offset
        t1 = offset + d
        z0 = 1.08 if gentle else 1.18
        z_amp = max(0.01, z0 - 1.0)
        z_expr = f"max(1.0\\,{z0:.4f} - ({z_amp:.4f})*(t-{t0:.4f})/{d:.4f})"

        if gentle:
            mid_fx = (
                f"eq=contrast=1.03:brightness=0.01:saturation=1.04,"
                f"scale=iw*({z_expr}):ih*({z_expr}):eval=frame,"
                f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
            )
        else:
            mid_fx = (
                "eq=contrast=1.06:brightness=0.02:saturation=1.08,"
                f"tmix=frames=5:weights='1 0.88 0.68 0.48 0.28':enable='between(t,{t0:.3f},{t1:.3f})',"
                f"scale=iw*({z_expr}):ih*({z_expr}):eval=frame,"
                f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
            )

        if with_audio:
            fc = (
                f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{mid_fx},format=yuv420p[v];"
                f"[0:a][1:a]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[a]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                out_path,
            ]
        else:
            fc = (
                f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{mid_fx},format=yuv420p[v]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-an",
                out_path,
            ]

        code, out = _run_process(cmd)
        if code != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Scale-style transition failed.\n\n{out}")


    def _stitch_zoom_transition(self, a: str, b: str, out_path: str, width: int, height: int, duration: float, with_audio: bool, zoom_in: bool) -> None:
        dur_a = max(0.0, _probe_media(a).get("duration") or 0.0)
        dur_b = max(0.0, _probe_media(b).get("duration") or 0.0)
        d = max(0.12, min(float(duration), min(dur_a, dur_b) * 0.45 if dur_a and dur_b else float(duration)))
        offset = max(0.0, dur_a - d)
        t0 = offset
        t1 = offset + d
        z1 = 1.12
        if zoom_in:
            z_expr = f"min({z1:.4f}\\,1.0 + ({z1 - 1.0:.4f})*(t-{t0:.4f})/{d:.4f})"
        else:
            z_expr = f"max(1.0\\,{z1:.4f} - ({z1 - 1.0:.4f})*(t-{t0:.4f})/{d:.4f})"
        mid_fx = (
            f"scale=trunc({width}*({z_expr})/2)*2:trunc({height}*({z_expr})/2)*2:eval=frame,"
            f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,"
            f"format=yuv420p"
        )

        if with_audio:
            fc = (
                f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{mid_fx}[v];"
                f"[0:a][1:a]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[a]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                out_path,
            ]
        else:
            fc = (
                f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{mid_fx}[v]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-an",
                out_path,
            ]
        code, out = _run_process(cmd)
        if code != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Zoom transition failed.\n\n{out}")

    def _stitch_simple_xfade(self, a: str, b: str, out_path: str, width: int, height: int, duration: float, with_audio: bool, transition_id: str) -> None:
        dur_a = max(0.0, _probe_media(a).get("duration") or 0.0)
        dur_b = max(0.0, _probe_media(b).get("duration") or 0.0)
        d = max(0.12, min(float(duration), min(dur_a, dur_b) * 0.45 if dur_a and dur_b else float(duration)))
        mode = self._transition_mode_to_xfade(transition_id)
        offset = max(0.0, dur_a - d)

        if transition_id == "t_curtain_open":
            expr = "if(lte(abs(X-W/2),(W*P/2)),B,A)"
            if with_audio:
                fc = (
                    f"[0:v][1:v]xfade=transition=custom:expr='{expr}':duration={d:.3f}:offset={offset:.3f}[x];"
                    f"[x]format=yuv420p[v];"
                    f"[0:a][1:a]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[a]"
                )
                cmd = [
                    self.ffmpeg, "-y", "-i", a, "-i", b, "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path,
                ]
            else:
                fc = f"[0:v][1:v]xfade=transition=custom:expr='{expr}':duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                cmd = [
                    self.ffmpeg, "-y", "-i", a, "-i", b, "-filter_complex", fc,
                    "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-an", out_path,
                ]
            code, out = _run_process(cmd)
            if code != 0 or not os.path.exists(out_path):
                raise RuntimeError(f"Transition stitch failed ({transition_id}).\n\n{out}")
            return

        # Small extra polish for a couple of modes so they feel closer to the music helper.
        post_fx = "format=yuv420p"
        if transition_id == "t_radial_burst":
            post_fx = (
                f"eq=brightness=0.04:contrast=1.06:enable='between(t,{offset:.3f},{offset + min(d, 0.14):.3f})',"
                "format=yuv420p"
            )
        elif transition_id == "t_shimmer_blur":
            post_fx = (
                f"eq=saturation=1.08:contrast=1.03:enable='between(t,{offset:.3f},{offset + d:.3f})',"
                "format=yuv420p"
            )
        elif transition_id == "t_wind":
            post_fx = (
                f"tmix=frames=4:weights='1 0.70 0.45 0.22':enable='between(t,{offset:.3f},{offset + d:.3f})',"
                "format=yuv420p"
            )

        if with_audio:
            fc = (
                f"[0:v][1:v]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{post_fx}[v];"
                f"[0:a][1:a]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[a]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                out_path,
            ]
        else:
            fc = (
                f"[0:v][1:v]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f}[x];"
                f"[x]{post_fx}[v]"
            )
            cmd = [
                self.ffmpeg, "-y",
                "-i", a,
                "-i", b,
                "-filter_complex", fc,
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-an",
                out_path,
            ]
        code, out = _run_process(cmd)
        if code != 0 or not os.path.exists(out_path):
            # safe fallback
            if with_audio:
                fc2 = (
                    f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v];"
                    f"[0:a][1:a]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[a]"
                )
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", a,
                    "-i", b,
                    "-filter_complex", fc2,
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    out_path,
                ]
            else:
                fc2 = f"[0:v][1:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", a,
                    "-i", b,
                    "-filter_complex", fc2,
                    "-map", "[v]",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    out_path,
                ]
            code, out = _run_process(cmd)
            if code != 0 or not os.path.exists(out_path):
                raise RuntimeError(f"Transition stitch failed ({transition_id}).\n\n{out}")

    def _stitch_pair(self, a: str, b: str, out_path: str, width: int, height: int, transition_id: str, duration: float, with_audio: bool) -> None:
        if transition_id == "hardcut":
            self._stitch_hardcut(a, b, out_path, with_audio)
            return
        if transition_id == "t_scale_punch":
            self._stitch_scale_like(a, b, out_path, width, height, duration, with_audio, gentle=False)
            return
        if transition_id == "t_smooth_zoom":
            self._stitch_scale_like(a, b, out_path, width, height, duration, with_audio, gentle=True)
            return
        if transition_id == "t_zoom_out":
            self._stitch_zoom_transition(a, b, out_path, width, height, duration, with_audio, zoom_in=False)
            return
        if transition_id == "t_zoom_in":
            self._stitch_zoom_transition(a, b, out_path, width, height, duration, with_audio, zoom_in=True)
            return
        self._stitch_simple_xfade(a, b, out_path, width, height, duration, with_audio, transition_id)

    def _build_single_pass_filter(self, normalized_paths: List[str], with_audio: bool, width: int, height: int) -> Tuple[str, str, Optional[str]]:
        durations = []
        for path in normalized_paths:
            meta = _probe_media(path)
            durations.append(max(0.001, float(meta.get("duration") or 0.0)))

        parts: List[str] = []
        for i in range(len(normalized_paths)):
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            if with_audio:
                parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")

        current_v = "v0"
        current_a = "a0" if with_audio else None
        current_duration = durations[0]

        for join_idx in range(1, len(normalized_paths)):
            transition_id = self._resolve_transition_for_join(join_idx - 1)
            dur_next = durations[join_idx]
            d = max(0.12, min(float(self.cfg.transition_seconds), min(current_duration, dur_next) * 0.45))
            in_v = f"v{join_idx}"
            in_a = f"a{join_idx}" if with_audio else None
            out_v = f"vx{join_idx}"
            out_a = f"ax{join_idx}" if with_audio else None
            label = _transition_label(transition_id)
            self._log(f"Joining {join_idx}/{len(normalized_paths)-1} with: {label} (single-pass)")

            if transition_id == "hardcut":
                if with_audio:
                    parts.append(f"[{current_v}][{current_a}][{in_v}][{in_a}]concat=n=2:v=1:a=1[{out_v}][{out_a}]")
                else:
                    parts.append(f"[{current_v}][{in_v}]concat=n=2:v=1:a=0[{out_v}]")
                current_duration = current_duration + dur_next
            else:
                offset = max(0.0, current_duration - d)
                if transition_id == "t_scale_punch":
                    z0 = 1.18
                    z_amp = max(0.01, z0 - 1.0)
                    z_expr = f"max(1.0\,{z0:.4f} - ({z_amp:.4f})*(t-{offset:.4f})/{d:.4f})"
                    mid_fx = (
                        f"eq=contrast=1.06:brightness=0.02:saturation=1.08,"
                        f"tmix=frames=5:weights='1 0.88 0.68 0.48 0.28':enable='between(t,{offset:.3f},{offset + d:.3f})',"
                        f"scale=trunc({width}*({z_expr})/2)*2:trunc({height}*({z_expr})/2)*2:eval=frame,"
                        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,format=yuv420p"
                    )
                    parts.append(f"[{current_v}][{in_v}]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    parts.append(f"[xv{join_idx}]{mid_fx}[{out_v}]")
                elif transition_id == "t_smooth_zoom":
                    z0 = 1.08
                    z_amp = max(0.01, z0 - 1.0)
                    z_expr = f"max(1.0\,{z0:.4f} - ({z_amp:.4f})*(t-{offset:.4f})/{d:.4f})"
                    mid_fx = (
                        f"eq=contrast=1.03:brightness=0.01:saturation=1.04,"
                        f"scale=trunc({width}*({z_expr})/2)*2:trunc({height}*({z_expr})/2)*2:eval=frame,"
                        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,format=yuv420p"
                    )
                    parts.append(f"[{current_v}][{in_v}]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    parts.append(f"[xv{join_idx}]{mid_fx}[{out_v}]")
                elif transition_id == "t_zoom_out":
                    z1 = 1.12
                    z_expr = f"max(1.0\\,{z1:.4f} - ({z1 - 1.0:.4f})*(t-{offset:.4f})/{d:.4f})"
                    parts.append(f"[{current_v}][{in_v}]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    parts.append(
                        f"[xv{join_idx}]scale=trunc({width}*({z_expr})/2)*2:trunc({height}*({z_expr})/2)*2:eval=frame,"
                        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,format=yuv420p[{out_v}]"
                    )
                elif transition_id == "t_zoom_in":
                    z1 = 1.12
                    z_expr = (
                        f"if(lt(t\,{offset:.4f})\,1.0\,"
                        f"if(gt(t\,{offset + d:.4f})\,{z1:.4f}\,"
                        f"1.0 + ({z1 - 1.0:.4f})*(t-{offset:.4f})/{d:.4f}))"
                    )
                    parts.append(f"[{current_v}][{in_v}]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    parts.append(
                        f"[xv{join_idx}]scale=trunc({width}*({z_expr})/2)*2:trunc({height}*({z_expr})/2)*2:eval=frame,"
                        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,format=yuv420p[{out_v}]"
                    )
                elif transition_id == "t_curtain_open":
                    expr = "if(lte(abs(X-W/2),(W*P/2)),B,A)"
                    parts.append(
                        f"[{current_v}][{in_v}]xfade=transition=custom:expr='{expr}':duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]"
                    )
                    parts.append(f"[xv{join_idx}]format=yuv420p[{out_v}]")
                elif transition_id == "t_mirror_slide":
                    parts.append(f"[{current_v}]split=2[msl{join_idx}a][msl{join_idx}b]")
                    parts.append(f"[msl{join_idx}b]hflip[msl{join_idx}bf]")
                    parts.append(f"[msl{join_idx}a][msl{join_idx}bf]blend=all_expr='if(lte(X,W/2),A,B)'[msa{join_idx}]")
                    parts.append(f"[{in_v}]split=2[msn{join_idx}a][msn{join_idx}b]")
                    parts.append(f"[msn{join_idx}b]hflip[msn{join_idx}bf]")
                    parts.append(f"[msn{join_idx}a][msn{join_idx}bf]blend=all_expr='if(lte(X,W/2),A,B)'[msb{join_idx}]")
                    parts.append(f"[msa{join_idx}][msb{join_idx}]xfade=transition=slideleft:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    parts.append(f"[xv{join_idx}]format=yuv420p[{out_v}]")
                else:
                    mode = self._transition_mode_to_xfade(transition_id)
                    parts.append(f"[{current_v}][{in_v}]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f}[xv{join_idx}]")
                    if transition_id == "t_radial_burst":
                        post_fx = f"eq=brightness=0.04:contrast=1.06:enable='between(t,{offset:.3f},{offset + min(d, 0.14):.3f})',format=yuv420p"
                    elif transition_id == "t_shimmer_blur":
                        post_fx = f"eq=saturation=1.08:contrast=1.03:enable='between(t,{offset:.3f},{offset + d:.3f})',format=yuv420p"
                    elif transition_id == "t_wind":
                        post_fx = f"tmix=frames=4:weights='1 0.70 0.45 0.22':enable='between(t,{offset:.3f},{offset + d:.3f})',format=yuv420p"
                    elif transition_id == "t_slitscan_push":
                        post_fx = f"tmix=frames=5:weights='1 0.86 0.62 0.38 0.18':enable='between(t,{offset:.3f},{offset + d:.3f})',format=yuv420p"
                    else:
                        post_fx = "format=yuv420p"
                    parts.append(f"[xv{join_idx}]{post_fx}[{out_v}]")

                if with_audio:
                    parts.append(f"[{current_a}][{in_a}]acrossfade=d={d:.3f}:curve1=tri:curve2=tri[{out_a}]")
                current_duration = current_duration + dur_next - d

            current_v = out_v
            current_a = out_a if with_audio else None

        return ";".join(parts), current_v, current_a

    def run(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="fv_transition_video_")
        try:
            self.progress.emit(1)
            metas = [_probe_media(item["path"]) for item in self.cfg.items]
            if not metas:
                raise RuntimeError("No clips loaded.")
            for meta in metas:
                if meta.get("duration", 0.0) <= 0.0:
                    raise RuntimeError(f"Could not read clip duration:\n{meta.get('path')}")
            fps = self._resolve_render_fps(metas)
            bitrate_kbps = self._resolve_render_bitrate_kbps(metas)
            width, height = _target_resolution(self.cfg.resolution_mode, metas)
            self._log(f"Render FPS: {fps:.3f}")
            self._log(f"Render bitrate: {bitrate_kbps} kbps")
            self._log(f"Render size: {width}x{height}")
            self._log(f"Sound in output: {'no' if self.cfg.mute_output else 'yes'}")

            normalized_paths: List[str] = []
            prep_total = max(1, len(metas))
            for idx, meta in enumerate(metas):
                self._log(f"Preparing clip {idx + 1}/{len(metas)}: {os.path.basename(meta['path'])}")
                out_path = os.path.join(temp_dir, f"norm_{idx:04d}.mp4")
                self._prepare_clip(meta, out_path, fps, width, height, bitrate_kbps)
                normalized_paths.append(out_path)
                self.progress.emit(5 + int(((idx + 1) / prep_total) * 35))

            output_folder = self.cfg.output_folder or _output_dir()
            _ensure_dir(output_folder)
            stem = _sanitize_stem(self.cfg.output_name or "transition_video")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(output_folder, f"{stem}_{stamp}.mp4")

            if len(normalized_paths) == 1:
                shutil.copy2(normalized_paths[0], final_path)
            else:
                self._log("Building single-pass ffmpeg timeline...")
                filter_complex, final_v, final_a = self._build_single_pass_filter(normalized_paths, not self.cfg.mute_output, width, height)
                cmd = [self.ffmpeg, "-y"]
                for p in normalized_paths:
                    cmd += ["-i", p]
                cmd += [
                    "-filter_complex", filter_complex,
                    "-map", f"[{final_v}]",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-b:v", f"{int(bitrate_kbps)}k",
                ]
                if self.cfg.mute_output:
                    cmd += ["-an"]
                else:
                    cmd += [
                        "-map", f"[{final_a}]",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-ar", "48000",
                        "-ac", "2",
                    ]
                cmd += [final_path]
                self.progress.emit(45)
                code, out = _run_process(cmd)
                if code != 0 or not os.path.exists(final_path):
                    raise RuntimeError(f"Single-pass render failed.\n\n{out}")

            self.progress.emit(100)
            self.done.emit(True, final_path)
        except Exception as e:
            self.done.emit(False, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TransitionVideoTool(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._loading_settings = True
        self._worker: Optional[TransitionVideoRenderWorker] = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings_now)
        self._build_ui()
        self._load_settings()
        self._loading_settings = False
        self._save_settings_later()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        body = QWidget(scroll)
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        title = QLabel("Transition Video", body)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        lay.addWidget(title)

        desc = QLabel(
            "Load a folder or list of clips, pick transitions, then save one stitched video.",
            body,
        )
        desc.setWordWrap(True)
        lay.addWidget(desc)

        # Clip buttons
        row_top = QHBoxLayout()
        self.btn_add_files = QPushButton("Add video files", body)
        self.btn_add_folder = QPushButton("Add folder", body)
        self.btn_remove = QPushButton("Remove selected", body)
        self.btn_clear = QPushButton("Clear list", body)
        self.btn_up = QPushButton("Move up", body)
        self.btn_down = QPushButton("Move down", body)
        self.btn_randomize_list = QPushButton("Randomize list", body)
        self.btn_sort = QPushButton("Sort", body)
        row_top.addWidget(self.btn_add_files)
        row_top.addWidget(self.btn_add_folder)
        row_top.addWidget(self.btn_remove)
        row_top.addWidget(self.btn_clear)
        row_top.addWidget(self.btn_up)
        row_top.addWidget(self.btn_down)
        row_top.addWidget(self.btn_randomize_list)
        row_top.addWidget(self.btn_sort)
        row_top.addStretch(1)
        lay.addLayout(row_top)

        # Clip list
        self.list_clips = QListWidget(body)
        self.list_clips.setViewMode(QListWidget.IconMode)
        self.list_clips.setIconSize(QSize(240, 135))
        self.list_clips.setResizeMode(QListWidget.Adjust)
        self.list_clips.setMovement(QListWidget.Static)
        self.list_clips.setWordWrap(True)
        self.list_clips.setSpacing(12)
        self.list_clips.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_clips.setMinimumHeight(290)
        lay.addWidget(self.list_clips)

        # Per-clip transition controls
        box_tr = QFrame(body)
        box_tr.setFrameShape(QFrame.StyledPanel)
        tr_lay = QVBoxLayout(box_tr)

        row_sel = QHBoxLayout()
        self.combo_transition = QComboBox(box_tr)
        for tid, label in TRANSITIONS:
            self.combo_transition.addItem(label, tid)
        self.btn_apply_selected = QPushButton("Apply to selected", box_tr)
        self.btn_apply_all = QPushButton("Apply to all clips", box_tr)
        row_sel.addWidget(QLabel("Selected clip transition:", box_tr))
        row_sel.addWidget(self.combo_transition, 1)
        row_sel.addWidget(self.btn_apply_selected)
        row_sel.addWidget(self.btn_apply_all)
        tr_lay.addLayout(row_sel)

        row_toggle = QHBoxLayout()
        self.check_randomize = QCheckBox("Randomize transitions", box_tr)
        self.check_one_for_all = QCheckBox("Use 1 transition for all clips", box_tr)
        self.combo_global_transition = QComboBox(box_tr)
        for tid, label in TRANSITIONS:
            self.combo_global_transition.addItem(label, tid)
        self.combo_global_transition.setCurrentIndex(0)
        row_toggle.addWidget(self.check_randomize)
        row_toggle.addWidget(self.check_one_for_all)
        row_toggle.addWidget(QLabel("Global transition:", box_tr))
        row_toggle.addWidget(self.combo_global_transition)
        row_toggle.addStretch(1)
        tr_lay.addLayout(row_toggle)

        self.box_random_pool = QFrame(box_tr)
        self.box_random_pool.setFrameShape(QFrame.StyledPanel)
        random_pool_lay = QVBoxLayout(self.box_random_pool)
        random_pool_lay.setContentsMargins(8, 8, 8, 8)

        row_pool_top = QHBoxLayout()
        row_pool_top.addWidget(QLabel("Randomizer uses only these transitions:", self.box_random_pool))
        self.btn_random_all_on = QPushButton("All on", self.box_random_pool)
        self.btn_random_all_off = QPushButton("All off", self.box_random_pool)
        row_pool_top.addStretch(1)
        row_pool_top.addWidget(self.btn_random_all_on)
        row_pool_top.addWidget(self.btn_random_all_off)
        random_pool_lay.addLayout(row_pool_top)

        self.random_transition_checks = {}
        grid_pool = QGridLayout()
        grid_pool.setHorizontalSpacing(12)
        grid_pool.setVerticalSpacing(4)
        for idx, (tid, label) in enumerate(TRANSITIONS):
            check = QCheckBox(label, self.box_random_pool)
            check.setChecked(True)
            check.toggled.connect(self._save_settings_later)
            self.random_transition_checks[tid] = check
            grid_pool.addWidget(check, idx // 3, idx % 3)
        random_pool_lay.addLayout(grid_pool)
        tr_lay.addWidget(self.box_random_pool)

        row_duration = QHBoxLayout()
        self.spin_transition_sec = QDoubleSpinBox(box_tr)
        self.spin_transition_sec.setDecimals(2)
        self.spin_transition_sec.setSingleStep(0.25)
        self.spin_transition_sec.setRange(0.50, 10.00)
        self.spin_transition_sec.setValue(1.00)
        row_duration.addWidget(QLabel("Transition duration (seconds):", box_tr))
        row_duration.addWidget(self.spin_transition_sec)
        row_duration.addStretch(1)
        tr_lay.addLayout(row_duration)

        lay.addWidget(box_tr)

        # Export / save
        box_save = QFrame(body)
        box_save.setFrameShape(QFrame.StyledPanel)
        save_lay = QFormLayout(box_save)

        self.combo_fps = QComboBox(box_save)
        self.combo_fps.addItem("Keep source (fallback 30)", "keep")
        self.combo_fps.addItem("30 fps", "30")

        self.combo_resolution = QComboBox(box_save)
        self.combo_resolution.addItem("Use source (fallback 720p)", "keep")
        self.combo_resolution.addItem("480p", "480")
        self.combo_resolution.addItem("720p", "720")
        self.combo_resolution.addItem("1080p", "1080")
        self.combo_resolution.setCurrentIndex(0)

        self.combo_quality = QComboBox(box_save)
        self.combo_quality.addItem("3500 kbps", "3500")
        self.combo_quality.addItem("5000 kbps", "5000")
        self.combo_quality.addItem("Keep source (fallback 3500)", "keep")

        self.check_mute = QCheckBox("Disable sound while saving", box_save)
        self.edit_output_name = QLineEdit("transition_video", box_save)
        self.edit_output_folder = QLineEdit(_output_dir(), box_save)
        self.btn_browse_output = QPushButton("Browse", box_save)

        row_out = QHBoxLayout()
        row_out.addWidget(self.edit_output_folder, 1)
        row_out.addWidget(self.btn_browse_output)

        save_lay.addRow("FPS:", self.combo_fps)
        save_lay.addRow("Resolution:", self.combo_resolution)
        save_lay.addRow("Quality:", self.combo_quality)
        save_lay.addRow("Output name:", self.edit_output_name)
        save_lay.addRow("Output folder:", row_out)
        save_lay.addRow("", self.check_mute)
        lay.addWidget(box_save)

        row_bottom = QHBoxLayout()
        self.btn_render = QPushButton("Create transition video", body)
        self.progress = QProgressBar(body)
        self.progress.setRange(0, 100)
        row_bottom.addWidget(self.btn_render)
        row_bottom.addWidget(self.progress, 1)
        lay.addLayout(row_bottom)

        self.logs = QTextEdit(body)
        self.logs.setReadOnly(True)
        self.logs.setMinimumHeight(180)
        lay.addWidget(self.logs)

        lay.addStretch(1)

        # Signals
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_list)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        self.btn_randomize_list.clicked.connect(self._randomize_list)
        self.btn_sort.clicked.connect(self._show_sort_menu)

        self.list_clips.itemDoubleClicked.connect(self._play_item)
        self.list_clips.currentItemChanged.connect(self._sync_transition_combo_from_current)
        self.list_clips.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_clips.customContextMenuRequested.connect(self._show_list_menu)

        self.btn_apply_selected.clicked.connect(self._apply_transition_to_selected)
        self.btn_apply_all.clicked.connect(self._apply_transition_to_all)
        self.check_randomize.toggled.connect(self._on_transition_mode_changed)
        self.check_one_for_all.toggled.connect(self._on_transition_mode_changed)
        self.combo_global_transition.currentIndexChanged.connect(self._save_settings_later)
        self.btn_random_all_on.clicked.connect(lambda: self._set_all_random_transitions(True))
        self.btn_random_all_off.clicked.connect(lambda: self._set_all_random_transitions(False))

        self.spin_transition_sec.valueChanged.connect(self._save_settings_later)
        self.combo_fps.currentIndexChanged.connect(self._save_settings_later)
        self.combo_resolution.currentIndexChanged.connect(self._save_settings_later)
        self.combo_quality.currentIndexChanged.connect(self._save_settings_later)
        self.check_mute.toggled.connect(self._save_settings_later)
        self.edit_output_name.textChanged.connect(self._save_settings_later)
        self.edit_output_folder.textChanged.connect(self._save_settings_later)
        self.btn_browse_output.clicked.connect(self._browse_output_folder)
        self.btn_render.clicked.connect(self._start_render)

        self._on_transition_mode_changed()

    def _log(self, text: str) -> None:
        self.logs.append(str(text))

    def _save_settings_later(self, *args) -> None:
        if self._loading_settings:
            return
        self._save_timer.start(250)

    def _save_settings_now(self) -> None:
        data = {
            "output_folder": self.edit_output_folder.text().strip(),
            "output_name": self.edit_output_name.text().strip(),
            "randomize_transitions": bool(self.check_randomize.isChecked()),
            "randomize_transition_ids": self._collect_randomizer_transition_ids(),
            "one_transition_for_all": bool(self.check_one_for_all.isChecked()),
            "global_transition_id": str(self.combo_global_transition.currentData() or "hardcut"),
            "transition_seconds": float(self.spin_transition_sec.value()),
            "fps_mode": str(self.combo_fps.currentData() or "keep"),
            "resolution_mode": str(self.combo_resolution.currentData() or "720"),
            "quality_mode": str(self.combo_quality.currentData() or "3500"),
            "mute_output": bool(self.check_mute.isChecked()),
            "clips": self._collect_items(),
        }
        try:
            _atomic_write_json(_settings_path(), data)
        except Exception:
            pass

    def _load_settings(self) -> None:
        data = _read_json(_settings_path())

        self.edit_output_folder.setText(str(data.get("output_folder") or _output_dir()))
        self.edit_output_name.setText(str(data.get("output_name") or "transition_video"))

        self._set_combo_by_data(self.combo_fps, str(data.get("fps_mode") or "keep"))
        self._set_combo_by_data(self.combo_resolution, str(data.get("resolution_mode") or "720"))
        self._set_combo_by_data(self.combo_quality, str(data.get("quality_mode") or "3500"))
        self.check_randomize.setChecked(bool(data.get("randomize_transitions", False)))
        self.check_one_for_all.setChecked(bool(data.get("one_transition_for_all", False)))
        self.check_mute.setChecked(bool(data.get("mute_output", False)))
        self.spin_transition_sec.setValue(float(data.get("transition_seconds") or 1.0))
        self._set_combo_by_data(self.combo_global_transition, str(data.get("global_transition_id") or "hardcut"))
        self._load_randomizer_transition_ids(data.get("randomize_transition_ids"))

        clips = data.get("clips") or []
        if isinstance(clips, list):
            for item in clips:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                transition_id = str(item.get("transition_id") or "hardcut")
                if path and os.path.isfile(path):
                    self._add_clip_item(path, transition_id=transition_id, save=False)

        self._on_transition_mode_changed()

    def _set_combo_by_data(self, combo: QComboBox, wanted: str) -> None:
        for i in range(combo.count()):
            if str(combo.itemData(i)) == str(wanted):
                combo.setCurrentIndex(i)
                return

    def _collect_items(self) -> List[dict]:
        out: List[dict] = []
        for i in range(self.list_clips.count()):
            item = self.list_clips.item(i)
            data = item.data(Qt.UserRole) or {}
            path = str(data.get("path") or "")
            transition_id = str(data.get("transition_id") or "hardcut")
            if path:
                out.append({"path": path, "transition_id": transition_id})
        return out

    def _collect_randomizer_transition_ids(self) -> List[str]:
        out: List[str] = []
        checks = getattr(self, "random_transition_checks", {}) or {}
        for tid, _label in TRANSITIONS:
            check = checks.get(tid)
            if check is not None and check.isChecked():
                out.append(str(tid))
        return out

    def _load_randomizer_transition_ids(self, ids) -> None:
        checks = getattr(self, "random_transition_checks", {}) or {}
        valid_ids = {str(tid) for tid, _label in TRANSITIONS}
        chosen = set()
        if isinstance(ids, (list, tuple, set)):
            chosen = {str(x) for x in ids if str(x) in valid_ids}
        if not chosen:
            chosen = set(valid_ids)
        for tid, _label in TRANSITIONS:
            check = checks.get(tid)
            if check is not None:
                check.setChecked(str(tid) in chosen)

    def _set_all_random_transitions(self, enabled: bool) -> None:
        checks = getattr(self, "random_transition_checks", {}) or {}
        for check in checks.values():
            check.setChecked(bool(enabled))
        self._save_settings_later()


    def _find_existing_paths(self) -> set:
        return {str(item.get("path") or "") for item in self._collect_items()}

    def _add_clip_item(self, path: str, transition_id: str = "hardcut", save: bool = True) -> None:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTS:
            return

        data = {"path": path, "transition_id": transition_id}
        item = QListWidgetItem(_make_item_text(path, transition_id), self.list_clips)
        item.setData(Qt.UserRole, data)

        thumb = _generate_thumbnail(path)
        if thumb and os.path.exists(thumb):
            item.setIcon(QIcon(thumb))
        else:
            pix = QPixmap(240, 135)
            pix.fill(Qt.black)
            item.setIcon(QIcon(pix))
        item.setToolTip(path)
        self.list_clips.addItem(item)
        if save:
            self._save_settings_later()

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add video files",
            _app_root(),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg *.m4v);;All files (*.*)",
        )
        if not files:
            return
        existing = self._find_existing_paths()
        added = 0
        for path in files:
            if os.path.abspath(path) in existing:
                continue
            self._add_clip_item(path, transition_id="hardcut", save=False)
            added += 1
        self._save_settings_later()
        if added:
            self._log(f"Added {added} clip(s).")

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder with video files", _app_root())
        if not folder:
            return
        existing = self._find_existing_paths()
        added = 0
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
                continue
            if os.path.abspath(path) in existing:
                continue
            self._add_clip_item(path, transition_id="hardcut", save=False)
            added += 1
        self._save_settings_later()
        if added:
            self._log(f"Added {added} clip(s) from folder.")

    def _selected_items(self) -> List[QListWidgetItem]:
        return list(self.list_clips.selectedItems() or [])

    def _remove_selected(self) -> None:
        rows = sorted((self.list_clips.row(it) for it in self._selected_items()), reverse=True)
        for row in rows:
            self.list_clips.takeItem(row)
        if rows:
            self._save_settings_later()

    def _clear_list(self) -> None:
        self.list_clips.clear()
        self._save_settings_later()

    def _move_up(self) -> None:
        row = self.list_clips.currentRow()
        if row <= 0:
            return
        item = self.list_clips.takeItem(row)
        self.list_clips.insertItem(row - 1, item)
        self.list_clips.setCurrentRow(row - 1)
        self._save_settings_later()

    def _move_down(self) -> None:
        row = self.list_clips.currentRow()
        if row < 0 or row >= self.list_clips.count() - 1:
            return
        item = self.list_clips.takeItem(row)
        self.list_clips.insertItem(row + 1, item)
        self.list_clips.setCurrentRow(row + 1)
        self._save_settings_later()

    def _randomize_list(self) -> None:
        count = self.list_clips.count()
        if count < 2:
            return
        items = [self.list_clips.takeItem(0) for _ in range(count)]
        random.shuffle(items)
        for item in items:
            self.list_clips.addItem(item)
        self.list_clips.setCurrentRow(0)
        self._save_settings_later()
        self._log("Clip order randomized.")

    def _show_sort_menu(self) -> None:
        menu = QMenu(self)
        options = [
            ("Name ↑", "name", False),
            ("Name ↓", "name", True),
            ("Date ↑", "date", False),
            ("Date ↓", "date", True),
            ("Size ↑", "size", False),
            ("Size ↓", "size", True),
        ]
        for label, mode, reverse in options:
            act = QAction(label, menu)
            act.triggered.connect(lambda checked=False, m=mode, r=reverse: self._sort_list(m, r))
            menu.addAction(act)
        menu.exec(self.btn_sort.mapToGlobal(self.btn_sort.rect().bottomLeft()))

    def _sort_key_for_item(self, item: QListWidgetItem, mode: str):
        path = str(self._item_data(item).get("path") or "")
        try:
            st = os.stat(path)
        except Exception:
            st = None
        name = os.path.basename(path).lower()
        if mode == "date":
            return (st.st_mtime if st else 0.0, name)
        if mode == "size":
            return (st.st_size if st else -1, name)
        return (name,)

    def _sort_list(self, mode: str, reverse: bool = False) -> None:
        count = self.list_clips.count()
        if count < 2:
            return
        items = [self.list_clips.takeItem(0) for _ in range(count)]
        items.sort(key=lambda it: self._sort_key_for_item(it, mode), reverse=bool(reverse))
        for item in items:
            self.list_clips.addItem(item)
        self.list_clips.setCurrentRow(0)
        self._save_settings_later()
        label = {"name": "name", "date": "date", "size": "size"}.get(mode, mode)
        direction = "descending" if reverse else "ascending"
        self._log(f"Clip order sorted by {label} ({direction}).")

    def _item_data(self, item: Optional[QListWidgetItem]) -> dict:
        if item is None:
            return {}
        data = item.data(Qt.UserRole)
        return data if isinstance(data, dict) else {}

    def _set_item_data(self, item: QListWidgetItem, data: dict) -> None:
        item.setData(Qt.UserRole, data)
        item.setText(_make_item_text(str(data.get("path") or ""), str(data.get("transition_id") or "hardcut")))
        item.setToolTip(str(data.get("path") or ""))

    def _play_item(self, item: Optional[QListWidgetItem]) -> None:
        if item is None:
            return
        path = str(self._item_data(item).get("path") or "")
        if not path or not os.path.exists(path):
            return
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception:
                pass

    def _show_list_menu(self, pos: QPoint) -> None:
        item = self.list_clips.itemAt(pos)
        menu = QMenu(self)

        if item is not None:
            act_play = QAction("Play clip", menu)
            act_play.triggered.connect(lambda: self._play_item(item))
            menu.addAction(act_play)

            menu.addSeparator()
            sub = menu.addMenu("Set transition")
            for tid, label in TRANSITIONS:
                act = QAction(label, sub)
                act.triggered.connect(lambda checked=False, transition_id=tid, target=item: self._set_transition_for_items([target], transition_id))
                sub.addAction(act)

            menu.addSeparator()
            act_remove = QAction("Remove clip", menu)
            act_remove.triggered.connect(lambda: self._remove_specific_item(item))
            menu.addAction(act_remove)
        else:
            act_add = QAction("Add video files", menu)
            act_add.triggered.connect(self._add_files)
            menu.addAction(act_add)

            act_folder = QAction("Add folder", menu)
            act_folder.triggered.connect(self._add_folder)
            menu.addAction(act_folder)

        menu.exec(self.list_clips.viewport().mapToGlobal(pos))

    def _remove_specific_item(self, item: QListWidgetItem) -> None:
        row = self.list_clips.row(item)
        if row >= 0:
            self.list_clips.takeItem(row)
            self._save_settings_later()

    def _sync_transition_combo_from_current(self, current=None, previous=None) -> None:
        item = current if current is not None else self.list_clips.currentItem()
        if item is None:
            return
        transition_id = str(self._item_data(item).get("transition_id") or "hardcut")
        self._set_combo_by_data(self.combo_transition, transition_id)

    def _set_transition_for_items(self, items: List[QListWidgetItem], transition_id: str) -> None:
        changed = 0
        for item in items:
            data = self._item_data(item)
            data["transition_id"] = transition_id
            self._set_item_data(item, data)
            changed += 1
        if changed:
            self._save_settings_later()

    def _apply_transition_to_selected(self) -> None:
        items = self._selected_items()
        if not items:
            cur = self.list_clips.currentItem()
            if cur is not None:
                items = [cur]
        if not items:
            return
        transition_id = str(self.combo_transition.currentData() or "hardcut")
        self._set_transition_for_items(items, transition_id)

    def _apply_transition_to_all(self) -> None:
        transition_id = str(self.combo_transition.currentData() or "hardcut")
        items = [self.list_clips.item(i) for i in range(self.list_clips.count())]
        self._set_transition_for_items(items, transition_id)

    def _on_transition_mode_changed(self) -> None:
        randomize = self.check_randomize.isChecked()
        one_for_all = self.check_one_for_all.isChecked()
        self.combo_transition.setEnabled(not (randomize or one_for_all))
        self.btn_apply_selected.setEnabled(not (randomize or one_for_all))
        self.btn_apply_all.setEnabled(not (randomize or one_for_all))
        self.combo_global_transition.setEnabled(one_for_all)
        self.box_random_pool.setEnabled(randomize)
        self.box_random_pool.setVisible(randomize)
        self._save_settings_later()

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.edit_output_folder.text().strip() or _output_dir())
        if folder:
            self.edit_output_folder.setText(folder)
            self._save_settings_later()

    def _validate_before_render(self) -> Optional[str]:
        if self.list_clips.count() < 1:
            return "Load at least one video clip first."
        if self.list_clips.count() > 1 and self.spin_transition_sec.value() <= 0.0:
            return "Transition duration must be above zero."
        ffmpeg = _ffmpeg_path()
        ffprobe = _ffprobe_path()
        if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
            return f"ffmpeg not found:\n{ffmpeg}"
        if not (os.path.exists(ffprobe) or shutil.which(ffprobe)):
            return f"ffprobe not found:\n{ffprobe}"
        return None

    def _start_render(self) -> None:
        err = self._validate_before_render()
        if err:
            QMessageBox.warning(self, "Transition Video", err)
            return

        cfg = RenderConfig(
            items=self._collect_items(),
            randomize_transitions=bool(self.check_randomize.isChecked()),
            randomize_transition_ids=self._collect_randomizer_transition_ids(),
            one_transition_for_all=bool(self.check_one_for_all.isChecked()),
            global_transition_id=str(self.combo_global_transition.currentData() or "hardcut"),
            transition_seconds=float(self.spin_transition_sec.value()),
            mute_output=bool(self.check_mute.isChecked()),
            fps_mode=str(self.combo_fps.currentData() or "keep"),
            resolution_mode=str(self.combo_resolution.currentData() or "720"),
            quality_mode=str(self.combo_quality.currentData() or "3500"),
            output_name=str(self.edit_output_name.text().strip() or "transition_video"),
            output_folder=str(self.edit_output_folder.text().strip() or _output_dir()),
        )

        self.progress.setValue(0)
        self.btn_render.setEnabled(False)
        self.logs.clear()
        self._log("Starting render...")

        self._worker = TransitionVideoRenderWorker(cfg, self)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.done.connect(self._render_finished)
        self._worker.start()

        self._save_settings_later()

    def _render_finished(self, ok: bool, payload: str) -> None:
        self.btn_render.setEnabled(True)
        if ok:
            self.progress.setValue(100)
            self._log(f"Saved:\n{payload}")
            QMessageBox.information(self, "Transition Video", f"Saved:\n{payload}")
        else:
            self._log(f"ERROR:\n{payload}")
            QMessageBox.warning(self, "Transition Video", str(payload))


def install_transition_video_tool(parent, section_or_layout):
    """
    Best-effort installer for FrameVision style sections/layouts.
    """
    tool = TransitionVideoTool(parent)
    target = section_or_layout

    # Common case: a collapsible section with a layout() or content layout.
    try:
        if hasattr(target, "content_layout"):
            lay = getattr(target, "content_layout")
            if lay is not None and hasattr(lay, "addWidget"):
                lay.addWidget(tool)
                return tool
    except Exception:
        pass

    try:
        if hasattr(target, "layout") and callable(target.layout):
            lay = target.layout()
            if lay is not None and hasattr(lay, "addWidget"):
                lay.addWidget(tool)
                return tool
    except Exception:
        pass

    try:
        if hasattr(target, "addWidget"):
            target.addWidget(tool)
            return tool
    except Exception:
        pass

    # Last resort: if it is a QWidget without a layout, give it one.
    try:
        lay = QVBoxLayout(target)
        lay.addWidget(tool)
    except Exception:
        pass
    return tool


if __name__ == "__main__":
    # Tiny standalone launcher for quick testing.
    try:
        from PySide6.QtWidgets import QApplication
        import sys

        app = QApplication(sys.argv)
        w = TransitionVideoTool()
        w.resize(1280, 900)
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        raise SystemExit(str(e))
