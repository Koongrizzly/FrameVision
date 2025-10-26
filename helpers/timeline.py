"""
OpenShot timeline backend integration (skeleton) — v2
- Looks for libopenshot*.dll and openshot.pyd next to the app in ./presets/bin on Windows.
- If openshot.pyd is not installed in site-packages, you can drop it into ./presets/bin.
"""
from __future__ import annotations

import os, sys, platform
from pathlib import Path
from typing import Optional, Any, List

_OPENSHOT_IMPORTED = False
_OPENSHOT = None

def _proj_root() -> Path:
    # helpers/ is one level below project root
    return Path(__file__).resolve().parent.parent

def _dll_dir_candidates() -> list[Path]:
    pr = _proj_root()
    return [
        pr / "presets" / "bin",
        Path.cwd() / "presets" / "bin",
    ]

def _add_path_once(p: Path):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

def _add_dll_directory_win(path: Path) -> None:
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))
    except Exception:
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")

def _find_openshot_pyd() -> Optional[Path]:
    # Common location we support: ./presets/bin/openshot.pyd
    for c in _dll_dir_candidates():
        cand = c / "openshot.pyd"
        if cand.exists():
            return cand
    return None

def ensure_openshot_loaded() -> None:
    """Attempt to load OpenShot libs and import the 'openshot' Python module (.pyd)."""
    global _OPENSHOT_IMPORTED, _OPENSHOT
    if _OPENSHOT_IMPORTED:
        return

    if os.name == "nt":
        for c in _dll_dir_candidates():
            if (c / "libopenshot.dll").exists():
                _add_dll_directory_win(c)
        # If a local openshot.pyd is present, add its folder to sys.path so 'import openshot' works
        pyd = _find_openshot_pyd()
        if pyd:
            _add_path_once(pyd.parent)

    # Try import now
    try:
        import openshot as _os
    except Exception as e:
        raise ImportError(
            "Could not import 'openshot' Python bindings. "
            "You need a matching 'openshot.pyd' for your Python version/arch, plus libopenshot DLLs. "
            "Place 'openshot.pyd', 'libopenshot.dll', and 'libopenshot-audio.dll' in ./presets/bin or install the bindings in your venv. "
            f"Original error: {e!r}"
        )
    _OPENSHOT = _os
    _OPENSHOT_IMPORTED = True

class OpenShotTimeline:
    def __init__(self, width:int=1280, height:int=720, fps:float=30.0, sample_rate:int=48000, channels:int=2):
        ensure_openshot_loaded()
        osmod = _OPENSHOT
        if abs(fps - 29.97) < 0.01:
            self.fps_num, self.fps_den = 30000, 1001
        else:
            self.fps_num, self.fps_den = int(round(fps)), 1
        self.width = int(width); self.height = int(height)
        self.sample_rate = int(sample_rate); self.channels = int(channels)
        self.timeline = osmod.Timeline(self.width, self.height, self.fps_num, self.fps_den, self.sample_rate, self.channels)
        self._os = osmod

    def add_clip(self, path: str | os.PathLike, layer:int=0, start_time:float=0.0, end_time:Optional[float]=None) -> Any:
        osmod = self._os
        reader = osmod.FFmpegReader(str(path))
        clip = osmod.Clip(reader)
        clip.Layer = int(layer)
        clip.Position = float(start_time)
        if end_time is not None:
            try:
                clip.End = float(end_time)
            except Exception:
                pass
        self.timeline.AddClip(clip)
        return clip

    def remove_clip(self, clip: Any) -> None:
        try:
            self.timeline.RemoveClip(clip)
        except Exception:
            try:
                clip.IsEnabled = False
            except Exception:
                pass

    def duration_seconds(self) -> float:
        try:
            return float(self.timeline.Duration())
        except Exception:
            return 0.0

    def export(self, output_path: str, video_codec: str="libx264", audio_codec: str="aac", crf:int=18, preset:str="medium"):
        osmod = self._os
        writer = osmod.FFmpegWriter(str(output_path))
        writer.SetVideoOptions(True, self.width, self.height, self.fps_num, self.fps_den, video_codec, crf, preset)
        writer.SetAudioOptions(True, self.sample_rate, self.channels, audio_codec, 192000)
        writer.Open()
        try:
            total_frames = int(self.duration_seconds() * (self.fps_num / self.fps_den)) + 1
            for n in range(total_frames):
                frame = self.timeline.GetFrame(n)
                writer.WriteFrame(frame)
        finally:
            writer.Close()

def _exists(p: Path) -> str:
    return "✓" if p.exists() else "✗"

def verify_environment(verbose: bool=False) -> str:
    """Return a diagnostic string explaining what's present/missing for OpenShot bindings."""
    pr = _proj_root()
    lines: List[str] = []
    lines.append(f"Python {platform.python_version()} ({platform.architecture()[0]}, {platform.python_implementation()})")
    lines.append(f"Executable: {sys.executable}")
    bin_dirs = _dll_dir_candidates()
    for b in bin_dirs:
        lines.append(f"Check {b}: "
                     f"openshot.pyd[{_exists(b/'openshot.pyd')}] "
                     f"libopenshot.dll[{_exists(b/'libopenshot.dll')}] "
                     f"libopenshot-audio.dll[{_exists(b/'libopenshot-audio.dll')}]")
    try:
        ensure_openshot_loaded()
        lines.append("Import 'openshot' -> OK")
    except Exception as e:
        lines.append(f"Import 'openshot' -> ERROR: {e}")
    if verbose:
        lines.append("sys.path (first 3):")
        for p in sys.path[:3]:
            lines.append(f" - {p}")
    return "\\n".join(lines)
