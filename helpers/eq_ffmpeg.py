
import os, sys, subprocess, shlex, time
from pathlib import Path

# Global singleton per process (simple)
_proc = None

def _guess_ffplay_path():
    # Try project_root/presets/bin/ffplay.exe
    try:
        here = Path(__file__).resolve()
        root = here
        for _ in range(5):
            if (root / "presets" / "bin" / "ffplay.exe").exists():
                return str(root / "presets" / "bin" / "ffplay.exe")
            root = root.parent
    except Exception:
        pass
    # Fall back to PATH
    return "ffplay"

def _pane_media_and_pos(pane):
    # Try common patterns to find current media path and position (ms)
    # file path
    for attr in ("media_path", "current_path", "video_path", "filepath", "source_path"):
        p = getattr(pane, attr, None)
        if isinstance(p, str) and p:
            path = p
            break
    else:
        # nested player with current media source
        path = None
        for name in ("player", "media_player", "video_player"):
            pl = getattr(pane, name, None)
            if pl is None: continue
            # QMediaPlayer API: source() -> QUrl
            try:
                url = pl.source()
                if hasattr(url, "toLocalFile"):
                    lf = url.toLocalFile()
                    if lf: path = lf; break
            except Exception:
                pass
    # position
    pos_ms = None
    for attr in ("position_ms", "pos_ms", "current_ms", "position"):
        v = getattr(pane, attr, None)
        if isinstance(v, int) and v >= 0:
            pos_ms = v; break
    if pos_ms is None:
        for name in ("player", "media_player", "video_player"):
            pl = getattr(pane, name, None)
            if pl is None: continue
            try:
                pos_ms = int(pl.position())
                break
            except Exception:
                pass
    return path, (pos_ms or 0)

def stop():
    global _proc
    if _proc and _proc.poll() is None:
        try:
            _proc.terminate()
        except Exception:
            pass
        try:
            _proc.kill()
        except Exception:
            pass
    _proc = None

def apply_filter(pane, filter_str):
    """Mute Qt audio and start/replace an ffplay sidecar to play audio with -af filter."""
    global _proc
    path, pos_ms = _pane_media_and_pos(pane)
    if not path or not os.path.exists(path):
        return False

    # Mute Qt audio if present
    for name in ("audio", "audio_output", "audioOutput", "player_audio"):
        a = getattr(pane, name, None)
        try:
            if a and hasattr(a, "setMuted"):
                a.setMuted(True)
        except Exception:
            pass

    # Build command
    ffplay = _guess_ffplay_path()
    # -nodisp: audio only; -autoexit at end; -ss seek; -af filter; -loglevel warning
    args = [ffplay, "-nodisp", "-autoexit", "-loglevel", "warning"]
    if pos_ms and pos_ms > 0:
        args += ["-ss", f"{pos_ms/1000.0:.3f}"]
    args += ["-af", filter_str, path]

    # Stop existing
    stop()

    try:
        _proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        _proc = None
        return False


def unmute_qt(pane):
    for name in ("audio", "audio_output", "audioOutput", "player_audio"):
        a = getattr(pane, name, None)
        try:
            if a and hasattr(a, "setMuted"):
                a.setMuted(False)
        except Exception:
            pass
