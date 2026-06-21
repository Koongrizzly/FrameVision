
# Media info and popup helpers for FrameVision
import json, subprocess, os, typing
from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QPushButton, QApplication, QMessageBox
from PySide6.QtCore import Qt, QObject, QTimer

IMAGE_EXTS = {'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff','.gif'}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.aif', '.aiff'}

def ffprobe_path():
    try:
        base = Path(".").resolve()
        cand = [base / "bin" / ("ffprobe.exe" if os.name == "nt" else "ffprobe"), "ffprobe"]
        for c in cand:
            try:
                subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
                return str(c)
            except Exception:
                continue
    except Exception:
        pass
    return "ffprobe"

def ffprobe_json(path: Path):
    try:
        out = subprocess.check_output([
            ffprobe_path(), "-v", "error",
            "-show_format", "-show_streams",
            "-print_format", "json",
            str(path)
        ], stderr=subprocess.STDOUT, universal_newlines=True)
        return json.loads(out)
    except Exception:
        return {}

def _num(v, default=None):
    try:
        if v is None: return default
        return float(v)
    except Exception:
        return default

def _to_kbps(bit_rate_val):
    try:
        br = float(bit_rate_val)
        return int(round(br/1000.0))
    except Exception:
        try:
            return int(round(float(str(bit_rate_val).strip())/1000.0))
        except Exception:
            return None

def _fps_from(fr):
    try:
        if isinstance(fr, (int, float)): return float(fr)
        if isinstance(fr, str) and "/" in fr:
            n, d = fr.split("/")
            d = float(d) if float(d) != 0 else 1.0
            return round(float(n)/d, 3)
        if fr:
            return float(fr)
    except Exception:
        pass
    return None

def probe_media_all(path: Path):
    info = {
        "file": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "kind": None,
        "duration": None,
        "bit_rate_kbps": None,
        "format": None,
        "video": {},
        "audio": {},
        "tags": {}
    }
    j = ffprobe_json(path)
    fmt = j.get("format") or {}
    streams = j.get("streams") or []
    info["format"] = fmt.get("format_long_name") or fmt.get("format_name")
    info["duration"] = _num(fmt.get("duration"))
    info["bit_rate_kbps"] = _to_kbps(fmt.get("bit_rate"))
    try:
        for k, v in (fmt.get("tags") or {}).items():
            info["tags"][k] = str(v)
    except Exception:
        pass
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v_stream:
        info["kind"] = "video"
        w = v_stream.get("width"); h = v_stream.get("height")
        fps = _fps_from(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate"))
        # Total frames (best-effort): prefer ffprobe's nb_frames when available.
        frames = None
        try:
            nf = v_stream.get("nb_frames")
            if nf is None:
                nf = v_stream.get("nb_read_frames")
            if nf is not None and str(nf).strip() != "N/A":
                frames = int(str(nf).strip())
        except Exception:
            frames = None
        if frames is None:
            try:
                if info.get("duration") and fps:
                    frames = int(round(float(info["duration"]) * float(fps)))
            except Exception:
                frames = None
        info["video"] = {
            "codec": v_stream.get("codec_name"),
            "width": w,
            "height": h,
            "fps": fps,
            "frames": frames,
            "pix_fmt": v_stream.get("pix_fmt"),
            "profile": v_stream.get("profile"),
        }
        if info["duration"] in (None, 0) and (fps in (None, 0)):
            if path.suffix.lower() in IMAGE_EXTS:
                info["kind"] = "image"
    elif a_stream:
        info["kind"] = "audio"
    else:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            info["kind"] = "image"
        elif ext in AUDIO_EXTS:
            info["kind"] = "audio"
        else:
            info["kind"] = "other"
    if a_stream:
        info["audio"] = {
            "codec": a_stream.get("codec_name"),
            "sample_rate_hz": _num(a_stream.get("sample_rate")),
            "channels": a_stream.get("channels"),
            "channel_layout": a_stream.get("channel_layout"),
            "bit_rate_kbps": _to_kbps(a_stream.get("bit_rate"))
        }
    return info

def _fmt_hms(seconds):
    try:
        if seconds is None: return "—"
        s = int(round(float(seconds)))
        h = s // 3600; m = (s % 3600) // 60; sec = s % 60
        if h: return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"
    except Exception:
        return "—"

def _human_size(bytes_):
    try:
        b = float(bytes_)
        for unit in ["B","KB","MB","GB","TB"]:
            if b < 1024.0 or unit == "TB":
                return f"{b:.1f} {unit}"
            b /= 1024.0
    except Exception:
        pass
    return "—"

def build_info_text(info: dict) -> str:
    lines = []
    lines.append(f"File       : {info.get('name','—')}")
    lines.append(f"Path       : {info.get('file','—')}")
    if info.get("format"):
        lines.append(f"Format     : {info.get('format')}")
    lines.append(f"Type       : {info.get('kind','—')}")
    if info.get("size_bytes") is not None:
        lines.append(f"Size       : {_human_size(info['size_bytes'])}")
    if info.get("duration") is not None:
        lines.append(f"Duration   : {_fmt_hms(info['duration'])}")
    if info.get('bit_rate_kbps'):
        lines.append(f"Bitrate    : {info['bit_rate_kbps']} kbps")
    v = info.get("video") or {}
    if v:
        res = (f"{v.get('width','?')}x{v.get('height','?')}" if (v.get('width') and v.get('height')) else '—')
        fps = v.get('fps')
        lines.append(f"Video      : {v.get('codec','—')} | {res} | {(str(fps)+' fps') if fps else 'fps —'}")
        fr = v.get('frames')
        if fr is not None:
            lines.append(f"Frames     : {fr}")
    a = info.get("audio") or {}
    if a:
        sr = a.get('sample_rate_hz'); ch = a.get('channels')
        br = a.get('bit_rate_kbps')
        lines.append(f"Audio      : {a.get('codec','—')} | {int(sr) if sr else '—'} Hz | {ch if ch else '—'} ch | {br if br else '—'} kbps")
    tags = info.get("tags") or {}
    if tags:
        lines.append("\nTags:")
        for k in sorted(tags.keys()):
            v = str(tags[k])
            if len(v) > 500:
                v = v[:500] + '…'
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
# -----------------------
# Auto-updating dialog
# -----------------------

_ACTIVE_DIALOG: 'MediaInfoDialog|None' = None

class MediaInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Media Info")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        self._txt = QTextEdit(self); self._txt.setReadOnly(True)
        layout.addWidget(self._txt)

        bb = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        btn_copy = QPushButton("Copy to clipboard", self)
        layout.addWidget(bb)
        layout.addWidget(btn_copy)

        def _copy():
            try:
                QApplication.clipboard().setText(self._txt.toPlainText())
            except Exception:
                pass
        btn_copy.clicked.connect(_copy)
        bb.rejected.connect(self.close)

        self.resize(640, 480)

        # Auto-refresh while open (no main-file hooks needed)
        self._last_path = None
        self._poll = QTimer(self)
        self._poll.setInterval(400)  # ms
        self._poll.timeout.connect(self._auto_refresh_tick)
        self._poll.start()

    def _auto_refresh_tick(self):
        try:
            # Search self, parents, and top-level widgets for a player/path
            obj_chain = []
            try:
                obj = self.parent()
                for _ in range(6):
                    if obj is None:
                        break
                    obj_chain.append(obj)
                    obj = obj.parent()
            except Exception:
                pass
            try:
                from PySide6.QtWidgets import QApplication
                for w in QApplication.topLevelWidgets():
                    obj_chain.append(w)
            except Exception:
                pass
            path = None
            for o in obj_chain:
                path = _extract_path_any(o)
                if path:
                    break
            if path is None:
                return
            if self._last_path is None or str(path) != str(self._last_path):
                self._last_path = path
                info = probe_media_all(Path(str(path)))
                self.set_info(info)
        except Exception:
            pass

    def set_info(self, info: dict):
        try:
            self._txt.setPlainText(build_info_text(info))
        except Exception:
            pass

def show_info_popup(parent, info: dict):
    """
    Open (or update) the Media Info window.
    If the window is already open, it will be updated in-place rather than spawning a new one.
    """
    global _ACTIVE_DIALOG
    try:
        if _ACTIVE_DIALOG is not None and _ACTIVE_DIALOG.isVisible():
            _ACTIVE_DIALOG.set_info(info)
            _ACTIVE_DIALOG.raise_()
            _ACTIVE_DIALOG.activateWindow()
            return
    except Exception:
        _ACTIVE_DIALOG = None  # stale reference; recreate below

    try:
        dlg = MediaInfoDialog(parent)
        dlg.set_info(info)
        _ACTIVE_DIALOG = dlg
        dlg.show()
    except Exception:
        # Fallback to a simple message box if the dialog cannot be created
        try:
            QMessageBox.information(parent, "Media Info", build_info_text(info))
        except Exception:
            pass

def update_info_popup(info: dict):
    """
    Update the already-open Media Info window (if any). Safe to call from your
    'current media changed' handler in the main app.
    """
    try:
        if _ACTIVE_DIALOG is not None and _ACTIVE_DIALOG.isVisible():
            _ACTIVE_DIALOG.set_info(info)
    except Exception:
        pass


# -----------------------
# Zero-risk auto-wire helpers (no main-file edits required)
# -----------------------
def _try_connect_signal(obj, signal_name: str, slot):
    try:
        sig = getattr(obj, signal_name, None)
        if sig is None:
            return False
        sig.connect(slot)
        return True
    except Exception:
        return False

def _wrap_method(obj, name: str, wrapper_attr: str):
    try:
        if hasattr(obj, wrapper_attr):
            return True
        orig = getattr(obj, name, None)
        if not callable(orig):
            return False
        def patched(*a, **kw):
            try:
                res = orig(*a, **kw)
            finally:
                try:
                    _safe_refresh_from(obj)
                except Exception:
                    pass
            return res
        setattr(obj, wrapper_attr, orig)
        setattr(obj, name, patched)
        return True
    except Exception:
        return False

def wire_auto_update(player_like) -> bool:
    hooked = False
    for sig in ("mediaChanged", "currentMediaChanged", "fileOpened", "mediaOpened"):
        hooked |= _try_connect_signal(player_like, sig, lambda *_: _safe_refresh_from(player_like))
    hooked |= _wrap_method(player_like, "open_file", "_mi_wrap_open_file")
    hooked |= _wrap_method(player_like, "open_path", "_mi_wrap_open_path")
    hooked |= _wrap_method(player_like, "load", "_mi_wrap_load")
    hooked |= _wrap_method(player_like, "setMedia", "_mi_wrap_setMedia")
    return bool(hooked)

def unwire_auto_update(player_like) -> int:
    count = 0
    for name, attr in (("open_file","_mi_wrap_open_file"),
                       ("open_path","_mi_wrap_open_path"),
                       ("load","_mi_wrap_load"),
                       ("setMedia","_mi_wrap_setMedia")):
        try:
            if hasattr(player_like, attr):
                orig = getattr(player_like, attr)
                setattr(player_like, name, orig)
                delattr(player_like, attr)
                count += 1
        except Exception:
            pass
    return count


def _extract_path_any(obj):
    """Return a Path if we can deduce a media path from this object, else None."""
    try:
        # Try getters/attributes
        for getter in ("current_path", "currentPath", "currentFile", "current_file",
                       "sourcePath", "mediaPath", "filePath", "currentUrl", "current_url",
                       "source", "currentMedia", "media"):
            try:
                if hasattr(obj, getter):
                    val = getattr(obj, getter)
                    if callable(val):
                        val = val()
                    try:
                        if hasattr(val, 'toLocalFile'):
                            val = val.toLocalFile()
                        elif hasattr(val, 'canonicalUrl'):
                            val = val.canonicalUrl()
                        elif hasattr(val, 'url'):
                            val = val.url()
                    except Exception:
                        pass
                    if val:
                        return Path(str(val))
            except Exception:
                pass
        # Try nested .player
        try:
            pl = getattr(obj, "player", None)
            if pl is not None:
                try:
                    src = getattr(pl, "source", lambda: None)()
                    if src is not None:
                        try:
                            val = src.toLocalFile()
                        except Exception:
                            val = getattr(src, "canonicalUrl", lambda: None)() or getattr(src, "url", lambda: None)()
                        if val:
                            return Path(str(val))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    return None
def _safe_refresh_from(player_like):
    try:
        # Search the object, then parents
        seen = set()
        cur = player_like
        for _ in range(6):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            pth = _extract_path_any(cur)
            if pth:
                info = probe_media_all(Path(str(pth)))
                update_info_popup(info)
                return
            try:
                cur = cur.parent()
            except Exception:
                cur = None
    except Exception:
        pass


def refresh_info_now(obj_or_path):
    """Force refresh the Info window from a player-like object or a file path."""
    try:
        if isinstance(obj_or_path, (str, Path)):
            info = probe_media_all(Path(str(obj_or_path)))
            update_info_popup(info)
            return True
        _safe_refresh_from(obj_or_path)
        return True
    except Exception:
        return False


# -----------------------
# Player signal hooks (no main-file edits)
# -----------------------

_PLAYER_HOOKED = False
_LAST_HOOK_COUNT = 0

def _connect_signal(obj, signal_name: str, slot) -> bool:
    try:
        sig = getattr(obj, signal_name, None)
        if sig is None:
            return False
        # Avoid duplicate connections by storing a marker
        marker = f"_mi_sig_{signal_name}_connected"
        if getattr(obj, marker, False):
            return True
        sig.connect(slot)
        setattr(obj, marker, True)
        return True
    except Exception:
        return False

def _iter_qobjects_bfs(root_obj, max_depth=4):
    try:
        from collections import deque
        dq = deque([(root_obj, 0)])
        seen = set([id(root_obj)])
        while dq:
            obj, d = dq.popleft()
            yield obj
            if d >= max_depth:
                continue
            try:
                for ch in obj.findChildren(QObject):
                    if id(ch) in seen:
                        continue
                    seen.add(id(ch))
                    dq.append((ch, d+1))
            except Exception:
                pass
    except Exception:
        yield root_obj

def _install_player_hooks():
    """
    Walk the Qt object tree and connect to common media player signals so the Info
    window refreshes whenever *any* player changes media or state.
    """
    global _PLAYER_HOOKED, _LAST_HOOK_COUNT
    count = 0
    try:
        from PySide6.QtWidgets import QApplication
        widgets = list(QApplication.topLevelWidgets())
    except Exception:
        widgets = []

    targets = []
    for w in widgets:
        for obj in _iter_qobjects_bfs(w, max_depth=5):
            # Duck-typing a "player": it should have any of these methods/signals
            has_source = hasattr(obj, "source") or hasattr(obj, "currentMedia") or hasattr(obj, "media")
            has_sig = any(hasattr(obj, nm) for nm in (
                "sourceChanged", "mediaStatusChanged", "currentMediaChanged",
                "mediaChanged", "stateChanged", "positionChanged"
            ))
            if has_source or has_sig:
                targets.append(obj)

    def on_any_change(*_):
        try:
            sender = None
            try:
                from PySide6.QtCore import QObject as _QO
                sender = _QO.sender()
            except Exception:
                pass
            _safe_refresh_from(sender or targets[0] if targets else None)
        except Exception:
            pass

    for obj in targets:
        # Hook a handful of useful signals if present
        for nm in ("sourceChanged", "mediaStatusChanged", "currentMediaChanged",
                   "mediaChanged", "stateChanged", "positionChanged"):
            if _connect_signal(obj, nm, on_any_change):
                count += 1

    _LAST_HOOK_COUNT = count
    _PLAYER_HOOKED = True
    return count
