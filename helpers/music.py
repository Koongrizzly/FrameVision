from __future__ import annotations

import os
import sys
import json
import math
import struct
import subprocess
import importlib.util
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from PySide6.QtCore import Qt, QTimer, QRect, QRectF, QSize, Signal, QObject, QEvent, QThread, QPropertyAnimation
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget
)

# Optional Qt multimedia probe
try:
    from PySide6.QtMultimedia import QAudioProbe, QMediaPlayer
    HAVE_AUDIO_PROBE = False
except Exception:
    QAudioProbe = None  # type: ignore
    QMediaPlayer = None  # type: ignore
    HAVE_AUDIO_PROBE = False

# ---------------- optional deps ----------------

def _try_numpy():
    try:
        import numpy as np
        return np
    except Exception:
        return None

_np = _try_numpy()

def _try_mutagen():
    try:
        from mutagen import File as MF
        return MF
    except Exception:
        return None

_MF = _try_mutagen()

ROOT = Path('.').resolve()
OUT_TEMP = ROOT / 'output' / '_temp'
OUT_TEMP.mkdir(parents=True, exist_ok=True)
MUSIC_STATE_PATH = OUT_TEMP / 'music_state.json'


# Playlists save/load directory
PLAYLIST_DIR = ROOT / 'presets' / 'setsave' / 'playlist'
try:
    PLAYLIST_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.aif', '.aiff'}

# ---------------- initial one-time scroll hack ----------------
# Workaround: on app start, the first audio track will seek forward by 100ms
# right after starting at 00:00. This mitigates a freeze when loading another
# track/video before any user scroll occurs.
_INITIAL_SCROLL_ARMED = False
_INITIAL_SCROLL_DONE = False


# ---------------- ffmpeg helpers ----------------

def ffmpeg_path() -> str:
    for c in [ROOT / 'bin' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'), 'ffmpeg']:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT, text=True)
            return str(c)
        except Exception:
            pass
    return 'ffmpeg'

def ffprobe_path() -> str:
    for c in [ROOT / 'bin' / ('ffprobe.exe' if os.name == 'nt' else 'ffprobe'), 'ffprobe']:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT, text=True)
            return str(c)
        except Exception:
            pass
    return 'ffprobe'


def _probe_duration_ms(path: Path) -> int:
    """Return duration in ms using ffprobe (0 if unknown)."""
    try:
        out = subprocess.check_output(
            [ffprobe_path(), '-v', 'error', '-select_streams', 'a', '-show_entries', 'format=duration', '-of', 'json', str(path)],
            text=True
        )
        data = json.loads(out or "{}")
        dur = float(data.get('format', {}).get('duration', 0.0))
        if dur and dur > 0:
            return int(dur * 1000)
    except Exception:
        pass
    return 0


def _ffprobe_full(path: Path) -> dict:
    """Return ffprobe json for format and streams."""
    try:
        out = subprocess.check_output(
            [ffprobe_path(), '-hide_banner', '-loglevel', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
            text=True
        )
        return json.loads(out or "{}")
    except Exception:
        return {}

def _make_human_info_from_ffprobe(path: Path) -> list[str]:
    data = _ffprobe_full(path)
    fmt = data.get('format', {}) if isinstance(data, dict) else {}
    streams = data.get('streams', []) or []
    tags = {}
    try:
        t = fmt.get('tags', {}) or {}
        if isinstance(t, dict):
            tags.update(t)
        for st in streams:
            if isinstance(st, dict):
                tt = st.get('tags', {}) or {}
                if isinstance(tt, dict):
                    for k, v in tt.items():
                        tags.setdefault(k, v)
    except Exception:
        pass

    def g(*keys):
        for k in keys:
            v = tags.get(k)
            if v:
                return v
        return None

    title = g('title', 'TIT2')
    artist = g('artist', 'TPE1', 'Contributing artists')
    album = g('album', 'TALB')
    album_artist = g('album_artist', 'TPE2', 'Album artist')
    year = g('date', 'year', 'TYER', 'TDRC')
    genre = g('genre', 'TCON')
    track = g('track', 'TRCK', 'tracknumber', 'TRACKNUMBER')
    disc = g('disc', 'TPOS', 'DISCNUMBER')
    bpm = g('TBPM', 'bpm', 'BPM')
    comment = g('comment', 'COMM::XXX', 'COMM')
    publisher = g('publisher', 'TPUB')
    encoded_by = g('encoded_by', 'encoder', 'TSSE', 'Encoded by')
    composer = g('composer', 'TCOM')

    bit_rate = None
    sr = None
    ch = None
    codec = None
    for st in streams:
        if st.get('codec_type') == 'audio':
            bit_rate = bit_rate or st.get('bit_rate') or fmt.get('bit_rate')
            sr = sr or st.get('sample_rate')
            ch = ch or st.get('channels')
            codec = codec or st.get('codec_name')

    dur_ms = _probe_duration_ms(path)
    if not dur_ms:
        try:
            d = float(fmt.get('duration', 0.0))
            if d > 0:
                dur_ms = int(d * 1000)
        except Exception:
            pass

    lines = []
    lines.append(f"File: {path.name}")
    lines.append(f"Path: {path}")
    try:
        size = path.stat().st_size
        units = ['B','KB','MB','GB','TB']
        v, i = float(size), 0
        while v >= 1024 and i < len(units)-1:
            v /= 1024.0
            i += 1
        size_str = f"{int(v)} {units[i]}" if i == 0 else f"{v:.2f} {units[i]}"
        lines.append(f"Size: {size_str}")
    except Exception:
        pass
    if dur_ms:
        sec = dur_ms/1000.0
        mm = int(sec//60); ss = int(sec%60)
        lines.append(f"Duration: {mm:02d}:{ss:02d} ({dur_ms} ms)")
    if bit_rate:
        try:
            br_k = int(int(bit_rate)/1000)
            lines.append(f"Bit rate: {br_k} kbps")
        except Exception:
            lines.append(f"Bit rate: {bit_rate}")
    if sr:
        lines.append(f"Sample rate: {sr} Hz")
    if ch:
        lines.append(f"Channels: {ch}")
    if codec:
        lines.append(f"Codec: {codec}")
    if fmt.get('format_name'):
        lines.append(f"Format: {fmt.get('format_name')}")

    tag_lines = []
    def add(name, val):
        if val:
            tag_lines.append(f"{name}: {val}")
    add('Title', title)
    add('Artist', artist)
    add('Album', album)
    add('Album artist', album_artist)
    add('Year', year)
    add('#', track)
    add('Disc', disc)
    add('Genre', genre)
    add('BPM', bpm)
    add('Composer', composer)
    add('Publisher', publisher)
    add('Encoded by', encoded_by)
    add('Comment', comment)

    if tag_lines:
        lines.append('')
        lines.append('Tags:')
        lines.extend(tag_lines)

    known = {'title','TIT2','artist','TPE1','album','TALB','album_artist','TPE2','date','year','TYER','TDRC','genre','TCON','track','TRCK','tracknumber','TRACKNUMBER','disc','TPOS','DISCNUMBER','TBPM','bpm','BPM','comment','COMM::XXX','COMM','publisher','TPUB','encoded_by','encoder','TSSE','composer','TCOM'}
    extras = []
    for k, v in sorted(tags.items()):
        if k not in known and str(v).strip():
            extras.append(f"{k}: {v}")
    if extras:
        lines.append('')
        lines.append('Other tags:')
        lines.extend(extras[:30])
    return lines
# ---------------- tag + cover helpers ----------------

def _read_tags_with_mutagen(path: Path) -> dict:
    if _MF is None:
        return {}
    try:
        m = _MF(str(path))
        if not m or not getattr(m, 'tags', None):
            return {}
        out = {}

        def getv(*keys):
            for k in keys:
                try:
                    v = m.tags.get(k)
                    if hasattr(v, 'text'):
                        v = v.text[0]
                    if isinstance(v, (list, tuple)):
                        v = v[0] if v else None
                    if v:
                        return str(v)
                except Exception:
                    pass
            return None

        out['title'] = getv('TIT2', 'title')
        out['artist'] = getv('TPE1', 'artist')
        out['album'] = getv('TALB', 'album')
        out['TBPM'] = getv('TBPM', 'bpm', 'BPM')

        # cover (APIC)
        try:
            for k in list(m.tags.keys()):
                if str(k).startswith('APIC'):
                    pic = m.tags.get(k)
                    data = getattr(pic, 'data', None)
                    if data:
                        img = QImage.fromData(data)
                        if img and not img.isNull():
                            out['_cover_qimage'] = img
                            break
        except Exception:
            pass

        return out
    except Exception:
        return {}

def _read_tags_with_ffprobe(path: Path) -> dict:
    try:
        out = subprocess.check_output(
            [ffprobe_path(), '-hide_banner', '-loglevel', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
            text=True
        )
        data = json.loads(out)
        tags = {}
        fmt = data.get('format', {}).get('tags', {}) if isinstance(data, dict) else {}
        if isinstance(fmt, dict):
            tags.update(fmt)
        for st in data.get('streams', []) or []:
            if isinstance(st, dict):
                tt = st.get('tags', {})
                if isinstance(tt, dict):
                    for k, v in tt.items():
                        tags.setdefault(k, v)
        return tags
    except Exception:
        return {}

def _extract_cover(path: Path) -> Optional[QPixmap]:
    # try mutagen first
    mt = _read_tags_with_mutagen(path)
    qi = mt.get('_cover_qimage')
    if qi is not None:
        return QPixmap.fromImage(qi)
    # ffprobe/ffmpeg attached pic
    try:
        out = subprocess.check_output(
            [ffprobe_path(), '-hide_banner', '-loglevel', 'error', '-select_streams', 'v', '-show_entries', 'stream=disposition,codec_type', '-of', 'json', str(path)],
            text=True
        )
        data = json.loads(out)
        ok = False
        for st in data.get('streams', []) or []:
            disp = st.get('disposition', {})
            if isinstance(disp, dict) and int(disp.get('attached_pic', 0)) == 1:
                ok = True
                break
        if ok:
            target = OUT_TEMP / f'{path.stem}_cover.jpg'
            subprocess.check_output([ffmpeg_path(), '-hide_banner', '-nostats', '-loglevel', 'error', '-y', '-i', str(path), '-an', '-vcodec', 'copy', str(target)],
                                    stderr=subprocess.STDOUT, text=True)
            if target.exists() and target.stat().st_size > 0:
                return QPixmap(str(target))
    except Exception:
        pass
    return None

def _read_all_tags(path: Path) -> dict:
    a = _read_tags_with_ffprobe(path)
    b = _read_tags_with_mutagen(path)
    a.update(b or {})
    return a

# ---------------- Visual plugin API ----------------

class BaseVisualizer:
    """Subclass this and implement paint(painter, rect, bands:list[float], rms:float, t:seconds)."""
    display_name = 'Custom Visual'

    def paint(self, p: QPainter, r: QRectF, bands, rms, t: float):
        pass

_VISUAL_REGISTRY: List[Tuple[str, type]] = []

def register_visualizer(cls):
    name = getattr(cls, 'display_name', None) or cls.__name__
    _VISUAL_REGISTRY.append((name, cls))
    return cls

def _load_visual_plugins():
    base = ROOT / 'presets' / 'viz'
    if not base.exists():
        return
    sys.path.insert(0, str(ROOT))
    for py in base.glob('*.py'):
        mod_name = f'fv_viz_{py.stem}'
        if mod_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = m
                spec.loader.exec_module(m)
        except Exception:
            continue

def _norm_name(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

# Hide built-ins Bars/Pulse and two plugins by display name (case/space-insensitive)
_BLOCKED = {
    _norm_name('bars'),
    _norm_name('pulse'),
    _norm_name('Reaction Diffusion'),
    _norm_name('Audio reactive fields'),
}

# ---------------- Analysis ----------------

class PreAnalyzer(QThread):
    """Background decoder + FFT to precompute bands and RMS aligned to time.
       Emits ready(times_ms, bands_frames, rms_frames).
    """
    ready = Signal(list, list, list)

    def __init__(self, path: Path, sr: int = 24000, bands: int = 48, hop_ms: int = 70, parent=None):
        self._ema_ref = None  # per-bin EMA reference for AGC
        self._stop_requested = False
        self._proc = None
        
        super().__init__(parent)
        self.path = Path(path)
        self.sr = int(sr)
        self.bands = int(bands)
        self.hop_ms = int(hop_ms)
        # Optional segment window (ms) for JIT mode
        self.start_ms = None
        self.dur_ms = None


    def request_stop(self):
        self._stop_requested = True
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass


    def _viz_rebuild_bag(self):
        """(Re)build the visual shuffle bag excluding the current visual,
        so we won't repeat until all visuals have been shown once."""
        if not self.overlay:
            self._viz_bag = []
            self._viz_bag_pos = -1
            return
        try:
            count = len(self.overlay._visual_modes)
            cur = self.overlay.cmb_visual.currentIndex()
            idxs = [i for i in range(count) if i != cur]
            import random
            random.shuffle(idxs)
            self._viz_bag = idxs
            self._viz_bag_pos = -1
        except Exception:
            # be safe; clear bag if something goes wrong
            self._viz_bag = []
            self._viz_bag_pos = -1

    def _viz_next_random_index(self) -> int:
        """Return next visual index from the shuffle bag; rebuild if needed."""
        if not self.overlay:
            return -1
        count = 0
        try:
            count = len(self.overlay._visual_modes)
        except Exception:
            pass
        if count <= 1:
            try:
                return self.overlay.cmb_visual.currentIndex()
            except Exception:
                return -1
        # Rebuild if bag empty or contains invalid indices
        if not isinstance(self._viz_bag, list) or not self._viz_bag or any((i < 0 or i >= count) for i in self._viz_bag):
            self._viz_rebuild_bag()
        # Advance
        self._viz_bag_pos += 1
        if self._viz_bag_pos >= len(self._viz_bag):
            # Completed a full pass; rebuild excluding current to avoid immediate repeat
            self._viz_rebuild_bag()
            self._viz_bag_pos = 0 if self._viz_bag else -1
        if 0 <= self._viz_bag_pos < len(self._viz_bag):
            return self._viz_bag[self._viz_bag_pos]
        try:
            return self.overlay.cmb_visual.currentIndex()
        except Exception:
            return -1

    def run(self):
        ff = ffmpeg_path()
        cmd = [ff, '-hide_banner', '-nostats', '-loglevel', 'error']
        if (getattr(self, 'start_ms', None) is not None) and (getattr(self, 'dur_ms', None) is not None):
            preroll = 200
            ss = max(0, int(self.start_ms) - preroll)
            tt = int(self.dur_ms) + preroll
            cmd += ['-ss', f'{ss/1000:.3f}', '-t', f'{tt/1000:.3f}']
        cmd += ['-i', str(self.path), '-vn', '-ac', '1', '-ar', str(self.sr), '-f', 'f32le', 'pipe:1']
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p = self._proc
        except Exception:
            self.ready.emit([], [], [])
            return

        hop = max(1, int(self.sr * self.hop_ms / 1000))
        # Use a larger FFT for better bass resolution
        n_fft = 1
        while n_fft < max(hop * 2, 2048):
            n_fft *= 2

        buf = b''
        times_ms: List[int] = []
        bands_mat: List[List[float]] = []
        rms_list: List[float] = []
        read = p.stdout.read  # type: ignore[attr-defined]
        idx = 0
        emitted_initial = False
        last_emit_ms = 0
        INITIAL_MS = 900  # first partial emit ~2.5s
        PERIODIC_MS = 1000  # then every ~2s

        freqs = None
        edges = None
        centers = None

        while True:
            if getattr(self, '_stop_requested', False):
                break
            chunk = read(8192)
            if not chunk:
                break
            buf += chunk

            while len(buf) >= max(hop, n_fft) * 4:
                if getattr(self, '_stop_requested', False):
                    break
                if len(buf) < n_fft * 4:
                    break
                window = buf[:n_fft * 4]
                buf = buf[hop * 4:]

                try:
                    if _np is None:
                        raise RuntimeError('numpy required for high-quality analysis')
                    import numpy as np
                    arr = np.frombuffer(window, dtype='<f4', count=n_fft)
                    win = np.hanning(arr.size)
                    spec_c = np.fft.rfft(arr * win, n=n_fft)
                    mag = np.abs(spec_c)
                    if mag.max() > 0:
                        # Per-bin EMA AGC + soft companding (replaces frame-wise max normalization)
                        if self._ema_ref is None:
                            self._ema_ref = mag.copy() + 1e-9
                        else:
                            self._ema_ref = 0.985 * self._ema_ref + 0.125 * mag
                        mag = np.clip(mag / (self._ema_ref + 1e-9), 0.0, 1.0)
                        # Soft companding keeps low-level motion visible without letting peaks dominate
                        mag = np.power(mag, 0.78)
# RMS over hop-sized slice
                    hop_samples = hop
                    td_chunk = arr[:hop_samples] if hop_samples <= arr.size else arr
                    rms = float(np.sqrt(np.mean(td_chunk**2))) if td_chunk.size else 0.0

                    if freqs is None:
                        freqs = np.linspace(0, self.sr / 2.0, mag.size)
                        lo_start = 20.0
                        edges = np.geomspace(lo_start, self.sr / 2.0, num=self.bands + 1)
                        centers = np.sqrt(edges[:-1] * edges[1:])

                    out = []
                    for i in range(self.bands):
                        lo = float(edges[i]); hi = float(edges[i + 1]); c = float(centers[i])
                        mask = (freqs >= lo) & (freqs <= hi)
                        if not mask.any():
                            val = float(np.interp(c, freqs, mag))
                            out.append(val)
                            continue
                        fsel = freqs[mask]; msel = mag[mask]
                        w = np.where(fsel <= c,
                                      (fsel - lo) / max(1e-9, (c - lo)),
                                      (hi - fsel) / max(1e-9, (hi - c)))
                        w = np.clip(w, 0.0, 1.0)
                        sw = np.sum(w)
                        val = float(np.sum(msel * w) / sw) if sw > 1e-9 else float(msel.mean())
                        out.append(val)

                except Exception:
                    # Fallback if numpy path fails
                    n = n_fft
                    try:
                        vals = struct.unpack('<' + 'f' * n, window[:n * 4])
                    except Exception:
                        self.ready.emit([], [], [])
                        try:
                            (self._proc or p).kill() if (self._proc or p) else None
                        except Exception:
                            pass
                        return
                    step = max(1, n // (self.bands * 2))
                    out = []
                    sm = 0.0
                    for i in range(self.bands):
                        seg = vals[i * step:(i + 1) * step]
                        mvv = 0.0
                        for x in seg:
                            mvv = max(mvv, abs(x))
                            sm += x*x
                        out.append(min(1.0, mvv * 2.0))
                    rms = (sm / max(1, n)) ** 0.5

                bands_mat.append(out)
                rms_list.append(min(1.0, rms * 2.2))
                t_ms = (int(self.start_ms) if getattr(self, 'start_ms', None) is not None else 0) + idx * self.hop_ms
                times_ms.append(t_ms)
                idx += 1
                # Incremental partial results
                try:
                    if (not emitted_initial) and (t_ms >= INITIAL_MS):
                        self.ready.emit(list(times_ms), list(bands_mat), list(rms_list))
                        emitted_initial = True
                        last_emit_ms = t_ms
                    elif emitted_initial and (t_ms - last_emit_ms) >= PERIODIC_MS:
                        self.ready.emit(list(times_ms), list(bands_mat), list(rms_list))
                        last_emit_ms = t_ms
                except Exception:
                    pass

        try:
            (self._proc or p).kill() if (self._proc or p) else None
        except Exception:
            pass
        self.ready.emit(times_ms, bands_mat, rms_list)

class HybridAnalyzer(QObject):
    """Position-synced from preanalysis; falls back to live probe; else time-based motion."""
    levelsReady = Signal(list)  # bands
    rmsReady = Signal(float)

    def __init__(self, player, bands: int = 48, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.player = player
        self.bands = bands
        self._times: List[int] = []
        self._bands: List[List[float]] = []
        self._rms: List[float] = []
        self._hop_ms = 80
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._seg_cache = {}
        self._seg_window_ms = 12000  # keep ~12s around last seeks
        try:
            if hasattr(self.player, 'positionChanged'):
                self.player.positionChanged.connect(self._on_pos_changed)
        except Exception:
            pass
        self._probe_ok = False
        self._probe_levels: Optional[List[float]] = None
        self._probe_rms: float = 0.0

        # Lighten QAudioProbe processing: process 1 of every 4 buffers
        self._probe_decim = 0
        self._probe_decim_mod = 4

        if HAVE_AUDIO_PROBE and player is not None:
            try:
                self._probe = QAudioProbe(self)  # type: ignore[call-arg]
                self._probe.audioBufferProbed.connect(self._on_buf)
                self._probe.setSource(player)
                self._probe_ok = True
            except Exception:
                self._probe_ok = False
        else:
            self._probe = None

        # --- JIT small-window analysis (no full preanalysis) ---
        self._win_ms = 8000
        self._preroll_ms = 800
        self._seg_cache = {}
        self._seg_worker = None
        self._seg_inflight = False
        self._win_start = 0
        self._win_end = 0
        try:
            if hasattr(self.player, 'positionChanged'):
                self.player.positionChanged.connect(self._on_pos_update_window)
        except Exception:
            pass

    def set_file(self, path: Path):
        # Stop any previous worker first
        try:
            if hasattr(self, '_worker') and getattr(self, '_worker', None):
                try:
                    self._worker.request_stop()
                except Exception:
                    pass
                try:
                    self._worker.wait(500)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            LONG_TRACK_MS = 20 * 60 * 1000
            dur_ms = 0
            try:
                dur_ms = _probe_duration_ms(Path(path))
            except Exception:
                dur_ms = 0
            if dur_ms >= LONG_TRACK_MS:
                # Skip preanalysis for long tracks; rely on live probe to avoid background CPU
                self._worker = None
                self._times, self._bands, self._rms = [], [], []
            else:
                self._worker = PreAnalyzer(path, bands=self.bands, hop_ms=self._hop_ms, parent=self)
                try:
                    self._worker.ready.connect(self._on_ready, getattr(Qt, 'UniqueConnection', 0))
                except Exception:
                    self._worker.ready.connect(self._on_ready)
                self._worker.start()
        except Exception:
            pass


        # JIT: cancel any previous preanalysis and prime small window
        try:
            if hasattr(self, '_worker') and getattr(self, '_worker', None):
                try: self._worker.request_stop()
                except Exception: pass
                try: self._worker.wait(400)
                except Exception: pass
                self._worker = None
        except Exception: pass
        try:
            self._seg_cache.clear()
        except Exception:
            self._seg_cache = {}
        self._cancel_window()
        self.path = Path(path)
        self._kick_window(0)

    def start(self):
        self._timer.start(90)

    def stop(self):
        self._timer.stop()
        try:
            if hasattr(self, '_worker') and getattr(self, '_worker', None):
                try:
                    self._worker.request_stop()
                except Exception:
                    pass
                try:
                    self._worker.wait(500)
                except Exception:
                    pass
                self._worker = None
        except Exception:
            pass

    def _on_ready(self, times_ms: list, bands: list, rms: list):
        self._times = times_ms or []
        self._bands = bands or []
        self._rms = rms or []

    def _on_buf(self, buf):
        try:
            
            # Decimate probe buffers to reduce CPU
            self._probe_decim = (self._probe_decim + 1) % max(1, self._probe_decim_mod)
            if self._probe_decim != 0:
                return
            fmt = buf.format()
            ch = getattr(fmt, 'channelCount', lambda: 1)()
            bps = getattr(fmt, 'bytesPerSample', lambda: 2)()
            data = buf.data()
            mv = memoryview(data)
            step = bps * ch
            if step <= 0:
                return
            vals = []
            for i in range(0, len(mv) - step, step):
                try:
                    if bps >= 4:
                        v = struct.unpack_from('<f', mv, i)[0]
                    else:
                        v = struct.unpack_from('<h', mv, i)[0] / 32768.0
                    vals.append(v)
                except Exception:
                    break
            if not vals:
                return
            n = len(vals)
            sm = 0.0
            for v in vals:
                sm += v * v
            self._probe_rms = min(1.0, (sm / max(1, n)) ** 0.5 * 2.2)
            step2 = max(1, n // self.bands)
            out = []
            for i in range(self.bands):
                seg = vals[i * step2:(i + 1) * step2]
                mvv = 0.0
                for x in seg:
                    if abs(x) > mvv:
                        mvv = abs(x)
                out.append(min(1.0, mvv * 2.0))
            self._probe_levels = out
        except Exception:
            pass

    def _tick(self):
        # JIT fast path: serve from tiny cache near playhead
        try:
            pos = int(getattr(self.player, 'position', lambda: 0)())
            b, r = self._lookup_from_cache(pos)
            if b is not None:
                self.levelsReady.emit(b)
                self.rmsReady.emit(float(r or 0.0))
                return
        except Exception:
            pass



        # Prefer cached segment data around current position
        try:
            pos = int(getattr(self.player, 'position', lambda: 0)())
        except Exception:
            pos = 0
        # look for exact/nearest t_ms within one hop
        near = None
        if getattr(self, '_seg_cache', None):
            ks = list(self._seg_cache.keys())
            if ks:
                near = min(ks, key=lambda k: abs(k - pos))
                if abs(near - pos) > max(1, self._hop_ms) * 100:  # relaxed: hold cached instead of falling back
                    near = None
        if near is not None:
            b, r = self._seg_cache.get(near, (None, None))
            if b is not None:
                self.levelsReady.emit(b)
                self.rmsReady.emit(float(r or 0.0))
                return
        # Fall back to position-synced preanalysis if available
        if self._times and self._bands:
            try:
                idx = int(round(pos / max(1, self._hop_ms)))
            except Exception:
                idx = 0
            if 0 <= idx < len(self._bands):
                bands = self._bands[idx]
                rms = self._rms[idx] if idx < len(self._rms) else 0.0
                self.levelsReady.emit(bands)
                self.rmsReady.emit(rms)
                return
            if getattr(self, '_probe_levels', None) is not None:
                self.levelsReady.emit(self._probe_levels)
                self.rmsReady.emit(self._probe_rms)
                return
        # Probe fallback when no preanalysis yet
        if getattr(self, '_probe_levels', None) is not None:
            self.levelsReady.emit(self._probe_levels)
            self.rmsReady.emit(self._probe_rms)
            return

    def run(self):
        try:
            ff = ffmpeg_path()
            preroll = 200  # ms preroll to stabilize decoder
            ss = max(0, self.start_ms - preroll)
            t = self.dur_ms + preroll
            cmd = [
                ff, '-hide_banner', '-nostats', '-loglevel', 'error',
                '-ss', f'{ss/1000:.3f}', '-t', f'{t/1000:.3f}',
                '-i', str(self.path), '-vn',
                '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p = self._proc
            import struct, math
            hop = int(round(0.001 * self.hop_ms * 24000))
            if hop <= 0: hop = 480
            win = hop * 2
            data = p.stdout.read()
            mv = memoryview(data)
            step = 4  # float32
            total = len(mv)//step
            frames = total // hop
            bands_mat, rms_list, times_ms = [], [], []
            for i in range(frames):
                base = i*hop
                end = min(total, base+win)
                if end <= base: continue
                peak = 0.0
                off = base*step
                for j in range(base, end):
                    v = struct.unpack_from('<f', mv, j*step)[0]
                    if abs(v) > peak: peak = abs(v)
                # simple peak-to-bands distribution (flat) to stay fast
                level = min(1.0, peak*2.0)
                out = [level for _ in range(self.bands)]
                bands_mat.append(out)
                rms_list.append(min(1.0, level*1.1))
                t_ms = self.start_ms + (i*self.hop_ms)
                times_ms.append(t_ms)
            try:
                (self._proc or p).kill() if (self._proc or p) else None
            except Exception:
                pass
            self.ready_segment.emit(self.start_ms, times_ms, bands_mat, rms_list)
        except Exception:
            pass

    def _cancel_window(self):
        try:
            if getattr(self, '_seg_worker', None):
                try: self._seg_worker.request_stop()
                except Exception: pass
                self._seg_worker = None
        except Exception:
            pass
        self._seg_inflight = False

    def _kick_window(self, center_ms: int):
        self._cancel_window()
        if not getattr(self, 'path', None):
            return
        start_ms = max(0, int(center_ms) - int(self._preroll_ms))
        dur_ms = int(self._win_ms)
        w = PreAnalyzer(self.path, sr=22050, bands=self.bands, hop_ms=self._hop_ms, parent=self)
        w.start_ms = start_ms
        w.dur_ms = dur_ms
        def _take(times, bands, rms):
            try:
                for t, b, r in zip(times, bands, rms):
                    self._seg_cache[int(t)] = (b, r)
                self._win_start = start_ms
                self._win_end = start_ms + dur_ms
                pos = int(getattr(self.player, 'position', lambda: 0)() or 0)
                if times:
                    idx = max(0, min(len(times)-1, int((pos - start_ms) / max(1, self._hop_ms))))
                    self.levelsReady.emit(bands[idx])
                    self.rmsReady.emit(rms[idx])
            except Exception:
                pass
            finally:
                self._seg_inflight = False
        try:
            w.ready.connect(_take, getattr(Qt, 'UniqueConnection', 0))
        except Exception:
            w.ready.connect(_take)
        self._seg_worker = w
        self._seg_inflight = True
        w.start()

    def _on_pos_update_window(self, pos: int):
        try:
            if pos < self._win_start or pos > self._win_end:
                self._kick_window(int(pos))
                return
            span = max(1, self._win_end - self._win_start)
            if not self._seg_inflight and (pos - self._win_start) >= int(0.70 * span):
                next_center = int(pos + 0.25 * span)
                self._kick_window(next_center)
        except Exception:
            pass

    def _lookup_from_cache(self, pos_ms: int):
        try:
            if not self._seg_cache:
                return None, None
            k = min(self._seg_cache.keys(), key=lambda t: abs(t - pos_ms))
            return self._seg_cache.get(k, (None, None))
        except Exception:
            return None, None

class VisualEngine(QObject):
    frameReady = Signal(QImage)

    def __init__(self, parent: Optional[QObject] = None, bars: int = 48):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.target_size = QSize(800, 800)
        self.start_time = _time.time()
        self.mode = 'spectrum'
        self.enabled = False  # visuals start OFF; analyzer/plugins still preload
        self._levels = [0.0] * bars
        self._rms = 0.0
        self._bars = bars
        # High-shelf visual tilt (0.0–0.5). 0.18 ≈ +1.5dB at top band
        self._high_tilt = 0.18
        self._plugin = None
        _load_visual_plugins()

    def available_modes(self) -> List[str]:
        base = ['spectrum']
        for name, cls in _VISUAL_REGISTRY:
            if _norm_name(name) in _BLOCKED:
                continue
            base.append(f'viz:{name}')
        return base

    def set_mode(self, mode: str):
        if mode.startswith('viz:'):
            disp = mode.split(':', 1)[1]
            self._plugin = None
            for name, cls in _VISUAL_REGISTRY:
                if name == disp:
                    try:
                        self._plugin = cls()
                    except Exception:
                        self._plugin = None
                    break
            self.mode = mode
            return
        self._plugin = None
        self.mode = mode

    
    
    
    
    def inject_levels(self, levels: List[float]):
        """Ultra-responsive bass: instant attack + fast release for lowest bands.
        - Bass bands: no attack lag (direct passthrough on rise), fast controlled decay.
        - Other bands: mild asymmetric EMA for smoothness.
        """
        if not levels:
            return
        if len(levels) < self._bars:
            vals = levels + [0.0] * (self._bars - len(levels))
        else:
            vals = levels[:self._bars]

        # Init buffers
        if not hasattr(self, "_levels_smooth") or not self._levels_smooth or len(self._levels_smooth) != self._bars:
            self._levels_smooth = list(vals)
            self._levels = list(vals)
            return

        out = [0.0] * self._bars

        # Define bass region size (captures kick fundamentals + first harmonics)
        bass_k = max(8, self._bars // 6)

        for i, v in enumerate(vals):
            prev = self._levels_smooth[i]
            v = float(v)

            if i < bass_k:
                # === Bass bands: zero-lag attack ===
                if v >= prev:
                    s = v  # immediate rise, no EMA
                else:
                    # Fast release per frame at ~30fps
                    decay = 0.40
                    s = max(v, prev - decay)
                out[i] = s
            else:
                # === Other bands: gentle asymmetry ===
                rising = v >= prev
                if i < max(16, self._bars // 4):        # low-mids
                    alpha_up, alpha_down = 0.34, 0.18
                elif i < max(24, self._bars // 2):      # mids
                    alpha_up, alpha_down = 0.30, 0.20
                else:                                    # highs
                    alpha_up, alpha_down = 0.36, 0.26
                a = alpha_up if rising else alpha_down
                out[i] = prev * (1.0 - a) + v * a

        # Minimal neighbor blend on the first few bins to avoid isolated spikes
        tiny = min(3, max(2, self._bars // 20))
        for i in range(tiny):
            left  = out[i - 1] if i - 1 >= 0 else out[i]
            right = out[i + 1] if i + 1 < self._bars else out[i]
            out[i] = 0.92 * out[i] + 0.04 * (left + right)

        # Apply gentle high-shelf tilt: progressively boost higher bands a little
        try:
            hb = float(getattr(self, '_high_tilt', 0.18))
        except Exception:
            hb = 0.18
        if hb > 0.0 and self._bars > 1:
            top = self._bars - 1
            for i in range(self._bars):
                t = i / top
                # perceptual-ish curve; raise highs up to ~+hb
                gain = 1.0 + hb * (t ** 0.9)
                out[i] = min(1.0, out[i] * gain)
        self._levels_smooth = out
        self._levels = out

    
    def inject_rms(self, rms: float):
        self._rms = float(max(0.0, min(1.0, rms)))

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_target(self, size: QSize):
        self.target_size = size

    def start(self, fps: int = 30):
        self.timer.start(int(1000 / max(1, fps)))

    def stop(self):
        self.timer.stop()

    def _tick(self):
        w = max(64, self.target_size.width())
        h = max(64, self.target_size.height())
        img = QImage(w, h, QImage.Format_RGBA8888)
        img.fill(QColor(10, 10, 12, 255))
        p = QPainter(img)
        try:
            if not self.enabled:
                p.fillRect(0, 0, w, h, QColor(8, 8, 10))
            else:
                if self.mode.startswith('viz:') and self._plugin is not None:
                    rect = QRectF(0, 0, w, h)
                    try:
                        self._plugin.paint(p, rect, self._levels, self._rms, _time.time() - self.start_time)
                    except Exception:
                        pass
                elif self.mode == 'spectrum':
                    self._draw_spectrum(p, w, h)
        finally:
            p.end()
        self.frameReady.emit(img)

    def _draw_spectrum(self, p: QPainter, w: int, h: int):
        bars = self._bars
        bw = max(2, w // (bars + 2))
        now = int(_time.time() * 40)
        for i in range(bars):
            v = max(0.02, min(1.0, self._levels[i]))
            bh = int(v * (h * 0.9))
            x = 1 + i * bw
            y = (h - bh) // 2
            col = QColor.fromHsv(((i * 7) + now) % 360, 200, 250, 255)
            p.fillRect(QRect(x, y, bw - 3, bh), col)
        p.fillRect(0, 0, w, 12, QColor(0, 0, 0, 120))
        p.fillRect(0, h - 12, w, 12, QColor(0, 0, 0, 120))

# ---------------- UI ----------------

@dataclass
class TrackInfo:
    path: Path
    title: str = ''
    artist: str = ''
    album: str = ''
    bpm: int = 120

class Collapser(QWidget):
    toggled = Signal(bool)  # True when collapsed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('fvMusicCard')
        self.setStyleSheet(
            '#fvMusicCard { background: rgba(0,0,0,110); border-radius: 14px; }'
            '#fvMusicCard QLabel, #fvMusicCard QListWidget, #fvMusicCard QCheckBox, #fvMusicCard QComboBox { color: palette(text);  background: palette(base);  border: 1px solid palette(mid); border-radius: 6px; }'
            '#fvMusicCard QPushButton { background: rgba(255,255,255,28); border: none; padding: 6px 10px; border-radius: 9px;}'
            '#fvMusicCard QPushButton:hover { background: rgba(255,255,255,40); }'
        )
        self._content = QWidget(self)
        self._fx = QGraphicsOpacityEffect(self._content)
        self._content.setGraphicsEffect(self._fx)
        self._fx.setOpacity(1.0)
        self._vl = QVBoxLayout(self._content)
        self._vl.setContentsMargins(10, 10, 10, 10)
        self._vl.setSpacing(8)
        self._toggle = QPushButton('♪', self)
        self._toggle.setToolTip('Show/Hide music controls')
        self._toggle.setFixedSize(30, 30)
        self._toggle.setStyleSheet('QPushButton {background: rgba(0,0,0,150); color:white; border-radius: 15px;} QPushButton:hover {background: rgba(0,0,0,200);}')
        self._toggle.clicked.connect(self.toggle)
        self._toggle.hide()  # Use external FAB instead
        self._is_collapsed = False

    def layout(self):
        return self._vl

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._toggle.isVisible():
            self._toggle.move(6, self.height() - self._toggle.height() - 6)

    def toggle(self):
        self._is_collapsed = not self._is_collapsed
        # fade animation
        try:
            anim = QPropertyAnimation(self._fx, b'opacity', self)
            anim.setDuration(180)
            anim.setStartValue(1.0 if not self._is_collapsed else 0.0)
            anim.setEndValue(0.0 if self._is_collapsed else 1.0)
            anim.start()
        except Exception:
            pass
        self._content.setVisible(not self._is_collapsed)  # type: ignore
        self.adjustSize()
        self.update()
        self.toggled.emit(self._is_collapsed)

    def setCollapsed(self, collapsed: bool):
        if self._is_collapsed != collapsed:
            self.toggle()

class MusicOverlay(QWidget):
    def __init__(self, video_pane, parent=None):
        super().__init__(parent or video_pane)
        self.suspended = False
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)
        self.setObjectName('musicOverlay')
        self.video = video_pane

        self.card = Collapser(self)
        self.card.move(0, 0)
        self.card.setCollapsed(False)  # start collapsed to keep visuals clean

        # floating FAB
        self.fab = QPushButton('♪', self)
        self.fab.setToolTip('Show/Hide music controls')
        self.fab.setFixedSize(40, 40)
        self.fab.clicked.connect(self.card.toggle)

        # update fab color on toggle
        self.card.toggled.connect(lambda _: self._update_fab_color())
        QTimer.singleShot(0, self._update_fab_color)

        # playlist
        self.playlist = QListWidget()
        self.playlist.setToolTip('Double-click a track to play it. In shuffle mode, sequence continues from the clicked track.')
        try:
            self.playlist.setMinimumWidth(260)
            self.playlist.setAlternatingRowColors(False)
            self.playlist.setStyleSheet('QListWidget { color: #ffffff;  background: transparent;} QListWidget::item{padding:6px 8px; margin:2px;} QListWidget::item:selected{ color: palette(highlighted-text); }\nQComboBox { color: palette(text);  background: palette(base);  border: 1px solid palette(mid); border-radius: 6px; }\nQAbstractItemView { color: palette(window-text); background: palette(window); }\nQAbstractItemView::item:selected { background: palette(highlight); color: palette(highlighted-text); }')
        except Exception:
            pass
        self.btn_add = QPushButton('Add…')
        self.btn_add.setToolTip('Add audio files to the playlist.')
        self.btn_clear = QPushButton('Clear')
        self.btn_clear.setToolTip('Clear playlist (also resets shuffle order).')
        self.btn_open_pl = QPushButton('Open playlist')
        self.btn_open_pl.setToolTip('Load a playlist file (.m3u/.m3u8/.json)')
        self.btn_save_pl = QPushButton('Save playlist')
        self.btn_save_pl.setToolTip('Save current playlist to presets/setsave/playlist')
        self.btn_prev = QPushButton('⏮')
        self.btn_prev.setToolTip('Previous track. In shuffle, goes back in the shuffle order.')
        self.btn_next = QPushButton('⏭')
        self.btn_next.setToolTip('Next track. In shuffle, follows the current shuffle order.')
        self.btn_repeat = QPushButton('Repeat: all')
        self._set_repeat_label_and_tooltip(self.repeat_mode())
        row_btns = QHBoxLayout()
        for b in (self.btn_add, self.btn_clear, self.btn_open_pl, self.btn_save_pl, self.btn_prev, self.btn_next, self.btn_repeat):
            row_btns.addWidget(b)
        row_btns.addStretch(1)

        # visuals controls
        self.cmb_visual = QComboBox()
        # Limit the visual selector width so it doesn't collide with the beats control
        try:
            self.cmb_visual.setMinimumWidth(160)
            self.cmb_visual.setMaximumWidth(280)
        except Exception:
            pass
        # Keep the visuals selector a stable width across show/hide
        try:
            self.cmb_visual.setFixedWidth(180)
        except Exception:
            pass
        # try to widen the dropdown; fall back if QSizePolicy isn't available
        try:
            from PySide6.QtWidgets import QSizePolicy as _QSP
        except Exception:
            _QSP = None
        try:
            if _QSP:
                self.cmb_visual.setSizePolicy(_QSP.Fixed, _QSP.Fixed)
        except Exception:
            pass
        # (removed) minimum width override
# Persist a fixed width for the visuals selector across show/hide toggles
        try:
            self._cmb_visual_fixed_w = getattr(self, "_cmb_visual_fixed_w", 180)
            self.cmb_visual.setFixedWidth(self._cmb_visual_fixed_w)
        except Exception:
            pass
        # Re-apply the width whenever the card is collapsed/expanded (prevents width 'snapping back')
        try:
            self.card.toggled.connect(lambda _c: self.cmb_visual.setFixedWidth(self._cmb_visual_fixed_w))
        except Exception:
            pass

        self.btn_vis_prev = QPushButton('◀'); self.btn_vis_prev.hide()
        self.btn_vis_next = QPushButton('▶'); self.btn_vis_next.hide()
        self.btn_visuals = QPushButton('Visuals on')
        self.btn_visuals.setToolTip('Visuals change rate (beats): 1 / random / all. (Unaffected by playlist repeat.)')
        self.btn_visuals.setCheckable(True)
        self.btn_visuals.setChecked(False)

        # auto-change controls
        self.chk_auto = QCheckBox('Auto-change visuals')
        self.cmb_beats = QComboBox()
        self.cmb_beats.addItems(['8', '16', '32', '64'])
        self.cmb_beats.setCurrentIndex(2)
        # Duration button cycles 8/16/32/64 beats
        self.btn_dur = QPushButton('32 beats')
        self.btn_dur.setToolTip('Change how long each visual stays (in beats)')
        def _cycle_dur():
            lst = ['8','16','32','64']
            cur = self.cmb_beats.currentText()
            idx = lst.index(cur) if cur in lst else 2
            idx = (idx + 1) % len(lst)
            self.cmb_beats.setCurrentIndex(idx)
            try:
                self.btn_dur.setText(f'{lst[idx]} beats')
            except Exception:
                pass
            try:
                if getattr(self.parent(), '_music_runtime', None):
                    self.parent()._music_runtime._persist_state()
            except Exception:
                pass
        self.btn_dur.clicked.connect(_cycle_dur)
        self.cmb_beats.currentIndexChanged.connect(lambda _i: self.btn_dur.setText(f'{self.cmb_beats.currentText()} beats'))
        self.btn_dur.setText(f'{self.cmb_beats.currentText()} beats')
        self.btn_vizmode = QPushButton('random')

        row_vis1 = QHBoxLayout()
        row_vis1.addWidget(self.btn_visuals)
        row_vis1.addSpacing(20)
        row_vis1.addWidget(self.btn_vizmode)
        row_vis1.addWidget(self.cmb_visual)
        row_vis1.addSpacing(25)
        row_vis1.addWidget(self.btn_dur)
                
        row_vis2 = QHBoxLayout()
        row_vis2.addWidget(self.chk_auto)
        row_vis2.addSpacing(8)
        row_vis2.addWidget(QLabel('Every'))
        row_vis2.addWidget(self.cmb_beats)
        row_vis2.addWidget(QLabel('beats'))
        row_vis2.addStretch(1)

        # assemble: playlist only in card top
        top = QHBoxLayout()
        top.addWidget(self.playlist, 1)
        L = self.card.layout()
        L.addLayout(top)
        L.addLayout(row_btns)
        L.addLayout(row_vis1)
        L.addLayout(row_vis2)
        # crossfade controls
        self.chk_xfade = QCheckBox('Crossfade visuals')
        self.cmb_fade = QComboBox()
        try:
            self.cmb_fade.addItems(['1.0s','1.5s','2.0s','3.0s','4.0s'])
            self.cmb_fade.setCurrentIndex(2)  # default 2.0s
        except Exception:
            pass
        row_vis3 = QHBoxLayout()
        row_vis3.addWidget(self.chk_xfade)
        row_vis3.addSpacing(8)
        row_vis3.addWidget(QLabel('Fade'))
        row_vis3.addWidget(self.cmb_fade)
        row_vis3.addStretch(1)
        L.addLayout(row_vis3)


        # signals
        self.btn_add.clicked.connect(self._add_files)
        self.btn_clear.clicked.connect(self._clear_playlist)
        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        self.playlist.itemDoubleClicked.connect(self._play_clicked)
        self.btn_repeat.clicked.connect(self._toggle_repeat)
        self.btn_vizmode.clicked.connect(self._toggle_vizmode)

        # new: open/save playlist
        try:
            self.btn_open_pl.clicked.connect(self._open_playlist_file)
            self.btn_save_pl.clicked.connect(self._save_playlist_file)
        except Exception:
            pass
        # context menu for playlist (widget + viewport) and event filter fallback
        try:
            self.playlist.setContextMenuPolicy(Qt.CustomContextMenu)
            self.playlist.customContextMenuRequested.connect(self._on_playlist_context_menu)
            vp = self.playlist.viewport()
            if vp is not None:
                vp.setContextMenuPolicy(Qt.CustomContextMenu)
                vp.customContextMenuRequested.connect(lambda pos: self._on_playlist_context_menu(self.playlist.mapFrom(vp, pos)))
                vp.installEventFilter(self)
        except Exception:
            pass

        # keyboard shortcuts
        try:
            QShortcut(QKeySequence('V'), self, activated=lambda: self.btn_visuals.toggle())
            QShortcut(QKeySequence('P'), self, activated=lambda: self.btn_prev.click())
            QShortcut(QKeySequence('N'), self, activated=lambda: self.btn_next.click())
        except Exception:
            pass

        self._tracks: List[TrackInfo] = []
        self._current_idx = -1
        # shuffle state (playlist only)
        self._shuffle_order = []  # type: list[int]
        self._shuffle_pos = -1
        self._visual_modes: List[str] = []
        self._visual_index = 0

        self.installEventFilter(self)
        self.video.installEventFilter(self)
        if hasattr(self.video, 'label'):
            self.video.label.installEventFilter(self)
        self._reposition()
        self.show()
        # ----- inactivity auto-hide (UI fades out after 5s; shows on activity) -----
        self._inactive_ms = 7500
        self._hidden = False
        self._anims = []
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(self._inactive_ms)
        self._hide_timer.timeout.connect(self._go_inactive)

        # ensure mouse move events arrive even without buttons pressed
        try:
            self.setMouseTracking(True)
            self.video.setMouseTracking(True)
            if hasattr(self.video, 'label') and self.video.label:
                self.video.label.setMouseTracking(True)
        except Exception:
            pass

        def _hook_window():
            try:
                w = self.window()
                if w:
                    w.installEventFilter(self)
                    try:
                        w.setMouseTracking(True)
                    except Exception:
                        pass
            except Exception:
                pass
        QTimer.singleShot(0, _hook_window)

        # Also install event filter on the combo popup view (for wheel/scroll)
        try:
            v = self.cmb_visual.view()
            if v:
                v.installEventFilter(self)
        except Exception:
            pass

        # start the inactivity timer immediately
        QTimer.singleShot(0, lambda: self._activity())

            # restore playlist from previous session
        try:
            QTimer.singleShot(0, self._restore_playlist_from_state)
        except Exception:
            pass
# ----- playlist persistence -----
    def _persist_playlist_state(self):
        try:
            st = _load_state()
        except Exception:
            st = {}
        try:
            st['playlist_paths'] = [str(t.path) for t in getattr(self, '_tracks', [])]
            st['playlist_current_idx'] = int(getattr(self, '_current_idx', -1))
            _save_state(st)
        except Exception:
            pass

    def _restore_playlist_from_state(self):
        try:
            st = _load_state()
            paths = st.get('playlist_paths') or []
            cur = int(st.get('playlist_current_idx', -1))
        except Exception:
            paths, cur = [], -1
        if paths:
            self._clear_playlist()
            for p in paths:
                pp = Path(p)
                if pp.exists():
                    try:
                        self.add_track(pp)
                    except Exception:
                        pass
            try:
                if 0 <= cur < len(self._tracks):
                    self._current_idx = cur
                    self.playlist.setCurrentRow(cur)
            except Exception:
                pass

    # ----- clear with shuffle reset + persist -----
    def _clear_playlist(self):
        try:
            self.playlist.clear()
        except Exception:
            pass
        try:
            self._tracks = []
            self._current_idx = -1
            self._shuffle_order = []
            self._shuffle_pos = -1
        except Exception:
            pass
        self._persist_playlist_state()

    # ----- context menu handlers -----
    def _on_playlist_context_menu(self, pos):
        try:
            item = self.playlist.itemAt(pos)
            if not item:
                return
            row = self.playlist.row(item)
            gp = self.playlist.mapToGlobal(pos)
            self._exec_playlist_menu(gp, row)
        except Exception:
            pass

    def eventFilter(self, obj, ev):
        # Right-click fallback on viewport
        try:
            from PySide6.QtCore import QEvent
            if obj is self.playlist.viewport():
                if ev.type() == QEvent.ContextMenu:
                    pos = ev.pos()
                    item = self.playlist.itemAt(pos)
                    if item:
                        row = self.playlist.row(item)
                        self._exec_playlist_menu(ev.globalPos(), row)
                        return True
                if ev.type() == QEvent.MouseButtonRelease and getattr(ev, 'button', lambda: None)() == Qt.RightButton:
                    pos = ev.pos()
                    item = self.playlist.itemAt(pos)
                    if item:
                        row = self.playlist.row(item)
                        gp = self.playlist.viewport().mapToGlobal(pos)
                        self._exec_playlist_menu(gp, row)
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    def _exec_playlist_menu(self, global_pos, row: int):
        try:
            menu = QMenu(self)
            a_top = menu.addAction('Move to top')
            a_up = menu.addAction('Move up')
            a_down = menu.addAction('Move down')
            a_bot = menu.addAction('Move to bottom')
            menu.addSeparator()
            a_rm = menu.addAction('Remove from playlist')
            a_info = menu.addAction('Show info')
            a_del = menu.addAction('Delete from disk…')
            act = menu.exec(global_pos)
            if act is None:
                return
            if act == a_top:
                self._move_item(row, 0)
            elif act == a_bot:
                self._move_item(row, self.playlist.count()-1)
            elif act == a_up:
                self._move_item(row, max(0, row-1))
            elif act == a_down:
                self._move_item(row, min(self.playlist.count()-1, row+1))
            elif act == a_rm:
                self._remove_item(row)
            elif act == a_info:
                self._show_track_info(row)
            elif act == a_del:
                self._delete_track_from_disk(row)
        except Exception:
            pass

    def _move_item(self, old_row: int, new_row: int):
        try:
            if old_row == new_row or not (0 <= old_row < len(self._tracks)):
                return
            new_row = max(0, min(len(self._tracks)-1, new_row))
            tr = self._tracks.pop(old_row)
            self._tracks.insert(new_row, tr)
            it = self.playlist.takeItem(old_row)
            self.playlist.insertItem(new_row, it)
            self.playlist.setCurrentRow(new_row)
            if self._current_idx == old_row:
                self._current_idx = new_row
            elif old_row < self._current_idx <= new_row:
                self._current_idx -= 1
            elif new_row <= self._current_idx < old_row:
                self._current_idx += 1
            self._shuffle_order = []
            self._shuffle_pos = -1
        except Exception:
            pass
        self._persist_playlist_state()

    def _remove_item(self, row: int):
        try:
            if not (0 <= row < len(self._tracks)):
                return
            self.playlist.takeItem(row)
            self._tracks.pop(row)
            if self._current_idx == row:
                self._current_idx = -1
            elif row < self._current_idx:
                self._current_idx -= 1
            self._shuffle_order = []
            self._shuffle_pos = -1
        except Exception:
            pass
        self._persist_playlist_state()

    def _show_track_info(self, row: int):
        try:
            if not (0 <= row < len(self._tracks)):
                return
            p = Path(self._tracks[row].path)
            lines = _make_human_info_from_ffprobe(p)
            msg = "\n".join(lines) if lines else f"File: {p.name}\nPath: {p}"
            try:
                QMessageBox.information(self, "Track info", msg)
            except Exception:
                print(msg)
        except Exception:
            pass
    def _delete_track_from_disk(self, row: int):
        try:
            if not (0 <= row < len(self._tracks)):
                return
            p = Path(self._tracks[row].path)
            try:
                ret = QMessageBox.question(self, "Delete file?",
                                           f"Really delete this file from disk?\n\n{p}",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if ret != QMessageBox.Yes:
                    return
            except Exception:
                return
            try:
                os.remove(p)
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Delete failed", f"Couldn't delete:\n{e}")
                except Exception:
                    pass
                return
            self._remove_item(row)
        except Exception:
            pass

    def _save_playlist_file(self):
        try:
            if not getattr(self, '_tracks', None):
                QMessageBox.information(self, "Save playlist", "Playlist is empty.")
                return
        except Exception:
            pass
        mw = self.video.window()
        try:
            default_name = "my_playlist.m3u8"
            fn, flt = QFileDialog.getSaveFileName(mw, "Save playlist", str(PLAYLIST_DIR / default_name),
                                                  "M3U Playlist (*.m3u8 *.m3u);;JSON (*.json)")
        except Exception:
            fn = ""
        if not fn:
            return
        path = Path(fn)
        try:
            if not str(path).startswith(str(PLAYLIST_DIR)):
                path = PLAYLIST_DIR / path.name
        except Exception:
            pass
        if path.suffix.lower() not in ('.m3u', '.m3u8', '.json'):
            path = path.with_suffix('.m3u8')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if path.suffix.lower() in ('.m3u', '.m3u8'):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("#EXTM3U\n")
                    for t in self._tracks:
                        f.write(str(Path(t.path).resolve()) + "\n")
            else:
                data = {"tracks":[str(Path(t.path).resolve()) for t in self._tracks]}
                path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            try:
                QMessageBox.information(self, "Playlist saved", f"Saved to:\n{path}")
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.warning(self, "Save failed", f"Couldn't save playlist:\n{e}")
            except Exception:
                pass

    def _open_playlist_file(self):
        mw = self.video.window()
        try:
            fn, flt = QFileDialog.getOpenFileName(mw, "Open playlist", str(PLAYLIST_DIR),
                                                  "Playlists (*.m3u *.m3u8 *.json);;All files (*.*)")
        except Exception:
            fn = ""
        if not fn:
            return
        path = Path(fn)
        tracks = []
        try:
            if path.suffix.lower() in ('.m3u', '.m3u8'):
                base = path.parent
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        pp = Path(s)
                        if not pp.is_absolute():
                            pp = base / s
                        tracks.append(pp)
            else:
                data = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(data, dict) and 'tracks' in data:
                    tracks = [Path(p) for p in data.get('tracks') or []]
                elif isinstance(data, list):
                    tracks = [Path(p) for p in data]
        except Exception as e:
            try:
                QMessageBox.warning(self, "Open failed", f"Couldn't read playlist:\n{e}")
            except Exception:
                pass
            return
        self._clear_playlist()
        for pp in tracks:
            if pp.exists():
                self.add_track(pp)
        try:
            if self.playlist.count() > 0:
                self.playlist.setCurrentRow(0)
                self._current_idx = 0
        except Exception:
            pass
        self._persist_playlist_state()

    # ----- helpers / UI integration -----
    def _update_fab_color(self):
        try:
            collapsed = getattr(self.card, '_is_collapsed', False) or not getattr(self.card, '_content', self.card).isVisible()
        except Exception:
            collapsed = False
        if collapsed:
            # green
            self.fab.setStyleSheet('QPushButton {background: rgba(40,180,99,200); color:white; border-radius: 20px;} QPushButton:hover {background: rgba(46,204,113,220);}')
        else:
            # red
            self.fab.setStyleSheet('QPushButton {background: rgba(231,76,60,200); color:white; border-radius: 20px;} QPushButton:hover {background: rgba(231,76,60,230);}')

    def eventFilter(self, obj, ev):
                # Ignore events while overlay is suspended (non-audio mode)
        if getattr(self, 'suspended', False):
            return False
# Any user activity resets timer and shows UI
        if ev.type() in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.Wheel, QEvent.KeyPress, QEvent.KeyRelease):
            try:
                self._activity()
            except Exception:
                pass
        if ev.type() in (QEvent.Resize, QEvent.Show, QEvent.WindowStateChange):
            QTimer.singleShot(0, self._reposition)
            QTimer.singleShot(0, self.raise_)
            QTimer.singleShot(0, self.show)
        return False

    def _reposition(self):
        try:
            r = self.video.label.geometry()
        except Exception:
            r = self.video.geometry()
        try:
            self.setGeometry(r)
        except Exception:
            pass

        # place card ~450px from left, but snap to center if wide enough
        w = r.width()
        h = r.height()
        W = min(560, int(w * 0.9))
        H = min(280, int(h * 0.6))
        desired_x = 450
        center_x = max(12, (w - W) // 2)
        max_x = max(12, w - W - 12)
        if center_x >= desired_x:
            x = center_x
        else:
            x = min(desired_x, max_x)
        y = max(12, h - H - 12)
        try:
            self.card.setFixedWidth(W)
            self.card.setFixedHeight(H)
        except Exception:
            pass
        self.card.move(x, y)

        # anchor FAB to header top-right (header rect ~ (18,18,width<=520,148))
        header_x = 18
        header_y = 18
        header_w = min(520, int(w * 0.7))
        fx = header_x + header_w - self.fab.width() - 8 + 75
        fy = header_y + 8 + 25
        fx = min(max(6, fx), max(6, w - self.fab.width() - 6))
        fy = min(max(6, fy), max(6, h - self.fab.height() - 6))
        self.fab.move(fx, fy)

        self.card.raise_()
        self.fab.raise_()
        self._update_fab_color()
        self.raise_()

    def _fade_widget(self, w, show: bool):
        try:
            eff = getattr(w, '_auto_fx', None)
            if eff is None:
                eff = QGraphicsOpacityEffect(w)
                try:
                    w.setGraphicsEffect(eff)
                except Exception:
                    pass
                w._auto_fx = eff  # type: ignore[attr-defined]
            anim = QPropertyAnimation(eff, b'opacity', self)
            anim.setDuration(320)
            try:
                start = 0.0 if show else 1.0
                end = 1.0 if show else 0.0
                if show:
                    try:
                        eff.setOpacity(0.0)
                        w.setVisible(True)
                    except Exception:
                        pass
                else:
                    try:
                        eff.setOpacity(1.0)
                    except Exception:
                        pass
                anim.setStartValue(start)
                anim.setEndValue(end)
                if not show:
                    def _after():
                        try:
                            w.setVisible(False)
                        except Exception:
                            pass
                    anim.finished.connect(_after)
                anim.start()
                if not hasattr(self, '_anims'):
                    self._anims = []
                self._anims.append(anim)
            except Exception:
                w.setVisible(show)
        except Exception:
            try:
                w.setVisible(show)
            except Exception:
                pass

    def _set_hidden(self, hidden: bool):
        self._hidden = bool(hidden)
        try:
            if getattr(self.parent(), '_music_runtime', None):
                self.parent()._music_runtime._ui_hidden = self._hidden
        except Exception:
            pass

    def _go_inactive(self):
        if self._hidden:
            return
        self._set_hidden(True)
        try:
            self._fade_widget(self.card, False)
            self._fade_widget(self.fab, False)
        except Exception:
            try:
                self.card.hide(); self.fab.hide()
            except Exception:
                pass

    def _activity(self):
        try:
            self._hide_timer.start(self._inactive_ms)
        except Exception:
            pass
        if self._hidden:
            self._set_hidden(False)
            try:
                self._fade_widget(self.card, True)
                self._fade_widget(self.fab, True)
            except Exception:
                try:
                    self.card.show(); self.fab.show()
                except Exception:
                    pass


    # ----- playlist helpers -----
    def _add_files(self):
        mw = self.video.window()
        # Load last directory from state (falls back to ROOT)
        try:
            st = _load_state()
            start_dir = st.get('last_add_dir', str(ROOT))
        except Exception:
            start_dir = str(ROOT)
        paths, _ = QFileDialog.getOpenFileNames(
            mw, 'Add audio files', start_dir,
            'Audio files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma *.aif *.aiff)')
        if paths:
            # Save the directory for next time
            try:
                st = _load_state()
                from pathlib import Path as _P
                st['last_add_dir'] = str(_P(paths[0]).parent)
                _save_state(st)
            except Exception:
                pass
        for p in paths:
            self.add_track(Path(p))

    def add_track(self, path: Path):
        p = Path(path)
        tags = _read_all_tags(p)
        title = tags.get('title') or tags.get('TITLE') or p.stem
        artist = tags.get('artist') or tags.get('ARTIST') or ''
        album = tags.get('album') or tags.get('ALBUM') or ''
        text = title
        sub = ' — '.join([s for s in (artist, album) if s])
        if sub:
            text += f"\n{sub}"
        self._tracks.append(TrackInfo(path=p, title=title, artist=artist, album=album))
        self.playlist.addItem(QListWidgetItem(text))
        try:
            self._persist_playlist_state()
        except Exception:
            pass

    def ensure_in_playlist(self, path: Path) -> int:
        for i, tr in enumerate(self._tracks):
            if Path(tr.path) == Path(path):
                self._current_idx = i
                try:
                    self.playlist.setCurrentRow(i)
                except Exception:
                    pass
                try:
                    self._persist_playlist_state()
                except Exception:
                    pass
                return i
        self.add_track(path)
        idx = len(self._tracks) - 1
        self._current_idx = idx
        try:
            self.playlist.setCurrentRow(idx)
        except Exception:
            pass
        try:
            self._persist_playlist_state()
        except Exception:
            pass
        return idx

    def current_track(self) -> Optional[Path]:
        if 0 <= self._current_idx < len(self._tracks):
            return self._tracks[self._current_idx].path
        return None

    def _play_clicked(self, item: QListWidgetItem):
        idx = self.playlist.row(item)
        if 0 <= idx < len(self._tracks):
            self._current_idx = idx
            if self.repeat_mode() == 'shuffle':
                self._build_shuffle_order(idx)
            self.video.open(self._tracks[idx].path)

    
    # ----- shuffle helpers (playlist only) -----
    def _build_shuffle_order(self, start_idx: int = 0):
        """
        Build a permutation of playlist indices with no repeats until all played.
        Places start_idx first; the rest are shuffled.
        """
        n = len(self._tracks)
        self._shuffle_order = []
        self._shuffle_pos = -1
        if n <= 0:
            return
        import random
        idxs = list(range(n))
        if 0 <= start_idx < n:
            idxs.remove(start_idx)
            random.shuffle(idxs)
            seq = [start_idx] + idxs
        else:
            random.shuffle(idxs)
            seq = idxs
        self._shuffle_order = seq
        self._shuffle_pos = 0

    def _shuffle_next_index(self, delta: int) -> int:
        """
        Move delta steps along the shuffle order.
        On completing a full pass, reshuffle and continue.
        """
        n = len(self._tracks)
        if n == 0:
            return -1
        # Ensure order exists and is valid
        if not self._shuffle_order or any(i >= n for i in self._shuffle_order):
            base = self._current_idx if self._current_idx >= 0 else 0
            self._build_shuffle_order(base)
        # Sync shuffle position to current index if unknown
        if self._shuffle_pos < 0 and self._current_idx >= 0:
            try:
                self._shuffle_pos = self._shuffle_order.index(self._current_idx)
            except ValueError:
                self._build_shuffle_order(self._current_idx)

        if delta > 0:
            self._shuffle_pos += 1
            if self._shuffle_pos >= len(self._shuffle_order):
                # Completed a pass: reshuffle from current
                current = self._shuffle_order[-1] if self._shuffle_order else self._current_idx
                self._build_shuffle_order(current)
                self._shuffle_pos = 1 if len(self._shuffle_order) > 1 else 0
        elif delta < 0:
            self._shuffle_pos -= 1
            if self._shuffle_pos < 0:
                current = self._shuffle_order[0] if self._shuffle_order else self._current_idx
                self._build_shuffle_order(current)
                self._shuffle_pos = len(self._shuffle_order) - 1 if self._shuffle_order else -1

        if 0 <= self._shuffle_pos < len(self._shuffle_order):
            return self._shuffle_order[self._shuffle_pos]
        return -1

    def _jump(self, delta: int):
            if not self._tracks:
                return
            if self.repeat_mode() == 'shuffle':
                if self._current_idx < 0:
                    self._current_idx = 0
                    self._build_shuffle_order(self._current_idx)
                idx = self._shuffle_next_index(delta)
                if idx >= 0:
                    self._current_idx = idx
            else:
                if self._current_idx < 0:
                    self._current_idx = 0
                else:
                    self._current_idx = (self._current_idx + delta) % len(self._tracks)
            self.video.open(self._tracks[self._current_idx].path)

    def _set_repeat_label_and_tooltip(self, mode: str):
        # Normalize
        mode = (mode or 'all').lower()
        if mode == 'one':
            self.btn_repeat.setText('Repeat: 1')
            self.btn_repeat.setToolTip('Repeats the current track continuously.')
        elif mode == 'random':
            self.btn_repeat.setText('Repeat: random')
            self.btn_repeat.setToolTip('Pure random pick each time; repeats possible.')
        elif mode == 'shuffle':
            self.btn_repeat.setText('Repeat: shuffle')
            self.btn_repeat.setToolTip('No repeat until all are played; reshuffles after a full pass. Next/Prev follow this order.')
        else:
            self.btn_repeat.setText('Repeat: all')
            self.btn_repeat.setToolTip('Repeats the entire playlist and starts again after the last track.')

    
    def _toggle_repeat(self):
        cur = self.repeat_mode()
        if cur == 'all':
            nxt = 'one'
        elif cur == 'one':
            nxt = 'random'
        elif cur == 'random':
            nxt = 'shuffle'
        else:
            nxt = 'all'
        self._set_repeat_label_and_tooltip(nxt)

    def repeat_mode(self) -> str:
        text = self.btn_repeat.text().lower()
        if 'shuffle' in text: return 'shuffle'
        if 'random' in text: return 'random'
        if '1' in text: return 'one'
        return 'all'

    # visuals UI helpers
    def set_meta(self, title: str, artist: str, album: str, cover_pm: Optional[QPixmap]):
        # NOTE: meta header is drawn over the visual; the card shows playlist only
        pass

    def set_modes(self, modes: List[str]):
        self._visual_modes = modes
        self.cmb_visual.clear()
        for m in modes:
            if m.startswith('viz:'):
                self.cmb_visual.addItem(m[4:])
            else:
                self.cmb_visual.addItem(m.capitalize())
        self.cmb_visual.setCurrentIndex(self._visual_index if self._visual_index < len(modes) else 0)
        # persist state on visual/beats changes
        try:
            self.cmb_visual.currentIndexChanged.connect(lambda _i: getattr(self.parent(), '_music_runtime', None) and self.parent()._music_runtime._persist_state())
            self.cmb_beats.currentIndexChanged.connect(lambda _i: getattr(self.parent(), '_music_runtime', None) and self.parent()._music_runtime._persist_state())
        except Exception:
            pass

    def selected_mode(self) -> Optional[str]:
        if not self._visual_modes:
            return None
        idx = max(0, self.cmb_visual.currentIndex())
        return self._visual_modes[idx]

    def _cycle_visual(self, delta: int):
        if not self._visual_modes:
            return
        self._visual_index = (self.cmb_visual.currentIndex() + delta) % len(self._visual_modes)
        self.cmb_visual.setCurrentIndex(self._visual_index)

    def current_auto_beats(self) -> int:
        try:
            return int(self.cmb_beats.currentText())
        except Exception:
            return 32

    def is_auto(self) -> bool:
        return bool(self.chk_auto.isChecked())

    def visuals_enabled(self) -> bool:
        try:
            return bool(self.btn_visuals.isChecked())
        except Exception:
            return False

    def current_viz_mode(self) -> str:
        txt = self.btn_vizmode.text().lower().strip()
        if txt in ('all','loop'): return 'all'
        if txt in ('1','one'): return 'one'
        return 'random'

    def _toggle_vizmode(self):
        m = self.current_viz_mode()
        if m == 'random':
            self.btn_vizmode.setText('all')
        elif m == 'all':
            self.btn_vizmode.setText('1')
        else:
            self.btn_vizmode.setText('random')

    def crossfade_enabled(self) -> bool:
        try:
            return bool(self.chk_xfade.isChecked())
        except Exception:
            return False

    def crossfade_ms(self) -> int:
        try:
            txt = self.cmb_fade.currentText().lower().replace('s','').strip()
            val = float(txt)
            val = max(1.0, min(4.0, val))
            return int(val * 1000)
        except Exception:
            return 2000
# ---------------- State helpers ----------------

def _load_state() -> dict:
    try:
        if MUSIC_STATE_PATH.exists():
            return json.loads(MUSIC_STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

def _save_state(d: dict):
    try:
        MUSIC_STATE_PATH.write_text(json.dumps(d, indent=2), encoding='utf-8')
    except Exception:
        pass

# ---------------- Runtime ----------------

class MusicRuntime(QObject):
    def __init__(self, video_pane):
        super().__init__(video_pane)
        self.video = video_pane
        self._ui_hidden = False
        self.overlay: Optional[MusicOverlay] = None
        self.visual = VisualEngine(self, bars=48)
        self.visual.frameReady.connect(self._on_visual_frame)
        # engine for outgoing-visual during crossfade (keeps old visual animating)
        self.visual_prev = VisualEngine(self, bars=48)
        self.visual_prev.frameReady.connect(self._on_prev_visual_frame)
        # not started by default; only during fades
        self._xfade_prev_running = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.cover: Optional[QPixmap] = None
        self.meta: TrackInfo = TrackInfo(path=Path('.'))
        self._last_beat_index = -1
        self.an = HybridAnalyzer(getattr(video_pane, 'player', None), bands=48, parent=self)
        self.an.levelsReady.connect(self.visual.inject_levels)
        self.an.levelsReady.connect(self._on_levels)
        self.an.levelsReady.connect(self._on_levels_return_flip)
        self.an.rmsReady.connect(self.visual.inject_rms)
        self.an.rmsReady.connect(self._on_rms)
        # responsive visuals: track label size
        self._last_label_size = QSize(0, 0)
        # kick/beat estimate
        self._ms_per_beat_est = 200
        self._kick_avg = 0.0
        self._kick_last = 0
        # return-flip detector state (separate; don't interfere with existing beat logic)
        self._kick_avg2 = 0.0
        self._kick_last2 = 0
        # silence/return detection state
        self._quiet_since_ms = None  # type: Optional[int]
        self._was_quiet = False
        self._return_cooldown_until_ms = 0
        # tunables (no UI): tweak if needed
        self._silence_gate = 0.30  # RMS threshold for 'quiet'
        self._silence_min_ms = 350  # how long it must stay below gate
        self._return_cooldown_ms = 1100  # min gap between return-triggered switches

    
        # crossfade state
        self._last_visual_img = None
        self._xfade_prev_img = None
        self._xfade_active = False
        self._xfade_t0 = 0.0
        self._xfade_ms = 2000

        # shuffle-bag for visuals random mode (non-repeating)
        self._viz_bag = []
        self._viz_bag_pos = -1
    def _viz_rebuild_bag(self):
        """(Re)build the visual shuffle bag excluding the current visual to avoid repeats before a full pass."""
        if not self.overlay:
            self._viz_bag = []
            self._viz_bag_pos = -1
            return
        try:
            count = len(self.overlay._visual_modes)
            cur = self.overlay.cmb_visual.currentIndex()
            idxs = [i for i in range(count) if i != cur]
            import random
            random.shuffle(idxs)
            self._viz_bag = idxs
            self._viz_bag_pos = -1
        except Exception:
            # be safe; clear bag if something goes wrong
            self._viz_bag = []
            self._viz_bag_pos = -1

    def _viz_next_random_index(self) -> int:
        """Return next visual index from the shuffle bag; rebuild if needed."""
        if not self.overlay:
            return -1
        try:
            count = len(self.overlay._visual_modes)
        except Exception:
            count = 0
        if count <= 1:
            try:
                return self.overlay.cmb_visual.currentIndex()
            except Exception:
                return -1
        # Rebuild if bag empty/invalid
        if not isinstance(self._viz_bag, list) or not self._viz_bag or any((i < 0 or i >= count) for i in self._viz_bag):
            self._viz_rebuild_bag()
        # Advance
        self._viz_bag_pos += 1
        if self._viz_bag_pos >= len(self._viz_bag):
            # Completed a full pass; rebuild excluding current to avoid immediate repeat
            self._viz_rebuild_bag()
            self._viz_bag_pos = 0 if self._viz_bag else -1
        if 0 <= self._viz_bag_pos < len(self._viz_bag):
            return self._viz_bag[self._viz_bag_pos]
        try:
            return self.overlay.cmb_visual.currentIndex()
        except Exception:
            return -1
    def start(self, path_for_analysis: Optional[Path] = None):
        try:
            self.visual.set_target(self.video.label.size())
        except Exception:
            pass
        self.visual.start(30)
        self.timer.start(33)
        self.an.start()
        if path_for_analysis:
            self.an.set_file(Path(path_for_analysis))

    def stop(self):
        self.timer.stop()
        self.visual.stop()
        self.an.stop()
    def _on_prev_visual_frame(self, img: QImage):
        """Capture live frames from the outgoing visual during crossfade."""
        try:
            if getattr(self, "_xfade_active", False):
                self._xfade_prev_img = img.copy()
        except Exception:
            pass



    def _draw_header_once(self):
        try:
            w = max(64, self.video.label.width())
            h = max(64, self.video.label.height())
            base = QImage(w, h, QImage.Format_RGBA8888)
            base.fill(QColor(5, 5, 7, 255))
            p = QPainter(base)
            try:
                p.fillRect(QRect(18, 18, min(520, int(w * 0.7)), 148), QColor(0, 0, 0, 150))
                if self.cover and not self.cover.isNull():
                    cpm = self.cover.scaled(QSize(110, 110), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    p.drawPixmap(26, 26, cpm)
                p.setPen(QColor(255, 255, 255))
                font = QFont(); font.setPointSize(14); font.setBold(True); p.setFont(font)
                p.drawText(QRect(160, 28, w - 180, 28), Qt.AlignLeft | Qt.AlignVCenter, self.meta.title or self.meta.path.name)
                font = QFont(); font.setPointSize(11); p.setFont(font)
                if self.meta.artist:
                    p.drawText(QRect(160, 60, w - 180, 24), Qt.AlignLeft | Qt.AlignVCenter, self.meta.artist)
                if self.meta.album:
                    p.drawText(QRect(160, 86, w - 180, 24), Qt.AlignLeft | Qt.AlignVCenter, self.meta.album)
            finally:
                p.end()
            self.video.label.setPixmap(QPixmap.fromImage(base))
        except Exception:
            pass

    def _on_visual_frame(self, img: QImage):
        w = max(64, self.video.label.width())
        h = max(64, self.video.label.height())
        base = QImage(w, h, QImage.Format_RGBA8888)
        base.fill(QColor(5, 5, 7, 255))
        p = QPainter(base)
        try:
            pm = None
            if self.overlay is None or self.overlay.visuals_enabled():
                pm = QPixmap.fromImage(img)
                pm = pm.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Crossfade blend if active
                cx = (w - pm.width()) // 2
                cy = (h - pm.height()) // 2
                if getattr(self, "_xfade_active", False) and self._xfade_prev_img is not None and (self.overlay and self.overlay.visuals_enabled()):
                    try:
                        prog = max(0.0, min(1.0, (((_time.time() - float(self._xfade_t0)) * 1000.0) / max(1, int(self._xfade_ms)))))
                    except Exception:
                        prog = 1.0
                    prev_pm = QPixmap.fromImage(self._xfade_prev_img)
                    prev_pm = prev_pm.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    p.save()
                    p.setOpacity(1.0 - prog)
                    p.drawPixmap((w - prev_pm.width()) // 2, (h - prev_pm.height()) // 2, prev_pm)
                    p.restore()
                    p.save()
                    p.setOpacity(prog)
                    p.drawPixmap(cx, cy, pm)
                    p.restore()
                    if prog >= 1.0:
                        self._xfade_active = False
                        self._xfade_prev_img = None
                        # stop the outgoing-visual engine
                        try:
                            if getattr(self, "_xfade_prev_running", False):
                                self.visual_prev.stop()
                                self.visual_prev.set_enabled(False)
                                self._xfade_prev_running = False
                        except Exception:
                            pass
                else:
                    p.drawPixmap(cx, cy, pm)
            if not getattr(self, '_ui_hidden', False):
                p.fillRect(QRect(18, 18, min(520, int(w * 0.7)), 148), QColor(0, 0, 0, 150))
                if self.cover and not self.cover.isNull():
                    cpm = self.cover.scaled(QSize(110, 110), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    p.drawPixmap(26, 26, cpm)
                p.setPen(QColor(255, 255, 255))
                font = QFont()
                font.setPointSize(14)
                font.setBold(True)
                p.setFont(font)
                p.drawText(QRect(160, 28, w - 180, 28), Qt.AlignLeft | Qt.AlignVCenter, self.meta.title or self.meta.path.name)
                font = QFont()
                font.setPointSize(11)
                p.setFont(font)
                if self.meta.artist:
                    p.drawText(QRect(160, 60, w - 180, 24), Qt.AlignLeft | Qt.AlignVCenter, self.meta.artist)
                if self.meta.album:
                    p.drawText(QRect(160, 86, w - 180, 24), Qt.AlignLeft | Qt.AlignVCenter, self.meta.album)
        finally:
            p.end()
        self.video.label.setPixmap(QPixmap.fromImage(base))
        try:
            self._last_visual_img = img.copy()
        except Exception:
            pass

    def _on_levels(self, bands: list):
        try:
            if not bands:
                return
            take = max(3, min(6, len(bands) // 6))
            low = sum(bands[:take]) / max(1, take)
            self._kick_avg = 0.98 * self._kick_avg + 0.1 * low
            pos = int(self.video.player.position()) if hasattr(self.video, 'player') else int(_time.time() * 1000)
            if low > self._kick_avg * 1.5 and (pos - self._kick_last) > 150:
                if self._kick_last > 0:
                    ib = pos - self._kick_last
                    if 250 <= ib <= 1500:
                        self._ms_per_beat_est = int(0.7 * self._ms_per_beat_est + 0.3 * ib)
                self._kick_last = pos
        except Exception:
            pass

    def _on_rms(self, rms: float):
        try:
            if hasattr(self.video, 'player') and hasattr(self.video.player, 'position'):
                pos = int(self.video.player.position())
            else:
                import time as _time
                pos = int(_time.time() * 1000)
            gate = float(getattr(self, "_silence_gate", 0.05))
            min_ms = int(getattr(self, "_silence_min_ms", 750))
            if float(rms) < gate:
                if getattr(self, "_quiet_since_ms", None) is None:
                    self._quiet_since_ms = pos
                else:
                    if (pos - int(self._quiet_since_ms)) >= max(0, min_ms):
                        self._was_quiet = True
            else:
                self._quiet_since_ms = None
        except Exception:
            pass

    def _on_levels_return_flip(self, bands: list):
        try:
            if not bands:
                return
            take = max(3, min(6, len(bands) // 6))
            low = sum(bands[:take]) / max(1, take)
            # separate EMA/guard so we don't affect the main detector
            self._kick_avg2 = 0.9 * getattr(self, "_kick_avg2", 0.0) + 0.1 * low
            if hasattr(self.video, 'player') and hasattr(self.video.player, 'position'):
                pos = int(self.video.player.position())
            else:
                import time as _time
                pos = int(_time.time() * 1000)
            if low > self._kick_avg2 * 1.8 and (pos - getattr(self, "_kick_last2", 0)) > 180:
                self._kick_last2 = pos
                now_ms = pos
                cooldown_until = int(getattr(self, "_return_cooldown_until_ms", 0))
                if getattr(self, "_was_quiet", False) and now_ms >= cooldown_until:
                    self._cycle_visual()
                    self._was_quiet = False
                    self._return_cooldown_until_ms = now_ms + int(getattr(self, "_return_cooldown_ms", 1500))
        except Exception:
            pass

    def _update(self):
        # responsive visuals target
        try:
            cur_size = self.video.label.size()
            if cur_size != self._last_label_size:
                self.visual.set_target(cur_size)
                self._last_label_size = QSize(cur_size)
        except Exception:
            pass

        # beat-based auto visual switching
        bpm = int(self.meta.bpm or 0)
        if bpm > 0:
            ms_per_beat = 60000 // max(1, bpm)
        else:
            ms_per_beat = max(250, min(1200, int(self._ms_per_beat_est or 200)))
        try:
            pos = int(self.video.player.position())
        except Exception:
            pos = 0
        beat_index = pos // ms_per_beat
        if self.overlay and self.overlay.visuals_enabled() and (self.overlay.current_viz_mode() in ('all','random') or self.overlay.is_auto()):
            N = max(1, self.overlay.current_auto_beats())
            if beat_index != self._last_beat_index and (beat_index % N) == 0 and beat_index > 0:
                self._cycle_visual()
        self._last_beat_index = beat_index

        if self.overlay:
            mode = self.overlay.selected_mode()
            if mode:
                self.visual.set_mode(mode)

        # auto-advance / repeat
        try:
            dur = int(self.video.player.duration())
        except Exception:
            dur = 0
        end_guard_ms = 800
        if dur > 0 and pos >= max(0, dur - end_guard_ms):
            if not getattr(self, '_end_fired', False):
                self._end_fired = True
                mode = self.overlay.repeat_mode() if self.overlay else 'all'
                if mode == 'one':
                    try:
                        self.video.player.setPosition(0)
                        self.video.player.play()
                    except Exception:
                        pass
                elif mode == 'random' and self.overlay and self.overlay._tracks:
                    import random
                    idx = random.randrange(0, len(self.overlay._tracks))
                    self.overlay._current_idx = idx
                    self.video.open(self.overlay._tracks[idx].path)
                else:  # all
                    if self.overlay:
                        try:
                            self._advance_playlist(+1)
                        except Exception:
                            pass
        else:
            self._end_fired = False

    def _advance_playlist(self, delta: int):
        try:
            self.overlay._jump(delta)
        except Exception:
            pass

    def _cycle_visual(self):
        if not self.overlay:
            return
        count = len(self.overlay._visual_modes)
        if count == 0:
            return
        mode = self.overlay.current_viz_mode()
        if mode == 'one':
            return
        if mode == 'random':
            nxt = self._viz_next_random_index()
            if nxt >= 0:
                self.overlay._visual_index = nxt
        else:  # all/loop
            self.overlay._visual_index = (self.overlay.cmb_visual.currentIndex() + 1) % count
        try:
            self._start_crossfade()
        except Exception:
            pass
        self.overlay.cmb_visual.setCurrentIndex(self.overlay._visual_index)

    def _start_crossfade(self):
        try:
            if not self.overlay or not self.overlay.visuals_enabled() or not self.overlay.crossfade_enabled():
                return
            # Seed with last image so first blended frame has content
            if getattr(self, "_last_visual_img", None) is not None:
                self._xfade_prev_img = self._last_visual_img
            # Configure the 'previous' engine to the current visual + size
            try:
                cur_size = self.video.label.size()
                self.visual_prev.set_target(cur_size)
            except Exception:
                pass
            try:
                prev_mode = self.overlay.selected_mode() or 'spectrum'
                self.visual_prev.set_mode(prev_mode)
            except Exception:
                pass
            try:
                # Connect analyzer feeds (idempotent connections are fine in Qt)
                self.an.levelsReady.connect(self.visual_prev.inject_levels)
                self.an.rmsReady.connect(self.visual_prev.inject_rms)
            except Exception:
                pass
            # Start the previous engine so it keeps animating during the fade
            try:
                self.visual_prev.set_enabled(True)
                self.visual_prev.start(30)
                self._xfade_prev_running = True
            except Exception:
                pass
            # Start fade clock
            self._xfade_t0 = _time.time()
            self._xfade_ms = int(self.overlay.crossfade_ms())
            self._xfade_active = True
        except Exception:
            self._xfade_active = False
        self._persist_state()


    def set_overlay(self, overlay: MusicOverlay):
        self.overlay = overlay
        self.overlay.set_modes(self.visual.available_modes())
        # visuals enabled toggle
        try:
            self.visual.set_enabled(self.overlay.visuals_enabled())
            self.overlay.btn_visuals.toggled.connect(self.visual.set_enabled)
            self.overlay.btn_visuals.toggled.connect(lambda _c: self._persist_state())
            # update button label text
            self.overlay.btn_visuals.toggled.connect(lambda c: self.overlay.btn_visuals.setText('Visuals off' if c else 'Visuals on'))
        except Exception:
            pass
        # load crossfade settings
        try:
            st = _load_state()
            cf_on = bool(st.get('xfade_on', True))
            cf_ms = int(st.get('xfade_ms', 2000))
            try:
                self.overlay.chk_xfade.setChecked(cf_on)
            except Exception:
                pass
            try:
                opts = ['1.0s','1.5s','2.0s','3.0s','4.0s']
                sec = max(1.0, min(4.0, cf_ms/1000.0))
                idx = min(range(len(opts)), key=lambda i: abs(float(opts[i][:-1]) - sec))
                self.overlay.cmb_fade.setCurrentIndex(idx)
            except Exception:
                pass
        except Exception:
            pass

        # Persist current viz mode (1/all/random) when user toggles the mode button

        # Persist current viz mode (1/all/random) and crossfade changes; also hook bag rebuild
        try:
            self.overlay.btn_vizmode.clicked.connect(lambda: (self._persist_state(), self._viz_rebuild_bag()))
            self.overlay.chk_xfade.toggled.connect(lambda _c: self._persist_state())
            self.overlay.cmb_fade.currentIndexChanged.connect(lambda _i: self._persist_state())
            # rebuild bag when user manually picks a visual
            self.overlay.cmb_visual.currentIndexChanged.connect(lambda _i: self._viz_rebuild_bag())
        except Exception:
            pass
        # initialize bag once overlay is wired
        self._viz_rebuild_bag()

        # load state
        st = _load_state()
        try:
            vis_on = bool(st.get('visuals_on', False))
            # OVERRIDE: always start OFF on app launch (analysis still runs when playback starts)
            self.overlay.btn_visuals.setChecked(False)
            self.visual.set_enabled(False)
            last_mode = st.get('visual_mode_name')
            if last_mode:
                # find index
                try:
                    if last_mode.startswith('viz:'):
                        name = last_mode[4:]
                    else:
                        name = last_mode
                    for i, m in enumerate(self.overlay._visual_modes):
                        if m == last_mode or (m.startswith('viz:') and m[4:] == name):
                            self.overlay._visual_index = i
                            self.overlay.cmb_visual.setCurrentIndex(i)
                            break
                except Exception:
                    pass
            beats = st.get('auto_beats')
            if beats and str(beats) in ['8','16','32','64']:
                idx = ['8','16','32','64'].index(str(beats))
                self.overlay.cmb_beats.setCurrentIndex(idx)
            mode = st.get('auto_mode')
            if mode in ('loop', 'all'):
                self.overlay.btn_vizmode.setText('all')
            elif mode == 'one':
                self.overlay.btn_vizmode.setText('1')
            else:
                self.overlay.btn_vizmode.setText('random')
        except Exception:
            pass

    def _persist_state(self):
        st = _load_state()
        try:
            st['visuals_on'] = bool(self.overlay.btn_visuals.isChecked())
            st['visual_mode_name'] = self.overlay.selected_mode() or ''
            st['auto_beats'] = self.overlay.cmb_beats.currentText()
            st['auto_mode'] = self.overlay.current_viz_mode()
        except Exception:
            pass
        
        # crossfade settings
        try:
            st['xfade_on'] = bool(self.overlay.chk_xfade.isChecked())
            st['xfade_ms'] = int(self.overlay.crossfade_ms())
        except Exception:
            pass
        _save_state(st)

    def set_cover(self, pm: Optional[QPixmap]):
        self.cover = pm

    def set_meta(self, meta: TrackInfo):
        self.meta = meta
        try:
            self._draw_header_once()
        except Exception:
            pass

# ---------------- Music teardown helper ----------------

def _teardown_music(video_pane):
    """Fully dispose the music runtime and overlay so they never pop back up."""
    try:
        rt = getattr(video_pane, '_music_runtime', None)
        if rt:
            # Stop periodic updates
            try:
                rt.stop()
            except Exception:
                pass
            # Stop visual engine and delete
            try:
                if getattr(rt, 'visual', None):
                    try:
                        rt.visual.stop()
                    except Exception:
                        pass
                    try:
                        rt.visual.deleteLater()
                    except Exception:
                        pass
            except Exception:
                pass
            # Stop analyzer, detach probe, kill worker thread
            try:
                if getattr(rt, 'an', None):
                    an = rt.an
                    try:
                        an.stop()
                    except Exception:
                        pass
                    try:
                        if getattr(an, '_probe', None):
                            try:
                                an._probe.setSource(None)
                            except Exception:
                                pass
                            try:
                                an._probe.deleteLater()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        if getattr(an, '_worker', None):
                            try:
                                an._worker.requestInterruption()
                            except Exception:
                                pass
                            try:
                                an._worker.quit()
                            except Exception:
                                pass
                            try:
                                an._worker.wait(150)
                            except Exception:
                                pass
                            try:
                                an._worker.deleteLater()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        an.deleteLater()
                    except Exception:
                        pass
            except Exception:
                pass
            # Stop and delete runtime timer
            try:
                if getattr(rt, 'timer', None):
                    try:
                        rt.timer.stop()
                    except Exception:
                        pass
                    try:
                        rt.timer.deleteLater()
                    except Exception:
                        pass
            except Exception:
                pass
            # Finally delete runtime object
            try:
                rt.deleteLater()
            except Exception:
                pass
    except Exception:
        pass

    # Overlay teardown
    try:
        ov = getattr(video_pane, '_music_overlay', None)
        if ov:
            try:
                # Remove event filters so resize/show won't revive it
                try:
                    video_pane.removeEventFilter(ov)
                except Exception:
                    pass
                try:
                    if hasattr(video_pane, 'label') and getattr(video_pane, 'label'):
                        video_pane.label.removeEventFilter(ov)
                except Exception:
                    pass
                try:
                    if getattr(ov, 'video', None):
                        ov.video.removeEventFilter(ov)
                except Exception:
                    pass
                try:
                    ov.removeEventFilter(ov)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                ov.hide()
            except Exception:
                pass
            try:
                ov.deleteLater()
            except Exception:
                pass
    except Exception:
        pass

    # Drop references
    try:
        video_pane._music_runtime = None
    except Exception:
        pass
    try:
        video_pane._music_overlay = None
    except Exception:
        pass


# ---------------- Wiring ----------------

def wire_to_videopane(VideoPaneClass):
    if getattr(VideoPaneClass, '_music_wired_full', False):
        return
    VideoPaneClass._music_wired_full = True
    orig_open = VideoPaneClass.open

    def open_wrapper(self, path):
        p = Path(str(path))
        ext = p.suffix.lower()
        if ext in AUDIO_EXTS:
            # Remember the folder of the opened track for next Add…
            try:
                st = _load_state()
                st['last_add_dir'] = str(p.parent)
                _save_state(st)
            except Exception:
                pass
            result = orig_open(self, path)  # keep existing player logic
            # runtime
            if not hasattr(self, '_music_runtime') or self._music_runtime is None:
                self._music_runtime = MusicRuntime(self)
            rt = self._music_runtime
            # overlay (wire once; don't reset visuals each track)
            if (not hasattr(self, '_music_overlay')) or self._music_overlay is None:
                self._music_overlay = MusicOverlay(self, parent=self)
                rt.set_overlay(self._music_overlay)
            else:
                if self._music_overlay.parent() is not self:
                    self._music_overlay.setParent(self)
                self._music_overlay.show()
                try:
                    # Ensure overlay is active for audio
                    self._music_overlay.suspended = False
                except Exception:
                    pass
                # Reattach event filters so overlay can track parent/label events again
                try:
                    try:
                        self.installEventFilter(self._music_overlay)
                    except Exception:
                        pass
                    if hasattr(self, 'label') and getattr(self, 'label', None):
                        try:
                            self.label.installEventFilter(self._music_overlay)
                        except Exception:
                            pass
                    try:
                        self._music_overlay.installEventFilter(self._music_overlay)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Make sure it is visible and above content
                try:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, self._music_overlay.show)
                    QTimer.singleShot(0, self._music_overlay.raise_)
                except Exception:
                    try:
                        self._music_overlay.raise_()
                    except Exception:
                        pass

                self._music_overlay._reposition()
                self._music_overlay.raise_()

            # tags/cover
            tags = _read_all_tags(p)
            title = tags.get('title') or tags.get('TITLE') or p.stem
            artist = tags.get('artist') or tags.get('ARTIST') or ''
            album = tags.get('album') or tags.get('ALBUM') or ''
            bpm = 0
            for key in ('TBPM', 'bpm', 'BPM', 'tbpm'):
                v = tags.get(key)
                if v:
                    try:
                        bpm = int(float(str(v).strip()))
                        break
                    except Exception:
                        pass
            if bpm <= 0:
                bpm = 0  # allow detector to estimate
            meta = TrackInfo(path=p, title=title, artist=artist, album=album, bpm=bpm)
            cover_pm = _extract_cover(p)

            rt.set_cover(cover_pm)
            rt.set_meta(meta)
            try:
                rt._cycle_visual()
            except Exception:
                pass

            # ensure the file appears in the playlist and is selected
            try:
                self._music_overlay.ensure_in_playlist(p)
            except Exception:
                pass
            # Rebind analyzer to the current player in case the VideoPane replaced it
            try:
                if getattr(rt, 'an', None):
                    try:
                        rt.an.rebind_player(getattr(self, 'player', None))
                    except Exception:
                        pass
            except Exception:
                pass
            rt.start(path_for_analysis=p)  # kick off hybrid preanalysis
            # one-time 100ms forward scroll on the very first audio after app opens
            try:
                from PySide6.QtCore import QTimer
                global _INITIAL_SCROLL_ARMED, _INITIAL_SCROLL_DONE
                if not _INITIAL_SCROLL_DONE and not _INITIAL_SCROLL_ARMED:
                    _INITIAL_SCROLL_ARMED = True
                    def _arm_scroll():
                        # wait until duration is known to avoid a no-op seek
                        try:
                            pl = getattr(self, 'player', None)
                            dur = int(pl.duration()) if pl else 0
                        except Exception:
                            dur = 0
                        if dur <= 0:
                            QTimer.singleShot(20, _arm_scroll)
                            return
                        # start at zero then nudge forward a tiny bit later
                        try:
                            pl.setPosition(0)
                        except Exception:
                            pass
                        def _nudge():
                            try:
                                d = int(pl.duration()) if pl else 0
                            except Exception:
                                d = 0
                            tgt = 150
                            if d > 0:
                                tgt = min(150, max(0, d - 2))
                            try:
                                pl.setPosition(tgt)
                            except Exception:
                                pass
                            # mark done so we never do this again
                            globals()['_INITIAL_SCROLL_DONE'] = True
                        QTimer.singleShot(50, _nudge)
                    QTimer.singleShot(60, _arm_scroll)
            except Exception:
                pass
            return result
        else:
            # Soft suspend overlay/runtime when opening non-audio (image/video/gif/etc)
            try:
                if getattr(self, "_music_runtime", None):
                    try:
                        self._music_runtime.stop()
                    except Exception:
                        pass
                if getattr(self, "_music_overlay", None):
                    try:
                        # Mark overlay as suspended so its eventFilter won't revive it
                        try:
                            self._music_overlay.suspended = True
                        except Exception:
                            pass
                        # Remove event filters to prevent auto-raise/show on parent resize/show
                        try:
                            try:
                                self.removeEventFilter(self._music_overlay)
                            except Exception:
                                pass
                            if hasattr(self, 'label') and getattr(self, 'label', None):
                                try:
                                    self.label.removeEventFilter(self._music_overlay)
                                except Exception:
                                    pass
                            try:
                                self._music_overlay.removeEventFilter(self._music_overlay)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        self._music_overlay.hide()
                    except Exception:
                        pass
            except Exception:
                pass
            return orig_open(self, path)

    VideoPaneClass.open = open_wrapper

# ----- Hotfix: silence watchdog for flat visuals (EQ goes flat on pause/stop or no audio) -----
try:
    import time as _time
    from PySide6.QtCore import QTimer as _QTimer
except Exception:
    _QTimer = None

# Extend/monkey-patch HybridAnalyzer with silence handling
def _HybridAnalyzer_mark_silent(self):
    # Zero RMS / peaks
    try:
        self._probe_rms = 0.0
    except Exception:
        pass
    try:
        self._probe_levels = None
    except Exception:
        pass
    # Try to zero any common arrays used by visuals
    for _name in ("_bars","_bars_smooth","_spectrum","_spectrum_smooth","_eq_bins","_levels","_levels_smooth"):
        try:
            _arr = getattr(self, _name, None)
            if isinstance(_arr, (list, tuple)):
                setattr(self, _name, [0.0] * len(_arr))
            elif hasattr(_arr, "fill"):
                _arr.fill(0.0)
        except Exception:
            pass
    # Nudge any known update hooks to repaint
    for _cb in ("_emit_levels","_emit_spectrum","_update_visuals","_tick","_repaint"):
        try:
            _f = getattr(self, _cb, None)
            if callable(_f):
                _f()
        except Exception:
            pass

def _HybridAnalyzer_on_player_state(self, *args):
    # Determine current state; treat anything not "Playing" as silence
    _state = None
    try:
        if hasattr(self, "player") and hasattr(self.player, "playbackState"):
            _state = self.player.playbackState()
        elif hasattr(self, "player") and hasattr(self.player, "state"):
            _state = self.player.state()
    except Exception:
        _state = None
    # Qt6 QMediaPlayer.PlaybackState: 0=Stopped, 1=Playing, 2=Paused
    try:
        if _state is None or int(_state) != 1:
            _HybridAnalyzer_mark_silent(self)
    except Exception:
        _HybridAnalyzer_mark_silent(self)

def _HybridAnalyzer_check_silence(self):
    # If no buffers recently, flatten
    try:
        now = _time.monotonic()
        last = getattr(self, "_last_buf_ts", None)
        # Default: if older than 0.6s, consider silence (covers pause and device stalls)
        timeout_s = getattr(self, "_silence_timeout_s", 0.6)
        if last is None or (now - last) > timeout_s:
            _HybridAnalyzer_mark_silent(self)
    except Exception:
        pass

def _HybridAnalyzer_hook_buf_timestamp(self):
    # One-time wrap of _on_buf to update _last_buf_ts
    try:
        if getattr(self, "_buf_wrapped", False):
            return
        _orig = getattr(self, "_on_buf", None)
        if callable(_orig):
            def _wrapped(buf, *a, **kw):
                try:
                    self._last_buf_ts = _time.monotonic()
                except Exception:
                    pass
                return _orig(buf, *a, **kw)
            self._on_buf = _wrapped
            self._buf_wrapped = True
    except Exception:
        pass

def _HybridAnalyzer_install_silence_watchdog(self):
    # Set timeout configurable
    try:
        if not hasattr(self, "_silence_timeout_s"):
            self._silence_timeout_s = 0.6
    except Exception:
        pass
    # Start periodic timer
    try:
        if _QTimer is not None:
            if getattr(self, "_silence_timer", None) is None:
                self._silence_timer = _QTimer(self)
                try:
                    self._silence_timer.setInterval(250)
                except Exception:
                    pass
                try:
                    self._silence_timer.timeout.connect(lambda: _HybridAnalyzer_check_silence(self))
                except Exception:
                    pass
                try:
                    self._silence_timer.start()
                except Exception:
                    pass
    except Exception:
        pass
    # Ensure we timestamp every audio buffer
    _HybridAnalyzer_hook_buf_timestamp(self)

def _HybridAnalyzer_rebind_plus(self, player):
    # Delegate to any existing rebind_player (if defined), then add silence hooks
    try:
        _base = getattr(HybridAnalyzer, "__rebind_base", None)
    except Exception:
        _base = None
    if callable(_base):
        try:
            _base(self, player)
        except Exception:
            pass
    else:
        # Fallback: just set refs and try to connect probes, similar to prior hotfix
        try:
            self.player = player
        except Exception:
            pass

    # Hook player state → flatten on pause/stop
    try:
        if player is not None:
            # Ensure storage for connection refs
            if not hasattr(self, "_fv_state_slot"):
                try:
                    self._fv_state_slot = None
                except Exception:
                    pass
            if not hasattr(self, "_fv_legacy_state_slot"):
                try:
                    self._fv_legacy_state_slot = None
                except Exception:
                    pass
            if not hasattr(self, "_fv_connected_player"):
                try:
                    self._fv_connected_player = None
                except Exception:
                    pass
            # Disconnect from previous player (if different)
            try:
                _prev = self._fv_connected_player
            except Exception:
                _prev = None
            try:
                if _prev is not None and _prev is not player:
                    try:
                        if hasattr(_prev, "playbackStateChanged") and self._fv_state_slot is not None:
                            try:
                                _prev.playbackStateChanged.disconnect(self._fv_state_slot)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        if hasattr(_prev, "stateChanged") and self._fv_legacy_state_slot is not None:
                            try:
                                _prev.stateChanged.disconnect(self._fv_legacy_state_slot)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            # Connect to current player's state signal (Qt6 or legacy)
            try:
                if hasattr(player, "playbackStateChanged"):
                    try:
                        if self._fv_state_slot is None:
                            self._fv_state_slot = (lambda *_: _HybridAnalyzer_on_player_state(self))
                    except Exception:
                        pass
                    player.playbackStateChanged.connect(self._fv_state_slot, Qt.ConnectionType.UniqueConnection)
                elif hasattr(player, "stateChanged"):
                    try:
                        if self._fv_legacy_state_slot is None:
                            self._fv_legacy_state_slot = (lambda *_: _HybridAnalyzer_on_player_state(self))
                    except Exception:
                        pass
                    player.stateChanged.connect(self._fv_legacy_state_slot, Qt.ConnectionType.UniqueConnection)
            except Exception:
                pass
            # Remember which player we're connected to
            try:
                self._fv_connected_player = player
            except Exception:
                pass
    except Exception:
        pass
    # Install periodic silence watchdog
    _HybridAnalyzer_install_silence_watchdog(self)
    # Consider immediately flattening if not in playing state
    _HybridAnalyzer_on_player_state(self)

# Preserve any existing rebind_player so we can call it first
try:
    if hasattr(HybridAnalyzer, "rebind_player"):
        HybridAnalyzer.__rebind_base = HybridAnalyzer.rebind_player  # type: ignore[attr-defined]
    HybridAnalyzer.rebind_player = _HybridAnalyzer_rebind_plus  # type: ignore[attr-defined]
    HybridAnalyzer.mark_silent = _HybridAnalyzer_mark_silent  # optional public helper
except Exception:
    pass