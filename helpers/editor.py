
# helpers/editor.py — enhanced build with trim handles, ghost snap, markers, ripple delete, media preview
from __future__ import annotations

import os, sys, json, subprocess, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

from PySide6.QtCore import Qt, QRectF, QPointF, QSize, QMimeData, QTimer, QEvent, Signal, QThread, QRect
from PySide6.QtGui import (QAction, QKeySequence, QIcon, QPixmap, QDrag, QPainter, QPen, QColor, QBrush,
                           QStandardItemModel, QStandardItem, QCursor, QImage, QImageReader, QPolygonF)
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QListWidget,
                               QListWidgetItem, QMenu, QToolButton, QComboBox, QSpinBox, QLineEdit,
                               QScrollArea, QFrame, QMessageBox, QSlider, QSizePolicy, QDialog,
                               QScrollBar,
                               QStyle,
                               QDialogButtonBox, QFormLayout, QAbstractItemView, QApplication, QTabWidget, QInputDialog, QColorDialog, QPushButton, QSplitter)

# Optional multimedia preview imports (Qt6+)
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_QTMULTI = True
except Exception:
    HAS_QTMULTI = False

# ---------- utils ----------
def _try_exec(c):
    try:
        subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
        return str(c)
    except Exception:
        return None

def ffmpeg_path():
    for c in [Path(".")/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), "ffmpeg"]:
        p = _try_exec(c)
        if p: return p
    return "ffmpeg"

def ffprobe_path():
    for c in [Path(".")/"bin"/("ffprobe.exe" if os.name=="nt" else "ffprobe"), "ffprobe"]:
        p = _try_exec(c)
        if p: return p
    return "ffprobe"

VIDEO = {".mp4",".mov",".mkv",".avi",".webm",".m4v",".mpg",".mpeg",".wmv",".ts",".m2ts"}
AUDIO = {".mp3",".wav",".aac",".m4a",".flac",".ogg",".opus",".aiff"}
IMAGE = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".gif"}

def media_kind(p: Path) -> str:
    s = p.suffix.lower()
    if s in VIDEO: return "video"
    if s in AUDIO: return "audio"
    if s in IMAGE: return "image"
    if s in {".txt",".srt",".ass",".vtt"}: return "text"
    return "unknown"

# ---------- data ----------
@dataclass
class MediaItem:
    id: str
    path: Path
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Clip:
    media_id: str
    start_ms: int = 0
    duration_ms: int = 1000
    gain_db: float = 0.0
    text: Optional[str] = None
    speed: float = 1.0
    rotation_deg: int = 0
    muted: bool = False
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    group_id: Optional[int] = None  # for grouping
    label_color: Optional[str] = None  # per-clip color label

@dataclass
class Track:
    name: str
    type: str  # "video","image","text","audio","any"
    enabled: bool = True
    clips: List[Clip] = field(default_factory=list)
    mute: bool = False
    solo: bool = False
    locked: bool = False
    color: str = "#3c3c3c"  # label color

@dataclass
class Project:
    version: int = 1
    media: Dict[str, MediaItem] = field(default_factory=dict)
    tracks: List[Track] = field(default_factory=list)
    fps: int = 30
    width: int = 1280
    height: int = 720
    sample_rate: int = 48000

    def to_json(self) -> Dict[str,Any]:
        return {
            "version": self.version,
            "media": {k: {"id": v.id, "path": str(v.path), "kind": v.kind, "meta": v.meta} for k,v in self.media.items()},
            "tracks": [{
                "name": tr.name, "type": tr.type, "enabled": tr.enabled,
                "mute": tr.mute, "solo": tr.solo, "locked": tr.locked, "color": tr.color,
                "clips": [asdict(c) for c in tr.clips]
            } for tr in self.tracks],
            "fps": self.fps, "width": self.width, "height": self.height, "sample_rate": self.sample_rate
        }

    @staticmethod
    def from_json(d: Dict[str,Any]) -> "Project":
        p = Project()
        p.version = int(d.get("version",1))
        p.fps = int(d.get("fps",30)); p.width = int(d.get("width",1280)); p.height=int(d.get("height",720))
        p.sample_rate = int(d.get("sample_rate",48000))
        p.media = {k: MediaItem(id=v["id"], path=Path(v["path"]), kind=v.get("kind","unknown"), meta=v.get("meta",{}))
                   for k,v in d.get("media",{}).items()}
        p.tracks = []
        for tr in d.get("tracks", []):
            t = Track(tr.get("name","Track"), tr.get("type","video"), tr.get("enabled",True))
            t.mute = bool(tr.get("mute", False)); t.solo = bool(tr.get("solo", False))
            t.locked = bool(tr.get("locked", False)); t.color = tr.get("color", "#3c3c3c")
            for c in tr.get("clips", []):
                # group_id might be missing in older saves
                if "group_id" not in c: c["group_id"] = None
                t.clips.append(Clip(**c))
            p.tracks.append(t)
        return p

# ---------- thumbnails ----------
class ThumbThread(QThread):
    ready = Signal(str, QPixmap)
    def __init__(self, items, size, parent=None):
        super().__init__(parent)
        self.items = list(items); self.size = size
    def run(self):
        for mid, m in self.items:
            try:
                pm = _gen_thumb(m, self.size)
                if pm: self.ready.emit(mid, pm)
            except Exception: pass

def _gen_thumb(media: MediaItem, size: QSize) -> Optional[QPixmap]:
    w, h = size.width(), size.height()
    def canvas(): pm = QPixmap(w,h); pm.fill(QColor(60,60,60)); return pm
    if media.kind in ("audio","text"): return _placeholder_icon(media.kind, size)
    if media.kind == "image" and media.path.exists():
        try:
            im = QImageReader(str(media.path)).read()
            if not im.isNull():
                im = im.scaled(w,h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = canvas(); p = QPainter(pm); p.drawImage((w-im.width())//2, (h-im.height())//2, im); p.end(); return pm

        except Exception: pass

    if media.kind == "video" and media.path.exists():
        try:
            data = subprocess.check_output([ffmpeg_path(), "-ss","1","-i",str(media.path), "-frames:v","1","-f","image2pipe","-vcodec","png","pipe:1"], stderr=subprocess.DEVNULL)
            im = QImage.fromData(data, "PNG")
            if not im.isNull():
                im = im.scaled(w,h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = canvas(); p = QPainter(pm); p.drawImage((w-im.width())//2, (h-im.height())//2, im); p.end(); return pm
        except Exception: pass
    return _placeholder_icon(media.kind, size)

def _placeholder_icon(kind: str, size: QSize) -> QPixmap:

    w,h = size.width(), size.height()
    pm = QPixmap(w,h)
    pm.fill(QColor(0,0,0,0))  # transparent background so audio looks compact
    p = QPainter(pm); p.setPen(Qt.NoPen)
    # base tile for non-audio kinds
    if kind != "audio":
        p.setBrush(QColor(70,120,200) if kind in ("video","image","text") else QColor(70,200,120))
        p.drawRoundedRect(0,0,w,h,6,6)
        p.setPen(QPen(QColor(255,255,255,220),2))
        if kind=="text":
            p.drawLine(int(w*0.2), int(h*0.25), int(w*0.8), int(h*0.25))
            p.drawLine(int(w*0.5), int(h*0.25), int(w*0.5), int(h*0.75))
        else:
            pts = [QPointF(w*0.35,h*0.25), QPointF(w*0.75,h*0.5), QPointF(w*0.35,h*0.75)]
            p.drawPolygon(QPolygonF(pts))
        p.end(); return pm
    # audio: draw a compact waveform band
    band_h = max(10, int(h*0.30))
    y = int((h-band_h)/2)
    p.setBrush(QColor(70,200,120))
    p.drawRoundedRect(4, y, w-8, band_h, 4,4)
    p.setPen(QPen(QColor(255,255,255,230), 2))
    # stylized 'note' + few bars
    p.drawLine(10, y+band_h-6, 10, y+4)
    p.drawLine(10, y+4, 28, y+2)
    p.drawLine(28, y+2, 28, y+band_h-8)
    for i in range(5):
        bx = int(w*0.45) + i*8
        bh = 6 + (i%3)*4
        p.drawLine(bx, y+band_h-4, bx, y+band_h-4-bh)
    p.end(); return pm

# ---------- media preview dialog ----------
class PreviewDialog(QDialog):
    def __init__(self, media: MediaItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle(media.path.name)
        self.setModal(False)
        try:
            from PySide6.QtCore import Qt as _Qt
            self.setAttribute(_Qt.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.resize(540, 320)
        lay = QVBoxLayout(self)
        # content area
        self.media = media
        self.player = None
        self.audio_out = None
        if HAS_QTMULTI and media.kind in ("video","audio"):
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_out)
            if media.kind == "video":
                vw = QVideoWidget(self); lay.addWidget(vw, 1); self.player.setVideoOutput(vw)
        if self.player is None and media.kind == "image":
            # static image preview
            lab = QLabel("")
            lab.setAlignment(Qt.AlignCenter)
            pm = _gen_thumb(media, QSize(520, 300))
            if pm: lab.setPixmap(pm.scaled(520,300,Qt.KeepAspectRatio,Qt.SmoothTransformation))
            lay.addWidget(lab, 1)
        ctrl = QHBoxLayout(); lay.addLayout(ctrl)
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0,1000)
        ctrl.addWidget(self.btn_play); ctrl.addWidget(self.btn_pause); ctrl.addWidget(self.slider, 1)
        self.btn_play.clicked.connect(self._play); self.btn_pause.clicked.connect(self._pause)
        if self.player:
            # file URL
            from PySide6.QtCore import QUrl
            self.player.setSource(QUrl.fromLocalFile(str(media.path)))
            self.player.playbackStateChanged.connect(self._sync)
            self.player.positionChanged.connect(self._pos_changed)
            self.player.durationChanged.connect(self._dur_changed)
            self.slider.sliderMoved.connect(self._seek)
        else:
            self.btn_play.setEnabled(False); self.btn_pause.setEnabled(False); self.slider.setEnabled(False)

    def _play(self):
        if self.player: self.player.play()

    def _pause(self):
        if self.player: self.player.pause()

    def _sync(self, *_): pass
    def _pos_changed(self, ms): 
        if self.player and self.player.duration()>0 and not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(1000 * ms/max(1,self.player.duration())))
            self.slider.blockSignals(False)
    def _dur_changed(self, ms): pass
    def _seek(self, val):
        if not self.player: return
        dur = max(1, self.player.duration()); pos = int((val/1000.0)*dur)
        self.player.setPosition(pos)

    def closeEvent(self, ev):
        try:
            if self.player:
                self.player.stop()
                # Detach source to release handles
                from PySide6.QtCore import QUrl
                self.player.setSource(QUrl())
        except Exception:
            pass
        try:
            if self.player: self.player.deleteLater()
            if self.audio_out: self.audio_out.deleteLater()
        except Exception:
            pass
        super().closeEvent(ev)
        self.player.setPosition(pos)

# ---------- UI: media strip ----------
class MediaList(QListWidget):
    MIME = "application/x-framevision-media-id"
    requestDelete = Signal(list)  # [media_ids]
    requestPreview = Signal(str)  # media_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setFlow(QListWidget.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setIconSize(QSize(120,64))
        self.setMaximumHeight(120)
        self.setSpacing(4)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setDefaultDropAction(Qt.CopyAction)
        try:
            self.setWordWrap(False)
            self.setUniformItemSizes(True)
        except Exception:
            pass

    def startDrag(self, actions):
        it = self.currentItem()
        if not it: return
        mid = it.data(Qt.UserRole)
        if not mid: return
        mime = QMimeData(); mime.setData(self.MIME, str(mid).encode("utf-8"))
        d = QDrag(self); d.setMimeData(mime)
        d.setPixmap(_placeholder_icon("video", self.iconSize()))
        d.exec(Qt.CopyAction)

    def set_thumb(self, media_id: str, pm: QPixmap):
        for i in range(self.count()):
            it = self.item(i)
            if it and it.data(Qt.UserRole) == media_id:
                it.setIcon(QIcon(pm)); break

    # delete selected thumbnails with Delete key
    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            ids = []
            for it in self.selectedItems():
                mid = it.data(Qt.UserRole)
                if mid: ids.append(mid)
            if ids: self.requestDelete.emit(ids)
            ev.accept(); return
        super().keyPressEvent(ev)

    # double-click preview
    def mouseDoubleClickEvent(self, ev):
        it = self.itemAt(ev.pos())
        if it:
            mid = it.data(Qt.UserRole)
            if mid: self.requestPreview.emit(mid)
        super().mouseDoubleClickEvent(ev)

# ---------- ruler ----------

class TimeRuler(QWidget):
    positionPicked = Signal(int)  # ms
    def __init__(self, parent=None):
        super().__init__(parent)
        self.px_per_s = 100.0
        self.duration_ms = 60000
        self.playhead_ms = 0
        self.fps = 30
        self._markers: List[int] = []  # ms
        self._in_ms: Optional[int] = None
        self._out_ms: Optional[int] = None
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        self._dragging = False
        self._offset_ms = 0

    # --- configuration ---
    def set_scale(self, px: float):
        self.px_per_s = max(0.01, float(px)); self.update()

    def set_duration(self, ms: int):
        self.duration_ms = max(1000, int(ms))
        try: self.update()
        except Exception: pass

    def set_offset(self, ms: int):
        self._offset_ms = max(0, int(ms))
        try: self.update()
        except Exception: pass

    def set_playhead(self, ms: int):
        self.playhead_ms = max(0, int(ms)); self.update()

    def set_markers(self, markers: List[int]):
        self._markers = list(sorted(set(int(m) for m in markers))); self.update()

    def set_fps(self, fps: int):
        self.fps = max(1, int(fps)); self.update()

    def set_in_out(self, in_ms: Optional[int], out_ms: Optional[int]):
        self._in_ms = int(in_ms) if in_ms is not None else None
        self._out_ms = int(out_ms) if out_ms is not None else None
        self.update()

    # --- helpers ---
    def _fmt_tc(self, ms: int) -> str:
        fps = max(1, int(self.fps))
        total_sec = max(0, int(ms)) / 1000.0
        hh = int(total_sec // 3600)
        mm = int((total_sec % 3600) // 60)
        ss = int(total_sec % 60)
        # frames from remainder
        frac = total_sec - int(total_sec)
        ff = int(round(frac * fps)) % fps
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    # --- event handling ---
    
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30,30,35))

        # visible offset in seconds
        off_s = float(self._offset_ms) / 1000.0

        # selection (in/out) shading
        if (self._in_ms is not None) and (self._out_ms is not None) and (self._out_ms > self._in_ms):
            x1 = int(((self._in_ms/1000.0) - off_s) * self.px_per_s)
            x2 = int(((self._out_ms/1000.0) - off_s) * self.px_per_s)
            sel_rect = QRect(min(x1,x2), 0, abs(x2-x1), self.rect().height())
            p.fillRect(sel_rect, QColor(70,120,200,60))

        # playhead
        xph = int(((self.playhead_ms/1000.0) - off_s) * self.px_per_s)
        p.setPen(QPen(QColor(230,80,80), 2))
        p.drawLine(xph, 0, xph, self.rect().height())

        # ticks
        w = self.rect().width()
        # desired px per major tick ~80
        target = 80.0
        step = max(0.1, target / max(1.0, self.px_per_s))
        stops = [0.1,0.2,0.5,1,2,5,10,15,30,60]
        tick = next((s for s in stops if s >= step), 60)
        sub = tick/5.0

        # start at the first sub-tick <= off_s
        import math
        start_sub_idx = int(math.floor(off_s / sub))  # integer index
        max_sub = int(math.ceil((off_s + (w / max(1.0, self.px_per_s))) / sub)) + 2  # draw slightly past right edge

        for i in range(start_sub_idx, max_sub+1):
            t = i * sub
            x = int((t - off_s) * self.px_per_s)
            if x > w:
                break
            if x < -2:
                continue
            # major every 'tick'
            major = (abs((t / tick) - round(t / tick)) < 1e-9)
            p.setPen(QPen(QColor(220,220,220) if major else QColor(140,140,150), 1))
            p.drawLine(x, 0 if major else 12, x, self.rect().height())
            if major:
                label = self._fmt_tc(int(t*1000))
                p.drawText(x+4, 18, label)

        # markers
        p.setPen(QPen(QColor(255,215,0), 2))
        for m in self._markers:
            x = int(((m/1000.0) - off_s) * self.px_per_s)
            p.drawLine(x, 0, x, self.rect().height())

        p.end()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            ms = int(max(0, (ev.position().x() / max(1e-6, self.px_per_s)) * 1000.0))
            # notify scrub start
            parent = self._find_editor()
            if parent is not None:
                parent.transport_request.emit("scrub:start")
            self.positionPicked.emit(ms)
            ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging:
            ms = int(max(0, (ev.position().x() / max(1e-6, self.px_per_s)) * 1000.0))
            self.positionPicked.emit(ms)
            ev.accept(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._dragging and ev.button() == Qt.LeftButton:
            self._dragging = False
            parent = self._find_editor()
            if parent is not None:
                parent.transport_request.emit("scrub:end")
            ev.accept(); return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ms = int(max(0, (ev.position().x() / max(1e-6, self.px_per_s)) * 1000.0))
            self.positionPicked.emit(ms); ev.accept(); return
        super().mouseDoubleClickEvent(ev)

    # zoom with wheel over ruler
    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        parent = self._find_editor()
        if parent is not None:
            if dy > 0: parent._set_zoom(parent.zoom*1.1)
            else: parent._set_zoom(parent.zoom/1.1)
            ev.accept(); return
        super().wheelEvent(ev)

    def _find_editor(self):
        parent = self.parent()
        while parent and not isinstance(parent, EditorPane):
            parent = parent.parent()
        return parent if isinstance(parent, EditorPane) else None
# ---------- timeline widgets ----------
class ClipWidget(QFrame):
    CLIP_MIME = "application/x-framevision-clip"
    HANDLE_W = 12
    HANDLE_HIT_W = 18

    def __init__(self, clip: Clip, media: MediaItem, row: "TrackRow"):
        super().__init__(row)
        self.clip = clip; self.media = media; self.row = row
        self.setFrameShape(QFrame.Panel); self.setFrameShadow(QFrame.Raised); self.setLineWidth(2)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.OpenHandCursor)
        self._press_pos = None; self._dragging = False
        self._resizing = False; self._resize_edge = None  # "L" or "R"
        self._orig_start = 0; self._orig_dur = 0
        self._hover_edge = None
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)  # use contextMenuEvent for reliability

    def contextMenuEvent(self, ev):
        self.row.editor._select_clip(self.clip, additive=False, toggle=False)
        self._show_ctx(ev.globalPos())

    def _show_ctx(self, global_pos):
        ed = self.row.editor; m = QMenu(self)
        # Default Delete: remove clip instance *from timeline* only
        try:
            act_del = m.addAction("Delete", lambda: ed._clip_remove_from_timeline(self.row.track, self.clip))
            try:
                from PySide6.QtGui import QKeySequence
                act_del.setShortcut(QKeySequence.Delete)
            except Exception:
                pass
        except Exception:
            pass
        # Copy/Paste
        m.addAction("Copy", lambda: ed._clip_copy(self.row.track, self.clip))
        act_paste = m.addAction("Paste", lambda: ed._clip_paste(self.row.track, self.mapToParent(self.rect().center()).x(), self.media.kind))
        act_paste.setEnabled(ed._clipboard is not None)
        # Cut
        m.addAction("Play", lambda: self.row.editor._play_clip_from_ruler(self))
        cut = m.addMenu("Cut")
        cut.addAction("Keep left side",  lambda: ed._clip_cut(self.row.track, self.clip, "left"))
        cut.addAction("Keep right side", lambda: ed._clip_cut(self.row.track, self.clip, "right"))
        cut.addAction("Keep both sides (split)", lambda: ed._clip_cut(self.row.track, self.clip, "both"))
        
        # Trim to playhead
        trim = m.addMenu("Trim to playhead")
        trim.addAction("Front → Playhead", lambda: (ed._push_undo(), ed._clip_trim_to_playhead(self.clip, 'L'), ed._refresh_tracks(), ed._update_time_range()))
        trim.addAction("End → Playhead",   lambda: (ed._push_undo(), ed._clip_trim_to_playhead(self.clip, 'R'), ed._refresh_tracks(), ed._update_time_range()))
        trim.addAction("Reset to maximum size", lambda: (ed._push_undo(), ed._clip_reset_to_max(self.clip), ed._refresh_tracks(), ed._update_time_range()))
# Fade
        fade = m.addMenu("Fade"); fi = fade.addMenu("Start"); fo = fade.addMenu("End")
        for label, ms in [("Short (1s)",1000),("Medium (2s)",2000),("Long (3s)",3000)]:
            fi.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, start_ms=ms))
            fo.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, end_ms=ms))
        # Volume
        vol = m.addMenu("Volume"); vol.setEnabled(self.media.kind in ("audio","video"))
        vol.addAction("Mute ON", lambda: ed._clip_set_mute(self.clip, True))
        vol.addAction("Mute OFF", lambda: ed._clip_set_mute(self.clip, False))
        vol.addAction("Set level (dB)…", lambda: ed._clip_set_gain(self.clip))
        # Time
        tm = m.addMenu("Time"); fast = tm.addMenu("Faster"); slow = tm.addMenu("Slower")
        for label, mult in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: fast.addAction(label, lambda mult=mult: ed._clip_speed(self.clip, mult))
        for label, div in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: slow.addAction(label, lambda div=div: ed._clip_speed(self.clip, 1.0/div))
        # Rotate
        rot = m.addMenu("Rotate"); rot.setEnabled(self.media.kind in ("image","video","text"))
        for d in [45,90,180]:
            rot.addAction(f"+{d}°", lambda d=d: ed._clip_rotate(self.clip, +d))
            rot.addAction(f"-{d}°", lambda d=d: ed._clip_rotate(self.clip, -d))
        # Separate sound
        sep = m.addAction("Separate sound from video"); sep.setEnabled(self.media.kind=="video")
        sep.triggered.connect(lambda: ed._clip_separate_audio(self.row.track, self.clip))
        # Remove submenu
        rm = m.addMenu("Remove")
        rm.addAction("From timeline", lambda: ed._clip_remove_from_timeline(self.row.track, self.clip))
        rm.addAction("From project", lambda: ed._clip_remove_from_project(self.row.track, self.clip))
        m.exec(global_pos)


    def _play_from_ruler(self):
        ed = self.row.editor
        try:
            dlg = PreviewDialog(self.media, ed)
            ed._playback_preview = dlg
            dlg.show()
            start_ms = int(getattr(ed.seek_slider, "value", lambda:0)())
            rel = max(0, start_ms - int(self.clip.start_ms))
            if dlg.player:
                try: dlg.player.setPosition(int(rel))
                except Exception: pass
                try: dlg.player.play()
                except Exception: pass
                def _sync(ms):
                    try:
                        g = int(self.clip.start_ms) + int(ms)
                        ed.seek_slider.blockSignals(True)
                        ed.seek_slider.setValue(int(g))
                        ed.seek_slider.blockSignals(False)
                        ed._ensure_playhead_visible()
                    except Exception:
                        pass
                try: dlg.player.positionChanged.connect(_sync)
                except Exception: pass
        except Exception:
            pass

    def paintEvent(self, ev):
        p = QPainter(self)
        # background tinted by per-clip color if set, else track color
        base = QColor(50,90,160) if self.media.kind in ("video","image","text") else QColor(60,160,90)
        try:
            tint_hex = getattr(self.clip, "label_color", None) or self.row.track.color
            tc = QColor(tint_hex)
            bg = QColor((base.red()+tc.red())//2, (base.green()+tc.green())//2, (base.blue()+tc.blue())//2)
        except Exception:
            bg = base
        p.fillRect(self.rect(), bg)
        # selection border
        try:
            if self.row.editor._is_selected(self.clip):
                p.setPen(QPen(QColor(255,220,80), 3)); p.drawRect(self.rect().adjusted(1,1,-2,-2))
        except Exception: pass
        # trim handles (visible area)
        p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,110))
        p.drawRect(0,0,self.HANDLE_W,self.height())
        p.drawRect(self.width()-self.HANDLE_W,0,self.HANDLE_W,self.height())
        # hover cue (slightly wider + brighter)
        if getattr(self, '_hover_edge', None) == 'L':
            p.setBrush(QColor(255,255,255,150)); p.drawRect(0,0,self.HANDLE_W+4,self.height())
        elif getattr(self, '_hover_edge', None) == 'R':
            p.setBrush(QColor(255,255,255,150)); p.drawRect(self.width()-self.HANDLE_W-4,0,self.HANDLE_W+4,self.height())
        # text
        p.setPen(QPen(QColor(255,255,255),1))
        name = self.media.path.name if self.media.kind!="text" else (self.clip.text or "Text")
        p.drawText(6, int(self.rect().height()/2)+5, name)
        p.end()

    def _edge_at_pos(self, posx: int) -> Optional[str]:
        if posx <= int(self.HANDLE_HIT_W): return "L"
        if posx >= int(self.width()-int(self.HANDLE_HIT_W)): return "R"
        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            add = (ev.modifiers() & Qt.ControlModifier) or (ev.modifiers() & Qt.ShiftModifier)
            self.row.editor._select_clip(self.clip, additive=add, toggle=True if add else False)
            ex = ev.position().toPoint().x()
            edge = self._edge_at_pos(ex)
            if edge and not self.row.track.locked:
                self._resizing = True; self._resize_edge = edge
                self._orig_start = int(self.clip.start_ms); self._orig_dur = int(self.clip.duration_ms)
                self.setCursor(Qt.SizeHorCursor)
            else:
                self._press_pos = ev.position().toPoint(); self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.row.track.locked: return
        if self._resizing:
            # trim with snapping
            x = self.mapToParent(QPointF(ev.position()).toPoint()).x()
            # convert x-in-row to ms
            start_ms = int(max(0, (x / max(1.0, self.row.px_per_s))*1000.0))
            try: start_ms = self.row.editor._snap_ms(start_ms)
            except Exception: pass
            # Determine max allowed duration for this media type
            max_ms = None
            try:
                if self.media.kind in ("audio","video"):
                    d = self.media.meta.get("duration")
                    if d is not None:
                        max_ms = max(1, int(float(d) * 1000 / max(0.001, float(getattr(self.clip,"speed",1.0)))))
                    else:
                        max_ms = max(1, int(self._orig_dur))  # unknown duration -> don't extend
            except Exception:
                max_ms = None
            if self._resize_edge == "L":
                # new start cannot exceed old end-1; clamp final duration to media length if applicable
                end_t = int(self._orig_start + self._orig_dur)
                new_start = min(start_ms, end_t - 1)
                new_dur = int(end_t - new_start)
                if max_ms is not None and new_dur > max_ms:
                    new_start = int(end_t - max_ms)
                    new_dur = int(max_ms)
                self.clip.start_ms = int(new_start)
                self.clip.duration_ms = max(1, int(new_dur))
            else:
                # right edge -> duration changes; clamp to max allowed duration
                end_t = max(start_ms, self._orig_start + 1)
                new_dur = int(end_t - self._orig_start)
                if max_ms is not None and new_dur > max_ms:
                    new_dur = int(max_ms)
                self.clip.duration_ms = max(1, int(new_dur))
            try:
                self.row._update_clip_widget_geometry(self)
            except Exception:
                pass
            self.row.editor._update_time_range(); self.row.update()
        elif self._press_pos and (ev.buttons() & Qt.LeftButton):
            if not self._dragging and (ev.position().toPoint() - self._press_pos).manhattanLength() > 6:
                # begin DnD for moving
                src_ti = self.row.editor._track_index(self.row.track)
                src_ci = self.row.index_of(self.clip)
                if src_ti is None or src_ci is None: return
                mime = QMimeData(); mime.setData(self.CLIP_MIME, f"{src_ti}:{src_ci}:{self.media.kind}".encode("utf-8"))
                d = QDrag(self); d.setMimeData(mime)
                ghost = QPixmap(self.width(), self.height()); ghost.fill(QColor(0,0,0,0))
                qp = QPainter(ghost); qp.fillRect(ghost.rect(), QColor(255,255,255,60)); qp.end()
                d.setPixmap(ghost)
                self._dragging = True; d.exec(Qt.MoveAction)
        else:
            # update cursor hover
            ex = ev.position().toPoint().x()
            edge = self._edge_at_pos(ex)
            self._hover_edge = edge
            if edge: self.setCursor(Qt.SizeHorCursor)
            else: self.setCursor(Qt.OpenHandCursor)
            self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._resizing:
            self._resizing = False; self._resize_edge = None; self.setCursor(Qt.OpenHandCursor)
            # finalize geometry with one relayout and snapshot for undo
            self.row.relayout(); self.row.editor._mark_dirty(); self.row.editor._push_undo()  # push after change
        self.setCursor(Qt.OpenHandCursor); self._press_pos = None; self._dragging = False
        super().mouseReleaseEvent(ev)


        # If this clip is currently playing, clamp background player to new end
        try:
            ed = self.row.editor
            if getattr(ed, "_playing_clip", None) is self and getattr(ed, "_bg_player", None):
                rel = int(ed._bg_player.position())
                if rel >= int(self.clip.duration_ms):
                    new_rel = int(self.clip.duration_ms) - 1
                    if new_rel < 0: new_rel = 0
                    try: ed._bg_player.setPosition(new_rel)
                    except Exception: pass
                    try: ed._bg_player.pause()
                    except Exception: pass
                    ed._is_playing = False
                    g = int(self.clip.start_ms) + new_rel
                    ed.seek_slider.blockSignals(True)
                    ed.seek_slider.setValue(int(g))
                    ed.seek_slider.blockSignals(False)
                    try: ed.ruler.set_playhead(int(g))
                    except Exception: pass
                    try: ed._ensure_playhead_visible()
                    except Exception: pass
        except Exception:
            pass

    def enterEvent(self, ev):
        try:
            pos = self.mapFromGlobal(QCursor.pos())
            self._hover_edge = self._edge_at_pos(pos.x())
            if self._hover_edge: self.setCursor(Qt.SizeHorCursor)
            else: self.setCursor(Qt.OpenHandCursor)
            self.update()
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover_edge = None
        try: self.setCursor(Qt.OpenHandCursor)
        except Exception: pass
        self.update()
        super().leaveEvent(ev)

class TrackHeader(QWidget):
    def __init__(self, track: Track, editor: "EditorPane", parent=None):
        super().__init__(parent); self.track=track; self.editor=editor
        lay = QHBoxLayout(self); lay.setContentsMargins(6,2,6,2); lay.setSpacing(4)
        self.name = QLabel(self._label_text())
        self.btn_m = QToolButton(); self.btn_m.setText("M"); self.btn_m.setCheckable(True); self.btn_m.setToolTip("Mute track")
        self.btn_s = QToolButton(); self.btn_s.setText("S"); self.btn_s.setCheckable(True); self.btn_s.setToolTip("Solo track")
        self.btn_l = QToolButton(); self.btn_l.setText("L"); self.btn_l.setCheckable(True); self.btn_l.setToolTip("Lock track")
        self.btn_color = QToolButton(); self.btn_color.setToolTip("Change selected clip color"); self.btn_color.setIcon(_palette_icon(16)); self.btn_color.setIconSize(QSize(16,16))
        for b in (self.btn_m,self.btn_s,self.btn_l):
            b.setFixedSize(24,20)
        self.btn_color.setFixedSize(28,22)
        self.btn_m.hide(); self.btn_s.hide()
        lay.addWidget(self.name); lay.addWidget(self.btn_l); lay.addWidget(self.btn_color)
        self.btn_l.setChecked(track.locked)
        self.btn_l.toggled.connect(self._set_lock); self.btn_color.clicked.connect(self._pick_color)
        try:
            self._apply_color()
        except Exception:
            pass
    def contextMenuEvent(self, ev):
        m = QMenu(self)
        m.addAction("Add new timeline above", lambda: self.editor._add_timeline_relative(self.track, where='above'))
        m.addAction("Add new timeline below", lambda: self.editor._add_timeline_relative(self.track, where='below'))
        m.addAction("Rename", self._rename_track)
        m.addSeparator()
        m.addAction("Clear timeline", lambda: self.editor._clear_timeline(self.track))
        m.addAction("Delete timeline from project", lambda: self.editor._delete_timeline(self.track))
        m.exec(ev.globalPos())
    def _rename_track(self):
        """Prompt to rename this timeline (max 16 characters)."""
        try:
            from PySide6.QtWidgets import QInputDialog
        except Exception:
            QInputDialog = None
        new_name = None
        if QInputDialog is not None:
            try:
                current = (getattr(self.track, "name", "") or "").strip()
                txt, ok = QInputDialog.getText(self, "Rename timeline", "New name (max 16):", text=current)
                if ok:
                    new_name = (str(txt) if txt is not None else "").strip()
            except Exception:
                new_name = None
        if not new_name:
            return  # cancelled or empty
        new_name = new_name[:16]
        try:
            self.track.name = new_name
        except Exception:
            pass
        try:
            self.name.setText(self._label_text())
        except Exception:
            pass
        try:
            self.editor._mark_dirty()
        except Exception:
            pass


    def _label_text(self):
        nm = (getattr(self.track, "name", "") or "").strip()
        typ = (getattr(self.track, "type", "") or "").strip()
        if nm:
            return nm
        return typ.capitalize() if typ else "Track"

    def _apply_color(self):
        try:
            self.setStyleSheet("")
        except Exception:
            pass

    def _set_mute(self, on): self.track.mute = bool(on); self.editor._mark_dirty()
    def _set_solo(self, on): self.track.solo = bool(on); self.editor._mark_dirty()
    def _set_lock(self, on): self.track.locked = bool(on); self.editor._mark_dirty()
    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self.track.color), self, "Clip color")
        if col.isValid():
            try:
                self.editor._apply_color_to_selection_on_track(self.track, col.name())
            except Exception:
                try:
                    self.editor._apply_color_to_selection(col.name())
                except Exception:
                    pass

class TrackRow(QWidget):
    def __init__(self, track: Track, project: Project, editor: "EditorPane", parent=None):
        super().__init__(parent)
        self.track = track; self.project = project; self.editor = editor
        self.px_per_s = 100
        self.setMinimumHeight(46)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._clip_widgets: List[ClipWidget] = []
        self._drag_hover = False; self._drag_can = False
        self._ghost_rect: Optional[QRect] = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_ctx)

    def mousePressEvent(self, ev):
        try:
            if ev.button() == Qt.LeftButton:
                self.editor._set_selected_track(self)
        except Exception:
            pass
        super().mousePressEvent(ev)

    def index_of(self, clip: Clip) -> Optional[int]:
        try: return self.track.clips.index(clip)
        except Exception: return None

    def set_scale(self, px): self.px_per_s = float(px); self.relayout()

    def relayout(self):
        for w in self._clip_widgets: w.setParent(None); w.deleteLater()
        self._clip_widgets = []
        for c in self.track.clips:
            m = self.project.media.get(c.media_id); 
            if not m: continue
            w = ClipWidget(c, m, self)
            x = int((c.start_ms/1000.0)*self.px_per_s); width = int((max(1,c.duration_ms)/1000.0)*self.px_per_s)
            w.setGeometry(x, 4, max(36,width), 38); w.show(); self._clip_widgets.append(w)
        total_ms = EditorPane.compute_project_duration(self.project)
        self.setMinimumWidth(int((total_ms/1000.0)*self.px_per_s)+80)
        try: self.editor._update_holder_width()
        except Exception: pass

    def _update_clip_widget_geometry(self, w: ClipWidget):
        """Update a single clip widget geometry without rebuilding all widgets."""
        try:
            c = w.clip
            x = int((c.start_ms/1000.0)*self.px_per_s)
            width = int((max(1,c.duration_ms)/1000.0)*self.px_per_s)
            w.setGeometry(x, 4, max(36,width), 38)
            w.update()
        except Exception:
            pass


    def paintEvent(self, ev):
        # locked overlay
        if self.track.locked:
            p0 = QPainter(self); p0.fillRect(self.rect(), QColor(40,40,40,60)); p0.end()
        # droppable highlight
        if self._drag_hover and self._drag_can:
            pbg = QPainter(self); pbg.fillRect(self.rect(), QColor(40, 90, 40, 80)); pbg.end()
        super().paintEvent(ev)
        
        # selection border (always, even when empty)
        try:
            if getattr(self.editor, "_selected_track_row", None) is self:
                psel = QPainter(self)
                psel.setPen(QPen(QColor(255, 200, 80), 2))
                rect = self.rect().adjusted(1, 1, -1, -1)
                psel.drawRect(rect)
                psel.end()
        except Exception:
            pass
# ghost preview
        if self._ghost_rect:
            p = QPainter(self); p.fillRect(self._ghost_rect, QColor(255,255,255,80)); p.setPen(QPen(QColor(255,255,255),1)); p.drawRect(self._ghost_rect); p.end()
        # playhead
        try:
            ms = int(self.editor.seek_slider.value()) if self.editor else 0
            x = int((ms/1000.0) * self.px_per_s)
            p = QPainter(self); p.setPen(QPen(QColor(230,80,80), 2)); p.drawLine(x, 0, x, self.height()); p.end()
        except Exception: pass

    def _kind_from_mid(self, mid: str) -> Optional[str]:
        m = self.project.media.get(mid); return m.kind if m else None

    # --- DnD ---
    def dragEnterEvent(self, ev):
        if self.track.locked: ev.ignore(); return
        mime = ev.mimeData(); can = False
        if mime and mime.hasFormat(MediaList.MIME):
            mid = bytes(mime.data(MediaList.MIME)).decode("utf-8")
            kind = self._kind_from_mid(mid); can = True  # allow; we'll auto-route to a compatible track on drop
        elif mime and mime.hasFormat(ClipWidget.CLIP_MIME):
            try:
                _,_, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":"); can = True  # allow; we'll auto-route to a compatible track on drop
            except Exception: can = False
        self._drag_hover = True; self._drag_can = bool(can); self.update()
        if can: ev.acceptProposedAction()
        else: ev.ignore()

    def dragMoveEvent(self, ev):
        if self.track.locked: ev.ignore(); return
        y = ev.position().y() if hasattr(ev,'position') else ev.pos().y()
        h = max(1,self.height()); band_ok = (h*0.10) <= y <= (h*0.90)
        self._drag_can = bool(band_ok) if self._drag_can else False
        # ghost preview rectangle
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        start_ms = int(max(0, (x / max(1.0, self.px_per_s)) * 1000.0))
        try: start_ms = self.editor._snap_ms(start_ms)
        except Exception: pass
        width_ms = 1500
        mime = ev.mimeData()
        if mime.hasFormat(ClipWidget.CLIP_MIME):
            src_ti, src_ci, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":")
            src_ti = int(src_ti); src_ci = int(src_ci)
            try:
                clip = self.editor.project.tracks[src_ti].clips[src_ci]
                width_ms = max(1, clip.duration_ms)
            except Exception: pass
        elif mime.hasFormat(MediaList.MIME):
            mid = bytes(mime.data(MediaList.MIME)).decode("utf-8")
            m = self.project.media.get(mid)
            if m and m.meta.get("duration"):
                try: width_ms = int(float(m.meta["duration"]) * 1000)
                except: pass
        rx = int((start_ms/1000.0)*self.px_per_s)
        rw = int((width_ms/1000.0)*self.px_per_s)
        self._ghost_rect = QRect(rx, 4, max(8,rw), 38)
        self.update(); ev.acceptProposedAction()

    def dragLeaveEvent(self, ev):
        self._drag_hover = False; self._drag_can = False; self._ghost_rect=None; self.update(); ev.accept()

    def dropEvent(self, ev):
        if self.track.locked: ev.ignore(); return
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        start_ms = int(max(0, (x / max(1.0, self.px_per_s)) * 1000.0))
        try:
            start_ms = self.editor._snap_ms(start_ms)
        except Exception: pass
        mime = ev.mimeData()
        if mime.hasFormat(MediaList.MIME):
            mid = bytes(mime.data(MediaList.MIME)).decode("utf-8")
            m = self.project.media.get(mid); 
            if not m: return
            # If this track can't accept it, find or create a compatible track
            target_tr = self.track
            if self.track.type not in ("any", m.kind):
                # find this row index in project to insert near it
                try:
                    near_idx = self.editor.project.tracks.index(self.track)
                except Exception:
                    near_idx = None
                target_tr = self.editor._ensure_track_for_kind(m.kind, near_idx)
            self.editor._push_undo()
            self.editor._add_media_clip(target_tr, mid, start_ms)
            self._drag_hover = False; self._drag_can = False; self._ghost_rect=None; self.update()
            ev.acceptProposedAction(); return

        if mime.hasFormat(ClipWidget.CLIP_MIME):
            src_ti, src_ci, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":")
            src_ti = int(src_ti); src_ci = int(src_ci)
            if self.track.type not in ("any", kind):
                QMessageBox.information(self, "Type mismatch", f"This track is locked to {self.track.type}."); return
            # band rule for cross-row moves
            y = ev.position().y() if hasattr(ev,'position') else ev.pos().y()
            h = max(1,self.height()); band_ok = (h*0.10) <= y <= (h*0.90)
            self.editor._push_undo()
            src_tr = self.editor.project.tracks[src_ti]
            if src_ci < 0 or src_ci >= len(src_tr.clips): return
            if src_tr is self.track or band_ok:
                clip = src_tr.clips[src_ci] if src_tr is self.track else src_tr.clips.pop(src_ci)
                if self.track.type == "any":
                    self.track.type = kind
                    try: self.editor._update_track_labels()
                    except Exception: pass
                clip.start_ms = start_ms
                if src_tr is self.track:
                    try: src_tr.clips.sort(key=lambda c: c.start_ms)
                    except Exception: pass
                else:
                    self.track.clips.append(clip)
                self.editor._refresh_tracks(); self.editor._update_time_range(); self.editor._mark_dirty()
                self._drag_hover = False; self._drag_can = False; self._ghost_rect=None; self.update()
                ev.acceptProposedAction(); return
            else:
                ev.ignore(); return

    # timeline empty-space menu
    def _on_ctx(self, pos):
        child = self.childAt(pos)
        if isinstance(child, ClipWidget): return
        m = QMenu(self)
        def add_files():
            ed = self.editor
            start_ms = int(getattr(ed.seek_slider, 'value', lambda:0)())
            fns, _ = QFileDialog.getOpenFileNames(self, "Add to timeline", getattr(ed, '_last_open_dir', str(Path.cwd())), "Media Files (*.*)")
            if not fns: return
            ed._last_open_dir = str(Path(fns[0]).parent)
            for fn in fns:
                pth = Path(fn); kind = media_kind(pth)
                for mid, item in ed.project.media.items():
                    if Path(item.path) == pth: break
                else:
                    mid = ed._unique_media_id(pth)
                    ed.project.media[mid] = MediaItem(id=mid, path=pth, kind=kind, meta=ed._probe(pth))
                if self.track.type in ("any", kind):
                    dur = 3000
                    md = ed.project.media[mid].meta
                    if md.get("duration"):
                        try: dur = int(float(md["duration"]) * 1000)
                        except: pass
                    if self.track.type == "any":
                        self.track.type = kind
                        try: ed._update_track_labels()
                        except Exception: pass
                    self.track.clips.append(Clip(media_id=mid, start_ms=start_ms, duration_ms=dur))
            ed._refresh_ui(); ed._update_time_range(); ed._mark_dirty()
        m.addAction("Add to timeline…", add_files)
        add_menu = m.addMenu("Add new timeline")
        add_menu.addAction("Video", lambda: self.editor._add_timeline_direct("video","Video"))
        add_menu.addAction("Image", lambda: self.editor._add_timeline_direct("image","Image"))
        add_menu.addAction("Text",  lambda: self.editor._add_timeline_direct("text","Text"))
        add_menu.addAction("Audio", lambda: self.editor._add_timeline_direct("audio","Audio"))
        def remove_tl():
            ed = self.editor
            idx = ed._track_index(self.track)
            if idx is not None and 0 <= idx < len(ed.project.tracks):
                ed._push_undo(); ed.project.tracks.pop(idx)
                ed._refresh_tracks(); ed._update_time_range(); ed._mark_dirty()
        m.addAction("Remove timeline", remove_tl)
        m.exec(self.mapToGlobal(pos))

    # zoom with wheel over timeline row
    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if abs(dy) > 0:
            if dy > 0: self.editor._set_zoom(self.editor.zoom*1.1)
            else: self.editor._set_zoom(self.editor.zoom/1.1)
            ev.accept(); return
        super().wheelEvent(ev)

# ---------- export dialog ----------
class ExportDialog(QDialog):
    def __init__(self, parent=None, default_w=1280, default_h=720):
        super().__init__(parent); self.setWindowTitle("Export Video"); self.setModal(True)
        lay = QFormLayout(self)
        self.combo = QComboBox()
        self.res_list = [("1280x720",1280,720),("1920x1080",1920,1080)]
        for label,w,h in self.res_list: self.combo.addItem(label,(w,h))
        self.fps = QSpinBox(); self.fps.setRange(10,120); self.fps.setValue(30)
        self.bitrate = QLineEdit("6M"); self.out = QLineEdit("export.mp4")
        btn = QToolButton(); btn.setText("…")
        def pick():
            fn,_=QFileDialog.getSaveFileName(self,"Save As",str(Path.cwd()/ "output"/"video"/"export.mp4"),"MP4 (*.mp4)")
            if fn: self.out.setText(fn)
        btn.clicked.connect(pick)
        h = QHBoxLayout(); h.addWidget(self.out); h.addWidget(btn)
        lay.addRow("Resolution:", self.combo); lay.addRow("FPS:", self.fps); lay.addRow("Video Bitrate:", self.bitrate); lay.addRow("Output:", h)
        bb = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); lay.addRow(bb)
    def values(self):
        w,h = self.combo.currentData()
        return {"width":w,"height":h,"fps":int(self.fps.value()),"bitrate":self.bitrate.text().strip() or "6M","output": self.out.text().strip() or "export.mp4"}

# ---------- editor pane ----------
class EditorPane(QWidget):
    preview_media = Signal(object, int)
    transport_request = Signal(str)





    # currently selected track row (for border highlight)
    _selected_track_row = None

    def _set_selected_track(self, row):
        try:
            self._selected_track_row = row
            for r in self.tracks_holder.findChildren(TrackRow):
                r.update()
        except Exception:
            pass
    def _step_ms(self) -> int:
        """Return step size in milliseconds based on the UI 'Step' control."""
        try:
            val = float(getattr(self.grid_step, "currentText", lambda: "1")())
            return max(1, int(val * 1000))
        except Exception:
            return 1000
    def _clipwidget_under_playhead(self):
        """Return the ClipWidget under the current playhead, preferring video when overlapping."""
        try:
            g = int(self.seek_slider.value())
        except Exception:
            g = 0
        best = None
        best_kind = None
        try:
            for cw in self.tracks_holder.findChildren(ClipWidget):
                try:
                    start = int(getattr(cw.clip, "start_ms", 0))
                    dur = int(getattr(cw.clip, "duration_ms", 0))
                    if start <= g < start + max(1, dur):
                        kind = getattr(cw.row.track, "kind", getattr(cw.media, "kind", "video"))
                        if best is None or (kind == "video" and best_kind != "video"):
                            best = cw; best_kind = kind
                except Exception:
                    pass
        except Exception:
            pass
        return best

    def _play_timeline_under_ruler_background(self):
        """Always playable fallback: find the data Clip under the ruler and play it via _bg_player."""
        try:
            g = int(self.seek_slider.value())
        except Exception:
            g = 0
        # Find a clip by data (prefer video kind tracks)
        best_clip = None
        best_kind = None
        try:
            for tr in getattr(self.project, "tracks", []):
                for cl in getattr(tr, "clips", []):
                    try:
                        st = int(getattr(cl, "start_ms", 0))
                        du = int(getattr(cl, "duration_ms", 0))
                        if st <= g < st + max(1, du):
                            kind = getattr(tr, "kind", "video")
                            if best_clip is None or (kind == "video" and best_kind != "video"):
                                best_clip = cl; best_kind = kind
                    except Exception:
                        pass
        except Exception:
            pass
        if not best_clip or not getattr(self, "_bg_player", None):
            return False
        # Resolve media
        try:
            media_id = getattr(best_clip, "media_id", "")
            media = self.project.media.get(media_id, None)
        except Exception:
            media = None
        if not media or getattr(media, "kind", "") not in ("audio","video"):
            return False
        try:
            from PySide6.QtCore import QUrl
            url = QUrl.fromLocalFile(str(media.path))
            self._bg_player.setSource(url)
        except Exception:
            return False
        # Compute relative seek
        rel = max(0, g - int(getattr(best_clip, "start_ms", 0)))
        try:
            dur = int(getattr(best_clip, "duration_ms", 0))
            rel = min(rel, max(0, dur-1))
        except Exception:
            pass
        try:
            self._bg_player.setPosition(int(rel))
            self._bg_player.play()
            # Small holder with .clip so _on_bg_pos logic works
            try:
                from types import SimpleNamespace
                self._playing_clip = SimpleNamespace(clip=best_clip)
            except Exception:
                self._playing_clip = type("X", (), {"clip": best_clip})()
            self._is_playing = True
            return True
        except Exception:
            return False
    def _clip_under_playhead(self):
        """Return the Clip under the current global playhead; prefer video over audio."""
        try:
            g = int(self.seek_slider.value())
        except Exception:
            g = 0
        best = None
        best_kind = None
        try:
            for tr in getattr(self.project, "tracks", []):
                for cl in getattr(tr, "clips", []):
                    try:
                        start = int(getattr(cl, "start_ms", 0))
                        dur = int(getattr(cl, "duration_ms", 0))
                        if start <= g < start + max(1, dur):
                            kind = getattr(tr, "kind", "video")
                            if best is None or (kind == "video" and best_kind != "video"):
                                best = cl; best_kind = kind
                    except Exception:
                        pass
        except Exception:
            pass
        return best
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.zoom = 1.0
        self.max_tracks = 8
        self._undo_stack: List[Dict[str,Any]] = []
        self._redo_stack: List[Dict[str,Any]] = []
        self._clipboard: Optional[dict] = None
        self._selected: Optional[Clip] = None
        self._selection: List[Clip] = []  # multi-select
        self._next_group_id = 1
        self.in_ms: Optional[int] = None
        self.out_ms: Optional[int] = None
        self._shuttle_rate: int = 0
        self._last_open_dir: str = str(Path.cwd())
        self.glue_enabled = True
        self.grid_enabled = False
        self.grid_step_s = 0.5
        self.snap_px_tol = 8
        self.ripple_enabled = False
        self.markers: List[int] = []
        self.header_col_w = 180
        self._init_ui()
        self._new_project()
        self._setup_autosave()
        try: self._start_autoplace_timer()
        except Exception: pass
        # restore autosave if present
        try:
            if self._autosave_path.exists():
                self.project = Project.from_json(json.loads(self._autosave_path.read_text(encoding="utf-8"))); self._refresh_ui()
        except Exception: pass

    # layout
    def _init_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(8,8,8,8)
        tb = QHBoxLayout()
        def btn(text, tip, cb): b=QToolButton(); b.setText(text); b.setToolTip(tip); b.clicked.connect(cb); return b
        self.btn_new=btn("New","New project (Ctrl+N)", self._new_project)
        self.btn_load=btn("Load","Load project (Ctrl+O)", self._load_project)
        self.btn_save=btn("Save","Save project (Ctrl+S)", self._save_project)
        self.btn_imp =btn("Load media","Import media", self._import_media)
        self.btn_export=btn("Export","Export MP4", self._export)
        self.btn_undo=btn("Undo","Undo (Ctrl+Z)", self._undo)
        self.btn_redo=btn("Redo","Redo (Ctrl+Y)", self._redo)
        self.btn_fit=btn("Fit","Zoom to fit (Ctrl+F)", self._zoom_to_fit)
        self.btn_sel=btn("Sel","Zoom to selection (Z)", self._zoom_to_selection)
        self.btn_glue=QToolButton(); self.btn_glue.setText("Glue"); self.btn_glue.setCheckable(True); self.btn_glue.setChecked(True); self.btn_glue.toggled.connect(lambda on: setattr(self,'glue_enabled',bool(on)))
        self.btn_grid=QToolButton(); self.btn_grid.setText("Grid"); self.btn_grid.setCheckable(True); self.btn_grid.toggled.connect(self._toggle_grid)
        self.grid_step = QComboBox(); [self.grid_step.addItem(s, float(s)) for s in ["0.25","0.5","1","2","5"]]; self.grid_step.setCurrentIndex(1); self.grid_step.currentIndexChanged.connect(lambda *_: setattr(self,'grid_step_s',float(self.grid_step.currentData())))
        self.snap_tol = QSpinBox(); self.snap_tol.setRange(1,32); self.snap_tol.setValue(self.snap_px_tol); self.snap_tol.valueChanged.connect(lambda v: setattr(self,'snap_px_tol',int(v)))
        self.btn_ripple=QToolButton(); self.btn_ripple.setText("Ripple"); self.btn_ripple.setCheckable(True); self.btn_ripple.toggled.connect(lambda on: setattr(self,'ripple_enabled',bool(on)))
        for w in [self.btn_new,self.btn_load,self.btn_save,self.btn_imp,self.btn_export,self.btn_undo,self.btn_redo]:
            tb.addWidget(w)
        tb.addStretch(1); root.addLayout(tb)
        # Splitter to resize between preview strip and timeline
        self.vsplit = QSplitter(Qt.Vertical, self)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.setHandleWidth(8)
        root.addWidget(self.vsplit, 1)

        # --- top: media strip container ---
        top = QWidget(); top_v = QVBoxLayout(top); top_v.setContentsMargins(0,0,0,0); top_v.setSpacing(2)
        self.media_list = MediaList(); top_v.addWidget(self.media_list)
        self.media_list.requestDelete.connect(self._delete_media_ids)
        self.media_list.requestPreview.connect(self._preview_media_id)
        top_div = QFrame(); top_div.setFrameShape(QFrame.HLine); top_div.setFrameShadow(QFrame.Sunken); top_v.addWidget(top_div)
        self.vsplit.addWidget(top)

        # --- bottom: timeline container ---
        bottom = QWidget(); bot_v = QVBoxLayout(bottom); bot_v.setContentsMargins(0,0,0,0); bot_v.setSpacing(4)
        # Top controls row (spreads UI so the lower row isn't crowded)
        ctrl_top = QHBoxLayout()
        ctrl_top.setSpacing(6)
        try:
            ctrl_top.addWidget(self.btn_fit)
            ctrl_top.addWidget(self.btn_sel)
            ctrl_top.addWidget(QLabel("Step"))
            ctrl_top.addWidget(self.grid_step)
            ctrl_top.addWidget(QLabel("Snap px"))
            ctrl_top.addWidget(self.snap_tol)
            ctrl_top.addWidget(self.btn_grid)
            ctrl_top.addWidget(self.btn_ripple)
            ctrl_top.addWidget(self.btn_glue)
        except Exception:
            pass
        ctrl_top.addStretch(1)
        bot_v.addLayout(ctrl_top)

        # controls
        ctrl = QHBoxLayout()
        # Zoom Fit button
        self.btn_zoom_fit = QToolButton(); self.btn_zoom_fit.setText("Fit"); self.btn_zoom_fit.setToolTip("Fit entire project")
        try: self.btn_zoom_fit.clicked.connect(self._zoom_fit_all)
        except Exception: pass
        self.btn_add_track = QToolButton(); self.btn_add_track.setText("Add timeline"); self.btn_add_track.clicked.connect(self._add_track)
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(10,500); self.zoom_slider.setValue(100); self.zoom_slider.valueChanged.connect(lambda v: self._set_zoom(v/10000.0))
        try:
            self.zoom_slider.setRange(1, 20000)
            self.zoom_slider.setValue(max(1, int(getattr(self, "zoom", 1.0)*10000)))
        except Exception:
            pass
        try:
            self.zoom_slider.setRange(1, 20000)  # allows zoom down to 0.0001
            self.zoom_slider.setValue(int(max(1, int(getattr(self, "zoom", 1.0)*10000))))
        except Exception:
            ctrl.addWidget(self.btn_zoom_fit)
            pass
        self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.setRange(0,60000); self.seek_slider.sliderMoved.connect(self._seek_changed); self.seek_slider.valueChanged.connect(self._seek_changed)
        # transport buttons (icons only; not wired yet)
        self.btn_to_start = QToolButton(); self.btn_to_start.setObjectName("btn_to_start")
        self.btn_back = QToolButton(); self.btn_back.setObjectName("btn_back")
        self.btn_play_pause = QToolButton(); self.btn_play_pause.setObjectName("btn_play_pause")
        self.btn_stop = QToolButton(); self.btn_stop.setObjectName("btn_stop")
        self.btn_forward = QToolButton(); self.btn_forward.setObjectName("btn_forward")
        self.btn_to_end = QToolButton(); self.btn_to_end.setObjectName("btn_to_end")
        try:
            st = self.style()
            self.btn_to_start.setIcon(st.standardIcon(QStyle.SP_MediaSkipBackward))
            self.btn_back.setIcon(st.standardIcon(QStyle.SP_MediaSeekBackward))
            self.btn_play_pause.setIcon(st.standardIcon(QStyle.SP_MediaPlay))
            self.btn_stop.setIcon(st.standardIcon(QStyle.SP_MediaStop))
            self.btn_forward.setIcon(st.standardIcon(QStyle.SP_MediaSeekForward))
            self.btn_to_end.setIcon(st.standardIcon(QStyle.SP_MediaSkipForward))
        except Exception:
            pass
        # tooltips for clarity
        self.btn_to_start.setToolTip("Go to start")
        self.btn_back.setToolTip("Step backward")
        self.btn_play_pause.setToolTip("Play/Pause")
        self.btn_stop.setToolTip("Stop")
        self.btn_forward.setToolTip("Step forward")
        self.btn_to_end.setToolTip("Go to end")
        # --- background player (no popup) ---
        self._bg_player = QMediaPlayer(self) if HAS_QTMULTI else None
        self._bg_audio = QAudioOutput(self) if HAS_QTMULTI else None
        if self._bg_player and self._bg_audio:
            self._bg_player.setAudioOutput(self._bg_audio)
            try: self._bg_player.positionChanged.connect(self._on_bg_pos)
            except Exception: pass
            # --- gap playback (black during empty timeline) ---
            self._gap_timer = QTimer(self)
            try: self._gap_timer.setInterval(33)
            except Exception: pass
            try: self._gap_timer.timeout.connect(self._gap_tick)
            except Exception: pass
            self._gap_playing = False
            self._play_continuous = False

        self._playing_clip = None  # ClipWidget currently driving playback
        self._is_playing = False
        self.transport_request.connect(self._on_transport)
        for w in [self.btn_add_track, QLabel("Zoom"), self.zoom_slider, QLabel("Seek"), self.seek_slider]:
            ctrl.addWidget(w)
        # timecode + in/out controls row (compact)
        tc_row = QHBoxLayout()
        self.tc_edit = QLineEdit("00:00:00:00"); self.tc_edit.setFixedWidth(110); self.tc_edit.setPlaceholderText("HH:MM:SS:FF")
        tc_row.addWidget(self.tc_edit)
        self.btn_set_in = QToolButton(); self.btn_set_in.setText("I"); self.btn_set_in.setToolTip("Set In (I)")
        self.btn_set_out = QToolButton(); self.btn_set_out.setText("O"); self.btn_set_out.setToolTip("Set Out (O)")
        self.btn_clear_io = QToolButton(); self.btn_clear_io.setText("Clear I/O"); self.btn_clear_io.setToolTip("Clear In/Out")
        self.lbl_io = QLabel("—"); 
        for w in [self.btn_set_in, self.btn_set_out, self.btn_clear_io, self.lbl_io]: tc_row.addWidget(w)
        bot_v.addLayout(tc_row)
        self.tc_edit.editingFinished.connect(self._jump_to_timecode)
        try:
            self.seek_slider.sliderPressed.connect(lambda: self.transport_request.emit("scrub:start"))
            self.seek_slider.sliderReleased.connect(lambda: self.transport_request.emit("scrub:end"))
        except Exception:
            pass
        self.btn_set_in.clicked.connect(self._set_in_here)
        self.btn_set_out.clicked.connect(self._set_out_here)
        self.btn_clear_io.clicked.connect(self._clear_in_out)


        # Add 'Start'/'End' buttons close together (kept from previous patch)
        self.btn_go_start = QToolButton(); self.btn_go_start.setObjectName("btn_go_start"); self.btn_go_start.setText("Start"); self.btn_go_start.setToolTip("Go to start (selection or timeline)")
        self.btn_go_end = QToolButton(); self.btn_go_end.setObjectName("btn_go_end"); self.btn_go_end.setText("End"); self.btn_go_end.setToolTip("Go to end (selection or timeline)")
        self.btn_go_start.setFixedWidth(58); self.btn_go_end.setFixedWidth(58)
        self.btn_go_start.clicked.connect(lambda: self._goto_edge('start'))
        self.btn_go_end.clicked.connect(lambda: self._goto_edge('end'))
        ctrl.setSpacing(6)
        ctrl.addWidget(self.btn_to_start); ctrl.addWidget(self.btn_back); ctrl.addWidget(self.btn_play_pause); ctrl.addWidget(self.btn_stop); ctrl.addWidget(self.btn_forward); ctrl.addWidget(self.btn_to_end)
        # wire transport buttons to internal player
        try:
            self.btn_play_pause.clicked.connect(self._tr_toggle)
            self.btn_stop.clicked.connect(self._tr_stop)
            self.btn_to_start.clicked.connect(self._tr_start)
            self.btn_to_end.clicked.connect(self._tr_end)
            self.btn_back.clicked.connect(lambda: self._tr_step(-1000))
            self.btn_forward.clicked.connect(lambda: self._tr_step(+1000))
            QTimer.singleShot(0, self._hook_player_signals)
        except Exception:
            pass
        # Move Glue/Grid/Ripple here, after Start/End
        ctrl.addStretch(1); bot_v.addLayout(ctrl)

        # ruler + tracks (align ruler zero after header column)
        self.ruler = TimeRuler()
        ruler_row = QWidget(); rh = QHBoxLayout(ruler_row); rh.setContentsMargins(0,0,0,0); rh.setSpacing(0)
        left_spacer = QWidget(); left_spacer.setFixedWidth(int(self.header_col_w)+1)
        rh.addWidget(left_spacer)
        vdiv_r = QFrame(); vdiv_r.setFrameShape(QFrame.VLine); vdiv_r.setFrameShadow(QFrame.Sunken)
        rh.addWidget(vdiv_r)
        rh.addWidget(self.ruler, 1)
        # keep the spacer matching the headers width
        self._ruler_left_spacer = left_spacer
        bot_v.addWidget(ruler_row)
        self.ruler.positionPicked.connect(self._set_playhead_from_ruler)
        self.ruler.set_fps(int(self.project.fps))
        self.ruler.set_in_out(self.in_ms, self.out_ms)
        self.ruler.set_markers(self.markers)

        
        # --- Split view: left fixed headers column, right scrollable rows column ---
        self.headers_area = QScrollArea(); self.headers_area.setWidgetResizable(True); self.headers_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.headers_area.setFrameShape(QFrame.NoFrame)
        try:
            self.headers_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception: pass
        self.headers_holder = QWidget(); self.headers_layout = QVBoxLayout(self.headers_holder)
        self.headers_layout.setContentsMargins(0,0,0,0); self.headers_layout.setSpacing(6)
        self.headers_holder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        self.headers_area.setWidget(self.headers_holder)
        self.tracks_area = QScrollArea(); self.tracks_area.setWidgetResizable(True)
        try:
            self.tracks_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        except Exception:
            pass
        holder = QWidget(); self.tracks_holder = holder
        self.tracks_layout = QVBoxLayout(holder); self.tracks_layout.setContentsMargins(0,0,0,0); self.tracks_layout.setSpacing(6)
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tracks_area.setWidget(holder)
        # Place headers_area and tracks_area side-by-side
        tracks_row = QWidget(); tracks_row_h = QHBoxLayout(tracks_row); tracks_row_h.setContentsMargins(0,0,0,0); tracks_row_h.setSpacing(0)
        # keep headers column width in sync
        try:
            self.headers_area.setFixedWidth(int(self.header_col_w))
            self._ruler_left_spacer.setFixedWidth(int(self.header_col_w)+1)
        except Exception:
            pass
        tracks_row_h.addWidget(self.headers_area)
        vdiv_main = QFrame(); vdiv_main.setFrameShape(QFrame.VLine); vdiv_main.setFrameShadow(QFrame.Sunken)
        tracks_row_h.addWidget(vdiv_main)
        tracks_row_h.addWidget(self.tracks_area, 1)
        bot_v.addWidget(tracks_row, 1)
        # External horizontal scrollbar under the timelines
        try: self.tracks_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception: pass
        self.hbar = QScrollBar(Qt.Horizontal)
        bot_v.addWidget(self.hbar)
        try:
            # Mirror inner horizontal scroll to external hbar
            self.tracks_area.horizontalScrollBar().valueChanged.connect(self._on_hscroll_changed)
            try:
                sb = self.tracks_area.horizontalScrollBar()
                sb.rangeChanged.connect(lambda *_: self._sync_hbar_from_inner())
                sb.valueChanged.connect(lambda v: self._sync_hbar_from_inner())
                self.hbar.valueChanged.connect(lambda v: self._sync_inner_from_hbar(v))
            except Exception:
                pass
            # Vertical scroll sync between headers and rows
            try:
                lv = self.headers_area.verticalScrollBar(); rv = self.tracks_area.verticalScrollBar()
                def _sync_v_from_r(v):
                    try:
                        if getattr(self, '_v_syncing', False): return
                        self._v_syncing = True; lv.setValue(v)
                    finally:
                        self._v_syncing = False
                def _sync_v_from_l(v):
                    try:
                        if getattr(self, '_v_syncing', False): return
                        self._v_syncing = True; rv.setValue(v)
                    finally:
                        self._v_syncing = False
                rv.valueChanged.connect(_sync_v_from_r)
                lv.valueChanged.connect(_sync_v_from_l)
            except Exception:
                pass
        except Exception:
            pass
        self.vsplit.addWidget(bottom)
        # default sizes
        try:
            self.vsplit.setStretchFactor(0, 0)
            self.vsplit.setStretchFactor(1, 1)
            # Keep the top (media strip) compact
            top.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            top.setMaximumHeight(self.media_list.maximumHeight() + 24)
        except Exception:
            pass

        self._add_shortcuts()
        # Apply unified styles for buttons and toggles
        self._apply_styles()

    def _add_shortcuts(self):
        def sc(seq, cb): a=QAction(self); a.setShortcut(QKeySequence(seq)); a.triggered.connect(cb); self.addAction(a)
        sc("Ctrl+N", self._new_project); sc("Ctrl+O", self._load_project); sc("Ctrl+S", self._save_project)
        sc("Ctrl+Z", self._undo); sc("Ctrl+Y", self._redo); sc("+", lambda: self._set_zoom(self.zoom*1.25)); sc("-", lambda: self._set_zoom(self.zoom/1.25))
        sc("Space", self._tr_toggle)
        sc("Ctrl+F", self._zoom_to_fit); sc("Z", self._zoom_to_selection)
        sc("M", self._add_marker_here);
        sc("J", lambda: self._shuttle("J"))
        sc("K", lambda: self._shuttle("K"))
        sc("L", lambda: self._shuttle("L"))
        sc("I", self._set_in_here)
        sc("O", self._set_out_here)
        sc("Ctrl+Shift+X", lambda: self._remove_range(ripple=True))  # Extract
        sc("Ctrl+L", lambda: self._remove_range(ripple=False))      # Lift
        sc("T", self._trim_to_range)
        sc(".", self._goto_next_marker); sc(",", self._goto_prev_marker)
        sc("Ctrl+G", self._group_selection); sc("Ctrl+Shift+G", self._ungroup_selection)

    # autosave
    def _setup_autosave(self):
        self._autosave_path = Path.cwd()/"presets"/"setsave"/"editor_temp.json"
        self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
        self._dirty=False; self._last_change_ts=0.0; self._last_save_ts=0.0
        self._autosave_timer = QTimer(self); self._autosave_timer.timeout.connect(self._autosave_tick); self._autosave_timer.start()

    # project ops
    def _new_project(self):
        self.project = Project(); self.project.tracks=[
            Track("Video","video"),
            Track("Video","video"),
            Track("Text","text"),
            Track("Audio","audio")
        ]
        self._undo_stack.clear(); self._redo_stack.clear(); self._refresh_ui(); self._mark_dirty()

    def _load_project(self):
        fn,_=QFileDialog.getOpenFileName(self,"Load Project",str(Path.cwd()/ "output"),"FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            self.project = Project.from_json(json.loads(Path(fn).read_text(encoding="utf-8")))
            self._undo_stack.clear(); self._redo_stack.clear(); self._refresh_ui(); self._mark_dirty()
        except Exception as e: 
            QMessageBox.critical(self,"Error",f"Failed to load: {e}")

    def _save_project(self):
        fn,_=QFileDialog.getSaveFileName(self,"Save Project",str(Path.cwd()/ "output"/"project.fvedit"),"FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            Path(fn).write_text(json.dumps(self.project.to_json(), indent=2), encoding="utf-8")
            QMessageBox.information(self,"Saved",f"Saved: {fn}")
        except Exception as e:
            QMessageBox.critical(self,"Error",f"Failed to save: {e}")

    # media importing & thumbs
    def _import_media(self):
        fns,_=QFileDialog.getOpenFileNames(self,"Import Media",self._last_open_dir,"Media Files (*.*)")
        if not fns: return
        self._last_open_dir = str(Path(fns[0]).parent)
        items=[]
        for fn in fns:
            p=Path(fn); kind=media_kind(p); mid=self._unique_media_id(p)
            self.project.media[mid]=MediaItem(id=mid,path=p,kind=kind,meta=self._probe(p))
            items.append((mid,self.project.media[mid]))
        self._refresh_media_list(items); self._mark_dirty()

    def _unique_media_id(self, p: Path) -> str:
        b = p.stem.lower().replace(" ","_"); i=1; mid=b
        while mid in self.project.media: i+=1; mid=f"{b}_{i}"
        return mid

    def _probe(self, path: Path) -> Dict[str,Any]:
        meta={"duration":None,"width":None,"height":None,"fps":None}
        try:
            out = subprocess.check_output([ffprobe_path(), "-v","error","-select_streams","v:0","-show_entries","stream=width,height,avg_frame_rate","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", str(path)], stderr=subprocess.STDOUT, universal_newlines=True)
            ls=[x.strip() for x in out.splitlines() if x.strip()]
            for x in ls:
                if meta["width"] is None and x.isdigit(): meta["width"]=int(x); continue
                if meta["height"] is None and x.isdigit(): meta["height"]=int(x); continue
                if "/" in x and meta["fps"] is None:
                    n,d=x.split("/"); 
                    try:
                        if float(d)!=0: meta["fps"]=round(float(n)/float(d),2)
                    except: pass
                if "." in x and meta["duration"] is None:
                    try: meta["duration"]=float(x)
                    except: pass
        except Exception: pass
        return meta


    def _short_media_label(self, path: Path) -> str:
        base = path.stem
        ext = path.suffix[1:] if path.suffix.startswith('.') else path.suffix
        if len(base) > 20:
            # Truncate base to 20 characters, then append ...ext (without the dot)
            return base[:20] + ('...' + ext if ext else '...')
        else:
            return base + (('.' + ext) if ext else '')


    def _refresh_media_list(self, new_items=None):
        self.media_list.clear()
        items=[]
        for m in self.project.media.values():
            li=QListWidgetItem(self._short_media_label(m.path)); li.setToolTip(m.path.name); li.setData(Qt.UserRole, m.id); li.setIcon(QIcon(_placeholder_icon(m.kind, self.media_list.iconSize()))); self.media_list.addItem(li); items.append((m.id,m))
        if new_items: items = new_items
        if items:
            self._t = ThumbThread(items, self.media_list.iconSize(), self); self._t.ready.connect(lambda mid,pm: self.media_list.set_thumb(mid,pm)); self._t.start()

    def _delete_media_ids(self, ids: List[str]):
        if not ids: return
        self._push_undo()
        # remove any clips referencing these media
        for mid in ids:
            for tr in self.project.tracks:
                tr.clips = [c for c in tr.clips if c.media_id != mid]
            self.project.media.pop(mid, None)
        self._refresh_ui(); self._update_time_range(); self._mark_dirty()

    def _preview_media_id(self, mid: str):
        m = self.project.media.get(mid)
        if not m: return
        dlg = PreviewDialog(m, self); dlg.show()

    # tracks/timeline rendering
    def _add_timeline_direct(self, kind, name): self.project.tracks.append(Track(name, kind)); self._refresh_tracks(); self._mark_dirty()

    def _add_track(self):
        m=QMenu(self); m.addAction("Video", lambda: self._add_timeline_direct("video","Video")); m.addAction("Image", lambda: self._add_timeline_direct("image","Image")); m.addAction("Text", lambda: self._add_timeline_direct("text","Text")); m.addAction("Audio", lambda: self._add_timeline_direct("audio","Audio")); m.exec(QCursor.pos())

    def _refresh_ui(self):
        self._refresh_media_list(); self._refresh_tracks(); self._update_time_range()

    def _update_track_labels(self):
        for w in self.tracks_holder.findChildren(TrackHeader):
            w.name.setText(w._label_text())

    
        # --- header actions & clip color ---
    def _apply_color_to_selection(self, color_hex: str):
        try:
            targets = self._selection[:] if self._selection else ([self._selected] if self._selected else [])
            if not targets:
                return
            self._push_undo()
            for c in targets:
                try:
                    c.label_color = color_hex
                except Exception:
                    pass
            self._mark_dirty(); self._refresh_tracks()
        except Exception:
            pass

    def _add_timeline_relative(self, track: Track, where: str = 'below'):
        try:
            idx = self._track_index(track)
            if idx is None:
                return
            ttype = getattr(track, 'type', 'video') or 'video'
            name = ''
            new_tr = Track(name or ttype.capitalize(), ttype)
            insert_at = idx if where == 'above' else idx + 1
            self._push_undo()
            self.project.tracks.insert(insert_at, new_tr)
            self._refresh_tracks(); self._mark_dirty()
        except Exception:
            pass

    def _clear_timeline(self, track: Track):
        try:
            self._push_undo()
            track.clips.clear()
            self._refresh_tracks(); self._update_time_range(); self._mark_dirty()
        except Exception:
            pass

    def _delete_timeline(self, track: Track):
        try:
            idx = self._track_index(track)
            if idx is None:
                return
            self._push_undo()
            self.project.tracks.pop(idx)
            self._refresh_tracks(); self._update_time_range(); self._mark_dirty()
        except Exception:
            pass

    def _refresh_tracks(self):
        # clear existing (both rows and headers)
        while self.tracks_layout.count():
            w = self.tracks_layout.takeAt(0).widget()
            if w:
                w.setParent(None); w.deleteLater()
        while self.headers_layout.count():
            w = self.headers_layout.takeAt(0).widget()
            if w:
                w.setParent(None); w.deleteLater()

        # rebuild rows with header column on the left
        for i, tr in enumerate(self.project.tracks):
            # Header (left column)
            header = TrackHeader(tr, self, self.headers_holder)
            header.setFixedWidth(int(self.header_col_w))
            try:
                header.setFixedHeight(46)
            except Exception:
                pass
            header.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.headers_layout.addWidget(header)
            # Row (right column)
            row = TrackRow(tr, self.project, self, self.tracks_holder); row.set_scale(self._px_per_s())
            self.tracks_layout.addWidget(row)
            if i < len(self.project.tracks) - 1:
                div_r = QFrame(); div_r.setFrameShape(QFrame.HLine); div_r.setFrameShadow(QFrame.Sunken)
                div_r.setStyleSheet("color:#2c2c2c;")
                self.tracks_layout.addWidget(div_r)
                div_l = QFrame(); div_l.setFrameShape(QFrame.HLine); div_l.setFrameShadow(QFrame.Sunken)
                div_l.setStyleSheet("color:#2c2c2c;")
                self.headers_layout.addWidget(div_l)
        self.tracks_layout.addStretch(1)
        self.headers_layout.addStretch(1)


    def _update_time_range(self):
        total = self.compute_project_duration(self.project); self.ruler.set_duration(total); self.ruler.set_markers(self.markers); self.seek_slider.setRange(0,int(total)); self._update_holder_width()

    
    # [scroll-sync] Bind ruler offset to track area's horizontal scroll
    def _on_hscroll_changed(self, value: int):
        try:
            pxs = getattr(self, "_px_per_s", lambda: 100.0)()
            ms = int(max(0, (int(value) / max(1.0, pxs)) * 1000.0))
            self.ruler.set_offset(ms)
        except Exception:
            pass
        except Exception:
            pass


    def _sync_hbar_from_inner(self):
        try:
            sb = self.tracks_area.horizontalScrollBar()
            if getattr(self, "_hbar_syncing", False): return
            self._hbar_syncing = True
            try:
                vpw = max(1, self.tracks_area.viewport().width())
                # Mirror inner scroll range
                self.hbar.setRange(sb.minimum(), sb.maximum())
                self.hbar.setPageStep(sb.pageStep() or vpw)
                self.hbar.setSingleStep(max(20, vpw//10))
                self.hbar.setValue(sb.value())
            finally:
                self._hbar_syncing = False
        except Exception:
            pass

    def _sync_inner_from_hbar(self, v: int):
        try:
            if getattr(self, "_hbar_syncing", False): return
            self._hbar_syncing = True
            try: self.tracks_area.horizontalScrollBar().setValue(int(v))
            finally: self._hbar_syncing = False
        except Exception:
            pass

    def _ensure_playhead_visible(self, margin_px: int = 80, center: bool = False):
        try:
            sb = self.tracks_area.horizontalScrollBar()
            vpw = max(1, self.tracks_area.viewport().width())
            pxs = getattr(self, "_px_per_s", lambda: 100.0)()
            ms = int(self.seek_slider.value())
            x = int((ms/1000.0) * pxs)
            target_x = int(getattr(self, "header_col_w", 0) + x)
            left = sb.value()
            right = left + vpw
            if center:
                sb.setValue(max(0, target_x - vpw//2)); return
            if target_x < left + margin_px:
                sb.setValue(max(0, target_x - margin_px))
            elif target_x > right - margin_px:
                sb.setValue(max(0, target_x - vpw + margin_px))
        except Exception:
            pass
    def _set_zoom(self, z):
        floor = self._min_zoom_fit(1.10)
        self.zoom = max(floor, min(50.0, float(z)))
        pxs=self._px_per_s()
        try: self.ruler.set_scale(pxs)
        except Exception: pass
        try:
            for row in self.tracks_holder.findChildren(TrackRow):
                row.set_scale(pxs)
        except Exception: pass
        self._update_holder_width(); self.zoom_slider.blockSignals(True); self.zoom_slider.setValue(int(self.zoom*10000)); self.zoom_slider.blockSignals(False)
        try: self.zoom_slider.setMinimum(int(max(1, floor*10000)))
        except Exception: pass

    def _px_per_s(self) -> float: return 100.0 * self.zoom

    def _min_zoom_fit(self, extra: float = 1.10) -> float:
        try:
            total_ms = int(self.compute_project_duration(self.project))
            total_s = max(0.001, total_ms/1000.0)
            vieww = max(1, self.tracks_area.viewport().width() - 120)
            pxs_fit = max(0.001, vieww / total_s)
            pxs_min = pxs_fit / max(1.01, float(extra))
            return max(1e-6, pxs_min/100.0)
        except Exception:
            return 1e-6


    def _update_holder_width(self):
        total_ms = self.compute_project_duration(self.project); need = int(self.header_col_w) + int((total_ms/1000.0)*self._px_per_s()) + 120
        if getattr(self,'tracks_holder',None) is not None: self.tracks_holder.setMinimumWidth(need)
        try: self._sync_hbar_from_inner()
        except Exception: pass

        try:
            f = self._min_zoom_fit(1.10)
            if self.zoom < f: self._set_zoom(f)
        except Exception: pass
    def _seek_changed(self, pos_ms: int):
        self.preview_media.emit(None, int(pos_ms)); self.transport_request.emit(f"seek:{int(pos_ms)}")
        try:
            self.ruler.set_playhead(int(pos_ms))
            for row in self.tracks_holder.findChildren(TrackRow):
                row.update()
        except Exception: pass
        try:
            self.tc_edit.setText(self._format_tc(int(pos_ms)))
        except Exception: pass
            # auto-follow
        try: self._ensure_playhead_visible()
        except Exception: pass

    def _goto_edge(self, which: str):
        """Jump playhead to start/end of selection if any, else timeline."""
        # Determine selection bounds
        sels = self._selection[:] if self._selection else ([self._selected] if self._selected else [])
        if sels:
            start = min(c.start_ms for c in sels)
            end = max(c.start_ms + max(1, c.duration_ms) for c in sels)
            target = start if which == 'start' else end
        else:
            target = 0 if which == 'start' else self.compute_project_duration(self.project)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(target))
        self.seek_slider.blockSignals(False)
        self._seek_changed(int(target))


    def _set_playhead_from_ruler(self, ms: int):
        self.seek_slider.blockSignals(True); self.seek_slider.setValue(int(ms)); self.seek_slider.blockSignals(False); self._seek_changed(int(ms))

    # selection helpers
    def _is_selected(self, clip: Clip) -> bool:
        return (clip in self._selection) or (self._selected is clip)

    def _select_clip(self, clip: Optional[Clip], additive=False, toggle=False):
        if clip is None:
            self._selected=None; self._selection.clear()
        else:
            if additive:
                if toggle and clip in self._selection:
                    self._selection.remove(clip)
                else:
                    if clip not in self._selection: self._selection.append(clip)
                self._selected = self._selection[-1] if self._selection else None
            else:
                self._selected = clip; self._selection = [clip]
        # refresh visuals
        for i in range(self.tracks_layout.count()):
            w=self.tracks_layout.itemAt(i).widget()
            if isinstance(w, TrackRow): w.update()

    def _clip_copy(self, track: Track, clip: Clip):
        self._clipboard={"media_id":clip.media_id,"duration_ms":clip.duration_ms,"gain_db":clip.gain_db,"text":clip.text,"speed":clip.speed,"rotation_deg":clip.rotation_deg,"muted":clip.muted,"fade_in_ms":clip.fade_in_ms,"fade_out_ms":clip.fade_out_ms}; self._mark_dirty()

    def _clip_paste(self, track: Track, x_in_row: int, kind_hint: str):
        if not self._clipboard: return
        start_ms=int(max(0, (x_in_row / max(1.0,self._px_per_s()))*1000.0)); 
        start_ms=self._snap_ms(start_ms)
        if track.type=="any": track.type = kind_hint
        if track.type not in ("any", kind_hint): return
        c = Clip(media_id=self._clipboard["media_id"], start_ms=start_ms, duration_ms=int(self._clipboard["duration_ms"]), gain_db=float(self._clipboard["gain_db"]), text=self._clipboard["text"], speed=float(self._clipboard["speed"]), rotation_deg=int(self._clipboard["rotation_deg"]), muted=bool(self._clipboard["muted"]), fade_in_ms=int(self._clipboard["fade_in_ms"]), fade_out_ms=int(self._clipboard["fade_out_ms"]))
        track.clips.append(c); self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

    def _clip_cut(self, track: Track, clip: Clip, mode: str):
        cut_ms=int(self.seek_slider.value()); c_start=clip.start_ms; c_end=clip.start_ms+clip.duration_ms
        if not (c_start < cut_ms < c_end): return
        self._push_undo()
        if mode=="left":
            clip.duration_ms=max(1, cut_ms-c_start)
        elif mode=="right":
            clip.duration_ms=max(1, c_end-cut_ms); clip.start_ms=cut_ms
        else:
            left=Clip(media_id=clip.media_id,start_ms=c_start,duration_ms=cut_ms-c_start,gain_db=clip.gain_db,text=clip.text,speed=clip.speed,rotation_deg=clip.rotation_deg,muted=clip.muted,fade_in_ms=clip.fade_in_ms,fade_out_ms=0)
            right=Clip(media_id=clip.media_id,start_ms=cut_ms,duration_ms=c_end-cut_ms,gain_db=clip.gain_db,text=clip.text,speed=clip.speed,rotation_deg=clip.rotation_deg,muted=clip.muted,fade_in_ms=0,fade_out_ms=clip.fade_out_ms)
            try: idx=track.clips.index(clip); track.clips.pop(idx); track.clips.insert(idx,left); track.clips.insert(idx+1,right)
            except Exception: pass
        self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

    def _clip_set_fade(self, clip: Clip, start_ms: Optional[int]=None, end_ms: Optional[int]=None):
        if start_ms is not None: clip.fade_in_ms=int(start_ms)
        if end_ms is not None: clip.fade_out_ms=int(end_ms)
        self._refresh_tracks(); self._mark_dirty()

    def _clip_set_mute(self, clip: Clip, mute: bool): clip.muted=bool(mute); self._mark_dirty()
    def _clip_set_gain(self, clip: Clip):
        try:
            val,ok=QInputDialog.getDouble(self,"Volume (dB)","Gain:", clip.gain_db, -60.0, 12.0, 1)
            if ok: clip.gain_db=float(val); self._mark_dirty()
        except Exception: pass

    def _clip_speed(self, clip: Clip, factor: float):
        factor=max(0.1, min(8.0, float(factor))); 
        if factor!=0: clip.duration_ms=int(max(1, round(clip.duration_ms / factor)))
        clip.speed*=factor; self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

    def _clip_rotate(self, clip: Clip, delta_deg: int):
        clip.rotation_deg=int((clip.rotation_deg+delta_deg)%360); self._mark_dirty()

    def _clip_separate_audio(self, track: Track, clip: Clip):
        tgt=None
        for tr in self.project.tracks:
            if tr.type=="audio": tgt=tr; break

    # ======= Internal media player wiring =======
    def _get_internal_player(self):
        """Try to find the app's main QMediaPlayer (video/audio) instance."""
        p = getattr(self, "_ext_player", None)
        try:
            if p and hasattr(p, "play") and hasattr(p, "position"):
                return p
        except Exception:
            pass
        try_objs = []
        try:
            w = self.window()
            if w: try_objs.append(w)
        except Exception:
            w = None
        candidates = []
        for obj in [self, w, getattr(self, "parent", lambda: None)()]:
            if obj: candidates += [obj, getattr(obj, "player", None), getattr(obj, "video", None),
                                   getattr(obj, "media_player", None), getattr(obj, "preview", None)]
        try_objs.extend([c for c in candidates if c])
        for obj in list(dict.fromkeys(try_objs)):
            try:
                if hasattr(obj, "play") and hasattr(obj, "position") and hasattr(obj, "setPosition"):
                    self._ext_player = obj
                    return obj
                pl = getattr(obj, "player", None)
                if pl and hasattr(pl, "play") and hasattr(pl, "position"):
                    self._ext_player = pl
                    return pl
            except Exception:
                pass
        try:
            from PySide6.QtCore import QObject
            queue = []
            if w and isinstance(w, QObject):
                queue = [w]
            seen = set()
            while queue:
                cur = queue.pop(0)
                if id(cur) in seen:
                    continue
                seen.add(id(cur))
                try:
                    for cand in [cur, getattr(cur, "player", None), getattr(cur, "media_player", None)]:
                        if cand and hasattr(cand, "play") and hasattr(cand, "position") and hasattr(cand, "setPosition"):
                            self._ext_player = cand
                            return cand
                    for ch in cur.children():
                        queue.append(ch)
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _refresh_play_icon(self):
        try:
            st = self.style()
            p = self._get_internal_player()
            if p and hasattr(p, "playbackState"):
                state = int(getattr(p, "playbackState")())
                if state == 1:
                    self.btn_play_pause.setIcon(st.standardIcon(QStyle.SP_MediaPause))
                else:
                    self.btn_play_pause.setIcon(st.standardIcon(QStyle.SP_MediaPlay))
            else:
                self.btn_play_pause.setIcon(st.standardIcon(QStyle.SP_MediaPlay))
        except Exception:
            pass

    def _hook_player_signals(self):
        p = self._get_internal_player()
        if not p:
            return
        try:
            if hasattr(p, "playbackStateChanged"):
                p.playbackStateChanged.connect(lambda _s: self._refresh_play_icon());
                getattr(p, 'positionChanged', lambda *_: None).connect(self._on_ext_pos)
        except Exception:
            pass
        self._refresh_play_icon()

    def _tr_toggle(self):
        """Play/Pause timeline from the ruler. Guarantees playback via internal player or background."""
        # If gap playback is running, pause it first
        try:
            if getattr(self, '_gap_playing', False):
                self._stop_gap_playback(); self._play_continuous = False
                self._refresh_play_icon(); return
        except Exception: pass
        started_ext = False
        started_bg = False
        p = self._get_internal_player()
        # If either player is currently playing, pause both (toggle behavior)
        try:
            if p and hasattr(p, "playbackState") and int(p.playbackState()) == 1:
                p.pause()
                try: self.transport_request.emit("pause")
                except Exception: pass
                try:
                    if getattr(self, "_bg_player", None): self._bg_player.pause()
                except Exception:
                    pass
                self._refresh_play_icon()
                return
        except Exception:
            pass
        # Try internal player first
        try:
            cw = getattr(self, '_clipwidget_under_playhead', lambda: None)()
            # If we already prepared this clip/source earlier, just resume without re-opening
            try:
                same = False
                clip = cw.clip if hasattr(cw, 'clip') else cw
                media_id = getattr(clip, 'media_id', '') if clip is not None else ''
                media = self.project.media.get(media_id, None) if hasattr(self.project, 'media') else None
                src = str(getattr(media, 'path', '')) if media else ''
                if src and src == getattr(self, '_ext_source_path', None) and clip is getattr(self, '_ext_playing_clip', None):
                    same = True
            except Exception:
                same = False
            if p and (same or self._ensure_ext_player_for_clip(cw)):
                p.play(); started_ext = True
                try: self.transport_request.emit("play")
                except Exception: pass
        except Exception:
            pass
        # If internal failed, ensure background playback
        if not started_ext:
            try:
                started_bg = self._play_timeline_under_ruler_background()
                if not started_bg and cw:
                    self._play_clip_from_ruler(cw); started_bg = True
            except Exception:
                pass
        # If nothing started at all, begin gap playback (black screen)
        if not started_ext and not started_bg:
            try: self._start_gap_playback()
            except Exception: pass
        # If both started, prefer internal and stop background to avoid double audio
        if started_ext and getattr(self, "_bg_player", None):
            try: self._bg_player.stop()
            except Exception: pass
        self._refresh_play_icon()

    def _tr_stop(self):
        p = self._get_internal_player()
        if p is not None:
            try:
                p.stop()
            except Exception:
                pass
        try:
            if getattr(self, "_bg_player", None):
                self._bg_player.stop()
        except Exception:
            pass
        self._ext_playing_clip = None
        try: self._stop_gap_playback()
        except Exception: pass
        try: self.preview_media.emit(None, int(getattr(self.seek_slider, 'value', lambda:0)()))
        except Exception: pass
        try: self.transport_request.emit("stop")
        except Exception: pass
        try: self._refresh_play_icon()
        except Exception: pass
    def _tr_seek(self, ms: int):
        try:
            ms = max(0, int(ms))
        except Exception:
            ms = 0
        try:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(ms)
            self.seek_slider.blockSignals(False)
        except Exception:
            pass
        try: self.ruler.set_playhead(int(ms))
        except Exception: pass
        try:
            for row in self.tracks_holder.findChildren(TrackRow):
                row.update()
        except Exception: pass
        try: self._ensure_playhead_visible()
        except Exception: pass
        try:
            if hasattr(self, "_ext_playing_clip") and self._ext_playing_clip:
                p = self._get_internal_player()
                if p:
                    rel = int(ms) - int(getattr(self._ext_playing_clip, "start_ms", 0))
                    if rel >= 0:
                        p.setPosition(int(rel))
        except Exception:
            pass
    def _tr_start(self):
        self._tr_seek(0)

    def _tr_end(self):
        try:
            total_ms = int(self.compute_project_duration(self.project))
        except Exception:
            total_ms = 0
        if total_ms > 0:
            self._tr_seek(max(0, total_ms - 1))
    def _tr_step(self, delta_ms: int = 1000):
        try:
            pos = int(self.seek_slider.value())
        except Exception:
            pos = 0
        self._tr_seek(max(0, pos + int(delta_ms)))
    def _clip_remove_from_timeline(self, track: Track, clip: Clip):
        try: self._push_undo(); track.clips.remove(clip); self._refresh_tracks(); self._update_time_range(); self._mark_dirty()
        except Exception: pass

    def _ripple_delete(self, track: Track, clip: Clip):
        # remove and close gap on the same track
        try:
            self._push_undo()
            idx = track.clips.index(clip)
            start = clip.start_ms; dur = clip.duration_ms; track.clips.pop(idx)
            for c in track.clips:
                if c.start_ms >= start + dur:
                    c.start_ms -= dur
            self._refresh_tracks(); self._update_time_range(); self._mark_dirty()
        except Exception: pass

    def _clip_remove_from_project(self, track: Track, clip: Clip):
        try:
            self._push_undo(); mid=clip.media_id
            try: track.clips.remove(clip)
            except Exception: pass
            for tr in self.project.tracks: tr.clips = [c for c in tr.clips if c.media_id != mid]
            self.project.media.pop(mid, None); self._refresh_ui(); self._update_time_range(); self._mark_dirty()
        except Exception: pass
    def _clip_trim_to_playhead(self, clip: Clip, side: str):
        """Trim front ('L') or end ('R') of clip to current playhead position."""
        cut_ms = int(self.seek_slider.value())
        c_start = int(clip.start_ms)
        c_end = int(clip.start_ms + max(1, clip.duration_ms))
        if side == 'L':
            # keep right side
            if cut_ms <= c_start: return
            if cut_ms >= c_end: 
                clip.duration_ms = 1
                return
            clip.duration_ms = max(1, c_end - cut_ms)
            clip.start_ms = cut_ms
        else:
            # side == 'R', keep left side
            if cut_ms >= c_end: return
            if cut_ms <= c_start:
                clip.duration_ms = 1
                return
            clip.duration_ms = max(1, cut_ms - c_start)
        self._mark_dirty()

    def _clip_reset_to_max(self, clip: Clip):
        """Reset clip duration to media's maximum known duration (if available)."""
        m = self.project.media.get(clip.media_id)
        if m and m.meta.get('duration'):
            try:
                max_ms = max(1, int(float(m.meta['duration']) * 1000))
                clip.duration_ms = max_ms
                self._mark_dirty()
            except Exception:
                pass


    # undo/redo
    def _snapshot(self) -> Dict[str,Any]: return self.project.to_json()
    def _push_undo(self): self._undo_stack.append(self._snapshot()); self._redo_stack.clear()
    def _undo(self):
        if not self._undo_stack: return
        self._redo_stack.append(self._snapshot()); js=self._undo_stack.pop(); self.project=Project.from_json(js); self._refresh_ui(); self._mark_dirty()
    def _redo(self):
        if not self._redo_stack: return
        self._undo_stack.append(self._snapshot()); js=self._redo_stack.pop(); self.project=Project.from_json(js); self._refresh_ui(); self._mark_dirty()

    # helpers
    def _track_index(self, track: Track) -> Optional[int]:
        try: return self.project.tracks.index(track)
        except ValueError: return None

    @staticmethod
    def compute_project_duration(p: Project) -> int:
        end=0
        for tr in p.tracks:
            for c in tr.clips:
                end=max(end, c.start_ms + max(1,c.duration_ms))
        return max(1000, end)

    # export (stub)

    # --- helpers for adding media ---
    def _ensure_track_for_kind(self, kind: str, near_index: int | None = None) -> Track:
        """Return a track that accepts 'kind' (video/image/audio/text). Create one if needed."""
        # prefer an existing compatible, unlocked track
        for i, tr in enumerate(self.project.tracks):
            if (tr.type in ("any", kind)) and not tr.locked:
                return tr
        # create new track near the drop location
        name = kind.capitalize()
        tr = Track(name, kind)
        if near_index is not None and 0 <= near_index <= len(self.project.tracks):
            self.project.tracks.insert(near_index+1, tr)
        else:
            self.project.tracks.append(tr)
        self._refresh_tracks()
        return tr

    def _add_media_clip(self, target_track: Track, media_id: str, start_ms: int):
        m = self.project.media.get(media_id)
        if not m:
            return
        # If target track is flexible, adopt the kind
        if target_track.type == "any":
            target_track.type = m.kind
            try: self._update_track_labels()
            except Exception: pass
        # duration from probe if available, else default 3s
        dur = 3000
        md = m.meta or {}
        if md.get("duration"):
            try: dur = int(float(md["duration"]) * 1000)
            except Exception: pass
        target_track.clips.append(Clip(media_id=media_id, start_ms=int(max(0, start_ms)), duration_ms=dur))
        self._refresh_tracks(); self._update_time_range(); self._mark_dirty()
    def _export(self):
        dlg = ExportDialog(self, self.project.width, self.project.height)
        if dlg.exec()!=QDialog.Accepted: return
        opts=dlg.values(); self.project.width=opts["width"]; self.project.height=opts["height"]; self.project.fps=opts["fps"]
        out=Path(opts["output"]); out.parent.mkdir(parents=True, exist_ok=True)
        QMessageBox.information(self,"Export",f"(Stub) Export graph wiring will go here. Range: " + ("IN-OUT" if (self.in_ms is not None and self.out_ms is not None and self.out_ms>self.in_ms) else "ALL"))

    # snapping
    def _toggle_grid(self, on): self.grid_enabled = bool(on); self._update_time_range()

    def _snap_ms(self, ms: int) -> int:
        if not self.glue_enabled and not self.grid_enabled: return int(ms)
        px_tol=float(self.snap_px_tol); tol=int(round((px_tol / max(1e-3, self._px_per_s())) * 1000.0))
        cands=[0, int(self.seek_slider.value())] + [int(m) for m in getattr(self,'markers',[])]
        for tr in self.project.tracks:
            for c in tr.clips:
                cands.extend([int(c.start_ms), int(c.start_ms + max(1,c.duration_ms))])
        # grid points
        if self.grid_enabled:
            total_ms = self.compute_project_duration(self.project)
            step_ms = int(self.grid_step_s * 1000)
            g = 0
            while g <= total_ms + step_ms:
                cands.append(int(g)); g += step_ms
        best=int(ms); bestd=tol+1
        for c in cands:
            d=abs(int(ms)-int(c))
            if d < bestd and d <= tol: best, bestd = int(c), d
        return best

    # autosave
    def _mark_dirty(self): self._dirty=True; self._last_change_ts=time.time()
    def _autosave_tick(self):
        now=time.time(); should = self._dirty and (now - self._last_save_ts >= 60)
        if self._dirty and (now - self._last_change_ts >= 120): should=True
        if should:
            try:
                self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
                self._autosave_path.write_text(json.dumps(self.project.to_json(), indent=2), encoding="utf-8")
                self._last_save_ts = now
                if now - self._last_change_ts >= 120: self._dirty=False
            except Exception: pass

    # autoplace in tab bar
    def _find_parent_tabwidget(self):
        p=self.parent()
        while p and not isinstance(p, QTabWidget): p=p.parent()
        return p
    def _find_tabwidget_anywhere(self):
        tw=self._find_parent_tabwidget()
        if tw: return tw
        try:
            win=self.window()
            if win: 
                tw=win.findChild(QTabWidget)
                if tw and tw.indexOf(self)!=-1: return tw
        except Exception: pass
        app=QApplication.instance()
        if app:
            for w in app.allWidgets():
                if isinstance(w, QTabWidget):
                    try:
                        if w.indexOf(self)!=-1: return w
                    except Exception: pass
        return None
    def _autoplace_try(self):
        tabs=self._find_tabwidget_anywhere()
        if not tabs: return False
        try:
            container=self; idx=-1
            while container is not None:
                try: idx=tabs.indexOf(container); 
                except Exception: pass
                if idx!=-1: break
                container=container.parent()
            if idx==-1: return False
            tabs.setTabText(idx,'Editor')
            try: tabs.tabBar().moveTab(idx,0)
            except Exception:
                w=tabs.widget(idx); text=tabs.tabText(idx) or 'Editor'; tabs.removeTab(idx); tabs.insertTab(0,w,text)
            tabs.setCurrentIndex(0); return True
        except Exception: return False
    def _start_autoplace_timer(self):
        self._autoplace_attempts=0; self._autoplace_timer=QTimer(self); self._autoplace_timer.setInterval(200)
        def tick():
            ok=self._autoplace_try(); self._autoplace_attempts+=1
            if ok or self._autoplace_attempts>30: self._autoplace_timer.stop()
        self._autoplace_timer.timeout.connect(tick); self._autoplace_timer.start(); QTimer.singleShot(0, tick)


    

    # --- background playback ---
    def _play_clip_from_ruler(self, clipw):
        """Play a clip using the background player starting from the current ruler position."""
        if not HAS_QTMULTI: return
        try:
            from PySide6.QtCore import QUrl
            media = clipw.media
            if media.kind not in ("audio","video"):
                return
            # source
            self._bg_player.setSource(QUrl.fromLocalFile(str(media.path)))
            self._playing_clip = clipw
            # compute relative start from global playhead
            g = int(self.seek_slider.value())
            rel = max(0, g - int(clipw.clip.start_ms))
            # cap within clip duration
            rel = min(rel, max(0, int(clipw.clip.duration_ms)-1))
            try: self._bg_player.setPosition(int(rel))
            except Exception: pass
            try: self._bg_player.play()
            except Exception: pass
            self._is_playing = True
        except Exception:
            pass

    def _on_bg_pos(self, ms):
        """Sync playhead/ruler to background player position and clamp to clip end."""
        try:
            if not self._is_playing or not self._playing_clip: return
            clip = self._playing_clip.clip
            # Stop at end of clip
            if int(ms) >= int(clip.duration_ms):
                try: self._bg_player.pause()
                except Exception: pass
                try:
                    if getattr(self, '_play_continuous', False):
                        self._start_gap_playback()
                except Exception: pass
                # clamp and update
                except Exception: pass
                self._is_playing = False
                ms = int(clip.duration_ms) - 1
            g = int(clip.start_ms) + int(ms)
            # update seek slider without feedback loops
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(g))
            self.seek_slider.blockSignals(False)
            # update ruler + rows
            try: self.ruler.set_playhead(int(g))
            except Exception: pass
            try:
                for row in self.tracks_holder.findChildren(TrackRow):
                    row.update()
            except Exception: pass
            try: self._ensure_playhead_visible()
            except Exception: pass
        except Exception:
            pass


    def _ensure_ext_player_for_clip(self, clipw=None):
        """Prepare the internal player to play the clip under the ruler and seek appropriately."""
        try:
            from PySide6.QtCore import QUrl
        except Exception:
            QUrl = None
        p = self._get_internal_player()
        if not p:
            return False
        clip = clipw.clip if hasattr(clipw, "clip") else clipw
        if clip is None:
            clip = self._clip_under_playhead()
            if clip is None:
                return False
        try:
            media_id = getattr(clip, "media_id", "")
            media = self.project.media.get(media_id, None)
        except Exception:
            media = None
        if not media or not hasattr(media, "path"):
            return False
        url = None
        try:
            if QUrl is not None:
                url = QUrl.fromLocalFile(str(media.path))
        except Exception:
            url = None
        try:
            if hasattr(p, "setSource") and url is not None:
                p.setSource(url)
            elif hasattr(p, "setMedia"):
                try:
                    from PySide6.QtMultimedia import QMediaContent
                    p.setMedia(QMediaContent(url))
                except Exception:
                    p.setMedia(url)
        except Exception:
            pass
        try:
            g = int(self.seek_slider.value())
        except Exception:
            g = 0
        rel = max(0, g - int(getattr(clip, "start_ms", 0)))
        try:
            dur = int(getattr(clip, "duration_ms", 0))
            rel = min(rel, max(0, dur-1))
        except Exception:
            pass
        try: p.setPosition(int(rel))
        except Exception: pass
        self._ext_playing_clip = clip
        try: self._ext_source_path = str(media.path)
        except Exception: self._ext_source_path = None
        try:
            self.preview_media.emit(getattr(media, "path", None), int(g))
        except Exception:
            pass
        try:
            if hasattr(p, "positionChanged"):
                p.positionChanged.connect(self._on_ext_pos)
        except Exception:
            pass
        return True

    def _on_ext_pos(self, ms):
        """Sync timeline UI to internal player position while previewing a clip."""
        try:
            clip = getattr(self, "_ext_playing_clip", None)
            if clip is None:
                return
            p = self._get_internal_player()
            if not p:
                return
            dur = int(getattr(clip, "duration_ms", 0))
            rel = int(ms)
            if dur and rel >= dur:
                try: p.pause()
                except Exception: pass
                rel = max(0, dur-1)
            g = int(getattr(clip, "start_ms", 0)) + int(rel)
            self.seek_slider.blockSignals(True); self.seek_slider.setValue(int(g)); self.seek_slider.blockSignals(False)
            try: self.ruler.set_playhead(int(g))
            except Exception: pass
            try:
                for row in self.tracks_holder.findChildren(TrackRow):
                    row.update()
            except Exception: pass
            try: self._ensure_playhead_visible()
            except Exception: pass
        except Exception:
            pass

    def _set_global_time(self, ms: int):
        """Set global playhead and, if a clip is active, keep player in sync."""
        try:
            ms = max(0, int(ms))
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(ms)
            self.seek_slider.blockSignals(False)
            if self._playing_clip and self._bg_player:
                rel = ms - int(self._playing_clip.clip.start_ms)
                if 0 <= rel < int(self._playing_clip.clip.duration_ms):
                    if abs(int(self._bg_player.position()) - int(rel)) > 60:
                        try: self._bg_player.setPosition(int(rel))
                        except Exception: pass
                    if not self._is_playing:
                        try: self._bg_player.play()
                        except Exception: pass
                        self._is_playing = True
                else:
                    try: self._bg_player.pause()
                    except Exception: pass
                    self._is_playing = False
            try: self._ensure_playhead_visible()
            except Exception: pass
        except Exception:
            pass

    def _on_transport(self, cmd: str):
        """Minimal transport handling for background player."""
        try:
            if cmd.startswith("seek:"):
                self._set_global_time(int(cmd.split(":")[1]))
                return
            if cmd == "toggle":
                if not self._bg_player: return
                if self._is_playing:
                    try: self._bg_player.pause()
                    except Exception: pass
                    self._is_playing = False
                else:
                    if self._playing_clip:
                        try: self._bg_player.play()
                        except Exception: pass
                        self._is_playing = True
                return
            if cmd == "pause":
                if self._bg_player:
                    try: self._bg_player.pause()
                    except Exception: pass
                    self._is_playing = False
                return
        except Exception:
            pass

    # --- gap playback helpers ---
    def _next_clip_start_after(self, g:int) -> int:
        try: g = int(g)
        except Exception: g = 0
        nxt = None
        try:
            for tr in getattr(self.project, 'tracks', []):
                for cl in getattr(tr, 'clips', []):
                    try:
                        st = int(getattr(cl, 'start_ms', 0))
                        if st >= g and (nxt is None or st < nxt):
                            nxt = st
                    except Exception: pass
        except Exception: pass
        if nxt is None:
            try: nxt = int(self.seek_slider.maximum())
            except Exception: nxt = g + 3600000  # 1h horizon
        return int(nxt)

    def _start_gap_playback(self):
        try: self.preview_media.emit(None, int(getattr(self.seek_slider, 'value', lambda:0)()))
        except Exception: pass
        self._gap_playing = True; self._play_continuous = True
        try: self._gap_timer.start()
        except Exception: pass

    def _stop_gap_playback(self):
        try:
            if getattr(self, '_gap_timer', None): self._gap_timer.stop()
        except Exception: pass
        self._gap_playing = False

    def _gap_tick(self):
        try:
            if not getattr(self, '_gap_playing', False): return
            cur = int(getattr(self.seek_slider, 'value', lambda:0)())
            end = int(self._next_clip_start_after(cur))
            step = 33
            new_g = cur + step
            if new_g >= end:
                new_g = end
            try: self._set_global_time(int(new_g))
            except Exception:
                try:
                    self.seek_slider.blockSignals(True); self.seek_slider.setValue(int(new_g)); self.seek_slider.blockSignals(False)
                except Exception: pass
            try: self.preview_media.emit(None, int(new_g))
            except Exception: pass
            if new_g >= end:
                self._stop_gap_playback()
                cw = None
                try: cw = self._clipwidget_under_playhead()
                except Exception: cw = None
                if cw:
                    try: self._play_timeline_under_ruler_background()
                    except Exception: pass
        except Exception:
            pass

    def _apply_styles(self):
        # Theme-aware greys so buttons look good on dark and light themes
        pal = self.palette()
        bg = pal.window().color()
        # Compute luminance
        def lum(qc):
            r,g,b = qc.redF(), qc.greenF(), qc.blueF()
            return 0.2126*r + 0.7152*g + 0.0722*b
        L = lum(bg)
        # Neutral greys blended from the current theme
        def mix_hex(qc, target=(60,65,72), alpha=0.35):
            tr,tg,tb = target
            r = int((1-alpha)*qc.red()   + alpha*tr)
            g = int((1-alpha)*qc.green() + alpha*tg)
            b = int((1-alpha)*qc.blue()  + alpha*tb)
            return f"#{r:02x}{g:02x}{b:02x}"
        def lighten_hex(qc, amt=0.15):
            r = min(255, int(qc.red()*(1-amt) + 255*amt))
            g = min(255, int(qc.green()*(1-amt) + 255*amt))
            b = min(255, int(qc.blue()*(1-amt) + 255*amt))
            return f"#{r:02x}{g:02x}{b:02x}"
        def darken_hex(qc, amt=0.20):
            r = max(0, int(qc.red()*(1-amt)))
            g = max(0, int(qc.green()*(1-amt)))
            b = max(0, int(qc.blue()*(1-amt)))
            return f"#{r:02x}{g:02x}{b:02x}"
        # Base shades
        btn_bg   = mix_hex(bg, target=(62,68,76), alpha=0.50)   # neutral grey
        btn_brd  = mix_hex(bg, target=(86,96,110), alpha=0.55)  # slightly brighter border
        btn_hov  = mix_hex(bg, target=(96,108,124), alpha=0.60)
        inp_bg   = mix_hex(bg, target=(64,70,78), alpha=0.35)
        text_col = "#0f172a" if L > 0.6 else "#e5e7eb"  # dark text on light themes
        green    = "#2ecc71"
        green_b  = "#25b864"
        green_t  = "#0b1a0f"
        self.setStyleSheet(f"""    QToolButton {{
            background: {btn_bg};
            border: 1px solid {btn_brd};
            padding: 4px 8px;
            border-radius: 8px;
            color: {text_col};
        }}
        QToolButton:hover {{
            border-color: {btn_hov};
        }}
        QToolButton:checked {{
            background: {green};
            border: 1px solid {green_b};
            color: {green_t};
        }}
        QToolButton:pressed {{
            background: {green};
        }}
        /* Start/End outline to remain visible on light themes */
        #btn_go_start, #btn_go_end {{
            border-color: {green_b};
        }}
        QComboBox, QSpinBox, QLineEdit {{
            background: {inp_bg};
            color: {text_col};
            border: 1px solid {btn_brd};
            border-radius: 8px;
            padding: 2px 6px;
        }}
        """)

    def keyPressEvent(self, ev):
        key = ev.key()
        mod = ev.modifiers()
        # nudge left/right
        if key in (Qt.Key_Left, Qt.Key_Right):
            frames = 10 if (mod & Qt.ShiftModifier) else 1
            step = int(round(1000.0 * frames / max(1,self.project.fps)))
            delta = -step if key==Qt.Key_Left else step
            targets = self._selection[:] if self._selection else ([self._selected] if self._selected else [])
            if not targets: return
            self._push_undo()
            for c in targets:
                c.start_ms = max(0, c.start_ms + delta)
            self._refresh_tracks(); self._update_time_range(); self._mark_dirty(); ev.accept(); return
        # move to prev/next compatible timeline
        if key in (Qt.Key_Up, Qt.Key_Down):
            up = (key==Qt.Key_Up)
            targets = self._selection[:] if self._selection else ([self._selected] if self._selected else [])
            if not targets: return
            self._push_undo()
            for c in targets:
                # find current track
                ti = None; ct=None
                for i,tr in enumerate(self.project.tracks):
                    if c in tr.clips: ti=i; ct=tr; break
                if ti is None: continue
                rng = range(ti-1, -1, -1) if up else range(ti+1, len(self.project.tracks))
                m = self.project.media.get(c.media_id); kind = m.kind if m else None
                for j in rng:
                    tr = self.project.tracks[j]
                    if tr.type in ("any", kind):
                        # move
                        try: ct.clips.remove(c)
                        except Exception: pass
                        if tr.type=="any": tr.type = kind or tr.type
                        tr.clips.append(c)
                        break
            self._refresh_tracks(); self._update_time_range(); self._mark_dirty(); ev.accept(); return
        # delete / ripple delete
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            targets = self._selection[:] if self._selection else ([self._selected] if self._selected else [])
            if not targets: return
            # find their track for ripple
            if (mod & Qt.ControlModifier) or (mod & Qt.ShiftModifier):
                for c in targets:
                    for tr in self.project.tracks:
                        if c in tr.clips:
                            self._ripple_delete(tr, c); break
            else:
                for c in targets:
                    for tr in self.project.tracks:
                        if c in tr.clips:
                            self._clip_remove_from_timeline(tr, c); break
            ev.accept(); return
        super().keyPressEvent(ev)

    
    # ---- timecode helpers ----
    def _format_tc(self, ms: int) -> str:
        fps = max(1, int(self.project.fps))
        total = max(0, int(ms))
        sec = total // 1000
        hh = sec // 3600
        mm = (sec % 3600) // 60
        ss = sec % 60
        ff = int(round(((total % 1000) / 1000.0) * fps)) % fps
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    def _parse_tc(self, s: str) -> Optional[int]:
        try:
            parts = s.strip().split(":")
            if len(parts) != 4: return None
            hh,mm,ss,ff = [int(x) for x in parts]
            fps = max(1, int(self.project.fps))
            total_ms = ((hh*3600 + mm*60 + ss) * 1000) + int(round((ff / float(fps)) * 1000.0))
            return max(0, total_ms)
        except Exception:
            return None

    def _jump_to_timecode(self):
        ms = self._parse_tc(self.tc_edit.text())
        if ms is None: return
        self._set_playhead_from_ruler(int(ms))

    # ---- in/out range ----
    def _set_in_here(self):
        self.in_ms = int(self.seek_slider.value()); self._update_io_ui()

    def _set_out_here(self):
        self.out_ms = int(self.seek_slider.value()); self._update_io_ui()

    def _clear_in_out(self):
        self.in_ms = None; self.out_ms = None; self._update_io_ui()

    def _update_io_ui(self):
        # normalize order
        if self.in_ms is not None and self.out_ms is not None and self.out_ms < self.in_ms:
            self.in_ms, self.out_ms = self.out_ms, self.in_ms
        try:
            if hasattr(self, "lbl_io"):
                if self.in_ms is not None or self.out_ms is not None:
                    t_in = self._format_tc(self.in_ms or 0)
                    t_out = self._format_tc(self.out_ms or 0)
                    self.lbl_io.setText(f"[{t_in} ➜ {t_out}]")
                else:
                    self.lbl_io.setText("—")
        except Exception: pass
        try:
            self.ruler.set_in_out(self.in_ms, self.out_ms)
            self.ruler.set_fps(self.project.fps)
        except Exception: pass

    def _remove_range(self, ripple: bool):
        if self.in_ms is None or self.out_ms is None: return
        a, b = int(self.in_ms), int(self.out_ms)
        if b <= a: return
        self._push_undo()
        delta = b - a
        for tr in self.project.tracks:
            new_clips = []
            for c in tr.clips:
                s = int(c.start_ms); e = int(c.start_ms + max(1, c.duration_ms))
                if e <= a:
                    new_clips.append(c)
                elif s >= b:
                    # after range
                    if ripple: c.start_ms -= delta
                    new_clips.append(c)
                else:
                    # overlaps
                    if s < a and e > b:
                        # split into two; keep both
                        left = Clip(media_id=c.media_id, start_ms=s, duration_ms=max(1, a - s),
                                    speed=c.speed, rotation_deg=c.rotation_deg, muted=c.muted,
                                    fade_in_ms=c.fade_in_ms, fade_out_ms=0, opacity=c.opacity, gain_db=c.gain_db, group_id=c.group_id)
                        right = Clip(media_id=c.media_id, start_ms=b, duration_ms=max(1, e - b),
                                     speed=c.speed, rotation_deg=c.rotation_deg, muted=c.muted,
                                     fade_in_ms=0, fade_out_ms=c.fade_out_ms, opacity=c.opacity, gain_db=c.gain_db, group_id=c.group_id)
                        if ripple: right.start_ms -= delta
                        new_clips.append(left); new_clips.append(right)
                    elif s < a < e <= b:
                        # keep left side only
                        c.duration_ms = max(1, a - s)
                        new_clips.append(c)
                    elif a <= s < b < e:
                        # keep right side only
                        c.start_ms = b
                        if ripple: c.start_ms -= delta
                        c.duration_ms = max(1, e - b)
                        new_clips.append(c)
                    else:
                        # fully inside: drop
                        pass
            tr.clips = sorted(new_clips, key=lambda x: x.start_ms)
        self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

    def _trim_to_range(self):
        if self.in_ms is None or self.out_ms is None: return
        a, b = int(self.in_ms), int(self.out_ms)
        if b <= a: return
        self._push_undo()
        targets = self._selection[:] if self._selection else None
        for tr in self.project.tracks:
            new_clips = []
            for c in tr.clips:
                if targets is not None and c not in targets:
                    new_clips.append(c); continue
                s = int(c.start_ms); e = int(c.start_ms + max(1,c.duration_ms))
                # intersection
                ns = max(s, a); ne = min(e, b)
                if ne <= ns:
                    # no overlap => drop (if targeting all, remove; if selection-only and no overlap, keep?)
                    if targets is None:
                        pass
                    else:
                        new_clips.append(c)
                    continue
                c.start_ms = ns; c.duration_ms = max(1, ne - ns)
                new_clips.append(c)
            tr.clips = sorted(new_clips, key=lambda x: x.start_ms)
        self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

    # ---- shuttle (J K L) ----
    def _shuttle(self, key: str):
        rates = [1,2,4,8]
        if key == "K":
            self._shuttle_rate = 0
            self.transport_request.emit("pause")
            return
        if key in ("J","L"):
            sign = -1 if key == "J" else 1
            if self._shuttle_rate == 0 or (self._shuttle_rate * sign) < 0:
                self._shuttle_rate = sign * rates[0]
            else:
                # increase magnitude
                mag = abs(self._shuttle_rate)
                try:
                    idx = rates.index(mag); idx = min(idx+1, len(rates)-1)
                except ValueError:
                    idx = 0
                self._shuttle_rate = sign * rates[idx]
            self.transport_request.emit(f"rate:{self._shuttle_rate}")
# ---- markers ----
    def _add_marker_here(self):
        ms = int(self.seek_slider.value())
        self.markers.append(ms); self.markers = sorted(set(self.markers)); self._update_time_range(); self._mark_dirty()

    def _goto_next_marker(self):
        ms = int(self.seek_slider.value())
        for m in self.markers:
            if m > ms: self._set_playhead_from_ruler(m); return

    def _goto_prev_marker(self):
        ms = int(self.seek_slider.value())
        for m in reversed(self.markers):
            if m < ms: self._set_playhead_from_ruler(m); return

    # ---- grouping ----
    def _group_selection(self):
        if len(self._selection) < 2: return
        gid = self._next_group_id; self._next_group_id += 1
        for c in self._selection: c.group_id = gid
        self._mark_dirty()

    def _ungroup_selection(self):
        for c in self._selection: c.group_id = None
        self._mark_dirty()

    # ---- zoom helpers ----
    def _zoom_to_fit(self):
        total_ms = self.compute_project_duration(self.project)
        if total_ms <= 0: return
        # compute px_per_s to fit into current viewport width
        vieww = max(1, self.tracks_area.viewport().width()-120)
        pxs = max(20.0, vieww / max(0.001, total_ms/1000.0))
        self._set_zoom(pxs/100.0)

    def _zoom_to_selection(self):
        target = self._selected or (self._selection[0] if self._selection else None)
        if not target: return
        dur = max(1, getattr(target, "duration_ms", 0) or 1)
        vieww = max(1, self.tracks_area.viewport().width()-120)
        pxs = max(20.0, vieww / max(0.001, dur/1000.0))
        self._set_zoom(pxs/100.0)


    def _zoom_fit_all(self):
        """Zoom out to fit the entire project into the visible viewport."""
        try:
            total_ms = int(self.compute_project_duration(self.project))
            total_s = max(0.001, total_ms/1000.0)
            vieww = max(1, self.tracks_area.viewport().width() - 120)
            pxs = max(0.001, vieww / total_s)
            self._set_zoom(pxs/100.0)
            try: self._ensure_playhead_visible(center=True)
            except Exception: pass
        except Exception:
            pass

        target = self._selected or (self._selection[0] if self._selection else None)
        if not target: return
        dur = max(1, target.duration_ms)
        vieww = max(1, self.tracks_area.viewport().width()-120)
        pxs = max(20.0, vieww / max(0.001, dur/1000.0))
        self._set_zoom(pxs/100.0)

# Standalone test
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = EditorPane(); w.resize(1200, 700); w.show()
    sys.exit(app.exec())

def _palette_icon(size:int=16) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(0,0,0,0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor(244, 208, 63))
    p.setPen(QPen(QColor(120, 90, 20), 1))
    p.drawEllipse(1, 1, size-2, size-2)
    dots = [QColor(231,76,60), QColor(46,204,113), QColor(52,152,219), QColor(155,89,182)]
    centers = [(int(size*0.35), int(size*0.35)), (int(size*0.60), int(size*0.30)), (int(size*0.65), int(size*0.60)), (int(size*0.35), int(size*0.65))]
    r = max(2, size//7)
    for c,(cx,cy) in zip(dots, centers):
        p.setBrush(c); p.setPen(Qt.NoPen)
        p.drawEllipse(cx-r, cy-r, r*2, r*2)
    p.end()
    return QIcon(pm)