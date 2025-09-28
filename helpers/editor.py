
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
                               QDialogButtonBox, QFormLayout, QAbstractItemView, QApplication, QTabWidget, QInputDialog, QColorDialog, QPushButton)

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
    pm = QPixmap(w,h); pm.fill(QColor(60,60,60))
    p = QPainter(pm); p.setPen(Qt.NoPen)
    p.setBrush(QColor(70,120,200) if kind in ("video","image","text") else QColor(70,200,120))
    p.drawRect(0,0,w,h)
    p.setPen(QPen(QColor(255,255,255,220),2))
    if kind=="audio":
        p.drawEllipse(int(w*0.25), int(h*0.2), int(w*0.18), int(w*0.18))
        p.drawLine(int(w*0.32), int(h*0.2), int(w*0.7), int(h*0.15))
        p.drawLine(int(w*0.7), int(h*0.15), int(w*0.7), int(h*0.7))
    elif kind=="text":
        p.drawLine(int(w*0.2), int(h*0.25), int(w*0.8), int(h*0.25))
        p.drawLine(int(w*0.5), int(h*0.25), int(w*0.5), int(h*0.75))
    else:
        pts = [QPointF(w*0.35,h*0.25), QPointF(w*0.75,h*0.5), QPointF(w*0.35,h*0.75)]
        p.drawPolygon(QPolygonF(pts))
    p.end(); return pm

# ---------- media preview dialog ----------
class PreviewDialog(QDialog):
    def __init__(self, media: MediaItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle(media.path.name)
        self.setModal(False)
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
        self.setIconSize(QSize(120,68))
        self.setMaximumHeight(148)
        self.setSpacing(8)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setDefaultDropAction(Qt.CopyAction)

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
        self.px_per_s = 100; self.duration_ms = 60000; self.playhead_ms = 0
        self.setFixedHeight(28)
        self._markers: List[int] = []  # ms

    def set_scale(self, px): self.px_per_s = max(5.0, float(px)); self.update()
    def set_duration(self, ms): self.duration_ms = max(1000, int(ms)); self.update()
    def set_playhead(self, ms): self.playhead_ms = max(0,int(ms)); self.update()
    def set_markers(self, markers: List[int]): self._markers = list(sorted(set(int(m) for m in markers))); self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.fillRect(self.rect(), QColor(30,30,35))
        # playhead
        x = int((self.playhead_ms/1000.0) * self.px_per_s)
        p.setPen(QPen(QColor(230,80,80), 2)); p.drawLine(x, 0, x, self.rect().height())
        # ticks
        w = self.rect().width(); target = 80.0
        step = max(0.1, target / max(1.0, self.px_per_s))
        stops = [0.1,0.2,0.5,1,2,5,10,15,30,60]
        tick = next((s for s in stops if s >= step), 60)
        sub = tick/5.0; t = 0.0
        while True:
            x = int(t * self.px_per_s)
            if x > w: break
            major = abs((t/tick) - round(t/tick)) < 1e-6
            p.setPen(QPen(QColor(220,220,220) if major else QColor(140,140,150), 1))
            p.drawLine(x, 0, x, 12 if major else 8)
            if major: p.drawText(x+2, 22, f"{t:.2f}s" if tick<1 else f"{int(t)}s")
            t += sub
        # markers
        for m in self._markers:
            x = int((m/1000.0) * self.px_per_s)
            p.setBrush(QColor(255,220,80)); p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x-4, 14), QPointF(x+4, 14), QPointF(x, 4)]))
            p.setPen(QPen(QColor(255,220,80),1)); p.drawLine(x, 14, x, self.height())
        p.end()

    def mouseDoubleClickEvent(self, ev):
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        ms = int(max(0, (x / max(1.0, self.px_per_s))*1000.0))
        self.positionPicked.emit(ms)
        super().mouseDoubleClickEvent(ev)

    # zoom with wheel over ruler
    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        parent = self.parent()
        while parent and not isinstance(parent, EditorPane):
            parent = parent.parent()
        if parent and isinstance(parent, EditorPane):
            if dy > 0: parent._set_zoom(parent.zoom*1.1)
            else: parent._set_zoom(parent.zoom/1.1)
            ev.accept(); return
        super().wheelEvent(ev)

# ---------- timeline widgets ----------
class ClipWidget(QFrame):
    CLIP_MIME = "application/x-framevision-clip"
    HANDLE_W = 6

    def __init__(self, clip: Clip, media: MediaItem, row: "TrackRow"):
        super().__init__(row)
        self.clip = clip; self.media = media; self.row = row
        self.setFrameShape(QFrame.Panel); self.setFrameShadow(QFrame.Raised); self.setLineWidth(2)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.OpenHandCursor)
        self._press_pos = None; self._dragging = False
        self._resizing = False; self._resize_edge = None  # "L" or "R"
        self._orig_start = 0; self._orig_dur = 0
        self.setContextMenuPolicy(Qt.DefaultContextMenu)  # use contextMenuEvent for reliability

    def contextMenuEvent(self, ev):
        self.row.editor._select_clip(self.clip, additive=False, toggle=False)
        self._show_ctx(ev.globalPos())

    def _show_ctx(self, global_pos):
        ed = self.row.editor; m = QMenu(self)
        # Copy/Paste
        m.addAction("Copy", lambda: ed._clip_copy(self.row.track, self.clip))
        act_paste = m.addAction("Paste", lambda: ed._clip_paste(self.row.track, self.mapToParent(self.rect().center()).x(), self.media.kind))
        act_paste.setEnabled(ed._clipboard is not None)
        # Cut
        cut = m.addMenu("Cut")
        cut.addAction("Keep left side",  lambda: ed._clip_cut(self.row.track, self.clip, "left"))
        cut.addAction("Keep right side", lambda: ed._clip_cut(self.row.track, self.clip, "right"))
        cut.addAction("Keep both sides (split)", lambda: ed._clip_cut(self.row.track, self.clip, "both"))
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

    def paintEvent(self, ev):
        p = QPainter(self)
        # background tinted by track color
        base = QColor(50,90,160) if self.media.kind in ("video","image","text") else QColor(60,160,90)
        try:
            tc = QColor(self.row.track.color)
            bg = QColor((base.red()+tc.red())//2, (base.green()+tc.green())//2, (base.blue()+tc.blue())//2)
        except Exception:
            bg = base
        p.fillRect(self.rect(), bg)
        # selection border
        try:
            if self.row.editor._is_selected(self.clip):
                p.setPen(QPen(QColor(255,220,80), 3)); p.drawRect(self.rect().adjusted(1,1,-2,-2))
        except Exception: pass
        # trim handles
        p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,100))
        p.drawRect(0,0,self.HANDLE_W,self.height())
        p.drawRect(self.width()-self.HANDLE_W,0,self.HANDLE_W,self.height())
        # text
        p.setPen(QPen(QColor(255,255,255),1))
        name = self.media.path.name if self.media.kind!="text" else (self.clip.text or "Text")
        p.drawText(6, int(self.rect().height()/2)+5, name)
        p.end()

    def _edge_at_pos(self, posx: int) -> Optional[str]:
        if posx <= self.HANDLE_W: return "L"
        if posx >= self.width()-self.HANDLE_W: return "R"
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
            start_ms = self.row.editor._snap_ms(start_ms)
            if self._resize_edge == "L":
                # new start cannot exceed old end-1
                new_start = min(start_ms, self._orig_start + self._orig_dur - 1)
                delta = new_start - self._orig_start
                self.clip.start_ms = new_start
                self.clip.duration_ms = max(1, self._orig_dur - delta)
            else:
                # right edge -> duration changes
                end = max(start_ms, self._orig_start + 1)
                self.clip.duration_ms = max(1, end - self._orig_start)
            self.row.relayout(); self.row.editor._update_time_range(); self.row.update()
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
            if edge: self.setCursor(Qt.SizeHorCursor)
            else: self.setCursor(Qt.OpenHandCursor)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._resizing:
            self._resizing = False; self._resize_edge = None; self.setCursor(Qt.OpenHandCursor)
            self.row.editor._mark_dirty(); self.row.editor._push_undo()  # push after change
        self.setCursor(Qt.OpenHandCursor); self._press_pos = None; self._dragging = False
        super().mouseReleaseEvent(ev)

class TrackHeader(QWidget):
    def __init__(self, track: Track, editor: "EditorPane", parent=None):
        super().__init__(parent); self.track=track; self.editor=editor
        lay = QHBoxLayout(self); lay.setContentsMargins(4,0,4,0); lay.setSpacing(6)
        self.swatch = QFrame(); self.swatch.setFixedSize(14,14); self._apply_color()
        self.name = QLabel(self._label_text())
        self.btn_m = QToolButton(); self.btn_m.setText("M"); self.btn_m.setCheckable(True); self.btn_m.setToolTip("Mute track")
        self.btn_s = QToolButton(); self.btn_s.setText("S"); self.btn_s.setCheckable(True); self.btn_s.setToolTip("Solo track")
        self.btn_l = QToolButton(); self.btn_l.setText("L"); self.btn_l.setCheckable(True); self.btn_l.setToolTip("Lock track")
        self.btn_color = QToolButton(); self.btn_color.setText("🎨"); self.btn_color.setToolTip("Track color")
        for b in (self.btn_m,self.btn_s,self.btn_l,self.btn_color):
            b.setFixedHeight(20)
        lay.addWidget(self.swatch); lay.addWidget(self.name,1); lay.addWidget(self.btn_m); lay.addWidget(self.btn_s); lay.addWidget(self.btn_l); lay.addWidget(self.btn_color)
        self.btn_m.setChecked(track.mute); self.btn_s.setChecked(track.solo); self.btn_l.setChecked(track.locked)
        self.btn_m.toggled.connect(self._set_mute); self.btn_s.toggled.connect(self._set_solo); self.btn_l.toggled.connect(self._set_lock); self.btn_color.clicked.connect(self._pick_color)

    def _label_text(self):
        return self.track.name if self.track.type=='any' else f"{self.track.name} ({self.track.type})"

    def _apply_color(self):
        try:
            self.swatch.setStyleSheet(f"background:{self.track.color}; border:1px solid #666;")
        except Exception:
            self.swatch.setStyleSheet("background:#3c3c3c; border:1px solid #666;")

    def _set_mute(self, on): self.track.mute = bool(on); self.editor._mark_dirty()
    def _set_solo(self, on): self.track.solo = bool(on); self.editor._mark_dirty()
    def _set_lock(self, on): self.track.locked = bool(on); self.editor._mark_dirty()
    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self.track.color), self, "Track color")
        if col.isValid():
            self.track.color = col.name()
            self._apply_color(); self.editor._refresh_tracks()

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
            w.setGeometry(x, 4, max(8,width), 38); w.show(); self._clip_widgets.append(w)
        total_ms = EditorPane.compute_project_duration(self.project)
        self.setMinimumWidth(int((total_ms/1000.0)*self.px_per_s)+80)
        try: self.editor._update_holder_width()
        except Exception: pass

    def paintEvent(self, ev):
        # locked overlay
        if self.track.locked:
            p0 = QPainter(self); p0.fillRect(self.rect(), QColor(40,40,40,60)); p0.end()
        # droppable highlight
        if self._drag_hover and self._drag_can:
            pbg = QPainter(self); pbg.fillRect(self.rect(), QColor(40, 90, 40, 80)); pbg.end()
        super().paintEvent(ev)
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
            kind = self._kind_from_mid(mid); can = (self.track.type in ("any", kind))
        elif mime and mime.hasFormat(ClipWidget.CLIP_MIME):
            try:
                _,_, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":"); can = (self.track.type in ("any", kind))
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
            if self.track.type == "any":
                self.track.type = m.kind
                try: self.editor._update_track_labels()
                except Exception: pass
            if self.track.type != m.kind:
                QMessageBox.information(self, "Type mismatch", f"This track is locked to {self.track.type}."); return
            self.editor._push_undo()
            dur = 3000
            if m.meta.get("duration"):
                try: dur = int(float(m.meta["duration"]) * 1000)
                except: pass
            self.track.clips.append(Clip(media_id=mid, start_ms=start_ms, duration_ms=dur))
            self.editor._refresh_tracks(); self.editor._update_time_range(); self.editor._mark_dirty()
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
        self._last_open_dir: str = str(Path.cwd())
        self.glue_enabled = True
        self.grid_enabled = False
        self.grid_step_s = 0.5
        self.snap_px_tol = 8
        self.ripple_enabled = False
        self.markers: List[int] = []
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
        self.zoom_in=btn("+","Zoom In", lambda: self._set_zoom(self.zoom*1.25))
        self.zoom_out=btn("-","Zoom Out", lambda: self._set_zoom(self.zoom/1.25))
        self.btn_fit=btn("Fit","Zoom to fit (Ctrl+F)", self._zoom_to_fit)
        self.btn_sel=btn("Sel","Zoom to selection (Z)", self._zoom_to_selection)
        self.btn_glue=QToolButton(); self.btn_glue.setText("Glue"); self.btn_glue.setCheckable(True); self.btn_glue.setChecked(True); self.btn_glue.toggled.connect(lambda on: setattr(self,'glue_enabled',bool(on)))
        self.btn_grid=QToolButton(); self.btn_grid.setText("Grid"); self.btn_grid.setCheckable(True); self.btn_grid.toggled.connect(self._toggle_grid)
        self.grid_step = QComboBox(); [self.grid_step.addItem(s, float(s)) for s in ["0.25","0.5","1","2","5"]]; self.grid_step.setCurrentIndex(1); self.grid_step.currentIndexChanged.connect(lambda *_: setattr(self,'grid_step_s',float(self.grid_step.currentData())))
        self.snap_tol = QSpinBox(); self.snap_tol.setRange(1,32); self.snap_tol.setValue(self.snap_px_tol); self.snap_tol.valueChanged.connect(lambda v: setattr(self,'snap_px_tol',int(v)))
        self.btn_ripple=QToolButton(); self.btn_ripple.setText("Ripple"); self.btn_ripple.setCheckable(True); self.btn_ripple.toggled.connect(lambda on: setattr(self,'ripple_enabled',bool(on)))
        for w in [self.btn_new,self.btn_load,self.btn_save,self.btn_imp,self.btn_export,self.btn_undo,self.btn_redo,self.zoom_in,self.zoom_out,self.btn_fit,self.btn_sel,self.btn_glue,self.btn_grid,QLabel("Step"),self.grid_step,QLabel("Snap px"),self.snap_tol,self.btn_ripple]:
            tb.addWidget(w)
        tb.addStretch(1); root.addLayout(tb)
        # media strip
        self.media_list = MediaList(); root.addWidget(self.media_list)
        self.media_list.requestDelete.connect(self._delete_media_ids)
        self.media_list.requestPreview.connect(self._preview_media_id)
        # controls
        ctrl = QHBoxLayout()
        self.btn_add_track = QToolButton(); self.btn_add_track.setText("Add timeline"); self.btn_add_track.clicked.connect(self._add_track)
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(10,500); self.zoom_slider.setValue(100); self.zoom_slider.valueChanged.connect(lambda v: self._set_zoom(v/100.0))
        self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.setRange(0,60000); self.seek_slider.sliderMoved.connect(self._seek_changed)
        for w in [self.btn_add_track, QLabel("Zoom"), self.zoom_slider, QLabel("Seek"), self.seek_slider]:
            ctrl.addWidget(w)
        ctrl.addStretch(1); root.addLayout(ctrl)
        # ruler + tracks
        self.ruler = TimeRuler(); root.addWidget(self.ruler); self.ruler.positionPicked.connect(self._set_playhead_from_ruler)
        self.tracks_area = QScrollArea(); self.tracks_area.setWidgetResizable(True); self.tracks_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        holder = QWidget(); self.tracks_holder = holder
        self.tracks_layout = QVBoxLayout(holder); self.tracks_layout.setContentsMargins(0,0,0,0); self.tracks_layout.setSpacing(6)
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tracks_area.setWidget(holder); root.addWidget(self.tracks_area, 1)
        self._add_shortcuts()

    def _add_shortcuts(self):
        def sc(seq, cb): a=QAction(self); a.setShortcut(QKeySequence(seq)); a.triggered.connect(cb); self.addAction(a)
        sc("Ctrl+N", self._new_project); sc("Ctrl+O", self._load_project); sc("Ctrl+S", self._save_project)
        sc("Ctrl+Z", self._undo); sc("Ctrl+Y", self._redo); sc("+", lambda: self._set_zoom(self.zoom*1.25)); sc("-", lambda: self._set_zoom(self.zoom/1.25))
        sc("Space", lambda: self.transport_request.emit("toggle"))
        sc("Ctrl+F", self._zoom_to_fit); sc("Z", self._zoom_to_selection)
        sc("M", self._add_marker_here); sc(".", self._goto_next_marker); sc(",", self._goto_prev_marker)
        sc("Ctrl+G", self._group_selection); sc("Ctrl+Shift+G", self._ungroup_selection)

    # autosave
    def _setup_autosave(self):
        self._autosave_path = Path.cwd()/"presets"/"setsave"/"editor_temp.json"
        self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
        self._dirty=False; self._last_change_ts=0.0; self._last_save_ts=0.0
        self._autosave_timer = QTimer(self); self._autosave_timer.setInterval(60_000); self._autosave_timer.timeout.connect(self._autosave_tick); self._autosave_timer.start()

    # project ops
    def _new_project(self):
        self.project = Project(); self.project.tracks=[Track("Timeline 1","any"), Track("Timeline 2","any"), Track("Timeline 3","any")]
        self._undo_stack.clear(); self._redo_stack.clear(); self._refresh_ui(); self._mark_dirty()

    def _load_project(self):
        fn,_=QFileDialog.getOpenFileName(self,"Load Project",str(Path.cwd()/ "output"),"FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            self.project = Project.from_json(json.loads(Path(fn).read_text(encoding="utf-8"))); self._undo_stack.clear(); self._redo_stack.clear(); self._refresh_ui(); self._mark_dirty()
        except Exception as e: QMessageBox.critical(self,"Error",f"Failed to load: {e}")

    def _save_project(self):
        fn,_=QFileDialog.getSaveFileName(self,"Save Project",str(Path.cwd()/ "output"/"project.fvedit"),"FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            Path(fn).parent.mkdir(parents=True, exist_ok=True); Path(fn).write_text(json.dumps(self.project.to_json(), indent=2), encoding="utf-8")
            QMessageBox.information(self,"Saved",f"Saved: {fn}")
        except Exception as e: QMessageBox.critical(self,"Error",f"Failed to save: {e}")

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
            out = subprocess.check_output([ffprobe_path(),"-v","error","-select_streams","v:0","-show_entries","stream=width,height,avg_frame_rate","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)], stderr=subprocess.STDOUT, universal_newlines=True)
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

    def _refresh_media_list(self, new_items=None):
        self.media_list.clear()
        items=[]
        for m in self.project.media.values():
            li=QListWidgetItem(m.path.name); li.setData(Qt.UserRole, m.id); li.setIcon(QIcon(_placeholder_icon(m.kind, self.media_list.iconSize()))); self.media_list.addItem(li); items.append((m.id,m))
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
        # handled by TrackHeader now
        for i in range(self.tracks_layout.count()):
            w=self.tracks_layout.itemAt(i).widget()
            if isinstance(w, TrackHeader):
                w.name.setText(w._label_text())

    def _refresh_tracks(self):
        while self.tracks_layout.count():
            w=self.tracks_layout.takeAt(0).widget()
            if w: w.setParent(None); w.deleteLater()
        for i,tr in enumerate(self.project.tracks):
            header=TrackHeader(tr, self); self.tracks_layout.addWidget(header)
            row=TrackRow(tr, self.project, self); row.set_scale(self._px_per_s()); self.tracks_layout.addWidget(row)
            if i < len(self.project.tracks)-1:
                div = QFrame(); div.setFrameShape(QFrame.HLine); div.setFrameShadow(QFrame.Sunken); div.setStyleSheet("color:#3c3c3c;"); self.tracks_layout.addWidget(div)
        self.tracks_layout.addStretch(1)

    def _update_time_range(self):
        total = self.compute_project_duration(self.project); self.ruler.set_duration(total); self.ruler.set_markers(self.markers); self.seek_slider.setRange(0,int(total)); self._update_holder_width()

    def _set_zoom(self, z):
        self.zoom = max(0.1, min(20.0, float(z))); pxs=self._px_per_s(); self.ruler.set_scale(pxs)
        for i in range(self.tracks_layout.count()):
            w=self.tracks_layout.itemAt(i).widget()
            if isinstance(w, TrackRow): w.set_scale(pxs)
        self._update_holder_width(); self.zoom_slider.blockSignals(True); self.zoom_slider.setValue(int(self.zoom*100)); self.zoom_slider.blockSignals(False)

    def _px_per_s(self) -> float: return 100.0 * self.zoom

    def _update_holder_width(self):
        total_ms = self.compute_project_duration(self.project); need = int((total_ms/1000.0)*self._px_per_s())+120
        if getattr(self,'tracks_holder',None) is not None: self.tracks_holder.setMinimumWidth(need)

    def _seek_changed(self, pos_ms: int):
        self.preview_media.emit(None, int(pos_ms)); self.transport_request.emit(f"seek:{int(pos_ms)}")
        try:
            self.ruler.set_playhead(int(pos_ms))
            for i in range(self.tracks_layout.count()):
                w=self.tracks_layout.itemAt(i).widget()
                if isinstance(w, TrackRow): w.update()
        except Exception: pass

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
        if tgt is None: tgt=Track("Audio","audio"); self.project.tracks.append(tgt)
        tgt.clips.append(Clip(media_id=clip.media_id, start_ms=clip.start_ms, duration_ms=clip.duration_ms))
        self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

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
    def _export(self):
        dlg = ExportDialog(self, self.project.width, self.project.height)
        if dlg.exec()!=QDialog.Accepted: return
        opts=dlg.values(); self.project.width=opts["width"]; self.project.height=opts["height"]; self.project.fps=opts["fps"]
        out=Path(opts["output"]); out.parent.mkdir(parents=True, exist_ok=True)
        QMessageBox.information(self,"Export","(Stub) Export graph wiring will go here.")

    # snapping
    def _toggle_grid(self, on): self.grid_enabled = bool(on); self._update_time_range()

    def _snap_ms(self, ms: int) -> int:
        if not self.glue_enabled and not self.grid_enabled: return int(ms)
        px_tol=float(self.snap_px_tol); tol=int(round((px_tol / max(1e-3, self._px_per_s())) * 1000.0))
        cands=[0, int(self.seek_slider.value())]
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

    # ---- keyboard control ----
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
