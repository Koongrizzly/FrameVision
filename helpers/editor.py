
# helpers/editor.py — stable build with selectable clips, robust DnD, right-click menus
from __future__ import annotations

import os, sys, json, subprocess, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import Qt, QRectF, QPointF, QSize, QMimeData, QTimer, QEvent, Signal, QThread
from PySide6.QtGui import (QAction, QKeySequence, QIcon, QPixmap, QDrag, QPainter, QPen, QColor, QBrush,
                           QStandardItemModel, QStandardItem, QCursor, QImage, QImageReader, QPolygonF)
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QListWidget,
                               QListWidgetItem, QMenu, QToolButton, QComboBox, QSpinBox, QLineEdit,
                               QScrollArea, QFrame, QMessageBox, QSlider, QSizePolicy, QDialog,
                               QDialogButtonBox, QFormLayout, QAbstractItemView, QApplication, QTabWidget, QInputDialog)

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

@dataclass
class Track:
    name: str
    type: str  # "video","image","text","audio","any"
    enabled: bool = True
    clips: List[Clip] = field(default_factory=list)

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
            for c in tr.get("clips", []):
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

# ---------- UI: media strip ----------
class MediaList(QListWidget):
    MIME = "application/x-framevision-media-id"
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

# ---------- ruler ----------
class TimeRuler(QWidget):
    positionPicked = Signal(int)  # ms
    def __init__(self, parent=None):
        super().__init__(parent)
        self.px_per_s = 100; self.duration_ms = 60000; self.playhead_ms = 0
        self.setFixedHeight(24)

    def set_scale(self, px_s): self.px_per_s = max(5.0, float(px_s)); self.update()
    def set_duration(self, ms): self.duration_ms = max(1000, int(ms)); self.update()
    def set_playhead(self, ms): self.playhead_ms = max(0,int(ms)); self.update()

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
            if major: p.drawText(x+2, 20, f"{t:.2f}s" if tick<1 else f"{int(t)}s")
            t += sub
        p.end()

    def mouseDoubleClickEvent(self, ev):
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        ms = int(max(0, (x / max(1.0, self.px_per_s))*1000.0))
        self.positionPicked.emit(ms)
        super().mouseDoubleClickEvent(ev)

# ---------- timeline widgets ----------
class ClipWidget(QFrame):
    CLIP_MIME = "application/x-framevision-clip"
    def __init__(self, clip: Clip, media: MediaItem, row: "TrackRow"):
        super().__init__(row)
        self.clip = clip; self.media = media; self.row = row
        self.setFrameShape(QFrame.Panel); self.setFrameShadow(QFrame.Raised); self.setLineWidth(2)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.OpenHandCursor)
        self._press_pos = None; self._dragging = False
        self.setContextMenuPolicy(Qt.DefaultContextMenu)  # use contextMenuEvent for reliability

    def contextMenuEvent(self, ev):
        self.row.editor._select_clip(self.clip)
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
        bg = QColor(50,90,160) if self.media.kind in ("video","image","text") else QColor(60,160,90)
        p.fillRect(self.rect(), bg)
        # selection border
        try:
            if self.row.editor._selected is self.clip:
                p.setPen(QPen(QColor(255,220,80), 3)); p.drawRect(self.rect().adjusted(1,1,-2,-2))
        except Exception: pass
        p.setPen(QPen(QColor(255,255,255),1))
        name = self.media.path.name if self.media.kind!="text" else (self.clip.text or "Text")
        p.drawText(6, int(self.rect().height()/2)+5, name)
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.row.editor._select_clip(self.clip)
            self._press_pos = ev.position().toPoint(); self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos and (ev.buttons() & Qt.LeftButton):
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
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.setCursor(Qt.OpenHandCursor); self._press_pos = None; self._dragging = False
        super().mouseReleaseEvent(ev)

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
        # droppable highlight
        if self._drag_hover and self._drag_can:
            pbg = QPainter(self); pbg.fillRect(self.rect(), QColor(40, 90, 40, 80)); pbg.end()
        super().paintEvent(ev)
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
        mime = ev.mimeData(); can = False
        if mime and mime.hasFormat(MediaListWidget.MIME if False else "application/x-framevision-media-id"):
            mid = bytes(mime.data("application/x-framevision-media-id")).decode("utf-8")
            kind = self._kind_from_mid(mid); can = (self.track.type in ("any", kind))
        elif mime and mime.hasFormat(ClipWidget.CLIP_MIME):
            try:
                _,_, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":"); can = (self.track.type in ("any", kind))
            except Exception: can = False
        self._drag_hover = True; self._drag_can = bool(can); self.update()
        if can: ev.acceptProposedAction()
        else: ev.ignore()

    def dragMoveEvent(self, ev):
        y = ev.position().y() if hasattr(ev,'position') else ev.pos().y()
        h = max(1,self.height()); band_ok = (h*0.10) <= y <= (h*0.90)
        self._drag_can = bool(band_ok) if self._drag_can else False
        self.update(); ev.acceptProposedAction()

    def dragLeaveEvent(self, ev):
        self._drag_hover = False; self._drag_can = False; self.update(); ev.accept()

    def dropEvent(self, ev):
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        start_ms = int(max(0, (x / max(1.0, self.px_per_s)) * 1000.0))
        try:
            start_ms = self.editor._snap_ms(start_ms)
        except Exception: pass
        mime = ev.mimeData()
        if mime.hasFormat("application/x-framevision-media-id"):
            mid = bytes(mime.data("application/x-framevision-media-id")).decode("utf-8")
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
            self._drag_hover = False; self._drag_can = False; self.update()
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
                self._drag_hover = False; self._drag_can = False; self.update()
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

# ---------- export dialog (kept) ----------
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
        self._last_open_dir: str = str(Path.cwd())
        self.glue_enabled = True
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
        self.btn_glue=QToolButton(); self.btn_glue.setText("Glue"); self.btn_glue.setCheckable(True); self.btn_glue.setChecked(True)
        self.btn_glue.toggled.connect(lambda on: setattr(self,'glue_enabled',bool(on)))
        for w in [self.btn_new,self.btn_load,self.btn_save,self.btn_imp,self.btn_export,self.btn_undo,self.btn_redo,self.zoom_in,self.zoom_out,self.btn_glue]:
            tb.addWidget(w)
        tb.addStretch(1); root.addLayout(tb)
        # media strip
        self.media_list = MediaList(); root.addWidget(self.media_list)
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

    # tracks/timeline rendering
    def _add_timeline_direct(self, kind, name): self.project.tracks.append(Track(name, kind)); self._refresh_tracks(); self._mark_dirty()

    def _add_track(self):
        m=QMenu(self); m.addAction("Video", lambda: self._add_timeline_direct("video","Video")); m.addAction("Image", lambda: self._add_timeline_direct("image","Image")); m.addAction("Text", lambda: self._add_timeline_direct("text","Text")); m.addAction("Audio", lambda: self._add_timeline_direct("audio","Audio")); m.exec(QCursor.pos())

    def _refresh_ui(self):
        self._refresh_media_list(); self._refresh_tracks(); self._update_time_range()

    def _update_track_labels(self):
        i=0
        for idx in range(self.tracks_layout.count()):
            w=self.tracks_layout.itemAt(idx).widget()
            if isinstance(w, QLabel):
                if i < len(self.project.tracks):
                    tr=self.project.tracks[i]; w.setText(tr.name if tr.type=="any" else f"{tr.name} ({tr.type})"); i+=1

    def _refresh_tracks(self):
        while self.tracks_layout.count():
            w=self.tracks_layout.takeAt(0).widget()
            if w: w.setParent(None); w.deleteLater()
        for i,tr in enumerate(self.project.tracks):
            row=TrackRow(tr, self.project, self); row.set_scale(self._px_per_s())
            label = tr.name if tr.type=='any' else f"{tr.name} ({tr.type})"
            self.tracks_layout.addWidget(QLabel(label)); self.tracks_layout.addWidget(row)
            if i < len(self.project.tracks)-1:
                div = QFrame(); div.setFrameShape(QFrame.HLine); div.setFrameShadow(QFrame.Sunken); div.setStyleSheet("color:#3c3c3c;"); self.tracks_layout.addWidget(div)
        self.tracks_layout.addStretch(1)

    def _update_time_range(self):
        total = self.compute_project_duration(self.project); self.ruler.set_duration(total); self.seek_slider.setRange(0,int(total)); self._update_holder_width()

    def _set_zoom(self, z):
        self.zoom = max(0.1, min(20.0, float(z))); pxs=self._px_per_s(); self.ruler.set_scale(pxs)
        for i in range(self.tracks_layout.count()):
            w=self.tracks_layout.itemAt(i).widget()
            if isinstance(w, TrackRow): w.set_scale(pxs)
        self._update_holder_width()

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

    # clipboard ops / ctx handlers
    def _select_clip(self, clip: Optional[Clip]):
        self._selected = clip
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

    # export (stub from earlier build)
    def _export(self):
        dlg = ExportDialog(self, self.project.width, self.project.height)
        if dlg.exec()!=QDialog.Accepted: return
        opts=dlg.values(); self.project.width=opts["width"]; self.project.height=opts["height"]; self.project.fps=opts["fps"]
        out=Path(opts["output"]); out.parent.mkdir(parents=True, exist_ok=True)
        QMessageBox.information(self,"Export","(Stub) Export graph wiring will go here.")

    # snapping
    def _snap_ms(self, ms: int) -> int:
        if not self.glue_enabled: return int(ms)
        px_tol=8.0; tol=int(round((px_tol / max(1e-3, self._px_per_s())) * 1000.0))
        cands=[0, int(self.seek_slider.value())]
        for tr in self.project.tracks:
            for c in tr.clips:
                cands.extend([int(c.start_ms), int(c.start_ms + max(1,c.duration_ms))])
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

# Standalone test
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = EditorPane(); w.resize(1200, 700); w.show()
    sys.exit(app.exec())
