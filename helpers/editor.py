# helpers/editor.py
# Mini Video Editor Tab for FrameVision (from uploaded base, with compact media strip and clip DnD)
# - Host preview remains external; we emit preview_media / transport_request like before.
# - Media "selection box" is now a compact strip above the timelines (IconMode, wrapping, max ~140px).
# - Timelines are below the media strip.
# - You can move clips along a track and drag to other (similar) tracks.
# - Export dialog and ffmpeg pipeline are kept.
from __future__ import annotations

import os, sys, json, subprocess, shutil, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import (Qt, QRectF, QPointF, QSize, QMimeData, QTimer, QEvent, Signal, QObject,
                            QAbstractTableModel, QModelIndex, QItemSelectionModel, QThread)
from PySide6.QtGui import (QAction, QKeySequence, QIcon, QPixmap, QDrag, QPainter, QPen, QColor, QBrush,
                           QStandardItemModel, QStandardItem, QCursor, QImage, QImageReader)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QMenu, QToolButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QScrollArea, QFrame, QMessageBox, QSlider, QSizePolicy, QDialog, QDialogButtonBox, QFormLayout,
    QAbstractItemView, QApplication, QTabWidget, QInputDialog
)

# --------- Utility: locate ffmpeg / ffprobe ----------
def _try_exec(cand: Path | str) -> Optional[str]:
    try:
        subprocess.check_output([str(cand), "-version"], stderr=subprocess.STDOUT)
        return str(cand)
    except Exception:
        return None

def ffmpeg_path() -> str:
    root = Path(".").resolve()
    cands = [root/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg"), "ffmpeg"]
    for c in cands:
        p = _try_exec(c)
        if p: return p
    return "ffmpeg"

def ffprobe_path() -> str:
    root = Path(".").resolve()
    cands = [root/"bin"/("ffprobe.exe" if os.name=="nt" else "ffprobe"), "ffprobe"]
    for c in cands:
        p = _try_exec(c)
        if p: return p
    return "ffprobe"

VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".webm",".m4v",".mpg",".mpeg",".wmv",".ts",".m2ts"}
AUDIO_EXTS = {".mp3",".wav",".aac",".m4a",".flac",".ogg",".opus",".aiff"}
IMAGE_EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".gif"}

def media_kind(path: Path) -> str:
    s = path.suffix.lower()
    if s in VIDEO_EXTS: return "video"
    if s in AUDIO_EXTS: return "audio"
    if s in IMAGE_EXTS: return "image"
    if s in {".txt",".srt",".ass",".vtt"}: return "text"
    return "unknown"

# --------- Data model ---------
@dataclass
class MediaItem:
    id: str
    path: Path
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)  # width/height/duration/fps/size

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

    def to_json(self) -> Dict[str, Any]:
        m = {k: {"id": v.id, "path": str(v.path), "kind": v.kind, "meta": v.meta} for k,v in self.media.items()}
        t = []
        for tr in self.tracks:
            t.append({"name": tr.name, "type": tr.type, "enabled": tr.enabled, "clips": [asdict(c) for c in tr.clips]})
        return {"version": self.version, "media": m, "tracks": t, "fps": self.fps, "width": self.width, "height": self.height, "sample_rate": self.sample_rate}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Project":
        p = Project()
        p.version = int(d.get("version", 1))
        p.fps = int(d.get("fps", 30))
        p.width = int(d.get("width", 1280))
        p.height = int(d.get("height", 720))
        p.sample_rate = int(d.get("sample_rate", 48000))
        for k, md in d.get("media", {}).items():
            p.media[k] = MediaItem(id=md["id"], path=Path(md["path"]), kind=md.get("kind","unknown"), meta=md.get("meta",{}))
        p.tracks = []
        for tr in d.get("tracks", []):
            t = Track(name=tr.get("name","Track"), type=tr.get("type","video"), enabled=tr.get("enabled",True))
            for c in tr.get("clips", []):
                t.clips.append(Clip(**c))
            p.tracks.append(t)
        return p

# ---------- Media strip (compact) ----------
class MediaListWidget(QListWidget):
    MIME = "application/x-framevision-media-id"
    def __init__(self, parent=None):
        super().__init__(parent)
        # Turn it into a compact horizontal strip
        self.setViewMode(QListWidget.IconMode)
        self.setFlow(QListWidget.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setIconSize(QSize(96, 54))
        self.setMaximumHeight(140)
        self.setSpacing(8)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx)

    def _ctx(self, pos):
        menu = QMenu(self)
        menu.addAction("Load media…", lambda: self.parent().parent()._import_media() if hasattr(self.parent().parent(), "_import_media") else None)
        it = self.itemAt(pos)
        if it and it.flags() & Qt.ItemIsEnabled:
            mid = it.data(Qt.UserRole)
            if mid:
                menu.addSeparator()
                menu.addAction("Remove from project", lambda: self._remove_mid(mid))
        menu.exec(self.mapToGlobal(pos))

    def _remove_mid(self, mid: str):
        ed = self._find_editor()
        if not ed: return
        # Remove from timelines and media list
        for tr in ed.project.tracks:
            tr.clips = [c for c in tr.clips if c.media_id != mid]
        ed.project.media.pop(mid, None)
        ed._refresh_ui()

    def _find_editor(self):
        p = self.parent()
        while p and not isinstance(p, EditorPane):
            p = p.parent()
        return p

    def startDrag(self, supportedActions):
        it = self.currentItem()
        if not it: return
        mid = it.data(Qt.UserRole)
        if not mid: return
        mime = QMimeData()
        mime.setData(self.MIME, str(mid).encode("utf-8"))
        drag = QDrag(self); drag.setMimeData(mime)
        # basic ghost
        pm = QPixmap(self.iconSize()); pm.fill(QColor(70,70,70))
        drag.setPixmap(pm)
        drag.exec(Qt.CopyAction)

    def set_thumb(self, media_id: str, pm: QPixmap):
        # Set the icon for the list item with this media id
        for i in range(self.count()):
            it = self.item(i)
            if it and it.data(Qt.UserRole) == media_id:
                it.setIcon(QIcon(pm))
                break

# ---------- Ruler ----------
class TimeRuler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.px_per_s = 100
        self.duration_ms = 60000
        self.setFixedHeight(24)

    def set_scale(self, px_per_s: float): self.px_per_s = max(5, min(4000, float(px_per_s))); self.update()
    def set_duration(self, ms: int): self.duration_ms = max(1000, int(ms)); self.update()
    def sizeHint(self): return self.minimumSizeHint()
    def minimumSizeHint(self): return QSize(int(self.px_per_s * (self.duration_ms/1000.0)) + 80, 24)

    def paintEvent(self, ev):
        p = QPainter(self); p.fillRect(self.rect(), QColor(30,30,35))
        total_s = self.duration_ms/1000.0; w = self.rect().width()
        # dynamic ticks ~80px apart
        target_px = 80.0
        step = max(0.1, target_px / max(1.0, self.px_per_s))
        stops = [0.1,0.2,0.5,1,2,5,10,15,30,60]
        tick = stops[-1]
        for s in stops:
            if s >= step: tick = s; break
        sub = tick/5.0
        t = 0.0
        while True:
            x = int(t * self.px_per_s)
            if x > w: break
            major = abs((t / tick) - round(t / tick)) < 1e-6
            p.setPen(QPen(QColor(220,220,220) if major else QColor(140,140,150), 1))
            p.drawLine(x, 0, x, 12 if major else 8)
            if major:
                p.drawText(x+2, 20, f"{t:.2f}s" if tick < 1 else f"{int(t)}s")
            t += sub
        p.end()

# ---------- Clip widget (move + cross-lane drag) ----------
class ClipWidget(QFrame):
    CLIP_MIME = "application/x-framevision-clip"
    def __init__(self, clip: Clip, media: MediaItem, track_row: "TrackRow"):
        super().__init__(track_row)
        self.clip = clip
        self.media = media
        self.track_row = track_row
        self.setFrameShape(QFrame.Panel); self.setFrameShadow(QFrame.Raised); self.setLineWidth(2)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(f"{media.kind.upper()}: {media.path.name}")
        self._press_pos = None
        self._dragging = False

    self.setContextMenuPolicy(Qt.CustomContextMenu)
    self.customContextMenuRequested.connect(self._on_ctx)

def _on_ctx(self, pos):
    # Build context menu with greyed-out options where not applicable
    m = QMenu(self)
    ed = self.track_row.editor
    media_kind = self.media.kind
    # Copy/Paste
    act_copy = m.addAction("Copy")
    act_paste = m.addAction("Paste")
    act_copy.triggered.connect(lambda: ed._clip_copy(self.track_row.track, self.clip))
    act_paste.setEnabled(ed._clipboard is not None)
    # Record the x position for paste placement
    x = self.mapToParent(pos).x()
    act_paste.triggered.connect(lambda: ed._clip_paste(self.track_row.track, x, self.media.kind))
    # Cut submenu
    cut_menu = m.addMenu("Cut")
    act_keep_left = cut_menu.addAction("Keep left side")
    act_keep_right = cut_menu.addAction("Keep right side")
    act_keep_both = cut_menu.addAction("Keep both sides (split)")
    act_keep_left.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="left"))
    act_keep_right.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="right"))
    act_keep_both.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="both"))
    # Fade submenu
    fade_menu = m.addMenu("Fade")
    fade_in_menu = fade_menu.addMenu("Start")
    fade_out_menu = fade_menu.addMenu("End")
    for label, ms in [("Short (1s)",1000),("Medium (2s)",2000),("Long (3s)",3000)]:
        fade_in_menu.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, start_ms=ms, end_ms=None))
        fade_out_menu.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, start_ms=None, end_ms=ms))
    # Volume submenu (audio + video only)
    vol_menu = m.addMenu("Volume"); 
    act_mute_on = vol_menu.addAction("Mute ON"); act_mute_off = vol_menu.addAction("Mute OFF")
    act_set_gain = vol_menu.addAction("Set level (dB)…")
    act_mute_on.triggered.connect(lambda: ed._clip_set_mute(self.clip, True))
    act_mute_off.triggered.connect(lambda: ed._clip_set_mute(self.clip, False))
    act_set_gain.triggered.connect(lambda: ed._clip_set_gain(self.clip))
    vol_menu.setEnabled(media_kind in ("audio","video"))
    # Time submenu (speed) for audio+video
    time_menu = m.addMenu("Time")
    faster = time_menu.addMenu("Faster")
    slower = time_menu.addMenu("Slower")
    for label, mult in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: faster.addAction(label, lambda mult=mult: ed._clip_speed(self.clip, mult))
    for label, div in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: slower.addAction(label, lambda div=div: ed._clip_speed(self.clip, 1.0/div))
    time_menu.setEnabled(media_kind in ("audio","video"))
    # Rotate submenu (images+video)
    rot_menu = m.addMenu("Rotate")
    for d in [45,90,180]:
        rot_menu.addAction(f"+{d}°", lambda d=d: ed._clip_rotate(self.clip, +d))
        rot_menu.addAction(f"-{d}°", lambda d=d: ed._clip_rotate(self.clip, -d))
    rot_menu.setEnabled(media_kind in ("image","video","text"))
    # Separate sound (video only)
    act_sep = m.addAction("Separate sound from video")
    act_sep.setEnabled(media_kind == "video")
    act_sep.triggered.connect(lambda: ed._clip_separate_audio(self.track_row.track, self.clip))
    # Edit text (text only)
    act_edit_text = m.addAction("Edit text…")
    act_edit_text.setEnabled(media_kind == "text")
    act_edit_text.triggered.connect(lambda: ed._clip_edit_text(self.clip))
    m.exec(self.mapToGlobal(pos))

    def paintEvent(self, ev):
        p = QPainter(self)
        bg = QColor(50,90,160) if self.media.kind in ("video","image","text") else QColor(60,160,90)
        p.fillRect(self.rect(), bg)
        p.setPen(QPen(QColor(255,255,255), 1))
        name = self.media.path.name if self.media.kind != "text" else (self.clip.text or "Text")
        p.drawText(6, int(self.rect().height()/2)+5, name)
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._press_pos = ev.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos and (ev.buttons() & Qt.LeftButton):
            if not self._dragging and (ev.position().toPoint() - self._press_pos).manhattanLength() > 6:
                # Start a DnD carrying track index + clip index + kind
                src_ti = self.track_row.editor._track_index(self.track_row.track)
                src_ci = self.track_row.index_of(self.clip)
                if src_ti is None or src_ci is None: return
                mime = QMimeData()
                mime.setData(self.CLIP_MIME, f"{src_ti}:{src_ci}:{self.media.kind}".encode("utf-8"))
                drag = QDrag(self); drag.setMimeData(mime)
                pm = QPixmap(self.width(), self.height()); pm.fill(QColor(0,0,0,0))
                qp = QPainter(pm); qp.fillRect(pm.rect(), QColor(255,255,255,60)); qp.end()
                drag.setPixmap(pm)
                self._dragging = True
                drag.exec(Qt.MoveAction)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.setCursor(Qt.OpenHandCursor)
        self._press_pos = None
        self._dragging = False

    self.setContextMenuPolicy(Qt.CustomContextMenu)
    self.customContextMenuRequested.connect(self._on_ctx)

def _on_ctx(self, pos):
    # Build context menu with greyed-out options where not applicable
    m = QMenu(self)
    ed = self.track_row.editor
    media_kind = self.media.kind
    # Copy/Paste
    act_copy = m.addAction("Copy")
    act_paste = m.addAction("Paste")
    act_copy.triggered.connect(lambda: ed._clip_copy(self.track_row.track, self.clip))
    act_paste.setEnabled(ed._clipboard is not None)
    # Record the x position for paste placement
    x = self.mapToParent(pos).x()
    act_paste.triggered.connect(lambda: ed._clip_paste(self.track_row.track, x, self.media.kind))
    # Cut submenu
    cut_menu = m.addMenu("Cut")
    act_keep_left = cut_menu.addAction("Keep left side")
    act_keep_right = cut_menu.addAction("Keep right side")
    act_keep_both = cut_menu.addAction("Keep both sides (split)")
    act_keep_left.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="left"))
    act_keep_right.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="right"))
    act_keep_both.triggered.connect(lambda: ed._clip_cut(self.track_row.track, self.clip, mode="both"))
    # Fade submenu
    fade_menu = m.addMenu("Fade")
    fade_in_menu = fade_menu.addMenu("Start")
    fade_out_menu = fade_menu.addMenu("End")
    for label, ms in [("Short (1s)",1000),("Medium (2s)",2000),("Long (3s)",3000)]:
        fade_in_menu.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, start_ms=ms, end_ms=None))
        fade_out_menu.addAction(label, lambda ms=ms: ed._clip_set_fade(self.clip, start_ms=None, end_ms=ms))
    # Volume submenu (audio + video only)
    vol_menu = m.addMenu("Volume"); 
    act_mute_on = vol_menu.addAction("Mute ON"); act_mute_off = vol_menu.addAction("Mute OFF")
    act_set_gain = vol_menu.addAction("Set level (dB)…")
    act_mute_on.triggered.connect(lambda: ed._clip_set_mute(self.clip, True))
    act_mute_off.triggered.connect(lambda: ed._clip_set_mute(self.clip, False))
    act_set_gain.triggered.connect(lambda: ed._clip_set_gain(self.clip))
    vol_menu.setEnabled(media_kind in ("audio","video"))
    # Time submenu (speed) for audio+video
    time_menu = m.addMenu("Time")
    faster = time_menu.addMenu("Faster")
    slower = time_menu.addMenu("Slower")
    for label, mult in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: faster.addAction(label, lambda mult=mult: ed._clip_speed(self.clip, mult))
    for label, div in [("+25%",1.25),("+50%",1.5),("+100%",2.0)]: slower.addAction(label, lambda div=div: ed._clip_speed(self.clip, 1.0/div))
    time_menu.setEnabled(media_kind in ("audio","video"))
    # Rotate submenu (images+video)
    rot_menu = m.addMenu("Rotate")
    for d in [45,90,180]:
        rot_menu.addAction(f"+{d}°", lambda d=d: ed._clip_rotate(self.clip, +d))
        rot_menu.addAction(f"-{d}°", lambda d=d: ed._clip_rotate(self.clip, -d))
    rot_menu.setEnabled(media_kind in ("image","video","text"))
    # Separate sound (video only)
    act_sep = m.addAction("Separate sound from video")
    act_sep.setEnabled(media_kind == "video")
    act_sep.triggered.connect(lambda: ed._clip_separate_audio(self.track_row.track, self.clip))
    # Edit text (text only)
    act_edit_text = m.addAction("Edit text…")
    act_edit_text.setEnabled(media_kind == "text")
    act_edit_text.triggered.connect(lambda: ed._clip_edit_text(self.clip))
    m.exec(self.mapToGlobal(pos))
        super().mouseReleaseEvent(ev)

# ---------- Track row ----------
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

    def index_of(self, clip: Clip) -> Optional[int]:
        try: return self.track.clips.index(clip)
        except Exception: return None

    def set_scale(self, px_per_s: float):
        self.px_per_s = float(px_per_s)
        self.relayout()

    def relayout(self):
        for w in self._clip_widgets:
            w.setParent(None); w.deleteLater()
        self._clip_widgets = []
        for c in self.track.clips:
            m = self.project.media.get(c.media_id)
            if not m: continue
            w = ClipWidget(c, m, self)
            x = int((c.start_ms/1000.0)*self.px_per_s); width = int((max(1,c.duration_ms)/1000.0)*self.px_per_s)
            w.setGeometry(x, 4, max(6,width), 38); w.show()
            self._clip_widgets.append(w)
        total_ms = EditorPane.compute_project_duration(self.project)
        self.setMinimumWidth(int((total_ms/1000.0)*self.px_per_s)+80)

    def _kind_from_mid(self, mid: str) -> Optional[str]:
        m = self.project.media.get(mid)
        return m.kind if m else None

    def dragEnterEvent(self, ev):
        mime = ev.mimeData()
        ok = False
        if mime.hasFormat(MediaListWidget.MIME):
            mid = bytes(mime.data(MediaListWidget.MIME)).decode("utf-8")
            kind = self._kind_from_mid(mid)
            ok = (self.track.type in ("any", kind))
        elif mime.hasFormat(ClipWidget.CLIP_MIME):
            try:
                src_ti, src_ci, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":")
                ok = (self.track.type in ("any", kind))
            except Exception:
                ok = False
        if ok: ev.acceptProposedAction()
        else: ev.ignore()

    def dragMoveEvent(self, ev): self.dragEnterEvent(ev)

    def dropEvent(self, ev):
        x = ev.position().x() if hasattr(ev,'position') else ev.pos().x()
        start_ms = int(max(0, (x / max(1.0, self.px_per_s)) * 1000.0))
        try:
            mime = ev.mimeData()
            if mime.hasFormat(MediaListWidget.MIME):
                mid = bytes(mime.data(MediaListWidget.MIME)).decode("utf-8")
                m = self.project.media.get(mid)
                if not m: return
                if self.track.type == "any":
                    self.track.type = m.kind
                    try: self.editor._update_track_labels()
                    except Exception: pass
                if self.track.type != m.kind:
                    QMessageBox.information(self, "Type mismatch", f"This track is locked to {self.track.type}.")
                    return
                self.editor._push_undo()
                dur = 3000
                if m.meta.get("duration"):
                    try: dur = int(float(m.meta["duration"]) * 1000)
                    except: pass
                self.track.clips.append(Clip(media_id=mid, start_ms=start_ms, duration_ms=dur))
                self.relayout(); self.editor._update_time_range(); self.editor._mark_dirty()
                ev.acceptProposedAction(); return
            if mime.hasFormat(ClipWidget.CLIP_MIME):
                src_ti, src_ci, kind = bytes(mime.data(ClipWidget.CLIP_MIME)).decode("utf-8").split(":")
                src_ti = int(src_ti); src_ci = int(src_ci)
                if self.track.type not in ("any", kind):
                    QMessageBox.information(self, "Type mismatch", f"This track is locked to {self.track.type}.")
                    return
                self.editor._push_undo()
                src_tr = self.editor.project.tracks[src_ti]
                if src_ci < 0 or src_ci >= len(src_tr.clips): return
                clip = src_tr.clips.pop(src_ci)
                # if target is 'any', lock to kind of clip media
                if self.track.type == "any":
                    self.track.type = kind
                    try: self.editor._update_track_labels()
                    except Exception: pass
                clip.start_ms = start_ms
                self.track.clips.append(clip)
                # Refresh both source and target rows
                self.editor._refresh_tracks()
                self.editor._update_time_range(); self.editor._mark_dirty()
                ev.acceptProposedAction(); return
        except Exception:
            pass
        ev.ignore()

    def wheelEvent(self, ev):
        try:
            delta = ev.angleDelta().y()
            factor = 1.1 if delta > 0 else (1/1.1)
            if self.editor is not None:
                self.editor._set_zoom(self.editor.zoom * factor)
            else:
                self.set_scale(self.px_per_s * (1.1 if delta>0 else 0.9))
        except Exception:
            pass
        super().wheelEvent(ev)

# ---------- Export dialog ----------
class ExportDialog(QDialog):
    def __init__(self, parent=None, default_w=1280, default_h=720):
        super().__init__(parent)
        self.setWindowTitle("Export Video"); self.setModal(True)
        lay = QFormLayout(self)
        self.combo_res = QComboBox()
        self.res_list = [("320x320",320,320),("480x480",480,480),("640x360",640,360),("854x480",854,480),
                         ("1280x720",1280,720),("1920x1080",1920,1080),("2560x1440",2560,1440),("3840x2160",3840,2160)]
        idx = 4
        for i,(label,w,h) in enumerate(self.res_list):
            self.combo_res.addItem(label, (w,h))
            if w==default_w and h==default_h: idx = i
        self.combo_res.setCurrentIndex(idx)
        self.fps = QSpinBox(); self.fps.setRange(10, 120); self.fps.setValue(30)
        self.bitrate = QLineEdit("6M")
        self.out = QLineEdit("export.mp4")
        btn = QToolButton(); btn.setText("…")
        def _pick():
            fn, _ = QFileDialog.getSaveFileName(self, "Save As", str(Path.cwd()/"output"/"video"/"export.mp4"), "MP4 (*.mp4)")
            if fn: self.out.setText(fn)
        btn.clicked.connect(_pick)
        h = QHBoxLayout(); h.addWidget(self.out); h.addWidget(btn)
        lay.addRow("Resolution:", self.combo_res)
        lay.addRow("FPS:", self.fps)
        lay.addRow("Video Bitrate (e.g., 6M):", self.bitrate)
        lay.addRow("Output:", h)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        lay.addRow(self.buttons)

    def values(self):
        w,h = self.combo_res.currentData()
        return {"width": w, "height": h, "fps": int(self.fps.value()), "bitrate": self.bitrate.text().strip() or "6M",
                "output": self.out.text().strip() or "export.mp4"}


# ---------- Thumbnail generation ----------
class ThumbThread(QThread):
    ready = Signal(str, QPixmap)  # media_id, pixmap
    def __init__(self, items, icon_size, parent=None):
        super().__init__(parent)
        self.items = list(items)  # list of (media_id, MediaItem)
        self.icon_size = icon_size

    def run(self):
        for mid, m in self.items:
            try:
                pm = _generate_thumbnail(m, self.icon_size)
                if pm is not None:
                    self.ready.emit(mid, pm)
            except Exception:
                pass

def _generate_thumbnail(media: MediaItem, icon_size: QSize) -> QPixmap | None:
    w, h = icon_size.width(), icon_size.height()
    bg = QColor(60,60,60)
    def _canvas():
        pix = QPixmap(w, h); pix.fill(bg); return pix
    kind = media.kind
    # Audio/Text: placeholders
    if kind in ("audio", "text"):
        pm = _placeholder_icon(kind, icon_size)
        return pm
    if kind == "image" and media.path.exists():
        try:
            reader = QImageReader(str(media.path))
            img = reader.read()
            if not img.isNull():
                img = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                canvas = _canvas()
                p = QPainter(canvas)
                x = (w - img.width())//2; y = (h - img.height())//2
                p.drawImage(x, y, img)
                p.end()
                return canvas
        except Exception:
            pass
    if kind == "video" and media.path.exists():
        # Use ffmpeg to capture a frame around 1s
        try:
            import subprocess
            args = [ffmpeg_path(), "-ss", "1", "-i", str(media.path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
            data = subprocess.check_output(args, stderr=subprocess.DEVNULL)
            img = QImage.fromData(data, "PNG")
            if not img.isNull():
                img = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                canvas = _canvas()
                p = QPainter(canvas); x = (w - img.width())//2; y = (h - img.height())//2; p.drawImage(x, y, img); p.end()
                return canvas
        except Exception:
            pass
    # Fallback
    return _placeholder_icon(kind, icon_size)

def _placeholder_icon(kind: str, size: QSize) -> QPixmap:
    w, h = size.width(), size.height()
    pix = QPixmap(w, h); pix.fill(QColor(60,60,60))
    p = QPainter(pix); p.setPen(Qt.NoPen)
    color = QColor(70,120,200) if kind in ("video","image","text") else QColor(70,200,120)
    p.setBrush(color); p.drawRect(0,0,w,h)
    # simple symbol
    p.setPen(QPen(QColor(255,255,255,220), 2))
    if kind == "audio":
        # draw a music note
        p.drawEllipse(int(w*0.25), int(h*0.2), int(w*0.18), int(w*0.18))
        p.drawLine(int(w*0.32), int(h*0.2), int(w*0.7), int(h*0.15))
        p.drawLine(int(w*0.7), int(h*0.15), int(w*0.7), int(h*0.7))
    elif kind == "text":
        # draw a 'T'
        p.drawLine(int(w*0.2), int(h*0.25), int(w*0.8), int(h*0.25))
        p.drawLine(int(w*0.5), int(h*0.25), int(w*0.5), int(h*0.75))
    else:
        # generic play triangle
        pts = [QPointF(w*0.35, h*0.25), QPointF(w*0.75, h*0.5), QPointF(w*0.35, h*0.75)]
        p.drawPolygon(*pts)
    p.end()
    return pix

# ---------- Editor pane ----------
class EditorPane(QWidget):
    # Keep the autoplace-first behavior from the uploaded file
    def _find_parent_tabwidget(self):
        p = self.parent()
        while p is not None and not isinstance(p, QTabWidget):
            p = p.parent()
        return p

    def _find_tabwidget_anywhere(self):
        tw = self._find_parent_tabwidget()
        if tw: return tw
        try:
            win = self.window()
            if win is not None:
                tw = win.findChild(QTabWidget)
                if tw and tw.indexOf(self) != -1: return tw
        except Exception: pass
        try:
            app = QApplication.instance()
            if app:
                for w in app.allWidgets():
                    if isinstance(w, QTabWidget):
                        try:
                            if w.indexOf(self) != -1: return w
                        except Exception: pass
        except Exception: pass
        return None

    def _autoplace_try(self):
        tabs = self._find_tabwidget_anywhere()
        if not tabs: return False
        try:
            container = self; idx = -1
            while container is not None:
                try:
                    idx = tabs.indexOf(container)
                    if idx != -1: break
                except Exception: pass
                container = container.parent()
            if idx == -1: return False
            try: tabs.setTabText(idx, 'Editor')
            except Exception: pass
            if idx != 0:
                try: tabs.tabBar().moveTab(idx, 0)
                except Exception:
                    try:
                        w = tabs.widget(idx); text = tabs.tabText(idx) or 'Editor'
                        tabs.removeTab(idx); tabs.insertTab(0, w, text)
                    except Exception: return False
            try: tabs.setCurrentIndex(0)
            except Exception: pass
            return True
        except Exception: return False

    def _start_autoplace_timer(self):
        try:
            self._autoplace_attempts = 0
            if getattr(self, '_autoplace_timer', None):
                try: self._autoplace_timer.stop()
                except Exception: pass
            self._autoplace_timer = QTimer(self); self._autoplace_timer.setInterval(200)
            def _tick():
                ok = self._autoplace_try()
                self._autoplace_attempts += 1
                if ok or self._autoplace_attempts > 30:
                    try: self._autoplace_timer.stop()
                    except Exception: pass
            self._autoplace_timer.timeout.connect(_tick); self._autoplace_timer.start(); QTimer.singleShot(0, _tick)
        except Exception: pass

    def event(self, e):
        try:
            if e and getattr(e, 'type', lambda: None)() in (QEvent.Show, QEvent.ParentChange, QEvent.ShowToParent):
                self._start_autoplace_timer()
        except Exception: pass
        return super().event(e)

    preview_media = Signal(object, int)     # Path | (text string), position_ms
    transport_request = Signal(str)         # "play","pause","seek:<ms>"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MiniEditorPane")
        self.project = Project()
        self.zoom = 1.0          # 1.0 -> px_per_s=100
        self.max_tracks = 8
        self._undo_stack: List[Dict[str,Any]] = []
        self._redo_stack: List[Dict[str,Any]] = []
        self._init_ui()
        self._new_project()
        try: self._start_autoplace_timer()
        except Exception: pass

# --- Auto-save setup ---
self._autosave_path = Path.cwd() / "presets" / "setsave" / "editor_temp.json"
self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
self._dirty = False
self._last_change_ts = 0.0
self._last_save_ts = 0.0
self._autosave_timer = QTimer(self)
self._autosave_timer.setInterval(60_000)  # 60 seconds
self._autosave_timer.timeout.connect(self._autosave_tick)
self._autosave_timer.start()

# Try to restore previous session automatically
try:
    if self._autosave_path.exists():
        data = json.loads(self._autosave_path.read_text(encoding="utf-8"))
        self.project = Project.from_json(data)
        self._refresh_ui()
except Exception:
    pass

    # -------- Layout: toolbar -> media strip -> controls -> ruler -> tracks ----
    def _init_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(8,8,8,8)

        # Toolbar
        tb = QHBoxLayout()
        def _btn(text, tip, cb):
            b = QToolButton(); b.setText(text); b.setToolTip(tip); b.clicked.connect(cb); return b
        self.btn_new = _btn("New", "New project (Ctrl+N)", self._new_project)
        self.btn_load= _btn("Load", "Load project (Ctrl+O)", self._load_project)
        self.btn_save= _btn("Save", "Save project (Ctrl+S)", self._save_project)
        self.btn_imp = _btn("Load media", "Import video/image/audio/text", self._import_media)
        self.btn_export = _btn("Export", "Export MP4 (H.264)", self._export)
        self.btn_undo = _btn("Undo", "Undo (Ctrl+Z)", self._undo)
        self.btn_redo = _btn("Redo", "Redo (Ctrl+Y)", self._redo)
        self.zoom_in = _btn("+", "Zoom In", lambda: self._set_zoom(self.zoom*1.25))
        self.zoom_out= _btn("-", "Zoom Out", lambda: self._set_zoom(self.zoom/1.25))
        # Transport buttons (wired later by host):
        self.btn_play = _btn("▶", "Play", lambda: self.transport_request.emit("play"))
        self.btn_pause = _btn("⏸", "Pause", lambda: self.transport_request.emit("pause"))
        self.btn_stop = _btn("⏹", "Stop", lambda: self.transport_request.emit("stop"))
        self.btn_rew = _btn("⏪", "Back 5s", lambda: self._jump_ms(-5000))
        self.btn_ff  = _btn("⏩", "Fwd 5s", lambda: self._jump_ms(+5000))
        for w in [self.btn_new,self.btn_load,self.btn_save,self.btn_imp,self.btn_export,
                  self.btn_undo,self.btn_redo,self.btn_rew,self.btn_play,self.btn_pause,self.btn_stop,self.btn_ff,
                  self.zoom_in,self.zoom_out]:
            tb.addWidget(w)
        tb.addStretch(1)
        root.addLayout(tb)

        # Media strip (compact)
        self.media_list = MediaListWidget()
        root.addWidget(self.media_list)

        # Controls row: add track, zoom slider, seek slider
        ctrl = QHBoxLayout()
        self.btn_add_track = QToolButton(); self.btn_add_track.setText("Add timeline"); self.btn_add_track.clicked.connect(self._add_track)
        self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.setRange(0, 60000); self.seek_slider.sliderMoved.connect(self._seek_changed)
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(10, 500); self.zoom_slider.setValue(100); self.zoom_slider.valueChanged.connect(lambda v: self._set_zoom(v/100.0))
        for w in [self.btn_add_track, QLabel("Zoom"), self.zoom_slider, QLabel("Seek"), self.seek_slider]:
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        # Ruler + tracks (vertical)
        self.ruler = TimeRuler(); root.addWidget(self.ruler)
        self.tracks_area = QScrollArea(); self.tracks_area.setWidgetResizable(True); self.tracks_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget(); self.tracks_layout = QVBoxLayout(holder); self.tracks_layout.setContentsMargins(0,0,0,0); self.tracks_layout.setSpacing(6)
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tracks_area.setWidget(holder); root.addWidget(self.tracks_area, 1)

        self._add_shortcuts()

    def _add_shortcuts(self):
        def sc(seq, cb):
            a = QAction(self); a.setShortcut(QKeySequence(seq)); a.triggered.connect(cb); self.addAction(a)
        sc("Ctrl+N", self._new_project); sc("Ctrl+O", self._load_project); sc("Ctrl+S", self._save_project)
        sc("Ctrl+Z", self._undo); sc("Ctrl+Y", self._redo)
        sc("+", lambda: self._set_zoom(self.zoom*1.25)); sc("-", lambda: self._set_zoom(self.zoom/1.25))
        # Transport shortcuts
        sc("Space", lambda: self.transport_request.emit("toggle"))
        sc("K", lambda: self.transport_request.emit("pause"))
        sc("L", lambda: self.btn_ff.click())
        sc("J", lambda: self.btn_rew.click())
        sc("Ctrl+Right", lambda: self._jump_ms(+5000))
        sc("Ctrl+Left",  lambda: self._jump_ms(-5000))

    # -------------------- Project ops --------------------
    def _new_project(self):
        self.project = Project()
        self.project.tracks = [Track("Timeline 1","any"), Track("Timeline 2","any"), Track("Timeline 3","any"), Track("Timeline 4","any")]
        self._undo_stack.clear(); self._redo_stack.clear()
        self._refresh_ui()
        self._mark_dirty()

    def _load_project(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Project", str(Path.cwd()/ "output"), "FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            data = json.loads(Path(fn).read_text(encoding="utf-8"))
            self.project = Project.from_json(data)
            self._undo_stack.clear(); self._redo_stack.clear()
            self._refresh_ui()
            self._mark_dirty()
            QMessageBox.information(self, "Loaded", f"Loaded project: {Path(fn).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _save_project(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Project", str(Path.cwd()/ "output"/"project.fvedit"), "FrameVision Edit (*.fvedit)")
        if not fn: return
        try:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            Path(fn).write_text(json.dumps(self.project.to_json(), indent=2), encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Saved: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    # -------------------- Media --------------------
    def _import_media(self):
        fns, _ = QFileDialog.getOpenFileNames(self, "Import Media", str(Path.cwd()), "Media Files (*.*)")
        if not fns: return
        for fn in fns:
            p = Path(fn); kind = media_kind(p)
            mid = self._unique_media_id(p)
            self.project.media[mid] = MediaItem(id=mid, path=p, kind=kind, meta=self._probe(p))
        self._refresh_media_list()
        self._mark_dirty()

    def _unique_media_id(self, p: Path) -> str:
        base = p.stem.lower().replace(" ","_"); i = 1; mid = base
        while mid in self.project.media:
            i += 1; mid = f"{base}_{i}"
        return mid

    def _probe(self, path: Path) -> Dict[str,Any]:
        meta = {"duration": None, "width": None, "height": None, "fps": None}
        try:
            out = subprocess.check_output([ffprobe_path(), "-v", "error",
                                           "-select_streams", "v:0",
                                           "-show_entries", "stream=width,height,avg_frame_rate",
                                           "-show_entries", "format=duration",
                                           "-of", "default=noprint_wrappers=1:nokey=1",
                                           str(path)], stderr=subprocess.STDOUT, universal_newlines=True)
            lines = [x.strip() for x in out.splitlines() if x.strip()]
            for x in lines:
                if x.isdigit() and meta["width"] is None: meta["width"] = int(x); continue
                if x.isdigit() and meta["height"] is None: meta["height"] = int(x); continue
                if "/" in x and meta["fps"] is None:
                    n,d = x.split("/"); 
                    try:
                        if float(d)!=0: meta["fps"] = round(float(n)/float(d),2)
                    except Exception: pass
                if "." in x and meta["duration"] is None:
                    try: meta["duration"] = float(x)
                    except Exception: pass
        except Exception: pass
        return meta

    def _refresh_media_list(self):
        self.media_list.clear()
        items = []
        for m in self.project.media.values():
            li = QListWidgetItem(m.path.name)
            li.setData(Qt.UserRole, m.id)
            # placeholder; real thumbnail set asynchronously
            placeholder = _placeholder_icon(m.kind, self.media_list.iconSize())
            li.setIcon(QIcon(placeholder))
            self.media_list.addItem(li)
            items.append((m.id, m))
        # spawn thumbnail thread
        if items:
            try:
                self._thumb_thread = ThumbThread(items, self.media_list.iconSize(), self)
                self._thumb_thread.ready.connect(lambda mid, pm: self.media_list.set_thumb(mid, pm))
                self._thumb_thread.start()
            except Exception:
                pass

    # -------------------- Tracks & timeline --------------------
    def _add_track(self):
        if len(self.project.tracks) >= self.max_tracks:
            QMessageBox.information(self, "Limit", f"Maximum {self.max_tracks} tracks.")
            return
        menu = QMenu(self)
        def add(kind, name):
            self.project.tracks.append(Track(name, kind)); self._refresh_tracks(); self._mark_dirty()
        menu.addAction("Video", lambda: add("video","Video"))
        menu.addAction("Image", lambda: add("image","Image"))
        menu.addAction("Text",  lambda: add("text","Text"))
        menu.addAction("Audio", lambda: add("audio","Audio"))
        menu.exec(QCursor.pos())

    def _refresh_ui(self):
        self._refresh_media_list()
        self._refresh_tracks()
        self._update_time_range()

    def _update_track_labels(self):
        # Update the "(type)" suffix on labels
        i = 0
        for idx in range(self.tracks_layout.count()):
            w = self.tracks_layout.itemAt(idx).widget()
            if isinstance(w, QLabel):
                # label precedes a TrackRow
                if i < len(self.project.tracks):
                    tr = self.project.tracks[i]
                    w.setText(tr.name if tr.type=="any" else f"{tr.name} ({tr.type})")
                    i += 1

    def _refresh_tracks(self):
        while self.tracks_layout.count():
            w = self.tracks_layout.takeAt(0).widget()
            if w: w.setParent(None); w.deleteLater()
        for tr in self.project.tracks:
            row = TrackRow(tr, self.project, editor=self)
            row.set_scale(self._px_per_s())
            label = tr.name if tr.type == 'any' else f"{tr.name} ({tr.type})"
            self.tracks_layout.addWidget(QLabel(label))
            self.tracks_layout.addWidget(row)
        self.tracks_layout.addStretch(1)

    def _update_time_range(self):
        total = self.compute_project_duration(self.project)
        self.ruler.set_duration(total)
        self.seek_slider.setRange(0, int(total))

    def _set_zoom(self, z: float):
        self.zoom = max(0.1, min(20.0, z))
        pxs = self._px_per_s()
        self.ruler.set_scale(pxs)
        for i in range(self.tracks_layout.count()):
            w = self.tracks_layout.itemAt(i).widget()
            if isinstance(w, TrackRow):
                w.set_scale(pxs)

    def _px_per_s(self) -> float: return 100.0 * self.zoom

def _mark_dirty(self):
    import time
    self._dirty = True
    self._last_change_ts = time.time()

def _autosave_tick(self):
    import time
    now = time.time()
    # Save every 60s if there were changes since last save
    should_save = self._dirty and (now - self._last_save_ts >= 60)
    # Also, if no movement for 120s, ensure one last save.
    if self._dirty and (now - self._last_change_ts >= 120):
        should_save = True
    if should_save:
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            self._autosave_path.write_text(json.dumps(self.project.to_json(), indent=2), encoding="utf-8")
            self._last_save_ts = now
            # If idle for 120s, clear dirty so we don't keep saving
            if now - self._last_change_ts >= 120:
                self._dirty = False
        except Exception:
            pass

    def _seek_changed(self, pos_ms: int):
        self.preview_media.emit(None, int(pos_ms))
        self.transport_request.emit(f"seek:{int(pos_ms)}")

# ------- Context menu handlers -------
_clipboard: Optional[dict] = None

def _clip_copy(self, track: Track, clip: Clip):
    self._clipboard = {"media_id": clip.media_id, "duration_ms": clip.duration_ms, "gain_db": clip.gain_db,
                       "text": clip.text, "speed": clip.speed, "rotation_deg": clip.rotation_deg,
                       "muted": clip.muted, "fade_in_ms": clip.fade_in_ms, "fade_out_ms": clip.fade_out_ms}
    self._mark_dirty()

def _clip_paste(self, track: Track, x_in_row: int, kind_hint: str):
    if not self._clipboard: return
    # compute start at mouse x
    start_ms = int(max(0, (x_in_row / max(1.0, self._px_per_s())) * 1000.0))
    # if track is 'any', lock to hint
    if track.type == "any": track.type = kind_hint
    # guard incompatible types
    if track.type not in ("any", kind_hint): return
    c = Clip(media_id=self._clipboard["media_id"],
             start_ms=start_ms,
             duration_ms=int(self._clipboard["duration_ms"]),
             gain_db=float(self._clipboard["gain_db"]),
             text=self._clipboard["text"],
             speed=float(self._clipboard["speed"]),
             rotation_deg=int(self._clipboard["rotation_deg"]),
             muted=bool(self._clipboard["muted"]),
             fade_in_ms=int(self._clipboard["fade_in_ms"]),
             fade_out_ms=int(self._clipboard["fade_out_ms"]))
    track.clips.append(c)
    self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

def _clip_cut(self, track: Track, clip: Clip, mode: str):
    # Cut at current playhead
    cut_ms = int(self.seek_slider.value())
    c_start = clip.start_ms; c_end = clip.start_ms + clip.duration_ms
    if not (c_start < cut_ms < c_end): return
    self._push_undo()
    if mode == "left":
        clip.duration_ms = max(1, cut_ms - c_start)
    elif mode == "right":
        clip.duration_ms = max(1, c_end - cut_ms)
        clip.start_ms = cut_ms
    else:  # both: split
        left = Clip(media_id=clip.media_id, start_ms=c_start, duration_ms=cut_ms - c_start,
                    gain_db=clip.gain_db, text=clip.text, speed=clip.speed, rotation_deg=clip.rotation_deg,
                    muted=clip.muted, fade_in_ms=clip.fade_in_ms, fade_out_ms=0)
        right = Clip(media_id=clip.media_id, start_ms=cut_ms, duration_ms=c_end - cut_ms,
                     gain_db=clip.gain_db, text=clip.text, speed=clip.speed, rotation_deg=clip.rotation_deg,
                     muted=clip.muted, fade_in_ms=0, fade_out_ms=clip.fade_out_ms)
        try:
            idx = track.clips.index(clip)
            track.clips.pop(idx)
            track.clips.insert(idx, left)
            track.clips.insert(idx+1, right)
        except Exception:
            pass
    self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

def _clip_set_fade(self, clip: Clip, start_ms: Optional[int]=None, end_ms: Optional[int]=None):
    if start_ms is not None: clip.fade_in_ms = int(start_ms)
    if end_ms is not None: clip.fade_out_ms = int(end_ms)
    self._refresh_tracks(); self._mark_dirty()

def _clip_set_mute(self, clip: Clip, mute: bool):
    clip.muted = bool(mute); self._mark_dirty()

def _clip_set_gain(self, clip: Clip):
    try:
        val, ok = QInputDialog.getDouble(self, "Volume (dB)", "Gain:", clip.gain_db, -60.0, 12.0, 1)
        if ok:
            clip.gain_db = float(val); self._mark_dirty()
    except Exception:
        pass

def _clip_speed(self, clip: Clip, factor: float):
    # limit speed to reasonable range
    factor = max(0.1, min(8.0, float(factor)))
    # speed up reduces duration; slow down increases
    new_speed = clip.speed * factor
    # recompute duration to maintain same source span assumption
    if factor != 0:
        clip.duration_ms = int(max(1, round(clip.duration_ms / factor)))
    clip.speed = new_speed
    self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

def _clip_rotate(self, clip: Clip, delta_deg: int):
    clip.rotation_deg = int((clip.rotation_deg + delta_deg) % 360)
    self._mark_dirty()

def _clip_separate_audio(self, track: Track, clip: Clip):
    # Find or create an audio track
    tgt = None
    for tr in self.project.tracks:
        if tr.type == "audio":
            tgt = tr; break
    if tgt is None:
        tgt = Track("Audio", "audio"); self.project.tracks.append(tgt)
    # Add an audio clip referencing same media
    tgt.clips.append(Clip(media_id=clip.media_id, start_ms=clip.start_ms, duration_ms=clip.duration_ms))
    self._refresh_tracks(); self._update_time_range(); self._mark_dirty()

def _clip_edit_text(self, clip: Clip):
    try:
        val, ok = QInputDialog.getText(self, "Edit text", "Text:", text=clip.text or "")
        if ok:
            clip.text = val or None; self._mark_dirty()
    except Exception:
        pass
    def _jump_ms(self, delta: int):
        # Relative seek helper used by Rew/FF buttons; emits absolute seek for compatibility.
        try:
            cur = int(self.seek_slider.value())
            newv = max(0, min(int(self.seek_slider.maximum()), cur + int(delta)))
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(newv)
            self.seek_slider.blockSignals(False)
            self.preview_media.emit(None, int(newv))
            self.transport_request.emit(f"seek:{int(newv)}")
        except Exception:
            pass


    # -------------------- Undo/Redo --------------------
    def _snapshot(self) -> Dict[str,Any]: return self.project.to_json()
    def _push_undo(self):
        self._undo_stack.append(self._snapshot()); self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack: return
        self._redo_stack.append(self._snapshot())
        js = self._undo_stack.pop()
        self.project = Project.from_json(js); self._refresh_ui(); self._mark_dirty()

    def _redo(self):
        if not self._redo_stack: return
        self._undo_stack.append(self._snapshot())
        js = self._redo_stack.pop()
        self.project = Project.from_json(js); self._refresh_ui(); self._mark_dirty()

    # -------------------- Helpers --------------------
    def _track_index(self, track: Track) -> Optional[int]:
        try: return self.project.tracks.index(track)
        except ValueError: return None

    @staticmethod
    def compute_project_duration(p: Project) -> int:
        end = 0
        for tr in p.tracks:
            for c in tr.clips:
                end = max(end, c.start_ms + max(1, c.duration_ms))
        return max(1000, end)

    # -------------------- Export --------------------
    def _export(self):
        dlg = ExportDialog(self, self.project.width, self.project.height)
        if dlg.exec()!=QDialog.Accepted: return
        opts = dlg.values()
        self.project.width = opts["width"]; self.project.height = opts["height"]; self.project.fps = opts["fps"]
        out = Path(opts["output"]); out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._export_ffmpeg(out, opts["bitrate"])
            QMessageBox.information(self, "Export", f"Export started:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _export_ffmpeg(self, out_path: Path, vbitrate: str = "6M"):
        p = self.project; ff = ffmpeg_path()
        args = [ff, "-y"]; filter_parts = []; v_streams = []; a_streams = []; input_index = 0
        def add_video_solid(label, color="black", dur_ms=1000):
            d = max(0.001, dur_ms/1000.0); idx = len(filter_parts)
            filter_parts.append(f"color=c={color}:s={p.width}x{p.height}:r={p.fps}:d={d}[{label}]")
        vc_labels_by_track: List[List[str]] = []
        for ti, tr in enumerate(p.tracks):
            if not tr.enabled: vc_labels_by_track.append([]); continue
            if tr.type in ("video","image","text","any"):
                lab_list = []
                for ci, c in enumerate(tr.clips):
                    # infer kind from media
                    kind = "video"
                    if c.text is not None: kind = "text"
                    else:
                        m = p.media.get(c.media_id)
                        if m: kind = m.kind
                    if kind=="text":
                        text = (c.text or "Text").replace(":", r'\:')
                        d = max(0.001, c.duration_ms/1000.0)
                        filter_parts.append(
                            f"color=black@0.0:s={p.width}x{p.height}:r={p.fps}:d={d},"
                            f"drawtext=text='{text}':x=(w-text_w)/2:y=(h-text_h)/2:fontcolor=white:fontsize=48:box=1:boxcolor=0x00000088"
                            f"[txt{ti}_{ci}]"
                        )
                        start_s = max(0.0, c.start_ms/1000.0); filter_parts.append(f"[txt{ti}_{ci}]tpad=start_duration={start_s}[txtp{ti}_{ci}]")
                        lab_list.append(f"txtp{ti}_{ci}")
                    elif kind=="image":
                        m = p.media.get(c.media_id); 
                        if not m: continue
                        args += ["-loop","1","-i", str(m.path)]
                        d = max(0.001, c.duration_ms/1000.0)
                        filter_parts.append(f"[{input_index}:v]fps={p.fps},scale={p.width}:{p.height},format=yuv420p,setsar=1,trim=duration={d}[img{ti}_{ci}]")
                        input_index += 1
                        start_s = max(0.0, c.start_ms/1000.0); filter_parts.append(f"[img{ti}_{ci}]tpad=start_duration={start_s}[imgp{ti}_{ci}]")
                        lab_list.append(f"imgp{ti}_{ci}")
                    else:  # video
                        m = p.media.get(c.media_id); 
                        if not m: continue
                        args += ["-i", str(m.path)]
                        d = max(0.001, c.duration_ms/1000.0)
                        filter_parts.append(f"[{input_index}:v]fps={p.fps},scale={p.width}:{p.height},format=yuv420p,setsar=1,trim=duration={d},setpts=PTS-STARTPTS[v{ti}_{ci}]")
                        input_index += 1
                        start_s = max(0.0, c.start_ms/1000.0); filter_parts.append(f"[v{ti}_{ci}]tpad=start_duration={start_s}[vp{ti}_{ci}]")
                        lab_list.append(f"vp{ti}_{ci}")
                vc_labels_by_track.append(lab_list)
            elif tr.type=="audio":
                for ci, c in enumerate(tr.clips):
                    m = p.media.get(c.media_id); 
                    if not m: continue
                    if m.kind in ("audio","video"):
                        args += ["-i", str(m.path)]
                        d = max(0.001, c.duration_ms/1000.0); start_s = max(0.0, c.start_ms/1000.0)
                        filter_parts.append(f"[{input_index}:a]atrim=duration={d},asetpts=PTS-STARTPTS,adelay={int(start_s*1000)}|{int(start_s*1000)}[a{ti}_{ci}]")
                        input_index += 1; a_streams.append(f"a{ti}_{ci}")
                vc_labels_by_track.append([])
            else:
                vc_labels_by_track.append([])

        track_vids = []
        for ti, labs in enumerate(vc_labels_by_track):
            add_video_solid(f"base{ti}", "black", dur_ms=self.compute_project_duration(p))
            cur = f"base{ti}"
            for li, lab in enumerate(labs):
                outlab = f"t{ti}_ov{li}"; filter_parts.append(f"[{cur}][{lab}]overlay=shortest=1[{outlab}]"); cur = outlab
            track_vids.append(cur)

        if track_vids:
            cur = track_vids[0]
            for i in range(1, len(track_vids)):
                outlab = f"vmerge{i}"; filter_parts.append(f"[{cur}][{track_vids[i]}]overlay=shortest=1[{outlab}]"); cur = outlab
            v_out = cur; v_streams.append(v_out)

        a_out = None
        if a_streams:
            if len(a_streams)==1: a_out = a_streams[0]
            else:
                n = len(a_streams)
                filter_parts.append(f"{''.join(f'[{a}]' for a in a_streams)}amix=inputs={n}:normalize=0[aout]")
                a_out = "aout"

        if filter_parts: args += ["-filter_complex", ";".join(filter_parts)]
        if v_streams:
            args += ["-map", f"[{v_streams[0]}]", "-c:v","libx264","-preset","medium","-b:v", vbitrate, "-pix_fmt", "yuv420p", "-r", str(p.fps)]
        if a_out: args += ["-map", f"[{a_out}]", "-c:a","aac","-b:a","192k"]
        else: args += ["-an"]
        args += [str(out_path)]
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Standalone quick test
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = EditorPane(); w.resize(1200, 700); w.show()
    sys.exit(app.exec())
