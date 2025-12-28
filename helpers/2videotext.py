# helpers/videotext.py
# Video Text Overlay tool (PySide6) - Multi-Track Creator

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from PySide6.QtCore import (
    Qt, QRect, QSize, QUrl, Signal, QTimer, QSizeF, QRectF, QPointF
)
from PySide6.QtGui import (
    QPainter, QPen, QFont, QColor, QBrush
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFontComboBox, QFileDialog, QSlider, QScrollArea,
    QFrame, QColorDialog, QToolButton, QSizePolicy, QApplication,
    QCheckBox, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsTextItem,
    QListWidget, QListWidgetItem, QSplitter
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem


def _project_root_from_helpers_file() -> Path:
    p = Path(__file__).resolve()
    return p.parents[1]


def _settings_path() -> Path:
    return _project_root_from_helpers_file() / "presets" / "setsave" / "videotext.json"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ms_to_s(ms: int) -> float:
    return max(0.0, float(ms) / 1000.0)


def _rgba_to_ffmpeg_color(rgba: list[int]) -> str:
    r, g, b, a = rgba
    alpha = _clamp(a / 255.0, 0.0, 1.0)
    return f"#{r:02x}{g:02x}{b:02x}@{alpha:.3f}"


def _ff_escape_value(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "\\'")
         .replace("%", "\\%")
         .replace("[", "\\[")
         .replace("]", "\\]")
         .replace("=", "\\=")
    )


def _split_nonword(s: str) -> List[str]:
    out: List[str] = []
    cur = ""
    for ch in s:
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.append(cur)
                cur = ""
    if cur:
        out.append(cur)
    return out


def _guess_windows_fontfile(family: str) -> Optional[str]:
    try:
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if not fonts_dir.exists():
            return None
        tokens = [t for t in _split_nonword(family.lower()) if t]
        if not tokens:
            return None

        exts = (".ttf", ".otf", ".ttc")
        candidates = []
        for p in fonts_dir.iterdir():
            if p.suffix.lower() not in exts:
                continue
            name = p.name.lower()
            score = 0
            for t in tokens:
                if t in name:
                    score += 1
            if score > 0:
                candidates.append((score, len(name), str(p)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]
    except Exception:
        return None


def _which(cmd: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    pathext = os.environ.get("PATHEXT", "").split(os.pathsep) if os.name == "nt" else [""]
    exts = pathext if os.name == "nt" else [""]
    for d in paths:
        d = d.strip('"')
        if not d:
            continue
        base = Path(d) / cmd
        if base.exists():
            return str(base)
        if os.name == "nt" and not cmd.lower().endswith(".exe"):
            for e in exts:
                if not e:
                    continue
                cand = Path(d) / (cmd + e)
                if cand.exists():
                    return str(cand)
    return None


def _find_ffmpeg_exe() -> Optional[str]:
    for exe in ("ffmpeg", "ffmpeg.exe"):
        if _which(exe):
            return exe

    root = _project_root_from_helpers_file()
    common = [
        root / "presets" / "bin" / "ffmpeg.exe",
        root / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "bin" / "ffmpeg.exe",
        root / "tools" / "ffmpeg.exe",
        root / "vendor" / "ffmpeg.exe",
        root / "ffmpeg.exe",
    ]
    for p in common:
        if p.exists():
            return str(p)
    return None


ANCHORS = [
    ("Top Left", "top_left"),
    ("Top Center", "top_center"),
    ("Top Right", "top_right"),
    ("Center Left", "center_left"),
    ("Center", "center"),
    ("Center Right", "center_right"),
    ("Bottom Left", "bottom_left"),
    ("Bottom Center", "bottom_center"),
    ("Bottom Right", "bottom_right"),
    ("Custom (top-left)", "custom"),
]


@dataclass
class TextSegmentData:
    uid: str = ""
    text: str = "Text Overlay"
    font_family: str = "Arial"
    font_size: int = 48
    color_rgba: list[int] | None = None
    
    font_file: str = ""
    
    anchor: str = "bottom_center"
    offset_x: int = 0
    offset_y: int = -40

    start_ms: int = 0
    duration_ms: int = 3000
    fade_in_ms: int = 500
    fade_out_ms: int = 500

    def __post_init__(self) -> None:
        if self.color_rgba is None:
            self.color_rgba = [255, 255, 255, 255]
        if not self.uid:
            self.uid = str(uuid.uuid4())


class VideoPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.view.setBackgroundBrush(QColor(0, 0, 0))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)

        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        self._items: Dict[str, Dict[str, QGraphicsTextItem]] = {}
        
        self._native_size = QSizeF(0, 0)
        self._video_rect = QRectF(0, 0, 1, 1)

        try:
            self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)
        except Exception:
            pass

        self.setMinimumSize(QSize(480, 270))

    def _on_native_size_changed(self, size) -> None:
        try:
            self._native_size = QSizeF(size)
        except Exception:
            try:
                self._native_size = size
            except Exception:
                return
        self._update_video_layout()
        self._refresh_all_positions()

    def update_segments(self, segments: List[TextSegmentData]) -> None:
        current_uids = set(s.uid for s in segments)
        existing_uids = set(self._items.keys())

        for uid in existing_uids - current_uids:
            item_group = self._items[uid]
            self.scene.removeItem(item_group['text'])
            self.scene.removeItem(item_group['shadow'])
            del self._items[uid]

        for s in segments:
            if s.uid not in self._items:
                shadow = QGraphicsTextItem()
                shadow.setDefaultTextColor(QColor(0, 0, 0, 170))
                self.scene.addItem(shadow)
                
                text = QGraphicsTextItem()
                self.scene.addItem(text)
                
                self._items[s.uid] = {'text': text, 'shadow': shadow}
            
            group = self._items[s.uid]
            font = QFont(s.font_family)
            font.setPixelSize(int(s.font_size))
            
            group['text'].setFont(font)
            group['shadow'].setFont(font)
            
            group['text'].setPlainText(s.text)
            group['shadow'].setPlainText(s.text)
            
            c = QColor(*s.color_rgba)
            group['text'].setDefaultTextColor(c)
            
        self._refresh_all_positions()

    def set_visibility(self, uid: str, visible: bool) -> None:
        if uid in self._items:
            self._items[uid]['text'].setVisible(visible)
            self._items[uid]['shadow'].setVisible(visible)

    def set_opacity(self, uid: str, opacity: float) -> None:
        if uid in self._items:
            self._items[uid]['text'].setOpacity(opacity)
            self._items[uid]['shadow'].setOpacity(opacity)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_video_layout()
        self._refresh_all_positions()

    def _update_video_layout(self) -> None:
        vp = self.view.viewport().rect()
        vw = float(vp.width())
        vh = float(vp.height())
        if vw <= 1 or vh <= 1:
            return

        self.scene.setSceneRect(0, 0, vw, vh)

        ns = self._native_size
        ns_w = float(ns.width()) if hasattr(ns, "width") else 0.0
        ns_h = float(ns.height()) if hasattr(ns, "height") else 0.0
        if ns_w <= 1 or ns_h <= 1:
            ns_w, ns_h = vw, vh

        scale = min(vw / ns_w, vh / ns_h) if ns_w > 0 and ns_h > 0 else 1.0
        out_w = max(1.0, ns_w * scale)
        out_h = max(1.0, ns_h * scale)
        x = (vw - out_w) / 2.0
        y = (vh - out_h) / 2.0

        self._video_rect = QRectF(x, y, out_w, out_h)

        try:
            self.video_item.setSize(QSizeF(out_w, out_h))
        except Exception:
            pass
        self.video_item.setPos(x, y)

    def _anchor_pos(self, rect: QRectF, text_w: float, text_h: float, anchor: str) -> Tuple[float, float]:
        x0 = rect.x()
        y0 = rect.y()
        w = rect.width()
        h = rect.height()

        if anchor == "top_left":
            return (x0, y0)
        if anchor == "top_center":
            return (x0 + (w - text_w) / 2.0, y0)
        if anchor == "top_right":
            return (x0 + (w - text_w), y0)

        if anchor == "center_left":
            return (x0, y0 + (h - text_h) / 2.0)
        if anchor == "center":
            return (x0 + (w - text_w) / 2.0, y0 + (h - text_h) / 2.0)
        if anchor == "center_right":
            return (x0 + (w - text_w), y0 + (h - text_h) / 2.0)

        if anchor == "bottom_left":
            return (x0, y0 + (h - text_h))
        if anchor == "bottom_center":
            return (x0 + (w - text_w) / 2.0, y0 + (h - text_h))
        if anchor == "bottom_right":
            return (x0 + (w - text_w), y0 + (h - text_h))

        return (x0, y0)

    def _refresh_all_positions(self) -> None:
        pass

    def update_positions_from_data(self, segments: List[TextSegmentData]) -> None:
        for s in segments:
            if s.uid not in self._items:
                continue
            
            br = self._items[s.uid]['text'].boundingRect()
            text_w = float(br.width())
            text_h = float(br.height())

            base_rect = self._video_rect
            if base_rect.width() <= 1 or base_rect.height() <= 1:
                base_rect = self.scene.sceneRect()

            base_x, base_y = self._anchor_pos(base_rect, text_w, text_h, s.anchor)

            if s.anchor == "custom":
                x = float(s.offset_x)
                y = float(s.offset_y)
            else:
                x = base_x + float(s.offset_x)
                y = base_y + float(s.offset_y)

            clamp_rect = base_rect.adjusted(-base_rect.width(), -base_rect.height(), base_rect.width(), base_rect.height())
            x = _clamp(x, clamp_rect.left(), clamp_rect.right())
            y = _clamp(y, clamp_rect.top(), clamp_rect.bottom())

            self._items[s.uid]['text'].setPos(x, y)
            self._items[s.uid]['shadow'].setPos(x + 2, y + 2)


class TimelineWidget(QWidget):
    seekRequested = Signal(int)
    segmentSelected = Signal(str) 
    segmentChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(88)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._duration_ms = 0
        self._zoom = 1.0
        self._px_per_sec_base = 80.0
        self._margin_left = 16
        self._margin_right = 16

        self._playhead_ms = 0
        self._segments: List[TextSegmentData] = []

        self._dragging_seg: Optional[TextSegmentData] = None
        self._drag_start_pos = 0
        self._drag_start_seg_ms = 0

        self.setMouseTracking(True)

    def set_duration_ms(self, ms: int) -> None:
        self._duration_ms = max(0, int(ms))
        self._recompute_width()
        self.update()

    def set_zoom(self, z: float) -> None:
        self._zoom = float(_clamp(z, 0.25, 10.0))
        self._recompute_width()
        self.update()

    def set_playhead_ms(self, ms: int) -> None:
        self._playhead_ms = int(_clamp(ms, 0, self._duration_ms if self._duration_ms else ms))
        self.update()

    def set_segments(self, segments: List[TextSegmentData]) -> None:
        self._segments = segments
        self.update()

    def highlight_segment(self, uid: str) -> None:
        self.update()

    def _recompute_width(self) -> None:
        dur_s = (self._duration_ms / 1000.0) if self._duration_ms else 0.0
        px_per_sec = self._px_per_sec_base * self._zoom
        w = int(self._margin_left + self._margin_right + (dur_s * px_per_sec))
        self.setMinimumWidth(max(320, w))

    def _px_per_sec(self) -> float:
        return self._px_per_sec_base * self._zoom

    def _time_to_x(self, ms: int) -> float:
        return self._margin_left + (ms / 1000.0) * self._px_per_sec()

    def _x_to_time(self, x: float) -> int:
        x = float(x) - self._margin_left
        if x <= 0:
            return 0
        ms = int((x / self._px_per_sec()) * 1000.0)
        if self._duration_ms > 0:
            ms = int(_clamp(ms, 0, self._duration_ms))
        return ms

    def _seg_rect(self, s: TextSegmentData) -> QRect:
        x1 = self._time_to_x(s.start_ms)
        x2 = self._time_to_x(s.start_ms + s.duration_ms)
        top = 28
        h = 34
        return QRect(int(x1), top, int(max(4.0, x2 - x1)), h)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.fillRect(self.rect(), QColor(20, 20, 20))
        track = QRect(self._margin_left, 18, self.width() - self._margin_left - self._margin_right, 56)
        p.fillRect(track, QColor(28, 28, 28))

        if self._duration_ms > 0:
            dur_s = self._duration_ms / 1000.0
            px_per_sec = self._px_per_sec()

            desired = 100.0
            step_s = max(0.5, round(desired / px_per_sec, 1))
            nice = [0.5, 1, 2, 5, 10, 15, 30, 60, 120]
            step_s = min(nice, key=lambda v: abs(v - step_s))
            major_s = step_s
            minor_s = major_s / 5.0

            p.setPen(QPen(QColor(55, 55, 55)))
            t = 0.0
            while t <= dur_s + 1e-6:
                x = self._margin_left + t * px_per_sec
                if int(round(t / minor_s)) % 5 != 0:
                    p.drawLine(int(x), track.bottom() - 10, int(x), track.bottom())
                t += minor_s

            p.setPen(QPen(QColor(85, 85, 85)))
            font = p.font()
            font.setPointSize(max(8, int(8 + (self._zoom - 1) * 1.2)))
            p.setFont(font)

            t = 0.0
            while t <= dur_s + 1e-6:
                x = self._margin_left + t * px_per_sec
                p.drawLine(int(x), track.top(), int(x), track.bottom())
                label = self._format_time(int(t * 1000.0))
                p.drawText(int(x) + 4, track.top() + 12, label)
                t += major_s

        for s in self._segments:
            seg = self._seg_rect(s)
            import hashlib
            hue = (int(hashlib.sha256(s.uid.encode()).hexdigest(), 16) % 360)
            color = QColor.fromHsv(hue, 200, 230)
            
            p.setPen(QPen(QColor(255, 255, 255, 40)))
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 170))
            p.drawRoundedRect(seg, 6, 6)

            p.setPen(QPen(QColor(255, 255, 255, 220)))
            lbl_txt = (s.text[:10] + '..') if len(s.text) > 10 else s.text
            seg_label = f"{lbl_txt} {self._format_time(s.start_ms)}"
            p.drawText(seg.x() + 4, seg.y() + 14, seg_label)

        if self._duration_ms > 0:
            x = self._time_to_x(self._playhead_ms)
            p.setPen(QPen(QColor(255, 80, 80, 220), 2))
            p.drawLine(int(x), track.top() - 6, int(x), track.bottom() + 6)

        p.setPen(QPen(QColor(45, 45, 45)))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()

        for s in reversed(self._segments):
            if self._seg_rect(s).contains(pos):
                self._dragging_seg = s
                self._drag_start_pos = pos.x()
                self._drag_start_seg_ms = s.start_ms
                self.segmentSelected.emit(s.uid)
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return

        ms = self._x_to_time(pos.x())
        self.seekRequested.emit(ms)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not self._duration_ms:
            return
        pos = event.position().toPoint()
        if self._dragging_seg:
            dx = pos.x() - self._drag_start_pos
            ms_delta = int((dx / self._px_per_sec()) * 1000.0)
            new_start = self._drag_start_seg_ms + ms_delta
            new_start = int(_clamp(new_start, 0, max(0, self._duration_ms - self._dragging_seg.duration_ms)))
            if new_start != self._dragging_seg.start_ms:
                self._dragging_seg.start_ms = new_start
                self.segmentChanged.emit(self._dragging_seg) 
                self.update()
            event.accept()
            return

        cursor = Qt.ArrowCursor
        for s in reversed(self._segments):
            if self._seg_rect(s).contains(pos):
                cursor = Qt.OpenHandCursor
                break
        self.setCursor(cursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging_seg:
            self._dragging_seg = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()

    @staticmethod
    def _format_time(ms: int) -> str:
        ms = int(max(0, ms))
        s, ms_rem = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}.{ms_rem:03d}"
        return f"{m:02d}:{s:02d}.{ms_rem:03d}"


class VideoTextPane(QWidget):
    settingsChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.segments: List[TextSegmentData] = []
        self.selected_uid: Optional[str] = None
        self.zoom: float = 1.0
        self.last_video_path: str = ""
        self.preview_always_show: bool = False

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)

        self.preview = VideoPreview(self)
        self.player.setVideoOutput(self.preview.video_item)

        self._tick = QTimer(self)
        self._tick.setInterval(33)
        self._tick.timeout.connect(self._update_overlay_visibility)
        self._tick.start()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_settings_now)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        controls = QHBoxLayout()
        self.btn_open = QPushButton("Open Video...")
        self.btn_export = QPushButton("Export with Text...")
        self.lbl_path = QLabel("No video loaded")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_play = QPushButton("Play/Pause")
        self.btn_stop = QPushButton("Stop")
        self.lbl_time = QLabel("00:00.000 / 00:00.000")

        controls.addWidget(self.btn_open, 0)
        controls.addWidget(self.btn_play, 0)
        controls.addWidget(self.btn_stop, 0)
        controls.addWidget(self.btn_export, 0)
        controls.addWidget(self.lbl_time, 0)
        controls.addWidget(self.lbl_path, 1)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.StyledPanel)
        vlay = QVBoxLayout(video_frame)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.addWidget(self.preview, 1)
        splitter.addWidget(video_frame)

        right_panel = QWidget()
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0,0,0,0)
        
        list_grp = QFrame()
        list_grp.setFrameShape(QFrame.StyledPanel)
        list_lay = QVBoxLayout(list_grp)
        list_lay.addWidget(QLabel("Text Segments:"))
        
        self.list_segments = QListWidget()
        self.list_segments.itemClicked.connect(self._on_list_selection)
        list_lay.addWidget(self.list_segments)
        
        btns_l = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_rem = QPushButton("- Remove")
        self.btn_add.clicked.connect(self._add_segment)
        self.btn_rem.clicked.connect(self._remove_segment)
        btns_l.addWidget(self.btn_add)
        btns_l.addWidget(self.btn_rem)
        list_lay.addLayout(btns_l)
        
        right_lay.addWidget(list_grp)

        form_grp = QFrame()
        form_grp.setFrameShape(QFrame.StyledPanel)
        form_lay = QVBoxLayout(form_grp)
        
        self.edit_text = QLineEdit()
        self.combo_font = QFontComboBox()
        self.spin_size = QSpinBox()
        self.spin_size.setRange(6, 300)
        self.btn_color = QToolButton()
        self.btn_color.setText("Color...")
        self.edit_fontfile = QLineEdit()
        self.btn_fontfile = QPushButton("Browse...")

        self.combo_anchor = QComboBox()
        for label, key in ANCHORS:
            self.combo_anchor.addItem(label, key)
        self.spin_x = QSpinBox()
        self.spin_x.setRange(-4000, 4000)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(-4000, 4000)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 24 * 60 * 60.0)
        self.spin_start.setDecimals(3)
        self.spin_start.setSingleStep(0.1)

        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(0.05, 24 * 60 * 60.0)
        self.spin_dur.setDecimals(3)
        self.spin_dur.setSingleStep(0.1)

        self.spin_fi = QSpinBox()
        self.spin_fi.setRange(0, 10000)
        self.spin_fi.setSuffix(" ms")
        self.spin_fi.setSingleStep(100)
        self.spin_fi.setToolTip("Fade In Duration")
        
        self.spin_fo = QSpinBox()
        self.spin_fo.setRange(0, 10000)
        self.spin_fo.setSuffix(" ms")
        self.spin_fo.setSingleStep(100)
        self.spin_fo.setToolTip("Fade Out Duration")

        self.chk_preview_always = QCheckBox("Always show overlay in preview")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setContentsMargins(10, 10, 10, 10)
        
        form.addRow(QLabel("Text"), self.edit_text)

        font_row = QHBoxLayout()
        font_row.addWidget(self.combo_font, 1)
        font_row.addWidget(QLabel("Sz"), 0)
        font_row.addWidget(self.spin_size, 0)
        font_row.addWidget(self.btn_color, 0)
        form.addRow(QLabel("Font"), font_row)

        ff_row = QHBoxLayout()
        ff_row.addWidget(self.edit_fontfile, 1)
        ff_row.addWidget(self.btn_fontfile, 0)
        form.addRow(QLabel("Font File"), ff_row)

        pos_row = QHBoxLayout()
        pos_row.addWidget(self.combo_anchor, 1)
        pos_row.addWidget(QLabel("X"), 0)
        pos_row.addWidget(self.spin_x, 0)
        pos_row.addWidget(QLabel("Y"), 0)
        pos_row.addWidget(self.spin_y, 0)
        form.addRow(QLabel("Position"), pos_row)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Start"), 0)
        time_row.addWidget(self.spin_start, 1)
        time_row.addWidget(QLabel("Dur"), 0)
        time_row.addWidget(self.spin_dur, 1)
        form.addRow(QLabel("Timing (s)"), time_row)

        fade_row = QHBoxLayout()
        fade_row.addWidget(QLabel("Fade In"), 0)
        fade_row.addWidget(self.spin_fi, 1)
        fade_row.addWidget(QLabel("Fade Out"), 0)
        fade_row.addWidget(self.spin_fo, 1)
        form.addRow(QLabel("Fades"), fade_row)

        form.addRow(self.chk_preview_always)
        
        form_lay.addLayout(form)
        form_lay.addStretch(1)
        right_lay.addWidget(form_grp)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        bottom = QVBoxLayout()
        bottom.setSpacing(6)

        zoom_row = QHBoxLayout()
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(0, 1000)
        self.slider_zoom.setValue(self._zoom_to_slider(self.zoom))
        self.lbl_zoom = QLabel("")
        zoom_row.addWidget(QLabel("Timeline zoom"), 0)
        zoom_row.addWidget(self.slider_zoom, 1)
        zoom_row.addWidget(self.lbl_zoom, 0)
        bottom.addLayout(zoom_row)

        self.timeline = TimelineWidget()
        self.timeline.segmentSelected.connect(self._select_segment_by_uid)
        self.timeline.segmentChanged.connect(self._on_segment_dragged)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.StyledPanel)
        self.scroll.setWidget(self.timeline)
        bottom.addWidget(self.scroll, 1)

        root.addLayout(bottom, 0)

        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self.player.stop)
        self.btn_export.clicked.connect(self._export_video)
        
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.timeline.seekRequested.connect(self._seek_to_ms)
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider)

        self.btn_color.clicked.connect(self._pick_color)
        self.btn_fontfile.clicked.connect(self._pick_fontfile)

        self.edit_text.textChanged.connect(self._on_inputs_changed)
        self.combo_font.currentFontChanged.connect(self._on_inputs_changed)
        self.spin_size.valueChanged.connect(self._on_inputs_changed)
        self.edit_fontfile.textChanged.connect(self._on_inputs_changed)
        self.combo_anchor.currentIndexChanged.connect(self._on_inputs_changed)
        self.spin_x.valueChanged.connect(self._on_inputs_changed)
        self.spin_y.valueChanged.connect(self._on_inputs_changed)
        self.spin_start.valueChanged.connect(self._on_inputs_changed)
        self.spin_dur.valueChanged.connect(self._on_inputs_changed)
        self.spin_fi.valueChanged.connect(self._on_inputs_changed)
        self.spin_fo.valueChanged.connect(self._on_inputs_changed)
        self.chk_preview_always.toggled.connect(self._on_inputs_changed)

        self._load_settings()
        self._refresh_segment_list()
        if self.segments:
            self._select_segment_by_uid(self.segments[0].uid)
        else:
            self._add_segment() 
        
        self._apply_selected_to_ui()
        self._update_zoom_label()
        self._update_overlay_visibility(force=True)

        if self.last_video_path:
            try:
                p = Path(self.last_video_path)
                if p.exists():
                    self._load_video(str(p))
            except Exception:
                pass

    # --- Video & Playback ---
    def _open_video(self) -> None:
        start_dir = ""
        if self.last_video_path:
            try:
                start_dir = str(Path(self.last_video_path).parent)
            except Exception:
                start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            start_dir,
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*.*)",
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        self.last_video_path = path
        self.lbl_path.setText(path)
        self.player.setSource(QUrl.fromLocalFile(path))
        self._schedule_save()

    def _toggle_play_pause(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _seek_to_ms(self, ms: int) -> None:
        self.player.setPosition(int(ms))

    def _on_position_changed(self, pos: int) -> None:
        dur = self.player.duration()
        self.lbl_time.setText(f"{TimelineWidget._format_time(pos)} / {TimelineWidget._format_time(dur)}")
        self.timeline.set_playhead_ms(pos)
        self._update_overlay_visibility()

    def _on_duration_changed(self, dur: int) -> None:
        self.timeline.set_duration_ms(dur)
        for s in self.segments:
            if s.start_ms > dur:
                s.start_ms = dur
            if s.start_ms + s.duration_ms > dur:
                s.duration_ms = max(1, dur - s.start_ms)
        
        self.timeline.set_segments(self.segments)
        self._apply_selected_to_ui() 
        self._schedule_save()
        self._emit_settings()

    # --- Segment Management ---
    def _add_segment(self) -> None:
        new_seg = TextSegmentData()
        new_seg.start_ms = int(self.player.position())
        dur = self.player.duration()
        if dur > 0:
            new_seg.duration_ms = min(3000, max(500, dur - new_seg.start_ms))
        else:
            new_seg.duration_ms = 3000
            
        self.segments.append(new_seg)
        self._refresh_segment_list()
        self._select_segment_by_uid(new_seg.uid)
        self._update_timeline()
        self._schedule_save()
        self._emit_settings()

    def _remove_segment(self) -> None:
        row = self.list_segments.currentRow()
        if row < 0 or not self.selected_uid:
            return
        self.segments = [s for s in self.segments if s.uid != self.selected_uid]
        self.selected_uid = None
        self._refresh_segment_list()
        self._update_timeline()
        self.preview.update_segments(self.segments)
        self._schedule_save()
        self._emit_settings()
        
        if self.segments:
            self._select_segment_by_uid(self.segments[0].uid)
        else:
            pass

    def _refresh_segment_list(self) -> None:
        self.list_segments.clear()
        for s in self.segments:
            item = QListWidgetItem(s.text)
            item.setData(Qt.UserRole, s.uid)
            if s.uid == self.selected_uid:
                item.setSelected(True)
                b = item.background()
                item.setBackground(QColor(50, 100, 150))
            self.list_segments.addItem(item)

    def _on_list_selection(self, item: QListWidgetItem) -> None:
        uid = item.data(Qt.UserRole)
        self._select_segment_by_uid(uid)

    def _select_segment_by_uid(self, uid: str) -> None:
        self.selected_uid = uid
        
        for r in range(self.list_segments.count()):
            item = self.list_segments.item(r)
            if item.data(Qt.UserRole) == uid:
                self.list_segments.setCurrentItem(item)
                break
        
        self._apply_selected_to_ui()
        self.timeline.highlight_segment(uid)

    def _get_selected_segment(self) -> Optional[TextSegmentData]:
        for s in self.segments:
            if s.uid == self.selected_uid:
                return s
        return None

    def _on_segment_dragged(self, segment: TextSegmentData) -> None:
        if segment.uid == self.selected_uid:
            self._apply_selected_to_ui()
        self._schedule_save()
        self._emit_settings()

    # --- Preview & Overlay Logic ---
    def _update_overlay_visibility(self, force: bool = False) -> None:
        pos = int(self.player.position())
        always = self.chk_preview_always.isChecked()

        for s in self.segments:
            visible = always or (pos >= s.start_ms and pos <= (s.start_ms + s.duration_ms))
            self.preview.set_visibility(s.uid, visible)
            
            if visible or force:
                opacity = 1.0
                if not always:
                    rel_t = pos - s.start_ms
                    
                    if rel_t < s.fade_in_ms:
                        if s.fade_in_ms > 0:
                            opacity = rel_t / s.fade_in_ms
                        else:
                            opacity = 1.0
                    
                    time_left = s.duration_ms - rel_t
                    if time_left < s.fade_out_ms:
                        if s.fade_out_ms > 0:
                            opacity = min(opacity, time_left / s.fade_out_ms)
                
                opacity = _clamp(opacity, 0.0, 1.0)
                self.preview.set_opacity(s.uid, opacity)

    # --- Form Handling ---
    def _apply_selected_to_ui(self) -> None:
        s = self._get_selected_segment()
        if not s:
            return

        self.edit_text.blockSignals(True)
        self.combo_font.blockSignals(True)
        self.spin_size.blockSignals(True)
        self.edit_fontfile.blockSignals(True)
        self.combo_anchor.blockSignals(True)
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_start.blockSignals(True)
        self.spin_dur.blockSignals(True)
        self.spin_fi.blockSignals(True)
        self.spin_fo.blockSignals(True)
        self.chk_preview_always.blockSignals(True)

        self.edit_text.setText(s.text)
        self.combo_font.setCurrentFont(QFont(s.font_family))
        self.spin_size.setValue(int(s.font_size))
        self.edit_fontfile.setText(s.font_file or "")

        idx = 0
        for i in range(self.combo_anchor.count()):
            if self.combo_anchor.itemData(i) == s.anchor:
                idx = i
                break
        self.combo_anchor.setCurrentIndex(idx)

        self.spin_x.setValue(int(s.offset_x))
        self.spin_y.setValue(int(s.offset_y))
        
        self.spin_start.setValue(s.start_ms / 1000.0)
        self.spin_dur.setValue(s.duration_ms / 1000.0)
        
        self.spin_fi.setValue(s.fade_in_ms)
        self.spin_fo.setValue(s.fade_out_ms)

        self.chk_preview_always.setChecked(self.preview_always_show)

        self.edit_text.blockSignals(False)
        self.combo_font.blockSignals(False)
        self.spin_size.blockSignals(False)
        self.edit_fontfile.blockSignals(False)
        self.combo_anchor.blockSignals(False)
        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)
        self.spin_start.blockSignals(False)
        self.spin_dur.blockSignals(False)
        self.spin_fi.blockSignals(False)
        self.spin_fo.blockSignals(False)
        self.chk_preview_always.blockSignals(False)

    def _on_inputs_changed(self, *args) -> None:
        s = self._get_selected_segment()
        if not s:
            return
            
        s.text = self.edit_text.text()
        s.font_family = self.combo_font.currentFont().family()
        s.font_size = int(self.spin_size.value())
        s.font_file = self.edit_fontfile.text().strip()

        s.anchor = str(self.combo_anchor.currentData())
        s.offset_x = int(self.spin_x.value())
        s.offset_y = int(self.spin_y.value())

        s.start_ms = int(self.spin_start.value() * 1000.0)
        s.duration_ms = int(max(1, self.spin_dur.value() * 1000.0))
        
        s.fade_in_ms = self.spin_fi.value()
        s.fade_out_ms = self.spin_fo.value()
        
        self.preview_always_show = self.chk_preview_always.isChecked()

        dur = int(self.player.duration())
        if dur > 0:
            if s.start_ms > dur:
                s.start_ms = dur
            if s.start_ms + s.duration_ms > dur:
                s.duration_ms = max(1, dur - s.start_ms)

        self.preview.update_segments(self.segments)
        self.preview.update_positions_from_data(self.segments)
        
        current_item = self.list_segments.currentItem()
        if current_item:
            current_item.setText(s.text)
            
        self._update_timeline()
        self._schedule_save()
        self._emit_settings()

    def _pick_color(self) -> None:
        s = self._get_selected_segment()
        if not s:
            return
        cur = QColor(*s.color_rgba)
        picked = QColorDialog.getColor(cur, self, "Pick text color")
        if picked.isValid():
            s.color_rgba = [picked.red(), picked.green(), picked.blue(), picked.alpha()]
            self.preview.update_segments(self.segments)
            self.preview.update_positions_from_data(self.segments)
            self._schedule_save()
            self._emit_settings()

    def _pick_fontfile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick a font file",
            "",
            "Fonts (*.ttf *.otf *.ttc);;All files (*.*)",
        )
        if path:
            self.edit_fontfile.setText(path)

    # --- Timeline & Zoom ---
    def _update_timeline(self) -> None:
        self.timeline.set_segments(self.segments)

    def _on_zoom_slider(self, _value: int) -> None:
        z = self._slider_to_zoom(self.slider_zoom.value())
        self.zoom = z
        self.timeline.set_zoom(z)
        self._update_zoom_label()
        self._schedule_save()
        self._emit_settings()

    def _update_zoom_label(self) -> None:
        self.lbl_zoom.setText(f"{self.zoom:.2f}x")

    # --- Export ---
    def _export_video(self) -> None:
        src = self.player.source()
        if not src or not src.isLocalFile():
            QMessageBox.information(self, "Export", "Please open a local video file first.")
            return
        in_path = src.toLocalFile()
        if not in_path or not Path(in_path).exists():
            QMessageBox.information(self, "Export", "Video file not found.")
            return

        ffmpeg = _find_ffmpeg_exe()
        if not ffmpeg:
            QMessageBox.warning(self, "Export", "ffmpeg not found (PATH or presets/bin).")
            return

        default_out = str(Path(in_path).with_name(Path(in_path).stem + "_text.mp4"))
        out_path, _ = QFileDialog.getSaveFileName(self, "Export video", default_out, "MP4 Video (*.mp4);;All files (*.*)")
        if not out_path:
            return

        filters = []
        for i, s in enumerate(self.segments):
            if not s.text.strip():
                continue
            
            filt_str, err = self._build_drawtext_filter(s)
            if err:
                QMessageBox.warning(self, "Export Error", f"Segment '{s.text}': {err}")
                return
            
            input_label = "[0:v]" if i == 0 else f"[v{i}]"
            output_label = f"[v{i+1}]" if i < len(self.segments) - 1 else ""
            
            filters.append(f"{input_label}{filt_str}{output_label}")

        if not filters:
            QMessageBox.warning(self, "Export", "No text to export.")
            return

        vf_filter = ";".join(filters)

        cmd = [
            ffmpeg,
            "-y",
            "-i", in_path,
            "-vf", vf_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            out_path,
        ]

        self.btn_export.setEnabled(False)
        self.btn_export.setText("Exporting...")

        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                msg = (p.stderr or "").strip()
                if len(msg) > 1400:
                    msg = msg[-1400:]
                QMessageBox.warning(self, "Export failed", msg or "ffmpeg returned an error.")
            else:
                QMessageBox.information(self, "Export complete", f"Saved:\n{out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
        finally:
            self.btn_export.setEnabled(True)
            self.btn_export.setText("Export with Text...")

    def _build_drawtext_filter(self, s: TextSegmentData) -> Tuple[str, Optional[str]]:
        fontfile = (s.font_file or "").strip()
        if not fontfile and os.name == "nt":
            guess = _guess_windows_fontfile(s.font_family)
            if guess:
                fontfile = guess
        
        start_s = _ms_to_s(s.start_ms)
        end_s = _ms_to_s(s.start_ms + s.duration_ms)
        fi_s = _ms_to_s(s.fade_in_ms)
        fo_s = _ms_to_s(s.fade_out_ms)
        
        ox = int(s.offset_x)
        oy = int(s.offset_y)

        if s.anchor == "custom":
            x_expr = f"{ox}"
            y_expr = f"{oy}"
        else:
            x_expr, y_expr = self._anchor_to_drawtext_xy(s.anchor, ox, oy)

        txt = _ff_escape_value(s.text)
        color = _rgba_to_ffmpeg_color(s.color_rgba)

        if fi_s > 0.0001:
             fade_in_expr = f"(t-{start_s:.3f})/{fi_s:.3f}"
        else:
             fade_in_expr = "1.0"

        if fo_s > 0.0001:
             fade_out_expr = f"({end_s:.3f}-t)/{fo_s:.3f}"
        else:
             fade_out_expr = "1.0"

        alpha_expr = (
            f"if(lt(t,{start_s:.3f}),0,"
            f"if(lt(t,{start_s+fi_s:.3f}),{fade_in_expr},"
            f"if(lt(t,{end_s-fo_s:.3f}),1,"
            f"if(lt(t,{end_s:.3f}),{fade_out_expr},0))))"
        )

        parts = ["drawtext="]
        if fontfile:
            ff = _ff_escape_value(fontfile)
            parts.append(f"fontfile='{ff}':")

        parts.append(f"text='{txt}':")
        parts.append(f"fontcolor={color}:")
        parts.append(f"fontsize={int(s.font_size)}:")
        parts.append(f"x={x_expr}:")
        parts.append(f"y={y_expr}:")
        parts.append(f"alpha='{alpha_expr}':")
        parts.append("shadowcolor=black@0.65:shadowx=2:shadowy=2")

        filt = "".join(parts)

        if not fontfile:
            return "", "Pick a font file first (Font file -> Browse...)."

        return filt, None

    def _anchor_to_drawtext_xy(self, anchor: str, ox: int, oy: int) -> Tuple[str, str]:
        if anchor == "top_left":
            return f"{ox}", f"{oy}"
        if anchor == "top_center":
            return f"(w-text_w)/2+({ox})", f"{oy}"
        if anchor == "top_right":
            return f"w-text_w+({ox})", f"{oy}"

        if anchor == "center_left":
            return f"{ox}", f"(h-text_h)/2+({oy})"
        if anchor == "center":
            return f"(w-text_w)/2+({ox})", f"(h-text_h)/2+({oy})"
        if anchor == "center_right":
            return f"w-text_w+({ox})", f"(h-text_h)/2+({oy})"

        if anchor == "bottom_left":
            return f"{ox}", f"h-text_h+({oy})"
        if anchor == "bottom_center":
            return f"(w-text_w)/2+({ox})", f"h-text_h+({oy})"
        if anchor == "bottom_right":
            return f"w-text_w+({ox})", f"h-text_h+({oy})"

        return f"{ox}", f"{oy}"

    # --- Settings Persistence ---
    def _load_settings(self) -> None:
        path = _settings_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                
                self.last_video_path = data.get("last_video_path", "")
                self.zoom = float(_clamp(float(data.get("zoom", 1.0)), 0.25, 10.0))
                self.preview_always_show = bool(data.get("preview_always_show", False))
                
                segs_data = data.get("segments", [])
                self.segments = []
                for sdata in segs_data:
                    seg = TextSegmentData()
                    for k, v in sdata.items():
                        if hasattr(seg, k):
                            setattr(seg, k, v)
                    self.segments.append(seg)
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.segments = []

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save_settings_now(self) -> None:
        path = _settings_path()
        _ensure_parent_dir(path)
        try:
            data = {
                "last_video_path": self.last_video_path,
                "zoom": self.zoom,
                "preview_always_show": self.preview_always_show,
                "segments": [asdict(s) for s in self.segments]
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _emit_settings(self) -> None:
        try:
            self.settingsChanged.emit({
                "zoom": self.zoom,
                "segments_count": len(self.segments)
            })
        except Exception:
            pass

    # --- Helpers ---
    @staticmethod
    def _slider_to_zoom(v: int) -> float:
        import math
        t = v / 1000.0
        lo, hi = 0.25, 10.0
        z = lo * ((hi / lo) ** t)
        return float(_clamp(z, lo, hi))

    @staticmethod
    def _zoom_to_slider(z: float) -> int:
        import math
        lo, hi = 0.25, 10.0
        z = float(_clamp(z, lo, hi))
        t = math.log(z / lo) / math.log(hi / lo)
        return int(_clamp(t, 0.0, 1.0) * 1000)

    def current_settings_dict(self) -> Dict[str, Any]:
        return {
            "zoom": self.zoom,
            "segments": [asdict(s) for s in self.segments]
        }


# Backwards-compatible alias
videotextPane = VideoTextPane

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = VideoTextPane()
    w.setWindowTitle("Video Text Creator (Multi-Track)")
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())