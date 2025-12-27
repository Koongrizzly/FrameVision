"""
helpers/videotext.py

PySide6 UI pane for adding text overlays to video with:
- Text content
- Font family + size (+ optional font file for export)
- Position (anchor + x/y offset)
- Start + duration
- Video preview with overlay (uses QGraphicsVideoItem so overlay renders reliably on Windows)
- Clickable/zoomable timeline that seeks the player + draggable segment
- Export video with text via ffmpeg drawtext (single overlay segment)

Settings persistence:
- /presets/setsave/videotext.json  (relative to project root)

Standalone runner available at bottom.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from PySide6.QtCore import (
    Qt, QRect, QSize, QPoint, QUrl, Signal, QTimer
)
from PySide6.QtGui import (
    QPainter, QPen, QFont, QColor
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFontComboBox, QFileDialog, QSlider, QScrollArea,
    QFrame, QColorDialog, QToolButton, QSizePolicy, QApplication,
    QCheckBox, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsTextItem
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem


def _project_root_from_helpers_file() -> Path:
    # Assuming this file is at: <root>/helpers/videotext.py
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
    """
    Escape for ffmpeg drawtext option values:
      - backslash must be doubled
      - ':' must be escaped as '\:'
      - "'" must be escaped as "\'"
      - '%' escaped to avoid expansions in some builds
    """
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "\\'")
         .replace("%", "\\%")
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
class VideoTextSettings:
    text: str = "Your text here"
    font_family: str = "Arial"
    font_size: int = 48
    color_rgba: list[int] | None = None  # [r,g,b,a]

    # Export-only: optional font file path for ffmpeg drawtext
    font_file: str = ""

    anchor: str = "bottom_center"
    offset_x: int = 0
    offset_y: int = -40

    start_ms: int = 0
    duration_ms: int = 2500

    zoom: float = 1.0
    last_video_path: str = ""

    # Preview behavior
    preview_always_show: bool = False

    def __post_init__(self) -> None:
        if self.color_rgba is None:
            self.color_rgba = [255, 255, 255, 255]


class VideoPreview(QWidget):
    """
    QGraphicsView-based preview so text overlay is guaranteed to render over video.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing | QPainter.TextAntialiasing)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)

        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        # Shadow + text (two items)
        self.text_shadow = QGraphicsTextItem()
        self.text_shadow.setDefaultTextColor(QColor(0, 0, 0, 170))
        self.scene.addItem(self.text_shadow)

        self.text_item = QGraphicsTextItem()
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.scene.addItem(self.text_item)

        self._settings = VideoTextSettings()
        self._visible_now = True

        self.setMinimumSize(QSize(480, 270))

    def set_settings(self, s: VideoTextSettings) -> None:
        self._settings = s
        self._apply_text_style()
        self._reposition_text()

    def set_visible_now(self, v: bool) -> None:
        self._visible_now = bool(v)
        self.text_item.setVisible(self._visible_now)
        self.text_shadow.setVisible(self._visible_now)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        r = self.view.viewport().rect()
        self.video_item.setSize(r.size())
        self.scene.setSceneRect(0, 0, r.width(), r.height())
        self._reposition_text()

    def _apply_text_style(self) -> None:
        s = self._settings
        txt = s.text or ""
        font = QFont(s.font_family, int(s.font_size))
        self.text_item.setFont(font)
        self.text_shadow.setFont(font)

        self.text_item.setPlainText(txt)
        self.text_shadow.setPlainText(txt)

        c = QColor(*s.color_rgba)
        self.text_item.setDefaultTextColor(c)
        self.text_shadow.setDefaultTextColor(QColor(0, 0, 0, 170))

    def _anchor_pos(self, scene_w: float, scene_h: float, text_w: float, text_h: float, anchor: str) -> Tuple[float, float]:
        if anchor == "top_left":
            return (0.0, 0.0)
        if anchor == "top_center":
            return ((scene_w - text_w) / 2.0, 0.0)
        if anchor == "top_right":
            return (scene_w - text_w, 0.0)

        if anchor == "center_left":
            return (0.0, (scene_h - text_h) / 2.0)
        if anchor == "center":
            return ((scene_w - text_w) / 2.0, (scene_h - text_h) / 2.0)
        if anchor == "center_right":
            return (scene_w - text_w, (scene_h - text_h) / 2.0)

        if anchor == "bottom_left":
            return (0.0, scene_h - text_h)
        if anchor == "bottom_center":
            return ((scene_w - text_w) / 2.0, scene_h - text_h)
        if anchor == "bottom_right":
            return (scene_w - text_w, scene_h - text_h)

        return (0.0, 0.0)

    def _reposition_text(self) -> None:
        s = self._settings
        if not (s.text or "").strip():
            self.text_item.setVisible(False)
            self.text_shadow.setVisible(False)
            return

        rect = self.scene.sceneRect()
        scene_w = rect.width()
        scene_h = rect.height()

        br = self.text_item.boundingRect()
        text_w = br.width()
        text_h = br.height()

        base_x, base_y = self._anchor_pos(scene_w, scene_h, text_w, text_h, s.anchor)

        if s.anchor == "custom":
            x = float(s.offset_x)
            y = float(s.offset_y)
        else:
            x = base_x + float(s.offset_x)
            y = base_y + float(s.offset_y)

        x = _clamp(x, -scene_w, scene_w)
        y = _clamp(y, -scene_h, scene_h)

        self.text_item.setPos(x, y)
        self.text_shadow.setPos(x + 2, y + 2)


class TimelineWidget(QWidget):
    seekRequested = Signal(int)
    segmentChanged = Signal(int, int)

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
        self._seg_start_ms = 0
        self._seg_dur_ms = 2500

        self._dragging_seg = False
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

    def set_segment(self, start_ms: int, dur_ms: int) -> None:
        self._seg_start_ms = int(max(0, start_ms))
        self._seg_dur_ms = int(max(1, dur_ms))
        if self._duration_ms > 0:
            self._seg_start_ms = int(_clamp(self._seg_start_ms, 0, self._duration_ms))
            if self._seg_start_ms + self._seg_dur_ms > self._duration_ms:
                self._seg_dur_ms = max(1, self._duration_ms - self._seg_start_ms)
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

    def _seg_rect(self) -> QRect:
        x1 = self._time_to_x(self._seg_start_ms)
        x2 = self._time_to_x(self._seg_start_ms + self._seg_dur_ms)
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

        seg = self._seg_rect()
        p.setPen(QPen(QColor(255, 255, 255, 40)))
        p.setBrush(QColor(70, 130, 255, 170))
        p.drawRoundedRect(seg, 6, 6)

        p.setPen(QPen(QColor(255, 255, 255, 220)))
        seg_label = f"Text: {self._format_time(self._seg_start_ms)}  +{self._format_time(self._seg_dur_ms)}"
        p.drawText(seg.x() + 10, seg.y() + 22, seg_label)

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

        if self._seg_rect().contains(pos):
            self._dragging_seg = True
            self._drag_start_pos = pos.x()
            self._drag_start_seg_ms = self._seg_start_ms
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
            new_start = int(_clamp(new_start, 0, max(0, self._duration_ms - self._seg_dur_ms)))
            if new_start != self._seg_start_ms:
                self._seg_start_ms = new_start
                self.segmentChanged.emit(self._seg_start_ms, self._seg_dur_ms)
                self.update()
            event.accept()
            return

        if self._seg_rect().contains(pos):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging_seg:
            self._dragging_seg = False
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

        self._settings = self._load_settings()

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)

        self.preview = VideoPreview(self)
        self.preview.set_settings(self._settings)

        # Attach video output to graphics item (fixes overlay not showing on Windows)
        self.player.setVideoOutput(self.preview.video_item)

        self._tick = QTimer(self)
        self._tick.setInterval(33)
        self._tick.timeout.connect(self._update_overlay_visibility)
        self._tick.start()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._save_settings_now)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        controls = QHBoxLayout()
        self.btn_open = QPushButton("Open Video…")
        self.btn_export = QPushButton("Export with Text…")
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

        mid = QHBoxLayout()
        mid.setSpacing(10)

        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.StyledPanel)
        vlay = QVBoxLayout(video_frame)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.addWidget(self.preview, 1)
        mid.addWidget(video_frame, 2)

        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(10, 10, 10, 10)
        panel_lay.setSpacing(10)

        self.edit_text = QLineEdit()
        self.edit_text.setPlaceholderText("Type the overlay text…")

        self.combo_font = QFontComboBox()
        self.spin_size = QSpinBox()
        self.spin_size.setRange(6, 300)

        self.btn_color = QToolButton()
        self.btn_color.setText("Color…")

        # Export font file row
        self.edit_fontfile = QLineEdit()
        self.edit_fontfile.setPlaceholderText("Optional: font file for export (ffmpeg drawtext)")
        self.btn_fontfile = QPushButton("Browse…")

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

        self.btn_start_now = QPushButton("Start = current")
        self.btn_end_now = QPushButton("End = current")
        self.btn_seek_start = QPushButton("Seek to start")

        self.chk_preview_always = QCheckBox("Always show overlay in preview")
        self.lbl_active = QLabel("Overlay: OFF")
        self.lbl_active.setStyleSheet("QLabel { color: #ff6b6b; }")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addRow(QLabel("Text"), self.edit_text)

        font_row = QHBoxLayout()
        font_row.addWidget(self.combo_font, 1)
        font_row.addWidget(QLabel("Size"), 0)
        font_row.addWidget(self.spin_size, 0)
        font_row.addWidget(self.btn_color, 0)
        font_wrap = QWidget()
        font_wrap.setLayout(font_row)
        form.addRow(QLabel("Font"), font_wrap)

        ff_row = QHBoxLayout()
        ff_row.addWidget(self.edit_fontfile, 1)
        ff_row.addWidget(self.btn_fontfile, 0)
        ff_wrap = QWidget()
        ff_wrap.setLayout(ff_row)
        form.addRow(QLabel("Font file"), ff_wrap)

        pos_row = QHBoxLayout()
        pos_row.addWidget(self.combo_anchor, 1)
        pos_row.addWidget(QLabel("X"), 0)
        pos_row.addWidget(self.spin_x, 0)
        pos_row.addWidget(QLabel("Y"), 0)
        pos_row.addWidget(self.spin_y, 0)
        pos_wrap = QWidget()
        pos_wrap.setLayout(pos_row)
        form.addRow(QLabel("Position"), pos_wrap)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Start (s)"), 0)
        dur_row.addWidget(self.spin_start, 0)
        dur_row.addSpacing(8)
        dur_row.addWidget(QLabel("Duration (s)"), 0)
        dur_row.addWidget(self.spin_dur, 0)
        dur_row.addStretch(1)
        dur_wrap = QWidget()
        dur_wrap.setLayout(dur_row)
        form.addRow(QLabel("Timing"), dur_wrap)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start_now, 0)
        btn_row.addWidget(self.btn_end_now, 0)
        btn_row.addWidget(self.btn_seek_start, 0)
        btn_row.addStretch(1)
        btn_wrap = QWidget()
        btn_wrap.setLayout(btn_row)
        form.addRow(QLabel("Quick set"), btn_wrap)

        prev_row = QHBoxLayout()
        prev_row.addWidget(self.chk_preview_always, 0)
        prev_row.addStretch(1)
        prev_row.addWidget(self.lbl_active, 0)
        prev_wrap = QWidget()
        prev_wrap.setLayout(prev_row)
        form.addRow(QLabel("Preview"), prev_wrap)

        panel_lay.addLayout(form)
        panel_lay.addStretch(1)

        mid.addWidget(panel, 1)
        root.addLayout(mid, 1)

        bottom = QVBoxLayout()
        bottom.setSpacing(6)

        zoom_row = QHBoxLayout()
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(0, 1000)
        self.slider_zoom.setValue(self._zoom_to_slider(self._settings.zoom))
        self.lbl_zoom = QLabel("")
        zoom_row.addWidget(QLabel("Timeline zoom"), 0)
        zoom_row.addWidget(self.slider_zoom, 1)
        zoom_row.addWidget(self.lbl_zoom, 0)
        bottom.addLayout(zoom_row)

        self.timeline = TimelineWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.StyledPanel)
        self.scroll.setWidget(self.timeline)
        bottom.addWidget(self.scroll, 1)

        root.addLayout(bottom, 0)

        # wiring
        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self.player.stop)
        self.btn_export.clicked.connect(self._export_video)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)

        self.timeline.seekRequested.connect(self._seek_to_ms)
        self.timeline.segmentChanged.connect(self._on_segment_dragged)

        self.slider_zoom.valueChanged.connect(self._on_zoom_slider)

        self.btn_color.clicked.connect(self._pick_color)
        self.btn_start_now.clicked.connect(self._set_start_to_current)
        self.btn_end_now.clicked.connect(self._set_end_to_current)
        self.btn_seek_start.clicked.connect(self._seek_to_start)

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

        self.chk_preview_always.toggled.connect(self._on_inputs_changed)

        # apply
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._update_zoom_label()
        self._update_overlay_visibility(force=True)

        if self._settings.last_video_path:
            try:
                p = Path(self._settings.last_video_path)
                if p.exists():
                    self._load_video(str(p))
            except Exception:
                pass

    # open/load
    def _open_video(self) -> None:
        start_dir = ""
        if self._settings.last_video_path:
            try:
                start_dir = str(Path(self._settings.last_video_path).parent)
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
        self._settings.last_video_path = path
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

    def _seek_to_start(self) -> None:
        self.player.setPosition(int(self._settings.start_ms))

    # segment/zoom
    def _on_segment_dragged(self, start_ms: int, dur_ms: int) -> None:
        self._settings.start_ms = int(start_ms)
        self._settings.duration_ms = int(dur_ms)
        self._sync_timing_to_spins()
        self.timeline.set_segment(self._settings.start_ms, self._settings.duration_ms)
        self.player.setPosition(int(self._settings.start_ms))
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _on_zoom_slider(self, _value: int) -> None:
        z = self._slider_to_zoom(self.slider_zoom.value())
        self._settings.zoom = z
        self.timeline.set_zoom(z)
        self._update_zoom_label()
        self._schedule_save()
        self._emit_settings()

    def _update_zoom_label(self) -> None:
        self.lbl_zoom.setText(f"{self._settings.zoom:.2f}×")

    # ui actions
    def _pick_color(self) -> None:
        cur = QColor(*self._settings.color_rgba)
        picked = QColorDialog.getColor(cur, self, "Pick text color")
        if not picked.isValid():
            return
        self._settings.color_rgba = [picked.red(), picked.green(), picked.blue(), picked.alpha()]
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _pick_fontfile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick a font file",
            "",
            "Fonts (*.ttf *.otf *.ttc);;All files (*.*)",
        )
        if not path:
            return
        self.edit_fontfile.setText(path)

    def _set_start_to_current(self) -> None:
        self._settings.start_ms = int(self.player.position())
        dur = self.player.duration()
        if dur > 0 and self._settings.start_ms + self._settings.duration_ms > dur:
            self._settings.duration_ms = max(1, dur - self._settings.start_ms)
        self._sync_timing_to_spins()
        self.timeline.set_segment(self._settings.start_ms, self._settings.duration_ms)
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _set_end_to_current(self) -> None:
        end = int(self.player.position())
        start = int(self._settings.start_ms)
        if end <= start:
            self._settings.duration_ms = 50
        else:
            self._settings.duration_ms = max(1, end - start)

        dur = self.player.duration()
        if dur > 0 and start + self._settings.duration_ms > dur:
            self._settings.duration_ms = max(1, dur - start)

        self._sync_timing_to_spins()
        self.timeline.set_segment(self._settings.start_ms, self._settings.duration_ms)
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    # player callbacks
    def _on_position_changed(self, pos: int) -> None:
        dur = self.player.duration()
        self.lbl_time.setText(f"{TimelineWidget._format_time(pos)} / {TimelineWidget._format_time(dur)}")
        self.timeline.set_playhead_ms(pos)
        self._update_overlay_visibility()

    def _on_duration_changed(self, dur: int) -> None:
        self.timeline.set_duration_ms(dur)
        if dur > 0:
            if self._settings.start_ms > dur:
                self._settings.start_ms = dur
            if self._settings.start_ms + self._settings.duration_ms > dur:
                self._settings.duration_ms = max(1, dur - self._settings.start_ms)
        self.timeline.set_segment(self._settings.start_ms, self._settings.duration_ms)
        self._sync_timing_to_spins()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _update_overlay_visibility(self, force: bool = False) -> None:
        pos = int(self.player.position())
        s = self._settings

        if s.preview_always_show:
            visible = True
        else:
            visible = (pos >= s.start_ms) and (pos <= (s.start_ms + s.duration_ms))

        self.preview.set_visible_now(visible)

        if visible:
            self.lbl_active.setText("Overlay: ON")
            self.lbl_active.setStyleSheet("QLabel { color: #7CFC98; }")
        else:
            self.lbl_active.setText("Overlay: OFF")
            self.lbl_active.setStyleSheet("QLabel { color: #ff6b6b; }")

        if force:
            self.preview.update()

    # settings io
    def _load_settings(self) -> VideoTextSettings:
        path = _settings_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                s = VideoTextSettings()
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)

                s.font_size = int(_clamp(int(getattr(s, "font_size", 48)), 6, 300))
                s.offset_x = int(_clamp(int(getattr(s, "offset_x", 0)), -4000, 4000))
                s.offset_y = int(_clamp(int(getattr(s, "offset_y", 0)), -4000, 4000))
                s.start_ms = int(max(0, int(getattr(s, "start_ms", 0))))
                s.duration_ms = int(max(1, int(getattr(s, "duration_ms", 2500))))
                s.zoom = float(_clamp(float(getattr(s, "zoom", 1.0)), 0.25, 10.0))

                if not isinstance(getattr(s, "color_rgba", None), list) or len(s.color_rgba) != 4:
                    s.color_rgba = [255, 255, 255, 255]

                if not isinstance(getattr(s, "preview_always_show", False), bool):
                    s.preview_always_show = False

                if not isinstance(getattr(s, "font_file", ""), str):
                    s.font_file = ""

                return s
        except Exception:
            pass
        return VideoTextSettings()

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save_settings_now(self) -> None:
        path = _settings_path()
        _ensure_parent_dir(path)
        try:
            path.write_text(json.dumps(asdict(self._settings), indent=2), encoding="utf-8")
        except Exception:
            pass

    # apply/sync
    def _apply_settings_to_ui(self) -> None:
        s = self._settings
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

        self.chk_preview_always.setChecked(bool(s.preview_always_show))

        self._sync_timing_to_spins()

        self.timeline.set_zoom(s.zoom)
        self.timeline.set_segment(s.start_ms, s.duration_ms)

        if s.last_video_path:
            self.lbl_path.setText(s.last_video_path)

    def _sync_timing_to_spins(self) -> None:
        self.spin_start.blockSignals(True)
        self.spin_dur.blockSignals(True)
        self.spin_start.setValue(self._settings.start_ms / 1000.0)
        self.spin_dur.setValue(self._settings.duration_ms / 1000.0)
        self.spin_start.blockSignals(False)
        self.spin_dur.blockSignals(False)

    def _apply_settings_to_preview(self) -> None:
        self.preview.set_settings(self._settings)
        self._update_overlay_visibility(force=True)

    def _emit_settings(self) -> None:
        try:
            self.settingsChanged.emit(asdict(self._settings))
        except Exception:
            pass

    def _on_inputs_changed(self, *args) -> None:
        s = self._settings
        s.text = self.edit_text.text()
        s.font_family = self.combo_font.currentFont().family()
        s.font_size = int(self.spin_size.value())
        s.font_file = self.edit_fontfile.text().strip()

        s.anchor = str(self.combo_anchor.currentData())
        s.offset_x = int(self.spin_x.value())
        s.offset_y = int(self.spin_y.value())

        s.start_ms = int(self.spin_start.value() * 1000.0)
        s.duration_ms = int(max(1, self.spin_dur.value() * 1000.0))

        s.preview_always_show = bool(self.chk_preview_always.isChecked())

        dur = int(self.player.duration())
        if dur > 0:
            if s.start_ms > dur:
                s.start_ms = dur
            if s.start_ms + s.duration_ms > dur:
                s.duration_ms = max(1, dur - s.start_ms)

        self.timeline.set_segment(s.start_ms, s.duration_ms)
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    # zoom mapping
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

    # export
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
            QMessageBox.warning(self, "Export", "ffmpeg not found (PATH or common app folders).")
            return

        default_out = str(Path(in_path).with_name(Path(in_path).stem + "_text.mp4"))
        out_path, _ = QFileDialog.getSaveFileName(self, "Export video", default_out, "MP4 Video (*.mp4);;All files (*.*)")
        if not out_path:
            return

        filt, warn = self._build_drawtext_filter()
        if warn:
            QMessageBox.warning(self, "Export", warn)
            return

        cmd = [
            ffmpeg,
            "-y",
            "-i", in_path,
            "-vf", filt,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            out_path,
        ]

        self.btn_export.setEnabled(False)
        self.btn_export.setText("Exporting…")

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
            self.btn_export.setText("Export with Text…")

    def _build_drawtext_filter(self) -> Tuple[str, Optional[str]]:
        s = self._settings
        if not (s.text or "").strip():
            return "", "Text is empty."

        fontfile = (s.font_file or "").strip()
        if not fontfile and os.name == "nt":
            guess = _guess_windows_fontfile(s.font_family)
            if guess:
                fontfile = guess

        start_s = _ms_to_s(s.start_ms)
        end_s = _ms_to_s(s.start_ms + s.duration_ms)
        if end_s <= start_s:
            end_s = start_s + 0.05

        ox = int(s.offset_x)
        oy = int(s.offset_y)

        if s.anchor == "custom":
            x_expr = f"{ox}"
            y_expr = f"{oy}"
        else:
            x_expr, y_expr = self._anchor_to_drawtext_xy(s.anchor, ox, oy)

        txt = _ff_escape_value(s.text)
        color = _rgba_to_ffmpeg_color(s.color_rgba)

        parts = ["drawtext="]
        if fontfile:
            ff = _ff_escape_value(fontfile)
            parts.append(f"fontfile='{ff}':")

        parts.append(f"text='{txt}':")
        parts.append(f"fontcolor={color}:")
        parts.append(f"fontsize={int(s.font_size)}:")
        parts.append(f"x={x_expr}:")
        parts.append(f"y={y_expr}:")
        parts.append("shadowcolor=black@0.65:shadowx=2:shadowy=2:")
        parts.append(f"enable='between(t,{start_s:.3f},{end_s:.3f})'")

        filt = "".join(parts)

        if not fontfile:
            # exporting may fail depending on ffmpeg build; require explicit fontfile to avoid a bad experience
            return "", "Pick a font file first (Font file → Browse…). Some ffmpeg builds need it for drawtext."

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

    def current_settings_dict(self) -> Dict[str, Any]:
        return asdict(self._settings)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = VideoTextPane()
    w.setWindowTitle("Video Text Overlay")
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())
