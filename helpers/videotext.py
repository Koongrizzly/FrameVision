"""
helpers/videotext.py

Video Text Overlay / Text-to-Video Creator (PySide6)

What you get:
- Reliable on-top preview overlay on Windows (no QVideoWidget child overlay issues)
  Uses QGraphicsVideoItem + multiple QGraphicsTextItem overlays
- Multiple text overlays (each with its own text/font/size/color/position/timing)
- Fade-in / fade-out per overlay (preview + export)
- Preview pane can be hidden (for more timeline space)
- Export baked-in text via ffmpeg drawtext

Settings saved to:
  /presets/setsave/videotext.json   (relative to project root)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from PySide6.QtCore import (
    Qt, QRect, QSize, QUrl, Signal, QTimer, QSizeF, QRectF, QEvent, QPointF
)
from PySide6.QtGui import (
    QPainter, QPen, QFont, QColor, QTransform, QPixmap, QFontMetrics
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


def _to_bool(v: Any, default: bool = False) -> bool:
    """Robust bool coercion for settings loaded from JSON (handles 'False' strings, 0/1, etc.)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return default



def _ms_to_s(ms: int) -> float:
    return max(0.0, float(ms) / 1000.0)


def _ff_escape_value(s: str) -> str:
    # Escape for ffmpeg drawtext option values.
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
    """
    Best-effort Windows font file guess for ffmpeg drawtext.
    Some ffmpeg builds on Windows require an explicit font file path.
    """
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
    """
    Preferred location for this app:
      (root) /presets/bin/ffmpeg.exe
    Falls back to PATH and a few common app folders.
    """
    root = _project_root_from_helpers_file()
    preferred = [
        root / "presets" / "bin" / "ffmpeg.exe",
        root / "presets" / "bin" / "ffmpeg",
    ]
    for p in preferred:
        if p.exists():
            return str(p)

    for exe in ("ffmpeg", "ffmpeg.exe"):
        found = _which(exe)
        if found:
            return exe  # use PATH-resolvable command

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


def _new_uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TextOverlay:
    uid: str = field(default_factory=_new_uid)

    text: str = "Your text here"
    font_family: str = "Arial"
    font_size: int = 48  # pixels
    color_rgba: list[int] = field(default_factory=lambda: [255, 255, 255, 255])

    # Effects
    # - none: static
    # - flash: blink on/off
    # - color_swap: alternate between Color and Alt Color
    effect: str = "none"
    effect_hz: float = 2.0
    effect_color2_rgba: list[int] = field(default_factory=lambda: [255, 64, 64, 255])

    # Export-only: optional font file path for ffmpeg drawtext
    font_file: str = ""

    anchor: str = "bottom_center"
    offset_x: int = 0
    offset_y: int = -40

    start_ms: int = 0
    duration_ms: int = 2500

    fade_in_ms: int = 250
    fade_out_ms: int = 250

    enabled: bool = True

    def normalize(self, video_duration_ms: int | None = None) -> None:
        self.font_size = int(_clamp(int(self.font_size), 6, 300))
        self.offset_x = int(_clamp(int(self.offset_x), -4000, 4000))
        self.offset_y = int(_clamp(int(self.offset_y), -4000, 4000))

        self.start_ms = int(max(0, int(self.start_ms)))
        self.duration_ms = int(max(1, int(self.duration_ms)))

        self.fade_in_ms = int(max(0, int(self.fade_in_ms)))
        self.fade_out_ms = int(max(0, int(self.fade_out_ms)))

        # Clamp to video duration
        if video_duration_ms is not None and video_duration_ms > 0:
            if self.start_ms > video_duration_ms:
                self.start_ms = int(video_duration_ms)
            if self.start_ms + self.duration_ms > video_duration_ms:
                self.duration_ms = max(1, int(video_duration_ms - self.start_ms))

        # Fade constraints
        self.fade_in_ms = int(_clamp(self.fade_in_ms, 0, self.duration_ms))
        self.fade_out_ms = int(_clamp(self.fade_out_ms, 0, self.duration_ms))
        if self.fade_in_ms + self.fade_out_ms > self.duration_ms:
            # Scale them down proportionally (keep ratio)
            total = max(1, self.fade_in_ms + self.fade_out_ms)
            self.fade_in_ms = int(self.duration_ms * (self.fade_in_ms / total))
            self.fade_out_ms = int(self.duration_ms - self.fade_in_ms)

        # Effects
        try:
            self.effect = str(getattr(self, "effect", "none") or "none")
        except Exception:
            self.effect = "none"
        if self.effect not in ("none", "flash", "color_swap"):
            self.effect = "none"

        try:
            self.effect_hz = float(getattr(self, "effect_hz", 2.0) or 0.0)
        except Exception:
            self.effect_hz = 0.0
        self.effect_hz = float(_clamp(self.effect_hz, 0.0, 20.0))

        try:
            c2 = list(getattr(self, "effect_color2_rgba", None) or [255, 64, 64, 255])
        except Exception:
            c2 = [255, 64, 64, 255]
        if len(c2) != 4:
            c2 = [255, 64, 64, 255]
        try:
            c2 = [int(_clamp(int(v), 0, 255)) for v in c2[:4]]
        except Exception:
            c2 = [255, 64, 64, 255]
        self.effect_color2_rgba = c2


@dataclass
class VideoTextSettings:
    overlays: List[TextOverlay] = field(default_factory=list)

    # UI state
    selected_uid: str = ""
    zoom: float = 1.0
    last_video_path: str = ""

    # Preview behavior
    preview_always_show: bool = False
    preview_hidden: bool = False

    def __post_init__(self) -> None:
        if not self.overlays:
            self.overlays = [TextOverlay()]
        self.zoom = float(_clamp(float(self.zoom), 0.25, 10.0))
        if not isinstance(self.preview_always_show, bool):
            self.preview_always_show = False
        if not isinstance(self.preview_hidden, bool):
            self.preview_hidden = False
        # Ensure selection is valid
        if self.selected_uid:
            if not any(o.uid == self.selected_uid for o in self.overlays):
                self.selected_uid = self.overlays[0].uid
        else:
            self.selected_uid = self.overlays[0].uid

    def get_selected(self) -> TextOverlay:
        for o in self.overlays:
            if o.uid == self.selected_uid:
                return o
        self.selected_uid = self.overlays[0].uid
        return self.overlays[0]


class _OverlayItems:
    def __init__(self, uid: str, shadow: QGraphicsTextItem, text: QGraphicsTextItem) -> None:
        self.uid = uid
        self.shadow = shadow
        self.text = text


class VideoPreview(QWidget):
    overlaySelected = Signal(str)
    overlayMoved = Signal(str, int, int)

    """
    Preview widget with reliable overlay:
    - Video renders into QGraphicsVideoItem
    - Multiple text overlays render as QGraphicsTextItem above it
    - We compute an aspect-fit video rect and anchor each text item to that rect
    """
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

        # Mouse interactions: click/drag overlays inside the preview
        self.view.setMouseTracking(True)
        try:
            self.view.viewport().setMouseTracking(True)
            self.view.viewport().installEventFilter(self)
        except Exception:
            pass

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)

        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        self._items: Dict[str, _OverlayItems] = {}
        self._item_to_uid: Dict[object, str] = {}

        # drag state
        self._drag_uid: str = ""
        self._drag_grab: QPointF = QPointF(0, 0)
        self._drag_anchor_base: QPointF = QPointF(0, 0)

        self._settings = VideoTextSettings()
        self._native_size = QSizeF(0, 0)
        self._video_rect = QRectF(0, 0, 1, 1)

        # When available, use nativeSizeChanged to recompute the aspect-fit rect.
        try:
            self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)  # type: ignore
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
        self._reposition_all()

    def set_settings(self, s: VideoTextSettings) -> None:
        self._settings = s
        self._sync_items()
        self._update_video_layout()
        self._apply_all_styles()
        self._reposition_all()

    def update_for_time(self, pos_ms: int) -> None:
        s = self._settings
        for ov in s.overlays:
            items = self._items.get(ov.uid)
            if not items:
                continue
            if not ov.enabled or not (ov.text or "").strip():
                items.text.setVisible(False)
                items.shadow.setVisible(False)
                continue

            if s.preview_always_show:
                alpha = (ov.color_rgba[3] / 255.0) if ov.color_rgba else 1.0
                # Apply flash even when "always show" is enabled (skip time-window, but still blink)
                try:
                    eff = str(getattr(ov, "effect", "none") or "none")
                except Exception:
                    eff = "none"
                if eff == "flash":
                    try:
                        hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
                    except Exception:
                        hz = 0.0
                    if hz > 0.001:
                        per = 1000.0 / hz
                        try:
                            phase = ((pos_ms - ov.start_ms) % per) / per
                        except Exception:
                            phase = 0.0
                        if phase >= 0.5:
                            alpha = 0.0
            else:
                alpha = self._alpha_at_ms(ov, pos_ms)

            if alpha <= 0.001:
                items.text.setVisible(False)
                items.shadow.setVisible(False)
            else:
                items.text.setVisible(True)
                items.shadow.setVisible(True)
                # Color effects (swap)
                try:
                    eff = str(getattr(ov, "effect", "none") or "none")
                except Exception:
                    eff = "none"
                if eff == "color_swap":
                    try:
                        hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
                    except Exception:
                        hz = 0.0
                    if hz > 0.001:
                        per = 1000.0 / hz
                        try:
                            phase = ((pos_ms - ov.start_ms) % per) / per
                        except Exception:
                            phase = 0.0
                        if phase >= 0.5:
                            try:
                                c2 = list(getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255])
                            except Exception:
                                c2 = [255, 64, 64, 255]
                            if len(c2) == 4:
                                rgb = (int(c2[0]), int(c2[1]), int(c2[2]))
                            else:
                                rgb = None
                        else:
                            rgb = None
                    else:
                        rgb = None
                else:
                    rgb = None

                try:
                    base_rgba = list(getattr(ov, "color_rgba", None) or [255, 255, 255, 255])
                except Exception:
                    base_rgba = [255, 255, 255, 255]
                try:
                    base_rgb = (int(base_rgba[0]), int(base_rgba[1]), int(base_rgba[2]))
                except Exception:
                    base_rgb = (255, 255, 255)

                want = rgb if rgb is not None else base_rgb
                try:
                    last = getattr(items.text, "_videotext_last_rgb", None)
                except Exception:
                    last = None
                if last != want:
                    try:
                        items.text.setDefaultTextColor(QColor(want[0], want[1], want[2]))
                    except Exception:
                        pass
                    try:
                        setattr(items.text, "_videotext_last_rgb", want)
                    except Exception:
                        pass

                items.text.setOpacity(alpha)
                items.shadow.setOpacity(alpha)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_video_layout()
        self._reposition_all()


    def _uid_at(self, scene_pos: QPointF) -> str:
        """Return overlay uid under scene_pos (hits text or shadow), else ''"""
        try:
            item = self.scene.itemAt(scene_pos, QTransform())
        except Exception:
            item = None
        if not item:
            return ""
        try:
            return str(self._item_to_uid.get(item, ""))
        except Exception:
            return ""

    def _overlay_by_uid(self, uid: str) -> Optional[TextOverlay]:
        for ov in self._settings.overlays:
            if ov.uid == uid:
                return ov
        return None

    def eventFilter(self, obj, event) -> bool:
        # Dragging and selection on the preview overlay text.
        try:
            if obj is self.view.viewport():
                et = event.type()

                if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    pos = event.position()
                    scene_pos = self.view.mapToScene(int(pos.x()), int(pos.y()))
                    uid = self._uid_at(scene_pos)
                    if uid:
                        self._drag_uid = uid

                        # Update selection
                        self._settings.selected_uid = uid
                        try:
                            self.overlaySelected.emit(uid)
                        except Exception:
                            pass

                        items = self._items.get(uid)
                        ov = self._overlay_by_uid(uid)
                        if items and ov:
                            item_pos = items.text.pos()
                            self._drag_grab = scene_pos - item_pos

                            # Precompute anchor base for offset math
                            br = items.text.boundingRect()
                            text_w = float(br.width())
                            text_h = float(br.height())

                            base_rect = self._video_rect
                            if base_rect.width() <= 1 or base_rect.height() <= 1:
                                base_rect = self.scene.sceneRect()

                            base_x, base_y = self._anchor_pos(base_rect, text_w, text_h, ov.anchor)
                            self._drag_anchor_base = QPointF(base_x, base_y)

                        self.view.setCursor(Qt.ClosedHandCursor)
                        event.accept()
                        return True

                if et == QEvent.MouseMove:
                    pos = event.position()
                    scene_pos = self.view.mapToScene(int(pos.x()), int(pos.y()))

                    if self._drag_uid and (event.buttons() & Qt.LeftButton):
                        uid = self._drag_uid
                        ov = self._overlay_by_uid(uid)
                        if ov:
                            new_pos = scene_pos - self._drag_grab

                            base_rect = self._video_rect
                            if base_rect.width() <= 1 or base_rect.height() <= 1:
                                base_rect = self.scene.sceneRect()

                            # Soft clamp (same as _reposition_overlay)
                            clamp_rect = base_rect.adjusted(
                                -base_rect.width(), -base_rect.height(),
                                base_rect.width(), base_rect.height()
                            )

                            x = float(_clamp(new_pos.x(), clamp_rect.left(), clamp_rect.right()))
                            y = float(_clamp(new_pos.y(), clamp_rect.top(), clamp_rect.bottom()))

                            if ov.anchor == "custom":
                                ov.offset_x = int(round(x))
                                ov.offset_y = int(round(y))
                            else:
                                ov.offset_x = int(round(x - float(self._drag_anchor_base.x())))
                                ov.offset_y = int(round(y - float(self._drag_anchor_base.y())))

                            ov.normalize(video_duration_ms=None)
                            self._reposition_overlay(ov)

                            try:
                                self.overlayMoved.emit(uid, int(ov.offset_x), int(ov.offset_y))
                            except Exception:
                                pass

                        event.accept()
                        return True

                    # Hover cursor
                    uid = self._uid_at(scene_pos)
                    if uid:
                        self.view.setCursor(Qt.OpenHandCursor)
                    else:
                        self.view.setCursor(Qt.ArrowCursor)

                if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    if self._drag_uid:
                        self._drag_uid = ""
                        self.view.setCursor(Qt.ArrowCursor)
                        event.accept()
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _sync_items(self) -> None:
        wanted = {ov.uid for ov in self._settings.overlays}

        # Remove old
        for uid in list(self._items.keys()):
            if uid not in wanted:
                it = self._items.pop(uid)
                try:
                    self._item_to_uid.pop(it.shadow, None)
                    self._item_to_uid.pop(it.text, None)
                except Exception:
                    pass
                self.scene.removeItem(it.shadow)
                self.scene.removeItem(it.text)

        # Add new
        for ov in self._settings.overlays:
            if ov.uid in self._items:
                continue

            shadow = QGraphicsTextItem()
            shadow.setDefaultTextColor(QColor(0, 0, 0, 170))
            shadow.setZValue(10.0)

            text = QGraphicsTextItem()
            text.setDefaultTextColor(QColor(255, 255, 255))
            text.setZValue(11.0)

            self.scene.addItem(shadow)
            self.scene.addItem(text)

            self._items[ov.uid] = _OverlayItems(ov.uid, shadow, text)
            self._item_to_uid[shadow] = ov.uid
            self._item_to_uid[text] = ov.uid

    def _update_video_layout(self) -> None:
        vp = self.view.viewport().rect()
        vw = float(vp.width())
        vh = float(vp.height())
        if vw <= 1 or vh <= 1:
            return

        # Scene matches the viewport.
        self.scene.setSceneRect(0, 0, vw, vh)

        ns = self._native_size
        ns_w = float(ns.width()) if hasattr(ns, "width") else 0.0
        ns_h = float(ns.height()) if hasattr(ns, "height") else 0.0
        if ns_w <= 1 or ns_h <= 1:
            ns_w, ns_h = vw, vh

        # Aspect-fit
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

    def _apply_all_styles(self) -> None:
        for ov in self._settings.overlays:
            self._apply_overlay_style(ov)

    def _apply_overlay_style(self, ov: TextOverlay) -> None:
        items = self._items.get(ov.uid)
        if not items:
            return

        txt = ov.text or ""

        font = QFont(ov.font_family)
        font.setPixelSize(int(ov.font_size))  # pixel-based sizing

        for it in (items.text, items.shadow):
            it.setScale(1.0)
            it.setFont(font)
            it.setPlainText(txt)

        if ov.color_rgba and len(ov.color_rgba) == 4:
            c = QColor(*ov.color_rgba)
        else:
            c = QColor(255, 255, 255, 255)
        items.text.setDefaultTextColor(c)
        items.shadow.setDefaultTextColor(QColor(0, 0, 0, 170))

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

    def _reposition_all(self) -> None:
        for ov in self._settings.overlays:
            self._reposition_overlay(ov)

    def _reposition_overlay(self, ov: TextOverlay) -> None:
        items = self._items.get(ov.uid)
        if not items:
            return

        if not (ov.text or "").strip():
            items.text.setVisible(False)
            items.shadow.setVisible(False)
            return

        br = items.text.boundingRect()
        text_w = float(br.width())
        text_h = float(br.height())

        base_rect = self._video_rect
        if base_rect.width() <= 1 or base_rect.height() <= 1:
            base_rect = self.scene.sceneRect()

        base_x, base_y = self._anchor_pos(base_rect, text_w, text_h, ov.anchor)

        if ov.anchor == "custom":
            x = float(ov.offset_x)
            y = float(ov.offset_y)
        else:
            x = base_x + float(ov.offset_x)
            y = base_y + float(ov.offset_y)

        # Soft clamp around the video rect so it doesn't fly off-screen.
        clamp_rect = base_rect.adjusted(-base_rect.width(), -base_rect.height(), base_rect.width(), base_rect.height())
        x = _clamp(x, clamp_rect.left(), clamp_rect.right())
        y = _clamp(y, clamp_rect.top(), clamp_rect.bottom())

        items.text.setPos(x, y)
        items.shadow.setPos(x + 2, y + 2)

    @staticmethod
    def _alpha_at_ms(ov: TextOverlay, pos_ms: int) -> float:
        if ov.duration_ms <= 0:
            return 0.0
        if pos_ms < ov.start_ms or pos_ms > (ov.start_ms + ov.duration_ms):
            return 0.0

        base_alpha = (ov.color_rgba[3] / 255.0) if ov.color_rgba else 1.0
        if base_alpha <= 0.001:
            return 0.0

        t = pos_ms - ov.start_ms
        dur = max(1, ov.duration_ms)
        fi = int(_clamp(ov.fade_in_ms, 0, dur))
        fo = int(_clamp(ov.fade_out_ms, 0, dur))

        a = 1.0

        if fi > 0 and t < fi:
            a = t / float(fi)

        if fo > 0 and t > (dur - fo):
            a2 = (dur - t) / float(fo)
            a = min(a, a2)

        a = float(_clamp(a, 0.0, 1.0))

        # Effects (alpha mods)
        try:
            eff = str(getattr(ov, "effect", "none") or "none")
        except Exception:
            eff = "none"
        if eff == "flash":
            try:
                hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
            except Exception:
                hz = 0.0
            if hz > 0.001:
                per = 1000.0 / hz
                try:
                    phase = ((pos_ms - ov.start_ms) % per) / per
                except Exception:
                    phase = 0.0
                if phase >= 0.5:
                    return 0.0

        return base_alpha * a


class TimelineWidget(QWidget):
    seekRequested = Signal(int)
    segmentChanged = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(176)
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

    def _ui_scale(self) -> float:
        # Base design height is 88px; scale UI to the actual height so the timeline can grow cleanly.
        try:
            return max(1.0, float(self.height()) / 88.0)
        except Exception:
            return 1.0

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

        s = self._ui_scale()
        top = int(round(28 * s))
        h = int(round(34 * s))
        return QRect(int(x1), top, int(max(4.0, x2 - x1)), h)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        s = self._ui_scale()

        p.fillRect(self.rect(), QColor(20, 20, 20))
        track_top = int(round(18 * s))
        track_h = int(round(56 * s))
        track = QRect(self._margin_left, track_top, self.width() - self._margin_left - self._margin_right, track_h)
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
                    p.drawLine(int(x), track.bottom() - int(round(10 * s)), int(x), track.bottom())
                t += minor_s

            p.setPen(QPen(QColor(85, 85, 85)))
            font = p.font()
            font.setPointSize(max(8, int((8 * s) + (self._zoom - 1) * (1.2 * s))))
            p.setFont(font)

            t = 0.0
            while t <= dur_s + 1e-6:
                x = self._margin_left + t * px_per_sec
                p.drawLine(int(x), track.top(), int(x), track.bottom())
                label = self._format_time(int(t * 1000.0))
                p.drawText(int(x) + int(round(4 * s)), track.top() + int(round(12 * s)), label)
                t += major_s

        seg = self._seg_rect()
        p.setPen(QPen(QColor(255, 255, 255, 40)))
        p.setBrush(QColor(70, 130, 255, 170))
        r = max(2, int(round(6 * s)))
        p.drawRoundedRect(seg, r, r)

        p.setPen(QPen(QColor(255, 255, 255, 220)))
        seg_label = f"Selected: {self._format_time(self._seg_start_ms)}  +{self._format_time(self._seg_dur_ms)}"
        p.drawText(seg.x() + int(round(10 * s)), seg.y() + int(round(22 * s)), seg_label)

        if self._duration_ms > 0:
            x = self._time_to_x(self._playhead_ms)
            p.setPen(QPen(QColor(255, 80, 80, 220), 2))
            pad = int(round(6 * s))
            p.drawLine(int(x), track.top() - pad, int(x), track.bottom() + pad)

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
        self.player.setVideoOutput(self.preview.video_item)

        # Big preview toggle (FrameVision integration)
        self._local_player = self.player
        self._main_player = None
        self._main_prev_state = None
        self._main_prev_source = None
        self._main_prev_pos = None
        self._main_prev_rate = None
        self._big_preview = None
        self._big_preview_active = False

        # Preview interactions (click/drag text)
        try:
            self.preview.overlaySelected.connect(self._on_preview_overlay_selected)
            self.preview.overlayMoved.connect(self._on_preview_overlay_moved)
        except Exception:
            pass

        self._tick = QTimer(self)
        self._tick.setInterval(33)
        self._tick.timeout.connect(self._tick_preview)
        self._tick.start()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._save_settings_now)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

                # Top controls (3 rows)
        controls = QVBoxLayout()
        controls.setSpacing(6)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row3 = QHBoxLayout()

        self.btn_open = QPushButton("Open Video…")
        self.btn_export = QPushButton("Export with Text…")
        self.btn_play = QPushButton("Play/Pause")
        self.btn_stop = QPushButton("Stop")

        self.btn_toggle_preview = QToolButton()
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.setText("Hide Preview")

        self.btn_big_preview = QToolButton()
        self.btn_big_preview.setCheckable(True)
        self.btn_big_preview.setChecked(False)
        self.btn_big_preview.setText("Preview on Main Player: OFF")
        self.btn_big_preview.setToolTip(
            "VideoText Preview OFF\n\n"
            "Turn ON to draw your text overlays on top of FrameVision’s main media player preview. "
            "While ON, the Play/Pause/Stop buttons and timeline scrubbing in this tool control the main player.\n\n"
            "Turn OFF to give control back to the app and to avoid double-text when watching exported (baked) videos."
        )

        self.lbl_time = QLabel("00:00.000 / 00:00.000")
        self.lbl_path = QLabel("No video loaded")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Row 1: Open / Play / Stop (+ Export)
        row1.addWidget(self.btn_open, 0)
        row1.addWidget(self.btn_play, 0)
        row1.addWidget(self.btn_stop, 0)
        row1.addWidget(self.btn_export, 0)
        row1.addStretch(1)

        # Row 2: Preview ON/OFF toggle + time (and optional "Hide Preview" for standalone mode)
        row2.addWidget(self.btn_big_preview, 0)
        row2.addSpacing(8)
        row2.addWidget(self.lbl_time, 0)
        row2.addStretch(1)
        row2.addWidget(self.btn_toggle_preview, 0)

        # Row 3: Loaded video path
        row3.addWidget(self.lbl_path, 1)

        controls.addLayout(row1)
        controls.addLayout(row2)
        controls.addLayout(row3)
        root.addLayout(controls)

        # Main area: splitter (preview | editor)
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.StyledPanel)
        vlay = QVBoxLayout(self.video_frame)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.addWidget(self.preview, 1)

        self.panel = QFrame()
        self.panel.setFrameShape(QFrame.StyledPanel)
        panel_lay = QVBoxLayout(self.panel)
        panel_lay.setContentsMargins(10, 10, 10, 10)
        panel_lay.setSpacing(10)

        # Overlay list toolbar
        ov_bar = QHBoxLayout()
        self.btn_add_overlay = QPushButton("Add")
        self.btn_dup_overlay = QPushButton("Duplicate")
        self.btn_del_overlay = QPushButton("Delete")
        self.btn_up_overlay = QPushButton("↑")
        self.btn_down_overlay = QPushButton("↓")
        ov_bar.addWidget(QLabel("Overlays"), 0)
        ov_bar.addStretch(1)
        ov_bar.addWidget(self.btn_add_overlay, 0)
        ov_bar.addWidget(self.btn_dup_overlay, 0)
        ov_bar.addWidget(self.btn_del_overlay, 0)
        ov_bar.addWidget(self.btn_up_overlay, 0)
        ov_bar.addWidget(self.btn_down_overlay, 0)

        self.list_overlays = QListWidget()
        self.list_overlays.setMinimumHeight(120)

        panel_lay.addLayout(ov_bar)
        panel_lay.addWidget(self.list_overlays, 0)

        # Editor widgets (for selected overlay)
        self.edit_text = QLineEdit()
        self.edit_text.setPlaceholderText("Type the overlay text…")

        self.chk_enabled = QCheckBox("Enabled")

        self.combo_font = QFontComboBox()
        self.spin_size = QSpinBox()
        self.spin_size.setRange(6, 300)

        self.btn_color = QToolButton()
        self.btn_color.setText("Color…")

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

        self.spin_fade_in = QDoubleSpinBox()
        self.spin_fade_in.setRange(0.0, 60.0)
        self.spin_fade_in.setDecimals(3)
        self.spin_fade_in.setSingleStep(0.05)

        self.spin_fade_out = QDoubleSpinBox()
        self.spin_fade_out.setRange(0.0, 60.0)
        self.spin_fade_out.setDecimals(3)
        self.spin_fade_out.setSingleStep(0.05)

        self.btn_start_now = QPushButton("Start = current")
        self.btn_end_now = QPushButton("End = current")
        self.btn_seek_start = QPushButton("Seek to start")

        self.chk_preview_always = QCheckBox("Always show overlays in preview")
        self.lbl_active = QLabel("Overlays: 0 active")
        self.lbl_active.setStyleSheet("QLabel { color: #bdbdbd; }")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addRow(QLabel("Text"), self.edit_text)
        form.addRow(QLabel(""), self.chk_enabled)

        font_row = QHBoxLayout()
        font_row.addWidget(self.combo_font, 1)
        font_row.addWidget(QLabel("Size"), 0)
        font_row.addWidget(self.spin_size, 0)
        font_row.addWidget(self.btn_color, 0)
        font_wrap = QWidget()
        font_wrap.setLayout(font_row)
        form.addRow(QLabel("Font"), font_wrap)

        # Effects
        self.combo_effect = QComboBox()
        self.combo_effect.addItem("None", "none")
        self.combo_effect.addItem("Flash", "flash")
        self.combo_effect.addItem("Color swap", "color_swap")

        self.spin_effect = QDoubleSpinBox()
        self.spin_effect.setRange(0.0, 20.0)
        self.spin_effect.setDecimals(2)
        self.spin_effect.setSingleStep(0.25)
        self.spin_effect.setSuffix(" Hz")

        self.btn_color2 = QToolButton()
        self.btn_color2.setText("Alt Color…")

        eff_row = QHBoxLayout()
        eff_row.addWidget(self.combo_effect, 1)
        eff_row.addWidget(QLabel("Speed"), 0)
        eff_row.addWidget(self.spin_effect, 0)
        eff_row.addWidget(self.btn_color2, 0)
        eff_wrap = QWidget()
        eff_wrap.setLayout(eff_row)
        form.addRow(QLabel("Effect"), eff_wrap)

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

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Start (s)"), 0)
        time_row.addWidget(self.spin_start, 0)
        time_row.addSpacing(8)
        time_row.addWidget(QLabel("Duration (s)"), 0)
        time_row.addWidget(self.spin_dur, 0)
        time_row.addStretch(1)
        time_wrap = QWidget()
        time_wrap.setLayout(time_row)
        form.addRow(QLabel("Timing"), time_wrap)

        fade_row = QHBoxLayout()
        fade_row.addWidget(QLabel("Fade in (s)"), 0)
        fade_row.addWidget(self.spin_fade_in, 0)
        fade_row.addSpacing(8)
        fade_row.addWidget(QLabel("Fade out (s)"), 0)
        fade_row.addWidget(self.spin_fade_out, 0)
        fade_row.addStretch(1)
        fade_wrap = QWidget()
        fade_wrap.setLayout(fade_row)
        form.addRow(QLabel("Fades"), fade_wrap)

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

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.video_frame)
        self.splitter.addWidget(self.panel)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        # Bottom: timeline
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

        # wiring (keep existing tool behavior / signals stable)
        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self._stop_current)
        self.btn_export.clicked.connect(self._export_video)
        self.btn_toggle_preview.toggled.connect(self._toggle_preview_pane)
        self.btn_big_preview.toggled.connect(self._toggle_big_preview)

        self._bind_player(self.player)

        # Some backends emit positionChanged sparsely; poll to keep overlays in sync.
        self._preview_poll = QTimer(self)
        self._preview_poll.setInterval(50)  # 20 FPS is enough for fades
        self._preview_poll.timeout.connect(self._tick_preview)
        self._preview_poll.start()

        self.timeline.seekRequested.connect(self._seek_to_ms)
        self.timeline.segmentChanged.connect(self._on_segment_dragged)

        self.slider_zoom.valueChanged.connect(self._on_zoom_slider)

        self.btn_color.clicked.connect(self._pick_color)
        self.btn_color2.clicked.connect(self._pick_color2)
        self.btn_start_now.clicked.connect(self._set_start_to_current)
        self.btn_end_now.clicked.connect(self._set_end_to_current)
        self.btn_seek_start.clicked.connect(self._seek_to_start)

        self.btn_fontfile.clicked.connect(self._pick_fontfile)

        self.btn_add_overlay.clicked.connect(self._add_overlay)
        self.btn_dup_overlay.clicked.connect(self._duplicate_overlay)
        self.btn_del_overlay.clicked.connect(self._delete_overlay)
        self.btn_up_overlay.clicked.connect(lambda: self._move_overlay(-1))
        self.btn_down_overlay.clicked.connect(lambda: self._move_overlay(+1))
        self.list_overlays.currentRowChanged.connect(self._on_overlay_selected)

        self.edit_text.textChanged.connect(self._on_inputs_changed)
        self.chk_enabled.toggled.connect(self._on_inputs_changed)
        self.combo_font.currentFontChanged.connect(self._on_inputs_changed)
        self.spin_size.valueChanged.connect(self._on_inputs_changed)
        self.combo_effect.currentIndexChanged.connect(self._on_inputs_changed)
        self.spin_effect.valueChanged.connect(self._on_inputs_changed)
        self.edit_fontfile.textChanged.connect(self._on_inputs_changed)

        self.combo_anchor.currentIndexChanged.connect(self._on_inputs_changed)
        self.spin_x.valueChanged.connect(self._on_inputs_changed)
        self.spin_y.valueChanged.connect(self._on_inputs_changed)

        self.spin_start.valueChanged.connect(self._on_inputs_changed)
        self.spin_dur.valueChanged.connect(self._on_inputs_changed)
        self.spin_fade_in.valueChanged.connect(self._on_inputs_changed)
        self.spin_fade_out.valueChanged.connect(self._on_inputs_changed)

        self.chk_preview_always.toggled.connect(self._on_inputs_changed)

        # apply
        self._rebuild_overlay_list(select_uid=self._settings.selected_uid)
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._update_zoom_label()
        self._tick_preview()

        # restore preview visibility
        self.btn_toggle_preview.blockSignals(True)
        self.btn_toggle_preview.setChecked(bool(self._settings.preview_hidden))
        self.btn_toggle_preview.blockSignals(False)
        self._apply_preview_hidden()

        if self._settings.last_video_path:
            try:
                p = Path(self._settings.last_video_path)
                if p.exists():
                    self._load_video(str(p))
            except Exception:
                pass

        # FrameVision integration: when embedded, always preview on the main media player
        # (ditch the tab preview pane entirely).
        try:
            QTimer.singleShot(0, self._auto_use_main_player_if_available)
        except Exception:
            pass



    # helpers

    # --- FrameVision integration: always use the main media player for preview (no tab preview pane) ---
    
    def _auto_use_main_player_if_available(self) -> None:
        """
        FrameVision integration (safe mode):
        - Detect the main window + VideoPane
        - Hide the tab preview (we preview on the big player only when the user explicitly enables it)
        - Patch VideoPane._refresh_label_pixmap once (does nothing unless preview is enabled)
        IMPORTANT: This function must NOT take control of the main player automatically.
        """
        if getattr(self, "_fv_checked", False):
            return
        self._fv_checked = True

        main = None
        try:
            main = self._find_framevision_main()
        except Exception:
            main = None
        if main is None:
            return

        vp = getattr(main, "video", None)
        if vp is None or (not hasattr(vp, "player")) or (not hasattr(vp, "label")):
            return

        # Mark embedded environment
        self._fv_main = main
        self._fv_video = vp
        self._fv_use_main_player = True  # backward-compat flag: "embedded detected"
        self._fv_attached = False

        # Keep local player as a harmless fallback (standalone behavior)
        try:
            self._local_player = self._local_player if getattr(self, "_local_player", None) is not None else self.player
        except Exception:
            self._local_player = self.player
        try:
            if self._local_player is not None:
                self._local_player.pause()
        except Exception:
            pass

        # Hide internal preview UI in embedded mode (avoid duplicate previews + empty windows)
        try:
            self.video_frame.hide()
        except Exception:
            pass
        try:
            self.btn_toggle_preview.hide()
        except Exception:
            pass
        try:
            # Editor fills the tab
            self.splitter.setSizes([0, 1])
        except Exception:
            pass

        # Disable playback controls until the user enables preview on the main player
        try:
            self._fv_apply_attach_state(False)
        except Exception:
            pass

        # Patch VideoPane once; overlay painting stays OFF until user enables it.
        try:
            self._patch_fv_video_pane(vp)
        except Exception:
            pass
        try:
            vp._videotext_overlay_provider = None
            vp._videotext_overlay_painter = None
        except Exception:
            pass

        # Stop background refresh timers (they can cause stutter in embedded mode)
        try:
            if getattr(self, "_tick", None) is not None:
                self._tick.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_preview_poll", None) is not None:
                self._preview_poll.stop()
        except Exception:
            pass

        # Ensure button state text
        try:
            self.btn_big_preview.blockSignals(True)
            self.btn_big_preview.setChecked(False)
            self.btn_big_preview.setText("Preview on Main Player: OFF")
            self.btn_big_preview.blockSignals(False)
        except Exception:
            pass

        # In embedded mode, do NOT auto-load last video into the main player.
        # (last_video_path is still used for export, but loading must be explicit.)

    def _fv_watch_player_swap(self) -> None:
        if not getattr(self, "_fv_use_main_player", False):
            return
        vp = getattr(self, "_fv_video", None)
        if vp is None:
            return
        try:
            mp = getattr(vp, "player", None)
        except Exception:
            mp = None
        if mp is None:
            return
        # Rebind when VideoPane recreates its player
        if mp is not getattr(self, "player", None):
            try:
                self._bind_player(mp)
            except Exception:
                pass
        # Keep provider attached while visible
        try:
            if self.isVisible():
                vp._videotext_overlay_provider = self._fv_overlay_provider
                vp._videotext_overlay_painter = self._fv_paint_overlays_on_pixmap
        except Exception:
            pass

    def _request_main_video_refresh(self) -> None:
        vp = getattr(self, "_fv_video", None)
        if vp is None:
            return
        try:
            if hasattr(vp, "_refresh_label_pixmap"):
                vp._refresh_label_pixmap()
        except Exception:
            pass

    @staticmethod
    
    def _fv_anchor_pos(base_rect, text_w: float, text_h: float, anchor: str):
        """
        Return (x,y) for the top-left corner of the text box inside base_rect.
        Anchor values are kept compatible with the VideoText settings.
        """
        a = (anchor or "").strip().lower()

        # normalize common variants
        if a == "top":
            a = "top_center"
        if a == "bottom":
            a = "bottom_center"
        if a == "left":
            a = "center_left"
        if a == "right":
            a = "center_right"

        if a == "top_left":
            return base_rect.left(), base_rect.top()
        if a == "top_center":
            return base_rect.center().x() - text_w / 2.0, base_rect.top()
        if a == "top_right":
            return base_rect.right() - text_w, base_rect.top()

        if a == "center_left":
            return base_rect.left(), base_rect.center().y() - text_h / 2.0
        if a in ("center", "middle"):
            return base_rect.center().x() - text_w / 2.0, base_rect.center().y() - text_h / 2.0
        if a == "center_right":
            return base_rect.right() - text_w, base_rect.center().y() - text_h / 2.0

        if a == "bottom_left":
            return base_rect.left(), base_rect.bottom() - text_h
        if a == "bottom_center":
            return base_rect.center().x() - text_w / 2.0, base_rect.bottom() - text_h
        if a == "bottom_right":
            return base_rect.right() - text_w, base_rect.bottom() - text_h

        # fallback
        return base_rect.center().x() - text_w / 2.0, base_rect.center().y() - text_h / 2.0


    def _fv_overlay_provider(self):
        # Called from the VideoPane draw wrapper; returns list[(TextOverlay, opacity)]
        out = []
        try:
            pos_ms = int(self.player.position())
        except Exception:
            pos_ms = 0
        s = self._settings
        for ov in s.overlays:
            try:
                if not ov.enabled or not (ov.text or "").strip():
                    continue
            except Exception:
                continue

            try:
                if s.preview_always_show:
                    alpha = (ov.color_rgba[3] / 255.0) if ov.color_rgba else 1.0
                    # Apply flash even when "always show" is enabled
                    try:
                        eff = str(getattr(ov, "effect", "none") or "none")
                    except Exception:
                        eff = "none"
                    if eff == "flash":
                        try:
                            hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
                        except Exception:
                            hz = 0.0
                        if hz > 0.001:
                            per = 1000.0 / hz
                            try:
                                phase = ((pos_ms - ov.start_ms) % per) / per
                            except Exception:
                                phase = 0.0
                            if phase >= 0.5:
                                alpha = 0.0
                else:
                    alpha = VideoPreview._alpha_at_ms(ov, pos_ms)
            except Exception:
                alpha = 0.0

            try:
                if alpha <= 0.001:
                    continue
            except Exception:
                continue

            out.append((ov, float(alpha)))
        return out

    
    def _fv_paint_overlays_on_pixmap(self, pm: QPixmap, overlays) -> None:
        """
        Paint overlays directly onto the already-scaled pixmap the VideoPane is showing.

        Performance notes:
        - Avoid forced refresh timers in embedded mode (done elsewhere).
        - Cache font metrics per overlay so each frame doesn't re-measure text.
        """
        if pm is None:
            return
        try:
            if pm.isNull():
                return
        except Exception:
            return
        if not overlays:
            return

        # layout cache: key -> (sig, font, lines, line_h, ascent, text_w, text_h)
        cache = getattr(self, "_fv_layout_cache", None)
        if cache is None:
            cache = {}
            self._fv_layout_cache = cache

        w = float(pm.width())  # logical px
        h = float(pm.height())
        base_rect = QRectF(0.0, 0.0, w, h)
        clamp_rect = base_rect.adjusted(-w, -h, w, h)

        painter = QPainter(pm)
        try:
            painter.setRenderHints(
                QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform, True
            )
        except Exception:
            pass

        try:
            pos_ms = int(self.player.position())
        except Exception:
            pos_ms = 0

        for ov, opacity in overlays:
            try:
                txt = (ov.text or "").strip()
            except Exception:
                txt = ""
            if not txt:
                continue

            # Cache text measurement for this overlay
            try:
                key = getattr(ov, "uid", None) or id(ov)
            except Exception:
                key = id(ov)

            try:
                ff = str(getattr(ov, "font_family", None) or "Arial")
            except Exception:
                ff = "Arial"
            try:
                fs = int(max(1, int(getattr(ov, "font_size", 48) or 48)))
            except Exception:
                fs = 48

            sig = (txt, ff, fs)
            c = cache.get(key, None)
            if c is None or c[0] != sig:
                try:
                    font = QFont(ff)
                except Exception:
                    font = QFont("Arial")
                try:
                    font.setPixelSize(fs)
                except Exception:
                    pass
                try:
                    fm = QFontMetrics(font)
                except Exception:
                    fm = None

                lines = txt.splitlines() or [txt]
                try:
                    line_h = float(fm.height()) if fm is not None else float(fs)
                except Exception:
                    line_h = float(fs)
                try:
                    ascent = float(fm.ascent()) if fm is not None else float(line_h * 0.8)
                except Exception:
                    ascent = float(line_h * 0.8)

                try:
                    max_w = 0.0
                    if fm is not None:
                        for ln in lines:
                            try:
                                max_w = max(max_w, float(fm.horizontalAdvance(ln)))
                            except Exception:
                                pass
                    text_w = max_w
                except Exception:
                    text_w = 0.0

                text_h = line_h * float(len(lines))
                cache[key] = (sig, font, lines, line_h, ascent, text_w, text_h)
                c = cache[key]

            _, font, lines, line_h, ascent, text_w, text_h = c

            try:
                painter.setFont(font)
            except Exception:
                pass

            # Position
            try:
                anchor = str(getattr(ov, "anchor", None) or "top_left")
            except Exception:
                anchor = "top_left"

            if anchor == "custom":
                try:
                    x = float(getattr(ov, "offset_x", 0.0))
                except Exception:
                    x = 0.0
                try:
                    y = float(getattr(ov, "offset_y", 0.0))
                except Exception:
                    y = 0.0
            else:
                ax, ay = self._fv_anchor_pos(base_rect, text_w, text_h, anchor)
                try:
                    x = float(ax) + float(getattr(ov, "offset_x", 0.0))
                except Exception:
                    x = float(ax)
                try:
                    y = float(ay) + float(getattr(ov, "offset_y", 0.0))
                except Exception:
                    y = float(ay)

            # Soft clamp like the tab preview
            try:
                x = float(_clamp(x, clamp_rect.left(), clamp_rect.right()))
                y = float(_clamp(y, clamp_rect.top(), clamp_rect.bottom()))
            except Exception:
                pass

            # Colors (match tab preview behavior: opacity multiplies the stored alpha)
            try:
                rgba = list(getattr(ov, "color_rgba", None) or [255, 255, 255, 255])
            except Exception:
                rgba = [255, 255, 255, 255]
            try:
                r, g, b, a = int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])
            except Exception:
                r, g, b, a = 255, 255, 255, 255

            # Color effects (swap)
            try:
                eff = str(getattr(ov, "effect", "none") or "none")
            except Exception:
                eff = "none"
            if eff == "color_swap":
                try:
                    hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
                except Exception:
                    hz = 0.0
                if hz > 0.001:
                    per = 1000.0 / hz
                    try:
                        phase = ((pos_ms - ov.start_ms) % per) / per
                    except Exception:
                        phase = 0.0
                    if phase >= 0.5:
                        try:
                            c2 = list(getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255])
                        except Exception:
                            c2 = [255, 64, 64, 255]
                        if len(c2) == 4:
                            try:
                                r, g, b = int(c2[0]), int(c2[1]), int(c2[2])
                            except Exception:
                                pass

            try:
                text_a = int(_clamp(a * float(opacity), 0, 255))
            except Exception:
                text_a = int(a)
            try:
                shadow_a = int(_clamp(170 * float(opacity), 0, 255))
            except Exception:
                shadow_a = 170

            text_col = QColor(r, g, b, int(text_a))
            sh_col = QColor(0, 0, 0, int(shadow_a))

            # Draw shadow then text (baseline-based)
            try:
                painter.setPen(sh_col)
                for i, ln in enumerate(lines):
                    painter.drawText(QPointF(x + 2.0, y + 2.0 + ascent + (float(i) * line_h)), ln)
            except Exception:
                pass

            try:
                painter.setPen(text_col)
                for i, ln in enumerate(lines):
                    painter.drawText(QPointF(x, y + ascent + (float(i) * line_h)), ln)
            except Exception:
                pass

        try:
            painter.end()
        except Exception:
            pass

    def _patch_fv_video_pane(self, vp) -> None:
        # Patch only once per VideoPane instance.
        if getattr(vp, "_videotext_overlay_patch_applied", False):
            return
        vp._videotext_overlay_patch_applied = True

        # Ensure attributes exist even when provider is disabled
        try:
            vp._videotext_overlay_provider = None
            vp._videotext_overlay_painter = None
        except Exception:
            pass

        try:
            orig = vp._refresh_label_pixmap
        except Exception:
            return

        def _wrapped_refresh():
            # Call original first (it sets the label pixmap from the current frame)
            try:
                orig()
            except Exception:
                pass

            try:
                # Only paint in normal video mode (skip compare mode)
                if getattr(vp, "_mode", None) != "video":
                    return
                if getattr(vp, "_compare_active", False):
                    return

                prov = getattr(vp, "_videotext_overlay_provider", None)
                paint_fn = getattr(vp, "_videotext_overlay_painter", None)
                if prov is None or paint_fn is None:
                    return

                overlays = prov()
                if not overlays:
                    return

                pm0 = None
                try:
                    pm0 = vp.label.pixmap()
                except Exception:
                    pm0 = None
                if pm0 is None:
                    return
                try:
                    if pm0.isNull():
                        return
                except Exception:
                    return

                pm = QPixmap(pm0)
                try:
                    paint_fn(pm, overlays)
                except Exception:
                    pass
                try:
                    vp.label.setPixmap(pm)
                except Exception:
                    pass
            except Exception:
                pass

        try:
            vp._videotext_orig_refresh_label_pixmap = orig
        except Exception:
            pass
        try:
            vp._refresh_label_pixmap = _wrapped_refresh
        except Exception:
            pass

    
    def showEvent(self, e):
        try:
            super().showEvent(e)
        except Exception:
            pass

        # In embedded mode, only attach overlay preview when the user explicitly enabled it.
        if getattr(self, "_fv_use_main_player", False) and getattr(self, "_fv_attached", False):
            vp = getattr(self, "_fv_video", None)
            if vp is not None:
                try:
                    vp._videotext_overlay_provider = self._fv_overlay_provider
                    vp._videotext_overlay_painter = self._fv_paint_overlays_on_pixmap
                except Exception:
                    pass
            try:
                self._request_main_video_refresh()
            except Exception:
                pass

    def hideEvent(self, e):
        # Always detach when leaving the tab (so VideoText never keeps controlling the app).
        if getattr(self, "_fv_use_main_player", False) and getattr(self, "_fv_attached", False):
            try:
                self.btn_big_preview.blockSignals(True)
                self.btn_big_preview.setChecked(False)
                self.btn_big_preview.blockSignals(False)
            except Exception:
                pass
            try:
                self._toggle_big_preview(False)
            except Exception:
                pass

        try:
            super().hideEvent(e)
        except Exception:
            pass

    def _current_overlay(self) -> TextOverlay:
        return self._settings.get_selected()

    def _index_for_uid(self, uid: str) -> int:
        for i, ov in enumerate(self._settings.overlays):
            if ov.uid == uid:
                return i
        return -1

    def _rebuild_overlay_list(self, select_uid: str | None = None) -> None:
        self.list_overlays.blockSignals(True)
        self.list_overlays.clear()

        for i, ov in enumerate(self._settings.overlays):
            label = self._overlay_label(ov, index=i)
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, ov.uid)
            self.list_overlays.addItem(it)

        # select
        if select_uid:
            idx = self._index_for_uid(select_uid)
            if idx >= 0:
                self.list_overlays.setCurrentRow(idx)
                self._settings.selected_uid = select_uid
            else:
                self.list_overlays.setCurrentRow(0)
                self._settings.selected_uid = self._settings.overlays[0].uid
        else:
            self.list_overlays.setCurrentRow(0)
            self._settings.selected_uid = self._settings.overlays[0].uid

        self.list_overlays.blockSignals(False)

    @staticmethod
    def _overlay_label(ov: TextOverlay, index: int) -> str:
        txt = (ov.text or "").strip()
        if len(txt) > 28:
            txt = txt[:28] + "…"
        st = TimelineWidget._format_time(ov.start_ms)
        en = TimelineWidget._format_time(ov.start_ms + ov.duration_ms)
        prefix = "✓" if ov.enabled else "×"
        return f"{prefix} {index + 1:02d}  [{st} – {en}]  {txt or '(empty)'}"

    def _refresh_overlay_list_labels(self) -> None:
        for row in range(self.list_overlays.count()):
            it = self.list_overlays.item(row)
            uid = it.data(Qt.UserRole)
            ov = next((o for o in self._settings.overlays if o.uid == uid), None)
            if ov:
                it.setText(self._overlay_label(ov, row))


    # --- Big preview integration (FrameVision) ----------------------------------
    def _stop_current(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass

    
    def _bind_player(self, p: QMediaPlayer) -> None:
        """
        (Re)bind time signals to the given player and make it the active player for this pane.

        Important: PySide can emit RuntimeWarning if you call disconnect() on a slot that was never connected.
        We avoid that by tracking connection state ourselves.
        """
        if p is None:
            return

        prev = getattr(self, "_bound_player", None)

        # Disconnect previous player if we were actually connected
        try:
            if prev is not None and prev is not p:
                if getattr(self, "_bound_pos_connected", False):
                    try:
                        prev.positionChanged.disconnect(self._on_position_changed)
                    except Exception:
                        pass
                if getattr(self, "_bound_dur_connected", False):
                    try:
                        prev.durationChanged.disconnect(self._on_duration_changed)
                    except Exception:
                        pass
        except Exception:
            pass

        # Set new player
        self.player = p
        self._bound_player = p
        self._bound_pos_connected = False
        self._bound_dur_connected = False

        # Connect signals (prefer UniqueConnection)
        try:
            try:
                p.positionChanged.connect(self._on_position_changed, Qt.UniqueConnection)
            except Exception:
                p.positionChanged.connect(self._on_position_changed)
            self._bound_pos_connected = True
        except Exception:
            self._bound_pos_connected = False

        try:
            try:
                p.durationChanged.connect(self._on_duration_changed, Qt.UniqueConnection)
            except Exception:
                p.durationChanged.connect(self._on_duration_changed)
            self._bound_dur_connected = True
        except Exception:
            self._bound_dur_connected = False

        # Sync UI immediately
        try:
            self._on_duration_changed(int(p.duration()))
        except Exception:
            pass
        try:
            self._on_position_changed(int(p.position()))
        except Exception:
            pass


    def _find_framevision_main(self):
        try:
            w = self.window()
        except Exception:
            w = None
        if w is None:
            return None
        # FrameVision main window convention: has .video (VideoPane) and the override helpers we add.
        if hasattr(w, "video") and hasattr(w, "set_left_override_widget"):
            return w
        return None

    def _ensure_big_preview(self):
        if getattr(self, "_big_preview", None) is not None:
            return self._big_preview
        try:
            bp = VideoPreview(None)
            try:
                bp.set_settings(self._settings)
            except Exception:
                pass
            try:
                bp.overlaySelected.connect(self._on_preview_overlay_selected)
                bp.overlayMoved.connect(self._on_preview_overlay_moved)
            except Exception:
                pass
            self._big_preview = bp
        except Exception:
            self._big_preview = None
        return self._big_preview

    
    def _fv_apply_attach_state(self, attached: bool) -> None:
        """
        Embedded-mode UX:
        - When detached: do not control the main player; keep playback controls disabled
        - When attached: this tool controls the main player (play/pause/seek) and draws overlays on it
        """
        try:
            self.btn_play.setEnabled(bool(attached))
        except Exception:
            pass
        try:
            self.btn_stop.setEnabled(bool(attached))
        except Exception:
            pass
        try:
            self.timeline.setEnabled(bool(attached))
        except Exception:
            pass
        try:
            self.slider_zoom.setEnabled(bool(attached))
        except Exception:
            pass

        # Make it very obvious what's happening
        try:
            if attached:
                self.btn_big_preview.setText("Preview on Main Player: ON")
            else:
                self.btn_big_preview.setText("Preview on Main Player: OFF")
        except Exception:
            pass

        # Helpful tooltip on Open Video in embedded mode
        try:
            if getattr(self, "_fv_use_main_player", False):
                if attached:
                    self.btn_open.setToolTip(
                        "Open Video…\n\n"
                        "Loads the selected video into FrameVision’s main media player and previews your overlays on top."
                    )
                else:
                    self.btn_open.setToolTip(
                        "Open Video…\n\n"
                        "Selects a video for export.\n"
                        "Tip: Turn ON “Preview on Main Player” if you want this tool to load/seek/play the main player."
                    )
        except Exception:
            pass

    def _toggle_big_preview(self, checked: bool) -> None:
        """
        Repurposed: this is now the safe ON/OFF switch for VideoText preview on the main player.
        OFF (default): VideoText does not touch the main player.
        ON: VideoText draws overlays on the main player and routes play/seek controls to it.
        """
        main = self._find_framevision_main()
        if main is None:
            # Standalone mode: this toggle has no meaning
            try:
                self.btn_big_preview.blockSignals(True)
                self.btn_big_preview.setChecked(False)
                self.btn_big_preview.setText("Preview on Main Player: OFF")
                self.btn_big_preview.blockSignals(False)
            except Exception:
                pass
            return

        # Ensure we ran embedded detection
        try:
            self._auto_use_main_player_if_available()
        except Exception:
            pass

        vp = getattr(self, "_fv_video", None)
        if vp is None or not hasattr(vp, "player"):
            try:
                self.btn_big_preview.blockSignals(True)
                self.btn_big_preview.setChecked(False)
                self.btn_big_preview.setText("Preview on Main Player: OFF")
                self.btn_big_preview.blockSignals(False)
            except Exception:
                pass
            return

        if not checked:
            # Detach
            try:
                vp._videotext_overlay_provider = None
                vp._videotext_overlay_painter = None
            except Exception:
                pass

            try:
                self._fv_attached = False
            except Exception:
                pass

            # Rebind tool controls back to local player so we don't hijack the app
            try:
                if getattr(self, "_local_player", None) is not None:
                    self._bind_player(self._local_player)
            except Exception:
                pass

            try:
                self._fv_apply_attach_state(False)
            except Exception:
                pass

            # Force a redraw to remove any last painted overlay from the current frame
            try:
                self._request_main_video_refresh()
            except Exception:
                pass
            return

        # Attach
        try:
            self._fv_attached = True
        except Exception:
            pass

        # Route time/seek controls to the main player
        try:
            self._bind_player(vp.player)
        except Exception:
            pass

        # Enable overlay painting (only while enabled)
        try:
            vp._videotext_overlay_provider = self._fv_overlay_provider
            vp._videotext_overlay_painter = self._fv_paint_overlays_on_pixmap
        except Exception:
            pass

        try:
            self._fv_apply_attach_state(True)
        except Exception:
            pass

        # If the user already selected a last video for this tool and the main player is empty,
        # we do NOT auto-load it (to avoid hijacking). They can click Open Video… to load explicitly.
        try:
            self._request_main_video_refresh()
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
        # Always remember the path (used for export in both standalone and embedded mode)
        self._settings.last_video_path = path
        try:
            self.lbl_path.setText(path)
        except Exception:
            pass
        self._schedule_save()

        embedded = bool(getattr(self, "_fv_use_main_player", False) and getattr(self, "_fv_video", None) is not None)

        # Embedded mode: only touch the main player when the user enabled preview.
        if embedded:
            if getattr(self, "_fv_attached", False):
                try:
                    vp = self._fv_video
                    try:
                        vp.open(Path(path))
                    except Exception:
                        # fallback: try to set source directly
                        try:
                            vp.player.setSource(QUrl.fromLocalFile(path))
                        except Exception:
                            pass
                    try:
                        self._bind_player(vp.player)
                    except Exception:
                        pass
                    try:
                        vp._videotext_overlay_provider = self._fv_overlay_provider
                        vp._videotext_overlay_painter = self._fv_paint_overlays_on_pixmap
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    self._request_main_video_refresh()
                except Exception:
                    pass
            else:
                # Preview is OFF; do not hijack the app by loading anything.
                try:
                    self.lbl_time.setText("Preview OFF (enable to control main player)")
                except Exception:
                    pass
            return

        # Standalone / non-embedded: use our local player & tab preview.
        try:
            self.player.setSource(QUrl.fromLocalFile(path))
        except Exception:
            pass



    def _toggle_play_pause(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _seek_to_ms(self, ms: int) -> None:
        self.player.setPosition(int(ms))

    def _seek_to_start(self) -> None:
        ov = self._current_overlay()
        self.player.setPosition(int(ov.start_ms))

    # overlays list actions
    def _add_overlay(self) -> None:
        ov = TextOverlay()
        ov.text = "New text"
        ov.start_ms = int(self.player.position())
        ov.duration_ms = 2500
        ov.fade_in_ms = 250
        ov.fade_out_ms = 250
        dur = int(self.player.duration())
        ov.normalize(video_duration_ms=dur if dur > 0 else None)

        self._settings.overlays.append(ov)
        self._settings.selected_uid = ov.uid

        self._rebuild_overlay_list(select_uid=ov.uid)
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _duplicate_overlay(self) -> None:
        src = self._current_overlay()
        new = TextOverlay()
        # copy fields
        new.text = src.text
        new.font_family = src.font_family
        new.font_size = src.font_size
        new.color_rgba = list(src.color_rgba) if src.color_rgba else [255, 255, 255, 255]
        new.font_file = src.font_file
        new.anchor = src.anchor
        new.offset_x = src.offset_x
        new.offset_y = src.offset_y
        new.start_ms = src.start_ms
        new.duration_ms = src.duration_ms
        new.fade_in_ms = src.fade_in_ms
        new.fade_out_ms = src.fade_out_ms
        new.enabled = src.enabled
        new.effect = getattr(src, "effect", "none")
        new.effect_hz = float(getattr(src, "effect_hz", 2.0) or 0.0)
        new.effect_color2_rgba = list(getattr(src, "effect_color2_rgba", None) or [255, 64, 64, 255])

        self._settings.overlays.append(new)
        self._settings.selected_uid = new.uid

        self._rebuild_overlay_list(select_uid=new.uid)
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _delete_overlay(self) -> None:
        if len(self._settings.overlays) <= 1:
            QMessageBox.information(self, "Delete overlay", "You need at least one overlay.")
            return
        uid = self._settings.selected_uid
        idx = self._index_for_uid(uid)
        if idx < 0:
            return
        self._settings.overlays.pop(idx)

        # select nearest
        idx = min(idx, len(self._settings.overlays) - 1)
        self._settings.selected_uid = self._settings.overlays[idx].uid

        self._rebuild_overlay_list(select_uid=self._settings.selected_uid)
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _move_overlay(self, delta: int) -> None:
        uid = self._settings.selected_uid
        idx = self._index_for_uid(uid)
        if idx < 0:
            return
        new_idx = idx + int(delta)
        if new_idx < 0 or new_idx >= len(self._settings.overlays):
            return
        ov = self._settings.overlays.pop(idx)
        self._settings.overlays.insert(new_idx, ov)
        self._rebuild_overlay_list(select_uid=uid)
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _on_overlay_selected(self, row: int) -> None:
        if row < 0 or row >= self.list_overlays.count():
            return
        it = self.list_overlays.item(row)
        uid = it.data(Qt.UserRole)
        if not uid:
            return
        self._settings.selected_uid = str(uid)
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()


    def _on_preview_overlay_selected(self, uid: str) -> None:
        uid = str(uid or "")
        if not uid:
            return
        self._settings.selected_uid = uid
        self._apply_settings_to_ui()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _on_preview_overlay_moved(self, uid: str, ox: int, oy: int) -> None:
        uid = str(uid or "")
        if not uid:
            return

        # If user dragged a non-selected overlay, select it (once).
        if uid != self._settings.selected_uid:
            self._settings.selected_uid = uid
            self._apply_settings_to_ui()
            self._apply_settings_to_preview()

        ov = self._current_overlay()
        ov.offset_x = int(ox)
        ov.offset_y = int(oy)

        # Update X/Y boxes without re-triggering full input sync loops.
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_x.setValue(int(ox))
        self.spin_y.setValue(int(oy))
        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)

        self._schedule_save()
        self._emit_settings()

    # segment/zoom
    def _on_segment_dragged(self, start_ms: int, dur_ms: int) -> None:
        ov = self._current_overlay()
        ov.start_ms = int(start_ms)
        ov.duration_ms = int(dur_ms)
        dur = int(self.player.duration())
        ov.normalize(video_duration_ms=dur if dur > 0 else None)
        self._sync_timing_to_spins()
        self.timeline.set_segment(ov.start_ms, ov.duration_ms)
        self.player.setPosition(int(ov.start_ms))
        self._refresh_overlay_list_labels()
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
        ov = self._current_overlay()
        cur = QColor(*ov.color_rgba)
        picked = QColorDialog.getColor(cur, self, "Pick text color")
        if not picked.isValid():
            return
        ov.color_rgba = [picked.red(), picked.green(), picked.blue(), picked.alpha()]
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _pick_color2(self) -> None:
        ov = self._current_overlay()
        try:
            cur_rgba = list(getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255])
        except Exception:
            cur_rgba = [255, 64, 64, 255]
        if len(cur_rgba) != 4:
            cur_rgba = [255, 64, 64, 255]

        cur = QColor(*cur_rgba)
        picked = QColorDialog.getColor(cur, self, "Pick alternate text color")
        if not picked.isValid():
            return
        ov.effect_color2_rgba = [picked.red(), picked.green(), picked.blue(), picked.alpha()]
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
        ov = self._current_overlay()
        ov.start_ms = int(self.player.position())
        dur = int(self.player.duration())
        ov.normalize(video_duration_ms=dur if dur > 0 else None)
        self._sync_timing_to_spins()
        self.timeline.set_segment(ov.start_ms, ov.duration_ms)
        self._refresh_overlay_list_labels()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    def _set_end_to_current(self) -> None:
        ov = self._current_overlay()
        end = int(self.player.position())
        start = int(ov.start_ms)
        if end <= start:
            ov.duration_ms = 50
        else:
            ov.duration_ms = max(1, end - start)

        dur = int(self.player.duration())
        ov.normalize(video_duration_ms=dur if dur > 0 else None)

        self._sync_timing_to_spins()
        self.timeline.set_segment(ov.start_ms, ov.duration_ms)
        self._refresh_overlay_list_labels()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    # preview pane
    def _toggle_preview_pane(self, checked: bool) -> None:
        self._settings.preview_hidden = bool(checked)
        self._apply_preview_hidden()
        self._schedule_save()
        self._emit_settings()

    def _apply_preview_hidden(self) -> None:
        hidden = bool(self._settings.preview_hidden)
        self.video_frame.setVisible(not hidden)
        self.btn_toggle_preview.setText("Show Preview" if hidden else "Hide Preview")

    # player callbacks
    def _on_position_changed(self, pos: int) -> None:
        dur = int(self.player.duration())
        self.lbl_time.setText(f"{TimelineWidget._format_time(pos)} / {TimelineWidget._format_time(dur)}")
        self.timeline.set_playhead_ms(pos)
        self._tick_preview()

    def _on_duration_changed(self, dur: int) -> None:
        self.timeline.set_duration_ms(dur)

        # normalize overlays against duration
        for ov in self._settings.overlays:
            ov.normalize(video_duration_ms=dur if dur > 0 else None)

        ov = self._current_overlay()
        self.timeline.set_segment(ov.start_ms, ov.duration_ms)
        self._sync_timing_to_spins()
        self._refresh_overlay_list_labels()
        self._apply_settings_to_preview()
        self._schedule_save()
        self._emit_settings()

    
    def _tick_preview(self) -> None:
        pos = int(self.player.position())

        # Standalone preview updates (tab preview and optional big preview widget).
        if not getattr(self, "_fv_use_main_player", False):
            try:
                self.preview.update_for_time(pos)
            except Exception:
                pass
            try:
                if self._big_preview is not None:
                    try:
                        self._big_preview.update_for_time(pos)
                    except Exception:
                        pass
            except Exception:
                pass

        # Embedded mode: do NOT force-refresh the main player every tick.
        # The main VideoPane will refresh on its own frames; overlays are painted there.
        # We only request refresh on user edits / attach-detach.

        if self._settings.preview_always_show:
            active = sum(1 for ov in self._settings.overlays if ov.enabled and (ov.text or "").strip())
        else:
            active = 0
            for ov in self._settings.overlays:
                if not ov.enabled or not (ov.text or "").strip():
                    continue
                if ov.start_ms <= pos <= (ov.start_ms + ov.duration_ms):
                    active += 1
        self.lbl_active.setText(f"Overlays: {active} active")


    # settings io
    def _load_settings(self) -> VideoTextSettings:
        path = _settings_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))

                # Backward compatibility: old single-overlay schema
                if isinstance(data, dict) and "overlays" not in data:
                    ov = TextOverlay()
                    ov.text = str(data.get("text", ov.text))
                    ov.font_family = str(data.get("font_family", ov.font_family))
                    ov.font_size = int(data.get("font_size", ov.font_size))
                    col = data.get("color_rgba", None)
                    if isinstance(col, list) and len(col) == 4:
                        ov.color_rgba = [int(col[0]), int(col[1]), int(col[2]), int(col[3])]
                    ov.font_file = str(data.get("font_file", data.get("fontfile", "")) or "")
                    ov.anchor = str(data.get("anchor", ov.anchor))
                    ov.offset_x = int(data.get("offset_x", ov.offset_x))
                    ov.offset_y = int(data.get("offset_y", ov.offset_y))
                    ov.start_ms = int(data.get("start_ms", ov.start_ms))
                    ov.duration_ms = int(data.get("duration_ms", ov.duration_ms))
                    # Older versions had no fades; keep defaults.
                    s = VideoTextSettings(
                        overlays=[ov],
                        selected_uid=ov.uid,
                        zoom=float(data.get("zoom", 1.0)),
                        last_video_path=str(data.get("last_video_path", "")),
                        preview_always_show=_to_bool(data.get("preview_always_show", False), default=False),
                        preview_hidden=_to_bool(data.get("preview_hidden", False), default=False),
                    )
                    return s

                # New schema
                s = VideoTextSettings()
                if isinstance(data, dict):
                    # overlays
                    ovs = []
                    raw_ovs = data.get("overlays", [])
                    if isinstance(raw_ovs, list):
                        for raw in raw_ovs:
                            if not isinstance(raw, dict):
                                continue
                            ov = TextOverlay(uid=str(raw.get("uid", _new_uid())))
                            ov.text = str(raw.get("text", ov.text))
                            ov.font_family = str(raw.get("font_family", ov.font_family))
                            ov.font_size = int(raw.get("font_size", ov.font_size))
                            col = raw.get("color_rgba", None)
                            if isinstance(col, list) and len(col) == 4:
                                ov.color_rgba = [int(col[0]), int(col[1]), int(col[2]), int(col[3])]
                            ov.font_file = str(raw.get("font_file", "") or "")
                            ov.anchor = str(raw.get("anchor", ov.anchor))
                            ov.offset_x = int(raw.get("offset_x", ov.offset_x))
                            ov.offset_y = int(raw.get("offset_y", ov.offset_y))
                            ov.start_ms = int(raw.get("start_ms", ov.start_ms))
                            ov.duration_ms = int(raw.get("duration_ms", ov.duration_ms))
                            ov.fade_in_ms = int(raw.get("fade_in_ms", ov.fade_in_ms))
                            ov.fade_out_ms = int(raw.get("fade_out_ms", ov.fade_out_ms))
                            ov.enabled = _to_bool(raw.get("enabled", True), default=True)
                            ov.effect = str(raw.get("effect", raw.get("effect_mode", getattr(ov, "effect", "none"))) or "none")
                            try:
                                ov.effect_hz = float(raw.get("effect_hz", getattr(ov, "effect_hz", 2.0)) or 0.0)
                            except Exception:
                                ov.effect_hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
                            try:
                                ov.effect_color2_rgba = list(raw.get("effect_color2_rgba", getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255]))
                            except Exception:
                                ov.effect_color2_rgba = list(getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255])
                            ovs.append(ov)

                    if not ovs:
                        ovs = [TextOverlay()]

                    s = VideoTextSettings(
                        overlays=ovs,
                        selected_uid=str(data.get("selected_uid", "")),
                        zoom=float(data.get("zoom", 1.0)),
                        last_video_path=str(data.get("last_video_path", "")),
                        preview_always_show=_to_bool(data.get("preview_always_show", False), default=False),
                        preview_hidden=_to_bool(data.get("preview_hidden", False), default=False),
                    )
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
            # Normalize before saving
            dur = int(self.player.duration())
            for ov in self._settings.overlays:
                ov.normalize(video_duration_ms=dur if dur > 0 else None)

            out = asdict(self._settings)
            path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        except Exception:
            pass

    # apply/sync
    def _apply_settings_to_ui(self) -> None:
        ov = self._current_overlay()

        # selected row should track selected_uid
        idx = self._index_for_uid(self._settings.selected_uid)
        if idx >= 0 and idx != self.list_overlays.currentRow():
            self.list_overlays.blockSignals(True)
            self.list_overlays.setCurrentRow(idx)
            self.list_overlays.blockSignals(False)

        self.edit_text.blockSignals(True)
        self.chk_enabled.blockSignals(True)
        self.combo_font.blockSignals(True)
        self.spin_size.blockSignals(True)
        self.combo_effect.blockSignals(True)
        self.spin_effect.blockSignals(True)
        self.edit_fontfile.blockSignals(True)
        self.combo_anchor.blockSignals(True)
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_start.blockSignals(True)
        self.spin_dur.blockSignals(True)
        self.spin_fade_in.blockSignals(True)
        self.spin_fade_out.blockSignals(True)
        self.chk_preview_always.blockSignals(True)

        self.edit_text.setText(ov.text)
        self.chk_enabled.setChecked(bool(ov.enabled))
        self.combo_font.setCurrentFont(QFont(ov.font_family))
        self.spin_size.setValue(int(ov.font_size))

        # Effects
        eff = str(getattr(ov, "effect", "none") or "none")
        idxe = 0
        for i in range(self.combo_effect.count()):
            if self.combo_effect.itemData(i) == eff:
                idxe = i
                break
        self.combo_effect.setCurrentIndex(idxe)
        try:
            self.spin_effect.setValue(float(getattr(ov, "effect_hz", 2.0) or 0.0))
        except Exception:
            self.spin_effect.setValue(0.0)
        try:
            self.btn_color2.setEnabled(eff == "color_swap")
        except Exception:
            pass

        self.edit_fontfile.setText(ov.font_file or "")

        idxa = 0
        for i in range(self.combo_anchor.count()):
            if self.combo_anchor.itemData(i) == ov.anchor:
                idxa = i
                break
        self.combo_anchor.setCurrentIndex(idxa)

        self.spin_x.setValue(int(ov.offset_x))
        self.spin_y.setValue(int(ov.offset_y))

        self.spin_start.setValue(ov.start_ms / 1000.0)
        self.spin_dur.setValue(ov.duration_ms / 1000.0)
        self.spin_fade_in.setValue(ov.fade_in_ms / 1000.0)
        self.spin_fade_out.setValue(ov.fade_out_ms / 1000.0)

        self.chk_preview_always.setChecked(bool(self._settings.preview_always_show))

        self.edit_text.blockSignals(False)
        self.chk_enabled.blockSignals(False)
        self.combo_font.blockSignals(False)
        self.spin_size.blockSignals(False)
        self.combo_effect.blockSignals(False)
        self.spin_effect.blockSignals(False)
        self.edit_fontfile.blockSignals(False)
        self.combo_anchor.blockSignals(False)
        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)
        self.spin_start.blockSignals(False)
        self.spin_dur.blockSignals(False)
        self.spin_fade_in.blockSignals(False)
        self.spin_fade_out.blockSignals(False)
        self.chk_preview_always.blockSignals(False)

        self.timeline.set_zoom(self._settings.zoom)
        self.timeline.set_segment(ov.start_ms, ov.duration_ms)

        if self._settings.last_video_path:
            self.lbl_path.setText(self._settings.last_video_path)

        self._update_zoom_label()
        self._refresh_overlay_list_labels()

    def _sync_timing_to_spins(self) -> None:
        ov = self._current_overlay()
        self.spin_start.blockSignals(True)
        self.spin_dur.blockSignals(True)
        self.spin_fade_in.blockSignals(True)
        self.spin_fade_out.blockSignals(True)
        self.spin_start.setValue(ov.start_ms / 1000.0)
        self.spin_dur.setValue(ov.duration_ms / 1000.0)
        self.spin_fade_in.setValue(ov.fade_in_ms / 1000.0)
        self.spin_fade_out.setValue(ov.fade_out_ms / 1000.0)
        self.spin_start.blockSignals(False)
        self.spin_dur.blockSignals(False)
        self.spin_fade_in.blockSignals(False)
        self.spin_fade_out.blockSignals(False)

    def _apply_settings_to_preview(self) -> None:
        try:
            self.preview.set_settings(self._settings)
        except Exception:
            pass
        try:
            if self._big_preview is not None:
                self._big_preview.set_settings(self._settings)
        except Exception:
            pass
        self._tick_preview()

    def _emit_settings(self) -> None:
        try:
            self.settingsChanged.emit(asdict(self._settings))
        except Exception:
            pass

    def _on_inputs_changed(self, *args) -> None:
        ov = self._current_overlay()

        # Ensure typed values are committed even if a spinbox still has focus.
        for w in (
            self.spin_start, self.spin_dur, self.spin_fade_in, self.spin_fade_out,
            self.spin_x, self.spin_y, self.spin_size,
        ):
            try:
                w.interpretText()
            except Exception:
                pass

        ov.text = self.edit_text.text()
        ov.enabled = bool(self.chk_enabled.isChecked())
        ov.font_family = self.combo_font.currentFont().family()
        ov.font_size = int(self.spin_size.value())

        # Effects
        try:
            ov.effect = str(self.combo_effect.currentData() or "none")
        except Exception:
            ov.effect = "none"
        try:
            ov.effect_hz = float(self.spin_effect.value())
        except Exception:
            ov.effect_hz = 0.0
        try:
            self.btn_color2.setEnabled(str(getattr(ov, "effect", "none") or "none") == "color_swap")
        except Exception:
            pass

        ov.font_file = self.edit_fontfile.text().strip()

        ov.anchor = str(self.combo_anchor.currentData())
        ov.offset_x = int(self.spin_x.value())
        ov.offset_y = int(self.spin_y.value())

        ov.start_ms = int(self.spin_start.value() * 1000.0)
        ov.duration_ms = int(max(1, self.spin_dur.value() * 1000.0))
        ov.fade_in_ms = int(max(0, self.spin_fade_in.value() * 1000.0))
        ov.fade_out_ms = int(max(0, self.spin_fade_out.value() * 1000.0))

        self._settings.preview_always_show = bool(self.chk_preview_always.isChecked())

        dur = int(self.player.duration())
        ov.normalize(video_duration_ms=dur if dur > 0 else None)

        self.timeline.set_segment(ov.start_ms, ov.duration_ms)
        self._refresh_overlay_list_labels()
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
            QMessageBox.warning(self, "Export", "ffmpeg not found. Expected in /presets/bin/ or on PATH.")
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
        dur = int(self.player.duration())
        missing_fonts: List[str] = []
        parts: List[str] = []

        for i, ov in enumerate(self._settings.overlays):
            if not ov.enabled:
                continue
            if not (ov.text or "").strip():
                continue

            ov.normalize(video_duration_ms=dur if dur > 0 else None)

            filt, warn = self._build_drawtext_for_overlay(ov)
            if warn == "MISSING_FONT":
                missing_fonts.append(f"Overlay {i + 1}")
                continue
            if warn:
                return "", warn
            if filt:
                parts.append(filt)

        if not parts:
            return "", "Nothing to export: all overlays are empty or disabled."

        if missing_fonts:
            return "", "Pick a font file for: " + ", ".join(missing_fonts) + "\n(Font file → Browse…)."

        # Chain drawtext filters
        return ",".join(parts), None

    def _build_drawtext_for_overlay(self, ov: TextOverlay) -> Tuple[str, Optional[str]]:
        if not (ov.text or "").strip():
            return "", "Text is empty."

        fontfile = (ov.font_file or "").strip()
        if not fontfile and os.name == "nt":
            guess = _guess_windows_fontfile(ov.font_family)
            if guess:
                fontfile = guess

        if not fontfile:
            return "", "MISSING_FONT"

        start_s = _ms_to_s(ov.start_ms)
        end_s = _ms_to_s(ov.start_ms + ov.duration_ms)
        if end_s <= start_s:
            end_s = start_s + 0.05

        fi_s = _ms_to_s(ov.fade_in_ms)
        fo_s = _ms_to_s(ov.fade_out_ms)
        # Keep them sane
        max_dur = max(0.05, end_s - start_s)
        fi_s = float(_clamp(fi_s, 0.0, max_dur))
        fo_s = float(_clamp(fo_s, 0.0, max_dur))
        if fi_s + fo_s > max_dur:
            if max_dur <= 1e-6:
                fi_s = 0.0
                fo_s = 0.0
            else:
                scale = max_dur / (fi_s + fo_s)
                fi_s *= scale
                fo_s *= scale

        ox = int(ov.offset_x)
        oy = int(ov.offset_y)

        if ov.anchor == "custom":
            x_expr = f"{ox}"
            y_expr = f"{oy}"
        else:
            x_expr, y_expr = self._anchor_to_drawtext_xy(ov.anchor, ox, oy)

        txt = _ff_escape_value(ov.text)

        r, g, b, a = ov.color_rgba if (ov.color_rgba and len(ov.color_rgba) == 4) else (255, 255, 255, 255)
        base_alpha = float(_clamp(a / 255.0, 0.0, 1.0))
        base_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        # Effects
        try:
            eff = str(getattr(ov, "effect", "none") or "none")
        except Exception:
            eff = "none"
        try:
            hz = float(getattr(ov, "effect_hz", 2.0) or 0.0)
        except Exception:
            hz = 0.0


        alpha_expr = self._fade_alpha_expr(start_s, end_s, fi_s, fo_s, base_alpha)

        parts = ["drawtext="]
        ff = _ff_escape_value(fontfile)
        parts.append(f"fontfile='{ff}':")
        parts.append(f"text='{txt}':")
        parts.append(f"fontcolor={base_color}:")
        parts.append(f"fontsize={int(ov.font_size)}:")
        parts.append(f"x={x_expr}:")
        parts.append(f"y={y_expr}:")
        enable_base = f"between(t,{start_s:.6f},{end_s:.6f})"
        enable_expr = enable_base
        if hz > 0.001 and eff in ("flash", "color_swap"):
            per = 1.0 / hz
            half = per / 2.0
            mod_expr = f"mod(t-{start_s:.6f},{per:.6f})"
            if eff == "flash":
                enable_expr = f"{enable_base}*lt({mod_expr},{half:.6f})"
        parts.append(f"enable='{enable_expr}':")
        parts.append("shadowcolor=black@0.65:shadowx=2:shadowy=2:")
        parts.append(f"alpha='{alpha_expr}'")

        filt = "".join(parts)

        if hz > 0.001 and eff == "color_swap":
            try:
                c2 = list(getattr(ov, "effect_color2_rgba", None) or [255, 64, 64, 255])
            except Exception:
                c2 = [255, 64, 64, 255]
            if len(c2) != 4:
                c2 = [255, 64, 64, 255]
            try:
                r2, g2, b2 = int(c2[0]), int(c2[1]), int(c2[2])
            except Exception:
                r2, g2, b2 = 255, 64, 64
            col2 = f"#{int(r2):02x}{int(g2):02x}{int(b2):02x}"

            per = 1.0 / hz
            half = per / 2.0
            enable_base = f"between(t,{start_s:.6f},{end_s:.6f})"
            mod_expr = f"mod(t-{start_s:.6f},{per:.6f})"
            enable2 = f"{enable_base}*gte({mod_expr},{half:.6f})"

            filt2 = filt.replace(f"fontcolor={base_color}:", f"fontcolor={col2}:", 1)
            filt2 = filt2.replace(f"enable='{enable_expr}':", f"enable='{enable2}':", 1)
            return filt + "," + filt2, None

        return filt, None

    @staticmethod
    def _fade_alpha_expr(start_s: float, end_s: float, fi_s: float, fo_s: float, base_alpha: float) -> str:
        """
        Returns an ffmpeg expression in [0..1] for drawtext alpha with fade in/out.
        base_alpha is multiplied in.
        """
        s = float(start_s)
        e = float(end_s)
        fi = float(max(0.0, fi_s))
        fo = float(max(0.0, fo_s))
        a0 = float(_clamp(base_alpha, 0.0, 1.0))

        # inner curve (0..1) inside [s,e]
        inner = "1"
        if fi > 1e-6:
            inner = f"if(lt(t,{s + fi:.6f}),(t-{s:.6f})/{fi:.6f},1)"
        if fo > 1e-6:
            inner = f"if(lt(t,{e - fo:.6f}),{inner},({e:.6f}-t)/{fo:.6f})"

        full = f"if(lt(t,{s:.6f}),0,if(lt(t,{e:.6f}),{inner},0))"
        if abs(a0 - 1.0) < 1e-6:
            return full
        return f"{a0:.6f}*({full})"

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


# Backwards-compatible alias (Tools tab expects `videotextPane`)
videotextPane = VideoTextPane

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = VideoTextPane()
    w.setWindowTitle("Video Text Overlay / Creator")
    w.resize(1400, 820)
    w.show()
    sys.exit(app.exec())
