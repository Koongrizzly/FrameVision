from __future__ import annotations

"""
helpers/zimage_sel_edit.py

Standalone PySide helper widget for Z-Image Turbo GGUF selective edit / inpaint.
Designed to be imported later into txt2img.py as a dedicated tab when the
"Z-Image Turbo (GGUF Low VRAM)" engine is selected.

What this file does now:
- Load a source image.
- Paint a grayscale mask over the image (left mouse = paint, right mouse = erase).
- Export source + mask into a portable temp folder under the app root.
- Build a Z-Image GGUF job payload.
- Optionally run helpers/zimage_cli.py via QProcess if no external callback is supplied.

Notes:
- This helper assumes a small future patch in helpers/zimage_cli.py so it accepts
  --mask-image and forwards it to sd-cli as --mask.
- Until that runner patch is added, direct execution from this helper may fail with
  an "unknown argument --mask-image" style error. The UI still prepares the correct
  source/mask assets and job payload cleanly.
"""

import json
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Any

from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QProcess, QProcessEnvironment
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl


# -----------------------------------------------------------------------------
# helpers / paths
# -----------------------------------------------------------------------------

def _app_root() -> Path:
    try:
        return Path(__file__).resolve().parent.parent
    except Exception:
        return Path.cwd()


def _settings_path() -> Path:
    p = _app_root() / "presets" / "setsave" / "z_edit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _temp_root() -> Path:
    p = _app_root() / "temp" / "zimage_sel_edit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_output_dir() -> Path:
    p = _app_root() / "output" / "zimage_selective_edit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_python() -> str:
    root = _app_root()
    candidates = [
        root / "environments" / ".images_models" / ("python.exe" if os.name == "nt" else "bin/python"),
        root / "environments" / ".qwen2512" / ("python.exe" if os.name == "nt" else "bin/python"),
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            pass
    return sys.executable


def _find_zimage_helper() -> str:
    root = _app_root()
    candidates = [
        root / "helpers" / "zimage_cli.py",
        Path(__file__).resolve().parent / "zimage_cli.py",
        Path.cwd() / "helpers" / "zimage_cli.py",
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            pass
    return str(root / "helpers" / "zimage_cli.py")


def _unique_path(p: Path) -> Path:
    try:
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix
        i = 1
        while True:
            cand = p.with_name(f"{stem}_{i:03d}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    except Exception:
        return p


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _json_load(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# painter widget
# -----------------------------------------------------------------------------

class _MaskPaintView(QWidget):
    changed = Signal()
    statusMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image = QImage()
        self._mask = QImage()
        self._paint_active = False
        self._last_point = QPoint()
        self._scale = 1.0
        self._min_scale = 0.2
        self._max_scale = 8.0
        self._brush_size = 36
        self._overlay_alpha = 120
        self._show_checker = True
        self._show_overlay = True
        self._overlay_cache = QImage()
        self._overlay_cache_dirty = True
        self._mouse_mode = "paint"   # "paint" or "erase"
        self._checker_a = QColor(45, 45, 45)
        self._checker_b = QColor(60, 60, 60)

    # ---- public API ----
    def has_image(self) -> bool:
        return not self._image.isNull()

    def image_size(self) -> QSize:
        return self._image.size() if not self._image.isNull() else QSize(0, 0)

    def clear_image(self) -> None:
        self._image = QImage()
        self._mask = QImage()
        self._overlay_cache = QImage()
        self._overlay_cache_dirty = True
        self._paint_active = False
        self._last_point = QPoint()
        self._scale = 1.0
        self.updateGeometry()
        self.update()
        self.changed.emit()
        self.statusMessage.emit("Source image cleared")

    def set_show_overlay(self, show: bool) -> None:
        self._show_overlay = bool(show)
        self.update()

    def set_overlay_alpha(self, alpha: int) -> None:
        self._overlay_alpha = max(0, min(255, int(alpha)))
        self._overlay_cache_dirty = True
        self.update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))
        self.statusMessage.emit(f"Brush size: {self._brush_size}px")
        self.update()

    def set_zoom(self, value: float) -> None:
        self._scale = max(self._min_scale, min(self._max_scale, float(value)))
        self.updateGeometry()
        self.update()
        self.statusMessage.emit(f"Zoom: {int(self._scale * 100)}%")

    def zoom(self) -> float:
        return self._scale

    def load_image(self, path: str) -> bool:
        img = QImage(path)
        if img.isNull():
            return False
        if img.format() != QImage.Format_ARGB32:
            img = img.convertToFormat(QImage.Format_ARGB32)
        self._image = img
        # Internal mask uses ARGB32 alpha, not Grayscale8.
        # QPainter is unreliable on Grayscale8 in some PySide6 builds, which made
        # brush strokes silently do nothing. Export converts this to grayscale.
        self._mask = QImage(self._image.size(), QImage.Format_ARGB32)
        self._mask.fill(Qt.transparent)
        self._overlay_cache = QImage()
        self._overlay_cache_dirty = True
        self._scale = 1.0
        self.updateGeometry()
        self.update()
        self.changed.emit()
        self.statusMessage.emit(f"Loaded image: {Path(path).name} ({img.width()}x{img.height()})")
        return True

    def clear_mask(self) -> None:
        if self._mask.isNull():
            return
        self._mask.fill(Qt.transparent)
        self._overlay_cache_dirty = True
        self.update()
        self.changed.emit()
        self.statusMessage.emit("Mask cleared")

    def invert_mask(self) -> None:
        if self._mask.isNull():
            return
        # Invert alpha mask: painted becomes erased, erased becomes painted.
        # This is rare enough that a pixel pass is fine, and keeps the normal
        # paint path fast.
        inv = QImage(self._mask.size(), QImage.Format_ARGB32)
        inv.fill(Qt.transparent)
        try:
            for y in range(self._mask.height()):
                for x in range(self._mask.width()):
                    a = QColor(self._mask.pixelColor(x, y)).alpha()
                    if a <= 0:
                        inv.setPixelColor(x, y, QColor(255, 255, 255, 255))
        except Exception:
            pass
        self._mask = inv
        self._overlay_cache_dirty = True
        self.update()
        self.changed.emit()
        self.statusMessage.emit("Mask inverted")

    def _build_export_mask(self, threshold: int = 1, grow_px: int = 2) -> QImage:
        """Return a hard black/white RGB mask for sd-cli.

        Keep the editable UI mask as ARGB32 so painting stays reliable, but do not
        export the soft antialiased edge directly. Soft gray edge pixels can make
        sd-cli only partially edit the border. This export path creates pure black
        and white pixels and grows the white area a little so replacement covers
        the painted edge.
        """
        if self._mask.isNull():
            return QImage()

        w = int(self._mask.width())
        h = int(self._mask.height())
        thr = max(0, min(255, int(threshold)))
        grow = max(0, int(grow_px))

        out = QImage(self._mask.size(), QImage.Format_RGB32)
        out.fill(QColor(0, 0, 0))

        white = QColor(255, 255, 255)
        painted: list[tuple[int, int]] = []

        try:
            for y in range(h):
                for x in range(w):
                    if self._mask.pixelColor(x, y).alpha() > thr:
                        painted.append((x, y))
        except Exception:
            # Last-resort fallback: keep old behavior but still avoid Grayscale8 writing.
            fallback = QImage(self._mask.size(), QImage.Format_RGB32)
            fallback.fill(QColor(0, 0, 0))
            qp = QPainter(fallback)
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            qp.drawImage(0, 0, self._mask)
            qp.end()
            return fallback

        if grow <= 0:
            try:
                for x, y in painted:
                    out.setPixelColor(x, y, white)
            except Exception:
                pass
            return out

        # Circular dilation. This is only run on export, not while painting, so the
        # small loop is acceptable and avoids new dependencies.
        offsets: list[tuple[int, int]] = []
        r2 = grow * grow
        for dy in range(-grow, grow + 1):
            for dx in range(-grow, grow + 1):
                if (dx * dx + dy * dy) <= r2:
                    offsets.append((dx, dy))

        try:
            for x, y in painted:
                for dx, dy in offsets:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        out.setPixelColor(nx, ny, white)
        except Exception:
            # If dilation failed for any reason, still export the non-grown binary mask.
            try:
                out.fill(QColor(0, 0, 0))
                for x, y in painted:
                    out.setPixelColor(x, y, white)
            except Exception:
                pass
        return out

    def export_source_and_mask(self, folder: Path) -> tuple[Optional[Path], Optional[Path]]:
        if self._image.isNull() or self._mask.isNull():
            return None, None
        folder.mkdir(parents=True, exist_ok=True)
        src_path = folder / "source.png"
        mask_path = folder / "mask.png"
        self._image.save(str(src_path), "PNG")
        export_mask = self._build_export_mask(threshold=1, grow_px=2)
        export_mask.save(str(mask_path), "PNG")
        return src_path, mask_path

    def save_mask_preview(self, path: str) -> bool:
        if self._mask.isNull():
            return False
        return self._build_export_mask(threshold=1, grow_px=2).save(path, "PNG")

    def set_mouse_mode(self, mode: str) -> None:
        self._mouse_mode = "erase" if str(mode).lower().startswith("erase") else "paint"
        self.statusMessage.emit(f"Mode: {'Erase' if self._mouse_mode == 'erase' else 'Paint'}")
        self.update()

    # ---- Qt overrides ----
    def sizeHint(self) -> QSize:
        if self._image.isNull():
            return QSize(780, 520)
        return QSize(max(1, int(self._image.width() * self._scale)), max(1, int(self._image.height() * self._scale)))

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 240)

    def wheelEvent(self, event: QWheelEvent) -> None:
        try:
            mods = event.modifiers()
            delta = event.angleDelta().y()
        except Exception:
            return super().wheelEvent(event)
        if mods & Qt.ControlModifier:
            step = 0.1 if delta > 0 else -0.1
            self.set_zoom(self._scale + step)
            event.accept()
            return
        if mods & Qt.ShiftModifier:
            step = 2 if delta > 0 else -2
            self.set_brush_size(self._brush_size + step)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._image.isNull():
            return
        if event.button() == Qt.LeftButton:
            self._paint_active = True
            self._last_point = self._to_image_pos(event.position().toPoint())
            self._stroke(self._last_point, self._last_point, paint=True)
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self._paint_active = True
            self._last_point = self._to_image_pos(event.position().toPoint())
            self._stroke(self._last_point, self._last_point, paint=False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._paint_active and not self._mask.isNull():
            pt = self._to_image_pos(event.position().toPoint())
            erase = bool(event.buttons() & Qt.RightButton) or self._mouse_mode == "erase"
            paint = not erase
            self._stroke(self._last_point, pt, paint=paint)
            self._last_point = pt
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._paint_active = False
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        if self._image.isNull():
            p.fillRect(self.rect(), QColor(30, 30, 30))
            p.setPen(QColor(210, 210, 210))
            p.drawText(self.rect(), Qt.AlignCenter, "Load an image to start selective editing")
            return

        target = QRect(0, 0, int(self._image.width() * self._scale), int(self._image.height() * self._scale))
        if self._show_checker:
            self._draw_checker(p, target)
        else:
            p.fillRect(target, QColor(22, 22, 22))

        p.drawImage(target, self._image)

        if self._show_overlay and not self._mask.isNull():
            overlay = self._get_overlay_cache()
            if not overlay.isNull():
                p.drawImage(target, overlay)

        # Draw a faint border.
        p.setPen(QColor(180, 180, 180, 120))
        p.drawRect(target.adjusted(0, 0, -1, -1))

    # ---- internals ----
    def _get_overlay_cache(self) -> QImage:
        """Return a cached red alpha overlay for the current mask.

        The normal paint path is fast: no per-pixel rebuild on every repaint.
        The mask itself stores alpha; the overlay is made by drawing a solid red
        layer and clipping it through the mask alpha.
        """
        try:
            if self._mask.isNull():
                return QImage()
            if (not self._overlay_cache_dirty
                    and not self._overlay_cache.isNull()
                    and self._overlay_cache.size() == self._mask.size()):
                return self._overlay_cache

            overlay = QImage(self._mask.size(), QImage.Format_ARGB32)
            overlay.fill(Qt.transparent)
            qp = QPainter(overlay)
            qp.fillRect(overlay.rect(), QColor(255, 60, 60, int(self._overlay_alpha)))
            qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            qp.drawImage(0, 0, self._mask)
            qp.end()

            self._overlay_cache = overlay
            self._overlay_cache_dirty = False
            return self._overlay_cache
        except Exception:
            return QImage()

    def _to_image_pos(self, view_pt: QPoint) -> QPoint:
        if self._image.isNull() or self._scale <= 0.0:
            return QPoint(0, 0)
        x = int(view_pt.x() / self._scale)
        y = int(view_pt.y() / self._scale)
        x = max(0, min(self._image.width() - 1, x))
        y = max(0, min(self._image.height() - 1, y))
        return QPoint(x, y)

    def _stroke(self, a: QPoint, b: QPoint, paint: bool = True) -> None:
        if self._mask.isNull():
            return
        qp = QPainter(self._mask)
        qp.setRenderHint(QPainter.Antialiasing, True)
        if paint:
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(QColor(255, 255, 255, 255), self._brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        else:
            qp.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(QColor(0, 0, 0, 0), self._brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        qp.setPen(pen)
        qp.drawLine(a, b)
        qp.end()
        self._overlay_cache_dirty = True
        self.update()
        self.changed.emit()

    def _draw_checker(self, p: QPainter, target: QRect) -> None:
        sz = 16
        for yy in range(0, target.height(), sz):
            for xx in range(0, target.width(), sz):
                idx = ((xx // sz) + (yy // sz)) % 2
                p.fillRect(target.x() + xx, target.y() + yy, sz, sz, self._checker_a if idx == 0 else self._checker_b)


# -----------------------------------------------------------------------------
# main widget
# -----------------------------------------------------------------------------

class ZImageSelectiveEditWidget(QWidget):
    fileReady = Signal(str)
    jobBuilt = Signal(dict)
    statusMessage = Signal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        embedded: bool = False,
        run_callback: Optional[Callable[[dict], Any]] = None,
        gguf_provider: Optional[Callable[[], dict]] = None,
        python_exe: Optional[str] = None,
        zimage_helper_path: Optional[str] = None,
    ):
        super().__init__(parent)
        self.embedded = bool(embedded)
        self.run_callback = run_callback
        self.gguf_provider = gguf_provider
        self.python_exe = python_exe or _find_python()
        self.zimage_helper_path = zimage_helper_path or _find_zimage_helper()
        self._proc: Optional[QProcess] = None
        self._last_job_folder: Optional[Path] = None
        self._last_image_path: Optional[str] = None
        self._build_ui()
        self._load_settings()

    # ---- ui ----
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        page_scroll = QScrollArea(self)
        page_scroll.setWidgetResizable(True)
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page_scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(page_scroll, 1)

        page = QWidget()
        page_scroll.setWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.banner = QLabel("Z-Image Turbo GGUF — Selective Edit")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setObjectName("zimgSelBanner")
        self.banner.setStyleSheet(
            "#zimgSelBanner {"
            "font-size: 15px; font-weight: 600; padding: 8px 14px; border-radius: 12px;"
            "color: white;"
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5c3cff, stop:1 #9e63ff);"
            "}"
        )
        root.addWidget(self.banner)

        # Source row
        source_row = QHBoxLayout()
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("Source image path")
        self.btn_browse_source = QPushButton("Browse")
        self.btn_clear_source = QPushButton("Clear")
        source_row.addWidget(QLabel("Source image"))
        source_row.addWidget(self.btn_browse_source)
        source_row.addWidget(self.btn_clear_source)
        source_row.addWidget(self.source_path, 1)
        root.addLayout(source_row)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # Left side: paint view
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        toolbar = QHBoxLayout()
        self.btn_clear_mask = QPushButton("Clear mask")
        self.btn_invert_mask = QPushButton("Invert mask")
        self.btn_fit_100 = QPushButton("100%")
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_in = QPushButton("+")
        self.lbl_zoom = QLabel("100%")
        toolbar.addWidget(self.btn_clear_mask)
        toolbar.addWidget(self.btn_invert_mask)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Zoom"))
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_fit_100)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.lbl_zoom)
        left_lay.addLayout(toolbar)

        self.paint_view = _MaskPaintView()
        self.paint_view.statusMessage.connect(self._set_status)
        self.paint_view.changed.connect(self._on_canvas_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        try:
            scroll.setStyleSheet("QScrollArea { background: #202020; }")
        except Exception:
            pass
        scroll.setWidget(self.paint_view)
        left_lay.addWidget(scroll, 1)

        hint = QLabel("Left mouse paints the editable region. Right mouse erases. Ctrl + mouse wheel = zoom. Shift + mouse wheel = brush size.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #bcbcbc;")
        left_lay.addWidget(hint)

        splitter.addWidget(left)

        # Right side controls
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        # Brush / canvas settings
        grp_brush = QGroupBox("Brush and overlay")
        brush_form = QFormLayout(grp_brush)
        self.brush_size = QSlider(Qt.Horizontal)
        self.brush_size.setRange(1, 256)
        self.brush_size.setValue(36)
        self.overlay_alpha = QSlider(Qt.Horizontal)
        self.overlay_alpha.setRange(10, 255)
        self.overlay_alpha.setValue(120)
        self.edit_mode = QComboBox()
        self.edit_mode.addItems(["Paint by default", "Erase by default"])
        self.show_overlay = QCheckBox("Show red overlay")
        self.show_overlay.setChecked(True)
        brush_form.addRow("Brush size", self.brush_size)
        brush_form.addRow("Overlay opacity", self.overlay_alpha)
        brush_form.addRow("Primary mouse mode", self.edit_mode)
        brush_form.addRow("Overlay", self.show_overlay)
        right_lay.addWidget(grp_brush)

        # Prompt settings
        grp_prompt = QGroupBox("Prompt")
        prompt_lay = QVBoxLayout(grp_prompt)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Describe what should appear in the painted region")
        self.negative = QTextEdit()
        self.negative.setPlaceholderText("Optional negative prompt")
        prompt_lay.addWidget(QLabel("Prompt"))
        prompt_lay.addWidget(self.prompt)
        prompt_lay.addWidget(QLabel("Negative"))
        prompt_lay.addWidget(self.negative)
        right_lay.addWidget(grp_prompt, 1)

        # Generation settings
        grp_gen = QGroupBox("Generation settings")
        gen_form = QFormLayout(grp_gen)
        self.width = QSpinBox(); self.width.setRange(256, 4096); self.width.setSingleStep(64); self.width.setValue(1024)
        self.height = QSpinBox(); self.height.setRange(256, 4096); self.height.setSingleStep(64); self.height.setValue(1024)
        self.match_source = QCheckBox("Match source image size")
        self.match_source.setChecked(True)
        self.steps = QSpinBox(); self.steps.setRange(1, 200); self.steps.setValue(12)
        self.guidance = QDoubleSpinBox(); self.guidance.setRange(0.0, 30.0); self.guidance.setDecimals(2); self.guidance.setSingleStep(0.25); self.guidance.setValue(4.0)
        self.strength = QDoubleSpinBox(); self.strength.setRange(0.0, 1.0); self.strength.setDecimals(2); self.strength.setSingleStep(0.05); self.strength.setValue(0.85)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(0)
        self.attn_slicing = QCheckBox("Low VRAM mode (attn slicing / CPU offload flags)")
        self.attn_slicing.setChecked(False)
        gen_form.addRow("Width", self.width)
        gen_form.addRow("Height", self.height)
        gen_form.addRow("Size", self.match_source)
        gen_form.addRow("Steps", self.steps)
        gen_form.addRow("Guidance", self.guidance)
        gen_form.addRow("Strength", self.strength)
        gen_form.addRow("Seed (0 = random)", self.seed)
        gen_form.addRow("Performance", self.attn_slicing)
        right_lay.addWidget(grp_gen)

        # Output
        grp_out = QGroupBox("Output")
        out_form = QFormLayout(grp_out)
        self.output_path = QLineEdit(str(_default_output_dir()))
        self.btn_browse_output = QPushButton("Browse")
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_path, 1)
        out_row.addWidget(self.btn_browse_output)
        out_form.addRow("Output folder", out_row)
        self.keep_temp = QCheckBox("Keep temp source/mask files")
        self.keep_temp.setChecked(True)
        out_form.addRow("Temp files", self.keep_temp)
        self.use_queue = QCheckBox("Use FrameVision queue")
        self.use_queue.setChecked(True)
        out_form.addRow("Queue", self.use_queue)
        right_lay.addWidget(grp_out)

        # Helpful hover tips for the selective-edit workflow.
        try:
            _tips = {
                "source_path": "Loaded source image path. Browse selects the image you want to edit.",
                "btn_browse_source": "Browse for the image to edit.",
                "btn_clear_source": "Clear the loaded source image and mask.",
                "btn_reload_source": "Reload the current source image from disk.",
                "btn_clear_mask": "Clear only the painted mask. The source image stays loaded.",
                "btn_invert_mask": "Invert the mask: painted areas become protected and protected areas become editable.",
                "btn_zoom_out": "Zoom out of the image preview.",
                "btn_fit_100": "Reset preview zoom to 100%.",
                "btn_zoom_in": "Zoom into the image preview.",
                "brush_size": "Brush size for painting the editable area. Left mouse paints, right mouse erases. Default: 36.",
                "overlay_alpha": "Red mask overlay opacity. This only changes visibility, not generation strength. Default: 120.",
                "edit_mode": "Choose whether left-drag paints or erases by default. Default: Paint by default.",
                "show_overlay": "Show or hide the red mask overlay. The mask still exists when hidden. Default: On.",
                "prompt": "Describe what should appear inside the painted mask. Be specific, for example: 'a small fluffy white puppy sitting on the DJ turntable'.",
                "negative": "Optional things to avoid inside the edit. Example: duplicate gear, cables, buttons, distorted paws.",
                "width": "Output width. Disabled when 'Match source image size' is on.",
                "height": "Output height. Disabled when 'Match source image size' is on.",
                "match_source": "Use the loaded source image size for output. Default: On.",
                "steps": "Generation steps. More can add detail but is slower. Default: 12.",
                "guidance": "Prompt guidance for Z-Image GGUF selective edit. Higher follows the prompt more, too high may look forced. Default: 4.00.",
                "strength": "How strongly the painted area is regenerated. Lower keeps more of the original; higher replaces more. Default: 0.85.",
                "seed": "Seed for repeatable tests. 0 means random seed. Use a fixed seed when comparing settings. Default: 0.",
                "attn_slicing": "Low VRAM mode adds CPU/offload flags. Slower, but can help on smaller GPUs. Default: Off.",
                "output_path": "Folder where edited images are saved.",
                "btn_browse_output": "Choose the output folder for selective edit results.",
                "keep_temp": "Keep temporary source/mask files under temp/zimage_sel_edit for debugging. Default: On.",
                "use_queue": "Send the job through the FrameVision queue. Default: On.",
            }
            for _name, _tip in _tips.items():
                _w = getattr(self, _name, None)
                if _w is not None and hasattr(_w, "setToolTip"):
                    _w.setToolTip(_tip)
        except Exception:
            pass


        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate selective edit")
        self.btn_open_output = QPushButton("Open output folder")
        btn_row.addWidget(self.btn_generate, 1)
        btn_row.addWidget(self.btn_open_output)
        right_lay.addLayout(btn_row)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        right_lay.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        right_lay.addWidget(self.log, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        # Wire up
        self.btn_browse_source.clicked.connect(self._pick_source_image)
        self.btn_clear_source.clicked.connect(self._clear_source_image)
        self.btn_browse_output.clicked.connect(self._pick_output_dir)
        self.btn_clear_mask.clicked.connect(self.paint_view.clear_mask)
        self.btn_invert_mask.clicked.connect(self.paint_view.invert_mask)
        self.btn_fit_100.clicked.connect(lambda: self._set_zoom(1.0))
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self.paint_view.zoom() - 0.1))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self.paint_view.zoom() + 0.1))
        self.brush_size.valueChanged.connect(self.paint_view.set_brush_size)
        self.overlay_alpha.valueChanged.connect(self.paint_view.set_overlay_alpha)
        self.show_overlay.toggled.connect(self.paint_view.set_show_overlay)
        self.edit_mode.currentIndexChanged.connect(self._sync_edit_mode)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_open_output.clicked.connect(self._open_output_dir)
        try:
            self.use_queue.toggled.connect(self._sync_queue_button)
        except Exception:
            pass
        self.source_path.editingFinished.connect(lambda: self._load_source_image(self.source_path.text().strip()))
        self.match_source.toggled.connect(self._update_size_lock_state)
        self._update_size_lock_state()
        self._sync_edit_mode()
        self._sync_queue_button()

    # ---- settings ----
    def _load_settings(self) -> None:
        s = _json_load(_settings_path())
        try:
            self.source_path.setText(str(s.get("source_path") or ""))
            self.output_path.setText(str(s.get("output_path") or _default_output_dir()))
            self.prompt.setPlainText(str(s.get("prompt") or ""))
            self.negative.setPlainText(str(s.get("negative") or ""))
            self.brush_size.setValue(int(s.get("brush_size") or 36))
            self.overlay_alpha.setValue(int(s.get("overlay_alpha") or 120))
            self.edit_mode.setCurrentIndex(int(s.get("edit_mode") or 0))
            self.show_overlay.setChecked(bool(s.get("show_overlay", True)))
            self.width.setValue(int(s.get("width") or 1024))
            self.height.setValue(int(s.get("height") or 1024))
            self.match_source.setChecked(bool(s.get("match_source", True)))
            self.steps.setValue(int(s.get("steps") or 12))
            self.guidance.setValue(float(s.get("guidance") or 4.0))
            self.strength.setValue(float(s.get("strength") or 0.85))
            self.seed.setValue(int(s.get("seed") or 0))
            self.attn_slicing.setChecked(bool(s.get("attn_slicing", False)))
            self.keep_temp.setChecked(bool(s.get("keep_temp", True)))
            if hasattr(self, "use_queue"):
                self.use_queue.setChecked(bool(s.get("use_queue", True)))
        except Exception:
            pass
        src = self.source_path.text().strip()
        if src:
            self._load_source_image(src, quiet=True)

    def _save_settings(self) -> None:
        payload = {
            "source_path": self.source_path.text().strip(),
            "output_path": self.output_path.text().strip(),
            "prompt": self.prompt.toPlainText(),
            "negative": self.negative.toPlainText(),
            "brush_size": self.brush_size.value(),
            "overlay_alpha": self.overlay_alpha.value(),
            "edit_mode": self.edit_mode.currentIndex(),
            "show_overlay": self.show_overlay.isChecked(),
            "width": self.width.value(),
            "height": self.height.value(),
            "match_source": self.match_source.isChecked(),
            "steps": self.steps.value(),
            "guidance": self.guidance.value(),
            "strength": self.strength.value(),
            "seed": self.seed.value(),
            "attn_slicing": self.attn_slicing.isChecked(),
            "keep_temp": self.keep_temp.isChecked(),
            "use_queue": bool(self.use_queue.isChecked()) if hasattr(self, "use_queue") else True,
        }
        try:
            _json_dump(_settings_path(), payload)
        except Exception:
            pass

    # ---- slots / helpers ----
    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(str(line))

    def _set_status(self, msg: str) -> None:
        self.status.setText(str(msg))
        self.statusMessage.emit(str(msg))

    def _on_canvas_changed(self) -> None:
        self._save_settings()

    def _set_zoom(self, value: float) -> None:
        self.paint_view.set_zoom(value)
        self.lbl_zoom.setText(f"{int(self.paint_view.zoom() * 100)}%")

    def _sync_edit_mode(self) -> None:
        self.paint_view.set_mouse_mode("erase" if self.edit_mode.currentIndex() == 1 else "paint")
        self._save_settings()

    def _update_size_lock_state(self) -> None:
        manual = not self.match_source.isChecked()
        self.width.setEnabled(manual)
        self.height.setEnabled(manual)
        if self.match_source.isChecked() and self.paint_view.has_image():
            size = self.paint_view.image_size()
            self.width.setValue(int(size.width()))
            self.height.setValue(int(size.height()))
        self._save_settings()

    def _pick_source_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select source image", self.source_path.text().strip() or str(_app_root()), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.source_path.setText(path)
            self._load_source_image(path)

    def _load_source_image(self, path: str, quiet: bool = False) -> None:
        path = str(path or "").strip()
        if not path:
            return
        if not Path(path).exists():
            if not quiet:
                self._set_status("Source image not found")
            return
        ok = self.paint_view.load_image(path)
        if ok:
            self._set_zoom(1.0)
            self._update_size_lock_state()
            self._save_settings()
        elif not quiet:
            self._set_status("Could not load source image")

    def _clear_source_image(self) -> None:
        try:
            self.source_path.clear()
        except Exception:
            pass
        try:
            self.paint_view.clear_image()
        except Exception:
            pass
        try:
            self.lbl_zoom.setText("100%")
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass
        self._set_status("Source image cleared")

    def _pick_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_path.text().strip() or str(_default_output_dir()))
        if folder:
            self.output_path.setText(folder)
            self._save_settings()

    def _open_output_dir(self) -> None:
        out = Path(self.output_path.text().strip() or str(_default_output_dir()))
        out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _prepare_job_assets(self) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        if not self.paint_view.has_image():
            self._set_status("Load a source image first")
            return None, None, None
        ts = time.strftime("%Y%m%d_%H%M%S")
        job_dir = _temp_root() / f"job_{ts}"
        src_path, mask_path = self.paint_view.export_source_and_mask(job_dir)
        if src_path is None or mask_path is None:
            self._set_status("Could not export source/mask")
            return None, None, None
        self._last_job_folder = job_dir
        return job_dir, src_path, mask_path

    def build_job(self) -> Optional[dict]:
        job_dir, src_path, mask_path = self._prepare_job_assets()
        if job_dir is None:
            return None

        out_dir = Path(self.output_path.text().strip() or str(_default_output_dir()))
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.match_source.isChecked() and self.paint_view.has_image():
            sz = self.paint_view.image_size()
            width = int(sz.width())
            height = int(sz.height())
        else:
            width = int(self.width.value())
            height = int(self.height.value())

        gguf_settings = {}
        try:
            if callable(self.gguf_provider):
                gguf_settings = self.gguf_provider() or {}
        except Exception:
            gguf_settings = {}

        job = {
            "engine": "zimage_gguf",
            "backend": "gguf",
            "prompt": self.prompt.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "width": width,
            "height": height,
            "steps": int(self.steps.value()),
            "guidance": float(self.guidance.value()),
            "seed": int(self.seed.value()),
            "strength": float(self.strength.value()),
            "attn_slicing": bool(self.attn_slicing.isChecked()),
            "source_image": str(src_path),
            "mask_image": str(mask_path),
            "init_image_enabled": True,
            "init_image": str(src_path),
            "img2img_strength": float(self.strength.value()),
            "outdir": str(out_dir),
            "output": str(out_dir),
            "batch": 1,
            "seed_policy": "fixed",
            "format": "png",
            "selective_edit": True,
            "model_name": "Z-Image-Turbo GGUF Selective Edit",
            "model": "Z-Image-Turbo GGUF Selective Edit",
            "filename_template": "zimage_sel_{seed}_{idx:03d}.png",
            "temp_job_dir": str(job_dir),
            "keep_temp": bool(self.keep_temp.isChecked()),
            "python_exe": self.python_exe,
            "zimage_helper_path": self.zimage_helper_path,
            "gguf_model_path": str(gguf_settings.get("gguf_model_path") or ""),
            "gguf_instruct_path": str(gguf_settings.get("gguf_instruct_path") or ""),
            "gguf_vae_path": str(gguf_settings.get("gguf_vae_path") or ""),
            "sd_cli_path": str(gguf_settings.get("sd_cli_path") or ""),
        }
        sidecar = Path(job_dir) / "job.json"
        _json_dump(sidecar, job)
        self.jobBuilt.emit(job)
        self._save_settings()
        return job

    def _sync_queue_button(self, *_):
        try:
            if hasattr(self, "btn_generate") and hasattr(self, "use_queue"):
                self.btn_generate.setText("Add selective edit to queue" if self.use_queue.isChecked() else "Generate selective edit")
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass

    def _enqueue_framevision(self, job: dict) -> bool:
        """Enqueue this selective-edit job as a normal FrameVision txt2img/Z-Image job."""
        try:
            try:
                from helpers.queue_adapter import enqueue_txt2img  # type: ignore
            except Exception:
                from queue_adapter import enqueue_txt2img  # type: ignore
        except Exception as e:
            self._append_log(f"Queue adapter not available: {e}")
            return False
        try:
            qjob = dict(job or {})
            qjob["engine"] = "zimage_gguf"
            qjob["backend"] = "gguf"
            qjob["output"] = str(qjob.get("output") or qjob.get("outdir") or _default_output_dir())
            qjob["init_image_enabled"] = True
            qjob["init_image"] = str(qjob.get("init_image") or qjob.get("source_image") or "")
            qjob["img2img_strength"] = float(qjob.get("img2img_strength", qjob.get("strength", 0.70)) or 0.70)
            qjob["batch"] = 1
            qjob["seed_policy"] = "fixed"
            qjob["format"] = "png"
            qjob["selective_edit"] = True
            qjob["run_now"] = False
            ok = bool(enqueue_txt2img(qjob))
            if ok:
                self._append_log("Added selective edit to FrameVision queue")
                self._set_status("Added selective edit to FrameVision queue")
            else:
                self._append_log("FrameVision queue enqueue returned False")
                self._set_status("Could not add selective edit to queue")
            return ok
        except Exception as e:
            self._append_log(f"Could not enqueue selective edit: {e}")
            self._set_status("Could not add selective edit to queue")
            return False

    def _on_generate(self) -> None:
        if not self.prompt.toPlainText().strip():
            QMessageBox.information(self, "Prompt required", "Please enter a prompt for the painted region.")
            return
        job = self.build_job()
        if not job:
            return
        self._append_log(f"Prepared selective-edit job in {job['temp_job_dir']}")
        self._append_log(f"Source : {job['source_image']}")
        self._append_log(f"Mask   : {job['mask_image']}")
        self._append_log(f"Output : {job['outdir']}")
        try:
            if job.get("gguf_model_path"):
                self._append_log(f"GGUF model   : {job.get('gguf_model_path')}")
            if job.get("gguf_instruct_path"):
                self._append_log(f"GGUF instruct: {job.get('gguf_instruct_path')}")
            if job.get("gguf_vae_path"):
                self._append_log(f"GGUF VAE     : {job.get('gguf_vae_path')}")
        except Exception:
            pass

        try:
            use_queue = bool(self.use_queue.isChecked()) if hasattr(self, "use_queue") else True
        except Exception:
            use_queue = True

        if use_queue:
            if self._enqueue_framevision(job):
                return
            self._append_log("Queue failed; falling back to local run")

        if callable(self.run_callback):
            try:
                self.run_callback(job)
                self._set_status("Selective edit job sent to external runner")
            except Exception as e:
                self._append_log(f"External callback failed: {e}")
                self._set_status("External callback failed")
            return

        self._run_local(job)

    def _run_local(self, job: dict) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

        py = str(job.get("python_exe") or self.python_exe)
        helper = str(job.get("zimage_helper_path") or self.zimage_helper_path)
        app_root = _app_root()
        args = [
            helper,
            "--backend", "gguf",
            "--prompt", str(job.get("prompt") or ""),
            "--negative", str(job.get("negative") or ""),
            "--height", str(job.get("height") or 1024),
            "--width", str(job.get("width") or 1024),
            "--steps", str(job.get("steps") or 12),
            "--guidance", str(job.get("guidance") or 4.0),
            "--seed", str(job.get("seed") or 0),
            "--outdir", str(job.get("outdir") or _default_output_dir()),
            "--filename_template", str(job.get("filename_template") or "zimage_sel_{seed}_{idx:03d}.png"),
            "--init-image", str(job.get("source_image") or ""),
            "--mask-image", str(job.get("mask_image") or ""),
            "--strength", str(job.get("strength") or 0.70),
        ]
        if str(job.get("gguf_model_path") or "").strip():
            args += ["--gguf-model", str(job.get("gguf_model_path") or "")]
        if str(job.get("gguf_instruct_path") or "").strip():
            args += ["--gguf-instruct", str(job.get("gguf_instruct_path") or "")]
        if str(job.get("gguf_vae_path") or "").strip():
            args += ["--gguf-vae", str(job.get("gguf_vae_path") or "")]
        if bool(job.get("attn_slicing", False)):
            args.append("--attn-slicing")

        self._append_log("Running local selective-edit helper...")
        self._append_log(" ".join([py] + args))
        self._set_status("Running selective edit...")

        proc = QProcess(self)
        self._proc = proc
        proc.setProgram(py)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(app_root))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        try:
            if str(job.get("sd_cli_path") or "").strip():
                env.insert("SD_CLI_PATH", str(job.get("sd_cli_path") or ""))
                env.insert("SD_CLI", str(job.get("sd_cli_path") or ""))
            if str(job.get("gguf_instruct_path") or "").strip():
                env.insert("ZIMAGE_GGUF_INSTRUCT_PATH", str(job.get("gguf_instruct_path") or ""))
                env.insert("ZIMAGE_LLM_PATH", str(job.get("gguf_instruct_path") or ""))
        except Exception:
            pass
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_proc_stdout)
        proc.readyReadStandardError.connect(self._on_proc_stderr)
        proc.finished.connect(self._on_proc_finished)
        proc.start()

    def _on_proc_stdout(self) -> None:
        if self._proc is None:
            return
        try:
            data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "ignore")
        except Exception:
            data = ""
        if not data:
            return
        for line in data.splitlines():
            line = line.rstrip()
            if not line:
                continue
            self._append_log(line)
            try:
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("files"):
                    files = payload.get("files") or []
                    if files:
                        self._last_image_path = str(files[0])
                        self.fileReady.emit(str(files[0]))
                        self._set_status(f"Done: {files[0]}")
            except Exception:
                pass

    def _on_proc_stderr(self) -> None:
        if self._proc is None:
            return
        try:
            data = bytes(self._proc.readAllStandardError()).decode("utf-8", "ignore")
        except Exception:
            data = ""
        if data:
            for line in data.splitlines():
                if line.strip():
                    self._append_log(line)

    def _on_proc_finished(self, exit_code: int, exit_status) -> None:
        self._append_log(f"Process finished with code {exit_code}")
        if exit_code != 0:
            self._set_status(
                "Selective edit finished with an error. If the log mentions '--mask-image', patch helpers/zimage_cli.py to accept mask-image and forward it to sd-cli as --mask."
            )
        else:
            self._set_status("Selective edit run completed")

        if self._last_job_folder and not self.keep_temp.isChecked():
            try:
                shutil.rmtree(self._last_job_folder, ignore_errors=True)
            except Exception:
                pass
        self._proc = None


# Convenience alias for future imports if you prefer the shorter class name.
ZImageSelEdit = ZImageSelectiveEditWidget


# -----------------------------------------------------------------------------
# standalone test launch
# -----------------------------------------------------------------------------

def _main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = ZImageSelectiveEditWidget(embedded=False)
    w.resize(1420, 900)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(_main())
