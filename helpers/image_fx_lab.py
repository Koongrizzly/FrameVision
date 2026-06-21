# FrameVision - Image FX Lab
# Standalone PySide6 helper for quick, non-AI image color/effect edits.
# Outputs: <FrameVision root>/output/edits/image_fx_lab/
# Undo/temp: <FrameVision root>/temp/image_fx_lab/
# Settings: <FrameVision root>/presets/setsave/image_fx_lab.json

from __future__ import annotations

import json
import copy
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

try:
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
except Exception as exc:  # pragma: no cover - shown in GUI when launched
    np = None  # type: ignore
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageFilter = None  # type: ignore
    ImageOps = None  # type: ignore
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "FrameVision - Image FX Lab"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _framevision_root() -> Path:
    """Return FrameVision root when this file lives in /helpers, else current work dir."""
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "helpers":
        return here.parent.parent
    return Path.cwd().resolve()


ROOT_DIR = _framevision_root()
TEMP_DIR = ROOT_DIR / "temp" / "image_fx_lab"
OUT_DIR = ROOT_DIR / "output" / "edits" / "image_fx_lab"
SETTINGS_PATH = ROOT_DIR / "presets" / "setsave" / "image_fx_lab.json"


def _ensure_dirs() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _safe_json_load(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _safe_json_save(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clamp_u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_to_pixmap(img: "Image.Image") -> QPixmap:
    rgba = img.convert("RGBA")
    w, h = rgba.size
    raw = rgba.tobytes("raw", "RGBA")
    qimg = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def keep_alpha(original: "Image.Image", rgb_img: "Image.Image") -> "Image.Image":
    if original.mode == "RGBA":
        out = rgb_img.convert("RGBA")
        out.putalpha(original.getchannel("A"))
        return out
    return rgb_img.convert("RGBA")


def apply_hue_shift(img: "Image.Image", degrees: int) -> "Image.Image":
    if not degrees:
        return img
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    hsv = rgba.convert("RGB").convert("HSV")
    arr = np.array(hsv, dtype=np.int16)
    arr[:, :, 0] = (arr[:, :, 0] + int(degrees / 360.0 * 255.0)) % 256
    shifted = Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGBA")
    shifted.putalpha(alpha)
    return shifted


def apply_temperature_tint(img: "Image.Image", temperature: int, tint: int) -> "Image.Image":
    if not temperature and not tint:
        return img
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)

    # Temperature: positive = warmer, negative = cooler.
    t = float(temperature) / 100.0
    if t > 0:
        arr[:, :, 0] *= 1.0 + 0.18 * t
        arr[:, :, 2] *= 1.0 - 0.14 * t
    elif t < 0:
        t = abs(t)
        arr[:, :, 2] *= 1.0 + 0.18 * t
        arr[:, :, 0] *= 1.0 - 0.14 * t

    # Tint: positive = magenta, negative = green.
    ti = float(tint) / 100.0
    if ti > 0:
        arr[:, :, 0] *= 1.0 + 0.08 * ti
        arr[:, :, 2] *= 1.0 + 0.08 * ti
        arr[:, :, 1] *= 1.0 - 0.08 * ti
    elif ti < 0:
        ti = abs(ti)
        arr[:, :, 1] *= 1.0 + 0.12 * ti
        arr[:, :, 0] *= 1.0 - 0.06 * ti
        arr[:, :, 2] *= 1.0 - 0.06 * ti

    return Image.fromarray(_clamp_u8(arr), "RGBA")


def apply_rgb_channels(img: "Image.Image", red: int, green: int, blue: int) -> "Image.Image":
    if not red and not green and not blue:
        return img
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    factors = [
        max(0.0, 1.0 + red / 100.0),
        max(0.0, 1.0 + green / 100.0),
        max(0.0, 1.0 + blue / 100.0),
    ]
    arr[:, :, 0] *= factors[0]
    arr[:, :, 1] *= factors[1]
    arr[:, :, 2] *= factors[2]
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def apply_gamma(img: "Image.Image", value: int) -> "Image.Image":
    if not value:
        return img
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32) / 255.0
    if value > 0:
        gamma = 1.0 / (1.0 + value / 100.0)
    else:
        gamma = 1.0 + abs(value) / 35.0
    arr[:, :, :3] = np.power(np.clip(arr[:, :, :3], 0, 1), gamma)
    arr[:, :, :3] *= 255.0
    arr[:, :, 3] *= 255.0
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def apply_vignette(img: "Image.Image", amount: int) -> "Image.Image":
    if amount <= 0:
        return img
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    h, w = arr.shape[:2]
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    dist = np.sqrt(x * x + y * y)
    strength = min(0.95, amount / 100.0)
    mask = 1.0 - np.clip((dist - 0.25) / 0.9, 0, 1) * strength
    arr[:, :, :3] *= mask[:, :, None]
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def apply_grain(img: "Image.Image", amount: int, seed: int) -> "Image.Image":
    if amount <= 0:
        return img
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount * 0.55, arr[:, :, :3].shape)
    arr[:, :, :3] += noise
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def apply_posterize(img: "Image.Image", amount: int) -> "Image.Image":
    if amount <= 0:
        return img
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bits = max(1, 8 - int(amount))
    out = ImageOps.posterize(rgba.convert("RGB"), bits).convert("RGBA")
    out.putalpha(alpha)
    return out


def apply_threshold(img: "Image.Image", level: int) -> "Image.Image":
    if level <= 0:
        return img
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    gray = ImageOps.grayscale(rgba)
    bw = gray.point(lambda p: 255 if p >= level else 0).convert("RGBA")
    bw.putalpha(alpha)
    return bw


def effect_invert(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    inv = ImageOps.invert(rgba.convert("RGB")).convert("RGBA")
    inv.putalpha(alpha)
    return inv


def effect_grayscale(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    out = ImageOps.grayscale(rgba).convert("RGBA")
    out.putalpha(alpha)
    return out


def effect_sepia(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    tr = 0.393 * r + 0.769 * g + 0.189 * b
    tg = 0.349 * r + 0.686 * g + 0.168 * b
    tb = 0.272 * r + 0.534 * g + 0.131 * b
    arr[:, :, 0] = tr
    arr[:, :, 1] = tg
    arr[:, :, 2] = tb
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def effect_auto_contrast(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    out = ImageOps.autocontrast(rgba.convert("RGB"), cutoff=1).convert("RGBA")
    out.putalpha(alpha)
    return out


# --------------------------- Creative one-click effects ---------------------------
def _rgba_array(img: "Image.Image"):
    return np.array(img.convert("RGBA")).astype(np.float32)


def _with_original_alpha(source: "Image.Image", result: "Image.Image") -> "Image.Image":
    src = source.convert("RGBA")
    out = result.convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return out


def _screen_blend(base_rgb, glow_rgb):
    base = np.clip(base_rgb / 255.0, 0, 1)
    glow = np.clip(glow_rgb / 255.0, 0, 1)
    return (1.0 - (1.0 - base) * (1.0 - glow)) * 255.0


def _colorize_gray(img: "Image.Image", black: str, white: str) -> "Image.Image":
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    gray = ImageOps.grayscale(rgba)
    out = ImageOps.colorize(gray, black=black, white=white).convert("RGBA")
    out.putalpha(alpha)
    return out


def effect_cyberpunk(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    base = ImageEnhance.Contrast(rgba).enhance(1.28)
    base = ImageEnhance.Color(base).enhance(1.42)
    base = apply_hue_shift(base, -8)
    arr = _rgba_array(base)
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]) / 255.0
    shadows = np.clip(1.0 - lum, 0, 1)[:, :, None]
    highs = np.clip(lum, 0, 1)[:, :, None]
    arr[:, :, :3] += shadows * np.array([0, 34, 72], dtype=np.float32)
    arr[:, :, :3] += highs * np.array([48, 0, 46], dtype=np.float32)
    out = Image.fromarray(_clamp_u8(arr), "RGBA")
    out = ImageEnhance.Sharpness(out).enhance(1.35)
    return apply_vignette(out, 22)


def effect_matrix_green(img: "Image.Image") -> "Image.Image":
    out = _colorize_gray(img, black="#001407", white="#99ff99")
    out = ImageEnhance.Contrast(out).enhance(1.32)
    out = ImageEnhance.Brightness(out).enhance(0.88)
    return apply_vignette(out, 30)


def effect_vhs(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()
    r = ImageChops.offset(r, 2, 0)
    b = ImageChops.offset(b, -2, 0)
    out = Image.merge("RGBA", (r, g, b, a))
    out = ImageEnhance.Color(out).enhance(0.82)
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = out.filter(ImageFilter.GaussianBlur(radius=0.35))
    arr = _rgba_array(out)
    arr[1::4, :, :3] *= 0.82
    arr[2::4, :, :3] *= 0.94
    h, w = arr.shape[:2]
    rng = np.random.default_rng(1234)
    arr[:, :, :3] += rng.normal(0, 4.5, (h, w, 3))
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def effect_old_photo(img: "Image.Image") -> "Image.Image":
    out = effect_sepia(img)
    out = ImageEnhance.Contrast(out).enhance(0.92)
    out = ImageEnhance.Brightness(out).enhance(1.04)
    out = apply_grain(out, 14, 1977)
    return apply_vignette(out, 28)


def effect_dreamy_glow(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    base = ImageEnhance.Brightness(rgba).enhance(1.06)
    base = ImageEnhance.Color(base).enhance(1.12)
    blur = base.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(base.size) / 90.0)))
    arr = _rgba_array(base)
    glow = _rgba_array(blur)
    arr[:, :, :3] = arr[:, :, :3] * 0.70 + _screen_blend(arr[:, :, :3], glow[:, :, :3]) * 0.30
    out = Image.fromarray(_clamp_u8(arr), "RGBA")
    return _with_original_alpha(img, out)


def effect_dark_moody(img: "Image.Image") -> "Image.Image":
    out = img.convert("RGBA")
    out = ImageEnhance.Brightness(out).enhance(0.72)
    out = ImageEnhance.Contrast(out).enhance(1.36)
    out = ImageEnhance.Color(out).enhance(0.78)
    out = apply_temperature_tint(out, -18, 4)
    return apply_vignette(out, 40)


def effect_neon_edge(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    base = ImageEnhance.Brightness(rgba).enhance(0.42)
    base = ImageEnhance.Contrast(base).enhance(1.25)
    edges = rgba.convert("RGB").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(ImageOps.grayscale(edges))
    e = np.array(edges).astype(np.float32) / 255.0
    arr = _rgba_array(base)
    cyan = np.array([0, 255, 230], dtype=np.float32)
    magenta = np.array([255, 0, 210], dtype=np.float32)
    h, w = e.shape
    xgrad = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    neon = cyan * (1.0 - xgrad) + magenta * xgrad
    arr[:, :, :3] = arr[:, :, :3] * (1.0 - e[:, :, None] * 0.72) + neon * (e[:, :, None] * 1.05)
    out = Image.fromarray(_clamp_u8(arr), "RGBA")
    return _with_original_alpha(img, out)


def effect_comic_outline(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    color = ImageEnhance.Color(rgba).enhance(1.28)
    color = ImageEnhance.Contrast(color).enhance(1.18)
    alpha = color.getchannel("A")
    poster = ImageOps.posterize(color.convert("RGB"), 4).convert("RGBA")
    poster.putalpha(alpha)
    edges = rgba.convert("RGB").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(ImageOps.grayscale(edges))
    mask = np.array(edges).astype(np.uint8) > 38
    arr = _rgba_array(poster)
    arr[mask, :3] = 0
    return Image.fromarray(_clamp_u8(arr), "RGBA")


def effect_high_contrast_silhouette(img: "Image.Image") -> "Image.Image":
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    gray = ImageOps.grayscale(rgba)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    arr = np.array(gray).astype(np.uint8)
    threshold = int(np.clip(np.mean(arr) * 0.92, 70, 185))
    bw = Image.fromarray(np.where(arr > threshold, 245, 8).astype(np.uint8), "L")
    out = bw.convert("RGBA")
    out.putalpha(alpha)
    return out


def effect_duotone_blue_orange(img: "Image.Image") -> "Image.Image":
    out = _colorize_gray(img, black="#061426", white="#ffb35a")
    out = ImageEnhance.Contrast(out).enhance(1.15)
    return out


def effect_duotone_teal_purple(img: "Image.Image") -> "Image.Image":
    out = _colorize_gray(img, black="#17091f", white="#53ffe0")
    out = ImageEnhance.Contrast(out).enhance(1.18)
    return out


def effect_duotone_green_black(img: "Image.Image") -> "Image.Image":
    out = _colorize_gray(img, black="#020805", white="#7dff65")
    out = ImageEnhance.Contrast(out).enhance(1.22)
    return out


class ImagePreview(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.StyledPanel)
        self.setText("Drop an image here or use Open")
        self.setAcceptDrops(False)
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, img: Optional["Image.Image"]) -> None:
        if img is None:
            self._pixmap = None
            self.setText("Drop an image here or use Open")
            self.setPixmap(QPixmap())
            return
        self._pixmap = pil_to_pixmap(img)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if not self._pixmap:
            return
        target = self.size() - QSize(18, 18)
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setPixmap(self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))


DIRECT_EFFECTS: Dict[str, Callable[["Image.Image"], "Image.Image"]] = {
    "effect_invert": effect_invert,
    "effect_grayscale": effect_grayscale,
    "effect_sepia": effect_sepia,
    "effect_auto_contrast": effect_auto_contrast,
    "effect_cyberpunk": effect_cyberpunk,
    "effect_matrix_green": effect_matrix_green,
    "effect_vhs": effect_vhs,
    "effect_old_photo": effect_old_photo,
    "effect_dreamy_glow": effect_dreamy_glow,
    "effect_dark_moody": effect_dark_moody,
    "effect_neon_edge": effect_neon_edge,
    "effect_comic_outline": effect_comic_outline,
    "effect_high_contrast_silhouette": effect_high_contrast_silhouette,
    "effect_duotone_blue_orange": effect_duotone_blue_orange,
    "effect_duotone_teal_purple": effect_duotone_teal_purple,
    "effect_duotone_green_black": effect_duotone_green_black,
}


class SliderRow(QWidget):
    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        default: int,
        tooltip: str,
        on_change: Callable[[], None],
    ) -> None:
        super().__init__()
        self.default = default
        self._on_change = on_change
        self.label = QLabel(label)
        self.label.setMinimumWidth(86)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(default)
        self.slider.setToolTip(tooltip)
        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(default)
        self.spin.setFixedWidth(72)
        self.spin.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(lambda _v: self._on_change())

    def value(self) -> int:
        return int(self.slider.value())

    def set_value(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(value))
        self.spin.setValue(int(value))
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def reset(self) -> None:
        self.set_value(self.default)


class ImageFxLab(QMainWindow):
    def __init__(self, parent=None, embedded: bool = False) -> None:
        super().__init__(parent)
        self._embedded = bool(embedded)
        _ensure_dirs()
        self.setWindowTitle(APP_TITLE)
        self.setAcceptDrops(True)
        if self._embedded:
            self.setWindowFlags(Qt.Widget)

        self.settings = _safe_json_load(SETTINGS_PATH)
        self.current_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None
        self.current_image: Optional[Image.Image] = None
        self.preview_image: Optional[Image.Image] = None
        self.applied_ops: list[dict] = []
        self.undo_paths: list[Path] = []
        self.redo_paths: list[Path] = []
        self.session_dir = TEMP_DIR / time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_index = 0
        self.grain_seed = int(time.time()) & 0xFFFF
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(60)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self.update_preview)

        self.preview = ImagePreview()
        self.rows: Dict[str, SliderRow] = {}
        self._build_ui()
        if self._embedded:
            self.resize(1180, 760)
        else:
            self._restore_window()
        self._set_controls_enabled(False)

        if PIL_IMPORT_ERROR is not None:
            QMessageBox.critical(
                self,
                "Missing dependency",
                "Image FX Lab needs Pillow and NumPy.\n\n"
                f"Import error:\n{PIL_IMPORT_ERROR}",
            )

    # --------------------------- UI ---------------------------
    def _build_ui(self) -> None:
        toolbar = QToolBar("Image FX")
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.open_action = QAction("Open", self)
        self.open_action.setToolTip("Load an image. You can also drag and drop one into the preview.")
        self.open_action.triggered.connect(self.open_image)
        toolbar.addAction(self.open_action)

        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save the current preview to /output/edits/image_fx_lab/.")
        self.save_action.triggered.connect(self.save_image_default)
        toolbar.addAction(self.save_action)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.setToolTip("Choose a custom save location.")
        self.save_as_action.triggered.connect(self.save_image_as)
        toolbar.addAction(self.save_as_action)

        toolbar.addSeparator()

        self.undo_action = QAction("Undo", self)
        self.undo_action.setToolTip("Go back to the previous applied state. States are stored in /temp/image_fx_lab/.")
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setToolTip("Restore the next applied state.")
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)

        toolbar.addSeparator()

        self.rotate_left_action = QAction("Rotate L", self)
        self.rotate_left_action.setToolTip("Rotate the image 90° counter-clockwise and add an undo state.")
        self.rotate_left_action.triggered.connect(self.rotate_left)
        toolbar.addAction(self.rotate_left_action)

        self.rotate_right_action = QAction("Rotate R", self)
        self.rotate_right_action.setToolTip("Rotate the image 90° clockwise and add an undo state.")
        self.rotate_right_action.triggered.connect(self.rotate_right)
        toolbar.addAction(self.rotate_right_action)

        self.flip_h_action = QAction("Flip H", self)
        self.flip_h_action.setToolTip("Flip the image horizontally and add an undo state.")
        self.flip_h_action.triggered.connect(self.flip_horizontal)
        toolbar.addAction(self.flip_h_action)

        self.flip_v_action = QAction("Flip V", self)
        self.flip_v_action.setToolTip("Flip the image vertically and add an undo state.")
        self.flip_v_action.triggered.connect(self.flip_vertical)
        toolbar.addAction(self.flip_v_action)

        self.batch_action = QAction("Batch", self)
        self.batch_action.setToolTip("Apply the current look to every supported image in a folder and save them under /output/edits/image_fx_lab/batch/.")
        self.batch_action.triggered.connect(self.batch_apply_folder)
        toolbar.addAction(self.batch_action)

        toolbar.addSeparator()

        self.compare_check = QCheckBox("Original")
        self.compare_check.setToolTip("Temporarily show the original loaded image.")
        self.compare_check.stateChanged.connect(lambda _v: self.update_preview(immediate=True))
        toolbar.addWidget(self.compare_check)

        main = QWidget()
        root_layout = QVBoxLayout(main)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.preview)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "WEBP"])
        self.format_combo.setToolTip("Default format used by Save.")
        self.format_combo.setCurrentText(str(self.settings.get("format", "PNG")).upper())
        self.format_combo.currentTextChanged.connect(lambda _v: self._save_settings())

        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(int(self.settings.get("quality", 95)))
        self.quality.setToolTip("JPG/WEBP quality. PNG ignores this.")
        self.quality.valueChanged.connect(lambda _v: self._save_settings())

        self.path_label = QLabel(str(OUT_DIR))
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setToolTip("Default output folder.")

        bottom.addWidget(QLabel("Format"))
        bottom.addWidget(self.format_combo)
        bottom.addSpacing(8)
        bottom.addWidget(QLabel("Quality"))
        bottom.addWidget(self.quality)
        bottom.addSpacing(12)
        bottom.addWidget(self.path_label, 1)
        root_layout.addLayout(bottom)

        self.setCentralWidget(main)
        self.setStatusBar(QStatusBar())
        self._update_status("Open an image to start.")

    def _build_side_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        layout.addWidget(self._tone_group())
        layout.addWidget(self._detail_group())
        layout.addWidget(self._quick_group())
        layout.addWidget(self._preset_group())
        layout.addStretch(1)

        scroll.setWidget(content)
        scroll.setMinimumWidth(355)
        return scroll

    def _add_row(self, key: str, label: str, mn: int, mx: int, default: int, tooltip: str, parent_layout: QVBoxLayout) -> None:
        row = SliderRow(label, mn, mx, default, tooltip, self.schedule_preview)
        self.rows[key] = row
        parent_layout.addWidget(row)

    def _tone_group(self) -> QGroupBox:
        group = QGroupBox("Tone / Color")
        layout = QVBoxLayout(group)
        self._add_row("brightness", "Brightness", -100, 100, 0, "Darker or brighter without changing the file until Apply is used.", layout)
        self._add_row("contrast", "Contrast", -100, 100, 0, "Lower for softer image, higher for stronger dark/light separation.", layout)
        self._add_row("saturation", "Saturation", -100, 100, 0, "Reduce or boost color strength.", layout)
        self._add_row("red", "Red", -100, 100, 0, "Reduce or boost the red channel.", layout)
        self._add_row("green", "Green", -100, 100, 0, "Reduce or boost the green channel.", layout)
        self._add_row("blue", "Blue", -100, 100, 0, "Reduce or boost the blue channel.", layout)
        self._add_row("hue", "Hue", -180, 180, 0, "Rotate colors around the hue wheel.", layout)
        self._add_row("gamma", "Gamma", -100, 100, 0, "Positive lifts midtones, negative darkens midtones.", layout)
        self._add_row("temperature", "Temp", -100, 100, 0, "Negative is cooler, positive is warmer.", layout)
        self._add_row("tint", "Tint", -100, 100, 0, "Negative adds green, positive adds magenta.", layout)
        return group

    def _detail_group(self) -> QGroupBox:
        group = QGroupBox("Detail / Style")
        layout = QVBoxLayout(group)
        self._add_row("sharpness", "Sharpness", -100, 100, 0, "Positive sharpens, negative softens.", layout)
        self._add_row("blur", "Blur", 0, 30, 0, "Adds a simple Gaussian blur.", layout)
        self._add_row("vignette", "Vignette", 0, 100, 0, "Darkens the edges.", layout)
        self._add_row("grain", "Grain", 0, 100, 0, "Adds stable film-like grain. Refresh Grain changes the pattern.", layout)
        self._add_row("posterize", "Posterize", 0, 7, 0, "Reduces color steps for a stylized look.", layout)
        self._add_row("threshold", "Threshold", 0, 255, 0, "Hard black/white cutoff. Leave at 0 to disable.", layout)

        row = QHBoxLayout()
        self.refresh_grain_btn = QPushButton("Refresh Grain")
        self.refresh_grain_btn.setToolTip("Generate a different grain pattern for the preview.")
        self.refresh_grain_btn.clicked.connect(self.refresh_grain)
        row.addWidget(self.refresh_grain_btn)
        layout.addLayout(row)
        return group

    def _quick_group(self) -> QGroupBox:
        group = QGroupBox("Quick")
        layout = QGridLayout(group)

        def add_button(text: str, tooltip: str, func: Callable[[], None], r: int, c: int) -> None:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(func)
            layout.addWidget(btn, r, c)

        add_button("Apply", "Bake the current slider preview into the image and add an undo state.", self.apply_current_preview, 0, 0)
        add_button("Reset", "Return to the original loaded image and reset all sliders.", self.reset_to_original, 0, 1)
        add_button("B/W", "Apply black and white as an undo state.", lambda: self.apply_direct(effect_grayscale, "Black and white"), 1, 0)
        add_button("Invert", "Invert colors as an undo state.", lambda: self.apply_direct(effect_invert, "Invert"), 1, 1)
        add_button("Sepia", "Apply a warm old-photo effect as an undo state.", lambda: self.apply_direct(effect_sepia, "Sepia"), 2, 0)
        add_button("Auto", "Apply a simple auto-contrast correction as an undo state.", lambda: self.apply_direct(effect_auto_contrast, "Auto contrast"), 2, 1)
        return group


    def _preset_group(self) -> QGroupBox:
        group = QGroupBox("Presets")
        layout = QGridLayout(group)

        presets = [
            ("Clean", "slider", {"brightness": 8, "contrast": 8, "saturation": 5, "sharpness": 18}, "Light cleanup: small brightness, contrast, saturation and sharpness boost."),
            ("Bright", "slider", {"brightness": 22, "contrast": 8, "saturation": 8, "gamma": 28}, "Brighter clean look. Use Apply to bake the slider preview."),
            ("Dark mood", "direct", effect_dark_moody, "Darker, colder, stronger contrast with vignette."),
            ("Cyber boost", "direct", effect_cyberpunk, "Boost contrast and neon blue/magenta color separation."),
            ("Matrix green", "direct", effect_matrix_green, "Green monochrome tint with stronger contrast and dark edges."),
            ("VHS", "direct", effect_vhs, "RGB channel offset, soft scanlines and light tape noise."),
            ("Old photo", "direct", effect_old_photo, "Sepia, soft contrast, grain and vignette."),
            ("Dream glow", "direct", effect_dreamy_glow, "Soft bright glow while keeping the original image readable."),
            ("Neon edge", "direct", effect_neon_edge, "Dark image with cyan/magenta edge glow."),
            ("Comic", "direct", effect_comic_outline, "Posterized color with simple black outlines."),
            ("Silhouette", "direct", effect_high_contrast_silhouette, "Hard high-contrast black and white silhouette look."),
            ("Duo warm", "direct", effect_duotone_blue_orange, "Blue shadows and warm orange highlights."),
            ("Duo teal", "direct", effect_duotone_teal_purple, "Purple shadows and teal highlights."),
            ("Duo green", "direct", effect_duotone_green_black, "Dark green/black duotone preset."),
        ]

        for i, (name, kind, payload, tooltip) in enumerate(presets):
            btn = QPushButton(name)
            btn.setToolTip(tooltip)
            if kind == "slider":
                btn.clicked.connect(lambda _checked=False, vals=payload: self.apply_preset(vals))
            else:
                btn.clicked.connect(lambda _checked=False, f=payload, label=name: self.apply_direct(f, label))
            layout.addWidget(btn, i // 2, i % 2)
        return group

    # --------------------------- File handling ---------------------------
    def open_image(self) -> None:
        if PIL_IMPORT_ERROR is not None:
            return
        start = self.settings.get("last_open_folder") or str(ROOT_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path) -> None:
        if PIL_IMPORT_ERROR is not None:
            return
        try:
            if path.suffix.lower() not in IMAGE_EXTS:
                raise ValueError("Unsupported image type")
            img = Image.open(path)
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGBA")
        except Exception as exc:
            QMessageBox.warning(self, "Could not open image", f"{path}\n\n{exc}")
            return

        self.current_path = path
        self.original_image = img.copy()
        self.current_image = img.copy()
        self.preview_image = img.copy()
        self.applied_ops = []
        self.undo_paths.clear()
        self.redo_paths.clear()
        self.reset_controls(update=False)
        self._clear_session()
        self._push_snapshot("loaded", clear_redo=True)
        self._set_controls_enabled(True)
        self.settings["last_open_folder"] = str(path.parent)
        self._save_settings()
        self.update_preview(immediate=True)
        self._update_status(f"Loaded: {path.name}  ({img.width} x {img.height})")

    def save_image_default(self) -> None:
        img = self._image_to_save()
        if img is None:
            return
        ext = self.format_combo.currentText().lower()
        stem = self.current_path.stem if self.current_path else "image"
        target = OUT_DIR / f"{stem}_fx_{time.strftime('%Y%m%d_%H%M%S')}.{ext.lower()}"
        self._save_to_path(target, img)

    def save_image_as(self) -> None:
        img = self._image_to_save()
        if img is None:
            return
        ext = self.format_combo.currentText().lower()
        stem = self.current_path.stem if self.current_path else "image"
        start = OUT_DIR / f"{stem}_fx.{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image as",
            str(start),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)",
        )
        if path:
            self._save_to_path(Path(path), img)

    def _save_to_path(self, path: Path, img: "Image.Image") -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                img.convert("RGB").save(path, quality=int(self.quality.value()), optimize=True)
            elif suffix == ".webp":
                img.convert("RGBA").save(path, quality=int(self.quality.value()), method=6)
            else:
                if suffix != ".png":
                    path = path.with_suffix(".png")
                img.convert("RGBA").save(path)
            self._update_status(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Could not save image", f"{path}\n\n{exc}")

    # --------------------------- Preview/effects ---------------------------
    def schedule_preview(self) -> None:
        self._preview_timer.start()

    def update_preview(self, immediate: bool = False) -> None:
        if immediate:
            self._preview_timer.stop()
        if self.current_image is None:
            self.preview.set_image(None)
            return
        if self.compare_check.isChecked() and self.original_image is not None:
            self.preview_image = self.original_image.copy()
        else:
            try:
                self.preview_image = self._render_from_controls(self.current_image)
            except Exception as exc:
                self.preview_image = self.current_image.copy()
                self._update_status(f"Preview failed: {exc}")
        self.preview.set_image(self.preview_image)
        self._update_actions()

    def _render_from_controls(self, source: "Image.Image", values: Optional[dict] = None) -> "Image.Image":
        img = source.convert("RGBA")
        values = values or self._current_slider_values()

        b = int(values.get("brightness", 0))
        c = int(values.get("contrast", 0))
        s = int(values.get("saturation", 0))
        sh = int(values.get("sharpness", 0))
        blur = int(values.get("blur", 0))

        if b:
            img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b / 100.0))
        if c:
            img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c / 100.0))
        if s:
            img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s / 100.0))

        img = apply_rgb_channels(img, int(values.get("red", 0)), int(values.get("green", 0)), int(values.get("blue", 0)))
        img = apply_hue_shift(img, int(values.get("hue", 0)))
        img = apply_gamma(img, int(values.get("gamma", 0)))
        img = apply_temperature_tint(img, int(values.get("temperature", 0)), int(values.get("tint", 0)))

        if sh:
            if sh > 0:
                img = ImageEnhance.Sharpness(img).enhance(1.0 + sh / 35.0)
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=abs(sh) / 35.0))
        if blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur / 8.0))

        img = apply_vignette(img, int(values.get("vignette", 0)))
        img = apply_grain(img, int(values.get("grain", 0)), self.grain_seed)
        img = apply_posterize(img, int(values.get("posterize", 0)))
        img = apply_threshold(img, int(values.get("threshold", 0)))
        return img.convert("RGBA")

    def apply_current_preview(self) -> None:
        if self.current_image is None:
            return
        self.update_preview(immediate=True)
        if self.preview_image is None:
            return
        self.current_image = self.preview_image.copy()
        self.reset_controls(update=False)
        self._push_snapshot("apply", clear_redo=True)
        self.update_preview(immediate=True)
        self._update_status("Applied.")

    def apply_direct(self, func: Callable[["Image.Image"], "Image.Image"], label: str) -> None:
        if self.current_image is None:
            return
        try:
            # Direct effects apply to the visible preview, so active sliders are not lost.
            self.update_preview(immediate=True)
            base = self.preview_image.copy() if self.preview_image is not None else self.current_image.copy()
            self.current_image = func(base).convert("RGBA")
            self.reset_controls(update=False)
            self._push_snapshot(label.lower().replace(" ", "_"), clear_redo=True)
            self.update_preview(immediate=True)
            self._update_status(f"Applied: {label}")
        except Exception as exc:
            QMessageBox.warning(self, "Effect failed", f"{label}\n\n{exc}")

    def apply_preset(self, values: dict) -> None:
        self.reset_controls(update=False)
        for key, value in values.items():
            row = self.rows.get(key)
            if row:
                row.set_value(int(value))
        self.update_preview(immediate=True)

    def reset_controls(self, update: bool = True) -> None:
        for row in self.rows.values():
            row.reset()
        if update:
            self.update_preview(immediate=True)

    def reset_to_original(self) -> None:
        """Reset the visible edit back to the originally loaded image.

        Internal calls still use reset_controls() when only the temporary slider
        values need to be cleared. The toolbar/button Reset should be stronger:
        users expect the image colors/effects to return too, not only the
        slider handles.
        """
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy().convert("RGBA")
        self.applied_ops = []
        self.reset_controls(update=False)
        if hasattr(self, "compare_check"):
            self.compare_check.blockSignals(True)
            self.compare_check.setChecked(False)
            self.compare_check.blockSignals(False)
        self._push_snapshot("reset_original", clear_redo=True)
        self.update_preview(immediate=True)
        self._update_status("Reset to original image.")

    def _current_slider_values(self) -> dict:
        return {key: row.value() for key, row in self.rows.items()}

    def _ops_sidecar_path(self, image_path: Path) -> Path:
        return Path(str(image_path) + ".json")

    def _save_ops_sidecar(self, image_path: Path) -> None:
        _safe_json_save(self._ops_sidecar_path(image_path), {"applied_ops": copy.deepcopy(self.applied_ops)})

    def _load_ops_sidecar(self, image_path: Path) -> list[dict]:
        data = _safe_json_load(self._ops_sidecar_path(image_path))
        ops = data.get("applied_ops")
        return ops if isinstance(ops, list) else []

    def _apply_op_chain(self, source: "Image.Image", ops: list[dict], slider_values: Optional[dict] = None) -> "Image.Image":
        img = source.convert("RGBA")
        for op in ops:
            kind = op.get("type")
            if kind == "slider":
                img = self._render_from_controls(img, op.get("values") or {})
            elif kind == "direct":
                func = DIRECT_EFFECTS.get(str(op.get("name") or ""))
                if func is not None:
                    img = func(img).convert("RGBA")
            elif kind == "transform":
                name = str(op.get("name") or "")
                if name == "rotate_left":
                    img = img.transpose(Image.Transpose.ROTATE_90)
                elif name == "rotate_right":
                    img = img.transpose(Image.Transpose.ROTATE_270)
                elif name == "flip_horizontal":
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif name == "flip_vertical":
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if slider_values is None:
            slider_values = self._current_slider_values()
        if slider_values and any(int(v) != 0 for v in slider_values.values()):
            img = self._render_from_controls(img, slider_values)
        return img.convert("RGBA")

    def _apply_transform(self, name: str, label: str) -> None:
        if self.current_image is None:
            return
        self.update_preview(immediate=True)
        base = self.preview_image.copy() if self.preview_image is not None else self.current_image.copy()
        if name == "rotate_left":
            result = base.transpose(Image.Transpose.ROTATE_90)
        elif name == "rotate_right":
            result = base.transpose(Image.Transpose.ROTATE_270)
        elif name == "flip_horizontal":
            result = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif name == "flip_vertical":
            result = base.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            return
        self.current_image = result.convert("RGBA")
        self.reset_controls(update=False)
        self.applied_ops.append({"type": "transform", "name": name})
        self._push_snapshot(name, clear_redo=True)
        self.update_preview(immediate=True)
        self._update_status(f"Applied: {label}")

    def rotate_left(self) -> None:
        self._apply_transform("rotate_left", "Rotate left")

    def rotate_right(self) -> None:
        self._apply_transform("rotate_right", "Rotate right")

    def flip_horizontal(self) -> None:
        self._apply_transform("flip_horizontal", "Flip horizontal")

    def flip_vertical(self) -> None:
        self._apply_transform("flip_vertical", "Flip vertical")

    def batch_apply_folder(self) -> None:
        if self.original_image is None or self.current_image is None:
            self._update_status("Open an image first.")
            return
        start = self.settings.get("last_batch_folder") or self.settings.get("last_open_folder") or str(ROOT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Choose folder to batch process", start)
        if not folder:
            return
        source_dir = Path(folder)
        files = [p for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not files:
            QMessageBox.information(self, APP_TITLE, "No supported images were found in that folder.")
            return

        slider_values = self._current_slider_values()
        batch_dir = OUT_DIR / "batch" / time.strftime("%Y%m%d_%H%M%S")
        batch_dir.mkdir(parents=True, exist_ok=True)

        ok_count = 0
        fail_count = 0
        for path in files:
            try:
                img = Image.open(path)
                img.load()
                img = ImageOps.exif_transpose(img).convert("RGBA")
                out = self._apply_op_chain(img, self.applied_ops, slider_values)
                ext = self.format_combo.currentText().lower()
                target = batch_dir / f"{path.stem}_fx.{ext}"
                self._save_to_path(target, out)
                ok_count += 1
            except Exception:
                fail_count += 1

        self.settings["last_batch_folder"] = str(source_dir)
        self._save_settings()
        self._update_status(f"Batch done: {ok_count} saved, {fail_count} failed. Output: {batch_dir}")
        QMessageBox.information(
            self,
            APP_TITLE,
            f"Batch finished.\n\nSaved: {ok_count}\nFailed: {fail_count}\n\nOutput:\n{batch_dir}",
        )

    def refresh_grain(self) -> None:
        self.grain_seed = (int(time.time() * 1000) ^ os.getpid()) & 0xFFFFFFFF
        self.update_preview(immediate=True)

    def _image_to_save(self) -> Optional["Image.Image"]:
        if self.current_image is None:
            self._update_status("Open an image first.")
            return None
        self.update_preview(immediate=True)
        return self.preview_image.copy() if self.preview_image is not None else self.current_image.copy()

    # --------------------------- Undo/redo ---------------------------
    def _clear_session(self) -> None:
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir, ignore_errors=True)
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.snapshot_index = 0
        except Exception:
            pass

    def _push_snapshot(self, label: str, clear_redo: bool) -> None:
        if self.current_image is None:
            return
        self.snapshot_index += 1
        safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)[:32]
        path = self.session_dir / f"{self.snapshot_index:04d}_{safe_label}.png"
        self.current_image.save(path)
        self._save_ops_sidecar(path)
        self.undo_paths.append(path)
        if clear_redo:
            self.redo_paths.clear()
        self._trim_undo()
        self._update_actions()

    def _trim_undo(self) -> None:
        max_states = int(self.settings.get("max_undo_states", 50))
        while len(self.undo_paths) > max_states:
            old = self.undo_paths.pop(0)
            try:
                old.unlink(missing_ok=True)
                self._ops_sidecar_path(old).unlink(missing_ok=True)
            except Exception:
                pass

    def undo(self) -> None:
        if len(self.undo_paths) <= 1 or self.current_image is None:
            return
        current = self.undo_paths.pop()
        self.redo_paths.append(current)
        self._load_snapshot(self.undo_paths[-1])
        self._update_status("Undo.")

    def redo(self) -> None:
        if not self.redo_paths:
            return
        path = self.redo_paths.pop()
        self.undo_paths.append(path)
        self._load_snapshot(path)
        self._update_status("Redo.")

    def _load_snapshot(self, path: Path) -> None:
        try:
            img = Image.open(path)
            img.load()
            self.current_image = img.convert("RGBA")
            self.applied_ops = self._load_ops_sidecar(path)
            self.reset_controls(update=False)
            self.update_preview(immediate=True)
        except Exception as exc:
            QMessageBox.warning(self, "Undo/redo failed", f"{path}\n\n{exc}")
        self._update_actions()

    # --------------------------- State/helpers ---------------------------
    def _set_controls_enabled(self, enabled: bool) -> None:
        for row in self.rows.values():
            row.setEnabled(enabled)
        for obj in [
            self.save_action,
            self.save_as_action,
            self.undo_action,
            self.redo_action,
            self.rotate_left_action,
            self.rotate_right_action,
            self.flip_h_action,
            self.flip_v_action,
            self.batch_action,
            self.compare_check,
            self.refresh_grain_btn,
            self.format_combo,
            self.quality,
        ]:
            obj.setEnabled(enabled)
        self._update_actions()

    def _update_actions(self) -> None:
        has_image = self.current_image is not None
        self.save_action.setEnabled(has_image)
        self.save_as_action.setEnabled(has_image)
        self.undo_action.setEnabled(has_image and len(self.undo_paths) > 1)
        self.redo_action.setEnabled(has_image and bool(self.redo_paths))
        for action in [self.rotate_left_action, self.rotate_right_action, self.flip_h_action, self.flip_v_action, self.batch_action]:
            action.setEnabled(has_image)
        self.compare_check.setEnabled(has_image and self.original_image is not None)

    def _update_status(self, text: str) -> None:
        if self.statusBar():
            self.statusBar().showMessage(text)

    def _save_settings(self) -> None:
        data = dict(self.settings)
        data["format"] = self.format_combo.currentText() if hasattr(self, "format_combo") else data.get("format", "PNG")
        data["quality"] = int(self.quality.value()) if hasattr(self, "quality") else int(data.get("quality", 95))
        if not getattr(self, "_embedded", False):
            data["geometry"] = bytes(self.saveGeometry()).hex()
            data["window_state"] = bytes(self.saveState()).hex()
        if "last_open_folder" in self.settings:
            data["last_open_folder"] = self.settings["last_open_folder"]
        data.setdefault("max_undo_states", 50)
        self.settings = data
        _safe_json_save(SETTINGS_PATH, data)

    def _restore_window(self) -> None:
        geo = self.settings.get("geometry")
        if isinstance(geo, str):
            try:
                self.restoreGeometry(bytes.fromhex(geo))
            except Exception:
                self.resize(1180, 760)
        else:
            self.resize(1180, 760)
        state = self.settings.get("window_state")
        if isinstance(state, str):
            try:
                self.restoreState(bytes.fromhex(state))
            except Exception:
                pass

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in IMAGE_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_EXTS:
                self.load_path(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_settings()
        super().closeEvent(event)


# Optional import helper for FrameVision main app later.
def create_image_fx_lab_widget(parent=None) -> ImageFxLab:
    """Create an embeddable Image FX Lab widget for FrameVision Tools."""
    win = ImageFxLab(parent=parent, embedded=True)
    try:
        win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        win.setMinimumHeight(760)
        win.setProperty("expand_min_h", 760)
    except Exception:
        pass
    return win


def open_image_fx_lab(parent=None) -> ImageFxLab:
    win = ImageFxLab(parent=None, embedded=False)
    if parent is not None:
        win.setWindowModality(Qt.NonModal)
    win.show()
    return win


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = ImageFxLab()
        win.show()
        return app.exec()
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, APP_TITLE, traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
