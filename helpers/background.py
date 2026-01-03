"""bg_remove.py

Standalone (or embeddable) background remover tool for PySide6.

Design goals:
- Can be imported as a widget (BgRemovePane) into a larger app later.
- Standalone main() available for quick testing.
- Uses ONNX models stored under: root/models/bg/
    - BiRefNet-COD-epoch_125.onnx
    - modnet.onnx
- Persists settings to: root/presets/setsave/bg_remove.json

Notes:
- This module intentionally avoids external heavy deps (e.g. OpenCV).
  It uses Pillow + numpy, and onnxruntime for inference.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from PySide6 import QtCore, QtGui, QtWidgets


# -------------------------
# Paths / Settings
# -------------------------

def _project_root() -> Path:
    # helpers/bg_remove.py -> root
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class SettingsStore(QtCore.QObject):
    """JSON-backed settings with debounced autosave."""

    changed = QtCore.Signal(dict)

    def __init__(self, settings_path: Path, defaults: Dict[str, Any]):
        super().__init__()
        self._path = settings_path
        self._defaults = defaults
        self._data: Dict[str, Any] = {}
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_now)
        self.load()

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    def load(self) -> None:
        _ensure_dir(self._path.parent)
        if self._path.exists():
            try:
                self._data = {**self._defaults, **json.loads(self._path.read_text(encoding="utf-8"))}
            except Exception:
                self._data = dict(self._defaults)
        else:
            self._data = dict(self._defaults)
            self._save_now()
        self.changed.emit(self.data)

    def set(self, key: str, value: Any) -> None:
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self.changed.emit(self.data)
        self._save_timer.start(250)

    def update(self, **kwargs: Any) -> None:
        changed = False
        for k, v in kwargs.items():
            if self._data.get(k) != v:
                self._data[k] = v
                changed = True
        if changed:
            self.changed.emit(self.data)
            self._save_timer.start(250)

    def _save_now(self) -> None:
        try:
            _ensure_dir(self._path.parent)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            # silent failure; parent app can hook logging
            pass


# -------------------------
# ONNX Inference
# -------------------------

@dataclass
class PadInfo:
    top: int
    left: int
    height: int
    width: int


def _pil_to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _normalize_minus1_to1(x: np.ndarray) -> np.ndarray:
    # x: float32 in [0,1]
    return (x - 0.5) / 0.5


def _resize_keep_aspect(w: int, h: int, long_side: int) -> Tuple[int, int, float]:
    if max(w, h) == long_side:
        return w, h, 1.0
    scale = long_side / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return nw, nh, scale


def _make_divisible(x: int, div: int) -> int:
    return int(np.ceil(x / div) * div)


def _letterbox(img: Image.Image, target_w: int, target_h: int) -> Tuple[Image.Image, PadInfo]:
    # Center pad to target.
    w, h = img.size
    if w == target_w and h == target_h:
        return img, PadInfo(0, 0, h, w)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    left = (target_w - w) // 2
    top = (target_h - h) // 2
    canvas.paste(img, (left, top))
    return canvas, PadInfo(top=top, left=left, height=h, width=w)


def _unletterbox(matte: np.ndarray, pad: PadInfo) -> np.ndarray:
    # matte: HxW
    y0, x0 = pad.top, pad.left
    y1, x1 = y0 + pad.height, x0 + pad.width
    return matte[y0:y1, x0:x1]


class OnnxModel:
    """Lazy ONNX Runtime session wrapper."""

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self._session = None
        self._input_name = None
        self._input_shape = None

    def is_available(self) -> bool:
        return self.path.exists()

    def ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required for background removal. "
                "Install it in your environment (CPU build is fine)."
            ) from e

        sess_options = ort.SessionOptions()
        # A small optimization for CPU-only use.
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.path), sess_options=sess_options, providers=providers)
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError(f"Model has no inputs: {self.path.name}")
        self._input_name = inputs[0].name
        self._input_shape = inputs[0].shape

    @property
    def input_name(self) -> str:
        if self._input_name is None:
            self.ensure_loaded()
        return str(self._input_name)

    @property
    def input_shape(self):
        if self._input_shape is None:
            self.ensure_loaded()
        return self._input_shape

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        assert self._session is not None
        out = self._session.run(None, {self.input_name: input_tensor})
        if not out:
            raise RuntimeError(f"No outputs from model: {self.path.name}")
        y = out[0]
        # Accept shapes like:
        #   (1,1,H,W), (1,H,W), (H,W), (1,H,W,1)
        y = np.asarray(y)
        if y.ndim == 4:
            if y.shape[1] == 1:
                y = y[0, 0]
            elif y.shape[-1] == 1:
                y = y[0, :, :, 0]
            else:
                # best-effort: pick first channel
                y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        elif y.ndim != 2:
            raise RuntimeError(f"Unexpected output shape {y.shape} from {self.path.name}")
        return y


def _infer_target_hw(model: OnnxModel, fallback_long_side: int, div: int = 32) -> Tuple[int, int, bool]:
    """Return (target_w, target_h, fixed)"""
    shape = model.input_shape
    # shape usually [N,3,H,W]
    fixed_h = None
    fixed_w = None
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        h = shape[2]
        w = shape[3]
        if isinstance(h, int) and h > 0:
            fixed_h = h
        if isinstance(w, int) and w > 0:
            fixed_w = w
    if fixed_h and fixed_w:
        return int(fixed_w), int(fixed_h), True

    # dynamic: use fallback long side and divisible.
    # We'll set both dimensions based on image aspect later, so return (0,0,False)
    return 0, 0, False


def _prepare_input(
    img_rgb: Image.Image,
    model: OnnxModel,
    ref_long_side: int,
    div: int,
    normalize: str,
) -> Tuple[np.ndarray, Tuple[int, int], Optional[PadInfo], float]:
    """Prepare NCHW float32 tensor.

    Returns: (tensor, resized_wh, pad_info, scale)
    - resized_wh: image size after resize (before pad)
    - pad_info: if padded to fixed/dynamic divisible size
    - scale: scale applied to original image
    """
    ow, oh = img_rgb.size
    fixed_w, fixed_h, fixed = _infer_target_hw(model, ref_long_side, div)
    pad_info: Optional[PadInfo] = None

    if fixed:
        # Resize to fixed input.
        img_r = img_rgb.resize((fixed_w, fixed_h), Image.BICUBIC)
        scale = fixed_w / float(ow)
        resized_wh = (fixed_w, fixed_h)
    else:
        # Resize by long side and pad to divisible.
        nw, nh, scale = _resize_keep_aspect(ow, oh, ref_long_side)
        img_r = img_rgb.resize((nw, nh), Image.BICUBIC)
        tw = _make_divisible(nw, div)
        th = _make_divisible(nh, div)
        if tw != nw or th != nh:
            img_r, pad_info = _letterbox(img_r, tw, th)
        resized_wh = (nw, nh)

    x = np.asarray(img_r).astype(np.float32) / 255.0  # HWC, 0..1
    if normalize == "-1..1":
        x = _normalize_minus1_to1(x)
    elif normalize == "none":
        pass
    else:
        # fallback
        x = _normalize_minus1_to1(x)
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    return x.astype(np.float32), resized_wh, pad_info, scale


def _postprocess_matte(
    matte: np.ndarray,
    original_size: Tuple[int, int],
    pad_info: Optional[PadInfo],
    resized_wh: Tuple[int, int],
    cutoff: float,
    gamma: float,
    feather_px: int,
    shrink_grow_px: int,
) -> Image.Image:
    """Convert model output to final alpha matte as PIL L image (0..255)."""
    matte = np.asarray(matte, dtype=np.float32)
    matte = np.nan_to_num(matte, nan=0.0, posinf=1.0, neginf=0.0)
    matte = np.clip(matte, 0.0, 1.0)

    # If padded, unpad.
    if pad_info is not None:
        matte = _unletterbox(matte, pad_info)

    # Resize to original.
    ow, oh = original_size
    # matte is in resized_wh
    mw, mh = resized_wh
    if matte.shape[1] != mw or matte.shape[0] != mh:
        # Some models output fixed size regardless. Resize matte to resized image size first.
        matte_img = Image.fromarray((matte * 255).astype(np.uint8), mode="L")
        matte_img = matte_img.resize((mw, mh), Image.BILINEAR)
    else:
        matte_img = Image.fromarray((matte * 255).astype(np.uint8), mode="L")

    matte_img = matte_img.resize((ow, oh), Image.BILINEAR)

    # Feather (blur) first.
    if feather_px > 0:
        matte_img = matte_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

    # Shrink/grow via min/max filter (cheap morphological op).
    if shrink_grow_px != 0:
        r = abs(int(shrink_grow_px))
        # Pillow Min/MaxFilter uses kernel size; must be odd.
        k = max(3, r * 2 + 1)
        if k % 2 == 0:
            k += 1
        if shrink_grow_px < 0:
            matte_img = matte_img.filter(ImageFilter.MinFilter(size=k))
        else:
            matte_img = matte_img.filter(ImageFilter.MaxFilter(size=k))

    # Cutoff + gamma shaping.
    matte_f = np.asarray(matte_img).astype(np.float32) / 255.0
    cutoff = float(np.clip(cutoff, 0.0, 0.999))
    matte_f = np.clip((matte_f - cutoff) / max(1e-6, 1.0 - cutoff), 0.0, 1.0)
    gamma = float(max(0.10, min(5.0, gamma)))
    matte_f = np.power(matte_f, gamma)
    matte_u8 = (matte_f * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(matte_u8, mode="L")


def remove_background(
    pil_image: Image.Image,
    model: OnnxModel,
    *,
    ref_long_side: int,
    normalize: str,
    cutoff: float,
    gamma: float,
    feather_px: int,
    shrink_grow_px: int,
) -> Tuple[Image.Image, Image.Image]:
    """Run background removal and return (rgba_image, matte_L)."""
    img_rgb = _pil_to_rgb(pil_image)
    ow, oh = img_rgb.size

    x, resized_wh, pad_info, _scale = _prepare_input(
        img_rgb,
        model,
        ref_long_side=ref_long_side,
        div=32,
        normalize=normalize,
    )

    matte = model.run(x)
    matte_L = _postprocess_matte(
        matte,
        original_size=(ow, oh),
        pad_info=pad_info,
        resized_wh=resized_wh,
        cutoff=cutoff,
        gamma=gamma,
        feather_px=int(feather_px),
        shrink_grow_px=int(shrink_grow_px),
    )

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(matte_L)
    return rgba, matte_L


# -------------------------
# UI Helpers
# -------------------------

def _qimage_from_pil(img: Image.Image) -> QtGui.QImage:
    # Ensure RGBA for display.
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, img.width, img.height, QtGui.QImage.Format_RGBA8888)
    # Deep copy so backing buffer isn't freed.
    return qimg.copy()


def _checker_brush(tile: int = 16) -> QtGui.QBrush:
    pm = QtGui.QPixmap(tile * 2, tile * 2)
    pm.fill(QtGui.QColor(230, 230, 230))
    p = QtGui.QPainter(pm)
    c1 = QtGui.QColor(230, 230, 230)
    c2 = QtGui.QColor(200, 200, 200)
    p.fillRect(0, 0, tile, tile, c2)
    p.fillRect(tile, tile, tile, tile, c2)
    p.fillRect(tile, 0, tile, tile, c1)
    p.fillRect(0, tile, tile, tile, c1)
    p.end()
    return QtGui.QBrush(pm)


class ImageView(QtWidgets.QGraphicsView):
    """Zoomable / pannable image view."""

    transformed = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(_checker_brush())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Wheel zoom (common editor behavior). Panning is via drag.
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.0 + (0.15 if angle > 0 else -0.15)
            self.scale(factor, factor)
            self.transformed.emit()
            event.accept()
            return
        super().wheelEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.transformed.emit()

    def fit_to_scene(self) -> None:
        r = self.sceneRect()
        if r.isNull() or r.width() < 2 or r.height() < 2:
            return
        self.fitInView(r, QtCore.Qt.KeepAspectRatio)
        self.transformed.emit()


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object, object, float, str)
    error = QtCore.Signal(str)


class BgRemoveWorker(QtCore.QRunnable):
    def __init__(self, pil_image: Image.Image, model: OnnxModel, settings: Dict[str, Any]):
        super().__init__()
        self.signals = _WorkerSignals()
        self.pil_image = pil_image
        self.model = model
        self.settings = settings

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            rgba, matte = remove_background(
                self.pil_image,
                self.model,
                ref_long_side=int(self.settings.get("ref_long_side", 1024)),
                normalize=str(self.settings.get("normalize", "-1..1")),
                cutoff=float(self.settings.get("cutoff", 0.15)),
                gamma=float(self.settings.get("gamma", 1.0)),
                feather_px=int(self.settings.get("feather_px", 6)),
                shrink_grow_px=int(self.settings.get("shrink_grow_px", 0)),
            )
            ms = (time.perf_counter() - t0) * 1000.0
            self.signals.finished.emit(rgba, matte, ms, self.model.label)
        except Exception as e:
            self.signals.error.emit(str(e))


# -------------------------
# Main Pane (embeddable)
# -------------------------

class BgRemovePane(QtWidgets.QWidget):
    """Embeddable background remover pane."""

    status = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        root = _project_root()
        self._models_dir = root / "models" / "bg"
        self._settings_path = root / "presets" / "setsave" / "bg_remove.json"

        # Default save location (used when no last_save_dir is stored yet)
        self._default_save_dir = root / "output" / "images" / "rm_backgrounds"
        try:
            self._default_save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.models: Dict[str, OnnxModel] = {
            "BiRefNet (general objects, slow)": OnnxModel(self._models_dir / "BiRefNet-COD-epoch_125.onnx", "BiRefNet"),
            "MODNet (portraits, fast)": OnnxModel(self._models_dir / "modnet.onnx", "MODNet"),
        }

        defaults = {
            "model_key": "BiRefNet (general objects)",
            "ref_long_side": 1024,
            "normalize": "-1..1",
            "cutoff": 0.15,
            "gamma": 1.0,
            "feather_px": 6,
            "shrink_grow_px": 0,
            "preview_bg": "checker",
            "preview_color": "#ffffff",
            "last_open_dir": "",
            "last_save_dir": "",
            "splitter_v_sizes": [],
            "splitter_h_sizes": [],
        }
        self.settings = SettingsStore(self._settings_path, defaults)

        self._threadpool = QtCore.QThreadPool.globalInstance()

        self._pil_original: Optional[Image.Image] = None
        self._pil_result: Optional[Image.Image] = None
        self._pil_matte: Optional[Image.Image] = None
        self._current_path: Optional[Path] = None

        self._build_ui()
        self._wire()
        self._apply_settings_to_ui(self.settings.data)
        self._refresh_model_status()

    # ---- UI

    def _build_ui(self) -> None:
        self.setObjectName("BgRemovePane")
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # Top bar (two lines)
        # Line 1: Open/Run + Fit
        # Line 2: Save actions (and optional injected "View results")
        top_box = QtWidgets.QWidget()
        top_v = QtWidgets.QVBoxLayout(top_box)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.setSpacing(6)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)

        self.btn_open = QtWidgets.QPushButton("Open image")
        self.btn_run = QtWidgets.QPushButton("Remove background")
        self.btn_save_png = QtWidgets.QPushButton("Save PNG")
        self.btn_save_mask = QtWidgets.QPushButton("Save mask")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setToolTip("Fit both previews to image")
        self.btn_run.setDefault(True)

        row1.addWidget(self.btn_open)
        row1.addWidget(self.btn_run)
        row1.addStretch(1)
        row1.addWidget(self.btn_fit)

        row2.addStretch(1)
        row2.addWidget(self.btn_save_png)
        row2.addWidget(self.btn_save_mask)

        top_v.addLayout(row1)
        top_v.addLayout(row2)
        main.addWidget(top_box)

        # Expose top bar rows for FrameVision integration (injecting extra buttons cleanly).
        self._top_row1 = row1
        self._top_row2 = row2

        # Main content: previews above settings (resizable splitter)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split_v = split

        # Top: previews
        previews = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(previews)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)

        titles = QtWidgets.QHBoxLayout()
        self.lbl_left = QtWidgets.QLabel("Original")
        self.lbl_right = QtWidgets.QLabel("Result")
        self.lbl_left.setStyleSheet("font-weight: 600;")
        self.lbl_right.setStyleSheet("font-weight: 600;")
        titles.addWidget(self.lbl_left)
        titles.addStretch(1)
        titles.addWidget(self.lbl_right)
        pv.addLayout(titles)

        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.setSpacing(6)
        self.btn_zoom_out = QtWidgets.QPushButton("-")
        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_reset = QtWidgets.QPushButton("100%")
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_reset):
            b.setFixedWidth(48)

        self.sl_zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_zoom.setRange(10, 800)
        self.sl_zoom.setSingleStep(10)
        self.sl_zoom.setPageStep(50)
        self.sl_zoom.setValue(100)
        self.sl_zoom.setToolTip("Zoom (mouse wheel also zooms).")

        self.lbl_zoom = QtWidgets.QLabel("100%")
        self.lbl_zoom.setMinimumWidth(56)
        self.lbl_zoom.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_zoom.setStyleSheet("color: #888;")

        zoom_row.addWidget(QtWidgets.QLabel("Zoom"))
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.sl_zoom, 1)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_reset)
        zoom_row.addWidget(self.lbl_zoom)
        pv.addLayout(zoom_row)

        self.view_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.scene_a = QtWidgets.QGraphicsScene(self)
        self.scene_b = QtWidgets.QGraphicsScene(self)
        self.view_a = ImageView()
        self.view_b = ImageView()
        self.view_a.setScene(self.scene_a)
        self.view_b.setScene(self.scene_b)
        self.view_split.addWidget(self.view_a)
        self.view_split.addWidget(self.view_b)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 1)
        pv.addWidget(self.view_split, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        pv.addWidget(self.progress)

        split.addWidget(previews)

        # Bottom: settings
        left = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(left)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(8)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(list(self.models.keys()))

        self.sp_ref = QtWidgets.QSpinBox()
        self.sp_ref.setRange(256, 2048)
        self.sp_ref.setSingleStep(64)
        self.sp_ref.setToolTip("Long-side resize before inference. Higher = sharper edges but slower.")

        self.cmb_norm = QtWidgets.QComboBox()
        self.cmb_norm.addItems(["-1..1", "none"])
        self.cmb_norm.setToolTip("Most matting models expect -1..1 normalization.")

        self.sl_cut = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_cut.setRange(0, 80)
        self.sl_cut.setToolTip("Foreground cutoff. Higher removes more background but can eat edges.")
        self.lbl_cut = QtWidgets.QLabel("0.15")

        self.sl_gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_gamma.setRange(10, 300)
        self.sl_gamma.setToolTip("Edge shaping. < 1 keeps more foreground; > 1 tightens.")
        self.lbl_gamma = QtWidgets.QLabel("1.00")

        self.sl_feather = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_feather.setRange(0, 50)
        self.sl_feather.setToolTip("Feather radius (blur) for smoother edges.")
        self.lbl_feather = QtWidgets.QLabel("6 px")

        self.sl_shrink = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_shrink.setRange(-30, 30)
        self.sl_shrink.setToolTip("Shrink (<0) or grow (>0) the matte.")
        self.lbl_shrink = QtWidgets.QLabel("0 px")

        self.cmb_preview_bg = QtWidgets.QComboBox()
        self.cmb_preview_bg.addItems(["checker", "white", "black", "color"])
        self.btn_color = QtWidgets.QPushButton("Pick color")
        self.btn_color.setToolTip("Used for 'color' preview background.")

        form.addRow("Model", self.cmb_model)
        form.addRow("Quality", self.sp_ref)
        form.addRow("Normalize", self.cmb_norm)

        row_cut = QtWidgets.QHBoxLayout()
        row_cut.addWidget(self.sl_cut, 1)
        row_cut.addWidget(self.lbl_cut)
        form.addRow("Cutoff", row_cut)

        row_gamma = QtWidgets.QHBoxLayout()
        row_gamma.addWidget(self.sl_gamma, 1)
        row_gamma.addWidget(self.lbl_gamma)
        form.addRow("Gamma", row_gamma)

        row_f = QtWidgets.QHBoxLayout()
        row_f.addWidget(self.sl_feather, 1)
        row_f.addWidget(self.lbl_feather)
        form.addRow("Feather", row_f)

        row_s = QtWidgets.QHBoxLayout()
        row_s.addWidget(self.sl_shrink, 1)
        row_s.addWidget(self.lbl_shrink)
        form.addRow("Shrink/Grow", row_s)

        row_bg = QtWidgets.QHBoxLayout()
        row_bg.addWidget(self.cmb_preview_bg, 1)
        row_bg.addWidget(self.btn_color)
        form.addRow("Preview BG", row_bg)

        self.lbl_info = QtWidgets.QLabel("Open an image to begin.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #888;")
        form.addRow(self.lbl_info)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left)

        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        main.addWidget(split, 1)

        # Status bar
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color: #888;")
        main.addWidget(self.lbl_status)

        self._pix_a: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._pix_b: Optional[QtWidgets.QGraphicsPixmapItem] = None

        self._updating_zoom = False

        # Restore splitter sizes (preview pane + left/right views)
        QtCore.QTimer.singleShot(0, self._restore_splitters)

    def _wire(self) -> None:
        self.btn_open.clicked.connect(self.open_image_dialog)
        self.btn_run.clicked.connect(self.run_remove)
        self.btn_save_png.clicked.connect(self.save_png_dialog)
        self.btn_save_mask.clicked.connect(self.save_mask_dialog)
        self.btn_fit.clicked.connect(self._fit_views)

        # Zoom controls
        self.sl_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        self.btn_zoom_out.clicked.connect(lambda: self._nudge_zoom(-10))
        self.btn_zoom_in.clicked.connect(lambda: self._nudge_zoom(10))
        self.btn_zoom_reset.clicked.connect(lambda: self._set_zoom_percent(100))

        self.cmb_model.currentTextChanged.connect(lambda v: self.settings.set("model_key", v))
        self.sp_ref.valueChanged.connect(lambda v: self.settings.set("ref_long_side", int(v)))
        self.cmb_norm.currentTextChanged.connect(lambda v: self.settings.set("normalize", v))
        self.cmb_preview_bg.currentTextChanged.connect(lambda v: self.settings.set("preview_bg", v))
        self.btn_color.clicked.connect(self._pick_color)

        self.sl_cut.valueChanged.connect(self._on_cutoff_changed)
        self.sl_gamma.valueChanged.connect(self._on_gamma_changed)
        self.sl_feather.valueChanged.connect(self._on_feather_changed)
        self.sl_shrink.valueChanged.connect(self._on_shrink_changed)

        self.settings.changed.connect(self._apply_settings_to_ui)

        # Persist splitter sizes (preview pane height + left/right preview widths)
        try:
            self.split_v.splitterMoved.connect(lambda *_: self.settings.set("splitter_v_sizes", list(self.split_v.sizes())))
        except Exception:
            pass
        try:
            self.view_split.splitterMoved.connect(lambda *_: self.settings.set("splitter_h_sizes", list(self.view_split.sizes())))
        except Exception:
            pass

        # Sync transforms between views.
        self._syncing = False
        self.view_a.transformed.connect(lambda: self._sync_views(self.view_a, self.view_b))
        self.view_b.transformed.connect(lambda: self._sync_views(self.view_b, self.view_a))

        # Keep zoom UI in sync with wheel zoom / fit.
        self.view_a.transformed.connect(self._on_view_transformed)
        self.view_b.transformed.connect(self._on_view_transformed)

        # Initial zoom display
        QtCore.QTimer.singleShot(0, lambda: self._update_zoom_ui(self._current_zoom_percent()))


    # ---- Splitter persistence

    def _restore_splitters(self) -> None:
        """Restore splitter sizes from settings (best-effort)."""
        s = self.settings.data

        vs = s.get("splitter_v_sizes")
        if isinstance(vs, (list, tuple)) and len(vs) >= 2:
            try:
                a, b = int(vs[0]), int(vs[1])
                if a > 0 and b > 0:
                    self.split_v.setSizes([a, b])
            except Exception:
                pass

        hs = s.get("splitter_h_sizes")
        if isinstance(hs, (list, tuple)) and len(hs) >= 2:
            try:
                a, b = int(hs[0]), int(hs[1])
                if a > 0 and b > 0:
                    self.view_split.setSizes([a, b])
            except Exception:
                pass

    # ---- Zoom

    def _current_zoom_percent(self) -> int:
        try:
            z = float(self.view_a.transform().m11())
        except Exception:
            z = 1.0
        return int(round(z * 100.0))

    def _clamp_zoom_percent(self, pct: int) -> int:
        return max(self.sl_zoom.minimum(), min(self.sl_zoom.maximum(), int(pct)))

    def _update_zoom_ui(self, pct: int) -> None:
        pct = self._clamp_zoom_percent(pct)
        self._updating_zoom = True
        try:
            self.lbl_zoom.setText(f"{pct}%")
            b = QtCore.QSignalBlocker(self.sl_zoom)
            _ = b
            self.sl_zoom.setValue(pct)
        finally:
            self._updating_zoom = False

        # Restore splitter sizes (preview pane + left/right views)
        QtCore.QTimer.singleShot(0, self._restore_splitters)

    @QtCore.Slot()
    def _on_view_transformed(self) -> None:
        if self._updating_zoom:
            return
        src = self.sender()
        if isinstance(src, ImageView):
            try:
                pct = int(round(float(src.transform().m11()) * 100.0))
            except Exception:
                pct = 100
        else:
            pct = self._current_zoom_percent()
        self._update_zoom_ui(pct)

    def _set_zoom_percent(self, pct: int) -> None:
        pct = self._clamp_zoom_percent(pct)
        cur_scale = float(self.view_a.transform().m11()) if self.view_a is not None else 1.0
        cur_scale = max(1e-6, cur_scale)
        desired_scale = pct / 100.0
        factor = desired_scale / cur_scale
        self.view_a.scale(factor, factor)
        self._sync_views(self.view_a, self.view_b, force=True)
        self._update_zoom_ui(pct)

    def _nudge_zoom(self, delta_pct: int) -> None:
        self._set_zoom_percent(self._current_zoom_percent() + int(delta_pct))

    def _on_zoom_slider_changed(self, v: int) -> None:
        if self._updating_zoom:
            return
        self._set_zoom_percent(int(v))

    # ---- Settings <-> UI

    def _apply_settings_to_ui(self, s: Dict[str, Any]) -> None:
        # Avoid signal loops by blocking while applying.
        b1 = QtCore.QSignalBlocker(self.cmb_model)
        b2 = QtCore.QSignalBlocker(self.sp_ref)
        b3 = QtCore.QSignalBlocker(self.cmb_norm)
        b4 = QtCore.QSignalBlocker(self.cmb_preview_bg)
        b5 = QtCore.QSignalBlocker(self.sl_cut)
        b6 = QtCore.QSignalBlocker(self.sl_gamma)
        b7 = QtCore.QSignalBlocker(self.sl_feather)
        b8 = QtCore.QSignalBlocker(self.sl_shrink)
        _ = (b1, b2, b3, b4, b5, b6, b7, b8)

        model_key = str(s.get("model_key", ""))
        idx = self.cmb_model.findText(model_key)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)

        self.sp_ref.setValue(int(s.get("ref_long_side", 1024)))

        norm = str(s.get("normalize", "-1..1"))
        idx = self.cmb_norm.findText(norm)
        if idx >= 0:
            self.cmb_norm.setCurrentIndex(idx)

        # sliders
        cutoff = float(s.get("cutoff", 0.15))
        self.sl_cut.setValue(int(round(cutoff * 100)))
        self.lbl_cut.setText(f"{cutoff:.2f}")

        gamma = float(s.get("gamma", 1.0))
        self.sl_gamma.setValue(int(round(gamma * 100)))
        self.lbl_gamma.setText(f"{gamma:.2f}")

        feather = int(s.get("feather_px", 6))
        self.sl_feather.setValue(feather)
        self.lbl_feather.setText(f"{feather} px")

        sg = int(s.get("shrink_grow_px", 0))
        self.sl_shrink.setValue(sg)
        self.lbl_shrink.setText(f"{sg} px")

        bg = str(s.get("preview_bg", "checker"))
        idx = self.cmb_preview_bg.findText(bg)
        if idx >= 0:
            self.cmb_preview_bg.setCurrentIndex(idx)
        self._apply_preview_bg()

    def _on_cutoff_changed(self, v: int) -> None:
        cutoff = float(v) / 100.0
        self.lbl_cut.setText(f"{cutoff:.2f}")
        self.settings.set("cutoff", cutoff)

    def _on_gamma_changed(self, v: int) -> None:
        gamma = float(v) / 100.0
        self.lbl_gamma.setText(f"{gamma:.2f}")
        self.settings.set("gamma", gamma)

    def _on_feather_changed(self, v: int) -> None:
        self.lbl_feather.setText(f"{v} px")
        self.settings.set("feather_px", int(v))

    def _on_shrink_changed(self, v: int) -> None:
        self.lbl_shrink.setText(f"{v} px")
        self.settings.set("shrink_grow_px", int(v))

    # ---- Preview background

    def _apply_preview_bg(self) -> None:
        s = self.settings.data
        mode = str(s.get("preview_bg", "checker"))
        if mode == "checker":
            brush = _checker_brush()
        elif mode == "white":
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        elif mode == "black":
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        else:
            col = QtGui.QColor(str(s.get("preview_color", "#ffffff")))
            brush = QtGui.QBrush(col)
        self.view_a.setBackgroundBrush(brush)
        self.view_b.setBackgroundBrush(brush)
        self.view_a.viewport().update()
        self.view_b.viewport().update()

    def _pick_color(self) -> None:
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self.settings.data.get("preview_color", "#ffffff")),
            self,
            "Pick preview background color",
        )
        if not col.isValid():
            return
        self.settings.set("preview_color", col.name())
        self.settings.set("preview_bg", "color")
        self._apply_preview_bg()

    # ---- Model status

    def _refresh_model_status(self) -> None:
        missing = [k for k, m in self.models.items() if not m.is_available()]
        if missing:
            self.lbl_info.setText(
                "Missing model files:\n"
                + "\n".join(f"- {self.models[k].path.name}" for k in missing)
                + "\n\nExpected location: models/bg/"
            )
            self.lbl_info.setStyleSheet("color: #b66;")
            self.btn_run.setEnabled(False)
        else:
            self.lbl_info.setText("Tip: Start with MODNet for portraits, BiRefNet for general objects.")
            self.lbl_info.setStyleSheet("color: #888;")
            self.btn_run.setEnabled(True)

    # ---- Load/Save

    def open_image_dialog(self) -> None:
        start_dir = self.settings.data.get("last_open_dir") or str(_project_root())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        self.settings.set("last_open_dir", str(p.parent))
        self.load_image(p)

    def load_image(self, path: Path) -> None:
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            self._set_status(f"Failed to open image: {e}")
            return
        self._current_path = path
        self._pil_original = img
        self._pil_result = None
        self._pil_matte = None
        self._show_original(img)
        self._show_result(None)
        self._set_status(f"Loaded: {path.name}  ({img.width}x{img.height})")
        self._fit_views()

    def save_png_dialog(self) -> None:
        if self._pil_result is None:
            self._set_status("Nothing to save. Run background removal first.")
            return
        start_dir = self.settings.data.get("last_save_dir") or str(self._default_save_dir)
        default_name = "result.png"
        if self._current_path:
            default_name = f"{self._current_path.stem}_cutout.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            str(Path(start_dir) / default_name),
            "PNG (*.png)",
        )
        if not path:
            return
        p = Path(path)
        try:
            self._pil_result.save(p, format="PNG")
            self.settings.set("last_save_dir", str(p.parent))
            self._set_status(f"Saved: {p.name}")
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    def save_mask_dialog(self) -> None:
        if self._pil_matte is None:
            self._set_status("No mask available. Run background removal first.")
            return
        start_dir = self.settings.data.get("last_save_dir") or str(self._default_save_dir)
        default_name = "mask.png"
        if self._current_path:
            default_name = f"{self._current_path.stem}_mask.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save mask",
            str(Path(start_dir) / default_name),
            "PNG (*.png)",
        )
        if not path:
            return
        p = Path(path)
        try:
            self._pil_matte.save(p, format="PNG")
            self.settings.set("last_save_dir", str(p.parent))
            self._set_status(f"Saved: {p.name}")
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    # ---- Run

    def run_remove(self) -> None:
        if self._pil_original is None:
            self._set_status("Open an image first.")
            return
        model_key = self.settings.data.get("model_key")
        model = self.models.get(str(model_key))
        if model is None:
            self._set_status("Invalid model selection.")
            return
        if not model.is_available():
            self._refresh_model_status()
            return

        self.progress.setVisible(True)
        self.btn_run.setEnabled(False)
        self._set_status("Running background removal...")

        worker = BgRemoveWorker(self._pil_original, model, self.settings.data)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        self._threadpool.start(worker)

    @QtCore.Slot(object, object, float, str)
    def _on_done(self, rgba: Image.Image, matte: Image.Image, ms: float, model_label: str) -> None:
        self._pil_result = rgba
        self._pil_matte = matte
        self._show_result(rgba)
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self._set_status(f"Done ({model_label}) in {ms:.0f} ms")
        self._fit_views()

    @QtCore.Slot(str)
    def _on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self._set_status(f"Error: {msg}")

    # ---- Scene updates

    def _show_original(self, img: Image.Image) -> None:
        self.scene_a.clear()
        qimg = _qimage_from_pil(_pil_to_rgb(img).convert("RGBA"))
        pix = QtGui.QPixmap.fromImage(qimg)
        self._pix_a = self.scene_a.addPixmap(pix)
        self.scene_a.setSceneRect(QtCore.QRectF(pix.rect()))

    def _show_result(self, img: Optional[Image.Image]) -> None:
        self.scene_b.clear()
        if img is None:
            self._pix_b = None
            self.scene_b.setSceneRect(QtCore.QRectF(0, 0, 1, 1))
            return
        qimg = _qimage_from_pil(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        self._pix_b = self.scene_b.addPixmap(pix)
        self.scene_b.setSceneRect(QtCore.QRectF(pix.rect()))

    def _fit_views(self) -> None:
        # Fit both, then sync transforms.
        self.view_a.fit_to_scene()
        self._sync_views(self.view_a, self.view_b, force=True)
        self._update_zoom_ui(self._current_zoom_percent())

    def _sync_views(self, src: ImageView, dst: ImageView, force: bool = False) -> None:
        if self._syncing and not force:
            return
        try:
            self._syncing = True
            dst.setTransform(src.transform())
            dst.horizontalScrollBar().setValue(src.horizontalScrollBar().value())
            dst.verticalScrollBar().setValue(src.verticalScrollBar().value())
        finally:
            self._syncing = False

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.status.emit(text)



# -------------------------
# FrameVision integration shim
# -------------------------
# This file is used both as a standalone tool and as a Tools-tab module inside FrameVision.
# The host app expects two public entry points:
#   - install_background_tool(pane, section_widget)
#   - remove_background_file(...)
#
# The standalone widget (BgRemovePane) remains the main implementation; the shim below
# adapts it to the Tools tab and provides a small compatibility one-shot API.

import os as _os

def _fv_project_root() -> Path:
    # Prefer the app's ROOT if it is already loaded (avoid circular imports).
    try:
        import sys as _sys
        _mod = _sys.modules.get("helpers.framevision_app")
        if _mod is not None and hasattr(_mod, "ROOT"):
            return Path(getattr(_mod, "ROOT"))
    except Exception:
        pass
    return _project_root()


# Common output folders (used by remove_background_file)
ROOT = _fv_project_root()
OUT_SHOTS = ROOT / "output" / "images" / "rm_backgrounds"
OUT_TEMP = ROOT / "output" / "_temp"
try:
    OUT_SHOTS.mkdir(parents=True, exist_ok=True)
    OUT_TEMP.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


def _ffmpeg_path() -> str:
    import os as _os
    win = _os.name == "nt"
    names = ["ffmpeg.exe"] if win else ["ffmpeg"]
    roots = [
        ROOT / "presets" / "bin",
        ROOT / "presets" / "tools",
        ROOT / "bin",
        ROOT,
    ]
    for base in roots:
        for n in names:
            cand = base / n
            try:
                if cand.exists():
                    return str(cand)
            except Exception:
                pass
    return "ffmpeg"


def _extract_video_frame(path: Path, time_s: float = 0.0) -> Optional[Image.Image]:
    import subprocess as _sp
    import tempfile as _tf

    out_png = Path(_tf.gettempdir()) / f"fv_bgtool_frame_{_os.getpid()}.png"
    cmd = [_ffmpeg_path(), "-y", "-ss", f"{max(0.0, float(time_s)):.3f}", "-i", str(path), "-frames:v", "1", str(out_png)]
    try:
        _sp.run(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, check=True)
        img = Image.open(out_png)
        img.load()
        return img
    except Exception:
        return None


def _current_media_from_app(pane) -> Tuple[Optional[Path], float]:
    """Best-effort: find current media path and current time (seconds) from the host app."""
    # Known attribute patterns in FrameVision:
    candidates = [
        getattr(pane, "current_path", None),
        getattr(getattr(pane, "main", None), "current_path", None),
    ]
    pth: Optional[Path] = None
    for c in candidates:
        if c:
            try:
                p = Path(str(c))
                if p.exists():
                    pth = p
                    break
            except Exception:
                pass

    # Time is optional; default to 0.
    t = 0.0
    try:
        main = getattr(pane, "main", None)
        vid = getattr(main, "video", None) if main is not None else None
        # Some players expose seconds; ignore if missing.
        for attr in ("current_time_s", "time_s", "pos_s", "position_s"):
            if vid is not None and hasattr(vid, attr):
                try:
                    t = float(getattr(vid, attr))
                    break
                except Exception:
                    pass
    except Exception:
        pass

    return pth, t


def _try_grab_qimage_from_host(pane) -> Optional["QtGui.QImage"]:
    """Try to grab the currently shown video frame as a QImage (if the host exposes one)."""
    try:
        main = getattr(pane, "main", None)
        vid = getattr(main, "video", None) if main is not None else None
        if vid is None:
            return None

        qimg = getattr(vid, "currentFrame", None)
        if qimg is not None:
            try:
                if hasattr(qimg, "isNull") and qimg.isNull():
                    qimg = None
            except Exception:
                pass
            if qimg is not None:
                return qimg

        # Fallback: pixmap from a QLabel
        lab = getattr(vid, "label", None)
        if lab is not None and hasattr(lab, "pixmap"):
            pm = lab.pixmap()
            if pm is not None:
                return pm.toImage()
    except Exception:
        pass
    return None


def _load_current_image_from_host(pane) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Return (PIL image, label) or (None, None).

    IMPORTANT: We prefer FULL-RES sources from disk (image path / ffmpeg video frame)
    over any UI snapshot (which is often downscaled to the preview widget).
    """
    # 1) Try to resolve the currently-open media path from the host app.
    media_path, time_s = _current_media_from_app(pane)

    # 1a) If it's an image file, load directly from disk (full resolution).
    if media_path is not None:
        try:
            if media_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                im = Image.open(media_path)
                im.load()
                return im, media_path.name
        except Exception:
            pass

        # 1b) If it's a video, extract a frame with ffmpeg at the current timestamp (full resolution).
        try:
            im = _extract_video_frame(media_path, time_s=time_s)
            if im is not None:
                return im, f"{media_path.name} @ {time_s:.2f}s"
        except Exception:
            pass

    # 2) Fallback: grab whatever the host is currently showing (often downscaled).
    try:
        qimg = _try_grab_qimage_from_host(pane)
        if qimg is not None:
            import tempfile as _tf
            from PySide6.QtGui import QPixmap as _QPixmap
            tmp = Path(_tf.gettempdir()) / f"fv_bgtool_qframe_{_os.getpid()}.png"
            _QPixmap.fromImage(qimg).save(str(tmp), "PNG")
            im = Image.open(tmp)
            im.load()
            return im, "current preview"
    except Exception:
        pass

    return None, None


def _bgpane_load_pil(self: "BgRemovePane", img: Image.Image, label: str = "image") -> None:
    """Load a PIL image directly into the pane (FrameVision 'Use current')."""
    try:
        img.load()
    except Exception:
        pass
    self._current_path = None
    self._pil_original = img
    self._pil_result = None
    self._pil_matte = None
    self._show_original(img)
    self._show_result(None)
    try:
        self._set_status(f"Loaded: {label}  ({img.width}x{img.height})")
    except Exception:
        self._set_status("Loaded")
    self._fit_views()


# Attach method without changing the class body (keeps standalone code intact).
try:
    if not hasattr(BgRemovePane, "load_pil"):
        BgRemovePane.load_pil = _bgpane_load_pil  # type: ignore[attr-defined]
except Exception:
    pass


def remove_background_file(
    inp_path: str,
    engine: str = "auto",
    mode: str = "keep_subject",
    threshold: int = 0,
    feather: int = 6,
    spill: bool = True,
    invert: bool = False,
    auto_level: bool = True,
    bias: int = 50,
    out_dir: Optional[str] = None,
    # Replacer (ignored here; kept for compatibility):
    repl_mode: str = "transparent",
    repl_color: Tuple[int, int, int] = (255, 255, 255),
    repl_blur: int = 20,
    repl_image: Optional[str] = None,
    drop_shadow: bool = False,
    shadow_alpha: int = 120,
    shadow_blur: int = 20,
    shadow_offx: int = 10,
    shadow_offy: int = 10,
    anti_halo: bool = True,
) -> Path:
    """One-shot API compatible with the old tool.

    Note: the new background remover focuses on producing a clean alpha cutout/matte.
    Background replacement/inpaint options from the legacy tool are not implemented here.
    """
    p = Path(inp_path)
    out_dir_p = Path(out_dir) if out_dir else OUT_SHOTS
    try:
        out_dir_p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    models_dir = ROOT / "models" / "bg"
    mod_modnet = OnnxModel(models_dir / "modnet.onnx", "MODNet")
    mod_biref = OnnxModel(models_dir / "BiRefNet-COD-epoch_125.onnx", "BiRefNet")

    eng = (engine or "auto").lower().strip()
    if eng in {"modnet", "portrait"}:
        model = mod_modnet
    elif eng in {"birefnet", "biref", "general"}:
        model = mod_biref
    else:
        # auto: pick first available
        model = mod_biref if mod_biref.is_available() else mod_modnet

    img = Image.open(p)
    img.load()

    rgba, matte = remove_background(
        img,
        model,
        ref_long_side=1024,
        normalize="-1..1",
        cutoff=0.15,
        gamma=1.0,
        feather_px=int(feather),
        shrink_grow_px=0,
    )

    # Convert matte (0..255) if needed
    try:
        a = np.asarray(matte.convert("L"), dtype=np.uint8)
    except Exception:
        a = np.asarray(matte, dtype=np.uint8)

    if invert:
        a = 255 - a

    if threshold and int(threshold) > 0:
        thr = int(threshold)
        a = np.where(a >= thr, a, 0).astype(np.uint8)

    if mode == "alpha_only":
        out_img = Image.fromarray(a, mode="L")
        suffix = "alpha"
    elif mode == "keep_bg":
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        inv = (255 - a).astype(np.float32) / 255.0
        bg_only = (rgb.astype(np.float32) * inv[..., None]).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(bg_only, mode="RGB")
        suffix = "bgonly"
    else:
        # keep_subject: apply updated alpha into the result
        r = np.asarray(rgba.convert("RGBA"), dtype=np.uint8)
        r[..., 3] = a
        out_img = Image.fromarray(r, mode="RGBA")
        suffix = "cutout"

    out = out_dir_p / f"{p.stem}_{suffix}.png"
    out_img.save(out)
    return out


def install_background_tool(pane, section_widget) -> None:
    """Tools-tab entry point expected by helpers.tools_tab."""
    try:
        from PySide6 import QtWidgets as _QtW
        from PySide6 import QtCore as _QtC
    except Exception:
        return

    # Make sure this tool can expand properly inside Tools tab without forcing the whole
    # section to reserve height when collapsed (same trick as the legacy tool).
    try:
        if hasattr(section_widget, "content"):
            try:
                section_widget.content.setProperty("expand_min_h", 520)
            except Exception:
                pass
            try:
                section_widget.content.setSizePolicy(_QtW.QSizePolicy.Expanding, _QtW.QSizePolicy.Preferred)
            except Exception:
                pass
            try:
                section_widget.setMinimumHeight(0)
            except Exception:
                pass
        else:
            section_widget.setSizePolicy(_QtW.QSizePolicy.Expanding, _QtW.QSizePolicy.Expanding)
            section_widget.setMinimumHeight(520)
    except Exception:
        pass

    container = _QtW.QWidget()
    outer = _QtW.QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(6)

    bg_pane = BgRemovePane(container)

    # Inject "Use current" + "View results" buttons into the pane's top bar.
    btn_use_current = _QtW.QPushButton("Use current")
    btn_use_current.setToolTip("Use the image (or current video frame) already loaded in FrameVision.")

    btn_view_results = _QtW.QPushButton("View results")
    btn_view_results.setToolTip("Open this tool's output folder in Media Explorer (inside FrameVision).")

    # Prefer putting these into the existing top bar rows (BgRemovePane exposes _top_row1/_top_row2).
    try:
        r1 = getattr(bg_pane, "_top_row1", None)
        r2 = getattr(bg_pane, "_top_row2", None)

        if r1 is not None:
            try:
                r1.insertWidget(0, btn_use_current)
            except Exception:
                r1.addWidget(btn_use_current)
        else:
            # Fallback: try the first layout item if it's a QHBoxLayout (older versions).
            top_item = bg_pane.layout().itemAt(0)
            top_lay = top_item.layout() if top_item is not None else None
            if top_lay is not None:
                try:
                    top_lay.insertWidget(0, btn_use_current)
                except Exception:
                    top_lay.addWidget(btn_use_current)
            else:
                row = _QtW.QHBoxLayout()
                row.addWidget(btn_use_current)
                row.addStretch(1)
                outer.addLayout(row)

        if r2 is not None:
            # "View results" should sit with the save buttons on line 2.
            r2.addWidget(btn_view_results)
        else:
            # Fallback: append to the same row we used above.
            top_item = bg_pane.layout().itemAt(0)
            top_lay = top_item.layout() if top_item is not None else None
            if top_lay is not None:
                top_lay.addWidget(btn_view_results)
            else:
                row = _QtW.QHBoxLayout()
                row.addStretch(1)
                row.addWidget(btn_view_results)
                outer.addLayout(row)
    except Exception:
        row = _QtW.QHBoxLayout()
        row.addWidget(btn_use_current)
        row.addStretch(1)
        row.addWidget(btn_view_results)
        outer.addLayout(row)


    def _on_use_current() -> None:
        im, label = _load_current_image_from_host(pane)
        if im is None:
            try:
                _QtW.QMessageBox.information(container, "No image", "Open an image or video in the app first.")
            except Exception:
                pass
            return
        try:
            bg_pane.load_pil(im, label or "current")  # type: ignore[attr-defined]
        except Exception:
            # last-resort: try saving to a temp file then load by path
            try:
                import tempfile as _tf
                tmp = Path(_tf.gettempdir()) / f"fv_bgtool_current_{_os.getpid()}.png"
                im.convert("RGBA").save(tmp)
                bg_pane.load_image(tmp)
            except Exception:
                pass

    def _resolve_main_window():
        # Most builds: tools host exposes .main -> MainWindow.
        try:
            m = getattr(pane, "main", None)
            if m is not None and hasattr(m, "open_media_explorer_folder"):
                return m
        except Exception:
            pass

        # Sometimes the main window is the QWidget window() parent.
        try:
            w = pane.window() if hasattr(pane, "window") else None
            if w is not None and hasattr(w, "open_media_explorer_folder"):
                return w
            m2 = getattr(w, "main", None) if w is not None else None
            if m2 is not None and hasattr(m2, "open_media_explorer_folder"):
                return m2
        except Exception:
            pass

        # Last-resort: active window.
        try:
            aw = _QtW.QApplication.activeWindow()
            if aw is not None and hasattr(aw, "open_media_explorer_folder"):
                return aw
        except Exception:
            pass
        return None

    def _results_folder() -> Path:
        try:
            d = str(bg_pane.settings.data.get("last_save_dir") or "").strip()
        except Exception:
            d = ""
        p = Path(d) if d else getattr(bg_pane, "_default_save_dir", (_project_root() / "output" / "images" / "rm_backgrounds"))
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def _on_view_results() -> None:
        folder = _results_folder()
        mw = _resolve_main_window()
        if mw is not None and hasattr(mw, "open_media_explorer_folder"):
            try:
                mw.open_media_explorer_folder(
                    str(folder),
                    preset="images",
                    include_subfolders=False,
                    activate=True,
                    rescan=True,
                    clear_first=True,
                )
                return
            except Exception:
                pass
        try:
            _QtW.QMessageBox.information(container, "View results", "Media Explorer is not available in this build.")
        except Exception:
            pass

    btn_use_current.clicked.connect(_on_use_current)
    btn_view_results.clicked.connect(_on_view_results)

    # Keep it nicely resizable inside the tools tab.
    try:
        bg_pane.setSizePolicy(_QtW.QSizePolicy.Expanding, _QtW.QSizePolicy.Expanding)
    except Exception:
        pass

    outer.addWidget(bg_pane, stretch=1)

    # Mount into the collapsible section.
    try:
        v = _QtW.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(container)
        section_widget.setContentLayout(v)
    except Exception:
        try:
            section_widget.setLayout(_QtW.QVBoxLayout())
            section_widget.layout().setContentsMargins(0, 0, 0, 0)
            section_widget.layout().addWidget(container)
        except Exception:
            pass

    # Prevent GC
    try:
        section_widget._bg_tool_container = container
        section_widget._bg_tool_pane = bg_pane
    except Exception:
        pass


# -------------------------
# Standalone test window
# -------------------------

class BgRemoveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Remover")
        self.resize(1200, 720)
        self.pane = BgRemovePane(self)
        self.setCentralWidget(self.pane)

        # Basic menu
        file_menu = self.menuBar().addMenu("File")
        act_open = QtGui.QAction("Open...", self)
        act_open.setShortcut(QtGui.QKeySequence.Open)
        act_open.triggered.connect(self.pane.open_image_dialog)
        file_menu.addAction(act_open)

        act_save = QtGui.QAction("Save PNG...", self)
        act_save.setShortcut(QtGui.QKeySequence.Save)
        act_save.triggered.connect(self.pane.save_png_dialog)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_quit = QtGui.QAction("Quit", self)
        act_quit.setShortcut(QtGui.QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)


def main() -> None:
    app = QtWidgets.QApplication([])
    # A light, professional look without enforcing any theme.
    app.setStyle("Fusion")
    win = BgRemoveWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
