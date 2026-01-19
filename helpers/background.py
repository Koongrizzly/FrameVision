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
from PIL import Image, ImageFilter, ImageOps

from PySide6 import QtCore, QtGui, QtWidgets


# -------------------------
# Optional-installs hide state (remove_hide)
# -------------------------

def _remove_hide_state_path() -> str:
    """Match helpers/remove_hide.py storage location (AppDataLocation)."""
    try:
        env_dir = os.environ.get("FRAMEVISION_STATE_DIR", "").strip()
        if env_dir:
            try:
                os.makedirs(env_dir, exist_ok=True)
            except Exception:
                pass
            return str(Path(env_dir) / "remove_hide_state.json")
    except Exception:
        pass

    try:
        base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
        if base:
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                pass
            return str(Path(base) / "remove_hide_state.json")
    except Exception:
        pass

    # Fallback: project root
    try:
        return str(_project_root() / "remove_hide_state.json")
    except Exception:
        return os.path.abspath("remove_hide_state.json")


def _hidden_ids() -> set:
    """Return hidden ids set; best-effort (never raises)."""
    sp = _remove_hide_state_path()
    try:
        if not os.path.exists(sp):
            return set()
        raw = Path(sp).read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
        ids = data.get("hidden_ids", []) if isinstance(data, dict) else []
        if not isinstance(ids, list):
            return set()
        return set(str(x) for x in ids)
    except Exception:
        return set()


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
    """Zoomable / pannable image view.

    Supports two interaction modes:
      - "pan": drag pans (hand tool), wheel zooms
      - "brush": drag paints (left = erase/remove, right = restore), wheel still zooms
    """

    transformed = QtCore.Signal()

    wheel_subject_scale = QtCore.Signal(int)

    subject_drag = QtCore.Signal(QtCore.QPointF)

    # scene_pos, restore(True=bring back), in scene coordinates (pixel-space)
    brush_begin = QtCore.Signal(QtCore.QPointF, bool)
    brush_move = QtCore.Signal(QtCore.QPointF, bool)
    brush_end = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(_checker_brush())

        self._wheel_mode = "view"

        self._subject_dragging = False
        self._subject_last_scene = QtCore.QPointF()

        self._interaction_mode = "pan"
        self.set_interaction_mode("pan")

    def interaction_mode(self) -> str:
        return str(self._interaction_mode)


    def wheel_mode(self) -> str:
        return str(getattr(self, "_wheel_mode", "view"))

    def set_wheel_mode(self, mode: str) -> None:
        mode = (mode or "view").lower().strip()
        if mode in ("subject", "subject_scale", "subjectsize", "subject-size"):
            mode = "subject"
        if mode not in ("view", "subject"):
            mode = "view"
        self._wheel_mode = mode

    def set_interaction_mode(self, mode: str) -> None:
        mode = (mode or "pan").lower().strip()
        if mode not in ("pan", "brush"):
            mode = "pan"
        self._interaction_mode = mode
        if mode == "pan":
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setCursor(QtCore.Qt.OpenHandCursor)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.setCursor(QtCore.Qt.CrossCursor)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Wheel zoom (common editor behavior).
        angle = event.angleDelta().y()
        if angle != 0:
            if getattr(self, "_wheel_mode", "view") == "subject":
                self.wheel_subject_scale.emit(int(angle))
                event.accept()
                return
            factor = 1.0 + (0.15 if angle > 0 else -0.15)
            self.scale(factor, factor)
            self.transformed.emit()
            event.accept()
            return
        super().wheelEvent(event)

    def _event_scene_pos(self, event: QtGui.QMouseEvent) -> QtCore.QPointF:
        try:
            pt = event.position()
            return self.mapToScene(int(pt.x()), int(pt.y()))
        except Exception:
            return self.mapToScene(event.pos())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # Subject move: when wheel mode is "subject", left-drag moves the cutout instead of panning the view.
        if self._interaction_mode == "pan" and getattr(self, "_wheel_mode", "view") == "subject":
            if event.button() == QtCore.Qt.LeftButton:
                self._subject_dragging = True
                self._subject_last_scene = self._event_scene_pos(event)
                event.accept()
                return

        if self._interaction_mode == "brush" and event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            restore = bool(event.button() == QtCore.Qt.RightButton)
            self.brush_begin.emit(self._event_scene_pos(event), restore)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if getattr(self, "_subject_dragging", False):
            if event.buttons() & QtCore.Qt.LeftButton:
                pos = self._event_scene_pos(event)
                delta = pos - self._subject_last_scene
                self._subject_last_scene = pos
                try:
                    self.subject_drag.emit(delta)
                except Exception:
                    pass
                event.accept()
                return
            # Safety: button released without a release event
            self._subject_dragging = False

        if self._interaction_mode == "brush":
            buttons = event.buttons()
            if buttons & (QtCore.Qt.LeftButton | QtCore.Qt.RightButton):
                restore = bool(buttons & QtCore.Qt.RightButton)
                self.brush_move.emit(self._event_scene_pos(event), restore)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if getattr(self, "_subject_dragging", False) and event.button() == QtCore.Qt.LeftButton:
            self._subject_dragging = False
            event.accept()
            return

        if self._interaction_mode == "brush" and event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self.brush_end.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

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
            "last_bg_dir": "",
            "splitter_v_sizes": [],
            "splitter_h_sizes": [],
            "edit_mode": "pan",
            "brush_size": 40,
            "subject_scale_enabled": False,
            "subject_scale": 1.0,
            "subject_offset_x": 0.0,
            "subject_offset_y": 0.0,
            "mask_export_mode": "white_subject",
        }
        self.settings = SettingsStore(self._settings_path, defaults)

        self._threadpool = QtCore.QThreadPool.globalInstance()

        self._pil_original: Optional[Image.Image] = None
        self._pil_result: Optional[Image.Image] = None
        self._pil_matte: Optional[Image.Image] = None

        # Result variants
        self._pil_cutout: Optional[Image.Image] = None  # RGBA cutout (transparent)
        self._pil_bg: Optional[Image.Image] = None      # Background image to composite behind cutout
        self._bg_path: Optional[Path] = None

        self._current_path: Optional[Path] = None

        # Brush-editable matte state (result mask)
        self._matte_arr: Optional[np.ndarray] = None
        self._orig_rgba_arr: Optional[np.ndarray] = None
        self._undo_stack: list[np.ndarray] = []
        self._undo_limit = 10
        self._brush_strength = 0.55
        self._brush_kernel: Optional[np.ndarray] = None
        self._brush_dirty = False
        self._brush_timer = QtCore.QTimer(self)
        self._brush_timer.setSingleShot(True)
        self._brush_timer.setInterval(30)
        self._brush_timer.timeout.connect(self._flush_brush_updates)

        # Subject scaling (for compositing with a new background)
        self._last_subject_scale = float(self.settings.data.get("subject_scale", 1.0))
        self._last_subject_enabled = bool(self.settings.data.get("subject_scale_enabled", False))

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

        # Top bar
        # (Mode + Save actions. The main Open/Run button row is docked at the bottom.)
        top_box = QtWidgets.QWidget()
        top_v = QtWidgets.QVBoxLayout(top_box)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.setSpacing(6)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)

        row_mode = QtWidgets.QHBoxLayout()
        row_mode.setSpacing(8)

        self.btn_open = QtWidgets.QPushButton("Open image")
        self.btn_run = QtWidgets.QPushButton("Remove background")
        self.btn_add_bg = QtWidgets.QPushButton("Add background")
        self.btn_add_bg.setToolTip("Place a new background image behind the cutout result.")
        self.btn_add_bg.setEnabled(False)
        self.btn_save_png = QtWidgets.QPushButton("Save PNG")
        self.btn_save_mask = QtWidgets.QPushButton("Save mask")
        self.cmb_mask_export = QtWidgets.QComboBox()
        self.cmb_mask_export.addItems(["White = subject", "White = background"])
        self.cmb_mask_export.setToolTip(
            "Choose how the saved mask is written.\n"
            "- White = subject: good when you want to keep the subject (common cutout mask).\n"
            "- White = background: good when your inpaint tool expects white = area to change."
        )
        self.cmb_mask_export.setFixedWidth(170)
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setToolTip("Fit both previews to image")
        self.btn_run.setDefault(True)

        # Edit controls (mask brush)
        self.btn_mode_pan = QtWidgets.QPushButton("Pan/Zoom")
        self.btn_mode_pan.setCheckable(True)
        self.btn_mode_pan.setToolTip("Pan with drag, zoom with wheel.")
        self.btn_mode_brush = QtWidgets.QPushButton("Brush")
        self.btn_mode_brush.setCheckable(True)
        self.btn_mode_brush.setToolTip("Paint on the mask: Left = remove, Right = restore.")
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_pan)
        self._mode_group.addButton(self.btn_mode_brush)
        self.btn_mode_pan.setChecked(True)

        # Make selected mode more obvious (custom highlight + badge)
        self.btn_mode_pan.setStyleSheet(
            'QPushButton{padding:6px 12px;} QPushButton:checked{background:#2563eb;color:white;font-weight:600;}'
        )
        self.btn_mode_brush.setStyleSheet(
            'QPushButton{padding:6px 12px;} QPushButton:checked{background:#d97706;color:white;font-weight:600;}'
        )

        self.lbl_mode_badge = QtWidgets.QLabel("PAN / ZOOM")
        self.lbl_mode_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_mode_badge.setMinimumWidth(120)
        self.lbl_mode_badge.setToolTip("Pan/Zoom mode: drag to pan, mouse wheel to zoom.")
        self.lbl_mode_badge.setStyleSheet(
            "QLabel{background:#2563eb;color:white;padding:4px 10px;border-radius:10px;font-weight:600;}"
        )

        self.sp_brush = QtWidgets.QSpinBox()
        self.sp_brush.setRange(3, 300)
        self.sp_brush.setSingleStep(2)
        self.sp_brush.setValue(40)
        self.sp_brush.setFixedWidth(72)
        self.sp_brush.setToolTip("Brush size (px). Left drag removes background, right drag restores it.")
        self.sp_brush.setEnabled(False)

        self.chk_subject_scale = QtWidgets.QCheckBox("Change subject size")
        self.chk_subject_scale.setToolTip(
            "When enabled, mouse wheel changes the cutout subject size relative to the background. "
            "Use the Zoom slider/buttons to zoom the view."
        )
        self.chk_subject_scale.setEnabled(False)

        self.lbl_subject_scale = QtWidgets.QLabel("Subject 100%")
        self.lbl_subject_scale.setMinimumWidth(100)
        self.lbl_subject_scale.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lbl_subject_scale.setStyleSheet("color: #888;")

        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last brush stroke (up to 10).")
        self.btn_undo.setEnabled(False)

        row1.addWidget(self.btn_open)
        row1.addWidget(self.btn_run)
        row1.addWidget(self.btn_add_bg)
        row1.addStretch(1)

        row_mode.addStretch(1)
        row_mode.addWidget(QtWidgets.QLabel("Mode"))
        row_mode.addWidget(self.btn_mode_pan)
        row_mode.addWidget(self.btn_mode_brush)
        row_mode.addWidget(self.lbl_mode_badge)
        row_mode.addWidget(QtWidgets.QLabel("Size"))
        row_mode.addWidget(self.sp_brush)

        row_subject = QtWidgets.QHBoxLayout()
        row_subject.setSpacing(8)
        row_subject.addStretch(1)
        row_subject.addWidget(self.chk_subject_scale)
        row_subject.addWidget(self.lbl_subject_scale)
        row_subject.addWidget(self.btn_fit)

        row2.addStretch(1)
        row2.addWidget(self.btn_undo)
        row2.addWidget(self.btn_save_png)
        row2.addWidget(QtWidgets.QLabel("Mask"))
        row2.addWidget(self.cmb_mask_export)
        row2.addWidget(self.btn_save_mask)

        top_v.addLayout(row1)
        top_v.addLayout(row_mode)
        top_v.addLayout(row_subject)
        top_v.addLayout(row2)
        main.addWidget(top_box)

        # Expose top bar rows for FrameVision integration (injecting extra buttons cleanly).
        self._top_row1 = row1
        self._top_row_mode = row_mode
        self._top_row_subject = row_subject
        self._top_row2 = row2

        # Main content: previews above settings (resizable splitter)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split_v = split

        # Top: previews
        previews = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(previews)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)

 #       titles = QtWidgets.QHBoxLayout()
  #      self.lbl_left = QtWidgets.QLabel("Original")
   #     self.lbl_right = QtWidgets.QLabel("Result")
    #    self.lbl_left.setStyleSheet("font-weight: 600;")
     #   self.lbl_right.setStyleSheet("font-weight: 600;")
 #       titles.addWidget(self.lbl_left)
  #      titles.addStretch(1)
   #     titles.addWidget(self.lbl_right)
    #    pv.addLayout(titles)

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
        self.btn_add_bg.clicked.connect(self.add_background_dialog)
        self.btn_save_png.clicked.connect(self.save_png_dialog)
        self.btn_save_mask.clicked.connect(self.save_mask_dialog)
        self.cmb_mask_export.currentIndexChanged.connect(self._on_mask_export_changed)
        self.btn_fit.clicked.connect(self._fit_views)

        # Edit mode + undo
        self.btn_mode_pan.clicked.connect(lambda: self.settings.set("edit_mode", "pan"))
        self.btn_mode_brush.clicked.connect(lambda: self.settings.set("edit_mode", "brush"))
        self.sp_brush.valueChanged.connect(lambda v: self.settings.set("brush_size", int(v)))
        self.chk_subject_scale.toggled.connect(lambda v: self.settings.set("subject_scale_enabled", bool(v)))
        self.btn_undo.clicked.connect(self.undo_brush)

        # Brush input (allow painting on either preview; result mask is shared)
        for _v in (self.view_a, self.view_b):
            _v.brush_begin.connect(self._on_brush_begin)
            _v.brush_move.connect(self._on_brush_move)
            _v.brush_end.connect(self._on_brush_end)
            _v.wheel_subject_scale.connect(self._on_subject_wheel)
            try:
                _v.subject_drag.connect(self._on_subject_drag)
            except Exception:
                pass

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
    # ---- Mask editing (brush)

    def _set_edit_mode(self, mode: str, *, update_settings: bool = True) -> None:
        mode = (mode or "pan").lower().strip()
        if mode not in ("pan", "brush"):
            mode = "pan"
        if update_settings:
            try:
                self.settings.set("edit_mode", mode)
            except Exception:
                pass

        # Subject scaling uses the wheel; disable it while brushing to avoid coordinate mismatch.
        if mode == "brush":
            try:
                if bool(self.settings.data.get("subject_scale_enabled", False)):
                    self.settings.set("subject_scale_enabled", False)
            except Exception:
                pass

        # Update UI controls (best-effort)
        try:
            self.btn_mode_pan.setChecked(mode == "pan")
            self.btn_mode_brush.setChecked(mode == "brush")
        except Exception:
            pass

        try:
            self.sp_brush.setEnabled(mode == "brush")
        except Exception:
            pass

        try:
            self._update_mode_badge(mode)
        except Exception:
            pass

        # Apply interaction mode to both views (keeps left/right synced)
        try:
            self.view_a.set_interaction_mode("pan" if mode == "pan" else "brush")
            self.view_b.set_interaction_mode("pan" if mode == "pan" else "brush")
        except Exception:
            pass


        try:
            self._update_subject_scale_controls()
        except Exception:
            pass

    def _update_mode_badge(self, mode: str) -> None:
        """Update the mode badge label (makes Pan/Zoom vs Brush much clearer)."""
        lab = getattr(self, "lbl_mode_badge", None)
        if lab is None:
            return

        mode = (mode or "pan").lower().strip()
        if mode == "brush":
            lab.setText("BRUSH MODE")
            lab.setToolTip("Brush mode: Left drag removes background, Right drag restores it.")
            lab.setStyleSheet(
                "QLabel{background:#d97706;color:white;padding:4px 10px;border-radius:10px;font-weight:600;}"
            )
        else:
            lab.setText("PAN / ZOOM")
            lab.setToolTip("Pan/Zoom mode: drag to pan, mouse wheel to zoom.")
            lab.setStyleSheet(
                "QLabel{background:#2563eb;color:white;padding:4px 10px;border-radius:10px;font-weight:600;}"
            )

    def _make_brush_kernel(self, size_px: int) -> np.ndarray:
        # size_px is a "diameter-like" value (UI friendly).
        size_px = int(max(3, min(300, size_px)))
        r = max(1, int(round(size_px / 2.0)))
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        dist = np.sqrt(x * x + y * y)
        k = np.clip(1.0 - (dist / float(r + 1e-6)), 0.0, 1.0)
        # Slight softness (feathered falloff)
        k = k * k
        return k.astype(np.float32)

    def _set_brush_size(self, size_px: int, *, update_settings: bool = True) -> None:
        size_px = int(max(3, min(300, int(size_px))))
        if update_settings:
            try:
                self.settings.set("brush_size", size_px)
            except Exception:
                pass
        try:
            self.sp_brush.setValue(size_px)
        except Exception:
            pass
        try:
            self._brush_kernel = self._make_brush_kernel(size_px)
        except Exception:
            self._brush_kernel = None

    def _update_undo_button(self) -> None:
        try:
            self.btn_undo.setEnabled(bool(self._undo_stack))
        except Exception:
            pass

    def undo_brush(self) -> None:
        """Undo last brush stroke (up to 10 snapshots)."""
        if not self._undo_stack:
            return
        try:
            prev = self._undo_stack.pop()
            self._matte_arr = np.asarray(prev, dtype=np.uint8).copy()
            self._update_undo_button()
            self._brush_dirty = True
            self._flush_brush_updates()
        except Exception:
            pass

    @QtCore.Slot(QtCore.QPointF, bool)
    def _on_brush_begin(self, scene_pos: QtCore.QPointF, restore: bool) -> None:
        if self._matte_arr is None or self._pil_original is None:
            return
        # Snapshot for undo (one per stroke)
        try:
            self._undo_stack.append(self._matte_arr.copy())
            if len(self._undo_stack) > int(self._undo_limit):
                self._undo_stack.pop(0)
        except Exception:
            self._undo_stack = self._undo_stack[-int(self._undo_limit) :]
        self._update_undo_button()

        self._apply_brush_at(scene_pos, restore)
        self._schedule_brush_updates()

    @QtCore.Slot(QtCore.QPointF, bool)
    def _on_brush_move(self, scene_pos: QtCore.QPointF, restore: bool) -> None:
        if self._matte_arr is None:
            return
        self._apply_brush_at(scene_pos, restore)
        self._schedule_brush_updates()

    @QtCore.Slot()
    def _on_brush_end(self) -> None:
        if self._matte_arr is None:
            return
        self._schedule_brush_updates()

    def _apply_brush_at(self, scene_pos: QtCore.QPointF, restore: bool) -> None:
        if self._matte_arr is None:
            return

        # Ensure kernel exists
        if self._brush_kernel is None:
            try:
                bs = int(self.settings.data.get("brush_size", 40))
            except Exception:
                bs = 40
            self._brush_kernel = self._make_brush_kernel(bs)

        k = self._brush_kernel
        if k is None:
            return

        r = (k.shape[0] - 1) // 2
        x = int(round(float(scene_pos.x())))
        y = int(round(float(scene_pos.y())))

        h, w = int(self._matte_arr.shape[0]), int(self._matte_arr.shape[1])
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        kx0 = x0 - (x - r)
        ky0 = y0 - (y - r)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        ks = k[ky0:ky1, kx0:kx1]
        if ks.size == 0:
            return

        a = self._matte_arr[y0:y1, x0:x1].astype(np.float32) / 255.0
        s = float(max(0.05, min(1.0, self._brush_strength)))

        if restore:
            a = a + (1.0 - a) * (ks * s)
        else:
            a = a * (1.0 - (ks * s))

        self._matte_arr[y0:y1, x0:x1] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
        self._brush_dirty = True

    def _schedule_brush_updates(self) -> None:
        if not self._brush_timer.isActive():
            self._brush_timer.start()

    # ---- Subject scaling (size vs background)

    def _subject_scale_allowed(self) -> bool:
        try:
            mode = str(self.settings.data.get("edit_mode", "pan")).lower().strip()
        except Exception:
            mode = "pan"
        if mode == "brush":
            return False
        return (self._pil_original is not None) and (self._matte_arr is not None)

    def _update_subject_scale_controls(self) -> None:
        """Enable/disable the subject scaling toggle based on current state."""
        allowed = self._subject_scale_allowed()
        try:
            self.chk_subject_scale.setEnabled(bool(allowed))
            self.lbl_subject_scale.setEnabled(bool(allowed))
        except Exception:
            pass
        self._apply_subject_wheel_mode()

    def _apply_subject_wheel_mode(self) -> None:
        enabled = False
        try:
            enabled = bool(self.settings.data.get("subject_scale_enabled", False))
        except Exception:
            enabled = False
        enabled = bool(enabled and self._subject_scale_allowed())
        mode = "subject" if enabled else "view"
        try:
            self.view_a.set_wheel_mode(mode)
        except Exception:
            pass
        try:
            self.view_b.set_wheel_mode(mode)
        except Exception:
            pass

    def _apply_subject_scale(self, cutout: Image.Image) -> Image.Image:
        """Scale the subject (RGBA cutout) onto a same-size transparent canvas."""
        try:
            enabled = bool(self.settings.data.get("subject_scale_enabled", False))
            scale = float(self.settings.data.get("subject_scale", 1.0))
        except Exception:
            enabled = False
            scale = 1.0

        try:
            ox = float(self.settings.data.get("subject_offset_x", 0.0))
            oy = float(self.settings.data.get("subject_offset_y", 0.0))
        except Exception:
            ox, oy = 0.0, 0.0

        if not enabled:
            return cutout

        if abs(scale - 1.0) < 1e-4 and abs(ox) < 1e-4 and abs(oy) < 1e-4:
            return cutout

        scale = float(max(0.10, min(4.00, scale)))

        try:
            resample = Image.Resampling.LANCZOS  # Pillow >= 9
        except Exception:
            resample = getattr(Image, "LANCZOS", Image.BICUBIC)

        cw, ch = int(cutout.width), int(cutout.height)
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))

        try:
            if abs(scale - 1.0) < 1e-4:
                scaled = cutout.convert("RGBA")
            else:
                scaled = cutout.convert("RGBA").resize((nw, nh), resample)
        except Exception:
            return cutout

        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        left = (cw - nw) // 2 + int(round(ox))
        top = (ch - nh) // 2 + int(round(oy))
        try:
            canvas.paste(scaled, (left, top), scaled)
        except Exception:
            try:
                return Image.alpha_composite(canvas, scaled)
            except Exception:
                return cutout
        return canvas
    @QtCore.Slot(int)
    def _on_subject_wheel(self, angle: int) -> None:
        """Mouse wheel adjusts subject scale when the toggle is enabled."""
        if not self._subject_scale_allowed():
            return
        try:
            if not bool(self.settings.data.get("subject_scale_enabled", False)):
                return
        except Exception:
            return

        try:
            steps = int(round(int(angle) / 120))
        except Exception:
            steps = 1 if angle > 0 else -1

        if steps == 0:
            return

        base = 1.05  # 5% per wheel step
        try:
            cur = float(self.settings.data.get("subject_scale", 1.0))
        except Exception:
            cur = 1.0

        if steps > 0:
            cur *= (base ** steps)
        else:
            cur /= (base ** (-steps))

        cur = float(max(0.10, min(4.00, cur)))
        self.settings.set("subject_scale", cur)

        # Force a recomposite for preview + save
        self._brush_dirty = True
        self._refresh_result_from_matte(force=True)


    @QtCore.Slot(QtCore.QPointF)
    def _on_subject_drag(self, delta: QtCore.QPointF) -> None:
        """Left-drag moves the subject (cutout) relative to the background."""
        if not self._subject_scale_allowed():
            return
        try:
            if not bool(self.settings.data.get("subject_scale_enabled", False)):
                return
        except Exception:
            return
    
        try:
            dx = float(delta.x())
            dy = float(delta.y())
        except Exception:
            return
    
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
    
        try:
            ox = float(self.settings.data.get("subject_offset_x", 0.0))
            oy = float(self.settings.data.get("subject_offset_y", 0.0))
        except Exception:
            ox, oy = 0.0, 0.0
    
        ox += dx
        oy += dy
    
        lim = 100000.0
        ox = float(max(-lim, min(lim, ox)))
        oy = float(max(-lim, min(lim, oy)))
    
        try:
            self.settings.update(subject_offset_x=ox, subject_offset_y=oy)
        except Exception:
            try:
                self.settings.set("subject_offset_x", ox)
                self.settings.set("subject_offset_y", oy)
            except Exception:
                return
    
        self._brush_dirty = True
        self._refresh_result_from_matte(force=True)
    
    def _resize_cover_rgb(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize an image to cover the target size (center-crop). Returns RGB."""
        img = _pil_to_rgb(img)
        w, h = img.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (target_w, target_h), (0, 0, 0))
        if w == target_w and h == target_h:
            return img

        scale = max(float(target_w) / float(w), float(target_h) / float(h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        try:
            resample = Image.Resampling.LANCZOS  # Pillow >= 9
        except Exception:
            resample = getattr(Image, "LANCZOS", Image.BICUBIC)

        img_r = img.resize((nw, nh), resample)
        left = max(0, (nw - target_w) // 2)
        top = max(0, (nh - target_h) // 2)
        return img_r.crop((left, top, left + target_w, top + target_h))

    def _composite_cutout_with_background(self, cutout: Image.Image) -> Image.Image:
        """Return a composited RGBA image (background + cutout)."""
        cutout2 = self._apply_subject_scale(cutout)
        if self._pil_bg is None:
            return cutout2
        bg_rgb = self._resize_cover_rgb(self._pil_bg, cutout2.width, cutout2.height)
        bg_rgba = bg_rgb.convert("RGBA")
        return Image.alpha_composite(bg_rgba, cutout2.convert("RGBA"))

    def _refresh_result_from_matte(self, force: bool = False) -> None:
        """Rebuild current cutout/result from the editable matte and optional background."""
        if (not force) and (not self._brush_dirty):
            return
        self._brush_dirty = False

        if self._pil_original is None or self._matte_arr is None:
            return

        # Cache original RGBA for fast reapply
        if self._orig_rgba_arr is None:
            try:
                self._orig_rgba_arr = np.asarray(_pil_to_rgb(self._pil_original).convert("RGBA"), dtype=np.uint8)
            except Exception:
                return

        rgba = self._orig_rgba_arr.copy()
        try:
            rgba[..., 3] = self._matte_arr
        except Exception:
            return

        try:
            cutout = Image.fromarray(rgba, mode="RGBA")
            self._pil_cutout = cutout
            self._pil_matte = Image.fromarray(self._matte_arr.copy(), mode="L")
        except Exception:
            return

        # Composite with background and optional subject scaling
        try:
            self._pil_result = self._composite_cutout_with_background(cutout)
        except Exception:
            self._pil_result = cutout

        self._update_result_pixmap(self._pil_result)

    def _flush_brush_updates(self) -> None:
        self._refresh_result_from_matte(force=False)



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
        b9 = QtCore.QSignalBlocker(self.btn_mode_pan)
        b10 = QtCore.QSignalBlocker(self.btn_mode_brush)
        b11 = QtCore.QSignalBlocker(self.sp_brush)
        b12 = QtCore.QSignalBlocker(self.chk_subject_scale)
        _ = (b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12)

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

        # Mask export mode (saved mask color convention)
        try:
            mexp = str(s.get("mask_export_mode", "white_subject"))
            self.cmb_mask_export.setCurrentIndex(1 if mexp == "white_background" else 0)
        except Exception:
            pass

        # Mask edit mode
        mode = str(s.get("edit_mode", "pan"))
        self._set_edit_mode(mode, update_settings=False)

        bs = int(s.get("brush_size", 40))
        self._set_brush_size(bs, update_settings=False)

        # Subject scaling UI
        enabled = bool(s.get("subject_scale_enabled", False))
        try:
            scale = float(s.get("subject_scale", 1.0))
        except Exception:
            scale = 1.0
        try:
            self.chk_subject_scale.setChecked(bool(enabled))
        except Exception:
            pass
        try:
            self.lbl_subject_scale.setText(f"Subject {int(round(scale * 100.0))}%")
        except Exception:
            pass

        # Enable/disable toggle based on current mode + whether we have a cutout.
        self._update_subject_scale_controls()

        # If subject scaling changed, re-composite result for preview/save.
        try:
            if (bool(enabled) != bool(getattr(self, "_last_subject_enabled", False))) or (abs(float(scale) - float(getattr(self, "_last_subject_scale", 1.0))) > 1e-4):
                self._last_subject_enabled = bool(enabled)
                self._last_subject_scale = float(scale)
                self._brush_dirty = True
                self._refresh_result_from_matte(force=True)
        except Exception:
            pass

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

    def _on_mask_export_changed(self, idx: int) -> None:
        """0 = white subject, 1 = white background (invert)."""
        try:
            i = int(idx)
        except Exception:
            i = 0
        mode = "white_subject" if i == 0 else "white_background"
        try:
            self.settings.set("mask_export_mode", mode)
        except Exception:
            pass

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

    
    def add_background_dialog(self) -> None:
        """Choose an image and place it behind the current cutout result."""
        # Make sure latest brush edits are applied first.
        try:
            self._flush_brush_updates()
        except Exception:
            pass

        if self._pil_original is None or self._matte_arr is None:
            self._set_status("Run background removal first.")
            return

        start_dir = self.settings.data.get("last_bg_dir") or self.settings.data.get("last_open_dir") or str(_project_root())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose background image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        p = Path(path)
        self.settings.set("last_bg_dir", str(p.parent))

        try:
            bg = Image.open(p)
            bg.load()
        except Exception as e:
            self._set_status(f"Failed to open background image: {e}")
            return

        self._pil_bg = bg
        self._bg_path = p

        # Rebuild composite.
        self._brush_dirty = True
        self._refresh_result_from_matte(force=True)
        self._set_status(f"Background set: {p.name}")

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
        self._pil_cutout = None
        self._pil_bg = None
        self._bg_path = None
        # Reset subject scaling for a new image (prevents confusing carry-over)
        try:
            self.settings.update(subject_scale_enabled=False, subject_scale=1.0)
        except Exception:
            pass
        try:
            self.btn_add_bg.setEnabled(False)
        except Exception:
            pass
        self._matte_arr = None
        self._orig_rgba_arr = None
        try:
            self._undo_stack.clear()
        except Exception:
            self._undo_stack = []
        self._update_undo_button()
        try:
            if self._brush_timer.isActive():
                self._brush_timer.stop()
        except Exception:
            pass
        self._show_original(img)
        self._show_result(None)
        self._set_status(f"Loaded: {path.name}  ({img.width}x{img.height})")
        self._fit_views()
    def save_png_dialog(self) -> None:
        """Save the current result immediately (no file dialog).

        Files are saved to last_save_dir (if set) or the default results folder.
        Filenames are unique and end with a date/time stamp.
        """
        self._flush_brush_updates()
        if self._pil_result is None:
            self._set_status("Nothing to save. Run background removal first.")
            return
        p = self._auto_save_image(self._pil_result, kind="cutout")
        if p is not None:
            self._set_status(f"Saved: {p.name}")
            self._toast(f"Saved to: {p}")
    def save_mask_dialog(self) -> None:
        """Save the current matte/mask immediately (no file dialog).

        Files are saved to last_save_dir (if set) or the default results folder.
        Filenames are unique and end with a date/time stamp.
        """
        self._flush_brush_updates()
        if self._pil_matte is None:
            self._set_status("No mask available. Run background removal first.")
            return
        # Optionally invert the saved mask so you can choose whether
        # white represents the subject or the background.
        mode = str(self.settings.data.get("mask_export_mode", "white_subject"))
        to_save = self._pil_matte
        try:
            to_save = to_save.convert("L")
        except Exception:
            pass
        if mode == "white_background":
            try:
                to_save = ImageOps.invert(to_save)
            except Exception:
                try:
                    a = np.asarray(to_save, dtype=np.uint8)
                    to_save = Image.fromarray((255 - a).astype(np.uint8), mode="L")
                except Exception:
                    pass
        p = self._auto_save_image(to_save, kind="mask")
        if p is not None:
            self._set_status(f"Saved: {p.name}")
            self._toast(f"Saved to: {p}")
    def _toast(self, text: str, msec: int = 2200) -> None:
        """Small 'toast' style popup using QToolTip (simple + dependency-free)."""
        try:
            pos = self.mapToGlobal(self.rect().center())
        except Exception:
            try:
                pos = QtGui.QCursor.pos()
            except Exception:
                pos = QtCore.QPoint(0, 0)
        try:
            QtWidgets.QToolTip.showText(pos, text, self, self.rect(), int(msec))
        except Exception:
            pass

    def _resolve_save_dir(self) -> Path:
        """Return a usable save directory (and create it if needed)."""
        try:
            d = str(self.settings.data.get("last_save_dir") or "").strip()
        except Exception:
            d = ""
        p = Path(d) if d else Path(getattr(self, "_default_save_dir", _project_root() / "output" / "images" / "rm_backgrounds"))
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # last resort
            p = _project_root() / "output"
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return p

    def _timestamp_suffix(self) -> str:
        """Return a date/time suffix (includes milliseconds) for unique filenames."""
        try:
            return QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        except Exception:
            import time as _time
            base = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
            ms = int((_time.time() % 1.0) * 1000.0)
            return f"{base}_{ms:03d}"

    def _unique_path(self, folder: Path, stem: str, kind: str, ext: str = "png") -> Path:
        ts = self._timestamp_suffix()
        safe_stem = (stem or "result").strip().replace(" ", "_")
        safe_kind = (kind or "out").strip().replace(" ", "_")
        # Keep the timestamp at the very end (as requested). If a collision happens, insert a counter BEFORE the timestamp.
        p = folder / f"{safe_stem}_{safe_kind}_{ts}.{ext}"
        if not p.exists():
            return p
        for i in range(1, 1000):
            cand = folder / f"{safe_stem}_{safe_kind}_{i:02d}_{ts}.{ext}"
            if not cand.exists():
                return cand
        # Extremely unlikely fallback
        try:
            extra = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        except Exception:
            extra = 0
        return folder / f"{safe_stem}_{safe_kind}_{ts}_{extra}.{ext}"

    def _auto_save_image(self, pil_img: Image.Image, *, kind: str) -> Optional[Path]:
        folder = self._resolve_save_dir()
        stem = "result"
        try:
            if self._current_path is not None:
                stem = self._current_path.stem
        except Exception:
            pass

        out_path = self._unique_path(folder, stem=stem, kind=kind, ext="png")
        try:
            pil_img.save(out_path, format="PNG")
            try:
                self.settings.set("last_save_dir", str(out_path.parent))
            except Exception:
                pass
            return out_path
        except Exception as e:
            self._set_status(f"Save failed: {e}")
            try:
                self._toast(f"Save failed: {e}")
            except Exception:
                pass
            return None

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
        # New base result: reset any previously chosen background.
        self._pil_cutout = rgba
        self._pil_bg = None
        self._bg_path = None
        # Reset subject scaling for a fresh cutout (user can enable again if needed)
        try:
            self.settings.update(subject_scale_enabled=False, subject_scale=1.0)
        except Exception:
            pass

        self._pil_result = rgba
        self._pil_matte = matte
        try:
            self.btn_add_bg.setEnabled(True)
        except Exception:
            pass

        # Cache editable matte for brush edits
        try:
            self._matte_arr = np.asarray(matte.convert("L"), dtype=np.uint8).copy()
        except Exception:
            self._matte_arr = None

        # Cache original RGBA for fast re-apply of edited alpha
        try:
            if self._pil_original is not None:
                self._orig_rgba_arr = np.asarray(_pil_to_rgb(self._pil_original).convert("RGBA"), dtype=np.uint8)
            else:
                self._orig_rgba_arr = None
        except Exception:
            self._orig_rgba_arr = None

        # Reset undo history (new base result)
        try:
            self._undo_stack.clear()
        except Exception:
            self._undo_stack = []
        self._update_undo_button()

        try:
            self._update_subject_scale_controls()
        except Exception:
            pass

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


    def _update_result_pixmap(self, img: Optional[Image.Image]) -> None:
        """Fast update for the Result preview.

        Brush edits can update very frequently; we avoid clearing/rebuilding the whole scene
        if a pixmap item already exists.
        """
        if img is None:
            self._show_result(None)
            return
        try:
            qimg = _qimage_from_pil(img)
            pix = QtGui.QPixmap.fromImage(qimg)
            if getattr(self, "_pix_b", None) is not None:
                try:
                    self._pix_b.setPixmap(pix)  # type: ignore[union-attr]
                    self.scene_b.setSceneRect(QtCore.QRectF(pix.rect()))
                    return
                except Exception:
                    pass
            # Fallback: rebuild result scene
            self._show_result(img)
        except Exception:
            self._show_result(img)


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
    """Best-effort: find current media path and current time (seconds) from the host app.

    Returns: (path, time_s)
      - time_s >= 0 : a best-effort position in seconds
      - time_s < 0  : unknown (host did not expose a usable position)
    """
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

    # Time is optional; default to UNKNOWN.
    t = -1.0

    def _as_float(v) -> Optional[float]:
        try:
            if callable(v):
                v = v()
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _try_seconds(obj) -> Optional[float]:
        if obj is None:
            return None
        # Direct seconds attributes used in some builds
        for attr in ("current_time_s", "time_s", "pos_s", "position_s", "playhead_s", "cursor_s"):
            if hasattr(obj, attr):
                vv = _as_float(getattr(obj, attr))
                if vv is not None:
                    return max(0.0, vv)
        # Some widgets expose a callable current_time_s()
        for meth in ("current_time_s", "time_s", "pos_s", "position_s"):
            if hasattr(obj, meth) and callable(getattr(obj, meth)):
                vv = _as_float(getattr(obj, meth))
                if vv is not None:
                    return max(0.0, vv)
        return None

    def _try_millis(obj) -> Optional[float]:
        if obj is None:
            return None
        # Common millisecond patterns
        for attr in ("position_ms", "pos_ms", "current_time_ms", "time_ms"):
            if hasattr(obj, attr):
                vv = _as_float(getattr(obj, attr))
                if vv is not None:
                    return max(0.0, vv) / 1000.0
        # QtMultimedia-style position() -> ms
        for meth in ("position",):
            if hasattr(obj, meth) and callable(getattr(obj, meth)):
                vv = _as_float(getattr(obj, meth))
                if vv is not None:
                    return max(0.0, vv) / 1000.0
        return None

    try:
        main = getattr(pane, "main", None)
        vid = getattr(main, "video", None) if main is not None else None

        # 1) Try seconds directly from the video widget
        vv = _try_seconds(vid)
        if vv is not None:
            t = vv
        else:
            # 2) Try ms from a player object attached to the widget / main
            player_candidates = [
                vid,
                getattr(vid, "player", None),
                getattr(vid, "_player", None),
                getattr(vid, "mediaPlayer", None),
                getattr(vid, "mediaplayer", None),
                getattr(main, "player", None) if main is not None else None,
                getattr(main, "_player", None) if main is not None else None,
                getattr(main, "mediaPlayer", None) if main is not None else None,
                getattr(main, "mediaplayer", None) if main is not None else None,
            ]
            for pc in player_candidates:
                vv = _try_millis(pc)
                if vv is not None:
                    t = vv
                    break
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

    Goal: when the user clicks **Use current**, grab the frame that is *currently shown/playing*.

    Strategy:
      - Images: load directly from disk (full resolution).
      - Video: prefer the *currently shown* QImage/frame from the player (correct moment),
               and only fall back to ffmpeg extraction when we have a reliable timestamp.
    """
    # 1) Try to resolve the currently-open media path + playhead time from the host app.
    media_path, time_s = _current_media_from_app(pane)

    # Quick helpers
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    vid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}

    # 1a) If it's an image file, load directly from disk (full resolution).
    if media_path is not None:
        try:
            if media_path.suffix.lower() in img_exts:
                im = Image.open(media_path)
                im.load()
                return im, media_path.name
        except Exception:
            pass

        # 1b) If it's a video, prefer the currently displayed frame (most accurate).
        if media_path.suffix.lower() in vid_exts:
            try:
                qimg = _try_grab_qimage_from_host(pane)
                if qimg is not None:
                    import tempfile as _tf
                    from PySide6.QtGui import QPixmap as _QPixmap
                    tmp = Path(_tf.gettempdir()) / f"fv_bgtool_qframe_{_os.getpid()}.png"
                    _QPixmap.fromImage(qimg).save(str(tmp), "PNG")
                    im = Image.open(tmp)
                    im.load()
                    return im, "current frame"
            except Exception:
                pass

            # 1c) If we couldn't grab a frame image, fall back to ffmpeg only if time is known.
            try:
                if float(time_s) >= 0.0:
                    im = _extract_video_frame(media_path, time_s=float(time_s))
                    if im is not None:
                        return im, f"{media_path.name} @ {float(time_s):.2f}s"
            except Exception:
                pass

            # Last fallback for video: first frame (still better than returning nothing).
            try:
                im0 = _extract_video_frame(media_path, time_s=0.0)
                if im0 is not None:
                    return im0, f"{media_path.name} @ 0.00s"
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
    self._pil_cutout = None
    self._pil_bg = None
    self._bg_path = None
    try:
        self.btn_add_bg.setEnabled(False)
    except Exception:
        pass
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


def _create_sdxl_inpaint_pane(parent=None):
    """Best-effort: create the SDXLInpaintPane widget.

    We try a few import paths so this works both when running inside FrameVision
    (helpers.* package) and when running this file standalone.
    """
    try:
        from PySide6 import QtWidgets as _QtW  # type: ignore
    except Exception:
        return None

    import importlib

    candidates = (
        "helpers.sdxl_inpaint",      # expected (this patch)
        "helpers.sdxl_inpaint_ui",   # older name referenced in some builds
        "sdxl_inpaint",
        "sdxl_inpaint_ui",
    )

    last_err = None
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            Pane = getattr(mod, "SDXLInpaintPane", None)
            if Pane is None:
                continue
            return Pane(parent)
        except Exception as e:
            last_err = e
            continue

    # Placeholder pane if file isn't available in this build.
    w = _QtW.QWidget(parent)
    lay = _QtW.QVBoxLayout(w)
    lay.setContentsMargins(12, 12, 12, 12)
    msg = "SDXL Inpaint tab is not available in this build.\n\n" \
          "Make sure helpers/sdxl_inpaint.py exists and dependencies (torch/diffusers) are installed."
    if last_err is not None:
        msg += f"\n\nImport error: {last_err}"
    lab = _QtW.QLabel(msg)
    lab.setWordWrap(True)
    lay.addWidget(lab)
    lay.addStretch(1)
    return w


def install_background_tool(pane, section_widget) -> None:
    """Tools-tab entry point expected by helpers.tools_tab.

    Adds a tab UI so you can quickly switch between:
      - Removal (ONNX background remover)
      - Inpaint (SDXL inpaint pane)
    """
    try:
        from PySide6 import QtWidgets as _QtW
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

    tabs = _QtW.QTabWidget(container)
    tabs.setObjectName("BackgroundToolTabs")

    bg_pane = BgRemovePane(tabs)
    _hide_inpaint = ("sdxl_inpaint" in _hidden_ids())
    inpaint_pane = None if _hide_inpaint else _create_sdxl_inpaint_pane(tabs)

    # Tools-tab "Remember settings" uses a generic snapshot/restore that can accidentally
    # shuffle values across unrelated widgets when layouts change. These panes have their
    # own persistence, so we opt them out to avoid settings corruption.
    try:
        for _w in (inpaint_pane,):
            if _w is not None:
                _w.setProperty("_fv_skip_restore", True)
                _w.setProperty("_fv_skip_snapshot", True)
    except Exception:
        pass


    tabs.addTab(bg_pane, "Background Removal")
    if not _hide_inpaint:
        if inpaint_pane is not None:
            tabs.addTab(inpaint_pane, "SDXL (Low Vram) Inpainter")
        else:
            # Should not happen, but keep tool usable.
            tabs.addTab(_QtW.QLabel("Inpaint tab unavailable (PySide6 not loaded)."), "Inpaint")


    # Inject "Use current" + "View results" buttons into the removal pane's top bar.
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
            try:
                tabs.setCurrentIndex(0)  # jump to Removal tab since it now has the loaded image
            except Exception:
                pass
        except Exception:
            # last-resort: try saving to a temp file then load by path
            try:
                import tempfile as _tf
                tmp = Path(_tf.gettempdir()) / f"fv_bgtool_current_{_os.getpid()}.png"
                im.convert("RGBA").save(tmp)
                bg_pane.load_image(tmp)
                try:
                    tabs.setCurrentIndex(0)
                except Exception:
                    pass
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
        tabs.setSizePolicy(_QtW.QSizePolicy.Expanding, _QtW.QSizePolicy.Expanding)
    except Exception:
        pass

    outer.addWidget(tabs, stretch=1)

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
        section_widget._bg_tool_tabs = tabs
        section_widget._bg_tool_pane = bg_pane
        section_widget._bg_tool_inpaint = inpaint_pane
    except Exception:
        pass


# -------------------------
# Standalone test window
# -------------------------

class BgRemoveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background: Removal + Inpaint")
        self.resize(1200, 780)

        self.tabs = QtWidgets.QTabWidget(self)
        self.pane = BgRemovePane(self.tabs)  # keep .pane for existing menu actions
        self._hide_inpaint = ("sdxl_inpaint" in _hidden_ids())
        self.inpaint = None if self._hide_inpaint else _create_sdxl_inpaint_pane(self.tabs)

        self.tabs.addTab(self.pane, "Removal")
        if not self._hide_inpaint:
            if self.inpaint is not None:
                self.tabs.addTab(self.inpaint, "Inpaint")
            else:
                self.tabs.addTab(QtWidgets.QLabel("Inpaint tab unavailable."), "Inpaint")

        self.setCentralWidget(self.tabs)

        # Basic menu (targets the Removal tab)
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
