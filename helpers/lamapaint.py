# -*- coding: utf-8 -*-
"""
LamaPaint (LaMa-ONNX) — lightweight inpainting pane for PySide6 apps.

Folder layout (expected):
  (root)
    /helpers/lamapaint.py
    /presets/setsave/lamapaint.json        (saved settings)
    /models/bg/                            (downloads/uses model here)
    /presets/bin/                          (optional exes, e.g., ffmpeg)

Notes:
- Uses Carve/LaMa-ONNX "lama_fp32.onnx" (512x512 fixed input).
- Designed to be VRAM-friendly by default via ROI/crop inpainting.
"""

from __future__ import annotations

import os
import json
import time
import traceback
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QMessageBox, QGroupBox, QSizePolicy
)

# Optional dependency. We import lazily in runtime to allow UI to open even if not installed.
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


# -----------------------------
# Paths / Settings
# -----------------------------

def _project_root_from_helpers() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _settings_path() -> str:
    return os.path.join(_project_root_from_helpers(), "presets", "setsave", "lamapaint.json")


def _models_dir() -> str:
    return os.path.join(_project_root_from_helpers(), "models", "bg")


def _model_path() -> str:
    return os.path.join(_models_dir(), "lama_fp32.onnx")


DEFAULT_SETTINGS = {
    "use_gpu_if_available": True,
    "crop_padding_percent": 0.18,     # extra context around mask bbox (0..1)
    "feather_px": 6,                  # feather mask edges for nicer blends
    "jpeg_quality": 95,
    "last_image_path": "",
    "last_mask_path": "",
    "last_out_dir": "",
}


def load_settings() -> dict:
    path = _settings_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_SETTINGS)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(data: dict) -> None:
    path = _settings_path()
    _ensure_dir(os.path.dirname(path))
    merged = dict(DEFAULT_SETTINGS)
    merged.update(data or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


# -----------------------------
# Download worker
# -----------------------------

@dataclass
class DownloadSpec:
    url: str
    dst_path: str
    tmp_path: str


class ModelDownloadThread(QThread):
    progress = Signal(int)          # 0..100
    status = Signal(str)            # text
    finished_ok = Signal(str)       # dst_path
    failed = Signal(str)            # error string

    def __init__(self, spec: DownloadSpec):
        super().__init__()
        self.spec = spec

    def run(self) -> None:  # pragma: no cover
        try:
            _ensure_dir(os.path.dirname(self.spec.dst_path))

            self.status.emit("Connecting…")

            req = urllib.request.Request(
                self.spec.url,
                headers={
                    "User-Agent": "LamaPaint/1.0 (PySide6)"
                }
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = resp.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else None

                self.status.emit("Downloading model…")
                downloaded = 0
                chunk = 1024 * 1024  # 1 MiB

                with open(self.spec.tmp_path, "wb") as out:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        out.write(buf)
                        downloaded += len(buf)
                        if total_bytes:
                            pct = int((downloaded / total_bytes) * 100)
                            self.progress.emit(max(0, min(100, pct)))
                        else:
                            # indeterminate-ish: pulse between 0-99
                            self.progress.emit(min(99, (downloaded // (5 * 1024 * 1024)) % 100))

            # Atomic-ish replace
            self.status.emit("Finalizing…")
            if os.path.exists(self.spec.dst_path):
                try:
                    os.remove(self.spec.dst_path)
                except Exception:
                    pass
            os.replace(self.spec.tmp_path, self.spec.dst_path)

            self.progress.emit(100)
            self.status.emit("Installed.")
            self.finished_ok.emit(self.spec.dst_path)
        except Exception as e:
            try:
                if os.path.exists(self.spec.tmp_path):
                    os.remove(self.spec.tmp_path)
            except Exception:
                pass
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# -----------------------------
# Inpaint worker
# -----------------------------

class InpaintThread(QThread):
    status = Signal(str)
    finished_ok = Signal(str)   # out_path
    failed = Signal(str)

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        out_path: str,
        use_gpu_if_available: bool,
        crop_padding_percent: float,
        feather_px: int,
        jpeg_quality: int,
        model_path: str,
    ):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.out_path = out_path
        self.use_gpu_if_available = use_gpu_if_available
        self.crop_padding_percent = float(crop_padding_percent)
        self.feather_px = int(feather_px)
        self.jpeg_quality = int(jpeg_quality)
        self.model_path = model_path

    def run(self) -> None:  # pragma: no cover
        try:
            if ort is None:
                raise RuntimeError("onnxruntime is not installed. Install onnxruntime (or onnxruntime-gpu).")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            self.status.emit("Loading images…")
            img = Image.open(self.image_path).convert("RGB")
            msk = Image.open(self.mask_path).convert("L")

            if msk.size != img.size:
                # Force mask to match image
                msk = msk.resize(img.size, Image.NEAREST)

            img_np = np.array(img)
            msk_np = np.array(msk)

            # Determine bbox of mask
            ys, xs = np.where(msk_np > 0)
            if len(xs) == 0 or len(ys) == 0:
                # Nothing to inpaint
                self.status.emit("Mask is empty — saving original.")
                img.save(self.out_path, quality=self.jpeg_quality)
                self.finished_ok.emit(self.out_path)
                return

            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1

            pad_x = int((x1 - x0) * self.crop_padding_percent)
            pad_y = int((y1 - y0) * self.crop_padding_percent)

            cx0 = max(0, x0 - pad_x)
            cy0 = max(0, y0 - pad_y)
            cx1 = min(img.width, x1 + pad_x)
            cy1 = min(img.height, y1 + pad_y)

            crop_img = img.crop((cx0, cy0, cx1, cy1))
            crop_msk = msk.crop((cx0, cy0, cx1, cy1))

            # Letterbox to square, then resize to 512
            self.status.emit("Preparing ROI…")
            w, h = crop_img.size
            side = max(w, h)
            sq_img = Image.new("RGB", (side, side), (0, 0, 0))
            sq_msk = Image.new("L", (side, side), 0)

            ox = (side - w) // 2
            oy = (side - h) // 2
            sq_img.paste(crop_img, (ox, oy))
            sq_msk.paste(crop_msk, (ox, oy))

            # Feather mask if requested
            if self.feather_px > 0:
                sq_msk = sq_msk.filter(ImageFilter.GaussianBlur(radius=self.feather_px))

            in_img = sq_img.resize((512, 512), Image.BICUBIC)
            in_msk = sq_msk.resize((512, 512), Image.BILINEAR)

            # Normalize
            image_arr = (np.array(in_img).astype(np.float32) / 255.0)  # HWC RGB
            mask_arr = (np.array(in_msk).astype(np.float32) / 255.0)   # HW

            # Model expects NCHW.
            image_nchw = np.transpose(image_arr, (2, 0, 1))[None, ...]  # 1x3x512x512
            mask_nchw = mask_arr[None, None, ...]                       # 1x1x512x512

            # Providers
            providers = ["CPUExecutionProvider"]
            if self.use_gpu_if_available:
                # Try CUDA, then DirectML (Windows), then CPU.
                # (If unavailable, onnxruntime will raise and we fall back.)
                gpu_try = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                providers = gpu_try

            self.status.emit("Loading model…")
            try:
                sess = ort.InferenceSession(self.model_path, providers=providers)
            except Exception:
                # Fall back to CPU if GPU provider is missing
                sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

            # Input mapping: detect image/mask inputs robustly
            inputs = sess.get_inputs()
            if len(inputs) < 2:
                raise RuntimeError("Unexpected LaMa ONNX model inputs (expected at least 2).")

            # Pick inputs by channel count if possible
            name_image = inputs[0].name
            name_mask = inputs[1].name

            def _channels(i):
                shp = i.shape
                # expecting [1,3,512,512] and [1,1,512,512]
                try:
                    return int(shp[1])
                except Exception:
                    return None

            chans0 = _channels(inputs[0])
            chans1 = _channels(inputs[1])
            if chans0 == 1 and chans1 == 3:
                name_image, name_mask = inputs[1].name, inputs[0].name

            self.status.emit("Inpainting…")
            out = sess.run(None, {name_image: image_nchw, name_mask: mask_nchw})
            out_arr = out[0]
            # Expect 1x3x512x512, clamp
            out_arr = np.clip(out_arr, 0.0, 1.0)
            out_hwc = np.transpose(out_arr[0], (1, 2, 0))  # 512x512x3
            out_img = Image.fromarray((out_hwc * 255.0).astype(np.uint8), mode="RGB")

            # Resize back to square side, then un-letterbox to crop size
            self.status.emit("Compositing…")
            out_sq = out_img.resize((side, side), Image.BICUBIC)
            out_crop = out_sq.crop((ox, oy, ox + w, oy + h))

            # Blend only masked region
            crop_img_np = np.array(crop_img).astype(np.float32)
            out_crop_np = np.array(out_crop).astype(np.float32)
            crop_msk_np = np.array(crop_msk).astype(np.float32) / 255.0

            if self.feather_px > 0:
                # we feathered sq_msk; approximate feather on crop mask too
                crop_msk_feather = crop_msk.filter(ImageFilter.GaussianBlur(radius=self.feather_px))
                crop_msk_np = np.array(crop_msk_feather).astype(np.float32) / 255.0

            alpha = np.clip(crop_msk_np[..., None], 0.0, 1.0)
            blended = (alpha * out_crop_np + (1.0 - alpha) * crop_img_np)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            blended_img = Image.fromarray(blended, mode="RGB")

            final = img.copy()
            final.paste(blended_img, (cx0, cy0))

            _ensure_dir(os.path.dirname(self.out_path))
            ext = os.path.splitext(self.out_path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                final.save(self.out_path, quality=self.jpeg_quality)
            else:
                final.save(self.out_path)

            self.status.emit("Done.")
            self.finished_ok.emit(self.out_path)

        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# -----------------------------
# Qt Widget
# -----------------------------

class LamaPaintWidget(QWidget):
    """
    Drop-in PySide6 widget:
      - "First time use" downloads LaMa ONNX model into /models/bg/
      - Inpaint from image + mask into output path
      - Saves settings in /presets/setsave/lamapaint.json
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._settings = load_settings()

        self._dl_thread: Optional[ModelDownloadThread] = None
        self._inpaint_thread: Optional[InpaintThread] = None

        self._build_ui()
        self._refresh_model_state()

    # ---------- UI

    def _build_ui(self) -> None:
        self.setObjectName("LamaPaintWidget")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("LamaPaint (LaMa-ONNX)")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        # Status / install
        install_box = QGroupBox("Model")
        install_layout = QVBoxLayout(install_box)

        self.model_status = QLabel("Model: (checking...)")
        self.model_status.setWordWrap(True)
        install_layout.addWidget(self.model_status)

        row = QHBoxLayout()
        self.btn_first_time = QPushButton("First time use: Download model")
        self.btn_first_time.clicked.connect(self._on_first_time_use)
        row.addWidget(self.btn_first_time)

        self.install_label = QLabel("")
        self.install_label.setMinimumWidth(140)
        row.addWidget(self.install_label)

        install_layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        install_layout.addWidget(self.progress)

        root.addWidget(install_box)

        # Paths
        paths_box = QGroupBox("Inputs / Output")
        paths_layout = QFormLayout(paths_box)
        paths_layout.setLabelAlignment(Qt.AlignLeft)

        # image
        self.edit_image = QLineEdit(self._settings.get("last_image_path", ""))
        self.btn_browse_image = QPushButton("Browse…")
        self.btn_browse_image.clicked.connect(self._browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.edit_image, 1)
        image_row.addWidget(self.btn_browse_image)
        paths_layout.addRow("Image:", image_row)

        # mask
        self.edit_mask = QLineEdit(self._settings.get("last_mask_path", ""))
        self.btn_browse_mask = QPushButton("Browse…")
        self.btn_browse_mask.clicked.connect(self._browse_mask)
        mask_row = QHBoxLayout()
        mask_row.addWidget(self.edit_mask, 1)
        mask_row.addWidget(self.btn_browse_mask)
        paths_layout.addRow("Mask (white = remove):", mask_row)

        # output dir + name
        last_out = self._settings.get("last_out_dir", "")
        self.edit_out_dir = QLineEdit(last_out)
        self.btn_browse_out = QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.edit_out_dir, 1)
        out_row.addWidget(self.btn_browse_out)
        paths_layout.addRow("Output folder:", out_row)

        self.edit_out_name = QLineEdit("inpainted.png")
        paths_layout.addRow("Output name:", self.edit_out_name)

        root.addWidget(paths_box)

        # Options
        opt_box = QGroupBox("Options (VRAM-friendly defaults)")
        opt = QFormLayout(opt_box)

        self.chk_gpu = QCheckBox("Use GPU if available (CUDA/DirectML)")
        self.chk_gpu.setChecked(bool(self._settings.get("use_gpu_if_available", True)))
        opt.addRow(self.chk_gpu)

        self.spin_padding = QDoubleSpinBox()
        self.spin_padding.setRange(0.0, 1.0)
        self.spin_padding.setSingleStep(0.02)
        self.spin_padding.setDecimals(2)
        self.spin_padding.setValue(float(self._settings.get("crop_padding_percent", 0.18)))
        opt.addRow("Crop padding %:", self.spin_padding)

        self.spin_feather = QSpinBox()
        self.spin_feather.setRange(0, 64)
        self.spin_feather.setValue(int(self._settings.get("feather_px", 6)))
        opt.addRow("Mask feather (px):", self.spin_feather)

        self.spin_jpegq = QSpinBox()
        self.spin_jpegq.setRange(50, 100)
        self.spin_jpegq.setValue(int(self._settings.get("jpeg_quality", 95)))
        opt.addRow("JPEG quality:", self.spin_jpegq)

        root.addWidget(opt_box)

        # Preview
        prev_box = QGroupBox("Preview")
        prev_layout = QHBoxLayout(prev_box)

        self.preview_img = QLabel("No image loaded.")
        self.preview_img.setAlignment(Qt.AlignCenter)
        self.preview_img.setMinimumHeight(180)
        self.preview_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.preview_mask = QLabel("No mask loaded.")
        self.preview_mask.setAlignment(Qt.AlignCenter)
        self.preview_mask.setMinimumHeight(180)
        self.preview_mask.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        prev_layout.addWidget(self.preview_img, 1)
        prev_layout.addWidget(self.preview_mask, 1)

        root.addWidget(prev_box)

        # Actions
        actions = QHBoxLayout()
        self.btn_run = QPushButton("Run inpaint")
        self.btn_run.clicked.connect(self._on_run)
        actions.addWidget(self.btn_run)

        self.run_status = QLabel("")
        self.run_status.setWordWrap(True)
        actions.addWidget(self.run_status, 1)

        root.addLayout(actions)

        # Initial previews
        self.edit_image.textChanged.connect(self._update_previews)
        self.edit_mask.textChanged.connect(self._update_previews)
        self._update_previews()

    # ---------- Model install

    def _refresh_model_state(self) -> None:
        mp = _model_path()
        if os.path.exists(mp) and os.path.getsize(mp) > 10 * 1024 * 1024:
            self.model_status.setText(f"Model: Installed\n{mp}")
            self.btn_first_time.setEnabled(False)
        else:
            self.model_status.setText(
                "Model: Not installed\n"
                "Click “First time use” to download LaMa (lama_fp32.onnx) into /models/bg/."
            )
            self.btn_first_time.setEnabled(True)

    def _set_installing(self, installing: bool, text: str = "") -> None:
        self.progress.setVisible(installing)
        self.install_label.setText(text if installing else "")
        self.btn_first_time.setEnabled(not installing and not os.path.exists(_model_path()))
        self.btn_run.setEnabled(not installing)

    def _on_first_time_use(self) -> None:
        if self._dl_thread and self._dl_thread.isRunning():
            return

        url = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true"
        dst = _model_path()
        tmp = dst + ".download"

        self._set_installing(True, "Installing…")
        self.progress.setValue(0)

        spec = DownloadSpec(url=url, dst_path=dst, tmp_path=tmp)
        self._dl_thread = ModelDownloadThread(spec)
        self._dl_thread.progress.connect(self.progress.setValue)
        self._dl_thread.status.connect(lambda s: self.install_label.setText(s))
        self._dl_thread.finished_ok.connect(self._on_install_ok)
        self._dl_thread.failed.connect(self._on_install_fail)
        self._dl_thread.start()

    def _on_install_ok(self, path: str) -> None:
        self._set_installing(False, "")
        self._refresh_model_state()
        QMessageBox.information(self, "LamaPaint", f"Model installed:\n{path}")

    def _on_install_fail(self, err: str) -> None:
        self._set_installing(False, "")
        self._refresh_model_state()
        QMessageBox.critical(self, "LamaPaint — install failed", err)

    # ---------- Browsing

    def _browse_image(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.edit_image.setText(p)

    def _browse_mask(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Choose mask (white = remove)", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.edit_mask.setText(p)

    def _browse_out_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose output folder", self.edit_out_dir.text() or "")
        if p:
            self.edit_out_dir.setText(p)

    # ---------- Preview

    def _update_previews(self) -> None:
        self._set_preview_label(self.preview_img, self.edit_image.text(), grayscale=False)
        self._set_preview_label(self.preview_mask, self.edit_mask.text(), grayscale=True)

    def _set_preview_label(self, label: QLabel, path: str, grayscale: bool) -> None:
        if not path or not os.path.exists(path):
            label.setPixmap(QPixmap())
            label.setText("No file.")
            return

        try:
            img = Image.open(path)
            if grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            # Convert to QPixmap
            qimg = pil_to_qimage(img)
            pix = QPixmap.fromImage(qimg)
            # Scale to label
            target = QSize(max(1, label.width()), max(1, label.height()))
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pix)
            label.setText("")
        except Exception:
            label.setPixmap(QPixmap())
            label.setText("Preview failed.")

    # ---------- Run inpaint

    def _gather_settings(self) -> dict:
        s = dict(self._settings)
        s["use_gpu_if_available"] = bool(self.chk_gpu.isChecked())
        s["crop_padding_percent"] = float(self.spin_padding.value())
        s["feather_px"] = int(self.spin_feather.value())
        s["jpeg_quality"] = int(self.spin_jpegq.value())
        s["last_image_path"] = self.edit_image.text().strip()
        s["last_mask_path"] = self.edit_mask.text().strip()
        s["last_out_dir"] = self.edit_out_dir.text().strip()
        return s

    def _validate(self) -> Optional[str]:
        img = self.edit_image.text().strip()
        msk = self.edit_mask.text().strip()
        out_dir = self.edit_out_dir.text().strip()
        out_name = self.edit_out_name.text().strip()

        if not img or not os.path.exists(img):
            return "Please select a valid image file."
        if not msk or not os.path.exists(msk):
            return "Please select a valid mask file."
        if not out_dir:
            return "Please choose an output folder."
        if not out_name:
            return "Please enter an output name."
        if not os.path.exists(_model_path()):
            return "Model not installed. Click “First time use: Download model” first."
        return None

    def _on_run(self) -> None:
        err = self._validate()
        if err:
            QMessageBox.warning(self, "LamaPaint", err)
            return

        if self._inpaint_thread and self._inpaint_thread.isRunning():
            return

        # Save settings now
        self._settings = self._gather_settings()
        save_settings(self._settings)

        img = self.edit_image.text().strip()
        msk = self.edit_mask.text().strip()
        out_dir = self.edit_out_dir.text().strip()
        out_name = self.edit_out_name.text().strip()
        out_path = os.path.join(out_dir, out_name)

        self._set_busy(True, "Inpainting…")

        self._inpaint_thread = InpaintThread(
            image_path=img,
            mask_path=msk,
            out_path=out_path,
            use_gpu_if_available=self._settings["use_gpu_if_available"],
            crop_padding_percent=self._settings["crop_padding_percent"],
            feather_px=self._settings["feather_px"],
            jpeg_quality=self._settings["jpeg_quality"],
            model_path=_model_path(),
        )
        self._inpaint_thread.status.connect(self._on_run_status)
        self._inpaint_thread.finished_ok.connect(self._on_run_ok)
        self._inpaint_thread.failed.connect(self._on_run_fail)
        self._inpaint_thread.start()

    def _on_run_status(self, s: str) -> None:
        self.run_status.setText(s)

    def _on_run_ok(self, out_path: str) -> None:
        self._set_busy(False, "")
        self.run_status.setText("Done.")
        QMessageBox.information(self, "LamaPaint", f"Saved:\n{out_path}")

    def _on_run_fail(self, err: str) -> None:
        self._set_busy(False, "")
        QMessageBox.critical(self, "LamaPaint — failed", err)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.btn_run.setEnabled(not busy)
        self.btn_first_time.setEnabled(not busy and not os.path.exists(_model_path()))
        self.btn_browse_image.setEnabled(not busy)
        self.btn_browse_mask.setEnabled(not busy)
        self.btn_browse_out.setEnabled(not busy)
        self.run_status.setText(text if busy else "")


# -----------------------------
# PIL <-> Qt helpers
# -----------------------------

def pil_to_qimage(img: Image.Image) -> QImage:
    if img.mode == "RGB":
        arr = np.array(img)
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    elif img.mode == "L":
        arr = np.array(img)
        h, w = arr.shape
        bytes_per_line = w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
    else:
        img = img.convert("RGB")
        return pil_to_qimage(img)


# -----------------------------
# Optional quick test harness
# -----------------------------
if __name__ == "__main__":  # pragma: no cover
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = LamaPaintWidget()
    w.resize(900, 700)
    w.show()
    sys.exit(app.exec())
