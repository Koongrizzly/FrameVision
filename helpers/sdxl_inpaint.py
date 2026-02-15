"""
SDXL Inpaint UI (PySide6) — standalone pane + basic inpaint pipeline runner.

- Models: scans <project_root>/models/inpaint/ for SDXL inpaint checkpoints (.safetensors/.ckpt) and/or Diffusers folders.
- Runs Diffusers StableDiffusionXLInpaintPipeline in a background thread to keep UI responsive.
- Optional simple mask painter (paint white = inpaint area).

Drop this file into: <project_root>/helpers/
Typical import: from helpers.sdxl_inpaint_ui import SDXLInpaintPane

Notes:
- Requires: PySide6, Pillow, torch, diffusers (and optionally safetensors).
- Uses local files only (no downloads).
"""

from __future__ import annotations

import os
import sys
import time
import math
import traceback
import json
import subprocess
import tempfile
import uuid
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (QMutex, QObject, QPoint, QRect, QSize, Qt, QThread,
                            QEvent,
                            QTimer, Signal, Slot)
from PySide6.QtGui import (QAction, QBrush, QColor, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QCheckBox,
                               QComboBox, QDialog, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QDoubleSpinBox,
                               QSpinBox, QPlainTextEdit, QProgressBar,
                               QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget)


# -------------------------
# Paths / helpers
# -------------------------

def project_root() -> Path:
    # helpers/sdxl_inpaint_ui.py -> project root is one level up from helpers/
    return Path(__file__).resolve().parent.parent


def models_inpaint_dir() -> Path:
    return project_root() / "models" / "inpaint"


def output_dir() -> Path:
    return project_root() / "output" / "inpaint"


def presets_bin_dir() -> Path:
    return project_root() / "presets" / "bin"

def setsave_dir() -> Path:
    # Persistent settings live here (requested): <project_root>/presets/setsave/
    return project_root() / "presets" / "setsave"


def inpaint_settings_path() -> Path:
    return setsave_dir() / "sdxl_inpaint_settings.json"


def venv_python_exe() -> Optional[Path]:
    # Windows-first; fall back to POSIX.
    root = project_root()
    win = root / ".venv" / "Scripts" / "python.exe"
    posix = root / ".venv" / "bin" / "python"
    if win.exists():
        return win
    if posix.exists():
        return posix
    return None


def sdxl_inpaint_python_exe() -> Optional[Path]:
    """Return the python executable for the dedicated SDXL-inpaint environment (.sdxl_inpaint)."""
    root = project_root()
    win = root / ".sdxl_inpaint" / "Scripts" / "python.exe"
    posix = root / ".sdxl_inpaint" / "bin" / "python"
    if win.exists():
        return win
    if posix.exists():
        return posix
    return None


def _ensure_sdxl_runner(tmp_dir: Path) -> Path:
    """
    Writes a tiny runner script (no PySide imports) into tmp_dir and returns its path.
    This script is executed under the dedicated .sdxl_inpaint environment.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    runner = tmp_dir / "_sdxl_inpaint_runner.py"
    tag = "# FRAMEVISION_SDXL_INPAINT_RUNNER_V2"
    if runner.exists():
        try:
            head = runner.read_text(encoding="utf-8", errors="ignore")[:2000]
            if tag in head:
                return runner
        except Exception:
            pass

    runner_code = r'''{tag}
import json, sys, traceback
from pathlib import Path

def _looks_like_linear_proj_mismatch(msg: str) -> bool:
    return ("proj_in.weight" in msg or "transformer_blocks" in msg) and ("expected shape" in msg or "expected shape tensor" in msg)

def _apply_scheduler(pipe, key: str):
    try:
        from diffusers import (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                               DPMSolverMultistepScheduler, DDIMScheduler,
                               HeunDiscreteScheduler)
    except Exception:
        return
    sched_map = {{
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "ddim": DDIMScheduler,
        "heun": HeunDiscreteScheduler,
    }}
    cls = sched_map.get(key)
    if cls is None:
        return
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config)
    except Exception:
        pass


    # -------------------------
    # Settings persistence
    # -------------------------

    def _load_settings_file(self) -> Dict[str, object]:
        path = getattr(self, "_settings_path", inpaint_settings_path())
        try:
            if not Path(path).exists():
                return {}
            raw = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _collect_settings(self) -> Dict[str, object]:
        try:
            model_data = str(self.cmb_model.currentData() or "")
        except Exception:
            model_data = ""
        try:
            sched_data = str(self.cmb_sched.currentData() or "")
        except Exception:
            sched_data = ""

        return {
            "model": model_data,
            "device": str(self.cmb_device.currentText()).strip(),
            "image": str(self.ed_image.text()),
            "mask": str(self.ed_mask.text()),
            "prompt": str(self.txt_prompt.toPlainText()),
            "negative": str(self.txt_negative.toPlainText()),

            "steps": int(self.sp_steps.value()),
            "cfg": float(self.sp_cfg.value()),
            "strength": float(self.sp_strength.value()),
            "seed": int(self.sp_seed.value()),
            "width": int(self.sp_w.value()),
            "height": int(self.sp_h.value()),

            "match_image": bool(self.chk_match_image.isChecked()),
            "cpu_offload": bool(self.chk_cpu_offload.isChecked()),
            "attention_slicing": bool(self.chk_attention_slicing.isChecked()),
            "vae_tiling": bool(self.chk_vae_tiling.isChecked()),
            "scheduler": sched_data,
            "invert_mask": bool(self.btn_invert_mask.isChecked()),

            "last_image_dir": str(getattr(self, "_last_image_dir", str(project_root()))),
            "last_mask_dir": str(getattr(self, "_last_mask_dir", str(project_root()))),
        }

    @Slot()
    def _schedule_save(self):
        if getattr(self, "_loading_settings", False):
            return
        if not getattr(self, "_persistence_enabled", True):
            return
        t = getattr(self, "_save_timer", None)
        if t is not None:
            t.start(400)

    def _save_settings_now(self):
        # Write to <project_root>/presets/setsave/sdxl_inpaint_settings.json
        try:
            setsave_dir().mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        path = Path(str(getattr(self, "_settings_path", inpaint_settings_path())))
        tmp = Path(str(path) + ".tmp")

        try:
            data = self._collect_settings()
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)  # atomic on most platforms
        except Exception:
            # Fallback to non-atomic write
            try:
                path.write_text(json.dumps(self._collect_settings(), indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    @Slot()
    def _apply_loaded_settings(self):
        s = getattr(self, "_loaded_settings", None)
        if not isinstance(s, dict) or not s:
            return

        self._loading_settings = True
        try:
            # Dialog folders
            try:
                self._last_image_dir = str(s.get("last_image_dir") or self._last_image_dir)
                self._last_mask_dir = str(s.get("last_mask_dir") or self._last_mask_dir)
            except Exception:
                pass

            # Match-image first to avoid autosize overriding saved size unexpectedly
            try:
                self.chk_match_image.setChecked(bool(s.get("match_image", True)))
            except Exception:
                pass

            # Text fields
            try:
                self.ed_image.setText(str(s.get("image") or ""))
            except Exception:
                pass
            try:
                self.ed_mask.setText(str(s.get("mask") or ""))
            except Exception:
                pass
            try:
                self.txt_prompt.setPlainText(str(s.get("prompt") or ""))
            except Exception:
                pass
            try:
                self.txt_negative.setPlainText(str(s.get("negative") or ""))
            except Exception:
                pass

            # Numbers (clamped)
            def _clamp_set(spin, key: str):
                try:
                    v = s.get(key, None)
                    if v is None:
                        return
                    v = int(float(v))
                    v = max(spin.minimum(), min(spin.maximum(), v))
                    spin.setValue(v)
                except Exception:
                    pass

            _clamp_set(self.sp_steps, "steps")
            _clamp_set(self.sp_seed, "seed")
            _clamp_set(self.sp_w, "width")
            _clamp_set(self.sp_h, "height")

            try:
                cfg = float(s.get("cfg", self.sp_cfg.value()))
                cfg = max(self.sp_cfg.minimum(), min(self.sp_cfg.maximum(), cfg))
                self.sp_cfg.setValue(cfg)
            except Exception:
                pass
            try:
                strength = float(s.get("strength", self.sp_strength.value()))
                strength = max(self.sp_strength.minimum(), min(self.sp_strength.maximum(), strength))
                self.sp_strength.setValue(strength)
            except Exception:
                pass

            # Device
            try:
                dev = str(s.get("device") or "").strip()
                if dev:
                    idx = self.cmb_device.findText(dev)
                    if idx >= 0:
                        self.cmb_device.setCurrentIndex(idx)
            except Exception:
                pass

            # Checkboxes
            try:
                self.chk_cpu_offload.setChecked(bool(s.get("cpu_offload", False)))
                self.chk_attention_slicing.setChecked(bool(s.get("attention_slicing", True)))
                self.chk_vae_tiling.setChecked(bool(s.get("vae_tiling", False)))
            except Exception:
                pass

            # Scheduler
            try:
                sched_key = str(s.get("scheduler") or "")
                if sched_key:
                    for i in range(self.cmb_sched.count()):
                        if str(self.cmb_sched.itemData(i)) == sched_key:
                            self.cmb_sched.setCurrentIndex(i)
                            break
            except Exception:
                pass

            # Invert mask
            try:
                self.btn_invert_mask.setChecked(bool(s.get("invert_mask", False)))
            except Exception:
                pass

            # Model selection (after refresh_models has populated cmb_model)
            try:
                model_path = str(s.get("model") or "")
                if model_path:
                    for i in range(self.cmb_model.count()):
                        if str(self.cmb_model.itemData(i)) == model_path:
                            self.cmb_model.setCurrentIndex(i)
                            break
            except Exception:
                pass

            # If match-image is on, apply autosize once now
            try:
                if self.chk_match_image.isChecked():
                    self._maybe_autosize()
            except Exception:
                pass

            try:
                self._update_mask_preview()
            except Exception:
                pass

        finally:
            self._loading_settings = False

    @Slot()
    def _enable_persistence(self):
        # Enable after initial load/app restores, then save once to lock in.
        self._persistence_enabled = True
        self._schedule_save()

    def closeEvent(self, event):  # type: ignore
        try:
            if getattr(self, "_persistence_enabled", False):
                self._save_settings_now()
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))

    from PIL import Image, ImageOps
    import torch

    model_path = Path(job["model_path"])
    image_path = Path(job["image_path"])
    mask_path = Path(job["mask_path"])
    out_path = Path(job["out_path"])

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    w = int(job["width"]); h = int(job["height"])
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
    if mask.size != (w, h):
        mask = mask.resize((w, h), resample=Image.Resampling.NEAREST)

    if job.get("invert_mask", False):
        mask = ImageOps.invert(mask)

    device = job.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    prompt = job.get("prompt", "")
    negative = job.get("negative_prompt") or None

    steps = int(job.get("steps", 30))
    cfg = float(job.get("cfg", 5.5))
    strength = float(job.get("strength", 0.75))
    seed = int(job.get("seed", -1))

    gen = None
    if seed >= 0:
        gen = torch.Generator(device=device).manual_seed(seed)

    from diffusers import StableDiffusionXLInpaintPipeline

    def _load_inpaint():
        if model_path.is_dir():
            return StableDiffusionXLInpaintPipeline.from_pretrained(
                str(model_path), torch_dtype=dtype, local_files_only=True
            )
        if not hasattr(StableDiffusionXLInpaintPipeline, "from_single_file"):
            raise RuntimeError("diffusers build lacks from_single_file()")
        return StableDiffusionXLInpaintPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=(model_path.suffix.lower() == ".safetensors"),
            local_files_only=True,
        )

    pipe = None
    mode = "inpaint"
    try:
        pipe = _load_inpaint()
    except Exception as e:
        msg = str(e)
        if not _looks_like_linear_proj_mismatch(msg):
            raise
        from diffusers import StableDiffusionXLImg2ImgPipeline
        mode = "img2img_compat"
        if model_path.is_dir():
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                str(model_path), torch_dtype=dtype, local_files_only=True
            )
        else:
            if not hasattr(StableDiffusionXLImg2ImgPipeline, "from_single_file"):
                raise
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(model_path),
                torch_dtype=dtype,
                use_safetensors=(model_path.suffix.lower() == ".safetensors"),
                local_files_only=True,
            )

    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
    except Exception:
        pass

    pipe = pipe.to(device)

    try:
        if job.get("attention_slicing", False) and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        if job.get("vae_tiling", False) and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
    except Exception:
        pass

    try:
        if job.get("cpu_offload", False) and device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
    except Exception:
        pass

    _apply_scheduler(pipe, job.get("scheduler", "euler"))

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        image=img,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
    )
    if gen is not None:
        kwargs["generator"] = gen

    if mode == "inpaint":
        kwargs["mask_image"] = mask

        # Force requested size. Some diffusers builds will otherwise default to 1024x1024.
    kwargs["width"] = w
    kwargs["height"] = h
    # SDXL micro-conditioning (helps preserve framing when supported).
    kwargs["original_size"] = (h, w)
    kwargs["target_size"] = (h, w)
    kwargs["crop_coords_top_left"] = (0, 0)

    def _safe_call(pipe, kwargs):
        try:
            return pipe(**kwargs)
        except TypeError:
            # Older/alternate pipelines may not accept micro-conditioning args.
            for k in ("original_size", "target_size", "crop_coords_top_left"):
                kwargs.pop(k, None)
            try:
                return pipe(**kwargs)
            except TypeError:
                # As a last resort, drop width/height too.
                for k in ("width", "height"):
                    kwargs.pop(k, None)
                return pipe(**kwargs)

    result = _safe_call(pipe, kwargs)
    out = result.images[0] if hasattr(result, "images") else None

    if mode != "inpaint" and out is not None:
        out = Image.composite(out, img, mask.convert("L"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print("OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)
'''.format(tag=tag)

    runner.write_text(runner_code, encoding="utf-8")
    return runner

def find_exe(name: str) -> Optional[Path]:
    # Prefer presets/bin, then PATH.
    p = presets_bin_dir() / name
    if p.exists():
        return p
    # Search PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        cand = Path(d) / name
        if cand.exists():
            return cand
    return None


def _is_diffusers_model_dir(p: Path) -> bool:
    # A heuristic: diffusers folders often have model_index.json.
    return p.is_dir() and (p / "model_index.json").exists()


def scan_inpaint_models() -> List[Tuple[str, Path]]:
    """
    Returns list of (display_name, path) in models/inpaint.
    Supports:
      - single-file checkpoints: .safetensors, .ckpt
      - diffusers folders (with model_index.json)
    """
    base = models_inpaint_dir()
    base.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, Path]] = []

    # Directories (diffusers)
    for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
        if _is_diffusers_model_dir(p):
            items.append((p.name, p))

    # Files
    exts = {".safetensors", ".ckpt"}
    for p in sorted(base.glob("*"), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in exts:
            items.append((p.stem, p))

    return items


def pil_from_qimage(img: QImage):
    # Convert QImage to PIL.Image without numpy dependency
    from PIL import Image  # type: ignore

    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(w * h * 4)
    return Image.frombuffer("RGBA", (w, h), bytes(ptr), "raw", "RGBA", 0, 1)


def qimage_from_pil(im):
    # Convert PIL.Image to QImage
    from PIL import Image  # type: ignore

    if im.mode not in ("RGBA", "RGB", "L"):
        im = im.convert("RGBA")
    if im.mode == "RGB":
        im = im.convert("RGBA")
    if im.mode == "L":
        im = im.convert("RGBA")
    w, h = im.size
    data = im.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format.Format_RGBA8888)
    # Make deep copy to own the memory
    return qimg.copy()


def ensure_mask_l_size(mask, size: Tuple[int, int]):
    """
    SDXL inpaint expects mask as PIL 'L', white=painted area.
    """
    from PIL import Image  # type: ignore

    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return mask


# -------------------------
# Simple mask painter
# -------------------------

class MaskPainter(QWidget):
    """
    Paints a binary mask on top of an image.
    White = inpaint area. Black = keep.

    Controls:
      - Left mouse: paint white
      - Right mouse: erase (paint black)
      - Mouse wheel: brush size
    """
    maskChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Settings persistence (requested):
        # Store all UI state in: <project_root>/presets/setsave/sdxl_inpaint_settings.json
        self._settings_path = inpaint_settings_path()
        self._loaded_settings: Dict[str, object] = self._load_settings_file()
        self._loading_settings = False

        # Separate remembered folders for image and mask dialogs
        self._last_image_dir = str(project_root())
        self._last_mask_dir = str(project_root())
        try:
            self._last_image_dir = str(self._loaded_settings.get("last_image_dir") or self._last_image_dir)
            self._last_mask_dir = str(self._loaded_settings.get("last_mask_dir") or self._last_mask_dir)
        except Exception:
            pass

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings_now)
        self.setMouseTracking(True)
        self._base: Optional[QImage] = None
        self._mask: Optional[QImage] = None
        self._brush_radius = 24
        self._last = QPoint()
        self._drawing = False
        self._draw_white = True
        self.setMinimumSize(512, 512)

    def set_image(self, img: QImage):
        self._base = img.copy()
        self._mask = QImage(img.size(), QImage.Format.Format_Grayscale8)
        self._mask.fill(0)
        self.update()
        self.maskChanged.emit()

    def has_image(self) -> bool:
        return self._base is not None and self._mask is not None

    def mask_qimage(self) -> Optional[QImage]:
        return self._mask.copy() if self._mask is not None else None

    def clear_mask(self):
        if self._mask is not None:
            self._mask.fill(0)
            self.update()
            self.maskChanged.emit()

    def brush_radius(self) -> int:
        return self._brush_radius

    def set_brush_radius(self, r: int):
        self._brush_radius = max(1, min(256, int(r)))
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        step = 2 if abs(delta) < 240 else 4
        if delta > 0:
            self.set_brush_radius(self._brush_radius + step)
        else:
            self.set_brush_radius(self._brush_radius - step)

    def mousePressEvent(self, event):
        if not self.has_image():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._draw_white = True
            self._last = event.position().toPoint()
            self._stroke(self._last, self._last, white=True)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drawing = True
            self._draw_white = False
            self._last = event.position().toPoint()
            self._stroke(self._last, self._last, white=False)

    def mouseMoveEvent(self, event):
        if not self.has_image():
            return
        pos = event.position().toPoint()
        if self._drawing:
            self._stroke(self._last, pos, white=self._draw_white)
            self._last = pos

    def mouseReleaseEvent(self, event):
        if self._drawing:
            self._drawing = False
            self.maskChanged.emit()

    def _stroke(self, a: QPoint, b: QPoint, white: bool):
        if self._mask is None:
            return
        painter = QPainter(self._mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255) if white else QColor(0, 0, 0))
        pen.setWidth(self._brush_radius * 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(a, b)
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if not self.has_image():
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load an image to paint a mask")
            painter.end()
            return

        base = self._base
        mask = self._mask
        assert base is not None and mask is not None

        # Fit image into widget
        target = self.rect().adjusted(8, 8, -8, -8)
        scaled = base.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        img_rect = QRect(x, y, scaled.width(), scaled.height())

        painter.drawImage(img_rect, scaled)

        # Overlay mask with alpha
        mask_rgba = QImage(mask.size(), QImage.Format.Format_ARGB32)
        mask_rgba.fill(Qt.GlobalColor.transparent)

        mp = QPainter(mask_rgba)
        mp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        mp.fillRect(mask_rgba.rect(), QColor(0, 0, 0, 0))
        mp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # White pixels -> red overlay
        # Convert grayscale mask to alpha overlay quickly
        # (simple per-pixel loop is OK for typical sizes, but avoid huge images here)
        mw, mh = mask.width(), mask.height()
        # Limit heavy work by using scaled version for display
        disp = mask.scaled(img_rect.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        dp = QPainter()
        dp.begin(painter)
        # Create colorized overlay from disp
        overlay = QImage(disp.size(), QImage.Format.Format_ARGB32)
        overlay.fill(Qt.GlobalColor.transparent)
        op = QPainter(overlay)
        op.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        op.fillRect(overlay.rect(), QColor(0, 0, 0, 0))
        op.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        # Use disp as alpha mask
        op.drawImage(0, 0, disp)
        op.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        op.fillRect(overlay.rect(), QColor(255, 0, 0, 120))
        op.end()

        dp.drawImage(img_rect.topLeft(), overlay)
        dp.end()

        # Brush preview circle
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawText(10, self.height() - 10,
                         f"Brush: {self._brush_radius}px  (wheel to change)   "
                         f"Left=paint  Right=erase")
        painter.end()


class MaskPainterDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Paint Mask (white = inpaint area)")
        self.setModal(True)
        self.resize(900, 700)

        self.painter = MaskPainter(self)
        self.btn_clear = QPushButton("Clear mask")
        self.btn_ok = QPushButton("Use mask")
        self.btn_cancel = QPushButton("Cancel")

        self.btn_ok.setDefault(True)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)

        layout = QVBoxLayout(self)
        layout.addWidget(self.painter, 1)
        layout.addLayout(btns)

        self.btn_clear.clicked.connect(self.painter.clear_mask)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def set_image(self, qimg: QImage):
        self.painter.set_image(qimg)

    def get_mask(self) -> Optional[QImage]:
        return self.painter.mask_qimage()


# -------------------------
# Pipeline worker
# -------------------------

@dataclass
class InpaintParams:
    model_path: Path
    prompt: str
    negative_prompt: str
    steps: int
    cfg: float
    strength: float
    seed: int
    width: int
    height: int
    match_image: bool
    device: str  # "auto"|"cuda"|"cpu"
    cpu_offload: bool
    attention_slicing: bool
    vae_tiling: bool
    scheduler: str  # key
    invert_mask: bool  # if True, invert white/black before inpainting (only if a mask is provided)


class InpaintWorker(QObject):
    progress = Signal(int)        # 0..100
    status = Signal(str)
    finished = Signal(object)     # PIL.Image or None
    failed = Signal(str)          # error string

    def __init__(self, image_path: Path, mask_path: Optional[Path], mask_qimage: Optional[QImage], params: InpaintParams):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mask_qimage = mask_qimage
        self.params = params
        self._cancel = False
        self._mutex = QMutex()

    @Slot()
    def cancel(self):
        self._mutex.lock()
        self._cancel = True
        self._mutex.unlock()

    def _is_cancelled(self) -> bool:
        self._mutex.lock()
        c = self._cancel
        self._mutex.unlock()
        return c

    @Slot()
    def run(self):
        try:
            self.progress.emit(1)
            self.status.emit("Loading image/mask…")

            from PIL import Image  # type: ignore

            img = Image.open(self.image_path).convert("RGB")

            # Target size
            src_w, src_h = img.size
            if bool(getattr(self.params, "match_image", False)):
                w, h = src_w, src_h
            else:
                w, h = int(self.params.width), int(self.params.height)
                if w <= 0 or h <= 0:
                    w, h = src_w, src_h

            if img.size != (w, h):
                img = img.resize((w, h), resample=Image.Resampling.LANCZOS)

            # Mask (SDXL inpaint expects L, white=painted area)
            has_mask = False
            if self.mask_qimage is not None:
                mask_pil = pil_from_qimage(self.mask_qimage).convert("L")
                has_mask = True
            elif self.mask_path is not None and self.mask_path.exists():
                mask_pil = Image.open(self.mask_path).convert("L")
                has_mask = True
            else:
                # default empty mask (no changes)
                mask_pil = Image.new("L", (w, h), 0)

            mask_pil = ensure_mask_l_size(mask_pil, (w, h))

            if self._is_cancelled():
                self.failed.emit("Cancelled.")
                return

            
            self.status.emit("Preparing SDXL-Inpaint job (external env)…")
            self.progress.emit(5)

            py = sdxl_inpaint_python_exe()
            if py is None:
                raise RuntimeError(
                    "SDXL-Inpaint environment not found. Expected: <project>/.sdxl_inpaint/.\n"
                    "Please run: presets/extra_env/sdxl_inpaint_install.bat"
                )

            # Prepare a temp job folder + runner
            tmp_dir = output_dir() / "_tmp_sdxl_inpaint"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            runner = _ensure_sdxl_runner(tmp_dir)

            # Write inputs to temp files (keep main env free of diffusers/torch imports)
            tag = uuid.uuid4().hex[:10]
            tmp_in = tmp_dir / f"in_{tag}.png"
            tmp_mask = tmp_dir / f"mask_{tag}.png"
            tmp_out = tmp_dir / f"out_{tag}.png"
            tmp_job = tmp_dir / f"job_{tag}.json"

            img.save(tmp_in)
            mask_pil.save(tmp_mask)

            job = {
                "model_path": str(self.params.model_path),
                "prompt": self.params.prompt.strip(),
                "negative_prompt": (self.params.negative_prompt.strip() or ""),
                "steps": int(self.params.steps),
                "cfg": float(self.params.cfg),
                "strength": float(self.params.strength),
                "seed": int(self.params.seed),
                "width": int(w),
                "height": int(h),
                "device": str(self.params.device),
                "cpu_offload": bool(self.params.cpu_offload),
                "attention_slicing": bool(self.params.attention_slicing),
                "vae_tiling": bool(self.params.vae_tiling),
                "scheduler": str(self.params.scheduler),
                "invert_mask": bool(getattr(self.params, "invert_mask", False)),
                "image_path": str(tmp_in),
                "mask_path": str(tmp_mask),
                "out_path": str(tmp_out),
            }
            tmp_job.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

            self.status.emit("Running SDXL-Inpaint in dedicated environment…")
            self.progress.emit(10)

            cmd = [str(py), str(runner), str(tmp_job)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            while True:
                if self._is_cancelled():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise RuntimeError("Cancelled by user")
                rc = proc.poll()
                if rc is not None:
                    break
                time.sleep(0.1)

            out_stdout, out_stderr = proc.communicate()

            if proc.returncode != 0:
                msg = (out_stderr or out_stdout or "").strip()
                if not msg:
                    msg = f"SDXL-Inpaint runner failed with exit code {proc.returncode}"
                raise RuntimeError(msg)

            if not tmp_out.exists():
                raise RuntimeError("SDXL-Inpaint runner produced no output image.")

            # Load output back into main process
            out = Image.open(tmp_out).convert("RGB")

            self.progress.emit(100)
            self.status.emit("Done.")
            self.finished.emit(out)
            return

            # Configure pipeline
            if hasattr(pipe, "safety_checker"):
                try:
                    pipe.safety_checker = None  # speed / avoid missing deps
                except Exception:
                    pass

            if device == "cuda":
                pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")
                if getattr(self, '_pipe_mode', 'inpaint') != 'inpaint':
                    self.status.emit('Compat mode: SDXL img2img + mask composite')

            # Optional memory features
            try:
                if self.params.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
            except Exception:
                pass

            try:
                if self.params.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
            except Exception:
                pass

            try:
                if self.params.cpu_offload and device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
            except Exception:
                pass

            # Scheduler
            self._apply_scheduler(pipe, self.params.scheduler)

            gen = None
            if self.params.seed >= 0:
                gen = torch.Generator(device=device).manual_seed(int(self.params.seed))

            # Progress callback
            total = max(1, int(self.params.steps))
            last_emit = 0

            def _cb(step: int, timestep: int, latents):
                nonlocal last_emit
                if self._is_cancelled():
                    raise RuntimeError("Cancelled by user")
                # step is 0..total-1
                pct = int(10 + (step + 1) * 85 / total)
                if pct != last_emit:
                    last_emit = pct
                    self.progress.emit(pct)

            self.status.emit("Running inpaint…")
            self.progress.emit(10)

            # Call pipe with best-effort compatibility across diffusers versions
            kwargs = dict(
                prompt=self.params.prompt.strip(),
                negative_prompt=self.params.negative_prompt.strip() or None,
                image=img,
                mask_image=mask_pil,
                num_inference_steps=int(self.params.steps),
                guidance_scale=float(self.params.cfg),
                strength=float(self.params.strength),
            )
            # If we're in compat mode (img2img), the pipeline doesn't accept mask_image.
            if getattr(self, '_pipe_mode', 'inpaint') != 'inpaint':
                kwargs.pop('mask_image', None)
            if gen is not None:
                kwargs["generator"] = gen

            # Prefer callback_on_step_end if available
            try:
                sig = pipe.__call__.__code__.co_varnames  # type: ignore
            except Exception:
                sig = ()

            # Ensure non-square sizes propagate: some diffusers versions default to 1024x1024 unless width/height are provided.
            if "width" in sig:
                kwargs["width"] = w
            if "height" in sig:
                kwargs["height"] = h
            # SDXL micro-conditioning (helps preserve framing when available).
            if "original_size" in sig:
                kwargs["original_size"] = (h, w)
            if "target_size" in sig:
                kwargs["target_size"] = (h, w)
            if "crop_coords_top_left" in sig:
                kwargs["crop_coords_top_left"] = (0, 0)

            result = None
            if "callback_on_step_end" in sig:
                kwargs["callback_on_step_end"] = _cb
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                result = pipe(**kwargs)
            elif "callback" in sig:
                kwargs["callback"] = _cb
                kwargs["callback_steps"] = 1
                result = pipe(**kwargs)
            else:
                result = pipe(**kwargs)

            out = result.images[0] if hasattr(result, "images") else None
            # Compat mode: SDXL img2img + mask composite
            if getattr(self, '_pipe_mode', 'inpaint') != 'inpaint':
                try:
                    from PIL import Image  # type: ignore
                    # Only apply generated pixels inside the (white) mask
                    if has_mask:
                        m = mask_pil.convert('L')
                        if m.size != out.size:
                            m = m.resize(out.size)
                        out = Image.composite(out, img, m)
                except Exception:
                    pass

            self.progress.emit(100)
            self.status.emit("Done.")
            self.finished.emit(out)

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.failed.emit(err)

    def _load_pipeline(self, model_path: Path, dtype):
        """
        Load pipeline for SDXL inpaint.
    
        Returns (pipe, mode) where mode is:
          - 'inpaint' for StableDiffusionXLInpaintPipeline
          - 'img2img_compat' for StableDiffusionXLImg2ImgPipeline + mask composite fallback
    
        Some community 'inpaint' checkpoints store attention proj weights as 2D Linear (640x640).
        Diffusers' SDXL-Inpaint loader expects Conv1x1 (640x640x1x1) and will raise a ValueError.
        In that case, we fall back to SDXL img2img and composite the result using the mask.
        """
        from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore
    
        def _looks_like_linear_proj_mismatch(err: Exception) -> bool:
            s = str(err)
            return (
                'proj_in.weight' in s and ('(640, 640, 1, 1)' in s or 'size=(640, 640, 1, 1)' in s)
                and ('torch.Size([640, 640])' in s or '(640, 640)' in s)
            )
    
        def _load_inpaint_dir(p: Path):
            return StableDiffusionXLInpaintPipeline.from_pretrained(
                str(p),
                torch_dtype=dtype,
                local_files_only=True,
            )
    
        def _load_inpaint_file(p: Path):
            if not hasattr(StableDiffusionXLInpaintPipeline, 'from_single_file'):
                raise RuntimeError(
                    "Your installed diffusers version doesn't support "
                    "StableDiffusionXLInpaintPipeline.from_single_file(). "
                    "Either use a Diffusers-format model folder (with model_index.json), "
                    "or upgrade diffusers in your .venv."
                )
            return StableDiffusionXLInpaintPipeline.from_single_file(
                str(p),
                torch_dtype=dtype,
                use_safetensors=p.suffix.lower() == '.safetensors',
                local_files_only=True,
            )
    
        # First try: native SDXL inpaint
        try:
            if model_path.is_dir():
                return _load_inpaint_dir(model_path), 'inpaint'
            return _load_inpaint_file(model_path), 'inpaint'
        except Exception as e:
            # Fallback: SDXL img2img + mask composite for linear-proj checkpoints
            if not _looks_like_linear_proj_mismatch(e):
                raise
            from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore
    
            def _load_img2img_dir(p: Path):
                return StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    str(p),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
    
            def _load_img2img_file(p: Path):
                if not hasattr(StableDiffusionXLImg2ImgPipeline, 'from_single_file'):
                    # Some diffusers builds don't expose this; let the original error bubble
                    raise
                return StableDiffusionXLImg2ImgPipeline.from_single_file(
                    str(p),
                    torch_dtype=dtype,
                    use_safetensors=p.suffix.lower() == '.safetensors',
                    local_files_only=True,
                )
    
            if model_path.is_dir():
                pipe = _load_img2img_dir(model_path)
            else:
                pipe = _load_img2img_file(model_path)
    
            # mark mode for UI/debug
            try:
                setattr(pipe, '_framevision_inpaint_mode', 'img2img_compat')
            except Exception:
                pass
            return pipe, 'img2img_compat'
    def _apply_scheduler(self, pipe, key: str):
        """
        Swap scheduler types while keeping config.
        """
        try:
            from diffusers import (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                   DPMSolverMultistepScheduler, DDIMScheduler,
                                   HeunDiscreteScheduler)  # type: ignore
        except Exception:
            return

        sched_map = {
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "ddim": DDIMScheduler,
            "heun": HeunDiscreteScheduler,
        }
        cls = sched_map.get(key)
        if cls is None:
            return
        try:
            pipe.scheduler = cls.from_config(pipe.scheduler.config)
        except Exception:
            pass


class _PipelineCache:
    """
    Simple per-process pipeline cache to avoid reload on every run.
    Keyed by absolute path + dtype.
    """
    _cache: Dict[Tuple[str, str], object] = {}

    @classmethod
    def _key(cls, model_path: Path, dtype) -> Tuple[str, str]:
        return (str(model_path.resolve()), str(dtype))

    @classmethod
    def get(cls, model_path: Path, dtype):
        return cls._cache.get(cls._key(model_path, dtype))

    @classmethod
    def put(cls, model_path: Path, dtype, pipe):
        cls._cache[cls._key(model_path, dtype)] = pipe


# -------------------------
# Main UI pane
# -------------------------

class SDXLInpaintPane(QWidget):
    """
    A drop-in QWidget pane for SDXL inpaint.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # FV: prevent parent scroll from hijacking the whole app when this tab is active.
        # Make this pane report a minimal size and let the parent layout decide the final height.
        # FV: prevent parent scroll (sizeHint)
        try:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setMinimumSize(0, 0)
        except Exception:
            pass


        # Settings persistence (requested):
        # Store all UI state in: <project_root>/presets/setsave/sdxl_inpaint_settings.json
        self._settings_path = inpaint_settings_path()
        self._loaded_settings: Dict[str, object] = self._load_settings_file()
        self._loading_settings = False
        self._persistence_enabled = False  # enable after initial load/app restores

        # Separate remembered folders for image and mask dialogs
        self._last_image_dir = str(project_root())
        self._last_mask_dir = str(project_root())
        try:
            self._last_image_dir = str(self._loaded_settings.get("last_image_dir") or self._last_image_dir)
            self._last_mask_dir = str(self._loaded_settings.get("last_mask_dir") or self._last_mask_dir)
        except Exception:
            pass

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings_now)

        self._model_items: List[Tuple[str, Path]] = []
        self._last_output_path: Optional[Path] = None
        self._mask_qimage: Optional[QImage] = None
        # Track which image a painted mask belongs to, so we can avoid stale masks
        self._painted_mask_source_image: str = ""
        self._last_image_path_text: str = ""

        # --- Controls
        self.cmb_model = QComboBox()
        self.btn_refresh_models = QPushButton("Refresh")
        self.lbl_model_path = QLabel("")
        self.lbl_model_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_model_path.setStyleSheet("color: #aaa;")

        self.ed_image = QLineEdit()
        self.btn_browse_image = QPushButton("Browse…")

        self.ed_mask = QLineEdit()
        self.btn_browse_mask = QPushButton("Browse…")
        self.btn_paint_mask = QPushButton("Paint…")
        self.btn_clear_paint = QPushButton("Clear painted")
        self.btn_invert_mask = QPushButton("Invert mask")
        self.btn_invert_mask.setCheckable(True)
        self.btn_invert_mask.setToolTip("Swap white/black in the mask before inpainting (white = area to change).")

        self.mask_preview = QLabel()
        self.mask_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_preview.setMinimumHeight(180)
        self.mask_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mask_preview.setStyleSheet("background: #111; border: 1px solid #333;")
        self.mask_preview.setText("Mask preview (white = will change)")

        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setPlaceholderText("Prompt…")
        self.txt_negative = QPlainTextEdit()
        self.txt_negative.setPlaceholderText("Negative prompt (optional)…")

        self.sp_steps = QSpinBox()
        self.sp_steps.setRange(1, 200)
        self.sp_steps.setValue(30)

        self.sp_cfg = QDoubleSpinBox()
        self.sp_cfg.setRange(0.0, 40.0)
        self.sp_cfg.setDecimals(2)
        self.sp_cfg.setValue(5.5)

        self.sp_strength = QDoubleSpinBox()
        self.sp_strength.setRange(0.0, 1.0)
        self.sp_strength.setDecimals(3)
        self.sp_strength.setSingleStep(0.05)
        self.sp_strength.setValue(0.75)

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(-1, 2_147_483_647)
        self.sp_seed.setValue(-1)

        self.sp_w = QSpinBox()
        self.sp_w.setRange(64, 4096)
        self.sp_w.setSingleStep(64)
        self.sp_w.setValue(1024)

        self.sp_h = QSpinBox()
        self.sp_h.setRange(64, 4096)
        self.sp_h.setSingleStep(64)
        self.sp_h.setValue(1024)

        self.chk_match_image = QCheckBox("Match size to loaded image")
        self.chk_match_image.setChecked(True)

        self.chk_mix_mask_preview = QCheckBox("Mix image & mask in preview")
        self.chk_mix_mask_preview.setChecked(False)

        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cuda", "cpu"])

        self.chk_cpu_offload = QCheckBox("CPU offload (if CUDA)")
        self.chk_attention_slicing = QCheckBox("Attention slicing")
        self.chk_vae_tiling = QCheckBox("VAE tiling")
        self.chk_attention_slicing.setChecked(True)
        self.chk_vae_tiling.setChecked(False)

        self.cmb_sched = QComboBox()
        self.cmb_sched.addItem("Euler", "euler")
        self.cmb_sched.addItem("Euler a", "euler_a")
        self.cmb_sched.addItem("DPM++ 2M", "dpmpp_2m")
        self.cmb_sched.addItem("DDIM", "ddim")
        self.cmb_sched.addItem("Heun", "heun")

        self.btn_run = QPushButton("Run Inpaint")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)

        self.btn_save_as = QPushButton("Save as…")
        self.btn_open_folder = QPushButton("Open folder")
        self.btn_save_as.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

        # Thread bits
        self._thread: Optional[QThread] = None
        self._worker: Optional[InpaintWorker] = None

        # --- Layout
        left = QWidget()
        lf = QVBoxLayout(left)
        lf.setContentsMargins(0, 0, 0, 0)

        gb_model = QGroupBox("Model")
        ml = QGridLayout(gb_model)
        ml.addWidget(QLabel("Inpaint model:"), 0, 0)
        ml.addWidget(self.cmb_model, 0, 1)
        ml.addWidget(self.btn_refresh_models, 0, 2)
        ml.addWidget(self.lbl_model_path, 1, 0, 1, 3)
        lf.addWidget(gb_model)

        gb_io = QGroupBox("Inputs")
        io = QGridLayout(gb_io)
        io.addWidget(QLabel("Image:"), 0, 0)
        io.addWidget(self.ed_image, 0, 1)
        io.addWidget(self.btn_browse_image, 0, 2)

        io.addWidget(QLabel("Mask:"), 1, 0)
        io.addWidget(self.ed_mask, 1, 1)
        io.addWidget(self.btn_browse_mask, 1, 2)

        row_mask_btns = QHBoxLayout()
        row_mask_btns.addWidget(self.btn_paint_mask)
        row_mask_btns.addWidget(self.btn_clear_paint)
        row_mask_btns.addWidget(self.btn_invert_mask)
        row_mask_btns.addStretch(1)
        io.addLayout(row_mask_btns, 2, 1, 1, 2)

        io.addWidget(self.chk_match_image, 3, 1, 1, 2)

        io.addWidget(self.chk_mix_mask_preview, 4, 1, 1, 2)

        io.addWidget(QLabel("Effective mask:"), 5, 0)
        io.addWidget(self.mask_preview, 5, 1, 1, 2)

        lf.addWidget(gb_io)

        gb_text = QGroupBox("Prompt")
        tl = QVBoxLayout(gb_text)
        tl.addWidget(QLabel("Prompt"))
        tl.addWidget(self.txt_prompt, 1)
        tl.addWidget(QLabel("Negative prompt (optional)"))
        tl.addWidget(self.txt_negative, 1)
        lf.addWidget(gb_text, 2)

        gb_params = QGroupBox("Parameters")
        pl = QFormLayout(gb_params)
        pl.addRow("Steps", self.sp_steps)
        pl.addRow("CFG", self.sp_cfg)
        pl.addRow("Strength", self.sp_strength)
        pl.addRow("Seed (-1=random)", self.sp_seed)

        size_row = QHBoxLayout()
        size_row.addWidget(self.sp_w)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.sp_h)
        size_row.addStretch(1)
        pl.addRow("Size", size_row)

        pl.addRow("Scheduler", self.cmb_sched)
        pl.addRow("Device", self.cmb_device)
        pl.addRow("", self.chk_cpu_offload)
        pl.addRow("", self.chk_attention_slicing)
        pl.addRow("", self.chk_vae_tiling)
        lf.addWidget(gb_params)

        # Put the left (controls) inside its own scroll area.


        # This avoids the "dead space" issue where a global QSS breaks QScrollBar geometry,


        # and it also keeps the footer stable while you scroll controls.


        left_scroll = QScrollArea()


        left_scroll.setWidgetResizable(True)

        # Prevent the inner scroll area from advertising its full content height as a sizeHint.
        # This stops the host (Tools tab) scroll container from scrolling the *entire* UI (including tabs).
        try:
            left_scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        except Exception:
            pass
        try:
            left_scroll.setMinimumHeight(0)
        except Exception:
            pass



        left_scroll.setFrameShape(QFrame.NoFrame)


        left_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)



        # Clamp scrollbar width/margins to avoid giant empty strips if a theme QSS goes wrong.


        try:


            vsb = left_scroll.verticalScrollBar()


            vsb.setFixedWidth(14)


        except Exception:


            pass


        try:


            left_scroll.setStyleSheet(


                "QScrollBar:vertical{width:14px;margin:0px;}"


                "QScrollBar:horizontal{height:14px;margin:0px;}"


            )


        except Exception:


            pass



        left_scroll.setWidget(left)



        layout = QVBoxLayout(self)


        layout.setContentsMargins(0, 0, 0, 0)






        layout.addWidget(left_scroll, 1)
        # --- Footer (action bar) — pinned at the bottom (does not scroll)
        footer = QWidget(self)
        footer.setObjectName("sdxl_inpaint_footer")
        footer_v = QVBoxLayout(footer)
        footer_v.setContentsMargins(8, 8, 8, 8)
        footer_v.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        footer_v.addWidget(sep)

        footer_v.addWidget(self.progress)
        footer_v.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save_as)
        btn_row.addWidget(self.btn_open_folder)
        footer_v.addLayout(btn_row)

        layout.addWidget(footer, 0)


        # Slightly larger UI font (requested)

        try:
            f = self.font()
            if f.pixelSize() and f.pixelSize() > 0:
                f.setPixelSize(f.pixelSize() + 3)
            elif f.pointSize() and f.pointSize() > 0:
                # ~3px-ish bump on typical DPI
                f.setPointSize(f.pointSize() + 2)
            self.setFont(f)
        except Exception:
            pass


        # --- Signals
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.btn_browse_image.clicked.connect(self._browse_image)
        self.btn_browse_mask.clicked.connect(self._browse_mask)
        self.btn_paint_mask.clicked.connect(self._paint_mask)
        self.btn_clear_paint.clicked.connect(self._clear_painted_mask)
        self.btn_invert_mask.toggled.connect(self._update_mask_preview)
        self.ed_mask.textChanged.connect(self._update_mask_preview)
        self.ed_mask.textChanged.connect(self._on_mask_path_changed)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_save_as.clicked.connect(self._save_as)
        self.btn_open_folder.clicked.connect(self._open_folder)

        self.ed_image.textChanged.connect(self._maybe_autosize)
        self.ed_image.textChanged.connect(self._on_image_path_changed)
        self.sp_w.valueChanged.connect(self._update_mask_preview)
        self.sp_h.valueChanged.connect(self._update_mask_preview)
        self.chk_match_image.toggled.connect(self._update_mask_preview)
        self.chk_mix_mask_preview.toggled.connect(self._update_mask_preview)

        # Settings persistence hooks
        self.cmb_model.currentIndexChanged.connect(self._schedule_save)
        self.ed_image.textChanged.connect(self._schedule_save)
        self.ed_mask.textChanged.connect(self._schedule_save)
        self.txt_prompt.textChanged.connect(self._schedule_save)
        self.txt_negative.textChanged.connect(self._schedule_save)

        self.sp_steps.valueChanged.connect(self._schedule_save)
        self.sp_cfg.valueChanged.connect(self._schedule_save)
        self.sp_strength.valueChanged.connect(self._schedule_save)
        self.sp_seed.valueChanged.connect(self._schedule_save)
        self.sp_w.valueChanged.connect(self._schedule_save)
        self.sp_h.valueChanged.connect(self._schedule_save)

        self.chk_match_image.toggled.connect(self._schedule_save)
        self.chk_mix_mask_preview.toggled.connect(self._schedule_save)
        self.cmb_device.currentIndexChanged.connect(self._schedule_save)
        self.chk_cpu_offload.toggled.connect(self._schedule_save)
        self.chk_attention_slicing.toggled.connect(self._schedule_save)
        self.chk_vae_tiling.toggled.connect(self._schedule_save)
        self.cmb_sched.currentIndexChanged.connect(self._schedule_save)
        self.btn_invert_mask.toggled.connect(self._schedule_save)

        # Init
        self.refresh_models()
        self._update_mask_preview()

        # Helpful path hints in status
        py = sdxl_inpaint_python_exe() or venv_python_exe()
        if sdxl_inpaint_python_exe():
            self.lbl_status.setText(f"Ready. SDXL-Inpaint env: {sdxl_inpaint_python_exe()}")
        elif py:
            self.lbl_status.setText(f"Ready. Detected venv python: {py}")
        else:
            self.lbl_status.setText("Ready. (No venv found; running in current interpreter.)")


        # Seed settings on first run so odd external Qt state restores can't inject weird defaults.
        if not self._loaded_settings:
            try:
                self._save_settings_now()
                self._loaded_settings = self._load_settings_file()
            except Exception:
                pass

        # Apply persisted settings AFTER the UI is fully constructed.
        QTimer.singleShot(0, self._apply_loaded_settings)

    # UI events
    # -------------------------

    @Slot()
    # -------------------------
    # Host scroll isolation
    # -------------------------
    def _nearest_outer_scroll_area(self) -> Optional[QScrollArea]:
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QScrollArea):
                return w
            w = w.parentWidget()
        return None

    def _outer_scroll_enable_passthrough(self, enable: bool):
        """
        This pane contains its own internal scroll. If the host wraps the entire tab stack in a QScrollArea
        (widgetResizable=False), the whole app (including tab bars) can start scrolling when this pane is shown.
        We temporarily force that nearest host scroll area to behave like a normal container, and restore on hide.
        """
        scroll = self._nearest_outer_scroll_area()
        if scroll is None:
            return

        if enable:
            # Snapshot once per-show so we can restore on hide.
            if getattr(self, '_outer_scroll_prev', None) is None or self._outer_scroll_prev[0] is not scroll:
                try:
                    self._outer_scroll_prev = (
                        scroll,
                        bool(scroll.widgetResizable()),
                        scroll.verticalScrollBarPolicy(),
                        scroll.horizontalScrollBarPolicy(),
                        scroll.frameShape(),
                    )
                except Exception:
                    self._outer_scroll_prev = (scroll, False, Qt.ScrollBarPolicy.ScrollBarAsNeeded, Qt.ScrollBarPolicy.ScrollBarAsNeeded, QFrame.Shape.StyledPanel)

            try:
                scroll.setWidgetResizable(True)
            except Exception:
                pass
            # Hide host scrollbars so only our internal scroll is used.
            try:
                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except Exception:
                pass
            try:
                scroll.setFrameShape(QFrame.NoFrame)
            except Exception:
                pass
        else:
            prev = getattr(self, '_outer_scroll_prev', None)
            if not prev:
                return
            try:
                prev_scroll, widget_resizable, vpol, hpol, frame = prev
                if prev_scroll is scroll:
                    scroll.setWidgetResizable(widget_resizable)
                    scroll.setVerticalScrollBarPolicy(vpol)
                    scroll.setHorizontalScrollBarPolicy(hpol)
                    scroll.setFrameShape(frame)
            except Exception:
                pass

    def showEvent(self, event):
        try:
            super().showEvent(event)
        except Exception:
            pass
        # Defer until parented + laid out.
        try:
            QTimer.singleShot(0, lambda: self._outer_scroll_enable_passthrough(True))
        except Exception:
            pass

    def hideEvent(self, event):
        try:
            self._outer_scroll_enable_passthrough(False)
        except Exception:
            pass
        try:
            super().hideEvent(event)
        except Exception:
            pass

    def refresh_models(self):
        self._model_items = scan_inpaint_models()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        for name, p in self._model_items:
            self.cmb_model.addItem(name, str(p))
        self.cmb_model.blockSignals(False)

        if self._model_items:
            self.cmb_model.setCurrentIndex(0)
            self._on_model_changed()
        else:
            self.lbl_model_path.setText(f"No models found in: {models_inpaint_dir()}")
            self.lbl_model_path.setToolTip(str(models_inpaint_dir()))

    @Slot()
    def _on_model_changed(self):
        p = self.current_model_path()
        if p:
            self.lbl_model_path.setText(str(p))
            self.lbl_model_path.setToolTip(str(p))

    def current_model_path(self) -> Optional[Path]:
        data = self.cmb_model.currentData()
        if not data:
            return None
        try:
            return Path(str(data))
        except Exception:
            return None

    @Slot()
    def _browse_image(self):
        start_dir = getattr(self, "_last_image_dir", str(project_root()))
        try:
            p = Path(str(start_dir))
            if not p.exists() or not p.is_dir():
                start_dir = str(project_root())
        except Exception:
            start_dir = str(project_root())

        fn, _ = QFileDialog.getOpenFileName(
            self, "Select image", start_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        )
        if fn:
            self.ed_image.setText(fn)
            try:
                self._last_image_dir = str(Path(fn).parent)
            except Exception:
                pass
            self._schedule_save()


    @Slot(str)
    def _on_image_path_changed(self, _txt: str):
        # Clear painted mask when the source image changes (prevents stale masks in embedded contexts)
        try:
            cur = self.ed_image.text().strip()
            cur_norm = str(Path(cur).resolve()) if cur else ""
        except Exception:
            cur_norm = self.ed_image.text().strip()

        prev = getattr(self, "_last_image_path_text", "")
        if cur_norm:
            self._last_image_path_text = cur_norm

        if self._mask_qimage is None:
            return

        src = getattr(self, "_painted_mask_source_image", "") or ""
        # If we know what image the painted mask was for, only clear when it differs
        should_clear = False
        if src and cur_norm and (cur_norm != src):
            should_clear = True
        elif not src and cur_norm and prev and (cur_norm != prev):
            should_clear = True
        elif not cur_norm:
            # Image cleared
            should_clear = True

        if should_clear:
            self._mask_qimage = None
            self._painted_mask_source_image = ""
            self.lbl_status.setText("Image changed. Painted mask cleared to avoid stale masking.")
            try:
                self._update_mask_preview()
            except Exception:
                pass
            self._schedule_save()

    @Slot(str)
    def _on_mask_path_changed(self, txt: str):
        # If a mask file path is set (including programmatic setText), prefer it over any painted mask.
        if not txt or not txt.strip():
            return
        if self._mask_qimage is None:
            return
        self._mask_qimage = None
        self._painted_mask_source_image = ""
        self.lbl_status.setText("Mask file selected. Painted mask cleared.")
        try:
            self._update_mask_preview()
        except Exception:
            pass
        self._schedule_save()


    @Slot()
    def _browse_mask(self):
        start_dir = getattr(self, "_last_mask_dir", str(project_root()))
        try:
            p = Path(str(start_dir))
            if not p.exists() or not p.is_dir():
                start_dir = str(project_root())
        except Exception:
            start_dir = str(project_root())

        fn, _ = QFileDialog.getOpenFileName(
            self, "Select mask (optional)", start_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        )
        if fn:
            self.ed_mask.setText(fn)
            try:
                self._last_mask_dir = str(Path(fn).parent)
            except Exception:
                pass
            self._mask_qimage = None  # prefer explicit mask file if set
            self._painted_mask_source_image = ""
            self.lbl_status.setText("Loaded mask file. (Painted mask cleared.)")
            self._schedule_save()


    @Slot()
    def _paint_mask(self):
        img_path = Path(self.ed_image.text().strip())
        if not img_path.exists():
            QMessageBox.warning(self, "Paint mask", "Load an image first.")
            return

        try:
            from PIL import Image  # type: ignore
            img = Image.open(img_path).convert("RGB")
            if self.chk_match_image.isChecked():
                self.sp_w.setValue(img.size[0])
                self.sp_h.setValue(img.size[1])
            img = img.resize((int(self.sp_w.value()), int(self.sp_h.value())))
            qimg = qimage_from_pil(img)
        except Exception as e:
            QMessageBox.critical(self, "Paint mask", f"Failed to load image:\n{e}")
            return

        dlg = MaskPainterDialog(self)
        dlg.set_image(qimg)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._mask_qimage = dlg.get_mask()
            try:
                self._painted_mask_source_image = str(Path(self.ed_image.text().strip()).resolve())
            except Exception:
                self._painted_mask_source_image = self.ed_image.text().strip()
            self.ed_mask.setText("")  # painted mask takes precedence
            self.lbl_status.setText("Painted mask set. (Mask file path cleared.)")
            self._update_mask_preview()
            self._schedule_save()

    @Slot()
    def _clear_painted_mask(self):
        self._mask_qimage = None
        self._painted_mask_source_image = ""
        self.lbl_status.setText("Painted mask cleared.")
        self._update_mask_preview()
        self._schedule_save()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep mask preview scaled nicely on resize
        try:
            self._update_mask_preview()
        except Exception:
            pass

    @Slot()
    def _update_mask_preview(self):
        """Update the mask preview to reflect current source + invert toggle."""
        try:
            from PIL import Image, ImageOps  # type: ignore
        except Exception:
            # Pillow missing — just skip preview.
            return

        w = int(self.sp_w.value())
        h = int(self.sp_h.value())

        mask_pil = None
        has_mask = False

        # Prefer painted mask, otherwise use mask file
        if self._mask_qimage is not None:
            try:
                mask_pil = pil_from_qimage(self._mask_qimage).convert("L")
                has_mask = True
            except Exception:
                mask_pil = None

        if mask_pil is None:
            ptxt = self.ed_mask.text().strip()
            if ptxt:
                p = Path(ptxt)
                if p.exists():
                    try:
                        mask_pil = Image.open(p).convert("L")
                        has_mask = True
                    except Exception:
                        mask_pil = None

        if mask_pil is None:
            # No user mask loaded
            self.mask_preview.setPixmap(QPixmap())
            self.mask_preview.setText("No mask loaded (nothing will change)")
            self.btn_invert_mask.blockSignals(True)
            self.btn_invert_mask.setChecked(False)
            self.btn_invert_mask.blockSignals(False)
            self.btn_invert_mask.setEnabled(False)
            return

        mask_pil = ensure_mask_l_size(mask_pil, (w, h))

        if self.btn_invert_mask.isChecked():
            mask_pil = ImageOps.invert(mask_pil)

        # Render preview
        preview_pil = mask_pil.convert("RGB")

        # Optional: show the source image behind the mask, but keep ONLY the black mask parts visible.
        # (white parts become "transparent" and reveal the image)
        try:
            if getattr(self, "chk_mix_mask_preview", None) is not None and self.chk_mix_mask_preview.isChecked():
                ip = Path(self.ed_image.text().strip())
                if ip.exists():
                    img_pil = Image.open(ip).convert("RGB")
                    # Match preview size
                    if img_pil.size != (w, h):
                        try:
                            res_lanczos = Image.Resampling.LANCZOS  # Pillow>=9
                        except Exception:
                            res_lanczos = Image.LANCZOS  # type: ignore
                        img_pil = img_pil.resize((w, h), resample=res_lanczos)

                    # Build a binary mask where original BLACK pixels become opaque (255)
                    mask_black = mask_pil.point(lambda v: 255 if v < 128 else 0)
                    black = Image.new("RGB", (w, h), 0)
                    preview_pil = Image.composite(black, img_pil, mask_black)
        except Exception:
            # Fall back to normal grayscale preview
            preview_pil = mask_pil.convert("RGB")

        qimg = qimage_from_pil(preview_pil)
        pix = QPixmap.fromImage(qimg)

        pix = pix.scaled(self.mask_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.mask_preview.setPixmap(pix)
        self.mask_preview.setText("")

        # Enable invert toggle unless running
        self.btn_invert_mask.setEnabled(self._thread is None)

    @Slot()
    def _maybe_autosize(self):
        if not self.chk_match_image.isChecked():
            return
        p = Path(self.ed_image.text().strip())
        if not p.exists():
            return
        try:
            from PIL import Image  # type: ignore
            img = Image.open(p)
            w, h = img.size
            # round to multiples of 8/64? SDXL likes multiples of 64.
            def _round64(x: int) -> int:
                return max(64, int(round(x / 64.0)) * 64)
            self.sp_w.blockSignals(True)
            self.sp_h.blockSignals(True)
            self.sp_w.setValue(_round64(w))
            self.sp_h.setValue(_round64(h))
            self.sp_w.blockSignals(False)
            self.sp_h.blockSignals(False)
        except Exception:
            pass

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_refresh_models.setEnabled(not running)
        self.cmb_model.setEnabled(not running)
        self.btn_browse_image.setEnabled(not running)
        self.btn_browse_mask.setEnabled(not running)
        self.btn_paint_mask.setEnabled(not running)
        self.btn_clear_paint.setEnabled(not running)
        self.btn_invert_mask.setEnabled((not running) and (bool(self.ed_mask.text().strip()) or (self._mask_qimage is not None)))
        self.btn_save_as.setEnabled((not running) and self._last_output_path is not None)
        self.btn_open_folder.setEnabled((not running) and self._last_output_path is not None)

    @Slot()
    def _run(self):
        if self._thread is not None:
            return

        model_path = self.current_model_path()
        if model_path is None or not model_path.exists():
            QMessageBox.warning(self, "Run inpaint", "Select a valid model first.")
            return

        image_path = Path(self.ed_image.text().strip())
        if not image_path.exists():
            QMessageBox.warning(self, "Run inpaint", "Select an input image.")
            return

        mask_path = Path(self.ed_mask.text().strip()) if self.ed_mask.text().strip() else None
        if mask_path and not mask_path.exists():
            QMessageBox.warning(self, "Run inpaint", "Mask file does not exist (or clear it).")
            return

        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Run inpaint", "Enter a prompt.")
            return

        params = InpaintParams(
            model_path=model_path,
            prompt=prompt,
            negative_prompt=self.txt_negative.toPlainText().strip(),
            steps=int(self.sp_steps.value()),
            cfg=float(self.sp_cfg.value()),
            strength=float(self.sp_strength.value()),
            seed=int(self.sp_seed.value()),
            width=int(self.sp_w.value()),
            height=int(self.sp_h.value()),
            match_image=bool(self.chk_match_image.isChecked()),
            device=str(self.cmb_device.currentText()),
            cpu_offload=bool(self.chk_cpu_offload.isChecked()),
            attention_slicing=bool(self.chk_attention_slicing.isChecked()),
            vae_tiling=bool(self.chk_vae_tiling.isChecked()),
            scheduler=str(self.cmb_sched.currentData()),
            invert_mask=bool(self.btn_invert_mask.isChecked()),
        )

        self.progress.setValue(0)
        self.lbl_status.setText("Starting…")
        self._last_output_path = None
        self.btn_save_as.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

        self._thread = QThread()
        self._worker = InpaintWorker(image_path=image_path, mask_path=mask_path, mask_qimage=self._mask_qimage, params=params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._worker.finished.connect(self._cleanup_thread)
        self._worker.failed.connect(self._cleanup_thread)

        self._set_running(True)
        self._thread.start()

    @Slot()
    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_status.setText("Cancelling…")

    @Slot(object)
    def _on_finished(self, pil_image):
        if pil_image is None:
            self._on_failed("Pipeline returned no image.")
            return

        try:
            out_dir = output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = out_dir / f"inpaint_{ts}.png"
            pil_image.save(fn)
            self._last_output_path = fn
            self.lbl_status.setText(f"Saved: {fn}")
            self.btn_save_as.setEnabled(True)
            self.btn_open_folder.setEnabled(True)
        except Exception as e:
            self._on_failed(f"Failed to save output:\n{e}")

    @Slot(str)
    def _on_failed(self, err: str):
        self.progress.setValue(0)
        self.lbl_status.setText("Failed.")
        QMessageBox.critical(self, "SDXL Inpaint failed", err)

    @Slot()
    def _cleanup_thread(self):
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None

    @Slot()
    def _save_as(self):
        if self._last_output_path is None or not self._last_output_path.exists():
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save output as", str(self._last_output_path),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;WEBP (*.webp);;All files (*.*)"
        )
        if not fn:
            return
        try:
            from PIL import Image  # type: ignore
            img = Image.open(self._last_output_path)
            img.save(fn)
            self.lbl_status.setText(f"Saved copy: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Save as", str(e))

    @Slot()
    def _open_folder(self):
        p = output_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception:
            QMessageBox.information(self, "Open folder", f"Folder: {p}")


    # -------------------------
    # Settings persistence
    # -------------------------

    def _load_settings_file(self) -> Dict[str, object]:
        """
        Load persisted UI state from:
          <project_root>/presets/setsave/sdxl_inpaint_settings.json
        """
        try:
            p = inpaint_settings_path()
            if not p.exists():
                return {}
            raw = p.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _gather_settings(self) -> Dict[str, object]:
        model_path = ""
        try:
            mp = self.current_model_path()
            model_path = str(mp) if mp else ""
        except Exception:
            model_path = ""

        return {
            "schema": 1,
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_path": model_path,

            # Remember folders separately (requested)
            "last_image_dir": str(getattr(self, "_last_image_dir", "")),
            "last_mask_dir": str(getattr(self, "_last_mask_dir", "")),

            # Optional: restore last selected files (useful + low risk)
            "image_path": self.ed_image.text().strip(),
            "mask_path": self.ed_mask.text().strip(),

            # Prompts
            "prompt": self.txt_prompt.toPlainText(),
            "negative_prompt": self.txt_negative.toPlainText(),

            # Core params
            "steps": int(self.sp_steps.value()),
            "cfg": float(self.sp_cfg.value()),
            "strength": float(self.sp_strength.value()),
            "seed": int(self.sp_seed.value()),
            "width": int(self.sp_w.value()),
            "height": int(self.sp_h.value()),
            "match_image": bool(self.chk_match_image.isChecked()),
            "mix_mask_preview": bool(self.chk_mix_mask_preview.isChecked()),
            "invert_mask": bool(self.btn_invert_mask.isChecked()),

            # Runtime options
            "device": str(self.cmb_device.currentText()),
            "cpu_offload": bool(self.chk_cpu_offload.isChecked()),
            "attention_slicing": bool(self.chk_attention_slicing.isChecked()),
            "vae_tiling": bool(self.chk_vae_tiling.isChecked()),
            "scheduler": str(self.cmb_sched.currentData()),
        }

    @Slot()
    def _schedule_save(self, *args, **kwargs):
        # Debounced saves to avoid writing on every keystroke.
        if getattr(self, "_loading_settings", False):
            return
        try:
            # Ensure the requested directory exists.
            setsave_dir().mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if hasattr(self, "_save_timer") and self._save_timer is not None:
            self._save_timer.start(400)

    @Slot()
    def _save_settings_now(self):
        if getattr(self, "_loading_settings", False):
            return
        try:
            setsave_dir().mkdir(parents=True, exist_ok=True)
            path = inpaint_settings_path()

            data = self._gather_settings()

            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                tmp.replace(path)
            except Exception:
                # Fallback if replace fails (e.g., Windows perms)
                path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
                try:
                    tmp.unlink(missing_ok=True)  # type: ignore
                except Exception:
                    pass
        except Exception:
            # Silent failure: never block UI on settings writes.
            pass

    def _apply_loaded_settings(self):
        """
        Apply settings loaded from disk, clamped to widget ranges.
        This is run via QTimer.singleShot(0, ...) so it wins over any external Qt state restore.
        """
        s = getattr(self, "_loaded_settings", None)
        if not isinstance(s, dict) or not s:
            return

        self._loading_settings = True
        try:
            # Folders
            try:
                lid = str(s.get("last_image_dir") or "")
                lmd = str(s.get("last_mask_dir") or "")
                if lid:
                    self._last_image_dir = lid
                if lmd:
                    self._last_mask_dir = lmd
            except Exception:
                pass

            # Model selection by path
            want_model = str(s.get("model_path") or "")
            if want_model:
                for i in range(self.cmb_model.count()):
                    try:
                        if str(self.cmb_model.itemData(i)) == want_model:
                            self.cmb_model.setCurrentIndex(i)
                            break
                    except Exception:
                        pass

            # Restore last file paths if they still exist (or keep blank).
            # IMPORTANT: only apply when the fields are empty so embedded/host code can prefill image/mask.
            imgp = str(s.get("image_path") or "").strip()
            if (not self.ed_image.text().strip()) and imgp and Path(imgp).exists():
                self.ed_image.setText(imgp)
                try:
                    self._last_image_dir = str(Path(imgp).parent)
                except Exception:
                    pass

            maskp = str(s.get("mask_path") or "").strip()
            if (not self.ed_mask.text().strip()) and maskp and Path(maskp).exists():
                self.ed_mask.setText(maskp)
                try:
                    self._last_mask_dir = str(Path(maskp).parent)
                except Exception:
                    pass
                # Ensure we don't keep an old painted mask when restoring a mask path
                self._mask_qimage = None
                self._painted_mask_source_image = ""

            # Text
            try:
                self.txt_prompt.setPlainText(str(s.get("prompt") or ""))
                self.txt_negative.setPlainText(str(s.get("negative_prompt") or ""))
            except Exception:
                pass

            # Params (clamped)
            def _set_spin(spin, key, cast=int):
                if key not in s:
                    return
                try:
                    v = cast(s.get(key))
                    v = max(spin.minimum(), min(spin.maximum(), v))
                    spin.setValue(v)
                except Exception:
                    pass

            def _set_dspin(spin, key):
                if key not in s:
                    return
                try:
                    v = float(s.get(key))
                    v = max(spin.minimum(), min(spin.maximum(), v))
                    spin.setValue(v)
                except Exception:
                    pass

            _set_spin(self.sp_steps, "steps", int)
            _set_dspin(self.sp_cfg, "cfg")
            _set_dspin(self.sp_strength, "strength")
            _set_spin(self.sp_seed, "seed", int)
            _set_spin(self.sp_w, "width", int)
            _set_spin(self.sp_h, "height", int)

            try:
                self.chk_match_image.setChecked(bool(s.get("match_image", True)))
            except Exception:
                pass

            try:
                self.chk_mix_mask_preview.setChecked(bool(s.get("mix_mask_preview", False)))
            except Exception:
                pass

            try:
                self.btn_invert_mask.setChecked(bool(s.get("invert_mask", False)))
            except Exception:
                pass

            # Runtime
            try:
                dev = str(s.get("device") or "")
                if dev in ("auto", "cuda", "cpu"):
                    self.cmb_device.setCurrentText(dev)
            except Exception:
                pass

            try:
                self.chk_cpu_offload.setChecked(bool(s.get("cpu_offload", False)))
                self.chk_attention_slicing.setChecked(bool(s.get("attention_slicing", True)))
                self.chk_vae_tiling.setChecked(bool(s.get("vae_tiling", False)))
            except Exception:
                pass

            try:
                sched_key = str(s.get("scheduler") or "")
                if sched_key:
                    for i in range(self.cmb_sched.count()):
                        if str(self.cmb_sched.itemData(i)) == sched_key:
                            self.cmb_sched.setCurrentIndex(i)
                            break
            except Exception:
                pass

        finally:
            self._loading_settings = False
            try:
                self._update_mask_preview()
            except Exception:
                pass

    def closeEvent(self, event):
        try:
            self._save_settings_now()
        except Exception:
            pass
        super().closeEvent(event)



# -------------------------
# Optional combined pane (Remove Background + Inpaint) for Tools tab
# -------------------------

def _try_make_remove_background_widget(parent: Optional[QWidget] = None) -> QWidget:
    """Best-effort loader for the Background Remover pane from common module/class names.
    If not found, returns a placeholder widget explaining what went wrong.
    """
    candidates = [
        (".background_remover", ["BackgroundRemoverPane", "BackgroundRemover", "RemoveBackgroundPane", "RemoveBgPane", "BgRemoverPane"]),
        (".bg_remove",          ["BackgroundRemoverPane", "RemoveBackgroundPane", "RemoveBgPane", "BgRemovePane"]),
        (".remove_bg",          ["RemoveBackgroundPane", "RemoveBgPane", "BackgroundRemoverPane"]),
        (".background_remove",  ["RemoveBackgroundPane", "BackgroundRemoverPane"]),
        (".bgremover",          ["BgRemoverPane", "BackgroundRemoverPane"]),
    ]

    errors: List[str] = []
    for mod_name, class_names in candidates:
        try:
            mod = importlib.import_module(mod_name, package=__package__)
        except Exception as e:
            errors.append(f"{mod_name}: {e}")
            continue

        for cls_name in class_names:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            try:
                return cls(parent)  # typical signature
            except TypeError:
                try:
                    w = cls()
                    if parent is not None and isinstance(w, QWidget) and w.parent() is None:
                        w.setParent(parent)
                    return w
                except Exception as e:
                    errors.append(f"{mod_name}.{cls_name}: {e}")
            except Exception as e:
                errors.append(f"{mod_name}.{cls_name}: {e}")

    # Placeholder
    w = QWidget(parent)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 12, 12, 12)
    title = QLabel("Remove background tool not found")
    title.setStyleSheet("font-weight: 600;")
    lay.addWidget(title)
    lay.addWidget(QLabel("FrameVision expected a background remover pane, but this module couldn't import it."))
    if errors:
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText("\n".join(errors[-12:]))
        box.setMinimumHeight(140)
        lay.addWidget(box)
    lay.addStretch(1)
    return w


class BackgroundRemoverAndInpaintPane(QWidget):
    """A combined two-tab pane meant for the Tools tab:
    - Remove background
    - Inpaint (SDXL)

    If your app already provides its own wrapper, you can ignore this class and instantiate SDXLInpaintPane directly.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)

        # Tab 1: background remover (best-effort import)
        tabs.addTab(_try_make_remove_background_widget(tabs), "Remove background")

        # Tab 2: inpaint
        tabs.addTab(SDXLInpaintPane(tabs), "Inpaint")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs, 1)


# Backwards-compatible aliases (different parts of FrameVision may import different names)
ToolsPane = BackgroundRemoverAndInpaintPane
BackgroundRemoverPane = BackgroundRemoverAndInpaintPane

def build_pane(parent: Optional[QWidget] = None) -> QWidget:
    return BackgroundRemoverAndInpaintPane(parent)


# -------------------------
# Standalone test runner
# -------------------------

def main():
    app = QApplication(sys.argv)
    w = QWidget()
    w.setWindowTitle("SDXL Inpaint (helpers/sdxl_inpaint_ui.py)")
    layout = QVBoxLayout(w)
    pane = SDXLInpaintPane(w)
    layout.addWidget(pane)
    w.resize(1200, 780)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()