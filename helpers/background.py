
"""
helpers/background.py — Background / Object Removal + Replacer (ONNX Runtime, offline)

Enhanced build:
- Engines: Auto / MODNet (portrait) / BiRefNet (general)
- Robust output handling (NCHW/NHWC; 1/3 channels); pad-to-32 for dynamic models; crop-back
- Engine-aware preprocessing (BiRefNet uses ImageNet mean/std)
- Post: threshold, feather, invert, auto-level; optional edge de-spill & anti-halo
- Background Replacer: Transparent / Color / Blur / Image
- Drop shadow with opacity/blur/offset
- NEW: Output folder chooser + Open button, remembered across restarts (ROOT/config/background_tool.json)
- Simplified actions: one **Process current image** button (no batch UI)
"""
from __future__ import annotations

import math, os, json, subprocess, sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageFilter

# ---- App paths fallbacks ----
try:
    from helpers.framevision_app import ROOT, OUT_SHOTS, OUT_TEMP
except Exception:
    ROOT = Path(".").resolve()
    OUT_SHOTS = ROOT / "output" / "images"
    OUT_TEMP  = ROOT / "output" / "_temp"

OUT_SHOTS.mkdir(parents=True, exist_ok=True)
OUT_TEMP.mkdir(parents=True, exist_ok=True)

# ---- Optional deps ----
try:
    import onnxruntime as ort
except Exception:
    ort = None  # handled later

try:
    import cv2
except Exception:
    cv2 = None

# -------------------- Preferences --------------------
def _prefs_path() -> Path:
    cfg = ROOT / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg / "background_tool.json"

def _load_prefs() -> dict:
    p = _prefs_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_prefs(d: dict) -> None:
    try:
        _prefs_path().write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def _get_out_dir_pref() -> Path:
    prefs = _load_prefs()
    val = prefs.get("out_dir")
    if val:
        q = Path(val)
        try:
            q.mkdir(parents=True, exist_ok=True)
        except Exception:
            return OUT_SHOTS
        return q
    return OUT_SHOTS

def _set_out_dir_pref(path: Path) -> None:
    prefs = _load_prefs()
    prefs["out_dir"] = str(path)
    _save_prefs(prefs)

# -------------------- Array & image helpers --------------------
def _np_from_pil(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))  # HWC uint8

def _pil_from_np(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr, "RGB")

def _gaussian(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return arr
    if cv2 is not None:
        k = max(1, int(2 * round(radius) + 1))
        return cv2.GaussianBlur(arr, (k, k), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_DEFAULT)
    im = _pil_from_np(arr)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=radius)))

def _pad_to_multiple(img_rgb: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """Pad H/W to nearest multiple with reflect padding. Returns (padded, (top,bottom,left,right))."""
    h, w = img_rgb.shape[:2]
    nh = int(math.ceil(h / multiple) * multiple)
    nw = int(math.ceil(w / multiple) * multiple)
    pt, pl = 0, 0
    pb, pr = nh - h, nw - w
    if pb == 0 and pr == 0:
        return img_rgb, (0, 0, 0, 0)
    if cv2 is not None:
        padded = cv2.copyMakeBorder(img_rgb, pt, pb, pl, pr, borderType=cv2.BORDER_REFLECT_101)
    else:
        padded = np.pad(img_rgb, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
    return padded, (pt, pb, pl, pr)

def _crop_from_padding(arr: np.ndarray, pads: tuple[int,int,int,int]) -> np.ndarray:
    """Crop padding back to original size."""
    pt, pb, pl, pr = pads
    h, w = arr.shape[:2]
    y0, y1 = pt, h - pb if pb > 0 else h
    x0, x1 = pl, w - pr if pr > 0 else w
    return arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, ...]

# -------------------- Model discovery --------------------
def _models_dir() -> Path:
    for p in (ROOT / "models", Path("models")):
        if p.exists():
            return p
    d = ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _find_model(names: List[str]) -> Optional[Path]:
    mdir = _models_dir()
    search_dirs = [mdir] + [mdir / x for x in ("bg", "MODNet", "modnet", "BiRefNet", "birefnet")]
    for folder in search_dirs:
        try:
            for nm in names:
                p = folder / nm
                if p.exists():
                    return p
        except Exception:
            pass
    return None

def _pick_engine(explicit: Optional[str] = None) -> tuple[str, Optional[Path]]:
    """Return (engine_id, model_path). engine_id ∈ {'modnet','birefnet'}"""
    if explicit:
        explicit = explicit.lower()
    if explicit == "modnet":
        p = _find_model(["modnet_photographic_portrait_matting.onnx", "modnet.onnx", "model.onnx"])
        return "modnet", p
    if explicit == "birefnet":
        p = _find_model(["BiRefNet-COD-epoch_125.onnx", "BiRefNet.onnx"])
        return "birefnet", p
    mod = _find_model(["modnet_photographic_portrait_matting.onnx", "modnet.onnx", "model.onnx"])
    if mod:
        return "modnet", mod
    bir = _find_model(["BiRefNet-COD-epoch_125.onnx", "BiRefNet.onnx"])
    if bir:
        return "birefnet", bir
    return "modnet", None

# -------------------- ONNX inference --------------------
def _ensure_ort() -> None:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Add onnxruntime==1.18.0 to requirements.")

def _preprocess_for_engine(x_nchw01: np.ndarray, engine_id: str) -> np.ndarray:
    """For BiRefNet, apply ImageNet normalization; for MODNet, pass-through."""
    if engine_id == "birefnet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (x_nchw01 - mean) / std
    return x_nchw01

def _auto_level_alpha(alpha01: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Contrast-stretch alpha via percentiles to expand dynamic range."""
    a = alpha01.astype(np.float32)
    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))
    if hi <= lo + 1e-6:
        return np.clip(alpha01, 0.0, 1.0)
    return np.clip((alpha01 - lo) / (hi - lo), 0.0, 1.0)

def _as_2d_alpha(arr: np.ndarray) -> np.ndarray:
    """Normalize any output to 2D float32 alpha in [0,1]."""
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)
    if a.ndim == 3:
        if a.shape[-1] in (1, 3):
            a = a[..., 0]
        elif a.shape[0] in (1, 3):
            a = a[0, ...]
        else:
            a = a.mean(axis=int(np.argmin(a.shape)))
    if a.ndim != 2:
        a = np.squeeze(a)
        if a.ndim != 2:
            a = a.reshape(a.shape[-2], a.shape[-1])
    a = a.astype(np.float32)
    if a.max() > 1.0 + 1e-5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)

def _infer_onnx_alpha(model_path: Path, img_rgb: np.ndarray, engine_id: str) -> np.ndarray:
    """Return alpha in range [0,1], shape HxW (original size)."""
    _ensure_ort()
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    shape = [s if isinstance(s, int) else -1 for s in (inp.shape or [])]
    fixed_hw: Optional[Tuple[int, int]] = None
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2] > 0 and shape[3] > 0:
        fixed_hw = (int(shape[2]), int(shape[3]))
    if fixed_hw is not None:
        x_img = img_rgb
        if (img_rgb.shape[0], img_rgb.shape[1]) != fixed_hw:
            if cv2 is not None:
                x_img = cv2.resize(img_rgb, (fixed_hw[1], fixed_hw[0]), interpolation=cv2.INTER_AREA)
            else:
                x_img = np.array(Image.fromarray(img_rgb).resize((fixed_hw[1], fixed_hw[0]), Image.BILINEAR))
        pads = (0, 0, 0, 0)
    else:
        x_img, pads = _pad_to_multiple(img_rgb, 32)
    x = x_img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    x = _preprocess_for_engine(x, engine_id)
    out = sess.run(None, {in_name: x})
    alpha = _as_2d_alpha(out[0] if len(out) == 1 else out[0])
    if alpha.shape != x_img.shape[:2]:
        if cv2 is not None:
            alpha = cv2.resize(alpha, (x_img.shape[1], x_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            alpha = np.array(
                Image.fromarray((alpha * 255).astype(np.uint8)).resize((x_img.shape[1], x_img.shape[0]), Image.BILINEAR)
            ) / 255.0
    if pads != (0, 0, 0, 0):
        alpha = _crop_from_padding(alpha, pads)
    if fixed_hw is not None and (img_rgb.shape[0], img_rgb.shape[1]) != fixed_hw:
        if cv2 is not None:
            alpha = cv2.resize(alpha, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            alpha = np.array(
                Image.fromarray((alpha * 255).astype(np.uint8)).resize((img_rgb.shape[1], img_rgb.shape[0]), Image.BILINEAR)
            ) / 255.0
    return np.clip(alpha, 0.0, 1.0)

# -------------------- Post, composite --------------------
def _apply_post(alpha: np.ndarray, thresh: int, feather: int) -> np.ndarray:
    """alpha 0..1 -> 0..255 uint8 with threshold + feather."""
    if alpha.ndim == 3 and alpha.shape[-1] in (1, 3):
        alpha = alpha[..., 0]
    a = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    if thresh > 0:
        a = (a >= int(thresh)).astype(np.uint8) * 255
    if feather > 0:
        a = _gaussian(a, float(feather))
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def _spill_suppress(rgb: np.ndarray, alpha_u8: np.ndarray, strength: float = 0.18) -> np.ndarray:
    """Light edge de-spill: desaturate along semi-transparent edges."""
    if strength <= 0:
        return rgb
    a = alpha_u8.astype(np.float32) / 255.0
    if a.ndim == 3 and a.shape[-1] in (1, 3):
        a = a[..., 0]
    if cv2 is not None:
        g = np.abs(cv2.Laplacian(a, cv2.CV_32F))
    else:
        g = np.abs(a - _gaussian(a, 1.5))
    edge = np.clip(g, 0, 1.0)
    if edge.ndim == 3:
        edge = edge[..., 0]
    edge = (edge > 0.02).astype(np.float32) * (edge / (edge.max() + 1e-6))
    rgbf = rgb.astype(np.float32)
    gray = rgbf.mean(axis=2, keepdims=True)
    e = edge[..., None]
    out = rgbf * (1.0 - strength * e) + gray * (1.0 * strength * e)
    return np.clip(out, 0, 255).astype(np.uint8)

def _edge_antihalo(rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Simple color decontamination around edges using background estimate from low-alpha pixels."""
    a = alpha_u8.astype(np.float32) / 255.0
    edge = (a > 0.02) & (a < 0.98)
    bg_mask = a < 0.05
    if not np.any(edge):
        return rgb
    if np.any(bg_mask):
        bg_est = rgb[bg_mask].astype(np.float32).mean(axis=0)
    else:
        bg_est = np.array([0, 0, 0], dtype=np.float32)
    out = rgb.astype(np.float32).copy()
    denom = (a[..., None] + 1e-6)
    corr = (out - (1.0 - a[..., None]) * bg_est) / denom
    out[edge] = corr[edge]
    return np.clip(out, 0, 255).astype(np.uint8)

def _resize_cover(img: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """Resize image to cover target size (center crop)."""
    th, tw = target_hw
    h, w = img.shape[:2]
    scale = max(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if cv2 is not None:
        res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        res = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    y0 = max(0, (nh - th) // 2); x0 = max(0, (nw - tw) // 2)
    return res[y0:y0+th, x0:x0+tw]

def _compose_with_bg(
    img_rgb: np.ndarray,
    a8: np.ndarray,
    mode: str,
    repl_mode: str,
    color_rgb: Tuple[int,int,int],
    blur_rad: int,
    bg_img_path: Optional[str],
    drop_shadow: bool,
    shadow_alpha: int,
    shadow_blur: int,
    shadow_offx: int,
    shadow_offy: int,
    anti_halo: bool,
) -> Image.Image:
    """Compose according to replacement settings. Returns PIL Image (RGBA when transparent)."""
    h, w = img_rgb.shape[:2]
    a = a8.astype(np.uint8)

    fg = img_rgb
    if anti_halo and mode != "alpha_only":
        fg = _edge_antihalo(fg, a)

    if repl_mode == "transparent":
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        out_alpha = np.zeros((h, w), dtype=np.uint8)
    elif repl_mode == "color":
        r,g,b = color_rgb
        bg = np.zeros((h, w, 3), dtype=np.uint8) + np.array([r,g,b], dtype=np.uint8)[None,None,:]
        out_alpha = np.full((h, w), 255, dtype=np.uint8)
    elif repl_mode == "blur":
        bg = _gaussian(img_rgb, float(max(0, blur_rad)))
        out_alpha = np.full((h, w), 255, dtype=np.uint8)
    else:
        try:
            im = Image.open(bg_img_path).convert("RGB")
            bg_np = np.asarray(im)
        except Exception:
            bg_np = np.zeros((h, w, 3), dtype=np.uint8)
        bg = _resize_cover(bg_np, (h, w))
        out_alpha = np.full((h, w), 255, dtype=np.uint8)

    if drop_shadow and mode != "alpha_only":
        s = a.astype(np.float32) / 255.0
        if shadow_blur > 0:
            s = _gaussian(s, float(shadow_blur))
            if s.ndim == 3: s = s[...,0]
        M = np.float32([[1,0,shadow_offx],[0,1,shadow_offy]])
        if cv2 is not None:
            s = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        else:
            s2 = np.zeros_like(s)
            ox, oy = int(shadow_offx), int(shadow_offy)
            y0 = max(0, oy); x0 = max(0, ox)
            y1 = min(h, h+oy); x1 = min(w, w+ox)
            s2[y0:y1, x0:x1] = s[0:h-abs(oy), 0:w-abs(ox)]
            s = s2
        s = np.clip(s * (shadow_alpha/255.0), 0.0, 1.0)
        bg = (bg.astype(np.float32) * (1.0 - s[...,None])).astype(np.uint8)
        if repl_mode == "transparent":
            out_alpha = np.maximum(out_alpha, (s*255).astype(np.uint8))

    if mode == "alpha_only":
        return _pil_from_np(a)

    if mode == "keep_bg":
        inv = 255 - a
        rgb = (img_rgb.astype(np.float32) * (inv/255.0)[...,None]).astype(np.uint8)
        return _pil_from_np(rgb)

    alpha01 = a.astype(np.float32) / 255.0
    comp = (fg.astype(np.float32) * alpha01[...,None]) + (bg.astype(np.float32) * (1.0 - alpha01[...,None]))
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    if repl_mode == "transparent":
        out_alpha = np.maximum(out_alpha, a)
        rgba = np.dstack([comp, out_alpha])
        return _pil_from_np(rgba)
    else:
        return _pil_from_np(comp)

# -------------------- Public processing API --------------------
def remove_background_file(
    inp_path: str,
    engine: str = "auto",
    mode: str = "keep_subject",
    threshold: int = 0,
    feather: int = 3,
    spill: bool = True,
    invert: bool = False,
    auto_level: bool = True,
    out_dir: Optional[str] = None,
    # Replacer:
    repl_mode: str = "transparent",
    repl_color: Tuple[int,int,int] = (255,255,255),
    repl_blur: int = 20,
    repl_image: Optional[str] = None,
    drop_shadow: bool = False,
    shadow_alpha: int = 120,
    shadow_blur: int = 20,
    shadow_offx: int = 10,
    shadow_offy: int = 10,
    anti_halo: bool = True,
) -> Path:
    """Process one image file and return output path."""
    p = Path(inp_path)
    out_dir_p = Path(out_dir) if out_dir else _get_out_dir_pref()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    eng, model = _pick_engine(None if engine == "auto" else engine)
    if model is None:
        raise FileNotFoundError("No ONNX model found in models/. Expected MODNet or BiRefNet.")

    img = Image.open(p)
    rgb = _np_from_pil(img)
    alpha = _infer_onnx_alpha(model, rgb, eng)
    if auto_level:
        alpha = _auto_level_alpha(alpha)
    if invert:
        alpha = 1.0 - alpha
    a8 = _apply_post(alpha, int(threshold), int(feather))
    if spill and mode != "alpha_only":
        rgb = _spill_suppress(rgb, a8, 0.18)

    out_img = _compose_with_bg(
        rgb, a8, mode,
        repl_mode, repl_color, int(repl_blur), repl_image,
        bool(drop_shadow), int(shadow_alpha), int(shadow_blur), int(shadow_offx), int(shadow_offy),
        bool(anti_halo),
    )

    suffix = {
        "transparent": "cutout",
        "color": "colorbg",
        "blur": "blurbg",
        "image": "imagebg",
    }.get(repl_mode, "cutout")
    if mode == "alpha_only":
        suffix = "alpha"
    elif mode == "keep_bg":
        suffix = "bgonly"

    out = out_dir_p / f"{p.stem}_{suffix}.png"
    out_img.save(out)
    return out

# -------------------- Qt UI wiring --------------------
def install_background_tool(pane, section_widget) -> None:
    """
    Build the UI under the provided CollapsibleSection.
    """
    try:
        from PySide6.QtWidgets import (
            QWidget, QFormLayout, QComboBox, QSpinBox, QHBoxLayout, QLineEdit,
            QPushButton, QCheckBox, QFileDialog, QMessageBox, QLabel
        )
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog
    except Exception:
        return

    w = QWidget()
    lay = QFormLayout(w)

    cmb_engine = QComboBox(); cmb_engine.addItems(["Auto", "MODNet (portrait)", "BiRefNet (general)"])
    cmb_mode   = QComboBox(); cmb_mode.addItems(["Keep subject (remove BG)", "Keep background (remove subject)", "Alpha only (mask)"])

    # Replacer UI
    cmb_repl   = QComboBox(); cmb_repl.addItems(["Transparent", "Color", "Blur", "Image"])
    s_blurbg   = QSpinBox();  s_blurbg.setRange(0, 200); s_blurbg.setValue(15)

    # Color picker
    color_preview = QLabel("●")
    color_preview.setStyleSheet("font-size:18px;")
    cho_color = QColor(255,255,255)
    btn_color = QPushButton("Pick color…")
    def pick_color():
        nonlocal cho_color
        try:
            c = QColorDialog.getColor(cho_color, w, "Background color")
        except Exception:
            c = cho_color
        if c.isValid():
            cho_color = c
            color_preview.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()}); font-size:18px;")
    btn_color.clicked.connect(pick_color)

    # Image picker
    lbl_img = QLabel("(none)")
    btn_img = QPushButton("Choose image…")
    bg_img_path = {"path": None}
    def pick_img():
        try:
            file, _ = QFileDialog.getOpenFileName(w, "Choose background image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        except Exception:
            file = ""
        if file:
            bg_img_path["path"] = file
            lbl_img.setText(Path(file).name)
    btn_img.clicked.connect(pick_img)

    # Post options
    s_thresh   = QSpinBox();  s_thresh.setRange(0,255); s_thresh.setValue(6)
    s_feath    = QSpinBox();  s_feath.setRange(0,50);  s_feath.setValue(3)
    cb_spill   = QCheckBox("Edge de-spill"); cb_spill.setChecked(True)
    cb_auto    = QCheckBox("Auto-level mask"); cb_auto.setChecked(True)
    cb_inv     = QCheckBox("Invert mask"); cb_inv.setChecked(False)
    cb_ah      = QCheckBox("Edge anti-halo"); cb_ah.setChecked(True)
    cb_shadow  = QCheckBox("Drop shadow"); cb_shadow.setChecked(False)
    s_salpha   = QSpinBox(); s_salpha.setRange(0,255); s_salpha.setValue(120)
    s_sblur    = QSpinBox(); s_sblur.setRange(0,200); s_sblur.setValue(20)
    s_sox      = QSpinBox(); s_sox.setRange(-500,500); s_sox.setValue(10)
    s_soy      = QSpinBox(); s_soy.setRange(-500,500); s_soy.setValue(10)

    # Output folder row with remember
    out_dir = _get_out_dir_pref()
    le_out = QLineEdit(str(out_dir)); le_out.setReadOnly(True)
    btn_out_change = QPushButton("Change…")
    btn_out_open   = QPushButton("Open folder")
    def change_out():
        try:
            d = QFileDialog.getExistingDirectory(w, "Choose output folder", str(out_dir))
        except Exception:
            d = ""
        if d:
            _set_out_dir_pref(Path(d))
            le_out.setText(d)
    btn_out_change.clicked.connect(change_out)

    def open_out():
        path = le_out.text().strip()
        if not path: return
        pth = Path(path)
        try: pth.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(pth))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.run(["open", str(pth)], check=False)
            else:
                subprocess.run(["xdg-open", str(pth)], check=False)
        except Exception:
            pass
    btn_out_open.clicked.connect(open_out)

    # Action button
    btn_run    = QPushButton("Process current image")

    # Layout
    lay.addRow("Engine", cmb_engine)
    lay.addRow("Mode", cmb_mode)
    lay.addRow("Background", cmb_repl)
    lay.addRow("Blur amount", s_blurbg)
    lay.addRow("BG color", btn_color)
    lay.addRow(" ", color_preview)
    lay.addRow("BG image", btn_img)
    lay.addRow(" ", lbl_img)
    lay.addRow("Output to", le_out)
    row_out = QHBoxLayout(); row_out.addWidget(btn_out_change); row_out.addWidget(btn_out_open)
    lay.addRow(row_out)
    lay.addRow("Hard threshold (0=off)", s_thresh)
    lay.addRow("Feather (px)", s_feath)
    lay.addRow(cb_spill); lay.addRow(cb_auto); lay.addRow(cb_inv); lay.addRow(cb_ah)
    lay.addRow(cb_shadow)
    row_shadow = QHBoxLayout()
    row_shadow.addWidget(QLabel("Opacity")); row_shadow.addWidget(s_salpha)
    row_shadow.addWidget(QLabel("Blur"));    row_shadow.addWidget(s_sblur)
    row_shadow.addWidget(QLabel("Offset X"));row_shadow.addWidget(s_sox)
    row_shadow.addWidget(QLabel("Offset Y"));row_shadow.addWidget(s_soy)
    lay.addRow(row_shadow)

    # Add the action row
    lay.addRow(btn_run)

    try:
        section_widget.setContentLayout(lay)
    except Exception:
        pass

    def _engine_key() -> str:
        i = cmb_engine.currentIndex()
        return {0: "auto", 1: "modnet", 2: "birefnet"}.get(i, "auto")

    def _mode_key() -> str:
        i = cmb_mode.currentIndex()
        return {0: "keep_subject", 1: "keep_bg", 2: "alpha_only"}.get(i, "keep_subject")

    def _repl_key() -> str:
        i = cmb_repl.currentIndex()
        return {0:"transparent",1:"color",2:"blur",3:"image"}[i]

    def _ensure_image_input() -> Optional[Path]:
        pth = getattr(pane.main, "current_path", None)
        if pth and pth.exists() and pth.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}:
            return pth
        try:
            files, _ = QFileDialog.getOpenFileNames(w, "Pick image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        except Exception:
            files = []
        if files: return Path(files[0])
        return None

    def on_run():
        try:
            pth = _ensure_image_input()
            if not pth:
                try: QMessageBox.warning(w, "No image", "Open an image first or pick one.")
                except Exception: pass
                return
            out = remove_background_file(
                str(pth),
                engine=_engine_key(),
                mode=_mode_key(),
                threshold=int(s_thresh.value()),
                feather=int(s_feath.value()),
                spill=bool(cb_spill.isChecked()),
                invert=bool(cb_inv.isChecked()),
                auto_level=bool(cb_auto.isChecked()),
                out_dir=le_out.text().strip(),
                repl_mode=_repl_key(),
                repl_color=(cho_color.red(), cho_color.green(), cho_color.blue()),
                repl_blur=int(s_blurbg.value()),
                repl_image=bg_img_path["path"],
                drop_shadow=bool(cb_shadow.isChecked()),
                shadow_alpha=int(s_salpha.value()),
                shadow_blur=int(s_sblur.value()),
                shadow_offx=int(s_sox.value()),
                shadow_offy=int(s_soy.value()),
                anti_halo=bool(cb_ah.isChecked()),
            )
            try: QMessageBox.information(w, "Done", f"Saved:\n{out}")
            except Exception: print("Saved:", out)
        except Exception as e:
            try: QMessageBox.critical(w, "Background tool", str(e))
            except Exception: print("Background tool error:", e)

    btn_run.clicked.connect(on_run)
