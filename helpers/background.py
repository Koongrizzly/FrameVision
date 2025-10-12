
"""
helpers/background.py — Background / Object Removal + Replacer with TOP PREVIEW (ONNX Runtime, offline)

Changes in this build
---------------------
- Preview panel is **on top**, controls below (vertical). It grows/shrinks with the tab.
- Added **Use current** button that grabs the image (or current video frame) already loaded in the app
  via `pane.main.current_path` (same behavior you had before).
- Still supports **Load external…** to pick an image/video from disk.
- Live preview (instant recomposite), Undo/Reset/Save, Recompute mask (run ONNX again on demand).
- Output folder Change/Open (remembers across restarts).

Everything else kept: MODNet/BiRefNet, pad-to-32, ImageNet norm for BiRefNet, de-spill, anti-halo, background
replacer (Transparent/Color/Blur/Image), drop shadow.
"""
from __future__ import annotations

import math, os, sys, json, subprocess, tempfile, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

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
try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
except Exception:
    torch = None
    StableDiffusionInpaintPipeline = None


# -------------------- Preferences --------------------
def _presets_store_path() -> Path:
    """Background tool presets path (single file)."""

    d = ROOT / "presets" / "setsave"
    d.mkdir(parents=True, exist_ok=True)
    return d / "background_tool.json"


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

# --- simple morphology helper for edge grow/shrink (px) on uint8 mask ---
def _morph_shift_u8(a_u8: np.ndarray, px: int) -> np.ndarray:
    try:
        k = int(px)
    except Exception:
        k = 0
    if k == 0:
        return a_u8
    k_abs = abs(k)
    # Ensure 2D + uint8
    a = a_u8
    if a.ndim == 3:
        a = a[...,0]
    a = np.clip(a, 0, 255).astype(np.uint8)
    if cv2 is not None:
        ksz = max(1, 2 * k_abs + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        if k > 0:
            a = cv2.dilate(a, kernel, iterations=1)
        else:
            a = cv2.erode(a, kernel, iterations=1)
    else:
        # PIL fallback via Min/MaxFilter, one pass per px
        from PIL import ImageFilter as _IF
        im = _pil_from_np(a)
        for _ in range(k_abs):
            im = im.filter(_IF.MaxFilter(3) if k > 0 else _IF.MinFilter(3))
        a = np.asarray(im)
    return a.astype(np.uint8)


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

# --- ONNX session cache (avoid rebuilding the model each click) ---
_ORT_SESSION_CACHE = {}

def _get_ort_session(model_path: Path):
    """Return cached onnxruntime.InferenceSession for model_path, creating it on first use.
    Provider priority: CUDA -> DirectML -> CPU with safe fallback."""
    _ensure_ort()
    try:
        avail = list(getattr(ort, "get_available_providers", lambda: [])())
    except Exception:
        avail = ["CPUExecutionProvider"]
    providers = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    elif "DmlExecutionProvider" in avail or "DMLExecutionProvider" in avail:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    key = str(Path(model_path).resolve()) + "|" + "|".join(providers)
    sess = _ORT_SESSION_CACHE.get(key)
    if sess is None:
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = getattr(ort, "GraphOptimizationLevel", None).ORT_ENABLE_ALL
        except Exception:
            pass
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        _ORT_SESSION_CACHE[key] = sess
    return sess


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
    sess = _get_ort_session(model_path)
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


# -------------------- SD Inpainting (Stable Diffusion 1.5) --------------------
_SD_PIPE_CACHE = {}

def _sd15_inpaint_available() -> bool:
    return ('torch' in globals() and torch is not None) and ('StableDiffusionInpaintPipeline' in globals() and StableDiffusionInpaintPipeline is not None)

def _default_sd15_inpaint_model_path() -> Path:
    # As provided by the user: root \models\bg\sd-v1-5-inpainting.safetensors
    return _models_dir() / "bg" / "sd-v1-5-inpainting.safetensors"


def _resolve_inpaint_model_path(user_text: Optional[str] = None) -> Path:
    # Resolve a valid SD inpaint model path with fallbacks
    import os
    # 1) From UI
    if user_text:
        s = user_text.strip().strip('"').strip("'")
        s = os.path.expanduser(os.path.expandvars(s))
        pth = Path(s)
        if pth.exists():
            return pth
    # 2) Preferred defaults under models/bg
    candidates = [
        _models_dir() / 'bg' / 'sd-v1-5-inpainting.safetensors',
        _models_dir() / 'bg' / 'sd-v1-5-inpainting.fp16.safetensors',
    ]
    # 3) Any sd*inpaint*.safetensors under models/bg
    try:
        for g in ['sd*-inpaint*.safetensors', 'sd*inpaint*.safetensors', '*inpaint*.safetensors']:
            for m in sorted((_models_dir() / 'bg').glob(g)):
                candidates.append(m)
    except Exception:
        pass
    for c in candidates:
        try:
            if Path(c).exists():
                return Path(c)
        except Exception:
            continue
    return Path(user_text) if user_text else _default_sd15_inpaint_model_path()

def _get_sd15_inpaint_pipe(model_path: Path):
    """Return a cached StableDiffusionInpaintPipeline from a single .safetensors file."""
    if not _sd15_inpaint_available():
        raise RuntimeError("diffusers/torch not installed. Please install: torch, diffusers, transformers, accelerate, safetensors.")
    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    key = str(model_path.resolve()) + "|" + device
    p = _SD_PIPE_CACHE.get(key)
    if p is None:
        p = StableDiffusionInpaintPipeline.from_single_file(
            str(model_path),
            local_files_only=True,
            safety_checker=None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        p = p.to(device)
        try:
            p.enable_attention_slicing()
        except Exception:
            pass
        _SD_PIPE_CACHE[key] = p
    return p

def inpaint_sd15(
    img_rgb: np.ndarray,
    remove_mask01: np.ndarray,
    prompt: str = "clean background, realistic",
    negative_prompt: str = "blurry, artifacts, distortion, text, watermark",
    steps: int = 30,
    guidance: float = 7.0,
    strength: float = 0.85,
    seed: int = -1,
    model_path: Optional[Path | str] = None,
) -> Image.Image:
    """Fill the masked (removed) regions using SD 1.5 inpainting."""
    from pathlib import Path as _PathAlias
    if model_path is None:
        model_path = _default_sd15_inpaint_model_path()
    model_path = _resolve_inpaint_model_path(str(model_path))
    if not _PathAlias(model_path).exists():
        raise FileNotFoundError(f"SD1.5 inpainting model not found at: {model_path}")
    h, w = img_rgb.shape[:2]
    m = np.clip(remove_mask01.astype(np.float32), 0.0, 1.0)
    m_u8 = (m * 255.0).astype(np.uint8)
    mask_pil = Image.fromarray(m_u8, mode="L").resize((w, h), Image.BILINEAR)
    img_pil = _pil_from_np(img_rgb)

    pipe = _get_sd15_inpaint_pipe(Path(model_path))
    g = None
    if isinstance(seed, int) and seed >= 0:
        g = torch.Generator(device=getattr(pipe, "device", "cpu")).manual_seed(int(seed))

    # First try: ask the pipeline to keep the original dimensions.
    try:
        out = pipe(
            prompt=prompt or "",
            negative_prompt=negative_prompt or None,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=g,
            width=int(w),
            height=int(h),
        )
        out_img = out.images[0].convert("RGB")
        if out_img.size != (w, h):
            raise RuntimeError("Pipeline returned different size; will retry with padded dimensions.")
        return out_img
    except Exception:
        pass

    # Fallback: pad to nearest multiple-of-8 (VAE factor), run, then crop back.
    padded_img, pads = _pad_to_multiple(img_rgb, 8)
    # Pad the mask consistently (single channel); reuse reflect padding via helper on a 3-channel view.
    m3 = np.repeat(m_u8[..., None], 3, axis=2)
    padded_m3, _ = _pad_to_multiple(m3, 8)
    padded_mask_u8 = padded_m3[..., 0]
    img_pil2 = Image.fromarray(padded_img)
    mask_pil2 = Image.fromarray(padded_mask_u8, mode="L")
    out2 = pipe(
        prompt=prompt or "",
        negative_prompt=negative_prompt or None,
        image=img_pil2,
        mask_image=mask_pil2,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=g,
        width=int(img_pil2.width),
        height=int(img_pil2.height),
    )
    out_img2 = out2.images[0].convert("RGB")
    out_np = _np_from_pil(out_img2)
    out_np_cropped = _crop_from_padding(out_np, pads)
    return _pil_from_np(out_np_cropped)

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

def _ensure_mask_hw(mask: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
    """Ensure a mask is HxW float32 in [0..1]. Returns None if mask is None."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    # Normalize dtype
    if m.dtype != np.float32 and m.dtype != np.float64:
        m = m.astype(np.float32)
        if m.max() > 1.0 + 1e-5:
            m = m / 255.0
    else:
        m = m.astype(np.float32)
    # Resize if needed
    if m.shape != (h, w):
        try:
            if cv2 is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                from PIL import Image as _PILImage
                m = np.array(_PILImage.fromarray((np.clip(m,0.0,1.0)*255).astype(np.uint8)).resize((w, h), _PILImage.BILINEAR)).astype(np.float32) / 255.0
        except Exception:
            from PIL import Image as _PILImage
            m = np.array(_PILImage.fromarray((np.clip(m,0.0,1.0)*255).astype(np.uint8)).resize((w, h), _PILImage.BILINEAR)).astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)

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
    # Normalize mask to uint8 HxW
    a = a8
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != (h, w):
        from PIL import Image as _PILImage
        a = np.array(_PILImage.fromarray(a.astype(np.uint8)).resize((w, h), _PILImage.BILINEAR))
    if a.dtype != np.uint8:
        try:
            vmax = float(a.max())
        except Exception:
            vmax = 255.0
        if vmax <= 1.0 + 1e-5:
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)

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
        # If a background image is provided (e.g., SD inpaint result), show it directly.
        if repl_mode == "image" and bg_img_path:
            try:
                im = Image.open(bg_img_path).convert("RGB")
                bg_np = np.asarray(im)
                bg = _resize_cover(bg_np, (h, w))
                return _pil_from_np(bg)
            except Exception:
                pass
        inv = 255 - a
        rgb = (img_rgb.astype(np.float32) * (inv/255.0)[...,None]).astype(np.uint8)
        return _pil_from_np(rgb)

    alpha01 = a.astype(np.float32) / 255.0
    comp = (fg.astype(np.float32) * alpha01[...,None]) + (bg.astype(np.float32) * (1.0 - alpha01[...,None]))
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    if repl_mode == "transparent":
        # Use straight alpha (un-multiplied RGB) so the subject doesn't look dark
        out_alpha = np.maximum(out_alpha, a)
        rgba = np.dstack([fg, out_alpha])
        return _pil_from_np(rgba)
    else:
        return _pil_from_np(comp)

# -------------------- I/O helpers --------------------
def _ffmpeg_path() -> Optional[str]:
    cand = ROOT / "presets" / "tools" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if cand.exists():
        return str(cand)
    return "ffmpeg"

def load_image_or_frame(path: str, time_s: float = 0.0) -> Image.Image:
    p = Path(path)
    if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}:
        return Image.open(p).convert("RGB")
    out_png = Path(tempfile.gettempdir()) / f"bgtool_frame_{os.getpid()}.png"
    cmd = [_ffmpeg_path(), "-y", "-ss", f"{max(0.0,time_s):.3f}", "-i", str(p), "-frames:v", "1", str(out_png)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return Image.open(out_png).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed to extract frame: {e}")

# Try to read the current media path/time from the host app
def _current_media_from_app(pane) -> tuple[Optional[Path], float]:
    # Known attribute in your app from earlier: pane.main.current_path
    candidates = [
        getattr(pane, "current_path", None),
        getattr(getattr(pane, "main", None), "current_path", None),
    ]
    for c in candidates:
        if c:
            p = Path(str(c))
            if p.exists():
                break
    else:
        return None, 0.0

    # try to find a current time (seconds) for videos
    time_keys = [
        "current_time_s", "video_time_s", "playhead_seconds", "seek_seconds",
        "current_frame_time", "frame_time_s"
    ]
    t = 0.0
    for k in time_keys:
        v = getattr(pane, k, None)
        if isinstance(v, (int, float)):
            t = float(v); break
        v = getattr(getattr(pane, "main", None), k, None)
        if isinstance(v, (int, float)):
            t = float(v); break
    return p, float(t)

# -------------------- Preview renderer --------------------
def remove_background_preview(
    rgb: np.ndarray,
    alpha_base: np.ndarray,
    mode: str,
    threshold: int,
    feather: int,
    spill: bool,
    invert: bool,
    auto_level: bool,
    bias: int,
    # Replacer:
    repl_mode: str,
    repl_color: Tuple[int,int,int],
    repl_blur: int,
    repl_image: Optional[str],
    drop_shadow: bool,
    shadow_alpha: int,
    shadow_blur: int,
    shadow_offx: int,
    shadow_offy: int,
    anti_halo: bool,
    edge_shift_px: int = 0,
) -> Image.Image:
    # Normalize alpha to 2D float [0..1] and match image size
    h_img, w_img = rgb.shape[:2]
    try:
        alpha = _as_2d_alpha(alpha_base)
    except Exception:
        alpha = np.asarray(alpha_base).astype(np.float32)
        if alpha.ndim == 3:
            alpha = alpha[...,0]
        if alpha.max() > 1.0 + 1e-5:
            alpha = alpha / 255.0
    # Merge live masks from preview canvas so brush strokes persist
    try:
        rm, km = preview.export_masks_to_image_size(w_img, h_img)
        if rm is not None:
            alpha = np.clip(np.maximum(alpha, rm.astype(np.float32)), 0.0, 1.0)
        if km is not None:
            alpha = np.clip(alpha - (km > 0.01).astype(np.float32), 0.0, 1.0)
    except Exception:
        pass
    if alpha.shape != (h_img, w_img):
        from PIL import Image as _PILImage
        alpha = np.array(
            _PILImage.fromarray((np.clip(alpha,0.0,1.0)*255).astype(np.uint8)).resize((w_img, h_img), _PILImage.BILINEAR)
        ).astype(np.float32) / 255.0
    if auto_level:
        alpha = _auto_level_alpha(alpha)
    if invert:
        alpha = 1.0 - alpha
    # Apply removal strength: map 0..100 (50 neutral) to a gamma curve on alpha for smooth control
    try:
        b = max(0, min(100, int(bias)))
        scale = (b - 50) / 50.0  # -1..1
        # gamma in [~2.8 (weaker removal)..1..~0.35 (stronger removal)]
        gamma = 2.0 ** (-1.5 * scale)
        alpha = np.clip(alpha, 0.0, 1.0) ** float(gamma)
    except Exception:
        pass
    a8 = _apply_post(alpha, int(threshold), int(feather))
    # Grow/shrink edges (post-threshold, pre-feather compose)
    try:
        if int(edge_shift_px) != 0:
            a8 = _morph_shift_u8(a8, int(edge_shift_px))
    except Exception:
        pass

    rgb2 = _spill_suppress(rgb, a8, 0.18) if (spill and mode != "alpha_only") else rgb
    return _compose_with_bg(
        rgb2, a8, mode,
        repl_mode, repl_color, int(repl_blur), repl_image,
        bool(drop_shadow), int(shadow_alpha), int(shadow_blur), int(shadow_offx), int(shadow_offy),
        bool(anti_halo),
    )

# -------------------- One-shot API (unchanged) --------------------
def remove_background_file(
    inp_path: str,
    engine: str = "auto",
    mode: str = "keep_subject",
    threshold: int = 0,
    feather: int = 3,
    spill: bool = True,
    invert: bool = False,
    auto_level: bool = True,
    bias: int = 50,
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
    p = Path(inp_path)
    out_dir_p = Path(out_dir) if out_dir else _get_out_dir_pref()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    eng, model = _pick_engine(None if engine == "auto" else engine)
    if model is None:
        raise FileNotFoundError("No ONNX model found in models/. Expected MODNet or BiRefNet.")

    img = Image.open(p).convert("RGB")
    rgb = _np_from_pil(img)
    alpha = _infer_onnx_alpha(model, rgb, eng)
    out_img = remove_background_preview(
        rgb, alpha, mode, threshold, feather, spill, invert, auto_level,
        repl_mode, repl_color, repl_blur, repl_image,
        drop_shadow, shadow_alpha, shadow_blur, shadow_offx, shadow_offy, anti_halo
    , edge_shift_px=0)
    suffix = {"transparent":"cutout","color":"colorbg","blur":"blurbg","image":"imagebg"}.get(repl_mode, "cutout")
    if mode == "alpha_only": suffix = "alpha"
    elif mode == "keep_bg": suffix = "bgonly"

    out = out_dir_p / f"{p.stem}_{suffix}.png"
    out_img.save(out)
    return out

# -------------------- Qt UI wiring (TOP Preview) --------------------
def install_background_tool(pane, section_widget) -> None:
    try:
        from PySide6.QtWidgets import (
            QWidget, QFormLayout, QComboBox, QSpinBox, QSlider, QHBoxLayout, QVBoxLayout, QDoubleSpinBox,
            QPushButton, QCheckBox, QFileDialog, QMessageBox, QLabel, QLineEdit
        , QButtonGroup)
        from PySide6.QtCore import Qt, QTimer, QSize, Signal, QThread
        from PySide6.QtGui import QPixmap, QImage, QColor
        from PySide6.QtWidgets import QColorDialog
    except Exception:
        return

    # --- Simple collapsible widget ---
    class CollapsibleBox(QWidget):
        def __init__(self, title: str):
            super().__init__()
            self._title = title
            self._btn = QPushButton(f"▾  {title}")
            self._btn.setCheckable(True)
            self._btn.setChecked(True)
            try:
                self._btn.setCursor(Qt.PointingHandCursor)
                self._btn.setStyleSheet('QPushButton { font-weight:600; text-align:left; padding:8px 10px; border-radius:10px; }')
            except Exception:
                pass
            self._body = QWidget()
            v = QVBoxLayout(self)
            v.setContentsMargins(0,0,0,0)
            v.addWidget(self._btn)
            v.addWidget(self._body)
            self._btn.toggled.connect(self._on_toggled)
        def _on_toggled(self, on: bool):
            self._body.setVisible(bool(on))
            try:
                self._btn.setText(('▾  ' if on else '▸  ') + self._title)
            except Exception:
                pass
        def setContentLayout(self, layout):
            try:
                self._body.setLayout(layout)
            except Exception:
                # Fallback: add as child layout
                self._body.layout().addLayout(layout)

    # --- Preview canvas at the TOP ---
    _ui_flags = {'auto_inpaint': False, 'freeze_unmasked': True}

    class Preview(QWidget):

        maskChanged = Signal()
        fileDropped = Signal(str)
        def setLockImageSize(self, on: bool):
            """When enabled, masks keep their size even if the source image size changes."""
            self._lock_image_size = bool(on)
            try:
                self.setFocus(Qt.OtherFocusReason)
            except Exception:
                pass

        def _clone_qimage(self, qimg):
            try:
                return qimg.copy()
            except Exception:
                return qimg

        def _push_undo_snapshot(self):
            try:
                if self._mask is None or self._mask.isNull():
                    return
                m1 = self._clone_qimage(self._mask)
                m2 = self._clone_qimage(self._mask_keep)
                self._undo_stack.append((m1, m2))
                if len(self._undo_stack) > 50:
                    self._undo_stack.pop(0)
                self._redo_stack.clear()
            except Exception:
                pass

        def undo(self):
            try:
                if not self._undo_stack:
                    return
                cur_m = self._clone_qimage(self._mask)
                cur_k = self._clone_qimage(self._mask_keep)
                m1, m2 = self._undo_stack.pop()
                self._redo_stack.append((cur_m, cur_k))
                self._mask = m1; self._mask_keep = m2
                self.maskChanged.emit(); self.update()
            except Exception:
                pass

        def redo(self):
            try:
                if not self._redo_stack:
                    return
                cur_m = self._clone_qimage(self._mask)
                cur_k = self._clone_qimage(self._mask_keep)
                m1, m2 = self._redo_stack.pop()
                self._undo_stack.append((cur_m, cur_k))
                self._mask = m1; self._mask_keep = m2
                self.maskChanged.emit(); self.update()
            except Exception:
                pass

        def __init__(self):
            super().__init__()

            # Enable keyboard focus (for Ctrl+Z / Ctrl+Y)
            try:
                self.setFocusPolicy(Qt.StrongFocus)
            except Exception:
                pass

            # State
            self._lock_image_size = False  # when True, masks don't resize if source image changes
            self._undo_stack = []          # list of (mask, mask_keep) QImages
            self._redo_stack = []
            self._stroke_active = False    # true while drawing; snapshot taken at press
            self._cursor_pos = None        # for brush outline

            self.setMinimumHeight(270)
            self._pix = QPixmap()
            self._bg = self._make_checker()
            # Image size in pixels (original content). We store masks in IMAGE coordinates.
            self._img_w = 0
            self._img_h = 0
            # Overlay tool state
            self._tool = 'brush'  # 'brush' or 'zoom'
            self._brush_radius = 20
            self._zoom = 1.0
            self._pan = [0, 0]
            self._drag_origin = None
            self._pan = [0, 0]  # pan offset in pixels (widget coords)
            self._drag_origin = None  # for panning in zoom tool
            self._draw_rect = None  # last drawn image rect for mapping
            # Mask images in IMAGE coords (allocated after image is set)
            from PySide6.QtGui import QImage
            self._mask = QImage()
            self._mask_keep = QImage()
            # Overlay opacity (0..1)
            self._overlay_opacity = 0.5
            # Accept drag & drop for quick load
            try: self.setAcceptDrops(True)
            except Exception: pass

            # Buttons
            self._btn_brush = QPushButton('🖌', self)
            self._btn_zoom  = QPushButton('🔍', self)
            for b in (self._btn_brush, self._btn_zoom):
                b.setFixedSize(36, 36)
                b.setCheckable(True)
                b.setCursor(Qt.PointingHandCursor)
                b.setStyleSheet(
                    'QPushButton { background: rgba(0,0,0,128); border-radius:8px; color:white; font-size:18px; }'
                    'QPushButton:checked { outline:2px solid #5aa0ff; }'
                )
            self._btn_brush.move(8, 8)
            self._btn_zoom.move(48, 8)
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            grp.addButton(self._btn_brush)
            grp.addButton(self._btn_zoom)
            self._btn_brush.setChecked(True)
            self._select_tool('brush')
            self._btn_brush.setChecked(True)
            self._btn_brush.clicked.connect(lambda: self._select_tool('brush'))
            self._btn_zoom.clicked.connect(lambda: self._select_tool('zoom'))

            # Small Undo / Redo buttons near tool icons
            self._btn_undo_small = QPushButton('↶', self)
            self._btn_redo_small = QPushButton('↷', self)
            for b in (self._btn_undo_small, self._btn_redo_small):
                b.setFixedSize(36, 36)
                b.setCursor(Qt.PointingHandCursor)
                b.setStyleSheet(
                    'QPushButton { background: rgba(0,0,0,128); border-radius:8px; color:white; font-size:18px; }'
                    'QPushButton:pressed { background: rgba(0,0,0,160); }'
                )
            try:
                self._btn_undo_small.move(88, 8)
                self._btn_redo_small.move(128, 8)
            except Exception:
                pass
            try:
                self._btn_undo_small.setToolTip('Undo (Ctrl+Z)')
                self._btn_redo_small.setToolTip('Redo (Ctrl+Y)')
            except Exception:
                pass
            self._btn_undo_small.clicked.connect(lambda: self.undo())
            self._btn_redo_small.clicked.connect(lambda: self.redo())

            # --- Tooltips for preview & tools ---
            try:
                self._btn_brush.setToolTip('Brush tool — left-drag = remove (make transparent), right-drag = keep. Mouse wheel changes brush size. Default radius: 20px.')
                self._btn_zoom.setToolTip('Zoom/Pan tool — mouse wheel to zoom, left-drag to pan. Default zoom: 1.0.')
                self.setToolTip('Preview — paint with 🖌 to remove/keep, or use 🔍 to zoom and pan. Overlay opacity is adjustable below.')
            except Exception:
                pass

        def _widget_to_image(self, pos):
            # Map widget position to image coordinates; return (x,y) or None if outside
            if not self._draw_rect or self._img_w <= 0 or self._img_h <= 0:
                return None
            x0, y0, w, h = self._draw_rect
            x = float(pos.x()) - float(x0)
            y = float(pos.y()) - float(y0)
            if x < 0 or y < 0 or x > w or y > h:
                return None
            ix = x * (self._img_w / float(w))
            iy = y * (self._img_h / float(h))
            return ix, iy, (self._img_w / float(w)), (self._img_h / float(h))

        def setImageSize(self, w: int, h: int):
            """Set the logical image size; allocate masks in image pixels."""
            from PySide6.QtGui import QImage
            w = int(max(1, w)); h = int(max(1, h))
            self._img_w, self._img_h = w, h
            self._mask = QImage(w, h, QImage.Format_Grayscale8); self._mask.fill(0)
            self._mask_keep = QImage(w, h, QImage.Format_Grayscale8); self._mask_keep.fill(0)
            self.update()

        def _ensure_masks(self):
            # Make sure we have image-sized masks allocated
            try:
                if (self._img_w <= 0 or self._img_h <= 0) and (not self._pix.isNull()):
                    self.setImageSize(self._pix.width(), self._pix.height())
                if self._mask.isNull() and self._img_w > 0 and self._img_h > 0:
                    from PySide6.QtGui import QImage
                    self._mask = QImage(self._img_w, self._img_h, QImage.Format_Grayscale8); self._mask.fill(0)
                if self._mask_keep.isNull() and self._img_w > 0 and self._img_h > 0:
                    from PySide6.QtGui import QImage
                    self._mask_keep = QImage(self._img_w, self._img_h, QImage.Format_Grayscale8); self._mask_keep.fill(0)
            except Exception:
                pass

        def _make_checker(self, s=8):
            img = QImage(s*2, s*2, QImage.Format_RGB32)
            c1 = QColor(200,200,200).rgb(); c2 = QColor(230,230,230).rgb()
            for y in range(s*2):
                for x in range(s*2):
                    img.setPixel(x,y, c1 if (x//s + y//s)%2==0 else c2)
            return QPixmap.fromImage(img)

        def setPixmap(self, pm: QPixmap):
            """Set the preview pixmap without nuking existing brush masks.
            Only (re)allocate/resize masks when the underlying IMAGE size changes.
            This fixes the bug where every recomposite wiped older strokes.
            """
            self._pix = pm
            try:
                w, h = int(pm.width()), int(pm.height())
                # First time: allocate masks
                if self._img_w <= 0 or self._img_h <= 0:
                    self.setImageSize(w, h)  # allocates blank masks intentionally
                else:
                    # If size changed, resample masks to new size to preserve strokes
                    
                    if (w != self._img_w) or (h != self._img_h):
                        if not getattr(self, "_lock_image_size", False):
                            from PySide6.QtCore import Qt as _Qt
                            try:
                                self._mask = self._mask.scaled(w, h, _Qt.IgnoreAspectRatio, _Qt.FastTransformation)
                            except Exception:
                                self._mask = self._mask.scaled(w, h)
                            try:
                                self._mask_keep = self._mask_keep.scaled(w, h, _Qt.IgnoreAspectRatio, _Qt.FastTransformation)
                            except Exception:
                                self._mask_keep = self._mask_keep.scaled(w, h)
                            self._img_w, self._img_h = w, h
                        else:
                            # Lock enabled: keep existing mask size; do not change _img_w/_img_h
                            pass

                # Otherwise: same size -> keep masks as-is
            except Exception:
                pass
            self.update()
        def setOverlayOpacity(self, v: float):
            try:
                self._overlay_opacity = max(0.0, min(1.0, float(v)))
            except Exception:
                self._overlay_opacity = 0.5
            self.update()

        def _select_tool(self, name: str):
            self._tool = name if name in ('brush','zoom') else 'brush'
            self.setCursor(Qt.CrossCursor if self._tool=='brush' else Qt.ArrowCursor)

        def resizeEvent(self, ev):
            # Widget resized; background checker will be regenerated automatically.
            # Masks are stored in image coordinates; no need to resize them here.
            return super().resizeEvent(ev)


        def wheelEvent(self, e):
            if self._tool == 'zoom':
                delta = e.angleDelta().y()
                factor = 1.0 + (0.0015 * delta)
                self._zoom = max(0.1, min(10.0, self._zoom * factor))
                # clamp pan so we don't lose the image completely
                try:
                    if not self._pix.isNull():
                        pm = self._pix
                        pm_fit = pm.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        zw = int(pm_fit.width() * self._zoom)
                        zh = int(pm_fit.height() * self._zoom)
                        max_off_x = max(0, (zw - self.width()) // 2 + 50)
                        max_off_y = max(0, (zh - self.height()) // 2 + 50)
                        self._pan[0] = max(-max_off_x, min(max_off_x, self._pan[0]))
                        self._pan[1] = max(-max_off_y, min(max_off_y, self._pan[1]))
                except Exception:
                    pass
                self.update()
            else:
                step = 2 if e.angleDelta().y() > 0 else -2
                self._brush_radius = max(1, self._brush_radius + step)
                self.update()

        def mousePressEvent(self, e):
            try: self.setFocus(Qt.MouseFocusReason)
            except Exception: pass

            if self._tool == 'brush' and e.button() == Qt.LeftButton:
                            try:
                                pth = chosen_bg.get('path') if isinstance(chosen_bg, dict) else None
                                from pathlib import Path as _P
                                if pth and _P(pth).name.startswith('inpaint_bg_'):
                                    chosen_bg['path'] = None
                                    try:
                                        lbl_img.setText('(none)')
                                    except Exception:
                                        pass
                                    try:
                                        # If replacer mode is Image, switch to Transparent
                                        if cmb_repl.currentIndex() == 3:
                                            cmb_repl.setCurrentIndex(0)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            _tmp = (not self._stroke_active and self._push_undo_snapshot()); self._stroke_active = True; self._paint_at(e.position() if hasattr(e, 'position') else e.pos())
            elif self._tool == 'brush' and e.button() == Qt.RightButton:
                            try:
                                pth = chosen_bg.get('path') if isinstance(chosen_bg, dict) else None
                                from pathlib import Path as _P
                                if pth and _P(pth).name.startswith('inpaint_bg_'):
                                    chosen_bg['path'] = None
                                    try:
                                        lbl_img.setText('(none)')
                                    except Exception:
                                        pass
                                    try:
                                        # If replacer mode is Image, switch to Transparent
                                        if cmb_repl.currentIndex() == 3:
                                            cmb_repl.setCurrentIndex(0)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            _tmp = (not self._stroke_active and self._push_undo_snapshot()); self._stroke_active = True; self._paint_keep_at(e.position() if hasattr(e, 'position') else e.pos())
            elif self._tool == 'zoom' and e.button() == Qt.LeftButton:
                self._drag_origin = ((e.position().x() if hasattr(e, 'position') else e.x()), (e.position().y() if hasattr(e, 'position') else e.y()))
                self.setCursor(Qt.ClosedHandCursor)
        def mouseMoveEvent(self, e):
            try: self._cursor_pos = (e.position().x() if hasattr(e,'position') else e.x(), e.position().y() if hasattr(e,'position') else e.y())
            except Exception: pass

            if self._tool == 'brush' and (e.buttons() & Qt.LeftButton):
                _tmp = (not self._stroke_active and self._push_undo_snapshot()); self._stroke_active = True; self._paint_at(e.position() if hasattr(e, 'position') else e.pos())
            if self._tool == 'brush' and (e.buttons() & Qt.RightButton):
                _tmp = (not self._stroke_active and self._push_undo_snapshot()); self._stroke_active = True; self._paint_keep_at(e.position() if hasattr(e, 'position') else e.pos())
            if self._tool == 'zoom' and (e.buttons() & Qt.LeftButton) and self._drag_origin is not None:
                dx = (e.position().x() if hasattr(e, 'position') else e.x()) - self._drag_origin[0]
                dy = (e.position().y() if hasattr(e, 'position') else e.y()) - self._drag_origin[1]
                self._pan[0] += int(dx)
                self._pan[1] += int(dy)
                self._drag_origin = ((e.position().x() if hasattr(e, 'position') else e.x()), (e.position().y() if hasattr(e, 'position') else e.y()))
                self.update()
            if self._tool == 'zoom' and (e.buttons() & Qt.LeftButton) and self._drag_origin is not None:
                dx = (e.position().x() if hasattr(e, 'position') else e.x()) - self._drag_origin[0]
                dy = (e.position().y() if hasattr(e, 'position') else e.y()) - self._drag_origin[1]
                self._pan[0] += int(dx)
                self._pan[1] += int(dy)
                self._drag_origin = ((e.position().x() if hasattr(e, 'position') else e.x()), (e.position().y() if hasattr(e, 'position') else e.y()))
                self.update()

        
        def mouseReleaseEvent(self, e):
            # End brush stroke and reset zoom drag
            try:
                if self._tool == 'brush' and e.button() in (Qt.LeftButton, Qt.RightButton):
                    self._stroke_active = False
            except Exception:
                pass
            # Auto-inpaint after finishing a LEFT-button remove stroke
            try:
                if self._tool == 'brush' and e.button() == Qt.LeftButton:
                    # Run SD1.5 inpaint using current settings if available
                    if Preview._ui_flags.get('auto_inpaint', False) and '_sd15_inpaint_available' in globals() and _sd15_inpaint_available():
                        try:
                            on_inpaint_sd15()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if self._tool == 'zoom' and e.button() == Qt.LeftButton:
                    self._drag_origin = None
                    self.setCursor(Qt.ArrowCursor)
            except Exception:
                pass
            return super().mouseReleaseEvent(e)
        def keyPressEvent(self, e):
            try:
                if (e.modifiers() & Qt.ControlModifier) and getattr(e, 'key', lambda: None)() == Qt.Key_Z:
                    self.undo(); return
                if (e.modifiers() & Qt.ControlModifier) and getattr(e, 'key', lambda: None)() == Qt.Key_Y:
                    self.redo(); return
            except Exception:
                pass
            try:
                super().keyPressEvent(e)
            except Exception:
                pass
        def _paint_at(self, pos):
            from PySide6.QtGui import QPainter
            from PySide6.QtCore import QPointF
            self._ensure_masks()
            mapped = self._widget_to_image(pos)
            if mapped is None: return
            ix, iy, sx, sy = mapped
            rad = int(max(1, round(self._brush_radius * (self._img_w / max(1.0, self._draw_rect[2])))))
            p = QPainter(self._mask)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen); p.setBrush(Qt.white)
            p.drawEllipse(QPointF(ix, iy), rad, rad)
            p.end()
            self.maskChanged.emit(); self.update()

        def _paint_keep_at(self, pos):
            from PySide6.QtGui import QPainter
            from PySide6.QtCore import QPointF
            self._ensure_masks()
            mapped = self._widget_to_image(pos)
            if mapped is None: return
            ix, iy, sx, sy = mapped
            rad = int(max(1, round(self._brush_radius * (self._img_w / max(1.0, self._draw_rect[2])))))
            p = QPainter(self._mask_keep)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen); p.setBrush(Qt.white)
            p.drawEllipse(QPointF(ix, iy), rad, rad)
            p.end()
            self.maskChanged.emit(); self.update()

        def clearMask(self):
            if not self._mask.isNull():
                self._mask.fill(0)
            if not self._mask_keep.isNull():
                try: self._mask_keep.fill(0)
                except Exception: pass
            self.maskChanged.emit(); self.update()

        
        def export_masks_to_image_size(self, target_w, target_h):
            """Return (remove_mask, keep_mask) as float32 [0..1] arrays at image size.
            Masks are stored in image coordinates; resize if target differs."""
            import numpy as _np
            from PySide6.QtGui import QImage
            from PIL import Image as _PILImage
            def _qimg_to_np(qimg, tw, th):
                if qimg is None or qimg.isNull(): return None
                q = qimg.convertToFormat(QImage.Format_Grayscale8)
                w = q.width(); h = q.height()
                bpl = q.bytesPerLine()
                ptr = q.constBits()
                try:
                    buf = ptr.tobytes()
                except AttributeError:
                    buf = bytes(ptr)
                arr = _np.frombuffer(buf, dtype=_np.uint8).reshape((h, bpl))[:, :w]
                pil = _PILImage.fromarray(arr, mode='L')
                if (w, h) != (tw, th):
                    pil = pil.resize((tw, th), _PILImage.BILINEAR)
                out = _np.asarray(pil).astype(_np.float32) / 255.0
                return out
            rm = _qimg_to_np(self._mask, target_w, target_h)
            km = _qimg_to_np(self._mask_keep, target_w, target_h)
            return rm, km
        def export_mask_to_image_size(self, target_w, target_h):
                    from PySide6.QtGui import QImage
                    import numpy as _np
                    from PIL import Image as _PILImage
                    q = self._mask.convertToFormat(QImage.Format_Grayscale8)
                    w = q.width(); h = q.height()
                    bpl = q.bytesPerLine()
                    ptr = q.constBits()
                    try:
                        buf = ptr.tobytes()
                    except AttributeError:
                        buf = bytes(ptr)
                    arr = _np.frombuffer(buf, dtype=_np.uint8).reshape((h, bpl))[:, :w]
                    pil = _PILImage.fromarray(arr, mode='L')
                    if (w, h) != (target_w, target_h):
                        pil = pil.resize((target_w, target_h), _PILImage.BILINEAR)
                    out = _np.asarray(pil).astype(_np.float32) / 255.0
                    return out

        def dragEnterEvent(self, e):
            try:
                if e.mimeData().hasUrls():
                    e.acceptProposedAction(); return
            except Exception:
                pass
            e.ignore()

        def dropEvent(self, e):
            try:
                urls = e.mimeData().urls()
                if urls:
                    p = urls[0].toLocalFile()
                    if p:
                        self.fileDropped.emit(p)
                        e.acceptProposedAction(); return
            except Exception:
                pass
            e.ignore()

        def paintEvent(self, ev):
            from PySide6.QtGui import QPainter
            p = QPainter(self)
            # draw checker
            for y in range(0, self.height(), self._bg.height()):
                for x in range(0, self.width(), self._bg.width()):
                    p.drawPixmap(x, y, self._bg)
            # draw pixmap with zoom, centered
            if not self._pix.isNull():
                pm = self._pix
                # base size to fit
                pm_fit = pm.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                zw = int(pm_fit.width() * self._zoom)
                zh = int(pm_fit.height() * self._zoom)
                pm_zoom = pm_fit.scaled(zw, zh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - pm_zoom.width()) // 2 + int(self._pan[0])
                y = (self.height() - pm_zoom.height()) // 2 + int(self._pan[1])
                self._draw_rect = (x, y, pm_zoom.width(), pm_zoom.height())
                p.drawPixmap(x, y, pm_zoom)
            # draw mask overlays anchored to the image rect
            try:
                p.setOpacity(self._overlay_opacity)
            except Exception:
                p.setOpacity(0.5)
            p.setPen(Qt.NoPen)
            from PySide6.QtGui import QPainter as _QP
            try:
                old_mode = p.compositionMode()
            except Exception:
                old_mode = None
            try:
                if self._draw_rect and not self._mask.isNull():
                    x, y, w, h = self._draw_rect
                    m1 = self._mask.scaled(w, h)
                    p.setCompositionMode(_QP.CompositionMode_Screen)
                    p.drawImage(int(x), int(y), m1)
                if self._draw_rect and not self._mask_keep.isNull():
                    x, y, w, h = self._draw_rect
                    m2 = self._mask_keep.scaled(w, h)
                    p.drawImage(int(x), int(y), m2)
            finally:
                try:
                    if old_mode is not None:
                        p.setCompositionMode(old_mode)
                except Exception:
                    pass
            
            # --- Brush outline + live radius label ---
            try:
                if self._tool == 'brush' and self._draw_rect and self._cursor_pos is not None:
                    cx, cy = int(self._cursor_pos[0]), int(self._cursor_pos[1])
                    x0, y0, w, h = self._draw_rect
                    if (x0 <= cx <= x0 + w) and (y0 <= cy <= y0 + h):
                        r_disp = int(max(1, self._brush_radius))
                        from PySide6.QtGui import QPen
                        pen1 = QPen(QColor(0,0,0,160)); pen1.setWidth(3)
                        pen2 = QPen(QColor(255,255,255,220)); pen2.setWidth(1)
                        p.setPen(pen1); p.setBrush(Qt.NoBrush)
                        p.drawEllipse(cx - r_disp, cy - r_disp, 2*r_disp, 2*r_disp)
                        p.setPen(pen2)
                        p.drawEllipse(cx - r_disp, cy - r_disp, 2*r_disp, 2*r_disp)
                        # Label text
                        try:
                            rad_img = int(round(self._brush_radius * (self._img_w / max(1.0, float(w)))))
                        except Exception:
                            rad_img = r_disp
                        txt = f"{rad_img} px"
                        fm = p.fontMetrics()
                        tw = fm.horizontalAdvance(txt) + 8
                        th = fm.height() + 4
                        tx = cx + r_disp + 8
                        ty = cy - r_disp - th - 6
                        tx = min(max(4, tx), self.width() - tw - 4)
                        ty = min(max(4, ty), self.height() - th - 4)
                        p.setBrush(QColor(0,0,0,160)); p.setPen(Qt.NoPen)
                        p.drawRoundedRect(tx, ty, tw, th, 6, 6)
                        p.setPen(QColor(255,255,255,230)); p.drawText(tx+4, ty+th-6, txt)
            except Exception:
                pass

            p.end()


    container = QWidget()
    vbox = QVBoxLayout(container)

    preview = Preview()

    vbox.addWidget(preview, stretch=3)  # preview gets most of the space

    # Controls panel below
    panel = QWidget()
    lay = QFormLayout(panel)
    vbox.addWidget(panel, stretch=0)

    # ---- Controls ----
    cmb_engine = QComboBox(); cmb_engine.addItems(["Auto", "MODNet (portrait)", "BiRefNet (general)"])
    cmb_mode   = QComboBox(); cmb_mode.addItems(["Keep subject (remove BG)", "Keep background (remove subject)", "Alpha only (mask)"])

    # Source row: Use current + Load external
    btn_use_current = QPushButton("Use current")
    cmb_source = QComboBox(); cmb_source.addItems(["Image", "Video"]); cmb_source.setCurrentIndex(0)
    t_seek = QDoubleSpinBox(); t_seek.setRange(0, 36000); t_seek.setDecimals(3); t_seek.setSingleStep(0.5); t_seek.setValue(0.0)
    btn_load = QPushButton("Load external…")
    source_row = QHBoxLayout()
    try:
        cmb_source.setVisible(False); t_seek.setVisible(False)
    except Exception:
        pass

    # Replacer
    cmb_repl = QComboBox(); cmb_repl.addItems(["Transparent", "Color", "Blur", "Image"])
    s_blurbg = QSpinBox(); s_blurbg.setRange(0, 200); s_blurbg.setValue(20)

    color_preview = QLabel("●"); color_preview.setStyleSheet("font-size:18px;"); color_val = QColor(255,255,255)
    btn_color = QPushButton("Pick color…")
    def pick_color():
        nonlocal color_val
        try:
            c = QColorDialog.getColor(color_val, panel, "Background color")
            if c.isValid():
                color_val = c; color_preview.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()}); font-size:18px;")
        except Exception:
            pass
    btn_color.clicked.connect(pick_color)

    lbl_img = QLabel("(none)"); btn_img = QPushButton("Choose image…"); chosen_bg = {"path": None}

    def pick_img():
        try:
            file, _ = QFileDialog.getOpenFileName(panel, "Choose background image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        except Exception:
            file = ""
        if file:
            chosen_bg["path"] = file; lbl_img.setText(Path(file).name); schedule_update()
    btn_img.clicked.connect(pick_img)

    # Post
    s_thresh = QSpinBox(); s_thresh.setRange(0,255); s_thresh.setValue(0)
    s_feath  = QSpinBox(); s_feath.setRange(0,50);  s_feath.setValue(3)
    sl_aggr = QSlider(Qt.Horizontal); sl_aggr.setRange(0,100); sl_aggr.setValue(50)
    s_edge   = QSpinBox(); s_edge.setRange(-50,50); s_edge.setValue(0)
    # --- Fine controls wrapped in collapsible 'remove background' ---
    row_top1 = QHBoxLayout()
    row_top1.addWidget(QLabel("Hard threshold (0=off)")); row_top1.addWidget(s_thresh)
    row_top1.addSpacing(12)
    row_top1.addWidget(QLabel("Feather (px)")); row_top1.addWidget(s_feath)

    row_top2 = QHBoxLayout()
    row_top2.addWidget(QLabel("Removal strength"))
    row_top2.addWidget(sl_aggr)

    row_top3 = QHBoxLayout()
    row_top3.addWidget(QLabel("Grow/ shrink edge (px)"))
    row_top3.addWidget(s_edge)

    remove_box = CollapsibleBox('remove background')
    _rm_lay = QVBoxLayout()
    _rm_lay.addLayout(row_top1)
    _rm_lay.addLayout(row_top2)
    _rm_lay.addLayout(row_top3)
    remove_box.setContentLayout(_rm_lay)
    vbox.insertWidget(0, remove_box)
    cb_spill = QCheckBox("Edge de-spill"); cb_spill.setChecked(True)
    cb_auto  = QCheckBox("Auto-level mask"); cb_auto.setChecked(True)
    cb_inv   = QCheckBox("Invert mask"); cb_inv.setChecked(False)
    cb_ah    = QCheckBox("Edge anti-halo"); cb_ah.setChecked(True)
    cb_locksize = QCheckBox("Lock image size"); cb_locksize.setChecked(False)
    try:
        cb_locksize.toggled.connect(lambda v: preview.setLockImageSize(v))
    except Exception:
        pass

    # Mask overlay
    s_moverlay = QSpinBox(); s_moverlay.setRange(0,100); s_moverlay.setValue(50)
    try:
        s_moverlay.valueChanged.connect(lambda v: preview.setOverlayOpacity(v/100.0))
        sl_aggr.valueChanged.connect(lambda _: schedule_update())
        s_edge.valueChanged.connect(lambda _: schedule_update())
    except Exception:
        pass


    # Shadow
    cb_shadow= QCheckBox("Drop shadow"); cb_shadow.setChecked(False)
    s_salpha= QSpinBox(); s_salpha.setRange(0,255); s_salpha.setValue(120)
    s_sblur = QSpinBox(); s_sblur.setRange(0,200); s_sblur.setValue(20)
    s_sox   = QSpinBox(); s_sox.setRange(-500,500); s_sox.setValue(10)
    s_soy   = QSpinBox(); s_soy.setRange(-500,500); s_soy.setValue(10)
    shadow_row = QHBoxLayout()
    # Opacity moved next to 'Mask overlay' row above Engine
    shadow_row.addWidget(QLabel("Blur"));    shadow_row.addWidget(s_sblur)
    shadow_row.addWidget(QLabel("Offset X"));shadow_row.addWidget(s_sox)
    shadow_row.addWidget(QLabel("Offset Y"));shadow_row.addWidget(s_soy)
    # Output folder row
    out_dir = _get_out_dir_pref()
    le_out = QLineEdit(str(out_dir)); le_out.setReadOnly(True)
    btn_out_change = QPushButton("Change…"); btn_out_open = QPushButton("Open folder")
    def change_out():
        try:
            d = QFileDialog.getExistingDirectory(panel, "Choose output folder", str(le_out.text()))
        except Exception:
            d = ""
        if d: _set_out_dir_pref(Path(d)); le_out.setText(d)
    btn_out_change.clicked.connect(change_out)
    def open_out():
        path = le_out.text().strip()
        if not path: return
        pth = Path(path); pth.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"): os.startfile(str(pth))
            elif sys.platform == "darwin": subprocess.run(["open", str(pth)], check=False)
            else: subprocess.run(["xdg-open", str(pth)], check=False)
        except Exception: pass
    btn_out_open.clicked.connect(open_out)
    out_row = QHBoxLayout(); out_row.addWidget(le_out); out_row.addWidget(btn_out_change); out_row.addWidget(btn_out_open)

    # Actions
    btn_recompute = QPushButton("Remove BG")
    btn_reset     = QPushButton("Reset")
    btn_undo      = QPushButton("Undo")
    try:
        btn_undo.clicked.connect(lambda: preview.undo())
    except Exception:
        pass
    btn_save      = QPushButton("Save")
    btns_row = QHBoxLayout(); btns_row.addWidget(btn_use_current); btns_row.addWidget(btn_load); btns_row.addWidget(btn_recompute); btns_row.addWidget(btn_undo); btns_row.addWidget(btn_reset); btns_row.addWidget(btn_save); btns_row.addStretch(1); btn_sd_inpaint = QPushButton("INPAINT"); btns_row.addWidget(btn_sd_inpaint);    # Combined row: Mask overlay + Shadow Opacity (moved here per request)
    row_overlay_opacity = QHBoxLayout()
    row_overlay_opacity.addWidget(QLabel("Mask overlay")); row_overlay_opacity.addWidget(s_moverlay)
    row_overlay_opacity.addSpacing(12)
    row_overlay_opacity.addWidget(QLabel("Shadow opacity")); row_overlay_opacity.addWidget(s_salpha)
    lay.addRow(row_overlay_opacity)
    lay.addRow("Lock image size", cb_locksize)

    # --- Tooltips (defaults in parentheses) ---
    try:
        cmb_engine.setToolTip('Segmentation engine used to compute the mask. Auto picks a working model (default: Auto).')
        cmb_mode.setToolTip('What to output: keep subject, keep background only, or show the alpha mask (default: Keep subject).')
        btn_use_current.setToolTip('Grab the image or current video frame already loaded in the app and use it here.')
        cmb_source.setToolTip('Input type for loading from disk (default: Image).')
        t_seek.setToolTip('Video timestamp to grab a frame when loading a video (default: 0.000 s).')
        btn_load.setToolTip('Load an external image or video from disk. If video, uses the Seek time.')

        cmb_repl.setToolTip('Background replacement mode: Transparent, Color, Blur, or Image (default: Transparent).')
        s_blurbg.setToolTip('Blur radius used when replacement mode is Blur (default: 20 px).')
        btn_color.setToolTip('Pick background color used when mode is Color (default: white #FFFFFF).')
        color_preview.setToolTip('Current background color (default: white #FFFFFF).')
        btn_img.setToolTip('Choose an image file used as replacement background.')
        lbl_img.setToolTip('Selected background image filename (default: none).')

        s_thresh.setToolTip('Hard threshold on the alpha. 0 disables thresholding (default: 0).')
        s_feath.setToolTip('Feather radius in pixels — softens edges (default: 3 px).')
        s_edge.setToolTip('Grow (>0) or shrink (<0) the cutout edge in pixels (default: 0).')
        sl_aggr.setToolTip('Removal strength — lower keeps more of the original, higher removes more (default: 50).')
        cb_spill.setToolTip('Edge de-spill — reduces color contamination near edges (default: enabled).')
        cb_auto.setToolTip('Auto-level mask — stretches mask contrast to improve edges (default: enabled).')
        cb_inv.setToolTip('Invert the mask (default: off).')
        cb_ah.setToolTip('Edge anti-halo — light color decontamination (default: enabled).')

        s_moverlay.setToolTip('Mask overlay opacity on the preview (default: 50%).')
        cb_shadow.setToolTip('Enable a simple drop shadow under the subject (default: off).')
        s_salpha.setToolTip('Shadow opacity (default: 120 / 255).')
        s_sblur.setToolTip('Shadow blur radius (default: 20 px).')
        s_sox.setToolTip('Shadow X offset in pixels (default: 10). Negative moves left.')
        s_soy.setToolTip('Shadow Y offset in pixels (default: 10). Negative moves up.')

        le_out.setToolTip('Output folder for saved results.')
        btn_out_change.setToolTip('Change the output folder and remember it for next time.')
        btn_out_open.setToolTip('Open the output folder in your file manager.')

        btn_recompute.setToolTip('Recompute the alpha mask using the selected engine.')
        btn_reset.setToolTip('Reset controls and masks back to defaults.')
        btn_undo.setToolTip('Undo the last brush stroke or change (where available).')
        btn_save.setToolTip('Save the current result as a PNG into the output folder.')
        btn_sd_inpaint.setToolTip('Run Stable Diffusion 1.5 Inpaint over removed areas to fill them realistically.')

        cmb_presets.setToolTip('Saved presets of all controls. Select to apply.')
        btn_preset_save.setToolTip('Save the current settings as a preset.')
        btn_preset_del.setToolTip('Delete the selected preset.')

        le_sd_prompt.setToolTip('Positive text prompt describing the desired fill (default: "clean background, realistic").')
        le_sd_neg.setToolTip('Negative prompt to avoid unwanted artifacts (default: "blurry, artifacts, distortion, text, watermark").')
        s_sd_steps.setToolTip('Number of diffusion steps (default: 30).')
        ds_sd_guid.setToolTip('Classifier-free guidance scale — higher follows the prompt more strongly (default: 7.0).')
        ds_sd_str.setToolTip('Inpaint strength — 0 keeps original, 1 replaces fully (default: 0.75).')
        chk_auto_inpaint.setToolTip('When enabled, runs inpaint automatically after each left-brush stroke (default: off).')
        chk_freeze_unmasked.setToolTip('Keep unmasked pixels exactly as-is; only paste the generated result where masked (default: on).')
        s_sd_pad.setToolTip('Dilate the inpaint mask by N pixels to give the model more context (default: 8 px).')
        s_sd_seed.setToolTip('Random seed; -1 uses a new random seed each run (default: -1).')
        le_sd_model.setToolTip('Path to sd-v1-5-inpainting model (.safetensors).')
        btn_sd_browse.setToolTip('Browse for the SD 1.5 inpaint model file.')
    except Exception:
        pass
        try:
            sd_form.addRow("", chk_sd_auto_bg)
        except Exception:
            pass

    # Layout wire
    # moved below: combined Engine + Model row
    lay.addRow("Mode",   cmb_mode)
    # Presets row
    cmb_presets = QComboBox(); btn_preset_save = QPushButton("Save preset"); btn_preset_del = QPushButton("Delete")
    # (moved) presets row combined below steps/seed
    # (moved) presets buttons now inline with combo
    # --- SD Inpaint (SD 1.5) controls ---
    sd_grp = QWidget(); sd_form = QFormLayout(sd_grp)
    le_sd_prompt = QLineEdit("clean background, realistic")
    chk_sd_auto_bg = QCheckBox("Auto apply as background")
    le_sd_neg    = QLineEdit("blurry, artifacts, distortion, text, watermark")
    s_sd_steps   = QSpinBox(); s_sd_steps.setRange(1,150); s_sd_steps.setValue(30)
    ds_sd_guid   = QDoubleSpinBox(); ds_sd_guid.setRange(0.0,30.0); ds_sd_guid.setSingleStep(0.5); ds_sd_guid.setValue(7.0)
    ds_sd_guid.setToolTip('Guidance (CFG): how strongly the fill follows your prompt. Lower = more natural; higher = more literal (default: 7.0).')
    ds_sd_str    = QDoubleSpinBox(); ds_sd_str.setRange(0.0,1.0); ds_sd_str.setSingleStep(0.05); ds_sd_str.setValue(0.85)
    ds_sd_str.setToolTip('Inpaint Strength: 0 keeps more original pixels, 1 replaces fully with generated content (default: 0.85).')
    s_sd_seed    = QSpinBox(); s_sd_seed.setRange(-1, 2147483647); s_sd_seed.setValue(-1)
    # Combine Steps+Seed on one line
    row_steps_seed = QHBoxLayout()
    row_steps_seed.addWidget(QLabel("Steps")); row_steps_seed.addWidget(s_sd_steps)
    row_steps_seed.addSpacing(12)
    row_steps_seed.addWidget(QLabel("Seed")); row_steps_seed.addWidget(s_sd_seed)
    # moved to inpaint_box

    # Guidance+Strength combined row (moved above Output area)
    row_guid_str = QHBoxLayout()
    row_guid_str.addWidget(QLabel("Guidance")); row_guid_str.addWidget(ds_sd_guid)
    row_guid_str.addSpacing(12)
    row_guid_str.addWidget(QLabel("Strength")); row_guid_str.addWidget(ds_sd_str)
    # We'll insert this container-level row before the Output row later.

    le_sd_model  = QLineEdit(str(_default_sd15_inpaint_model_path())); btn_sd_browse = QPushButton("…")
    def _browse_sd_model():
        try:
            f, _ = QFileDialog.getOpenFileName(panel, "Choose SD1.5 Inpaint model", str(le_sd_model.text()), "Safetensors (*.safetensors)")
        except Exception:
            f = ""
        if f: le_sd_model.setText(f)
    btn_sd_browse.clicked.connect(_browse_sd_model)

    row_sd_model = QHBoxLayout(); row_sd_model.addWidget(le_sd_model); row_sd_model.addWidget(btn_sd_browse)
    # Combined Engine + SD Inpaint Model on one line
    row_engine_model = QHBoxLayout()
    row_engine_model.addWidget(QLabel("Engine")); row_engine_model.addWidget(cmb_engine)
    row_engine_model.addSpacing(12)
    row_engine_model.addWidget(QLabel("Model")); row_engine_model.addLayout(row_sd_model)
    lay.addRow(row_engine_model)

    # --- Inpaint controls group ---
    inpaint_box = CollapsibleBox('inpaint')
    inpaint_form = QFormLayout()
    inpaint_box.setContentLayout(inpaint_form)

    # Add requested rows inside 'inpaint'
    inpaint_form.addRow(row_guid_str)
    inpaint_form.addRow(row_steps_seed)
    inpaint_form.addRow("Prompt", le_sd_prompt)
    inpaint_form.addRow("Negative", le_sd_neg)
    chk_auto_inpaint = QCheckBox("Auto inpaint after brush"); chk_auto_inpaint.setChecked(False)
    chk_freeze_unmasked = QCheckBox("Freeze unmasked area (composite only masked)"); chk_freeze_unmasked.setChecked(True)
    s_sd_pad = QSpinBox(); s_sd_pad.setRange(0,64); s_sd_pad.setValue(8)
    row_tog = QHBoxLayout(); row_tog.addWidget(chk_auto_inpaint); row_tog.addWidget(chk_freeze_unmasked)
    row_pad = QHBoxLayout(); row_pad.addWidget(QLabel("Pad mask edges (px)")); row_pad.addWidget(s_sd_pad)
    inpaint_form.addRow(row_tog)
    inpaint_form.addRow(row_pad)
    try:
        chk_auto_inpaint.toggled.connect(lambda v: setattr(Preview, '_ui_flags', {**Preview._ui_flags, 'auto_inpaint': bool(v)}))
        chk_freeze_unmasked.toggled.connect(lambda v: setattr(Preview, '_ui_flags', {**Preview._ui_flags, 'freeze_unmasked': bool(v)}))
    except Exception:
        pass
    lay.addRow(inpaint_box)

    # --- Background presets and settings group ---
    bg_box = CollapsibleBox('background presets and settings')
    bg_form = QFormLayout()
    bg_box.setContentLayout(bg_form)
    lay.addRow(bg_box)

    # moved to inpaint_box
    # moved to inpaint_box
    # moved: Steps sits next to Seed
    # moved: Guidance sits with Strength above Output
    # moved: Strength sits with Guidance above Output
    # moved: Seed sits next to Steps
    # moved: Model sits on the Engine row
    lay.addRow(sd_grp)

    # Presets combo + buttons on one line, placed below Steps/Seed
    row_presets = QHBoxLayout(); row_presets.addWidget(cmb_presets); row_presets.addWidget(btn_preset_save); row_presets.addWidget(btn_preset_del)
    bg_form.addRow("Presets", row_presets)

    # hidden: source row (type/seek) removed from UI
    bg_form.addRow("Background", cmb_repl)
    bg_form.addRow("Blur amount", s_blurbg)
    bg_form.addRow("BG color", btn_color); bg_form.addRow(" ", color_preview)


        
    bg_form.addRow("BG image", btn_img);   bg_form.addRow(" ", lbl_img)
    # Removal strength (bias)
    row_tog1 = QHBoxLayout(); row_tog1.addWidget(cb_spill); row_tog1.addWidget(cb_inv); row_tog1.addWidget(cb_shadow)
    row_tog2 = QHBoxLayout(); row_tog2.addWidget(cb_auto);  row_tog2.addWidget(cb_ah)
    bg_form.addRow(row_tog1)
    bg_form.addRow(row_tog2)
    bg_form.addRow(shadow_row)
    vbox.insertLayout(1, btns_row)
    # Move Output path controls directly under the preview (before the main form panel)
    out_line = QHBoxLayout(); out_line.addWidget(QLabel("Output")); out_line.addLayout(out_row)
    # moved into inpaint_box
    vbox.insertLayout(4, out_line)


    # Mount into CollapsibleSection
    try:
        section_widget.setContentLayout(vbox)
    except Exception:
        pass

    # --- state ---
    current_rgb: Optional[np.ndarray] = None
    current_alpha: Optional[np.ndarray] = None
    current_path: Optional[Path] = None
    undo_stack: List[Dict] = []

    # --- helpers ---
    def qimage_from_pil(im: Image.Image) -> QImage:
        arr = np.array(im.convert("RGBA"))  # HWC uint8
        h,w = arr.shape[:2]
        return QImage(arr.data, w, h, 4*w, QImage.Format_RGBA8888).copy()

    def update_preview_pix(pim: Image.Image):
        pm = QPixmap.fromImage(qimage_from_pil(pim)); preview.setPixmap(pm)

    

    def get_params() -> Dict:
        return dict(
            engine={0:"auto",1:"modnet",2:"birefnet"}[cmb_engine.currentIndex()],
            mode={0:"keep_subject",1:"keep_bg",2:"alpha_only"}[cmb_mode.currentIndex()],
            threshold=int(s_thresh.value()), feather=int(s_feath.value()), bias=int(sl_aggr.value()),
            spill=bool(cb_spill.isChecked()), invert=bool(cb_inv.isChecked()), auto_level=bool(cb_auto.isChecked()),
            repl_mode={0:"transparent",1:"color",2:"blur",3:"image"}.get(cmb_repl.currentIndex(), "transparent"),
            repl_color=(color_val.red(), color_val.green(), color_val.blue()),
            repl_blur=int(s_blurbg.value()), repl_image=chosen_bg["path"],
            drop_shadow=bool(cb_shadow.isChecked()), shadow_alpha=int(s_salpha.value()), shadow_blur=int(s_sblur.value()),
            shadow_offx=int(s_sox.value()), shadow_offy=int(s_soy.value()), anti_halo=bool(cb_ah.isChecked()),
            overlay=int(s_moverlay.value()),
            out_dir=le_out.text().strip(),
            sd_prompt=le_sd_prompt.text().strip(),
            sd_negative=le_sd_neg.text().strip(),
            sd_steps=int(s_sd_steps.value()),
            sd_guidance=float(ds_sd_guid.value()),
            sd_strength=float(ds_sd_str.value()),
            sd_seed=int(s_sd_seed.value()),
            sd_model=le_sd_model.text().strip(),
            sd_auto_bg=bool(chk_sd_auto_bg.isChecked()),
        )

    def set_params(d: Dict):
        nonlocal color_val
        cmb_engine.setCurrentIndex({"auto":0,"modnet":1,"birefnet":2}.get(d.get("engine","auto"),0))
        cmb_mode.setCurrentIndex({"keep_subject":0,"keep_bg":1,"alpha_only":2}.get(d.get("mode","keep_subject"),0))
        s_thresh.setValue(int(d.get("threshold",0))); s_feath.setValue(int(d.get("feather",3)))
        cb_spill.setChecked(bool(d.get("spill",True))); cb_inv.setChecked(bool(d.get("invert",False))); cb_auto.setChecked(bool(d.get("auto_level",True)))
        cmb_repl.setCurrentIndex({"transparent":0,"color":1,"blur":2,"image":3}.get(d.get("repl_mode","transparent"),0))
        r,g,b = d.get("repl_color",(255,255,255)); color_val = QColor(int(r),int(g),int(b)); color_preview.setStyleSheet(f"color: rgb({int(r)},{int(g)},{int(b)}); font-size:18px;")
        s_blurbg.setValue(int(d.get("repl_blur",20)))
        chosen_bg["path"] = d.get("repl_image"); lbl_img.setText(Path(d.get("repl_image")).name if d.get("repl_image") else "(none)")
        cb_shadow.setChecked(bool(d.get("drop_shadow",False))); s_salpha.setValue(int(d.get("shadow_alpha",120))); s_sblur.setValue(int(d.get("shadow_blur",20)))
        s_sox.setValue(int(d.get("shadow_offx",10))); s_soy.setValue(int(d.get("shadow_offy",10))); cb_ah.setChecked(bool(d.get("anti_halo",True)))
        try: sl_aggr.setValue(int(d.get("bias", 50)))
        except Exception: pass
        try: s_moverlay.setValue(int(d.get("overlay", 50))); preview.setOverlayOpacity(s_moverlay.value()/100.0)
        except Exception: pass
        try: s_edge.setValue(int(d.get("edge_shift", 0)))
        except Exception: pass
        # SD fields
        try: le_sd_prompt.setText(str(d.get("sd_prompt","clean background, realistic")))
        except Exception: pass
        try: le_sd_neg.setText(str(d.get("sd_negative","blurry, artifacts, distortion, text, watermark")))
        except Exception: pass
        try: s_sd_steps.setValue(int(d.get("sd_steps",30)))
        except Exception: pass
        try: ds_sd_guid.setValue(float(d.get("sd_guidance",7.0)))
        except Exception: pass
        try: ds_sd_str.setValue(float(d.get("sd_strength",0.85)))
        except Exception: pass
        try: s_sd_seed.setValue(int(d.get("sd_seed",-1)))
        except Exception: pass
        try: le_sd_model.setText(str(d.get("sd_model", le_sd_model.text())))
        except Exception: pass
        try: chk_sd_auto_bg.setChecked(bool(d.get("sd_auto_bg", False)))
        except Exception: pass
        try: chk_auto_inpaint.setChecked(bool(d.get("sd_auto_inpaint", False)))
        except Exception: pass
        try: chk_freeze_unmasked.setChecked(bool(d.get("sd_freeze_unmasked", True)))
        except Exception: pass
        try: s_sd_pad.setValue(int(d.get("sd_pad_px", 8)))
        except Exception: pass
        # out dir
        try:
            txt = d.get("out_dir")
            if txt:
                le_out.setText(str(txt))
        except Exception:
            pass

    def defaults() -> Dict:
        return dict(engine="auto", mode="keep_subject", threshold=0, feather=3, bias=50, edge_shift=0, spill=True, invert=False, auto_level=True,
                    repl_mode="transparent", repl_color=(255,255,255), repl_blur=20, repl_image=None,
                    drop_shadow=False, shadow_alpha=120, shadow_blur=20, shadow_offx=10, shadow_offy=10, anti_halo=True,
                    sd_prompt="clean background, realistic", sd_negative="blurry, artifacts, distortion, text, watermark",
                    sd_steps=35, sd_guidance=8.5, sd_strength=0.75, sd_seed=-1, sd_model=str(_default_sd15_inpaint_model_path()),
                sd_auto_bg=False, sd_auto_inpaint=False, sd_freeze_unmasked=True, sd_pad_px=8, overlay=50, out_dir=str(_get_out_dir_pref()))

    def _load_presets_list():
            p = _presets_store_path()
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    return data.get("bg_presets", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                except Exception:
                    return []
            # Optional one-time migration from old configackground_tool.json
            try:
                old = _prefs_path()
                if old.exists():
                    data = json.loads(old.read_text(encoding="utf-8"))
                    lst = data.get("bg_presets", [])
                    if lst:
                        _save_presets_list(lst)
                        return lst
            except Exception:
                pass
            return []

    def _save_presets_list(lst):
            p = _presets_store_path()
            try:
                payload = {"bg_presets": lst}
                p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
    def _refresh_presets_ui():
        try:
            cmb_presets.blockSignals(True); cmb_presets.clear()
            lst = _load_presets_list()
            for item in lst:
                cmb_presets.addItem(item.get("name","Preset"))
        finally:
            cmb_presets.blockSignals(False)
    def on_preset_save():
        lst = _load_presets_list()
        name = f"Preset {len(lst)+1}"
        try:
            from PySide6.QtWidgets import QInputDialog
            n, ok = QInputDialog.getText(panel, "Save preset", "Name:", text=name)
            if ok and n: name = n
        except Exception:
            pass
        lst.append({"name": name, "params": get_params()})
        _save_presets_list(lst); _refresh_presets_ui()
    def on_preset_apply(idx):
        lst = _load_presets_list()
        if 0 <= idx < len(lst):
            set_params(lst[idx]["params"]); schedule_update()
    def on_preset_delete():
        lst = _load_presets_list()
        idx = cmb_presets.currentIndex()
        if 0 <= idx < len(lst):
            del lst[idx]; _save_presets_list(lst); _refresh_presets_ui()
    try:
        btn_preset_save.clicked.connect(on_preset_save)
        btn_preset_del.clicked.connect(on_preset_delete)
        cmb_presets.currentIndexChanged.connect(on_preset_apply)
        _refresh_presets_ui()
    except Exception:
        pass

    # --- recomposition (no ONNX run) ---
    from PySide6.QtCore import QTimer
    _debounce = QTimer(); _debounce.setSingleShot(True); _debounce.setInterval(200)
    def schedule_update():
        _debounce.stop(); _debounce.start()
    def do_update():
        if current_rgb is None or current_alpha is None: return
        p = get_params()
        # Apply brush mask from preview to remove regions (set alpha=0 where painted)
        a_eff = current_alpha.copy()
        try:
            h, w = a_eff.shape[:2]
            rm, km = None, None
            try:
                rm, km = preview.export_masks_to_image_size(w, h)
            except Exception:
                rm = preview.export_mask_to_image_size(w, h)
                km = None
            # Ensure masks match image size
            rm = _ensure_mask_hw(rm, h, w)
            km = _ensure_mask_hw(km, h, w)
            if rm is not None:
                a_eff = a_eff * (1.0 - (rm > 0.01).astype(a_eff.dtype))
            if km is not None:
                a_eff = np.clip(a_eff + (km > 0.01).astype(a_eff.dtype) * (1.0 - a_eff), 0.0, 1.0)
        except Exception:
            pass
        try:
            pim = remove_background_preview(
                current_rgb, a_eff, p["mode"], p["threshold"], p["feather"], p["spill"],
                p["invert"], p["auto_level"], p.get("bias", 50),
                p["repl_mode"], p["repl_color"], p["repl_blur"], p["repl_image"],
                p["drop_shadow"], p["shadow_alpha"], p["shadow_blur"], p["shadow_offx"], p["shadow_offy"], p["anti_halo"]
            )
            update_preview_pix(pim)
        except Exception as e:
            try: QMessageBox.critical(panel, "Preview error", str(e))
            except Exception: print("Preview error:", e)
    _debounce.timeout.connect(do_update)
    # --- Background inference worker to keep UI responsive ---
    class _InferWorker(QThread):
        done = Signal(object, object)  # (alpha ndarray or None, error str or None)
        def __init__(self, model_path: Path, img_rgb: np.ndarray, engine_id: str):
            super().__init__()
            self._model_path = model_path
            self._rgb = img_rgb
            self._engine = engine_id
        def run(self):
            try:
                alpha = _infer_onnx_alpha(self._model_path, self._rgb, self._engine)
                self.done.emit(alpha, None)
            except Exception as e:
                self.done.emit(None, str(e))

    _worker_ref = {'w': None}

    def _set_busy(is_busy: bool):
        try:
            btn_recompute.setEnabled(not is_busy)
            btn_use_current.setEnabled(not is_busy)
            btn_load.setEnabled(not is_busy)
        except Exception:
            pass

    def _start_infer(model_path: Path, reset_params: bool):
        if current_rgb is None or model_path is None:
            return
        w = _worker_ref.get('w')
        try:
            if w is not None and w.isRunning():
                return
        except Exception:
            pass
        _set_busy(True)
        eng_id = {0:"auto",1:"modnet",2:"birefnet"}[cmb_engine.currentIndex()]
        worker = _InferWorker(model_path, current_rgb, eng_id)
        _worker_ref['w'] = worker
        def _on_done(alpha, err):
            nonlocal current_alpha
            _set_busy(False)
            if err:
                try: QMessageBox.critical(panel, "ONNX error", str(err))
                except Exception: print("ONNX error:", err)
                return
            if alpha is not None:
                current_alpha = alpha
                if reset_params:
                    try:
                        undo_stack.clear(); set_params(defaults())
                    except Exception:
                        pass
                schedule_update()
        try:
            worker.done.connect(_on_done)
        except Exception:
            pass
        worker.start()


    # Mask change triggers a preview update
    try:
        preview.maskChanged.connect(schedule_update)
    except Exception:
        pass

    # Hook updates
    for w in [cmb_mode, cmb_repl, s_blurbg, s_thresh, s_feath, s_edge, cb_spill, cb_auto, cb_inv, cb_ah, cb_shadow, s_salpha, s_sblur, s_sox, s_soy]:
        try:
            if hasattr(w, "valueChanged"): w.valueChanged.connect(schedule_update)
            if hasattr(w, "currentIndexChanged"): w.currentIndexChanged.connect(schedule_update)
            if hasattr(w, "toggled"): w.toggled.connect(schedule_update)
        except Exception: pass

    # --- Use current (from app) ---
    def on_use_current():
        nonlocal current_rgb, current_alpha, current_path
        media_path, time_s = _current_media_from_app(pane)
        if media_path is None:
            try:
                QMessageBox.information(panel, "No image", "Open an image in the app (left panel) first.")
                return
            except Exception:
                return
        current_path = media_path
        try:
            img = load_image_or_frame(str(media_path), float(time_s))
        except Exception as e:
            try:
                QMessageBox.critical(panel, "Load error", str(e))
                return
            except Exception:
                return
        current_rgb = _np_from_pil(img)
        h, w = current_rgb.shape[:2]
        try:
            import numpy as _np_local
            current_alpha = _np_local.ones((h, w), dtype=_np_local.float32)
        except Exception:
            current_alpha = None
        try:
            preview.setImageSize(w, h)
            preview.clearMask()
        except Exception:
            pass
        try:
            update_preview_pix(_pil_from_np(current_rgb).convert("RGBA"))
        except Exception:
            try:
                pm = QPixmap.fromImage(qimage_from_pil(_pil_from_np(current_rgb)))
                preview.setPixmap(pm)
            except Exception:
                pass
    
    btn_use_current.clicked.connect(on_use_current)

    # --- Load external ---
    def on_load_external():
        nonlocal current_rgb, current_alpha, current_path
        try:
            if cmb_source.currentIndex()==0:
                files, _ = QFileDialog.getOpenFileNames(panel, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
            else:
                files, _ = QFileDialog.getOpenFileNames(panel, "Choose video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm)")
        except Exception:
            files = []
        if not files:
            return
        current_path = Path(files[0])
        try:
            img = load_image_or_frame(str(current_path), float(t_seek.value()))
        except Exception as e:
            try:
                QMessageBox.critical(panel, "Load error", str(e))
                return
            except Exception:
                return
        current_rgb = _np_from_pil(img)
        h, w = current_rgb.shape[:2]
        try:
            import numpy as _np_local
            current_alpha = _np_local.ones((h, w), dtype=_np_local.float32)
        except Exception:
            current_alpha = None
        try:
            preview.setImageSize(w, h)
            preview.clearMask()
        except Exception:
            pass
        try:
            update_preview_pix(_pil_from_np(current_rgb).convert("RGBA"))
        except Exception:
            try:
                pm = QPixmap.fromImage(qimage_from_pil(_pil_from_np(current_rgb)))
                preview.setPixmap(pm)
            except Exception:
                pass
    
    btn_load.clicked.connect(on_load_external)

    # Drag & drop from preview
    def _on_drop_path(pathstr: str):
        nonlocal current_rgb, current_alpha, current_path
        try:
            pth = Path(pathstr)
            current_path = pth
            ext = pth.suffix.lower()
            is_video = ext in ['.mp4','.mov','.mkv','.avi','.webm']
            img = load_image_or_frame(str(current_path), float(t_seek.value()) if is_video else 0.0)
            current_rgb = _np_from_pil(img)
            h, w = current_rgb.shape[:2]
            import numpy as _np_local
            current_alpha = _np_local.ones((h, w), dtype=_np_local.float32)
            preview.setImageSize(w, h)
            preview.clearMask()
            update_preview_pix(_pil_from_np(current_rgb).convert("RGBA"))
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(panel, "Drop error", str(e))
            except Exception:
                pass
    
    try:
        preview.fileDropped.connect(_on_drop_path)
    except Exception:
        pass
    # --- Recompute, Reset, Undo, Save ---
    def on_recompute():
        nonlocal current_alpha
        if current_rgb is None:
            try: QMessageBox.information(panel, "Info", "Load or use current image first."); return
            except Exception: return
        eng, model = _pick_engine({0:"auto",1:"modnet",2:"birefnet"}[cmb_engine.currentIndex()])
        if model is None:
            try: QMessageBox.critical(panel, "Model", "No ONNX model found in models/."); return
            except Exception: return
        try:
            _start_infer(model, reset_params=False)
        except Exception as e:
            try: QMessageBox.critical(panel, "ONNX error", str(e)); return
            except Exception: return
    btn_recompute.clicked.connect(on_recompute)

    def on_reset():
        if current_rgb is None: return
        set_params(defaults()); preview.clearMask(); schedule_update()
    btn_reset.clicked.connect(on_reset)

    def on_undo():
        if not undo_stack: return
        d = undo_stack.pop(); set_params(d); schedule_update()
    btn_undo.clicked.connect(on_undo)

    def snapshot():
        d = get_params().copy(); undo_stack.append(d)
        if len(undo_stack) > 20: undo_stack.pop(0)
    for w in [cmb_mode, cmb_repl, s_blurbg, s_thresh, s_feath, s_edge, cb_spill, cb_auto, cb_inv, cb_ah, cb_shadow, s_salpha, s_sblur, s_sox, s_soy]:
        try:
            if hasattr(w, "editingFinished"): w.editingFinished.connect(snapshot)
        except Exception: pass
        try:
            if hasattr(w, "currentIndexChanged"): w.currentIndexChanged.connect(snapshot)
            if hasattr(w, "toggled"): w.toggled.connect(snapshot)
        except Exception: pass

    def on_save():
        if current_rgb is None or current_alpha is None:
            try: QMessageBox.information(panel, "Info", "Nothing to save. Load or use current image first."); return
            except Exception: return
        p = get_params()
        try:
            pim = remove_background_preview(
                current_rgb, current_alpha, p["mode"], p["threshold"], p["feather"], p["spill"],
                p["invert"], p["auto_level"], p.get("bias",50),
                p["repl_mode"], p["repl_color"], p["repl_blur"], p["repl_image"],
                p["drop_shadow"], p["shadow_alpha"], p["shadow_blur"], p["shadow_offx"], p["shadow_offy"], p["anti_halo"]
            )
            out_dir = Path(le_out.text().strip()) if le_out.text().strip() else _get_out_dir_pref()
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = current_path.stem if current_path else "preview"
            suffix = {"transparent":"cutout","color":"colorbg","blur":"blurbg","image":"imagebg"}[p["repl_mode"]]
            if p["mode"] == "alpha_only": suffix = "alpha"
            elif p["mode"] == "keep_bg": suffix = "bgonly"
            out = out_dir / f"{stem}_{suffix}.png"
            pim.save(out)
            try: QMessageBox.information(panel, "Saved", f"Saved:\n{out}")
            except Exception: print("Saved:", out)
        except Exception as e:
            try: QMessageBox.critical(panel, "Save error", str(e))
            except Exception: print("Save error:", e)
    btn_save.clicked.connect(on_save)
    # --- Inpaint (SD 1.5) ---
    # --- Inpaint worker to keep UI responsive ---
    class _InpaintWorker(QThread):
        done = Signal(object, object)  # (PIL.Image.Image or None, error str or None)
        def __init__(self, img_rgb, rm01, prompt, negative, steps, guidance, strength, seed, model_path):
            super().__init__()
            self._img_rgb = img_rgb
            self._rm01 = rm01
            self._prompt = prompt
            self._negative = negative
            self._steps = int(steps)
            self._guidance = float(guidance)
            self._strength = float(strength)
            self._seed = int(seed)
            self._model_path = model_path
        def run(self):
            try:
                out_img = inpaint_sd15(
                    self._img_rgb,
                    remove_mask01=self._rm01,
                    prompt=self._prompt,
                    negative_prompt=self._negative,
                    steps=self._steps,
                    guidance=self._guidance,
                    strength=self._strength,
                    seed=self._seed,
                    model_path=self._model_path,
                )
                self.done.emit(out_img, None)
            except Exception as e:
                self.done.emit(None, str(e))

    _inpaint_worker_ref = {'w': None}

    
    _inpaint_queue = {'pending': False}
    def on_inpaint_sd15():


        # Mini-queue: coalesce rapid inpaint requests so only one runs at a time
        try:
            w = _inpaint_worker_ref.get('w') if isinstance(_inpaint_worker_ref, dict) else None
            if w is not None:
                try:
                    is_running = bool(getattr(w, "isRunning", lambda: False)())
                except Exception:
                    is_running = False
            else:
                is_running = False
            if is_running:
                _inpaint_queue['pending'] = True
                return
        except Exception:
            pass
    
        nonlocal current_rgb, current_alpha, current_path
        if current_rgb is None:
            try:
                QMessageBox.information(panel, "Info", "Load or use current image first."); return
            except Exception:
                return
        if not _sd15_inpaint_available():
            try:
                QMessageBox.critical(panel, "SD Inpaint", "Missing dependencies. Please install torch, diffusers, transformers, accelerate, safetensors."); return
            except Exception:
                return
        try:
            h, w = current_rgb.shape[:2]
            # Export remove/keep masks at image size
            rm, km = None, None
            try:
                rm, km = preview.export_masks_to_image_size(w, h)
            except Exception:
                try:
                    rm = preview.export_mask_to_image_size(w, h)
                except Exception:
                    rm = None
                km = None
            rm = _ensure_mask_hw(rm, h, w)
            km = _ensure_mask_hw(km, h, w)
            if rm is None or (rm <= 0.01).sum() == (rm.size):
                if current_alpha is None:
                    try: QMessageBox.information(panel, "Inpaint", "No masked area to fill."); return
                    except Exception: return
                else:
                    import numpy as _np_local
                    rm = (1.0 - _np_local.clip(current_alpha, 0.0, 1.0)).astype(_np_local.float32)
            if km is not None:
                import numpy as _np_local
                rm = _np_local.clip(rm - (km > 0.01).astype(_np_local.float32), 0.0, 1.0)

            # Optionally dilate the remove mask to give the model room
            try:
                _pad_px = int(s_sd_pad.value())
            except Exception:
                _pad_px = 0
            if _pad_px > 0:
                try:
                    rm = (_morph_shift_u8((np.clip(rm,0.0,1.0)*255).astype(np.uint8), int(_pad_px)) / 255.0).astype(np.float32)
                except Exception:
                    pass
            # Gather params
            _prompt  = le_sd_prompt.text().strip()
            _neg     = le_sd_neg.text().strip()
            _steps   = int(s_sd_steps.value())
            _guid    = float(ds_sd_guid.value())
            _stren   = float(ds_sd_str.value())
            _seed    = int(s_sd_seed.value())
            _mpath   = _resolve_inpaint_model_path(le_sd_model.text().strip())

            # Disable UI, start worker
            try: btn_sd_inpaint.setEnabled(False)
            except Exception: pass

            wkr = _InpaintWorker(current_rgb.copy(), rm, _prompt, _neg, _steps, _guid, _stren, _seed, _mpath)
            _inpaint_worker_ref['w'] = wkr
            def _on_inpaint_done(img, err):
                try: btn_sd_inpaint.setEnabled(True)
                except Exception: pass
                if err is not None:
                    try: QMessageBox.critical(panel, "Inpaint error", str(err))
                    except Exception: print("Inpaint error:", err)
                    return
                if img is None:
                    return
                try:
                    out_img = img
                    import numpy as _np_local
                    # Update current image; paste only where masked when 'freeze unmasked' is on
                    crgb = _np_from_pil(out_img.convert("RGB"))
                    hh, ww = crgb.shape[:2]
                    nonlocal current_rgb, current_alpha
                    try:
                        _freeze = bool(chk_freeze_unmasked.isChecked())
                    except Exception:
                        _freeze = bool(getattr(Preview, '_ui_flags', {}).get('freeze_unmasked', True))
                    if _freeze and rm is not None:
                        m = np.clip(rm, 0.0, 1.0).astype(np.float32)
                        if m.shape != (hh, ww):
                            try:
                                if cv2 is not None:
                                    m = cv2.resize(m, (ww, hh), interpolation=cv2.INTER_LINEAR)
                                else:
                                    from PIL import Image as _PIL
                                    m = np.array(_PIL.fromarray((m*255).astype(np.uint8)).resize((ww, hh), _PIL.BILINEAR)) / 255.0
                            except Exception:
                                pass
                        m3 = np.dstack([m, m, m]).astype(np.float32)
                        base = current_rgb.astype(np.float32)
                        gen  = crgb.astype(np.float32)
                        crgb = np.clip(gen * m3 + base * (1.0 - m3), 0, 255).astype(np.uint8)
                    # Assign back, keep alpha as-is
                    current_rgb = crgb
                    preview.setImageSize(ww, hh)
                    preview.clearMask()
                    update_preview_pix(out_img.convert("RGBA"))
                    schedule_update()

                    # If more inpaint requests arrived while this one was running,
                    # trigger exactly one more run with the latest mask/settings.
                    try:
                        from PySide6.QtCore import QTimer as _QTimer
                        if _inpaint_queue.get('pending'):
                            _inpaint_queue['pending'] = False
                            _QTimer.singleShot(0, on_inpaint_sd15)
                    except Exception:
                        pass
                except Exception as e:
                    try: QMessageBox.critical(panel, "Inpaint error", str(e))
                    except Exception: print("Inpaint error:", e)

            wkr.done.connect(_on_inpaint_done)
            wkr.start()
        except Exception as e:
            try: QMessageBox.critical(panel, "Inpaint error", str(e))
            except Exception: print("Inpaint error:", e)
    
    try:
        btn_sd_inpaint.clicked.connect(on_inpaint_sd15)
    except Exception:
        pass



    # Hand layout back to section
    # section_widget.setContentLayout(vbox) was called above.
    return