import os, glob
from math import sin
from random import random, choice
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPixmap
)
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_logo_pix = None
_last_t = None
_prev_spec = []
_env_lo = 0.0
_env_mid = 0.0
_env_hi = 0.0
_prev_lo_env = 0.0
_sweep_y = None

def _load_logo():
    """Load ONE random logo from presets/startup/ once per run."""
    global _logo_pix
    if _logo_pix is not None:
        return _logo_pix

    base_dir = os.path.dirname(os.path.abspath(__file__))
    startup_dir = os.path.normpath(os.path.join(base_dir, "..", "startup"))

    # look for any logo_*.jpg/png/jpeg etc.
    patterns = [
        "logo_*.jpg", "logo_*.png", "logo_*.jpeg",
        "logo_*.JPG", "logo_*.PNG", "logo_*.JPEG",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(startup_dir, pat)))

    if candidates:
        pick = choice(candidates)
        pm = QPixmap(pick)
        if not pm.isNull():
            _logo_pix = pm
            return _logo_pix

    # fallback original logic
    for nm in ["logo_2.jpg", "logo_2.png", "logo_1.jpg", "logo_1.png"]:
        pth = os.path.join(startup_dir, nm)
        if os.path.exists(pth):
            pm = QPixmap(pth)
            if not pm.isNull():
                _logo_pix = pm
                break
    return _logo_pix

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if (not _prev_spec) or (len(_prev_spec) != len(bands)):
        _prev_spec = [0.0] * len(bands)
    f = 0.0
    L = max(1, len(bands) - 1)
    for i, (x, px) in enumerate(zip(bands, _prev_spec)):
        d = x - px
        if d > 0:
            f += d * (0.35 + 0.65 * (i / L))
    _prev_spec = [0.82 * px + 0.18 * x for x, px in zip(bands, _prev_spec)]
    return f / max(1, len(bands))

def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n // 6)
    b = max(a + 1, n // 2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b - a))
    hi = sum(bands[b:]) / max(1, (n - b))
    return lo, mid, hi

def _env_step(env, target, up=0.34, down=0.14):
    if target > env:
        return (1 - up) * env + up * target
    else:
        return (1 - down) * env + down * target

@register_visualizer
class SliceGlitchWall(BaseVisualizer):
    display_name = "Slice Glitch Wall"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t, _env_lo, _env_mid, _env_hi, _prev_lo_env, _sweep_y

        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        if _last_t is None:
            _last_t = t
        dt = max(0.0, min(0.05, t - _last_t))
        _last_t = t

        logo_pm = _load_logo()

        lo, mid, hi = _split(bands)
        fx = _flux(bands)

        _env_lo = _env_step(_env_lo, lo + 0.4 * rms, up=0.5, down=0.25)
        _env_mid = _env_step(_env_mid, mid + 0.4 * fx, up=0.4, down=0.20)
        _env_hi = _env_step(_env_hi, hi + 1.5 * fx, up=0.6, down=0.30)

        # bass punch triggers sweep bar
        if _env_lo > _prev_lo_env + 0.12:
            _sweep_y = 0.0
        _prev_lo_env = _env_lo

        # update sweep bar
        if _sweep_y is not None:
            _sweep_y += dt * (h * 2.0)
            if _sweep_y > h:
                _sweep_y = None

        # fill background dark
        p.fillRect(r, QBrush(QColor(5, 5, 8)))

        cx, cy = w * 0.5, h * 0.5

        # global CRT-ish transform (rotate, zoom, shear)
        rot = 2.0 * sin(t * 0.5) * min(1.0, _env_mid * 1.2)
        scale_val = 1.0 + 0.03 * _env_lo
        shear_x = 0.02 * sin(t * 2.0)

        p.save()
        p.translate(cx, cy)
        p.rotate(rot)
        p.scale(scale_val, scale_val)
        p.shear(shear_x, 0.0)
        p.translate(-cx, -cy)

        # draw 3x3 tiled logo wall with horizontal slice glitching
        if logo_pm is not None and (not logo_pm.isNull()):
            tile_w = w / 3.0
            tile_h = h / 3.0
            # scale once for performance
            scaled_pm = logo_pm.scaled(int(tile_w), int(tile_h),
                                       Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)

            slices = 12
            slice_h = tile_h / slices

            for gr in range(3):
                for gc in range(3):
                    dest_x = gc * tile_w
                    dest_y = gr * tile_h
                    for si in range(slices):
                        y0 = dest_y + si * slice_h
                        h0 = slice_h + 1.0

                        jitter = 0.0
                        # hi-freq energy = glitch slices jumping sideways
                        if random() < 0.3 * _env_hi:
                            jitter = (random() * 2.0 - 1.0) * (8.0 + 24.0 * _env_hi)

                        p.save()
                        p.setClipRect(QRectF(dest_x, y0, tile_w, h0))
                        # draw one slice of the scaled logo at (dest_x+jitter,dest_y)
                        p.drawPixmap(QRectF(dest_x + jitter, dest_y,
                                            tile_w, tile_h),
                                     scaled_pm,
                                     QRectF(0, 0,
                                            scaled_pm.width(),
                                            scaled_pm.height()))
                        p.restore()

        p.restore()  # end of CRT transform

        # CRT scanlines overlay
        pen_scan = QPen(QColor(0, 0, 0, 60), 1)
        pen_scan.setCosmetic(True)
        p.setPen(pen_scan)
        for yy in range(0, h, 3):
            p.drawLine(0, yy, w, yy)

        # bass sweep bar
        if _sweep_y is not None:
            bar_h = h * 0.03
            fade_k = max(0.0, min(1.0, 1.0 - (_sweep_y / max(1.0, h))))
            alpha = int(180 * fade_k)
            hue = int((200 + 120 * _env_hi) % 360)
            col = QColor.fromHsv(hue, 80, 255, alpha)
            p.fillRect(QRectF(0, _sweep_y - bar_h * 0.5, w, bar_h),
                       QBrush(col))
