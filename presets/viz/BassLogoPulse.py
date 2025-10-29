import os, glob
from math import sin
from random import random, choice
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPixmap, QRadialGradient
)
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# persistent module-level state
_logo_pix = None
_rings = []           # list of {"t0": float}
_last_t = None
_prev_spec = []
_env_lo = 0.0
_env_mid = 0.0
_env_hi = 0.0
_last_lo_env = 0.0
_shake_timer = 0.0

def _load_logo():
    """Load ONE random logo from presets/startup/ once, cache it for this run."""
    global _logo_pix
    if _logo_pix is not None:
        return _logo_pix

    base_dir = os.path.dirname(os.path.abspath(__file__))
    startup_dir = os.path.normpath(os.path.join(base_dir, "..", "startup"))

    # try all logo_*.jpg/png/jpeg etc. and pick a random one
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

    # fallback to older fixed names if no glob match
    for nm in ["logo_1.jpg", "logo_2.jpg", "logo_1.png", "logo_2.png"]:
        candidate = os.path.join(startup_dir, nm)
        if os.path.exists(candidate):
            pm = QPixmap(candidate)
            if not pm.isNull():
                _logo_pix = pm
                break
    return _logo_pix

def _flux(bands):
    """Rough 'onset energy' measure = how much spectrum jumped up this frame."""
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
    """Split spectrum -> (low, mid, high) energy buckets."""
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
    """Attack/decay smoothing envelope."""
    if target > env:
        return (1 - up) * env + up * target
    else:
        return (1 - down) * env + down * target

@register_visualizer
class BassLogoPulse(BaseVisualizer):
    display_name = "Bass Logo Pulse"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t, _env_lo, _env_mid, _env_hi, _last_lo_env, _shake_timer

        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        # time step
        if _last_t is None:
            _last_t = t
        dt = max(0.0, min(0.05, t - _last_t))
        _last_t = t

        # audio analysis
        lo, mid, hi = _split(bands)
        fx = _flux(bands)

        _env_lo = _env_step(_env_lo, lo + 0.4 * rms, up=0.5, down=0.25)
        _env_mid = _env_step(_env_mid, mid + 0.4 * fx, up=0.4, down=0.20)
        _env_hi = _env_step(_env_hi, hi + 1.5 * fx, up=0.6, down=0.30)

        # detect bass hit to spawn ring + shake camera
        if _env_lo > _last_lo_env + 0.12:
            _rings.append({ "t0": t })
            _shake_timer = 0.07
        _last_lo_env = _env_lo

        if _shake_timer > 0.0:
            _shake_timer = max(0.0, _shake_timer - dt)

        shake_amt = (_shake_timer / 0.07) * 6.0 if _shake_timer > 0 else 0.0
        shakex = (random() * 2 - 1) * shake_amt
        shakey = (random() * 2 - 1) * shake_amt

        # background
        p.fillRect(r, QBrush(QColor(5, 5, 8)))

        cx, cy = w * 0.5, h * 0.5
        logo_pm = _load_logo()
        if logo_pm is None or logo_pm.isNull():
            # fallback: simple pulsing circle
            rad = min(w, h) * (0.15 + 0.12 * _env_lo)
            glow_col = QColor.fromHsv(int((200 + 120 * _env_hi) % 360), 200, 255, 160)
            grad = QRadialGradient(QPointF(cx+shakex, cy+shakey), rad * 1.6)
            grad.setColorAt(0.0, glow_col)
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx+shakex, cy+shakey), rad * 1.4, rad * 1.4)
            p.setBrush(QBrush(QColor(255,255,255,220)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx+shakex, cy+shakey), rad, rad)
        else:
            # how big the logo should be on screen
            base_h = min(w, h) * 0.30
            scale_factor = base_h / max(1, logo_pm.height())
            scale_factor *= (1.0 + 0.20 * _env_lo)

            draw_w = int(logo_pm.width() * scale_factor)
            draw_h = int(logo_pm.height() * scale_factor)
            if draw_w < 1 or draw_h < 1:
                return
            scaled_logo = logo_pm.scaled(draw_w, draw_h,
                                         Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)

            # glow aura behind the logo (radial gradient ellipse)
            hue = int((200 + 120 * _env_hi) % 360)
            glow_rad = max(draw_w, draw_h) * (0.8 + 0.5 * _env_lo)
            grad = QRadialGradient(QPointF(cx+shakex, cy+shakey), glow_rad)
            grad.setColorAt(0.0, QColor.fromHsv(hue, 180, 220, 190))
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx+shakex, cy+shakey),
                          glow_rad * 0.9,
                          glow_rad * 0.9)

            # draw logo with slight rotation driven by mid
            rot_angle = 4.0 * sin(t * 1.2) * min(1.0, _env_mid * 1.2)
            p.save()
            p.translate(cx+shakex, cy+shakey)
            p.rotate(rot_angle)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.setOpacity(1.0)
            p.drawPixmap(-scaled_logo.width() / 2,
                         -scaled_logo.height() / 2,
                         scaled_logo)
            p.restore()

        # shockwave rings expanding from center on bass
        alive_rings = []
        for ring in _rings:
            age = t - ring["t0"]
            lifetime = 0.7
            if age < lifetime:
                alive_rings.append(ring)
                k = age / lifetime
                k = max(0.0, min(1.0, k))
                radius = (k ** 0.4) * min(w, h) * 0.8
                alpha = int(180 * (1.0 - k))
                thick = 2 + 4 * (1.0 - k)
                hue = int((200 + 120 * _env_hi) % 360)
                pen = QPen(QColor.fromHsv(hue, 120, 255, alpha), thick)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx+shakex, cy+shakey), radius, radius)
        _rings[:] = alive_rings
