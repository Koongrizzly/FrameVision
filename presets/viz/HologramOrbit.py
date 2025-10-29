import os, glob
from math import sin, cos, pi
from random import random, choice, shuffle
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPixmap, QRadialGradient
)
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# persistent state
_logo_pixmaps = []   # list[QPixmap]
_orbs = []           # list of dicts {angle,speed_base,pix_i,trail: list[(x,y,depth)]}
_last_t = None
_prev_spec = []
_env_lo = 0.0
_env_mid = 0.0
_env_hi = 0.0

def _load_logos():
    """Load all logo_*.jpg/png/etc. once and cache them for this run."""
    global _logo_pixmaps
    if _logo_pixmaps:
        return _logo_pixmaps

    base_dir = os.path.dirname(os.path.abspath(__file__))
    startup_dir = os.path.normpath(os.path.join(base_dir, "..", "startup"))

    patterns = [
        "logo_*.jpg", "logo_*.png", "logo_*.jpeg",
        "logo_*.JPG", "logo_*.PNG", "logo_*.JPEG",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(startup_dir, pat)))

    # shuffle so each app start gives different ordering / assignment
    if files:
        shuffle(files)

    for pth in files:
        pm = QPixmap(pth)
        if not pm.isNull():
            _logo_pixmaps.append(pm)

    # fallback to older fixed names if nothing matched
    if not _logo_pixmaps:
        names = [
            "logo_1.jpg", "logo_1.png",
            "logo_2.jpg", "logo_2.png",
        ]
        for nm in names:
            pth = os.path.join(startup_dir, nm)
            if os.path.exists(pth):
                pm = QPixmap(pth)
                if not pm.isNull():
                    _logo_pixmaps.append(pm)

    return _logo_pixmaps

def _init_orbs(count=10):
    """Create orbiters once. Each orb keeps its own trail."""
    global _orbs
    if _orbs:
        return
    logos = _load_logos()
    if not logos:
        # create placeholder "ghost" orbs anyway, they won't draw images
        logos = [None]
    for i in range(count):
        _orbs.append({
            "angle": random() * 2 * pi,
            "speed_base": 0.3 + random() * 1.0,
            "pix_i": i % len(logos),
            "trail": [],
        })

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
class HologramOrbit(BaseVisualizer):
    display_name = "Hologram Orbit"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t, _env_lo, _env_mid, _env_hi

        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        if _last_t is None:
            _last_t = t
        dt = max(0.0, min(0.05, t - _last_t))
        _last_t = t

        logos = _load_logos()
        _init_orbs(10)

        lo, mid, hi = _split(bands)
        fx = _flux(bands)

        _env_lo = _env_step(_env_lo, lo + 0.4 * rms, up=0.45, down=0.22)
        _env_mid = _env_step(_env_mid, mid + 0.4 * fx, up=0.4, down=0.20)
        _env_hi = _env_step(_env_hi, hi + 1.5 * fx, up=0.6, down=0.30)

        cx, cy = w * 0.5, h * 0.5

        # background: dark radial glow
        bg_grad = QRadialGradient(QPointF(cx, cy), max(w, h) * 0.8)
        bg_grad.setColorAt(0.0, QColor(5, 8, 16))
        bg_grad.setColorAt(1.0, QColor(0, 0, 0))
        p.fillRect(r, QBrush(bg_grad))

        # subtle HUD crosshair + orbit ring
        p.setRenderHint(QPainter.Antialiasing, True)
        hud_col = QColor(0, 200, 255, 40)
        p.setPen(QPen(hud_col, 1))
        p.setBrush(Qt.NoBrush)
        p.drawLine(cx - 40, cy, cx + 40, cy)
        p.drawLine(cx, cy - 40, cx, cy + 40)

        R = min(w, h) * 0.35 * (1.0 + 0.10 * _env_lo)
        orbit_col = QColor(0, 200, 255, 20)
        p.setPen(QPen(orbit_col, 1))
        p.drawEllipse(QPointF(cx, cy), R, R * 0.4)

        # update orbs physics
        orbs_draw = []
        speed_scale = 0.5 + 2.0 * _env_mid
        for orb in _orbs:
            orb["angle"] += orb["speed_base"] * speed_scale * dt

            ang = orb["angle"]
            ox = cx + cos(ang) * R
            oy = cy + sin(ang) * R * 0.4
            depth = 0.5 + 0.5 * sin(ang)  # front when sin(ang) > 0

            orb["pos"] = (ox, oy)
            orb["depth"] = depth

            trail = orb["trail"]
            trail.append((ox, oy, depth))
            if len(trail) > 10:
                del trail[0]

            orbs_draw.append(orb)

        # sort so farther orbs draw first
        orbs_draw.sort(key=lambda o: o["depth"])

        hue = int((180 + 120 * _env_hi) % 360)
        glow_col = QColor.fromHsv(hue, 200, 255, 110)

        for orb in orbs_draw:
            ox, oy = orb["pos"]
            depth = orb["depth"]
            pix_i = orb["pix_i"]

            # trail lines
            tr = orb["trail"]
            if len(tr) >= 2:
                for idx in range(1, len(tr)):
                    x0, y0, _ = tr[idx - 1]
                    x1, y1, _ = tr[idx]
                    a = int(100 * (idx / len(tr)))
                    pen = QPen(QColor(0, 255, 255, a), 2)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    p.drawLine(x0, y0, x1, y1)

            if not logos or pix_i >= len(logos):
                continue
            pm = logos[pix_i]
            if pm.isNull():
                continue

            base_size = min(w, h) * 0.18
            size_scale = (0.6 + 0.4 * depth) * (1.0 + 0.15 * _env_lo)
            draw_w = int(pm.width() * (base_size / max(1, pm.height())) * size_scale)
            draw_h = int(pm.height() * (base_size / max(1, pm.height())) * size_scale)
            if draw_w < 1 or draw_h < 1:
                continue
            scaled_pm = pm.scaled(draw_w, draw_h,
                                  Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)

            # glow at orb position
            p.save()
            p.translate(ox, oy)
            glow_r = max(draw_w, draw_h) * 0.8
            grad = QRadialGradient(QPointF(0, 0), glow_r)
            grad.setColorAt(0.0, glow_col)
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(0, 0), glow_r, glow_r)

            opacity = (0.3 + 0.7 * depth) * (0.6 + 0.4 * _env_hi)
            if opacity > 1.0:
                opacity = 1.0
            p.setOpacity(opacity)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawPixmap(-scaled_pm.width() / 2,
                         -scaled_pm.height() / 2,
                         scaled_pm)
            p.restore()
