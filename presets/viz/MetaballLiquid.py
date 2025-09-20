from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# Beat-reactive envelope (mid/high + flux-like via simple diff)
_prev = []
_env = 0.0
def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    acc = 0.0; cnt = 0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        acc += w*bands[i]; cnt += 1
    return acc/max(1, cnt)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = 0.0; cnt = 0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d * (0.3 + 0.7*((i-cut)/max(1, n-cut)))
        cnt += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, cnt)

def _env_drive(bands, rms):
    global _env
    target = 0.55*_midhi(bands) + 1.25*_flux(bands) + 0.20*rms
    target = target / (1.0 + 0.7*target)
    if target > _env:
        _env = 0.68*_env + 0.32*target
    else:
        _env = 0.92*_env + 0.08*target
    if _env < 0: _env = 0.0
    if _env > 1.0: _env = 1.0
    return _env

@register_visualizer
class MetaballLiquid(BaseVisualizer):
    display_name = "Metaball Liquid Metal"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = r.width(), r.height()
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()

        env = _env_drive(bands, rms)

        p.setPen(QPen(QColor(220, 230, 255, 80), 2))

        blobs = 10
        base_rad = min(w, h)*0.12*(0.9 + 0.6*env)
        ring = min(w, h)*0.22*(0.9 + 0.6*env)

        # Pulse hue with time + env
        hue_base = int((t*70 + 160*env) % 360)

        # big soft center glow (fake field blending)
        g0 = QRadialGradient(QPointF(cx, cy), ring*1.6)
        g0.setColorAt(0.0, QColor.fromHsv((hue_base+180)%360, 120, 200, 60))
        g0.setColorAt(1.0, QColor.fromHsv((hue_base+180)%360, 120, 0, 0))
        p.setBrush(QBrush(g0))
        p.drawEllipse(QPointF(cx, cy), ring*0.7, ring*0.7)

        # orbiting blobs with color gradients
        for i in range(blobs):
            ang = i/blobs*2*pi + t*(0.5 + 1.6*env)
            rad = base_rad * (0.85 + 0.30*sin(t*1.3 + i))
            x = cx + (ring)*cos(ang)
            y = cy + (ring*0.85)*sin(ang)

            hue = (hue_base + i*25) % 360
            g = QRadialGradient(QPointF(x, y), rad*2.0)
            g.setColorAt(0.0, QColor.fromHsv(hue, 220, 255, 220))
            g.setColorAt(1.0, QColor.fromHsv(hue, 220, 0, 0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,25), 1))
            p.drawEllipse(QPointF(x, y), rad, rad)
