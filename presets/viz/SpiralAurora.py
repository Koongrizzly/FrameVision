from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QConicalGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Shared smoothing state
_s_bass = 0.0

def bass_level(bands, rms):
    global _s_bass
    if bands:
        n = len(bands)
        lo = max(1, n//6)  # focus on lowest ~1/6th
        bass = sum(bands[:lo]) / lo
    else:
        bass = 0.0
    # Combine bass + rms, clamp
    lvl = 0.25 + 0.75*min(1.0, 0.9*bass + 0.5*rms)
    # Smooth
    _s_bass = 0.85*_s_bass + 0.15*lvl
    return _s_bass

@register_visualizer
class SpiralAurora(BaseVisualizer):
    display_name = "Spiral Aurora"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        cx, cy = r.center().x(), r.center().y()
        turns = 6
        steps = 800
        lvl = bass_level(bands, rms)
        px, py = None, None
        for i in range(steps):
            frac = i/steps
            ang = 2*pi*turns*frac + t*0.3
            rad = frac * min(w,h)*0.5 * (0.7 + 0.6*lvl)
            x = cx + rad*cos(ang)
            y = cy + rad*sin(ang)
            hue = int((t*30 + 360*frac) % 360)
            alpha = int(120*(1-frac))
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, alpha), 2))
            if px is not None:
                p.drawLine(QPointF(px,py), QPointF(x,y))
            px, py = x,y
