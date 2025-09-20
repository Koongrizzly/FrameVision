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
class PlasmaBloom(BaseVisualizer):
    display_name = "Plasma Bloom"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.fillRect(r, QBrush(QColor(10, 8, 20)))
        lvl = bass_level(bands, rms)
        step = max(6, int(min(w,h)*0.02))
        for y in range(0, h, step):
            for x in range(0, w, step):
                u = x / max(1.0, w)
                v = y / max(1.0, h)
                val = (0.5+0.5*sin(6*u + t*0.6)) + (0.5+0.5*sin(6*v - t*0.8))
                val += (0.5+0.5*sin(6*(u+v) + t*0.4))
                if bands:
                    val += 0.8 * lvl
                val *= 0.25
                hv = int((val*360) % 360)
                col = QColor.fromHsv(hv, 200, 230, 210)
                p.fillRect(QRectF(x, y, step+1, step+1), QBrush(col))
