from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QConicalGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Smoothed bass emphasis to keep idle motion while popping on bass
_s_bass = 0.0
def bass_level(bands, rms):
    global _s_bass
    if bands:
        n = len(bands)
        lo = max(1, n//6)
        bass = sum(bands[:lo]) / lo
    else:
        bass = 0.0
    lvl = 0.22 + 0.78 * min(1.0, 0.9*bass + 0.35*rms)
    _s_bass = 0.82*_s_bass + 0.18*lvl
    return _s_bass

@register_visualizer
class PolarAurora(BaseVisualizer):
    display_name = "Polar Aurora"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()
        lvl = bass_level(bands, rms)
        rays = 160
        n = len(bands) if bands else 1
        for i in range(rays):
            ang = 2*pi*i/rays
            v = bands[i % n] if bands else 0.0
            L = (0.28 + 0.4*v)*(0.7 + 0.8*lvl)*min(w,h)
            hue = int((t*30 + i*3) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 240, 170), 2))
            x = cx + L*cos(ang + 0.15*sin(t*0.8 + i*0.05))
            y = cy + L*sin(ang + 0.15*sin(t*0.8 + i*0.05))
            p.drawLine(QPointF(cx, cy), QPointF(x,y))
