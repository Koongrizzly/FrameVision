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
class PetalBloom(BaseVisualizer):
    display_name = "Petal Bloom"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()
        lvl = bass_level(bands, rms)
        petals = 7
        loops = 3
        R = min(w,h)*0.38*(0.75+0.5*lvl)
        N = 800
        for j in range(loops):
            hue = int((t*30 + j*40) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 200, 255, 180), 2))
            prev = None
            for i in range(N):
                th = 2*pi*i/N
                rads = R * (0.6 + 0.4*sin(petals*th + t*0.9 + j*0.6))
                x = cx + rads*cos(th)
                y = cy + rads*sin(th)
                if prev:
                    p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
                prev = (x,y)
