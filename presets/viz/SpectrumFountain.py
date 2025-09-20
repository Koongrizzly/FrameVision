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
class SpectrumFountain(BaseVisualizer):
    display_name = "Spectrum Fountain"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        p.fillRect(r, QBrush(QColor(6,7,14)))
        lvl = bass_level(bands, rms)
        n = len(bands) if bands else 0
        cols = max(48, w//12)
        for i in range(0, cols):
            x = r.left() + i*(w/cols)
            v = 0.0
            if n:
                bi = int(i/cols * n)
                bi = max(0, min(n-1, bi))
                v = bands[bi]
            hgt = (0.12 + 0.6*v)*(0.6+0.7*lvl)*h
            hue = int((t*35 + i*5) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 245, 200), 3))
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom()-hgt))
