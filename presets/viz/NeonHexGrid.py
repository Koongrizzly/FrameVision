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
class NeonHexGrid(BaseVisualizer):
    display_name = "Neon Hex Grid"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        p.fillRect(r, QBrush(QColor(8,9,16)))
        lvl = bass_level(bands, rms)
        size = max(14, int(min(w,h)*0.045))
        hx = size*3/2
        hy = size*sqrt(3)/2
        rows = int(h/hy) + 2
        cols = int(w/hx) + 3
        for row in range(rows):
            for col in range(cols):
                cx = r.left() + col*hx + (hx/2 if row%2 else 0)
                cy = r.top()  + row*hy
                hue = int((row*20 + col*12 + t*30) % 360)
                p.setPen(QPen(QColor.fromHsv(hue, 220, 240, 180), 2))
                pts = []
                for k in range(6):
                    ang = k*pi/3 + 0.25*sin(t*0.8 + (row+col)*0.15) * (0.5+0.5*lvl)
                    pts.append(QPointF(cx + size*cos(ang), cy + size*sin(ang)))
                for k in range(6):
                    p.drawLine(pts[k], pts[(k+1)%6])
