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
class SunburstBeat(BaseVisualizer):
    display_name = "Sunburst Beat"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        cx, cy = r.center().x(), r.center().y()
        rays = 120
        lvl = bass_level(bands, rms)
        base = 0.55 + 0.9*lvl
        for i in range(rays):
            ang = 2*pi*i/rays + t*0.1
            v = 0.0
            if bands:
                v = bands[i % len(bands)]
            length = min(w,h)*0.45*(base + 0.4*v)
            hue = int((i*3 + t*40) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 240, 160), 2))
            p.drawLine(QPointF(cx, cy), QPointF(cx + length*cos(ang), cy + length*sin(ang)))
        g = QRadialGradient(QPointF(cx,cy), min(w,h)*0.25)
        g.setColorAt(0.0, QColor(255,255,255,60))
        g.setColorAt(1.0, QColor(0,0,0,0))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(255,255,255,20), 1))
        p.drawEllipse(QPointF(cx,cy), min(w,h)*0.25*(0.7+0.5*lvl), min(w,h)*0.25*(0.7+0.5*lvl))
