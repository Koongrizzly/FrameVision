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
class RibbonTendrils(BaseVisualizer):
    display_name = "Ribbon Tendrils"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.fillRect(r, QBrush(QColor(8,8,16)))
        lvl = bass_level(bands, rms)
        strands = 7
        for s in range(strands):
            phase = t*0.5 + s*0.6
            hue = int((s*40 + t*25)%360)
            p.setPen(QPen(QColor.fromHsv(hue, 200, 240, 180), 2))
            prev = None
            step = max(6, w//220*8)
            for i in range(0, w, step):
                v = 0.0
                if bands:
                    idx = int(i/ max(1, w) * len(bands))
                    idx = max(0, min(len(bands)-1, idx))
                    v = bands[idx]
                y = r.center().y() + (h*0.22 + s*4) * sin(i*0.01 + phase + v*1.0 + 1.2*lvl)
                if prev:
                    p.drawLine(QPointF(prev[0],prev[1]), QPointF(i, y))
                prev = (i, y)
