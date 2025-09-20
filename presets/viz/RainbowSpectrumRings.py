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
class RainbowSpectrumRings(BaseVisualizer):
    display_name = "Rainbow Spectrum Rings"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        cx, cy = r.center().x(), r.center().y()
        p.fillRect(r, QBrush(QColor(8, 10, 18)))
        max_r = min(w, h) * 0.5
        rings = 10
        lvl = bass_level(bands, rms)
        for i in range(rings):
            frac = (i+1)/rings
            radius = frac * max_r * (0.7 + 0.6*lvl)
            hue = int((t*20 + i*36) % 360)
            val = int(180 + 60 * (0.5 + 0.5*sin(t*0.7 + i)))
            color = QColor.fromHsv(hue, 255, max(0, min(255, val)), 190)
            pen = QPen(color, 2 + frac*4)
            p.setPen(pen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)
        # center glow
        g = QConicalGradient(QPointF(cx, cy), (t*40) % 360)
        for j in range(6):
            g.setColorAt(j/6.0, QColor.fromHsv(int((j*60 + t*50)%360), 200, 220, 120))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(255,255,255,10), 1))
        p.drawEllipse(QPointF(cx, cy), max_r*0.25*(0.8+0.4*lvl), max_r*0.25*(0.8+0.4*lvl))
