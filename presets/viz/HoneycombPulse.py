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
class HoneycombPulse(BaseVisualizer):
    display_name = "Honeycomb Pulse"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.fillRect(r, QBrush(QColor(12,14,26)))
        size = max(16, int(min(w,h)*0.05))
        hx = size * 3/2
        hy = size * sqrt(3) / 2
        rows = int(h / hy) + 2
        cols = int(w / hx) + 3
        lvl = bass_level(bands, rms)
        nb = len(bands) if bands else 1
        for row in range(rows):
            for col in range(cols):
                cx = r.left() + col*hx + (hx/2 if row%2 else 0)
                cy = r.top() + row*hy
                idx = (row*cols + col) % nb
                v = bands[idx] if bands else 0.0
                pulse = 0.4 + 0.6*lvl + 0.3*v
                hue = int((idx*7 + t*30) % 360)
                colr = QColor.fromHsv(hue, 220, min(255, int(140 + 100*pulse)), 200)
                p.setPen(QPen(colr, 2))
                pts = []
                for k in range(6):
                    ang = k*pi/3
                    pts.append(QPointF(cx + size*cos(ang), cy + size*sin(ang)))
                for k in range(6):
                    p.drawLine(pts[k], pts[(k+1)%6])
