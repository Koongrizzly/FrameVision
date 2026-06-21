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
class AuroraRibbon(BaseVisualizer):
    display_name = "Aurora Ribbon"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        # Background
        bg = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
        bg.setColorAt(0.0, QColor(10, 15, 30))
        bg.setColorAt(1.0, QColor(25, 30, 55))
        p.fillRect(r, QBrush(bg))

        lvl = bass_level(bands, rms)
        layers = 5
        for k in range(layers):
            phase = t*0.6 + k*0.8
            amp = (0.10 + 0.08*k) * h * (0.6 + 0.8*lvl)
            pen = QPen(QColor(80+30*k, 200-20*k, 255, 160), 2 + k*0.6)
            p.setPen(pen)
            prev = None
            step = max(4, w//300*2)  # adaptive density
            for i in range(0, w, step):
                # sample spectrum smoothly
                v = 0.0
                if bands:
                    idx = int(i / max(1, w) * len(bands))
                    idx = max(0, min(len(bands)-1, idx))
                    v = bands[idx]
                x = r.left() + i
                y = r.center().y() + amp * sin(i*0.012 + phase + v*1.2)
                if prev:
                    p.drawLine(QPointF(prev[0], prev[1]), QPointF(x, y))
                prev = (x, y)
