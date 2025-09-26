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
class SpectrumSpline(BaseVisualizer):
    display_name = "Spectrum Spline"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.fillRect(r, QBrush(QColor(6, 8, 16)))
        if not bands:
            return
        n = len(bands)
        lvl = bass_level(bands, rms)
        lg = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
        lg.setColorAt(0.0, QColor(30,50,120,160))
        lg.setColorAt(1.0, QColor(180,80,220,120))
        path_pts = []
        step = max(3, w//320*4)
        for i in range(0, w, step):
            bi = int(i * n / max(1, w))
            v = max(0.0, min(1.0, bands[min(n-1, bi)]))
            x = r.left() + i
            y = r.bottom() - v*h*0.75*(0.7+0.6*lvl)
            path_pts.append((x,y))
        if path_pts:
            pp = QPainterPath(QPointF(path_pts[0][0], r.bottom()))
            for (x,y) in path_pts:
                pp.lineTo(x,y)
            pp.lineTo(path_pts[-1][0], r.bottom())
            pp.closeSubpath()
            p.setPen(QPen(QColor(255,255,255,25), 1))
            p.fillPath(pp, QBrush(lg))
        pen = QPen(QColor(255, 200, 240, 230), 2 + int(2*lvl))
        p.setPen(pen)
        prev = None
        for (x,y) in path_pts:
            if prev: p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
            prev = (x,y)
