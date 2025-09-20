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
class BezierWaves(BaseVisualizer):
    display_name = "Bézier Waves"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        p.fillRect(r, QBrush(QColor(5,6,12)))
        lvl = bass_level(bands, rms)
        rows = 8
        for iy in range(rows):
            hue = int((t*25 + iy*35) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 210, 245, 200), 2))
            y = r.top() + (iy+0.5)*h/rows
            x1 = r.left(); x2 = r.right()
            ctrlx = (x1+x2)/2
            amp = (0.04*h)*(1+2*lvl)
            v = 0.0
            if bands:
                v = bands[iy % len(bands)]
            c1 = QPointF(ctrlx, y - amp*(sin(t+iy*0.6)+ v*2))
            c2 = QPointF(ctrlx, y + amp*(cos(t+iy*0.5)+ v*2))
            path = QPainterPath(QPointF(x1,y))
            path.cubicTo(c1, c2, QPointF(x2,y))
            p.drawPath(path)
