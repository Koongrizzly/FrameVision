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
class OceanSurf(BaseVisualizer):
    display_name = "Ocean Surf"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        # dusk gradient
        bg = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
        bg.setColorAt(0.0, QColor(8, 14, 28))
        bg.setColorAt(1.0, QColor(20, 30, 60))
        p.fillRect(r, QBrush(bg))

        lvl = bass_level(bands, rms)
        layers = 6
        for k in range(layers):
            phase = t*0.45 + k*0.5
            amp = (0.05 + 0.03*k) * h * (0.7 + 0.7*lvl)
            hue = int((200 + k*18 + t*10) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 180, 240, 200), 2))
            prev = None
            step = max(4, w//240*4)
            for i in range(0, w+step, step):
                x = r.left() + i
                # sample mid bands for detail
                v = 0.0
                if bands:
                    idx = int((i/w) * len(bands))
                    idx = max(0, min(len(bands)-1, idx))
                    v = bands[idx]
                y = r.center().y() + amp*sin(i*0.012 + phase) + (h*0.02)*sin(i*0.05 + t*0.7) - v*14
                if prev:
                    p.drawLine(QPointF(prev[0], prev[1]), QPointF(x,y))
                prev = (x,y)
