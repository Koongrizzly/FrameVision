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
class VortexFlow(BaseVisualizer):
    display_name = "Vortex Flow"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()
        lvl = bass_level(bands, rms)
        arms = 5
        steps = 650
        for a in range(arms):
            hue = int((t*25 + a*60) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 250, 170), 2))
            px, py = None, None
            for i in range(steps):
                f = i/steps
                ang = 2*pi*(a/arms) + 4.5*pi*f + t*0.25
                rad = f*min(w,h)*0.48*(0.65 + 0.7*lvl)
                x = cx + rad*cos(ang)
                y = cy + rad*sin(ang)
                if px is not None:
                    p.drawLine(QPointF(px,py), QPointF(x,y))
                px,py = x,y
