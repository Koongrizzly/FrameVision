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

_rng = Random(77)
comets = [(_rng.random(), _rng.random(), 2*pi*_rng.random(), 0.01 + 0.02*_rng.random()) for _ in range(60)]
@register_visualizer
class CometTrails(BaseVisualizer):
    display_name = "Comet Trails"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        p.fillRect(r, QBrush(QColor(5,6,12)))
        lvl = bass_level(bands, rms)
        base = 0.02 + 0.03*lvl
        for i,(ux, uy, ph, rad) in enumerate(comets):
            hue = int((t*50 + i*6) % 360)
            val = 190 + int(50*(0.5+0.5*sin(t*2 + ph)))
            steps = 5
            for k in range(steps, -1, -1):
                fade = (k+1)/(steps+1)
                x = r.left() + ((ux + base*t + rad*cos(t*1.5 + ph) - k*0.012) % 1.0) * w
                y = r.top()  + ((uy + base*0.8*t + rad*sin(t*1.5 + ph) - k*0.008) % 1.0) * h
                g = QRadialGradient(QPointF(x,y), 8 + 6*lvl)
                g.setColorAt(0.0, QColor.fromHsv(hue, 230, min(255,val), int(220*fade)))
                g.setColorAt(1.0, QColor.fromHsv(hue, 230, 0, 0))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(255,255,255,10), 1))
                p.drawEllipse(QPointF(x,y), 2.2+1.4*lvl, 2.2+1.4*lvl)
