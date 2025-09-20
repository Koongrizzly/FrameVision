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
class OrbitalRings(BaseVisualizer):
    display_name = "Orbital Rings"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()
        lvl = bass_level(bands, rms)
        rings = 4
        base = min(w,h)*0.18*(0.8+0.6*lvl)
        for i in range(rings):
            rad = base*(1 + 0.5*i)
            hue = int((t*25 + i*50) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 245, 160), 2))
            p.drawEllipse(QPointF(cx,cy), rad, rad*0.85)
            # moving beads on ring
            beads = 10
            for b in range(beads):
                ang = 2*pi*(b/beads) + t*(0.4 + 0.15*i)
                x = cx + rad*cos(ang)
                y = cy + rad*0.85*sin(ang)
                g = QRadialGradient(QPointF(x,y), 8+6*lvl)
                g.setColorAt(0.0, QColor.fromHsv(hue, 220, 255, 200))
                g.setColorAt(1.0, QColor.fromHsv(hue, 220, 0, 0))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(255,255,255,10), 1))
                p.drawEllipse(QPointF(x,y), 2.0+1.2*lvl, 2.0+1.2*lvl)
