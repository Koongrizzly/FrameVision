from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(42)
sites = [(_rng.random(), _rng.random()) for _ in range(140)]

@register_visualizer
class VoronoiShatter(BaseVisualizer):
    display_name = "Voronoi Shatter"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = r.width(), r.height()
        p.setPen(QPen(QColor(140, 180, 255, 180), 1))
        for (ux, uy) in sites:
            x = r.left() + ux*w; y = r.top() + uy*h
            x2 = x + 18*sin(t + x*0.01 + rms*2.0)
            y2 = y + 18*cos(t*0.8 + y*0.01 + rms*2.0)
            p.drawLine(x, y, x2, y2)
