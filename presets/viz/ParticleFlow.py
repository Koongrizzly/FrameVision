from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(1337)
pts = [(_rng.random(), _rng.random()) for _ in range(800)]

@register_visualizer
class ParticleFlow(BaseVisualizer):
    display_name = "Particle Flow Field"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = r.width(), r.height()
        p.setPen(QPen(QColor(180, 220, 255, 160), 1))
        flow = 1.5 + 2.0*rms
        for (ux, uy) in pts:
            x = r.left() + ux*w; y = r.top() + uy*h
            ang = sin( (x+y)*0.01 + t*flow ) + cos( (x-y)*0.008 - t*0.8 )
            dx = 10*cos(ang); dy = 10*sin(ang)
            p.drawLine(QPointF(x,y), QPointF(x+dx,y+dy))
