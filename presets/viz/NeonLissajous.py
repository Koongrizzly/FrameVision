from math import sin, pi
from PySide6.QtGui import QPainter, QPen, QColor
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class NeonLissajous(BaseVisualizer):
    display_name = "Neon Lissajous XY"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = r.width(), r.height()
        cx, cy = r.center().x(), r.center().y()
        a = 2*pi*(0.5 + 0.2*rms)
        b = 2*pi*(1.0 + 0.3*rms)
        pen = QPen(QColor(0, 255, 240), 2 + int(4*rms))
        p.setPen(pen)
        prev = None
        N = 800
        for i in range(N):
            ph = i/N*2*pi
            x = cx + (w*0.35)*sin(a*ph + t*0.7)
            y = cy + (h*0.35)*sin(b*ph + t)
            if prev: p.drawLine(prev[0], prev[1], x, y)
            prev = (x,y)
