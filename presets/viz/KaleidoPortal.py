from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QTransform
from PySide6.QtCore import QRectF
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class KaleidoPortal(BaseVisualizer):
    display_name = "Kaleido-Portal"
    def paint(self, p: QPainter, r, bands, rms, t):
        cx, cy = r.center().x(), r.center().y()
        slices = 10
        radius = min(r.width(), r.height())*0.45*(0.8+0.2*sin(t*0.7))
        pen = QPen(QColor(180, 220, 255), 2)
        p.setPen(pen)
        for i in range(slices):
            ang = 2*pi*i/slices + t*0.4*(0.6+0.4*rms)
            x = cx + radius*cos(ang)
            y = cy + radius*sin(ang)
            p.drawLine(cx, cy, x, y)
        if bands:
            n = len(bands)
            for i in range(n):
                v = max(0.0, min(1.0, bands[i]))
                rr = radius*(0.2+0.75*v)
                ang = 2*pi*i/n + t*0.2
                x = cx + rr*cos(ang); y = cy + rr*sin(ang)
                p.setPen(QPen(QColor(120+int(120*v), 80, 255), 3))
                p.drawPoint(x,y)
