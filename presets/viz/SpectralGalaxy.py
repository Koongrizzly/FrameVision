from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class SpectralGalaxy(BaseVisualizer):
    display_name = "3D Spectral Galaxy"
    def paint(self, p: QPainter, r, bands, rms, t):
        cx, cy = r.center().x(), r.center().y()
        p.setPen(QPen(QColor(200, 220, 255), 2))
        arms = 4
        points = 1400
        for i in range(points):
            a = (i%arms)/arms*2*pi + t*0.1
            rad = (i/points)*min(r.width(), r.height())*0.48*(0.6 + 0.4*rms)
            x = cx + rad*cos(a + 0.5*sin(t*0.6))
            y = cy + rad*sin(a + 0.5*cos(t*0.6))
            p.drawPoint(x,y)
