from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class TunnelWarp(BaseVisualizer):
    display_name = "Tunnel / Starfield Warp"
    def paint(self, p: QPainter, r, bands, rms, t):
        cx, cy = r.center().x(), r.center().y()
        p.setPen(QPen(QColor(170, 210, 255), 2))
        rings = 24
        for i in range(rings):
            rad = (i+1)/rings * min(r.width(), r.height())*0.5
            rad *= 0.9 + 0.1*sin(t*1.5 + i*0.3 + rms*2.0)
            p.drawEllipse(cx-rad, cy-rad, rad*2, rad*2)
