from math import sin, cos, pi, hypot
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class MetaballLiquid(BaseVisualizer):
    display_name = "Metaball Liquid Metal"
    def paint(self, p: QPainter, r, bands, rms, t):
        cx, cy = r.center().x(), r.center().y()
        p.setPen(QPen(QColor(200, 210, 255), 3))
        count = 8
        for i in range(count):
            ang = i/count*2*pi + t*(0.4 + 0.3*rms)
            rad = min(r.width(), r.height())*(0.15 + 0.1*sin(t+i))
            x = cx + (rad*2.2)*cos(ang); y = cy + (rad*2.2)*sin(ang)
            p.drawEllipse(x-rad, y-rad, rad*2, rad*2)
