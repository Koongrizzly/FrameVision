from math import sin, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF
from helpers.music import register_visualizer, BaseVisualizer

@register_visualizer
class EqualizerCity(BaseVisualizer):
    display_name = "Equalizer City"
    def paint(self, p: QPainter, r, bands, rms, t):
        if not bands: return
        w,h = r.width(), r.height()
        n = len(bands); bw = max(2, int(w/(n*1.1))); gap = max(1, int(bw*0.1))
        x = r.left()
        for i,v in enumerate(bands):
            v = max(0.0, min(1.0, v))
            bh = int(v*h*0.85*(0.8+0.2*sin(t+ i*0.1)))
            p.setBrush(QBrush(QColor(40,40,60)))
            p.setPen(QPen(QColor(100, 160, 255), 1))
            p.drawRect(QRectF(x, h-bh, bw, bh))
            x += bw + gap
