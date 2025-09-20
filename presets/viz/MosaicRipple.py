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
class MosaicRipple(BaseVisualizer):
    display_name = "Mosaic Ripple"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return
        p.fillRect(r, QBrush(QColor(10,10,18)))
        lvl = bass_level(bands, rms)
        step = max(10, int(min(w,h)*0.04))
        cx, cy = r.center().x(), r.center().y()
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx = x - cx; dy = y - cy
                d = sqrt(dx*dx + dy*dy) / max(1.0, min(w,h))
                wave = 0.5 + 0.5*sin(10*d - t*1.2)
                hue = int((t*40 + 360*d) % 360)
                val = int(120 + 120*(wave*(0.6+0.6*lvl)))
                col = QColor.fromHsv(hue, 210, min(255,val), 210)
                p.fillRect(QRectF(x,y,step,step), QBrush(col))
