from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n = len(bands)
    a = max(1, n//6); b = max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.54, down=0.2):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class VinylPulse(BaseVisualizer):
    display_name = "Vinyl Pulse"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rot = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2, 4, 8))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.8*rms, 0.72, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)
        self._rot += (25 + 60*self._env_mid)*(1/60.0)

        cx, cy = w*0.5, h*0.5
        radius = min(w, h)*0.38

        # Vinyl disk
        p.setPen(Qt.NoPen)
        base = QColor(8, 10, 16)
        p.setBrush(QBrush(base))
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Grooves
        p.setPen(QPen(QColor(40, 50, 70), 1))
        rings = 22
        for i in range(rings):
            d = i/float(rings)
            rr = radius*(0.15 + 0.8*d)
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        # Colored pulse spokes
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        spokes = 64
        for i in range(spokes):
            k = i/float(spokes)
            ang = self._rot*2*pi + k*2*pi
            amp = 0.3*self._env_lo + 0.6*self._env_mid + 0.4*self._env_hi
            length = radius*0.4 + radius*0.5*amp*sin(t*2 + i*0.4)

            x1 = cx + cos(ang)*radius*0.1
            y1 = cy + sin(ang)*radius*0.1
            x2 = cx + cos(ang)*(radius*0.1 + length)
            y2 = cy + sin(ang)*(radius*0.1 + length)

            hue = (int(t*20) + i*4) % 360
            col = QColor.fromHsv(hue, 220, 255, int(40 + 160*self._env_hi))
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(x1,y1), QPointF(x2,y2))

        # Center label
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        hue = (int(t*30) % 360)
        label_col = QColor.fromHsv(hue, 220, 255, 220)
        p.setPen(QPen(QColor(10,10,18), 2))
        p.setBrush(QBrush(label_col))
        p.drawEllipse(QPointF(cx, cy), radius*0.18, radius*0.18)
        p.setBrush(QBrush(QColor(10,10,18)))
        p.drawEllipse(QPointF(cx, cy), radius*0.04, radius*0.04)
