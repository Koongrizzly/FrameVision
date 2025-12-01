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
class OrbitWaves(BaseVisualizer):
    display_name = "Orbit Waves"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rot = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 6, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)
        self._rot += (8 + 32*self._env_mid)*(1/60.0)

        cx, cy = w*0.5, h*0.5
        base_r = min(w, h)*0.18

        p.setCompositionMode(QPainter.CompositionMode_Plus)

        # Central orbit rings
        for i in range(5):
            d = i/4.0
            radius = base_r*(1.0 + 0.35*i + 0.2*self._env_lo*sin(t*1.3 + i))
            hue = (int(t*25) + i*40) % 360
            col = QColor.fromHsv(hue, 220, 255, int(60 + 120*(1-d)*self._env_hi))
            p.setPen(QPen(col, 2))
            p.drawEllipse(QPointF(cx, cy), radius, radius*0.95)

        # Orbiting blobs
        satellites = 9
        for i in range(satellites):
            k = i/float(satellites)
            ang = self._rot*2*pi + k*2*pi + 0.5*sin(t + i)
            radius = base_r*1.4 + 40*sin(t*0.7 + i*1.2)
            x = cx + cos(ang)*radius
            y = cy + sin(ang)*radius*0.7

            size = 10 + 12*self._env_hi*(0.5 + 0.5*sin(t*2 + i))
            hue = (int(t*40) + i*35) % 360
            col = QColor.fromHsv(hue, 220, 255, 150)
            glow = QColor.fromHsv(hue, 180, 255, 60)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(x, y), size*1.6, size*1.1)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(x, y), size*0.7, size*0.7)

        # Subtle crosshair
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self._env_hi > 0.5:
            a = int(40 + 80*self._env_hi)
            p.setPen(QPen(QColor(200, 230, 255, a), 1))
            p.drawLine(cx-20, cy, cx+20, cy)
            p.drawLine(cx, cy-20, cx, cy+20)
