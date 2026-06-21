
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n // 6)
    b = max(a + 1, n // 2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b - a))
    hi = sum(bands[b:]) / max(1, (n - b))
    return lo, mid, hi


def _env_step(env, target, up=0.5, down=0.2):
    if target > env:
        return (1 - up) * env + up * target
    else:
        return (1 - down) * env + down * target



@register_visualizer
class ConstellationNet(BaseVisualizer):
    display_name = "Constellation Net"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self.N = 64
        self.phase = [random()*6.28 for _ in range(self.N)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4*rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(1, 1, 6))

        cx, cy = w*0.5, h*0.5
        rx = w*0.42*(0.9 + 0.2*self._env_lo)
        ry = h*0.36*(0.9 + 0.2*self._env_mid)

        pts = []
        for i in range(self.N):
            a = 1.3*sin(0.7*t + self.phase[i]) + self.phase[i]
            b = 1.7*cos(0.6*t + self.phase[i]*1.3)
            x = cx + rx * sin(a)
            y = cy + ry * cos(b)
            pts.append((x,y))

        # draw connections
        p.setPen(QPen(QColor(120, 210, 255, 90), 1))
        thresh = (60 + 120*self._env_hi)
        for i in range(self.N):
            xi, yi = pts[i]
            for j in range(i+1, self.N):
                xj, yj = pts[j]
                dx, dy = xi-xj, yi-yj
                d2 = dx*dx + dy*dy
                if d2 < thresh*thresh:
                    p.drawLine(xi, yi, xj, yj)

        # draw stars
        p.setPen(Qt.NoPen)
        for i,(x,y) in enumerate(pts):
            a = 140 + int(100*self._env_hi)
            hue = int((210 + 2*i) % 360)
            p.setBrush(QBrush(QColor.fromHsv(hue, 180, 255, a)))
            s = 2.0 + 1.5*self._env_hi
            p.drawEllipse(QPointF(x,y), s, s)
