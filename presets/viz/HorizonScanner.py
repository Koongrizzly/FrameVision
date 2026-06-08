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

def _env_step(env, target, up=0.56, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class HorizonScanner(BaseVisualizer):
    display_name = "Horizon Scanner"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2, 4, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7*rms, 0.7, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)
        self._phase += 0.6 + 1.5*self._env_mid

        horizon_y = int(h*0.6)

        # Grid
        p.setPen(QPen(QColor(20, 40, 70), 1))
        for y in range(horizon_y, h, 18):
            p.drawLine(0, y, w, y)
        for x in range(0, w, 40):
            p.drawLine(x, horizon_y, x, h)

        # Peaks from spectrum
        if bands:
            step = w / float(len(bands))
            p.setPen(QPen(QColor(100, 220, 255), 2))
            for i, v in enumerate(bands):
                x = i*step
                a = (0.4*self._env_lo + 0.7*self._env_mid + 0.5*v)
                height = min(h*0.45, h*a*0.7)
                p.drawLine(x, horizon_y, x, horizon_y - height)

        # Scanning line
        p.setPen(QPen(QColor(180, 240, 255, 180), 3))
        y = horizon_y - (20 + 80*sin(self._phase*0.08))
        p.drawLine(0, int(y), w, int(y))

        # Sky glow
        p.setPen(Qt.NoPen)
        a = int(40 + 140*self._env_hi)
        if a > 0:
            p.setBrush(QBrush(QColor(80, 140, 255, a)))
            p.drawRect(0, 0, w, horizon_y)
