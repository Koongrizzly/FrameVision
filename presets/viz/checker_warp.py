
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
class CheckerWarp(BaseVisualizer):
    display_name = "Checker Warp"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(8, 8, 10))

        cells = 22
        cw = w / cells
        ch = h / cells
        amp = 0.35 * min(cw, ch) * (0.6 + self._env_lo)

        for j in range(cells):
            for i in range(cells):
                # warp center
                wx = amp * sin(0.9*t + 0.4*i + 0.6*j + 4*self._env_mid)
                wy = amp * cos(0.8*t + 0.3*j + 0.5*i + 5*self._env_hi)
                x = i*cw + wx
                y = j*ch + wy
                hue = int(( (i*7 + j*5) + 180*self._env_hi ) % 360)
                val = 200 if (i+j)%2==0 else 80
                sat = 160 + int(60*self._env_mid)
                col = QColor.fromHsv(hue, sat, val, 200)
                p.fillRect(QRectF(x, y, cw+2, ch+2), col)
