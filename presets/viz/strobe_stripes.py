
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
class StrobeStripes(BaseVisualizer):
    display_name = "Strobe Stripes"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.65, 0.32)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(0, 0, 0))

        rows = 20
        stripe_h = h / rows
        base_freq = 3.0 + 5.0 * self._env_mid
        for i in range(rows):
            f = i / max(1, rows - 1)
            phase = t * base_freq + f * 4.0
            osc = 0.5 + 0.5 * sin(phase)
            lvl = osc * (0.4 + 0.6 * (self._env_lo + self._env_mid))

            hue = int((200 + 90 * f + 140 * self._env_hi) % 360)
            sat = int(140 + 100 * lvl)
            val = int(40 + 210 * lvl)
            alpha = int(80 + 140 * lvl)

            col = QColor.fromHsv(hue, sat, val, alpha)
            y = i * stripe_h
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRect(QRectF(0, y, w, stripe_h + 1))

        edge_y = h * (0.5 + 0.3 * sin(t * (4.0 + 4.0 * self._env_hi)))
        a = int(50 + 180 * self._env_hi)
        p.setPen(QPen(QColor(255, 255, 255, a), 2))
        p.setBrush(Qt.NoBrush)
        p.drawLine(0, edge_y, w, edge_y)
