
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
class PixelCloud(BaseVisualizer):
    display_name = "Pixel Cloud"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, False)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid + 0.2 * rms, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(0, 0, 0))

        cell = max(6, int(min(w, h) * 0.02))
        cols = int(w / cell) + 1
        rows = int(h / cell) + 1

        base_hue = (200 + 60 * self._env_hi) % 360
        motion_speed = 0.8 + 2.5 * self._env_mid

        for iy in range(rows):
            ny = iy * 0.18
            for ix in range(cols):
                nx = ix * 0.22
                x = ix * cell
                y = iy * cell

                v = 0.5 + 0.5 * sin(
                    t * motion_speed
                    + nx * (0.7 + 0.6 * self._env_lo)
                    + ny * (0.9 + 0.8 * self._env_mid)
                )
                v *= 0.3 + 1.8 * (self._env_lo + self._env_mid + rms)

                if v < 0.12:
                    continue

                hue = int((base_hue + 40 * sin(nx * 0.3 + ny * 0.5)) % 360)
                sat = 140 + int(90 * self._env_hi)
                val = int(60 + 195 * min(1.0, v))
                alpha = int(40 + 210 * min(1.0, v))
                col = QColor.fromHsv(hue, sat, val, alpha)

                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawRect(QRectF(x, y, cell + 1, cell + 1))
