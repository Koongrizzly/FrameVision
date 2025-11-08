
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
class RipplePoints(BaseVisualizer):
    display_name = "Ripple Points"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.23)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(1, 3, 8))

        cx, cy = w * 0.5, h * 0.5
        max_r = (min(w, h) * 0.6) * (1.0 + 0.2 * self._env_lo)

        rings = 5
        points_per_ring = 32

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(rings):
            f_ring = j / max(1, rings - 1)
            radius = max_r * (0.2 + 0.75 * f_ring)
            phase = t * (0.6 + 1.2 * self._env_mid) + f_ring * 2.4

            for i in range(points_per_ring):
                f = i / points_per_ring
                ang = 2 * pi * f
                wobble = 0.22 * sin(ang * 3.0 + phase)
                r2 = radius * (1.0 + wobble * (0.5 + 0.5 * self._env_mid))

                x = cx + cos(ang) * r2
                y = cy + sin(ang) * r2

                depth = 0.3 + 0.7 * f_ring
                hue = int((180 + 140 * depth + 140 * self._env_hi) % 360)
                sat = int(150 + 60 * self._env_mid)
                val = int(120 + 60 * self._env_lo)
                alpha = int(40 + 120 * (1.0 - f_ring) * (0.5 + 0.5 * self._env_mid))

                col = QColor.fromHsv(hue, sat, val, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                size = 2.0 + 3.0 * (1.0 - f_ring)
                p.drawEllipse(QPointF(x, y), size, size)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
