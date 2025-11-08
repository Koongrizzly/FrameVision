
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
class KaleidoTiles(BaseVisualizer):
    display_name = "Kaleido Tiles"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(3, 3, 10))

        self._phase += (20 + 80 * self._env_mid) * (1.0 / 60.0)
        cols = 10
        rows = 6
        tile_w = w / cols
        tile_h = h / rows

        for iy in range(rows):
            for ix in range(cols):
                cx = (ix + 0.5) * tile_w - w * 0.5
                cy = (iy + 0.5) * tile_h - h * 0.5
                angle = (self._phase + (cx * 0.03) + (cy * 0.04)) * pi / 180.0
                rad = (abs(cx) + abs(cy)) * 0.3
                distort = sin(angle * 1.7) * cos(angle * 1.3)
                f = 0.5 + 0.5 * sin(rad * 0.02 + distort * 2.0)

                hue = int((180 + 180 * f + 160 * self._env_hi) % 360)
                sat = int(150 + 80 * self._env_mid)
                val = int(80 + 160 * (0.3 + self._env_lo))
                alpha = 180

                col = QColor.fromHsv(hue, sat, val, alpha)

                x0 = ix * tile_w
                y0 = iy * tile_h
                path = QPainterPath()
                path.moveTo(x0 + tile_w * 0.5, y0)
                path.lineTo(x0 + tile_w, y0 + tile_h * 0.5)
                path.lineTo(x0 + tile_w * 0.5, y0 + tile_h)
                path.lineTo(x0, y0 + tile_h * 0.5)
                path.closeSubpath()

                p.setPen(QPen(QColor(10, 10, 10, 220), 1))
                p.setBrush(QBrush(col))
                p.drawPath(path)
