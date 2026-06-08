
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
class ShatteredGrid(BaseVisualizer):
    display_name = "Shattered Grid"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.23)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(4, 4, 8))

        cols = 9
        rows = 5
        cell_w = w / cols
        cell_h = h / rows

        for iy in range(rows):
            for ix in range(cols):
                x0 = ix * cell_w
                y0 = iy * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h

                # jitter corners based on time and env
                j = 0.18 * min(cell_w, cell_h) * (0.4 + 0.6 * self._env_mid)
                phase = t * 0.8
                c1 = QPointF(x0 + j * sin(phase + (ix + iy) * 0.7),
                             y0 + j * cos(phase * 1.1 + ix * 0.4))
                c2 = QPointF(x1 + j * sin(phase * 1.2 + (ix + 1) * 0.6),
                             y0 + j * cos(phase * 0.9 + iy * 0.5))
                c3 = QPointF(x1 + j * sin(phase * 0.7 + (ix + iy + 3) * 0.5),
                             y1 + j * cos(phase * 1.3 + ix * 0.3))
                c4 = QPointF(x0 + j * sin(phase * 1.4 + (ix + 2) * 0.4),
                             y1 + j * cos(phase * 1.0 + iy * 0.6))

                f = (ix + iy) / max(1, cols + rows - 2)
                lvl = 0.4 + 0.6 * (self._env_lo + self._env_mid) * (0.5 + 0.5 * sin(phase + f * 3.0))
                hue = int((260 + 80 * f + 120 * self._env_hi) % 360)
                sat = int(140 + 70 * self._env_mid)
                val = int(80 + 150 * lvl)
                alpha = int(130 + 100 * lvl)

                col = QColor.fromHsv(hue, sat, val, alpha)
                edge_col = QColor.fromHsv(hue, sat, 255, 180)

                path = QPainterPath()
                path.moveTo(c1)
                path.lineTo(c2)
                path.lineTo(c3)
                path.lineTo(c4)
                path.closeSubpath()

                p.setPen(QPen(edge_col, 1.2))
                p.setBrush(QBrush(col))
                p.drawPath(path)
