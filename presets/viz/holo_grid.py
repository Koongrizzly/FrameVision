
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
class HoloGrid(BaseVisualizer):
    display_name = "Holo Grid"

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
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.25)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(2, 4, 10))

        cols = 46
        rows = 26
        cell_w = w / cols
        cell_h = h / rows

        base_phase = t * (0.5 + 1.6 * self._env_mid)

        for iy in range(rows):
            fy = iy / max(1, rows - 1)
            for ix in range(cols):
                fx = ix / max(1, cols - 1)
                phase = base_phase + fx * 5.3 + fy * 7.1
                wave = 0.5 + 0.5 * sin(phase)
                pulse = (0.2 + 0.8 * wave) * (0.25 + 0.75 * self._env_lo)
                brightness = min(1.0, pulse + 0.3 * self._env_hi)

                if brightness < 0.08:
                    continue

                hue = int((180 + 110 * fx + 80 * self._env_hi) % 360)
                val = int(40 + 190 * brightness)
                alpha = int(35 + 185 * brightness)

                col = QColor.fromHsv(hue, 200, val, alpha)
                x = ix * cell_w
                y = iy * cell_h
                margin = 0.25
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawRect(QRectF(x + cell_w * margin,
                                  y + cell_h * margin,
                                  cell_w * (1 - 2 * margin),
                                  cell_h * (1 - 2 * margin)))
