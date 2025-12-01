
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
class ChromaticGrid(BaseVisualizer):
    display_name = "Chromatic Grid"

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
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.28)

        p.fillRect(r, QColor(3, 3, 10))

        cols = 22
        rows = 12
        cw = w / float(cols)
        ch = h / float(rows)

        self._phase += (0.5 + 2.5 * self._env_mid) * (1 / 60.0)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for iy in range(rows):
            for ix in range(cols):
                x = ix * cw
                y = iy * ch

                fx = ix / max(1, cols - 1)
                fy = iy / max(1, rows - 1)

                v = (
                    0.4
                    + 0.6
                    * (
                        0.5
                        + 0.5
                        * sin(
                            self._phase * 2.0
                            + fx * 6.0
                            + fy * 4.0
                            + (self._env_lo + self._env_mid * 1.5) * 3.0
                        )
                    )
                )
                v *= (0.3 + 0.7 * (self._env_lo + self._env_mid + self._env_hi) * 0.5)

                if v < 0.08:
                    continue

                hue = int((220 + 160 * fx + 180 * self._env_hi) % 360)
                alpha = int(40 + 200 * v)
                sat = 80 + int(150 * (self._env_mid + self._env_hi) * 0.6)
                val = 100 + int(155 * v)
                col = QColor.fromHsv(hue, sat, val, alpha)

                margin = 0.12 + 0.18 * (1.0 - v)
                rx = x + cw * margin
                ry = y + ch * margin
                rw = cw * (1.0 - margin * 2)
                rh = ch * (1.0 - margin * 2)

                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawRoundedRect(QRectF(rx, ry, rw, rh), 2, 2)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
