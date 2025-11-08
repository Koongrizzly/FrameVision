
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
class WaveField(BaseVisualizer):
    display_name = "Wave Field"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.24)

        p.fillRect(r, QColor(1, 6, 10))

        rows = 5
        for row in range(rows):
            y0 = h * (0.3 + 0.1 * row)
            amp = h * (0.02 + 0.05 * self._env_lo * (1 + row * 0.3))
            path = QPainterPath(QPointF(0, y0))
            steps = 80
            for i in range(1, steps + 1):
                x = (i / steps) * w
                y = y0 + amp * sin(
                    t * 1.2 + i * 0.3 + row * 0.7
                ) + amp * self._env_mid * sin(i * 0.15 + t * 2)
                path.lineTo(x, y)
            hue = int((160 + row * 25 + 150 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 200, 255, 170)
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
