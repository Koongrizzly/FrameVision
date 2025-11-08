
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
class OrigamiWaves(BaseVisualizer):
    display_name = "Origami Waves"

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
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(4, 5, 10))

        layers = 6
        for layer in range(layers):
            ybase = h * (0.2 + layer * 0.12)
            amp = h * (0.02 + 0.04 * (layers - layer) * self._env_lo)
            steps = 40
            path = QPainterPath(QPointF(0, h))
            path.lineTo(0, ybase)
            for i in range(steps + 1):
                x = (i / steps) * w
                y = ybase + amp * sin(
                    t * 0.9 + i * 0.4 + layer
                ) + amp * self._env_mid * sin(i * 0.2 + t * 1.7)
                path.lineTo(x, y)
            path.lineTo(w, h)
            path.closeSubpath()
            hue = int((140 + layer * 22 + 160 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 160, 230, 120 + int(20 * layer))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawPath(path)
