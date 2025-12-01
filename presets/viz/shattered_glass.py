
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
class ShatteredGlass(BaseVisualizer):
    display_name = "Shattered Glass"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.3 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.fillRect(r, QColor(2, 2, 8))

        cx, cy = w * 0.5, h * 0.5
        radius = min(w, h) * (0.3 + 0.15 * self._env_lo)
        shards = 26
        spin = t * (10 + 40 * self._env_mid)

        p.setCompositionMode(QPainter.CompositionMode_Screen)

        for i in range(shards):
            base_ang = (360.0 / shards) * i + spin
            offset = 10 + 35 * sin(t * 1.4 + i * 0.7)
            a0 = (base_ang - offset * 0.5) * pi / 180.0
            a1 = (base_ang + offset * 0.5) * pi / 180.0

            inner = radius * (0.1 + 0.25 * (1.0 + sin(t * 2.0 + i)))
            outer = radius * (0.9 + 0.3 * self._env_hi * sin(t * 3.1 + i))

            x0, y0 = cx, cy
            x1 = cx + cos(a0) * inner
            y1 = cy + sin(a0) * inner
            x2 = cx + cos(a1) * outer
            y2 = cy + sin(a1) * outer
            x3 = cx + cos(a0) * outer
            y3 = cy + sin(a0) * outer

            path = QPainterPath(QPointF(x0, y0))
            path.lineTo(x1, y1)
            path.lineTo(x2, y2)
            path.lineTo(x3, y3)
            path.closeSubpath()

            f = i / max(1, shards - 1)
            hue = int((180 + 80 * f + 140 * self._env_hi) % 360)
            alpha = int(40 + 160 * (0.4 + 0.6 * self._env_mid))
            col = QColor.fromHsv(hue, 80 + int(140 * self._env_hi), 255, alpha)

            p.setPen(QPen(col, 1))
            p.setBrush(QBrush(col))
            p.drawPath(path)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
