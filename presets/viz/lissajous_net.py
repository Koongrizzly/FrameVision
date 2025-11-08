
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
class LissajousNet(BaseVisualizer):
    display_name = "Lissajous Net"

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
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(3, 5, 12))

        self._phase += (0.8 + 1.6 * self._env_mid) * (1.0 / 60.0)

        points = []
        nodes = 40
        for i in range(nodes):
            f = i / max(1, nodes - 1)
            ax = 1.0 + 2.0 * self._env_lo
            ay = 2.0 + 1.5 * self._env_lo
            x = 0.5 + 0.42 * sin(2 * pi * f * ax + self._phase * 3.1)
            y = 0.5 + 0.42 * sin(2 * pi * f * ay + self._phase * 2.3)
            points.append((x * w, y * h))

        for i in range(nodes):
            x1, y1 = points[i]
            for j in range(i + 1, nodes):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                dist2 = dx * dx + dy * dy
                if dist2 > (min(w, h) * 0.3) ** 2:
                    continue
                f = max(0.0, 1.0 - dist2 / (min(w, h) * 0.3) ** 2)
                alpha = int(15 + 160 * f * (0.5 + 0.5 * self._env_mid))
                hue = int((190 + 80 * f + 140 * self._env_hi) % 360)
                col = QColor.fromHsv(hue, 210, 230, alpha)
                p.setPen(QPen(col, 1))
                p.drawLine(x1, y1, x2, y2)

        for (x, y) in points:
            hue = int((210 + 80 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 230, 255, 220)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            s = 3 + 2 * self._env_hi
            p.drawEllipse(QPointF(x, y), s, s)
