
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
class CosmicSpiral(BaseVisualizer):
    display_name = "Cosmic Spiral"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._angle = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.24)

        p.fillRect(r, QColor(3, 5, 12))
        cx, cy = w * 0.5, h * 0.5

        self._angle += (15 + 40 * self._env_mid) * (1 / 60.0)
        p.setPen(Qt.NoPen)

        arms = 5
        max_r = min(w, h) * 0.55
        for a in range(arms):
            base = self._angle + a * (360.0 / arms)
            for i in range(80):
                f = i / 80.0
                ang = (base + 260 * f) * pi / 180.0
                radius = max_r * (f ** 0.8)
                x = cx + cos(ang) * radius
                y = cy + sin(ang) * radius
                alpha = int(40 + 200 * (1.0 - f))
                hue = int((base + 120 * f + 200 * self._env_hi) % 360)
                col = QColor.fromHsv(hue, 180, 220, alpha)
                size = 2 + 3 * (1.0 - f) + 3 * self._env_lo
                p.setBrush(col)
                p.drawEllipse(QPointF(x, y), size, size * 0.85)

        core_r = min(w, h) * (0.06 + 0.08 * self._env_lo)
        p.setBrush(QBrush(QColor(255, 255, 255, 200)))
        p.drawEllipse(QPointF(cx, cy), core_r, core_r)
