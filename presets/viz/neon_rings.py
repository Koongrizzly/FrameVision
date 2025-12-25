
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
class NeonRings(BaseVisualizer):
    display_name = "Neon Rings"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(0, 0, 0))
        cx, cy = w * 0.5, h * 0.5

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        rings = 12
        for i in range(rings):
            f = i / (rings - 1)
            radius = (min(w, h) * 0.1 + f * min(w, h) * 0.45) * (
                1.0 + 0.18 * self._env_lo
            )
            width = 4 + 6 * (1.0 - f)
            hue = int((220 + 80 * f + 140 * self._env_hi) % 360)
            alpha = int(30 + 220 * (1.0 - f) * (0.5 + 0.5 * self._env_mid))
            col = QColor.fromHsv(hue, 230, 255, alpha)
            p.setPen(QPen(col, width))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), radius, radius)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
