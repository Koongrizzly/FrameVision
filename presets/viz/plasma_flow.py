
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
class PlasmaFlow(BaseVisualizer):
    display_name = "Plasma Flow"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(2, 2, 6))

        layers = 6
        blobs = 18

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(layers):
            f_layer = j / max(1, layers - 1)
            radius_base = min(w, h) * (0.22 + 0.25 * f_layer) * (1.0 + 0.25 * self._env_lo)
            for i in range(blobs):
                ang = (t * 0.3 + i * 40 + j * 60) * pi / 180.0
                wob = sin(t * (0.7 + 0.2 * j) + i * 0.9)
                cx = w * (0.5 + 0.35 * cos(ang) * (0.3 + 0.7 * self._env_mid))
                cy = h * (0.5 + 0.35 * sin(ang) * (0.3 + 0.7 * self._env_mid))
                radius = radius_base * (0.7 + 0.3 * wob)

                hue = int((220 + 80 * f_layer + 120 * self._env_hi + 20 * wob) % 360)
                sat = int(170 + 60 * self._env_mid)
                val = int(100 + 120 * (0.4 + 0.6 * self._env_lo))
                alpha = int(40 + 70 * (1.0 - f_layer))

                col = QColor.fromHsv(hue, sat, val, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(cx, cy), radius, radius)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
