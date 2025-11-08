
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
class NeonTriangles(BaseVisualizer):
    display_name = "Neon Triangles"

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
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.23)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(0, 0, 0))

        cx, cy = w * 0.5, h * 0.5
        base_r = min(w, h) * (0.18 + 0.18 * self._env_lo)

        self._angle += (50 + 90 * self._env_mid) * (1.0 / 60.0)
        base_angle = self._angle * pi / 180.0

        layers = 4
        per_layer = 10
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(layers):
            f_layer = j / max(1, layers - 1)
            radius = base_r * (1.0 + 0.7 * f_layer)
            thickness = 1.5 + 1.0 * (layers - j)
            for i in range(per_layer):
                a = base_angle + (2 * pi * i / per_layer) + f_layer * 0.7
                spread = 0.33 + 0.15 * sin(a * 2.0 + t * 0.7)
                a1 = a - spread
                a2 = a + spread

                p1 = QPointF(cx, cy)
                p2 = QPointF(cx + cos(a1) * radius, cy + sin(a1) * radius)
                p3 = QPointF(cx + cos(a2) * radius, cy + sin(a2) * radius)

                hue = int((320 - 70 * f_layer + 160 * self._env_hi) % 360)
                sat = int(170 + 60 * self._env_mid)
                val = int(180 + 60 * self._env_lo)
                alpha = int(80 + 150 * (1.0 - f_layer) * (0.5 + 0.5 * self._env_mid))

                col = QColor.fromHsv(hue, sat, val, alpha)
                p.setPen(QPen(col, thickness, Qt.SolidLine, Qt.RoundCap))
                p.setBrush(QBrush(QColor(0, 0, 0, 0)))

                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(p2)
                path.lineTo(p3)
                path.closeSubpath()
                p.drawPath(path)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
