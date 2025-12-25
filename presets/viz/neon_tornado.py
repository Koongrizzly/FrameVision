
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
class NeonTornado(BaseVisualizer):
    display_name = "Neon Tornado"

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
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.26)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.28)

        p.fillRect(r, QColor(1, 3, 10))

        cx, cy = w * 0.5, h * 0.55

        layers = 22
        height = h * 0.8

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(layers):
            f = i / max(1, layers - 1)
            y = cy + (f - 0.5) * height

            swirl = sin(t * 1.2 + f * 8.0) * (0.3 + 0.3 * self._env_mid)
            radius = (0.15 + 0.4 * (1.0 - f) * (0.4 + self._env_lo)) * min(w, h)
            x = cx + swirl * radius

            band = 8 + 18 * (1.0 - f)
            hue = int((190 + 180 * f + 120 * self._env_hi) % 360)
            alpha = int(30 + 180 * (0.3 + 0.7 * self._env_mid))
            col = QColor.fromHsv(hue, 230, 255, alpha)

            p.setPen(QPen(col, band, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(x - radius * 0.6, y, x + radius * 0.6, y)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
