
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
class DiagonalRibbons(BaseVisualizer):
    display_name = "Diagonal Ribbons"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._offset = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.23)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(2, 4, 8))

        speed = 90 + 220 * self._env_mid
        self._offset += speed * (1.0 / 60.0)
        spacing = h * 0.32

        ribbons = 7
        for i in range(-2, ribbons + 2):
            y_off = (i * spacing + self._offset) % (spacing * ribbons)
            f = (i + 2) / max(1, ribbons + 4)
            hue = int((190 + 100 * f + 140 * self._env_hi) % 360)
            sat = int(160 + 80 * self._env_mid)
            val = int(120 + 120 * self._env_lo)
            alpha = int(80 + 130 * (0.2 + 0.8 * self._env_lo))

            col = QColor.fromHsv(hue, sat, val, alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(col)

            path = QPainterPath()
            path.moveTo(-w * 0.3, y_off - spacing)
            path.lineTo(w * 1.3, y_off - spacing * 0.2)
            path.lineTo(w * 1.3, y_off + spacing * 0.2)
            path.lineTo(-w * 0.3, y_off + spacing)
            path.closeSubpath()
            p.drawPath(path)
