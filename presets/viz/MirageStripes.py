from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient
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

def _env_step(env, target, up=0.6, down=0.23):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

@register_visualizer
class MirageStripes(BaseVisualizer):
    display_name = "Mirage Stripes"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 5, 12))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.66, 0.25)

        bands_count = 24
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for i in range(bands_count):
            y0 = (i / float(bands_count)) * h
            thickness = h / bands_count * (0.9 + 0.6 * sin(t * 0.7 + i * 0.5) * self._env_lo)
            y = y0 + sin(t * 0.9 + i) * 8 * self._env_mid

            hue = (int(t * 24) + i * 7) % 360
            col1 = QColor.fromHsv(hue, 240, 255, int(40 + 120 * self._env_hi))
            col2 = QColor.fromHsv((hue + 120) % 360, 200, 200, int(20 + 70 * self._env_hi))

            grad = QLinearGradient(0, y, w, y + thickness)
            grad.setColorAt(0.0, col1)
            grad.setColorAt(0.5, QColor(255, 255, 255, 40))
            grad.setColorAt(1.0, col2)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRect(0, y, w, thickness)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
