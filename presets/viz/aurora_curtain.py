
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
class AuroraCurtain(BaseVisualizer):
    display_name = "Aurora Curtain"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.3 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(2, 4, 10))

        bands_count = 80
        band_w = w / bands_count
        for i in range(bands_count):
            f = i / (bands_count - 1)
            x = i * band_w
            height = h * (
                0.25
                + 0.45 * sin(t * 0.8 + f * 4) * self._env_mid
                + 0.4 * self._env_lo * f
            )
            hue = int((180 + 80 * f + 160 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 200, 255, 190)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(x, h - height, band_w + 1, height))
