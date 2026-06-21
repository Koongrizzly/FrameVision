
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
class CitySpectrum(BaseVisualizer):
    display_name = "City Spectrum"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        p.fillRect(r, QColor(2, 4, 10))

        base_y = h * 0.82
        max_height = h * 0.7

        cols = 64
        col_w = w / cols

        for i in range(cols):
            f = i / max(1, cols - 1)
            idx = int(f * (len(bands) - 1)) if bands else 0
            lvl = bands[idx] if bands else 0.0
            height = max_height * (0.05 + 0.9 * (lvl * 0.6 + self._env_mid * 0.8))

            x = i * col_w
            y = base_y - height

            hue = int((210 + 90 * f + 140 * self._env_hi) % 360)
            brightness = int(140 + 110 * self._env_lo)
            col = QColor.fromHsv(hue, 200, brightness, 230)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRect(QRectF(x, y, col_w * 0.8, height))

            p.setBrush(QColor(10, 10, 10, 180))
            win_h = 6
            step = 18
            wy = base_y - win_h * 0.5
            while wy > y + 10:
                alpha = 60 + int(120 * random() * (0.4 + self._env_hi))
                p.setBrush(QColor(0, 0, 0, alpha))
                p.drawRect(QRectF(x + col_w * 0.08, wy, col_w * 0.64, win_h))
                wy -= step

        grad_h = h * 0.22
        for i in range(40):
            f = i / 39.0
            alpha = int(80 * (1.0 - f) * (0.4 + self._env_lo))
            col = QColor(10, 40, 70, alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRect(QRectF(0, base_y + f * grad_h, w, grad_h / 40.0))
