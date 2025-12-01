
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
class SpectrumCity(BaseVisualizer):
    display_name = "Spectrum City"

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
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.24)

        p.fillRect(r, QColor(3, 5, 14))

        col_count = 42
        col_w = w / float(col_count)
        drift = sin(t * 0.15) * col_w * 3.0

        for i in range(col_count + 2):
            x = (i - 1) * col_w + drift
            if x > w or x + col_w < 0:
                continue

            f = i / max(1, col_count - 1)
            base = 0.15 + 0.75 * (self._env_lo * 0.7 + self._env_mid * 0.3)
            wave = 0.3 + 0.35 * sin(t * 0.6 + f * 5.2)
            height = h * max(0.05, min(0.95, base * wave))

            y = h - height
            hue = int((210 + 40 * f + 120 * self._env_hi) % 360)
            body_col = QColor.fromHsv(hue, 80, 100, 255)
            p.setPen(Qt.NoPen)
            p.setBrush(body_col)
            p.drawRect(QRectF(x, y, col_w * 0.9, height))

            neon = QColor.fromHsv(hue, 220, 255, 200)
            p.setPen(QPen(neon, 2))
            p.setBrush(Qt.NoBrush)
            p.drawLine(x, y, x + col_w * 0.9, y)

            p.setPen(Qt.NoPen)
            win_col = QColor(255, 220, 160, 120 + int(80 * self._env_mid))
            p.setBrush(win_col)
            rows = int(max(2, height / 18))
            cols = 3
            for ry in range(rows):
                if random() > 0.55 + 0.3 * (1.0 - self._env_mid):
                    continue
                wy = y + 6 + ry * (height / rows)
                for cx in range(cols):
                    if random() > 0.6:
                        continue
                    wx = x + 4 + cx * (col_w * 0.25)
                    p.drawRect(QRectF(wx, wy, 4, 6))
