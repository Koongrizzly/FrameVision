
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
class HexPulse(BaseVisualizer):
    display_name = "Hex Pulse"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(4, 4, 8))

        size = min(w, h) * 0.06 * (1.0 + 0.2 * self._env_lo)
        cols = int(w / (size * 0.9)) + 2
        rows = int(h / (size * 0.8)) + 2

        for iy in range(rows):
            for ix in range(cols):
                x = ix * size * 0.9 + (iy % 2) * size * 0.45
                y = iy * size * 0.8
                if x > w + size or y > h + size:
                    continue
                dx = x - w * 0.5
                dy = y - h * 0.5
                dist = (dx * dx + dy * dy) ** 0.5
                f = max(0.0, 1.0 - dist / (max(w, h) * 0.7))
                hue = int((200 + 40 * self._env_hi + 80 * f) % 360)
                alpha = int(20 + 180 * f + 80 * self._env_mid)
                col = QColor.fromHsv(hue, 220, 200, alpha)
                p.setPen(QPen(col, 1))
                p.setBrush(Qt.NoBrush)
                path = QPainterPath()
                for k in range(7):
                    ang = (60 * k + t * 40 * self._env_mid) * pi / 180.0
                    xx = x + cos(ang) * size * 0.5
                    yy = y + sin(ang) * size * 0.5
                    if k == 0:
                        path.moveTo(xx, yy)
                    else:
                        path.lineTo(xx, yy)
                p.drawPath(path)
