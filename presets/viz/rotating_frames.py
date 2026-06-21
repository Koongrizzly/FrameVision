
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
class RotatingFrames(BaseVisualizer):
    display_name = "Rotating Frames"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.25)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(2, 3, 9))

        cx, cy = w * 0.5, h * 0.5
        base_size = min(w, h) * 0.15
        layers = 8

        p.save()
        p.translate(cx, cy)

        base_rot = t * (18 + 40 * self._env_mid)
        for i in range(layers):
            f = i / max(1, layers - 1)
            size = base_size * (1.3 + 2.2 * f * (1.0 + 0.25 * self._env_lo))
            angle = base_rot * (1.0 + 0.15 * i) * (1 if i % 2 == 0 else -1)
            p.save()
            p.rotate(angle)

            rect = QRectF(
                -size,
                -size * (0.7 + 0.2 * sin(t * 0.6 + i)),
                size * 2,
                size * 2 * (0.7 + 0.2 * sin(t * 0.6 + i)),
            )

            hue = int((210 + 20 * i + 140 * self._env_hi) % 360)
            val = int(110 + 130 * (0.3 + 0.7 * self._env_mid))
            alpha = int(40 + 160 * (1.0 - f))
            col = QColor.fromHsv(hue, 190, val, alpha)

            p.setPen(QPen(col, 2.0 + 2.5 * (1.0 - f)))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

            p.restore()

        p.restore()
