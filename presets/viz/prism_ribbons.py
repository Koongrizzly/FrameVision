
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
class PrismRibbons(BaseVisualizer):
    display_name = "Prism Ribbons"

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

        ribbons = 4
        steps = 64
        base_amp = h * (0.04 + 0.12 * self._env_lo)

        for idx in range(ribbons):
            offset = idx / max(1, ribbons - 1)
            y_base = h * (0.2 + 0.6 * offset)
            phase_shift = offset * 3.2

            path = QPainterPath(QPointF(-0.1 * w, h + 10))
            path.lineTo(-0.1 * w, y_base)

            for i in range(steps + 1):
                fx = i / max(1, steps)
                x = -0.1 * w + (1.2 * w) * fx
                wave = sin(t * (0.8 + 0.7 * self._env_mid) + fx * 5.0 + phase_shift)
                wave2 = sin(t * 1.7 + fx * 9.0 + phase_shift * 1.3)
                y = y_base + base_amp * wave + base_amp * 0.5 * self._env_mid * wave2
                path.lineTo(x, y)

            path.lineTo(1.1 * w, h + 10)
            path.closeSubpath()

            hue = int((200 + 60 * offset + 160 * self._env_hi) % 360)
            val = int(120 + 120 * self._env_mid)
            alpha = int(80 + 80 * (1.0 - offset) + 40 * self._env_lo)
            col = QColor.fromHsv(hue, 180, val, alpha)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawPath(path)
