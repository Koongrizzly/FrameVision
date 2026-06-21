
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
class CircuitFlow(BaseVisualizer):
    display_name = "Circuit Flow"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.28)

        # dark circuit-board background
        p.fillRect(r, QColor(0, 8, 6))

        margin = 0.08 * min(w, h)
        cols = 12
        rows = 7
        if cols < 2 or rows < 2:
            return

        span_w = w - 2 * margin
        span_h = h - 2 * margin

        # grid of pads
        for iy in range(rows):
            for ix in range(cols):
                x = margin + (ix / (cols - 1)) * span_w
                y = margin + (iy / (rows - 1)) * span_h

                lvl = 0.4 + 0.6 * (self._env_lo + self._env_mid) * random()
                hue = int((130 + 40 * lvl + 80 * self._env_hi) % 360)
                sat = int(140 + 80 * lvl)
                val = int(100 + 120 * lvl)
                alpha = int(160 + 80 * lvl)
                col = QColor.fromHsv(hue, sat, val, alpha)

                size = 4 + 4 * lvl
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawRect(QRectF(x - size * 0.5, y - size * 0.5, size, size))

        # traces between pads (horizontal & vertical)
        base_thick = 1.0 + 2.5 * self._env_mid
        for iy in range(rows):
            for ix in range(cols):
                x = margin + (ix / (cols - 1)) * span_w
                y = margin + (iy / (rows - 1)) * span_h

                # right
                if ix < cols - 1:
                    x2 = margin + ((ix + 1) / (cols - 1)) * span_w
                    # brightness pulses with mid env and time
                    osc = 0.5 + 0.5 * sin(t * 3.0 + (ix + iy * 2) * 0.7)
                    lvl = osc * (0.3 + 0.7 * self._env_mid)
                    alpha = int(40 + 180 * lvl)
                    col = QColor(40, 180, 120 + int(80 * self._env_hi), alpha)
                    p.setPen(QPen(col, base_thick, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(x, y, x2, y)

                # down
                if iy < rows - 1:
                    y2 = margin + ((iy + 1) / (rows - 1)) * span_h
                    osc = 0.5 + 0.5 * sin(t * 2.6 + (ix * 1.3 + iy) * 0.8)
                    lvl = osc * (0.3 + 0.7 * self._env_mid)
                    alpha = int(40 + 180 * lvl)
                    col = QColor(40, 150, 200 + int(40 * self._env_hi), alpha)
                    p.setPen(QPen(col, base_thick, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(x, y, x, y2)
