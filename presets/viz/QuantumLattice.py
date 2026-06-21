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

def _env_step(env, target, up=0.58, down=0.22):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

@register_visualizer
class QuantumLattice(BaseVisualizer):
    display_name = "Quantum Lattice"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 6, 14))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.58, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.64, 0.24)

        cols = 12
        rows = 7
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for j in range(rows):
            for i in range(cols):
                tx = (i + 0.5) / float(cols)
                ty = (j + 0.5) / float(rows)
                x = tx * w
                y = ty * h

                jitter_x = sin(t * 0.7 + tx * 8 + ty * 4) * 10 * self._env_lo
                jitter_y = cos(t * 0.9 + tx * 5 + ty * 6) * 12 * self._env_mid
                x += jitter_x
                y += jitter_y

                pop = (0.3 + 0.7 * self._env_hi) * (0.6 + 0.4 * sin(t * 1.4 + i * 2 + j))
                sz = 3 + 12 * pop

                hue = (int(t * 28) + i * 14 + j * 9) % 360
                col = QColor.fromHsv(hue, 230, 255, int(60 + 160 * pop))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(x, y), sz, sz)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
