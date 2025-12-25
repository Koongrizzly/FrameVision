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

def _env_step(env, target, up=0.6, down=0.23):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

@register_visualizer
class MetaCubes(BaseVisualizer):
    display_name = "Meta Cubes"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def _cube(self, size):
        s = size
        path = QPainterPath()
        path.moveTo(-s, -s * 0.5)
        path.lineTo(0, -s)
        path.lineTo(s, -s * 0.5)
        path.lineTo(0, 0)
        path.closeSubpath()

        front = QPainterPath()
        front.moveTo(-s, 0)
        front.lineTo(0, s * 0.5)
        front.lineTo(s, 0)
        front.lineTo(0, -s * 0.5)
        front.closeSubpath()

        return path, front

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 5, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.62, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.66, 0.25)

        cols = 9
        rows = 6
        step_x = w / (cols + 1.0)
        step_y = h / (rows + 1.0)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(rows):
            for i in range(cols):
                cx = step_x * (i + 1)
                cy = step_y * (j + 1)
                wobble = sin(t * 0.9 + i * 0.6 + j * 0.4) * 12 * self._env_lo
                cy += wobble

                base_size = min(step_x, step_y) * (0.3 + 0.2 * self._env_mid)
                size = base_size * (0.8 + 0.4 * sin(t * 1.4 + i + j))

                hue = (int(t * 26) + i * 11 + j * 13) % 360
                edge = QColor.fromHsv(hue, 240, 255, int(80 + 100 * self._env_hi))
                fill = QColor.fromHsv((hue + 30) % 360, 200, 160, int(40 + 80 * self._env_hi))

                p.save()
                p.translate(cx, cy)
                p.rotate(sin(t * 0.6 + i * 0.5 + j) * 10 * self._env_mid)

                top, front = self._cube(size)
                p.setPen(QPen(edge, 1.6))
                p.setBrush(QBrush(fill))
                p.drawPath(front)
                p.drawPath(top)
                p.restore()

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
