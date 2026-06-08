from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QRadialGradient
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

def _env_step(env, target, up=0.55, down=0.22):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

@register_visualizer
class NeonGridOrbit(BaseVisualizer):
    display_name = "Neon Grid Orbit"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def _draw_grid(self, p, w, h):
        horizon = h * 0.45
        p.save()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(4, 8, 24))
        grad.setColorAt(0.4, QColor(6, 10, 40))
        grad.setColorAt(1.0, QColor(2, 2, 10))
        p.fillRect(0, 0, w, h, grad)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(80, 140, 255, 120), 1))

        rows = 26
        for i in range(rows):
            t = i / (rows - 1)
            y = horizon + (h - horizon) * t * t
            p.drawLine(0, y, w, y)

        cols = 18
        cx = w * 0.5
        for i in range(-cols, cols + 1):
            x = i / float(cols) * w * 0.9
            p.drawLine(cx + x * 0.08, horizon, cx + x, h)
        p.restore()

    def _draw_orbits(self, p, w, h):
        cx, cy = w * 0.5, h * 0.45
        p.save()
        p.translate(cx, cy)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        rings = 6
        for i in range(rings):
            d = (i + 1) / float(rings + 1)
            rad = d * min(w, h) * (0.35 + 0.2 * self._env_lo)
            col = QColor.fromHsv((int(self._phase * 30) + i * 40) % 360,
                                 220,
                                 255,
                                 int(80 + 120 * (1.0 - d)))
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), rad, rad * 0.7)

            angle = self._phase * (1.2 + 0.4 * i) + i * 0.6
            px = cos(angle) * rad
            py = sin(angle) * rad * 0.7
            glow = QRadialGradient(px, py, 18 + 16 * self._env_hi)
            glow.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(), 220))
            glow.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            s = 8 + 10 * self._env_hi
            p.drawEllipse(QPointF(px, py), s, s)

        p.restore()

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)
        self._phase += (0.15 + 0.8 * self._env_mid)

        self._draw_grid(p, w, h)
        self._draw_orbits(p, w, h)
