from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
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
class SolarFlareHalo(BaseVisualizer):
    display_name = "Solar Flare Halo"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2, 3, 8))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.62, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.66, 0.25)
        self._phase += 0.12 + 0.9 * self._env_mid

        cx, cy = w * 0.5, h * 0.5

        bg = QRadialGradient(cx, cy, min(w, h) * 0.7)
        bg.setColorAt(0.0, QColor(8, 10, 26))
        bg.setColorAt(0.6, QColor(6, 4, 20))
        bg.setColorAt(1.0, QColor(0, 0, 0))
        p.fillRect(r, bg)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        base_r = min(w, h) * (0.18 + 0.15 * self._env_lo)

        gl = QRadialGradient(cx, cy, base_r * 1.8)
        gl.setColorAt(0.0, QColor(255, 230, 160, int(150 + 80 * self._env_hi)))
        gl.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, gl)

        rings = 7
        for i in range(rings):
            d = (i + 1) / float(rings + 1)
            radius = base_r + d * base_r * 2.2
            hue = (30 + i * 20 + int(self._phase * 15)) % 360
            col = QColor.fromHsv(hue, 240, 255, int(80 + 130 * (1.0 - d)))
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            spokes = 40
            p.setPen(QPen(col, 1))
            for k in range(spokes):
                aa = (k / float(spokes)) * 2 * pi + self._phase * 0.4 + i * 0.2
                jitter = 0.25 * self._env_hi
                seg = radius * (1.3 + 0.3 * sin(aa * 5 + self._phase))
                x1 = cx + cos(aa) * radius
                y1 = cy + sin(aa) * radius
                x2 = cx + cos(aa + jitter) * seg
                y2 = cy + sin(aa + jitter) * seg
                p.drawLine(x1, y1, x2, y2)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
