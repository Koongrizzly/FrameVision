from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QRadialGradient
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
class CrystalBloom(BaseVisualizer):
    display_name = "Crystal Bloom"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rot = 0.0

    def _petal(self, length, width_scale):
        path = QPainterPath()
        path.moveTo(0, 0)
        path.quadTo(length * 0.15, -length * 0.25 * width_scale, length * 0.6, 0)
        path.quadTo(length * 0.15, length * 0.25 * width_scale, 0, 0)
        path.closeSubpath()
        return path

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 6, 12))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.62, 0.23)
        self._env_mid = _env_step(self._env_mid, mid, 0.58, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)
        self._rot += 0.05 + 0.7 * self._env_mid

        cx, cy = w * 0.5, h * 0.5
        p.save()
        p.translate(cx, cy)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        rings = 4
        for ring in range(rings):
            petals = 10 + ring * 4
            radius = 22 + ring * 40 + 18 * self._env_lo
            base_len = 40 + ring * 12 + 30 * self._env_mid
            for i in range(petals):
                ang = (360.0 / petals) * i + self._rot * (15 + ring * 8)
                p.save()
                p.rotate(ang)
                p.translate(radius, 0)
                hue = (int(t * 30) + ring * 30 + i * 4) % 360
                col = QColor.fromHsv(hue, 210, 255, int(60 + 150 * self._env_hi))
                path = self._petal(base_len, 0.6 + 0.3 * sin(t * 0.7 + i))
                p.setPen(QPen(col, 1.6))
                grad = QRadialGradient(0, 0, base_len)
                grad.setColorAt(0.0, QColor(255, 255, 255, 80))
                grad.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
                p.setBrush(QBrush(grad))
                p.drawPath(path)
                p.restore()

        p.restore()

        core = QRadialGradient(cx, cy, min(w, h) * 0.25)
        core.setColorAt(0.0, QColor(230, 255, 255, int(180 * self._env_hi)))
        core.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, core)
