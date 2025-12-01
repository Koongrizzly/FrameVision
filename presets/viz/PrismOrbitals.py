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

def _env_step(env, target, up=0.6, down=0.22):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

@register_visualizer
class PrismOrbitals(BaseVisualizer):
    display_name = "Prism Orbitals"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def _poly(self, r):
        path = QPainterPath()
        for i in range(6):
            a = i * pi / 3.0 + pi / 6.0
            x = cos(a) * r
            y = sin(a) * r
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 4, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)

        self._phase += 0.09 + 0.7 * self._env_mid

        cx, cy = w * 0.5, h * 0.5

        p.save()
        p.translate(cx, cy)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        layers = 7
        for i in range(layers):
            d = (i + 1) / float(layers + 1)
            radius = d * min(w, h) * (0.4 + 0.3 * self._env_lo)
            scale = 0.6 + d * 0.8
            angle = self._phase * (0.6 + 0.2 * i)
            hue = (int(t * 40) + i * 32) % 360
            col = QColor.fromHsv(hue, 230, 255, int(80 + 150 * d * self._env_hi))

            p.save()
            p.rotate(angle * 40)
            p.scale(scale, scale)
            path = self._poly(30 + 18 * self._env_mid + i * 6)
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.restore()

            orb_angle = angle * 1.4 + i
            ox = cos(orb_angle) * radius * 0.4
            oy = sin(orb_angle) * radius * 0.4
            glow = QRadialGradient(ox, oy, 16 + 12 * self._env_hi)
            c1 = QColor(col.red(), col.green(), col.blue(), 220)
            c2 = QColor(col.red(), col.green(), col.blue(), 0)
            glow.setColorAt(0.0, c1)
            glow.setColorAt(1.0, c2)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(ox, oy), 10 + 8 * self._env_hi, 10 + 8 * self._env_hi)

        p.restore()
