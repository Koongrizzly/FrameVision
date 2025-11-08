
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
class TunnelGrid(BaseVisualizer):
    display_name = "Tunnel Grid"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._z = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.24)

        p.fillRect(r, QColor(0, 0, 0))
        cx, cy = w * 0.5, h * 0.5

        self._z += (120 + 280 * self._env_lo) * (1 / 60.0)

        depth_steps = 16
        for i in range(depth_steps):
            z = self._z + i * 60
            scale = 1.0 / (1.0 + z * 0.002)
            sx = w * scale
            sy = h * scale
            alpha = int(30 + 200 * scale)
            p.setPen(QPen(QColor(80, 160, 255, alpha), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(cx - sx / 2, cy - sy / 2, sx, sy))

        p.setPen(QPen(QColor(120, 220, 255, 180), 2))
        for angle in range(0, 360, 30):
            ang = angle * pi / 180.0
            x = cx + cos(ang) * w
            y = cy + sin(ang) * h
            p.drawLine(cx, cy, x, y)
