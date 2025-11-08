
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
class MoireWeave(BaseVisualizer):
    display_name = "Moire Weave"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.3*rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.22)

        p.fillRect(r, QColor(0,0,0))
        p.setPen(QPen(QColor(200,200,255,60), 1))

        # two rotating line families
        spacing = max(6.0, min(w,h) / 40.0)
        ang1 = (10 + 50*self._env_mid) * t * pi/180.0
        ang2 = -ang1 * (0.6 + 0.4*self._env_hi)

        def draw_family(angle, phase):
            ca, sa = cos(angle), sin(angle)
            limit = int(max(w,h)*1.5)
            i = -limit
            while i <= limit:
                x0 = w/2 + ca*i + phase*20
                y0 = h/2 + sa*i
                x1 = x0 - sa*2000
                y1 = y0 + ca*2000
                x2 = x0 + sa*2000
                y2 = y0 - ca*2000
                p.drawLine(x1, y1, x2, y2)
                i += spacing

        draw_family(ang1, self._env_lo)
        p.setPen(QPen(QColor(120,220,255,70), 1))
        draw_family(ang2, self._env_hi)
