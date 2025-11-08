
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
class TriCrystalMesh(BaseVisualizer):
    display_name = "Tri Crystal Mesh"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4*rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.22)

        p.fillRect(r, QColor(5, 6, 9))
        cols, rows = 18, 12
        sx, sy = w/(cols-1), h/(rows-1)

        p.setPen(Qt.NoPen)
        for j in range(rows-1):
            for i in range(cols-1):
                # jittered corners for facet look
                def pt(ix, iy):
                    x = ix*sx + 8*sin(0.8*t + 0.4*iy + 0.3*ix) * (1+0.8*self._env_mid)
                    y = iy*sy + 8*cos(0.7*t + 0.5*ix + 0.2*iy) * (1+0.8*self._env_lo)
                    return QPointF(x, y)

                p1 = pt(i, j); p2 = pt(i+1,j); p3 = pt(i, j+1); p4 = pt(i+1, j+1)

                # triangle A
                hue = int((140 + 70*self._env_hi + 3*(i+j)) % 360)
                col = QColor.fromHsv(hue, 180, 230, 160)
                p.setBrush(col)
                path = QPainterPath(p1); path.lineTo(p2); path.lineTo(p3); path.closeSubpath()
                p.drawPath(path)

                # triangle B
                hue2 = int((hue + 20 + 10*sin(t + i)) % 360)
                col2 = QColor.fromHsv(hue2, 180, 240, 140)
                path2 = QPainterPath(p2); path2.lineTo(p4); path2.lineTo(p3); path2.closeSubpath()
                p.setBrush(col2); p.drawPath(path2)
