
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
class FlowRibbons(BaseVisualizer):
    display_name = "Flow Ribbons"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4*rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.55, 0.22)

        p.fillRect(r, QColor(3, 3, 7))

        ribbons = 5
        steps = 60
        base_thick = h*0.05*(0.7 + 0.6*self._env_lo)

        p.setPen(Qt.NoPen)
        for k in range(ribbons):
            ybase = h * (0.2 + 0.15*k) + 20*sin(0.7*t + k)
            thick = base_thick * (1.0 - 0.12*k)
            hue = int((100 + 30*k + 160*self._env_hi) % 360)
            col = QColor.fromHsv(hue, 200, 255, 90+int(30*self._env_mid))
            p.setBrush(QBrush(col))

            path = QPainterPath()
            path.moveTo(0, ybase)
            for i in range(1, steps+1):
                x = (i/steps)*w
                y = ybase + thick*0.5*sin(1.3*t + 0.2*i + k) + thick*0.3*sin(0.5*t + 0.7*i + 2*k)*self._env_mid
                path.lineTo(x, y)
            # close thickness
            for i in range(steps, -1, -1):
                x = (i/steps)*w
                y = ybase + thick + thick*0.35*cos(0.6*t + 0.25*i + k)
                path.lineTo(x, y)
            path.closeSubpath()
            p.drawPath(path)
