
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
class ElectricMaze(BaseVisualizer):
    display_name = "Electric Maze"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self.seed = random()*1000.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4*rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(0, 0, 0))

        grid = 28
        gw = w/grid
        gh = h/grid

        # draw glowing grid segments that open/close like a maze
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(grid+1):
            for i in range(grid+1):
                # decide whether to draw right/down segment
                phase = (i*0.37 + j*0.23 + self.seed)
                open_right = (sin(phase + 1.2*t + 6*self._env_mid) > 0.1 - 0.6*self._env_lo)
                open_down  = (cos(phase + 1.1*t + 5*self._env_hi) > 0.1 - 0.6*self._env_lo)

                x, y = i*gw, j*gh
                hue = int((180 + 100*self._env_hi) % 360)
                col = QColor.fromHsv(hue, 200, 255, 110)
                p.setPen(QPen(col, 2))
                if open_right and i < grid:
                    p.drawLine(x, y, x+gw, y)
                if open_down and j < grid:
                    p.drawLine(x, y, x, y+gh)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
