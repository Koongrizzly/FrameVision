from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n = len(bands)
    a = max(1, n//6); b = max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.6, down=0.24):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class PixelFire(BaseVisualizer):
    display_name = "Pixel Fire"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._grid = []
        self._cols = self._rows = 0

    def _init(self, w, h):
        self._cols = 40
        self._rows = 26
        self._grid = [[0.0 for _ in range(self._cols)] for _ in range(self._rows)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        if not self._grid:
            self._init(w, h)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 4, 8))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.8*rms, 0.72, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        # Seed bottom row from bass
        base_energy = min(1.0, self._env_lo*1.6)
        for x in range(self._cols):
            jitter = random()*0.4
            self._grid[self._rows-1][x] = min(1.0, base_energy + jitter)

        # Diffuse upwards (simple cellular fire)
        for y in range(self._rows-2, -1, -1):
            for x in range(self._cols):
                s = self._grid[y+1][x]
                if x > 0:
                    s += self._grid[y+1][x-1]
                if x < self._cols-1:
                    s += self._grid[y+1][x+1]
                self._grid[y][x] = max(0.0, s/3.1 - 0.02)

        cell_w = w / float(self._cols)
        cell_h = h / float(self._rows)

        p.setPen(Qt.NoPen)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for y in range(self._rows):
            for x in range(self._cols):
                v = self._grid[y][x]
                if v <= 0.01:
                    continue
                # Map value to fire–like HSV
                hue = 30 - int(24*v)  # orange -> yellow
                sat = 220
                val = int(120 + 135*v*(0.6 + 0.4*self._env_hi))
                col = QColor.fromHsv(hue, sat, val, int(80 + 150*v))
                p.setBrush(QBrush(col))
                p.drawRect(QRectF(x*cell_w, y*cell_h, cell_w+1, cell_h+1))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
