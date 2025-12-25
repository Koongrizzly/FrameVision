
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
class PlasmaMetaballs(BaseVisualizer):
    display_name = "Plasma Metaballs"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self.blobs = [{'x':0.3,'y':0.4,'vx':0.18,'vy':0.22,'r':0.14},
                      {'x':0.7,'y':0.5,'vx':-0.16,'vy':0.15,'r':0.12},
                      {'x':0.5,'y':0.7,'vx':0.12,'vy':-0.18,'r':0.16}]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.fillRect(r, QColor(2, 3, 7))

        # Update blob positions in normalized space [0,1]
        dt = 1/60.0
        for b in self.blobs:
            b['x'] += b['vx'] * dt * (0.5 + 0.8 * (0.5 + self._env_lo))
            b['y'] += b['vy'] * dt * (0.5 + 0.8 * (0.5 + self._env_mid))
            b['r'] *= (0.995 + 0.02 * self._env_hi)
            if b['x'] < 0.1 or b['x'] > 0.9: b['vx'] *= -1
            if b['y'] < 0.1 or b['y'] > 0.9: b['vy'] *= -1
            b['r'] = max(0.08, min(b['r'], 0.22))

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setPen(Qt.NoPen)

        for i, b in enumerate(self.blobs):
            cx, cy, rr = b['x'] * w, b['y'] * h, b['r'] * min(w, h) * (0.9 + 0.7*self._env_lo)
            hue = int((180 + 50*i + 200*self._env_hi) % 360)
            for k in range(6, 0, -1):
                a = int(18 + 32 * k)
                col = QColor.fromHsv(hue, 220, 255, a)
                p.setBrush(QBrush(col))
                s = rr * (k/6.0)
                p.drawEllipse(QPointF(cx, cy), s, s)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
