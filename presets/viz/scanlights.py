
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class ScanLights(BaseVisualizer):
    display_name = "Moving Scanlights"

    def __init__(self):
        super().__init__()
        self._phases = [0.0, 1.3, 2.1, 3.7]
        self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2, 3, 8))

        hi = 0.0
        if bands:
            n = len(bands)
            a = max(1, int(n*0.65))
            hi = sum(bands[a:]) / max(1, (n-a))

        self._env_hi = (1-0.36)*self._env_hi + 0.36*(hi + rms*0.6)

        # Dance floor gradient
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(10, 10, 20)))
        p.drawRect(QRectF(0, h*0.45, w, h*0.55))

        # Scanlights
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, ph in enumerate(self._phases):
            phase = t*0.6 + ph
            x = (0.1 + 0.8*(0.5+0.5*sin(phase))) * w
            hue = (int(t*30) + i*70) % 360
            alpha = int(40 + 180*min(1.0, self._env_hi*2.5))
            col = QColor.fromHsv(hue, 240, 255, alpha)
            p.setPen(QPen(col, w*0.018))

            # Beam from top to floor
            y1 = 0
            y2 = h*0.9
            p.drawLine(QPointF(x, y1), QPointF(x, y2))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
