
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class CrossfaderGlow(BaseVisualizer):
    display_name = "Crossfader Glow Bar"

    def __init__(self):
        super().__init__()
        self._pos = 0.5
        self._env = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(5, 6, 12))

        # Background bar
        bar_h = h * 0.08
        y = h * 0.5 - bar_h * 0.5
        radius = bar_h * 0.5

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(15, 18, 30)))
        p.drawRoundedRect(QRectF(w*0.08, y, w*0.84, bar_h), radius, radius)

        # Position from low vs high
        if bands:
            n = len(bands)
            a = max(1, n//4)
            lo = sum(bands[:a]) / a
            hi = sum(bands[-a:]) / a
            target = 0.5 if (lo+hi) <= 0 else lo / (lo + hi)
        else:
            target = 0.5

        k = 0.15
        self._pos = (1-k)*self._pos + k*target
        self._env = (1-0.25)*self._env + 0.25*(rms + max(bands) if bands else rms)

        # Glow
        x = w*0.08 + self._pos * w*0.84
        glow = min(1.0, 0.3 + self._env*3.0)
        rad = bar_h * (1.6 + glow*2.0)
        alpha = int(60 + 140*glow)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        grad_col = QColor.fromHsv(int((t*40) % 360), 230, 255, alpha)
        p.setBrush(QBrush(grad_col))
        p.drawEllipse(QPointF(x, y+bar_h*0.5), rad, rad)

        # Knob
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        knob_r = bar_h*0.55
        p.setBrush(QBrush(QColor(230, 230, 240)))
        p.setPen(QPen(QColor(10, 10, 16), 1.6))
        p.drawEllipse(QPointF(x, y+bar_h*0.5), knob_r, knob_r)
