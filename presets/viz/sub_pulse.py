
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class SubPulse(BaseVisualizer):
    display_name = "Bass Cone Pulse"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._ring_phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 4, 8))

        lo = 0.0
        if bands:
            n = len(bands)
            a = max(1, int(n*0.35))
            lo = sum(bands[:a]) / a

        target = lo + rms*0.9
        k_up, k_down = 0.38, 0.2
        self._env_lo = (1-k_up)*self._env_lo + k_up*target if target > self._env_lo else (1-k_down)*self._env_lo + k_down*target
        self._ring_phase += 1.5 + 4*self._env_lo

        cx, cy = w*0.5, h*0.55
        radius = min(w, h)*0.32

        # Speaker housing
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(8, 10, 16)))
        p.drawRoundedRect(QRectF(cx-radius*1.4, cy-radius*1.4, radius*2.8, radius*2.8), 28, 28)

        # Cone base
        p.setBrush(QBrush(QColor(20, 22, 30)))
        p.drawEllipse(QPointF(cx, cy), radius*1.05, radius*1.05)

        # Moving cone
        depth = radius*0.4*self._env_lo
        p.setBrush(QBrush(QColor(230, 230, 238)))
        p.setPen(QPen(QColor(12, 12, 20), 2))
        p.drawEllipse(QPointF(cx, cy), radius*0.6+depth, radius*0.6+depth)

        # Center cap
        p.setBrush(QBrush(QColor(40, 42, 60)))
        p.drawEllipse(QPointF(cx, cy), radius*0.3+depth*0.4, radius*0.3+depth*0.4)

        # Radiating rings
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(5):
            d = i + self._ring_phase*0.12
            frac = (d % 5)/5.0
            rr = radius*0.7 + radius*0.6*frac
            alpha = int(20 + 120*(1-frac)*min(1.0, self._env_lo*3.0))
            hue = (int(t*20) + i*30) % 360
            col = QColor.fromHsv(hue, 200, 255, alpha)
            p.setPen(QPen(col, 2.0))
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
