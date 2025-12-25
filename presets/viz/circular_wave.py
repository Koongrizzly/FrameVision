
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class CircularWaveRing(BaseVisualizer):
    display_name = "Circular Waveform Ring"

    def __init__(self):
        super().__init__()
        self._env = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 5, 10))

        val = rms
        if bands:
            val = max(rms, sum(bands)/len(bands))
        k_up, k_down = 0.32, 0.16
        self._env = (1-k_up)*self._env + k_up*val if val > self._env else (1-k_down)*self._env + k_down*val

        cx, cy = w*0.5, h*0.55
        radius = min(w, h)*0.3

        # Base ring
        p.setPen(QPen(QColor(30, 34, 60), 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Samples around circle
        if not bands:
            return

        n = len(bands)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, v in enumerate(bands):
            frac = i/float(n)
            ang = frac*2*pi + t*0.2
            amp = min(1.0, v*3.0 + self._env)
            inner = radius*0.75
            outer = radius*(0.75 + 0.45*amp)
            x1 = cx + cos(ang)*inner
            y1 = cy + sin(ang)*inner
            x2 = cx + cos(ang)*outer
            y2 = cy + sin(ang)*outer

            hue = (int(t*40) + int(frac*360)) % 360
            alpha = int(40 + 190*amp)
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, alpha), 1.4))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
