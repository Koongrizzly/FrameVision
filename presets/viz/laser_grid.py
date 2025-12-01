
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class LaserGrid(BaseVisualizer):
    display_name = "Laser Grid"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(1, 2, 6))

        val = rms
        if bands:
            val = max(rms, sum(bands)/len(bands))
        self._env = (1-0.32)*self._env + 0.32*val
        self._phase += 0.4 + 3.0*self._env

        floor_top = h*0.25
        floor_bottom = h
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(5, 6, 12)))
        p.drawRect(QRectF(0, floor_top, w, floor_bottom-floor_top))

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        rows = 12
        cols = 14

        for i in range(rows+1):
            f = i/float(rows)
            y = floor_top + (floor_bottom-floor_top)*f
            depth = f
            offset = sin(self._phase*0.2 + f*4.0)*w*0.03*self._env
            hue = (int(t*25) + i*10) % 360
            alpha = int(40 + 180*(1-f)*min(1.0, self._env*2.5))
            col = QColor.fromHsv(hue, 240, 255, alpha)
            p.setPen(QPen(col, 1.4 + 1.6*(1-f)))
            p.drawLine(QPointF(0+offset, y), QPointF(w-offset, y))

        for j in range(cols+1):
            f = j/float(cols)
            x = w*f
            hue = (int(t*30) + j*8) % 360
            alpha = int(30 + 200*min(1.0, self._env*2.5))
            col = QColor.fromHsv(hue, 220, 255, alpha)
            p.setPen(QPen(col, 1.2))
            bend = sin(self._phase*0.15 + f*5.0)*h*0.03*self._env
            p.drawLine(QPointF(x, floor_top), QPointF(x+bend, floor_bottom))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
