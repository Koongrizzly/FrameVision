
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class EQCityscape(BaseVisualizer):
    display_name = "EQ Cityscape"

    def __init__(self):
        super().__init__()
        self._bars = []

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 4, 9))

        # Prepare bands into ~40 bars
        if not bands:
            return

        target_count = 40
        n = len(bands)
        step = max(1, n//target_count)
        values = []
        for i in range(0, n, step):
            chunk = bands[i:i+step]
            values.append(sum(chunk)/max(1, len(chunk)))

        if not self._bars:
            self._bars = [0.0]*len(values)
        elif len(self._bars) != len(values):
            self._bars = [0.0]*len(values)

        k_up, k_down = 0.4, 0.2
        for i, v in enumerate(values):
            v = min(1.0, v*3.0 + rms*0.6)
            e = self._bars[i]
            self._bars[i] = (1-k_up)*e + k_up*v if v > e else (1-k_down)*e + k_down*v

        ground = h*0.82
        max_height = h*0.7
        width = w / max(1, len(self._bars))

        # Skyline reflection background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(8, 8, 18)))
        p.drawRect(QRectF(0, ground, w, h-ground))

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, v in enumerate(self._bars):
            bar_h = max_height * v
            x = i*width
            hue = (int(t*20) + i*3) % 360
            alpha = int(50 + 180*v)
            col = QColor.fromHsv(hue, 230, 255, alpha)
            rect = QRectF(x+1, ground-bar_h, width-2, bar_h)
            p.setBrush(QBrush(col))
            p.drawRect(rect)

            # Window lights
            step_y = max(4.0, bar_h/12.0)
            p.setBrush(QBrush(QColor(255, 250, 220, int(60+120*v))))
            for yy in [ground - step_y*k for k in range(2, int(bar_h/step_y))]:
                if random() < 0.3 + 0.4*v:
                    ww = (width-4)*0.5
                    wx = x+2 + random()*(width-4-ww)
                    p.drawRect(QRectF(wx, yy-step_y*0.35, ww, step_y*0.3))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
