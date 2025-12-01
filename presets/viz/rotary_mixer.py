
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class RotaryMixer(BaseVisualizer):
    display_name = "Rotary DJ Mixer"

    def __init__(self):
        super().__init__()
        self._env = [0.0]*4

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(6, 7, 11))

        # Split into 4 rough bands
        vals = [0.0]*4
        if bands:
            n = len(bands)
            step = max(1, n//4)
            for i in range(4):
                s = i*step
                e = n if i == 3 else min(n, s+step)
                vals[i] = sum(bands[s:e]) / max(1, (e-s))

        k_up, k_down = 0.3, 0.16
        def env_step(env, target):
            return (1-k_up)*env + k_up*target if target > env else (1-k_down)*env + k_down*target

        for i in range(4):
            self._env[i] = env_step(self._env[i], vals[i] + rms*0.4)

        cx = w*0.5
        top = h*0.35
        spacing = w*0.18
        radius = min(w, h)*0.09

        p.setPen(Qt.NoPen)
        for i, val in enumerate(self._env):
            x = cx + (i-1.5)*spacing
            y = top
            base_rect = QRectF(x-spacing*0.4, y-radius*1.8, spacing*0.8, radius*3.5)
            p.setBrush(QBrush(QColor(12, 13, 20)))
            p.drawRoundedRect(base_rect, 10, 10)

            # Halo
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            hue = (int(t*25) + i*60) % 360
            halo_col = QColor.fromHsv(hue, 220, 255, int(60 + 130*min(1.0, val*2.5)))
            p.setBrush(QBrush(halo_col))
            p.drawEllipse(QPointF(x, y), radius*1.4, radius*1.4)

            # Knob
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setBrush(QBrush(QColor(230, 230, 240)))
            p.setPen(QPen(QColor(18, 18, 26), 1.4))
            p.drawEllipse(QPointF(x, y), radius, radius)

            # Indicator
            ang = -120 + 240*min(1.0, val*2.2)
            ang_rad = ang * pi / 180.0
            inner = radius*0.25
            outer = radius*0.95
            x1 = x + cos(ang_rad)*inner
            y1 = y + sin(ang_rad)*inner
            x2 = x + cos(ang_rad)*outer
            y2 = y + sin(ang_rad)*outer
            p.setPen(QPen(halo_col, 2.0))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
