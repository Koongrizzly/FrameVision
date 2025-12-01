
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


def _split_lo_hi(bands):
    if not bands:
        return 0.0, 0.0
    n = len(bands)
    a = max(1, int(n*0.55))
    lo = sum(bands[:a]) / a
    hi = sum(bands[a:]) / max(1, (n-a))
    return lo, hi

@register_visualizer
class DualTurntables(BaseVisualizer):
    display_name = "Dual Turntables"

    def __init__(self):
        super().__init__()
        self._rot_a = 0.0
        self._rot_b = 0.0
        self._env_lo = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 5, 10))

        lo, hi = _split_lo_hi(bands)
        k_up, k_down = 0.36, 0.18
        def env(e, target):
            return (1-k_up)*e + k_up*target if target > e else (1-k_down)*e + k_down*target
        self._env_lo = env(self._env_lo, lo + rms*0.7)
        self._env_hi = env(self._env_hi, hi + rms*0.5)

        self._rot_a += (20 + 70*self._env_lo)*(1/60.0)
        self._rot_b -= (22 + 80*self._env_hi)*(1/60.0)

        radius = min(w*0.35, h*0.42)
        cy = h*0.55
        cx_a = w*0.28
        cx_b = w*0.72

        def draw_deck(cx, rot, env, hue_offset):
            # Deck base
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(10, 12, 18)))
            p.drawRoundedRect(QRectF(cx-radius*1.3, cy-radius*1.3, radius*2.6, radius*2.0), 18, 18)

            # Disk
            p.setBrush(QBrush(QColor(12, 14, 22)))
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            # Grooves
            p.setPen(QPen(QColor(40, 46, 70), 1))
            rings = 18
            for i in range(rings):
                d = i/float(rings)
                rr = radius*(0.15 + 0.82*d)
                p.drawEllipse(QPointF(cx, cy), rr, rr)

            # Colored arcs
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            spokes = 42
            for i in range(spokes):
                k = i/float(spokes)
                ang = rot*2*pi + k*2*pi
                amp = 0.2 + 0.8*env
                length = radius*(0.22 + 0.35*amp*sin(t*3 + i*0.25))
                x1 = cx + cos(ang)*radius*0.25
                y1 = cy + sin(ang)*radius*0.25
                x2 = cx + cos(ang)*(radius*0.25 + length)
                y2 = cy + sin(ang)*(radius*0.25 + length)
                hue = (int(t*20) + i*5 + hue_offset) % 360
                col = QColor.fromHsv(hue, 220, 255, int(40 + 150*env))
                p.setPen(QPen(col, 1.7))
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Label
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            label_col = QColor.fromHsv((int(t*30)+hue_offset) % 360, 220, 255, 220)
            p.setBrush(QBrush(label_col))
            p.setPen(QPen(QColor(8, 8, 16), 1.6))
            p.drawEllipse(QPointF(cx, cy), radius*0.22, radius*0.22)
            p.setBrush(QBrush(QColor(8, 8, 16)))
            p.drawEllipse(QPointF(cx, cy), radius*0.05, radius*0.05)

        draw_deck(cx_a, self._rot_a, self._env_lo, 0)
        draw_deck(cx_b, self._rot_b, self._env_hi, 140)
