
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


def _split_three(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n//3)
    b = max(a+1, 2*n//3)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

@register_visualizer
class ThreeBandKnobs(BaseVisualizer):
    display_name = "3-Band EQ Knobs"

    def __init__(self):
        super().__init__()
        self._lo = self._mid = self._hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 6, 12))

        lo, mid, hi = _split_three(bands)
        k_up, k_down = 0.35, 0.18

        def env_step(env, target):
            return (1-k_up)*env + k_up*target if target > env else (1-k_down)*env + k_down*target

        self._lo = env_step(self._lo, lo + rms*0.6)
        self._mid = env_step(self._mid, mid + rms*0.4)
        self._hi = env_step(self._hi, hi + rms*0.3)

        cx = w*0.5
        cy = h*0.55
        spacing = w*0.22
        radius = min(w, h)*0.11

        bands_vals = [self._lo, self._mid, self._hi]
        colors = [
            QColor.fromHsv(140, 220, 200),
            QColor.fromHsv(210, 220, 220),
            QColor.fromHsv(20, 220, 230),
        ]

        p.setPen(Qt.NoPen)
        for i, val in enumerate(bands_vals):
            x = cx + (i-1) * spacing
            base = QColor(12, 14, 24)
            p.setBrush(QBrush(base))
            p.drawEllipse(QPointF(x, cy), radius*1.25, radius*1.25)

            # Halo
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            halo = min(1.0, 0.2 + val*3.0)
            p.setBrush(QBrush(QColor(colors[i].red(), colors[i].green(), colors[i].blue(), int(60+120*halo))))
            p.drawEllipse(QPointF(x, cy), radius*1.4, radius*1.4)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)

            # Knob
            p.setBrush(QBrush(QColor(230, 230, 240)))
            p.setPen(QPen(QColor(18, 18, 28), 1.4))
            p.drawEllipse(QPointF(x, cy), radius, radius)

            # Indicator line
            ang = -140 + 220*min(1.0, val*2.4)
            ang_rad = ang * pi / 180.0
            inner = radius*0.3
            outer = radius*0.9
            x1 = x + cos(ang_rad)*inner
            y1 = cy + sin(ang_rad)*inner
            x2 = x + cos(ang_rad)*outer
            y2 = cy + sin(ang_rad)*outer
            p.setPen(QPen(colors[i], 2.0))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
