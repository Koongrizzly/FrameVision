
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class DualSpeakers(BaseVisualizer):
    display_name = "Dual Speakers"

    def __init__(self):
        super().__init__()
        self._env_left = 0.0
        self._env_right = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 5, 10))

        # Rough split of spectrum: left=low+mid, right=mid+hi
        lo = mid = hi = 0.0
        if bands:
            n = len(bands)
            a = max(1, n//3)
            b = max(a+1, 2*n//3)
            lo = sum(bands[:a]) / a
            mid = sum(bands[a:b]) / max(1, (b-a))
            hi = sum(bands[b:]) / max(1, (n-b))
        left_target = lo + mid*0.7 + rms*0.5
        right_target = mid*0.5 + hi + rms*0.5

        k_up, k_down = 0.32, 0.18
        def env(e, t):
            return (1-k_up)*e + k_up*t if t > e else (1-k_down)*e + k_down*t

        self._env_left = env(self._env_left, left_target)
        self._env_right = env(self._env_right, right_target)

        cx = w*0.5
        cy = h*0.55
        spacing = w*0.26
        body_w = w*0.22
        body_h = h*0.56

        def draw_speaker(x, env, hue_offset):
            rect = QRectF(x-body_w*0.5, cy-body_h*0.5, body_w, body_h)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(10, 12, 18)))
            p.drawRoundedRect(rect, 18, 18)

            # Two cones
            for j in range(2):
                yy = rect.top() + body_h*(0.3 if j == 0 else 0.7)
                rr = min(body_w, body_h)*0.14*(1.0 + env*0.2)
                p.setBrush(QBrush(QColor(24, 26, 34)))
                p.drawEllipse(QPointF(x, yy), rr*1.1, rr*1.1)
                p.setBrush(QBrush(QColor(230, 230, 238)))
                p.setPen(QPen(QColor(14, 14, 24), 1.6))
                p.drawEllipse(QPointF(x, yy), rr*0.8+rr*0.5*env, rr*0.8+rr*0.5*env)
                p.setBrush(QBrush(QColor(40, 42, 58)))
                p.drawEllipse(QPointF(x, yy), rr*0.4+rr*0.3*env, rr*0.4+rr*0.3*env)

            # Glow strip
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            hue = (int(t*30) + hue_offset) % 360
            alpha = int(40 + 150*min(1.0, env*3.0))
            col = QColor.fromHsv(hue, 230, 255, alpha)
            p.setBrush(QBrush(col))
            g_rect = QRectF(rect.left()+body_w*0.15, rect.center().y()-body_h*0.02, body_w*0.7, body_h*0.04)
            p.drawRoundedRect(g_rect, 8, 8)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        draw_speaker(cx-spacing*0.5, self._env_left, 0)
        draw_speaker(cx+spacing*0.5, self._env_right, 140)
