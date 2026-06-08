
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


@register_visualizer
class RetroReels(BaseVisualizer):
    display_name = "Retro Reels"

    def __init__(self):
        super().__init__()
        self._rot = 0.0
        self._env = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(6, 7, 12))

        val = rms
        if bands:
            val = max(rms, sum(bands)/len(bands))
        k_up, k_down = 0.36, 0.18
        self._env = (1-k_up)*self._env + k_up*val if val > self._env else (1-k_down)*self._env + k_down*val
        self._rot += (20 + 80*self._env)*(1/60.0)

        cx, cy = w*0.5, h*0.55
        body_w = w*0.7
        body_h = h*0.45

        # Tape body
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(18, 20, 30)))
        p.drawRoundedRect(QRectF(cx-body_w*0.5, cy-body_h*0.5, body_w, body_h), 20, 20)

        # Window
        p.setBrush(QBrush(QColor(10, 12, 22)))
        win_h = body_h*0.4
        win_rect = QRectF(cx-body_w*0.4, cy-win_h*0.5, body_w*0.8, win_h)
        p.drawRoundedRect(win_rect, 12, 12)

        # Reels
        reel_r = win_h*0.35
        left_x = win_rect.left() + win_rect.width()*0.3
        right_x = win_rect.right() - win_rect.width()*0.3
        y = win_rect.center().y()

        def draw_reel(x, angle, hue_offset):
            # Outer
            p.setBrush(QBrush(QColor(26, 28, 38)))
            p.setPen(QPen(QColor(8, 8, 16), 1.4))
            p.drawEllipse(QPointF(x, y), reel_r, reel_r)
            # Spokes
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            hue = (int(t*25) + hue_offset) % 360
            col = QColor.fromHsv(hue, 230, 255, int(80 + 150*self._env))
            p.setPen(QPen(col, 2.0))
            spokes = 5
            for i in range(spokes):
                a = angle*2*pi + i*2*pi/spokes
                x1 = x + cos(a)*reel_r*0.15
                y1 = y + sin(a)*reel_r*0.15
                x2 = x + cos(a)*reel_r*0.8
                y2 = y + sin(a)*reel_r*0.8
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            # Hub
            p.setBrush(QBrush(QColor(230, 230, 240)))
            p.drawEllipse(QPointF(x, y), reel_r*0.18, reel_r*0.18)

        draw_reel(left_x, self._rot, 0)
        draw_reel(right_x, -self._rot*1.05, 160)

        # Tape connection
        p.setPen(QPen(QColor(40, 42, 60), 3.0))
        p.drawLine(QPointF(left_x+reel_r*0.9, y),
                   QPointF(right_x-reel_r*0.9, y))

        # Edge screws
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(210, 210, 220)))
        s = min(body_w, body_h)*0.03
        for sx in (-1, 1):
            for sy in (-1, 1):
                p.drawEllipse(QPointF(cx+sx*body_w*0.42, cy+sy*body_h*0.35), s*0.5, s*0.5)
