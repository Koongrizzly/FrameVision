
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n // 6)
    b = max(a + 1, n // 2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b - a))
    hi = sum(bands[b:]) / max(1, (n - b))
    return lo, mid, hi


def _env_step(env, target, up=0.5, down=0.2):
    if target > env:
        return (1 - up) * env + up * target
    else:
        return (1 - down) * env + down * target



@register_visualizer
class PrismBeams(BaseVisualizer):
    display_name = "Prism Beams"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.fillRect(r, QColor(2, 4, 12))

        cx, cy = w * 0.5, h * 0.5

        prism_h = min(w, h) * (0.15 + 0.15 * self._env_lo)
        prism_w = prism_h * 0.7
        prism_rect = QRectF(cx - prism_w * 0.5, cy - prism_h * 0.5, prism_w, prism_h)
        prism_col = QColor(230, 230, 240, 255)
        p.setPen(QPen(QColor(140, 140, 160, 220), 2))
        p.setBrush(QBrush(prism_col))
        p.drawRoundedRect(prism_rect, 6, 6)

        beam_y = cy - prism_h * 0.05
        p.setPen(QPen(QColor(240, 240, 255, 230), 4))
        p.drawLine(0, beam_y, prism_rect.left(), beam_y)

        count = 7
        spread = 0.7 + 0.4 * self._env_mid
        length = w * (0.4 + 0.5 * (self._env_lo + self._env_mid))
        base_ang = -0.1 + 0.2 * sin(t * 0.5)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(count):
            f = (i / max(1, count - 1)) - 0.5
            ang = base_ang + f * spread
            x1 = prism_rect.right()
            y1 = beam_y
            x2 = x1 + cos(ang) * length
            y2 = y1 + sin(ang) * length

            hue = int((200 + 220 * (i / max(1, count - 1)) + 140 * self._env_hi) % 360)
            alpha = int(50 + 180 * (0.4 + 0.6 * self._env_mid))
            width = 6 + 10 * (1.0 - abs(f))
            col = QColor.fromHsv(hue, 230, 255, alpha)
            p.setPen(QPen(col, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(x1, y1, x2, y2)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
