
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
class GlitchTiles(BaseVisualizer):
    display_name = "Glitch Tiles"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.3 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.25)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(2, 3, 8))

        cols = 6
        rows = 4
        tile_w = w / cols
        tile_h = h / rows

        glitch_phase = t * (1.0 + 1.5 * self._env_mid)
        max_shift = 0.03 * min(w, h) * (0.3 + 1.5 * self._env_hi)
        max_scale = 0.08 * (0.2 + self._env_hi)

        for iy in range(rows):
            fy = iy / max(1, rows - 1)
            for ix in range(cols):
                fx = ix / max(1, cols - 1)

                base_x = ix * tile_w
                base_y = iy * tile_h

                n = sin(glitch_phase + fx * 3.7 + fy * 5.1)
                n2 = sin(glitch_phase * 1.7 + fx * 2.3 - fy * 4.0)
                shift_x = n * max_shift
                shift_y = n2 * max_shift

                scale = 1.0 + max_scale * sin(glitch_phase * 2.0 + fx * 8.0 + fy * 11.0)

                w_scaled = tile_w * scale
                h_scaled = tile_h * scale
                x = base_x + (tile_w - w_scaled) * 0.5 + shift_x
                y = base_y + (tile_h - h_scaled) * 0.5 + shift_y

                base_hue = 200 + 40 * fx + 80 * fy
                hue = int((base_hue + 80 * self._env_hi) % 360)
                val = int(80 + 160 * (0.3 + 0.7 * self._env_mid))
                alpha = int(140 + 100 * (abs(n) * 0.6 + abs(n2) * 0.4))

                col = QColor.fromHsv(hue, 180, val, alpha)
                p.setPen(QPen(QColor(5, 10, 18, int(alpha * 0.6)), 2))
                p.setBrush(QBrush(col))
                p.drawRect(QRectF(x, y, w_scaled, h_scaled))
