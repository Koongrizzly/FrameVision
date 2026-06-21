
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
class CityEqualizer(BaseVisualizer):
    display_name = "City Equalizer"

    def __init__(self):
        super().__init__()
        self.cols = 48
        self._env = [0.0] * self.cols
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid + 0.2 * rms, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(4, 6, 12))

        n_bands = max(1, len(bands))
        for i in range(self.cols):
            idx0 = int(i / self.cols * n_bands)
            idx1 = min(n_bands, idx0 + max(1, n_bands // self.cols))
            band_val = sum(bands[idx0:idx1]) / max(1, (idx1 - idx0)) if bands else 0.0
            target = 0.05 + 2.0 * band_val + 1.2 * rms
            self._env[i] = _env_step(self._env[i], target, 0.5, 0.18)

        col_w = w / self.cols
        ground_h = h * 0.15
        horizon_y = h - ground_h

        p.fillRect(QRectF(0, 0, w, horizon_y), QColor(2, 8, 20))

        for i in range(self.cols):
            strength = min(1.8, self._env[i])
            b_h = (0.15 + 0.7 * strength) * h
            x = i * col_w
            y = horizon_y - b_h
            hue = int((210 + 40 * self._env_hi + 40 * (i / self.cols)) % 360)
            base_col = QColor.fromHsv(hue, 180, 220, 230)
            p.setPen(Qt.NoPen)
            p.setBrush(base_col)
            p.drawRect(QRectF(x + 1, y, col_w - 2, b_h))

            win_h = 6
            win_w = col_w * 0.18
            rows = int(b_h / (win_h * 2.2))
            cols = max(1, int((col_w - win_w) / (win_w * 1.6)))
            for ry in range(rows):
                wy = horizon_y - (ry + 1) * win_h * 2.2
                if wy < 0:
                    continue
                for cx in range(cols):
                    wx = x + win_w * 0.6 + cx * win_w * 1.6
                    phase = sin(t * 3.0 + i * 0.7 + ry * 1.3 + cx * 2.1)
                    if phase * strength + self._env_mid * 0.5 < 0.1:
                        continue
                    a = int(80 + 160 * max(0.0, strength))
                    window_col = QColor(255, 240, 200, a)
                    p.setBrush(window_col)
                    p.drawRect(QRectF(wx, wy, win_w, win_h))

        p.setBrush(QColor(5, 10, 18))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(0, horizon_y, w, ground_h))
