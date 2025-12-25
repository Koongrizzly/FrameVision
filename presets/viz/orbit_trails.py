
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
class OrbitTrails(BaseVisualizer):
    display_name = "Orbit Trails"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._angles = [random() * 2 * pi for _ in range(6)]
        self._history = [[] for _ in range(6)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.fillRect(r, QColor(2, 3, 9))

        cx, cy = w * 0.5, h * 0.5
        max_r = min(w, h) * 0.38 * (1.0 + 0.2 * self._env_lo)

        dt = 1.0 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for idx in range(len(self._angles)):
            base_r = max_r * (0.25 + 0.12 * idx)
            speed = (0.3 + 0.08 * idx) * (1.0 + 1.2 * self._env_mid)
            wobble = 0.3 + 0.3 * sin(t * 0.6 + idx)
            self._angles[idx] += speed * dt * 2 * pi

            a = self._angles[idx]
            ex = base_r * (1.0 + 0.4 * wobble)
            ey = base_r * (1.0 - 0.3 * wobble)
            x = cx + cos(a) * ex
            y = cy + sin(a) * ey

            hist = self._history[idx]
            hist.append((x, y))
            max_len = 40 + int(60 * self._env_lo)
            if len(hist) > max_len:
                del hist[0]

            hue = int((180 + 40 * idx + 160 * self._env_hi) % 360)

            if len(hist) > 1:
                for i in range(1, len(hist)):
                    f = i / max(1, len(hist) - 1)
                    alpha = int(30 + 190 * (1.0 - f) * (0.5 + 0.5 * self._env_mid))
                    col = QColor.fromHsv(hue, 210, 255, alpha)
                    p.setPen(QPen(col, 2))
                    x1, y1 = hist[i - 1]
                    x2, y2 = hist[i]
                    p.drawLine(x1, y1, x2, y2)

            col_head = QColor.fromHsv(hue, 230, 255, 230)
            p.setPen(Qt.NoPen)
            p.setBrush(col_head)
            s = 4 + 3 * self._env_hi
            p.drawEllipse(QPointF(x, y), s, s)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
