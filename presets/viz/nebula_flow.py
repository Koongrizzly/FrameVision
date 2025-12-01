
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
class NebulaFlow(BaseVisualizer):
    display_name = "Nebula Flow"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._blobs = []
        self._init_blobs()

    def _init_blobs(self):
        self._blobs = []
        for _ in range(34):
            self._blobs.append(
                {
                    "x": random(),
                    "y": random(),
                    "r": 0.08 + 0.18 * random(),
                    "h": 180 + 80 * random(),
                    "dx": (random() - 0.5) * 0.02,
                    "dy": (random() - 0.5) * 0.02,
                }
            )

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.25)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.3)

        p.fillRect(r, QColor(1, 2, 8))

        dt = 1 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for b in self._blobs:
            speed_boost = 0.3 + 1.4 * self._env_mid
            b["x"] += b["dx"] * dt * speed_boost
            b["y"] += b["dy"] * dt * speed_boost

            if b["x"] < -0.3 or b["x"] > 1.3 or b["y"] < -0.3 or b["y"] > 1.3:
                b["x"] = random()
                b["y"] = random()
                b["dx"] = (random() - 0.5) * 0.02
                b["dy"] = (random() - 0.5) * 0.02
                b["r"] = 0.08 + 0.18 * random()
                b["h"] = 180 + 80 * random()

            radius = b["r"] * min(w, h) * (0.9 + 0.8 * self._env_lo)
            cx = b["x"] * w
            cy = b["y"] * h

            hue = int((b["h"] + 60 * self._env_hi) % 360)
            strength = 0.45 + 0.55 * self._env_mid
            alpha = int(40 + 170 * strength)
            val = int(80 + 150 * strength)
            col = QColor.fromHsv(hue, 160, val, alpha)

            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
