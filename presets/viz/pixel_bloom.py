
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
class PixelBloom(BaseVisualizer):
    display_name = "Pixel Bloom"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._blooms = []

    def _spawn_bloom(self, w, h, power):
        cx = random() * w
        cy = random() * h
        max_r = (0.2 + 0.35 * power) * min(w, h)
        self._blooms.append(
            {
                "x": cx,
                "y": cy,
                "r": 0.0,
                "max_r": max_r,
                "speed": 140 + 260 * power,
            }
        )
        if len(self._blooms) > 40:
            self._blooms = self._blooms[-40:]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.fillRect(r, QColor(4, 4, 10))

        power = (self._env_mid * 0.6 + self._env_hi * 0.9 + rms * 0.4)
        if hi > 0.25 or rms > 0.22 or random() < 0.02 * power:
            self._spawn_bloom(w, h, min(1.5, power + self._env_lo))

        dt = 1 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        alive = []
        for b in self._blooms:
            b["r"] += b["speed"] * dt * (0.4 + 0.8 * self._env_mid)
            if b["r"] > b["max_r"]:
                continue
            alive.append(b)

            steps = 36
            for i in range(steps):
                ang = 2 * pi * i / steps
                jitter = 0.85 + 0.3 * sin(t * 2.3 + i)
                rr = b["r"] * jitter
                x = b["x"] + cos(ang) * rr
                y = b["y"] + sin(ang) * rr

                size = 8 + 10 * (1.0 - b["r"] / max(1.0, b["max_r"]))
                f = b["r"] / max(1.0, b["max_r"])
                alpha = int(25 + 180 * (1.0 - f))
                hue = int((210 + 80 * self._env_hi + 140 * f) % 360)
                col = QColor.fromHsv(hue, 220, 255, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawRect(QRectF(x - size * 0.5, y - size * 0.5, size, size))

        self._blooms = alive
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
