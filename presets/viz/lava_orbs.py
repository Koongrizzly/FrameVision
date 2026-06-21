
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
class LavaOrbs(BaseVisualizer):
    display_name = "Lava Orbs"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._orbs = []
        self._init_done = False

    def _ensure_orbs(self, w, h):
        if self._init_done or w <= 0 or h <= 0:
            return
        self._init_done = True
        count = 6
        for _ in range(count):
            self._orbs.append(
                {
                    "x": random() * w,
                    "y": random() * h,
                    "vx": (random() - 0.5) * 40,
                    "vy": (random() - 0.5) * 40,
                    "r": (0.18 + 0.08 * random()) * min(w, h),
                }
            )

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        self._ensure_orbs(w, h)
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.55, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        bg = QColor(5, 3, 12)
        p.fillRect(r, bg)

        dt = 1 / 60.0

        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for o in self._orbs:
            speed_boost = 1.0 + 0.9 * self._env_lo
            o["x"] += o["vx"] * dt * speed_boost
            o["y"] += o["vy"] * dt * speed_boost

            cx, cy = w * 0.5, h * 0.5
            o["vx"] += (cx - o["x"]) * 0.02 * dt
            o["vy"] += (cy - o["y"]) * 0.02 * dt

            if o["x"] - o["r"] < 0 or o["x"] + o["r"] > w:
                o["vx"] *= -0.9
            if o["y"] - o["r"] < 0 or o["y"] + o["r"] > h:
                o["vy"] *= -0.9

            base_r = o["r"] * (0.7 + 0.5 * (0.4 + self._env_lo + 0.3 * sin(t * 1.2)))
            layers = 4
            for layer in range(layers):
                f = layer / max(1, layers - 1)
                rr = base_r * (0.45 + f * 0.8)
                hue = int((20 + 40 * self._env_lo + 140 * self._env_hi + 30 * layer) % 360)
                sat = 220
                val = 200 + int(40 * (1.0 - f))
                alpha = int(40 + 90 * (1.0 - f) * (0.6 + 0.4 * self._env_mid))
                col = QColor.fromHsv(hue, sat, val, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(o["x"], o["y"]), rr, rr)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
