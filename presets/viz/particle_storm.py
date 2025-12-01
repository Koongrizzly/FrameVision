
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
class ParticleStorm(BaseVisualizer):
    display_name = "Particle Storm (Huge)"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._parts = []

    def _spawn(self, w, h, amount, power):
        # Spawn in a large disc around the center so the effect fills most of the viewport
        cx, cy = w * 0.5, h * 0.5
        radius = 0.25 * min(w, h)
        for _ in range(amount):
            ang = random() * 2 * pi
            r = radius * (random() ** 0.6)
            sx = cx + cos(ang) * r
            sy = cy + sin(ang) * r

            speed = (220 + 720 * power) * (0.7 + random())
            dir_ang = random() * 2 * pi
            self._parts.append(
                {
                    "x": sx,
                    "y": sy,
                    "vx": cos(dir_ang) * speed,
                    "vy": sin(dir_ang) * speed,
                    "a": 255,
                    "s": 8.0 + 12.0 * random(),
                }
            )
        if len(self._parts) > 2500:
            self._parts = self._parts[-2500:]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(3, 3, 8))

        # Much more eager spawning for a massive effect
        if lo > 0.12 or rms > 0.06:
            burst_power = self._env_lo + rms * 0.7
            count = int(140 + 260 * burst_power)
            self._spawn(w, h, count, burst_power)

        dt = 1 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        alive = []
        for a in self._parts:
            a["x"] += a["vx"] * dt
            a["y"] += a["vy"] * dt
            a["vx"] *= 0.988
            a["vy"] *= 0.988
            # Fade very slowly so the storm stays dense
            a["a"] = max(0, a["a"] - 1 - int(self._env_hi * 2))
            if 0 <= a["x"] < w and 0 <= a["y"] < h and a["a"] > 4:
                hue = int((200 + 80 * self._env_mid + 40 * random()) % 360)
                col = QColor.fromHsv(hue, 220, 255, a["a"])
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                s = a["s"]
                p.drawEllipse(QPointF(a["x"], a["y"]), s, s)
                alive.append(a)
        self._parts = alive
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
