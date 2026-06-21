
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
class StarBursts(BaseVisualizer):
    display_name = "Star Bursts"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._stars = []

    def _spawn_star(self, w, h, power):
        x = random() * w
        y = random() * h
        max_age = 0.8 + 1.5 * power
        size = (0.04 + 0.14 * power) * min(w, h)
        self._stars.append({"x": x, "y": y, "age": 0.0, "max_age": max_age, "size": size})
        if len(self._stars) > 40:
            self._stars = self._stars[-40:]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.26)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.28)

        p.fillRect(r, QColor(1, 2, 8))

        power = max(rms, hi)
        if power > 0.18 or random() < 0.015 * (self._env_lo + self._env_mid + self._env_hi):
            self._spawn_star(w, h, min(1.5, power * 2.0))

        dt = 1 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        alive = []
        for s in self._stars:
            s["age"] += dt
            if s["age"] >= s["max_age"]:
                continue
            alive.append(s)

            phase = s["age"] / max(1e-3, s["max_age"])
            radius = s["size"] * (0.2 + 1.4 * phase)
            core = s["size"] * (0.15 + 0.25 * (1.0 - phase))

            rays = 12
            for i in range(rays):
                a = 2 * pi * i / rays
                jitter = 0.65 + 0.5 * sin(t * 3.0 + i * 1.7)
                r1 = core * 0.4
                r2 = radius * jitter
                x1 = s["x"] + cos(a) * r1
                y1 = s["y"] + sin(a) * r1
                x2 = s["x"] + cos(a) * r2
                y2 = s["y"] + sin(a) * r2

                hue = int((40 + 220 * phase + 160 * self._env_hi) % 360)
                alpha = int(40 + 180 * (1.0 - phase))
                col = QColor.fromHsv(hue, 200, 255, alpha)
                p.setPen(QPen(col, 2 + 3 * (1.0 - phase), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.drawLine(x1, y1, x2, y2)

            hue_core = int((30 + 80 * self._env_hi) % 360)
            alpha_core = int(180 * (1.0 - 0.7 * phase))
            core_col = QColor.fromHsv(hue_core, 60, 255, alpha_core)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(core_col))
            p.drawEllipse(QPointF(s["x"], s["y"]), core, core)

        self._stars = alive
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
