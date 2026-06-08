
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
class NetworkMesh(BaseVisualizer):
    display_name = "Network Mesh"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._pts = []
        self._count = 70

    def _ensure_points(self, w, h):
        if self._pts and len(self._pts) == self._count:
            return
        self._pts = []
        for _ in range(self._count):
            self._pts.append(
                {
                    "x": random() * w,
                    "y": random() * h,
                    "vx": (random() - 0.5) * 40.0,
                    "vy": (random() - 0.5) * 40.0,
                }
            )

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        self._ensure_points(w, h)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.4 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid + 0.1 * rms, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(1, 4, 10))

        dt = 1 / 60.0
        speed_boost = 1.0 + 2.0 * (self._env_lo + rms)
        for pt in self._pts:
            pt["vx"] += (random() - 0.5) * 10.0 * dt
            pt["vy"] += (random() - 0.5) * 10.0 * dt
            pt["x"] += pt["vx"] * dt * speed_boost
            pt["y"] += pt["vy"] * dt * speed_boost

            if pt["x"] < 0:
                pt["x"] = 0
                pt["vx"] = abs(pt["vx"])
            elif pt["x"] > w:
                pt["x"] = w
                pt["vx"] = -abs(pt["vx"])
            if pt["y"] < 0:
                pt["y"] = 0
                pt["vy"] = abs(pt["vy"])
            elif pt["y"] > h:
                pt["y"] = h
                pt["vy"] = -abs(pt["vy"])

        max_dist = 0.24 * max(w, h) * (1.0 + 0.8 * self._env_mid)
        max_dist2 = max_dist * max_dist

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        n = len(self._pts)
        for i in range(n):
            xi, yi = self._pts[i]["x"], self._pts[i]["y"]
            for j in range(i + 1, n):
                xj, yj = self._pts[j]["x"], self._pts[j]["y"]
                dx = xj - xi
                dy = yj - yi
                d2 = dx * dx + dy * dy
                if d2 > max_dist2:
                    continue
                d = d2 ** 0.5
                f = 1.0 - d / max_dist
                alpha = int(15 + 200 * f * (0.3 + self._env_mid + self._env_hi))
                if alpha <= 5:
                    continue
                hue = int((190 + 80 * self._env_hi + 40 * f) % 360)
                col = QColor.fromHsv(hue, 170, 255, alpha)
                p.setPen(QPen(col, 1.2))
                p.drawLine(xi, yi, xj, yj)

        for pt in self._pts:
            pulse = 0.5 + 0.5 * sin(t * 3.0 + pt["x"] * 0.01 + pt["y"] * 0.01)
            size = 2.0 + 2.0 * pulse * (0.6 + self._env_hi)
            alpha = int(80 + 150 * pulse)
            hue = int((210 + 60 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 140, 255, alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(pt["x"], pt["y"]), size, size)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
