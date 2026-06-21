
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
class CrystalShards(BaseVisualizer):
    display_name = "Crystal Shards"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._shards = []

    def _spawn_shard(self, w, h, power):
        edge = int(random() * 4)
        if edge == 0:
            x, y = random() * w, -0.1 * h
            angle = random() * pi + pi / 2
        elif edge == 1:
            x, y = random() * w, h * 1.1
            angle = random() * pi - pi / 2
        elif edge == 2:
            x, y = -0.1 * w, random() * h
            angle = random() * pi - pi
        else:
            x, y = w * 1.1, random() * h
            angle = random() * pi

        size = (0.12 + 0.3 * power) * min(w, h)
        rot_speed = (random() - 0.5) * (0.5 + 4.0 * power)
        vx = cos(angle) * (40 + 220 * power)
        vy = sin(angle) * (40 + 220 * power)
        life = 3.5 + 3.0 * power

        self._shards.append(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "angle": random() * 2 * pi,
                "rot_speed": rot_speed,
                "size": size,
                "life": life,
                "age": 0.0,
            }
        )

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5 * rms, 0.6, 0.24)
        self._env_mid = _env_step(self._env_mid, mid + 0.1 * rms, 0.5, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        p.fillRect(r, QColor(2, 4, 10))

        power = max(self._env_lo, rms)
        spawn_rate = 1 + int(6 * power)
        for _ in range(spawn_rate):
            if random() < 0.35 + 0.6 * power and len(self._shards) < 120:
                self._spawn_shard(w, h, power)

        dt = 1 / 60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        alive = []
        for s in self._shards:
            s["age"] += dt
            if s["age"] > s["life"]:
                continue

            s["x"] += s["vx"] * dt
            s["y"] += s["vy"] * dt
            s["vx"] *= 0.995
            s["vy"] *= 0.995
            s["angle"] += s["rot_speed"] * dt

            age_t = s["age"] / max(0.001, s["life"])
            alpha_f = max(0.0, 1.0 - age_t * age_t)
            hue = int((190 + 80 * self._env_hi + 40 * age_t) % 360)
            sat = 150 + int(80 * self._env_mid)
            val = 200 + int(40 * (1.0 - age_t))
            alpha = int(40 + 215 * alpha_f)
            col_fill = QColor.fromHsv(hue, sat, val, alpha)
            col_edge = QColor.fromHsv(hue, min(255, sat + 40), 255, alpha)

            cx, cy = s["x"], s["y"]
            size = s["size"]
            angle = s["angle"]

            pts_local = [
                (0.0, -size * 0.7),
                (-size * 0.4, size * 0.5),
                (size * 0.4, size * 0.5),
            ]
            path = QPainterPath()
            for i, (lx, ly) in enumerate(pts_local):
                rx = cx + cos(angle) * lx - sin(angle) * ly
                ry = cy + sin(angle) * lx + cos(angle) * ly
                if i == 0:
                    path.moveTo(rx, ry)
                else:
                    path.lineTo(rx, ry)
            path.closeSubpath()

            p.setPen(QPen(col_edge, 1.5))
            p.setBrush(QBrush(col_fill))
            p.drawPath(path)
            alive.append(s)

        self._shards = alive
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
