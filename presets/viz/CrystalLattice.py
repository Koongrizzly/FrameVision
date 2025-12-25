from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
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

def _env_step(env, target, up=0.55, down=0.20):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

def _is_playing(bands, rms, eps=1e-4):
    if rms is not None and rms > 0.002:
        return True
    if not bands:
        return False
    # treat max-band as a simple "activity" signal
    return max(bands) > 0.002

@register_visualizer
class CrystalLattice(BaseVisualizer):
    display_name = "Crystal Lattice"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "CrystalLattice")))

    def _step_time(self, playing: bool, intensity: float):
        # assume ~60 fps; keep motion stable even if paint() is called faster/slower
        if playing:
            self._tick += (1.0 / 60.0) * (0.35 + 1.65 * max(0.0, min(1.0, intensity)))

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6 * (rms or 0.0), 0.62, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.58, 0.20)
        self._env_hi = _env_step(self._env_hi, hi, 0.60, 0.22)

        playing = _is_playing(bands, rms)
        intensity = max(self._env_lo, self._env_mid, self._env_hi)
        self._step_time(playing, intensity)


        p.fillRect(r, QColor(3, 5, 10))
        cx, cy = w * 0.5, h * 0.5
        size = min(w, h) * (0.38 + 0.12 * self._env_mid)
        rot = self._tick * (0.55 + 1.1 * self._env_mid)

        # lattice points in 3 rings
        rings = 3
        pts = []
        for R in range(1, rings + 1):
            m = 10 + R * 6
            rad = size * (0.25 + 0.25 * R)
            for i in range(m):
                u = i / float(m)
                a = 2 * pi * u + rot * (0.9 + 0.15 * R)
                bob = sin(self._tick * (0.9 + 0.25 * R) + i * 0.6) * (0.07 + 0.10 * self._env_hi)
                x = cx + cos(a) * rad * (1.0 + bob)
                y = cy + sin(a) * rad * (0.92 + 0.08 * sin(self._tick + u * 3.0))
                pts.append((x, y, R, u))

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        # connect near neighbors by angle proximity
        for i in range(len(pts)):
            x1, y1, R1, u1 = pts[i]
            for j in range(i + 1, len(pts)):
                x2, y2, R2, u2 = pts[j]
                if abs(u1 - u2) < 0.09 and abs(R1 - R2) <= 1:
                    d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if d < size * 0.55:
                        hue = int((175 + 140 * (u1 + u2) + 60 * self._env_hi) % 360)
                        alpha = int(14 + 85 * (0.25 + 0.75 * self._env_mid))
                        col = QColor.fromHsv(hue, 140, 255, alpha)
                        p.setPen(QPen(col, 1))
                        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # nodes
        for x, y, Rn, u in pts:
            hue = int((190 + 220 * u + 90 * self._env_hi) % 360)
            col = QColor.fromHsv(hue, 200, 255, int(18 + 90 * self._env_mid))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(x, y), 2 + Rn, 2 + Rn)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

