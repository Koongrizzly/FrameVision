from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient
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

def _env_step(env, target, up=0.55, down=0.22):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

def _activity(bands, rms):
    # 0..~1, used to gate animation. When paused, bands/rms usually collapse to ~0.
    if not bands:
        return max(0.0, float(rms))
    avg = sum(bands) / max(1, len(bands))
    return max(float(rms), avg)

def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

@register_visualizer
class PrismShards(BaseVisualizer):
    display_name = "Prism Shards"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._spin = 0.0
        self._rng = Random(20207)
        self._shards = [(self._rng.random(), self._rng.random(), self._rng.random()*2*pi) for _ in range(42)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.45 * mid + 0.55 * hi + 0.25 * rms
        target = target / (1.0 + 0.8 * target)
        self._env = _env_step(self._env, target, 0.62, 0.22)

        act = _activity(bands, rms)
        if act > 0.005:
            self._spin += (0.8 + 4.0 * self._env) * (1.0 / 60.0)

        p.fillRect(r, QColor(4, 3, 8))
        cx, cy = w * 0.5, h * 0.5
        radius = min(w, h) * 0.46

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, (u, v, a0) in enumerate(self._shards):
            ang = a0 + self._spin * (0.8 + 0.4 * u)
            rr = radius * (0.15 + 0.85 * v) * (0.85 + 0.25 * self._env)
            x = cx + cos(ang) * rr
            y = cy + sin(ang) * rr

            # triangle shard
            size = (6 + 26 * u) * (0.45 + 0.90 * self._env)
            rot = ang + sin(self._spin + i * 0.3) * 0.8
            pth = QPainterPath()
            pth.moveTo(QPointF(x + cos(rot) * size, y + sin(rot) * size))
            pth.lineTo(QPointF(x + cos(rot + 2*pi/3) * size, y + sin(rot + 2*pi/3) * size))
            pth.lineTo(QPointF(x + cos(rot + 4*pi/3) * size, y + sin(rot + 4*pi/3) * size))
            pth.closeSubpath()

            hue = int((40 + 280 * u + 140 * self._env) % 360)
            alpha = int(25 + 95 * (0.35 + 0.65 * self._env))
            col = QColor.fromHsv(hue, 220, 255, alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawPath(pth)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
