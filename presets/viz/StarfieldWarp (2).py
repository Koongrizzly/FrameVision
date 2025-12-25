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
class StarfieldWarp(BaseVisualizer):
    display_name = "Starfield Warp"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._z = 0.0
        self._rng = Random(80808)
        self._stars = [(self._rng.random()*2-1, self._rng.random()*2-1, 0.2 + 0.8*self._rng.random()) for _ in range(260)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        target = 0.35 * lo + 0.55 * mid + 0.25 * rms
        target = target / (1.0 + 0.9 * target)
        self._env = _env_step(self._env, target, 0.62, 0.22)

        act = _activity(bands, rms)
        if act > 0.005:
            self._z += (0.6 + 5.5 * self._env) * (1.0 / 60.0)

        p.fillRect(r, QColor(0, 0, 5))
        cx, cy = w * 0.5, h * 0.5
        scale = min(w, h) * 0.55

        p.setRenderHint(QPainter.Antialiasing, False)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, (sx, sy, z0) in enumerate(self._stars):
            z = (z0 + self._z) % 1.0
            # perspective
            k = 1.0 / (0.15 + 0.85 * z)
            x = cx + sx * scale * k
            y = cy + sy * scale * k
            if x < -40 or x > w + 40 or y < -40 or y > h + 40:
                continue
            size = 1 + int(2 * (1.0 - z) + 2 * self._env)
            hue = int((210 + 90 * self._env + i * 0.3) % 360)
            alpha = int(12 + 70 * (1.0 - z) * (0.35 + 0.65 * self._env))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor.fromHsv(hue, 160, 255, alpha)))
            p.drawEllipse(QPointF(x, y), size, size)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
