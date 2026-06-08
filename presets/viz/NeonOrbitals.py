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
class NeonOrbitals(BaseVisualizer):
    display_name = "Neon Orbitals"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._phase = 0.0
        self._rng = Random(10101)

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        lo, mid, hi = _split(bands)
        act = _activity(bands, rms)
        target = 0.55 * mid + 0.35 * hi + 0.25 * lo + 0.65 * rms
        target = target / (1.0 + 0.7 * target)
        self._env = _env_step(self._env, target, 0.62, 0.24)

        # advance only when music is active
        if act > 0.005:
            self._phase += (0.9 + 3.0 * self._env) * (1.0 / 60.0)

        p.fillRect(r, QColor(3, 4, 8))

        cx, cy = w * 0.5, h * 0.5
        base = min(w, h) * 0.36
        rings = 5

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for k in range(rings):
            f = k / max(1, rings - 1)
            rad = base * (0.45 + 0.55 * f) * (1.0 + 0.20 * self._env)
            dots = 24 + 10 * k
            for i in range(dots):
                a = (i / dots) * 2 * pi + self._phase * (0.6 + 0.3 * k)
                wob = 0.45 + 0.55 * sin(self._phase * (1.2 + 0.2 * k) + i * 0.23)
                x = cx + cos(a) * rad * (0.85 + 0.15 * wob)
                y = cy + sin(a) * rad * (0.85 + 0.15 * wob)
                size = (2.0 + 5.0 * f) * (0.6 + 0.8 * self._env)
                hue = int((200 + 120 * f + 90 * sin(i * 0.12 + self._phase)) % 360)
                col = QColor.fromHsv(hue, int(170 + 70 * self._env), 255, int(35 + 75 * (1.0 - f)))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(x, y), size, size)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
