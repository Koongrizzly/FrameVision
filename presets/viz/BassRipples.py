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
class BassRipples(BaseVisualizer):
    display_name = "Bass Ripples"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.95 * lo + 0.35 * rms
        target = target / (1.0 + 0.9 * target)
        self._env = _env_step(self._env, target, 0.72, 0.22)

        act = _activity(bands, rms)
        if act > 0.005:
            self._phase += (0.7 + 3.2 * self._env) * (1.0 / 60.0)

        p.fillRect(r, QColor(1, 2, 6))

        cx, cy = w * 0.5, h * 0.5
        maxr = min(w, h) * 0.48

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        rings = 16
        for i in range(rings):
            f = i / max(1, rings - 1)
            rr = maxr * (0.08 + 0.92 * f)
            wave = sin(self._phase * 2.1 - f * 5.0)
            thick = 2 + int(4 * self._env * (1.0 - f))
            hue = int((200 + 120 * wave + 120 * f) % 360)
            alpha = int(20 + 120 * (1.0 - f) * (0.35 + 0.65 * self._env))
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, alpha), thick))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), rr * (0.92 + 0.08 * wave), rr * (0.92 + 0.08 * wave))

        # center throb
        center = maxr * 0.10 * (0.6 + 1.2 * self._env)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor.fromHsv(int(320 * self._env) % 360, 230, 255, int(90 + 130 * self._env))))
        p.drawEllipse(QPointF(cx, cy), center, center)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
