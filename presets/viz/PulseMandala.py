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
class PulseMandala(BaseVisualizer):
    display_name = "Pulse Mandala"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._theta = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7 * rms, 0.72, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.62, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.28)

        act = _activity(bands, rms)
        if act > 0.005:
            self._theta += (0.8 + 3.6 * self._env_mid + 1.2 * self._env_hi) * (1.0 / 60.0)

        p.fillRect(r, QColor(2, 2, 7))

        cx, cy = w * 0.5, h * 0.5
        R = min(w, h) * 0.44

        # spokes
        spokes = 72
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        amp = 0.35 * self._env_lo + 0.55 * self._env_mid + 0.45 * self._env_hi
        for i in range(spokes):
            k = i / spokes
            ang = self._theta * 0.8 + k * 2 * pi
            length = R * (0.30 + 0.70 * amp) * (0.85 + 0.15 * sin(self._theta * 2.0 + i * 0.25))
            x1 = cx + cos(ang) * R * 0.12
            y1 = cy + sin(ang) * R * 0.12
            x2 = cx + cos(ang) * (R * 0.12 + length)
            y2 = cy + sin(ang) * (R * 0.12 + length)
            hue = int((60 + 260 * k + 120 * self._env_hi) % 360)
            alpha = int(22 + 110 * (0.35 + 0.65 * self._env_hi))
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, alpha), 2))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # rings
        rings = 10
        for j in range(rings):
            f = j / max(1, rings - 1)
            rr = R * (0.22 + 0.78 * f) * (0.92 + 0.22 * amp * sin(self._theta * 1.8 - j))
            hue = int((200 + 140 * f + 120 * self._env_mid) % 360)
            alpha = int(16 + 55 * (1.0 - f) * (0.35 + 0.65 * self._env_mid))
            p.setPen(QPen(QColor.fromHsv(hue, 210, 255, alpha), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
