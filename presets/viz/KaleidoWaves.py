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
class KaleidoWaves(BaseVisualizer):
    display_name = "Kaleido Waves"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._phi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        lo, mid, hi = _split(bands)
        target = 0.45 * lo + 0.55 * mid + 0.35 * hi + 0.35 * rms
        target = target / (1.0 + 0.75 * target)
        self._env = _env_step(self._env, target, 0.62, 0.22)

        act = _activity(bands, rms)
        if act > 0.005:
            self._phi += (0.7 + 3.8 * self._env) * (1.0 / 60.0)

        p.fillRect(r, QColor(3, 2, 8))

        cx, cy = w * 0.5, h * 0.5
        R = min(w, h) * 0.47
        petals = 10

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for k in range(petals):
            a0 = (k / petals) * 2 * pi
            path = QPainterPath()
            steps = 80
            for i in range(steps + 1):
                u = i / steps
                a = a0 + u * 2 * pi / petals
                wav = sin(self._phi * 2.2 + u * 7.0 + k * 0.6)
                rr = R * (0.35 + 0.65 * u) * (0.85 + 0.15 * wav) * (0.75 + 0.55 * self._env)
                x = cx + cos(a) * rr
                y = cy + sin(a) * rr
                if i == 0:
                    path.moveTo(QPointF(x, y))
                else:
                    path.lineTo(QPointF(x, y))
            hue = int((260 + 110 * sin(self._phi + k * 0.4) + 120 * self._env) % 360)
            alpha = int(18 + 55 * (0.35 + 0.65 * self._env))
            pen = QPen(QColor.fromHsv(hue, 230, 255, alpha), 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawPath(path)

        # center glow
        glow = QRadialGradient(QPointF(cx, cy), R * 0.45)
        glow.setColorAt(0.0, QColor.fromHsv(int(330 * self._env) % 360, 200, 255, int(65 + 90 * self._env)))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, cy), R * 0.45, R * 0.45)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
