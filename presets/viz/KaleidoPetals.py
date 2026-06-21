from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient, QPolygonF
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

def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _drive_from_audio(bands, rms, lo_w=0.65, mid_w=0.85, hi_w=0.55):
    lo, mid, hi = _split(bands)
    drive = lo_w * lo + mid_w * mid + hi_w * hi + 0.6 * rms
    drive = drive / (1.0 + 0.9 * drive)
    return _clamp01(drive)

@register_visualizer
class KaleidoPetals(BaseVisualizer):
    display_name = "Kaleido Petals"
    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._spin = 0.0
        self._pulse = 0.0
        self._prev_t = None

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        drive = _drive_from_audio(bands, rms, 0.70, 0.75, 0.65)
        self._env = _env_step(self._env, drive, 0.66, 0.30)

        if self._prev_t is None:
            self._prev_t = t
        dt = max(0.0, min(0.08, t - self._prev_t))
        self._prev_t = t
        if drive > 0.02:
            self._spin += dt * (0.25 + 2.8 * self._env)
            self._pulse += dt * (0.9 + 5.0 * self._env)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 3, 7))

        cx, cy = r.center().x(), r.center().y()
        R = min(w, h) * (0.42 + 0.10 * self._env)

        petals = 18
        layers = 3

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(layers):
            fj = j / max(1, layers - 1)
            for i in range(petals):
                a0 = (i / petals) * 2 * pi + self._spin * (0.6 + 0.35 * fj)
                a1 = a0 + (pi / petals) * (0.9 + 0.2 * sin(self._pulse + i))
                # petal geometry
                r0 = R * (0.22 + 0.10 * fj)
                r1 = R * (0.98 - 0.18 * fj) * (0.78 + 0.22 * (0.5 + 0.5 * sin(self._pulse * 0.9 + i * 0.7)))
                pth = QPainterPath()
                pth.moveTo(QPointF(cx + cos(a0) * r0, cy + sin(a0) * r0))
                # control points
                c1 = QPointF(cx + cos(a0) * (r0 + r1 * 0.35), cy + sin(a0) * (r0 + r1 * 0.35))
                c2 = QPointF(cx + cos(a1) * (r0 + r1 * 0.35), cy + sin(a1) * (r0 + r1 * 0.35))
                tip = QPointF(cx + cos((a0 + a1) * 0.5) * r1, cy + sin((a0 + a1) * 0.5) * r1)
                pth.cubicTo(c1, QPointF(tip.x() * 0.8 + c2.x() * 0.2, tip.y() * 0.8 + c2.y() * 0.2), tip)
                pth.cubicTo(c2, QPointF(tip.x() * 0.8 + c1.x() * 0.2, tip.y() * 0.8 + c1.y() * 0.2), QPointF(cx + cos(a1) * r0, cy + sin(a1) * r0))
                pth.closeSubpath()

                hue = int((40 + 240 * (i / petals) + 120 * fj + 140 * self._env) % 360)
                alpha = int(18 + 70 * (1.0 - 0.35 * fj) * self._env)
                col = QColor.fromHsv(hue, int(170 + 70 * self._env), int(85 + 140 * (0.35 + 0.65 * self._env)), alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawPath(pth)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        # Center gem
        gem = QRadialGradient(QPointF(cx, cy), R * 0.22)
        gem.setColorAt(0.0, QColor(255, 255, 255, int(60 + 130 * self._env)))
        gem.setColorAt(1.0, QColor(30, 30, 60, 220))
        p.setBrush(QBrush(gem))
        p.setPen(QPen(QColor(255, 255, 255, int(40 + 140 * self._env)), 2))
        p.drawEllipse(QPointF(cx, cy), R * 0.11, R * 0.11)
