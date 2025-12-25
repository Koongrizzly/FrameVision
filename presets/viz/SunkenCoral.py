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
class SunkenCoral(BaseVisualizer):
    display_name = "Sunken Coral"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "SunkenCoral")))

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


        # Deep ocean background
        bg = QLinearGradient(r.left(), r.bottom(), r.right(), r.top())
        bg.setColorAt(0.0, QColor(1, 6, 14))
        bg.setColorAt(1.0, QColor(4, 2, 10))
        p.fillRect(r, bg)

        base_y = r.bottom() - h * 0.12
        trunk_x = r.center().x()
        sway = sin(self._tick * (0.7 + 1.0 * self._env_mid)) * (8 + 24 * self._env_lo)

        branches = 10
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(branches):
            f = i / max(1, branches - 1)
            x0 = trunk_x + (f - 0.5) * w * 0.55
            y0 = base_y
            height = h * (0.28 + 0.36 * (0.35 + self._env_mid)) * (0.7 + 0.6 * (1.0 - abs(f - 0.5) * 2))
            bend = sway * (0.6 + 0.8 * (1.0 - f))

            path = QPainterPath()
            path.moveTo(x0, y0)
            # 3 control points
            c1 = QPointF(x0 + bend * 0.7, y0 - height * 0.35)
            c2 = QPointF(x0 - bend * 0.4, y0 - height * 0.70)
            tip = QPointF(x0 + bend * 0.9, y0 - height)
            path.cubicTo(c1, c2, tip)

            hue = int((320 - 220 * f + 80 * self._env_hi + self._tick * 8) % 360)
            alpha = int(30 + 120 * (0.25 + 0.75 * self._env_lo))
            col = QColor.fromHsv(hue, 210, 255, alpha)
            p.setPen(QPen(col, 4 + int(3 * (0.2 + self._env_mid))))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

            # polyp sparks
            if self._env_hi > 0.12:
                sparks = 3 + int(8 * self._env_hi)
                for k in range(sparks):
                    tt = (k / max(1, sparks - 1))
                    px = (1 - tt) ** 2 * x0 + 2 * (1 - tt) * tt * c2.x() + tt ** 2 * tip.x()
                    py = (1 - tt) ** 2 * y0 + 2 * (1 - tt) * tt * c2.y() + tt ** 2 * tip.y()
                    rr = 2 + 6 * self._env_hi
                    g = QRadialGradient(QPointF(px, py), rr * 6)
                    g.setColorAt(0.0, QColor.fromHsv(hue, 180, 255, int(55 + 120 * self._env_hi)))
                    g.setColorAt(1.0, QColor(0, 0, 0, 0))
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(g))
                    p.drawEllipse(QPointF(px, py), rr * 2, rr * 2)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

