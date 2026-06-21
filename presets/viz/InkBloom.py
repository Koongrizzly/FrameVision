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
class InkBloom(BaseVisualizer):
    display_name = "Ink Bloom"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "InkBloom")))

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


        p.fillRect(r, QColor(6, 6, 12))
        cx, cy = w * 0.5, h * 0.52
        base = min(w, h) * 0.33

        petals = 11
        rings = 6
        swell = 0.55 + 0.95 * self._env_lo

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(rings):
            f = j / max(1, rings - 1)
            rr = base * (0.55 + 0.22 * j) * swell
            wob = 0.06 + 0.12 * self._env_mid + 0.06 * f

            path = QPainterPath()
            first = True
            steps = 140
            for i in range(steps + 1):
                u = i / float(steps)
                a = 2 * pi * u
                petals_wob = sin(a * petals + self._tick * (0.9 + 0.2 * j)) * (0.55 + 0.45 * self._env_hi)
                ripple = sin(self._tick * (0.7 + 0.3 * f) - a * 2.0) * 0.6
                rad = rr * (1.0 + wob * (petals_wob + 0.4 * ripple))
                x = cx + cos(a) * rad
                y = cy + sin(a) * rad * (0.86 + 0.14 * sin(self._tick + f))
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
            path.closeSubpath()

            hue = int((280 + 90 * f + self._tick * 10) % 360)
            alpha = int(18 + 70 * (1.0 - f) * (0.25 + 0.75 * self._env_lo))
            col = QColor.fromHsv(hue, int(160 + 70 * self._env_hi), int(120 + 120 * self._env_mid), alpha)
            p.setPen(QPen(QColor.fromHsv(hue, 30, 255, alpha), 2))
            p.setBrush(QBrush(col))
            p.drawPath(path)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

