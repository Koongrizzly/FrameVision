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
class KaleidoShards(BaseVisualizer):
    display_name = "Kaleido Shards"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "KaleidoShards")))

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


        p.fillRect(r, QColor(3, 3, 7))
        cx, cy = w * 0.5, h * 0.5
        base = min(w, h) * 0.45

        segments = 9
        lines = 40
        spin = self._tick * (0.55 + 1.1 * self._env_hi)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for s in range(segments):
            a0 = (2 * pi / segments) * s + spin
            for i in range(lines):
                u = i / float(lines - 1)
                rr1 = base * (0.08 + 0.85 * u) * (0.9 + 0.4 * self._env_mid)
                rr2 = rr1 + base * (0.08 + 0.12 * self._env_lo) * (0.2 + 0.8 * sin(self._tick * 1.3 + u * 7))

                a = a0 + (u - 0.5) * (0.35 + 0.8 * self._env_hi)
                x1 = cx + cos(a) * rr1
                y1 = cy + sin(a) * rr1
                x2 = cx + cos(a + 0.12 * sin(self._tick + s)) * rr2
                y2 = cy + sin(a + 0.12 * sin(self._tick + s)) * rr2

                hue = int((20 + 330 * u + 55 * s + 80 * self._env_hi) % 360)
                alpha = int(14 + 90 * (0.25 + 0.75 * self._env_hi))
                col = QColor.fromHsv(hue, 210, 255, alpha)
                p.setPen(QPen(col, 1))
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # subtle center gem
        g = QRadialGradient(QPointF(cx, cy), base * 0.35)
        g.setColorAt(0.0, QColor(255, 255, 255, int(30 + 60 * self._env_mid)))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.fillRect(r, g)

