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
class NeonRain(BaseVisualizer):
    display_name = "Neon Rain"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "NeonRain")))

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


        p.fillRect(r, QColor(2, 2, 8))

        # persistent drops
        if not hasattr(self, "_drops"):
            self._drops = []
            cols = 22
            for i in range(cols):
                x = (i + 0.5) / cols
                for _ in range(6):
                    y = self._rng.random()
                    sp = 0.25 + self._rng.random() * 1.4
                    sz = 0.7 + self._rng.random() * 1.6
                    self._drops.append([x, y, sp, sz, self._rng.random()])

        speed = 0.6 + 2.0 * self._env_mid + 2.4 * self._env_hi
        tw = self._tick

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for d in self._drops:
            x, y0, sp, sz, seed = d
            y = (y0 + tw * sp * speed * 0.25) % 1.2 - 0.1
            px = r.left() + x * w
            py = r.top() + y * h

            # streak length reacts to highs
            L = (12 + 60 * self._env_hi) * sz
            hue = int((120 + 220 * seed + 90 * self._env_hi + tw * 10) % 360)
            alpha = int(35 + 150 * (0.25 + 0.75 * self._env_hi))
            col = QColor.fromHsv(hue, 220, 255, alpha)
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(px, py - L), QPointF(px, py + L * 0.4))

            # head glow
            g = QRadialGradient(QPointF(px, py), 10 + 14 * self._env_hi)
            g.setColorAt(0.0, QColor.fromHsv(hue, 180, 255, int(40 + 140 * self._env_hi)))
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(QPointF(px, py), 10 + 14 * self._env_hi, 10 + 14 * self._env_hi)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

