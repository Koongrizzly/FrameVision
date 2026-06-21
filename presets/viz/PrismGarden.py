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
class PrismGarden(BaseVisualizer):
    display_name = "Prism Garden"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._tick = 0.0   # advances only while music is playing
        self._rng = Random(20251 + sum(map(ord, "PrismGarden")))

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


        p.fillRect(r, QColor(4, 4, 10))
        cx, cy = w * 0.5, h * 0.5
        base = min(w, h) * 0.42

        # soft radial glow
        rg = QRadialGradient(QPointF(cx, cy), base * 1.2)
        rg.setColorAt(0.0, QColor(20, 18, 40, 120))
        rg.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, rg)

        petals = 24
        layers = 3
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for L in range(layers):
            fL = L / max(1, layers - 1)
            rad = base * (0.35 + 0.22 * L) * (0.9 + 0.5 * self._env_mid)
            twist = self._tick * (0.55 + 0.2 * L)

            for i in range(petals):
                u = i / float(petals)
                a = 2 * pi * u + twist
                flare = 0.35 + 0.95 * (0.35 * self._env_lo + 0.65 * self._env_hi)
                px = cx + cos(a) * rad
                py = cy + sin(a) * rad

                # triangle shard
                tip = QPointF(px + cos(a) * rad * 0.45 * flare, py + sin(a) * rad * 0.45 * flare)
                left = QPointF(px + cos(a + 1.7) * rad * 0.22, py + sin(a + 1.7) * rad * 0.22)
                right = QPointF(px + cos(a - 1.7) * rad * 0.22, py + sin(a - 1.7) * rad * 0.22)

                path = QPainterPath()
                path.moveTo(tip)
                path.lineTo(left)
                path.lineTo(right)
                path.closeSubpath()

                hue = int((40 + 300 * u + 70 * fL + self._tick * 12) % 360)
                alpha = int(22 + 70 * (1.0 - fL) * (0.25 + 0.75 * self._env_mid))
                col = QColor.fromHsv(hue, int(170 + 60 * self._env_hi), 255, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawPath(path)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

