from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient, QFont
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


def _env_step(env, target, up=0.55, down=0.2):
    if target > env:
        return (1.0 - up) * env + up * target
    return (1.0 - down) * env + down * target


class _PlayGatedTime:
    """Keeps an internal animation time that only advances when audio energy is present."""
    def __init__(self):
        self.t_prev = None
        self.anim_t = 0.0

    def step(self, t_now: float, energy: float) -> float:
        if self.t_prev is None:
            self.t_prev = t_now
            return 0.0
        dt = t_now - self.t_prev
        self.t_prev = t_now
        # Clamp to avoid huge jumps if the UI hiccups.
        if dt < 0.0:
            dt = 0.0
        if dt > 0.05:
            dt = 0.05
        if energy > 0.001:
            self.anim_t += dt
        return dt

@register_visualizer
class LotusMandala(BaseVisualizer):
    display_name = "Lotus Mandala"

    def __init__(self):
        super().__init__()
        self._tg = _PlayGatedTime()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7 * rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.25)

        energy = max(rms, (lo + mid + hi) / 3.0)
        self._tg.step(t, energy)
        tt = self._tg.anim_t

        p.fillRect(r, QColor(3, 2, 8))
        cx, cy = w * 0.5, h * 0.5
        R = min(w, h) * 0.42

        # Soft center glow
        g = QRadialGradient(cx, cy, R * 0.9)
        g.setColorAt(0.0, QColor(180, 120, 255, int(50 + 90 * self._env_lo)))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), R * 0.9, R * 0.9)

        petals = 16
        layers = 4
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(layers):
            fj = j / max(1, layers - 1)
            rr = R * (0.45 + 0.55 * fj) * (1.0 + 0.12 * self._env_lo)
            width = rr * (0.20 + 0.12 * (1.0 - fj))
            length = rr * (0.55 + 0.30 * self._env_mid)
            rot = tt * (0.25 + 0.7 * self._env_hi) + fj * 0.45

            for i in range(petals):
                a = rot + i * (2 * pi / petals)
                # Petal shape using a simple bezier leaf
                x0 = cx + cos(a) * rr
                y0 = cy + sin(a) * rr
                tip = QPointF(cx + cos(a) * (rr + length), cy + sin(a) * (rr + length))
                left = QPointF(cx + cos(a + pi/2) * width + cos(a) * (rr + length*0.35),
                               cy + sin(a + pi/2) * width + sin(a) * (rr + length*0.35))
                right = QPointF(cx + cos(a - pi/2) * width + cos(a) * (rr + length*0.35),
                                cy + sin(a - pi/2) * width + sin(a) * (rr + length*0.35))

                path = QPainterPath()
                path.moveTo(QPointF(x0, y0))
                path.quadTo(left, tip)
                path.quadTo(right, QPointF(x0, y0))

                hue = int((280 - 40 * fj + 90 * self._env_hi + i * 4) % 360)
                alpha = int(18 + 55 * (1.0 - fj) + 70 * self._env_hi)
                col = QColor.fromHsv(hue, 180 + int(50 * self._env_mid), 255, alpha)

                p.setPen(QPen(QColor.fromHsv(hue, 120, 255, int(alpha * 0.75)), 1))
                p.setBrush(QBrush(col))
                p.drawPath(path)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
