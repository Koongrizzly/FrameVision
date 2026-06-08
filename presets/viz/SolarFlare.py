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
class SolarFlare(BaseVisualizer):
    display_name = "Solar Flare"

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
        self._env_lo = _env_step(self._env_lo, lo + 0.9 * rms, 0.7, 0.26)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.64, 0.26)

        energy = max(rms, (lo + mid + hi) / 3.0)
        self._tg.step(t, energy)
        tt = self._tg.anim_t

        p.fillRect(r, QColor(6, 2, 4))
        cx, cy = w * 0.5, h * 0.5
        R = min(w, h) * 0.18 * (0.85 + 0.6 * self._env_lo)

        # Core
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        core = QRadialGradient(cx, cy, R * 2.6)
        core.setColorAt(0.0, QColor(255, 255, 255, int(70 + 120 * self._env_hi)))
        core.setColorAt(0.25, QColor(255, 210, 80, int(70 + 140 * self._env_mid)))
        core.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(core))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), R * 2.6, R * 2.6)

        # Flares
        spokes = 64
        for i in range(spokes):
            k = i / spokes
            ang = k * 2 * pi + tt * (0.25 + 0.9 * self._env_mid)
            mod = 0.35 + 0.65 * sin(tt * (1.6 + 1.2 * self._env_hi) + i * 0.45)
            length = min(w, h) * (0.18 + 0.34 * mod) * (0.35 + 1.2 * self._env_mid)
            x1 = cx + cos(ang) * R * 0.8
            y1 = cy + sin(ang) * R * 0.8
            x2 = cx + cos(ang) * (R * 0.8 + length)
            y2 = cy + sin(ang) * (R * 0.8 + length)

            hue = int((20 + 25 * mod + 40 * self._env_hi) % 360)  # warm
            alpha = int(10 + 80 * mod * (0.25 + 0.75 * self._env_hi))
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, alpha), 2))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
