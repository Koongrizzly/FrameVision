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
class SkylinePulse(BaseVisualizer):
    display_name = "Skyline Pulse"

    def __init__(self):
        super().__init__()
        self._tg = _PlayGatedTime()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rng = Random(77331)

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.85 * rms, 0.7, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.26)

        energy = max(rms, (lo + mid + hi) / 3.0)
        self._tg.step(t, energy)
        tt = self._tg.anim_t

        # dusk sky gradient
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QColor(4, 6, 16))
        bg.setColorAt(1.0, QColor(2, 2, 8))
        p.fillRect(r, QBrush(bg))

        horizon_y = h * 0.72
        # skyline base
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 160)))
        p.drawRect(QRectF(0, horizon_y, w, h - horizon_y))

        # audio bars become "buildings"
        n = max(16, min(72, len(bands) if bands else 48))
        bw = w / n
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(n):
            v = (bands[i] if bands and i < len(bands) else 0.0)
            # emphasize mid/high for city energy
            vv = (0.25 + 0.75 * (i / max(1, n - 1))) * v
            height = (h * 0.40) * (0.05 + 0.9 * vv) * (0.65 + 0.55 * self._env_lo)
            x = i * bw
            y = horizon_y - height
            hue = int((210 + 120 * (i / n) + 140 * self._env_hi) % 360)
            alpha = int(30 + 90 * (0.3 + 0.7 * self._env_mid))
            col = QColor.fromHsv(hue, 210, 255, alpha)

            p.setBrush(QBrush(col))
            p.drawRoundedRect(QRectF(x + bw * 0.12, y, bw * 0.76, height), 2.0, 2.0)

            # windows sparkle (t gated)
            if (i % 3) == 0:
                tw = 0.5 + 0.5 * sin(tt * (1.4 + 0.02 * i) + i)
                win_alpha = int(20 + 140 * tw * self._env_hi)
                p.setBrush(QBrush(QColor(255, 255, 255, win_alpha)))
                wx = x + bw * 0.28
                wy = y + height * 0.18
                for k in range(4):
                    p.drawRect(QRectF(wx, wy + k * height * 0.18, bw * 0.20, height * 0.08))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
