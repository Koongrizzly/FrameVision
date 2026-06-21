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
class LiquidColumns(BaseVisualizer):
    display_name = "Liquid Columns"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.85 * rms, 0.72, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.62, 0.24)

        act = _activity(bands, rms)
        if act > 0.005:
            self._phase += (0.7 + 2.8 * self._env_mid) * (1.0 / 60.0)

        p.fillRect(r, QColor(2, 2, 8))
        cols = 22
        pad = w * 0.08
        span = w - 2 * pad
        cw = span / cols

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(cols):
            x = pad + i * cw + cw * 0.5
            band_i = int(i / max(1, cols - 1) * max(1, len(bands) - 1)) if bands else 0
            b = bands[band_i] if bands else 0.0
            amp = _clamp01(0.55 * b + 0.35 * self._env_lo + 0.15 * sin(self._phase + i * 0.35))
            height = h * (0.08 + 0.78 * amp)
            y0 = h * 0.92
            y1 = y0 - height

            hue = int((180 + 140 * amp + 80 * self._env_mid) % 360)
            alpha = int(35 + 90 * (0.35 + 0.65 * self._env_lo))
            p.setPen(Qt.NoPen)

            grad = QLinearGradient(QPointF(x, y0), QPointF(x, y1))
            grad.setColorAt(0.0, QColor.fromHsv(hue, 210, 255, 0))
            grad.setColorAt(0.5, QColor.fromHsv(hue, 220, 255, alpha))
            grad.setColorAt(1.0, QColor.fromHsv((hue + 30) % 360, 220, 255, int(alpha * 0.8)))
            p.setBrush(QBrush(grad))

            bw = cw * 0.65
            p.drawRoundedRect(QRectF(x - bw * 0.5, y1, bw, height), bw * 0.35, bw * 0.35)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
