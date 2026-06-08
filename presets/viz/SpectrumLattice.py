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
class SpectrumLattice(BaseVisualizer):
    display_name = "Spectrum Lattice"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._scroll = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7 * rms, 0.70, 0.28)
        self._env_mid = _env_step(self._env_mid, mid, 0.62, 0.24)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.28)

        act = _activity(bands, rms)
        if act > 0.005:
            self._scroll += (50 + 220 * self._env_mid) * (1.0 / 60.0)

        p.fillRect(r, QColor(2, 3, 7))

        cols = 16
        rows = 10
        cell_w = w / cols
        cell_h = h / rows

        # lattice lines that only shift when _scroll advances
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(rows + 1):
            y = j * cell_h
            hue = int((220 + j * 10 + 120 * self._env_hi) % 360)
            alpha = int(35 + 55 * self._env_mid)
            p.setPen(QPen(QColor.fromHsv(hue, 200, 255, alpha), 2))
            p.drawLine(QPointF(0, y), QPointF(w, y))

        for i in range(cols + 1):
            x = i * cell_w
            hue = int((160 + i * 12 + 140 * self._env_lo) % 360)
            alpha = int(30 + 60 * self._env_mid)
            p.setPen(QPen(QColor.fromHsv(hue, 210, 255, alpha), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, h))

        # moving highlight diagonal "pulse" (gated)
        diag = (self._scroll % (w + h)) if act > 0.005 else (self._scroll % (w + h))
        p.setPen(QPen(QColor.fromHsv(int(300 * self._env_hi) % 360, 220, 255, int(70 + 90 * self._env_hi)), 5))
        p.drawLine(QPointF(-h + diag, h), QPointF(diag, 0))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
