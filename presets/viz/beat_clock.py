from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, Qt
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

def _is_playing(bands, rms, eps=1e-3):
    if rms and rms > eps:
        return True
    if bands:
        # keep it cheap
        s = 0.0
        for i in range(0, len(bands), max(1, len(bands)//12)):
            s += bands[i]
        return s > eps
    return False

@register_visualizer
class BeatClock(BaseVisualizer):
    display_name = "Beat Clock (CPU)"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0
        self._last_t = None

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        playing = _is_playing(bands, rms)
        if self._last_t is None:
            self._last_t = t
        dt = max(0.0, float(t - self._last_t))
        self._last_t = t

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.45 * rms, 0.65, 0.26)
        self._env_mid = _env_step(self._env_mid, mid, 0.60, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.28)

        if playing:
            self._phase += dt * (0.8 + 3.4 * self._env_mid)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2, 2, 6))

        cx, cy = w * 0.5, h * 0.5
        R = min(w, h) * 0.38

        # outer ring
        p.setPen(QPen(QColor(70, 90, 130, 90), 2))
        p.drawEllipse(QPointF(cx, cy), R, R)

        ticks = 48
        for i in range(ticks):
            k = i / float(ticks)
            ang = k * 2 * pi + (self._phase * 0.6)
            b = bands[i % len(bands)] if bands else 0.0
            strength = (0.25 + 0.75 * b) * (0.35 + 0.65 * (0.35 * self._env_lo + 0.45 * self._env_mid + 0.55 * self._env_hi))
            L = R * (0.10 + 0.28 * strength)
            x1 = cx + cos(ang) * (R - L)
            y1 = cy + sin(ang) * (R - L)
            x2 = cx + cos(ang) * (R)
            y2 = cy + sin(ang) * (R)
            hue = int((180 + 180 * k + 80 * self._env_hi + (t * 10 if playing else 0)) % 360)
            alpha = int(70 + 160 * strength) if playing else int(60 + 110 * strength)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, alpha), 2))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # center dot
        p.setPen(Qt.NoPen)
        hue = int((t * 40) % 360)
        p.setBrush(QBrush(QColor.fromHsv(hue, 220, 255, 200 if playing else 130)))
        p.drawEllipse(QPointF(cx, cy), R * 0.06, R * 0.06)
