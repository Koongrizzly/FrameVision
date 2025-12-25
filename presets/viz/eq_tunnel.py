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
class EQTunnel(BaseVisualizer):
    display_name = "EQ Tunnel (CPU)"

    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._scroll = 0.0
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
        self._env_lo = _env_step(self._env_lo, lo + 0.55 * rms, 0.62, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.58, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.28)

        if playing:
            self._scroll += dt * (0.8 + 2.8 * self._env_mid)

        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(r, QColor(2, 2, 7))

        cx, cy = w * 0.5, h * 0.5
        bars = 40
        bar_w = max(2, int(w * 0.75 / bars))
        left = cx - (bars * bar_w) * 0.5

        # perspective lines (cheap)
        p.setPen(QPen(QColor(30, 40, 70, 110), 1))
        for i in range(9):
            k = i / 8.0
            x = w * (0.15 + 0.70 * k)
            p.drawLine(int(x), int(h * 0.15), int(cx), int(cy))
            p.drawLine(int(x), int(h * 0.85), int(cx), int(cy))

        # bars
        hue_base = int((t * 20) % 360)
        for i in range(bars):
            b = bands[i % len(bands)] if bands else 0.0
            amp = (0.35 * self._env_lo + 0.55 * self._env_mid + 0.45 * self._env_hi) * (0.35 + 0.65 * b)
            height = (h * (0.10 + 0.55 * amp))
            x = left + i * bar_w
            # slight moving "tunnel" offset only when playing
            z = (i / bars + self._scroll * 0.08) % 1.0
            yoff = (z * z) * (h * (0.10 + 0.20 * self._env_lo)) if playing else 0.0

            hue = (hue_base + i * 5) % 360
            col = QColor.fromHsv(hue, 220, 255, 110 if playing else 70)
            p.fillRect(int(x), int(cy - height * 0.5 + yoff), bar_w - 1, int(height), col)
