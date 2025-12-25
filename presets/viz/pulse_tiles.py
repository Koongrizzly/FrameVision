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
class PulseTiles(BaseVisualizer):
    display_name = "Pulse Tiles (CPU)"

    def __init__(self):
        super().__init__()
        self._env = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        playing = _is_playing(bands, rms)
        lo, mid, hi = _split(bands)
        self._env = _env_step(self._env, 0.45 * lo + 0.45 * mid + 0.25 * rms, 0.62, 0.28)

        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(r, QColor(4, 4, 8))

        cols = 12
        rows = 7
        pad = 2
        tw = max(1, int((w - (cols + 1) * pad) / cols))
        th = max(1, int((h - (rows + 1) * pad) / rows))

        base_hue = int((t * 30) % 360) if playing else 210
        for y in range(rows):
            for x in range(cols):
                idx = (y * cols + x)
                b = bands[idx % len(bands)] if bands else 0.0
                v = (0.15 + 0.85 * b) * (0.25 + 0.75 * self._env)
                hue = (base_hue + idx * 7) % 360
                alpha = int(70 + 140 * v) if playing else int(60 + 90 * v)
                col = QColor.fromHsv(hue, 220, int(80 + 175 * v), alpha)
                xx = pad + x * (tw + pad)
                yy = pad + y * (th + pad)
                p.fillRect(xx, yy, tw, th, col)
