from math import sin, cos, pi, sqrt
from random import random, Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QRadialGradient, QLinearGradient
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
    # fast attack, slower release
    if target > env:
        return (1 - up) * env + up * target
    return (1 - down) * env + down * target

def _active(bands, rms, thr=0.012):
    # If rms is provided by the engine, trust it as the "playing" signal.
    # This makes visuals fully freeze when playback is paused.
    if rms is not None:
        return rms > thr
    if bands:
        s = 0.0
        for v in bands:
            s += v
        return (s / max(1, len(bands))) > (thr * 0.8)
    return False


@register_visualizer
class RippleGrid(BaseVisualizer):
    display_name = "Ripple Grid"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._anim = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 1 or h <= 1:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.55*lo + 0.55*mid + 0.75*hi + 0.45*(rms or 0.0)
        self._env = _env_step(self._env, target, 0.6, 0.22)

        if _active(bands, rms):
            self._anim += 1.0/60.0

        p.fillRect(r, QColor(2, 2, 6))

        cols = 18
        rows = 10
        dx = w / float(cols-1)
        dy = h / float(rows-1)

        amp = (0.03 + 0.10*self._env) * min(w, h)
        freq = 1.2 + 2.2*self._env

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for j in range(rows):
            for i in range(cols):
                x0 = i*dx
                y0 = j*dy
                d = sqrt((x0 - w*0.5)**2 + (y0 - h*0.5)**2) / max(1.0, min(w, h))
                # ripple displacement
                s = sin((d*10.0)*freq - self._anim*3.0)
                x = x0 + amp*0.25*s*cos(self._anim + d*4.0)
                y = y0 + amp*0.25*s*sin(self._anim + d*4.0)

                hue = int((210 + 120*s + 160*d + self._anim*25) % 360)
                alpha = int(30 + 120*(0.5 + 0.5*s))
                col = QColor.fromHsv(hue, 210, 255, alpha)

                size = 1.2 + 2.2*(0.5 + 0.5*s) + 2.8*self._env
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(x, y), size, size)

        # A faint gridline overlay
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(QPen(QColor(60, 80, 140, 35), 1))
        for i in range(cols):
            p.drawLine(int(i*dx), 0, int(i*dx), h)
        for j in range(rows):
            p.drawLine(0, int(j*dy), w, int(j*dy))
