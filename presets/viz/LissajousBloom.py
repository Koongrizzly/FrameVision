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
class LissajousBloom(BaseVisualizer):
    display_name = "Lissajous Bloom"

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
        target = 0.40*lo + 0.65*mid + 0.55*hi + 0.65*(rms or 0.0)
        self._env = _env_step(self._env, target, 0.6, 0.22)

        if _active(bands, rms):
            self._anim += 1.0/60.0

        p.fillRect(r, QColor(1, 1, 5))
        cx, cy = w*0.5, h*0.5
        R = min(w, h)*0.40

        # bloom curve
        a = 3
        b = 4
        phi = self._anim*(1.2 + 1.8*self._env)

        path = QPainterPath()
        pts = 260
        for i in range(pts+1):
            t0 = (i/pts) * 2*pi
            x = sin(a*t0 + phi)
            y = sin(b*t0)
            rr = (0.35 + 0.65*(0.5 + 0.5*sin(t0*2 + self._anim))) * (0.65 + 0.7*self._env)
            px = cx + x * R * rr
            py = cy + y * R * rr
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        hue = int((260 + 120*self._env + self._anim*55) % 360)
        pen = QPen(QColor.fromHsv(hue, 210, 255, 190), 3 + int(4*self._env))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # dots along curve
        p.setPen(Qt.NoPen)
        for i in range(24):
            t0 = (i/24.0) * 2*pi
            x = sin(a*t0 + phi)
            y = sin(b*t0)
            px = cx + x * R * (0.75 + 0.35*self._env)
            py = cy + y * R * (0.75 + 0.35*self._env)
            hue2 = int((hue + i*14) % 360)
            p.setBrush(QBrush(QColor.fromHsv(hue2, 220, 255, 170)))
            p.drawEllipse(QPointF(px, py), 3.0 + 5.0*self._env, 3.0 + 5.0*self._env)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
