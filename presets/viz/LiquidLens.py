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
class LiquidLens(BaseVisualizer):
    display_name = "Liquid Lens"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_hi = 0.0
        self._anim = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 1 or h <= 1:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.55*(rms or 0.0), 0.65, 0.25)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        if _active(bands, rms):
            self._anim += 1.0/60.0

        # Background gradient
        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor(1, 2, 8))
        bg.setColorAt(1.0, QColor(0, 0, 0))
        p.fillRect(r, QBrush(bg))

        cx, cy = w*0.5, h*0.5
        R = min(w, h)*0.33

        # Blob shape (wobble)
        wob1 = 0.18 + 0.35*self._env_lo
        wob2 = 0.12 + 0.28*self._env_hi
        pts = 80
        path = QPainterPath()
        for i in range(pts+1):
            a = (i/pts)*2*pi
            wob = 1.0 + wob1*sin(a*3 + self._anim*1.7) + wob2*sin(a*5 - self._anim*1.1)
            rr = R*wob
            x = cx + cos(a)*rr
            y = cy + sin(a)*rr
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()

        p.setCompositionMode(QPainter.CompositionMode_Plus)

        hue = int((280 + 100*self._env_hi + self._anim*35) % 360)
        col = QColor.fromHsv(hue, 210, 255, 130)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(col))
        p.drawPath(path)

        # Inner highlight
        g = QRadialGradient(QPointF(cx - R*0.12, cy - R*0.18), R*0.9)
        g.setColorAt(0.0, QColor(255, 255, 255, int(35 + 80*self._env_lo)))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), R*0.85, R*0.85)

        # Rim
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(QPen(QColor(220, 230, 255, int(25 + 90*self._env_hi)), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R*1.02, R*1.02)
