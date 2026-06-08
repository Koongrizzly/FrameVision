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
class CircuitWaves(BaseVisualizer):
    display_name = "Circuit Waves"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._anim = 0.0
        self._rng = Random(777)

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 1 or h <= 1:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.25*lo + 0.65*mid + 0.55*hi + 0.55*(rms or 0.0)
        self._env = _env_step(self._env, target, 0.6, 0.22)

        if _active(bands, rms):
            self._anim += 1.0/60.0

        p.fillRect(r, QColor(1, 2, 6))

        p.setCompositionMode(QPainter.CompositionMode_Plus)

        lines = 10
        for k in range(lines):
            f = k / max(1, lines-1)
            y = h*(0.10 + 0.80*f)
            phase = self._anim*(0.9 + 0.35*k) + k*2.1
            amp = (0.03 + 0.06*self._env)*(0.4 + 0.6*(1.0-f))*h
            path = QPainterPath()
            path.moveTo(0, y)
            steps = 22
            for i in range(1, steps+1):
                x = w*(i/steps)
                s = sin(phase + i*0.65) + 0.35*sin(phase*1.7 + i*1.2)
                yy = y + amp*s
                # 'circuit' corners
                if i % 3 == 0:
                    path.lineTo(x, yy)
                else:
                    path.quadTo(QPointF(x - w/steps*0.5, yy), QPointF(x, yy))
            hue = int((190 + 160*f + 120*self._env + self._anim*35) % 360)
            col = QColor.fromHsv(hue, 210, 255, int(35 + 120*(1.0-f)))
            p.setPen(QPen(col, 2 + int(3*self._env)))
            p.drawPath(path)

            # nodes
            for i in range(0, steps+1, 3):
                x = w*(i/steps)
                s = sin(phase + i*0.65) + 0.35*sin(phase*1.7 + i*1.2)
                yy = y + amp*s
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor.fromHsv((hue+50)%360, 220, 255, 120)))
                p.drawEllipse(QPointF(x, yy), 2.0 + 4.0*self._env, 2.0 + 4.0*self._env)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
