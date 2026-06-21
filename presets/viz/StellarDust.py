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
class StellarDust(BaseVisualizer):
    display_name = "Stellar Dust"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._anim = 0.0
        self._rng = Random(20201)
        self._stars = [(self._rng.random()*2-1, self._rng.random()*2-1, 0.2 + 0.8*self._rng.random()) for _ in range(420)]

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 1 or h <= 1:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.35*lo + 0.55*mid + 0.7*hi + 0.6*(rms or 0.0)
        self._env = _env_step(self._env, target, 0.6, 0.22)

        active = _active(bands, rms)
        if active:
            self._anim += 1.0/60.0

        # Background
        p.fillRect(r, QColor(0, 0, 0))

        cx, cy = w*0.5, h*0.5
        scale = min(w, h)*0.5
        speed = (0.08 + 0.55*self._env)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i, (sx, sy, z) in enumerate(self._stars):
            # update only when active (so pause freezes)
            if active:
                # drift outward
                sx *= (1.0 + speed*0.006*(0.4+z))
                sy *= (1.0 + speed*0.006*(0.4+z))
                # respawn if out of bounds
                if abs(sx) > 1.2 or abs(sy) > 1.2:
                    sx, sy, z = (self._rng.random()*2-1, self._rng.random()*2-1, 0.2 + 0.8*self._rng.random())
                self._stars[i] = (sx, sy, z)

            x = cx + sx*scale
            y = cy + sy*scale

            tw = 0.5 + 0.5*sin(self._anim*3.0 + i*0.08)
            size = (0.8 + 2.2*z) * (0.8 + 0.9*self._env) * (0.7 + 0.6*tw)
            hue = int((200 + 140*z + 90*self._env + self._anim*20) % 360)
            alpha = int(35 + 150*z*(0.7+0.3*tw))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor.fromHsv(hue, 180, 255, alpha)))
            p.drawEllipse(QPointF(x, y), size, size)

        # soft center glow
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        g = QRadialGradient(QPointF(cx, cy), scale*0.9)
        g.setColorAt(0.0, QColor(40, 80, 160, int(45 + 90*self._env)))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, QBrush(g))
