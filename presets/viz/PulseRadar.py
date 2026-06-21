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


def _onset_gate(bands):
    # simple flux-based gate, self-contained
    if not bands:
        return 0.0
    n = len(bands)
    cut = max(1, n // 6)
    # local flux approximation: mid/high concentration
    s = 0.0
    for i in range(cut, n):
        s += bands[i]
    return (s / max(1, n-cut))

@register_visualizer
class PulseRadar(BaseVisualizer):
    display_name = "Pulse Radar"

    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._anim = 0.0
        self._blips = []  # (age, ang, dist)

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 1 or h <= 1:
            return
        p.setRenderHint(QPainter.Antialiasing, True)

        lo, mid, hi = _split(bands)
        target = 0.35*lo + 0.55*mid + 0.85*hi + 0.55*(rms or 0.0)
        self._env = _env_step(self._env, target, 0.6, 0.22)

        active = _active(bands, rms)
        if active:
            self._anim += 1.0/60.0

        p.fillRect(r, QColor(0, 0, 0))

        cx, cy = w*0.5, h*0.5
        R = min(w, h)*0.44

        # radar grid
        p.setPen(QPen(QColor(40, 120, 90, 70), 1))
        p.setBrush(Qt.NoBrush)
        for k in range(5):
            rr = R*(0.2 + 0.2*k)
            p.drawEllipse(QPointF(cx, cy), rr, rr)
        p.drawLine(int(cx-R), int(cy), int(cx+R), int(cy))
        p.drawLine(int(cx), int(cy-R), int(cx), int(cy+R))

        # sweep
        sweep = (self._anim*1.8) % (2*pi)
        x2 = cx + cos(sweep) * R
        y2 = cy + sin(sweep) * R

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        hue = int((120 + 120*self._env + self._anim*30) % 360)
        p.setPen(QPen(QColor.fromHsv(hue, 220, 255, 180), 3))
        p.drawLine(QPointF(cx, cy), QPointF(x2, y2))

        # fade wedge
        for i in range(12):
            a = sweep - i*(0.10 + 0.06*self._env)
            alpha = int(18 + 10*i)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, alpha), 6))
            p.drawLine(QPointF(cx, cy), QPointF(cx + cos(a)*R, cy + sin(a)*R))

        # blips on strong highs (only while active)
        gate = _onset_gate(bands) if bands else 0.0
        if active and gate > 0.22 and (len(self._blips) < 18):
            self._blips.append([0.0, (random()*2*pi), (0.15 + 0.85*random())])

        # age blips only when active (so they freeze when paused)
        if active:
            for b in self._blips:
                b[0] += 1.0/60.0
            self._blips = [b for b in self._blips if b[0] < (1.2 + 1.2*self._env)]

        p.setPen(Qt.NoPen)
        for age, ang, dist in self._blips:
            rr = R*dist
            x = cx + cos(ang)*rr
            y = cy + sin(ang)*rr
            a = max(0.0, 1.0 - age/(1.2 + 1.2*self._env))
            col = QColor.fromHsv((hue+60) % 360, 220, 255, int(210*a))
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(x, y), 3.0 + 6.0*self._env, 3.0 + 6.0*self._env)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
