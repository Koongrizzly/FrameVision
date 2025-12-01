from math import sin, cos, pi
from random import random, randint
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.52, down=0.2):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

@register_visualizer
class HexPulse(BaseVisualizer):
    display_name = "Hex Pulse"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._centers = []
        self._last_w = self._last_h = -1
        self._ring = 0.0

    def _hex_points(self, cx, cy, r):
        pts = []
        for k in range(6):
            a = pi/3*k
            pts.append(QPointF(cx + r*cos(a), cy + r*sin(a)))
        return pts

    def _build(self, w, h):
        self._centers.clear()
        size = 28
        dy = size * 1.5
        dx = size * 0.866
        y = size
        row = 0
        while y < h - size:
            x = size + (dx if row%2 else 0)
            while x < w - size:
                self._centers.append((x,y))
                x += dx*2
            y += dy
            row += 1

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if self._last_w!=w or self._last_h!=h or not self._centers:
            self._last_w, self._last_h = w, h
            self._build(w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(5,7,12))

        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        cx,cy = w*0.5, h*0.5
        self._ring += (120 + 260*self._env_lo) * (1/60.0)

        p.setPen(QPen(QColor(120,200,255,60), 2))
        p.setBrush(Qt.NoBrush)

        for (x,y) in self._centers:
            d = ((x-cx)**2 + (y-cy)**2)**0.5
            rhex = 12 + 4*sin(0.02*d + t) + 8*self._env_mid
            glow = max(0, 1.0 - abs(d - self._ring)%220 / 220.0)
            a = int(40 + 160*glow)
            p.setPen(QPen(QColor(120,200,255,a), 2))
            p.drawPolygon(self._hex_points(x,y,rhex))
