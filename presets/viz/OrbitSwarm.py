from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1, n//6); b=max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.5, down=0.2):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class OrbitSwarm(BaseVisualizer):
    display_name = "Orbit Swarm"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._orbits = []
        self._built = False

    def _build(self, w, h):
        self._orbits.clear()
        cx, cy = w*0.5, h*0.5
        layers = 6
        for L in range(layers):
            r = min(w, h) * (0.12 + 0.1*L)
            count = 6 + L*3
            for i in range(count):
                ang = random()*2*pi
                speed = (0.15 + 0.05*L) * (1 if i%2==0 else -1)
                size = 3 + 2*random()
                hue = (40*L + i*12) % 360
                self._orbits.append({
                    "r": r,
                    "ang": ang,
                    "speed": speed,
                    "size": size,
                    "hue": hue,
                })
        self._built = True

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        if not self._built:
            self._build(w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 6, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        cx, cy = w*0.5, h*0.5

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(8, 14, 26)))
        p.drawEllipse(QPointF(cx, cy), min(w, h)*0.18, min(w, h)*0.18)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        dt = 1/60.0
        for orb in self._orbits:
            orb["ang"] += orb["speed"] * (1.0 + 0.7*self._env_mid) * dt
            rr = orb["r"] * (1.0 + 0.15*self._env_lo*sin(orb["ang"]*2.0))
            x = cx + cos(orb["ang"]) * rr
            y = cy + sin(orb["ang"]*0.9) * rr*0.9
            hue = (orb["hue"] + int(self._env_hi*80)) % 360
            col = QColor.fromHsv(hue, 220, 255, int(80 + 130*self._env_hi))
            p.setPen(QPen(col, 1))
            p.drawLine(QPointF(cx, cy), QPointF(x, y))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(x, y), orb["size"], orb["size"])

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
