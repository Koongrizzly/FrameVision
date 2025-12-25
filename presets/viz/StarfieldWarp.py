from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor
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

def _env_step(env, target, up=0.55, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class StarfieldWarp(BaseVisualizer):
    display_name = "Starfield Warp"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._stars = []
        self._built = False

    def _build(self, w, h):
        self._stars.clear()
        count = 260
        for i in range(count):
            angle = random()*2*pi
            dist = random()
            speed = 0.35 + 0.9*dist
            self._stars.append({
                "angle": angle,
                "dist": dist,
                "speed": speed,
            })
        self._built = True

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        if not self._built:
            self._build(w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(1, 2, 6))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)

        cx, cy = w*0.5, h*0.5
        max_r = (w**2 + h**2)**0.5 * 0.6
        dt = 1/60.0

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for s in self._stars:
            s["dist"] += s["speed"] * dt * (0.6 + 2.0*self._env_lo)
            if s["dist"] > 1.3:
                s["dist"] = random()*0.1
                s["angle"] = random()*2*pi
                s["speed"] = 0.35 + 0.9*s["dist"]
            r0 = max_r * (s["dist"] - 0.05)
            r1 = max_r * s["dist"]
            x0 = cx + cos(s["angle"])*r0
            y0 = cy + sin(s["angle"])*r0
            x1 = cx + cos(s["angle"])*r1
            y1 = cy + sin(s["angle"])*r1
            brightness = min(1.0, 0.2 + s["dist"]*1.1 + 0.4*self._env_hi)
            alpha = int(60 + 180*brightness)
            col = QColor(170, 220, 255, alpha)
            p.setPen(QPen(col, 1.3))
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
