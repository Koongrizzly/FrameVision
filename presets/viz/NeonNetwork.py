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

def _env_step(env, target, up=0.55, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class NeonNetwork(BaseVisualizer):
    display_name = "Neon Network"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._nodes = []
        self._built = False

    def _build(self, w, h):
        self._nodes.clear()
        count = 40
        for i in range(count):
            x = random()*w
            y = random()*h
            dx = (random()-0.5)*0.6
            dy = (random()-0.5)*0.6
            hue = int(random()*360)
            self._nodes.append({
                "x": x, "y": y,
                "dx": dx, "dy": dy,
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
        p.fillRect(r, QColor(3, 5, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.24)

        dt = 1/60.0
        for n in self._nodes:
            n["x"] += n["dx"] * (1.0 + 1.2*self._env_lo) * dt * 60
            n["y"] += n["dy"] * (1.0 + 1.2*self._env_lo) * dt * 60
            if n["x"] < 0 or n["x"] > w:
                n["dx"] *= -1
            if n["y"] < 0 or n["y"] > h:
                n["dy"] *= -1

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        # Draw edges between close nodes
        node_count = len(self._nodes)
        for i in range(node_count):
            ni = self._nodes[i]
            xi, yi = ni["x"], ni["y"]
            for j in range(i+1, node_count):
                nj = self._nodes[j]
                dx = xi - nj["x"]
                dy = yi - nj["y"]
                d2 = dx*dx + dy*dy
                if d2 < (w*h*0.03)/(1.0 + 2.0*self._env_mid):
                    strength = max(0.0, 1.0 - d2/(w*h*0.03))
                    hue = (ni["hue"] + nj["hue"])//2
                    alpha = int(40 + 160*strength*self._env_mid)
                    col = QColor.fromHsv(hue % 360, 210, 255, alpha)
                    p.setPen(QPen(col, 1.5))
                    p.drawLine(QPointF(xi, yi), QPointF(nj["x"], nj["y"]))

        # Draw nodes on top
        for n in self._nodes:
            hue = n["hue"]
            glow = QColor.fromHsv(hue, 220, 255, int(80 + 100*self._env_hi))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(n["x"], n["y"]), 3.5 + 2*self._env_hi, 3.5 + 2*self._env_hi)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
