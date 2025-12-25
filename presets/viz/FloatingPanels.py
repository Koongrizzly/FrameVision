from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, Qt, QRectF
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
class FloatingPanels(BaseVisualizer):
    display_name = "Floating Panels"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._panels = []
        self._built = False

    def _build(self, w, h):
        self._panels.clear()
        for i in range(40):
            pw = w*0.12 + random()*w*0.08
            ph = h*0.08 + random()*h*0.08
            self._panels.append({
                "x": random()*w,
                "y": random()*h,
                "w": pw,
                "h": ph,
                "vx": (random()-0.5)*20,
                "vy": (random()-0.5)*20,
                "phase": random()*2*pi,
            })
        self._built = True

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        if not self._built:
            self._build(w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 6, 12))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)

        dt = 1/60.0
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for idx, pan in enumerate(self._panels):
            pan["phase"] += dt*(0.5 + self._env_mid)
            pan["x"] += pan["vx"]*dt*(0.6 + 1.0*self._env_lo)
            pan["y"] += pan["vy"]*dt*(0.6 + 1.0*self._env_lo)
            if pan["x"] + pan["w"] < 0: pan["x"] = w
            if pan["x"] > w: pan["x"] = -pan["w"]
            if pan["y"] + pan["h"] < 0: pan["y"] = h
            if pan["y"] > h: pan["y"] = -pan["h"]
            flicker = 0.5 + 0.5*sin(pan["phase"]*2.0)
            alpha = int(40 + 150*flicker*(0.4 + 0.6*self._env_hi))
            hue = (int(150 + idx*7) + int(t*10)) % 360
            col = QColor.fromHsv(hue, 210, 255, alpha)
            p.setPen(QPen(QColor(10, 20, 40, alpha), 1.2))
            p.setBrush(QBrush(col))
            rect = QRectF(pan["x"], pan["y"], pan["w"], pan["h"])
            p.drawRoundedRect(rect, 8, 8)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
