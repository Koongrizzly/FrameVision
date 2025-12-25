from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n = len(bands)
    a = max(1, n//6); b = max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.58, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class BlobField(BaseVisualizer):
    display_name = "Blob Field"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._points = []
        self._seeded = False

    def _seed(self, w, h):
        self._points = []
        for _ in range(60):
            self._points.append({
                "x": random()*w,
                "y": random()*h,
                "ox": random()*2*pi,
                "oy": random()*2*pi,
            })
        self._seeded = True

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        if not self._seeded:
            self._seed(w, h)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 5, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7*rms, 0.7, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)

        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for i, pt in enumerate(self._points):
            k = i/float(max(1,len(self._points)-1))
            local = bands[i % len(bands)] if bands else 0.0

            jitter_x = 30*sin(t*0.9 + pt["ox"])
            jitter_y = 30*sin(t*1.1 + pt["oy"])
            x = pt["x"] + jitter_x*self._env_mid
            y = pt["y"] + jitter_y*self._env_mid

            size = 6 + 40*(0.4*self._env_lo + 0.5*self._env_mid + 0.4*local)
            size *= (0.7 + 0.6*sin(t*1.5 + k*6.0))

            hue = (int(t*15) + int(k*260)) % 360
            glow = QColor.fromHsv(hue, 200, 255, int(40 + 120*self._env_hi))
            core = QColor.fromHsv(hue, 220, 255, 220)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(x, y), size*1.1, size*1.1)
            p.setBrush(QBrush(core))
            p.drawEllipse(QPointF(x, y), size*0.55, size*0.55)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
