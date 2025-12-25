from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF
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
class PrismStrata(BaseVisualizer):
    display_name = "Prism Strata"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 4, 10))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)

        layers = 6
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for L in range(layers):
            y_base = h*(0.1 + 0.15*L)
            height = h*0.18
            segs = 18
            hue = (int(t*20) + L*45) % 360
            for s in range(segs):
                x0 = (s / segs) * w
                x1 = ((s+1) / segs) * w
                a0 = 0.4*sin(t*0.6 + s*0.4 + L)
                a1 = 0.4*sin(t*0.6 + (s+1)*0.4 + L)
                y0 = y_base + height*(0.1 + 0.9*(0.5 + 0.5*a0*self._env_mid))
                y1 = y_base + height*(0.1 + 0.9*(0.5 + 0.5*a1*self._env_mid))
                apex_y = y_base - height*(0.2 + 0.6*self._env_mid)
                poly = QPolygonF([QPointF(x0, y0), QPointF(x1, y1), QPointF((x0+x1)*0.5, apex_y)])
                col = QColor.fromHsv(hue, 220, 200 + int(55*self._env_hi), int(60 + 150*(L/layers)))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawPolygon(poly)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
