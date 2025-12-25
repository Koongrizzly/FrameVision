from math import sin, cos, pi
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
class RingMosaic(BaseVisualizer):
    display_name = "Ring Mosaic"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3, 4, 8))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)

        cx, cy = w*0.5, h*0.5
        max_r = min(w, h)*0.5
        rings = 6
        segs = 32

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for R in range(rings):
            frac = R / max(1, rings-1)
            base_r = max_r*(0.1 + 0.8*frac)
            for s in range(segs):
                ang0 = 2*pi*s/segs
                ang1 = 2*pi*(s+1)/segs
                amp = 0.3*self._env_lo + 0.5*self._env_mid
                r_inner = base_r*(0.85 + 0.15*sin(ang0*3 + t*0.7 + R))
                r_outer = base_r*(1.03 + 0.22*amp)
                x0 = cx + cos(ang0)*r_inner
                y0 = cy + sin(ang0)*r_inner
                x1 = cx + cos(ang1)*r_inner
                y1 = cy + sin(ang1)*r_inner
                x2 = cx + cos(ang1)*r_outer
                y2 = cy + sin(ang1)*r_outer
                x3 = cx + cos(ang0)*r_outer
                y3 = cy + sin(ang0)*r_outer
                hue = (int(ang0*180/pi) + int(t*30) + R*30) % 360
                alpha = int(40 + 140*(frac + 0.4*self._env_hi))
                col = QColor.fromHsv(hue, 220, 255, alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col))
                p.drawPolygon([QPointF(x0,y0), QPointF(x1,y1), QPointF(x2,y2), QPointF(x3,y3)])

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
