from math import sin, cos, pi
from random import random, randint
from PySide6.QtGui import QPainter, QPen, QColor
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
class SpiralAurora(BaseVisualizer):
    display_name = "Spiral Aurora"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rot = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(3,5,9))

        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.24)

        cx, cy = w*0.5, h*0.5
        self._rot += (10 + 30*self._env_mid) * (1/60.0)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        arms = 8
        for arm in range(arms):
            p.save()
            p.translate(cx, cy)
            p.rotate(self._rot*40 + arm*(360.0/arms))
            hue = (int(t*20) + arm*30) % 360
            col = QColor.fromHsv(hue, 220, 255, 160)
            p.setPen(QPen(col, 3))
            last = QPointF(0,0)
            for i in range(1, 200, 4):
                rad = i*4*(1.0+0.12*self._env_lo)
                ang = i*0.25
                x = cos(ang)*rad
                y = sin(ang)*rad*0.8
                p.drawLine(last, QPointF(x,y))
                last = QPointF(x,y)
            p.restore()
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
