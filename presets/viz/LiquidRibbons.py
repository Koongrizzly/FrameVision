from math import sin, cos, pi
from random import random, randint
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
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
class LiquidRibbons(BaseVisualizer):
    display_name = "Liquid Ribbons"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2,4,8))

        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.24)

        layers = 3
        for L in range(layers):
            path = QPainterPath(QPointF(0, h*0.5 + 40*sin(t + L)))
            segs = 18
            amp = 50 + 70*self._env_mid + L*12
            for i in range(1, segs+1):
                x = (i/segs)*w
                y = h*0.5 + amp*sin(t*0.6 + i*0.7 + L) + 20*self._env_lo*sin(t*1.5 + i*0.3)
                path.lineTo(x, y)
            hue = (int(t*28) + L*60) % 360
            col = QColor.fromHsv(hue, 220, 255, 160)
            p.setPen(QPen(col, 4))
            p.setBrush(QBrush(QColor.fromHsv((hue+40)%360, 160, 255, 60)))
            p.drawPath(path)
