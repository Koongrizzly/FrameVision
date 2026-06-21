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
class WaveTunnel(BaseVisualizer):
    display_name = "Wave Tunnel"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._rot = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(2,4,8))

        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.24)

        cx, cy = w*0.5, h*0.5
        self._rot += (10 + 40*self._env_mid)*(1/60.0)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(26):
            d = i/25.0
            rad = (min(w,h)*0.1 + i*18) * (1.0 + 0.08*self._env_lo*sin(t*1.2+i))
            hue = (int(t*30) + i*10) % 360
            col = QColor.fromHsv(hue, 220, 255, int(40 + 190*(1.0-d)))
            p.setPen(QPen(col, 3))
            p.save()
            p.translate(cx, cy)
            p.rotate(self._rot* (1.0 + 0.1*i))
            p.drawEllipse(QPointF(0,0), rad, rad*0.7)
            p.restore()
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
