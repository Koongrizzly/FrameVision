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
class KaleidoBloom(BaseVisualizer):
    display_name = "Kaleido Bloom"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4,6,10))

        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.62, 0.24)

        cx, cy = w*0.5, h*0.5
        petals = 12
        base = QPainterPath()
        base.moveTo(0, 0)
        for i in range(1, 60):
            ang = i*0.2 + t*0.9
            rad = 40 + 30*sin(i*0.25 + t) + 70*self._env_mid
            base.lineTo(cos(ang)*rad, sin(ang)*rad)

        p.save()
        p.translate(cx, cy)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for k in range(petals):
            p.save()
            p.rotate(k*(360.0/petals) + 20*self._env_lo*sin(t*0.6+k))
            hue = (int(t*30) + k*20) % 360
            col = QColor.fromHsv(hue, 220, 255, int(120 + 100*self._env_hi))
            p.setPen(QPen(col, 3))
            p.setBrush(Qt.NoBrush)
            p.drawPath(base)
            p.restore()
        p.restore()

        p.setPen(QPen(QColor(200,220,255,120), 2))
        for rr in range(4):
            rad = 18 + rr*8 + 14*self._env_lo
            p.drawEllipse(QPointF(cx,cy), rad, rad)
