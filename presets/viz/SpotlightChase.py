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
class SpotlightChase(BaseVisualizer):
    display_name = "Spotlight Chase"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._ang = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.setRenderHint(QPainter.Antialiasing, True)
        lo,mid,hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.6*rms, 0.65, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.24)
        self._ang += (20 + 60*self._env_mid) * (1/60.0)
        p.fillRect(r, QColor(6,10,16))
        stage_y = h*0.78
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(12,14,22))); p.drawRect(0, stage_y, w, h-stage_y)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for side in (-1,1):
            bx = w*0.5 + side*w*0.28; by = h*0.12
            sweep = 40 + 25*sin(self._ang*0.9 + side)
            for i in range(6):
                a = (-60 + i*24) + sweep*sin(self._ang + i*0.6)*side
                length = h*0.95; spread = 0.22 + 0.18*self._env_lo
                tipx = bx + length*cos(a*pi/180.0); tipy = by + length*sin(a*pi/180.0)
                p.setPen(Qt.NoPen); col = QColor(120, 200, 255, int(70 + 150*self._env_hi)); p.setBrush(QBrush(col))
                p.drawPolygon([QPointF(bx,by), QPointF(tipx + spread*60, tipy + spread*10), QPointF(tipx - spread*60, tipy - spread*10)])
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(Qt.NoPen)
        for i in range(28):
            x = (i/27.0)*w; hh = (h-stage_y)*0.5*(0.6 + 0.4*random())*(1.0+0.2*self._env_lo)
            p.setBrush(QBrush(QColor(10,12,16,230))); p.drawEllipse(QPointF(x, stage_y+(h-stage_y)*0.2), w*0.03, hh*0.4)
        if self._env_hi>0.5:
            p.setPen(QPen(QColor(255,255,255,80), 2)); step = 6
            for x in range(0, w, step*2): p.drawLine(x, 0, x, h)
