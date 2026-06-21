# Auto-generated music visualizer
from math import sin, cos, pi, sqrt
from random import random, randint, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QFont
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_prev_spec = []
_last_t = None

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if not _prev_spec or len(_prev_spec)!=len(bands):
        _prev_spec = [0.0]*len(bands)
    f = 0.0
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d>0: f += d * (0.35 + 0.65*(i/max(1,len(bands)-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f / max(1,len(bands))

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.34, down=0.14):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

_offset=0.0
@register_visualizer
class MoireMasks(BaseVisualizer):
    display_name = "Moiré Masks"
    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_offset
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:return
        if _last_t is None:_last_t=t
        dt=max(0.0,min(0.05,t-_last_t)); _last_t=t

        lo,mid,hi=_split(bands); fx=_flux(bands)
        speed = 40 + 260*min(1.0, mid + 0.4*fx)
        _offset = (_offset + dt*speed) % (w*2)

        p.fillRect(r,QBrush(QColor(12,12,14)))
        # layer 1: vertical bars
        bar_w = 12
        p.setBrush(QBrush(QColor(220,220,230,90)))
        p.setPen(QPen(QColor(0,0,0,0),0))
        for x in range(-w, 2*w, bar_w*2):
            p.drawRect(QRectF(x + _offset*0.6, 0, bar_w, h))

        # layer 2: angled mask (reveals hidden words on highs)
        p.setBrush(QBrush(QColor(220,220,230,90)))
        gap = 18
        for i in range(-h, h, gap):
            x0 = 0; y0 = i + _offset*0.3
            x1 = w; y1 = i + w*0.5 + _offset*0.3
            p.setPen(QPen(QColor(180,180,200,80), 2))
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        if hi+fx>0.2:
            p.setPen(QPen(QColor(255,255,255,200), 3))
            p.setFont(QFont("Arial", int(min(w,h)*0.11), QFont.Bold))
            p.drawText(r, Qt.AlignCenter, "FrameVision")
