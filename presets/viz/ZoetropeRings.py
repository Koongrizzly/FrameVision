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

_phase=0.0
@register_visualizer
class ZoetropeRings(BaseVisualizer):
    display_name = "Zoetrope Rings"
    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_phase
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:return
        if _last_t is None:_last_t=t
        dt=max(0.0,min(0.05,t-_last_t)); _last_t=t

        lo,mid,hi=_split(bands); fx=_flux(bands)
        strobe = 1 if (hi+fx)>0.16 else 0
        _phase += dt * (0.3 + 2.0*mid + 1.0*lo)

        p.fillRect(r,QBrush(QColor(10,10,12)))
        cx,cy=w/2,h/2
        rings=6
        for k in range(rings):
            rad = min(w,h)*0.45*(k+1)/rings
            segs = 24 + int(36*min(1.0,mid+0.2))
            for s in range(segs):
                a0 = (s/segs + (0 if strobe else _phase*0.1))*2*pi
                a1 = a0 + 2*pi/segs*0.6
                x0 = cx + rad*cos(a0); y0 = cy + rad*sin(a0)
                x1 = cx + rad*cos(a1); y1 = cy + rad*sin(a1)
                p.setPen(QPen(QColor.fromHsv(int((k*45 + s*3 + t*30)%360), 180, 230, 220), 3))
                p.drawLine(QPointF(x0,y0), QPointF(x1,y1))
