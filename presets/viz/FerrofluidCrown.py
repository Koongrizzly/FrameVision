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

_env_lo=_env_mid=_env_hi=0.0
@register_visualizer
class FerrofluidCrown(BaseVisualizer):
    display_name = "Ferrofluid Crown"
    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_env_lo,_env_mid,_env_hi
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:return
        if _last_t is None:_last_t=t
        dt=max(0.0,min(0.05,t-_last_t)); _last_t=t

        lo,mid,hi=_split(bands); fx=_flux(bands)
        _env_lo=_env_step(_env_lo, lo+0.2*rms, up=0.45, down=0.2)
        _env_mid=_env_step(_env_mid, mid+0.4*fx, up=0.35, down=0.15)
        _env_hi=_env_step(_env_hi, 0.7*hi + 1.6*fx, up=0.6, down=0.18)

        p.fillRect(r,QBrush(QColor(6,8,12)))
        cx,cy=w/2,h/2*1.05
        base = min(w,h)/6
        R= base*(1.0+0.8*min(1.0,_env_lo))
        spikes = int(20 + 50*min(1.0,_env_mid*1.4))
        p.setPen(QPen(QColor(20,30,40,150),1))
        p.setBrush(QBrush(QColor.fromHsv(int((t*30+220*_env_mid)%360), 180, 230, 240)))
        poly = QPolygonF()
        for k in range(spikes):
            ang = (k/spikes)*2*pi
            crown = 0.5 + 0.5*sin(ang*3 + t*2.5) + 0.9*_env_hi*sin(ang*9 + t*8.0)
            rr = R * (1.0 + 0.28*crown)
            poly.append(QPointF(cx + rr*cos(ang), cy + rr*sin(ang)))
        p.drawPolygon(poly)
        # inner shadow
        p.setBrush(QBrush(QColor(0,0,0,80)))
        p.drawEllipse(QPointF(cx,cy), R*0.62, R*0.62)
