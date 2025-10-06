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

_env = 0.0
pin_field = None

def _setup(w,h,cell=10):
    cols=max(24,int(w/cell))
    rows=max(16,int(h/cell))
    global pin_field
    if pin_field is None or len(pin_field)!=rows or len(pin_field[0])!=cols:
        pin_field = [[0.0]*cols for _ in range(rows)]
    return cols,rows,cell

@register_visualizer
class PinscreenImprint(BaseVisualizer):
    display_name = "Pinscreen Imprint"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t,_env,pin_field
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:return
        if _last_t is None:_last_t=t
        dt=max(0.0,min(0.05,t-_last_t)); _last_t=t

        lo,mid,hi=_split(bands); fx=_flux(bands)
        _env=_env_step(_env, 0.8*lo + 0.5*mid + 0.9*fx + 0.1*rms, up=0.5, down=0.18)

        p.fillRect(r,QBrush(QColor(8,9,12)))
        cols,rows,cell=_setup(w,h,cell=max(8,int(min(w,h)/40)))
        ox=(w-cols*cell)/2; oy=(h-rows*cell)/2

        # drive imprint: low frequency pushes a soft "hand" circle across
        cx = (sin(t*0.6)*0.5+0.5)*(cols-1)
        cy = (cos(t*0.45)*0.5+0.5)*(rows-1)
        rad = 3 + 9*min(1.0,_env)

        for j in range(rows):
            for i in range(cols):
                dx=i-cx; dy=j-cy
                d= sqrt(dx*dx+dy*dy)
                push = max(0.0, 1.0 - d/(rad+0.001))
                # highs create sharp spikes
                spike = 1.0 if random()<0.02*min(1.0,hi*4.0) else 0.0
                target = 0.55*push + 0.35*_env + 0.6*spike
                pin_field[j][i]= 0.82*pin_field[j][i] + 0.18*target

                # draw pin
                x=ox+i*cell; y=oy+j*cell
                depth = pin_field[j][i]
                z = int(50 + 160*depth)
                p.setBrush(QBrush(QColor(z,z,z,255)))
                p.setPen(QPen(QColor(0,0,0,150),1))
                p.drawRect(QRectF(x+1,y+1,cell-2,cell-2))
