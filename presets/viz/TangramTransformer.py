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

pieces=None
phase=0.0
def _init():
    global pieces
    # seven tangram polygons (triangles, square, parallelogram) in normalized coords
    # We'll define as simple triangles/quads and animate transforms.
    pieces=[
        [QPointF(0,0), QPointF(1,0), QPointF(0,1)], # tri 1
        [QPointF(1,1), QPointF(1,0), QPointF(0,1)], # tri 2
        [QPointF(0,0.5), QPointF(0.5,1), QPointF(1,0.5), QPointF(0.5,0)], # square
        [QPointF(0,0), QPointF(0.5,0.5), QPointF(1,0)], # tri 3
        [QPointF(0,1), QPointF(0.5,0.5), QPointF(1,1)], # tri 4
        [QPointF(0.25,0.25), QPointF(0.75,0.25), QPointF(0.5,0.75)], # tri small 1
        [QPointF(0.25,0.75), QPointF(0.75,0.75), QPointF(0.5,0.25)], # tri small 2
    ]

@register_visualizer
class TangramTransformer(BaseVisualizer):
    display_name = "Tangram Transformer"
    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t, pieces, phase
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:return
        if pieces is None: _init()
        if _last_t is None:_last_t=t
        dt=max(0.0,min(0.05,t-_last_t)); _last_t=t
        lo,mid,hi=_split(bands); fx=_flux(bands)
        phase += dt * (0.5 + 2.0*mid + 0.5*lo)

        p.fillRect(r,QBrush(QColor(10,12,14)))
        cx,cy=w/2,h/2; S=min(w,h)*0.55
        for idx,poly in enumerate(pieces):
            hue = int((idx*40 + t*20)%360)
            p.setBrush(QBrush(QColor.fromHsv(hue, 170, 230, 255)))
            p.setPen(QPen(QColor(10,10,10),2))
            # transform: rotate and place in a morphing arrangement
            ang = phase + idx*0.7
            ox = cx + S*0.25*sin(phase*0.7 + idx)
            oy = cy + S*0.25*cos(phase*0.9 + idx*0.5)
            pts = []
            for q in poly:
                x = (q.x()-0.5)*S*0.5; y=(q.y()-0.5)*S*0.5
                xr = x*cos(ang) - y*sin(ang)
                yr = x*sin(ang) + y*cos(ang)
                pts.append(QPointF(ox+xr, oy+yr))
            p.drawPolygon(QPolygonF(pts))
        # snap sparks on highs
        if hi+fx>0.18:
            p.setPen(QPen(QColor(255,255,255,150),3))
            p.drawLine(QPointF(cx-S*0.3, cy), QPointF(cx+S*0.3, cy))
