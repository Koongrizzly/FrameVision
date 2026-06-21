from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Aggressive mid/high + spectral flux env with onset gate
_rng = Random(13579)
_prev = []
_env = 0.0
_gate = 0.0

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=0.0; c=0
    for i in range(cut,n):
        w=0.4+0.6*((i-cut)/max(1,n-cut))
        s += w*bands[i]; c += 1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n:
        _prev=[0.0]*n
    cut=max(1,n//6); f=0.0; c=0
    for i in range(cut,n):
        d=bands[i]-_prev[i]
        if d>0: f += (0.3+0.7*((i-cut)/max(1,n-cut)))*d
        c+=1
    _prev=[0.85*_prev[i]+0.15*bands[i] for i in range(n)]
    return f/max(1,c)

def music_env(bands,rms):
    global _env,_gate
    target=0.55*_midhi(bands)+1.35*_flux(bands)+0.2*rms
    target=target/(1+0.7*target)
    if target>_env: _env=0.65*_env+0.35*target
    else: _env=0.90*_env+0.10*target
    # onset gate with hysteresis
    f=_flux(bands)
    hi=0.30; lo=0.18
    g=1.0 if f>hi else (0.0 if f<lo else _gate)
    _gate=0.78*_gate+0.22*g
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate))

@register_visualizer
class NeonHexGrid(BaseVisualizer):
    display_name = "Neon Hex Grid"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(8,9,16)))
        env,gate=music_env(bands,rms)
        size=max(14,int(min(w,h)*0.045))
        hx=size*3/2; hy=size*sqrt(3)/2
        rows=int(h/hy)+2; cols=int(w/hx)+3
        twist=0.25 + 0.9*env + (0.3 if gate>0.6 else 0.0)
        for row in range(rows):
            for col in range(cols):
                cx=r.left()+col*hx+(hx/2 if row%2 else 0)
                cy=r.top()+row*hy
                hue=int((row*20+col*12+t*50)%360)
                p.setPen(QPen(QColor.fromHsv(hue,220,250,200),2))
                pts=[]
                for k in range(6):
                    ang=k*pi/3 + twist*sin(t*0.9 + (row+col)*0.2)
                    rad=size*(0.9+0.5*env)
                    pts.append(QPointF(cx+rad*cos(ang), cy+rad*sin(ang)))
                for k in range(6):
                    p.drawLine(pts[k], pts[(k+1)%6])
