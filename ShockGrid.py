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
class ShockGrid(BaseVisualizer):
    display_name = "Shock Grid"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(7,7,12)))
        env,gate=music_env(bands,rms)
        cols,rows=20,12
        kick=0.0 + (1.0 if gate>0.55 else 0.0)
        for iy in range(rows):
            for ix in range(cols):
                x=r.left()+ix*(w/(cols-1)); y=r.top()+iy*(h/(rows-1))
                burst=env*(0.7+0.3*sin(t*2+(ix+iy)*0.25)) + 0.8*kick
                length=10 + 48*burst
                hue=int((t*60 + ix*9 + iy*7)%360)
                p.setPen(QPen(QColor.fromHsv(hue,230,255,220), 2))
                p.drawLine(QPointF(x,y), QPointF(x+length,y))
                p.drawLine(QPointF(x,y), QPointF(x-length,y))
                p.drawLine(QPointF(x,y), QPointF(x,y+length))
                p.drawLine(QPointF(x,y), QPointF(x,y-length))
