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
class HyperBloom(BaseVisualizer):
    display_name = "Hyper Bloom"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        cx,cy=r.center().x(), r.center().y()
        env,gate=music_env(bands,rms)
        petals=11
        loops=3
        R=min(w,h)*0.45*(0.7+1.2*env + (0.25 if gate>0.6 else 0.0))
        N=800
        rot=t*(0.8 + 2.2*env)
        for j in range(loops):
            hue=int((t*40 + j*50)%360)
            p.setPen(QPen(QColor.fromHsv(hue,220,255,200), 2))
            prev=None
            for i in range(N):
                th=2*pi*i/N + rot + j*0.4
                rads=R*(0.45+0.55*sin(petals*th + 0.8*t + 1.2*env))
                x=cx+rads*cos(th); y=cy+rads*sin(th)
                if prev: p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
                prev=(x,y)
