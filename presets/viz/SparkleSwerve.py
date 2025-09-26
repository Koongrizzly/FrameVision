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

_pts=[(_rng.random(), _rng.random(), 2*pi*_rng.random()) for _ in range(220)]
@register_visualizer
class SparkleSwerve(BaseVisualizer):
    display_name = "Sparkle Swerve"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,12)))
        env,gate=music_env(bands,rms)
        speed=0.05+0.25*env + (0.08 if gate>0.6 else 0.0)
        for i,(ux,uy,ph) in enumerate(_pts):
            x=r.left()+((ux + speed*sin(t*0.9+ph))%1.0)*w
            y=r.top() +((uy + speed*cos(t*0.8+ph))%1.0)*h
            hue=int((i*3 + t*90) % 360)
            val=170 + int(60*(0.5+0.5*sin(t*4+ph))) + int(60*env) + (40 if gate>0.6 else 0)
            g=QRadialGradient(QPointF(x,y), 8+8*env+(4 if gate>0.6 else 0))
            g.setColorAt(0.0, QColor.fromHsv(hue,220,min(255,val), 220))
            g.setColorAt(1.0, QColor.fromHsv(hue,220,0,0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,10),1))
            p.drawEllipse(QPointF(x,y), 2.2+1.6*env, 2.2+1.6*env)
