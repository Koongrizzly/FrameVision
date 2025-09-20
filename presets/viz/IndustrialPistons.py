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
class IndustrialPistons(BaseVisualizer):
    display_name = "Industrial Pistons"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        env,gate=music_env(bands,rms)
        p.fillRect(r,QBrush(QColor(10,10,12)))
        lanes=4
        for i in range(lanes):
            y=r.top()+(i+0.5)*h/lanes
            hue=int((t*20+i*40)%360)
            stroke=(w*0.35)*(0.6+1.4*env)
            speed=1.0+0.8*env + (0.6 if i%2==0 else 0.0)
            x=r.center().x()+stroke*sin(t*speed + i)
            p.setPen(QPen(QColor.fromHsv(hue,220,255,200), 4+int(1.5*env)))
            p.drawLine(QPointF(x-90,y), QPointF(x+90,y))
            p.setPen(QPen(QColor.fromHsv(hue,220,200,150), 6))
            p.drawLine(QPointF(x,y-22), QPointF(x,y+22))
            if gate>0.55:
                for k in range(8):
                    ang=2*pi*_rng.random(); L=25+70*_rng.random()
                    col=QColor.fromHsv(int(_rng.random()*360),220,255,220)
                    p.setPen(QPen(col,2))
                    p.drawLine(QPointF(x,y), QPointF(x+L*cos(ang), y+L*sin(ang)))
