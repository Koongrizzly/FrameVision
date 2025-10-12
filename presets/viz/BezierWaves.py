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
class BezierWaves(BaseVisualizer):
    display_name = "Bézier Waves"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(5,6,12)))
        env,gate=music_env(bands,rms)
        rows=10
        for iy in range(rows):
            hue=int((t*40+iy*35)%360)
            val=220 if gate>0.6 and iy%2==0 else 200
            p.setPen(QPen(QColor.fromHsv(hue,220,val,220), 2+int(1.5*env)))
            y=r.top()+(iy+0.5)*h/rows
            x1=r.left(); x2=r.right()
            ctrlx=(x1+x2)/2
            amp=(0.06*h)*(1+2.4*env)  # stronger
            v=bands[iy%len(bands)] if bands else 0.0
            c1=QPointF(ctrlx, y - amp*(sin(t*1.2+iy*0.7)+2.0*v))
            c2=QPointF(ctrlx, y + amp*(cos(t*1.1+iy*0.6)+2.0*v))
            path=QPainterPath(QPointF(x1,y))
            path.cubicTo(c1,c2,QPointF(x2,y))
            p.drawPath(path)
            # subtle RGB mis-register on peaks
            if gate>0.55:
                p.setPen(QPen(QColor.fromHsv((hue+120)%360,220,200,140),1))
                p.drawPath(path.translated(2,-1))
                p.setPen(QPen(QColor.fromHsv((hue+240)%360,220,200,140),1))
                p.drawPath(path.translated(-2,1))
