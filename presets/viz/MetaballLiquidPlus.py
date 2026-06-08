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
class MetaballLiquidPlus(BaseVisualizer):
    display_name = "Metaball Liquid (Color+)"
    def paint(self,p:QPainter,r,bands,rms,t):
        # This is a simple replacement if the original isn't available.
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        env,gate=music_env(bands,rms)
        p.fillRect(r,QBrush(QColor(6,6,10)))
        cx,cy=r.center().x(), r.center().y()
        blobs=8
        for i in range(blobs):
            ang=2*pi*i/blobs + t*(0.4+1.6*env)
            rad= min(w,h)*0.18*(0.8+0.6*env)
            x=cx + rad*cos(ang); y=cy + 0.85*rad*sin(ang)
            hue=int((t*70 + i*30) % 360)
            g=QRadialGradient(QPointF(x,y), 40+20*env+(10 if gate>0.6 else 0))
            g.setColorAt(0.0, QColor.fromHsv(hue, 220, 255, 200))
            g.setColorAt(1.0, QColor.fromHsv(hue, 220, 0, 0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,20),1))
            p.drawEllipse(QPointF(x,y), 20+10*env, 20+10*env)
