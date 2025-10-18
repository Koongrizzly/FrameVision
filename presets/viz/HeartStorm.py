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
    target=0.55*_midhi(bands)+1.35*_flux(bands)+0.3*rms
    target=target/(1+0.7*target)
    if target>_env: _env=0.65*_env+0.35*target
    else: _env=0.90*_env+0.10*target
    # onset gate with hysteresis
    f=_flux(bands)
    hi=0.30; lo=0.18
    g=1.0 if f>hi else (0.0 if f<lo else _gate)
    _gate=0.78*_gate+0.22*g
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate))

def _heart_path(cx, cy, s):
    path=QPainterPath()
    x0,y0=cx, cy-0.3*s
    path.moveTo(x0,y0)
    path.cubicTo(cx-0.5*s, cy-0.9*s, cx-1.2*s, cy+0.2*s, cx, cy+0.9*s)
    path.cubicTo(cx+1.2*s, cy+0.2*s, cx+0.5*s, cy-0.9*s, x0, y0)
    return path

@register_visualizer
class HeartStorm(BaseVisualizer):
    display_name = "Heart Storm"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        env,gate=music_env(bands,rms)
        p.fillRect(r,QBrush(QColor(8,8,14)))
        cx,cy=r.center().x(), r.center().y()
        s=32*(1+0.9*env + (0.25 if gate>0.6 else 0.0))
        p.setPen(QPen(QColor(255,150,180,230), 3))
        p.setBrush(QBrush(QColor(255,80,120,90)))
        p.drawPath(_heart_path(cx,cy,s))
        # outer ring pumps harder with env^1.3 + gate kick
        pump=(env**1.8)*1.0 + (0.6 if gate>0.6 else 0.0)
        count=28
        for i in range(count):
            ang=2*pi*i/count + t*0.7
            L=(60 + 220*pump)
            x=cx+L*cos(ang); y=cy+L*sin(ang)
            hue=int((340 + i*7 + t*50) % 360)
            p.setPen(QPen(QColor.fromHsv(hue,220,255,220), 2))
            p.setBrush(QBrush(QColor.fromHsv(hue,180,230,80)))
            p.drawPath(_heart_path(x,y, 12+8*sin(t*2+i)))
