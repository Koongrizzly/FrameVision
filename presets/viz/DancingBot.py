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
class DancingBot(BaseVisualizer):
    display_name = "Dancing Bot"
    def paint(self,p:QPainter,r,bands,rms,t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(8,9,16)))
        env,gate=music_env(bands,rms)
        cx,cy=r.center().x(), r.center().y() + 20*sin(t*0.8)
        scale=(0.9+0.7*env)
        body_w=90*scale; body_h=120*scale
        p.setBrush(QBrush(QColor(100,180,255)))
        p.setPen(QPen(QColor(255,255,255,150),2))
        p.drawRoundedRect(QRectF(cx-body_w/2, cy-body_h/2, body_w, body_h), 14,14)
        head_r=30*scale*(1+0.25*env + (0.15 if gate>0.6 else 0.0))
        hx=cx+10*sin(t*1.8); hy=cy-body_h/2-head_r*1.3+10*sin(t*4+env*6)
        p.setBrush(QBrush(QColor(255,220,120)))
        p.setPen(QPen(QColor(255,255,255,180),2))
        p.drawEllipse(QPointF(hx,hy), head_r, head_r)
        # eyes
        p.setBrush(QBrush(QColor(30,30,30)))
        p.setPen(QPen(QColor(0,0,0,0),0))
        p.drawEllipse(QPointF(hx-8*scale, hy-3*scale), 3*scale,3*scale)
        p.drawEllipse(QPointF(hx+8*scale, hy-3*scale), 3*scale,3*scale)
        # arms swing harder
        arm_len=65*scale*(1+0.4*env)
        ang=1.1*sin(t*2.5) + 1.6*env*sin(t*4.2)
        p.setPen(QPen(QColor(255,180,200), 5))
        ax1=cx-body_w/2; ay1=cy-18*scale
        p.drawLine(QPointF(ax1,ay1), QPointF(ax1-arm_len*cos(ang), ay1-arm_len*sin(ang)))
        bx1=cx+body_w/2; by1=cy-18*scale
        p.drawLine(QPointF(bx1,by1), QPointF(bx1+arm_len*cos(ang), by1-arm_len*sin(ang)))
        # feet stomp with shocks
        p.setPen(QPen(QColor(180,255,200), 5))
        spread=34*scale
        yfoot=cy+body_h/2 + (14*env + (10 if gate>0.6 else 0))*sin(t*6)
        p.drawLine(QPointF(cx-spread,yfoot), QPointF(cx-spread-16, yfoot+9))
        p.drawLine(QPointF(cx+spread,yfoot), QPointF(cx+spread+16, yfoot+9))
        if gate>0.65:
            for i in range(12):
                ang2=2*pi*_rng.random()
                d=(30+60*_rng.random())*(0.8+0.6*env)
                col=QColor.fromHsv(int(_rng.random()*360),220,255,220)
                p.setPen(QPen(col,2))
                p.drawLine(QPointF(hx,hy), QPointF(hx+d*cos(ang2), hy+d*sin(ang2)))
