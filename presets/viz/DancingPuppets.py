
# ---- shared beat + spring helpers ----
TUNE_KICK = 1.2
TUNE_BOOM = 1.0
SPR_K = 28.0
SPR_C = 6.0
SPR_MAX = 4.0

from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(999)
_prev=[]; _env=0.0; _gate=0.0; _punch=0.0; _pt=None; _beat_count=0

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut, n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s+=w*bands[i]; c+=1
    return s/max(1,c)

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    return sum(bands[:cut])/max(1,cut)

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n:
        _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_beat_count
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target=0.56*e + 1.20*f + 0.18*rms + 0.26*lo*TUNE_BOOM
    target=target/(1+0.7*target)
    if target>_env: _env=0.72*_env + 0.28*target
    else: _env=0.92*_env + 0.08*target
    hi,lo_thr=0.27,0.16
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    if g>0.6 and _gate<=0.6: _beat_count+=1
    _gate=0.82*_gate + 0.18*g
    boom=min(1.0, max(0.0, lo*1.3 + 0.45*rms))
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    decay = pow(0.80, dt/0.016) if dt>0 else 0.80
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch)), _beat_count, dt

_sdict={}
def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=0.25, hi=SPR_MAX):
    s, v, pt = _sdict.get(key, (1.0, 0.0, None))
    if pt is None:
        _sdict[key]=(s,v,t); return 1.0
    dt=max(0.0, min(0.033, t-pt))
    a=-k*(s-target) - c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

_pose=0
@register_visualizer
class DancingPuppets(BaseVisualizer):
    display_name = "Dancing Puppets"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _pose
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,10)))
        env, gate, boom, punch, bc, dt = beat_drive(bands, rms, t)
        if gate>0.6: _pose=( _pose+1 )%6  # more distinct poses
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("dp_amp", 1.0+0.7*env + TUNE_KICK*punch + 0.6*boom, t, hi=3.4)
        # puppet bar + strings
        bar_w = min(w,600)*0.5
        p.setPen(QPen(QColor(180,200,255,180),5)); p.drawLine(QPointF(cx-bar_w/2, cy-180*amp), QPointF(cx+bar_w/2, cy-180*amp))
        # body anchor points
        anchors=[(cx-60*amp, cy-100*amp),(cx+60*amp, cy-100*amp),(cx, cy-135*amp)]
        p.setPen(QPen(QColor(200,220,255,140),2))
        for ax,ay in anchors:
            p.drawLine(QPointF(ax, cy-180*amp), QPointF(ax, ay))
        # body
        p.setPen(QPen(QColor(120,240,180,220),8))
        p.drawLine(QPointF(cx, cy-110*amp), QPointF(cx, cy+40*amp))  # spine
        # head
        p.setBrush(QBrush(QColor(250,250,255,230))); p.setPen(QPen(QColor(40,40,60,220),3))
        p.drawEllipse(QPointF(cx, cy-140*amp), 22*amp, 26*amp)
        # limbs per pose
        poses=[(-1.1,0.9,  1.0,-0.6),   # L-arm,R-arm,L-leg,R-leg angles
               (-0.2,1.4,  0.4,-1.1),
               (1.2,-0.3,  -0.6,0.9),
               (-1.5,-0.5, 0.9,0.9),
               (0.2,0.2,   -1.0,1.0),
               (1.3,-1.1,  -0.2,0.4)]
        la,ra,ll,rl = poses[_pose]
        L = 70*amp; p.setPen(QPen(QColor(255,120,220,220),7))
        # arms
        p.drawLine(QPointF(cx, cy-90*amp), QPointF(cx+L*cos(la), cy-90*amp - L*sin(la)))
        p.drawLine(QPointF(cx, cy-90*amp), QPointF(cx+L*cos(ra), cy-90*amp - L*sin(ra)))
        # legs
        p.setPen(QPen(QColor(120,200,255,220),9))
        p.drawLine(QPointF(cx, cy+40*amp), QPointF(cx+L*0.9*cos(ll), cy+40*amp + L*0.9*sin(ll)))
        p.drawLine(QPointF(cx, cy+40*amp), QPointF(cx+L*0.9*cos(rl), cy+40*amp + L*0.9*sin(rl)))
