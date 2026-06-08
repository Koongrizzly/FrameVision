
TUNE_KICK   = 1.8
TUNE_BOOM   = 1.5
SPR_K       = 40.0
SPR_C       = 3.0
SPR_MAX     = 6.2

EYE_PERIOD_LEFT   = 1
EYE_PERIOD_RIGHT  = 4
MOUTH_PERIOD      = 2

EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0
MOUTH_INTERVAL_SEC     = 0.85

IDLE_TO_CENTER_SEC     = 60.0

from math import sin, cos, pi
from random import Random, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(424242)

_prev=[]; _env=_gate=_punch=0.0; _pt=None; _beat_count=0; _last_onset_t=0.0

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
    s=c=0.0
    for i in range(cut):
        w=1.0 - 0.4*(i/max(1,cut-1))
        s+=w*bands[i]; c+=1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands: _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n: _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i]+0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_beat_count,_last_onset_t
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target=0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target=target/(1+0.7*target)
    if target>_env: _env=0.72*_env+0.28*target
    else: _env=0.92*_env+0.08*target
    hi,lo_thr=0.06,0.08
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    if g>0.6 and _gate<=0.6:
        _beat_count+=1
        _last_onset_t=t
    _gate=0.82*_gate+0.18*g
    boom=min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    decay=pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch=max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return _env,_gate,boom,_punch,_beat_count,dt,_last_onset_t

_sdict={}
def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=-1e9, hi=1e9):
    s,v,pt=_sdict.get(key,(target,0.0,None))
    if pt is None: _sdict[key]=(target,0.0,t); return target
    dt=max(0.0, min(0.033, t-pt))
    a=-k*(s-target) - c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

# shared eye/mouth state
_last_bc_eye_L=_last_bc_eye_R=_last_bc_mouth=-1
_eye_state_L=1   # 0=left 1=center 2=right (or equivalent per avatar)
_eye_state_R=2
_mouth_state=0   # 0=open 1=close 2=smile
_eye_last_t_L=_eye_last_t_R=_mouth_last_t=None

def tick_states(bc,t,idle):
    global _last_bc_eye_L,_last_bc_eye_R,_last_bc_mouth
    global _eye_state_L,_eye_state_R,_mouth_state
    global _eye_last_t_L,_eye_last_t_R,_mouth_last_t
    if _eye_last_t_L is None: _eye_last_t_L=t
    if _eye_last_t_R is None: _eye_last_t_R=t
    if _mouth_last_t is None: _mouth_last_t=t
    # beat-driven
    if not idle and bc% EYE_PERIOD_LEFT==0 and bc>0 and _last_bc_eye_L!=bc:
        _last_bc_eye_L=bc
        ns=choice([0,1,2])
        if ns==_eye_state_L: ns=(ns+choice([1,2]))%3
        _eye_state_L=ns; _eye_last_t_L=t
    if not idle and bc% EYE_PERIOD_RIGHT==0 and bc>0 and _last_bc_eye_R!=bc:
        _last_bc_eye_R=bc
        ns=choice([0,1,2])
        if ns==_eye_state_R: ns=(ns+choice([1,2]))%3
        _eye_state_R=ns; _eye_last_t_R=t
    if bc% MOUTH_PERIOD==0 and bc>0 and _last_bc_mouth!=bc:
        _last_bc_mouth=bc
        ns=choice([0,1,2])
        if ns==_mouth_state: ns=(ns+choice([1,2]))%3
        _mouth_state=ns; _mouth_last_t=t
    # fallbacks
    if not idle and t-_eye_last_t_L> EYE_INTERVAL_LEFT_SEC: 
        _eye_state_L=choice([0,1,2]); _eye_last_t_L=t
    if not idle and t-_eye_last_t_R> EYE_INTERVAL_RIGHT_SEC: 
        _eye_state_R=choice([0,1,2]); _eye_last_t_R=t
    if not idle and t-_mouth_last_t> MOUTH_INTERVAL_SEC:
        _mouth_state=choice([0,1,2]); _mouth_last_t=t
    # idle force center eyes
    if idle:
        _eye_state_L=_eye_state_R=1
        _eye_last_t_L=_eye_last_t_R=t
    _mouth_state=1
    _mouth_last_t=t

@register_visualizer
class NeonOniMask(BaseVisualizer):
    display_name = "Neon Oni Mask"
    def paint(self, p:QPainter, r, bands, rms, t):
        env,gate,boom,punch,bc,dt,last_on = beat_drive(bands,rms,t)
        idle=(t-last_on)>IDLE_TO_CENTER_SEC
        tick_states(bc,t,idle)
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(4,4,8)))
        cx,cy=r.center().x(), r.center().y()
        s=spring_to("s", 1.0+0.6*env+0.9*boom, t, hi=4.0)

        rad=min(w,h)*0.33*s
        p.setPen(QPen(QColor(255,120,160,220),6)); p.setBrush(QBrush(QColor(12,8,14,230)))
        p.drawEllipse(QPointF(cx,cy), rad, rad*1.1)

        # horn glow
        p.setPen(QPen(QColor(255,120,160,200),8))
        p.drawLine(QPointF(cx-rad*0.4, cy-rad*0.9), QPointF(cx-rad*0.7, cy-rad*1.3))
        p.drawLine(QPointF(cx+rad*0.4, cy-rad*0.9), QPointF(cx+rad*0.7, cy-rad*1.3))

        # eyes as flames: state→horizontal shift of flame center
        for side,state in ((-1,_eye_state_L),(1,_eye_state_R)):
            x=cx+side*rad*0.35 + {0:-rad*0.10,1:0,2:rad*0.10}[state]
            y=cy-rad*0.15
            p.setPen(QPen(QColor(255,160,200,220),4)); p.setBrush(QBrush(QColor(80,20,60,200)))
            p.drawEllipse(QPointF(x,y), rad*0.22, rad*0.14)
            p.setPen(QPen(QColor(255,200,240,220), max(3,int(rad*0.10))))
            p.drawArc(QRectF(x-rad*0.18,y-rad*0.10, rad*0.36, rad*0.20), 0, 16*180)

        # jaw plate
        ms=_mouth_state; mw, mh = rad*1.0, rad*0.28; base_y=cy+rad*0.55
        if ms==0: mh*=1.2*(1.0+0.4*(env+punch))
        elif ms==1: mh*=0.15
        else:
            path=QPainterPath(QPointF(cx-mw*0.5, base_y))
            curve=rad*0.20*(1.0+0.6*(env+punch))
            path.cubicTo(QPointF(cx-mw*0.2, base_y-curve), QPointF(cx+mw*0.2, base_y-curve), QPointF(cx+mw*0.5, base_y))
            p.setPen(QPen(QColor(255,160,200,220), max(4,int(rad*0.10)))); p.setBrush(QBrush(QColor(0,0,0,0)))
            p.drawPath(path)
        if ms!=2:
            p.setBrush(QBrush(QColor(40,10,30,230))); p.setPen(QPen(QColor(255,160,200,220),4))
            p.drawRoundedRect(QRectF(cx-mw/2, base_y-mh/2, mw, mh), 10,10)
