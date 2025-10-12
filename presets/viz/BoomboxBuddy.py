
TUNE_KICK   = 1.9
TUNE_BOOM   = 1.8
SPR_K       = 90.0
SPR_C       = 5.0
SPR_MAX     = 3.2

EYE_PERIOD_LEFT   = 1
EYE_PERIOD_RIGHT  = 1
MOUTH_PERIOD      = 1

EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0
MOUTH_INTERVAL_SEC     = 0.55

IDLE_TO_CENTER_SEC     = 5.0
FREEZE_WHEN_IDLE   = True
IDLE_ENV_THR       = 0.04

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
    hi,lo_thr=0.22,0.10
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    if g>0.5 and _gate<=0.5:
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
    if False and t-_eye_last_t_L> EYE_INTERVAL_LEFT_SEC: 
        _eye_state_L=choice([0,1,2]); _eye_last_t_L=t
    if False and t-_eye_last_t_R> EYE_INTERVAL_RIGHT_SEC: 
        _eye_state_R=choice([0,1,2]); _eye_last_t_R=t
    if False and t-_mouth_last_t> MOUTH_INTERVAL_SEC:
        _mouth_state=choice([0,1,2]); _mouth_last_t=t
    # idle force center eyes
if idle:
    _eye_state_L=_eye_state_R=1

@register_visualizer
class BoomboxBuddy(BaseVisualizer):
    display_name = "Boombox Buddy"
    def paint(self, p:QPainter, r, bands, rms, t):
        env,gate,boom,punch,bc,dt,last_on = beat_drive(bands,rms,t)
        idle = (FREEZE_WHEN_IDLE and ((t-last_on) > IDLE_TO_CENTER_SEC) and (env < IDLE_ENV_THR))
        tick_states(bc,t,idle)
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,8,10)))
        cx,cy=r.center().x(), r.center().y()
        s=spring_to("s", 1.0+0.5*env+0.9*boom, t, hi=3.8)
        bw,bh=min(w,h)*0.78*s, min(w,h)*0.38*s
        p.setBrush(QBrush(QColor(20,22,30,220))); p.setPen(QPen(QColor(160,200,255,180),5))
        p.drawRoundedRect(QRectF(cx-bw/2,cy-bh/2,bw,bh),16,16)
        # handle
        p.setPen(QPen(QColor(160,200,255,150),6))
        p.drawLine(QPointF(cx-bw*0.3, cy-bh*0.55), QPointF(cx+bw*0.3, cy-bh*0.55))

        # speakers as eyes
        rr=bh*0.36*(1.0+0.25*(env+punch))
        for side,state in ((-1,_eye_state_L),(1,_eye_state_R)):
            x=cx+side*bw*0.30; y=cy-bh*0.02
            depth = 0.25 + 0.5*(env + (0.6 if state==1 else 0.0))
            p.setPen(QPen(QColor(90,120,255,200),3)); p.setBrush(QBrush(QColor(10,12,20,180)))
            p.drawEllipse(QPointF(x,y), rr, rr)
            p.setBrush(QBrush(QColor(90,120,255,120)))
            p.drawEllipse(QPointF(x,y), rr*0.65, rr*0.65*(1.0-depth))
            p.setBrush(QBrush(QColor(200,220,255,220)))
            p.drawEllipse(QPointF(x,y), rr*0.22, rr*0.22)

        # mouth: deck slot
        ms=_mouth_state; mw,mh=bw*0.42,bh*0.18
        if ms==0: mh*=1.2*(1.0+0.4*(env+punch))
        elif ms==1: mh*=0.18
        else:
            path=QPainterPath(QPointF(cx-mw*0.5, cy+bh*0.15))
            curve=bh*0.16*(1.0+0.5*(env+punch))
            path.cubicTo(QPointF(cx-mw*0.2, cy+bh*0.15-curve),
                         QPointF(cx+mw*0.2, cy+bh*0.15-curve),
                         QPointF(cx+mw*0.5, cy+bh*0.15))
            p.setPen(QPen(QColor(180,255,180,220), max(4,int(bh*0.10))))
            p.setBrush(QBrush(QColor(0,0,0,0))); p.drawPath(path)
        if ms!=2:
            p.setBrush(QBrush(QColor(30,40,30,200))); p.setPen(QPen(QColor(180,255,180,220),4))
            p.drawRoundedRect(QRectF(cx-mw/2, cy+bh*0.02 - mh/2, mw, mh),8,8)
