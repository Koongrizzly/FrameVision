
TUNE_KICK   = 1.2
TUNE_BOOM   = 1.0
SPR_K       = 30.0
SPR_C       = 6.0
SPR_MAX     = 4.2

EYE_PERIOD_LEFT   = 3
EYE_PERIOD_RIGHT  = 4
MOUTH_PERIOD      = 2

EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0
MOUTH_INTERVAL_SEC     = 0.55

IDLE_TO_CENTER_SEC     = 2.0

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
    hi,lo_thr=0.30,0.18
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
    if t-_mouth_last_t> MOUTH_INTERVAL_SEC:
        _mouth_state=choice([0,1,2]); _mouth_last_t=t
    # idle force center eyes
    if idle:
        _eye_state_L=_eye_state_R=1
        _eye_last_t_L=_eye_last_t_R=t

@register_visualizer
class PixelRobot(BaseVisualizer):
    display_name = "Pixel Robot"
    def paint(self, p:QPainter, r, bands, rms, t):
        env,gate,boom,punch,bc,dt,last_on = beat_drive(bands,rms,t)
        idle=(t-last_on)>IDLE_TO_CENTER_SEC
        tick_states(bc,t,idle)
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,10)))
        cx,cy=r.center().x(), r.center().y()
        s=spring_to("s", 1.0+0.6*env+0.9*boom, t, hi=3.8)
        size=min(w,h)*0.60*s

        # head box
        p.setBrush(QBrush(QColor(20,26,36,230))); p.setPen(QPen(QColor(140,200,255,220),6))
        p.drawRoundedRect(QRectF(cx-size/2, cy-size*0.40, size, size*0.80), 16,16)

        # camera lens eyes (state→aperture x offset or iris size)
        for side,state in ((-1,_eye_state_L),(1,_eye_state_R)):
            x=cx+side*size*0.28; y=cy-size*0.15
            p.setBrush(QBrush(QColor(12,16,24,230))); p.setPen(QPen(QColor(160,220,255,200),4))
            p.drawEllipse(QPointF(x,y), size*0.14, size*0.14)
            iris=size*0.06*(1.0 + (0.3 if state==1 else 0.0))
            px = x + {0:-size*0.03,1:0,2:size*0.03}[state]
            p.setBrush(QBrush(QColor(180,240,255,230))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(px,y), iris, iris)

        # LED matrix mouth
        ms=_mouth_state
        grid_w, grid_h = 10, 4
        cw = size*0.045; ch = size*0.045
        ox = cx - cw*grid_w/2; oy = cy + size*0.10
        for gy in range(grid_h):
            for gx in range(grid_w):
                on=False
                if ms==0:  # OPEN rectangle
                    on = 1<=gx<=8 and 1<=gy<=2
                elif ms==1:  # CLOSE line
                    on = gy==2
                else:  # SMILE curve
                    on = (gy==3 and gx in (2,3,6,7)) or (gy==2 and gx in (1,4,5,8))
                col = QColor(120,220,180,220) if on else QColor(30,40,40,140)
                p.setBrush(QBrush(col)); p.setPen(QPen(QColor(0,0,0,0),0))
                p.drawRoundedRect(QRectF(ox+gx*cw, oy+gy*ch, cw-3, ch-3), 3,3)
