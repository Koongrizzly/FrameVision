
# --- Neon Cassette Idol (No Mouth) ---
# Drop-in replacement of NeonCassetteIdol with the mouth/tape-window removed.
# Eyes = reels; everything else unchanged (glow, tape spill on punches).
TUNE_KICK   = 1.2
TUNE_BOOM   = 1.0
SPR_K       = 30.0
SPR_C       = 6.0

EYE_PERIOD_LEFT   = 3
EYE_PERIOD_RIGHT  = 4
MOUTH_PERIOD      = 2     # kept for compatibility but unused (no mouth)

EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0

IDLE_TO_CENTER_SEC     = 2.0

from math import sin, cos, pi
from random import Random, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient
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

# shared eye state (no mouth state needed but kept harmlessly)
_last_bc_eye_L=_last_bc_eye_R=-1
_eye_state_L=1   # 0=left 1=center 2=right
_eye_state_R=2
_eye_last_t_L=_eye_last_t_R=None

def tick_states(bc,t,idle):
    global _last_bc_eye_L,_last_bc_eye_R
    global _eye_state_L,_eye_state_R
    global _eye_last_t_L,_eye_last_t_R
    if _eye_last_t_L is None: _eye_last_t_L=t
    if _eye_last_t_R is None: _eye_last_t_R=t
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
    # fallbacks
    if not idle and t-_eye_last_t_L> EYE_INTERVAL_LEFT_SEC: 
        _eye_state_L=choice([0,1,2]); _eye_last_t_L=t
    if not idle and t-_eye_last_t_R> EYE_INTERVAL_RIGHT_SEC: 
        _eye_state_R=choice([0,1,2]); _eye_last_t_R=t
    # idle force center eyes
    if idle:
        _eye_state_L=_eye_state_R=1
        _eye_last_t_L=_eye_last_t_R=t

@register_visualizer
class NeonCassetteIdol(BaseVisualizer):
    display_name = "Neon Cassette Idol"
    def paint(self, p:QPainter, r, bands, rms, t):
        env,gate,boom,punch,bc,dt,last_on = beat_drive(bands,rms,t)
        idle = (t-last_on)>IDLE_TO_CENTER_SEC
        tick_states(bc,t,idle)
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(8,8,12)))
        cx,cy=r.center().x(), r.center().y()
        scale = spring_to("scale", 1.0+0.6*env+0.9*boom+0.6*punch, t, hi=4.0)

        # body
        bw,bh=min(w,h)*0.70*scale, min(w,h)*0.38*scale
        grad=QLinearGradient(cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
        hue=int((t*50+120*env)%360)
        grad.setColorAt(0,QColor.fromHsv(hue,230,255,200))
        grad.setColorAt(1,QColor.fromHsv((hue+80)%360,230,255,200))
        p.setBrush(QBrush(grad)); p.setPen(QPen(QColor(255,255,255,160),6))
        p.drawRoundedRect(QRectF(cx-bw/2,cy-bh/2,bw,bh),20,20)

        # reels as eyes
        rx = bw*0.28; rr = bh*0.30
        base_rot = spring_to("reel_rot", (env*2.5 + 1.0*punch + 1.8*boom)*t, t, k=10,c=6)
        for side,state,key in ((-1,_eye_state_L,"L"),(1,_eye_state_R,"R")):
            x = cx + side*rx; y=cy-bh*0.05
            p.setPen(QPen(QColor(255,255,255,160),3))
            p.setBrush(QBrush(QColor(10,10,18,180)))
            p.drawEllipse(QPointF(x,y), rr, rr)
            # pupil dot moves L/C/R around the reel
            ang = {0:-pi/2,1:0.0,2:pi/2}[state] + base_rot
            px = x + rr*0.55*cos(ang); py = y + rr*0.55*sin(ang)
            p.setBrush(QBrush(QColor(255,255,255,220))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(px,py), rr*0.10, rr*0.10)

        # *** Mouth removed intentionally ***

        # tape spill on punches (kept)
        if punch>0.6:
            for i in range(12):
                ang=_rng.random()*2*pi
                length = 20+60*_rng.random()
                col=QColor.fromHsv((hue+_rng.randint(0,80))%360,230,255,200)
                p.setPen(QPen(col,2))
                p.drawLine(QPointF(cx, cy), QPointF(cx+length*cos(ang), cy+length*sin(ang)))
