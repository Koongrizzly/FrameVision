
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

_pts=[]
@register_visualizer
class ParticleFountain(BaseVisualizer):
    display_name = "Particle Fountain"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _pts
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,10)))
        env, gate, boom, punch, bc, dt = beat_drive(bands, rms, t)
        # emit continuously
        n_emit = int(20 + 120*env + (60 if gate>0.6 else 0))
        for i in range(n_emit):
            ang = -pi/2 + (_rng.random()-0.5)*0.7
            spd = 140+240*_rng.random() + 200*env
            col = QColor.fromHsv(int(_rng.random()*360),230,255,200)
            _pts.append([w/2, h-10, spd*cos(ang), spd*sin(ang), col, 1.0])
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        nxt=[]
        for x,y,vx,vy,col,life in _pts:
            vy += 200*dt
            x += vx*dt; y += vy*dt
            life *= 0.985
            p.setPen(QPen(col,2)); p.drawPoint(QPointF(x,y))
            if life>0.2 and y<h+50: nxt.append([x,y,vx,vy,col,life])
        _pts=nxt[-1800:]  # cap
