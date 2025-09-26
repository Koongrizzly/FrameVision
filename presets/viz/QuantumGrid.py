
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

_state={}; _twinkle=0.0
@register_visualizer
class QuantumGrid(BaseVisualizer):
    display_name = "Quantum Grid"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _state,_twinkle
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        cols,rows=20,12
        cw,ch=w/cols,h/rows
        p.fillRect(r,QBrush(QColor(4,6,10)))
        env, gate, boom, punch, bc, dt = beat_drive(bands, rms, t)
        # idle random flips
        for _ in range(3 + int(20*env)):
            x=int(_rng.random()*cols); y=int(_rng.random()*rows)
            _state[(x,y)]=1
        # gate burst
        if gate>0.6:
            for _ in range(60):
                x=int(_rng.random()*cols); y=int(_rng.random()*rows)
                _state[(x,y)]=1
        # propagation on boom
        new_state={}
        for y in range(rows):
            for x in range(cols):
                v=_state.get((x,y),0)
                n=0
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        if dx==0 and dy==0: continue
                        if _state.get(((x+dx)%cols,(y+dy)%rows),0)>0: n+=1
                if v or (boom>0.4 and n>=2) or (env>0.25 and n>=3):
                    new_state[(x,y)]=1
        _state=new_state
        # draw grid faint
        p.setPen(QPen(QColor(30,40,60,120),1))
        for y in range(rows):
            for x in range(cols):
                p.drawRect(QRectF(x*cw, y*ch, cw, ch))
        # draw active cells
        for (x,y),v in _state.items():
            hue=int((t*80 + (x+y)*3)%360)
            p.setBrush(QBrush(QColor.fromHsv(hue,230,255,220)))
            p.setPen(QPen(QColor(30,40,60,120),1))
            p.drawRect(QRectF(x*cw+2, y*ch+2, cw-4, ch-4))
