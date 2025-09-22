
# --- Chromaforge (rotating ingot + hue shifts) ---
TUNE_KICK = 1.2
TUNE_BOOM = 1.0
SPR_K = 30.0
SPR_C = 6.0
SPR_MAX = 4.0

from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QTransform
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(2025)
_prev=[]; _env=_gate=_punch=0.0; _pt=None
_phi=0.0; _omega=0.0; _last_t=None
_shards=[]

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
    global _env,_gate,_punch,_pt
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target=0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target=target/(1+0.7*target)
    if target>_env: _env=0.72*_env+0.28*target
    else: _env=0.92*_env+0.08*target
    hi,lo_thr=0.30,0.18
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate=0.82*_gate+0.18*g
    boom=min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    # fast punch
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return _env,_gate,boom,_punch,dt

@register_visualizer
class Chromaforge(BaseVisualizer):
    display_name = "Chromaforge"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _phi,_omega,_last_t,_shards
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,6,6)))
        env, gate, boom, punch, dt = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()

        # angular velocity kicks on onsets, damped over time
        if gate>0.6: _omega += 2.6 + 3.4*boom + 1.2*punch
        _omega *= 0.965
        _phi += _omega * dt  # integrate

        # hue shift with energy + onset flash
        base_hue = (t*50 + 120*env + (40 if gate>0.6 else 0)) % 360

        # draw rotated ingot
        iw, ih = min(w,h)*0.45, min(w,h)*0.12
        p.save()
        p.translate(cx, cy)
        p.rotate(_phi*180/pi)
        grad = QLinearGradient(-iw/2, -ih/2, iw/2, ih/2)
        grad.setColorAt(0.0, QColor.fromHsv(int(base_hue)%360, 230, 255, 220))
        grad.setColorAt(1.0, QColor.fromHsv(int(base_hue+40)%360, 230, 255, 220))
        p.setBrush(QBrush(grad)); p.setPen(QPen(QColor(255,255,255,160),3))
        p.drawRoundedRect(QRectF(-iw/2, -ih/2, iw, ih), 14, 14)
        p.restore()

        # spawn shards on onsets (emit oriented by rotation)
        if gate>0.6:
            for i in range(32):
                ang = _phi + _rng.random()*2*pi
                spd = 200+400*_rng.random()
                hue = int((base_hue + _rng.random()*60)%360)
                _shards.append([cx,cy, spd*cos(ang), spd*sin(ang), hue, 1.0])
        # draw shards
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        nxt=[]
        for s in _shards:
            s[0]+=s[2]*dt; s[1]+=s[3]*dt; s[5]*=0.985; s[3]+=80*dt  # gravity
            col = QColor.fromHsv(int(s[4])%360,230,255,int(200*(0.6+0.4*punch)))
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(s[0],s[1]), QPointF(s[0]-s[2]*dt*2, s[1]-s[3]*dt*2))
            if s[5]>0.25: nxt.append(s)
        _shards = nxt
