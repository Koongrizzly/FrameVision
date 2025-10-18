
# --- common beat + spring kit (tweak here) ---
TUNE_KICK = 1.3   # onset kick (bumped from 1.2 to 1.3)
TUNE_BOOM = 1.0   # bass boom weight
SPR_K = 30.0      # spring stiffness
SPR_C = 6.0       # spring damping
SPR_MAX = 4.2     # max scale

from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(1337)
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None

def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = c = 0.0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += w*bands[i]; c += 1
    return s/max(1, c)

def _low(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = c = 0.0
    for i in range(0, cut):
        w = 1.0 - 0.4*(i/max(1,cut-1))
        s += w*bands[i]; c += 1
    return s/max(1, c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = c = 0.0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d*(0.3 + 0.7*((i-cut)/max(1, n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, c)

def beat_drive(bands, rms, t):
    global _env, _gate, _punch, _pt
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target = target/(1+0.7*target)
    if target > _env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi, lo_thr = 0.30, 0.18
    g = 1.0 if f > hi else (0.0 if f < lo_thr else _gate)
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    # fast-decay punch on onsets
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

_sdict = {}
def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=0.25, hi=SPR_MAX):
    s, v, pt = _sdict.get(key, (1.0, 0.0, None))
    if pt is None:
        _sdict[key] = (s, v, t)
        return 1.0
    dt = max(0.0, min(0.033, t-pt))
    a = -k*(s-target) - c*v
    v += a*dt
    s += v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key] = (s, v, t)
    return s

# Binary Pinball
_balls = []
@register_visualizer
class BinaryPinball(BaseVisualizer):
    display_name = "Binary Pinball"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _balls
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        p.fillRect(r, QBrush(QColor(5,6,10)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        if gate>0.6 and len(_balls)<24:
            for i in range(3):
                ang = _rng.random()*2*pi
                spd = 160+260*_rng.random()
                col = QColor.fromHsv(int(_rng.random()*360), 240, 255, 220)
                _balls.append([cx,cy, spd*cos(ang), spd*sin(ang), col, 1.0])
        R = min(w,h)*0.36*(1+0.2*env)
        for a in range(6):
            th = a*2*pi/6 + t*(0.4+1.0*env)
            x = cx + R*cos(th); y = cy + R*sin(th)
            p.setBrush(QBrush(QColor(20,40,80)))
            p.setPen(QPen(QColor(180,220,255,200),3))
            p.drawEllipse(QPointF(x,y), 14, 14)
        dt = 0.016
        for b in _balls:
            b[0] += b[2]*dt; b[1] += b[3]*dt; b[5] *= 0.992
            if b[0]<r.left()+10 or b[0]>r.right()-10: b[2]*=-1
            if b[1]<r.top()+10 or b[1]>r.bottom()-10: b[3]*=-1
            gx = (cx - b[0])*boom*2.0; gy = (cy - b[1])*boom*2.0
            b[2] += gx*dt*60; b[3] += gy*dt*60
            p.setPen(QPen(b[4], 3))
            p.drawLine(QPointF(b[0],b[1]), QPointF(b[0]-b[2]*dt*2, b[1]-b[3]*dt*2))
            p.setBrush(QBrush(b[4])); p.setPen(QPen(QColor(255,255,255,30),1))
            p.drawEllipse(QPointF(b[0],b[1]), 6+4*punch, 6+4*punch)
        _balls = [b for b in _balls if b[5]>0.2]
