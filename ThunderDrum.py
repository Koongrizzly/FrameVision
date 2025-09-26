
# --- common beat + spring kit (tweak here) ---
TUNE_KICK = 1.2   # onset kick amount (1.2 = solid hit; raise to 1.3 for harder)
TUNE_BOOM = 1.0   # bass boom weight
SPR_K = 30.0      # spring stiffness
SPR_C = 6.0       # spring damping
SPR_MAX = 4.2     # max scale (screen fill)

from math import sin, cos, pi, floor
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(4242)
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None
_beat_count = 0

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
    global _env, _gate, _punch, _pt, _beat_count
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target = target/(1+0.7*target)
    if target > _env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi, lo_thr = 0.30, 0.18
    g = 1.0 if f > hi else (0.0 if f < lo_thr else _gate)
    if g>0.6 and _gate<=0.6: _beat_count += 1
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    # fast-decay punch on onsets
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch)), _beat_count

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

_rings = []
_stars = []
@register_visualizer
class ThunderDrum(BaseVisualizer):
    display_name = "Thunder Drum"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _rings,_stars
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        if not _stars:
            for i in range(120):
                _stars.append([_rng.random()*w, _rng.random()*h, _rng.random()*0.8+0.2])
        p.fillRect(r,QBrush(QColor(4,4,8)))
        env, gate, boom, punch, bc = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()
        # membrane
        R = min(w,h)*0.22*spring_to("drum", 1.0+0.7*env + TUNE_KICK*punch + 1.2*boom, t, hi=4.0)
        p.setBrush(QBrush(QColor(20,20,30))); p.setPen(QPen(QColor(200,220,255,200),3))
        p.drawEllipse(QPointF(cx,cy), R,R)
        if gate>0.6: _rings.append([0.0, QColor.fromHsv(int((t*80)%360),230,255,220)])
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        nxt=[]
        for age,col in _rings:
            rad = R + age*160 + 120*boom
            p.setPen(QPen(col,3)); p.setBrush(QBrush(QColor(0,0,0,0)))
            p.drawEllipse(QPointF(cx,cy), rad, rad)
            age += 0.05
            if rad<max(w,h)*1.2: nxt.append([age,col])
        _rings=nxt
        # starfield distortion
        for i in range(len(_stars)):
            x,y,s=_stars[i]
            y += (0.2+boom)*s
            if y>h: y-=h
            _stars[i][1]=y
            if ((x-cx)**2+(y-cy)**2)**0.5 < R*1.1:
                p.setPen(QPen(QColor(255,255,255,200),1))
            else:
                p.setPen(QPen(QColor(140,160,255,120),1))
            p.drawPoint(QPointF(x,y))
