
# --- common beat + spring kit (tweak here) ---
TUNE_KICK = 1.2   # onset kick amount (1.2 felt best earlier)
TUNE_BOOM = 1.0   # bass boom weight
SPR_K = 30.0      # spring stiffness
SPR_C = 6.0       # spring damping
SPR_MAX = 4.2     # max scale (how huge things get)

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

_trails = []
@register_visualizer
class IonDrummer(BaseVisualizer):
    display_name = "Ion Drummer"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _trails
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(5,6,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()+10*sin(t*1.2)
        amp = spring_to("drummer", 1.0+0.8*env + TUNE_KICK*punch + 0.7*boom, t, hi=3.6)
        # torso
        p.setPen(QPen(QColor(220,230,255,220),4))
        p.drawLine(QPointF(cx,cy-40*amp), QPointF(cx, cy+40*amp))
        # sticks
        for sgn in (-1,1):
            ang = 0.8*sgn + 1.4*sgn*sin(t*3.2) + 1.3*env*sgn
            L = 90*amp
            x2 = cx + L*cos(ang); y2 = cy - 20*amp - L*sin(ang)
            col = QColor.fromHsv(int((t*80 + (0 if sgn<0 else 180))%360),230,255,220)
            _trails.append([(x2,y2), col, 1.0])
            p.setPen(QPen(col,4)); p.drawLine(QPointF(cx, cy-20*amp), QPointF(x2,y2))
        # draw trails
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        nxt=[]
        for (x,y),col,life in _trails:
            p.setPen(QPen(col,3)); p.drawEllipse(QPointF(x,y), 6, 6)
            life*=0.96
            if life>0.2: nxt.append([(x,y),col,life])
        _trails=nxt
