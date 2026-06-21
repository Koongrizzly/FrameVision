
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

_strokes = []
@register_visualizer
class EchoConductor(BaseVisualizer):
    display_name = "Echo Conductor"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _strokes
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("conductor", 1.0+0.8*env + TUNE_KICK*punch + 0.6*boom, t, hi=3.5)
        # maestro body
        p.setPen(QPen(QColor(220,230,255,200), 4))
        p.drawLine(QPointF(cx, cy-60*amp), QPointF(cx, cy+60*amp))
        # arm
        ang = 1.0*sin(t*2.4) + 1.8*env*sin(t*4.2) + (0.7 if gate>0.6 else 0.0)
        L = 90*amp
        hx = cx + L*cos(ang); hy = cy - 30*amp - L*sin(ang)
        p.drawLine(QPointF(cx, cy-30*amp), QPointF(hx, hy))
        # add stroke trail
        if gate>0.5 or env>0.2:
            _strokes.append([[(hx,hy)], QColor.fromHsv(int((t*80)%360),230,255,200), 1.0])
        # extend and draw trails
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        nxt=[]
        for path,col,life in _strokes:
            if len(path)<80: path.append((hx,hy))
            p.setPen(QPen(col, 3))
            for i in range(1, len(path)):
                x1,y1=path[i-1]; x2,y2=path[i]
                p.drawLine(QPointF(x1,y1), QPointF(x2,y2))
            life*=0.985
            if life>0.15: nxt.append([path,col,life])
        _strokes=nxt
