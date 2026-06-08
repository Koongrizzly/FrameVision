
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

_polarity = 1.0
@register_visualizer
class LumenMagnets(BaseVisualizer):
    display_name = "Lumen Magnets"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _polarity
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        env, gate, boom, punch = beat_drive(bands, rms, t)
        if gate>0.6: _polarity *= -1.0
        p.fillRect(r,QBrush(QColor(4,6,10)))
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("magnets", 1.0+0.7*env + TUNE_KICK*punch + 1.1*boom, t, hi=4.0)
        d = min(w,h)*0.30*amp
        a = QPointF(cx-d, cy); b = QPointF(cx+d, cy)
        p.setPen(QPen(QColor(200,220,255,160),3))
        p.drawEllipse(a, 18, 18); p.drawEllipse(b,18,18)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for k in range(24):
            yy = cy + (k-12)*8
            ctrl = QPointF(cx, yy + _polarity*40*sin(t*2+ k*0.2))
            col = QColor.fromHsv(int((t*80 + k*10)%360),230,255,200)
            p.setPen(QPen(col,2))
            p.drawLine(QPointF(a.x(), yy), ctrl)
            p.drawLine(ctrl, QPointF(b.x(), yy))
