
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

def branch(p, x, y, ang, lenr, depth, hue):
    if depth<=0 or lenr<6: return
    x2 = x + lenr*cos(ang); y2 = y + lenr*sin(ang)
    col = QColor.fromHsv(int(hue)%360, 230, 200, 220)
    p.setPen(QPen(col, max(1,int(lenr*0.04))))
    p.drawLine(QPointF(x,y), QPointF(x2,y2))
    branch(p, x2, y2, ang+0.5, lenr*0.66, depth-1, hue+18)
    branch(p, x2, y2, ang-0.6, lenr*0.6, depth-1, hue+28)

@register_visualizer
class FractalBloom(BaseVisualizer):
    display_name = "Fractal Bloom"
    def paint(self, p:QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,10)))
        env, gate, boom, punch, bc = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()
        depth = 3 + int(2*boom) + (1 if gate>0.6 else 0)
        base = min(w,h)*0.12*(1+0.6*env + 0.8*punch)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(6):
            ang = i*2*pi/6 + 0.3*sin(t*0.7+i)
            hue = t*40 + i*60
            branch(p, cx, cy, ang, base, depth, hue)
        # leaf sparks
        if _rng.random()<0.3*(env+0.2):
            hue = int((t*90)%360)
            p.setPen(QPen(QColor.fromHsv(hue,230,255,220),2))
            p.drawPoint(QPointF(cx+_rng.random()*w*0.4-w*0.2, cy+_rng.random()*h*0.4-h*0.2))
