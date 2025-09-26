from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Mid/High envelope + spectral flux + onset gate ---
_rng = Random(424242)
_prev = []
_env = 0.0
_gate = 0.0

def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = 0.0; c = 0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += w*bands[i]; c += 1
    return s/max(1, c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev) != n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = 0.0; c = 0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d * (0.3 + 0.7*((i-cut)/max(1, n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, c)

def music_env(bands, rms):
    global _env, _gate
    e = _midhi(bands)
    f = _flux(bands)
    target = 0.55*e + 1.30*f + 0.20*rms
    target = target/(1+0.7*target)
    if target > _env: _env = 0.70*_env + 0.30*target
    else: _env = 0.92*_env + 0.08*target
    thr_hi, thr_lo = 0.30, 0.18
    g = 1.0 if f > thr_hi else (0.0 if f < thr_lo else _gate)
    _gate = 0.82*_gate + 0.18*g
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate))

# Simple spring utility for bouncy moves
_s_state = {"s":1.0,"v":0.0,"t":None}
def spring_to(target, t, k=24.0, c=6.0, smin=0.3, smax=2.5):
    s=_s_state["s"]; v=_s_state["v"]; pt=_s_state["t"]
    if pt is None: _s_state["t"]=t; _s_state["s"]=s; _s_state["v"]=v; return 1.0
    dt = max(0.0, min(0.033, t-pt)); _s_state["t"]=t
    a = -k*(s-target) - c*v
    v += a*dt; s += v*dt
    if s<smin: s=smin
    if s>smax: s=smax
    _s_state["s"]=s; _s_state["v"]=v
    return s

@register_visualizer
class ToyRockets(BaseVisualizer):
    display_name = "Toy Rockets"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(3,5,10)))
        env, gate = music_env(bands, rms)
        cx, cy = r.center().x(), r.center().y()
        count = 5
        for i in range(count):
            ang = 2*pi*i/count + t*(0.4+1.2*env)
            R = min(w,h)*0.28*(0.9+0.6*env)
            x = cx + R*cos(ang); y = cy + R*sin(ang)
            hue = int((t*50 + i*40) % 360)
            # body
            p.setPen(QPen(QColor.fromHsv(hue,220,255,200), 2))
            p.setBrush(QBrush(QColor.fromHsv(hue,200,220,120)))
            p.drawEllipse(QPointF(x,y), 12, 24)
            # flame on peaks
            if gate>0.55:
                p.setPen(QPen(QColor(255,200,120,200),3))
                p.drawLine(QPointF(x, y+24), QPointF(x, y+24+20+20*env))
