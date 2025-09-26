from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(777)
_prev = []
_env = 0.0
_gate = 0.0
def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s=c=0.0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += w*bands[i]; c += 1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6); f=c=0.0
    for i in range(cut, n):
        d = bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def music_env(bands, rms):
    global _env,_gate
    e=_midhi(bands); f=_flux(bands)
    target = 0.55*e + 1.30*f + 0.20*rms
    target = target/(1+0.7*target)
    if target>_env: _env = 0.70*_env + 0.30*target
    else: _env = 0.92*_env + 0.08*target
    hi,lo = 0.30, 0.18
    g = 1.0 if f>hi else (0.0 if f<lo else _gate)
    _gate = 0.82*_gate + 0.18*g
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate))

_sdict = {}
def spring_to(key, target, t, k=28.0, c=6.5, lo=0.3, hi=3.0):
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

@register_visualizer
class StickGroove(BaseVisualizer):
    display_name = "Stick Groove"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,9,16)))
        env, gate = music_env(bands, rms)
        amp = spring_to("stick_amp", 1.0 + 0.8*env + (1.2 if gate>0.55 else 0.0), t, hi=2.6)
        cx, cy = r.center().x(), r.center().y() + 12*sin(t*1.0*amp)
        scale = 1.3 * amp
        p.setPen(QPen(QColor(220,240,255,220), 6))
        p.drawLine(QPointF(cx, cy-50*scale), QPointF(cx, cy+50*scale))
        hr = 20*scale*(1+0.2*env)
        p.setBrush(QBrush(QColor(255,220,150)))
        p.setPen(QPen(QColor(255,255,255,200), 2))
        p.drawEllipse(QPointF(cx, cy-70*scale), hr, hr)
        p.setPen(QPen(QColor(255,180,200), 7))
        ang = 1.4*sin(t*2.8*amp) + 1.6*env*sin(t*4.6)
        L = 60*scale
        p.drawLine(QPointF(cx,cy-24*scale), QPointF(cx-L*cos(ang), cy-24*scale - L*sin(ang)))
        p.drawLine(QPointF(cx,cy-24*scale), QPointF(cx+L*cos(ang), cy-24*scale - L*sin(ang)))
        p.setPen(QPen(QColor(180,255,200), 7))
        spread = 34*scale
        yfoot = cy+50*scale + (18*env + 14*(1 if gate>0.55 else 0))*sin(t*7.0*amp)
        p.drawLine(QPointF(cx-spread, cy+50*scale), QPointF(cx-spread, yfoot))
        p.drawLine(QPointF(cx+spread, cy+50*scale), QPointF(cx+spread, yfoot))
