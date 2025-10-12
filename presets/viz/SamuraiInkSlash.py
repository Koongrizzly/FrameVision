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
class SamuraiInkSlash(BaseVisualizer):
    display_name = "Samurai Ink Slash"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(5,6,12)))
        slashes = 8 + int(10*env)
        for s in range(slashes):
            ph = t*(0.9+1.0*env) + s*0.45
            x1 = r.left() - 60 + (w+120)*( (s*0.19 + 0.25*sin(ph)) % 1.0 )
            y1 = r.top()  + h*(0.15 + 0.7*( (s*0.33 + 0.35*cos(ph)) % 1.0 ))
            ang = 0.8*sin(ph) + 1.1*cos(ph*1.3)
            lenr = 160 + 360*env + (60 if gate>0.55 else 0)
            hue = int((t*40 + s*37) % 360)
            core = QColor.fromHsv(hue, 240, 255, 220)
            trail = QColor.fromHsv((hue+40)%360, 230, 240, 150)
            p.setPen(QPen(core, 5))
            p.drawLine(QPointF(x1,y1), QPointF(x1+lenr*cos(ang), y1+lenr*sin(ang)))
            p.setPen(QPen(trail, 2))
            for k in range(1,6):
                a2 = ang + 0.05*k
                p.drawLine(QPointF(x1-12*k*cos(a2), y1-12*k*sin(a2)),
                           QPointF(x1+(lenr-12*k)*cos(a2), y1+(lenr-12*k)*sin(a2)))
        if gate > 0.5:
            for i in range(22):
                ang = 2*pi*_rng.random()
                d = (40+140*_rng.random())*(0.6+0.8*env)
                col = QColor.fromHsv(int(_rng.random()*360), 230, 255, 220)
                p.setPen(QPen(col, 2))
                x = r.center().x(); y = r.center().y()
                p.drawLine(QPointF(x,y), QPointF(x+d*cos(ang), y+d*sin(ang)))
