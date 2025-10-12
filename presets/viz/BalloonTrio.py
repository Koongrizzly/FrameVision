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
class BalloonTrio(BaseVisualizer):
    display_name = "Balloon Trio"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(8,8,16)))
        env, gate = music_env(bands, rms)
        tgt = 1.0 + 0.8*env + (1.2 if gate>0.55 else 0.0)
        amp = spring_to("ball_amp", tgt, t, hi=2.6)
        cx, cy = r.center().x(), r.center().y()
        offs = [-140, 0, 140]
        for i,dx in enumerate(offs):
            x = cx + dx
            y = cy - 60*sin(t*(0.9+0.4*i)*amp) - 40*env
            rad = 30*amp
            hue = int((t*45 + i*100) % 360)
            g = QRadialGradient(QPointF(x, y), rad*1.8)
            g.setColorAt(0.0, QColor.fromHsv(hue,220,255,230))
            g.setColorAt(1.0, QColor.fromHsv(hue,220,0,0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,35), 2))
            p.drawEllipse(QPointF(x,y), rad, rad*1.25)
            p.setPen(QPen(QColor.fromHsv(hue,220,255,220),2))
            p.drawLine(QPointF(x, y+rad*1.2), QPointF(x, y+rad*1.2 + 70 + 30*sin(t+i)))
