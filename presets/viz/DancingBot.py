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
class DancingBot(BaseVisualizer):
    display_name = "Dancing Bot"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,9,16)))
        env, gate = music_env(bands, rms)
        tgt = 1.0 + 0.8*env + (1.2 if gate>0.55 else 0.0)
        amp = spring_to("bot_amp", tgt, t, hi=2.6)
        cx, cy = r.center().x(), r.center().y() + 12*sin(t*1.0)
        scale = 1.0 * amp
        body_w = 90*scale; body_h = 130*scale
        p.setBrush(QBrush(QColor(100,180,255)))
        p.setPen(QPen(QColor(255,255,255,160), 2))
        p.drawRoundedRect(QRectF(cx-body_w/2, cy-body_h/2, body_w, body_h), 16, 16)
        head_r = 30*scale*(0.9 + 0.2*env)
        hx = cx + 12*sin(t*2.0*amp)
        hy = cy - body_h/2 - head_r*1.3 + 12*sin(t*5 + env*6)
        p.setBrush(QBrush(QColor(255,220,120)))
        p.setPen(QPen(QColor(255,255,255,200), 2))
        p.drawEllipse(QPointF(hx,hy), head_r, head_r)
        p.setBrush(QBrush(QColor(30,30,30)))
        p.setPen(QPen(QColor(0,0,0,0),0))
        p.drawEllipse(QPointF(hx-8*scale, hy-3*scale), 3*scale,3*scale)
        p.drawEllipse(QPointF(hx+8*scale, hy-3*scale), 3*scale,3*scale)
        arm_len = 70*scale*(1+0.5*env)
        ang = 1.2*sin(t*2.6*amp) + 1.6*env*sin(t*4.4)
        p.setPen(QPen(QColor(255,180,200), 6))
        ax1=cx-body_w/2; ay1=cy-18*scale
        p.drawLine(QPointF(ax1,ay1), QPointF(ax1-arm_len*cos(ang), ay1-arm_len*sin(ang)))
        bx1=cx+body_w/2; by1=cy-18*scale
        p.drawLine(QPointF(bx1,by1), QPointF(bx1+arm_len*cos(ang), by1-arm_len*sin(ang)))
        p.setPen(QPen(QColor(180,255,200), 6))
        spread = 38*scale
        yfoot = cy+body_h/2 + (16*env + 12*(1 if gate>0.55 else 0))*sin(t*6.5*amp)
        p.drawLine(QPointF(cx-spread,yfoot), QPointF(cx-spread-18, yfoot+10))
        p.drawLine(QPointF(cx+spread,yfoot), QPointF(cx+spread+18, yfoot+10))
        if gate>0.6:
            for i in range(16):
                ang2 = 2*pi*_rng.random()
                d = (30+70*_rng.random())*(0.8+0.8*env)
                col = QColor.fromHsv(int(_rng.random()*360), 220, 255, 220)
                p.setPen(QPen(col,2))
                p.drawLine(QPointF(hx,hy), QPointF(hx+d*cos(ang2), hy+d*sin(ang2)))
