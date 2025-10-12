
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut, n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s += w*bands[i]; c+=1
    return s/max(1,c)

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(0, cut):
        w=1.0 - 0.4*(i/max(1,cut-1))
        s += w*bands[i]; c+=1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n:
        _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.60*e + 1.35*f + 0.18*rms + 0.22*lo
    target = target/(1+0.7*target)
    if target>_env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi,lo_thr = 0.30, 0.18
    g = 1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, _low(bands)*1.3 + 0.45*rms))
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

_sdict = {}
def spring_to(key, target, t, k=32.0, c=5.8, lo=0.25, hi=4.2):
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

def star_path(cx, cy, r_outer, r_inner, points=5, rot=0.0):
    path = QPainterPath()
    for i in range(points*2):
        r = r_outer if i%2==0 else r_inner
        th = rot + i*pi/points
        x = cx + r*cos(th); y = cy + r*sin(th)
        if i==0: path.moveTo(x,y)
        else: path.lineTo(x,y)
    path.closeSubpath()
    return path

@register_visualizer
class NeonStarburstPulse(BaseVisualizer):
    display_name = "Neon Starburst Pulse"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(5,6,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        target = 1.0 + 1.0*env + (2.0 if gate>0.55 else 0.9)*boom + 0.70*punch
        scale = spring_to("star_amp", target, t)
        cx,cy=r.center().x(), r.center().y()
        spacing = min(w,h)*0.25
        base_r = min(w,h)*0.095
        positions = (cx-spacing, cx, cx+spacing)
        left_col = QColor(120,230,255,235); right_col = QColor(255,120,240,235)
        hue = int((t*105 + 220*env) % 360)
        mid_col = QColor.fromHsv(hue,235,255,235)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i,x in enumerate(positions):
            ro = base_r * scale
            ri = ro * 0.45
            fill = (mid_col if i==1 else (left_col if i==0 else right_col))
            stroke = QColor(255,255,255,255)
            g = QRadialGradient(QPointF(x, cy), ro*2.5)
            c0 = QColor(fill.red(), fill.green(), fill.blue(), 160)
            c1 = QColor(fill.red(), fill.green(), fill.blue(), 0)
            g.setColorAt(0.0, c0); g.setColorAt(1.0, c1)
            p.setBrush(QBrush(g)); p.setPen(QPen(QColor(255,255,255,14),1))
            p.drawEllipse(QPointF(x, cy), ro*1.45, ro*1.45)
            p.setBrush(QBrush(fill)); p.setPen(QPen(stroke, 4))
            p.drawPath(star_path(x, cy, ro, ri, points=5, rot=t*(0.8+1.6*env)))
            p.drawPath(star_path(x, cy, ro*1.02, ri*1.02, points=5, rot=t*(0.8+1.6*env)))
