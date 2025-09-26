
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_prev = []; _env=_gate=_punch=0.0; _pt=None; _sdict={}

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut, n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s+=w*bands[i]; c+=1
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
    if not _prev or len(_prev)!=n: _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target=0.58*e + 1.30*f + 0.18*rms + 0.22*lo
    target=target/(1+0.7*target)
    if target>_env: _env=0.72*_env+0.28*target
    else: _env=0.92*_env+0.08*target
    hi,lo_thr=0.30,0.18
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate=0.82*_gate + 0.18*g
    boom=min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

def spring_to(key, target, t, k=30.0, c=6.0, lo=0.25, hi=4.0):
    s, v, pt = _sdict.get(key, (1.0, 0.0, None))
    if pt is None:
        _sdict[key]=(s,v,t); return 1.0
    dt=max(0.0, min(0.033, t-pt))
    a=-k*(s-target) - c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

def smiley(p:QPainter, cx, cy, s, fill:QColor, stroke:QColor):
    # Draw with NORMAL composition to avoid additive washout
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    # outer shadow glow (plus)
    g = QRadialGradient(QPointF(cx, cy), s*1.6)
    c0 = QColor(fill.red(), fill.green(), fill.blue(), 90)
    g.setColorAt(0.0, c0); g.setColorAt(1.0, QColor(0,0,0,0))
    p.setCompositionMode(QPainter.CompositionMode_Plus)
    p.setBrush(QBrush(g)); p.setPen(QPen(QColor(0,0,0,0),0))
    p.drawEllipse(QPointF(cx,cy), s*1.2, s*1.2)
    # face
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(20,20,30,255), 6))
    p.drawEllipse(QPointF(cx,cy), s, s)
    # eyes + mouth for contrast
    er = max(2.0, s*0.13)
    ex = s*0.35; ey = -s*0.10
    p.setBrush(QBrush(QColor(20,20,30))); p.setPen(QPen(QColor(20,20,30), 2))
    p.drawEllipse(QPointF(cx-ex, cy+ey), er, er)
    p.drawEllipse(QPointF(cx+ex, cy+ey), er, er)
    path = QPainterPath(QPointF(cx - s*0.52, cy + s*0.22))
    path.cubicTo(QPointF(cx - s*0.25, cy + s*0.55),
                 QPointF(cx + s*0.25, cy + s*0.55),
                 QPointF(cx + s*0.52, cy + s*0.22))
    p.setPen(QPen(QColor(20,20,30), max(2,int(s*0.13))))
    p.setBrush(QBrush(QColor(0,0,0,0)))
    p.drawPath(path)

@register_visualizer
class SmileySlamPulse(BaseVisualizer):
    display_name = "Smiley Slam Pulse"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        target = 1.0 + 1.0*env + (2.0 if gate>0.55 else 1.0)*boom + 0.70*punch
        scale = spring_to("smiley_amp", target, t)
        cx,cy=r.center().x(), r.center().y()
        spacing = min(w,h)*0.26
        base_s = min(w,h)*0.10
        positions = (cx-spacing, cx, cx+spacing)
        left_fill = QColor(255,215,80,255)   # strong yellow
        right_fill= QColor(255,120,80,255)   # orange for contrast
        hue = int((t*110 + 220*env) % 360)
        mid_fill = QColor.fromHsv(hue,235,255,255)
        for i,x in enumerate(positions):
            s = base_s * scale
            fill = (mid_fill if i==1 else (left_fill if i==0 else right_fill))
            smiley(p, x, cy, s, fill, QColor(0,0,0,255))
