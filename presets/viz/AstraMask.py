
# --- common beat + spring kit (tweak here) ---
TUNE_KICK = 1.3   # onset kick (bumped from 1.2 to 1.3)
TUNE_BOOM = 1.0   # bass boom weight
SPR_K = 30.0      # spring stiffness
SPR_C = 6.0       # spring damping
SPR_MAX = 4.2     # max scale

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

# Astra Mask
@register_visualizer
class AstraMask(BaseVisualizer):
    display_name = "Astra Mask"
    def paint(self, p:QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()
        target = 1.0 + 0.9*env + TUNE_KICK*punch + 1.0*boom
        amp = spring_to("astra", target, t, hi=4.0)
        face_w = min(w,h)*0.35*amp
        face_h = min(w,h)*0.46*amp
        g = QRadialGradient(QPointF(cx, cy), face_h*1.2)
        hue = int((t*50+220*env)%360)
        g.setColorAt(0.0, QColor.fromHsv(hue,230,255,180))
        g.setColorAt(1.0, QColor(0,0,0,0))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(255,255,255,30),1))
        p.drawEllipse(QPointF(cx,cy), face_w*0.6, face_h*0.6)
        p.setBrush(QBrush(QColor(40,40,60,220))); p.setPen(QPen(QColor(160,180,255,220),3))
        p.drawRoundedRect(QRectF(cx-face_w/2, cy-face_h/2, face_w, face_h), 28, 32)
        eye_w = face_w*0.18; eye_h = max(3.0, face_h*0.04 + 10*env + 14*punch)
        p.setBrush(QBrush(QColor(255,240,200))); p.setPen(QPen(QColor(255,255,255,200),2))
        p.drawRoundedRect(QRectF(cx-face_w*0.22-eye_w/2, cy-face_h*0.10-eye_h/2, eye_w, eye_h), 6, 6)
        p.drawRoundedRect(QRectF(cx+face_w*0.22-eye_w/2, cy-face_h*0.10-eye_h/2, eye_w, eye_h), 6, 6)
        jd = face_h*(0.10 + 0.20*boom)
        p.setBrush(QBrush(QColor(90,90,130,220)))
        p.drawRoundedRect(QRectF(cx-face_w*0.28, cy+face_h*0.08, face_w*0.56, jd), 12, 12)
        p.setPen(QPen(QColor.fromHsv((hue+100)%360,230,255,220),3))
        for k in range(6):
            off = (k-2.5)*face_h*0.06
            p.drawLine(QPointF(cx-face_w*0.35, cy+off), QPointF(cx-face_w*0.20, cy+off+10*sin(t*4+off)))
            p.drawLine(QPointF(cx+face_w*0.35, cy+off), QPointF(cx+face_w*0.20, cy+off+10*sin(t*4+off)))
