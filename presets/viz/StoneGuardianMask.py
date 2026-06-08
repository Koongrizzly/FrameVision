

from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# ======== Config ========
FREEZE_WHEN_IDLE   = True    # freeze when no music
IDLE_TO_CENTER_SEC = 2.0     # seconds since last beat -> idle
IDLE_ENV_THR       = 0.04    # energy floor to consider idle
POP_HALF_LIFE      = 0.09    # seconds — visual pop persistence

# ======== Shared state ========
_prev_spec = []
_last_t = None
_onset_avg = 0.0
_onset_peak = 1e-3
_gate = 0.0
_since_on = 10.0  # seconds since last beat (start idle)

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if (not _prev_spec) or (len(_prev_spec) != len(bands)):
        _prev_spec = [0.0]*len(bands)
    f = 0.0; n = len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d>0:
            f += d * (0.35 + 0.65*(i/max(1,n-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f/max(1,n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _onset_features(bands, rms, dt):
    global _onset_avg, _onset_peak, _gate, _since_on
    lo,mid,hi = _split(bands)
    fx = _flux(bands)
    env = 0.5*rms + 0.35*lo + 0.15*mid
    onset = 0.6*hi + 1.5*fx + 0.2*rms
    # adaptive normalization
    _onset_avg = 0.98*_onset_avg + 0.02*onset
    _onset_peak = max(_onset_peak*pow(0.5, dt/0.6), onset)
    denom = max(1e-3, _onset_peak - 0.6*_onset_avg)
    norm = max(0.0, (onset - 0.8*_onset_avg) / denom)  # ~0..1+
    beat = (norm > 0.55) and (_gate <= 0.5)
    _gate = 0.78*_gate + 0.22*(1.0 if norm>0.55 else 0.0)
    if beat:
        _since_on = 0.0
    return lo, mid, hi, fx, env, norm, beat

def _decay(x, dt, half_life):
    return x * pow(0.5, dt/max(1e-6, half_life))

_pop = 0.0
@register_visualizer
class StoneGuardianMask(BaseVisualizer):
    display_name = "Stone Guardian Mask"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _pop
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t
        _since_on += dt

        lo,mid,hi,fx,env,norm,beat = _onset_features(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on>IDLE_TO_CENTER_SEC) and (env<IDLE_ENV_THR))
        if beat and not idle: _pop = 1.0
        _pop = _decay(_pop, dt, POP_HALF_LIFE)

        # basalt bg
        p.fillRect(r, QBrush(QColor(14,14,16)))
        cx,cy = w/2, h/2; S=min(w,h)

        # stone face
        p.setPen(QPen(QColor(120,130,140,220), 6)); p.setBrush(QBrush(QColor(28,28,34)))
        p.drawRoundedRect(QRectF(cx-S*0.26, cy-S*0.30, S*0.52, S*0.60), 18,18)

        # cracks that glow on beat
        glow = int(60 + 160*_pop)
        p.setPen(QPen(QColor(220,180,120,glow), int(2+4*_pop)))
        for dx in (-S*0.18, 0, S*0.18):
            p.drawLine(QPointF(cx+dx, cy-S*0.22), QPointF(cx+dx+S*0.06, cy-S*0.10))
            p.drawLine(QPointF(cx+dx+S*0.06, cy-S*0.10), QPointF(cx+dx-S*0.02, cy))

        # eye hollows + inner glow
        for sgn in (-1,1):
            p.setBrush(QBrush(QColor(18,18,20)))
            p.setPen(QPen(QColor(60,60,70,200), 3))
            p.drawEllipse(QPointF(cx+sgn*S*0.12, cy-S*0.05), S*0.06, S*0.04)
            p.setBrush(QBrush(QColor(230,210,160,glow)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx+sgn*S*0.12, cy-S*0.05), S*0.03*(1.0+0.5*_pop), S*0.02*(1.0+0.5*_pop))

        # jaw slab drops slightly on beat
        p.setPen(QPen(QColor(150,160,170,220), 5)); p.setBrush(QBrush(QColor(30,30,36)))
        p.drawRoundedRect(QRectF(cx-S*0.16, cy+S*(0.14+0.05*_pop), S*0.32, S*0.10), 8,8)
