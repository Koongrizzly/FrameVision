

from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# ======== Config ========
FREEZE_WHEN_IDLE   = True    # freeze when no music
IDLE_TO_CENTER_SEC = 3.0     # seconds since last beat -> idle
IDLE_ENV_THR       = 0.02    # energy floor to consider idle
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
class LanternGhostFace(BaseVisualizer):
    display_name = "Lantern Ghost Face"
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

        # dark teal bg
        p.fillRect(r, QBrush(QColor(8,16,18)))

        cx,cy = w/2, h/2; S=min(w,h)
        # lantern skull
        p.setPen(QPen(QColor(110,160,170,220), 6))
        p.setBrush(QBrush(QColor(22,28,30)))
        p.drawEllipse(QPointF(cx, cy), S*0.28, S*0.34)

        # eyes glow on beats
        glow = int(80 + 140*_pop)
        for sgn in (-1,1):
            p.setBrush(QBrush(QColor(160, 240, 240, glow)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx + sgn*S*0.1, cy - S*0.05), S*0.06*(1.0+0.5*_pop), S*0.06*(1.0+0.5*_pop))

        # mouth gap opens on beat
        p.setPen(QPen(QColor(180,230,240,220), 4))
        p.setBrush(QBrush(QColor(12,16,18)))
        p.drawRoundedRect(QRectF(cx-S*0.10, cy+S*0.12, S*0.20, S*(0.04+0.08*_pop)), 6,6)

        # hanging tassel flicker
        if _pop>0.05:
            p.setPen(QPen(QColor(140, 220, 220, 140), 3))
            p.drawLine(QPointF(cx, cy+S*0.34), QPointF(cx, cy+S*(0.34+0.05*_pop)))
