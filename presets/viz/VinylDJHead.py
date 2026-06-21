

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

_spin = 0.0
@register_visualizer
class VinylDJHead(BaseVisualizer):
    display_name = "Vinyl DJ Head"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _spin
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t
        _since_on += dt

        lo,mid,hi,fx,env,norm,beat = _onset_features(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on>IDLE_TO_CENTER_SEC) and (env<IDLE_ENV_THR))
        if beat and not idle:
            _spin += 2.2  # add angular velocity only on beats
        _spin *= pow(0.5, dt/0.35)  # friction

        # dark club bg
        p.fillRect(r, QBrush(QColor(10,12,18)))

        cx,cy = w/2, h/2; S=min(w,h)
        # head silhouette
        p.setPen(QPen(QColor(160,180,200,200), 5)); p.setBrush(QBrush(QColor(20,24,30)))
        p.drawEllipse(QPointF(cx, cy), S*0.30, S*0.34)

        # eyes: small arcs flash with spin
        p.setPen(QPen(QColor(220,240,255,int(120+100*min(1.0,_spin))), 4))
        p.drawArc(int(cx-S*0.18), int(cy-S*0.06), int(S*0.12), int(S*0.08), 30*16, 120*16)
        p.drawArc(int(cx+S*0.06), int(cy-S*0.06), int(S*0.12), int(S*0.08), 30*16, 120*16)

        # mouth: vinyl record that rotates
        p.setBrush(QBrush(QColor(30,32,40)))
        p.setPen(QPen(QColor(240,240,245,200), 3))
        p.drawEllipse(QPointF(cx, cy+S*0.18), S*0.12, S*0.12)
        # grooves
        for k in range(3):
            p.setPen(QPen(QColor(90,100,120,180), 2))
            p.drawArc(int(cx-S*0.12), int(cy+S*0.06), int(S*0.24), int(S*0.24), int((_spin*120+k*60)%360)*16, 180*16)
        # label
        p.setBrush(QBrush(QColor(220,80,120)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy+S*0.18), S*0.035, S*0.035)
