

from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# ======== Config ========
FREEZE_WHEN_IDLE   = True    # freeze when no music
IDLE_TO_CENTER_SEC = 2.0     # time since last beat to consider idle
IDLE_ENV_THR       = 0.04    # energy floor to consider idle
POP_HALF_LIFE      = 0.08    # seconds — visual pop persistence

# ======== Shared state ========
_prev_spec = []
_last_t = None
_onset_avg = 0.0
_onset_peak = 1e-3
_gate = 0.0
_since_on = 10.0  # seconds since last beat (start "idle")

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

_pop_eye = 0.0
_pop_jaw = 0.0

@register_visualizer
class MechaMaskBeats(BaseVisualizer):
    display_name = "Mecha Mask (Beats)"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _pop_eye, _pop_jaw
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t
        _since_on += dt

        lo,mid,hi,fx,env,norm,beat = _onset_features(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on>IDLE_TO_CENTER_SEC) and (env<IDLE_ENV_THR))

        if beat and not idle:
            if hi + fx > lo*0.8:
                _pop_eye = 1.0
            if lo > 0.08:
                _pop_jaw = 1.0
        _pop_eye = _decay(_pop_eye, dt, POP_HALF_LIFE)
        _pop_jaw = _decay(_pop_jaw, dt, POP_HALF_LIFE)

        # draw
        p.fillRect(r, QBrush(QColor(8,9,12)))
        cx,cy = w/2, h/2
        S = min(w,h)
        # mask
        p.setBrush(QBrush(QColor(22,24,30)))
        p.setPen(QPen(QColor(180,210,230,220), 6))
        p.drawRoundedRect(QRectF(cx-S*0.25, cy-S*0.28, S*0.5, S*0.56), 22, 22)

        # eye slits + shutters (pop controls shutter thickness)
        slit_w = S*0.16; slit_h = S*0.05; offset_y = S*0.06
        for sgn in (-1,1):
            x0 = cx + sgn*S*0.12 - slit_w/2; y0 = cy - offset_y - slit_h/2
            p.setBrush(QBrush(QColor(240,250,255))); p.setPen(QPen(QColor(40,50,80,220), 3))
            p.drawRoundedRect(QRectF(x0, y0, slit_w, slit_h), 6, 6)
            thick = S*0.02*(1.0+1.5*_pop_eye)
            p.setBrush(QBrush(QColor(30,34,40))); p.setPen(Qt.NoPen)
            p.drawRect(QRectF(x0, y0, slit_w, min(slit_h, thick)))
            p.drawRect(QRectF(x0, y0+slit_h-min(slit_h, thick), slit_w, min(slit_h, thick)))

        # segmented jaw (expands on low pop)
        jaw_h = S*0.12*(1.0+0.9*_pop_jaw)
        seg_w = S*0.12
        p.setPen(QPen(QColor(160,200,160,220), 6)); p.setBrush(QBrush(QColor(28,32,36)))
        for i in range(-2,3):
            x = cx + i*seg_w*0.52 - seg_w/2
            p.drawRoundedRect(QRectF(x, cy+S*0.18, seg_w*0.9, max(1.0, jaw_h*0.45)), 6,6)

        # forehead beacon on strong hits
        if _pop_eye>0.05 or _pop_jaw>0.05:
            p.setBrush(QBrush(QColor.fromHsv(int((t*30+200)%360), 200, 255, 200)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy - S*0.26), S*0.02*(1.0+_pop_eye), S*0.02*(1.0+_pop_eye))
