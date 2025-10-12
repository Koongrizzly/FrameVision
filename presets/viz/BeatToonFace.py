

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

_pop = 0.0
_cheek_idx = 0

@register_visualizer
class BeatToonFace(BaseVisualizer):
    display_name = "Beat Toon Face"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _pop, _cheek_idx
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t
        _since_on += dt

        lo,mid,hi,fx,env,norm,beat = _onset_features(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on>IDLE_TO_CENTER_SEC) and (env<IDLE_ENV_THR))

        # only animate on beats
        if beat and not idle:
            _pop = 1.0
            _cheek_idx = (_cheek_idx + 1) % 3
        _pop = _decay(_pop, dt, POP_HALF_LIFE)

        # --- draw ---
        p.fillRect(r, QBrush(QColor(10,12,16)))
        cx, cy = w/2, h/2
        head_r = min(w,h)*0.32
        p.setPen(QPen(QColor(180,200,220,220), 6))
        p.setBrush(QBrush(QColor(24,28,34)))
        p.drawEllipse(QPointF(cx, cy), head_r, head_r)

        # eyes (pop opens lids; static when idle)
        eye_r = head_r*0.28
        ex = head_r*0.48
        open_amt = head_r*0.18*(0.2 + 0.8*_pop)
        for sgn in (-1, 1):
            p.setPen(QPen(QColor(220,230,240,220), 5))
            p.setBrush(QBrush(QColor(220,230,240)))
            p.drawEllipse(QPointF(cx + sgn*ex, cy - head_r*0.18), eye_r*0.55, eye_r*0.55)
            # eyelid rectangle
            p.setBrush(QBrush(QColor(24,28,34)))
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(cx + sgn*ex - eye_r*0.6, cy - head_r*0.18 - eye_r*0.6,
                              eye_r*1.2, max(0.0, eye_r*1.2 - open_amt)))
            # pupil pulse
            p.setBrush(QBrush(QColor.fromHsv(int((200+60*sgn)%360), 160, 255, int(200*_pop))))
            p.setPen(QPen(QColor(0,0,0,180), 2))
            p.drawEllipse(QPointF(cx + sgn*ex, cy - head_r*0.18), eye_r*0.18*(1.0+0.6*_pop), eye_r*0.18*(1.0+0.6*_pop))

        # cheeks RGB cycle on beats
        cheek_colors = [(255,60,60),(60,255,60),(80,150,255)]
        cr,cg,cb = cheek_colors[_cheek_idx]
        cheek_r = head_r*0.14*(1.0 + 0.6*_pop)
        for sgn in (-1,1):
            p.setPen(QPen(QColor(cr//2, cg//2, cb//2, 200), 3))
            p.setBrush(QBrush(QColor(cr, cg, cb, int(120+120*_pop))))
            p.drawEllipse(QPointF(cx + sgn*head_r*0.58, cy + head_r*0.05), cheek_r, cheek_r)

        # mouth: line -> big "O" only on beat
        if _pop>0.03:
            p.setPen(QPen(QColor(250,120,160,230), int(6+6*_pop)))
            p.setBrush(QBrush(QColor(30,20,28)))
            p.drawEllipse(QPointF(cx, cy + head_r*0.32), head_r*0.18*(1.0+0.7*_pop), head_r*0.16*(1.0+0.7*_pop))
        else:
            p.setPen(QPen(QColor(200,240,200,220), 5))
            p.drawLine(QPointF(cx - head_r*0.18, cy + head_r*0.35), QPointF(cx + head_r*0.18, cy + head_r*0.35))
