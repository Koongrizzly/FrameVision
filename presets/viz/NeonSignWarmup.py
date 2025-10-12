# NeonSignWarmup.py — adaptive beat-pop (size really changes)
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import Qt, QPointF
from helpers.music import register_visualizer, BaseVisualizer

# --- customize here ---
SIGN_TEXT = "FrameVision"   # change to your app name
BASE_SCALE = 0.12           # base font size as fraction of min(w,h)
POP_STRENGTH = 0.35         # pop amount (bigger = more size change)
GLOW_LEVEL = 1.0            # 1.0 default; increase for brighter glow

# --- state ---
_prev = []
_last_t = None
_warm = 0.0
_gate = 0.0
_pop = 0.0
_onset_avg = 0.0
_onset_peak = 1e-3  # avoid div by 0

def _flux(bands):
    global _prev
    if not bands:
        _prev = []
        return 0.0
    if (not _prev) or (len(_prev) != len(bands)):
        _prev = [0.0]*len(bands)
    f = 0.0
    n = len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev)):
        d = x - px
        if d>0:
            f += d * (0.35 + 0.65*(i/max(1,n-1)))
    _prev = [0.82*px + 0.18*x for x,px in zip(bands,_prev)]
    return f / max(1,n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.34, down=0.14):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

@register_visualizer
class NeonSignWarmup(BaseVisualizer):
    display_name = "Neon Sign Warm-Up (Pop)"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t,_warm,_gate,_pop,_onset_avg,_onset_peak
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t=t
        dt=max(0.0, min(0.05, t-_last_t)); _last_t=t

        # features
        lo,mid,hi = _split(bands)
        fx = _flux(bands)

        # neon "warmth" envelope
        _warm = _env_step(_warm, 0.8*mid + 1.1*fx + 0.2*hi, up=0.7, down=0.25)

        # --- adaptive onset detection (so it works at any loudness) ---
        onset = 0.6*hi + 1.5*fx + 0.2*rms
        _onset_avg = 0.98*_onset_avg + 0.02*onset
        _onset_peak = max(_onset_peak*pow(0.5, dt/0.6), onset)  # peak decays ~0.6s half-life
        denom = max(1e-3, _onset_peak - _onset_avg*0.6)
        norm = max(0.0, (onset - _onset_avg*0.8) / denom)  # 0..~1
        # gate + impulse pop
        rising = (norm > 0.55) and (_gate <= 0.5)
        _gate = 0.78*_gate + 0.22*(1.0 if norm > 0.55 else 0.0)
        _pop = 1.0 if rising else _pop * pow(0.5, dt/0.07)  # ~70ms half-life

        # combine impulse with continuous norm so size ALWAYS reacts
        pop_level = min(1.0, 0.35*norm + _pop)

        # --- draw ---
        p.fillRect(r, QBrush(QColor(4,7,10)))

        # dynamic font size
        base_px = int(min(w,h) * BASE_SCALE)
        size_px = max(8, int(base_px * (1.0 + POP_STRENGTH * pop_level)))
        p.setFont(QFont("Arial", size_px, QFont.Bold))

        # color + glow
        hue = int((t*20 + 300) % 360)
        base = QColor.fromHsv(hue, 200, 255, 255)
        glow = int(60 + 180*min(1.0, _warm*1.2) * GLOW_LEVEL)

        # multi-stroke neon glow; thickness also scales slightly with pop
        for thick in (18, 14, 10, 6, 3):
            tsc = max(1, int(thick * (1.0 + 0.25*pop_level)))
            p.setPen(QPen(QColor(base.red(), base.green(), base.blue(), max(12, glow//tsc)), tsc))
            p.drawText(r, Qt.AlignCenter, SIGN_TEXT)

        # small starter flicks
        if norm > 0.65:
            p.setPen(QPen(QColor(255,255,255,90), 2))
            for i in range(10):
                a = 2*pi*i/10
                R = min(w,h)*0.37
                p.drawLine(QPointF(w/2+R*cos(a), h/2+R*sin(a)), QPointF(w/2+R*cos(a*1.02), h/2+R*sin(a*1.02)))
