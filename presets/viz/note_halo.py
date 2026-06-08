
# Colorful note visual — uses only: ♫ ♪ ♩ ♬ ♭ ♮ ♯
# Depends on: PySide6 + helpers.music (register_visualizer, BaseVisualizer)
from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# colorful music notes + accidentals
_notes = list("♫♪♩♬♭♮♯")

# envelope + onset + boom driver
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None
_sdict = {}

TUNE_KICK = 1.2
TUNE_BOOM = 1.0
SPR_K = 30.0
SPR_C = 6.0
SPR_MAX = 4.2

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
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

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

def hue_color(base_h, sat=230, val=255, alpha=220):
    h = int(base_h)%360
    return QColor.fromHsv(h, sat, val, alpha)


@register_visualizer
class NoteHalo(BaseVisualizer):
    display_name = "Note Halo"
    def paint(self, p:QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,12)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("note_halo", 1.0+0.8*env + TUNE_KICK*punch + 0.8*boom, t, hi=3.6)
        R = min(w,h)*0.32*amp
        p.setFont(QFont("DejaVu Sans", int(24+12*amp)))
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        n = len(_notes)
        for i,ch in enumerate(_notes):
            th = i*2*pi/n + t*(0.7+1.0*env)
            x = cx + R*cos(th); y = cy + R*sin(th)
            col = hue_color(t*60 + i*30)
            p.setPen(QPen(col, 2))
            p.drawText(QPointF(x,y), ch)
        if gate>0.6:
            p.setPen(QPen(hue_color(t*120, val=255, alpha=255), 4))
            p.drawEllipse(QPointF(cx,cy), R*0.55, R*0.55)
