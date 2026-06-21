
import random, math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Symbols (notes + accidentals)
_NOTES = list("♫♪♩♬♭♮♯")

# ===== Audio helpers (style-matched to your examples) =====
_prev=[]; _env=_gate=_punch=0.0; _pt=None
_f_short=_f_long=0.0

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i,x in enumerate(bands):
        if i>=cut: s+=x; c+=1.0
    return (s/c) if c>0.0 else 0.0

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//8)
    s=c=0.0
    for i,x in enumerate(bands):
        if i<cut: s+=x; c+=1.0
    return (s/c) if c>0.0 else 0.0

def _flux(bands):
    global _prev
    if not _prev or len(_prev)!=len(bands):
        _prev=list(bands); return 0.0
    s=0.0
    for a,b in zip(_prev,bands):
        d=b-a
        if d>0: s+=d
    _prev=list(bands)
    return s/(len(bands)+1e-6)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_f_short,_f_long
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target = 0.50*e + 1.00*f + 0.10*rms + 0.15*lo
    target = target/(1+1.20*target)
    if target>_env: _env=0.70*_env+0.30*target
    else:           _env=0.45*_env+0.55*target
    hi,lo_thr=0.24,0.14
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate=0.70*_gate+0.30*g
    mix=0.70*lo+0.30*f
    _f_short=0.65*_f_short+0.35*mix
    _f_long =0.95*_f_long +0.05*mix
    onset=max(0.0,_f_short-_f_long)
    norm=_f_long+1e-4
    onset_n=min(1.0, onset/(0.30*norm))
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    _punch=max(_punch*pow(0.30, dt/0.05), onset_n)
    boom=min(1.0, max(0.0, 0.60*lo + 0.20*rms))
    return max(0, min(1,_env)), max(0, min(1,_gate)), boom, max(0, min(1,_punch))

def _font(px):
    f = QFont("DejaVu Sans", pointSize=1)
    f.setPixelSize(int(max(10, px)))
    return f

def _hsv(h,s=230,v=255,a=230):
    return QColor.fromHsv(int(h)%360, int(max(0,min(255,s))), int(max(0,min(255,v))), int(max(0,min(255,a))))

# --- Visual 1: Note Ribbon ---
# Notes flow left->right along multiple sine lanes. Bursts on onsets.
# SourceOver composition for crisp text.

_particles=[]; _t_prev=None; _last_spawn=0.0

@register_visualizer
class NoteRibbon(BaseVisualizer):
    display_name = "Note Ribbon"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _particles, _t_prev, _last_spawn
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _t_prev is None: _t_prev=t
        dt = max(0.0, min(0.05, t-_t_prev)); _t_prev=t

        p.fillRect(r, QBrush(QColor(8,8,12)))

        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        lanes = 5
        lane_y = [h*0.25 + i*(h*0.5/(lanes-1.0)) for i in range(lanes)]
        amp = h*(0.03 + 0.10*env)

        # base rate + onset bursts + keepalive
        base_rate = 3.0 + 6.0*env + 3.0*boom
        _last_spawn += base_rate*dt
        spawn = int(_last_spawn)
        if spawn>10: spawn=10
        _last_spawn -= spawn
        if punch > 0.55:
            spawn += 6

        for _ in range(spawn):
            lane = random.randint(0, lanes-1)
            spd = (0.22 + 0.55*random.random())*(0.9+0.7*env)*w
            size = int(min(w,h)*0.035 + 10*random.random())
            _particles.append({
                "x": -80.0,
                "y0": lane_y[lane] + random.uniform(-h*0.02, h*0.02),
                "phase": random.uniform(0, 2*math.pi),
                "speed": spd,
                "size": size,
                "life": 1.0,
                "h": random.uniform(0,360),
                "ch": random.choice(_NOTES),
            })

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        alive=[]
        for it in _particles:
            it["x"] += it["speed"]*dt
            it["life"] *= (0.994 - 0.06*dt)
            it["h"] += 40*dt
            y = it["y0"] + amp * math.sin((it["x"]/w)*2*math.pi + it["phase"])
            if -100 < it["x"] < w+100 and it["life"]>0.12:
                alive.append(it)
                p.setFont(_font(it["size"]))
                col = _hsv(it["h"], a=int(210*it["life"])+40)
                p.setPen(QPen(col, 2))
                p.drawText(int(it["x"]), int(y), it["ch"])
        _particles = alive
