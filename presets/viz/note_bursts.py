
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

# --- Visual 2: Note Bursts ---
# Radial bursts of notes from the center on onsets, with a gentle ambient drizzle.

_bursts=[]; _amb=[]; _t_prev=None

@register_visualizer
class NoteBursts(BaseVisualizer):
    display_name = "Note Bursts"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _bursts, _amb, _t_prev
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _t_prev is None: _t_prev=t
        dt = max(0.0, min(0.05, t-_t_prev)); _t_prev=t

        p.fillRect(r, QBrush(QColor(7,7,11)))

        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        cx, cy = r.center().x(), r.center().y()

        # ambient drizzle: always a few notes drifting downward
        amb_spawn = int((1.5 + 3.0*env) * dt * 10)
        for _ in range(amb_spawn):
            _amb.append({
                "x": random.uniform(0, w),
                "y": -20.0,
                "vy": (80 + 220*random.random())*(0.8 + 0.6*env),
                "size": int(min(w,h)*0.03 + 8*random.random()),
                "h": random.uniform(0,360),
                "life": 1.0,
                "ch": random.choice(_NOTES),
            })

        # burst on onset
        if punch > 0.55:
            count = 40
            base_h = random.uniform(0,360)
            for i in range(count):
                ang = (2*math.pi*i/count) + random.uniform(-0.08,0.08)
                spd = (180 + 420*random.random())*(1.0 + 0.8*env + 0.6*punch)
                _bursts.append({
                    "x": cx, "y": cy, "vx": spd*math.cos(ang), "vy": spd*math.sin(ang),
                    "size": int(min(w,h)*0.028 + 10*random.random()),
                    "h": base_h + i*6.0,
                    "life": 1.0,
                    "ch": random.choice(_NOTES),
                })

        # draw ambient
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        alive_amb=[]
        for a in _amb:
            a["y"] += a["vy"]*dt
            a["life"] *= (0.996 - 0.05*dt)
            a["h"] += 30*dt
            if -40<a["y"]<h+40 and a["life"]>0.1:
                alive_amb.append(a)
                p.setFont(_font(a["size"]))
                p.setPen(QPen(_hsv(a["h"], a=int(180*a["life"])+40), 2))
                p.drawText(int(a["x"]), int(a["y"]), a["ch"])
        _amb = alive_amb

        # draw bursts
        alive=[]
        for b in _bursts:
            b["x"] += b["vx"]*dt
            b["y"] += b["vy"]*dt
            b["vy"] += 20.0*dt  # tiny gravity so arcs feel nicer
            b["life"] *= (0.990 - 0.10*dt)
            b["h"] += 90*dt
            if -100<b["x"]<w+100 and -100<b["y"]<h+100 and b["life"]>0.1:
                alive.append(b)
                p.setFont(_font(b["size"]))
                p.setPen(QPen(_hsv(b["h"], a=int(220*b["life"])), 2))
                p.drawText(int(b["x"]), int(b["y"]), b["ch"])
        _bursts = alive
