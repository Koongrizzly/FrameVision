
# Note Lines — Bold: fewer but MUCH larger notes (+50%) moving left→right on staff lines.
# Symbols: ♫ ♪ ♩ ♬ ♭ ♮ ♯
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

NOTES = list("♫♪♩♬♭♮♯")

def HSV(h,s=230,v=255,a=220):
    return QColor.fromHsv(int(h)%360, int(max(0,min(255,s))), int(max(0,min(255,v))), int(max(0,min(255,a))))

# ===== Beat driver (as per your working examples) =====
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
def drive(bands, rms, t):
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
    dt=max(0.0, min(0.05, t-_pt)); _pt=t
    _punch=max(_punch*pow(0.30, dt/0.05), onset_n)
    boom=min(1.0, max(0.0, 0.60*lo + 0.20*rms))
    return max(0,min(1,_env)), max(0,min(1,_gate)), boom, max(0,min(1,_punch))

def font_px(px):
    f = QFont("DejaVu Sans", pointSize=1)
    f.setPixelSize(int(max(12, px)))
    return f

# ===== Module-level state =====
_ACTORS=[]; _EMIT=0.0; _LAST_T=None; _LAST_PUNCH=0.0; _LINES=[]

@register_visualizer
class NoteLinesBold(BaseVisualizer):
    display_name = "Note Lines — Bold"

    def paint(self, p:QPainter, r, bands, rms, t):
        global _ACTORS, _EMIT, _LAST_T, _LAST_PUNCH, _LINES
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        if _LAST_T is None: _LAST_T = t
        dt = max(0.0, min(0.05, t-_LAST_T)); _LAST_T = t

        p.fillRect(r, QBrush(QColor(7,7,11)))

        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = drive(bands or [0.0]*n, rms or 0.0, t)

        if not _LINES or len(_LINES)!=5:
            top = h*0.28; bottom = h*0.72
            spacing = (bottom-top)/4.0
            _LINES = [top + i*spacing for i in range(5)]

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for i,y in enumerate(_LINES):
            p.setPen(QPen(HSV(200+i*10, s=120, v=130, a=160), 3))
            p.drawLine(0, int(y), w, int(y))

        # Fewer spawns; sizes increased by +50%
        base_rate = 0.9 + 2.2*env + 0.9*boom
        _EMIT += base_rate * dt
        spawn = int(_EMIT)
        if spawn>6: spawn=6
        _EMIT -= spawn
        if punch>0.55 and _LAST_PUNCH<=0.55:
            spawn += 3

        for _ in range(spawn):
            y = random.choice(_LINES)
            speed = (0.18 + 0.45*random.random())*(0.75+0.7*env)*w
            base_size = (min(w,h)*0.065 + 14*random.random())
            size = int(1.5 * base_size)  # +50% size
            _ACTORS.append({
                "x": -80.0, "y": y,
                "speed": speed,
                "size": size, "life": 1.0,
                "h": random.uniform(0,360),
                "ch": random.choice(NOTES),
            })

        alive=[]
        for it in _ACTORS:
            it["x"] += it["speed"]*dt
            it["life"] *= (0.994 - 0.06*dt)
            it["h"] += 40*dt
            if -120 < it["x"] < w+120 and it["life"]>0.12:
                alive.append(it)
                p.setFont(font_px(it["size"]))
                p.setPen(QPen(HSV(it["h"], a=int(220*it["life"])), 2))
                p.drawText(int(it["x"]), int(it["y"]+3), it["ch"])
        _ACTORS = alive
        _LAST_PUNCH = punch
