
# Shared utilities + beat driver (matched to your examples)
import math, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

def HSV(h,s=230,v=255,a=230):
    return QColor.fromHsv(int(h)%360, int(max(0,min(255,s))), int(max(0,min(255,v))), int(max(0,min(255,a))))

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
    onset=max(0.0,_f_short-_f_long); norm=_f_long+1e-4
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

# ===== Solar Winds (Rework: Lemniscate 1-each) =====
SUNS = list("☼☀")
STARS = list("★☆✦✧✩✪❇✨")
ICONS = SUNS + STARS

_S_LAST_T=None
_S_ITEMS=[]

def lemniscate(cx, cy, a, s):
    x = cx + a * math.sin(s)
    y = cy + a * math.sin(s) * math.cos(s) * 0.6
    return x, y

@register_visualizer
class SolarWinds(BaseVisualizer):
    display_name = "Solar Winds — Lemniscate (1-each)"

    def paint(self, p:QPainter, r, bands, rms, t):
        global _S_LAST_T, _S_ITEMS
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        if _S_LAST_T is None: _S_LAST_T=t
        dt=max(0.0, min(0.05, t-_S_LAST_T)); _S_LAST_T=t

        p.fillRect(r, QBrush(QColor(5,6,10)))

        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = drive(bands or [0.0]*n, rms or 0.0, t)

        cx, cy = r.center().x(), r.center().y()
        a = min(w,h)*0.35*(1.0 + 0.10*env)

        if not _S_ITEMS:
            base_phase = 2*math.pi/len(ICONS)
            for i,ch in enumerate(ICONS):
                spd = (0.8 + 0.8*random.random())
                size = int(min(w,h)*(0.08 if ch in SUNS else 0.06))
                _S_ITEMS.append({
                    "ch": ch, "s": i*base_phase, "spd": spd,
                    "size": size, "h": random.uniform(0,360)
                })

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for it in _S_ITEMS:
            it["s"] += (it["spd"] * (1.0 + 0.6*punch + 0.3*env)) * dt * 2.0
            it["h"] += (50 + 40*env) * dt
            x,y = lemniscate(cx, cy, a, it["s"])
            scale = 1.0 + 0.25*punch + (0.15 if it["ch"] in SUNS else 0.0)
            size = int(it["size"] * scale)
            p.setFont(font_px(size))
            p.setPen(QPen(HSV(it["h"], a=230), 2))
            p.drawText(int(x), int(y), it["ch"])
