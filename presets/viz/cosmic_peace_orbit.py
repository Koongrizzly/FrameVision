
# Cosmic Peace Orbit — orbiting peace/yin‑yang/sun/star glyphs that pulse to the beat.
# Symbols included: ☮ ☯ ☼ ☀ ★ ☆ ✦ ✧ ✩ ✪ ❇ ✨
import math, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

ICONS = list("☮☯☼☀★☆✦✧✩✪❇✨")

def HSV(h,s=230,v=255,a=230):
    return QColor.fromHsv(int(h)%360, int(max(0,min(255,s))), int(max(0,min(255,v))), int(max(0,min(255,a))))

# ===== Beat driver (matches your working examples) =====
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
    f.setPixelSize(int(max(14, px)))
    return f

# ===== Module-level state =====
_LAST_T=None
_ANG=0.0
_RAD=1.0
_H=0.0
_TWINK=[]  # ambient twinkles

@register_visualizer
class CosmicPeaceOrbit(BaseVisualizer):
    display_name = "Cosmic Peace Orbit"

    def paint(self, p:QPainter, r, bands, rms, t):
        global _LAST_T, _ANG, _RAD, _H, _TWINK
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # dt from engine
        if _LAST_T is None: _LAST_T = t
        dt = max(0.0, min(0.05, t-_LAST_T)); _LAST_T = t

        # background
        p.fillRect(r, QBrush(QColor(6,6,10)))

        # audio
        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = drive(bands or [0.0]*n, rms or 0.0, t)

        cx, cy = r.center().x(), r.center().y()
        base_R = min(w,h)*0.32
        target_R = base_R*(1.0 + 0.35*env + 0.45*punch + 0.25*boom)
        _RAD = 0.85*_RAD + 0.15*target_R
        _ANG += dt*(0.6 + 1.2*env + 0.4*punch)
        _H += 60*dt

        # Ambient twinkles (stars drifting slowly)
        # steady low spawn
        tw_rate = 1.0 + 3.0*env
        if random.random() < tw_rate*dt:
            _TWINK.append({
                "x": random.uniform(0,w),
                "y": random.uniform(0,h),
                "life": 1.0,
                "h": random.uniform(0,360),
                "ch": random.choice(list("★☆✦✧✩✪✨❇"))
            })
        alive=[]
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for twn in _TWINK:
            twn["life"] *= (0.994 - 0.04*dt)
            twn["h"] += 30*dt
            if twn["life"]>0.12:
                alive.append(twn)
                px = int(min(w,h)*0.028 + 6*env)
                p.setFont(font_px(px))
                p.setPen(QPen(HSV(twn["h"], a=int(160*twn["life"])+40), 2))
                p.drawText(int(twn["x"]), int(twn["y"]), twn["ch"])
        _TWINK = alive

        # Orbit of main icons
        icons = ICONS
        N = len(icons)
        size = int(min(w,h)*0.08 + 10*env + 6*punch)
        p.setFont(font_px(size))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for i,ch in enumerate(icons):
            ang = _ANG + (2*math.pi*i/N)
            x = cx + _RAD*math.cos(ang)
            y = cy + _RAD*math.sin(ang)
            p.setPen(QPen(HSV(_H + i*20, a=230), 2))
            p.drawText(int(x), int(y), ch)

        # optional inner glow ring on strong beats
        if punch > 0.55:
            p.setPen(QPen(HSV(_H+120, a=200), 3))
            p.drawEllipse(QPointF(cx,cy), _RAD*0.6, _RAD*0.6)
