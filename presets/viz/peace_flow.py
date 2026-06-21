
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

# ===== Peace Flow (Rework: Pendulum) — only ☮ =====
ICON="☮"

# Module state
_P_LAST_T=None
_P_TH=0.0      # pendulum phase
_P_THSPD=0.0   # angular velocity
_P_RAD=1.0
_P_H=0.0
_P_ECHOS=[]    # list of echo dicts

@register_visualizer
class PeaceFlow(BaseVisualizer):
    display_name = "Peace Flow — Pendulum"

    def paint(self, p:QPainter, r, bands, rms, t):
        global _P_LAST_T, _P_TH, _P_THSPD, _P_RAD, _P_H, _P_ECHOS
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        if _P_LAST_T is None: _P_LAST_T=t
        dt=max(0.0, min(0.05, t-_P_LAST_T)); _P_LAST_T=t

        p.fillRect(r, QBrush(QColor(7,7,11)))

        n = max(24, len(bands) if bands else 32)
        env, gate, boom, punch = drive(bands or [0.0]*n, rms or 0.0, t)

        cx, cy = r.center().x(), r.center().y()
        base_R = min(w,h)*0.32
        target_R = base_R*(1.0 + 0.15*env + 0.20*boom)
        _P_RAD = 0.90*_P_RAD + 0.10*target_R
        target_spd = (0.8 + 1.2*env)
        _P_THSPD = 0.92*_P_THSPD + 0.08*target_spd
        _P_TH += _P_THSPD * dt
        _P_H += 50*dt

        x = cx + _P_RAD*math.sin(_P_TH)
        y = cy + 0.25*_P_RAD*math.sin(2*_P_TH)

        if punch>0.55:
            for k in (0.0, 0.10, 0.20):
                _P_ECHOS.append({
                    "x": x, "y": y, "life": 1.0-k, "h": _P_H+120*k,
                    "size": int(min(w,h)*(0.10 + 0.04*env))
                })

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        echoes=[]
        for e in _P_ECHOS:
            e["life"] *= (0.96 - 0.10*dt)
            if e["life"]>0.10:
                echoes.append(e)
                p.setFont(font_px(e["size"]))
                p.setPen(QPen(HSV(e["h"], a=int(120*e["life"])), 2))
                p.drawText(int(e["x"]), int(e["y"]), ICON)
        _P_ECHOS = echoes

        size = int(min(w,h)*(0.12 + 0.06*punch + 0.04*env))
        p.setFont(font_px(size))
        p.setPen(QPen(HSV(_P_H, a=230), 2))
        p.drawText(int(x), int(y), ICON)
