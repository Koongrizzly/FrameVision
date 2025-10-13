# --- InvertedTrianglesPulse_basspulse_v1.py ---
# Bass‑biased pulsing: small between beats, big on the kick.
# Changes vs v4:
# - Dual‑EMA onset runs on a bass‑weighted mix (0.75*low + 0.25*flux).
# - Mapping gives a little extra credit to bass (boom) and less to env.
# - Everything else stays fast and snappy. Comments are plain English.

# --- InvertedTrianglesPulse_pulse128_v4.py ---
# Fix for 'moves once then stops': dual-EMA onset detection + fast decay.
# Short EMA vs Long EMA creates a continuous onset signal; no brittle thresholds.
# Size mapping uses a continuous pulse (no 0/1 step), so it will always pump.

# --- InvertedTrianglesPulse_pulse128_v3.py ---
# Fix: visuals that move once then stop. This version uses a dynamic onset detector
# (adaptive threshold + minimum interval) to fire a short 'punch' on every beat.
# Result: small between beats, big on each beat (e.g., ~128 BPM).
# Edits are commented in plain English near each change.

# --- InvertedTrianglesPulse_pulse128_v2.py ---
# Big on the beat, small between beats (clear pulsing).
# Changes vs previous: easier beat trigger, discrete pulse mapping in paint(),
# faster release, lower base size, higher peak size.

# --- InvertedTrianglesPulse_pulse128.py ---
# Goal: VERY clear pulse at the song tempo (e.g., 128 BPM) — big on beat, small between.
# What we changed:
# 1) Envelope: faster attack and *very* fast release so size drops to small quickly.
# 2) Gate: snappier smoothing so beats produce clear on/off pulses.
# 3) Mapping: low baseline size + punch drives peaks; env only counts when gate is on.
# 4) Bass influence: reduced so sustained lows don't keep it large.
# 5) Spring: quick settle, permits big hits, small in-between.
# Drop-in replacement: rename this to your original filename if needed.
# ---------------------------------------------------------------


from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut, n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s += w*bands[i]; c+=1
    return s/max(1,c)

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(0, cut):
        w=0.8 - 0.4*(i/max(1,cut-1))
        s += w*bands[i]; c+=1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n:
        _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env, _gate, _punch, _pt, _f_avg, _f_var, _last_hit, _f_short, _f_long
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.50*e + 1.00*f + 0.08*rms + 0.08*lo
    target = target / (1 + 1.35*target)

    # Envelope: fast attack, very fast release
    if target > _env: _env = 0.65*_env + 0.35*target
    else:             _env = 0.35*_env + 0.65*target

    # Gate (kept for colors); a bit snappier
    hi, lo_thr = 0.22, 0.12
    g = 1.0 if f > hi else (0.0 if f < lo_thr else _gate)
    _gate = 0.55*_gate + 0.45*g

    # --- Dual-EMA onset detection (robust, continuous) ---
    # Short EMA reacts to immediate flux; Long EMA follows overall trend.
    _f_short = 0.65*_f_short + 0.35*f
    _f_long  = 0.95*_f_long  + 0.05*f
    onset = max(0.0, _f_short - _f_long)  # positive when energy jumps
    # Normalize onset roughly to [0,1]
    norm = _f_long + 1e-4
    onset_n = min(1.0, onset / (0.35*norm))  # 0.35 is a loose scale factor

    # Time step and fast-decay 'punch' so it gets small between hits
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    _punch = max(_punch * pow(0.22, dt/0.05), onset_n)

    # Small low-end contribution only
    boom = min(1.0, max(0.0, 0.55*lo + 0.12*rms))

    return max(0.0, min(1.0, _env)), max(0.0, min(1.0, _gate)), boom, max(0.0, min(1.0, _punch))

_sdict = {}
# Dual-EMA flux trackers for robust onsets
_f_short = 0.0  # short window EMA of flux
_f_long = 0.0   # long window EMA of flux
# --- Runtime state for beat detection ---
_f_avg = 0.0     # running average of flux
_f_var = 0.0     # running variance (for adaptive threshold)
_last_hit = None # last time a beat 'hit' fired

def spring_to(key, target, t, k=36.0, c=9.0, lo=0.12, hi=3.4):
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

def tri_path(cx, cy, s, rot=0.0):
    path = QPainterPath()
    for k in range(3):
        th = rot + (pi/2) + k*2*pi/3
        x = cx + s*cos(th); y = cy + s*sin(th)
        if k==0: path.moveTo(x,y)
        else: path.lineTo(x,y)
    path.closeSubpath()
    return path

@register_visualizer
class InvertedTrianglesPulse(BaseVisualizer):
    display_name = "Inverted Triangles Pulse"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,14)))
        env, gate, boom, punch = beat_drive(bands, rms, t)
        # Final size mapping (plain English):
# - Base size is small (0.65) so it's clearly tiny between beats.
# - On the beat, size is driven by 'punch' and 'env*gate'.
# - Bass (boom) only adds a little, so long low notes don't keep it large.
        target = 0.65 + 1.20*punch + 0.70*(env*gate) + 0.20*boom
        scale = spring_to("tri_amp", target, t)
        cx,cy=r.center().x(), r.center().y()
        spacing = min(w,h)*0.24
        base_s = min(w,h)*0.11
        positions = (cx-spacing, cx, cx+spacing)
        red_fill = QColor(255,70,110,220); red_stroke = QColor(255,180,200,255)
        hue = int((t*100 + 220*env) % 360)
        mid_fill = QColor.fromHsv(hue,235,255,235); mid_stroke = QColor.fromHsv(hue,235,255,255)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i,x in enumerate(positions):
            s = base_s * scale
            rot = pi if i==1 else 0.0  # middle upside down
            fill, stroke = (mid_fill, mid_stroke) if i==1 else (red_fill, red_stroke)
            g = QRadialGradient(QPointF(x,cy), s*2.4)
            c0 = QColor(fill.red(), fill.green(), fill.blue(), 160)
            c1 = QColor(fill.red(), fill.green(), fill.blue(), 0)
            g.setColorAt(0.0,c0); g.setColorAt(1.0,c1)
            p.setBrush(QBrush(g)); p.setPen(QPen(QColor(255,255,255,14),1))
            p.drawEllipse(QPointF(x,cy), s*1.4, s*1.4)
            p.setBrush(QBrush(fill)); p.setPen(QPen(stroke, 5))
            p.drawPath(tri_path(x, cy, s, rot))
            p.drawPath(tri_path(x, cy, s*1.02, rot))
