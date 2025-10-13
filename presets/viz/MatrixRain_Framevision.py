
from math import sin, cos, pi
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Direction + previous-beat state for the logo kick
_logo_dir = 1   # +1 right, -1 left
_p_prev_logo = 0.0
KICK_SHIFT = 0.60  # how far the logo shifts horizontally per beat (in units of base_px)

# --- What this visual does (plain English) ---
# - Draws colorful 'Matrix' code that rains down the screen in columns.
# - Shows a 'Framevision' logo that gets bigger on the beat and small between beats.
# - Font size for the logo is based on your screen: size = 0.12 * min(width, height).
# - The beat detector uses lows (kick) + flux, so the logo pulses to the kick.
# ------------------------------------------------

# --- Small audio feature helpers (same idea as your other file) ---
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None

# Dual-EMA trackers for robust onsets (short vs long average)
_f_short = 0.0
_f_long  = 0.0

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
        _prev = list(bands)
        return 0.0
    s=0.0
    for a,b in zip(_prev,bands):
        d=b-a
        if d>0: s+=d
    _prev = list(bands)
    return s/(len(bands)+1e-6)

# Simple spring to smooth the logo scaling
_sdict = {}
def spring_to(key, target, t, k=34.0, c=9.0, lo=0.10, hi=4.0):
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

# Beat drive: fast-release envelope + bass-biased onset that turns into a snappy 'punch'
def beat_drive(bands, rms, t):
    global _env, _gate, _punch, _pt, _f_short, _f_long
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)

    # Energy target (kept gentle to avoid sticking at max)
    target = 0.50*e + 1.00*f + 0.08*rms + 0.10*lo
    target = target/(1 + 1.20*target)  # soft ceiling

    # Envelope: quick up, very quick down
    if target>_env: _env = 0.70*_env + 0.30*target
    else:           _env = 0.40*_env + 0.60*target

    # Gate (for color variation etc.)
    hi, lo_thr = 0.24, 0.14
    g = 1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate = 0.65*_gate + 0.35*g

    # Dual-EMA onset focusing on lows (kick leads), but with some flux
    mix = 0.70*lo + 0.30*f
    _f_short = 0.65*_f_short + 0.35*mix
    _f_long  = 0.95*_f_long  + 0.05*mix
    onset = max(0.0, _f_short - _f_long)
    norm = _f_long + 1e-4
    onset_n = min(1.0, onset / (0.30*norm))  # 0..1

    # Time + quick punch decay (shrinks between beats)
    global _pt
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t-_pt)); _pt = t
    _punch = max(_punch * pow(0.22, dt/0.05), onset_n)

    boom = min(1.0, max(0.0, 0.60*lo + 0.12*rms))
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

# --- Matrix rain state ---
# We keep per-column position/speed and color. Rebuilt if size changes.
_mrain = {
    "w": 0, "h": 0, "cols": [], "char_h": 0.0, "last_t": None
}

def _ensure_rain(w, h, t):
    global _mrain
    # If size changed or not initialized, (re)build columns
    if _mrain["w"]!=w or _mrain["h"]!=h or not _mrain["cols"]:
        cols = []
        # Character height relative to screen (tweakable)
        char_h = max(10.0, min(w,h)*0.035)
        # Column width ~ 0.6 of char height (monospace look)
        col_w = int(char_h*0.6)
        ncols = max(8, int(w/col_w))
        for i in range(ncols):
            speed = random.uniform(60.0, 220.0)  # pixels per second
            y = random.uniform(-h, h)            # start spread
            hue = random.randint(0,359)          # colorful columns
            cols.append({"x": int(i*col_w + col_w*0.5), "y": y, "speed": speed, "hue": hue})
        _mrain = {"w": w, "h": h, "cols": cols, "char_h": char_h, "last_t": t}
    elif _mrain["last_t"] is None:
        _mrain["last_t"] = t

def _update_rain(t):
    dt = 0.016
    if _mrain["last_t"] is not None:
        dt = max(0.0, min(0.05, t - _mrain["last_t"]))
    _mrain["last_t"] = t
    for c in _mrain["cols"]:
        c["y"] += c["speed"]*dt
        tail = 14 * _mrain["char_h"]
        if c["y"] - tail > _mrain["h"]:
            c["y"] = -random.uniform(0.0, _mrain["h"]*0.5)
            c["speed"] = random.uniform(60.0, 220.0)
            c["hue"] = (c["hue"] + random.randint(20,140)) % 360

# Character set (mix of digits and some katakana-looking chars)
_CHARS = "01アカサタナハマヤラワ0123456789カタハマラﾊﾗｱﾀﾅ"

@register_visualizer
class MatrixRainFramevision(BaseVisualizer):
    display_name = "Matrix Rain + Framevision"

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Background: very dark to make neon colors pop
        p.fillRect(r, QBrush(QColor(8,8,14)))

        # Beat analysis
        env, gate, boom, punch = beat_drive(bands, rms, t)

        # Prepare matrix rain
        _ensure_rain(w, h, t)
        _update_rain(t)

        # Choose a readable monospace-like font for the rain
        char_h = _mrain["char_h"]
        rain_font = QFont("Monospace", pointSize=1)
        rain_font.setPixelSize(int(char_h))
        p.setFont(rain_font)

        # Draw columns
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for c in _mrain["cols"]:
            x = c["x"]; y_head = c["y"]; hue = c["hue"]
            # Make the head brighter on the beat
            brightness_boost = int(60 + 140*punch)  # 60..200
            for k in range(0, 16):  # short tail
                yk = y_head - k*char_h
                if yk < -char_h: break
                if yk > h + char_h: continue
                # random glyph per frame (gives shimmering rainfall)
                ch = random.choice(_CHARS)
                sat = 220
                val = max(60, brightness_boost - k*10)  # fade along tail
                color = QColor.fromHsv((hue + k*3) % 360, sat, min(255, val), max(60, 230-k*10))
                p.setPen(QPen(color))
                p.drawText(int(x), int(yk), ch)

        # Framevision logo that kicks to the beat
        # Base font size = 0.12 * min(w,h) as requested
        base_px = int(min(w,h) * 0.12)
        logo_font = QFont("Montserrat", pointSize=1)
        logo_font.setPixelSize(int(base_px))

        # Scale the logo with a spring so it feels punchy but not jittery
        # target scale: 0.90 small → 1.60 big
        target = 0.90 + 0.70*punch
        scale = spring_to("framevision_logo", target, t)

        p.setFont(logo_font)
        # Center logo
        text = "Framevision"
        # Measure approximate width by drawing to a rect centered; we place it near the bottom third
        cx, cy = r.center().x(), int(h*0.70)
        # Glow color changes slightly with beat
        glow_hue = int((120 + 200*punch) % 360)
        glow_col = QColor.fromHsv(glow_hue, 180, 255, 200)
        text_col = QColor(240, 255, 250, 255)

        # Draw a soft glow behind the text
        glow_radius = int(base_px * 3.0 * scale)
        g = QRadialGradient(QPointF(cx, cy), glow_radius)
        g.setColorAt(0.0, QColor(glow_col.red(), glow_col.green(), glow_col.blue(), 120))
        g.setColorAt(1.0, QColor(glow_col.red(), glow_col.green(), glow_col.blue(), 0))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(0,0,0,0),1))
        p.drawEllipse(QPointF(cx,cy), glow_radius*0.65, glow_radius*0.40)

        # Draw the logo text with stroke + fill so it pops
        p.setPen(QPen(QColor(255,255,255,230), max(2, int(base_px*0.07*scale))))
        # Decide left/right randomly on NEW beats (when punch crosses a threshold)
        global _p_prev_logo, _logo_dir
        if punch > 0.60 and _p_prev_logo <= 0.60:
            _logo_dir = random.choice([-1, 1])
        _p_prev_logo = punch
        # Base font size = 0.12 * min(w,h). We SCALE the font by 'scale' so it actually grows on the beat.
        logo_font.setPixelSize(int(base_px * scale))
        p.setFont(logo_font)
        # Center the text using font metrics, then apply a small beat-driven horizontal shift left OR right.
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        tx = int(cx - tw/2) + int(_logo_dir * KICK_SHIFT * base_px * punch)
        ty = cy + int(base_px * 0.35 * scale)
        # Move the glow with the text so it stays behind the logo.
        glow_radius = int(base_px * 3.0 * scale)
        g = QRadialGradient(QPointF(tx + tw/2, cy), glow_radius)
        g.setColorAt(0.0, QColor(glow_col.red(), glow_col.green(), glow_col.blue(), 120))
        g.setColorAt(1.0, QColor(glow_col.red(), glow_col.green(), glow_col.blue(), 0))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(0,0,0,0),1))
        p.drawEllipse(QPointF(tx + tw/2, cy), glow_radius*0.65, glow_radius*0.40)
        # Draw the logo text with stroke + fill (centered)
        p.setPen(QPen(QColor(255,255,255,230), max(2, int(base_px*0.07*scale))))
        p.drawText(tx, ty, text)
        p.setPen(QPen(text_col, 1))
        p.drawText(tx, ty, text)
