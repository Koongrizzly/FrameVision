
from math import pi
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# --- Colored-by-frequency Matrix Rain ---
# - Different frequency bands have different, FIXED colors (bass = blue).
# - Columns span full screen width (one column per band).
# - New streams spawn at top when a band "hits"; streams fall with fast dt and fade near bottom.
# - Hard cap on active streams to keep performance stable.

# ====== Config ======
MAX_STREAMS   = 100    # maximum on-screen streams
KEEPALIVE_SEC = 0.80   # ensure each band spawns occasionally even if quiet

# ====== Audio helpers ======
_prev_bands = []
_env = 0.0
_punch = 0.0
_pt = None

# Per-band short/long EMA for onsets
_emaS = []
_emaL = []
_last_spawn = []

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//8)
    return sum(bands[:cut]) / float(cut)

def _flux_global(bands):
    global _prev_bands
    if not _prev_bands or len(_prev_bands)!=len(bands):
        _prev_bands = list(bands)
        return 0.0
    s=0.0
    for a,b in zip(_prev_bands, bands):
        d = b - a
        if d>0: s += d
    _prev_bands = list(bands)
    return s/(len(bands)+1e-6)

def beat_drive(bands, rms, t):
    global _env, _punch, _pt
    lo = _low(bands)
    fx = _flux_global(bands)
    target = 0.25*lo + 0.75*fx + 0.10*rms
    target = target/(1 + 1.2*target)
    if target>_env: _env = 0.70*_env + 0.30*target
    else:           _env = 0.45*_env + 0.55*target
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t-_pt)); _pt = t
    _punch = max(_punch * pow(0.25, dt/0.05), _env)
    return max(0.0, min(1.0, _punch))

# ====== Layout & state ======
_streams = []  # each: {i, x, y, speed, hue, sat}
_layout = {"w":0, "h":0, "n":0, "char_h":18.0, "col_w":12.0}
_t_prev = None

def _band_hue(i, n):
    """Return a fixed hue (0-359) for band i. Bass (low i) is blue.~210°
    We travel from blue→cyan→green→yellow→orange→pink across the spectrum."""
    if n <= 1: return 210
    x = i / float(max(1, n-1))
    # piecewise stops: (position -> hue)
    stops = [(0.0, 210),   # blue (bass)
             (0.18, 190),  # blue-cyan
             (0.35, 140),  # green
             (0.55, 90),   # yellow-green
             (0.75, 40),   # orange
             (1.00, 320)]  # pink/magenta
    # find segment
    for a,(pa,ha) in enumerate(stops[:-1]):
        pb,hb = stops[a+1]
        if x <= pb:
            t = (x - pa)/max(1e-6, (pb-pa))
            return int(ha + t*(hb-ha))
    return stops[-1][1]

def _ensure_layout(w, h, bands_len):
    global _layout, _emaS, _emaL, _last_spawn, _streams
    if _layout["w"]==w and _layout["h"]==h and _layout["n"]==bands_len and _layout["char_h"]>0:
        return
    # Character size scales with screen; columns span FULL width (one per band)
    char_h = max(10.0, min(w,h)*0.035)   # height of one glyph in pixels
    col_w  = max(8.0, w/float(max(1, bands_len)))  # full-width packing
    _layout = {"w":w, "h":h, "n":bands_len, "char_h":char_h, "col_w":col_w}
    # Reset EMAs and spam limiter per band
    _emaS = [0.0]*bands_len
    _emaL = [0.0]*bands_len
    _last_spawn = [None]*bands_len
    # Re-center existing streams to their band column after resize
    for s in _streams:
        i = s.get("i", 0)
        s["x"] = int((i + 0.5)*col_w)

def _spawn_stream(i, t):
    # Create a new falling stream for band index i, starting at the top.
    x = int((i + 0.5) * _layout["col_w"])  # center of this band's column across full width
    # randomize inside the band column a bit
    x += int((_layout["col_w"]*0.40) * (random.random()-0.5))
    speed = random.uniform(420.0, 900.0)  # px/sec (fast)
    hue = _band_hue(i, max(1, _layout["n"]))  # fixed per-band color
    sat = 230  # strong color without being neon harsh
    tail = random.randint(8, 14)  # number of glyphs behind the head
    _streams.append({"i":i, "x":x, "y": -random.uniform(0, _layout['char_h']*tail),
                     "speed":speed, "hue":hue, "sat":sat, "tail":tail})
    # Enforce cap
    if len(_streams) > MAX_STREAMS:
        _streams.pop(0)

# Character set (binary digits + a few katakana for flavor)
_CHARS = "01アカサタナハマヤラワ01ﾊﾗｱﾀﾅ"

@register_visualizer
class MatrixRainBandColors(BaseVisualizer):
    display_name = "Matrix Rain (Band Colors)"

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Background: near-black
        p.fillRect(r, QBrush(QColor(6,6,8)))

        # Setup layout for current size/band count
        n = max(1, len(bands) if bands else 32)
        _ensure_layout(w, h, n)
        # If layout just reset, seed last_spawn to now to avoid burst
        for j in range(n):
            if _last_spawn[j] is None:
                _last_spawn[j] = t

        # Beat punch modulates global speed
        punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        # Per-band onsets for spawning
        for i in range(n):
            v = bands[i] if (bands and i < len(bands)) else 0.0
            _emaS[i] = 0.60*_emaS[i] + 0.40*v
            _emaL[i] = 0.96*_emaL[i] + 0.04*v
            onset = _emaS[i] - _emaL[i]
            margin = 0.10*_emaL[i] + 0.02
            min_interval = 0.20
            last = _last_spawn[i]
            can_spawn = (last is None) or ((t - last) > min_interval)
            # keepalive ensures occasional spawns even if a band is quiet
            if (onset > margin and can_spawn) or (can_spawn and (t - last if last is not None else 1e9) > KEEPALIVE_SEC):
                _spawn_stream(i, t)
                _last_spawn[i] = t

        # Additive glow for overlapping glyphs
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        # Font for glyphs
        f = QFont("Monospace", pointSize=1)
        f.setPixelSize(int(_layout["char_h"]))
        p.setFont(f)

        # Real dt
        global _t_prev
        dt = 0.016
        if _t_prev is not None:
            dt = max(0.0, min(0.06, t - _t_prev))
        _t_prev = t

        # Speed multiplier responds to beat
        speed_mul = 1.0 + 1.20*punch

        # Draw streams
        for s in list(_streams):
            s["y"] += s["speed"] * speed_mul * dt
            # remove once fully off-screen
            if s["y"] - s["tail"]*_layout["char_h"] > h + _layout["char_h"]:
                _streams.remove(s)
                continue

            # head bright; tail fades + bottom fade
            for k in range(0, s["tail"]):
                yy = s["y"] - k*_layout["char_h"]
                if yy < -_layout["char_h"] or yy > h + _layout["char_h"]:
                    continue
                ch = random.choice(_CHARS)
                fade_pos = max(0.0, min(1.0, 1.0 - (yy / float(h))))  # fades near bottom
                hue = s["hue"]         # fixed band color
                sat = s["sat"]
                val = int(max(40, (200 - k*12) * (0.35 + 0.65*fade_pos)))
                alpha = int(max(30, (220 - k*14) * (0.25 + 0.75*fade_pos)))
                color = QColor.fromHsv(hue, sat, min(255, val), min(255, alpha))
                p.setPen(QPen(color))
                p.drawText(int(s["x"]), int(yy), ch)
