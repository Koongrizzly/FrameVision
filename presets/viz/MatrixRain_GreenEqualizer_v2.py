# --- MatrixRain_GreenEqualizer_v2.py ---
# Changes requested:
# - Hard cap: at most 100 streams on screen at once.
# - Much faster fall speed (uses real dt, not a fixed step).
# - Streams fade out by the time they reach the bottom.
# - Slightly stricter spawn rate to avoid flooding and keep FPS healthy.


from math import pi
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# --- What this visual does (plain English) ------------------------------------
# - Only green "Matrix" code, raining downward.
# - Acts like an upside-down equalizer: when a frequency band hits, a NEW stream
#   of code appears at the TOP for that band and falls down the screen.
# - Existing streams keep falling while new ones spawn at the top.
# - Speed subtly speeds up on strong beats (so the whole rain pumps to the music).
# -----------------------------------------------------------------------------

# ====== Audio helpers (lightweight) ======
_prev_bands = []
_env = 0.0
_punch = 0.0
_pt = None

# Short/Long EMA per-band for robust onset detection (so it keeps pulsing forever)
_emaS = []   # short window (reacts fast)
_emaL = []   # long window (slow baseline)
_last_spawn = []  # per-band: last time we spawned a stream, to avoid spam

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
    # Goal: create a 'punch' that rises on beats and drops fast between beats.
    global _env, _punch, _pt
    lo = _low(bands)
    fx = _flux_global(bands)
    # Energy target (gentle, to avoid pinning)
    target = 0.25*lo + 0.75*fx + 0.10*rms
    target = target/(1 + 1.2*target)
    # Envelope: quick rise, quicker fall
    if target>_env: _env = 0.70*_env + 0.30*target
    else:           _env = 0.45*_env + 0.55*target
    # Time step + fast-decaying punch
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t-_pt)); _pt = t
    # 'Punch' pulls towards envelope on hits and decays quickly
    _punch = max(_punch * pow(0.25, dt/0.05), _env)
    return max(0.0, min(1.0, _punch))

# ====== Matrix rain state ======
MAX_STREAMS = 100  # hard cap on simultaneous streams
KEEPALIVE_SEC = 0.80  # ensure each band spawns occasionally
_t_prev = None      # last timestamp for dt
_streams = []  # each: {i, x, y, speed, hue, tail, jitter}
_layout = {"w":0, "h":0, "n":0, "char_h":18.0, "col_w":12.0}

def _ensure_layout(w, h, bands_len):
    # Recompute character metrics and per-band spacing if size or band count changed
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
    # Re-center existing streams to their band column so resizes don't blank the screen
    for s in _streams:
        i = s.get("i", 0)
        s["x"] = int((i + 0.5)*col_w)


def _spawn_stream(i, t):
    # Create a new falling stream for band index i, starting at the top.
    x = int((i + 0.5) * _layout["col_w"])  # center of this band's column across full width
    # randomize inside the band column a bit
    x += int((_layout["col_w"]*0.40) * (random.random()-0.5))
    speed = random.uniform(420.0, 900.0)  # much faster  # px/sec
    hue = 120  # pure green
    tail = random.randint(8, 14)  # shorter tail for clarity/perf        # number of glyphs behind the head
    jitter = random.uniform(0.0, 0.25)   # tiny hue jitter for variety
    _streams.append({"i":i, "x":x, "y": -random.uniform(0, _layout['char_h']*tail),
                     "speed":speed, "hue":hue, "tail":tail, "jitter":jitter})
    # Enforce cap: if too many streams, remove oldest
    if len(_streams) > MAX_STREAMS:
        _streams.pop(0)

# Character set (binary digits + a few katakana for flavor)
_CHARS = "01アカサタナハマヤラワ01ﾊﾗｱﾀﾅ"

@register_visualizer
class MatrixRainGreenEqualizer(BaseVisualizer):
    display_name = "Matrix Rain (Green Equalizer)"

    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Background: near-black for neon green pop
        p.fillRect(r, QBrush(QColor(6,6,8)))

        # Layout & per-band EMAs
        n = max(1, len(bands) if bands else 32)
        _ensure_layout(w, h, n)
        # If layout just reset (None entries), seed last_spawn to now to avoid a burst
        for j in range(n):
            if _last_spawn[j] is None:
                _last_spawn[j] = t

        # Beat 'punch' for global speed modulation
        punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        # Per-band onsets: short vs long EMA over the *band value* (simple & robust)
        # If short EMA rises above long EMA by a margin, spawn a stream for that band.
        # Margin scales with long EMA so it adapts to volume changes.
        for i in range(n):
            v = bands[i] if (bands and i < len(bands)) else 0.0
            # EMA updates (0..1-ish). Short reacts fast, Long tracks trend.
            _emaS[i] = 0.60*_emaS[i] + 0.40*v
            _emaL[i] = 0.96*_emaL[i] + 0.04*v
            onset = _emaS[i] - _emaL[i]
            margin = 0.10*_emaL[i] + 0.02  # adaptive + small floor
            # simple spam limiter per band
            min_interval = 0.20  # seconds (allows up to ~8 spawns/sec per band if really active)
            last = _last_spawn[i]
            can_spawn = (last is None) or ((t - last) > min_interval)
            if (onset > margin and can_spawn) or (can_spawn and (t - last if last is not None else 1e9) > KEEPALIVE_SEC):
                _spawn_stream(i, t)
                _last_spawn[i] = t

        # Update & draw streams
        # Use CompositionMode_Plus so overlapping greens glow
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        # Monospace-like font for the rain
        f = QFont("Monospace", pointSize=1)
        f.setPixelSize(int(_layout["char_h"]))
        p.setFont(f)

        # Compute real dt and speed multiplier (faster overall)
        global _t_prev
        dt = 0.016
        if _t_prev is not None:
            dt = max(0.0, min(0.06, t - _t_prev))
        _t_prev = t
        speed_mul = 1.0 + 1.20*punch
        # Draw each stream head + short tail
        for s in list(_streams):
            s["y"] += s["speed"] * speed_mul * dt  # approx frame step; renderer caps dt anyway
            # Remove when off-screen
            if s["y"] - s["tail"]*_layout["char_h"] > h + _layout["char_h"]:
                _streams.remove(s)
                continue

            # Head brighter, tail fades
            for k in range(0, s["tail"]):
                yy = s["y"] - k*_layout["char_h"]
                if yy < -_layout["char_h"] or yy > h + _layout["char_h"]:
                    continue
                ch = random.choice(_CHARS)
                # Classic Matrix green with slight variations
                hue = (s["hue"] + int(s["jitter"]*30)) % 360  # stay greenish
                sat = 255
                val = max(60, 220 - k*12)  # fade along tail
                alpha = max(50, 240 - k*14)
                color = QColor.fromHsv(hue, sat, val, alpha)
                p.setPen(QPen(color))
                p.drawText(int(s["x"]), int(yy), ch)
