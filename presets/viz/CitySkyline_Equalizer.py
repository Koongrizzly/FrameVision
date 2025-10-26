
from math import sin, pi
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QPolygonF, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- City Skyline Equalizer (v2) ---
# - **Much faster response** (stiffer spring) so it feels like a real EQ (low latency).
# - **Exactly 24 bands** are drawn (input bands are auto-compressed).
# - **Windows**: up to 50 "window pops" spawn on beats at the bottom and rise to the top.
# - Color palette still sweeps the hue wheel, with a beat nudge.

# ====== Audio helpers ======
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None  # last audio time for punch decay

# Dual-EMA trackers for robust onsets
_f_short = 0.0
_f_long  = 0.0

# ====== Motion/window state ======
_sdict = {}            # springs for per‑building heights
_rt = None             # last render time for dt (windows)
_win_particles = []    # rising windows {i, col, row, speed}
MAX_WIN = 80           # hard cap

TARGET_BANDS = 12      # draw exactly 12 buildings

# --- band utilities ---
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

def _compress_bands(bands, target=TARGET_BANDS):
    """Group adjacent bins so we always have exactly 'target' bands."""
    if not bands: return [0.0]*target
    m=len(bands)
    if m==target: return list(bands)
    out=[]
    for i in range(target):
        a=int(i*m/target)
        b=int((i+1)*m/target)
        if b<=a: b=a+1
        seg = bands[a:b]
        out.append(sum(seg)/float(len(seg)))
    return out

def spring_to(key, target, t, k=90.0, c=150.0, lo=0.03, hi=0.9):
    # Stiffer + more damping -> fast, near‑critical response (low lag)
    s, v, pt = _sdict.get(key, (target, 0.0, None))
    if pt is None:
        _sdict[key] = (target, 0.0, t)
        return target
    dt = max(0.0, min(0.033, t-pt))
    a = -k*(s-target) - c*v
    v += a*dt
    s += v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key] = (s, v, t)
    return s

def beat_drive(bands, rms, t):
    """Bass‑biased dual‑EMA onset for a clean 'punch' envelope."""
    global _env, _gate, _punch, _pt, _f_short, _f_long
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)

    target = 0.50*e + 1.00*f + 0.10*rms + 0.15*lo
    target = target / (1 + 1.20*target)

    # Faster release so it shrinks between hits
    if target>_env: _env = 0.70*_env + 0.30*target
    else:           _env = 0.45*_env + 0.55*target

    hi, lo_thr = 0.24, 0.14
    g = 1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate = 0.70*_gate + 0.30*g

    mix = 0.70*lo + 0.30*f
    _f_short = 0.65*_f_short + 0.35*mix
    _f_long  = 0.95*_f_long  + 0.05*mix
    onset = max(0.0, _f_short - _f_long)
    norm = _f_long + 1e-4
    onset_n = min(1.0, onset / (0.30*norm))

    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    _punch = max(_punch * pow(0.30, dt/0.05), onset_n)

    boom = min(1.0, max(0.0, 0.60*lo + 0.20*rms))
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch))

def _hsv(h,s,v,a=255):
    c = QColor.fromHsv(int(h)%360, int(max(0,min(255,s))), int(max(0,min(255,v))), int(max(0,min(255,a))))
    return c

def _ease(x): return (1.0 - (1.0-x)*(1.0-x))

# --- Visualizer ---------------------------------------------------------------
@register_visualizer
class CitySkylineEqualizerV2(BaseVisualizer):
    display_name = "City Skyline EQ (24‑band, fast)"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _rt, _win_particles
        w, h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Render dt for windows
        if _rt is None: _rt = t
        dt = max(0.0, min(0.06, t-_rt)); _rt = t

        p.fillRect(r, QBrush(QColor(8,8,12)))

        # Compress to 24 bands for drawing + analysis
        cb = _compress_bands(bands or [0.0]*TARGET_BANDS, TARGET_BANDS)
        env, gate, boom, punch = beat_drive(cb, rms or 0.0, t)

        # Layout
        margin = int(w*0.04)
        usable = max(1, w - 2*margin)
        n = TARGET_BANDS
        col_w = max(10, int(usable/float(n)))
        gap = max(1, int(col_w*0.06))
        depth = int(col_w*0.35)

        # Spawn a few window pops on beat (respect MAX_WIN)
        if punch > 0.60 and random.random() < 0.9:  # almost every strong beat
            spawn = min(6, MAX_WIN - len(_win_particles))
            for _ in range(spawn):
                _win_particles.append({"i": random.randrange(n), "col": None, "row": 0.0, "speed": random.uniform(6.0, 10.0)})
        if len(_win_particles) > MAX_WIN:
            _win_particles = _win_particles[-MAX_WIN:]  # keep newest

        # Color sweep base hue (fast enough to feel lively; beats nudge it)
        hue_base = (int((t*18) % 360) + int(140*punch)) % 360

        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Precompute baseline/mx height for speed
        base_h = 0.08*h
        max_h  = 0.90*h

        # Draw buildings
        for i in range(n):
            x0 = margin + i*col_w
            band_v = max(0.0, min(1.0, cb[i]))

            # **Very fast** height target (more from band, plus punch)
            tnorm = min(1.0, 0.06 + 0.94*(0.80*band_v + 0.50*punch + 0.30*env))
            h_norm = spring_to(f"h{i}", tnorm, t, k=90.0, c=14.0, lo=0.05, hi=1.0)
            height = base_h + (max_h-base_h)*h_norm

            # Colors for this building
            band_hue = (hue_base + int(360.0*(i/max(1,n)))) % 360
            front_col1 = _hsv(band_hue, 180, 220, 255)
            front_col2 = _hsv((band_hue+18)%360, 200, 130, 255)
            side_col   = _hsv((band_hue+330)%360, 180, 110, 255)
            edge_col   = QColor(0,0,0,180)

            # Geometry
            bx = x0 + gap//2
            by = int(h - height)
            bw = col_w - gap

            # Front face
            grad = QLinearGradient(QPointF(bx, by), QPointF(bx, by+height))
            grad.setColorAt(0.0, front_col1)
            grad.setColorAt(1.0, front_col2)
            p.setBrush(QBrush(grad)); p.setPen(QPen(edge_col, 2))
            p.drawRect(QRectF(bx, by, bw, height))

            # Right side & roof
            side = QPolygonF([QPointF(bx+bw, by),
                              QPointF(bx+bw+depth, by - depth*0.4),
                              QPointF(bx+bw+depth, by - depth*0.4 + height),
                              QPointF(bx+bw, by+height)])
            p.setBrush(QBrush(side_col)); p.setPen(QPen(edge_col, 2))
            p.drawPolygon(side)
            roof = QPolygonF([QPointF(bx, by),
                              QPointF(bx+bw, by),
                              QPointF(bx+bw+depth, by - depth*0.4),
                              QPointF(bx+depth, by - depth*0.4)])
            p.setBrush(QBrush(_hsv(band_hue, 80, 230, 220))); p.setPen(QPen(edge_col, 2))
            p.drawPolygon(roof)

            # Windows grid
            cell_w = max(6, int(bw*0.16))
            cell_h = max(8, int(cell_w*1.1))
            cols = max(2, int((bw - cell_w) / (cell_w + 4)))
            rows = max(3, int((height*0.85) / (cell_h + 4)))
            ox = bx + int((bw - (cols*cell_w + (cols-1)*4)) * 0.5)
            oy = by + int(height*0.10)

            # Draw rising window pops that belong to this building
            alive = []
            for wp in _win_particles:
                if wp["i"] != i:
                    alive.append(wp)
                    continue
                # Assign a column once we know how many cols this building has
                if wp["col"] is None:
                    wp["col"] = random.randint(0, max(0, cols-1))
                wp["row"] += wp["speed"] * dt * (1.0 + 0.8*punch)  # rows/sec
                r_int = int(wp["row"])
                if r_int >= rows:
                    # reached the top -> drop this particle
                    continue
                # Draw the lit window at (r_int, col)
                wx = ox + wp["col"]*(cell_w+4)
                wy = oy + r_int*(cell_h+4)
                a = 200  # bright
                win_col = QColor(255, 245, 200, a)
                p.setBrush(QBrush(win_col)); p.setPen(QPen(QColor(0,0,0,120), 1))
                p.drawRect(QRectF(wx, wy, cell_w, cell_h))
                alive.append(wp)
            _win_particles = alive

        # Ground line glow
        base_glow = QColor(80, 220, 140, int(40 + 120*punch))
        p.setPen(QPen(base_glow, max(2, int(h*0.006))))
        p.drawLine(int(margin*0.5), h-1, w-int(margin*0.5), h-1)
