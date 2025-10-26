
import random, math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# --- Rubik's Cube Pulse (no-glow) + Silence Gate ---
_prev=[]; _env=_gate=_punch=0.0; _pt=None
_f_short=_f_long=0.0

# Pause/static detection (so we can freeze motion)
_prev_frame_bands=None
_prev_rms=0.0
_static_quiet_time=0.0
_last_active_t=None
_silence_mode=False
_pt_gate=None

# Gate tunables (similar to your other visuals)
PAUSE_STATIC_SEC = 0.35
STATIC_FLUX_THR  = 0.015
STATIC_BANDS_EPS = 0.002
STATIC_RMS_EPS   = 0.002
SILENCE_RMS_THR  = 0.015
SILENCE_HOLD_SEC = 0.60
WAKE_EAGER_SEC   = 0.10

# Integrated rotation state (so we can *freeze* when silent)
_ax=0.0
_ay=0.0

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
    return max(0, min(1,_env)), max(0, min(1,_gate)), boom, max(0, min(1,_punch)), dt, f

def _rotate3d(pt, ax, ay):
    x,y,z = pt
    cx=math.cos(ax); sx=math.sin(ax)
    y,z = y*cx - z*sx, y*sx + z*cx
    cy=math.cos(ay); sy=math.sin(ay)
    x,z = x*cy + z*sy, -x*sy + z*cy
    return x,y,z

def _project(pt, w, h, scale=1.0):
    x,y,z = pt
    dz = max(0.6, 3.0 + z)
    px = x*scale/dz; py = y*scale/dz
    cx, cy = w*0.5, h*0.55
    return QPointF(cx + px, cy + py)

def _face_quads(size=1.0):
    step = size/3.0
    offs = [-size/2 + step*i for i in range(3)]
    faces = []
    # +Z
    f0=[]; z=size/2
    for iy,y in enumerate(reversed(offs)):
        row=[]
        for ix,x in enumerate(offs):
            row.append([(x,y,z),(x+step,y,z),(x+step,y+step,z),(x,y+step,z)])
        f0.append(row)
    faces.append(f0)
    # +X
    f1=[]; x=size/2
    for iy,y in enumerate(reversed(offs)):
        row=[]
        for iz,z in enumerate(reversed(offs)):
            row.append([(x,y,z),(x,y+step,z),(x,y+step,z+step),(x,y,z+step)])
        f1.append(row)
    faces.append(f1)
    # -Z
    f2=[]; z=-size/2
    for iy,y in enumerate(reversed(offs)):
        row=[]
        for ix,x in enumerate(reversed(offs)):
            row.append([(x,y,z),(x,y+step,z),(x+step,y+step,z),(x+step,y,z)])
        f2.append(row)
    faces.append(f2)
    # -X
    f3=[]; x=-size/2
    for iy,y in enumerate(reversed(offs)):
        row=[]
        for iz,z in enumerate(offs):
            row.append([(x,y,z),(x,y,z+step),(x,y+step,z+step),(x,y+step,z)])
        f3.append(row)
    faces.append(f3)
    # +Y
    f4=[]; y=size/2
    for iz,z in enumerate(reversed(offs)):
        row=[]
        for ix,x in enumerate(offs):
            row.append([(x,y,z),(x+step,y,z),(x+step,y,z+step),(x,y,z+step)])
        f4.append(row)
    faces.append(f4)
    # -Y
    f5=[]; y=-size/2
    for iz,z in enumerate(offs):
        row=[]
        for ix,x in enumerate(offs):
            row.append([(x,y,z),(x,y,z+step),(x+step,y,z+step),(x+step,y,z)])
        f5.append(row)
    faces.append(f5)
    return faces

_faces_geo = _face_quads(1.6)

# Sticker colors (start randomized but pleasant)
_faces = []
def _init_faces():
    global _faces
    if _faces: return
    palette = [QColor(240,240,240), QColor(255,80,80), QColor(80,160,255),
               QColor(255,210,0), QColor(80,200,120), QColor(255,120,220)]
    rnd = random.Random(42)
    _faces = [[ [rnd.choice(palette) for _ in range(3)] for __ in range(3) ] for _ in range(6)]

def _rand_color():
    hue = random.randint(0,359)
    return QColor.fromHsv(hue, random.randint(120,220), random.randint(180,255))

def _activity_gate(bands, rms, t, flux):
    """Detects when audio is paused or extremely static."""
    global _prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t,_silence_mode,_pt_gate
    if _pt_gate is None: _pt_gate=t
    dt = max(0.0, min(0.033, t-_pt_gate)); _pt_gate=t

    avg_delta = 0.0
    if bands and _prev_frame_bands and len(_prev_frame_bands)==len(bands):
        s=0.0
        for a,b in zip(_prev_frame_bands, bands): s += abs(a-b)
        avg_delta = s/len(bands)
    if (avg_delta < STATIC_BANDS_EPS) and (abs(rms-_prev_rms) < STATIC_RMS_EPS) and (flux < STATIC_FLUX_THR):
        _static_quiet_time += dt
    else:
        _static_quiet_time = 0.0
    _prev_frame_bands = list(bands) if bands else None
    _prev_rms = rms

    active_now = (rms > SILENCE_RMS_THR) or (flux > 0.18)
    if _last_active_t is None: _last_active_t = t
    if active_now: _last_active_t = t

    prev = _silence_mode
    _silence_mode = (_static_quiet_time > PAUSE_STATIC_SEC) or ((t - _last_active_t) > SILENCE_HOLD_SEC)
    if _silence_mode and active_now and (t - _last_active_t) <= WAKE_EAGER_SEC:
        _silence_mode = False

    return (not _silence_mode), dt

@register_visualizer
class RubiksCubePulseNoGlow(BaseVisualizer):
    display_name = "Rubik's Cube Pulse (no glow)"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _ax,_ay
        _init_faces()
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Solid background
        p.fillRect(r, QBrush(QColor(8,8,12)))

        # Beat / onset
        env, gate, boom, punch, dt_drive, flux = beat_drive(bands or [0.0]*24, rms or 0.0, t)

        # Activity gate
        is_active, dt_gate = _activity_gate(bands or [0.0]*24, rms or 0.0, t, flux)

        # Recolor stickers only when *active* and on strong onset
        if is_active and punch > 0.6 and random.random() < 0.8:
            for _ in range(5):
                f = random.randrange(6)
                i = random.randrange(3); j = random.randrange(3)
                _faces[f][i][j] = _rand_color()

        # --- Rotation: integrate only while active (freezes when silent) ---
        if is_active:
            # velocities inspired by original t-based motion, plus small punch kick
            vax = 0.35*math.cos(0.7*t)  # ≈ d/dt of 0.5*sin(0.7t)
            vay = 0.40 + 0.075*math.cos(0.3*t)  # ≈ d/dt of (0.4*t + 0.25*sin(0.3t))
            _ax += (vax + 0.60*punch) * dt_gate
            _ay += (vay + 1.20*punch) * dt_gate
        # else: keep _ax/_ay unchanged → frozen pose

        # Build depth-sorted faces
        draw_faces = []
        scale = min(w,h)*0.9
        for fi, face in enumerate(_faces_geo):
            quads2d = []
            zsum = 0.0; count = 0
            for i,row in enumerate(face):
                for j,quad in enumerate(row):
                    poly = []
                    for pt in quad:
                        x,y,z = _rotate3d(pt, _ax, _ay)
                        zsum += z; count += 1
                        poly.append(_project((x,y,z), w, h, scale=scale))
                    quads2d.append((i,j,poly))
            depth = zsum / max(1,count)
            draw_faces.append((depth, fi, quads2d))

        p.setRenderHint(QPainter.Antialiasing, True)
        draw_faces.sort(key=lambda x: x[0])

        edge = QPen(QColor(0,0,0,220), max(1,int(min(w,h)*0.005)))
        sep  = max(1,int(min(w,h)*0.009))

        # Draw stickers
        for depth, fi, quads in draw_faces:
            for i,j,poly in quads:
                c = QColor(_faces[fi][i][j]); c.setAlpha(240)
                p.setBrush(QBrush(c))
                p.setPen(edge)
                p.drawPolygon(QPolygonF(poly))
                p.setPen(QPen(QColor(20,20,24,230), sep))
                p.setBrush(QBrush(QColor(0,0,0,0)))  # ensure no fill for outline strokes
                p.drawPolygon(QPolygonF(poly))
