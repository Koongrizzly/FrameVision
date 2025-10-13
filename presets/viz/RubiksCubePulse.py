
import random, math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF
from PySide6.QtCore import QPointF
from helpers.music import register_visualizer, BaseVisualizer

# --- Rubik's Cube Pulse (no-glow) ---
# Same as before, but **no big circle/glow**. Nothing will cover the cube.
# Also forces a transparent brush before any outlines so no accidental fills happen.

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
    return max(0, min(1,_env)), max(0, min(1,_gate)), boom, max(0, min(1,_punch))

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

@register_visualizer
class RubiksCubePulseNoGlow(BaseVisualizer):
    display_name = "Rubik's Cube Pulse (no glow)"

    def paint(self, p: QPainter, r, bands, rms, t):
        _init_faces()
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # Solid background (no overlays)
        p.fillRect(r, QBrush(QColor(8,8,12)))

        # Beat (only for color changes)
        env, gate, boom, punch = beat_drive(bands or [0.0]*24, rms or 0.0, t)

        # On beat, recolor a few stickers
        if punch > 0.6 and random.random() < 0.8:
            for _ in range(5):
                f = random.randrange(6)
                i = random.randrange(3); j = random.randrange(3)
                _faces[f][i][j] = _rand_color()

        # Rotation
        ax = 0.5*math.sin(t*0.7) + 0.15*punch
        ay = t*0.4 + 0.25*math.sin(t*0.3) + 0.3*punch

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
                        x,y,z = _rotate3d(pt, ax, ay)
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

        # No circle / no glow here — nothing to cover the cube.
