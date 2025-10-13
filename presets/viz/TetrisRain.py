
import random, math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Tetris Rain ---
# Colorful tetrominoes fall from the sky. New pieces spawn on beats.
# Speed is nudged by the beat so heavy parts rain harder.
# Pieces fade out when they hit the ground; hard cap keeps it light.

# ===== Audio helpers =====
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

# ===== Shapes =====
# Tetromino definitions as sets of (x,y) blocks in a 4×4 bounding box
_TETROS = {
    'I': [(0,1),(1,1),(2,1),(3,1)],
    'O': [(1,1),(2,1),(1,2),(2,2)],
    'T': [(1,1),(0,2),(1,2),(2,2)],
    'S': [(1,1),(2,1),(0,2),(1,2)],
    'Z': [(0,1),(1,1),(1,2),(2,2)],
    'J': [(0,1),(0,2),(1,2),(2,2)],
    'L': [(2,1),(0,2),(1,2),(2,2)],
}
_PALETTE = [
    QColor(70,190,255), QColor(255,90,110), QColor(255,190,80),
    QColor(140,255,140), QColor(190,110,255), QColor(255,120,220),
    QColor(120,230,200),
]

# Piece state: {shape:str, blocks:[(x,y)], color:QColor, x,y(float), rot:int, spd:float, alpha:int}
_pieces=[]; _last_t=None
MAX_PIECES=80

def _rot_pt(pt, r):
    x,y = pt
    r%=4
    if r==0: return x,y
    if r==1: return 3-y, x
    if r==2: return 3-x, 3-y
    if r==3: return y, 3-x

def _spawn_piece(w, cols):
    shape = random.choice(list(_TETROS.keys()))
    color = random.choice(_PALETTE)
    rot = random.randint(0,3)
    blocks = [_rot_pt(pt,rot) for pt in _TETROS[shape]]
    # column grid spawn
    col = random.randint(0, cols-4)
    return { 'shape':shape, 'blocks':blocks, 'color':color, 'x':float(col), 'y':-4.0, 'rot':rot,
             'spd': random.uniform(4.0, 9.0), 'alpha':255 }

@register_visualizer
class TetrisRain(BaseVisualizer):
    display_name = "Tetris Rain"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _pieces, _last_t
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t=t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t=t

        p.fillRect(r, QBrush(QColor(10,10,14)))

        # Beat analysis
        n = max(16, len(bands) if bands else 32)
        env, gate, boom, punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        # Grid layout
        cols = 16
        margin = int(w*0.05)
        usable = w - 2*margin
        tile = max(8, int(min(usable/cols, h/20)))
        left = margin + (usable - cols*tile)//2
        ground = h - int(tile*0.5)

        # Spawn on strong beats
        if punch > 0.85:
            for _ in range(random.randint(1,3)):
                if len(_pieces) < MAX_PIECES:
                    _pieces.append(_spawn_piece(w, cols))

        # Update & draw
        alive=[]
        for pc in _pieces:
            # gravity with beat nudge
            pc['y'] += (pc['spd'] + 10.0*punch) * dt
            ypix = int(pc['y']*tile)
            # hit the ground -> start fading
            if left >= 0 and (ypix + 4*tile) >= ground:
                pc['alpha'] = max(0, pc['alpha'] - int(400*dt))
                pc['y'] += 2.0*dt  # slight slide
            # cull if fully gone
            if pc['alpha'] <= 0 or ypix > h + 4*tile:
                continue

            # draw blocks
            col = QColor(pc['color']); col.setAlpha(max(30, pc['alpha']))
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor(0,0,0,160), 2))
            for bx,by in pc['blocks']:
                x = left + int((pc['x'] + bx)*tile)
                y = int(pc['y']*tile + by*tile)
                p.drawRect(QRectF(x, y, tile, tile))

            alive.append(pc)
        _pieces = alive

        # Subtle ground glow on beat
        p.setPen(QPen(QColor(120,220,255,int(30+150*punch)), max(2,int(tile*0.25))))
        p.drawLine(left, ground, left + cols*tile, ground)
