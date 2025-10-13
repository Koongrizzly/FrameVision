
import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Tetris Stacker (black bg + neon BORDER, no top line) ---
# Same as TetrisStacker_blackBorder but the TOP border line is removed.

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

TETROS = {
    'I': [(0,1),(1,1),(2,1),(3,1)],
    'O': [(1,1),(2,1),(1,2),(2,2)],
    'T': [(1,1),(0,2),(1,2),(2,2)],
    'S': [(1,1),(2,1),(0,2),(1,2)],
    'Z': [(0,1),(1,1),(1,2),(2,2)],
    'J': [(0,1),(0,2),(1,2),(2,2)],
    'L': [(2,1),(0,2),(1,2),(2,2)],
}
PALETTE = [
    QColor(70,190,255), QColor(255,90,110), QColor(255,190,80),
    QColor(140,255,140), QColor(190,110,255), QColor(255,120,220),
    QColor(120,230,200),
]

COLS, ROWS = 12, 22
GRID = [[None for _ in range(COLS)] for __ in range(ROWS)]
ACTIVE = []
CLEARING = []   # (row_index, time_remaining)
_last_t = None
_board_flash = 0.0

def _rot(pt, r):
    x,y = pt
    r%=4
    if r==0: return x,y
    if r==1: return 3-y, x
    if r==2: return 3-x, 3-y
    if r==3: return y, 3-x

def _spawn():
    shape = random.choice(list(TETROS.keys()))
    color = random.choice(PALETTE)
    rot = random.randint(0,3)
    blocks = [_rot(pt,rot) for pt in TETROS[shape]]
    x = random.randint(0, max(0, COLS-4))
    return {'blocks':blocks, 'x':float(x), 'y':-3.5, 'spd': random.uniform(3.5, 7.0), 'color':color}

def _collides(pc, nx, ny):
    for bx,by in pc['blocks']:
        gx = int(nx + bx); gy = int(ny + by)
        if gx < 0 or gx >= COLS: return True
        if gy >= ROWS: return True
        if gy >= 0 and GRID[gy][gx] is not None: return True
    return False

def _lock(pc):
    global GRID, ACTIVE, _board_flash
    for bx,by in pc['blocks']:
        gx = int(pc['x'] + bx); gy = int(pc['y'] + by)
        if 0 <= gx < COLS and 0 <= gy < ROWS:
            GRID[gy][gx] = QColor(pc['color'])
    if any(GRID[0][c] is not None for c in range(COLS)):
        GRID = [[None for _ in range(COLS)] for __ in range(ROWS)]
        ACTIVE.clear()
        _board_flash = 0.25

def _check_and_clear():
    global GRID, CLEARING
    full = [ri for ri in range(ROWS) if all(GRID[ri][c] is not None for c in range(COLS))]
    if not full: return
    CLEARING = [(ri, 0.18) for ri in full]
    remaining = [GRID[ri] for ri in range(ROWS) if ri not in set(full)]
    GRID = [[None for _ in range(COLS)] for _ in range(len(full))] + remaining

@register_visualizer
class TetrisStackerBlackBorderNoTop(BaseVisualizer):
    display_name = "Tetris Stacker (black + neon border, no top)"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t, ACTIVE, CLEARING, _board_flash
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t=t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t=t

        p.fillRect(r, QBrush(QColor(0,0,0)))

        n = max(16, len(bands) if bands else 32)
        env, gate, boom, punch = beat_drive(bands or [0.0]*n, rms or 0.0, t)

        tile = max(12, int(min(w*0.8/COLS, h*0.92/ROWS)))
        left = int((w - COLS*tile)/2)
        top  = int(h - ROWS*tile - tile*0.04)
        pf_rect = QRectF(left, top, COLS*tile, ROWS*tile)

        spawn_prob = 0.10 + 0.70*punch
        if len(ACTIVE) < 6 and random.random() < spawn_prob:
            ACTIVE.append(_spawn())

        survivors = []
        for pc in ACTIVE:
            vy = (pc['spd'] + 8.0*punch) * dt
            nx, ny = pc['x'], pc['y'] + vy
            if _collides(pc, nx, ny):
                _lock(pc)
            else:
                pc['y'] = ny
                survivors.append(pc)
        ACTIVE = survivors

        _check_and_clear()

        block_edge = QPen(QColor(0,0,0,230), max(1, int(tile*0.10)))
        p.setPen(block_edge)
        for rri in range(ROWS):
            for cc in range(COLS):
                col = GRID[rri][cc]
                if col:
                    x = left + cc*tile; y = top + rri*tile
                    p.setBrush(QBrush(col))
                    p.drawRect(QRectF(x, y, tile, tile))

        if CLEARING:
            keep=[]
            for ri, tleft in CLEARING:
                a = int(255 * max(0.0, tleft/0.18))
                p.setBrush(QBrush(QColor(255,255,255,a)))
                p.setPen(QPen(QColor(255,255,255,a), 2))
                p.drawRect(QRectF(left, top + ri*tile, COLS*tile, tile))
                if tleft - dt > 0: keep.append((ri, tleft-dt))
            CLEARING = keep

        p.setPen(block_edge)
        for pc in ACTIVE:
            col = QColor(pc['color']); col.setAlpha(240)
            p.setBrush(QBrush(col))
            for bx,by in pc['blocks']:
                x = left + int((pc['x'] + bx)*tile)
                y = top + int((pc['y'] + by)*tile)
                p.drawRect(QRectF(x, y, tile, tile))

        # NEON BORDER (no top line)
        base_hue = int((t*30) % 360)
        hues = [base_hue, (base_hue+90)%360, (base_hue+180)%360]
        colors = [QColor.fromHsv(h, 200, 230, 220) for h in hues]
        thick = max(2, int(tile*0.15))
        # right
        p.setPen(QPen(colors[0], thick))
        p.drawLine(left+COLS*tile, top, left+COLS*tile, top+ROWS*tile)
        # bottom
        p.setPen(QPen(colors[1], thick))
        p.drawLine(left, top+ROWS*tile, left+COLS*tile, top+ROWS*tile)
        # left
        p.setPen(QPen(colors[2], thick))
        p.drawLine(left, top, left, top+ROWS*tile)

        if _board_flash > 0.0:
            a = int(255 * min(1.0, _board_flash / 0.25))
            p.setBrush(QBrush(QColor(255,255,255,a)))
            p.setPen(QPen(QColor(255,255,255,a), 1))
            p.drawRect(pf_rect)
            _board_flash = max(0.0, _board_flash - dt)
