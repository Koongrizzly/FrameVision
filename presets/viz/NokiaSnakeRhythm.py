
# --- Nokia Snake Rhythm (Beat-Step v1.1) ---
# - Snake advances exactly ONE grid cell every 2 beats.
# - Direction is randomly picked from {UP,DOWN,LEFT,RIGHT} on each 4-beat tick
#   (won't immediately reverse into itself).
# - Smooth interpolation between cells so motion is visible between beats.
# - Apples spawn on bass booms; eating grows the snake.
# - Colors pulse to env; grid glints on hits.
from random import Random, choice, randint
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer


# ---- beat kit (same flavor as other viz) ----
_prev = []
_env=_gate=_punch=0.0
_pt=None
_beat_count=2
_since_bc = 0.0  # seconds since last beat (fallback)
_FBK_BPM = 126.0  # fallback BPM for movement


def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut,n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s+=w*bands[i]; c+=1
    return s/max(1,c)

def _low(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(0,cut):
        w=1.0 - 0.4*(i/max(1,cut-1))
        s+=w*bands[i]; c+=1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands: _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n: _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut,n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3 + 0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i]+0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_beat_count,_since_bc
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target=0.58*e + 1.30*f + 0.18*rms + 0.22*lo
    target=target/(1+0.7*target)
    if target>_env: _env=0.72*_env+0.28*target
    else: _env=0.92*_env+0.08*target
    hi,lo_thr=0.30,0.18
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    rising = g>0.6 and _gate<=0.6
    _gate=0.82*_gate+0.18*g
    boom=min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    if _pt is None: _pt=t
    dt=max(0.0, min(0.033, t-_pt)); _pt=t
    decay=pow(0.78, dt/0.016) if dt>0 else 0.78
    # Fallback beat synthesis to keep snake moving
    _since_bc = max(0.0, _since_bc + dt)
    beat_sec = 60.0 / _FBK_BPM if _FBK_BPM>0 else 0.5
    if rising:
        _beat_count += 1
        _since_bc = 0.0
    elif beat_sec>0 and _since_bc >= beat_sec:
        add = int(_since_bc // beat_sec)
        if add>0:
            _beat_count += add
            _since_bc -= add*beat_sec
    _punch=max(_punch*decay, 1.0 if rising else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch)), _beat_count, dt

# ---- snake state ----
_rng = Random(9090)
GRID_W = 28
GRID_H = 18
STEP_PERIOD_BEATS = 1  # move once every 2 beats

snake = []
dir_vec = (1,0)
last_move_bc = -1
grow_left = 0
apple = None
head_px = (0.0,0.0)

def reset(w,h):
    global snake, dir_vec, last_move_bc, grow_left, apple, head_px
    cx, cy = GRID_W//2, GRID_H//2
    snake = [(cx-i, cy) for i in range(6)]
    dir_vec = (1,0)
    last_move_bc = -1
    grow_left = 0
    apple = (min(GRID_W-2, cx+5), cy)
    head_px = (0.0,0.0)

def rand_dir(not_back=None):
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    if not_back:
        dirs=[d for d in dirs if (d[0],d[1])!=(-not_back[0], -not_back[1])]
    return choice(dirs)

def step_snake(bc):
    global snake, dir_vec, grow_left, apple, last_move_bc
    if bc%STEP_PERIOD_BEATS!=0 or bc<=0 or last_move_bc==bc:
        return False
    last_move_bc = bc
    # choose new random direction (avoid immediate reversal)
    dir_vec = rand_dir(not_back=dir_vec)
    hx,hy = snake[0]
    nx = (hx + dir_vec[0]) % GRID_W
    ny = (hy + dir_vec[1]) % GRID_H
    new_head=(nx,ny)
    # eat?
    ate = (apple is not None and new_head==apple)
    if ate:
        # grow 2-4
        add = 2+_rng.randint(0,2)
        grow_left += add
    snake.insert(0, new_head)
    if grow_left>0:
        grow_left -= 1
    else:
        snake.pop()  # remove tail
    # respawn apple on eat
    if ate:
        spawn_apple()
    return True

def spawn_apple():
    global apple
    # place apple not on snake
    tries=0
    while True:
        ax = _rng.randint(0, GRID_W-1)
        ay = _rng.randint(0, GRID_H-1)
        if (ax,ay) not in snake:
            apple=(ax,ay); break
        tries+=1
        if tries>200:
            apple=None; break

@register_visualizer
class NokiaSnakeRhythm(BaseVisualizer):
    display_name = "Nokia Snake Rhythm"
    def paint(self, p:QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        if not snake: reset(w,h)

        env, gate, boom, punch, bc, dt = beat_drive(bands, rms, t)

        # burst apples on strong boom
        if boom>0.75 and _rng.random()<0.12:
            spawn_apple()
        # --- Catch up snake steps if beat counter skipped multiples ---
        prev_block = last_move_bc // STEP_PERIOD_BEATS if last_move_bc >= 0 else -1
        curr_block = bc // STEP_PERIOD_BEATS
        if curr_block > prev_block:
            for _blk in range(prev_block + 1, curr_block + 1):
                step_snake(_blk * STEP_PERIOD_BEATS)
        # --- Time-based safety: ensure at least one step happens on a steady clock ---
        global _move_accum
        try:
            _move_accum += dt
        except NameError:
            _move_accum = dt
        secs_per_step = (60.0 / max(1e-3, _FBK_BPM)) * STEP_PERIOD_BEATS
        if _move_accum >= secs_per_step:
            _next_blk = (last_move_bc // STEP_PERIOD_BEATS + 1) if last_move_bc >= 0 else 1
            step_snake(_next_blk * STEP_PERIOD_BEATS)
            _move_accum -= secs_per_step
        # background
        p.fillRect(r, QBrush(QColor(8,10,12)))
        # subtle grid
        cell = min(w/max(1,GRID_W), h/max(1,GRID_H))
        gx = (w - GRID_W*cell)/2
        gy = (h - GRID_H*cell)/2
        gl = int(40 + 80*boom)
        p.setPen(QPen(QColor(80,120,160,gl//4), 1))
        for i in range(GRID_W+1):
            x = gx + i*cell
            p.drawLine(QPointF(x,gy), QPointF(x, gy+GRID_H*cell))
        for j in range(GRID_H+1):
            y = gy + j*cell
            p.drawLine(QPointF(gx,y), QPointF(gx+GRID_W*cell, y))

        # draw apple
        if apple is not None:
            ax,ay=apple
            axp = gx + (ax+0.5)*cell; ayp = gy + (ay+0.5)*cell
            hue = int((t*60 + 260*env) % 360)
            col = QColor.fromHsv(hue, 240, 255, 230)
            p.setBrush(QBrush(col)); p.setPen(QPen(QColor(255,255,255,180), 2))
            p.drawEllipse(QPointF(axp, ayp), cell*0.25*(1.0+0.2*punch), cell*0.25*(1.0+0.2*punch))

        # snake color
        hue = int((t*40 + 140*env) % 360)
        body_col = QColor.fromHsv(hue, 220, 230, 240)
        edge_col = QColor.fromHsv((hue+40)%360, 180, 255, 180)
        head_col = QColor.fromHsv((hue+80)%360, 240, 255, 255)

        # draw snake segments
        p.setPen(QPen(edge_col, max(2,int(cell*0.10))))
        for idx,(sx,sy) in enumerate(snake):
            x = gx + sx*cell; y = gy + sy*cell
            rect = QRectF(x+cell*0.08, y+cell*0.08, cell*0.84, cell*0.84)
            if idx==0:
                p.setBrush(QBrush(head_col))
            else:
                p.setBrush(QBrush(body_col))
            p.drawRoundedRect(rect, cell*(0.30 if idx==0 else 0.20), cell*(0.30 if idx==0 else 0.20))

        # eyes on head indicate current chosen dir by shifting pupils
        if snake:
            hx,hy = snake[0]
            x = gx + hx*cell; y=gy + hy*cell
            eye_r = cell*0.10
            ex1 = x + cell*0.35; ey1 = y + cell*0.35
            ex2 = x + cell*0.65; ey2 = y + cell*0.35
            dx,dy = dir_vec
            px_off = dx*cell*0.12; py_off = dy*cell*0.12
            p.setBrush(QBrush(QColor(20,24,32,255))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(ex1,ey1), eye_r, eye_r)
            p.drawEllipse(QPointF(ex2,ey2), eye_r, eye_r)
            p.setBrush(QBrush(QColor(230,240,255,230)))
            p.drawEllipse(QPointF(ex1+px_off,ey1+py_off), eye_r*0.55, eye_r*0.55)
            p.drawEllipse(QPointF(ex2+px_off,ey2+py_off), eye_r*0.55, eye_r*0.55)
