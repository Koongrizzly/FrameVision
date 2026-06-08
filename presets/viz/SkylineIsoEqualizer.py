# SkylineIsoEqualizer.py
# Isometric mini-city equalizer.
# Lows -> building height pump
# Mids -> window light patterns
# Highs -> fireworks + rooftop beacons
from math import sin, cos, pi
from random import Random, randint, random, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# ---- audio features (fast + snappy) ----
_prev_spec = []
_env_lo = _env_mid = _env_hi = 0.0
_gate_hi = 0.0
_last_t = None

_rng = Random(4242)

def _band_split(bands):
    if not bands: return 0.0,0.0,0.0
    n = len(bands)
    a = max(1, n//6)         # lows ~ first 1/6
    b = max(a+1, n//2)       # mids ~ middle up to half
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if not _prev_spec or len(_prev_spec)!=len(bands):
        _prev_spec = [0.0]*len(bands)
    f = 0.0
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d>0: f += d * (0.35 + 0.65*(i/len(bands)))
    _prev_spec = [0.85*px + 0.15*x for x,px in zip(bands,_prev_spec)]
    return f / max(1,len(bands))

def _features(bands, rms, t):
    global _env_lo, _env_mid, _env_hi, _gate_hi, _last_t
    lo, mid, hi = _band_split(bands)
    fx = _flux(bands)

    # snappy envelopes (quick rise, modest fall)
    def env_update(env, target, up=0.30, down=0.10):
        return (1-up)*env + up*target if target>env else (1-down)*env + down*target

    _env_lo = env_update(_env_lo, 1.2*lo + 0.2*rms)
    _env_mid = env_update(_env_mid, 1.1*mid + 0.1*rms)
    # high channel prioritizes increases (transients)
    hi_target = 0.8*hi + 1.4*fx
    _env_hi = env_update(_env_hi, hi_target, up=0.5, down=0.15)

    # short gate on high transients (for fireworks)
    thr_hi = 0.18
    rising = hi_target > thr_hi and _gate_hi <= 0.6
    _gate_hi = 0.78*_gate_hi + 0.22*(1.0 if hi_target>thr_hi else 0.0)

    if _last_t is None: _last_t = t
    dt = max(0.0, min(0.05, t - _last_t)); _last_t = t

    return _env_lo, _env_mid, _env_hi, _gate_hi, rising, dt

# ---- city config ----
GRID_X = 9
GRID_Y = 7
MAX_H = 8.0           # max blocks high
BASE_H = 1.0
H_PUMP = 6.0          # how much lows affect height
WIN_RAND_RATE = 0.15  # window flicker responsiveness
FIRE_RATE = 0.5       # chance per strong hit to spawn fireworks somewhere
BEACON_DECAY = 0.85

# ---- state ----
_windows = {}         # (x,y) -> window bits seed
_beacons = {}         # (x,y) -> intensity 0..1
_fireworks = []       # list of dicts with particle info

def _iso_params(w,h):
    # Fit city to screen using width & height constraints with generous fill ratios.
    fill_w = 0.94
    fill_h = 0.92
    # Height scale relative to tile height (smaller -> shorter buildings -> larger width fit)
    ZH_TILE_MULT = 0.8

    # From width: city_w = (GX+GY) * (tile_w * 0.5) <= w * fill_w
    tw_by_width = (2.0 * w * fill_w) / max(1.0, (GRID_X + GRID_Y))

    # From height: city_h = (GX+GY)*(tile_h*0.5) + MAX_H*zh
    # tile_h = tile_w*0.5 ; zh = tile_h * ZH_TILE_MULT = tile_w * 0.5 * ZH_TILE_MULT
    factor = (GRID_X + GRID_Y)/4.0 + (MAX_H * ZH_TILE_MULT / 2.0)
    tw_by_height = (h * fill_h) / max(1e-6, factor)

    tile_w = min(tw_by_width, tw_by_height)
    tile_h = tile_w * 0.5
    z_h    = tile_h * ZH_TILE_MULT

    # Center
    city_w = (GRID_X + GRID_Y) * (tile_w * 0.5)
    city_h = (GRID_X + GRID_Y) * (tile_h * 0.5) + MAX_H * z_h

    ox = (w - city_w) / 2.0
    oy = (h - city_h) / 2.0 + z_h * 0.6
    return tile_w, tile_h, z_h, ox, oy

def _iso_project(x,y,z, tw,th,zh, ox,oy):
    # isometric projection (x right, y down-left, z up)
    px = ox + (x - y) * (tw*0.5)
    py = oy + (x + y) * (th*0.5) - z*zh
    return QPointF(px, py)

def _building_faces(x,y,h, tw,th,zh, ox,oy):
    # returns polygons for top, left, right faces (QPolygonF)
    # corners of top face at height h
    A = _iso_project(x, y, h, tw,th,zh,ox,oy)
    B = _iso_project(x+1, y, h, tw,th,zh,ox,oy)
    C = _iso_project(x+1, y+1, h, tw,th,zh,ox,oy)
    D = _iso_project(x, y+1, h, tw,th,zh,ox,oy)
    # bottom rim at h=0 to make sides
    A0 = _iso_project(x, y, 0, tw,th,zh,ox,oy)
    B0 = _iso_project(x+1, y, 0, tw,th,zh,ox,oy)
    C0 = _iso_project(x+1, y+1, 0, tw,th,zh,ox,oy)
    D0 = _iso_project(x, y+1, 0, tw,th,zh,ox,oy)
    top = QPolygonF([A,B,C,D])
    right = QPolygonF([B,B0,C0,C])
    left  = QPolygonF([A,A0,D0,D])
    return top, left, right, A

def _window_grid_rects(face_poly, cols=3, rows=4):
    # create small rectangles inside the left/right faces bounding box
    br = face_poly.boundingRect()
    rects = []
    if br.width() < 6 or br.height() < 6: return rects
    cw = br.width() / (cols+1)
    ch = br.height() / (rows+1)
    for i in range(1, cols+1):
        for j in range(1, rows+1):
            rects.append(QRectF(br.x()+i*cw - cw*0.25, br.y()+j*ch - ch*0.25, cw*0.5, ch*0.5))
    return rects

def _spawn_firework(px, py, hi):
    # a compact radial burst
    count = 10 + int(14*min(1.0, hi*1.4))
    for i in range(count):
        ang = (i/count) * 2*pi + (random()*0.2)
        speed = 70 + 160*random()* (0.6 + 0.4*hi)
        _fireworks.append({
            'x': px, 'y': py,
            'vx': speed*cos(ang), 'vy': speed*sin(ang)-30,
            'life': 0.8 + random()*0.6,
            'age': 0.0
        })

@register_visualizer
class SkylineIsoEqualizer(BaseVisualizer):
    display_name = "Skyline Equalizer (Isometric)"
    def paint(self, p:QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        lo, mid, hi, gate_hi, rising, dt = _features(bands, rms, t)

        # params and background
        p.fillRect(r, QBrush(QColor(8,10,14)))
        tw,th,zh,ox,oy = _iso_params(w,h)

        # draw ground grid (subtle)
        p.setPen(QPen(QColor(40,60,90,80), 1))
        for gx in range(0, GRID_X+GRID_Y+1, 2):
            a = _iso_project(gx, 0, 0, tw,th,zh,ox,oy)
            b = _iso_project(0, gx, 0, tw,th,zh,ox,oy)
            p.drawLine(a,b)

        # compute height pump from lows
        pump = BASE_H + H_PUMP * min(1.0, max(0.0, lo))
        hue_base = int((t*20 + 240*mid) % 360)

        # draw buildings back-to-front for isometric depth (x+y order)
        order = [(x,y) for x in range(GRID_X) for y in range(GRID_Y)]
        order.sort(key=lambda xy: xy[0]+xy[1], reverse=True)

        # mids affect windows; refresh occasionally
        for (x,y) in order:
            if (x,y) not in _windows:
                _windows[(x,y)] = _rng.randint(0, 2**16-1)

        # update beacons decay
        for k in list(_beacons.keys()):
            _beacons[k] *= BEACON_DECAY
            if _beacons[k] < 0.02:
                del _beacons[k]

        # fireworks update
        alive = []
        for fw in _fireworks:
            fw['age'] += dt
            if fw['age'] < fw['life']:
                # simple physics
                fw['vy'] += 100*dt
                fw['x'] += fw['vx']*dt
                fw['y'] += fw['vy']*dt
                alive.append(fw)
        _fireworks[:] = alive

        # on strong high transient, spawn fireworks at a random tall building
        if rising and random() < FIRE_RATE:
            rx, ry = choice(order)
            height = pump
            px = _iso_project(rx+0.5, ry+0.5, height+0.1, tw,th,zh,ox,oy)
            _spawn_firework(px.x(), px.y(), hi)
            _beacons[(rx,ry)] = 1.0

        # draw each building
        for (x,y) in order:
            # height
            hval = pump * (0.7 + 0.3*( (x+y) % 3 )/2.0)
            hval = max(0.3, min(MAX_H, hval))

            top, left, right, top_center = _building_faces(x,y,hval, tw,th,zh,ox,oy)

            # colors (slight hue shift per tile)
            hue = (hue_base + (x*13 + y*23)) % 360
            col_top = QColor.fromHsv(hue, 80, 220, 255)
            col_left = QColor.fromHsv((hue+20)%360, 110, 180, 255)
            col_right= QColor.fromHsv((hue+35)%360, 130, 200, 255)

            # faces
            p.setPen(QPen(QColor(0,0,0,140), 1))
            p.setBrush(QBrush(col_left));  p.drawPolygon(left)
            p.setBrush(QBrush(col_right)); p.drawPolygon(right)
            p.setBrush(QBrush(col_top));   p.drawPolygon(top)

            # windows flicker (mids)
            if mid>0.02:
                seed = _windows[(x,y)]
                # update some bits based on mid
                if random() < WIN_RAND_RATE * min(1.0, mid*2.5):
                    seed ^= 1 << randint(0,15)
                    _windows[(x,y)] = seed
                # draw small rects lit by seed bits
                rects_L = _window_grid_rects(left, cols=3, rows=4)
                rects_R = _window_grid_rects(right, cols=3, rows=4)
                bit = seed
                on_col = QColor.fromHsv((hue+60)%360, 40, int(180+70*mid), 230)
                off_col= QColor(0,0,0,0)
                # keep current pen
                for i,rc in enumerate(rects_L + rects_R):
                    p.setBrush(QBrush(on_col if (bit>> (i%16)) & 1 else off_col))
                    p.drawRect(rc)

            # beacons (highs)
            if (x,y) in _beacons:
                inten = _beacons[(x,y)] * (0.6 + 0.6*hi)
                rad = 3 + 8*inten
                p.setBrush(QBrush(QColor.fromHsv((hue+90)%360, 10, 255, int(120*inten+40))))
                p.setPen(QPen(QColor(255,255,255,120), 1))
                p.drawEllipse(top_center, rad, rad)

        # fireworks render
        for fw in _fireworks:
            life = max(0.0, 1.0 - fw['age']/fw['life'])
            a = int(255*life)
            hue = int((hue_base + 180*fw['age']) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 200, 255, min(255,a)), 2))
            p.drawPoint(QPointF(fw['x'], fw['y']))
