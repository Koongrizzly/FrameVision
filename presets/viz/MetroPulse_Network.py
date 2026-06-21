
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont
import random, math
from helpers.music import register_visualizer, BaseVisualizer

# --- MetroPulse Network ---
# A neon subway map. Several colored lines span the screen.
# - Each line 'owns' some frequency bands; it thickens with energy.
# - On each beat, glowing pulses (trains) spawn and travel along the lines.
# - Pulses move faster on strong hits and fade out as they travel.
# - Subtle color drift keeps it alive without being noisy.

# ======= Audio helpers =======
_prev = []
_env=_gate=_punch=0.0
_pt=None
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
        _prev = list(bands)
        return 0.0
    s=0.0
    for a,b in zip(_prev,bands):
        d=b-a
        if d>0: s+=d
    _prev = list(bands)
    return s/(len(bands)+1e-6)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_f_short,_f_long
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    target = 0.50*e + 1.00*f + 0.10*rms + 0.15*lo
    target = target/(1+1.25*target)
    if target>_env: _env=0.70*_env+0.30*target
    else:           _env=0.45*_env+0.55*target
    hi,lo_thr=0.24,0.14
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    _gate=0.70*_gate+0.30*g
    # robust onset (bass-biased dual EMA)
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

# ======= Geometry helpers =======
# We procedurally generate 'routes' across the screen. Each is a smooth polyline.
_routes = []          # each: {pts:[QPointF...], hue:int, bands:(a,b), pulses:[{t,spd,life}]}
_last_t = None
_routes_w = 0
_routes_h = 0

def _compress_bands(bands, target):
    if not bands: return [0.0]*target
    m=len(bands)
    if m==target: return list(bands)
    out=[]
    for i in range(target):
        a=int(i*m/target); b=int((i+1)*m/target); b=max(b,a+1)
        seg=bands[a:b]
        out.append(sum(seg)/float(len(seg)))
    return out

def _gen_routes(w,h, n_routes=6, points_per=140):
    global _routes, _routes_w, _routes_h
    _routes = []
    _routes_w = w
    _routes_h = h
    rnd = random.Random(1337)
    # frequency ownership slices across the (compressed) 24 bands
    bands_per = max(1, 24//n_routes)
    for r in range(n_routes):
        hue = int((r*360.0/n_routes)) % 360
        pts=[]
        margin = int(min(w,h)*0.08)
        for k in range(points_per):
            x = margin + (w-2*margin) * (k/(points_per-1.0))
            y0 = h*0.30 + (r/(n_routes-1.0))*h*0.40
            y = y0 + 0.10*h*math.sin( (x*0.004) + r*0.7 ) + 0.05*h*math.sin( (x*0.010) + r*1.8 )
            pts.append(QPointF(x,y))
        bands=(r*bands_per, min(24,(r+1)*bands_per))
        _routes.append({"pts":pts, "hue":hue, "bands":bands, "pulses":[]})

def _ensure_routes(w,h):
    global _routes_w, _routes_h
    if (not _routes or len(_routes[0]["pts"]) == 0 or
        abs(w - _routes_w) > 4 or abs(h - _routes_h) > 4):
        _gen_routes(w,h)

def _path_len(pts):
    L=0.0
    for i in range(1,len(pts)):
        dx=pts[i].x()-pts[i-1].x(); dy=pts[i].y()-pts[i-1].y()
        L+=math.hypot(dx,dy)
    return max(1.0, L)

def _point_on_path(pts, t):
    # t in [0,1] along the polyline
    target = t * _path_len(pts)
    L=0.0
    for i in range(1,len(pts)):
        ax,ay=pts[i-1].x(), pts[i-1].y()
        bx,by=pts[i].x(), pts[i].y()
        seg=math.hypot(bx-ax,by-ay)
        if L+seg>=target:
            u=(target-L)/seg if seg>1e-6 else 0.0
            x=ax+(bx-ax)*u; y=ay+(by-ay)*u
            return x,y
        L+=seg
    return pts[-1].x(), pts[-1].y()

@register_visualizer
class MetroPulseNetwork(BaseVisualizer):
    display_name = "MetroPulse Network"

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,12)))

        # dt for pulse motion
        if _last_t is None: _last_t=t
        dt=max(0.0, min(0.06, t-_last_t)); _last_t=t

        _ensure_routes(w,h)

        # analyze audio (compress to 24 for cleaner grouping)
        cb = _compress_bands(bands or [0.0]*24, 24)
        env, gate, boom, punch = beat_drive(cb, rms or 0.0, t)

        # slight hue drift + beat nudge
        hue_drift = (int((t*12) % 360) + int(90*punch)) % 360

        # Spawn pulses on strong beats
        if punch > 0.60:
            for route in _routes:
                if random.random()<0.5 and len(route["pulses"])<10:
                    route["pulses"].append({"t":0.0, "spd": random.uniform(0.20,0.45)*(1.0+0.8*punch), "life":1.0})

        # Grid
        p.setPen(QPen(QColor(20,20,24,140), 1))
        step=max(40, int(min(w,h)*0.06))
        for x in range(step, w, step): p.drawLine(x, 0, x, h)
        for y in range(step, h, step): p.drawLine(0, y, w, y)

        # Draw routes
        p.setRenderHint(QPainter.Antialiasing, True)
        for ri, route in enumerate(_routes):
            a,b = route["bands"]
            energy = sum(cb[a:b])/max(1,(b-a))
            thickness = 2.0 + 10.0*energy + 6.0*punch
            hue = (route["hue"] + hue_drift) % 360
            line_col = QColor.fromHsv(hue, 200, 230, 255)
            glow_col = QColor.fromHsv(hue, 180, 180, 90)

            # Glow
            p.setPen(QPen(glow_col, thickness*2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for i in range(1,len(route["pts"])):
                p.drawLine(route["pts"][i-1], route["pts"][i])
            # Line
            p.setPen(QPen(line_col, thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for i in range(1,len(route["pts"])):
                p.drawLine(route["pts"][i-1], route["pts"][i])

            # Pulses
            new_p=[]
            for pl in route["pulses"]:
                pl["t"] += pl["spd"]*dt
                pl["life"] *= pow(0.92, dt/0.016)
                if pl["t"]>1.0 or pl["life"]<0.05:
                    continue
                x,y = _point_on_path(route["pts"], pl["t"]) 
                rad = 5 + 10*punch
                alpha = int(200*pl["life"]) 
                glow = QColor.fromHsv(hue, 150, 255, max(60, alpha//2))
                dot  = QColor.fromHsv(hue, 180, 255, alpha)
                p.setBrush(QBrush(glow)); p.setPen(QPen(QColor(0,0,0,0),1))
                p.drawEllipse(QPointF(x,y), rad*1.8, rad*1.8)
                p.setBrush(QBrush(dot)); p.setPen(QPen(QColor(0,0,0,0),1))
                p.drawEllipse(QPointF(x,y), rad, rad)
                new_p.append(pl)
            route["pulses"] = new_p
