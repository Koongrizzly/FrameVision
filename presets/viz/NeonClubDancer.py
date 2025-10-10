# NeonClubDancer.py — neon-rim silhouette, lasers & floor glow (beat-only)
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# ======== Config ========
FREEZE_WHEN_IDLE   = True    # do not move if no beats
IDLE_TO_CENTER_SEC = 2.0     # seconds since last beat to treat as idle
IDLE_ENV_THR       = 0.04    # gate noise floor
HL_TREB            = 0.12    # half-life (s) for treble pops (arms/hair/lasers)
HL_BASS            = 0.22    # half-life (s) for bass pops (hips/steps/floor)
HL_SWAY            = 0.35    # skirt/hair inertia

# ======== State ========
_prev_spec = []
_last_t = None
_onset_avg = 0.0
_onset_peak = 1e-3
_gate = 0.0
_since_on = 10.0

_pop_treb = 0.0   # high / flux controlled
_pop_bass = 0.0   # low controlled
_arm_dir  = 1     # alternate arm lead
_step_dir = 1     # alternate leg lead
_sway     = 0.0   # skirt/hair lag state
_hue      = 210.0 # neon hue base (deg)

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if (not _prev_spec) or (len(_prev_spec) != len(bands)):
        _prev_spec = [0.0]*len(bands)
    f = 0.0; n=len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x-px
        if d>0:
            f += d*(0.35 + 0.65*(i/max(1,n-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f/max(1,n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _onset(bands, rms, dt):
    global _onset_avg, _onset_peak, _gate, _since_on
    lo,mid,hi = _split(bands)
    fx = _flux(bands)
    env = 0.5*rms + 0.35*lo + 0.15*mid
    onset = 0.6*hi + 1.5*fx + 0.2*rms
    _onset_avg  = 0.98*_onset_avg  + 0.02*onset
    _onset_peak = max(_onset_peak*pow(0.5, dt/0.6), onset)
    denom = max(1e-3, _onset_peak - 0.6*_onset_avg)
    norm = max(0.0, (onset - 0.8*_onset_avg)/denom)
    beat = (norm>0.55) and (_gate<=0.5)
    _gate = 0.78*_gate + 0.22*(1.0 if norm>0.55 else 0.0)
    if beat: _since_on = 0.0
    return lo,hi,fx,env,beat

def _decay(x, dt, hl):
    return x * pow(0.5, dt/max(1e-6, hl))

def _rot(x,y,a):
    from math import cos, sin
    ca,sa = cos(a), sin(a)
    return x*ca - y*sa, x*sa + y*ca

def _neon(h, s=230, v=255, a=220):
    return QColor.fromHsv(int(h)%360, max(0,min(255,int(s))), max(0,min(255,int(v))), max(0,min(255,int(a))))

@register_visualizer
class NeonClubDancer(BaseVisualizer):
    display_name = "Neon Club Dancer"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _pop_treb, _pop_bass, _arm_dir, _step_dir, _sway, _hue
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t
        _since_on += dt

        lo,hi,fx,env,beat = _onset(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on>IDLE_TO_CENTER_SEC) and (env<IDLE_ENV_THR))

        if beat and not idle:
            if lo > 0.08:
                _pop_bass = 1.0; _step_dir *= -1; _sway += 0.4*_step_dir
            if hi + fx > lo*0.6:
                _pop_treb = 1.0; _arm_dir  *= -1; _sway += 0.2*_arm_dir
            _hue = (_hue + 22.0) % 360.0

        if idle:
            _pop_treb = 0.0; _pop_bass = 0.0; _sway *= 0.0
        else:
            _pop_treb = _decay(_pop_treb, dt, HL_TREB)
            _pop_bass = _decay(_pop_bass, dt, HL_BASS)
            _sway     = _decay(_sway, dt, HL_SWAY)

        # background
        p.fillRect(r, QBrush(QColor(6,8,12)))
        cx,cy = w/2, int(h*0.80); S = min(w,h)
        bass_glow = int(60 + 180*_pop_bass)
        p.setBrush(QBrush(_neon(_hue+180, 200, 110, bass_glow)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), S*0.22*(1.0+0.15*_pop_bass), S*0.05*(1.0+0.15*_pop_bass))

        # rear lasers
        if _pop_treb>0.02:
            beams = 7; hue0 = _hue
            for i in range(beams):
                ang = (-0.5 + i/(beams-1))*0.9
                length = S*(0.42 + 0.18*_pop_treb)
                x2 = cx + length * cos(ang)
                y2 = cy - S*0.48 + length * sin(ang)
                c = _neon(hue0 + i*12, 255, 255, int(80 + 140*_pop_treb))
                p.setPen(QPen(c, max(2, int(S*0.006*(1.0+0.5*_pop_treb)))))
                p.drawLine(QPointF(cx, cy-S*0.48), QPointF(x2, y2))

        # dancer
        hip_x = cx; hip_y = cy - S*0.32
        p.save()
        p.translate(hip_x, hip_y); p.rotate((_sway*0.10)*180/pi); p.translate(-hip_x, -hip_y)

        rim  = _neon(_hue, 255, 255, 220)
        fill = QColor(12,14,18)

        waist_w   = S*0.10; shoulder_w= S*0.22
        hem_w     = S*(0.24 + 0.10*abs(_sway))
        hem_drop  = S*(0.28 + 0.03*_pop_bass)

        torso = QPainterPath(QPointF(hip_x - waist_w/2, hip_y))
        torso.lineTo(hip_x + waist_w/2, hip_y)
        torso.lineTo(hip_x + shoulder_w/2, hip_y - S*0.22 + _pop_treb*S*0.02*_arm_dir)
        torso.lineTo(hip_x - shoulder_w/2, hip_y - S*0.22 - _pop_treb*S*0.02*_arm_dir)
        torso.closeSubpath()

        skirt = QPainterPath()
        skirt.moveTo(hip_x - waist_w/2, hip_y)
        skirt.lineTo(hip_x + waist_w/2, hip_y)
        skirt.lineTo(hip_x + hem_w/2 + _sway*S*0.05, hip_y + hem_drop)
        skirt.lineTo(hip_x - hem_w/2 + _sway*S*0.05, hip_y + hem_drop)
        skirt.closeSubpath()

        p.setBrush(QBrush(fill)); p.setPen(Qt.NoPen); p.drawPath(skirt); p.drawPath(torso)
        for wmult, alpha in ((1.0,220),(2.0,120),(3.2,60)):
            p.setPen(QPen(_neon(_hue, 255, 255, alpha), int(S*0.006*wmult)))
            p.setBrush(Qt.NoBrush); p.drawPath(torso); p.drawPath(skirt)

        head_r = S*0.052*(1.0 + 0.25*_pop_treb)
        hx = hip_x; hy = hip_y - S*0.29 - _pop_treb*S*0.02
        p.setBrush(QBrush(fill)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(hx, hy), head_r, head_r)
        for wmult, alpha in ((1.0,220),(2.0,120)):
            p.setPen(QPen(_neon(_hue+8, 255, 255, alpha), int(S*0.006*wmult)))
            p.drawEllipse(QPointF(hx, hy), head_r, head_r)

        tail = QPainterPath(QPointF(hx, hy - head_r*0.1))
        swing = _sway + 0.7*_pop_treb*_arm_dir
        tail.cubicTo(QPointF(hx + S*0.04*swing, hy - head_r*0.8),
                     QPointF(hx + S*0.06*swing, hy - head_r*1.4),
                     QPointF(hx + S*0.02*swing, hy - head_r*1.8))
        p.setPen(QPen(_neon(_hue+12, 255, 255, 180), int(S*0.008))); p.setBrush(Qt.NoBrush); p.drawPath(tail)

        def limb(pivot_x, pivot_y, a1, l1, a2, l2, thick=8):
            from math import cos, sin
            x1,y1 = 0, -l1
            ca,sa = cos(a1), sin(a1); x1,y1 = x1*ca - y1*sa, x1*sa + y1*ca
            x2,y2 = 0, -l2
            ca2,sa2 = cos(a1+a2), sin(a1+a2); x2,y2 = x2*ca2 - y2*sa2, x2*sa2 + y2*ca2
            p.setPen(QPen(fill, thick)); p.drawLine(QPointF(pivot_x, pivot_y), QPointF(pivot_x+x1, pivot_y+y1))
            p.drawLine(QPointF(pivot_x+x1, pivot_y+y1), QPointF(pivot_x+x1+x2, pivot_y+y1+y2))
            p.setPen(QPen(rim, thick*0.6)); p.drawLine(QPointF(pivot_x, pivot_y), QPointF(pivot_x+x1, pivot_y+y1))
            p.drawLine(QPointF(pivot_x+x1, pivot_y+y1), QPointF(pivot_x+x1+x2, pivot_y+y1+y2))
            return pivot_x+x1+x2, pivot_y+y1+y2

        Lsx, Lsy = hip_x - shoulder_w/2, hip_y - S*0.22 - _pop_treb*S*0.02*_arm_dir
        Rsx, Rsy = hip_x + shoulder_w/2, hip_y - S*0.22 + _pop_treb*S*0.02*_arm_dir
        base = -0.25; flick = 0.7*_pop_treb
        limb(Lsx, Lsy, base - 0.3*_arm_dir, S*0.12,  0.6 + (flick if _arm_dir<0 else 0.0), S*0.13, int(S*0.014))
        limb(Rsx, Rsy, base + 0.3*_arm_dir, S*0.12, -0.6 - (flick if _arm_dir>0 else 0.0), S*0.13, int(S*0.014))

        Lhx, Rhx = hip_x - S*0.045, hip_x + S*0.045
        thigh = S*0.14; shin = S*0.16; lift = 0.32*_pop_bass
        limb(Lhx, hip_y,  0.25 + (lift if _step_dir<0 else 0.0), thigh, -0.40 - lift*0.6, shin, int(S*0.016))
        limb(Rhx, hip_y, -0.25 + (lift if _step_dir>0 else 0.0), thigh,  0.40 + lift*0.6, shin, int(S*0.016))

        p.restore()
