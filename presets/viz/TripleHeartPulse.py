
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Beat envelope (mid/high + spectral flux) ---
_prev = []
_env = 0.0
_gate = 0.0

def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = 0.0; c = 0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))  # favor highs
        s += bands[i]*w; c += 1
    return s/max(1, c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = 0.0; c = 0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d * (0.3 + 0.7*((i-cut)/max(1, n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, c)

def _env_drive(bands, rms):
    global _env, _gate
    e = _midhi(bands); f = _flux(bands)
    target = 0.55*e + 1.30*f + 0.20*rms
    target = target / (1.0 + 0.7*target)  # soft compression
    # attack/release
    if target > _env: _env = 0.68*_env + 0.32*target
    else: _env = 0.92*_env + 0.08*target
    # onset gate (binary-ish) for extra punch
    hi, lo = 0.30, 0.18
    g = 1.0 if f > hi else (0.0 if f < lo else _gate)
    _gate = 0.80*_gate + 0.20*g
    # clamp
    if _env < 0: _env = 0.0
    if _env > 1: _env = 1.0
    if _gate < 0: _gate = 0.0
    if _gate > 1: _gate = 1.0
    return _env, _gate

def _heart_path(cx, cy, s):
    # Make a heart at (cx,cy) with size s (roughly half-width).
    path = QPainterPath()
    x0, y0 = cx, cy - 0.3*s
    path.moveTo(x0, y0)
    path.cubicTo(cx - 0.5*s, cy - 0.9*s, cx - 1.2*s, cy + 0.2*s, cx, cy + 0.9*s)
    path.cubicTo(cx + 1.2*s, cy + 0.2*s, cx + 0.5*s, cy - 0.9*s, x0, y0)
    return path

@register_visualizer
class TripleHeartPulse(BaseVisualizer):
    display_name = "Triple Heart Pulse"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0: return

        # background
        p.fillRect(r, QBrush(QColor(8, 8, 14)))

        env, gate = _env_drive(bands, rms)

        # layout positions (left, center, right)
        cx = r.center().x()
        cy = r.center().y()
        spacing = min(w, h) * 0.22
        positions = (cx - spacing, cx, cx + spacing)

        # base size and scale with beat (outer and center all scale)
        base_s = min(w, h) * 0.075
        scale = 0.95 + 1.35*(env**1.15) + (0.25 if gate > 0.6 else 0.0)

        # colors
        red_fill = QColor(255, 70, 110, 180)
        red_stroke = QColor(255, 160, 190, 230)

        # center heart hue cycles; outer hearts stay red
        hue_center = int((t*90 + 200*env) % 360)
        center_fill = QColor.fromHsv(hue_center, 220, 255, 200)
        center_stroke = QColor.fromHsv(hue_center, 220, 255, 230)

        # draw hearts with soft glow
        for i, x in enumerate(positions):
            s = base_s * scale
            if i == 1:  # center — color changing
                fill, stroke = center_fill, center_stroke
            else:       # outer — fixed red
                fill, stroke = red_fill, red_stroke

            # background glow
            g = QRadialGradient(QPointF(x, cy), s*2.2)
            c0 = QColor(fill.red(), fill.green(), fill.blue(), 120)
            c1 = QColor(fill.red(), fill.green(), fill.blue(), 0)
            g.setColorAt(0.0, c0); g.setColorAt(1.0, c1)
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,10), 1))
            p.drawEllipse(QPointF(x, cy), s*1.3, s*1.3)

            # heart
            path = _heart_path(x, cy, s)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(stroke, 3))
            p.drawPath(path)

        # tiny sparkle on strong peaks over center heart
        if gate > 0.65:
            s = base_s * scale
            for k in range(10):
                ang = 2*pi*k/10 + t*1.2
                L = s*0.9
                col = QColor.fromHsv((hue_center + k*12) % 360, 220, 255, 200)
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(cx, cy), QPointF(cx + L*cos(ang), cy + L*sin(ang)))
