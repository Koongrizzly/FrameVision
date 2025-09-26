
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Beat envelope (mid/high + spectral flux) ---
_prev = []
_env = 0.0
_gate = 0.0

# extra: onset-driven bounce impulse
_pulse = 0.0
_last_gate = 0.0

# tuning constants (feel free to tweak)
BOUNCE_FREQ_BASE = 10.0   # Hz-like feel (scaled by time)
IMPULSE_ADD = 1.10        # amount of bounce added per onset
PULSE_DECAY = 0.90        # 0..1, closer to 1 = longer bounce

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
    global _env, _gate, _pulse, _last_gate
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

    # onset detection (rising edge) -> add impulse
    if _gate > 0.60 and _last_gate <= 0.60:
        _pulse += IMPULSE_ADD
        if _pulse > 2.5: _pulse = 2.5
    _last_gate = _gate

    # decay pulse every frame
    _pulse *= PULSE_DECAY

    # clamp
    if _env < 0: _env = 0.0
    if _env > 1: _env = 1.0
    if _gate < 0: _gate = 0.0
    if _gate > 1: _gate = 1.0
    return _env, _gate, _pulse

def _heart_path(cx, cy, s):
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

        env, gate, pulse = _env_drive(bands, rms)

        # layout positions (left, center, right)
        cx = r.center().x()
        cy = r.center().y()
        spacing = min(w, h) * 0.22
        positions = (cx - spacing, cx, cx + spacing)

        # base size
        base_s = min(w, h) * 0.075

        # hard bounce scaler: clearly smaller AND bigger
        freq = BOUNCE_FREQ_BASE * (1.0 + 0.3*env + 0.6*pulse)
        osc = sin(t * freq)
        ampl = 0.55 + 0.65*env + 0.95*pulse   # amplitude of bounce
        scale = 1.0 + ampl * osc              # oscillate around 1.0
        if scale < 0.42: scale = 0.42         # clamp min so it doesn't invert

        # colors
        red_fill = QColor(255, 70, 110, 200)
        red_stroke = QColor(255, 160, 190, 240)

        # center heart hue cycles; outer hearts stay red
        hue_center = int((t*120 + 240*env + 60*pulse) % 360)
        center_fill = QColor.fromHsv(hue_center, 230, 255, 220)
        center_stroke = QColor.fromHsv(hue_center, 230, 255, 245)

        # draw hearts with stronger glow
        for i, x in enumerate(positions):
            s = base_s * scale
            if i == 1:  # center — color changing
                fill, stroke = center_fill, center_stroke
            else:       # outer — fixed red
                fill, stroke = red_fill, red_stroke

            # background glow
            g = QRadialGradient(QPointF(x, cy), s*2.6)
            c0 = QColor(fill.red(), fill.green(), fill.blue(), 150)
            c1 = QColor(fill.red(), fill.green(), fill.blue(), 0)
            g.setColorAt(0.0, c0); g.setColorAt(1.0, c1)
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,10), 1))
            p.drawEllipse(QPointF(x, cy), s*1.4, s*1.4)

            # heart
            path = _heart_path(x, cy, s)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(stroke, 3))
            p.drawPath(path)

        # starburst on very strong peaks
        if pulse > 0.8:
            s = base_s * scale
            for k in range(12):
                ang = 2*pi*k/12 + t*1.5
                L = s
                col = QColor.fromHsv((hue_center + k*12) % 360, 230, 255, 220)
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(cx, cy), QPointF(cx + L*cos(ang), cy + L*sin(ang)))
