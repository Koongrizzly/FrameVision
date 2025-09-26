from math import sin, pi
from PySide6.QtGui import QPainter, QPen, QColor
from helpers.music import register_visualizer, BaseVisualizer

# Mid/high energy + spectral-flux envelope for stronger, snappier response
_prev = []
_env = 0.0

def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    acc = 0.0; cnt = 0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        acc += w*bands[i]; cnt += 1
    return acc/max(1, cnt)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = 0.0; cnt = 0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d * (0.3 + 0.7*((i-cut)/max(1, n-cut)))
        cnt += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, cnt)

def _env_drive(bands, rms):
    global _env
    target = 0.55*_midhi(bands) + 1.30*_flux(bands) + 0.20*rms
    target = target / (1.0 + 0.7*target)  # soft compression
    if target > _env:
        _env = 0.70*_env + 0.30*target   # fast attack
    else:
        _env = 0.92*_env + 0.08*target   # slower release
    if _env < 0: _env = 0.0
    if _env > 1.0: _env = 1.0
    return _env

@register_visualizer
class NeonLissajous(BaseVisualizer):
    display_name = "Neon Lissajous XY"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = r.width(), r.height()
        if w <= 0 or h <= 0: return
        cx, cy = r.center().x(), r.center().y()

        env = _env_drive(bands, rms)

        # Frequencies unchanged, but path amplitude & width scale with env
        a = 2*pi*(0.5 + 0.2*rms)
        b = 2*pi*(1.0 + 0.3*rms)
        ax = (w*0.35) * (0.9 + 0.7*env)
        ay = (h*0.35) * (0.9 + 0.7*env)
        width = 2 + int(3*env)

        N = 900
        prev = None
        base_hue = (int(t*80) % 360)
        for i in range(N):
            ph = i/N*2*pi
            x = cx + ax*sin(a*ph + t*(0.9 + 1.6*env))
            y = cy + ay*sin(b*ph + t*(0.8 + 1.4*env))
            # Color cycles across the curve + to the beat
            hue = (base_hue + int(360*i/N)) % 360
            val = 230
            p.setPen(QPen(QColor.fromHsv(hue, 230, val, 200), width))
            if prev: p.drawLine(prev[0], prev[1], x, y)
            prev = (x, y)
