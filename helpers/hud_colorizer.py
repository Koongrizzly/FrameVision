# helpers/hud_colorizer.py — colorize % and temperatures (+highlight & hysteresis)
from __future__ import annotations
import re
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel

YELLOW = "#e0c600"
ORANGE = "#d98a00"
RED    = "#d70022"
BG_RED = "#ffe6e6"  # subtle background for ≥70°C (with hysteresis)

# Percent patterns (one each)
_PAT_GPU = re.compile(r"(GPU\s*:.*?\s)(\d{1,3})%", re.IGNORECASE)
_PAT_DDR = re.compile(r"(DDR\s*.*?\s)(\d{1,3})%", re.IGNORECASE)
_PAT_CPU = re.compile(r"(CPU\s*)(\d{1,3})%", re.IGNORECASE)

# Temperature tokens like '62°C' or '62 C' or '62c'
_PAT_TEMP = re.compile(r"""
    (?<![\dA-Za-z])        # not part of a larger token
    (\d{2,3})              # temperature value (2–3 digits)
    \s*                    # optional space
    (?:°\s*)?              # optional degree symbol
    ([cC])                  # 'C' or 'c'
    (?![\dA-Za-z])         # not followed by alnum
""", re.VERBOSE)

def _wrap_percent(num_str: str) -> str:
    try:
        v = int(num_str)
    except Exception:
        return num_str + "%"
    if v >= 95:
        color = RED
    elif v >= 85:
        color = ORANGE
    else:
        return num_str + "%"
    return f"<span style='color:{color};font-weight:600'>{v}%</span>"

def _class_for_temp(value: int, in_red_hys: bool) -> str | None:
    # Return color hex or None. Applies hysteresis only for red band.
    if in_red_hys and value >= 68:
        return RED
    if value >= 70:
        return RED
    if value >= 65:
        return ORANGE
    if value >= 60:
        return YELLOW
    return None

def _wrap_temp_token(full_token: str, value_str: str, in_red_hys: bool) -> str:
    """Wrap a temperature token (e.g., '65°C') based on thresholds.
    Hysteresis: once any temp has reached ≥70°C, we stay in a 'red state' until all temps drop below 68°C.
    In red state, values ≥68°C render red; otherwise thresholds are:
      <60  : no change
      60-64: yellow
      65-69: orange
      ≥70  : red
    """
    try:
        v = int(value_str)
    except Exception:
        return full_token
    color = _class_for_temp(v, in_red_hys)
    if not color:
        return full_token
    return f"<span style='color:{color};font-weight:600'>{full_token}</span>"

def _colorize_percents(text: str) -> str:
    s = text
    for pat in (_PAT_GPU, _PAT_DDR, _PAT_CPU):
        def repl(m):
            prefix, num = m.group(1), m.group(2)
            return prefix + _wrap_percent(num)
        s = pat.sub(repl, s, count=1)  # one token each
    return s

def _colorize_temps(text: str, in_red_hys: bool) -> str:
    def repl(m: re.Match) -> str:
        value = m.group(1)
        return _wrap_temp_token(m.group(0), value, in_red_hys)
    return _PAT_TEMP.sub(repl, text)

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)

def _find_header_label() -> QLabel | None:
    app = QApplication.instance()
    if not app: return None
    # Find a QLabel whose text contains GPU, DDR and CPU (your header usually has all three)
    for w in app.allWidgets():
        if isinstance(w, QLabel):
            try:
                t = _strip_tags(w.text())
                if "GPU" in t and "DDR" in t and "CPU" in t:
                    return w
            except Exception:
                continue
    return None

def _parse_all_temps(plain: str) -> list[int]:
    vals = []
    for m in _PAT_TEMP.finditer(plain):
        try:
            vals.append(int(m.group(1)))
        except Exception:
            pass
    return vals

def auto_install_hud_colorizer(interval_ms: int = 900):
    lbl = _find_header_label()
    if not lbl:
        QTimer.singleShot(600, lambda: auto_install_hud_colorizer(interval_ms))
        return
    if getattr(lbl, "_fv_hud_colorizer_active", False):
        return
    lbl._fv_hud_colorizer_active = True
    lbl.setTextFormat(Qt.RichText)

    # Persistent state on the label for hysteresis + change detection
    state = {"in_red": False, "last_plain": None}
    lbl._fv_hud_colorizer_state = state  # for debugging

    timer = QTimer(lbl)
    def tick():
        try:
            plain = _strip_tags(lbl.text())
            if plain == state.get("last_plain") and not state.get("force_next", False):
                return
            temps = _parse_all_temps(plain)
            max_t = max(temps) if temps else None

            # Hysteresis for red band: enter at ≥70, leave only when max < 68
            next_in_red = False
            if max_t is not None:
                if state["in_red"]:
                    next_in_red = max_t >= 68
                else:
                    next_in_red = max_t >= 70

            # Colorize: temps first (with hysteresis), then percents
            colored = _colorize_temps(plain, in_red_hys=next_in_red)
            colored = _colorize_percents(colored)

            # Background highlight if in (or staying in) red zone
            if next_in_red:
                colored = f"<span style='background:{BG_RED};border-radius:4px;padding:0 3px'>{colored}</span>"

            if colored != lbl.text():
                lbl.setText(colored)

            # Update state
            state["in_red"] = next_in_red
            state["last_plain"] = plain
            state["force_next"] = False
        except Exception:
            pass

    timer.setInterval(interval_ms)
    timer.timeout.connect(tick)
    timer.start()
    # run once immediately
    state["force_next"] = True
    tick()
