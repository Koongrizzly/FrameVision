# helpers/hud_colorizer.py — colorize % and temperatures (flashless ≥70°C)
from __future__ import annotations
import re
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel

YELLOW = "#e0c600"
ORANGE = "#d98a00"
RED    = "#d70022"
# NOTE: Removed background highlight and hysteresis-based 'red alert' to stop any flashing at ≥70°C.

# Percent patterns (one each)
_PAT_GPU = re.compile(r"(GPU\s*:.*?\s)(\d{1,3})%", re.IGNORECASE)
_PAT_DDR = re.compile(r"(DDR\s*.*?\s)(\d{1,3})%", re.IGNORECASE)
_PAT_CPU = re.compile(r"(CPU\s*)(\d{1,3})%", re.IGNORECASE)

# VRAM amount pattern like 'DDR 5.3/24.0 GB'
_PAT_VRAM_AMOUNT = re.compile(
    r"(DDR\s*:?\s*)(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(\s*[GMK]i?B)",
    re.IGNORECASE,
)

# Temperature tokens like '62°C' or '62 C' or '62c'
_PAT_TEMP = re.compile(r"""    (?<![\dA-Za-z])        # not part of a larger token
    (\d{2,3})              # temperature value (2–3 digits)
    \s*                    # optional space
    (?:°\s*)?              # optional degree symbol
    ([cC])                  # 'C' or 'c'
    (?![\dA-Za-z])         # not followed by alnum
""", re.VERBOSE)

## Net speed tokens like 'DL 123 KB/s', 'UL 456 KB/s', or 'DL 12.3 MB/s'
_PAT_NET = re.compile(r"((?:DL|UL)\s*)(\d+(?:\.\d+)?)\s*(KB/s|MB/s)", re.IGNORECASE)

def _net_color_for_mbs(mbs: float) -> str | None:
    # Thresholds:
    #   >= 10 MB/s : yellow
    #   >= 50 MB/s : orange
    #   >= 95 MB/s : red
    try:
        m = float(mbs)
    except Exception:
        m = 0.0
    if m >= 95.0:
        return RED
    if m >= 50.0:
        return ORANGE
    if m >= 10.0:
        return YELLOW
    return None

def _colorize_net_speeds(text: str) -> str:
    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        num = m.group(2)
        unit = (m.group(3) or "KB/s").strip()
        try:
            v = float(num)
        except Exception:
            return m.group(0)
        # Convert to MB/s for thresholding (KB/s is treated as 1024-based).
        if unit.lower().startswith("mb"):
            mbs = v
        else:
            mbs = v / 1024.0
        color = _net_color_for_mbs(mbs)
        if not color:
            return m.group(0)
        tok = f"{num} {unit}"
        return prefix + f"<span style='color:{color};font-weight:600'>{tok}</span>"
    return _PAT_NET.sub(repl, text)


def _wrap_percent(num_str: str) -> str:
    """Generic percent wrapper (GPU / CPU), orange ≥85%, red ≥95%."""
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


def _vram_color_for_percent(pct: float) -> str | None:
    """Color thresholds specifically for VRAM fullness.

    When VRAM is:
      - ≥85% and <90% : yellow
      - ≥90% and <95% : orange
      - ≥95%          : red
    """
    if pct >= 95:
        return RED
    if pct >= 90:
        return ORANGE
    if pct >= 85:
        return YELLOW
    return None


def _wrap_vram_percent(num_str: str) -> str:
    """Percent wrapper for VRAM usage, using VRAM thresholds (85/90/95)."""
    try:
        v = int(num_str)
    except Exception:
        return num_str + "%"
    color = _vram_color_for_percent(float(v))
    if not color:
        return num_str + "%"
    return f"<span style='color:{color};font-weight:600'>{v}%</span>"


def _wrap_vram_amount(used_str: str, total_str: str, unit_tail: str) -> str:
    """Wraps the VRAM 'used/total' amount based on how full it is.

    Example matched text (no HTML tags here):
        'DDR 5.3/24.0 GB 22%'

    We compute 5.3 / 24.0 ≈ 22% and apply:
      - yellow when ≥85%
      - orange when ≥90%
      - red when ≥95%
    """
    try:
        used = float(used_str)
        total = float(total_str)
        if total <= 0:
            raise ValueError
        pct = used / total * 100.0
    except Exception:
        return f"{used_str}/{total_str}{unit_tail}"

    color = _vram_color_for_percent(pct)
    if not color:
        return f"{used_str}/{total_str}{unit_tail}"

    return (
        f"<span style='color:{color};font-weight:600'>"
        f"{used_str}/{total_str}{unit_tail}"
        f"</span>"
    )


def _class_for_temp(value: int, in_red_hys: bool) -> str | None:
    # Hysteresis no longer used; argument kept for signature compatibility.
    if value >= 70:
        return RED
    if value >= 65:
        return ORANGE
    if value >= 60:
        return YELLOW
    return None


def _wrap_temp_token(full_token: str, value_str: str, in_red_hys: bool) -> str:
    """Wrap a temperature token (e.g., '65°C') based on thresholds.
    Flashless behavior: no background highlight, no hysteresis-based state.
    Thresholds:
      <60  : no change
      60-64: yellow
      65-69: orange
      ≥70  : red (text only)
    """
    try:
        v = int(value_str)
    except Exception:
        return full_token
    color = _class_for_temp(v, in_red_hys)
    if not color:
        return full_token
    return f"<span style='color:{color};font-weight:600'>{full_token}</span>"


def _colorize_vram_amount(text: str) -> str:
    """Colorize the VRAM 'used/total' amount based on fullness.

    Matches sequences like:
        'DDR 5.3/24.0 GB'
        'DDR: 7.8/8.0GiB'
    and wraps the whole numeric portion with a colored span when ≥85% full.
    """
    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        used = m.group(2)
        total = m.group(3)
        tail = m.group(4) or ""
        wrapped = _wrap_vram_amount(used, total, tail)
        return prefix + wrapped

    # Only one DDR block expected, so count=1 is enough
    return _PAT_VRAM_AMOUNT.sub(repl, text, count=1)


def _colorize_percents(text: str) -> str:
    """Colorize GPU/DDR/CPU percentage tokens.

    - GPU and CPU use the generic _wrap_percent thresholds.
    - DDR (VRAM) uses VRAM-specific 85/90/95 thresholds.
    """
    s = text

    def repl_gpu(m: re.Match) -> str:
        prefix, num = m.group(1), m.group(2)
        return prefix + _wrap_percent(num)

    def repl_ddr(m: re.Match) -> str:
        prefix, num = m.group(1), m.group(2)
        return prefix + _wrap_vram_percent(num)

    def repl_cpu(m: re.Match) -> str:
        prefix, num = m.group(1), m.group(2)
        return prefix + _wrap_percent(num)

    s = _PAT_GPU.sub(repl_gpu, s, count=1)
    s = _PAT_DDR.sub(repl_ddr, s, count=1)
    s = _PAT_CPU.sub(repl_cpu, s, count=1)
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
    if not app:
        return None
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


def auto_install_hud_colorizer(interval_ms: int = 2500):
    lbl = _find_header_label()
    if not lbl:
        QTimer.singleShot(600, lambda: auto_install_hud_colorizer(interval_ms))
        return
    if getattr(lbl, "_fv_hud_colorizer_active", False):
        return
    lbl._fv_hud_colorizer_active = True
    lbl.setTextFormat(Qt.RichText)

    # Persistent state on the label for change detection (no hysteresis)
    state = {"last_plain": None}
    lbl._fv_hud_colorizer_state = state  # for debugging

    timer = QTimer(lbl)

    def tick():
        try:
            plain = _strip_tags(lbl.text())
            if plain == state.get("last_plain") and not state.get("force_next", False):
                return

            # Colorize: temps (no hysteresis), VRAM amount, then percents
            colored = _colorize_temps(plain, in_red_hys=False)
            colored = _colorize_vram_amount(colored)
            colored = _colorize_percents(colored)

            colored = _colorize_net_speeds(colored)

            # No background highlight; text-only colorization to avoid flashing.
            if colored != lbl.text():
                lbl.setText(colored)

            # Update state
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
