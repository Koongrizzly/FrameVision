
# helpers/hud_colorizer.py — sticky 1s clock; strips any existing time/date before appending one clock
from __future__ import annotations
import re, time
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel

# Colors (kept if you want to re-enable styling later)
YELLOW = "#e0c600"; ORANGE = "#d98a00"; RED = "#d70022"; BG_RED = "#ffe6e6"

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

# Accepts a wide range of time/date stamps that other writers might append:
# - " | 12:05:38"           (standard)
# - " | 12:05.38"           (locale with dot before seconds)
# - " Sun. 28 Sep 12:05:36" (weekday + date + time)
# - "Sun 28 Sep 12:05.36"   (no separators, dot seconds)
# We strip these (and the adjacent separator) before adding our own HH:MM:SS.
_PAT_TRAIL_CLOCK = re.compile(
    r"""
    (?:\s*                # optional space
       [\|\-•]\s*         # common separator before a clock
    )?
    (?:                   # a trailing time/date token group
       (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?\s+   # optional weekday
    )?
    (?:\d{1,2}\s+[A-Za-z]{3}\s+)?              # optional '28 Sep'
    \d{1,2}:\d{2}[:\.]\d{2}                    # 12:05:36 or 12:05.36
    \s*$                                       # to the end
    """,
    re.VERBOSE
)

def _strip_any_trailing_time(s: str) -> str:
    # Remove multiple possible trailing time/date tokens (if someone appended twice)
    prev = None
    while prev != s:
        prev = s
        s = _PAT_TRAIL_CLOCK.sub("", s).rstrip()
    return s

def _now_clock() -> str:
    return time.strftime("%H:%M:%S")

def _merge_clock(base_plain: str) -> str:
    clean = _strip_any_trailing_time(base_plain)
    return f"{clean} | {_now_clock()}" if clean else _now_clock()

def _find_header_label() -> Optional[QLabel]:
    app = QApplication.instance()
    if not app: return None
    for w in app.allWidgets():
        try:
            if isinstance(w, QLabel):
                t = _strip_tags(w.text() or "")
                if "GPU" in t and "CPU" in t:
                    return w
        except Exception: pass
    return None

def auto_install_hud_colorizer(interval_ms: int = 1000):
    lbl = _find_header_label()
    if not lbl:
        QTimer.singleShot(600, lambda: auto_install_hud_colorizer(interval_ms))
        return
    if getattr(lbl, "_fv_hud_colorizer_active", False):
        return

    lbl._fv_hud_colorizer_active = True
    lbl.setTextFormat(Qt.RichText)

    state = {"base_plain": _strip_tags(lbl.text() or "")}
    orig_setText = lbl.setText

    # Intercept any external writer and re-append a single clock immediately
    def sticky_setText(new_text: str):
        base = _strip_tags(new_text or "")
        state["base_plain"] = base
        try:
            orig_setText(_merge_clock(base))
        except Exception:
            orig_setText(new_text)

    try:
        lbl.setText = sticky_setText  # type: ignore
        lbl._fv_setText_wrapped = True
    except Exception:
        pass

    # 1s precise timer to tick the clock; uses orig_setText to avoid recursion
    timer = QTimer(lbl)
    timer.setTimerType(Qt.PreciseTimer)
    timer.setInterval(1000)
    timer.timeout.connect(lambda: orig_setText(_merge_clock(state["base_plain"])))
    timer.start()

    # immediate apply
    orig_setText(_merge_clock(state["base_plain"]))
