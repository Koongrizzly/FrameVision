
from __future__ import annotations
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

# Day palette
DAY_BG = "#F2F2F2"
DAY_GROUP_BG = "#FFFFFF"
DAY_INPUT_BG = "#DBDBDB"
DAY_TAB_BG = "#CDEEFF"
DAY_TEXT = "#1A1A1A"
DAY_BORDER_SOFT = "#E7E7E7"
DAY_BORDER_SOFTER = "#EFEFEF"
DAY_SLIDER = "#66FF66"

# "Evening" palette — Gradio-like dark with BLUE in place of the usual orange
# This becomes the new default Evening theme requested by the user
EVE_BG = "#111827"          # slate-900-ish window background
EVE_GROUP_BG = "#0f172a"     # slightly darker panels
EVE_HEADER = "#1d4ed8"       # BLUE accent instead of orange
EVE_TEXT = "#E5E7EB"         # light gray text for contrast
EVE_BORDER_SOFT = "#334155"   # slate-600 border
EVE_BORDER_SOFTER = "#475569" # slate-500 border/hover
EVE_TAB_BG = "#111827"       # tabs in line with bg
EVE_TAB_HOVER = "#0b1220"
EVE_BTN_PRESSED = "#172554"
EVE_SLIDER = "#38bdf8"       # cyan-ish handle for visibility

# Night palette
NIGHT_BG = "#000000"
NIGHT_GROUP_BG = "#0A0A0A"
NIGHT_HEADER = "#00D6F7"
NIGHT_TEXT = "#EAEFF5"
NIGHT_BORDER = "#1F2A3A"
NIGHT_TAB_BG = "#121829"
NIGHT_TAB_HOVER = "#192136"
NIGHT_BTN_HOVER = "#182034"
NIGHT_SLIDER = "#00F708"

QSS_DAY = f"""
QAbstractScrollArea {{ background: {DAY_BG}; }}
QScrollArea QWidget#qt_scrollarea_viewport {{ background: {DAY_BG}; }}
QWidget {{ background: {DAY_BG}; color: {DAY_TEXT}; }}
QMainWindow, QDialog {{ background: {DAY_BG}; }}

QGroupBox {{ background: {DAY_GROUP_BG}; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {DAY_TAB_BG}; color: {DAY_TEXT}; border-radius: 6px; font-weight: 600; }}

QTabWidget::pane {{ border: 1px solid {DAY_BORDER_SOFT}; top: -1px; background: {DAY_GROUP_BG}; }}
QTabBar::tab {{ background: {DAY_TAB_BG}; padding: 7px 14px; border: 1px solid {DAY_BORDER_SOFT}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: #D8F0FF; }}
QTabBar::tab:selected {{ background: {DAY_GROUP_BG}; color: {DAY_TEXT}; font-weight: 600; }}

QPushButton {{ background: {DAY_TAB_BG}; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; padding: 6px 12px; }}
QPushButton:hover {{ background: #EAF6FF; }}
QPushButton:pressed {{ background: #DAECFB; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: {DAY_INPUT_BG}; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; padding: 5px 8px; color: {DAY_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}

/* Checkboxes/radios: transparent rows, no loud borders */
QCheckBox, QRadioButton {{ background: transparent; border: none; }}

QSlider::groove:horizontal {{ height: 6px; background: {DAY_BORDER_SOFT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: {DAY_SLIDER}; border: 1px solid #6bdc6b; border-radius: 8px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {DAY_TAB_BG}; border: 1px solid {DAY_BORDER_SOFTER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {DAY_BORDER_SOFT}; border-radius: 6px; }}
"""

QSS_EVENING = f"""
QWidget {{ background: {EVE_BG}; color: {EVE_TEXT}; }}
QMainWindow, QDialog {{ background: {EVE_BG}; }}

QGroupBox {{ background: {EVE_GROUP_BG}; border: 1px solid {EVE_BORDER_SOFT}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {EVE_HEADER}; color: #E5E7EB; border-radius: 6px; font-weight: 700; }}

QTabWidget::pane {{ border: 1px solid {EVE_BORDER_SOFT}; top: -1px; background: {EVE_GROUP_BG}; }}
QTabBar::tab {{ background: {EVE_TAB_BG}; padding: 7px 14px; border: 1px solid {EVE_BORDER_SOFT}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: {EVE_TAB_HOVER}; }}
QTabBar::tab:selected {{ background: {EVE_GROUP_BG}; color: {EVE_TEXT}; font-weight: 700; }}

QPushButton {{ background: {EVE_TAB_BG}; border: 1px solid {EVE_BORDER_SOFT}; border-radius: 8px; padding: 6px 12px; color: {EVE_TEXT}; }}
QPushButton:hover {{ background: {EVE_TAB_HOVER}; }}
QPushButton:pressed {{ background: {EVE_BTN_PRESSED}; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: {EVE_GROUP_BG}; border: 1px solid {EVE_BORDER_SOFT}; border-radius: 8px; padding: 5px 8px; color: {EVE_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}

QCheckBox, QRadioButton {{ background: transparent; border: none; color: {EVE_TEXT}; }}

QSlider::groove:horizontal {{ height: 6px; background: {EVE_BORDER_SOFTER}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: {EVE_SLIDER}; border: 1px solid #079a2a; border-radius: 8px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {EVE_TAB_BG}; border: 1px solid {EVE_BORDER_SOFTER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {EVE_BORDER_SOFT}; border-radius: 6px; }}
"""

QSS_NIGHT = f"""
QWidget {{ background: {NIGHT_BG}; color: {NIGHT_TEXT}; }}
QMainWindow, QDialog {{ background: {NIGHT_BG}; }}

QGroupBox {{ background: {NIGHT_GROUP_BG}; border: 1px solid {NIGHT_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {NIGHT_HEADER}; color: #003349; border-radius: 6px; font-weight: 800; }}

QTabWidget::pane {{ border: 1px solid {NIGHT_BORDER}; top: -1px; background: {NIGHT_GROUP_BG}; }}
QTabBar::tab {{ background: {NIGHT_TAB_BG}; padding: 7px 14px; border: 1px solid {NIGHT_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; color: #C8D1DA; }}
QTabBar::tab:hover {{ background: {NIGHT_TAB_HOVER}; }}
QTabBar::tab:selected {{ background: {NIGHT_GROUP_BG}; color: {NIGHT_TEXT}; font-weight: 800; }}

QPushButton {{ background: {NIGHT_TAB_BG}; border: 1px solid {NIGHT_BORDER}; border-radius: 8px; padding: 6px 12px; color: #C8D1DA; }}
QPushButton:hover {{ background: {NIGHT_BTN_HOVER}; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: #0B1018; border: 1px solid {NIGHT_BORDER}; border-radius: 8px; padding: 5px 8px; color: {NIGHT_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}

QCheckBox, QRadioButton {{ background: transparent; border: none; color: {NIGHT_TEXT}; }}

QSlider::groove:horizontal {{ height: 6px; background: {NIGHT_BORDER}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: {NIGHT_SLIDER}; border: 1px solid #06b607; border-radius: 8px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {NIGHT_TAB_BG}; border: 1px solid {NIGHT_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {NIGHT_BORDER}; border-radius: 6px; }}
"""

def qss_for_theme(name: str) -> str:
    # Normalize
    raw = (name or "Evening")
    s = raw.strip().lower()
    # replace non-alphanum with single space, collapse
    import re as _re
    s = _re.sub(r"[^a-z0-9]+", " ", s).strip()

    # canonical keys map
    _MAP = {
        "day": QSS_DAY,
        "evening": QSS_EVENING,
        "night": QSS_NIGHT,
        "slate": QSS_SLATE,
        "high contrast": QSS_HIGH_CONTRAST,
        "highcontrast": QSS_HIGH_CONTRAST,
        "contrast": QSS_HIGH_CONTRAST,
        "cyberpunk": QSS_CYBERPUNK,
        "neon": QSS_NEON,
        "ocean": QSS_OCEAN,
        "solarized light": QSS_SOLARIZED_LIGHT,
        "solarized": QSS_SOLARIZED_LIGHT,
        "crt": QSS_CRT,
        "tropical fiesta": QSS_TROPICAL_FIESTA,
        "tropical": QSS_TROPICAL_FIESTA,
        "fiesta": QSS_TROPICAL_FIESTA,
        "color mix": QSS_COLOR_MIX,
        "colormix": QSS_COLOR_MIX,
        "aurora": QSS_AURORA,
        "mardi gras": QSS_MARDI_GRAS,
        "mardi grass": QSS_MARDI_GRAS,
        "sunburst": QSS_SUNBURST,
        "candy pop": QSS_CANDY_POP,
        "rainbow riot": QSS_RAINBOW_RIOT,
        "pastel light": QSS_PASTEL_LIGHT,
        "pastel": QSS_PASTEL_LIGHT,
        "mgraphite dusk": QSS_GRAPHITE_DUSK,
        "graphite dusk": QSS_GRAPHITE_DUSK,
        "graphite": QSS_GRAPHITE_DUSK,
        "cloud grey": QSS_CLOUD_GREY,
        "cloud gray": QSS_CLOUD_GREY,
        "signal grey": QSS_SIGNAL_GREY,
        "signal gray": QSS_SIGNAL_GREY,
    }

    # Exact mapping first
    if s in _MAP:
        return _MAP[s]

    # Prefix helpers (for friendly matching like "daylight", "evening glow", etc.)
    if s.startswith("day"): return QSS_DAY
    if s.startswith("even"): return QSS_EVENING
    if s.startswith("night"): return QSS_NIGHT
    if s.startswith("slate"): return QSS_SLATE
    if s.startswith("high") or "contrast" in s: return QSS_HIGH_CONTRAST
    if s.startswith("cyber"): return QSS_CYBERPUNK
    if s.startswith("neon"): return QSS_NEON
    if s.startswith("ocean"): return QSS_OCEAN
    if s.startswith("solar"): return QSS_SOLARIZED_LIGHT
    if s.startswith("crt"): return QSS_CRT
    if "tropic" in s or "fiesta" in s: return QSS_TROPICAL_FIESTA
    if "color" in s and "mix" in s: return QSS_COLOR_MIX
    if "aurora" in s: return QSS_AURORA
    if "mardi" in s: return QSS_MARDI_GRAS
    if "sunburst" in s: return QSS_SUNBURST
    if "pastel" in s: return QSS_PASTEL_LIGHT
    if "cloud" in s: return QSS_CLOUD_GREY
    if "signal" in s: return QSS_SIGNAL_GREY
    if "graphite" in s or s.endswith("dusk"): return QSS_GRAPHITE_DUSK

    # Fallback
    return QSS_EVENING

def _menu_qss(name: str) -> str:
    """
    Extra QSS that ensures visible hover/pressed backgrounds for menus and menubars,
    tuned per theme. Keeps day themes light; makes dark themes lighter on hover;
    gives colorful themes strong, on-brand hover tints; and sets Cyberpunk to violet.
    """
    s = (name or "").strip().lower()

    def rule(hover_bg: str, pressed_bg: str, panel_bg: str | None = None, separator: str | None = None) -> str:
        parts = []
        if panel_bg:
            parts.append(f"QMenu {{ background: {panel_bg}; }}")
        if separator:
            parts.append(f"QMenu::separator {{ height: 1px; background: {separator}; margin: 6px 8px; }}")
        parts.append(f"QMenu::item:selected {{ background: {hover_bg}; }}")
        parts.append(f"QMenu::item:pressed {{ background: {pressed_bg}; }}")
        parts.append(f"QMenuBar::item:selected {{ background: {hover_bg}; }}")
        parts.append(f"QMenuBar::item:pressed {{ background: {pressed_bg}; }}")
        return "\n".join(parts) + "\n"

    # Colorful themes (explicit request)
    if 'color mix' in s or 'colormix' in s:
        return rule('#22c55e', '#16a34a')  # green
    if 'candy pop' in s:
        return rule('#f97316', '#ea580c')  # orange
    if 'rainbow riot' in s:
        return rule('#3b82f6', '#2563eb')  # blue
    if s.startswith('cyber'):
        return rule('#6d28d9', '#5b21b6')  # violet

    # Day-like themes — keep light and subtle
    if s.startswith('day'):
        return rule('#EAF6FF', '#D8F0FF')
    if s.startswith('solar'):
        return rule('#e6dfc9', '#e0d7bd')
    if 'pastel' in s:
        return rule('#F3E8FF', '#E9D5FF')  # soft lavender
    if 'sunburst' in s:
        return rule('#FFF3C4', '#FFE8A1')  # pale yellow
    if 'cloud grey' in s or 'cloud gray' in s:
        return rule('#E6F0FF', '#DDE6FF')
    if 'signal grey' in s or 'signal gray' in s or s.startswith('signal'):
        return rule('#FFE8D1', '#E2F0FF')

    # Dark-ish themes — provide lighter, clearer hover states
    # Evening, Night, Slate, Graphite Dusk, High Contrast, Neon, Ocean, CRT, Aurora, Mardi Gras, Tropical Fiesta
    return rule('#1f2937', '#172554')




def _button_hover_qss(name: str) -> str:
    """
    Extra QSS that gives QPushButton (and key toolbuttons) a clear, theme-aware
    hover/pressed outline so buttons always feel "alive" when you move the mouse
    over them. Neon/color-bomb themes that already define strong button hover
    effects keep their original styling (this helper returns an empty string).
    """
    s = (name or "").strip().lower()

    # Skip themes that already define flashy hover borders in their own QSS
    if any(key in s for key in ("cyber", "rainbow riot", "candy pop", "candypop", "color mix", "colormix")):
        return ""

    # --- Light / day-like themes ------------------------------------------------------
    if s.startswith("day"):
        return f"""
QPushButton:hover {{
    border-color: #3b82f6;
    color: {DAY_TEXT};
}}
QPushButton:pressed {{
    border-color: #2563eb;
}}
"""

    if "pastel" in s:
        return """
QPushButton:hover {
    border-color: #e879f9;
    color: #374151;
}
QPushButton:pressed {
    border-color: #d946ef;
}
"""

    if "sunburst" in s:
        return f"""
QPushButton:hover {{
    border-color: {SUN_ACCENT};
    color: {SUN_TEXT};
}}
QPushButton:pressed {{
    border-color: {SUN_SLIDER};
}}
"""

    if "solar" in s:
        return """
QPushButton:hover {
    border-color: #268bd2;
    color: #073642;
}
QPushButton:pressed {
    border-color: #1c6fa8;
}
"""

    if "cloud" in s:
        return f"""
QPushButton:hover {{
    border-color: {CLOUD_ACCENT};
    color: {CLOUD_TEXT};
}}
QPushButton:pressed {{
    border-color: {CLOUD_SLIDER};
}}
"""

    if "signal" in s:
        return f"""
QPushButton:hover {{
    border-color: {SIG_ACCENT_ORANGE};
    color: {SIG_TEXT};
}}
QPushButton:pressed {{
    border-color: {SIG_ACCENT_BLUE};
}}
"""

    # --- Evening / neutral darks ------------------------------------------------------
    if s.startswith("even"):
        return f"""
QPushButton:hover {{
    border-color: {EVE_SLIDER};
    color: {EVE_TEXT};
}}
QPushButton:pressed {{
    border-color: {EVE_HEADER};
}}
"""

    if s.startswith("night"):
        return f"""
QPushButton:hover {{
    border-color: {NIGHT_HEADER};
    color: {NIGHT_TEXT};
}}
QPushButton:pressed {{
    border-color: {NIGHT_SLIDER};
}}
"""

    if s.startswith("slate"):
        return f"""
QPushButton:hover {{
    border-color: {SLATE_ACCENT};
    color: {SLATE_TEXT};
}}
QPushButton:pressed {{
    border-color: {SLATE_ACCENT};
}}
"""

    if s.startswith("high") or "contrast" in s:
        return f"""
QPushButton:hover {{
    border-color: {HC_ACCENT};
    color: {HC_TEXT};
}}
QPushButton:pressed {{
    border-color: {HC_ACCENT};
}}
"""

    if "graphite" in s or "dusk" in s:
        return f"""
QPushButton:hover {{
    border-color: {GRA_ACCENT};
    color: {GRA_TEXT};
}}
QPushButton:pressed {{
    border-color: {GRA_ACCENT};
}}
"""

    # --- Other dark / colorful themes ------------------------------------------------
    if s.startswith("neon"):
        return """
QPushButton:hover {
    border-color: #22c55e;
    color: #eaffea;
}
QPushButton:pressed {
    border-color: #16a34a;
}
"""

    if s.startswith("ocean"):
        return """
QPushButton:hover {
    border-color: #14b8a6;
    color: #e6f7ff;
}
QPushButton:pressed {
    border-color: #0ea5a7;
}
"""

    if s.startswith("crt"):
        return """
QPushButton:hover {
    border-color: #16a34a;
    color: #9aff9a;
}
QPushButton:pressed {
    border-color: #22c55e;
}
"""

    if "tropic" in s or "fiesta" in s:
        return """
QPushButton:hover, QToolButton:hover {
    border-color: #22d3ee;
    color: #e8fff4;
}
QPushButton:pressed, QToolButton:pressed {
    border-color: #f59e0b;
}
"""

    if "aurora" in s:
        return f"""
QPushButton:hover {{
    border-color: {AURORA_ACCENT};
    color: {AURORA_TEXT};
}}
QPushButton:pressed {{
    border-color: {AURORA_ACCENT};
}}
"""

    if "mardi" in s:
        return f"""
QPushButton:hover {{
    border-color: {MARDI_ACCENT};
    color: {MARDI_TEXT};
}}
QPushButton:pressed {{
    border-color: {MARDI_ACCENT};
}}
"""

    # Fallback: use the evening accent as a sensible default
    return f"""
QPushButton:hover {{
    border-color: {EVE_SLIDER};
    color: {EVE_TEXT};
}}
QPushButton:pressed {{
    border-color: {EVE_HEADER};
}}
"""


def apply_theme(app: QApplication, name: str) -> None:
    if (name or "").strip().lower() == "auto":
        from .framevision_app import pick_auto_theme
        try:
            name = pick_auto_theme()
        except Exception:
            name = "Evening"
    # Handle special theme selectors before building QSS
    s = (name or "").strip().lower()
    if s == "random":
        try:
            import random as _rand
            _pool = ["Day","Solarized Light","Sunburst","Cloud Grey","Signal Grey","Evening","Night","Slate","High Contrast","Cyberpunk","Neon","Ocean","CRT","Aurora","Mardi Gras","Tropical Fiesta","Color Mix"]
            name = _rand.choice(_pool)
        except Exception:
            name = "Evening"
    elif s == "auto":
        from .framevision_app import pick_auto_theme
        try:
            name = pick_auto_theme()
        except Exception:
            name = "Evening"
    qss = qss_for_theme(name) + _menu_qss(name) + _button_hover_qss(name)
    try:
        app.setStyleSheet(qss)
        pal = app.palette()
        if name.lower().startswith("night"):
            pal.setColor(QPalette.Window, QColor(NIGHT_BG))
            pal.setColor(QPalette.Base, QColor(NIGHT_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(NIGHT_TAB_BG))
            pal.setColor(QPalette.Text, Qt.white)
            pal.setColor(QPalette.WindowText, Qt.white)
        elif (name.lower().startswith("day") or "solar" in name.lower() or "sunburst" in name.lower() or "sky" in name.lower() or "pastel" in name.lower() or "cloud" in name.lower() or "signal" in name.lower()):
            pal.setColor(QPalette.Window, QColor(DAY_BG))
            pal.setColor(QPalette.Base, QColor(DAY_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(DAY_TAB_BG))
            pal.setColor(QPalette.Text, QColor(DAY_TEXT))
            pal.setColor(QPalette.WindowText, QColor(DAY_TEXT))
        elif ("graphite" in name.lower() or "dusk" in name.lower()):
            pal.setColor(QPalette.Window, QColor(GRA_BG))
            pal.setColor(QPalette.Base, QColor(GRA_GROUP_BG))
            pal.setColor(QPalette.Button, QColor("#1C2024"))
            pal.setColor(QPalette.Text, QColor(GRA_TEXT))
            pal.setColor(QPalette.WindowText, QColor(GRA_TEXT))
        else:
            pal.setColor(QPalette.Window, QColor(EVE_BG))
            pal.setColor(QPalette.Base, QColor(EVE_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(EVE_TAB_BG))
            pal.setColor(QPalette.Text, QColor(EVE_TEXT))
            pal.setColor(QPalette.WindowText, QColor(EVE_TEXT))
        app.setPalette(pal)

        # Fast theme refresh: avoid re-polishing every widget (can freeze the UI on large apps).
        # Updating top-level widgets is typically enough; Qt will propagate style changes to children.
        for w in app.topLevelWidgets():
            try:
                w.style().unpolish(w); w.style().polish(w); w.update()
            except Exception:
                pass
    except Exception:
        pass



# --- Cloud Grey (mid-light neutral greys with cool blue accent) ---------------------------
CLOUD_BG = "#DDE0E5"
CLOUD_GROUP_BG = "#ECEEF1"
CLOUD_TEXT = "#1A1A1A"
CLOUD_BORDER = "#C6CBD4"
CLOUD_BORDER_SOFT = "#D2D6DE"
CLOUD_TAB_BG = "#E6EBF5"
CLOUD_TAB_HOVER = "#DDE6FF"
CLOUD_SLIDER = "#7FD6FF"
CLOUD_ACCENT = "#AFCBFF"

QSS_CLOUD_GREY = f"""
QAbstractScrollArea {{ background: {CLOUD_BG}; }}
QScrollArea QWidget#qt_scrollarea_viewport {{ background: {CLOUD_BG}; }}
QWidget {{ background: {CLOUD_BG}; color: {CLOUD_TEXT}; }}
QMainWindow, QDialog {{ background: {CLOUD_BG}; }}

QGroupBox {{
    background: {CLOUD_GROUP_BG};
    border: 1px solid {CLOUD_BORDER_SOFT};
    border-radius: 8px;
    margin-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 3px 8px;
    background: {CLOUD_ACCENT};
    color: {CLOUD_TEXT};
    border-radius: 6px;
    font-weight: 600;
}}

QTabWidget::pane {{
    border: 1px solid {CLOUD_BORDER_SOFT};
    top: -1px;
    background: {CLOUD_GROUP_BG};
}}
QTabBar::tab {{
    background: {CLOUD_TAB_BG};
    padding: 7px 14px;
    border: 1px solid {CLOUD_BORDER_SOFT};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: {CLOUD_TEXT};
}}
QTabBar::tab:hover {{ background: {CLOUD_TAB_HOVER}; }}
QTabBar::tab:selected {{
    background: {CLOUD_GROUP_BG};
    color: {CLOUD_TEXT};
    font-weight: 700;
}}

QPushButton {{
    background: {CLOUD_TAB_BG};
    border: 1px solid {CLOUD_BORDER_SOFT};
    border-radius: 8px;
    padding: 6px 12px;
    color: {CLOUD_TEXT};
}}
QPushButton:hover {{ background: {CLOUD_TAB_HOVER}; }}
QPushButton:pressed {{ background: #D1EBFF; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: #D7DADF;
    border: 1px solid {CLOUD_BORDER_SOFT};
    border-radius: 8px;
    padding: 5px 8px;
    color: {CLOUD_TEXT};
}}
QComboBox::drop-down {{ width: 24px; }}

QCheckBox, QRadioButton {{
    background: transparent;
    border: none;
    color: {CLOUD_TEXT};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {CLOUD_BORDER_SOFT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    background: {CLOUD_SLIDER};
    border: 1px solid #78c3ff;
    border-radius: 8px;
    margin: -6px 0;
}}

QScrollBar:vertical {{
    width: 12px;
    background: {CLOUD_TAB_BG};
    border: 1px solid {CLOUD_BORDER_SOFT};
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {CLOUD_BORDER};
    border-radius: 6px;
}}
"""


# --- Signal Grey (mid-light greys with orange & blue accents) -----------------------------
SIG_BG = "#CFCFD4"
SIG_GROUP_BG = "#DADADD"
SIG_TEXT = "#1A1A1A"
SIG_BORDER = "#B8B8BD"
SIG_BORDER_SOFT = "#C4C4C8"
SIG_ACCENT_ORANGE = "#FFB766"
SIG_ACCENT_BLUE = "#6CAEFF"
SIG_TAB_BG = "#E2E4EA"
SIG_TAB_HOVER = "#DADDE6"
SIG_SLIDER = "#6CAEFF"

QSS_SIGNAL_GREY = f"""
QAbstractScrollArea {{ background: {SIG_BG}; }}
QScrollArea QWidget#qt_scrollarea_viewport {{ background: {SIG_BG}; }}
QWidget {{ background: {SIG_BG}; color: {SIG_TEXT}; }}
QMainWindow, QDialog {{ background: {SIG_BG}; }}

QGroupBox {{
    background: {SIG_GROUP_BG};
    border: 1px solid {SIG_BORDER_SOFT};
    border-radius: 8px;
    margin-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 3px 8px;
    background: {SIG_ACCENT_ORANGE};
    color: {SIG_TEXT};
    border-radius: 6px;
    font-weight: 600;
}}

QTabWidget::pane {{
    border: 1px solid {SIG_BORDER_SOFT};
    top: -1px;
    background: {SIG_GROUP_BG};
}}
QTabBar::tab {{
    background: {SIG_TAB_BG};
    padding: 7px 14px;
    border: 1px solid {SIG_BORDER_SOFT};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: {SIG_TEXT};
}}
QTabBar::tab:hover {{ background: {SIG_TAB_HOVER}; }}
QTabBar::tab:selected {{
    background: {SIG_ACCENT_BLUE};
    color: #0F1B2E;
    font-weight: 700;
}}

QPushButton {{
    background: {SIG_TAB_BG};
    border: 1px solid {SIG_BORDER_SOFT};
    border-radius: 8px;
    padding: 6px 12px;
    color: {SIG_TEXT};
}}
QPushButton:hover {{ background: {SIG_TAB_HOVER}; }}
QPushButton:pressed {{
    background: {SIG_ACCENT_ORANGE};
    color: #2A1500;
}}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: #D4D4D9;
    border: 1px solid {SIG_BORDER_SOFT};
    border-radius: 8px;
    padding: 5px 8px;
    color: {SIG_TEXT};
}}
QComboBox::drop-down {{ width: 24px; }}

QCheckBox, QRadioButton {{
    background: transparent;
    border: none;
    color: {SIG_TEXT};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {SIG_BORDER_SOFT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    background: {SIG_SLIDER};
    border: 1px solid {SIG_ACCENT_ORANGE};
    border-radius: 8px;
    margin: -6px 0;
}}

QScrollBar:vertical {{
    width: 12px;
    background: {SIG_TAB_BG};
    border: 1px solid {SIG_BORDER_SOFT};
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {SIG_BORDER};
    border-radius: 6px;
}}
"""

# --- Graphite Dusk (cool neutral dusk) ----------------------------------------------------
GRA_BG = "#202428"           # window background (graphite)
GRA_GROUP_BG = "#252A30"     # panels
GRA_TEXT = "#E6E9ED"         # light text
GRA_BORDER = "#343A42"       # borders
GRA_ACCENT = "#8B95A1"       # graphite accent (neutral)

QSS_GRAPHITE_DUSK = f"""
QWidget {{ background: {GRA_BG}; color: {GRA_TEXT}; }}
QMainWindow, QDialog {{ background: {GRA_BG}; }}

QGroupBox {{ background: {GRA_GROUP_BG}; border: 1px solid {GRA_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {GRA_ACCENT}; color: #101215; border-radius: 6px; font-weight: 700; }}

QTabWidget::pane {{ border: 1px solid {GRA_BORDER}; top: -1px; background: {GRA_GROUP_BG}; }}
QTabBar::tab {{ background: #1C2024; padding: 7px 14px; border: 1px solid {GRA_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: #171B1F; }}
QTabBar::tab:selected {{ background: {GRA_GROUP_BG}; color: {GRA_TEXT}; font-weight: 700; }}

QPushButton {{ background: #1C2024; border: 1px solid {GRA_BORDER}; border-radius: 8px; padding: 6px 12px; color: {GRA_TEXT}; }}
QPushButton:hover {{ background: #171B1F; }}
QPushButton:pressed {{ background: #13171B; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: #171B1F; border: 1px solid {GRA_BORDER}; border-radius: 8px; padding: 5px 8px; color: {GRA_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}

QSlider::groove:horizontal {{ height: 6px; background: {GRA_BORDER}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: {GRA_ACCENT}; border: 1px solid #7a828e; border-radius: 8px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: #1C2024; border: 1px solid {GRA_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {GRA_BORDER}; border-radius: 6px; }}
"""

# --- Extra themes to reach 5 total -------------------------------------------------------
# Slate (neutral mid-dark) — good for editors
SLATE_BG = "#2B2F36"
SLATE_GROUP_BG = "#333842"
SLATE_TEXT = "#E8EEF4"
SLATE_BORDER = "#3C4450"
SLATE_ACCENT = "#4F46E5"  # indigo

QSS_SLATE = f"""
QWidget {{ background: {SLATE_BG}; color: {SLATE_TEXT}; }}
QMainWindow, QDialog {{ background: {SLATE_BG}; }}
QGroupBox {{ background: {SLATE_GROUP_BG}; border: 1px solid {SLATE_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {SLATE_ACCENT}; color: #e6eaff; border-radius: 6px; font-weight: 700; }}
QTabWidget::pane {{ border: 1px solid {SLATE_BORDER}; top: -1px; background: {SLATE_GROUP_BG}; }}
QTabBar::tab {{ background: #262a31; padding: 7px 14px; border: 1px solid {SLATE_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: #20242b; }}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4338ca, stop:1 {SLATE_ACCENT});
    color: {SLATE_TEXT}; font-weight: 700; border-color: {SLATE_ACCENT};
}}
QPushButton {{ background: #262a31; border: 1px solid {SLATE_BORDER}; border-radius: 8px; padding: 6px 12px; color: {SLATE_TEXT}; }}
QPushButton:hover {{ background: #20242b; }}
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: #22262d; border: 1px solid {SLATE_BORDER}; border-radius: 8px; padding: 5px 8px; color: {SLATE_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}
QCheckBox, QRadioButton {{ background: transparent; border: none; color: {SLATE_TEXT}; }}
QSlider::groove:horizontal {{ height: 6px; background: {SLATE_BORDER}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: {SLATE_ACCENT}; border: 1px solid #4338ca; border-radius: 8px; margin: -6px 0; }}
QScrollBar:vertical {{ width: 12px; background: #262a31; border: 1px solid {SLATE_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {SLATE_BORDER}; border-radius: 6px; }}
"""

# High Contrast (accessible) — bright text, sharp borders
HC_BG = "#000000"
HC_GROUP_BG = "#0d0d0d"
HC_TEXT = "#FFFFFF"
HC_BORDER = "#7f7f7f"
HC_ACCENT = "#00B0FF"

QSS_HIGH_CONTRAST = f"""
QWidget {{ background: {HC_BG}; color: {HC_TEXT}; }}
QMainWindow, QDialog {{ background: {HC_BG}; }}
QGroupBox {{ background: {HC_GROUP_BG}; border: 2px solid {HC_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {HC_ACCENT}; color: #00111a; border-radius: 6px; font-weight: 800; }}
QTabWidget::pane {{ border: 2px solid {HC_BORDER}; top: -2px; background: {HC_GROUP_BG}; }}
QTabBar::tab {{ background: #0a0a0a; padding: 8px 16px; border: 2px solid {HC_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: #101010; }}
QTabBar::tab:selected {{ background: {HC_GROUP_BG}; color: {HC_TEXT}; font-weight: 900; }}
QPushButton {{ background: #0a0a0a; border: 2px solid {HC_BORDER}; border-radius: 8px; padding: 7px 14px; color: {HC_TEXT}; }}
QPushButton:hover {{ background: #101010; }}
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: #0a0a0a; border: 2px solid {HC_BORDER}; border-radius: 8px; padding: 6px 9px; color: {HC_TEXT}; }}
QComboBox::drop-down {{ width: 28px; }}
QCheckBox, QRadioButton {{ background: transparent; border: none; color: {HC_TEXT}; }}
QSlider::groove:horizontal {{ height: 8px; background: {HC_BORDER}; border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 18px; background: {HC_ACCENT}; border: 2px solid #006b99; border-radius: 9px; margin: -6px 0; }}
QScrollBar:vertical {{ width: 14px; background: #0a0a0a; border: 2px solid {HC_BORDER}; border-radius: 7px; }}
QScrollBar::handle:vertical {{ background: {HC_BORDER}; border-radius: 7px; }}
"""



# Cyberpunk — high-contrast dark, neon cyan + purple (no pink)
# --- Cyberpunk (Fluorescent) ---------------------------------------------------------------
# Deep near-black background to make neon pop
CYBER_BG = "#07010F"
CYBER_GROUP_BG = "#0C0A1A"
CYBER_TEXT = "#EAEAFB"
CYBER_BORDER = "#2B2450"
# Neon accents
CYBER_NEON_PURPLE = "#9D00FF"   # violet / purple
CYBER_NEON_VIOLET = "#B000FF"
CYBER_NEON_BLUE = "#00E5FF"
CYBER_NEON_PINK = "#FF00A8"
CYBER_NEON_YELLOW = "#F8FF00"

QSS_CYBERPUNK = f"""
QWidget {{ background: {CYBER_BG}; color: {CYBER_TEXT}; }}
QMainWindow, QDialog {{ background: {CYBER_BG}; }}

QGroupBox {{ background: {CYBER_GROUP_BG}; border: 1px solid {CYBER_BORDER}; border-radius: 10px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {CYBER_NEON_BLUE}; }}

QTabWidget::pane {{ border: 1px solid {CYBER_BORDER}; top: -1px; background: {CYBER_GROUP_BG}; }}
QTabBar::tab {{
    background: {CYBER_GROUP_BG};
    padding: 8px 16px;
    border: 1px solid {CYBER_BORDER};
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}}
QTabBar::tab:hover {{ background: #120F28; border-color: {CYBER_NEON_PURPLE}; }}
QTabBar::tab:selected {{
    background: #120F28;
    color: {CYBER_NEON_YELLOW};
    border-color: {CYBER_NEON_PINK};
    font-weight: 700;
}}

QPushButton {{
    background: #0E0A22;
    color: {CYBER_TEXT};
    border: 1px solid {CYBER_BORDER};
    border-radius: 10px;
    padding: 8px 12px;
}}
QPushButton:hover {{ border-color: {CYBER_NEON_BLUE}; color: {CYBER_NEON_BLUE}; }}
QPushButton:pressed {{ background: {CYBER_NEON_PURPLE}; color: #0B0011; }}

QLineEdit, QPlainTextEdit, QTextEdit {{
    background: #0E0A22;
    border: 1px solid {CYBER_BORDER};
    border-radius: 8px;
    selection-background-color: {CYBER_NEON_PINK};
    selection-color: #1C0023;
}}
QComboBox {{
    background: #0E0A22;
    border: 1px solid {CYBER_BORDER};
    border-radius: 8px; padding: 6px;
}}
QComboBox QAbstractItemView {{
    background: {CYBER_GROUP_BG};
    color: {CYBER_TEXT};
    selection-background-color: {CYBER_NEON_BLUE};
    selection-color: #00181A;
    border: 1px solid {CYBER_BORDER};
}}

QProgressBar {{
    background: #0E0A22;
    border: 1px solid {CYBER_BORDER};
    border-radius: 10px;
    text-align: center;
    color: {CYBER_TEXT};
}}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {CYBER_NEON_PURPLE}, stop:0.5 {CYBER_NEON_PINK}, stop:1 {CYBER_NEON_BLUE}); border-radius: 10px; }}

QSlider::groove:horizontal {{ height: 6px; background: #0E0A22; border: 1px solid {CYBER_BORDER}; border-radius: 4px; }}
QSlider::handle:horizontal {{
    width: 18px;
    background: {CYBER_NEON_YELLOW};
    border: 1px solid {CYBER_NEON_PINK};
    border-radius: 9px;
    margin: -6px 0;
}}

QScrollBar:vertical {{
    width: 12px; background: #0E0A22; border: 1px solid {CYBER_BORDER}; border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {CYBER_NEON_VIOLET}; border-radius: 6px;
}}
"""



# Neon — dark slate with neon green highlights
QSS_NEON = """
QWidget { background: #0a0f0a; color: #eaffea; }
QMainWindow, QDialog { background: #0a0f0a; }
QGroupBox { background: #0f1a0f; border: 1px solid #1f3a1f; border-radius: 8px; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: #22c55e; color: #07230e; border-radius: 6px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #1f3a1f; top: -1px; background: #0f1a0f; }
QTabBar::tab { background: #101b10; padding: 7px 14px; border: 1px solid #1f3a1f; color: #cfead0; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:hover { background: #142114; }
QTabBar::tab:selected { background: #22c55e; color: #07230e; font-weight: 700; }
QPushButton { background: #0f1a0f; border: 1px solid #2a4a2a; border-radius: 8px; padding: 6px 12px; color: #eaffea; }
QPushButton:hover { background: #142114; }
QPushButton:pressed { background: #0e260e; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox { background: #0f1a0f; border: 1px solid #2a4a2a; color: #eaffea; border-radius: 8px; }
QSlider::groove:horizontal { height: 8px; background: #142114; border-radius: 4px; }
QSlider::handle:horizontal { width: 16px; height: 16px; background: #22c55e; border: 1px solid #16a34a; border-radius: 8px; margin: -6px 0; }
"""



# Ocean — cool blue/teal, medium-dark
QSS_OCEAN = """
QWidget { background: #06151a; color: #e6f7ff; }
QMainWindow, QDialog { background: #06151a; }
QGroupBox { background: #0b2027; border: 1px solid #134b5f; border-radius: 8px; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: #0ea5a7; color: #032022; border-radius: 6px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #104556; top: -1px; background: #0b2027; }
QTabBar::tab { background: #0b2027; padding: 7px 14px; border: 1px solid #104556; color: #cfe8ef; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:hover { background: #0e2c38; }
QTabBar::tab:selected { background: #14b8a6; color: #032022; font-weight: 700; }
QPushButton { background: #0b242d; border: 1px solid #104556; border-radius: 8px; padding: 6px 12px; color: #e6f7ff; }
QPushButton:hover { background: #0e2c38; }
QPushButton:pressed { background: #10323b; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox { background: #0b2027; border: 1px solid #134b5f; color: #e6f7ff; border-radius: 8px; }
QSlider::groove:horizontal { height: 8px; background: #10323b; border-radius: 4px; }
QSlider::handle:horizontal { width: 16px; height: 16px; background: #14b8a6; border: 1px solid #0d9488; border-radius: 8px; margin: -6px 0; }
"""



# Solarized Light — gentle light theme
QSS_SOLARIZED_LIGHT = """
QAbstractScrollArea { background: #fdf6e3; }
QScrollArea QWidget#qt_scrollarea_viewport { background: #fdf6e3; }
QWidget { background: #fdf6e3; color: #073642; }
QMainWindow, QDialog { background: #fdf6e3; }
QGroupBox { background: #eee8d5; border: 1px solid #d6cdb8; border-radius: 8px; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: #268bd2; color: #fdf6e3; border-radius: 6px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #d6cdb8; top: -1px; background: #eee8d5; }
QTabBar::tab { background: #eee8d5; padding: 7px 14px; border: 1px solid #d6cdb8; color: #073642; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:hover { background: #e6dfc9; }
QTabBar::tab:selected { background: #268bd2; color: #fdf6e3; font-weight: 700; }
QPushButton { background: #eee8d5; border: 1px solid #d3c7a7; border-radius: 8px; padding: 6px 12px; color: #073642; }
QPushButton:hover { background: #e6dfc9; }
QPushButton:pressed { background: #e0d7bd; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox { background: #fffffb; border: 1px solid #d3c7a7; color: #073642; border-radius: 8px; }
QSlider::groove:horizontal { height: 8px; background: #eae2cc; border-radius: 4px; }
QSlider::handle:horizontal { width: 16px; height: 16px; background: #268bd2; border: 1px solid #1c6fa8; border-radius: 8px; margin: -6px 0; }
"""



# CRT (retro terminal green on black)
QSS_CRT = """
QWidget { background: #020402; color: #9aff9a; }
QMainWindow, QDialog { background: #020402; }
QGroupBox { background: #061006; border: 1px solid #0c3a0c; border-radius: 8px; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: #16a34a; color: black; border-radius: 6px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #0c3a0c; top: -1px; background: #061006; }
QTabBar::tab { background: #061006; padding: 7px 14px; border: 1px solid #0c3a0c; color: #9aff9a; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:hover { background: #082108; }
QTabBar::tab:selected { background: #16a34a; color: black; font-weight: 700; }
QPushButton { background: #061006; border: 1px solid #093309; border-radius: 8px; padding: 6px 12px; color: #9aff9a; }
QPushButton:hover { background: #082108; }
QPushButton:pressed { background: #0a2f0a; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox { background: #061006; border: 1px solid #0c3a0c; color: #9aff9a; border-radius: 8px; selection-background-color: #0c3a0c; }
QSlider::groove:horizontal { height: 8px; background: #0b1a0b; border-radius: 4px; }
QSlider::handle:horizontal { width: 16px; height: 16px; background: #16a34a; border: 1px solid #0e7a34; border-radius: 8px; margin: -6px 0; }
"""


# Tropical Fiesta — teal/green/orange accents on a dark "jungle" base
QSS_TROPICAL_FIESTA = """
QWidget { background: #0b1210; color: #e8fff4; }
QMainWindow, QDialog { background: #0b1210; }
QGroupBox { background: #0f1e19; border: 1px solid #1e3a2f; border-radius: 10px; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 4px 10px; background: #10b981; color: #03140d; border-radius: 8px; font-weight: 800; }
QTabWidget::pane { border: 1px solid #1e3a2f; top: -1px; background: #0f1e19; }
QTabBar::tab { background: #0e1a16; padding: 8px 16px; border: 1px solid #1e3a2f; color: #d7ffee; border-top-left-radius: 10px; border-top-right-radius: 10px; }
QTabBar::tab:hover { background: #11241e; }
QTabBar::tab:selected {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22d3ee, stop:1 #f59e0b);
    color: #00171b; font-weight: 800; border-color:#22d3ee;
}
QPushButton, QToolButton { background: #0e1a16; border: 1px solid #2a5d4c; border-radius: 12px; padding: 8px 14px; color: #e8fff4; }
QPushButton:hover, QToolButton:hover { background: #143126; }
QPushButton:pressed, QToolButton:pressed { background: #0f271f; }
QPushButton#btn_upscale_quick { background: #f59e0b; border: 1px solid #d97706; color: #261300; }
QPushButton#btn_upscale_quick:hover { background: #fbbf24; }
QToolButton#btn_play    { background: #10b981; color: #03140d; border: 1px solid #059669; }
QToolButton#btn_pause   { background: #f59e0b; color: #261300; border: 1px solid #d97706; }
QToolButton#btn_stop    { background: #ef4444; color: #2a0a0a; border: 1px solid #dc2626; }
QToolButton#btn_fullscreen, QToolButton#btn_screenshot { background: #22d3ee; color: #00171b; border: 1px solid #0891b2; }

/* Ensure icon-only toolbuttons are visible (even when autoRaise/flat) */
QToolButton {
  background: #0e1a16;
  border: 1px solid #2a5d4c;
  color: #e8fff4;
  min-width: 28px; min-height: 28px;
  border-radius: 14px;
}
QToolButton:hover { background: #143126; }
QToolButton:disabled { color: #3d6a57; border-color: #23483b; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox { background: #0c1814; border: 1px solid #2a5d4c; color: #e8fff4; border-radius: 10px; }
QComboBox QAbstractItemView { background: #0c1814; color: #e8fff4; selection-background-color: #143126; }
QSlider::groove:horizontal { height: 9px; background: #193a2f; border-radius: 5px; }
QSlider::handle:horizontal { width: 18px; height: 18px; background: #22d3ee; border: 1px solid #0891b2; border-radius: 9px; margin: -6px 0; }
QProgressBar { border: 1px solid #2a5d4c; border-radius: 8px; background: #0e1a16; text-align: center; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #10b981, stop:1 #f59e0b); }
"""


# Color Mix — multicolor UI with readable contrast
# --- Color Mix (Fluorescent Multicolor) ----------------------------------------------------
CM_BG = "#06090F"           # deep space
CM_GROUP_BG = "#0B1220"     # panel bg
CM_TEXT = "#F4F7FF"         # bright text
CM_BORDER = "#1F2A44"       # subtle outline

# Neon palette
CM_NEON_PINK = "#FF00A8"
CM_NEON_PURPLE = "#A100FF"
CM_NEON_VIOLET = "#C000FF"
CM_NEON_BLUE = "#00E5FF"
CM_NEON_CYAN = "#00FFD1"
CM_NEON_GREEN = "#00FF87"
CM_NEON_YELLOW = "#F8FF00"
CM_NEON_ORANGE = "#FF8A00"

QSS_COLOR_MIX = f"""
QWidget {{ background: {CM_BG}; color: {CM_TEXT}; }}
QMainWindow, QDialog {{ background: {CM_BG}; }}

QGroupBox {{
    background: {CM_GROUP_BG};
    border: 1px solid {CM_BORDER};
    border-radius: 12px;
    margin-top: 16px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px; padding: 0 8px;
    color: {CM_NEON_CYAN};
}}

QTabWidget::pane {{ border: 1px solid {CM_BORDER}; top: -1px; background: {CM_GROUP_BG}; }}
QTabBar::tab {{
    background: #0D1528;
    padding: 9px 18px;
    border: 1px solid {CM_BORDER};
    border-bottom: none;
    border-top-left-radius: 12px; border-top-right-radius: 12px;
}}
QTabBar::tab:hover {{ border-color: {CM_NEON_PURPLE}; }}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {CM_NEON_PINK}, stop:0.2 {CM_NEON_PURPLE},
        stop:0.4 {CM_NEON_BLUE}, stop:0.6 {CM_NEON_GREEN},
        stop:0.8 {CM_NEON_YELLOW}, stop:1 {CM_NEON_ORANGE});
    color: #0A0A0A;
    font-weight: 700;
    border-color: {CM_NEON_YELLOW};
}}

QPushButton {{
    background: #0D1528;
    color: {CM_TEXT};
    border: 1px solid {CM_BORDER};
    border-radius: 12px;
    padding: 9px 14px;
}}
QPushButton:hover {{
    border-color: {CM_NEON_BLUE};
    color: {CM_NEON_BLUE};
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {CM_NEON_GREEN}, stop:1 {CM_NEON_PINK});
    color: #0B0B0B;
}}

QLineEdit, QPlainTextEdit, QTextEdit {{
    background: #0D1528;
    border: 1px solid {CM_BORDER};
    border-radius: 10px;
    selection-background-color: {CM_NEON_PINK};
    selection-color: #120012;
}}
QComboBox {{
    background: #0D1528;
    border: 1px solid {CM_BORDER};
    border-radius: 10px; padding: 6px;
}}
QComboBox QAbstractItemView {{
    background: {CM_GROUP_BG};
    color: {CM_TEXT};
    selection-background-color: {CM_NEON_BLUE};
    selection-color: #001818;
    border: 1px solid {CM_BORDER};
}}

QProgressBar {{
    background: #0D1528;
    border: 1px solid {CM_BORDER};
    border-radius: 12px;
    text-align: center;
    color: {CM_TEXT};
}}
QProgressBar::chunk {{
    border-radius: 12px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {CM_NEON_PINK}, stop:0.16 {CM_NEON_PURPLE},
        stop:0.33 {CM_NEON_BLUE}, stop:0.5 {CM_NEON_CYAN},
        stop:0.66 {CM_NEON_GREEN}, stop:0.83 {CM_NEON_YELLOW},
        stop:1 {CM_NEON_ORANGE});
}}

QSlider::groove:horizontal {{
    height: 6px; background: #0D1528; border: 1px solid {CM_BORDER}; border-radius: 4px;
}}
QSlider::handle:horizontal {{
    width: 18px;
    background: {CM_NEON_YELLOW};
    border: 1px solid {CM_NEON_PINK};
    border-radius: 9px; margin: -6px 0;
}}

QScrollBar:vertical {{
    width: 12px; background: #0D1528; border: 1px solid {CM_BORDER}; border-radius: 6px;
}}
QScrollBar::handle:vertical {{ background: {CM_NEON_PURPLE}; border-radius: 6px; }}

QCheckBox::indicator:unchecked {{
    border: 1px solid {CM_BORDER}; background: #0D1528; width: 16px; height: 16px; border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    border: 1px solid {CM_NEON_BLUE}; background: {CM_NEON_BLUE};
}}
"""



# --- Aurora theme (cool aurora greens/purples) ---------------------------------------------
AURORA_BG = "#0A0F1E"
AURORA_GROUP_BG = "#111931"
AURORA_TEXT = "#E9F5FF"
AURORA_BORDER = "#25314A"
AURORA_TAB_BG = "#16223B"
AURORA_TAB_HOVER = "#1D2A48"
AURORA_BTN = "#16223B"
AURORA_BTN_HOVER = "#1E2F55"
AURORA_ACCENT = "#7CFFCB"
AURORA_ACCENT_TEXT = "#0B1A16"
AURORA_SLIDER = "#7CFFCB"

QSS_AURORA = f"""
QWidget {{ background: {AURORA_BG}; color: {AURORA_TEXT}; }}
QMainWindow, QDialog {{ background: {AURORA_BG}; }}

QGroupBox {{ background: {AURORA_GROUP_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; background: transparent; color: {AURORA_TEXT}; }}

QTabWidget::pane {{ border: 1px solid {AURORA_BORDER}; top: -1px; background: {AURORA_GROUP_BG}; }}
QTabBar::tab {{ background: {AURORA_TAB_BG}; padding: 7px 14px; border: 1px solid {AURORA_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: {AURORA_TAB_HOVER}; }}
QTabBar::tab:selected {{ background: {AURORA_ACCENT}; color: {AURORA_ACCENT_TEXT}; font-weight: 700; }}

QPushButton {{ background: {AURORA_BTN}; border: 1px solid {AURORA_BORDER}; border-radius: 8px; padding: 6px 10px; }}
QPushButton:hover {{ background: {AURORA_BTN_HOVER}; }}
QPushButton:pressed {{ background: {AURORA_ACCENT}; color: {AURORA_ACCENT_TEXT}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {AURORA_TAB_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 6px; selection-background-color: {AURORA_ACCENT}; selection-color: {AURORA_ACCENT_TEXT}; }}
QComboBox {{ background: {AURORA_TAB_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 6px; padding: 4px; }}
QComboBox QAbstractItemView {{ background: {AURORA_GROUP_BG}; color: {AURORA_TEXT}; selection-background-color: {AURORA_ACCENT}; selection-color: {AURORA_ACCENT_TEXT}; }}

QProgressBar {{ background: {AURORA_TAB_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {AURORA_ACCENT}; border-radius: 8px; }}

QSlider::groove:horizontal {{ height: 6px; background: {AURORA_TAB_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 18px; background: {AURORA_SLIDER}; border: 1px solid #0ebfa1; border-radius: 9px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {AURORA_TAB_BG}; border: 1px solid {AURORA_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {AURORA_BORDER}; border-radius: 6px; }}
"""



# --- Mardi Gras theme (purple/green/gold) --------------------------------------------------
MARDI_BG = "#1B0F2A"
MARDI_GROUP_BG = "#24123A"
MARDI_TEXT = "#FDE68A"
MARDI_BORDER = "#3A215C"
MARDI_TAB_BG = "#2D1747"
MARDI_TAB_HOVER = "#3A1F5C"
MARDI_BTN = "#2D1747"
MARDI_BTN_HOVER = "#3E2268"
MARDI_ACCENT = "#22C55E"
MARDI_ACCENT_TEXT = "#06140B"
MARDI_SLIDER = "#10B981"

QSS_MARDI_GRAS = f"""
QWidget {{ background: {MARDI_BG}; color: {MARDI_TEXT}; }}
QMainWindow, QDialog {{ background: {MARDI_BG}; }}

QGroupBox {{ background: {MARDI_GROUP_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; background: transparent; color: {MARDI_TEXT}; }}

QTabWidget::pane {{ border: 1px solid {MARDI_BORDER}; top: -1px; background: {MARDI_GROUP_BG}; }}
QTabBar::tab {{ background: {MARDI_TAB_BG}; padding: 7px 14px; border: 1px solid {MARDI_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: {MARDI_TAB_HOVER}; }}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {MARDI_TAB_BG}, stop:1 {MARDI_ACCENT});
    color: {MARDI_ACCENT_TEXT}; font-weight: 700; border-color: {MARDI_ACCENT};
}}

QPushButton {{ background: {MARDI_BTN}; border: 1px solid {MARDI_BORDER}; border-radius: 8px; padding: 6px 10px; }}
QPushButton:hover {{ background: {MARDI_BTN_HOVER}; }}
QPushButton:pressed {{ background: {MARDI_ACCENT}; color: {MARDI_ACCENT_TEXT}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {MARDI_TAB_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 6px; selection-background-color: {MARDI_ACCENT}; selection-color: {MARDI_ACCENT_TEXT}; }}
QComboBox {{ background: {MARDI_TAB_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 6px; padding: 4px; }}
QComboBox QAbstractItemView {{ background: {MARDI_GROUP_BG}; color: {MARDI_TEXT}; selection-background-color: {MARDI_ACCENT}; selection-color: {MARDI_ACCENT_TEXT}; }}

QProgressBar {{ background: {MARDI_TAB_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {MARDI_ACCENT}; border-radius: 8px; }}

QSlider::groove:horizontal {{ height: 6px; background: {MARDI_TAB_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 18px; background: {MARDI_SLIDER}; border: 1px solid #0a6e50; border-radius: 9px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {MARDI_TAB_BG}; border: 1px solid {MARDI_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {MARDI_BORDER}; border-radius: 6px; }}
"""



# --- Sunburst theme (warm light oranges/yellows) ------------------------------------------
SUNBG = "#FFF7E6"
SUN_GROUP_BG = "#FFFFFF"
SUN_TEXT = "#3B2F2F"
SUN_BORDER = "#FFE0B2"
SUN_TAB_BG = "#FFE8C2"
SUN_TAB_HOVER = "#FFE1A6"
SUN_BTN = "#FFE8C2"
SUN_BTN_HOVER = "#FFE1A6"
SUN_ACCENT = "#FFB74D"
SUN_ACCENT_TEXT = "#3B2F2F"
SUN_SLIDER = "#FF8A00"

QSS_SUNBURST = f"""
QAbstractScrollArea {{ background: {SUNBG}; }}
QScrollArea QWidget#qt_scrollarea_viewport {{ background: {SUNBG}; }}
QWidget {{ background: {SUNBG}; color: {SUN_TEXT}; }}
QMainWindow, QDialog {{ background: {SUNBG}; }}

QGroupBox {{ background: {SUN_GROUP_BG}; border: 1px solid {SUN_BORDER}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; background: transparent; color: {SUN_TEXT}; }}

QTabWidget::pane {{ border: 1px solid {SUN_BORDER}; top: -1px; background: {SUN_GROUP_BG}; }}
QTabBar::tab {{ background: {SUN_TAB_BG}; padding: 7px 14px; border: 1px solid {SUN_BORDER}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: {SUN_TAB_HOVER}; }}
QTabBar::tab:selected {{ background: {SUN_ACCENT}; color: {SUN_ACCENT_TEXT}; font-weight: 700; }}

QPushButton {{ background: {SUN_BTN}; border: 1px solid {SUN_BORDER}; border-radius: 8px; padding: 6px 10px; }}
QPushButton:hover {{ background: {SUN_BTN_HOVER}; }}
QPushButton:pressed {{ background: {SUN_ACCENT}; color: {SUN_ACCENT_TEXT}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {SUN_GROUP_BG}; border: 1px solid {SUN_BORDER}; border-radius: 6px; selection-background-color: {SUN_ACCENT}; selection-color: {SUN_ACCENT_TEXT}; }}
QComboBox {{ background: {SUN_GROUP_BG}; border: 1px solid {SUN_BORDER}; border-radius: 6px; padding: 4px; }}
QComboBox QAbstractItemView {{ background: {SUN_GROUP_BG}; color: {SUN_TEXT}; selection-background-color: {SUN_ACCENT}; selection-color: {SUN_ACCENT_TEXT}; }}

QProgressBar {{ background: {SUN_GROUP_BG}; border: 1px solid {SUN_BORDER}; border-radius: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {SUN_ACCENT}; border-radius: 8px; }}

QSlider::groove:horizontal {{ height: 6px; background: {SUN_TAB_BG}; border: 1px solid {SUN_BORDER}; border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 18px; background: {SUN_SLIDER}; border: 1px solid #e67e22; border-radius: 9px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: {SUN_TAB_BG}; border: 1px solid {SUN_BORDER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {SUN_BORDER}; border-radius: 6px; }}
"""



# --- Candy Pop (flashy neon candy palette on deep magenta) --------------------------------
QSS_CANDY_POP = """
QWidget { background: #120018; color: #FFEAFE; }
QMainWindow, QDialog { background: #120018; }

QGroupBox { background: #1E0026; border: 1px solid #3A0A57; border-radius: 12px; margin-top: 16px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 2px 10px; background: #FF67C0; color: #2A002F; border-radius: 8px; font-weight: 900; }

QTabWidget::pane { border: 1px solid #3A0A57; top: -1px; background: #1E0026; }
QTabBar::tab { background: #240033; padding: 9px 18px; border: 1px solid #3A0A57; border-bottom: none; border-top-left-radius: 12px; border-top-right-radius: 12px; color:#FFEAFE; }
QTabBar::tab:hover { background: #2C0042; border-color: #FF67C0; }
QTabBar::tab:selected {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #FF67C0, stop:1 #00FFD1);
    color:#1A001F; font-weight: 900; border-color:#FF67C0;
}

QPushButton { background: #240033; color: #FFEAFE; border: 1px solid #3A0A57; border-radius: 12px; padding: 9px 14px; }
QPushButton:hover { border-color: #FF67C0; color: #FF67C0; }
QPushButton:pressed { background: #FF67C0; color:#2A002F; border-color:#00FFD1; }

QLineEdit, QPlainTextEdit, QTextEdit { background: #240033; border: 1px solid #3A0A57; border-radius: 10px; selection-background-color: #FF67C0; selection-color:#2A002F; }
QComboBox { background: #240033; border: 1px solid #3A0A57; border-radius: 10px; padding: 6px; color:#FFEAFE; }

QProgressBar { background:#240033; border:1px solid #3A0A57; border-radius: 12px; text-align:center; color:#FFEAFE; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00FFD1, stop:1 #FF67C0); border-radius:12px; }

QSlider::groove:horizontal { height: 6px; background: #240033; border: 1px solid #3A0A57; border-radius: 4px; }
QSlider::handle:horizontal { width: 18px; background: #00FFD1; border: 1px solid #FF67C0; border-radius: 9px; margin: -6px 0; }

QScrollBar:vertical { width: 12px; background: #240033; border: 1px solid #3A0A57; border-radius: 6px; }
QScrollBar::handle:vertical { background: #FF67C0; border-radius: 6px; }
"""


# --- Rainbow Riot (bold multi-color gradients on deep navy) --------------------------------
QSS_RAINBOW_RIOT = """
QWidget { background: #05070A; color: #F5F7FF; }
QMainWindow, QDialog { background: #05070A; }

QGroupBox { background: #0A0F1A; border: 1px solid #1D2A3A; border-radius: 12px; margin-top: 16px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 2px 10px; 
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, 
        stop:0 #FF007A, stop:0.2 #FF8A00, stop:0.4 #F8FF00, stop:0.6 #00FF87, stop:0.8 #00E5FF, stop:1 #A100FF);
    color: #0B0E14; border-radius: 8px; font-weight: 900; 
}

QTabWidget::pane { border: 1px solid #1D2A3A; top: -1px; background: #0A0F1A; }
QTabBar::tab { background: #0C1320; padding: 9px 18px; border: 1px solid #1D2A3A; border-bottom: none; border-top-left-radius: 12px; border-top-right-radius: 12px; color:#E8EEFF; }
QTabBar::tab:hover { background: #0E1829; }
QTabBar::tab:selected { 
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, 
        stop:0 #FF8A00, stop:0.25 #F8FF00, stop:0.5 #00FF87, stop:0.75 #00E5FF, stop:1 #A100FF);
    color:#0B0E14; font-weight: 900; border-color:#00E5FF; 
}

QPushButton { background: #0C1320; color: #F5F7FF; border: 1px solid #1D2A3A; border-radius: 12px; padding: 9px 14px; }
QPushButton:hover { border-color: #00E5FF; color:#00E5FF; }
QPushButton:pressed { 
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00FF87, stop:1 #FF007A);
    color:#0B0E14; border-color:#F8FF00;
}

QLineEdit, QPlainTextEdit, QTextEdit { background: #0C1320; border: 1px solid #1D2A3A; border-radius: 10px; selection-background-color: #FF007A; selection-color:#0D0F16; }
QComboBox { background: #0C1320; border: 1px solid #1D2A3A; border-radius: 10px; padding: 6px; color:#F5F7FF; }

QProgressBar { background:#0C1320; border:1px solid #1D2A3A; border-radius: 12px; text-align:center; color:#F5F7FF; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #FF007A, stop:0.2 #FF8A00, stop:0.4 #F8FF00, stop:0.6 #00FF87, stop:0.8 #00E5FF, stop:1 #A100FF); border-radius:12px; }

QSlider::groove:horizontal { height: 6px; background: #0C1320; border: 1px solid #1D2A3A; border-radius: 4px; }
QSlider::handle:horizontal { width: 18px; background: #F8FF00; border: 1px solid #FF007A; border-radius: 9px; margin: -6px 0; }

QScrollBar:vertical { width: 12px; background: #0C1320; border: 1px solid #1D2A3A; border-radius: 6px; }
QScrollBar::handle:vertical { background: #00E5FF; border-radius: 6px; }
"""


# ----------------------------------------------------------------------------------------
# Added: Sky Light & Pastel Light (light themes that work in Settings tab)
# These reuse the Day palette colors with different gentle accents.
# ----------------------------------------------------------------------------------------
# --- Pastel Light — soft lilac/peach accents on light base --------------------------------
QSS_PASTEL_LIGHT = f"""
QAbstractScrollArea {{ background: {DAY_BG}; }}
QScrollArea QWidget#qt_scrollarea_viewport {{ background: {DAY_BG}; }}
QWidget {{ background: {DAY_BG}; color: {DAY_TEXT}; }}
QMainWindow, QDialog {{ background: {DAY_BG}; }}

QGroupBox {{ background: {DAY_GROUP_BG}; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; margin-top: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: #EADCFD; color: {DAY_TEXT}; border-radius: 6px; font-weight: 600; }}

QTabWidget::pane {{ border: 1px solid {DAY_BORDER_SOFT}; top: -1px; background: {DAY_GROUP_BG}; }}
QTabBar::tab {{ background: #FFF6FA; padding: 7px 14px; border: 1px solid {DAY_BORDER_SOFT}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:hover {{ background: #FDECF4; }}
QTabBar::tab:selected {{ background: {DAY_GROUP_BG}; color: {DAY_TEXT}; font-weight: 700; }}

QPushButton {{ background: #FFF6FA; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; padding: 6px 12px; }}
QPushButton:hover {{ background: #FDECF4; }}
QPushButton:pressed {{ background: #F8E4EF; }}

QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {{ background: {DAY_INPUT_BG}; border: 1px solid {DAY_BORDER_SOFT}; border-radius: 8px; padding: 5px 8px; color: {DAY_TEXT}; }}
QComboBox::drop-down {{ width: 24px; }}

QCheckBox, QRadioButton {{ background: transparent; border: none; }}

QSlider::groove:horizontal {{ height: 6px; background: {DAY_BORDER_SOFT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; background: #F5BFD4; border: 1px solid #eaa3bd; border-radius: 8px; margin: -6px 0; }}

QScrollBar:vertical {{ width: 12px; background: #FFF6FA; border: 1px solid {DAY_BORDER_SOFTER}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {DAY_BORDER_SOFT}; border-radius: 6px; }}
"""
