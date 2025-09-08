
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
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 3px 8px; background: {EVE_HEADER}; color: #003349; border-radius: 6px; font-weight: 700; }}

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
        "contrast": QSS_HIGH_CONTRAST,
        "cyberpunk": QSS_CYBERPUNK,
        "neon": QSS_NEON,
        "ocean": QSS_OCEAN,
        "solarized light": QSS_SOLARIZED_LIGHT,
        "solarized": QSS_SOLARIZED_LIGHT,
        "solar": QSS_SOLARIZED_LIGHT,
        "crt": QSS_CRT,
        "tropical fiesta": QSS_TROPICAL_FIESTA,
        "tropical": QSS_TROPICAL_FIESTA,
        "color mix": QSS_COLOR_MIX,
        "colormix": QSS_COLOR_MIX,
        "aurora": QSS_AURORA,
        "mardi gras": QSS_MARDI_GRAS,
        "mardi grass": QSS_MARDI_GRAS,
        "sunburst": QSS_SUNBURST,
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
    # Fallback
    return QSS_EVENING
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
            _pool = ["Day","Solarized Light","Sunburst","Evening","Night","Slate","High Contrast","Cyberpunk","Neon","Ocean","CRT","Aurora","Mardi Gras","Tropical Fiesta","Color Mix"]
            name = _rand.choice(_pool)
        except Exception:
            name = "Evening"
    elif s == "auto":
        from .framevision_app import pick_auto_theme
        try:
            name = pick_auto_theme()
        except Exception:
            name = "Evening"
    qss = qss_for_theme(name)
    try:
        app.setStyleSheet(qss)
        pal = app.palette()
        if name.lower().startswith("night"):
            pal.setColor(QPalette.Window, QColor(NIGHT_BG))
            pal.setColor(QPalette.Base, QColor(NIGHT_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(NIGHT_TAB_BG))
            pal.setColor(QPalette.Text, Qt.white)
            pal.setColor(QPalette.WindowText, Qt.white)
        elif (name.lower().startswith("day") or "solar" in name.lower() or "sunburst" in name.lower()):
            pal.setColor(QPalette.Window, QColor(DAY_BG))
            pal.setColor(QPalette.Base, QColor(DAY_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(DAY_TAB_BG))
            pal.setColor(QPalette.Text, QColor(DAY_TEXT))
            pal.setColor(QPalette.WindowText, QColor(DAY_TEXT))
        else:
            pal.setColor(QPalette.Window, QColor(EVE_BG))
            pal.setColor(QPalette.Base, QColor(EVE_GROUP_BG))
            pal.setColor(QPalette.Button, QColor(EVE_TAB_BG))
            pal.setColor(QPalette.Text, QColor(EVE_TEXT))
            pal.setColor(QPalette.WindowText, QColor(EVE_TEXT))
        app.setPalette(pal)
        for w in app.allWidgets():
            try:
                w.style().unpolish(w); w.style().polish(w); w.update()
            except Exception:
                pass
    except Exception:
        pass


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
QTabBar::tab:selected {{ background: {SLATE_GROUP_BG}; color: {SLATE_TEXT}; font-weight: 700; }}
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
QTabBar::tab:selected { background: #22d3ee; color: #00171b; font-weight: 800; }
QPushButton, QToolButton { background: #0e1a16; border: 1px solid #2a5d4c; border-radius: 12px; padding: 8px 14px; color: #e8fff4; }
QPushButton:hover, QToolButton:hover { background: #143126; }
QPushButton:pressed, QToolButton:pressed { background: #0f271f; }
QPushButton#btn_upscale_quick { background: #f59e0b; border: 1px solid #d97706; color: #261300; }
QPushButton#btn_upscale_quick:hover { background: #fbbf24; }
QToolButton#btn_play    { background: #10b981; color: #03140d; border: 1px solid #059669; }
QToolButton#btn_pause   { background: #f59e0b; color: #261300; border: 1px solid #d97706; }
QToolButton#btn_stop    { background: #ef4444; color: #2a0a0a; border: 1px solid #dc2626; }
QToolButton#btn_fullscreen, QToolButton#btn_screenshot { background: #22d3ee; color: #00171b; border: 1px solid #0891b2; }
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
QTabBar::tab:selected {{ background: {MARDI_ACCENT}; color: {MARDI_ACCENT_TEXT}; font-weight: 700; }}

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

