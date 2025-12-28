from PySide6.QtWidgets import QMessageBox
import subprocess
# helpers/tools_tab.py — extracted Tools pane (modular)
import os, re, subprocess, sys
from pathlib import Path
from helpers.meme_tool import MemeToolPane
from helpers.trim_tool import install_trim_tool
from helpers.cropper import install_cropper_tool
from helpers.audiotool import install_audio_tool
from helpers.prompt import install_prompt_tool
from helpers.background import install_background_tool
from helpers.renam import RenamPane
from helpers.batch import BatchSelectDialog
from helpers.frames import install_frames_tool
from helpers.musicedit import MusicEditWidget
from helpers.split_glue_video import SpliglueVideoTool
# from helpers.whisper import WhisperWidget
from helpers.metadata import MetadataEditorWidget

import re

def _slug_title(t:str)->str:
    try:
        return re.sub(r'[^a-z0-9]+','_', (t or '').lower()).strip('_')
    except Exception:
        return 'section'
# Ensure widget classes are imported for use below
from PySide6.QtWidgets import QSpinBox, QSlider, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSettings, QTimer, QEvent
from PySide6.QtCore import QDate, QTime, QDateTime
from PySide6.QtCore import Signal, QMimeData
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFormLayout, QSizePolicy, QToolButton, QGroupBox, QLineEdit, QComboBox, QCheckBox, QMessageBox, QFileDialog)

from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QRadioButton, QDateEdit, QTimeEdit, QDateTimeEdit, QMessageBox
from helpers import gif as gif_backend

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS, OUT_TRIMS, OUT_SHOTS, OUT_TEMP
except Exception:
    from pathlib import Path as _Path
    ROOT = _Path('.').resolve()
    BASE = ROOT
    OUT_VIDEOS = BASE/'output'/'video'
    OUT_TRIMS  = BASE/'output'/'video'/'trims'
    OUT_SHOTS  = BASE/'output'/'screenshots'
    OUT_TEMP   = BASE/'output'/'_temp'

# --- Normalize trims output folder (new default) ---
try:
    OUT_TRIMS = (OUT_VIDEOS / 'trims')
except Exception:
    try:
        OUT_TRIMS = ROOT/'output'/'video'/'trims'
    except Exception:
        pass
try:
    OUT_TRIMS.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

try:
    OUT_REVERSE
except Exception:
    try:
        OUT_REVERSE = ROOT/'output'/'video'/'reverse'
    except Exception:
        from pathlib import Path as _Path
        OUT_REVERSE = _Path('output')/'video'/'reverse'


def ffmpeg_path():
    """Resolve ffmpeg, preferring app-local presets/bin first, then bin, then PATH."""
    exe = "ffmpeg.exe" if os.name=="nt" else "ffmpeg"
    candidates = [
        ROOT/"presets"/"bin"/exe,
        ROOT/"bin"/exe,
        "ffmpeg",
    ]
    for c in candidates:
        try:
            subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return "ffmpeg"


def ffprobe_path():
    """Resolve ffprobe, preferring app-local presets/bin first, then bin, then PATH."""
    exe = 'ffprobe.exe' if os.name=='nt' else 'ffprobe'
    candidates = [
        ROOT/"presets"/"bin"/exe,
        ROOT/"bin"/exe,
        'ffprobe',
    ]
    for c in candidates:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return 'ffprobe'

def probe_media(path: Path):
    info = {"width": None, "height": None, "fps": None, "duration": None, "size": None}
    try:
        out = subprocess.check_output([ ffprobe_path(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-show_entries", "format=duration,size",
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(path) ], stderr=subprocess.STDOUT, universal_newlines=True)
        for line in out.splitlines():
            if line.startswith("width="): info["width"] = int(line.split("=")[1])
            if line.startswith("height="): info["height"] = int(line.split("=")[1])
            if line.startswith("avg_frame_rate="):
                fr = line.split("=")[1]
                if "/" in fr:
                    n,d = fr.split("/")
                    if float(d)!=0: info["fps"] = round(float(n)/float(d),2)
            if line.startswith("duration="):
                try: info["duration"] = float(line.split("=")[1])
                except: pass
            if line.startswith("size="):
                try: info["size"] = int(line.split("=")[1])
                except: pass
    except Exception:
        pass
    return info


def has_audio(path: Path) -> bool:
    """Return True if the media file has at least one audio stream."""
    try:
        out = subprocess.check_output([
            ffprobe_path(), "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path)
        ], stderr=subprocess.STDOUT, universal_newlines=True)
        return bool((out or "").strip())
    except Exception:
        return False

# --- Themes (QSS condensed for size)
QSS_DAY = """
QWidget { background:#f3f7fb; color:#111; }
QPushButton { background:#e8f0ff; border:1px solid #a8c4ff; border-radius:8px; padding:6px 10px; }
QPushButton:hover { background:#dbe7ff; }
QPushButton:pressed { background:#cbd9fb; }
QMenuBar, QMenu { background:#f3f7fb; color:#111; }
QMenu::item:selected { background:#e0e8ff; }
QGroupBox { background:transparent; border:1px solid #cfd8ea; border-radius:8px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#f3f7fb; }
QTabWidget::pane { background:#f3f7fb; border:1px solid #cfd8ea; border-radius:10px; }
QTabBar::tab { background:#e9effa; color:#111; padding:6px 12px; border:1px solid #cfd8ea; border-bottom: none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#ffffff; }
QTabBar::tab:hover { background:#f6fbff; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#ffffff; color:#111; border:1px solid #cfd8ea; border-radius:8px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#ffffff; color:#111; border:1px solid #cfd8ea; border-radius:8px; }
QSlider::groove:horizontal { height:6px; background:#c9cfd9; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#6a8cff; border:1px solid #3d5bd6; margin:-5px 0; border-radius:7px; }
QToolTip { background:#ffffff; color:#111; border:1px solid #a8c4ff; }
QMenuBar::item { background:#f3f7fb; color:#111; padding:4px 8px; }
QMenuBar::item:selected { background:#e0e8ff; }
QMenuBar::item:pressed { background:#e0e8ff; }

"""
QSS_EVENING = """


QWidget { background:#808080; color:#e7eefc; }              /* dark gray */
QPushButton { background:#2b3540; color:#e7eefc; border:1px solid #3a4a58; border-radius:12px; padding:8px 12px; }
QPushButton:hover { background:#34404c; }
QPushButton:pressed { background:#2a333d; }
QMenuBar, QMenu { background:#808080; color:#e7eefc; }
QMenu::item:selected { background:#2b3540; }
QGroupBox { background:transparent; border:1px solid #2f3a45; border-radius:10px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#808080; }
QTabWidget::pane { background:#808080; border:1px solid #2f3a45; border-radius:10px; }
QTabBar::tab { background:#222a33; color:#e7eefc; padding:6px 12px; border:1px solid #2f3a45; border-bottom:none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#2b3540; }
QTabBar::tab:hover { background:#2a333d; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#141920; color:#e7eefc; border:1px solid #2f3a45; border-radius:10px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#141920; color:#e7eefc; border:1px solid #2f3a45; border-radius:10px; }
QSlider::groove:horizontal { height:6px; background:#26303a; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#49617a; border:1px solid #394d62; margin:-5px 0; border-radius:7px; }
QToolTip { background:#222a33; color:#e7eefc; border:1px solid #3a4a58; }
QMenuBar::item { background:#1a1f26; color:#e7eefc; padding:4px 8px; }
QMenuBar::item:selected { background:#2b3540; }
QMenuBar::item:pressed { background:#2b3540; }



"""
QSS_NIGHT = """
QWidget { background:#0b1118; color:#d8e6ff; }
QPushButton { background:#0f1b2a; color:#d8e6ff; border:1px solid #1f2e45; border-radius:12px; padding:8px 12px; }
QPushButton:hover { background:#122136; }
QPushButton:pressed { background:#0e1a2a; }
QMenuBar, QMenu { background:#0b1118; color:#d8e6ff; }
QMenu::item:selected { background:#142238; }
QGroupBox { background:transparent; border:1px solid #1a2a40; border-radius:10px; margin-top:8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#0b1118; }
QTabWidget::pane { background:#0b1118; border:1px solid #1a2a40; border-radius:10px; }
QTabBar::tab { background:#0f1926; color:#d8e6ff; padding:6px 12px; border:1px solid #1a2a40; border-bottom:none; border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }
QTabBar::tab:selected { background:#142238; }
QTabBar::tab:hover { background:#122136; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#0b141f; color:#d8e6ff; border:1px solid #1a2a40; border-radius:10px; padding:4px 6px; }
QListView, QTreeView, QTableView { background:#0b141f; color:#d8e6ff; border:1px solid #1a2a40; border-radius:10px; }
QSlider::groove:horizontal { height:6px; background:#12213a; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#1f3963; border:1px solid #12213a; margin:-5px 0; border-radius:7px; }
QToolTip { background:#0f1b2a; color:#d8e6ff; border:1px solid #1f2e45; }
QMenuBar::item { background:#0b1118; color:#d8e6ff; padding:4px 8px; }
QMenuBar::item:selected { background:#142238; }
QMenuBar::item:pressed { background:#142238; }

"""


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None, expanded=False):
        super().__init__(parent)
        try:
            self.setObjectName('sec_' + _slug_title(title))
        except Exception:
            pass
        self._expanded = bool(expanded)
        self.toggle = QToolButton(self)
        self.toggle.setStyleSheet("QToolButton { border:none; text-align:left; padding-left:0px; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        try:
            self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        self.toggle.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)

        # --- Reorder grip (shown only when reorder mode is enabled) ---
        self.title = title
        self._reorder_enabled = False
        self._drag_press_pos = None
        self.grip = QLabel("⠿", self)
        try:
            self.grip.setObjectName("reorder_grip")
        except Exception:
            pass
        self.grip.setFixedWidth(18)
        self.grip.setAlignment(Qt.AlignCenter)
        self.grip.setToolTip("Drag to reorder tools")
        self.grip.setCursor(Qt.ArrowCursor)
        self.grip.setVisible(False)
        self.grip.installEventFilter(self)

        self.header = QWidget(self)
        hlay = QHBoxLayout(self.header)
        hlay.setContentsMargins(0,0,0,0)
        hlay.setSpacing(6)
        try:
            hlay.setAlignment(Qt.AlignLeft)
        except Exception:
            pass
        hlay.addWidget(self.grip, 0)
        hlay.addWidget(self.toggle, 1)

        self.content = QWidget(self)
        self.content.setLayout(QVBoxLayout()); self.content.layout().setSpacing(8)
        self.content.layout().setContentsMargins(12, 6, 12, 12)
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.content.setVisible(self._expanded)
        self.content.setMaximumHeight(16777215 if self._expanded else 0)

        self.anim = QPropertyAnimation(self.content, b"maximumHeight", self)
        self.anim.setDuration(160)
        self.anim.finished.connect(self._on_anim_finished)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6,6,6,6)
        lay.setSpacing(6)
        lay.addWidget(self.header)
        lay.addWidget(self.content)

        def _on_toggled(on):
            self._expanded = on
            self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

            # Critical: when collapsed, ensure the content does not reserve space in layouts.
            if not on:
                try:
                    self.content.setMinimumHeight(0)
                except Exception:
                    pass

            self.content.setVisible(True)  # visible during animation
            start = self.content.maximumHeight()

            if on:
                min_h = 0
                try:
                    min_h = int(self.content.property("expand_min_h") or 0)
                except Exception:
                    min_h = 0
                if not min_h:
                    try:
                        lay0 = self.content.layout()
                        if lay0 is not None and lay0.count() > 0:
                            w0 = lay0.itemAt(0).widget()
                            if w0 is not None:
                                min_h = int(w0.property("expand_min_h") or 0)
                    except Exception:
                        pass
                end = max(self.content.sizeHint().height(), min_h)
            else:
                end = 0

            self.anim.stop()
            self.anim.setStartValue(start)
            self.anim.setEndValue(end)
            self.anim.start()
        self.toggle.toggled.connect(_on_toggled)

    def _on_anim_finished(self):
        if self._expanded:
            self.content.setMaximumHeight(16777215)  # give natural height so scrollbars can appear
            min_h = 0
            try:
                min_h = int(self.content.property("expand_min_h") or 0)
            except Exception:
                min_h = 0
            if not min_h:
                try:
                    lay0 = self.content.layout()
                    if lay0 is not None and lay0.count() > 0:
                        w0 = lay0.itemAt(0).widget()
                        if w0 is not None:
                            min_h = int(w0.property("expand_min_h") or 0)
                except Exception:
                    pass
            if min_h:
                try:
                    self.content.setMinimumHeight(min_h)
                except Exception:
                    pass
        else:
            # Fully collapse without reserving layout space.
            try:
                self.content.setMinimumHeight(0)
            except Exception:
                pass
            self.content.setMaximumHeight(0)
            self.content.setVisible(False)

    def setContentLayout(self, layout):
        QWidget().setLayout(self.content.layout())
        self.content.setLayout(layout)
        if self._expanded:
            self.content.setMaximumHeight(16777215)
        else:
            self.content.setMaximumHeight(0)
    def setReorderEnabled(self, on: bool):
        """Show/hide the grip and enable drag-start only from the grip."""
        try:
            self._reorder_enabled = bool(on)
            self.grip.setVisible(self._reorder_enabled)
            self.grip.setCursor(Qt.OpenHandCursor if self._reorder_enabled else Qt.ArrowCursor)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # Only intercept events on the grip label while reorder mode is enabled.
        try:
            if obj is getattr(self, "grip", None) and getattr(self, "_reorder_enabled", False):
                et = event.type()
                if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    try:
                        self._drag_press_pos = event.position().toPoint()
                    except Exception:
                        self._drag_press_pos = event.pos()
                    try:
                        self.grip.setCursor(Qt.ClosedHandCursor)
                    except Exception:
                        pass
                    return True
                elif et == QEvent.MouseMove and self._drag_press_pos is not None:
                    try:
                        pos = event.position().toPoint()
                    except Exception:
                        pos = event.pos()
                    try:
                        dist = (pos - self._drag_press_pos).manhattanLength()
                    except Exception:
                        dist = 0
                    if dist >= QApplication.startDragDistance():
                        self._drag_press_pos = None
                        self._start_reorder_drag()
                        try:
                            self.grip.setCursor(Qt.OpenHandCursor)
                        except Exception:
                            pass
                        return True
                    return True
                elif et == QEvent.MouseButtonRelease:
                    self._drag_press_pos = None
                    try:
                        self.grip.setCursor(Qt.OpenHandCursor)
                    except Exception:
                        pass
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _start_reorder_drag(self):
        try:
            drag = QDrag(self.grip)
            mime = QMimeData()
            mime.setData("application/x-framevision-tool-section", (self.title or "").encode("utf-8"))
            drag.setMimeData(mime)
            drag.exec(Qt.MoveAction)
        except Exception:
            pass



# --- Add safe helpers to expose expanded state for saving/restoring
def _cs_isChecked(self):
    try:
        return bool(self._expanded)
    except Exception:
        return False

def _cs_setChecked(self, on: bool):
    try:
        on = bool(on)
        # Drive via the toggle so animation/state stays consistent
        if getattr(self, '_expanded', None) is None:
            self._expanded = on
        if getattr(self, 'toggle', None) is not None:
            if self._expanded != on:
                self.toggle.setChecked(on)
            else:
                # ensure visuals are correct
                self.content.setVisible(on)
                self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        else:
            # Fallback: directly set
            self._expanded = on
    except Exception:
        pass

try:
    CollapsibleSection.isChecked = _cs_isChecked
    CollapsibleSection.setChecked = _cs_setChecked
except Exception:
    pass



class ToolsReorderHost(QWidget):
    """Drop target that reorders CollapsibleSection widgets inside the Tools scroll layout."""
    MIME = "application/x-framevision-tool-section"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reorder_enabled = False
        self._pane = None
        self._layout = None
        try:
            self.setAcceptDrops(False)
        except Exception:
            pass

    def setContext(self, pane, layout):
        self._pane = pane
        self._layout = layout

    def setReorderEnabled(self, on: bool):
        self._reorder_enabled = bool(on)
        try:
            self.setAcceptDrops(self._reorder_enabled)
        except Exception:
            pass

    def dragEnterEvent(self, event):
        try:
            if self._reorder_enabled and event.mimeData().hasFormat(self.MIME):
                event.acceptProposedAction()
                return
        except Exception:
            pass
        try:
            event.ignore()
        except Exception:
            pass

    def dragMoveEvent(self, event):
        try:
            if self._reorder_enabled and event.mimeData().hasFormat(self.MIME):
                event.acceptProposedAction()
                return
        except Exception:
            pass
        try:
            event.ignore()
        except Exception:
            pass

    def dropEvent(self, event):
        try:
            if not (self._reorder_enabled and event.mimeData().hasFormat(self.MIME)):
                event.ignore()
                return

            try:
                raw = bytes(event.mimeData().data(self.MIME))
                title = raw.decode("utf-8", "ignore")
            except Exception:
                title = ""

            if not title or self._pane is None or self._layout is None:
                event.ignore()
                return

            w = None
            try:
                # Prefer a map keyed by the actual visible section titles.
                # (The remember-menu map uses friendlier names that don't always match the header text.)
                w = getattr(self._pane, "_reorder_by_title", {}).get(title, None)
            except Exception:
                w = None
            if w is None:
                # Fallback (older builds)
                try:
                    w = getattr(self._pane, "_sections_map", {}).get(title, None)
                except Exception:
                    w = None
            if w is None:
                event.ignore()
                return

            # Current widget order (ignore stretch/spacers)
            widgets = []
            for i in range(self._layout.count()):
                it = self._layout.itemAt(i)
                ww = it.widget() if it is not None else None
                if ww is not None and isinstance(ww, CollapsibleSection):
                    widgets.append(ww)

            if w not in widgets:
                event.ignore()
                return

            # Drop y position (Qt6: position(), Qt5: pos())
            try:
                y = event.position().toPoint().y()
            except Exception:
                y = event.pos().y()

            # Remove moving widget from list, then compute insertion index
            widgets_wo = [x for x in widgets if x is not w]
            insert_at = len(widgets_wo)
            for i, ww in enumerate(widgets_wo):
                cy = ww.y() + (ww.height() / 2.0)
                if y < cy:
                    insert_at = i
                    break

            new_order = list(widgets_wo)
            new_order.insert(insert_at, w)

            # Rebuild layout in the new order (keep widgets alive)
            # Remove widgets
            for ww in widgets:
                try:
                    self._layout.removeWidget(ww)
                except Exception:
                    pass
            # Remove trailing stretch/spacers
            for i in reversed(range(self._layout.count())):
                it = self._layout.itemAt(i)
                if it is not None and it.spacerItem() is not None:
                    try:
                        self._layout.removeItem(it)
                    except Exception:
                        pass

            for ww in new_order:
                self._layout.addWidget(ww)
            self._layout.addStretch(1)

            try:
                self._pane._save_tools_order()
            except Exception:
                pass

            event.acceptProposedAction()
        except Exception:
            try:
                event.ignore()
            except Exception:
                pass


class InstantToolsPane(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.setSpacing(12)

        # Fancy banner at the top for Multi Tool tab
        self.banner = QLabel("Multi Tool")
        self.banner.setObjectName("toolsBanner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.banner.setFixedHeight(48)
        self.banner.setStyleSheet(
            "#toolsBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: #e8f5e9;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #424242,"
            "   stop:0.5 #1e88e5,"
            "   stop:1 #1b5e20"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        outer.addWidget(self.banner)
        outer.addSpacing(4)

        # --- Global remember toggle for Tools tab ---
        try:
            self._qs = QSettings("FrameVision","FrameVision")
        except Exception:
            from PySide6.QtCore import QSettings as _QS
            self._qs = _QS("FrameVision","FrameVision")
        try:
            remember_enabled = bool(self._qs.value("ToolsPane/remember_enabled", True, type=bool))
        except Exception:
            remember_enabled = True
        topbar = QHBoxLayout(); topbar.setContentsMargins(0,0,0,0); topbar.addStretch(1)
        try:
            self.cb_remember = QCheckBox("Remember settings"); self.cb_remember.setChecked(remember_enabled)
            self.btn_remember_menu = QPushButton("Remember…")
            self.remember_menu = QMenu(self.btn_remember_menu)
            self.btn_remember_menu.setMenu(self.remember_menu)
            topbar.addWidget(self.cb_remember); topbar.addWidget(self.btn_remember_menu)
            outer.addLayout(topbar)
            self.cb_remember.toggled.connect(lambda v: self._qs.setValue("ToolsPane/remember_enabled", bool(v)))
        except Exception:
            pass

        # --- Reorder tools toggle (drag via grip/handle) ---
        try:
            reorder_mode = bool(self._qs.value("ToolsPane/reorder_tools_enabled", False, type=bool))
        except Exception:
            reorder_mode = False
        self._reorder_tools_enabled = bool(reorder_mode)

        try:
            rb = QHBoxLayout()
            rb.setContentsMargins(6, 0, 6, 0)
            rb.setSpacing(8)
            self.cb_reorder_tools = QCheckBox("Reorder tools")
            self.cb_reorder_tools.setChecked(self._reorder_tools_enabled)
            rb.addWidget(self.cb_reorder_tools)
            rb.addStretch(1)
            outer.addLayout(rb)
            self.cb_reorder_tools.toggled.connect(self._set_reorder_tools_enabled)
        except Exception:
            pass

        # --- Scroll container (shows vertical scrollbar only when needed) ---
        from PySide6.QtWidgets import QScrollArea
        self.tools_scroll = QScrollArea(self)
        self.tools_scroll.setWidgetResizable(True)
        self.tools_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        _scroll_host = ToolsReorderHost(self.tools_scroll)
        root = QVBoxLayout(_scroll_host)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(10)

        # Reorder-drop host context
        self._tools_root_layout = root
        self._tools_reorder_host = _scroll_host
        try:
            _scroll_host.setContext(self, root)
        except Exception:
            pass

        self.tools_scroll.setWidget(_scroll_host)
        outer.addWidget(self.tools_scroll, 1)

        # --- Auto arrange width / prevent hidden horizontal scrolling ---
        # Sometimes touchpad/Qt can scroll the Tools area horizontally even when the bar is hidden,
        # making the left side look like it jumped off-screen. We hard-clamp horizontal scroll back
        # to 0 and keep the scroll-host at least as wide as the viewport.
        self._tools_scroll_host = _scroll_host
        self._install_tools_auto_width_guard()


# Sections
        sec_speed = CollapsibleSection("Slow motion - Speedup Video", expanded=False)
        sec_reverse = CollapsibleSection("Reverse video - Boomerang", expanded=False)
        sec_resize = CollapsibleSection("Resize/convert - Images/Video", expanded=False)
        sec_splitglue = CollapsibleSection("Video Split and glue together", expanded=False)
        sec_gif = CollapsibleSection("Create animated gifs from images or video", expanded=False)
        sec_extract = CollapsibleSection("Extract frames", expanded=False)
        sec_trim = CollapsibleSection("Video Trim Lab", expanded=False)
        sec_crop = CollapsibleSection("Cropping", expanded=False)
        sec_describe = CollapsibleSection("Describe anything with Qwen3 VL", expanded=False)
        _desc_wrap = QWidget(); _descl = QVBoxLayout(_desc_wrap); _descl.setContentsMargins(0,0,0,0)
        _descl.setSpacing(6)
        _desc = None
        try:
            from helpers.describer import DescriberWidget
            _desc = DescriberWidget()
            try:
                _desc.setParent(self)
            except Exception:
                pass
        except Exception:
            _desc = None
        if _desc is not None:
            try:
                _desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass
            try:
                _desc.setMinimumHeight(560)
            except Exception:
                pass
            _descl.addWidget(_desc, 1)
        else:
            _fallback = QLabel("Describe tool is unavailable (missing helpers/describer.py).")
            _fallback.setWordWrap(True)
            _descl.addWidget(_fallback)
        sec_describe.setContentLayout(_descl)
        try:
            sec_describe.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        sec_upscale = CollapsibleSection("Upscale Video and images", expanded=False)
        _upsc_wrap = QWidget(); _upscl = QVBoxLayout(_upsc_wrap); _upscl.setContentsMargins(0,0,0,0)
        _upscl.setSpacing(6)
        try:
            from helpers.upsc import UpscPane
            _upsc = UpscPane(self)
        except Exception:
            try:
                from helpers.upsc import UpscPane
                _upsc = UpscPane(None)
            except Exception:
                _upsc = None
        # Wire the main-window "Upscale" button to this embedded Upscale tool.
        # (This prevents the old legacy quick-upscale popup from firing.)
        try:
            self.upsc_pane = _upsc
        except Exception:
            pass
        try:
            if _upsc is not None and hasattr(_upsc, "set_main"):
                _upsc.set_main(self.main)
        except Exception:
            pass
        # Deterministic re-wire: target the player's quick button by objectName if possible.
        try:
            from PySide6.QtWidgets import QPushButton as _FVQPushButton
            _btnq = None
            try:
                _btnq = self.main.findChild(_FVQPushButton, "btn_upscale_quick")
            except Exception:
                _btnq = None
            if _btnq is None:
                try:
                    _btnq = getattr(getattr(self.main, "video", None), "btn_upscale", None)
                except Exception:
                    _btnq = None
            if _btnq is not None and _upsc is not None:
                try:
                    _btnq.clicked.disconnect()
                except Exception:
                    pass
                def _fv_tools_upscale_quick():
                    try:
                        if hasattr(_upsc, "btn_upscale"):
                            _upsc.btn_upscale.click()
                        else:
                            _upsc._do_single()
                    except Exception:
                        try:
                            _upsc._do_single()
                        except Exception:
                            pass
                _btnq.clicked.connect(_fv_tools_upscale_quick)
        except Exception:
            pass

        if _upsc is not None:
            try:
                _upsc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass
            # Important: when embedded in a collapsible Tools section, Qt may otherwise pick a tiny sizeHint
            try:
                _upsc.setMinimumHeight(560)
            except Exception:
                pass
            _upscl.addWidget(_upsc, 1)
        else:
            _fallback = QLabel("Upscale tool is unavailable (missing helpers/upsc.py).")
            _fallback.setWordWrap(True)
            _upscl.addWidget(_fallback)
        sec_upscale.setContentLayout(_upscl)
        try:
            sec_upscale.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        

        # ---- RIFE FPS (Interpolation) ----
        sec_rife = CollapsibleSection("Rife interpolation", expanded=False)
        _rife_wrap = QWidget(); _rifel = QVBoxLayout(_rife_wrap); _rifel.setContentsMargins(0,0,0,0)
        _rifel.setSpacing(6)
        try:
            from helpers.interp import InterpPane
            _rife = InterpPane(self.main, {"ROOT": str(ROOT)}, parent=self)
        except Exception:
            try:
                from helpers.interp import InterpPane
                _rife = InterpPane(None, {"ROOT": str(ROOT)}, parent=self)
            except Exception:
                _rife = None
        if _rife is not None:
            try:
                _rife.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass
            # Important: when embedded in a collapsible Tools section, Qt may otherwise pick a tiny sizeHint
            try:
                _rife.setMinimumHeight(640)
            except Exception:
                pass
            _rifel.addWidget(_rife, 1)
        else:
            _rfallback = QLabel("Rife Fps tool is unavailable (missing helpers/interp.py).")
            _rfallback.setWordWrap(True)
            _rifel.addWidget(_rfallback)
        sec_rife.setContentLayout(_rifel)
        try:
            sec_rife.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

# ---- Meme / Caption ----
        sec_meme = CollapsibleSection("Thumbnail / Meme Creator", expanded=True)
        _meme_wrap = QWidget(); _memel = QVBoxLayout(_meme_wrap); _memel.setContentsMargins(0,0,0,0)
        try:
            _meme = MemeToolPane(self.main, self)
        except Exception:
            _meme = MemeToolPane(None, self)
        _memel.addWidget(_meme)
        sec_meme.setContentLayout(_memel)
        
        
        # ---- Prompt Generator ----
        sec_prompt = CollapsibleSection("Prompt Enhancement", expanded=False)
        try:
            install_prompt_tool(self, sec_prompt)
        except Exception:
            try:
                _p = QWidget(); _pl = QVBoxLayout(_p); _pl.setContentsMargins(0,0,0,0); _pl.addWidget(QLabel("Prompt tool failed to load."))
                sec_prompt.setContentLayout(_pl)
            except Exception:
                pass


        # ---- Background Remover / Inpainter ----
        sec_bg = CollapsibleSection("Background Remover / Inpainter", expanded=False)
        try:
            install_background_tool(self, sec_bg)
        except Exception:
            try:
                _bg_wrap = QWidget(); _bg_l = QVBoxLayout(_bg_wrap); _bg_l.setContentsMargins(0,0,0,0)
                _bg_l.addWidget(QLabel("Background Remover tool failed to load (missing helpers/background.py)."))
                sec_bg.setContentLayout(_bg_l)
            except Exception:
                pass

        # ---- HunyuanVideo 1.5 (Diffusers) ----
        sec_hunyuan15 = None
        try:
            _hy_env = Path(str(ROOT)) / ".hunyuan15_env"
        except Exception:
            _hy_env = Path(".hunyuan15_env")
        _hunyuan15_installed = False
        try:
            _hunyuan15_installed = bool(_hy_env.exists() and _hy_env.is_dir())
        except Exception:
            _hunyuan15_installed = False
        if _hunyuan15_installed:
            sec_hunyuan15 = CollapsibleSection("HunyuanVideo 1.5", expanded=False)
            try:
                from helpers.hunyuan15 import install_hunyuan15_tool
                install_hunyuan15_tool(self, sec_hunyuan15)
            except Exception:
                _hy_wrap = QWidget(); _hy_l = QVBoxLayout(_hy_wrap); _hy_l.setContentsMargins(0,0,0,0)
                _hy_l.addWidget(QLabel("HunyuanVideo 1.5 tool failed to load."))
                sec_hunyuan15.setContentLayout(_hy_l)

        # ---- Ace-Step Music creation ----
        sec_ace = CollapsibleSection("Ace Step Music creation", expanded=False)
        _ace_wrap = QWidget(); _ace_layout = QVBoxLayout(_ace_wrap); _ace_layout.setContentsMargins(0,0,0,0)
        _ace_layout.setSpacing(6)
        try:
            from helpers.ace import acePane as _AcePane
        except Exception:
            _AcePane = None
        _ace_widget = None
        if _AcePane is not None:
            try:
                _ace_widget = _AcePane(self)
            except Exception:
                try:
                    _ace_widget = _AcePane(None)
                except Exception:
                    _ace_widget = None
        if _ace_widget is not None:
            try:
                _ace_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass
            try:
                _ace_widget.setMinimumHeight(640)
            except Exception:
                pass
            _ace_layout.addWidget(_ace_widget, 1)
        else:
            _ace_fallback = QLabel("Ace-Step tool failed to load (missing helpers/ace.py).")
            _ace_fallback.setWordWrap(True)
            _ace_layout.addWidget(_ace_fallback)
        sec_ace.setContentLayout(_ace_layout)
        try:
            sec_ace.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        # ---- Multi Rename (moved to helpers/renam.py) ----
        sec_rename = CollapsibleSection("Multi Rename/Replace", expanded=False)
        _rn_wrap = QWidget(); _rn_layout = QVBoxLayout(_rn_wrap); _rn_layout.setContentsMargins(0,0,0,0)
        try:
            _rn_widget = RenamPane(self.main, self)
        except Exception:
            _rn_widget = RenamPane(None, self)
        _rn_layout.addWidget(_rn_widget)
        sec_rename.setContentLayout(_rn_layout)

        # Whisper Lab (Faster-Whisper)
        sec_whisper = CollapsibleSection("Whisper Lab", expanded=False)
        _whisper_wrap = QWidget(); _whisper_layout = QVBoxLayout(_whisper_wrap); _whisper_layout.setContentsMargins(0,0,0,0)
        try:
            _whisper_widget = WhisperWidget(self)
        except Exception:
            try:
                _whisper_widget = WhisperWidget(None)
            except Exception:
                _whisper_widget = QLabel("Whisper Lab failed to load.")
        _whisper_layout.addWidget(_whisper_widget)
        sec_whisper.setContentLayout(_whisper_layout)

        # Metadata editor (embedded directly in this section)
        sec_metadata = CollapsibleSection("Metadata editor", expanded=False)
        _meta_wrap = QWidget()
        _meta_layout = QVBoxLayout(_meta_wrap)
        _meta_layout.setContentsMargins(0, 0, 0, 0)
        try:
            # If the main window exposes a shared log QTextEdit, pass it through.
            if hasattr(self.main, "log_widget") and isinstance(getattr(self.main, "log_widget"), QTextEdit):
                _meta_widget = MetadataEditorWidget(parent=self, external_log_widget=self.main.log_widget)
            else:
                _meta_widget = MetadataEditorWidget(parent=self)
        except Exception:
            _meta_widget = QLabel("Metadata Editor failed to load.")
        _meta_layout.addWidget(_meta_widget)
        sec_metadata.setContentLayout(_meta_layout)

        # Speed
        self.speed = QSlider(Qt.Horizontal); self.speed.setRange(25, 250); self.speed.setValue(150)
        self.lbl_speed = QLabel("1.50x"); self.speed.valueChanged.connect(lambda v: self.lbl_speed.setText(f"{v/100:.2f}x"))
        from PySide6.QtWidgets import QDoubleSpinBox
        self.spin_speed = QDoubleSpinBox(); self.spin_speed.setRange(0.25, 2.50); self.spin_speed.setDecimals(2); self.spin_speed.setSingleStep(0.01); self.spin_speed.setValue(1.50)
        # sync both ways
        self.speed.valueChanged.connect(lambda v: self.spin_speed.setValue(round(v/100.0,2)))
        self.spin_speed.valueChanged.connect(lambda val: self.speed.setValue(int(round(val*100))))
        self.btn_speed = QPushButton("Change Speed"); self.btn_speed.setToolTip("Change playback speed. If Sound sync is on, audio pitch is preserved."); self.btn_speed.setToolTip("Change playback speed. 1.00x keeps pitch if Sound sync is on."); self.btn_speed_batch = QPushButton("Batch…"); self.btn_speed_batch.setToolTip("Batch with current Speed settings."); self.btn_speed_batch.setToolTip("Select multiple videos or a folder; one job per file using current Speed settings.")
        lay_speed = QFormLayout();
        row_speed = QHBoxLayout(); row_speed.addWidget(self.speed); row_speed.addWidget(self.spin_speed); lay_speed.addRow("Speed factor", row_speed); lay_speed.addRow("", self.lbl_speed); row_b = QHBoxLayout(); row_b.addWidget(self.btn_speed); row_b.addWidget(self.btn_speed_batch); lay_speed.addRow(row_b)

        # Audio options
        try:
            self.cb_speed_sync = QCheckBox("Sound sync (adjust audio tempo)"); self.cb_speed_sync.setToolTip("Adjust audio tempo to match new speed."); self.cb_speed_sync.setToolTip("When enabled, audio tempo is adjusted to match the new speed.")
            self.cb_speed_sync.setChecked(True)
        except Exception:
            pass
        try:
            self.cb_speed_mute = QCheckBox("Mute audio"); self.cb_speed_mute.setToolTip("Output video without audio."); self.cb_speed_mute.setToolTip("Mute the audio track in the output.")
            self.cb_speed_mute.setChecked(False)
        except Exception:
            pass
        try:
            lay_speed.addRow(self.cb_speed_sync)
            lay_speed.addRow(self.cb_speed_mute)
        except Exception:
            pass
        
        try:
            lay_speed.addRow("Factor (type)", self.spin_speed)
        except Exception:
            pass
        sec_speed.setContentLayout(lay_speed)
        # Presets buttons
        row = QHBoxLayout(); btn_ss = QPushButton("Save preset"); btn_ls = QPushButton("Load preset"); row.addWidget(btn_ss); row.addWidget(btn_ls)
        lay_speed.addRow(row)
        btn_ss.clicked.connect(lambda: self._save_preset_speed())
        btn_ls.clicked.connect(lambda: self._load_preset_speed())

        
        # Reverse video
        self.cb_reverse_mute = QCheckBox("Mute audio")
        self.cb_reverse_mute.setToolTip("Output reversed video without audio.")
        self.btn_reverse_load = QPushButton("Load video…")
        self.btn_reverse_load.setToolTip("Pick a video file to use with the Reverse video tool.")
        self.btn_reverse = QPushButton("Reverse video")
        self.btn_reverse.setToolTip("Create a reversed playback copy of the selected video or the main player video.")
        self.btn_reverse_batch = QPushButton("Batch…")
        self.btn_reverse_batch.setToolTip("Reverse multiple videos at once using the current Reverse settings.")
        self.btn_reverse_open_folder = QPushButton("View results")
        self.btn_reverse_open_folder.setToolTip("Open these results in Media Explorer.")
        self.lbl_reverse_info = QLabel("Using main Media Player video. No separate file loaded.")
        try:
            self.lbl_reverse_info.setWordWrap(True)
        except Exception:
            pass
        lay_reverse = QFormLayout()
        lay_reverse.addRow(self.btn_reverse_load)
        lay_reverse.addRow(self.lbl_reverse_info)
        # Boomerang controls
        self.cb_reverse_boom = QCheckBox("Boomerang")
        self.cb_reverse_boom.setToolTip("Create a forward+backward boomerang loop instead of a simple reverse.")
        self.spin_reverse_loops = QSpinBox()
        self.spin_reverse_loops.setRange(1, 9)
        self.spin_reverse_loops.setValue(1)
        self.spin_reverse_loops.setToolTip("Number of forward+backward boomerang repeats (1–9).")
        try:
            self.spin_reverse_loops.setVisible(False)
        except Exception:
            pass
        try:
            self.cb_reverse_boom.toggled.connect(self.spin_reverse_loops.setVisible)
        except Exception:
            pass
        row_boom = QHBoxLayout()
        row_boom.addWidget(self.cb_reverse_boom)
        row_boom.addWidget(self.spin_reverse_loops)
        lay_reverse.addRow(row_boom)
        lay_reverse.addRow(self.cb_reverse_mute)
        row_reverse = QHBoxLayout()
        row_reverse.addWidget(self.btn_reverse)
        row_reverse.addWidget(self.btn_reverse_batch)
        row_reverse.addWidget(self.btn_reverse_open_folder)
        lay_reverse.addRow(row_reverse)
        sec_reverse.setContentLayout(lay_reverse)

        # Resize (moved to helpers/resize.py)

        try:

            from helpers.resize import install_resize_tool

            install_resize_tool(self, sec_resize)

        except Exception as e:

            try:

                QMessageBox.critical(self, 'Resize init failed', str(e))

            except Exception:

                pass
# GIF
        self.gif_fps = QSlider(Qt.Horizontal); self.gif_fps.setRange(5,30); self.gif_fps.setValue(12)
        self.lbl_gif_fps = QLabel("12"); self.gif_fps.valueChanged.connect(lambda v: self.lbl_gif_fps.setText(str(v)))
        self.spin_gif_fps = QSpinBox(); self.spin_gif_fps.setRange(5,30); self.spin_gif_fps.setValue(12)
        self.gif_fps.valueChanged.connect(lambda v: self.spin_gif_fps.setValue(int(v)))
        self.spin_gif_fps.valueChanged.connect(lambda val: self.gif_fps.setValue(int(val)))
        self.gif_same = QCheckBox("Same as video"); self.gif_same.setToolTip("Use the source video FPS for the GIF."); self.gif_same.toggled.connect(lambda on: (self.gif_fps.setEnabled(not on), self.spin_gif_fps.setEnabled(not on)))
        self.btn_gif = QPushButton("Export GIF"); self.btn_gif.setToolTip("Queue a GIF export using the settings above."); self.btn_gif_batch = QPushButton("Batch…"); self.btn_gif_batch.setToolTip("Batch with current GIF settings."); self.btn_gif_batch.setToolTip("Batch GIF export with current FPS setting.")
        self.btn_gif.setToolTip("Exports a GIF from the video using the chosen FPS. Lower FPS = smaller file.")
        lay_gif = QFormLayout();
        row_gfps = QHBoxLayout(); row_gfps.addWidget(self.gif_fps); row_gfps.addWidget(self.spin_gif_fps); lay_gif.addRow("GIF fps", row_gfps); lay_gif.addRow("", self.gif_same); row_gb = QHBoxLayout(); row_gb.addWidget(self.btn_gif); row_gb.addWidget(self.btn_gif_batch); lay_gif.addRow(row_gb)
        
        try:
            lay_gif.addRow("GIF fps (type)", self.spin_gif_fps)
        except Exception:
            pass
                # --- Advanced GIF options ---
        gif_backend.install_ui(self, lay_gif, sec_gif)
        sec_gif.setContentLayout(lay_gif)

        # Split & Glue Video (near frames tools)
        _sg_wrap = QWidget()
        _sg_layout = QVBoxLayout(_sg_wrap)
        _sg_layout.setContentsMargins(0, 0, 0, 0)
        try:
            _sg_widget = SpliglueVideoTool(self)
        except Exception:
            try:
                _sg_widget = SpliglueVideoTool(None)
            except Exception:
                _sg_widget = QLabel("Split & Glue tool failed to load.")
        _sg_layout.addWidget(_sg_widget)
        sec_splitglue.setContentLayout(_sg_layout)

        # Music Clip Creator (auto music sync) — embedded like other tools
        sec_musicclip = CollapsibleSection("VideoClip Creator", expanded=False)
        try:
            from helpers import auto_music_sync as _mcc
            # Preferred installer-style API
            if hasattr(_mcc, "install_auto_music_sync_tool"):
                _mcc.install_auto_music_sync_tool(self, sec_musicclip)
            elif hasattr(_mcc, "install_music_clip_creator"):
                _mcc.install_music_clip_creator(self, sec_musicclip)
            else:
                # Fallback to a direct widget, if provided
                _mcc_wrap = QWidget(); _mcc_layout = QVBoxLayout(_mcc_wrap); _mcc_layout.setContentsMargins(0,0,0,0)
                try:
                    widget_cls = getattr(_mcc, "MusicClipCreatorWidget", None) or getattr(_mcc, "MusicClipCreator", None)
                    if widget_cls is not None:
                        try:
                            _mcc_widget = widget_cls(self)
                        except Exception:
                            _mcc_widget = widget_cls(None)
                    else:
                        _mcc_widget = QLabel("Music Clip Creator tool loaded, but no UI entrypoint was found.")
                except Exception:
                    _mcc_widget = QLabel("Music Clip Creator tool failed to load.")
                _mcc_layout.addWidget(_mcc_widget)
                sec_musicclip.setContentLayout(_mcc_layout)
        except Exception:
            _mcc_wrap = QWidget(); _mcc_layout = QVBoxLayout(_mcc_wrap); _mcc_layout.setContentsMargins(0,0,0,0)
            _mcc_layout.addWidget(QLabel("Music Clip Creator tool failed to load."))
            sec_musicclip.setContentLayout(_mcc_layout)

        # Audio (moved to helpers/audiotool.py)
        sec_audio = CollapsibleSection("Sound Mixer", expanded=False)
        try:
            install_audio_tool(self, sec_audio)
        except Exception:
            pass

        # Music Edit
        sec_music = CollapsibleSection("Sound Edit", expanded=False)
        _music_wrap = QWidget(); _music_layout = QVBoxLayout(_music_wrap); _music_layout.setContentsMargins(0,0,0,0)
        try:
            _music_widget = MusicEditWidget(self)
        except Exception:
            try:
                _music_widget = MusicEditWidget(None)
            except Exception:
                _music_widget = QLabel("Music Edit tool failed to load.")
        _music_layout.addWidget(_music_widget)
        sec_music.setContentLayout(_music_layout)


        # moved below to reorder; see tuple later
        # root.addWidget(sec_audio)
        row = QHBoxLayout(); btn_sg = QPushButton("Save preset"); btn_lg = QPushButton("Load preset"); row.addWidget(btn_sg); row.addWidget(btn_lg)
        lay_gif.addRow(row)
        btn_sg.clicked.connect(lambda: self._save_preset_gif())
        btn_lg.clicked.connect(lambda: self._load_preset_gif())
        # Extract (moved to helpers/frames.py)
        try:
            install_frames_tool(self, sec_extract)
        except Exception:
            pass




        # Trim (moved to helpers/trim_tool.py)
        install_trim_tool(self, sec_trim)


        # Crop (moved to helpers/cropper.py)
        try:
            from helpers.cropper import install_cropper_tool
            install_cropper_tool(self, sec_crop)
        except Exception:
            pass
        default_sections = [sec_prompt, sec_bg, sec_ace, sec_describe, sec_meme, sec_music, sec_audio, sec_speed, sec_reverse, sec_upscale, sec_rife,
                            sec_resize, sec_trim, sec_crop, sec_splitglue, sec_gif, sec_extract, sec_rename, sec_metadata]

# sec_musicclip,  # to re add put this back in default_sections

        _sections = self._apply_tools_order(default_sections)

        # --- Reorder helpers (keyed by the visible header titles) ---
        # The remember-menu map uses "friendly" names that do not always match the section header text.
        # Drag/drop uses the header title, so we keep a separate map for reorder.
        try:
            self._reorder_sections = [s for s in (_sections or []) if isinstance(s, CollapsibleSection)]
        except Exception:
            self._reorder_sections = []
        self._reorder_by_title = {}
        for s in (self._reorder_sections or []):
            try:
                t = getattr(s, "title", None) or (s.toggle.text() if hasattr(s, "toggle") else "")
            except Exception:
                t = ""
            if t and t not in self._reorder_by_title:
                self._reorder_by_title[t] = s
        for sec in _sections:
            root.addWidget(sec)
        root.addStretch(1)
        # --- Remember settings (per-tool + global) ---
        def _sec_name_map():
            d = {
                "Sound Lab": sec_audio,
                "Music Edit": sec_music,
                "Music Clip Creator": sec_musicclip,
                "Slow motion - Speedup Video": sec_speed,
                "Reverse video": sec_reverse,
                "Upscale": sec_upscale,
                "Resize Images & Video": sec_resize,
                "Split & Glue Video": sec_splitglue,
                "Animated Frames Lab": sec_gif,
                "Extract frames": sec_extract,
                "Trim Lab": sec_trim,
                "Cropping": sec_crop,

                "Thumbnail / Meme Creator": sec_meme,
                "Prompt Enhancement": sec_prompt,
                "Background Remover": sec_bg,
                "Ace Step Music creation": sec_ace,
                "Multi Rename": sec_rename,
                "Whisper Lab": sec_whisper,
                "Metadata editor": sec_metadata
            }
            if sec_hunyuan15 is not None:
                # Keep it near prompt-related tools in the menu ordering.
                try:
                    _items = list(d.items())
                    _nd = {}
                    for _k, _v in _items:
                        _nd[_k] = _v
                        if _k == "Prompt Enhancement":
                            _nd["HunyuanVideo 1.5"] = sec_hunyuan15
                    d = _nd
                except Exception:
                    d["HunyuanVideo 1.5"] = sec_hunyuan15
            return d

        self._sections_map = _sec_name_map()
        # Apply current reorder mode to show grips + enable drop handling
        try:
            self._set_reorder_tools_enabled(getattr(self, "_reorder_tools_enabled", False))
        except Exception:
            pass
        # Build default whitelist (all except Trim and Audio)
        default_whitelist = [k for k in self._sections_map.keys() if k not in ("Trim Lab","Sound Lab")]
        try:
            import json as _json
            wl_txt = self._qs.value("ToolsPane/remember_whitelist_json", "", type=str) or ""
            self._remember_whitelist = set(_json.loads(wl_txt)) if wl_txt else set(default_whitelist)
        except Exception:
            self._remember_whitelist = set(default_whitelist)
        # populate menu
        try:
            if self.remember_menu:
                self.remember_menu.clear()
                for name in self._sections_map.keys():
                    if name in ("Trim Lab","Sound Lab"): 
                        continue
                    act = self.remember_menu.addAction(name)
                    act.setCheckable(True); act.setChecked(name in self._remember_whitelist)
                    act.toggled.connect(lambda on, nm=name: self._toggle_remember_tool(nm, on))
        except Exception:
            pass

        def _widget_snapshot(w):
            from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                                           QSpinBox, QDoubleSpinBox, QComboBox,
                                           QCheckBox, QRadioButton, QSlider,
                                           QDateEdit, QTimeEdit, QDateTimeEdit)
            from PySide6.QtCore import Qt
            classes = (QLineEdit, QPlainTextEdit, QTextEdit,
                       QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
                       QCheckBox, QRadioButton, QDateEdit, QTimeEdit, QDateTimeEdit)
            snap = []
            try:
                for cls in classes:
                    children = w.findChildren(cls)
                    for idx, child in enumerate(children):
                        name = child.objectName() or ""
                        try:
                            if isinstance(child, QLineEdit):
                                val = child.text()
                            elif isinstance(child, (QPlainTextEdit, QTextEdit)):
                                val = child.toPlainText()
                            elif isinstance(child, (QSpinBox, QSlider)):
                                val = int(child.value())
                            elif isinstance(child, QDoubleSpinBox):
                                val = float(child.value())
                            elif isinstance(child, QComboBox):
                                val = int(child.currentIndex())
                            elif isinstance(child, (QCheckBox, QRadioButton)):
                                val = bool(child.isChecked())
                            elif isinstance(child, QDateEdit):
                                val = child.date().toString(Qt.ISODate)
                            elif isinstance(child, QTimeEdit):
                                val = child.time().toString(Qt.ISODate)
                            elif isinstance(child, QDateTimeEdit):
                                val = child.dateTime().toString(Qt.ISODate)
                            else:
                                continue
                            snap.append((cls.__name__, idx, name, val))
                        except Exception:
                            continue
            except Exception:
                pass
            return snap

        def _widget_restore(w, snap):
            try:
                from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                                               QSpinBox, QDoubleSpinBox, QComboBox,
                                               QCheckBox, QRadioButton, QSlider,
                                               QDateEdit, QTimeEdit, QDateTimeEdit)
                from PySide6.QtCore import Qt, QDate, QTime, QDateTime
            except Exception:
                return
            cls_map = {"QLineEdit":QLineEdit, "QPlainTextEdit":QPlainTextEdit, "QTextEdit":QTextEdit,
                       "QSpinBox":QSpinBox, "QDoubleSpinBox":QDoubleSpinBox, "QSlider":QSlider,
                       "QComboBox":QComboBox, "QCheckBox":QCheckBox, "QRadioButton":QRadioButton,
                       "QDateEdit":QDateEdit, "QTimeEdit":QTimeEdit, "QDateTimeEdit":QDateTimeEdit}
            for item in (snap or []):
                try:
                    cls_name, idx, name, val = item
                except Exception:
                    continue
                cls = cls_map.get(str(cls_name))
                if not cls:
                    continue
                target = None
                if name:
                    target = w.findChild(cls, name)
                if target is None:
                    children = w.findChildren(cls)
                    try:
                        idx_i = int(idx)
                    except Exception:
                        idx_i = -1
                    if 0 <= idx_i < len(children):
                        target = children[idx_i]
                if target is None:
                    continue
                try:
                    if isinstance(target, QLineEdit):
                        target.setText(str(val))
                    elif isinstance(target, (QPlainTextEdit, QTextEdit)):
                        target.setPlainText(str(val))
                    elif isinstance(target, (QSpinBox, QSlider)):
                        target.setValue(int(val))
                    elif isinstance(target, QDoubleSpinBox):
                        target.setValue(float(val))
                    elif isinstance(target, QComboBox):
                        target.setCurrentIndex(int(val))
                    elif isinstance(target, (QCheckBox, QRadioButton)):
                        target.setChecked(bool(val))
                    elif isinstance(target, QDateEdit):
                        d = QDate.fromString(str(val), Qt.ISODate)
                        if d.isValid(): target.setDate(d)
                    elif isinstance(target, QTimeEdit):
                        t = QTime.fromString(str(val), Qt.ISODate)
                        if t.isValid(): target.setTime(t)
                    elif isinstance(target, QDateTimeEdit):
                        dt = QDateTime.fromString(str(val), Qt.ISODate)
                        if dt.isValid(): target.setDateTime(dt)
                except Exception:
                    continue

        def _save_all_tools():
            try:
                import json as _json
                data = {"sections": {}}
                for name, sec in self._sections_map.items():
                    try:
                        expanded = bool(getattr(sec, "isChecked", lambda: True)())
                    except Exception:
                        expanded = True
                    content = getattr(sec, "content", sec)
                    data["sections"][name] = {"expanded": bool(expanded), "snap": _widget_snapshot(content)}
                self._qs.setValue("ToolsPane/saved_json", _json.dumps(data))
            except Exception:
                pass

        def _apply_saved():
            try:
                import json as _json
                txt = self._qs.value("ToolsPane/saved_json", "", type=str) or ""
                if not txt:
                    return
                data = _json.loads(txt)
                for name, sec in self._sections_map.items():
                    entry = (data.get("sections") or {}).get(name)
                    if not entry:
                        continue
                    snap = entry.get("snap")
                    if snap:
                        content = getattr(sec, "content", sec)
                        _widget_restore(content, snap)
                    if "expanded" in entry:
                        try:
                            sec.setChecked(bool(entry["expanded"]))
                        except Exception:
                            pass
            except Exception:
                pass

        def _autosave():
            """Debounced autosave.
            Creates a single QTimer once and restarts it on every call.
            When the timer fires, it performs one _save_all_tools() and stops.
            This avoids spawning/connecting new timers repeatedly.
            """
            try:
                if getattr(self, '_autosave_timer', None) is None:
                    self._autosave_timer = QTimer(self)
                    try:
                        # Use a coarse timer on Windows to reduce WndProc load
                        self._autosave_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
                    except Exception:
                        pass
                    self._autosave_timer.setSingleShot(True)
                    self._autosave_timer.setInterval(1500)
                    self._autosave_timer.timeout.connect(lambda: _save_all_tools())
                # debounce restart
                self._autosave_timer.start()
            except Exception:
                # Fallback: save immediately if timer can't start
                try:
                    _save_all_tools()
                except Exception:
                    pass

        def _wire_watchers():
            try:
                from PySide6.QtWidgets import (QLineEdit, QPlainTextEdit, QTextEdit,
                    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton,
                    QSlider, QDateEdit, QTimeEdit, QDateTimeEdit)
            except Exception:
                return
            def _wire(root):
                try:
                    for w in root.findChildren(QLineEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QPlainTextEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QTextEdit): w.textChanged.connect(_autosave)
                    for w in root.findChildren(QSpinBox): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDoubleSpinBox): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QComboBox): w.currentIndexChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QCheckBox): w.toggled.connect(lambda *_: _autosave())
                    for w in root.findChildren(QRadioButton): w.toggled.connect(lambda *_: _autosave())
                    for w in root.findChildren(QSlider): w.valueChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDateEdit): w.dateChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QTimeEdit): w.timeChanged.connect(lambda *_: _autosave())
                    for w in root.findChildren(QDateTimeEdit): w.dateTimeChanged.connect(lambda *_: _autosave())
                except Exception:
                    pass
            try:
                for _name, _sec in self._sections_map.items():
                    _root = getattr(_sec, 'content', _sec)
                    _wire(_root)
            except Exception:
                pass
        def _toggle_remember_tool(name, on):
            try:
                if on: self._remember_whitelist.add(name)
                else: self._remember_whitelist.discard(name)
                import json as _json
                self._qs.setValue("ToolsPane/remember_whitelist_json", _json.dumps(sorted(self._remember_whitelist)))
                _save_all_tools()
            except Exception:
                pass
        self._toggle_remember_tool = _toggle_remember_tool

        # Apply saved settings on init
        try:
            _apply_saved()
        except Exception:
            pass

        try:
            _wire_watchers()
            _save_all_tools()
        except Exception:
            pass
        # removed quality/img tool wiring
        # removed quality/img tool wiring
        # removed: batch_folder button
        # removed quality/img tool wiring
        # removed quality/img tool wiring
        # removed: batch_folder button
# Wire
        self.btn_reverse_load.clicked.connect(self._reverse_load_video)
        self.btn_reverse.clicked.connect(self.run_reverse)
        self.btn_reverse_batch.clicked.connect(self.run_reverse_batch)
        try:
            self.btn_reverse_open_folder.clicked.connect(self._reverse_open_folder)
        except Exception:
            pass
        self.btn_speed.clicked.connect(self.run_speed)
        self.btn_speed_batch.clicked.connect(self.run_speed_batch)
        self.btn_gif.clicked.connect(self.run_gif)
        self.btn_gif_batch.clicked.connect(self.run_gif_batch)

        self.btn_trim.clicked.connect(self.run_trim)
        # Let the Trim tool own the batch popup; wire here only if provided.
        try:
            if hasattr(self, 'run_trim_batch'):
                self.btn_trim_batch.clicked.connect(self.run_trim_batch)
        except Exception:
            pass    # Helpers and actions stay the same as your previous version
    

    # ------------------------------
    # Tools tab auto-arrange width guard
    # ------------------------------
    def _install_tools_auto_width_guard(self):
        """Prevent the Tools tab scroll area from drifting horizontally off-screen."""
        try:
            sa = getattr(self, "tools_scroll", None)
            host = getattr(self, "_tools_scroll_host", None)
            if sa is None or host is None:
                return

            # Hide horizontal bar and force its value to 0 if anything nudges it.
            try:
                sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                hbar = sa.horizontalScrollBar()
                hbar.valueChanged.connect(self._on_tools_hscroll_changed)
            except Exception:
                pass

            # Track viewport changes so the host always stays at least as wide as the viewport.
            try:
                vp = sa.viewport()
                if vp is not None:
                    vp.installEventFilter(self)
            except Exception:
                pass

            QTimer.singleShot(0, self._sync_tools_host_width)
            QTimer.singleShot(0, self._force_tools_scroll_left)
        except Exception:
            pass

    # ------------------------------
    # Tools tab: reorder tools (drag via grip/handle)
    # ------------------------------
    def _apply_tools_order(self, default_sections):
        """Return sections ordered by saved user order, with new tools appended."""
        secs = [s for s in (default_sections or []) if s is not None]
        # Map by visible title
        by_title = {}
        default_titles = []
        for s in secs:
            try:
                t = getattr(s, "title", None) or (s.toggle.text() if hasattr(s, "toggle") else "")
            except Exception:
                t = ""
            by_title[t] = s
            default_titles.append(t)

        saved = []
        try:
            import json as _json
            txt = self._qs.value("ToolsPane/tools_order_json", "", type=str) or ""
            if txt:
                saved = _json.loads(txt)
                if not isinstance(saved, list):
                    saved = []
        except Exception:
            saved = []

        ordered = []
        seen = set()
        for t in saved:
            if t in by_title and t not in seen:
                ordered.append(by_title[t])
                seen.add(t)
        for t in default_titles:
            if t in by_title and t not in seen:
                ordered.append(by_title[t])
                seen.add(t)
        return ordered

    def _save_tools_order(self):
        """Persist current UI order."""
        try:
            import json as _json
            lay = getattr(self, "_tools_root_layout", None)
            if lay is None:
                return
            titles = []
            for i in range(lay.count()):
                it = lay.itemAt(i)
                w = it.widget() if it is not None else None
                if w is not None and isinstance(w, CollapsibleSection):
                    try:
                        t = getattr(w, "title", None) or (w.toggle.text() if hasattr(w, "toggle") else "")
                    except Exception:
                        t = ""
                    if t:
                        titles.append(t)
            self._qs.setValue("ToolsPane/tools_order_json", _json.dumps(titles))
        except Exception:
            pass

    def _set_reorder_tools_enabled(self, on: bool):
        self._reorder_tools_enabled = bool(on)
        try:
            self._qs.setValue("ToolsPane/reorder_tools_enabled", bool(self._reorder_tools_enabled))
        except Exception:
            pass

        try:
            host = getattr(self, "_tools_reorder_host", None)
            if host is not None:
                host.setReorderEnabled(self._reorder_tools_enabled)
        except Exception:
            pass

        try:
            secs = getattr(self, "_reorder_sections", None)
            if isinstance(secs, (list, tuple)) and secs:
                for sec in secs:
                    try:
                        sec.setReorderEnabled(self._reorder_tools_enabled)
                    except Exception:
                        pass
            else:
                # Fallback (older builds)
                smap = getattr(self, "_sections_map", None)
                if isinstance(smap, dict):
                    for sec in smap.values():
                        try:
                            sec.setReorderEnabled(self._reorder_tools_enabled)
                        except Exception:
                            pass
        except Exception:
            pass

    def _sync_tools_host_width(self):
        try:
            sa = getattr(self, "tools_scroll", None)
            host = getattr(self, "_tools_scroll_host", None)
            if sa is None or host is None:
                return
            vp = sa.viewport()
            if vp is None:
                return
            w = int(vp.width())
            if w > 0 and int(host.minimumWidth()) != w:
                host.setMinimumWidth(w)
        except Exception:
            pass

    def _force_tools_scroll_left(self):
        try:
            sa = getattr(self, "tools_scroll", None)
            if sa is None:
                return
            sa.horizontalScrollBar().setValue(0)
        except Exception:
            pass

    def _on_tools_hscroll_changed(self, v):
        try:
            if int(v) != 0:
                self._force_tools_scroll_left()
        except Exception:
            pass

    def eventFilter(self, obj, ev):
        try:
            sa = getattr(self, "tools_scroll", None)
            if sa is not None and obj == sa.viewport():
                et = ev.type()
                if et == QEvent.Resize:
                    QTimer.singleShot(0, self._sync_tools_host_width)
                    QTimer.singleShot(0, self._force_tools_scroll_left)
                elif et == QEvent.Wheel:
                    # Trackpads can sneak in horizontal deltas; clamp after the event.
                    QTimer.singleShot(0, self._force_tools_scroll_left)
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    def _pick_batch_files(self, for_videos=True):
        try:
            from PySide6.QtWidgets import QFileDialog
            import os
            files, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Media files (*.*)")
            paths = [p for p in files or [] if os.path.isfile(p)]
            if not paths:
                folder = QFileDialog.getExistingDirectory(self, "Or select folder")
                if folder:
                    exts_vid = {".mp4",".mov",".mkv",".webm",".avi"}
                    exts_img = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
                    for name in os.listdir(folder):
                        p=os.path.join(folder,name); ext=os.path.splitext(name)[1].lower()
                        if os.path.isfile(p) and ((for_videos and ext in exts_vid) or ((not for_videos) and ext in exts_img)):
                            paths.append(p)
            return paths
        except Exception:
            return []

    def _ensure_input(self, allow_images: bool=False):
        if not self.main.current_path:
            QMessageBox.warning(self, "No file", "Open a video first.");
            return None
        return self.main.current_path

    def _run(self, cmd_list, outfile):
        # Enqueue instead of executing immediately
        inp = self._ensure_input()
        if not inp:
            return
        from pathlib import Path as _P
        out_path = outfile if isinstance(outfile, _P) else _P(str(outfile))
        try:
            try:
                from helpers.queue_adapter import enqueue_tool_job as enq
            except Exception:
                from queue_adapter import enqueue_tool_job as enq
            jid = enq("tools_ffmpeg", str(inp), str(out_path.parent), {"ffmpeg_cmd": cmd_list, "outfile": str(out_path)}, priority=600)
            try:
                QMessageBox.information(self, "Queued", f"Job queued to jobs/pending.\n\n{jid}")
            except Exception:
                pass
            # Switch to Queue tab
            try:
                win = self.window()
                tabs = win.findChild(QTabWidget)
                if tabs:
                    for i in range(tabs.count()):
                        if tabs.tabText(i).strip().lower() == "queue":
                            tabs.setCurrentIndex(i); break
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.warning(self, "Queue error", str(e))
            except Exception:
                pass
        return



    def run_speed(self):
        inp = self._ensure_input()
        if not inp:
            return
        factor = self.speed.value() / 100.0
        try:
            out_dir = OUT_VIDEOS
        except Exception:
            out_dir = Path('.')
        out = out_dir / f"{inp.stem}_spd_{factor:.2f}x.mp4"
        setpts = 1.0 / float(factor if factor != 0 else 1.0)
        mute = False
        sync = True
        try:
            mute = bool(self.cb_speed_mute.isChecked())
        except Exception:
            pass
        try:
            sync = bool(self.cb_speed_sync.isChecked())
        except Exception:
            pass
        cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", f"setpts={setpts:.6f}*PTS"]
        if mute:
            cmd += ["-an"]
        else:
            if sync:
                atempos = []
                f = float(factor)
                while f > 2.0:
                    atempos.append("2.0"); f /= 2.0
                while f < 0.5:
                    atempos.append("0.5"); f *= 2.0
                atempos.append(f"{f:.3f}")
                cmd += ["-filter:a", "atempo=" + ",".join(atempos)]
            else:
                cmd += ["-c:a", "copy", "-shortest"]
        # Match source bitrate so output keeps similar quality/size profile
        br_flags = []
        try:
            from pathlib import Path as _P
            info = probe_media(_P(str(inp)))
            dur = float(info.get("duration") or 0.0)
            size_b = float(info.get("size") or 0.0)
            if dur > 0.0 and size_b > 0.0:
                br_kbps = int((size_b * 8.0) / (dur * 1000.0))
                if br_kbps < 300:
                    br_kbps = 300
                if br_kbps > 50000:
                    br_kbps = 50000
                br_flags = [
                    "-b:v", f"{br_kbps}k",
                    "-maxrate", f"{br_kbps}k",
                    "-bufsize", f"{br_kbps * 2}k",
                ]
        except Exception:
            br_flags = []

        cmd += ["-c:v", "libx264", "-preset", "veryfast"]
        if br_flags:
            cmd += br_flags
        cmd += ["-movflags", "+faststart", str(out)]
        self._run(cmd, out)

    
    def run_gif(self):
        inp = self._ensure_input()
        if not inp:
            return
        fmt = self.gif_fmt.currentText().strip().lower() if hasattr(self, "gif_fmt") else "gif"
        opts = gif_backend.options_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self, "gif_fps") else 0, batch=False)
        out = OUT_VIDEOS / gif_backend.output_name_for(inp.stem, fmt, batch=False)
        # Progress + ETA like interp.py
        try:
            gif_backend.encode_with_progress(self, inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
        except Exception:
            gif_backend.encode(inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
        # Remember output to enable "Play last" button in the GIF UI
        try:
            self._gif_last_out = out
        except Exception:
            pass
        inp = self._ensure_input();
        if not inp: return
        out=OUT_VIDEOS / f"{inp.stem}_lastframe.png"
        cmd=[ffmpeg_path(),"-y","-sseof","-1","-i",str(inp),"-update","1","-frames:v","1",str(out)]
        self._run(cmd,out)
        inp = self._ensure_input();
        if not inp: return

    def _reverse_set_input(self, path):
        """Remember reverse tool input and update the info label."""
        try:
            from pathlib import Path as _P
            p = _P(str(path)) if path else None
        except Exception:
            p = None
        try:
            self._reverse_input_path = p
        except Exception:
            pass
        try:
            label = getattr(self, "lbl_reverse_info", None)
        except Exception:
            label = None
        if label is None:
            return
        if not p or not p.exists():
            try:
                label.setText("Using main Media Player video. No separate file loaded.")
            except Exception:
                pass
            return
        info = probe_media(p)
        try:
            name = p.name
        except Exception:
            name = str(p)
        parts = []
        try:
            w = info.get("width")
            h = info.get("height")
            if w and h:
                parts.append(f"{w}x{h}")
        except Exception:
            pass
        try:
            dur = info.get("duration")
            if dur is not None:
                parts.append(self._fmt_duration(dur))
        except Exception:
            pass
        try:
            size_b = info.get("size")
            if size_b is not None:
                parts.append(self._fmt_size(size_b))
        except Exception:
            pass
        summary = " • ".join(parts) if parts else "No details"
        try:
            label.setText(f"{name} — {summary}")
        except Exception:
            pass

    def _reverse_get_input(self):
        """Prefer a manually loaded video; fall back to the main Media Player video."""
        try:
            p = getattr(self, "_reverse_input_path", None)
        except Exception:
            p = None
        if p:
            try:
                from pathlib import Path as _P
                return _P(str(p))
            except Exception:
                pass
        return self._ensure_input()

    def _reverse_load_video(self):
        """Pick a video file to use just for the Reverse tool."""
        try:
            from PySide6.QtWidgets import QFileDialog
        except Exception:
            QFileDialog = None
        if QFileDialog is None:
            return
        exts = "Videos (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.mpg *.mpeg *.ts *.m2ts)"
        files, _ = QFileDialog.getOpenFileNames(self, "Choose video for Reverse tool", "", exts)
        if not files:
            return
        try:
            path = files[0]
        except Exception:
            return
        self._reverse_set_input(path)

    def _fmt_duration(self, seconds):
        try:
            s = int(round(float(seconds)))
            h = s // 3600
            m = (s % 3600) // 60
            s = s % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            return "?"

    def _fmt_size(self, size_bytes):
        try:
            b = float(size_bytes)
        except Exception:
            return "?"
        units = ["B","KB","MB","GB","TB"]
        i = 0
        while b >= 1024 and i < len(units)-1:
            b /= 1024.0
            i += 1
        try:
            return f"{b:.1f} {units[i]}"
        except Exception:
            return f"{int(b)} {units[i]}"

    def _open_folder_in_os(self, folder_path):
        """Open a folder in the system file manager (Explorer/Finder/etc.)."""
        try:
            from pathlib import Path as _P
            fp = _P(str(folder_path))
        except Exception:
            fp = None
        if fp is None:
            return
        try:
            fp.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if os.name == "nt":
                os.startfile(str(fp))  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        # macOS / Linux fallback
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(fp)])
            else:
                subprocess.Popen(["xdg-open", str(fp)])
        except Exception as e:
            try:
                QMessageBox.warning(self, "Open folder", str(e))
            except Exception:
                pass

    def _reverse_open_folder(self):
        """Open Reverse tool results in Media Explorer (fallback: OS folder)."""
        folder = None
        try:
            folder = getattr(self, "_reverse_last_out_dir", None)
        except Exception:
            folder = None
        if not folder:
            try:
                folder = OUT_REVERSE
            except Exception:
                try:
                    folder = OUT_VIDEOS
                except Exception:
                    folder = Path(".")
        try:
            from pathlib import Path as _P
            fp = folder if isinstance(folder, _P) else _P(str(folder))
            try:
                fp.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        except Exception:
            fp = None

        # Prefer Media Explorer (single shared entry-point) if available on the main window.
        main = None
        try:
            main = getattr(self, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = self.window() if hasattr(self, "window") else None
            except Exception:
                main = None

        if main is not None and hasattr(main, "open_media_explorer_folder") and fp is not None:
            try:
                main.open_media_explorer_folder(str(fp), preset="videos", include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(fp))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: open in OS file browser
        if fp is not None:
            self._open_folder_in_os(fp)

    def run_reverse(self):
        inp = self._reverse_get_input()
        if not inp:
            return
        # Ensure info label is in sync with the file we are actually using
        try:
            self._reverse_set_input(inp)
        except Exception:
            pass
        from pathlib import Path as _P
        try:
            out_dir = OUT_REVERSE
        except Exception:
            try:
                out_dir = OUT_VIDEOS
            except Exception:
                out_dir = _P(".")
        try:
            _P(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        inp_path = _P(str(inp))
        out = _P(out_dir) / f"{inp_path.stem}_reversed.mp4"

        try:
            self._reverse_last_out_dir = _P(str(out)).parent
        except Exception:
            pass

        # Options
        mute = False
        try:
            mute = bool(self.cb_reverse_mute.isChecked())
        except Exception:
            pass

        audio_ok = False
        try:
            if not mute:
                audio_ok = bool(has_audio(inp_path))
        except Exception:
            audio_ok = False
        boomerang = False
        loops = 1
        try:
            boomerang = bool(self.cb_reverse_boom.isChecked())
            if boomerang:
                try:
                    loops = int(self.spin_reverse_loops.value())
                except Exception:
                    loops = 1
        except Exception:
            boomerang = False
        if loops < 1:
            loops = 1
        if loops > 9:
            loops = 9

        # Try to approximate source bitrate so reversed output keeps similar quality/size
        br_flags = []
        try:
            info = probe_media(inp_path)
            dur = float(info.get("duration") or 0.0)
            size_b = float(info.get("size") or 0.0)
            if dur > 0.0 and size_b > 0.0:
                br_kbps = int((size_b * 8.0) / (dur * 1000.0))
                if br_kbps < 300:
                    br_kbps = 300
                if br_kbps > 50000:
                    br_kbps = 50000
                br_flags = [
                    "-b:v", f"{br_kbps}k",
                    "-maxrate", f"{br_kbps}k",
                    "-bufsize", f"{br_kbps * 2}k",
                ]
        except Exception:
            br_flags = []

        ff = ffmpeg_path()
        cmd = [ff, "-y", "-i", str(inp_path)]

        if boomerang and loops > 0:
            # Forward + reverse boomerang (repeatable): duplicate streams with split/asplit so FFmpeg
            # doesn't "consume" [fwd]/[rev] once and then collapse to a single direction.
            segments = 2 * loops
            with_audio = (not mute) and audio_ok

            parts = []
            parts.append("[0:v]split=2[fv0][vtmp]")
            parts.append("[vtmp]reverse[rv0]")

            if loops > 1:
                fvouts = "".join(f"[fv{i}]" for i in range(loops))
                rvouts = "".join(f"[rv{i}]" for i in range(loops))
                parts.append(f"[fv0]split={loops}{fvouts}")
                parts.append(f"[rv0]split={loops}{rvouts}")
                v_inter = "".join(f"[fv{i}][rv{i}]" for i in range(loops))
            else:
                v_inter = "[fv0][rv0]"

            parts.append(f"{v_inter}concat=n={segments}:v=1:a=0[vout]")

            if with_audio:
                parts.append("[0:a]asplit=2[fa0][atmp]")
                parts.append("[atmp]areverse[ra0]")

                if loops > 1:
                    faouts = "".join(f"[fa{i}]" for i in range(loops))
                    raouts = "".join(f"[ra{i}]" for i in range(loops))
                    parts.append(f"[fa0]asplit={loops}{faouts}")
                    parts.append(f"[ra0]asplit={loops}{raouts}")
                    a_inter = "".join(f"[fa{i}][ra{i}]" for i in range(loops))
                else:
                    a_inter = "[fa0][ra0]"

                parts.append(f"{a_inter}concat=n={segments}:v=0:a=1[aout]")

            vf = ";".join(parts)
            cmd += ["-filter_complex", vf, "-map", "[vout]"]

            if mute or (not audio_ok):
                cmd += ["-an"]
            else:
                cmd += ["-map", "[aout]"]
        else:
            # Simple full reverse
            cmd += ["-vf", "reverse"]
            if mute:
                cmd += ["-an"]
            else:
                if audio_ok:
                    cmd += ["-af", "areverse"]

        cmd += ["-c:v", "libx264", "-preset", "veryfast"]
        if br_flags:
            cmd += br_flags
        cmd += ["-movflags", "+faststart", str(out)]

        # Use the selected input path directly so manually loaded videos work even if the main player is empty
        ok = False
        try:
            ok = bool(self._enqueue_cmd_for_input(inp_path, cmd, out))
        except Exception:
            ok = False
        if not ok:
            # Fallback to legacy path that uses the main Media Player input
            self._run(cmd, out)




    def run_reverse_batch(self):
        """Batch reverse multiple videos using current Reverse settings."""
        try:
            paths = self._batch_paths_with_dialog(title="Batch Reverse", exts=BatchSelectDialog.VIDEO_EXTS)
        except Exception:
            paths = []
        if not paths:
            return
        # Confirm with user (optional)
        try:
            if QMessageBox.question(
                self,
                "Batch Reverse",
                f"Add {len(paths)} video(s) with current Reverse settings to the queue?"
            ) != QMessageBox.Yes:
                return
        except Exception:
            pass

        mute = False
        try:
            mute = bool(self.cb_reverse_mute.isChecked())
        except Exception:
            pass
        boomerang = False
        loops = 1
        try:
            boomerang = bool(self.cb_reverse_boom.isChecked())
            if boomerang:
                try:
                    loops = int(self.spin_reverse_loops.value())
                except Exception:
                    loops = 1
        except Exception:
            boomerang = False
        if loops < 1:
            loops = 1
        if loops > 9:
            loops = 9

        from pathlib import Path as _P

        for p in paths:
            try:
                inp = _P(str(p))
            except Exception:
                continue
            if not inp.exists():
                continue
            try:
                out_dir = OUT_REVERSE
            except Exception:
                try:
                    out_dir = OUT_VIDEOS
                except Exception:
                    out_dir = _P(".")
            try:
                _P(out_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            out = _P(out_dir) / f"{inp.stem}_reversed.mp4"

            try:
                self._reverse_last_out_dir = _P(str(out)).parent
            except Exception:
                pass

            # Bitrate estimation per input file
            br_flags = []
            try:
                info = probe_media(inp)
                dur = float(info.get("duration") or 0.0)
                size_b = float(info.get("size") or 0.0)
                if dur > 0.0 and size_b > 0.0:
                    br_kbps = int((size_b * 8.0) / (dur * 1000.0))
                    if br_kbps < 300:
                        br_kbps = 300
                    if br_kbps > 50000:
                        br_kbps = 50000
                    br_flags = [
                        "-b:v", f"{br_kbps}k",
                        "-maxrate", f"{br_kbps}k",
                        "-bufsize", f"{br_kbps * 2}k",
                    ]
            except Exception:
                br_flags = []

            ff = ffmpeg_path()
            cmd = [ff, "-y", "-i", str(inp)]

            audio_ok = False
            try:
                if not mute:
                    audio_ok = bool(has_audio(inp))
            except Exception:
                audio_ok = False

            if boomerang and loops > 0:
                # Forward + reverse boomerang (repeatable): duplicate streams with split/asplit
                # so FFmpeg doesn't reuse a consumed label and end up repeating one direction.
                segments = 2 * loops
                with_audio = (not mute) and audio_ok

                parts = []
                parts.append("[0:v]split=2[fv0][vtmp]")
                parts.append("[vtmp]reverse[rv0]")

                if loops > 1:
                    fvouts = "".join(f"[fv{i}]" for i in range(loops))
                    rvouts = "".join(f"[rv{i}]" for i in range(loops))
                    parts.append(f"[fv0]split={loops}{fvouts}")
                    parts.append(f"[rv0]split={loops}{rvouts}")
                    v_inter = "".join(f"[fv{i}][rv{i}]" for i in range(loops))
                else:
                    v_inter = "[fv0][rv0]"

                parts.append(f"{v_inter}concat=n={segments}:v=1:a=0[vout]")

                if with_audio:
                    parts.append("[0:a]asplit=2[fa0][atmp]")
                    parts.append("[atmp]areverse[ra0]")

                    if loops > 1:
                        faouts = "".join(f"[fa{i}]" for i in range(loops))
                        raouts = "".join(f"[ra{i}]" for i in range(loops))
                        parts.append(f"[fa0]asplit={loops}{faouts}")
                        parts.append(f"[ra0]asplit={loops}{raouts}")
                        a_inter = "".join(f"[fa{i}][ra{i}]" for i in range(loops))
                    else:
                        a_inter = "[fa0][ra0]"

                    parts.append(f"{a_inter}concat=n={segments}:v=0:a=1[aout]")

                vf = ";".join(parts)
                cmd += ["-filter_complex", vf, "-map", "[vout]"]

                if mute or (not audio_ok):
                    cmd += ["-an"]
                else:
                    cmd += ["-map", "[aout]"]
            else:
                cmd += ["-vf", "reverse"]
                if mute:
                    cmd += ["-an"]
                else:
                    if audio_ok:
                        cmd += ["-af", "areverse"]

            cmd += ["-c:v", "libx264", "-preset", "veryfast"]
            if br_flags:
                cmd += br_flags
            cmd += ["-movflags", "+faststart", str(out)]
            try:
                self._enqueue_cmd_for_input(inp, cmd, out)
            except Exception:
                continue


    def run_trim(self):
        inp = self._ensure_input();
        if not inp: return
        start=self.trim_start.text().strip(); end=self.trim_end.text().strip()
        out=OUT_TRIMS / f"{inp.stem}_trim_{start.replace(':','-').replace('.','_')}_{(end or 'end').replace(':','-').replace('.','_')}.mp4"
        if self.trim_mode.currentIndex()==0:
            cmd=[ffmpeg_path(),"-y"]+ (["-ss",start] if start else []) + (["-to",end] if end else []) + ["-i",str(inp),"-c","copy","-movflags","+faststart",str(out)]
        else:
            cmd=[ffmpeg_path(),"-y"] + (["-ss",start] if start else []) + ["-i",str(inp)] + (["-to",end] if end else []) + ["-c:v","libx264","-preset","veryfast","-c:a","aac","-movflags","+faststart",str(out)]
        self._run(cmd,out)
    # ---- Presets helpers ----
    def _preset_dir(self):
        try:
            base = Path(config.get("last_preset_dir", str(ROOT / "presets" / "Tools")))
        except Exception:
            base = ROOT / "presets" / "Tools"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _choose_save_path(self, suggested_name):
        d = self._preset_dir()
        path = d / suggested_name
        fn, _ = QFileDialog.getSaveFileName(self, "Save preset", str(path), "FrameVision Preset (*.json)")
        if not fn:
            return None
        p = Path(fn)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        # overwrite confirmation
        if p.exists():
            res = QMessageBox.question(self, "Overwrite?", f"{p.name} already exists. Overwrite?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if res != QMessageBox.Yes:
                return None
        # remember folder
        try:
            config["last_preset_dir"] = str(p.parent)
            save_config()
        except Exception:
            pass
        return p

    def _choose_open_path(self):
        d = self._preset_dir()
        fn, _ = QFileDialog.getOpenFileName(self, "Load preset", str(d), "FrameVision Preset (*.json)")
        if not fn:
            return None
        p = Path(fn)
        try:
            config["last_preset_dir"] = str(p.parent)
            save_config()
        except Exception:
            pass
        return p

    # ---- Speed ----
    def _save_preset_speed(self):
        factor = self.speed.value()/100.0
        name = f"speed_{int(round(factor*100))}pct_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data = {"tool":"speed","factor":factor}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_speed(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="speed": raise ValueError("Wrong preset type")
            factor = float(data.get("factor", 1.0))
            self.speed.setValue(int(round(factor*100)))
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Resize ----
    def _save_preset_gif(self):
        inp = self._ensure_input(silent=True)
        if not inp:
            # we can still save UI-only presets without an input
            inp = Path('dummy.mp4')
        name = 'gif_preset.json'
        p = self._choose_save_path(name)
        if not p:
            return
        data = gif_backend.preset_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self,'gif_fps') else 0)
        p.write_text(json.dumps(data, indent=2), encoding='utf-8')
        try:
            QMessageBox.information(self, 'Preset saved', str(p))
        except Exception:
            pass

    def _load_preset_gif(self):
        p = self._choose_open_path()
        if not p:
            return
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            gif_backend.apply_preset(self, data)
            try:
                QMessageBox.information(self, 'Preset loaded', str(p))
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.critical(self, 'Preset error', str(e))
            except Exception:
                pass
        name = "extract_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data={"tool":"extract"}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="extract": raise ValueError("Wrong preset type")
            QMessageBox.information(self, "Preset", "Loaded extract preset (no parameters).")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Trim ----
    def _save_preset_trim(self):
        start=self.trim_start.text().strip(); end=self.trim_end.text().strip()
        mode="copy" if self.trim_mode.currentIndex()==0 else "encode"
        def safe(s):
            return (s or "none").replace(":", "-").replace(".", "_").replace(" ", "")
        name = f"trim_{safe(start)}_{safe(end)}_{mode}_preset.json"
        p = self._choose_save_path(name)
        if not p: return
        data={"tool":"trim","mode":mode,"start":start,"end":end}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Preset saved", str(p))

    def _load_preset_trim(self):
        p = self._choose_open_path()
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool")!="trim": raise ValueError("Wrong preset type")
            mode=data.get("mode","copy"); self.trim_mode.setCurrentIndex(0 if mode=="copy" else 1)
            self.trim_start.setText(data.get("start",""))
            self.trim_end.setText(data.get("end",""))
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ---- Crop ----
    def _map_preset(self, enc: str, preset_txt: str) -> list[str]:
        # Map generic preset to encoder-specific flags
        p = preset_txt.lower()
        if enc in ("libx264","libx265"):
            return ["-preset", p]
        if enc.endswith("_nvenc"):
            # map to p1..p7; medium->p5
            m = {"ultrafast":"p1","superfast":"p2","veryfast":"p3","faster":"p4","fast":"p4","medium":"p5","slow":"p6","slower":"p6","veryslow":"p7"}
            return ["-preset", m.get(p, "p5")]
        if enc == "libsvtav1":
            # 0 best quality/slowest .. 13 fastest; map around medium=6
            m = {"ultrafast":"12","superfast":"11","veryfast":"10","faster":"9","fast":"8","medium":"6","slow":"4","slower":"3","veryslow":"2"}
            return ["-preset", m.get(p, "6")]
        return []

    def _audio_flags(self):
        a = (self.q_audio.currentText() if hasattr(self, 'q_audio') else "copy").lower()
        if a.startswith("copy"):
            return ["-c:a","copy"]
        if "opus" in a:
            kb = re.findall(r"(\d+)", a)
            return ["-c:a","libopus","-b:a", f"{kb[0]}k" if kb else "128k"]
        # default aac
        kb = re.findall(r"(\d+)", a)
        return ["-c:a","aac","-b:a", f"{kb[0]}k" if kb else "128k"]

    def _build_quality_cmd_for(self, inp_path):
        from pathlib import Path as _P
        enc, short = self._codec_selected()
        fmt = (self.q_format.currentText() if hasattr(self,'q_format') else "mp4").lower()
        mode = 0 if (hasattr(self,'q_mode') and self.q_mode.currentIndex()==0) else 1
        crf = int(self.q_crf.value()) if hasattr(self,'q_crf') else 23
        br = int(self.q_bitrate.value()) if hasattr(self,'q_bitrate') else 3500
        preset = self._map_preset(enc, self.q_preset.currentText() if hasattr(self,'q_preset') else "medium")

        inp = _P(inp_path)
        suf = (f"{short}_crf{crf}" if mode==0 else f"{short}_b{br}k")
        out = (OUT_VIDEOS if 'OUT_VIDEOS' in globals() else _P('.')) / f"{inp.stem}_{suf}.{fmt}"
        cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-c:v", enc]

        # presets
        cmd += preset

        if mode == 0:
            if enc.endswith("_nvenc"):
                # CQ mode for NVENC
                cmd += ["-rc","vbr","-cq", str(crf), "-b:v","0"]
            elif enc == "libsvtav1":
                cmd += ["-crf", str(crf)]
            else:
                cmd += ["-crf", str(crf)]
        else:
            cmd += ["-b:v", f"{br}k", "-maxrate", f"{br}k", "-bufsize", f"{br*2}k"]

        # audio
        cmd += self._audio_flags()

        # container tweaks
        if fmt == "mp4":
            cmd += ["-movflags","+faststart"]

        return cmd, out

    def run_quality(self):
        inp = self._ensure_input()
        if not inp: return
        try:
            cmd, out = self._build_quality_cmd_for(inp)
            self._run(cmd, out)
        except Exception as e:
            try: QMessageBox.warning(self, "Quality tool", str(e))
            except Exception: pass
    def run_quality_batch_popup(self):
        # Use the new batch dialog directly
        return self.run_quality_batch()

    def run_quality_batch(self):
        paths = self._batch_paths_with_dialog(title="Batch Quality/Size", exts=BatchSelectDialog.VIDEO_EXTS)
        if not paths:
            return
        ok = 0
        for p in paths:
            try:
                cmd, out = self._build_quality_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch Quality", f"Queued {ok} item(s).")
        except Exception: pass

    def _probe_codecs_and_disable_unavailable(self):
        """Probe ffmpeg encoders once and grey-out unsupported codec choices in the Quality/Size tool."""
        try:
            ff = ffmpeg_path()
            out = subprocess.check_output([ff, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True, creationflags=0 if os.name!="nt" else 0x08000000)
        except Exception:
            out = ""
        encs = set()
        for ln in out.splitlines():
            m = re.match(r"\s*[A-Z\.]{6}\s+(\S+)", ln)
            if m:
                encs.add(m.group(1).strip().lower())
        # Map combobox display -> encoder id
        map_disp = {
            "H.264 (x264)": "libx264",
            "H.265 (x265)": "libx265",
            "AV1 (SVT-AV1)": "libsvtav1",
            "H.264 (NVENC)": "h264_nvenc",
            "HEVC (NVENC)": "hevc_nvenc",
            "AV1 (NVENC)": "av1_nvenc",
        }
        for i in range(self.q_codec.count()):
            disp = self.q_codec.itemText(i)
            enc = map_disp.get(disp, "").lower()
            available = (enc in encs) if enc else True
            try:
                # Grey-out
                itm = self.q_codec.model().item(i)
                if itm is not None:
                    itm.setEnabled(bool(available))
                    if not available:
                        itm.setToolTip("Not available in this ffmpeg build")
                else:
                    # Fallback method
                    from PySide6.QtCore import Qt
                    self.q_codec.setItemData(i, bool(available), Qt.ItemDataRole.EnabledRole)
                    if not available:
                        self.q_codec.setItemData(i, "Not available in this ffmpeg build", Qt.ItemDataRole.ToolTipRole)
            except Exception:
                pass
    

    # ---- Image conversion helpers ----

    def _batch_paths_prompt(self, for_videos: bool|str, title: str):
        """Return a list of paths chosen via a Files/Folder dialog.
        for_videos: True -> videos only, False -> images only, "both" -> videos+images."""
        choice = self._ask_files_or_folder(title)
        allow_both = (for_videos == "both")
        if choice == "files":
            if allow_both:
                exts = "Videos & Images (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.mpg *.mpeg *.ts *.m2ts *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif)"
                files, _ = QFileDialog.getOpenFileNames(self, "Pick files", "", exts)
                return [str(p) for p in files]
            return self._pick_video_files() if for_videos else self._pick_image_files()
        elif choice == "folder":
            title_txt = "Pick a folder with videos" if (for_videos is True) else ("Pick a folder with images" if (for_videos is False) else "Pick a folder with media")
            d = QFileDialog.getExistingDirectory(self, title_txt, "")
            if not d:
                return []
            from pathlib import Path as _P
            if allow_both:
                exts = {".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts",".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
            elif for_videos:
                exts = {".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts"}
            else:
                exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
            paths = []
            try:
                for name in os.listdir(d):
                    pth = _P(d) / name
                    if pth.is_file() and pth.suffix.lower() in exts:
                        paths.append(str(pth))
            except Exception:
                pass
            return paths
        else:
            return []

    def _ask_files_or_folder(self, title: str = "Batch") -> str | None:
        """Ask user to choose 'files' or 'folder'. Returns 'files', 'folder', or None."""
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle(title)
            msg.setText("Choose what to batch")
            btn_files = msg.addButton("Files…", QMessageBox.AcceptRole)
            btn_folder = msg.addButton("Folder…", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_files:
                return "files"
            if clicked == btn_folder:
                return "folder"
            return None
        except Exception:
            return None

    def _pick_video_files(self):

        exts = "Videos (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.mpg *.mpeg *.ts *.m2ts)"
        files, _ = QFileDialog.getOpenFileNames(self, "Pick videos", "", exts)
        return [str(p) for p in files]
    def _pick_image_files(self):
        try:
            files, _ = BatchSelectDialog.pick(self, title="Batch images", exts=getattr(BatchSelectDialog, "IMAGE_EXTS", {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif'}))
            return list(files or [])
        except Exception:
            # Fallback to native file dialog if the shared dialog isn't available
            exts = "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif)"
            files, _ = QFileDialog.getOpenFileNames(self, "Pick images", "", exts)
            return [str(p) for p in files]

    
    def _estimate_quality_for_target_image(self, inp_path: str, fmt: str, target_bytes: int, mw: int, mh: int) -> int:
        """Estimate quality (1..100) for a target size after optional resize using heuristic bits-per-pixel curves."""
        import subprocess, json, os
        # Probe size
        w = h = None
        try:
            ff = ffmpeg_path()
            probe = ff.replace("ffmpeg", "ffprobe")
            out = subprocess.check_output([probe, "-v", "error", "-select_streams", "v:0",
                                           "-show_entries", "stream=width,height",
                                           "-of", "json", inp_path], text=True, creationflags=0x08000000 if os.name=="nt" else 0)
            data = json.loads(out)
            st = (data.get("streams") or [{}])[0]
            w = int(st.get("width") or 0) or None
            h = int(st.get("height") or 0) or None
        except Exception:
            pass
        if not w or not h:
            w, h = 1920, 1080
        # Apply resize box conservatively (decrease)
        if mw or mh:
            sx = (mw / w) if mw else 1.0
            sy = (mh / h) if mh else 1.0
            s = min(sx if mw else 1.0, sy if mh else 1.0)
            if s > 0: w, h = max(1, int(w*s)), max(1, int(h*s))
        pixels = max(1, w*h)
        bpp = (target_bytes * 8.0) / pixels

        def jpg_q(b):
            pts = [(0.15,50),(0.25,60),(0.35,70),(0.55,80),(0.9,90),(1.3,95),(2.0,98)]
            if b <= pts[0][0]: return 45
            for i in range(len(pts)-1):
                x1,y1=pts[i]; x2,y2=pts[i+1]
                if b<=x2: return int(round(y1+(y2-y1)*(b-x1)/(x2-x1)))
            return 98
        def webp_q(b):
            pts = [(0.10,45),(0.20,60),(0.30,70),(0.45,80),(0.65,88),(0.85,93),(1.1,97)]
            if b <= pts[0][0]: return 40
            for i in range(len(pts)-1):
                x1,y1=pts[i]; x2,y2=pts[i+1]
                if b<=x2: return int(round(y1+(y2-y1)*(b-x1)/(x2-x1)))
            return 97

        if (fmt or "jpg").lower() == "webp":
            return max(1, min(100, webp_q(bpp)))
        return max(1, min(100, jpg_q(bpp)))
    def _build_image_cmd_for(self, inp_path):
            from pathlib import Path as _P
            fmt = (self.img_format.currentText() if hasattr(self,'img_format') else "jpg")
            q = int(self.img_quality.value()) if hasattr(self,'img_quality') else 85
            mw = int(self.img_w.value()) if hasattr(self,'img_w') else 0
            mh = int(self.img_h.value()) if hasattr(self,'img_h') else 0
            keep_meta = bool(self.img_keep_meta.isChecked()) if hasattr(self,'img_keep_meta') else False
            use_target = bool(getattr(self,'img_mode',None) and self.img_mode.currentIndex()==1 and (self.img_format.currentText().lower()!="png"))
            if use_target:
                # Convert target to bytes
                unit = (self.img_target_unit.currentText() if hasattr(self,'img_target_unit') else 'KB')
                mult = 1024 if unit=='KB' else 1024*1024
                target_bytes = int(self.img_target.value()) * mult if hasattr(self,'img_target') else 400*1024
                q = self._estimate_quality_for_target_image(str(inp_path), fmt, target_bytes, mw, mh)
    
            inp = _P(inp_path)
            suf = (f"{fmt}_t{int(self.img_target.value())}{self.img_target_unit.currentText().lower()}" if (use_target and hasattr(self,'img_target')) else f"{fmt}_q{q}") + (f"_w{mw}" if mw else "") + (f"_h{mh}" if mh else "")
            out = (OUT_IMAGES if 'OUT_IMAGES' in globals() else _P('.')) / f"{inp.stem}_{suf}.{fmt}"
    
            cmd = [ffmpeg_path(), "-y", "-i", str(inp)]
    
            # Resize if requested
            if mw or mh:
                if mw == 0: mw = -1
                if mh == 0: mh = -1
                filt = f"scale=w={mw if mw>0 else 'iw'}:h={mh if mh>0 else 'ih'}:force_original_aspect_ratio=decrease"
                cmd += ["-vf", filt]
    
            # Map format & quality
            if fmt == "jpg":
                qscale = max(2, min(31, int(round(31 - (q/100.0)*29))))
                cmd += ["-q:v", str(qscale)]
            elif fmt == "webp":
                cmd += ["-c:v", "libwebp", "-q:v", str(q), "-compression_level", "4"]
            elif fmt == "png":
                clevel = max(0, min(9, int(round((100-q)/100.0 * 9))))
                cmd += ["-compression_level", str(clevel)]
    
            if keep_meta:
                cmd += ["-map_metadata", "0"]
            else:
                cmd += ["-map_metadata", "-1"]
    
            return cmd, out

    def run_img_convert(self):
        inp = self._ensure_input(allow_images=True)
        if not inp:
            files = self._pick_image_files()
            if not files: return
            inp = files[0]
        cmd, out = self._build_image_cmd_for(inp)
        self._run(cmd, out)
    def run_img_batch_popup(self):
        # Unified batch: use shared dialog for images
        return self.run_img_batch()
    def run_img_batch_popup(self):
        choice = self._ask_files_or_folder("Batch images")
        if choice == "files":
            return self.run_img_batch()
        elif choice == "folder":
            return self.run_img_batch_folder()
        return None





    def run_img_batch(self):
        files = self._pick_image_files()
        if not files: return
        ok = 0
        for p in files:
            try:
                cmd, out = self._build_image_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch images", f"Queued {ok} image(s).")
        except Exception: pass
    def run_img_batch_folder(self):
        files = self._pick_image_files()
        return
        import os
        img_exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
        paths = []
        for name in os.listdir(d):
            p = Path(d) / name
            if p.is_file() and p.suffix.lower() in img_exts:
                paths.append(str(p))
        if not paths:
            try: QMessageBox.information(self, "Batch folder", "No images found in that folder.")
            except Exception: pass
            return
        ok = 0
        for p in paths:
            try:
                cmd, out = self._build_image_cmd_for(p)
                self._enqueue_cmd_for_input(p, cmd, out); ok += 1
            except Exception:
                continue
        try: QMessageBox.information(self, "Batch folder", f"Queued {ok} image(s).")
        except Exception: pass
    def _enqueue_cmd_for_input(self, inp_path, cmd_list, outfile):
                try:
                    from pathlib import Path as _P
                    out_path = outfile if isinstance(outfile, _P) else _P(str(outfile))
                    try:
                        from helpers.queue_adapter import enqueue_tool_job as enq
                    except Exception:
                        from queue_adapter import enqueue_tool_job as enq
                    enq("tools_ffmpeg", str(inp_path), str(out_path.parent), {"ffmpeg_cmd": cmd_list, "outfile": str(out_path)}, priority=600)
                    return True
                except Exception:
                    return False

    def _batch_paths_with_dialog(self, title="Batch", exts=None):
        """Use the shared BatchSelectDialog to pick files. Returns list of paths (or [])."""
        try:
            files, _ = BatchSelectDialog.pick(self, title=title, exts=exts or BatchSelectDialog.VIDEO_EXTS)
            return list(files or [])
        except Exception:
            return []
    

    def run_speed_batch(self):
        paths = self._batch_paths_with_dialog(title="Batch Speed", exts=BatchSelectDialog.VIDEO_EXTS)
        if not paths:
            return
        try:
            if QMessageBox.question(self, "Batch Speed", f"Add {len(paths)} video(s) with current Speed settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        for p in paths:
            try:
                from pathlib import Path as _P
                inp = _P(p)
                factor = self.speed.value() / 100.0
                setpts = 1.0 / float(factor if factor != 0 else 1.0)
                mute = bool(self.cb_speed_mute.isChecked()) if hasattr(self, "cb_speed_mute") else False
                sync = bool(self.cb_speed_sync.isChecked()) if hasattr(self, "cb_speed_sync") else True
                out = OUT_VIDEOS / f"{inp.stem}_spd_{factor:.2f}x.mp4"
                cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", f"setpts={setpts:.6f}*PTS"]
                if mute:
                    cmd += ["-an"]
                else:
                    if sync:
                        atempos = []
                        f = float(factor)
                        while f > 2.0:
                            atempos.append("2.0"); f /= 2.0
                        while f < 0.5:
                            atempos.append("0.5"); f *= 2.0
                        atempos.append(f"{f:.3f}")
                        cmd += ["-filter:a", "atempo=" + ",".join(atempos)]
                    else:
                        cmd += ["-c:a", "copy", "-shortest"]
                # Match source bitrate so output keeps similar quality/size profile
                br_flags = []
                try:
                    info = probe_media(inp)
                    dur = float(info.get("duration") or 0.0)
                    size_b = float(info.get("size") or 0.0)
                    if dur > 0.0 and size_b > 0.0:
                        br_kbps = int((size_b * 8.0) / (dur * 1000.0))
                        if br_kbps < 300:
                            br_kbps = 300
                        if br_kbps > 50000:
                            br_kbps = 50000
                        br_flags = [
                            "-b:v", f"{br_kbps}k",
                            "-maxrate", f"{br_kbps}k",
                            "-bufsize", f"{br_kbps * 2}k",
                        ]
                except Exception:
                    br_flags = []

                cmd += ["-c:v", "libx264", "-preset", "veryfast"]
                if br_flags:
                    cmd += br_flags
                cmd += ["-movflags", "+faststart", str(out)]
                self._enqueue_cmd_for_input(inp, cmd, out)
            except Exception:
                continue

    def run_gif_batch(self):
        paths = self._batch_paths_with_dialog(title="Batch GIF", exts=BatchSelectDialog.VIDEO_EXTS)
        if not paths:
            return
        fmt = self.gif_fmt.currentText().strip().lower() if hasattr(self, "gif_fmt") else "gif"
        for p in paths:
            try:
                inp = Path(p)
                opts = gif_backend.options_from_ui(self, inp, self.gif_same.isChecked(), int(self.gif_fps.value()) if hasattr(self, "gif_fps") else 0, batch=True)
                out = OUT_VIDEOS / gif_backend.output_name_for(inp.stem, fmt, batch=True)
                cmds = gif_backend.build_commands(inp, out, opts, ffmpeg=ffmpeg_path(), work_dir=OUT_TEMP)
                self._enqueue_cmd_for_input(inp, cmds[0], out)
            except Exception:
                continue
        # Use shared batch picker for videos
        try:
            paths, _ = BatchSelectDialog.pick(self, title="Batch Crop", exts=getattr(BatchSelectDialog, "VIDEO_EXTS", {'.mp4','.mov','.mkv','.avi','.m4v','.webm','.ts','.m2ts','.wmv','.flv','.mpg','.mpeg','.3gp','.3g2','.ogv'}))
        except Exception:
            paths = []
        paths = list(paths or [])
        if not paths:
            return
        try:
            if QMessageBox.question(self, "Batch Crop", f"Add {len(paths)} video(s) with current Crop settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        for p in paths:
            try:
                from pathlib import Path as _P
                inp=_P(p)
                y=int(self.crop_y.value()); h=int(self.crop_h.value())
                out=OUT_VIDEOS / f"{inp.stem}_crop_y{y}_h{h}.mp4"
                filter_str = f"crop=iw:{h}:0:{y}"
                cmd=[ffmpeg_path(),"-y","-i",str(inp),"-vf",filter_str,"-c:v","libx264","-preset","veryfast","-movflags","+faststart",str(out)]
                self._enqueue_cmd_for_input(inp, cmd, out)
            except Exception:
                continue
        try:
            QMessageBox.information(self, "Batch Crop", f"Queued {len(paths)} item(s).")
        except Exception:
            pass