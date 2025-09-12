
from __future__ import annotations

import re
from typing import Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Slot, Qt
from PySide6.QtWidgets import QTabWidget, QPushButton, QWidget, QApplication

ROLE_NONE = "NONE"
ROLE_UPSCALE = "UPSCALE"
ROLE_RIFE = "RIFE"
ROLE_DESCRIBE = "DESCRIBE"

LABELS = {
    ROLE_UPSCALE: "Upscale",
    ROLE_RIFE: "Add FPS",
    ROLE_DESCRIBE: "Describe",
    ROLE_NONE: "Upscale",
}

TOOLTIPS = {
    ROLE_UPSCALE: "Upscale using current tab settings",
    ROLE_RIFE: "Interpolate frames (RIFE) using current tab settings",
    ROLE_DESCRIBE: "Describe the current frame now",
    ROLE_NONE: "No one-click action for this tab",
}

# Fallback inline styles (we try to respect themes; these are only used if app QSS doesn't override)
STYLE_DEFAULT = "padding:6px 12px; border-radius:8px; font-weight:600;"
STYLE_PRIMARY = "background:#3b82f6; color:white;"  # original button color as safe default
STYLE_SUCCESS = "background:#22c55e; color:white;"  # green
STYLE_WARNING = "background:#f59e0b; color:white;"  # orange
STYLE_DISABLED = "background:#9ca3af; color:white;"  # gray

def _merge_styles(*parts: str) -> str:
    return " ".join([p for p in parts if p])

class QuickActionDriver(QObject):

    def _enqueue_upscale_from_player(self):
        """Queue upscaling for the current player media; avoid in-tab direct runs."""
        try:
            main = self.main
            if not hasattr(main, "current_path") or not main.current_path:
                return
            p = str(main.current_path)
            is_video = p.lower().endswith((".mp4",".mov",".mkv",".avi",".webm",".m4v",".gif"))
            try:
                from helpers.queue_adapter import enqueue, default_outdir
            except Exception:
                from queue_adapter import enqueue, default_outdir
            outdir = default_outdir(is_video, "upscale")
            # Try to read model/scale from visible controls
            model = "RealESRGAN-general-x4v3"
            factor = 4
            try:
                edit = getattr(main, "edit", None)
                for obj in (getattr(edit, "inner", None), edit):
                    if obj is None: 
                        continue
                    cmb = getattr(obj, "cmb_model", None)
                    if cmb is not None and hasattr(cmb, "currentText"):
                        t = str(cmb.currentText() or "").strip()
                        if t: model = t; break
            except Exception:
                pass
            try:
                edit = getattr(main, "edit", None)
                for obj in (getattr(edit, "inner", None), edit):
                    if obj is None:
                        continue
                    sp = getattr(obj, "spn_scale", None)
                    if sp is not None and hasattr(sp, "value"):
                        v = int(sp.value())
                        if v >= 1: factor = v; break
            except Exception:
                pass
            job_type = "upscale_video" if is_video else "upscale_photo"
            enqueue(job_type, p, outdir, factor, model)
            try:
                from helpers.queue_system import QueueSystem
            except Exception:
                try:
                    from queue_system import QueueSystem
                except Exception:
                    QueueSystem = None
            try:
                if QueueSystem:
                    from pathlib import Path as _P
                    QueueSystem(_P(".").resolve()).nudge_pending()
            except Exception:
                pass
        except Exception as e:
            # (disabled) print(GAB log removed)
    """
    Runtime driver that repurposes a single global button into a context-aware action
    based on the active tab. No tab logic duplicated; it simply triggers each tab's
    own primary Start/Run button (or method) with the user's current settings.
    """
    def __init__(self, main_window: QWidget, tabs: QTabWidget, button: QPushButton, parent: Optional[QObject]=None):
        super().__init__(parent)
        self.main = main_window
        self.tabs = tabs
        self.button = button

        # a tiny poll to mirror target-button enabled/disabled while jobs run
        self._mirror_timer = QTimer(self)
        self._mirror_timer.setInterval(400)  # ms
        self._mirror_timer.timeout.connect(self.update_state)

        self._current_role = ROLE_NONE

    # ---- public ------------------------------------------------------------

    def install(self):
        """Attach to the existing button & tab widget."""
        try:
            # Replace any existing click handlers on the Upscale button
            try:
                self.button.clicked.disconnect()
            except Exception:
                pass
            self.button.clicked.connect(self._on_click)

            # Track tab changes
            self.tabs.currentChanged.connect(self.update_state)

            # Initial state
            self.update_state()

            # Start mirror timer
            self._mirror_timer.start()
        except Exception as e:
            print("QuickActionDriver.install failed:", e)

    # ---- internals ---------------------------------------------------------

    def _detect_role(self) -> Tuple[str, QWidget]:
        """Return (role, current_tab_widget)."""
        try:
            tab = self.tabs.currentWidget()
            label = self.tabs.tabText(self.tabs.currentIndex()).lower()
        except Exception:
            tab = None
            label = ""

        # Prefer class/feature-based detection first
        try:
            from helpers.interp import InterpPane
            if isinstance(tab, InterpPane):
                return ROLE_RIFE, tab
        except Exception:
            pass

        # Describe if the tab exposes a describe_now action or name says Describe
        if hasattr(tab, "describe_now"):
            return ROLE_DESCRIBE, tab
        if label and re.search(r"describe", label, re.I):
            return ROLE_DESCRIBE, tab

        # Otherwise, default to Upscale everywhere
        return ROLE_UPSCALE, tab

    def _target_button_for_role(self, role: str, tab: QWidget) -> Optional[QPushButton]:
        """Find the tab's own primary button we should mirror/click."""
        if tab is None:
            return None

        try:
            from PySide6.QtWidgets import QPushButton
        except Exception:
            return None

        # RIFE / Interpolation
        if role == ROLE_RIFE:
            # Preferred: attribute on InterpPane
            btn = getattr(tab, "btn_start", None)
            if isinstance(btn, QPushButton):
                return btn
            # Fallback: search by text
            for b in tab.findChildren(QPushButton):
                txt = (b.text() or "").strip()
                if re.match(r"^(Start(\s*\(Queue\))?|Interpolate)$", txt, re.I):
                    return b
            return None

        # Describe
        if role == ROLE_DESCRIBE:
            # Preferred: call method directly; button may not be exposed as attribute
            return None  # we'll call method directly in _do_action
        # Upscale (if there's an Upscale tab someday)
        if role == ROLE_UPSCALE:
            self._enqueue_upscale_from_player()
            return