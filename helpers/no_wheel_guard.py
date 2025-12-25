"""
No-wheel guard for FrameVision (class-level patch).

Goal
----
- Let you scroll the UI with the mouse wheel without accidentally changing
  sliders, spin boxes or combo boxes.
- These widgets will *never* react to the mouse wheel; you adjust them by
  clicking/dragging or opening the dropdown instead.

Bonus: "auto-resize" safety net (optional)
------------------------------------------
Qt can only "auto rearrange" widgets that are inside proper layouts.
If some panels still go out of view when the window gets small, the most
robust app-wide safety net is to make those panels scrollable.

This module can optionally patch QTabWidget (and optionally QMainWindow) so that
any page/panel you add is automatically wrapped in a QScrollArea. That way,
nothing disappears off-screen; it becomes reachable via scrolling.

This is intentionally conservative:
- It does *not* try to rewrite layouts or change geometry-managed widgets.
- It avoids double-wrapping and includes an opt-out property.

Opt-out
-------
Set a widget property to skip wrapping:

    some_widget.setProperty("_fv_no_autoscroll_wrap", True)

Implementation
--------------
- We patch the `wheelEvent` method on the *classes* QAbstractSlider,
  QAbstractSpinBox and QComboBox once.
- For auto-resize safety net, we patch QTabWidget.addTab/insertTab (and optionally
  QMainWindow.setCentralWidget) to wrap child widgets in QScrollArea.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QMainWindow,
    QScrollArea,
    QTabWidget,
)


_patched = False
_auto_resize_patched = True


def _patch_class(cls):
    """Patch wheelEvent on a Qt widget class to always ignore the wheel."""
    if getattr(cls, "_fv_no_wheel_patched_cls", False):
        return

    orig_wheel = cls.wheelEvent

    def wheel_event(self, event, _orig=orig_wheel):
        # Do not change the widget's value via mouse wheel.
        # Ignore the event so that any parent scroll area can scroll instead.
        event.ignore()

    cls.wheelEvent = wheel_event
    cls._fv_no_wheel_patched_cls = True


def install_no_wheel_guard(_root=None):
    """Disable mouse wheel changes for sliders/spinboxes/combos app-wide.

    This function is safe to call multiple times; the patch is applied only once.
    """
    global _patched
    if _patched:
        return
    _patched = True

    for cls in (QAbstractSlider, QAbstractSpinBox, QComboBox):
        try:
            _patch_class(cls)
        except Exception:
            # Never break startup because of a patch failure
            continue


# ------------------------------
# Auto-resize safety net (scroll)
# ------------------------------

def _should_skip_wrap(widget) -> bool:
    try:
        # Explicit opt-out
        if widget.property("_fv_no_autoscroll_wrap"):
            return True
    except Exception:
        pass
    return False


def _wrap_in_scroll_area(widget):
    """Wrap a widget into a QScrollArea (if not already wrapped)."""
    if widget is None:
        return widget

    # Don't wrap a scroll area or anything that opted out
    if isinstance(widget, QScrollArea) or _should_skip_wrap(widget):
        return widget


    # Best-effort: allow shrinking. (Some pages ship with large minimum sizes.)
    try:
        widget.setMinimumSize(0, 0)
    except Exception:
        pass
    try:
        widget.updateGeometry()
    except Exception:
        pass

    # Already wrapped? (our wrappers tag themselves)
    try:
        if widget.property("_fv_autoscroll_wrapped"):
            return widget
    except Exception:
        pass

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    # Attach the widget
    scroll.setWidget(widget)

    # Tag wrapper + wrapped widget to avoid double wraps
    try:
        scroll.setProperty("_fv_autoscroll_wrapper", True)
        widget.setProperty("_fv_autoscroll_wrapped", True)
    except Exception:
        pass

    # Best effort: preserve objectName for debugging styles
    try:
        name = widget.objectName() or widget.__class__.__name__
        scroll.setObjectName(f"{name}__scroll")
    except Exception:
        pass

    return scroll


def _patch_tab_widget_autowrap():
    """Patch QTabWidget so added pages are automatically scroll-wrapped."""
    if getattr(QTabWidget, "_fv_autoscroll_patched_cls", False):
        return

    orig_addTab = QTabWidget.addTab
    orig_insertTab = QTabWidget.insertTab

    def addTab(self, widget, label, _orig=orig_addTab):
        wrapped = _wrap_in_scroll_area(widget)
        return _orig(self, wrapped, label)

    def insertTab(self, index, widget, label, _orig=orig_insertTab):
        wrapped = _wrap_in_scroll_area(widget)
        return _orig(self, index, wrapped, label)

    QTabWidget.addTab = addTab
    QTabWidget.insertTab = insertTab
    QTabWidget._fv_autoscroll_patched_cls = True


def _patch_mainwindow_central_autowrap():
    """Patch QMainWindow.setCentralWidget to scroll-wrap the central widget."""
    if getattr(QMainWindow, "_fv_autoscroll_patched_cls", False):
        return

    orig_setCentralWidget = QMainWindow.setCentralWidget

    def setCentralWidget(self, widget, _orig=orig_setCentralWidget):
        wrapped = _wrap_in_scroll_area(widget)
        return _orig(self, wrapped)

    QMainWindow.setCentralWidget = setCentralWidget
    QMainWindow._fv_autoscroll_patched_cls = True


def _is_our_scroll_wrapper(widget) -> bool:
    try:
        return isinstance(widget, QScrollArea) and bool(widget.property("_fv_autoscroll_wrapper"))
    except Exception:
        return isinstance(widget, QScrollArea)


def _retrofit_existing_tabs():
    """Wrap pages already added to QTabWidget instances (best effort)."""
    app = QApplication.instance()
    if not app:
        return

    for tab in app.allWidgets():
        if not isinstance(tab, QTabWidget):
            continue

        try:
            current = tab.currentIndex()
        except Exception:
            current = -1

        # Walk backwards because we remove/insert tabs
        try:
            count = tab.count()
        except Exception:
            continue

        for i in range(count - 1, -1, -1):
            try:
                page = tab.widget(i)
            except Exception:
                continue

            if page is None or _is_our_scroll_wrapper(page) or _should_skip_wrap(page):
                continue

            try:
                label = tab.tabText(i)
                tooltip = tab.tabToolTip(i)
                whats = tab.tabWhatsThis(i)
                icon = tab.tabIcon(i)
                enabled = tab.isTabEnabled(i)
            except Exception:
                label = ""
                tooltip = ""
                whats = ""
                icon = None
                enabled = True

            try:
                tab.removeTab(i)
            except Exception:
                continue

            wrapped = _wrap_in_scroll_area(page)

            # Insert and restore metadata
            try:
                new_index = tab.insertTab(i, wrapped, label)
            except TypeError:
                # Some bindings prefer the overload with icon
                try:
                    new_index = tab.insertTab(i, wrapped, icon, label)
                except Exception:
                    new_index = tab.insertTab(i, wrapped, label)
            except Exception:
                # If insert fails, try to put the original page back
                try:
                    tab.insertTab(i, page, label)
                except Exception:
                    pass
                continue

            try:
                if icon is not None:
                    tab.setTabIcon(new_index, icon)
                tab.setTabToolTip(new_index, tooltip)
                tab.setTabWhatsThis(new_index, whats)
                tab.setTabEnabled(new_index, enabled)
            except Exception:
                pass

        # Restore selection
        try:
            if current >= 0 and current < tab.count():
                tab.setCurrentIndex(current)
        except Exception:
            pass


def _retrofit_existing_mainwindows():
    """Wrap current QMainWindow central widgets (best effort)."""
    app = QApplication.instance()
    if not app:
        return

    for win in app.allWidgets():
        if not isinstance(win, QMainWindow):
            continue
        try:
            cw = win.centralWidget()
        except Exception:
            continue

        if cw is None or _is_our_scroll_wrapper(cw) or _should_skip_wrap(cw):
            continue

        try:
            win.setCentralWidget(_wrap_in_scroll_area(cw))
        except Exception:
            pass


def _kick_layouts():
    """Force a layout refresh after wrapping, so it works even on the first show."""
    app = QApplication.instance()
    if not app:
        return

    # Activate layouts in visible top-level windows
    try:
        top_levels = app.topLevelWidgets()
    except Exception:
        top_levels = []

    for w in top_levels:
        try:
            lay = w.layout()
            if lay:
                lay.invalidate()
                lay.activate()
        except Exception:
            pass
        try:
            w.updateGeometry()
            w.repaint()
        except Exception:
            pass

    # Give Qt a chance to recompute size hints
    try:
        app.processEvents()
    except Exception:
        pass


def install_auto_resize_guard(patch_tabs=True, patch_mainwindow=False):
    """Install a conservative "auto resize" safety net.

    What it does:
    - Wraps QTabWidget pages (and optionally the QMainWindow central widget)
      in a QScrollArea automatically.
    - This prevents UI from disappearing off-screen on small windows; it becomes
      scrollable instead.

    Safe to call multiple times.
    """
    global _auto_resize_patched
    if _auto_resize_patched:
        return
    _auto_resize_patched = True

    try:
        if patch_tabs:
            _patch_tab_widget_autowrap()
    except Exception:
        pass

    try:
        if patch_mainwindow:
            _patch_mainwindow_central_autowrap()
    except Exception:
        pass


    # If this is installed after some UI is already built, wrap existing tabs/central widgets too.
    # Also kick layouts so the first render respects the new scroll wrappers without needing a
    # manual "make window bigger once" step.
    try:
        QTimer.singleShot(0, _retrofit_existing_tabs)
        if patch_mainwindow:
            QTimer.singleShot(0, _retrofit_existing_mainwindows)
        QTimer.singleShot(0, _kick_layouts)
        QTimer.singleShot(80, _kick_layouts)
    except Exception:
        pass


def install_ui_guards():
    """Convenience: install both wheel guard + auto-resize scroll safety net."""
    install_no_wheel_guard()
    install_auto_resize_guard()
