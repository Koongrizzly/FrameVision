"""
No-wheel guard for FrameVision (class-level patch).

Goal
----
- Let you scroll the UI with the mouse wheel without accidentally changing
  sliders, spin boxes or combo boxes.
- These widgets will *never* react to the mouse wheel; you adjust them by
  clicking/dragging or opening the dropdown instead.

Implementation
--------------
- We patch the `wheelEvent` method on the *classes* QAbstractSlider,
  QAbstractSpinBox and QComboBox once.
- For all instances (existing and future), wheel events are simply ignored.
  Qt then lets the parent scroll area handle scrolling normally.
"""

from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
)


_patched = False


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
