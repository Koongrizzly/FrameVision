"""
Global wheel guard for FrameVision.

Goal:
- Allow normal scrolling with the mouse wheel anywhere in the app.
- Prevent sliders, spin boxes and combo boxes from changing values
  when you just scroll past them.
- Only let these widgets react to the wheel when they have focus
  (i.e. after you click them).

When a protected widget does not have focus, the wheel event is
forwarded to the nearest scroll area (if any) so scrolling still works.
"""

from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QAbstractScrollArea,
)


class WheelGuard(QObject):
    def eventFilter(self, obj, event):
        # Only care about wheel events
        if event.type() != QEvent.Wheel:
            return False

        # Only protect typical "setting" widgets
        if not isinstance(obj, (QAbstractSlider, QAbstractSpinBox, QComboBox)):
            return False

        # If the widget has keyboard focus, let it handle the wheel normally.
        # This means: click to focus, then wheel to fine‑tune.
        if obj.hasFocus():
            return False

        # Widget does NOT have focus: do NOT let it change value,
        # but still allow the user to scroll the view by forwarding the
        # wheel event to the nearest scroll area ancestor (if any).
        scroll_area = self._find_scroll_area_ancestor(obj)
        if scroll_area is not None:
            # Call the scroll area's wheel handler directly so it scrolls.
            scroll_area.wheelEvent(event)
            # Returning True prevents the widget itself from reacting.
            return True

        # No scroll area found: safest option is to block the wheel so
        # settings do not change unexpectedly.
        return True

    def _find_scroll_area_ancestor(self, widget):
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                return parent
            parent = parent.parent()
        return None
