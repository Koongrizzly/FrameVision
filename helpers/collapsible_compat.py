
# helpers/collapsible_compat.py
# Clean, compatibility-safe CollapsibleSection for use across the app.
# - No forced defaults (expanded is accepted but IGNORED for "remove defaults" policy).
# - Provides a QToolButton header and a content area you can fill via setContentLayout().
# - Exposes setChecked()/isChecked() to mimic checkable containers.

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt

class CollapsibleSection(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None, expanded=None, **kwargs):
        super().__init__(parent)
        self._title = title or ""
        self._checked = False  # start closed; will be restored by UI-state logic

        # Header toggle button
        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setText(self._title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(self._checked)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggled)

        # Content area
        self._content = QtWidgets.QWidget(self)
        self._content.setVisible(self._checked)
        self._content.setObjectName(f"content_{self.objectName() or 'collapsible'}")

        # Layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)

        # Optional: light style hints (let theme QSS do the real work)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

    # --- API compatibility ----------------------------------------------------
    def setContentLayout(self, layout: QtWidgets.QLayout) -> None:
        """Place a layout inside the collapsible content area."""
        self._content.setLayout(layout)

    def contentWidget(self) -> QtWidgets.QWidget:
        return self._content

    def headerButton(self) -> QtWidgets.QToolButton:
        return self._toggle

    def setChecked(self, checked: bool) -> None:
        """Mirror a checkable-container API, forwards to the header button."""
        self._checked = bool(checked)
        self._toggle.blockSignals(True)
        try:
            self._toggle.setChecked(self._checked)
        finally:
            self._toggle.blockSignals(False)
        self._sync_visual()

    def isChecked(self) -> bool:
        return bool(self._checked)

    def isCheckable(self) -> bool:
        """Provided so generic persistence code can treat it like a checkable container."""
        return True

    # --- Internals ------------------------------------------------------------
    def _on_toggled(self, checked: bool) -> None:
        self._checked = bool(checked)
        self._sync_visual()
        self.toggled.emit(self._checked)

    def _sync_visual(self) -> None:
        self._content.setVisible(self._checked)
        self._toggle.setArrowType(Qt.DownArrow if self._checked else Qt.RightArrow)
        # Resize to fit open/closed states nicely
        self._content.adjustSize()
        self.adjustSize()

# Backwards-compatible alias used by some tabs
class ToolsCollapsibleSection(CollapsibleSection):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None, expanded=None, **kwargs):
        # Ignore expanded to avoid forcing defaults
        super().__init__(title=title, parent=parent, expanded=None, **kwargs)
