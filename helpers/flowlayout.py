
# helpers/flowlayout.py — lightweight FlowLayout for PySide6
from __future__ import annotations
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QSize, QRect, QPoint

class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=0, hspacing=-1, vspacing=-1):
        super().__init__(parent)
        self._items = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem):
        self._items.append(item)

    def addWidget(self, w: QtWidgets.QWidget):
        self.addItem(QtWidgets.QWidgetItem(w))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        l, t, r, b = self.getContentsMargins()
        size += QSize(l + r, t + b)
        return size

    def _hspacing(self) -> int:
        if self._hspace >= 0:
            return self._hspace
        return self.smartSpacing(QtWidgets.QStyle.PM_LayoutHorizontalSpacing)

    def _vspacing(self) -> int:
        if self._vspace >= 0:
            return self._vspace
        return self.smartSpacing(QtWidgets.QStyle.PM_LayoutVerticalSpacing)

    def smartSpacing(self, pm) -> int:
        parent = self.parent()
        if isinstance(parent, QtWidgets.QWidget):
            return parent.style().pixelMetric(pm, None, parent)
        return QtWidgets.QApplication.style().pixelMetric(pm)

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        l, t, r, b = self.getContentsMargins()
        x = rect.x() + l
        y = rect.y() + t
        effective_rect = QRect(rect.x() + l, rect.y() + t, rect.width() - l - r, rect.height() - t - b)

        hspace = self._hspacing()
        vspace = self._vspacing()

        for item in self._items:
            wid = item.widget()
            if wid and not wid.isVisible():
                # still reserve space for layout consistency
                hint = item.sizeHint()
            else:
                hint = item.sizeHint()

            next_x = x + hint.width() + hspace
            if next_x - hspace > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + vspace
                next_x = x + hint.width() + hspace
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))

            x = next_x
            line_height = max(line_height, hint.height())

        return (y + line_height + b) - rect.y()
