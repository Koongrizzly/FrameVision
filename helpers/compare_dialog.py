# helpers/compare_dialog.py
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QMessageBox
)

_IMG_EXTS = {"png","jpg","jpeg","webp","bmp","tif","tiff","gif"}
_VID_EXTS = {"mp4","mkv","mov","webm","avi","m4v","mpg","mpeg","wmv"}

def _kind(path: str) -> str:
    try:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
    except Exception:
        ext = ""
    if ext in _VID_EXTS:
        return "video"
    if ext in _IMG_EXTS:
        return "image"
    return "image"


class ComparePickDialog(QDialog):
    """
    Simple picker dialog:
    - Select Left file
    - Select Right file
    - Enforce same kind (image+image or video+video)
    """
    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Compare (Side-by-side)")
        self.setModal(True)

        self._start_dir = start_dir or ""
        self._left = ""
        self._right = ""
        self._kind = None

        root = QVBoxLayout(self)

        info = QLabel("Select two files to compare.\nBoth must be the same type (both images or both videos).")
        info.setWordWrap(True)
        root.addWidget(info)

        # Left row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Left:"))
        self.left_edit = QLineEdit()
        self.left_edit.setReadOnly(True)
        row1.addWidget(self.left_edit, 1)
        self.btn_left = QPushButton("Browse…")
        self.btn_left.clicked.connect(self._pick_left)
        row1.addWidget(self.btn_left)
        root.addLayout(row1)

        # Right row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Right:"))
        self.right_edit = QLineEdit()
        self.right_edit.setReadOnly(True)
        row2.addWidget(self.right_edit, 1)
        self.btn_right = QPushButton("Browse…")
        self.btn_right.clicked.connect(self._pick_right)
        row2.addWidget(self.btn_right)
        root.addLayout(row2)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_ok)
        root.addLayout(buttons)

        self.resize(640, 180)

    def _pick_left(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select left media", self._start_dir, "Media Files (*.*)")
        if not path:
            return
        self._left = path
        self.left_edit.setText(path)
        self._start_dir = os.path.dirname(path) or self._start_dir
        self._validate()

    def _pick_right(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select right media", self._start_dir, "Media Files (*.*)")
        if not path:
            return
        self._right = path
        self.right_edit.setText(path)
        self._start_dir = os.path.dirname(path) or self._start_dir
        self._validate()

    def _validate(self):
        if not self._left or not self._right:
            self.status.setText("")
            self.btn_ok.setEnabled(False)
            self._kind = None
            return

        k1 = _kind(self._left)
        k2 = _kind(self._right)

        if k1 != k2:
            self.status.setText("⚠️ Files must be the same type (both images or both videos).")
            self.btn_ok.setEnabled(False)
            self._kind = None
            return

        self._kind = k1
        self.status.setText(f"Ready: {k1} compare")
        self.btn_ok.setEnabled(True)

    def _on_ok(self):
        if not self._left or not self._right or not self._kind:
            QMessageBox.warning(self, "Compare", "Please select two files of the same type.")
            return
        self.accept()

    def get_selection(self):
        return self._left, self._right, self._kind



def open_with_files(parent, left_path: str, right_path: str):
    """Programmatic entry path.

    - Detect media type (image/video)
    - Validate both sides exist and are same type
    - Load viewer immediately (bypass manual selection UI)

    Returns True when opened, False otherwise.
    """
    try:
        left = str(left_path or '').strip()
        right = str(right_path or '').strip()
        if not left or not right:
            return False
        if not os.path.exists(left) or not os.path.exists(right):
            return False
        k1 = _kind(left)
        k2 = _kind(right)
        if k1 != k2:
            return False
        # Open immediately using the existing viewer hook (parent is expected to implement open_compare).
        if parent is not None and hasattr(parent, 'open_compare'):
            try:
                parent.open_compare(left, right, k1)
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False
