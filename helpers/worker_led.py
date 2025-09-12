from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QFrame
from PySide6.QtCore import Qt

class WorkerStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "unknown"
        self.led = QFrame(); self.led.setFixedSize(14,14)
        self.led.setFrameShape(QFrame.NoFrame); self.led.setStyleSheet(self._style_for("unknown"))
        self.label = QLabel("Worker: unknown")
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        lay.addWidget(self.led); lay.addWidget(self.label); lay.addStretch(1)

    def _style_for(self, state):
        color = "#888"
        if state == "running": color = "green"
        elif state == "idle": color = "gold"
        elif state == "stopped": color = "red"
        elif state == "error": color = "orange"
        return f"border-radius:7px; background:{color};"

    def set_state(self, state, tip=""):
        self._state = state
        self.led.setStyleSheet(self._style_for(state))
        self.label.setText(f"Worker: {state}")
        if tip:
            self.setToolTip(tip)
