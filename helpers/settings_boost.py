
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QCheckBox, QLabel,
    QDialogButtonBox, QMessageBox
)

def _make_cache_dialog(parent) -> QDialog:
    dlg = QDialog(parent)
    dlg.setWindowTitle("Clear program cache")
    lay = QVBoxLayout(dlg); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)

    group = QGroupBox(dlg)
    v = QVBoxLayout(group); v.setContentsMargins(10,10,10,10); v.setSpacing(8)

    chk_pyc = QCheckBox("Python bytecode (__pycache__, *.pyc/*.pyo)"); chk_pyc.setChecked(False)  # default removed; will restore via UI
    chk_logs = QCheckBox("Logs folder"); chk_logs.setChecked(False)  # default removed; will restore via UI
    chk_thumbs = QCheckBox("Thumbnails (Frames/Previews)"); chk_thumbs.setChecked(False)  # default removed; will restore via UI
    chk_qt = QCheckBox("Qt cache (QSettings, etc.)")
    chk_hf = QCheckBox("HuggingFace cache (models)")

    for cb in (chk_pyc, chk_logs, chk_thumbs, chk_qt, chk_hf):
        v.addWidget(cb)

    tip = QLabel("Tip: This does not remove your rendered outputs or presets.")
    tip.setWordWrap(True)
    tip.setStyleSheet("color:#888;")
    v.addWidget(tip)

    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)

    def update_ok():
        bb.button(QDialogButtonBox.Ok).setEnabled(any(cb.isChecked() for cb in (chk_pyc, chk_logs, chk_thumbs, chk_qt, chk_hf)))
    for cb in (chk_pyc, chk_logs, chk_thumbs, chk_qt, chk_hf):
        cb.toggled.connect(update_ok)
    update_ok()

    def on_accept():
        # lazy import so helpers path doesn't matter
        try:
            from helpers.cleanup_cache import run_cleanup
        except Exception:
            from cleanup_cache import run_cleanup
        res = run_cleanup(
            project_root=str(Path.cwd()),
            clean_pyc=chk_pyc.isChecked(),
            clean_logs=chk_logs.isChecked(),
            clean_thumbs=chk_thumbs.isChecked(),
            clean_qt_cache=chk_qt.isChecked(),
            clean_hf_cache=chk_hf.isChecked()
        )
        msg = "Cache cleared:\n" + "\n".join(f"  {k}: {v}" for k, v in res.items())
        QMessageBox.information(parent or dlg, "Clear cache", msg)
        dlg.accept()

    bb.accepted.connect(on_accept)
    bb.rejected.connect(dlg.reject)

    lay.addWidget(group)
    lay.addWidget(bb)
    return dlg
