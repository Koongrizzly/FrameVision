# helpers/settings_diagnostics.py — robust installer for the "Developer Diagnostics" group
from __future__ import annotations
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QWidget, QTabWidget, QScrollArea, QVBoxLayout, QGroupBox, QHBoxLayout,
    QCheckBox, QPushButton, QLabel, QFileDialog
)
from helpers import diagnostics as diag

def _find_settings_container(root: QWidget) -> QWidget | None:
    """Try to locate the inner QWidget that holds Settings controls."""
    if root is None:
        return None

    # If root itself is a QTabWidget, look for "Settings" tab
    if isinstance(root, QTabWidget):
        for i in range(root.count()):
            if root.tabText(i).strip().lower() == "settings":
                w = root.widget(i)
                # If the tab content is a scroll area, unwrap to inner widget
                if isinstance(w, QScrollArea):
                    return w.widget()
                return w

    # Otherwise, search for a QTabWidget among children
    tabs = root.findChildren(QTabWidget)
    for t in tabs:
        for i in range(t.count()):
            if t.tabText(i).strip().lower() == "settings":
                w = t.widget(i)
                if isinstance(w, QScrollArea):
                    return w.widget()
                return w

    # Fallback: if the widget tree already has a group box with the right name, return its parent
    existing = root.findChild(QGroupBox, "dev_diag_group")
    if existing:
        return existing.parentWidget()

    return None

def _make_group() -> QGroupBox:
    g = QGroupBox("Developer Diagnostics")
    g.setObjectName("dev_diag_group")
    lay = QVBoxLayout(g)

    chk = QCheckBox("Enable diagnostics logging (console + logs/framevision.log)")
    try:
        chk.setChecked(diag.diag_enabled())
    except Exception:
        pass
    def on_tog(v: bool):
        try:
            diag.set_diag_enabled(v)
        except Exception:
            pass
    chk.toggled.connect(on_tog)
    lay.addWidget(chk)

    row = QHBoxLayout()
    btn_dump = QPushButton("Dump QSettings now")
    def do_dump():
        try:
            diag.dump_qsettings("manual")
        except Exception as e:
            diag.log("Dump QSettings button failed:", e)
    btn_dump.clicked.connect(do_dump)
    row.addWidget(btn_dump)

    btn_open = QPushButton("Open log folder")
    def do_open():
        try:
            # Try to use native dialog to open the logs dir; fallback to message box
            import os
            from helpers.diagnostics import LOG_DIR
            if os.path.isdir(LOG_DIR):
                QtWidgets.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(LOG_DIR))
            else:
                QtWidgets.QMessageBox.information(g, "Logs", f"Log folder not found:\n{LOG_DIR}")
        except Exception as e:
            diag.log("Open log folder failed:", e)
    btn_open.clicked.connect(do_open)
    row.addWidget(btn_open)

    row.addStretch(1)
    lay.addLayout(row)


    return g

def install_diagnostics_settings(root_widget: QWidget | None) -> None:
    """Idempotently add the Developer Diagnostics group into the Settings tab."""
    try:
        if root_widget is None:
            return
        container = _find_settings_container(root_widget)
        if container is None:
            return
        # Ensure container has a layout
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            container.setLayout(lay)

        # Already present?
        if container.findChild(QGroupBox, "dev_diag_group"):
            return

        # Insert before a bottom spacer if present, else append
        grp = _make_group()
        inserted = False
        # Try to append cleanly
        lay.addWidget(grp)
        inserted = True

        if inserted:
            diag.log("Diagnostics settings group added to Settings tab.")
    except Exception as e:
        try:
            diag.log("install_diagnostics_settings failed:", e)
        except Exception:
            pass
