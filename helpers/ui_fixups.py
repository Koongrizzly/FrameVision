
from __future__ import annotations

from PySide6.QtCore import QTimer


def _safe_call(installer_name: str, main_window):
    try:
        module = __import__(f"helpers.{installer_name}", fromlist=["*"])
        installer = getattr(module, f"install_{installer_name.split('.')[-1]}")
        installer(main_window)
    except Exception:
        # Silent fail to avoid startup crashes; other fixes continue.
        pass


def _apply_all(main_window):
    # Keep this list small and resilient. It is OK if some are missing.
    # These calls are wrapped to never crash the app.
    # File menu (existing)
    try:
        mod = __import__("helpers.menu_file", fromlist=["*"])
        if hasattr(mod, "install_file_menu"):
            mod.install_file_menu(main_window)
    except Exception:
        pass

    # Info menu (new)
    try:
        mod = __import__("helpers.menu_info", fromlist=["*"])
        if hasattr(mod, "install_info_menu"):
            mod.install_info_menu(main_window)
    except Exception:
        pass


def schedule_all_ui_fixes(main_window):
    """Schedule non-intrusive UI fixes after the main window is created.
    Safe to call multiple times.
    """
    QTimer.singleShot(0, lambda: _apply_all(main_window))
