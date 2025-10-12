try:
    import helpers.diagnostics  # bootstrap Qt message handler ASAP
except Exception:
    pass

# --- FrameVision: quiet specific Qt warnings ---
def _fv_msg_filter(mode, context, message):
    try:
        msg = str(message)
    except Exception:
        msg = message
    # Hide noisy stylesheet warning from Settings content widget only
    if "Could not parse stylesheet" in msg and "FvSettingsContent" in msg:
        return
    # Forward others to stderr (optional)
    try:
        import sys
        sys.stderr.write(msg + "\n")
    except Exception:
        pass

try:
    QtCore.qInstallMessageHandler(_fv_msg_filter)
except Exception:
    pass
# --- end quiet block ---


# --- FrameVision: global warnings/verbosity quiet patch ---
# Set env + filters BEFORE importing torch/transformers/diffusers
try:
    import os as _os, warnings as _warnings
    # Environment toggles
    _os.environ.setdefault("PYTORCH_ENABLE_FLASH_SDP", "0")  # force non-flash SDPA globally
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Warning filters
    _warnings.filterwarnings(
        "ignore",
        message=r".*not compiled with flash attention.*",
        category=UserWarning,
        module=r"transformers\.integrations\.sdpa_attention",
    )
    _warnings.filterwarnings(
        "ignore",
        message=r".*`callback`.*deprecated.*",
        category=FutureWarning,
        module=r"diffusers\..*",
    )
    _warnings.filterwarnings(
        "ignore",
        message=r".*`callback_steps`.*deprecated.*",
        category=FutureWarning,
        module=r"diffusers\..*",
    )
    # Quiet HF loggers
    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from diffusers.utils import logging as _df_logging
        _df_logging.set_verbosity_error()
    except Exception:
        pass
except Exception:
    pass
# --- end warnings quiet patch ---




import sys
from PySide6.QtWidgets import QApplication
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence

import helpers.framevision_app as app_mod
from helpers.ui_fixups import schedule_all_ui_fixes
from helpers.intro_screen import run_intro_if_enabled

def _toggle_fullscreen(win):
    if win.windowState() & Qt.WindowFullScreen:
        win.showMaximized()
    else:
        win.showFullScreen()

def main():
    app = QApplication(sys.argv)
    win = app_mod.MainWindow()
    # Delay-load tab integrations after QApplication exists
    try:
                try:
                    from helpers.tab_integration import integrate_into_main_window
                    integrate_into_main_window(win)
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure Settings tab re-layout gets applied after UI builds
    try:
        from PySide6.QtCore import QTimer
        from helpers.settings_tab import install_settings_tab
        QTimer.singleShot(0, lambda: install_settings_tab(win))
    except Exception:
        pass
    # Apply saved/auto theme at startup
    try:
        from helpers.framevision_app import apply_theme, config
        apply_theme(app, config.get("theme", "Auto"))

    except Exception:
        pass

    integrate_into_main_window(win)
    run_intro_if_enabled(win)
    win.show()  # do not auto-maximize via timers

    # Auto-switch theme when set to 'Auto'
    try:
        from helpers.framevision_app import apply_theme, config
        def _auto_theme_tick():
            try:
                if config.get("theme", "Auto") == "Auto":
                    apply_theme(app, "Auto")

            except Exception:
                pass
        win._auto_theme_timer = QTimer(win)
        win._auto_theme_timer.setInterval(60000)  # 1 min
        win._auto_theme_timer.timeout.connect(_auto_theme_tick)
        win._auto_theme_timer.start()
    except Exception:
        pass


    # -- Heartbeat for worker auto-shutdown --
    try:
        from pathlib import Path as _Path
        import time as _time
        hb_path = _Path(".").resolve() / "app_alive.ping"
        def _write_hb():
            try:
                hb_path.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            except Exception:
                pass
        _write_hb()
        win._heartbeat_timer = QTimer(win)
        win._heartbeat_timer.setInterval(2000)
        win._heartbeat_timer.timeout.connect(_write_hb)
        win._heartbeat_timer.start()
        app.aboutToQuit.connect(lambda: (hb_path.exists() and hb_path.unlink()))
    except Exception:
        pass

    # F11 fullscreen toggle
    QShortcut(QKeySequence("F11"), win, activated=lambda: _toggle_fullscreen(win))

    schedule_all_ui_fixes(win)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()