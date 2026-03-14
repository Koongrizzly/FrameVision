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

    noisy_prefixes = (
        "QPainter::begin: A paint device can only be painted by one painter at a time.",
        "QPainter::translate: Painter not active",
        "QColor::fromHsv: HSV parameters out of range",
    )

    if any(msg.startswith(p) for p in noisy_prefixes):
        return

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
    from PySide6 import QtCore as _QtCore
    _QtCore.qInstallMessageHandler(_fv_msg_filter)
except Exception:
    pass
# --- end quiet block ---

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.distributed\.reduce_op is deprecated, please use torch\.distributed\.ReduceOp instead"
)

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
        message=r".*`QColor`.*fromHsv.*",
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
import helpers.save_guard  # force unique filenames on save



# --- FrameVision: stderr noise filter (FFmpeg/Qt spam) ---
try:
    import io as _io
    import os as _os
    import sys as _sys
    import threading as _threading

    _FV_NOISY_PREFIXES = (
        "[mp3float @",
        "[mp3 @",
        "QPainter::begin: A paint device can only be painted by one painter at a time.",
        "QPainter::translate: Painter not active",
        "QColor::fromHsv: HSV parameters out of range",
    )

    class _FVStreamPrefixFilter:
        def __init__(self, stream, prefixes):
            self._stream = stream
            self._buf = ""
            self._prefixes = tuple(prefixes)

        def _should_drop(self, line):
            s = line.lstrip()
            return any(s.startswith(p) for p in self._prefixes)

        def write(self, data):
            try:
                s = data if isinstance(data, str) else str(data)
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    if self._should_drop(line):
                        continue
                    self._stream.write(line + "\n")
            except Exception:
                pass

        def flush(self):
            try:
                if self._buf:
                    line = self._buf
                    self._buf = ""
                    if not self._should_drop(line):
                        self._stream.write(line)
                if hasattr(self._stream, "flush"):
                    self._stream.flush()
            except Exception:
                pass

        def __getattr__(self, name):
            return getattr(self._stream, name)

    _orig_stderr = _sys.stderr
    _sys.stderr = _FVStreamPrefixFilter(_orig_stderr, prefixes=_FV_NOISY_PREFIXES)

    # Also filter native writes that bypass Python's sys.stderr by redirecting fd 2 through a pipe.
    try:
        _stderr_fd = 2
        _saved_stderr_fd = _os.dup(_stderr_fd)
        _read_fd, _write_fd = _os.pipe()
        _os.dup2(_write_fd, _stderr_fd)
        _os.close(_write_fd)

        def _fv_stderr_pump():
            buf = b""
            try:
                while True:
                    chunk = _os.read(_read_fd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        try:
                            line = raw.decode(errors="replace")
                        except Exception:
                            line = str(raw)
                        if any(line.lstrip().startswith(p) for p in _FV_NOISY_PREFIXES):
                            continue
                        _os.write(_saved_stderr_fd, raw + b"\n")
                if buf:
                    try:
                        line = buf.decode(errors="replace")
                    except Exception:
                        line = str(buf)
                    if not any(line.lstrip().startswith(p) for p in _FV_NOISY_PREFIXES):
                        _os.write(_saved_stderr_fd, buf)
            except Exception:
                pass

        _thr = _threading.Thread(target=_fv_stderr_pump, name="FVStderrFilter", daemon=True)
        _thr.start()
    except Exception:
        pass
except Exception:
    pass
# --- end stderr noise filter ---





import sys
from PySide6.QtWidgets import QApplication
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence

import helpers.framevision_app as app_mod
from helpers.no_wheel_guard import install_no_wheel_guard
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

    # Disable mouse-wheel changes on sliders/spinboxes/combos (so scrolling is safe)
    try:
        install_no_wheel_guard(win)
    except Exception:
        pass

    schedule_all_ui_fixes(win)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()