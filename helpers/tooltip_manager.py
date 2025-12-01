# helpers/tooltip_manager.py
#
# Global runtime on/off switch for ALL Qt tooltips in the app.
#
# How it works
# ------------
# We install a global eventFilter on the QApplication that intercepts
# QEvent.ToolTip and swallows it when tooltips are disabled.
#
# Lazy install
# ------------
# You do NOT *need* to call init() at startup for it to work anymore.
# The first time Settings calls TooltipManager.set_enabled(...) or
# TooltipManager.is_enabled(), we grab QApplication.instance(),
# install the blocker if it's missing, and start enforcing the rule.
#
# Boot-time disable (recommended polish)
# --------------------------------------
# If you ALSO want tooltips to start disabled before the user ever
# opens Settings, add this right after you create QApplication:
#
#     from PySide6.QtCore import QSettings
#     from helpers.tooltip_manager import TooltipManager
#
#     app = QApplication(sys.argv)
#
#     s = QSettings("FrameVision","FrameVision")
#     start_enabled = s.value("tooltips_enabled", True, type=bool)
#
#     TooltipManager.init(app, enabled=bool(start_enabled))
#
# That pre-installs the blocker immediately and uses the saved state.
#
# Runtime toggle
# --------------
# In Settings we connect the checkbox to TooltipManager.set_enabled(b),
# and also update QSettings("tooltips_enabled", b). This instantly
# turns tooltips on/off everywhere, including inside Settings itself.

from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import QApplication


class _TooltipBlocker(QObject):
    """Internal event filter that swallows tooltip events when disabled."""

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QObject
        if not isinstance(obj, QObject):
            return False
        # QEvent.ToolTip fires right before Qt shows a tooltip bubble.
        if event.type() == QEvent.ToolTip and not TooltipManager.tooltips_enabled:
            # Returning True means "we handled it", so Qt will NOT show the tooltip.
            return True
        # Otherwise allow normal behavior.
        return super().eventFilter(obj, event)


class TooltipManager:
    """Singleton-style manager for global tooltip enable/disable."""

    tooltips_enabled = True
    _blocker = None

    @classmethod
    def _ensure_blocker(cls):
        """
        Make sure our global eventFilter is installed on the current
        QApplication. Safe to call multiple times.
        """
        if cls._blocker is None:
            app = QApplication.instance()
            if app is None:
                # QApplication not created yet. We'll try again later.
                return
            cls._blocker = _TooltipBlocker()
            app.installEventFilter(cls._blocker)

    @classmethod
    def init(cls, app: QApplication, enabled: bool = True):
        """
        Optional boot-time setup.

        Call once after QApplication is created (before main window shows).
        This will install the blocker immediately and use the provided
        enabled state so tooltips can start OFF on first frame.
        """
        cls.tooltips_enabled = bool(enabled)
        cls._ensure_blocker()

    @classmethod
    def set_enabled(cls, enabled: bool):
        """Turn tooltips on/off at runtime and ensure blocker is active."""
        cls.tooltips_enabled = bool(enabled)
        cls._ensure_blocker()

    @classmethod
    def enable(cls):
        cls.set_enabled(True)

    @classmethod
    def disable(cls):
        cls.set_enabled(False)

    @classmethod
    def is_enabled(cls) -> bool:
        cls._ensure_blocker()
        return bool(cls.tooltips_enabled)