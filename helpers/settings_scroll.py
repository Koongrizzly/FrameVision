# helpers/settings_scroll.py — tag Settings scroll content for scoped styling
from __future__ import annotations
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget, QScrollArea, QVBoxLayout, QLayoutItem

def _find_settings_page() -> QWidget | None:
    app = QApplication.instance()
    if not app: return None
    try:
        from PySide6.QtWidgets import QTabWidget
        for tl in app.topLevelWidgets():
            for tw in tl.findChildren(QTabWidget):
                for i in range(tw.count()):
                    try:
                        if tw.tabText(i).strip().lower() == "settings":
                            return tw.widget(i)
                    except Exception: continue
    except Exception:
        pass
    return None

def _move_all_items(src_layout: QVBoxLayout, dst_layout: QVBoxLayout):
    while True:
        item: QLayoutItem = src_layout.takeAt(0)
        if item is None: break
        w = item.widget(); l = item.layout(); s = item.spacerItem()
        if w is not None: dst_layout.addWidget(w)
        elif l is not None: dst_layout.addLayout(l)
        elif s is not None: dst_layout.addItem(s)

def _wrap_page_in_scroll(page: QWidget):
    if getattr(page, "_fv_settings_scrolled", False): return
    lay = page.layout()
    if lay is None:
        lay = QVBoxLayout(page); page.setLayout(lay)
    for i in range(lay.count()):
        w = lay.itemAt(i).widget()
        from PySide6.QtWidgets import QScrollArea
        if isinstance(w, QScrollArea):
            page._fv_settings_scrolled = True; return

    from PySide6.QtWidgets import QScrollArea
    scroll = QScrollArea(page)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.viewport().setAutoFillBackground(True)
    scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)

    content = QWidget(scroll)
    content.setAttribute(Qt.WA_StyledBackground, True)
    content.setStyleSheet("background: palette(window);")
    content.setObjectName("FvSettingsContent")  # <— tag for scoped spacing
    inner = QVBoxLayout(content); inner.setContentsMargins(8,8,8,8); inner.setSpacing(8)

    _move_all_items(lay, inner)

    scroll.setWidget(content)
    lay.addWidget(scroll)
    page._fv_settings_scrolled = True

def auto_wrap_settings_scroll(retries: int = 12, delay_ms: int = 300):
    page = _find_settings_page()
    if page is None:
        if retries <= 0: return
        QTimer.singleShot(delay_ms, lambda: auto_wrap_settings_scroll(retries-1, delay_ms)); return
    try: _wrap_page_in_scroll(page)
    except Exception: pass
