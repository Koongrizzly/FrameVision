# helpers/tooltips.py — FIXED: correctly walks all descendants (QWidget) for upgrades/applies
from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable

REGISTRY: Dict[str, Tuple[str, str]] = {}

def _esc(s: str) -> str:
    s = s or ""
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def rich(title: str, body: str) -> str:
    if not title and not body:
        return ""
    if not title:
        return _esc(body)
    return f"<b>{_esc(title)}</b><br/>{_esc(body)}"

def _title_from(w) -> str:
    try:
        if hasattr(w, "text"):
            t = w.text() or ""
            if t:
                return t.replace("&","").strip()
    except Exception:
        pass
    try:
        if hasattr(w, "placeholderText"):
            p = w.placeholderText() or ""
            if p:
                return p
    except Exception:
        pass
    try:
        n = w.objectName()
        if n:
            return n
    except Exception:
        pass
    try:
        return w.__class__.__name__
    except Exception:
        return "Widget"

def _descendants(parent):
    try:
        from PySide6.QtWidgets import QWidget
        try:
            return parent.findChildren(QWidget)
        except Exception:
            pass
    except Exception:
        pass
    try:
        return parent.children()
    except Exception:
        return []

def set_tip(widget, title: str = "", body: str = "", *, key: Optional[str]=None) -> None:
    if key and (not title and not body) and key in REGISTRY:
        title, body = REGISTRY[key]
    try:
        widget.setToolTip(rich(title, body))  # type: ignore[attr-defined]
    except Exception:
        pass

def apply_registry(parent, scope: Optional[str]=None) -> int:
    n = 0
    for w in _descendants(parent) or []:
        try:
            tip = w.toolTip() if hasattr(w, "toolTip") else ""
        except Exception:
            tip = ""
        if tip and "<b>" in tip:
            continue
        name = ""
        try: name = w.objectName() or ""
        except Exception: name = ""
        if name and f"object:{name}" in REGISTRY:
            t,b = REGISTRY[f"object:{name}"]
            try: w.setToolTip(rich(t,b)); n += 1
            except Exception: pass
            continue
        text = ""
        try: text = (w.text() if hasattr(w,"text") else "") or ""
        except Exception: text = ""
        text = text.strip()
        if text and f"text:{text}" in REGISTRY:
            t,b = REGISTRY[f"text:{text}"]
            try: w.setToolTip(rich(t,b)); n += 1
            except Exception: pass
            continue
        if text:
            cname = ""
            try: cname = w.__class__.__name__
            except Exception: cname = ""
            key = f"class:{cname}:{text}"
            if key in REGISTRY:
                t,b = REGISTRY[key]
                try: w.setToolTip(rich(t,b)); n += 1
                except Exception: pass
                continue
    return n

def upgrade_children_tooltips(parent) -> int:
    n = 0
    for w in _descendants(parent) or []:
        try:
            tip = w.toolTip() if hasattr(w, "toolTip") else ""
        except Exception:
            tip = ""
        if not tip or "<b>" in tip:
            continue
        title = _title_from(w)
        try:
            w.setToolTip(rich(title, tip))
            n += 1
        except Exception:
            pass
    return n

__all__ = ["REGISTRY", "rich", "set_tip", "apply_registry", "upgrade_children_tooltips"]
