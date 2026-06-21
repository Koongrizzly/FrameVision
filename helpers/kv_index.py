# helpers/kv_index.py — dad-jokes only (keeps API used by intro/settings eggs)
from __future__ import annotations
import os, json, random, glob, re
from typing import List
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtWidgets import QWidget, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

ROOT = os.path.dirname(os.path.dirname(__file__))
ORG="FrameVision"; APP="FrameVision"

# Sources to scan (dad-only)
DB_JSON = os.path.join(ROOT, "assets", "lookup_table.json")
TXT_PATHS = [
    os.path.join(ROOT, "assets", "dad_jokes.txt"),
    os.path.join(ROOT, "assets", "dad.txt"),
]
FOLDERS_DAD = [
    os.path.join(ROOT, "assets", "dad_jokes"),
    os.path.join(ROOT, "assets", "jokes", "dad"),
    os.path.join(ROOT, "assets", "jokes", "dads"),
]
FOLDER_GENERIC = os.path.join(ROOT, "assets", "jokes")  # infer from file/subfolder names or line prefixes

def _is_dad_path(path: str) -> bool:
    p = path.lower().replace("\\", "/")
    return any(seg in p for seg in ("/dad_jokes", "/jokes/dad", "/jokes/dads", "_dad.txt", "dad_jokes.txt", "/dad/"))

def _maybe_strip_prefix(line: str) -> tuple[str, bool]:
    """Return (text, is_dad) where prefixes like 'dad: ...' or '[dad] ...' tag the line as dad."""
    m = re.match(r"^\s*(?:\[(?P<b>\w+)\]|(?P<a>\w+)\s*[:\-])\s*(?P<j>.+)$", line.strip())
    if not m:
        return (line.strip(), False)
    tag = (m.group('a') or m.group('b') or '').strip().lower()
    return (m.group('j').strip(), tag.startswith('dad'))

_cached: List[str] | None = None

def _load_jokes() -> List[str]:
    """Aggregate dad jokes from multiple locations. Returns a list of strings (no categories)."""
    global _cached
    if _cached is not None:
        return _cached
    jokes: List[str] = []

    # 1) JSON: only sets.dad (ignore other sets/jokes)
    try:
        with open(DB_JSON, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            sets = (data.get("sets") or {})
            arr = sets.get("dad")
            if isinstance(arr, list):
                for s in arr:
                    if isinstance(s, str) and s.strip():
                        jokes.append(s.strip())
    except Exception:
        pass

    # 2) Plain TXT dad files
    for path in TXT_PATHS:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: continue
                    txt, is_dad = _maybe_strip_prefix(ln)
                    if is_dad or path:  # these files are explicitly dad
                        jokes.append(txt)
        except Exception:
            continue

    # 3) Dad folders (explicit)
    for folder in FOLDERS_DAD:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln: continue
                        txt, _ = _maybe_strip_prefix(ln)
                        jokes.append(txt)
            except Exception:
                continue

    # 4) Generic jokes folder: include only files we can infer as dad (by path or line prefix)
    if os.path.isdir(FOLDER_GENERIC):
        for path in glob.glob(os.path.join(FOLDER_GENERIC, "**", "*.txt"), recursive=True):
            try:
                is_dad_file = _is_dad_path(path)
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln: continue
                        txt, is_dad_line = _maybe_strip_prefix(ln)
                        if is_dad_file or is_dad_line:
                            jokes.append(txt)
            except Exception:
                continue

    # Deduplicate while preserving order
    seen=set(); dedup=[]
    for j in jokes:
        if j not in seen:
            seen.add(j); dedup.append(j)

    if not dedup:
        dedup = ["I used to be a banker but I lost interest."]

    _cached = dedup
    return dedup

def get_random_joke(kind: str | None = None) -> str:
    """Return a random dad joke string. (kind is ignored to preserve API compatibility.)"""
    s = QSettings(ORG, APP)
    key = "dad_jokes_queue_dad"
    jokes = _load_jokes()
    # Keep a shuffle queue in settings to avoid quick repeats
    try:
        import json as _json
        queue = _json.loads(s.value(key, "[]"))
        if not isinstance(queue, list): queue = []
    except Exception:
        queue = []
    if not queue:
        import random as _r
        queue = list(range(len(jokes)))
        _r.shuffle(queue)
    idx = queue.pop(0)
    s.setValue(key, _json.dumps(queue))
    return jokes[idx % len(jokes)]

# UI helpers (unchanged API)
def _nice_dialog_style():
    s = QSettings(ORG, APP); theme = (s.value("theme","") or "").lower()
    if "night" in theme or "dark" in theme:
        return """
            QDialog{ background:#0e0e0e; }
            QLabel#joke{ background:#1e1e1e; color:#e6e6e6; padding:12px; border-radius:12px; font-size:14pt; }
            QPushButton{ padding:8px 14px; }
        """
    return """
        QDialog{ background:#f3f3f3; }
        QLabel#joke{ background:#ffffff; color:#111; padding:12px; border-radius:12px; font-size:14pt; }
        QPushButton{ padding:8px 14px; }
    """

def _show_joke_dialog(parent: QWidget, text: str):
    try:
        dlg = QDialog(parent); dlg.setWindowTitle("FrameVision — Dad joke"); dlg.setModal(True)
        dlg.setStyleSheet(_nice_dialog_style())
        lay = QVBoxLayout(dlg); lay.setContentsMargins(18,18,18,18); lay.setSpacing(12)
        lbl = QLabel(text); lbl.setObjectName("joke"); lbl.setWordWrap(True); lay.addWidget(lbl)
        row = QHBoxLayout(); row.addStretch(1); btn = QPushButton("😂  Nice"); row.addWidget(btn); lay.addLayout(row)
        btn.clicked.connect(dlg.accept); dlg.adjustSize(); dlg.setAttribute(Qt.WA_DeleteOnClose, True); dlg.exec()
    except Exception:
        pass

def attach_click_hint(widget: QWidget, threshold:int=4, window_ms:int=2500, kind: str | None = None):
    """Attach a click-listener to the widget; after X clicks within window_ms, show a random dad joke."""
    clicks = {"n":0}
    timer = QTimer(widget); timer.setSingleShot(True); timer.setInterval(window_ms)
    def reset(): clicks["n"]=0
    def on_click(_ev=None):
        clicks["n"] += 1
        if clicks["n"] >= threshold:
            reset()
            _show_joke_dialog(widget, get_random_joke())
        else:
            timer.start()
    old = getattr(widget, "mousePressEvent", None)
    def handler(ev):
        try:
            on_click(ev)
        finally:
            if callable(old):
                try: old(ev)
                except Exception: pass
    setattr(widget, "mousePressEvent", handler)
    timer.timeout.connect(reset)
