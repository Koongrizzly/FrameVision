
from __future__ import annotations
from typing import Optional, Dict, Any
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSettings

ORG="FrameVision"; APP="FrameVision"

def _S() -> QSettings:
    s = QSettings(ORG, APP); s.sync(); return s

def get_detail() -> str: return _S().value("describe_detail_level","Medium", type=str) or "Medium"
def set_detail(v:str)->None: _S().setValue("describe_detail_level", v)
def get_style() -> str: return _S().value("describe_decode_style","Deterministic", type=str) or "Deterministic"
def set_style(v:str)->None: _S().setValue("describe_decode_style", v)
def get_promptify() -> bool: return _S().value("describe_promptify", True, type=bool)
def set_promptify(v:bool)->None: _S().setValue("describe_promptify", v)
def get_negative() -> str: return _S().value("describe_negative","", type=str) or ""
def set_negative(v:str)->None: _S().setValue("describe_negative", v)

def build_prompt(raw:str) -> str:
    base = (raw or "").strip().rstrip(".")
    if not base: return ""
    extras = ["high detail", "sharp focus", "cinematic", "natural soft light", "balanced colors", "intricate texture"]
    prompt = f"{base}, " + ", ".join(extras) + ", 8k"
    neg = get_negative().strip()
    if neg: prompt += f"\nNegative prompt: {neg}"
    return prompt

class _PromptifyBox(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Promptify & Decode", parent)
        self.setObjectName("FvPromptifyBox")
        v = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.detail = QtWidgets.QComboBox(); self.detail.addItems(["Short","Medium","Long"]); self.detail.setCurrentText(get_detail())
        self.style  = QtWidgets.QComboBox(); self.style.addItems(["Deterministic","Creative"]); self.style.setCurrentText(get_style())
        hrow = QtWidgets.QHBoxLayout()
        self.chk_promptify = QtWidgets.QCheckBox("Promptify output"); self.chk_promptify.setChecked(get_promptify())
        btn_copy = QtWidgets.QPushButton("Copy prompt")
        btn_apply = QtWidgets.QPushButton("Apply to output")
        hrow.addWidget(self.chk_promptify); hrow.addStretch(1); hrow.addWidget(btn_apply); hrow.addWidget(btn_copy)
        self.neg = QtWidgets.QPlainTextEdit(get_negative()); self.neg.setPlaceholderText("Optional negative prompt (comma-separated)"); self.neg.setFixedHeight(54)
        form.addRow("Detail level:", self.detail); form.addRow("Decode style:", self.style)
        v.addLayout(form); v.addLayout(hrow); v.addWidget(self.neg)
        self.detail.currentTextChanged.connect(set_detail); self.style.currentTextChanged.connect(set_style)
        self.chk_promptify.toggled.connect(set_promptify); self.neg.textChanged.connect(lambda: set_negative(self.neg.toPlainText()))
        btn_copy.clicked.connect(self.copy_prompt); btn_apply.clicked.connect(self.apply_to_output)

    def _find_output_widget(self) -> Optional[QtWidgets.QPlainTextEdit]:
        parent = self.window()
        if not parent: return None
        candidates = parent.findChildren(QtWidgets.QPlainTextEdit)
        for w in candidates:
            name = (w.objectName() or "").lower()
            if "desc" in name or "output" in name:
                return w
        return candidates[-1] if candidates else None

    def _current_text(self) -> str:
        out = self._find_output_widget()
        return out.toPlainText() if out else ""

    @QtCore.Slot()
    def copy_prompt(self):
        raw = self._current_text()
        text = build_prompt(raw) if self.chk_promptify.isChecked() else raw
        QtWidgets.QApplication.clipboard().setText(text or "")

    @QtCore.Slot()
    def apply_to_output(self):
        out = self._find_output_widget()
        if not out: return
        raw = out.toPlainText()
        text = build_prompt(raw) if self.chk_promptify.isChecked() else raw
        if text: out.setPlainText(text)

def _find_describe_container(app: QtWidgets.QApplication) -> Optional[QtWidgets.QWidget]:
    for w in app.allWidgets():
        try:
            if isinstance(w, QtWidgets.QGroupBox) and ("describe" in (w.title() or "").lower() or "describe" in (w.objectName() or "").lower()):
                return w
        except Exception:
            pass
    for w in app.allWidgets():
        if isinstance(w, QtWidgets.QTabWidget):
            for i in range(w.count()):
                t = (w.tabText(i) or "").lower()
                if "describe" in t or "describer" in t:
                    return w.widget(i)
    return None

def _install_ui():
    app = QtWidgets.QApplication.instance()
    if not app: return
    parent = _find_describe_container(app)
    if not parent:
        QtCore.QTimer.singleShot(800, _install_ui); return
    if parent.findChild(QtWidgets.QGroupBox, "FvPromptifyBox"): return
    lay = parent.layout(); box = _PromptifyBox(parent)
    if isinstance(lay, QtWidgets.QVBoxLayout): lay.addWidget(box)
    elif isinstance(lay, QtWidgets.QFormLayout): lay.addRow(box)
    else:
        v = QtWidgets.QVBoxLayout(parent); v.addWidget(box); parent.setLayout(v)

def _decode_params() -> Dict[str, Any]:
    detail = get_detail(); style = get_style()
    presets = {
        "Short": dict(max_new_tokens=32, min_length=10, no_repeat_ngram_size=3),
        "Medium": dict(max_new_tokens=96, min_length=30, no_repeat_ngram_size=3),
        "Long": dict(max_new_tokens=160, min_length=60, no_repeat_ngram_size=4),
    }
    dec = presets.get(detail, presets["Medium"]).copy()
    if style == "Deterministic":
        dec.update(dict(num_beams=5, length_penalty=1.2, temperature=1.0, top_p=1.0, top_k=0))
    else:
        dec.update(dict(num_beams=1, temperature=0.8, top_p=0.9, top_k=50))
    dec.setdefault("repetition_penalty", 1.1)
    return dec

def _try_patch_describer():
    try:
        import helpers.interp as interp
    except Exception:
        return
    target = getattr(interp, "describe_image", None)
    if callable(target):
        def wrapped(*a, **kw):
            params = _decode_params()
            for k,v in params.items(): kw.setdefault(k, v)
            return target(*a, **kw)
        setattr(interp, "describe_image", wrapped)
        print("[fv] describer: patched helpers.interp.describe_image with decode params")
        return
    D = getattr(interp, "Describer", None)
    if D and hasattr(D, "describe"):
        orig = D.describe
        def wrap(self, *a, **kw):
            params = _decode_params()
            for k,v in params.items(): kw.setdefault(k, v)
            return orig(self, *a, **kw)
        setattr(D, "describe", wrap)
        print("[fv] describer: patched helpers.interp.Describer.describe with decode params")
        return

def _start():
    try:
        QtCore.QTimer.singleShot(1200, _install_ui)
        QtCore.QTimer.singleShot(1000, _try_patch_describer)
    except Exception:
        _install_ui(); _try_patch_describer()

try:
    _start()
except Exception:
    pass
