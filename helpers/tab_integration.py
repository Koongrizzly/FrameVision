
from __future__ import annotations
import importlib, inspect, json, os, traceback, tempfile, webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QPlainTextEdit,
    QHBoxLayout, QPushButton, QMessageBox, QFrame, QSizePolicy, QFileDialog
)


def _sync_from_main_path(main):
    """Push main.current_path into Upscaler/Describer fields if available."""
    try:
        p = getattr(main, "current_path", None)
        p = str(p) if p else ""
        if not p:
            return
        # Describer test image
        try:
            inner = getattr(main, "describer_tab", None).inner
            for name in ("edit_test_image","line_image","lineEdit_image"):
                w = getattr(inner, name, None)
                if w and getattr(w, "setText", None): w.setText(p)
        except Exception:
            pass
    except Exception:
        pass

def _wrap_video_open(main):
    """Wrap VideoPane.open() to sync path whenever the player source changes."""
    try:
        if hasattr(main, "video") and hasattr(main.video, "open"):
            _orig = main.video.open
            def _wrap(path):
                res = _orig(path)
                try: _sync_from_main_path(main)
                except Exception: pass
                return res
            main.video.open = _wrap
    except Exception:
        pass



def _apply_profile_preferences(inner):
    return

def _sync_from_main_path(main):
    """Copy the player's current file path into Upscaler/Describer input fields."""
    try:
        p = getattr(main, "current_path", None)
        p = str(p) if p else ""
        # Upscaler
        try:
            inner = getattr(main, "upscaler_tab", None).inner
            for name in ("edit_input","line_input","lineEdit_input"):
                w = getattr(inner, name, None)
                if w and hasattr(w, "setText"): w.setText(p)
        except Exception:
            pass
        # Describer
        try:
            inner = getattr(main, "describer_tab", None).inner
            for name in ("edit_test_image","line_image","lineEdit_image"):
                w = getattr(inner, name, None)
                if w and hasattr(w, "setText"): w.setText(p)
        except Exception:
            pass
    except Exception:
        pass


def _import_module(module_candidates):
    last_err = None
    for name in module_candidates:
        try:
            return importlib.import_module(name), name
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import any of {module_candidates}: {last_err}")

# ----------------- monkey-patch helpers -----------------

def _read_val(self, obj, attr, default=""):
    w = getattr(self, obj, None)
    if w is None:
        return default
    try:
        if attr == "text":
            return w.text().strip()
        if attr == "value":
            return int(w.value())
        if attr == "currentText":
            return w.currentText().strip()
    except Exception:
        return default
    return default

def _set_val(self, obj, attr, value):
    w = getattr(self, obj, None)
    if w is None:
        return False
    try:
        if attr == "text":
            w.setText(str(value))
            return True
        if attr == "value":
            w.setValue(int(value))
            return True
        if attr == "setCurrentText":
            w.setCurrentText(str(value))
            return True
    except Exception:
        return False
    return False

def _mk_build_cmd_stub_for(cls):
    if hasattr(cls, "_build_command_dialog"):
        return
    def _build_command_dialog(self):
        engine = _read_val(self, "edit_engine", "text", _read_val(self, "combo_engine", "currentText", "realesrgan-ncnn-vulkan.exe"))
        inp    = _read_val(self, "edit_input", "text")
        outd   = _read_val(self, "edit_outdir", "text")
        model  = _read_val(self, "edit_model", "text", _read_val(self, "combo_model", "currentText", "realesr-general-x4v3"))
        scale  = _read_val(self, "spin_scale", "value", 4)

        parts = []
        if engine: parts += [f'"{engine}"']
        if inp:    parts += ['-i', f'"{inp}"']
        if outd:   parts += ['-o', f'"{outd}"']
        if model:  parts += ['-n', model]
        if scale:  parts += ['-s', str(scale)]
        cmd = " ".join(parts) if parts else "(no fields set)"

        try:
            QMessageBox.information(self, "Command preview", cmd)
        except Exception as e:
            QMessageBox.warning(self, "Build command failed", str(e))
    setattr(cls, "_build_command_dialog", _build_command_dialog)

def _mk_save_json_stub_for(cls):
    if hasattr(cls, "_save_json"):
        return
    def _save_json(self):
        data = {
            "engine": _read_val(self, "edit_engine", "text", _read_val(self, "combo_engine", "currentText", "")),
            "input": _read_val(self, "edit_input", "text"),
            "outdir": _read_val(self, "edit_outdir", "text"),
            "model": _read_val(self, "edit_model", "text", _read_val(self, "combo_model", "currentText", "")),
            "scale": _read_val(self, "spin_scale", "value", 4),
        }
        default_dir = data["outdir"] or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(self, "Save settings JSON", default_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
    setattr(cls, "_save_json", _save_json)

def _mk_load_json_stub_for(cls):
    if hasattr(cls, "_load_json"):
        return
    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings JSON", os.getcwd(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"{e}")
            return
        _set_val(self, "edit_engine", "text", data.get("engine"))
        if not _set_val(self, "edit_engine", "text", data.get("engine")):
            _set_val(self, "combo_engine", "setCurrentText", data.get("engine"))
        _set_val(self, "edit_input", "text", data.get("input"))
        _set_val(self, "edit_outdir", "text", data.get("outdir"))
        if not _set_val(self, "edit_model", "text", data.get("model")):
            _set_val(self, "combo_model", "setCurrentText", data.get("model"))
        try:
            _set_val(self, "spin_scale", "value", data.get("scale", 4))
        except Exception:
            pass
        QMessageBox.information(self, "Loaded", f"Settings loaded from:\n{path}")
    setattr(cls, "_load_json", _load_json)

def _mk_persist_model_choice_stub_for(cls):
    if hasattr(cls, "_persist_dml_model_choice"):
        return
    def _persist_dml_model_choice(self):
        try:
            from PySide6.QtCore import QSettings
            val = _read_val(self, "combo_dml_model", "currentText", "")
            s = QSettings("FrameVision", "Upscaler")
            s.setValue("dml_model", val)
            s.sync()
            lbl = getattr(self, "lbl_model_hint", None)
            if lbl:
                lbl.setText(f"Saved choice: {val}")
        except Exception:
            pass
    setattr(cls, "_persist_dml_model_choice", _persist_dml_model_choice)

# ---- Real download actions: open correct web pages and the models folder ----
_REAL_SR_URL = "https://github.com/xinntao/Real-ESRGAN/releases"
_REALESRGAN_NCNN_URL = "https://github.com/nihui/realesrgan-ncnn-vulkan/releases"
_RIFE_NCNN_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases"
_SWINIR_INFO_URL = "https://github.com/JingyunLiang/SwinIR"  # docs; user can find ONNX/weights links here

def _mk_download_real_for(cls):
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QDesktopServices

    def _ensure_models_dir(self):
        try:
            s = QSettings("FrameVision", "Upscaler")
            root = s.value("ModelsRoot", "", str)
        except Exception:
            root = ""
        if not root:
            root = os.path.join(os.getcwd(), "models")
        try:
            os.makedirs(root, exist_ok=True)
        except Exception:
            pass
        return root

    def _open_url(url: str):
        try:
            webbrowser.open(url)
        except Exception:
            try:
                QDesktopServices.openUrl(QUrl(url))
            except Exception:
                pass

    def _decide_urls(model_text: str):
        lt = (model_text or "").lower()
        urls = []
        if "rife" in lt:
            urls.append(_RIFE_NCNN_URL)
        if "realesrgan" in lt or "real-esr" in lt or "realesr-" in lt or "x4v3" in lt:
            # show both nihui's NCNN tool (binary+models) and xinntao release page
            urls.extend([_REALESRGAN_NCNN_URL, _REAL_SR_URL])
        if "swinir" in lt or "onnx" in lt:
            urls.append(_SWINIR_INFO_URL)
        # Fallback: if we didn't match anything, show the two most common:
        if not urls:
            urls = [_REALESRGAN_NCNN_URL, _RIFE_NCNN_URL]
        return urls

    if not hasattr(cls, "_choose_models_dir"):
        def _choose_models_dir(self):
            path = QFileDialog.getExistingDirectory(self, "Choose models directory", _ensure_models_dir(self))
            if not path:
                return
            try:
                s = QSettings("FrameVision", "Upscaler")
                s.setValue("ModelsRoot", path); s.sync()
            except Exception:
                pass
            msg = getattr(self, "lbl_model_hint", None)
            if msg:
                msg.setText(f"Models folder set: {path}")
        setattr(cls, "_choose_models_dir", _choose_models_dir)

    if not hasattr(cls, "_open_models_folder"):
        def _open_models_folder(self):
            path = _ensure_models_dir(self)
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception:
                QMessageBox.information(self, "Models folder", path)
        setattr(cls, "_open_models_folder", _open_models_folder)

    def _do_selected(self):
        root = _ensure_models_dir(self)
        # try common sources based on current combo text
        model = ""
        for name in ("combo_model","edit_model","combo_dml_model"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    model = w.currentText().strip() if hasattr(w,"currentText") else w.text().strip()
                except Exception:
                    pass
            if model:
                break
        urls = _decide_urls(model)
        for u in urls:
            _open_url(u)
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(root))
        except Exception:
            pass
        QMessageBox.information(self, "Open downloads",
            "Opened the recommended download pages in your browser.\n"
            f"Models folder:\n{root}\n\n"
            "Download the model files to that folder, then click Rescan.\n"
        )

    def _do_all(self):
        root = _ensure_models_dir(self)
        # open the major sources for both upscalers
        for u in (_REALESRGAN_NCNN_URL, _RIFE_NCNN_URL, _REAL_SR_URL, _SWINIR_INFO_URL):
            _open_url(u)
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(root))
        except Exception:
            pass
        QMessageBox.information(self, "Open downloads",
            "Opened all major model sources in your browser.\n"
            f"Models folder:\n{root}\n\n"
            "Grab what you need and click Rescan afterwards."
        )

    # Replace placeholders (or add if missing)
    setattr(cls, "_download_selected", _do_selected)
    setattr(cls, "_download_all", _do_all)

def _mk_rescan_models_stub_for(cls):
    need_rescan = not hasattr(cls, "_rescan_models")
    need_scan_fill = not hasattr(cls, "_scan_and_fill_models")
    if not (need_rescan or need_scan_fill):
        return

    def _rescan_impl(self):
        roots = [Path("models"), Path("tools")/"realesrgan", Path("tools")/"rife", Path.cwd()]
        found = set()
        for r in roots:
            if not r.exists():
                continue
            for p in r.rglob("*.onnx"):
                found.add(str(p))
        combo = getattr(self, "combo_dml_model", None)
        if combo is not None:
            combo.clear()
            if found:
                for s in sorted(found):
                    combo.addItem(s)
                hint = getattr(self, "lbl_model_hint", None)
                if hint:
                    hint.setText(f"Found {len(found)} .onnx model(s)")
            else:
                combo.addItem("(no .onnx models found)")

    if need_rescan:
        setattr(cls, "_rescan_models", _rescan_impl)
    if need_scan_fill:
        setattr(cls, "_scan_and_fill_models", _rescan_impl)

def _mk_apply_settings_stub_for(cls):
    if hasattr(cls, "_apply_settings"):
        return
    def _apply_settings(self, settings: dict):
        if not isinstance(settings, dict):
            return
        try:
            _set_val(self, "edit_engine", "text", settings.get("engine"))
            if not _set_val(self, "edit_engine", "text", settings.get("engine")):
                _set_val(self, "combo_engine", "setCurrentText", settings.get("engine"))
            _set_val(self, "edit_input", "text", settings.get("input"))
            _set_val(self, "edit_outdir", "text", settings.get("outdir"))
            if not _set_val(self, "edit_model", "text", settings.get("model")):
                _set_val(self, "combo_model", "setCurrentText", settings.get("model"))
            try:
                _set_val(self, "spin_scale", "value", int(settings.get("scale", 4)))
            except Exception:
                pass
            dml = settings.get("dml_model") or settings.get("directml_model")
            if dml:
                _set_val(self, "combo_dml_model", "setCurrentText", dml)
            lbl = getattr(self, "lbl_model_hint", None)
            if lbl: lbl.setText("Settings applied")
        except Exception:
            pass
    setattr(cls, "_apply_settings", _apply_settings)

def _monkey_patch_upscaler_class(cls):
    _mk_build_cmd_stub_for(cls)
    _mk_save_json_stub_for(cls)
    _mk_load_json_stub_for(cls)
    _mk_persist_model_choice_stub_for(cls)
    _mk_rescan_models_stub_for(cls)
    _mk_apply_settings_stub_for(cls)
    _mk_download_real_for(cls)  # now opens real pages, not placeholders


def _relax_sizes(root_widget):
    """Walk the upscaler widget tree and relax any hard size constraints so it scrolls instead of clipping."""
    try:
        from PySide6.QtWidgets import QWidget, QSizePolicy, QLayout
        QWIDGETSIZE_MAX = 16777215
        def _tune(w: QWidget):
            try:
                w.setMinimumSize(0, 0)
                w.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
                sp = w.sizePolicy()
                sp.setHorizontalPolicy(QSizePolicy.Preferred)
                sp.setVerticalPolicy(QSizePolicy.Preferred)
                sp.setHorizontalStretch(0); sp.setVerticalStretch(0)
                w.setSizePolicy(sp)
                lay = w.layout()
                if isinstance(lay, QLayout):
                    lay.setContentsMargins(8, 8, 8, 8)
                    lay.setSpacing(max(6, lay.spacing()))
                    lay.setSizeConstraint(QLayout.SetMinAndMaxSize)
            except Exception:
                pass
        _tune(root_widget)
        for ch in root_widget.findChildren(QWidget):
            _tune(ch)
        try:
            root_widget.adjustSize()
        except Exception:
            pass
    except Exception:
        pass
# ----------------- generic loader -----------------

def _instantiate(obj, module_hint: str | None = None):
    if inspect.isclass(obj):
        if module_hint == "upscaler":
            _monkey_patch_upscaler_class(obj)
        return obj()
    if callable(obj):
        return obj()
    return obj

def _find_widget(mod, names, module_hint: str | None = None):
    for n in names:
        if hasattr(mod, n):
            return _instantiate(getattr(mod, n), module_hint), f"class_or_factory:{n}"
    for fn in ("create_widget","build_widget","main_widget","create_ui","build_ui","create","build"):
        if hasattr(mod, fn):
            return _instantiate(getattr(mod, fn), module_hint), f"factory:{fn}"
    from PySide6.QtWidgets import QWidget as _QW
    cands = []
    for n, obj in mod.__dict__.items():
        try:
            if inspect.isclass(obj) and issubclass(obj, _QW):
                score = 0; ln=n.lower()
                if module_hint == "describer" and ("describe" in ln or "caption" in ln): score += 10
                if module_hint == "upscaler" and ("upscale" in ln or "realesr" in ln or "swinir" in ln): score += 10
                if ln.endswith("widget") or ln.endswith("window"): score += 2
                cands.append((score, n, obj))
        except Exception:
            pass
    if cands:
        cands.sort(reverse=True)
        _, n, obj = cands[0]
        return _instantiate(obj, module_hint), f"heuristic:{n}"
    raise ImportError("No QWidget subclass found in module.")

def _load(module_candidates, class_candidates, module_hint: str | None = None):
    mod, used = _import_module(module_candidates)
    w, how = _find_widget(mod, class_candidates, module_hint)
    return w, f"{used} via {how}"

def _error_panel(title: str, exc: Exception):
    box = QWidget()
    v = QVBoxLayout(box); v.setContentsMargins(8,8,8,8); v.setSpacing(6)
    lbl = QLabel(f"{title}"); lbl.setStyleSheet("color:#b91c1c; font-weight:600;")
    v.addWidget(lbl)
    ed = QPlainTextEdit(); ed.setReadOnly(True); ed.setMinimumHeight(230)
    ed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ed.setPlainText(tb)
    v.addWidget(ed, 1)
    row = QHBoxLayout(); btn = QPushButton("Copy details"); row.addWidget(btn); row.addStretch(1); v.addLayout(row)
    btn.clicked.connect(lambda: (ed.selectAll(), ed.copy()))
    return box


class UpscTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        try:
            try:
                from helpers import upsc as _upsc_mod
            except Exception as _e1:
                try:
                    from helpers.framevision_app import ROOT as _ROOT
                except Exception:
                    _ROOT = Path(__file__).resolve().parent.parent
                _p = _ROOT / "helpers" / "upsc.py"
                if _p.exists():
                    import importlib.util
                    _spec = importlib.util.spec_from_file_location("helpers._upsc_dyn", str(_p))
                    _upsc_mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_upsc_mod)  # type: ignore
                else:
                    raise _e1
            # Find pane class
            _cls = None
            for _name in ("UpscPane","UpscTab","UpscalePane","UpscaleTab"):
                _cls = getattr(_upsc_mod, _name, None)
                if _cls: break
            if _cls is None:
                raise ImportError("helpers.upsc found but no pane class exported")
            self.inner = _cls(self)
            try:
                setter = getattr(self.inner, "set_main", None)
                if setter: setter(self.parent())
            except Exception: pass
            try:
                from PySide6.QtWidgets import QMainWindow
                if isinstance(self.inner, QMainWindow):
                    self.inner.setParent(self); self.inner.setWindowFlags(Qt.Widget); lay.addWidget(self.inner)
                else:
                    sc = QScrollArea(); sc.setWidgetResizable(True)
                    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                    sc.setFrameShape(QFrame.NoFrame); sc.setFocusPolicy(Qt.NoFocus)
                    sc.setWidget(self.inner); lay.addWidget(sc)
            except Exception:
                lay.addWidget(self.inner)
        except Exception as e:
            lay.addWidget(_error_panel("Failed to load Upscale tab", e))

class DescriberTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.inner = None
        try:
            inner, note = _load(["helpers.describer","describer"],
                                ["DescriberWidget","DescriberPanel","SettingsWindow","DescriberWindow"],
                                module_hint="describer")
            self.inner = inner
            try:
                from PySide6.QtWidgets import QMainWindow
                if isinstance(self.inner, QMainWindow):
                    self.inner.setParent(self); self.inner.setWindowFlags(Qt.Widget)
                    lay.addWidget(self.inner)
                else:
                    sc = QScrollArea()
                    sc.setWidgetResizable(True)
                    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                    sc.setFocusPolicy(Qt.NoFocus); sc.setFrameShape(QFrame.NoFrame)
                    self.inner.setParent(sc); sc.setWidget(self.inner)
                    sc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    lay.addWidget(sc)
                    try:
                        _relax_sizes(self.inner)
                    except Exception:
                        pass  # relax sizes in Upscaler
            except Exception:
                lay.addWidget(self.inner)

            note_lbl = QLabel(f"Loaded: {note}"); note_lbl.setStyleSheet("color:#6b7280; font-size:11px;")
            lay.addWidget(note_lbl)

            # Aggressive reflow: run now and a bit later to beat late resizes
            try:
                self.btn_fix_layout.clicked.connect(lambda: _relax_sizes(self.inner))
                QTimer.singleShot(0, lambda: _relax_sizes(self.inner))
                QTimer.singleShot(1500, lambda: _relax_sizes(self.inner))
                QTimer.singleShot(4000, lambda: _relax_sizes(self.inner))
            except Exception:
                pass


        except Exception as e:
            lay.addWidget(_error_panel("Describer failed to load", e))

    def on_pause_capture(self, qimg):
        if self.inner is None: return
        try:
            if hasattr(self.inner, 'caption_from_qimage'): self.inner.caption_from_qimage(qimg, mode='short'); return
            import uuid
            tmp = Path(tempfile.gettempdir())/f"framevision_cap_{uuid.uuid4().hex}.png"
            qimg.save(str(tmp), "PNG")
            if hasattr(self.inner, 'set_image_path') and hasattr(self.inner, 'generate'):
                self.inner.set_image_path(str(tmp)); self.inner.generate(mode='short'); return
            if hasattr(self.inner, 'edit_test_image'): self.inner.edit_test_image.setText(str(tmp))
            if hasattr(self.inner, '_gen_short_desc'): self.inner._gen_short_desc()
        except Exception:
            pass



def integrate_into_main_window(main):
    """Ensure Upscaler, Rife Fps (legacy Edit), Describe at front and remove duplicate 'Rife Fps' tabs."""
    tabs = getattr(main, 'tabs', None)
    if tabs is None:
        return
    # Remove any non-Edit 'Rife Fps' tabs
    try:
        for i in sorted(range(tabs.count()), reverse=True):
            nm = (tabs.tabText(i) or '').strip().lower()
            if nm == 'rife fps':
                if getattr(main, 'edit', None) is None or tabs.widget(i) is not main.edit:
                    tabs.removeTab(i)
    except Exception:
        pass
    
    # Ensure "Upscale" tab at index 0 (only upscaling tab)
    upsc_idx = None
    for i in range(tabs.count()):
        if (tabs.tabText(i) or '').strip().lower() == 'upscale':
            upsc_idx = i; break
    if upsc_idx is None:
        try:
            upsc_tab = UpscTab(main); tabs.insertTab(0, upsc_tab, 'Upscale'); upsc_idx = 0; main.upsc_tab = upsc_tab
        except Exception:
            pass
    elif upsc_idx != 0:
        try:
            w = tabs.widget(upsc_idx); tabs.removeTab(upsc_idx); tabs.insertTab(0, w, 'Upscale'); main.upsc_tab = w
        except Exception:
            pass

    # Rename/move legacy Edit -> Rife Fps at index 1

    try:
        legacy = getattr(main, 'edit', None)
        if legacy is not None:
            for i in range(tabs.count()):
                if tabs.widget(i) is legacy:
                    tabs.removeTab(i); break
            tabs.insertTab(1, legacy, 'Rife Fps')
    except Exception:
        pass
    # Ensure Describe at index 2
    desc_idx = None
    for i in range(tabs.count()):
        if (tabs.tabText(i) or '').strip().lower() == 'describe':
            desc_idx = i; break
    if desc_idx is None:
        try:
            dt = DescriberTab(main); tabs.insertTab(2, dt, 'Describe'); main.describer_tab = dt
        except Exception:
            pass
    elif desc_idx != 2:
        try:
            w = tabs.widget(desc_idx); tabs.removeTab(desc_idx); tabs.insertTab(2, w, 'Describe'); main.describer_tab = w
        except Exception:
            pass
    try: _enforce_scroll_policies(main)
    except Exception: pass
    try: _wrap_video_open(main)
    except Exception: pass
    try: _apply_profile_preferences(main)
    except Exception: pass
    return main
def _enforce_scroll_policies(widget):
    # Apply to all QScrollArea descendants
    try:
        from PySide6.QtWidgets import QScrollArea
        from PySide6.QtCore import Qt
        for sa in widget.findChildren(QScrollArea):
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sa.setFocusPolicy(Qt.NoFocus)
    except Exception:
        pass