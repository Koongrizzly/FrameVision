# helpers/describer_impl.py
from __future__ import annotations

import os, json, zipfile, traceback
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import QEvent, QSettings, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget
)




# --- Robust import guard for QCheckBox (may be used by warm-on-load option) ---
if 'QCheckBox' not in globals():
    try:
        from PySide6.QtWidgets import QCheckBox  # type: ignore
    except Exception:
        try:
            from PyQt5.QtWidgets import QCheckBox  # type: ignore
        except Exception:
            pass
# --- end guard ---
# ------------------------- Utilities -------------------------

APP_ORG = "FrameVision"
APP_NAME = "Describe"

def _settings() -> QSettings:
    return QSettings(APP_ORG, APP_NAME)

def human(path: Path) -> str:
    return str(path.resolve())

def find_project_root() -> Path:
    return Path(".").resolve()

def models_root() -> Path:
    s = _settings().value("models_root", "", type=str)
    if s:
        return Path(s)
    # default under project models/
    return find_project_root() / "models"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_bundled(pattern: str) -> Optional[Path]:
    root = find_project_root() / "models" / "_bundled"
    for f in root.glob(pattern):
        if f.is_file():
            return f
    return None

def auto_extract_if_missing(target_dir: Path, bundle_patterns: list[str], progress: Optional[QProgressBar]=None) -> bool:
    """Extract first matching bundled zip into target_dir if missing. Returns True if ready."""
    if target_dir.exists() and any(target_dir.iterdir()):
        return True
    for pat in bundle_patterns:
        z = find_bundled(pat)
        if z and z.exists():
            try:
                ensure_dir(target_dir)
                if progress:
                    progress.setMaximum(0); progress.setValue(-1)
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(target_dir)
                if progress:
                    progress.setMaximum(1); progress.setValue(1)
                return True
            except Exception:
                pass
    return target_dir.exists() and any(target_dir.iterdir())

def file_exists_any(p: Path, names: list[str]) -> bool:
    return any((p / n).exists() for n in names)

# ------------------------- Engines -------------------------

ENGINE_CATALOG: Dict[str, Dict[str, Any]] = {
    "blip-base": {
        "label": "BLIP (base, Transformers)",
        "size": "~0.9 GB cache",
        "best": "Short + simple captions",
        "notes": "Salesforce/blip-image-captioning-base",
        "runs_on": "CPU or CUDA",
        "folder": "describe/default/blip-image-captioning-base",
        "bundle": ["*blip*base*.zip"],
        "required": ["config.json", "pytorch_model.bin", "tokenizer_config.json"],
        "type": "hf_blip",
    },
    "blip-large": {
        "label": "BLIP (large, Transformers)",
        "size": "~1.2 GB cache",
        "best": "More detailed captions",
        "notes": "Salesforce/blip-image-captioning-large",
        "runs_on": "CPU or CUDA",
        "folder": "describe/default/blip-image-captioning-large",
        "bundle": ["*blip*large*.zip"],
        "required": ["config.json", "pytorch_model.bin", "tokenizer_config.json"],
        "type": "hf_blip",
    },
    "qwen2-vl-2b": {
        "label": "Qwen2-VL (2B Instruct)",
        "size": "~3.0 GB cache",
        "best": "Long, multi-sentence, rich descriptions",
        "notes": "Qwen/Qwen2-VL-2B-Instruct",
        "runs_on": "CUDA recommended; CPU works (slower)",
        "folder": "describe/default/qwen2-vl-2b-instruct",
        "bundle": ["*qwen2*vl*2b*instruct*.zip"],
        "required": ["config.json", "model.safetensors", "tokenizer.json"],
        "type": "hf_qwen2vl",
    },
}

# ------------------------- Widget -------------------------

class DescriberWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

        s = _settings()
        self.engine_key = "qwen2-vl-2b"

        self.detail_level = s.value("detail_level", "Long", type=str)
        if self.detail_level not in ("Short","Medium","Long"):
            self.detail_level = "Long"

        self.decode_style = s.value("decode_style", "Deterministic", type=str)
        if self.decode_style not in ("Deterministic","Creative"):
            self.decode_style = "Deterministic"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        # Hide host-injected "Loaded: helpers.describer..." footer label
        def __hide_loader_blurb():
            try:
                w = self.window()
                if not w:
                    return
                from PySide6.QtWidgets import QLabel as _QLabel
                for _lbl in w.findChildren(_QLabel):
                    try:
                        if _lbl.text().strip().startswith("Loaded:"):
                            _lbl.setVisible(False)
                    except Exception:
                        pass
            except Exception:
                pass
        QTimer.singleShot(0, __hide_loader_blurb)


        
        # --- Engine info (Qwen-only) ---
        eng_widget = QWidget()
        # Hide engine panel entirely per user request
        eng_widget.setVisible(False)
        eng_widget.setMaximumHeight(0)
        eng_widget.setMinimumHeight(0)
        eng_grid = QGridLayout(eng_widget)
        row = 0

        eng_grid.addWidget(QLabel("Engine:"), row, 0)
        self.lbl_engine_name = QLabel("Qwen2-VL (2B Instruct)")
        eng_grid.addWidget(self.lbl_engine_name, row, 1, 1, 3); row += 1

        self.lbl_size = QLabel(""); eng_grid.addWidget(QLabel("Size:"), row, 0); eng_grid.addWidget(self.lbl_size, row, 1)
        self.lbl_runs = QLabel(""); eng_grid.addWidget(QLabel("Runs on:"), row, 2); eng_grid.addWidget(self.lbl_runs, row, 3); row += 1

        self.lbl_best = QLabel(""); eng_grid.addWidget(QLabel("Best for:"), row, 0); eng_grid.addWidget(self.lbl_best, row, 1, 1, 3); row += 1
        self.lbl_notes = QLabel(""); eng_grid.addWidget(QLabel("Notes:"), row, 0); eng_grid.addWidget(self.lbl_notes, row, 1, 1, 3); row += 1

        eng_grid.addWidget(QLabel("Models root:"), row, 0)
        self.lbl_models_root = QLabel(human(models_root())); eng_grid.addWidget(self.lbl_models_root, row, 1, 1, 2)
        self.lbl_ready = QLabel(""); self.lbl_ready.setMinimumWidth(120)
        eng_grid.addWidget(self.lbl_ready, row, 3); row += 1

        self.status_chip = self.lbl_ready
        self.prep_progress = QProgressBar(); self.prep_progress.setVisible(False)
        lay.addWidget(eng_widget)

        # --- Actions ---
        act_box = QGroupBox("Actions")
        act_row = QHBoxLayout(act_box)
        self.btn_describe = QPushButton("Describe")
        self.btn_desc_folder = QPushButton("Describe folder...")
        self.btn_use_current = QPushButton("Use current frame")
        act_row.addWidget(self.btn_describe); act_row.addWidget(self.btn_desc_folder); act_row.addWidget(self.btn_use_current)
        lay.addWidget(act_box)

        # --- Output ---
        out_box = QGroupBox("Generated description")
        out_v = QVBoxLayout(out_box)
        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMinimumHeight(360)
        out_v.addWidget(self.output)
        # Auto-size output to fully fit wrapped content; clamp 120–960; persist height when Save settings is ON
        try:
            self.output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # keep width flexible but height controlled by our autosize
            self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        def _persist_output_height(h:int):
            try:
                # Only persist if user opted in
                if hasattr(self, "chk_save_settings") and not self.chk_save_settings.isChecked():
                    return
                from pathlib import Path as _P; import json as _J
                pdir = _P(__file__).resolve().parent.parent / "presets" / "setsave"; pdir.mkdir(parents=True, exist_ok=True)
                pp = pdir / "describer.json"
                data = {}
                if pp.exists():
                    try: data = _J.loads(pp.read_text(encoding="utf-8"))
                    except Exception: data = {}
                data["output_height"] = int(h)
                pp.write_text(_J.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
        def _autosize_output():
            try:
                doc = self.output.document()
                # Make layout aware of current wrap width
                try:
                    vw = max(10, self.output.viewport().width() - 2)
                    doc.setTextWidth(vw)
                except Exception:
                    pass
                # Prefer layout-computed height (handles wrapped lines precisely)
                try:
                    h = int(doc.documentLayout().documentSize().toSize().height())
                except Exception:
                    fm = self.output.fontMetrics(); h = fm.lineSpacing() * max(1, doc.blockCount())
                try:
                    m = self.output.contentsMargins(); fw = getattr(self.output, "frameWidth", lambda: 0)()
                    h += m.top() + m.bottom() + fw*2
                except Exception:
                    pass
                h = max(120, min(960, h))
                self.output.setFixedHeight(h)
                _persist_output_height(h)
            except Exception:
                pass
        # Expose as method so eventFilter can call it
        self._auto_size_output = _autosize_output
        try:
            self.output.textChanged.connect(lambda: QTimer.singleShot(0, _autosize_output))
            self.output.installEventFilter(self)
        except Exception:
            pass
        try:
            # Initial height: use saved height if present, then autosize to content
            from pathlib import Path as _P; import json as _J
            pp = _P(__file__).resolve().parent.parent / "presets" / "setsave" / "describer.json"
            if pp.exists():
                try:
                    data = _J.loads(pp.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "output_height" in data:
                        h = max(120, min(960, int(data["output_height"]))); self.output.setFixedHeight(h)
                except Exception:
                    pass
            QTimer.singleShot(0, _autosize_output)
        except Exception:
            pass

        row2 = QHBoxLayout(); out_v.addLayout(row2)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy_prompt = QPushButton("Copy as Prompt")
        self.btn_save_txt = QPushButton("Save .txt")
        self.btn_save_json = QPushButton("Save .json")
        row2.addWidget(self.btn_copy); row2.addWidget(self.btn_copy_prompt); row2.addWidget(self.btn_save_txt); row2.addWidget(self.btn_save_json)

        # Output options 

        row3 = QHBoxLayout(); out_v.addLayout(row3)

        row3.addWidget(QLabel("Detail level:")); self.combo_detail = QComboBox(); self.combo_detail.addItems(["Short","Medium","Long"]); self.combo_detail.setCurrentText(self.detail_level); row3.addWidget(self.combo_detail)

        row3.addWidget(QLabel("Decode style:")); self.combo_decode = QComboBox(); self.combo_decode.addItems(["Deterministic","Creative"]); self.combo_decode.setCurrentText(self.decode_style); row3.addWidget(self.combo_decode)

        row3.addStretch(1)
        self.negative = QLineEdit(); self.negative.setPlaceholderText("Negative prompt (optional)..."); out_v.addWidget(self.negative)
        lay.addWidget(out_box)

        # --- Paths ---
        path_box = QGroupBox("Paths")
        g = QGridLayout(path_box)
        g.addWidget(QLabel("Test image path:"), 0, 0)
        self.txt_image = QLineEdit(); g.addWidget(self.txt_image, 0, 1)
        self.btn_browse = QPushButton("Browse..."); g.addWidget(self.btn_browse, 0, 2)
        self.lbl_player = QLabel(""); self.lbl_player.setVisible(False); self.lbl_player.setMaximumHeight(0)  # hidden
        lay.addWidget(path_box)

        # --- Advanced (collapsible) ---
        open_state = _settings().value("advanced_open", True, bool)
        self.btn_adv = QToolButton()
        self.btn_adv.setText("Advanced settings")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_adv.setArrowType(Qt.DownArrow if open_state else Qt.RightArrow)
        self.btn_adv.setChecked(bool(open_state))
        adv = QGroupBox()
        adv.setTitle("")
        adv.setFlat(False)
        adv.setVisible(bool(open_state))
        gg = QGridLayout(adv)
        r = 0
        # --- Settings persistence controls (top of Advanced) ---
        row_save = QWidget(self); hsave = QHBoxLayout(row_save); hsave.setContentsMargins(0,0,0,0)
        self.chk_save_settings = QCheckBox("Save settings")
        self.chk_save_settings.setToolTip("When ON, any change updates presets\\setsave\\describer.json and will be restored on next launch.")
        self.btn_defaults = QPushButton("Default settings")
        self.btn_defaults.setToolTip("Restore all Advanced options to built-in defaults.")
        hsave.addWidget(self.chk_save_settings); hsave.addStretch(1); hsave.addWidget(self.btn_defaults)
        gg.addWidget(row_save, r, 0, 1, 2); r+=1

        gg.addWidget(QLabel("max_new_tokens"), r,0); self.sp_max = QSpinBox(); self.sp_max.setRange(16,2048); self.sp_max.setValue(_settings().value("max_new_tokens", 320, int)); gg.addWidget(self.sp_max, r,1); r+=1
        gg.addWidget(QLabel("min_length"), r,0); self.sp_min = QSpinBox(); self.sp_min.setRange(0,1024); self.sp_min.setValue(_settings().value("min_length", 60, int)); gg.addWidget(self.sp_min, r,1); r+=1
        gg.addWidget(QLabel("no_repeat_ngram_size"), r,0); self.sp_ngram = QSpinBox(); self.sp_ngram.setRange(0,10); self.sp_ngram.setValue(_settings().value("no_repeat_ngram_size", 3, int)); gg.addWidget(self.sp_ngram, r,1); r+=1
        gg.addWidget(QLabel("temperature"), r,0); self.sp_temp = QDoubleSpinBox(); self.sp_temp.setRange(0.0,2.0); self.sp_temp.setSingleStep(0.1); self.sp_temp.setValue(_settings().value("temperature", 0.7, float)); gg.addWidget(self.sp_temp, r,1); r+=1
        gg.addWidget(QLabel("top_p"), r,0); self.sp_topp = QDoubleSpinBox(); self.sp_topp.setRange(0.0,1.0); self.sp_topp.setSingleStep(0.05); self.sp_topp.setValue(_settings().value("top_p", 0.9, float)); gg.addWidget(self.sp_topp, r,1); r+=1
        gg.addWidget(QLabel("top_k"), r,0); self.sp_topk = QSpinBox(); self.sp_topk.setRange(0,200); self.sp_topk.setValue(_settings().value("top_k", 50, int)); gg.addWidget(self.sp_topk, r,1); r+=1
        gg.addWidget(QLabel("repetition_penalty"), r,0); self.sp_rep = QDoubleSpinBox(); self.sp_rep.setRange(0.0, 4.0); self.sp_rep.setValue(_settings().value("repetition_penalty", 1.1, float)); gg.addWidget(self.sp_rep, r,1); r+=1
        self.chk_warm = QCheckBox("Warm model on load"); self.chk_warm.setChecked(_settings().value("warm_on_load", True, bool)); gg.addWidget(self.chk_warm, r,0,1,2); r+=1
        self.chk_keep = QCheckBox("Keep in memory"); self.chk_keep.setChecked(_settings().value("keep_in_memory", True, bool)); gg.addWidget(self.chk_keep, r,0,1,2); r+=1
        lay.addWidget(self.btn_adv)
        lay.addWidget(adv)

        # Signals
        pass  # engine selection removed; Qwen-only
        self.btn_browse.clicked.connect(self._browse)
        self.btn_describe.clicked.connect(self._describe_clicked)
        self.btn_copy.clicked.connect(lambda: self._copy(self.output.toPlainText()))
        self.btn_copy_prompt.clicked.connect(self._copy_prompt)
        self.btn_save_txt.clicked.connect(self._save_txt)
        self.btn_save_json.clicked.connect(self._save_json)
        # --- Settings persistence: helpers (scoped in __init__) ---
        def _setsave_path():
            root = Path(__file__).resolve().parent.parent
            pdir = root / "presets" / "setsave"
            try: pdir.mkdir(parents=True, exist_ok=True)
            except Exception: pass
            return pdir / "describer.json"
        def _gather():
            d = {}
            try:
                if hasattr(self, "spin_cpu"): d["spin_cpu"] = int(self.spin_cpu.value())
            except Exception: pass
            for name in ("sp_max","sp_min","sp_ngram","sp_temp","sp_topp","sp_topk","sp_rep"):
                try:
                    if hasattr(self,name):
                        w = getattr(self,name)
                        d[name] = float(w.value()) if "Double" in type(w).__name__ else int(w.value())
                except Exception: pass
            for name in ("chk_warm","chk_keep"):
                try:
                    if hasattr(self,name): d[name] = bool(getattr(self,name).isChecked())
                except Exception: pass
            for name in ("combo_detail","combo_decode"):
                try:
                    if hasattr(self,name): d[name] = str(getattr(self,name).currentText())
                except Exception: pass
            for name in ("negative","txt_image"):
                try:
                    if hasattr(self,name): d[name] = str(getattr(self,name).text())
                except Exception: pass
            try:
                if "output_height" in d and hasattr(self, "output"):
                    h = max(120, min(960, int(d["output_height"])))
                    self.output.setFixedHeight(h)
            except Exception:
                pass
            try: d["save_settings"] = bool(self.chk_save_settings.isChecked())
            except Exception: pass
            try:
                if hasattr(self, "output"): d["output_height"] = int(max(120, min(960, self.output.height())))
            except Exception:
                pass
            return d
        def _apply(d):
            try:
                if "spin_cpu" in d and hasattr(self,"spin_cpu"): self.spin_cpu.setValue(int(d["spin_cpu"]))
            except Exception: pass
            for name in ("sp_max","sp_min","sp_ngram","sp_temp","sp_topp","sp_topk","sp_rep"):
                try:
                    if name in d and hasattr(self,name):
                        w = getattr(self,name)
                        val = d[name]
                        w.setValue(float(val) if "Double" in type(w).__name__ else int(val))
                except Exception: pass
            for name in ("chk_warm","chk_keep"):
                try:
                    if name in d and hasattr(self,name): getattr(self,name).setChecked(bool(d[name]))
                except Exception: pass
            for name in ("combo_detail","combo_decode"):
                try:
                    if name in d and hasattr(self,name): getattr(self,name).setCurrentText(str(d[name]))
                except Exception: pass
            for name in ("negative","txt_image"):
                try:
                    if name in d and hasattr(self,name): getattr(self,name).setText(str(d[name]))
                except Exception: pass
            try:
                if "save_settings" in d and hasattr(self,"chk_save_settings"): self.chk_save_settings.setChecked(bool(d["save_settings"]))
            except Exception: pass
        def _save_now():
            try:
                if hasattr(self,"chk_save_settings") and not self.chk_save_settings.isChecked(): return
                pp = _setsave_path(); data = _gather()
                pp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception: pass
        def _load_now():
            try:
                pp = _setsave_path()
                if pp.exists():
                    data = json.loads(pp.read_text(encoding="utf-8"))
                    _apply(data)
                    try: self.chk_save_settings.setChecked(bool(data.get("save_settings", True)))
                    except Exception: pass
                else:
                    try: self.chk_save_settings.setChecked(False)
                    except Exception: pass
            except Exception: pass
        def _restore_defaults():
            defaults = {"sp_max": 320, "sp_min":60, "sp_ngram":3, "sp_temp":0.7, "sp_topp":0.9, "sp_topk":50, "sp_rep":1.1, "chk_warm":True, "chk_keep":True}
            for k,v in defaults.items():
                try:
                    if hasattr(self,k):
                        w = getattr(self,k)
                        (w.setChecked(bool(v)) if k.startswith("chk_") else w.setValue(float(v) if "Double" in type(w).__name__ else int(v)))
                except Exception: pass
            _save_now()
        def _hook_autosave(w):
            try:
                if hasattr(w, "valueChanged"): w.valueChanged.connect(_save_now)
            except Exception: pass
            try:
                if hasattr(w, "textChanged"): w.textChanged.connect(_save_now)
            except Exception: pass
            try:
                if hasattr(w, "currentTextChanged"): w.currentTextChanged.connect(_save_now)
            except Exception: pass
            try:
                if hasattr(w, "toggled"): w.toggled.connect(_save_now)
            except Exception: pass
        try:
            self.btn_defaults.clicked.connect(_restore_defaults)
            self.chk_save_settings.toggled.connect(lambda _v: _save_now())
        except Exception: pass
        for nm in ("spin_cpu","sp_max","sp_min","sp_ngram","sp_temp","sp_topp","sp_topk","sp_rep","chk_warm","chk_keep","combo_detail","combo_decode","negative","txt_image"):
            try:
                if hasattr(self, nm): _hook_autosave(getattr(self, nm))
            except Exception: pass
        _tips = {"sp_max":"Upper bound on generated tokens.","sp_min":"Minimum generated length.","sp_ngram":"Blocks repeats of this n-gram length.","sp_temp":"Randomness: higher = more creative.","sp_topp":"Nucleus sampling cut-off.","sp_topk":"Top-K sampling cut-off.","sp_rep":"Penalty for repeated tokens.","chk_warm":"Warm up the model at startup.","chk_keep":"Keep model in memory between runs.","spin_cpu":"Threads for CPU inference (if used).","combo_detail":"Verbosity of output text.","combo_decode":"Decoding strategy for generation.","negative":"Negative prompt tokens to avoid.","txt_image":"Path of test image to describe."}
        for k,v in _tips.items():
            try:
                if hasattr(self,k): getattr(self,k).setToolTip(v)
            except Exception: pass
        _load_now()

        self.combo_detail.currentTextChanged.connect(lambda v: _settings().setValue("detail_level", v))
        self.combo_decode.currentTextChanged.connect(lambda v: _settings().setValue("decode_style", v))
        self.btn_adv.toggled.connect(lambda v: (adv.setVisible(bool(v)), self.btn_adv.setArrowType(Qt.DownArrow if v else Qt.RightArrow), _settings().setValue("advanced_open", bool(v))))

        # Initialize labels + autoload models
        self._refresh_engine_labels()
        self._ensure_ready_async()

    # Autosize support: recompute when output viewport resizes (wrap-aware)
    def eventFilter(self, obj, ev):
        try:
            if hasattr(self, "output") and obj is self.output and ev.type() == QEvent.Resize:
                fn = getattr(self, "_auto_size_output", None)
                if callable(fn):
                    QTimer.singleShot(0, fn)
        except Exception:
            pass
        try:
            return super().eventFilter(obj, ev)
        except Exception:
            return False

    # ----------------- Drag&drop -----------------
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
    def dropEvent(self, e: QDropEvent) -> None:
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".webp"):
                self.txt_image.setText(str(p))
                break

    # ----------------- Helpers -----------------
    def _on_engine_change(self) -> None:
        self.engine_key = self.combo_engine.currentData()
        _settings().setValue("engine_key", self.engine_key)
        self._refresh_engine_labels()
        self._ensure_ready_async()

    def _refresh_engine_labels(self) -> None:
        cfg = ENGINE_CATALOG[self.engine_key]
        self.lbl_size.setText(cfg["size"])
        self.lbl_runs.setText(cfg["runs_on"])
        self.lbl_best.setText(cfg["best"])
        self.lbl_notes.setText(cfg["notes"])
        self.lbl_models_root.setText(str(models_root()))
        self.status_chip.setText("Checking…"); self.status_chip.setStyleSheet("QLabel{background:#ffe9b3;color:#6a4;padding:3px 8px;border-radius:8px;}")

    def _ensure_ready_async(self) -> None:
        # lightweight check+extract on UI thread (fast)
        cfg = ENGINE_CATALOG[self.engine_key]
        target = models_root() / cfg["folder"]
        ready = auto_extract_if_missing(target, cfg["bundle"], self.prep_progress)
        self.prep_progress.setVisible(False)
        if ready and file_exists_any(target, cfg["required"]):
            self.status_chip.setText("Ready"); self.status_chip.setStyleSheet("QLabel{background:#dff5d9;color:#27632a;padding:3px 8px;border-radius:8px;}")
        else:
            self.status_chip.setText("Missing"); self.status_chip.setStyleSheet("QLabel{background:#f9dede;color:#a33;padding:3px 8px;border-radius:8px;}")

    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Choose image", str(find_project_root()), "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if p:
            self.txt_image.setText(p)

    def _copy(self, text: str) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text or "")

    def _copy_prompt(self) -> None:
        desc = self.output.toPlainText().strip()
        if not desc:
            return
        self._copy(desc)
    def _(self, desc: str, negative: str) -> str:
        bits = [
            "Describe the scene precisely with subjects, materials, colors and spatial relationships.",
            "Write in multi-sentence prose; avoid hallucinations; do not invent objects.",
            f"Observation: {desc}",
        ]
        if negative:
            bits.append(f"Do not include: {negative}")
        return " ".join(bits)

    def _save_txt(self) -> None:
        text = self.output.toPlainText()
        if not text: return
        p, _ = QFileDialog.getSaveFileName(self, "Save .txt", "description.txt", "Text (*.txt)")
        if p:
            Path(p).write_text(text, encoding="utf-8")

    def _save_json(self) -> None:
        text = self.output.toPlainText().strip()
        if not text: return
        data = {"engine": self.engine_key, "detail": self.combo_detail.currentText(), "decode": self.combo_decode.currentText(), "text": text}
        p, _ = QFileDialog.getSaveFileName(self, "Save .json", "description.json", "JSON (*.json)")
        if p:
            Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------- Inference -----------------
    def _describe_clicked(self) -> None:
        img = self.txt_image.text().strip()
        if not img or not Path(img).exists():
            self.output.setPlainText("Please choose a valid image path or use current frame.")
            return
        try:
            text = self._run_inference(Path(img))
        except Exception as e:
            text = f"Describe failed: {e}"
        self.output.setPlainText(text)

    def _gen_params(self) -> Dict[str, Any]:
        # Defaults tuned for long output if Long+Deterministic
        detail = self.combo_detail.currentText()
        dec = self.combo_decode.currentText()
        max_new = int(self.sp_max.value())
        min_len = int(self.sp_min.value())
        ngram = int(self.sp_ngram.value())
        temperature = float(self.sp_temp.value())
        top_p = float(self.sp_topp.value())
        top_k = int(self.sp_topk.value())
        rep = float(self.sp_rep.value())

        if dec == "Deterministic":
            return dict(max_new_tokens=max_new, min_length=min_len, no_repeat_ngram_size=ngram, do_sample=False, repetition_penalty=rep)
        else:
            return dict(max_new_tokens=max_new, min_length=min_len, no_repeat_ngram_size=ngram, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=rep)

    def _run_inference(self, img_path: Path) -> str:
        cfg = ENGINE_CATALOG[self.engine_key]
        folder = models_root() / cfg["folder"]
        if not (folder.exists() and any(folder.iterdir())):
            raise RuntimeError("Model not prepared")

        # Lazy import transformers to avoid startup cost
        from PIL import Image
        image = Image.open(str(img_path)).convert("RGB")

        eng_type = cfg["type"]
        gen_kwargs = self._gen_params()

        if eng_type == "hf_blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            # load from local folder
            processor = BlipProcessor.from_pretrained(str(folder), local_files_only=True)
            model = BlipForConditionalGeneration.from_pretrained(str(folder), local_files_only=True)
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs, **gen_kwargs)
            from transformers import AutoTokenizer
            tok = processor.tokenizer
            text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        elif eng_type == "hf_qwen2vl":
            # Qwen2-VL via AutoProcessor/AutoModelForVision2Seq
            from transformers import AutoProcessor, AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)
            model = AutoModelForVision2Seq.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)
            inputs = processor(images=image, text="Describe this image in rich detail.", return_tensors="pt")
            gen = model.generate(**inputs, **gen_kwargs)
            text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        else:
            raise RuntimeError(f"Unknown engine type: {eng_type}")

        # Heuristic length control per detail level (append guidance if too short)
        if self.combo_detail.currentText() == "Long" and len(text.split()) < 40:
            text = text + " " + "(expand: focus on subjects, materials, colors, lighting, composition, background and spatial relationships.)"
        return text

# === FrameVision: Describer ALL-IN-ONE v8 ===
try:
    DescriberWidget  # type: ignore[name-defined]
except Exception:
    pass
else:
    from pathlib import Path as _Path
    import os, tempfile, weakref
    from PIL import Image
    import torch
    try:
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLMModel
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq as _VLMModel
            except Exception:
                _VLMModel = None
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            StoppingCriteria = object
            def StoppingCriteriaList(x): return [x]
    except Exception:
        AutoProcessor = None
        _VLMModel = None
        StoppingCriteria = object
        def StoppingCriteriaList(x): return [x]

    from PySide6 import QtCore
    from PySide6.QtCore import Qt, QSettings, QBuffer, QByteArray, QThread, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap, QGuiApplication
    from PySide6.QtWidgets import (
        QWidget, QHBoxLayout, QLabel, QComboBox, QSizePolicy, QPushButton, QProgressBar, QSpinBox, QFileDialog
    )

    # ---------- Device selection ----------
    def _fv_choose_device():
        try:
            import torch_directml as _dml
        except Exception:
            _dml = None
        if torch.cuda.is_available():
            return "cuda", f"CUDA:{torch.cuda.get_device_name(0)}", torch.float16
        if _dml is not None:
            return _dml.device(), "DirectML", torch.float32
        return "cpu", "CPU", torch.float32

    # ---------- Player helpers ----------
    def _fv_get_video(self):
        try:
            return getattr(self.window(), "video", None)
        except Exception:
            return None

    def _fv_grab_qimage_from_player(self):
        vid = _fv_get_video(self)
        if vid is None:
            return None
        # Prefer the live pixmap from the video label (updates each frame)
        try:
            pm = vid.label.pixmap() if hasattr(vid, "label") else None
            if pm is not None and not pm.isNull():
                return pm.toImage()
        except Exception:
            pass
        # Fallback to any cached frame set via _fv_player_show_image
        try:
            img = getattr(vid, "currentFrame", None)
            if img is not None and (not hasattr(img, "isNull") or not img.isNull()):
                return img
        except Exception:
            pass
        return None

    def _fv_player_show_image(self, qimg: QImage):
        try:
            vid = _fv_get_video(self)
            if vid is None or qimg is None or qimg.isNull():
                return
            vid.currentFrame = qimg
            if hasattr(vid, "label") and vid.label is not None:
                pm = QPixmap.fromImage(qimg).scaled(vid.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                vid.label.setPixmap(pm)
        except Exception:
            pass

    def _fv_save_temp_png(qimg: QImage) -> str:
        ba = QByteArray(); buf = QBuffer(ba); buf.open(QBuffer.WriteOnly)
        qimg.save(buf, "PNG"); buf.close()
        p = _Path(tempfile.gettempdir()) / "framevision_describer_current.png"
        with open(p, "wb") as f:
            f.write(bytes(ba))
        return str(p)

    # ---------- Presets ----------
    _PRESETS = [
        "General (detail-based)",
        "Literal caption (6–12 words)",
        "Photo prompt (SDXL-style tags)",
        "Cinematic still",
        "Product listing",
        "Object list",
        "Scene breakdown (bullets)",
        "ALT text (accessibility)",
        "Anime still",
        "E-commerce (white background)",
        "Architecture interior",
        "Architecture exterior",
        "3D render (Octane-style)",
        "Food styling",
    ]

    def _fv_prompt_for_preset(self, preset: str) -> str:
        detail = self.combo_detail.currentText() if hasattr(self, "combo_detail") else "Medium"
        if preset == "General (detail-based)":
            if detail == "Short":
                return "Provide a short, punchy caption for this image."
            if detail == "Medium":
                return "Describe this image clearly with key subjects, attributes, and setting."
            return "Describe this image thoroughly: subjects, actions, materials, colors, lighting, mood, background, and composition."
        if preset == "Literal caption (6–12 words)":
            return ("Write a literal 6–12 word caption for this image. Only describe what is clearly visible. No imagination or opinions.")
        if preset == "Photo prompt (SDXL-style tags)":
            neg = ""
            try: neg = (self.negative.text() or "").strip()
            except Exception: neg = ""
            return ("Return only a single comma-separated list of 25–45 concise, lowercase tags for a photo-realistic text-to-image prompt. "
                    "Include subject, setting, camera, lens, lighting, mood, and style keywords that are visible. "
                    "Do not invent objects, and do not add explanations, quotes, or line breaks."
                   ) + (f" Avoid these concepts: {neg}." if neg else "")
        if preset == "Cinematic still":
            return ("Describe the image as a cinematic still. Be literal to the scene but include lens length, aperture, depth of field, "
                    "lighting style, framing, color grading, and mood that are visually evident. No invented story.")
        if preset == "Product listing":
            return ("Write a clear product-style description of what is visible: item type, materials, color, condition, size cues, and background. "
                    "Avoid brand names unless legible in the image. Keep it factual and neutral.")
        if preset == "Object list":
            return ("List the clearly visible objects in this image as a short bullet list using hyphens. "
                    "One object per line, no commentary, no invented items.")
        if preset == "Scene breakdown (bullets)":
            return ("Break down the scene as concise bullet points with hyphens for: subject, secondary subjects, background, lighting, colors, "
                    "composition/framing, mood. Be factual and only include what is clearly visible.")
        if preset == "ALT text (accessibility)":
            return ("Write one concise, literal ALT-text sentence (max 150 characters) that accurately describes the image content for screen readers.")
        if preset == "Anime still":
            return ("Describe the image as an anime still. Focus on character pose, facial features, hair, outfit, background elements, color palette, "
                    "and lighting. Use genre terms only if visually evident. No invented story.")
        if preset == "E-commerce (white background)":
            return ("Write a clean e‑commerce product description for a white/neutral background photo. List item type, material, color, finishes, "
                    "notable features, size cues, and condition. No brand claims unless text is legible.")
        if preset == "Architecture interior":
            return ("Describe the interior scene: room type, layout, key furniture, materials, textures, color palette, lighting type, "
                    "style (e.g., modern, Scandinavian), and composition. Keep it factual and visual.")
        if preset == "Architecture exterior":
            return ("Describe the exterior architecture: building type, massing, façade materials, glazing, roof, surrounding context, time of day, "
                    "and weather. Note perspective if visible.")
        if preset == "3D render (Octane-style)":
            return ("Describe the image as a 3D render. Mention materials/shaders, reflections, depth of field, lighting (HDRI/area lights), "
                    "and camera framing. Keep to visible cues.")
        if preset == "Food styling":
            return ("Describe the food scene: dish type, ingredients, textures, garnishes, plating, props, surface, lighting (soft/hard, direction), "
                    "and mood. No imagined flavors or claims.")
        return "Describe this image clearly with key visible subjects, attributes, setting, lighting, and composition."

    def _fv_adjust_gen_for_preset(preset: str, gk: dict) -> dict:
        g = dict(gk)
        short_caps = ("Literal caption (6–12 words)", "ALT text (accessibility)", "Photo prompt (SDXL-style tags)")
        bullets = ("Object list", "Scene breakdown (bullets)")
        concise = ("E-commerce (white background)", "Anime still", "Architecture interior", "Architecture exterior", "3D render (Octane-style)", "Food styling")
        if preset in short_caps:
            g["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                g.pop(k, None)
            g["max_new_tokens"] = min(int(g.get("max_new_tokens", 48)), 56)
            g["min_length"] = max(1, min(int(g.get("min_length", 3)), 16))
            g["no_repeat_ngram_size"] = max(2, int(g.get("no_repeat_ngram_size", 3)))
            g["repetition_penalty"] = max(1.1, float(g.get("repetition_penalty", 1.2)))
        if preset in bullets:
            g["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                g.pop(k, None)
            g["max_new_tokens"] = max(140, int(g.get("max_new_tokens", 180)))
            g["min_length"] = max(40, int(g.get("min_length", 60)))
        if preset in concise and preset not in short_caps and preset not in bullets:
            g["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                g.pop(k, None)
            g["max_new_tokens"] = min(160, int(g.get("max_new_tokens", 180)))
            g["min_length"] = max(30, int(g.get("min_length", 40)))
        return g

    def _fv_build_generation_config(model, gk):
        try:
            gc = model.generation_config.clone()
        except Exception:
            gc = model.generation_config
        do_sample = bool(gk.get("do_sample", False))
        try: setattr(gc, "do_sample", do_sample)
        except Exception: pass
        for k, v in list(gk.items()):
            if hasattr(gc, k):
                try: setattr(gc, k, v)
                except Exception: pass
        if hasattr(gc, "do_sample") and getattr(gc, "do_sample") is False:
            for bad in ("temperature", "top_p", "top_k"):
                if hasattr(gc, bad):
                    try: setattr(gc, bad, None)
                    except Exception: pass
        else:
            if hasattr(gc, "temperature"):
                try:
                    if gc.temperature is None or gc.temperature <= 0:
                        gc.temperature = 0.7
                except Exception:
                    pass
        return gc

    class _FVCancelStop(StoppingCriteria):  # type: ignore[misc]
        def __init__(self, w):
            self._wref = weakref.ref(w)
        def __call__(self, input_ids, scores, **kwargs):
            w = self._wref()
            return bool(w and getattr(w, "_fv_cancelled", False))

    # ---------- Async workers ----------
    class _FVDescribeWorker(QtCore.QObject):
        finished = Signal(str)
        failed = Signal(str)
        def __init__(self, w, img_path: _Path, cpu_cap: int):
            super().__init__(); self._wref = weakref.ref(w); self._path = _Path(img_path); self._cap = max(1, int(cpu_cap or 1))
        @Slot()
        def run(self):
            w = self._wref(); 
            if w is None: self.failed.emit("Describe failed: widget disposed"); return
            try:
                torch.set_num_threads(self._cap)
            except Exception:
                pass
            try:
                text = w._run_inference(self._path)
                self.finished.emit(text)
            except Exception as e:
                self.failed.emit(f"Describe failed: {e}")

    class _FVFolderWorker(QtCore.QObject):
        progress = Signal(int, int)
        finished = Signal(str)
        failed = Signal(str)
        def __init__(self, w, files, cpu_cap: int):
            super().__init__(); self._wref = weakref.ref(w); self._files = list(files); self._cap = max(1, int(cpu_cap or 1))
        @Slot()
        def run(self):
            w = self._wref()
            if w is None: self.failed.emit("Describe folder failed: widget disposed"); return
            try:
                torch.set_num_threads(self._cap)
            except Exception:
                pass
            total = len(self._files); done = 0
            try:
                for f in self._files:
                    if getattr(w, "_fv_cancelled", False):
                        self.finished.emit(f"Cancelled at {done}/{total}."); return
                    text = w._run_inference(_Path(f))
                    try:
                        _Path(f).with_suffix(".txt").write_text(text, encoding="utf-8")
                    except Exception:
                        pass
                    done += 1; self.progress.emit(done, total)
                self.finished.emit(f"Done: {done}/{total} files.")
            except Exception as e:
                self.failed.emit(f"Describe folder failed: {e}")

    # ---------- UI: insert/move widgets & connect signals ----------
    _old_init_v8 = DescriberWidget.__init__  # type: ignore[attr-defined]
    def _fv_init_v8(self, *a, **kw):  # type: ignore[no-redef]
        _old_init_v8(self, *a, **kw)
        root = self.layout()
        if root is None: return

        # Bigger output
        try:
            if hasattr(self, "output"):
                self.output.setMinimumHeight(max(460, self.output.minimumHeight()))
                self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        # Preset row between Actions and "Generated description"
        combo = getattr(self, "combo_preset", None)
        if not isinstance(combo, QComboBox):
            combo = QComboBox(); combo.addItems(_PRESETS)
            s = QSettings("FrameVision","FrameVision")
            saved = s.value("describe_preset", _PRESETS[0], str) or _PRESETS[0]
            if saved in _PRESETS: combo.setCurrentText(saved)
            combo.currentTextChanged.connect(lambda v: QSettings("FrameVision","FrameVision").setValue("describe_preset", v))
            self.combo_preset = combo
        row = QWidget(self); hl = QHBoxLayout(row); hl.setContentsMargins(0,0,0,0)
        hl.addWidget(QLabel("Preset:")); combo.setParent(row); hl.addWidget(combo)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # find insertion index
        insert_index = None
        for i in range(root.count()):
            it = root.itemAt(i); w = it.widget() if it else None
            if isinstance(w, QLabel) and getattr(w, "text", lambda: "")().strip().lower() == "generated description":
                insert_index = i; break
        if insert_index is None:
            try:
                actw = self.btn_describe.parentWidget()
                for i in range(root.count()):
                    it = root.itemAt(i)
                    if it and it.widget() is actw: insert_index = i+1; break
            except Exception: insert_index = 1
        root.insertWidget(insert_index, row)

        # CPU threads + backend row under engine info (top)
        try:
            w = getattr(self, "lbl_engine_name", None)
            ins_after = None
            if w is not None:
                gp = w.parentWidget().parentWidget()
                for i in range(root.count()):
                    it = root.itemAt(i)
                    if it and it.widget() is gp: ins_after = i+1; break
            if ins_after is None: ins_after = 1
            row2 = QWidget(self); h2 = QHBoxLayout(row2); h2.setContentsMargins(0,0,0,0)
            h2.addWidget(QLabel("CPU threads:"))
            if not hasattr(self, "spin_cpu"):
                self.spin_cpu = QSpinBox()
                mx = max(1, os.cpu_count() or 8)
                s = QSettings("FrameVision","FrameVision")
                cap = int(s.value("describe_cpu_threads", max(1, mx//2)))
                self.spin_cpu.setRange(1, mx); self.spin_cpu.setValue(min(mx, max(1, cap)))
                self.spin_cpu.valueChanged.connect(lambda v: QSettings("FrameVision","FrameVision").setValue("describe_cpu_threads", int(v)))
            h2.addWidget(self.spin_cpu)
            # Move toggles from Advanced to top row, after CPU threads spinner
            try:
                if hasattr(self, "chk_warm"):
                    self.chk_warm.setParent(row2)
                    h2.addSpacing(12)
                    h2.addWidget(self.chk_warm)
                if hasattr(self, "chk_keep"):
                    self.chk_keep.setParent(row2)
                    h2.addSpacing(12)
                    h2.addWidget(self.chk_keep)
            except Exception:
                pass

            h2.addStretch(1)
            h2.addWidget(QLabel("Backend:"))
            if not hasattr(self, "lbl_backend"):
                self.lbl_backend = QLabel("—")
            h2.addWidget(self.lbl_backend)
            row2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            root.insertWidget(ins_after, row2)
        except Exception:
            pass

        # Compact actions & add Cancel as 4th small button
        try:
            act = self.btn_describe.parentWidget(); lay = act.layout()
            if not hasattr(self, "btn_cancel"):
                self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False); self.btn_cancel.clicked.connect(self._fv_on_cancel_clicked); lay.addWidget(self.btn_cancel)
            for b in (self.btn_describe, self.btn_desc_folder, self.btn_use_current, self.btn_cancel):
                try:
                    b.setMinimumHeight(32); f=b.font(); f.setPointSize(max(8, f.pointSize()-2)); b.setFont(f)
                except Exception: pass
        except Exception:
            pass

        # Create hidden progress bar under actions
        if not hasattr(self, "prog"):
            self.prog = QProgressBar(); self.prog.setTextVisible(False); self.prog.setFixedHeight(3); self.prog.setRange(0,1); self.prog.setValue(0); self.prog.setVisible(False)
            try:
                actw = self.btn_describe.parentWidget()
                idx=None
                for i in range(root.count()):
                    it = root.itemAt(i)
                    if it and it.widget() is actw: idx=i; break
                root.insertWidget((idx+1) if idx is not None else 1, self.prog)
            except Exception:
                pass

        # Wire buttons/signals with safe disconnect
        pairs = [
            (getattr(self,"btn_describe",None), getattr(self, "_fv_on_click_describe_async", None)),
            (getattr(self,"btn_use_current",None), getattr(self, "_fv_use_current_clicked", None)),
            (getattr(self,"btn_desc_folder",None), getattr(self, "_fv_on_click_describe_folder_async", None)),
        ]
        for btn, slot in pairs:
            if btn is None or slot is None:
                continue
            try:
                reg = getattr(btn, '_fv_connected', None)
                if reg is None:
                    reg = set(); setattr(btn, '_fv_connected', reg)
                key = (id(slot), getattr(slot, '__name__', str(slot)))
                if key not in reg:
                    btn.clicked.connect(slot)
                    reg.add(key)
            except Exception:
                pass

        # Wrap Browse to preview chosen image in player
        if hasattr(self, "_browse"):
            orig = self._browse
            def _wrap_browse():
                orig()
                p = (self.txt_image.text() or "").strip() if hasattr(self,"txt_image") else ""
                if p and _Path(p).exists():
                    try:
                        im = Image.open(p).convert("RGB")
                        qimg = QImage(im.tobytes(), im.width, im.height, im.width*3, QImage.Format_RGB888)
                        _fv_player_show_image(self, qimg)
                    except Exception: pass
            self._browse = _wrap_browse  # type: ignore[assignment]

    DescriberWidget.__init__ = _fv_init_v8  # type: ignore[attr-defined]

    # ---------- Button handlers ----------
    def _fv_on_cancel_clicked(self):
        self._fv_cancelled = True
        try:
            self.btn_cancel.setEnabled(False)
            if hasattr(self,"output") and self.output.toPlainText().strip() == "Creating the description, this could take a minute":
                self.output.setPlainText("Cancelling…")
        except Exception: pass

    DescriberWidget._fv_on_cancel_clicked = _fv_on_cancel_clicked  # type: ignore[attr-defined]

    def _fv_on_click_describe_async(self):
        p = ""
        try: p = (self.txt_image.text() or "").strip()
        except Exception: p = ""
        # Always prefer grabbing the current video frame (overrides stale path)
        try:
            qimg = _fv_grab_qimage_from_player(self)
        except Exception:
            qimg = None
        if qimg is not None and (not hasattr(qimg, "isNull") or not qimg.isNull()):
            p = _fv_save_temp_png(qimg)
            try: self.txt_image.setText(p)
            except Exception: pass
            # Clear any cached frame so next describe re-grabs fresh
            try:
                vid = _fv_get_video(self)
                if vid is not None and hasattr(vid, "currentFrame"): vid.currentFrame = None
            except Exception: pass
        if not p or not _Path(p).exists():
            try: self.output.setPlainText("Please choose a valid image path or use current frame.")
            except Exception: pass
            return
        if getattr(self, "_fv_busy", False):
            try: self.output.setPlainText("A describe job is already running…")
            except Exception: pass
            return
        self._fv_busy = True; self._fv_cancelled = False
        try: self.btn_describe.setEnabled(False); self.btn_use_current.setEnabled(False); self.btn_cancel.setEnabled(True)
        except Exception: pass
        try: QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        except Exception: pass
        try: self.output.setPlainText("Creating the description, this could take a minute")
        except Exception: pass
        try: self.prog.setVisible(True); self.prog.setRange(0,0); self.prog.setValue(0)
        except Exception: pass
        try: cap = int(self.spin_cpu.value())
        except Exception: cap = max(1, (os.cpu_count() or 8)//2)
        self._fv_thread = QThread(self)
        self._fv_worker = _FVDescribeWorker(self, _Path(p), cap)
        self._fv_worker.moveToThread(self._fv_thread)
        self._fv_thread.started.connect(self._fv_worker.run)
        self._fv_worker.finished.connect(self._fv_on_worker_finished)
        self._fv_worker.failed.connect(self._fv_on_worker_failed)
        self._fv_worker.finished.connect(self._fv_thread.quit)
        self._fv_worker.failed.connect(self._fv_thread.quit)
        self._fv_thread.finished.connect(self._fv_thread.deleteLater)
        self._fv_thread.start()

    DescriberWidget._fv_on_click_describe_async = _fv_on_click_describe_async  # type: ignore[attr-defined]

    def _fv__reset_busy_ui(self):
        try: self.btn_describe.setEnabled(True); self.btn_use_current.setEnabled(True); self.btn_cancel.setEnabled(False)
        except Exception: pass
        try: self.prog.setVisible(False); self.prog.setRange(0,1); self.prog.setValue(0)
        except Exception: pass
        try: QGuiApplication.restoreOverrideCursor()
        except Exception: pass
        self._fv_busy = False; self._fv_cancelled = False

    DescriberWidget._fv__reset_busy_ui = _fv__reset_busy_ui  # type: ignore[attr-defined]

    def _fv_on_worker_finished(self, text: str):
        try: self.output.setPlainText(text)
        except Exception: pass
        _fv__reset_busy_ui(self)

    def _fv_on_worker_failed(self, msg: str):
        try: self.output.setPlainText(msg)
        except Exception: pass
        _fv__reset_busy_ui(self)

    DescriberWidget._fv_on_worker_finished = _fv_on_worker_finished  # type: ignore[attr-defined]
    DescriberWidget._fv_on_worker_failed = _fv_on_worker_failed  # type: ignore[attr-defined]

    # Folder async
    def _fv_on_click_describe_folder_async(self):
        try: folder = QFileDialog.getExistingDirectory(self, "Choose folder to describe")
        except Exception: folder = ""
        if not folder: return
        p = _Path(folder); 
        if not p.exists(): return
        exts = (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")
        files = [str(q) for q in sorted(p.rglob("*")) if q.suffix.lower() in exts]
        if not files:
            try: self.output.setPlainText("No images found in folder.")
            except Exception: pass
            return
        self._fv_busy = True; self._fv_cancelled = False
        try:
            self.btn_describe.setEnabled(False); self.btn_use_current.setEnabled(False); self.btn_cancel.setEnabled(True); self.btn_desc_folder.setEnabled(False)
        except Exception: pass
        try: self.prog.setVisible(True); self.prog.setRange(0, len(files)); self.prog.setValue(0)
        except Exception: pass
        try: self.output.setPlainText(f"Describing folder… 0/{len(files)}")
        except Exception: pass
        try: cap = int(self.spin_cpu.value())
        except Exception: cap = max(1, (os.cpu_count() or 8)//2)
        self._fv_thread = QThread(self)
        self._fv_folder = _FVFolderWorker(self, files, cap)
        self._fv_folder.moveToThread(self._fv_thread)
        self._fv_thread.started.connect(self._fv_folder.run)
        self._fv_folder.progress.connect(self._fv_on_folder_progress)
        self._fv_folder.finished.connect(self._fv_on_folder_finished)
        self._fv_folder.failed.connect(self._fv_on_folder_failed)
        self._fv_folder.finished.connect(self._fv_thread.quit)
        self._fv_folder.failed.connect(self._fv_thread.quit)
        self._fv_thread.finished.connect(self._fv_thread.deleteLater)
        self._fv_thread.start()

    DescriberWidget._fv_on_click_describe_folder_async = _fv_on_click_describe_folder_async  # type: ignore[attr-defined]

    def _fv_on_folder_progress(self, done:int, total:int):
        try: self.prog.setVisible(True); self.prog.setRange(0,total); self.prog.setValue(done); self.output.setPlainText(f"Describing folder… {done}/{total}")
        except Exception: pass

    def _fv_on_folder_finished(self, msg:str):
        try: self.output.setPlainText(msg)
        except Exception: pass
        try: self.btn_desc_folder.setEnabled(True)
        except Exception: pass
        _fv__reset_busy_ui(self)

    def _fv_on_folder_failed(self, msg:str):
        try: self.output.setPlainText(msg)
        except Exception: pass
        try: self.btn_desc_folder.setEnabled(True)
        except Exception: pass
        _fv__reset_busy_ui(self)

    DescriberWidget._fv_on_folder_progress = _fv_on_folder_progress  # type: ignore[attr-defined]
    DescriberWidget._fv_on_folder_finished = _fv_on_folder_finished  # type: ignore[attr-defined]
    DescriberWidget._fv_on_folder_failed = _fv_on_folder_failed  # type: ignore[attr-defined]

    # ---------- Inference override ----------
    def _run_inference(self, img_path: _Path) -> str:  # type: ignore[no-redef]
        cfg = ENGINE_CATALOG[self.engine_key]
        folder = models_root() / cfg["folder"]
        if not (folder.exists() and any(folder.iterdir())):
            raise RuntimeError("Model not prepared")

        image = Image.open(str(img_path)).convert("RGB")
        eng_type = cfg["type"]
        gen_kwargs = self._gen_params()

        preset_name = getattr(self, "combo_preset", None).currentText() if getattr(self, "combo_preset", None) else "General (detail-based)"
        gen_kwargs = _fv_adjust_gen_for_preset(preset_name, gen_kwargs)

        if eng_type == "hf_blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(str(folder), local_files_only=True)
            model = BlipForConditionalGeneration.from_pretrained(str(folder), local_files_only=True)
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)
            text = processor.decode(out_ids[0], skip_special_tokens=True).strip()

        elif eng_type == "hf_qwen2vl":
            device, backend, dtype = _fv_choose_device()
            try:
                if hasattr(self, "lbl_backend"): self.lbl_backend.setText(str(backend))
            except Exception: pass
            print(f"[fv] describer backend: {backend}")
            if AutoProcessor is None or _VLMModel is None:
                raise RuntimeError("Transformers not available")
            processor = AutoProcessor.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True, use_fast=True)
            model = _VLMModel.from_pretrained(
                str(folder), trust_remote_code=True, local_files_only=True, torch_dtype=dtype
            ).to(device)

            base_prompt = _fv_prompt_for_preset(self, preset_name)
            if preset_name in ("Object list", "Scene breakdown (bullets)"):
                base_prompt += " Use hyphen bullets starting with '- ' on each line."

            messages = [{"role": "user", "content": [{"type":"image","image": image}, {"type":"text","text": base_prompt}]}]
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[-1]
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}

            gc = _fv_build_generation_config(model, gen_kwargs)
            stop_list = StoppingCriteriaList([_FVCancelStop(self)]) if isinstance(StoppingCriteria, type) else None
            with torch.inference_mode():
                out = model.generate(**inputs, generation_config=gc, return_dict_in_generate=True, stopping_criteria=stop_list)
            out_ids = out.sequences if hasattr(out, "sequences") else out
            new_ids = out_ids[:, prompt_len:]
            text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        else:
            raise RuntimeError(f"Unknown engine type: {eng_type}")

        try:
            if preset_name == "General (detail-based)" and self.combo_detail.currentText() == "Long" and len(text.split()) < 40:
                text = text + " (expand: mention subjects, materials, colors, lighting, background, composition)"
        except Exception:
            pass
        return text

    DescriberWidget._run_inference = _run_inference  # type: ignore[attr-defined]

# === FrameVision: dynamic model size on disk (engine Size label) ===
try:
    DescriberWidget  # type: ignore[name-defined]
except Exception:
    pass
else:
    from PySide6 import QtCore
    from PySide6.QtCore import QObject, QThread, Signal, Slot

    class _FVFolderSizeWorker(QtCore.QObject):
        finished = Signal(object)  # bytes
        failed = Signal(str)
        def __init__(self, path):
            super().__init__(); self._path = path
        @Slot()
        def run(self):
            try:
                p = self._path
                total = 0
                if p.exists():
                    for q in p.rglob("*"):
                        if q.is_file():
                            try:
                                total += q.stat().st_size
                            except Exception:
                                pass
                self.finished.emit(total)
            except Exception as e:
                self.failed.emit(str(e))

    def _fv_human_bytes(n: int) -> str:
        try:
            x = float(max(0, n))
            for unit in ("B","KB","MB","GB","TB"):
                if x < 1024 or unit == "TB":
                    return (f"~{x:.1f} {unit}" if unit != "B" else f"{int(x)} B") + " on disk"
                x = x/1024.0
        except Exception:
            pass
        return f"{n} B on disk"

    _old_refresh_size = getattr(DescriberWidget, "_refresh_engine_labels", None)

    def _fv_refresh_with_size(self):
        # call original to set base texts
        if callable(_old_refresh_size):
            _old_refresh_size(self)
        # then compute real on-disk size async
        try:
            cfg = ENGINE_CATALOG[self.engine_key]
            folder = models_root() / cfg["folder"]
        except Exception:
            return
        # mark while scanning
        try:
            base_txt = getattr(self, "lbl_size").text().strip()
            if base_txt:
                self.lbl_size.setText(base_txt + "  (scanning…)")
        except Exception:
            pass

        try:
            if getattr(self, "_fv_size_thread", None):
                self._fv_size_thread.quit(); self._fv_size_thread.wait(50)
        except Exception:
            pass

        th = QThread(self); wk = _FVFolderSizeWorker(folder)
        wk.moveToThread(th)
        def _done(bytes_total: int):
            try:
                self.lbl_size.setText(_fv_human_bytes(bytes_total))
            except Exception:
                pass
        wk.finished.connect(_done)
        wk.failed.connect(lambda msg: None)
        th.started.connect(wk.run)
        th.finished.connect(th.deleteLater)
        self._fv_size_thread = th
        self._fv_size_worker = wk
        th.start()

    DescriberWidget._refresh_engine_labels = _fv_refresh_with_size  # type: ignore[attr-defined]

# === FrameVision: Size on disk (models folder only, strict) ===
try:
    DescriberWidget  # type: ignore[name-defined]
except Exception:
    pass
else:
    from PySide6 import QtCore
    from PySide6.QtCore import QThread, Signal, Slot

    class _FVStrictSizeWorker(QtCore.QObject):
        finished = Signal(object)  # bytes
        failed = Signal(str)
        def __init__(self, path):
            super().__init__(); self._path = path

        @Slot()
        def run(self):
            try:
                p = self._path
                total = 0
                files = 0
                if p.exists():
                    for q in p.rglob("*"):
                        if q.is_file():
                            try:
                                sz = q.stat().st_size
                                total += int(sz); files += 1
                            except Exception:
                                pass
                # debug print in console for verification
                try:
                    # Also compute full app size (root + subfolders) and print in MB
                    root = find_project_root()
                    app_files = 0
                    app_total = 0
                    if root.exists():
                        for qq in root.rglob("*"):
                            if qq.is_file():
                                try:
                                    app_total += int(qq.stat().st_size)
                                    app_files += 1
                                except Exception:
                                    pass
                    mb = app_total / (1024.0 * 1024.0)
                    print(f"[fv] size scan: {root} -> {app_files} files, {mb:.1f} MB")
                except Exception:
                    pass
                self.finished.emit(total)
            except Exception as e:
                self.failed.emit(str(e))

    def _fv_bytes_human_strict(n: int) -> str:
        x = float(max(0, n))
        for u in ("B","KB","MB","GB","TB"):
            if x < 1024 or u == "TB":
                return (f"~{x:.1f} {u}" if u != "B" else f"{int(x)} B") + " on disk"
            x /= 1024.0

    _old_refresh_size_strict = getattr(DescriberWidget, "_refresh_engine_labels", None)

    def _fv_refresh_with_size_strict(self):
        if callable(_old_refresh_size_strict):
            _old_refresh_size_strict(self)
        try:
            cfg = ENGINE_CATALOG[self.engine_key]
            folder = models_root() / cfg["folder"]
        except Exception:
            return
        try:
            base_txt = getattr(self, "lbl_size").text().strip()
            if base_txt:
                self.lbl_size.setText(base_txt + "  (scanning models folder…)")
        except Exception:
            pass
        # stop previous
        try:
            if getattr(self, "_fv_size_thread", None):
                self._fv_size_thread.quit(); self._fv_size_thread.wait(50)
        except Exception:
            pass
        th = QThread(self); wk = _FVStrictSizeWorker(folder)
        wk.moveToThread(th)
        def _done(total:int):
            try:
                self.lbl_size.setText(_fv_bytes_human_strict(total))
            except Exception:
                pass
        wk.finished.connect(_done)
        wk.failed.connect(lambda msg: None)
        th.started.connect(wk.run)
        th.finished.connect(th.deleteLater)
        self._fv_size_thread = th
        self._fv_size_worker = wk
        th.start()

    DescriberWidget._refresh_engine_labels = _fv_refresh_with_size_strict  # type: ignore[attr-defined]