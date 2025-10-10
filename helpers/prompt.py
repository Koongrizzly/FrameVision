from __future__ import annotations

import os, json, traceback, inspect, datetime, threading
from pathlib import Path
from typing import Any, Dict, Callable, List, Tuple, Optional

from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFormLayout, QDoubleSpinBox, QSpinBox, QLineEdit,
    QComboBox, QFileDialog, QMessageBox
)

# ---- Shared root path from app ----
try:
    from helpers.framevision_app import ROOT
except Exception:
    ROOT = Path(".").resolve()

SET_PATH = ROOT / "presets" / "setsave" / "prompt.json"
SET_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_STYLE = ""
DEFAULT_TEMPLATE_BASE = (
    "Expand the seed into one vivid, {length_words} single-sentence prompt for text-to-{target}. "
    "Include subject, environment, time of day, lighting, camera/lens, composition, action, textures, mood, and color palette. "
    "Return only the final prompt, no lists or markup."
)
DEFAULT_NEG = ""
OLD_DEFAULT_NEG = "text, watermark, vending machines, rubber ducks, disco balls, flamingos"
NEW_DEFAULT_NEG = "bad eyes, deformed body parts, bad teeth"

LENGTH_PRESETS = {
    "Short (40–60 words)": ( "40–60 words", 160 ),
    "Medium (80–120 words)": ( "80–120 words", 280 ),
    "Long (140–200 words)": ( "140–200 words", 420 ),
}

# ---------- Lightweight caching for model + processor ----------
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()

def _lines_to_px(widget, lines:int)->int:
    fm = widget.fontMetrics()
    return max(48, int(lines * fm.lineSpacing()) + 8)

def _load_settings()->Dict[str, Any]:
    try:
        if SET_PATH.exists():
            data = json.loads(SET_PATH.read_text(encoding="utf-8"))
            try:
                neg = (data.get("negatives") or "").strip()
                # Map the legacy noisy default list to the new cleaner set
                # Accept either exact match or the same items in any order.
                legacy = OLD_DEFAULT_NEG.lower()
                cur = neg.lower()
                def _norm_set(s):
                    return {x.strip() for x in s.split(",") if x.strip()}
                if cur == legacy or _norm_set(cur) == _norm_set(legacy):
                    data["negatives"] = NEW_DEFAULT_NEG
            except Exception:
                pass
                        # Normalize legacy forced styles: drop "" defaults
            try:
                st = (data.get("style") or "").strip()
                if st and "pixar" in st.lower():
                    data["style"] = ""
            except Exception:
                pass
            return data
    except Exception:
        pass
    return {
        "style": DEFAULT_STYLE,
        "length_choice": "Medium (80–120 words)",
        "target": "image",
        "model_key": "",
        "negatives": DEFAULT_NEG,
        "temperature": 0.85,
        "max_new_tokens": LENGTH_PRESETS["Medium (80–120 words)"][1],
        "seed_text": "",
    }

def _save_settings(data:Dict[str, Any])->None:
    try:
        SET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ---------- Model discovery & generation ----------
def _list_qwen_vl_models() -> List[Tuple[str,str]]:
    """Return list of (key, label) for engines of type 'hf_qwen2vl' from helpers.describer."""
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
    except Exception:
        return []
    out = []
    try:
        cat = getattr(D, "ENGINE_CATALOG", {})
        for k, meta in cat.items():
            if str(meta.get("type","")).lower() in ("hf_qwen2vl","hf_qwen2_vl","qwen2vl","qwen_vl"):
                label = meta.get("label", k)
                out.append((k, label))
    except Exception:
        pass
    return out

def _models_root_path()->Path|None:
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
    except Exception:
        return None
    try:
        return Path(getattr(D, "models_root")())
    except Exception:
        return None

def _folder_for_model_key(model_key:str)->Path|None:
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
    except Exception:
        return None
    try:
        meta = getattr(D, "ENGINE_CATALOG")[model_key]
        root = _models_root_path()
        if root is None: return None
        return root / meta.get("folder","")
    except Exception:
        return None

def _choose_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        try:
            import torch_directml as _dml  # type: ignore
            return _dml.device(), torch.float32
        except Exception:
            return "cpu", torch.float32
    except Exception:
        return "cpu", None

def _get_qwen_text_model(model_path: Path):
    """
    Load (or reuse cached) processor+model for text-only use.
    Returns (processor, model, device, dtype).
    """
    key = str(model_path.resolve())
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
    # Load lazily outside the lock to avoid long hold
    device, dtype = _choose_device()
    try:
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLMModel
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq as _VLMModel
            except Exception:
                _VLMModel = None  # type: ignore
        if _VLMModel is None:
            raise RuntimeError("Transformers model class not available")
    except Exception as e:
        raise RuntimeError(f"Transformers unavailable: {e}")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True
    )
    # Keep memory usage modest; let HF handle placement
    model = _VLMModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)
    try:
        model.eval()
    except Exception:
        pass

    with _MODEL_LOCK:
        _MODEL_CACHE[key] = (processor, model, device, dtype)
    return processor, model, device, dtype

def _generate_with_qwen_text(
    model_path: Path,
    system_prompt:str,
    user_prompt:str,
    temperature:float,
    max_new_tokens:int,
    cancel_check: Optional[Callable[[], bool]] = None
)->str:
    """
    Text-only generation for Qwen2-VL like models using Transformers, at the given model folder.
    Runs in a worker thread; supports cooperative cancel via stopping criteria.
    """
    if model_path is None or not (model_path.exists() and any(model_path.iterdir())):
        raise RuntimeError("Model folder not found")
    try:
        processor, model, device, dtype = _get_qwen_text_model(model_path)
    except Exception as e:
        raise

    messages = []
    if system_prompt:
        messages.append({"role":"system","content":[{"type":"text","text": system_prompt}]})
    messages.append({"role":"user","content":[{"type":"text","text": user_prompt}]})

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat_text], return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]
    try:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}
    except Exception:
        pass

    # Optional cooperative cancel
    stopping_criteria = None
    try:
        if cancel_check is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class _CancelStopper(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    try:
                        return bool(cancel_check())
                    except Exception:
                        return False
            stopping_criteria = StoppingCriteriaList([_CancelStopper()])
    except Exception:
        stopping_criteria = None

    import torch
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(max(0.01, temperature or 0.7)),
            top_p=0.9,
            max_new_tokens=int(max_new_tokens or 200),
            repetition_penalty=1.1,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria
        )
    seq = out.sequences if hasattr(out, "sequences") else out
    new_ids = seq[:, prompt_len:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return text

# ---------- Worker (non-blocking) ----------
class _PromptGenWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_folder: Path, system_prompt: str, user_prompt: str, temperature: float, max_new_tokens: int):
        super().__init__()
        self._model_folder = model_folder
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = _generate_with_qwen_text(
                self._model_folder,
                self._system_prompt,
                self._user_prompt,
                self._temperature,
                self._max_new_tokens,
                cancel_check=lambda: self._cancel
            )
            if self._cancel:
                self.failed.emit("Cancelled")
                return
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

# ---------- UI ----------
class PromptToolPane(QWidget):
    """
    Prompt Generator that expands short seeds into long prompts.
    - Saves settings to presets/setsave/prompt.json
    - Lets you choose Length, Target (image/video), and Model (Qwen2-VL engines discovered)
    - Can save outputs as .json (and .txt)
    - Now runs generation in a background thread with Cancel support and a small model cache.
    """
    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        self._saving_timer = QTimer(self); self._saving_timer.setInterval(600); self._saving_timer.setSingleShot(True)
        self._saving_timer.timeout.connect(self._save_now)

        self._thread: Optional[QThread] = None
        self._worker: Optional[_PromptGenWorker] = None

        self.state = _load_settings()
        # Ensure full cleanup when the widget is destroyed (prevents hidden windows / timers leaks)
        try:
            self.destroyed.connect(self._teardown)
        except Exception:
            pass

        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(8)

        # ---- Controls ----
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignLeft); form.setFormAlignment(Qt.AlignLeft|Qt.AlignTop)

        # Model & target
        self.combo_model = QComboBox()
        models = _list_qwen_vl_models()
        if not models:
            self.combo_model.addItem("Qwen2-VL (auto)", "")
        else:
            for k,label in models:
                self.combo_model.addItem(label, k)
        # restore selection
        want_key = self.state.get("model_key","")
        if want_key:
            idx = max(0, self.combo_model.findData(want_key))
            self.combo_model.setCurrentIndex(idx)

        self.combo_target = QComboBox(); self.combo_target.addItems(["image","video"])
        tgt = (self.state.get("target") or "image").lower()
        if tgt in ("image","video"):
            self.combo_target.setCurrentText(tgt)

        # Length preset
        self.combo_len = QComboBox()
        for label in LENGTH_PRESETS.keys():
            self.combo_len.addItem(label)
        self.combo_len.setCurrentText(self.state.get("length_choice","Medium (80–120 words)"))

        # Style/negatives and generation params
        self.style = QLineEdit(self.state.get("style", DEFAULT_STYLE)); self.style.setPlaceholderText("Style tags (Cinematic, Photoreal, Anime)…")
        self.neg = QLineEdit(self.state.get("negatives", DEFAULT_NEG)); self.neg.setPlaceholderText("Negatives (optional; leave empty if not needed)")
        self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setSingleStep(0.05); self.temp.setValue(float(self.state.get("temperature", 0.85)))
        self.max_new = QSpinBox(); self.max_new.setRange(64, 4096); self.max_new.setValue(int(self.state.get("max_new_tokens", 280)))

        form.addRow("Model", self.combo_model)
        form.addRow("Target", self.combo_target)
        form.addRow("Length", self.combo_len)
        form.addRow("Style", self.style)
        form.addRow("Negatives", self.neg)
        row_params = QHBoxLayout(); row_params.addWidget(QLabel("Temperature")); row_params.addWidget(self.temp); row_params.addSpacing(12); row_params.addWidget(QLabel("Max tokens")); row_params.addWidget(self.max_new); row_params.addStretch(1)
        form.addRow(row_params)

        root.addLayout(form)

        # ---- Seed / Buttons / Result ----
        self.seed = QTextEdit(self.state.get("seed_text",""))
        self.seed.setPlaceholderText("Enter seed words, e.g. 'a cat in a tree'")
        self.seed.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.seed.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.seed.setFixedHeight(_lines_to_px(self.seed, 2))
        root.addWidget(QLabel("Seed words"))
        root.addWidget(self.seed)

        btns = QHBoxLayout()
        self.btn_gen = QPushButton("Generate")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_copy = QPushButton("Copy result")
        self.btn_clear = QPushButton("Clear")
        self.btn_save_txt = QPushButton("Save .txt")
        self.btn_save_json = QPushButton("Save .json")
        for b in (self.btn_gen, self.btn_cancel, self.btn_copy, self.btn_clear, self.btn_save_txt, self.btn_save_json):
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)

        self.out = QTextEdit("")
        self.out.setReadOnly(True)
        self.out.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.out.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.out.setFixedHeight(_lines_to_px(self.out, 15))
        root.addWidget(QLabel("Result"))
        root.addWidget(self.out)

        # ---- hooks ----
        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_copy.clicked.connect(lambda: self._copy_to_clipboard())
        self.btn_clear.clicked.connect(lambda: self.out.clear())
        self.btn_save_txt.clicked.connect(self._save_txt)
        self.btn_save_json.clicked.connect(self._save_json)

        for w in (self.style, self.neg, self.combo_model, self.combo_len, self.combo_target):
            try:
                w.currentIndexChanged.connect(self._schedule_save)  # for combos
            except Exception:
                w.textChanged.connect(self._schedule_save)  # for lineedits
        self.temp.valueChanged.connect(self._schedule_save)
        self.max_new.valueChanged.connect(self._schedule_save)
        self.seed.textChanged.connect(self._schedule_save)

        # auto-adjust tokens if user hasn't changed them manually
        self._user_touched_tokens = False
        self.max_new.valueChanged.connect(lambda *_: setattr(self, "_user_touched_tokens", True))
        self.combo_len.currentIndexChanged.connect(self._maybe_adjust_tokens)

    def _maybe_adjust_tokens(self):
        if not self._user_touched_tokens:
            _, tokens = LENGTH_PRESETS[self.combo_len.currentText()]
            self.max_new.setValue(tokens)

    # ---- persistence ----
    def _schedule_save(self):
        self._saving_timer.start()

    def _save_now(self):
        data = {
            "style": self.style.text().strip(),
            "length_choice": self.combo_len.currentText(),
            "target": self.combo_target.currentText(),
            "model_key": self.combo_model.currentData() or "",
            "negatives": self.neg.text().strip(),
            "temperature": float(self.temp.value()),
            "max_new_tokens": int(self.max_new.value()),
            "seed_text": self.seed.toPlainText(),
        }
        _save_settings(data)

    # ---- generation ----
    def _compose_prompts(self, seed:str)->tuple[str,str]:
        target = self.combo_target.currentText().strip().lower()
        if target not in ("image","video"):
            target = "image"
        length_label = self.combo_len.currentText()
        length_words, _ = LENGTH_PRESETS.get(length_label, LENGTH_PRESETS["Medium (80–120 words)"])
        template = DEFAULT_TEMPLATE_BASE.format(length_words=length_words, target=("image" if target=="image" else "video"))
        style = self.style.text().strip() or DEFAULT_STYLE
        negatives = self.neg.text().strip() or DEFAULT_NEG
        # Video adds motion cues to nudge the model
        if target == "video":
            style = f"{style}, cinematic, motion-aware, dynamic framing"
        sys = (
            "You are a visual prompt engineer. "
            "Expand short seeds into a single richly detailed prompt. "
            "Follow the instruction template and style hints exactly."
        )
        import re as _re
        _neg_clean = negatives if _re.search(r"[A-Za-z0-9]", negatives) else ""
        user = (
            (f"{template} ")
            + (f"Use the style: {style}. ")
            + (f"If and only if negatives are provided, append at the end: 'negative: {_neg_clean}'. " if _neg_clean else "")
            + (f"Seed: {seed}")
        )
        return sys, user

    def _set_busy(self, busy: bool):
        self.btn_gen.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        self.btn_copy.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        self.btn_save_txt.setEnabled(not busy)
        self.btn_save_json.setEnabled(not busy)
        self.combo_model.setEnabled(not busy)
        self.combo_target.setEnabled(not busy)
        self.combo_len.setEnabled(not busy)
        self.style.setEnabled(not busy)
        self.neg.setEnabled(not busy)
        self.temp.setEnabled(not busy)
        self.max_new.setEnabled(not busy)
        self.seed.setEnabled(not busy)
        self.btn_gen.setText("Generating…" if busy else "Generate")

    def _on_generate(self):
        if self._thread is not None:
            # already running
            return
        seed = (self.seed.toPlainText() or "").strip()
        if not seed:
            QMessageBox.warning(self, "Prompt", "Please enter a seed (e.g., 'a cat').")
            return
        sys, usr = self._compose_prompts(seed)
        model_key = self.combo_model.currentData() or ""
        folder = _folder_for_model_key(model_key) if model_key else None
        if folder is None:
            # try auto fallback to first available
            models = _list_qwen_vl_models()
            if models:
                folder = _folder_for_model_key(models[0][0])
        if folder is None:
            # No model available — produce local expand quickly
            style = self.style.text().strip() or DEFAULT_STYLE
            import re as _re
            _neg_raw = self.neg.text().strip()
            neg = _neg_raw if _re.search(r"[A-Za-z0-9]", _neg_raw) else ""
            tail = (f" negative: {neg}." if neg else "")
            out = f"{seed.strip().capitalize()}" + (f" in {style};" if style else ";") + " rich lighting, camera details, textures, mood, palette." + (tail or "")
            self.out.setPlainText(str(out).strip())
            return

        # Spin up background worker
        self._thread = QThread(self)
        self._worker = _PromptGenWorker(folder, sys, usr, float(self.temp.value()), int(self.max_new.value()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        # Ensure cleanup
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.failed.connect(self._cleanup_worker)

        self._set_busy(True)
        self._thread.start()

    def _on_worker_finished(self, text: str):
        self.out.setPlainText(str(text).strip())
        self._schedule_save()

    def _on_worker_failed(self, err: str):
        try:
            QMessageBox.critical(self, "Prompt generator", f"{err}\n\nFalling back to local expand.")
        except Exception:
            pass
        seed = (self.seed.toPlainText() or "").strip()
        style = self.style.text().strip() or DEFAULT_STYLE
        neg = (self.neg.text().strip() or DEFAULT_NEG)
        tail = (f" negative: {neg}." if neg else "")
        out = f"{seed.strip().capitalize()}" + (f" in {style};" if style else ";") + " rich lighting, camera details, textures, mood, palette." + (tail or "")
        self.out.setPlainText(str(out).strip())
        self._schedule_save()

    def _cleanup_worker(self):
        # Reset busy UI and teardown thread/worker
        self._set_busy(False)
        try:
            if self._worker is not None:
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
                self._thread.deleteLater()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.btn_gen.setText("Cancelling…")

    def _copy_to_clipboard(self):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(self.out.toPlainText())
            QMessageBox.information(self, "Copied", "Result copied to clipboard.")
        except Exception:
            pass

    def _save_txt(self):
        text = self.out.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Save .txt", "No result to save yet.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save prompt as .txt", str(ROOT / "output_prompt.txt"), "Text files (*.txt)")
        if not fn: return
        try:
            Path(fn).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Save .txt", f"Failed: {e}")

    def _save_json(self):
        text = self.out.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Save .json", "No result to save yet.")
            return
        meta = {
            "seed": (self.seed.toPlainText() or "").strip(),
            "prompt": text,
            "style": self.style.text().strip(),
            "length": self.combo_len.currentText(),
            "target": self.combo_target.currentText(),
            "model_key": self.combo_model.currentData() or "",
            "negatives": self.neg.text().strip(),
            "temperature": float(self.temp.value()),
            "max_new_tokens": int(self.max_new.value()),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        fn, _ = QFileDialog.getSaveFileName(self, "Save prompt as .json", str(ROOT / "prompt.json"), "JSON files (*.json)")
        if not fn: return
        try:
            Path(fn).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Save .json", f"Failed: {e}")

    def _teardown(self, *args, **kwargs):
        # Stop debounce timer
        try:
            self._saving_timer.stop()
        except Exception:
            pass
        # Cancel running worker (if any)
        try:
            if self._worker is not None:
                self._worker.cancel()
        except Exception:
            pass
        # Ensure background thread is fully terminated
        try:
            if self._thread is not None:
                try:
                    self._thread.quit()
                except Exception:
                    pass
                self._thread.wait(2000)
                try:
                    self._thread.terminate()  # last resort on Windows to avoid leaked hidden msg windows
                except Exception:
                    pass
        except Exception:
            pass
        # Clear refs
        self._thread = None
        self._worker = None


def install_prompt_tool(owner, section_widget):
    """Install the Prompt tool into a CollapsibleSection, matching other Tools UI conventions."""
    try:
        widget = PromptToolPane(getattr(owner, "main", None), section_widget)
    except Exception:
        widget = PromptToolPane(None, section_widget)
    wrap = QWidget(); lay = QVBoxLayout(wrap); lay.setContentsMargins(0,0,0,0); lay.addWidget(widget)
    try:
        section_widget.setContentLayout(lay)
    except Exception:
        try:
            section_widget.content.setLayout(lay)
        except Exception:
            pass
    return widget
