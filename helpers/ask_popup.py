from __future__ import annotations
import os, time, threading, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QTextCursor, QImage, QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel as _QLabel, QApplication
)

# === Model path (offline) ===
MODELS_FOLDER = Path(".").resolve() / "models" / "describe" / "default" / "qwen2-vl-2b-instruct"

# === Singletons to avoid VRAM growth ===
_GLOBAL = {"device": None, "dtype": None, "processor": None, "model": None}

# === Local knowledge (optional) ===
JOKES_PATH = Path('.') / 'assets' / 'dad_jokes.txt'
INFO_DIR = Path('.') / 'presets' / 'info'
_LOCAL = {'jokes': None, 'j_order': [], 'j_idx': 0, 'info_text': None}

def _choose_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        try:
            import torch_directml as dml  # type: ignore
            return dml.device(), torch.float32
        except Exception:
            return "cpu", torch.float32
    except Exception:
        return "cpu", None

def _qimage_to_pil(img: QImage):
    try:
        from PIL import Image
    except Exception:
        return None
    if img is None or img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.bits(); ptr.setsize(img.sizeInBytes())
    pil = Image.frombuffer("RGBA", (w, h), bytes(ptr), "raw", "RGBA", 0, 1).convert("RGB")
    return pil


def _ensure_model_loaded():
    if _GLOBAL["model"] is not None and _GLOBAL["processor"] is not None:
        return
    device, dtype = _choose_device()
    _GLOBAL["device"], _GLOBAL["dtype"] = device, dtype
    from transformers import AutoProcessor, AutoModelForCausalLM
    if not (MODELS_FOLDER.exists() and any(MODELS_FOLDER.iterdir())):
        raise RuntimeError(f"Model not found at {MODELS_FOLDER}")
    _GLOBAL["processor"] = AutoProcessor.from_pretrained(
        str(MODELS_FOLDER), trust_remote_code=True, local_files_only=True, use_fast=True
    )
    _GLOBAL["model"] = AutoModelForCausalLM.from_pretrained(
        str(MODELS_FOLDER),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)
    # Console debug
    try:
        _cls = type(_GLOBAL['model']).__name__
        has_vision = hasattr(_GLOBAL['model'], 'vision_tower') or hasattr(getattr(_GLOBAL['model'], 'config', object()), 'vision_config')
        print(f"[DEBUG] model_class={_cls} has_vision={has_vision}")
    except Exception as e:
        print(f"[DEBUG] model_class_check_error: {e}")

@dataclass
class GenConfig:
    temperature: float = 0.7
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05

def _build_messages(system_prompt: str, history: List[Dict[str, Any]], user_text: str, pil_image) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    sys = (system_prompt or "").strip()
    if sys:
        msgs.append({"role":"system","content":[{"type":"text","text": sys}]})
    msgs.extend(history)
    content: List[Dict[str, Any]] = []
    if pil_image is not None:
        # Use image placeholder only; actual PIL is passed to processor(images=[...])
        content.append({"type":"image"})
    content.append({"type":"text","text": user_text})
    msgs.append({"role":"user","content": content})
    return msgs

# ---------------- Local knowledge helpers ----------------
def _load_jokes():
    if _LOCAL['jokes'] is not None:
        return
    jokes = []
    try:
        txt = JOKES_PATH.read_text(encoding='utf-8', errors='ignore')
        for line in txt.splitlines():
            s = line.strip()
            if s:
                jokes.append(s)
    except Exception:
        jokes = []
    _LOCAL['jokes'] = jokes
    _LOCAL['j_order'] = list(range(len(jokes)))
    random.shuffle(_LOCAL['j_order'])
    _LOCAL['j_idx'] = 0

def _next_joke() -> str:
    _load_jokes()
    if not _LOCAL['jokes']:
        return "I don't have local jokes available."
    i = _LOCAL['j_order'][_LOCAL['j_idx'] % len(_LOCAL['j_order'])]
    _LOCAL['j_idx'] += 1
    return _LOCAL['jokes'][i]

def _strip_html(s: str) -> str:
    try:
        s = re.sub(r'<script.*?>.*?</script>', '', s, flags=re.DOTALL|re.IGNORECASE)
        s = re.sub(r'<style.*?>.*?</style>', '', s, flags=re.DOTALL|re.IGNORECASE)
        s = re.sub(r'<[^>]+>', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    except Exception:
        return s

def _load_info_text():
    if _LOCAL['info_text'] is not None:
        return
    parts = []
    try:
        if INFO_DIR.exists():
            for pth in sorted(INFO_DIR.glob('*')):
                if pth.suffix.lower() == '.json':
                    try:
                        parts.append(pth.read_text(encoding='utf-8', errors='ignore'))
                    except Exception:
                        pass
                elif pth.suffix.lower() in ('.html', '.htm'):
                    try:
                        parts.append(_strip_html(pth.read_text(encoding='utf-8', errors='ignore')))
                    except Exception:
                        pass
    except Exception:
        pass
    text = '\n'.join(parts).strip()
    if len(text) > 2000:
        text = text[:2000] + '…'
    _LOCAL['info_text'] = text or ""

def _maybe_local_response(user_text: str) -> Optional[str]:
    t = (user_text or "").strip().lower()
    if t.startswith('/reloadlocal'):
        _LOCAL['jokes'] = None; _LOCAL['info_text'] = None; _load_jokes(); _load_info_text()
        return "Local data reloaded."
    if t.startswith('/joke') or ('joke' in t and len(t) < 80):
        return _next_joke()
    if t.startswith('/info') or 'what is framevision' in t or 'about this app' in t or 'help with framevision' in t:
        _load_info_text()
        return _LOCAL['info_text'] or "No local app info found."
    return None

# ---------------- Worker ----------------
class ChatWorker(QThread):
    chunk = Signal(str)
    done = Signal()
    error = Signal(str)

    def __init__(self, messages: List[Dict[str, Any]], pil_image, gen: GenConfig, stream: bool = True, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.pil_image = pil_image
        self.gen = gen
        self.stream = bool(stream)

    def run(self):
        try:
            _ensure_model_loaded()
            try:
                m = _GLOBAL.get('model')
                _cls = type(m).__name__ if m is not None else 'None'
                has_vision = (hasattr(m, 'vision_tower') or hasattr(getattr(m, 'config', object()), 'vision_config')) if m is not None else False
                self.chunk.emit(f"[DEBUG] model_class={_cls} has_vision={has_vision}\n")
            except Exception:
                pass
            from transformers import TextIteratorStreamer
            import torch

            processor = _GLOBAL["processor"]; model = _GLOBAL["model"]
            device = _GLOBAL["device"]

            if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
                self.error.emit("Processor has no tokenizer; cannot decode.")
                return
            tokenizer = processor.tokenizer

            chat_text = processor.apply_chat_template(self.messages, add_generation_prompt=True)

            # DEBUG: confirm image count passed to processor
            try:
                img_count = 1 if (self.pil_image is not None) else 0
                self.chunk.emit(f"[DEBUG] processor images={img_count}\\n")
            except Exception:
                pass

            if self.pil_image is not None:
                inputs = processor(text=[chat_text], images=[self.pil_image], return_tensors="pt")
            else:
                inputs = processor(text=[chat_text], return_tensors="pt")

            try:
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}
            except Exception:
                pass

            emitted = False

            if self.stream:
                streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})
                eos_id = getattr(tokenizer, "eos_token_id", None)
                pad_id = getattr(tokenizer, "pad_token_id", eos_id)

                gen_kwargs = dict(
                    **inputs,
                    do_sample=True if (self.gen.temperature and self.gen.temperature>0.01) else False,
                    temperature=float(max(0.01, self.gen.temperature)),
                    top_p=0.9,
                    repetition_penalty=float(max(1.0, self.gen.repetition_penalty)),
                    max_new_tokens=int(max(1, self.gen.max_new_tokens)),
                    streamer=streamer,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    return_dict_in_generate=True
                )

                def _generate_stream():
                    with torch.inference_mode():
                        model.generate(**gen_kwargs)

                th = threading.Thread(target=_generate_stream, daemon=True)
                th.start()
                for piece in streamer:
                    if piece:
                        emitted = True
                        self.chunk.emit(piece)

            if (not self.stream) or (not emitted):
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        do_sample=True if (self.gen.temperature and self.gen.temperature>0.01) else False,
                        temperature=float(max(0.01, self.gen.temperature)),
                        top_p=0.9,
                        repetition_penalty=float(max(1.0, self.gen.repetition_penalty)),
                        max_new_tokens=int(max(1, self.gen.max_new_tokens)),
                        return_dict_in_generate=True
                    )
                sequences = out.sequences if hasattr(out, "sequences") else out[0]
                text = processor.batch_decode(sequences if isinstance(sequences, list) else [sequences], skip_special_tokens=True)[0]
                if text:
                    self.chunk.emit(text)

            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

DEFAULT_SYSTEM = (
    "You are a friendly assistant inside a video/photo app. Be concise. "
    "When a frame is attached, ground your answer in it."
)

class AskPopup(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ask")
        self.setMinimumSize(680, 600)

        self._history: List[Dict[str, Any]] = []

        root = QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(8)

        hdr = QHBoxLayout(); hdr.addWidget(QLabel("Ask anything.")); hdr.addStretch(1)
        hdr.addWidget(QLabel("Temp")); self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.0,2.0); self.spin_temp.setSingleStep(0.05); self.spin_temp.setValue(0.70); hdr.addWidget(self.spin_temp)
        hdr.addWidget(QLabel("Max")); self.spin_max = QSpinBox(); self.spin_max.setRange(16,4096); self.spin_max.setValue(1024); hdr.addWidget(self.spin_max)
        root.addLayout(hdr)

        self.system = QPlainTextEdit(self); self.system.setPlainText(DEFAULT_SYSTEM); self.system.setMaximumHeight(64); root.addWidget(self.system)

        self.transcript = QPlainTextEdit(self); self.transcript.setReadOnly(True); self.transcript.setPlaceholderText("Assistant responses will appear here..."); root.addWidget(self.transcript, 1)

        self.input = QPlainTextEdit(self); self.input.setPlaceholderText("Type and press Send… (Shift+Enter for newline)"); self.input.setFixedHeight(100); self.input.installEventFilter(self); root.addWidget(self.input)

        # Simple status/debug line
        self.status = QLabel("")
        root.addWidget(self.status)

        row = QHBoxLayout()
        self.chk_attach = QCheckBox("Attach current frame"); row.addWidget(self.chk_attach)
        self.chk_stream = QCheckBox("Stream"); self.chk_stream.setChecked(True); row.addWidget(self.chk_stream)
        row.addStretch(1)
        self.btn_reset = QPushButton("Reset"); self.btn_close = QPushButton("Close"); self.btn_send = QPushButton("Send")
        row.addWidget(self.btn_reset); row.addWidget(self.btn_close); row.addWidget(self.btn_send); root.addLayout(row)

        self.btn_close.clicked.connect(self.close)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_send.clicked.connect(self._on_send)

    def _append(self, who: str, text: str = ""):
        if who: self.transcript.appendPlainText(f"{who}: {text}")
        else: self.transcript.appendPlainText(text)
        self.transcript.moveCursor(QTextCursor.End)

    def _append_inline(self, text: str):
        c = self.transcript.textCursor(); c.movePosition(QTextCursor.End); c.insertText(text); self.transcript.setTextCursor(c); self.transcript.ensureCursorVisible()

    def _debug(self, msg: str):
        self.transcript.appendPlainText(f"[DEBUG] {msg}")
        self.transcript.moveCursor(QTextCursor.End)
        self.status.setText(msg)

    def _find_main_with_video(self):
        try:
            p = self.parent()
            while p is not None:
                if hasattr(p, "video"):
                    return p
                try:
                    p = p.parent()
                except Exception:
                    break
        except Exception:
            pass
        try:
            for w in QApplication.topLevelWidgets():
                if hasattr(w, "video"):
                    return w
        except Exception:
            pass
        return None

    def _grab_qimage(self) -> Optional[QImage]:
        try:
            main = self._find_main_with_video()
            if main is None:
                self._debug("No main window with .video found."); return None

            video = getattr(main, "video", None)

            # 1) currentFrame
            img = getattr(video, "currentFrame", None)
            if isinstance(img, QImage) and not img.isNull():
                self._debug(f"Found currentFrame: {img.width()}x{img.height()} from main.video")
                return img

            # 2) label.pixmap()
            try:
                label = getattr(video, "label", None)
                if label is not None and hasattr(label, "pixmap"):
                    pm = label.pixmap()
                    if pm is not None and not pm.isNull():
                        qimg = pm.toImage()
                        self._debug(f"Found label pixmap: {qimg.width()}x{qimg.height()} from main.video.label")
                        return qimg
            except Exception:
                pass

            # 3) any QLabel pixmap
            try:
                labels = main.findChildren(_QLabel)
                for lb in labels[::-1]:
                    if hasattr(lb, "pixmap"):
                        pm = lb.pixmap()
                        if pm is not None and not pm.isNull() and pm.width() > 32 and pm.height() > 32:
                            qimg = pm.toImage()
                            self._debug(f"Found fallback QLabel pixmap: {qimg.width()}x{qimg.height()}")
                            return qimg
            except Exception:
                pass

            self._debug("No frame source found.")
        except Exception as e:
            self._debug(f"Frame grab error: {e}")
        return None

    def _on_reset(self):
        self.transcript.clear(); self.input.clear(); self._history.clear(); self.status.setText("")

    def _on_send(self):
        q = (self.input.toPlainText() or "").strip()
        if not q: return
        self.input.clear()

        # local knowledge hooks
        local = _maybe_local_response(q)
        if local is not None:
            self._append("You", q)
            self._append("Assistant", local)
            return

        self._append("You", q + ("  [frame attached]" if self.chk_attach.isChecked() else ""))
        self._append("Assistant", "")

        qimg = self._grab_qimage() if self.chk_attach.isChecked() else None
        pil = _qimage_to_pil(qimg) if qimg is not None else None
        if self.chk_attach.isChecked():
            if qimg is None:
                self._debug("Attach requested but no QImage available.")
            else:
                self._debug(f"Attach ON: QImage {qimg.width()}x{qimg.height()} -> PIL {'OK' if pil is not None else 'FAIL'}")

        # If asking about the app, include local info as extra system context
        extra_sys = self.system.toPlainText()
        if 'framevision' in q.lower() or 'this app' in q.lower():
            _load_info_text()
            if _LOCAL['info_text']:
                extra_sys = (extra_sys + '\\n\\n' + 'APP_INFO:\\n' + _LOCAL['info_text']).strip()

        gen = GenConfig(temperature=float(self.spin_temp.value()), max_new_tokens=int(self.spin_max.value()), repetition_penalty=1.05)
        msgs = _build_messages(extra_sys, self._history, q, pil)

        self.btn_send.setEnabled(False)
        self.worker = ChatWorker(msgs, pil, gen, stream=self.chk_stream.isChecked(), parent=self)
        self.worker.chunk.connect(self._append_inline)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self):
        last_line = self.transcript.toPlainText().splitlines()[-1] if self.transcript.toPlainText().splitlines() else ""
        asst = last_line[len("Assistant: "):] if last_line.startswith("Assistant: ") else last_line
        self._history.append({"role":"assistant","content":[{"type":"text","text": asst}] })
        self._append("", ""); self.btn_send.setEnabled(True); self.status.setText("")

    def _on_error(self, msg: str):
        self._append_inline(f"[error: {msg}]"); self._append("", ""); self.btn_send.setEnabled(True); self.status.setText(msg)

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            ke: QKeyEvent = ev  # type: ignore
            if ke.key() in (Qt.Key_Return, Qt.Key_Enter):
                if ke.modifiers() & Qt.ShiftModifier: return False
                self._on_send(); return True
        return super().eventFilter(obj, ev)
