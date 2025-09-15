from __future__ import annotations
import os, time, threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QTextCursor, QImage, QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox
)

# === Model path (offline) ===
MODELS_FOLDER = Path(".").resolve() / "models" / "describe" / "default" / "qwen2-vl-2b-instruct"

# === Singletons to avoid VRAM growth ===
_GLOBAL = {"device": None, "dtype": None, "processor": None, "model": None}

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
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as VLM
        except Exception:
            VLM = None
    if VLM is None:
        raise RuntimeError("Transformers VLM class not available.")
    if not (MODELS_FOLDER.exists() and any(MODELS_FOLDER.iterdir())):
        raise RuntimeError(f"Model not found at {MODELS_FOLDER}")
    _GLOBAL["processor"] = AutoProcessor.from_pretrained(
        str(MODELS_FOLDER), trust_remote_code=True, local_files_only=True, use_fast=True
    )
    _GLOBAL["model"] = VLM.from_pretrained(
        str(MODELS_FOLDER), trust_remote_code=True, local_files_only=True, torch_dtype=dtype
    ).to(device)

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
        content.append({"type":"image","image": pil_image})
    content.append({"type":"text","text": user_text})
    msgs.append({"role":"user","content": content})
    return msgs

class ChatWorker(QThread):
    chunk = Signal(str)
    done = Signal()
    error = Signal(str)

    def __init__(self, messages: List[Dict[str, Any]], pil_image, gen: GenConfig, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.pil_image = pil_image
        self.gen = gen

    def run(self):
        try:
            _ensure_model_loaded()
            from transformers import TextIteratorStreamer
            import torch

            processor = _GLOBAL["processor"]; model = _GLOBAL["model"]
            device = _GLOBAL["device"]

            if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
                self.error.emit("Processor has no tokenizer; cannot stream.")
                return
            tokenizer = processor.tokenizer

            chat_text = processor.apply_chat_template(self.messages, add_generation_prompt=True)

            if self.pil_image is not None:
                inputs = processor(text=[chat_text], images=[self.pil_image], return_tensors="pt")
            else:
                inputs = processor(text=[chat_text], return_tensors="pt")

            try:
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}
            except Exception:
                pass

            # Streamer must use the tokenizer (not the model)
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

            def _generate():
                with torch.inference_mode():
                    model.generate(**gen_kwargs)

            th = threading.Thread(target=_generate, daemon=True)
            th.start()
            for piece in streamer:
                if piece:
                    self.chunk.emit(piece)
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
        self.setMinimumSize(640, 520)

        self._history: List[Dict[str, Any]] = []

        root = QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(8)

        hdr = QHBoxLayout(); hdr.addWidget(QLabel("Ask anything.")); hdr.addStretch(1)
        hdr.addWidget(QLabel("Temp")); self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.0,2.0); self.spin_temp.setSingleStep(0.05); self.spin_temp.setValue(0.70); hdr.addWidget(self.spin_temp)
        hdr.addWidget(QLabel("Max")); self.spin_max = QSpinBox(); self.spin_max.setRange(16,4096); self.spin_max.setValue(1024); hdr.addWidget(self.spin_max)
        root.addLayout(hdr)

        self.system = QPlainTextEdit(self); self.system.setPlainText(DEFAULT_SYSTEM); self.system.setMaximumHeight(64); root.addWidget(self.system)

        self.transcript = QPlainTextEdit(self); self.transcript.setReadOnly(True); self.transcript.setPlaceholderText("Assistant responses will appear here..."); root.addWidget(self.transcript, 1)

        self.input = QPlainTextEdit(self); self.input.setPlaceholderText("Type and press Send… (Shift+Enter for newline)"); self.input.setFixedHeight(100); self.input.installEventFilter(self); root.addWidget(self.input)

        row = QHBoxLayout()
        self.chk_attach = QCheckBox("Attach current frame"); row.addWidget(self.chk_attach); row.addStretch(1)
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

    def _grab_qimage(self) -> Optional[QImage]:
        try:
            vid = getattr(self.window(), "video", None)
            if vid is None: return None
            try:
                pm = vid.label.pixmap() if hasattr(vid, "label") else None
                if pm is not None and not pm.isNull(): return pm.toImage()
            except Exception: pass
            img = getattr(vid, "currentFrame", None)
            if isinstance(img, QImage) and not img.isNull(): return img
        except Exception: pass
        return None

    def _on_reset(self):
        self.transcript.clear(); self.input.clear(); self._history.clear()

    def _on_send(self):
        q = (self.input.toPlainText() or "").strip()
        if not q: return
        self.input.clear()
        self._append("You", q + ("  [frame attached]" if self.chk_attach.isChecked() else ""))
        self._append("Assistant", "")

        pil = _qimage_to_pil(self._grab_qimage()) if self.chk_attach.isChecked() else None
        gen = GenConfig(temperature=float(self.spin_temp.value()), max_new_tokens=int(self.spin_max.value()), repetition_penalty=1.05)

        msgs = _build_messages(self.system.toPlainText(), self._history, q, pil)

        self.btn_send.setEnabled(False)
        self.worker = ChatWorker(msgs, pil, gen, parent=self)
        self.worker.chunk.connect(self._append_inline)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self):
        last_line = self.transcript.toPlainText().splitlines()[-1] if self.transcript.toPlainText().splitlines() else ""
        asst = last_line[len("Assistant: "):] if last_line.startswith("Assistant: ") else last_line
        self._history.append({"role":"assistant","content":[{"type":"text","text": asst}] })
        self._append("", ""); self.btn_send.setEnabled(True)

    def _on_error(self, msg: str):
        self._append_inline(f"[error: {msg}]"); self._append("", ""); self.btn_send.setEnabled(True)

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            ke: QKeyEvent = ev  # type: ignore
            if ke.key() in (Qt.Key_Return, Qt.Key_Enter):
                if ke.modifiers() & Qt.ShiftModifier: return False
                self._on_send(); return True
        return super().eventFilter(obj, ev)
