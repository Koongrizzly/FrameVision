from __future__ import annotations
import os, time, threading, random, re, ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent, QEventLoop, QRect, QPoint, QTimer, QEventLoop, QRect, QTimer, QEventLoop, QRect, QTimer, QProcess
from PySide6.QtGui import QTextCursor, QImage, QKeyEvent, QGuiApplication, QPainter, QPainter, QGuiApplication, QPainter, QClipboard
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel as _QLabel, QApplication, QInputDialog, QRubberBand, QInputDialog, QRubberBand
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
    """
    Robust QImage -> PIL conversion that works with PySide6/PyQt5 memoryview/sip.voidptr,
    handles stride (bytesPerLine), and falls back to a copy without requiring NumPy.
    """
    try:
        from PIL import Image
    except Exception:
        try:
            _GLOBAL["pillow_missing"] = True
        except Exception:
            pass
        return None
    if img is None or img.isNull():
        return None

    qimg = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()

    ptr = qimg.bits()  # memoryview (PySide6) or sip.voidptr (PyQt5)
    try:
        # PyQt5 path
        ptr.setsize(qimg.sizeInBytes())
    except Exception:
        # PySide6 memoryview: size is already known
        pass

    buf = ptr.tobytes()

    # Try zero-copy path that understands stride
    try:
        pil = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", bpl, 1)
        return pil.convert("RGB")
    except Exception:
        # Fallback: build a contiguous RGBA buffer row by row (no stride)
        row = w * 4
        contiguous = bytearray(row * h)
        for y in range(h):
            start = y * bpl
            contiguous[y*row:(y+1)*row] = buf[start:start+row]
        pil = Image.frombuffer("RGBA", (w, h), bytes(contiguous), "raw", "RGBA", 0, 1)
        return pil.convert("RGB")


def _ensure_model_loaded():
    if _GLOBAL["model"] is not None and _GLOBAL["processor"] is not None:
        return
    device, dtype = _choose_device()
    _GLOBAL["device"], _GLOBAL["dtype"] = device, dtype
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    if not (MODELS_FOLDER.exists() and any(MODELS_FOLDER.iterdir())):
        raise RuntimeError(f"Model not found at {MODELS_FOLDER}")
    _GLOBAL["processor"] = AutoProcessor.from_pretrained(
        str(MODELS_FOLDER), trust_remote_code=True, local_files_only=True, use_fast=True
    )
    _GLOBAL["model"] = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODELS_FOLDER),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)
    # Debug: show model class and vision capability
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
            # Show model class + vision capability in the chat transcript
            try:
                m = _GLOBAL.get('model')
                _cls = type(m).__name__ if m is not None else 'None'
                has_vision = (hasattr(m, 'vision_tower') or hasattr(getattr(m, 'config', object()), 'vision_config')) if m is not None else False
                self.chunk.emit(f"[DEBUG] model_class={_cls} has_vision={has_vision}\n")
            except Exception as e:
                self.chunk.emit(f"[DEBUG] model_class_check_error: {e}\n")
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
                text = processor.batch_decode(sequences[0] if isinstance(sequences[0], list) else [sequences[0]], skip_special_tokens=True)[0]
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



# === In-app region overlay (child of main window) ===
class _LocalRegionOverlay(QWidget):
    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = None

        # Fill entire parent area
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def paintEvent(self, ev):
        # Dim only the app content, not the entire desktop
        painter = QPainter(self)
        painter.setOpacity(0.15)
        painter.fillRect(self.rect(), Qt.black)
        painter.end()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self._rubber.hide()
            self.cancelled.emit()
            self.close()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._origin = ev.pos()
            self._rubber.setGeometry(QRect(self._origin, self._origin))
            self._rubber.show()

    def mouseMoveEvent(self, ev):
        if self._rubber.isVisible() and self._origin is not None:
            rect = QRect(self._origin, ev.pos()).normalized()
            self._rubber.setGeometry(rect)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._rubber.isVisible():
            rect = self._rubber.geometry().normalized()
            self._rubber.hide()
            self.selected.emit(rect)
            self.close()

    @staticmethod
    def pick_in_parent(parent: QWidget) -> QRect:
        # Modal mini-loop like in your menu approach
        loop = QEventLoop()
        result = {"rect": QRect()}
        ov = _LocalRegionOverlay(parent)

        ov.selected.connect(lambda r: (result.update(rect=r), loop.quit()))
        ov.cancelled.connect(lambda: loop.quit())

        # Ensure it's on top of the parent content
        QTimer.singleShot(0, ov.raise_)
        loop.exec()
        return result["rect"]


def _wait_ms(ms: int):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

# --- Windows BitBlt screen capture (avoids black screens) ---
def _win_grab_rect_to_pil(x: int, y: int, w: int, h: int):
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        if os.name != "nt":
            return None
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        hdc_screen = user32.GetDC(0)
        if not hdc_screen:
            return None
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        gdi32.SelectObject(hdc_mem, hbm)

        SRCCOPY = 0x00CC0020
        CAPTUREBLT = 0x40000000  # capture layered windows too
        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, SRCCOPY | CAPTUREBLT)

        # Prepare BITMAPINFO for 32-bit BGRA
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]

        bi = BITMAPINFO()
        ctypes.memset(ctypes.byref(bi), 0, ctypes.sizeof(bi))
        bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.bmiHeader.biWidth = w
        bi.bmiHeader.biHeight = -h  # top-down
        bi.bmiHeader.biPlanes = 1
        bi.bmiHeader.biBitCount = 32
        bi.bmiHeader.biCompression = 0  # BI_RGB

        buflen = w * h * 4
        buffer = (ctypes.c_ubyte * buflen)()
        got = gdi32.GetDIBits(hdc_mem, hbm, 0, h, ctypes.byref(buffer), ctypes.byref(bi), 0)
        # Cleanup
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        if got == 0:
            return None

        # Build PIL image from BGRA buffer
        img = Image.frombuffer("RGBA", (w, h), bytes(buffer), "raw", "BGRA", 0, 1).convert("RGB")
        return img
    except Exception:
        return None

# === Windows global snip helpers (use native Snipping Tool / Screen Snipping) ===
def _win_launch_screenclip() -> bool:
    """Try to start the native region snipping UI. Returns True if started."""
    try:
        if os.name != "nt":
            return False
        # Prefer the URI handler via explorer (Windows 10/11)
        try:
            ok = QProcess.startDetached("explorer.exe", ["ms-screenclip:"])
            if ok:
                return True
        except Exception:
            pass
        # Fallbacks
        try:
            snip = os.path.join(os.environ.get("WINDIR","C:\\Windows"), "System32", "ScreenSnipping.exe")
            if os.path.exists(snip):
                return QProcess.startDetached(snip, [])
        except Exception:
            pass
        try:
            snip2 = os.path.join(os.environ.get("WINDIR","C:\\Windows"), "System32", "SnippingTool.exe")
            if os.path.exists(snip2):
                return QProcess.startDetached(snip2, ["/clip"])
        except Exception:
            pass
    except Exception:
        return False
    return False

def _wait_clipboard_qimage(timeout_ms: int = 15000):
    """Wait until an image appears on the clipboard or timeout. Returns QImage or None."""
    cb = QGuiApplication.clipboard()  # type: QClipboard
    try:
        cb.clear()
    except Exception:
        pass

    elapsed = 0
    step = 50
    while elapsed < timeout_ms:
        QGuiApplication.processEvents()
        img = cb.image()
        if img is not None and not img.isNull() and img.width() > 0 and img.height() > 0:
            return img
        QTimer.singleShot(step, lambda: None)
        loop = QEventLoop(); QTimer.singleShot(step, loop.quit); loop.exec()
        elapsed += step
    return None

def _win_global_snip_to_qimage(max_wait_ms: int = 15000):
    """Launch native snip, wait for user region selection, and return clipboard image as QImage."""
    if os.name != "nt":
        return None
    started = _win_launch_screenclip()
    if not started:
        return None
    return _wait_clipboard_qimage(timeout_ms=max_wait_ms)
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
        self.btn_screenshot = QPushButton("Screenshot")
        row.addWidget(self.btn_screenshot)
        self.btn_reset = QPushButton("Reset"); self.btn_close = QPushButton("Close"); self.btn_send = QPushButton("Send")
        row.addWidget(self.btn_reset); row.addWidget(self.btn_close); row.addWidget(self.btn_send); root.addLayout(row)

        self.btn_close.clicked.connect(self.close)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_send.clicked.connect(self._on_send)
        self.btn_screenshot.clicked.connect(self._on_screenshot)

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

    def _on_screenshot(self):
        # Ask for delay in seconds
        sec, ok = QInputDialog.getInt(self, "Screenshot", "How long before I take the screenshot (seconds)?", 3, 0, 300, 1)
        if not ok:
            return
        self.status.setText(f"Waiting {sec}s…")
        def _do_capture():
            # Delay
            if sec > 0:
                _wait_ms(int(sec*1000))

            # On Windows: Use native global snip so user can select ANYWHERE on screen
            if os.name == "nt":
                self.hide(); QGuiApplication.processEvents()
                qimg = _win_global_snip_to_qimage(max_wait_ms=20000)
                self.show(); QGuiApplication.processEvents()
                if qimg is not None and not qimg.isNull():
                    self._send_image_question(qimg)
                    return
                # If it failed, fall back to in-app overlay
                self.status.setText("Global snip unavailable or cancelled. Falling back to app-window capture…")

            # Fallback: in-app overlay (select inside this app window)
            main = self.window()
            if main is None or not isinstance(main, QWidget):
                self.status.setText("No main window to capture."); return

            rect = _LocalRegionOverlay.pick_in_parent(main)
            if rect.isNull() or rect.width() == 0 or rect.height() == 0:
                self.status.setText("Screenshot cancelled."); return

            pm = main.grab(rect)
            if pm.isNull():
                self.status.setText("Grab failed."); return
            self._send_image_question(pm.toImage())
        QTimer.singleShot(0, _do_capture)

    def _send_image_question(self, img):
        # Convert and send as: 'what do you see'
        from PIL import Image
        pil = None
        try:
            if isinstance(img, Image.Image):
                pil = img
            else:
                pil = _qimage_to_pil(img)
        except Exception:
            pil = _qimage_to_pil(img)
        if pil is None:
            self._debug("Failed to convert screenshot to PIL."); return

        # Heuristic: avoid sending all-black/empty captures
        try:
            extrema = pil.convert("L").getextrema()
            if extrema and extrema[0] == extrema[1] and extrema[0] in (0, 1):
                self._debug("Screenshot appears black; not sending."); return
        except Exception:
            pass

        q = "what do you see"

        self._append("You", q + "  [screenshot attached]")
        self._append("Assistant", "")

        extra_sys = self.system.toPlainText()
        gen = GenConfig(temperature=float(self.spin_temp.value()), max_new_tokens=int(self.spin_max.value()), repetition_penalty=1.05)
        msgs = _build_messages(extra_sys, self._history, q, pil)

        self.btn_send.setEnabled(False)
        self.worker = ChatWorker(msgs, pil, gen, stream=self.chk_stream.isChecked(), parent=self)
        self.worker.chunk.connect(self._append_inline)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.status.setText("Sending screenshot…")

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
