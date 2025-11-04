from __future__ import annotations
import os, time, threading, random, re, ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent, QEventLoop, QRect, QPoint, QTimer, QEventLoop, QRect, QTimer, QEventLoop, QRect, QTimer, QProcess
from PySide6.QtGui import QTextCursor, QImage, QKeyEvent, QGuiApplication, QPainter, QPainter, QGuiApplication, QPainter, QClipboard
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel as _QLabel, QApplication, QInputDialog, QRubberBand, QInputDialog, QRubberBand, QComboBox, QMenu, QFileDialog
)
from transformers import AutoProcessor
from transformers import TextIteratorStreamer

# === Model path (offline) ===

# === Framie: helpers ===
BASE_DIR = Path(__file__).resolve().parent.parent
SETSAVE_DIR = BASE_DIR / "presets" / "setsave"
SETSAVE_DIR.mkdir(parents=True, exist_ok=True)
PREFS_PATH = SETSAVE_DIR / "prefs.json"
PINS_PATH = SETSAVE_DIR / "pins.json"

def _prefs_load():
    try:
        import json
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"greeting":"Professional","mode":"Coach","auto_frame":True,"backend":"Transformers (VL)","device":"auto","preset":"q5","gguf":"","llama_bin":""}

def _prefs_save(p):
    try:
        import json
        PREFS_PATH.write_text(json.dumps(p, indent=2), encoding="utf-8")
    except Exception:
        pass

def _pins_load():
    try:
        import json
        pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
        return pins if isinstance(pins, list) else []
    except Exception:
        return []

def _pins_save(pins):
    try:
        import json
        PINS_PATH.write_text(json.dumps(pins, indent=2), encoding="utf-8")
    except Exception:
        pass

def _qimage_sha1(img: QImage) -> str:
    try:
        from PySide6.QtCore import QByteArray, QBuffer, QIODevice
        ba = QByteArray()
        buff = QBuffer(ba); buff.open(QIODevice.WriteOnly)
        img.save(buff, "PNG")
        data = bytes(ba)
        import hashlib
        return hashlib.sha1(data).hexdigest()
    except Exception:
        return ""

def _downscale_qimage(img: QImage, max_dim: int = 1024) -> QImage:
    try:
        w, h = img.width(), img.height()
        if max(w,h) <= max_dim: return img
        if w >= h:
            new_w = max_dim; new_h = int(h * (max_dim / float(w)))
        else:
            new_h = max_dim; new_w = int(w * (max_dim / float(h)))
        return img.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        return img

def _gguf_ram_hint(path: str, ctx: int = 4096) -> str:
    try:
        sz = Path(path).stat().st_size
        kv = ctx * 2 * 2 * 32 * 4096 / (1024*1024)  # rough MB
        return f"{sz/1_073_741_824:.1f} GB model + KV≈{kv:.0f} MB"
    except Exception:
        return "unknown"

MODELS_FOLDER = Path(".").resolve() / "models" / "describe" / "default" / "qwen3vl2b"

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

    # Ensure local model exists
    if not (MODELS_FOLDER.exists() and any(MODELS_FOLDER.iterdir())):
        raise RuntimeError(f"Model not found at {MODELS_FOLDER}")

    # Load processor (tokenizer + processors)
    _GLOBAL["processor"] = AutoProcessor.from_pretrained(
        str(MODELS_FOLDER), trust_remote_code=True, local_files_only=True, use_fast=True
    )

    # Try multiple possible model classes for Qwen VL families (3, 2.5/2) with graceful fallbacks
    model = None
    last_err = None
    try:
        from transformers import Qwen3VLForConditionalGeneration  # Preferred for Qwen3 VL
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODELS_FOLDER),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype
        )
    except Exception as e1:
        last_err = e1
        try:
            from transformers import Qwen2VLForConditionalGeneration  # Qwen2/2.5 VL path
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(MODELS_FOLDER),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype
            )
        except Exception as e2:
            last_err = e2
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    str(MODELS_FOLDER),
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=dtype
                )
            except Exception as e3:
                last_err = e3
                try:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODELS_FOLDER),
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=dtype
                    )
                except Exception as e4:
                    last_err = e4

    if model is None:
        raise RuntimeError(f"Failed to load model from {MODELS_FOLDER}: {last_err}")

    _GLOBAL["model"] = model.to(device)

    # Debug: show model class and vision capability flags, if any
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



def _build_messages(extra_system: str, history: List[Dict[str, Any]], user_text: str, pil_image):
    """
    Build messages for a vision-capable chat model without leaking system text.
    - System: DEFAULT_SYSTEM + optional extra_system (modes/app info, without duplicates)
    - History: pass through as-is
    - User: user_text (+ image placeholder only if present)
    """
    base = DEFAULT_SYSTEM.strip()
    extra = (extra_system or "").strip()
    # Remove any copy of DEFAULT_SYSTEM from extra to avoid duplication
    if extra.replace("\r", "").replace("\n", " ").strip() == base.replace("\r", "").replace("\n", " ").strip():
        extra = ""
    else:
        extra = extra.replace(base, "").strip()

    sys_txt = base if not extra else (base + "\n\n" + extra).strip()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type":"text","text": sys_txt}]}
    ]

    for m in history:
        if m.get("role") in ("user","assistant"):
            clean_parts = []
            for c in m.get("content", []):
                if c.get("type") == "text":
                    clean_parts.append({"type":"text","text": c.get("text","")})
            if clean_parts:
                messages.append({"role": m.get("role"), "content": clean_parts})

    content = []
    if pil_image is not None:
        content.append({"type":"image"})
    content.append({"type":"text","text": user_text})
    messages.append({"role": "user", "content": content})
    return messages


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

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return text
        lines = []
        for line in str(text).splitlines():
            if line.startswith("[DEBUG]"):
                continue
            if line.strip().lower() in {"system", "user", "assistant"}:
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = cleaned.replace("", "")
        return cleaned

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ask")
        self.setMinimumSize(680, 600)

        self._history: List[Dict[str, Any]] = []

        root = QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(8)

        hdr = QHBoxLayout(); hdr.addWidget(QLabel("Ask anything.")); hdr.addStretch(1)
        hdr.addWidget(QLabel("Temp")); self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.0,2.0); self.spin_temp.setSingleStep(0.05); self.spin_temp.setValue(0.70); hdr.addWidget(self.spin_temp)
        self.spin_temp.setToolTip("Sampling temperature: higher = more random, lower = more deterministic.")
        hdr.addWidget(QLabel("Max")); self.spin_max = QSpinBox(); self.spin_max.setRange(16,4096); self.spin_max.setValue(1024); hdr.addWidget(self.spin_max)
        self.spin_max.setToolTip("Maximum number of tokens to generate in the reply.")
        root.addLayout(hdr)

        # Framie toolbar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Greeting"))
        self.combo_greet = QComboBox(); self.combo_greet.addItems(["Professional","Casual","Playful"])
        self.combo_greet.setToolTip("Tone/style of assistant replies.")
        bar.addWidget(self.combo_greet)
        bar.addWidget(QLabel("Mode"))
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Coach","Power User","ELI5","Bug hunt"])
        self.combo_mode.setToolTip("Operating mode: Coach, Power User, ELI5, or Bug hunt.")
        bar.addWidget(self.combo_mode)
        self.chk_auto_frame = QCheckBox("Auto ‘this frame’ image (≤1024px)"); self.chk_auto_frame.setChecked(True)
        self.chk_auto_frame.setToolTip("Automatically attach a downscaled snapshot of the current frame (\u22641024px) to your message.")
        bar.addWidget(self.chk_auto_frame)
        self.btn_pin = QPushButton("📌 Pin last"); self.btn_pin.clicked.connect(self._pin_last)
        self.btn_pin.setToolTip("Pin the last assistant reply for quick access later.")
        bar.addWidget(self.btn_pin)
        self.btn_pins = QPushButton("📂 Pinned"); self.btn_pins.clicked.connect(self._show_pins)
        self.btn_pins.setToolTip("Open your pinned messages.")
        bar.addWidget(self.btn_pins)
        self.btn_tools = QPushButton("Tools")
        self.btn_tools.setToolTip("Open quick tools and utilities.")
        _menu = QMenu(self.btn_tools)
        _menu.addAction("Analyze Clip…", self._tool_analyze_clip)
        _menu.addAction("Dry-run Export…", self._tool_dry_run)
        _menu.addSeparator()
        _menu.addAction("Apply last ffmpeg…", self._tool_apply_ffmpeg_from_last)
        _menu.addSeparator()
        _menu.addAction("Select llama binary…", self._select_llama_bin)
        self.btn_tools.setMenu(_menu)
        bar.addWidget(self.btn_tools)
        self.btn_tools.setVisible(False)
        bar.addStretch(1)
        root.addLayout(bar)

        # Models
        mm = QHBoxLayout()
        mm.addWidget(QLabel("Backend"))
        self.combo_backend = QComboBox(); self.combo_backend.addItems(["Transformers (VL)","llama.cpp (GGUF)"])
        self.combo_backend.setToolTip("Choose the inference backend (e.g., Transformers or local llama.cpp).")
        mm.addWidget(self.combo_backend)
        mm.addWidget(QLabel("Device")); self.combo_device = QComboBox(); self.combo_device.addItems(["auto","cpu","cuda"])
        self.combo_device.setToolTip("Select the compute device (auto/CPU/CUDA).")
        mm.addWidget(self.combo_device)
        mm.addWidget(QLabel("Preset")); self.combo_preset = QComboBox(); self.combo_preset.addItems(["Tiny (Q4)","Balanced (Q5)","Quality (Q8)"])
        self.combo_preset.setToolTip("Choose a quality/speed preset for responses.")
        mm.addWidget(self.combo_preset)
        mm.addWidget(QLabel("GGUF"))
        self.gguf_path = QPlainTextEdit(self); self.gguf_path.setMaximumHeight(28)
        self.gguf_path.setToolTip("Path to your GGUF model file for llama.cpp.")
        mm.addWidget(self.gguf_path)
        self.btn_browse_gguf = QPushButton("…"); self.btn_browse_gguf.clicked.connect(self._browse_gguf)
        self.btn_browse_gguf.setToolTip("Browse for a GGUF model file.")
        mm.addWidget(self.btn_browse_gguf)
        root.addLayout(mm)

        # Hide backend/device/preset/GGUF row (user preference)
        try:
            for _i in range(mm.count()):
                _item = mm.itemAt(_i)
                _w = _item.widget()
                if _w is not None:
                    _w.setVisible(False)
        except Exception:
            pass

        # HUD
        hud = QHBoxLayout()
        self.hud = QLabel("ctx: –, out: –, tok/s: –, t: –"); hud.addWidget(self.hud); hud.addStretch(1)
        root.addLayout(hud)


        self.system = QPlainTextEdit(self); self.system.setPlainText(DEFAULT_SYSTEM); self.system.setMaximumHeight(64); root.addWidget(self.system)

        self.transcript = QPlainTextEdit(self); self.transcript.setReadOnly(True); self.transcript.setPlaceholderText("Assistant responses will appear here..."); root.addWidget(self.transcript, 1)

        self.input = QPlainTextEdit(self); self.input.setPlaceholderText("Type and press Send… (Shift+Enter for newline)"); self.input.setFixedHeight(100); self.input.installEventFilter(self); root.addWidget(self.input)

        # Simple status/debug line
        self.status = QLabel("")
        root.addWidget(self.status)

        row = QHBoxLayout()
        self.chk_attach = QCheckBox("Attach current frame"); row.addWidget(self.chk_attach)
        self.chk_attach.setToolTip("Attach the current preview frame to your next message.")
        self.chk_stream = QCheckBox("Stream"); self.chk_stream.setChecked(True); row.addWidget(self.chk_stream)
        self.chk_stream.setToolTip("Stream tokens as they generate for lower latency.")
        self.chk_enter_send = QCheckBox("Enter sends message"); row.addWidget(self.chk_enter_send)
        self.chk_enter_send.setToolTip("Press Enter to send; Shift+Enter for a newline.")
        row.addStretch(1)
        self.btn_screenshot = QPushButton("Screenshot")
        self.btn_screenshot.setToolTip("Capture and attach a screenshot of the app window.")
        row.addWidget(self.btn_screenshot)
        self.btn_reset = QPushButton("Reset"); self.btn_close = QPushButton("Close"); self.btn_send = QPushButton("Send")
        self.btn_close.setToolTip("Close the Ask window.")
        self.btn_send.setToolTip("Send your message.")
        self.btn_reset.setToolTip("Reset input/toggles to defaults.")
        row.addWidget(self.btn_reset); row.addWidget(self.btn_close); row.addWidget(self.btn_send); root.addLayout(row)

        self.btn_close.clicked.connect(self.close)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_send.clicked.connect(self._on_send)
        self.btn_screenshot.clicked.connect(self._on_screenshot)

        # Framie prefs/pins
        self._prefs = _prefs_load()
        self.combo_greet.setCurrentText(self._prefs.get("greeting","Professional"))
        self.combo_mode.setCurrentText(self._prefs.get("mode","Coach"))
        self.chk_auto_frame.setChecked(bool(self._prefs.get("auto_frame", True)))
        self.combo_backend.setCurrentText(self._prefs.get("backend","Transformers (VL)"))
        self.combo_device.setCurrentText(self._prefs.get("device","auto"))
        self.combo_preset.setCurrentText({"q4":"Tiny (Q4)","q5":"Balanced (Q5)","q8":"Quality (Q8)"}.get(self._prefs.get("preset","q5"), "Balanced (Q5)"))
        self.gguf_path.setPlainText(self._prefs.get("gguf",""))
        
        self.chk_stream.setChecked(bool(self._prefs.get("stream", True)))
        self.chk_attach.setChecked(bool(self._prefs.get("attach", False)))# New: Enter sends message preference
        try:
            self.chk_enter_send.setChecked(bool(self._prefs.get("enter_send", True)))
        except Exception:
            pass

        self._pins = _pins_load()
        self._last_img_hash = ""
        self._pending_img = None

        # Greeting variants
        g = self.combo_greet.currentText()
        greet_map = {
            "Professional": "Hi. I’m Framie — your FrameVision assistant. How can I help?",
            "Casual": "Hi! I’m Framie — your FrameVision assistant. How can I help?",
            "Playful": "hey, I’m Framie — your FrameVision sidekick. what are we fixing today?"
        }
        self._append("Assistant", greet_map.get(g, greet_map["Casual"]))

        self._update_input_placeholder()

        # Save-on-change
        self.chk_stream.toggled.connect(lambda v: self._prefs.__setitem__("stream", bool(v)) or _prefs_save(self._prefs))
        self.chk_attach.toggled.connect(lambda v: self._prefs.__setitem__("attach", bool(v)) or _prefs_save(self._prefs))
        self.combo_greet.currentTextChanged.connect(lambda v: self._prefs.__setitem__("greeting", v) or _prefs_save(self._prefs))
        self.combo_mode.currentTextChanged.connect(lambda v: self._prefs.__setitem__("mode", v) or _prefs_save(self._prefs))
        self.chk_auto_frame.toggled.connect(lambda v: self._prefs.__setitem__("auto_frame", bool(v)) or _prefs_save(self._prefs))
        self.combo_backend.currentTextChanged.connect(lambda v: self._prefs.__setitem__("backend", v) or _prefs_save(self._prefs))
        self.combo_device.currentTextChanged.connect(lambda v: self._prefs.__setitem__("device", v) or _prefs_save(self._prefs))
        self.combo_preset.currentTextChanged.connect(lambda v: self._prefs.__setitem__("preset", "q4" if v.startswith("Tiny") else "q8" if v.startswith("Quality") else "q5") or _prefs_save(self._prefs))
        # New: enter sends toggle persistence + placeholder refresh
        self.chk_enter_send.toggled.connect(lambda v: self._prefs.__setitem__("enter_send", bool(v)) or _prefs_save(self._prefs))
        self.chk_enter_send.toggled.connect(lambda _: self._update_input_placeholder())

        

    def _append(self, who: str, text: str = ""):
        text = self._sanitize_text(text)
        if who: self.transcript.appendPlainText(f"{who}: {text}")
        else: self.transcript.appendPlainText(text)
        self.transcript.moveCursor(QTextCursor.End)
        try: self._update_hud()
        except Exception: pass

    

    def _current_asst_prefix(self) -> str:
        try:
            t = self.transcript.toPlainText().splitlines()
            if t and t[-1].startswith("Assistant: "):
                return t[-1][len("Assistant: "):]
        except Exception:
            pass
        return ""

    def _strip_system_leak(self, incoming: str) -> str:
        try:
            if not getattr(self, "_filter_active", False):
                return incoming

            existing = self._current_asst_prefix()
            probe = (existing + incoming)

            prefixes = []
            try:
                prefixes.append(DEFAULT_SYSTEM.strip())
            except Exception:
                pass
            xsys = getattr(self, "_filter_extra_sys", "")
            if xsys:
                prefixes.append(xsys.strip())
            prefixes += [
                "Be terse, command-like, prefer concrete commands and code.",
                "Coach the user step-by-step, friendly and encouraging.",
                "Explain like I am five: simple words and short sentences.",
                "Proactively ask for logs",
            ]
            q = getattr(self, "_last_user_q", "")
            for pref in [q, f"You: {q}", f"User: {q}", f"Human: {q}"]:
                if pref:
                    prefixes.append(pref)

            def strip_prefix(s, pref):
                s2 = s.lstrip()
                if s2.startswith(pref):
                    lead = len(s) - len(s2)
                    return s[:lead] + s2[len(pref):], True
                return s, False

            changed = False
            for _ in range(3):
                for pref in prefixes:
                    if not pref: continue
                    probe2, did = strip_prefix(probe, pref)
                    if did:
                        probe = probe2; changed = True

            if probe.strip() == "":
                return ""

            if any(ch.isalnum() for ch in probe[:32]):
                self._filter_active = False

            tail = probe[len(existing):]
            if changed and tail.startswith("\n"):
                tail = tail[1:]
            return tail
        except Exception:
            return incoming
    def _append_inline(self, text: str):
        text = self._sanitize_text(text)
        text = self._strip_system_leak(text)
        c = self.transcript.textCursor(); c.movePosition(QTextCursor.End); c.insertText(text); self.transcript.setTextCursor(c); self.transcript.ensureCursorVisible()
        try: self._update_hud()
        except Exception: pass

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
                    # Queue this screenshot for the next question instead of auto-sending
                    self._pending_img = qimg
                    try:
                        self.chk_attach.setChecked(True)
                    except Exception:
                        pass
                    self.status.setText("Screenshot captured. Ask a question and press Send — I'll include it.")
                    self.input.setFocus()
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
            self._pending_img = pm.toImage()
            try:
                self.chk_attach.setChecked(True)
            except Exception:
                pass
            self.status.setText("Screenshot captured. Ask a question and press Send — I'll include it.")
            self.input.setFocus()
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

        backend = self.combo_backend.currentText() if hasattr(self, "combo_backend") else "Transformers (VL)"

        attach_needed = self.chk_attach.isChecked() or (getattr(self, "chk_auto_frame", None) and self.chk_auto_frame.isChecked() and any(k in q.lower() for k in ["this frame","current frame","this shot","this image"]))
        qimg = self._grab_qimage() if attach_needed else None
        if qimg is not None:
            qimg = _downscale_qimage(qimg, 1024)
            h = _qimage_sha1(qimg)
            # Only dedup if we successfully computed a hash
            if h and h == getattr(self, "_last_img_hash", ""):
                qimg = None
            else:
                if h:
                    self._last_img_hash = h

        pil = _qimage_to_pil(qimg) if qimg is not None else None
        if self.chk_attach.isChecked():
            if qimg is None:
                self._debug("Attach requested but no QImage available.")
            else:
                self._debug(f"Attach ON: QImage {qimg.width()}x{qimg.height()} -> PIL {'OK' if pil is not None else 'FAIL'}")

        # If asking about the app, include local info as extra system context
        extra_sys = self.system.toPlainText()
        # ##dedup_default: prevent DEFAULT_SYSTEM duplication
        try:
            ds = DEFAULT_SYSTEM.strip()
            if extra_sys.strip().replace('\r','').replace('\n',' ') == ds.replace('\r','').replace('\n',' '):
                extra_sys = ''
            else:
                extra_sys = extra_sys.replace(ds, '').strip()
        except Exception:
            pass
        mode = getattr(self, "combo_mode", None).currentText() if hasattr(self, "combo_mode") else "Coach"
        if mode == "Coach":
            extra_sys = "Coach the user step-by-step, friendly and encouraging.\n" + extra_sys
        elif mode == "Power User":
            extra_sys = "Be terse, command-like, prefer concrete commands and code.\n" + extra_sys
        elif mode == "ELI5":
            extra_sys = "Explain like I am five: simple words and short sentences.\n" + extra_sys
        elif mode == "Bug hunt":
            extra_sys = "Proactively ask for logs, reproduction steps, and show a checklist to isolate the bug.\n" + extra_sys

        if 'framevision' in q.lower() or 'this app' in q.lower():
            _load_info_text()
            if _LOCAL['info_text']:
                extra_sys = (extra_sys + '\\n\\n' + 'APP_INFO:\\n' + _LOCAL['info_text']).strip()

        gen = GenConfig(temperature=float(self.spin_temp.value()), max_new_tokens=int(self.spin_max.value()), repetition_penalty=1.05)
        self._last_user_q = q
        self._filter_extra_sys = extra_sys
        self._filter_active = True
        self._assist_chars = 0
        msgs = _build_messages(extra_sys, self._history, q, pil)

        self.btn_send.setEnabled(False)
        self._gen_start = time.time()
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

    

    def _browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF model", "", "GGUF (*.gguf)")
        if path:
            self.gguf_path.setPlainText(path)
            self._prefs["gguf"] = path; _prefs_save(self._prefs)
            self.status.setText(f"GGUF selected ({_gguf_ram_hint(path)})")

    def _select_llama_bin(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select llama.cpp binary", "", "Executables (*)")
        if path:
            self._prefs["llama_bin"] = path; _prefs_save(self._prefs)
            self.status.setText("llama.cpp binary set.")

    def _pin_last(self):
        lines = self.transcript.toPlainText().splitlines()
        last = ""
        for s in reversed(lines):
            if s.startswith("Assistant: "):
                last = s[len("Assistant: "):].strip(); break
        if not last:
            self.status.setText("Nothing to pin."); return
        self._pins.append({"role":"assistant","text": last, "ts": time.time()})
        _pins_save(self._pins); self.status.setText("Pinned.")

    def _show_pins(self):
        items = [f"{i+1}. {p['text'][:80]}…" if len(p['text'])>80 else f"{i+1}. {p['text']}" for i,p in enumerate(self._pins)]
        if not items:
            QInputDialog.getText(self, "Pinned", "No pins yet. (Use 📌 Pin last)")
            return
        idx, ok = QInputDialog.getInt(self, "Pinned", "Pick # to insert:", 1, 1, len(items), 1)
        if ok:
            pin = self._pins[idx-1]["text"]
            self.input.insertPlainText(pin)

    def _update_hud(self):
        try:
            tok = getattr(_GLOBAL.get("processor", None), "tokenizer", None)
            ctx_tokens, out_tokens = 0, 0
            if tok is not None:
                ctx_txt = ""
                for m in self._history:
                    for c in m.get("content", []):
                        if c.get("type") == "text":
                            ctx_txt += c.get("text", "") + "\n"
                last_asst = ""
                lines = self.transcript.toPlainText().splitlines()
                for s in reversed(lines):
                    if s.startswith("Assistant: "):
                        last_asst = s[len("Assistant: "):]; break
                try:
                    out_tokens = len(tok.encode(last_asst)) if hasattr(tok,"encode") else len(tok(last_asst)["input_ids"])
                    ctx_tokens = len(tok.encode(ctx_txt)) if hasattr(tok,"encode") else len(tok(ctx_txt)["input_ids"])
                except Exception:
                    pass
            elapsed = (time.time() - getattr(self, "_gen_start", time.time())) if hasattr(self, "_gen_start") else 0.0
            rate = (out_tokens/elapsed) if elapsed>0 else 0.0
            self.hud.setText(f"ctx≈{ctx_tokens} • out {out_tokens} • {rate:.1f} tok/s • {elapsed:.1f}s")
        except Exception:
            pass

    def _tool_analyze_clip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Analyze clip (ffprobe)", "", "Video files (*.mp4 *.mov *.mkv *.avi *.mxf);;All files (*.*)")
        if not path: return
        args = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
        try:
            import subprocess
            out = subprocess.run(args, capture_output=True, text=True, timeout=20)
            rep = out.stdout.strip() or out.stderr.strip()
            self._append("Assistant", "[ffprobe report]\n" + rep[:16000])
            self._update_hud()
        except Exception as e:
            self._on_error(f"ffprobe error: {e}")

    def _tool_dry_run(self):
        path, _ = QFileDialog.getOpenFileName(self, "Dry-run export (ffmpeg null)", "", "Video files (*.mp4 *.mov *.mkv *.avi *.mxf);;All files (*.*)")
        if not path: return
        args = ["ffmpeg", "-v", "info", "-i", path, "-t", "1", "-f", "null", "-"]
        try:
            import subprocess
            out = subprocess.run(args, capture_output=True, text=True, timeout=30)
            log = (out.stdout + "\n" + out.stderr)[-16000:]
            self._append("Assistant", "[ffmpeg dry-run log]\n" + log)
            self._update_hud()
        except Exception as e:
            self._on_error(f"ffmpeg error: {e}")

    def _tool_apply_ffmpeg_from_last(self):
        last_block = ""
        lines = self.transcript.toPlainText().splitlines()
        collect = False
        for s in lines:
            if s.startswith("Assistant: "):
                last_block = s[len("Assistant: "):] + "\n"; collect = True
            elif collect:
                last_block += s + "\n"
        cmd = None
        for ln in last_block.splitlines():
            if ln.strip().startswith("ffmpeg "):
                cmd = ln.strip(); break
        if not cmd:
            self.status.setText("No ffmpeg command found in last answer."); return
        QApplication.clipboard().setText(cmd)
        self.status.setText("ffmpeg command copied to clipboard.")


def _askpopup_update_input_placeholder(self):
    try:
        if getattr(self, "chk_enter_send", None) and self.chk_enter_send.isChecked():
            self.input.setPlaceholderText("Type and press Enter to send (Shift+Enter for newline)")
        else:
            self.input.setPlaceholderText("Type and press Send… (Enter for newline)")
    except Exception:
        pass


def eventFilter(self, obj, ev):
    # Key handling for the chat input
    if getattr(self, "input", None) is obj and getattr(ev, "type", lambda: None)() == QEvent.KeyPress:
        ke = ev  # QKeyEvent
        try:
            key = ke.key()
        except Exception:
            key = None
        if key in (Qt.Key_Return, Qt.Key_Enter):
            # If "Enter sends message" is OFF, let Enter insert a newline
            if not getattr(self, "chk_enter_send", None) or not self.chk_enter_send.isChecked():
                return False
            # Shift+Enter always makes a newline
            if getattr(ke, "modifiers", lambda: 0)() & Qt.ShiftModifier:
                return False
            # Otherwise, send the message
            try:
                self._on_send()
            except Exception:
                pass
            return True
    # Fallback to default behavior
    try:
        return super(AskPopup, self).eventFilter(obj, ev)
    except Exception:
        return False
class LlamaWorker(QThread):
    chunk = Signal(str)
    done = Signal()
    error = Signal(str)

    def __init__(self, prompt: str, gguf_path: str, max_new_tokens: int = 512, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.gguf = gguf_path
        self.max_new_tokens = max_new_tokens

    def run(self):
        try:
            # Resolve llama binary from prefs or PATH
            import json, shutil, subprocess
            prefs = json.loads(PREFS_PATH.read_text(encoding="utf-8")) if PREFS_PATH.exists() else {}
            cand = prefs.get("llama_bin", "")
            binpath = cand if (cand and Path(cand).exists()) else (shutil.which("llama") or shutil.which("llama-cli") or shutil.which("main"))
            if not binpath:
                self.error.emit("llama.cpp binary not found. Set it in Tools menu or PATH."); return
            args = [binpath, "-m", self.gguf, "-p", self.prompt, "-n", str(int(self.max_new_tokens))]
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                if line: self.chunk.emit(line.rstrip())
            proc.wait()
            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))

# Bind patched helpers onto AskPopup
try:
    AskPopup._update_input_placeholder = _askpopup_update_input_placeholder
    AskPopup.eventFilter = eventFilter
except Exception:
    pass
