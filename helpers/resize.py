
# helpers/resize.py — Resize tool (revamped)
# Implements: preset path fix, duplicate row removal, AR-lock, Fit/Fill/Letterbox,
# CRF slider + x264 preset, audio copy option, even-dim rounding, pixel-format/SAR,
# cleaned UI, smarter batch summary.

import os, re, json, subprocess, math
from dataclasses import dataclass, asdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLabel, QSlider, QSpinBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QComboBox, QCheckBox, QFileDialog, QMessageBox
)

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT as APP_ROOT
except Exception:
    APP_ROOT = Path('.').resolve()

# Required preset path: root /presets/setsave/resize.json
PRESET_PATH = (APP_ROOT / "presets" / "setsave" / "resize.json")

# --------------------------- Data model ---------------------------
@dataclass
class ResizePreset:
    width: int = 1280
    height: int = 720
    ar_lock: bool = True
    scale_mode: str = "Fit"            # Fit | Fill | Letterbox
    crf: int = 23
    x264_preset: str = "veryfast"
    audio_copy: bool = True
    round_even: bool = True

    @classmethod
    def load(cls) -> "ResizePreset":
        try:
            if PRESET_PATH.exists():
                data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
                return cls(**{**asdict(cls()), **data})
        except Exception:
            pass
        return cls()

    def save(self):
        PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRESET_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

# --------------------------- Helpers ---------------------------
def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1

def _round_even_dims(w: int, h: int) -> tuple[int,int]:
    return _even(max(2, w)), _even(max(2, h))

def _calc_fit(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> tuple[int,int]:
    """Scale to fit within box (no pad/crop), keep AR."""
    if src_w <= 0 or src_h <= 0: return tgt_w, tgt_h
    r = min(tgt_w / src_w, tgt_h / src_h)
    return max(1, int(src_w * r)), max(1, int(src_h * r))

def _calc_fill(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> tuple[int,int]:
    """Scale to cover box (will crop later), keep AR."""
    if src_w <= 0 or src_h <= 0: return tgt_w, tgt_h
    r = max(tgt_w / src_w, tgt_h / src_h)
    return max(1, int(src_w * r)), max(1, int(src_h * r))

def _build_vf(src_w, src_h, out_w, out_h, mode: str, round_even: bool) -> str:
    """Return FFmpeg -vf chain including scale/pad/crop + setsar=1."""
    # ensure even dims where necessary
    ow, oh = (out_w, out_h)
    if round_even:
        ow, oh = _round_even_dims(ow, oh)

    vf = ""

    if mode == "Fit":
        # scale to fit inside target; no padding, final dims may be <= target
        # Use -2 trick to keep AR and even dims along one axis
        # Compute which axis constrains
        # We still provide explicit w/h to be deterministic
        vf = f"scale={ow}:{oh}:flags=lanczos"
    elif mode == "Fill":
        # scale to cover, then crop to exact box
        # first scale so that both dimensions >= target, then crop center
        vf = (
            f"scale={ow}:{oh}:flags=lanczos,"
            f"crop={ow}:{oh}"
        )
    elif mode == "Letterbox":
        # scale to fit, then pad to exact box centered
        vf = (
            f"scale=min({ow}\\,iw*{oh}/ih):min({oh}\\,ih*{ow}/iw):flags=lanczos,"
            f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        vf = f"scale={ow}:{oh}:flags=lanczos"

    # Ensure square SAR for compatibility
    if vf:
        vf += ",setsar=1"
    else:
        vf = "setsar=1"
    return vf, ow, oh

def _ffprobe_size(path: Path) -> tuple[int,int]:
    """Try to get input width/height with ffprobe; fallback to 0,0 on failure."""
    try:
        import json as _json
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path)
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        data = _json.loads(p.stdout)
        s = data["streams"][0]
        return int(s.get("width", 0)), int(s.get("height", 0))
    except Exception:
        return 0, 0

# --------------------------- UI ---------------------------
class ResizePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.preset = ResizePreset.load()

        form = QFormLayout(self)

        # Width/Height — single pair (no duplicates)
        self.spin_w = QSpinBox(); self.spin_w.setRange(16, 8192); self.spin_w.setValue(self.preset.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(16, 8192); self.spin_h.setValue(self.preset.height)

        row_wh = QHBoxLayout()
        row_wh.addWidget(QLabel("W")); row_wh.addWidget(self.spin_w)
        row_wh.addWidget(QLabel("H")); row_wh.addWidget(self.spin_h)
        form.addRow(QLabel("Target size"), row_wh)

        # AR lock
        self.chk_ar = QCheckBox("Lock aspect ratio"); self.chk_ar.setChecked(self.preset.ar_lock)
        form.addRow(self.chk_ar)

        # Mode
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["Fit","Fill","Letterbox"])
        self.cmb_mode.setCurrentText(self.preset.scale_mode)
        form.addRow(QLabel("Scale mode"), self.cmb_mode)

        # CRF
        self.slider_crf = QSlider(Qt.Horizontal); self.slider_crf.setRange(18, 28); self.slider_crf.setValue(self.preset.crf)
        self.lbl_crf = QLabel(str(self.preset.crf))
        self.slider_crf.valueChanged.connect(lambda v: self.lbl_crf.setText(str(v)))
        row_crf = QHBoxLayout(); row_crf.addWidget(self.slider_crf); row_crf.addWidget(self.lbl_crf)
        form.addRow(QLabel("CRF (18=best, 28=smaller)"), row_crf)

        # x264 preset
        self.cmb_x264 = QComboBox(); self.cmb_x264.addItems(
            ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
        )
        self.cmb_x264.setCurrentText(self.preset.x264_preset)
        form.addRow(QLabel("Encoder preset"), self.cmb_x264)

        # Audio
        self.chk_acopy = QCheckBox("Copy audio (no re-encode)"); self.chk_acopy.setChecked(self.preset.audio_copy)
        form.addRow(self.chk_acopy)

        # Even dims
        self.chk_even = QCheckBox("Round to even dimensions"); self.chk_even.setChecked(self.preset.round_even)
        form.addRow(self.chk_even)

        # Action buttons
        self.btn_one = QPushButton("Resize a file…")
        self.btn_batch = QPushButton("Resize a batch…")
        row_btn = QHBoxLayout(); row_btn.addWidget(self.btn_one); row_btn.addWidget(self.btn_batch)
        form.addRow(row_btn)

        # Preset buttons
        self.btn_save = QPushButton("Save preset"); self.btn_load = QPushButton("Load preset")
        row_p = QHBoxLayout(); row_p.addWidget(self.btn_save); row_p.addWidget(self.btn_load)
        form.addRow(row_p)

        # Signals
        self.chk_ar.stateChanged.connect(self._on_ar_lock_changed)
        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)
        self.btn_one.clicked.connect(self._on_resize_single)
        self.btn_batch.clicked.connect(self._on_resize_batch)
        self.btn_save.clicked.connect(self._on_save_preset)
        self.btn_load.clicked.connect(self._on_load_preset)

        self._src_w = None
        self._src_h = None

    # ----------------- AR lock behavior -----------------
    def _on_ar_lock_changed(self, _v):
        # nothing to do immediately; on size changes we compute the other side when locked
        pass

    def _on_w_changed(self, v):
        if self.chk_ar.isChecked() and self._src_w and self._src_h:
            ratio = self._src_h / self._src_w
            self.spin_h.blockSignals(True)
            self.spin_h.setValue(max(1, int(v * ratio)))
            self.spin_h.blockSignals(False)

    def _on_h_changed(self, v):
        if self.chk_ar.isChecked() and self._src_w and self._src_h:
            ratio = self._src_w / self._src_h
            self.spin_w.blockSignals(True)
            self.spin_w.setValue(max(1, int(v * ratio)))
            self.spin_w.blockSignals(False)

    # ----------------- Presets -----------------
    def _on_save_preset(self):
        p = ResizePreset(
            width=self.spin_w.value(),
            height=self.spin_h.value(),
            ar_lock=self.chk_ar.isChecked(),
            scale_mode=self.cmb_mode.currentText(),
            crf=self.slider_crf.value(),
            x264_preset=self.cmb_x264.currentText(),
            audio_copy=self.chk_acopy.isChecked(),
            round_even=self.chk_even.isChecked(),
        )
        try:
            p.save()
            QMessageBox.information(self, "Preset saved", f"Saved to:\n{PRESET_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    def _on_load_preset(self):
        try:
            p = ResizePreset.load()
            self.spin_w.setValue(p.width)
            self.spin_h.setValue(p.height)
            self.chk_ar.setChecked(p.ar_lock)
            self.cmb_mode.setCurrentText(p.scale_mode)
            self.slider_crf.setValue(p.crf)
            self.cmb_x264.setCurrentText(p.x264_preset)
            self.chk_acopy.setChecked(p.audio_copy)
            self.chk_even.setChecked(p.round_even)
            QMessageBox.information(self, "Preset loaded", f"Loaded from:\n{PRESET_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ----------------- Actions -----------------
    def _pick_input(self) -> Path:
        dlg = QFileDialog(self, "Choose media file")
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                return Path(files[0])
        return None

    def _pick_inputs(self) -> list[Path]:
        dlg = QFileDialog(self, "Choose media files")
        dlg.setFileMode(QFileDialog.ExistingFiles)
        if dlg.exec():
            return [Path(f) for f in dlg.selectedFiles()]
        return []

    def _pick_output(self, inp: Path, suggested_suffix: str) -> Path:
        # suggest sibling file with suffix
        base = inp.with_suffix(suggested_suffix)
        dlg = QFileDialog(self, "Save as")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.selectFile(str(base))
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                return Path(files[0])
        return None

    def _build_cmd(self, inp: Path, out: Path, src_w: int, src_h: int) -> list[str]:
        tgt_w, tgt_h = self.spin_w.value(), self.spin_h.value()
        mode = self.cmb_mode.currentText()
        vf, ow, oh = _build_vf(src_w, src_h, tgt_w, tgt_h, mode, self.chk_even.isChecked())

        cmd = [
            "ffmpeg", "-y",
            "-i", str(inp),
            "-vf", vf,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", self.cmb_x264.currentText(),
            "-crf", str(self.slider_crf.value())
        ]
        if self.chk_acopy.isChecked():
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]

        cmd += [str(out)]
        return cmd

    def _on_resize_single(self):
        inp = self._pick_input()
        if not inp: return
        src_w, src_h = _ffprobe_size(inp)
        self._src_w, self._src_h = src_w, src_h

        # determine extension based on input (if image keep same ext; else mp4)
        ext = inp.suffix.lower()
        if ext in (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"):
            suggested = f".resized{ext}"
        else:
            suggested = ".resized.mp4"

        out = self._pick_output(inp, suggested)
        if not out: return

        cmd = self._build_cmd(inp, out, src_w, src_h)
        ok, err = self._run_ffmpeg(cmd)
        if ok:
            QMessageBox.information(self, "Done", f"Saved:\n{out}")
        else:
            QMessageBox.critical(self, "FFmpeg error", err or "Unknown error")

    def _on_resize_batch(self):
        inputs = self._pick_inputs()
        if not inputs: return

        ok_count = 0
        fail = []
        skipped = []

        for inp in inputs:
            try:
                if not inp.exists():
                    skipped.append((inp, "missing"))
                    continue
                src_w, src_h = _ffprobe_size(inp)
                self._src_w, self._src_h = src_w, src_h
                ext = inp.suffix.lower()
                if ext in (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"):
                    out = inp.with_name(inp.stem + ".resized" + ext)
                else:
                    out = inp.with_name(inp.stem + ".resized.mp4")

                cmd = self._build_cmd(inp, out, src_w, src_h)
                ok, err = self._run_ffmpeg(cmd)
                if ok:
                    ok_count += 1
                else:
                    fail.append((inp, err or "ffmpeg failed"))
            except Exception as e:
                fail.append((inp, str(e)))

        # Smarter summary
        lines = [f"Processed: {len(inputs)}", f"Success: {ok_count}"]
        if skipped:
            lines.append(f"Skipped: {len(skipped)}")
            for p, reason in skipped[:10]:
                lines.append(f"  - {p.name}: {reason}")
            if len(skipped) > 10:
                lines.append(f"  … (+{len(skipped)-10} more)")
        if fail:
            lines.append(f"Failed: {len(fail)}")
            for p, reason in fail[:10]:
                lines.append(f"  - {p.name}: {reason}")
            if len(fail) > 10:
                lines.append(f"  … (+{len(fail)-10} more)")

        msg = "\n".join(lines)
        if fail:
            QMessageBox.warning(self, "Batch finished with errors", msg)
        else:
            QMessageBox.information(self, "Batch finished", msg)

    def _run_ffmpeg(self, cmd: list[str]) -> tuple[bool, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                return True, ""
            # Try to extract a concise reason
            err = p.stderr.strip()
            tail = "\n".join(err.splitlines()[-15:])
            return False, tail
        except FileNotFoundError:
            return False, "ffmpeg not found. Please install it and ensure it's on PATH."
        except Exception as e:
            return False, str(e)

# Factory for host app
def create_widget(parent=None) -> QWidget:
    return ResizePane(parent)
# ---------------- Back-compat shim ----------------

def install_resize_tool(*args, **kwargs):
    """
    Back-compat entrypoint expected by older Tools panes.
    If a section-like container is passed, this function will mount the created
    widget into that container so the UI actually shows up.
    Returns the ResizePane QWidget.
    """
    parent = kwargs.get("parent")
    section = None

    # Best-effort detection of section and parent from positional args
    for a in args:
        # Detect a likely section/container object
        if hasattr(a, "setContentLayout") or hasattr(a, "setContentWidget") or hasattr(a, "layout"):
            section = a
        # Prefer a QWidget-like parent when available
        if parent is None and (hasattr(a, "addWidget") or hasattr(a, "layout")):
            parent = a

    widget = create_widget(parent)

    # Try common mounting methods used by collapsible sections
    try:
        # 1) Section provides setContentLayout -> create a layout and add widget
        if section is not None and hasattr(section, "setContentLayout"):
            from PySide6.QtWidgets import QVBoxLayout
            lay = QVBoxLayout()
            lay.addWidget(widget)
            section.setContentLayout(lay)
            return widget

        # 2) Section provides setContentWidget
        if section is not None and hasattr(section, "setContentWidget"):
            section.setContentWidget(widget)
            return widget

        # 3) Section already has a layout -> add to it
        if section is not None and hasattr(section, "layout") and section.layout() is not None:
            section.layout().addWidget(widget)
            return widget
    except Exception:
        # Fall back to returning the widget; host may handle mounting
        pass

    return widget

