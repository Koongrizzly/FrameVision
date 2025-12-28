# helpers/cropper.py — Crop tool (Video & Image) with working buttons
import os, sys, json, subprocess
from pathlib import Path
from PySide6.QtWidgets import (
    QLabel, QSlider, QPushButton, QFormLayout, QHBoxLayout, QVBoxLayout,
    QSpinBox, QMessageBox, QFileDialog, QComboBox, QWidget
)
from PySide6.QtCore import Qt

# --- Imports from app (with fallbacks) ---
try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS, OUT_PHOTOS, config, save_config
except Exception:
    ROOT = Path('.').resolve()
    OUT_VIDEOS = ROOT / 'output' / 'video'
    OUT_PHOTOS = ROOT / 'output' / 'photo'
    class _Cfg(dict): pass
    config = _Cfg()
    def save_config(): pass

VIDEO_EXTS = {'.mp4','.mov','.mkv','.avi','.m4v','.webm','.ts','.m2ts','.wmv','.flv','.mpg','.mpeg','.3gp','.3g2','.ogv'}
IMAGE_EXTS = {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}

# --- Local helpers (no circular imports) ---
def ffmpeg_path():
    candidates = [ROOT / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"), "ffmpeg"]
    for c in candidates:
        try:
            subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return "ffmpeg"

def ffprobe_path():
    candidates = [ROOT / "bin" / ("ffprobe.exe" if os.name == "nt" else "ffprobe"), "ffprobe"]
    for c in candidates:
        try:
            subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return "ffprobe"

def probe_media(path: Path):
    info = {"width": None, "height": None}
    try:
        out = subprocess.check_output([
            ffprobe_path(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(path)
        ], stderr=subprocess.STDOUT, universal_newlines=True)
        for line in out.splitlines():
            if line.startswith("width="):
                info["width"] = int(line.split("=")[1])
            elif line.startswith("height="):
                info["height"] = int(line.split("=")[1])
    except Exception:
        # fallback for images
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
            info["width"] = w; info["height"] = h
        except Exception:
            pass
    return info

# --- Preset path helpers (use app's config folder if available) ---
def _preset_dir():
    try:
        base = Path(config.get("last_preset_dir", str(ROOT / "presets" / "Tools")))
    except Exception:
        base = ROOT / "presets" / "Tools"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _choose_save_path(parent, suggested_name):
    d = _preset_dir()
    path = d / suggested_name
    fn, _ = QFileDialog.getSaveFileName(parent, "Save preset", str(path), "FrameVision Preset (*.json)")
    if not fn:
        return None
    p = Path(fn)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    if p.exists():
        try:
            res = QMessageBox.question(parent, "Overwrite?",
                                       f"{p.name} already exists. Overwrite?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if res != QMessageBox.Yes:
                return None
        except Exception:
            return None
    try:
        config["last_preset_dir"] = str(p.parent); save_config()
    except Exception:
        pass
    return p

def _choose_open_path(parent):
    d = _preset_dir()
    fn, _ = QFileDialog.getOpenFileName(parent, "Load preset", str(d), "FrameVision Preset (*.json)")
    if not fn:
        return None
    p = Path(fn)
    try:
        config["last_preset_dir"] = str(p.parent); save_config()
    except Exception:
        pass
    return p

# --- Installer ---
def install_cropper_tool(host, section_widget):
    """Attach a working Crop tool into the given section.
    New buttons:
      - Use current: use the app's currently loaded media, set max W/H, auto-range, auto-mode
      - Add: pick a media file to use for cropping, set max W/H, auto-range, auto-mode
    Crop requires one of these to be used first; Batch remains independent.
    """
    # Keep a selected media path on the host
    try:
        host._cropper_selected_path
    except Exception:
        host._cropper_selected_path = None

    # Mode
    host.mode_combo = QComboBox(); host.mode_combo.addItems(["Video","Image"])

    # Width/Height rows: [SpinBox][Slider]
    host.crop_w = QSlider(Qt.Horizontal); host.crop_w.setRange(16, 8192)
    host.spin_crop_w = QSpinBox(); host.spin_crop_w.setRange(16, 8192); host.spin_crop_w.setValue(1280)
    host.crop_h = QSlider(Qt.Horizontal); host.crop_h.setRange(16, 8192)
    host.spin_crop_h = QSpinBox(); host.spin_crop_h.setRange(16, 8192); host.spin_crop_h.setValue(720)

    host.crop_w.valueChanged.connect(lambda v: host.spin_crop_w.setValue(int(v)))
    host.spin_crop_w.valueChanged.connect(lambda v: host.crop_w.setValue(int(v)))
    host.crop_h.valueChanged.connect(lambda v: host.spin_crop_h.setValue(int(v)))
    host.spin_crop_h.valueChanged.connect(lambda v: host.crop_h.setValue(int(v)))

    # Video options
    host.vid_opts = QWidget(); vlay = QFormLayout(host.vid_opts)
    host.combo_x264_preset = QComboBox(); host.combo_x264_preset.addItems(["ultrafast","fast","medium","slow"])
    host.combo_x264_preset.setCurrentText("fast")
    vlay.addRow("x264 preset", host.combo_x264_preset)

    # Image options
    host.img_opts = QWidget(); ilay = QFormLayout(host.img_opts)
    host.combo_img_format = QComboBox(); host.combo_img_format.addItems(["PNG","JPG"])
    host.spin_jpg_quality = QSpinBox(); host.spin_jpg_quality.setRange(1,100); host.spin_jpg_quality.setValue(92)
    host.spin_png_compression = QSpinBox(); host.spin_png_compression.setRange(0,9); host.spin_png_compression.setValue(6)
    host.lbl_jpg_quality = QLabel("JPG quality")
    host.lbl_png_compression = QLabel("PNG compression")
    ilay.addRow("Output format", host.combo_img_format)
    ilay.addRow(host.lbl_jpg_quality, host.spin_jpg_quality)
    ilay.addRow(host.lbl_png_compression, host.spin_png_compression)

    # Buttons
    host.btn_crop = QPushButton("Crop")
    host.btn_crop_batch = QPushButton("Batch…")
    host.btn_use_current = QPushButton("Use current")
    host.btn_add = QPushButton("Add")
    host.btn_crop_open_folder = QPushButton("View results")
    host.btn_crop_open_folder.setToolTip("Open these results in Media Explorer.")
    btn_sc = QPushButton("Save preset"); btn_lc = QPushButton("Load preset")

    # Layout
    lay = QFormLayout()
    lay.addRow("Mode", host.mode_combo)

    row_w = QHBoxLayout(); row_w.addWidget(host.spin_crop_w); row_w.addWidget(host.crop_w)
    lay.addRow("Width :", row_w)
    row_h = QHBoxLayout(); row_h.addWidget(host.spin_crop_h); row_h.addWidget(host.crop_h)
    lay.addRow("Height :", row_h)

    row_cb = QHBoxLayout()
    row_cb.addWidget(host.btn_crop); row_cb.addWidget(host.btn_crop_batch)
    row_cb.addWidget(host.btn_use_current); row_cb.addWidget(host.btn_add); row_cb.addWidget(host.btn_crop_open_folder)
    lay.addRow(row_cb)

    lay.addRow("Video options", host.vid_opts)
    lay.addRow("Image options", host.img_opts)

    row_p = QHBoxLayout(); row_p.addWidget(btn_sc); row_p.addWidget(btn_lc)
    lay.addRow(row_p)

    try:
        section_widget.setContentLayout(lay)
    except Exception:
        if hasattr(section_widget, "setLayout"):
            section_widget.setLayout(lay)

    # Mode visibility
    def _update_img_quality_ui():
        fmt = host.combo_img_format.currentText().lower()
        is_jpg = (fmt == "jpg")
        host.lbl_jpg_quality.setVisible(is_jpg)
        host.spin_jpg_quality.setVisible(is_jpg)
        host.lbl_png_compression.setVisible(not is_jpg)
        host.spin_png_compression.setVisible(not is_jpg)

    def _update_mode_ui():
        is_vid = host.mode_combo.currentText().lower() == "video"
        host.vid_opts.setVisible(is_vid); host.img_opts.setVisible(not is_vid)

    host.mode_combo.currentIndexChanged.connect(_update_mode_ui); _update_mode_ui()
    host.combo_img_format.currentIndexChanged.connect(_update_img_quality_ui); _update_img_quality_ui()

    # --- Helpers ---
    IMAGE_EXTS = {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
    VIDEO_EXTS = {'.mp4','.mov','.mkv','.avi','.m4v','.webm','.ts','.m2ts','.wmv','.flv','.mpg','.mpeg','.3gp','.3g2','.ogv'}

    def _auto_mode_for_path(path: Path):
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS: host.mode_combo.setCurrentText("Image")
        elif ext in VIDEO_EXTS: host.mode_combo.setCurrentText("Video")

    def _sync_ranges_to_media(inp: Path):
        info = probe_media(inp)
        iw = int(info.get("width") or 0); ih = int(info.get("height") or 0)
        if iw and ih:
            host.crop_w.setRange(16, max(16, iw)); host.spin_crop_w.setRange(16, max(16, iw))
            host.crop_h.setRange(16, max(16, ih)); host.spin_crop_h.setRange(16, max(16, ih))
        return iw, ih

    def _set_dims_to_max(iw, ih):
        # Set to maximum, adjusting to even sizes for codec-friendliness
        w = max(16, iw); h = max(16, ih)
        if w % 2: w -= 1
        if h % 2: h -= 1
        host.spin_crop_w.setValue(w); host.spin_crop_h.setValue(h)

    def _warn_and_clamp(iw, ih):
        w = int(host.spin_crop_w.value()); h = int(host.spin_crop_h.value())
        ow, oh = w, h
        if iw and w > iw: w = iw
        if ih and h > ih: h = ih
        if w % 2: w -= 1
        if h % 2: h -= 1
        changed = (w != ow) or (h != oh)
        if changed:
            host.spin_crop_w.setValue(max(16, w)); host.spin_crop_h.setValue(max(16, h))
            try:
                QMessageBox.warning(host, "Crop too big",
                    f"Selected crop {ow}x{oh} exceeds/needs adjust for source {iw}x{ih}.\n"
                    f"Maximum even crop for this media is {w}x{h}.\n"
                    "The values and sliders have been adjusted accordingly.")
            except Exception:
                pass
        return int(host.spin_crop_w.value()), int(host.spin_crop_h.value())

    def _center(iw, ih, w, h):
        x = max(0, (iw - w) // 2); y = max(0, (ih - h) // 2)
        return x, y

    # --- New Buttons ---
    def use_current():
        try:
            inp = host._ensure_input()
        except Exception:
            inp = None
        if not inp:
            try: QMessageBox.information(host, "No current media", "Nothing is loaded in the viewer.")
            except Exception: pass
            return
        host._cropper_selected_path = inp
        _auto_mode_for_path(inp)
        iw, ih = _sync_ranges_to_media(inp)
        if iw and ih:
            _set_dims_to_max(iw, ih)

    def add_media():
        try:
            # Build a simple filter string
            media_exts = " *.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts *.m2ts *.wmv *.flv *.mpg *.mpeg *.3gp *.3g2 *.ogv *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"
            fn, _ = QFileDialog.getOpenFileName(host, "Select media", str(Path.home()), f"Media files ({media_exts})")
        except Exception:
            fn = ""
        if not fn:
            return
        inp = Path(fn)
        host._cropper_selected_path = inp
        _auto_mode_for_path(inp)
        iw, ih = _sync_ranges_to_media(inp)
        if iw and ih:
            _set_dims_to_max(iw, ih)

    # --- Actions ---
    def run_crop_video(inp: Path):
        iw, ih = _sync_ranges_to_media(inp)
        w, h = _warn_and_clamp(iw, ih)
        x, y = _center(iw, ih, w, h)
        out = OUT_VIDEOS / f"{inp.stem}_crop_{w}x{h}.mp4"
        preset = host.combo_x264_preset.currentText()
        filter_str = f"crop={w}:{h}:{x}:{y}"
        cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", filter_str,
               "-c:v", "libx264", "-preset", preset, "-movflags", "+faststart", str(out)]
        host._run(cmd, out)
        try: host._cropper_last_out_dir = out.parent
        except Exception: pass

    def run_crop_image(inp: Path):
        iw, ih = _sync_ranges_to_media(inp)
        w, h = _warn_and_clamp(iw, ih)
        x, y = _center(iw, ih, w, h)
        fmt = host.combo_img_format.currentText().lower()
        suffix = ".png" if fmt == "png" else ".jpg"
        out = OUT_PHOTOS / f"{inp.stem}_crop_{w}x{h}{suffix}"
        filter_str = f"crop={w}:{h}:{x}:{y}"
        if fmt == "jpg":
            q = 2 + int((31 - 2) * (100 - host.spin_jpg_quality.value()) / 100); q = max(2, min(31, q))
            cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", filter_str, "-q:v", str(q), str(out)]
        else:
            # PNG: use compression level 0-9 (higher = smaller file, slower)
            try:
                comp = int(host.spin_png_compression.value())
            except Exception:
                comp = 6
            comp = max(0, min(9, comp))
            cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-vf", filter_str, "-compression_level", str(comp), str(out)]
        host._run(cmd, out)
        try: host._cropper_last_out_dir = out.parent
        except Exception: pass

    def run_crop():
        inp = host._cropper_selected_path
        if not inp:
            try: QMessageBox.warning(host, "No media selected", "Nothing is selected. Use 'Use current' or 'Add' first.")
            except Exception: pass
            return
        # auto-mode check
        _auto_mode_for_path(inp)
        if host.mode_combo.currentText().lower() == "image":
            run_crop_image(inp)
        else:
            run_crop_video(inp)

    def run_crop_batch():
        is_vid = host.mode_combo.currentText().lower() == "video"
        paths = []
        try:
            from helpers.batch import BatchSelectDialog as _BSD
            exts = getattr(_BSD, "VIDEO_EXTS", VIDEO_EXTS) if is_vid else getattr(_BSD, "IMAGE_EXTS", IMAGE_EXTS)
            title = "Batch Crop (Video)" if is_vid else "Batch Crop (Image)"
            paths, _ = _BSD.pick(host, title=title, exts=exts)
        except Exception:
            pass
        paths = list(paths or [])
        if not paths: return
        try:
            if QMessageBox.question(host, "Batch Crop", f"Add {len(paths)} item(s) with current Crop settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        ok = 0
        for p in paths:
            try:
                _p = Path(p)
                info = probe_media(_p); iw = int(info.get("width") or 0); ih = int(info.get("height") or 0)
                # clamp to current W/H, centered
                w, h = _warn_and_clamp(iw, ih)
                x, y = _center(iw, ih, w, h)
                if is_vid:
                    out = OUT_VIDEOS / f"{_p.stem}_crop_{w}x{h}.mp4"
                    filter_str = f"crop={w}:{h}:{x}:{y}"
                    preset = host.combo_x264_preset.currentText()
                    cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-c:v", "libx264", "-preset", preset, "-movflags", "+faststart", str(out)]
                else:
                    fmt = host.combo_img_format.currentText().lower()
                    suffix = ".png" if fmt == "png" else ".jpg"
                    out = OUT_PHOTOS / f"{_p.stem}_crop_{w}x{h}{suffix}"
                    filter_str = f"crop={w}:{h}:{x}:{y}"
                    if fmt == "jpg":
                        q = 2 + int((31 - 2) * (100 - host.spin_jpg_quality.value()) / 100); q = max(2, min(31, q))
                        cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-q:v", str(q), str(out)]
                    else:
                        # PNG: use compression level 0-9 (higher = smaller file, slower)
                        try:
                            comp = int(host.spin_png_compression.value())
                        except Exception:
                            comp = 6
                        comp = max(0, min(9, comp))
                        cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-compression_level", str(comp), str(out)]
                host._enqueue_cmd_for_input(_p, cmd, out); ok += 1
                try: host._cropper_last_out_dir = out.parent
                except Exception: pass
            except Exception:
                continue
        try:
            QMessageBox.information(host, "Batch Crop", f"Queued {ok} item(s)." )
        except Exception:
            pass

    def save_preset_crop():
        w = int(host.spin_crop_w.value()); h = int(host.spin_crop_h.value())
        mode = host.mode_combo.currentText().lower()
        name = f"crop_{mode}_{w}x{h}_preset.json"
        pth = _choose_save_path(host, name)
        if not pth: return
        data = {
            "tool":"crop","mode":mode,
            "w":w,"h":h,
            "x264_preset":host.combo_x264_preset.currentText(),
            "img_format":host.combo_img_format.currentText(),
            "jpg_quality":int(host.spin_jpg_quality.value()),
            "png_compression":int(host.spin_png_compression.value())
        }
        pth.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try: QMessageBox.information(host, "Preset saved", str(pth))
        except Exception: pass

    def load_preset_crop():
        pth = _choose_open_path(host)
        if not pth: return
        try:
            data = json.loads(pth.read_text(encoding="utf-8"))
            if data.get("tool") != "crop": raise ValueError("Wrong preset type")
            mode = data.get("mode","video").capitalize()
            if mode not in ("Video","Image"): mode = "Video"
            host.mode_combo.setCurrentText(mode)
            host.spin_crop_w.setValue(int(data.get("w", data.get("y", host.spin_crop_w.value()))))
            host.spin_crop_h.setValue(int(data.get("h", host.spin_crop_h.value())))
            preset_val = data.get("x264_preset", host.combo_x264_preset.currentText())
            if preset_val not in {"ultrafast","fast","medium","slow"}: preset_val = "fast"
            host.combo_x264_preset.setCurrentText(preset_val)
            host.combo_img_format.setCurrentText(data.get("img_format", host.combo_img_format.currentText()))
            host.spin_jpg_quality.setValue(int(data.get("jpg_quality", host.spin_jpg_quality.value())))
            host.spin_png_compression.setValue(int(data.get("png_compression", host.spin_png_compression.value())))
            try:
                _update_img_quality_ui()
            except Exception:
                pass
        except Exception as e:
            try: QMessageBox.critical(host, "Preset error", str(e))
            except Exception: pass

    
    # --- View results (Media Explorer preferred) ---
    def _open_folder_in_os(folder_path: Path):
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if os.name == "nt":
                os.startfile(str(folder_path))  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder_path)])
            else:
                subprocess.Popen(["xdg-open", str(folder_path)])
        except Exception:
            pass

    def _crop_open_folder():
        # Prefer last output dir if known; else mode-based default
        try:
            folder = getattr(host, "_cropper_last_out_dir", None)
        except Exception:
            folder = None
        if not folder:
            try:
                is_img = (host.mode_combo.currentText().lower() == "image")
            except Exception:
                is_img = False
            folder = (OUT_PHOTOS if is_img else OUT_VIDEOS)

        try:
            fp = folder if isinstance(folder, Path) else Path(str(folder))
        except Exception:
            fp = None

        # Prefer Media Explorer on main window if available
        main = None
        try:
            main = getattr(host, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = host.window() if hasattr(host, "window") else None
            except Exception:
                main = None

        if fp is not None and main is not None and hasattr(main, "open_media_explorer_folder"):
            try:
                preset = "images" if (host.mode_combo.currentText().lower() == "image") else "videos"
            except Exception:
                preset = "videos"
            try:
                main.open_media_explorer_folder(str(fp), preset=preset, include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(fp))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        if fp is not None:
            _open_folder_in_os(fp)

# --- Wiring ---
    host.btn_crop.clicked.connect(run_crop)
    host.btn_crop_batch.clicked.connect(run_crop_batch)
    host.btn_use_current.clicked.connect(use_current)
    host.btn_add.clicked.connect(add_media)
    btn_sc.clicked.connect(save_preset_crop)
    btn_lc.clicked.connect(load_preset_crop)
    try:
        host.btn_crop_open_folder.clicked.connect(_crop_open_folder)
    except Exception:
        pass

    # Compatibility exposure
    try:
        host.run_crop = run_crop
        host.run_crop_batch = run_crop_batch
        host._save_preset_crop = save_preset_crop
        host._load_preset_crop = load_preset_crop
    except Exception:
        pass
        paths = list(paths or [])
        if not paths: return
        try:
            if QMessageBox.question(host, "Batch Crop", f"Add {len(paths)} item(s) with current Crop settings to the queue?") != QMessageBox.Yes:
                return
        except Exception:
            pass
        ok = 0
        for p in paths:
            try:
                _p = Path(p)
                info = probe_media(_p); iw = int(info.get("width") or 0); ih = int(info.get("height") or 0)
                w, h = _warn_and_clamp(iw, ih)
                x, y = _x_y_center(iw, ih, w, h)
                if is_vid:
                    out = OUT_VIDEOS / f"{_p.stem}_crop_{w}x{h}.mp4"
                    filter_str = f"crop={w}:{h}:{x}:{y}"
                    preset = host.combo_x264_preset.currentText()
                    cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-c:v", "libx264", "-preset", preset, "-movflags", "+faststart", str(out)]
                else:
                    fmt = host.combo_img_format.currentText().lower()
                    suffix = ".png" if fmt == "png" else ".jpg"
                    out = OUT_PHOTOS / f"{_p.stem}_crop_{w}x{h}{suffix}"
                    filter_str = f"crop={w}:{h}:{x}:{y}"
                    if fmt == "jpg":
                        q = 2 + int((31 - 2) * (100 - host.spin_jpg_quality.value()) / 100); q = max(2, min(31, q))
                        cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-q:v", str(q), str(out)]
                    else:
                        # PNG: use compression level 0-9 (higher = smaller file, slower)
                        try:
                            comp = int(host.spin_png_compression.value())
                        except Exception:
                            comp = 6
                        comp = max(0, min(9, comp))
                        cmd = [ffmpeg_path(), "-y", "-i", str(_p), "-vf", filter_str, "-compression_level", str(comp), str(out)]
                host._enqueue_cmd_for_input(_p, cmd, out); ok += 1
                try: host._cropper_last_out_dir = out.parent
                except Exception: pass
            except Exception:
                continue
        try:
            QMessageBox.information(host, "Batch Crop", f"Queued {ok} item(s)." )
        except Exception:
            pass

    def save_preset_crop():
        w = int(host.spin_crop_w.value()); h = int(host.spin_crop_h.value())
        mode = host.mode_combo.currentText().lower()
        name = f"crop_{mode}_{w}x{h}_preset.json"
        p = _choose_save_path(host, name)
        if not p: return
        data = {
            "tool":"crop","mode":mode,
            "w":w,"h":h,
            "x264_preset":host.combo_x264_preset.currentText(),
            "img_format":host.combo_img_format.currentText(),
            "jpg_quality":int(host.spin_jpg_quality.value()),
            "png_compression":int(host.spin_png_compression.value())
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try: QMessageBox.information(host, "Preset saved", str(p))
        except Exception: pass

    def load_preset_crop():
        p = _choose_open_path(host)
        if not p: return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("tool") != "crop": raise ValueError("Wrong preset type")
            mode = data.get("mode", "video").capitalize()
            if mode not in ("Video","Image"): mode = "Video"
            host.mode_combo.setCurrentText(mode)
            # accept legacy keys 'y'/'h' as width/height
            host.spin_crop_w.setValue(int(data.get("w", data.get("y", host.spin_crop_w.value()))))
            host.spin_crop_h.setValue(int(data.get("h", host.spin_crop_h.value())))
            # per-mode options
            preset_val = data.get("x264_preset", host.combo_x264_preset.currentText())
            if preset_val not in {"ultrafast","fast","medium","slow"}: preset_val = "fast"
            host.combo_x264_preset.setCurrentText(preset_val)
            host.combo_img_format.setCurrentText(data.get("img_format", host.combo_img_format.currentText()))
            host.spin_jpg_quality.setValue(int(data.get("jpg_quality", host.spin_jpg_quality.value())))
            host.spin_png_compression.setValue(int(data.get("png_compression", host.spin_png_compression.value())))
            try:
                _update_img_quality_ui()
            except Exception:
                pass
        except Exception as e:
            try: QMessageBox.critical(host, "Preset error", str(e))
            except Exception: pass

    
    # --- View results (Media Explorer preferred) ---
    def _open_folder_in_os(folder_path: Path):
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if os.name == "nt":
                os.startfile(str(folder_path))  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder_path)])
            else:
                subprocess.Popen(["xdg-open", str(folder_path)])
        except Exception:
            pass

    def _crop_open_folder():
        # Prefer last output dir if known; else mode-based default
        try:
            folder = getattr(host, "_cropper_last_out_dir", None)
        except Exception:
            folder = None
        if not folder:
            try:
                is_img = (host.mode_combo.currentText().lower() == "image")
            except Exception:
                is_img = False
            folder = (OUT_PHOTOS if is_img else OUT_VIDEOS)

        try:
            fp = folder if isinstance(folder, Path) else Path(str(folder))
        except Exception:
            fp = None

        # Prefer Media Explorer on main window if available
        main = None
        try:
            main = getattr(host, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = host.window() if hasattr(host, "window") else None
            except Exception:
                main = None

        if fp is not None and main is not None and hasattr(main, "open_media_explorer_folder"):
            try:
                preset = "images" if (host.mode_combo.currentText().lower() == "image") else "videos"
            except Exception:
                preset = "videos"
            try:
                main.open_media_explorer_folder(str(fp), preset=preset, include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(fp))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        if fp is not None:
            _open_folder_in_os(fp)

# --- Wiring ---
    host.btn_crop.clicked.connect(run_crop)
    host.btn_crop_batch.clicked.connect(run_crop_batch)
    btn_sc.clicked.connect(save_preset_crop)
    btn_lc.clicked.connect(load_preset_crop)
    try:
        host.btn_crop_open_folder.clicked.connect(_crop_open_folder)
    except Exception:
        pass
    # Compatibility exposure
    try:
        host.run_crop = run_crop
        host.run_crop_batch = run_crop_batch
        host._save_preset_crop = save_preset_crop
        host._load_preset_crop = load_preset_crop
    except Exception:
        pass
