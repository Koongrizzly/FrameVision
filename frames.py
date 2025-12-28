import os
import json
import subprocess
import platform
from pathlib import Path

def _frames_get_input(owner):
    """
    Resolve the active input path for this tool WITHOUT popping host dialogs.
    Priority:
      1) owner._frames_loaded_path
      2) Common path attrs on owner/main: current_path, input_path, video_path
      3) owner._ensure_input() (last resort)
    """
    def _first_existing(paths):
        from pathlib import Path
        for cand in paths:
            if isinstance(cand, (str, Path)) and cand:
                pc = Path(cand)
                if pc.exists():
                    return str(pc)
        return None

    try:
        # 1) direct
        direct = getattr(owner, "_frames_loaded_path", None)
        if direct:
            from pathlib import Path as _P
            return str(_P(direct))

        # 2) common attrs on owner and owner.main
        main = getattr(owner, "main", None)
        probes = []
        for obj in (owner, main):
            if obj is None: 
                continue
            for attr in ("current_path", "input_path", "video_path"):
                try:
                    probes.append(getattr(obj, attr, None))
                except Exception:
                    pass
        found = _first_existing(probes)
        if found:
            return found

        # 3) last resort: host ensure (may show a dialog)
        ensure = getattr(owner, "_ensure_input", None)
        if callable(ensure):
            try:
                ensured = ensure()
            except Exception:
                ensured = None
            if ensured:
                return _first_existing([ensured])

        return None
    except Exception:
        return None
from typing import Optional, Tuple

from PySide6.QtCore import QSettings
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QFileDialog, QLabel, QComboBox, QSpinBox
)

# --- Safe imports for shared paths/constants (mirrors tools_tab fallbacks) ---
try:
    from helpers.framevision_app import ROOT, OUT_FRAMES
    
    # New default for frames output
    OUT_FRAMES = ROOT/'output'/'frames'
except Exception:
    ROOT = Path('.').resolve()
    BASE = ROOT
    OUT_FRAMES = BASE/'output'/'video'
    OUT_FRAMES = BASE/'output'/'frames'

VIDEO_EXTS = (".mp4",".mov",".mkv",".avi",".m4v",".webm",".ts",".m2ts",".wmv",".mpg",".mpeg",".3gp",".3g2",".ogv")
IMAGE_EXTS = (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".gif")

# ---- binaries ----
def _which_binary(name: str) -> str:
    exe = f"{name}.exe" if os.name == "nt" else name
    candidates = [
        ROOT/"presets"/"bin"/exe,          # user's preferred location
        ROOT/"bin"/exe,                    # legacy
        Path("./presets/bin")/exe,         # relative fallback
        exe                                # hope in PATH
    ]
    for c in candidates:
        try:
            subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return name

def ffmpeg_path() -> str:
    return _which_binary("ffmpeg")

def ffprobe_path() -> str:
    # ffprobe may respond to "-version" or "-h"; be generous
    exe = f"ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidates = [
        ROOT/"presets"/"bin"/exe,
        ROOT/"bin"/exe,
        Path("./presets/bin")/exe,
        exe
    ]
    for c in candidates:
        try:
            subprocess.check_output([str(c), "-h"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return "ffprobe"

# ---------------- helpers ----------------
def _human_bytes(n: int) -> str:
    try:
        step = 1024.0
        units = ["B","KB","MB","GB","TB","PB"]
        i = 0
        x = float(n)
        while x >= step and i < len(units)-1:
            x /= step; i += 1
        return f"{x:.2f} {units[i]}"
    except Exception:
        return f"{n} B"

def _ensure_outdir(p: Path) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _set_owner_input(owner, path: Path) -> None:
    """Best-effort: stash selected path so downstream tools and the host can find it."""
    try:
        spath = str(path)
        # Fill common attributes on owner and main
        for obj in (owner, getattr(owner, "main", None)):
            if obj is None:
                continue
            try:
                setattr(obj, "current_path", spath)
                setattr(obj, "input_path", spath)
                setattr(obj, "video_path", spath)
            except Exception:
                pass

        # If owner already has _ensure_input, do NOT override it (respect host behavior)
        if not hasattr(owner, "_ensure_input"):
            owner._ensure_input = lambda: spath  # type: ignore[attr-defined]

        # Also keep a named attr for other tools
        owner._frames_loaded_path = path  # type: ignore[attr-defined]
    except Exception:
        pass

def _probe_media(path: Path) -> Tuple[float, int]:
    """Return (duration_sec, frame_count). duration=0 and frames=1 for images."""
    low = path.suffix.lower()
    if low in IMAGE_EXTS:
        return (0.0, 1)
    # Try ffprobe JSON
    try:
        cmd = [
            ffprobe_path(), "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(out.decode("utf-8", errors="ignore"))
        dur = 0.0
        try:
            dur = float(info.get("format", {}).get("duration") or 0.0)
        except Exception:
            dur = 0.0
        frames = 0
        # pick the first video stream
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams:
            vs = vstreams[0]
            if "nb_frames" in vs and str(vs["nb_frames"]).isdigit():
                frames = int(vs["nb_frames"])
            else:
                # derive from avg_frame_rate if present
                rate_str = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0/0"
                try:
                    if "/" in rate_str:
                        a, b = rate_str.split("/", 1)
                        a = float(a or 0.0); b = float(b or 1.0)
                        fps = a / b if b else 0.0
                    else:
                        fps = float(rate_str)
                except Exception:
                    fps = 0.0
                if dur and fps:
                    frames = int(round(dur * fps))
        return (dur, max(1, frames))
    except Exception:
        return (0.0, 0)


# ---------------- info line helper ----------------
def _set_info(owner, message: str):
    """Set the info line text under the top buttons; create if missing."""
    try:
        lbl = getattr(owner, "frames_info_label", None)
        if lbl is not None:
            lbl.setText(message)
    except Exception:
        pass

# ---------------- format helpers ----------------
def _selected_ext(owner) -> str:
    """Return 'png' or 'jpg' based on the UI combo; defaults to 'png'."""
    try:
        combo = getattr(owner, "frames_format_combo", None)
        if combo is not None:
            t = combo.currentText().strip().lower()
            return "jpg" if t in ("jpg","jpeg") else "png"
    except Exception:
        pass
    return "png"

def _codec_args_for_ext(ext: str):
    """Optional quality args for ffmpeg when saving single frames."""
    if ext == "jpg":
        return ["-q:v", "2"]
    return []
# ---------------- actions ----------------
def _open_frames_folder(owner):
    """Open the frames output folder in Media Explorer (fallback: OS file explorer)."""
    try:
        last_dir = getattr(owner, "_frames_last_dir", None)
    except Exception:
        last_dir = None
    base = Path(last_dir) if last_dir else OUT_FRAMES
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Prefer Media Explorer on main window if available
    main = None
    try:
        main = getattr(owner, "main", None)
    except Exception:
        main = None
    if main is None:
        try:
            main = owner.window() if hasattr(owner, "window") else None
        except Exception:
            main = None

    if main is not None and hasattr(main, "open_media_explorer_folder"):
        try:
            main.open_media_explorer_folder(str(base), preset="images", include_subfolders=False)
            _set_info(owner, f"Opened in Media Explorer: {base}")
            return
        except TypeError:
            try:
                main.open_media_explorer_folder(str(base))
                _set_info(owner, f"Opened in Media Explorer: {base}")
                return
            except Exception:
                pass
        except Exception:
            pass

    # Fallback: OS folder open
    try:
        sysname = platform.system().lower()
        if sysname.startswith("win"):
            os.startfile(str(base))  # type: ignore[attr-defined]
        elif sysname == "darwin":
            subprocess.Popen(["open", str(base)])
        else:
            subprocess.Popen(["xdg-open", str(base)])
        _set_info(owner, f"Opened folder: {base}")
    except Exception:
        _set_info(owner, f"Folder open failed: {base}")


def _run_last(owner):
    inp = _frames_get_input(owner)
    if not inp:
        QMessageBox.information(owner, "Extract", "No input selected."); return
    ext = _selected_ext(owner)
    out = OUT_FRAMES / f"{Path(inp).stem}_lastframe.{ext}"
    _ensure_outdir(out)
    setattr(owner, "_frames_last_dir", out.parent)
    cmd = [ffmpeg_path(), "-y", "-sseof", "-1", "-i", str(inp)] + _codec_args_for_ext(ext) + ["-update", "1", "-frames:v", "1", str(out)]
    getattr(owner, "_run", lambda *_: None)(cmd, out)
    _set_info(owner, f"Queued last-frame extraction → {out}")

def _run_all(owner):
    inp = _frames_get_input(owner)
    if not inp:
        QMessageBox.information(owner, "Extract", "No input selected."); return
    outdir = OUT_FRAMES / f"{Path(inp).stem}"
    outdir.mkdir(parents=True, exist_ok=True)
    setattr(owner, "_frames_last_dir", outdir)
    ext = _selected_ext(owner)
    out = outdir / f"frame_%06d.{ext}"
    # Build step-aware filter if needed
    try:
        step_val = int(getattr(owner, "frames_step_spin").value()) if hasattr(owner, "frames_step_spin") else 1
    except Exception:
        step_val = 1
    vf = []
    if step_val > 1:
        vf = [f"select='not(mod(n\\,{step_val}))',setpts=N/FRAME_RATE/TB"]
    cmd = [ffmpeg_path(), '-y', '-i', str(inp)]
    if vf:
        cmd += ['-vf', ','.join(vf), '-vsync', 'vfr']
    cmd += _codec_args_for_ext(ext)
    cmd += [str(out)]
    getattr(owner, "_run", lambda *_: None)(cmd, outdir)
    _set_info(owner, f"Queued all-frames extraction (every N={getattr(owner, 'frames_step_spin').value() if hasattr(owner, 'frames_step_spin') else 1}) → {outdir}")


def _join_frames(owner):
    """Pick a folder with image frames and join them into a video using ffmpeg."""
    # Remember last used folder
    try:
        settings = QSettings("FrameVision", "FrameVision")
    except Exception:
        settings = None
    start = ""
    if settings is not None:
        try:
            start = settings.value("frames/join_last_dir", "") or ""
        except Exception:
            start = ""
    if not start:
        start = str(OUT_FRAMES)

    try:
        folder = QFileDialog.getExistingDirectory(owner, "Select frames folder", start)
    except Exception:
        folder = ""
    if not folder:
        return

    if settings is not None:
        try:
            settings.setValue("frames/join_last_dir", folder)
        except Exception:
            pass

    p = Path(folder)
    # Detect which image extension is present
    candidates = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff"]
    pattern = None
    for glob_pat in candidates:
        try:
            if any(p.glob(glob_pat)):
                pattern = str(p / glob_pat)
                break
        except Exception:
            continue

    if not pattern:
        try:
            QMessageBox.warning(owner, "Join frames", "No image frames (png/jpg/webp/bmp/tif) were found in that folder.")
        except Exception:
            pass
        return

    # Determine FPS from UI spinbox if available
    fps_spin = getattr(owner, "frames_join_fps_spin", None)
    try:
        fps = int(fps_spin.value()) if fps_spin is not None else 30
    except Exception:
        fps = 30
    if fps < 1:
        fps = 1

    out = OUT_FRAMES / f"{p.name}_joined.mp4"
    _ensure_outdir(out)

    cmd = [
        ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out)
    ]

    runner = getattr(owner, "_run", None)
    if callable(runner):
        runner(cmd, out)
    else:
        try:
            subprocess.Popen(cmd)
        except Exception:
            pass

    _set_info(owner, f"Queued join-frames → video at {fps} fps → {out}")

def _get_current_frame(owner):
    """
    Grab the CURRENT frame from the media player or image preview, save to OUT_FRAMES.
    Mirrors meme_tool.py behavior for grabbing currentFrame / label.pixmap().
    """
    main = getattr(owner, "main", owner)
    # Best-effort: detect 'current path' to name file
    cur_path = None
    for attr in ("current_path", "current_image"):
        p = getattr(main, attr, None)
        if isinstance(p, str) and os.path.exists(p):
            cur_path = p; break
    vid = getattr(main, "video", None)
    pm = None
    # Prefer full-res QImage (e.g., video.currentFrame)
    try:
        qimg = getattr(vid, "currentFrame", None) if vid is not None else None
        if qimg is not None and (not hasattr(qimg, "isNull") or not qimg.isNull()):
            pm = QPixmap.fromImage(qimg)
    except Exception:
        pm = None
    # Fallback: whatever is on the label
    if pm is None and vid is not None and hasattr(vid, "label") and hasattr(vid.label, "pixmap"):
        try:
            ppm = vid.label.pixmap()
            if ppm is not None and (not hasattr(ppm, "isNull") or not ppm.isNull()):
                pm = ppm
        except Exception:
            pass
    # If still nothing and current path is an image, load it
    if pm is None and cur_path and str(cur_path).lower().endswith(IMAGE_EXTS):
        try:
            tmp = QPixmap(cur_path)
            if not tmp.isNull():
                pm = tmp
        except Exception:
            pass
    if pm is None or pm.isNull():
        QMessageBox.warning(owner, "Get current frame", "Couldn't capture the current frame."); return
    stem = Path(cur_path).stem if cur_path else "frame"
    ext = _selected_ext(owner)
    out = OUT_FRAMES / f"{stem}_current.{ext}"
    _ensure_outdir(out)
    try:
        fmt = _selected_ext(owner).upper()
        ok = pm.save(str(out), "JPG" if fmt.startswith("JP") else "PNG")
        if not ok:
            raise RuntimeError("save failed")
        _set_info(owner, f"Saved current frame: {out}")
    except Exception:
        _set_info(owner, "Couldn't save image.")

def _on_batch(owner):
    """Use helpers/batch.py to create a batch of extract-all-frames jobs for the queue."""
    try:
        from helpers.batch import BatchSelectDialog
    except Exception:
        QMessageBox.warning(owner, "Batch", "BatchSelectDialog is unavailable."); return
    settings = QSettings("FrameVision", "FrameVision")
    start = settings.value("frames/last_dir", "") or ""
    files, conflict = BatchSelectDialog.pick(owner, title="Batch extract frames", exts=BatchSelectDialog.VIDEO_EXTS, start_dir=start)
    if files is None:
        return  # cancelled
    if files:
        try:
            settings.setValue("frames/last_dir", os.path.dirname(files[0]))
        except Exception:
            pass
    for f in files:
        fpath = Path(f)
        outdir = OUT_FRAMES / f"{fpath.stem}"
        if conflict == "skip" and outdir.exists():
            continue
        if conflict == "version":
            base = outdir
            i = 1
            while outdir.exists():
                outdir = Path(str(base) + f"_{i:02d}")
                i += 1
        outdir.mkdir(parents=True, exist_ok=True)
        setattr(owner, "_frames_last_dir", outdir)
        ext = _selected_ext(owner)
        out = outdir / f"frame_%06d.{ext}"
        try:
            step_val = int(getattr(owner, "frames_step_spin").value()) if hasattr(owner, "frames_step_spin") else 1
        except Exception:
            step_val = 1
        vf = []
        if step_val > 1:
            vf = [f"select='not(mod(n\\,{step_val}))',setpts=N/FRAME_RATE/TB"]
        cmd = [ffmpeg_path(), '-y', '-i', str(fpath)]
        if vf:
            cmd += ['-vf', ','.join(vf), '-vsync', 'vfr']
        cmd += _codec_args_for_ext(ext)
        cmd += [str(out)]
        getattr(owner, "_run", lambda *_: None)(cmd, outdir)
    _set_info(owner, f"Queued all-frames extraction (every N={getattr(owner, 'frames_step_spin').value() if hasattr(owner, 'frames_step_spin') else 1}) → {outdir}")

def _on_load(owner):
    """Let the user pick a video; remember last dir; then show info (name, size, duration, frames)."""
    settings = QSettings("FrameVision", "FrameVision")
    start = settings.value("frames/last_dir", "") or ""
    filt = "Video files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts *.m2ts *.wmv *.mpg *.mpeg *.3gp *.3g2 *.ogv);;All files (*)"
    path, _ = QFileDialog.getOpenFileName(owner, "Load video", start, filt)
    if not path:
        return
    try:
        settings.setValue("frames/last_dir", os.path.dirname(path))
    except Exception:
        pass
    p = Path(path)
    _set_owner_input(owner, p)
    size = p.stat().st_size if p.exists() else 0
    dur, frames = _probe_media(p)
    # Pretty duration
    def _fmt_dur(s: float) -> str:
        if s <= 0:
            return "0.00 s"
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s - (h*3600 + m*60)
        if h:
            return f"{h:d}:{m:02d}:{sec:05.2f}"
        return f"{m:d}:{sec:05.2f}" if m else f"{sec:.2f} s"
    msg = (
        f"Name: {p.name}\n"
        f"Size: {_human_bytes(size)}\n"
        f"Duration: {_fmt_dur(dur)}\n"
        f"Frames: {frames:,d}"
    )
    _set_info(owner, msg)

# ---------------- UI installer ----------------
def install_frames_tool(owner, section_widget):
    """Installs the 'Extract frames' UI into the provided CollapsibleSection."""
    lay_ext = QVBoxLayout()

    # Row with the three requested buttons (same line)
    row_top = QHBoxLayout()
    btn_get = QPushButton("Get current frame")
    btn_batch = QPushButton("Batch")
    btn_load = QPushButton("Load video")
    btn_open = QPushButton("Open folder")
    fmt_combo = QComboBox()
    fmt_combo.addItems(["PNG", "JPG"])
    fmt_combo.setToolTip("Choose output image format for frames")
    step_label = QLabel("Every :")
    step_spin = QSpinBox()
    step_spin.setRange(1, 500)
    step_spin.setValue(1)
    step_spin.setToolTip("Save one frame every ... frames (1 = every frame)")
    row_top.addWidget(btn_get); row_top.addWidget(btn_batch); row_top.addWidget(btn_load); row_top.addWidget(btn_open); row_top.addWidget(fmt_combo); row_top.addWidget(step_label); row_top.addWidget(step_spin); row_top.addStretch(1)
    lay_ext.addLayout(row_top)
    # Persist selected format (PNG/JPG) via QSettings
    settings = QSettings("FrameVision", "FrameVision")
    saved_fmt = str(settings.value("frames/format", "png")).lower()
    if saved_fmt not in ("png","jpg","jpeg"):
        saved_fmt = "png"
    fmt_combo.setCurrentText("JPG" if saved_fmt.startswith("jp") else "PNG")
    def _on_fmt_change(t):
        settings.setValue("frames/format", t.lower())
    fmt_combo.currentTextChanged.connect(_on_fmt_change)
    # Persist frame step (Every Nth)
    saved_step = int(settings.value("frames/step", 1)) if settings.value("frames/step", None) is not None else 1
    if saved_step < 1:
        saved_step = 1
    step_spin.setValue(saved_step)
    def _on_step_change(v):
        settings.setValue("frames/step", int(v))
    step_spin.valueChanged.connect(_on_step_change)



    # Join frames → video row
    row_join = QHBoxLayout()
    btn_join = QPushButton("Join frames to video")
    join_fps_label = QLabel("FPS:")
    join_fps_spin = QSpinBox()
    join_fps_spin.setRange(1, 240)
    join_fps_spin.setToolTip("Playback framerate for the joined video")
    row_join.addWidget(btn_join)
    row_join.addWidget(join_fps_label)
    row_join.addWidget(join_fps_spin)
    row_join.addStretch(1)
    lay_ext.addLayout(row_join)

    # Persist join FPS
    saved_join_fps = settings.value("frames/join_fps", 30)
    try:
        saved_join_fps = int(saved_join_fps)
    except Exception:
        saved_join_fps = 25
    if saved_join_fps < 1:
        saved_join_fps = 25
    join_fps_spin.setValue(saved_join_fps)
    def _on_join_fps_change(v):
        settings.setValue("frames/join_fps", int(v))
    join_fps_spin.valueChanged.connect(_on_join_fps_change)

    # Info line just under the buttons
    info_row = QHBoxLayout()
    info_label = QLabel("")
    info_label.setObjectName("frames_info_label")
    info_label.setWordWrap(True)
    info_row.addWidget(info_label)
    lay_ext.addLayout(info_row)

    # Existing extract buttons
    btn_last = QPushButton("Extract Last Frame")
    btn_all = QPushButton("Extract All Frames")
    btn_all.setToolTip("Export every frame to images. Large output!")
    lay_ext.addWidget(btn_last)
    lay_ext.addWidget(btn_all)

    # Preset row (hidden preset buttons)
    row = QHBoxLayout()
    btn_se = QPushButton("Save preset")
    btn_le = QPushButton("Load preset")
    row.addWidget(btn_se)
    row.addWidget(btn_le)
    lay_ext.addLayout(row)
    # Hide preset buttons from UI (keep functionality available for future use)
    btn_se.hide()
    btn_le.hide()

    section_widget.setContentLayout(lay_ext)

    # Wire actions
    btn_get.clicked.connect(lambda: _get_current_frame(owner))
    btn_batch.clicked.connect(lambda: _on_batch(owner))
    btn_load.clicked.connect(lambda: _on_load(owner))
    btn_last.clicked.connect(lambda: _run_last(owner))
    btn_all.clicked.connect(lambda: _run_all(owner))
    btn_join.clicked.connect(lambda: _join_frames(owner))
    btn_se.clicked.connect(lambda: _save_preset(owner))
    btn_le.clicked.connect(lambda: _load_preset(owner))
    btn_open.clicked.connect(lambda: _open_frames_folder(owner))

    # Optionally expose buttons on owner (back-compat)
    try:
        owner.btn_last = btn_last
        owner.btn_all = btn_all
        owner.btn_get_current_frame = btn_get
        owner.btn_batch_extract = btn_batch
        owner.btn_load_video = btn_load
        owner.btn_open_frames = btn_open
        owner.btn_join_frames = btn_join
        owner.frames_format_combo = fmt_combo
        owner.frames_info_label = info_label
        owner.frames_step_spin = step_spin
        owner.frames_join_fps_spin = join_fps_spin
    except Exception:
        pass

# ---------------- preset helpers (unchanged) ----------------
def _save_preset(owner):
    # Prefer owner's preset helpers if available
    chooser = getattr(owner, "_choose_save_path", None)
    if chooser is None:
        # Minimal fallback
        try:
            pth, _ = QFileDialog.getSaveFileName(owner, "Save preset", str(ROOT/"presets"/"Tools"/"extract_preset.json"), "FrameVision Preset (*.json)")
            if not pth:
                return
            p = Path(pth)
        except Exception:
            return
    else:
        p = chooser("extract_preset.json")
        if not p:
            return
    try:
        data = {"tool":"extract"}
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            QMessageBox.information(owner, "Preset saved", str(p))
        except Exception:
            pass
    except Exception:
        pass

def _load_preset(owner):
    chooser = getattr(owner, "_choose_open_path", None)
    if chooser is None:
        try:
            pth, _ = QFileDialog.getOpenFileName(owner, "Load preset", str(ROOT/"presets"/"Tools"), "FrameVision Preset (*.json)")
            if not pth:
                return
            p = Path(pth)
        except Exception:
            return
    else:
        p = chooser()
        if not p:
            return
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if data.get("tool") != "extract":
            raise ValueError("Wrong preset type")
        try:
            QMessageBox.information(owner, "Preset", "Loaded extract preset (no parameters).")
        except Exception:
            pass
    except Exception as e:
        try:
            QMessageBox.critical(owner, "Preset error", str(e))
        except Exception:
            pass
