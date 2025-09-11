# helpers/audiotool.py — Audio tool extracted from Tools tab
import os, re, subprocess
from pathlib import Path
from PySide6.QtWidgets import (QLineEdit, QToolButton, QCheckBox, QPushButton,
                               QFormLayout, QHBoxLayout, QFileDialog, QMessageBox)

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS
except Exception:
    from pathlib import Path as _Path
    ROOT = _Path('.').resolve()
    BASE = ROOT
    OUT_VIDEOS = BASE/'output'/'video'

def ffmpeg_path():
    candidates = [ROOT/'bin'/('ffmpeg.exe' if os.name=='nt' else 'ffmpeg'), 'ffmpeg']
    for c in candidates:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return 'ffmpeg'

def install_audio_tool(pane, sec_audio):
    """Install the Audio UI/logic into an existing CollapsibleSection.
    - pane: the InstantToolsPane instance (provides _ensure_input and _run)
    - sec_audio: the CollapsibleSection to populate
    """
    edit_audio = QLineEdit()
    btn_pick_audio = QToolButton(); btn_pick_audio.setText("…")
    cb_mix = QCheckBox("Mix with original audio (instead of replacing)")
    btn_audio = QPushButton("Add Audio to Video")

    lay_audio = QFormLayout()
    row = QHBoxLayout(); row.addWidget(edit_audio); row.addWidget(btn_pick_audio)
    lay_audio.addRow("Audio file", row)
    lay_audio.addRow(cb_mix)
    lay_audio.addRow(btn_audio)
    sec_audio.setContentLayout(lay_audio)

    def _pick_audio_file():
        path, _ = QFileDialog.getOpenFileName(pane, "Choose audio file...", "", 
                    "Audio files (*.mp3 *.wav *.m4a *.aac *.flac);;All files (*)")
        if path:
            edit_audio.setText(path)

    def run_add_audio():
        inp = pane._ensure_input()
        if not inp:
            return
        audio = (edit_audio.text().strip() if edit_audio.text() else "")
        if not audio or not os.path.isfile(audio):
            try:
                QMessageBox.warning(pane, "Add Audio", "Please choose a valid audio file.")
            except Exception:
                pass
            return
        try:
            out_dir = OUT_VIDEOS
        except Exception:
            out_dir = Path('.')
        out = out_dir / f"{inp.stem}_withaudio.mp4"
        replace = not bool(cb_mix.isChecked())
        if replace:
            cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-i", str(audio),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
        else:
            cmd = [ffmpeg_path(), "-y", "-i", str(inp), "-i", str(audio),
                   "-filter_complex", "[0:a]volume=1.0[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=shortest[aout]",
                   "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
        pane._run(cmd, out)

    btn_pick_audio.clicked.connect(_pick_audio_file)
    btn_audio.clicked.connect(run_add_audio)

    # Expose widgets on pane for snapshot/restore and potential external use (optional)
    try:
        pane.edit_audio = edit_audio
        pane.btn_pick_audio = btn_pick_audio
        pane.cb_mix = cb_mix
        pane.btn_audio = btn_audio
    except Exception:
        pass
