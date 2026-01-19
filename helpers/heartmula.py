"""HeartMuLa (heartlib) PySide6 UI

Drop this file into:  helpers/heartmula.py
Settings saved to:   presets/setsave/mula.json
Outputs saved to:    output/music/heartmula/

This UI calls the upstream example script:
  models/HeartMuLa/_heartlib_src/examples/run_music_generation.py

The one-click installer (presets/extra_env/mula_install.bat) downloads
heartlib and the required checkpoints into the expected folders.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QUrl, QTime, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QDoubleSpinBox,
    QSpinBox,
    QTimeEdit,
    QComboBox,
    QVBoxLayout,
    QTabWidget,
    QWidget,
)


def _ms_to_time(ms: int) -> QTime:
    """Convert milliseconds to a QTime (hh:mm:ss)."""
    total_seconds = max(1, int(round(ms / 1000.0)))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return QTime(h % 24, m, s)


def _time_to_ms(t: QTime) -> int:
    """Convert a QTime (hh:mm:ss) to milliseconds."""
    return int((t.hour() * 3600 + t.minute() * 60 + t.second()) * 1000)


def _root_dir() -> Path:
    # helpers/heartmula.py -> helpers -> root
    return Path(__file__).resolve().parent.parent


def _settings_path() -> Path:
    return _root_dir() / "presets" / "setsave" / "mula.json"


def _presets_path() -> Path:
    return _root_dir() / "presets" / "setsave" / "mulapresets.json"


def _ensure_dirs() -> None:
    root = _root_dir()
    (root / ".mula_env").mkdir(parents=True, exist_ok=True)
    (root / "models" / "HeartMuLa").mkdir(parents=True, exist_ok=True)
    (root / "output" / "music" / "heartmula").mkdir(parents=True, exist_ok=True)
    (root / "presets" / "setsave").mkdir(parents=True, exist_ok=True)


def _slugify(text: str, max_len: int = 42) -> str:
    """Make a filesystem-friendly slug (lowercase, underscores)."""
    s = (text or '').strip().lower()
    # normalize common dash variants
    s = s.replace('–', '-').replace('—', '-')
    # keep alnum, convert everything else to underscores
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip('_')
    return s


def _default_mula_presets() -> dict:
    """Default presets written to mulapresets.json the first time."""
    return {
        "version": 1,
        "genres": [
            {
                "name": "EDM",
                "presets": [
                    {
                        "name": "House – Club",
                        "tags": "club house, 128 bpm, four-on-the-floor kick, offbeat open hi-hat, rolling bassline, sidechain compression, bright supersaw, riser, snare build, drop, festival energy, wide stereo",
                    },
                    {
                        "name": "Tech House",
                        "tags": "tech house, 128 bpm, punchy kick and snare, groovy bass, syncopated percussion, shaker loop, minimal vocals, tension build, big drop, club mix",
                    },
                    {
                        "name": "Melodic Techno",
                        "tags": "melodic techno, 126 bpm, driving kick, hypnotic arp, dark atmosphere, long buildup, impact hit, drop, cinematic pads",
                    },
                    {
                        "name": "Big Room",
                        "tags": "big room, 130 bpm, huge kick, snare roll buildup, white noise riser, supersaw lead, massive drop, festival",
                    },
                    {
                        "name": "Drum & Bass – Dancefloor",
                        "tags": "drum and bass, 174 bpm, fast breakbeat, punchy kick and snare, rolling sub bass, reese bass, high energy, tension build, riser, impact hit, big drop, energetic synth stabs, atmospheric pad, DJ friendly, loop-based, minimal vocal chops",
                    },
                    {
                        "name": "Drum & Bass – Neurofunk",
                        "tags": "neurofunk, drum and bass, 172 bpm, tight snare, aggressive reese bass, growl bass, syncopated drums, gritty texture, dark atmosphere, industrial sound design, long buildup, snare roll, heavy drop, bass variation, minimal vocals, club mix, hard hitting",
                    },
                    {
                        "name": "Drum & Bass – Liquid",
                        "tags": "liquid drum and bass, 170 bpm, crisp breakbeat, warm sub bass, airy pads, emotional chords, clean mix, gentle riser, smooth drop, melodic lead, spacious reverb, minimal vocals, DJ friendly intro and outro",
                    },
                    {
                        "name": "Techno – Driving Warehouse",
                        "tags": "techno, 130 bpm, four-on-the-floor kick, rumbling bass, rolling percussion, offbeat hi-hat, hypnotic loop, dark warehouse, minimal vocals, repetitive groove, long buildup, tension, impact hit, drop, DJ friendly",
                    },
                    {
                        "name": "Techno – Peak Time",
                        "tags": "peak time techno, 132 bpm, powerful kick, driving bassline, big synth stab, build-up with snare roll, white noise riser, huge drop, energetic percussion, club mix, loop-based, crowd energy",
                    },
                    {
                        "name": "Techno – Melodic",
                        "tags": "melodic techno, 126 bpm, driving kick, arpeggiated synth, deep bass, cinematic pad, gradual buildup, impact hit, drop, wide stereo, minimal vocal chops, DJ friendly",
                    },
                    {
                        "name": "Hard Techno – Rave",
                        "tags": "hard techno, 150 bpm, hard four-on-the-floor kick, distorted rumble bass, aggressive percussion, rave stabs, intense energy, fast hats, build-up, snare roll, white noise riser, brutal drop, warehouse rave, relentless",
                    },
                    {
                        "name": "Hard Techno – Industrial",
                        "tags": "industrial hard techno, 145 bpm, heavy distorted kick, metallic percussion, dark atmosphere, gritty texture, pounding groove, relentless drive, tension build, impact hit, drop, harsh synth, underground",
                    },
                    {
                        "name": "Hard Techno – Hardgroove",
                        "tags": "hardgroove techno, 145 bpm, hard kick, groovy tom percussion, syncopated loops, fast hats, tribal percussion, hypnotic repetition, long buildup, drop, rave energy, DJ friendly",
                    },
                ],
            },
            {
                "name": "Rock",
                "presets": [
                    {
                        "name": "Rock – Classic",
                        "tags": "classic rock, 120 bpm, live drum kit, steady rock groove, crunchy rhythm guitars, melodic lead guitar, electric bass, verse chorus structure, big chorus, warm analog mix, arena feel",
                    },
                    {
                        "name": "Rock – Hard Rock",
                        "tags": "hard rock, 140 bpm, driving drums, punchy kick and snare, distorted power chords, palm-muted riffs, big chorus, guitar solo, gritty vocals, aggressive energy, modern rock mix, wide guitars",
                    },
                    {
                        "name": "Rock – Metal",
                        "tags": "heavy metal, 160 bpm, double kick, tight snare, fast riffs, palm-muted chugs, aggressive guitar tone, heavy bass, breakdown, solo section, intense vocals, raw power",
                    },
                ],
            },
            {
                "name": "Pop",
                "presets": [
                    {
                        "name": "Pop – Modern",
                        "tags": "modern pop, 118 bpm, clean punchy drums, bright synths, catchy hook, verse chorus structure, big chorus, polished vocal production, radio-ready mix, uplifting mood, wide stereo",
                    },
                    {
                        "name": "Pop – Synthpop",
                        "tags": "synthpop, 112 bpm, retro drum machine, gated snare, warm analog synths, arpeggiator, catchy melody, dreamy chords, smooth vocals, nostalgic 80s vibe, clean mix",
                    },
                    {
                        "name": "Pop – Dance Pop",
                        "tags": "dance pop, 124 bpm, four-on-the-floor kick, offbeat open hat, bright chords, sidechain compression, vocal hook, pre-chorus build, drop-style chorus, club-friendly, glossy mix",
                    },
                ],
            },
            {
                "name": "R&B",
                "presets": [
                    {
                        "name": "R&B – Contemporary",
                        "tags": "contemporary r&b, 92 bpm, smooth drums, deep sub bass, lush chords, soulful vocals, modern vocal layers, relaxed groove, intimate vibe, polished mix",
                    },
                    {
                        "name": "R&B – Neo Soul",
                        "tags": "neo soul, 85 bpm, swung groove, warm electric piano, jazzy chords, live bass feel, crisp snare, intimate vocals, laid-back pocket, organic texture, smooth mix",
                    },
                    {
                        "name": "R&B – Trap Soul",
                        "tags": "trap soul, 70 bpm, trap hats, 808 bass, minimal chords, airy pads, moody vibe, melodic vocals, emotional hook, sparse arrangement, late-night atmosphere",
                    },
                ],
            },
            {
                "name": "Rap",
                "presets": [
                    {
                        "name": "Rap – Boom Bap",
                        "tags": "boom bap rap, 92 bpm, classic hip hop drums, punchy kick, snappy snare, sampled vibe, chopped loop, gritty texture, head-nod groove, rap-focused, raw mix",
                    },
                    {
                        "name": "Rap – Trap",
                        "tags": "trap rap, 140 bpm, rapid hi-hats, 808 bass, hard snare, dark synth melody, aggressive energy, hook section, beat switch, modern mix, rap-focused",
                    },
                    {
                        "name": "Rap – Drill",
                        "tags": "drill rap, 145 bpm, sliding 808, sharp snare, syncopated hi-hats, dark piano or bell melody, tense vibe, aggressive rhythm, rap-forward, hard mix",
                    },
                ],
            },
            {
                "name": "Ballad / Slow",
                "presets": [
                    {
                        "name": "Ballad – Piano",
                        "tags": "piano ballad, 72 bpm, emotional piano chords, soft drums, gentle strings, intimate vocals, big chorus lift, gradual build, heartfelt mood, warm reverb",
                    },
                    {
                        "name": "Ballad – Acoustic",
                        "tags": "acoustic ballad, 78 bpm, acoustic guitar fingerpicking, soft percussion, warm bass, intimate vocal, emotional chorus, natural room sound, organic performance",
                    },
                    {
                        "name": "Ballad – Cinematic",
                        "tags": "cinematic ballad, 68 bpm, orchestral strings, piano, soft drums, emotional swells, dramatic build, big climax, film score feel, lush reverb",
                    },
                ],
            },
            {
                "name": "Instrumental",
                "presets": [
                    {
                        "name": "Instrumental – Ambient",
                        "tags": "ambient instrumental, 70 bpm, evolving pads, soft textures, spacious reverb, slow movement, atmospheric drones, minimal percussion, calming mood, cinematic soundscape, no vocals",
                    },
                    {
                        "name": "Instrumental – Lo-fi",
                        "tags": "lofi instrumental, 82 bpm, dusty drums, vinyl crackle texture, jazzy chords, warm bass, relaxed groove, soft melody, chill mood, loop-based, no vocals",
                    },
                    {
                        "name": "Instrumental – Orchestral",
                        "tags": "orchestral instrumental, 90 bpm, strings brass woodwinds, cinematic percussion, emotional theme, dynamic build, heroic climax, film score style, wide stereo, no vocals",
                    },
                ],
            },
            {
                "name": "Reggae",
                "presets": [
                    {
                        "name": "Reggae – Roots",
                        "tags": "roots reggae, 74 bpm, one drop rhythm, skank guitar on offbeat, warm bassline, live drums, relaxed groove, sunny vibe, soulful vocals, organic mix",
                    },
                    {
                        "name": "Reggae – Dancehall",
                        "tags": "dancehall, 96 bpm, modern drum pattern, heavy bass, syncopated rhythm, energetic vibe, catchy chant hook, club-ready, bright synth accents, punchy mix",
                    },
                    {
                        "name": "Reggae – Dub",
                        "tags": "dub reggae, 72 bpm, deep bass, sparse drums, skank guitar, heavy reverb and delay, echo effects, spacey atmosphere, minimalist groove, instrumental focus",
                    },
                ],
            },
            {"name": "Custom", "presets": []},
        ],
    }


class MulaPresetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {}

    def load_or_create(self) -> dict:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(self.data, dict) and "genres" in self.data:
                    return self.data
            except Exception:
                pass
        self.data = _default_mula_presets()
        self.save()
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class MulaSettings:
    model_path: str = "models/HeartMuLa"
    version: str = "3B"
    max_audio_length_ms: int = 240000
    topk: int = 50
    temperature: float = 1.0
    cfg_scale: float = 1.5
    lyrics_text: str = "[Verse]\nWrite your lyrics here...\n"
    tags_text: str = "piano,happy,wedding,synthesizer,romantic"
    output_dir: str = "output/music/heartmula"

    @staticmethod
    def load(path: Path) -> "MulaSettings":
        if not path.exists():
            return MulaSettings()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return MulaSettings()
        s = MulaSettings()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_path": self.model_path,
            "version": self.version,
            "max_audio_length_ms": int(self.max_audio_length_ms),
            "topk": int(self.topk),
            "temperature": float(self.temperature),
            "cfg_scale": float(self.cfg_scale),
            "lyrics_text": self.lyrics_text,
            "tags_text": self.tags_text,
            "output_dir": self.output_dir,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class HeartMuLaUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        _ensure_dirs()

        self.setWindowTitle("HeartMuLa (offline) – Music Generator")
        self.setMinimumWidth(920)

        self.settings_path = _settings_path()
        self.settings = MulaSettings.load(self.settings_path)

        # --- Presets ---
        self.presets_path = _presets_path()
        self.preset_store = MulaPresetStore(self.presets_path)
        self.preset_data = self.preset_store.load_or_create()

        # --- Process runner ---
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)

        # --- Widgets ---
        self.model_path = QLineEdit(self.settings.model_path)
        self.btn_browse_model = QPushButton("Browse…")
        self.btn_browse_model.clicked.connect(self._browse_model)

        self.version = QComboBox()
        self.version.addItems(["3B"])
        self.version.setCurrentText(self.settings.version if self.settings.version else "3B")

        # Length is stored as milliseconds for the upstream script, but shown as mm:ss for humans.
        self.max_len = QTimeEdit()
        self.max_len.setDisplayFormat("mm:ss")
        self.max_len.setMinimumTime(_ms_to_time(1_000))
        self.max_len.setMaximumTime(_ms_to_time(1_200_000))
        self.max_len.setTime(_ms_to_time(int(self.settings.max_audio_length_ms)))

        self.topk = QSpinBox()
        self.topk.setRange(1, 200)
        self.topk.setValue(int(self.settings.topk))

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 3.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.temperature.setValue(float(self.settings.temperature))

        self.cfg_scale = QDoubleSpinBox()
        self.cfg_scale.setRange(0.0, 10.0)
        self.cfg_scale.setSingleStep(0.1)
        self.cfg_scale.setDecimals(2)
        self.cfg_scale.setValue(float(self.settings.cfg_scale))

        self.output_dir = QLineEdit(self.settings.output_dir)
        self.btn_browse_out = QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_output)

        self.output_name = QLineEdit("")
        self.output_name.setPlaceholderText("Optional filename (blank = auto preset + timestamp)")

        self.lyrics = QPlainTextEdit(self.settings.lyrics_text)
        self.lyrics.setPlaceholderText("Paste lyrics here. Use section headers like [Verse], [Chorus], etc.")

        self.tags = QLineEdit(self.settings.tags_text)
        self.tags.setPlaceholderText("Comma-separated tags, no spaces. Example: piano,happy,wedding,synthesizer,romantic")

        # Preset UI
        self.genre_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.btn_preset_add = QPushButton("Add")
        self.btn_preset_remove = QPushButton("Remove")

        self.genre_combo.currentIndexChanged.connect(self._on_genre_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.btn_preset_add.clicked.connect(self._add_preset)
        self.btn_preset_remove.clicked.connect(self._remove_preset)

        self.btn_generate = QPushButton("Generate music")
        self.btn_generate.clicked.connect(self._generate)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)

        self.btn_open_out = QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(self._open_output_folder)

        self.btn_save = QPushButton("Save settings")
        self.btn_save.clicked.connect(self._save_settings)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10_000)

        # --- Layout ---
        main = QVBoxLayout(self)

        cfg = QGroupBox("Settings")
        form = QFormLayout(cfg)

        row_model = QHBoxLayout()
        row_model.addWidget(self.model_path, 1)
        row_model.addWidget(self.btn_browse_model)
        form.addRow("Model folder", row_model)

        form.addRow("Model version", self.version)
        form.addRow("Length (mm:ss)", self.max_len)
        form.addRow("Top-k", self.topk)
        form.addRow("Temperature", self.temperature)
        form.addRow("CFG scale", self.cfg_scale)

        row_out = QHBoxLayout()
        row_out.addWidget(self.output_dir, 1)
        row_out.addWidget(self.btn_browse_out)
        form.addRow("Output folder", row_out)

        form.addRow("Output filename", self.output_name)

        main.addWidget(cfg)

        main.addWidget(QLabel("Lyrics"))
        main.addWidget(self.lyrics, 2)

        tags_row = QHBoxLayout()
        tags_row.addWidget(QLabel("Tags"))
        tags_row.addWidget(self.tags, 1)
        main.addLayout(tags_row)

        presets_box = QGroupBox("Presets")
        presets_form = QFormLayout(presets_box)
        self.genre_combo.addItem("All")
        for g in self.preset_data.get("genres", []):
            name = str(g.get("name", "")).strip()
            if name and name != "All":
                self.genre_combo.addItem(name)
        presets_form.addRow("Genre", self.genre_combo)

        row_p = QHBoxLayout()
        row_p.addWidget(self.preset_combo, 1)
        row_p.addWidget(self.btn_preset_add)
        row_p.addWidget(self.btn_preset_remove)
        presets_form.addRow("Preset", row_p)

        main.addWidget(presets_box)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_open_out)
        main.addLayout(btns)

        main.addWidget(QLabel("Log"))
        main.addWidget(self.log, 1)

        # Build the preset dropdown contents.
        self._on_genre_changed()

        self._append_log("Ready. If this is your first time, run presets/extra_env/mula_install.bat")

    # --------------------- helpers ---------------------

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _show_toast(self, text: str, *, error: bool = False, ms: int = 2200) -> None:
        """Non-blocking toast message (prefers main-window toast if available)."""
        # Prefer any existing app-level toast API.
        try:
            win = self.window()
            for name in (
                "show_toast",
                "toast",
                "push_toast",
                "notify_toast",
                "notify",
            ):
                fn = getattr(win, name, None)
                if callable(fn):
                    try:
                        fn(text)
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # Fallback: lightweight in-widget toast label.
        try:
            lbl = getattr(self, "_toast_lbl", None)
            if lbl is None:
                lbl = QLabel(self)
                lbl.setObjectName("heartmulaToast")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setWordWrap(True)
                lbl.setContentsMargins(12, 8, 12, 8)
                # No theme assumptions: keep it readable on both light/dark.
                lbl.setStyleSheet(
                    "#heartmulaToast {"
                    " color: white;"
                    " border-radius: 10px;"
                    " padding: 8px 12px;"
                    " background: rgba(40, 40, 40, 220);"
                    "}"
                    "#heartmulaToast[err='1'] {"
                    " background: rgba(140, 40, 40, 230);"
                    "}"
                )
                lbl.hide()
                self._toast_lbl = lbl

            try:
                lbl.setProperty("err", "1" if error else "0")
                # Force stylesheet refresh
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)
            except Exception:
                pass

            lbl.setText(str(text))

            # Size and position (bottom-center, inside this widget).
            max_w = max(260, min(int(self.width() * 0.92), 560))
            lbl.setMaximumWidth(max_w)
            lbl.adjustSize()
            w = lbl.width()
            h = lbl.height()
            x = max(10, int((self.width() - w) / 2))
            y = max(10, int(self.height() - h - 18))
            lbl.move(x, y)
            lbl.raise_()
            lbl.show()

            # Hide after a short delay.
            try:
                t = getattr(self, "_toast_timer", None)
                if t is None:
                    t = QTimer(self)
                    t.setSingleShot(True)
                    t.timeout.connect(lambda: getattr(self, "_toast_lbl", None) and self._toast_lbl.hide())
                    self._toast_timer = t
                t.stop()
                t.start(max(800, int(ms)))
            except Exception:
                pass
        except Exception:
            # Last resort: do nothing (no blocking dialogs).
            return

    def _browse_model(self) -> None:
        root = _root_dir()
        start = (root / self.model_path.text()).resolve() if self.model_path.text() else (root / "models" / "HeartMuLa")
        d = QFileDialog.getExistingDirectory(self, "Select model folder", str(start))
        if d:
            self.model_path.setText(os.path.relpath(d, str(root)))

    def _browse_output(self) -> None:
        root = _root_dir()
        start = (root / self.output_dir.text()).resolve() if self.output_dir.text() else (root / "output" / "music" / "heartmula")
        d = QFileDialog.getExistingDirectory(self, "Select output folder", str(start))
        if d:
            self.output_dir.setText(os.path.relpath(d, str(root)))

    def _open_output_folder(self) -> None:
        out_dir = (_root_dir() / self.output_dir.text()).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))


    def _auto_name_slug(self) -> str:
        """Best-effort slug for the auto filename based on preset/genre."""
        # Prefer selected preset (sub-genre / style).
        try:
            data = self.preset_combo.currentData(Qt.UserRole)
            if isinstance(data, dict):
                pname = str(data.get('name', '')).strip()
                if pname:
                    return _slugify(pname)
        except Exception:
            pass

        # Fallback to selected genre (if not All).
        try:
            g = (self.genre_combo.currentText() or '').strip()
            if g and g.lower() != 'all':
                return _slugify(g)
        except Exception:
            pass

        return ''


    # --------------------- presets ---------------------

    def _get_genre(self, name: str) -> dict | None:
        for g in self.preset_data.get("genres", []):
            if str(g.get("name", "")).strip() == name:
                return g
        return None

    def _rebuild_preset_combo(self) -> None:
        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem("Select preset…")
            selected_genre = self.genre_combo.currentText().strip() or "All"

            def add_preset_item(genre_name: str, preset: dict, prefix_genre: bool) -> None:
                pname = str(preset.get("name", "")).strip() or "(unnamed)"
                label = f"{genre_name} • {pname}" if prefix_genre else pname
                payload = {
                    "genre": genre_name,
                    "name": pname,
                    "tags": str(preset.get("tags", "")),
                }
                self.preset_combo.addItem(label)
                self.preset_combo.setItemData(self.preset_combo.count() - 1, payload, Qt.UserRole)

            if selected_genre == "All":
                first_group = True
                for g in self.preset_data.get("genres", []):
                    gname = str(g.get("name", "")).strip()
                    presets = g.get("presets", []) if isinstance(g.get("presets", []), list) else []
                    if not gname or not presets:
                        continue
                    if not first_group:
                        self.preset_combo.insertSeparator(self.preset_combo.count())
                    first_group = False
                    for p in presets:
                        if isinstance(p, dict):
                            add_preset_item(gname, p, prefix_genre=True)
            else:
                g = self._get_genre(selected_genre)
                presets = g.get("presets", []) if isinstance(g, dict) else []
                if isinstance(presets, list):
                    for p in presets:
                        if isinstance(p, dict):
                            add_preset_item(selected_genre, p, prefix_genre=False)
        finally:
            self.preset_combo.blockSignals(False)

        self.preset_combo.setCurrentIndex(0)
        self.btn_preset_remove.setEnabled(False)

    def _on_genre_changed(self) -> None:
        self._rebuild_preset_combo()

    def _on_preset_changed(self) -> None:
        data = self.preset_combo.currentData(Qt.UserRole)
        if not isinstance(data, dict) or not data.get("tags"):
            self.btn_preset_remove.setEnabled(False)
            return
        self.btn_preset_remove.setEnabled(True)
        self.tags.setText(str(data.get("tags", "")).strip())

    def _add_preset(self) -> None:
        tags = self.tags.text().strip()
        if not tags:
            QMessageBox.information(self, "No tags", "Enter some tags first, then click Add.")
            return

        genre = self.genre_combo.currentText().strip() or "Custom"
        if genre == "All":
            genre = "Custom"

        name, ok = QInputDialog.getText(self, "Add preset", "Preset name:")
        name = (name or "").strip()
        if not ok or not name:
            return

        g = self._get_genre(genre)
        if g is None:
            g = {"name": genre, "presets": []}
            self.preset_data.setdefault("genres", []).append(g)
            # Add to dropdown (keep 'All' on top).
            if self.genre_combo.findText(genre) < 0:
                self.genre_combo.addItem(genre)

        g.setdefault("presets", []).append({"name": name, "tags": tags})
        self.preset_store.data = self.preset_data
        self.preset_store.save()
        self._append_log(f"Preset added: {genre} / {name}")

        # Jump to the new preset.
        self.genre_combo.setCurrentText(genre)
        self._rebuild_preset_combo()
        for i in range(self.preset_combo.count()):
            d = self.preset_combo.itemData(i, Qt.UserRole)
            if isinstance(d, dict) and d.get("genre") == genre and d.get("name") == name:
                self.preset_combo.setCurrentIndex(i)
                break

    def _remove_preset(self) -> None:
        data = self.preset_combo.currentData(Qt.UserRole)
        if not isinstance(data, dict):
            return
        genre = str(data.get("genre", "")).strip()
        name = str(data.get("name", "")).strip()
        if not genre or not name:
            return

        if QMessageBox.question(
            self,
            "Remove preset",
            f"Remove preset '{name}' from genre '{genre}'?",
        ) != QMessageBox.Yes:
            return

        g = self._get_genre(genre)
        if not isinstance(g, dict):
            return
        presets = g.get("presets", [])
        if not isinstance(presets, list):
            return

        new_list = [p for p in presets if not (isinstance(p, dict) and str(p.get("name", "")).strip() == name)]
        g["presets"] = new_list
        self.preset_store.data = self.preset_data
        self.preset_store.save()
        self._append_log(f"Preset removed: {genre} / {name}")
        self._rebuild_preset_combo()

    def _save_settings(self) -> None:
        self.settings.model_path = self.model_path.text().strip() or "models/HeartMuLa"
        self.settings.version = self.version.currentText()
        self.settings.max_audio_length_ms = _time_to_ms(self.max_len.time())
        self.settings.topk = int(self.topk.value())
        self.settings.temperature = float(self.temperature.value())
        self.settings.cfg_scale = float(self.cfg_scale.value())
        self.settings.lyrics_text = self.lyrics.toPlainText()
        self.settings.tags_text = self.tags.text().strip()
        self.settings.output_dir = self.output_dir.text().strip() or "output/music/heartmula"
        self.settings.save(self.settings_path)
        self._append_log(f"Saved: {self.settings_path}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._save_settings()
        except Exception:
            pass
        super().closeEvent(event)

    def _env_python(self) -> Path:
        root = _root_dir()
        vpy = root / ".mula_env" / "Scripts" / "python.exe"
        if vpy.exists():
            return vpy
        return Path(sys.executable)

    def _heartlib_script(self) -> Path:
        root = _root_dir()
        # installer downloads heartlib here
        return (root / "models" / "HeartMuLa" / "_heartlib_src" / "examples" / "run_music_generation.py")

    def _validate_install(self) -> tuple[bool, str]:
        root = _root_dir()
        model_dir = (root / self.model_path.text()).resolve()
        if not model_dir.exists():
            return False, f"Model folder not found: {model_dir}"
        # Check minimal expected files per upstream README
        needed = [
            model_dir / "HeartCodec-oss",
            model_dir / "HeartMuLa-oss-3B",
            model_dir / "gen_config.json",
            model_dir / "tokenizer.json",
        ]
        missing = [p for p in needed if not p.exists()]
        if missing:
            return False, "Missing model files/folders:\n" + "\n".join(str(p) for p in missing)

        script = self._heartlib_script()
        if not script.exists():
            return False, f"heartlib example script missing: {script}\nRun presets/extra_env/mula_install.bat"

        return True, ""
    # --------------------- generation ---------------------

    def _goto_queue_tab(self) -> bool:
        """Best-effort: switch the main UI to the Queue tab."""
        try:
            win = self.window()
            tabs = win.findChild(QTabWidget)
            if not tabs:
                return False
            for i in range(tabs.count()):
                try:
                    if tabs.tabText(i).strip().lower() == "queue":
                        tabs.setCurrentIndex(i)
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def _generate(self) -> None:
        """Queue-first generation (no direct-run).

        HeartMuLa is VRAM-heavy, so we enqueue by default so it won't collide
        with other jobs.
        """
        ok, err = self._validate_install()
        if not ok:
            QMessageBox.critical(self, "Not installed", err)
            return

        self._save_settings()

        root = _root_dir()
        tmp_dir = root / "presets" / "setsave" / "_mula_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        lyrics_path = tmp_dir / "lyrics.txt"
        tags_path = tmp_dir / "tags.txt"
        lyrics_txt = self.lyrics.toPlainText().strip()
        tags_txt = self.tags.text().strip()
        lyrics_path.write_text((lyrics_txt or "") + "\n", encoding="utf-8")
        tags_path.write_text((tags_txt or "") + "\n", encoding="utf-8")

        out_dir = (root / (self.output_dir.text().strip() or "output/music/heartmula")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self.output_name.text().strip()
        if not name:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            slug = self._auto_name_slug()
            if slug:
                name = f"heartmula_{slug}_{ts}.mp3"
            else:
                name = f"heartmula_{ts}.mp3"
        if not name.lower().endswith(".mp3"):
            name += ".mp3"
        out_path = out_dir / name

        py = str(self._env_python())
        script = str(self._heartlib_script())
        model_dir = str((root / (self.model_path.text().strip() or "models/HeartMuLa")).resolve())

        cmd = [
            py,
            script,
            f"--model_path={model_dir}",
            f"--version={self.version.currentText()}",
            f"--lyrics={str(lyrics_path)}",
            f"--tags={str(tags_path)}",
            f"--save_path={str(out_path)}",
            f"--max_audio_length_ms={_time_to_ms(self.max_len.time())}",
            f"--topk={int(self.topk.value())}",
            f"--temperature={float(self.temperature.value())}",
            f"--cfg_scale={float(self.cfg_scale.value())}",
        ]

        # Placeholder input so queue rows have something to show.
        placeholder = tmp_dir / "queued_job.txt"
        try:
            preview = (tags_txt or "(no tags)")
            placeholder.write_text(preview + "\n", encoding="utf-8")
        except Exception:
            pass

        # Queue job
        try:
            from helpers.queue_adapter import enqueue_tool_job
        except Exception:
            # Fallback for standalone runs
            from queue_adapter import enqueue_tool_job  # type: ignore

        label = "HeartMuLa: " + ((tags_txt.replace("\n", " ").strip()[:60]) if tags_txt else "music")
        job_args = {
            "label": label,
            "cmd": cmd,
            "cwd": str(root),
            "outfile": str(out_path),
            "prepend_path": str((root / "presets" / "bin").resolve()),
            # extras (useful for metadata / debugging)
            "version": self.version.currentText(),
            "model_path": model_dir,
            "tags": tags_txt,
        }

        ok = enqueue_tool_job(
            "heartmula_generate",
            str(placeholder),
            str(out_dir),
            job_args,
            priority=450,
        )

        self._append_log("\n---")
        self._append_log("Queued HeartMuLa job")
        self._append_log("Command: " + " ".join(str(x) for x in cmd))

        # Queue switch + UX
        try:
            self._goto_queue_tab()
        except Exception:
            pass

        if not ok:
            self._show_toast("Failed to add job to queue.", error=True)
        else:
            self._show_toast("Added to queue.")

        # No direct process running in the tool UI
        try:
            self.btn_stop.setEnabled(False)
        except Exception:
            pass

    def _stop(self) -> None:
        # Queue-controlled (cancel via Queue right-click).
        self._show_toast("Cancel this job from the Queue tab.")


    def _on_proc_output(self) -> None:
        # Legacy direct-run hook (kept for compatibility; queue mode doesn't use it).
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            data = ""
        if data:
            for line in data.splitlines():
                self._append_log(line)

    def _on_proc_finished(self, exit_code: int, _status) -> None:  # type: ignore[override]
        # Legacy direct-run hook (kept for compatibility; queue mode doesn't use it).
        try:
            self.btn_generate.setEnabled(True)
        except Exception:
            pass
        try:
            self.btn_stop.setEnabled(False)
        except Exception:
            pass
        try:
            self._append_log(f"Finished (exit code {exit_code}).")
        except Exception:
            pass


def main() -> int:
    app = QApplication(sys.argv)
    w = HeartMuLaUI()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
