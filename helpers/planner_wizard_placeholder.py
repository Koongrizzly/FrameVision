# helpers/planner_wizard_placeholder.py
from __future__ import annotations

import os
import time
import random
from dataclasses import asdict
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    # Normal package import (when helpers is a real package)
    from .planner_config import PlannerConfig, CharacterSpec
except Exception:
    try:
        # If executed directly from the same folder
        from planner_config import PlannerConfig, CharacterSpec  # type: ignore
    except Exception:
        # If executed from project root, ensure root is on sys.path then import helpers.*
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
        from helpers.planner_config import PlannerConfig, CharacterSpec  # type: ignore

# ------------------------------
# Utilities
# ------------------------------

def _hbox(*widgets: QtWidgets.QWidget, stretch_last: bool = True) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, wd in enumerate(widgets):
        lay.addWidget(wd)
    if stretch_last:
        lay.addStretch(1)
    return w

def _vbox(*widgets: QtWidgets.QWidget, stretch_last: bool = True) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for wd in widgets:
        lay.addWidget(wd)
    if stretch_last:
        lay.addStretch(1)
    return w

def _label(title: str, bold: bool = True) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(title)
    if bold:
        f = lab.font()
        f.setBold(True)
        lab.setFont(f)
    return lab


# ------------------------------
# Fake pipeline worker (placeholder)
# Replace with your real queue/worker integration later.
# ------------------------------

class PlannerWorker(QtCore.QObject):
    """
    Placeholder "pipeline" that emits preview artifacts progressively.
    It supports:
      - pause at checkpoint (between items)
      - regenerate selected item (queued)
      - staged flow: images stage then videos stage
    """
    sig_stage_changed = QtCore.Signal(str)
    sig_item_done = QtCore.Signal(str, int, str, dict)  # stage, index, path, meta
    sig_progress = QtCore.Signal(str, int, int)         # stage, done, total
    sig_log = QtCore.Signal(str)
    sig_paused = QtCore.Signal(str, int, int)           # stage, current, total
    sig_finished = QtCore.Signal(bool)

    def __init__(self, cfg: PlannerConfig, out_dir: str):
        super().__init__()
        self.cfg = cfg
        self.out_dir = out_dir
        self._abort = False

        self.pause_requested = False
        self._paused = False
        self._regen_queue: List[Tuple[str, int]] = []

        self._current_stage = ""
        self._total_images = 8
        self._total_videos = 4

    @QtCore.Slot()
    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            # "Resolve" models (placeholder)
            self.cfg.resolved["image_model"] = (
                "AUTO" if self.cfg.image_model_strategy == "auto"
                else (self.cfg.fixed_image_model or "IMAGE_MODEL_PLACEHOLDER")
            )
            self.cfg.resolved["video_model"] = (
                "AUTO" if self.cfg.video_model_strategy == "auto"
                else (self.cfg.fixed_video_model or "VIDEO_MODEL_PLACEHOLDER")
            )

            # Stage 1: images
            self._current_stage = "images"
            self.sig_stage_changed.emit(self._current_stage)
            self.sig_log.emit("Starting images stage…")

            self._generate_items(stage="images", total=self._total_images, kind="image")

            if self._abort:
                self.sig_finished.emit(False)
                return

            # Optional: pause after images stage completes so user can inspect/regenerate before moving on
            if (not self._abort) and bool(getattr(self.cfg, "allow_edit_images", False)):
                self.sig_log.emit("Images complete — editing pause enabled. Inspect results and regenerate if needed, then press Resume.")
                self._paused = True
                self.sig_paused.emit("images", self._total_images, self._total_images)
                while self._paused and (not self._abort):
                    # Allow queued regenerations while paused
                    self._process_regens("images", "image")
                    time.sleep(0.05)

            # Stage 2: videos
            self._current_stage = "videos"
            self.sig_stage_changed.emit(self._current_stage)
            self.sig_log.emit("Starting videos stage…")

            self._generate_items(stage="videos", total=self._total_videos, kind="video")

            if self._abort:
                self.sig_finished.emit(False)
                return

            # Optional: pause after videos stage completes so user can inspect/regenerate before finishing
            if (not self._abort) and bool(getattr(self.cfg, "allow_edit_videos", False)):
                self.sig_log.emit("Videos complete — editing pause enabled. Inspect results and regenerate if needed, then press Resume to finish.")
                self._paused = True
                self.sig_paused.emit("videos", self._total_videos, self._total_videos)
                while self._paused and (not self._abort):
                    self._process_regens("videos", "video")
                    time.sleep(0.05)

            self.sig_log.emit("Pipeline finished (placeholder).")
            self.sig_finished.emit(True)
        except Exception as e:
            self.sig_log.emit(f"Worker error: {e}")
            self.sig_finished.emit(False)

    def _generate_items(self, stage: str, total: int, kind: str):
        done = 0
        for idx in range(total):
            if self._abort:
                return

            # Honor pause requests ONLY if requested before last item begins (per your rule).
            if self.pause_requested and idx < (total - 1):
                self._paused = True
                self.sig_paused.emit(stage, idx, total)
                self.sig_log.emit(f"Paused at checkpoint before item {idx+1}/{total}.")
                while self._paused and (not self._abort):
                    time.sleep(0.05)

            # Simulate work
            time.sleep(0.25 + random.random() * 0.25)

            path, meta = self._write_placeholder_artifact(stage, idx, kind)
            done += 1
            self.sig_item_done.emit(stage, idx, path, meta)
            self.sig_progress.emit(stage, done, total)

            # If there are regen requests queued for this stage, process them between items.
            self._process_regens(stage, kind)

    def _process_regens(self, stage: str, kind: str):
        # Process queued regens (FIFO), but only while we're still inside the stage.
        # Each regen is treated like a quick re-run for that one item.
        while self._regen_queue and (not self._abort):
            st, idx = self._regen_queue.pop(0)
            if st != stage:
                continue

            self.sig_log.emit(f"Regenerating {stage} item {idx+1}…")
            time.sleep(0.15 + random.random() * 0.2)
            path, meta = self._write_placeholder_artifact(stage, idx, kind, regen=True)
            self.sig_item_done.emit(stage, idx, path, meta)
            # progress isn't incremented; it's a replacement version

    def _write_placeholder_artifact(self, stage: str, idx: int, kind: str, regen: bool = False):
        # Make a colored pixmap as an image artifact; for "video" just write a dummy txt path
        seed = random.randint(0, 999999)
        version = random.randint(2, 9) if regen else 1
        meta = {
            "seed": seed,
            "version": version,
            "model": self.cfg.resolved.get("image_model" if stage == "images" else "video_model", "AUTO"),
            "kind": kind,
            "stage": stage,
        }

        if kind == "image":
            w, h = (768, 432) if self.cfg.aspect == "16:9" else (432, 768) if self.cfg.aspect == "9:16" else (640, 640)
            pm = QtGui.QPixmap(w, h)
            pm.fill(QtGui.QColor(random.randint(20, 230), random.randint(20, 230), random.randint(20, 230)))

            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 4))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 160)))
            painter.drawRoundedRect(20, 20, w - 40, h - 40, 24, 24)

            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
            font = QtGui.QFont()
            font.setBold(True)
            font.setPointSize(18)
            painter.setFont(font)
            painter.drawText(QtCore.QRect(40, 40, w - 80, h - 80),
                             QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap,
                             f"{stage.upper()}  #{idx+1}\n"
                             f"v{version}  seed {seed}\n"
                             f"model: {meta['model']}\n"
                             f"prompt: {self.cfg.prompt[:80]}")
            painter.end()

            fname = f"{stage}_{idx+1:02d}_v{version}_seed{seed}.png"
            fpath = os.path.join(self.out_dir, fname)
            pm.save(fpath)
            return fpath, meta

        # "video" placeholder
        fname = f"{stage}_{idx+1:02d}_v{version}_seed{seed}.txt"
        fpath = os.path.join(self.out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"Placeholder video artifact for {stage} #{idx+1}\n")
            f.write(f"v{version} seed {seed}\n")
            f.write(f"model: {meta['model']}\n")
            f.write(f"prompt: {self.cfg.prompt}\n")
        return fpath, meta

    # Control API
    @QtCore.Slot()
    def request_pause(self):
        self.pause_requested = True

    @QtCore.Slot()
    def resume(self):
        self.pause_requested = False
        self._paused = False

    @QtCore.Slot(str, int)
    def request_regen(self, stage: str, index: int):
        self._regen_queue.append((stage, index))

    @QtCore.Slot()
    def abort(self):
        self._abort = True
        self._paused = False


# ------------------------------
# Wizard pages
# ------------------------------

class WelcomePage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Welcome")
        self.setSubTitle("A guided planner that can turn a prompt into story, visuals, audio, and a finished video.")

        self.chk_advanced = QtWidgets.QCheckBox("Show advanced options")
        self.chk_advanced.setChecked(False)

        bullets = QtWidgets.QLabel(
            "What this planner can do (overview):\n"
            "• Build a storyline from your prompt\n"
            "• Generate storyboard images and/or videos\n"
            "• Create background music (Ace Step) or lyric music (HeartMula), or use your own\n"
            "• Add narration (TTS or your own voice) and mix music/voice\n"
            "• Show previews while running, and let you pause/regenerate items before the stage ends\n"
        )
        bullets.setWordWrap(True)

        note = QtWidgets.QLabel("Tip: Default mode keeps it minimal. Advanced mode reveals model routing and extra controls.")
        note.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(bullets)
        lay.addWidget(note)
        lay.addSpacing(10)
        lay.addWidget(self.chk_advanced)
        lay.addStretch(1)

        self.registerField("advanced_mode", self.chk_advanced)

    def validatePage(self) -> bool:
        self.cfg.advanced_mode = bool(self.field("advanced_mode"))
        return True


class IdeaDetailsPage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Idea")
        self.setSubTitle("Enter your prompt and any extra details you want included.")

        self.txt_prompt = QtWidgets.QPlainTextEdit()
        self.txt_prompt.setPlaceholderText("Your main idea/prompt (required)…")
        self.txt_prompt.setMinimumHeight(90)

        self.txt_details = QtWidgets.QPlainTextEdit()
        self.txt_details.setPlaceholderText("Extra details (optional): night, snow, nature, vibe, era, camera style…")
        self.txt_details.setMinimumHeight(90)

        self.lbl_hint = QtWidgets.QLabel("Next is enabled when the prompt is not empty.")
        self.lbl_hint.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_label("Prompt"))
        lay.addWidget(self.txt_prompt)
        lay.addWidget(_label("Extra details"))
        lay.addWidget(self.txt_details)
        lay.addWidget(self.lbl_hint)
        lay.addStretch(1)

        self.txt_prompt.textChanged.connect(self.completeChanged)

    def isComplete(self) -> bool:
        return bool(self.txt_prompt.toPlainText().strip())

    def validatePage(self) -> bool:
        self.cfg.prompt = self.txt_prompt.toPlainText().strip()
        self.cfg.extra_details = self.txt_details.toPlainText().strip()
        return True


class CharactersPage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Characters")
        self.setSubTitle("Optional: add up to 2 characters. Leave empty to keep it simple.")

        self.chk_enable = QtWidgets.QCheckBox("Add characters")
        self.chk_enable.setChecked(False)

        self.grp1 = self._make_char_group("Character 1")
        self.grp2 = self._make_char_group("Character 2")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.chk_enable)
        lay.addWidget(self.grp1)
        lay.addWidget(self.grp2)
        lay.addStretch(1)

        self.chk_enable.toggled.connect(self._update_enabled)
        self._update_enabled(False)

    def _make_char_group(self, title: str) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox(title)
        name = QtWidgets.QLineEdit()
        desc = QtWidgets.QPlainTextEdit()
        desc.setPlaceholderText("Short description (traits, role, clothing, vibe)…")
        desc.setMinimumHeight(70)

        btn_pick = QtWidgets.QPushButton("Pick reference image…")
        lbl_path = QtWidgets.QLabel("")
        lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # store on widget for easy access
        grp._name = name
        grp._desc = desc
        grp._btn_pick = btn_pick
        grp._lbl_path = lbl_path

        btn_pick.clicked.connect(lambda: self._pick_image(grp))

        form = QtWidgets.QFormLayout(grp)
        form.addRow("Name", name)
        form.addRow("Description", desc)

        # Advanced-only row: ref image (still optional)
        roww = QtWidgets.QWidget()
        rowl = QtWidgets.QVBoxLayout(roww)
        rowl.setContentsMargins(0, 0, 0, 0)
        rowl.addWidget(_hbox(btn_pick, lbl_path, stretch_last=True))
        grp._ref_row = roww
        form.addRow("Reference (advanced)", roww)

        return grp

    def _pick_image(self, grp: QtWidgets.QGroupBox):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            grp._lbl_path.setText(path)

    def initializePage(self) -> None:
        # Hide/show advanced-only ref rows based on welcome checkbox
        adv = bool(self.wizard().field("advanced_mode"))
        for grp in (self.grp1, self.grp2):
            grp._ref_row.setVisible(adv)

    def _update_enabled(self, enabled: bool):
        self.grp1.setEnabled(enabled)
        self.grp2.setEnabled(enabled)

    def validatePage(self) -> bool:
        enabled = self.chk_enable.isChecked()

        def read_grp(grp: QtWidgets.QGroupBox) -> CharacterSpec:
            return CharacterSpec(
                enabled=enabled and bool(grp._name.text().strip() or grp._desc.toPlainText().strip() or grp._lbl_path.text().strip()),
                name=grp._name.text().strip(),
                description=grp._desc.toPlainText().strip(),
                reference_image_path=grp._lbl_path.text().strip(),
            )

        self.cfg.character_1 = read_grp(self.grp1)
        self.cfg.character_2 = read_grp(self.grp2)
        return True


class OutputTypePage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("What do you want to create?")
        self.setSubTitle("Choose between a narrated storyline video or a music videoclip.")

        self.rad_narrated = QtWidgets.QRadioButton("Storyline video (with narration)")
        self.rad_music = QtWidgets.QRadioButton("Music videoclip")
        self.rad_narrated.setChecked(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.rad_narrated)
        lay.addWidget(self.rad_music)
        lay.addStretch(1)

    def validatePage(self) -> bool:
        self.cfg.output_type = "narrated_story" if self.rad_narrated.isChecked() else "music_videoclip"
        return True


class AudioPage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Audio")
        self.setSubTitle("Pick music and/or narration. Default stays simple, advanced reveals more control.")

        # Narration
        self.chk_narration = QtWidgets.QCheckBox("Narration")
        self.chk_narration.setChecked(True)
        self.cmb_narration = QtWidgets.QComboBox()
        self.cmb_narration.addItems(["TTS (default)", "User voice file"])
        self.btn_voice = QtWidgets.QPushButton("Pick voice file…")
        self.lbl_voice = QtWidgets.QLabel("")
        self.lbl_voice.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.btn_voice.clicked.connect(self._pick_voice)

        # Music
        self.chk_music = QtWidgets.QCheckBox("Music")
        self.chk_music.setChecked(False)
        self.cmb_music = QtWidgets.QComboBox()
        self.cmb_music.addItems(["Auto (recommended)", "Ace Step", "HeartMula", "User music file"])
        self.btn_music = QtWidgets.QPushButton("Pick music file…")
        self.lbl_music = QtWidgets.QLabel("")
        self.lbl_music.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.btn_music.clicked.connect(self._pick_music)

        # Lyric settings (music videoclip path)
        self.grp_lyrics = QtWidgets.QGroupBox("Lyrics (music videoclip)")
        self.rad_auto_lyrics = QtWidgets.QRadioButton("Auto lyrics based on storyline")
        self.rad_user_lyrics = QtWidgets.QRadioButton("Use my own lyrics")
        self.rad_user_audio = QtWidgets.QRadioButton("Use my own audio (skip lyric generation)")
        self.rad_auto_lyrics.setChecked(True)
        self.txt_lyrics = QtWidgets.QPlainTextEdit()
        self.txt_lyrics.setPlaceholderText("Paste your lyrics here…")
        self.txt_lyrics.setMinimumHeight(80)

        ly = QtWidgets.QVBoxLayout(self.grp_lyrics)
        ly.addWidget(self.rad_auto_lyrics)
        ly.addWidget(self.rad_user_lyrics)
        ly.addWidget(self.txt_lyrics)
        ly.addWidget(self.rad_user_audio)

        # Mix (advanced)
        self.grp_mix = QtWidgets.QGroupBox("Mix (advanced)")
        self.sld_music = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_music.setRange(0, 100)
        self.sld_music.setValue(45)
        self.sld_voice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_voice.setRange(0, 100)
        self.sld_voice.setValue(70)
        form = QtWidgets.QFormLayout(self.grp_mix)
        form.addRow("Music volume", self.sld_music)
        form.addRow("Voice volume", self.sld_voice)

        # layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_label("Narration"))
        lay.addWidget(_hbox(self.chk_narration, self.cmb_narration, stretch_last=True))
        lay.addWidget(_hbox(self.btn_voice, self.lbl_voice, stretch_last=True))
        lay.addSpacing(10)
        lay.addWidget(_label("Music"))
        lay.addWidget(_hbox(self.chk_music, self.cmb_music, stretch_last=True))
        lay.addWidget(_hbox(self.btn_music, self.lbl_music, stretch_last=True))
        lay.addSpacing(10)
        lay.addWidget(self.grp_lyrics)
        lay.addWidget(self.grp_mix)
        lay.addStretch(1)

        # react
        self.chk_narration.toggled.connect(self._sync_enabled)
        self.chk_music.toggled.connect(self._sync_enabled)
        self.cmb_music.currentIndexChanged.connect(self._sync_enabled)
        self.cmb_narration.currentIndexChanged.connect(self._sync_enabled)
        self._sync_enabled()

    def _pick_music(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select music file", "", "Audio (*.mp3 *.wav *.flac *.ogg)")
        if path:
            self.lbl_music.setText(path)

    def _pick_voice(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select voice file", "", "Audio (*.mp3 *.wav *.flac *.ogg)")
        if path:
            self.lbl_voice.setText(path)

    def initializePage(self) -> None:
        # Show/hide lyric options based on output type
        is_music_vclip = (self.cfg.output_type == "music_videoclip")
        self.grp_lyrics.setVisible(is_music_vclip)

        # Narration default differs by output type
        if is_music_vclip:
            self.chk_narration.setChecked(False)
        else:
            self.chk_narration.setChecked(True)

        # Advanced: show mix group only if advanced enabled
        adv = bool(self.wizard().field("advanced_mode"))
        self.grp_mix.setVisible(adv)

        self._sync_enabled()

    def _sync_enabled(self):
        self.cmb_narration.setEnabled(self.chk_narration.isChecked())
        use_voice = self.chk_narration.isChecked() and (self.cmb_narration.currentIndex() == 1)
        self.btn_voice.setEnabled(use_voice)
        self.lbl_voice.setEnabled(use_voice)

        self.cmb_music.setEnabled(self.chk_music.isChecked())
        use_user_music = self.chk_music.isChecked() and (self.cmb_music.currentIndex() == 3)
        self.btn_music.setEnabled(use_user_music)
        self.lbl_music.setEnabled(use_user_music)

        # Lyrics text only enabled when "Use my own lyrics" selected
        self.txt_lyrics.setEnabled(self.grp_lyrics.isVisible() and self.rad_user_lyrics.isChecked())

    def validatePage(self) -> bool:
        is_music_vclip = (self.cfg.output_type == "music_videoclip")

        self.cfg.want_narration = self.chk_narration.isChecked()
        self.cfg.narration_mode = "tts" if self.cmb_narration.currentIndex() == 0 else "user_voice"
        self.cfg.user_voice_path = self.lbl_voice.text().strip()

        self.cfg.want_music = self.chk_music.isChecked() or is_music_vclip
        # map music combo
        idx = self.cmb_music.currentIndex()
        self.cfg.music_mode = ["auto", "ace_step", "heartmula", "user_file"][idx]
        self.cfg.user_music_path = self.lbl_music.text().strip()

        if is_music_vclip:
            if self.rad_auto_lyrics.isChecked():
                self.cfg.lyrics_mode = "auto_from_story"
            elif self.rad_user_lyrics.isChecked():
                self.cfg.lyrics_mode = "user_lyrics"
            else:
                self.cfg.lyrics_mode = "user_audio"
            self.cfg.user_lyrics = self.txt_lyrics.toPlainText().strip()
        else:
            self.cfg.lyrics_mode = "auto_from_story"
            self.cfg.user_lyrics = ""

        # Mix
        self.cfg.music_volume = int(self.sld_music.value())
        self.cfg.voice_volume = int(self.sld_voice.value())
        return True


class ModelsPage(QtWidgets.QWizardPage):
    """
    Placeholder: one page for Image models and one for Video models.
    In default mode, keep it minimal (Auto / Fixed).
    In advanced mode, reveal priority list selection.
    """
    def __init__(self, cfg: PlannerConfig, kind: str):
        super().__init__()
        self.cfg = cfg
        self.kind = kind  # "image" or "video"
        self.setTitle("Models" if kind == "image" else "Models")
        self.setSubTitle(("Choose how image models are selected." if kind == "image" else "Choose how video models are selected.")
                         + " (Placeholder list, wire your real model registry later.)")

        self.cmb_strategy = QtWidgets.QComboBox()
        self.cmb_strategy.addItems(["Auto (recommended)", "Pick one model", "Pick multiple (priority order)"])

        self.cmb_fixed = QtWidgets.QComboBox()
        self.cmb_fixed.addItems(["", "Model A (placeholder)", "Model B (placeholder)", "Model C (placeholder)"])

        self.list_priority = QtWidgets.QListWidget()
        self.list_priority.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for m in ["Model A (placeholder)", "Model B (placeholder)", "Model C (placeholder)", "Model D (placeholder)"]:
            it = QtWidgets.QListWidgetItem(m)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.list_priority.addItem(it)

        self.btn_up = QtWidgets.QPushButton("Up")
        self.btn_down = QtWidgets.QPushButton("Down")
        self.btn_up.clicked.connect(lambda: self._move_selected(-1))
        self.btn_down.clicked.connect(lambda: self._move_selected(1))

        self.grp_adv = QtWidgets.QGroupBox("Priority list (advanced)")
        adv_lay = QtWidgets.QVBoxLayout(self.grp_adv)
        adv_lay.addWidget(self.list_priority)
        adv_lay.addWidget(_hbox(self.btn_up, self.btn_down, stretch_last=True))

        # Allow editing (advanced): pause after stage completes so the user can inspect/regenerate
        self.chk_allow_edit = QtWidgets.QCheckBox("Allow editing")
        self.lbl_allow_edit = QtWidgets.QLabel("")
        self.lbl_allow_edit.setWordWrap(True)
        self.grp_allow_edit = QtWidgets.QGroupBox("Editing (advanced)")
        ed_lay = QtWidgets.QVBoxLayout(self.grp_allow_edit)
        ed_lay.addWidget(self.chk_allow_edit)
        ed_lay.addWidget(self.lbl_allow_edit)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_label("Selection strategy"))
        lay.addWidget(self.cmb_strategy)
        lay.addWidget(_label("Fixed model (if picking one)"))
        lay.addWidget(self.cmb_fixed)
        lay.addWidget(self.grp_allow_edit)
        lay.addWidget(self.grp_adv)
        lay.addStretch(1)

        self.cmb_strategy.currentIndexChanged.connect(self._sync)
        self._sync()

    def initializePage(self) -> None:
        adv = bool(self.wizard().field("advanced_mode"))
        self.grp_adv.setVisible(adv)
        self.grp_allow_edit.setVisible(adv)

        # In default mode, hide multi-model strategy
        if not adv and self.cmb_strategy.currentIndex() == 2:
            self.cmb_strategy.setCurrentIndex(0)

        # Label text depends on whether this is the image or video models page
        if self.kind == "image":
            self.lbl_allow_edit.setText(
                "This will pause the planner after all images are created so you can inspect them, "
                "fine-tune prompts if needed, or regenerate bad images before continuing."
            )
            self.chk_allow_edit.setChecked(bool(getattr(self.cfg, "allow_edit_images", False)))
        else:
            self.lbl_allow_edit.setText(
                "This will pause the planner after all videos are created so you can inspect them, "
                "fine-tune prompts if needed, or regenerate bad videos before finishing."
            )
            self.chk_allow_edit.setChecked(bool(getattr(self.cfg, "allow_edit_videos", False)))

        self._sync()

    def _sync(self):
        strat = self.cmb_strategy.currentIndex()
        self.cmb_fixed.setEnabled(strat == 1)
        self.grp_adv.setEnabled(strat == 2)

    def _move_selected(self, delta: int):
        row = self.list_priority.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.list_priority.count():
            return
        item = self.list_priority.takeItem(row)
        self.list_priority.insertItem(new_row, item)
        self.list_priority.setCurrentRow(new_row)

    def validatePage(self) -> bool:
        strat_idx = self.cmb_strategy.currentIndex()
        strategy = ["auto", "fixed", "priority_list"][strat_idx]
        fixed = self.cmb_fixed.currentText().strip()

        priority: List[str] = []
        for i in range(self.list_priority.count()):
            it = self.list_priority.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                priority.append(it.text())

        if self.kind == "image":
            self.cfg.image_model_strategy = strategy
            self.cfg.fixed_image_model = fixed
            self.cfg.image_model_priority = priority
            self.cfg.allow_edit_images = bool(self.chk_allow_edit.isChecked())
        else:
            self.cfg.video_model_strategy = strategy
            self.cfg.fixed_video_model = fixed
            self.cfg.video_model_priority = priority
            self.cfg.allow_edit_videos = bool(self.chk_allow_edit.isChecked())
        return True


class OutputSettingsPage(QtWidgets.QWizardPage):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Output settings")
        self.setSubTitle("Keep it simple: quality, duration, and aspect ratio.")

        self.cmb_quality = QtWidgets.QComboBox()
        self.cmb_quality.addItems(["Fast", "Balanced (recommended)", "Best"])
        self.cmb_quality.setCurrentIndex(1)

        self.spin_duration = QtWidgets.QSpinBox()
        self.spin_duration.setRange(5, 600)
        self.spin_duration.setValue(30)
        self.spin_duration.setSuffix(" sec")

        self.cmb_aspect = QtWidgets.QComboBox()
        self.cmb_aspect.addItems(["16:9 (landscape)", "9:16 (portrait)", "1:1 (square)"])
        self.cmb_aspect.setCurrentIndex(0)

        form = QtWidgets.QFormLayout(self)
        form.addRow("Quality", self.cmb_quality)
        form.addRow("Duration", self.spin_duration)
        form.addRow("Aspect ratio", self.cmb_aspect)

    def validatePage(self) -> bool:
        self.cfg.quality = ["fast", "balanced", "best"][self.cmb_quality.currentIndex()]
        self.cfg.duration_sec = int(self.spin_duration.value())
        self.cfg.aspect = ["16:9", "9:16", "1:1"][self.cmb_aspect.currentIndex()]
        return True


class LivePreviewPage(QtWidgets.QWizardPage):
    """
    Page 7: live preview + pause/regenerate (placeholder).
    In your merge, this page can hook into your real queue worker / planner run.
    """
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Preview & last-minute changes")
        self.setSubTitle("Watch results appear, pause at checkpoints, and regenerate selected items before the stage ends.")

        # UI: stage + progress
        self.lbl_stage = _label("Stage: -")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)

        # Gallery
        self.list_items = QtWidgets.QListWidget()
        self.list_items.setIconSize(QtCore.QSize(128, 72))
        self.list_items.currentItemChanged.connect(self._on_selection)

        # Preview
        self.preview = QtWidgets.QLabel("No preview yet")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setStyleSheet("border: 1px solid rgba(0,0,0,60); border-radius: 10px;")

        # Controls
        self.btn_pause = QtWidgets.QPushButton("Pause after current")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_regen = QtWidgets.QPushButton("Regenerate selected")
        self.btn_open = QtWidgets.QPushButton("Open output folder")
        self.btn_cancel = QtWidgets.QPushButton("Cancel run")

        self.btn_resume.setEnabled(False)
        self.btn_regen.setEnabled(False)

        # Details (advanced)
        self.grp_details = QtWidgets.QGroupBox("Details (advanced)")
        self.txt_details = QtWidgets.QPlainTextEdit()
        self.txt_details.setReadOnly(True)
        dlay = QtWidgets.QVBoxLayout(self.grp_details)
        dlay.addWidget(self.txt_details)

        # Log
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(400)

        left = _vbox(_label("Results"), self.list_items, stretch_last=True)
        mid = _vbox(_label("Preview"), self.preview, stretch_last=True)
        right = _vbox(_label("Run log"), self.txt_log, self.grp_details, stretch_last=True)

        top_controls = _hbox(self.btn_pause, self.btn_resume, self.btn_regen, self.btn_open, self.btn_cancel, stretch_last=True)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.lbl_stage)
        root.addWidget(self.pb)
        root.addWidget(top_controls)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(mid)
        split.addWidget(right)
        split.setStretchFactor(0, 30)
        split.setStretchFactor(1, 40)
        split.setStretchFactor(2, 30)
        root.addWidget(split)

        # Runtime
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[PlannerWorker] = None
        self._out_dir = os.path.join(os.path.expanduser("~"), "FrameVision_Planner_Placeholder_Out")
        self._stage_totals = {}
        self._stage_done = {}
        self._selected_meta = None
        self._current_stage = ""

        # Connect buttons
        self.btn_pause.clicked.connect(self._pause)
        self.btn_resume.clicked.connect(self._resume)
        self.btn_regen.clicked.connect(self._regen_selected)
        self.btn_open.clicked.connect(self._open_out)
        self.btn_cancel.clicked.connect(self._cancel)

    def initializePage(self) -> None:
        adv = bool(self.wizard().field("advanced_mode"))
        self.grp_details.setVisible(adv)

        # Start worker when page becomes visible
        self._start_run()

    def cleanupPage(self) -> None:
        # Ensure worker stops if user goes back
        self._cancel()

    def isComplete(self) -> bool:
        # allow Finish only when worker done
        return getattr(self, "_finished_ok", False)

    def _start_run(self):
        self._finished_ok = False
        self.list_items.clear()
        self.preview.setText("Starting…")
        self.txt_log.setPlainText("")
        self.txt_details.setPlainText("")

        # thread + worker
        self._thread = QtCore.QThread(self)
        self._worker = PlannerWorker(self.cfg, self._out_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.sig_stage_changed.connect(self._on_stage)
        self._worker.sig_item_done.connect(self._on_item_done)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_log.connect(self._log)
        self._worker.sig_paused.connect(self._on_paused)
        self._worker.sig_finished.connect(self._on_finished)

        self._thread.start()

    def _on_stage(self, stage: str):
        self._current_stage = stage
        self.lbl_stage.setText(f"Stage: {stage}")
        self._stage_totals[stage] = 0
        self._stage_done[stage] = 0
        self._log(f"Stage changed: {stage}")
        # Hint: stage boundary coming warning
        self._update_controls()

    def _on_progress(self, stage: str, done: int, total: int):
        self._stage_totals[stage] = total
        self._stage_done[stage] = done

        pct = int((done / max(total, 1)) * 100)
        self.pb.setValue(pct)

        # Warning when last item is next
        if done == (total - 1):
            self._log("Last item is next — pause/regenerate now if you want changes.")

        self._update_controls()

    def _on_item_done(self, stage: str, idx: int, path: str, meta: dict):
        # Add/replace list item for this stage+index (support versions)
        key = f"{stage}:{idx}"
        existing = self._find_item_by_key(key)
        title = f"{stage.upper()} #{idx+1}  v{meta.get('version', 1)}"
        it = existing or QtWidgets.QListWidgetItem(title)
        it.setText(title)
        it.setData(QtCore.Qt.UserRole, {"key": key, "path": path, "meta": meta})

        if meta.get("kind") == "image" and os.path.exists(path):
            pm = QtGui.QPixmap(path)
            it.setIcon(QtGui.QIcon(pm.scaled(128, 72, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)))
        else:
            it.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))

        if existing is None:
            self.list_items.addItem(it)

        # auto-select newest
        self.list_items.setCurrentItem(it)
        self._update_controls()

    def _find_item_by_key(self, key: str) -> Optional[QtWidgets.QListWidgetItem]:
        for i in range(self.list_items.count()):
            it = self.list_items.item(i)
            data = it.data(QtCore.Qt.UserRole) or {}
            if data.get("key") == key:
                return it
        return None

    def _on_selection(self, cur: Optional[QtWidgets.QListWidgetItem], prev: Optional[QtWidgets.QListWidgetItem]):
        self.btn_regen.setEnabled(cur is not None and (not self._finished_ok))
        if not cur:
            return
        data = cur.data(QtCore.Qt.UserRole) or {}
        path = data.get("path", "")
        meta = data.get("meta", {})
        self._selected_meta = meta

        if meta.get("kind") == "image" and os.path.exists(path):
            pm = QtGui.QPixmap(path)
            self.preview.setPixmap(pm.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.preview.setText(f"Placeholder video artifact:\n{os.path.basename(path)}")

        # advanced details
        self.txt_details.setPlainText(
            f"Path: {path}\n"
            f"Stage: {meta.get('stage')}\n"
            f"Index: {data.get('key')}\n"
            f"Model: {meta.get('model')}\n"
            f"Seed: {meta.get('seed')}\n"
            f"Version: {meta.get('version')}\n"
            f"\nConfig (snapshot):\n{self.cfg.as_dict()}"
        )

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        # refresh preview scaling
        cur = self.list_items.currentItem()
        if cur:
            self._on_selection(cur, None)

    def _pause(self):
        if self._worker:
            self._worker.request_pause()
            self._log("Pause requested (will pause at next checkpoint, not mid-item).")
        self._update_controls()

    def _resume(self):
        if self._worker:
            self._worker.resume()
            self._log("Resuming…")
        self._update_controls()

    def _regen_selected(self):
        cur = self.list_items.currentItem()
        if not cur:
            return
        data = cur.data(QtCore.Qt.UserRole) or {}
        key = data.get("key", "")
        if ":" not in key:
            return
        stage, idxs = key.split(":", 1)
        try:
            idx = int(idxs)
        except ValueError:
            return

        # Enforce your rule: must happen before stage completes (before last item finishes).
        total = self._stage_totals.get(stage, 0)
        done = self._stage_done.get(stage, 0)
        if done >= total:  # stage complete
            # Allow post-stage regenerations if 'Allow editing' was enabled and we're still on that stage.
            if not (
                (stage == "images" and bool(getattr(self.cfg, "allow_edit_images", False)) and self._current_stage == "images")
                or (stage == "videos" and bool(getattr(self.cfg, "allow_edit_videos", False)) and self._current_stage == "videos")
            ):
                self._log("Too late: stage already finished. (In real app, offer rerun stage.)")
                return

        if self._worker:
            self._worker.request_regen(stage, idx)
            self._log(f"Queued regenerate for {stage} item #{idx+1}.")
        self._update_controls()

    def _open_out(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._out_dir))

    def _cancel(self):
        if self._worker:
            self._worker.abort()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self._update_controls()

    def _on_paused(self, stage: str, idx: int, total: int):
        self.btn_resume.setEnabled(True)
        self._log(f"Paused at checkpoint (before item {idx+1}/{total}).")

    def _on_finished(self, ok: bool):
        self._finished_ok = bool(ok)
        self._log("Finished." if ok else "Stopped / canceled.")
        self.btn_regen.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.completeChanged.emit()

        # stop thread
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] {msg}")

    def _update_controls(self):
        paused = False
        if self._worker:
            # can't reliably read internal paused; keep button states simple
            pass
        self.btn_pause.setEnabled((not self._finished_ok) and (self._worker is not None))
        self.btn_resume.setEnabled((not self._finished_ok) and (self._worker is not None))
        self.btn_cancel.setEnabled(self._worker is not None)
        self.btn_regen.setEnabled((self.list_items.currentItem() is not None) and (not self._finished_ok))


# ------------------------------
# Wizard shell
# ------------------------------

class PlannerWizard(QtWidgets.QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Planner Wizard (Placeholder)")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage, True)

        self._cfg = PlannerConfig()

        self.addPage(WelcomePage(self._cfg))
        self.addPage(IdeaDetailsPage(self._cfg))
        self.addPage(CharactersPage(self._cfg))
        self.addPage(OutputTypePage(self._cfg))
        self.addPage(AudioPage(self._cfg))

        # Image models page (advanced-friendly)
        img_models = ModelsPage(self._cfg, kind="image")
        img_models.setTitle("Image models")
        self.addPage(img_models)

        # Video models page (advanced-friendly)
        vid_models = ModelsPage(self._cfg, kind="video")
        vid_models.setTitle("Video models")
        self.addPage(vid_models)

        self.addPage(OutputSettingsPage(self._cfg))
        self.addPage(LivePreviewPage(self._cfg))

        # Buttons labeling
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next")
        self.setButtonText(QtWidgets.QWizard.BackButton, "Back")
        self.setButtonText(QtWidgets.QWizard.FinishButton, "Finish")
        self.setButtonText(QtWidgets.QWizard.CancelButton, "Close")

    @property
    def cfg(self) -> PlannerConfig:
        return self._cfg


def open_planner_wizard(parent=None) -> Optional[PlannerConfig]:
    dlg = PlannerWizard(parent)
    res = dlg.exec()
    if res == QtWidgets.QDialog.Accepted:
        return dlg.cfg
    return None


# ------------------------------
# Plugin entry point (required by the app's dynamic loader)
# ------------------------------

class PlannerWidget(QtWidgets.QWidget):
    """Small embedded launcher that opens the wizard as a modal dialog.

    Your app's loader expects either `create_widget()` or a QWidget class named
    PlannerWindow/PlannerDialog/PlannerPane/PlannerWidget/Planner.

    This keeps the wizard behaviour intact (Next/Back/Finish) while still
    providing a QWidget that can live inside a tab.
    """

    # Optional: your main app can connect to this if it wants the config.
    sig_config_ready = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_cfg: Optional[PlannerConfig] = None
        self.setObjectName("PlannerWidget")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)

        title = QtWidgets.QLabel("Planner (Placeholder)")
        f = title.font()
        f.setBold(True)
        f.setPointSize(max(10, f.pointSize() + 4))
        title.setFont(f)

        subtitle = QtWidgets.QLabel(
            "This module is a placeholder. Click below to open the guided planner wizard."
        )
        subtitle.setWordWrap(True)

        self.btn_open = QtWidgets.QPushButton("Open Planner Wizard…")
        self.btn_open.setMinimumHeight(36)
        self.btn_open.clicked.connect(self._open)

        self.txt_status = QtWidgets.QPlainTextEdit()
        self.txt_status.setReadOnly(True)
        self.txt_status.setPlaceholderText(
            "When you finish the wizard, the chosen settings will appear here."
        )
        self.txt_status.setMinimumHeight(160)

        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addSpacing(10)
        lay.addWidget(self.btn_open, 0, QtCore.Qt.AlignLeft)
        lay.addSpacing(10)
        lay.addWidget(self.txt_status)
        lay.addStretch(1)

    def _open(self):
        cfg = open_planner_wizard(self)
        if cfg is None:
            self.txt_status.setPlainText("Wizard closed.")
            return

        self._last_cfg = cfg

        # Show a readable snapshot of the config.
        try:
            d = cfg.as_dict() if hasattr(cfg, "as_dict") else asdict(cfg)  # type: ignore
        except Exception:
            d = {
                "prompt": getattr(cfg, "prompt", ""),
                "output_type": getattr(cfg, "output_type", ""),
            }

        try:
            import json
            self.txt_status.setPlainText(json.dumps(d, indent=2, ensure_ascii=False))
        except Exception:
            self.txt_status.setPlainText(str(d))

        self.sig_config_ready.emit(d)


def create_widget(parent=None) -> QtWidgets.QWidget:
    """Entry point used by the dynamic loader."""
    return PlannerWidget(parent)


# Extra alias (some loaders also accept this name)
Planner = PlannerWidget
