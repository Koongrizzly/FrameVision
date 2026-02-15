
import json
import os
import sys
from pathlib import Path

# FrameVision root (helpers/ace.py -> root)
ROOT_DIR = Path(__file__).resolve().parents[1]
# Preferred install location (root):  <root>/.ace_env/ACE-Step/
ACE_ENV_DIR = ROOT_DIR / ".ace_env"
# Legacy location (older installs): <root>/presets/extra_env/.ace_env/ACE-Step/
ACE_ENV_DIR_LEGACY = ROOT_DIR / "presets" / "extra_env" / ".ace_env"

def _detect_ace_env_dir() -> Path:
    """Return the active ACE virtualenv dir if installed, otherwise the preferred location."""
    for cand in (ACE_ENV_DIR, ACE_ENV_DIR_LEGACY):
        try:
            if (cand / "ACE-Step").exists():
                return cand
        except Exception:
            continue
    return ACE_ENV_DIR

ACE_ACTIVE_ENV_DIR = _detect_ace_env_dir()
ACE_REPO_DIR = ACE_ACTIVE_ENV_DIR / "ACE-Step"  # ACE-Step repo inside the env

from PySide6 import QtCore, QtWidgets

# NOTE:
# We intentionally do NOT import ACEStepPipeline (or 'acestep') in this process,
# because ACE-Step depends on its own 'diffusers' version which may be
# incompatible with FrameVision's main environment. Instead, AceWorker will
# spawn a small helper script inside the dedicated ACE-Step virtual environment.
# That script imports 'acestep.pipeline_ace_step.ACEStepPipeline' entirely
# inside the ACE env and saves the generated audio to disk.


class AceConfig:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.config_path = self.root_dir / "presets" / "setsave" / "ace.json"
        self.data = {}

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


# Presets now live in JSON via AcePresetManager.
# These dicts are kept for backwards compatibility but unused by the UI.
GENRE_PRESETS = {}
GENRE_PARAM_PRESETS = {}


class AcePresetManager:
    """JSON-based preset manager for ACE-Step.

    JSON path (relative to ROOT_DIR):
        presets/setsave/ace/presets/ace_presets.json
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.presets_path = (
            self.root_dir
            / "presets"
            / "setsave"
            / "ace"
            / "presets"
            / "ace_presets.json"
        )
        # UI genres (filter). "All" is only used in the UI, never stored in the JSON.
        self.genres = ["All", "EDM", "Rock", "Hiphop/RnB", "Other"]
        self.presets = []
        self.favorites = set()

    def load_presets(self):
        """Load presets + favorites from JSON, or seed a minimal default file."""
        if self.presets_path.exists():
            try:
                with open(self.presets_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}

        presets = data.get("presets")
        if not isinstance(presets, list):
            presets = [
                {
                    "name": "Custom (manual tags)",
                    "genre": "Other",
                    "prompt": "",
                    "params": {},
                }
            ]

        favs = data.get("favorites", [])
        if not isinstance(favs, list):
            favs = []

        cleaned = []
        for p in presets:
            name = str(p.get("name", "")).strip()
            if not name:
                continue
            genre = p.get("genre", "Other")
            if genre not in self.genres:
                genre = "Other"
            prompt_text = p.get("prompt", "")
            neg_text = p.get("negative_prompt", "")
            params = p.get("params", {}) or {}
            cleaned.append(
                {
                    "name": name,
                    "genre": genre,
                    "prompt": prompt_text,
                    "negative_prompt": neg_text,
                    "params": params,
                }
            )
        self.presets = cleaned
        all_names = {p["name"] for p in self.presets}
        self.favorites = {str(n) for n in favs if n in all_names}

        if not self.presets_path.exists():
            self.save_presets()

    def save_presets(self):
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "presets": self.presets,
            "favorites": sorted(self.favorites),
        }
        with open(self.presets_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_presets_for_genre(self, genre: str):
        if genre == "All":
            items = list(self.presets)
        else:
            items = [p for p in self.presets if p.get("genre") == genre]

        fav = []
        nonfav = []
        for p in items:
            if p["name"] in self.favorites:
                fav.append(p)
            else:
                nonfav.append(p)

        fav.sort(key=lambda p: p["name"].lower())
        nonfav.sort(key=lambda p: p["name"].lower())
        return fav + nonfav

    def get_preset_by_name(self, name: str):
        for p in self.presets:
            if p["name"] == name:
                return p
        return None

    def is_favorite(self, name: str) -> bool:
        return name in self.favorites

    def toggle_favorite(self, name: str):
        if name in self.favorites:
            self.favorites.remove(name)
        else:
            self.favorites.add(name)
        self.save_presets()



class AceWorker(QtCore.QThread):
    finished_with_result = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(self, root_dir: Path, config: AceConfig, ui_state: dict, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.config = config
        self.ui_state = ui_state

    def run(self):
        try:
            self.progress.emit("Preparing pipeline...")

            # Compute checkpoint directory (relative to the FrameVision root)
            checkpoint_rel = self.config.get("checkpoint_path", ".ace_env/ACE-Step/checkpoints")
            checkpoint_path = str((self.root_dir / checkpoint_rel).resolve())

            bf16 = bool(self.config.get("bf16", True))
            torch_compile = False  # force-disable torch.compile; Triton not available on this setup
            cpu_offload = bool(self.config.get("cpu_offload", False))
            overlapped_decode = bool(self.config.get("overlapped_decode", False))
            device_id = int(self.config.get("device_id", 0))

            prompt = self.ui_state["prompt"]
            negative_prompt = self.ui_state.get("negative_prompt", "")
            lyrics = self.ui_state["lyrics"]
            audio_duration = float(self.ui_state["audio_duration"])
            infer_step = int(self.ui_state["infer_step"])
            guidance_scale = float(self.ui_state["guidance_scale"])
            scheduler_type = self.ui_state["scheduler_type"]
            cfg_type = self.ui_state["cfg_type"]

            # Reference audio from UI state / config
            ref_audio_input = str(self.ui_state.get("ref_audio_input", "") or "").strip()
            ref_audio_strength = float(
                self.ui_state.get(
                    "ref_audio_strength",
                    self.config.get("ref_audio_strength", 0.5),
                )
            )
            if ref_audio_strength < 0.0:
                ref_audio_strength = 0.0
            if ref_audio_strength > 1.0:
                ref_audio_strength = 1.0
            audio2audio_enable = bool(self.ui_state.get("audio2audio_enable", bool(ref_audio_input)))

            # Normalize reference path to absolute if provided
            if ref_audio_input:
                ref_path = Path(ref_audio_input)
                if not ref_path.is_absolute():
                    ref_path = (self.root_dir / ref_path).resolve()
                ref_audio_input = str(ref_path)
                audio2audio_enable = True
            else:
                ref_audio_input = None
                audio2audio_enable = False

            seed = int(self.ui_state["seed"])
            if seed == 0:
                seed = QtCore.QRandomGenerator.global_().generate()
            actual_seeds = [int(seed)]

            omega_scale = float(self.config.get("omega_scale", 10.0))
            guidance_interval = float(self.config.get("guidance_interval", 0.5))
            guidance_interval_decay = float(self.config.get("guidance_interval_decay", 0.0))
            min_guidance_scale = float(self.config.get("min_guidance_scale", 3.0))
            use_erg_tag = bool(self.config.get("use_erg_tag", True))
            use_erg_lyric = bool(self.config.get("use_erg_lyric", False))  # default off for lyrics
            use_erg_diffusion = bool(self.config.get("use_erg_diffusion", True))
            oss_steps = self.config.get("oss_steps", [])
            guidance_scale_text = float(self.config.get("guidance_scale_text", 5.0))
            guidance_scale_lyric = float(self.config.get("guidance_scale_lyric", 1.5))

            manual_seeds = ", ".join(map(str, actual_seeds))
            oss_steps_str = ", ".join(map(str, oss_steps)) if oss_steps else ""

            out_dir = self.root_dir / self.ui_state["output_rel"]
            out_dir.mkdir(parents=True, exist_ok=True)

            # Build descriptive filename using user track name, seed and preset
            base_seed = actual_seeds[0] if actual_seeds else int(seed)
            track_name_raw = str(self.ui_state.get("track_name", "") or "").strip()
            preset_name_raw = str(self.config.get("preset_name", "") or "").strip()
            import re as _re
            def _slugify_name(value: str, default: str) -> str:
                value = (value or "").strip()
                if not value:
                    return default
                value = _re.sub(r"[^a-zA-Z0-9_]+", "_", value)
                value = value.strip("_") or default
                return value.lower()
            track_slug = _slugify_name(track_name_raw, "track")
            preset_slug = _slugify_name(preset_name_raw, "preset")

            filename = f"{track_slug}_{base_seed}_{preset_slug}.wav"
            output_path = str(out_dir / filename)

            # Prepare a small JSON config to hand off to the ACE env runner script.
            import json as _json
            import subprocess as _subprocess
            import tempfile as _tempfile
            import time as _time

            job = {
                "checkpoint_path": checkpoint_path,
                "dtype": "bfloat16" if bf16 else "float32",
                "torch_compile": bool(torch_compile),
                "cpu_offload": bool(cpu_offload),
                "overlapped_decode": bool(overlapped_decode),
                "device_id": int(device_id),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "cfg_type": cfg_type,
                "manual_seeds": manual_seeds,
                "omega_scale": omega_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
                "min_guidance_scale": min_guidance_scale,
                "use_erg_tag": use_erg_tag,
                "use_erg_lyric": use_erg_lyric,
                "use_erg_diffusion": use_erg_diffusion,
                "oss_steps": oss_steps_str,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "audio2audio_enable": bool(audio2audio_enable),
                "ref_audio_strength": float(ref_audio_strength),
                "ref_audio_input": ref_audio_input,
                "output_path": output_path,
            }

            tmp_dir = Path(_tempfile.gettempdir())
            job_path = tmp_dir / f"framevision_ace_job_{os.getpid()}_{int(_time.time())}.json"
            with open(job_path, "w", encoding="utf-8") as f:
                _json.dump(job, f)

            # Determine the ACE env Python and runner script.
            ace_env_dir = _detect_ace_env_dir()
            ace_repo_dir = ace_env_dir / "ACE-Step"

            if sys.platform.startswith("win"):
                ace_python = ace_env_dir / "Scripts" / "python.exe"
            else:
                ace_python = ace_env_dir / "bin" / "python"

            runner_script = ace_repo_dir / "framevision_ace_runner.py"


            if not ace_python.exists():
                raise RuntimeError(f"ACE env python not found at: {ace_python}")
            if not runner_script.exists():
                raise RuntimeError(
                    f"ACE runner script not found at: {runner_script}. "
                    "Make sure this file exists inside ACE-Step."
                )

            self.progress.emit("Running ACE-Step in external process...")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device_id)

            cmd = [str(ace_python), str(runner_script), str(job_path)]
            proc = _subprocess.run(
                cmd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
                text=True,
                env=env,
            )

            if proc.returncode != 0:
                err_tail = proc.stderr[-2000:] if proc.stderr else "Unknown error"
                raise RuntimeError(f"ACE-Step runner failed (code {proc.returncode}):\n{err_tail}")

            self.progress.emit("Generation finished.")
            self.finished_with_result.emit(output_path)

        except Exception as e:
            self.failed.emit(str(e))


class AceUI(QtWidgets.QWidget):
    playRequested = QtCore.Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ACE-Step Helper")
        self.root_dir = ROOT_DIR
        self.config = AceConfig(self.root_dir)
        self.last_output_path = None
        self.config.load()

        # JSON-backed preset manager (presets/setsave/ace/presets/ace_presets.json)
        self.preset_manager = AcePresetManager(self.root_dir)
        self.preset_manager.load_presets()

        self._build_ui()
        self._load_config_into_ui()

    def _build_ui(self):
        # Top-level layout: scrollable content + fixed bottom bar
        main_layout = QtWidgets.QVBoxLayout(self)

        # Fancy blue banner at the top
        self.banner = QtWidgets.QLabel("Generate Music with Ace-Step")
        self.banner.setObjectName("aceBanner")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.banner.setFixedHeight(45)
        self.banner.setStyleSheet(
            "#aceBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: #082c33;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #97e7f5,"
            "   stop:0.5 #7fd6eb,"
            "   stop:1 #5ac2dd"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )

        # User preference: hide the fancy banner (keep it defined in case we re-enable later).
        self.banner.setVisible(False)

        # --- Fixed top area (status + buttons) ---
        self.status_label = QtWidgets.QLabel("Status: idle")
        self.status_label.setWordWrap(True)
        self.status_label.setToolTip("Shows progress messages and any errors from ACE-Step.")
        main_layout.addWidget(self.status_label)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate Music")
        self.btn_generate.setToolTip("Start ACE-Step generation with the current settings.")
        self.btn_play_last = QtWidgets.QPushButton("Play last result")
        self.btn_play_last.setToolTip("Play the most recently generated track in the main player.")
        self.btn_open_folder = QtWidgets.QPushButton("View results")
        self.btn_open_folder.setToolTip("Open the ACE output folder in Media Explorer and scan for results.")

        # Enlarge button fonts (+3px) and add blue hover on Generate Music
        for b in (self.btn_generate, self.btn_play_last, self.btn_open_folder):
            f = b.font()
            sz = f.pointSize()
            if sz <= 0:
                sz = 10
            f.setPointSize(sz + 3)
            b.setFont(f)
            b.setMinimumHeight(32)
            b.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        self.btn_generate.setObjectName("aceGenerateButton")
        self.btn_generate.setStyleSheet(
            "QPushButton#aceGenerateButton {"
            " font-weight: 600;"
            "}"
            "QPushButton#aceGenerateButton:hover {"
            " background-color: #97e7f5;"
            " color: #082c33;"
            "}"
        )

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_play_last)
        btn_layout.addWidget(self.btn_open_folder)
        btn_layout.addStretch(1)
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(6)

        # Scroll area for all main controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        main_layout.addWidget(scroll_area)

        # Container widget inside the scroll area
        scroll_widget = QtWidgets.QWidget()
        scroll_area.setWidget(scroll_widget)

        layout = QtWidgets.QVBoxLayout(scroll_widget)
        layout.setContentsMargins(6,6,6,6)

        # Preset + tags row
    #    prompt_label = QtWidgets.QLabel("Prompt / Tags (style, genre, mood):")
     #   layout.addWidget(prompt_label)
        # Lyrics
        lyrics_label = QtWidgets.QLabel(
            "Lyrics (optional; use [instrumental] or [inst] for instrumentals):"
        )
        layout.addWidget(lyrics_label)

        self.lyrics_edit = QtWidgets.QPlainTextEdit()
        self.lyrics_edit.setPlaceholderText(
            "[verse]\nNeon lights above the crowd...\n\n"
            "[chorus]\nWhere can we go now...\n\n"
            "Use [instrumental] or [inst] for an instrumental-only track."
        )
        self.lyrics_edit.setToolTip(
            "Paste lyrics with [verse]/[chorus] tags, or type [instrumental] for a track without vocals."
        )
        layout.addWidget(self.lyrics_edit)
        self.lyrics_edit.setMinimumHeight(130)  # try 240–360 for a chunkier bar

        # Genre filter + preset list
        genre_layout = QtWidgets.QHBoxLayout()
        genre_layout.addWidget(QtWidgets.QLabel("Genre:"))
        self.genre_combo = QtWidgets.QComboBox()
        self.genre_combo.addItems(self.preset_manager.genres)
        self.genre_combo.setToolTip(
            "Filter presets by high-level genre.\n"
            "Choose 'All' to see every preset."
        )
        genre_layout.addWidget(self.genre_combo)
        layout.addLayout(genre_layout)

        presets_label = QtWidgets.QLabel("Presets (right-click to toggle favorite):")
        layout.addWidget(presets_label)

        self.preset_list = QtWidgets.QListWidget()
        self.preset_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.preset_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.preset_list.setToolTip(
            "Select a preset to fill in the tags below.\n"
            "Right-click a preset to mark it as a favorite.\n"
            "Favorites are shown at the top (and also in the 'All' view)."
        )
        layout.addWidget(self.preset_list)
        self.preset_list.setMinimumHeight(60)  # or 60, 100, etc.

        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "Example: electronic, club, 128 BPM, punchy kick, sidechained bass, "
            "bright hi-hats, english female vocal, teasing tone"
        )
        self.prompt_edit.setToolTip(
            "Describe the overall style: genre, tempo, mood and key sound design clues.\n"
            "ACE uses this to shape the groove, instrumentation and mix."
        )
        layout.addWidget(self.prompt_edit)
        self.prompt_edit.setMinimumHeight(40)  # try 40–60 for a chunkier bar

        # Optional user-defined track name (used in output filename).
        track_layout = QtWidgets.QHBoxLayout()
        track_label = QtWidgets.QLabel("Track name (optional):")
        track_layout.addWidget(track_label)
        self.track_name_edit = QtWidgets.QLineEdit()
        self.track_name_edit.setPlaceholderText("Example: summer_dance")
        self.track_name_edit.setToolTip(
            "Optional name for this track. If set, finished files are named like "
            "'trackname_seed_preset.wav' (for example: summer_dance_165445_deep_house.wav)."
        )
        track_layout.addWidget(self.track_name_edit)
        layout.addLayout(track_layout)

        negative_label = QtWidgets.QLabel("Negative tags (optional, things to avoid):")
        layout.addWidget(negative_label)

        self.negative_edit = QtWidgets.QLineEdit()
        self.negative_edit.setPlaceholderText(
            "Example: off-key vocals, harsh distortion, random drift, noisy artifacts"
        )
        self.negative_edit.setToolTip(
            "Describe what you DON'T want in the track: unwanted instruments, moods or artifacts."
        )
        layout.addWidget(self.negative_edit)

        # Reference audio (optional)
        ref_layout = QtWidgets.QHBoxLayout()
        ref_layout.addWidget(QtWidgets.QLabel("Reference track (optional):"))
        self.ref_audio_edit = QtWidgets.QLineEdit()
        self.ref_audio_edit.setPlaceholderText(
            "Optional: pick an existing track or stem to guide ACE (leave empty for pure text → music)."
        )
        self.ref_audio_edit.setToolTip(
            "Optional reference song or stem. When set, ACE tries to keep its groove, energy and structure."
        )
        ref_layout.addWidget(self.ref_audio_edit)
        self.ref_audio_browse_btn = QtWidgets.QPushButton("Browse...")
        self.ref_audio_browse_btn.setToolTip("Browse for a reference audio file on disk.")
        ref_layout.addWidget(self.ref_audio_browse_btn)
        layout.addLayout(ref_layout)

        ref_strength_layout = QtWidgets.QHBoxLayout()
        self.ref_strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ref_strength_slider.setRange(0, 100)
        self.ref_strength_slider.setSingleStep(5)
        self.ref_strength_slider.setValue(50)
        self.ref_strength_slider.setToolTip(
            "How much of the reference track to keep.\n"
            "0% = ignore reference, 100% = stay very close to it."
        )
        ref_strength_layout.addWidget(QtWidgets.QLabel("Reference influence:"))
        ref_strength_layout.addWidget(self.ref_strength_slider)
        self.ref_strength_label = QtWidgets.QLabel("50%")
        self.ref_strength_label.setToolTip("Numeric display of the reference influence slider.")
        ref_strength_layout.addWidget(self.ref_strength_label)
        layout.addLayout(ref_strength_layout)

        # Duration + steps
        duration_layout = QtWidgets.QHBoxLayout()
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(10, 240)
        self.duration_spin.setSingleStep(10)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip(
            "Target track length in seconds. Longer durations take more VRAM and generation time."
        )
        duration_layout.addWidget(QtWidgets.QLabel("Duration:"))
        duration_layout.addWidget(self.duration_spin)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(10, 120)
        self.steps_spin.setValue(60)
        self.steps_spin.setToolTip(
            "Number of diffusion steps. Higher = more detail and stability, but slower. 60 is a good default."
        )
        duration_layout.addWidget(QtWidgets.QLabel("Steps:"))
        duration_layout.addWidget(self.steps_spin)
        layout.addLayout(duration_layout)

        # Main guidance
        guidance_layout = QtWidgets.QHBoxLayout()
        self.guidance_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 30.0)
        self.guidance_spin.setDecimals(1)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(15.0)
        self.guidance_spin.setToolTip(
            "Global guidance strength (CFG). Higher values follow your text more exactly,\n"
            "but can sound less natural if pushed too far."
        )
        guidance_layout.addWidget(QtWidgets.QLabel("Guidance scale:"))
        guidance_layout.addWidget(self.guidance_spin)
        layout.addLayout(guidance_layout)

        # Text vs lyric guidance
        guidance_detail_layout = QtWidgets.QHBoxLayout()
        self.guidance_text_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_text_spin.setRange(0.0, 10.0)
        self.guidance_text_spin.setDecimals(1)
        self.guidance_text_spin.setSingleStep(0.1)
        self.guidance_text_spin.setValue(5.0)
        self.guidance_text_spin.setToolTip(
            "How strongly ACE should follow the tags/prompt when generating audio."
        )
        guidance_detail_layout.addWidget(QtWidgets.QLabel("Text guidance:"))
        guidance_detail_layout.addWidget(self.guidance_text_spin)

        self.guidance_lyric_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_lyric_spin.setRange(0.0, 10.0)
        self.guidance_lyric_spin.setDecimals(1)
        self.guidance_lyric_spin.setSingleStep(0.1)
        self.guidance_lyric_spin.setValue(1.5)
        self.guidance_lyric_spin.setToolTip(
            "How tightly the audio should follow the rhythm and phrasing of your lyrics."
        )
        guidance_detail_layout.addWidget(QtWidgets.QLabel("Lyric guidance:"))
        guidance_detail_layout.addWidget(self.guidance_lyric_spin)
        layout.addLayout(guidance_detail_layout)

        # Scheduler + CFG type
        scheduler_layout = QtWidgets.QHBoxLayout()
        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems(["euler", "ddim", "heun"])
        self.scheduler_combo.setToolTip(
            "Sampler / scheduler used during diffusion. 'euler' is a solid default;\n"
            "other options slightly change noise behaviour and texture."
        )
        scheduler_layout.addWidget(QtWidgets.QLabel("Scheduler:"))
        scheduler_layout.addWidget(self.scheduler_combo)

        self.cfg_combo = QtWidgets.QComboBox()
        self.cfg_combo.addItems(["apg", "vanilla"])
        self.cfg_combo.setToolTip(
            "CFG algorithm. 'apg' is recommended for ACE; 'vanilla' is standard classifier-free guidance."
        )
        scheduler_layout.addWidget(QtWidgets.QLabel("CFG type:"))
        scheduler_layout.addWidget(self.cfg_combo)
        layout.addLayout(scheduler_layout)

        # Seed
        seed_layout = QtWidgets.QHBoxLayout()
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip(
            "Random seed. 0 = new random result each time. Reuse the same number to reproduce a track."
        )
        seed_layout.addWidget(QtWidgets.QLabel("Seed (0 for random):"))
        seed_layout.addWidget(self.seed_spin)
        layout.addLayout(seed_layout)

        # Output folder
        out_layout = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setToolTip(
            "Where to save the generated .wav file, relative to the FrameVision root folder."
        )
        out_layout.addWidget(QtWidgets.QLabel("Output folder (relative to root):"))
        out_layout.addWidget(self.output_edit)
        layout.addLayout(out_layout)


        # Advanced settings
        advanced_label = QtWidgets.QLabel("Advanced settings")
        adv_font = advanced_label.font()
        adv_font.setBold(True)
        advanced_label.setFont(adv_font)
        advanced_label.setToolTip(
            "Extra ACE-Step controls for fine-tuning diffusion and ERG behaviour. "
            "If unsure, just leave these at their defaults."
        )
        layout.addWidget(advanced_label)

        # Advanced: diffusion knobs
        adv_diff_layout = QtWidgets.QGridLayout()
        adv_row = 0

        self.omega_spin = QtWidgets.QDoubleSpinBox()
        self.omega_spin.setRange(0.0, 50.0)
        self.omega_spin.setDecimals(1)
        self.omega_spin.setSingleStep(0.5)
        self.omega_spin.setToolTip(
            "Omega scale: balances randomness vs. structure. Higher values can sound more experimental."
        )
        adv_diff_layout.addWidget(QtWidgets.QLabel("Omega scale:"), adv_row, 0)
        adv_diff_layout.addWidget(self.omega_spin, adv_row, 1)
        adv_row += 1

        self.min_guidance_spin = QtWidgets.QDoubleSpinBox()
        self.min_guidance_spin.setRange(0.0, 30.0)
        self.min_guidance_spin.setDecimals(1)
        self.min_guidance_spin.setSingleStep(0.5)
        self.min_guidance_spin.setToolTip(
            "Minimum guidance scale used when guidance is decayed over time."
        )
        adv_diff_layout.addWidget(QtWidgets.QLabel("Min guidance:"), adv_row, 0)
        adv_diff_layout.addWidget(self.min_guidance_spin, adv_row, 1)
        adv_row += 1

        self.guidance_interval_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_interval_spin.setRange(0.1, 5.0)
        self.guidance_interval_spin.setDecimals(2)
        self.guidance_interval_spin.setSingleStep(0.1)
        self.guidance_interval_spin.setToolTip(
            "How often ACE re-applies text/ERG guidance during diffusion. "
            "Lower = tighter to the prompt, higher = looser and more free."
        )
        adv_diff_layout.addWidget(QtWidgets.QLabel("Guidance interval:"), adv_row, 0)
        adv_diff_layout.addWidget(self.guidance_interval_spin, adv_row, 1)
        adv_row += 1

        self.guidance_decay_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_decay_spin.setRange(0.0, 1.0)
        self.guidance_decay_spin.setDecimals(2)
        self.guidance_decay_spin.setSingleStep(0.05)
        self.guidance_decay_spin.setToolTip(
            "Decay factor for guidance over time. 0 = constant guidance; "
            "small values let the model relax later in the generation."
        )
        adv_diff_layout.addWidget(QtWidgets.QLabel("Guidance decay:"), adv_row, 0)
        adv_diff_layout.addWidget(self.guidance_decay_spin, adv_row, 1)

        layout.addLayout(adv_diff_layout)

        # Advanced: ERG toggles
        erg_layout = QtWidgets.QHBoxLayout()
        self.erg_tag_check = QtWidgets.QCheckBox("ERG on tags")
        self.erg_tag_check.setToolTip("Use ERG when interpreting style / tag text.")
        self.erg_lyric_check = QtWidgets.QCheckBox("ERG on lyrics")
        self.erg_lyric_check.setToolTip("Use ERG to align audio to lyric rhythm.")
        self.erg_diffusion_check = QtWidgets.QCheckBox("ERG on diffusion")
        self.erg_diffusion_check.setToolTip("Embed ERG signals directly into diffusion for tighter grooves.")
        erg_layout.addWidget(self.erg_tag_check)
        erg_layout.addWidget(self.erg_lyric_check)
        erg_layout.addWidget(self.erg_diffusion_check)
        erg_layout.addStretch(1)
        layout.addLayout(erg_layout)

        # Advanced: OSS steps
        oss_layout = QtWidgets.QHBoxLayout()
        self.oss_steps_edit = QtWidgets.QLineEdit()
        self.oss_steps_edit.setPlaceholderText("Optional: comma-separated step numbers, e.g. 10, 20, 40")
        self.oss_steps_edit.setToolTip(
            "Experimental override for ACE OSS steps. Leave blank unless you know what you want."
        )
        oss_layout.addWidget(QtWidgets.QLabel("OSS steps:"))
        oss_layout.addWidget(self.oss_steps_edit)
        layout.addLayout(oss_layout)

        # Signals
        self.btn_generate.clicked.connect(self.generate_clicked)
        self.btn_play_last.clicked.connect(self._on_play_last_clicked)
        self.btn_open_folder.clicked.connect(self._on_view_results_clicked)
        self.genre_combo.currentTextChanged.connect(self._on_genre_changed)
        self.preset_list.currentItemChanged.connect(self._on_preset_item_changed)
        self.preset_list.customContextMenuRequested.connect(self._on_preset_context_menu)
        self.ref_audio_browse_btn.clicked.connect(self._browse_ref_audio)
        self.ref_strength_slider.valueChanged.connect(self._on_ref_strength_changed)

    def _load_config_into_ui(self):
        # Avoid preset callbacks firing while we restore values
        if hasattr(self, "genre_combo"):
            self.genre_combo.blockSignals(True)
        try:
            self.prompt_edit.setText(self.config.get("prompt", ""))
            if hasattr(self, "negative_edit"):
                self.negative_edit.setText(self.config.get("negative_prompt", ""))
            if hasattr(self, "track_name_edit"):
                self.track_name_edit.setText(self.config.get("track_name", ""))
            self.lyrics_edit.setPlainText(self.config.get("lyrics", ""))

            self.duration_spin.setValue(int(self.config.get("audio_duration", 60)))
            self.steps_spin.setValue(int(self.config.get("infer_step", 60)))
            self.guidance_spin.setValue(float(self.config.get("guidance_scale", 15.0)))

            self.guidance_text_spin.setValue(
                float(self.config.get("guidance_scale_text", 5.0))
            )
            self.guidance_lyric_spin.setValue(
                float(self.config.get("guidance_scale_lyric", 1.5))
            )

            # Advanced settings
            try:
                self.omega_spin.setValue(float(self.config.get("omega_scale", 10.0)))
                self.min_guidance_spin.setValue(float(self.config.get("min_guidance_scale", 3.0)))
                self.guidance_interval_spin.setValue(float(self.config.get("guidance_interval", 0.5)))
                self.guidance_decay_spin.setValue(float(self.config.get("guidance_interval_decay", 0.0)))

                self.erg_tag_check.setChecked(bool(self.config.get("use_erg_tag", True)))
                self.erg_lyric_check.setChecked(bool(self.config.get("use_erg_lyric", False)))
                self.erg_diffusion_check.setChecked(bool(self.config.get("use_erg_diffusion", True)))

                oss_steps_val = self.config.get("oss_steps", [])
                if isinstance(oss_steps_val, str):
                    oss_text = oss_steps_val
                elif isinstance(oss_steps_val, (list, tuple)):
                    oss_text = ", ".join(str(x) for x in oss_steps_val)
                else:
                    oss_text = ""
                self.oss_steps_edit.setText(oss_text)
            except Exception:
                pass

            # Reference audio config
            ref_audio_input = self.config.get("ref_audio_input", "")
            if ref_audio_input is None:
                ref_audio_input = ""
            if hasattr(self, "ref_audio_edit"):
                self.ref_audio_edit.setText(ref_audio_input)

            ref_strength = float(self.config.get("ref_audio_strength", 0.5))
            if ref_strength < 0.0:
                ref_strength = 0.0
            if ref_strength > 1.0:
                ref_strength = 1.0
            if hasattr(self, "ref_strength_slider"):
                self.ref_strength_slider.setValue(int(ref_strength * 100))
                self._on_ref_strength_changed(self.ref_strength_slider.value())

            scheduler = self.config.get("scheduler_type", "euler")
            idx = self.scheduler_combo.findText(scheduler)
            if idx >= 0:
                self.scheduler_combo.setCurrentIndex(idx)

            cfg_type = self.config.get("cfg_type", "apg")
            idx = self.cfg_combo.findText(cfg_type)
            if idx >= 0:
                self.cfg_combo.setCurrentIndex(idx)

            # Restore last-used preset selection
            selected_preset_name = self.config.get("preset_name")
            if not selected_preset_name:
                selected_preset_name = self.config.get(
                    "genre_preset", "Custom (manual tags)"
                )

            saved_genre_filter = self.config.get("preset_genre_filter", "All")
            if saved_genre_filter not in self.preset_manager.genres:
                saved_genre_filter = "All"
            if hasattr(self, "genre_combo"):
                idx = self.genre_combo.findText(saved_genre_filter)
                if idx >= 0:
                    self.genre_combo.setCurrentIndex(idx)
        finally:
            if hasattr(self, "genre_combo"):
                self.genre_combo.blockSignals(False)

        # Populate preset list after genre filter has been restored
        self._refresh_preset_list(select_name=selected_preset_name)

        self.seed_spin.setValue(0)
        self.output_edit.setText(self.config.get("output_folder", "output/ace"))


    def _store_config_from_ui(self):
        self.config.set("prompt", self.prompt_edit.text())
        if hasattr(self, "track_name_edit"):
            self.config.set("track_name", self.track_name_edit.text().strip())
        if hasattr(self, "negative_edit"):
            self.config.set("negative_prompt", self.negative_edit.text())
        self.config.set("lyrics", self.lyrics_edit.toPlainText())
        self.config.set("audio_duration", float(self.duration_spin.value()))
        self.config.set("infer_step", int(self.steps_spin.value()))
        self.config.set("guidance_scale", float(self.guidance_spin.value()))
        self.config.set("guidance_scale_text", float(self.guidance_text_spin.value()))
        self.config.set("guidance_scale_lyric", float(self.guidance_lyric_spin.value()))
        self.config.set("scheduler_type", self.scheduler_combo.currentText())
        self.config.set("cfg_type", self.cfg_combo.currentText())

        # Persist preset selection
        preset_name = ""
        if hasattr(self, "preset_list"):
            current_item = self.preset_list.currentItem()
            if current_item is not None:
                preset_name = (
                    current_item.data(QtCore.Qt.UserRole)
                    or current_item.text().lstrip("★ ").strip()
                )
        self.config.set("preset_name", preset_name)
        if hasattr(self, "genre_combo"):
            self.config.set("preset_genre_filter", self.genre_combo.currentText())
        # Legacy key for backwards compatibility
        self.config.set("genre_preset", preset_name)

        self.config.set("output_folder", self.output_edit.text())

        # Reference audio config
        if hasattr(self, "ref_audio_edit"):
            ref_audio_input = self.ref_audio_edit.text().strip()
        else:
            ref_audio_input = ""
        self.config.set("ref_audio_input", ref_audio_input)
        if hasattr(self, "ref_strength_slider"):
            self.config.set(
                "ref_audio_strength", float(self.ref_strength_slider.value()) / 100.0
            )

        # Advanced settings
        try:
            if hasattr(self, "omega_spin"):
                self.config.set("omega_scale", float(self.omega_spin.value()))
            if hasattr(self, "min_guidance_spin"):
                self.config.set("min_guidance_scale", float(self.min_guidance_spin.value()))
            if hasattr(self, "guidance_interval_spin"):
                self.config.set("guidance_interval", float(self.guidance_interval_spin.value()))
            if hasattr(self, "guidance_decay_spin"):
                self.config.set("guidance_interval_decay", float(self.guidance_decay_spin.value()))
            if hasattr(self, "erg_tag_check"):
                self.config.set("use_erg_tag", bool(self.erg_tag_check.isChecked()))
            if hasattr(self, "erg_lyric_check"):
                self.config.set("use_erg_lyric", bool(self.erg_lyric_check.isChecked()))
            if hasattr(self, "erg_diffusion_check"):
                self.config.set("use_erg_diffusion", bool(self.erg_diffusion_check.isChecked()))
            if hasattr(self, "oss_steps_edit"):
                raw = self.oss_steps_edit.text().strip()
                steps_list = []
                if raw:
                    for part in raw.split(","):
                        p = part.strip()
                        if not p:
                            continue
                        try:
                            steps_list.append(int(p))
                        except Exception:
                            # Ignore tokens that are not valid integers
                            continue
                self.config.set("oss_steps", steps_list)
        except Exception:
            pass

        self.config.save()

    def _collect_ui_state(self):
        if hasattr(self, "ref_audio_edit"):
            ref_audio_input = self.ref_audio_edit.text().strip()
        else:
            ref_audio_input = ""
        if hasattr(self, "ref_strength_slider"):
            ref_strength = float(self.ref_strength_slider.value()) / 100.0
        else:
            ref_strength = 0.5
        if ref_strength < 0.0:
            ref_strength = 0.0
        if ref_strength > 1.0:
            ref_strength = 1.0
        return {
            "prompt": self.prompt_edit.text(),
            "negative_prompt": self.negative_edit.text() if hasattr(self, "negative_edit") else "",
            "track_name": self.track_name_edit.text().strip() if hasattr(self, "track_name_edit") else "",
            "lyrics": self.lyrics_edit.toPlainText(),
            "audio_duration": float(self.duration_spin.value()),
            "infer_step": int(self.steps_spin.value()),
            "guidance_scale": float(self.guidance_spin.value()),
            "scheduler_type": self.scheduler_combo.currentText(),
            "cfg_type": self.cfg_combo.currentText(),
            "seed": int(self.seed_spin.value()),
            "output_rel": self.output_edit.text().strip() or "output/ace",
            "ref_audio_input": ref_audio_input,
            "ref_audio_strength": ref_strength,
            "audio2audio_enable": bool(ref_audio_input),
        }

    def _browse_ref_audio(self):
        start_dir = str(self.root_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select reference audio",
            start_dir,
            "Audio files (*.wav *.mp3 *.flac *.ogg);;All files (*)",
        )
        if path:
            try:
                root_str = str(self.root_dir.resolve())
                abs_path = Path(path).resolve()
                if str(abs_path).startswith(root_str):
                    rel = os.path.relpath(str(abs_path), root_str)
                    self.ref_audio_edit.setText(rel)
                else:
                    self.ref_audio_edit.setText(str(abs_path))
            except Exception:
                self.ref_audio_edit.setText(path)

    def _on_ref_strength_changed(self, value: int):
        if hasattr(self, "ref_strength_label"):
            self.ref_strength_label.setText(f"{value}%")


    def _ensure_ace_models_installed(self) -> bool:
        """Safety check: ensure ACE-Step is installed before starting generation."""
        try:
            repo_root = (self.root_dir / ".ace_env" / "ACE-Step")
            legacy_repo_root = (
                self.root_dir / "presets" / "extra_env" / ".ace_env" / "ACE-Step"
            )
            if repo_root.exists() or legacy_repo_root.exists():
                return True
        except Exception:
            pass

        # Inform the user and guide them to Optional Downloads.
        try:
            from PySide6.QtWidgets import QMessageBox
            msg = (
                "Models are not installed yet, please select 'Ace step music' model from the "
                "'optional downloads' menu to download the correct model for this tool ."
            )
            QMessageBox.information(self, "ACE-Step", msg)
        except Exception:
            pass

        # Open the Optional Installs script (helpers/opt_installs.py) after the user clicks OK.
        try:
            opt_file = (self.root_dir / "helpers" / "opt_installs.py").resolve()
            if opt_file.exists():
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(opt_file)))
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "ACE-Step", f"Could not find: {opt_file}")
        except Exception:
            pass

        return False

    def generate_clicked(self):
        """Start ACE generation – prefer background queue, fall back to in-process worker."""
        # Safety: make sure the ACE-Step model environment exists before doing anything.
        if not self._ensure_ace_models_installed():
            return

        # First, persist the current UI settings
        self._store_config_from_ui()
        ui_state = self._collect_ui_state()

        # Try to enqueue in the background worker queue using helpers.queue_adapter.
        enqueue_ok = False
        try:
            try:
                from helpers.queue_adapter import enqueue_ace_from_widget
            except Exception:
                # Fallback: try local import if running helpers/ace.py directly
                try:
                    import queue_adapter as _qa  # type: ignore
                    enqueue_ace_from_widget = _qa.enqueue_ace_from_widget  # type: ignore
                except Exception:
                    enqueue_ace_from_widget = None  # type: ignore
            if enqueue_ace_from_widget is not None:  # type: ignore
                enqueue_ok = bool(enqueue_ace_from_widget(self))  # type: ignore
        except Exception as e:
            # If queue enqueue fails for any reason, we fall back to the legacy in-process worker.
            # You can inspect this in the console if needed.
            print('[ACE] enqueue_ace_from_widget failed, falling back to direct run:', e)

        if enqueue_ok:
            # Queue path: inform the user and do not run ACE inline
            self.status_label.setText(
                "Status: ACE job enqueued in background worker (see Queue tab)."
            )
            return

        # Fallback: run ACE directly in a QThread (legacy behaviour)
        self.status_label.setText("Status: preparing pipeline...")

        self.worker = AceWorker(self.root_dir, self.config, ui_state, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_with_result.connect(self._on_generation_success)
        self.worker.failed.connect(self._on_generation_failed)
        self.worker.start()

    def _refresh_preset_list(self, select_name=None):
        """Refresh the preset list widget based on the current genre filter.

        Favorites are always shown at the top (both per-genre and in 'All').
        """
        if not hasattr(self, "preset_list"):
            return

        self.preset_list.blockSignals(True)
        try:
            self.preset_list.clear()
            genre = (
                self.genre_combo.currentText()
                if hasattr(self, "genre_combo")
                else "All"
            )
            presets = self.preset_manager.get_presets_for_genre(genre)

            for preset in presets:
                name = preset.get("name", "")
                if not name:
                    continue
                is_fav = self.preset_manager.is_favorite(name)
                label = f"★ {name}" if is_fav else name
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.UserRole, name)
                self.preset_list.addItem(item)

            # Decide which preset to select
            target_name = select_name or self.config.get("preset_name")
            selected_item = None
            if target_name:
                for i in range(self.preset_list.count()):
                    it = self.preset_list.item(i)
                    if it.data(QtCore.Qt.UserRole) == target_name:
                        selected_item = it
                        break
            if selected_item is None and self.preset_list.count() > 0:
                selected_item = self.preset_list.item(0)

            if selected_item is not None:
                self.preset_list.setCurrentItem(selected_item)
        finally:
            self.preset_list.blockSignals(False)

        # Apply the selected preset (if any)
        current = self.preset_list.currentItem()
        if current is not None:
            preset_name = (
                current.data(QtCore.Qt.UserRole)
                or current.text().lstrip("★ ").strip()
            )
            preset = self.preset_manager.get_preset_by_name(preset_name)
            if preset:
                self._apply_preset(preset)

    def _on_genre_changed(self, genre_text: str):
        # When the genre filter changes, rebuild the list but keep the same preset if possible
        current_name = ""
        if hasattr(self, "preset_list"):
            current_item = self.preset_list.currentItem()
            if current_item is not None:
                current_name = current_item.data(QtCore.Qt.UserRole) or ""
        self._refresh_preset_list(select_name=current_name or None)

    def _on_preset_item_changed(self, current, previous):
        if current is None:
            return
        preset_name = (
            current.data(QtCore.Qt.UserRole)
            or current.text().lstrip("★ ").strip()
        )
        preset = self.preset_manager.get_preset_by_name(preset_name)
        if not preset:
            return
        self._apply_preset(preset)
        # Keep config in sync so the last-used preset is remembered
        self.config.set("preset_name", preset_name)
        if hasattr(self, "genre_combo"):
            self.config.set("preset_genre_filter", self.genre_combo.currentText())
        self.config.set("genre_preset", preset_name)
        self.config.save()

    def _on_preset_context_menu(self, pos):
        if not hasattr(self, "preset_list"):
            return
        item = self.preset_list.itemAt(pos)
        if item is None:
            return
        preset_name = (
            item.data(QtCore.Qt.UserRole)
            or item.text().lstrip("★ ").strip()
        )
        if not preset_name:
            return

        menu = QtWidgets.QMenu(self)
        is_fav = self.preset_manager.is_favorite(preset_name)
        toggle_text = "Remove from favorites" if is_fav else "Add to favorites"
        toggle_action = menu.addAction(toggle_text)

        chosen = menu.exec_(self.preset_list.mapToGlobal(pos))
        if chosen == toggle_action:
            self.preset_manager.toggle_favorite(preset_name)
            # Rebuild list so the updated favorite ordering is visible
            self._refresh_preset_list(select_name=preset_name)

    def _apply_preset(self, preset: dict):
        # Fill tags from the JSON-backed presets
        prompt = preset.get("prompt", "")
        if prompt is not None:
            self.prompt_edit.setText(str(prompt))

        # Fill negative tags from preset (if provided)
        neg_prompt = preset.get("negative_prompt")
        if hasattr(self, "negative_edit") and neg_prompt is not None:
            self.negative_edit.setText(str(neg_prompt))

        # Optional: also adjust core generation parameters for certain presets
        params = preset.get("params") or {}
        if not isinstance(params, dict):
            return

        # Duration
        if "audio_duration" in params:
            try:
                self.duration_spin.setValue(int(params["audio_duration"]))
            except Exception:
                pass

        # Diffusion steps
        if "infer_step" in params:
            try:
                self.steps_spin.setValue(int(params["infer_step"]))
            except Exception:
                pass

        # Main guidance scale
        if "guidance_scale" in params:
            try:
                self.guidance_spin.setValue(float(params["guidance_scale"]))
            except Exception:
                pass

        # Scheduler + CFG type (only if the requested value exists in the combos)
        if "scheduler_type" in params:
            idx = self.scheduler_combo.findText(str(params["scheduler_type"]))
            if idx >= 0:
                self.scheduler_combo.setCurrentIndex(idx)

        if "cfg_type" in params:
            idx = self.cfg_combo.findText(str(params["cfg_type"]))
            if idx >= 0:
                self.cfg_combo.setCurrentIndex(idx)

        # Optional extended guidance fields
        if "guidance_scale_text" in params:
            try:
                self.guidance_text_spin.setValue(float(params["guidance_scale_text"]))
            except Exception:
                pass
        if "guidance_scale_lyric" in params:
            try:
                self.guidance_lyric_spin.setValue(float(params["guidance_scale_lyric"]))
            except Exception:
                pass
        if "ref_audio_strength" in params and hasattr(self, "ref_strength_slider"):
            try:
                v = float(params["ref_audio_strength"])
                v = max(0.0, min(1.0, v))
                self.ref_strength_slider.setValue(int(v * 100))
            except Exception:
                pass

    def _on_progress(self, msg: str):
        self.status_label.setText(f"Status: {msg}")

    def _on_generation_success(self, out_path: str):
        # Remember last output path so the 'Play last result' button can use it
        try:
            if out_path:
                from pathlib import Path as _P
                p = _P(str(out_path))
                self.last_output_path = str(p)
        except Exception:
            self.last_output_path = out_path or None
        self.status_label.setText(f"Status: done! Output: {out_path}")



    def _on_play_last_clicked(self):
        """Play the most recently generated track in the main FrameVision player."""
        # Prefer the exact path reported by the worker, if we still have it
        path = None
        try:
            if getattr(self, "last_output_path", None):
                from pathlib import Path as _P
                p = _P(str(self.last_output_path))
                if p.exists():
                    path = p
        except Exception:
            path = None

        # Fallback: scan the configured output folder for the newest .wav
        if path is None:
            try:
                out_rel = self.output_edit.text().strip() or "output/ace"
            except Exception:
                out_rel = "output/ace"
            try:
                out_dir = (self.root_dir / out_rel).resolve()
            except Exception:
                out_dir = self.root_dir / "output" / "ace"
            candidates = []
            try:
                for p in out_dir.glob("*.wav"):
                    try:
                        candidates.append((p.stat().st_mtime, p))
                    except Exception:
                        continue
            except Exception:
                candidates = []
            if candidates:
                candidates.sort(key=lambda t: t[0], reverse=True)
                path = candidates[0][1]

        if path is None:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "ACE", "No generated track found to play yet. Generate one first.")
            except Exception:
                pass
            return

        # Hand off to the main window's VideoPane (unified image/video/audio player)
        main = self.window() if hasattr(self, "window") else None
        try:
            from pathlib import Path as _P
            if main is not None and hasattr(main, "video"):
                main.current_path = _P(str(path))
                try:
                    main.video.open(main.current_path)
                except Exception:
                    # As a last resort, try direct QUrl open if available
                    try:
                        from PySide6.QtCore import QUrl
                        main.video.player.setSource(QUrl.fromLocalFile(str(path)))
                        main.video.player.play()
                    except Exception:
                        raise
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "ACE", f"Could not send track to player:\n{e}")
            except Exception:
                pass

    def _on_view_results_clicked(self):
        """Open ACE results in Media Explorer (and trigger a scan)."""
        # Prefer the exact file we generated last (if we have it), otherwise fall back
        # to the configured output folder (relative to FrameVision root by default).
        out_dir = None
        try:
            if getattr(self, 'last_output_path', None):
                p = Path(str(self.last_output_path)).expanduser()
                if p.exists():
                    out_dir = p.parent
        except Exception:
            out_dir = None

        if out_dir is None:
            try:
                out_rel = self.output_edit.text().strip() or 'output/ace'
            except Exception:
                out_rel = 'output/ace'
            try:
                p = Path(out_rel).expanduser()
                out_dir = p if p.is_absolute() else (self.root_dir / p).resolve()
            except Exception:
                out_dir = (self.root_dir / 'output' / 'ace').resolve()

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Ask the main window to switch to Media Explorer and scan this folder.
        main = self.window() if hasattr(self, 'window') else None
        if main is not None and hasattr(main, 'open_media_explorer_folder'):
            try:
                # Newer signature: open_media_explorer_folder(folder, **options)
                main.open_media_explorer_folder(str(out_dir))
                return
            except TypeError:
                # Older signature fallback (if the method expects different params)
                try:
                    main.open_media_explorer_folder(str(out_dir), None)
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: open the folder in the OS file browser.
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
        except Exception:
            try:
                self.status_label.setText(f'Status: output folder: {out_dir}')
            except Exception:
                pass


    def _on_generation_failed(self, message: str):
        self.status_label.setText(f"Status: generation failed: {message}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AceUI()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# FrameVision integration alias: main app expects 'acePane' from helpers.ace
acePane = AceUI