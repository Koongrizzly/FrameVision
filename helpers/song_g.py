"""
SongGeneration (LeVo) Controller UI (PySide6)

Layout (app root):
  presets/extra_env/SG_install.bat
  presets/extra_env/SG_req.txt
  helpers/song_g.py
  helpers/song_g_download.py
  presets/setsave/song_g.json
  output/music/song_g/
  .song_g_env/
  models/song_generation/    (git repo + assets + ckpt + third_party)

Important:
- Upstream generate.py is executed with cwd = repo_dir.
- It loads config via: OmegaConf.load(os.path.join(args.ckpt_path, "config.yaml"))
  Therefore args.ckpt_path must be something like:  ckpt/songgeneration_base
  (relative to the repo root), not an absolute path and not only the model name.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtWidgets


AUTO_PROMPT_TYPES = [
    "Auto",
    "Pop",
    "R&B",
    "Dance",
    "Jazz",
    "Folk",
    "Rock",
    "Chinese Style",
    "Chinese Tradition",
    "Metal",
    "Reggae",
    "Chinese Opera",
]

# Display -> internal
GENERATE_MODES = [
    ("Mixed (vocals + accompaniment) [default]", "mixed"),
    ("BGM only (instrumental) --bgm", "bgm"),
    ("Vocal only (a cappella) --vocal", "vocal"),
    ("Separate tracks --separate", "separate"),
]

DOWNLOAD_OPTIONS = [
    ("Tencent base pack (ckpt + third_party, ~16GB)", "tencent_base"),
    ("Runtime pack (ckpt + third_party)", "runtime"),
    ("Checkpoint: large (model.pt + config)", "ckpt:large"),
    ("Checkpoint: base-full (model.pt + config)", "ckpt:base-full"),
    ("Checkpoint: base-new (model.pt + config)", "ckpt:base-new"),
    ("Checkpoint: base (model.pt + config)", "ckpt:base"),
]


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def norm_slashes(p: str) -> str:
    return (p or "").replace("\\", "/")


def load_json(path: Path, fallback: dict) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return dict(fallback)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_abs_path(p: str) -> bool:
    if not p:
        return False
    p = p.strip()
    return (
        (len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"))  # C:\ or C:/
        or p.startswith("\\\\")  # UNC
        or p.startswith("/")  # unix
    )


def ckpt_arg_path(ckpt_input: str, repo_dir: Path) -> tuple[str, Path, str]:
    """
    Accept:
      - full path to .../ckpt/<model_name>
      - relative path like ckpt/<model_name>
      - folder name <model_name>

    Return:
      (model_name, resolved_ckpt_dir, arg_for_cli)
    """
    ckpt_input = (ckpt_input or "").strip().strip('"')
    repo_dir = repo_dir.resolve()
    ckpt_root = repo_dir / "ckpt"

    # Full/absolute directory
    if ckpt_input and (is_abs_path(ckpt_input) or os.path.isdir(ckpt_input)):
        p = Path(ckpt_input)
        if p.exists():
            model_name = p.name.lower()
            resolved = p.resolve()
            try:
                rel = resolved.relative_to(repo_dir)
                arg = rel.as_posix().replace("\\", "/")
            except Exception:
                arg = str(resolved)
            return model_name, resolved, arg

    # Relative path provided
    low = ckpt_input.lower().replace("\\", "/")
    if low.startswith("ckpt/"):
        p = (repo_dir / ckpt_input).resolve()
        model_name = Path(low).name.lower()
        return model_name, p, low

    # Folder name
    model_name = ckpt_input.lower()
    resolved = (ckpt_root / ckpt_input).resolve()
    arg = f"ckpt/{ckpt_input}".replace("\\", "/")
    return model_name, resolved, arg


@dataclass
class Paths:
    root: Path

    @property
    def settings_json(self) -> Path:
        return self.root / "presets" / "setsave" / "song_g.json"

    @property
    def installer_bat(self) -> Path:
        return self.root / "presets" / "extra_env" / "SG_install.bat"

    @property
    def venv_python(self) -> Path:
        return self.root / ".song_g_env" / "Scripts" / "python.exe"

    @property
    def repo_dir_default(self) -> Path:
        return self.root / "models" / "song_generation"

    @property
    def output_dir_default(self) -> Path:
        return self.root / "output" / "music" / "song_g"

    @property
    def downloader_py(self) -> Path:
        return self.root / "helpers" / "song_g_download.py"


class LogBox(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)
        f = self.font()
        f.setFamily("Consolas" if sys.platform.startswith("win") else "Monospace")
        f.setPointSize(max(9, f.pointSize() - 1))
        self.setFont(f)

    def add(self, text: str):
        self.appendPlainText(text.rstrip("\n"))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class SongGWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths = Paths(app_root())
        self.settings = load_json(self.paths.settings_json, {})

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_out)
        self.proc.finished.connect(self._on_proc_done)

        self.dl_proc = QtCore.QProcess(self)
        self.dl_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.dl_proc.readyReadStandardOutput.connect(self._on_dl_out)
        self.dl_proc.finished.connect(self._on_dl_done)

        self._dirty = False
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

        self._build_ui()
        self._load_into_ui()
        self._refresh_status()

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("Status: ...")
        self.status_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.status_lbl, 1)

        self.btn_install = QtWidgets.QPushButton("Run 1-click installer")
        self.btn_install.clicked.connect(self._run_installer)
        top.addWidget(self.btn_install)

        self.dl_combo = QtWidgets.QComboBox()
        for label, val in DOWNLOAD_OPTIONS:
            self.dl_combo.addItem(label, val)
        self.btn_download = QtWidgets.QPushButton("Download assets")
        self.btn_download.clicked.connect(self._download_assets)
        top.addWidget(self.dl_combo)
        top.addWidget(self.btn_download)

        self.btn_open_repo = QtWidgets.QPushButton("Open repo")
        self.btn_open_repo.clicked.connect(lambda: self._open_folder(self.repo_dir()))
        top.addWidget(self.btn_open_repo)

        self.btn_open_out = QtWidgets.QPushButton("Open output")
        self.btn_open_out.clicked.connect(lambda: self._open_folder(self.output_dir()))
        top.addWidget(self.btn_open_out)

        layout.addLayout(top)

        grid = QtWidgets.QGridLayout()
        r = 0

        grid.addWidget(QtWidgets.QLabel("Repo path"), r, 0)
        self.repo_edit = QtWidgets.QLineEdit()
        self.repo_browse = QtWidgets.QPushButton("Browse")
        self.repo_browse.clicked.connect(self._browse_repo)
        grid.addWidget(self.repo_edit, r, 1)
        grid.addWidget(self.repo_browse, r, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Checkpoint (folder name, ckpt/..., or full path)"), r, 0)
        self.ckpt_edit = QtWidgets.QLineEdit()
        self.ckpt_browse = QtWidgets.QPushButton("Browse")
        self.ckpt_browse.clicked.connect(self._browse_ckpt)
        grid.addWidget(self.ckpt_edit, r, 1)
        grid.addWidget(self.ckpt_browse, r, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Output folder"), r, 0)
        self.out_edit = QtWidgets.QLineEdit()
        self.out_browse = QtWidgets.QPushButton("Browse")
        self.out_browse.clicked.connect(self._browse_out)
        grid.addWidget(self.out_edit, r, 1)
        grid.addWidget(self.out_browse, r, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Song id (idx)"), r, 0)
        self.idx_edit = QtWidgets.QLineEdit()
        grid.addWidget(self.idx_edit, r, 1, 1, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Generate mode"), r, 0)
        self.mode_combo = QtWidgets.QComboBox()
        for label, val in GENERATE_MODES:
            self.mode_combo.addItem(label, val)
        grid.addWidget(self.mode_combo, r, 1, 1, 2)
        r += 1

        # Advanced generation params
        grid.addWidget(QtWidgets.QLabel("Seed (0 = random)"), r, 0)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(0)
        grid.addWidget(self.seed_spin, r, 1)

        grid.addWidget(QtWidgets.QLabel("CFG"), r, 2)
        self.cfg_spin = QtWidgets.QDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 50.0)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setSingleStep(0.25)
        self.cfg_spin.setValue(1.5)
        grid.addWidget(self.cfg_spin, r, 3)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Temperature"), r, 0)
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.9)
        grid.addWidget(self.temp_spin, r, 1)

        grid.addWidget(QtWidgets.QLabel("Top-K"), r, 2)
        self.topk_spin = QtWidgets.QSpinBox()
        self.topk_spin.setRange(0, 1000)
        self.topk_spin.setValue(50)
        grid.addWidget(self.topk_spin, r, 3)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Top-P"), r, 0)
        self.topp_spin = QtWidgets.QDoubleSpinBox()
        self.topp_spin.setRange(0.0, 1.0)
        self.topp_spin.setDecimals(3)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(0.0)
        grid.addWidget(self.topp_spin, r, 1)

        self.quiet_chk = QtWidgets.QCheckBox("Quiet warnings")
        self.quiet_chk.setChecked(True)
        grid.addWidget(self.quiet_chk, r, 2, 1, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Auto prompt audio type"), r, 0)
        self.auto_prompt_combo = QtWidgets.QComboBox()
        self.auto_prompt_combo.addItems(AUTO_PROMPT_TYPES)
        grid.addWidget(self.auto_prompt_combo, r, 1, 1, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Prompt audio (10s ref, optional)"), r, 0)
        self.prompt_audio_edit = QtWidgets.QLineEdit()
        self.prompt_browse = QtWidgets.QPushButton("Browse")
        self.prompt_browse.clicked.connect(self._browse_prompt_audio)
        self.prompt_clear = QtWidgets.QPushButton("Clear")
        self.prompt_clear.clicked.connect(lambda: self.prompt_audio_edit.setText(""))
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.prompt_audio_edit, 1)
        h.addWidget(self.prompt_browse)
        h.addWidget(self.prompt_clear)
        grid.addLayout(h, r, 1, 1, 2)
        r += 1

        self.low_mem_chk = QtWidgets.QCheckBox("Low memory mode (--low_mem)")
        self.flash_chk = QtWidgets.QCheckBox("Use Flash Attention (--use_flash_attn)")
        opt = QtWidgets.QHBoxLayout()
        opt.addWidget(self.low_mem_chk)
        opt.addWidget(self.flash_chk)
        opt.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("Options"), r, 0)
        grid.addLayout(opt, r, 1, 1, 2)
        r += 1

        layout.addLayout(grid)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        desc_box = QtWidgets.QWidget()
        desc_l = QtWidgets.QVBoxLayout(desc_box)
        desc_l.addWidget(QtWidgets.QLabel("Descriptions (optional)"))
        self.desc_edit = QtWidgets.QPlainTextEdit()
        self.desc_edit.setPlaceholderText("e.g. female, dark, pop, sad, piano and drums, bpm 120")
        desc_l.addWidget(self.desc_edit, 1)
        split.addWidget(desc_box)

        lyr_box = QtWidgets.QWidget()
        lyr_l = QtWidgets.QVBoxLayout(lyr_box)
        lyr_l.addWidget(QtWidgets.QLabel("Lyrics (gt_lyric) — sections separated by ';'"))
        self.lyr_edit = QtWidgets.QPlainTextEdit()
        lyr_l.addWidget(self.lyr_edit, 1)
        split.addWidget(lyr_box)

        split.setSizes([140, 260])
        layout.addWidget(split, 2)

        btns = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)

        self.open_on_finish = QtWidgets.QCheckBox("Open output folder when finished")
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        btns.addWidget(self.open_on_finish)
        layout.addLayout(btns)

        self.log = LogBox()
        layout.addWidget(self.log, 2)

        # Dirty tracking
        for w in [self.repo_edit, self.ckpt_edit, self.out_edit, self.idx_edit, self.prompt_audio_edit]:
            w.textChanged.connect(self._mark_dirty)
        self.mode_combo.currentIndexChanged.connect(self._mark_dirty)
        self.auto_prompt_combo.currentIndexChanged.connect(self._mark_dirty)
        self.low_mem_chk.stateChanged.connect(self._mark_dirty)
        self.flash_chk.stateChanged.connect(self._mark_dirty)
        self.seed_spin.valueChanged.connect(self._mark_dirty)
        self.cfg_spin.valueChanged.connect(self._mark_dirty)
        self.temp_spin.valueChanged.connect(self._mark_dirty)
        self.topk_spin.valueChanged.connect(self._mark_dirty)
        self.topp_spin.valueChanged.connect(self._mark_dirty)
        self.quiet_chk.stateChanged.connect(self._mark_dirty)
        self.desc_edit.textChanged.connect(self._mark_dirty)
        self.lyr_edit.textChanged.connect(self._mark_dirty)
        self.open_on_finish.stateChanged.connect(self._mark_dirty)

    def _mark_dirty(self):
        self._dirty = True

    def _autosave(self):
        if not self._dirty:
            return
        self._dirty = False
        self._save_from_ui()

    # ---------------- Paths ----------------
    def repo_dir(self) -> Path:
        raw = self.repo_edit.text().strip()
        if raw:
            p = Path(raw)
            if not is_abs_path(raw):
                p = self.paths.root / raw
            return p.resolve()
        return self.paths.repo_dir_default

    def output_dir(self) -> Path:
        raw = self.out_edit.text().strip()
        if raw:
            p = Path(raw)
            if not is_abs_path(raw):
                p = self.paths.root / raw
            return p.resolve()
        return self.paths.output_dir_default

    # ---------------- Settings I/O ----------------
    def _load_into_ui(self):
        s = self.settings
        self.repo_edit.setText(s.get("repo_path", "models/song_generation"))
        self.ckpt_edit.setText(s.get("ckpt_input", "songgeneration_base"))
        self.out_edit.setText(s.get("output_dir", "output/music/song_g"))
        self.idx_edit.setText(s.get("idx", "song_001"))
        self.desc_edit.setPlainText(s.get("descriptions", ""))
        self.lyr_edit.setPlainText(s.get("gt_lyric", ""))
        self.prompt_audio_edit.setText(s.get("prompt_audio_path", ""))

        ap = s.get("auto_prompt_audio_type", "Auto")
        if ap in AUTO_PROMPT_TYPES:
            self.auto_prompt_combo.setCurrentText(ap)

        mode = s.get("mode", "mixed")
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == mode:
                self.mode_combo.setCurrentIndex(i)
                break

        self.low_mem_chk.setChecked(bool(s.get("low_mem", True)))
        self.flash_chk.setChecked(bool(s.get("use_flash_attn", False)))
        self.open_on_finish.setChecked(bool(s.get("open_output_on_finish", True)))
        self.seed_spin.setValue(int(s.get("seed", 0) or 0))
        self.cfg_spin.setValue(float(s.get("cfg_coef", 1.5) or 1.5))
        self.temp_spin.setValue(float(s.get("temperature", 0.9) or 0.9))
        self.topk_spin.setValue(int(s.get("top_k", 50) or 50))
        self.topp_spin.setValue(float(s.get("top_p", 0.0) or 0.0))
        self.quiet_chk.setChecked(bool(s.get("quiet_warnings", True)))


    def _save_from_ui(self):
        data = {
            "repo_path": norm_slashes(self.repo_edit.text().strip()),
            "ckpt_input": norm_slashes(self.ckpt_edit.text().strip()),
            "output_dir": norm_slashes(self.out_edit.text().strip()),
            "idx": self.idx_edit.text().strip(),
            "descriptions": self.desc_edit.toPlainText().strip(),
            "gt_lyric": self.lyr_edit.toPlainText().strip(),
            "prompt_audio_path": norm_slashes(self.prompt_audio_edit.text().strip()),
            "auto_prompt_audio_type": self.auto_prompt_combo.currentText(),
            "mode": self.mode_combo.currentData(),
            "low_mem": bool(self.low_mem_chk.isChecked()),
            "use_flash_attn": bool(self.flash_chk.isChecked()),
            "seed": int(self.seed_spin.value()),
            "cfg_coef": float(self.cfg_spin.value()),
            "temperature": float(self.temp_spin.value()),
            "top_k": int(self.topk_spin.value()),
            "top_p": float(self.topp_spin.value()),
            "quiet_warnings": bool(self.quiet_chk.isChecked()),
            "open_output_on_finish": bool(self.open_on_finish.isChecked()),
        }
        self.settings = data
        save_json(self.paths.settings_json, data)

    # ---------------- Status ----------------
    def _refresh_status(self):
        repo = self.repo_dir()
        vpy = self.paths.venv_python
        parts = []
        parts.append("venv: OK" if vpy.exists() else "venv: MISSING (.song_g_env)")
        parts.append("repo: OK" if repo.exists() else "repo: MISSING")
        parts.append("assets: OK" if (repo / "third_party").exists() else "assets: missing third_party")
        self.status_lbl.setText("Status: " + " | ".join(parts))

    # ---------------- Actions ----------------
    def _run_installer(self):
        bat = self.paths.installer_bat
        if not bat.exists():
            QtWidgets.QMessageBox.critical(self, "Missing installer", f"Not found:\n{bat}")
            return
        self.log.add(f"[UI] Launching installer: {bat}")
        try:
            os.startfile(str(bat))
        except Exception:
            pass

    def _download_assets(self):
        if self.dl_proc.state() != QtCore.QProcess.NotRunning:
            return
        py = self.paths.venv_python
        if not py.exists():
            QtWidgets.QMessageBox.critical(self, "Missing env", "Missing .song_g_env. Run installer first.")
            return
        script = self.paths.downloader_py
        if not script.exists():
            QtWidgets.QMessageBox.critical(self, "Missing downloader", f"Not found:\n{script}")
            return

        mode = self.dl_combo.currentData()
        args = [str(script)]
        if mode.startswith("ckpt:"):
            args += ["ckpt", "--model", mode.split(":", 1)[1]]
        else:
            args.append(mode)

        self.log.add("")
        self.log.add("[DL] " + str(py) + " " + " ".join(shlex.quote(a) for a in args))
        self.btn_download.setEnabled(False)
        self.dl_combo.setEnabled(False)

        self.dl_proc.setWorkingDirectory(str(self.paths.root))
        self.dl_proc.start(str(py), args)

    def _validate(self) -> tuple[bool, str, str, str, Path, str]:
        self._save_from_ui()
        self._refresh_status()

        vpy = self.paths.venv_python
        repo = self.repo_dir()
        if not vpy.exists():
            return False, "Env missing (.song_g_env). Run installer.", "", "", repo, ""
        if not repo.exists():
            return False, "Repo missing (models/song_generation). Run installer.", "", "", repo, ""

        model_name, ckpt_dir, ckpt_arg = ckpt_arg_path(self.settings.get("ckpt_input", ""), repo)
        if not model_name:
            return False, "Checkpoint not set. Use 'songgeneration_base' or browse the folder.", "", "", repo, ""

        cfg = ckpt_dir / "config.yaml"
        mpt = ckpt_dir / "model.pt"
        if not (cfg.exists() and mpt.exists()):
            msg = (
                "Checkpoint folder invalid.\n"
                "Expected:\n"
                f"{cfg}\n"
                f"{mpt}"
            )
            return False, msg, "", "", repo, ""

        if not (repo / "third_party").exists():
            return False, "Missing assets folder: repo/third_party.\nUse Download assets.", "", "", repo, ""

        if not self.settings.get("gt_lyric", "").strip():
            return False, "Lyrics (gt_lyric) is empty.", "", "", repo, ""

        return True, "", model_name, ckpt_arg, repo, str(vpy)

    def _run(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            return

        ok, err, model_name, ckpt_arg, repo, vpy = self._validate()
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Can't start", err)
            return

        outd = self.output_dir()
        outd.mkdir(parents=True, exist_ok=True)
        (outd / "jsonl").mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        idx = (self.settings.get("idx") or "song_001").strip() or "song_001"
        idx = f"{idx}_{ts}"

        raw_lyr = (self.settings.get("gt_lyric", "") or "").strip()
        # If user didn't provide structured tokens, wrap into a verse and ensure sentence punctuation.
        if raw_lyr and "[" not in raw_lyr:
            t = " ".join([x.strip() for x in raw_lyr.replace("\r", "\n").split("\n") if x.strip()])
            if not t.endswith("."):
                t = t + "."
            raw_lyr = "[verse] " + t + " ;"
        item = {"idx": idx, "gt_lyric": raw_lyr}

        desc = self.settings.get("descriptions", "").strip()
        if desc:
            item["descriptions"] = desc

        prompt_audio = self.settings.get("prompt_audio_path", "").strip()
        if prompt_audio:
            item["prompt_audio_path"] = prompt_audio
        else:
            item["auto_prompt_audio_type"] = self.settings.get("auto_prompt_audio_type", "Auto")

        jsonl_path = outd / "jsonl" / f"{idx}.jsonl"
        jsonl_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

        cmd = [
            str(self.paths.root / "helpers" / "song_g_generate_fv.py"),
            "--ckpt_path", ckpt_arg,
            "--input_jsonl", str(jsonl_path),
            "--save_dir", str(outd),
            "--generate_type", (self.settings.get("mode") or "mixed").lower(),
            "--cfg_coef", str(float(self.settings.get("cfg_coef", 1.5))),
            "--temperature", str(float(self.settings.get("temperature", 0.9))),
            "--top_k", str(int(self.settings.get("top_k", 50))),
            "--top_p", str(float(self.settings.get("top_p", 0.0))),
        ]
        seed_val = int(self.settings.get("seed", 0) or 0)
        if seed_val > 0:
            cmd += ["--seed", str(seed_val)]

        if self.settings.get("low_mem", True):
            cmd.append("--low_mem")
        if self.settings.get("use_flash_attn", False):
            cmd.append("--use_flash_attn")

        env = QtCore.QProcessEnvironment.systemEnvironment()

        # HuggingFace cache
        hf_home = repo / "third_party" / "hub"
        hf_home.mkdir(parents=True, exist_ok=True)
        env.insert("HF_HOME", str(hf_home))
        env.remove("TRANSFORMERS_CACHE") if env.contains("TRANSFORMERS_CACHE") else None
        if self.settings.get("quiet_warnings", True):
            env.insert("TRANSFORMERS_VERBOSITY", "error")
            env.insert("PYTHONWARNINGS", ",".join([
                "ignore:Using `TRANSFORMERS_CACHE` is deprecated.*:FutureWarning",
                "ignore:Using `is_flash_attn_available` is deprecated.*:UserWarning",
                "ignore:Special tokens have been added.*:UserWarning",
                "ignore:You are using an old version of the checkpointing format.*:UserWarning",
            ]))

        # Ensure repo is importable
        py_paths = [
            str(repo),
            str(repo / "codeclm" / "tokenizer"),
            str(repo / "codeclm" / "tokenizer" / "Flow1dVAE"),
        ]
        existing = env.value("PYTHONPATH")
        merged = ";".join(py_paths + ([existing] if existing else []))
        env.insert("PYTHONPATH", merged)

        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(repo))

        self.log.add("")
        self.log.add("[RUN] " + vpy + " " + " ".join(shlex.quote(x) for x in cmd))
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.start(vpy, cmd)

    def _stop(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self.log.add("[UI] Stopping...")
            self.proc.kill()

    # ---------------- Browse helpers ----------------
    def _browse_repo(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select SongGeneration repo folder", str(self.repo_dir()))
        if d:
            try:
                rel = os.path.relpath(d, str(self.paths.root))
                self.repo_edit.setText(norm_slashes(rel))
            except Exception:
                self.repo_edit.setText(norm_slashes(d))

    def _browse_ckpt(self):
        start = str(self.repo_dir() / "ckpt")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select checkpoint model folder", start)
        if d:
            self.ckpt_edit.setText(norm_slashes(d))

    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", str(self.output_dir()))
        if d:
            try:
                rel = os.path.relpath(d, str(self.paths.root))
                self.out_edit.setText(norm_slashes(rel))
            except Exception:
                self.out_edit.setText(norm_slashes(d))

    def _browse_prompt_audio(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select prompt audio",
            str(self.paths.root),
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg);;All files (*.*)"
        )
        if f:
            self.prompt_audio_edit.setText(norm_slashes(f))

    def _open_folder(self, folder: Path):
        folder = folder.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception:
            pass

    # ---------------- Process output ----------------
    def _on_proc_out(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.add(data)

    def _on_proc_done(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        ok = (exit_code == 0 and exit_status == QtCore.QProcess.NormalExit)
        self.log.add(f"[DONE] {'OK' if ok else 'FAILED'} (exit_code={exit_code})")
        if ok and self.open_on_finish.isChecked():
            self._open_folder(self.output_dir())
        self._refresh_status()

    def _on_dl_out(self):
        data = bytes(self.dl_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.add(data)

    def _on_dl_done(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        self.btn_download.setEnabled(True)
        self.dl_combo.setEnabled(True)
        ok = (exit_code == 0 and exit_status == QtCore.QProcess.NormalExit)
        self.log.add(f"[DL DONE] {'OK' if ok else 'FAILED'} (exit_code={exit_code})")
        self._refresh_status()


class SongGWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SongGeneration (LeVo) Controller")
        self.setMinimumSize(1100, 760)
        self.setCentralWidget(SongGWidget(self))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SongGWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
