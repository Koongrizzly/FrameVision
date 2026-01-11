"""
SongGeneration (LeVo) controller UI (PySide6)

- Uses a dedicated venv in:      .song_g_env/
- Expects repo at:               models/song_generation/
- Saves UI settings to:          presets/setsave/song_g.json
- Writes outputs to:             output/music/song_g/

This file can run standalone:
    python helpers/song_g.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import shlex
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

GENERATE_TYPES = [
    ("Mixed (vocals + accompaniment)", "mixed"),
    ("BGM only (instrumental)", "bgm"),
    ("Vocal only (a cappella)", "vocal"),
    ("Separate (vocal + accompaniment tracks)", "separate"),
]

DOWNLOAD_OPTIONS = [
    ("Tencent base pack (ckpt + third_party, ~16GB)", "tencent_base"),
    ("Runtime pack (ckpt + third_party)", "runtime"),
    ("Checkpoint: large (model.pt + config)", "ckpt:large"),
    ("Checkpoint: base-full (model.pt + config)", "ckpt:base-full"),
    ("Checkpoint: base-new (model.pt + config)", "ckpt:base-new"),
    ("Checkpoint: base (model.pt + config)", "ckpt:base"),
]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path, fallback: dict) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return dict(fallback)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _norm(p: str) -> str:
    return p.replace("\\", "/") if p else p


@dataclass
class SongGPaths:
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
    def default_repo(self) -> Path:
        return self.root / "models" / "song_generation"

    @property
    def default_output(self) -> Path:
        return self.root / "output" / "music" / "song_g"

    @property
    def download_script(self) -> Path:
        return self.root / "helpers" / "song_g_download.py"


class LogBox(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(8000)
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
        self.paths = SongGPaths(_root_dir())
        self.defaults = _load_json(self.paths.settings_json, {})
        self.settings = dict(self.defaults)

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_out)
        self.proc.finished.connect(self._on_proc_finished)

        self.dl_proc = QtCore.QProcess(self)
        self.dl_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.dl_proc.readyReadStandardOutput.connect(self._on_dl_out)
        self.dl_proc.finished.connect(self._on_dl_finished)

        self._build_ui()
        self._load_into_ui()
        self._refresh_status()

        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave_if_dirty)
        self._autosave_timer.start()
        self._dirty = False

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

        self.btn_open_repo = QtWidgets.QPushButton("Open repo folder")
        self.btn_open_repo.clicked.connect(lambda: self._open_folder(self._repo_path()))
        top.addWidget(self.btn_open_repo)

        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(lambda: self._open_folder(self._output_dir()))
        top.addWidget(self.btn_open_out)

        layout.addLayout(top)

        grid = QtWidgets.QGridLayout()
        row = 0

        grid.addWidget(QtWidgets.QLabel("Repo path"), row, 0)
        self.repo_path = QtWidgets.QLineEdit()
        self.repo_browse = QtWidgets.QPushButton("Browse")
        self.repo_browse.clicked.connect(self._browse_repo)
        grid.addWidget(self.repo_path, row, 1)
        grid.addWidget(self.repo_browse, row, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Checkpoint folder (contains config.yaml + model.pt)"), row, 0)
        self.ckpt_path = QtWidgets.QLineEdit()
        self.ckpt_browse = QtWidgets.QPushButton("Browse")
        self.ckpt_browse.clicked.connect(self._browse_ckpt)
        grid.addWidget(self.ckpt_path, row, 1)
        grid.addWidget(self.ckpt_browse, row, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Output folder"), row, 0)
        self.output_dir = QtWidgets.QLineEdit()
        self.out_browse = QtWidgets.QPushButton("Browse")
        self.out_browse.clicked.connect(self._browse_out)
        grid.addWidget(self.output_dir, row, 1)
        grid.addWidget(self.out_browse, row, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Song id (idx)"), row, 0)
        self.idx_edit = QtWidgets.QLineEdit()
        grid.addWidget(self.idx_edit, row, 1, 1, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Generate type"), row, 0)
        self.gen_type = QtWidgets.QComboBox()
        for label, val in GENERATE_TYPES:
            self.gen_type.addItem(label, val)
        grid.addWidget(self.gen_type, row, 1, 1, 2)
        row += 1

        self.low_mem = QtWidgets.QCheckBox("Low memory mode (--low_mem)")
        self.use_flash = QtWidgets.QCheckBox("Use Flash Attention (--use_flash_attn) if installed")
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(self.low_mem)
        opts.addWidget(self.use_flash)
        opts.addStretch(1)
        grid.addLayout(opts, row, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Options"), row, 0)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Auto prompt audio type"), row, 0)
        self.auto_prompt = QtWidgets.QComboBox()
        self.auto_prompt.addItems(AUTO_PROMPT_TYPES)
        grid.addWidget(self.auto_prompt, row, 1, 1, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Prompt audio (10s ref, optional)"), row, 0)
        self.prompt_audio = QtWidgets.QLineEdit()
        self.prompt_browse = QtWidgets.QPushButton("Browse")
        self.prompt_browse.clicked.connect(self._browse_prompt_audio)
        self.prompt_clear = QtWidgets.QPushButton("Clear")
        self.prompt_clear.clicked.connect(lambda: self.prompt_audio.setText(""))
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.prompt_audio, 1)
        h.addWidget(self.prompt_browse)
        h.addWidget(self.prompt_clear)
        grid.addLayout(h, row, 1, 1, 2)
        row += 1

        layout.addLayout(grid)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        desc_box = QtWidgets.QWidget()
        desc_layout = QtWidgets.QVBoxLayout(desc_box)
        desc_layout.addWidget(QtWidgets.QLabel("Descriptions (optional)"))
        self.desc_edit = QtWidgets.QPlainTextEdit()
        self.desc_edit.setPlaceholderText("e.g. female, dark, pop, sad, piano and drums, the bpm is 120")
        desc_layout.addWidget(self.desc_edit, 1)
        split.addWidget(desc_box)

        lyr_box = QtWidgets.QWidget()
        lyr_layout = QtWidgets.QVBoxLayout(lyr_box)
        lyr_layout.addWidget(QtWidgets.QLabel("Lyrics (gt_lyric) — sections separated by ';'"))
        self.lyr_edit = QtWidgets.QPlainTextEdit()
        lyr_layout.addWidget(self.lyr_edit, 1)
        split.addWidget(lyr_box)

        split.setSizes([120, 220])
        layout.addWidget(split, 2)

        btns = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate")
        self.btn_generate.clicked.connect(self._generate)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)

        self.open_on_finish = QtWidgets.QCheckBox("Open output folder when finished")
        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        btns.addWidget(self.open_on_finish)
        layout.addLayout(btns)

        self.log = LogBox()
        layout.addWidget(self.log, 2)

        for w in [self.repo_path, self.ckpt_path, self.output_dir, self.idx_edit, self.prompt_audio]:
            w.textChanged.connect(self._mark_dirty)
        self.gen_type.currentIndexChanged.connect(self._mark_dirty)
        self.auto_prompt.currentIndexChanged.connect(self._mark_dirty)
        self.low_mem.stateChanged.connect(self._mark_dirty)
        self.use_flash.stateChanged.connect(self._mark_dirty)
        self.desc_edit.textChanged.connect(self._mark_dirty)
        self.lyr_edit.textChanged.connect(self._mark_dirty)
        self.open_on_finish.stateChanged.connect(self._mark_dirty)

    def _mark_dirty(self):
        self._dirty = True

    def _autosave_if_dirty(self):
        if not self._dirty:
            return
        self._dirty = False
        self._save_from_ui()

    def _repo_path(self) -> Path:
        p = self.repo_path.text().strip()
        return (self.paths.root / p).resolve() if p and not Path(p).is_absolute() else Path(p).resolve() if p else self.paths.default_repo

    def _output_dir(self) -> Path:
        p = self.output_dir.text().strip()
        return (self.paths.root / p).resolve() if p and not Path(p).is_absolute() else Path(p).resolve() if p else self.paths.default_output

    def _venv_python(self) -> Path:
        return self.paths.venv_python

    def _save_from_ui(self):
        data = {
            "repo_path": _norm(self.repo_path.text().strip()),
            "ckpt_path": _norm(self.ckpt_path.text().strip()),
            "output_dir": _norm(self.output_dir.text().strip()),
            "idx": self.idx_edit.text().strip(),
            "descriptions": self.desc_edit.toPlainText().strip(),
            "gt_lyric": self.lyr_edit.toPlainText().strip(),
            "prompt_audio_path": _norm(self.prompt_audio.text().strip()),
            "auto_prompt_audio_type": self.auto_prompt.currentText(),
            "generate_type": self.gen_type.currentData(),
            "use_flash_attn": bool(self.use_flash.isChecked()),
            "low_mem": bool(self.low_mem.isChecked()),
            "open_output_on_finish": bool(self.open_on_finish.isChecked()),
        }
        self.settings = data
        _save_json(self.paths.settings_json, data)

    def _load_into_ui(self):
        s = self.settings
        self.repo_path.setText(s.get("repo_path", "models/song_generation"))
        self.ckpt_path.setText(s.get("ckpt_path", ""))
        self.output_dir.setText(s.get("output_dir", "output/music/song_g"))
        self.idx_edit.setText(s.get("idx", "song_001"))
        self.desc_edit.setPlainText(s.get("descriptions", ""))
        self.lyr_edit.setPlainText(s.get("gt_lyric", ""))
        self.prompt_audio.setText(s.get("prompt_audio_path", ""))
        ap = s.get("auto_prompt_audio_type", "Auto")
        self.auto_prompt.setCurrentText(ap if ap in AUTO_PROMPT_TYPES else "Auto")

        gt = s.get("generate_type", "mixed")
        for i in range(self.gen_type.count()):
            if self.gen_type.itemData(i) == gt:
                self.gen_type.setCurrentIndex(i)
                break

        self.use_flash.setChecked(False)
        self.low_mem.setChecked(True)
        self.open_on_finish.setChecked(True)

    def _run_installer(self):
        bat = self.paths.installer_bat
        if not bat.exists():
            QtWidgets.QMessageBox.critical(self, "Missing installer", f"Not found:\n{bat}")
            return
        self.log.add(f"[UI] Launching installer: {bat}")
        try:
            os.startfile(str(bat))
        except Exception:
            import subprocess
            subprocess.Popen(["cmd", "/c", "start", "", str(bat)], cwd=str(self.paths.root))

    def _download_assets(self):
        if self.dl_proc.state() != QtCore.QProcess.NotRunning:
            return

        py = self._venv_python()
        if not py.exists():
            QtWidgets.QMessageBox.critical(self, "Missing env", "The SongGeneration env (.song_g_env) is missing.\nRun the 1-click installer first.")
            return

        script = self.paths.download_script
        if not script.exists():
            QtWidgets.QMessageBox.critical(self, "Missing helper", f"Not found:\n{script}")
            return

        mode = self.dl_combo.currentData()
        args = [str(script)]
        if mode.startswith("ckpt:"):
            args.append("ckpt")
            args += ["--model", mode.split(":", 1)[1]]
        else:
            args.append(mode)

        self.log.add("")
        self.log.add(f"[DL] {py} " + " ".join(shlex.quote(a) for a in args))
        self.btn_download.setEnabled(False)
        self.dl_combo.setEnabled(False)
        self.dl_proc.setWorkingDirectory(str(self.paths.root))
        self.dl_proc.start(str(py), args)

    def _browse_repo(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select SongGeneration repo folder", str(self._repo_path()))
        if d:
            self.repo_path.setText(_norm(os.path.relpath(d, str(self.paths.root))))

    def _browse_ckpt(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select checkpoint folder (config.yaml + model.pt)", str(self.paths.root))
        if d:
            self.ckpt_path.setText(_norm(d))

    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", str(self._output_dir()))
        if d:
            self.output_dir.setText(_norm(os.path.relpath(d, str(self.paths.root))))

    def _browse_prompt_audio(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select prompt audio", str(self.paths.root), "Audio (*.wav *.flac *.mp3 *.m4a *.ogg);;All files (*.*)")
        if f:
            self.prompt_audio.setText(_norm(f))

    def _open_folder(self, folder: Path):
        folder = folder.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception:
            pass

    def _refresh_status(self):
        repo = self._repo_path()
        py = self._venv_python()
        parts = []
        parts.append("venv: OK" if py.exists() else "venv: MISSING (.song_g_env)")
        parts.append("repo: OK" if repo.exists() else "repo: MISSING (models/song_generation)")
        parts.append("assets: OK" if (repo / "third_party").exists() else "assets: MISSING (third_party)")
        self.status_lbl.setText("Status: " + " | ".join(parts))

    def _validate(self) -> tuple[bool, str]:
        self._save_from_ui()
        py = self._venv_python()
        repo = self._repo_path()
        ckpt = Path(self.settings.get("ckpt_path") or "")
        outd = self._output_dir()

        if not py.exists():
            return False, "Env missing (.song_g_env). Run installer."
        if not repo.exists():
            return False, "Repo missing (models/song_generation). Run installer."
        if not ckpt.exists():
            return False, "Checkpoint folder not set."
        if not (ckpt / "config.yaml").exists() or not (ckpt / "model.pt").exists():
            return False, "Checkpoint folder must contain config.yaml and model.pt."
        if not (repo / "third_party").exists():
            return False, "Runtime assets missing: third_party folder. Use Download assets."

        if not self.settings.get("gt_lyric", "").strip():
            return False, "Lyrics is empty."

        outd.mkdir(parents=True, exist_ok=True)
        (outd / "audios").mkdir(parents=True, exist_ok=True)
        (outd / "jsonl").mkdir(parents=True, exist_ok=True)
        return True, ""

    def _generate(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            return

        ok, err = self._validate()
        self._refresh_status()
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Can't start", err)
            return

        repo = self._repo_path()
        py = self._venv_python()
        outd = self._output_dir()
        ckpt = Path(self.settings["ckpt_path"])

        ts = time.strftime("%Y%m%d_%H%M%S")
        idx = self.settings.get("idx", "song_001").strip() or "song_001"
        idx = f"{idx}_{ts}"

        item = {"idx": idx, "gt_lyric": self.settings.get("gt_lyric", "").strip()}

        desc = self.settings.get("descriptions", "").strip()
        if desc:
            item["descriptions"] = desc

        prompt_audio = self.settings.get("prompt_audio_path", "").strip()
        if prompt_audio:
            item["prompt_audio_path"] = prompt_audio
        else:
            ap = self.settings.get("auto_prompt_audio_type", "Auto")
            if ap:
                item["auto_prompt_audio_type"] = ap

        jsonl_path = outd / "jsonl" / f"{idx}.jsonl"
        jsonl_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

        args = [
            str(repo / "generate.py"),
            "--ckpt_path", str(ckpt),
            "--input_jsonl", str(jsonl_path),
            "--save_dir", str(outd),
            "--generate_type", str(self.settings.get("generate_type", "mixed")),
        ]
        if self.settings.get("use_flash_attn", False):
            args.append("--use_flash_attn")
        if self.settings.get("low_mem", False):
            args.append("--low_mem")

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONDONTWRITEBYTECODE", "1")
        env.insert("TRANSFORMERS_CACHE", str(repo / "third_party" / "hub"))

        sep = ";"
        py_paths = [
            str(repo / "codeclm" / "tokenizer"),
            str(repo),
            str(repo / "codeclm" / "tokenizer" / "Flow1dVAE"),
            str(repo / "codeclm" / "tokenizer"),
        ]
        existing = env.value("PYTHONPATH")
        merged = sep.join(py_paths + ([existing] if existing else []))
        env.insert("PYTHONPATH", merged)

        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(repo))

        self.log.add("")
        self.log.add(f"[RUN] {py} " + " ".join(shlex.quote(a) for a in args))
        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.start(str(py), args)

    def _stop(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self.log.add("[UI] Stopping...")
            self.proc.kill()

    def _on_proc_out(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.add(data)

    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        status = "OK" if exit_code == 0 and exit_status == QtCore.QProcess.NormalExit else "FAILED"
        self.log.add(f"[DONE] {status} (exit_code={exit_code})")
        if status == "OK" and self.open_on_finish.isChecked():
            self._open_folder(self._output_dir())
        self._refresh_status()

    def _on_dl_out(self):
        data = bytes(self.dl_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.add(data)

    def _on_dl_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        self.btn_download.setEnabled(True)
        self.dl_combo.setEnabled(True)
        status = "OK" if exit_code == 0 and exit_status == QtCore.QProcess.NormalExit else "FAILED"
        self.log.add(f"[DL DONE] {status} (exit_code={exit_code})")
        self._refresh_status()


class SongGWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SongGeneration (LeVo) Controller")
        self.setMinimumSize(1050, 740)
        self.setCentralWidget(SongGWidget(self))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SongGWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
