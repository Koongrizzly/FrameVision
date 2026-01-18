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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QVBoxLayout,
    QWidget,
)


def _root_dir() -> Path:
    # helpers/heartmula.py -> helpers -> root
    return Path(__file__).resolve().parent.parent


def _settings_path() -> Path:
    return _root_dir() / "presets" / "setsave" / "mula.json"


def _ensure_dirs() -> None:
    root = _root_dir()
    (root / ".mula_env").mkdir(parents=True, exist_ok=True)
    (root / "models" / "HeartMuLa").mkdir(parents=True, exist_ok=True)
    (root / "output" / "music" / "heartmula").mkdir(parents=True, exist_ok=True)
    (root / "presets" / "setsave").mkdir(parents=True, exist_ok=True)


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

        self.max_len = QSpinBox()
        self.max_len.setRange(1_000, 1_200_000)
        self.max_len.setSingleStep(5_000)
        self.max_len.setValue(int(self.settings.max_audio_length_ms))

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
        self.output_name.setPlaceholderText("Optional filename (blank = auto timestamp)")

        self.lyrics = QPlainTextEdit(self.settings.lyrics_text)
        self.lyrics.setPlaceholderText("Paste lyrics here. Use section headers like [Verse], [Chorus], etc.")

        self.tags = QLineEdit(self.settings.tags_text)
        self.tags.setPlaceholderText("Comma-separated tags, no spaces. Example: piano,happy,wedding,synthesizer,romantic")

        self.btn_generate = QPushButton("Generate")
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
        form.addRow("Max audio length (ms)", self.max_len)
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

        btns = QHBoxLayout()
        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_open_out)
        main.addLayout(btns)

        main.addWidget(QLabel("Log"))
        main.addWidget(self.log, 1)

        self._append_log("Ready. If this is your first time, run presets/extra_env/mula_install.bat")

    # --------------------- helpers ---------------------

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

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

    def _save_settings(self) -> None:
        self.settings.model_path = self.model_path.text().strip() or "models/HeartMuLa"
        self.settings.version = self.version.currentText()
        self.settings.max_audio_length_ms = int(self.max_len.value())
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

    def _generate(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A generation is already running.")
            return

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
        lyrics_path.write_text(self.lyrics.toPlainText().strip() + "\n", encoding="utf-8")
        tags_path.write_text(self.tags.text().strip() + "\n", encoding="utf-8")

        out_dir = (root / self.output_dir.text()).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        name = self.output_name.text().strip()
        if not name:
            name = f"heartmula_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        if not name.lower().endswith(".mp3"):
            name += ".mp3"
        out_path = out_dir / name

        py = str(self._env_python())
        script = str(self._heartlib_script())
        model_dir = str((root / self.model_path.text()).resolve())

        args = [
            script,
            f"--model_path={model_dir}",
            f"--version={self.version.currentText()}",
            f"--lyrics={str(lyrics_path)}",
            f"--tags={str(tags_path)}",
            f"--save_path={str(out_path)}",
            f"--max_audio_length_ms={int(self.max_len.value())}",
            f"--topk={int(self.topk.value())}",
            f"--temperature={float(self.temperature.value())}",
            f"--cfg_scale={float(self.cfg_scale.value())}",
        ]

        self._append_log("\n---")
        self._append_log(f"Python: {py}")
        self._append_log(f"Command: {py} { ' '.join(args) }")

        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Ensure bundled binaries (ffmpeg/ffprobe, etc.) are discoverable by any
        # subprocesses the upstream script may spawn.
        env = QProcessEnvironment.systemEnvironment()
        bin_dir = (root / "presets" / "bin").resolve()
        old_path = env.value("PATH")
        env.insert("PATH", str(bin_dir) + os.pathsep + old_path)
        self.proc.setProcessEnvironment(env)

        self.proc.setWorkingDirectory(str(root))
        self.proc.start(py, args)

        if not self.proc.waitForStarted(5_000):
            self.btn_generate.setEnabled(True)
            self.btn_stop.setEnabled(False)
            QMessageBox.critical(self, "Failed", "Could not start process. Check installer/log.")

    def _stop(self) -> None:
        if self.proc.state() == QProcess.NotRunning:
            return
        self._append_log("Stopping…")
        self.proc.kill()

    def _on_proc_output(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            for line in data.splitlines():
                self._append_log(line)

    def _on_proc_finished(self, exit_code: int, _status) -> None:  # type: ignore[override]
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"Finished (exit code {exit_code}).")


def main() -> int:
    app = QApplication(sys.argv)
    w = HeartMuLaUI()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
