from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QProcess, QTimer, Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
        QVBoxLayout, QWidget
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required to display this installer."
    ) from exc


REPO_URL = "https://github.com/ostris/ai-toolkit.git"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES = [
    "torch==2.9.1",
    "torchvision==0.24.1",
    "torchaudio==2.9.1",
]


def find_app_root() -> Path:
    here = Path(__file__).resolve()
    # Expected: root/presets/extra_env/ostris_install.py
    candidate = here.parents[2]
    if (candidate / "helpers").exists() or (candidate / "presets").exists():
        return candidate
    return Path.cwd().resolve()


APP_ROOT = find_app_root()
MODEL_ROOT = APP_ROOT / "models" / "ostris"
REPO_ROOT = MODEL_ROOT / "ai-toolkit"
ENV_ROOT = APP_ROOT / "environments" / ".ostris"
LOG_ROOT = APP_ROOT / "logs"
SETTINGS_PATH = APP_ROOT / "presets" / "setsave" / "ostris_installer.json"


def conda_executable() -> Optional[str]:
    candidates = [
        shutil.which("conda"),
        str(APP_ROOT / "miniconda3" / "Scripts" / "conda.exe"),
        str(APP_ROOT / "conda" / "Scripts" / "conda.exe"),
        str(APP_ROOT / "installer_files" / "conda" / "Scripts" / "conda.exe"),
    ]
    for item in candidates:
        if item and Path(item).exists():
            return item
    return None


def env_python() -> Path:
    return ENV_ROOT / "python.exe" if os.name == "nt" else ENV_ROOT / "bin" / "python"


def quote_cmd(parts: list[str]) -> str:
    return subprocess.list2cmdline([str(p) for p in parts])


class OstrisInstaller(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LoRA Trainer - Ostris AI Toolkit Installer")
        self.resize(920, 720)
        self.process: Optional[QProcess] = None
        self.queue: list[tuple[str, list[str], Optional[Path]]] = []
        self.log_file = LOG_ROOT / f"ostris_install_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self._build_ui()
        self._load_settings()
        self._refresh_status()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("LoRA Trainer")
        font = QFont()
        font.setPointSize(17)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        credit = QLabel(
            "Standalone hell for Ostris AI Toolkit. "
            "The underlying training engine is developed by Ostris."
        )
        credit.setWordWrap(True)
        layout.addWidget(credit)

        form = QFormLayout()
        self.repo_path = QLineEdit(str(REPO_ROOT))
        self.repo_path.setReadOnly(True)
        self.repo_path.setToolTip("AI Toolkit source code is installed here.")
        form.addRow("Repository:", self.repo_path)

        self.env_path = QLineEdit(str(ENV_ROOT))
        self.env_path.setReadOnly(True)
        self.env_path.setToolTip("Dedicated Conda environment used only by the trainer backend.")
        form.addRow("Environment:", self.env_path)

        self.conda_path = QLineEdit(conda_executable() or "")
        self.conda_path.setPlaceholderText("Conda executable was not detected")
        self.conda_path.setToolTip(
            "Path to conda.exe. The installer checks PATH and common locations."
        )
        form.addRow("Conda:", self.conda_path)
        layout.addLayout(form)

        options = QHBoxLayout()
        self.update_repo = QCheckBox("Update repository when already installed")
        self.update_repo.setChecked(True)
        self.update_repo.setToolTip("Runs git pull and refreshes submodules.")
        options.addWidget(self.update_repo)

        self.install_ui_deps = QCheckBox("Install original web UI dependencies")
        self.install_ui_deps.setChecked(False)
        self.install_ui_deps.setToolTip(
            "Optional. Requires Node.js 20 or newer. The helper does not need Node."
        )
        options.addWidget(self.install_ui_deps)
        options.addStretch(1)
        layout.addLayout(options)

        buttons = QHBoxLayout()
        self.install_btn = QPushButton("Install / Repair")
        self.install_btn.clicked.connect(self.install_or_repair)
        buttons.addWidget(self.install_btn)

        self.verify_btn = QPushButton("Verify")
        self.verify_btn.clicked.connect(self.verify_install)
        buttons.addWidget(self.verify_btn)

        self.launch_btn = QPushButton("Launch Trainer")
        self.launch_btn.clicked.connect(self.launch_helper)
        buttons.addWidget(self.launch_btn)

        self.web_ui_btn = QPushButton("Launch Ostris Web UI")
        self.web_ui_btn.clicked.connect(self.launch_web_ui)
        buttons.addWidget(self.web_ui_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)
        buttons.addWidget(self.stop_btn)
        layout.addLayout(buttons)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Installer output")
        layout.addWidget(self.log, 1)

        self.setCentralWidget(central)

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if data.get("conda"):
            self.conda_path.setText(data["conda"])
        self.install_ui_deps.setChecked(bool(data.get("install_ui_deps", False)))
        self.update_repo.setChecked(bool(data.get("update_repo", True)))

    def _save_settings(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps({
            "conda": self.conda_path.text().strip(),
            "install_ui_deps": self.install_ui_deps.isChecked(),
            "update_repo": self.update_repo.isChecked(),
        }, indent=2), encoding="utf-8")

    def append_log(self, text: str) -> None:
        text = text.rstrip()
        if not text:
            return
        self.log.appendPlainText(text)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text + "\n")

    def _refresh_status(self) -> None:
        repo_ok = (REPO_ROOT / "run.py").exists()
        env_ok = env_python().exists()
        torch_ok = False
        if env_ok:
            result = subprocess.run(
                [str(env_python()), "-c", "import torch; print(torch.__version__)"],
                capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            torch_ok = result.returncode == 0
        states = [
            f"Repository: {'ready' if repo_ok else 'missing'}",
            f"Conda environment: {'ready' if env_ok else 'missing'}",
            f"PyTorch: {'ready' if torch_ok else 'not verified'}",
        ]
        self.status.setText("  |  ".join(states))
        self.launch_btn.setEnabled(repo_ok and env_ok)
        self.web_ui_btn.setEnabled(repo_ok and env_ok)

    def _validate_conda(self) -> Optional[str]:
        value = self.conda_path.text().strip()
        if value and Path(value).exists():
            return value
        detected = conda_executable()
        if detected:
            self.conda_path.setText(detected)
            return detected
        QMessageBox.critical(
            self, "Conda not found",
            "Conda could not be found. Enter the path to conda.exe and try again."
        )
        return None

    def install_or_repair(self) -> None:
        if self.process:
            return
        conda = self._validate_conda()
        if not conda:
            return
        self._save_settings()
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.queue.clear()

        git = shutil.which("git")
        if not git:
            QMessageBox.critical(self, "Git not found", "Git is required to download AI Toolkit.")
            return

        if not (REPO_ROOT / ".git").exists():
            self.queue.append(("Clone AI Toolkit", [git, "clone", REPO_URL, str(REPO_ROOT)], APP_ROOT))
        elif self.update_repo.isChecked():
            self.queue.append(("Update AI Toolkit", [git, "-C", str(REPO_ROOT), "pull", "--ff-only"], APP_ROOT))

        self.queue.append((
            "Initialize repository submodules",
            [git, "-C", str(REPO_ROOT), "submodule", "update", "--init", "--recursive"],
            APP_ROOT
        ))

        if not env_python().exists():
            self.queue.append((
                "Create Python 3.12 Conda environment",
                [conda, "create", "-y", "-p", str(ENV_ROOT), "python=3.12", "pip"],
                APP_ROOT
            ))

        py = str(env_python())
        self.queue.extend([
            ("Upgrade packaging tools", [py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], APP_ROOT),
            ("Install CUDA PyTorch", [py, "-m", "pip", "install", "--no-cache-dir", *TORCH_PACKAGES, "--index-url", TORCH_INDEX], APP_ROOT),
            ("Install AI Toolkit requirements", [py, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")], REPO_ROOT),
            ("Install helper dependencies", [py, "-m", "pip", "install", "PySide6", "PyYAML", "Pillow"], REPO_ROOT),
        ])

        if self.install_ui_deps.isChecked():
            npm = shutil.which("npm")
            if not npm:
                QMessageBox.warning(
                    self, "Node.js not found",
                    "The backend will be installed, but original web UI dependencies are skipped "
                    "because npm was not found."
                )
            else:
                self.queue.append(("Install original web UI packages", [npm, "install"], REPO_ROOT / "ui"))

        self.progress.setRange(0, 0)
        self.install_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log("=== LoRA Trainer installation started ===")
        self._run_next()

    def _run_next(self) -> None:
        if not self.queue:
            self.process = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.install_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.append_log("=== Installation completed ===")
            self._refresh_status()
            QMessageBox.information(self, "Completed", "AI Toolkit installation completed.")
            return

        label, command, cwd = self.queue.pop(0)
        self.append_log(f"\n--- {label} ---")
        self.append_log(quote_cmd(command))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        if cwd:
            self.process.setWorkingDirectory(str(cwd))
        self.process.readyReadStandardOutput.connect(self._read_process)
        self.process.finished.connect(self._process_finished)
        self.process.start(command[0], command[1:])

    def _read_process(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(data)

    def _process_finished(self, code: int, _status) -> None:
        self._read_process()
        self.process = None
        if code != 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.install_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.queue.clear()
            self.append_log(f"ERROR: command exited with code {code}")
            self._refresh_status()
            QMessageBox.critical(
                self, "Installation failed",
                f"A command failed with exit code {code}.\nSee:\n{self.log_file}"
            )
            return
        QTimer.singleShot(100, self._run_next)

    def stop_process(self) -> None:
        self.queue.clear()
        if self.process:
            self.append_log("Stopping current installer process...")
            self.process.kill()
        self.install_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def verify_install(self) -> None:
        if not env_python().exists() or not (REPO_ROOT / "run.py").exists():
            QMessageBox.warning(self, "Not installed", "The repository or environment is missing.")
            return
        checks = [
            [str(env_python()), "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
            [str(env_python()), "-c", "import yaml, PIL; print('yaml and Pillow ready')"],
        ]
        output = []
        ok = True
        for cmd in checks:
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            output.append(result.stdout.strip() or result.stderr.strip())
            ok &= result.returncode == 0
        self.append_log("\n".join(output))
        self._refresh_status()
        QMessageBox.information(
            self, "Verification",
            ("Installation verified successfully." if ok else "One or more checks failed.") +
            "\n\n" + "\n".join(output)
        )

    def launch_helper(self) -> None:
        helper = APP_ROOT / "helpers" / "ostris_lora_train.py"
        if not helper.exists():
            QMessageBox.critical(self, "Missing helper", f"Missing:\n{helper}")
            return
        subprocess.Popen([str(env_python()), str(helper)], cwd=APP_ROOT)

    def launch_web_ui(self) -> None:
        npm = shutil.which("npm")
        if not npm:
            QMessageBox.warning(self, "Node.js not found", "Install Node.js 20 or newer to use the original web UI.")
            return
        subprocess.Popen(
            [npm, "run", "build_and_start"],
            cwd=REPO_ROOT / "ui",
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OstrisInstaller()
    window.show()
    sys.exit(app.exec())
