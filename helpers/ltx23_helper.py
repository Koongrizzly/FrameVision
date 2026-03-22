from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QSize, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleBox(QWidget):
    def __init__(self, title: str, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=checked)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggle.clicked.connect(self._on_toggled)

        self.content = QWidget()
        self.content.setVisible(checked)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 6, 8, 6)
        self.content_layout.setSpacing(8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class LTX23Helper(QWidget):
    SETTINGS_NAME = "ltx23_helper.json"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._loading_settings = False
        self._process: QProcess | None = None

        self.framevision_root = self._guess_framevision_root()
        self.settings_path = self.framevision_root / "presets" / "setsave" / self.SETTINGS_NAME
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._apply_defaults()
        self._load_settings()
        self._autodetect_paths(fill_missing_only=True)
        QTimer.singleShot(0, self._save_settings)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        banner = QFrame()
        banner.setObjectName("banner")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("LTX 2.3 GGUF Helper")
        tf = QFont()
        tf.setPointSize(14)
        tf.setBold(True)
        title.setFont(tf)
        subtitle = QLabel(
            "Experimental helper for env setup, repo checks, model validation, CLI probing, and early distilled testing."
        )
        subtitle.setWordWrap(True)
        banner_layout.addWidget(title)
        banner_layout.addWidget(subtitle)
        root.addWidget(banner)

        info = QLabel(
            "Notes: official LTX-2 docs say the codebase targets Python >= 3.12, CUDA > 12.7, and PyTorch ~= 2.7. "
            "The pipelines README exposes CLI modules such as ltx_pipelines.distilled, while the Unsloth GGUF page mostly documents model/support files and a ComfyUI path. "
            "This helper therefore focuses on getting the environment honest and readable first."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # Summary row
        summary = QGroupBox("Quick actions")
        srow = QHBoxLayout(summary)
        self.btn_autodetect = QPushButton("Autodetect paths")
        self.btn_autodetect.clicked.connect(lambda: self._autodetect_paths(fill_missing_only=False))
        self.btn_save = QPushButton("Save settings")
        self.btn_save.clicked.connect(self._save_settings)
        self.btn_open_output = QPushButton("Open output")
        self.btn_open_output.clicked.connect(lambda: self._open_path(self.output_dir_edit.text().strip()))
        self.btn_open_repo = QPushButton("Open repo")
        self.btn_open_repo.clicked.connect(lambda: self._open_path(self.repo_dir_edit.text().strip()))
        for w in [self.btn_autodetect, self.btn_save, self.btn_open_output, self.btn_open_repo]:
            srow.addWidget(w)
        srow.addStretch(1)
        root.addWidget(summary)

        # Paths
        paths_box = QGroupBox("Paths")
        paths = QFormLayout(paths_box)
        self.root_edit = self._path_row(paths, "FrameVision root", browse_dir=True)
        self.env_dir_edit = self._path_row(paths, "Environment folder", browse_dir=True)
        self.repo_dir_edit = self._path_row(paths, "LTX-2 repo folder", browse_dir=True)
        self.output_dir_edit = self._path_row(paths, "Output folder", browse_dir=True)
        root.addWidget(paths_box)

        # Environment
        env_box = QGroupBox("Environment & repo")
        env_layout = QVBoxLayout(env_box)
        env_buttons = QHBoxLayout()
        self.btn_check_tools = QPushButton("Check tools")
        self.btn_check_tools.clicked.connect(self.check_tools)
        self.btn_create_env = QPushButton("Create env")
        self.btn_create_env.clicked.connect(self.create_env)
        self.btn_clone_repo = QPushButton("Clone / update repo")
        self.btn_clone_repo.clicked.connect(self.clone_or_update_repo)
        self.btn_install_base = QPushButton("Install base deps")
        self.btn_install_base.clicked.connect(self.install_base_deps)
        self.btn_install_repo = QPushButton("Install repo packages")
        self.btn_install_repo.clicked.connect(self.install_repo_packages)
        for w in [self.btn_check_tools, self.btn_create_env, self.btn_clone_repo, self.btn_install_base, self.btn_install_repo]:
            env_buttons.addWidget(w)
        env_buttons.addStretch(1)
        env_layout.addLayout(env_buttons)

        env_flags = QGridLayout()
        self.use_uv_check = QCheckBox("Prefer uv when available")
        self.use_uv_check.setChecked(True)
        self.upgrade_pip_check = QCheckBox("Upgrade pip/setuptools/wheel first")
        self.upgrade_pip_check.setChecked(True)
        self.cuda_torch_check = QCheckBox("Use CUDA torch install hint")
        self.cuda_torch_check.setChecked(True)
        self.editable_install_check = QCheckBox("Use editable install for ltx packages")
        self.editable_install_check.setChecked(True)
        env_flags.addWidget(self.use_uv_check, 0, 0)
        env_flags.addWidget(self.upgrade_pip_check, 0, 1)
        env_flags.addWidget(self.cuda_torch_check, 1, 0)
        env_flags.addWidget(self.editable_install_check, 1, 1)
        env_layout.addLayout(env_flags)
        root.addWidget(env_box)

        # Model files
        model_box = QGroupBox("Model files")
        model_layout = QFormLayout(model_box)
        self.checkpoint_edit = self._path_row(model_layout, "Distilled GGUF checkpoint", browse_file=True)
        self.video_vae_edit = self._path_row(model_layout, "Video VAE", browse_file=True)
        self.audio_vae_edit = self._path_row(model_layout, "Audio VAE", browse_file=True)
        self.embed_connector_edit = self._path_row(model_layout, "Embeddings connector", browse_file=True)
        self.gemma_edit = self._path_row(model_layout, "Gemma QAT GGUF", browse_file=True)
        self.mmproj_edit = self._path_row(model_layout, "mmproj GGUF", browse_file=True)
        self.upscaler_edit = self._path_row(model_layout, "Spatial upscaler", browse_file=True)
        mbtns = QHBoxLayout()
        self.btn_find_models = QPushButton("Find model files")
        self.btn_find_models.clicked.connect(lambda: self._autodetect_paths(fill_missing_only=False))
        self.btn_validate_models = QPushButton("Validate model files")
        self.btn_validate_models.clicked.connect(self.validate_model_files)
        mbtns.addWidget(self.btn_find_models)
        mbtns.addWidget(self.btn_validate_models)
        mbtns.addStretch(1)
        model_layout.addRow("", self._wrap_layout(mbtns))
        root.addWidget(model_box)

        # Generation
        gen_box = QGroupBox("Generation / test")
        gen_layout = QFormLayout(gen_box)
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItems([
            "distilled",
            "ti2vid_two_stages",
            "ti2vid_two_stages_hq",
        ])
        self.pipeline_combo.setToolTip("Start with distilled. The others are there to compare CLI help and error output.")
        gen_layout.addRow("Pipeline", self.pipeline_combo)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(80)
        gen_layout.addRow("Prompt", self.prompt_edit)

        self.start_image_edit = self._browse_lineedit(file_mode=True)
        gen_layout.addRow("Optional start image", self.start_image_edit[0])

        dims = QWidget()
        dims_layout = QHBoxLayout(dims)
        dims_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QSpinBox(); self.width_spin.setRange(32, 4096); self.width_spin.setSingleStep(32)
        self.height_spin = QSpinBox(); self.height_spin.setRange(32, 4096); self.height_spin.setSingleStep(32)
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(9, 4097); self.frames_spin.setSingleStep(8)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 120)
        dims_layout.addWidget(QLabel("W")); dims_layout.addWidget(self.width_spin)
        dims_layout.addWidget(QLabel("H")); dims_layout.addWidget(self.height_spin)
        dims_layout.addWidget(QLabel("Frames")); dims_layout.addWidget(self.frames_spin)
        dims_layout.addWidget(QLabel("FPS")); dims_layout.addWidget(self.fps_spin)
        dims_layout.addStretch(1)
        gen_layout.addRow("Video", dims)

        misc = QWidget()
        misc_layout = QHBoxLayout(misc)
        misc_layout.setContentsMargins(0, 0, 0, 0)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 2147483647)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 200)
        self.cfg_spin = QSpinBox(); self.cfg_spin.setRange(1, 50)
        misc_layout.addWidget(QLabel("Seed")); misc_layout.addWidget(self.seed_spin)
        misc_layout.addWidget(QLabel("Steps")); misc_layout.addWidget(self.steps_spin)
        misc_layout.addWidget(QLabel("CFG")); misc_layout.addWidget(self.cfg_spin)
        misc_layout.addStretch(1)
        gen_layout.addRow("Sampling", misc)

        self.output_name_edit = QLineEdit()
        gen_layout.addRow("Output name", self.output_name_edit)

        self.alloc_conf_check = QCheckBox("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        self.alloc_conf_check.setChecked(True)
        self.help_first_check = QCheckBox("Probe module with --help before launch")
        self.help_first_check.setChecked(True)
        self.no_run_check = QCheckBox("Dry run only (show command, do not launch)")
        self.no_run_check.setChecked(True)
        gen_layout.addRow("", self.alloc_conf_check)
        gen_layout.addRow("", self.help_first_check)
        gen_layout.addRow("", self.no_run_check)

        gen_btns = QHBoxLayout()
        self.btn_import_test = QPushButton("Import smoke test")
        self.btn_import_test.clicked.connect(self.import_smoke_test)
        self.btn_help = QPushButton("Run module --help")
        self.btn_help.clicked.connect(self.run_module_help)
        self.btn_build = QPushButton("Build command")
        self.btn_build.clicked.connect(self.build_command_preview)
        self.btn_run = QPushButton("Run test")
        self.btn_run.clicked.connect(self.run_test)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_process)
        for w in [self.btn_import_test, self.btn_help, self.btn_build, self.btn_run, self.btn_stop]:
            gen_btns.addWidget(w)
        gen_btns.addStretch(1)
        gen_layout.addRow("", self._wrap_layout(gen_btns))
        root.addWidget(gen_box)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setPlaceholderText("Command preview will show here.")
        self.command_preview.setMinimumHeight(120)
        root.addWidget(QLabel("Command preview"))
        root.addWidget(self.command_preview)

        # Collapsible lower sections
        self.paths_box = CollapsibleBox("Resolved paths", checked=False)
        self.paths_text = QPlainTextEdit()
        self.paths_text.setReadOnly(True)
        self.paths_text.setMinimumHeight(150)
        self.paths_box.content_layout.addWidget(self.paths_text)
        root.addWidget(self.paths_box)

        self.logs_box = CollapsibleBox("Logs", checked=False)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)
        self.logs_box.content_layout.addWidget(self.log_edit)
        root.addWidget(self.logs_box)

        root.addWidget(QLabel("Mini log"))
        self.mini_log = QPlainTextEdit()
        self.mini_log.setReadOnly(True)
        self.mini_log.setMaximumHeight(110)
        root.addWidget(self.mini_log)
        root.addStretch(1)

        self.setStyleSheet(
            """
            QFrame#banner {
                border: 1px solid rgba(160, 160, 160, 90);
                border-radius: 10px;
                background: rgba(120, 120, 160, 30);
            }
            QPlainTextEdit, QTextEdit, QLineEdit, QComboBox, QSpinBox {
                min-height: 28px;
            }
            QGroupBox {
                font-weight: 600;
            }
            """
        )

    def _path_row(self, layout: QFormLayout, label: str, browse_dir: bool = False, browse_file: bool = False) -> QLineEdit:
        widget, edit = self._browse_lineedit(file_mode=browse_file, dir_mode=browse_dir)
        layout.addRow(label, widget)
        return edit

    def _browse_lineedit(self, file_mode: bool = False, dir_mode: bool = False):
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.textChanged.connect(self._on_setting_changed)
        lay.addWidget(edit)
        browse = QPushButton("Browse")
        browse.setMaximumWidth(80)

        def _browse() -> None:
            start = edit.text().strip() or str(self.framevision_root)
            if dir_mode:
                path = QFileDialog.getExistingDirectory(self, "Select folder", start)
            elif file_mode:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
            else:
                path = ""
            if path:
                edit.setText(path)

        browse.clicked.connect(_browse)
        lay.addWidget(browse)
        return wrap, edit

    def _wrap_layout(self, layout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    # ---------- defaults / settings ----------
    def _guess_framevision_root(self) -> Path:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / "helpers").exists() or (parent / "presets").exists():
                return parent
        return here.parent

    def _apply_defaults(self) -> None:
        self.root_edit.setText(str(self.framevision_root))
        self.env_dir_edit.setText(str(self.framevision_root / "environments" / ".ltx23"))
        self.repo_dir_edit.setText(str(self.framevision_root / "models" / "ltx23" / "LTX-2"))
        self.output_dir_edit.setText(str(self.framevision_root / "output" / "ltx23"))
        self.prompt_edit.setPlainText("a cinematic shot of a florist arranging flowers in a small sunlit shop, soft camera movement, natural colors")
        self.width_spin.setValue(640)
        self.height_spin.setValue(384)
        self.frames_spin.setValue(17)
        self.fps_spin.setValue(24)
        self.seed_spin.setValue(42)
        self.steps_spin.setValue(8)
        self.cfg_spin.setValue(1)
        self.output_name_edit.setText("ltx23_test.mp4")
        self._refresh_paths_text()

    def _load_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Failed to load settings: {exc}")
            return
        self._loading_settings = True
        try:
            mapping = {
                "root": self.root_edit,
                "env_dir": self.env_dir_edit,
                "repo_dir": self.repo_dir_edit,
                "output_dir": self.output_dir_edit,
                "checkpoint": self.checkpoint_edit,
                "video_vae": self.video_vae_edit,
                "audio_vae": self.audio_vae_edit,
                "embed_connector": self.embed_connector_edit,
                "gemma": self.gemma_edit,
                "mmproj": self.mmproj_edit,
                "upscaler": self.upscaler_edit,
                "prompt": self.prompt_edit,
                "start_image": self.start_image_edit[1],
                "output_name": self.output_name_edit,
            }
            for key, widget in mapping.items():
                if key not in data:
                    continue
                value = data[key]
                if isinstance(widget, QTextEdit):
                    widget.setPlainText(str(value))
                else:
                    widget.setText(str(value))

            self.pipeline_combo.setCurrentText(data.get("pipeline", self.pipeline_combo.currentText()))
            self.width_spin.setValue(int(data.get("width", self.width_spin.value())))
            self.height_spin.setValue(int(data.get("height", self.height_spin.value())))
            self.frames_spin.setValue(int(data.get("frames", self.frames_spin.value())))
            self.fps_spin.setValue(int(data.get("fps", self.fps_spin.value())))
            self.seed_spin.setValue(int(data.get("seed", self.seed_spin.value())))
            self.steps_spin.setValue(int(data.get("steps", self.steps_spin.value())))
            self.cfg_spin.setValue(int(data.get("cfg", self.cfg_spin.value())))
            self.use_uv_check.setChecked(bool(data.get("use_uv", self.use_uv_check.isChecked())))
            self.upgrade_pip_check.setChecked(bool(data.get("upgrade_pip", self.upgrade_pip_check.isChecked())))
            self.cuda_torch_check.setChecked(bool(data.get("cuda_torch", self.cuda_torch_check.isChecked())))
            self.editable_install_check.setChecked(bool(data.get("editable_install", self.editable_install_check.isChecked())))
            self.alloc_conf_check.setChecked(bool(data.get("alloc_conf", self.alloc_conf_check.isChecked())))
            self.help_first_check.setChecked(bool(data.get("help_first", self.help_first_check.isChecked())))
            self.no_run_check.setChecked(bool(data.get("dry_run", self.no_run_check.isChecked())))
        finally:
            self._loading_settings = False
            self._refresh_paths_text()

    def _save_settings(self) -> None:
        if self._loading_settings:
            return
        data = {
            "root": self.root_edit.text().strip(),
            "env_dir": self.env_dir_edit.text().strip(),
            "repo_dir": self.repo_dir_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "checkpoint": self.checkpoint_edit.text().strip(),
            "video_vae": self.video_vae_edit.text().strip(),
            "audio_vae": self.audio_vae_edit.text().strip(),
            "embed_connector": self.embed_connector_edit.text().strip(),
            "gemma": self.gemma_edit.text().strip(),
            "mmproj": self.mmproj_edit.text().strip(),
            "upscaler": self.upscaler_edit.text().strip(),
            "prompt": self.prompt_edit.toPlainText(),
            "start_image": self.start_image_edit[1].text().strip(),
            "pipeline": self.pipeline_combo.currentText(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "frames": self.frames_spin.value(),
            "fps": self.fps_spin.value(),
            "seed": self.seed_spin.value(),
            "steps": self.steps_spin.value(),
            "cfg": self.cfg_spin.value(),
            "output_name": self.output_name_edit.text().strip(),
            "use_uv": self.use_uv_check.isChecked(),
            "upgrade_pip": self.upgrade_pip_check.isChecked(),
            "cuda_torch": self.cuda_torch_check.isChecked(),
            "editable_install": self.editable_install_check.isChecked(),
            "alloc_conf": self.alloc_conf_check.isChecked(),
            "help_first": self.help_first_check.isChecked(),
            "dry_run": self.no_run_check.isChecked(),
        }
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._log(f"Saved settings -> {self.settings_path}")
        except Exception as exc:
            self._log(f"Failed to save settings: {exc}")
        self._refresh_paths_text()

    def _on_setting_changed(self, *_args) -> None:
        if self._loading_settings:
            return
        self._refresh_paths_text()
        self._save_settings()

    # ---------- logging ----------
    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self.log_edit.appendPlainText(line)
        self.mini_log.appendPlainText(line)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())
        self.mini_log.verticalScrollBar().setValue(self.mini_log.verticalScrollBar().maximum())

    def _refresh_paths_text(self) -> None:
        info = {
            "FrameVision root": self.root_edit.text().strip(),
            "Environment folder": self.env_dir_edit.text().strip(),
            "Repo folder": self.repo_dir_edit.text().strip(),
            "Output folder": self.output_dir_edit.text().strip(),
            "Settings JSON": str(self.settings_path),
            "Python exe": self._env_python(),
            "Checkpoint": self.checkpoint_edit.text().strip(),
            "Video VAE": self.video_vae_edit.text().strip(),
            "Audio VAE": self.audio_vae_edit.text().strip(),
            "Embeddings connector": self.embed_connector_edit.text().strip(),
            "Gemma GGUF": self.gemma_edit.text().strip(),
            "mmproj GGUF": self.mmproj_edit.text().strip(),
            "Spatial upscaler": self.upscaler_edit.text().strip(),
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        self.paths_text.setPlainText("\n".join(lines))

    # ---------- path detection ----------
    def _autodetect_paths(self, fill_missing_only: bool = True) -> None:
        root = Path(self.root_edit.text().strip() or self.framevision_root)
        self.framevision_root = root

        candidates = [
            root / "models" / "ltx23_gguf",
            root / "models" / "ltx23_gguf" / "unet",
            root / "models" / "ltx23_gguf" / "vae",
            root / "models" / "ltx23_gguf" / "text_encoders",
            root / "models" / "ltx23_gguf" / "latent_upscale_models",
            root / "models",
            root,
        ]

        def set_if_needed(edit: QLineEdit, value: Path | None) -> None:
            if value is None:
                return
            if fill_missing_only and edit.text().strip():
                return
            edit.setText(str(value))

        # Repo autodetect
        repo_candidates = [
            root / "models" / "ltx23_gguf" / "LTX-2",
            root / "models" / "ltx23" / "LTX-2",
            root / "LTX-2",
        ]
        repo_found = next((p for p in repo_candidates if p.exists()), None)
        set_if_needed(self.repo_dir_edit, repo_found or repo_candidates[0])

        def _is_real_model_file(p: Path) -> bool:
            if not p.is_file():
                return False
            lower = str(p).lower().replace('\\', '/')
            bad_parts = (
                '/manifests/',
                '/.cache/',
                '/cache/huggingface/',
                '/download/',
                '/snapshots/',
                '/refs/',
                '/blobs/',
            )
            bad_suffixes = ('.json', '.metadata', '.lock', '.incomplete', '.tmp', '.part')
            if any(part in lower for part in bad_parts):
                return False
            if lower.endswith(bad_suffixes):
                return False
            return True

        all_files = []
        for base in candidates:
            if base.exists():
                try:
                    for p in base.rglob('*'):
                        if _is_real_model_file(p):
                            all_files.append(p)
                except Exception:
                    pass

        def first_match(*needles: str) -> Path | None:
            lowered = [n.lower() for n in needles]
            for p in all_files:
                name = str(p).lower().replace('\\', '/')
                if all(n in name for n in lowered):
                    return p
            return None

        set_if_needed(self.checkpoint_edit, first_match('/unet/', 'distilled', '.gguf') or first_match('ltx-2.3', 'distilled', '.gguf'))
        set_if_needed(self.video_vae_edit, first_match('/vae/', 'video_vae', '.safetensors'))
        set_if_needed(self.audio_vae_edit, first_match('/vae/', 'audio_vae', '.safetensors'))
        set_if_needed(self.embed_connector_edit, first_match('embeddings_connectors', '.safetensors'))
        set_if_needed(self.gemma_edit, first_match('/text_encoders/', 'gemma', '.gguf'))
        set_if_needed(self.mmproj_edit, first_match('/text_encoders/', 'mmproj', '.gguf'))
        set_if_needed(self.upscaler_edit, first_match('/latent_upscale_models/', 'spatial-upscaler', '.safetensors') or first_match('/latent_upscale_models/', 'spatial_upscaler', '.safetensors'))
        self._refresh_paths_text()
        self._log("Autodetect finished.")

    # ---------- command helpers ----------
    def _env_python(self) -> str:
        env_dir = Path(self.env_dir_edit.text().strip())
        if platform.system().lower().startswith("win"):
            return str(env_dir / "Scripts" / "python.exe")
        return str(env_dir / "bin" / "python")

    def _env_pip(self) -> str:
        env_dir = Path(self.env_dir_edit.text().strip())
        if platform.system().lower().startswith("win"):
            return str(env_dir / "Scripts" / "pip.exe")
        return str(env_dir / "bin" / "pip")

    def _tool_exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def _open_path(self, path_text: str) -> None:
        if not path_text:
            return
        p = Path(path_text)
        if p.is_file():
            target = p.parent
        else:
            target = p
        target.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            self._log(f"Open path failed: {exc}")

    def _run_qprocess(self, command: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy", "Another process is still running.")
            return
        self._process = QProcess(self)
        if cwd:
            self._process.setWorkingDirectory(cwd)
        proc_env = self._process.processEnvironment()
        for key, value in (env or {}).items():
            proc_env.insert(key, value)
        self._process.setProcessEnvironment(proc_env)
        self._process.readyReadStandardOutput.connect(lambda: self._log(bytes(self._process.readAllStandardOutput()).decode(errors="ignore").rstrip()))
        self._process.readyReadStandardError.connect(lambda: self._log(bytes(self._process.readAllStandardError()).decode(errors="ignore").rstrip()))
        self._process.finished.connect(lambda code, status: self._log(f"Process finished with exit code {code}, status {status}."))
        self._log("Launching: " + self._quote_cmd(command))
        self._process.start(command[0], command[1:])

    def _run_subprocess_capture(self, command: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
        try:
            proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
            out = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
            return proc.returncode, out.strip()
        except Exception as exc:
            return 1, str(exc)

    def _quote_cmd(self, command: list[str]) -> str:
        parts = []
        for part in command:
            if " " in part or "\t" in part:
                parts.append(f'"{part}"')
            else:
                parts.append(part)
        return " ".join(parts)

    def _common_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HF_HOME"] = str(Path(self.root_edit.text().strip()) / "cache" / "huggingface")
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(self.root_edit.text().strip()) / "cache" / "huggingface" / "hub")
        if self.alloc_conf_check.isChecked():
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return env

    def _validate_video_shape(self) -> list[str]:
        problems = []
        if self.width_spin.value() % 32 != 0:
            problems.append("Width should be divisible by 32.")
        if self.height_spin.value() % 32 != 0:
            problems.append("Height should be divisible by 32.")
        if (self.frames_spin.value() - 1) % 8 != 0:
            problems.append("Frame count should follow 8k+1 (17, 25, 33, ...).")
        return problems

    def _build_pipeline_command(self, help_mode: bool = False) -> list[str]:
        py = self._env_python()
        module = f"ltx_pipelines.{self.pipeline_combo.currentText()}"
        if help_mode:
            return [py, "-m", module, "--help"]

        out_dir = Path(self.output_dir_edit.text().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (self.output_name_edit.text().strip() or "ltx23_test.mp4")

        cmd = [py, "-m", module]

        # Shared best-guess args based on the current LTX pipelines README examples.
        if self.checkpoint_edit.text().strip():
            cmd += ["--checkpoint-path", self.checkpoint_edit.text().strip()]
        if self.upscaler_edit.text().strip():
            cmd += ["--spatial-upsampler-path", self.upscaler_edit.text().strip()]
        if self.gemma_edit.text().strip():
            gemma_root = str(Path(self.gemma_edit.text().strip()).parent)
            cmd += ["--gemma-root", gemma_root]
        if self.prompt_edit.toPlainText().strip():
            cmd += ["--prompt", self.prompt_edit.toPlainText().strip()]
        cmd += ["--output-path", str(output_path)]
        cmd += ["--seed", str(self.seed_spin.value())]
        cmd += ["--height", str(self.height_spin.value())]
        cmd += ["--width", str(self.width_spin.value())]
        cmd += ["--num-frames", str(self.frames_spin.value())]
        cmd += ["--frame-rate", str(float(self.fps_spin.value()))]
        cmd += ["--num-inference-steps", str(self.steps_spin.value())]

        start_image = self.start_image_edit[1].text().strip()
        if start_image:
            # Kept intentionally simple; exact image-conditioning flags can be confirmed by --help.
            cmd += ["--image-path", start_image]

        return cmd

    def build_command_preview(self) -> None:
        issues = self._validate_video_shape()
        if issues:
            self._log("Shape warning(s): " + " | ".join(issues))
        cmd = self._build_pipeline_command(help_mode=False)
        text = self._quote_cmd(cmd)
        self.command_preview.setPlainText(text)
        self._log("Command preview updated.")

    # ---------- actions ----------
    def check_tools(self) -> None:
        py = Path(self._env_python())
        rows = [
            f"System Python: {sys.executable}",
            f"Target env python exists: {py.exists()} -> {py}",
            f"Git found: {self._tool_exists('git')}",
            f"uv found: {self._tool_exists('uv')}",
            f"hf found: {self._tool_exists('hf')}",
            f"Python version now: {platform.python_version()}",
            f"OS: {platform.platform()}",
        ]
        self._log("Tool check:\n" + "\n".join(rows))

    def create_env(self) -> None:
        env_dir = Path(self.env_dir_edit.text().strip())
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.use_uv_check.isChecked() and self._tool_exists("uv"):
            cmd = ["uv", "venv", str(env_dir), "--python", "3.12"]
        else:
            cmd = [sys.executable, "-m", "venv", str(env_dir)]
        self._run_qprocess(cmd, cwd=str(Path(self.root_edit.text().strip())))

    def clone_or_update_repo(self) -> None:
        repo_dir = Path(self.repo_dir_edit.text().strip())
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if (repo_dir / ".git").exists():
            cmd = ["git", "-C", str(repo_dir), "pull"]
            self._run_qprocess(cmd)
        else:
            cmd = ["git", "clone", "https://github.com/Lightricks/LTX-2.git", str(repo_dir)]
            self._run_qprocess(cmd)

    def install_base_deps(self) -> None:
        pip = self._env_pip()
        if not Path(pip).exists():
            self._log("Target pip not found. Create the env first.")
            return
        cmds: list[list[str]] = []
        if self.upgrade_pip_check.isChecked():
            cmds.append([pip, "install", "-U", "pip", "setuptools", "wheel"])
        if self.cuda_torch_check.isChecked():
            cmds.append([
                pip, "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu128",
            ])
        cmds.append([
            pip, "install",
            "accelerate",
            "transformers",
            "diffusers",
            "safetensors",
            "sentencepiece",
            "huggingface_hub[cli]",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "Pillow",
            "numpy",
            "scipy",
            "einops",
            "librosa",
            "soundfile",
            "omegaconf",
            "pyyaml",
            "tqdm",
            "packaging",
            "requests",
        ])
        for cmd in cmds:
            self._run_qprocess(cmd, cwd=str(Path(self.root_edit.text().strip())), env=self._common_env())
            break
        if len(cmds) > 1:
            self._log("Base dependency install starts with the first command. Re-run the next ones after completion if needed.")

    def install_repo_packages(self) -> None:
        repo_dir = Path(self.repo_dir_edit.text().strip())
        pip = self._env_pip()
        py = self._env_python()
        if not repo_dir.exists():
            self._log("Repo folder not found. Clone/update the repo first.")
            return
        if self.use_uv_check.isChecked() and self._tool_exists("uv") and (repo_dir / "pyproject.toml").exists():
            cmd = ["uv", "sync"]
            self._run_qprocess(cmd, cwd=str(repo_dir), env=self._common_env())
            return
        if self.editable_install_check.isChecked():
            cmd = [pip, "install", "-e", "packages/ltx-core", "-e", "packages/ltx-pipelines"]
        else:
            cmd = [py, "-m", "pip", "install", "packages/ltx-core", "packages/ltx-pipelines"]
        self._run_qprocess(cmd, cwd=str(repo_dir), env=self._common_env())

    def validate_model_files(self) -> None:
        checks = {
            "Distilled checkpoint": self.checkpoint_edit.text().strip(),
            "Video VAE": self.video_vae_edit.text().strip(),
            "Audio VAE": self.audio_vae_edit.text().strip(),
            "Embeddings connector": self.embed_connector_edit.text().strip(),
            "Gemma GGUF": self.gemma_edit.text().strip(),
            "mmproj": self.mmproj_edit.text().strip(),
            "Spatial upscaler": self.upscaler_edit.text().strip(),
        }
        missing = []
        bad = []
        for label, path_text in checks.items():
            p = Path(path_text) if path_text else None
            exists = bool(path_text) and p.exists()
            lower = str(p).lower().replace('\\', '/') if p else ''
            suspicious = lower.endswith(('.json', '.metadata', '.lock', '.incomplete', '.tmp', '.part')) or any(part in lower for part in ('/manifests/', '/.cache/', '/cache/huggingface/', '/download/', '/snapshots/', '/refs/', '/blobs/'))
            if exists and suspicious:
                self._log(f"{label}: WRONG FILE TYPE -> {path_text}")
                bad.append(label)
            else:
                self._log(f"{label}: {'OK' if exists else 'MISSING'} -> {path_text or '(empty)'}")
            if (not exists) or suspicious:
                missing.append(label)
        if not missing:
            self._log("All tracked model files look present.")
        else:
            if bad:
                self._log("These entries point to cache/manifest/metadata files instead of real model files: " + ", ".join(bad))
            self._log("Missing items: " + ", ".join(missing))

    def import_smoke_test(self) -> None:
        py = self._env_python()
        code = (
            "import sys\n"
            "mods=['torch','transformers','diffusers','safetensors','PIL','imageio_ffmpeg','ltx_pipelines.distilled']\n"
            "bad=[]\n"
            "for m in mods:\n"
            "    try:\n"
            "        __import__(m)\n"
            "        print(f'OK: {m}')\n"
            "    except Exception as e:\n"
            "        bad.append((m, str(e)))\n"
            "        print(f'FAIL: {m} -> {e}')\n"
            "print('DONE')\n"
            "sys.exit(1 if bad else 0)\n"
        )
        self._run_qprocess([py, "-c", code], cwd=self.repo_dir_edit.text().strip() or None, env=self._common_env())

    def run_module_help(self) -> None:
        cmd = self._build_pipeline_command(help_mode=True)
        self.command_preview.setPlainText(self._quote_cmd(cmd))
        self._run_qprocess(cmd, cwd=self.repo_dir_edit.text().strip() or None, env=self._common_env())

    def run_test(self) -> None:
        issues = self._validate_video_shape()
        if issues:
            self._log("Shape warning(s): " + " | ".join(issues))
        cmd = self._build_pipeline_command(help_mode=False)
        self.command_preview.setPlainText(self._quote_cmd(cmd))
        if self.help_first_check.isChecked():
            self._log("Tip: run module --help first and compare the accepted flags with the command preview.")
        if self.no_run_check.isChecked():
            self._log("Dry run enabled. Command was not launched.")
            return
        Path(self.output_dir_edit.text().strip()).mkdir(parents=True, exist_ok=True)
        self._run_qprocess(cmd, cwd=self.repo_dir_edit.text().strip() or None, env=self._common_env())

    def stop_process(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._log("Process killed.")


class LTX23HelperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LTX 2.3 GGUF Helper")
        self.resize(QSize(1180, 920))
        self.setCentralWidget(LTX23Helper(self))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = LTX23HelperWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
