import json
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
)

import torch
from diffusers import ZImagePipeline


class GenerateThread(QThread):
    finished = Signal(str)   # path to saved image
    error = Signal(str)      # error message / traceback
    status = Signal(str)     # status text

    def __init__(self, pipe, prompt, height, width, steps, guidance, seed, out_dir, parent=None):
        super().__init__(parent)
        self.pipe = pipe
        self.prompt = prompt
        self.height = height
        self.width = width
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.out_dir = Path(out_dir)

    def run(self):
        try:
            self.status.emit("Starting generation ...")

            # Choose generator device
            if torch.cuda.is_available():
                gen_device = "cuda"
            else:
                gen_device = "cpu"

            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=gen_device).manual_seed(int(self.seed))

            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.status.emit("Running Z-Image-Turbo pipeline ...")

            result = self.pipe(
                prompt=self.prompt,
                height=int(self.height),
                width=int(self.width),
                num_inference_steps=int(self.steps),
                guidance_scale=float(self.guidance),
                generator=generator,
            )

            image = result.images[0]
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.out_dir / f"zimage_{ts}.png"
            image.save(out_path)

            self.status.emit("Done.")
            self.finished.emit(str(out_path))

        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)


class ZImageWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Paths
        self.root_dir = Path(__file__).resolve().parents[1]
        self.model_dir = self.root_dir / "models" / "Z-Image-Turbo"
        self.settings_path = self.root_dir / "presets" / "setsave" / "zimage.json"

        self.settings = self._load_settings()
        self.pipe = None
        self.worker = None

        self._build_ui()
        self._load_pipeline()

    # ---------- Settings helpers ----------

    def _load_settings(self):
        default = {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "seed": 42,
            "output_dir": "output/zimage",
            "last_prompt": "",
        }
        try:
            if self.settings_path.is_file():
                with self.settings_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                default.update(data)
        except Exception:
            # Ignore errors and fallback to defaults
            pass
        return default

    def _save_settings(self):
        try:
            self.settings["height"] = int(self.height_spin.value())
            self.settings["width"] = int(self.width_spin.value())
            self.settings["num_inference_steps"] = int(self.steps_spin.value())
            self.settings["guidance_scale"] = float(self.guidance_spin.value())
            self.settings["seed"] = int(self.seed_spin.value())
            self.settings["output_dir"] = self.output_edit.text().strip()
            self.settings["last_prompt"] = self.prompt_edit.toPlainText()

            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            # Non-fatal
            pass

    # ---------- UI ----------

    def _build_ui(self):
        self.setWindowTitle("Z-Image-Turbo Helper")
        self.resize(640, 480)

        layout = QVBoxLayout(self)

        # Prompt
        layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the image you want to generate...")
        if self.settings.get("last_prompt"):
            self.prompt_edit.setPlainText(self.settings["last_prompt"])
        layout.addWidget(self.prompt_edit)

        # Parameters row 1 (size)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(int(self.settings.get("width", 1024)))
        size_row.addWidget(self.width_spin)

        size_row.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(int(self.settings.get("height", 1024)))
        size_row.addWidget(self.height_spin)

        layout.addLayout(size_row)

        # Parameters row 2 (steps, guidance, seed)
        params_row = QHBoxLayout()

        params_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(int(self.settings.get("num_inference_steps", 9)))
        params_row.addWidget(self.steps_spin)

        params_row.addWidget(QLabel("Guidance:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setValue(float(self.settings.get("guidance_scale", 0.0)))
        params_row.addWidget(self.guidance_spin)

        params_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(int(self.settings.get("seed", 42)))
        params_row.addWidget(self.seed_spin)

        layout.addLayout(params_row)

        # Output folder
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit(self.settings.get("output_dir", "output/zimage"))
        out_row.addWidget(self.output_edit)
        self.output_button = QPushButton("Browse...")
        self.output_button.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_button)
        layout.addLayout(out_row)

        # Buttons
        button_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        button_row.addWidget(self.generate_button)

        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.clicked.connect(self._on_open_output_folder)
        button_row.addWidget(self.open_folder_button)

        layout.addLayout(button_row)

        # Status + progress
        self.status_label = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

    # ---------- Pipeline loading ----------

    def _load_pipeline(self):
        if not self.model_dir.exists():
            QMessageBox.critical(
                self,
                "Model not found",
                f"Could not find the Z-Image-Turbo model at:\n{self.model_dir}\n\n"
                "Please run presets/extra_env/zimage_install.bat first to download it.",
            )
            self.status_label.setText("Model not found. Run installer first.")
            return

        try:
            self.status_label.setText("Loading Z-Image-Turbo pipeline ...")
            QApplication.processEvents()

            if torch.cuda.is_available():
                # Prefer bfloat16 if supported, otherwise float16
                dtype = torch.bfloat16
                if not hasattr(torch.cuda, 'is_bf16_supported') or not torch.cuda.is_bf16_supported():
                    dtype = torch.float16
                device = "cuda"
            else:
                dtype = torch.float32
                device = "cpu"

            self.pipe = ZImagePipeline.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            self.pipe.to(device)

            self.status_label.setText(f"Pipeline loaded on {device}. Ready.")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(
                self,
                "Error loading pipeline",
                f"Failed to load Z-Image-Turbo:\n{e}\n\n{tb}",
            )
            self.status_label.setText("Failed to load pipeline. See error.")
            self.pipe = None

    # ---------- Actions ----------

    def _browse_output(self):
        base = (self.root_dir / "output").resolve()
        base.mkdir(parents=True, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", str(base))
        if folder:
            self.output_edit.setText(folder)
            self._save_settings()

    def _on_open_output_folder(self):
        folder = Path(self.output_edit.text().strip() or "output/zimage")
        if not folder.is_absolute():
            folder = self.root_dir / folder
        folder.mkdir(parents=True, exist_ok=True)

        # Open folder with system file explorer
        try:
            if sys.platform.startswith("win"):
                import subprocess
                subprocess.Popen(["explorer", str(folder)])
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(folder)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            QMessageBox.information(self, "Output folder", f"Images are saved in:\n{folder}")

    def _on_generate_clicked(self):
        if self.pipe is None:
            QMessageBox.warning(self, "Pipeline not ready", "Pipeline is not loaded. Check earlier errors or run the installer.")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Generation is already in progress.")
            return

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No prompt", "Please enter a prompt first.")
            return

        self._save_settings()

        # Resolve output folder
        out_dir = self.output_edit.text().strip()
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            out_path = self.root_dir / out_path

        self.worker = GenerateThread(
            pipe=self.pipe,
            prompt=prompt,
            height=self.height_spin.value(),
            width=self.width_spin.value(),
            steps=self.steps_spin.value(),
            guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(),
            out_dir=str(out_path),
        )
        self.worker.finished.connect(self._on_generation_finished)
        self.worker.error.connect(self._on_generation_error)
        self.worker.status.connect(self._on_worker_status)

        self.generate_button.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Generating image ...")

        self.worker.start()

    def _on_worker_status(self, text):
        self.status_label.setText(text)

    def _on_generation_finished(self, path):
        self.progress.setVisible(False)
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Saved: {path}")
        QMessageBox.information(self, "Done", f"Image saved to:\n{path}")

    def _on_generation_error(self, tb):
        self.progress.setVisible(False)
        self.generate_button.setEnabled(True)
        self.status_label.setText("Error during generation.")
        QMessageBox.critical(self, "Generation error", tb)

    # ---------- Qt overrides ----------

    def closeEvent(self, event):
        try:
            self._save_settings()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = ZImageWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
