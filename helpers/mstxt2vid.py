import os
import json
import sys
import threading
from datetime import datetime

from PySide6 import QtWidgets, QtCore

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

print("[MS-TXT2VID] Python executable:", sys.executable)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETSAVE_PATH = os.path.join(ROOT_DIR, "presets", "setsave", "ms.json")


class MSTxt2VidConfig(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.data = {
            "prompt": "A cute robot walking through a neon city at night, cinematic, 3D render",
            "num_frames": 16,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": -1,
            "fps": 8,
            "use_cpu_offload": True,
            "use_vae_slicing": True,
            "output_dir": os.path.join("output", "video"),
            "model_local_dir": os.path.join("models", "text-to-video-ms-1.7b"),
        }
        self.load()

    def load(self):
        if os.path.isfile(SETSAVE_PATH):
            try:
                with open(SETSAVE_PATH, "r", encoding="utf-8") as f:
                    incoming = json.load(f)
                self.data.update(incoming)
            except Exception as e:
                print("[MS-TXT2VID] Failed to load config:", e, file=sys.stderr)

    def save(self):
        try:
            os.makedirs(os.path.dirname(SETSAVE_PATH), exist_ok=True)
            with open(SETSAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print("[MS-TXT2VID] Failed to save config:", e, file=sys.stderr)

    def __getitem__(self, item):
        return self.data.get(item)

    def __setitem__(self, key, value):
        self.data[key] = value
        self.changed.emit()


class MSTxt2VidWindow(QtWidgets.QWidget):
    log_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MS Text-to-Video (ali-vilab/text-to-video-ms-1.7b)")
        self.resize(640, 480)

        torch.set_grad_enabled(False)

        self.config = MSTxt2VidConfig()
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_generating = False

        self._build_ui()
        self._load_from_config()

        # connect cross-thread logging signal
        self.log_message.connect(self._log_status)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Prompt
        prompt_label = QtWidgets.QLabel("Prompt:")
        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the video you want to generate...")
        layout.addWidget(prompt_label)
        layout.addWidget(self.prompt_edit, 1)

        # Numeric options
        grid = QtWidgets.QGridLayout()

        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(4, 256)
        self.frames_spin.setValue(16)
        grid.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        grid.addWidget(self.frames_spin, 0, 1)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(25)
        grid.addWidget(QtWidgets.QLabel("Inference steps:"), 0, 2)
        grid.addWidget(self.steps_spin, 0, 3)

        self.guidance_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 30.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.25)
        self.guidance_spin.setValue(7.5)
        grid.addWidget(QtWidgets.QLabel("Guidance scale:"), 1, 0)
        grid.addWidget(self.guidance_spin, 1, 1)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)
        grid.addWidget(QtWidgets.QLabel("Seed (-1 = random):"), 1, 2)
        grid.addWidget(self.seed_spin, 1, 3)

        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(8)
        grid.addWidget(QtWidgets.QLabel("FPS:"), 2, 0)
        grid.addWidget(self.fps_spin, 2, 1)

        layout.addLayout(grid)

        # Checkboxes
        checks_layout = QtWidgets.QHBoxLayout()
        self.cpu_offload_check = QtWidgets.QCheckBox("Enable CPU offload")
        self.vae_slicing_check = QtWidgets.QCheckBox("Enable VAE slicing")
        self.cpu_offload_check.setChecked(True)
        self.vae_slicing_check.setChecked(True)
        checks_layout.addWidget(self.cpu_offload_check)
        checks_layout.addWidget(self.vae_slicing_check)
        layout.addLayout(checks_layout)

        # Output dir
        out_layout = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit()
        self.output_browse_btn = QtWidgets.QPushButton("Browse...")
        out_layout.addWidget(QtWidgets.QLabel("Output folder:"))
        out_layout.addWidget(self.output_edit, 1)
        out_layout.addWidget(self.output_browse_btn)
        layout.addLayout(out_layout)

        # Model dir
        model_layout = QtWidgets.QHBoxLayout()
        self.model_dir_edit = QtWidgets.QLineEdit()
        self.model_browse_btn = QtWidgets.QPushButton("Browse...")
        model_layout.addWidget(QtWidgets.QLabel("Model folder:"))
        model_layout.addWidget(self.model_dir_edit, 1)
        model_layout.addWidget(self.model_browse_btn)
        layout.addLayout(model_layout)

        # Status + button
        bottom_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready.")
        self.generate_btn = QtWidgets.QPushButton("Generate video")
        bottom_layout.addWidget(self.status_label, 1)
        bottom_layout.addWidget(self.generate_btn)
        layout.addLayout(bottom_layout)

        # Connections
        self.output_browse_btn.clicked.connect(self._choose_output_dir)
        self.model_browse_btn.clicked.connect(self._choose_model_dir)
        self.generate_btn.clicked.connect(self._on_generate_clicked)

    def _load_from_config(self):
        cfg = self.config.data
        self.prompt_edit.setPlainText(cfg.get("prompt", ""))
        self.frames_spin.setValue(int(cfg.get("num_frames", 16)))
        self.steps_spin.setValue(int(cfg.get("num_inference_steps", 25)))
        self.guidance_spin.setValue(float(cfg.get("guidance_scale", 7.5)))
        self.seed_spin.setValue(int(cfg.get("seed", -1)))
        self.fps_spin.setValue(int(cfg.get("fps", 8)))
        self.cpu_offload_check.setChecked(bool(cfg.get("use_cpu_offload", True)))
        self.vae_slicing_check.setChecked(bool(cfg.get("use_vae_slicing", True)))
        self.output_edit.setText(cfg.get("output_dir", os.path.join("output", "video")))
        self.model_dir_edit.setText(cfg.get("model_local_dir", os.path.join("models", "text-to-video-ms-1.7b")))

    def _update_config_from_ui(self):
        self.config["prompt"] = self.prompt_edit.toPlainText().strip()
        self.config["num_frames"] = int(self.frames_spin.value())
        self.config["num_inference_steps"] = int(self.steps_spin.value())
        self.config["guidance_scale"] = float(self.guidance_spin.value())
        self.config["seed"] = int(self.seed_spin.value())
        self.config["fps"] = int(self.fps_spin.value())
        self.config["use_cpu_offload"] = bool(self.cpu_offload_check.isChecked())
        self.config["use_vae_slicing"] = bool(self.vae_slicing_check.isChecked())
        self.config["output_dir"] = self.output_edit.text().strip()
        self.config["model_local_dir"] = self.model_dir_edit.text().strip()
        self.config.save()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _choose_output_dir(self):
        current = self.output_edit.text().strip()
        if not os.path.isabs(current):
            current = os.path.join(ROOT_DIR, current)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", current)
        if path:
            rel = os.path.relpath(path, ROOT_DIR)
            self.output_edit.setText(rel)
            self._update_config_from_ui()

    def _choose_model_dir(self):
        current = self.model_dir_edit.text().strip()
        if not os.path.isabs(current):
            current = os.path.join(ROOT_DIR, current)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose model folder", current)
        if path:
            rel = os.path.relpath(path, ROOT_DIR)
            self.model_dir_edit.setText(rel)
            self._update_config_from_ui()

    @QtCore.Slot(str)
    def _log_status(self, text):
        print("[MS-TXT2VID]", text)
        self.status_label.setText(text)

    def _resolve_path(self, p: str) -> str:
        """Resolve config/ui path to an absolute path under ROOT_DIR if relative."""
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return os.path.join(ROOT_DIR, p)

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        model_dir = self._resolve_path(self.model_dir_edit.text().strip())
        if not model_dir:
            raise RuntimeError("Model folder not set. Run ms_install.bat first.")
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"Model folder does not exist: {model_dir}")

        self._log_status("Loading pipeline... (first time can take a while)")
        QtWidgets.QApplication.processEvents()

        # Choose dtype & variant depending on device
        if self.device == "cuda":
            dtype = torch.float16
            variant = "fp16"
        else:
            dtype = torch.float32
            variant = None

        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            variant=variant,
            local_files_only=True,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Device placement / memory options
        if self.device == "cuda":
            if self.cpu_offload_check.isChecked():
                # Layers are moved between CPU and GPU to save VRAM
                pipe.enable_model_cpu_offload()
            else:
                # Full GPU mode
                pipe.to("cuda")

        # VAE slicing is safe on both CPU and GPU
        if self.vae_slicing_check.isChecked():
            pipe.enable_vae_slicing()

        pipe.set_progress_bar_config(disable=False)

        self.pipeline = pipe
        return self.pipeline

    def _on_generate_clicked(self):
        if self.is_generating:
            QtWidgets.QMessageBox.information(self, "Busy", "Generation already in progress.")
            return

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QtWidgets.QMessageBox.warning(self, "No prompt", "Please enter a prompt first.")
            return

        self._update_config_from_ui()
        self.is_generating = True
        self.generate_btn.setEnabled(False)
        self._log_status("Starting generation...")

        thread = threading.Thread(target=self._run_generation, daemon=True)
        thread.start()

    def _run_generation(self):
        try:
            pipe = self._ensure_pipeline()

            cfg = self.config.data
            prompt = cfg["prompt"]
            num_frames = int(cfg["num_frames"])
            steps = int(cfg["num_inference_steps"])
            guidance = float(cfg["guidance_scale"])
            seed = int(cfg["seed"])
            fps = int(cfg["fps"])
            output_dir_rel = cfg["output_dir"]

            output_dir = self._resolve_path(output_dir_rel)
            os.makedirs(output_dir, exist_ok=True)

            self._log_safe("Generating video...")
            generator = None
            if seed >= 0:
                # For CUDA we want a cuda generator; for CPU, a cpu generator
                gen_device = self.device if self.device == "cuda" else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(seed)

            result = pipe(
                prompt,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            frames = result.frames[0]

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ms_txt2vid_{ts}.mp4"
            out_path = os.path.join(output_dir, filename)
            export_to_video(frames, output_video_path=out_path, fps=fps)

            self._log_safe(f"Done: {out_path}")
        except Exception as e:
            self._log_safe(f"Error: {e}")
        finally:
            QtCore.QMetaObject.invokeMethod(
                self,
                "_on_generation_finished",
                QtCore.Qt.QueuedConnection,
            )

    @QtCore.Slot()
    def _on_generation_finished(self):
        self.is_generating = False
        self.generate_btn.setEnabled(True)
        if self.status_label.text().startswith("Starting"):
            self.status_label.setText("Ready.")

    def _log_safe(self, text):
        # emit a signal so Qt does a thread-safe queued call to _log_status
        self.log_message.emit(text)



class mstxt2vidPane(MSTxt2VidWindow):
    """Thin wrapper so FrameVision can import this as a tab pane."""
    def __init__(self, parent=None):
        super().__init__(parent=parent)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MSTxt2VidWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()