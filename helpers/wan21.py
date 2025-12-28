import sys
import os
import time
import random
import json
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QGroupBox,
    QSlider,
    QComboBox,
    QGridLayout,
)

# Lazy imports for heavy ML libs happen inside the worker.
_WAN21_T2V_PIPELINE = None
_WAN21_T2V_DEVICE = None

_WAN21_I2V_PIPELINE = None
_WAN21_I2V_DEVICE = None


def _get_root_dir() -> Path:
    """Return the app root (two levels up: root/helpers/wan21.py -> root)."""
    return Path(__file__).resolve().parents[1]


def _get_models_dir() -> Path:
    return _get_root_dir() / "models" / "wan21"


def _get_models_dir_gguf() -> Path:
    return _get_root_dir() / "models" / "wan21gguf"


def _get_sdcpp_dir() -> Path:
    return _get_root_dir() / ".wan21gguf_env" / "sdcpp"


def _get_sdcli_path() -> Path:
    d = _get_sdcpp_dir()
    # common locations after extracting sd.cpp zips
    candidates = [
        d / "sd-cli.exe",
        d / "bin" / "Release" / "sd-cli.exe",
        d / "bin" / "sd-cli.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _get_gguf_settings_path() -> Path:
    return _get_root_dir() / "presets" / "setsave" / "wan21gguf.json"


# Keep this in sync with presets/extra_env/wan21_guff.py
_GGUF_CATALOG = {
    # diffusion
    "wan21_t2v14b_q4_k_m": ("diffusion_models", "wan2.1-t2v-14b-Q4_K_M.gguf"),
    "wan21_t2v14b_q5_k_m": ("diffusion_models", "wan2.1-t2v-14b-Q5_K_M.gguf"),
    "wan21_t2v14b_q6_k":   ("diffusion_models", "wan2.1-t2v-14b-Q6_K.gguf"),
    "wan21_t2v14b_q8_0":   ("diffusion_models", "wan2.1-t2v-14b-Q8_0.gguf"),

    "wan21_i2v14b_480p_q4_k_m": ("diffusion_models", "wan2.1-i2v-14b-480p-Q4_K_M.gguf"),
    "wan21_i2v14b_480p_q5_k_m": ("diffusion_models", "wan2.1-i2v-14b-480p-Q5_K_M.gguf"),
    "wan21_i2v14b_480p_q6_k":   ("diffusion_models", "wan2.1-i2v-14b-480p-Q6_K.gguf"),
    "wan21_i2v14b_480p_q8_0":   ("diffusion_models", "wan2.1-i2v-14b-480p-Q8_0.gguf"),

    # encoders / companions
    "umt5_xxl_q6_k": ("text_encoders", "umt5-xxl-encoder-Q6_K.gguf"),
    "umt5_xxl_q8_0": ("text_encoders", "umt5-xxl-encoder-Q8_0.gguf"),
    "wan21_vae": ("vae", "wan_2.1_vae.safetensors"),
    "clip_vision_h": ("clip_vision", "clip_vision_h.safetensors"),
}


def _gguf_path_for_key(key: str) -> Path:
    sub, name = _GGUF_CATALOG[key]
    return _get_models_dir_gguf() / sub / name

def _get_default_output_dir() -> Path:
    return _get_root_dir() / "output" / "wan21"


def _get_i2v_model_dir() -> Path:
    """Expected location of the Wan I2V diffusers checkpoint.

    You can place for example:
      Wan-AI/Wan2.1-I2V-1.3B-480P-diffusers

    into: models/wan21/Wan2.1-I2V-1.3B-480P-diffusers
    """
    return _get_models_dir() / "Wan2.1-I2V-1.3B-480P-diffusers"


class GenerateConfig:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int | None,
        use_random_seed: bool,
        output_dir: Path,
        backend: str = "diffusers",  # "diffusers" or "gguf"
        mode: str = "t2v",  # "t2v" or "i2v"
        init_image_path: str | None = None,
        gguf_t2v_model_key: str = "wan21_t2v14b_q4_k_m",
        gguf_i2v_model_key: str = "wan21_i2v14b_480p_q4_k_m",
        gguf_t5_key: str = "umt5_xxl_q6_k",
        gguf_vae_key: str = "wan21_vae",
        gguf_clip_vision_key: str = "clip_vision_h",
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.use_random_seed = use_random_seed
        self.output_dir = output_dir
        self.backend = backend
        self.mode = mode
        self.init_image_path = init_image_path
        self.gguf_t2v_model_key = gguf_t2v_model_key
        self.gguf_i2v_model_key = gguf_i2v_model_key
        self.gguf_t5_key = gguf_t5_key
        self.gguf_vae_key = gguf_vae_key
        self.gguf_clip_vision_key = gguf_clip_vision_key


class Wan21Worker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, config: GenerateConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config

    # Logging ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.progress.emit(msg)

    # Pipelines -------------------------------------------------------
    def _ensure_t2v_pipeline(self, device: str):
        """Load Wan 2.1 T2V 1.3B Diffusers pipeline on demand."""
        global _WAN21_T2V_PIPELINE, _WAN21_T2V_DEVICE

        if _WAN21_T2V_PIPELINE is not None:
            return _WAN21_T2V_PIPELINE

        self._log("Importing T2V diffusers / torch...")
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        import torch

        models_dir = _get_models_dir()
        model_path = models_dir / "Wan2.1-T2V-1.3B-Diffusers"
        if not model_path.exists():
            raise RuntimeError(
                f"Missing T2V model directory: {model_path}\n"
                "Run presets/extra_env/wan21_install.bat first to download the models."
            )

        self._log(f"Loading Wan2.1 T2V pipeline from:\n  {model_path}")
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        flow_shift = 3.0  # 3.0 for 480p, 5.0 for 720p
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift,
        )

        pipe = WanPipeline.from_pretrained(
            str(model_path),
            vae=vae,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        pipe.scheduler = scheduler

        self._log(f"Moving T2V pipeline to device: {device}")
        pipe.to(device)

        _WAN21_T2V_PIPELINE = pipe
        _WAN21_T2V_DEVICE = device
        self._log("T2V model loaded and ready.")
        return pipe

    def _ensure_i2v_pipeline(self, device: str):
        """Load Wan 2.1 I2V 1.3B Diffusers pipeline on demand.

        This expects a diffusers-style I2V model directory such as
        'Wan2.1-I2V-1.3B-480P-diffusers' under models/wan21.
        """
        global _WAN21_I2V_PIPELINE, _WAN21_I2V_DEVICE

        if _WAN21_I2V_PIPELINE is not None:
            return _WAN21_I2V_PIPELINE

        self._log("Importing I2V diffusers / torch...")
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel
        import torch

        model_path = _get_i2v_model_dir()
        if not model_path.exists():
            raise RuntimeError(
                f"Missing I2V model directory: {model_path}\n"
                "Please download a Wan2.1 I2V diffusers checkpoint (for example, Wan-AI/Wan2.1-I2V-1.3B-480P-diffusers)\n"
                "into models/wan21 and retry."
            )

        self._log(f"Loading Wan2.1 I2V pipeline from:\n  {model_path}")
        image_encoder = CLIPVisionModel.from_pretrained(
            str(model_path),
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        pipe = WanImageToVideoPipeline.from_pretrained(
            str(model_path),
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        self._log(f"Moving I2V pipeline to device: {device}")
        pipe.to(device)

        _WAN21_I2V_PIPELINE = pipe
        _WAN21_I2V_DEVICE = device
        self._log("I2V model loaded and ready.")
        return pipe


    # GGUF (stable-diffusion.cpp / sd-cli) -------------------------------
    def _run_gguf(self) -> None:
        """
        Run Wan2.1 via stable-diffusion.cpp (sd-cli.exe).

        This tries a couple flag variants to stay compatible with sd-cli changes.
        """
        cfg = self.config

        sdcli = _get_sdcli_path()
        if not sdcli.exists():
            raise RuntimeError(
                "sd-cli.exe not found.\n\n"
                "Run presets/extra_env/wan21_guff_install.bat first (it downloads stable-diffusion.cpp)."
            )

        # resolve model files
        model_key = cfg.gguf_i2v_model_key if cfg.mode == "i2v" else cfg.gguf_t2v_model_key
        diffusion = _gguf_path_for_key(model_key)
        t5 = _gguf_path_for_key(cfg.gguf_t5_key)
        vae = _gguf_path_for_key(cfg.gguf_vae_key)
        clipv = _gguf_path_for_key(cfg.gguf_clip_vision_key)

        missing = [p for p in [diffusion, t5, vae] if not p.exists()]
        if cfg.mode == "i2v":
            missing += [clipv] if not clipv.exists() else []
        if missing:
            msg = "Missing GGUF companion files:\n" + "\n".join(f"- {m}" for m in missing) + "\n\n"
            msg += "Use the downloader:\n"
            msg += "  .\\.wan21gguf_env\\venv\\Scripts\\python.exe presets\\extra_env\\wan21_guff.py download <key> --with-deps\n"
            raise RuntimeError(msg)

        # output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_suffix = "i2v" if cfg.mode == "i2v" else "t2v"
        seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31 - 1)
        out_path = cfg.output_dir / f"wan21_{out_suffix}_gguf_{timestamp}_{seed}.mp4"

        # base command
        base = [
            str(sdcli),
            "-M", "vid_gen",
            "--diffusion-model", str(diffusion),
            "--vae", str(vae),
            "--t5xxl", str(t5),
            "-p", cfg.prompt,
            "--cfg-scale", str(cfg.guidance_scale),
            "--sampling-method", "euler",
            "--seed", str(seed),
        ]
        if cfg.negative_prompt.strip():
            base += ["-n", cfg.negative_prompt.strip()]

        if cfg.mode == "i2v":
            if not cfg.init_image_path:
                raise ValueError("Image-to-video mode enabled but no image was selected.")
            image_path = Path(cfg.init_image_path)
            if not image_path.exists():
                raise ValueError(f"Input image does not exist: {image_path}")
            base += ["--clip_vision", str(clipv)]
            # init-image flag differs across builds; we will try a few.
            init_flag_candidates = ["--init-img", "--init-image", "--image", "--input"]
        else:
            init_flag_candidates = []

        # Some flags vary across sd-cli versions; try a few combinations.
        steps_flags = ["--steps", "--num-steps"]
        width_flags = ["--width", "-W"]
        height_flags = ["--height", "-H"]
        frames_flags = ["--frames", "--num-frames", "--num_frames"]
        fps_flags = ["--fps", "--frame-rate"]
        out_flags = ["-o", "--output", "--out"]

        def try_run(cmd: list[str]) -> tuple[int, str, str]:
            self._log("[GGUF] Running:\n  " + " ".join(cmd))
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.stdout:
                self._log(p.stdout.strip())
            if p.stderr:
                self._log(p.stderr.strip())
            return p.returncode, p.stdout, p.stderr

        last_err = ""
        for steps_f in steps_flags:
            for wf in width_flags:
                for hf in height_flags:
                    for ff in frames_flags:
                        for fpsf in fps_flags:
                            for outf in out_flags:
                                cmd = base.copy()
                                cmd += [steps_f, str(cfg.num_inference_steps)]
                                cmd += [wf, str(cfg.width), hf, str(cfg.height)]
                                cmd += [ff, str(cfg.num_frames)]
                                cmd += [fpsf, str(cfg.fps)]
                                if cfg.mode == "i2v":
                                    # insert init-image flag as one of candidates
                                    ok = False
                                    for initf in init_flag_candidates:
                                        cmd2 = cmd.copy()
                                        cmd2 += [initf, str(Path(cfg.init_image_path))]
                                        cmd2 += [outf, str(out_path)]
                                        rc, _, err = try_run(cmd2)
                                        if rc == 0:
                                            ok = True
                                            break
                                        last_err = err or last_err
                                        # if unknown-arg, keep trying; else also try.
                                    if ok:
                                        break
                                    else:
                                        continue
                                else:
                                    cmd += [outf, str(out_path)]
                                    rc, _, err = try_run(cmd)
                                    if rc == 0:
                                        break
                                    last_err = err or last_err
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break

        if not out_path.exists():
            raise RuntimeError(
                "GGUF run finished but output file was not found:\n"
                f"  {out_path}\n\n"
                "sd-cli error output (last):\n"
                f"{last_err}"
            )

        self._log(f"Saving video to: {out_path}")
        self.finished.emit(str(out_path))

    # Main worker -----------------------------------------------------
    def run(self) -> None:
        try:
            from diffusers.utils import export_to_video, load_image
            import torch

            cfg = self.config

            if getattr(cfg, "backend", "diffusers") == "gguf":
                self._run_gguf()
                return

            if not cfg.prompt.strip():
                raise ValueError("Prompt is empty. Please enter a prompt.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                self._log("[WARNING] CUDA GPU not detected. Falling back to CPU (very slow and may OOM).")

            # Seed handling
            if cfg.use_random_seed or cfg.seed is None:
                seed = random.randint(0, 2**31 - 1)
            else:
                seed = int(cfg.seed)

            self._log(f"Using seed: {seed}")
            generator = torch.Generator(device=device).manual_seed(seed)

            # Ensure output directory exists
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_suffix = "i2v" if cfg.mode == "i2v" else "t2v"
            out_path = cfg.output_dir / f"wan21_{out_suffix}_{timestamp}_{seed}.mp4"

            if cfg.mode == "i2v":
                # ----------------- I2V branch ---------------------
                if not cfg.init_image_path:
                    raise ValueError("Image-to-video mode enabled but no image was selected.")

                image_path = Path(cfg.init_image_path)
                if not image_path.exists():
                    raise ValueError(f"Input image does not exist: {image_path}")

                self._log(f"[I2V] Loading init image:\n  {image_path}")
                init_image = load_image(str(image_path))

                pipe = self._ensure_i2v_pipeline(device)
                self._log(
                    f"[I2V] Generating video at {cfg.width}x{cfg.height}, {cfg.num_frames} frames, "
                    f"guidance_scale={cfg.guidance_scale}, steps={cfg.num_inference_steps}..."
                )

                result = pipe(
                    prompt=cfg.prompt,
                    image=init_image,
                    negative_prompt=cfg.negative_prompt or None,
                    num_frames=cfg.num_frames,
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=generator,
                )
                frames = result.frames[0]
            else:
                # ----------------- T2V branch ---------------------
                pipe = self._ensure_t2v_pipeline(device)
                self._log(
                    f"[T2V] Generating video at {cfg.width}x{cfg.height}, {cfg.num_frames} frames, "
                    f"guidance_scale={cfg.guidance_scale}, steps={cfg.num_inference_steps}..."
                )

                result = pipe(
                    prompt=cfg.prompt,
                    negative_prompt=cfg.negative_prompt or None,
                    height=cfg.height,
                    width=cfg.width,
                    num_frames=cfg.num_frames,
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=generator,
                )
                frames = result.frames[0]

            self._log(f"Saving video to: {out_path}")
            export_to_video(frames, str(out_path), fps=cfg.fps)

            self.finished.emit(str(out_path))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class Wan21Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wan 2.1 - T2V / I2V")
        self.resize(1024, 768)

        self._thread: QThread | None = None
        self._worker: Wan21Worker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)


        # Backend (Diffusers / GGUF)
        backend_group = QGroupBox("Backend", self)
        backend_layout = QHBoxLayout(backend_group)
        backend_layout.addWidget(QLabel("Engine:", backend_group))

        self.backend_combo = QComboBox(backend_group)
        self.backend_combo.addItem("Diffusers (normal)", userData="diffusers")
        self.backend_combo.addItem("GGUF (sd.cpp)", userData="gguf")
        backend_layout.addWidget(self.backend_combo)

        self.gguf_group = QGroupBox("GGUF models (sd.cpp)", self)
        gguf_layout = QGridLayout(self.gguf_group)

        gguf_layout.addWidget(QLabel("T2V diffusion:", self.gguf_group), 0, 0)
        self.gguf_t2v_combo = QComboBox(self.gguf_group)
        gguf_layout.addWidget(self.gguf_t2v_combo, 0, 1)

        gguf_layout.addWidget(QLabel("I2V diffusion:", self.gguf_group), 1, 0)
        self.gguf_i2v_combo = QComboBox(self.gguf_group)
        gguf_layout.addWidget(self.gguf_i2v_combo, 1, 1)

        gguf_layout.addWidget(QLabel("UMT5 encoder:", self.gguf_group), 2, 0)
        self.gguf_t5_combo = QComboBox(self.gguf_group)
        gguf_layout.addWidget(self.gguf_t5_combo, 2, 1)

        gguf_layout.addWidget(QLabel("VAE:", self.gguf_group), 3, 0)
        self.gguf_vae_combo = QComboBox(self.gguf_group)
        gguf_layout.addWidget(self.gguf_vae_combo, 3, 1)

        gguf_layout.addWidget(QLabel("CLIP-Vision:", self.gguf_group), 4, 0)
        self.gguf_clipv_combo = QComboBox(self.gguf_group)
        gguf_layout.addWidget(self.gguf_clipv_combo, 4, 1)

        btn_row = QHBoxLayout()
        self.gguf_refresh_btn = QPushButton("Refresh", self.gguf_group)
        self.gguf_refresh_btn.clicked.connect(self._refresh_gguf_lists)
        btn_row.addWidget(self.gguf_refresh_btn)

        self.gguf_download_btn = QPushButton("Download selected (with deps)", self.gguf_group)
        self.gguf_download_btn.clicked.connect(self._download_selected_gguf)
        btn_row.addWidget(self.gguf_download_btn)

        btn_row.addStretch(1)
        gguf_layout.addLayout(btn_row, 5, 0, 1, 2)

        layout.addWidget(backend_group)
        layout.addWidget(self.gguf_group)

        # Prompt group
        prompt_group = QGroupBox("Prompt", self)
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QTextEdit(prompt_group)
        self.prompt_edit.setPlaceholderText("Describe the video you want Wan 2.1 to generate...")
        prompt_layout.addWidget(self.prompt_edit)

        # Negative prompt
        neg_group = QGroupBox("Negative prompt", self)
        neg_layout = QVBoxLayout(neg_group)
        self.negative_edit = QTextEdit(neg_group)
        self.negative_edit.setPlaceholderText("Things you do NOT want in the video (optional)...")
        neg_layout.addWidget(self.negative_edit)

        # Settings group
        settings_group = QGroupBox("Settings", self)
        settings_layout = QVBoxLayout(settings_group)

        # Resolution / frames
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Width:", self))
        self.width_spin = QSpinBox(self)
        self.width_spin.setRange(256, 1920)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(832)
        row1.addWidget(self.width_spin)

        self.width_slider = QSlider(Qt.Horizontal, self)
        self.width_slider.setRange(256, 1920)
        self.width_slider.setSingleStep(64)
        self.width_slider.setPageStep(64)
        self.width_slider.setValue(self.width_spin.value())
        row1.addWidget(self.width_slider)

        row1.addWidget(QLabel("Height:", self))
        self.height_spin = QSpinBox(self)
        self.height_spin.setRange(256, 1080)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(480)
        row1.addWidget(self.height_spin)

        self.height_slider = QSlider(Qt.Horizontal, self)
        self.height_slider.setRange(256, 1080)
        self.height_slider.setSingleStep(64)
        self.height_slider.setPageStep(64)
        self.height_slider.setValue(self.height_spin.value())
        row1.addWidget(self.height_slider)

        row1.addWidget(QLabel("Frames:", self))
        self.frames_spin = QSpinBox(self)
        self.frames_spin.setRange(8, 161)
        self.frames_spin.setValue(81)
        row1.addWidget(self.frames_spin)

        settings_layout.addLayout(row1)

        # FPS row
        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("FPS:", self))
        self.fps_spin = QSpinBox(self)
        self.fps_spin.setRange(4, 60)
        self.fps_spin.setValue(16)
        row1b.addWidget(self.fps_spin)

        self.fps_slider = QSlider(Qt.Horizontal, self)
        self.fps_slider.setRange(4, 60)
        self.fps_slider.setSingleStep(1)
        self.fps_slider.setPageStep(5)
        self.fps_slider.setValue(self.fps_spin.value())
        row1b.addWidget(self.fps_slider)

        row1b.addStretch(1)
        settings_layout.addLayout(row1b)

        # Guidance / steps / seed
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Guidance scale:", self))
        self.guidance_spin = QDoubleSpinBox(self)
        self.guidance_spin.setDecimals(1)
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(5.0)
        row2.addWidget(self.guidance_spin)

        self.guidance_slider = QSlider(Qt.Horizontal, self)
        self.guidance_slider.setRange(0, 200)  # 0.0 - 20.0 mapped as *0.1
        self.guidance_slider.setSingleStep(5)  # 0.5 steps
        self.guidance_slider.setPageStep(10)
        self.guidance_slider.setValue(int(self.guidance_spin.value() * 10))
        row2.addWidget(self.guidance_slider)

        row2.addWidget(QLabel("Steps:", self))
        self.steps_spin = QSpinBox(self)
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(50)
        row2.addWidget(self.steps_spin)

        self.steps_slider = QSlider(Qt.Horizontal, self)
        self.steps_slider.setRange(1, 200)
        self.steps_slider.setSingleStep(1)
        self.steps_slider.setPageStep(10)
        self.steps_slider.setValue(self.steps_spin.value())
        row2.addWidget(self.steps_slider)

        self.random_seed_check = QCheckBox("Random seed", self)
        self.random_seed_check.setChecked(True)
        row2.addWidget(self.random_seed_check)

        row2.addWidget(QLabel("Seed:", self))
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setEnabled(False)
        row2.addWidget(self.seed_spin)

        self.random_seed_check.toggled.connect(self.seed_spin.setDisabled)

        settings_layout.addLayout(row2)

        # I2V group
        i2v_group = QGroupBox("Image-to-Video (optional)", self)
        i2v_layout = QHBoxLayout(i2v_group)
        self.i2v_enable_check = QCheckBox("Enable image-to-video mode", i2v_group)
        i2v_layout.addWidget(self.i2v_enable_check)
        self.i2v_path_edit = QLineEdit(i2v_group)
        self.i2v_path_edit.setPlaceholderText("Select an input image to enable I2V...")
        i2v_layout.addWidget(self.i2v_path_edit)
        i2v_browse_btn = QPushButton("Browse image...", i2v_group)
        i2v_browse_btn.clicked.connect(self._choose_i2v_image)
        i2v_layout.addWidget(i2v_browse_btn)

        settings_layout.addWidget(i2v_group)


        # Output directory
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Output folder:", self))
        self.output_edit = QLineEdit(self)
        self.output_edit.setText(str(_get_default_output_dir()))
        row3.addWidget(self.output_edit)
        browse_btn = QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._choose_output_dir)
        row3.addWidget(browse_btn)
        settings_layout.addLayout(row3)

        # Run controls
        row4 = QHBoxLayout()
        self.run_btn = QPushButton("Generate video", self)
        self.run_btn.clicked.connect(self._on_run_clicked)
        row4.addWidget(self.run_btn)

        self.open_btn = QPushButton("Open output folder", self)
        self.open_btn.clicked.connect(self._open_output_folder)
        row4.addWidget(self.open_btn)

        row4.addStretch(1)
        settings_layout.addLayout(row4)

        # Progress + log
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # indeterminate by default
        self.progress_bar.setVisible(False)

        self.log_edit = QTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("Log output will appear here...")

        # Assemble layout
        layout.addWidget(prompt_group)
        layout.addWidget(neg_group)
        layout.addWidget(settings_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_edit, stretch=1)

        # Sync sliders <-> spinboxes
        self._wire_spin_slider_sync()

    # Slider / spin sync -------------------------------------------------
    def _wire_spin_slider_sync(self) -> None:
        # Width
        self.width_slider.valueChanged.connect(self._on_width_slider_changed)
        self.width_spin.valueChanged.connect(self._on_width_spin_changed)

        # Height
        self.height_slider.valueChanged.connect(self._on_height_slider_changed)
        self.height_spin.valueChanged.connect(self._on_height_spin_changed)

        # GGUF UI init
        self._refresh_gguf_lists()
        self.backend_combo.currentIndexChanged.connect(self._update_backend_visibility)
        self._update_backend_visibility()
        self._load_settings()


        # FPS
        self.fps_slider.valueChanged.connect(self.fps_spin.setValue)
        self.fps_spin.valueChanged.connect(self.fps_slider.setValue)

        # Guidance (0-20 -> slider 0-200)
        self.guidance_slider.valueChanged.connect(self._on_guidance_slider_changed)
        self.guidance_spin.valueChanged.connect(self._on_guidance_spin_changed)

        # Steps
        self.steps_slider.valueChanged.connect(self.steps_spin.setValue)
        self.steps_spin.valueChanged.connect(self.steps_slider.setValue)

    def _quantize_dim(self, value: int, minimum: int, step: int) -> int:
        # Snap resolution to multiples of step from minimum
        if value < minimum:
            return minimum
        offset = value - minimum
        snapped = step * round(offset / step) + minimum
        return snapped

    def _on_width_slider_changed(self, value: int) -> None:
        snapped = self._quantize_dim(value, 256, 64)
        if snapped != value:
            self.width_slider.blockSignals(True)
            self.width_slider.setValue(snapped)
            self.width_slider.blockSignals(False)
        self.width_spin.setValue(snapped)

    def _on_width_spin_changed(self, value: int) -> None:
        snapped = self._quantize_dim(value, 256, 64)
        if snapped != value:
            self.width_spin.blockSignals(True)
            self.width_spin.setValue(snapped)
            self.width_spin.blockSignals(False)
        self.width_slider.setValue(snapped)

    def _on_height_slider_changed(self, value: int) -> None:
        snapped = self._quantize_dim(value, 256, 64)
        if snapped != value:
            self.height_slider.blockSignals(True)
            self.height_slider.setValue(snapped)
            self.height_slider.blockSignals(False)
        self.height_spin.setValue(snapped)

    def _on_height_spin_changed(self, value: int) -> None:
        snapped = self._quantize_dim(value, 256, 64)
        if snapped != value:
            self.height_spin.blockSignals(True)
            self.height_spin.setValue(snapped)
            self.height_spin.blockSignals(False)
        self.height_slider.setValue(snapped)

    def _on_guidance_slider_changed(self, value: int) -> None:
        # Slider is 0-200 -> 0.0-20.0 in 0.1 steps
        v = value / 10.0
        self.guidance_spin.setValue(v)

    def _on_guidance_spin_changed(self, value: float) -> None:
        slider_val = int(round(value * 10.0))
        if slider_val < self.guidance_slider.minimum():
            slider_val = self.guidance_slider.minimum()
        if slider_val > self.guidance_slider.maximum():
            slider_val = self.guidance_slider.maximum()
        self.guidance_slider.setValue(slider_val)

    # UI helpers ---------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_edit.append(text)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def _update_backend_visibility(self) -> None:
        backend = self.backend_combo.currentData() or "diffusers"
        is_gguf = backend == "gguf"
        self.gguf_group.setVisible(is_gguf)

    def _refresh_gguf_lists(self) -> None:
        # Populate GGUF comboboxes from the in-file catalog.
        try:
            models_dir = _get_models_dir_gguf()
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Don't crash the UI if the folder can't be created.
            pass

        def fill_combo(combo: QComboBox, entries: list[tuple[str, str]]) -> None:
            combo.blockSignals(True)
            combo.clear()
            for key, filename in entries:
                try:
                    p = _gguf_path_for_key(key)
                    label = f"{filename}  ✓" if p.exists() else f"{filename}  (missing)"
                except Exception:
                    label = filename
                combo.addItem(label, userData=key)
            combo.blockSignals(False)

        # Build lists
        t2v = []
        i2v = []
        t5 = []
        vae = []
        clipv = []

        for key, (category, filename) in _GGUF_CATALOG.items():
            if category == "diffusion_models":
                if "_t2v" in key:
                    t2v.append((key, filename))
                elif "_i2v" in key:
                    i2v.append((key, filename))
            elif category == "text_encoders":
                t5.append((key, filename))
            elif category == "vae":
                vae.append((key, filename))
            elif category == "clip_vision":
                clipv.append((key, filename))

        # Stable order: prefer smaller quants first (roughly)
        def sort_key(item: tuple[str, str]) -> tuple[int, str]:
            k = item[0]
            # crude quant ordering
            order = 99
            if "q4" in k:
                order = 4
            elif "q5" in k:
                order = 5
            elif "q6" in k:
                order = 6
            elif "q8" in k:
                order = 8
            return (order, k)

        t2v.sort(key=sort_key)
        i2v.sort(key=sort_key)
        t5.sort(key=lambda x: x[0])
        vae.sort(key=lambda x: x[0])
        clipv.sort(key=lambda x: x[0])

        fill_combo(self.gguf_t2v_combo, t2v)
        fill_combo(self.gguf_i2v_combo, i2v)
        fill_combo(self.gguf_t5_combo, t5)
        fill_combo(self.gguf_vae_combo, vae)
        fill_combo(self.gguf_clipv_combo, clipv)

    def _download_selected_gguf(self) -> None:
        # Runs presets/extra_env/wan21_guff.py to fetch the selected model and companions.
        root = _get_root_dir()
        downloader = root / "presets" / "extra_env" / "wan21_guff.py"
        if not downloader.exists():
            QMessageBox.critical(self, "WAN 2.1 GGUF", f"Downloader not found:\n{downloader}")
            return

        backend = self.backend_combo.currentData() or "diffusers"
        if backend != "gguf":
            # Switch automatically to GGUF to reduce confusion.
            for i in range(self.backend_combo.count()):
                if self.backend_combo.itemData(i) == "gguf":
                    self.backend_combo.setCurrentIndex(i)
                    break

        key = (self.gguf_i2v_combo.currentData() if self.i2v_enable_check.isChecked() else self.gguf_t2v_combo.currentData())
        key = key or ("wan21_i2v14b_480p_q4_k_m" if self.i2v_enable_check.isChecked() else "wan21_t2v14b_q4_k_m")

        t5_key = self.gguf_t5_combo.currentData() or "umt5_xxl_q6_k"
        vae_key = self.gguf_vae_combo.currentData() or "wan21_vae"
        clipv_key = self.gguf_clipv_combo.currentData() or "clip_vision_h"

        # Prefer the dedicated env python if present.
        env_py = root / ".wan21gguf_env" / "venv" / "Scripts" / "python.exe"
        py = str(env_py) if env_py.exists() else sys.executable

        cmd = [
            py,
            "-u",
            str(downloader),
            "download",
            str(key),
            "--root",
            str(root),
            "--with-deps",
            "--t5-key",
            str(t5_key),
            "--vae-key",
            str(vae_key),
            "--clip-vision-key",
            str(clipv_key),
        ]

        self._append_log("=== Downloading GGUF model (and deps) ===")
        self._append_log(" ".join(cmd))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "WAN 2.1 GGUF", f"Failed to run downloader:\n{exc}")
            return

        if proc.stdout:
            self._append_log(proc.stdout.rstrip())
        if proc.stderr:
            self._append_log(proc.stderr.rstrip())

        if proc.returncode != 0:
            QMessageBox.critical(self, "WAN 2.1 GGUF", f"Downloader failed (code {proc.returncode}). Check the log.")
            return

        self._append_log("Download complete.")
        self._refresh_gguf_lists()

    def _load_settings(self) -> None:
        # Settings file for WAN 2.1 GGUF (and basic UI state)
        path = _get_gguf_settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}

        # Backend
        backend = data.get("backend", "diffusers")
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == backend:
                self.backend_combo.setCurrentIndex(i)
                break

        # Basics
        try:
            self.width_spin.setValue(int(data.get("width", self.width_spin.value())))
            self.height_spin.setValue(int(data.get("height", self.height_spin.value())))
            self.frames_spin.setValue(int(data.get("num_frames", self.frames_spin.value())))
            self.fps_spin.setValue(int(data.get("fps", self.fps_spin.value())))
            self.steps_spin.setValue(int(data.get("num_inference_steps", self.steps_spin.value())))
            self.guidance_spin.setValue(float(data.get("guidance_scale", self.guidance_spin.value())))
        except Exception:
            pass

        # Seed
        try:
            use_random = bool(data.get("use_random_seed", True))
            self.random_seed_check.setChecked(use_random)
            if not use_random and "seed" in data:
                self.seed_spin.setValue(int(data.get("seed", self.seed_spin.value())))
        except Exception:
            pass

        # Output dir
        try:
            out_dir = str(data.get("output_dir", "")).strip()
            if out_dir:
                self.output_edit.setText(out_dir)
        except Exception:
            pass

        # I2V
        try:
            self.i2v_enable_check.setChecked(bool(data.get("i2v_enabled", False)))
            i2v_path = str(data.get("i2v_image", "")).strip()
            if i2v_path:
                self.i2v_path_edit.setText(i2v_path)
        except Exception:
            pass

        # Text fields
        try:
            p = str(data.get("prompt", "")).strip()
            if p:
                self.prompt_edit.setPlainText(p)
            n = str(data.get("negative_prompt", "")).strip()
            if n:
                self.negative_edit.setPlainText(n)
        except Exception:
            pass

        # GGUF selections
        def set_combo_by_key(combo: QComboBox, wanted: str) -> None:
            for i in range(combo.count()):
                if combo.itemData(i) == wanted:
                    combo.setCurrentIndex(i)
                    return

        set_combo_by_key(self.gguf_t2v_combo, data.get("gguf_t2v_model_key", "wan21_t2v14b_q4_k_m"))
        set_combo_by_key(self.gguf_i2v_combo, data.get("gguf_i2v_model_key", "wan21_i2v14b_480p_q4_k_m"))
        set_combo_by_key(self.gguf_t5_combo, data.get("gguf_t5_key", "umt5_xxl_q6_k"))
        set_combo_by_key(self.gguf_vae_combo, data.get("gguf_vae_key", "wan21_vae"))
        set_combo_by_key(self.gguf_clipv_combo, data.get("gguf_clip_vision_key", "clip_vision_h"))

        self._update_backend_visibility()

    def _save_settings(self) -> None:
        path = _get_gguf_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "backend": (self.backend_combo.currentData() or "diffusers"),
            "prompt": self.prompt_edit.toPlainText().strip(),
            "negative_prompt": self.negative_edit.toPlainText().strip(),
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "num_frames": int(self.frames_spin.value()),
            "fps": int(self.fps_spin.value()),
            "guidance_scale": float(self.guidance_spin.value()),
            "num_inference_steps": int(self.steps_spin.value()),
            "use_random_seed": bool(self.random_seed_check.isChecked()),
            "seed": None if self.random_seed_check.isChecked() else int(self.seed_spin.value()),
            "output_dir": self.output_edit.text().strip(),
            "i2v_enabled": bool(self.i2v_enable_check.isChecked()),
            "i2v_image": self.i2v_path_edit.text().strip(),
            "gguf_t2v_model_key": (self.gguf_t2v_combo.currentData() or "wan21_t2v14b_q4_k_m"),
            "gguf_i2v_model_key": (self.gguf_i2v_combo.currentData() or "wan21_i2v14b_480p_q4_k_m"),
            "gguf_t5_key": (self.gguf_t5_combo.currentData() or "umt5_xxl_q6_k"),
            "gguf_vae_key": (self.gguf_vae_combo.currentData() or "wan21_vae"),
            "gguf_clip_vision_key": (self.gguf_clipv_combo.currentData() or "clip_vision_h"),
        }

        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            # Don't block generation if writing settings fails.
            pass

    def _choose_output_dir(self) -> None:
        current = self.output_edit.text().strip() or str(_get_default_output_dir())
        directory = QFileDialog.getExistingDirectory(self, "Select output folder", current)
        if directory:
            self.output_edit.setText(directory)

    def _open_output_folder(self) -> None:
        path_str = self.output_edit.text().strip() or str(_get_default_output_dir())
        path = Path(path_str)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            os.system(f"open '{path}'")
        else:
            os.system(f"xdg-open '{path}'")

    def _choose_i2v_image(self) -> None:
        current = self.i2v_path_edit.text().strip() or str(_get_root_dir())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input image for I2V",
            current,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )
        if file_path:
            self.i2v_path_edit.setText(file_path)
            self.i2v_enable_check.setChecked(True)

    # Run / thread handling ----------------------------------------------
    def _on_run_clicked(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "WAN 2.1", "Please enter a prompt first.")
            return

        backend = self.backend_combo.currentData() or "diffusers"
        self._save_settings()

        try:
            width = int(self.width_spin.value())
            height = int(self.height_spin.value())
            num_frames = int(self.frames_spin.value())
            fps = int(self.fps_spin.value())
            guidance = float(self.guidance_spin.value())
            num_steps = int(self.steps_spin.value())
            use_random = self.random_seed_check.isChecked()
            seed = None if use_random else int(self.seed_spin.value())
            out_dir = Path(self.output_edit.text().strip() or str(_get_default_output_dir()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "WAN 2.1", f"Invalid settings: {exc}")
            return

        mode = "i2v" if self.i2v_enable_check.isChecked() else "t2v"
        init_image_path = self.i2v_path_edit.text().strip() or None

        cfg = GenerateConfig(
            prompt=prompt,
            negative_prompt=self.negative_edit.toPlainText().strip(),
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            seed=seed,
            use_random_seed=use_random,
            output_dir=out_dir,
            backend=backend,
            gguf_t2v_model_key=(self.gguf_t2v_combo.currentData() or "wan21_t2v14b_q4_k_m"),
            gguf_i2v_model_key=(self.gguf_i2v_combo.currentData() or "wan21_i2v14b_480p_q4_k_m"),
            gguf_t5_key=(self.gguf_t5_combo.currentData() or "umt5_xxl_q6_k"),
            gguf_vae_key=(self.gguf_vae_combo.currentData() or "wan21_vae"),
            gguf_clip_vision_key=(self.gguf_clipv_combo.currentData() or "clip_vision_h"),
            mode=mode,
            init_image_path=init_image_path,
        )

        self._start_worker(cfg)

    def _start_worker(self, cfg: GenerateConfig) -> None:
        if self._thread is not None:
            QMessageBox.information(
                self,
                "WAN 2.1",
                "Generation is already running. Please wait for it to finish.",
            )
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.run_btn.setEnabled(False)
        mode_label = "I2V" if cfg.mode == "i2v" else "T2V"
        self._append_log(f"=== Starting Wan 2.1 {mode_label} generation ===")


        self._thread = QThread(self)
        self._worker = Wan21Worker(cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._append_log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _cleanup_thread(self) -> None:
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self._thread = None
        self._worker = None

    def _on_worker_finished(self, out_path: str) -> None:
        self._append_log(f"Finished! Saved video to:\n  {out_path}")
        QMessageBox.information(self, "WAN 2.1", f"Video saved to:\n{out_path}")
        self._cleanup_thread()

    def _on_worker_error(self, message: str) -> None:
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "WAN 2.1 - Error", message)
        self._cleanup_thread()


def main() -> None:
    app = QApplication(sys.argv)
    window = Wan21Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
