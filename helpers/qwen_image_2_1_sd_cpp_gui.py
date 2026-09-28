
from __future__ import annotations

import json
import re
import sys
import time
import struct
import hashlib
import os
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, Signal, QUrl, QThread
from PySide6.QtGui import QDesktopServices, QPixmap, QImageReader, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QGroupBox, QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QFileDialog, QMessageBox, QTabWidget,
    QScrollArea, QListWidget, QListWidgetItem, QAbstractItemView, QSplitter,
    QFrame, QProgressBar
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRESETS_BIN = ROOT / "presets" / "bin"
DEFAULT_MODELS_ROOT = ROOT / "models"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "qwen_image_2_1"
DEFAULT_SETTINGS = ROOT / "settings" / "qwen_image_2_1_sd_cpp_gui.json"


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class ElidedPathLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def setPath(self, text: str):
        self._full_text = text
        self.setToolTip(text)
        self._update_elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elide()

    def _update_elide(self):
        fm = self.fontMetrics()
        self.setText(fm.elidedText(self._full_text, Qt.ElideMiddle, max(80, self.width() - 8)))


@dataclass
class LoraEntry:
    path: str
    strength: float = 1.0


class ReferenceTile(QFrame):
    remove_requested = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(190)
        self.setMaximumWidth(240)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(170, 150)
        self.preview.setStyleSheet("background:#171a1f; border:1px solid #303741;")
        lay.addWidget(self.preview)

        name = QLabel(Path(path).name)
        name.setWordWrap(True)
        name.setToolTip(path)
        lay.addWidget(name)

        btn = QPushButton("Remove")
        btn.clicked.connect(lambda: self.remove_requested.emit(self.path))
        lay.addWidget(btn)
        self._load()

    def _load(self):
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            self.preview.setText("Unable to preview")
            return
        pix = QPixmap.fromImage(img)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._load()


def _qwen3vl_rewrite_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def _read_safetensors_header(path: Path):
    with path.open("rb") as f:
        hdr_len_raw = f.read(8)
        if len(hdr_len_raw) != 8:
            raise IOError(f"Invalid safetensors header: {path}")
        hdr_len = struct.unpack("<Q", hdr_len_raw)[0]
        hdr_bytes = f.read(hdr_len)
    return json.loads(hdr_bytes), 8 + hdr_len


def _collect_qwen3vl_shards(path: Path):
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            return sorted({path / n for n in idx["weight_map"].values()})
        single = path / "model.safetensors"
        if single.is_file():
            return [single]
        raise FileNotFoundError(
            f"No model.safetensors.index.json or model.safetensors found in:\n{path}"
        )
    if path.is_file():
        return [path]
    raise FileNotFoundError(str(path))


def convert_qwen3vl_hf_to_sd_cpp(input_path: Path, output_path: Path, progress_cb=None):
    """Consolidate original HF Qwen3-VL safetensors for stable-diffusion.cpp."""
    shards = _collect_qwen3vl_shards(input_path)

    entries = []
    for shard_index, shard_path in enumerate(shards):
        if progress_cb:
            progress_cb(f"Reading shard {shard_index + 1}/{len(shards)}: {shard_path.name}")
        hdr, data_off = _read_safetensors_header(shard_path)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            entries.append((_qwen3vl_rewrite_key(key), shard_path, data_off, info))

    entries.sort(key=lambda e: e[0])
    new_header = {}
    cur_offset = 0
    for new_key, shard_path, data_off, info in entries:
        start, end = info["data_offsets"]
        size = end - start
        new_header[new_key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [cur_offset, cur_offset + size],
        }
        cur_offset += size

    header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    header_json += b" " * ((-len(header_json)) % 8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    with temp_path.open("wb") as out:
        out.write(struct.pack("<Q", len(header_json)))
        out.write(header_json)

        total = len(entries)
        for i, (_, shard_path, data_off, info) in enumerate(entries):
            start, end = info["data_offsets"]
            remaining = end - start
            with shard_path.open("rb") as inp:
                inp.seek(data_off + start)
                while remaining > 0:
                    chunk = inp.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise IOError(f"Truncated tensor data in {shard_path}")
                    out.write(chunk)
                    remaining -= len(chunk)

            if progress_cb and (i % 64 == 0 or i == total - 1):
                progress_cb(f"Converting Qwen3-VL tensors: {i + 1}/{total}")

    temp_path.replace(output_path)
    return output_path




class SdCliDownloadWorker(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    completed = Signal(str, str)
    failed = Signal(str)

    RELEASES_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=30"
    CUDA_ASSET_RE = re.compile(r"^sd-.*-bin-win-cuda12-x64\.zip$", re.IGNORECASE)

    def __init__(self, target_dir: Path, parent=None):
        super().__init__(parent)
        self.target_dir = Path(target_dir)

    @staticmethod
    def _request(url: str):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "FrameVision-QwenImage21-sd-cli-updater",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        return urllib.request.urlopen(req, timeout=45)

    @classmethod
    def _find_newest_cuda_asset(cls):
        with cls._request(cls.RELEASES_API) as response:
            releases = json.loads(response.read().decode("utf-8"))

        if not isinstance(releases, list):
            raise RuntimeError("GitHub returned an unexpected releases response.")

        # GitHub returns releases newest first. Deliberately skip any newer release
        # that did not publish the Windows CUDA 12 x64 build.
        for release in releases:
            if release.get("draft"):
                continue
            for asset in release.get("assets") or []:
                name = str(asset.get("name") or "")
                if cls.CUDA_ASSET_RE.match(name) and "cudart" not in name.lower():
                    url = asset.get("browser_download_url")
                    if url:
                        return str(release.get("tag_name") or release.get("name") or "unknown"), name, str(url)

        raise RuntimeError(
            "No Windows CUDA 12 x64 stable-diffusion.cpp build was found in the newest 30 releases."
        )

    @staticmethod
    def _safe_extract_overwrite(zip_path: Path, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        root = target_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            for index, member in enumerate(members, 1):
                dest = (target_dir / member.filename).resolve()
                try:
                    dest.relative_to(root)
                except ValueError:
                    raise RuntimeError(f"Unsafe path in downloaded archive: {member.filename}")

                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(dest, "wb") as dst:
                        while True:
                            block = src.read(1024 * 1024)
                            if not block:
                                break
                            dst.write(block)

    def run(self):
        zip_path = None
        try:
            self.status.emit("Checking stable-diffusion.cpp releases for newest CUDA build…")
            tag, asset_name, url = self._find_newest_cuda_asset()

            self.target_dir.mkdir(parents=True, exist_ok=True)
            zip_path = self.target_dir / asset_name
            self.status.emit(f"Downloading {asset_name} ({tag})…")

            with self._request(url) as response, open(zip_path, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    self.progress.emit(done, total)

            self.status.emit("Extracting CUDA sd-cli into /presets/bin…")
            self.progress.emit(0, 0)
            self._safe_extract_overwrite(zip_path, self.target_dir)

            try:
                zip_path.unlink()
            except FileNotFoundError:
                pass

            self.completed.emit(tag, asset_name)
        except Exception as exc:
            if zip_path is not None:
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            self.failed.emit(str(exc))


class QwenImage21Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: Optional[QProcess] = None
        self.references: List[str] = []
        self.loras: List[LoraEntry] = []
        self.last_output: Optional[str] = None
        self.settings_path = DEFAULT_SETTINGS

        self._build_ui()
        self._load_settings()
        self._autodetect_cli()
        self._refresh_lora_list()
        self._update_ref_strip()
        self._set_running(False)
        self._on_use_queue_toggled(self.use_queue.isChecked())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("Qwen Image 2.1 • stable-diffusion.cpp")
        f = title.font()
        f.setPointSize(max(11, f.pointSize() + 2))
        f.setBold(True)
        title.setFont(f)
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.backend_status = QLabel("Backend: not checked")
        title_row.addWidget(self.backend_status)
        root.addLayout(title_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.generation_tab = self._make_generation_tab()
        self.models_tab = self._make_models_tab()
        self.lora_tab = self._make_lora_tab()
        self.log_tab = self._make_log_tab()
        self.settings_tab = self._make_settings_tab()

        self.tabs.addTab(self.generation_tab, "Generate")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.lora_tab, "LoRAs")
        self.tabs.addTab(self.log_tab, "Log")
        self.tabs.addTab(self.settings_tab, "Settings")

        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(180)
        footer.addWidget(self.progress)

        self.state_label = QLabel("Ready")
        footer.addWidget(self.state_label)
        footer.addStretch(1)

        self.use_queue = QCheckBox("Use queue")
        self.use_queue.setToolTip(
            "When enabled, Generate adds a Qwen Image 2.1 job to FrameVision's queue instead of starting sd-cli directly."
        )
        self.use_queue.toggled.connect(self._on_use_queue_toggled)
        footer.addWidget(self.use_queue)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        footer.addWidget(self.cancel_btn)

        self.open_output_btn = QPushButton("Open output folder")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        footer.addWidget(self.open_output_btn)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumWidth(150)
        self.generate_btn.clicked.connect(self.generate)
        footer.addWidget(self.generate_btn)

        root.addLayout(footer)

    def _scroll_page(self, content: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sc.setWidget(content)
        return sc

    def _make_generation_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)

        left_content = QWidget()
        left = QVBoxLayout(left_content)
        left.setContentsMargins(6, 6, 8, 6)

        grp = QGroupBox("Prompt")
        gl = QVBoxLayout(grp)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Describe the image or edit...")
        self.prompt.setMinimumHeight(120)
        gl.addWidget(self.prompt)

        self.negative = QTextEdit()
        self.negative.setPlaceholderText("Negative prompt (optional)")
        self.negative.setMaximumHeight(75)
        gl.addWidget(self.negative)

        self.alpha_helper = QCheckBox("Transparent / RGBA prompt helper")
        self.alpha_helper.setToolTip(
            "Adds Qwen Image 2.1's recommended alpha/transparency prompt wording."
        )
        gl.addWidget(self.alpha_helper)
        left.addWidget(grp)

        grp = QGroupBox("Reference images / image editing")
        v = QVBoxLayout(grp)
        r = QHBoxLayout()
        add_ref = QPushButton("Add reference images")
        add_ref.clicked.connect(self.add_references)
        clear_ref = QPushButton("Clear")
        clear_ref.clicked.connect(self.clear_references)
        r.addWidget(add_ref)
        r.addWidget(clear_ref)
        r.addStretch(1)
        v.addLayout(r)

        self.ref_scroll = QScrollArea()
        self.ref_scroll.setWidgetResizable(True)
        self.ref_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ref_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ref_strip = QWidget()
        self.ref_layout = QHBoxLayout(self.ref_strip)
        self.ref_layout.setContentsMargins(2, 2, 2, 2)
        self.ref_scroll.setWidget(self.ref_strip)
        self.ref_scroll.setMinimumHeight(235)
        v.addWidget(self.ref_scroll)
        left.addWidget(grp)

        grp = QGroupBox("Generation")
        form = QGridLayout(grp)

        self.size_preset = NoWheelComboBox()
        self.size_preset.addItems([
            "1024 × 1024", "1344 × 768", "768 × 1344",
            "1152 × 896", "896 × 1152", "1536 × 1024", "1024 × 1536",
            "Custom"
        ])
        self.size_preset.currentTextChanged.connect(self._apply_size_preset)

        self.width = NoWheelSpinBox()
        self.width.setRange(256, 4096)
        self.width.setSingleStep(32)
        self.width.setValue(1024)

        self.height = NoWheelSpinBox()
        self.height.setRange(256, 4096)
        self.height.setSingleStep(32)
        self.height.setValue(1024)

        self.steps = NoWheelSpinBox()
        self.steps.setRange(1, 200)
        self.steps.setValue(20)

        self.cfg = NoWheelDoubleSpinBox()
        self.cfg.setRange(0.0, 30.0)
        self.cfg.setDecimals(2)
        self.cfg.setSingleStep(0.25)
        self.cfg.setValue(6.0)

        self.seed = NoWheelSpinBox()
        self.seed.setRange(-1, 2147483647)
        self.seed.setValue(-1)
        self.seed.setToolTip("-1 = random")

        self.batch = NoWheelSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(1)

        self.sampler = NoWheelComboBox()
        self.sampler.addItems([
            "euler", "euler_a", "heun", "dpm2",
            "dpm++2s_a", "dpm++2m", "dpm++2mv2", "er_sde", "lcm"
        ])
        self.sampler.setCurrentText("euler")

        self.scheduler = NoWheelComboBox()
        self.scheduler.addItems(["discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple", "smoothstep", "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal", "flux2", "flux", "beta", "llada_image"])

        form.addWidget(QLabel("Size preset"), 0, 0)
        form.addWidget(self.size_preset, 0, 1, 1, 3)
        form.addWidget(QLabel("Width"), 1, 0)
        form.addWidget(self.width, 1, 1)
        form.addWidget(QLabel("Height"), 1, 2)
        form.addWidget(self.height, 1, 3)
        form.addWidget(QLabel("Steps"), 2, 0)
        form.addWidget(self.steps, 2, 1)
        form.addWidget(QLabel("CFG"), 2, 2)
        form.addWidget(self.cfg, 2, 3)
        form.addWidget(QLabel("Seed"), 3, 0)
        form.addWidget(self.seed, 3, 1)
        form.addWidget(QLabel("Batch"), 3, 2)
        form.addWidget(self.batch, 3, 3)
        form.addWidget(QLabel("Sampler"), 4, 0)
        form.addWidget(self.sampler, 4, 1)
        form.addWidget(QLabel("Scheduler"), 4, 2)
        form.addWidget(self.scheduler, 4, 3)
        left.addWidget(grp)

        grp = QGroupBox("Performance / VRAM")
        form = QGridLayout(grp)

        self.flash_attention = QCheckBox("Flash attention")
        self.flash_attention.setChecked(True)
        self.offload_cpu = QCheckBox("Offload to CPU")
        self.offload_cpu.setChecked(True)
        self.vae_tiling = QCheckBox("VAE tiling")
        self.mmap = QCheckBox("Memory map model files")
        self.mmap.setChecked(True)

        self.max_vram = NoWheelDoubleSpinBox()
        self.max_vram.setRange(0.0, 64.0)
        self.max_vram.setDecimals(1)
        self.max_vram.setSingleStep(0.5)
        self.max_vram.setValue(0.0)
        self.max_vram.setSpecialValueText("Automatic")

        self.rng = NoWheelComboBox()
        self.rng.addItems(["cuda", "cpu"])

        self.prefix_cache = NoWheelComboBox()
        self.prefix_cache.addItems(["auto", "f32", "f16", "bf16", "q8_0", "q6_K", "q4_0", "q4_K", "off"])

        form.addWidget(self.flash_attention, 0, 0, 1, 2)
        form.addWidget(self.offload_cpu, 0, 2, 1, 2)
        form.addWidget(self.vae_tiling, 1, 0, 1, 2)
        form.addWidget(self.mmap, 1, 2, 1, 2)
        form.addWidget(QLabel("Max VRAM (GiB)"), 2, 0)
        form.addWidget(self.max_vram, 2, 1)
        form.addWidget(QLabel("RNG"), 2, 2)
        form.addWidget(self.rng, 2, 3)
        form.addWidget(QLabel("Qwen prefix cache"), 3, 0)
        form.addWidget(self.prefix_cache, 3, 1, 1, 3)
        left.addWidget(grp)

        grp = QGroupBox("Advanced")
        form = QFormLayout(grp)
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Optional extra sd-cli arguments")
        form.addRow("Extra CLI arguments", self.extra_args)
        left.addWidget(grp)
        left.addStretch(1)

        splitter.addWidget(self._scroll_page(left_content))

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 6, 6, 6)

        out_grp = QGroupBox("Last output preview")
        ov = QVBoxLayout(out_grp)
        self.output_preview = QLabel("Generated image preview")
        self.output_preview.setAlignment(Qt.AlignCenter)
        self.output_preview.setMinimumSize(360, 360)
        self.output_preview.setStyleSheet("background:#121418; border:1px solid #303741;")
        ov.addWidget(self.output_preview, 1)
        self.last_output_label = ElidedPathLabel("")
        ov.addWidget(self.last_output_label)
        rv.addWidget(out_grp, 1)

        cmd_grp = QGroupBox("Command preview")
        cv = QVBoxLayout(cmd_grp)
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(150)
        cv.addWidget(self.command_preview)
        refresh = QPushButton("Refresh command preview")
        refresh.clicked.connect(self.refresh_command_preview)
        cv.addWidget(refresh)
        rv.addWidget(cmd_grp)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)
        return page

    def _make_models_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "Qwen Image 2.1 needs the diffusion model, its own Qwen Image 2.1 VAE, "
            "and Qwen3-VL-8B as text encoder. Reference-image editing with a GGUF "
            "text encoder also needs the matching mmproj / vision file."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Model files")
        form = QGridLayout(grp)
        self.diffusion_model = QLineEdit()
        self.vae_model = QLineEdit()
        self.llm_model = QLineEdit()
        self.llm_vision_model = QLineEdit()

        rows = [
            ("Diffusion GGUF", self.diffusion_model, "Model files (*.gguf *.safetensors);;All files (*)"),
            ("Qwen Image 2.1 VAE", self.vae_model, "Model files (*.safetensors *.gguf);;All files (*)"),
        ]

        for row, (label, edit, filt) in enumerate(rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(edit, row, 1)
            btn = QPushButton("Browse…")
            btn.clicked.connect(lambda _, e=edit, f=filt: self._browse_file(e, f))
            form.addWidget(btn, row, 2)

        form.addWidget(QLabel("Qwen3-VL-8B text encoder"), 2, 0)
        form.addWidget(self.llm_model, 2, 1)
        llm_buttons = QHBoxLayout()

        llm_file_btn = QPushButton("File…")
        llm_file_btn.clicked.connect(
            lambda: self._browse_file(
                self.llm_model,
                "Model files (*.gguf *.safetensors);;All files (*)"
            )
        )
        llm_buttons.addWidget(llm_file_btn)

        llm_folder_btn = QPushButton("HF folder…")
        llm_folder_btn.setToolTip(
            "Select the original Hugging Face Qwen3-VL folder containing "
            "model.safetensors.index.json and its shard files."
        )
        llm_folder_btn.clicked.connect(self._browse_llm_folder)
        llm_buttons.addWidget(llm_folder_btn)
        form.addLayout(llm_buttons, 2, 2)

        form.addWidget(QLabel("Qwen3-VL vision / mmproj"), 3, 0)
        form.addWidget(self.llm_vision_model, 3, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(
            lambda: self._browse_file(
                self.llm_vision_model,
                "Model files (*.gguf *.safetensors);;All files (*)"
            )
        )
        form.addWidget(btn, 3, 2)

        layout.addWidget(grp)

        grp = QGroupBox("Model notes")
        v = QVBoxLayout(grp)
        note = QLabel(
            "• Use Qwen Image 2.1 VAE weights; older Qwen Image / Wan VAEs are not interchangeable.\n"
            "• The linked GGUF repository contains transformer quantizations; text encoder and VAE are still required.\n"
            "• Original Hugging Face Qwen3-VL folders with sharded safetensors are accepted and auto-converted/cached.\n"
            "• Q4_K_M is a practical size/quality choice from the linked GGUF set."
        )
        note.setWordWrap(True)
        v.addWidget(note)
        layout.addWidget(grp)
        layout.addStretch(1)
        return self._scroll_page(content)

    def _make_lora_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("LoRA directory")
        row = QHBoxLayout(grp)
        self.lora_dir = QLineEdit(str(DEFAULT_MODELS_ROOT / "loras"))
        row.addWidget(self.lora_dir, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_dir(self.lora_dir))
        row.addWidget(b)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_lora_list)
        row.addWidget(refresh)
        layout.addWidget(grp)

        grp = QGroupBox("Available LoRAs")
        v = QVBoxLayout(grp)
        self.available_loras = QListWidget()
        self.available_loras.setSelectionMode(QAbstractItemView.SingleSelection)
        self.available_loras.itemDoubleClicked.connect(self._add_selected_available_lora)
        v.addWidget(self.available_loras)
        add = QPushButton("Add selected LoRA")
        add.clicked.connect(self._add_selected_available_lora)
        v.addWidget(add)
        layout.addWidget(grp)

        grp = QGroupBox("Active LoRAs")
        v = QVBoxLayout(grp)
        self.active_loras = QListWidget()
        v.addWidget(self.active_loras)
        row = QHBoxLayout()

        self.lora_strength = NoWheelDoubleSpinBox()
        self.lora_strength.setRange(-4.0, 4.0)
        self.lora_strength.setDecimals(2)
        self.lora_strength.setSingleStep(0.05)
        self.lora_strength.setValue(1.0)
        row.addWidget(QLabel("Strength"))
        row.addWidget(self.lora_strength)

        apply_strength = QPushButton("Apply strength")
        apply_strength.clicked.connect(self._change_lora_strength)
        row.addWidget(apply_strength)

        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_active_lora)
        row.addWidget(remove)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(grp)

        note = QLabel(
            "LoRAs are enabled using stable-diffusion.cpp prompt tags. "
            "The GUI appends active <lora:name:strength> tags only when launching sd-cli."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return self._scroll_page(content)

    def _make_log_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        v.addWidget(self.log)
        row = QHBoxLayout()
        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log.clear)
        row.addWidget(clear)
        row.addStretch(1)
        v.addLayout(row)
        return page

    def _make_settings_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("Application paths")
        form = QGridLayout(grp)
        self.cli_path = QLineEdit()
        self.output_dir = QLineEdit(str(DEFAULT_OUTPUT_DIR))

        form.addWidget(QLabel("sd-cli executable"), 0, 0)
        form.addWidget(self.cli_path, 0, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_file(self.cli_path, "Executables (*.exe);;All files (*)"))
        form.addWidget(b, 0, 2)

        detect = QPushButton("Auto-detect in /presets/bin")
        detect.clicked.connect(lambda: self._autodetect_cli(force=True))
        form.addWidget(detect, 1, 1, 1, 2)

        self.sd_cli_download_btn = QPushButton("Download / Update CUDA sd-cli")
        self.sd_cli_download_btn.setToolTip(
            "Find the newest stable-diffusion.cpp release that actually contains a Windows CUDA 12 x64 build, "
            "download it to /presets/bin, overwrite older files, extract it, and delete the ZIP."
        )
        self.sd_cli_download_btn.clicked.connect(self._download_or_update_sd_cli)
        form.addWidget(self.sd_cli_download_btn, 2, 1, 1, 2)

        self.sd_cli_download_progress = QProgressBar()
        self.sd_cli_download_progress.setRange(0, 100)
        self.sd_cli_download_progress.setValue(0)
        self.sd_cli_download_progress.setTextVisible(True)
        self.sd_cli_download_progress.hide()
        form.addWidget(self.sd_cli_download_progress, 3, 1, 1, 2)

        self.sd_cli_download_status = QLabel("")
        self.sd_cli_download_status.setWordWrap(True)
        self.sd_cli_download_status.hide()
        form.addWidget(self.sd_cli_download_status, 4, 1, 1, 2)

        form.addWidget(QLabel("Output folder"), 5, 0)
        form.addWidget(self.output_dir, 5, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_dir(self.output_dir))
        form.addWidget(b, 5, 2)
        layout.addWidget(grp)

        grp = QGroupBox("GUI behavior")
        form = QFormLayout(grp)

        self.auto_preview = QCheckBox("Preview generated image")
        self.auto_preview.setChecked(True)
        self.auto_open_output = QCheckBox("Open output folder after generation")
        self.remember_prompt = QCheckBox("Remember prompt between sessions")
        self.remember_prompt.setChecked(True)
        self.verbose_log = QCheckBox("Verbose backend logging")
        self.verbose_log.setChecked(True)

        form.addRow(self.auto_preview)
        form.addRow(self.auto_open_output)
        form.addRow(self.remember_prompt)
        form.addRow(self.verbose_log)
        layout.addWidget(grp)

        grp = QGroupBox("Persistence")
        row = QHBoxLayout(grp)
        save = QPushButton("Save settings now")
        save.clicked.connect(self._save_settings)
        row.addWidget(save)
        reset = QPushButton("Reset GUI settings")
        reset.clicked.connect(self._reset_settings)
        row.addWidget(reset)
        row.addStretch(1)
        layout.addWidget(grp)
        layout.addStretch(1)
        return self._scroll_page(content)

    def _browse_llm_folder(self):
        start = self.llm_model.text().strip()
        if start and Path(start).is_file():
            start = str(Path(start).parent)
        elif not start:
            start = str(DEFAULT_MODELS_ROOT)

        p = QFileDialog.getExistingDirectory(
            self,
            "Select original Qwen3-VL Hugging Face model folder",
            start
        )
        if p:
            self.llm_model.setText(p)
            self.state_label.setText("Qwen3-VL HF folder selected")

    def _resolved_llm_path(self, allow_convert=True) -> str:
        raw = self.llm_model.text().strip()
        if not raw:
            return ""

        p = Path(raw)
        if p.is_file():
            return str(p)

        if not p.is_dir():
            return raw

        index_path = p / "model.safetensors.index.json"
        single_path = p / "model.safetensors"
        if not index_path.exists() and not single_path.exists():
            raise FileNotFoundError(
                "The selected Qwen3-VL folder does not contain "
                "model.safetensors.index.json or model.safetensors."
            )

        signature_source = index_path if index_path.exists() else single_path
        stat = signature_source.stat()
        sig_text = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        digest = hashlib.sha1(sig_text.encode("utf-8")).hexdigest()[:10]

        cache_dir = DEFAULT_MODELS_ROOT / "text_encoders" / "sd_cpp_converted"
        out_path = cache_dir / f"qwen3vl_8b_sd_cpp_{digest}.safetensors"

        if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
            return str(out_path)

        if not allow_convert:
            return str(out_path)

        self.log.append(
            "\\n[GUI] Original Hugging Face Qwen3-VL folder detected. "
            "Converting shards to one stable-diffusion.cpp compatible file..."
        )
        QApplication.processEvents()

        def report(msg):
            self.state_label.setText(msg)
            self.log.append(f"[Qwen3-VL convert] {msg}")
            QApplication.processEvents()

        try:
            converted = convert_qwen3vl_hf_to_sd_cpp(p, out_path, report)
        except Exception:
            partial = out_path.with_suffix(out_path.suffix + ".part")
            if partial.exists():
                try:
                    partial.unlink()
                except Exception:
                    pass
            raise

        self.log.append(f"[GUI] Qwen3-VL conversion complete: {converted}")
        self.state_label.setText("Qwen3-VL conversion complete")
        return str(converted)

    def _browse_file(self, edit: QLineEdit, filt: str):
        start = edit.text().strip() or str(DEFAULT_MODELS_ROOT)
        p, _ = QFileDialog.getOpenFileName(self, "Select file", start, filt)
        if p:
            edit.setText(p)

    def _browse_dir(self, edit: QLineEdit):
        start = edit.text().strip() or str(ROOT)
        p = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if p:
            edit.setText(p)
            if edit is self.lora_dir:
                self._refresh_lora_list()

    def add_references(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add reference images", str(ROOT),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"
        )
        for p in files:
            if p not in self.references:
                self.references.append(p)
        self._update_ref_strip()

    def clear_references(self):
        self.references.clear()
        self._update_ref_strip()

    def _remove_reference(self, path: str):
        self.references = [p for p in self.references if p != path]
        self._update_ref_strip()

    def _update_ref_strip(self):
        while self.ref_layout.count():
            item = self.ref_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self.references:
            lbl = QLabel("No reference images loaded — text-to-image mode")
            lbl.setAlignment(Qt.AlignCenter)
            self.ref_layout.addWidget(lbl)
            return

        for p in self.references:
            tile = ReferenceTile(p)
            tile.remove_requested.connect(self._remove_reference)
            self.ref_layout.addWidget(tile)
        self.ref_layout.addStretch(1)

    def _refresh_lora_list(self):
        if not hasattr(self, "available_loras"):
            return
        self.available_loras.clear()
        folder_text = self.lora_dir.text().strip()
        if not folder_text:
            return
        folder = Path(folder_text)
        if not folder.exists():
            return

        exts = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}
        for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() in exts:
                item = QListWidgetItem(p.name)
                item.setData(Qt.UserRole, str(p))
                self.available_loras.addItem(item)

    def _add_selected_available_lora(self, *_):
        item = self.available_loras.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        if any(Path(x.path).name == Path(path).name for x in self.loras):
            return
        self.loras.append(LoraEntry(path=path, strength=float(self.lora_strength.value())))
        self._refresh_active_loras()

    def _refresh_active_loras(self):
        self.active_loras.clear()
        for entry in self.loras:
            item = QListWidgetItem(f"{Path(entry.path).name}    strength={entry.strength:g}")
            item.setData(Qt.UserRole, entry.path)
            self.active_loras.addItem(item)

    def _change_lora_strength(self):
        item = self.active_loras.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        for entry in self.loras:
            if entry.path == path:
                entry.strength = float(self.lora_strength.value())
                break
        self._refresh_active_loras()

    def _remove_active_lora(self):
        item = self.active_loras.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        self.loras = [x for x in self.loras if x.path != path]
        self._refresh_active_loras()

    def _lora_prompt_tags(self) -> str:
        return " ".join(
            f"<lora:{Path(entry.path).stem}:{entry.strength:g}>"
            for entry in self.loras
        )

    def _download_or_update_sd_cli(self):
        if getattr(self, "_sd_cli_download_worker", None) is not None and self._sd_cli_download_worker.isRunning():
            return

        PRESETS_BIN.mkdir(parents=True, exist_ok=True)
        self.sd_cli_download_btn.setEnabled(False)
        self.sd_cli_download_progress.show()
        self.sd_cli_download_status.show()
        self.sd_cli_download_progress.setRange(0, 0)
        self.sd_cli_download_status.setText("Checking releases for a Windows CUDA build…")

        worker = SdCliDownloadWorker(PRESETS_BIN, self)
        self._sd_cli_download_worker = worker
        worker.status.connect(self._on_sd_cli_download_status)
        worker.progress.connect(self._on_sd_cli_download_progress)
        worker.completed.connect(self._on_sd_cli_download_complete)
        worker.failed.connect(self._on_sd_cli_download_failed)
        worker.finished.connect(self._on_sd_cli_download_finished)
        worker.start()

    def _on_sd_cli_download_status(self, message: str):
        self.sd_cli_download_status.setText(message)

    def _on_sd_cli_download_progress(self, done: int, total: int):
        if total > 0:
            self.sd_cli_download_progress.setRange(0, 100)
            percent = max(0, min(100, int(done * 100 / total)))
            self.sd_cli_download_progress.setValue(percent)
            self.sd_cli_download_progress.setFormat(f"{percent}%")
        else:
            self.sd_cli_download_progress.setRange(0, 0)

    def _on_sd_cli_download_complete(self, tag: str, asset_name: str):
        self.sd_cli_download_progress.setRange(0, 100)
        self.sd_cli_download_progress.setValue(100)
        self.sd_cli_download_progress.setFormat("100%")
        self.sd_cli_download_status.setText(
            f"Installed newest available CUDA build: {tag}\n{asset_name}"
        )
        self._autodetect_cli(force=False)
        self._save_settings()
        self.state_label.setText("CUDA sd-cli installed / updated")

    def _on_sd_cli_download_failed(self, message: str):
        self.sd_cli_download_progress.setRange(0, 100)
        self.sd_cli_download_progress.setValue(0)
        self.sd_cli_download_progress.setFormat("Failed")
        self.sd_cli_download_status.setText(f"Download/update failed: {message}")
        QMessageBox.warning(
            self,
            "CUDA sd-cli download failed",
            f"Could not download or install stable-diffusion.cpp CUDA sd-cli:\n\n{message}",
        )

    def _on_sd_cli_download_finished(self):
        self.sd_cli_download_btn.setEnabled(True)

    def _autodetect_cli(self, force=False):
        if not force and self.cli_path.text().strip() and Path(self.cli_path.text().strip()).exists():
            self._update_backend_status()
            return

        candidates = [
            PRESETS_BIN / "sd-cli.exe",
            PRESETS_BIN / "sd-cli",
            PRESETS_BIN / "sd.exe",
            PRESETS_BIN / "sd",
        ]
        for p in candidates:
            if p.exists():
                self.cli_path.setText(str(p))
                self._update_backend_status()
                return

        if PRESETS_BIN.exists():
            names = {"sd-cli.exe", "sd-cli", "sd.exe", "sd"}
            for p in PRESETS_BIN.rglob("*"):
                if p.is_file() and p.name.lower() in names:
                    self.cli_path.setText(str(p))
                    self._update_backend_status()
                    return

        self.backend_status.setText("Backend: sd-cli not found")
        if force:
            QMessageBox.warning(
                self, "sd-cli not found",
                f"No stable-diffusion.cpp CLI executable was found under:\n{PRESETS_BIN}"
            )

    def _update_backend_status(self):
        p = Path(self.cli_path.text().strip()) if self.cli_path.text().strip() else None
        if p and p.exists():
            self.backend_status.setText(f"Backend: {p.name}")
            self.backend_status.setToolTip(str(p))
        else:
            self.backend_status.setText("Backend: missing")

    def _validate(self) -> Optional[str]:
        cli_text = self.cli_path.text().strip()
        if not cli_text or not Path(cli_text).exists():
            return "stable-diffusion.cpp sd-cli executable was not found."

        required = [
            ("Diffusion model", self.diffusion_model.text().strip()),
            ("Qwen Image 2.1 VAE", self.vae_model.text().strip()),
            ("Qwen3-VL text encoder", self.llm_model.text().strip()),
        ]
        for label, value in required:
            if not value:
                return f"{label} is not selected."
            if not Path(value).exists():
                return f"{label} does not exist:\n{value}"

        try:
            self._resolved_llm_path(allow_convert=True)
        except Exception as e:
            return f"Could not prepare Qwen3-VL text encoder:\n{e}"

        if self.references:
            vision = self.llm_vision_model.text().strip()
            if not vision or not Path(vision).exists():
                return "Reference-image editing is active, but the Qwen3-VL vision/mmproj file is missing."
            for p in self.references:
                if not Path(p).exists():
                    return f"Reference image does not exist:\n{p}"

        if self.width.value() % 32 != 0 or self.height.value() % 32 != 0:
            return "Qwen Image 2.1 width and height must both be divisible by 32."

        if not self.prompt.toPlainText().strip():
            return "Prompt is empty."

        try:
            Path(self.output_dir.text().strip()).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"Cannot create output folder:\n{e}"

        return None

    @staticmethod
    def _split_extra_args(text: str) -> List[str]:
        if not text.strip():
            return []
        pattern = r"(?:[^\s\"']+|\"[^\"]*\"|'[^']*')+"
        parts = re.findall(pattern, text)
        cleaned = []
        for p in parts:
            if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                p = p[1:-1]
            cleaned.append(p)
        return cleaned

    def _effective_prompt(self) -> str:
        p = self.prompt.toPlainText().strip()
        if self.alpha_helper.isChecked():
            p = (
                "This is an RGBA image with transparency. "
                + p
                + ". The image has alpha channel and the background is transparent."
            )
        tags = self._lora_prompt_tags()
        if tags:
            p = f"{p} {tags}"
        return p

    def _next_output_path(self) -> str:
        out_dir = Path(self.output_dir.text().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return str(out_dir / f"qwen21_{stamp}.png")

    def build_command(self, output_path: Optional[str] = None):
        cli = self.cli_path.text().strip()
        if output_path is None:
            output_path = self._next_output_path()

        args = [
            "--diffusion-model", self.diffusion_model.text().strip(),
            "--vae", self.vae_model.text().strip(),
            "--llm", self._resolved_llm_path(allow_convert=True),
            "-p", self._effective_prompt(),
            "--cfg-scale", f"{self.cfg.value():g}",
            "--sampling-method", self.sampler.currentText(),
            "--scheduler", self.scheduler.currentText(),
            "--steps", str(self.steps.value()),
            "-W", str(self.width.value()),
            "-H", str(self.height.value()),
            "-s", str(self.seed.value()),
            "-b", str(self.batch.value()),
            "--rng", self.rng.currentText(),
            "-o", output_path,
        ]

        neg = self.negative.toPlainText().strip()
        if neg:
            args += ["-n", neg]

        if self.references:
            args += ["--llm_vision", self.llm_vision_model.text().strip()]
            for p in self.references:
                args += ["-r", p]

        if self.loras and self.lora_dir.text().strip():
            args += ["--lora-model-dir", self.lora_dir.text().strip()]

        if self.flash_attention.isChecked():
            args.append("--fa")
        if self.offload_cpu.isChecked():
            args.append("--offload-to-cpu")
        if self.vae_tiling.isChecked():
            args.append("--vae-tiling")
        if self.mmap.isChecked():
            args.append("--mmap")
        if self.max_vram.value() > 0.0:
            args += ["--max-vram", f"{self.max_vram.value():g}"]

        cache = self.prefix_cache.currentText()
        if cache == "off":
            args += ["--model-args", "qwen_image_2_1_prefix_cache=false"]
        elif cache != "auto":
            args += ["--model-args", f"qwen_image_2_1_prefix_cache_type={cache}"]

        if self.verbose_log.isChecked():
            args.append("-v")

        args += self._split_extra_args(self.extra_args.text())
        return cli, args, output_path

    def refresh_command_preview(self):
        cli, args, _ = self.build_command(output_path="<output>.png")
        self.command_preview.setPlainText(self._format_command(cli, args))

    def generate(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return

        error = self._validate()
        if error:
            QMessageBox.warning(self, "Cannot generate", error)
            return

        self._save_settings()

        if self.use_queue.isChecked():
            self._enqueue_current_job()
            return

        cli, args, output_path = self.build_command()
        self.last_output = output_path
        self.last_output_label.setPath(output_path)

        self.log.append("\n=== Generation started ===")
        self.log.append(self._format_command(cli, args))

        self.process = QProcess(self)
        self.process.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        self.process.setProgram(cli)
        self.process.setArguments(args)
        self.process.setWorkingDirectory(str(Path(cli).parent))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self._set_running(True)
        self.process.start()

    def _on_use_queue_toggled(self, checked: bool):
        checked = bool(checked)
        self.generate_btn.setText("Add to queue" if checked else "Generate")
        if checked:
            self.generate_btn.setToolTip(
                "Adds this Qwen Image 2.1 job to FrameVision's pending queue. Requires queue_adapter support."
            )
        else:
            self.generate_btn.setToolTip("Run Qwen Image 2.1 directly with stable-diffusion.cpp.")
        # Direct-run cancel does not apply to a job that has only been queued.
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.cancel_btn.setEnabled(False)

    def _enqueue_current_job(self):
        """Queue through FrameVision when its adapter exposes Qwen Image 2.1 support."""
        try:
            try:
                from helpers import queue_adapter as _qa  # type: ignore
            except Exception:
                import queue_adapter as _qa  # type: ignore

            enq = getattr(_qa, "enqueue_qwen_image_2_1_from_widget", None)
            if not callable(enq):
                raise RuntimeError(
                    "queue_adapter.py does not yet provide enqueue_qwen_image_2_1_from_widget(widget)."
                )

            job_id = enq(self)
            self.state_label.setText("Queued")
            self.log.append(f"\n=== Added to queue: {job_id} ===")
            QMessageBox.information(
                self,
                "Qwen Image 2.1 queued",
                f"Job added to FrameVision's queue.\n\nJob: {job_id}",
            )
        except Exception as exc:
            self.state_label.setText("Queue unavailable")
            self.log.append("\n[Queue] " + str(exc))
            QMessageBox.warning(
                self,
                "Qwen Image 2.1 queue support",
                "The Qwen Image 2.1 tab is ready for queue integration, but the current "
                "queue adapter/worker does not expose Qwen Image 2.1 yet.\n\n"
                f"Reason: {exc}",
            )

    def cancel_generation(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.log.append("\n[GUI] Cancel requested.")
            self.process.terminate()
            if not self.process.waitForFinished(2500):
                self.process.kill()

    def _read_process_output(self):
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.insertPlainText(data)
            self.log.ensureCursorVisible()

    def _process_finished(self, exit_code, exit_status):
        self._set_running(False)
        self.log.append(f"\n=== Finished: exit code {exit_code} ===")

        candidates = self._find_recent_outputs()
        if candidates:
            self.last_output = str(candidates[0])
            self.last_output_label.setPath(self.last_output)
            if self.auto_preview.isChecked():
                self._show_output_preview(self.last_output)

        if exit_code == 0:
            self.state_label.setText("Finished")
            if self.auto_open_output.isChecked():
                self.open_output_folder()
        else:
            self.state_label.setText(f"Failed ({exit_code})")
            QMessageBox.warning(
                self, "Generation failed",
                "sd-cli returned a non-zero exit code. Open the Log tab for details."
            )

    def _process_error(self, err):
        self._set_running(False)
        self.log.append(f"\n[QProcess error] {err}")
        self.state_label.setText("Backend error")

    def _set_running(self, running: bool):
        self.generate_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.progress.setRange(0, 0)
            self.state_label.setText("Generating…")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _find_recent_outputs(self) -> List[Path]:
        folder_text = self.output_dir.text().strip()
        if not folder_text:
            return []
        folder = Path(folder_text)
        if not folder.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:20]

    def _show_output_preview(self, path: str):
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            self.output_preview.setText("Preview unavailable")
            return
        pix = QPixmap.fromImage(img)
        self.output_preview.setPixmap(
            pix.scaled(self.output_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def open_output_folder(self):
        folder = Path(self.output_dir.text().strip() or DEFAULT_OUTPUT_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def _format_command(cli: str, args: List[str]) -> str:
        def q(x):
            x = str(x)
            return f'"{x}"' if any(c.isspace() for c in x) else x
        return " ".join([q(cli)] + [q(a) for a in args])

    def _apply_size_preset(self, text: str):
        if text == "Custom":
            return
        m = re.match(r"(\d+)\s*×\s*(\d+)", text)
        if m:
            self.width.setValue(int(m.group(1)))
            self.height.setValue(int(m.group(2)))

    def _settings_dict(self):
        return {
            "cli_path": self.cli_path.text(),
            "output_dir": self.output_dir.text(),
            "diffusion_model": self.diffusion_model.text(),
            "vae_model": self.vae_model.text(),
            "llm_model": self.llm_model.text(),
            "llm_vision_model": self.llm_vision_model.text(),
            "lora_dir": self.lora_dir.text(),
            "loras": [asdict(x) for x in self.loras],
            "width": self.width.value(),
            "height": self.height.value(),
            "steps": self.steps.value(),
            "cfg": self.cfg.value(),
            "seed": self.seed.value(),
            "batch": self.batch.value(),
            "sampler": self.sampler.currentText(),
            "scheduler": self.scheduler.currentText(),
            "flash_attention": self.flash_attention.isChecked(),
            "offload_cpu": self.offload_cpu.isChecked(),
            "vae_tiling": self.vae_tiling.isChecked(),
            "mmap": self.mmap.isChecked(),
            "max_vram": self.max_vram.value(),
            "rng": self.rng.currentText(),
            "prefix_cache": self.prefix_cache.currentText(),
            "extra_args": self.extra_args.text(),
            "auto_preview": self.auto_preview.isChecked(),
            "auto_open_output": self.auto_open_output.isChecked(),
            "remember_prompt": self.remember_prompt.isChecked(),
            "verbose_log": self.verbose_log.isChecked(),
            "use_queue": self.use_queue.isChecked(),
            "prompt": self.prompt.toPlainText() if self.remember_prompt.isChecked() else "",
            "negative": self.negative.toPlainText() if self.remember_prompt.isChecked() else "",
        }

    def _save_settings(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(
                json.dumps(self._settings_dict(), indent=2), encoding="utf-8"
            )
            self.state_label.setText("Settings saved")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Could not save settings:\n{e}")

    def _load_settings(self):
        if not self.settings_path.exists():
            return
        try:
            d = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return

        def set_text(widget, key):
            if key in d and d[key] is not None:
                widget.setText(str(d[key]))

        set_text(self.cli_path, "cli_path")
        set_text(self.output_dir, "output_dir")
        set_text(self.diffusion_model, "diffusion_model")
        set_text(self.vae_model, "vae_model")
        set_text(self.llm_model, "llm_model")
        set_text(self.llm_vision_model, "llm_vision_model")
        set_text(self.lora_dir, "lora_dir")
        set_text(self.extra_args, "extra_args")

        for widget, key in [
            (self.width, "width"), (self.height, "height"), (self.steps, "steps"),
            (self.seed, "seed"), (self.batch, "batch")
        ]:
            if key in d:
                widget.setValue(int(d[key]))

        for widget, key in [(self.cfg, "cfg"), (self.max_vram, "max_vram")]:
            if key in d:
                widget.setValue(float(d[key]))

        for widget, key in [
            (self.sampler, "sampler"), (self.scheduler, "scheduler"),
            (self.rng, "rng"), (self.prefix_cache, "prefix_cache")
        ]:
            if key in d and widget.findText(str(d[key])) >= 0:
                widget.setCurrentText(str(d[key]))

        for widget, key in [
            (self.flash_attention, "flash_attention"),
            (self.offload_cpu, "offload_cpu"),
            (self.vae_tiling, "vae_tiling"),
            (self.mmap, "mmap"),
            (self.auto_preview, "auto_preview"),
            (self.auto_open_output, "auto_open_output"),
            (self.remember_prompt, "remember_prompt"),
            (self.verbose_log, "verbose_log"),
            (self.use_queue, "use_queue"),
        ]:
            if key in d:
                widget.setChecked(bool(d[key]))

        if d.get("remember_prompt", True):
            self.prompt.setPlainText(d.get("prompt", ""))
            self.negative.setPlainText(d.get("negative", ""))

        self.loras = []
        for x in d.get("loras", []):
            try:
                self.loras.append(
                    LoraEntry(path=str(x["path"]), strength=float(x.get("strength", 1.0)))
                )
            except Exception:
                pass
        self._refresh_active_loras()

    def _reset_settings(self):
        reply = QMessageBox.question(
            self, "Reset GUI settings",
            "Delete the saved Qwen Image 2.1 GUI settings file?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            if self.settings_path.exists():
                self.settings_path.unlink()
            QMessageBox.information(
                self, "Reset",
                "Saved settings were removed. Restart this panel to reload defaults."
            )
        except Exception as e:
            QMessageBox.warning(self, "Reset", str(e))

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


class QwenImage21Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FrameVision • Qwen Image 2.1")
        self.resize(1320, 860)
        self.setCentralWidget(QwenImage21Widget(self))


def create_qwen_image_2_1_widget(parent=None) -> QwenImage21Widget:
    return QwenImage21Widget(parent)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    win = QwenImage21Window()
    win.show()
    sys.exit(app.exec())
