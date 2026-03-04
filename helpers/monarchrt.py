"""FrameVision MonarchRT helper pane (PySide6).

This module is designed to live at: <FrameVisionRoot>/helpers/monarchrt.py

It provides a QWidget-based UI that:
  - Collects a single prompt (plus optional extended prompt)
  - Lets the user pick a MonarchRT config YAML and checkpoint
  - Runs MonarchRT's inference.py via the dedicated environment Python
  - Renames the produced MP4(s) into: MonRT_<seed>_<date>.mp4 in /output/video/

Paths assumed (relative to FrameVision root):
  - Python:   environments/.monarchrt/Scripts/python.exe
  - Repo:     models/Wan2.1-T2V-1.3B/repo/MonarchRT
  - Model dir models/Wan2.1-T2V-1.3B/
  - Output:   output/video/

Notes about "defaults": MonarchRT exposes many generation settings via config YAMLs.
This UI reads defaults from the selected config (merged over configs/default_config.yaml)
and allows overriding common ones (seed / num_samples / guidance / negative prompt).

MonarchRT's public inference.py currently hardcodes output FPS to 16.
We show an FPS field for visibility, but it cannot change FPS unless MonarchRT changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


def _fv_root() -> Path:
    # helpers/monarchrt.py -> FrameVision root is parent of helpers
    return Path(__file__).resolve().parents[1]


def _safe_int(text: str, default: int) -> int:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _now_stamp() -> str:
    # Date part in filename should be stable and sortable.
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_simple_yaml(path: Path) -> Dict:
    """Lightweight YAML reader.

    We avoid importing PyYAML to keep this helper self-contained.
    The MonarchRT configs are simple key: value + small lists, so we parse
    only what's needed for UI defaults.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # Ultra-minimal YAML-ish parser for this repo's config style.
    # Supports:
    #   key: value
    #   key: 'string'
    #   key: "string"
    #   key:
    #     - item
    #     - item
    data: Dict[str, object] = {}
    current_list_key: Optional[str] = None
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # list item
        if current_list_key and line.startswith("-"):
            item = line[1:].strip()
            # try int/float
            val: object = item
            if re.fullmatch(r"-?\d+", item):
                val = int(item)
            else:
                try:
                    val = float(item)
                except Exception:
                    val = item.strip("\"'")
            data.setdefault(current_list_key, []).append(val)
            continue

        current_list_key = None
        if ":" not in line:
            continue

        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # handle block-list start
        if val == "":
            current_list_key = key
            data[key] = []
            continue

        # strip inline comments safely
        if " #" in val:
            val = val.split(" #", 1)[0].rstrip()

        # unquote
        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            val = val[1:-1]

        # bool
        if val.lower() in {"true", "false"}:
            data[key] = (val.lower() == "true")
            continue

        # int
        if re.fullmatch(r"-?\d+", val):
            data[key] = int(val)
            continue

        # float
        try:
            f = float(val)
            data[key] = f
            continue
        except Exception:
            pass

        data[key] = val

    return data


def _merge_dicts(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class MonarchRTPaths:
    root: Path
    python_exe: Path
    repo_dir: Path
    model_dir: Path
    out_dir: Path


def resolve_paths() -> MonarchRTPaths:
    root = _fv_root()
    python_exe = root / "environments" / ".monarchrt" / "Scripts" / "python.exe"
    model_dir = root / "models" / "Wan2.1-T2V-1.3B"
    repo_dir = model_dir / "repo" / "MonarchRT"
    out_dir = root / "output" / "video"
    return MonarchRTPaths(root=root, python_exe=python_exe, repo_dir=repo_dir, model_dir=model_dir, out_dir=out_dir)


class _ProcWorker(QtCore.QObject):
    """Runs a subprocess and streams its output."""

    sig_line = QtCore.Signal(str)
    sig_done = QtCore.Signal(int, str)  # returncode, last_output_path (may be empty)

    def __init__(self, cmd: list[str], cwd: Path, env: Dict[str, str], expected_out_dir: Path):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd
        self._env = env
        self._expected_out_dir = expected_out_dir
        self._proc: Optional[subprocess.Popen] = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    @QtCore.Slot()
    def run(self) -> None:
        last_seen = self._snapshot_outputs()

        try:
            self._proc = subprocess.Popen(
                self._cmd,
                cwd=str(self._cwd),
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=(0x08000000 if os.name == "nt" else 0),  # CREATE_NO_WINDOW
            )
        except Exception as e:
            self.sig_line.emit(f"[MonarchRT] Failed to start: {e}\n")
            self.sig_done.emit(1, "")
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop:
                break
            self.sig_line.emit(line.rstrip("\n"))

        # If we requested stop, ensure process is dead.
        if self._stop and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

        rc = self._proc.wait()
        newest = self._diff_newest(last_seen)
        self.sig_done.emit(rc, str(newest) if newest else "")

    def _snapshot_outputs(self) -> Dict[Path, float]:
        out: Dict[Path, float] = {}
        try:
            for p in self._expected_out_dir.glob("*.mp4"):
                try:
                    out[p] = p.stat().st_mtime
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def _diff_newest(self, before: Dict[Path, float]) -> Optional[Path]:
        candidates: list[Tuple[float, Path]] = []
        try:
            for p in self._expected_out_dir.glob("*.mp4"):
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    continue
                if p not in before or mt > before.get(p, 0):
                    candidates.append((mt, p))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]


class MonarchRTPane(QtWidgets.QWidget):
    """Standalone pane; can be embedded in FrameVision."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.paths = resolve_paths()
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_ProcWorker] = None
        self._last_generated: list[Path] = []

        self._build_ui()
        self._refresh_configs()
        self._apply_defaults_from_selected_config()
        self._validate_paths(update_ui=True)

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.setObjectName("MonarchRTPane")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        scroll_body = QtWidgets.QWidget()
        scroll.setWidget(scroll_body)

        root = QtWidgets.QVBoxLayout(scroll_body)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)


        title = QtWidgets.QLabel("MonarchRT (Wan2.1-T2V-1.3B)")
        title.setFont(QtGui.QFont(title.font().family(), 12, QtGui.QFont.Weight.DemiBold))
        root.addWidget(title)

        # --- Prompt box
        prompt_box = QtWidgets.QGroupBox("Prompt")
        prompt_layout = QtWidgets.QVBoxLayout(prompt_box)
        prompt_layout.setContentsMargins(10, 10, 10, 10)

        self.txt_prompt = QtWidgets.QPlainTextEdit()
        self.txt_prompt.setPlaceholderText("Describe the video you want to generate...")
        self.txt_prompt.setMinimumHeight(72)
        prompt_layout.addWidget(self.txt_prompt)

        self.chk_extended = QtWidgets.QCheckBox("Use extended prompt (optional)")
        prompt_layout.addWidget(self.chk_extended)

        self.txt_extended = QtWidgets.QPlainTextEdit()
        self.txt_extended.setPlaceholderText("Extended prompt (if enabled)")
        self.txt_extended.setMinimumHeight(56)
        self.txt_extended.setEnabled(False)
        prompt_layout.addWidget(self.txt_extended)

        self.chk_extended.toggled.connect(self.txt_extended.setEnabled)
        root.addWidget(prompt_box)

        # --- Settings grid
        settings_box = QtWidgets.QGroupBox("Settings")
        grid = QtWidgets.QGridLayout(settings_box)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        row = 0

        self.cmb_config = QtWidgets.QComboBox()
        self.cmb_config.setMinimumWidth(280)
        self.cmb_config.currentIndexChanged.connect(self._apply_defaults_from_selected_config)
        grid.addWidget(QtWidgets.QLabel("Config"), row, 0)
        grid.addWidget(self.cmb_config, row, 1, 1, 3)
        row += 1

        self.ed_checkpoint = QtWidgets.QLineEdit()
        self.btn_browse_ckpt = QtWidgets.QPushButton("Browse")
        self.btn_browse_ckpt.clicked.connect(self._browse_checkpoint)
        grid.addWidget(QtWidgets.QLabel("Checkpoint"), row, 0)
        grid.addWidget(self.ed_checkpoint, row, 1, 1, 2)
        grid.addWidget(self.btn_browse_ckpt, row, 3)
        row += 1

        self.ed_seed = QtWidgets.QLineEdit("0")
        self.ed_seed.setValidator(QtGui.QIntValidator(-1, 2**31 - 1, self))
        self.btn_random_seed = QtWidgets.QPushButton("Random")
        self.btn_random_seed.clicked.connect(self._randomize_seed)
        grid.addWidget(QtWidgets.QLabel("Seed"), row, 0)
        grid.addWidget(self.ed_seed, row, 1)
        grid.addWidget(self.btn_random_seed, row, 2)
        row += 1

        self.spn_samples = QtWidgets.QSpinBox()
        self.spn_samples.setRange(1, 16)
        self.spn_samples.setValue(1)
        grid.addWidget(QtWidgets.QLabel("Samples"), row, 0)
        grid.addWidget(self.spn_samples, row, 1)
        row += 1

        self.ed_guidance = QtWidgets.QDoubleSpinBox()
        self.ed_guidance.setRange(0.0, 20.0)
        self.ed_guidance.setSingleStep(0.1)
        self.ed_guidance.setValue(3.0)
        grid.addWidget(QtWidgets.QLabel("Guidance"), row, 0)
        grid.addWidget(self.ed_guidance, row, 1)
        row += 1

        self.txt_negative = QtWidgets.QPlainTextEdit()
        self.txt_negative.setPlaceholderText("Negative prompt (optional; pulled from config when available)")
        self.txt_negative.setMinimumHeight(54)
        grid.addWidget(QtWidgets.QLabel("Negative"), row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.txt_negative, row, 1, 2, 3)
        row += 2

        self.lbl_res = QtWidgets.QLabel("—")
        self.lbl_frames = QtWidgets.QLabel("—")
        self.lbl_steps = QtWidgets.QLabel("—")
        self.lbl_fps = QtWidgets.QLabel("16 (fixed in MonarchRT inference.py)")

        grid.addWidget(QtWidgets.QLabel("Resolution"), row, 0)
        grid.addWidget(self.lbl_res, row, 1)
        grid.addWidget(QtWidgets.QLabel("Frames"), row, 2)
        grid.addWidget(self.lbl_frames, row, 3)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Steps"), row, 0)
        grid.addWidget(self.lbl_steps, row, 1)
        grid.addWidget(QtWidgets.QLabel("FPS"), row, 2)
        grid.addWidget(self.lbl_fps, row, 3)
        row += 1

        self.chk_use_ema = QtWidgets.QCheckBox("Use EMA")
        self.chk_torch_compile = QtWidgets.QCheckBox("torch.compile")
        self.chk_disable_offload = QtWidgets.QCheckBox("Disable offload")
        grid.addWidget(self.chk_use_ema, row, 0)
        grid.addWidget(self.chk_torch_compile, row, 1)
        grid.addWidget(self.chk_disable_offload, row, 2)
        row += 1

        self.spn_overlap = QtWidgets.QSpinBox()
        self.spn_overlap.setRange(1, 200)
        self.spn_overlap.setValue(21)
        self.spn_overlap.setToolTip("Maps to --num_output_frames in MonarchRT inference.py. Default 21.")
        grid.addWidget(QtWidgets.QLabel("Overlap frames"), row, 0)
        grid.addWidget(self.spn_overlap, row, 1)
        row += 1

        root.addWidget(settings_box)

        # --- Run controls
        controls = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_open_output = QtWidgets.QPushButton("Open output folder")
        self.btn_open_output.clicked.connect(self._open_output_folder)

        self.btn_run.clicked.connect(self._on_generate)
        self.btn_cancel.clicked.connect(self._on_cancel)

        controls.addWidget(self.btn_run)
        controls.addWidget(self.btn_cancel)
        controls.addStretch(1)
        controls.addWidget(self.btn_open_output)
        root.addLayout(controls)

        # --- Status + log
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(160)
        root.addWidget(self.txt_log, 1)

    # ------------- Defaults / config -------------

    def _configs_dir(self) -> Path:
        return self.paths.repo_dir / "configs"

    def _refresh_configs(self) -> None:
        self.cmb_config.blockSignals(True)
        self.cmb_config.clear()

        cfg_dir = self._configs_dir()
        if cfg_dir.exists():
            configs = sorted(cfg_dir.glob("*.yaml"), key=lambda p: p.name.lower())
            for p in configs:
                self.cmb_config.addItem(p.name, str(p))

        # pick a sensible default
        preferred = [
            "self_forcing_dmd.yaml",
            "self_forcing_monarch_dmd.yaml",
            "self_forcing_monarch_dmd_optimized.yaml",
        ]
        for name in preferred:
            idx = self.cmb_config.findText(name)
            if idx >= 0:
                self.cmb_config.setCurrentIndex(idx)
                break

        self.cmb_config.blockSignals(False)

        # checkpoint default
        guess = self.paths.repo_dir / "checkpoints" / "self_forcing_dmd.pt"
        if guess.exists() and not self.ed_checkpoint.text().strip():
            self.ed_checkpoint.setText(str(guess))

    def _selected_config_path(self) -> Optional[Path]:
        p = self.cmb_config.currentData()
        return Path(p) if p else None

    def _load_effective_config(self) -> Dict:
        """default_config.yaml merged with selected config yaml."""
        cfg_dir = self._configs_dir()
        base_path = cfg_dir / "default_config.yaml"
        base = _read_simple_yaml(base_path) if base_path.exists() else {}

        selected = self._selected_config_path()
        over = _read_simple_yaml(selected) if selected and selected.exists() else {}
        return _merge_dicts(base, over)

    @QtCore.Slot()
    def _apply_defaults_from_selected_config(self) -> None:
        cfg = self._load_effective_config()

        # resolution / frames
        w = int(cfg.get("width", 832) or 832)
        h = int(cfg.get("height", 480) or 480)
        nf = int(cfg.get("num_frames", 81) or 81)
        self.lbl_res.setText(f"{w}×{h}")
        self.lbl_frames.setText(str(nf))

        # steps: denoising_step_list exists for few-step configs
        dsl = cfg.get("denoising_step_list")
        if isinstance(dsl, list) and dsl:
            self.lbl_steps.setText(f"{len(dsl)} (denoising_step_list)")
        else:
            # multi-step diffusion uses num_train_timestep or similar
            nts = cfg.get("num_train_timestep")
            self.lbl_steps.setText(str(nts) if nts is not None else "—")

        # guidance
        gs = cfg.get("guidance_scale")
        if isinstance(gs, (int, float)):
            self.ed_guidance.setValue(float(gs))

        # negative prompt if present
        neg = cfg.get("negative_prompt")
        if isinstance(neg, str) and neg.strip() and not self.txt_negative.toPlainText().strip():
            self.txt_negative.setPlainText(neg)

        # overlap frames (training frames) default from config if present
        ntf = cfg.get("num_training_frames")
        if isinstance(ntf, int) and ntf > 0:
            self.spn_overlap.setValue(int(ntf))

        self._validate_paths(update_ui=True)

    # ------------- Path validation -------------

    def _validate_paths(self, update_ui: bool = False) -> bool:
        missing = []
        if not self.paths.python_exe.exists():
            missing.append(f"Missing python: {self.paths.python_exe}")
        if not self.paths.repo_dir.exists():
            missing.append(f"Missing repo: {self.paths.repo_dir}")

        ckpt = Path(self.ed_checkpoint.text().strip()) if self.ed_checkpoint.text().strip() else None
        if ckpt and not ckpt.exists():
            missing.append(f"Checkpoint not found: {ckpt}")

        ok = not missing
        if update_ui:
            if ok:
                self.lbl_status.setText(
                    f"Ready. Output → {self.paths.out_dir}"
                )
                self.lbl_status.setStyleSheet("")
            else:
                self.lbl_status.setText("\n".join(missing))
                self.lbl_status.setStyleSheet("color: #c0392b;")
        return ok

    # ------------- Actions -------------

    def _browse_checkpoint(self) -> None:
        start_dir = str(self.paths.model_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select checkpoint (.pt / .safetensors)",
            start_dir,
            "Checkpoint (*.pt *.pth *.bin *.safetensors);;All files (*.*)",
        )
        if path:
            self.ed_checkpoint.setText(path)
            self._validate_paths(update_ui=True)

    def _randomize_seed(self) -> None:
        # -1 is commonly used as "random" in other tools; MonarchRT expects an int,
        # but its set_seed uses this value. We'll map -1 to a time-based seed.
        self.ed_seed.setText(str(int(time.time() * 1000) % (2**31 - 1)))

    def _open_output_folder(self) -> None:
        out = self.paths.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(out))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(out)])
        else:
            subprocess.Popen(["xdg-open", str(out)])

    def _append_log(self, line: str) -> None:
        self.txt_log.appendPlainText(line)
        # keep cursor at bottom
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.txt_log.setTextCursor(cursor)

    def _make_prompt_files(self, prompt: str, extended: Optional[str]) -> Tuple[Path, Optional[Path]]:
        tmp_dir = self.paths.root / "output" / "_tmp" / "monarchrt"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_stamp()
        prompt_path = tmp_dir / f"prompt_{stamp}.txt"
        prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")

        ext_path: Optional[Path] = None
        if extended and extended.strip():
            ext_path = tmp_dir / f"extended_{stamp}.txt"
            ext_path.write_text(extended.strip() + "\n", encoding="utf-8")
        return prompt_path, ext_path

    def _make_override_config(self, base_cfg_path: Path, guidance: float, negative: str) -> Path:
        # We create a small overlay YAML that includes the base config via copy
        # and overrides a few keys. Since MonarchRT uses OmegaConf, we can just
        # write a merged YAML as a flat file.
        base = _read_simple_yaml(self._configs_dir() / "default_config.yaml")
        over = _read_simple_yaml(base_cfg_path)
        merged = _merge_dicts(base, over)
        merged["guidance_scale"] = float(guidance)
        if negative.strip():
            merged["negative_prompt"] = negative.strip()

        # Keep height/width/num_frames consistent with the merged config.
        # Changing these may break inference.py tensor shapes.

        tmp_dir = self.paths.root / "output" / "_tmp" / "monarchrt"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"cfg_{_now_stamp()}.yaml"

        # Serialize in a simple YAML style
        lines = []
        for k, v in merged.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for it in v:
                    lines.append(f"  - {it}")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k}: {v}")
            else:
                s = str(v)
                # quote strings with special chars
                if any(ch in s for ch in [":", "#", "\n", "\t"]):
                    s = '"' + s.replace('"', "\\\"") + '"'
                lines.append(f"{k}: {s}")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    def _rename_outputs(self, generated_paths: list[Path], seed: int) -> list[Path]:
        self.paths.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_stamp()
        out_paths: list[Path] = []
        for i, src in enumerate(generated_paths):
            suffix = "" if len(generated_paths) == 1 else f"_{i+1:02d}"
            dst = self.paths.out_dir / f"MonRT_{seed}_{stamp}{suffix}.mp4"
            try:
                if src.resolve() != dst.resolve():
                    shutil.move(str(src), str(dst))
                out_paths.append(dst)
            except Exception:
                # fallback: copy
                try:
                    shutil.copy2(str(src), str(dst))
                    out_paths.append(dst)
                except Exception:
                    pass
        return out_paths

    def _collect_new_mp4s(self, out_dir: Path, since_ts: float) -> list[Path]:
        found: list[Path] = []
        for p in out_dir.glob("*.mp4"):
            try:
                if p.stat().st_mtime >= since_ts - 0.5:
                    found.append(p)
            except Exception:
                pass
        found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return found

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_browse_ckpt.setEnabled(not running)
        self.cmb_config.setEnabled(not running)

    def _on_generate(self) -> None:
        if not self._validate_paths(update_ui=True):
            return

        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            self.lbl_status.setText("Please enter a prompt.")
            self.lbl_status.setStyleSheet("color: #c0392b;")
            return

        seed = _safe_int(self.ed_seed.text(), 0)
        if seed == -1:
            seed = int(time.time() * 1000) % (2**31 - 1)

        samples = int(self.spn_samples.value())
        guidance = float(self.ed_guidance.value())
        negative = self.txt_negative.toPlainText()
        overlap = int(self.spn_overlap.value())

        cfg_path = self._selected_config_path()
        if not cfg_path:
            self._append_log("[MonarchRT] No config selected.")
            return

        ckpt_path = Path(self.ed_checkpoint.text().strip())

        ext = self.txt_extended.toPlainText() if self.chk_extended.isChecked() else None
        prompt_file, ext_file = self._make_prompt_files(prompt, ext)
        merged_cfg = self._make_override_config(cfg_path, guidance, negative)

        self.paths.out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.paths.python_exe),
            "-u",
            "inference.py",
            "--config_path",
            str(merged_cfg),
            "--checkpoint_path",
            str(ckpt_path),
            "--data_path",
            str(prompt_file),
            "--output_folder",
            str(self.paths.out_dir),
            "--seed",
            str(seed),
            "--num_samples",
            str(samples),
            "--num_output_frames",
            str(overlap),
            "--save_with_index",
        ]
        if ext_file:
            cmd += ["--extended_prompt_path", str(ext_file)]

        if self.chk_use_ema.isChecked():
            cmd.append("--use_ema")
        if self.chk_torch_compile.isChecked():
            cmd.append("--use_torch_compile")
        if self.chk_disable_offload.isChecked():
            cmd.append("--disable_offload")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        # Help imports when running inside FrameVision.
        # inference.py uses relative imports like "from pipeline import ...".
        # Setting cwd to repo is the key, but PYTHONPATH helps too.
        env["PYTHONPATH"] = str(self.paths.repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

        self.txt_log.clear()
        self._append_log("[MonarchRT] Starting…")
        self._append_log("[MonarchRT] " + " ".join(cmd))

        self._set_running(True)
        self.lbl_status.setText("Generating…")
        self.lbl_status.setStyleSheet("")

        started = time.time()

        # worker thread
        self._worker_thread = QtCore.QThread(self)
        self._worker = _ProcWorker(cmd=cmd, cwd=self.paths.repo_dir, env=env, expected_out_dir=self.paths.out_dir)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.sig_line.connect(self._append_log)

        def _on_done(rc: int, newest: str) -> None:
            self._set_running(False)

            # Collect all MP4s written since start (handles multi-sample)
            new_files = self._collect_new_mp4s(self.paths.out_dir, since_ts=started)

            # If worker saw a newest file, prioritize it
            if newest:
                p = Path(newest)
                if p.exists() and p not in new_files:
                    new_files.insert(0, p)

            # Heuristic: MonarchRT uses save_with_index => 0-0_*.mp4 etc.
            # Keep only those created recently.
            if rc == 0 and new_files:
                renamed = self._rename_outputs(new_files[: self.spn_samples.value()], seed)
                self._last_generated = renamed
                self.lbl_status.setText(
                    "Done. Saved: " + ", ".join([p.name for p in renamed])
                )
            else:
                self.lbl_status.setText(f"Finished with code {rc}. See log for details.")
                self.lbl_status.setStyleSheet("color: #c0392b;" if rc != 0 else "")

            # cleanup thread
            if self._worker_thread:
                self._worker_thread.quit()
                self._worker_thread.wait(2000)
                self._worker_thread.deleteLater()
            self._worker_thread = None
            self._worker = None

        self._worker.sig_done.connect(_on_done)
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._append_log("[MonarchRT] Cancelling…")
            self._worker.stop()
        self._set_running(False)
        self.lbl_status.setText("Cancelled.")
        self.lbl_status.setStyleSheet("color: #c0392b;")


# --------- Optional: tiny standalone runner for dev/testing ---------

def _standalone() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MonarchRTPane()
    w.resize(860, 720)
    w.setWindowTitle("MonarchRT - FrameVision Helper")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone()
