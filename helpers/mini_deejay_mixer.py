import json
import math
import os
import subprocess
import shlex
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

APP_TITLE = "FrameVision Mini Deejay Mixer"
SETTINGS_REL = Path("presets/setsave/mini_deejay_mixer.json")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
WAV_SAMPLE_RATES = [22050, 32000, 44100, 48000]
MP3_BITRATES = [128, 160, 192, 224, 256, 320]


@dataclass
class TrackAnalysis:
    bpm: float = 0.0
    beat_offset_sec: float = 0.0
    intro_start_sec: float = 0.0
    duration_sec: float = 0.0
    sample_rate: int = 0
    onset_strength: float = 0.0
    source_path: str = ""
    analysis_ok: bool = False
    analysis_note: str = ""


@dataclass
class TrackItem:
    path: str
    analysis: TrackAnalysis


class BeatAnalyzer:
    def __init__(self, ffmpeg_path: Path, ffprobe_path: Path, temp_root: Path):
        self.ffmpeg_path = str(ffmpeg_path)
        self.ffprobe_path = str(ffprobe_path)
        self.temp_root = temp_root
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    def _probe_duration(self, src: str) -> float:
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src,
        ]
        cp = self._run(cmd)
        try:
            return float(cp.stdout.strip())
        except Exception:
            return 0.0

    def _decode_preview_wav(self, src: str) -> Optional[Path]:
        out = self.temp_root / f"analysis_{abs(hash((src, time.time()))) % 10_000_000}.wav"
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", src,
            "-ac", "1",
            "-ar", "22050",
            "-t", "180",
            str(out),
        ]
        cp = self._run(cmd)
        if cp.returncode != 0 or not out.exists():
            return None
        return out

    def _read_wav(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n)
        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        return arr, sr

    def _detect_intro_start(self, x: np.ndarray, sr: int) -> float:
        if len(x) < sr:
            return 0.0
        frame = max(512, int(sr * 0.025))
        hop = max(128, int(sr * 0.010))
        energies = []
        for i in range(0, len(x) - frame, hop):
            seg = x[i:i + frame]
            energies.append(float(np.sqrt(np.mean(seg * seg) + 1e-10)))
        if not energies:
            return 0.0
        e = np.array(energies, dtype=np.float32)
        baseline = np.percentile(e[: min(len(e), 100)], 20) if len(e) else 0.01
        threshold = max(baseline * 3.0, baseline + 0.01, np.percentile(e, 60) * 0.35)
        hits = np.where(e >= threshold)[0]
        if len(hits) == 0:
            return 0.0
        return float(hits[0] * hop / sr)

    def _detect_beats(self, x: np.ndarray, sr: int) -> Tuple[float, float, float]:
        if len(x) < sr * 2:
            return 0.0, 0.0, 0.0
        frame = 1024
        hop = 256
        x = x.astype(np.float32)
        diff_env = []
        for i in range(frame, len(x), hop):
            prev = np.abs(x[i - frame:i])
            cur = np.abs(x[i:i + frame])
            if len(cur) < frame:
                break
            pos_diff = np.maximum(cur - prev, 0.0)
            diff_env.append(float(np.sum(pos_diff)))
        env = np.array(diff_env, dtype=np.float32)
        if len(env) < 32:
            return 0.0, 0.0, 0.0
        env -= np.mean(env)
        env = np.maximum(env, 0.0)
        onset_strength = float(np.percentile(env, 90)) if len(env) else 0.0
        if np.max(env) <= 0:
            return 0.0, 0.0, onset_strength
        min_bpm = 70
        max_bpm = 180
        min_lag = int((60.0 / max_bpm) * sr / hop)
        max_lag = int((60.0 / min_bpm) * sr / hop)
        ac = np.correlate(env, env, mode="full")
        ac = ac[len(ac) // 2:]
        if max_lag >= len(ac):
            max_lag = len(ac) - 1
        if min_lag >= max_lag:
            return 0.0, 0.0, onset_strength
        segment = ac[min_lag:max_lag + 1]
        if len(segment) == 0:
            return 0.0, 0.0, onset_strength
        best = int(np.argmax(segment)) + min_lag
        spb = (best * hop) / sr
        bpm = 60.0 / spb if spb > 0 else 0.0

        smooth = env.copy()
        if len(smooth) > 4:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            smooth = np.convolve(smooth, kernel, mode="same")
        percentile = np.percentile(smooth, 85)
        candidates = np.where(smooth >= percentile)[0]
        if len(candidates) == 0:
            beat_offset = 0.0
        else:
            period_frames = max(1, int(round(spb * sr / hop)))
            first = int(candidates[0])
            best_phase = first
            best_score = -1.0
            for c in candidates[: min(24, len(candidates))]:
                idxs = np.arange(c, len(smooth), period_frames)
                score = float(np.sum(smooth[idxs]))
                if score > best_score:
                    best_score = score
                    best_phase = int(c)
            beat_offset = float(best_phase * hop / sr)
        return float(bpm), float(beat_offset), onset_strength

    def analyze(self, src: str) -> TrackAnalysis:
        duration = self._probe_duration(src)
        wav_path = self._decode_preview_wav(src)
        if not wav_path:
            return TrackAnalysis(duration_sec=duration, source_path=src, analysis_ok=False, analysis_note="ffmpeg decode failed")
        try:
            x, sr = self._read_wav(wav_path)
            intro = self._detect_intro_start(x, sr)
            bpm, beat_offset, onset_strength = self._detect_beats(x, sr)
            note = "ok" if bpm > 0 else "beat detection fallback needed"
            return TrackAnalysis(
                bpm=bpm,
                beat_offset_sec=beat_offset,
                intro_start_sec=intro,
                duration_sec=duration,
                sample_rate=sr,
                onset_strength=onset_strength,
                source_path=src,
                analysis_ok=bpm > 0,
                analysis_note=note,
            )
        except Exception as exc:
            return TrackAnalysis(duration_sec=duration, source_path=src, analysis_ok=False, analysis_note=str(exc))
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass


class AnalysisWorker(QThread):
    progress = Signal(str)
    finished_with_tracks = Signal(list)

    def __init__(self, analyzer: BeatAnalyzer, paths: List[str]):
        super().__init__()
        self.analyzer = analyzer
        self.paths = paths

    def run(self):
        out = []
        total = len(self.paths)
        for idx, path in enumerate(self.paths, start=1):
            self.progress.emit(f"Analyzing {idx}/{total}: {Path(path).name}")
            analysis = self.analyzer.analyze(path)
            out.append(TrackItem(path=path, analysis=analysis))
        self.finished_with_tracks.emit(out)


class RenderWorker(QThread):
    log_line = Signal(str)
    progress_value = Signal(int)
    finished_path = Signal(str)
    failed = Signal(str)

    def __init__(self, mixer: 'MixerEngine', tracks: List[TrackItem], settings: Dict, output_path: str):
        super().__init__()
        self.mixer = mixer
        self.tracks = tracks
        self.settings = settings
        self.output_path = output_path

    def run(self):
        try:
            self.log_line.emit("Planning mix timeline...")
            plan = self.mixer.build_mix_plan(self.tracks, self.settings)
            self.log_line.emit(f"Timeline contains {len(plan)} entries")
            self.progress_value.emit(15)
            out = self.mixer.render(plan, self.output_path, self.settings, self.log_line.emit)
            self.progress_value.emit(100)
            self.finished_path.emit(out)
        except Exception as exc:
            self.failed.emit(str(exc))


class MixerEngine:
    def __init__(self, ffmpeg_path: Path, ffprobe_path: Path):
        self.ffmpeg_path = str(ffmpeg_path)
        self.ffprobe_path = str(ffprobe_path)

    @staticmethod
    def _safe_bpm(a: TrackAnalysis, fallback: float = 124.0) -> float:
        if a.bpm and 60 <= a.bpm <= 200:
            return a.bpm
        return fallback

    @staticmethod
    def quantize_time(target_sec: float, beat_grid_start_sec: float, bpm: float) -> float:
        beat = 60.0 / max(1e-6, bpm)
        if target_sec <= beat_grid_start_sec:
            return beat_grid_start_sec
        n = math.ceil((target_sec - beat_grid_start_sec) / beat)
        return beat_grid_start_sec + (n * beat)

    @staticmethod
    def _compute_base_target_bpm(tracks: List[TrackItem], settings: Dict) -> float:
        manual_bpm = int(settings.get("target_bpm", 0) or 0)
        valid_bpms = [t.analysis.bpm for t in tracks if 60 <= t.analysis.bpm <= 200]
        if manual_bpm > 0:
            return float(manual_bpm)
        if valid_bpms:
            return float(sum(valid_bpms) / len(valid_bpms))
        return 124.0

    @staticmethod
    def _target_for_track(original_bpm: float, requested_target: float, max_delta_bpm: float = 10.0) -> float:
        low = max(60.0, original_bpm - max_delta_bpm)
        high = min(200.0, original_bpm + max_delta_bpm)
        return min(high, max(low, requested_target))

    @staticmethod
    def _entry_source_start(a: TrackAnalysis, original_bpm: float) -> float:
        beat_len = 60.0 / max(1e-6, original_bpm)
        base = max(0.0, a.beat_offset_sec)
        intro = max(0.0, a.intro_start_sec)
        if beat_len <= 0.0:
            return max(base, intro)
        # Enter on the first beat on or after the audible intro/start section.
        if intro <= base:
            return base
        steps = math.ceil((intro - base) / beat_len)
        return max(0.0, base + (steps * beat_len))

    def build_mix_plan(self, tracks: List[TrackItem], settings: Dict) -> List[Dict]:
        if not tracks:
            raise RuntimeError("No tracks loaded")
        fade_sec = float(settings.get("fade_sec", 8))
        handoff_remaining_sec = float(settings.get("time_left_sec", 24))
        requested_target_bpm = self._compute_base_target_bpm(tracks, settings)
        plan: List[Dict] = []
        current_start = 0.0

        for idx, item in enumerate(tracks):
            a = item.analysis
            original_bpm = self._safe_bpm(a, requested_target_bpm)
            target_bpm = self._target_for_track(original_bpm, requested_target_bpm, 10.0)
            atempo = target_bpm / max(original_bpm, 1e-6)
            atempo = max(0.5, min(2.0, atempo))

            # The old helper delayed the raw file from t=0 and only *imagined* the first beat
            # would land on the right grid. For DJ-style beatmatching that is too weak.
            # Instead we actually start each incoming track from its detected first beat,
            # so the audible material that enters the mix is the beat-aligned section.
            source_start_sec = self._entry_source_start(a, original_bpm)
            remaining_source_sec = max(0.001, (a.duration_sec - source_start_sec) if a.duration_sec > 0 else 0.001)
            adjusted_duration = remaining_source_sec / atempo

            entry = {
                "index": idx,
                "path": item.path,
                "name": Path(item.path).name,
                "bpm_original": original_bpm,
                "bpm_target": target_bpm,
                "requested_target_bpm": requested_target_bpm,
                "atempo": atempo,
                "source_start_sec": source_start_sec,
                "beat_offset_sec": 0.0,
                "intro_start_sec": max(0.0, a.intro_start_sec - source_start_sec) / atempo if a.intro_start_sec > source_start_sec else 0.0,
                "duration_sec": adjusted_duration,
                "mix_start_sec": current_start,
                "fade_in_sec": fade_sec if idx > 0 else 0.0,
                "fade_out_sec": 0.0,
                "source_analysis_ok": a.analysis_ok,
                "analysis_note": a.analysis_note,
            }
            plan.append(entry)
            if idx < len(tracks) - 1:
                desired = current_start + max(0.0, adjusted_duration - handoff_remaining_sec)
                # Quantize on the shared target grid, not the raw track grid.
                beat_aligned_time = self.quantize_time(desired, current_start, target_bpm)
                entry["fade_out_sec"] = fade_sec
                current_start = max(0.0, beat_aligned_time)
            else:
                current_start += adjusted_duration
        return plan

    @staticmethod
    def _tempo_filters(tempo: float) -> List[str]:
        tempo = max(0.5, min(2.0, float(tempo)))
        if abs(tempo - 1.0) <= 0.0001:
            return []
        return [f"atempo={tempo:.6f}"]

    def render(self, plan: List[Dict], output_path: str, settings: Dict, log) -> str:
        output_path_obj = Path(output_path)
        ext = output_path_obj.suffix.lower()
        output_format = settings.get("output_format", "wav")
        if ext not in {".wav", ".mp3"}:
            ext = ".wav" if output_format == "wav" else ".mp3"
            output_path_obj = output_path_obj.with_suffix(ext)
            output_path = str(output_path_obj)

        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = int(settings.get("wav_sample_rate", 48000))
        mp3_bitrate = int(settings.get("mp3_bitrate", 320))
        normalize_audio = bool(settings.get("normalize_audio", False))

        cmd = [self.ffmpeg_path, "-y", "-hide_banner"]
        for entry in plan:
            cmd += ["-i", entry["path"]]

        if not plan:
            raise RuntimeError("Nothing to render. Add at least one analyzed track.")

        filter_parts = []
        mix_inputs = []
        for entry in plan:
            i = entry["index"]
            a_lbl = f"a{i}"
            dur = max(0.001, float(entry.get("duration_sec", 0.0) or 0.0))
            delay_ms = int(max(0.0, float(entry.get("mix_start_sec", 0.0) or 0.0)) * 1000)
            fade_in = max(0.0, float(entry.get("fade_in_sec", 0.0) or 0.0))
            fade_out = max(0.0, float(entry.get("fade_out_sec", 0.0) or 0.0))

            source_start = max(0.0, float(entry.get("source_start_sec", 0.0) or 0.0))
            source_end = source_start + (dur * max(0.5, min(2.0, float(entry["atempo"]))))
            filters = [
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                f"aresample={sample_rate}",
                f"atrim={source_start:.3f}:{source_end:.3f}",
                "asetpts=N/SR/TB",
            ]
            filters.extend(self._tempo_filters(entry["atempo"]))
            if normalize_audio:
                filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
            if fade_in > 0:
                filters.append(f"afade=t=in:st=0:d={min(fade_in, dur):.3f}")
            if fade_out > 0 and dur > 0.05:
                fade_out = min(fade_out, max(0.01, dur - 0.01))
                filters.append(f"afade=t=out:st={max(0.0, dur - fade_out):.3f}:d={fade_out:.3f}")
            if delay_ms > 0:
                filters.append(f"adelay={delay_ms}|{delay_ms}")
            filter_parts.append(f"[{i}:a]" + ",".join(filters) + f"[{a_lbl}]")
            mix_inputs.append(f"[{a_lbl}]")

        filter_parts.append(
            "".join(mix_inputs) + f"amix=inputs={len(plan)}:normalize=0:dropout_transition=0,alimiter=limit=0.97[aout]"
        )
        filter_complex = ";".join(filter_parts)
        cmd += ["-filter_complex", filter_complex, "-map", "[aout]"]

        if ext == ".wav":
            cmd += ["-c:a", "pcm_s16le", "-ar", str(sample_rate), output_path]
            log(f"Rendering WAV at {sample_rate} Hz")
        elif ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", f"{mp3_bitrate}k", "-ar", str(sample_rate), output_path]
            log(f"Rendering MP3 at {mp3_bitrate} kbps")
        else:
            raise RuntimeError("Only WAV and MP3 are supported in this helper.")

        if normalize_audio:
            log("Normalize enabled: applying loudnorm per track before mixing")
        log("Running ffmpeg render...")
        log("FFmpeg command: " + " ".join(shlex.quote(part) for part in cmd))

        cp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        log_lines = []
        if cp.stdout:
            for line in cp.stdout:
                line = line.rstrip()
                if line:
                    log_lines.append(line)
                    if len(log_lines) > 120:
                        log_lines = log_lines[-120:]
                    log(line)
        rc = cp.wait()
        if rc != 0 or not output_path_obj.exists():
            tail = "\n".join(log_lines[-25:]) if log_lines else "No ffmpeg output captured."
            raise RuntimeError(f"ffmpeg render failed with code {rc}\n\nLast ffmpeg output:\n{tail}")
        return str(output_path_obj)


class MiniDeejayMixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1460, 940)

        self.root_dir = self._guess_framevision_root()
        self.bin_dir = self.root_dir / "presets" / "bin"
        self.ffmpeg_path = self.bin_dir / "ffmpeg.exe"
        self.ffprobe_path = self.bin_dir / "ffprobe.exe"
        self.ffplay_path = self.bin_dir / "ffplay.exe"
        self.python_path = self.root_dir / "environments" / ".qwen3tts" / "Scripts" / "python.exe"
        self.settings_path = self.root_dir / SETTINGS_REL
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_root = self.root_dir / "temp" / "mini_deejay_mixer"
        self.temp_root.mkdir(parents=True, exist_ok=True)

        self.track_items: List[TrackItem] = []
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.render_worker: Optional[RenderWorker] = None
        self.preview_process: Optional[subprocess.Popen] = None
        self._loading_settings = False

        self._build_ui()
        self._load_settings()
        self._refresh_paths()
        self._sync_controls()
        self._update_output_controls()
        self._update_bpm_info()

    def _guess_framevision_root(self) -> Path:
        env = os.environ.get("FRAMEVISION_ROOT", "").strip()
        if env:
            return Path(env)
        helper_dir = Path(__file__).resolve().parent
        if helper_dir.name.lower() == "helpers":
            return helper_dir.parent
        return helper_dir.parent

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        main = QVBoxLayout(body)

        top = QSplitter(Qt.Horizontal)
        main.addWidget(top, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        right = QWidget()
        right_l = QVBoxLayout(right)
        top.addWidget(left)
        top.addWidget(right)
        top.setSizes([760, 650])

        track_group = QGroupBox("Track manager")
        track_l = QVBoxLayout(track_group)
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load tracks")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear all")
        self.btn_up = QPushButton("Move up")
        self.btn_down = QPushButton("Move down")
        self.btn_preview = QPushButton("Preview")
        self.btn_stop_preview = QPushButton("Stop preview")
        for b in [self.btn_load, self.btn_remove, self.btn_clear, self.btn_up, self.btn_down, self.btn_preview, self.btn_stop_preview]:
            btn_row.addWidget(b)
        track_l.addLayout(btn_row)
        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QAbstractItemView.SingleSelection)
        track_l.addWidget(self.track_list, 1)
        self.analysis_label = QLabel("No tracks loaded")
        track_l.addWidget(self.analysis_label)
        left_l.addWidget(track_group, 2)

        analysis_group = QGroupBox("Auto sync and quantize analysis")
        analysis_l = QVBoxLayout(analysis_group)
        self.analysis_table = QTableWidget(0, 7)
        self.analysis_table.setHorizontalHeaderLabels(["Track", "BPM", "Beat offset", "Intro start", "Duration", "Status", "Note"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        analysis_l.addWidget(self.analysis_table)
        left_l.addWidget(analysis_group, 2)

        settings_group = QGroupBox("Mix settings")
        settings_l = QVBoxLayout(settings_group)
        form = QFormLayout()

        self.fade_slider = QSlider(Qt.Horizontal)
        self.fade_slider.setRange(0, 60)
        self.fade_spin = QSpinBox()
        self.fade_spin.setRange(0, 60)
        fade_box = QHBoxLayout()
        fade_box.addWidget(self.fade_slider)
        fade_box.addWidget(self.fade_spin)
        fade_wrap = QWidget(); fade_wrap.setLayout(fade_box)
        form.addRow("Fade length (sec)", fade_wrap)

        self.left_slider = QSlider(Qt.Horizontal)
        self.left_slider.setRange(0, 180)
        self.left_spin = QSpinBox()
        self.left_spin.setRange(0, 180)
        left_box = QHBoxLayout()
        left_box.addWidget(self.left_slider)
        left_box.addWidget(self.left_spin)
        left_wrap = QWidget(); left_wrap.setLayout(left_box)
        form.addRow("Time left before next track (sec)", left_wrap)

        self.chk_normalize = QCheckBox("Normalize all tracks to similar loudness")
        form.addRow("Normalize", self.chk_normalize)

        bpm_row = QHBoxLayout()
        self.target_bpm_spin = QSpinBox()
        self.target_bpm_spin.setRange(0, 220)
        self.target_bpm_spin.setSpecialValueText("Auto average")
        bpm_row.addWidget(self.target_bpm_spin)
        bpm_help = QLabel("0 = average the detected BPMs. Each track is only pushed up to max ±10 BPM from its own detected BPM.")
        bpm_help.setWordWrap(True)
        bpm_wrap = QWidget(); bpm_wrap.setLayout(bpm_row)
        form.addRow("Set BPM", bpm_wrap)
        form.addRow("BPM behavior", bpm_help)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["wav", "mp3"])
        form.addRow("Save as", self.output_format_combo)

        self.wav_rate_combo = QComboBox()
        for rate in WAV_SAMPLE_RATES:
            self.wav_rate_combo.addItem(str(rate), rate)
        form.addRow("WAV sample rate", self.wav_rate_combo)

        self.mp3_bitrate_combo = QComboBox()
        for br in MP3_BITRATES:
            self.mp3_bitrate_combo.addItem(str(br), br)
        form.addRow("MP3 bitrate (kbps)", self.mp3_bitrate_combo)

        self.fade_spin.setValue(8)
        self.left_spin.setValue(24)
        wav_default_idx = self.wav_rate_combo.findData(44100)
        if wav_default_idx >= 0:
            self.wav_rate_combo.setCurrentIndex(wav_default_idx)
        mp3_default_idx = self.mp3_bitrate_combo.findData(320)
        if mp3_default_idx >= 0:
            self.mp3_bitrate_combo.setCurrentIndex(mp3_default_idx)

        self.behavior_label = QLabel(
            "When the remaining-time trigger is reached, the next track is shifted so its first detected beat lands on the outgoing track's next beat after tempo matching."
        )
        self.behavior_label.setWordWrap(True)
        form.addRow("Behavior", self.behavior_label)

        self.target_bpm_info = QLabel("")
        self.target_bpm_info.setWordWrap(True)
        form.addRow("Current target BPM", self.target_bpm_info)

        settings_l.addLayout(form)
        right_l.addWidget(settings_group)

        path_group = QGroupBox("Paths")
        path_l = QFormLayout(path_group)
        self.root_edit = QLineEdit(); self.root_edit.setReadOnly(True)
        self.ffmpeg_edit = QLineEdit(); self.ffmpeg_edit.setReadOnly(True)
        self.ffprobe_edit = QLineEdit(); self.ffprobe_edit.setReadOnly(True)
        self.ffplay_edit = QLineEdit(); self.ffplay_edit.setReadOnly(True)
        self.python_edit = QLineEdit(); self.python_edit.setReadOnly(True)
        path_l.addRow("FrameVision root", self.root_edit)
        path_l.addRow("ffmpeg", self.ffmpeg_edit)
        path_l.addRow("ffprobe", self.ffprobe_edit)
        path_l.addRow("ffplay", self.ffplay_edit)
        path_l.addRow("Python env", self.python_edit)
        right_l.addWidget(path_group)

        plan_group = QGroupBox("Mix plan")
        plan_l = QVBoxLayout(plan_group)
        self.plan_table = QTableWidget(0, 8)
        self.plan_table.setHorizontalHeaderLabels(["#", "Track", "Start", "Orig BPM", "Target BPM", "Tempo", "Fade in", "Fade out"])
        self.plan_table.horizontalHeader().setStretchLastSection(True)
        plan_l.addWidget(self.plan_table)
        right_l.addWidget(plan_group, 2)

        bottom = QHBoxLayout()
        self.btn_refresh_plan = QPushButton("Refresh plan")
        self.btn_render = QPushButton("Render mixed track")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        bottom.addWidget(self.btn_refresh_plan)
        bottom.addWidget(self.btn_render)
        bottom.addWidget(self.progress)
        right_l.addLayout(bottom)

        log_group = QGroupBox("Logs")
        log_l = QVBoxLayout(log_group)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        log_l.addWidget(self.logs)
        right_l.addWidget(log_group, 1)

        self.btn_load.clicked.connect(self.load_tracks)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_tracks)
        self.btn_up.clicked.connect(self.move_up)
        self.btn_down.clicked.connect(self.move_down)
        self.btn_preview.clicked.connect(self.preview_selected)
        self.btn_stop_preview.clicked.connect(self.stop_preview)
        self.btn_refresh_plan.clicked.connect(self.refresh_plan)
        self.btn_render.clicked.connect(self.render_mix)
        self.fade_slider.valueChanged.connect(self.fade_spin.setValue)
        self.fade_spin.valueChanged.connect(self.fade_slider.setValue)
        self.left_slider.valueChanged.connect(self.left_spin.setValue)
        self.left_spin.valueChanged.connect(self.left_slider.setValue)
        self.output_format_combo.currentTextChanged.connect(self._update_output_controls)
        self.target_bpm_spin.valueChanged.connect(self._on_settings_changed)
        self.chk_normalize.stateChanged.connect(self._on_settings_changed)
        self.fade_spin.valueChanged.connect(self._on_settings_changed)
        self.left_spin.valueChanged.connect(self._on_settings_changed)
        self.wav_rate_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.mp3_bitrate_combo.currentIndexChanged.connect(self._on_settings_changed)

    def _append_log(self, text: str):
        self.logs.appendPlainText(text)

    def _refresh_paths(self):
        self.root_edit.setText(str(self.root_dir))
        self.ffmpeg_edit.setText(str(self.ffmpeg_path))
        self.ffprobe_edit.setText(str(self.ffprobe_path))
        self.ffplay_edit.setText(str(self.ffplay_path))
        self.python_edit.setText(str(self.python_path))

    def _sync_controls(self):
        self.fade_slider.setValue(self.fade_spin.value())
        self.left_slider.setValue(self.left_spin.value())

    def _update_output_controls(self):
        is_wav = self.output_format_combo.currentText().lower() == "wav"
        self.wav_rate_combo.setEnabled(is_wav)
        self.mp3_bitrate_combo.setEnabled(not is_wav)

    def _update_bpm_info(self):
        valid = [t.analysis.bpm for t in self.track_items if 60 <= t.analysis.bpm <= 200]
        if self.target_bpm_spin.value() > 0:
            self.target_bpm_info.setText(f"Manual target: {self.target_bpm_spin.value()} BPM. Effective target per track is clamped to max ±10 BPM.")
        elif valid:
            avg = sum(valid) / len(valid)
            self.target_bpm_info.setText(f"Auto target average: {avg:.2f} BPM, then clamped per track to max ±10 BPM.")
        else:
            self.target_bpm_info.setText("Auto target fallback: 124 BPM until tracks are analyzed.")

    def _on_settings_changed(self):
        if self._loading_settings:
            return
        self._update_bpm_info()
        self._update_output_controls()
        self._save_settings()
        if self.track_items:
            self.refresh_plan()

    def closeEvent(self, event):
        self.stop_preview()
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self):
        data = {
            "fade_sec": self.fade_spin.value(),
            "time_left_sec": self.left_spin.value(),
            "root_dir": str(self.root_dir),
            "normalize_audio": self.chk_normalize.isChecked(),
            "target_bpm": self.target_bpm_spin.value(),
            "output_format": self.output_format_combo.currentText(),
            "wav_sample_rate": int(self.wav_rate_combo.currentData() or 48000),
            "mp3_bitrate": int(self.mp3_bitrate_combo.currentData() or 320),
        }
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"Failed to save settings: {exc}")

    def _load_settings(self):
        if not self.settings_path.exists():
            return
        self._loading_settings = True
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            self.fade_spin.setValue(int(data.get("fade_sec", self.fade_spin.value())))
            self.left_spin.setValue(int(data.get("time_left_sec", self.left_spin.value())))
            self.chk_normalize.setChecked(bool(data.get("normalize_audio", False)))
            self.target_bpm_spin.setValue(int(data.get("target_bpm", 0)))
            fmt = str(data.get("output_format", self.output_format_combo.currentText())).lower()
            idx = self.output_format_combo.findText(fmt)
            if idx >= 0:
                self.output_format_combo.setCurrentIndex(idx)
            wav_rate = int(data.get("wav_sample_rate", int(self.wav_rate_combo.currentData() or 44100)))
            idx = self.wav_rate_combo.findData(wav_rate)
            if idx >= 0:
                self.wav_rate_combo.setCurrentIndex(idx)
            mp3_br = int(data.get("mp3_bitrate", int(self.mp3_bitrate_combo.currentData() or 320)))
            idx = self.mp3_bitrate_combo.findData(mp3_br)
            if idx >= 0:
                self.mp3_bitrate_combo.setCurrentIndex(idx)
        except Exception as exc:
            self._append_log(f"Failed to load settings: {exc}")
        finally:
            self._loading_settings = False

    def _ensure_bins(self) -> bool:
        missing = []
        for p in [self.ffmpeg_path, self.ffprobe_path, self.ffplay_path, self.python_path]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            QMessageBox.warning(self, APP_TITLE, "Missing required paths:\n\n" + "\n".join(missing))
            return False
        return True

    def load_tracks(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load tracks", str(self.root_dir), "Audio files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.wma);;All files (*.*)")
        if not files:
            return
        if not self._ensure_bins():
            return
        analyzer = BeatAnalyzer(self.ffmpeg_path, self.ffprobe_path, self.temp_root)
        self.analysis_worker = AnalysisWorker(analyzer, files)
        self.analysis_worker.progress.connect(self._append_log)
        self.analysis_worker.finished_with_tracks.connect(self._add_analyzed_tracks)
        self.analysis_label.setText(f"Analyzing {len(files)} track(s)...")
        self.analysis_worker.start()

    def _add_analyzed_tracks(self, tracks: List[TrackItem]):
        for item in tracks:
            self.track_items.append(item)
            lw = QListWidgetItem(Path(item.path).name)
            lw.setToolTip(item.path)
            self.track_list.addItem(lw)
        self.analysis_label.setText(f"Loaded {len(self.track_items)} track(s)")
        self.refresh_analysis_table()
        self._update_bpm_info()
        self.refresh_plan()

    def refresh_analysis_table(self):
        self.analysis_table.setRowCount(len(self.track_items))
        for row, item in enumerate(self.track_items):
            a = item.analysis
            vals = [
                Path(item.path).name,
                f"{a.bpm:.2f}" if a.bpm else "?",
                f"{a.beat_offset_sec:.3f}s",
                f"{a.intro_start_sec:.3f}s",
                f"{a.duration_sec:.2f}s",
                "OK" if a.analysis_ok else "Heuristic",
                a.analysis_note,
            ]
            for col, val in enumerate(vals):
                self.analysis_table.setItem(row, col, QTableWidgetItem(str(val)))

    def current_settings(self) -> Dict:
        return {
            "fade_sec": self.fade_spin.value(),
            "time_left_sec": self.left_spin.value(),
            "normalize_audio": self.chk_normalize.isChecked(),
            "target_bpm": self.target_bpm_spin.value(),
            "output_format": self.output_format_combo.currentText().lower(),
            "wav_sample_rate": int(self.wav_rate_combo.currentData() or 48000),
            "mp3_bitrate": int(self.mp3_bitrate_combo.currentData() or 320),
        }

    def refresh_plan(self):
        self.plan_table.setRowCount(0)
        if not self.track_items:
            self._update_bpm_info()
            return
        try:
            engine = MixerEngine(self.ffmpeg_path, self.ffprobe_path)
            plan = engine.build_mix_plan(self.track_items, self.current_settings())
            self.plan_table.setRowCount(len(plan))
            for row, entry in enumerate(plan):
                vals = [
                    entry["index"] + 1,
                    entry["name"],
                    f"{entry['mix_start_sec']:.2f}s",
                    f"{entry['bpm_original']:.2f}",
                    f"{entry['bpm_target']:.2f}",
                    f"{entry['atempo']:.4f}",
                    f"{entry['fade_in_sec']:.2f}s",
                    f"{entry['fade_out_sec']:.2f}s",
                ]
                for col, val in enumerate(vals):
                    self.plan_table.setItem(row, col, QTableWidgetItem(str(val)))
        except Exception as exc:
            self._append_log(f"Plan refresh failed: {exc}")

    def remove_selected(self):
        row = self.track_list.currentRow()
        if row < 0:
            return
        self.track_list.takeItem(row)
        del self.track_items[row]
        self.refresh_analysis_table()
        self._update_bpm_info()
        self.refresh_plan()
        self.analysis_label.setText(f"Loaded {len(self.track_items)} track(s)")

    def clear_tracks(self):
        self.track_items.clear()
        self.track_list.clear()
        self.analysis_table.setRowCount(0)
        self.plan_table.setRowCount(0)
        self.analysis_label.setText("No tracks loaded")
        self._update_bpm_info()

    def move_up(self):
        row = self.track_list.currentRow()
        if row <= 0:
            return
        self.track_items[row - 1], self.track_items[row] = self.track_items[row], self.track_items[row - 1]
        item = self.track_list.takeItem(row)
        self.track_list.insertItem(row - 1, item)
        self.track_list.setCurrentRow(row - 1)
        self.refresh_analysis_table()
        self.refresh_plan()

    def move_down(self):
        row = self.track_list.currentRow()
        if row < 0 or row >= len(self.track_items) - 1:
            return
        self.track_items[row + 1], self.track_items[row] = self.track_items[row], self.track_items[row + 1]
        item = self.track_list.takeItem(row)
        self.track_list.insertItem(row + 1, item)
        self.track_list.setCurrentRow(row + 1)
        self.refresh_analysis_table()
        self.refresh_plan()

    def preview_selected(self):
        row = self.track_list.currentRow()
        if row < 0:
            return
        if not self._ensure_bins():
            return
        self.stop_preview()
        src = self.track_items[row].path
        try:
            self.preview_process = subprocess.Popen([str(self.ffplay_path), "-nodisp", "-autoexit", src])
            self._append_log(f"Previewing: {Path(src).name}")
        except Exception as exc:
            QMessageBox.warning(self, APP_TITLE, f"Preview failed:\n{exc}")

    def stop_preview(self):
        if self.preview_process and self.preview_process.poll() is None:
            try:
                self.preview_process.terminate()
            except Exception:
                pass
        self.preview_process = None

    def render_mix(self):
        if not self.track_items:
            QMessageBox.information(self, APP_TITLE, "Load at least one track first.")
            return
        if not self._ensure_bins():
            return
        fmt = self.output_format_combo.currentText().lower()
        default_name = "mini_deejay_mix.wav" if fmt == "wav" else "mini_deejay_mix.mp3"
        filter_text = "WAV (*.wav)" if fmt == "wav" else "MP3 (*.mp3)"
        out, _ = QFileDialog.getSaveFileName(self, "Render mixed track", str(self.root_dir / "output" / default_name), filter_text)
        if not out:
            return
        self.progress.setValue(0)
        engine = MixerEngine(self.ffmpeg_path, self.ffprobe_path)
        self.render_worker = RenderWorker(engine, self.track_items, self.current_settings(), out)
        self.render_worker.log_line.connect(self._append_log)
        self.render_worker.progress_value.connect(self.progress.setValue)
        self.render_worker.finished_path.connect(self._render_done)
        self.render_worker.failed.connect(self._render_failed)
        self.render_worker.start()

    def _render_done(self, path: str):
        self.progress.setValue(100)
        QMessageBox.information(self, APP_TITLE, f"Mix created:\n{path}")

    def _render_failed(self, msg: str):
        self.progress.setValue(0)
        QMessageBox.warning(self, APP_TITLE, f"Render failed:\n{msg}")


def main():
    app = QApplication(sys.argv)
    w = MiniDeejayMixer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
