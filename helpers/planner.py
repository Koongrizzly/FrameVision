
"""
Planner (PySide6) - Prompt -> Finished Video pipeline UI placeholder

Goal:
- Provide a clear, wire-ready UI and pipeline skeleton that can later be connected to:
  - Qwen3-VL (story + prompts + describe)
  - Whisper (transcript)
  - Qwen3 TTS (voice)
  - Image generation (Z-Image, Qwen 2512, SDXL, etc.)
  - Video generation (Qwen 2.2 5B, HunyuanVideo, etc.)
  - Videoclip Creator preset runner (FrameVision internal tool)

This file is intentionally "placeholder" and focuses on:
- Options + UX
- A staged pipeline runner (QThread) with logs + progress
- A job config dict that can be serialized and passed to later code
"""

from __future__ import annotations

import os
import json
import time
import uuid
import hashlib
import sys
import io
import contextlib
import traceback
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple


# -----------------------------
# Project root / path helpers
# -----------------------------
# Planner lives in <root>/helpers/planner.py
# We want outputs and model paths rooted at <root>, not inside /helpers.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]  # one folder up from helpers/

def _root() -> Path:
    return _PROJECT_ROOT

def _default_output_base() -> str:
    return str((_root() / "output" / "placeholdr").resolve())

def _abspath_from_root(p: str) -> str:
    try:
        pp = Path(p)
        if not pp.is_absolute():
            pp = _root() / pp
        return str(pp.resolve())
    except Exception:
        return str((_root() / p).resolve())

def _qwen_model_dir() -> str:
    return str((_root() / "models" / "describe" / "default" / "qwen3vl2b").resolve())


# -----------------------------
# Qwen3-VL text generation helper (reuse helpers.prompt)
# -----------------------------
# Ensure project root and helpers are importable even when this file is run standalone.
for _p in (str(_root()), str((_root() / "helpers"))):
    try:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    except Exception:
        pass

_HAVE_QWEN_TEXT = False
_QWEN_IMPORT_ERROR: Optional[Exception] = None
try:
    # Preferred path (same one used by Prompt tab / CLI)
    from helpers.prompt import _generate_with_qwen_text as _qwen_generate_text  # type: ignore
    _HAVE_QWEN_TEXT = True
except Exception as _e1:
    try:
        from prompt import _generate_with_qwen_text as _qwen_generate_text  # type: ignore
        _HAVE_QWEN_TEXT = True
    except Exception as _e2:
        _QWEN_IMPORT_ERROR = _e2 or _e1
        _qwen_generate_text = None  # type: ignore




# -----------------------------
# Video generation presets (placeholders, but real numbers)
# -----------------------------
# These represent "generation targets" (proxy). Final export resolution/quality is handled later (upscale/encode).
_HUNYUAN_PRESETS = {
    "low":    {"model": "hunyuan", "quality": "low",    "res": "368p", "fps": 15, "steps": 9, "min_sec": 2.5, "max_sec": 7.0,
               "model_variant": "distilled_8step", "note": "Fast proxy, longer clips possible"},
    "medium": {"model": "hunyuan", "quality": "medium", "res": "384p", "fps": 20, "steps": 9, "min_sec": 2.5, "max_sec": 5.0,
               "model_variant": "distilled_8step", "note": "Default proxy"},
    "high":   {"model": "hunyuan", "quality": "high",   "res": "432p", "fps": 18, "steps": 9, "min_sec": 2.5, "max_sec": 4.0,
               "model_variant": "distilled_8step", "note": "Heavier proxy, shorter clips"},
}

_WAN22_PRESETS = {
    "normal": {"model": "wan22", "quality": "normal", "res": "704p", "fps": 15, "steps": 25, "min_sec": 2.5, "max_sec": 5.0,
               "note": "Practical proxy (interpolate later)"},
    "high":   {"model": "wan22", "quality": "high",   "res": "704p", "fps": 24, "steps": 30, "min_sec": 2.5, "max_sec": 3.5,
               "note": "Best native motion, heavier"},
}

def _normalize_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

def _video_model_key(label: str) -> str:
    l = (label or "").lower()
    if "wan" in l and "2.2" in l:
        return "wan22"
    if "hunyuan" in l:
        return "hunyuan"
    # default (also for Auto / Qwen entries for now)
    return "hunyuan"

def _resolve_generation_profile(video_model_label: str, gen_quality_label: str) -> Dict[str, Any]:
    mk = _video_model_key(video_model_label)
    gk = _normalize_key(gen_quality_label)
    if mk == "wan22":
        if gk not in _WAN22_PRESETS:
            gk = "normal"
        return dict(_WAN22_PRESETS[gk])
    # hunyuan default
    if gk not in _HUNYUAN_PRESETS:
        gk = "medium"
    return dict(_HUNYUAN_PRESETS[gk])

def _stable_uniform(seed_text: str, lo: float, hi: float) -> float:
    # deterministic float in [lo, hi]
    r = random.Random(_sha1_text(seed_text))
    if hi <= lo:
        return float(lo)
    return lo + (hi - lo) * r.random()


from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QSlider,
    QComboBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QMessageBox,
    QSplitter,
    QToolButton,
    QMenu,
    QSizePolicy,
)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class PlaceholdrJob:
    job_id: str
    created_at: float

    # Step 1 inputs
    prompt: str
    storytelling: bool
    music_background: bool
    silent: bool
    storytelling_volume: int  # 0-100
    music_volume: int         # 0-100

    approx_duration_sec: int  # 5 .. 600
    resolution_preset: str    # e.g. "720p Landscape (16:9)"
    quality_preset: str       # "Low" | "Medium" | "High"
    extra_info: str

    # Optional "future" attachments
    attachments: Dict[str, List[str]]  # keys: json/images/videos/text/transcripts

    # Internal: derived encoding targets (placeholder mapping)
    encoding: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _quality_defaults(name: str) -> Dict[str, Any]:
    """
    Placeholder quality mapping:
    - Low: smaller disk, lower bitrate / higher CRF
    - Medium: default-ish: CRF 18 OR ~3000k
    - High: close to max but not extreme: CRF 16 OR ~6000k
    """
    n = (name or "").strip().lower()
    if n == "low":
        return {"mode": "crf", "crf": 22, "bitrate_kbps": 1500, "note": "Small files, faster encode"}
    if n == "high":
        return {"mode": "crf", "crf": 16, "bitrate_kbps": 6000, "note": "Near-high quality, larger files"}
    # Medium default
    return {"mode": "crf", "crf": 18, "bitrate_kbps": 3000, "note": "Balanced (default)"}


def _duration_label(seconds: int) -> str:
    m = seconds // 60
    s = seconds % 60
    if m <= 0:
        return f"{seconds}s"
    return f"{m}m {s:02d}s"




def _sha1_text(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8", errors="ignore")).hexdigest()


def _safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_shots_list(path: str) -> List[Dict[str, Any]]:
    """Read shots.json accepting either:
    - top-level list of shot dicts (preferred)
    - legacy wrapper: { "shots": [ ... ] }
    """
    obj = _safe_read_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        v = obj.get("shots")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _file_ok(path: str, min_bytes: int = 2) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def _extract_first_json(raw: str) -> Optional[Any]:
    """Robustly extract the first JSON object/array from a model response."""
    if not raw:
        return None
    s = raw.strip()
    # Strip fenced blocks if present
    if "```" in s:
        # try to keep content inside the first fence
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1] if parts[1].strip() else parts[2]
            s = s.strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

    # Find first { or [
    start = None
    for k,ch in enumerate(s):
        if ch in "{[":
            start = k
            break
    if start is None:
        return None

    def _balanced_extract(ss: str, st: int) -> Optional[str]:
        depth = 0
        in_str = False
        esc = False
        open_ch = ss[st]
        close_ch = "}" if open_ch == "{" else "]"
        for i in range(st, len(ss)):
            ch = ss[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return ss[st:i+1]
        return None

    cand = _balanced_extract(s, start)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass

    # Fallback: try progressively from start
    for end in range(len(s), start+1, -1):
        chunk = s[start:end].strip()
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def _append_prompt_used(path: str, title: str, system_prompt: str, user_prompt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write("-"*80 + "\n")
        f.write("SYSTEM:\n" + (system_prompt or "") + "\n\n")
        f.write("USER:\n" + (user_prompt or "") + "\n")
        f.write("="*80 + "\n\n")


def _qwen_json_call(step_name: str, system_prompt: str, user_prompt: str, raw_path: str, prompts_used_path: str,
                    error_path: str, temperature: float = 0.2, max_new_tokens: int = 2048) -> Tuple[Optional[Any], str]:
    """Call local Qwen3-VL text path and force JSON-only. Returns (parsed_json_or_none, raw_text).

    Reliability rules:
    - Always writes raw_path (plan_raw/shots_raw) even if parsing fails.
    - Writes prompts_used_path with system/user prompts.
    - Writes error_path with traceback on failure.
    - Robust JSON extraction: first parseable {...} or [...] block.
    """
    _append_prompt_used(prompts_used_path, step_name, system_prompt, user_prompt)
    raw_text = ""
    try:
        if not _HAVE_QWEN_TEXT or _qwen_generate_text is None:
            raise RuntimeError(f"Qwen text generator not available: {_QWEN_IMPORT_ERROR!r}")
        model_path = Path(_qwen_model_dir())
        if not (model_path.exists() and any(model_path.iterdir())):
            raise RuntimeError(f"Qwen3-VL model folder not found or empty: {model_path}")
        # Prevent model/transformers prints from spamming logs; capture for debug.
        cap_out = io.StringIO()
        cap_err = io.StringIO()
        with contextlib.redirect_stdout(cap_out), contextlib.redirect_stderr(cap_err):
            txt = _qwen_generate_text(
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                cancel_check=None,
            )  # type: ignore
        raw_text = (txt or "").strip()
        stderr_txt = (cap_err.getvalue() or "").strip()

        # Save raw (optionally append stderr without affecting JSON extraction)
        save_blob = raw_text
        if stderr_txt:
            save_blob = (save_blob + "\n\n[stderr]\n" + stderr_txt).strip()
        _safe_write_text(raw_path, save_blob + ("\n" if save_blob else ""))

        parsed = _extract_first_json(raw_text)
        if parsed is None:
            raise RuntimeError("Model response did not contain parseable JSON.")
        return parsed, raw_text

    except Exception:
        # Always write whatever we have + error traceback
        try:
            if raw_text:
                _safe_write_text(raw_path, raw_text + "\n")
        except Exception:
            pass
        _safe_write_text(error_path, traceback.format_exc())
        return None, raw_text




# -----------------------------
# Pipeline runner (placeholder)
# -----------------------------

class PipelineSignals(QObject):
    log = Signal(str)
    stage = Signal(str)
    progress = Signal(int)  # 0-100
    finished = Signal(dict)  # result payload
    failed = Signal(str)


class PipelineWorker(QThread):
    """
    Placeholder threaded runner: simulates the stages and emits logs.
    Replace stage bodies with real calls later.
    """

    def __init__(self, job: PlaceholdrJob, out_dir: str):
        super().__init__()
        self.job = job
        self.out_dir = out_dir
        self.signals = PipelineSignals()
        self._stop_requested = False
        self._last_pct = 0  # last emitted progress percent (for skip logs)


    def request_stop(self) -> None:
        self._stop_requested = True

    def _tick(self, msg: str, pct: int, sleep_s: float = 0.25) -> None:
        if self._stop_requested:
            raise RuntimeError("Cancelled by user.")
        self.signals.log.emit(msg)
        self._last_pct = int(pct)
        self.signals.progress.emit(pct)
        time.sleep(sleep_s)

    def run(self) -> None:
        try:
            os.makedirs(self.out_dir, exist_ok=True)

            # NOTE:
            # The Planner originally simulated the full pipeline with placeholder stage logs.
            # Plan + Shots are now real (Qwen3-VL JSON), so we drive progress from the
            # stepwise artifact pipeline below and avoid misleading "not wired" messages.

            self.signals.stage.emit("Starting")
            self.signals.log.emit(f"Job: {self.job.job_id}")
            self.signals.log.emit(f"Project root: {_root()}")
            self.signals.log.emit(f"Output directory: {self.out_dir}")
            self.signals.log.emit("----")
            self.signals.progress.emit(0)
            # -----------------------------
            # Stepwise, idempotent artifact pipeline (placeholder)
            # Each step writes its own output. Future wiring simply replaces a step body.
            # -----------------------------
            story_dir = os.path.join(self.out_dir, "story")
            prompts_dir = os.path.join(self.out_dir, "prompts")
            audio_dir = os.path.join(self.out_dir, "audio")
            images_dir = os.path.join(self.out_dir, "images")
            clips_dir = os.path.join(self.out_dir, "clips")
            final_dir = os.path.join(self.out_dir, "final")
            errors_dir = os.path.join(self.out_dir, "errors")

            for d in (story_dir, prompts_dir, audio_dir, images_dir, clips_dir, final_dir, errors_dir):
                os.makedirs(d, exist_ok=True)

            manifest_path = os.path.join(self.out_dir, "manifest.json")
            readme_path = os.path.join(self.out_dir, "README_PLACEHOLDER.txt")

            # Inputs fingerprint for this job
            input_fingerprint = _sha1_text(json.dumps({
                "prompt": self.job.prompt,
                "extra_info": self.job.extra_info,
                "duration": self.job.approx_duration_sec,
                "resolution": self.job.resolution_preset,
                "quality": self.job.quality_preset,
                "storytelling": self.job.storytelling,
                "music_background": self.job.music_background,
                "silent": self.job.silent,
                "storytelling_volume": self.job.storytelling_volume,
                "music_volume": self.job.music_volume,
                "image_model": self.job.encoding.get("image_model"),
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                "generation_profile": self.job.encoding.get("generation_profile"),
                "image_model": self.job.encoding.get("image_model"),
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
            }, sort_keys=True))

            manifest = _safe_read_json(manifest_path) or {}
            manifest.setdefault("job_id", self.job.job_id)
            manifest["project_root"] = str(_root())
            manifest["output_dir"] = self.out_dir
            manifest["input_fingerprint"] = input_fingerprint
            manifest.setdefault("steps", {})
            manifest.setdefault("paths", {})
            manifest.setdefault("settings", {})
            manifest["settings"].update({
                "duration_sec": self.job.approx_duration_sec,
                "resolution_preset": self.job.resolution_preset,
                "quality_preset": self.job.quality_preset,
                "storytelling": self.job.storytelling,
                "music_background": self.job.music_background,
                "silent": self.job.silent,
                "storytelling_volume": self.job.storytelling_volume,
                "music_volume": self.job.music_volume,

                "image_model": self.job.encoding.get("image_model"),
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                "videoclip_preset": self.job.encoding.get("videoclip_preset"),
                "generation_profile": self.job.encoding.get("generation_profile"),
            })
            manifest["paths"].update({
                "story_dir": story_dir,
                "prompts_dir": prompts_dir,
                "audio_dir": audio_dir,
                "images_dir": images_dir,
                "clips_dir": clips_dir,
                "final_dir": final_dir,
                "errors_dir": errors_dir,
            })

            def _set_step(name: str, status: str, note: str = "") -> None:
                cur = manifest["steps"].get(name) or {}
                cur.update({
                    "status": status,
                    "ts": time.time(),
                    "note": note,
                })
                manifest["steps"][name] = cur
                _safe_write_json(manifest_path, manifest)

            def _skip(name: str, why: str) -> None:
                self._tick(f"[SKIP] {name}: {why}", self._last_pct, 0.05)
                _set_step(name, "skipped", why)

            def _run(name: str, fn, pct: int) -> None:
                if self._stop_requested:
                    raise RuntimeError("Cancelled by user.")
                self.signals.stage.emit(name)
                self._tick(f"[RUN] {name}", pct, 0.05)
                _set_step(name, "running")
                try:
                    fn()
                    # Don't overwrite a step that marked itself as failed/stale.
                    cur = manifest["steps"].get(name) or {}
                    if cur.get("status") == "running":
                        _set_step(name, "done")
                except Exception as e:
                    err_path = os.path.join(errors_dir, f"{name.replace(' ', '_').lower()}.txt")
                    _safe_write_text(err_path, str(e))
                    _set_step(name, "failed", str(e))
                    raise

            # Step A: Plan
            plan_path = os.path.join(story_dir, "plan.json")
            plan_raw_path = os.path.join(story_dir, "plan_raw.txt")
            qwen_prompts_used = os.path.join(story_dir, "qwen_prompts_used.txt")
            qwen_plan_err = os.path.join(errors_dir, "qwen_plan_error.txt")

            # Resolve generation profile (proxy targets)
            gen_profile = self.job.encoding.get("generation_profile")
            if not isinstance(gen_profile, dict):
                gen_profile = _resolve_generation_profile(
                    self.job.encoding.get("video_model") or "",
                    self.job.encoding.get("gen_quality_preset") or "",
                )
                self.job.encoding["generation_profile"] = gen_profile

            plan_fingerprint = _sha1_text(json.dumps({
                "prompt": self.job.prompt,
                "extra_info": self.job.extra_info,
                "duration_sec": self.job.approx_duration_sec,
                "audio": {
                    "storytelling": self.job.storytelling,
                    "music_background": self.job.music_background,
                    "silent": self.job.silent,
                    "storytelling_volume": self.job.storytelling_volume,
                    "music_volume": self.job.music_volume,
                },
                "models": {
                    "image_model": self.job.encoding.get("image_model"),
                    "video_model": self.job.encoding.get("video_model"),
                    "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                },
                "generation_profile": gen_profile,
                "final_export": {
                    "resolution_preset": self.job.resolution_preset,
                    "quality_preset": self.job.quality_preset,
                    "encode": self.job.encoding,
                },
            }, sort_keys=True))

            def step_plan() -> None:
                # Determine audio mode for story
                audio_mode = ("silent" if self.job.silent else
                              ("both" if (self.job.storytelling and self.job.music_background) else
                               ("storytelling" if self.job.storytelling else
                                ("music" if self.job.music_background else "none"))))

                # Force JSON-only plan from Qwen
                sys_p = (
                    "You are a planning assistant for an offline video generator. "
                    "Return ONLY valid JSON. No markdown, no commentary, no code fences. "
                    "The JSON must be a single object."
                )
                user_p = (
                    "Create a concise but useful story plan and constraints for this video project.\n"
                    "Requirements:\n"
                    "- Output MUST be JSON only.\n"
                    "- Include: title, logline, setting, characters (list), tone, continuity_rules (list), beats (list).\n"
                    "- Keep it short and actionable for prompt generation.\n\n"
                    f"PROMPT: {self.job.prompt}\n"
                    + (f"EXTRA_INFO: {self.job.extra_info}\n" if (self.job.extra_info or '').strip() else "")
                    + f"TARGET_DURATION_SEC: {self.job.approx_duration_sec}\n"
                    + f"AUDIO_MODE: {audio_mode}\n"
                    + f"VIDEO_MODEL: {self.job.encoding.get('video_model')}\n"
                    + f"GEN_QUALITY: {self.job.encoding.get('gen_quality_preset')}\n"
                    + f"GEN_PROFILE: {json.dumps(gen_profile, ensure_ascii=False)}\n"
                )

                parsed, raw = _qwen_json_call(
                    step_name="Plan (Qwen3-VL JSON)",
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    raw_path=plan_raw_path,
                    prompts_used_path=qwen_prompts_used,
                    error_path=qwen_plan_err,
                    temperature=0.2,
                    max_new_tokens=1800,
                )

                if not isinstance(parsed, dict):
                    # Fallback placeholder, but mark failed and keep debug files
                    plan = {
                        "title": f"Planner plan {self.job.job_id}",
                        "prompt": self.job.prompt,
                        "extra_info": self.job.extra_info,
                        "logline": self.job.prompt,
                        "setting": "",
                        "characters": [],
                        "tone": "",
                        "continuity_rules": [
                            "Keep characters, outfits, and key props consistent across shots.",
                            "Keep the main location style consistent unless the shot says otherwise.",
                        ],
                        "beats": [
                            {"index": 1, "purpose": "setup", "moment": "Introduce the scene and main subject."},
                            {"index": 2, "purpose": "escalation", "moment": "Increase energy and visual novelty."},
                            {"index": 3, "purpose": "payoff", "moment": "Deliver a satisfying highlight moment."},
                        ],
                    }
                    _safe_write_json(plan_path, plan)
                    manifest["paths"]["plan_json"] = plan_path
                    # Record debug paths + fingerprint and mark as failed
                    manifest["paths"]["plan_raw_txt"] = plan_raw_path
                    manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                    manifest["paths"]["qwen_plan_error_txt"] = qwen_plan_err
                    srec = manifest["steps"].get("Plan (story + constraints)") or {}
                    srec.update({
                        "status": "failed",
                        "fingerprint": plan_fingerprint,
                        "debug": {"raw": plan_raw_path, "prompts_used": qwen_prompts_used, **({"error": qwen_plan_err} if _file_ok(qwen_plan_err, 1) else {})},
                        "note": "Qwen JSON failed; wrote placeholder plan.json",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Plan (story + constraints)"] = srec
                    _safe_write_json(manifest_path, manifest)
                    return

                # Merge resolved settings into plan
                parsed.setdefault("title", f"Planner plan {self.job.job_id}")
                parsed["prompt"] = self.job.prompt
                parsed["extra_info"] = self.job.extra_info
                parsed["duration_sec"] = int(self.job.approx_duration_sec)
                parsed["audio_mode"] = audio_mode
                parsed["generation"] = {
                    "video_model": self.job.encoding.get("video_model"),
                    "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                    "profile": gen_profile,
                }
                parsed["final_export"] = {
                    "resolution_preset": self.job.resolution_preset,
                    "quality_preset": self.job.quality_preset,
                    "encode": {
                        "mode": self.job.encoding.get("mode"),
                        "crf": self.job.encoding.get("crf"),
                        "bitrate_kbps": self.job.encoding.get("bitrate_kbps"),
                        "note": self.job.encoding.get("note"),
                    },
                }
                parsed["style_notes"] = {
                    "image_model": self.job.encoding.get("image_model"),
                    "video_model": self.job.encoding.get("video_model"),
                    "videoclip_preset": self.job.encoding.get("videoclip_preset"),
                }
                parsed["planner_plan_fingerprint"] = plan_fingerprint

                _safe_write_json(plan_path, parsed)
                manifest["paths"]["plan_json"] = plan_path
                manifest["paths"]["plan_raw_txt"] = plan_raw_path
                manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                # Only store error path if an error file exists
                if _file_ok(qwen_plan_err, 1):
                    manifest["paths"]["qwen_plan_error_txt"] = qwen_plan_err
                else:
                    manifest["paths"].pop("qwen_plan_error_txt", None)

                srec = manifest["steps"].get("Plan (story + constraints)") or {}
                srec.update({
                    "status": "done",
                    "fingerprint": plan_fingerprint,
                    "debug": {"raw": plan_raw_path, "prompts_used": qwen_prompts_used, **({"error": qwen_plan_err} if _file_ok(qwen_plan_err, 1) else {})},
                    "note": "Generated with Qwen (JSON enforced).",
                    "ts": time.time(),
                })
                manifest["steps"]["Plan (story + constraints)"] = srec
                _safe_write_json(manifest_path, manifest)

            # Idempotency: skip only if plan exists AND fingerprint matches AND last status was done
            plan_prev = (manifest.get("steps") or {}).get("Plan (story + constraints)") or {}
            if _file_ok(plan_path, 10) and plan_prev.get("fingerprint") == plan_fingerprint and plan_prev.get("status") == "done":
                _skip("Plan (story + constraints)", "plan.json up-to-date (fingerprint match)")
                plan_changed = False
            else:
                _run("Plan (story + constraints)", step_plan, 12)
                plan_changed = True

# Step B: Shots
            shots_path = os.path.join(story_dir, "shots.json")
            shots_raw_path = os.path.join(story_dir, "shots_raw.txt")
            qwen_shots_err = os.path.join(errors_dir, "qwen_shots_error.txt")

            # Shots fingerprint depends on plan fingerprint + generation profile
            shots_fingerprint = _sha1_text(json.dumps({
                "plan_fingerprint": plan_fingerprint,
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                "generation_profile": gen_profile,
            }, sort_keys=True))

            def step_shots() -> None:
                # Load plan for context (best effort)
                plan_obj = _safe_read_json(plan_path) or {}
                # Choose approximate number of shots based on desired duration
                # (avg duration chosen from generation preset; stable per job)
                avg_sec = float((gen_profile.get("min_sec", 2.5) + gen_profile.get("max_sec", 5.0)) / 2.0)
                n_shots = max(6, min(80, int(max(1, self.job.approx_duration_sec) / max(1.5, avg_sec))))

                sys_p = (
                    "You are a shot-list generator for an offline video pipeline. "
                    "Return ONLY valid JSON. No markdown, no commentary, no code fences. "
                    "The JSON must be a single ARRAY (top-level list) of shot objects. "
                    "Each shot MUST include fields: id, seed, camera, mood, lighting, notes."
                )
                user_p = (
                    "Generate a concise seeded shot list for the plan below.\n"
                    "Rules:\n"
                    "- Output MUST be JSON only.\n"
                    "- Output a top-level JSON array of {id, seed, camera, mood, lighting, notes} objects.\n"
                    "- Use ids like S01, S02, ... up to the requested count.\n"
                    "- Keep each shot to ONE clear action and ONE subject.\n"
                    "- Keep continuity across shots.\n\n"
                    f"REQUESTED_SHOTS: {n_shots}\n"
                    f"GEN_PROFILE: {json.dumps(gen_profile, ensure_ascii=False)}\n"
                    "PLAN_JSON:\n"
                    + json.dumps(plan_obj, ensure_ascii=False)
                )

                parsed, raw = _qwen_json_call(
                    step_name="Shots (Qwen3-VL JSON)",
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    raw_path=shots_raw_path,
                    prompts_used_path=qwen_prompts_used,
                    error_path=qwen_shots_err,
                    temperature=0.25,
                    max_new_tokens=2400,
                )

                shots_list: List[Dict[str, Any]] = []

                if isinstance(parsed, list):
                    shots_list = [x for x in parsed if isinstance(x, dict)]
                elif isinstance(parsed, dict):
                    v = parsed.get("shots")
                    if isinstance(v, list):
                        shots_list = [x for x in v if isinstance(x, dict)]
                    else:
                        # try treat dict itself as shot map (unlikely)
                        pass

                if not shots_list:
                    # Fallback placeholder list
                    shots_list = []
                    for i in range(1, n_shots + 1):
                        sid = f"S{i:02d}"
                        shots_list.append({
                            "id": sid,
                            "seed": f"{self.job.prompt} — moment {i} of {n_shots}",
                            "camera": "medium shot",
                            "mood": "neutral",
                            "lighting": "cinematic lighting",
                            "notes": "One clear action, one subject. Keep continuity with prior shots.",
                        })
                    # mark failed but continue
                    manifest["paths"]["shots_raw_txt"] = shots_raw_path
                    manifest["paths"]["qwen_shots_error_txt"] = qwen_shots_err

                    srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                    srec.update({
                        "status": "failed",
                        "fingerprint": shots_fingerprint,
                        "debug": {"raw": shots_raw_path, "prompts_used": qwen_prompts_used, **({"error": qwen_shots_err} if _file_ok(qwen_shots_err, 1) else {})},
                        "note": "Qwen JSON failed; wrote placeholder shots.json",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Shots (seeded shot list)"] = srec

                # Normalize + add generation intent fields
                out_shots: List[Dict[str, Any]] = []
                for idx, sh in enumerate(shots_list, start=1):
                    sid = sh.get("id") or f"S{idx:02d}"
                    seed = sh.get("seed") or f"{self.job.prompt} — moment {idx}"
                    cam = sh.get("camera") or "medium shot"
                    mood = sh.get("mood") or "neutral"
                    light = sh.get("lighting") or "cinematic lighting"
                    notes = sh.get("notes") or "One clear action, one subject. Keep continuity."

                    # deterministic duration per shot within preset range
                    dur = _stable_uniform(str(seed), float(gen_profile.get("min_sec", 2.5)), float(gen_profile.get("max_sec", 5.0)))
                    dur = round(dur, 2)

                    out_shots.append({
                        "id": str(sid),
                        "seed": str(seed),
                        "seed_int": int(_sha1_text(str(seed))[:8], 16) % 2147483647,
                        "camera": str(cam),
                        "mood": str(mood),
                        "lighting": str(light),
                        "notes": str(notes),
                        # placeholders for downstream compilers
                        "duration_sec": dur,
                        "gen_fps": int(gen_profile.get("fps", 20)),
                        "gen_res": str(gen_profile.get("res", "384p")),
                        "steps": int(gen_profile.get("steps", 9)),
                        "video_model_key": str(gen_profile.get("model", "hunyuan")),
                    })

                _safe_write_json(shots_path, out_shots)
                manifest["paths"]["shots_json"] = shots_path
                manifest["paths"]["shots_raw_txt"] = shots_raw_path
                manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                # Only store error path if an error file exists
                if _file_ok(qwen_shots_err, 1):
                    manifest["paths"]["qwen_shots_error_txt"] = qwen_shots_err
                else:
                    manifest["paths"].pop("qwen_shots_error_txt", None)
                # Only set n_shots if we have list
                manifest["settings"]["n_shots"] = len(out_shots)

                # If step not already marked failed above, mark done now
                srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                if srec.get("status") != "failed":
                    srec.update({
                        "status": "done",
                        "fingerprint": shots_fingerprint,
                        "debug": {"raw": shots_raw_path, "prompts_used": qwen_prompts_used, **({"error": qwen_shots_err} if _file_ok(qwen_shots_err, 1) else {})},
                        "note": "Generated with Qwen (JSON enforced).",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Shots (seeded shot list)"] = srec

                _safe_write_json(manifest_path, manifest)

            shots_prev = (manifest.get("steps") or {}).get("Shots (seeded shot list)") or {}

            # Shots depend on plan: if plan changed, always regenerate shots
            if not plan_changed and _file_ok(shots_path, 10) and shots_prev.get("fingerprint") == shots_fingerprint and shots_prev.get("status") == "done":
                _skip("Shots (seeded shot list)", "shots.json up-to-date (fingerprint match)")
            else:
                if plan_changed and _file_ok(shots_path, 10):
                    # Mark stale in manifest
                    srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                    srec.update({"status": "stale", "note": "Plan changed; regenerating shots", "ts": time.time()})
                    manifest["steps"]["Shots (seeded shot list)"] = srec
                    _safe_write_json(manifest_path, manifest)
                _run("Shots (seeded shot list)", step_shots, 28)

# Step C: Image prompts
            image_prompts_path = os.path.join(prompts_dir, "image_prompts.txt")
            def step_image_prompts() -> None:
                shots = _read_shots_list(shots_path)
                base_style = "Cinematic, detailed, consistent character design, clean composition."
                if self.job.extra_info.strip():
                    base_style += f" Extra notes: {self.job.extra_info.strip()}"
                out = []
                for sh in shots:
                    seed = sh.get("seed", "")
                    camera = sh.get("camera", "medium shot")
                    lighting = sh.get("lighting", "cinematic lighting")
                    mood = sh.get("mood", "neutral")
                    out.append(
                        f"{seed}. {camera}. {lighting}. Mood: {mood}. "
                        f"High detail, sharp focus. {base_style}"
                    )
                _safe_write_text(image_prompts_path, "\n\n".join(out).strip() + "\n")
                manifest["paths"]["image_prompts_txt"] = image_prompts_path
                _safe_write_json(manifest_path, manifest)

            if _file_ok(image_prompts_path, 10):
                _skip("Image prompts (from shots)", "image_prompts.txt already exists")
            else:
                _run("Image prompts (from shots)", step_image_prompts, 44)

            # Step D: Transcript (optional)
            transcript_path = os.path.join(audio_dir, "transcript.txt")
            def step_transcript() -> None:
                if self.job.storytelling and not self.job.silent:
                    _safe_write_text(
                        transcript_path,
                        "PLACEHOLDER TRANSCRIPT (will be generated by Whisper/Qwen later)\n\n"
                        f"Prompt: {self.job.prompt}\n"
                        + (f"Extra: {self.job.extra_info.strip()}\n" if self.job.extra_info.strip() else "")
                    )
                else:
                    _safe_write_text(transcript_path, "SKIPPED (storytelling off or silent)\n")
                manifest["paths"]["transcript_txt"] = transcript_path
                _safe_write_json(manifest_path, manifest)

            if _file_ok(transcript_path, 2):
                _skip("Transcript (optional)", "transcript.txt already exists")
            else:
                _run("Transcript (optional)", step_transcript, 55)

            # Step E: I2V prompts
            i2v_prompts_path = os.path.join(prompts_dir, "i2v_prompts.txt")
            def step_i2v_prompts() -> None:
                shots = _read_shots_list(shots_path)
                lines = []
                for sh in shots:
                    sid = sh.get("id", "S??")
                    lines.append(
                        f"{sid}: Slow camera move, subtle motion, keep subject stable. "
                        f"Match the image exactly; add gentle parallax and atmospheric movement."
                    )
                _safe_write_text(i2v_prompts_path, "\n".join(lines).strip() + "\n")
                manifest["paths"]["i2v_prompts_txt"] = i2v_prompts_path
                _safe_write_json(manifest_path, manifest)

            if _file_ok(i2v_prompts_path, 10):
                _skip("I2V prompts (from shots)", "i2v_prompts.txt already exists")
            else:
                _run("I2V prompts (from shots)", step_i2v_prompts, 70)

            # Step F: Final placeholder video
            final_video = os.path.join(final_dir, f"{self.job.job_id}_final.mp4")
            def step_final() -> None:
                with open(final_video, "wb") as f:
                    f.write(b"")
                manifest["final_video"] = final_video
                manifest["paths"]["final_video"] = final_video
                _safe_write_json(manifest_path, manifest)

            if os.path.isfile(final_video):
                _skip("Finalize output", "final mp4 already exists")
            else:
                _run("Finalize output", step_final, 100)

            # README (always refresh, tiny)
            _safe_write_text(
                readme_path,
                "This job folder contains placeholder artifacts for wiring later:\n"
                "- story/plan.json\n"
                "- story/shots.json\n"
                "- prompts/image_prompts.txt\n"
                "- prompts/i2v_prompts.txt\n"
                "- audio/transcript.txt\n"
                "- final/<jobid>_final.mp4 (empty placeholder)\n"
                "- manifest.json (paths + step statuses)\n"
            )

            self._tick(f"Wrote/updated manifest: {manifest_path}", 100, 0.05)
            self._tick(f"Final video placeholder: {final_video}", 100, 0.10)

            result = {
                "job_id": self.job.job_id,
                "output_dir": self.out_dir,
                "final_video": final_video,
                "job_config": asdict(self.job),
            }
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.failed.emit(str(e))


# -----------------------------
# UI
# -----------------------------

class PlaceholdrPane(QWidget):
    """
    A wire-ready pane for "Prompt -> Finished Video".
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[PipelineWorker] = None
        self._last_result: Optional[dict] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Planner — Prompt → Finished Video (Placeholder)")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)

        subtitle = QLabel("UI + pipeline skeleton. Everything is stubbed, ready to wire to your models and Videoclip Creator later.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("opacity: 0.85;")

        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: Step 1 options
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(self._build_step1_group())
        left_layout.addWidget(self._build_optional_group())
        left_layout.addStretch(1)

        # Right: Run + logs
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        right_layout.addWidget(self._build_run_group())
        right_layout.addWidget(self._build_log_group())

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

        self._sync_toggle_visibility()
        self._sync_silent_logic()

    # -------------------------
    # Step 1 UI
    # -------------------------

    def _build_step1_group(self) -> QGroupBox:
        box = QGroupBox("Step 1 — What do you want to make?")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Prompt
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Example: aliens go out to a nightclub and have fun")
        self.prompt_edit.setFixedHeight(70)

        lay.addWidget(QLabel("Prompt"))
        lay.addWidget(self.prompt_edit)

        # Toggles row
        toggles = QHBoxLayout()
        toggles.setSpacing(12)

        self.chk_story = QCheckBox("Story telling")
        self.chk_music = QCheckBox("Music background")
        self.chk_silent = QCheckBox("Silent")

        self.chk_story.toggled.connect(self._sync_toggle_visibility)
        self.chk_music.toggled.connect(self._sync_toggle_visibility)
        self.chk_silent.toggled.connect(self._sync_silent_logic)

        toggles.addWidget(self.chk_story)
        toggles.addWidget(self.chk_music)
        toggles.addWidget(self.chk_silent)
        toggles.addStretch(1)

        lay.addLayout(toggles)

        # Volume controls
        self.story_vol_row = QWidget()
        svl = QHBoxLayout(self.story_vol_row)
        svl.setContentsMargins(0, 0, 0, 0)
        svl.setSpacing(10)
        svl.addWidget(QLabel("Story telling volume"))
        self.sld_story_vol = QSlider(Qt.Horizontal)
        self.sld_story_vol.setRange(0, 100)
        self.sld_story_vol.setValue(100)  # default 100%
        self.lbl_story_vol = QLabel("100%")
        self.lbl_story_vol.setMinimumWidth(48)
        self.sld_story_vol.valueChanged.connect(lambda v: self.lbl_story_vol.setText(f"{v}%"))
        svl.addWidget(self.sld_story_vol, 1)
        svl.addWidget(self.lbl_story_vol)

        self.music_vol_row = QWidget()
        mvl = QHBoxLayout(self.music_vol_row)
        mvl.setContentsMargins(0, 0, 0, 0)
        mvl.setSpacing(10)
        mvl.addWidget(QLabel("Music background volume"))
        self.sld_music_vol = QSlider(Qt.Horizontal)
        self.sld_music_vol.setRange(0, 100)
        self.sld_music_vol.setValue(25)  # default 25%
        self.lbl_music_vol = QLabel("25%")
        self.lbl_music_vol.setMinimumWidth(48)
        self.sld_music_vol.valueChanged.connect(lambda v: self.lbl_music_vol.setText(f"{v}%"))
        mvl.addWidget(self.sld_music_vol, 1)
        mvl.addWidget(self.lbl_music_vol)

        lay.addWidget(self.story_vol_row)
        lay.addWidget(self.music_vol_row)

        # Duration slider (5s - 10 min)
        duration_row = QWidget()
        dr = QHBoxLayout(duration_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.setSpacing(10)
        dr.addWidget(QLabel("Approx duration"))

        self.sld_duration = QSlider(Qt.Horizontal)
        self.sld_duration.setRange(5, 600)
        self.sld_duration.setValue(30)
        self.lbl_duration = QLabel(_duration_label(30))
        self.lbl_duration.setMinimumWidth(70)
        self.sld_duration.valueChanged.connect(lambda v: self.lbl_duration.setText(_duration_label(v)))

        dr.addWidget(self.sld_duration, 1)
        dr.addWidget(self.lbl_duration)
        lay.addWidget(duration_row)

        # Resolution + quality grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Resolution & aspect"), 0, 0)
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems([
            "480p Landscape (16:9)",
            "480p Portrait (9:16)",
            "720p Landscape (16:9)",
            "720p Portrait (9:16)",
            "1080p Landscape (16:9)",
            "1080p Portrait (9:16)",
        ])
        self.cmb_resolution.setCurrentText("720p Landscape (16:9)")
        grid.addWidget(self.cmb_resolution, 0, 1)

        grid.addWidget(QLabel("Quality"), 1, 0)
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["Low", "Medium", "High"])
        self.cmb_quality.setCurrentText("Medium")
        grid.addWidget(self.cmb_quality, 1, 1)

        self.lbl_quality_hint = QLabel("Medium default: CRF 18 (~3000 kbps target).")
        self.lbl_quality_hint.setStyleSheet("opacity: 0.8;")
        self.cmb_quality.currentTextChanged.connect(self._sync_quality_hint)
        grid.addWidget(self.lbl_quality_hint, 2, 0, 1, 2)

        lay.addLayout(grid)

        # End of user work note
        end_note = QLabel("When you press Generate, the app will run the full pipeline (story → images → video clips → assembly).")
        end_note.setWordWrap(True)
        end_note.setStyleSheet("opacity: 0.9;")
        lay.addWidget(end_note)

        return box

    def _build_optional_group(self) -> QGroupBox:
        box = QGroupBox("Optional — Extra info and future inputs")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        self.extra_info = QTextEdit()
        self.extra_info.setPlaceholderText(
            "Optional: style, colors, background info, mood (happy/sad), cinematic notes, etc.\n"
            "Example: neon cyberpunk club, playful mood, vibrant colors, fast cuts."
        )
        self.extra_info.setFixedHeight(80)

        lay.addWidget(QLabel("Extra info (free text)"))
        lay.addWidget(self.extra_info)

        # Attachment quick-add row
        attach_row = QHBoxLayout()
        attach_row.setSpacing(8)

        self.btn_add_json = QPushButton("Add JSON")
        self.btn_add_images = QPushButton("Add Images")
        self.btn_add_videos = QPushButton("Add Video")
        self.btn_add_text = QPushButton("Add Text")
        self.btn_add_transcript = QPushButton("Add Transcript")

        self.btn_add_json.clicked.connect(lambda: self._add_files("json"))
        self.btn_add_images.clicked.connect(lambda: self._add_files("images"))
        self.btn_add_videos.clicked.connect(lambda: self._add_files("videos"))
        self.btn_add_text.clicked.connect(lambda: self._add_files("text"))
        self.btn_add_transcript.clicked.connect(lambda: self._add_files("transcripts"))

        attach_row.addWidget(self.btn_add_json)
        attach_row.addWidget(self.btn_add_images)
        attach_row.addWidget(self.btn_add_videos)
        attach_row.addWidget(self.btn_add_text)
        attach_row.addWidget(self.btn_add_transcript)
        attach_row.addStretch(1)

        lay.addLayout(attach_row)

        self.attach_list = QListWidget()
        self.attach_list.setMinimumHeight(110)
        self.attach_list.setToolTip("Optional inputs that can later be used for story guidance / references / editing.")
        lay.addWidget(self.attach_list)

        # Clear/remove actions
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.btn_remove_selected = QPushButton("Remove selected")
        self.btn_clear_all = QPushButton("Clear all")
        self.btn_remove_selected.clicked.connect(self._remove_selected_attachment)
        self.btn_clear_all.clicked.connect(self.attach_list.clear)

        row2.addWidget(self.btn_remove_selected)
        row2.addWidget(self.btn_clear_all)
        row2.addStretch(1)
        lay.addLayout(row2)

        return box

    # -------------------------
    # Run + logs UI
    # -------------------------

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Model selection (placeholder)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Image model"), 0, 0)
        self.cmb_image_model = QComboBox()
        self.cmb_image_model.addItems([
            "Auto (later)",
            "Z-Image Turbo (planned)",
            "Qwen Image 2512 (planned)",
            "SDXL (planned)",
        ])
        self.cmb_image_model.setCurrentIndex(0)
        grid.addWidget(self.cmb_image_model, 0, 1)

        grid.addWidget(QLabel("Video model"), 1, 0)
        self.cmb_video_model = QComboBox()
        self.cmb_video_model.addItems([
            "Auto (later)",
            "WAN 2.2 (planned)",
            "HunyuanVideo 1.5 (planned)",
            "Qwen 2.2 5B (planned)",
        ])
        self.cmb_video_model.setCurrentIndex(0)
        grid.addWidget(self.cmb_video_model, 1, 1)

        grid.addWidget(QLabel("Generation quality"), 2, 0)
        self.cmb_gen_quality = QComboBox()
        # populated dynamically based on video model selection
        grid.addWidget(self.cmb_gen_quality, 2, 1)

        def _refresh_gen_quality() -> None:
            vm = (self.cmb_video_model.currentText() or "").lower()
            cur = (self.cmb_gen_quality.currentText() or "").strip()
            self.cmb_gen_quality.blockSignals(True)
            self.cmb_gen_quality.clear()
            if "wan" in vm and "2.2" in vm:
                self.cmb_gen_quality.addItems(["Normal (default)", "High"])
                if cur.lower().startswith("high"):
                    self.cmb_gen_quality.setCurrentIndex(1)
                else:
                    self.cmb_gen_quality.setCurrentIndex(0)
            else:
                # hunyuan + fallback
                self.cmb_gen_quality.addItems(["Low", "Medium (default)", "High"])
                if cur.lower().startswith("low"):
                    self.cmb_gen_quality.setCurrentIndex(0)
                elif cur.lower().startswith("high"):
                    self.cmb_gen_quality.setCurrentIndex(2)
                else:
                    self.cmb_gen_quality.setCurrentIndex(1)
            self.cmb_gen_quality.blockSignals(False)

        self.cmb_video_model.currentIndexChanged.connect(lambda _=None: _refresh_gen_quality())
        _refresh_gen_quality()

        grid.addWidget(QLabel("Videoclip Creator preset"), 3, 0)
        self.cmb_videoclip_preset = QComboBox()
        self.cmb_videoclip_preset.addItems([
            "Preset A (placeholder)",
            "Preset B (placeholder)",
            "Preset C (placeholder)",
        ])
        self.cmb_videoclip_preset.setCurrentIndex(0)
        grid.addWidget(self.cmb_videoclip_preset, 3, 1)

        lay.addLayout(grid)

        # Output folder
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("Output folder (optional). Defaults to ./output/placeholdr/")
        self.btn_browse_out = QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_out_dir)

        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(self.btn_browse_out)
        lay.addLayout(out_row)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_generate = QPushButton("Generate")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_export_job = QPushButton("Export job JSON")
        self.btn_open_output = QPushButton("Open output folder")

        self.btn_cancel.setEnabled(False)
        self.btn_open_output.setEnabled(False)

        self.btn_generate.clicked.connect(self._start_pipeline)
        self.btn_cancel.clicked.connect(self._cancel_pipeline)
        self.btn_export_job.clicked.connect(self._export_job_json)
        self.btn_open_output.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_export_job)
        btn_row.addWidget(self.btn_open_output)

        lay.addLayout(btn_row)

        # Progress
        self.lbl_stage = QLabel("Stage: —")
        self.lbl_stage.setStyleSheet("opacity: 0.9;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        lay.addWidget(self.lbl_stage)
        lay.addWidget(self.progress)

        # Dry-run hint
        hint = QLabel("Placeholder runner simulates outputs and writes an empty *_final.mp4 file in the chosen output folder.")
        hint.setWordWrap(True)
        hint.setStyleSheet("opacity: 0.8;")
        lay.addWidget(hint)

        return box

    def _build_log_group(self) -> QGroupBox:
        box = QGroupBox("Logs")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline logs will appear here…")
        self.log.setMinimumHeight(260)

        lay.addWidget(self.log)

        # Log actions
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_copy_log = QPushButton("Copy log")
        self.btn_clear_log.clicked.connect(self.log.clear)
        self.btn_copy_log.clicked.connect(self._copy_log)

        row.addWidget(self.btn_clear_log)
        row.addWidget(self.btn_copy_log)
        row.addStretch(1)

        lay.addLayout(row)
        return box

    # -------------------------
    # Behavior / wiring
    # -------------------------

    def _sync_quality_hint(self) -> None:
        q = self.cmb_quality.currentText()
        d = _quality_defaults(q)
        self.lbl_quality_hint.setText(
            f"{q} preset: {d.get('note')} — CRF {d.get('crf')} (target ~{d.get('bitrate_kbps')} kbps)."
        )

    def _sync_toggle_visibility(self) -> None:
        self.story_vol_row.setVisible(self.chk_story.isChecked())
        self.music_vol_row.setVisible(self.chk_music.isChecked())

    def _sync_silent_logic(self) -> None:
        silent = self.chk_silent.isChecked()
        if silent:
            # If silent: it doesn't make sense to keep audio toggles on.
            self.chk_story.setChecked(False)
            self.chk_music.setChecked(False)
            self.chk_story.setEnabled(False)
            self.chk_music.setEnabled(False)
        else:
            self.chk_story.setEnabled(True)
            self.chk_music.setEnabled(True)
        self._sync_toggle_visibility()

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.out_dir_edit.setText(d)

    def _copy_log(self) -> None:
        QApplication.clipboard().setText(self.log.toPlainText())

    def _add_files(self, kind: str) -> None:
        kind = kind.lower().strip()
        caption = f"Select {kind} files"
        filters = "All files (*.*)"

        if kind == "images":
            filters = "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        elif kind == "videos":
            filters = "Video (*.mp4 *.mov *.mkv *.webm *.avi);;All files (*.*)"
        elif kind == "json":
            filters = "JSON (*.json);;All files (*.*)"
        elif kind == "text":
            filters = "Text (*.txt *.md);;All files (*.*)"
        elif kind == "transcripts":
            filters = "Transcript (*.srt *.vtt *.txt *.json);;All files (*.*)"

        files, _ = QFileDialog.getOpenFileNames(self, caption, "", filters)
        if not files:
            return

        for p in files:
            item = QListWidgetItem(f"{kind}: {p}")
            item.setData(Qt.UserRole, {"kind": kind, "path": p})
            self.attach_list.addItem(item)

    def _remove_selected_attachment(self) -> None:
        for item in self.attach_list.selectedItems():
            row = self.attach_list.row(item)
            self.attach_list.takeItem(row)

    def _collect_attachments(self) -> Dict[str, List[str]]:
        out = {"json": [], "images": [], "videos": [], "text": [], "transcripts": []}
        for i in range(self.attach_list.count()):
            item = self.attach_list.item(i)
            data = item.data(Qt.UserRole) or {}
            k = data.get("kind", "")
            p = data.get("path", "")
            if k in out and p:
                out[k].append(p)
        return out

    def _build_job(self) -> PlaceholdrJob:
        prompt = (self.prompt_edit.toPlainText() or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        storytelling = self.chk_story.isChecked()
        music = self.chk_music.isChecked()
        silent = self.chk_silent.isChecked()

        q = self.cmb_quality.currentText()
        enc = _quality_defaults(q)
        enc.update({
            "resolution_preset": self.cmb_resolution.currentText(),
            "quality_preset": q,
            "image_model": self.cmb_image_model.currentText(),
            "video_model": self.cmb_video_model.currentText(),
            "videoclip_preset": self.cmb_videoclip_preset.currentText(),
            "gen_quality_preset": (self.cmb_gen_quality.currentText() if hasattr(self, "cmb_gen_quality") else ""),
        })
        # Resolve generation profile (proxy targets) now so fingerprints stay stable
        try:
            enc["generation_profile"] = _resolve_generation_profile(enc.get("video_model",""), enc.get("gen_quality_preset",""))
        except Exception:
            enc["generation_profile"] = {}


        job = PlaceholdrJob(
            job_id=str(uuid.uuid4())[:8],
            created_at=time.time(),
            prompt=prompt,
            storytelling=storytelling,
            music_background=music,
            silent=silent,
            storytelling_volume=int(self.sld_story_vol.value()),
            music_volume=int(self.sld_music_vol.value()),
            approx_duration_sec=int(self.sld_duration.value()),
            resolution_preset=self.cmb_resolution.currentText(),
            quality_preset=q,
            extra_info=(self.extra_info.toPlainText() or "").strip(),
            attachments=self._collect_attachments(),
            encoding=enc,
        )
        return job

    def _default_out_dir(self) -> str:
        # Default output path at project root
        return _default_output_base()

    def _append_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {line}")

    def _set_running(self, running: bool) -> None:
        self.btn_generate.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_export_job.setEnabled(not running)
        self.btn_open_output.setEnabled(not running and bool(self._last_result))
        self.progress.setEnabled(True)

    @Slot()
    def _start_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        try:
            job = self._build_job()
        except Exception as e:
            QMessageBox.warning(self, "Cannot start", str(e))
            return

        out_dir = (self.out_dir_edit.text() or "").strip() or self._default_out_dir()
        out_dir = _abspath_from_root(out_dir)
        out_dir = os.path.join(out_dir, job.job_id)

        # Reset UI
        self.progress.setValue(0)
        self.lbl_stage.setText("Stage: Starting")
        self.log.clear()
        self._last_result = None
        self.btn_open_output.setEnabled(False)

        self._set_running(True)

        self._worker = PipelineWorker(job, out_dir)
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.stage.connect(lambda s: self.lbl_stage.setText(f"Stage: {s}"))
        self._worker.signals.progress.connect(self.progress.setValue)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.failed.connect(self._on_failed)

        self._append_log("Starting pipeline…")
        self._worker.start()

    @Slot()
    def _cancel_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            self._append_log("Cancel requested…")
            self._worker.request_stop()

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        self._last_result = result
        self._append_log("----")
        self._append_log("Done.")
        self._append_log(f"Final video: {result.get('final_video')}")
        self._set_running(False)
        self.btn_open_output.setEnabled(True)

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self._append_log("----")
        self._append_log(f"FAILED: {err}")
        self._set_running(False)
        QMessageBox.warning(self, "Pipeline failed", err)

    @Slot()
    def _export_job_json(self) -> None:
        try:
            job = self._build_job()
        except Exception as e:
            QMessageBox.warning(self, "Cannot export", str(e))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save job JSON", f"placeholdr_job_{job.job_id}.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(job.to_json())
            QMessageBox.information(self, "Saved", f"Job JSON saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    @Slot()
    def _open_output_folder(self) -> None:
        if not self._last_result:
            return
        out_dir = self._last_result.get("output_dir", "")
        if not out_dir:
            return
        # Cross-platform open folder
        try:
            if os.name == "nt":
                os.startfile(out_dir)  # type: ignore[attr-defined]
            elif os.name == "posix":
                # macOS or Linux
                import subprocess
                subprocess.Popen(["xdg-open", out_dir])
            else:
                QMessageBox.information(self, "Output folder", out_dir)
        except Exception:
            QMessageBox.information(self, "Output folder", out_dir)


# -----------------------------
# Standalone window
# -----------------------------

class PlaceholdrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planner — Prompt → Finished Video (Placeholder)")
        self.setMinimumSize(1050, 650)

        pane = PlaceholdrPane(self)
        self.setCentralWidget(pane)

        # Simple menu
        m = self.menuBar().addMenu("File")
        act_export = QAction("Export job JSON…", self)
        act_export.triggered.connect(pane._export_job_json)
        m.addAction(act_export)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)


def main() -> int:
    app = QApplication([])
    w = PlaceholdrWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())