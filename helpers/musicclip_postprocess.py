from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Tuple


def _root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "helpers":
        return here.parent.parent
    return here.parent


def _ffmpeg(root: Path, supplied: str = "") -> str:
    if supplied and Path(supplied).is_file():
        return supplied
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for p in (root / "presets" / "bin" / exe, root / "bin" / exe, root / exe):
        if p.is_file():
            return str(p)
    return supplied or exe


def _ffprobe(root: Path, supplied: str = "") -> str:
    if supplied and Path(supplied).is_file():
        return supplied
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    for p in (root / "presets" / "bin" / exe, root / "bin" / exe, root / exe):
        if p.is_file():
            return str(p)
    return supplied or exe


def _probe_fps(ffprobe: str, src: Path) -> str:
    try:
        out = subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", str(src),
        ], text=True, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(out or "{}")
        stream = (data.get("streams") or [{}])[0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = str(stream.get(key) or "").strip()
            if val and val not in {"0/0", "0"}:
                return val
    except Exception:
        pass
    return "30"


def _hypir_paths(root: Path) -> Tuple[Path, Path, Path, Path, Optional[Path]]:
    env_py = root / "environments" / ".hypir" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Compatibility with an older Windows layout that placed python.exe directly in .hypir.
    if os.name == "nt" and not env_py.is_file():
        alt = root / "environments" / ".hypir" / "python.exe"
        if alt.is_file():
            env_py = alt
    repo = root / "presets" / "extra_env" / "hypir_src" / "HYPIR"
    weight = root / "models" / "hypir" / "HYPIR_sd2.pth"
    runner = root / "helpers" / "hypir_runner.py"
    model_root = root / "models" / "hypir"
    candidates = [
        model_root / "stable-diffusion-2-1-base",
        model_root / "sd2-community" / "stable-diffusion-2-1-base",
        model_root / "stabilityai" / "stable-diffusion-2-1-base",
        model_root / "sd2_diffusers",
    ]
    env_model = (os.environ.get("FRAMEVISION_HYPIR_BASE_MODEL") or "").strip()
    if env_model:
        candidates.insert(0, Path(env_model))
    base = next((p for p in candidates if p.exists()), None)
    return env_py, repo, weight, runner, base


def _run(cmd, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    cp = subprocess.run(
        [str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if cp.returncode != 0:
        tail = "\n".join((cp.stdout or "").splitlines()[-20:])
        raise RuntimeError(tail or f"Command failed with exit code {cp.returncode}")


def _hypir_x1(src: Path, ffmpeg: str, ffprobe: str, progress: Callable[[str], None]) -> None:
    root = _root()
    env_py, repo, weight, runner, base = _hypir_paths(root)
    missing = []
    for label, p in (("HYPIR Python", env_py), ("HYPIR repository", repo), ("HYPIR weights", weight), ("HYPIR runner", runner)):
        if not p.exists():
            missing.append(f"{label}: {p}")
    if base is None:
        missing.append(f"Stable Diffusion 2.1 base model under: {root / 'models' / 'hypir'}")
    if missing:
        raise RuntimeError("HyPiR x1 is enabled but its runtime is incomplete:\n" + "\n".join(missing))

    fps = _probe_fps(ffprobe, src)
    work = Path(tempfile.mkdtemp(prefix="fv_musicclip_hypir_x1_"))
    in_dir, out_dir = work / "in", work / "out"
    in_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = src.with_name(f".{src.stem}.hypirx1.{uuid.uuid4().hex[:8]}.mp4")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1"); env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_HUB_OFFLINE", "1"); env.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        progress("HyPiR x1: extracting frames...")
        _run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src), "-map", "0:v:0", "-vsync", "0", str(in_dir / "f_%08d.png")])
        progress("HyPiR x1: restoring frames (face details may change slightly)...")
        _run([
            env_py, "-X", "utf8", runner,
            "--repo", repo, "--base-model", base, "--weight", weight,
            "--input", in_dir, "--output", out_dir, "--scale", "1",
            "--prompt", "", "--seed", "-1", "--patch-size", "512", "--stride", "256", "--device", "cuda",
        ], cwd=repo, env=env)
        progress("HyPiR x1: rebuilding video...")
        _run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            "-framerate", fps, "-i", str(out_dir / "f_%08d.png"), "-i", str(src),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(tmp),
        ])
        if not tmp.is_file() or tmp.stat().st_size <= 0:
            raise RuntimeError("HyPiR x1 rebuild did not create a valid video.")
        os.replace(str(tmp), str(src))
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        shutil.rmtree(work, ignore_errors=True)


def _lanczos_x2(src: Path, ffmpeg: str, progress: Callable[[str], None]) -> None:
    tmp = src.with_name(f".{src.stem}.lanczos2x.{uuid.uuid4().hex[:8]}.mp4")
    try:
        progress("Lanczos x2: upsampling final video...")
        _run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=iw*2:ih*2:flags=lanczos",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", str(tmp),
        ])
        if not tmp.is_file() or tmp.stat().st_size <= 0:
            raise RuntimeError("Lanczos x2 did not create a valid video.")
        os.replace(str(tmp), str(src))
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


def postprocess_music_video(
    video_path: str,
    *,
    use_hypir_x1: bool = False,
    use_lanczos_x2: bool = False,
    ffmpeg: str = "",
    ffprobe: str = "",
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Post-process a completed music video in-place.

    Order is intentional: HyPiR x1 restoration first, Lanczos x2 second.
    """
    src = Path(str(video_path or "")).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Music clip post-processing source is missing: {src}")
    if not use_hypir_x1 and not use_lanczos_x2:
        return str(src)
    root = _root()
    ffmpeg = _ffmpeg(root, ffmpeg)
    ffprobe = _ffprobe(root, ffprobe)
    cb = progress if callable(progress) else (lambda _text: None)
    if use_hypir_x1:
        _hypir_x1(src, ffmpeg, ffprobe, cb)
    if use_lanczos_x2:
        _lanczos_x2(src, ffmpeg, cb)
    cb("Music clip post-processing complete.")
    return str(src)
