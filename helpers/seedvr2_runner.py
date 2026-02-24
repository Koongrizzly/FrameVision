# SeedVR2 runner (FrameVision)
# Backward-compatible wrapper that supports both:
# 1) Legacy FrameVision call: --seed_cmd_json ... --root ... --input ... --output ... --is_video ...
# 2) New call: --cli ... --input ... --ffmpeg ... --ffprobe ... --work_root ... [--keep]
#
# Always logs into: <root>/output/_logs/seedvr2_runner.log

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _logfile(root: Path) -> Path:
    return root / "output" / "_logs" / "seedvr2_runner.log"


def _write_log(root: Path, msg: str) -> None:
    lf = _logfile(root)
    _safe_mkdir(lf.parent)
    try:
        with lf.open("a", encoding="utf-8") as f:
            f.write(f"[{_now()}] {msg}\n")
    except Exception:
        # last resort
        try:
            sys.stderr.write(msg + "\n")
        except Exception:
            pass



def _utf8_env(base: Optional[dict] = None) -> dict:
    env = dict(base) if base else dict(os.environ)
    # Force UTF-8 so SeedVR2's emoji/log prints won't crash on Windows cp1252 pipes.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _ensure_utf8_flag(cmd: List[str]) -> List[str]:
    # Insert "-X utf8" right after the python executable when appropriate.
    try:
        if not cmd:
            return cmd
        exe = cmd[0].lower()
        if exe.endswith("python.exe") or exe.endswith("python"):
            if len(cmd) >= 2 and cmd[1] == "-X" and (len(cmd) >= 3 and cmd[2].lower() == "utf8"):
                return cmd
            return [cmd[0], "-X", "utf8"] + cmd[1:]
    except Exception:
        pass
    return cmd

def _run_cmd(cmd: List[str], cwd: Optional[Path], root: Path) -> int:
    cmd = _ensure_utf8_flag(list(cmd))
    _write_log(root, "RUN: " + " ".join(shlex.quote(x) for x in cmd))
    try:
        # Stream child output so FrameVision worker/progress parsers can see live logs.
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_utf8_env(),
            text=True,
            errors="replace",
            bufsize=1,
            universal_newlines=True,
        )
        out_lines: List[str] = []
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            out_lines.append(line)
            try:
                print(line, flush=True)
            except Exception:
                pass
        p.wait()
        if out_lines:
            _write_log(root, "STDOUT/ERR:\n" + "\n".join(out_lines))
        _write_log(root, f"EXIT: {p.returncode}")

        rc = int(p.returncode or 0)
        if rc == 0:
            try:
                out_s = _parse_output_from_forward_args(forward)
                if out_s:
                    out_p = Path(out_s)
                    if not out_p.is_absolute():
                        out_p = (cli.parent / out_p).resolve()
                    _write_log(root, f"NEW MODE parsed output: {out_p}")
                    src_p = Path(args.input)
                    if _looks_video_path(src_p) or _looks_video_path(out_p):
                        _repair_video_output(root, src_p, out_p, args.ffmpeg, args.ffprobe)
                else:
                    _write_log(root, "NEW MODE post-fix skipped: --output not found in forwarded args")
            except Exception as e:
                _write_log(root, f"NEW MODE post-fix exception: {e!r}")
        return rc
    except Exception as e:
        _write_log(root, f"EXCEPTION while running subprocess: {e!r}")
        return 1

def _ffprobe_json(ffprobe: str, path: Path, root: Path) -> Dict[str, Any]:
    try:
        cmd = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        p = subprocess.run(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        if p.returncode != 0:
            if p.stderr:
                _write_log(root, f"ffprobe failed for {path}: {p.stderr.strip()}")
            return {}
        return json.loads(p.stdout or "{}") if (p.stdout or "").strip() else {}
    except Exception as e:
        _write_log(root, f"ffprobe exception for {path}: {e!r}")
        return {}


def _media_meta(ffprobe: str, path: Path, root: Path) -> Dict[str, Any]:
    j = _ffprobe_json(ffprobe, path, root)
    streams = j.get("streams") or []
    fmt = j.get("format") or {}
    v = next((st for st in streams if st.get("codec_type") == "video"), {})
    a = next((st for st in streams if st.get("codec_type") == "audio"), None)

    def _to_float(x: Any) -> Optional[float]:
        try:
            if x in (None, "", "N/A"):
                return None
            return float(x)
        except Exception:
            return None

    def _ratio_to_float(x: Any) -> Optional[float]:
        try:
            s = str(x or "").strip()
            if not s or s in ("0/0", "0", "N/A"):
                return None
            if "/" in s:
                a1, b1 = s.split("/", 1)
                a1f = float(a1); b1f = float(b1)
                if b1f == 0:
                    return None
                return a1f / b1f
            return float(s)
        except Exception:
            return None

    dur = _to_float(v.get("duration")) or _to_float(fmt.get("duration"))
    avg_fps = _ratio_to_float(v.get("avg_frame_rate")) or _ratio_to_float(v.get("r_frame_rate"))
    return {
        "duration": dur,
        "fps": avg_fps,
        "has_audio": bool(a),
        "audio_codec": (a or {}).get("codec_name") if a else None,
        "video_codec": v.get("codec_name"),
    }


def _run_cmd_env(cmd: List[str], cwd: Optional[Path], root: Path, env: Optional[dict] = None) -> int:
    cmd = _ensure_utf8_flag(list(cmd))
    _write_log(root, "RUN: " + " ".join(shlex.quote(x) for x in cmd))
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_utf8_env(env),
            text=True,
            errors="replace",
        )
        if p.stdout:
            _write_log(root, "STDOUT:\n" + p.stdout.strip())
        if p.stderr:
            _write_log(root, "STDERR:\n" + p.stderr.strip())
        _write_log(root, f"EXIT: {p.returncode}")
        return int(p.returncode)
    except Exception as e:
        _write_log(root, f"EXCEPTION while running subprocess: {e!r}")
        return 1


def _repair_video_output(root: Path, src: Path, out_path: Path, ffmpeg_path: Optional[str], ffprobe_path: Optional[str]) -> None:
    try:
        if not src.exists() or not out_path.exists():
            return
        if src.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv"}:
            return
        ffmpeg = str(ffmpeg_path or "ffmpeg")
        ffprobe = str(ffprobe_path or "ffprobe")
        src_meta = _media_meta(ffprobe, src, root)
        out_meta = _media_meta(ffprobe, out_path, root)
        _write_log(root, f"SRC META: {src_meta}")
        _write_log(root, f"OUT META: {out_meta}")

        src_d = src_meta.get("duration") or 0.0
        out_d = out_meta.get("duration") or 0.0
        dur_delta = abs(float(src_d) - float(out_d)) if (src_d and out_d) else 0.0
        need_audio = bool(src_meta.get("has_audio")) and not bool(out_meta.get("has_audio"))

        src_fps = src_meta.get("fps") or 0.0
        out_fps = out_meta.get("fps") or 0.0
        fps_delta = abs(float(src_fps) - float(out_fps)) if (src_fps and out_fps) else 0.0
        # Use a tighter threshold than 1s so short test clips also get repaired.
        # Trigger on either absolute duration drift (>100ms), relative drift (>0.5%), or fps mismatch (>0.01 fps).
        dur_rel = (dur_delta / float(src_d)) if (src_d and dur_delta is not None) else 0.0
        need_duration_fix = bool(src_d and out_d and (dur_delta > 0.10 or dur_rel > 0.005))
        need_fps_fix = bool(src_fps and out_fps and fps_delta > 0.01)

        _write_log(root, f"DURATION CHECK: src={src_d:.3f}s out={out_d:.3f}s delta={dur_delta:.3f}s rel={dur_rel:.4f} src_fps={src_fps} out_fps={out_fps} fps_delta={fps_delta}")
        if not need_audio and not need_duration_fix and not need_fps_fix:
            _write_log(root, "Post-fix: no repair needed.")
            return

        tmp = out_path.with_name(out_path.stem + "_postfix" + out_path.suffix)
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(out_path), "-i", str(src), "-map", "0:v:0"]
        if src_meta.get("has_audio"):
            cmd += ["-map", "1:a?"]

        # If duration/fps drift exists, retime video to source duration / fps and re-encode video.
        if (need_duration_fix or need_fps_fix) and src_d and out_d and out_d > 0:
            factor = float(src_d) / float(out_d)
            # Bound to avoid absurd values on corrupt probes.
            if factor < 0.25 or factor > 4.0:
                _write_log(root, f"Post-fix skipped retime due to suspicious factor={factor}")
                need_duration_fix = False
            else:
                _write_log(root, f"Post-fix retime enabled: src={src_d:.3f}s out={out_d:.3f}s factor={factor:.9f}")
                cmd += ["-vf", f"setpts={factor:.9f}*PTS"]
                if src_fps and 1.0 <= float(src_fps) <= 240.0:
                    fps_txt = (f"{float(src_fps):.6f}").rstrip("0").rstrip(".")
                    cmd += ["-r", fps_txt]
                cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        if not (need_duration_fix or need_fps_fix):
            cmd += ["-c:v", "copy"]

        if src_meta.get("has_audio"):
            cmd += ["-c:a", "copy", "-shortest"]
        cmd += [str(tmp)]

        rc = _run_cmd_env(cmd, cwd=root, root=root)
        if rc != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
            _write_log(root, f"Post-fix failed rc={rc}; keeping original output")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return

        bak = out_path.with_name(out_path.stem + "_prepostfix" + out_path.suffix)
        try:
            if bak.exists():
                bak.unlink()
        except Exception:
            pass
        try:
            out_path.replace(bak)
            tmp.replace(out_path)
            try:
                if bak.exists() and bak.stat().st_size > 0:
                    bak.unlink()
            except Exception:
                pass
            _write_log(root, "Post-fix applied successfully (audio/duration repair).")
        except Exception as e:
            _write_log(root, f"Post-fix replace failed: {e!r}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    except Exception as e:
        _write_log(root, f"Post-fix exception: {e!r}")


def _detect_root_from_args_or_env(root_value: str) -> Path:
    # Accept explicit root when provided; otherwise try common FrameVision locations relative to this file/cwd.
    if str(root_value or "").strip():
        try:
            return Path(root_value).resolve()
        except Exception:
            pass
    candidates = []
    try:
        here = Path(__file__).resolve()
        candidates += [here.parent.parent, here.parent]
    except Exception:
        pass
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    # Prefer a folder that looks like FrameVision root.
    for c in candidates:
        try:
            if (c / "presets" / "bin").exists() or (c / "helpers").exists() or (c / "output").exists():
                return c.resolve()
        except Exception:
            continue
    # Last resort
    try:
        return candidates[0].resolve()
    except Exception:
        return Path(".").resolve()

def _legacy_mode(args: argparse.Namespace) -> int:
    root = _detect_root_from_args_or_env(getattr(args, 'root', ''))
    _write_log(root, f"MODE: legacy; py={sys.executable}; cwd={os.getcwd()}")
    try:
        print(f"[seedvr2_runner] input={args.input}", flush=True)
    except Exception:
        pass
    try:
        seed_cfg = json.loads(args.seed_cmd_json) if args.seed_cmd_json else {}
    except Exception as e:
        _write_log(root, f"Failed to parse --seed_cmd_json: {e!r}")
        seed_cfg = {}

    # Some older FrameVision integrations pass a full command list in JSON.
    # Support that for maximum backward compatibility.
    if isinstance(seed_cfg, list) and seed_cfg and all(isinstance(x, str) for x in seed_cfg):
        cmd = _ensure_utf8_flag(list(seed_cfg))

        # If the CLI expects --dit_model to be a filename (argparse choices),
        # ensure we don't pass an absolute path here.
        try:
            if "--dit_model" in cmd:
                i = cmd.index("--dit_model")
                if i + 1 < len(cmd):
                    cmd[i + 1] = Path(cmd[i + 1]).name
        except Exception:
            pass

        # Pick a sensible cwd: the directory containing inference_cli.py if present.
        cwd = None
        try:
            for part in cmd:
                if isinstance(part, str) and part.lower().endswith("inference_cli.py"):
                    p = Path(part)
                    if p.exists():
                        cwd = p.parent
                    break
        except Exception:
            cwd = None

        _write_log(root, "legacy JSON was a command list; executing directly")
        rc = _run_cmd(cmd, cwd=cwd, root=root)
        if rc == 0 and str(args.is_video).strip() not in ("0", "false", "False", "") and args.output:
            _repair_video_output(root, Path(args.input), Path(args.output), None, None)
        return rc

    # Normalize dict-based config.
    if not isinstance(seed_cfg, dict):
        _write_log(root, f"Unexpected seed_cfg type: {type(seed_cfg)}; falling back to empty config")
        seed_cfg = {}

    # Determine CLI path. Prefer explicit in JSON, otherwise default expected repo location.
    cli_path = seed_cfg.get("cli") or str(
        root / "presets" / "extra_env" / "seedvr2_src" / "ComfyUI-SeedVR2_VideoUpscaler" / "inference_cli.py"
    )
    cli = Path(cli_path)
    if not cli.exists():
        _write_log(root, f"Missing SeedVR2 CLI: {cli}")
        return 2

    # Determine model_dir default
    model_dir = Path(seed_cfg.get("model_dir") or (root / "models" / "SEEDVR2"))

    # Build command: python inference_cli.py <input> --output <output> --model_dir <...> plus extra args from json
    cmd = [sys.executable, str(cli), str(args.input)]
    out_path = args.output or seed_cfg.get("output")
    if out_path:
        cmd += ["--output", str(out_path)]

    # output_format can be passed via json, but don't force it
    if seed_cfg.get("output_format"):
        cmd += ["--output_format", str(seed_cfg["output_format"])]

    # Model dir + model selection if provided
    cmd += ["--model_dir", str(model_dir)]
    if seed_cfg.get("dit_model"):
        # The CLI's argparse uses 'choices' and expects a filename, not a full path.
        cmd += ["--dit_model", Path(str(seed_cfg["dit_model"])).name]

    # Resolution / max_resolution
    if seed_cfg.get("resolution") is not None:
        cmd += ["--resolution", str(seed_cfg["resolution"])]
    if seed_cfg.get("max_resolution") is not None:
        cmd += ["--max_resolution", str(seed_cfg["max_resolution"])]

    # Pass through other known knobs if present
    passthru_flags = [
        "batch_size",
        "uniform_batch_size",
        "seed",
        "chunk_size",
        "temporal_overlap",
        "color_correction",
        "cuda_device",
        "attention_mode",
        "vae_encode_tiled",
        "vae_decode_tiled",
        "vae_encode_tile_size",
        "vae_encode_tile_overlap",
        "vae_decode_tile_size",
        "vae_decode_tile_overlap",
        "tile_debug",
        "debug",
    ]
    for k in passthru_flags:
        v = seed_cfg.get(k, None)
        if v is None or v is False:
            continue
        flag = "--" + k
        if v is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    # Video backend: if json says ffmpeg and paths exist, pass it
    vb = seed_cfg.get("video_backend")
    if vb:
        cmd += ["--video_backend", str(vb)]
        if vb == "ffmpeg":
            # Only pass 10bit if requested
            if seed_cfg.get("10bit") is True:
                cmd.append("--10bit")

    # Choose cwd as repo root (where inference_cli.py lives)
    cwd = cli.parent

    rc = _run_cmd(cmd, cwd=cwd, root=root)
    if rc == 0 and str(args.is_video).strip() not in ("0", "false", "False", "") and out_path:
        _repair_video_output(root, Path(args.input), Path(str(out_path)), None, None)
    return rc




def _parse_output_from_forward_args(forward: List[str]) -> Optional[str]:
    """Extract --output path from forwarded CLI args."""
    try:
        for i, a in enumerate(forward or []):
            if a == "--output" and i + 1 < len(forward):
                return str(forward[i + 1])
            if isinstance(a, str) and a.startswith("--output="):
                return a.split("=", 1)[1]
    except Exception:
        pass
    return None


def _looks_video_path(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv"}

def _new_mode(args: argparse.Namespace) -> int:
    root = Path(args.work_root).resolve()
    _write_log(root, f"MODE: new; py={sys.executable}; cwd={os.getcwd()} [PATCH active newmode-postfix-v2]")
    try:
        print(f"[seedvr2_runner] input={args.input}", flush=True)
    except Exception:
        pass

    cli = Path(args.cli)
    if not cli.exists():
        _write_log(root, f"Missing SeedVR2 CLI: {cli}")
        return 2

    # Remaining args after '--' are forwarded to inference_cli.py (except output which may be present there)
    forward = list(args.forward_args or [])
    # Ensure input is first positional for inference_cli.py
    cmd = _ensure_utf8_flag([sys.executable, str(cli), str(args.input)] + forward)

    # In "new" mode we may need ffmpeg/ffprobe on PATH for --video_backend ffmpeg.
    # We'll prepend their folder to PATH if provided.
    env = _utf8_env(os.environ.copy())
    ffmpeg_dir = str(Path(args.ffmpeg).resolve().parent) if args.ffmpeg else ""
    ffprobe_dir = str(Path(args.ffprobe).resolve().parent) if args.ffprobe else ""
    extra_paths = [p for p in [ffmpeg_dir, ffprobe_dir] if p]
    if extra_paths:
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])

    _write_log(root, "PATH+ (ffmpeg): " + ";".join(extra_paths))

    try:
        p = subprocess.Popen(
            cmd,
            cwd=str(cli.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_utf8_env(env),
            text=True,
            errors="replace",
            bufsize=1,
            universal_newlines=True,
        )
        out_lines: List[str] = []
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            out_lines.append(line)
            try:
                print(line, flush=True)
            except Exception:
                pass
        p.wait()
        if out_lines:
            _write_log(root, "STDOUT/ERR:\n" + "\n".join(out_lines))
        _write_log(root, f"EXIT: {p.returncode}")
        return int(p.returncode or 0)
    except Exception as e:
        _write_log(root, f"EXCEPTION while running subprocess: {e!r}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Detect legacy mode by presence of --seed_cmd_json or --root or --is_video
    is_legacy = any(a in argv for a in ["--seed_cmd_json", "--root", "--is_video"])
    if (not is_legacy) and ("--input" in argv) and ("--cli" not in argv) and ("--work_root" not in argv):
        is_legacy = True
    if is_legacy:
        ap = argparse.ArgumentParser(prog="seedvr2_runner.py (legacy)")
        ap.add_argument("--seed_cmd_json", default="")
        ap.add_argument("--root", default="")
        ap.add_argument("--input", required=True)
        ap.add_argument("--output", default="")
        ap.add_argument("--is_video", default="0")
        args = ap.parse_args(argv)
        return _legacy_mode(args)

    # New mode with optional passthrough after '--'
    ap = argparse.ArgumentParser(prog="seedvr2_runner.py")
    ap.add_argument("--cli", required=True, help="Path to SeedVR2 inference_cli.py")
    ap.add_argument("--input", required=True, help="Input file (image or video)")
    ap.add_argument("--ffmpeg", required=False, default="", help="Path to ffmpeg")
    ap.add_argument("--ffprobe", required=False, default="", help="Path to ffprobe")
    ap.add_argument("--work_root", required=True, help="Root folder for temp work dirs / logs (FrameVision root)")
    ap.add_argument("--keep", action="store_true", help="Keep temp folder for debugging")

    # collect unknown args to forward to inference_cli.py, supporting "--" separator
    if "--" in argv:
        idx = argv.index("--")
        known = argv[:idx]
        forward = argv[idx+1:]
        args = ap.parse_args(known)
        args.forward_args = forward
    else:
        args, forward = ap.parse_known_args(argv)
        args.forward_args = forward

    return _new_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
