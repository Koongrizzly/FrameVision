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
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_utf8_env(),
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


def _legacy_mode(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    _write_log(root, f"MODE: legacy; py={sys.executable}; cwd={os.getcwd()}")
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
        return _run_cmd(cmd, cwd=cwd, root=root)

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

    return _run_cmd(cmd, cwd=cwd, root=root)


def _new_mode(args: argparse.Namespace) -> int:
    root = Path(args.work_root).resolve()
    _write_log(root, f"MODE: new; py={sys.executable}; cwd={os.getcwd()}")

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
        p = subprocess.run(
            cmd,
            cwd=str(cli.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_utf8_env(),
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


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Detect legacy mode by presence of --seed_cmd_json or --root or --is_video
    is_legacy = any(a in argv for a in ["--seed_cmd_json", "--root", "--is_video", "--output"])
    if is_legacy:
        ap = argparse.ArgumentParser(prog="seedvr2_runner.py (legacy)")
        ap.add_argument("--seed_cmd_json", default="")
        ap.add_argument("--root", required=True)
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
