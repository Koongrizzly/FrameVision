from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]


def _root() -> Path:
    return _PROJECT_ROOT


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _norm(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    s = os.path.expandvars(os.path.expanduser(str(path).strip().strip('"')))
    try:
        p = Path(s)
        if not p.is_absolute():
            p = (_root() / p).resolve()
        return str(p)
    except Exception:
        return os.path.normpath(s)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return obj


def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def _find_conda_bat() -> str:
    conda_exe = os.environ.get('CONDA_EXE', '')
    guesses = []
    if conda_exe:
        ce = str(conda_exe)
        guesses.extend([
            ce.replace('\\Scripts\\conda.exe', '\\condabin\\conda.bat'),
            ce.replace('/Scripts/conda.exe', '/condabin/conda.bat'),
            ce.replace('\\Scripts\\conda.exe', '\\Scripts\\conda.bat'),
            ce.replace('/Scripts/conda.exe', '/Scripts/conda.bat'),
        ])
    guesses.extend([
        os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'condabin', 'conda.bat'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'Scripts', 'conda.bat'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'condabin', 'conda.bat'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'Scripts', 'conda.bat'),
        r'C:\ProgramData\miniconda3\condabin\conda.bat',
        r'C:\ProgramData\miniconda3\Scripts\conda.bat',
        r'C:\ProgramData\anaconda3\condabin\conda.bat',
        r'C:\ProgramData\anaconda3\Scripts\conda.bat',
    ])
    seen = set()
    for g in guesses:
        if not g:
            continue
        ng = os.path.normpath(g)
        if ng in seen:
            continue
        seen.add(ng)
        if os.path.isfile(ng):
            return ng
    return ''



def _wangp_cli_args(args: argparse.Namespace, settings_json: str, output_dir: str) -> List[str]:
    """Arguments that must be passed to WanGP's wgp.py for headless processing."""
    cmd_tail: List[str] = [
        '--process', settings_json,
        '--output-dir', output_dir,
    ]
    if args.attention:
        cmd_tail += ['--attention', str(args.attention)]
    if args.bf16:
        cmd_tail.append('--bf16')
    elif args.fp16:
        cmd_tail.append('--fp16')
    if args.verbose is not None:
        cmd_tail += ['--verbose', str(int(args.verbose))]
    return cmd_tail


def _wangp_setup_available(wangp_root: str) -> bool:
    root = Path(str(wangp_root or ''))
    return (root / 'setup.py').is_file() and (root / 'envs.json').is_file()


def _wangp_active_env_label(wangp_root: str) -> str:
    try:
        envs_path = Path(wangp_root) / 'envs.json'
        obj = json.loads(envs_path.read_text(encoding='utf-8'))
        active = str(obj.get('active') or '').strip()
        envs = obj.get('envs') if isinstance(obj.get('envs'), dict) else {}
        data = envs.get(active) if active and isinstance(envs.get(active), dict) else {}
        typ = str(data.get('type') or '').strip()
        path = str(data.get('path') or '').strip()
        if active and typ and path:
            return f"setup.py run active env: {active} ({typ}) {path}"
        if active:
            return f"setup.py run active env: {active}"
    except Exception:
        pass
    return 'setup.py run active env'


def _write_wangp_args_txt(wangp_root: str, cli_args: List[str]) -> Tuple[str, bool, bytes]:
    """Temporarily write scripts/args.txt for WanGP's new setup.py runner.

    The new WanGP installer launches through `python setup.py run`, which reads
    scripts/args.txt and appends it to `wgp.py`. We replace it only while the
    Planner clip is running, then restore the user's original file.
    """
    scripts_dir = Path(wangp_root) / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    args_path = scripts_dir / 'args.txt'
    existed = args_path.exists()
    old_data = b''
    if existed:
        try:
            old_data = args_path.read_bytes()
        except Exception:
            old_data = b''
    args_line = subprocess.list2cmdline(cli_args).strip()
    args_path.write_text(args_line + '\n', encoding='utf-8')
    return str(args_path), existed, old_data


def _restore_wangp_args_txt(args_path: str, existed: bool, old_data: bytes) -> None:
    try:
        p = Path(args_path)
        if existed:
            p.write_bytes(old_data or b'')
        else:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    except Exception:
        pass


def _build_wangp_launch(args: argparse.Namespace, wangp_root: str, wgp_py: str, settings_json: str, output_dir: str) -> Tuple[List[str], bool, List[str], str, str, List[str]]:
    cli_args = _wangp_cli_args(args, settings_json, output_dir)
    launch_mode = str(getattr(args, 'launch_mode', 'auto') or 'auto').strip().lower()

    if args.wangp_python or launch_mode == 'direct':
        python_exe = _norm(args.wangp_python) if args.wangp_python else sys.executable
        if not os.path.isfile(python_exe):
            raise FileNotFoundError(f"WanGP python not found: {args.wangp_python}")
        return [python_exe, wgp_py] + cli_args, False, [python_exe], python_exe, 'direct', []

    # New WanGP installer path: setup.py owns the active env (venv/uv/conda) via envs.json.
    # It appends scripts/args.txt to wgp.py, so use that instead of forcing the old `wan2gp` conda env.
    if launch_mode in ('auto', 'setup') and _wangp_setup_available(wangp_root):
        setup_py = str((Path(wangp_root) / 'setup.py').resolve())
        runner_bat = str((Path(wangp_root) / 'scripts' / 'run.bat').resolve())
        desc = f'{sys.executable} setup.py run  (via scripts/args.txt; runner ref: {runner_bat})'
        return [sys.executable, setup_py, 'run'], False, [desc], _wangp_active_env_label(wangp_root), 'setup', cli_args

    if launch_mode == 'setup':
        raise FileNotFoundError(f"WanGP setup runner not available. Expected setup.py and envs.json in: {wangp_root}")

    conda_bat = _find_conda_bat()
    if not conda_bat:
        raise FileNotFoundError('Could not find conda.bat needed to launch WanGP via the wan2gp environment.')

    cmd_tail: List[str] = ['python', wgp_py] + cli_args
    quoted_tail = subprocess.list2cmdline(cmd_tail)
    run_cmd = (
        f'call "{conda_bat}" activate wan2gp '
        f'&& cd /d "{wangp_root}" '
        f'&& set PYTORCH_NVFUSER_DISABLE=1 '
        f'&& set TORCH_COMPILE_DEBUG=0 '
        f'&& set TORCH_COMPILE_DEBUG_LOG=0 '
        f'&& {quoted_tail}'
    )
    launcher_desc = f'cmd.exe /d /c call "{conda_bat}" activate wan2gp && cd /d "{wangp_root}" && python {Path(wgp_py).name} ...'
    return [run_cmd], True, [launcher_desc], 'wan2gp (via conda activate)', 'conda', []


def _find_generated_video(output_dir: str, started_at: float, requested_out: str = "") -> str:
    out_dir = Path(output_dir)
    if requested_out:
        req = Path(requested_out)
        if req.is_file() and req.stat().st_size > 0:
            return str(req)

    candidates: List[Path] = []
    for pattern in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        candidates.extend(out_dir.glob(pattern))
    if not candidates:
        return ""

    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return ""

    candidates.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    for cand in candidates:
        if cand.stat().st_mtime >= started_at - 2.0:
            return str(cand)
    return str(candidates[0])


def _copy_or_move_video(src: str, dst: str) -> None:
    _ensure_parent(dst)
    src_p = Path(src)
    dst_p = Path(dst)
    if src_p.resolve() == dst_p.resolve():
        return
    try:
        shutil.move(str(src_p), str(dst_p))
        return
    except Exception:
        pass
    shutil.copy2(str(src_p), str(dst_p))




def _is_wangp_settings_json(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if any(k in obj for k in ("settings_version", "model_type", "activated_loras", "loras_multipliers")):
        return True
    return False


def _merge_known_settings(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    allowed = {
        "settings_version", "resolution", "flow_shift", "sliding_window_size", "sliding_window_overlap",
        "denoising_strength", "masking_strength", "audio_prompt_type", "perturbation_layers",
        "audio_scale", "guidance_phases", "num_inference_steps", "video_length", "negative_prompt",
        "activated_loras", "loras_multipliers", "lset_name", "force_fps", "seed", "image_mode",
        "image_prompt_type", "image_start", "image_end", "image_refs", "video_prompt_type",
        "video_source", "keep_frames_video_source", "input_video_strength", "output_filename",
        "audio_guide", "audio_guide2", "audio_source"
    }
    for k, v in src.items():
        if k in allowed:
            dst[k] = v


def _parse_resolution(text_value: str, fallback: Tuple[int, int] = (1280, 720)) -> Tuple[int, int]:
    s = str(text_value or '').strip().lower().replace(' ', '')
    if 'x' not in s:
        return fallback
    try:
        w, h = s.split('x', 1)
        wi = max(64, int(w))
        hi = max(64, int(h))
        return wi, hi
    except Exception:
        return fallback


def _detect_transition_recipe(obj: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "is_recipe": False,
        "width": None,
        "height": None,
        "fps": None,
        "frames": None,
        "uses_firstframe": False,
        "uses_middleframe": False,
        "uses_ref_image": False,
    }
    if not isinstance(obj, dict) or not isinstance(obj.get("nodes"), list):
        return info
    info["is_recipe"] = True
    for node in obj.get("nodes", []):
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or '')
        widgets = node.get("widgets_values") or []
        ttl = title.lower()
        if title == 'Get_width' and widgets:
            try: info['width'] = int(widgets[0])
            except Exception: pass
        elif title == 'Get_height' and widgets:
            try: info['height'] = int(widgets[0])
            except Exception: pass
        elif title == 'Get_fps' and widgets:
            try: info['fps'] = int(float(widgets[0]))
            except Exception: pass
        elif title == 'Get_frames' and widgets:
            try: info['frames'] = int(widgets[0])
            except Exception: pass
        elif 'firstframe' in ttl:
            info['uses_firstframe'] = True
        elif 'middleframe' in ttl:
            info['uses_middleframe'] = True
        elif 'ref_image' in ttl:
            info['uses_ref_image'] = True
    return info


def _load_lora_json_info(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not path:
        return {}, {}
    obj = _safe_read_json(path)
    if _is_wangp_settings_json(obj):
        return obj, {}
    recipe = _detect_transition_recipe(obj)
    return {}, recipe

def _build_base_settings(template_path: str = "") -> Dict[str, Any]:
    if template_path:
        return _safe_read_json(template_path)

    return {
        "settings_version": 2.55,
        "resolution": "1280x720",
        "flow_shift": 5.0,
        "sliding_window_size": 481,
        "sliding_window_overlap": 17,
        "denoising_strength": 1.0,
        "masking_strength": 0,
        "audio_prompt_type": "",
        "perturbation_layers": [28],
        "audio_scale": 1.0,
        "guidance_phases": 1,
        "num_inference_steps": 8,
        "video_length": 241,
    }


def _build_settings_payload(args: argparse.Namespace) -> Dict[str, Any]:
    settings = _build_base_settings(args.settings_template)

    lora_json_path = _norm(args.lora_json)
    lora_json_settings: Dict[str, Any] = {}
    lora_recipe_info: Dict[str, Any] = {}
    if lora_json_path:
        if not os.path.isfile(lora_json_path):
            raise FileNotFoundError(f"LoRA JSON not found: {args.lora_json}")
        lora_json_settings, lora_recipe_info = _load_lora_json_info(lora_json_path)
        if lora_json_settings:
            _merge_known_settings(settings, lora_json_settings)

    image_path = _norm(args.image)
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Start image not found: {args.image}")

    # WanGP wraps a plain settings dict into a single-task manifest and requires model_type.
    settings["model_type"] = str(args.model_type or "ltx2_22B_distilled").strip()
    settings["prompt"] = str(args.prompt or "").strip()
    settings["negative_prompt"] = str(args.negative or settings.get("negative_prompt") or "").strip()

    resolved_resolution = str(args.resolution or settings.get("resolution") or "1280x720").strip()
    if lora_recipe_info.get("is_recipe"):
        rw = lora_recipe_info.get("width")
        rh = lora_recipe_info.get("height")
        if isinstance(rw, int) and isinstance(rh, int) and rw > 0 and rh > 0:
            resolved_resolution = f"{rw}x{rh}"
    settings["resolution"] = resolved_resolution
    settings["num_inference_steps"] = int(args.steps)
    settings["video_length"] = int(args.frames)
    if lora_recipe_info.get("is_recipe") and isinstance(lora_recipe_info.get("frames"), int) and int(args.frames) == 241:
        settings["video_length"] = int(lora_recipe_info["frames"])
    # WanGP's get_computed_fps() expects force_fps to behave like a string
    # (it calls len(force_fps) before numeric checks), so do not store this as int.
    settings["force_fps"] = str(int(args.fps))
    if lora_recipe_info.get("is_recipe") and isinstance(lora_recipe_info.get("fps"), int) and int(args.fps) == 24:
        settings["force_fps"] = str(int(lora_recipe_info["fps"]))
    settings["seed"] = int(args.seed) if args.seed is not None else -1

    # Image-to-video.
    settings["image_mode"] = 0
    image_end_path = _norm(getattr(args, "image_end", ""))
    if image_end_path and not os.path.isfile(image_end_path):
        raise FileNotFoundError(f"End image not found: {args.image_end}")

    settings["image_prompt_type"] = "SE" if image_end_path else "S"
    settings["image_start"] = [image_path]
    settings["image_end"] = [image_end_path] if image_end_path else None
    settings["image_refs"] = None
    settings["video_prompt_type"] = ""
    settings["video_source"] = None
    settings["keep_frames_video_source"] = ""
    settings["input_video_strength"] = float(args.input_video_strength)

    # Optional LoRA activation / transition recipe hints.
    lora_file_path = _norm(args.lora_file)
    if lora_file_path:
        if not os.path.isfile(lora_file_path):
            raise FileNotFoundError(f"LoRA file not found: {args.lora_file}")
        settings["activated_loras"] = [os.path.basename(lora_file_path)]
        settings["loras_multipliers"] = str(args.lora_multiplier)
        settings["ltx_transition_lora_file"] = lora_file_path

    if lora_json_path:
        settings["ltx_transition_json"] = lora_json_path
        settings["ltx_transition_json_name"] = os.path.basename(lora_json_path)
        if lora_json_settings:
            settings["lset_name"] = os.path.basename(lora_json_path)
        elif lora_recipe_info.get("is_recipe"):
            settings["ltx_transition_recipe"] = lora_recipe_info
            if lora_recipe_info.get("uses_firstframe"):
                settings["ltx_transition_firstframe"] = image_path
            if lora_recipe_info.get("uses_ref_image"):
                settings["image_refs"] = [image_path]
            if lora_recipe_info.get("uses_middleframe"):
                settings["ltx_transition_middleframe"] = image_path

    # Output control.
    out_stem = Path(args.output).stem
    settings["output_filename"] = out_stem

    # Known-good distilled defaults from the user's WanGP settings export.
    settings["flow_shift"] = float(args.flow_shift)
    settings["sliding_window_size"] = int(args.sliding_window_size)
    settings["sliding_window_overlap"] = int(args.sliding_window_overlap)
    settings["guidance_phases"] = int(args.guidance_phases)
    settings["denoising_strength"] = float(args.denoising_strength)
    settings["masking_strength"] = float(args.masking_strength)
    audio_path = _norm(getattr(args, "audio", ""))
    if audio_path:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio guide not found: {args.audio}")
        settings["audio_prompt_type"] = "A"
        settings["audio_guide"] = audio_path
    else:
        settings["audio_prompt_type"] = str(settings.get("audio_prompt_type") or "")
        settings["audio_guide"] = None

    settings["audio_scale"] = float(args.audio_scale)
    settings["perturbation_layers"] = [int(x) for x in args.perturbation_layers]

    return settings


def _run_wangp_generate(args: argparse.Namespace) -> int:
    wangp_root = _norm(args.wangp_root)
    if not wangp_root or not os.path.isdir(wangp_root):
        raise FileNotFoundError(f"WanGP root not found: {args.wangp_root}")

    wgp_py = _norm(args.wgp_py) if args.wgp_py else os.path.join(wangp_root, "wgp.py")
    if not os.path.isfile(wgp_py):
        raise FileNotFoundError(f"WanGP runner not found: {wgp_py}")

    python_exe = ''

    output_path = _norm(args.output)
    output_dir = str(Path(output_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = _build_settings_payload(args)

    with tempfile.TemporaryDirectory(prefix="ltx23_cli_") as td:
        settings_json = os.path.join(td, "ltx23_task.json")
        _safe_write_json(settings_json, payload)

        launch_cmd, launch_shell, tried_launches, python_exe, launch_mode, setup_cli_args = _build_wangp_launch(
            args=args,
            wangp_root=wangp_root,
            wgp_py=wgp_py,
            settings_json=settings_json,
            output_dir=output_dir,
        )

        print(f"[ltx23_cli] WanGP root: {wangp_root}")
        print(f"[ltx23_cli] Python/launcher: {python_exe}")
        print(f"[ltx23_cli] Runner: {wgp_py}")
        print(f"[ltx23_cli] Model: {payload['model_type']}")
        if payload.get("activated_loras"):
            print(f"[ltx23_cli] LoRA: {payload.get('activated_loras')}")
        else:
            print("[ltx23_cli] LoRA: disabled")
        if payload.get("ltx_transition_json_name"):
            print(f"[ltx23_cli] Transition JSON: {payload.get('ltx_transition_json_name')}")
        else:
            print("[ltx23_cli] Transition JSON: disabled")
        print(f"[ltx23_cli] Image: {payload['image_start'][0]}")
        if payload.get("audio_guide"):
            print(f"[ltx23_cli] Audio guide: {payload.get('audio_guide')}")
        else:
            print("[ltx23_cli] Audio guide: disabled")
        print(f"[ltx23_cli] Output target: {output_path}")
        print(f"[ltx23_cli] Frames={payload['video_length']} Steps={payload['num_inference_steps']} Res={payload['resolution']}")
        print("[ltx23_cli] Launching WanGP...")
        if tried_launches:
            print('[ltx23_cli] Launch candidates considered:')
            for one in tried_launches:
                print(f'  - {one}')

        started_at = time.time()
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUTF8", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
        args_backup: Optional[Tuple[str, bool, bytes]] = None
        try:
            if launch_mode == 'setup':
                args_backup = _write_wangp_args_txt(wangp_root, setup_cli_args)
                print(f"[ltx23_cli] Wrote temporary WanGP scripts/args.txt: {args_backup[0]}")
            if launch_shell:
                cp = subprocess.run(launch_cmd[0], cwd=wangp_root, check=False, shell=True, env=child_env)
            else:
                cp = subprocess.run(launch_cmd, cwd=wangp_root, check=False, env=child_env)
        finally:
            if args_backup is not None:
                _restore_wangp_args_txt(*args_backup)
                print("[ltx23_cli] Restored WanGP scripts/args.txt")
        if cp.returncode != 0:
            raise RuntimeError(f"WanGP returned non-zero exit code: {cp.returncode}")

        produced = _find_generated_video(output_dir=output_dir, started_at=started_at, requested_out=output_path)
        if not produced:
            raise FileNotFoundError(
                "WanGP finished without a detectable output video in the target folder. "
                f"Checked: {output_dir}"
            )

        _copy_or_move_video(produced, output_path)
        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            raise RuntimeError(f"Expected output video missing after copy/move: {output_path}")

        print(f"[ltx23_cli] Done: {output_path}")
        return 0


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ltx23_cli.py",
        description="Private bridge CLI: run WanGP LTX 2.3 22B distilled from FrameVision/planner.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate one LTX clip through WanGP")
    g.add_argument("--wangp-root", required=True, help="Path to the WanGP root folder")
    g.add_argument("--wangp-python", default="", help="Path to WanGP's python.exe; auto-guessed when omitted")
    g.add_argument("--wgp-py", default="", help="Optional explicit path to WanGP wgp.py")
    g.add_argument("--launch-mode", choices=["auto", "setup", "conda", "direct"], default="auto", help="WanGP launch mode. auto prefers the new setup.py/envs.json runner, then falls back to old conda mode.")
    g.add_argument("--settings-template", default="", help="Optional WanGP settings JSON to use as a template")
    g.add_argument("--lora-file", default="", help="Optional LoRA safetensors to activate for this run")
    g.add_argument("--lora-json", default="", help="Optional companion JSON. WanGP settings JSON is merged; workflow JSON is stored as transition metadata/hints")
    g.add_argument("--lora-multiplier", type=float, default=1.0, help="LoRA multiplier string/value for activated_loras")
    g.add_argument("--model-type", default="ltx2_22B_distilled", help="WanGP model type key (default: ltx2_22B_distilled)")
    g.add_argument("--prompt", required=True, help="Main prompt")
    g.add_argument("--negative", default="", help="Negative prompt (kept for later wiring)")
    g.add_argument("--image", required=True, help="Start image path")
    g.add_argument("--image-end", default="", help="Optional target end image path")
    g.add_argument("--audio", default="", help="Optional audio guide path passed into WanGP as audio_guide")
    g.add_argument("--output", required=True, help="Target output video path")
    g.add_argument("--frames", type=int, default=241, help="Video length / frames for WanGP")
    g.add_argument("--fps", type=int, default=24, help="Force FPS field passed into WanGP settings")
    g.add_argument("--steps", type=int, default=8, help="Inference steps")
    g.add_argument("--seed", type=int, default=None, help="Seed; omit for random")
    g.add_argument("--resolution", default="1280x720", help="Resolution string, e.g. 1280x720")
    g.add_argument("--attention", default="", help="Optional WanGP attention override")
    g.add_argument("--bf16", action="store_true", help="Pass --bf16 to WanGP")
    g.add_argument("--fp16", action="store_true", help="Pass --fp16 to WanGP")
    g.add_argument("--verbose", type=int, default=None, help="Optional WanGP verbose level")

    # Distilled defaults; exposed mainly so you can tweak if WanGP behaves differently later.
    g.add_argument("--flow-shift", type=float, default=5.0)
    g.add_argument("--sliding-window-size", type=int, default=481)
    g.add_argument("--sliding-window-overlap", type=int, default=17)
    g.add_argument("--guidance-phases", type=int, default=1)
    g.add_argument("--denoising-strength", type=float, default=1.0)
    g.add_argument("--masking-strength", type=float, default=0.0)
    g.add_argument("--audio-scale", type=float, default=1.0)
    g.add_argument("--input-video-strength", type=float, default=1.0)
    g.add_argument("--perturbation-layers", type=int, nargs="*", default=[28])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            return _run_wangp_generate(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except subprocess.CalledProcessError as e:
        _eprint(f"[ltx23_cli] External process failed: {e}")
        return int(getattr(e, "returncode", 1) or 1)
    except Exception as e:
        _eprint(f"[ltx23_cli] ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
