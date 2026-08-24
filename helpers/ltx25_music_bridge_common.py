"""Shared Music Clip Creator adapter for native LTX 2.5 backends.

The existing Music Clip Creator planning/review/assembly contract lives in
``clip2ltx_cli.py``.  This adapter deliberately reuses that mature planner and
assembly code, but replaces only the final LTX generation command builder.
That keeps scene plans, audio chunks, resolution/aspect settings, trimming,
review/recreate, queue behavior and final assembly identical to the existing
LTX workflow while the actual clip renderer is LTX 2.5.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent if HERE.name.lower() == "helpers" else HERE
BASE_PATH = HERE / "clip2ltx_cli.py"
LTX25_HELPER = HERE / "ltx25_helper.py"
LTX25_CONVROT_WORKER = HERE / "ltx25_convrot_worker.py"
LTX25_MSR_PIPELINE = HERE / "ltx25_msr_pipeline.py"

FP16_ENV = APP_ROOT / "environments" / "ltx25" / "Scripts" / "python.exe"
CONVROT_ENV = APP_ROOT / "environments" / "ltx25_convrot" / "Scripts" / "python.exe"
FP16_MODELS = APP_ROOT / "models" / "ltx-2.5"
CONVROT_MODELS = APP_ROOT / "models" / "ltx_2_5_convrot"
MSR_MODEL = FP16_MODELS / "msr" / "LTX-2.5-Licon-MSR-V1.safetensors"


def _load_base(tag: str):
    if not BASE_PATH.is_file():
        raise FileNotFoundError(f"Music Clip Creator base bridge is missing: {BASE_PATH}")
    name = f"_framevision_ltx25_{tag}_base_{abs(hash(str(BASE_PATH)))}"
    spec = importlib.util.spec_from_file_location(name, str(BASE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bridge module spec: {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _paths(mode: str) -> dict[str, Path]:
    if mode == "convrot":
        root = CONVROT_MODELS
        return {
            "transformer": root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors",
            "text_encoder": root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors",
            "video_vae": root / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
            "audio_vae": root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
            "upsampler": root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        }
    root = FP16_MODELS
    return {
        "transformer": root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder": root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": root / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
        "audio_vae": root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
        "upsampler": root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    }


def install_status(mode: str) -> dict[str, Any]:
    paths = _paths(mode)
    env_python = CONVROT_ENV if mode == "convrot" else FP16_ENV
    helper = LTX25_CONVROT_WORKER if mode == "convrot" else LTX25_HELPER
    missing = []
    if not BASE_PATH.is_file():
        missing.append(str(BASE_PATH))
    if not env_python.is_file():
        missing.append(str(env_python))
    if not helper.is_file():
        missing.append(str(helper))
    for p in paths.values():
        if not p.is_file():
            missing.append(str(p))
    if mode == "convrot":
        comfy = CONVROT_MODELS / "ComfyUI" / "comfy" / "sd.py"
        if not comfy.is_file():
            missing.append(str(comfy))
    return {
        "ok": not missing,
        "mode": mode,
        "missing": missing,
        "message": "LTX 2.5 backend ready." if not missing else "Missing: " + " | ".join(missing),
    }


def _option(cmd: list[str], *names: str, default: str = "") -> str:
    for i, token in enumerate(cmd):
        if token in names and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return default


def _images_from_command(cmd: list[str]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    i = 0
    while i < len(cmd):
        if cmd[i] != "--image":
            i += 1
            continue
        if i + 3 >= len(cmd):
            break
        path = str(cmd[i + 1])
        try:
            frame_idx = int(float(cmd[i + 2]))
        except Exception:
            frame_idx = 0
        try:
            strength = float(cmd[i + 3])
        except Exception:
            strength = 1.0
        if path and Path(path).expanduser().is_file():
            images.append({"path": str(Path(path).expanduser().resolve()), "frame_idx": frame_idx, "strength": strength})
        # Old LTX 2.3 ImageConditioningAction sometimes has a fourth CRF value.
        i += 4
        if i < len(cmd) and not str(cmd[i]).startswith("--"):
            try:
                float(cmd[i])
                i += 1
            except Exception:
                pass
    return images


def _iter_objects(value: Any, depth: int = 0, seen: set[int] | None = None):
    if depth > 5:
        return
    if seen is None:
        seen = set()
    try:
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
    except Exception:
        pass
    yield value
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_objects(v, depth + 1, seen)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _iter_objects(v, depth + 1, seen)
    elif hasattr(value, "__dict__"):
        try:
            yield from _iter_objects(vars(value), depth + 1, seen)
        except Exception:
            pass


def _find_named_value(values: Iterable[Any], names: Iterable[str]) -> Any:
    wanted = {str(n) for n in names}
    for root in values:
        for obj in _iter_objects(root):
            if isinstance(obj, dict):
                for key in wanted:
                    if key in obj and obj[key] not in (None, "", [], ()):
                        return obj[key]
    return None


def _image_paths(value: Any) -> list[str]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    result = []
    seen = set()
    for item in raw:
        if isinstance(item, dict):
            item = item.get("path") or item.get("image") or item.get("source") or ""
        try:
            p = Path(str(item or "").strip().strip('"')).expanduser()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                rp = str(p.resolve())
                key = os.path.normcase(rp)
                if key not in seen:
                    seen.add(key)
                    result.append(rp)
        except Exception:
            pass
    return result


def _extract_sources_from_json(obj: Any) -> tuple[list[str], str]:
    refs: list[str] = []
    background = ""
    seen = set()

    def walk(value: Any, key_hint: str = ""):
        nonlocal background
        if isinstance(value, dict):
            path_value = value.get("path") or value.get("source_path") or value.get("source") or value.get("file")
            role = str(value.get("role") or value.get("label") or value.get("type") or key_hint or "").lower()
            if path_value:
                paths = _image_paths(path_value)
                for p in paths:
                    if "background" in role or role in {"bg", "scene"}:
                        if not background:
                            background = p
                    elif p not in seen:
                        seen.add(p)
                        refs.append(p)
            for k, v in value.items():
                if str(k) in {"path", "source_path", "source", "file"}:
                    continue
                walk(v, str(k))
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v, key_hint)
        elif isinstance(value, str):
            paths = _image_paths(value)
            for p in paths:
                low = key_hint.lower()
                if "background" in low or low == "bg":
                    if not background:
                        background = p
                elif p not in seen:
                    seen.add(p)
                    refs.append(p)

    walk(obj)
    return refs[:4], background


def _sources_from_reference_video(cmd: list[str]) -> tuple[list[str], str]:
    ref_video = _option(cmd, "--video-conditioning", default="")
    if not ref_video:
        return [], ""
    try:
        rp = Path(ref_video).expanduser().resolve()
        folder = rp.parent
    except Exception:
        return [], ""
    candidates = []
    for pattern in ("*.json", "**/*.json"):
        try:
            candidates.extend(folder.glob(pattern))
        except Exception:
            pass
    try:
        candidates = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: abs(p.stat().st_mtime - rp.stat().st_mtime))
    except Exception:
        pass
    for meta in candidates[:12]:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            refs, bg = _extract_sources_from_json(data)
            if refs:
                return refs, bg
        except Exception:
            continue
    return [], ""


def _msr_sources(cmd: list[str], call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> tuple[list[str], str]:
    # First prefer the exact per-shot metadata created by the existing MSR builder.
    refs, bg = _sources_from_reference_video(cmd)
    if refs:
        return refs[:4], bg
    roots: list[Any] = list(call_args) + [call_kwargs]
    refs = _image_paths(_find_named_value(roots, ("msr_reference_paths", "msr_refs", "reference_paths")))[:4]
    backgrounds = _image_paths(_find_named_value(roots, ("msr_background_paths", "msr_background", "background_paths")))
    return refs, (backgrounds[0] if backgrounds else "")


def _safe_int(text: str, default: int) -> int:
    try:
        return int(float(text))
    except Exception:
        return int(default)


def _safe_float(text: str, default: float) -> float:
    try:
        return float(text)
    except Exception:
        return float(default)


def _write_job(mode: str, old_cmd: list[str], call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    paths = _paths(mode)
    prompt = _option(old_cmd, "--prompt", default="").strip()
    output = _option(old_cmd, "--output-path", "--output", default="").strip()
    audio = _option(old_cmd, "--audio-path", default="").strip()
    width = _safe_int(_option(old_cmd, "--width", default="832"), 832)
    height = _safe_int(_option(old_cmd, "--height", default="512"), 512)
    frames = _safe_int(_option(old_cmd, "--num-frames", "--frames", default="241"), 241)
    fps = _safe_float(_option(old_cmd, "--frame-rate", "--fps", default="24"), 24.0)
    seed = _safe_int(_option(old_cmd, "--seed", default="-1"), -1)
    if seed < 0:
        seed = random.randint(0, 2147483647)
    if not output:
        raise RuntimeError("LTX 2.5 adapter could not determine the per-shot output path from the existing workflow.")
    if not prompt:
        raise RuntimeError("LTX 2.5 adapter could not determine the per-shot video prompt from the existing workflow.")
    if not audio:
        audio_value = _find_named_value(list(call_args) + [call_kwargs], ("audio_path", "audio_chunk_path", "audio_file"))
        if audio_value:
            audio = str(audio_value)
    if audio:
        try:
            audio = str(Path(audio).expanduser().resolve())
        except Exception:
            pass

    # The existing 2.3 planner has already resolved 480p/720p + landscape/portrait
    # into exact width/height values. Reusing those values is what makes every
    # Music Clip Creator resolution/aspect setting carry across to 2.5.
    job = {
        "cmd": "generate",
        "model_type": "W4A8 ConvRot (recommended)" if mode == "convrot" else "Full FP16 / BF16",
        "prompt": prompt,
        "workflow": "two_phase",
        "audio_path": audio,
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "output": str(Path(output).expanduser()),
        "images": _images_from_command(old_cmd),
        "paths": {k: str(v) for k, v in paths.items()},
        "offload": "cpu",
        "quantization": "none" if mode == "convrot" else "fp8-cast",
        "max_batch_size": 1,
        "use_sage_attention": False,
        "use_int8_transformer": False,
        "int8_transformer_bundle": "",
        "use_int8_text_encoder": False,
        "int8_text_encoder_bundle": "",
        "enhance_prompt": False,
        "defer_trim": True,
        "cache_prompt_embeddings": True,
    }
    payload_dir = APP_ROOT / "temp" / "musicclip_ltx25_jobs"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"musicclip_ltx25_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{seed}.json"
    payload_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload_path, job


def _msr_command(old_cmd: list[str], call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> list[str]:
    if not FP16_ENV.is_file():
        raise RuntimeError(f"LTX 2.5 FP16 environment Python is missing: {FP16_ENV}")
    if not LTX25_MSR_PIPELINE.is_file():
        raise RuntimeError(f"LTX 2.5 MSR pipeline helper is missing: {LTX25_MSR_PIPELINE}")
    if not MSR_MODEL.is_file():
        raise RuntimeError(f"LTX 2.5 MSR model is missing: {MSR_MODEL}. Run the LTX 2.5 MSR installer first.")
    refs, background = _msr_sources(old_cmd, call_args, call_kwargs)
    if not refs:
        raise RuntimeError(
            "LTX 2.5 MSR was selected, but the adapter could not recover the individual reference images "
            "from this shot. Keep the normal MSR reference lists populated and recreate the director plan."
        )
    p = _paths("fp16")
    prompt = _option(old_cmd, "--prompt", default="").strip()
    output = _option(old_cmd, "--output-path", "--output", default="").strip()
    audio = _option(old_cmd, "--audio-path", default="").strip()
    width = _safe_int(_option(old_cmd, "--width", default="832"), 832)
    height = _safe_int(_option(old_cmd, "--height", default="512"), 512)
    frames = _safe_int(_option(old_cmd, "--num-frames", "--frames", default="241"), 241)
    fps = _safe_float(_option(old_cmd, "--frame-rate", "--fps", default="24"), 24.0)
    seed = _safe_int(_option(old_cmd, "--seed", default="-1"), -1)
    roots: list[Any] = list(call_args) + [call_kwargs]
    requested_ref_frames = _safe_int(str(_find_named_value(roots, ("msr_reference_frames",)) or 33), 33)
    ref_frames = 25 if requested_ref_frames <= 28 else 33
    strength = _safe_float(str(_find_named_value(roots, ("msr_strength", "msr_reference_strength")) or 1.0), 1.0)

    cmd = [
        str(FP16_ENV), str(LTX25_MSR_PIPELINE),
        "--transformer-path", str(p["transformer"]),
        "--text-encoder-path", str(p["text_encoder"]),
        "--video-vae-path", str(p["video_vae"]),
        "--audio-vae-path", str(p["audio_vae"]),
        "--spatial-upsampler-path", str(p["upsampler"]),
        "--offload", "cpu",
        "--quantization", "fp8-cast",
        "--height", str(height), "--width", str(width),
        "--num-frames", str(frames), "--frame-rate", str(fps),
        "--seed", str(seed), "--prompt", prompt,
        "--output-path", output,
        "--msr-lora-path", str(MSR_MODEL),
        "--msr-reference-strength", str(strength),
        "--msr-reference-frames", str(ref_frames),
        "--msr-tile-size", "256", "--msr-tile-overlap", "64",
    ]
    if audio:
        cmd += ["--audio-path", str(Path(audio).expanduser().resolve())]
    else:
        raise RuntimeError("LTX 2.5 MSR Music Clip Creator requires the shot audio chunk, but no --audio-path was present.")
    for idx, ref in enumerate(refs[:4], 1):
        cmd += [f"--msr-ref-{idx}", ref]
    if background:
        cmd += ["--msr-background", background]
    for image in _images_from_command(old_cmd):
        # Keep optional matching end-frame/start-frame conditions alongside MSR.
        cmd += ["--image", image["path"], str(image["frame_idx"]), str(image["strength"])]
    return cmd


def install_generation_patch(base, mode: str):
    original = getattr(base, "_ltx23_build_vramlab_direct_args", None)
    if not callable(original):
        raise RuntimeError("clip2ltx_cli.py does not expose _ltx23_build_vramlab_direct_args; cannot safely replace only the renderer.")

    def build(*args, **kwargs):
        old_cmd = [str(x) for x in list(original(*args, **kwargs) or [])]
        is_msr = "--video-conditioning" in old_cmd or any("msr" in str(x).lower() and "pipeline" in str(x).lower() for x in old_cmd)
        if is_msr:
            if mode != "fp16":
                raise RuntimeError(
                    "LTX 2.5 ConvRot MSR is not available in this backend yet. "
                    "Select LTX 2.5 FP16 for Licon MSR, or turn MSR off for ConvRot."
                )
            return _msr_command(old_cmd, args, kwargs)

        payload_path, job = _write_job(mode, old_cmd, args, kwargs)
        if mode == "convrot":
            return [str(CONVROT_ENV), str(LTX25_CONVROT_WORKER), "--job", str(payload_path)]
        return [str(FP16_ENV), str(LTX25_HELPER), "--queue-job", str(payload_path)]

    setattr(base, "_ltx23_build_vramlab_direct_args", build)
    return build


def install_settings_patch(base, mode: str):
    original = getattr(base, "_ltx23_vramlab_ui_settings", None)
    if not callable(original):
        return

    def settings(*args, **kwargs):
        try:
            out = original(*args, **kwargs)
            result = dict(out) if isinstance(out, dict) else {}
        except Exception:
            result = {}
        p = _paths(mode)
        # Keep the old bridge's key names populated with valid 2.5 paths.  The
        # command builder itself is replaced, but this avoids preflight rejecting
        # the new selection merely because an old 2.3 checkpoint path is absent.
        result.update({
            "checkpoint_path": str(p["transformer"]),
            "transformer_path": str(p["transformer"]),
            "text_encoder_path": str(p["text_encoder"]),
            "video_vae_path": str(p["video_vae"]),
            "audio_vae_path": str(p["audio_vae"]),
            "spatial_upsampler_path": str(p["upsampler"]),
            "quantization": "none" if mode == "convrot" else "fp8-cast",
            "offload": "cpu",
        })
        return result

    setattr(base, "_ltx23_vramlab_ui_settings", settings)


def export_base_api(namespace: dict[str, Any], base) -> None:
    for name in (
        "export_musicclip_scene_plan", "create_prompt_plan", "create_ltx_shot_plan",
        "create_ltx_director_plan", "apply_ltx_start_end_duration_safety_to_plan",
        "generate_ltx_start_image_for_shot", "load_ltx_review_state",
        "run_single_ltx_shot_test", "run_all_ltx_director_shots", "assemble_ltx_music_video",
    ):
        fn = getattr(base, name, None)
        if callable(fn):
            namespace[name] = fn
