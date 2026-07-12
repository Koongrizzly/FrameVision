from __future__ import annotations

"""FrameVision HiDream batch CLI for Music Clip Creator.

Loads one HiDream model a single time and generates many start images from a
JSON manifest. This is intentionally a one-run batch worker, not a permanent
server. It keeps the model warm for the image phase, writes per-shot logs and
optional payload JSON files, then exits so VRAM is released before LTX video
jobs begin.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

HELPERS_DIR = Path(__file__).resolve().parent
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

import hidream_cli as hcli  # noqa: E402


def _safe_str(value: Any, default: str = "") -> str:
    try:
        text = str(value if value is not None else "").strip()
    except Exception:
        text = ""
    return text if text else str(default or "")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _existing_unique_paths(paths: Any, *, limit: int = 5) -> List[str]:
    out: List[str] = []
    hard_limit = max(1, int(limit or 5))
    for raw in _as_list(paths):
        path = _safe_str(raw).strip().strip('"')
        if not path:
            continue
        try:
            if os.path.isfile(path) and path not in out:
                out.append(path)
        except Exception:
            continue
        if len(out) >= hard_limit:
            break
    return out


def _write_json(path: Any, data: Dict[str, Any]) -> None:
    p = Path(_safe_str(path)).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_debug_payload(job: Dict[str, Any], *, command_summary: Dict[str, Any], model_files: Dict[str, Any], reference_handoff: Dict[str, Any], actual_reference_paths: List[str], output_image: str, error: str = "") -> None:
    payload_path = _safe_str(job.get("payload_json_path") or job.get("payload_path"))
    if not payload_path:
        return
    base = dict(job.get("debug_payload_base") or {})
    if not isinstance(base, dict):
        base = {}
    character_reference_passed = bool(actual_reference_paths)
    skipped_reference_reason = ""
    if not character_reference_passed:
        skipped_reference_reason = _safe_str(
            base.get("skipped_reference_reason")
            or ("selected_reference_paths_not_added_to_hidream_batch" if _as_list(base.get("selected_reference_sheet_paths")) else "")
        )
    image_model_reference_mode = "direct_reference_image" if character_reference_passed else _safe_str(base.get("image_model_reference_mode") or "text_only_reference_not_passed")

    base.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": output_image or _safe_str(base.get("output_path")),
        "command_summary": command_summary,
        "model_files_used": model_files,
        "actual_image_model_reference_paths_passed": list(actual_reference_paths),
        "reference_handoff": dict(reference_handoff),
        "character_reference_passed_to_image_model": character_reference_passed,
        "image_model_reference_mode": image_model_reference_mode,
        "skipped_reference_reason": skipped_reference_reason,
    })
    if error:
        base["error"] = _safe_str(error)
        base["generation_failed"] = True
    else:
        base.pop("error", None)
        base["generation_failed"] = False

    cref = dict(base.get("character_reference") or {}) if isinstance(base.get("character_reference"), dict) else {}
    cref.update({
        "actual_image_model_reference_paths_passed": list(actual_reference_paths),
        "passed_to_model": character_reference_passed,
        "model_reference_mode": image_model_reference_mode,
        "skipped_reference_reason": skipped_reference_reason,
    })
    base["character_reference"] = cref
    _write_json(payload_path, base)


def _log_write(handle, text: str) -> None:
    line = str(text or "")
    try:
        print(line, flush=True)
    except Exception:
        pass
    if handle is not None:
        try:
            handle.write(line + "\n")
            handle.flush()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FrameVision HiDream batch CLI")
    parser.add_argument("--manifest_json", type=str, required=True)
    parser.add_argument("--results_json", type=str, default="")
    parser.add_argument("--device_map", type=str, default="cuda", choices=["cuda", "auto"])
    parser.add_argument("--offload_folder", type=str, default=str(hcli.APP_ROOT / "temp" / "hidream_offload"))
    parser.add_argument("--resolution_mode", choices=["native", "framevision"], default="framevision")
    parser.add_argument("--disable_ref_safe_resolution", action="store_true")
    parser.add_argument("--disable_ref_no_crop", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"HiDream batch manifest was not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = list(raw.get("jobs") or [])
    if not jobs:
        raise RuntimeError("HiDream batch manifest does not contain any jobs.")
    results_path = Path(_safe_str(args.results_json) or str(manifest_path.with_name(manifest_path.stem + "_results.json"))).expanduser().resolve()

    if not hcli.REPO_DIR.exists():
        raise RuntimeError(f"HiDream repo folder was not found: {hcli.REPO_DIR}")
    sys.path.insert(0, str(hcli.REPO_DIR))
    hcli.patch_resolution_picker(args.resolution_mode)

    import models.qwen3_vl_transformers as qwen3_vl_transformers  # type: ignore
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # type: ignore
    import models.pipeline as pipeline_mod  # type: ignore
    from models.pipeline import DEFAULT_TIMESTEPS, generate_image  # type: ignore
    from inference import add_special_tokens, get_tokenizer  # type: ignore

    def parse_timesteps(mode: str):
        if _safe_str(mode) == "dev":
            return DEFAULT_TIMESTEPS
        return None

    model_key = _safe_str(raw.get("model_key"), "dev")
    if model_key not in hcli.MODEL_MAP:
        raise RuntimeError(f"Unknown HiDream model key in batch manifest: {model_key}")
    info = hcli.MODEL_MAP[model_key]
    model_dir = hcli.HIDREAM_ROOT / info["folder"]
    if not model_dir.exists():
        raise RuntimeError(f"Selected HiDream model is not installed: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU mode is not useful for this model.")

    any_refs = any(_existing_unique_paths(job.get("reference_paths") or job.get("ref_images"), limit=5) for job in jobs)
    if any_refs and not args.disable_ref_no_crop:
        hcli._patch_hidream_reference_resize_no_crop(pipeline_mod)

    file_dtype = hcli._dtype_from_safetensors(model_dir)
    weight_dtype = hcli.resolve_weight_dtype(model_dir, str(info.get("weight_dtype", "bf16")))
    compute_dtype = hcli.compute_dtype_from_weight_dtype(weight_dtype)

    print("[HiDream batch] Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("[HiDream batch] Repo:", hcli.REPO_DIR)
    print("[HiDream batch] Model:", model_dir)
    print("[HiDream batch] Model key:", model_key, "-", info.get("label"))
    print("[HiDream batch] Jobs:", len(jobs))
    print("[HiDream batch] Weight dtype:", hcli.dtype_label(weight_dtype))
    print("[HiDream batch] Compute dtype:", hcli.dtype_label(compute_dtype))
    print(f"[HiDream batch] Loading processor and model once ({hcli.dtype_label(weight_dtype)} weights)...")

    processor = hcli.AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = hcli.ensure_processor_chat_template(processor, model_dir, get_tokenizer)
    add_special_tokens(tokenizer)

    load_dtype = compute_dtype if hcli.is_float8_dtype(weight_dtype) else weight_dtype
    load_kwargs: Dict[str, Any] = {"dtype": load_dtype}
    if args.device_map == "auto":
        offload_folder = Path(args.offload_folder).expanduser()
        offload_folder.mkdir(parents=True, exist_ok=True)
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = str(offload_folder)
    else:
        load_kwargs["device_map"] = "cuda"

    fp8_ops = hcli.build_manual_fp8_operations(compute_dtype) if hcli.is_float8_dtype(weight_dtype) else None
    if fp8_ops is not None:
        print("[HiDream batch] Using standalone FP8 manual-cast ops.")
        operations = fp8_ops
        nn_proxy = hcli._TorchNNProxy(qwen3_vl_transformers.nn, operations)
        original_nn = qwen3_vl_transformers.nn
        try:
            qwen3_vl_transformers.nn = nn_proxy
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs).eval()
        finally:
            qwen3_vl_transformers.nn = original_nn
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs).eval()

    if hcli.is_float8_dtype(weight_dtype):
        if fp8_ops is not None:
            fp8_params = sum(1 for param in model.parameters() if param.dtype == weight_dtype)
            if file_dtype != weight_dtype or fp8_params == 0:
                converted = hcli._convert_matrix_params_to_dtype(model, weight_dtype)
                print(f"[HiDream batch] Converted {converted} large tensors to {hcli.dtype_label(weight_dtype)} after load.")
            recast = hcli._fp8_safety_recast(model, compute_dtype)
            if recast:
                print(f"[HiDream batch] Recast {recast} small/bias FP8 tensors to {hcli.dtype_label(compute_dtype)} for stable math.")
            visual_recast = hcli._recast_float8_prefixes(model, ["model.visual."], compute_dtype)
            if visual_recast:
                print(f"[HiDream batch] Recast {visual_recast} vision-path FP8 tensors to {hcli.dtype_label(compute_dtype)} for reference-image compatibility.")
        else:
            recast = hcli._recast_all_float8_params(model, compute_dtype)
            if recast:
                print(f"[HiDream batch] Recast {recast} FP8 tensors to {hcli.dtype_label(compute_dtype)} because the FP8 execution path was unavailable.")

    results: Dict[str, Any] = {
        "ok": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest_json": str(manifest_path),
        "results_json": str(results_path),
        "model_key": model_key,
        "model_dir": str(model_dir),
        "job_count": len(jobs),
        "jobs": [],
        "failed_count": 0,
        "finished_count": 0,
    }

    for idx, job in enumerate(jobs, start=1):
        shot_id = _safe_str(job.get("shot_id") or job.get("job_id") or f"job_{idx:03d}")
        output_path = Path(_safe_str(job.get("output_image"))).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = _safe_str(job.get("prompt"))
        negative_prompt = _safe_str(job.get("negative_prompt"))
        ref_paths = _existing_unique_paths(job.get("reference_paths") or job.get("ref_images"), limit=5)
        requested_ref_paths = [_safe_str(x) for x in _as_list(job.get("reference_paths") or job.get("ref_images")) if _safe_str(x)]
        width = _safe_int(job.get("width"), 1280)
        height = _safe_int(job.get("height"), 720)
        seed = _safe_int(job.get("seed"), -1)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        steps = _safe_int(job.get("steps"), int(info.get("default_steps") or 28))
        guidance_scale = _safe_float(job.get("guidance_scale"), float(info.get("default_guidance") or 0.0))
        shift = _safe_float(job.get("shift"), float(info.get("default_shift") or 1.0))
        scheduler_name = _safe_str(job.get("scheduler_name"), _safe_str(info.get("default_scheduler"), "flash"))
        timesteps_mode = _safe_str(job.get("timesteps"), _safe_str(info.get("default_timesteps"), "none"))
        keep_original_aspect = _safe_bool(job.get("keep_original_aspect"), False)
        noise_scale_start = _safe_float(job.get("noise_scale_start"), 7.5)
        noise_scale_end = _safe_float(job.get("noise_scale_end"), 7.5)
        noise_clip_std = _safe_float(job.get("noise_clip_std"), 2.5)
        ref_count = len(ref_paths)
        reference_safe_active = bool(ref_count and args.resolution_mode == "framevision" and not args.disable_ref_safe_resolution)
        reference_no_crop_active = bool(ref_count and not args.disable_ref_no_crop)
        if args.resolution_mode == "framevision":
            if reference_safe_active:
                selected_w, selected_h = hcli._reference_safe_resolution(width, height, ref_count)
            else:
                selected_w, selected_h = hcli._closest_resolution(width, height)
        else:
            selected_w, selected_h = (width, height)

        log_path = Path(_safe_str(job.get("log_path"))).expanduser().resolve() if _safe_str(job.get("log_path")) else None
        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        started = time.time()
        job_result: Dict[str, Any] = {
            "shot_id": shot_id,
            "job_index": idx,
            "ok": False,
            "output_image": str(output_path),
            "log_path": str(log_path) if log_path is not None else "",
        }
        reference_handoff = {
            "reference_paths_requested": requested_ref_paths,
            "reference_paths_passed": list(ref_paths),
            "reference_arg_name": "batch_manifest.ref_images",
            "reference_arg_source": "hidream_batch_cli_manifest",
            "reference_handoff_supported": bool(ref_paths),
            "reference_handoff_reason": "direct_reference_paths_added_to_batch_job" if ref_paths else "no_valid_reference_paths_to_pass",
        }
        command_summary = {
            "type": "hidream_batch_cli",
            "manifest_json": str(manifest_path),
            "results_json": str(results_path),
            "job_index": idx,
            "job_count": len(jobs),
            "model_key": model_key,
        }
        model_files = {
            "hidream_model_key": model_key,
            "hidream_model_dir": str(model_dir),
            "hidream_batch_cli": str(Path(__file__).resolve()),
            "python": sys.executable,
            "defaults": {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "scheduler_name": scheduler_name,
                "timesteps": timesteps_mode,
                "label": _safe_str(info.get("label") or model_key),
            },
        }
        try:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass
            _log_write(log_handle, "[musicclip bridge] HiDream batch start image generation")
            _log_write(log_handle, f"Batch job: {idx}/{len(jobs)}")
            _log_write(log_handle, f"Shot: {shot_id}")
            _log_write(log_handle, f"Image model: HiDream {_safe_str(info.get('label') or model_key)}")
            _log_write(log_handle, f"Output: {output_path}")
            _log_write(log_handle, f"Requested size: {width}x{height}")
            _log_write(log_handle, f"Selected generation size: {selected_w}x{selected_h}")
            _log_write(log_handle, f"Seed: {seed}")
            _log_write(log_handle, f"Steps: {steps}")
            if ref_count:
                _log_write(log_handle, f"Reference workflow: {ref_count} reference image(s)")
                if reference_safe_active:
                    _log_write(log_handle, "Reference-safe bucket override: enabled")
                if reference_no_crop_active:
                    _log_write(log_handle, "Reference no-crop resize: enabled")
            _log_write(log_handle, "Generating...")
            image = generate_image(
                model=model,
                processor=processor,
                prompt=prompt,
                ref_image_paths=ref_paths,
                height=selected_h,
                width=selected_w,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                shift=shift,
                timesteps_list=parse_timesteps(timesteps_mode),
                scheduler_name=scheduler_name,
                seed=seed,
                noise_scale_start=noise_scale_start,
                noise_scale_end=noise_scale_end,
                noise_clip_std=noise_clip_std,
                keep_original_aspect=keep_original_aspect,
            )
            image.save(output_path)
            dt = time.time() - started
            job_result.update({
                "ok": True,
                "seed": seed,
                "requested_width": width,
                "requested_height": height,
                "selected_width": selected_w,
                "selected_height": selected_h,
                "reference_paths_passed": list(ref_paths),
                "reference_handoff": dict(reference_handoff),
                "command_summary": dict(command_summary),
                "model_files": dict(model_files),
                "generation_time_seconds": round(float(dt), 3),
            })
            results["finished_count"] = _safe_int(results.get("finished_count"), 0) + 1
            _log_write(log_handle, f"Saved: {output_path}")
            try:
                _log_write(log_handle, f"Final image size: {image.size[0]}x{image.size[1]}")
            except Exception:
                pass
            _log_write(log_handle, f"Generation time: {dt:.1f}s")
            _update_debug_payload(job, command_summary=command_summary, model_files=model_files, reference_handoff=reference_handoff, actual_reference_paths=ref_paths, output_image=str(output_path), error="")
            del image
        except Exception as exc:
            msg = f"HiDream batch job failed for {shot_id}: {exc}"
            job_result.update({
                "ok": False,
                "error": str(exc),
                "reference_paths_passed": list(ref_paths),
                "reference_handoff": dict(reference_handoff),
                "command_summary": dict(command_summary),
                "model_files": dict(model_files),
            })
            results["ok"] = False
            results["failed_count"] = _safe_int(results.get("failed_count"), 0) + 1
            _log_write(log_handle, msg)
            _update_debug_payload(job, command_summary=command_summary, model_files=model_files, reference_handoff=reference_handoff, actual_reference_paths=ref_paths, output_image=str(output_path), error=str(exc))
            if args.stop_on_error:
                results["jobs"].append(job_result)
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                _write_json(results_path, results)
                raise
        finally:
            results["jobs"].append(job_result)
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass

    _write_json(results_path, results)

    try:
        del model
        del processor
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not results.get("ok"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
