import argparse
import os
import sys
import inspect
import subprocess
import re
import importlib
from pathlib import Path

DEFAULT_MODELS = {
    "480p_t2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "720p_t2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "480p_t2v_distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
    "480p_i2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
    "720p_i2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
    "480p_i2v_distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
    "720p_i2v_distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
    "480p_i2v_step_distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
}


def _ffmpeg_exe(root: Path) -> str | None:
    """Best-effort lookup for ffmpeg.exe inside the FrameVision/FrameLab-style tree."""
    candidates = [
        root / "bin" / "ffmpeg.exe",
        root / "ffmpeg.exe",
        root / "presets" / "bin" / "ffmpeg.exe",
        root / "presets" / "ffmpeg.exe",
        root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "presets" / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _reencode_h264_mp4(ffmpeg: str, src: Path, dst: Path, bitrate_kbps: int) -> None:
    """Re-encode MP4 using ffmpeg with a (near) CBR target bitrate."""
    br = max(250, int(bitrate_kbps))
    buf = max(500, int(br) * 2)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-b:v",
        f"{br}k",
        "-minrate",
        f"{br}k",
        "-maxrate",
        f"{br}k",
        "-bufsize",
        f"{buf}k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _supports_kw(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False


def _encode_prompt_embeds_cpu(pipe, torch, prompt: str, negative: str = ""):
    """Encode prompt (and negative) on CPU.

    Returns a 4-tuple:
        (prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask)

    If we can't reliably obtain the required masks, returns None so the caller can fall back to normal prompt encoding.
    """
    try:
        if not hasattr(pipe, "encode_prompt"):
            return None
        enc_fn = getattr(pipe, "encode_prompt")

        kw = {}
        if _supports_kw(enc_fn, "device"):
            kw["device"] = "cpu"
        if _supports_kw(enc_fn, "do_classifier_free_guidance"):
            kw["do_classifier_free_guidance"] = True
        if _supports_kw(enc_fn, "negative_prompt"):
            kw["negative_prompt"] = negative if negative is not None else ""

        if _supports_kw(enc_fn, "num_videos_per_prompt"):
            kw["num_videos_per_prompt"] = 1
        elif _supports_kw(enc_fn, "num_images_per_prompt"):
            kw["num_images_per_prompt"] = 1

        want_dict = _supports_kw(enc_fn, "return_dict")
        if want_dict:
            kw["return_dict"] = True

        try:
            out = enc_fn(prompt=prompt, **kw)
        except TypeError:
            try:
                if "device" in kw:
                    out = enc_fn(prompt, kw["device"])
                else:
                    out = enc_fn(prompt)
            except Exception:
                out = enc_fn(prompt=prompt)

        prompt_embeds = None
        negative_embeds = None
        prompt_mask = None
        negative_mask = None

        if want_dict and not isinstance(out, (tuple, list)):
            def _get(o, name: str):
                if isinstance(o, dict):
                    return o.get(name)
                return getattr(o, name, None)

            prompt_embeds = _get(out, "prompt_embeds")
            negative_embeds = _get(out, "negative_prompt_embeds")
            prompt_mask = _get(out, "prompt_embeds_mask") or _get(out, "prompt_attention_mask")
            negative_mask = _get(out, "negative_prompt_embeds_mask") or _get(out, "negative_prompt_attention_mask")
        else:
            # Tuple/list forms usually don't include masks; treat as unsupported for HunyuanVideo 1.5 prompt_embeds path.
            if isinstance(out, (tuple, list)):
                if len(out) >= 1:
                    prompt_embeds = out[0]
                if len(out) >= 2:
                    negative_embeds = out[1]

        if prompt_embeds is None or prompt_mask is None:
            return None
        if negative_embeds is not None and negative_mask is None:
            return None

        try:
            prompt_embeds = prompt_embeds.to("cpu")
        except Exception:
            pass
        try:
            prompt_mask = prompt_mask.to("cpu")
        except Exception:
            pass
        if negative_embeds is not None:
            try:
                negative_embeds = negative_embeds.to("cpu")
            except Exception:
                pass
        if negative_mask is not None:
            try:
                negative_mask = negative_mask.to("cpu")
            except Exception:
                pass

        return (prompt_embeds, negative_embeds, prompt_mask, negative_mask)
    except Exception:
        return None


def _from_pretrained(pipe_cls, source: str, dtype):
    """Compatibility shim for diffusers deprecation: prefer dtype= over torch_dtype=."""
    fp = getattr(pipe_cls, "from_pretrained")
    if _supports_kw(fp, "dtype"):
        return fp(source, dtype=dtype)
    return fp(source, torch_dtype=dtype)

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_cuda_or_die():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project is NVIDIA CUDA-only.")
    return torch

def set_hf_home(root: Path):
    # Keep all HF caches in ./models/hf_cache
    hf_home = root / "models" / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "info")

def _ensure_model_download(model_id: str, out_dir: Path) -> None:
    """Download model repo to out_dir with visible per-file logging.

    snapshot_download can be silent in non-TTY runs (like QProcess). This implementation
    prints progress per file so the UI never looks stuck.
    """
    try:
        if out_dir.exists() and any(out_dir.iterdir()):
            return
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] missing locally; downloading to: {out_dir}", flush=True)
    print("[download] note: large models can take a while. You'll see file-by-file progress here.", flush=True)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        print(f"[download] ERROR importing huggingface_hub: {e}", flush=True)
        raise

    api = HfApi()
    files: list[tuple[str, int | None]] = []
    total_bytes = 0

    # Prefer sizes from model_info (if available) for a % estimate.
    try:
        info = api.model_info(model_id)
        siblings = getattr(info, "siblings", []) or []
        for s in siblings:
            fn = getattr(s, "rfilename", None) or getattr(s, "path", None)
            sz = getattr(s, "size", None)
            if fn:
                files.append((str(fn), int(sz) if isinstance(sz, int) else None))
        total_bytes = sum(sz for _, sz in files if isinstance(sz, int))
    except Exception:
        files = []

    # Fallback: just list filenames (no sizes).
    if not files:
        try:
            for fn in api.list_repo_files(model_id):
                files.append((str(fn), None))
        except Exception as e:
            print(f"[download] ERROR listing repo files: {e}", flush=True)
            raise

    done_bytes = 0
    n = len(files)
    for i, (fn, sz) in enumerate(files, 1):
        print(f"[download] {i}/{n}: {fn}", flush=True)
        # hf_hub_download handles LFS blobs and resumes automatically.
        hf_hub_download(
            repo_id=model_id,
            filename=fn,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if isinstance(sz, int):
            done_bytes += sz
            if total_bytes > 0:
                pct = (done_bytes / total_bytes) * 100.0
                print(f"[download] progress: {pct:.1f}%", flush=True)
    print("[download] done", flush=True)

def cmd_download(args):
    root = project_root()
    set_hf_home(root)

    model_key = args.model
    model_id = DEFAULT_MODELS.get(model_key, model_key)

    out_dir = root / "models" / model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] model: {model_id}", flush=True)
    print(f"[download] local_dir: {out_dir}", flush=True)
    _ensure_model_download(model_id, out_dir)
    print("[download] done", flush=True)


def _ensure_snapshot_download(model_id: str, out_dir: Path):
    """Download (or resume) a model repo into out_dir with visible logs.

    Kept for backward compatibility: we implement per-file logging so QProcess runs
    don't look stuck.
    """
    _ensure_model_download(model_id, out_dir)

def _load_and_fit_image(image_path: str, target_w: int | None, target_h: int | None):
    """Load a start image and fit it to target size using center-crop to aspect."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if not target_w or not target_h:
        return img
    # center-crop to target aspect ratio, then resize
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return img
    target_ar = float(target_w) / float(target_h)
    in_ar = float(iw) / float(ih)
    if in_ar > target_ar:
        # crop width
        new_w = int(round(ih * target_ar))
        left = max(0, (iw - new_w) // 2)
        img = img.crop((left, 0, left + new_w, ih))
    else:
        # crop height
        new_h = int(round(iw / target_ar))
        top = max(0, (ih - new_h) // 2)
        img = img.crop((0, top, iw, top + new_h))
    img = img.resize((int(target_w), int(target_h)), resample=Image.LANCZOS)
    return img

def _pick_dtype(torch):
    # Prefer bfloat16 on Ampere+ when available, otherwise fp16.
    # torch.cuda.is_bf16_supported exists on newer torch; keep it safe.
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.bfloat16 if bf16_ok else torch.float16

def _flash_attn_available():
    """Return (available, version_str) for flash_attn (if installed in this env)."""
    try:
        import flash_attn  # type: ignore
        ver = getattr(flash_attn, "__version__", "")
        return True, str(ver) if ver is not None else ""
    except Exception:
        return False, ""

def _set_attention_backend(pipe, backend: str):
    """Set Diffusers attention backend.

    Notes:
      - `flash_hub` / `flash_varlen_hub` / `sage_hub` are *kernels Hub* backends (require `kernels`
        and an installable kernel for your platform).
      - `flash` is the native FlashAttention backend (requires `flash_attn` package).
      - On Windows, Hub kernels may not exist; if Hub selection fails but `flash_attn` is installed,
        we fall back to `flash`.
    """
    have_flash, _flash_ver = _flash_attn_available()

    def _try(b: str) -> bool:
        try:
            pipe.transformer.set_attention_backend(b)
            return True
        except Exception:
            return False

    if backend == "auto":
        # Prefer kernels Hub backends first (if available), then native FlashAttention, then SDPA.
        for b in ("flash_hub", "flash_varlen_hub", "sage_hub"):
            if _try(b):
                return b
        if have_flash and _try("flash"):
            return "flash"
        if _try("sdpa"):
            return "sdpa"
        return "default"

    # If user explicitly selected a Hub backend, try it first, then fall back.
    if backend in ("flash_hub", "flash_varlen_hub", "_flash_2_hub", "_flash_3_hub"):
        if _try(backend):
            return backend
        if have_flash and _try("flash"):
            return "flash"
        if _try("sdpa"):
            return "sdpa"
        return "default"

    if backend and backend != "default":
        if _try(backend):
            return backend
        return "default"

    return "default"

def cmd_generate(args):
    root = project_root()
    set_hf_home(root)

    torch = ensure_cuda_or_die()
    from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
    from diffusers.utils import export_to_video

    model_key = args.model
    model_id = DEFAULT_MODELS.get(model_key, model_key)
    local_dir = root / "models" / model_id.replace("/", "_")

    # Detect I2V vs T2V based on the resolved repo id / key.
    model_is_i2v = ("_i2v" in str(model_id).lower()) or ("i2v" in str(model_key).lower())
    image_path = (getattr(args, "image", "") or "").strip()

    if image_path and (not model_is_i2v):
        raise RuntimeError("Start image was provided but the selected model is T2V. Select an *_i2v model.")
    if model_is_i2v and (not image_path):
        raise RuntimeError("An *_i2v model was selected but no --image was provided.")

    # Ensure the model is downloaded into ./models on first use (shows progress).
    _ensure_snapshot_download(model_id, local_dir)


    dtype = _pick_dtype(torch)
    print(f"[gen] dtype: {dtype}", flush=True)
    print(f"[gen] model: {model_id}", flush=True)
    print(f"[gen] local_dir: {local_dir if local_dir.exists() else '(HF cache)'}", flush=True)

    # Prefer local_dir if user downloaded; otherwise let HF handle it.
    PipeCls = HunyuanVideo15ImageToVideoPipeline if model_is_i2v else HunyuanVideo15Pipeline
    pipe = _from_pretrained(PipeCls, str(local_dir), dtype)

    # I2V resolution override:
    # Diffusers' HunyuanVideo15ImageToVideoPipeline does not accept height/width kwargs.
    # Instead it uses pipe.target_size to decide the internal working resolution.
    if model_is_i2v and (args.height or args.width):
        req_h = int(args.height) if args.height else None
        req_w = int(args.width) if args.width else None

        # target_size bucket override:
        # - If --target-size is provided, it wins (useful for auto-aspect portrait/square).
        # - Otherwise fall back to the old behavior: derive from min(requested_w, requested_h).
        req = int(getattr(args, "target_size", 0) or 0)
        if req <= 0:
            req = min(v for v in (req_h, req_w) if v is not None)

        try:
            vae_sf = int(getattr(pipe, "vae_scale_factor_spatial", 16) or 16)
        except Exception:
            vae_sf = 16

        # Align to the VAE spatial compression factor (usually 16) to avoid shape errors.
        req_aligned = max(vae_sf, (req // vae_sf) * vae_sf)
        try:
            pipe.target_size = int(req_aligned)
            src = "target_size" if getattr(args, "target_size", None) else "min side"
            print(f"[i2v] override target_size: {pipe.target_size} (source={src}, req={req})", flush=True)
        except Exception as e:
            print(f"[i2v] WARNING: could not override target_size: {e}", flush=True)

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    # Optional: Diffusers hook toggles (advanced / experimental)
    # Group Offload can conflict with enable_model_cpu_offload, so we skip if --offload is enabled.
    if getattr(args, "group_offload", False):
        if getattr(args, "offload", False):
            print("[gen] group offload: skipped (model CPU offload is enabled)", flush=True)
        else:
            try:
                try:
                    from diffusers.hooks import apply_group_offloading
                except Exception:
                    from diffusers.hooks.group_offloading import apply_group_offloading  # type: ignore
                target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None) or pipe
                apply_group_offloading(
                    target,
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    use_stream=False,
                )
                print("[gen] group offload: enabled", flush=True)
            except Exception as e:
                print(f"[gen] group offload: failed ({e})", flush=True)

    if getattr(args, "first_block_cache", False):
        try:
            thr = float(getattr(args, "first_block_cache_threshold", 0.05) or 0.05)
        except Exception:
            thr = 0.05
        if thr < 0.0:
            thr = 0.0
        if thr > 0.5:
            thr = 0.5
        try:
            try:
                from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
            except Exception:
                from diffusers.hooks.first_block_cache import apply_first_block_cache, FirstBlockCacheConfig  # type: ignore
            target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None) or pipe
            cfg = FirstBlockCacheConfig(threshold=float(thr))
            apply_first_block_cache(target, cfg)
            print(f"[gen] first block cache: enabled (threshold={thr:.3f})", flush=True)
        except Exception as e:
            # Some diffusers builds don't register Hunyuan block classes for HookRegistry.
            # Try a best-effort runtime registration and retry once.
            msg = str(e)
            retried = False
            if "not registered" in msg or "not registered." in msg:
                try:
                    # Pull the class path out of the error, then patch common registries.
                    cls = None
                    mm = re.search(r"<class '([^']+)'>", msg)
                    if mm:
                        mod_name, cls_name = mm.group(1).rsplit(".", 1)
                        mod = importlib.import_module(mod_name)
                        cls = getattr(mod, cls_name, None)

                    if cls is not None:
                        try:
                            import diffusers.hooks.hooks as dh  # type: ignore
                        except Exception:
                            dh = None

                        try:
                            from diffusers.hooks import HookRegistry  # type: ignore
                        except Exception:
                            HookRegistry = None  # type: ignore

                        if dh is not None and HookRegistry is not None:
                            # Dict-based registries (class -> registry type)
                            for name in (
                                "MODEL_CLASS_TO_HOOK_REGISTRY",
                                "_MODEL_CLASS_TO_HOOK_REGISTRY",
                                "HOOK_REGISTRY_BY_MODEL_CLASS",
                                "_HOOK_REGISTRY_BY_MODEL_CLASS",
                                "MODEL_HOOK_REGISTRY",
                                "_MODEL_HOOK_REGISTRY",
                            ):
                                d = getattr(dh, name, None)
                                if isinstance(d, dict):
                                    d[cls] = HookRegistry

                            # Set/list-based registries (supported classes)
                            for name in (
                                "HOOKABLE_MODEL_CLASSES",
                                "_HOOKABLE_MODEL_CLASSES",
                                "SUPPORTED_HOOK_MODELS",
                                "_SUPPORTED_HOOK_MODELS",
                                "SUPPORTED_MODELS",
                                "_SUPPORTED_MODELS",
                            ):
                                s = getattr(dh, name, None)
                                if isinstance(s, set):
                                    s.add(cls)
                                elif isinstance(s, list):
                                    if cls not in s:
                                        s.append(cls)

                            # Function-based registration helpers
                            for fn_name in dir(dh):
                                if "register" in fn_name.lower() and "hook" in fn_name.lower():
                                    fn = getattr(dh, fn_name, None)
                                    if callable(fn):
                                        try:
                                            fn(cls)
                                        except Exception:
                                            pass

                        # Retry once
                        apply_first_block_cache(target, cfg)
                        print(f"[gen] first block cache: enabled (threshold={thr:.3f})", flush=True)
                        retried = True
                except Exception:
                    retried = False

            if not retried:
                print(f"[gen] first block cache: failed ({e})", flush=True)

    pab_enabled = False
    pab_step_end_cb = None
    pab_simple_cb = None
    if getattr(args, "pyramid_attn_broadcast", False):
        try:
            try:
                from diffusers.hooks import apply_pyramid_attention_broadcast, PyramidAttentionBroadcastConfig
            except Exception:
                from diffusers.hooks.pyramid_attention_broadcast import apply_pyramid_attention_broadcast, PyramidAttentionBroadcastConfig  # type: ignore

            # Provide a stable place for hooks to read the current timestep
            try:
                setattr(pipe, "current_timestep", int(getattr(pipe, "current_timestep", 0) or 0))
            except Exception:
                pass

            def _pab_get_timestep() -> int:
                try:
                    return int(getattr(pipe, "current_timestep", 0) or 0)
                except Exception:
                    return 0

            # Use conservative defaults (enable all 3 attention types).
            cfg = PyramidAttentionBroadcastConfig(
                spatial_attention_block_skip_range=2,
                temporal_attention_block_skip_range=2,
                cross_attention_block_skip_range=3,
                current_timestep_callback=_pab_get_timestep,
            )
            target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None) or pipe
            apply_pyramid_attention_broadcast(target, cfg)
            pab_enabled = True

            # If the pipeline supports callbacks, we update pipe.current_timestep each step.
            if _supports_kw(pipe.__call__, "callback_on_step_end"):
                def _pab_on_step_end(p, step, timestep, callback_kwargs):
                    try:
                        t = timestep
                        if hasattr(t, "item"):
                            t = t.item()
                        t = int(t)
                    except Exception:
                        t = 0
                    try:
                        setattr(p, "current_timestep", t)
                    except Exception:
                        pass
                    return callback_kwargs
                pab_step_end_cb = _pab_on_step_end
            elif _supports_kw(pipe.__call__, "callback") and _supports_kw(pipe.__call__, "callback_steps"):
                def _pab_callback(step, timestep, latents=None):
                    try:
                        t = timestep
                        if hasattr(t, "item"):
                            t = t.item()
                        t = int(t)
                    except Exception:
                        t = 0
                    try:
                        setattr(pipe, "current_timestep", t)
                    except Exception:
                        pass
                pab_simple_cb = _pab_callback

            print("[gen] pyramid attention broadcast: enabled", flush=True)
        except Exception as e:
            print(f"[gen] pyramid attention broadcast: failed ({e})", flush=True)

    if getattr(args, "attn_slicing", False):
        try:
            if hasattr(pipe, "enable_attention_slicing"):
                fn = getattr(pipe, "enable_attention_slicing")
                if _supports_kw(fn, "slice_size"):
                    fn(slice_size="auto")
                else:
                    fn()
                print("[gen] attention slicing: enabled", flush=True)
        except Exception:
            pass

    if getattr(args, "vae_slicing", False):
        try:
            pipe.vae.enable_slicing()
            print("[gen] VAE slicing: enabled", flush=True)
        except Exception:
            pass

    if args.tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    chosen_backend = _set_attention_backend(pipe, args.attn)
    print(f"[gen] attention backend: {chosen_backend}", flush=True)
    # If we ended up on native FlashAttention, mention whether flash_attn is present.
    if chosen_backend == "flash":
        ok, ver = _flash_attn_available()
        if ok:
            print(f"[gen] flash_attn package: {ver or 'installed'}", flush=True)

    prompt = args.prompt
    negative = (getattr(args, 'negative', '') or '').strip()
    out_dir = root / "output" / "video" / "hunyuan15"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = out_dir / out_path

    # Optional: encode text on CPU to avoid text-encoder VRAM spikes.
    prompt_embeds = None
    negative_prompt_embeds = None
    prompt_embeds_mask = None
    negative_prompt_embeds_mask = None
    if getattr(args, "cpu_text_encode", False):
        try:
            enc = _encode_prompt_embeds_cpu(pipe, torch, prompt, negative=negative)
            # This pipeline requires prompt_embeds_mask when using prompt_embeds.
            if enc and _supports_kw(pipe.__call__, "prompt_embeds") and _supports_kw(pipe.__call__, "prompt_embeds_mask"):
                prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask = enc

                # Move embeddings/masks to CUDA for the actual generation run (small VRAM cost vs. text encoder weights).
                try:
                    prompt_embeds = prompt_embeds.to("cuda")
                except Exception:
                    pass
                try:
                    prompt_embeds_mask = prompt_embeds_mask.to("cuda")
                except Exception:
                    pass

                if negative_prompt_embeds is not None:
                    try:
                        negative_prompt_embeds = negative_prompt_embeds.to("cuda")
                    except Exception:
                        pass
                if negative_prompt_embeds_mask is not None:
                    try:
                        negative_prompt_embeds_mask = negative_prompt_embeds_mask.to("cuda")
                    except Exception:
                        pass

                print("[gen] cpu-text-encode: enabled (using prompt_embeds + prompt_embeds_mask)", flush=True)
            else:
                print("[gen] cpu-text-encode: requested but unsupported; falling back to normal prompt encoding", flush=True)
                prompt_embeds = None
                negative_prompt_embeds = None
                prompt_embeds_mask = None
                negative_prompt_embeds_mask = None
        except Exception:
            print("[gen] cpu-text-encode: failed; falling back to normal prompt encoding", flush=True)
            prompt_embeds = None
            negative_prompt_embeds = None
            prompt_embeds_mask = None
            negative_prompt_embeds_mask = None

    gen = None

    if args.seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)

    
    # Defaults follow the Diffusers docs example
    use_embeds = (
        prompt_embeds is not None
        and prompt_embeds_mask is not None
        and _supports_kw(pipe.__call__, "prompt_embeds")
        and _supports_kw(pipe.__call__, "prompt_embeds_mask")
    )

    if use_embeds:
        kwargs = dict(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            generator=gen,
        )
        if negative_prompt_embeds is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds"):
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        if negative_prompt_embeds_mask is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask"):
            kwargs["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
    else:
        kwargs = dict(
            prompt=prompt,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            generator=gen,
        )
        if negative and _supports_kw(pipe.__call__, "negative_prompt"):
            kwargs["negative_prompt"] = negative

    if args.height and _supports_kw(pipe.__call__, "height"):
        kwargs["height"] = args.height
    if args.width and _supports_kw(pipe.__call__, "width"):
        kwargs["width"] = args.width

    # Pyramid Attention Broadcast needs the current timestep during inference.
    if pab_enabled:
        if pab_step_end_cb is not None and _supports_kw(pipe.__call__, "callback_on_step_end"):
            kwargs["callback_on_step_end"] = pab_step_end_cb
            if _supports_kw(pipe.__call__, "callback_on_step_end_tensor_inputs"):
                kwargs["callback_on_step_end_tensor_inputs"] = []
        elif pab_simple_cb is not None and _supports_kw(pipe.__call__, "callback") and _supports_kw(pipe.__call__, "callback_steps"):
            kwargs["callback"] = pab_simple_cb
            kwargs["callback_steps"] = 1


    # I2V: provide start image
    if model_is_i2v:
        init_img = _load_and_fit_image(image_path, args.width, args.height)
        try:
            eff_h, eff_w = pipe.video_processor.calculate_default_height_width(
                height=init_img.size[1], width=init_img.size[0], target_size=getattr(pipe, "target_size", None)
            )
            print(f"[i2v] effective resolution: {eff_w}x{eff_h} (target_size={getattr(pipe, 'target_size', None)})", flush=True)
        except Exception:
            pass
        if _supports_kw(pipe.__call__, "image"):
            kwargs["image"] = init_img
        elif _supports_kw(pipe.__call__, "images"):
            kwargs["images"] = init_img
        else:
            raise RuntimeError("This pipeline does not accept an image input (no 'image'/'images' kwarg).")

    print(f"[gen] running: frames={args.frames} steps={args.steps}", flush=True)
    video = pipe(**kwargs).frames[0]

    # Diffusers' export_to_video uses a very conservative default encode, which can look low-bitrate.
    # We optionally re-encode with ffmpeg for a predictable bitrate.
    br = int(getattr(args, "bitrate_kbps", 0) or 0)
    if br > 0:
        tmp = out_path.with_name(f"{out_path.stem}__raw{out_path.suffix or '.mp4'}")
        export_to_video(video, str(tmp), fps=args.fps)

        ff = _ffmpeg_exe(root)
        if ff:
            print(f"[gen] re-encode: {br} kbps (h264)", flush=True)
            try:
                _reencode_h264_mp4(ff, tmp, out_path, br)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        else:
            # If ffmpeg is missing, fall back to the raw export.
            print("[gen] WARNING: ffmpeg.exe not found; keeping raw export (bitrate may be low).", flush=True)
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            try:
                tmp.replace(out_path)
            except Exception:
                # last resort: copy bytes
                out_path.write_bytes(tmp.read_bytes())
                try:
                    tmp.unlink()
                except Exception:
                    pass
    else:
        export_to_video(video, str(out_path), fps=args.fps)

    print(f"[gen] saved: {out_path}", flush=True)

def main():
    p = argparse.ArgumentParser(prog="hunyuan15_cli", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download model weights into ./models/")
    d.add_argument("--model", default="480p_t2v", help="Model key or HF repo id")
    d.set_defaults(fn=cmd_download)

    g = sub.add_parser("generate", help="Generate a short video")
    g.add_argument("--model", default="480p_t2v", help="Model key or HF repo id")
    g.add_argument("--prompt", required=True, help="Text prompt")
    g.add_argument("--negative", default="", help="Negative prompt (optional)")
    g.add_argument("--image", default="", help="(I2V) Path to a start image. Requires an *_i2v model.")
    g.add_argument("--output", default="hunyuan15_output.mp4", help="Output filename (saved to ./output/video/hunyuan15/) or full path")
    g.add_argument("--frames", type=int, default=61)
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--fps", type=int, default=15)
    g.add_argument("--bitrate-kbps", type=int, default=2000, help="Target H.264 bitrate in kbps for the final MP4 (0=disable re-encode)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--height", type=int, default=0)
    g.add_argument("--width", type=int, default=0)
    g.add_argument("--auto-aspect", action="store_true", help="Use input image aspect ratio (no fit/crop); ignore --width/--height overrides")
    g.add_argument("--target-size", type=int, default=0, help="(Auto aspect / I2V) Override pipeline target_size bucket (single number). 0=use default")
    g.add_argument("--offload", action="store_true", help="Enable model CPU offloading (lower VRAM)")
    g.add_argument("--group-offload", action="store_true", help="Enable Group Offload (experimental; lower VRAM, may slow)")
    g.add_argument("--first-block-cache", action="store_true", help="Enable FirstBlockCache (experimental speedup)")
    g.add_argument("--first-block-cache-threshold", type=float, default=0.05, help="FirstBlockCache threshold (0.000-0.500). Only used when --first-block-cache is set.")
    g.add_argument("--pyramid-attn-broadcast", action="store_true", help="Enable Pyramid Attention Broadcast (experimental speedup)")

    g.add_argument("--tiling", action="store_true", help="Enable VAE tiling (lower VRAM)")
    g.add_argument("--attn-slicing", action="store_true", help="Enable attention slicing (lower VRAM, slower)")
    g.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing (lower VRAM, slower)")
    g.add_argument("--cpu-text-encode", action="store_true", help="Encode prompt on CPU and pass embeddings to avoid text-encoder VRAM spikes")
    g.add_argument("--attn", default="auto", help="Attention backend: auto|flash_hub|flash_varlen_hub|sage_hub|flash|sdpa|default")
    g.set_defaults(fn=cmd_generate)

    args = p.parse_args()

    # normalize height/width
    if args.cmd == "generate":
        if args.height == 0: args.height = None
        if args.width == 0: args.width = None
        if getattr(args, 'target_size', 0) == 0: args.target_size = None

    try:
        args.fn(args)
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())