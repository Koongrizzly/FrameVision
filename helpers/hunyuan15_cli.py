import argparse
import os
import sys
import inspect
import subprocess
import re
import importlib
import time
import gc
from pathlib import Path
from types import SimpleNamespace

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
    """Encode Hunyuan prompt on CPU and return a small dict of tensors.

    Diffusers HunyuanVideo 1.5 builds disagree on names/order. Some current
    builds require `prompt_embeds_2` instead of `prompt_embeds` at __call__ time.
    Returning a dict lets the caller pass whichever names this installed pipeline
    actually accepts instead of accidentally leaving the required embed undefined.
    """
    def _is_tensor(x):
        return hasattr(x, "to") and hasattr(x, "shape") and hasattr(x, "dtype")

    def _rank(x):
        try:
            return len(tuple(x.shape))
        except Exception:
            return 0

    def _dtype_str(x):
        try:
            return str(x.dtype).lower()
        except Exception:
            return ""

    def _looks_mask(x):
        if not _is_tensor(x):
            return False
        ds = _dtype_str(x)
        return _rank(x) <= 2 and ("bool" in ds or "int" in ds or "long" in ds)

    def _looks_embed(x):
        if not _is_tensor(x):
            return False
        ds = _dtype_str(x)
        return _rank(x) >= 3 and ("float" in ds or "bfloat" in ds or "half" in ds)

    def _cpu(x):
        if x is None:
            return None
        try:
            return x.to("cpu")
        except Exception:
            return x

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

        result = {
            "prompt_embeds": None,
            "prompt_embeds_2": None,
            "prompt_embeds_mask": None,
            "prompt_embeds_2_mask": None,
            "prompt_embeds_mask_2": None,
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_2": None,
            "negative_prompt_embeds_mask": None,
            "negative_prompt_embeds_2_mask": None,
            "negative_prompt_embeds_mask_2": None,
        }

        def _get(o, name: str):
            if isinstance(o, dict):
                return o.get(name)
            return getattr(o, name, None)

        if want_dict and not isinstance(out, (tuple, list)):
            for k in list(result.keys()):
                result[k] = _get(out, k)
            # aliases seen across Diffusers builds
            if result["prompt_embeds_mask"] is None:
                result["prompt_embeds_mask"] = _get(out, "prompt_attention_mask")
            if result["prompt_embeds_2_mask"] is None:
                result["prompt_embeds_2_mask"] = _get(out, "prompt_attention_mask_2")
            if result["negative_prompt_embeds_mask"] is None:
                result["negative_prompt_embeds_mask"] = _get(out, "negative_prompt_attention_mask")
            if result["negative_prompt_embeds_2_mask"] is None:
                result["negative_prompt_embeds_2_mask"] = _get(out, "negative_prompt_attention_mask_2")
        elif isinstance(out, (tuple, list)):
            vals = [v for v in list(out) if _is_tensor(v)]
            embeds = [v for v in vals if _looks_embed(v)]
            masks = [v for v in vals if _looks_mask(v)]
            # Current Hunyuan __call__ may require prompt_embeds_2. The second
            # floating 3D tensor from encode_prompt is usually that required path.
            if len(embeds) >= 1:
                result["prompt_embeds"] = embeds[0]
            if len(embeds) >= 2:
                result["prompt_embeds_2"] = embeds[1]
            if len(embeds) >= 3:
                result["negative_prompt_embeds"] = embeds[2]
            if len(embeds) >= 4:
                result["negative_prompt_embeds_2"] = embeds[3]
            if len(masks) >= 1:
                result["prompt_embeds_mask"] = masks[0]
            if len(masks) >= 2:
                result["prompt_embeds_2_mask"] = masks[1]
            if len(masks) >= 3:
                result["negative_prompt_embeds_mask"] = masks[2]
            if len(masks) >= 4:
                result["negative_prompt_embeds_2_mask"] = masks[3]

        def _ones_mask_for_embed(embed):
            if not _is_tensor(embed):
                return None
            try:
                shape = tuple(embed.shape)
                if len(shape) < 2:
                    return None
                return torch.ones((int(shape[0]), int(shape[1])), dtype=torch.bool, device=getattr(embed, "device", "cpu"))
            except Exception:
                return None

        # If this pipeline requires prompt_embeds_2 and encode_prompt only gave
        # one prompt embedding, use it for prompt_embeds_2 rather than building
        # an invalid __call__ without prompt/prompt_embeds_2.
        if result["prompt_embeds_2"] is None and result["prompt_embeds"] is not None and _supports_kw(pipe.__call__, "prompt_embeds_2"):
            result["prompt_embeds_2"] = result["prompt_embeds"]
        if result["prompt_embeds_2_mask"] is None and result["prompt_embeds_mask"] is not None and _supports_kw(pipe.__call__, "prompt_embeds_2_mask"):
            result["prompt_embeds_2_mask"] = result["prompt_embeds_mask"]

        # Some Hunyuan Diffusers builds return embeddings without masks, but
        # __call__ refuses prompt_embeds unless the matching attention mask is
        # also present. For a single already-tokenized prompt, an all-true mask
        # is the safe fallback and keeps us out of the CUDA text-encode path.
        if result["prompt_embeds"] is not None and result["prompt_embeds_mask"] is None and _supports_kw(pipe.__call__, "prompt_embeds_mask"):
            result["prompt_embeds_mask"] = _ones_mask_for_embed(result["prompt_embeds"])
        if result["prompt_embeds_2"] is not None and result["prompt_embeds_2_mask"] is None and _supports_kw(pipe.__call__, "prompt_embeds_2_mask"):
            result["prompt_embeds_2_mask"] = _ones_mask_for_embed(result["prompt_embeds_2"])
        if result["negative_prompt_embeds"] is not None and result["negative_prompt_embeds_mask"] is None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask"):
            result["negative_prompt_embeds_mask"] = _ones_mask_for_embed(result["negative_prompt_embeds"])
        if result["negative_prompt_embeds_2"] is not None and result["negative_prompt_embeds_2_mask"] is None and _supports_kw(pipe.__call__, "negative_prompt_embeds_2_mask"):
            result["negative_prompt_embeds_2_mask"] = _ones_mask_for_embed(result["negative_prompt_embeds_2"])
        if result["prompt_embeds_2"] is not None and result["prompt_embeds_mask_2"] is None and _supports_kw(pipe.__call__, "prompt_embeds_mask_2"):
            result["prompt_embeds_mask_2"] = _ones_mask_for_embed(result["prompt_embeds_2"])
        if result["negative_prompt_embeds_2"] is not None and result["negative_prompt_embeds_mask_2"] is None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask_2"):
            result["negative_prompt_embeds_mask_2"] = _ones_mask_for_embed(result["negative_prompt_embeds_2"])

        # Diffusers Hunyuan builds disagree on whether the second-encoder mask is
        # named prompt_embeds_2_mask or prompt_embeds_mask_2. Keep both aliases
        # populated so __call__ receives the exact spelling required by the
        # installed package.
        if result["prompt_embeds_mask_2"] is None and result["prompt_embeds_2_mask"] is not None:
            result["prompt_embeds_mask_2"] = result["prompt_embeds_2_mask"]
        if result["prompt_embeds_2_mask"] is None and result["prompt_embeds_mask_2"] is not None:
            result["prompt_embeds_2_mask"] = result["prompt_embeds_mask_2"]
        if result["negative_prompt_embeds_mask_2"] is None and result["negative_prompt_embeds_2_mask"] is not None:
            result["negative_prompt_embeds_mask_2"] = result["negative_prompt_embeds_2_mask"]
        if result["negative_prompt_embeds_2_mask"] is None and result["negative_prompt_embeds_mask_2"] is not None:
            result["negative_prompt_embeds_2_mask"] = result["negative_prompt_embeds_mask_2"]

        # Require at least the embed path demanded by the installed pipeline.
        if _supports_kw(pipe.__call__, "prompt_embeds_2"):
            if result["prompt_embeds_2"] is None:
                return None
        elif result["prompt_embeds"] is None:
            return None

        for k, v in list(result.items()):
            result[k] = _cpu(v)
        return result
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

_VRAM_LAB_CONTEXT = None


class _VRAMLabReportBuffer:
    """Tiny reporter object compatible with tools/vram_lab reporter calls."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(str(text))
        if text:
            print(f"[vram-lab] {text}", flush=True)

    def section(self, title: str) -> None:
        self.line("")
        self.line("=" * 78)
        self.line(str(title))
        self.line("=" * 78)


def _vram_lab_report_path(root: Path) -> Path:
    # Timestamp every Hunyuan VRAM Lab report so long tests do not overwrite each other.
    try:
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    except Exception:
        stamp = str(int(time.time()))
    return root / "tools" / "vram_lab" / f"hunyuan_vram_lab_integration_report_{stamp}.txt"


def _fmt_cuda_bytes(n: int | None) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if abs(v) < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}" if u != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{n} B"


def _cuda_mem_line(torch) -> str:
    try:
        if torch is None or not torch.cuda.is_available():
            return "n/a"
        torch.cuda.synchronize()
        alloc = int(torch.cuda.memory_allocated())
        reserv = int(torch.cuda.memory_reserved())
        free, total = torch.cuda.mem_get_info()
        return f"allocated={_fmt_cuda_bytes(alloc)}, reserved={_fmt_cuda_bytes(reserv)}, driver_free={_fmt_cuda_bytes(int(free))}, driver_total={_fmt_cuda_bytes(int(total))}"
    except Exception as e:
        return f"n/a ({e})"



def _vram_lab_stage(ctx: dict | None, key: str, torch, note: str = "") -> None:
    """Record a best-effort CUDA stage memory line for the integration report."""
    if not isinstance(ctx, dict):
        return
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    ctx[str(key)] = _cuda_mem_line(torch)
    if note:
        ctx.setdefault("stage_notes", []).append(f"{key}: {note}; {ctx.get(str(key), 'n/a')}")


def _vram_lab_cleanup_cuda(ctx: dict | None, torch, label: str) -> None:
    """Best-effort CUDA cleanup used only when VRAM Lab mode is active."""
    if not isinstance(ctx, dict):
        return
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        ctx.setdefault("stage_notes", []).append(f"{label}: synchronized + empty_cache; {_cuda_mem_line(torch)}")
    except Exception as e:
        ctx.setdefault("stage_notes", []).append(f"{label}: cleanup skipped/failed ({e})")


def _vram_lab_force_memory_savers(args, ctx: dict | None) -> None:
    """Force memory-saver flags only for VRAM Lab modes; Off behavior stays untouched."""
    if not isinstance(ctx, dict):
        return
    forced = []
    if not getattr(args, "attn_slicing", False):
        args.attn_slicing = True
        forced.append("attention slicing")
    if not getattr(args, "vae_slicing", False):
        args.vae_slicing = True
        forced.append("VAE slicing")
    if not getattr(args, "tiling", False):
        args.tiling = True
        forced.append("VAE tiling")
    ctx["memory_saver_policy"] = "forced for VRAM Lab mode" if forced else "already enabled by user/settings"
    ctx["memory_saver_forced"] = ", ".join(forced) if forced else "none"
    ctx["attention_slicing_forced"] = "YES" if "attention slicing" in forced else "already enabled"
    ctx["vae_slicing_forced"] = "YES" if "VAE slicing" in forced else "already enabled"
    ctx["vae_tiling_forced"] = "YES" if "VAE tiling" in forced else "already enabled"


def _vram_lab_target_size_limit(mode: str) -> int:
    """Quality guard for older scan-only VRAM Lab experiments.

    VRAM Lab 0.5 is the forward-hook gate. It must not fake progress by
    lowering Hunyuan target_size, frames, or resolution. Return 0 so the
    user's selected generation settings stay intact in Safe/Balanced modes.
    """
    return 0


def _normalize_vram_lab_mode(mode: str) -> str:
    """Normalize UI/legacy VRAM Lab values without changing the non-VRAM path."""
    mode = str(mode or "off").lower().strip()
    if mode in {"on", "safe", "balanced", "aggressive"}:
        return "safe"
    return "off"


def _vram_lab_enabled(mode: str) -> bool:
    return _normalize_vram_lab_mode(mode) != "off"


def _ensure_vram_lab_allocator_env(mode: str) -> str:
    """Set allocator config before torch import when VRAM Lab mode is selected."""
    mode = _normalize_vram_lab_mode(mode)
    if not _vram_lab_enabled(mode):
        return "not requested"
    key = "PYTORCH_CUDA_ALLOC_CONF"
    cur = os.environ.get(key, "").strip()
    wanted = "expandable_segments:True"
    if not cur:
        os.environ[key] = wanted
        return f"set before torch import: {wanted}"
    if "expandable_segments" not in cur:
        os.environ[key] = cur + "," + wanted
        return f"appended before torch import: {os.environ[key]}"
    return f"already set before torch import: {cur}"


def _generation_safe_vram_policy(mode: str, total_vram_bytes: int) -> tuple[int, int, str]:
    """Return (budget_bytes, reserve_bytes, note) for real Hunyuan generation.

    The 0.3.1 stress gate used lower/residency budgets. Real generation also needs
    latents/attention/temporary tensors, so do not use the stress gate cap as the
    generation cap. Keep this policy generic: it returns a process budget and
    workspace reserve, while the actual model runner remains Hunyuan.
    """
    gib = 1024 ** 3
    total = int(total_vram_bytes or 0)
    mode = _normalize_vram_lab_mode(mode)
    aggressive_extra = int(2 * gib)
    if total <= 0:
        if mode == "aggressive":
            return 24 * gib, 0, "fallback aggressive: balanced + 2 GB advisory allowance"
        if mode == "balanced":
            return 22 * gib, 2 * gib, "fallback balanced: 22 GB budget / 2 GB reserve"
        return 20 * gib, 4 * gib, "fallback safe: 20 GB budget / 4 GB reserve"
    # RTX 3090 / 24 GB class defaults. Clamp to card size if used elsewhere.
    if total >= 23 * gib:
        if mode == "aggressive":
            base = min(22 * gib, int(total * 0.94))
            budget = min(total - int(0.5 * gib), base + aggressive_extra)
            return budget, max(0, total - budget), "24GB generation aggressive: balanced + 2 GB advisory allowance for step phase"
        if mode == "balanced":
            return min(22 * gib, int(total * 0.94)), max(1 * gib, total - min(22 * gib, int(total * 0.94))), "24GB generation balanced: 22 GB cap / about 2 GB driver reserve"
        return min(20 * gib, int(total * 0.86)), max(3 * gib, total - min(20 * gib, int(total * 0.86))), "24GB generation safe: 20 GB cap / about 4 GB driver reserve"
    # Smaller cards: keep more conservative headroom.
    if mode == "aggressive":
        budget = min(total - int(0.5 * gib), int(total * 0.88) + aggressive_extra)
    elif mode == "balanced":
        budget = int(total * 0.88)
    else:
        budget = int(total * 0.80)
    return budget, max(0, total - budget), f"generic generation {mode}: {budget / gib:.1f} GB advisory cap"




def _vram_lab_refresh_forward_hook_status(ctx: dict | None) -> None:
    """Copy live VRAM hook runtime counters into the integration report context."""
    if not isinstance(ctx, dict):
        return
    runtime = ctx.get("_vram_runtime")
    if runtime is not None and hasattr(runtime, "update_context"):
        try:
            runtime.update_context(ctx)
        except Exception as e:
            ctx["vram_hook_status_error"] = f"update_context failed: {e}"


def _vram_lab_detach_forward_hooks(ctx: dict | None) -> None:
    """Detach VRAM Lab forward hooks at process cleanup/finalization."""
    if not isinstance(ctx, dict):
        return
    runtime = ctx.get("_vram_runtime")
    if runtime is not None and hasattr(runtime, "detach_vram_hooks"):
        try:
            runtime.detach_vram_hooks()
            ctx["vram_hooks_detached"] = "YES"
        except Exception as e:
            ctx["vram_hooks_detached"] = f"FAILED: {e}"


def _vram_lab_make_finalize_guard(ctx: dict | None, torch, label: str = "hunyuan_finalize"):
    """Create the shared VRAM Lab finalize guard when VRAM Lab mode is active."""
    if not isinstance(ctx, dict):
        return None
    try:
        root = project_root()
        lab_dir = root / "tools" / "vram_lab"
        if str(lab_dir) not in sys.path:
            sys.path.insert(0, str(lab_dir))
        import vram_forward_hooks as vfh  # type: ignore
        if hasattr(vfh, "make_finalize_guard"):
            return vfh.make_finalize_guard(ctx=ctx, label=label, torch_module=torch)
    except Exception as e:
        ctx["finalize_guard_enabled"] = f"FAILED: {type(e).__name__}: {e}"
        ctx.setdefault("finalize_guard_notes", []).append(f"finalize guard import/create failed: {e}")
    return None



def _hunyuan_vram_env_float(name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a Hunyuan-only VRAM Lab float from env without touching shared VRAM Lab defaults."""
    try:
        value = float(os.environ.get(name, str(default)) or str(default))
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _hunyuan_vram_env_str(name: str, default: str, allowed: set[str] | None = None) -> str:
    """Read a Hunyuan-only VRAM Lab string from env and normalize it."""
    value = str(os.environ.get(name, default) or default).strip().lower()
    if allowed and value not in allowed:
        value = str(default).strip().lower()
    return value

def _hunyuan_vram_ctx_float(ctx: dict | None, key: str, env_name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read Hunyuan-local VRAM Lab value from CLI/UI ctx first, then env fallback."""
    try:
        if isinstance(ctx, dict) and ctx.get(key) is not None:
            v = float(ctx.get(key))
            if min_value is not None:
                v = max(float(min_value), v)
            if max_value is not None:
                v = min(float(max_value), v)
            return v
    except Exception:
        pass
    return _hunyuan_vram_env_float(env_name, default, min_value, max_value)


def _hunyuan_vram_ctx_str(ctx: dict | None, key: str, env_name: str, default: str, allowed: set[str] | None = None) -> str:
    try:
        if isinstance(ctx, dict) and ctx.get(key) is not None:
            value = str(ctx.get(key) or default).strip().lower()
            if allowed and value not in allowed:
                return str(default).strip().lower()
            return value
    except Exception:
        pass
    return _hunyuan_vram_env_str(env_name, default, allowed)

def _hunyuan_vram_ctx_bool(ctx: dict | None, key: str, env_name: str, default: bool) -> bool:
    def _parse(v):
        if isinstance(v, bool):
            return v
        s = str(v or '').strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off'):
            return False
        return None
    try:
        if isinstance(ctx, dict) and key in ctx:
            parsed = _parse(ctx.get(key))
            if parsed is not None:
                return bool(parsed)
        env = os.environ.get(env_name, None)
        if env is not None:
            parsed = _parse(env)
            if parsed is not None:
                return bool(parsed)
    except Exception:
        pass
    return bool(default)


def _vram_lab_force_cpu_object(obj):
    """Best-effort CPU move for tensors/lists/tuples/dicts without changing PIL/numpy frames."""
    try:
        if hasattr(obj, "detach") and hasattr(obj, "to"):
            try:
                return obj.detach().to("cpu")
            except Exception:
                return obj.to("cpu")
        if isinstance(obj, list):
            return [_vram_lab_force_cpu_object(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_vram_lab_force_cpu_object(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _vram_lab_force_cpu_object(v) for k, v in obj.items()}
        # Diffusers decode can return small output objects/dataclasses with a
        # tensor stored on .sample instead of returning a raw tuple. Preserve the
        # object when possible and only move the tensor field to CPU.
        if hasattr(obj, "sample"):
            try:
                obj.sample = _vram_lab_force_cpu_object(getattr(obj, "sample"))
                return obj
            except Exception:
                pass
    except Exception:
        pass
    return obj


def _vram_lab_tensor_devices(obj, limit: int = 12) -> list[str]:
    """Small tensor-device sampler for diagnostics; avoids dumping shapes/values."""
    out: list[str] = []
    seen: set[int] = set()

    def add(label: str, value) -> None:
        if len(out) >= limit:
            return
        try:
            if hasattr(value, "device") and hasattr(value, "dtype"):
                ident = id(value)
                if ident in seen:
                    return
                seen.add(ident)
                shape = tuple(getattr(value, "shape", ()) or ())
                out.append(f"{label}: device={getattr(value, 'device', 'n/a')}, dtype={getattr(value, 'dtype', 'n/a')}, shape={shape}")
        except Exception:
            pass

    def walk(label: str, value, depth: int = 0) -> None:
        if len(out) >= limit or depth > 3:
            return
        add(label, value)
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value[:6]):
                walk(f"{label}[{i}]", item, depth + 1)
        elif isinstance(value, dict):
            for i, (k, item) in enumerate(list(value.items())[:6]):
                walk(f"{label}.{k}", item, depth + 1)

    walk("arg", obj)
    return out


def _vram_lab_param_devices(module, limit: int = 12) -> list[str]:
    out: list[str] = []
    try:
        for i, (name, param) in enumerate(module.named_parameters(recurse=True)):
            if i >= limit:
                break
            out.append(f"param {name}: device={getattr(param, 'device', 'n/a')}, dtype={getattr(param, 'dtype', 'n/a')}, shape={tuple(getattr(param, 'shape', ()) or ())}")
    except Exception as e:
        out.append(f"param scan failed: {type(e).__name__}: {e}")
    try:
        for i, (name, buf) in enumerate(module.named_buffers(recurse=True)):
            if len(out) >= limit:
                break
            out.append(f"buffer {name}: device={getattr(buf, 'device', 'n/a')}, dtype={getattr(buf, 'dtype', 'n/a')}, shape={tuple(getattr(buf, 'shape', ()) or ())}")
    except Exception as e:
        out.append(f"buffer scan failed: {type(e).__name__}: {e}")
    return out


def _vram_lab_primary_tensor_device(args, kwargs):
    """Return the first real tensor device from a module call, preferring CUDA.

    Hunyuan + Diffusers model_cpu_offload can leave block weights on CPU while
    hidden states are already on cuda. VRAM Lab hooks should control residency,
    but this tiny fallback lets the currently executing transformer block align
    itself to the real input device instead of crashing in addmm.
    """
    found = None

    def walk(value, depth: int = 0):
        nonlocal found
        if found is not None or depth > 3:
            return
        try:
            if hasattr(value, "device") and hasattr(value, "dtype"):
                dev = getattr(value, "device", None)
                if dev is not None:
                    if str(dev).startswith("cuda"):
                        found = dev
                        return
                    if found is None:
                        found = dev
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            for item in value[:8]:
                walk(item, depth + 1)
                if found is not None and str(found).startswith("cuda"):
                    return
        elif isinstance(value, dict):
            for item in list(value.values())[:8]:
                walk(item, depth + 1)
                if found is not None and str(found).startswith("cuda"):
                    return

    walk(args)
    if found is None or not str(found).startswith("cuda"):
        walk(kwargs)
    return found


def _vram_lab_module_device_summary(module, limit: int = 64) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        for i, param in enumerate(module.parameters(recurse=True)):
            if i >= limit:
                break
            key = str(getattr(param, "device", "n/a"))
            counts[key] = counts.get(key, 0) + 1
    except Exception:
        pass
    return counts


def _vram_lab_align_current_block_to_inputs(module, args, kwargs, ctx: dict, block_name: str) -> None:
    """Fallback residency aligner for the active block only.

    This does not disable VRAM Lab hooks. It runs inside the already-hooked block
    forward and only moves that one transformer block to the input tensor device
    when Diffusers/Accelerate left submodule parameters on CPU.
    """
    try:
        dev = _vram_lab_primary_tensor_device(args, kwargs)
        if dev is None or not str(dev).startswith("cuda"):
            return
        counts = _vram_lab_module_device_summary(module, limit=96)
        if not counts or set(counts.keys()) == {str(dev)}:
            return
        if any(k == "cpu" or k.startswith("meta") for k in counts.keys()) or str(dev) not in counts:
            ctx["vram_device_align_active_block"] = block_name
            ctx["vram_device_align_last_before"] = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            try:
                n = int(ctx.get("vram_device_align_count", 0) or 0) + 1
            except Exception:
                n = 1
            ctx["vram_device_align_count"] = n
            if n <= 6:
                ctx.setdefault("vram_device_align_notes", []).append(f"{block_name}: moved block to {dev} from {ctx.get('vram_device_align_last_before', 'n/a')}")
            module.to(dev)
            after = _vram_lab_module_device_summary(module, limit=96)
            ctx["vram_device_align_last_after"] = ", ".join(f"{k}:{v}" for k, v in sorted(after.items()))
    except Exception as e:
        ctx["vram_device_align_error"] = f"{type(e).__name__}: {e}"




def _vram_lab_driver_free_bytes(torch) -> int | None:
    try:
        if torch is None or not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def _vram_lab_note_low_driver_free(ctx: dict | None, torch, label: str) -> None:
    """Track low driver-free VRAM as a proxy for shared-memory danger."""
    if not isinstance(ctx, dict):
        return
    free = _vram_lab_driver_free_bytes(torch)
    if free is None:
        return
    gib = 1024 ** 3
    try:
        low = ctx.get("vram_lowest_driver_free_bytes")
        if low is None or int(free) < int(low):
            ctx["vram_lowest_driver_free_bytes"] = int(free)
            ctx["vram_lowest_driver_free"] = _fmt_cuda_bytes(int(free))
            ctx["vram_lowest_driver_free_stage"] = str(label)
        notes = ctx.setdefault("vram_shared_spill_risk_notes", [])
        if int(free) < int(1.0 * gib):
            msg = f"{label}: driver_free below 1 GB ({_fmt_cuda_bytes(int(free))}); Windows shared-memory spill likely"
            if msg not in notes:
                notes.append(msg)
        elif int(free) < int(2.0 * gib):
            msg = f"{label}: driver_free below 2 GB ({_fmt_cuda_bytes(int(free))}); close to shared-memory spill"
            if msg not in notes:
                notes.append(msg)
    except Exception:
        pass


def _vram_lab_component_device_summary(obj, limit: int = 64) -> str:
    try:
        counts = {}
        n = 0
        for p in obj.parameters(recurse=True):
            d = str(getattr(p, "device", "?"))
            counts[d] = counts.get(d, 0) + 1
            n += 1
            if n >= int(limit):
                break
        return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no parameters sampled"
    except Exception as e:
        return f"n/a ({type(e).__name__}: {e})"


def _vram_lab_predenoise_component_purge(pipe, torch, ctx: dict | None, trigger: str = "first transformer block") -> None:
    """Move non-denoiser Hunyuan components off CUDA once denoise is about to start.

    Called from the first live transformer block. At that point Diffusers has
    already built prompt/image conditioning, so text/image encoders and VAE
    should not need to sit on CUDA during denoise. The transformer itself is not
    moved here; VRAM Lab hooks still control that path.
    """
    if not isinstance(ctx, dict):
        return
    if ctx.get("vram_predenoise_purge_done"):
        return
    ctx["vram_predenoise_purge_done"] = "YES"
    ctx["vram_predenoise_purge_trigger"] = str(trigger)
    try:
        ctx["cuda_before_predenoise_purge"] = _cuda_mem_line(torch)
        moved = []
        skipped = []
        for name in ("text_encoder", "text_encoder_2", "image_encoder", "vision_encoder", "vae"):
            comp = getattr(pipe, name, None)
            if comp is None:
                continue
            try:
                before = _vram_lab_component_device_summary(comp, limit=48)
                comp.to("cpu")
                moved.append(f"{name} ({before} -> cpu)")
            except Exception as e:
                skipped.append(f"{name}: {type(e).__name__}: {e}")
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        ctx["vram_predenoise_purge_moved"] = "; ".join(moved) if moved else "none"
        ctx["vram_predenoise_purge_skipped"] = "; ".join(skipped) if skipped else "none"
        ctx["cuda_after_predenoise_purge"] = _cuda_mem_line(torch)
        _vram_lab_note_low_driver_free(ctx, torch, "after pre-denoise component purge")
        print(f"[vram-lab] pre-denoise purge: moved {len(moved)} non-transformer component(s) to CPU", flush=True)
    except Exception as e:
        ctx["vram_predenoise_purge_error"] = f"{type(e).__name__}: {e}"



def _hunyuan_vram_move_components_to_cpu(pipe, component_names: tuple[str, ...], torch, ctx: dict | None, label: str) -> tuple[int, int]:
    """Move selected Hunyuan pipeline components to CPU and record compact telemetry."""
    moved: list[str] = []
    skipped: list[str] = []
    for name in component_names:
        try:
            comp = getattr(pipe, name, None)
        except Exception:
            comp = None
        if comp is None:
            continue
        try:
            before = _vram_lab_component_device_summary(comp, limit=48)
            comp.to("cpu")
            moved.append(f"{name} ({before} -> cpu)")
        except Exception as exc:
            skipped.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    if isinstance(ctx, dict):
        ctx[f"{label}_components_moved"] = "; ".join(moved) if moved else "none"
        ctx[f"{label}_components_skipped"] = "; ".join(skipped) if skipped else "none"
        ctx[f"cuda_after_{label}"] = _cuda_mem_line(torch)
        _vram_lab_note_low_driver_free(ctx, torch, f"after {label}")
    return (len(moved), len(skipped))


def _hunyuan_vram_cleanup_cuda_passes(torch, passes: int = 1) -> None:
    try:
        passes = max(0, min(int(passes), 8))
    except Exception:
        passes = 1
    for _ in range(passes):
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


def _hunyuan_vram_force_transformer_blocks_cpu(pipe, torch, ctx: dict | None, label: str = "hunyuan_decode_block_purge") -> tuple[int, int]:
    moved: list[str] = []
    skipped: list[str] = []
    try:
        transformer = getattr(pipe, "transformer", None)
    except Exception:
        transformer = None
    blocks = getattr(transformer, "transformer_blocks", None) if transformer is not None else None
    if blocks is None:
        if isinstance(ctx, dict):
            ctx[f"{label}_moved"] = "none"
            ctx[f"{label}_skipped"] = "transformer_blocks not found"
        return (0, 1)
    try:
        seq = list(blocks)
    except Exception:
        seq = []
    for idx, block in enumerate(seq):
        try:
            counts = _vram_lab_module_device_summary(block, limit=96)
            if counts and any(str(k).startswith("cuda") for k in counts.keys()):
                block.to("cpu")
                moved.append(f"transformer_blocks.{idx} ({', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))} -> cpu)")
        except Exception as exc:
            skipped.append(f"transformer_blocks.{idx}: {type(exc).__name__}: {exc}")
    _hunyuan_vram_cleanup_cuda_passes(torch, 1)
    if isinstance(ctx, dict):
        ctx[f"{label}_moved"] = "; ".join(moved) if moved else "none"
        ctx[f"{label}_skipped"] = "; ".join(skipped) if skipped else "none"
        ctx[f"{label}_count"] = str(len(moved))
        ctx[f"cuda_after_{label}"] = _cuda_mem_line(torch)
        _vram_lab_note_low_driver_free(ctx, torch, f"after {label}")
    return (len(moved), len(skipped))



def _hunyuan_vram_tensor_shape(obj) -> str:
    try:
        shp = tuple(getattr(obj, "shape", ()))
        return str(shp) if shp else "n/a"
    except Exception:
        return "n/a"


def _hunyuan_vram_decode_chunk_dim(tensor, chunk_frames: int) -> int | None:
    """Best-effort Hunyuan VAE latent time dimension detector.

    Most video VAEs use B,C,T,H,W, so dim=2 is the first choice. This is kept
    Hunyuan-local and optional because chunking decode is experimental.
    """
    try:
        shape = tuple(tensor.shape)
    except Exception:
        return None
    if len(shape) >= 5:
        # Hunyuan 1.5 VAE latents are B,C,T,H,W in the observed Diffusers build.
        # IMPORTANT: never fallback to dim=1 here, because dim=1 is channels (32).
        # The failed chunk test split channels into 16 and the VAE expected 32.
        if int(shape[2]) > int(chunk_frames):
            return 2
    return None


def _hunyuan_vram_first_tensor(obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)) and obj:
            return _hunyuan_vram_first_tensor(obj[0])
        if isinstance(obj, dict):
            for key in ("sample", "latents", "frames", "video"):
                if key in obj:
                    t = _hunyuan_vram_first_tensor(obj.get(key))
                    if t is not None:
                        return t
        sample = getattr(obj, "sample", None)
        if sample is not None:
            return _hunyuan_vram_first_tensor(sample)
    except Exception:
        return None
    return None


def _hunyuan_vram_slice_decode_result(obj, dim: int, start: int, end: int):
    """Slice a decoded VAE chunk result along the same temporal dimension."""
    try:
        import torch
        if obj is None:
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.ndim > int(dim):
                n = int(obj.shape[int(dim)])
                s = max(0, min(int(start), n))
                e = max(s + 1, min(int(end), n))
                return obj.narrow(int(dim), s, e - s).contiguous()
            return obj
        if isinstance(obj, tuple):
            return tuple(_hunyuan_vram_slice_decode_result(x, dim, start, end) for x in obj)
        if isinstance(obj, list):
            return [_hunyuan_vram_slice_decode_result(x, dim, start, end) for x in obj]
        if isinstance(obj, dict):
            out = dict(obj)
            for k, v in list(out.items()):
                out[k] = _hunyuan_vram_slice_decode_result(v, dim, start, end)
            return out
        sample = getattr(obj, "sample", None)
        if isinstance(sample, torch.Tensor):
            sliced = _hunyuan_vram_slice_decode_result(sample, dim, start, end)
            try:
                return obj.__class__(sample=sliced)
            except Exception:
                try:
                    obj.sample = sliced
                    return obj
                except Exception:
                    return sliced
    except Exception:
        pass
    return obj


def _hunyuan_vram_decode_result_len(obj, dim: int) -> int | None:
    try:
        t = _hunyuan_vram_first_tensor(obj)
        if t is not None and t.ndim > int(dim):
            return int(t.shape[int(dim)])
    except Exception:
        pass
    return None


def _hunyuan_vram_concat_decode_chunks(chunks, dim: int):
    """Concatenate VAE decode chunk results while preserving common Diffusers outputs."""
    if not chunks:
        return None
    try:
        import torch
        first = chunks[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(list(chunks), dim=dim)
        if isinstance(first, tuple):
            vals = []
            for i, item in enumerate(first):
                if isinstance(item, torch.Tensor):
                    vals.append(torch.cat([c[i] for c in chunks], dim=dim))
                else:
                    vals.append(item)
            return tuple(vals)
        if isinstance(first, list):
            vals = []
            for i, item in enumerate(first):
                if isinstance(item, torch.Tensor):
                    vals.append(torch.cat([c[i] for c in chunks], dim=dim))
                else:
                    vals.append(item)
            return vals
        if isinstance(first, dict):
            out = dict(first)
            for key, item in list(first.items()):
                if isinstance(item, torch.Tensor):
                    out[key] = torch.cat([c[key] for c in chunks], dim=dim)
            return out
        sample = getattr(first, "sample", None)
        if isinstance(sample, torch.Tensor):
            cat_sample = torch.cat([getattr(c, "sample") for c in chunks], dim=dim)
            try:
                return first.__class__(sample=cat_sample)
            except Exception:
                try:
                    first.sample = cat_sample
                    return first
                except Exception:
                    return cat_sample
    except Exception:
        pass
    return chunks[0]

def _hunyuan_vram_early_predecode_purge(pipe, torch, ctx: dict | None, trigger: str = "last transformer block") -> None:
    """Run the decode cleanup a little earlier, at the last transformer block.

    The decode guard works, but report 20260616_130438 showed the pipeline could
    already be near the shared-memory edge before the VAE decode wrapper starts.
    This Hunyuan-local guard runs after the final transformer block of the final
    denoise step, while still inside pipe(...), and before Diffusers enters the
    hidden VAE decode/postprocess section.
    """
    if not isinstance(ctx, dict):
        return
    if ctx.get("hunyuan_early_predecode_purge_done"):
        return
    ctx["hunyuan_early_predecode_purge_done"] = "YES"
    ctx["hunyuan_early_predecode_purge_trigger"] = str(trigger)
    started = time.perf_counter()
    try:
        _vram_lab_refresh_forward_hook_status(ctx)
    except Exception:
        pass
    try:
        ctx["cuda_before_hunyuan_early_predecode_purge"] = _cuda_mem_line(torch)
        _vram_lab_note_low_driver_free(ctx, torch, "before early pre-decode purge")
    except Exception:
        pass

    try:
        rt = ctx.get("_vram_runtime")
        if rt is not None and hasattr(rt, "detach_vram_hooks"):
            rt.detach_vram_hooks()
            ctx["hunyuan_early_predecode_detached_runtime"] = "YES"
            ctx["_vram_runtime"] = None
    except Exception as exc:
        ctx["hunyuan_early_predecode_detached_runtime"] = f"FAILED: {type(exc).__name__}: {exc}"

    try:
        moved_blocks, skipped_blocks = _hunyuan_vram_force_transformer_blocks_cpu(
            pipe, torch, ctx, "hunyuan_early_predecode_block_purge"
        )
        ctx["hunyuan_early_predecode_block_purge_result"] = f"moved={moved_blocks}, skipped={skipped_blocks}"
    except Exception as exc:
        ctx["hunyuan_early_predecode_block_purge_error"] = f"{type(exc).__name__}: {exc}"

    try:
        moved, skipped = _hunyuan_vram_move_components_to_cpu(
            pipe,
            ("transformer", "text_encoder", "text_encoder_2", "image_encoder", "vision_encoder"),
            torch,
            ctx,
            "hunyuan_early_predecode_purge",
        )
        ctx["hunyuan_early_predecode_purge_result"] = f"moved={moved}, skipped={skipped}"
    except Exception as exc:
        ctx["hunyuan_early_predecode_purge_error"] = f"{type(exc).__name__}: {exc}"

    try:
        passes = int(max(1, min(8, _hunyuan_vram_ctx_float(
            ctx,
            "hunyuan_ui_vram_decode_cleanup_passes",
            "HUNYUAN_VRAM_LAB_DECODE_CLEANUP_PASSES",
            1.0,
            0.0,
            8.0,
        ))))
    except Exception:
        passes = 1
    try:
        _hunyuan_vram_cleanup_cuda_passes(torch, passes)
    except Exception:
        pass
    try:
        ctx["cuda_after_hunyuan_early_predecode_purge"] = _cuda_mem_line(torch)
        _vram_lab_note_low_driver_free(ctx, torch, "after early pre-decode purge")
    except Exception:
        pass
    ctx["hunyuan_early_predecode_purge_duration"] = f"{(time.perf_counter() - started):.3f}s"
    try:
        print("[vram-lab] early pre-decode purge: final denoise block reached; cleared non-VAE residency", flush=True)
    except Exception:
        pass


def _hunyuan_vram_install_decode_guard(pipe, torch, ctx: dict | None, args=None) -> None:
    """Install a Hunyuan-local pre-decode guard.

    The Hunyuan Diffusers pipeline spends several minutes after the denoise progress
    bar reaches 100%, before pipe(...) returns. The normal finalize guard is too late
    for that phase. This wrapper catches VAE decode from inside the pipeline call,
    releases transformer/text/image residency first, and records the hidden memory
    stage without touching shared VRAM Lab files.
    """
    if not isinstance(ctx, dict):
        return
    try:
        vae = getattr(pipe, "vae", None)
        if vae is None:
            ctx["hunyuan_decode_guard_installed"] = "NO: pipe.vae not found"
            return
        candidates = []
        # Prefer the public decode() wrapper. Wrapping both decode and _decode causes
        # nested chunking: decode chunks the time dimension, then wrapped _decode sees
        # the smaller chunk and may try to chunk the wrong dimension. Only wrap _decode
        # as a fallback when decode() is unavailable.
        for method_name in ("decode",):
            try:
                fn = getattr(vae, method_name, None)
            except Exception:
                fn = None
            if fn is not None and callable(fn) and not getattr(fn, "_fv_hunyuan_decode_guard_wrapped", False):
                candidates.append((method_name, fn))
        if not candidates:
            for method_name in ("_decode",):
                try:
                    fn = getattr(vae, method_name, None)
                except Exception:
                    fn = None
                if fn is not None and callable(fn) and not getattr(fn, "_fv_hunyuan_decode_guard_wrapped", False):
                    candidates.append((method_name, fn))
        if not candidates:
            ctx["hunyuan_decode_guard_installed"] = "NO: no VAE decode method found"
            return

        def make_wrapped(method_name: str, original):
            def _wrapped_decode(*d_args, **d_kwargs):
                started = time.perf_counter()
                try:
                    ctx["hunyuan_decode_guard_calls"] = int(ctx.get("hunyuan_decode_guard_calls", 0) or 0) + 1
                except Exception:
                    ctx["hunyuan_decode_guard_calls"] = 1
                call_no = ctx.get("hunyuan_decode_guard_calls", 1)
                ctx["hunyuan_decode_guard_active_method"] = str(method_name)
                ctx["cuda_before_hunyuan_vae_decode"] = _cuda_mem_line(torch)
                _vram_lab_note_low_driver_free(ctx, torch, f"before VAE decode call {call_no}")

                decode_min_free_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_decode_min_free_gb", "HUNYUAN_VRAM_LAB_DECODE_MIN_FREE_GB", 6.0, 0.0, 20.0)
                decode_cleanup_passes = int(max(0, min(8, _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_decode_cleanup_passes", "HUNYUAN_VRAM_LAB_DECODE_CLEANUP_PASSES", 1.0, 0.0, 8.0))))
                decode_chunk_frames = int(max(0, min(241, _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_decode_chunk_frames", "HUNYUAN_VRAM_LAB_DECODE_CHUNK_FRAMES", 0.0, 0.0, 241.0))))
                decode_chunk_overlap = int(max(0, min(64, _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_decode_chunk_overlap", "HUNYUAN_VRAM_LAB_DECODE_CHUNK_OVERLAP", 0.0, 0.0, 64.0))))
                if decode_chunk_frames > 0:
                    decode_chunk_overlap = min(decode_chunk_overlap, max(0, decode_chunk_frames - 1))
                decode_unload_blocks = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_decode_unload_blocks", "HUNYUAN_VRAM_LAB_DECODE_UNLOAD_BLOCKS", True)
                decode_output_cpu = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_decode_output_cpu", "HUNYUAN_VRAM_LAB_DECODE_OUTPUT_CPU", True)
                ctx["hunyuan_decode_guard_min_free_gb"] = f"{decode_min_free_gb:.1f} GB"
                ctx["hunyuan_decode_guard_cleanup_passes"] = str(decode_cleanup_passes)
                ctx["hunyuan_decode_guard_chunk_frames"] = str(decode_chunk_frames)
                ctx["hunyuan_decode_guard_chunk_overlap"] = str(decode_chunk_overlap)
                ctx["hunyuan_decode_guard_unload_blocks"] = "YES" if decode_unload_blocks else "NO"
                ctx["hunyuan_decode_guard_output_cpu_enabled"] = "YES" if decode_output_cpu else "NO"

                # Freeze/update counters, then detach the transformer hook runtime before
                # VAE decode/postprocess. The denoise blocks are finished by the time VAE
                # decode starts, so keeping runtime references only increases churn.
                try:
                    _vram_lab_refresh_forward_hook_status(ctx)
                    rt = ctx.get("_vram_runtime")
                    if rt is not None and hasattr(rt, "detach_vram_hooks"):
                        rt.detach_vram_hooks()
                        ctx["hunyuan_decode_guard_detached_runtime"] = "YES"
                        ctx["_vram_runtime"] = None
                except Exception as exc:
                    ctx["hunyuan_decode_guard_detached_runtime"] = f"FAILED: {type(exc).__name__}: {exc}"

                if decode_unload_blocks:
                    try:
                        moved_blocks, skipped_blocks = _hunyuan_vram_force_transformer_blocks_cpu(pipe, torch, ctx, f"hunyuan_decode_block_purge_{call_no}")
                        print(f"[vram-lab] pre-decode block purge: moved {moved_blocks} hooked block(s) to CPU", flush=True)
                    except Exception as exc:
                        ctx[f"hunyuan_decode_block_purge_error_{call_no}"] = f"{type(exc).__name__}: {exc}"

                # Important: do NOT move the VAE here. It is the component about to run.
                # Move everything else that should not be needed during decode/postprocess.
                try:
                    moved, skipped = _hunyuan_vram_move_components_to_cpu(
                        pipe,
                        ("transformer", "text_encoder", "text_encoder_2", "image_encoder", "vision_encoder"),
                        torch,
                        ctx,
                        "hunyuan_pre_decode_purge",
                    )
                    print(f"[vram-lab] pre-decode guard: moved {moved} non-VAE component(s) to CPU", flush=True)
                except Exception as exc:
                    ctx["hunyuan_pre_decode_purge_error"] = f"{type(exc).__name__}: {exc}"

                _hunyuan_vram_cleanup_cuda_passes(torch, decode_cleanup_passes)
                ctx["cuda_at_hunyuan_vae_decode_start"] = _cuda_mem_line(torch)
                _vram_lab_note_low_driver_free(ctx, torch, f"VAE decode start {call_no}")
                try:
                    free = _vram_lab_driver_free_bytes(torch)
                    target = int(float(decode_min_free_gb) * (1024 ** 3))
                    if free is not None and target > 0 and int(free) < target:
                        ctx.setdefault("hunyuan_decode_guard_notes", []).append(
                            f"decode call {call_no}: driver_free {_fmt_cuda_bytes(int(free))} below decode target {decode_min_free_gb:.1f} GB before VAE decode"
                        )
                except Exception:
                    pass
                input_tensor = _hunyuan_vram_first_tensor(d_args[0] if d_args else None)
                ctx[f"hunyuan_decode_call_{call_no}_input_shape"] = _hunyuan_vram_tensor_shape(input_tensor)
                chunk_dim = _hunyuan_vram_decode_chunk_dim(input_tensor, decode_chunk_frames) if decode_chunk_frames > 0 and input_tensor is not None else None
                ctx[f"hunyuan_decode_call_{call_no}_chunk_dim"] = "n/a" if chunk_dim is None else str(chunk_dim)
                try:
                    if chunk_dim is not None:
                        total = int(tuple(input_tensor.shape)[int(chunk_dim)])
                        chunks = []
                        chunk_count = 0
                        for start_i in range(0, total, max(1, int(decode_chunk_frames))):
                            end_i = min(total, start_i + int(decode_chunk_frames))
                            ext_start = max(0, start_i - int(decode_chunk_overlap))
                            ext_end = min(total, end_i + int(decode_chunk_overlap))
                            chunk_count += 1
                            part_args = list(d_args)
                            part_args[0] = input_tensor.narrow(int(chunk_dim), int(ext_start), int(ext_end - ext_start))
                            _hunyuan_vram_cleanup_cuda_passes(torch, decode_cleanup_passes)
                            ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_range"] = f"core={start_i}:{end_i}/{total}, ext={ext_start}:{ext_end}, overlap={decode_chunk_overlap}"
                            ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_before"] = _cuda_mem_line(torch)
                            part_result = original(*tuple(part_args), **d_kwargs)
                            ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_after_raw"] = _cuda_mem_line(torch)

                            if int(decode_chunk_overlap) > 0:
                                out_len = _hunyuan_vram_decode_result_len(part_result, int(chunk_dim))
                                ext_len = max(1, int(ext_end - ext_start))
                                if out_len is not None and out_len > 1:
                                    crop_start = int(round(((int(start_i) - int(ext_start)) / float(ext_len)) * float(out_len)))
                                    crop_end = int(round(((int(end_i) - int(ext_start)) / float(ext_len)) * float(out_len)))
                                    crop_start = max(0, min(crop_start, out_len - 1))
                                    crop_end = max(crop_start + 1, min(crop_end, out_len))
                                    ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_crop"] = f"{crop_start}:{crop_end}/{out_len}"
                                    part_result = _hunyuan_vram_slice_decode_result(part_result, int(chunk_dim), crop_start, crop_end)
                                else:
                                    ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_crop"] = "n/a: output length unavailable"

                            if decode_output_cpu:
                                part_result = _vram_lab_force_cpu_object(part_result)
                            chunks.append(part_result)
                            _hunyuan_vram_cleanup_cuda_passes(torch, decode_cleanup_passes)
                            ctx[f"hunyuan_decode_call_{call_no}_chunk_{chunk_count}_after_cleanup"] = _cuda_mem_line(torch)
                        result = _hunyuan_vram_concat_decode_chunks(chunks, int(chunk_dim))
                        if int(decode_chunk_overlap) > 0:
                            ctx[f"hunyuan_decode_call_{call_no}_chunked"] = f"YES: {chunk_count} overlapped chunk(s), dim={chunk_dim}, frames={decode_chunk_frames}, overlap={decode_chunk_overlap}"
                        else:
                            ctx[f"hunyuan_decode_call_{call_no}_chunked"] = f"YES: {chunk_count} hard chunk(s), dim={chunk_dim}, frames={decode_chunk_frames}"
                    else:
                        result = original(*d_args, **d_kwargs)
                        ctx[f"hunyuan_decode_call_{call_no}_chunked"] = "NO"
                except Exception as exc:
                    ctx["hunyuan_decode_guard_exception"] = f"{type(exc).__name__}: {exc}"
                    ctx["cuda_after_hunyuan_vae_decode_exception"] = _cuda_mem_line(torch)
                    raise

                ctx["cuda_after_hunyuan_vae_decode_raw"] = _cuda_mem_line(torch)
                _vram_lab_note_low_driver_free(ctx, torch, f"after VAE decode raw {call_no}")

                # Move decoded tensors to CPU before Diffusers postprocess when possible.
                # PIL/numpy frames pass through untouched. If the pipeline expects a CUDA
                # tensor later, the failure report will point to this boundary clearly.
                if decode_output_cpu:
                    try:
                        result = _vram_lab_force_cpu_object(result)
                        ctx["hunyuan_decode_output_to_cpu"] = "YES"
                    except Exception as exc:
                        ctx["hunyuan_decode_output_to_cpu"] = f"FAILED: {type(exc).__name__}: {exc}"
                else:
                    ctx["hunyuan_decode_output_to_cpu"] = "NO: disabled by setting"

                try:
                    # After decode, the VAE can leave CUDA too; postprocess/export should
                    # work from CPU tensors/arrays/frames and this attacks the 100%-bar spike.
                    _hunyuan_vram_move_components_to_cpu(pipe, ("vae",), torch, ctx, "hunyuan_post_decode_vae_purge")
                except Exception as exc:
                    ctx["hunyuan_post_decode_vae_purge_error"] = f"{type(exc).__name__}: {exc}"
                _hunyuan_vram_cleanup_cuda_passes(torch, decode_cleanup_passes)
                ctx["cuda_after_hunyuan_decode_guard_cleanup"] = _cuda_mem_line(torch)
                ctx["hunyuan_decode_guard_last_duration"] = f"{(time.perf_counter() - started):.3f}s"
                _vram_lab_note_low_driver_free(ctx, torch, f"after VAE decode guard cleanup {call_no}")
                return result
            setattr(_wrapped_decode, "_fv_hunyuan_decode_guard_wrapped", True)
            return _wrapped_decode

        wrapped_names = []
        for method_name, original in candidates:
            try:
                setattr(vae, method_name, make_wrapped(method_name, original))
                wrapped_names.append(method_name)
            except Exception as exc:
                ctx.setdefault("hunyuan_decode_guard_wrap_errors", []).append(f"{method_name}: {type(exc).__name__}: {exc}")
        ctx["hunyuan_decode_guard_installed"] = "YES: " + ", ".join(wrapped_names) if wrapped_names else "NO: wrapping failed"
    except Exception as exc:
        ctx["hunyuan_decode_guard_installed"] = f"FAILED: {type(exc).__name__}: {exc}"

def _vram_lab_apply_hunyuan_step_policy(policy: dict, ctx: dict | None, torch) -> dict:
    """Tighten VRAM Lab residency for Hunyuan denoise steps.

    The generic/LTX-like profile was too loose for Hunyuan 121-frame tests on a
    24 GB card: hooks fired, but Windows still spilled into shared memory.
    """
    if not isinstance(policy, dict):
        return policy
    gib = 1024 ** 3
    # Hunyuan has much larger live activation/workspace pressure than the LTX-style
    # generic profile. Report 22 proved that 16 GB resident/hot with a nominal
    # 6 GB free target still dropped to ~1.5 GB driver-free. Keep this strictly
    # Hunyuan-local: do not change shared VRAM Lab defaults for other models.
    text_hot_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_text_hot_gb", "HUNYUAN_VRAM_LAB_TEXT_HOT_GB", 9.0, 0.0, 20.0)
    strict_hot_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_step_hot_gb", "HUNYUAN_VRAM_LAB_HOT_GB", 1.0, 0.25, 20.0)
    after_hot_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_after_hot_gb", "HUNYUAN_VRAM_LAB_AFTER_HOT_GB", 0.0, 0.0, 20.0)
    min_free_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_min_free_gb", "HUNYUAN_VRAM_LAB_MIN_FREE_GB", 0.5, 0.5, 16.0)
    step_cleanup_every = int(max(0, min(64, _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_step_cleanup_every", "HUNYUAN_VRAM_LAB_STEP_CLEANUP_EVERY", 4.0, 0.0, 64.0))))
    step_empty_cache = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_empty_cache", "HUNYUAN_VRAM_LAB_STEP_EMPTY_CACHE", True)
    step_release_after_forward = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_release_after_forward", "HUNYUAN_VRAM_LAB_STEP_RELEASE_AFTER_FORWARD", True)
    step_unload_before_load = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_unload_before_load", "HUNYUAN_VRAM_LAB_STEP_UNLOAD_BEFORE_LOAD", True)
    # Hunyuan stage-1 stability default: rolling/single-block residency.
    # The previous planned-hotset path passed the memory boundary, but did it by
    # fighting: hundreds of forced unloads/emergency cleanups. Rolling mode avoids
    # keeping a stable prefix resident while Diffusers/Accelerate is also moving
    # modules, so we can test whether step memory becomes smoother instead of
    # only lowering the budget another tiny amount.
    residency_strategy = _hunyuan_vram_ctx_str(
        ctx,
        "hunyuan_ui_vram_residency",
        "HUNYUAN_VRAM_LAB_RESIDENCY",
        "planned_hotset",
        {"rolling", "planned_hotset"},
    )
    stable_fraction = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_stable_fraction", "HUNYUAN_VRAM_LAB_STABLE_FRACTION", 1.15, 0.0, 2.0)
    stable_budget_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_stable_budget_gb", "HUNYUAN_VRAM_LAB_STABLE_BUDGET_GB", 1.0, 0.0, 20.0)
    strict_hot = int(strict_hot_gb * gib)
    min_free = int(min_free_gb * gib)
    stable_budget = int(stable_budget_gb * gib)
    for key in (
        "hot_block_budget_bytes", "hot_blocks_budget_bytes", "hot_budget_bytes",
        "block_hot_budget_bytes", "resident_budget_bytes", "max_resident_bytes",
        "max_cuda_resident_bytes", "block_residency_budget_bytes",
    ):
        policy[key] = strict_hot
    for key in (
        "min_driver_free_bytes", "driver_free_floor_bytes", "emergency_driver_free_floor_bytes",
        "emergency_trim_driver_free_bytes", "shared_spill_guard_free_bytes",
        "trim_when_driver_free_below_bytes", "free_floor_bytes", "min_free_bytes",
    ):
        policy[key] = min_free
    # Force a conservative Hunyuan-local residency mode. The shared safe profile
    # intentionally keeps blocks resident for speed, but Hunyuan's 480p+ jobs need
    # stronger shared-memory protection before speed tuning.
    policy["release_after_forward"] = bool(step_release_after_forward)
    policy["unload_other_blocks_before_load"] = bool(step_unload_before_load)
    policy["synchronize_after_unload"] = True
    policy["empty_cache_after_unload"] = bool(step_empty_cache)
    policy["empty_cache_every"] = int(step_cleanup_every)
    policy["cleanup_every_n_blocks"] = int(step_cleanup_every)
    policy["residency_strategy"] = residency_strategy
    policy["stable_hotset_fraction"] = stable_fraction
    policy["stable_hotset_budget_bytes"] = stable_budget
    policy["hunyuan_step_policy"] = "pressure_safe_rolling_stage1"
    if isinstance(ctx, dict):
        ctx["hunyuan_text_stage_hot_budget"] = f"{text_hot_gb:.1f} GB"
        ctx["hunyuan_after_stage_hot_budget"] = f"{after_hot_gb:.1f} GB"
        ctx["hunyuan_step_policy"] = f"pressure-safe budget: {strict_hot_gb:.1f} GB planned weight/cache; keep about {min_free_gb:.1f} GB driver-free workspace"
        ctx["hunyuan_step_policy_hot_budget"] = _fmt_cuda_bytes(strict_hot)
        ctx["hunyuan_step_policy_min_driver_free"] = _fmt_cuda_bytes(min_free)
        ctx["hunyuan_step_cleanup_every"] = str(step_cleanup_every)
        ctx["hunyuan_step_empty_cache"] = "YES" if step_empty_cache else "NO"
        ctx["hunyuan_step_release_after_forward"] = "YES" if step_release_after_forward else "NO"
        ctx["hunyuan_step_unload_before_load"] = "YES" if step_unload_before_load else "NO"
        ctx["hunyuan_step_residency_strategy"] = residency_strategy
        ctx["hunyuan_step_stable_hotset_fraction"] = f"{stable_fraction:.2f}"
        ctx["hunyuan_step_stable_hotset_budget"] = _fmt_cuda_bytes(stable_budget)
        ctx.setdefault("notes", []).append(
            f"Hunyuan VRAM Lab uses a Hunyuan-local stage-1 stability profile: "
            f"{strict_hot_gb:.1f}GB hot budget, {min_free_gb:.1f}GB driver-free floor, "
            f"residency={residency_strategy}, release-after-forward enabled; shared VRAM Lab helper left untouched."
        )
    return policy


def _vram_lab_force_runtime_hunyuan_budget(runtime, ctx: dict | None) -> None:
    """Best-effort override for VRAM hook runtimes that copy profile values
    into object attributes instead of reading the policy dict on each call.

    This is intentionally narrow: it only lowers resident/hot budgets for
    Hunyuan VRAM Lab runs and does not disable hooks. The runtime value must
    match the Hunyuan step policy; otherwise the report can say 9 GB while
    the live runtime still trims at 13 GB.
    """
    if runtime is None or not isinstance(ctx, dict):
        return
    try:
        gib = 1024 ** 3

        def _ctx_gb_from_bytes(key: str, fallback_gb: float) -> float:
            try:
                raw = ctx.get(key)
                if isinstance(raw, str):
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw.replace(",", "."))
                    if m:
                        return float(m.group(1))
                if raw is not None:
                    return float(raw) / float(gib)
            except Exception:
                pass
            return float(fallback_gb)

        # Keep one source of truth. _vram_lab_apply_hunyuan_step_policy writes
        # ctx['hunyuan_step_policy_hot_budget']; the live runtime override now
        # follows that unless an explicit environment override is provided.
        default_hot_gb = _ctx_gb_from_bytes("hunyuan_step_policy_hot_budget", 9.0)
        default_min_gb = _ctx_gb_from_bytes("hunyuan_step_policy_min_driver_free", 6.5)
        hot_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_step_hot_gb", "HUNYUAN_VRAM_LAB_HOT_GB", default_hot_gb, 0.25, 20.0)
        min_free_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_min_free_gb", "HUNYUAN_VRAM_LAB_MIN_FREE_GB", default_min_gb, 1.0, 16.0)
        residency_strategy = _hunyuan_vram_ctx_str(
            ctx,
            "hunyuan_ui_vram_residency",
            "HUNYUAN_VRAM_LAB_RESIDENCY",
            str(ctx.get("hunyuan_step_residency_strategy", "rolling") or "rolling"),
            {"rolling", "planned_hotset"},
        )
        stable_fraction = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_stable_fraction", "HUNYUAN_VRAM_LAB_STABLE_FRACTION", 1.15, 0.0, 2.0)
        stable_budget_gb = _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_stable_budget_gb", "HUNYUAN_VRAM_LAB_STABLE_BUDGET_GB", 1.0, 0.0, 20.0)
        step_cleanup_every = int(max(0, min(64, _hunyuan_vram_ctx_float(ctx, "hunyuan_ui_vram_step_cleanup_every", "HUNYUAN_VRAM_LAB_STEP_CLEANUP_EVERY", 4.0, 0.0, 64.0))))
        step_empty_cache = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_empty_cache", "HUNYUAN_VRAM_LAB_STEP_EMPTY_CACHE", True)
        step_release_after_forward = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_release_after_forward", "HUNYUAN_VRAM_LAB_STEP_RELEASE_AFTER_FORWARD", True)
        step_unload_before_load = _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_unload_before_load", "HUNYUAN_VRAM_LAB_STEP_UNLOAD_BEFORE_LOAD", True)
        hot = int(hot_gb * gib)
        min_free = int(min_free_gb * gib)
        stable_budget = int(stable_budget_gb * gib)
        hot_names = (
            "hot_block_budget_bytes", "hot_blocks_budget_bytes", "hot_budget_bytes",
            "block_hot_budget_bytes", "resident_budget_bytes", "max_resident_bytes",
            "max_cuda_resident_bytes", "block_residency_budget_bytes", "hot_block_budget",
            "hot_blocks_budget", "resident_budget", "max_cuda_resident",
        )
        min_names = (
            "min_driver_free_bytes", "driver_free_floor_bytes", "emergency_driver_free_floor_bytes",
            "emergency_trim_driver_free_bytes", "shared_spill_guard_free_bytes",
            "trim_when_driver_free_below_bytes", "free_floor_bytes", "min_free_bytes",
        )
        bool_overrides = {
            "release_after_forward": bool(step_release_after_forward),
            "unload_other_blocks_before_load": bool(step_unload_before_load),
            "synchronize_after_unload": True,
            "empty_cache_after_unload": bool(step_empty_cache),
        }
        int_overrides = {
            "empty_cache_every": int(step_cleanup_every),
            "cleanup_every_n_blocks": int(step_cleanup_every),
            "stable_hotset_budget_bytes": stable_budget,
        }
        float_overrides = {
            "stable_hotset_fraction": stable_fraction,
        }
        str_overrides = {
            "residency_strategy": residency_strategy,
        }
        touched = []
        for obj in (runtime, getattr(runtime, "policy", None), getattr(runtime, "config", None)):
            if obj is None:
                continue
            if isinstance(obj, dict):
                for name in hot_names:
                    obj[name] = hot
                for name in min_names:
                    obj[name] = min_free
                for name, value in bool_overrides.items():
                    obj[name] = value
                for name, value in int_overrides.items():
                    obj[name] = value
                for name, value in float_overrides.items():
                    obj[name] = value
                for name, value in str_overrides.items():
                    obj[name] = value
                touched.append("dict")
            else:
                for name in hot_names:
                    try:
                        setattr(obj, name, hot)
                        touched.append(name)
                    except Exception:
                        pass
                for name in min_names:
                    try:
                        setattr(obj, name, min_free)
                        touched.append(name)
                    except Exception:
                        pass
                for name, value in bool_overrides.items():
                    try:
                        setattr(obj, name, value)
                        touched.append(name)
                    except Exception:
                        pass
                for name, value in int_overrides.items():
                    try:
                        setattr(obj, name, value)
                        touched.append(name)
                    except Exception:
                        pass
                for name, value in float_overrides.items():
                    try:
                        setattr(obj, name, value)
                        touched.append(name)
                    except Exception:
                        pass
                for name, value in str_overrides.items():
                    try:
                        setattr(obj, name, value)
                        touched.append(name)
                    except Exception:
                        pass
        try:
            if residency_strategy == "rolling":
                setattr(runtime, "_stable_hotset_names", set())
                setattr(runtime, "_stable_hotset_order", [])
                setattr(runtime, "_stable_hotset_bytes", 0)
            elif hasattr(runtime, "_plan_stable_hotset"):
                runtime._plan_stable_hotset()
        except Exception as _plan_e:
            ctx.setdefault("vram_shared_spill_risk_notes", []).append(f"Hunyuan residency replan warning: {type(_plan_e).__name__}: {_plan_e}")
        ctx["hunyuan_runtime_budget_override"] = (
            f"hot={_fmt_cuda_bytes(hot)}, min_free={_fmt_cuda_bytes(min_free)}, "
            f"residency={residency_strategy}, stable_fraction={stable_fraction:.2f}, "
            f"stable_budget={_fmt_cuda_bytes(stable_budget)}, release_after_forward={step_release_after_forward}, "
            f"empty_cache_after_unload={step_empty_cache}, cleanup_every={step_cleanup_every}, "
            f"unload_before_load={step_unload_before_load}, touched={len(touched)}"
        )
    except Exception as e:
        ctx["hunyuan_runtime_budget_override"] = f"FAILED: {type(e).__name__}: {e}"


def _hunyuan_vram_live_snapshot(ctx: dict | None, torch_module, label: str, *, print_line: bool = False) -> None:
    """Refresh report counters and optionally print a compact live stage line."""
    if not isinstance(ctx, dict):
        return
    try:
        _vram_lab_refresh_forward_hook_status(ctx)
    except Exception:
        pass
    try:
        _vram_lab_note_low_driver_free(ctx, torch_module, label)
    except Exception:
        pass
    try:
        snap = _cuda_mem_line(torch_module)
        key = "hunyuan_live_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_").lower()
        ctx[key] = snap
    except Exception:
        snap = "n/a"
    if print_line:
        try:
            print(
                f"[vram-lab] {label}: {snap}; "
                f"hooks pre/post={ctx.get('vram_pre_forward_calls','0')}/{ctx.get('vram_post_forward_calls','0')}, "
                f"loads={ctx.get('vram_block_load_count','0')}, unloads={ctx.get('vram_block_unload_count','0')}, "
                f"forced={ctx.get('vram_forced_unload_count','0')}, emergency={ctx.get('vram_emergency_cleanup_count','0')}",
                flush=True,
            )
        except Exception:
            pass


def _hunyuan_vram_profiler_enabled(ctx: dict | None) -> bool:
    if not isinstance(ctx, dict):
        return False
    return _hunyuan_vram_ctx_bool(ctx, "hunyuan_ui_vram_step_profiler", "HUNYUAN_VRAM_LAB_STEP_PROFILER", False)


def _hunyuan_vram_profiler_steps(ctx: dict | None) -> int:
    try:
        return max(1, min(3, int(float(ctx.get("hunyuan_ui_vram_step_profiler_steps", 1) if isinstance(ctx, dict) else 1))))
    except Exception:
        return 1


def _hunyuan_vram_profiler_counters(ctx: dict | None) -> dict:
    if not isinstance(ctx, dict):
        return {}
    out = {}
    for k in ("vram_block_load_count", "vram_block_unload_count", "vram_forced_unload_count", "vram_emergency_cleanup_count", "vram_cache_cleanup_count"):
        try:
            out[k] = int(ctx.get(k, 0) or 0)
        except Exception:
            out[k] = 0
    return out


def _hunyuan_vram_profiler_counter_delta(before: dict, ctx: dict | None) -> str:
    if not isinstance(ctx, dict):
        return "n/a"
    parts = []
    for key, label in (
        ("vram_block_load_count", "load"),
        ("vram_block_unload_count", "unload"),
        ("vram_forced_unload_count", "forced"),
        ("vram_emergency_cleanup_count", "emergency"),
        ("vram_cache_cleanup_count", "cache"),
    ):
        try:
            now = int(ctx.get(key, 0) or 0)
            old = int(before.get(key, 0) or 0)
            delta = now - old
            if delta:
                parts.append(f"{label}+{delta}")
        except Exception:
            pass
    return ", ".join(parts) if parts else "no hook counter delta"


def _hunyuan_vram_profiler_record(ctx: dict | None, torch_module, block_name: str, step_index: int, block_index: int, before_cuda: str, after_cuda: str, elapsed_ms: float, before_counts: dict) -> None:
    if not isinstance(ctx, dict):
        return
    rows = ctx.setdefault("hunyuan_step_profiler_rows", [])
    if not isinstance(rows, list) or len(rows) >= 220:
        return
    try:
        _vram_lab_refresh_forward_hook_status(ctx)
    except Exception:
        pass
    try:
        free = _vram_lab_driver_free_bytes(torch_module)
        free_txt = _fmt_cuda_bytes(int(free)) if free is not None else "n/a"
    except Exception:
        free_txt = "n/a"
    rows.append({
        "step": int(step_index),
        "block": int(block_index),
        "name": str(block_name),
        "ms": float(elapsed_ms),
        "before": str(before_cuda),
        "after": str(after_cuda),
        "driver_free": str(free_txt),
        "delta": _hunyuan_vram_profiler_counter_delta(before_counts, ctx),
    })


def _hunyuan_vram_make_step_end_callback(pipe, ctx: dict | None, torch_module, previous_callback=None):
    """Create a Hunyuan-local step telemetry callback and preserve any existing callback."""
    if not isinstance(ctx, dict):
        return previous_callback

    def _callback(p, step, timestep, callback_kwargs):
        if previous_callback is not None:
            try:
                callback_kwargs = previous_callback(p, step, timestep, callback_kwargs)
            except Exception as exc:
                ctx.setdefault("vram_shared_spill_risk_notes", []).append(
                    f"previous step callback warning at step {step}: {type(exc).__name__}: {exc}"
                )
        try:
            idx = int(step) + 1
        except Exception:
            idx = 0
        _hunyuan_vram_live_snapshot(ctx, torch_module, f"after denoise step {idx}", print_line=True)
        # Keep the report useful during long tests without waiting for final export.
        try:
            _write_vram_lab_integration_report(ctx, result="WARN")
        except Exception:
            pass
        return callback_kwargs

    return _callback

def _hunyuan_vram_install_nontransformer_release_guards(pipe, ctx: dict | None, torch_module=None) -> None:
    """Release text/image/VAE components immediately after their forward calls.

    This tries to control the first pre-denoise spike before parent transformer.forward
    by releasing non-transformer components as soon as they finish their own forward.
    The transformer itself is not moved here.
    """
    if not isinstance(ctx, dict):
        return
    try:
        component_names = ("text_encoder", "text_encoder_2", "image_encoder", "vae")
        wrapped = []
        for comp_name in component_names:
            comp = getattr(pipe, comp_name, None)
            if comp is None:
                continue
            if getattr(comp, "_fv_hunyuan_release_guard_wrapped", False):
                wrapped.append(f"{comp_name}:already")
                continue
            orig_forward = getattr(comp, "forward", None)
            if orig_forward is None or not callable(orig_forward):
                wrapped.append(f"{comp_name}:no-forward")
                continue

            def make_forward(_orig, _name):
                def _wrapped_forward(*args, **kwargs):
                    try:
                        n = int(ctx.get(f"hunyuan_release_guard_{_name}_calls", 0) or 0) + 1
                    except Exception:
                        n = 1
                    ctx[f"hunyuan_release_guard_{_name}_calls"] = n
                    if n <= 3:
                        try:
                            ctx[f"hunyuan_release_guard_{_name}_{n}_before"] = _cuda_mem_line(torch_module)
                        except Exception:
                            pass
                    result = _orig(*args, **kwargs)
                    if n <= 3:
                        try:
                            ctx[f"hunyuan_release_guard_{_name}_{n}_after_raw"] = _cuda_mem_line(torch_module)
                        except Exception:
                            pass
                    try:
                        moved, skipped = _hunyuan_vram_move_components_to_cpu(pipe, (_name,), torch_module, ctx, f"hunyuan_release_guard_{_name}_{n}")
                        ctx[f"hunyuan_release_guard_{_name}_{n}_result"] = f"moved={moved}, skipped={skipped}"
                    except Exception as move_exc:
                        ctx[f"hunyuan_release_guard_{_name}_{n}_result"] = f"FAILED: {type(move_exc).__name__}: {move_exc}"
                    try:
                        _hunyuan_vram_cleanup_cuda_passes(torch_module, 1)
                    except Exception:
                        pass
                    if n <= 3:
                        try:
                            ctx[f"hunyuan_release_guard_{_name}_{n}_after_cleanup"] = _cuda_mem_line(torch_module)
                        except Exception:
                            pass
                    return result
                return _wrapped_forward

            try:
                setattr(comp, "forward", make_forward(orig_forward, comp_name))
                setattr(comp, "_fv_hunyuan_release_guard_wrapped", True)
                wrapped.append(comp_name)
            except Exception as wrap_exc:
                wrapped.append(f"{comp_name}:FAILED:{type(wrap_exc).__name__}")
        ctx["hunyuan_nontransformer_release_guard"] = "YES: " + ", ".join(wrapped) if wrapped else "NO: no components wrapped"
        try:
            print(f"[vram-lab] non-transformer release guards: {ctx['hunyuan_nontransformer_release_guard']}", flush=True)
        except Exception:
            pass
    except Exception as exc:
        ctx["hunyuan_nontransformer_release_guard"] = f"FAILED: {type(exc).__name__}: {exc}"


def _hunyuan_vram_install_parent_transformer_guard(pipe, ctx: dict | None, torch_module=None) -> None:
    """Wrap the parent transformer.forward boundary.

    Hunyuan/Accelerate can create the first denoise VRAM spikes after the
    progress bar reaches 0/N but before transformer block 0 enters the existing
    block hooks. This guard records that parent boundary and runs the safe
    non-transformer purge/cache cleanup as early as we can without moving the
    transformer itself mid-forward.
    """
    if not isinstance(ctx, dict):
        return
    try:
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            ctx["hunyuan_parent_guard_status"] = "NO: transformer not found"
            return
        orig_forward = getattr(transformer, "forward", None)
        if orig_forward is None or not callable(orig_forward):
            ctx["hunyuan_parent_guard_status"] = "NO: transformer.forward not found"
            return
        if getattr(transformer, "_fv_hunyuan_parent_guard_wrapped", False):
            ctx["hunyuan_parent_guard_status"] = "YES: already wrapped"
            return

        def _wrapped_parent_forward(*args, **kwargs):
            try:
                call_no = int(ctx.get("hunyuan_parent_forward_calls", 0) or 0) + 1
            except Exception:
                call_no = 1
            ctx["hunyuan_parent_forward_calls"] = call_no
            ctx["hunyuan_parent_guard_active"] = "YES"
            try:
                ctx[f"hunyuan_parent_call_{call_no}_before"] = _cuda_mem_line(torch_module)
                if call_no == 1:
                    ctx["cuda_before_parent_transformer_forward_first"] = ctx[f"hunyuan_parent_call_{call_no}_before"]
                _vram_lab_note_low_driver_free(ctx, torch_module, f"before parent transformer forward {call_no}")
            except Exception:
                pass

            try:
                _vram_lab_predenoise_component_purge(pipe, torch_module, ctx, trigger=f"parent_transformer.forward.{call_no}")
            except Exception as purge_exc:
                ctx["hunyuan_parent_pre_forward_purge_error"] = f"{type(purge_exc).__name__}: {purge_exc}"
            try:
                _hunyuan_vram_cleanup_cuda_passes(torch_module, 1)
                ctx[f"hunyuan_parent_call_{call_no}_after_guard_cleanup"] = _cuda_mem_line(torch_module)
                if call_no == 1:
                    ctx["cuda_after_parent_transformer_pre_forward_guard_first"] = ctx[f"hunyuan_parent_call_{call_no}_after_guard_cleanup"]
            except Exception:
                pass

            try:
                result = orig_forward(*args, **kwargs)
            except Exception as exc:
                ctx["hunyuan_parent_forward_exception"] = f"{type(exc).__name__}: {exc}"
                try:
                    ctx[f"hunyuan_parent_call_{call_no}_exception_cuda"] = _cuda_mem_line(torch_module)
                except Exception:
                    pass
                raise

            try:
                ctx[f"hunyuan_parent_call_{call_no}_after"] = _cuda_mem_line(torch_module)
                if call_no == 1:
                    ctx["cuda_after_parent_transformer_forward_first"] = ctx[f"hunyuan_parent_call_{call_no}_after"]
                _vram_lab_note_low_driver_free(ctx, torch_module, f"after parent transformer forward {call_no}")
            except Exception:
                pass
            return result

        setattr(_wrapped_parent_forward, "_fv_hunyuan_parent_guard_wrapped", True)
        setattr(transformer, "forward", _wrapped_parent_forward)
        setattr(transformer, "_fv_hunyuan_parent_guard_wrapped", True)
        ctx["hunyuan_parent_guard_status"] = "YES: transformer.forward wrapped"
        try:
            print("[vram-lab] parent transformer pre-forward guard installed", flush=True)
        except Exception:
            pass
    except Exception as exc:
        ctx["hunyuan_parent_guard_status"] = f"FAILED: {type(exc).__name__}: {exc}"


def _vram_lab_install_device_mismatch_diagnostics(pipe, ctx: dict | None, torch_module=None) -> None:
    """Wrap only Hunyuan transformer blocks with tiny failure diagnostics.

    This does not replace or disable VRAM Lab hooks. It only records where a
    CPU/CUDA mismatch occurs so the next patch can target the right boundary.
    """
    if not isinstance(ctx, dict):
        return
    try:
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "named_modules"):
            ctx["vram_device_diag_status"] = "NO: transformer not found"
            return
        wrapped = 0
        named = list(transformer.named_modules())
        selected = []
        for name, module in named:
            full_name = "transformer" if not name else f"transformer.{name}"
            if re.match(r"^transformer\.transformer_blocks\.\d+$", full_name) is not None:
                selected.append((full_name, module))
        if selected:
            try:
                ctx["hunyuan_last_transformer_block_name"] = str(selected[-1][0])
            except Exception:
                pass
        if not selected:
            # Some Diffusers/Hunyuan versions expose a different block naming tree.
            # Fall back to leaf-ish modules so we still learn where the mismatch happens.
            for name, module in named:
                full_name = "transformer" if not name else f"transformer.{name}"
                try:
                    child_count = sum(1 for _ in module.children())
                except Exception:
                    child_count = 1
                if child_count == 0 and hasattr(module, "forward"):
                    selected.append((full_name, module))
                if len(selected) >= 256:
                    break
        for full_name, module in selected[:256]:
            if getattr(module, "_fv_hunyuan_vram_diag_wrapped", False):
                continue
            orig_forward = getattr(module, "forward", None)
            if orig_forward is None:
                continue

            def make_forward(_orig, _name, _module):
                def _wrapped_forward(*args, **kwargs):
                    ctx["vram_active_diag_block"] = _name
                    try:
                        ctx["vram_diag_entered_blocks"] = int(ctx.get("vram_diag_entered_blocks", 0) or 0) + 1
                    except Exception:
                        ctx["vram_diag_entered_blocks"] = 1
                    if not ctx.get("vram_first_block_input_devices"):
                        try:
                            samples = []
                            samples.extend(_vram_lab_tensor_devices(args, limit=8))
                            samples.extend(_vram_lab_tensor_devices(kwargs, limit=8))
                            ctx["vram_first_block_name"] = _name
                            ctx["vram_first_block_input_devices"] = samples[:12] or ["no tensor args sampled"]
                            ctx["vram_first_block_param_devices"] = _vram_lab_param_devices(_module, limit=12)
                        except Exception as e:
                            ctx["vram_first_block_diag_error"] = f"{type(e).__name__}: {e}"
                    try:
                        _vram_lab_predenoise_component_purge(pipe, torch_module, ctx, trigger=_name)
                    except Exception:
                        pass
                    try:
                        _vram_lab_note_low_driver_free(ctx, torch_module, f"enter {_name}")
                    except Exception:
                        pass
                    profile_this_block = False
                    profile_before_cuda = "n/a"
                    profile_before_counts = {}
                    profile_started = 0.0
                    profile_step_index = 0
                    profile_block_index = -1
                    try:
                        if _hunyuan_vram_profiler_enabled(ctx):
                            m_prof = re.search(r"transformer_blocks\.(\d+)", str(_name))
                            profile_block_index = int(m_prof.group(1)) if m_prof else -1
                            total_blocks = int(ctx.get("vram_hooked_block_count_int", 54) or 54)
                            entered = int(ctx.get("vram_diag_entered_blocks", 0) or 0)
                            profile_step_index = int((max(1, entered) - 1) // max(1, total_blocks)) + 1
                            profile_this_block = profile_step_index <= _hunyuan_vram_profiler_steps(ctx)
                            if profile_this_block:
                                profile_before_cuda = _cuda_mem_line(torch_module)
                                profile_before_counts = _hunyuan_vram_profiler_counters(ctx)
                                profile_started = time.perf_counter()
                    except Exception:
                        profile_this_block = False
                    try:
                        _vram_lab_align_current_block_to_inputs(_module, args, kwargs, ctx, _name)
                    except Exception:
                        pass
                    try:
                        result = _orig(*args, **kwargs)
                        if profile_this_block:
                            try:
                                _hunyuan_vram_profiler_record(
                                    ctx,
                                    torch_module,
                                    _name,
                                    profile_step_index,
                                    profile_block_index,
                                    profile_before_cuda,
                                    _cuda_mem_line(torch_module),
                                    (time.perf_counter() - profile_started) * 1000.0,
                                    profile_before_counts,
                                )
                            except Exception as prof_exc:
                                ctx["hunyuan_step_profiler_error"] = f"{type(prof_exc).__name__}: {prof_exc}"
                        try:
                            last_name = str(ctx.get("hunyuan_last_transformer_block_name", ""))
                            if last_name and _name == last_name:
                                try:
                                    n_last = int(ctx.get("hunyuan_last_block_forward_calls", 0) or 0) + 1
                                except Exception:
                                    n_last = 1
                                ctx["hunyuan_last_block_forward_calls"] = n_last
                                try:
                                    target_steps = int(ctx.get("hunyuan_requested_steps", 0) or 0)
                                except Exception:
                                    target_steps = 0
                                if target_steps > 0 and n_last >= target_steps and not ctx.get("hunyuan_early_predecode_purge_done"):
                                    # Do not move transformer/components here. This location is still inside
                                    # the final transformer block wrapper and Diffusers can still use CUDA
                                    # tensors after the block returns. The earlier v1 purge caused CPU/CUDA
                                    # mismatches here. Keep this as a marker only; the real cleanup remains
                                    # at the VAE decode wrapper, where it is safe.
                                    ctx["hunyuan_early_predecode_purge_done"] = "NO: disabled; marker only"
                                    ctx["hunyuan_early_predecode_purge_trigger"] = f"{_name} call {n_last}/{target_steps}"
                                    ctx["hunyuan_early_predecode_marker_cuda"] = _cuda_mem_line(torch_module)
                                    _vram_lab_note_low_driver_free(ctx, torch_module, "early pre-decode marker")
                        except Exception as early_exc:
                            ctx["hunyuan_early_predecode_purge_error"] = f"{type(early_exc).__name__}: {early_exc}"
                        return result
                    except Exception as e:
                        msg = str(e)
                        ctx["vram_diag_exception_block"] = _name
                        ctx["vram_diag_exception_type"] = type(e).__name__
                        ctx["vram_diag_exception_message"] = msg[:1000]
                        try:
                            samples = []
                            samples.extend(_vram_lab_tensor_devices(args, limit=10))
                            samples.extend(_vram_lab_tensor_devices(kwargs, limit=10))
                            ctx["vram_failing_input_devices"] = samples[:16] or ["no tensor args sampled"]
                            ctx["vram_failing_param_devices"] = _vram_lab_param_devices(_module, limit=16)
                        except Exception as de:
                            ctx["vram_diag_exception_sampling_error"] = f"{type(de).__name__}: {de}"
                        raise
                    finally:
                        ctx["vram_last_diag_block"] = _name
                return _wrapped_forward

            module.forward = make_forward(orig_forward, full_name, module)  # type: ignore[method-assign]
            setattr(module, "_fv_hunyuan_vram_diag_wrapped", True)
            wrapped += 1
        ctx["vram_device_diag_status"] = f"YES: wrapped {wrapped} transformer blocks"
        print(f"[vram-lab] device mismatch diagnostics wrapped {wrapped} transformer blocks", flush=True)
    except Exception as e:
        ctx["vram_device_diag_status"] = f"FAILED: {type(e).__name__}: {e}"


def _attach_vram_lab_forward_hooks(pipe, torch, ctx: dict | None) -> None:
    """Attach real forward-time VRAM Lab hooks to Hunyuan transformer blocks.

    Hunyuan remains the runner. VRAM Lab only wraps live module/block forwards so
    the report can prove whether real execution entered VRAM-managed blocks.
    """
    if not isinstance(ctx, dict):
        return
    try:
        root = project_root()
        lab_dir = root / "tools" / "vram_lab"
        if str(lab_dir) not in sys.path:
            sys.path.insert(0, str(lab_dir))
        import vram_forward_hooks as vfh  # type: ignore

        transformer = getattr(pipe, "transformer", None)
        component_map = {"transformer": transformer} if transformer is not None else {}
        mode = _normalize_vram_lab_mode(str(ctx.get("mode", "safe") or "safe"))
        policy = vfh.apply_vram_lab_profile_defaults({
            "mode": mode,
            "device": "cuda",
            # Start with the main heavy transformer blocks. Context/refiner blocks
            # can remain under Diffusers offload until the core path is stable.
            "hook_name_regex": r"^transformer\.transformer_blocks\.\d+$",
            "max_blocks": 256,
        }, mode)
        policy = _vram_lab_apply_hunyuan_step_policy(policy, ctx, torch)
        ctx["vram_profile_note"] = str(policy.get("profile_note", "n/a"))
        ctx["vram_aggressive_extra_gb"] = f"{float(policy.get('aggressive_extra_gb', 0.0) or 0.0):.1f}"
        runtime = vfh.attach_vram_hooks(component_map, policy=policy, torch_module=torch)
        ctx["_vram_runtime"] = runtime
        _vram_lab_force_runtime_hunyuan_budget(runtime, ctx)
        runtime.update_context(ctx)
        # update_context may report the runtime's original profile fields; keep a
        # clear Hunyuan override marker in the report so we can verify whether the
        # stricter step policy was applied to the live runtime object.
        _vram_lab_force_runtime_hunyuan_budget(runtime, ctx)
        _hunyuan_vram_install_nontransformer_release_guards(pipe, ctx, torch)
        _hunyuan_vram_install_parent_transformer_guard(pipe, ctx, torch)
        _vram_lab_install_device_mismatch_diagnostics(pipe, ctx, torch)
        if int(ctx.get("vram_hooked_block_count_int", 0) or 0) > 0:
            ctx["vram_forward_hooks_attached"] = "YES"
            ctx["activation_status"] = str(ctx.get("activation_status", "PASS")) + "; PASS: forward-time block hooks attached"
            print(f"[vram-lab] forward hooks attached: {ctx.get('vram_hooked_block_count', '0')} blocks", flush=True)
        else:
            ctx["vram_forward_hooks_attached"] = "NO: no transformer blocks discovered"
            ctx["activation_status"] = str(ctx.get("activation_status", "PASS")) + "; WARN: no forward-time blocks discovered"
            print("[vram-lab] WARNING: no transformer blocks discovered for forward hooks", flush=True)
    except Exception as e:
        ctx["vram_forward_hooks_attached"] = f"FAILED: {type(e).__name__}: {e}"
        ctx["activation_status"] = str(ctx.get("activation_status", "PASS")) + "; FAIL: forward-hook attach failed"
        print(f"[vram-lab] forward hook attach failed: {e}", flush=True)

def _write_vram_lab_integration_report(ctx: dict, result: str = "WARN", error: str = "") -> None:
    """Write the Hunyuan→VRAM Lab integration report.

    This report is intentionally separate from the standalone 0.3.1 stress report.
    Hunyuan remains the video runner; VRAM Lab only reports activation/planning status.
    """
    if not isinstance(ctx, dict) or not ctx.get("report_path"):
        return
    _vram_lab_refresh_forward_hook_status(ctx)
    result = str(result or "WARN").upper().strip()
    if result not in {"PASS", "WARN", "FAIL"}:
        result = "FAIL"

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("FrameVision Hunyuan → VRAM Lab Integration")
    lines.append("=" * 78)
    lines.append("Scope: Hunyuan owns generation; VRAM Lab owns memory planning/control.")
    lines.append(f"Started: {ctx.get('started', 'n/a')}")
    lines.append(f"Updated: {ctx.get('updated', 'n/a')}")
    lines.append(f"Python: {sys.executable}")
    lines.append("")
    lines.append("Hunyuan")
    lines.append("-" * 78)
    lines.append(f"selected Hunyuan model: {ctx.get('model_key', 'n/a')}")
    lines.append(f"resolved model id: {ctx.get('model_id', 'n/a')}")
    lines.append(f"selected VRAM Lab mode: {ctx.get('mode', 'off')}")
    lines.append(f"aggressive extra allowance: {ctx.get('aggressive_extra_gb', ctx.get('vram_aggressive_extra_gb', '0.0'))} GB")
    lines.append(f"model root: {ctx.get('model_root', 'n/a')}")
    lines.append(f"Hunyuan generation status: {ctx.get('generation_status', 'not started')}")
    lines.append(f"output path: {ctx.get('output_path', 'n/a')}")
    lines.append("")
    lines.append("VRAM Lab")
    lines.append("-" * 78)
    lines.append(f"GPU name: {ctx.get('gpu_name', 'n/a')}")
    lines.append(f"total VRAM: {ctx.get('total_vram', 'n/a')}")
    lines.append(f"selected VRAM Lab profile: {ctx.get('profile', 'n/a')}")
    lines.append(f"configured planning budget: {ctx.get('budget', 'n/a')}")
    lines.append(f"planning workspace reserve: {ctx.get('workspace_reserve', 'n/a')}")
    lines.append(f"configured generation-safe budget: {ctx.get('generation_budget', 'n/a')}")
    lines.append(f"generation workspace reserve: {ctx.get('generation_reserve', 'n/a')}")
    lines.append(f"generation budget policy: {ctx.get('generation_policy_note', 'n/a')}")
    lines.append(f"step hook profile note: {ctx.get('vram_profile_note', 'n/a')}")
    lines.append(f"Hunyuan step policy: {ctx.get('hunyuan_step_policy', 'n/a')}")
    lines.append(f"Hunyuan step hot budget: {ctx.get('hunyuan_step_policy_hot_budget', 'n/a')}")
    lines.append(f"Hunyuan step min driver-free: {ctx.get('hunyuan_step_policy_min_driver_free', 'n/a')}")
    lines.append(f"Hunyuan runtime budget override: {ctx.get('hunyuan_runtime_budget_override', 'n/a')}")
    lines.append(f"Hunyuan UI startup/text guard: {ctx.get('hunyuan_text_stage_hot_budget', ctx.get('hunyuan_ui_vram_text_hot_gb', 'n/a'))}")
    lines.append(f"Hunyuan UI after/finalize guard: {ctx.get('hunyuan_after_stage_hot_budget', ctx.get('hunyuan_ui_vram_after_hot_gb', 'n/a'))}")
    lines.append(f"Hunyuan decode min-free guard: {ctx.get('hunyuan_decode_guard_min_free_gb', ctx.get('hunyuan_ui_vram_decode_min_free_gb', 'n/a'))}")
    lines.append(f"Hunyuan decode cleanup passes: {ctx.get('hunyuan_decode_guard_cleanup_passes', ctx.get('hunyuan_ui_vram_decode_cleanup_passes', 'n/a'))}")
    lines.append(f"Hunyuan decode chunk frames: {ctx.get('hunyuan_decode_guard_chunk_frames', ctx.get('hunyuan_ui_vram_decode_chunk_frames', 'n/a'))}")
    lines.append(f"Hunyuan decode chunk overlap: {ctx.get('hunyuan_decode_guard_chunk_overlap', ctx.get('hunyuan_ui_vram_decode_chunk_overlap', 'n/a'))}")
    lines.append("Hunyuan decode chunk dimension rule: B,C,T,H,W -> chunk dim 2 only")
    lines.append(f"Hunyuan decode unload blocks: {ctx.get('hunyuan_decode_guard_unload_blocks', ctx.get('hunyuan_ui_vram_decode_unload_blocks', 'n/a'))}")
    lines.append(f"Hunyuan decode output to CPU setting: {ctx.get('hunyuan_decode_guard_output_cpu_enabled', ctx.get('hunyuan_ui_vram_decode_output_cpu', 'n/a'))}")
    lines.append(f"Hunyuan step profiler: {ctx.get('hunyuan_ui_vram_step_profiler', False)}")
    lines.append(f"Hunyuan step profiler steps: {ctx.get('hunyuan_ui_vram_step_profiler_steps', 'n/a')}")
    lines.append(f"Hunyuan step cleanup every N blocks: {ctx.get('hunyuan_step_cleanup_every', ctx.get('hunyuan_ui_vram_step_cleanup_every', 'n/a'))}")
    lines.append(f"Hunyuan step empty cache after unload: {ctx.get('hunyuan_step_empty_cache', ctx.get('hunyuan_ui_vram_step_empty_cache', 'n/a'))}")
    lines.append(f"Hunyuan step release after forward: {ctx.get('hunyuan_step_release_after_forward', ctx.get('hunyuan_ui_vram_step_release_after_forward', 'n/a'))}")
    lines.append(f"Hunyuan step unload before load: {ctx.get('hunyuan_step_unload_before_load', ctx.get('hunyuan_ui_vram_step_unload_before_load', 'n/a'))}")
    lines.append(f"Hunyuan residency strategy: {ctx.get('hunyuan_step_residency_strategy', 'n/a')}")
    lines.append(f"Hunyuan stable hotset fraction: {ctx.get('hunyuan_step_stable_hotset_fraction', 'n/a')}")
    lines.append(f"Hunyuan stable hotset budget: {ctx.get('hunyuan_step_stable_hotset_budget', 'n/a')}")
    lines.append(f"allocator config requested: {ctx.get('allocator_config_requested', 'n/a')}")
    lines.append(f"allocator config effective/too-late status: {ctx.get('allocator_config_status', 'n/a')}")
    lines.append(f"full pipe.to(\"cuda\") avoided: {ctx.get('pipe_to_cuda_avoided', 'n/a')}")
    lines.append(f"CPU/offload-aware loading path used: {ctx.get('cpu_offload_path_used', 'n/a')}")
    lines.append(f"CPU text encode forced by VRAM Lab: {ctx.get('cpu_text_encode_forced', 'n/a')}")
    lines.append(f"CPU text embeddings kept on CPU: {ctx.get('cpu_text_embeddings_kept_on_cpu', 'n/a')}")
    lines.append(f"CPU text encode components to CPU: {ctx.get('cpu_text_encode_components_to_cpu', 'n/a')}")
    lines.append(f"CPU text encode call embed keys: {ctx.get('cpu_text_encode_call_embed_keys', 'n/a')}")
    lines.append(f"memory-saver policy: {ctx.get('memory_saver_policy', 'n/a')}")
    lines.append(f"memory-saver forced/enabled: {ctx.get('memory_saver_forced', 'n/a')}")
    lines.append(f"attention slicing forced/enabled: {ctx.get('attention_slicing_forced', 'n/a')}")
    lines.append(f"attention slicing applied: {ctx.get('attention_slicing_applied', 'n/a')}")
    lines.append(f"VAE slicing forced/enabled: {ctx.get('vae_slicing_forced', 'n/a')}")
    lines.append(f"VAE slicing applied: {ctx.get('vae_slicing_applied', 'n/a')}")
    lines.append(f"VAE tiling forced/enabled: {ctx.get('vae_tiling_forced', 'n/a')}")
    lines.append(f"VAE tiling applied: {ctx.get('vae_tiling_applied', 'n/a')}")
    lines.append(f"Hunyuan pre-decode guard installed: {ctx.get('hunyuan_decode_guard_installed', 'n/a')}")
    lines.append(f"Hunyuan pre-decode guard calls: {ctx.get('hunyuan_decode_guard_calls', '0')}")
    lines.append(f"Hunyuan decode output to CPU: {ctx.get('hunyuan_decode_output_to_cpu', 'n/a')}")
    lines.append(f"Hunyuan decode guard duration: {ctx.get('hunyuan_decode_guard_last_duration', 'n/a')}")
    lines.append(f"Hunyuan early pre-decode purge: {ctx.get('hunyuan_early_predecode_purge_done', 'NO')}")
    lines.append(f"Hunyuan early pre-decode trigger: {ctx.get('hunyuan_early_predecode_purge_trigger', 'n/a')}")
    lines.append(f"Hunyuan early pre-decode marker CUDA: {ctx.get('hunyuan_early_predecode_marker_cuda', 'n/a')}")
    lines.append(f"Hunyuan early pre-decode duration: {ctx.get('hunyuan_early_predecode_purge_duration', 'n/a')}")
    lines.append(f"model/group/sequential offload used: {ctx.get('offload_mode_used', 'n/a')}")
    lines.append(f"generation setting adjustment: {ctx.get('generation_setting_adjustment', 'none')}")
    lines.append(f"Hunyuan step telemetry callback: {ctx.get('hunyuan_step_telemetry_callback', 'n/a')}")
    lines.append(f"Hunyuan non-transformer release guard: {ctx.get('hunyuan_nontransformer_release_guard', 'n/a')}")
    lines.append(f"Hunyuan parent transformer guard: {ctx.get('hunyuan_parent_guard_status', 'n/a')}")
    lines.append(f"Hunyuan parent transformer forward calls: {ctx.get('hunyuan_parent_forward_calls', 'n/a')}")
    lines.append(f"CUDA before parent transformer forward first: {ctx.get('cuda_before_parent_transformer_forward_first', 'n/a')}")
    lines.append(f"CUDA after parent pre-forward guard first: {ctx.get('cuda_after_parent_transformer_pre_forward_guard_first', 'n/a')}")
    lines.append(f"CUDA after parent transformer forward first: {ctx.get('cuda_after_parent_transformer_forward_first', 'n/a')}")
    for _cg_name in ("text_encoder", "text_encoder_2", "image_encoder", "vae"):
        _cg_calls = ctx.get(f"hunyuan_release_guard_{_cg_name}_calls", "n/a")
        lines.append(f"release guard {_cg_name} calls: {_cg_calls}")
        if str(_cg_calls) not in ("n/a", "0"):
            lines.append(f"  first before: {ctx.get(f'hunyuan_release_guard_{_cg_name}_1_before', 'n/a')}")
            lines.append(f"  first after raw: {ctx.get(f'hunyuan_release_guard_{_cg_name}_1_after_raw', 'n/a')}")
            lines.append(f"  first after cleanup: {ctx.get(f'hunyuan_release_guard_{_cg_name}_1_after_cleanup', 'n/a')}")
            lines.append(f"  first result: {ctx.get(f'hunyuan_release_guard_{_cg_name}_1_result', 'n/a')}")
    lines.append(f"VRAM Lab forward hooks attached: {ctx.get('vram_forward_hooks_attached', 'n/a')}")
    lines.append(f"hooked component names: {ctx.get('vram_hooked_component_names', 'n/a')}")
    lines.append(f"hooked block count: {ctx.get('vram_hooked_block_count', 'n/a')}")
    lines.append(f"pre-forward hook calls: {ctx.get('vram_pre_forward_calls', 'n/a')}")
    lines.append(f"post-forward hook calls: {ctx.get('vram_post_forward_calls', 'n/a')}")
    lines.append(f"block load count: {ctx.get('vram_block_load_count', 'n/a')}")
    lines.append(f"block unload count: {ctx.get('vram_block_unload_count', 'n/a')}")
    lines.append(f"forced unload count: {ctx.get('vram_forced_unload_count', 'n/a')}")
    lines.append(f"cache cleanup count: {ctx.get('vram_cache_cleanup_count', 'n/a')}")
    lines.append(f"emergency cleanup count: {ctx.get('vram_emergency_cleanup_count', 'n/a')}")
    lines.append(f"hot-block trim count: {ctx.get('vram_hot_block_trim_count', 'n/a')}")
    lines.append(f"hot-block budget: {ctx.get('vram_hot_block_budget', 'n/a')}")
    lines.append(f"hook block pattern: {ctx.get('vram_hook_block_pattern', 'n/a')}")
    lines.append(f"hooked blocks currently on CUDA: {ctx.get('vram_blocks_currently_cuda', 'n/a')}")
    lines.append(f"VRAM Lab retained CUDA tensors: {ctx.get('vram_retained_cuda_refs', 'n/a')}")
    lines.append(f"per-process hard cap status: {ctx.get('per_process_cap_status', 'n/a')}")
    lines.append(f"sample hooked block names: {ctx.get('vram_sample_hooked_block_names', 'n/a')}")
    lines.append(f"active/last hooked block: {ctx.get('vram_active_block_name', 'n/a')}")
    lines.append(f"device diagnostic wrapper: {ctx.get('vram_device_diag_status', 'n/a')}")
    lines.append(f"device diagnostic entered blocks: {ctx.get('vram_diag_entered_blocks', '0')}")
    lines.append(f"device diagnostic active/last block: {ctx.get('vram_active_diag_block', ctx.get('vram_last_diag_block', 'n/a'))}")
    lines.append(f"device diagnostic exception block: {ctx.get('vram_diag_exception_block', 'n/a')}")
    lines.append(f"device diagnostic exception: {ctx.get('vram_diag_exception_type', 'n/a')}: {ctx.get('vram_diag_exception_message', 'n/a')}")
    lines.append(f"device align fallback count: {ctx.get('vram_device_align_count', '0')}")
    lines.append(f"device align active/last block: {ctx.get('vram_device_align_active_block', 'n/a')}")
    lines.append(f"device align last before: {ctx.get('vram_device_align_last_before', 'n/a')}")
    lines.append(f"device align last after: {ctx.get('vram_device_align_last_after', 'n/a')}")
    if ctx.get('vram_device_align_error'):
        lines.append(f"device align error: {ctx.get('vram_device_align_error')}")
    if ctx.get('vram_device_align_notes'):
        lines.append("device align notes:")
        for item in list(ctx.get('vram_device_align_notes') or [])[:8]:
            lines.append(f"  - {item}")
    if ctx.get('vram_first_block_input_devices'):
        lines.append("first block input tensor devices:")
        for item in list(ctx.get('vram_first_block_input_devices') or [])[:12]:
            lines.append(f"  - {item}")
    if ctx.get('vram_first_block_param_devices'):
        lines.append("first block direct parameter/buffer devices:")
        for item in list(ctx.get('vram_first_block_param_devices') or [])[:12]:
            lines.append(f"  - {item}")
    if ctx.get('vram_failing_input_devices'):
        lines.append("failing block input tensor devices:")
        for item in list(ctx.get('vram_failing_input_devices') or [])[:16]:
            lines.append(f"  - {item}")
    if ctx.get('vram_failing_param_devices'):
        lines.append("failing block direct parameter/buffer devices:")
        for item in list(ctx.get('vram_failing_param_devices') or [])[:16]:
            lines.append(f"  - {item}")
    lines.append(f"peak CUDA during hooked execution: {ctx.get('vram_peak_cuda_during_hooked_execution', 'n/a')}")
    lines.append(f"VRAM hook failures: {ctx.get('vram_hook_failures', 'none')}")
    lines.append(f"failure stage: {ctx.get('failure_stage', 'n/a')}")
    lines.append(f"CUDA before pipeline load: {ctx.get('cuda_before_pipeline_load', 'n/a')}")
    lines.append(f"CUDA after pipeline load: {ctx.get('cuda_after_pipeline_load', 'n/a')}")
    lines.append(f"CUDA after offload hook: {ctx.get('cuda_after_offload_hook', 'n/a')}")
    lines.append(f"CUDA before generation: {ctx.get('cuda_before_generation_call', 'n/a')}")
    lines.append(f"CUDA after generation attempt: {ctx.get('cuda_after_generation_attempt', 'n/a')}")
    lines.append(f"CUDA before pre-denoise purge: {ctx.get('cuda_before_predenoise_purge', 'n/a')}")
    lines.append(f"CUDA after pre-denoise purge: {ctx.get('cuda_after_predenoise_purge', 'n/a')}")
    lines.append(f"pre-denoise purge trigger: {ctx.get('vram_predenoise_purge_trigger', 'n/a')}")
    lines.append(f"pre-denoise purge moved: {ctx.get('vram_predenoise_purge_moved', 'n/a')}")
    lines.append(f"pre-denoise purge skipped: {ctx.get('vram_predenoise_purge_skipped', 'n/a')}")
    if ctx.get('vram_predenoise_purge_error'):
        lines.append(f"pre-denoise purge error: {ctx.get('vram_predenoise_purge_error')}")
    lines.append("early pre-decode purge status: disabled after CPU/CUDA mismatch; decode wrapper guard remains active")
    lines.append(f"CUDA before early pre-decode purge: {ctx.get('cuda_before_hunyuan_early_predecode_purge', 'n/a')}")
    lines.append(f"CUDA after early pre-decode purge: {ctx.get('cuda_after_hunyuan_early_predecode_purge', 'n/a')}")
    lines.append(f"early pre-decode block purge: {ctx.get('hunyuan_early_predecode_block_purge_result', 'n/a')}")
    lines.append(f"early pre-decode purge moved: {ctx.get('hunyuan_early_predecode_purge_components_moved', 'n/a')}")
    lines.append(f"early pre-decode purge skipped: {ctx.get('hunyuan_early_predecode_purge_components_skipped', 'n/a')}")
    lines.append(f"CUDA before hidden VAE decode: {ctx.get('cuda_before_hunyuan_vae_decode', 'n/a')}")
    lines.append(f"CUDA after pre-decode purge: {ctx.get('cuda_after_hunyuan_pre_decode_purge', 'n/a')}")
    lines.append(f"pre-decode purge moved: {ctx.get('hunyuan_pre_decode_purge_components_moved', 'n/a')}")
    lines.append(f"pre-decode purge skipped: {ctx.get('hunyuan_pre_decode_purge_components_skipped', 'n/a')}")
    lines.append(f"decode block purge 1 moved: {ctx.get('hunyuan_decode_block_purge_1_moved', 'n/a')}")
    lines.append(f"decode block purge 1 skipped: {ctx.get('hunyuan_decode_block_purge_1_skipped', 'n/a')}")
    lines.append(f"decode block purge 2 moved: {ctx.get('hunyuan_decode_block_purge_2_moved', 'n/a')}")
    lines.append(f"decode block purge 2 skipped: {ctx.get('hunyuan_decode_block_purge_2_skipped', 'n/a')}")
    lines.append(f"CUDA at VAE decode start: {ctx.get('cuda_at_hunyuan_vae_decode_start', 'n/a')}")
    lines.append(f"CUDA after VAE decode raw: {ctx.get('cuda_after_hunyuan_vae_decode_raw', 'n/a')}")
    lines.append(f"decode call 1 input shape: {ctx.get('hunyuan_decode_call_1_input_shape', 'n/a')}")
    lines.append(f"decode call 1 chunked: {ctx.get('hunyuan_decode_call_1_chunked', 'n/a')}")
    lines.append(f"decode call 2 input shape: {ctx.get('hunyuan_decode_call_2_input_shape', 'n/a')}")
    lines.append(f"decode call 2 chunked: {ctx.get('hunyuan_decode_call_2_chunked', 'n/a')}")
    lines.append(f"CUDA after post-decode VAE purge: {ctx.get('cuda_after_hunyuan_post_decode_vae_purge', 'n/a')}")
    lines.append(f"CUDA after decode guard cleanup: {ctx.get('cuda_after_hunyuan_decode_guard_cleanup', 'n/a')}")
    if ctx.get('hunyuan_decode_guard_exception'):
        lines.append(f"Hunyuan decode guard exception: {ctx.get('hunyuan_decode_guard_exception')}")
    lines.append(f"lowest driver-free VRAM: {ctx.get('vram_lowest_driver_free', 'n/a')} at {ctx.get('vram_lowest_driver_free_stage', 'n/a')}")
    if ctx.get('vram_shared_spill_risk_notes'):
        lines.append("shared-memory spill risk notes:")
        for item in list(ctx.get('vram_shared_spill_risk_notes') or [])[:16]:
            lines.append(f"  - {item}")
    prof_rows = ctx.get("hunyuan_step_profiler_rows")
    if isinstance(prof_rows, list) and prof_rows:
        lines.append("Hunyuan step profiler rows:")
        for row in prof_rows[:160]:
            try:
                lines.append(
                    "  - "
                    f"step {row.get('step')} block {int(row.get('block', -1)):>2} "
                    f"{row.get('name')} | {float(row.get('ms', 0.0)):.1f} ms | "
                    f"driver_free={row.get('driver_free')} | {row.get('delta')} | "
                    f"before=[{row.get('before')}] after=[{row.get('after')}]"
                )
            except Exception:
                lines.append(f"  - {row}")
    if ctx.get("hunyuan_step_profiler_error"):
        lines.append(f"Hunyuan step profiler error: {ctx.get('hunyuan_step_profiler_error')}")
    live_keys = sorted(k for k in ctx.keys() if str(k).startswith("hunyuan_live_"))
    if live_keys:
        lines.append("Hunyuan live step telemetry:")
        for k in live_keys[-16:]:
            lines.append(f"  - {k.replace('hunyuan_live_', '').replace('_', ' ')}: {ctx.get(k)}")
    lines.append(f"finalize guard enabled: {ctx.get('finalize_guard_enabled', 'NO')}")
    lines.append(f"finalize hooks detached: {ctx.get('finalize_hooks_detached', ctx.get('vram_hooks_detached', 'n/a'))}")
    lines.append(f"finalize components to CPU: {ctx.get('finalize_components_to_cpu', 'n/a')}")
    lines.append(f"finalize generated output moved to CPU: {ctx.get('finalize_output_to_cpu', 'n/a')}")
    lines.append(f"CUDA after denoise/generation: {ctx.get('cuda_after_denoise_generation', 'n/a')}")
    lines.append(f"CUDA before save/re-encode: {ctx.get('cuda_before_save_reencode', 'n/a')}")
    lines.append(f"CUDA after save/re-encode: {ctx.get('cuda_after_save_reencode', 'n/a')}")
    lines.append(f"CUDA after final cleanup: {ctx.get('finalize_guard_end_memory', 'n/a')}")
    lines.append(f"finalize save/re-encode duration: {ctx.get('finalize_save_reencode_duration', 'n/a')}")
    lines.append(f"finalize guard total duration: {ctx.get('finalize_guard_total_duration', 'n/a')}")
    lines.append(f"safetensors files found: {ctx.get('safetensors_count', 'n/a')}")
    for sf in ctx.get("safetensors_files", [])[:20]:
        lines.append(f"  - {sf}")
    extra = max(0, int(ctx.get("safetensors_count_int", 0) or 0) - 20)
    if extra:
        lines.append(f"  ... {extra} more safetensors files")
    lines.append(f"combined estimated model size: {ctx.get('estimated_size', 'n/a')}")
    lines.append(f"whether full CUDA load would be refused: {ctx.get('full_refused', 'n/a')}")
    lines.append(f"adapter/contract status: {ctx.get('adapter_status', 'n/a')}")
    lines.append(f"hook/activation status: {ctx.get('activation_status', 'n/a')}")
    lines.append(f"component/group summary: {ctx.get('group_summary', 'n/a')}")
    notes = ctx.get("notes", [])
    stage_notes = ctx.get("stage_notes", [])
    finalize_stages = ctx.get("finalize_guard_stages", [])
    finalize_notes = ctx.get("finalize_guard_notes", [])
    if notes or stage_notes or finalize_stages or finalize_notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-" * 78)
        for note in notes:
            lines.append(str(note))
        if stage_notes:
            lines.append("")
            lines.append("Best-effort stage memory notes")
            for note in stage_notes:
                lines.append(str(note))
        if finalize_stages:
            lines.append("")
            lines.append("VRAM Lab finalize guard stages")
            for note in finalize_stages:
                lines.append(str(note))
        if finalize_notes:
            lines.append("")
            lines.append("VRAM Lab finalize guard notes")
            for note in finalize_notes:
                lines.append(str(note))
    if error:
        lines.append("")
        lines.append("Error")
        lines.append("-" * 78)
        lines.append(str(error))
    lines.append("")
    lines.append(f"PASS/WARN/FAIL decision: {result}")
    lines.append(f"Next recommended step: {ctx.get('next_step', 'Run a tiny Hunyuan job with VRAM Lab mode enabled and inspect this report/log.')}")
    lines.append("")
    lines.append(f"HUNYUAN VRAM LAB INTEGRATION RESULT: {result}")

    path = Path(str(ctx["report_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_vram_lab_integration(result: str, generation_status: str, error: str = "") -> None:
    global _VRAM_LAB_CONTEXT
    ctx = _VRAM_LAB_CONTEXT
    if not isinstance(ctx, dict):
        return
    ctx["updated"] = str(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ctx["generation_status"] = generation_status
    try:
        import torch as _torch  # type: ignore
        ctx["cuda_after_generation_attempt"] = _cuda_mem_line(_torch)
    except Exception:
        pass

    _vram_lab_refresh_forward_hook_status(ctx)
    try:
        pre_calls = int(ctx.get("vram_pre_forward_calls_int", 0) or 0)
        post_calls = int(ctx.get("vram_post_forward_calls_int", 0) or 0)
        block_count = int(ctx.get("vram_hooked_block_count_int", 0) or 0)
    except Exception:
        pre_calls = post_calls = block_count = 0

    low_free_warn = False
    try:
        low_free = ctx.get("vram_lowest_driver_free_bytes")
        if low_free is not None and int(low_free) < int(4.0 * (1024 ** 3)):
            low_free_warn = True
    except Exception:
        low_free_warn = False

    if generation_status == "completed" and pre_calls > 0 and low_free_warn:
        result = "WARN"
        ctx["failure_stage"] = "completed but memory boundary was not safe"
        ctx["next_step"] = "Hunyuan completed, but driver-free VRAM dropped below the reference safety margin during hooked execution. Treat this as VRAM Lab not yet protecting shared-memory spill for large jobs."
    elif generation_status == "completed" and pre_calls > 0:
        result = "PASS"
        ctx["failure_stage"] = "n/a"
        ctx["next_step"] = "Hunyuan completed with VRAM Lab forward hooks active and no recorded driver-free danger. Next work can optimize residency policy and remove remaining Diffusers/Accelerate overlap where useful."
    elif generation_status != "completed" and pre_calls > 0:
        result = "WARN"
        if ctx.get("vram_diag_exception_block"):
            ctx["failure_stage"] = f"inside hooked block: {ctx.get('vram_diag_exception_block')}"
        elif post_calls < pre_calls:
            ctx["failure_stage"] = "during hooked transformer execution"
        else:
            ctx["failure_stage"] = "after hooked transformer execution or during VAE decode/export"
        ctx["next_step"] = "Forward hooks fired, so VRAM Lab reached live Hunyuan execution. Next patch should focus only on the reported failure stage, not scan/planning/cap tweaks."
    elif block_count > 0 and pre_calls <= 0:
        ctx["failure_stage"] = "before hooks"
        ctx["next_step"] = "Forward hooks attached but were never called. Find why generation failed before transformer execution or why Diffusers bypassed those modules."
    else:
        result = "FAIL"
        ctx["failure_stage"] = "hooks did not attach"
        ctx["next_step"] = "Hooks did not attach or counters stayed zero; inspect transformer module discovery and avoid more scan-only patches."

    _write_vram_lab_integration_report(ctx, result=result, error=error)
    _vram_lab_detach_forward_hooks(ctx)


def _detect_total_vram_gb_cli() -> float:
    """Best-effort GPU VRAM detection for Hunyuan CLI profile defaults."""
    try:
        import subprocess as _subprocess
        out = _subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=_subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        vals = []
        for line in str(out).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line.split()[0]) / 1024.0)
            except Exception:
                pass
        if vals:
            return max(vals)
    except Exception:
        pass
    return 0.0


def _resolve_vram_lab_profile_cli(profile: str | None) -> str:
    key = str(profile or "auto").lower().strip()
    if key not in {"auto", "16", "24"}:
        key = "auto"
    if key == "auto":
        # Below 23.01 GB -> 16 GB profile; above 23 GB -> 24 GB profile.
        return "24" if _detect_total_vram_gb_cli() > 23.0 else "16"
    return key


def _profile_hotset_budget_gb_cli(profile: str | None) -> float:
    return 9.0 if _resolve_vram_lab_profile_cli(profile) == "24" else 1.0

def _profile_step_hot_gb_cli(profile: str | None) -> float:
    return 9.0 if _resolve_vram_lab_profile_cli(profile) == "24" else 1.0


def _activate_vram_lab_memory_control(root: Path, torch, args, model_key: str, model_id: str, local_dir: Path) -> dict | None:
    """Activate the Hunyuan→VRAM Lab integration gate.

    This is deliberately a Hunyuan-side call into the standalone VRAM Lab API.
    It does not make VRAM Lab a Hunyuan runner and it does not generate video.
    """
    global _VRAM_LAB_CONTEXT
    raw_mode = str(getattr(args, "vram_lab", "off") or "off").lower().strip()
    mode = _normalize_vram_lab_mode(raw_mode)
    try:
        setattr(args, "vram_lab", mode)
    except Exception:
        pass
    if not _vram_lab_enabled(mode):
        return None

    lab_dir = root / "tools" / "vram_lab"
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))

    started = str(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ctx: dict = {
        "started": started,
        "updated": started,
        "mode": mode,
        "aggressive_extra_gb": "5.0" if mode == "aggressive" else "0.0",
        "model_key": str(model_key),
        "model_id": str(model_id),
        "model_root": str(local_dir),
        "report_path": str(_vram_lab_report_path(root)),
        "generation_status": "starting",
        "adapter_status": "not started",
        "activation_status": "not started",
        "allocator_config_requested": "expandable_segments:True",
        "allocator_config_status": str(getattr(args, "_vram_lab_allocator_env_status", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set"))),
        "hunyuan_ui_vram_text_hot_gb": float(getattr(args, "vram_lab_text_hot_gb", 9.0) or 9.0),
        "hunyuan_ui_vram_step_hot_gb": float(
            _profile_step_hot_gb_cli(getattr(args, "vram_lab_profile", "auto"))
            if getattr(args, "vram_lab_step_hot_gb", None) is None
            else getattr(args, "vram_lab_step_hot_gb")
        ),
        "hunyuan_ui_vram_after_hot_gb": float(getattr(args, "vram_lab_after_hot_gb", 0.0)),
        "hunyuan_ui_vram_decode_min_free_gb": float(getattr(args, "vram_lab_decode_min_free_gb", 0.5) or 0.5),
        "hunyuan_ui_vram_decode_cleanup_passes": int(getattr(args, "vram_lab_decode_cleanup_passes", 2) or 2),
        "hunyuan_ui_vram_decode_chunk_frames": int(getattr(args, "vram_lab_decode_chunk_frames", 8) or 8),
        "hunyuan_ui_vram_decode_chunk_overlap": int(getattr(args, "vram_lab_decode_chunk_overlap", 4) or 4),
        "hunyuan_ui_vram_decode_unload_blocks": bool(getattr(args, "vram_lab_decode_unload_blocks", True)),
        "hunyuan_ui_vram_decode_output_cpu": bool(getattr(args, "vram_lab_decode_output_cpu", True)),
        "hunyuan_ui_vram_min_free_gb": float(getattr(args, "vram_lab_min_free_gb", 0.5) or 0.5),
        "hunyuan_requested_steps": int(getattr(args, "steps", 0) or 0),
        "hunyuan_ui_vram_step_profiler": bool(getattr(args, "vram_lab_step_profiler", False)),
        "hunyuan_ui_vram_step_profiler_steps": int(getattr(args, "vram_lab_step_profiler_steps", 1) or 1),
        "hunyuan_ui_vram_step_cleanup_every": int(getattr(args, "vram_lab_step_cleanup_every", 4) or 0),
        "hunyuan_ui_vram_step_empty_cache": bool(getattr(args, "vram_lab_step_empty_cache", True)),
        "hunyuan_ui_vram_step_release_after_forward": bool(getattr(args, "vram_lab_step_release_after_forward", True)),
        "hunyuan_ui_vram_step_unload_before_load": bool(getattr(args, "vram_lab_step_unload_before_load", True)),
        "hunyuan_ui_vram_residency": str(getattr(args, "vram_lab_residency", "planned_hotset") or "planned_hotset"),
        "hunyuan_ui_vram_stable_fraction": float(getattr(args, "vram_lab_stable_fraction", 1.15) or 1.15),
        "hunyuan_ui_vram_stable_budget_gb": float(
            _profile_hotset_budget_gb_cli(getattr(args, "vram_lab_profile", "auto"))
            if getattr(args, "vram_lab_stable_budget_gb", None) is None
            else getattr(args, "vram_lab_stable_budget_gb")
        ),
        "hunyuan_ui_vram_profile": _resolve_vram_lab_profile_cli(getattr(args, "vram_lab_profile", "auto")),
        "per_process_cap_status": "pending",
        "pipe_to_cuda_avoided": "pending",
        "cpu_offload_path_used": "pending",
        "cpu_text_encode_forced": "pending",
        "cuda_before_pipeline_load": _cuda_mem_line(torch),
        "notes": [
            "Hunyuan remains the video runner.",
            "VRAM Lab remains the memory manager / planner.",
            "This integration gate runs before the Diffusers pipeline is loaded.",
            "VRAM Lab 0.5 adds forward-time hooks; hook counters must be above zero to prove real execution control.",
        ],
    }

    try:
        import vram_forward_hooks as vlab  # type: ignore

        profile = "24GB_SAFE" if mode == "safe" else "24GB_BALANCED"
        reporter = _VRAMLabReportBuffer()

        # Keep this activation step compatible with both VRAM Lab generations:
        # - older helper files exposed build_budget/SafetensorsCPUWeightStore/HotColdPlanner
        # - newer shared vram_forward_hooks.py only exposes the runtime hook/profile API
        # Hunyuan should not require changes to the shared helper just to build its report.
        has_legacy_planner_api = all(
            hasattr(vlab, name)
            for name in (
                "build_budget",
                "SafetensorsCPUWeightStore",
                "GenericSafetensorsFolderAdapter",
                "HotColdPlanner",
            )
        )

        gpu_name = "unknown GPU"
        total = 0
        budget_profile = profile
        budget_profile_note = "Hunyuan-local compatibility budget"
        budget_bytes = 0
        workspace_reserve_bytes = 0
        safetensors_files: list[str] = []
        safetensors_count = 0
        estimated_size_bytes = 0
        full_cuda_load_refused = True
        group_summary = "components=0, tensors=n/a, original_groups=0, packed_groups=0, largest_group=0 B"
        adapter_status = "PASS: Hunyuan-local safetensors metadata scan"

        if has_legacy_planner_api:
            budget_args = SimpleNamespace(profile=profile, budget_gb=None, workspace_reserve_gb=None)
            budget = vlab.build_budget(budget_args, torch, reporter)

            # Scan and plan the real Hunyuan safetensors store. This is metadata/header
            # planning only, not another streaming stress test and not video generation.
            store = vlab.SafetensorsCPUWeightStore(torch, Path(local_dir))
            tensors = store.list_tensors()
            adapter = vlab.GenericSafetensorsFolderAdapter(
                "hunyuan_720_i2v_distilled" if str(model_key) == "720p_i2v_distilled" else str(model_key)
            )
            adapter_info = adapter.build_info(store, tensors)
            plan = vlab.HotColdPlanner(budget, reporter).build(tensors)

            gpu_name = str(getattr(budget, "gpu_name", gpu_name))
            total = int(getattr(budget, "total_vram_bytes", 0) or 0)
            budget_profile = str(getattr(budget, "profile", profile))
            budget_profile_note = str(getattr(budget, "profile_note", "legacy planner API"))
            budget_bytes = int(getattr(budget, "budget_bytes", 0) or 0)
            workspace_reserve_bytes = int(getattr(budget, "workspace_reserve_bytes", 0) or 0)
            safetensors_count = int(getattr(adapter_info, "safetensors_file_count", 0) or 0)
            safetensors_files = list(getattr(adapter_info, "safetensors_files", []) or [])
            estimated_size_bytes = int(getattr(plan, "full_logical_bytes", 0) or 0)
            full_cuda_load_refused = bool(getattr(plan, "full_cuda_load_refused", True))
            group_summary = (
                f"components={getattr(adapter_info, 'component_count', 0)}, tensors={len(tensors)}, "
                f"original_groups={getattr(plan, 'original_cold_group_count', 0)}, "
                f"packed_groups={getattr(plan, 'packed_cold_group_count', 0)}, "
                f"largest_group={_fmt_cuda_bytes(int(getattr(plan, 'largest_group_bytes', 0) or 0))}"
            )
            adapter_status = "PASS: model adapter contract built from real safetensors folder"
        else:
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_name = str(getattr(props, "name", gpu_name))
                total = int(getattr(props, "total_memory", 0) or 0)
            except Exception:
                try:
                    _free, _total = torch.cuda.mem_get_info()
                    total = int(_total or 0)
                except Exception:
                    total = 0

            try:
                policy = vlab.apply_vram_lab_profile_defaults({"mode": mode}, mode)
            except Exception:
                policy = {"mode": mode}

            budget_profile = f"{profile}_COMPAT"
            budget_profile_note = str(policy.get("profile_note") or "new vram_forward_hooks runtime API; legacy planner API not present")
            budget_bytes = int(
                policy.get("hot_block_budget_bytes")
                or policy.get("hot_blocks_budget_bytes")
                or policy.get("resident_budget_bytes")
                or policy.get("max_cuda_resident_bytes")
                or 0
            )
            workspace_reserve_bytes = int(
                policy.get("emergency_driver_free_floor_bytes")
                or policy.get("driver_free_floor_bytes")
                or 0
            )
            if budget_bytes <= 0 and total > 0:
                budget_bytes, workspace_reserve_bytes, _compat_note = _generation_safe_vram_policy(mode, total)
                budget_profile_note = f"{budget_profile_note}; {_compat_note}"

            file_rows: list[tuple[str, int]] = []
            try:
                for fp in sorted(Path(local_dir).rglob("*.safetensors")):
                    try:
                        rel = str(fp.relative_to(local_dir)).replace("\\", "/")
                    except Exception:
                        rel = str(fp.name)
                    try:
                        size = int(fp.stat().st_size)
                    except Exception:
                        size = 0
                    file_rows.append((rel, size))
            except Exception as scan_exc:
                ctx.setdefault("notes", []).append(f"Hunyuan-local safetensors scan failed: {type(scan_exc).__name__}: {scan_exc}")

            safetensors_files = [name for name, _size in file_rows]
            safetensors_count = len(file_rows)
            estimated_size_bytes = sum(size for _name, size in file_rows)
            largest_size = max((size for _name, size in file_rows), default=0)
            group_summary = (
                f"components=n/a, tensors=n/a, original_groups=n/a, packed_groups=n/a, "
                f"safetensors_files={safetensors_count}, largest_file={_fmt_cuda_bytes(largest_size)}"
            )
            adapter_status = "PASS: Hunyuan-local compatibility scan; shared VRAM hook helper left untouched"
            ctx.setdefault("notes", []).append("Legacy VRAM Lab planner API not found; using Hunyuan-local compatibility scan before attaching the newer forward hooks.")

        gen_budget, gen_reserve, gen_note = _generation_safe_vram_policy(mode, total)
        if total > 0:
            fraction = max(0.05, min(1.0, int(gen_budget) / float(total)))
            hard_cap_requested = str(os.environ.get("VRAM_LAB_HARD_CAP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
            if hard_cap_requested:
                try:
                    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                    ctx["per_process_cap_status"] = f"ACTIVE: fraction={fraction:.4f} from VRAM_LAB_HARD_CAP"
                    ctx["notes"].append(f"Applied generation-safe torch CUDA per-process memory fraction: {fraction:.4f}")
                except Exception as e:
                    ctx["per_process_cap_status"] = f"FAILED: {e}"
                    ctx["notes"].append(f"Could not set torch CUDA memory fraction: {e}")
            else:
                # 0.5 reports showed the artificial 22 GB fraction killed real
                # generation while normal Hunyuan can complete. Keep the VRAM Lab
                # budget as telemetry/safety policy, but do not hard-cap PyTorch
                # during real hooked execution unless explicitly requested.
                ctx["per_process_cap_status"] = f"DISABLED: advisory budget only (would have been fraction={fraction:.4f}); set VRAM_LAB_HARD_CAP=1 to enforce"
                ctx["notes"].append("Per-process hard cap disabled for real hooked generation; VRAM Lab budget is advisory telemetry.")

        ctx.update(
            {
                "gpu_name": str(gpu_name),
                "total_vram": _fmt_cuda_bytes(total),
                "profile": f"{budget_profile} ({budget_profile_note})",
                "budget": _fmt_cuda_bytes(budget_bytes),
                "workspace_reserve": _fmt_cuda_bytes(workspace_reserve_bytes),
                "generation_budget": _fmt_cuda_bytes(gen_budget),
                "generation_reserve": _fmt_cuda_bytes(gen_reserve),
                "generation_policy_note": gen_note,
                "allocator_config_status": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set"),
                "safetensors_count": str(safetensors_count),
                "safetensors_count_int": int(safetensors_count),
                "safetensors_files": list(safetensors_files),
                "estimated_size": _fmt_cuda_bytes(estimated_size_bytes),
                "full_refused": "YES" if full_cuda_load_refused else "NO",
                "adapter_status": adapter_status,
                "activation_status": "PASS: VRAM Lab budget/profile selected before Diffusers pipeline load",
                "group_summary": group_summary,
                "next_step": "Hunyuan will continue generation with the available forward-hook runtime. If it succeeds, tune Hunyuan-specific hook budgets from the report/logs.",
            }
        )
        _VRAM_LAB_CONTEXT = ctx
        _write_vram_lab_integration_report(ctx, result="WARN")
        print(f"[vram-lab] mode={mode} profile={profile} activated before pipeline load", flush=True)
        print(f"[vram-lab] report: {ctx['report_path']}", flush=True)
        return ctx
    except Exception as e:
        ctx["adapter_status"] = "FAIL"
        ctx["activation_status"] = "FAIL"
        ctx["generation_status"] = "not started"
        _VRAM_LAB_CONTEXT = ctx
        _write_vram_lab_integration_report(ctx, result="FAIL", error=f"{type(e).__name__}: {e}")
        raise RuntimeError(f"VRAM Lab activation failed: {e}") from e


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

    # Optional VRAM Lab integration gate. This happens before the Diffusers
    # pipeline load so full-load risk can be detected/avoided early.
    vram_lab_ctx = _activate_vram_lab_memory_control(root, torch, args, str(model_key), str(model_id), local_dir)
    vram_lab_mode = _normalize_vram_lab_mode(getattr(args, "vram_lab", "off"))
    if isinstance(vram_lab_ctx, dict):
        vram_lab_ctx["cuda_before_pipeline_load"] = _cuda_mem_line(torch)
        # VRAM Lab mode is allowed to choose lower-memory Hunyuan options, while
        # Hunyuan remains the runner and VRAM Lab remains the planner/manager.
        args.offload = True
        # Reference-trace correction: do not force prompt-embed injection for
        # Hunyuan VRAM Lab. This Diffusers build has unstable prompt/mask names
        # and the failed embed path was silently falling back into the exact
        # shared-memory crawl we are trying to prevent. Use Diffusers' normal
        # prompt path under enable_model_cpu_offload(), while VRAM Lab controls
        # component residency and transformer block streaming.
        if getattr(args, "cpu_text_encode", False):
            args.cpu_text_encode = False
            vram_lab_ctx["cpu_text_encode_forced"] = "NO: disabled for Hunyuan reference profile; prompt embeds path is unstable in this Diffusers build"
        else:
            vram_lab_ctx["cpu_text_encode_forced"] = "NO: Hunyuan reference profile uses normal prompt path with model CPU offload"
        vram_lab_ctx["cpu_text_embeddings_kept_on_cpu"] = "n/a: prompt-embed injection disabled"
        vram_lab_ctx["cpu_text_encode_call_embed_keys"] = "n/a: prompt-embed injection disabled"
        _vram_lab_force_memory_savers(args, vram_lab_ctx)
        vram_lab_ctx["pipe_to_cuda_avoided"] = "YES: VRAM Lab mode uses Diffusers CPU offload path, not pipe.to(\"cuda\")"
        vram_lab_ctx["cpu_offload_path_used"] = "pending"
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

    # Prefer local_dir if user downloaded; otherwise let HF handle it.
    PipeCls = HunyuanVideo15ImageToVideoPipeline if model_is_i2v else HunyuanVideo15Pipeline
    pipe = _from_pretrained(PipeCls, str(local_dir), dtype)
    if isinstance(vram_lab_ctx, dict):
        vram_lab_ctx["cuda_after_pipeline_load"] = _cuda_mem_line(torch)
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

    # I2V resolution override:
    # Diffusers' HunyuanVideo15ImageToVideoPipeline does not accept height/width kwargs.
    # Instead it uses pipe.target_size to decide the internal working resolution.
    if model_is_i2v and ((args.height or args.width) or (getattr(args, 'target_size', 0) and int(getattr(args, 'target_size', 0) or 0) > 0)):
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

    # VRAM Lab generation policy may clamp I2V target_size to prevent the first
    # real generation from dying in the denoise/VAE spike. This is reported.
    if isinstance(vram_lab_ctx, dict) and model_is_i2v:
        try:
            limit = int(_vram_lab_target_size_limit(_normalize_vram_lab_mode(getattr(args, "vram_lab", "off"))) or 0)
            cur_ts = getattr(pipe, "target_size", None)
            if cur_ts is not None and limit > 0 and int(cur_ts) > limit:
                old_ts = int(cur_ts)
                try:
                    vae_sf = int(getattr(pipe, "vae_scale_factor_spatial", 16) or 16)
                except Exception:
                    vae_sf = 16
                limit_aligned = max(vae_sf, (limit // vae_sf) * vae_sf)
                pipe.target_size = int(limit_aligned)
                msg = f"VRAM Lab {_normalize_vram_lab_mode(getattr(args, 'vram_lab', 'off'))} clamped I2V target_size {old_ts} -> {int(pipe.target_size)}"
                print(f"[vram-lab] {msg}", flush=True)
                vram_lab_ctx["generation_setting_adjustment"] = msg
            else:
                vram_lab_ctx.setdefault("generation_setting_adjustment", "none; VRAM Lab 0.5 does not lower target_size/frames for hook proof")
        except Exception as e:
            vram_lab_ctx["generation_setting_adjustment"] = f"target_size guard skipped/failed: {e}"
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

    vram_lab_mode = _normalize_vram_lab_mode(getattr(args, "vram_lab", "off"))
    if args.offload or _vram_lab_enabled(vram_lab_mode):
        if _vram_lab_enabled(vram_lab_mode):
            print("[vram-lab] using Diffusers CPU/offload path; full pipe.to(\"cuda\") is avoided", flush=True)
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["pipe_to_cuda_avoided"] = "YES"
            # Hunyuan + VRAM Lab must use Diffusers model CPU offload.
            # Do NOT use enable_sequential_cpu_offload() here: that creates
            # Accelerate meta-device placeholders and can crash before the
            # transformer hooks ever run with:
            #   Tensor on device meta is not on the expected device cuda:0!
            # Older working Hunyuan VRAM Lab runs used enable_model_cpu_offload(),
            # so keep safe/balanced/aggressive on that same path.
            pipe.enable_model_cpu_offload()
            used_offload = "YES: pipe.enable_model_cpu_offload()"
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["cpu_offload_path_used"] = used_offload
                vram_lab_ctx["offload_mode_used"] = used_offload
        else:
            pipe.enable_model_cpu_offload()
        if isinstance(vram_lab_ctx, dict):
            vram_lab_ctx["cuda_after_offload_hook"] = _cuda_mem_line(torch)
            _vram_lab_cleanup_cuda(vram_lab_ctx, torch, "after offload hook")
            _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")
    else:
        pipe = pipe.to("cuda")

    # VRAM Lab 0.5: attach real forward-time hooks after the pipeline/offload
    # objects exist, but before Hunyuan generation starts. This does not make
    # VRAM Lab a video runner; Hunyuan still owns the call, VRAM Lab only wraps
    # the live transformer block forwards.
    if isinstance(vram_lab_ctx, dict):
        _attach_vram_lab_forward_hooks(pipe, torch, vram_lab_ctx)
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

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
                if isinstance(vram_lab_ctx, dict):
                    vram_lab_ctx["attention_slicing_applied"] = "YES"
            elif isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["attention_slicing_applied"] = "NO: pipeline has no enable_attention_slicing"
        except Exception as e:
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["attention_slicing_applied"] = f"FAILED: {e}"

    if getattr(args, "vae_slicing", False):
        try:
            pipe.vae.enable_slicing()
            print("[gen] VAE slicing: enabled", flush=True)
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["vae_slicing_applied"] = "YES"
        except Exception as e:
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["vae_slicing_applied"] = f"FAILED: {e}"

    if args.tiling:
        try:
            pipe.vae.enable_tiling()
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["vae_tiling_applied"] = "YES"
        except Exception as e:
            if isinstance(vram_lab_ctx, dict):
                vram_lab_ctx["vae_tiling_applied"] = f"FAILED: {e}"
    if isinstance(vram_lab_ctx, dict):
        _vram_lab_cleanup_cuda(vram_lab_ctx, torch, "after memory savers")
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

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
    prompt_embeds_2 = None
    negative_prompt_embeds = None
    negative_prompt_embeds_2 = None
    prompt_embeds_mask = None
    prompt_embeds_2_mask = None
    prompt_embeds_mask_2 = None
    negative_prompt_embeds_mask = None
    negative_prompt_embeds_2_mask = None
    negative_prompt_embeds_mask_2 = None
    if getattr(args, "cpu_text_encode", False):
        try:
            enc = _encode_prompt_embeds_cpu(pipe, torch, prompt, negative=negative)
            required_embed_ok = bool(enc) and (
                (not _supports_kw(pipe.__call__, "prompt_embeds_2") or enc.get("prompt_embeds_2") is not None)
                and (not _supports_kw(pipe.__call__, "prompt_embeds") or enc.get("prompt_embeds") is not None or _supports_kw(pipe.__call__, "prompt_embeds_2"))
            )
            if required_embed_ok:
                prompt_embeds = enc.get("prompt_embeds")
                prompt_embeds_2 = enc.get("prompt_embeds_2")
                negative_prompt_embeds = enc.get("negative_prompt_embeds")
                negative_prompt_embeds_2 = enc.get("negative_prompt_embeds_2")
                prompt_embeds_mask = enc.get("prompt_embeds_mask")
                prompt_embeds_2_mask = enc.get("prompt_embeds_2_mask") or enc.get("prompt_embeds_mask_2")
                prompt_embeds_mask_2 = enc.get("prompt_embeds_mask_2") or enc.get("prompt_embeds_2_mask")
                negative_prompt_embeds_mask = enc.get("negative_prompt_embeds_mask")
                negative_prompt_embeds_2_mask = enc.get("negative_prompt_embeds_2_mask") or enc.get("negative_prompt_embeds_mask_2")
                negative_prompt_embeds_mask_2 = enc.get("negative_prompt_embeds_mask_2") or enc.get("negative_prompt_embeds_2_mask")

                # In normal/off mode, preserve the old behavior: move prompt tensors
                # to CUDA before the pipeline call. In VRAM Lab mode, keep them on
                # CPU and push the text encoders back to CPU immediately after
                # encoding; otherwise the next large frame-count call starts already
                # at the VRAM cliff before transformer hooks can do anything.
                if isinstance(vram_lab_ctx, dict):
                    vram_lab_ctx["cpu_text_embeddings_kept_on_cpu"] = "YES"
                    moved_txt = []
                    for _name in ("text_encoder", "text_encoder_2"):
                        _comp = getattr(pipe, _name, None)
                        if _comp is None:
                            continue
                        try:
                            _before = _vram_lab_component_device_summary(_comp, limit=32)
                            _comp.to("cpu")
                            moved_txt.append(f"{_name} ({_before} -> cpu)")
                        except Exception as _e:
                            moved_txt.append(f"{_name}: skip {type(_e).__name__}: {_e}")
                    vram_lab_ctx["cpu_text_encode_components_to_cpu"] = "; ".join(moved_txt) if moved_txt else "none"
                    _vram_lab_cleanup_cuda(vram_lab_ctx, torch, "after CPU prompt/text encode + text encoders to CPU")
                else:
                    try:
                        if prompt_embeds is not None:
                            prompt_embeds = prompt_embeds.to("cuda")
                    except Exception:
                        pass
                    try:
                        if prompt_embeds_2 is not None:
                            prompt_embeds_2 = prompt_embeds_2.to("cuda")
                    except Exception:
                        pass
                    try:
                        if negative_prompt_embeds is not None:
                            negative_prompt_embeds = negative_prompt_embeds.to("cuda")
                    except Exception:
                        pass
                    try:
                        if negative_prompt_embeds_2 is not None:
                            negative_prompt_embeds_2 = negative_prompt_embeds_2.to("cuda")
                    except Exception:
                        pass
                    try:
                        if prompt_embeds_mask is not None:
                            prompt_embeds_mask = prompt_embeds_mask.to("cuda")
                    except Exception:
                        pass
                    try:
                        if prompt_embeds_2_mask is not None:
                            prompt_embeds_2_mask = prompt_embeds_2_mask.to("cuda")
                    except Exception:
                        pass
                    try:
                        if prompt_embeds_mask_2 is not None:
                            prompt_embeds_mask_2 = prompt_embeds_mask_2.to("cuda")
                    except Exception:
                        pass
                    try:
                        if negative_prompt_embeds_mask is not None:
                            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to("cuda")
                    except Exception:
                        pass
                    try:
                        if negative_prompt_embeds_2_mask is not None:
                            negative_prompt_embeds_2_mask = negative_prompt_embeds_2_mask.to("cuda")
                    except Exception:
                        pass
                    try:
                        if negative_prompt_embeds_mask_2 is not None:
                            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to("cuda")
                    except Exception:
                        pass

                used_names = []
                for _k, _v in (enc or {}).items():
                    if _v is not None and (_k.startswith("prompt_embeds") or _k.startswith("negative_prompt_embeds")):
                        used_names.append(_k)
                print(f"[gen] cpu-text-encode: enabled ({', '.join(used_names) or 'embeds'})", flush=True)
            else:
                print("[gen] cpu-text-encode: requested but required embed names were not available; falling back to normal prompt encoding", flush=True)
                prompt_embeds = None
                prompt_embeds_2 = None
                negative_prompt_embeds = None
                negative_prompt_embeds_2 = None
                prompt_embeds_mask = None
                prompt_embeds_2_mask = None
                prompt_embeds_mask_2 = None
                negative_prompt_embeds_mask = None
                negative_prompt_embeds_2_mask = None
                negative_prompt_embeds_mask_2 = None
        except Exception:
            print("[gen] cpu-text-encode: failed; falling back to normal prompt encoding", flush=True)
            prompt_embeds = None
            prompt_embeds_2 = None
            negative_prompt_embeds = None
            negative_prompt_embeds_2 = None
            prompt_embeds_mask = None
            prompt_embeds_2_mask = None
            negative_prompt_embeds_mask = None
            negative_prompt_embeds_2_mask = None

    gen = None

    if args.seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)

    
    # Defaults follow the Diffusers docs example, but HunyuanVideo 1.5
    # Diffusers builds may require prompt_embeds_2 rather than prompt_embeds.
    call_accepts_prompt_embeds = _supports_kw(pipe.__call__, "prompt_embeds")
    call_accepts_prompt_embeds_2 = _supports_kw(pipe.__call__, "prompt_embeds_2")
    use_embeds = False
    if call_accepts_prompt_embeds_2:
        use_embeds = prompt_embeds_2 is not None
    elif call_accepts_prompt_embeds:
        use_embeds = prompt_embeds is not None

    if use_embeds:
        kwargs = dict(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            generator=gen,
        )
        if prompt_embeds is not None and call_accepts_prompt_embeds:
            kwargs["prompt_embeds"] = prompt_embeds
        if prompt_embeds_2 is not None and call_accepts_prompt_embeds_2:
            kwargs["prompt_embeds_2"] = prompt_embeds_2
        if prompt_embeds_mask is not None and _supports_kw(pipe.__call__, "prompt_embeds_mask"):
            kwargs["prompt_embeds_mask"] = prompt_embeds_mask
        if prompt_embeds_2_mask is not None and _supports_kw(pipe.__call__, "prompt_embeds_2_mask"):
            kwargs["prompt_embeds_2_mask"] = prompt_embeds_2_mask
        if prompt_embeds_mask_2 is not None and _supports_kw(pipe.__call__, "prompt_embeds_mask_2"):
            kwargs["prompt_embeds_mask_2"] = prompt_embeds_mask_2
        elif prompt_embeds_2_mask is not None and _supports_kw(pipe.__call__, "prompt_embeds_mask_2"):
            kwargs["prompt_embeds_mask_2"] = prompt_embeds_2_mask
        if negative_prompt_embeds is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds"):
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        if negative_prompt_embeds_2 is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_2"):
            kwargs["negative_prompt_embeds_2"] = negative_prompt_embeds_2
        if negative_prompt_embeds_mask is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask"):
            kwargs["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
        if negative_prompt_embeds_2_mask is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_2_mask"):
            kwargs["negative_prompt_embeds_2_mask"] = negative_prompt_embeds_2_mask
        if negative_prompt_embeds_mask_2 is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask_2"):
            kwargs["negative_prompt_embeds_mask_2"] = negative_prompt_embeds_mask_2
        elif negative_prompt_embeds_2_mask is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds_mask_2"):
            kwargs["negative_prompt_embeds_mask_2"] = negative_prompt_embeds_2_mask
        if isinstance(vram_lab_ctx, dict):
            vram_lab_ctx["cpu_text_encode_call_embed_keys"] = ", ".join(sorted(k for k in kwargs.keys() if "embed" in k)) or "none"
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

    if isinstance(vram_lab_ctx, dict) and _supports_kw(pipe.__call__, "callback_on_step_end"):
        prev_cb = kwargs.get("callback_on_step_end")
        kwargs["callback_on_step_end"] = _hunyuan_vram_make_step_end_callback(pipe, vram_lab_ctx, torch, prev_cb)
        if _supports_kw(pipe.__call__, "callback_on_step_end_tensor_inputs"):
            kwargs.setdefault("callback_on_step_end_tensor_inputs", [])
        vram_lab_ctx["hunyuan_step_telemetry_callback"] = "YES: callback_on_step_end"
    elif isinstance(vram_lab_ctx, dict):
        vram_lab_ctx["hunyuan_step_telemetry_callback"] = "NO: pipeline has no callback_on_step_end kwarg"


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
    if isinstance(vram_lab_ctx, dict):
        _vram_lab_stage(vram_lab_ctx, "cuda_before_generation_call", torch, "before pipeline __call__")
        _vram_lab_note_low_driver_free(vram_lab_ctx, torch, "before generation call")
        _vram_lab_cleanup_cuda(vram_lab_ctx, torch, "before generation call")
        # Reference-profile guard: if the job is already below the safe
        # workspace reserve before denoise begins, do not let Windows crawl
        # through shared GPU memory for minutes. This is a real VRAM Lab FAIL,
        # not a reason to fall back to normal CUDA behavior.
        try:
            _free = _vram_lab_driver_free_bytes(torch)
            _floor_gb = float(getattr(args, "vram_lab_min_free_gb", 0.5) or 0.5)
            _floor = int(_floor_gb * (1024 ** 3))
            vram_lab_ctx["hunyuan_reference_workspace_floor"] = _fmt_cuda_bytes(_floor)
            if _free is not None and int(_free) < _floor:
                vram_lab_ctx["failure_stage"] = "before denoise: workspace reserve already below Hunyuan reference safe floor"
                vram_lab_ctx["next_step"] = "Do not continue into shared-memory crawl. Reduce resident components or fix pre-denoise ownership before running larger frame counts."
                _write_vram_lab_integration_report(vram_lab_ctx, result="FAIL", error=f"VRAM Lab reference guard: driver_free {_fmt_cuda_bytes(int(_free))} below floor {_fmt_cuda_bytes(_floor)} before denoise")
                raise RuntimeError(f"VRAM Lab reference guard: driver_free {_fmt_cuda_bytes(int(_free))} below safe floor {_fmt_cuda_bytes(_floor)} before denoise")
        except RuntimeError:
            raise
        except Exception as _e:
            vram_lab_ctx.setdefault("vram_shared_spill_risk_notes", []).append(f"reference pre-denoise guard warning: {type(_e).__name__}: {_e}")
        _hunyuan_vram_live_snapshot(vram_lab_ctx, torch, "before pipeline call", print_line=True)
        _hunyuan_vram_install_decode_guard(pipe, torch, vram_lab_ctx, args)
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")
    try:
        video = pipe(**kwargs).frames[0]
    except Exception as e:
        if isinstance(vram_lab_ctx, dict):
            _vram_lab_refresh_forward_hook_status(vram_lab_ctx)
            try:
                pre = int(vram_lab_ctx.get("vram_pre_forward_calls_int", 0) or 0)
                post = int(vram_lab_ctx.get("vram_post_forward_calls_int", 0) or 0)
            except Exception:
                pre = post = 0
            if vram_lab_ctx.get("vram_diag_exception_block"):
                vram_lab_ctx["failure_stage"] = f"inside hooked block: {vram_lab_ctx.get('vram_diag_exception_block')}"
            elif pre <= 0:
                vram_lab_ctx["failure_stage"] = "before first transformer hook"
            elif post < pre:
                vram_lab_ctx["failure_stage"] = "inside hooked transformer execution"
            else:
                vram_lab_ctx["failure_stage"] = "after hooks / pipeline post-processing"
            vram_lab_ctx["generation_exception_type"] = type(e).__name__
            vram_lab_ctx["generation_exception_message"] = str(e)[:1200]
            _vram_lab_stage(vram_lab_ctx, "cuda_after_generation_exception", torch, f"pipeline __call__ raised {type(e).__name__}")
            _write_vram_lab_integration_report(vram_lab_ctx, result="WARN", error=f"{type(e).__name__}: {e}")
        raise
    finalize_guard = None
    if isinstance(vram_lab_ctx, dict):
        _vram_lab_stage(vram_lab_ctx, "cuda_after_generation_attempt", torch, "after pipeline __call__ returned frames")
        # The heavy denoise/generation path is done. From here on, protect the
        # final save/re-encode stage by releasing VRAM Lab's runtime references and
        # moving components off CUDA before exporting frames.
        finalize_guard = _vram_lab_make_finalize_guard(vram_lab_ctx, torch, "hunyuan_finalize")
        if finalize_guard is not None:
            try:
                finalize_guard.stage("cuda_after_denoise_generation", "pipeline returned frames")
                rt = vram_lab_ctx.get("_vram_runtime")
                finalize_guard.detach_runtimes([rt] if rt is not None else [], clear_context_key="_vram_runtime")
                comps = [
                    getattr(pipe, "transformer", None),
                    getattr(pipe, "text_encoder", None),
                    getattr(pipe, "text_encoder_2", None),
                    getattr(pipe, "image_encoder", None),
                    getattr(pipe, "vae", None),
                ]
                try:
                    if hasattr(pipe, "maybe_free_model_hooks"):
                        pipe.maybe_free_model_hooks()
                except Exception:
                    pass
                finalize_guard.move_components_to_cpu(comps, label="before_save_reencode_cleanup")
                try:
                    video = finalize_guard.tensor_to_cpu(video, label="generated_video_to_cpu")
                    vram_lab_ctx["finalize_output_to_cpu"] = "YES"
                except Exception as e:
                    vram_lab_ctx["finalize_output_to_cpu"] = f"FAILED: {e}"
                # Release the pipeline reference before export/re-encode. The frame
                # list/tensor should be independent at this point.
                try:
                    del pipe
                except Exception:
                    pass
                gc.collect()
                finalize_guard.cleanup_cuda("before_save_reencode")
                finalize_guard.stage("cuda_before_save_reencode", "before export_to_video / ffmpeg re-encode")
            except Exception as e:
                vram_lab_ctx.setdefault("finalize_guard_notes", []).append(f"hunyuan finalize pre-save guard failed: {type(e).__name__}: {e}")
        _write_vram_lab_integration_report(vram_lab_ctx, result="WARN")

    # Diffusers' export_to_video uses a very conservative default encode, which can look low-bitrate.
    # We optionally re-encode with ffmpeg for a predictable bitrate.
    br = int(getattr(args, "bitrate_kbps", 0) or 0)
    _save_start = time.perf_counter()
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
    _save_secs = time.perf_counter() - _save_start

    print(f"[gen] saved: {out_path}", flush=True)
    if isinstance(vram_lab_ctx, dict):
        vram_lab_ctx["output_path"] = str(out_path)
        vram_lab_ctx["finalize_save_reencode_duration"] = f"{_save_secs:.3f}s"
        if finalize_guard is not None:
            try:
                finalize_guard.stage("cuda_after_save_reencode", "after export_to_video / ffmpeg re-encode")
                finalize_guard.finish()
            except Exception as e:
                vram_lab_ctx.setdefault("finalize_guard_notes", []).append(f"hunyuan finalize post-save guard failed: {type(e).__name__}: {e}")
        else:
            _vram_lab_cleanup_cuda(vram_lab_ctx, torch, "after VAE decode/export")
    _finalize_vram_lab_integration("PASS", "completed")

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
    g.add_argument("--vram-lab", default="off", choices=["off", "on", "safe", "balanced", "aggressive"], help="Use VRAM Lab memory control: off|on (on maps to safe/default internally; legacy safe/balanced/aggressive also map to safe)")
    g.add_argument("--vram-lab-profile", default="auto", choices=["auto", "16", "24"], help="Auto chooses 16 GB profile below 23.01 GB VRAM and 24 GB profile above 23 GB VRAM. If step-hot/stable-budget are omitted, profile 16 uses 1.0 GB and profile 24 uses 9.0 GB.")
    g.add_argument("--vram-lab-text-hot-gb", type=float, default=9.0, help="Hunyuan-local startup/text/image stage guard budget in GB")
    g.add_argument("--vram-lab-step-hot-gb", type=float, default=None, help="Hunyuan-local denoise transformer hot/resident budget in GB. If omitted, profile 16 uses 1.0 GB and profile 24 uses 9.0 GB.")
    g.add_argument("--vram-lab-after-hot-gb", type=float, default=0.0, help="Hunyuan-local after-steps/finalize guard budget in GB")
    g.add_argument("--vram-lab-decode-min-free-gb", type=float, default=0.5, help="Hunyuan-local minimum driver-free VRAM target to try to preserve before each hidden VAE decode/postprocess call")
    g.add_argument("--vram-lab-decode-cleanup-passes", type=int, default=2, help="Extra cleanup / empty-cache passes to run around the hidden VAE decode phase")
    g.add_argument("--vram-lab-decode-chunk-frames", type=int, default=8, help="Experimental hidden VAE decode chunk size along time/frame dimension. 0=disabled/default; try 8/16/24/32 if decode spills into shared memory.")
    g.add_argument("--vram-lab-decode-chunk-overlap", type=int, default=4, help="Latent-time overlap for chunked VAE decode. 0=hard chunks; try 2 to reduce chunk-boundary glitches.")
    g.add_argument("--vram-lab-decode-unload-blocks", action=argparse.BooleanOptionalAction, default=True, help="Force any transformer blocks still on CUDA back to CPU before each hidden VAE decode call")
    g.add_argument("--vram-lab-decode-output-cpu", action=argparse.BooleanOptionalAction, default=True, help="Move decoded tensors/frames to CPU immediately after each hidden VAE decode call when possible")
    g.add_argument("--vram-lab-min-free-gb", type=float, default=0.5, help="Hunyuan-local minimum driver-free VRAM target in GB")
    g.add_argument("--vram-lab-residency", default="planned_hotset", choices=["rolling", "planned_hotset"], help="Hunyuan-local denoise residency strategy")
    g.add_argument("--vram-lab-stable-fraction", type=float, default=1.15, help="Hunyuan-local stable hotset fraction for planned_hotset mode")
    g.add_argument("--vram-lab-stable-budget-gb", type=float, default=None, help="Hunyuan-local stable hotset budget in GB for planned_hotset mode. If omitted, profile 16 uses 1.0 GB and profile 24 uses 9.0 GB.")
    g.add_argument("--vram-lab-step-profiler", action="store_true", help="Profile per-block time/VRAM for the first denoise step")
    g.add_argument("--vram-lab-step-profiler-steps", type=int, default=1, help="How many early denoise steps to profile, normally 1")
    g.add_argument("--vram-lab-step-cleanup-every", type=int, default=4, help="Denoise cleanup/cache frequency in blocks. 1=current safe every block; 2/4/8=smoother but riskier; 0=runtime/emergency default only.")
    g.add_argument("--vram-lab-step-empty-cache", action=argparse.BooleanOptionalAction, default=True, help="Empty CUDA cache after block unload during denoise")
    g.add_argument("--vram-lab-step-release-after-forward", action=argparse.BooleanOptionalAction, default=True, help="Release block after forward during denoise")
    g.add_argument("--vram-lab-step-unload-before-load", action=argparse.BooleanOptionalAction, default=True, help="Unload other blocks before loading the next denoise block")
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
        args.vram_lab = _normalize_vram_lab_mode(getattr(args, "vram_lab", "off"))

    try:
        if args.cmd == "generate":
            setattr(args, "_vram_lab_allocator_env_status", _ensure_vram_lab_allocator_env(getattr(args, "vram_lab", "off")))
            if _vram_lab_enabled(getattr(args, "vram_lab", "off")):
                print(f"[vram-lab] allocator env: {getattr(args, '_vram_lab_allocator_env_status', 'n/a')}", flush=True)
        args.fn(args)
        return 0
    except Exception as e:
        try:
            _finalize_vram_lab_integration("FAIL", "failed", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())