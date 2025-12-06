#!/usr/bin/env python
"""
Z-Image Turbo txt2img helper for FrameVision.

This version is fully OFFLINE and uses the local model folder:
    <root>/models/Z-Image-Turbo

Requirements (handled by zimage_install.bat):
  * models/Z-Image-Turbo/ directory must contain a full snapshot of
    Tongyi-MAI/Z-Image-Turbo, with the transformer weights replaced by
    an FP8 checkpoint at:
        transformer/diffusion_pytorch_model.safetensors

It:
  * Chooses CUDA + bfloat16/float16 when available, else CPU + float32.
  * Loads ZImagePipeline from the LOCAL directory only (local_files_only=True).
  * Uses Diffusers' enable_model_cpu_offload() on CUDA for lower VRAM.
  * Saves one or more images to the requested output directory.
  * Prints a single JSON line to stdout: { "files": [paths], "model": "Z-Image-Turbo", "error": ... }
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
import sys


def _unique_path(p: Path) -> Path:
    """Return a unique path by appending _### if needed."""
    try:
        p = Path(p)
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix or ".png"
        i = 1
        while True:
            cand = p.with_name(f"{stem}_{i:03d}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    except Exception:
        return Path(p)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fmt", default="png")
    parser.add_argument("--filename_template", default="zimage_{seed}_{idx:03d}.png")
    # Optional: explicit attention-slicing flag from UI / caller.
    parser.add_argument("--attn-slicing", action="store_true", default=False)
    args = parser.parse_args(argv)

    try:
        import torch  # type: ignore
        from diffusers import ZImagePipeline  # type: ignore
    except Exception as e:
        payload = {"files": [], "error": f"import_failed: {e}"}
        print(json.dumps(payload))
        return 1

    # Device / dtype selection
    try:
        if torch.cuda.is_available():
            dtype = getattr(torch, "bfloat16", torch.float16)
            try:
                if not (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()):
                    dtype = torch.float16
            except Exception:
                dtype = torch.float16
            device = "cuda"
        else:
            dtype = torch.float32
            device = "cpu"
    except Exception:
        dtype = torch.float32
        device = "cpu"


    # Simple VRAM-based heuristic for low-VRAM GPUs: treat GPUs with
    # less than ~16 GiB as low-VRAM and default to attention slicing.
    auto_low_vram = False
    try:
        if device == "cuda" and hasattr(torch, "cuda"):
            props = torch.cuda.get_device_properties(0)
            total_gib = float(getattr(props, "total_memory", 0)) / float(1024 ** 3)
            if total_gib < 16.0:
                auto_low_vram = True
    except Exception:
        auto_low_vram = False

    # Resolve LOCAL model directory: <root>/models/Z-Image-Turbo
    try:
        script_path = Path(__file__).resolve()
        root = script_path.parents[1]  # helpers/ -> project root
        model_dir = root / "models" / "Z-Image-Turbo"
    except Exception:
        model_dir = Path("models/Z-Image-Turbo")

    # Load pipeline from LOCAL directory only, no HF cache/network.
    try:
        if not model_dir.exists():
            raise RuntimeError(f"model_dir_not_found: {model_dir}")

        pipe = ZImagePipeline.from_pretrained(
            str(model_dir),
            torch_dtype=dtype if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

        if device == "cuda":
            # Prefer Diffusers' CPU offload to keep VRAM reasonable.
            used_offload = False
            try:
                pipe.enable_model_cpu_offload()
                used_offload = True
            except Exception:
                used_offload = False

            if not used_offload:
                # Fallback: move entire pipeline to CUDA if offload is unavailable.
                pipe.to(device)

            # Extra memory tweaks (safe no-ops if unsupported)
            try:
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
            except Exception:
                pass

            # Attention slicing: reduce VRAM by splitting attention into smaller
            # chunks. We enable this automatically on low-VRAM GPUs (auto_low_vram)
            # and whenever the caller passes --attn-slicing from the UI.
            try:
                want_attn_slicing = bool(getattr(args, "attn_slicing", False) or auto_low_vram)
            except Exception:
                want_attn_slicing = auto_low_vram
            try:
                if want_attn_slicing and hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing("max")
            except Exception:
                pass

        # Turn off safety checker if present to save a bit of memory.
        try:
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
        except Exception:
            pass

    except Exception as e:
        payload = {
            "files": [],
            "error": f"load_failed: {e}",
            "trace": traceback.format_exc(),
        }
        print(json.dumps(payload))
        return 1

    out_dir = Path(args.outdir).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    fmt = (args.fmt or "png").lower().strip()
    if fmt not in ("png", "jpg", "jpeg", "webp", "bmp"):
        fmt = "png"

    files = []
    base_seed = int(args.seed or 0)
    for i in range(int(args.batch or 1)):
        # Generator
        gen = None
        try:
            if base_seed:
                gen_dev = device if device != "cpu" else "cpu"
                gen = torch.Generator(device=gen_dev).manual_seed(base_seed + i)
        except Exception:
            gen = None

        try:
            kwargs = dict(
                prompt=args.prompt,
                height=int(args.height),
                width=int(args.width),
                num_inference_steps=max(1, int(args.steps or 1)),
                guidance_scale=float(args.guidance or 0.0),
                generator=gen,
            )
            neg = (args.negative or "").strip()
            if neg:
                kwargs["negative_prompt"] = neg
            result = pipe(**kwargs)
            img = result.images[0]
        except Exception as e:
            payload = {
                "files": files,
                "error": f"generate_failed: {e}",
                "trace": traceback.format_exc(),
            }
            print(json.dumps(payload))
            return 1

        # Filename from template
        try:
            fname = args.filename_template.format(
                seed=(base_seed if base_seed else 0) + i,
                idx=i,
                width=int(args.width),
                height=int(args.height),
            )
        except Exception:
            fname = f"zimage_{int(time.time())}_{i:03d}.{fmt}"

        if "." not in fname.split("/")[-1]:
            fname = fname.rstrip(".") + f".{fmt}"

        fpath = _unique_path(out_dir / fname)
        try:
            img.save(str(fpath))
        except Exception as e:
            payload = {
                "files": files,
                "error": f"save_failed: {e}",
                "trace": traceback.format_exc(),
            }
            print(json.dumps(payload))
            return 1

        files.append(str(fpath))

    payload = {"files": files, "model": "Z-Image-Turbo"}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
