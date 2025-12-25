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
import subprocess
import datetime
import json
import time
import traceback
from pathlib import Path
import sys
import inspect


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


def _root_dir() -> "Path":
    # helpers/zimage_cli.py -> <root>/helpers
    return Path(__file__).resolve().parent.parent


def _run_gguf(args) -> dict:
    """
    Run Z-Image Turbo GGUF via stable-diffusion.cpp sd-cli.exe.
    Returns: {"files":[...], ...} or {"files":[], "error": "..."}.
    """
    root = _root_dir()
    gguf_dir = root / "models" / "Z-Image-Turbo GGUF"

    exe = gguf_dir / "sd-cli.exe"
    if not exe.exists():
        exe = gguf_dir / "bin" / "sd-cli.exe"
    if not exe.exists():
        return {"files": [], "error": f"gguf_missing_sdcli: {exe} not found. Re-run the GGUF installer."}

    default_diffusion = gguf_dir / "z_image_turbo-Q5_0.gguf"
    default_vae = gguf_dir / "ae.safetensors"
    llm = gguf_dir / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    diffusion = default_diffusion
    vae = default_vae
    # Transport for GGUF model/VAE comes via existing args.lora/args.lora2 (UI repurposes these in GGUF mode).
    try:
        if getattr(args, "lora", "") and str(getattr(args, "lora")).lower().endswith(".gguf"):
            p = Path(str(getattr(args, "lora")))
            diffusion = (gguf_dir / p).resolve() if not p.is_absolute() else p
    except Exception:
        pass
    try:
        if getattr(args, "lora2", "") and str(getattr(args, "lora2")).lower().endswith((".safetensors",".pt")):
            p = Path(str(getattr(args, "lora2")))
            vae = (gguf_dir / p).resolve() if not p.is_absolute() else p
    except Exception:
        pass
    missing = [p for p in (diffusion, llm, vae) if not p.exists()]
    if missing:
        return {"files": [], "error": "gguf_missing_files: " + ", ".join(str(p) for p in missing)}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Best-effort log file
    log_path = None
    try:
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"zimage_gguf_{ts}_{args.seed}.log"
    except Exception:
        log_path = None

    def _log(line: str) -> None:
        if log_path is None:
            return
        try:
            with open(log_path, "a", encoding="utf-8", errors="ignore") as f:
                f.write(line.rstrip("\\n") + "\\n")
        except Exception:
            pass

    files = []
    batch = max(1, int(getattr(args, "batch", 1)))
    for idx in range(batch):
        seed = int(getattr(args, "seed", 42))
        if seed == 0:
            seed = -1  # sd-cli uses random for <0
        elif idx > 0:
            seed = seed + idx

        # Resolve filename using template
        try:
            fname = str(getattr(args, "filename_template", "z_img_{seed}_{idx:03d}.png")).format(
                seed=(seed if seed >= 0 else int(time.time())),
                idx=idx,
            )
        except Exception:
            fname = f"z_img_{seed}_{idx:03d}.png"
        fpath = outdir / fname

        prompt = str(getattr(args, "prompt", ""))
        neg = str(getattr(args, "negative", "") or "")
        h = int(getattr(args, "height", 1024))
        w = int(getattr(args, "width", 1024))
        steps = int(getattr(args, "steps", 20))
        guidance = float(getattr(args, "guidance", 0.0))

        cmd = [
            str(exe),
            "--diffusion-model", str(diffusion),
            "--vae", str(vae),
            "--llm", str(llm),
            "-p", prompt,
            "-n", neg,
            "-H", str(h),
            "-W", str(w),
            "--steps", str(steps),
            "-s", str(seed),
            "-o", str(fpath),
            "-v",
            "--diffusion-fa",
            "--cfg-scale", "1.0",
            "--guidance", str(guidance),
        ]

        if getattr(args, "attn_slicing", False):
            # Use as "low-vram mode" for gguf backend
            cmd += ["--vae-tiling", "--offload-to-cpu", "--clip-on-cpu", "--vae-on-cpu"]

        _log("CMD: " + " ".join(cmd))
        print("[zimage_cli][gguf] " + " ".join(cmd), flush=True)

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            assert p.stdout is not None
            for line in p.stdout:
                _log(line)
                print(line.rstrip("\\n"), flush=True)
            rc = p.wait()
        except Exception as e:
            _log(f"EXCEPTION: {e}")
            return {"files": [], "error": f"gguf_run_failed: {e}"}

        if rc != 0:
            _log(f"EXIT CODE: {rc}")
            return {"files": [], "error": f"gguf_sdcli_exit_{rc}: see {log_path or 'logs/'}"}

        if not fpath.exists():
            return {"files": [], "error": f"gguf_no_output: {fpath}"}

        files.append(str(fpath))

    return {"files": files, "model": "Z-Image-Turbo", "backend": "gguf"}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="diffusers", choices=["diffusers", "gguf"], help="Backend: diffusers (default) or gguf (stable-diffusion.cpp)")
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
    # Seed variance enhancer (Turbo): optional prompt-embedding noise.
    # These flags are safe no-ops if the pipeline does not support prompt_embeds.
    parser.add_argument("--seed-variance", action="store_true", default=False)
    parser.add_argument("--seed-variance-enabled", action="store_true", dest="seed_variance_enabled", default=False)
    parser.add_argument("--seed-variance-strength", type=float, default=0.0)
    # Optional LoRA slots (1 & 2) from the UI / caller.
    # These are applied only if the underlying pipeline supports Diffusers LoRA.
    parser.add_argument("--lora", default="")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--lora2", default="")
    parser.add_argument("--lora2_scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    if getattr(args, 'backend', 'diffusers') == 'gguf':
        try:
            payload = _run_gguf(args)
            print(json.dumps(payload))
            return 0 if payload.get('files') else 1
        except Exception as e:
            payload = {'files': [], 'error': f'gguf_exception: {e}', 'trace': traceback.format_exc()}
            print(json.dumps(payload))
            return 1

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


    # ------------------------------------------------------------------
    # Seed variance enhancer (Turbo)
    #
    # This is a lightweight, fully-offline implementation that adds
    # controlled randomness to the *positive* prompt embeddings to help
    # increase visual diversity when Z-Image Turbo shows low seed variance.
    #
    # If the underlying Diffusers pipeline does not expose encode_prompt
    # or does not accept prompt_embeds, this feature becomes a safe no-op.
    # ------------------------------------------------------------------
    def _supports_kw(fn, name: str) -> bool:
        try:
            sig = inspect.signature(fn)
            return name in sig.parameters
        except Exception:
            return False

    
    # ------------------------------------------------------------------
    # Optional LoRA support (slots 1 & 2)
    #
    # Applies Diffusers-style LoRAs to the loaded ZImagePipeline when supported.
    # If the pipeline (or this Diffusers version) does not support LoRA, this
    # becomes a safe no-op.
    # ------------------------------------------------------------------
    lora_mode = None
    lora_paths = []
    lora_scales = []
    lora_names = []

    try:
        from pathlib import Path as _Path

        def _try_load_lora(p: str, adapter_name: str):
            try:
                if not p:
                    return False
                pp = _Path(p).expanduser()
                if not pp.exists():
                    return False
                if not hasattr(pipe, "load_lora_weights"):
                    return False

                kw = {}
                try:
                    if _supports_kw(pipe.load_lora_weights, "adapter_name"):
                        kw["adapter_name"] = adapter_name
                except Exception:
                    pass

                pipe.load_lora_weights(str(pp), **kw)
                return True
            except Exception as _e:
                try:
                    print(f"[zimage_cli] LoRA load failed: {p} ({_e})")
                except Exception:
                    pass
                return False

        # Collect up to two LoRAs from args (same key names as FrameVision UI)
        p1 = str(getattr(args, "lora", "") or "").strip()
        p2 = str(getattr(args, "lora2", "") or "").strip()
        s1 = float(getattr(args, "lora_scale", 1.0) or 1.0)
        s2 = float(getattr(args, "lora2_scale", 1.0) or 1.0)

        if p1 and _try_load_lora(p1, "lora1"):
            lora_paths.append(p1)
            lora_scales.append(s1)
            lora_names.append("lora1")
        if p2 and _try_load_lora(p2, "lora2"):
            lora_paths.append(p2)
            lora_scales.append(s2)
            lora_names.append("lora2")

        if lora_names:
            # Prefer multi-adapter API when available.
            try:
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters(lora_names, lora_scales)
                    lora_mode = "adapters"
                    try:
                        print(f"[zimage_cli] LoRA applied via adapters: {list(zip(lora_paths, lora_scales))}")
                    except Exception:
                        pass
                elif hasattr(pipe, "fuse_lora"):
                    # Fallback: fuse sequentially (some pipelines support only single adapter).
                    for _i, _p in enumerate(lora_paths):
                        try:
                            scale = lora_scales[_i] if _i < len(lora_scales) else 1.0
                            try:
                                if _supports_kw(pipe.fuse_lora, "lora_scale"):
                                    pipe.fuse_lora(lora_scale=scale)
                                else:
                                    pipe.fuse_lora()
                            except Exception:
                                pipe.fuse_lora()
                        except Exception:
                            pass
                    lora_mode = "fused"
                    try:
                        print(f"[zimage_cli] LoRA applied via fuse: {list(zip(lora_paths, lora_scales))}")
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        lora_mode = None
    def _maybe_build_noised_embeds(pipe, prompt: str, negative: str, strength: float, noise_seed: int, device: str):
        try:
            if not strength or strength <= 0:
                return None
            if not hasattr(pipe, "encode_prompt"):
                return None

            import torch  # local import to keep startup resilient

            # Deterministic noise per image when a base seed is provided.
            try:
                gen_dev = device if device != "cpu" else "cpu"
                gen = torch.Generator(device=gen_dev).manual_seed(int(noise_seed))
            except Exception:
                gen = torch.Generator().manual_seed(int(noise_seed))

            # Try common encode_prompt signatures used across Diffusers pipelines.
            try:
                enc = pipe.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative if negative else None,
                )
            except TypeError:
                enc = pipe.encode_prompt(prompt=prompt, device=device)

            prompt_embeds = None
            negative_embeds = None
            if isinstance(enc, (tuple, list)):
                if len(enc) >= 1:
                    prompt_embeds = enc[0]
                if len(enc) >= 2:
                    negative_embeds = enc[1]
            else:
                prompt_embeds = enc

            if prompt_embeds is None:
                return None

            # Map UI-like 0-100 strength to a modest embedding noise scale.
            # Users can tune this if they want stronger/weaker variation.
            scale = max(0.0, float(strength)) / 100.0 * 0.30
            if scale <= 0:
                return (prompt_embeds, negative_embeds)

            noise = torch.randn_like(prompt_embeds, generator=gen) * scale
            prompt_embeds = prompt_embeds + noise

            return (prompt_embeds, negative_embeds)
        except Exception:
            return None
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
    # Seed variance flags from UI/CLI
    try:
        sv_enabled = bool(getattr(args, "seed_variance", False) or getattr(args, "seed_variance_enabled", False))
        sv_strength = float(getattr(args, "seed_variance_strength", 0.0) or 0.0)
    except Exception:
        sv_enabled = False
        sv_strength = 0.0

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
            # Optional: seed variance enhancer (Turbo)
            if sv_enabled and sv_strength > 0:
                try:
                    noise_seed = (base_seed if base_seed else 0) + i + 1337
                    embeds = _maybe_build_noised_embeds(pipe, args.prompt, neg, sv_strength, noise_seed, device)
                    if embeds and _supports_kw(pipe.__call__, "prompt_embeds"):
                        pe, ne = embeds
                        kwargs.pop("prompt", None)
                        kwargs.pop("negative_prompt", None)
                        kwargs["prompt_embeds"] = pe
                        if ne is not None and _supports_kw(pipe.__call__, "negative_prompt_embeds"):
                            kwargs["negative_prompt_embeds"] = ne
                except Exception:
                    pass
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

    payload = {"files": files, "model": "Z-Image-Turbo", "lora_mode": lora_mode, "loras": lora_paths, "lora_scales": lora_scales, "lora_names": lora_names}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())