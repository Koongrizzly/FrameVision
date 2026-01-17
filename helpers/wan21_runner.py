import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _get_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_models_dir() -> Path:
    return _get_root_dir() / "models" / "wan21"


def _get_t2v_model_dir() -> Path:
    return _get_models_dir() / "Wan2.1-T2V-1.3B-Diffusers"


def _get_i2v_model_dir() -> Path:
    return _get_models_dir() / "Wan2.1-I2V-1.3B-480P-diffusers"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config produced by FrameVision.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))

    mode = str(cfg.get("mode", "t2v")).lower().strip()
    prompt = str(cfg.get("prompt", "")).strip()
    negative_prompt = (cfg.get("negative_prompt") or "").strip() or None
    width = int(cfg.get("width", 512))
    height = int(cfg.get("height", 512))
    num_frames = int(cfg.get("num_frames", 49))
    fps = int(cfg.get("fps", 12))
    guidance_scale = float(cfg.get("guidance_scale", 5.0))
    num_inference_steps = int(cfg.get("num_inference_steps", 30))
    seed = cfg.get("seed", None)
    use_random_seed = bool(cfg.get("use_random_seed", False))
    init_image_path = cfg.get("init_image_path", None)
    output_dir = Path(cfg.get("output_dir") or (_get_root_dir() / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prompt:
        raise ValueError("Prompt is empty. Please enter a prompt.")

    import torch
    from diffusers.utils import export_to_video, load_image
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("[WARN] CUDA not available. Running on CPU will be extremely slow.")

    if use_random_seed or seed is None:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed)

    log(f"Using seed: {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_suffix = "i2v" if mode == "i2v" else "t2v"
    out_path = output_dir / f"wan21_{out_suffix}_{timestamp}_{seed}.mp4"

    if mode == "i2v":
        if not init_image_path:
            raise ValueError("Image-to-video mode enabled but no image was selected.")
        image_path = Path(init_image_path)
        if not image_path.exists():
            raise ValueError(f"Input image does not exist: {image_path}")

        log(f"[I2V] Loading init image:\n  {image_path}")
        init_image = load_image(str(image_path))

        from diffusers import WanImageToVideoPipeline
        from transformers import CLIPVisionModel

        model_path = _get_i2v_model_dir()
        if not model_path.exists():
            raise RuntimeError(
                f"Missing I2V model directory: {model_path}\n"
                "Please download a Wan2.1 I2V diffusers checkpoint (e.g. Wan-AI/Wan2.1-I2V-1.3B-480P-diffusers)\n"
                "into models/wan21 and retry."
            )

        log(f"[I2V] Loading model from:\n  {model_path}")
        image_encoder = CLIPVisionModel.from_pretrained(
            str(model_path),
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        pipe = WanImageToVideoPipeline.from_pretrained(
            str(model_path),
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        log(f"[I2V] Moving pipeline to device: {device}")
        pipe.to(device)

        result = pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        frames = result.frames[0]
    else:
        from diffusers import WanPipeline

        model_path = _get_t2v_model_dir()
        if not model_path.exists():
            raise RuntimeError(
                f"Missing T2V model directory: {model_path}\n"
                "Run presets/extra_env/wan21_install.bat first to download the models."
            )

        log(f"[T2V] Loading model from:\n  {model_path}")
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        flow_shift = 3.0
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift,
        )

        pipe = WanPipeline.from_pretrained(
            str(model_path),
            vae=vae,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        pipe.scheduler = scheduler

        log(f"[T2V] Moving pipeline to device: {device}")
        pipe.to(device)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        frames = result.frames[0]

    log(f"Saving video to: {out_path}")
    export_to_video(frames, str(out_path), fps=fps)

    print(f"__WAN21_RESULT__ {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Print the exception plainly (FrameVision captures stdout)
        print(f"[ERROR] {e}", flush=True)
        raise
