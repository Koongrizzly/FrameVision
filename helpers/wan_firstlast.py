#!/usr/bin/env python3
"""FrameVision WAN 2.2 Turbo first/last-frame helper.

This helper keeps the Turbo repo untouched. It runs the Wan2.2 Turbo/few-step
pipeline, but adds a native first/last-frame control layer around it.

The helper supports:
- an optional start image
- an optional end image
- delayed end-image influence during denoising
- optional exact final-frame replacement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def _app_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "helpers":
            return here.parent.parent
        return here.parent
    except Exception:
        return Path.cwd()


APP_ROOT = _app_root()
STATE_FILE = APP_ROOT / "presets" / "setsave" / "wan_firstlast.json"


def _write_state(stage: str, **data) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        payload.update(data or {})
        tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            tmp.replace(STATE_FILE)
        except Exception:
            STATE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _fail(msg: str, code: int = 2) -> None:
    _write_state("error", error=msg)
    print(f"[wan_firstlast] ERROR: {msg}", flush=True)
    raise SystemExit(code)


def _preprocess_image(path: str, width: int, height: int, torch, TF, Image, device: str = "cuda"):
    p = Path(str(path or "")).expanduser()
    if not p.exists():
        _fail(f"image not found: {p}")
    img = Image.open(str(p)).convert("RGB")
    # Match the Turbo runner. It resizes directly to the requested canvas.
    img = img.resize((int(width), int(height)), Image.LANCZOS)
    return TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1).to(dtype=torch.bfloat16)


def _encode_single_image(pipe, path: str, width: int, height: int, torch, TF, Image):
    img = _preprocess_image(path, width, height, torch, TF, Image)
    # encode_to_latent expects [B, C, T, H, W].
    latent = pipe.vae.encode_to_latent(img.unsqueeze(0))
    # Expected Turbo shape: [B, T, C, H, W]. A single image normally gives T=1.
    if latent.ndim != 5:
        _fail(f"unexpected encoded latent shape for {path}: {tuple(latent.shape)}")
    return latent


def _pixel_frame_tensor(path: str, width: int, height: int, torch, TF, Image, device: str = "cuda"):
    p = Path(str(path or "")).expanduser()
    if not p.exists():
        _fail(f"image not found: {p}")
    img = Image.open(str(p)).convert("RGB")
    img = img.resize((int(width), int(height)), Image.LANCZOS)
    frame = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device=device, dtype=torch.bfloat16)
    return frame


def _apply_start_end_constraints(latent, start_latent=None, end_latent=None, end_weight: float = 0.0):
    out = latent
    if start_latent is not None:
        out[:, 0:1, :, :, :] = start_latent[:, :1, :, :, :]
    if end_latent is not None and float(end_weight) > 0.0:
        ew = float(max(0.0, min(1.0, end_weight)))
        out[:, -1:, :, :, :] = (1.0 - ew) * out[:, -1:, :, :, :] + ew * end_latent[:, -1:, :, :, :]
    return out


def _end_weight_at_step(step_index: int, total_steps: int, start_fraction: float, strength: float) -> float:
    try:
        total = max(1, int(total_steps))
        idx = max(0, int(step_index))
        start_fraction = float(max(0.0, min(0.98, start_fraction)))
        strength = float(max(0.0, min(1.0, strength)))
        start_idx = int(round((total - 1) * start_fraction))
        if idx <= start_idx:
            return 0.0
        denom = max(1, (total - 1) - start_idx)
        ramp = float(idx - start_idx) / float(denom)
        return float(max(0.0, min(1.0, ramp * strength)))
    except Exception:
        return 0.0


def _needed_seq_len(noise) -> int:
    # Turbo wrapper flattens mask2[:, :, 0, ::2, ::2].
    # This is the same sequence length needed by the Wan2.2 generator.
    try:
        return int(noise.shape[1] * ((noise.shape[3] + 1) // 2) * ((noise.shape[4] + 1) // 2))
    except Exception:
        return 0


def _firstlast_inference(
    pipe,
    noise,
    text_prompts: List[str],
    start_latent=None,
    end_latent=None,
    start_mask=None,
    end_influence_start: float = 0.7,
    end_influence_strength: float = 1.0,
):
    import torch
    from tqdm import tqdm

    conditional_dict = pipe.text_encoder(text_prompts=text_prompts)
    noisy_image_or_video = noise

    mask2 = start_mask.to(noise.device, dtype=noise.dtype) if start_mask is not None else torch.ones_like(noisy_image_or_video, dtype=noise.dtype, device=noise.device)
    if start_latent is not None:
        start_latent = start_latent.to(noise.device, dtype=noise.dtype)
    if end_latent is not None:
        end_latent = end_latent.to(noise.device, dtype=noise.dtype)

    noisy_image_or_video = _apply_start_end_constraints(
        noisy_image_or_video,
        start_latent=start_latent,
        end_latent=end_latent,
        end_weight=0.0,
    ).to(noise.device, dtype=noise.dtype)

    progress_bar = tqdm(
        enumerate(pipe.denoising_step_list),
        total=len(pipe.denoising_step_list),
        desc="Denoising Steps",
        unit="step",
    )

    pred_image_or_video = noisy_image_or_video
    total_steps = len(pipe.denoising_step_list)
    for index, current_timestep in progress_bar:
        end_weight = _end_weight_at_step(index, total_steps, end_influence_start, end_influence_strength)
        if end_latent is not None and end_weight > 0.0:
            noisy_image_or_video = _apply_start_end_constraints(
                noisy_image_or_video,
                start_latent=start_latent,
                end_latent=end_latent,
                end_weight=end_weight,
            ).to(noise.device, dtype=noise.dtype)

        wan22_input_timestep = torch.tensor([current_timestep.item()], device=noise.device, dtype=noise.dtype)
        temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)
        temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
        seq_len = int(getattr(pipe.generator, "seq_len", temp_ts.size(1)))
        if seq_len < temp_ts.size(1):
            pipe.generator.seq_len = int(temp_ts.size(1))
            seq_len = int(temp_ts.size(1))
        temp_ts = torch.cat([
            temp_ts,
            temp_ts.new_ones(temp_ts.shape[0], seq_len - temp_ts.size(1)) * wan22_input_timestep.unsqueeze(-1),
        ], dim=1)
        wan22_input_timestep = temp_ts.to(noise.device, dtype=torch.long)

        _, pred_image_or_video = pipe.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
            wan22_input_timestep=wan22_input_timestep,
            mask2=mask2,
            wan22_image_latent=start_latent,
        )

        pred_image_or_video = _apply_start_end_constraints(
            pred_image_or_video,
            start_latent=start_latent,
            end_latent=end_latent,
            end_weight=end_weight,
        ).to(noise.device, dtype=noise.dtype)

        if index < len(pipe.denoising_step_list) - 1:
            next_timestep = pipe.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2], dtype=torch.long, device=noise.device
            )
            noisy_image_or_video = pipe.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1),
            ).unflatten(0, noise.shape[:2])
            noisy_image_or_video = _apply_start_end_constraints(
                noisy_image_or_video,
                start_latent=start_latent,
                end_latent=end_latent,
                end_weight=end_weight,
            ).to(noise.device, dtype=noise.dtype)

    final_end_weight = float(max(0.0, min(1.0, end_influence_strength))) if end_latent is not None else 0.0
    pred_image_or_video = _apply_start_end_constraints(
        pred_image_or_video,
        start_latent=start_latent,
        end_latent=end_latent,
        end_weight=final_end_weight,
    ).to(noise.device, dtype=noise.dtype)

    video = pipe.vae.decode_to_pixel(pred_image_or_video)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FrameVision Wan2.2 Turbo first/last-frame helper")
    parser.add_argument("--repo_root", "--wan-root", dest="repo_root", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image", default=None, help="Alias for --start_image")
    parser.add_argument("--start_image", default=None)
    parser.add_argument("--end_image", "--last_image", dest="end_image", default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--h", type=int, default=704)
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--end-influence-start", dest="end_influence_start", type=float, default=0.70, help="When end-image influence begins during denoising (0.0-1.0)")
    parser.add_argument("--end-influence-strength", dest="end_influence_strength", type=float, default=1.00, help="Maximum end-image influence strength (0.0-1.0)")
    parser.add_argument("--force-exact-last-frame", dest="force_exact_last_frame", action="store_true", help="Replace the final visible frame with the selected end image")
    ns = parser.parse_args(argv)

    if int(ns.num_frames) % 4 != 1:
        _fail("num_frames must be 1 more than a multiple of 4")

    start_image = str(ns.start_image or ns.image or "").strip() or None
    end_image = str(ns.end_image or "").strip() or None
    if not start_image and not end_image:
        _fail("first/last helper needs --start_image/--image and/or --end_image")

    repo_root = Path(ns.repo_root).expanduser().resolve()
    if not repo_root.exists():
        _fail(f"Turbo repo root not found: {repo_root}")
    config_path = Path(ns.config_path).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    checkpoint_folder = Path(ns.checkpoint_folder).expanduser()
    if not checkpoint_folder.is_absolute():
        checkpoint_folder = repo_root / checkpoint_folder
    output_path = Path(ns.output_path).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    _write_state(
        "starting",
        repo_root=str(repo_root),
        config_path=str(config_path),
        checkpoint_folder=str(checkpoint_folder),
        output_path=str(output_path),
        start_image=str(start_image or ""),
        end_image=str(end_image or ""),
        width=int(ns.w),
        height=int(ns.h),
        frames=int(ns.num_frames),
        fps=int(ns.fps),
        seed=int(ns.seed),
        end_influence_start=float(ns.end_influence_start),
        end_influence_strength=float(ns.end_influence_strength),
        force_exact_last_frame=bool(ns.force_exact_last_frame),
    )

    old_cwd = Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    try:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        from diffusers.utils import export_to_video
        from omegaconf import OmegaConf
        from pipeline import Wan22FewstepInferencePipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)

        if not config_path.exists():
            _fail(f"config not found: {config_path}")
        if not (checkpoint_folder / "model.pt").exists():
            _fail(f"Turbo model.pt not found: {checkpoint_folder / 'model.pt'}")

        config = OmegaConf.load(str(config_path))
        pipe = Wan22FewstepInferencePipeline(config)
        state_dict = torch.load(str(checkpoint_folder / "model.pt"), map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")
            new_key = new_key.replace("_checkpoint_wrapped_module.", "")
            new_key = new_key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        missing, unexpected = pipe.generator.load_state_dict(new_state_dict, strict=False)
        if len(unexpected) != 0:
            _fail(f"unexpected keys in Turbo state_dict: {unexpected}")
        if len(missing) != 0:
            print(f"[wan_firstlast] missing keys while loading Turbo state_dict: {missing}", flush=True)
        pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

        latent_frames = (int(ns.num_frames) - 1) // 4 + 1
        noise = torch.randn(
            1,
            latent_frames,
            48,
            int(ns.h) // 16,
            int(ns.w) // 16,
            generator=torch.Generator(device="cuda").manual_seed(int(ns.seed)),
            dtype=torch.bfloat16,
            device="cuda",
        )
        need_seq = _needed_seq_len(noise)
        try:
            if need_seq and int(getattr(pipe.generator, "seq_len", 0) or 0) < need_seq:
                print(f"[wan_firstlast] raising generator seq_len {getattr(pipe.generator, 'seq_len', 0)} -> {need_seq}", flush=True)
                pipe.generator.seq_len = int(need_seq)
        except Exception:
            pass

        start_latent = None
        start_mask = torch.ones_like(noise, dtype=noise.dtype, device=noise.device)
        end_latent = None

        if start_image:
            z_start = _encode_single_image(pipe, start_image, int(ns.w), int(ns.h), torch, TF, Image).to(noise.device, dtype=noise.dtype)
            start_latent = torch.zeros_like(noise, dtype=noise.dtype, device=noise.device)
            start_latent[:, 0:1, :, :, :] = z_start[:, :1, :, :, :]
            start_mask[:, 0:1, :, :, :] = 0.0
        if end_image:
            z_end = _encode_single_image(pipe, end_image, int(ns.w), int(ns.h), torch, TF, Image).to(noise.device, dtype=noise.dtype)
            end_latent = torch.zeros_like(noise, dtype=noise.dtype, device=noise.device)
            end_latent[:, -1:, :, :, :] = z_end[:, -1:, :, :, :]

        _write_state("denoising", output_path=str(output_path))
        video_tensor = _firstlast_inference(
            pipe,
            noise=noise,
            text_prompts=[ns.prompt or "A cinematic video, realistic motion"],
            start_latent=start_latent,
            end_latent=end_latent,
            start_mask=start_mask,
            end_influence_start=float(ns.end_influence_start),
            end_influence_strength=float(ns.end_influence_strength),
        )

        if end_image and bool(ns.force_exact_last_frame):
            try:
                exact_last = _pixel_frame_tensor(end_image, int(ns.w), int(ns.h), torch, TF, Image, device="cpu")
                exact_last = (exact_last * 0.5 + 0.5).clamp(0, 1).to(video_tensor.device, dtype=video_tensor.dtype)
                if video_tensor.ndim == 5:
                    video_tensor[0, -1, :, :, :] = exact_last
                elif video_tensor.ndim == 4:
                    video_tensor[-1, :, :, :] = exact_last
            except Exception as e:
                print(f"[wan_firstlast] warning: failed to force exact last frame: {e}", flush=True)

        video = video_tensor[0].permute(0, 2, 3, 1).cpu().numpy()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_state("saving", output_path=str(output_path))
        export_to_video(video, str(output_path), fps=max(1, int(ns.fps or 24)))
        if not output_path.exists():
            _fail(f"output was not created: {output_path}", code=1)
        _write_state("done", output_path=str(output_path), size_bytes=output_path.stat().st_size)
        return 0
    finally:
        try:
            os.chdir(str(old_cwd))
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
