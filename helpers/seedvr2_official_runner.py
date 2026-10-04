from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path


def _install_flash_attn_fallback_if_needed():
    """Provide the tiny subset of flash_attn used by the official SeedVR code.

    The official repository imports flash_attn unconditionally, but the published
    environment/apex wheels target Linux. Native Windows builds commonly have no
    flash-attn package. PyTorch SDPA is semantically suitable for this inference
    path; it is slower but lets the official model run without a Linux-only wheel.
    """
    try:
        import flash_attn  # noqa: F401
        print("Attention backend: flash_attn package found.", flush=True)
        return False
    except Exception:
        pass

    import types
    import torch
    import torch.nn.functional as F

    mod = types.ModuleType("flash_attn")

    def flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None, causal=False,
        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
        deterministic=False, return_attn_probs=False,
        block_table=None, **kwargs,
    ):
        # q/k/v use FlashAttention's packed varlen layout: [total_tokens, heads, dim].
        # Process each packed sequence independently with torch SDPA and concatenate.
        outs = []
        nseq = int(cu_seqlens_q.numel()) - 1
        for i in range(nseq):
            qs, qe = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
            ks, ke = int(cu_seqlens_k[i].item()), int(cu_seqlens_k[i + 1].item())
            qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # 1,h,L,d
            ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
            vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
            sdpa_kwargs = dict(dropout_p=float(dropout_p or 0.0), is_causal=bool(causal))
            if softmax_scale is not None:
                sdpa_kwargs["scale"] = float(softmax_scale)
            oi = F.scaled_dot_product_attention(qi, ki, vi, **sdpa_kwargs)
            outs.append(oi.squeeze(0).transpose(0, 1).contiguous())
        out = torch.cat(outs, dim=0) if outs else q.new_empty(q.shape)
        if return_attn_probs:
            # SeedVR does not request attention probabilities; keep a compatible tuple.
            return out, None, None
        return out

    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod
    print("Attention backend: flash_attn unavailable; using PyTorch SDPA fallback.", flush=True)
    return True


def _replace_linux_fused_norms(config):
    """Replace Apex fused norm names with equivalent portable PyTorch/diffusers norms."""
    replacements = {"fusedrms": "rms", "fusedln": "layer"}

    def walk(node):
        try:
            from omegaconf import DictConfig, ListConfig
        except Exception:
            DictConfig = ListConfig = ()
        if isinstance(node, str):
            return replacements.get(node, node)
        if DictConfig and isinstance(node, DictConfig):
            for key in list(node.keys()):
                val = node[key]
                if isinstance(val, str) and val in replacements:
                    node[key] = replacements[val]
                else:
                    walk(val)
            return node
        if ListConfig and isinstance(node, ListConfig):
            for i in range(len(node)):
                val = node[i]
                if isinstance(val, str) and val in replacements:
                    node[i] = replacements[val]
                else:
                    walk(val)
            return node
        if isinstance(node, dict):
            for key, val in list(node.items()):
                node[key] = walk(val)
        elif isinstance(node, list):
            for i, val in enumerate(node):
                node[i] = walk(val)
        return node

    walk(config)
    print("Normalization backend: Apex fused norms replaced with portable RMSNorm/LayerNorm.", flush=True)


def _repo_imports(repo: Path):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    _install_flash_attn_fallback_if_needed()

    import datetime
    import torch
    import mediapy
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchvision.io.video import read_video
    from torchvision.io import read_image
    from torchvision.transforms import Compose, Lambda, Normalize

    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    from common.distributed import get_device, init_torch
    from common.distributed.advanced import (
        get_sequence_parallel_rank,
        init_sequence_parallel,
    )
    from common.seed import set_seed
    from common.config import load_config
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    return locals()


def _load_state(path: Path, torch):
    ext = path.suffix.lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                "This checkpoint is .safetensors but the safetensors package is not installed. "
                "Install it in the SeedVR environment with: pip install safetensors"
            ) from e
        return load_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu", mmap=True)


def _configure_runner(repo: Path, model_size: str, dit_checkpoint: Path, vae_checkpoint: Path, sp_size: int):
    ns = _repo_imports(repo)
    torch = ns["torch"]
    OmegaConf = ns["OmegaConf"]
    load_config = ns["load_config"]
    VideoDiffusionInfer = ns["VideoDiffusionInfer"]
    init_torch = ns["init_torch"]
    init_sequence_parallel = ns["init_sequence_parallel"]
    get_device = ns["get_device"]
    datetime = ns["datetime"]

    cfg_dir = "configs_3b" if model_size == "3B" else "configs_7b"
    config = load_config(str(repo / cfg_dir / "main.yaml"))
    # The official config hard-codes ./ckpts/ema_vae.pth. Make the selected path explicit.
    OmegaConf.set_readonly(config, False)
    config.vae.checkpoint = str(vae_checkpoint)
    _replace_linux_fused_norms(config)

    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)

    # The official scripts are launched under torchrun and initialize a
    # distributed process group even for code paths that later behave as a
    # single rank. Native Windows does not need (and may not support) that
    # process-group setup for this standalone sp_size=1 launcher.
    #
    # Most SeedVR helpers already return rank=0/world_size=1 when no sequence
    # parallel group exists. One helper is the exception:
    # get_sequence_parallel_global_ranks() calls dist.get_rank() unconditionally.
    # Patch only that helper to return [0] when torch.distributed is not
    # initialized. This preserves the official single-rank VAE logic while
    # avoiding fake Gloo/NCCL initialization and any communication.
    if int(sp_size) == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the official SeedVR2 inference code.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        import torch.distributed as dist
        import common.distributed.advanced as advanced_dist
        _official_get_sp_global_ranks = advanced_dist.get_sequence_parallel_global_ranks

        def _standalone_get_sequence_parallel_global_ranks():
            if not dist.is_available() or not dist.is_initialized():
                return [0]
            return _official_get_sp_global_ranks()

        advanced_dist.get_sequence_parallel_global_ranks = _standalone_get_sequence_parallel_global_ranks
        print(
            "SeedVR2 standalone mode: single GPU rank=0; distributed communication disabled "
            "with official sequence-parallel rank fallback patched for Windows.",
            flush=True,
        )
    else:
        if os.name == "nt":
            raise RuntimeError(
                "Sequence-parallel SeedVR2 requires the official distributed torchrun/NCCL setup, "
                "which is not supported by this native Windows standalone launcher. Use sp_size=1."
            )
        # Non-Windows multi-GPU launches still follow the official code path and
        # therefore require the torchrun environment (MASTER_ADDR, RANK, etc.).
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        init_sequence_parallel(sp_size)

    # Same model creation path as the official VideoDiffusionInfer, but make the
    # checkpoint configurable and allow a state-dict-compatible safetensors file.
    init_device = "cpu"
    with torch.device(init_device):
        from common.config import create_object
        runner.dit = create_object(runner.config.dit.model)
    runner.dit.set_gradient_checkpointing(runner.config.dit.gradient_checkpoint)
    state = _load_state(dit_checkpoint, torch)
    loading_info = runner.dit.load_state_dict(state, strict=True, assign=True)
    print(f"Loading pretrained DiT ckpt from {dit_checkpoint}", flush=True)
    print(f"Loading info: {loading_info}", flush=True)
    try:
        from common.distributed.meta_init_utils import meta_non_persistent_buffer_init_fn
        runner.dit = meta_non_persistent_buffer_init_fn(runner.dit)
    except Exception:
        pass
    # Keep the DiT on CPU after loading. It is moved to CUDA only for the diffusion stage.
    runner.dit.to("cpu")

    # Create/load the VAE following the official configure_vae_model() path,
    # but use _load_state() so compatible .safetensors VAE checkpoints work too.
    from common.config import create_object
    dtype = getattr(torch, runner.config.vae.dtype)
    runner.vae = create_object(runner.config.vae.model)
    runner.vae.requires_grad_(False).eval()
    # Keep VAE on CPU after loading; the staged memory manager moves it to CUDA only for encode/decode.
    runner.vae.to(device="cpu", dtype=dtype)
    vae_state = _load_state(vae_checkpoint, torch)
    vae_loading_info = runner.vae.load_state_dict(vae_state, strict=True)
    print(f"Loading pretrained VAE ckpt from {vae_checkpoint}", flush=True)
    print(f"VAE loading info: {vae_loading_info}", flush=True)
    if hasattr(runner.vae, "set_causal_slicing") and hasattr(runner.config.vae, "slicing"):
        runner.vae.set_causal_slicing(**runner.config.vae.slicing)
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner, ns


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _cuda_mem_report(torch, label: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        alloc_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()
        gb = 1024 ** 3
        print(
            f"VRAM [{label}]: free={free_b/gb:.2f} GiB / {total_b/gb:.2f} GiB, "
            f"allocated={alloc_b/gb:.2f} GiB, reserved={reserved_b/gb:.2f} GiB",
            flush=True,
        )
    except Exception:
        pass


def _cuda_cleanup(torch, label: str = "cleanup") -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    _cuda_mem_report(torch, label)


def run_one(args) -> Path:
    repo = Path(args.repo).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    dit_checkpoint = Path(args.dit).resolve()
    vae_checkpoint = Path(args.vae).resolve()

    for p, label in [
        (repo, "SeedVR repo"), (input_path, "input"),
        (dit_checkpoint, "DiT checkpoint"), (vae_checkpoint, "VAE checkpoint")
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    runner, ns = _configure_runner(repo, args.model_size, dit_checkpoint, vae_checkpoint, args.sp_size)
    torch = ns["torch"]
    rearrange = ns["rearrange"]
    mediapy = ns["mediapy"]
    Compose, Lambda, Normalize = ns["Compose"], ns["Lambda"], ns["Normalize"]
    DivisibleCrop, NaResize, Rearrange = ns["DivisibleCrop"], ns["NaResize"], ns["Rearrange"]
    read_video, read_image = ns["read_video"], ns["read_image"]
    get_device = ns["get_device"]
    get_sequence_parallel_rank = ns["get_sequence_parallel_rank"]
    set_seed = ns["set_seed"]

    # Official SeedVR2 is a one-step model. Keep official defaults unless user overrides.
    runner.config.diffusion.cfg.scale = float(args.cfg_scale)
    runner.config.diffusion.cfg.rescale = float(args.cfg_rescale)
    runner.config.diffusion.timesteps.sampling.steps = int(args.steps)
    runner.configure_diffusion()
    set_seed(int(args.seed), same_across_ranks=True)

    # Prefer embeddings stored beside the official DiT checkpoint. This lets the
    # GUI keep all Hugging Face model assets under <FrameVision>/models/seedvr2_3b_official/.
    model_dir = dit_checkpoint.parent
    pos_path = model_dir / "pos_emb.pt"
    neg_path = model_dir / "neg_emb.pt"
    if not pos_path.exists():
        pos_path = repo / "pos_emb.pt"
    if not neg_path.exists():
        neg_path = repo / "neg_emb.pt"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            "SeedVR2 text embeddings are missing. Expected pos_emb.pt and neg_emb.pt "
            f"beside the DiT checkpoint ({model_dir}) or in the SeedVR repo root ({repo})."
        )
    text_pos_embeds = torch.load(str(pos_path), map_location="cpu")
    text_neg_embeds = torch.load(str(neg_path), map_location="cpu")
    text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}

    video_transform = Compose([
        NaResize(resolution=(args.height * args.width) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

    if _is_image_file(input_path):
        video = read_image(str(input_path)).unsqueeze(0) / 255.0
        if args.sp_size > 1:
            raise ValueError("sp_size must be 1 for image input")
        save_fps = float(args.out_fps or 24.0)
    else:
        video, _, info = read_video(str(input_path), output_format="TCHW")
        video = video / 255.0
        save_fps = float(args.out_fps or info["video_fps"])

    print(f"Read input size: {tuple(video.size())}", flush=True)
    # Preprocess on CPU. Keeping the full source video on CUDA wastes VRAM before VAE encode.
    cond = video_transform(video)
    ori_length = cond.size(1)
    input_cond = cond.detach().cpu()
    del video

    # This padding rule is copied from the official SeedVR2 inference script.
    t = cond.size(1)
    sp_size = int(args.sp_size)
    if t != 1:
        if t <= 4 * sp_size:
            padding = [cond[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            cond = torch.cat([cond, torch.cat(padding, dim=1)], dim=1)
        elif (t - 1) % (4 * sp_size) != 0:
            pad_count = 4 * sp_size - ((t - 1) % (4 * sp_size))
            padding = [cond[:, -1].unsqueeze(1)] * pad_count
            cond = torch.cat([cond, torch.cat(padding, dim=1)], dim=1)

    print(f"Encoding video tensor: {tuple(cond.size())}", flush=True)
    print("MEMORY STAGE 1/3: VAE encode only", flush=True)
    _cuda_cleanup(torch, "before VAE encode")
    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_cuda = cond.to(get_device(), non_blocking=False)
    _cuda_mem_report(torch, "VAE encode resident")
    cond_latents_gpu = runner.vae_encode([cond_cuda])
    # Latents are much smaller than RGB frames; move them to CPU before unloading VAE.
    cond_latents = [x.detach().cpu() for x in cond_latents_gpu]
    del cond_latents_gpu, cond_cuda, cond
    runner.vae.to("cpu")
    _cuda_cleanup(torch, "after VAE encode / VAE offloaded")

    print("MEMORY STAGE 2/3: DiT inference only", flush=True)
    runner.dit.to(get_device())
    _cuda_mem_report(torch, "DiT resident")

    for k in ("texts_pos", "texts_neg"):
        text_embeds[k] = [e.to(get_device()) for e in text_embeds[k]]

    latent_blur = cond_latents[0].to(get_device())
    noise = torch.randn_like(latent_blur)
    aug_noise = torch.randn_like(latent_blur)
    cond_noise_scale = 0.0
    timestep = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
    shape = torch.tensor(latent_blur.shape[1:], device=get_device())[None]
    timestep = runner.timestep_transform(timestep, shape)
    latent_blur = runner.schedule.forward(latent_blur, aug_noise, timestep)
    condition = runner.get_condition(noise, task="sr", latent_blur=latent_blur)

    # The official inference helper performs VAE decode near the end. Wrap that exact
    # decode entry point so the DiT is evicted before the VAE returns to CUDA. This
    # prevents WDDM shared-memory spill on 24 GB cards.
    original_vae_decode = getattr(runner, "vae_decode", None)
    if callable(original_vae_decode):
        def _staged_vae_decode(*va_args, **va_kwargs):
            print("MEMORY STAGE 3/3: VAE decode only (DiT -> CPU first)", flush=True)
            runner.dit.to("cpu")
            _cuda_cleanup(torch, "DiT offloaded before VAE decode")
            runner.vae.to(get_device())
            _cuda_mem_report(torch, "VAE decode resident")
            try:
                out = original_vae_decode(*va_args, **va_kwargs)
            finally:
                runner.vae.to("cpu")
                _cuda_cleanup(torch, "after VAE decode / VAE offloaded")
            return out
        runner.vae_decode = _staged_vae_decode

    print("Running SeedVR2 inference with strict staged VRAM residency...", flush=True)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=[noise],
            conditions=[condition],
            # Do NOT use the official generic DiT offload option in staged mode; on
            # Windows/WDDM it can continuously migrate model pages into shared memory.
            dit_offload=False,
            **text_embeds,
        )

    # Ensure no large model remains resident after inference/decode.
    runner.dit.to("cpu")
    runner.vae.to("cpu")
    _cuda_cleanup(torch, "inference complete / models offloaded")

    sample = video_tensors[0]
    sample = rearrange(sample[:, None], "c t h w -> t c h w") if sample.ndim == 3 else rearrange(sample, "c t h w -> t c h w")
    if ori_length < sample.shape[0]:
        sample = sample[:ori_length]

    # Official optional color fix, if the user has installed the same color_fix.py.
    if args.color_fix:
        color_fix_path = repo / "projects" / "video_diffusion_sr" / "color_fix.py"
        if color_fix_path.exists():
            from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
            inp = rearrange(input_cond[:, None], "c t h w -> t c h w") if input_cond.ndim == 3 else rearrange(input_cond, "c t h w -> t c h w")
            sample = wavelet_reconstruction(sample.to("cpu"), inp[:sample.size(0)].to("cpu"))
        else:
            print("Color fix requested but projects/video_diffusion_sr/color_fix.py is absent; continuing without it.", flush=True)
            sample = sample.to("cpu")
    else:
        sample = sample.to("cpu")

    sample = rearrange(sample[:, None], "t c h w -> t h w c") if sample.ndim == 3 else rearrange(sample, "t c h w -> t h w c")
    sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sample.shape[0] == 1:
        mediapy.write_image(str(output_path), sample.squeeze(0))
    else:
        mediapy.write_video(str(output_path), sample, fps=save_fps)

    del sample, video_tensors, cond_latents
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DONE: {output_path}", flush=True)
    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean PySide6 companion runner for the official ByteDance SeedVR2 repo")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--model-size", choices=["3B", "7B"], default="3B")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--cfg-rescale", type=float, default=0.0)
    ap.add_argument("--sp-size", type=int, default=1)
    ap.add_argument("--out-fps", type=float, default=None)
    ap.add_argument("--dit-offload", action="store_true", default=False, help="Legacy compatibility flag. Strict staged offload is always used on this standalone runner.")
    ap.add_argument("--color-fix", action="store_true", default=False)
    args = ap.parse_args()
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
