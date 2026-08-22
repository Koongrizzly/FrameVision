
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_ORG = "FrameVision"
APP_NAME = "LTX25Helper"
HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent if HERE.name.lower() == "helpers" else HERE
DEFAULT_ROOT = APP_ROOT
DEFAULT_REPO = APP_ROOT / "presets" / "extra_env" / "LTX-2"
DEFAULT_ENV = APP_ROOT / "environments" / "ltx25"
DEFAULT_MODELS = APP_ROOT / "models" / "ltx-2.5"
DEFAULT_CACHE = DEFAULT_MODELS / "cache"
DEFAULT_OUTPUT = APP_ROOT / "output"
DEFAULT_SETTINGS_PATH = APP_ROOT / "presets" / "setsave" / "ltx25.json"
DEFAULT_LOGS_DIR = APP_ROOT / "logs"
DEFAULT_INSTALLER = APP_ROOT / "presets" / "extra_env" / "install_ltx_2_5_distilled.bat"
DEFAULT_PROMPT = "A red sports car races along a coastal highway at sunset, cinematic tracking shot, realistic lighting, detailed reflections, fast natural motion."

# ----------------------------- Worker mode ---------------------------------

def run_worker():
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    import torch
    import copy
    import gc
    import dataclasses
    from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
    from ltx_core.loader.helpers import create_meta_model
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.model.transformer import LTXModelConfigurator
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, get_gemma_ops
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.blocks import PromptEncoder, AudioConditioner
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import combined_image_conditionings, ensure_tiling_config, tiling_scale_factors_for_vae
    from ltx_pipelines.utils.media_io import encode_video, decode_audio_from_file
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode, ModalitySpec
    from optimum.quanto import requantize
    from safetensors.torch import load_file as load_safetensors_file

    cache: dict[tuple, object] = {}
    shared_registry = ModelRegistry(cache_weights=True, cache_models=True)

    def _reset_shared_registry(reason=""):
        nonlocal shared_registry
        try:
            shared_registry.clear()
        except Exception:
            pass
        gc.collect()
        shared_registry = ModelRegistry(cache_weights=True, cache_models=True)
        if reason:
            print(f"[RAM CACHE] Registry reset: {reason}", flush=True)

    prompt_embedding_cache: dict[str, object] = {}

    def _map_tensors(obj, device, *, clone=False):
        if torch.is_tensor(obj):
            out = obj.detach().to(device)
            return out.clone() if clone else out
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            values = {f.name: _map_tensors(getattr(obj, f.name), device, clone=clone) for f in dataclasses.fields(obj)}
            try:
                return dataclasses.replace(obj, **values)
            except Exception:
                return type(obj)(**values)
        if isinstance(obj, dict):
            return {k: _map_tensors(v, device, clone=clone) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_map_tensors(v, device, clone=clone) for v in obj]
        if isinstance(obj, tuple):
            return type(obj)(*[_map_tensors(v, device, clone=clone) for v in obj]) if hasattr(obj, '_fields') else tuple(_map_tensors(v, device, clone=clone) for v in obj)
        return copy.deepcopy(obj) if clone else obj

    class CachedPromptEncoder:
        """Small CPU cache for encoded prompt contexts; never keeps prompt tensors in VRAM."""
        def __init__(self, inner, device):
            self._inner = inner
            self._device = device

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, *args, **kwargs):
            key = repr((args, sorted(kwargs.items(), key=lambda kv: kv[0])))
            cached = prompt_embedding_cache.get(key)
            if cached is not None:
                print('[PROMPT CACHE] Reusing encoded prompt embeddings from RAM.', flush=True)
                return _map_tensors(cached, self._device, clone=False)
            result = self._inner(*args, **kwargs)
            prompt_embedding_cache[key] = _map_tensors(result, torch.device('cpu'), clone=True)
            print('[PROMPT CACHE] Cached encoded prompt embeddings in RAM.', flush=True)
            return result

    def _find_cacheable_builder(builder):
        seen = set()
        cur = builder
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            checkpoint = getattr(cur, 'checkpoint', None) or getattr(cur, 'model_path', None)
            if checkpoint:
                return cur
            nxt = None
            for attr in ('inner', '_inner', 'builder', '_builder'):
                candidate = getattr(cur, attr, None)
                if candidate is not None and candidate is not cur:
                    nxt = candidate
                    break
            cur = nxt
        return None

    def _preload_builder_state(builder, label):
        base = _find_cacheable_builder(builder)
        if base is None:
            print(f'[RAM CACHE] Could not find a cacheable {label} builder; it will cache on first real use.', flush=True)
            return
        checkpoint = getattr(base, 'checkpoint', None) or getattr(base, 'model_path', None)
        paths = list(checkpoint) if isinstance(checkpoint, (list, tuple)) else [str(checkpoint)]
        sd_ops = getattr(base, 'model_sd_ops', None)
        if shared_registry.get(paths, sd_ops) is not None:
            print(f'[RAM CACHE] {label} state dict already cached.', flush=True)
            return
        print(f'[RAM CACHE] Preloading {label} state dict from disk...', flush=True)
        loader = SafetensorsModelStateDictLoader()
        state = loader.load(paths if len(paths) > 1 else paths[0], sd_ops=sd_ops, device=torch.device('cpu'))
        try:
            shared_registry.add(paths, sd_ops, state)
        except ValueError:
            pass
        size = getattr(state, 'size', 0) or 0
        print(f'[RAM CACHE] {label} cached in RAM ({size / (1024**3):.2f} GiB).', flush=True)

    def _preload_pipeline(pipeline, job):
        print('[RAM CACHE] Preload requested. This can use roughly 60+ GiB RAM for transformer + Gemma.', flush=True)
        _preload_builder_state(pipeline.stage._transformer_builder, 'transformer')
        text_builder = getattr(pipeline.prompt_encoder, '_streaming_text_encoder_builder', None) or pipeline.prompt_encoder._text_encoder_builder
        _preload_builder_state(text_builder, 'text encoder')
        if job.get('cache_prompt_embeddings', False) and job.get('prompt'):
            print('[PROMPT CACHE] Pre-encoding the current prompt...', flush=True)
            pipeline.prompt_encoder([job['prompt']], enhance_first_prompt=bool(job.get('enhance_prompt', False)), enhance_prompt_image=None)
        print('[RAM CACHE] Preload complete.', flush=True)

    class QuantoBundleBuilder:
        """LTX builder for the INT8 Quanto bundles made by the companion quantizer."""
        _state_cache: dict[str, dict] = {}
        _qmap_cache: dict[str, dict] = {}

        def __init__(self, bundle_dir, model_class_configurator, *, module_ops=(), model_sd_ops=None, loras=()):
            self._bundle_dir = str(Path(bundle_dir))
            self._weights = str(Path(bundle_dir) / "model.safetensors")
            self._qmap = str(Path(bundle_dir) / "quantization_map.json")
            self._model_class_configurator = model_class_configurator
            self._module_ops = tuple(module_ops)
            self._model_sd_ops = model_sd_ops
            self._loras = tuple(loras)
            self._loader = SafetensorsModelStateDictLoader()
            if not Path(self._weights).is_file() or not Path(self._qmap).is_file():
                raise FileNotFoundError(
                    f"Invalid INT8 Quanto bundle: {bundle_dir}\\n"
                    "Expected model.safetensors and quantization_map.json."
                )

        @property
        def model_sd_ops(self): return self._model_sd_ops
        @property
        def module_ops(self): return self._module_ops
        @property
        def loras(self): return self._loras
        @property
        def checkpoint(self): return self._weights
        @property
        def model_path(self): return self._weights
        @property
        def keeps_gpu_resident_weights(self): return False
        @property
        def fuse_rule(self): return None

        def model_metadata(self):
            return self._loader.metadata(self._weights)

        def model_config(self):
            return self.model_metadata().get("config", {})

        def with_module_ops(self, module_ops):
            clone = copy.copy(self); clone._module_ops = tuple(module_ops); return clone

        def with_sd_ops(self, sd_ops):
            clone = copy.copy(self); clone._model_sd_ops = sd_ops; return clone

        def with_loras(self, loras):
            if loras:
                raise ValueError("LoRA fusion is not supported with INT8 Quanto bundles yet.")
            clone = copy.copy(self); clone._loras = (); return clone

        def with_fuse_rule(self, fuse_rule):
            return copy.copy(self)

        @classmethod
        def _load_bundle_state(cls, weights_path, qmap_path):
            state = cls._state_cache.get(weights_path)
            if state is None:
                print(f"[INT8] Reading quantized weights into RAM: {weights_path}", flush=True)
                state = load_safetensors_file(weights_path, device="cpu")
                cls._state_cache[weights_path] = state
                print(f"[INT8] Cached quantized state dict in RAM: {Path(weights_path).name}", flush=True)
            else:
                print(f"[INT8] Reusing RAM-cached quantized weights: {Path(weights_path).name}", flush=True)
            qmap = cls._qmap_cache.get(qmap_path)
            if qmap is None:
                with open(qmap_path, "r", encoding="utf-8") as f:
                    qmap = json.load(f)
                cls._qmap_cache[qmap_path] = qmap
            return state, qmap

        def build(self, device=None, dtype=None, **kwargs):
            if self._loras:
                raise ValueError("LoRA fusion is not supported with INT8 Quanto bundles yet.")
            target = torch.device("cuda" if device is None else device)
            model = create_meta_model(self._model_class_configurator, self.model_metadata(), self._module_ops)
            state, qmap = self._load_bundle_state(self._weights, self._qmap)
            print(f"[INT8] Requantizing module shell on {target}...", flush=True)
            requantize(model, state, qmap, device=target)
            return model.eval()

    class ReleaseAfterPromptEncoder:
        """Wrap PromptEncoder and force temporary CUDA model allocations out after encoding.

        The INT8 Quanto state dict remains cached in system RAM by QuantoBundleBuilder.
        This only releases the temporary requantized Gemma CUDA module / allocator cache
        before LTX computes the DiffVAE memory budget.
        """
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, *args, **kwargs):
            try:
                return self._inner(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                # PromptEncoder's build/dispose context has ended at this point. Collect
                # Python refs first, then release cached CUDA blocks even when the global
                # allocator strategy is DEFER.
                gc.collect()
                if torch.cuda.is_available():
                    before_free, total = torch.cuda.mem_get_info()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    after_free, _ = torch.cuda.mem_get_info()
                    print(
                        "[INT8] Prompt complete; forced Gemma CUDA cleanup. "
                        f"GPU free {before_free / (1024**3):.2f} -> "
                        f"{after_free / (1024**3):.2f} GiB / {total / (1024**3):.2f} GiB. "
                        "Quantized state dict remains cached in RAM.",
                        flush=True,
                    )

    class SageAttentionLTX:
        """Adapter from LTX's [B,T,H*D] attention ABI to SageAttention HND.

        LTX logs its attention backends inside the transformer constructor, before
        create_meta_model() applies module_ops. Therefore the stock transformer log
        can still say SDPA even when this callable is installed immediately after
        construction. The first-call message below is the runtime source of truth.
        """
        label = "SageAttention2"

        def __init__(self):
            try:
                from sageattention import sageattn
            except Exception as exc:
                raise RuntimeError(
                    "Use SageAttention is enabled, but sageattention could not be imported. "
                    "Install Triton-Windows and SageAttention in the LTX UV environment first."
                ) from exc
            self._sageattn = sageattn
            self._verified_runtime = False
            self._call_count = 0

        def __call__(self, q, k, v, heads: int):
            self._call_count += 1
            b, _, inner = q.shape
            if inner % heads != 0:
                raise RuntimeError(f"Invalid LTX attention shape: inner={inner}, heads={heads}")
            head_dim = inner // heads
            qh = q.view(b, -1, heads, head_dim).transpose(1, 2).contiguous()
            kh = k.view(b, -1, heads, head_dim).transpose(1, 2).contiguous()
            vh = v.view(b, -1, heads, head_dim).transpose(1, 2).contiguous()

            if not self._verified_runtime:
                print(
                    "[ATTENTION] VERIFIED: SageAttention2 ACTIVE on real transformer tensors | "
                    f"q={tuple(qh.shape)} k={tuple(kh.shape)} v={tuple(vh.shape)} "
                    f"dtype={qh.dtype} device={qh.device}",
                    flush=True,
                )
                self._verified_runtime = True

            out = self._sageattn(qh, kh, vh, tensor_layout="HND", is_causal=False)
            return out.transpose(1, 2).contiguous().view(b, -1, inner)

    def emit(obj: dict):
        print("@@RESULT@@" + json.dumps(obj, ensure_ascii=False), flush=True)

    def pipeline_key(job: dict) -> tuple:
        p = job["paths"]
        return (
            p["transformer"], p["text_encoder"], p["video_vae"], p["audio_vae"], p["upsampler"],
            job["offload"], job["quantization"], bool(job.get("defer_trim", True)),
            bool(job.get("use_sage_attention", False)),
            bool(job.get("use_int8_transformer", False)), job.get("int8_transformer_bundle", ""),
            bool(job.get("use_int8_text_encoder", False)), job.get("int8_text_encoder_bundle", ""),
            bool(job.get("cache_model_weights", False)),
            bool(job.get("cache_prompt_embeddings", True)),
            str(job.get("workflow", "two_phase")),
        )

    def build_pipeline(job: dict):
        p = job["paths"]
        use_int8_transformer = bool(job.get("use_int8_transformer", False))
        use_int8_text = bool(job.get("use_int8_text_encoder", False))

        model_paths = ModelPaths.from_split(
            transformer_path=p["transformer"],
            text_encoder_path=p["text_encoder"],
            video_vae_path=p["video_vae"],
            audio_vae_path=p["audio_vae"],
        )

        quant_policy = None
        q = job.get("quantization", "none")
        if use_int8_transformer:
            q = "prequantized-int8-quanto"
        elif q and q != "none":
            quant_policy = QuantizationKind(q).to_policy(checkpoint_path=p["transformer"])

        trim = AllocatorTrimStrategy.DEFER if job.get("defer_trim", True) else AllocatorTrimStrategy.TRIM
        print(
            f"[WARM] Building LTX pipeline | offload={job['offload']} | quantization={q} | "
            f"allocator={'DEFER' if job.get('defer_trim', True) else 'TRIM'}",
            flush=True,
        )

        registry = shared_registry if job.get("cache_model_weights", False) else None
        pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=p["upsampler"],
            loras=[],
            registry=registry,
            quantization=quant_policy,
            offload_mode=OffloadMode(job["offload"]),
            alloc_trim_strategy=trim,
        )
        if registry is not None:
            print("[RAM CACHE] LTX ModelRegistry enabled; model weights and reusable model shells are retained in system RAM.", flush=True)

        pipeline.audio_conditioner = AudioConditioner(
            model_paths.audio_vae(), pipeline.dtype, pipeline.device,
            registry=registry, alloc_trim_strategy=trim,
        )

        if job.get("cache_prompt_embeddings", True):
            pipeline.prompt_encoder = CachedPromptEncoder(pipeline.prompt_encoder, pipeline.device)
            print("[PROMPT CACHE] Encoded prompt cache enabled (CPU RAM).", flush=True)

        if use_int8_transformer:
            print(
                "[INT8] Using prequantized Quanto transformer. Native LTX block streaming "
                "cannot stream Quanto companion weights, so the transformer uses the standard "
                "INT8 builder while other components retain the selected offload mode.",
                flush=True,
            )
            pipeline.stage = pipeline.stage.with_builder(
                QuantoBundleBuilder(job["int8_transformer_bundle"], LTXModelConfigurator)
            )

        if use_int8_text:
            print(
                "[INT8] Using prequantized Quanto Gemma text encoder. The original packed BF16 "
                "file is retained for metadata and embeddings-processor connector weights.",
                flush=True,
            )
            gemma_sd_ops, gemma_module_ops = get_gemma_ops(p["text_encoder"])
            text_builder = QuantoBundleBuilder(
                job["int8_text_encoder_bundle"],
                GemmaTextEncoderConfigurator.with_gemma_model_path(p["text_encoder"]),
                module_ops=gemma_module_ops,
                model_sd_ops=gemma_sd_ops,
            )
            pipeline.prompt_encoder = ReleaseAfterPromptEncoder(
                PromptEncoder(
                    model_paths,
                    pipeline.dtype,
                    pipeline.device,
                    offload_mode=OffloadMode.NONE,
                    text_encoder_builder=text_builder,
                    alloc_trim_strategy=trim,
                )
            )
            print(
                "[INT8] INT8 Gemma prompt cleanup guard enabled: temporary CUDA allocations "
                "will be released after prompt encoding while quantized weights stay cached in RAM.",
                flush=True,
            )

        if job.get("use_sage_attention", False):
            sage = SageAttentionLTX()
            pipeline.stage = pipeline.stage.with_attention(sage)
            print(
                "[ATTENTION] SageAttention override installed for unmasked attention; masked attention remains SDPA.",
                flush=True,
            )
            print(
                "[ATTENTION] Runtime verification will appear on the first real SageAttention call. "
                "Ignore LTX's constructor-time SDPA label for this override.",
                flush=True,
            )
        else:
            print("[ATTENTION] Using LTX automatic attention backend.", flush=True)
        return pipeline


    def _encode_frozen_audio(pipeline, audio_path: str, num_frames: int, fps: float):
        decoded = decode_audio_from_file(audio_path, pipeline.device, 0.0, num_frames / fps)
        if decoded is None:
            raise ValueError(f"Failed to decode supplied audio: {audio_path}")
        latent = pipeline.audio_conditioner(lambda enc: vae_encode_audio(decoded, enc, None))
        shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / fps, channels=8, mel_bins=16
        )
        latent = latent[:, :, : shape.frames]
        original = Audio(
            waveform=decoded.waveform.squeeze(0),
            sampling_rate=decoded.sampling_rate,
        )
        print(f"[AUDIO] Supplied soundtrack encoded and frozen ({latent.shape[2]} latent frames).", flush=True)
        return latent, original

    def _prepare_common(pipeline, job, images):
        seed = int(job["seed"])
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        (ctx_p,) = pipeline.prompt_encoder(
            [job["prompt"]],
            enhance_first_prompt=bool(job.get("enhance_prompt", False)),
            enhance_prompt_image=images[0][0] if images else None,
        )
        return generator, noiser, ctx_p.video_encoding, ctx_p.audio_encoding

    def _run_distilled_one_phase(pipeline, job, images):
        """Distilled one-pass route using the same native stage as DistilledPipeline.

        This intentionally does not use TI2VidOneStagePipeline because that stock
        class is documented for the full/non-distilled checkpoint. We keep the
        distilled SimpleDenoiser + distilled sigma schedule and simply run it at
        target resolution, with no latent upsampler/refine pass.
        """
        width, height = int(job["width"]), int(job["height"])
        frames, fps = int(job["frames"]), float(job["fps"])
        if width % 32 or height % 32:
            raise ValueError("One-phase LTX requires width and height divisible by 32.")
        generator, noiser, vctx, actx = _prepare_common(pipeline, job, images)
        scale_factors = tiling_scale_factors_for_vae(pipeline.video_decoder.checkpoint_path)
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=pipeline.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps),
            diffvae_optimization=pipeline.video_decoder.diffvae_optimization,
            device=pipeline.device,
        )
        cond = pipeline.image_conditioner(lambda enc: combined_image_conditionings(
            images=images, height=height, width=width, video_encoder=enc,
            dtype=pipeline.dtype, device=pipeline.device,
        ))
        audio_path = str(job.get("audio_path") or "").strip()
        frozen_latent = original_audio = None
        if audio_path:
            frozen_latent, original_audio = _encode_frozen_audio(pipeline, audio_path, frames, fps)
        audio_spec = ModalitySpec(context=actx)
        if frozen_latent is not None:
            audio_spec = ModalitySpec(context=actx, frozen=True, noise_scale=0.0, initial_latent=frozen_latent)
        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
        stage_kwargs = pipeline._stage_1_sampler_kwargs(int(job["seed"]))
        video_state, audio_state = pipeline.stage(
            denoiser=SimpleDenoiser(vctx, actx), sigmas=sigmas, noiser=noiser,
            width=width, height=height, frames=frames, fps=fps,
            video=ModalitySpec(context=vctx, conditionings=cond),
            audio=audio_spec, **stage_kwargs,
        )
        video = pipeline.video_decoder(video_state.latent, tiling, generator)
        audio = original_audio if original_audio is not None else pipeline.audio_decoder(audio_state.latent)
        print("[WORKFLOW] One-phase distilled pass complete; stage 2/upscaler not used.", flush=True)
        return video, audio, frames, tiling

    def _run_distilled_two_phase_audio(pipeline, job, images):
        """Distilled two-stage pipeline with supplied audio frozen in both stages."""
        width, height = int(job["width"]), int(job["height"])
        frames, fps = int(job["frames"]), float(job["fps"])
        if width % 64 or height % 64:
            raise ValueError("Two-phase LTX requires width and height divisible by 64.")
        generator, noiser, vctx, actx = _prepare_common(pipeline, job, images)
        scale_factors = tiling_scale_factors_for_vae(pipeline.video_decoder.checkpoint_path)
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=pipeline.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps),
            diffvae_optimization=pipeline.video_decoder.diffvae_optimization,
            device=pipeline.device,
        )
        frozen_audio, original_audio = _encode_frozen_audio(
            pipeline, str(job["audio_path"]), frames, fps
        )
        s1w, s1h = width // 2, height // 2
        cond1 = pipeline.image_conditioner(lambda enc: combined_image_conditionings(
            images=images, height=s1h, width=s1w, video_encoder=enc,
            dtype=pipeline.dtype, device=pipeline.device,
        ))
        sig1 = DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
        video_state, _ = pipeline.stage(
            denoiser=SimpleDenoiser(vctx, actx), sigmas=sig1, noiser=noiser,
            width=s1w, height=s1h, frames=frames, fps=fps,
            video=ModalitySpec(context=vctx, conditionings=cond1),
            audio=ModalitySpec(context=actx, frozen=True, noise_scale=0.0, initial_latent=frozen_audio),
            **pipeline._stage_1_sampler_kwargs(int(job["seed"])),
        )
        upscaled = pipeline.upsampler(video_state.latent[:1])
        cond2 = pipeline.image_conditioner(lambda enc: combined_image_conditionings(
            images=images, height=height, width=width, video_encoder=enc,
            dtype=pipeline.dtype, device=pipeline.device,
        ))
        sig2 = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
        video_state, _ = pipeline.stage(
            denoiser=SimpleDenoiser(vctx, actx), sigmas=sig2, noiser=noiser,
            width=width, height=height, frames=frames, fps=fps,
            video=ModalitySpec(context=vctx, conditionings=cond2, noise_scale=sig2[0].item(), initial_latent=upscaled),
            audio=ModalitySpec(context=actx, frozen=True, noise_scale=0.0, initial_latent=frozen_audio),
        )
        video = pipeline.video_decoder(video_state.latent, tiling, generator)
        print("[AUDIO] Original supplied soundtrack preserved in output.", flush=True)
        return video, original_audio, frames, tiling

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            job = json.loads(raw)
            cmd = job.get("cmd")
            if cmd == "quit":
                print("[WARM] Quit requested.", flush=True)
                break
            if cmd == "clear_caches":
                shared_registry.clear()
                prompt_embedding_cache.clear()
                cache.clear()
                gc.collect()
                print("[RAM CACHE] Model and prompt caches cleared.", flush=True)
                continue
            if cmd not in ("generate", "preload"):
                continue

            if not bool(job.get("cache_model_weights", False)):
                shared_registry.clear()
                print("[RAM CACHE] ModelRegistry weight/model cache disabled and cleared.", flush=True)
            if not bool(job.get("cache_prompt_embeddings", True)):
                prompt_embedding_cache.clear()
                print("[PROMPT CACHE] Prompt embedding cache disabled and cleared.", flush=True)

            key = pipeline_key(job)
            if key not in cache:
                cache.clear()
                cache[key] = build_pipeline(job)
            else:
                print("[WARM] Reusing persistent LTX pipeline.", flush=True)

            pipeline = cache[key]
            if cmd == "preload":
                try:
                    if job.get("cache_model_weights", False):
                        _preload_pipeline(pipeline, job)
                    elif job.get("cache_prompt_embeddings", True) and job.get("prompt"):
                        print("[PROMPT CACHE] Pre-encoding current prompt (model-state RAM cache disabled)...", flush=True)
                        pipeline.prompt_encoder(
                            [job["prompt"]],
                            enhance_first_prompt=bool(job.get("enhance_prompt", False)),
                            enhance_prompt_image=None,
                        )
                    print("@@PRELOAD_DONE@@", flush=True)
                except Exception as exc:
                    # On Windows, a failed safetensors mmap/preload can leave
                    # cached storage invalid. Discard it before any real job.
                    _reset_shared_registry(f"preload failed: {type(exc).__name__}")
                    prompt_embedding_cache.clear()
                    cache.clear()
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print(f"[RAM CACHE] Preload aborted safely: {type(exc).__name__}: {exc}", flush=True)
                    print("@@PRELOAD_FAILED@@", flush=True)
                continue

            images = [
                ImageConditioningInput(
                    path=item["path"],
                    frame_idx=int(item.get("frame_idx", 0)),
                    strength=float(item.get("strength", 1.0)),
                    crf=None,
                )
                for item in job.get("images", [])
            ]

            print(
                f"[WARM] Generate {job['width']}x{job['height']} | {job['frames']} frames | "
                f"{job['fps']} fps | seed {job['seed']}",
                flush=True,
            )

            with torch.inference_mode():
                workflow = str(job.get("workflow", "two_phase"))
                has_audio = bool(str(job.get("audio_path") or "").strip())
                if workflow == "one_phase":
                    print("[WORKFLOW] 1 phase (distilled, target-resolution pass)", flush=True)
                    video, audio, num_frames, tiling_config = _run_distilled_one_phase(pipeline, job, images)
                elif has_audio:
                    print("[WORKFLOW] 2 phases + supplied soundtrack conditioning", flush=True)
                    video, audio, num_frames, tiling_config = _run_distilled_two_phase_audio(pipeline, job, images)
                else:
                    print("[WORKFLOW] 2 phases (native DistilledPipeline)", flush=True)
                    video, audio, num_frames, tiling_config = pipeline(
                        prompt=job["prompt"],
                        seed=int(job["seed"]),
                        height=int(job["height"]),
                        width=int(job["width"]),
                        num_frames=int(job["frames"]),
                        frame_rate=float(job["fps"]),
                        images=images,
                        enhance_prompt=bool(job.get("enhance_prompt", False)),
                        tiling_config=AUTO_TILING,
                    )

                out_path = Path(job["output"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                encode_video(
                    video=video,
                    fps=float(job["fps"]),
                    audio=audio,
                    output_path=str(out_path),
                    video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
                )

            emit({"ok": True, "output": job["output"]})
        except Exception as exc:
            traceback.print_exc()
            emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# --------------------------- GUI bootstrap ---------------------------------

def ensure_pyside6():
    try:
        import PySide6  # noqa: F401
        return
    except Exception:
        pass
    print("[SETUP] PySide6 is missing; installing it into the current environment...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-warn-script-location", "PySide6"
    ])


def run_gui(parent=None, embedded=False):
    ensure_pyside6()

    from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QUrl, Signal, QTimer
    from PySide6.QtGui import QDesktopServices, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
        QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
        QMessageBox, QPushButton, QPlainTextEdit, QScrollArea, QSpinBox, QDoubleSpinBox,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget, QProgressBar
    )

    class JsonSettings:
        """Small QSettings-like JSON store kept inside FrameVision."""
        def __init__(self, path: Path):
            self.path = Path(path)
            self.data = {}
            try:
                if self.path.is_file():
                    obj = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        self.data = obj
            except Exception:
                self.data = {}

        def value(self, key, default=None):
            return self.data.get(key, default)

        def setValue(self, key, value):
            self.data[str(key)] = value

        def sync(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    class WheelGuardMixin:
        def wheelEvent(self, event):
            # Never allow mouse-wheel scrolling to edit a setting.
            # Ignoring the event lets the surrounding QScrollArea scroll instead.
            event.ignore()

    class SafeSpinBox(WheelGuardMixin, QSpinBox):
        pass

    class SafeDoubleSpinBox(WheelGuardMixin, QDoubleSpinBox):
        pass

    class SafeComboBox(WheelGuardMixin, QComboBox):
        pass

    class ClickableImage(QLabel):
        clicked = Signal()
        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton and self.pixmap() is not None:
                self.clicked.emit()
            super().mousePressEvent(event)

    class ImagePreviewDialog(QDialog):
        def __init__(self, image_path: str, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Image preview")
            self.resize(1000, 760)
            root = QVBoxLayout(self)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(image_path)
            if not pix.isNull():
                label.setPixmap(pix)
            scroll.setWidget(label)
            root.addWidget(scroll)

    class LTX25Window(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.settings = JsonSettings(DEFAULT_SETTINGS_PATH)
            self.process: QProcess | None = None
            self.install_process: QProcess | None = None
            self._one_shot_worker = False
            self._loading_settings = True
            self._settings_timer = QTimer(self)
            self._settings_timer.setSingleShot(True)
            self._settings_timer.setInterval(2000)
            self._settings_timer.timeout.connect(self._save_settings)
            self.image_path = str(self.settings.value("image_path", ""))
            self.audio_path = str(self.settings.value("audio_path", ""))
            self.last_result = str(self.settings.value("last_result", ""))

            DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = DEFAULT_LOGS_DIR / f"ltx25_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
            self._log_file.write(f"LTX 2.5 helper log started {datetime.now().isoformat()}\n")
            self._log_file.flush()

            self.setWindowTitle("LTX")
            self.resize(980, 820)
            self.setMinimumSize(760, 620)

            self._build_ui()
            self._load_settings()
            self._loading_settings = False
            self._connect_setting_autosave()
            self._apply_theme()
            self._update_image_preview()
            self._update_audio_label()
            self._refresh_install_button()
            self._log(f"[LOG] Session log: {self.log_path}")

        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(10, 10, 10, 10)
            outer.setSpacing(8)

            self.tabs = QTabWidget()
            outer.addWidget(self.tabs, 1)

            self.tabs.addTab(self._make_generation_tab(), "Generation")
            self.tabs.addTab(self._make_settings_tab(), "Settings")

            footer = QFrame()
            fl = QHBoxLayout(footer)
            fl.setContentsMargins(0, 6, 0, 0)
            self.generate_btn = QPushButton("Generate")
            self.generate_btn.setMinimumHeight(42)
            self.generate_btn.clicked.connect(self.generate)
            self.open_output_btn = QPushButton("Open output folder")
            self.open_output_btn.setMinimumHeight(42)
            self.open_output_btn.clicked.connect(self.open_output_folder)
            self.open_last_btn = QPushButton("Open last result")
            self.open_last_btn.setMinimumHeight(42)
            self.open_last_btn.clicked.connect(self.open_last_result)
            self.install_btn = QPushButton("Install LTX 2.5")
            self.install_btn.setMinimumHeight(42)
            self.install_btn.clicked.connect(self.install_ltx25)
            self.install_progress = QProgressBar()
            self.install_progress.setRange(0, 0)
            self.install_progress.setTextVisible(False)
            self.install_progress.setFixedWidth(130)
            self.install_progress.setVisible(False)

            fl.addWidget(self.generate_btn, 2)
            fl.addWidget(self.install_btn, 1)
            fl.addWidget(self.install_progress, 0)
            fl.addWidget(self.open_output_btn, 1)
            fl.addWidget(self.open_last_btn, 1)
            outer.addWidget(footer, 0)

        def _scroll_container(self, content: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(content)
            return scroll

        def _make_generation_tab(self):
            content = QWidget()
            lay = QVBoxLayout(content)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(12)

            basic = QGroupBox("Generation")
            form = QFormLayout(basic)

            self.workflow_combo = SafeComboBox()
            self.workflow_combo.addItems(["2 phases", "1 phase"])
            self.workflow_combo.currentIndexChanged.connect(self._workflow_changed)
            form.addRow("Workflow", self.workflow_combo)

            self.mode_combo = SafeComboBox()
            self.mode_combo.addItems(["Text to Video", "Image to Video"])
            self.mode_combo.currentIndexChanged.connect(self._toggle_image_group)
            form.addRow("Mode", self.mode_combo)

            self.prompt_edit = QTextEdit()
            self.prompt_edit.setMinimumHeight(150)
            form.addRow("Prompt", self.prompt_edit)

            dims = QWidget()
            dg = QGridLayout(dims)
            dg.setContentsMargins(0, 0, 0, 0)

            self.width_spin = SafeSpinBox()
            self.width_spin.setRange(64, 2048)
            self.width_spin.setSingleStep(64)
            self.width_spin.setValue(832)

            self.height_spin = SafeSpinBox()
            self.height_spin.setRange(64, 2048)
            self.height_spin.setSingleStep(64)
            self.height_spin.setValue(512)

            self.frames_spin = SafeSpinBox()
            self.frames_spin.setRange(9, 2001)
            self.frames_spin.setSingleStep(8)
            self.frames_spin.setValue(121)

            self.fps_spin = SafeDoubleSpinBox()
            self.fps_spin.setRange(1.0, 120.0)
            self.fps_spin.setDecimals(2)
            self.fps_spin.setValue(24.0)

            self.seed_spin = SafeSpinBox()
            self.seed_spin.setRange(-1, 2147483647)
            self.seed_spin.setValue(-1)
            self.seed_spin.setSpecialValueText("Random")

            dg.addWidget(QLabel("Width"), 0, 0)
            dg.addWidget(self.width_spin, 0, 1)
            dg.addWidget(QLabel("Height"), 0, 2)
            dg.addWidget(self.height_spin, 0, 3)
            dg.addWidget(QLabel("Frames"), 1, 0)
            dg.addWidget(self.frames_spin, 1, 1)
            dg.addWidget(QLabel("FPS"), 1, 2)
            dg.addWidget(self.fps_spin, 1, 3)
            dg.addWidget(QLabel("Seed"), 2, 0)
            dg.addWidget(self.seed_spin, 2, 1)
            form.addRow(dims)
            lay.addWidget(basic)

            self.image_group = QGroupBox("Image")
            ig = QHBoxLayout(self.image_group)

            self.image_preview = ClickableImage()
            self.image_preview.setFixedSize(220, 150)
            self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_preview.setFrameShape(QFrame.Shape.StyledPanel)
            self.image_preview.setText("No image")
            self.image_preview.setToolTip("Click the thumbnail to view the full image")
            self.image_preview.clicked.connect(self.show_full_image)

            image_side = QVBoxLayout()
            self.choose_image_btn = QPushButton("Choose image")
            self.choose_image_btn.clicked.connect(self.choose_image)
            self.clear_image_btn = QPushButton("Clear image")
            self.clear_image_btn.clicked.connect(self.clear_image)
            self.image_strength = SafeDoubleSpinBox()
            self.image_strength.setRange(0.0, 2.0)
            self.image_strength.setSingleStep(0.05)
            self.image_strength.setDecimals(2)
            self.image_strength.setValue(1.0)
            image_side.addWidget(self.choose_image_btn)
            image_side.addWidget(self.clear_image_btn)
            image_side.addWidget(QLabel("Conditioning strength"))
            image_side.addWidget(self.image_strength)
            image_side.addStretch(1)

            ig.addWidget(self.image_preview)
            ig.addLayout(image_side, 1)
            lay.addWidget(self.image_group)

            self.audio_group = QGroupBox("Optional soundtrack conditioning")
            ag = QHBoxLayout(self.audio_group)
            self.audio_label = QLabel("No soundtrack selected")
            self.audio_label.setWordWrap(True)
            self.choose_audio_btn = QPushButton("Choose audio")
            self.choose_audio_btn.clicked.connect(self.choose_audio)
            self.clear_audio_btn = QPushButton("Clear audio")
            self.clear_audio_btn.clicked.connect(self.clear_audio)
            self.play_audio_btn = QPushButton("Play audio")
            self.play_audio_btn.clicked.connect(self.play_audio)
            ag.addWidget(self.audio_label, 1)
            ag.addWidget(self.choose_audio_btn)
            ag.addWidget(self.clear_audio_btn)
            ag.addWidget(self.play_audio_btn)
            lay.addWidget(self.audio_group)

            lay.addStretch(1)
            return self._scroll_container(content)

        def _path_row(self, target: QLineEdit, folder=False):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(target, 1)
            b = QPushButton("Browse")
            if folder:
                b.clicked.connect(lambda: self._browse_folder(target))
            else:
                b.clicked.connect(lambda: self._browse_file(target))
            l.addWidget(b)
            return w

        def _make_settings_tab(self):
            content = QWidget()
            lay = QVBoxLayout(content)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(12)

            paths = QGroupBox("Folders and model files")
            form = QFormLayout(paths)

            self.runtime_edit = QLineEdit()
            self.models_edit = QLineEdit()
            self.output_edit = QLineEdit()
            self.transformer_edit = QLineEdit()
            self.text_encoder_edit = QLineEdit()
            self.video_vae_edit = QLineEdit()
            self.audio_vae_edit = QLineEdit()
            self.upsampler_edit = QLineEdit()

            form.addRow("LTX runtime folder", self._path_row(self.runtime_edit, folder=True))
            form.addRow("LTX model folder", self._path_row(self.models_edit, folder=True))
            form.addRow("Output folder", self._path_row(self.output_edit, folder=True))
            form.addRow("Transformer", self._path_row(self.transformer_edit))
            form.addRow("Text encoder", self._path_row(self.text_encoder_edit))
            form.addRow("Video VAE", self._path_row(self.video_vae_edit))
            form.addRow("Audio VAE", self._path_row(self.audio_vae_edit))
            form.addRow("Spatial upscaler", self._path_row(self.upsampler_edit))
            lay.addWidget(paths)

            int8_group = QGroupBox("INT8 Quanto bundles")


            int8_group.setVisible(False)
            int8_form = QFormLayout(int8_group)

            self.use_int8_transformer_cb = QCheckBox("Use INT8 transformer bundle")
            self.use_int8_transformer_cb.setChecked(False)
            self.use_int8_transformer_cb.setVisible(False)
            self.int8_transformer_edit = QLineEdit()
            self.int8_transformer_edit.setEnabled(False)
            int8_form.addRow("", self.use_int8_transformer_cb)
            int8_form.addRow("Transformer INT8 folder", self._path_row(self.int8_transformer_edit, folder=True))

            self.use_int8_text_cb = QCheckBox("Use INT8 text encoder bundle")
            self.use_int8_text_cb.setChecked(False)
            self.use_int8_text_cb.setVisible(False)
            self.int8_text_edit = QLineEdit()
            self.int8_text_edit.setEnabled(False)
            int8_form.addRow("", self.use_int8_text_cb)
            int8_form.addRow("Text encoder INT8 folder", self._path_row(self.int8_text_edit, folder=True))

            int8_note = QLabel(
                "INT8 bundle weights are loaded with Quanto. The original BF16 paths above remain "
                "configured because LTX still uses their metadata and small connector/projection weights."
            )
            int8_note.setWordWrap(True)
            int8_form.addRow(int8_note)
            lay.addWidget(int8_group)

            perf = QGroupBox("Memory and performance")
            pf = QFormLayout(perf)

            self.offload_combo = SafeComboBox()
            self.offload_combo.addItems(["cpu", "none", "disk"])
            pf.addRow("Offload", self.offload_combo)

            self.quant_combo = SafeComboBox()
            self.quant_combo.addItems(["fp8-cast", "none"])
            pf.addRow("Transformer quantization", self.quant_combo)

            self.use_int8_transformer_cb.toggled.connect(self._update_int8_controls)
            self.use_int8_text_cb.toggled.connect(self._update_int8_controls)

            self.batch_spin = SafeSpinBox()
            self.batch_spin.setRange(1, 4)
            self.batch_spin.setValue(1)
            pf.addRow("Max batch size", self.batch_spin)

            self.sage_attention_cb = QCheckBox("Use SageAttention")
            self.sage_attention_cb.setChecked(False)
            self.sage_attention_cb.setToolTip(
                "Uses installed SageAttention for LTX unmasked transformer attention. "
                "Masked attention remains on LTX's stock SDPA backend. LTX prints its constructor-time "
                "SDPA label before module overrides are applied, so verify activation with the "
                "'VERIFIED: SageAttention2 ACTIVE' runtime line."
            )
            pf.addRow("", self.sage_attention_cb)

            self.use_framevision_queue_cb = QCheckBox("Use FrameVision queue")
            self.use_framevision_queue_cb.setChecked(False)
            self.use_framevision_queue_cb.setToolTip(
                "When enabled, Generate adds the LTX 2.5 job to FrameVision jobs/pending "
                "instead of running it immediately in this GUI."
            )
            pf.addRow("", self.use_framevision_queue_cb)

            self.keep_warm_cb = QCheckBox("Keep pipeline warm between generations")
            self.keep_warm_cb.setChecked(True)
            self.keep_warm_cb.setToolTip(
                "Keeps one persistent Python worker and its built LTX pipeline object alive between clips. "
                "This is a real process/pipeline reuse switch; it does not guarantee all weights remain in VRAM."
            )
            pf.addRow("", self.keep_warm_cb)



            self.cache_prompt_cb = QCheckBox("Cache prompt embeddings")
            self.cache_prompt_cb.setChecked(True)
            self.cache_prompt_cb.setToolTip(
                "Keeps encoded prompt contexts in CPU RAM. Reusing the exact same prompt can skip Gemma prompt encoding."
            )
            pf.addRow("", self.cache_prompt_cb)

            self.defer_trim_cb = QCheckBox("Keep CUDA allocator cache warm")
            self.defer_trim_cb.setChecked(True)
            self.defer_trim_cb.setToolTip(
                "Uses the repo's AllocatorTrimStrategy.DEFER so LTX does not call its normal CUDA cleanup after each model context. "
                "This keeps allocator blocks reusable, not model weights permanently resident."
            )
            pf.addRow("", self.defer_trim_cb)


            lay.addWidget(perf)

            logs = QGroupBox("Logs")
            ll = QVBoxLayout(logs)
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setMinimumHeight(260)
            ll.addWidget(self.log_view)
            br = QHBoxLayout()
            clear_btn = QPushButton("Clear log")
            clear_btn.clicked.connect(self.log_view.clear)
            unload_btn = QPushButton("Unload warm worker")
            unload_btn.clicked.connect(self.unload_worker)
            br.addWidget(clear_btn)
            br.addWidget(unload_btn)
            br.addStretch(1)
            ll.addLayout(br)
            lay.addWidget(logs)

            lay.addStretch(1)
            return self._scroll_container(content)

        def _apply_theme(self):
            self.setStyleSheet("""
                QWidget {
                    background: #0d1117;
                    color: #d8e6f3;
                    font-size: 10pt;
                }
                QTabWidget::pane {
                    border: 1px solid #263848;
                    border-radius: 5px;
                }
                QTabBar::tab {
                    background: #111923;
                    color: #a9bfd2;
                    border: 1px solid #263848;
                    padding: 9px 22px;
                }
                QTabBar::tab:selected {
                    background: #153145;
                    color: #66e3ff;
                }
                QGroupBox {
                    border: 1px solid #263848;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 6px;
                    color: #66e3ff;
                }
                QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background: #080c11;
                    border: 1px solid #2d4658;
                    border-radius: 4px;
                    padding: 6px;
                    selection-background-color: #1d6680;
                }
                QPushButton {
                    background: #142532;
                    border: 1px solid #2d5970;
                    border-radius: 5px;
                    padding: 7px 12px;
                }
                QPushButton:hover {
                    background: #1b3547;
                    border-color: #66e3ff;
                }
                QPushButton:disabled {
                    color: #61707c;
                    background: #10161c;
                    border-color: #26313b;
                }
                QScrollBar:vertical {
                    background: #0a0f14;
                    width: 14px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background: #315064;
                    min-height: 28px;
                    border-radius: 6px;
                }
            """)

        def _defaults(self):
            root = DEFAULT_ROOT
            runtime = DEFAULT_REPO
            models = DEFAULT_MODELS
            return {
                "runtime": str(runtime),
                "models": str(models),
                "output": str(DEFAULT_OUTPUT),
                "transformer": str(models / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
                "text_encoder": str(models / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
                "video_vae": str(models / "vae" / "ltx-2.5-video-vae-bf16.safetensors"),
                "audio_vae": str(models / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"),
                "upsampler": str(models / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
                "int8_transformer": str(models / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-int8-quanto"),
                "int8_text": str(models / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-int8-quanto"),
            }

        @staticmethod
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("1", "true", "yes", "on")

        def _load_settings(self):
            d = self._defaults()
            self.runtime_edit.setText(str(self.settings.value("runtime", d["runtime"])))
            self.models_edit.setText(str(self.settings.value("models", d["models"])))
            self.output_edit.setText(str(self.settings.value("output", d["output"])))
            self.transformer_edit.setText(str(self.settings.value("transformer", d["transformer"])))
            self.text_encoder_edit.setText(str(self.settings.value("text_encoder", d["text_encoder"])))
            self.video_vae_edit.setText(str(self.settings.value("video_vae", d["video_vae"])))
            self.audio_vae_edit.setText(str(self.settings.value("audio_vae", d["audio_vae"])))
            self.upsampler_edit.setText(str(self.settings.value("upsampler", d["upsampler"])))
            self.int8_transformer_edit.setText(str(self.settings.value("int8_transformer", d["int8_transformer"])))
            self.int8_text_edit.setText(str(self.settings.value("int8_text", d["int8_text"])))
            self.use_int8_transformer_cb.setChecked(False)
            self.use_int8_text_cb.setChecked(False)

            self.prompt_edit.setPlainText(str(self.settings.value("prompt", DEFAULT_PROMPT)))
            self.width_spin.setValue(int(self.settings.value("width", 832)))
            self.height_spin.setValue(int(self.settings.value("height", 512)))
            self.frames_spin.setValue(int(self.settings.value("frames", 121)))
            self.fps_spin.setValue(float(self.settings.value("fps", 24.0)))
            self.seed_spin.setValue(int(self.settings.value("seed", -1)))
            self.mode_combo.setCurrentIndex(int(self.settings.value("mode", 0)))
            self.workflow_combo.setCurrentIndex(int(self.settings.value("workflow", 0)))
            self.image_strength.setValue(float(self.settings.value("image_strength", 1.0)))

            self.offload_combo.setCurrentText(str(self.settings.value("offload", "cpu")))
            self.quant_combo.setCurrentText(str(self.settings.value("quant", "fp8-cast")))
            self.batch_spin.setValue(int(self.settings.value("batch", 1)))
            self.sage_attention_cb.setChecked(self._to_bool(self.settings.value("use_sage_attention", False)))
            self.use_framevision_queue_cb.setChecked(self._to_bool(self.settings.value("use_framevision_queue", False)))
            self.keep_warm_cb.setChecked(self._to_bool(self.settings.value("keep_warm", True)))
            self.cache_prompt_cb.setChecked(self._to_bool(self.settings.value("cache_prompt_embeddings", True)))
            self.defer_trim_cb.setChecked(self._to_bool(self.settings.value("defer_trim", True)))
            self._toggle_image_group()
            self._update_audio_label()
            self._workflow_changed()
            self._update_int8_controls()

        def _save_settings(self):
            vals = {
                "runtime": self.runtime_edit.text(),
                "models": self.models_edit.text(),
                "output": self.output_edit.text(),
                "transformer": self.transformer_edit.text(),
                "text_encoder": self.text_encoder_edit.text(),
                "video_vae": self.video_vae_edit.text(),
                "audio_vae": self.audio_vae_edit.text(),
                "upsampler": self.upsampler_edit.text(),
                "int8_transformer": self.int8_transformer_edit.text(),
                "int8_text": self.int8_text_edit.text(),
                "use_int8_transformer": False,
                "use_int8_text_encoder": False,
                "prompt": self.prompt_edit.toPlainText(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "frames": self.frames_spin.value(),
                "fps": self.fps_spin.value(),
                "seed": self.seed_spin.value(),
                "mode": self.mode_combo.currentIndex(),
                "workflow": self.workflow_combo.currentIndex(),
                "audio_path": self.audio_path,
                "image_strength": self.image_strength.value(),
                "offload": self.offload_combo.currentText(),
                "quant": self.quant_combo.currentText(),
                "batch": self.batch_spin.value(),
                "use_sage_attention": self.sage_attention_cb.isChecked(),
                "use_framevision_queue": self.use_framevision_queue_cb.isChecked(),
                "keep_warm": self.keep_warm_cb.isChecked(),
                "cache_prompt_embeddings": self.cache_prompt_cb.isChecked(),
                "defer_trim": self.defer_trim_cb.isChecked(),
                "image_path": self.image_path,
                "last_result": self.last_result,
            }
            for k, v in vals.items():
                self.settings.setValue(k, v)
            self.settings.sync()

        def _schedule_settings_save(self, *args):
            if self._loading_settings:
                return
            self._settings_timer.start(2000)

        def _connect_setting_autosave(self):
            line_edits = [
                self.runtime_edit, self.models_edit, self.output_edit, self.transformer_edit,
                self.text_encoder_edit, self.video_vae_edit, self.audio_vae_edit, self.upsampler_edit,
                self.int8_transformer_edit, self.int8_text_edit,
            ]
            for w in line_edits:
                w.textChanged.connect(self._schedule_settings_save)
            self.prompt_edit.textChanged.connect(self._schedule_settings_save)
            for w in [self.width_spin, self.height_spin, self.frames_spin, self.fps_spin, self.seed_spin, self.image_strength, self.batch_spin]:
                w.valueChanged.connect(self._schedule_settings_save)
            for w in [self.mode_combo, self.workflow_combo, self.offload_combo, self.quant_combo]:
                w.currentIndexChanged.connect(self._schedule_settings_save)
            for w in [self.sage_attention_cb, self.use_framevision_queue_cb, self.keep_warm_cb, self.cache_prompt_cb, self.defer_trim_cb]:
                w.toggled.connect(self._schedule_settings_save)

        def _update_int8_controls(self):
            use_int8 = False
            if use_int8:
                self.quant_combo.setCurrentText("none")
            self.quant_combo.setEnabled(not use_int8)
            self.quant_combo.setToolTip(
                "Runtime quantization for the BF16 transformer."
                if not use_int8
                else "Disabled because the transformer bundle is already INT8 Quanto."
            )

        def _toggle_image_group(self):
            self.image_group.setVisible(self.mode_combo.currentText() == "Image to Video")

        def _workflow_changed(self):
            one = self.workflow_combo.currentText() == "1 phase"
            self.upsampler_edit.setEnabled(not one)
            self._schedule_settings_save()

        def choose_audio(self):
            p, _ = QFileDialog.getOpenFileName(
                self, "Choose soundtrack", "",
                "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus);;All files (*.*)"
            )
            if p:
                self.audio_path = p
                self._update_audio_label()
                self._schedule_settings_save()

        def clear_audio(self):
            self.audio_path = ""
            self._update_audio_label()
            self._schedule_settings_save()

        def play_audio(self):
            if self.audio_path and Path(self.audio_path).is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.audio_path))
            else:
                QMessageBox.information(self, "Soundtrack", "No soundtrack is selected.")

        def _update_audio_label(self):
            if self.audio_path and Path(self.audio_path).is_file():
                self.audio_label.setText(Path(self.audio_path).name)
                self.audio_label.setToolTip(self.audio_path)
                self.play_audio_btn.setEnabled(True)
            else:
                self.audio_label.setText("No soundtrack selected")
                self.audio_label.setToolTip("")
                self.play_audio_btn.setEnabled(False)

        def choose_image(self):
            p, _ = QFileDialog.getOpenFileName(
                self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
            )
            if p:
                self.image_path = p
                self._update_image_preview()
                self._schedule_settings_save()

        def clear_image(self):
            self.image_path = ""
            self._update_image_preview()
            self._schedule_settings_save()

        def _update_image_preview(self):
            if not self.image_path or not Path(self.image_path).is_file():
                self.image_preview.setPixmap(QPixmap())
                self.image_preview.setText("No image")
                return
            pix = QPixmap(self.image_path)
            if pix.isNull():
                self.image_preview.setPixmap(QPixmap())
                self.image_preview.setText("Cannot preview")
                return
            thumb = pix.scaled(
                self.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_preview.setText("")
            self.image_preview.setPixmap(thumb)

        def show_full_image(self):
            if self.image_path and Path(self.image_path).is_file():
                dlg = ImagePreviewDialog(self.image_path, self)
                dlg.exec()

        def _browse_folder(self, target: QLineEdit):
            p = QFileDialog.getExistingDirectory(self, "Choose folder", target.text() or str(DEFAULT_ROOT))
            if p:
                target.setText(p)
                self._schedule_settings_save()

        def _browse_file(self, target: QLineEdit):
            p, _ = QFileDialog.getOpenFileName(
                self, "Choose file", target.text(), "Safetensors (*.safetensors);;All files (*.*)"
            )
            if p:
                target.setText(p)
                self._schedule_settings_save()

        def _validate(self):
            width = self.width_spin.value()
            height = self.height_spin.value()
            frames = self.frames_spin.value()
            if self.workflow_combo.currentText() == "1 phase":
                if width % 32 or height % 32:
                    return False, "One-phase LTX requires width and height to be multiples of 32."
            else:
                if width % 64 or height % 64:
                    return False, "Two-phase LTX requires width and height to be multiples of 64."
            if frames % 8 != 1:
                return False, "LTX frame count must satisfy frames % 8 == 1 (for example 121, 241, 481)."
            if not self.prompt_edit.toPlainText().strip():
                return False, "Enter a prompt."
            if self.mode_combo.currentText() == "Image to Video":
                if not self.image_path or not Path(self.image_path).is_file():
                    return False, "Choose an input image."
            if self.audio_path and not Path(self.audio_path).is_file():
                return False, f"Soundtrack file does not exist:\n{self.audio_path}"
            checks = [
                ("Runtime folder", self.runtime_edit.text()),
                ("Transformer", self.transformer_edit.text()),
                ("Text encoder", self.text_encoder_edit.text()),
                ("Video VAE", self.video_vae_edit.text()),
                ("Audio VAE", self.audio_vae_edit.text()),
                *(([("Spatial upscaler", self.upsampler_edit.text())]) if self.workflow_combo.currentText() != "1 phase" else []),
            ]
            for name, p in checks:
                if not Path(p).exists():
                    return False, f"{name} path does not exist:\n{p}"

            bundles = []
            if False:
                bundles.append(("Transformer INT8", self.int8_transformer_edit.text()))
            if False:
                bundles.append(("Text encoder INT8", self.int8_text_edit.text()))
            for name, folder in bundles:
                root = Path(folder)
                missing = [x for x in ("model.safetensors", "quantization_map.json") if not (root / x).is_file()]
                if missing:
                    return False, f"{name} bundle is incomplete:\n{folder}\nMissing: {', '.join(missing)}"
            return True, ""

        def _build_job(self):
            seed = self.seed_spin.value()
            if seed < 0:
                seed = random.randint(0, 2147483647)

            out_dir = Path(self.output_edit.text())
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"ltx25_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed_{seed}.mp4"

            images = []
            if self.mode_combo.currentText() == "Image to Video":
                images.append({
                    "path": self.image_path,
                    "frame_idx": 0,
                    "strength": self.image_strength.value(),
                })

            return {
                "cmd": "generate",
                "prompt": self.prompt_edit.toPlainText().strip(),
                "workflow": "one_phase" if self.workflow_combo.currentText() == "1 phase" else "two_phase",
                "audio_path": self.audio_path,
                "seed": seed,
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "frames": self.frames_spin.value(),
                "fps": self.fps_spin.value(),
                "output": str(out_path),
                "images": images,
                "paths": {
                    "transformer": self.transformer_edit.text(),
                    "text_encoder": self.text_encoder_edit.text(),
                    "video_vae": self.video_vae_edit.text(),
                    "audio_vae": self.audio_vae_edit.text(),
                    "upsampler": self.upsampler_edit.text(),
                },
                "offload": self.offload_combo.currentText(),
                "quantization": self.quant_combo.currentText(),
                "max_batch_size": self.batch_spin.value(),
                "use_sage_attention": self.sage_attention_cb.isChecked(),
                "use_int8_transformer": False,
                "int8_transformer_bundle": self.int8_transformer_edit.text(),
                "use_int8_text_encoder": False,
                "int8_text_encoder_bundle": self.int8_text_edit.text(),
                "enhance_prompt": False,
                "defer_trim": self.defer_trim_cb.isChecked(),
                "cache_prompt_embeddings": self.cache_prompt_cb.isChecked(),
            }

        def _enqueue_framevision_job(self, job: dict) -> bool:
            """Queue this exact LTX job through FrameVision's normal jobs/pending adapter."""
            try:
                try:
                    from helpers.queue_adapter import enqueue_tool_job as enqueue_job
                except Exception:
                    from queue_adapter import enqueue_tool_job as enqueue_job

                payload_dir = APP_ROOT / "temp" / "ltx25_queue_payloads"
                payload_dir.mkdir(parents=True, exist_ok=True)
                payload_path = payload_dir / (
                    f"ltx25_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
                    f"seed_{int(job.get('seed', 0))}.json"
                )
                payload_path.write_text(
                    json.dumps(job, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                python_exe = str(self._python_exe())
                helper_path = str(Path(__file__).resolve())
                cmd = [
                    python_exe,
                    helper_path,
                    "--queue-job",
                    str(payload_path),
                ]

                # queue_adapter uses input mainly for display/metadata. Prefer a
                # real image/audio source if one exists, otherwise use the helper.
                input_path = ""
                if job.get("images"):
                    input_path = str(job["images"][0].get("path") or "")
                if not input_path and job.get("audio_path"):
                    input_path = str(job.get("audio_path") or "")
                if not input_path:
                    input_path = helper_path

                out_path = Path(job["output"])
                args = {
                    "cmd": cmd,
                    "outfile": str(out_path),
                    "cwd": str(self.runtime_edit.text()),
                    "engine": "ltx25",
                    "scan_dir": str(out_path.parent),
                    "scan_ext": out_path.suffix or ".mp4",
                }
                jid = enqueue_job(
                    "ltx25_generate",
                    input_path,
                    str(out_path.parent),
                    args,
                    priority=600,
                )
                self._log(f"[QUEUE] Added LTX 2.5 job to FrameVision queue: {jid}")
                self._log(f"[QUEUE] Output: {out_path}")
                return True
            except Exception as exc:
                self._log(f"[QUEUE ERROR] {type(exc).__name__}: {exc}")
                QMessageBox.warning(self, "Queue error", str(exc))
                return False

        def generate(self):
            ok, msg = self._validate()
            if not ok:
                QMessageBox.warning(self, "Cannot generate", msg)
                self._log("[ERROR] " + msg)
                return

            self._save_settings()
            job = self._build_job()

            if self.use_framevision_queue_cb.isChecked():
                self._log("=" * 72)
                self._log(
                    f"[JOB] {self.mode_combo.currentText()} | {self.workflow_combo.currentText()} | "
                    f"{job['width']}x{job['height']} | {job['frames']} frames | "
                    f"{job['fps']} fps | seed {job['seed']}"
                )
                if job.get("audio_path"):
                    self._log(f"[AUDIO] Conditioning soundtrack: {job['audio_path']}")
                if self._enqueue_framevision_job(job):
                    return
                # Queueing failed: do not silently run locally.
                return

            self.generate_btn.setEnabled(False)
            self._log("=" * 72)
            self._log(f"[JOB] {self.mode_combo.currentText()} | {self.workflow_combo.currentText()} | {job['width']}x{job['height']} | {job['frames']} frames | {job['fps']} fps | seed {job['seed']}")
            if job.get("audio_path"):
                self._log(f"[AUDIO] Conditioning soundtrack: {job['audio_path']}")
            self._log(f"[JOB] Output: {job['output']}")
            self._log(
                f"[OPTIONS] warm={'ON' if self.keep_warm_cb.isChecked() else 'OFF'} | "
                f"SageAttention={'ON' if job.get('use_sage_attention') else 'OFF'} | "
                f"RAM model cache={'ON' if job.get('cache_model_weights') else 'OFF'} | "
                f"prompt cache={'ON' if job.get('cache_prompt_embeddings') else 'OFF'} | "
                f"CUDA allocator={'DEFER' if job.get('defer_trim') else 'TRIM'}"
            )
            if job.get("use_int8_transformer"):
                self._log(f"[INT8] Transformer bundle: {job['int8_transformer_bundle']}")
            if job.get("use_int8_text_encoder"):
                self._log(f"[INT8] Text encoder bundle: {job['int8_text_encoder_bundle']}")

            needs_python_worker = (
                job.get("use_sage_attention", False)
                or job.get("use_int8_transformer", False)
                or job.get("use_int8_text_encoder", False)
                or job.get("cache_prompt_embeddings", False)
            )
            if self.keep_warm_cb.isChecked() or needs_python_worker:
                # SageAttention and Quanto bundles require the Python API worker.
                self._one_shot_worker = not self.keep_warm_cb.isChecked()
                self._generate_warm(job)
            else:
                self._one_shot_worker = False
                self._generate_cli(job)


        def _python_exe(self):
            candidate = DEFAULT_ENV / "Scripts" / "python.exe"
            return str(candidate if candidate.exists() else Path(sys.executable))

        def _apply_ltx_process_environment(self, process: QProcess):
            env = QProcessEnvironment.systemEnvironment()
            cache = DEFAULT_CACHE
            (cache / "uv").mkdir(parents=True, exist_ok=True)
            (cache / "huggingface").mkdir(parents=True, exist_ok=True)
            (cache / "torch_extensions").mkdir(parents=True, exist_ok=True)
            (cache / "triton").mkdir(parents=True, exist_ok=True)
            env.insert("UV_CACHE_DIR", str(cache / "uv"))
            env.insert("HF_HOME", str(cache / "huggingface"))
            env.insert("HF_HUB_CACHE", str(cache / "huggingface" / "hub"))
            env.insert("HF_XET_CACHE", str(cache / "huggingface" / "xet"))
            env.insert("TORCH_EXTENSIONS_DIR", str(cache / "torch_extensions"))
            env.insert("TRITON_CACHE_DIR", str(cache / "triton"))
            env.insert("XDG_CACHE_HOME", str(cache))
            env.insert("PIP_CACHE_DIR", str(cache / "pip"))
            process.setProcessEnvironment(env)

        def _generate_warm(self, job: dict):
            if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
                self._start_worker()
            if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
                self.generate_btn.setEnabled(True)
                return
            payload = json.dumps(job, ensure_ascii=False) + "\n"
            self.process.write(payload.encode("utf-8"))

        def _start_worker(self):
            self.process = QProcess(self)
            self.process.setWorkingDirectory(self.runtime_edit.text())
            self.process.setProgram(self._python_exe())
            self.process.setArguments([str(Path(__file__).resolve()), "--worker"])
            self._apply_ltx_process_environment(self.process)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._read_worker_output)
            self.process.finished.connect(self._worker_finished)
            self._log("[WARM] Starting persistent LTX worker...")
            self.process.start()
            if not self.process.waitForStarted(10000):
                self._log("[ERROR] Could not start warm worker.")
                self.process = None

        def _read_worker_output(self):
            if not self.process:
                return
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            for line in text.replace("\r", "\n").splitlines():
                if not line.strip():
                    continue
                if "Building transformer with attention backends --" in line:
                    continue
                self._log(line)
                if line == "@@PRELOAD_DONE@@":
                    self._log("[RAM CACHE] GUI-start preload finished.")
                    continue
                if line == "@@PRELOAD_FAILED@@":
                    self._log("[RAM CACHE] Preload failed; its registry and pipeline were discarded so the next generation starts clean.")
                    continue
                if line.startswith("@@RESULT@@"):
                    try:
                        obj = json.loads(line[len("@@RESULT@@"):])
                    except Exception:
                        obj = {}
                    if obj.get("ok"):
                        self.last_result = obj.get("output", "")
                        self.settings.setValue("last_result", self.last_result)
                        self._log(f"[DONE] Saved: {self.last_result}")
                    else:
                        self._log("[ERROR] " + obj.get("error", "Generation failed"))
                    self.generate_btn.setEnabled(True)
                    if self._one_shot_worker and self.process:
                        self._one_shot_worker = False
                        self.process.write(b'{"cmd":"quit"}\n')

        def _worker_finished(self, code, status):
            self._log(f"[WARM] Worker exited with code {code}.")
            self.generate_btn.setEnabled(True)
            self.process = None
            self._one_shot_worker = False

        def unload_worker(self):
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self._log("[WARM] Unloading persistent worker; RAM model/prompt caches will be released too...")
                self.process.write(b'{"cmd":"quit"}\n')
                self.process.waitForFinished(5000)
                if self.process.state() != QProcess.ProcessState.NotRunning:
                    self.process.kill()
            self.process = None

        def _generate_cli(self, job: dict):
            args = [
                "-m", "ltx_pipelines.distilled",
                "--transformer-path", job["paths"]["transformer"],
                "--text-encoder-path", job["paths"]["text_encoder"],
                "--video-vae-path", job["paths"]["video_vae"],
                "--audio-vae-path", job["paths"]["audio_vae"],
                "--spatial-upsampler-path", job["paths"]["upsampler"],
                "--offload", job["offload"],
                "--max-batch-size", str(job["max_batch_size"]),
                "--height", str(job["height"]),
                "--width", str(job["width"]),
                "--num-frames", str(job["frames"]),
                "--frame-rate", str(job["fps"]),
                "--seed", str(job["seed"]),
                "--prompt", job["prompt"],
                "--output-path", job["output"],
            ]
            if job["quantization"] != "none":
                args += ["--quantization", job["quantization"]]
            if job["enhance_prompt"]:
                args += ["--enhance-prompt"]
            for image in job["images"]:
                args += ["--image", image["path"], str(image["frame_idx"]), str(image["strength"])]

            self.process = QProcess(self)
            self.process.setWorkingDirectory(self.runtime_edit.text())
            self.process.setProgram(self._python_exe())
            self.process.setArguments(args)
            self._apply_ltx_process_environment(self.process)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._read_cli_output)
            self.process.finished.connect(lambda code, status: self._cli_finished(code, job["output"]))
            self.process.start()

        def _read_cli_output(self):
            if not self.process:
                return
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._log_raw(text)

        def _cli_finished(self, code, output_path):
            if code == 0 and Path(output_path).exists():
                self.last_result = output_path
                self.settings.setValue("last_result", output_path)
                self._log(f"[DONE] Saved: {output_path}")
            else:
                self._log(f"[ERROR] Generation process exited with code {code}.")
            self.process = None
            self.generate_btn.setEnabled(True)

        def _log(self, text: str):
            self.log_view.appendPlainText(text)
            try:
                self._log_file.write(str(text) + "\n")
                self._log_file.flush()
            except Exception:
                pass
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _log_raw(self, text: str):
            for line in text.replace("\r", "\n").splitlines():
                if not line.strip():
                    continue
                if "Building transformer with attention backends --" in line:
                    continue
                self._log(line)

        def _install_complete(self):
            env_ok = (DEFAULT_ENV / "Scripts" / "python.exe").is_file()
            required = [
                DEFAULT_MODELS / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                DEFAULT_MODELS / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                DEFAULT_MODELS / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
                DEFAULT_MODELS / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
                DEFAULT_MODELS / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            ]
            models_ok = DEFAULT_MODELS.is_dir() and all(p.is_file() for p in required)
            return env_ok and models_ok

        def _refresh_install_button(self):
            complete = self._install_complete()
            self.install_btn.setVisible(not complete)
            self.install_btn.setEnabled(not complete and self.install_process is None)
            if complete:
                self.install_progress.setVisible(False)

        def install_ltx25(self):
            if self.install_process is not None:
                return
            if not DEFAULT_INSTALLER.is_file():
                QMessageBox.warning(self, "Install LTX 2.5", f"Installer was not found:\n{DEFAULT_INSTALLER}")
                return
            self.tabs.setCurrentIndex(1)
            self._log("=" * 72)
            self._log(f"[INSTALL] Starting LTX 2.5 installer: {DEFAULT_INSTALLER}")
            self.install_btn.setEnabled(False)
            self.install_progress.setVisible(True)
            proc = QProcess(self)
            self.install_process = proc
            proc.setWorkingDirectory(str(DEFAULT_INSTALLER.parent))
            proc.setProgram("cmd.exe")
            proc.setArguments(["/d", "/c", str(DEFAULT_INSTALLER), "--no-pause"])
            self._apply_ltx_process_environment(proc)
            proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            proc.readyReadStandardOutput.connect(self._read_installer_output)
            proc.finished.connect(self._installer_finished)
            proc.start()
            if not proc.waitForStarted(10000):
                self._log("[INSTALL] ERROR: installer process could not be started.")
                self.install_process = None
                self.install_progress.setVisible(False)
                self._refresh_install_button()

        def _read_installer_output(self):
            if not self.install_process:
                return
            data = bytes(self.install_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._log_raw(data)

        def _installer_finished(self, code, status):
            self._log(f"[INSTALL] Installer exited with code {code}.")
            self.install_process = None
            self.install_progress.setVisible(False)
            if code == 0:
                # Re-apply canonical root-relative paths in case this is a fresh install.
                d = self._defaults()
                self.runtime_edit.setText(d["runtime"])
                self.models_edit.setText(d["models"])
                self.output_edit.setText(d["output"])
                self.transformer_edit.setText(d["transformer"])
                self.text_encoder_edit.setText(d["text_encoder"])
                self.video_vae_edit.setText(d["video_vae"])
                self.audio_vae_edit.setText(d["audio_vae"])
                self.upsampler_edit.setText(d["upsampler"])
                self._save_settings()
            self._refresh_install_button()

        def open_output_folder(self):
            path = Path(self.output_edit.text())
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

        def open_last_result(self):
            if self.last_result and Path(self.last_result).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_result))
            else:
                QMessageBox.information(self, "Last result", "No generated result is available yet.")

        def closeEvent(self, event):
            self._save_settings()
            self.unload_worker()
            if self.install_process and self.install_process.state() != QProcess.ProcessState.NotRunning:
                self.install_process.terminate()
                self.install_process.waitForFinished(2000)
                if self.install_process.state() != QProcess.ProcessState.NotRunning:
                    self.install_process.kill()
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            super().closeEvent(event)

    if embedded:
        win = LTX25Window(parent=parent)
        try:
            win.setWindowFlags(Qt.Widget)
            win.setMinimumSize(0, 0)
            win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        return win

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    win = LTX25Window()
    win.show()
    if owns_app:
        sys.exit(app.exec())
    return win


def _run_queue_job_file(job_file: str) -> None:
    """Run one queued LTX job through the same native worker code as the GUI."""
    import io
    payload_path = Path(job_file)
    job = json.loads(payload_path.read_text(encoding="utf-8"))
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(
            json.dumps(job, ensure_ascii=False) + "\n" +
            json.dumps({"cmd": "quit"}) + "\n"
        )
        run_worker()
    finally:
        sys.stdin = old_stdin
        try:
            payload_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--queue-job", default="")
    args, _ = parser.parse_known_args()
    if args.queue_job:
        _run_queue_job_file(args.queue_job)
    elif args.worker:
        run_worker()
    else:
        run_gui()


if __name__ == "__main__":
    main()
