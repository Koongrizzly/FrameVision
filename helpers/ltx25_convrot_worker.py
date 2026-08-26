from __future__ import annotations
import argparse, gc, hashlib, json, os, subprocess, sys, tempfile, traceback
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

# LTX 2.5 ConvRot uses its own current ComfyUI backend.
# Do not import the older shared /vendor Comfy backend used by MiniMax.
COMFY_ROOT = APP_ROOT / 'models' / 'ltx_2_5_convrot' / 'ComfyUI'
if not (COMFY_ROOT / 'comfy' / 'sd.py').is_file():
    raise RuntimeError(
        'LTX 2.5 current Comfy backend was not found at '
        f'{COMFY_ROOT}. Run presets/extra_env/download_ltx25_comfy_backend.bat first.'
    )

# Put the isolated ComfyUI root first so imports such as comfy.*, node_helpers
# and folder_paths all resolve from the same current backend.
sys.path.insert(0, str(COMFY_ROOT))
os.chdir(APP_ROOT)

# Preflight the binary support package required by the isolated LTX Comfy backend.
# It is installed permanently by the LTX ConvRot installer; this fallback repairs
# older installs made before comfy-aimdo was included.
try:
    import comfy_aimdo  # noqa: F401
except ModuleNotFoundError:
    uv = APP_ROOT / 'presets' / 'bin' / 'uv' / 'uv.exe'
    if not uv.is_file():
        raise RuntimeError(
            'LTX Comfy backend requires comfy-aimdo, but it is missing from '
            f'{sys.executable} and presets/bin/uv/uv.exe was not found.'
        )
    print('[CONVROT SETUP] Installing missing Comfy dependency: comfy-aimdo', flush=True)
    subprocess.check_call([
        str(uv), 'pip', 'install', '--python', sys.executable, 'comfy-aimdo'
    ])
    import comfy_aimdo  # noqa: F401

# Match the embedded Comfy pattern used by the working MiniMax backend.
_saved_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import comfy.options
comfy.options.enable_args_parsing(True)
import comfy.cli_args
import torch
import numpy as np
from PIL import Image
from scipy.io import wavfile
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_detection
# The clean ConvRot installer predates Comfy's sampler dependency on torchsde.
# Install this one small missing runtime package into the dedicated ConvRot env once, if needed.
try:
    import torchsde  # noqa: F401
except ModuleNotFoundError:
    uv = APP_ROOT / 'presets' / 'bin' / 'uv' / 'uv.exe'
    if not uv.is_file():
        raise RuntimeError('ConvRot backend needs torchsde and uv.exe was not found under presets/bin/uv')
    print('[CONVROT SETUP] Installing missing sampler dependency: torchsde', flush=True)
    subprocess.check_call([str(uv), 'pip', 'install', '--python', sys.executable, 'torchsde'])
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import folder_paths
import comfy_extras.nodes_lt as nodes_lt
from comfy_extras.nodes_lt import LTXVDualCFGGuider
from comfy_extras.nodes_custom_sampler import RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_audio import load as load_audio_file, VAEEncodeAudio
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
import node_helpers
sys.argv = _saved_argv

SIGMAS_DISTILLED = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
SIGMAS_STAGE2 = [0.909375, 0.725, 0.421875, 0.0]
NEGATIVE = 'worst quality, inconsistent motion, blurry, distorted, artifacts, cartoon, video game'

# Conservative tiled VAE decode defaults for LTX 2.5 video decode on 24 GB cards.
VIDEO_VAE_TILE_T = 8
VIDEO_VAE_TILE_X = 32
VIDEO_VAE_TILE_Y = 32
VIDEO_VAE_OVERLAP = 8
VIDEO_VAE_OVERLAP_T = 4


def _flush():
    try: comfy.model_management.unload_all_models()
    except Exception: pass
    gc.collect()
    try: comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        if torch.cuda.is_available(): torch.cuda.empty_cache()



def _load_quantized_diffusion_model(path: str):
    """
    Load a standalone Comfy quantized diffusion checkpoint without losing the
    per-layer comfy_quant markers.

    Some ComfyUI builds call convert_old_quants() only after stripping the
    model.diffusion_model. prefix. Checkpoints whose _quantization_metadata
    still names fully-qualified layers then get orphaned comfy_quant keys.
    Convert the metadata first, while the original prefix is still present,
    and pass scrubbed metadata to the stock state-dict loader so it does not
    repeat the broken conversion a second time.
    """
    sd, metadata = comfy.utils.load_torch_file(
        str(path),
        safe_load=True,
        return_metadata=True,
    )

    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    quant_meta_present = bool(
        metadata and (
            "_quantization_metadata" in metadata
            or "quantization_metadata" in metadata
        )
    )

    before_markers = sum(1 for k in sd.keys() if k.endswith(".comfy_quant"))

    # This is the important ordering fix: convert BEFORE prefix stripping.
    sd, converted_metadata = comfy.utils.convert_old_quants(
        sd,
        prefix,
        metadata=metadata,
    )

    after_markers = sum(1 for k in sd.keys() if k.endswith(".comfy_quant"))
    print(
        f"[CONVROT] Quant metadata pre-conversion | prefix={prefix!r} | "
        f"metadata={'yes' if quant_meta_present else 'no'} | "
        f"markers={before_markers}->{after_markers}",
        flush=True,
    )

    # The stock loader must not regenerate markers from the original,
    # fully-qualified metadata after it strips the prefix.
    clean_metadata = dict(converted_metadata or {})
    clean_metadata.pop("_quantization_metadata", None)
    clean_metadata.pop("quantization_metadata", None)

    try:
        model = comfy.sd.load_diffusion_model_state_dict(
            sd,
            model_options={},
            metadata=clean_metadata,
        )
    except TypeError:
        # Compatibility with Comfy builds where metadata is supplied through
        # model_options rather than as a named argument.
        model = comfy.sd.load_diffusion_model_state_dict(
            sd,
            model_options={"metadata": clean_metadata},
        )

    if model is None:
        raise RuntimeError(f"Comfy could not detect LTX diffusion model: {path}")

    return model


def _load_vae(path: str, audio=False):
    sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    if audio:
        sd = comfy.utils.state_dict_prefix_replace(sd, {'audio_vae.': 'autoencoder.', 'vocoder.': 'vocoder.'}, filter_keys=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    if hasattr(vae, 'throw_exception_if_invalid'): vae.throw_exception_if_invalid()
    return vae


def _conditioning_fingerprint(cond):
    try:
        chunks = []
        shapes = []
        for entry in cond:
            tensor = entry[0]
            shapes.append(tuple(tensor.shape))
            t = tensor.detach().float().cpu().contiguous()
            chunks.append(t.numpy().tobytes())
        digest = hashlib.sha256(b''.join(chunks)).hexdigest()[:16]
        return digest, shapes
    except Exception as exc:
        return f'error:{type(exc).__name__}', []


def _encode_conditioning(clip, prompt: str, fps: float):
    print(f'[CONVROT] Prompt received: {prompt!r}', flush=True)
    # LTX 2.5 expects the Gemma conditioning sequence padded to 1024 tokens.
    # Generic Gemma4 auto-detection defaults to a minimal sequence length, which
    # produced shapes such as (1, 10, 6144) and effectively broke prompt guidance.
    positive_tokens = clip.tokenize(prompt, min_length=1024)
    negative_tokens = clip.tokenize(NEGATIVE, min_length=1024)
    positive = clip.encode_from_tokens_scheduled(positive_tokens)
    negative = clip.encode_from_tokens_scheduled(negative_tokens)
    pos_hash, pos_shapes = _conditioning_fingerprint(positive)
    neg_hash, neg_shapes = _conditioning_fingerprint(negative)
    print(f'[CONVROT] Positive conditioning fingerprint: {pos_hash} shapes={pos_shapes}', flush=True)
    print(f'[CONVROT] Negative conditioning fingerprint: {neg_hash} shapes={neg_shapes}', flush=True)
    def add_values(cond):
        out = node_helpers.conditioning_set_values(cond, {'frame_rate': float(fps)})
        # Compatibility with LTXAV builds where the attention mask is left inside model_conds.
        fixed=[]
        for item in out:
            item=[item[0], item[1].copy()]
            mc=item[1].get('model_conds', {})
            am=mc.get('attention_mask')
            if am is not None and 'attention_mask' not in item[1]:
                item[1]['attention_mask'] = getattr(am, 'cond', am)
            fixed.append(item)
        return fixed
    return add_values(positive), add_values(negative)



def _node_value(result, index=0):
    """Return one value from either Comfy io.NodeOutput or a legacy tuple."""
    if isinstance(result, tuple):
        return result[index]
    if hasattr(result, "args"):
        return result.args[index]
    try:
        return tuple(result)[index]
    except Exception:
        if index == 0:
            return result
        raise


def _encode_soundtrack(audio_path, audio_vae, frames, fps):
    """Decode the user soundtrack, trim/pad to video duration, and VAE-encode it."""
    waveform, sample_rate = load_audio_file(str(audio_path))
    # load_audio_file returns [channels, samples]; Comfy AUDIO uses [batch, channels, samples].
    waveform = waveform.unsqueeze(0)
    wanted = max(1, int(round((float(frames) / float(fps)) * sample_rate)))
    if waveform.shape[-1] > wanted:
        waveform = waveform[..., :wanted]
    elif waveform.shape[-1] < wanted:
        pad = wanted - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    audio = {"waveform": waveform, "sample_rate": int(sample_rate)}
    with torch.inference_mode():
        encoded = VAEEncodeAudio.execute(audio_vae, audio)
    latent = _node_value(encoded, 0)
    # Freeze supplied audio: video may diffuse, audio must remain unchanged.
    latent["noise_mask"] = torch.zeros_like(latent["samples"])
    print(
        f"[CONVROT] Soundtrack encoded/frozen for AV conditioning | "
        f"{float(frames)/float(fps):.2f}s | latent={tuple(latent['samples'].shape)}",
        flush=True,
    )
    return latent



_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)


def _clone_conditioning(conditioning):
    return [[item[0], dict(item[1])] for item in conditioning]


def _load_msr_checkpoint(path):
    """Load Licon MSR LoRA + learned slot embedding using Comfy's native loader."""
    lora, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    metadata = metadata or {}
    slot_state = {}
    normal_lora = {}
    for key, value in lora.items():
        matched = False
        for prefix in _SLOT_PREFIXES:
            if key.startswith(prefix):
                slot_state[key[len(prefix):]] = value.detach().cpu()
                matched = True
                break
        if not matched:
            normal_lora[key] = value

    required = {
        "frequencies",
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    }
    missing = sorted(required.difference(slot_state))
    if missing:
        raise ValueError(
            "MSR checkpoint is missing reference slot tensors: " + ", ".join(missing)
        )
    if metadata.get("reference_token_order", "prepend") != "prepend":
        raise ValueError("Unsupported MSR reference_token_order; expected prepend.")
    if metadata.get(
        "reference_slot_time_offsets", "pic1_based_negative_time"
    ) != "pic1_based_negative_time":
        raise ValueError(
            "Unsupported MSR reference_slot_time_offsets; "
            "expected pic1_based_negative_time."
        )

    downscale = max(1, round(float(metadata.get("reference_downscale_factor", 1))))
    print(
        f"[CONVROT-MSR] Loaded MSR checkpoint | "
        f"adapter_tensors={len(normal_lora)} slot_tensors={len(slot_state)} "
        f"downscale={downscale}",
        flush=True,
    )
    return normal_lora, slot_state, metadata, downscale


def _apply_msr_lora(model, normal_lora, metadata, strength_model):
    """Apply the MSR adapter to the already-loaded quantized ConvRot model."""
    if float(strength_model) == 0.0:
        return model
    loaded_model, _ = comfy.sd.load_lora_for_models(
        model,
        None,
        normal_lora,
        float(strength_model),
        0.0,
        lora_metadata=metadata,
    )
    return loaded_model


def _slot_embedding(slot_id, state, device, dtype):
    frequencies = state["frequencies"].to(device=device, dtype=torch.float32)
    slot_value = torch.tensor(float(slot_id), device=device, dtype=torch.float32)
    scaled = slot_value / 16.0
    phases = scaled * frequencies
    features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
    weight0 = state["net.0.weight"].to(device=device, dtype=torch.float32)
    bias0 = state["net.0.bias"].to(device=device, dtype=torch.float32)
    hidden = torch.nn.functional.silu(
        torch.nn.functional.linear(features, weight0, bias0)
    )
    weight2 = state["net.2.weight"].to(device=device, dtype=torch.float32)
    bias2 = state["net.2.bias"].to(device=device, dtype=torch.float32)
    embedding = torch.nn.functional.linear(hidden, weight2, bias2)
    return embedding.to(dtype=dtype)


def _conditioning_get(conditioning, key, default=None):
    for _, values in conditioning:
        if key in values:
            return values[key]
    return default


def _append_attention_entry(conditioning, pre_filter_count, latent_shape, strength):
    existing = _conditioning_get(conditioning, "guide_attention_entries", [])
    entry = {
        "pre_filter_count": int(pre_filter_count),
        "strength": float(strength),
        "pixel_mask": None,
        "latent_shape": list(latent_shape),
    }
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": [*existing, entry]}
    )


def _resize_msr_reference(images, target_width, target_height, is_background):
    """Mirror Licon's official ComfyUI-LTX2.5-MSR resize policy."""
    if is_background:
        return comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(target_width),
            int(target_height),
            "bilinear",
            crop="center",
        ).movedim(1, -1)

    source_height, source_width = images.shape[1:3]
    if source_width == target_width and source_height == target_height:
        return images

    def aspect_family(width, height):
        ratio = float(width) / float(height)
        if ratio >= 1.25:
            return "landscape"
        if ratio <= 0.8:
            return "portrait"
        return "square"

    same_family = (
        aspect_family(source_width, source_height)
        == aspect_family(target_width, target_height)
    )
    source_is_smaller = (
        source_width <= target_width and source_height <= target_height
    )
    if same_family and not source_is_smaller:
        return comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(target_width),
            int(target_height),
            "bilinear",
            crop="center",
        ).movedim(1, -1)

    scale = min(
        float(target_width) / float(source_width),
        float(target_height) / float(source_height),
    )
    resized_width = max(1, min(int(target_width), round(source_width * scale)))
    resized_height = max(1, min(int(target_height), round(source_height * scale)))
    resized = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        resized_width,
        resized_height,
        "bilinear",
        crop="disabled",
    ).movedim(1, -1)
    canvas = torch.ones(
        (images.shape[0], int(target_height), int(target_width), images.shape[-1]),
        dtype=images.dtype,
        device=images.device,
    )
    left = (int(target_width) - resized_width) // 2
    top = (int(target_height) - resized_height) // 2
    canvas[:, top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _encode_msr_reference(
    vae,
    latent_width,
    latent_height,
    image_path,
    reference_frames,
    downscale,
    is_background,
    tile_size,
    tile_overlap,
):
    image = _load_image(image_path)
    repeated = image.repeat(int(reference_frames), 1, 1, 1)
    time_scale, width_scale, height_scale = vae.downscale_index_formula
    keep = ((repeated.shape[0] - 1) // int(time_scale)) * int(time_scale) + 1
    repeated = repeated[:keep]
    target_width = int(latent_width * width_scale / downscale)
    target_height = int(latent_height * height_scale / downscale)
    pixels = _resize_msr_reference(
        repeated, target_width, target_height, bool(is_background)
    )[..., :3]
    with torch.inference_mode():
        guide_latent = vae.encode_tiled(
            pixels,
            tile_x=int(tile_size),
            tile_y=int(tile_size),
            overlap=int(tile_overlap),
        )
    return guide_latent


def _apply_msr_guides(
    positive,
    negative,
    video_latent,
    vae,
    references,
    slot_state,
    downscale,
    strength,
    reference_frames,
    tile_size,
    tile_overlap,
):
    """
    Native Comfy/Licon 2.5 MSR path:
      independent VAE encode -> learned slot embedding -> negative time offset
      -> append_keyframe -> guide attention metadata.
    """
    positive = _clone_conditioning(positive)
    negative = _clone_conditioning(negative)
    latent_image = video_latent["samples"]
    noise_mask = nodes_lt.get_noise_mask(video_latent)

    if latent_image.ndim != 5 or latent_image.shape[1] != 128:
        raise ValueError(
            "ConvRot MSR needs video latent [B,128,F,H,W], "
            f"got {tuple(latent_image.shape)}"
        )
    if latent_image.shape[0] != 1:
        raise ValueError("ConvRot MSR currently requires batch_size=1.")

    _, _, _, latent_height, latent_width = latent_image.shape
    if latent_height % int(downscale) or latent_width % int(downscale):
        raise ValueError(
            f"Target latent grid {latent_width}x{latent_height} is not divisible "
            f"by MSR reference_downscale_factor={downscale}."
        )

    num_slots = len(references)
    if not 1 <= num_slots <= 5:
        raise ValueError(f"MSR requires 1-5 references, got {num_slots}.")

    scale_factors = vae.downscale_index_formula
    print(
        f"[CONVROT-MSR] Applying {num_slots} references | "
        f"frames_each={reference_frames} | target_latent={tuple(latent_image.shape)}",
        flush=True,
    )

    for slot_index, ref in enumerate(references):
        label = ref["label"]
        path = ref["path"]
        is_background = bool(ref.get("background", False))
        slot_id = slot_index + 1

        guide_latent = _encode_msr_reference(
            vae,
            latent_width,
            latent_height,
            path,
            reference_frames,
            downscale,
            is_background,
            tile_size,
            tile_overlap,
        )

        embedding = _slot_embedding(
            slot_id, slot_state, guide_latent.device, guide_latent.dtype
        )
        if embedding.numel() != guide_latent.shape[1]:
            raise ValueError(
                f"MSR slot embedding dimension {embedding.numel()} does not "
                f"match LTX latent channels {guide_latent.shape[1]}."
            )
        guide_latent = guide_latent + embedding.view(1, -1, 1, 1, 1)

        original_shape = list(guide_latent.shape[2:])
        guide_mask = None
        if int(downscale) > 1:
            guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(
                guide_latent, int(downscale)
            )

        frame_offset = -(num_slots - slot_index)
        positive, negative, latent_image, noise_mask = (
            nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_offset,
                latent_image,
                noise_mask,
                guide_latent,
                float(strength),
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=int(downscale),
                causal_fix=True,
            )
        )

        token_count = (
            guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        )
        positive = _append_attention_entry(
            positive, token_count, original_shape, strength
        )
        negative = _append_attention_entry(
            negative, token_count, original_shape, strength
        )
        print(
            f"[CONVROT-MSR] {label} slot={slot_id} offset={frame_offset} "
            f"latent={tuple(guide_latent.shape)}",
            flush=True,
        )

    out = dict(video_latent)
    out["samples"] = latent_image
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    return positive, negative, out


def _crop_msr_guides(positive, negative, video_samples):
    """Remove prepended MSR guide slots exactly through native LTXVCropGuides."""
    result = nodes_lt.LTXVCropGuides.execute(
        positive, negative, {"samples": video_samples}
    )
    cropped = _node_value(result, 2)
    return cropped["samples"]


def _load_latent_upsampler(path):
    """Load FrameVision's selected LTX latent x2 upscaler through current Comfy."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"LTX latent spatial upscaler not found: {p}")
    folder_paths.add_model_folder_path("latent_upscale_models", str(p.parent))
    result = LatentUpscaleModelLoader.execute(model_name=p.name)
    return _node_value(result, 0)


def _upscale_video_latent(video_latent, video_vae, upsampler_path):
    print("[CONVROT] Loading LTX x2 latent spatial upscaler...", flush=True)
    upscale_model = _load_latent_upsampler(upsampler_path)
    with torch.inference_mode():
        result = LTXVLatentUpsampler.execute(
            {"samples": video_latent},
            upscale_model,
            video_vae,
        )
    upscaled = _node_value(result, 0)
    print(
        f"[CONVROT] Latent spatial upscale complete | "
        f"{tuple(video_latent.shape)} -> {tuple(upscaled['samples'].shape)}",
        flush=True,
    )
    del upscale_model
    _flush()
    return upscaled["samples"]


def _empty_video(width, height, frames):
    z = torch.zeros((1,128,((int(frames)-1)//8)+1,int(height)//32,int(width)//32), device=comfy.model_management.intermediate_device())
    return {'samples': z, 'downscale_ratio_spacial': 32}


def _empty_audio(audio_vae, frames, fps):
    fs=audio_vae.first_stage_model
    n=fs.num_of_latents_from_frames(int(frames), float(fps))
    z=torch.zeros((1,audio_vae.latent_channels,n,fs.latent_frequency_bins), device=comfy.model_management.intermediate_device())
    return {'samples': z, 'type':'audio'}


def _load_image(path):
    im=Image.open(path).convert('RGB')
    a=np.asarray(im,dtype=np.float32)/255.0
    return torch.from_numpy(a).unsqueeze(0)


def _apply_i2v(video_latent, video_vae, image_path, strength):
    image=_load_image(image_path)
    samples=video_latent['samples'].clone()
    _, hs, ws = video_vae.downscale_index_formula
    h=samples.shape[-2]*hs; w=samples.shape[-1]*ws
    if image.shape[1] != h or image.shape[2] != w:
        image=comfy.utils.common_upscale(image.movedim(-1,1), w, h, 'bilinear', 'center').movedim(1,-1)
    t=video_vae.encode(image[...,:3])
    samples[:,:,:t.shape[2]]=t
    mask=torch.ones((samples.shape[0],1,samples.shape[2],1,1),dtype=torch.float32,device=samples.device)
    mask[:,:,:t.shape[2]]=1.0-float(strength)
    out=video_latent.copy(); out['samples']=samples; out['noise_mask']=mask
    return out


def _concat_av(video_latent, audio_latent):
    v=video_latent['samples']; a=audio_latent['samples']
    out={}; out.update(video_latent); out.update(audio_latent)
    out['samples']=comfy.nested_tensor.NestedTensor((v,a))
    vm=video_latent.get('noise_mask'); am=audio_latent.get('noise_mask')
    if vm is not None or am is not None:
        if vm is None: vm=torch.ones_like(v)
        if am is None: am=torch.ones_like(a)
        out['noise_mask']=comfy.nested_tensor.NestedTensor((vm,am))
    return out


def _sample(model, positive, negative, latent, seed, sigma_values=None):
    # Mirror the current official LTX 2.5 Comfy workflow:
    # RandomNoise -> LTXVDualCFGGuider(video_cfg=1,audio_cfg=1)
    # -> euler_ancestral -> SamplerCustomAdvanced.
    #
    # The previous direct comfy.sample.sample_custom(...) shortcut used the
    # generic CFG path and bypassed LTX's AV-aware guider. With a NestedTensor
    # containing video+audio latents that can produce deterministic but
    # effectively unconditioned output.
    sigmas = torch.tensor(
        SIGMAS_DISTILLED if sigma_values is None else sigma_values,
        dtype=torch.float32,
        device=comfy.model_management.intermediate_device(),
    )
    sampler = comfy.samplers.sampler_object('euler_ancestral')
    # These Comfy nodes expose class-level execute(...) in the backend version
    # shipped with the isolated LTX 2.5 Comfy install.
    noise = RandomNoise.execute(int(seed))[0]
    guider = LTXVDualCFGGuider.execute(
        model,
        positive,
        negative,
        1.0,
        1.0,
    )[0]
    sampled_latent = SamplerCustomAdvanced.execute(
        noise,
        guider,
        sampler,
        sigmas,
        latent,
    )[0]
    return sampled_latent['samples']


def _decode_video(latent, vae):
    # LTX 2.5 video decode is very VRAM-hungry on 24 GB cards. Decode with
    # explicit 3D tiling under inference mode instead of trying a full-frame
    # decode first. The tile values are in latent space, matching Comfy's VAE
    # decode_tiled(...) interface for 3D/video latents.
    temporal = int(latent.shape[2]) if getattr(latent, "ndim", 0) >= 3 else -1
    print(
        f"[CONVROT] VAE decode tiling | temporal={temporal} "
        f"tile_t={VIDEO_VAE_TILE_T} overlap_t={VIDEO_VAE_OVERLAP_T} "
        f"tile_xy={VIDEO_VAE_TILE_X} overlap_xy={VIDEO_VAE_OVERLAP}",
        flush=True,
    )
    with torch.inference_mode():
        images = vae.decode_tiled(
            latent,
            tile_t=VIDEO_VAE_TILE_T,
            tile_x=VIDEO_VAE_TILE_X,
            tile_y=VIDEO_VAE_TILE_Y,
            overlap=VIDEO_VAE_OVERLAP,
            overlap_t=VIDEO_VAE_OVERLAP_T,
        )
    if images.ndim==5: images=images.reshape(-1,*images.shape[-3:])
    return images.detach().cpu()


def _decode_audio(latent, vae):
    # Keep audio VAE decode inference-only for the same reason and to avoid
    # building an unnecessary autograd graph during generation.
    with torch.inference_mode():
        audio=vae.decode(latent).movedim(-1,1)
    sr=int(getattr(vae.first_stage_model,'output_sample_rate',32000))
    return audio.detach().cpu(), sr


def _find_ffmpeg():
    root=APP_ROOT/'presets'/'bin'
    direct=[root/'ffmpeg.exe',root/'ffmpeg'/'ffmpeg.exe']
    for p in direct:
        if p.is_file(): return p
    for p in root.rglob('ffmpeg.exe') if root.exists() else []:
        return p
    raise FileNotFoundError('ffmpeg.exe was not found under FrameVision/presets/bin')


def _save(images, audio, sr, out_path, fps, soundtrack_path=None):
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=_find_ffmpeg()
    arr=(images.clamp(0,1).numpy()*255.0+0.5).astype(np.uint8)
    h,w=arr.shape[1:3]
    with tempfile.TemporaryDirectory(prefix='ltx25_convrot_') as td:
        if soundtrack_path:
            # Preserve the user's original soundtrack for the final mux.
            audio_input = str(soundtrack_path)
        else:
            wave=audio.numpy()
            if wave.ndim==3: wave=wave[0]
            if wave.ndim==2 and wave.shape[0] in (1,2): wave=wave.T
            wave=np.clip(wave,-1,1).astype(np.float32)
            wav=Path(td)/'audio.wav'; wavfile.write(wav,sr,wave)
            audio_input = str(wav)

        cmd=[
            str(ffmpeg),'-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(float(fps)),'-i','pipe:0',
            '-i',audio_input,
            '-c:v','libx264','-pix_fmt','yuv420p','-crf','18',
            '-c:a','aac','-b:a','256k','-shortest',str(out)
        ]
        # Use communicate()/subprocess.run semantics instead of manually writing
        # to stdin. With -shortest, ffmpeg can intentionally finish as soon as
        # the soundtrack ends and close the raw-video pipe before every generated
        # frame has been written. A manual stdin.write then raises BrokenPipeError
        # even though ffmpeg successfully created a valid output file.
        proc = subprocess.run(cmd, input=arr.tobytes())
        code = int(proc.returncode)
        if code:
            raise RuntimeError(f'ffmpeg exited with code {code}')
        if not out.is_file() or out.stat().st_size <= 0:
            raise RuntimeError('ffmpeg returned success but no output video was created')
    return out


def generate(job):
    paths=job['paths']
    width=int(job['width'])//32*32
    height=int(job['height'])//32*32
    frames=int(job['frames'])
    fps=float(job['fps'])
    seed=int(job['seed'])
    two_phase = job.get('workflow') == 'two_phase'
    soundtrack = str(job.get('audio_path') or '').strip()
    msr_enabled = bool(job.get("msr_enabled", False))

    if two_phase and (width % 64 or height % 64):
        raise ValueError("Two-phase ConvRot requires width and height divisible by 64.")

    msr_refs = []
    msr_normal_lora = msr_slot_state = msr_metadata = None
    msr_downscale = 1
    msr_strength = float(job.get("msr_reference_strength", 1.0))
    msr_model_strength = float(job.get("msr_strength_model", 1.0))
    msr_reference_frames = int(job.get("msr_reference_frames", 33))
    msr_tile_size = int(job.get("msr_tile_size", 256))
    msr_tile_overlap = int(job.get("msr_tile_overlap", 64))

    if msr_enabled:
        if msr_reference_frames not in (25, 33):
            raise ValueError(
                f"ConvRot MSR reference_frames must be 25 or 33, got {msr_reference_frames}."
            )
        raw_refs = [str(x).strip() for x in (job.get("msr_refs") or []) if str(x).strip()]
        background = str(job.get("msr_background") or "").strip()
        for idx, path in enumerate(raw_refs[:4], 1):
            msr_refs.append({"label": f"pic{idx}", "path": path, "background": False})
        if background:
            msr_refs.append({"label": "background", "path": background, "background": True})
        if not msr_refs:
            raise ValueError("ConvRot MSR was requested but no references were supplied.")
        msr_path = str(job.get("msr_lora_path") or "").strip()
        if not msr_path or not Path(msr_path).is_file():
            raise FileNotFoundError(f"LTX 2.5 MSR checkpoint not found: {msr_path}")
        msr_normal_lora, msr_slot_state, msr_metadata, msr_downscale = (
            _load_msr_checkpoint(msr_path)
        )

    print(
        f'[CONVROT] Embedded Comfy backend | {job.get("model_type")} | '
        f'{"2 phase" if two_phase else "1 phase"} | '
        f'{width}x{height} | {frames} frames | seed {seed}'
        + (f' | MSR refs={len(msr_refs)}' if msr_enabled else ''),
        flush=True
    )

    print('[CONVROT] Loading LTX 2.5 Gemma4 text encoder + projection through Comfy auto-detection...',flush=True)
    clip=comfy.sd.load_clip(
        [str(paths['text_encoder'])],
        clip_type=None,
    )
    base_positive,base_negative=_encode_conditioning(clip,job['prompt'],fps)
    del clip; _flush()

    print('[CONVROT] Loading audio VAE...',flush=True)
    av=_load_vae(paths['audio_vae'],audio=True)
    if soundtrack:
        audio_latent = _encode_soundtrack(soundtrack, av, frames, fps)
    else:
        audio_latent = _empty_audio(av,frames,fps)
    del av; _flush()

    stage_w = width // 2 if two_phase else width
    stage_h = height // 2 if two_phase else height
    video_latent=_empty_video(stage_w,stage_h,frames)

    # Accept both the worker-native ``images`` list and the music-clip
    # bridge's singular ``image`` field.  The bridge payloads currently pass
    # a single start-image path, so without this compatibility normalization
    # the I2V branch below was silently skipped and the run became text/audio
    # to video even though the payload contained a valid image path.
    images=job.get('images') or []
    if not images:
        single_image = job.get('image')
        if isinstance(single_image, str) and single_image.strip():
            images=[{'path': single_image.strip(), 'strength': float(job.get('image_strength', 1.0))}]
        elif isinstance(single_image, dict):
            image_path = str(single_image.get('path') or single_image.get('image') or '').strip()
            if image_path:
                images=[{
                    'path': image_path,
                    'strength': float(single_image.get('strength', job.get('image_strength', 1.0))),
                }]
    if images:
        print(
            f"[CONVROT] Start-image route active | {images[0].get('path')} | "
            f"strength={float(images[0].get('strength', 1.0)):.3f}",
            flush=True,
        )
    else:
        print('[CONVROT] Start-image route inactive | no image supplied', flush=True)

    vv = None
    if images or msr_enabled:
        vv=_load_vae(paths['video_vae'])

    # Preserve ordinary first/last-frame conditioning if the existing planner
    # supplied it. MSR references are added separately through native guides.
    if images:
        print(
            f'[CONVROT] Encoding first-frame image conditioning at '
            f'{stage_w}x{stage_h}...',
            flush=True
        )
        image=images[0]
        video_latent=_apply_i2v(video_latent,vv,image['path'],image.get('strength',1.0))

    if msr_enabled:
        print('[CONVROT-MSR] Encoding Stage 1 multi-reference guides...', flush=True)
        positive, negative, video_latent = _apply_msr_guides(
            base_positive,
            base_negative,
            video_latent,
            vv,
            msr_refs,
            msr_slot_state,
            msr_downscale,
            msr_strength,
            msr_reference_frames,
            msr_tile_size,
            msr_tile_overlap,
        )
    else:
        positive, negative = base_positive, base_negative

    if vv is not None:
        del vv
        _flush()

    latent=_concat_av(video_latent,audio_latent)

    print('[CONVROT] Loading quantized LTX 2.5 diffusion model through comfy.sd...',flush=True)
    model=_load_quantized_diffusion_model(str(paths['transformer']))
    if msr_enabled:
        print('[CONVROT-MSR] Applying MSR LoRA to quantized ConvRot transformer...', flush=True)
        model=_apply_msr_lora(model, msr_normal_lora, msr_metadata, msr_model_strength)

    print(
        f'[CONVROT] Stage 1 sampling | {stage_w}x{stage_h} | '
        f'{len(SIGMAS_DISTILLED)-1} steps...',
        flush=True
    )
    sampled=_sample(model,positive,negative,latent,seed,SIGMAS_DISTILLED)
    streams=sampled.unbind()
    video_z,audio_z=streams[0],streams[1]
    if msr_enabled:
        video_z = _crop_msr_guides(positive, negative, video_z)
        print(
            f'[CONVROT-MSR] Stage 1 guide slots cropped | video_latent={tuple(video_z.shape)}',
            flush=True,
        )

    del model,latent,sampled,video_latent
    _flush()

    if two_phase:
        print('[CONVROT] Stage 1 complete; preparing x2 latent spatial upscale...', flush=True)
        vv=_load_vae(paths['video_vae'])
        video_z=_upscale_video_latent(video_z,vv,paths['upsampler'])

        stage2_video = {'samples': video_z, 'downscale_ratio_spacial': 32}
        if images:
            image=images[0]
            print(
                '[CONVROT] Re-applying original first-frame conditioning '
                'for Stage 2...',
                flush=True,
            )
            stage2_video=_apply_i2v(
                stage2_video,
                vv,
                image['path'],
                image.get('strength',1.0),
            )
            video_z=stage2_video['samples']

        if msr_enabled:
            print('[CONVROT-MSR] Re-encoding Stage 2 multi-reference guides...', flush=True)
            positive2, negative2, stage2_video = _apply_msr_guides(
                base_positive,
                base_negative,
                stage2_video,
                vv,
                msr_refs,
                msr_slot_state,
                msr_downscale,
                msr_strength,
                msr_reference_frames,
                msr_tile_size,
                msr_tile_overlap,
            )
        else:
            positive2, negative2 = base_positive, base_negative

        del vv
        _flush()

        stage2_audio = audio_latent
        stage2_latent = _concat_av(stage2_video, stage2_audio)

        print('[CONVROT] Reloading quantized LTX model for Stage 2 refine...', flush=True)
        model=_load_quantized_diffusion_model(str(paths['transformer']))
        if msr_enabled:
            print('[CONVROT-MSR] Re-applying MSR LoRA for Stage 2 refine...', flush=True)
            model=_apply_msr_lora(model, msr_normal_lora, msr_metadata, msr_model_strength)

        print(
            f'[CONVROT] Stage 2 refine | {width}x{height} | '
            f'{len(SIGMAS_STAGE2)-1} steps...',
            flush=True
        )
        sampled2=_sample(model,positive2,negative2,stage2_latent,seed,SIGMAS_STAGE2)
        streams2=sampled2.unbind()
        video_z,audio_z=streams2[0],streams2[1]
        if msr_enabled:
            video_z = _crop_msr_guides(positive2, negative2, video_z)
            print(
                f'[CONVROT-MSR] Stage 2 guide slots cropped | video_latent={tuple(video_z.shape)}',
                flush=True,
            )

        del model,stage2_latent,sampled2,stage2_video
        _flush()
        print('[CONVROT] Two-phase latent-upscale workflow complete.', flush=True)

    # The MSR adapter/checkpoint is no longer needed once diffusion finishes.
    if msr_enabled:
        del msr_normal_lora, msr_slot_state, msr_metadata
        _flush()

    print('[CONVROT] Decoding video...',flush=True)
    vv=_load_vae(paths['video_vae'])
    decoded=_decode_video(video_z,vv)
    del vv,video_z
    _flush()

    if soundtrack:
        print('[CONVROT] Preserving original supplied soundtrack in output...',flush=True)
        audio = torch.zeros((1,2,1), dtype=torch.float32)
        sr = 48000
    else:
        print('[CONVROT] Decoding generated audio...',flush=True)
        av=_load_vae(paths['audio_vae'],audio=True)
        audio,sr=_decode_audio(audio_z,av)
        del av
        _flush()

    del audio_z, audio_latent
    _flush()

    out=_save(
        decoded,audio,sr,job['output'],fps,
        soundtrack_path=soundtrack if soundtrack else None,
    )
    return {'ok':True,'output':str(out),'seed':seed}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--job',required=True); ap.add_argument('--msr-enabled', action='store_true'); ns=ap.parse_args()
    p=Path(ns.job)
    try:
        job=json.loads(p.read_text(encoding='utf-8'))
        result=generate(job)
    except Exception as e:
        traceback.print_exc(); result={'ok':False,'error':f'{type(e).__name__}: {e}'}
    finally:
        try:p.unlink(missing_ok=True)
        except Exception:pass
    print('@@RESULT@@'+json.dumps(result,ensure_ascii=False),flush=True)
    raise SystemExit(0 if result.get('ok') else 1)

if __name__=='__main__': main()
