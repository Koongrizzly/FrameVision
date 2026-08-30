from __future__ import annotations
import argparse, os, sys, tempfile, subprocess, shutil, time
from pathlib import Path

# This launcher lives in helpers/, while runtime/ remains in the standalone root.
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from runtime.paths import ROOT
from runtime.validate_models import validate
from runtime import vram_manager as _vram_manager_module

_EXPECTED_VRAM_SIGNATURE = "V11_5_EXTREME_DIFFUSION_SAFE_20260826A"

def _verify_vram_runtime():
    actual = getattr(_vram_manager_module, "VRAM_MANAGER_SIGNATURE", None)
    if actual != _EXPECTED_VRAM_SIGNATURE:
        path = getattr(_vram_manager_module, "__file__", "unknown")
        raise RuntimeError(
            "VRAM Manager patch mismatch: the launcher is V11.5 but the loaded "
            f"runtime/vram_manager.py is not. Loaded: {path} | signature={actual!r}. "
            "Re-extract the patch into the MiniMax app root so both helpers/ and runtime/ are replaced."
        )
    return str(getattr(_vram_manager_module, "__file__", "unknown"))
from runtime.ffmpeg_tools import ensure_ffmpeg_tools, tool_path


def main():
    ap = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA W4A8 standalone generator")
    ap.add_argument("--prompt", required=True); ap.add_argument("--width", type=int, default=832); ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=362); ap.add_argument("--experimental-long-duration", action="store_true", help="Allow H3 native-grid research durations beyond the normal 719-frame range, up to 2385 frames"); ap.add_argument("--steps", type=int, default=15); ap.add_argument("--cfg", type=float, default=1.0); ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--shift", type=float, default=12.0); ap.add_argument("--audio-shift", type=float, default=3.0); ap.add_argument("--sampler", default="euler"); ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--ref-image-size", choices=["match", "max"], default="match"); ap.add_argument("--ref-image", action="append", default=[]); ap.add_argument("--ref-video", action="append", default=[]); ap.add_argument("--ref-audio", action="append", default=[]); ap.add_argument("--ref-audio-subject", action="append", type=int, default=[], help="Optional H3 Subject number (1-9) for each standalone --ref-audio; 0 keeps it generic."); ap.add_argument("--output")
    ap.add_argument("--fl2va-checkpoint"); ap.add_argument("--ref2va-checkpoint"); ap.add_argument("--text-encoder"); ap.add_argument("--video-vae"); ap.add_argument("--audio-vae")
    ap.add_argument("--lora", action="append", default=[]); ap.add_argument("--lora-strength", action="append", type=float, default=[])
    ap.add_argument("--extended-logging", action="store_true")
    ap.add_argument("--tile-debugging", action="store_true")
    ap.add_argument("--video-vae-tile-size", type=int, default=256)
    ap.add_argument("--video-vae-tile-overlap", type=int, default=64)
    ap.add_argument("--vram-manager", action="store_true")
    ap.add_argument("--vram-manager-auto", action="store_true", help="Automatically bypass VRAM Lab when native sampling is estimated to fit the detected GPU")
    ap.add_argument("--spectrum", action="store_true", help="Enable experimental bundled MiniMax H3 Spectrum feature forecasting")
    ap.add_argument("--sage-attention", action="store_true", help="Use SageAttention for the sampling worker")
    ap.add_argument("--disable-comfy-kitchen", action="store_true", help="Disable Comfy Kitchen quantized W4A8 / ConvRot acceleration for worker processes")
    ap.add_argument("--vram-residency-engine", choices=["static", "dynamic"], default="static")
    ap.add_argument("--vram-runtime-free-gb", type=float, default=0.5)
    ap.add_argument("--vram-text-headroom-gb", type=float, default=1.0)
    ap.add_argument("--vram-diffusion-headroom-gb", type=float, default=1.0)
    ap.add_argument("--vram-offload-chunk-mb", type=int, default=512)
    ap.add_argument("--vram-max-resident-weights-gb", type=float, default=0.0)
    ap.add_argument("--vram-block-check-interval", type=int, default=1)
    ap.add_argument("--vram-async-streams", type=int, default=2)
    ap.add_argument("--vram-video-vae-reserve-gb", type=float, default=2.0)
    ap.add_argument("--vram-audio-vae-reserve-gb", type=float, default=1.0)
    ap.add_argument("--vram-residency-fill", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vram-residency-target-free-gb", type=float, default=0.5)
    ap.add_argument("--vram-residency-warmup-blocks", type=int, default=2)
    ap.add_argument("--vram-residency-refill-interval", type=int, default=1)
    ap.add_argument("--vram-keep-text-encoder", action="store_true")
    ns = ap.parse_args()
    # Worker processes import the vendored Comfy stack afresh, so this environment
    # switch cleanly controls whether comfy.quant_ops may activate Comfy Kitchen.
    if ns.disable_comfy_kitchen:
        os.environ["H3_DISABLE_COMFY_KITCHEN"] = "1"
        print("Comfy Kitchen W4A8 acceleration: disabled", flush=True)
    else:
        os.environ.pop("H3_DISABLE_COMFY_KITCHEN", None)
        try:
            import comfy_kitchen as _ck
            _backends = _ck.list_backends()
            _cuda = _backends.get("cuda", {}) if isinstance(_backends, dict) else {}
            _state = "CUDA" if _cuda.get("available") and not _cuda.get("disabled") else "available fallback"
            print(f"Comfy Kitchen W4A8 acceleration: enabled ({_state})", flush=True)
        except Exception as _exc:
            print(f"Comfy Kitchen W4A8 acceleration: requested but unavailable ({type(_exc).__name__}: {_exc})", flush=True)
    if ns.vram_manager:
        runtime_path = _verify_vram_runtime()
        print(f"[VRAM-MGR] V11.5 runtime verified: {runtime_path}", flush=True)
    if ns.video_vae_tile_size < 128: print("ERROR: video VAE tile size must be at least 128 px"); return 2
    if ns.video_vae_tile_overlap < 0 or ns.video_vae_tile_overlap >= ns.video_vae_tile_size: print("ERROR: video VAE tile overlap must be >= 0 and smaller than tile size"); return 2
    if len(ns.lora) != len(ns.lora_strength): print("ERROR: each --lora needs one matching --lora-strength"); return 2
    if len(ns.lora) > 3: print("ERROR: maximum 3 LoRAs are supported"); return 2
    for lp in ns.lora:
        if not Path(lp).is_file(): print(f"ERROR: LoRA not found: {lp}"); return 2
    max_frames = 2385 if ns.experimental_long_duration else 719
    if ns.frames > max_frames:
        mode_name = "experimental long-duration" if ns.experimental_long_duration else "normal"
        print(f"ERROR: maximum allowed requested frame count in {mode_name} mode is {max_frames} ({max_frames / 24.0:.3f} seconds at 24 FPS)"); return 2
    if ns.width % 32 or ns.height % 32: print("ERROR: width and height must be valid preset values divisible by 32"); return 2
    if not (ns.ref_image or ns.ref_video or ns.ref_audio): print("ERROR: Ref2VA needs at least one reference"); return 2

    errors, info, diff, ref, te, vv, av = validate(
        require_vae=True, mode="ref2va",
        fl2va_path=ns.fl2va_checkpoint, ref2va_path=ns.ref2va_checkpoint,
        text_encoder_path=ns.text_encoder, video_vae_path=ns.video_vae, audio_vae_path=ns.audio_vae,
    )
    if errors:
        print("Model validation failed before generation:")
        for e in errors: print(" -", e)
        return 2
    if ns.seed < 0: ns.seed = int.from_bytes(os.urandom(4), "little") % 99_000_000
    else: ns.seed = int(ns.seed) % 99_000_000
    out = Path(ns.output) if ns.output else ROOT / "output" / f"minimax_h3_ref2va_int4_{time.strftime('%Y%m%d_%H%M%S')}.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"MiniMax-H3 Ref2VA W4A8 | {ns.width}x{ns.height} | requested frames={ns.frames} | steps={ns.steps} | CFG={ns.cfg:g} | shift={ns.shift:g} | audio shift={ns.audio_shift:g} | seed={ns.seed}", flush=True)
    print(f"Using checkpoint: {Path(ref).name}", flush=True)
    print(f"Using text encoder: {Path(te).name}", flush=True)
    print(f"Using video VAE: {Path(vv).name}", flush=True)
    print(f"Using audio VAE: {Path(av).name}", flush=True)
    py = sys.executable

    use_vram_manager = bool(ns.vram_manager)
    stage_plan = None
    managed_sample_stages = ["reference", "text", "diffusion"] if use_vram_manager else []
    if ns.vram_manager_auto:
        from runtime.vram_auto import decide_vram_stages, print_vram_stage_decisions
        # Ref2VA encodes one reference at a time; reference videos can span up to
        # the requested generation frame count and dominate this stage's transient.
        reference_frames = ns.frames if ns.ref_video else 1
        stage_plan = decide_vram_stages(
            ns.width, ns.height, ns.frames, "ref2va",
            text_encoder_path=te, video_vae_path=vv, audio_vae_path=av,
            reference_needed=True, reference_frames=reference_frames,
            reference_audio=bool(ns.ref_audio or ns.ref_video),
            ref_image_count=len(ns.ref_image),
            ref_video_count=len(ns.ref_video),
            ref_audio_count=len(ns.ref_audio),
            ref_image_size=ns.ref_image_size,
        )
        print_vram_stage_decisions(stage_plan, ns.width, ns.height, ns.frames)
        managed_sample_stages = [x for x in ("reference", "text", "diffusion") if stage_plan["stages"][x]["use_manager"]]
        use_vram_manager = bool(managed_sample_stages)


    with tempfile.TemporaryDirectory(prefix="h3_ref_isolated_", dir=str(ROOT / "output")) as td:
        td = Path(td); lat = td / "latents.pt"; frames_dir = td / "frames"; wav = td / "audio.wav"
        cmd = [py, "-m", "runtime.sample_worker_ref", "--diffusion", str(ref), "--text-encoder", str(te), "--video-vae", str(vv), "--audio-vae", str(av), "--prompt", ns.prompt, "--width", str(ns.width), "--height", str(ns.height), "--frames", str(ns.frames), "--steps", str(ns.steps), "--cfg", str(ns.cfg), "--seed", str(ns.seed), "--shift", str(ns.shift), "--audio-shift", str(ns.audio_shift), "--sampler", ns.sampler, "--scheduler", ns.scheduler, "--ref-image-size", ns.ref_image_size, "--out", str(lat)]
        if ns.experimental_long_duration: cmd += ["--experimental-long-duration"]
        if ns.spectrum: cmd += ["--spectrum"]
        sample_env = os.environ.copy()
        comfy_args = []
        if ns.sage_attention:
            comfy_args += ["--use-sage-attention"]
            print("SageAttention: enabled for sampling worker", flush=True)
        else:
            print("SageAttention: disabled", flush=True)
        if use_vram_manager:
            cmd += ["--vram-manager", "--vram-runtime-free-gb", str(ns.vram_runtime_free_gb), "--vram-text-headroom-gb", str(ns.vram_text_headroom_gb), "--vram-diffusion-headroom-gb", str(ns.vram_diffusion_headroom_gb), "--vram-offload-chunk-mb", str(ns.vram_offload_chunk_mb), "--vram-max-resident-weights-gb", str(ns.vram_max_resident_weights_gb), "--vram-block-check-interval", str(ns.vram_block_check_interval), "--vram-residency-target-free-gb", str(ns.vram_residency_target_free_gb), "--vram-residency-warmup-blocks", str(ns.vram_residency_warmup_blocks), "--vram-residency-refill-interval", str(ns.vram_residency_refill_interval)]
            if ns.vram_manager_auto:
                for stage in managed_sample_stages:
                    cmd += ["--vram-managed-stage", stage]
            cmd += ["--vram-residency-fill" if ns.vram_residency_fill else "--no-vram-residency-fill"]
            if ns.vram_keep_text_encoder: cmd += ["--vram-keep-text-encoder"]
            # Keep the process-wide Comfy baseline small. VRAMManager.set_stage()
            # raises EXTRA_RESERVED_VRAM only for stages selected as managed and
            # restores this baseline for native stages.
            initial_reserve = max(ns.vram_runtime_free_gb, 0.5)
            comfy_args += ["--reserve-vram", str(initial_reserve)]
            if "diffusion" in managed_sample_stages and ns.vram_residency_engine == "static":
                # Classic/estimate-based ModelPatcher. Unlike DynamicVRAM/VBAR this gives
                # Comfy a concrete resident-weight budget and avoids intentionally mapping
                # most transformer weights through Windows shared GPU memory.
                comfy_args += ["--disable-dynamic-vram"]
            elif "diffusion" in managed_sample_stages:
                comfy_args += ["--enable-dynamic-vram"]
            effective_async_streams = ns.vram_async_streams
            diffusion_plan = stage_plan["stages"]["diffusion"] if stage_plan else None
            physical_total = stage_plan.get("total_gib") if stage_plan else None
            extreme_diffusion = bool(
                "diffusion" in managed_sample_stages
                and diffusion_plan
                and physical_total is not None
                and float(diffusion_plan.get("required_gib", 0.0)) > float(physical_total)
            )
            if extreme_diffusion and effective_async_streams > 0:
                effective_async_streams = 0
                print(
                    f"[VRAM-MGR] V11.5 extreme diffusion safety: estimate={diffusion_plan['required_gib']:.2f} GiB "
                    f"> physical VRAM={physical_total:.2f} GiB; async weight offload automatically disabled for this sampling process",
                    flush=True,
                )
            if effective_async_streams <= 0: comfy_args += ["--disable-async-offload"]
            else: comfy_args += ["--async-offload", str(effective_async_streams)]
            print(f"VRAM Manager V11.5: sample stages={','.join(managed_sample_stages) if ns.vram_manager_auto else 'forced all'} | engine={ns.vram_residency_engine} | runtime free={ns.vram_runtime_free_gb:g} GB | text load headroom={ns.vram_text_headroom_gb:g} GB | diffusion load headroom={ns.vram_diffusion_headroom_gb:g} GB | chunk={ns.vram_offload_chunk_mb} MB | async streams={effective_async_streams}", flush=True)
        if comfy_args:
            sample_env["H3_COMFY_ARGS"] = " ".join(comfy_args)
        for lp, strength in zip(ns.lora, ns.lora_strength): cmd += ["--lora", str(Path(lp).resolve()), "--lora-strength", str(strength)]
        if ns.extended_logging: cmd += ["--extended-logging"]
        for p in ns.ref_image[:9]: cmd += ["--ref-image", str(Path(p).resolve())]
        for p in ns.ref_video[:3]: cmd += ["--ref-video", str(Path(p).resolve())]
        for i, p in enumerate(ns.ref_audio[:3]):
            cmd += ["--ref-audio", str(Path(p).resolve())]
            subject_n = ns.ref_audio_subject[i] if i < len(ns.ref_audio_subject) else 0
            cmd += ["--ref-audio-subject", str(subject_n)]
        subprocess.check_call(cmd, cwd=ROOT, env=sample_env)
        print("Sampling process exited completely. Starting VAE with clean memory...", flush=True)

        ve = os.environ.copy()
        video_decode_managed = bool(stage_plan and stage_plan["stages"]["video_decode"]["use_manager"])
        if ns.vram_manager_auto and not video_decode_managed:
            ve["H3_COMFY_ARGS"] = f"--reserve-vram {max(0.1, ns.vram_runtime_free_gb):g} --fp16-vae"
            print("[VRAM-AUTO] video VAE decode launching NATIVE", flush=True)
        else:
            ve["H3_COMFY_ARGS"] = f"--lowvram --reserve-vram {max(0.1, ns.vram_video_vae_reserve_gb):g} --fp16-vae"
            if ns.vram_manager_auto: print("[VRAM-AUTO] video VAE decode launching MANAGED (--lowvram)", flush=True)
        video_cmd = [py, "-m", "runtime.video_decode_worker", "--latents", str(lat), "--vae", str(vv), "--frames-dir", str(frames_dir), "--tile-size", str(ns.video_vae_tile_size), "--tile-overlap", str(ns.video_vae_tile_overlap)]
        print(f"Video VAE tiling: {ns.video_vae_tile_size}px tile / {ns.video_vae_tile_overlap}px overlap", flush=True)
        if ns.extended_logging: video_cmd += ["--extended-logging"]
        if ns.tile_debugging: video_cmd += ["--tile-debugging"]
        subprocess.check_call(video_cmd, cwd=ROOT, env=ve)

        if ns.tile_debugging:
            debug_dir = out.parent / (out.stem + "_tile_debug"); debug_dir.mkdir(parents=True, exist_ok=True); copied = 0
            for name in ("tile_debug_plan.txt", "tile_debug_grid.png", "tile_debug_overlay_frame0.png", "frame_000000.png"):
                src = frames_dir / name
                if src.is_file(): shutil.copy2(src, debug_dir / name); copied += 1
            print(f"Persistent tile diagnostics: {debug_dir} ({copied} files copied)", flush=True)

        ok, msg = ensure_ffmpeg_tools(lambda x: print(x, flush=True))
        if not ok:
            print(f"ERROR: FFmpeg tools unavailable: {msg}", flush=True); return 2
        ff = str(tool_path("ffmpeg.exe")); ae = os.environ.copy(); audio_ok = False
        audio_decode_managed = bool(stage_plan and stage_plan["stages"]["audio_decode"]["use_manager"])
        if ns.vram_manager_auto and not audio_decode_managed:
            ae["H3_COMFY_ARGS"] = f"--reserve-vram {max(0.1, ns.vram_runtime_free_gb):g}"
            print("[VRAM-AUTO] audio VAE decode launching NATIVE", flush=True)
        else:
            ae["H3_COMFY_ARGS"] = f"--novram --reserve-vram {max(0.1, ns.vram_audio_vae_reserve_gb):g}"
            if ns.vram_manager_auto: print("[VRAM-AUTO] audio VAE decode launching MANAGED (--novram)", flush=True)
        try:
            audio_cmd = [py, "-m", "runtime.audio_decode_worker", "--latents", str(lat), "--vae", str(av), "--wav", str(wav)]
            if ns.extended_logging: audio_cmd += ["--extended-logging"]
            subprocess.check_call(audio_cmd, cwd=ROOT, env=ae)
            print('Muxing video and audio...', flush=True)
            tmp = td / "mux.mp4"; subprocess.check_call([ff, "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "256k", "-shortest", str(tmp)])
            if out.exists(): out.unlink()
            shutil.move(str(tmp), str(out)); audio_ok = True
        except subprocess.CalledProcessError:
            if out.exists(): out.unlink()
            subprocess.check_call([ff, "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)])
        print(("Saved with audio: " if audio_ok else "Saved video-only fallback: ") + str(out), flush=True); return 0 if audio_ok else 3

if __name__ == "__main__": raise SystemExit(main())
