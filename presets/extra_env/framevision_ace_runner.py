import json
import os
import sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: framevision_ace_runner.py <job_json_path>", file=sys.stderr)
        return 1

    job_path = Path(sys.argv[1])
    if not job_path.exists():
        print(f"Job JSON not found: {job_path}", file=sys.stderr)
        return 1

    with open(job_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Ensure we stay inside the ACE env and use its packages.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.get("device_id", 0)))

    from acestep.pipeline_ace_step import ACEStepPipeline

    checkpoint_path = cfg["checkpoint_path"]

    pipe = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype=cfg.get("dtype", "bfloat16"),
        torch_compile=bool(cfg.get("torch_compile", False)),
        cpu_offload=bool(cfg.get("cpu_offload", False)),
        overlapped_decode=bool(cfg.get("overlapped_decode", False)),
    )

    out_path = cfg["output_path"]

    pipe(
        audio_duration=float(cfg["audio_duration"]),
        prompt=cfg["prompt"],
        lyrics=cfg["lyrics"],
        infer_step=int(cfg["infer_step"]),
        guidance_scale=float(cfg["guidance_scale"]),
        scheduler_type=cfg["scheduler_type"],
        cfg_type=cfg["cfg_type"],
        omega_scale=float(cfg["omega_scale"]),
        manual_seeds=cfg["manual_seeds"],
        guidance_interval=float(cfg["guidance_interval"]),
        guidance_interval_decay=float(cfg["guidance_interval_decay"]),
        min_guidance_scale=float(cfg["min_guidance_scale"]),
        use_erg_tag=bool(cfg["use_erg_tag"]),
        use_erg_lyric=bool(cfg["use_erg_lyric"]),
        use_erg_diffusion=bool(cfg["use_erg_diffusion"]),
        oss_steps=cfg["oss_steps"],
        guidance_scale_text=float(cfg["guidance_scale_text"]),
        guidance_scale_lyric=float(cfg["guidance_scale_lyric"]),
        audio2audio_enable=bool(cfg.get("audio2audio_enable", False)),
        ref_audio_strength=float(cfg.get("ref_audio_strength", 0.5)),
        ref_audio_input=cfg.get("ref_audio_input") or None,
        save_path=out_path,
    )

    print(f"ACE-Step generation finished. Output written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
