#!/usr/bin/env python
"""
Small helper script that runs Z-Image Turbo txt2img in a dedicated environment.

It is meant to be called from helpers/txt2img.py via the python.exe inside
<root>/.zimage_env/, and returns a single line of JSON with the list of files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

def _unique_path(p: Path) -> Path:
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
    args = parser.parse_args(argv)

    try:
        import torch  # type: ignore
        from diffusers import ZImagePipeline  # type: ignore
    except Exception as e:
        payload = {"files": [], "error": f"import_failed: {e}"}
        print(json.dumps(payload))
        return 1

    # Resolve root + model dir
    try:
        root_dir = Path(__file__).resolve().parents[1]
    except Exception:
        root_dir = Path(".").resolve()
    model_dir = root_dir / "models" / "Z-Image-Turbo"
    if not model_dir.exists():
        payload = {"files": [], "error": f"model_not_found: {model_dir}"}
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

    try:
        pipe = ZImagePipeline.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        pipe = pipe.to(device)
        try:
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
        except Exception:
            pass
    except Exception as e:
        payload = {"files": [], "error": f"load_failed: {e}", "trace": traceback.format_exc()}
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
            result = pipe(
                prompt=args.prompt,
                height=int(args.height),
                width=int(args.width),
                num_inference_steps=max(1, int(args.steps or 1)),
                guidance_scale=float(args.guidance or 0.0),
                generator=gen,
            )
            img = result.images[0]
        except Exception as e:
            payload = {"files": files, "error": f"generate_failed: {e}", "trace": traceback.format_exc()}
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
            payload = {"files": files, "error": f"save_failed: {e}", "trace": traceback.format_exc()}
            print(json.dumps(payload))
            return 1

        files.append(str(fpath))

    payload = {"files": files, "model": "Z-Image-Turbo"}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
