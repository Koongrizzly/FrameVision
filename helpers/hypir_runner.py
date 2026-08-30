#!/usr/bin/env python3
"""FrameVision HYPIR image restoration/upscale runner.

This process is intentionally launched with FrameVision's isolated .hypir
Python environment so HYPIR's Torch/Diffusers dependencies do not touch the
main FrameVision interpreter.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrameVision HYPIR runner")
    p.add_argument("--repo", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, default=4, choices=range(1, 9))
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1, help="Number of same-size frames/images to process together")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    base_model = Path(args.base_model).resolve()
    weight = Path(args.weight).resolve()
    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()

    for label, path in (
        ("HYPIR repository", repo),
        ("Stable Diffusion 2.1 base model", base_model),
        ("HYPIR weight", weight),
        ("input path", inp),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Imports stay in this isolated subprocess by design.
    import torch
    from accelerate.utils import set_seed
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from HYPIR.enhancer.sd2 import SD2Enhancer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU in the .hypir environment.")

    seed = int(args.seed)
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    patch_size = max(512, min(1024, int(args.patch_size)))
    stride = max(128, min(patch_size, int(args.stride)))
    batch_size = max(1, min(8, int(args.batch_size)))

    print("[HYPIR] FrameVision runner")
    print(f"[HYPIR] device={args.device} seed={seed} scale={args.scale}x")
    print(f"[HYPIR] patch_size={patch_size} stride={stride} batch_size={batch_size}")
    print(f"[HYPIR] base_model={base_model}")
    print(f"[HYPIR] weight={weight}")
    print(f"[HYPIR] input={inp}")
    print(f"[HYPIR] output={out}")

    lora_modules = [
        "to_k", "to_q", "to_v", "to_out.0",
        "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
        "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
    ]

    model = SD2Enhancer(
        base_model_path=str(base_model),
        weight_path=str(weight),
        lora_modules=lora_modules,
        lora_rank=256,
        model_t=200,
        coeff_t=200,
        device=args.device,
    )
    print("[HYPIR] loading model...")
    model.init_models()
    print("[HYPIR] model loaded")

    def _save_result(result, out_file: Path) -> None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_file.suffix.lower()
        save_kwargs = {}
        if suffix in {".jpg", ".jpeg"}:
            save_kwargs.update(quality=100, subsampling=0)
        result.save(out_file, **save_kwargs)
        print(f"[HYPIR] saved: {out_file}")

    def _enhance_batch(in_files: list[Path], out_files: list[Path]) -> None:
        images = [Image.open(path).convert("RGB") for path in in_files]
        try:
            # Video frames extracted by FFmpeg have identical dimensions. Keep
            # an explicit check here so ordinary directory use fails clearly.
            sizes = {img.size for img in images}
            if len(sizes) != 1:
                raise ValueError(
                    "HYPIR batching requires all images in a batch to have the same dimensions. "
                    f"Got: {sorted(sizes)}"
                )
            image_tensor = torch.stack([to_tensor(img) for img in images], dim=0)
            with torch.inference_mode():
                results = model.enhance(
                    lq=image_tensor,
                    prompt=args.prompt,
                    upscale=int(args.scale),
                    patch_size=patch_size,
                    stride=stride,
                    return_type="pil",
                )
            if len(results) != len(out_files):
                raise RuntimeError(
                    f"HYPIR returned {len(results)} result(s) for a batch of {len(out_files)} image(s)."
                )
            for result, out_file in zip(results, out_files):
                _save_result(result, out_file)
        finally:
            # Do not empty CUDA's allocator cache between batches. Reusing the
            # allocations is considerably better for long video runs.
            try:
                del images
            except Exception:
                pass
            try:
                del image_tensor
            except Exception:
                pass
            try:
                del results
            except Exception:
                pass

    def _enhance_one(in_file: Path, out_file: Path) -> None:
        _enhance_batch([in_file], [out_file])

    if inp.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
        if not files:
            raise FileNotFoundError(f"No supported image frames found in input directory: {inp}")
        total = len(files)
        print(f"[HYPIR] processing directory with {total} frames; batch_size={batch_size}")
        for start_idx in range(0, total, batch_size):
            batch_files = files[start_idx:start_idx + batch_size]
            batch_out = [out / path.name for path in batch_files]
            first = start_idx + 1
            last = start_idx + len(batch_files)
            if len(batch_files) == 1:
                print(f"[HYPIR] frame {first}/{total}: {batch_files[0].name}")
            else:
                print(f"[HYPIR] frames {first}-{last}/{total}: batch={len(batch_files)}")
            _enhance_batch(batch_files, batch_out)
    else:
        _enhance_one(inp, out)

    # Explicitly release GPU allocations before subprocess exit. This matters
    # when FrameVision launches several jobs in sequence.
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
