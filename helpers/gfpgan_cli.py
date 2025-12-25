# helpers/gfpgan_cli.py
# Minimal GFPGAN single-image CLI wrapper for FrameVision / FrameLab style tools.
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

def _imread_unicode(path: Path):
    """cv2.imread that works with unicode paths on Windows."""
    import numpy as np
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data is None or data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def _imwrite_unicode(path: Path, img) -> bool:
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = (path.suffix or ".png").lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="GFPGAN one-shot face restoration (single image).")
    ap.add_argument("--in", dest="inp", required=True, help="Input image path")
    ap.add_argument("--out", dest="out", required=True, help="Output image path")
    ap.add_argument("--model", dest="model", required=True, help="GFPGAN .pth weights path")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    ap.add_argument("--upscale", type=int, default=1, help="GFPGAN upscale factor (keep at 1; upscaling is done by other engines)")
    ap.add_argument("--strength", type=float, default=float(os.environ.get("FV_GFPGAN_STRENGTH", "0.45")), help="Blend restored result with original. 0=original, 1=full GFPGAN")
    args = ap.parse_args(argv)

    in_path = Path(args.inp)
    out_path = Path(args.out)
    model_path = Path(args.model)

    if not in_path.exists():
        print(f"[gfpgan_cli] missing input: {in_path}", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"[gfpgan_cli] missing model: {model_path}", file=sys.stderr)
        return 3

    # Resolve device
    device = "cpu"
    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    try:
        import cv2
        from gfpgan import GFPGANer
    except Exception as e:
        print(f"[gfpgan_cli] import error: {e}", file=sys.stderr)
        return 4

    img = _imread_unicode(in_path)
    if img is None:
        print(f"[gfpgan_cli] failed to read: {in_path}", file=sys.stderr)
        return 5

    try:
        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=max(1, int(args.upscale)),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        # enhance returns (cropped_faces, restored_faces, restored_img)
        _cropped, _restored, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # Blend restored result with original to reduce over-stylization
        try:
            strength = float(args.strength)
        except Exception:
            strength = 0.45
        if strength < 0.0: strength = 0.0
        if strength > 1.0: strength = 1.0
        if strength < 1.0 and restored_img is not None:
            import numpy as np
            # Ensure same shape
            if restored_img.shape[:2] != img.shape[:2]:
                restored_img = cv2.resize(restored_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            # addWeighted expects same dtype
            if restored_img.dtype != img.dtype:
                restored_img = restored_img.astype(img.dtype, copy=False)
            restored_img = cv2.addWeighted(restored_img, strength, img, 1.0 - strength, 0.0)
        if restored_img is None:
            print("[gfpgan_cli] enhance returned no image", file=sys.stderr)
            return 6
        if not _imwrite_unicode(out_path, restored_img):
            print(f"[gfpgan_cli] failed to write: {out_path}", file=sys.stderr)
            return 7
        print(f"[gfpgan_cli] ok -> {out_path}")
        return 0
    except Exception as e:
        print(f"[gfpgan_cli] runtime error: {e}", file=sys.stderr)
        return 8

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))