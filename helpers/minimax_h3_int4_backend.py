"""Small helper intended for PySide6 integration.
The GUI can call generate_video() directly or launch generate.py as a subprocess.
"""
from __future__ import annotations
from pathlib import Path

def validate_install():
    from runtime.validate_models import validate
    errors, info, diff, ref, te, vv, av = validate(require_vae=True)
    return {"ok": not errors, "errors": errors, "diffusion": str(diff) if diff else None,
            "text_encoder": str(te) if te else None, "video_vae": str(vv) if vv else None,
            "audio_vae": str(av) if av else None}

def generate_video(prompt: str, **kwargs):
    from runtime.validate_models import validate
    errors, info, diff, ref, te, vv, av = validate(require_vae=True)
    if errors: raise RuntimeError("MiniMax-H3 model validation failed: " + " | ".join(errors))
    from runtime.headless_h3 import generate
    return generate(diffusion_model=diff, text_encoder=te, video_vae=vv, audio_vae=av, prompt=prompt, **kwargs)
