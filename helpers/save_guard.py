
"""
helpers/save_guard.py - Force unique image saves globally.
Loads early (import helpers.save_guard) and monkey-patches PIL.Image.Image.save
to auto-rename if the target path already exists.
"""
from pathlib import Path
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # tolerate missing PIL in some contexts

def _unique_path(p: Path) -> Path:
    try:
        p = Path(p)
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix
        i = 1
        while True:
            cand = p.with_name(f"{stem}_{i:03d}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    except Exception:
        return Path(p)

if Image is not None and getattr(Image.Image, "save", None):
    _orig_save = Image.Image.save
    def _safe_save(self, fp, *args, **kwargs):
        # If fp is a file-like object, let PIL handle it (no path)
        try:
            if hasattr(fp, "write"):
                return _orig_save(self, fp, *args, **kwargs)
        except Exception:
            pass
        try:
            p = Path(str(fp))
            # Only uniquify for local filesystem paths
            p = _unique_path(p)
            return _orig_save(self, str(p), *args, **kwargs)
        except Exception:
            # Last resort: fall back to original
            return _orig_save(self, fp, *args, **kwargs)
    Image.Image.save = _safe_save
