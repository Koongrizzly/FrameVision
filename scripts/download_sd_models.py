"""
Stable Diffusion (SD1.5 / SDXL) optional downloader helpers for FrameVision.

This module was extracted from scripts/download_externals.py to keep that script smaller.
Nothing here runs unless called explicitly (or via run_txt2img_auto()).

Behavior:
- If env FVS_TXT2IMG=1 OR any .urls/<family>.txt exists, it downloads listed URLs into models/.
- Otherwise it can optionally try a "quick demo" fetch via huggingface_hub (internet required).

URL list format (.urls/<family>.txt):
  URL
  URL -> relative\path\inside\models
Lines starting with # are ignored.
"""
from __future__ import annotations

import os
import glob
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Iterable, Tuple, List

def _dl(url: str, dst: Path, timeout: int = 180) -> bool:
    """Simple downloader with a stable UA; returns True on success."""
    try:
        print("[externals][txt2img] download:", url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "framevision-installer"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print("[externals][txt2img] download failed:", e, "@", url)
        try:
            dst.unlink()
        except Exception:
            pass
        return False

def _txt2img_families_default() -> List[str]:
    # Families your UI can list. They map to .urls/<family>.txt, not to fixed vendors.
    return [
        "sd15_photoreal",
        "sd15_anime",
        "sd15_vae",
        "sdxl_base",
        "sdxl_refiner",
        "sdxl_feature_models",
        "embeddings_neg",
        "sd15_inpaint",
        "sdxl_inpaint",
        "controlnet_sd15",
        "controlnet_sdxl",
    ]

def _parse_mapping_line(line: str) -> Tuple[str, Optional[Path]]:
    """Parses 'URL [-> relative\\path]' mapping lines."""
    if "->" in line:
        url, rel = line.split("->", 1)
        relp = rel.strip()
        return url.strip(), (Path(relp) if relp else None)
    return line.strip(), None

def _download_to(models_root: Path, url: str, relpath: Optional[Path]) -> bool:
    # Choose a destination path
    if relpath is None or getattr(relpath, "name", "") == "":
        name = url.split("?")[0].rstrip("/").split("/")[-1]
        relpath = Path(name)
    dst = models_root / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)

    ok = _dl(url, dst)
    if not ok:
        return False

    # If it's a zip, extract and delete the zip
    try:
        if dst.suffix.lower() == ".zip":
            with zipfile.ZipFile(dst, "r") as z:
                z.extractall(dst.parent)
            try:
                dst.unlink()
            except Exception:
                pass
    except Exception as e:
        print("[externals][txt2img] zip extract error:", e)
    return True

def ensure_txt2img_models(
    families: Optional[Iterable[str]] = None,
    urls_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> None:
    root = Path(__file__).resolve().parent.parent
    urls_dir = urls_dir or (root / ".urls")
    models_dir = models_dir or (root / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    urls_dir.mkdir(parents=True, exist_ok=True)

    fams = list(families) if families is not None else _txt2img_families_default()

    total = 0
    ok = 0
    miss = 0
    for fam in fams:
        ftxt = urls_dir / f"{fam}.txt"
        if not ftxt.exists():
            print(f"[externals][txt2img] no .urls/{fam}.txt — skipping")
            miss += 1
            continue
        lines = [
            ln.strip()
            for ln in ftxt.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not lines:
            print(f"[externals][txt2img] .urls/{fam}.txt is empty — skipping")
            miss += 1
            continue
        print(f"[externals][txt2img] downloading family '{fam}' ({len(lines)} items)")
        for ln in lines:
            url, rel = _parse_mapping_line(ln)
            total += 1
            if _download_to(models_dir, url, rel):
                ok += 1
    print(f"[externals][txt2img] done: {ok}/{total} files fetched; {miss} families skipped")

def _txt2img_should_run(urls_dir: Optional[Path] = None) -> bool:
    # Try both project root (parent of 'scripts') and local script directory
    candidates: List[Path] = []
    if urls_dir is not None:
        candidates.append(Path(urls_dir))

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir.parent / ".urls")  # project root
    candidates.append(script_dir / ".urls")         # next to scripts/

    fams = _txt2img_families_default()

    if os.environ.get("FVS_TXT2IMG", "0") == "1":
        print("[externals][txt2img] env FVS_TXT2IMG=1 — enabling TXT→IMG downloads")
        return True

    for udir in candidates:
        try:
            for fam in fams:
                if (udir / f"{fam}.txt").exists():
                    print(f"[externals][txt2img] detected URLs file: {udir / (fam + '.txt')}")
                    return True
        except Exception:
            pass

    print("[externals][txt2img] no .urls/* files detected; skipping TXT→IMG downloads")
    return False

def _txt2img_demo_try() -> bool:
    """Best-effort demo fetch for a quick test. Requires huggingface_hub and internet."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print("[externals][txt2img] demo fetch: huggingface_hub not available:", e)
        return False

    try:
        models_root = Path(__file__).resolve().parent.parent / "models"
        sdxl_dir = models_root / "SDXL"
        sd15_dir = models_root / "SD15"
        sdxl_dir.mkdir(parents=True, exist_ok=True)
        sd15_dir.mkdir(parents=True, exist_ok=True)

        # SDXL candidates
        sdxl_targets = [
            ("RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"),
            ("RunDiffusion/Juggernaut-XL-Lightning", "Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors"),
            ("SG161222/RealVisXL_V4.0", "RealVisXL_V4.0_B1_fp16-no-ema.safetensors"),
        ]
        for repo_id, filename in sdxl_targets:
            dst = sdxl_dir / filename
            if dst.exists():
                print("[externals][txt2img] demo model ready at", dst)
                return True
            try:
                print(f"[externals][txt2img] fetching SDXL demo: {repo_id}/{filename} …")
                fp = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(sdxl_dir),
                    local_dir_use_symlinks=False,
                )
                # Move to exact filename if HF used a cache path
                try:
                    p = Path(fp)
                    if p.is_file() and not dst.exists():
                        p.replace(dst)
                except Exception:
                    pass
                if dst.exists():
                    print("[externals][txt2img] demo SDXL model ready at", dst)
                    return True
            except Exception as e:
                print("[externals][txt2img] SDXL attempt failed:", e)

        # SD1.5 fallbacks
        ds_file = sd15_dir / "DreamShaper_8_pruned.safetensors"
        if not ds_file.exists():
            print("[externals][txt2img] fetching fallback SD1.5: Lykon/DreamShaper …")
            fp = hf_hub_download(
                repo_id="Lykon/DreamShaper",
                filename="DreamShaper_8_pruned.safetensors",
                local_dir=str(sd15_dir),
                local_dir_use_symlinks=False,
            )
            try:
                p = Path(fp)
                if p.is_file() and not ds_file.exists():
                    p.replace(ds_file)
            except Exception:
                pass
        if ds_file.exists():
            print("[externals][txt2img] demo model ready at", ds_file)
            return True

        rv_file = sd15_dir / "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
        if not rv_file.exists():
            print("[externals][txt2img] DreamShaper not available; fetching SG161222/Realistic_Vision_V5.1_noVAE (fp16)…")
            fp = hf_hub_download(
                repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
                filename="Realistic_Vision_V5.1_fp16-no-ema.safetensors",
                local_dir=str(sd15_dir),
                local_dir_use_symlinks=False,
            )
            try:
                p = Path(fp)
                if p.is_file() and not rv_file.exists():
                    p.replace(rv_file)
            except Exception:
                pass
        if rv_file.exists():
            print("[externals][txt2img] demo model ready at", rv_file)
            return True

        print("[externals][txt2img] demo fetch: no models available")
        return False
    except Exception as e:
        print("[externals][txt2img] demo fetch failed:", e)
        return False

def run_txt2img_auto() -> None:
    """Mirror the old download_externals.py behavior for SD15/SDXL."""
    try:
        if _txt2img_should_run():
            ensure_txt2img_models()
        else:
            _txt2img_demo_try()
    except Exception as e:
        print("[externals][txt2img] auto-run error:", e)

if __name__ == "__main__":
    run_txt2img_auto()
