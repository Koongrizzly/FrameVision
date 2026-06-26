"""
FrameVision Optional Installs UI (PySide6)

This module provides a small dialog that lets users select and run optional
install add-ons. It mirrors the "optional installs" strategies from
install_menu.bat by invoking the same extra installer scripts (when present).

Intended usage:
- Import in FrameVision and open the dialog from a menu item.
- Can also be run standalone for testing: `python -m helpers.opt_installs`
"""

from __future__ import annotations

import os
import sys
import shutil
import stat
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

from PySide6 import QtCore, QtGui, QtWidgets

# Optional: licenses viewer
try:
    from helpers.licenses_viewer import show_3rd_party_licenses
except Exception:
    show_3rd_party_licenses = None  # type: ignore



# -----------------------------
# Models / Tasks
# -----------------------------

@dataclass(frozen=True)
class OptionalInstall:
    key: str
    title: str
    description: str
    # Returns (program, args, working_dir) or None when missing prerequisites.
    runner: Callable[[Path], Optional[Tuple[str, List[str], Path]]]


def _root_from_this_file() -> Path:
    """
    Determine the FrameVision root folder.

    - If this file lives in <root>/helpers/, return <root>
    - If this file lives directly in <root>/, return <root>
    """
    try:
        p = Path(__file__).resolve()
        if p.parent.name.lower() == "helpers":
            return p.parent.parent
        return p.parent
    except Exception:
        return Path.cwd().resolve()


def _venv_python(root: Path) -> Optional[Path]:
    """
    Prefer <root>/.venv python. Fall back to current interpreter.
    """
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",   # Windows
        root / ".venv" / "bin" / "python",           # *nix
        root / ".venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return c
    # If we're running inside FrameVision already, sys.executable is fine.
    if Path(sys.executable).exists():
        return Path(sys.executable)
    return None


def _cmd_call_bat(script_path: Path, cwd: Path) -> Tuple[str, List[str], Path]:
    """
    On Windows, run .bat through cmd.exe /c call
    """
    return ("cmd.exe", ["/c", "call", str(script_path)], cwd)


# -----------------------------
# Shared model cleanup
# -----------------------------

_SHARED_MODEL_TARGETS: Dict[str, Tuple[str, ...]] = {
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "models/wan22",
        "models/hiar/HiAR/wan_models/Wan2.1-T2V-1.3B",
    ),
    "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf": (
        "models/Qwen-Image-2512 GGUF",
        "models/FireRed-Image-Edit-1.1",
        "models/qwen2511gguf/text_encoders",
    ),
    "Qwen3-8B-Q5_K_M.gguf": (
        "models/klein4b_gguf/text_encoders",
        "models/klein9b_gguf/text_encoders",
    ),
    "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf": (
        "models/FireRed-Image-Edit-1.1",
        "models/qwen2511gguf/text_encoders",
    ),
}


def _cleanup_shared_model_files(root: Path) -> List[str]:
    """Move known FrameVision shared files into models/shared.

    This is intentionally narrow: only the known FrameVision subfolders above are
    checked. Custom/user model folders elsewhere are not scanned or touched.
    """
    moved: List[str] = []
    try:
        models_dir = root / "models"
        shared_dir = models_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        for filename, rel_folders in _SHARED_MODEL_TARGETS.items():
            dest = shared_dir / filename
            for rel in rel_folders:
                src = root / rel / filename
                try:
                    if not src.exists() or not src.is_file():
                        continue
                    if src.resolve() == dest.resolve():
                        continue
                    if dest.exists():
                        try:
                            dest.unlink()
                        except Exception:
                            pass
                    shutil.move(str(src), str(dest))
                    moved.append(f"{filename} -> models/shared")
                except Exception:
                    # Quiet cleanup: never block opening Optional Installs.
                    pass
    except Exception:
        pass
    return moved


def _run_wan22(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "wan22_setup.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_wan22_turbo(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install WAN 2.2 Turbo using the combined normal + Turbo installer.

    The installer itself must do the full combined install by default:
    full normal WAN 2.2 model/repo first, then Turbo additions.
    Do not require hidden console arguments for the normal UI button path.
    """
    script = root / "presets" / "extra_env" / "wan22_turbo_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--app-root", str(root)], root)


def _zimage_unified_script(root: Path) -> Optional[Path]:
    """Return the unified Z-Image installer script introduced by the cleanup patch."""
    script = root / "presets" / "extra_env" / "zimage_install.py"
    if script.exists():
        return script
    return None


def _run_zimage_installer(root: Path, args_extra: List[str]) -> Optional[Tuple[str, List[str], Path]]:
    """Run presets/extra_env/zimage_install.py with the current app Python."""
    script = _zimage_unified_script(root)
    if script is None:
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    args = ["-u", str(script), str(root), *args_extra]
    return (str(py), args, root)


def _run_zimage(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Create/update the shared image-model environment in environments/.images_models."""
    return _run_zimage_installer(root, ["--mode", "env"])


def _run_chroma(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install/update SPARK.Chroma using the portable Chroma installer.

    The installer reuses the shared image-model environment when present and
    downloads SPARK.Chroma into models/chroma/SPARK.Chroma_v1.
    """
    script = root / "presets" / "extra_env" / "chroma_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script)], root)


def _run_zimage_fp16(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install the Z-Image Turbo full 16-bit Diffusers model with automatic BF16/FP16 runtime selection."""
    target = (root / "models" / "Z-Image-Turbo").resolve()
    return _run_zimage_installer(
        root,
        [
            str(target),
            "--mode",
            "diffusers",
            "--precision",
            "auto",
        ],
    )


def _run_zimage_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Z-Image Turbo GGUF installer through the unified installer.
    Supported quants: Q4_0, Q5_0, Q6_K, Q8_0
    """
    target = (root / "models" / "Z-Image-Turbo GGUF").resolve()
    return _run_zimage_installer(
        root,
        [
            str(target),
            "--mode",
            "gguf",
            "--quant",
            quant,
            "--match-text-quant",
            "1",
        ],
    )


def _run_zimage_gguf5(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    # Back-compat key: was "GGUF5 model" -> now installs diffusion Q5_0.
    return _run_zimage_gguf(root, "Q5_0")


def _run_zimage_gguf4(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q4_0")


def _run_zimage_gguf6(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q6_K")


def _run_zimage_gguf8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q8_0")


def _run_zimage_base_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """Download Z-Image (base) GGUF for the selected quant + required VAE + text encoder."""
    script = root / "presets" / "extra_env" / "z-image_base_gguf_download.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    target = (root / "models" / "z-image_base_GGuf").resolve()
    args = [
        "-u",
        str(script),
        str(root),
        str(target),
        "--quant",
        str(quant),
    ]
    return (str(py), args, root)


def _run_zimage_base_q2k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q2_K")


def _run_zimage_base_q3kl(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q3_K_L")


def _run_zimage_base_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q4_K_M")


def _run_zimage_base_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q5_K_M")


def _run_zimage_base_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q6_K")


def _run_zimage_base_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_base_gguf(root, "Q8_0")




def _klein_script(root: Path) -> Optional[Path]:
    """Return the Klein GGUF downloader script when present.

    Supported script filenames (preferred order):
      - presets/extra_env/klein_gguf_download.py   (unified 4B/9B)
      - presets/extra_env/klein4b_gguf_download.py (legacy 4B-only)
    """
    candidates = [
        root / "presets" / "extra_env" / "klein_gguf_download.py",
        root / "presets" / "extra_env" / "klein4b_gguf_download.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _klein_script_supports_variant(script: Path) -> bool:
    # Unified script supports --variant (4b/9b)
    return script.name.lower() == "klein_gguf_download.py"


def _klein_base_dir(root: Path, variant: str, script: Path) -> Path:
    """Return the models base folder used by the downloader."""
    if _klein_script_supports_variant(script):
        v = variant.lower().strip()
        if v in ("9b", "9", "klein9b"):
            return root / "models" / "klein9b_gguf"
        return root / "models" / "klein4b_gguf"
    # Legacy script: always uses klein4b_gguf
    return root / "models" / "klein4b_gguf"


def _klein_has_vae(root: Path, variant: str) -> bool:
    """Check whether the FLUX2 VAE is already present for the given variant."""
    script = _klein_script(root)
    if not script:
        return False
    base = _klein_base_dir(root, variant, script)
    p = base / "vae" / "split_files" / "vae" / "flux2-vae.safetensors"
    return p.exists()


def _klein4b_script(root: Path) -> Optional[Path]:
    """Return a Klein downloader script usable for 4B installs."""
    return _klein_script(root)
def _run_klein4b_unet(root: Path, quant_token: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Download a single FLUX.2-klein-4B UNet GGUF by substring token (e.g. Q4_K_M).
    Auto-downloads the required VAE if it's not present yet.
    """
    script = _klein4b_script(root)
    if not script:
        return None

    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    args = ["-u", str(script)]
    if _klein_script_supports_variant(script):
        args += ["--variant", "4b"]
    args += ["--download-unet", str(quant_token)]
    if not _klein_has_vae(root, "4b"):
        args.append("--download-vae")

    return (str(py), args, root)


def _run_klein4b_textenc(root: Path, source: str, token: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Download a text encoder for Klein 4B. Source must be: comfy / gguf / cordux.
    Token can be an index (e.g. "1") or a substring (e.g. "Q4_K_M") depending on source.
    """
    script = _klein4b_script(root)
    if not script:
        return None

    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    args = ["-u", str(script)]
    if _klein_script_supports_variant(script):
        args += ["--variant", "4b"]
    args += ["--download-textenc", f"{source}:{token}"]

    return (str(py), args, root)


def _run_klein4b_unet_q2k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q2_K")


def _run_klein4b_unet_q3kl(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q3_K_L")


def _run_klein4b_unet_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q4_K_M")


def _run_klein4b_unet_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q5_K_M")


def _run_klein4b_unet_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q6_K")


def _run_klein4b_unet_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_unet(root, "Q8_0")


def _run_klein4b_te_comfy(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    # Comfy bundle typically has a single safetensors weight; use index 1.
    return _run_klein4b_textenc(root, "comfy", "1")


def _run_klein4b_te_qwen_gguf_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_textenc(root, "gguf", "Q4_K_M")


def _run_klein4b_te_qwen_gguf_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_textenc(root, "gguf", "Q5_K_M")


def _run_klein4b_te_qwen_gguf_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein4b_textenc(root, "gguf", "Q8_0")


def _run_klein4b_te_cordux(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    # Gated repo – may require accepting terms on Hugging Face. Use index 1.
    return _run_klein4b_textenc(root, "cordux", "1")


def _run_klein9b_unet(root: Path, quant_token: str) -> Optional[Tuple[str, List[str], Path]]:
    """Download a single FLUX.2-klein-9B UNet GGUF by substring token (e.g. Q4_K_M).

    Requires the unified downloader script (klein_gguf_download.py) that supports --variant 9b.
    Auto-downloads the required VAE if it's not present yet.
    """
    script = _klein_script(root)
    if not script or (not _klein_script_supports_variant(script)):
        return None

    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    args = ["-u", str(script), "--variant", "9b", "--download-unet", str(quant_token)]
    if not _klein_has_vae(root, "9b"):
        args.append("--download-vae")

    return (str(py), args, root)


def _run_klein9b_textenc(root: Path, token: str) -> Optional[Tuple[str, List[str], Path]]:
    """Download a Klein 9B text encoder (Qwen3-8B GGUF) by substring token."""
    script = _klein_script(root)
    if not script or (not _klein_script_supports_variant(script)):
        return None

    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    args = ["-u", str(script), "--variant", "9b", "--download-textenc", f"gguf:{token}"]
    return (str(py), args, root)


def _run_klein9b_unet_q2k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q2_K")


def _run_klein9b_unet_q3km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q3_K_M")


def _run_klein9b_unet_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q4_K_M")


def _run_klein9b_unet_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q5_K_M")


def _run_klein9b_unet_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q6_K")


def _run_klein9b_unet_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_unet(root, "Q8_0")


def _run_klein9b_te_q2k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_textenc(root, "Q2_K")


def _run_klein9b_te_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_textenc(root, "Q4_K_M")


def _run_klein9b_te_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_textenc(root, "Q5_K_M")


def _run_klein9b_te_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_textenc(root, "Q6_K")


def _run_klein9b_te_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_klein9b_textenc(root, "Q8_0")



def _run_ace(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "ace_setup.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_ace15(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Ace Step 1.5 Music Creation (turbo repo installer)."""
    script = root / "presets" / "extra_env" / "install_ace_step_15.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_gfpgan(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "gfpgan_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_seedvr2_env(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install / update SeedVR2 GGUF environment and dependencies (no model downloads)."""
    script = root / "presets" / "extra_env" / "seedvr2_gguf_installer.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    args = ["-u", str(script), "--skip-models"]
    return (str(py), args, root)


def _run_seedvr2_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """Download SeedVR2 3B GGUF for a given quant (will create/reuse env as needed)."""
    script = root / "presets" / "extra_env" / "seedvr2_gguf_installer.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    args = ["-u", str(script), "--quant", str(quant)]
    return (str(py), args, root)


def _run_seedvr2_q3(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_seedvr2_gguf(root, "Q3")


def _run_seedvr2_q4(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_seedvr2_gguf(root, "Q4")


def _run_seedvr2_q5(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_seedvr2_gguf(root, "Q5")


def _run_seedvr2_q6(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_seedvr2_gguf(root, "Q6")


def _run_seedvr2_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_seedvr2_gguf(root, "Q8")


def _run_hunyuan15(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install/update Hunyuan 1.5 using the portable conda-prefix installer.

    Env target: <root>/environments/.hunyuan15_official/
    The installer creates a real conda env, so the runtime Python is:
        <root>/environments/.hunyuan15_official/python.exe
    """
    script = root / "presets" / "extra_env" / "hunyuan15_install.py"
    if script.exists():
        py = _venv_python(root)
        if py is None or (not py.exists()):
            return None
        return (str(py), ["-u", str(script), "--root", str(root)], root)

    # Legacy fallback only for old builds that still ship the BAT.
    bat = root / "presets" / "extra_env" / "hunyuan15_install.bat"
    if bat.exists():
        return _cmd_call_bat(bat, root)
    return None


def _run_hunyuan15_flashattn(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install optional FlashAttention wheel into the Hunyuan 1.5 environment."""
    candidates = [
        # New conda-prefix installer location.
        root / "environments" / ".hunyuan15_official" / "python.exe",
        # Legacy/current venv layouts kept as fallback only.
        root / "environments" / ".hunyuan15_official" / "Scripts" / "python.exe",
        root / "environments" / ".hunyuan15" / "python.exe",
        root / "environments" / ".hunyuan15" / "Scripts" / "python.exe",
        root / ".hunyuan15_env" / "Scripts" / "python.exe",
    ]
    py = next((p for p in candidates if p.exists()), candidates[0])
    if not py.exists():
        return None
    wheel_url = (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.19/"
        "flash_attn-2.8.3%2Bcu124torch2.5-cp311-cp311-win_amd64.whl"
    )
    return (str(py), ["-m", "pip", "install", wheel_url], root)



def _run_bernini_r_1p3b(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install/update Bernini-R 1.3B using the portable conda-prefix installer.

    Env target: <root>/environments/.bernini_r/
    Models/repo target: <root>/models/bernini_r_1p3b/
    """
    script = root / "presets" / "extra_env" / "install_bernini_r_1p3b.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--root", str(root), "--yes"], root)

def _run_hiar(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install HiAR Wan 2.1 long-format video environment + repo + model files."""
    script = root / "presets" / "extra_env" / "hiar_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    args = ["-u", str(script), "--root", str(root)]
    return (str(py), args, root)


def _run_ltx23(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install/update LTX 2.3 FP16 using the portable FrameVision installer.

    Env target: <root>/environments/.ltx23/
    Models/repo target: <root>/models/ltx23/
    The installer repairs missing pieces and reuses existing downloads by default.
    """
    script = root / "presets" / "extra_env" / "ltx23_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--root", str(root), "--repair", "--model-variant", "fp16"], root)


def _run_ltx23_fp8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install/update LTX 2.3 FP8 checkpoint using the shared portable LTX env/repo.

    Env target: <root>/environments/.ltx23/
    FP8 checkpoint target: <root>/models/ltx23/fp8/
    Shared pieces: repo, Gemma text encoder, FFmpeg tools, and acceleration kernels.
    """
    script = root / "presets" / "extra_env" / "ltx23_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--root", str(root), "--repair", "--model-variant", "fp8"], root)


# (HeartMula optional installs removed)

def _run_whisper(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install Whisper (Faster-Whisper) env + download default model."""
    script = root / "presets" / "extra_env" / "whisper_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)

def _run_qwen3tts(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install Qwen3-TTS environment + base model(s)."""
    script = root / "presets" / "extra_env" / "install_qwentts.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_dotstts(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install dots.tts conda environment + repo + turbo model."""
    script = root / "presets" / "extra_env" / "dottstts_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--root", str(root)], root)


def _run_qwen3tts_models(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Download extra Qwen3-TTS models (uses the existing Qwen3-TTS env)."""
    script = root / "presets" / "extra_env" / "download_qwentts_models.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)



def _run_qwen3tts_flashattn(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install FlashAttention for Qwen3-TTS (optional speed-up)."""
    script = root / "presets" / "extra_env" / "install_flashattn_optional.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)



def _run_qwen2512_env(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """
    Legacy Qwen-Image-2512 Python env installer.
    Not shown in Optional Installs anymore; Qwen GGUF only needs model files + the Qwen sd-cli bin.
    """
    script = root / "presets" / "extra_env" / "qwen2512_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _find_qwen2512_downloader_script(root: Path) -> Optional[Path]:
    """
    Try common locations for the Qwen2512 downloader script.
    """
    candidates = [
        root / "scripts" / "qwen2512_download.py",
        root / "qwen2512_download.py",
        root / "presets" / "extra_env" / "qwen2512_download.py",
        root / "presets" / "extra_env" / "qwen2512_download_quants.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_qwen2512_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Qwen-Image-2512 GGUF model downloader.

    Note: This runs inside FrameVision's .venv (same as the Z-image GGUF downloader),
    so it does NOT require re-installing the Qwen2512 env.
    """
    script = _find_qwen2512_downloader_script(root)
    if script is None:
        return None

    py = _venv_python(root)
    if not py:
        return None

    target = (root / "models" / "Qwen-Image-2512 GGUF").resolve()

    # Support both the newer quant-selectable downloader and the older baseline downloader.
    # Newer: supports --qwen-quant (q2/q3/q4/q5/q6/q8)
    # Older: downloads the default bundle without quant selection.
    try:
        src = script.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        src = ""

    args = ["-u", str(script)]
    if "--qwen-quant" in src or "qwen-quant" in src:
        # Newer script signature (preferred)
        args += ["--qwen-quant", quant]
    # Common args (existing in the original downloader)
    args += ["--root", str(root), "--models-dir", str(target)]

    return (str(py), args, root)


def _run_qwen2512_q2(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q2")


def _run_qwen2512_q3(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q3")


def _run_qwen2512_q4(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q4")


def _run_qwen2512_q5(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q5")


def _run_qwen2512_q6(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q6")


def _run_qwen2512_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2512_gguf(root, "q8")




def _find_qwen2511_downloader_script(root: Path) -> Optional[Path]:
    """Try common locations for the Qwen2511 (Image Edit) GGUF downloader script."""
    candidates = [
        root / "presets" / "extra_env" / "qwen2511_download.py",
        root / "scripts" / "qwen2511_download.py",
        root / "qwen2511_download.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_qwen2511_gguf(root: Path, variant: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Qwen-Image-Edit-2511 GGUF model downloader.

    NOTE: Qwen2511 and Qwen2512 share the same stable-diffusion.cpp bin folder,
    but they do not require a private Python environment for downloads/generation.
    """
    script = _find_qwen2511_downloader_script(root)
    if script is None:
        return None

    py = _venv_python(root)
    if not py:
        return None

    target = (root / "models" / "qwen2511gguf").resolve()

    args = [
        "-u",
        str(script),
        "--variants",
        variant,
        "--root",
        str(root),
        "--models-dir",
        str(target),
    ]
    return (str(py), args, root)


def _run_qwen2511_q3(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2511_gguf(root, "Q3_K_S")


def _run_qwen2511_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2511_gguf(root, "Q4_K_M")


def _run_qwen2511_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2511_gguf(root, "Q5_K_M")


def _run_qwen2511_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2511_gguf(root, "Q6_K")


def _run_qwen2511_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_qwen2511_gguf(root, "Q8_0")



def _find_ideogram4_gguf_installer(root: Path) -> Optional[Path]:
    """Return the Ideogram 4 GGUF installer script when present."""
    candidates = [
        root / "presets" / "extra_env" / "ideogram4_gguf_install.py",
        root / "scripts" / "ideogram4_gguf_install.py",
        root / "ideogram4_gguf_install.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_ideogram4_gguf(root: Path, component: str, conditional_quant: str = "Q5_0", unconditional_quant: str = "Q2_K") -> Optional[Tuple[str, List[str], Path]]:
    """Install Ideogram 4 GGUF runtime/shared files or one selected GGUF model."""
    script = _find_ideogram4_gguf_installer(root)
    if script is None:
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    args = [
        "-u",
        str(script),
        "--component",
        str(component),
        "--conditional-quant",
        str(conditional_quant),
        "--unconditional-quant",
        str(unconditional_quant),
    ]
    return (str(py), args, root)


def _run_ideogram4_runtime(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "runtime")


def _run_ideogram4_cond_q5(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "conditional", conditional_quant="Q5_0")


def _run_ideogram4_cond_q6(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "conditional", conditional_quant="Q6_K")


def _run_ideogram4_cond_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "conditional", conditional_quant="Q8_0")


def _run_ideogram4_uncond_q2(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "unconditional", unconditional_quant="Q2_K")


def _run_ideogram4_uncond_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_ideogram4_gguf(root, "unconditional", unconditional_quant="Q8_0")




def _run_hidream_edit(root: Path, models: str) -> Optional[Tuple[str, List[str], Path]]:
    """Install HiDream environment/repo and selected BF16/FP8 model(s)."""
    script = root / "presets" / "extra_env" / "hidream_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    # hidream_install.py is responsible for reusing environments/.hidream_dev,
    # the official repo, and already-downloaded model files. These switches keep
    # optional installs non-interactive while still allowing model-only additions.
    args = ["-u", str(script), "--models", str(models), "--no-prompt"]
    return (str(py), args, root)


def _run_hidream_edit_base(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "base")


def _run_hidream_edit_dev(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "dev")


def _run_hidream_edit_dev_2604_bf16(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "dev_2604")


def _run_hidream_edit_both(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "both")


def _run_hidream_edit_base_fp8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "base_fp8")


def _run_hidream_edit_dev_fp8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "dev_fp8")


def _run_hidream_edit_both_fp8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "both_fp8")


def _run_hidream_edit_all(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_hidream_edit(root, "all")

def _run_lens_turbo_u4(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install the Lens Turbo U4 environment/repo and cache required runtime kernels."""
    script = root / "presets" / "extra_env" / "lens_turbo_u4.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script)], root)


def _run_lens_turbo_u4_model_cache(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Download/cache Lens Turbo U4 model files now, instead of waiting for first use."""
    script = root / "presets" / "extra_env" / "lens_turbo_u4.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    return (str(py), ["-u", str(script), "--download-model"], root)


def _run_sdxl_juggernaut(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "download_sd_models.py"
    py = _venv_python(root)
    if (not script.exists()) or (py is None):
        return None
    return (str(py), ["-u", str(script)], root / "scripts")


def _run_background_removal_models(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "background_download.py"
    py = _venv_python(root)
    if (not script.exists()) or (py is None):
        return None
    # Downloads only ONNX model files into models/bg.
    # No extra env is installed here; background removal uses the main FrameVision env.
    return (str(py), ["-u", str(script)], root)


# Backwards-compatible internal name; the install now downloads background-removal models only.
_run_background_remover_inpainter = _run_background_removal_models


def _run_sdxl_inpaint_env(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """
    Install the SDXL inpaint environment used by the Background Remover + SDXL Inpaint tool.

    Expected to create: <root>/environments/.sdxl_inpaint/
    """
    script = root / "presets" / "extra_env" / "sdxl_inpaint_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_firered_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Download FireRed Image Edit 1.1 GGUF plus the shared VAE, text encoder and mmproj.

    Files are stored flat in:
      <root>/models/FireRed-Image-Edit-1.1/
    """
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None

    target = (root / "models" / "FireRed-Image-Edit-1.1").resolve()

    downloader = r"""
import sys
import time
import shutil
import urllib.request
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
shared = root / "models" / "shared"
quant = sys.argv[3].strip()

GGUF_REPO = "https://huggingface.co/vantagewithai/FireRed-Image-Edit-1.1-GGUF/resolve/main"
TEXT_ENCODER_URL = "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf?download=1"
MMPROJ_URL = "https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/resolve/main/mmproj/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf?download=1"
VAE_URL = "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=1"

files = [
    (f"{GGUF_REPO}/FireRed-Image-Edit-1.1-{quant}.gguf?download=1", target / f"FireRed-Image-Edit-1.1-{quant}.gguf"),
    (VAE_URL, target / "qwen_image_vae.safetensors"),
    (MMPROJ_URL, shared / "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"),
    (TEXT_ENCODER_URL, shared / "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"),
]

target.mkdir(parents=True, exist_ok=True)
shared.mkdir(parents=True, exist_ok=True)
headers = {"User-Agent": "FrameVision Optional Installs/1.0"}

def remote_size(url: str):
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None

def fmt_size(num):
    if num is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(num)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{num} B"

def download(url: str, dest: Path) -> None:
    size = remote_size(url)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f"[OK] already present: {dest.name} ({fmt_size(size)})", flush=True)
        return
    if dest.exists() and size is None and dest.stat().st_size > 0:
        print(f"[OK] already present: {dest.name}", flush=True)
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"[DL] {dest.name} -> {dest}", flush=True)
    if size is not None:
        print(f"[INFO] expected size: {fmt_size(size)}", flush=True)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as src, open(tmp, "wb") as out:
        total = 0
        last = 0.0
        while True:
            chunk = src.read(1024 * 1024 * 4)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            now = time.time()
            if now - last >= 1.2:
                if size:
                    pct = (100.0 * total / size)
                    print(f"[DL] {dest.name}: {pct:.1f}% ({fmt_size(total)} / {fmt_size(size)})", flush=True)
                else:
                    print(f"[DL] {dest.name}: {fmt_size(total)}", flush=True)
                last = now

    if size is not None and tmp.stat().st_size != size:
        raise RuntimeError(f"Downloaded size mismatch for {dest.name}: got {tmp.stat().st_size}, expected {size}")

    if dest.exists():
        dest.unlink()
    shutil.move(str(tmp), str(dest))
    print(f"[OK] downloaded: {dest.name}", flush=True)

print("[INFO] FireRed 1.1 GGUF download started", flush=True)
print(f"[INFO] target folder: {target}", flush=True)
print(f"[INFO] selected quant: {quant}", flush=True)
for url, dest in files:
    download(url, dest)
print("[DONE] FireRed 1.1 GGUF files are ready.", flush=True)
"""

    args = ["-u", "-c", downloader, str(root), str(target), str(quant)]
    return (str(py), args, root)


def _run_firered_q3km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q3_K_M")


def _run_firered_q3ks(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q3_K_S")


def _run_firered_q4_0(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q4_0")


def _run_firered_q4_1(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q4_1")


def _run_firered_q4ks(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q4_K_S")


def _run_firered_q4km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q4_K_M")


def _run_firered_q5_0(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q5_0")


def _run_firered_q5_1(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q5_1")


def _run_firered_q5ks(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q5_K_S")


def _run_firered_q5km(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q5_K_M")


def _run_firered_q6k(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q6_K")


def _run_firered_q8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_firered_gguf(root, "Q8_0")



def _default_installs() -> List[OptionalInstall]:
    # Titles/descriptions copied from install_menu.bat "extra options" page.
    return [
        OptionalInstall(
            key="qwen3tts",
            title="Qwen3-TTS-12Hz-1.7B-CustomVoice",
            description="Qwen 3 Text to speech. Installs environment + Qwen TTS Custom Voice model.",
            runner=_run_qwen3tts,
        ),
        OptionalInstall(
            key="dotstts",
            title="dots.tts Text to Speech",
            description=(
                "Installs the dots.tts conda environment, repo and turbo model. "
                "Total disk space is about 16 GB. Needs less than 8 GB VRAM for small text fragments, "
                "but VRAM use can rise quickly when adding too much text."
            ),
            runner=_run_dotstts,
        ),
        OptionalInstall(
            key="qwen3tts_models",
            title="installs only Qwen TTS Custom Voice model. More models can be downloaded inside the tool.",
            description="If the environment is missing, it will be installed first.",
            runner=_run_qwen3tts_models,
        ),

        
        OptionalInstall(
            key="qwen3tts_flashattn",
            title="Install flash attention",
            description="optional but advised for faster generation, credits to Get Going Fast (https://www.youtube.com/@cognibuild)  for the help",
            runner=_run_qwen3tts_flashattn,
        ),
OptionalInstall(
            key="wan22_turbo",
            title="WAN 2.2 Turbo 5B (installs normal + Turbo)",
            description=(
                "Installs both the full original WAN 2.2 5B model/repo, including the large diffusion safetensor shards, and the Turbo files needed for the 4-step Turbo workflow. "
                "Text/image/video to video with extender, creates up to 10 second long videos. "
                "The normal WAN 2.2 path is included too; it is slower, but can still use the 8-step LoRA to become a little faster while keeping original FP16 quality."
            ),
            runner=_run_wan22_turbo,
        ),
        OptionalInstall(
            key="ltx23",
            title="LTX 2.3 FP16 Text/Image to Video (For RTX 30XX series)",
            description=(
                "Installs or repairs the portable LTX 2.3 FP16 environment, official repo, distilled 1.1 model, text encoder, FFmpeg tools, and optional acceleration kernels. "
                "Reuses existing files when possible and stores everything inside FrameVision: environments/.ltx23 and models/ltx23."
            ),
            runner=_run_ltx23,
        ),
        OptionalInstall(
            key="ltx23_fp8",
            title="LTX 2.3 FP8 Text/Image/Video to Video (RTX 40XX and 50XX only)",
            description=(
                "Installs or repairs the shared portable LTX 2.3 environment, official repo, FP8 distilled model, text encoder, FFmpeg tools, and optional acceleration kernels. "
                "Shares environments/.ltx23 and models/ltx23 with the FP16 install, then stores the FP8 checkpoint in models/ltx23/fp8."
            ),
            runner=_run_ltx23_fp8,
        ),
        OptionalInstall(
            key="hunyuan15",
            title="HunyuanVideo 1.5, text/image/video to Video with extender",
            description="Installs with Distilled 8 step 480p model, other models will be downloaded directly in the app at first time use",
            runner=_run_hunyuan15,
        ),

        OptionalInstall(
            key="bernini_r_1p3b",
            title="Bernini-R 1.3B",
            description=(
                "1.3B is the Low(er) VRAM version of Bernini 14B. "
                "It can do small edits in video but no complete inpainting. "
                "It can also be used for image editing (fast and low VRAM), text to image, and text to video."
            ),
            runner=_run_bernini_r_1p3b,
        ),
        OptionalInstall(
            key="hiar",
            title="Hiar wan 2.1 long format Video",
            description="Experimental model for long consistency, about 20 gigabyte for model + repo. Enabling this optional install checks existing files and installs/downloads everything needed to get started.",
            runner=_run_hiar,
        ),
        OptionalInstall(
            key="lens_turbo_u4",
            title="Lens Turbo U4 Text to Image environment install",
            description="Installs the Lens Turbo U4 environment, Microsoft Lens repo, and caches the required runtime kernels in the FrameVision models/lens folders. Model files download automatically on first use unless you run the cache option below.",
            runner=_run_lens_turbo_u4,
        ),
        OptionalInstall(
            key="lens_turbo_u4_model_cache",
            title="Lens Turbo U4 download model cache",
            description="Optional extra step: downloads the Lens Turbo U4 model files into FrameVision's portable models/lens Hugging Face cache now, so first use can run offline/faster. Also refreshes the kernel cache.",
            runner=_run_lens_turbo_u4_model_cache,
        ),
        OptionalInstall(
            key="hidream_edit_base",
            title="HiDream BF16 (Base / Full)",
            description="HiDream model. Installs/reuses environments/.hidream_dev, the official repo, and downloads only the Base / Full BF16 model when missing.",
            runner=_run_hidream_edit_base,
        ),
        OptionalInstall(
            key="hidream_edit_dev",
            title="HiDream BF16 (Dev)",
            description="HiDream model. Installs/reuses environments/.hidream_dev, the official repo, and downloads only the Dev BF16 model when missing.",
            runner=_run_hidream_edit_dev,
        ),
        OptionalInstall(
            key="hidream_edit_dev_2604_bf16",
            title="HiDream BF16 (Dev 2604)",
            description="Updated Dev 2604 BF16 model. Installs/reuses environments/.hidream_dev and the HiDream repo, then downloads the separate Dev 2604 BF16 folder without replacing older HiDream models.",
            runner=_run_hidream_edit_dev_2604_bf16,
        ),
        OptionalInstall(
            key="hidream_edit_both",
            title="HiDream BF16 (Base + Dev)",
            description="Installs/reuses the HiDream environment/repo and downloads both Base / Full BF16 and Dev BF16 models, skipping files that already exist.",
            runner=_run_hidream_edit_both,
        ),
        OptionalInstall(
            key="hidream_edit_base_fp8",
            title="HiDream FP8 (Base / Full)",
            description="HiDream model. Installs/reuses environments/.hidream_dev, the official repo, and downloads only the Base / Full FP8 model when missing.",
            runner=_run_hidream_edit_base_fp8,
        ),
        OptionalInstall(
            key="hidream_edit_dev_fp8",
            title="HiDream FP8 (Dev)",
            description="HiDream model. Installs/reuses environments/.hidream_dev, the official repo, and downloads only the Dev FP8 model when missing.",
            runner=_run_hidream_edit_dev_fp8,
        ),
        OptionalInstall(
            key="hidream_edit_both_fp8",
            title="HiDream FP8 (Base + Dev)",
            description="Installs/reuses the HiDream environment/repo and downloads both Base / Full FP8 and Dev FP8 models, skipping files that already exist.",
            runner=_run_hidream_edit_both_fp8,
        ),
        OptionalInstall(
            key="hidream_edit_all",
            title="HiDream BF16 + FP8 (All)",
            description="Installs/reuses the HiDream environment/repo and downloads Base / Full + Dev for both BF16 and FP8 plus Dev 2604 BF16, skipping files that already exist.",
            runner=_run_hidream_edit_all,
        ),
        OptionalInstall(
            key="qwen2512_q2",
            title="Qwen2512 GGUF (Q2)",
            description="Lowest VRAM / smallest. Fastest download, lowest quality.",
            runner=_run_qwen2512_q2,
        ),
        OptionalInstall(
            key="qwen2512_q3",
            title="Qwen2512 GGUF (Q3)",
            description="Low VRAM / small. Better than Q2, still lightweight.",
            runner=_run_qwen2512_q3,
        ),
        OptionalInstall(
            key="qwen2512_q4",
            title="Qwen2512 GGUF (Q4)",
            description="Balanced size/quality. Recommended starting point.",
            runner=_run_qwen2512_q4,
        ),
        OptionalInstall(
            key="qwen2512_q5",
            title="Qwen2512 GGUF (Q5)",
            description="Higher quality / larger download.",
            runner=_run_qwen2512_q5,
        ),
        OptionalInstall(
            key="qwen2512_q6",
            title="Qwen2512 GGUF (Q6)",
            description="High quality / larger. Needs more VRAM.",
            runner=_run_qwen2512_q6,
        ),
        OptionalInstall(
            key="qwen2512_q8",
            title="Qwen2512 GGUF (Q8)",
            description="Best quality / largest GGUF. Needs the most VRAM.",
            runner=_run_qwen2512_q8,
        ),
        OptionalInstall(
            key="qwen2511_q3",
            title="Qwen2511 Image Edit GGUF (Q3_K_S)",
            description="Low VRAM / Don't expect perfect results.",
            runner=_run_qwen2511_q3,
        ),
        OptionalInstall(
            key="qwen2511_q4km",
            title="Qwen2511 Image Edit GGUF (Q4_K_M)",
            description="runs in 10-12 gig vram, still not great quality.",
            runner=_run_qwen2511_q4km,
        ),
        OptionalInstall(
            key="qwen2511_q5km",
            title="Qwen2511 Image Edit GGUF (Q5_K_M)",
            description="16 gig vram advised, quality is better.",
            runner=_run_qwen2511_q5km,
        ),
        OptionalInstall(
            key="qwen2511_q6k",
            title="Qwen2511 Image Edit GGUF (Q6_K)",
            description="High quality / 24 gog vram without offloading",
            runner=_run_qwen2511_q6k,
        ),
        OptionalInstall(
            key="qwen2511_q8",
            title="Qwen2511 Image Edit GGUF (Q8_0)",
            description="Quality almost identical to original model.",
            runner=_run_qwen2511_q8,
        ),

        OptionalInstall(
            key="ideogram4_runtime",
            title="Ideogram 4 GGUF runtime / shared files",
            description="Installs or refreshes sd-cli plus the shared Qwen3-VL text encoder and Flux2 VAE used by Ideogram 4 GGUF. Model buttons below also install these shared files if missing.",
            runner=_run_ideogram4_runtime,
        ),
        OptionalInstall(
            key="ideogram4_cond_q5",
            title="Ideogram 4 conditional GGUF (Q5_0)",
            description="Conditional/main Ideogram 4 model. Balanced quality and size. Downloads into models/ideogram4_gguf and reuses existing shared files.",
            runner=_run_ideogram4_cond_q5,
        ),
        OptionalInstall(
            key="ideogram4_cond_q6",
            title="Ideogram 4 conditional GGUF (Q6_K)",
            description="Conditional/main Ideogram 4 model. Higher quality and larger than Q5.",
            runner=_run_ideogram4_cond_q6,
        ),
        OptionalInstall(
            key="ideogram4_cond_q8",
            title="Ideogram 4 conditional GGUF (Q8_0)",
            description="Conditional/main Ideogram 4 model. Best quality / largest conditional GGUF.",
            runner=_run_ideogram4_cond_q8,
        ),
        OptionalInstall(
            key="ideogram4_uncond_q2",
            title="Ideogram 4 unconditional GGUF (Q2_K)",
            description="Unconditional Ideogram 4 model. Smaller low-VRAM companion file.",
            runner=_run_ideogram4_uncond_q2,
        ),
        OptionalInstall(
            key="ideogram4_uncond_q8",
            title="Ideogram 4 unconditional GGUF (Q8_0)",
            description="Unconditional Ideogram 4 model. Best quality / largest unconditional companion file.",
            runner=_run_ideogram4_uncond_q8,
        ),

        OptionalInstall(
            key="firered_q3km",
            title="FireRed 1.1 Edit 20B GGUF (Q3_K_M)",
            description="Lowest VRAM / smallest. Downloads the main GGUF plus shared VAE, text encoder and mmproj into models/FireRed-Image-Edit-1.1/.",
            runner=_run_firered_q3km,
        ),
        OptionalInstall(
            key="firered_q3ks",
            title="FireRed 1.1 Edit 20B GGUF (Q3_K_S)",
            description="Very low VRAM / smaller. Downloads the main GGUF plus shared VAE, text encoder and mmproj.",
            runner=_run_firered_q3ks,
        ),
        OptionalInstall(
            key="firered_q4_0",
            title="FireRed 1.1 Edit 20B GGUF (Q4_0)",
            description="Balanced lower-memory option. Downloads the main GGUF plus shared VAE, text encoder and mmproj.",
            runner=_run_firered_q4_0,
        ),
        OptionalInstall(
            key="firered_q4_1",
            title="FireRed 1.1 Edit 20B GGUF (Q4_1)",
            description="Balanced lower-memory option with slightly more quality than Q4_0.",
            runner=_run_firered_q4_1,
        ),
        OptionalInstall(
            key="firered_q4ks",
            title="FireRed 1.1 Edit 20B GGUF (Q4_K_S)",
            description="Good starting point for smaller VRAM systems.",
            runner=_run_firered_q4ks,
        ),
        OptionalInstall(
            key="firered_q4km",
            title="FireRed 1.1 Edit 20B GGUF (Q4_K_M)",
            description="Recommended balanced choice. High quality image editor, on average about 20 GB disk space needed once shared files are included.",
            runner=_run_firered_q4km,
        ),
        OptionalInstall(
            key="firered_q5_0",
            title="FireRed 1.1 Edit 20B GGUF (Q5_0)",
            description="Higher quality / larger download.",
            runner=_run_firered_q5_0,
        ),
        OptionalInstall(
            key="firered_q5_1",
            title="FireRed 1.1 Edit 20B GGUF (Q5_1)",
            description="Higher quality / larger download than Q5_0.",
            runner=_run_firered_q5_1,
        ),
        OptionalInstall(
            key="firered_q5ks",
            title="FireRed 1.1 Edit 20B GGUF (Q5_K_S)",
            description="High quality / larger. Downloads the main GGUF plus shared VAE, text encoder and mmproj.",
            runner=_run_firered_q5ks,
        ),
        OptionalInstall(
            key="firered_q5km",
            title="FireRed 1.1 Edit 20B GGUF (Q5_K_M)",
            description="High quality balanced choice. Close to the example layout from your screenshot.",
            runner=_run_firered_q5km,
        ),
        OptionalInstall(
            key="firered_q6k",
            title="FireRed 1.1 Edit 20B GGUF (Q6_K)",
            description="Very high quality / larger. Needs more VRAM and disk.",
            runner=_run_firered_q6k,
        ),
        OptionalInstall(
            key="firered_q8",
            title="FireRed 1.1 Edit 20B GGUF (Q8_0)",
            description="Best quality / largest GGUF. Main file alone is about 21.8 GB.",
            runner=_run_firered_q8,
        ),

        OptionalInstall(
            key="zimage",
            title="Z-image Turbo Text to Image",
            description="Shared image-model environment. For best quality install the full 16-bit model; Auto will pick BF16/FP16. If you have less than 16GB VRAM, use the GGUF options (Q4–Q8).",
            runner=_run_zimage,
        ),
        OptionalInstall(
            key="zimage_fp16",
            title="Full 16-bit model (Diffusers Auto BF16/FP16)",
            description="VRAM: 16GB recommended. Downloads the official Z-Image Turbo Diffusers folder layout (model_index.json + repo folders). Runtime automatically uses BF16 when the GPU/PyTorch setup supports it; otherwise FP16. Does not download GGUF.",
            runner=_run_zimage_fp16,
        ),
        OptionalInstall(
            key="chroma",
            title="SPARK.Chroma Text to Image",
            description="Installs or repairs SPARK.Chroma. Reuses the shared image-model environment when available and stores the model in models/chroma/SPARK.Chroma_v1.",
            runner=_run_chroma,
        ),
        OptionalInstall(
            key="zimage_gguf4",
            title="Z-Image Turbo GGUF (Q4_0)",
            description="VRAM: ~6GB+. Smallest/fastest.",
            runner=_run_zimage_gguf4,
        ),
        OptionalInstall(
            key="zimage_gguf5",
            title="Z-Image Turbo GGUF (Q5_0)",
            description="VRAM: ~8GB+. Balanced size/quality.",
            runner=_run_zimage_gguf5,
        ),
        OptionalInstall(
            key="zimage_gguf6",
            title="Z-Image Turbo GGUF (Q6_K)",
            description="VRAM: ~10–12GB+. Higher quality.",
            runner=_run_zimage_gguf6,
        ),
        OptionalInstall(
            key="zimage_gguf8",
            title="Z-Image Turbo GGUF (Q8_0)",
            description="VRAM: ~12–16GB+ for 1080p. Best quality / largest. almost same like fp16 but a little faster",
            runner=_run_zimage_gguf8,
        ),
        
        OptionalInstall(
            key="klein4b_unet_q2k",
            title="Flux Klein edit 4B UNet GGUF (Q2_K)",
            description="Smallest/fastest. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q2k,
        ),
        OptionalInstall(
            key="klein4b_unet_q3kl",
            title="Flux Klein edit 4B UNet GGUF (Q3_K_L)",
            description="Low VRAM / small. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q3kl,
        ),
        OptionalInstall(
            key="klein4b_unet_q4km",
            title="Flux Klein edit 4B UNet GGUF (Q4_K_M)",
            description="Balanced size/quality. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q4km,
        ),
        OptionalInstall(
            key="klein4b_unet_q5km",
            title="Flux Klein edit 4B UNet GGUF (Q5_K_M)",
            description="Higher quality / larger download. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q5km,
        ),
        OptionalInstall(
            key="klein4b_unet_q6k",
            title="Flux Klein edit 4B UNet GGUF (Q6_K)",
            description="High quality / larger. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q6k,
        ),
        OptionalInstall(
            key="klein4b_unet_q8",
            title="Flux Klein edit 4B UNet GGUF (Q8_0)",
            description="Best quality / largest. Auto-downloads the required VAE if missing.",
            runner=_run_klein4b_unet_q8,
        ),

        OptionalInstall(
            key="klein4b_te_comfy",
            title="Flux Klein edit 4B Text Encoder (Comfy-Org safetensors)",
            description="Recommended default text encoder download (safetensors).",
            runner=_run_klein4b_te_comfy,
        ),
        OptionalInstall(
            key="klein4b_te_qwen_gguf_q4km",
            title="Flux Klein edit 4B Text Encoder (Qwen3-4B GGUF Q4_K_M)",
            description="Lower VRAM text encoder (GGUF).",
            runner=_run_klein4b_te_qwen_gguf_q4km,
        ),
        OptionalInstall(
            key="klein4b_te_qwen_gguf_q5km",
            title="Flux Klein edit 4B Text Encoder (Qwen3-4B GGUF Q5_K_M)",
            description="Balanced GGUF text encoder.",
            runner=_run_klein4b_te_qwen_gguf_q5km,
        ),
        OptionalInstall(
            key="klein4b_te_qwen_gguf_q8",
            title="Flux Klein edit 4B Text Encoder (Qwen3-4B GGUF Q8_0)",
            description="Best quality / largest GGUF text encoder.",
            runner=_run_klein4b_te_qwen_gguf_q8,
        ),
        OptionalInstall(
            key="klein4b_te_cordux",
            title="Flux Klein edit 4B Text Encoder (Cordux uncensored - gated)",
            description="Gated Hugging Face repo. You may need to accept terms and set HF_TOKEN.",
            runner=_run_klein4b_te_cordux,
        ),

OptionalInstall(
    key="klein9b_unet_q2k",
    title="Flux Klein edit 9B UNet GGUF (Q2_K)",
    description="Smallest setup. Needs at least ~8GB VRAM for 1024×1024. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q2k,
),
OptionalInstall(
    key="klein9b_unet_q3km",
    title="Flux Klein edit 9B UNet GGUF (Q3_K_M)",
    description="Low VRAM / small. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q3km,
),
OptionalInstall(
    key="klein9b_unet_q4km",
    title="Flux Klein edit 9B UNet GGUF (Q4_K_M)",
    description="Balanced size/quality. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q4km,
),
OptionalInstall(
    key="klein9b_unet_q5km",
    title="Flux Klein edit 9B UNet GGUF (Q5_K_M)",
    description="Higher quality / larger download. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q5km,
),
OptionalInstall(
    key="klein9b_unet_q6k",
    title="Flux Klein edit 9B UNet GGUF (Q6_K)",
    description="High quality / larger. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q6k,
),
OptionalInstall(
    key="klein9b_unet_q8",
    title="Flux Klein edit 9B UNet GGUF (Q8_0)",
    description="Best quality / largest. Needs about ~22GB VRAM for a 1024×1024 edit. Auto-downloads the required VAE if missing.",
    runner=_run_klein9b_unet_q8,
),

OptionalInstall(
    key="klein9b_te_q2k",
    title="Flux Klein edit 9B Text Encoder (Qwen3-8B GGUF Q2_K)",
    description="Smallest/fastest GGUF text encoder.",
    runner=_run_klein9b_te_q2k,
),
OptionalInstall(
    key="klein9b_te_q4km",
    title="Flux Klein edit 9B Text Encoder (Qwen3-8B GGUF Q4_K_M)",
    description="Balanced GGUF text encoder.",
    runner=_run_klein9b_te_q4km,
),
OptionalInstall(
    key="klein9b_te_q5km",
    title="Flux Klein edit 9B Text Encoder (Qwen3-8B GGUF Q5_K_M)",
    description="Higher quality GGUF text encoder.",
    runner=_run_klein9b_te_q5km,
),
OptionalInstall(
    key="klein9b_te_q6k",
    title="Flux Klein edit 9B Text Encoder (Qwen3-8B GGUF Q6_K)",
    description="High quality GGUF text encoder.",
    runner=_run_klein9b_te_q6k,
),
OptionalInstall(
    key="klein9b_te_q8",
    title="Flux Klein edit 9B Text Encoder (Qwen3-8B GGUF Q8_0)",
    description="Best quality / largest GGUF text encoder.",
    runner=_run_klein9b_te_q8,
),
OptionalInstall(
            key="ace15",
            title="Ace Step 1.5 Music Creation",
            description=(
                "Downloads and installs the Ace Step 1.5 repo with the turbo model. "
                "All other models can be used and will be downloaded at first use."
            ),
            runner=_run_ace15,
        ),
        OptionalInstall(
            key="gfpgan",
            title="GFPGAN Face restorer/enhancer",
            description="VRAM: optional (CPU works). Download <400MB; env uses ~5GiB disk.",
            runner=_run_gfpgan,
        ),
        OptionalInstall(
            key="seedvr2_env",
            title="SeedVR2 (GGUF) environment install",
            description="Installs the SeedVR2 GGUF environment + dependencies. Creates: environments/.seedvr2/",
            runner=_run_seedvr2_env,
        ),

        OptionalInstall(
            key="seedvr2_gguf_q4",
            title="SeedVR2 3B GGUF (Q4_K_M)",
            description="Balanced size/quality. Recommended starting point. If the env is missing, it will be installed first.",
            runner=_run_seedvr2_q4,
        ),

        OptionalInstall(
            key="seedvr2_gguf_q8",
            title="SeedVR2 3B GGUF (Q8_0)",
            description="Best quality / largest GGUF. If the env is missing, it will be installed first.",
            runner=_run_seedvr2_q8,
        ),
        OptionalInstall(
            key="whisper",
            title="Whisper (voice to text / subtitles / transcript)",
            description="Offline speech-to-text with Faster-Whisper. Runs on any PC. Installs environment + downloads the default model (~1.7 GB).",
            runner=_run_whisper,
        ),
                OptionalInstall(
            key="bgrem_inpaint",
            title="Background removal models (no env)",
            description="Downloads only the MODNet and BiRefNet ONNX background-removal models into models/bg. No extra Python environment is installed or required for this option.",
            runner=_run_background_removal_models,
        ),
        OptionalInstall(
            key="sdxl_inpaint_env",
            title="SDXL Inpainter environment",
            description="Separate optional install for the SDXL/Juggernaut inpaint tool. Background removal keeps working without this environment; install this only when you want to use the SDXL inpainter workflow.",
            runner=_run_sdxl_inpaint_env,
        ),
OptionalInstall(
            key="sdxljugg",
            title="Juggernaut XL V9. Model for SDXL Text to image",
            description="VRAM: 6–12GB. Disk: ~6.5GB. SDXL model for txt2img. find more (CyberRealisticXL, DreamshaperXL, EpicRealismXL,...) at https://civitai.com/",
            runner=_run_sdxl_juggernaut,
        ),
    ]


# -----------------------------
# Environment folders (optional installs)
# -----------------------------
# These folders are safe to delete when re-installing an optional component:
# they contain only the Python environment, not the downloaded model weights.
_ENV_DIR_BY_KEY = {
    "qwen3tts": Path("environments") / ".qwen3tts",
    "dotstts": Path("environments") / ".dots_tts",
    "whisper": Path("environments") / ".whisper",
    "ace15": Path("environments") / ".ace_15",
    "hunyuan15": Path("environments") / ".hunyuan15_official",
    "hiar": Path("environments") / ".hiar",
    "ltx23": Path("environments") / ".ltx23",
    "ltx23_fp8": Path("environments") / ".ltx23",
    "lens_turbo_u4": Path("environments") / ".lens_turbo_u4",
    "gfpgan": Path("models") / "gfpgan" / ".GFPGAN",
    "seedvr2_env": Path("environments") / ".seedvr2",
    "wan22_turbo": Path("environments") / ".wan22_i2v",
    "zimage": Path("environments") / ".images_models",
    "chroma": Path("environments") / ".images_models",
    "hidream_edit_base": Path("environments") / ".hidream_dev",
    "hidream_edit_dev": Path("environments") / ".hidream_dev",
    "hidream_edit_dev_2604_bf16": Path("environments") / ".hidream_dev",
    "hidream_edit_both": Path("environments") / ".hidream_dev",
    "hidream_edit_base_fp8": Path("environments") / ".hidream_dev",
    "hidream_edit_dev_fp8": Path("environments") / ".hidream_dev",
    "hidream_edit_both_fp8": Path("environments") / ".hidream_dev",
    "hidream_edit_all": Path("environments") / ".hidream_dev",
    "sdxl_inpaint_env": Path("environments") / ".sdxl_inpaint",
    # Not on the UI list yet, but reserved for future use.
    "comfui": Path(".comfui_env"),
}

# -----------------------------
# UI
# -----------------------------

class _OptionRow(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, opt: OptionalInstall, parent: Optional[QtWidgets.QWidget] = None, indent: int = 0) -> None:
        super().__init__(parent)
        self.opt = opt

        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.stateChanged.connect(lambda s: self.toggled.emit(s == QtCore.Qt.Checked))

        title = QtWidgets.QLabel(opt.title)
        f = title.font()
        f.setBold(True)
        title.setFont(f)

        desc = QtWidgets.QLabel(opt.description)
        desc.setWordWrap(True)
        desc.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)
        left.addWidget(title)
        left.addWidget(desc)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10 + max(0, int(indent)), 8, 10, 8)
        lay.setSpacing(10)
        lay.addWidget(self.checkbox, 0, QtCore.Qt.AlignTop)
        lay.addLayout(left, 1)

        self.setObjectName("OptionalInstallRow")
        self.setStyleSheet("""
        QWidget#OptionalInstallRow {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
        }
        """)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)


class _ToastWidget(QtWidgets.QFrame):
    """Lightweight, non-modal toast popup shown inside the dialog."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("ToastWidget")
        self.setVisible(False)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self._lbl = QtWidgets.QLabel("")
        self._lbl.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(self._lbl)

        self.setStyleSheet("""
        QFrame#ToastWidget {
            background: rgba(0, 0, 0, 0.82);
            color: white;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
        }
        """)

    def show_message(self, msg: str, msec: int = 2800) -> None:
        self._lbl.setText(msg)
        self.adjustSize()

        # Bottom-right corner with padding.
        parent = self.parentWidget()
        if parent is not None:
            pad = 18
            x = max(pad, parent.width() - self.width() - pad)
            y = max(pad, parent.height() - self.height() - pad)
            self.move(x, y)

        self.setVisible(True)
        self.raise_()
        QtCore.QTimer.singleShot(max(800, int(msec)), self.hide)




class _CollapsibleSection(QtWidgets.QWidget):
    """Simple collapsible container for grouping optional installs."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, *, start_collapsed: bool = False) -> None:
        super().__init__(parent)

        self._btn = QtWidgets.QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(not start_collapsed)
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(QtCore.Qt.DownArrow if (not start_collapsed) else QtCore.Qt.RightArrow)
        self._btn.clicked.connect(self._on_toggled)

        f = self._btn.font()
        f.setBold(True)
        self._btn.setFont(f)

        self._content = QtWidgets.QWidget()
        self._content_lay = QtWidgets.QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(12, 6, 0, 0)
        self._content_lay.setSpacing(10)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._btn)
        outer.addWidget(self._content)

        self._content.setVisible(not start_collapsed)

        # Subtle divider + padding
        self.setObjectName("CollapsibleSection")
        self.setStyleSheet("""
        QWidget#CollapsibleSection {
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 10px;
        }
        QToolButton {
            border: none;
            padding: 6px 6px;
        }
        QToolButton:hover {
            background: rgba(255,255,255,0.06);
            border-radius: 8px;
        }
        """)

    def layout_content(self) -> QtWidgets.QVBoxLayout:
        return self._content_lay

    def _on_toggled(self) -> None:
        expanded = self._btn.isChecked()
        self._btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._content.setVisible(expanded)



class OptionalInstallsDialog(QtWidgets.QDialog):
    """
    Standalone dialog that runs selected optional installs sequentially and shows live logs.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, root_dir: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optional installs")
        self.setMinimumSize(820, 560)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        self.root_dir: Path = (root_dir or _root_from_this_file()).resolve()
        self._shared_cleanup_notes: List[str] = _cleanup_shared_model_files(self.root_dir)
        self.installs: List[OptionalInstall] = _default_installs()

        self._process: Optional[QtCore.QProcess] = None
        self._queue: List[OptionalInstall] = []
        self._running: Optional[OptionalInstall] = None

        # Log handling
        self._log_fp = None  # type: ignore
        self._log_file_path: Optional[Path] = None

        # Auto-continue / pause handling
        self._pause_key_sent: bool = False

        # Process log buffering. Some installers (especially conda) can emit huge
        # bursts/progress updates while creating an environment. Appending every
        # line directly to QPlainTextEdit can starve the Qt event loop and make
        # the Optional Installs window look frozen until conda is done. Buffer
        # process output and flush it in small UI batches instead.
        self._proc_log_buffer: List[str] = []
        self._proc_log_timer = QtCore.QTimer(self)
        self._proc_log_timer.setSingleShot(True)
        self._proc_log_timer.setInterval(80)
        self._proc_log_timer.timeout.connect(self._flush_proc_log_buffer)


        # Header
        title = QtWidgets.QLabel("Optional installs")
        tf = title.font()
        tf.setPointSize(tf.pointSize() + 6)
        tf.setBold(True)
        title.setFont(tf)

        subtitle = QtWidgets.QLabel(
            "Select extra installs for FrameVision. When you press Start, installs run one-by-one and the log appears below."
        )
        subtitle.setWordWrap(True)

        # Header window controls (minimize / maximize / close)
        header_row = QtWidgets.QWidget()
        header_lay = QtWidgets.QHBoxLayout(header_row)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(8)

        header_lay.addWidget(title)
        header_lay.addStretch(1)

        def _mk_hdr_btn(std_icon: QtWidgets.QStyle.StandardPixmap, tooltip: str) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton()
            btn.setFlat(True)
            btn.setToolTip(tooltip)
            btn.setIcon(self.style().standardIcon(std_icon))
            btn.setFixedSize(34, 28)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            return btn

        self._btn_min = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarMinButton, "Minimize")
        self._btn_max = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarMaxButton, "Maximize / Restore")
        self._btn_x = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarCloseButton, "Close")

        self._btn_min.clicked.connect(self.showMinimized)
        self._btn_max.clicked.connect(lambda: self.showNormal() if self.isMaximized() else self.showMaximized())
        self._btn_x.clicked.connect(self.close)

        header_lay.addWidget(self._btn_min, 0, QtCore.Qt.AlignTop)
        header_lay.addWidget(self._btn_max, 0, QtCore.Qt.AlignTop)
        header_lay.addWidget(self._btn_x, 0, QtCore.Qt.AlignTop)

        # Light hover styling; close button gets a stronger hover
        header_row.setStyleSheet("""
            QPushButton { border: none; background: transparent; border-radius: 6px; }
            QPushButton:hover { background: rgba(255,255,255,0.08); }
            QPushButton:pressed { background: rgba(255,255,255,0.14); }
        """)
        self._btn_x.setStyleSheet("""
            QPushButton { border: none; background: transparent; border-radius: 6px; }
            QPushButton:hover { background: rgba(255, 60, 60, 0.55); }
            QPushButton:pressed { background: rgba(255, 60, 60, 0.75); }
        """)

        
        # Options list (grouped)
        self.rows: List[_OptionRow] = []
        opts_box = QtWidgets.QWidget()
        opts_lay = QtWidgets.QVBoxLayout(opts_box)
        opts_lay.setContentsMargins(0, 0, 0, 0)
        opts_lay.setSpacing(10)

        row_by_key: Dict[str, _OptionRow] = {}

        def _mk_section_label(txt: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(txt)
            f = lbl.font()
            f.setBold(True)
            f.setPointSize(f.pointSize() + 1)
            lbl.setFont(f)
            lbl.setStyleSheet("padding: 6px 2px;")
            return lbl

        def _add_opt(opt: OptionalInstall, parent_lay: QtWidgets.QVBoxLayout) -> None:
            # Indent model download options consistently.
            is_zimage_extra = opt.key.startswith("zimage_") and opt.key != "zimage"
            is_qwen2512_extra = opt.key.startswith("qwen2512_") and opt.key != "qwen2512"
            is_qwen2511_extra = opt.key.startswith("qwen2511_")
            is_qwen_extra = is_qwen2512_extra or is_qwen2511_extra
            is_qwen3tts_extra = opt.key.startswith("qwen3tts_") and opt.key != "qwen3tts"
            is_seedvr2_extra = opt.key.startswith("seedvr2_gguf_")
            is_hunyuan15_extra = opt.key.startswith("hunyuan15_") and opt.key != "hunyuan15"
            is_hidream_extra = opt.key.startswith("hidream_edit_")
            is_ideogram4_extra = opt.key.startswith("ideogram4_") and opt.key != "ideogram4_runtime"

            row = _OptionRow(
                opt,
                indent=(26 if (is_zimage_extra or is_qwen_extra or is_qwen3tts_extra or is_seedvr2_extra or is_hunyuan15_extra or is_hidream_extra or is_ideogram4_extra) else 0),
            )

            if is_zimage_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_zimage_model_toggled(k, checked))
            if is_qwen2512_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_qwen2512_model_toggled(k, checked))
            if is_qwen2511_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_qwen2511_model_toggled(k, checked))
            if is_qwen3tts_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_qwen3tts_model_toggled(k, checked))
            if opt.key.startswith("firered_"):
                row.toggled.connect(lambda checked, k=opt.key: self._on_firered_model_toggled(k, checked))
            if opt.key.startswith("ideogram4_"):
                row.toggled.connect(lambda checked, k=opt.key: self._on_ideogram4_model_toggled(k, checked))
            if opt.key.startswith("hidream_edit_"):
                row.toggled.connect(lambda checked, k=opt.key: self._on_hidream_edit_model_toggled(k, checked))

            if is_seedvr2_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_seedvr2_model_toggled(k, checked))
            if is_hunyuan15_extra:
                row.toggled.connect(lambda checked, k=opt.key: self._on_hunyuan15_extra_toggled(k, checked))

            self.rows.append(row)
            row_by_key[opt.key] = row
            parent_lay.addWidget(row)

        def _add_group_label(txt: str, parent_lay: QtWidgets.QVBoxLayout) -> None:
            lbl = QtWidgets.QLabel(txt)
            f = lbl.font()
            f.setBold(True)
            lbl.setFont(f)
            lbl.setStyleSheet("color: rgba(255,255,255,0.88); padding: 4px 2px;")
            parent_lay.addWidget(lbl)

        # Build a quick lookup for installs by key
        by_key = {i.key: i for i in self.installs}

        # ---- Video models
        opts_lay.addWidget(_mk_section_label("Video models"))

        for k in ("wan22_turbo", "ltx23", "ltx23_fp8", "hunyuan15", "bernini_r_1p3b", "hiar"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, opts_lay)

        # ---- Image models
        opts_lay.addWidget(_mk_section_label("Image models"))

        # Lens Turbo U4 group (collapsible)
        lens_sec = _CollapsibleSection("Lens Turbo U4 Text to Image", start_collapsed=True)
        lens_lay = lens_sec.layout_content()

        _add_group_label("Environment and optional model cache", lens_lay)
        for k in ("lens_turbo_u4", "lens_turbo_u4_model_cache"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, lens_lay)

        opts_lay.addWidget(lens_sec)

        # SPARK.Chroma group (collapsible)
        chroma_sec = _CollapsibleSection("SPARK.Chroma Text to Image", start_collapsed=True)
        chroma_lay = chroma_sec.layout_content()

        _add_group_label("Environment and model", chroma_lay)
        for k in ("chroma",):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, chroma_lay)

        opts_lay.addWidget(chroma_sec)

        # HiDream group (collapsible)
        hidream_sec = _CollapsibleSection("HiDream", start_collapsed=True)
        hidream_lay = hidream_sec.layout_content()

        _add_group_label("BF16 models (choose one)", hidream_lay)
        for k in ("hidream_edit_base", "hidream_edit_dev", "hidream_edit_dev_2604_bf16", "hidream_edit_both"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, hidream_lay)

        _add_group_label("FP8 models (choose one)", hidream_lay)
        for k in ("hidream_edit_base_fp8", "hidream_edit_dev_fp8", "hidream_edit_both_fp8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, hidream_lay)

        _add_group_label("Everything", hidream_lay)
        for k in ("hidream_edit_all",):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, hidream_lay)

        opts_lay.addWidget(hidream_sec)

        # Qwen group (collapsible)
        qwen_sec = _CollapsibleSection("Qwen models", start_collapsed=True)
        qwen_lay = qwen_sec.layout_content()

        # Qwen Edit 2511
        _add_group_label("Qwen Edit 2511", qwen_lay)
        for k in ("qwen2511_q3", "qwen2511_q4km", "qwen2511_q5km", "qwen2511_q6k", "qwen2511_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, qwen_lay)

        # Qwen Image 2512
        _add_group_label("Qwen Image 2512", qwen_lay)
        for k in ("qwen2512_q2", "qwen2512_q3", "qwen2512_q4", "qwen2512_q5", "qwen2512_q6", "qwen2512_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, qwen_lay)

        opts_lay.addWidget(qwen_sec)

        # Ideogram 4 GGUF group (collapsible)
        ideogram4_sec = _CollapsibleSection("Ideogram 4 GGUF", start_collapsed=True)
        ideogram4_lay = ideogram4_sec.layout_content()

        _add_group_label("Runtime / shared files", ideogram4_lay)
        for k in ("ideogram4_runtime",):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, ideogram4_lay)

        _add_group_label("Conditional model (choose one)", ideogram4_lay)
        for k in ("ideogram4_cond_q5", "ideogram4_cond_q6", "ideogram4_cond_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, ideogram4_lay)

        _add_group_label("Unconditional model (choose one)", ideogram4_lay)
        for k in ("ideogram4_uncond_q2", "ideogram4_uncond_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, ideogram4_lay)

        opts_lay.addWidget(ideogram4_sec)

        # FireRed group (collapsible)
        firered_sec = _CollapsibleSection("FireRed 1.1 edit 20B GGUF", start_collapsed=True)
        firered_lay = firered_sec.layout_content()

        _add_group_label("Main model (choose one)", firered_lay)
        for k in (
            "firered_q3km",
            "firered_q3ks",
            "firered_q4_0",
            "firered_q4_1",
            "firered_q4ks",
            "firered_q4km",
            "firered_q5_0",
            "firered_q5_1",
            "firered_q5ks",
            "firered_q5km",
            "firered_q6k",
            "firered_q8",
        ):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, firered_lay)

        opts_lay.addWidget(firered_sec)

        # Z-Image group (collapsible)
        zimg_sec = _CollapsibleSection("Z-Image models", start_collapsed=True)
        zimg_lay = zimg_sec.layout_content()

        _add_group_label("Z-Image Turbo (full 16-bit)", zimg_lay)
        for k in ("zimage", "zimage_fp16"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, zimg_lay)

        _add_group_label("Z-Image Turbo GGUF (low VRAM)", zimg_lay)
        for k in ("zimage_gguf4", "zimage_gguf5", "zimage_gguf6", "zimage_gguf8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, zimg_lay)

        opts_lay.addWidget(zimg_sec)

        # Flux Klein edit 4B group (collapsible)
        klein_sec = _CollapsibleSection("Flux Klein edit 4B", start_collapsed=True)
        klein_lay = klein_sec.layout_content()

        _add_group_label("UNet GGUF (choose one)", klein_lay)
        for k in ("klein4b_unet_q2k", "klein4b_unet_q3kl", "klein4b_unet_q4km", "klein4b_unet_q5km", "klein4b_unet_q6k", "klein4b_unet_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, klein_lay)

        _add_group_label("Text encoder (choose one)", klein_lay)
        for k in ("klein4b_te_comfy", "klein4b_te_qwen_gguf_q4km", "klein4b_te_qwen_gguf_q5km", "klein4b_te_qwen_gguf_q8", "klein4b_te_cordux"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, klein_lay)

        opts_lay.addWidget(klein_sec)


        # Flux Klein edit 9B group (collapsible)
        klein9_sec = _CollapsibleSection("Flux Klein edit 9B", start_collapsed=True)
        klein9_lay = klein9_sec.layout_content()

        _add_group_label("UNet GGUF (choose one)", klein9_lay)
        for k in ("klein9b_unet_q2k", "klein9b_unet_q3km", "klein9b_unet_q4km", "klein9b_unet_q5km", "klein9b_unet_q6k", "klein9b_unet_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, klein9_lay)

        _add_group_label("Text encoder (choose one)", klein9_lay)
        for k in ("klein9b_te_q2k", "klein9b_te_q4km", "klein9b_te_q5km", "klein9b_te_q6k", "klein9b_te_q8"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, klein9_lay)

        opts_lay.addWidget(klein9_sec)



        # Other image-related installs
        # Keep SeedVR2 close to GFPGAN (requested: "bottom" grouping).
        for k in (
            "bgrem_inpaint",
            "sdxl_inpaint_env",
            "sdxljugg",
            "gfpgan",
            "seedvr2_env",
            "seedvr2_gguf_q3",
            "seedvr2_gguf_q4",
            "seedvr2_gguf_q5",
            "seedvr2_gguf_q6",
            "seedvr2_gguf_q8",
        ):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, opts_lay)

        # ---- Audio models
        opts_lay.addWidget(_mk_section_label("Audio models"))

        for k in ("qwen3tts", "qwen3tts_models", "qwen3tts_flashattn", "ace15"):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, opts_lay)

        # ---- Utilities
        opts_lay.addWidget(_mk_section_label("Utilities"))

        for k in ("whisper",):
            opt = by_key.get(k)
            if opt:
                _add_opt(opt, opts_lay)

        self._row_by_key = row_by_key

        opts_lay.addStretch(1)

        scroll = QtWidgets.QScrollArea()

        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(opts_box)

        # Log + progress
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Install log will appear here...")
        if self._shared_cleanup_notes:
            self.log.appendPlainText(f"[cleanup] moved {len(self._shared_cleanup_notes)} shared model file(s) to models/shared")

        # Log tools (below progress bar)
        self.auto_continue_chk = QtWidgets.QCheckBox('Auto-continue when installer says "Press any key to continue"')
        self.auto_continue_chk.setChecked(True)
        self.auto_continue_chk.setToolTip(
            "Some installers call PAUSE or ask you to press a key/button, which can hang in this embedded log.\n"
            "When enabled, FrameVision will try to auto-send Enter to continue."
        )

        self.save_log_chk = QtWidgets.QCheckBox("Save log to file")
        self.save_log_chk.setChecked(False)
        self.save_log_chk.setToolTip("Write the full optional-installs log to a file in <root>/logs/.")

        self.open_logs_btn = QtWidgets.QPushButton("Open logs folder")

        self.licenses_btn = QtWidgets.QPushButton("Third-party licenses")
        self.licenses_btn.setToolTip("Open the third-party licenses list (presets/info/3rd_party_licenses.json)")
        self.licenses_btn.clicked.connect(self._open_licenses_viewer)

        self.open_logs_btn.setToolTip("Open the FrameVision logs folder.")
        self.open_logs_btn.clicked.connect(self._open_logs_folder)

        log_tools = QtWidgets.QHBoxLayout()
        log_tools.setContentsMargins(0, 0, 0, 0)
        log_tools.setSpacing(10)
        log_tools.addWidget(self.auto_continue_chk)
        log_tools.addWidget(self.save_log_chk)
        log_tools.addStretch(1)
        log_tools.addWidget(self.open_logs_btn)
        log_tools.addWidget(self.licenses_btn)

        # Buttons
        self.start_btn = QtWidgets.QPushButton("Start optional installs")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.continue_btn = QtWidgets.QPushButton("Send Enter")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setEnabled(False)

        self.continue_btn.setEnabled(False)
        self.continue_btn.setToolTip("If an installer shows 'Press any key to continue', click this to continue.")
        self.continue_btn.clicked.connect(self._send_enter_to_process)

        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.close_btn.clicked.connect(self.accept)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.continue_btn)
        btns.addWidget(self.close_btn)

        # Layout
        top = QtWidgets.QVBoxLayout(self)
        top.setContentsMargins(14, 14, 14, 14)
        top.setSpacing(10)
        top.addWidget(header_row)
        top.addWidget(subtitle)
        top.addWidget(scroll, 2)
        top.addWidget(self.status_lbl)
        top.addWidget(self.progress)
        top.addLayout(log_tools)
        top.addWidget(self.log, 1)
        top.addLayout(btns)

        # Nice default font size for log
        lf = self.log.font()
        lf.setPointSize(max(9, lf.pointSize()))
        self.log.setFont(lf)
        try:
            # Keep long conda/pip logs from making the UI sluggish over time.
            self.log.document().setMaximumBlockCount(12000)
        except Exception:
            pass

        # Toast popup (non-modal)
        self._toast_widget = _ToastWidget(self)

        self._append_line(f"Root: {self.root_dir}")

    # ---- logging helpers

    def _append_line(self, s: str) -> None:
        self._append_text_block((s or "") + "\n", mirror_to_file=True)

    def _append_text_block(self, text: str, *, mirror_to_file: bool = True) -> None:
        """Append a block of text to the log with one UI update.

        This is intentionally separate from _append_line so process output can be
        flushed in batches. Conda environment creation can print a lot of status
        text quickly; batching prevents the optional-installs dialog from freezing
        just because the log widget is being updated too often.
        """
        if not text:
            return
        try:
            cursor = self.log.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(text)
            self.log.setTextCursor(cursor)
            sb = self.log.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            try:
                self.log.appendPlainText(text.rstrip("\n"))
            except Exception:
                pass

        if not mirror_to_file:
            return
        # Optional: mirror log output to a file.
        try:
            fp = getattr(self, "_log_fp", None)
            if fp is not None:
                fp.write(text)
                fp.flush()
        except Exception:
            pass

    def _queue_proc_log_line(self, s: str) -> None:
        """Queue one process-output line for throttled UI logging."""
        try:
            self._proc_log_buffer.append(s)
            if not self._proc_log_timer.isActive():
                self._proc_log_timer.start()
        except Exception:
            self._append_line(s)

    def _flush_proc_log_buffer(self) -> None:
        """Flush queued process log lines without monopolising the UI thread."""
        try:
            if not self._proc_log_buffer:
                return
            # Limit each paint/update batch. If conda dumps thousands of lines,
            # continue flushing on the next event-loop tick instead of blocking.
            batch = self._proc_log_buffer[:250]
            del self._proc_log_buffer[:250]
            self._append_text_block("\n".join(batch) + "\n", mirror_to_file=True)
            if self._proc_log_buffer:
                self._proc_log_timer.start(0)
        except Exception:
            pass


    def _toast(self, msg: str, msec: int = 2800) -> None:
        """Show a small, non-blocking toast message."""
        try:
            tw = getattr(self, "_toast_widget", None)
            if tw is not None:
                tw.show_message(msg, msec=msec)
        except Exception:
            # Toasts should never crash the installer UI.
            pass


    def _logs_dir(self) -> Path:
        return (self.root_dir / "logs").resolve()

    def _open_logs_folder(self) -> None:
        try:
            p = self._logs_dir()
            p.mkdir(parents=True, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
        except Exception:
            # Best-effort only.
            pass

    def _start_log_file_if_enabled(self) -> None:
        # Called at the start of a run.
        self._close_log_file()
        try:
            if not getattr(self, "save_log_chk", None) or (not self.save_log_chk.isChecked()):
                return
            logs_dir = self._logs_dir()
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file_path = logs_dir / f"optional_installs_{ts}.log"
            self._log_fp = open(self._log_file_path, "w", encoding="utf-8", errors="replace")
            self._append_line(f"[INFO] Saving log to: {self._log_file_path}")
        except Exception:
            # If log file fails, continue without it.
            self._log_fp = None
            self._log_file_path = None

    def _open_licenses_viewer(self) -> None:
        """Open the third-party licenses viewer (JSON-backed)."""
        try:
            from helpers.licenses_viewer import LicensesViewerDialog
            dlg = LicensesViewerDialog(parent=self, root_dir=self.root_dir)
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Third-party licenses", f"Failed to open licenses viewer:\n{e}")

    def _close_log_file(self) -> None:
        try:
            self._flush_proc_log_buffer()
        except Exception:
            pass
        try:
            fp = getattr(self, "_log_fp", None)
            if fp is not None:
                fp.flush()
                fp.close()
        except Exception:
            pass
        self._log_fp = None
        self._log_file_path = None

    def _send_enter_to_process(self) -> None:
        """Manually send Enter to the running process (useful for PAUSE)."""
        if self._process is None:
            return
        try:
            self._process.write(b"\r\n")
            self._process.waitForBytesWritten(250)
            self._append_line("[INFO] Sent Enter to installer input.")
            self._toast("Sent Enter to continue…")
        except Exception:
            pass

    def _maybe_auto_continue(self, text_line: str) -> None:
        """Detect common PAUSE prompts and auto-continue if enabled."""
        try:
            if self._pause_key_sent:
                return
            if not self.auto_continue_chk.isChecked():
                return
            s = (text_line or "").lower()
            pause_markers = (
                "press any key to continue",
                "press a button to continue",
                "press enter to continue",
                "press return to continue",
                "hit any key to continue",
            )
            if any(marker in s for marker in pause_markers):
                self._pause_key_sent = True
                self._append_line("[AUTO] Detected continue prompt — sending Enter…")
                try:
                    if self._process is not None:
                        self._process.write(b"\r\n")
                        self._process.waitForBytesWritten(250)
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """If an installer is running, confirm before closing so we don't leave background processes."""
        try:
            running = (self._process is not None and self._process.state() != QtCore.QProcess.NotRunning) or bool(self._queue)
        except Exception:
            running = False

        if running:
            mb = QtWidgets.QMessageBox(self)
            mb.setIcon(QtWidgets.QMessageBox.Warning)
            mb.setWindowTitle("Installer running")
            mb.setText("An optional install is still running.")
            mb.setInformativeText("Do you want to cancel the install and close this window?")
            btn_keep = mb.addButton("Keep running", QtWidgets.QMessageBox.RejectRole)
            btn_close = mb.addButton("Cancel & close", QtWidgets.QMessageBox.AcceptRole)
            mb.setDefaultButton(btn_keep)
            mb.exec()

            if mb.clickedButton() == btn_keep:
                event.ignore()
                return

            # Cancel the run, then close.
            self._on_cancel()

        super().closeEvent(event)


    # ---- env folder safety (optional installs)

    def _env_dir_for(self, opt: OptionalInstall) -> Optional[Path]:
        """Return the environment folder for an optional install.

        Some optional installs share environments (e.g. Qwen2511 re-uses Qwen2512),
        and some UI keys represent model downloads (qwen2512_q4, seedvr2_gguf_q4, ...)
        that still belong to a single environment.

        This function centralizes that mapping so the "keep / delete env" prompt always
        points at the correct folder.
        """

        k = (getattr(opt, "key", "") or "").strip()

        # Qwen2511/Qwen2512 GGUF installs download model files and use the Qwen sd-cli bin.
        # They do not own a Python environment, so do not show an env delete/reinstall prompt.
        if k == "qwen2512" or k.startswith("qwen2512_") or k.startswith("qwen2511_"):
            return None

        # SeedVR2: GGUF downloads use the SeedVR2 env.
        if k.startswith("seedvr2_gguf_"):
            rel_seed = _ENV_DIR_BY_KEY.get("seedvr2_env", Path("environments") / ".seedvr2")
            return (self.root_dir / rel_seed).resolve()

        # Default: direct mapping.
        rel = _ENV_DIR_BY_KEY.get(k)
        if rel is None:
            return None
        return (self.root_dir / rel).resolve()

    def _rmtree_safe(self, p: Path) -> Tuple[bool, str]:
        """
        Robust recursive delete for Windows (handles read-only files).
        """
        try:
            if not p.exists():
                return True, ""
            if p.is_file():
                p.unlink()
                return True, ""
            def _onerror(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    raise
            shutil.rmtree(p, onerror=_onerror)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _maybe_reset_env(self, opt: OptionalInstall) -> str:
        """Handle existing optional-install environments.

        Returns:
            "proceed"   -> continue and run the installer as normal
            "keep"      -> user chose to keep the existing env folder
            "reinstall" -> env folder was deleted; proceed to reinstall
            "cancel"    -> user cancelled (queue cleared / UI updated)
        """
        env_dir = self._env_dir_for(opt)
        if env_dir is None:
            return "proceed"
        if not env_dir.exists():
            return "proceed"

        # Pause and warn user.
        mb = QtWidgets.QMessageBox(self)
        mb.setIcon(QtWidgets.QMessageBox.Warning)
        mb.setWindowTitle("Existing environment found")
        mb.setText(f"An existing environment folder was found for:\n\n{opt.title}")
        mb.setInformativeText(
            "FrameVision found an existing environment folder for this component.\n\n"
            "Choose what to do:\n"
            "• Continue without reinstalling environment: keep the env and proceed (fastest).\n"
            "• Delete and reinstall environment: deletes ONLY the env folder below, then reinstalls it.\n"
            "Downloaded models/weights will not be removed.\n\n"
            f"Environment folder:\n{env_dir}"
        )
        btn_cancel = mb.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        btn_keep = mb.addButton("Continue without reinstalling environment", QtWidgets.QMessageBox.AcceptRole)
        btn_reinstall = mb.addButton("Delete and reinstall environment", QtWidgets.QMessageBox.DestructiveRole)
        mb.setDefaultButton(btn_keep)
        mb.exec()

        if mb.clickedButton() == btn_cancel:
            self._append_line("")
            self._append_line("[CANCEL] User cancelled (existing env found).")
            self._queue.clear()
            self._running = None
            self.status_lbl.setText("Cancelled.")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return "cancel"

        if mb.clickedButton() == btn_keep:
            self._append_line("")
            self._append_line("[INFO] Keeping existing env folder (no reinstall).")
            self._toast(f"Keeping existing environment: {opt.title}")
            return "keep"

        # User chose delete + reinstall: delete environment folder and proceed.
        self._append_line("")
        self._toast(f"Deleting existing environment to reinstall: {opt.title}")

        self._append_line(f"[INFO] Deleting existing env folder: {env_dir}")
        ok, err = self._rmtree_safe(env_dir)
        if not ok:
            self._append_line(f"[ERROR] Failed to delete env folder: {err}")
            QtWidgets.QMessageBox.critical(
                self,
                "Could not delete environment folder",
                f"Failed to delete:\n{env_dir}\n\nError:\n{err}",
            )
            # Safer to cancel the whole run than continue in a half-broken state.
            self._queue.clear()
            self._running = None
            self.status_lbl.setText("Cancelled (delete failed).")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return "cancel"

        self._append_line("[OK] Existing env folder deleted.")
        self._toast(f"Environment deleted. Reinstalling: {opt.title}")
        return "reinstall"

    # ---- selection helpers



    def _on_zimage_model_toggled(self, key: str, checked: bool) -> None:
        """Keep Z-image model selections sane and warn when prerequisites are missing."""
        if not checked:
            return


        # Make Turbo GGUF quants mutually exclusive to avoid accidentally downloading 2–4 large models.
        if key.startswith("zimage_gguf"):
            try:
                for k, row in getattr(self, "_row_by_key", {}).items():
                    if k.startswith("zimage_gguf") and k != key and row.is_checked():
                        row.set_checked(False)
            except Exception:
                pass

        # Prereqs for the Turbo model downloader.
        script = self.root_dir / "presets" / "extra_env" / "zimage_install.py"
        if not script.exists():
            self._toast(
                "Z-Image installer missing: presets/extra_env/zimage_install.py. "
                "Apply the Z-Image unified installer patch first."
            )
            return


    def _on_firered_model_toggled(self, key: str, checked: bool) -> None:
        """Keep FireRed GGUF selections sane and show a short info toast."""
        if not checked:
            return

        try:
            for k, row in getattr(self, "_row_by_key", {}).items():
                if k.startswith("firered_") and k != key and row.is_checked():
                    row.set_checked(False)
        except Exception:
            pass

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        self._toast("FireRed 1.1 files will be downloaded flat into models/FireRed-Image-Edit-1.1 when you press Start.")


    def _on_seedvr2_model_toggled(self, key: str, checked: bool) -> None:
        """Keep SeedVR2 GGUF selections sane and warn when prerequisites are missing."""
        if not checked:
            return

        # Make GGUF quants mutually exclusive (big downloads).
        if key.startswith("seedvr2_gguf_"):
            try:
                for k, row in getattr(self, "_row_by_key", {}).items():
                    if k.startswith("seedvr2_gguf_") and k != key and row.is_checked():
                        row.set_checked(False)
            except Exception:
                pass

        # Prereqs for the installer.
        script = self.root_dir / "presets" / "extra_env" / "seedvr2_gguf_installer.py"
        if not script.exists():
            self._toast(
                "SeedVR2 installer missing: presets/extra_env/seedvr2_gguf_installer.py. "
                "Update your FrameVision install or reinstall optional installs scripts."
            )
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        env_dir = (self.root_dir / "environments" / ".seedvr2").resolve()
        if not env_dir.exists():
            self._toast("SeedVR2 environment not found — it will be installed first.")
        else:
            self._toast("SeedVR2 GGUF model will download when you press Start.")

    def _on_qwen2512_model_toggled(self, key: str, checked: bool) -> None:
        """Warn when Qwen2512 model prerequisites are missing."""
        if not checked:
            return

        # Prereqs for the downloader.
        script = _find_qwen2512_downloader_script(self.root_dir)
        if script is None:
            self._toast("Missing downloader: qwen2512_download.py")
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        # Info toast
        self._toast("Qwen2512 GGUF model will download when you press Start.")





    def _on_qwen2511_model_toggled(self, key: str, checked: bool) -> None:
        """Warn when Qwen2511 model prerequisites are missing."""
        if not checked:
            return

        script = _find_qwen2511_downloader_script(self.root_dir)
        if script is None:
            self._toast("Missing downloader: qwen2511_download.py")
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        # Qwen2511 re-uses the Qwen2512 stable-diffusion.cpp bin folder when present,
        # but it does not need a private Python environment.
        self._toast("Qwen2511 GGUF model will download when you press Start.")


    def _on_qwen3tts_model_toggled(self, key: str, checked: bool) -> None:
        """Warn when Qwen3-TTS model downloader prerequisites are missing."""
        if not checked:
            return

        # FlashAttention add-on (optional)
        if key == "qwen3tts_flashattn":
            bat = self.root_dir / "presets" / "extra_env" / "install_flashattn_optional.bat"
            if not bat.exists():
                self._toast("Missing installer: install_flashattn_optional.bat")
                return

            env_dir = (self.root_dir / "environments" / ".qwen3tts").resolve()
            if not env_dir.exists():
                self._toast("Qwen 3 TTS environment not found — it will be installed first.")

            self._toast("FlashAttention will install when you press Start (env install only needed once).")
            return


        bat = self.root_dir / "presets" / "extra_env" / "download_qwentts_models.bat"
        if not bat.exists():
            self._toast("Missing downloader: download_qwentts_models.bat")
            return

        # Qwen3-TTS uses its own environment folder.
        env_dir = (self.root_dir / "environments" / ".qwen3tts").resolve()
        if not env_dir.exists():
            self._toast("Qwen 3 TTS environment not found — it will be installed first.")

        self._toast("Qwen 3 TTS models will download when you press Start (env install only needed once).")


    def _on_ideogram4_model_toggled(self, key: str, checked: bool) -> None:
        """Keep Ideogram 4 GGUF selections sane and warn when prerequisites are missing."""
        if not checked:
            return

        try:
            for k, row in getattr(self, "_row_by_key", {}).items():
                if key.startswith("ideogram4_cond_") and k.startswith("ideogram4_cond_") and k != key and row.is_checked():
                    row.set_checked(False)
                if key.startswith("ideogram4_uncond_") and k.startswith("ideogram4_uncond_") and k != key and row.is_checked():
                    row.set_checked(False)
        except Exception:
            pass

        script = _find_ideogram4_gguf_installer(self.root_dir)
        if script is None:
            self._toast("Missing installer: presets/extra_env/ideogram4_gguf_install.py")
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        if key == "ideogram4_runtime":
            self._toast("Ideogram 4 runtime/shared files will install when you press Start.")
        elif key.startswith("ideogram4_cond_"):
            self._toast("Ideogram 4 conditional GGUF will download when you press Start.")
        elif key.startswith("ideogram4_uncond_"):
            self._toast("Ideogram 4 unconditional GGUF will download when you press Start.")


    def _on_hidream_edit_model_toggled(self, key: str, checked: bool) -> None:
        """Keep HiDream model selections sane and warn when the installer is missing."""
        if not checked:
            return

        # All HiDream entries share the same installer/env. Treat the options as
        # mutually exclusive so users do not accidentally queue duplicate runs for
        # the same shared environment/model root.
        try:
            for k, row in getattr(self, "_row_by_key", {}).items():
                if k.startswith("hidream_edit_") and k != key and row.is_checked():
                    row.set_checked(False)
        except Exception:
            pass

        script = self.root_dir / "presets" / "extra_env" / "hidream_install.py"
        if not script.exists():
            self._toast("HiDream installer missing: presets/extra_env/hidream_install.py")
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        env_dir = (self.root_dir / "environments" / ".hidream_dev").resolve()
        if not env_dir.exists():
            self._toast("HiDream environment not found — it will be installed first.")
        else:
            self._toast("HiDream will reuse the existing env/repo and only download missing model files.")


    def _on_hunyuan15_extra_toggled(self, key: str, checked: bool) -> None:
        """Warn when Hunyuan 1.5 optional add-ons are selected without the env."""
        if not checked:
            return

        if key == "hunyuan15_flashattn":
            env_dir = (self.root_dir / "environments" / ".hunyuan15_official").resolve()
            if not env_dir.exists():
                self._toast("Hunyuan 1.5 environment not found — it will be installed first.")
            self._toast("Hunyuan FlashAttention will install when you press Start (env install only needed once).")
            return


    def selected_installs(self) -> List[OptionalInstall]:
        # Preserve install order from self.installs.
        checked_keys = [row.opt.key for row in self.rows if row.is_checked()]
        checked_set = set(checked_keys)

        ordered: List[OptionalInstall] = []
        for opt in self.installs:
            if opt.key in checked_set:
                ordered.append(opt)

        # Qwen2511/Qwen2512 GGUF downloads do not auto-install a private Python env.
        # They only need the model files and the Qwen stable-diffusion.cpp bin folder.
        self._auto_added_qwen2512_env = False
        self._auto_added_qwen2511_env = False

        # Qwen3-TTS:
        # If user selects the models-only downloader, ensure the Qwen3-TTS environment exists first.
        self._auto_added_qwen3tts_env = False
        try:
            wants_qwen3tts_extras = ("qwen3tts_models" in checked_set) or ("qwen3tts_flashattn" in checked_set)
            if wants_qwen3tts_extras and ("qwen3tts" not in checked_set):
                env_dir = (self.root_dir / "environments" / ".qwen3tts").resolve()
                if not env_dir.exists():
                    first_idx = None
                    for i, opt in enumerate(ordered):
                        if opt.key in ("qwen3tts_models", "qwen3tts_flashattn"):
                            first_idx = i
                            break
                    for opt in self.installs:
                        if opt.key == "qwen3tts":
                            if first_idx is None:
                                ordered.insert(0, opt)
                            else:
                                ordered.insert(first_idx, opt)
                            self._auto_added_qwen3tts_env = True
                            break
        except Exception:
            pass


        # Hunyuan 1.5 optional extras:
        # If user selects Hunyuan extras (e.g. FlashAttention), ensure the Hunyuan env exists first.
        self._auto_added_hunyuan15_env = False
        try:
            wants_hunyuan15_extras = False  # Hunyuan FlashAttention optional install is hidden/disabled
            if wants_hunyuan15_extras and ("hunyuan15" not in checked_set):
                env_dir = (self.root_dir / "environments" / ".hunyuan15_official").resolve()
                if not env_dir.exists():
                    first_idx = None
                    for i, opt in enumerate(ordered):
                        if opt.key == "hunyuan15_flashattn":
                            first_idx = i
                            break
                    for opt in self.installs:
                        if opt.key == "hunyuan15":
                            if first_idx is None:
                                ordered.insert(0, opt)
                            else:
                                ordered.insert(first_idx, opt)
                            self._auto_added_hunyuan15_env = True
                            break
        except Exception:
            pass


        #  install the env first.
        # This is silent: we only auto-add the env step when it is missing, so no "existing env"
        # prompt will appear.
        self._auto_added_zimage_env = False
        try:
            wants_zimage_models = any(k.startswith("zimage_") and k != "zimage" for k in checked_set)
            if wants_zimage_models and ("zimage" not in checked_set):
                env_rel = _ENV_DIR_BY_KEY.get("zimage")
                env_dir = (self.root_dir / env_rel).resolve() if env_rel is not None else None
                if env_dir is not None and (not env_dir.exists()):
                    for opt in self.installs:
                        if opt.key == "zimage":
                            ordered.insert(0, opt)
                            self._auto_added_zimage_env = True
                            break
        except Exception:
            pass


        # SeedVR2 GGUF:
        # If user selects a SeedVR2 GGUF quant without selecting the env toggle,
        # silently add the env install when missing.
        # (Requested behavior: if env isn't there, always install it even when only GGUF is selected.)
        self._auto_added_seedvr2_env = False
        try:
            wants_seedvr2_models = any(k.startswith("seedvr2_gguf_") for k in checked_set)
            if wants_seedvr2_models and ("seedvr2_env" not in checked_set):
                env_dir = (self.root_dir / "environments" / ".seedvr2").resolve()
                if not env_dir.exists():
                    # Insert env install right before the first selected SeedVR2 model.
                    first_idx = None
                    for i, opt in enumerate(ordered):
                        if opt.key.startswith("seedvr2_gguf_"):
                            first_idx = i
                            break
                    for opt in self.installs:
                        if opt.key == "seedvr2_env":
                            if first_idx is None:
                                ordered.insert(0, opt)
                            else:
                                ordered.insert(first_idx, opt)
                            self._auto_added_seedvr2_env = True
                            break
        except Exception:
            pass

        return ordered

    # ---- run control

    def _set_running_ui(self, running: bool) -> None:
        for r in self.rows:
            r.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(True)
        # Enable manual "Send Enter" only while a process is expected to be running.
        try:
            self.continue_btn.setEnabled(running)
        except Exception:
            pass
        self.close_btn.setEnabled(not running and (self._running is None) and (not self._queue))

    def _on_start(self) -> None:
        selected = self.selected_installs()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Optional installs", "No optional installs selected.")
            return

        self._queue = list(selected)
        self._running = None
        self._pause_key_sent = False
        self._start_log_file_if_enabled()
        self._append_line("")
        self._append_line("=== Starting optional installs ===")
        if getattr(self, "_auto_added_zimage_env", False):
            self._append_line("[INFO] Shared image-model env not found — installing it first so Z-image can run after model download.")

        if getattr(self, "_auto_added_qwen3tts_env", False):
            self._append_line("[INFO] Qwen3-TTS env not found — installing it first because you selected extra model downloads.")

        if getattr(self, "_auto_added_hunyuan15_env", False):
            self._append_line("[INFO] Hunyuan 1.5 env not found — installing it first because you selected FlashAttention.")

        if getattr(self, "_auto_added_seedvr2_env", False):
            self._append_line("[INFO] SeedVR2 env not found — installing it first because you selected a GGUF quant.")

        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)

        self._set_running_ui(True)
        self._run_next()

    def _on_cancel(self) -> None:
        # If currently running, attempt to terminate.
        if self._process is not None and self._process.state() != QtCore.QProcess.NotRunning:
            self._append_line("")
            self._append_line("[CANCEL] Terminating current installer...")
            self._process.kill()
            self._process = None

        try:
            self._flush_proc_log_buffer()
            self._proc_log_buffer.clear()
        except Exception:
            pass

        # If we were mid-queue, clear it.
        self._queue.clear()
        self._running = None
        self.status_lbl.setText("Cancelled.")
        self._set_running_ui(False)
        self.close_btn.setEnabled(True)
        self._close_log_file()

    def _run_next(self) -> None:
        if not self._queue:
            try:
                self._flush_proc_log_buffer()
            except Exception:
                pass
            self._append_line("")
            self._append_line("=== Optional installs completed ===")
            self._close_log_file()
            self.status_lbl.setText("Optional installs completed.")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return

        self._running = self._queue.pop(0)
        idx_done = self.progress.value()
        self.status_lbl.setText(f"Running: {self._running.title}")
        self._append_line("")
        self._append_line(f"[RUN] {self._running.title}")

        cmd = self._running.runner(self.root_dir)
        if cmd is None:
            self._append_line(f"[WARN] Missing installer/script for: {self._running.title}")
            self.progress.setValue(idx_done + 1)
            self._run_next()
            return

        decision = self._maybe_reset_env(self._running)
        if decision == "cancel":
            return

        # Special case:
        # If the user chooses to keep an existing Z-Image env, do NOT re-run the env install.
        # Keeping means we should skip straight to the model download step.
        if decision == "keep" and self._running.key == "zimage":
            self._append_line("[INFO] Skipping shared image-model environment install (keeping existing env).")
            self._toast("Z-image env kept. Continuing to downloads…")
            self.progress.setValue(idx_done + 1)
            self._running = None
            self._run_next()
            return

        # Same idea for SeedVR2 env: "keep" means do NOT rerun pip installs.
        if decision == "keep" and self._running.key == "seedvr2_env":
            self._append_line("[INFO] Skipping SeedVR2 environment install (keeping existing env).")
            self._toast("SeedVR2 env kept. Continuing…")
            self.progress.setValue(idx_done + 1)
            self._running = None
            self._run_next()
            return

        # Special case:
        # Whisper installer always recreates the env; if the user chooses to keep an existing env,
        # we should skip running whisper_install.bat to avoid deleting it again.
        if decision == "keep" and self._running.key == "whisper":
            self._append_line("[INFO] Skipping Whisper install (keeping existing env).")
            self._toast("Whisper env kept. Skipping reinstall.")
            self.progress.setValue(idx_done + 1)
            self._running = None
            self._run_next()
            return


        program, args, cwd = cmd

        # QProcess setup
        self._pause_key_sent = False
        proc = QtCore.QProcess(self)
        self._process = proc
        proc.setProgram(program)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(cwd))

        # Set ROOT env var like install_menu.bat does.
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("ROOT", str(self.root_dir) + (os.sep if not str(self.root_dir).endswith(os.sep) else ""))
        proc.setProcessEnvironment(env)

        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(self._on_proc_output)
        proc.readyReadStandardError.connect(self._on_proc_output)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        # Start
        proc.start()

        if not proc.waitForStarted(3000):
            self._append_line(f"[ERROR] Failed to start: {program}")
            self.progress.setValue(idx_done + 1)
            self._run_next()

    def _on_proc_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        if data:
            # Conda often uses carriage-return progress updates. Treat them as
            # line breaks for readable logs, but queue them so the UI stays live.
            data = data.replace("\r", "\n")
            for line in data.splitlines():
                self._queue_proc_log_line(line)
                self._maybe_auto_continue(line)

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError) -> None:
        self._append_line(f"[ERROR] Process error: {err}")

    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        try:
            self._flush_proc_log_buffer()
        except Exception:
            pass
        done_count = self.progress.value() + 1
        self.progress.setValue(done_count)

        if exit_status != QtCore.QProcess.NormalExit:
            self._append_line(f"[ERROR] Installer crashed (exit_code={exit_code}).")
        elif exit_code != 0:
            self._append_line(f"[WARN] Installer finished with exit_code={exit_code}.")
        else:
            self._append_line("[OK] Finished.")

        notes = _cleanup_shared_model_files(self.root_dir)
        if notes:
            self._append_line(f"[cleanup] moved {len(notes)} shared model file(s) to models/shared")

        self._process = None
        self._running = None
        self._run_next()


# -----------------------------
# Public API (import friendly)
# -----------------------------

def show_optional_installs(parent: Optional[QtWidgets.QWidget] = None, root_dir: Optional[str] = None) -> int:
    """
    Import-friendly entry point.

    Returns QDialog.exec() result.
    """
    root_path = Path(root_dir).resolve() if root_dir else None

    # If FrameVision already has a QApplication, don't create another.
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        created_app = True
        app = QtWidgets.QApplication(sys.argv)

    dlg = OptionalInstallsDialog(parent=parent, root_dir=root_path)

    # If we're running inside FrameVision already, show modeless so the main app stays usable.
    if not created_app:
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return 0

    # Standalone testing: run as a modal dialog.
    res = dlg.exec()

    # If we created the app, close it cleanly.
    app.quit()
    return res


def main() -> None:
    show_optional_installs(parent=None, root_dir=None)


if __name__ == "__main__":
    main()
