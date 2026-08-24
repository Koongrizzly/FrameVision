"""Music Clip Creator adapter: LTX 2.5 native FP16/BF16 renderer."""
from __future__ import annotations

try:
    from .ltx25_music_bridge_common import _load_base, export_base_api, install_generation_patch, install_settings_patch, install_status
except Exception:
    from ltx25_music_bridge_common import _load_base, export_base_api, install_generation_patch, install_settings_patch, install_status

_BASE = _load_base("fp16")
install_settings_patch(_BASE, "fp16")
_build_ltx25_args = install_generation_patch(_BASE, "fp16")
export_base_api(globals(), _BASE)


def is_available() -> bool:
    return bool(install_status("fp16").get("ok"))


def ltx25_install_status(root_dir=None):
    return install_status("fp16")
