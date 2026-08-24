"""Music Clip Creator adapter: LTX 2.5 W4A8 ConvRot renderer."""
from __future__ import annotations

try:
    from .ltx25_music_bridge_common import _load_base, export_base_api, install_generation_patch, install_settings_patch, install_status
except Exception:
    from ltx25_music_bridge_common import _load_base, export_base_api, install_generation_patch, install_settings_patch, install_status

_BASE = _load_base("convrot")
install_settings_patch(_BASE, "convrot")
_build_ltx25_args = install_generation_patch(_BASE, "convrot")
export_base_api(globals(), _BASE)


def is_available() -> bool:
    return bool(install_status("convrot").get("ok"))


def ltx25_install_status(root_dir=None):
    return install_status("convrot")
