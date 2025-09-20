# helpers/visualizer_catalog.py
# Registry of eye-candy visualizers to populate your selector.
# Import and use:
#   from helpers.visualizer_catalog import to_combo_items, find_by_key
#   for name, key in to_combo_items(): combo.addItem(name, userData=key)
#   spec = find_by_key(key)  # dict with metadata + defaults

from collections import OrderedDict

VISUALIZERS = [
    {
        "key": "kaleido_portal",
        "name": "Kaleido-Portal",
        "short": "Mirrored radial slices with BPM rotation + bloom.",
        "engine": "moderngl",
        "req": ["moderngl", "numpy"],
        "audio_map": {"bass->rotation_speed": 1.0, "mids->hue_shift": 0.8, "highs->bloom": 0.6},
        "defaults": {"slices": 10, "bloom": True, "hue_cycle": True, "rotate_per_bpm": True},
        "version": "1.0",
    },
    {
        "key": "lissajous_neon",
        "name": "Neon Lissajous XY",
        "short": "Stereo X/Y oscilloscope with neon trails.",
        "engine": "vispy",
        "req": ["vispy", "numpy"],
        "audio_map": {"rms->thickness": 1.0, "tempo->trail_decay": 0.7, "mids->color_cycle": 0.5},
        "defaults": {"trail_seconds": 3.0, "neon_trails": True, "thickness_base": 2.0},
        "version": "1.0",
    },
    {
        "key": "particle_flow",
        "name": "Particle Flow Field",
        "short": "Millions of particles advected by curl noise.",
        "engine": "moderngl",
        "req": ["moderngl", "numpy"],
        "audio_map": {"bass->flow_strength": 1.0, "highs->jitter": 0.8},
        "defaults": {"count": 500_000, "curl_noise": True, "chromatic": True},
        "version": "1.0",
    },
    {
        "key": "audio_fluids",
        "name": "Audio-Reactive Fluids",
        "short": "Stable fluids dye injection on kicks, curl on snares.",
        "engine": "taichi",
        "req": ["taichi", "numpy"],
        "audio_map": {"kick->dye", "snare->curl", "mids->viscosity"},
        "defaults": {"grid": 512, "dye_on_kick": True, "curl_on_snare": True},
        "version": "1.0",
    },
    {
        "key": "voronoi_shatter",
        "name": "Voronoi Shatter",
        "short": "Voronoi/Delaunay tessellation that cracks on beats.",
        "engine": "moderngl",
        "req": ["scipy", "numpy", "moderngl"],
        "audio_map": {"onset->split", "mids->cell_brightness", "bass->cell_scale"},
        "defaults": {"cells": 400, "crack_threshold": 0.7},
        "version": "1.0",
    },
    {
        "key": "spectral_galaxy",
        "name": "3D Spectral Galaxy",
        "short": "Instanced point galaxy; radius from frequency bins.",
        "engine": "moderngl",
        "req": ["moderngl", "numpy"],
        "audio_map": {"bass->z_wobble", "mids->radius", "highs->chroma"},
        "defaults": {"points": 120_000, "spiral_arms": 4, "depth_of_field": True},
        "version": "1.0",
    },
    {
        "key": "tunnel_warp",
        "name": "Tunnel / Starfield Warp",
        "short": "Raymarched vortex with twist and chromatic aberration.",
        "engine": "glsl_raymarch",
        "req": ["moderngl", "numpy"],
        "audio_map": {"kick->radius_pulse", "hihat->twist", "mids->aberration"},
        "defaults": {"vortex_segments": 64, "aberration": True, "fog": True},
        "version": "1.0",
    },
    {
        "key": "reaction_diffusion",
        "name": "Reaction-Diffusion (Gray–Scott)",
        "short": "Pattern bloom synced to bands and sections.",
        "engine": "compute_shader",
        "req": ["moderngl", "numpy"],
        "audio_map": {"mids->feed", "highs->kill", "sections->palette_shift"},
        "defaults": {"size": 768, "iterations_per_frame": 10, "palette_morph": True},
        "version": "1.0",
    },
    {
        "key": "metaball_blobs",
        "name": "Metaball Liquid Metal",
        "short": "3D blobs merge/split with sub-bass envelopes.",
        "engine": "moderngl",
        "req": ["moderngl", "numpy"],
        "audio_map": {"bass->merge_strength", "highs->surface_roughness"},
        "defaults": {"blob_count": 12, "ibl": True, "ssr": True},
        "version": "1.0",
    },
    {
        "key": "equalizer_city",
        "name": "Equalizer City",
        "short": "Infinite skyline; building heights from bands.",
        "engine": "moderngl",
        "req": ["moderngl", "numpy"],
        "audio_map": {"bands->building_heights", "transients->window_flicker"},
        "defaults": {"blocks": 64, "night_mode": True, "fog": True},
        "version": "1.0",
    },
]

def to_combo_items():
    """Return [(display_name, key), ...] for populating a QComboBox."""
    return [(v["name"], v["key"]) for v in VISUALIZERS]

def get_visualizers():
    return VISUALIZERS[:]

def find_by_key(key:str):
    for v in VISUALIZERS:
        if v["key"] == key:
            return v
    return None

__all__ = ["VISUALIZERS", "to_combo_items", "get_visualizers", "find_by_key"]
