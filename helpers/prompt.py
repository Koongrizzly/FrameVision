
from __future__ import annotations

import os, json, traceback, inspect, datetime, threading, random
from pathlib import Path
from typing import Any, Dict, Callable, List, Tuple, Optional

from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFormLayout, QDoubleSpinBox, QSpinBox, QLineEdit,
    QComboBox, QFileDialog, QMessageBox
)

# ---- Shared root path from app ----
try:
    from helpers.framevision_app import ROOT
except Exception:
    ROOT = Path(".").resolve()

SET_PATH = ROOT / "presets" / "setsave" / "prompt.json"
SET_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Qwen3-VL 2B local support ---
QWEN3_LOCAL_KEY = "__local_qwen3vl2b__"

def _qwen3_local_folder() -> Path:
    try:
        base = ROOT
    except Exception:
        base = Path(".").resolve()
    return (base / "models" / "describe" / "default" / "qwen3vl2b").resolve()

# --- Qwen3-VL 2B local model support ---
QWEN3_LOCAL_KEY = "__local_qwen3vl2b__"

def _qwen3_local_folder() -> Path:
    # Expected path per user: models\describe\default\qwen3vl2b (relative to app root)
    try:
        base = ROOT
    except Exception:
        base = Path(".").resolve()
    return (base / "models" / "describe" / "default" / "qwen3vl2b").resolve()

DEFAULT_STYLE = ""
DEFAULT_TEMPLATE_BASE = (
    "Expand the seed into one vivid, {length_words} single-sentence prompt for text-to-{target}. "
    "Include subject, environment, time of day, lighting, camera/lens, composition, action, textures, mood, and color palette. "
    "Return only the final prompt, no lists or markup."
)
DEFAULT_NEG = ""
OLD_DEFAULT_NEG = "text, watermark, vending machines, rubber ducks, disco balls, flamingos"
NEW_DEFAULT_NEG = "bad eyes, deformed body parts, bad teeth"

LENGTH_PRESETS = {
    "Short (40–60 words)": ( "40–60 words", 160 ),
    "Medium (80–120 words)": ( "80–120 words", 280 ),
    "Long (140–200 words)": ( "140–200 words", 420 ),
}


# Opening phrase variants used to vary the very start of the generated sentence
OPENING_VARIANTS = [
    "A cinematic shot of",
    "Close-up of",
    "Wide shot of",
    "Portrait of",
    "Overhead view of",
    "Low-angle view of",
    "Nighttime scene of",
    "Golden-hour scene of",
    "Macro shot of",
    "Editorial-style shot of",
    "Moody scene of",
    "Vivid scene of",
    "Dramatic view of",
    "Ultra-realistic photo of",
]


def _merge_negatives(base:str, extra:str)->str:
    "Combine and deduplicate comma-separated negatives while preserving order (case-insensitive)."
    parts = []
    for chunk in (base or "", extra or ""):
        if not chunk: 
            continue
        for t in [x.strip() for x in str(chunk).split(",")]:
            if t and t.lower() not in [p.lower() for p in parts]:
                parts.append(t)
    return ", ".join(parts)

# ---- Prompt Presets ----
# Each preset can inject style tags and optional negatives and guidance text added to the instruction template.
# New: Each preset may define a 'category' (People / Styles / Tech / Scenes) and 'defaults' for target/length/temperature/negatives.
PRESET_DEFS: Dict[str, Dict[str, Any]] = {
    "Default": {
        "style": "",
        "negatives": "watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed",
        "guide": "",
        "category": "Styles",
        "defaults": {"negatives": "watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed", },
    },
    # Core set
    "Architectural": {
        "style": "archviz, ultra-clean lines, PBR materials, global illumination, precise perspective, 35mm tilt-shift feel, concrete glass steel",
        "guide": "State the architectural style, materials, geometry, and camera height.",
        "category": "Tech",
    },
    "Anime": {
        "style": "anime, cel-shaded, crisp line art, dynamic posing, flat shading with gentle bloom, bokeh sparkles, vibrant color palette",
        "guide": "Emphasize emotive expressions and iconic framing; keep lines clean.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.9, "negatives":"muddy shading, painterly smears, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Landscapes": {
        "style": "landscape photography, wide-angle 16–24mm, sweeping vista, atmospheric perspective, weather detail, golden hour light",
        "guide": "Describe foreground, midground, background, and sky explicitly.",
        "category": "Scenes",
    },
    "Macro Nature": {
        "style": "macro photography, 100mm lens, extreme close-up, shallow depth of field, fine textures, dew drops, backlit translucency",
        "guide": "Call out scale and tiny environmental context like moss, bark, or petals.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.75, "negatives":"focus stacking artifacts, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Photo Shoot": {
        "style": "studio photography, 3-point lighting with softboxes and rim light, seamless backdrop, editorial styling, 85mm lens, f/2 portrait look",
        "guide": "Specify pose, wardrobe, makeup/hair, backdrop color, and lighting setup.",
        "category": "People",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.7, "negatives":"overexposure, blown highlights, harsh shadows, asymmetrical eyes, cross-eyed, lazy eye, extra eyes, deformed eyes, blank or empty eyes, malformed nose, wide nostrils, nostril deformity, extra nostrils, crooked nose, fused nose, deformed nose, missing fingers, extra fingers, fused fingers, disfigured hands, deformed hands, long fingers, short fingers, missing hand, extra hand, deformed feet, fused toes, extra toes, missing toes, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Pixar-style 3D": {
        "style": "3D-rendered Pixar-style, soft global illumination, subsurface scattering, stylized proportions, expressive eyes, filmic color grading, depth of field, volumetric god rays",
        "guide": "Lean into playful charm and clean shapes; keep it wholesome and cinematic.",
        "category": "Styles",
    },
    "Product Render": {
        "style": "product photography, hero shot, seamless infinity backdrop, studio softboxes, specular control cards, fingerprint-free surfaces",
        "guide": "Include packaging/branding, surface finish, and any props for scale.",
        "category": "Tech",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.65, "negatives":"dirt, scratches, fingerprints, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Cyberpunk": {
        "style": "cyberpunk neon, rain-slick streets, holograms, chromatic aberration, dramatic rim lighting, night city haze, flying particles",
        "guide": "Use moody nighttime ambiance and dense urban detail.",
        "category": "Styles",
    },
    "Ultra-Realistic": {
        "style": "photorealistic, ultra-detailed, 8k, physically based rendering, natural color science, realistic imperfections, fine surface texture",
        "guide": "Prefer natural light behavior and believable materials.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"cartoonish, over-smooth skin, plastic look, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },

    # User additions (with categories + defaults)
    "Astrophotography": {
        "style": "night sky long exposure, Milky Way core, starfields, faint nebulae, foreground silhouette, light pollution control",
        "guide": "Use for: epic night landscapes.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.7, "negatives":"fake neon stars, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Blueprint / Tech Drawing": {
        "style": "blueprint schematic, orthographic projection, white linework on cobalt grid, callouts/labels",
        "guide": "Use for: products, architecture plans.",
        "category": "Tech",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.65, "negatives":"perspective lens effects, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Cinematic Film Still (Anamorphic)": {
        "style": "cinematic film still, anamorphic bokeh, lens breathing, shallow DoF, natural grain, 35/50mm, golden hour",
        "guide": "Use for: moody, movie-like frames.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"harsh HDR, oversaturation, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Claymation / Stop-Motion": {
        "style": "claymation look, thumbprint texture, tiny set lighting, shallow tabletop DoF, handmade charm",
        "guide": "Use for: cute, tactile characters.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.85, "negatives":"hyper-smooth surfaces, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Fashion Editorial": {
        "style": "glossy editorial, beauty lighting, gel accents, couture styling, 85mm f/2, magazine cover layout hints",
        "guide": "Use for: people, outfits, makeup.",
        "category": "People",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.75, "negatives":"skin smoothing/plastic look, asymmetrical eyes, cross-eyed, lazy eye, extra eyes, deformed eyes, blank or empty eyes, malformed nose, wide nostrils, nostril deformity, extra nostrils, crooked nose, fused nose, deformed nose, missing fingers, extra fingers, fused fingers, disfigured hands, deformed hands, long fingers, short fingers, missing hand, extra hand, deformed feet, fused toes, extra toes, missing toes, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Food & Beverage": {
        "style": "food photography, soft top-down + bounce, steam highlights, appetizing color science, props for context",
        "guide": "Use for: dishes, drinks, packaging.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.75, "negatives":"cold/gray tones, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Noir / Monochrome": {
        "style": "black-and-white noir, high contrast, hard key light, venetian shadows, 1940s vibe, film grain",
        "guide": "Use for: dramatic characters and city alleys.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.7, "negatives":"color tint, soft focus, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Real-Estate Interior": {
        "style": "interior archviz, balanced white, bounce-lit, window highlight control, verticals corrected, 16–24mm",
        "guide": "Use for: rooms, décor.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.7, "negatives":"blown windows, barrel distortion, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Sports / Action Burst": {
        "style": "high shutter speed freeze, sideline angle, motion trails, stadium lights, dynamic crop",
        "guide": "Use for: athletes, stunts.",
        "category": "Scenes",
        "defaults": {"target":"video", "length":"Short (40–60 words)", "temperature":0.9, "negatives":"motion blur smear, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Street / Documentary": {
        "style": "candid street photo, 35mm, available light, mild film grain, imperfect framing, real-world textures",
        "guide": "Use for: authentic people/places.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.75, "negatives":"staged studio vibe, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Synthwave / Vaporwave": {
        "style": "retro-futurist neon, gradient sun, grid horizon, chromatic glow, 80s album cover energy",
        "guide": "Use for: bold, graphic posters.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Short (40–60 words)", "temperature":0.85, "negatives":"muddy colors, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"},
    },
    "Underwater": {
        "style": "underwater photography, caustic light patterns, suspended particles, blue-green shift, dome port perspective",
        "guide": "Use for: oceans, pools, surreal swims.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"dry studio look, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Wildlife Documentary": {
        "style": "nature docu look, 400mm telephoto compression, dust-in-air, natural behavior, field textures",
        "guide": "Use for: animals in motion.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"studio lighting, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"},
    },
    "Disco": {
        "style": "disco-themed, vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed",
        "negatives": "minimalist, rustic, monochrome, contemporary, simplistic, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed",
        "guide": "Use for: retro nightclub, dance floor, party scenes.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.9, "negatives":"minimalist, rustic, monochrome, contemporary, simplistic, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"}
    },
    "Alien": {
        "style": "extraterrestrial, cosmic, otherworldly, mysterious, sci-fi, highly detailed",
        "negatives": "earthly, mundane, common, realistic, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed",
        "guide": "Use for: strange planets, UFOs, cosmic encounters.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.85, "negatives":"earthly, mundane, common, realistic, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"}
    },
    "HDR Photo": {
        "style": "high dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed",
        "negatives": "flat, low contrast, oversaturated, underexposed, overexposed, blurred, noisy, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon",
        "guide": "Use for: punchy photography with extended dynamic range.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"flat, low contrast, oversaturated, underexposed, overexposed, blurred, noisy, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"}
    },
    "Futuristic": {
        "style": "sleek, modern, ultramodern, high tech, detailed",
        "negatives": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vintage, antique, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed",
        "guide": "Use for: sci‑fi design, advanced interfaces, cityscapes.",
        "category": "Tech",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.85, "negatives":"ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vintage, antique, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed"}
    },
    "Artstyle Hyperrealism": {
        "style": "extremely high‑resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike",
        "negatives": "simplified, abstract, unrealistic, impressionistic, low resolution, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon",
        "guide": "Use for: lifelike portraits/products with extreme detail.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"simplified, abstract, unrealistic, impressionistic, low resolution, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"}
    },
    "Artstyle Graffiti": {
        "style": "street art, vibrant, urban, detailed, tag, mural",
        "negatives": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed",
        "guide": "Use for: walls, street scenes, bold tags and murals.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.9, "negatives":"ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed"}
    },
    "Artstyle Cubist": {
        "style": "geometric shapes, abstract, innovative, revolutionary",
        "negatives": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed",
        "guide": "Use for: fragmented forms, multiple viewpoints.",
        "category": "Styles",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.85, "negatives":"anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed"}
    },
    "Automotive": {
        "style": "sleek, dynamic, professional, commercial, vehicle‑focused, high‑resolution, highly detailed",
        "negatives": "noisy, blurry, unattractive, sloppy, unprofessional, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon",
        "guide": "Use for: car ads, beauty shots, motion rigs.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"noisy, blurry, unattractive, sloppy, unprofessional, watermark, signature, text, logo, low-res, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"}
    },
    "Car Enthusiast": {
        "style": "natural light car photography, tasteful angles, rolling shot or panning blur, golden hour, back roads or urban meet, clean reflections, high-resolution detail",
        "negatives": "studio light sweep, showroom backdrop, ad layout, heavy logo placement, overpolished paint, exaggerated reflections, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon",
        "guide": "Use for: beautiful cars in real settings (meets, roads, garages), not ad/commercial layouts.",
        "category": "Scenes",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"studio light sweep, showroom backdrop, ad layout, heavy logo placement, overpolished paint, exaggerated reflections, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"}
    },
    "Ultra-Real Animals": {
        "style": "ultra‑realistic photography, animals in tailored human clothing, natural fabric texture, plausible posture and gesture, candid human-like activity, skin/fur detail, cinematic depth of field",
        "negatives": "costume seams, mascot suit, plush, fursuit, taxidermy, uncanny valley, caricature, exaggerated features, asymmetrical eyes, cross-eyed, lazy eye, extra eyes, deformed eyes, blank or empty eyes, malformed nose, wide nostrils, nostril deformity, extra nostrils, crooked nose, fused nose, deformed nose, missing fingers, extra fingers, fused fingers, disfigured hands, deformed hands, long fingers, short fingers, missing hand, extra hand, deformed feet, fused toes, extra toes, missing toes, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon",
        "guide": "Use for: animals dressed and acting like people, but shot like serious photojournalism.",
        "category": "People",
        "defaults": {"target":"image", "length":"Medium (80–120 words)", "temperature":0.8, "negatives":"costume seams, mascot suit, plush, fursuit, taxidermy, uncanny valley, caricature, exaggerated features, asymmetrical eyes, cross-eyed, lazy eye, extra eyes, deformed eyes, blank or empty eyes, malformed nose, wide nostrils, nostril deformity, extra nostrils, crooked nose, fused nose, deformed nose, missing fingers, extra fingers, fused fingers, disfigured hands, deformed hands, long fingers, short fingers, missing hand, extra hand, deformed feet, fused toes, extra toes, missing toes, watermark, signature, text, logo, low-res, blurry, jpeg artifacts, heavy noise, overprocessed, anime, manga, cartoon, cel-shaded, line art, illustration, chibi, kawaii, toon"}
    }
}

CATEGORIES_ORDER = ["All", "People", "Styles", "Tech", "Scenes"]

# ---------- Lightweight caching for model + processor ----------
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()

def _lines_to_px(widget, lines:int)->int:
    fm = widget.fontMetrics()
    return max(48, int(lines * fm.lineSpacing()) + 8)

def _load_settings()->Dict[str, Any]:
    try:
        if SET_PATH.exists():
            data = json.loads(SET_PATH.read_text(encoding="utf-8"))
            try:
                neg = (data.get("negatives") or "").strip()
                legacy = OLD_DEFAULT_NEG.lower()
                cur = neg.lower()
                def _norm_set(s):
                    return {x.strip() for x in s.split(",") if x.strip()}
                if cur == legacy or _norm_set(cur) == _norm_set(legacy):
                    data["negatives"] = NEW_DEFAULT_NEG
            except Exception:
                pass
            try:
                st = (data.get("style") or "").strip()
                if st and "pixar" in st.lower():
                    data["style"] = ""
            except Exception:
                pass
            if "preset" not in data:
                data["preset"] = "Default"
            if "favorites" not in data:
                data["favorites"] = []
            if "preset_category" not in data:
                data["preset_category"] = "All"
            if "last_used_preset" not in data:
                data["last_used_preset"] = data.get("preset","Default")
            return data
    except Exception:
        pass
    return {
        "style": DEFAULT_STYLE,
        "length_choice": "Medium (80–120 words)",
        "target": "image",
        "model_key": "",
        "negatives": DEFAULT_NEG,
        "temperature": 0.85,
        "max_new_tokens": LENGTH_PRESETS["Medium (80–120 words)"][1],
        "seed_text": "",
        "preset": "Default",
        "favorites": [],
        "preset_category": "All",
        "last_used_preset": "Default",
    }

def _save_settings(data:Dict[str, Any])->None:
    try:
        SET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ---------- Model discovery & generation ----------

def _list_qwen_vl_models() -> List[Tuple[str,str]]:
    """
    Discover available Qwen-VL models, including the new local Qwen3‑VL 2B folder
    at models/describe/default/qwen3vl2b.
    """
    out: List[Tuple[str, str]] = []
    # Prefer the new local Qwen3‑VL 2B if present
    try:
        q3 = _qwen3_local_folder()
        if q3.exists() and any(q3.iterdir()):
            out.append((QWEN3_LOCAL_KEY, "Qwen3-VL 2B (local)"))
    except Exception:
        pass
    # Also include any engines from helpers.describer / describer catalog
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
        cat = getattr(D, "ENGINE_CATALOG", {})
        for k, meta in cat.items():
            t = str(meta.get("type","")).lower()
            if t in ("hf_qwen3vl","hf_qwen3_vl","qwen3vl","qwen3_vl","hf_qwen2vl","hf_qwen2_vl","qwen2vl","qwen_vl"):
                label = meta.get("label", k)
                if (k, label) not in out:
                    out.append((k, label))
    except Exception:
        pass
    return out

def _models_root_path()->Optional[Path]:
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
    except Exception:
        return None
    try:
        return Path(getattr(D, "models_root")())
    except Exception:
        return None

def _folder_for_model_key(model_key:str)->Optional[Path]:
    try:
        try:
            import helpers.describer as D  # type: ignore
        except Exception:
            import describer as D  # type: ignore
    except Exception:
        return None
    try:
        meta = getattr(D, "ENGINE_CATALOG")[model_key]
        root = _models_root_path()
        if root is None: return None
        return root / meta.get("folder","")
    except Exception:
        return None

def _choose_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        try:
            import torch_directml as _dml  # type: ignore
            return _dml.device(), torch.float32
        except Exception:
            return "cpu", torch.float32
    except Exception:
        return "cpu", None

def _get_qwen_text_model(model_path: Path):
    key = str(model_path.resolve())
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
    device, dtype = _choose_device()
    try:
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLMModel
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq as _VLMModel
            except Exception:
                _VLMModel = None  # type: ignore
        if _VLMModel is None:
            raise RuntimeError("Transformers model class not available")
    except Exception as e:
        raise RuntimeError(f"Transformers unavailable: {e}")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True
    )
    model = _VLMModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)
    try:
        model.eval()
    except Exception:
        pass

    with _MODEL_LOCK:
        _MODEL_CACHE[key] = (processor, model, device, dtype)
    return processor, model, device, dtype

def _generate_with_qwen_text(
    model_path: Path,
    system_prompt:str,
    user_prompt:str,
    temperature:float,
    max_new_tokens:int,
    cancel_check: Optional[Callable[[], bool]] = None
)->str:
    if model_path is None or not (model_path.exists() and any(model_path.iterdir())):
        raise RuntimeError("Model folder not found")
    try:
        processor, model, device, dtype = _get_qwen_text_model(model_path)
    except Exception as e:
        raise

    messages = []
    if system_prompt:
        messages.append({"role":"system","content":[{"type":"text","text": system_prompt}]})
    messages.append({"role":"user","content":[{"type":"text","text": user_prompt}]})

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat_text], return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]
    try:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}
    except Exception:
        pass

    stopping_criteria = None
    try:
        if cancel_check is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class _CancelStopper(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    try:
                        return bool(cancel_check())
                    except Exception:
                        return False
            stopping_criteria = StoppingCriteriaList([_CancelStopper()])
    except Exception:
        stopping_criteria = None

    import torch
    with torch.inference_mode():
        # Fresh randomness per Generate click
        try:
            import time as _t, random as _r
            _seed = int(_t.time_ns() % 2147483647)
            _r.seed(_seed)
            try:
                import numpy as _np
                _np.random.seed(_seed)
            except Exception:
                pass
            try:
                torch.manual_seed(_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_seed)
            except Exception:
                pass
        except Exception:
            pass
        # Slight jitter to sampling params to ensure variety even if a backend is overly deterministic
        _temp = float(max(0.01, temperature or 0.7))
        _top_p = 0.9
        _rep = 1.1
        try:
            import random as _r
            _temp *= _r.uniform(0.96, 1.08)
            _top_p *= _r.uniform(0.96, 1.03)
            _rep   *= _r.uniform(0.97, 1.05)
            _temp = min(max(_temp, 0.2), 1.6)
            _top_p = min(max(_top_p, 0.75), 0.99)
            _rep   = min(max(_rep, 1.0), 1.3)
        except Exception:
            pass
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=_temp,
            top_p=_top_p,
            max_new_tokens=int(max_new_tokens or 200),
            repetition_penalty=_rep,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria
        )

    seq = out.sequences if hasattr(out, "sequences") else out
    new_ids = seq[:, prompt_len:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return text

# ---------- Worker (non-blocking) ----------
class _PromptGenWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_folder: Path, system_prompt: str, user_prompt: str, temperature: float, max_new_tokens: int):
        super().__init__()
        self._model_folder = model_folder
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = _generate_with_qwen_text(
                self._model_folder,
                self._system_prompt,
                self._user_prompt,
                self._temperature,
                self._max_new_tokens,
                cancel_check=lambda: self._cancel
            )
            if self._cancel:
                self.failed.emit("Cancelled")
                return
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

# ---------- UI ----------
class PromptToolPane(QWidget):
    """
    Prompt Generator that expands short seeds into long prompts.
    - Saves settings to presets/setsave/prompt.json
    - Lets you choose Length, Target (image/video), Model (Qwen2-VL engines discovered), and Presets
    - Favorite presets (⭐) pin to top; remembers last used
    - Category filter: All / People / Styles / Tech / Scenes
    - Per-preset defaults: target, length, temperature, negatives (applied when preset changes)
    - Runs generation in a background thread with Cancel support and a small model cache.
    """
    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        self._saving_timer = QTimer(self); self._saving_timer.setInterval(600); self._saving_timer.setSingleShot(True)
        self._saving_timer.timeout.connect(self._save_now)

        self._thread: Optional[QThread] = None
        self._last_combined_neg: str = ""
        self._worker: Optional[_PromptGenWorker] = None

        self.state = _load_settings()
        try:
            self.destroyed.connect(self._teardown)
        except Exception:
            pass

        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(8)

        # ---- Controls ----
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignLeft); form.setFormAlignment(Qt.AlignLeft|Qt.AlignTop)

        # Model & target
        self.combo_model = QComboBox()
        models = _list_qwen_vl_models()
        if not models:
            self.combo_model.addItem("Qwen2-VL (auto)", "")
        else:
            for k,label in models:
                self.combo_model.addItem(label, k)
        want_key = self.state.get("model_key","")
        if want_key:
            idx = max(0, self.combo_model.findData(want_key))
            self.combo_model.setCurrentIndex(idx)

        # Target
        self.combo_target = QComboBox(); self.combo_target.addItems(["image","video"])
        tgt = (self.state.get("target") or "image").lower()
        if tgt in ("image","video"):
            self.combo_target.setCurrentText(tgt)

        # Category filter
        self.combo_category = QComboBox()
        for c in CATEGORIES_ORDER:
            self.combo_category.addItem(c)
        cat = self.state.get("preset_category","All")
        if cat not in CATEGORIES_ORDER:
            cat = "All"
        self.combo_category.setCurrentText(cat)

        # Presets (will be populated via rebuild; Default pinned)
        self.combo_preset = QComboBox()

        # Favorite star
        self.btn_star = QPushButton("☆")
        self.btn_star.setFixedWidth(28)
        self.btn_star.setToolTip("Toggle favorite")

        # Put Target, Category, Presets, Star on the same row
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Target"))
        top_row.addWidget(self.combo_target)
        top_row.addSpacing(12)
        top_row.addWidget(QLabel("Category"))
        top_row.addWidget(self.combo_category)
        top_row.addSpacing(12)
        top_row.addWidget(QLabel("Presets"))
        top_row.addWidget(self.combo_preset, 1)
        top_row.addWidget(self.btn_star)
        top_row.addStretch(1)

        # Length preset
        self.combo_len = QComboBox()
        for label in LENGTH_PRESETS.keys():
            self.combo_len.addItem(label)
        self.combo_len.setCurrentText(self.state.get("length_choice","Medium (80–120 words)"))

        # Style/negatives and generation params
        self.style = QLineEdit(self.state.get("style", DEFAULT_STYLE)); self.style.setPlaceholderText("Style tags (Cinematic, Photoreal, Anime)…")
        self.neg = QLineEdit(self.state.get("negatives", DEFAULT_NEG)); self.neg.setPlaceholderText("Negatives (optional; leave empty if not needed)")
        self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setSingleStep(0.05); self.temp.setValue(float(self.state.get("temperature", 0.85)))
        self.max_new = QSpinBox(); self.max_new.setRange(64, 4096); self.max_new.setValue(int(self.state.get("max_new_tokens", 280)))

        form.addRow("Model", self.combo_model)
        form.addRow(top_row)
        form.addRow("Length", self.combo_len)
        form.addRow("Style", self.style)
        form.addRow("Negatives", self.neg)
        row_params = QHBoxLayout(); row_params.addWidget(QLabel("Temperature")); row_params.addWidget(self.temp); row_params.addSpacing(12); row_params.addWidget(QLabel("Max tokens")); row_params.addWidget(self.max_new); row_params.addStretch(1)
        form.addRow(row_params)

        root.addLayout(form)

        # ---- Seed / Buttons / Result ----
        self.seed = QTextEdit(self.state.get("seed_text",""))
        self.seed.setPlaceholderText("Enter seed words, e.g. 'a cat in a tree'")
        self.seed.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.seed.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.seed.setFixedHeight(_lines_to_px(self.seed, 2))
        root.addWidget(QLabel("Seed words"))
        root.addWidget(self.seed)

        btns = QHBoxLayout()
        self.btn_gen = QPushButton("Generate")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_copy = QPushButton("Copy result")
        self.btn_clear = QPushButton("Clear")
        self.btn_save_txt = QPushButton("Save .txt")
        self.btn_save_json = QPushButton("Save .json")
        for b in (self.btn_gen, self.btn_cancel, self.btn_copy, self.btn_clear, self.btn_save_txt, self.btn_save_json):
            btns.addWidget(b)
        btns.addStretch(1)
        # Variation dial + synonym shuffler
        self.var_count = QSpinBox(); self.var_count.setRange(1, 8); self.var_count.setValue(int(self.state.get("var_count", 3)))
        self.btn_rephrase = QPushButton("Rephrase")
        self.btn_shuffle = QPushButton("Shuffle Synonyms")
        btns.addSpacing(12)
        btns.addWidget(QLabel("Variations"))
        btns.addWidget(self.var_count)
        btns.addWidget(self.btn_rephrase)
        btns.addWidget(self.btn_shuffle)
        root.addLayout(btns)

        self.out = QTextEdit("")
        self.out.setReadOnly(True)
        self.out.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.out.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.out.setFixedHeight(_lines_to_px(self.out, 15))
        root.addWidget(QLabel("Result"))
        root.addWidget(self.out)
        
        # New: separate Negatives box (no longer appended to the prompt itself)
        self.out_neg = QTextEdit("")
        self.out_neg.setReadOnly(True)
        self.out_neg.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.out_neg.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.out_neg.setFixedHeight(_lines_to_px(self.out_neg, 3))
        root.addWidget(QLabel("Negatives"))
        root.addWidget(self.out_neg)

        # ---- populate + hooks ----
        self._rebuild_preset_combo()
        self.combo_preset.setCurrentText(self.state.get("preset","Default"))
        self._sync_star_button()

        self.btn_star.clicked.connect(self._toggle_favorite)

        self.combo_category.currentIndexChanged.connect(self._on_category_changed)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)

        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_copy.clicked.connect(lambda: self._copy_to_clipboard())
        self.btn_clear.clicked.connect(self._clear_outputs)
        self.btn_save_txt.clicked.connect(self._save_txt)
        self.btn_save_json.clicked.connect(self._save_json)

        self.btn_rephrase.clicked.connect(self._on_rephrase_click)
        self.btn_shuffle.clicked.connect(self._on_shuffle_click)
        self.var_count.valueChanged.connect(self._schedule_save)

        for w in (self.style, self.neg, self.combo_model, self.combo_len, self.combo_target, self.combo_preset):
            try:
                w.currentIndexChanged.connect(self._schedule_save)  # for combos
            except Exception:
                w.textChanged.connect(self._schedule_save)  # for lineedits
        self.temp.valueChanged.connect(self._schedule_save)
        self.max_new.valueChanged.connect(self._schedule_save)
        self.seed.textChanged.connect(self._schedule_save)

        self._user_touched_tokens = False
        self._variations_list = []
        self._variation_index = 0
        self.max_new.valueChanged.connect(lambda *_: setattr(self, "_user_touched_tokens", True))
        self.combo_len.currentIndexChanged.connect(self._maybe_adjust_tokens)

    # ---------- Favorites & categories ----------
    def _on_category_changed(self):
        self.state["preset_category"] = self.combo_category.currentText()
        self._rebuild_preset_combo()
        self._schedule_save()

    def _is_favorite(self, name:str)->bool:
        favs = set(self.state.get("favorites",[]) or [])
        return name in favs

    def _toggle_favorite(self):
        name = self.combo_preset.currentText()
        if not name:
            return
        favs = set(self.state.get("favorites",[]) or [])
        if name in favs:
            favs.remove(name)
        else:
            favs.add(name)
        self.state["favorites"] = sorted(list(favs), key=str.lower)
        self._rebuild_preset_combo(keep_current=True)
        self._sync_star_button()
        self._schedule_save()

    def _sync_star_button(self):
        name = self.combo_preset.currentText()
        on = self._is_favorite(name)
        self.btn_star.setText("★" if on else "☆")
        self.btn_star.setToolTip("Unfavorite" if on else "Favorite")

    def _rebuild_preset_combo(self, keep_current:bool=False):
        cur = self.combo_preset.currentText()
        cat_filter = self.combo_category.currentText() or "All"
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        # Always pin Default
        self.combo_preset.addItem("Default")
        # Build candidates by category
        names = [k for k in PRESET_DEFS.keys() if k != "Default"]
        if cat_filter != "All":
            names = [n for n in names if PRESET_DEFS.get(n,{}).get("category","Styles") == cat_filter]
        # Split into favorites and others
        favs = [n for n in names if self._is_favorite(n)]
        non = [n for n in names if not self._is_favorite(n)]
        for lst in (sorted(favs, key=str.lower), sorted(non, key=str.lower)):
            for n in lst:
                self.combo_preset.addItem(n)
        # Restore selection
        if keep_current and cur:
            idx = self.combo_preset.findText(cur)
            if idx >= 0:
                self.combo_preset.setCurrentIndex(idx)
        self.combo_preset.blockSignals(False)

    # ---------- Preset change behavior ----------
    def _on_preset_changed(self):
        name = self.combo_preset.currentText()
        self.state["preset"] = name
        self.state["last_used_preset"] = name
        # Apply per-preset defaults (target, length, temperature, negatives) if available
        preset = PRESET_DEFS.get(name, {})
        defaults = preset.get("defaults", {}) or {}
        if isinstance(defaults, dict):
            tgt = defaults.get("target")
            if tgt in ("image","video"):
                self.combo_target.setCurrentText(tgt)
            length = defaults.get("length")
            if length in LENGTH_PRESETS:
                self.combo_len.setCurrentText(length)
                if not self._user_touched_tokens:
                    # also auto-adjust tokens for new length
                    _, tokens = LENGTH_PRESETS[length]
                    self.max_new.setValue(tokens)
            temp = defaults.get("temperature")
            if isinstance(temp, (int,float)):
                self.temp.setValue(float(temp))
            negs = defaults.get("negatives")
            if isinstance(negs, str) and negs.strip():
                self.neg.setText(negs.strip())
        # Update star
        self._sync_star_button()
        self._schedule_save()

    def _maybe_adjust_tokens(self):
        if not self._user_touched_tokens:
            _, tokens = LENGTH_PRESETS[self.combo_len.currentText()]
            self.max_new.setValue(tokens)

    # ---- persistence ----
    def _schedule_save(self):
        self._saving_timer.start()

    def _save_now(self):
        data = {
            "style": self.style.text().strip(),
            "length_choice": self.combo_len.currentText(),
            "target": self.combo_target.currentText(),
            "model_key": self.combo_model.currentData() or "",
            "negatives": self.neg.text().strip(),
            "applied_negatives": self._last_combined_neg,
            "temperature": float(self.temp.value()),
            "max_new_tokens": int(self.max_new.value()),
            "seed_text": self.seed.toPlainText(),
            "preset": self.combo_preset.currentText(),
            "favorites": self.state.get("favorites",[]),
            "preset_category": self.combo_category.currentText(),
            "last_used_preset": self.state.get("last_used_preset", self.combo_preset.currentText()),
            "var_count": int(self.var_count.value()),
        }
        _save_settings(data)

    # ---- generation ----
    def _compose_prompts(self, seed:str)->tuple[str,str,str]:
        target = self.combo_target.currentText().strip().lower()
        if target not in ("image","video"):
            target = "image"
        length_label = self.combo_len.currentText()
        length_words, _ = LENGTH_PRESETS.get(length_label, LENGTH_PRESETS["Medium (80–120 words)"])
        template = DEFAULT_TEMPLATE_BASE.format(length_words=length_words, target=("image" if target=="image" else "video"))

        # Pull base style/negatives from UI
        style = self.style.text().strip() or DEFAULT_STYLE
        negatives = self.neg.text().strip() or DEFAULT_NEG

        # Video adds motion cues to nudge the model
        if target == "video":
            style = f"{style}, cinematic, motion-aware, dynamic framing".strip(", ")

        # Apply preset augmentations (non-destructive for fields)
        preset_name = self.combo_preset.currentText().strip() or "Default"
        preset = PRESET_DEFS.get(preset_name, PRESET_DEFS["Default"])
        p_style = (preset.get("style","") or "").strip()
        p_negs = (preset.get("negatives","") or "").strip()
        p_guide = (preset.get("guide","") or "").strip()

        if p_style:
            style = (f"{style}, {p_style}" if style else p_style)
        if p_negs:
            negatives = _merge_negatives(negatives, p_negs) if negatives else p_negs
        if p_guide:
            template = template + " " + p_guide



        # --- Opening phrase rotation to vary the very first tokens ---


        try:


            import random as _r


            opener = _r.choice(OPENING_VARIANTS)


        except Exception:


            opener = "A cinematic shot of"


        


        # Light style synonym shuffle / rotation so presets don't pin the start


        try:


            import random as _r


            if _r.random() < 0.5:


                style = self._shuffle_synonyms(style, 1)


            segs = [s.strip() for s in (style or "").split(",") if s.strip()]


            if len(segs) > 2 and _r.random() < 0.4:


                k = _r.randint(1, min(2, len(segs)-1))


                style = ", ".join(segs[k:] + segs[:k])


        except Exception:


            pass
        sys = (
            "You are a visual prompt engineer. "
            "Expand short seeds into a single richly detailed prompt. "
            "Follow the instruction template and style hints exactly."
        )
        import re as _re
        _neg_clean = negatives if _re.search(r"[A-Za-z0-9]", negatives) else ""
        user = (
            (f"{template} ") 
            + (f"Begin the single sentence with {opener} (no quotes). ")
            + (f"Use the style: {style}. " if style else "")
            + (f"Seed: {seed}")
        )
        # Return combined negatives separately (no longer appended to prompt)
        return sys, user, _neg_clean

    def _set_busy(self, busy: bool):
        self.btn_gen.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        self.btn_copy.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        self.btn_save_txt.setEnabled(not busy)
        self.btn_save_json.setEnabled(not busy)
        self.combo_model.setEnabled(not busy)
        self.combo_target.setEnabled(not busy)
        self.combo_preset.setEnabled(not busy)
        self.combo_category.setEnabled(not busy)
        self.combo_len.setEnabled(not busy)
        self.style.setEnabled(not busy)
        self.neg.setEnabled(not busy)
        self.temp.setEnabled(not busy)
        self.max_new.setEnabled(not busy)
        self.seed.setEnabled(not busy)
        try:
            self.out_neg.setEnabled(not busy)
        except Exception:
            pass
        self.btn_gen.setText("Generating…" if busy else "Generate")
    # ---------- Variation & Synonym tools ----------
    _SYNMAP = {
        "vibrant": ["vivid", "lively", "saturated"],
        "moody": ["atmospheric", "brooding", "dramatic"],
        "cinematic": ["filmic", "movie-like", "cinema-grade"],
        "detailed": ["intricate", "finely detailed", "high detail"],
        "realistic": ["lifelike", "naturalistic", "true to life"],
        "clean": ["pristine", "polished", "refined"],
        "dramatic": ["striking", "intense", "high-impact"],
        "soft": ["gentle", "subtle", "delicate"],
        "neon": ["luminous", "glowing", "electric"],
        "futuristic": ["cutting-edge", "ultramodern", "forward-looking"],
        "highly detailed": ["richly detailed", "finely textured", "ultra-detailed"],
        "photorealistic": ["photo-real", "hyper-real", "true to life"],
        "beautiful": ["stunning", "gorgeous", "eye-catching"],
        "sharp": ["crisp", "tack-sharp", "pin-sharp"],
        "colorful": ["vivid", "chromatic", "richly colored"],
    }

    def _shuffle_synonyms(self, text, max_swaps):
        try:
            s = (text or "").strip()
            if not s: return text
            pairs = []
            low = s.lower()
            for key, alts in self._SYNMAP.items():
                if key in low:
                    import random as _r
                    pairs.append((key, _r.choice(alts)))
            import random as _r
            _r.shuffle(pairs)
            swaps = 0
            out = s
            import re as _re
            for key, repl in pairs:
                if swaps >= int(max_swaps or 1):
                    break
                pattern = _re.compile(_re.escape(key), _re.IGNORECASE)
                out = pattern.sub(repl, out, count=1)
                swaps += 1
            return out
        except Exception:
            return text

    def _segments(self, s):
        try:
            import re as _re
            parts = _re.split(r';|,|\band\b', s)
            parts = [p.strip() for p in parts if p and p.strip()]
            return parts if parts else [s]
        except Exception:
            return [s]

    def _compose_seed_variant(self, base):
        try:
            parts = self._segments(base)
            import random as _r
            if len(parts) > 1:
                if _r.random() < 0.5:
                    k = _r.randint(1, max(1, len(parts)-1))
                    parts = parts[k:] + parts[:k]
                else:
                    i = _r.randrange(len(parts))
                    j = _r.randrange(len(parts))
                    parts[i], parts[j] = parts[j], parts[i]
            draft = ", ".join(parts)
            swaps = _r.randint(1, 3)
            draft = self._shuffle_synonyms(draft, swaps)
            boosters = [" richly textured", " with thoughtful composition", " under nuanced lighting", " with natural imperfections"]
            if _r.random() < 0.5:
                draft += _r.choice(boosters)
            import re as _re
            return _re.sub(r'\s+', ' ', draft).strip()
        except Exception:
            return base

    def _make_variations(self, seed, n):
        seen = set()
        out = []
        tries = 0
        base = seed or ""
        while len(out) < int(n or 1) and tries < max(10, int(n or 1)*10):
            tries += 1
            v = self._compose_seed_variant(base)
            low = (v or "").lower().strip()
            if low and low not in seen and low != base.lower().strip():
                out.append(v)
                seen.add(low)
        if not out:
            out = [base]
        return out

    def _on_rephrase_click(self):
        seed = (self.seed.toPlainText() or "").strip()
        if not seed:
            try:
                QMessageBox.warning(self, "Rephrase", "Please enter a seed first.")
            except Exception:
                pass
            return
        import random as _r
        n = int(self.var_count.value())
        if not self._variations_list or self._variation_index >= len(self._variations_list):
            _r.seed()
            self._variations_list = self._make_variations(seed, n)
            self._variation_index = 0
        variant = self._variations_list[self._variation_index]
        self._variation_index += 1
        self.seed.setPlainText(variant)
        self._schedule_save()

    def _on_shuffle_click(self):
        try:
            swaps = max(1, min(5, int(self.var_count.value())))
        except Exception:
            swaps = 2
        try:
            self.style.setText(self._shuffle_synonyms(self.style.text(), swaps))
        except Exception:
            pass
        try:
            cur = (self.seed.toPlainText() or "").strip()
            if cur:
                self.seed.setPlainText(self._shuffle_synonyms(cur, 1))
        except Exception:
            pass
        self._schedule_save()


    def _on_generate(self):
        if self._thread is not None:
            return
        seed = (self.seed.toPlainText() or "").strip()
        if not seed:
            QMessageBox.warning(self, "Prompt", "Please enter a seed (e.g., 'a cat').")
            return
        sys, usr, combined_negs = self._compose_prompts(seed)
        self._last_combined_neg = combined_negs
        model_key = self.combo_model.currentData() or ""
        folder = _folder_for_model_key(model_key) if model_key else None
        if folder is None:
            models = _list_qwen_vl_models()
            if models:
                folder = _folder_for_model_key(models[0][0])
        if folder is None:
            # No model available — produce local expand quickly
            style = self.style.text().strip() or DEFAULT_STYLE
            # apply preset locally as well
            preset_name = self.combo_preset.currentText().strip() or "Default"
            p = PRESET_DEFS.get(preset_name, PRESET_DEFS["Default"])
            if p.get("style"):
                style = (f"{style}, {p['style']}" if style else p["style"])
            import re as _re
            _neg_raw = self.neg.text().strip()
            neg = _neg_raw if _re.search(r"[A-Za-z0-9]", _neg_raw) else ""
            if p.get("negatives"):
                neg = (f"{neg}, {p['negatives']}" if neg else p["negatives"])
            out = f"{seed.strip().capitalize()}" + (f" in {style};" if style else ";") + " rich lighting, camera details, textures, mood, palette."
            self.out.setPlainText(str(out).strip())
            self.out_neg.setPlainText(self._last_combined_neg)
            return

        self._thread = QThread(self)
        self._worker = _PromptGenWorker(folder, sys, usr, float(self.temp.value()), int(self.max_new.value()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.failed.connect(self._cleanup_worker)

        self._set_busy(True)
        self._thread.start()

    def _on_worker_finished(self, text: str):
        self.out.setPlainText(str(text).strip())
        try:
            self.out_neg.setPlainText(self._last_combined_neg)
        except Exception:
            pass
        self._schedule_save()

    def _on_worker_failed(self, err: str):
        try:
            QMessageBox.critical(self, "Prompt generator", f"{err}\n\nFalling back to local expand.")
        except Exception:
            pass
        seed = (self.seed.toPlainText() or "").strip()
        style = self.style.text().strip() or DEFAULT_STYLE
        # apply preset locally
        preset_name = self.combo_preset.currentText().strip() or "Default"
        p = PRESET_DEFS.get(preset_name, PRESET_DEFS["Default"])
        if p.get("style"):
            style = (f"{style}, {p['style']}" if style else p["style"])
        neg = (self.neg.text().strip() or DEFAULT_NEG)
        if p.get("negatives"):
            neg = (f"{neg}, {p['negatives']}" if neg else p["negatives"])
        out = f"{seed.strip().capitalize()}" + (f" in {style};" if style else ";") + " rich lighting, camera details, textures, mood, palette."
        self.out.setPlainText(str(out).strip())
        try:
            # Re-compute combined negatives for display
            _ = self._compose_prompts(seed)
            self.out_neg.setPlainText(self._last_combined_neg)
        except Exception:
            self.out_neg.setPlainText(self._last_combined_neg)
        self._schedule_save()

    def _cleanup_worker(self):
        self._set_busy(False)
        try:
            if self._worker is not None:
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
                self._thread.deleteLater()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.btn_gen.setText("Cancelling…")

    def _copy_to_clipboard(self):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(self.out.toPlainText())
            QMessageBox.information(self, "Copied", "Result copied to clipboard.")
        except Exception:
            pass

    
    def _clear_outputs(self):
        try:
            self.out.clear()
        except Exception:
            pass
        try:
            self.out_neg.clear()
        except Exception:
            pass

    def _save_txt(self):
        text = self.out.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Save .txt", "No result to save yet.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save prompt as .txt", str(ROOT / "output_prompt.txt"), "Text files (*.txt)")
        if not fn: return
        try:
            Path(fn).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Save .txt", f"Failed: {e}")

    def _save_json(self):
        text = self.out.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Save .json", "No result to save yet.")
            return
        meta = {
            "seed": (self.seed.toPlainText() or "").strip(),
            "prompt": text,
            "style": self.style.text().strip(),
            "length": self.combo_len.currentText(),
            "target": self.combo_target.currentText(),
            "preset": self.combo_preset.currentText(),
            "model_key": self.combo_model.currentData() or "",
            "negatives": self.neg.text().strip(),
            "applied_negatives": self._last_combined_neg,
            "temperature": float(self.temp.value()),
            "max_new_tokens": int(self.max_new.value()),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "category": self.combo_category.currentText(),
            "favorites": self.state.get("favorites",[]),
        }
        fn, _ = QFileDialog.getSaveFileName(self, "Save prompt as .json", str(ROOT / "prompt.json"), "JSON files (*.json)")
        if not fn: return
        try:
            Path(fn).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Save .json", f"Failed: {e}")

    def _teardown(self, *args, **kwargs):
        try:
            self._saving_timer.stop()
        except Exception:
            pass
        try:
            if self._worker is not None:
                self._worker.cancel()
        except Exception:
            pass
        try:
            if self._thread is not None:
                try:
                    self._thread.quit()
                except Exception:
                    pass
                self._thread.wait(2000)
                try:
                    self._thread.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        self._thread = None
        self._worker = None


def install_prompt_tool(owner, section_widget):
    try:
        widget = PromptToolPane(getattr(owner, "main", None), section_widget)
    except Exception:
        widget = PromptToolPane(None, section_widget)
    wrap = QWidget(); lay = QVBoxLayout(wrap); lay.setContentsMargins(0,0,0,0); lay.addWidget(widget)
    try:
        section_widget.setContentLayout(lay)
    except Exception:
        try:
            section_widget.content.setLayout(lay)
        except Exception:
            pass
    return widget