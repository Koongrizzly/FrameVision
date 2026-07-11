#!/usr/bin/env python3
"""Tower-Vision: a complete single-file neon tower-defense game for FrameVision.

Run with: python Tower-vision.py
Optional developer smoke test: python Tower-vision.py --selftest
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


# -----------------------------------------------------------------------------
# Core constants and helpers
# -----------------------------------------------------------------------------
DESIGN_W, DESIGN_H = 1280, 720
BOARD_RECT = pygame.Rect(20, 76, 930, 620)
SIDE_RECT = pygame.Rect(968, 76, 292, 620)
FPS = 60
FIXED_STEP = 1.0 / 60.0
MAX_FRAME_DT = 0.15
SPATIAL_CELL = 96
MAX_PARTICLES = 700
MAX_FLOATING_TEXTS = 140
MAX_BEAMS = 90
SAVE_VERSION = 3
TARGET_MODES = ("First", "Last", "Strongest", "Weakest", "Fastest", "Closest")
DIFFICULTIES = ("Easy", "Normal", "Hard")
MAP_IDS = ("neon_circuit", "split_junction", "reactor_spiral")
TOWER_KEYS = ("pulse", "rail", "arc", "cryo", "missile", "drone")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

C_BG = (8, 12, 20)
C_PANEL = (11, 18, 29)
C_PANEL_2 = (18, 27, 43)
C_TEXT = (224, 239, 255)
C_MUTED = (135, 160, 183)
C_CYAN = (52, 225, 255)
C_BLUE = (76, 135, 255)
C_GREEN = (86, 245, 169)
C_YELLOW = (255, 216, 88)
C_ORANGE = (255, 142, 73)
C_RED = (255, 80, 92)
C_PURPLE = (159, 112, 255)
C_WHITE = (250, 253, 255)
C_BLACK = (0, 0, 0)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def v2(value) -> pygame.Vector2:
    return value if isinstance(value, pygame.Vector2) else pygame.Vector2(value)


def distance_point_segment(point: pygame.Vector2, a: pygame.Vector2, b: pygame.Vector2) -> float:
    ab = b - a
    denom = ab.length_squared()
    if denom <= 1e-9:
        return point.distance_to(a)
    t = clamp((point - a).dot(ab) / denom, 0.0, 1.0)
    return point.distance_to(a + ab * t)


def draw_glow_circle(surface: pygame.Surface, color, pos, radius: int, width: int = 0, alpha: int = 180):
    if radius <= 0:
        return
    glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    center = pygame.Vector2(radius * 2, radius * 2)
    for scale, a in ((1.8, alpha // 8), (1.45, alpha // 5), (1.15, alpha // 3)):
        pygame.draw.circle(glow, (*color, a), center, int(radius * scale), max(1, width))
    pygame.draw.circle(glow, (*color, alpha), center, radius, width)
    surface.blit(glow, (pos[0] - radius * 2, pos[1] - radius * 2), special_flags=pygame.BLEND_PREMULTIPLIED)


def text(surface, font, content: str, pos, color=C_TEXT, anchor="topleft", shadow=True):
    img = font.render(str(content), True, color)
    rect = img.get_rect()
    setattr(rect, anchor, pos)
    if shadow:
        shadow_img = font.render(str(content), True, (0, 0, 0))
        surface.blit(shadow_img, rect.move(2, 2))
    surface.blit(img, rect)
    return rect


def wrap_text(font: pygame.font.Font, content: str, max_width: int) -> list[str]:
    words = content.split()
    lines, line = [], ""
    for word in words:
        test = word if not line else line + " " + word
        if font.size(test)[0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines



def tower_level_cap(campaign_level: int) -> int:
    """Internal upgrade index cap: level 1 has indices 0..3, then two more tiers per cycle."""
    return 3 + max(0, int(campaign_level) - 1) * 2


def campaign_hp_scale(campaign_level: int) -> float:
    """Each cleared 30-wave cycle is meaningfully tougher without relying only on enemy count."""
    return 2.65 ** min(249, max(0, int(campaign_level) - 1))


def campaign_reward_scale(campaign_level: int) -> float:
    return 1.0 + max(0, int(campaign_level) - 1) * 0.28


def total_campaign_wave(campaign_level: int, wave: int) -> int:
    return max(0, (max(1, int(campaign_level)) - 1) * 30 + max(0, int(wave)))


def safe_int(value, default, low=None, high=None):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if low is not None and result < low:
        return default
    if high is not None and result > high:
        return default
    return result


def safe_float(value, default, low=None, high=None):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    if low is not None and result < low:
        return default
    if high is not None and result > high:
        return default
    return result


# -----------------------------------------------------------------------------
# Save and sound systems
# -----------------------------------------------------------------------------
class SaveManager:
    def __init__(self, script_path: Path):
        self.root = script_path.resolve().parent.parent
        self.save_path = self.root / "presets" / "setsave" / "tower_defense_save.json"
        self.data = self.default_data()
        self.load()

    @staticmethod
    def default_data():
        return {
            "version": SAVE_VERSION,
            "high_score": 0,
            "best_wave": 0,
            "best_level": 1,
            "levels_cleared": 0,
            "total_enemies_defeated": 0,
            "bosses_defeated": 0,
            "games_won": 0,
            "games_lost": 0,
            "towers_built": 0,
            "towers_upgraded": 0,
            "credits_earned": 0,
            "total_play_time": 0.0,
            "selected_difficulty": "Normal",
            "selected_map": "neon_circuit",
            "audio_volume": 0.55,
            "muted": False,
            "fullscreen": False,
            "auto_wave": False,
            "campaign": None,
        }

    def load(self):
        defaults = self.default_data()
        try:
            raw = json.loads(self.save_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("save root is not an object")
        except (OSError, ValueError, json.JSONDecodeError):
            raw = {}
        self.data = defaults
        self.data["high_score"] = safe_int(raw.get("high_score"), 0, 0)
        self.data["best_wave"] = safe_int(raw.get("best_wave"), 0, 0)
        self.data["best_level"] = safe_int(raw.get("best_level"), 1, 1)
        self.data["levels_cleared"] = safe_int(raw.get("levels_cleared"), 0, 0)
        for key in ("total_enemies_defeated", "bosses_defeated", "games_won", "games_lost",
                    "towers_built", "towers_upgraded", "credits_earned"):
            self.data[key] = safe_int(raw.get(key), 0, 0)
        self.data["total_play_time"] = safe_float(raw.get("total_play_time"), 0.0, 0.0)
        self.data["selected_difficulty"] = raw.get("selected_difficulty") if raw.get("selected_difficulty") in DIFFICULTIES else "Normal"
        self.data["selected_map"] = raw.get("selected_map") if raw.get("selected_map") in MAP_IDS else "neon_circuit"
        self.data["audio_volume"] = safe_float(raw.get("audio_volume"), 0.55, 0.0, 1.0)
        self.data["muted"] = bool(raw.get("muted", False))
        self.data["fullscreen"] = bool(raw.get("fullscreen", False))
        self.data["auto_wave"] = bool(raw.get("auto_wave", False))
        self.data["campaign"] = self.validate_campaign(raw.get("campaign"))

    def validate_campaign(self, camp):
        if not isinstance(camp, dict):
            return None
        map_id = camp.get("map_id")
        difficulty = camp.get("difficulty")
        if map_id not in MAP_IDS or difficulty not in DIFFICULTIES:
            return None
        wave = safe_int(camp.get("wave"), -1, 0, 29)
        campaign_level = safe_int(camp.get("level"), 1, 1, 250)
        health = safe_float(camp.get("reactor_health"), -1, 1, 10000)
        max_health = safe_float(camp.get("reactor_max"), -1, 1, 10000)
        credits = safe_int(camp.get("credits"), -1, 0, 1_000_000_000_000_000)
        score = safe_int(camp.get("score"), -1, 0, 9_000_000_000_000_000_000)
        if min(wave, health, max_health, credits, score) < 0 or health > max_health:
            return None
        towers_raw = camp.get("towers", [])
        if not isinstance(towers_raw, list) or len(towers_raw) > 80:
            return None
        towers = []
        for item in towers_raw:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            level = safe_int(item.get("level"), -1, 0, tower_level_cap(campaign_level))
            x = safe_float(item.get("x"), -1, BOARD_RECT.left, BOARD_RECT.right)
            y = safe_float(item.get("y"), -1, BOARD_RECT.top, BOARD_RECT.bottom)
            target_mode = item.get("targeting") if item.get("targeting") in TARGET_MODES else "First"
            pad_index = safe_int(item.get("pad_index"), -1, 0, 999)
            if kind in TOWER_KEYS and level >= 0 and x >= 0 and y >= 0 and pad_index >= 0:
                towers.append({"type": kind, "level": level, "x": x, "y": y,
                               "targeting": target_mode, "pad_index": pad_index})
        cooldowns = camp.get("ability_cooldowns", {})
        if not isinstance(cooldowns, dict):
            cooldowns = {}
        return {
            "map_id": map_id,
            "difficulty": difficulty,
            "level": campaign_level,
            "wave": wave,
            "reactor_health": health,
            "reactor_max": max_health,
            "credits": credits,
            "score": score,
            "towers": towers,
            "ability_cooldowns": {
                "emp": safe_float(cooldowns.get("emp"), 0.0, 0.0, 9999.0),
                "orbital": safe_float(cooldowns.get("orbital"), 0.0, 0.0, 9999.0),
                "repair": safe_float(cooldowns.get("repair"), 0.0, 0.0, 9999.0),
            },
        }

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, sort_keys=True)
        fd, temp_name = tempfile.mkstemp(prefix="tower_save_", suffix=".tmp", dir=str(self.save_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.save_path)
        except OSError:
            try:
                os.unlink(temp_name)
            except OSError:
                pass

    def reset(self):
        self.data = self.default_data()
        self.save()


class SoundManager:
    def __init__(self, volume=0.55, muted=False):
        self.volume = clamp(volume, 0.0, 1.0)
        self.muted = muted
        self.available = False
        self.sounds = {}
        self.last_played = {}
        self.sound_gaps = {"pulse": 0.025, "drone": 0.045, "enemy_down": 0.035,
                           "cryo": 0.035, "missile": 0.045, "lightning": 0.05,
                           "explosion": 0.055}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.available = True
            self._build_sounds()
            self.apply_volume()
        except pygame.error:
            self.available = False

    @staticmethod
    def _tone(freq=440, duration=0.08, volume=0.35, sweep=0.0, noise=0.0):
        rate = 22050
        count = max(1, int(rate * duration))
        samples = array.array("h")
        rng = random.Random(int(freq * 17 + duration * 1000))
        for i in range(count):
            t = i / rate
            f = freq + sweep * (i / count)
            envelope = min(1.0, i / max(1, count * 0.04)) * (1.0 - i / count) ** 1.6
            wave = math.sin(math.tau * f * t)
            wave += noise * rng.uniform(-1.0, 1.0)
            samples.append(int(clamp(wave * envelope * volume, -1, 1) * 32767))
        return pygame.mixer.Sound(buffer=samples.tobytes())

    def _build_sounds(self):
        recipes = {
            "menu_move": (520, .035, .18, 80, 0), "menu_confirm": (760, .07, .24, 220, 0),
            "place": (350, .09, .28, 450, 0), "upgrade": (540, .16, .26, 620, 0),
            "sell": (520, .12, .22, -330, 0), "pulse": (680, .04, .20, -160, 0),
            "rail": (155, .13, .38, 900, .12), "lightning": (1020, .08, .24, -700, .2),
            "missile": (230, .11, .26, 120, .08), "explosion": (85, .18, .42, -30, .65),
            "enemy_down": (430, .055, .16, -250, 0), "escape": (160, .17, .30, -60, .12),
            "reactor": (110, .26, .43, -40, .28), "ability": (310, .24, .33, 860, .05),
            "wave_start": (480, .22, .30, 520, 0), "wave_complete": (620, .28, .28, 680, 0),
            "boss": (92, .45, .44, -15, .24), "victory": (620, .7, .25, 740, 0),
            "defeat": (220, .75, .28, -160, .12), "cryo": (980, .06, .17, -280, 0),
            "drone": (800, .035, .13, -80, 0), "hit": (300, .025, .08, 0, .12),
        }
        for name, args in recipes.items():
            self.sounds[name] = self._tone(*args)

    def apply_volume(self):
        if not self.available:
            return
        level = 0.0 if self.muted else self.volume
        for sound in self.sounds.values():
            sound.set_volume(level)

    def play(self, name):
        if not (self.available and not self.muted and name in self.sounds):
            return
        now = time.perf_counter()
        gap = self.sound_gaps.get(name, 0.0)
        if gap and now - self.last_played.get(name, -999.0) < gap:
            return
        self.last_played[name] = now
        self.sounds[name].play()

    def set_volume(self, value):
        self.volume = clamp(value, 0.0, 1.0)
        self.apply_volume()

    def toggle_mute(self):
        self.muted = not self.muted
        self.apply_volume()


# -----------------------------------------------------------------------------
# Map definitions
# -----------------------------------------------------------------------------
@dataclass
class MapDefinition:
    id: str
    name: str
    description: str
    routes: list[list[tuple[float, float]]]
    pads: list[tuple[float, float]]
    blocked: list[pygame.Rect]
    accent: tuple[int, int, int]

    def route_vectors(self, route_id=0):
        return [pygame.Vector2(p) for p in self.routes[route_id]]


def create_maps() -> dict[str, MapDefinition]:
    return {
        "neon_circuit": MapDefinition(
            "neon_circuit", "Neon Circuit",
            "A readable winding route with balanced build positions. Best for learning.",
            [[(24, 160), (170, 160), (170, 300), (350, 300), (350, 145), (545, 145),
              (545, 420), (735, 420), (735, 245), (890, 245), (930, 245)]],
            [(95, 245), (105, 390), (245, 220), (265, 390), (430, 245), (450, 510),
             (635, 245), (640, 510), (810, 340), (830, 160), (870, 520), (300, 560)],
            [pygame.Rect(45, 490, 135, 90), pygame.Rect(755, 500, 115, 70)], C_CYAN),
        "split_junction": MapDefinition(
            "split_junction", "Split Junction",
            "Two stable branches split and reconnect. Central pads can cover both lanes.",
            [[(24, 325), (185, 325), (260, 205), (470, 205), (575, 325), (740, 325), (840, 245), (930, 245)],
             [(24, 325), (185, 325), (260, 445), (470, 445), (575, 325), (740, 325), (840, 245), (930, 245)]],
            [(105, 215), (110, 445), (305, 325), (405, 325), (515, 130), (515, 520),
             (620, 235), (620, 415), (760, 190), (775, 470), (860, 345), (890, 145)],
            [pygame.Rect(300, 270, 180, 110), pygame.Rect(50, 105, 110, 70)], C_GREEN),
        "reactor_spiral": MapDefinition(
            "reactor_spiral", "Reactor Spiral",
            "A tightening spiral toward a central reactor. Powerful inner pads, scarce outer space.",
            [[(24, 115), (875, 115), (875, 585), (120, 585), (120, 205), (785, 205),
              (785, 495), (220, 495), (220, 295), (690, 295), (690, 405), (465, 405)]],
            [(70, 350), (190, 160), (340, 160), (560, 160), (820, 350), (700, 550),
             (490, 550), (290, 545), (170, 400), (315, 250), (595, 250), (615, 455),
             (365, 450), (455, 350)],
            [pygame.Rect(35, 615, 110, 45), pygame.Rect(800, 35, 95, 50)], C_PURPLE),
    }


# -----------------------------------------------------------------------------
# UI primitives
# -----------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label, callback: Optional[Callable] = None, tooltip="", hotkey=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.tooltip = tooltip
        self.hotkey = hotkey
        self.enabled = True
        self.selected = False

    def draw(self, surf, fonts, mouse):
        hover = self.enabled and self.rect.collidepoint(mouse)
        base = C_PANEL_2 if self.enabled else (28, 34, 42)
        border = C_CYAN if self.selected else C_GREEN if hover else (65, 92, 118)
        pygame.draw.rect(surf, base, self.rect, border_radius=8)
        pygame.draw.rect(surf, border, self.rect, 2, border_radius=8)
        if hover:
            glow = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow, (*border, 35), glow.get_rect(), border_radius=8)
            surf.blit(glow, self.rect)
        color = C_TEXT if self.enabled else (92, 104, 116)
        text(surf, fonts["small"], self.label, self.rect.center, color, "center")
        if self.hotkey:
            text(surf, fonts["tiny"], self.hotkey, (self.rect.right - 8, self.rect.top + 5), C_MUTED, "topright", False)

    def click(self, pos):
        if self.enabled and self.rect.collidepoint(pos):
            if self.callback:
                self.callback()
            return True
        return False


class TooltipManager:
    def __init__(self):
        self.message = ""
        self.pos = (0, 0)

    def update(self, mouse, items: Iterable):
        self.message = ""
        self.pos = mouse
        for item in items:
            rect = getattr(item, "rect", None)
            tip = getattr(item, "tooltip", "")
            if rect and tip and rect.collidepoint(mouse):
                self.message = tip
                break

    def draw(self, surf, fonts):
        if not self.message:
            return
        lines = wrap_text(fonts["tiny"], self.message, 300)
        w = min(320, max(fonts["tiny"].size(line)[0] for line in lines) + 24)
        h = len(lines) * 18 + 18
        x = min(self.pos[0] + 16, DESIGN_W - w - 8)
        y = min(self.pos[1] + 18, DESIGN_H - h - 8)
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (5, 10, 18), rect, border_radius=7)
        pygame.draw.rect(surf, C_CYAN, rect, 1, border_radius=7)
        for i, line in enumerate(lines):
            text(surf, fonts["tiny"], line, (x + 12, y + 9 + i * 18), C_TEXT, shadow=False)


# -----------------------------------------------------------------------------
# Effects
# -----------------------------------------------------------------------------
class Particle:
    def __init__(self, pos, velocity, color, life, radius=3, drag=0.92, glow=True):
        self.pos = v2(pos).copy()
        self.vel = v2(velocity).copy()
        self.color = color
        self.life = self.max_life = life
        self.radius = radius
        self.drag = drag
        self.glow = glow
        self.dead = False

    def update(self, dt):
        self.life -= dt
        if self.life <= 0:
            self.dead = True
            return
        self.pos += self.vel * dt
        self.vel *= self.drag ** (dt * 60)

    def draw(self, surf):
        alpha = int(255 * clamp(self.life / self.max_life, 0, 1))
        r = max(1, int(self.radius * (0.4 + self.life / self.max_life)))
        if self.glow:
            draw_glow_circle(surf, self.color, self.pos, r, alpha=alpha)
        else:
            pygame.draw.circle(surf, self.color, self.pos, r)


class FloatingText:
    def __init__(self, pos, content, color=C_TEXT, life=0.8, size="tiny"):
        self.pos = v2(pos).copy()
        self.content = content
        self.color = color
        self.life = self.max_life = life
        self.size = size
        self.dead = False

    def update(self, dt):
        self.life -= dt
        self.pos.y -= 34 * dt
        self.dead = self.life <= 0

    def draw(self, surf, fonts):
        alpha = int(255 * clamp(self.life / self.max_life, 0, 1))
        img = fonts[self.size].render(self.content, True, self.color)
        img.set_alpha(alpha)
        surf.blit(img, img.get_rect(center=self.pos))


class BeamEffect:
    def __init__(self, points, color, life=0.13, width=3):
        self.points = [v2(p).copy() for p in points]
        self.color = color
        self.life = self.max_life = life
        self.width = width
        self.dead = False

    def update(self, dt):
        self.life -= dt
        self.dead = self.life <= 0

    def draw(self, surf):
        if len(self.points) < 2:
            return
        alpha = int(255 * clamp(self.life / self.max_life, 0, 1))
        layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        pygame.draw.lines(layer, (*self.color, alpha // 3), False, self.points, self.width * 5)
        pygame.draw.lines(layer, (*self.color, alpha), False, self.points, self.width)
        surf.blit(layer, (0, 0))


# -----------------------------------------------------------------------------
# Enemy system
# -----------------------------------------------------------------------------
ENEMY_SPECS = {
    "runner": {"name": "Runner", "hp": 68, "speed": 92, "armor": 0, "reward": 8, "damage": 1, "score": 14, "radius": 9, "color": C_YELLOW},
    "grunt": {"name": "Grunt", "hp": 145, "speed": 58, "armor": 3, "reward": 12, "damage": 1, "score": 22, "radius": 12, "color": C_CYAN},
    "tank": {"name": "Tank", "hp": 560, "speed": 30, "armor": 12, "reward": 28, "damage": 3, "score": 56, "radius": 17, "color": C_ORANGE},
    "swarm": {"name": "Swarm Drone", "hp": 42, "speed": 72, "armor": 0, "reward": 4, "damage": 1, "score": 9, "radius": 7, "color": C_GREEN},
    "shield": {"name": "Shield Unit", "hp": 190, "shield": 180, "speed": 45, "armor": 5, "reward": 22, "damage": 2, "score": 42, "radius": 13, "color": C_BLUE},
    "splitter": {"name": "Splitter", "hp": 300, "speed": 48, "armor": 5, "reward": 19, "damage": 2, "score": 38, "radius": 14, "color": C_PURPLE},
    "regenerator": {"name": "Regenerator", "hp": 440, "speed": 37, "armor": 7, "reward": 27, "damage": 2, "score": 52, "radius": 14, "color": (84, 255, 132)},
    "phase": {"name": "Phase Unit", "hp": 275, "speed": 62, "armor": 3, "reward": 24, "damage": 2, "score": 48, "radius": 12, "color": (140, 190, 255)},
}

BOSS_SPECS = {
    "siege": {"name": "Siege Walker", "hp": 7200, "speed": 24, "armor": 18, "reward": 440, "damage": 12, "score": 1050, "radius": 27, "color": C_ORANGE},
    "serpent": {"name": "Null Serpent", "hp": 14500, "speed": 32, "armor": 14, "reward": 820, "damage": 18, "score": 2100, "radius": 25, "color": C_PURPLE},
    "devourer": {"name": "Core Devourer", "hp": 27000, "shield": 7600, "speed": 27, "armor": 16, "reward": 1850, "damage": 35, "score": 4900, "radius": 31, "color": C_RED},
}


class Enemy:
    next_id = 1

    def __init__(self, kind, route, route_id, difficulty, wave, hp_scale=1.0, reward_scale=1.0,
                 start_progress=0.0, required=True, child=False, campaign_level=1):
        spec = ENEMY_SPECS[kind]
        diff = DIFFICULTY_DATA[difficulty]
        self.id = Enemy.next_id
        Enemy.next_id += 1
        self.kind = kind
        self.name = spec["name"]
        self.route = [v2(p).copy() for p in route]
        self.route_id = route_id
        self.segment_lengths = [self.route[i].distance_to(self.route[i + 1]) for i in range(len(self.route) - 1)]
        self.total_length = max(1.0, sum(self.segment_lengths))
        self.segment = 0
        self.distance_on_route = 0.0
        self.pos = self.route[0].copy()
        self.wave = wave
        self.campaign_level = max(1, int(campaign_level))
        # Health rises aggressively enough that upgraded batteries do not erase later waves at the portal.
        growth = 1.0 + (wave - 1) * 0.11 + max(0, wave - 15) * 0.040
        cycle_hp = campaign_hp_scale(self.campaign_level)
        self.max_hp = spec["hp"] * growth * diff["enemy_hp"] * hp_scale * cycle_hp
        self.hp = self.max_hp
        self.max_shield = spec.get("shield", 0) * growth * diff["enemy_hp"] * cycle_hp
        self.shield = self.max_shield
        self.shield_delay = 0.0
        cycle_speed = 1.0 + min(0.42, (self.campaign_level - 1) * 0.025)
        self.base_speed = spec["speed"] * diff["enemy_speed"] * cycle_speed
        self.armor = spec["armor"] + max(0, wave - 12) * 0.12 + (self.campaign_level - 1) * 1.35
        cycle_reward = campaign_reward_scale(self.campaign_level)
        self.reward = max(1, int(spec["reward"] * reward_scale * diff["reward"] * cycle_reward))
        self.core_damage = spec["damage"] + (self.campaign_level - 1) // 3
        self.score_value = int(spec["score"] * diff["score_mult"] * cycle_reward)
        self.radius = spec["radius"]
        self.color = spec["color"]
        self.required = required
        self.child = child
        self.alive = True
        self.escaped = False
        self.kill_processed = False
        self.escape_processed = False
        self.spawn_protection = 0.08
        self.damage_flash = 0.0
        self.slow_strength = 0.0
        self.slow_timer = 0.0
        self.freeze_timer = 0.0
        self.stun_timer = 0.0
        self.burn_dps = 0.0
        self.burn_timer = 0.0
        self.armor_reduction = 0.0
        self.armor_reduction_timer = 0.0
        self.ability_scale = {"Easy": 1.12, "Normal": 1.0, "Hard": 0.88}[difficulty]
        self.phase_timer = random.uniform(2.2, 4.2) * self.ability_scale if kind == "phase" else 0.0
        self.phase_duration = 0.0
        self.phased = False
        self.anim = random.random() * math.tau
        self.last_damage_source = None
        self.regen_rate = self.max_hp * 0.018 if kind == "regenerator" else 0.0
        if start_progress > 0:
            self.set_progress(start_progress)

    @property
    def is_boss(self):
        return False

    @property
    def targetable(self):
        return self.alive and not self.escaped and not self.phased and self.spawn_protection <= 0

    @property
    def progress(self):
        return clamp(self.distance_on_route / self.total_length, 0.0, 1.0)

    @property
    def durability(self):
        return max(0.0, self.hp) + max(0.0, self.shield)

    @property
    def effective_speed(self):
        if self.freeze_timer > 0 or self.stun_timer > 0:
            return 0.0
        return max(7.0, self.base_speed * (1.0 - clamp(self.slow_strength, 0.0, 0.65)))

    def set_progress(self, progress):
        target = clamp(progress, 0.0, 0.98) * self.total_length
        self.segment = 0
        self.distance_on_route = 0.0
        remaining = target
        for i, length in enumerate(self.segment_lengths):
            if remaining <= length:
                self.segment = i
                t = remaining / max(length, 1e-6)
                self.pos = self.route[i].lerp(self.route[i + 1], t)
                self.distance_on_route = target
                return
            remaining -= length
        self.segment = len(self.route) - 2
        self.pos = self.route[-2].copy()
        self.distance_on_route = target

    def apply_status(self, kind, amount, duration):
        if not self.alive or self.escaped:
            return
        if kind == "slow":
            self.slow_strength = max(self.slow_strength, clamp(amount, 0, 0.65))
            self.slow_timer = max(self.slow_timer, duration)
        elif kind == "freeze":
            self.freeze_timer = max(self.freeze_timer, min(duration, 1.6 if not self.is_boss else 0.65))
        elif kind == "stun":
            self.stun_timer = max(self.stun_timer, min(duration, 2.5 if not self.is_boss else 0.8))
        elif kind == "burn":
            self.burn_dps = max(self.burn_dps, amount)
            self.burn_timer = max(self.burn_timer, duration)
        elif kind == "armor_break":
            self.armor_reduction = max(self.armor_reduction, min(amount, self.armor + 8))
            self.armor_reduction_timer = max(self.armor_reduction_timer, duration)

    def take_damage(self, game, raw_damage, armor_pen=0.0, source=None, ignore_phase=False, color=C_WHITE, critical=False):
        if not self.alive or self.escaped or (self.phased and not ignore_phase):
            return 0.0
        raw_damage = max(0.0, raw_damage)
        if raw_damage <= 0:
            return 0.0
        effective_armor = max(-5.0, self.armor - self.armor_reduction - armor_pen)
        multiplier = 100.0 / (100.0 + max(-45.0, effective_armor * 7.0))
        damage = max(0.5, raw_damage * multiplier)
        dealt = 0.0
        if self.shield > 0:
            shield_hit = min(self.shield, damage)
            self.shield -= shield_hit
            damage -= shield_hit
            dealt += shield_hit
            self.shield_delay = 3.0
        if damage > 0:
            hp_hit = min(self.hp, damage)
            self.hp -= hp_hit
            dealt += hp_hit
        self.damage_flash = 0.11
        self.last_damage_source = source
        game.add_floating(FloatingText(self.pos + pygame.Vector2(0, -self.radius - 8),
                                                ("CRIT " if critical else "") + str(int(dealt)),
                                                C_YELLOW if critical else color, .55))
        if self.hp <= 0 and self.alive:
            self.die(game, source)
        return dealt

    def die(self, game, source=None):
        if not self.alive or self.kill_processed:
            return
        self.alive = False
        self.kill_processed = True
        game.credits += self.reward
        game.score += self.score_value
        game.save.data["total_enemies_defeated"] += 1
        game.save.data["credits_earned"] += self.reward
        game.add_floating(FloatingText(self.pos, f"+{self.reward}", C_GREEN, .85, "small"))
        game.sound.play("enemy_down")
        game.add_burst(self.pos, self.color, 8, 120)
        if self.kind == "splitter":
            # Children are required enemies but grant their own small rewards; the parent reward is not duplicated.
            for offset in (-0.012, 0.012):
                child = Enemy("swarm", self.route, self.route_id, game.difficulty, self.wave,
                              hp_scale=1.15, reward_scale=.65, start_progress=self.progress + offset,
                              required=self.required, child=True, campaign_level=self.campaign_level)
                game.enemies_to_add.append(child)
        if self.is_boss:
            game.save.data["bosses_defeated"] += 1

    def escape(self, game):
        if not self.alive or self.escape_processed:
            return
        self.alive = False
        self.escaped = True
        self.escape_processed = True
        game.damage_reactor(self.core_damage)
        game.sound.play("escape")

    def update_status(self, game, dt):
        self.spawn_protection = max(0.0, self.spawn_protection - dt)
        self.damage_flash = max(0.0, self.damage_flash - dt)
        if self.slow_timer > 0:
            self.slow_timer -= dt
            if self.slow_timer <= 0:
                self.slow_strength = 0.0
        if self.freeze_timer > 0:
            self.freeze_timer -= dt
        if self.stun_timer > 0:
            self.stun_timer -= dt
        if self.burn_timer > 0:
            self.burn_timer -= dt
            self.take_damage(game, self.burn_dps * dt, armor_pen=2, source="burn", ignore_phase=True, color=C_ORANGE)
            if self.burn_timer <= 0:
                self.burn_dps = 0.0
        if self.armor_reduction_timer > 0:
            self.armor_reduction_timer -= dt
            if self.armor_reduction_timer <= 0:
                self.armor_reduction = 0.0
        if self.kind == "shield" and self.max_shield > 0:
            self.shield_delay = max(0.0, self.shield_delay - dt)
            if self.shield_delay <= 0 and self.shield < self.max_shield:
                self.shield = min(self.max_shield, self.shield + self.max_shield * (.10 / self.ability_scale) * dt)
        if self.kind == "regenerator" and self.hp > 0:
            self.hp = min(self.max_hp, self.hp + self.regen_rate * dt)
        if self.kind == "phase":
            if self.phased:
                self.phase_duration -= dt
                if self.phase_duration <= 0:
                    self.phased = False
                    self.phase_timer = random.uniform(2.6, 4.4) * self.ability_scale
            else:
                self.phase_timer -= dt
                if self.phase_timer <= 0:
                    self.phased = True
                    self.phase_duration = 0.85

    def move(self, dt):
        remaining = self.effective_speed * dt
        while remaining > 0 and self.segment < len(self.route) - 1:
            target = self.route[self.segment + 1]
            dist = self.pos.distance_to(target)
            if dist <= 1e-8:
                self.segment += 1
                continue
            step = min(remaining, dist)
            self.pos += (target - self.pos) * (step / dist)
            self.distance_on_route = min(self.total_length, self.distance_on_route + step)
            remaining -= step
            if step >= dist - 1e-7:
                self.pos = target.copy()
                self.segment += 1
        return self.segment >= len(self.route) - 1

    def update(self, game, dt):
        if not self.alive:
            return
        self.anim += dt * (4 + self.base_speed / 40)
        self.update_status(game, dt)
        if not self.alive:
            return
        if self.move(dt):
            self.escape(game)

    def draw(self, surf):
        if not self.alive:
            return
        pos = self.pos
        base = C_WHITE if self.damage_flash > 0 else self.color
        color = tuple(int(c * .42) for c in base) if self.phased else base
        if self.kind in ("runner", "phase"):
            pts = [pos + pygame.Vector2(math.cos(self.anim + i * math.tau / 3),
                                        math.sin(self.anim + i * math.tau / 3)) * self.radius for i in range(3)]
            pygame.draw.polygon(surf, color, pts)
        elif self.kind == "tank":
            rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 1.5)
            rect.center = pos
            pygame.draw.rect(surf, color, rect, border_radius=4)
            pygame.draw.circle(surf, (30, 35, 44), pos, self.radius // 2)
        elif self.kind == "swarm":
            pygame.draw.circle(surf, color, pos, self.radius)
            pygame.draw.line(surf, C_WHITE, pos - pygame.Vector2(8, 0), pos + pygame.Vector2(8, 0), 2)
        elif self.kind == "shield":
            pygame.draw.circle(surf, color, pos, self.radius)
            if self.shield > 0:
                pygame.draw.circle(surf, C_BLUE, pos, self.radius + 5, 2)
        elif self.kind == "splitter":
            pygame.draw.rect(surf, color, pygame.Rect(pos.x-self.radius, pos.y-self.radius,
                                                      self.radius*2, self.radius*2), border_radius=5)
            pygame.draw.line(surf, C_WHITE, pos + (-7, -7), pos + (7, 7), 2)
            pygame.draw.line(surf, C_WHITE, pos + (-7, 7), pos + (7, -7), 2)
        elif self.kind == "regenerator":
            pygame.draw.circle(surf, color, pos, self.radius)
            pygame.draw.line(surf, C_WHITE, pos + (-7, 0), pos + (7, 0), 3)
            pygame.draw.line(surf, C_WHITE, pos + (0, -7), pos + (0, 7), 3)
        else:
            pygame.draw.circle(surf, color, pos, self.radius)
        if self.freeze_timer > 0:
            pygame.draw.circle(surf, C_CYAN, pos, self.radius + 4, 2)
        if self.stun_timer > 0:
            for i in range(3):
                ang = self.anim * 2 + i * math.tau / 3
                pygame.draw.circle(surf, C_YELLOW, pos + pygame.Vector2(math.cos(ang), math.sin(ang)) * (self.radius + 7), 2)
        if self.burn_timer > 0:
            pygame.draw.circle(surf, C_ORANGE, pos + (0, -self.radius), 3)
        self.draw_bars(surf)

    def draw_bars(self, surf):
        width = max(24, self.radius * 2 + 8)
        x = self.pos.x - width / 2
        y = self.pos.y - self.radius - 12
        pygame.draw.rect(surf, (20, 24, 30), (x, y, width, 4))
        pygame.draw.rect(surf, C_GREEN if self.hp / self.max_hp > .35 else C_RED,
                         (x, y, width * clamp(self.hp / self.max_hp, 0, 1), 4))
        if self.max_shield > 0:
            pygame.draw.rect(surf, (20, 24, 30), (x, y - 5, width, 3))
            pygame.draw.rect(surf, C_BLUE, (x, y - 5, width * clamp(self.shield / self.max_shield, 0, 1), 3))


class BossEnemy(Enemy):
    def __init__(self, kind, route, route_id, difficulty, wave, campaign_level=1):
        spec = BOSS_SPECS[kind]
        # Initialize manually to retain Enemy behavior while using boss data.
        self.id = Enemy.next_id
        Enemy.next_id += 1
        self.kind = kind
        self.name = spec["name"]
        self.route = [v2(p).copy() for p in route]
        self.route_id = route_id
        self.segment_lengths = [self.route[i].distance_to(self.route[i + 1]) for i in range(len(self.route) - 1)]
        self.total_length = max(1.0, sum(self.segment_lengths))
        self.segment = 0
        self.distance_on_route = 0.0
        self.pos = self.route[0].copy()
        self.wave = wave
        self.campaign_level = max(1, int(campaign_level))
        diff = DIFFICULTY_DATA[difficulty]
        cycle_hp = campaign_hp_scale(self.campaign_level)
        self.max_hp = spec["hp"] * diff["enemy_hp"] * cycle_hp
        self.hp = self.max_hp
        self.max_shield = spec.get("shield", 0) * diff["enemy_hp"] * cycle_hp
        self.shield = self.max_shield
        self.shield_delay = 0.0
        cycle_speed = 1.0 + min(0.38, (self.campaign_level - 1) * 0.022)
        self.base_speed = spec["speed"] * diff["enemy_speed"] * cycle_speed
        self.armor = spec["armor"] + (self.campaign_level - 1) * 1.6
        cycle_reward = campaign_reward_scale(self.campaign_level)
        self.reward = int(spec["reward"] * diff["reward"] * cycle_reward)
        self.core_damage = spec["damage"] + (self.campaign_level - 1) * 2
        self.score_value = int(spec["score"] * diff["score_mult"] * cycle_reward)
        self.radius = spec["radius"]
        self.color = spec["color"]
        self.required = True
        self.child = False
        self.alive = True
        self.escaped = False
        self.kill_processed = False
        self.escape_processed = False
        self.spawn_protection = .12
        self.damage_flash = 0
        self.slow_strength = 0
        self.slow_timer = 0
        self.freeze_timer = 0
        self.stun_timer = 0
        self.burn_dps = 0
        self.burn_timer = 0
        self.armor_reduction = 0
        self.armor_reduction_timer = 0
        self.phase_timer = 0
        self.phase_duration = 0
        self.phased = False
        self.anim = 0
        self.last_damage_source = None
        self.regen_rate = 0
        self.ability_scale = {"Easy": 1.12, "Normal": 1.0, "Hard": 0.88}[difficulty]
        self.ability_timer = 4.0 * self.ability_scale
        self.telegraph = 0.0
        self.resistance_timer = 0.0
        self.fast_phase = False
        self.phase_stage = 0
        self.reinforcement_triggered = False
        self.enraged = False

    @property
    def is_boss(self):
        return True

    @property
    def effective_speed(self):
        if self.freeze_timer > 0 or self.stun_timer > 0:
            return 0
        mult = 1.0
        if self.kind == "serpent" and self.fast_phase:
            mult = 1.8
        if self.kind == "devourer" and self.enraged:
            mult = 1.45
        return max(9.0, self.base_speed * mult * (1.0 - min(self.slow_strength, .35)))

    def take_damage(self, game, raw_damage, armor_pen=0, source=None, ignore_phase=False, color=C_WHITE, critical=False):
        if self.kind == "siege" and self.resistance_timer > 0:
            raw_damage *= .48
        if self.kind == "devourer" and self.phase_stage == 0 and self.shield <= 0:
            self.phase_stage = 1
            self.reinforcement_triggered = False
            self.telegraph = 1.2
        dealt = super().take_damage(game, raw_damage, armor_pen, source, ignore_phase, color, critical)
        if self.kind == "devourer" and self.alive:
            hp_ratio = self.hp / self.max_hp
            if self.phase_stage == 1 and hp_ratio <= .58:
                self.phase_stage = 2
                self.enraged = True
                self.telegraph = 1.5
                game.add_message("CORE DEVOURER: ENRAGED FINAL PHASE", C_RED)
        return dealt

    def update(self, game, dt):
        if not self.alive:
            return
        self.anim += dt * 2
        self.update_status(game, dt)
        self.resistance_timer = max(0.0, self.resistance_timer - dt)
        self.telegraph = max(0.0, self.telegraph - dt)
        self.ability_timer -= dt
        if self.ability_timer <= 0:
            if self.kind == "siege":
                if self.resistance_timer <= 0:
                    self.resistance_timer = 3.2
                    self.telegraph = .8
                    game.add_message("Siege Walker resistance field active", C_ORANGE)
                self.spawn_support(game, "grunt", 3)
                self.ability_timer = 8.0 * self.ability_scale
            elif self.kind == "serpent":
                self.fast_phase = not self.fast_phase
                self.telegraph = .9
                game.add_message("Null Serpent accelerating" if self.fast_phase else "Null Serpent stabilizing", C_PURPLE)
                if self.fast_phase:
                    for tower in game.towers:
                        if tower.pos.distance_to(self.pos) <= 210:
                            tower.disable_timer = max(tower.disable_timer, 3.0)
                self.ability_timer = 5.2 * self.ability_scale
            elif self.kind == "devourer":
                if self.phase_stage >= 1:
                    self.spawn_support(game, random.choice(("tank", "shield", "phase")), 3 if self.phase_stage == 1 else 5)
                self.ability_timer = (7.0 if not self.enraged else 4.8) * self.ability_scale
        if self.kind == "devourer" and self.phase_stage == 1 and not self.reinforcement_triggered:
            self.reinforcement_triggered = True
            self.spawn_support(game, "shield", 5)
            self.spawn_support(game, "runner", 6)
            game.add_message("Core Devourer reinforcement phase", C_RED)
        if self.move(dt):
            self.escape(game)

    def spawn_support(self, game, kind, count):
        for i in range(count):
            route_id = self.route_id if len(game.map.routes) == 1 else (self.route_id + i) % len(game.map.routes)
            route = game.map.route_vectors(route_id)
            e = Enemy(kind, route, route_id, game.difficulty, self.wave,
                      hp_scale=.72, reward_scale=.65,
                      start_progress=max(0.0, self.progress - .04 * (i + 1)), required=True,
                      campaign_level=self.campaign_level)
            game.enemies_to_add.append(e)

    def draw(self, surf):
        if not self.alive:
            return
        pos = self.pos
        pulse = 1 + .08 * math.sin(self.anim * 3)
        r = int(self.radius * pulse)
        draw_glow_circle(surf, self.color, pos, r, alpha=210)
        pygame.draw.circle(surf, C_PANEL, pos, int(r * .55))
        for i in range(6):
            ang = self.anim + i * math.tau / 6
            p1 = pos + pygame.Vector2(math.cos(ang), math.sin(ang)) * (r * .55)
            p2 = pos + pygame.Vector2(math.cos(ang), math.sin(ang)) * (r * 1.15)
            pygame.draw.line(surf, self.color, p1, p2, 4)
        if self.resistance_timer > 0:
            pygame.draw.circle(surf, C_YELLOW, pos, r + 8, 3)
        if self.telegraph > 0:
            pygame.draw.circle(surf, C_RED, pos, r + 14 + int(5 * math.sin(self.anim * 8)), 2)
        self.draw_bars(surf)


# -----------------------------------------------------------------------------
# Towers, drones and projectiles
# -----------------------------------------------------------------------------
TOWER_SPECS = {
    "pulse": {"name": "Pulse Cannon", "cost": 90, "damage": 18, "range": 142, "rate": 3.1, "speed": 520,
              "color": C_CYAN, "role": "Balanced rapid fire", "desc": "Reliable early defense. Fast shots, medium range.",
              "special": "Critical chance rises with upgrades."},
    "rail": {"name": "Railgun", "cost": 230, "damage": 130, "range": 255, "rate": .55, "speed": 1100,
             "color": C_YELLOW, "role": "Armor-piercing sniper", "desc": "Slow, very high damage. Pierces lined-up targets.",
             "special": "High armor penetration and limited piercing."},
    "arc": {"name": "Arc Tower", "cost": 185, "damage": 46, "range": 160, "rate": .95, "speed": 0,
            "color": C_PURPLE, "role": "Chain damage", "desc": "Instant lightning jumps through clustered enemies.",
            "special": "Each jump deals reduced damage; never repeats a target."},
    "cryo": {"name": "Cryo Tower", "cost": 145, "damage": 10, "range": 150, "rate": 1.4, "speed": 430,
             "color": (100, 220, 255), "role": "Crowd control", "desc": "Slows enemies and may briefly freeze at higher levels.",
             "special": "Strongest slow wins; freeze has a safe duration cap."},
    "missile": {"name": "Missile Battery", "cost": 260, "damage": 82, "range": 210, "rate": .62, "speed": 250,
                "color": C_ORANGE, "role": "Area damage", "desc": "Homing missiles explode across dense groups.",
                "special": "Splash damage and burn; each explosion hits once."},
    "drone": {"name": "Drone Hub", "cost": 215, "damage": 14, "range": 180, "rate": 0, "speed": 500,
              "color": C_GREEN, "role": "Autonomous coverage", "desc": "Deploys orbiting drones that independently acquire targets.",
              "special": "Upgrades add drones and improve their fire rate."},
}

DIFFICULTY_DATA = {
    "Easy": {"credits": 650, "reactor": 32, "enemy_hp": .92, "enemy_speed": .94, "count": .90,
             "tower_price": .92, "reward": 1.14, "wave_bonus": 1.15, "score_mult": .8, "ability": .9},
    "Normal": {"credits": 500, "reactor": 24, "enemy_hp": 1.0, "enemy_speed": 1.0, "count": 1.06,
               "tower_price": 1.04, "reward": 1.04, "wave_bonus": 1.0, "score_mult": 1.0, "ability": 1.0},
    "Hard": {"credits": 410, "reactor": 18, "enemy_hp": 1.30, "enemy_speed": 1.10, "count": 1.24,
             "tower_price": 1.14, "reward": 1.0, "wave_bonus": .94, "score_mult": 1.35, "ability": 1.1},
}


class Tower:
    next_id = 1

    def __init__(self, kind, pos, pad_index, difficulty, level=0, targeting="First", level_cap=3):
        self.id = Tower.next_id
        Tower.next_id += 1
        self.kind = kind
        self.pos = v2(pos).copy()
        self.pad_index = pad_index
        self.difficulty = difficulty
        self.level = 0
        self.level_cap = max(3, int(level_cap))
        self.targeting = targeting if targeting in TARGET_MODES else "First"
        self.cooldown = random.random() * .2
        self.rotation = 0.0
        self.disable_timer = 0.0
        self.flash = 0.0
        self.total_investment = self.base_cost
        self.drones: list[Drone] = []
        for _ in range(level):
            self.level_up(free=True)
        if self.kind == "drone":
            self.sync_drones()

    @property
    def spec(self):
        return TOWER_SPECS[self.kind]

    @property
    def base_cost(self):
        return int(round(self.spec["cost"] * DIFFICULTY_DATA[self.difficulty]["tower_price"]))

    @property
    def maxed(self):
        return self.level >= self.level_cap

    @property
    def upgrade_cost(self):
        if self.maxed:
            return None
        return int(self.base_cost * (0.72 + self.level * .52))

    @property
    def sale_value(self):
        return int(self.total_investment * .68)

    @property
    def damage(self):
        # Early upgrades feel substantial; later endless tiers grow more slowly so enemy scaling can keep pace.
        early_mult = {"pulse": 1.34, "rail": 1.39, "arc": 1.33, "cryo": 1.28, "missile": 1.36, "drone": 1.31}[self.kind]
        early_levels = min(self.level, 3)
        late_levels = max(0, self.level - 3)
        return self.spec["damage"] * early_mult ** early_levels * 1.22 ** late_levels

    @property
    def range(self):
        return self.spec["range"] * min(1.80, 1 + .075 * self.level)

    @property
    def fire_rate(self):
        base = self.spec["rate"]
        return base * (1 + .16 * min(self.level, 4) + .09 * max(0, self.level - 4))

    @property
    def projectile_speed(self):
        return self.spec["speed"] * min(2.6, 1 + .10 * self.level)

    def level_up(self, free=False):
        if self.maxed:
            return False
        cost = self.upgrade_cost or 0
        self.level += 1
        self.total_investment += cost
        if self.kind == "drone":
            self.sync_drones()
        return True

    def set_level_cap(self, level_cap):
        self.level_cap = max(self.level, 3, int(level_cap))
        if self.kind == "drone":
            self.sync_drones()

    def sync_drones(self):
        desired = 2 + min(6, (self.level + 1) // 2)
        while len(self.drones) < desired:
            self.drones.append(Drone(self, len(self.drones)))
        while len(self.drones) > desired:
            self.drones.pop().alive = False

    def valid_targets(self, game):
        radius_sq = self.range * self.range
        return [e for e in game.query_enemies(self.pos, self.range)
                if e.targetable and e.pos.distance_squared_to(self.pos) <= radius_sq]

    def select_target(self, game):
        candidates = self.valid_targets(game)
        if not candidates:
            return None
        if self.targeting == "First":
            return max(candidates, key=lambda e: e.progress)
        if self.targeting == "Last":
            return min(candidates, key=lambda e: e.progress)
        if self.targeting == "Strongest":
            return max(candidates, key=lambda e: e.durability)
        if self.targeting == "Weakest":
            return min(candidates, key=lambda e: e.durability)
        if self.targeting == "Fastest":
            return max(candidates, key=lambda e: e.effective_speed)
        return min(candidates, key=lambda e: e.pos.distance_squared_to(self.pos))

    def update(self, game, dt):
        self.flash = max(0, self.flash - dt)
        self.disable_timer = max(0, self.disable_timer - dt)
        if self.kind == "drone":
            for drone in list(self.drones):
                drone.update(game, dt)
            return
        self.cooldown -= dt
        target = self.select_target(game)
        if target:
            delta = target.pos - self.pos
            if delta.length_squared() > 0:
                self.rotation = math.atan2(delta.y, delta.x)
        if self.disable_timer > 0 or self.cooldown > 0 or not target:
            return
        self.fire(game, target)
        self.cooldown += 1.0 / max(.05, self.fire_rate)

    def fire(self, game, target):
        self.flash = .09
        if self.kind == "pulse":
            crit_chance = min(.46, .06 + .035 * self.level)
            crit = game.rng.random() < crit_chance
            damage = self.damage * (1.8 if crit else 1.0)
            game.projectiles.append(Projectile("bullet", self.pos, target, damage, self.projectile_speed,
                                               C_CYAN, owner_id=self.id, critical=crit))
            game.sound.play("pulse")
        elif self.kind == "rail":
            direction = (target.pos - self.pos).normalize() if target.pos != self.pos else pygame.Vector2(1, 0)
            game.projectiles.append(Projectile("rail", self.pos, target, self.damage, self.projectile_speed,
                                               C_YELLOW, owner_id=self.id, direction=direction,
                                               armor_pen=min(90, 18 + 6 * self.level),
                                               pierce=min(14, 2 + self.level)))
            game.sound.play("rail")
            game.screen_shake = max(game.screen_shake, 3.5)
        elif self.kind == "arc":
            self.fire_arc(game, target)
            game.sound.play("lightning")
        elif self.kind == "cryo":
            game.projectiles.append(Projectile("cryo", self.pos, target, self.damage, self.projectile_speed,
                                               self.spec["color"], owner_id=self.id,
                                               status=("slow", .30 + .07 * self.level, 2.2 + .35 * self.level),
                                               freeze_chance=min(.48, .08 * self.level)))
            game.sound.play("cryo")
        elif self.kind == "missile":
            game.projectiles.append(Projectile("missile", self.pos, target, self.damage, self.projectile_speed,
                                               C_ORANGE, owner_id=self.id, splash=min(185, 58 + 12 * self.level),
                                               burn=(8 + 5 * self.level, 2.8)))
            game.sound.play("missile")

    def fire_arc(self, game, target):
        hit_ids = set()
        points = [self.pos.copy()]
        current = target
        damage = self.damage
        jumps = min(14, 3 + self.level)
        retention = min(.93, .72 + .035 * self.level)
        for _ in range(jumps):
            if not current or current.id in hit_ids or not current.targetable:
                break
            hit_ids.add(current.id)
            points.append(current.pos.copy())
            current.take_damage(game, damage, armor_pen=4, source=self, color=C_PURPLE)
            jump_range = 88 + self.level * 8
            jump_sq = jump_range * jump_range
            nearby = [e for e in game.query_enemies(current.pos, jump_range)
                      if e.targetable and e.id not in hit_ids and
                      e.pos.distance_squared_to(current.pos) <= jump_sq]
            current = min(nearby, key=lambda e: e.pos.distance_squared_to(points[-1])) if nearby else None
            damage *= retention
        if len(game.beams) < MAX_BEAMS:
            game.beams.append(BeamEffect(points, C_PURPLE, .16, 2 + self.level // 2))

    def draw(self, surf, selected=False):
        color = self.spec["color"]
        if selected:
            range_layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            pygame.draw.circle(range_layer, (*color, 24), self.pos, int(self.range))
            pygame.draw.circle(range_layer, (*color, 110), self.pos, int(self.range), 1)
            surf.blit(range_layer, (0, 0))
        visual_level = min(self.level, 8)
        draw_glow_circle(surf, color, self.pos, 17 + visual_level * 2, alpha=170)
        pygame.draw.circle(surf, C_PANEL, self.pos, 13 + visual_level)
        if self.kind == "pulse":
            end = self.pos + pygame.Vector2(math.cos(self.rotation), math.sin(self.rotation)) * 18
            pygame.draw.line(surf, color, self.pos, end, 5)
        elif self.kind == "rail":
            end = self.pos + pygame.Vector2(math.cos(self.rotation), math.sin(self.rotation)) * 23
            pygame.draw.line(surf, color, self.pos, end, 7)
            pygame.draw.line(surf, C_WHITE, self.pos, end, 2)
        elif self.kind == "arc":
            pts = []
            for i in range(6):
                ang = i * math.tau / 6 + pygame.time.get_ticks() * .001
                pts.append(self.pos + pygame.Vector2(math.cos(ang), math.sin(ang)) * 13)
            pygame.draw.polygon(surf, color, pts, 3)
        elif self.kind == "cryo":
            pygame.draw.circle(surf, color, self.pos, 10, 2)
            pygame.draw.line(surf, color, self.pos + (-12, 0), self.pos + (12, 0), 2)
            pygame.draw.line(surf, color, self.pos + (0, -12), self.pos + (0, 12), 2)
        elif self.kind == "missile":
            pygame.draw.rect(surf, color, pygame.Rect(self.pos.x - 12, self.pos.y - 8, 24, 16), 2, border_radius=3)
            pygame.draw.circle(surf, C_ORANGE, self.pos + (9, 0), 3)
        elif self.kind == "drone":
            pygame.draw.circle(surf, color, self.pos, 12, 3)
            for drone in self.drones:
                drone.draw(surf)
        shown = min(6, self.level + 1)
        for i in range(shown):
            pygame.draw.circle(surf, C_WHITE, self.pos + pygame.Vector2(-15 + i * 6, 19), 2)
        if self.level + 1 > shown:
            pygame.draw.circle(surf, color, self.pos, 24 + min(5, (self.level - 5) // 2), 1)
        if self.flash > 0:
            pygame.draw.circle(surf, C_WHITE, self.pos, 20, 2)
        if self.disable_timer > 0:
            pygame.draw.line(surf, C_RED, self.pos + (-15, -15), self.pos + (15, 15), 4)
            pygame.draw.line(surf, C_RED, self.pos + (-15, 15), self.pos + (15, -15), 4)


class Drone:
    def __init__(self, owner: Tower, index: int):
        self.owner = owner
        self.index = index
        self.pos = owner.pos.copy()
        self.angle = index * math.tau / 4
        self.cooldown = random.random()
        self.alive = True

    def update(self, game, dt):
        if not self.alive or self.owner not in game.towers:
            self.alive = False
            return
        self.angle += dt * (1.7 + .15 * self.owner.level)
        orbit = 29 + (self.index % 2) * 8
        target_pos = self.owner.pos + pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * orbit
        self.pos += (target_pos - self.pos) * min(1, dt * 7)
        self.cooldown -= dt
        if self.owner.disable_timer > 0 or self.cooldown > 0:
            return
        range_sq = self.owner.range * self.owner.range
        candidates = [e for e in game.query_enemies(self.owner.pos, self.owner.range)
                      if e.targetable and e.pos.distance_squared_to(self.owner.pos) <= range_sq]
        if candidates:
            target = min(candidates, key=lambda e: e.pos.distance_squared_to(self.pos))
            game.projectiles.append(Projectile("drone", self.pos, target, self.owner.damage,
                                               self.owner.projectile_speed, C_GREEN, owner_id=self.owner.id))
            game.sound.play("drone")
            self.cooldown = max(.16, .78 - .075 * self.owner.level)

    def draw(self, surf):
        if self.alive:
            draw_glow_circle(surf, C_GREEN, self.pos, 5, alpha=180)
            pygame.draw.line(surf, C_GREEN, self.pos + (-5, 0), self.pos + (5, 0), 2)


class Projectile:
    def __init__(self, kind, pos, target, damage, speed, color, owner_id=None, direction=None,
                 armor_pen=0, pierce=0, splash=0, status=None, freeze_chance=0, burn=None, critical=False):
        self.kind = kind
        self.pos = v2(pos).copy()
        self.prev = self.pos.copy()
        self.target = target
        self.damage = max(0, damage)
        self.speed = max(1, speed)
        self.color = color
        self.owner_id = owner_id
        self.direction = v2(direction).normalize() if direction is not None else None
        self.armor_pen = armor_pen
        self.pierce = pierce
        self.splash = splash
        self.status = status
        self.freeze_chance = freeze_chance
        self.burn = burn
        self.critical = critical
        self.life = 3.2
        self.dead = False
        self.hit_ids = set()
        self.trail = []

    def update(self, game, dt):
        if self.dead:
            return
        self.life -= dt
        if self.life <= 0:
            self.dead = True
            return
        self.prev = self.pos.copy()
        if self.kind == "rail":
            self.pos += self.direction * self.speed * dt
            self.check_piercing(game)
            if not BOARD_RECT.inflate(120, 120).collidepoint(self.pos):
                self.dead = True
        else:
            if not self.target or not self.target.alive or self.target.escaped:
                self.dead = True
                return
            to_target = self.target.pos - self.pos
            dist = to_target.length()
            step = self.speed * dt
            if dist <= step + self.target.radius:
                self.pos = self.target.pos.copy()
                self.impact(game, self.target)
            elif dist > 0:
                self.pos += to_target * (step / dist)
                if distance_point_segment(self.target.pos, self.prev, self.pos) <= self.target.radius + 3:
                    self.impact(game, self.target)
        self.trail.append(self.pos.copy())
        if len(self.trail) > 7:
            self.trail.pop(0)

    def check_piercing(self, game):
        mid = (self.prev + self.pos) * 0.5
        radius = self.prev.distance_to(self.pos) * 0.5 + 40
        for enemy in game.query_enemies(mid, radius):
            if enemy.id in self.hit_ids or not enemy.targetable:
                continue
            if distance_point_segment(enemy.pos, self.prev, self.pos) <= enemy.radius + 4:
                self.hit_ids.add(enemy.id)
                enemy.take_damage(game, self.damage, self.armor_pen, source=self.owner_id, color=C_YELLOW,
                                  critical=self.critical)
                enemy.apply_status("armor_break", 3 + self.pierce, 2.5)
                if len(self.hit_ids) >= self.pierce:
                    self.dead = True
                    break

    def impact(self, game, enemy):
        if self.dead:
            return
        if self.kind == "missile":
            game.explode(self.pos, self.splash, self.damage, owner=self.owner_id, burn=self.burn)
        else:
            enemy.take_damage(game, self.damage, self.armor_pen, source=self.owner_id,
                              color=self.color, critical=self.critical)
            if self.status:
                enemy.apply_status(*self.status)
            if self.freeze_chance > 0 and game.rng.random() < self.freeze_chance:
                enemy.apply_status("freeze", 0, .55 + .12 * self.freeze_chance * 10)
        game.add_burst(self.pos, self.color, 4, 70)
        self.dead = True

    def draw(self, surf):
        if self.dead:
            return
        if len(self.trail) >= 2:
            for i in range(1, len(self.trail)):
                strength = .25 + .65 * i / len(self.trail)
                trail_color = tuple(int(c * strength) for c in self.color)
                pygame.draw.line(surf, trail_color, self.trail[i - 1], self.trail[i], 2)
        if self.kind == "missile":
            pygame.draw.circle(surf, self.color, self.pos, 5)
            pygame.draw.circle(surf, C_WHITE, self.pos, 2)
        elif self.kind == "rail":
            pygame.draw.line(surf, C_WHITE, self.prev, self.pos, 3)
            pygame.draw.line(surf, self.color, self.prev, self.pos, 7)
        else:
            draw_glow_circle(surf, self.color, self.pos, 3, alpha=220)


# -----------------------------------------------------------------------------
# Abilities and waves
# -----------------------------------------------------------------------------
ABILITY_DATA = {
    "emp": {"name": "EMP Pulse", "cooldown": 32.0, "color": C_CYAN,
            "desc": "Stuns every normal enemy. Bosses receive a shorter stun."},
    "orbital": {"name": "Orbital Strike", "cooldown": 44.0, "color": C_ORANGE,
                "desc": "Target a board location. A delayed blast deals heavy area damage."},
    "repair": {"name": "Emergency Repair", "cooldown": 55.0, "color": C_GREEN,
               "desc": "Restores limited reactor health. Cannot be used at full health."},
}


class AbilityManager:
    def __init__(self, difficulty):
        mult = DIFFICULTY_DATA[difficulty]["ability"]
        self.cooldowns = {k: 0.0 for k in ABILITY_DATA}
        self.max_cooldowns = {k: v["cooldown"] * mult for k, v in ABILITY_DATA.items()}
        self.targeting = None
        self.strikes = []
        self.emp_wave = 0.0

    def update(self, game, dt):
        for key in self.cooldowns:
            self.cooldowns[key] = max(0.0, self.cooldowns[key] - dt)
        self.emp_wave = max(0.0, self.emp_wave - dt)
        for strike in list(self.strikes):
            strike["timer"] -= dt
            if strike["timer"] <= 0:
                game.explode(strike["pos"], 112,
                             (520 + game.wave_manager.current_wave * 10) * campaign_hp_scale(game.campaign_level) * .82,
                             owner="orbital", ignore_phase=True, burn=(35, 3.5))
                game.sound.play("explosion")
                game.screen_shake = max(game.screen_shake, 10)
                self.strikes.remove(strike)

    def ready(self, key):
        return self.cooldowns[key] <= 0

    def activate_emp(self, game):
        if not self.ready("emp") or game.wave_manager.phase != "active":
            return False
        for enemy in game.enemies:
            if enemy.alive:
                enemy.apply_status("stun", 0, .65 if enemy.is_boss else 2.0)
        self.cooldowns["emp"] = self.max_cooldowns["emp"]
        self.emp_wave = .65
        game.sound.play("ability")
        game.add_message("EMP PULSE DEPLOYED", C_CYAN)
        return True

    def begin_orbital(self):
        if not self.ready("orbital"):
            return False
        self.targeting = "orbital"
        return True

    def place_orbital(self, game, pos):
        if self.targeting != "orbital" or not self.ready("orbital") or not BOARD_RECT.collidepoint(pos):
            return False
        self.strikes.append({"pos": v2(pos).copy(), "timer": 1.25})
        self.cooldowns["orbital"] = self.max_cooldowns["orbital"]
        self.targeting = None
        game.sound.play("ability")
        game.add_message("ORBITAL STRIKE INBOUND", C_ORANGE)
        return True

    def repair(self, game):
        if not self.ready("repair") or game.reactor_health >= game.reactor_max or game.reactor_health <= 0:
            return False
        amount = max(3, int(game.reactor_max * .26))
        game.reactor_health = min(game.reactor_max, game.reactor_health + amount)
        self.cooldowns["repair"] = self.max_cooldowns["repair"]
        game.sound.play("ability")
        game.add_floating(FloatingText(game.map.routes[0][-1], f"+{amount} CORE", C_GREEN, 1.2, "small"))
        return True

    def cancel(self):
        self.targeting = None

    def draw(self, surf, mouse):
        if self.emp_wave > 0:
            radius = int(650 * (1 - self.emp_wave / .65))
            layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            pygame.draw.circle(layer, (*C_CYAN, 120), BOARD_RECT.center, radius, 5)
            surf.blit(layer, (0, 0))
        for strike in self.strikes:
            p = strike["pos"]
            pulse = 4 * math.sin(pygame.time.get_ticks() * .015)
            pygame.draw.circle(surf, C_RED, p, int(112 + pulse), 2)
            pygame.draw.line(surf, C_RED, p + (-16, 0), p + (16, 0), 2)
            pygame.draw.line(surf, C_RED, p + (0, -16), p + (0, 16), 2)
        if self.targeting == "orbital" and BOARD_RECT.collidepoint(mouse):
            layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            pygame.draw.circle(layer, (*C_ORANGE, 40), mouse, 112)
            pygame.draw.circle(layer, (*C_ORANGE, 210), mouse, 112, 2)
            surf.blit(layer, (0, 0))


def build_wave_definitions():
    waves = []
    for wave in range(1, 31):
        if wave == 10:
            groups = [("grunt", 8, .55), ("shield", 4, .8), ("boss:siege", 1, 1.0)]
        elif wave == 20:
            groups = [("runner", 12, .28), ("phase", 7, .7), ("boss:serpent", 1, 1.0)]
        elif wave == 30:
            groups = [("tank", 6, .85), ("shield", 8, .55), ("phase", 6, .65), ("boss:devourer", 1, 1.0)]
        elif wave <= 4:
            groups = [("grunt" if wave > 1 else "runner", 6 + wave * 2, max(.35, .72 - wave * .06))]
            if wave >= 3:
                groups.append(("runner", 4 + wave, .35))
        elif wave == 5:
            groups = [("grunt", 10, .55), ("runner", 10, .28), ("tank", 2, 1.1)]
        elif wave <= 9:
            groups = [("grunt", 8 + wave, .45), ("runner", 5 + wave, .26)]
            if wave >= 7: groups.append(("shield", 3 + wave // 2, .68))
            if wave >= 8: groups.append(("tank", 2 + wave // 4, .95))
        elif wave <= 14:
            groups = [("shield", 5 + wave // 2, .52), ("tank", 3 + wave // 3, .85),
                      ("swarm", 12 + wave, .18)]
        elif wave == 15:
            groups = [("tank", 9, .72), ("shield", 12, .42), ("runner", 22, .19)]
        elif wave <= 19:
            groups = [("splitter", 5 + wave // 2, .62), ("regenerator", 4 + wave // 3, .78),
                      ("phase", 4 + wave // 3, .66), ("swarm", 15 + wave, .16)]
        elif wave <= 24:
            groups = [("swarm", 22 + wave, .12), ("runner", 14 + wave, .17),
                      ("shield", 8 + wave // 2, .40), ("splitter", 7 + wave // 3, .50)]
        elif wave == 25:
            groups = [("tank", 16, .58), ("regenerator", 13, .52), ("phase", 14, .44), ("shield", 16, .38)]
        else:
            groups = [("tank", 8 + wave // 2, .52), ("shield", 12 + wave // 2, .34),
                      ("splitter", 9 + wave // 3, .40), ("regenerator", 8 + wave // 3, .48),
                      ("phase", 8 + wave // 3, .43), ("runner", 12 + wave, .15)]
        waves.append(groups)
    return waves


WAVE_DEFINITIONS = build_wave_definitions()


class WaveManager:
    def __init__(self, start_wave=0, auto_wave=False):
        self.current_wave = start_wave
        self.phase = "prep"
        self.prep_timer = 12.0
        self.spawn_queue = []
        self.spawn_timer = 0.0
        self.auto_wave = auto_wave
        self.wave_started = False
        self.early_bonus_awarded = False
        self.flawless_health = None
        self.total_to_spawn = 0
        self.spawned = 0

    def preview(self):
        index = min(29, self.current_wave)
        return [g[0].replace("boss:", "") for g in WAVE_DEFINITIONS[index]][:5]

    def start_wave(self, game, early=False):
        if self.phase != "prep" or self.current_wave >= 30 or self.wave_started:
            return False
        # Persist the exact preparation state before any early bonus or live entities exist.
        game.capture_safe_snapshot()
        game.save.save()
        self.current_wave += 1
        self.phase = "active"
        self.wave_started = True
        self.flawless_health = game.reactor_health
        groups = WAVE_DEFINITIONS[self.current_wave - 1]
        queue = []
        count_mult = DIFFICULTY_DATA[game.difficulty]["count"] * (1.0 + min(.90, (game.campaign_level - 1) * .10))
        for kind, count, interval in groups:
            actual = 1 if kind.startswith("boss:") else max(1, int(round(count * count_mult)))
            for _ in range(actual):
                queue.append((kind, interval))
        self.spawn_queue = list(reversed(queue))
        self.total_to_spawn = len(queue)
        self.spawned = 0
        self.spawn_timer = .35
        if early and self.prep_timer > 0 and not self.early_bonus_awarded:
            bonus = int(8 + self.prep_timer * 2.2 + self.current_wave)
            game.credits += bonus
            game.save.data["credits_earned"] += bonus
            game.add_floating(FloatingText((BOARD_RECT.centerx, BOARD_RECT.top + 30),
                                                    f"EARLY +{bonus}", C_YELLOW, 1.2, "small"))
            self.early_bonus_awarded = True
        game.sound.play("boss" if self.current_wave in (10, 20, 30) else "wave_start")
        game.add_message(f"LEVEL {game.campaign_level} • WAVE {self.current_wave} INCOMING",
                         C_RED if self.current_wave in (10,20,30) else C_CYAN)
        return True

    def update(self, game, dt):
        if self.phase == "prep":
            self.prep_timer -= dt
            if self.prep_timer <= 0 and self.auto_wave:
                self.start_wave(game, early=False)
            return
        if self.phase != "active":
            return
        if self.spawn_queue:
            self.spawn_timer -= dt
            while self.spawn_queue and self.spawn_timer <= 0:
                kind, interval = self.spawn_queue.pop()
                route_id = game.route_spawn_counter % len(game.map.routes)
                if kind.startswith("boss:"):
                    boss_kind = kind.split(":", 1)[1]
                    enemy = BossEnemy(boss_kind, game.map.route_vectors(route_id), route_id,
                                      game.difficulty, self.current_wave, game.campaign_level)
                else:
                    enemy = Enemy(kind, game.map.route_vectors(route_id), route_id,
                                  game.difficulty, self.current_wave, campaign_level=game.campaign_level)
                game.enemies_to_add.append(enemy)
                game.route_spawn_counter += 1
                self.spawned += 1
                self.spawn_timer += max(.08, interval)
        living_required = any(e.alive and e.required for e in game.enemies) or any(e.required for e in game.enemies_to_add)
        if not self.spawn_queue and not living_required:
            self.complete_wave(game)

    def complete_wave(self, game):
        if self.phase != "active":
            return
        self.phase = "prep"
        self.wave_started = False
        self.early_bonus_awarded = False
        reward_scale = campaign_reward_scale(game.campaign_level)
        base = int((32 + self.current_wave * 7) * DIFFICULTY_DATA[game.difficulty]["wave_bonus"] * reward_scale)
        savings = min(70 + game.campaign_level * 15, int(game.credits * .035))
        flawless = (int((18 + self.current_wave * 1.5) * reward_scale)
                    if self.flawless_health is not None and game.reactor_health >= self.flawless_health else 0)
        level_cleared = self.current_wave >= 30
        level_bonus = int((360 + game.campaign_level * 140) * DIFFICULTY_DATA[game.difficulty]["wave_bonus"]) if level_cleared else 0
        total = base + savings + flawless + level_bonus
        game.credits += total
        game.save.data["credits_earned"] += total
        game.score += int(total * 2 * DIFFICULTY_DATA[game.difficulty]["score_mult"] * (1 + .08 * (game.campaign_level - 1)))
        game.sound.play("victory" if level_cleared else "wave_complete")

        if level_cleared:
            cleared_level = game.campaign_level
            game.save.data["games_won"] += 1
            game.save.data["levels_cleared"] += 1
            repair = max(2, int(math.ceil(game.reactor_max * .20)))
            game.reactor_health = min(game.reactor_max, game.reactor_health + repair)
            game.campaign_level += 1
            self.current_wave = 0
            self.prep_timer = 14.0
            new_cap = tower_level_cap(game.campaign_level)
            for tower in game.towers:
                tower.set_level_cap(new_cap)
            for key in game.abilities.cooldowns:
                game.abilities.cooldowns[key] *= .65
            game.add_message(
                f"LEVEL {cleared_level} CLEARED +{total}  •  LEVEL {game.campaign_level}  •  TOWER TIERS {new_cap + 1}",
                C_GREEN, 4.5)
        else:
            self.prep_timer = 9.0
            game.add_message(f"WAVE CLEARED  +{total}", C_GREEN)

        absolute_wave = total_campaign_wave(game.campaign_level, self.current_wave)
        game.save.data["best_wave"] = max(game.save.data["best_wave"], absolute_wave)
        game.save.data["best_level"] = max(game.save.data["best_level"], game.campaign_level)
        game.capture_safe_snapshot()
        game.save.save()

    def remaining(self, game):
        return len(self.spawn_queue) + sum(1 for e in game.enemies if e.alive and e.required) + sum(1 for e in game.enemies_to_add if e.required)

    def progress(self, game):
        total = max(1, self.total_to_spawn)
        remaining = self.remaining(game)
        return clamp(1 - remaining / total, 0, 1)


# -----------------------------------------------------------------------------
# Main Game
# -----------------------------------------------------------------------------
class Game:
    def __init__(self, headless=False):
        pygame.init()
        pygame.font.init()
        self.headless = headless
        self.save = SaveManager(Path(__file__))
        self.fullscreen = self.save.data["fullscreen"] and not headless
        flags = pygame.RESIZABLE | (pygame.FULLSCREEN if self.fullscreen else 0)
        try:
            self.window = pygame.display.set_mode((1280, 720), flags)
        except pygame.error:
            self.window = pygame.Surface((1280, 720))
        pygame.display.set_caption("Tower-Vision")
        self.canvas = pygame.Surface((DESIGN_W, DESIGN_H)).convert()
        self.fonts = {
            "title": pygame.font.Font(None, 72), "h1": pygame.font.Font(None, 44),
            "h2": pygame.font.Font(None, 31), "body": pygame.font.Font(None, 24),
            "small": pygame.font.Font(None, 20), "tiny": pygame.font.Font(None, 17),
        }
        self.clock = pygame.time.Clock()
        self.sound = SoundManager(self.save.data["audio_volume"], self.save.data["muted"])
        self.maps = create_maps()
        self.selected_map_id = self.save.data["selected_map"]
        self.difficulty = self.save.data["selected_difficulty"]
        self.background = self.load_background()
        self.background_particles = [
            [pygame.Vector2(random.randrange(DESIGN_W), random.randrange(DESIGN_H)), random.uniform(8, 24),
             random.choice((C_CYAN, C_BLUE, C_GREEN, C_PURPLE)), random.uniform(1, 3)] for _ in range(34)
        ]
        self.tooltip = TooltipManager()
        self.running = True
        self.state = "title"
        self.dialog = None
        self.speed = 1
        self.accumulator = 0.0
        self.last_real_time = time.perf_counter()
        self.play_session = 0.0
        self.rng = random.Random()
        self.message = ""
        self.message_color = C_TEXT
        self.message_timer = 0.0
        self.screen_shake = 0.0
        self.core_flash = 0.0
        self.hover_pad = None
        self.placing_kind = None
        self.selected_tower = None
        self.pending_sell = None
        self.menu_hover_id = None
        self.buttons = []
        self.title_buttons = []
        self.map_buttons = []
        self.diff_buttons = []
        self.game_buttons = []
        self._build_static_buttons()
        self.reset_runtime()

    def load_background(self):
        startup = Path(__file__).resolve().parent.parent / "presets" / "startup"
        candidates = []
        if startup.is_dir():
            candidates = [p for p in startup.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            self.rng = random.Random(time.time_ns())
            self.rng.shuffle(candidates)
        for path in candidates:
            try:
                img = pygame.image.load(str(path)).convert()
                iw, ih = img.get_size()
                if iw <= 0 or ih <= 0:
                    continue
                scale = max(DESIGN_W / iw, DESIGN_H / ih)
                size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
                img = pygame.transform.smoothscale(img, size)
                crop = pygame.Rect((size[0] - DESIGN_W) // 2, (size[1] - DESIGN_H) // 2, DESIGN_W, DESIGN_H)
                return img.subsurface(crop).copy()
            except (pygame.error, OSError, ValueError):
                continue
        # Generated fallback: dark blue-green energy field.
        surf = pygame.Surface((DESIGN_W, DESIGN_H))
        for y in range(DESIGN_H):
            t = y / DESIGN_H
            color = (int(7 + 10 * t), int(13 + 16 * t), int(24 + 23 * t))
            pygame.draw.line(surf, color, (0, y), (DESIGN_W, y))
        rng = random.Random(9917)
        for _ in range(140):
            p = (rng.randrange(DESIGN_W), rng.randrange(DESIGN_H))
            pygame.draw.circle(surf, rng.choice((C_CYAN, C_BLUE, C_GREEN)), p, rng.choice((1, 1, 2)))
        return surf

    def _build_static_buttons(self):
        self.title_buttons = [
            Button((495, 235, 290, 46), "New Game", lambda: self.enter_state("map"), "Start a fresh endless run built from escalating 30-wave levels."),
            Button((495, 289, 290, 46), "Continue Game", self.continue_game, "Restore the latest safe wave-preparation snapshot."),
            Button((495, 343, 290, 42), "Instructions", lambda: self.enter_state("instructions"), "Read controls, tower roles, enemies and saving details."),
            Button((495, 393, 140, 42), "Difficulty", lambda: self.enter_state("difficulty"), "Choose Easy, Normal or Hard."),
            Button((645, 393, 140, 42), "Map", lambda: self.enter_state("map"), "Choose one of three strategic maps."),
            Button((495, 443, 140, 42), "Statistics", lambda: self.enter_state("statistics"), "View persistent career statistics."),
            Button((645, 443, 140, 42), "Sound", self.toggle_mute, "Toggle all generated game audio. Shortcut: M."),
            Button((495, 493, 140, 42), "Fullscreen", self.toggle_fullscreen, "Toggle fullscreen. Shortcut: F11."),
            Button((645, 493, 140, 42), "Quit", self.request_quit, "Save settings and quit."),
            Button((495, 543, 140, 38), "Volume −", lambda: self.adjust_volume(-0.10), "Lower generated sound-effect volume."),
            Button((645, 543, 140, 38), "Volume +", lambda: self.adjust_volume(0.10), "Raise generated sound-effect volume."),
        ]
        self.map_buttons = [
            Button((120 + i * 355, 230, 330, 260), self.maps[mid].name,
                   lambda m=mid: self.choose_map(m), self.maps[mid].description) for i, mid in enumerate(MAP_IDS)
        ]
        self.diff_buttons = [
            Button((350, 230 + i * 95, 580, 72), d, lambda value=d: self.choose_difficulty(value),
                   self.difficulty_tooltip(d)) for i, d in enumerate(DIFFICULTIES)
        ]

    def reset_runtime(self):
        self.map = self.maps[self.selected_map_id]
        self.towers: list[Tower] = []
        self.enemies: list[Enemy] = []
        self.enemies_to_add: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.particles: list[Particle] = []
        self.floating_texts: list[FloatingText] = []
        self.beams: list[BeamEffect] = []
        self.enemy_grid: dict[tuple[int, int], list[Enemy]] = {}
        self.campaign_level = 1
        self.credits = DIFFICULTY_DATA[self.difficulty]["credits"]
        self.score = 0
        self.reactor_max = DIFFICULTY_DATA[self.difficulty]["reactor"]
        self.reactor_health = self.reactor_max
        self.wave_manager = WaveManager(0, self.save.data["auto_wave"])
        self.abilities = AbilityManager(self.difficulty)
        self.selected_tower = None
        self.placing_kind = None
        self.pending_sell = None
        self.route_spawn_counter = 0
        self.outcome_decided = False
        self.safe_snapshot = None
        self.hover_pad = None
        self.message = ""
        self.message_timer = 0
        self.screen_shake = 0
        self.core_flash = 0

    def difficulty_tooltip(self, difficulty):
        d = DIFFICULTY_DATA[difficulty]
        return (f"{difficulty}: {d['credits']} starting credits, {d['reactor']} reactor health, "
                f"enemy health ×{d['enemy_hp']:.2f}, speed ×{d['enemy_speed']:.2f}, tower price ×{d['tower_price']:.2f}.")

    def enter_state(self, state):
        self.state = state
        self.dialog = None
        self.sound.play("menu_confirm")

    def choose_map(self, map_id):
        self.selected_map_id = map_id
        self.save.data["selected_map"] = map_id
        self.save.save()
        self.enter_state("difficulty")

    def choose_difficulty(self, difficulty):
        self.difficulty = difficulty
        self.save.data["selected_difficulty"] = difficulty
        self.save.save()
        self.start_new_game()

    def start_new_game(self):
        self.reset_runtime()
        self.state = "game"
        self.capture_safe_snapshot()
        self.save.save()
        self.sound.play("menu_confirm")

    def continue_game(self):
        camp = self.save.data.get("campaign")
        if not camp:
            self.add_message("No valid campaign snapshot found", C_RED)
            return
        self.selected_map_id = camp["map_id"]
        self.difficulty = camp["difficulty"]
        self.reset_runtime()
        self.map = self.maps[self.selected_map_id]
        self.campaign_level = camp.get("level", 1)
        self.wave_manager.current_wave = camp["wave"]
        self.reactor_max = camp["reactor_max"]
        self.reactor_health = camp["reactor_health"]
        self.credits = camp["credits"]
        self.score = camp["score"]
        used_pads = set()
        for td in camp["towers"]:
            if td["pad_index"] >= len(self.map.pads) or td["pad_index"] in used_pads:
                continue
            pad = pygame.Vector2(self.map.pads[td["pad_index"]])
            if pad.distance_to((td["x"], td["y"])) > 5:
                continue
            tower = Tower(td["type"], pad, td["pad_index"], self.difficulty,
                          level=td["level"], targeting=td["targeting"],
                          level_cap=tower_level_cap(self.campaign_level))
            self.towers.append(tower)
            used_pads.add(td["pad_index"])
        for key in self.abilities.cooldowns:
            self.abilities.cooldowns[key] = camp["ability_cooldowns"].get(key, 0.0)
        self.capture_safe_snapshot()
        self.state = "game"
        self.sound.play("menu_confirm")

    def capture_safe_snapshot(self, pre_wave=False):
        # Never replace a known-safe preparation snapshot with partial live-wave state.
        if self.wave_manager.phase == "active" and not pre_wave:
            return False
        wave = max(0, self.wave_manager.current_wave - 1) if pre_wave else self.wave_manager.current_wave
        snapshot = {
            "map_id": self.selected_map_id,
            "difficulty": self.difficulty,
            "level": self.campaign_level,
            "wave": min(29, wave),
            "reactor_health": self.reactor_health,
            "reactor_max": self.reactor_max,
            "credits": max(0, int(self.credits)),
            "score": max(0, int(self.score)),
            "towers": [{"type": t.kind, "level": t.level, "x": t.pos.x, "y": t.pos.y,
                        "targeting": t.targeting, "pad_index": t.pad_index} for t in self.towers],
            "ability_cooldowns": {k: float(v) for k, v in self.abilities.cooldowns.items()},
        }
        self.safe_snapshot = snapshot
        self.save.data["campaign"] = snapshot
        return True

    def request_quit(self):
        self.dialog = {"title": "Quit Tower-Vision?", "message": "Your latest safe preparation snapshot and statistics will be saved.",
                       "yes": self.quit_game, "no": lambda: setattr(self, "dialog", None)}

    def quit_game(self):
        self.save_settings()
        self.running = False

    def save_settings(self):
        self.save.data["selected_difficulty"] = self.difficulty
        self.save.data["selected_map"] = self.selected_map_id
        self.save.data["audio_volume"] = self.sound.volume
        self.save.data["muted"] = self.sound.muted
        self.save.data["fullscreen"] = self.fullscreen
        self.save.data["auto_wave"] = self.wave_manager.auto_wave
        self.save.data["high_score"] = max(self.save.data["high_score"], int(self.score))
        self.save.data["total_play_time"] += self.play_session
        self.play_session = 0.0
        if self.safe_snapshot and not self.outcome_decided:
            self.save.data["campaign"] = self.safe_snapshot
        self.save.save()

    def return_to_title(self):
        self.capture_safe_snapshot()
        self.save_settings()
        self.enter_state("title")

    def reset_save_confirm(self):
        self.dialog = {"title": "Reset all save data?", "message": "High score, statistics, settings and campaign progress will be erased.",
                       "yes": self.reset_save, "no": lambda: setattr(self, "dialog", None)}

    def reset_save(self):
        self.save.reset()
        self.selected_map_id = "neon_circuit"
        self.difficulty = "Normal"
        self.sound.muted = False
        self.sound.set_volume(.55)
        self.dialog = None
        self._build_static_buttons()
        self.add_message("Save data reset", C_GREEN)

    def toggle_mute(self):
        self.sound.toggle_mute()
        self.save.data["muted"] = self.sound.muted
        self.save.save()

    def adjust_volume(self, delta):
        self.sound.set_volume(self.sound.volume + delta)
        if self.sound.muted and self.sound.volume > 0:
            self.sound.muted = False
            self.sound.apply_volume()
        self.save.data["audio_volume"] = self.sound.volume
        self.save.data["muted"] = self.sound.muted
        self.sound.play("menu_move")
        self.save.save()

    def toggle_fullscreen(self):
        if self.headless:
            return
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        size = (0, 0) if self.fullscreen else (1280, 720)
        try:
            self.window = pygame.display.set_mode(size, flags)
        except pygame.error:
            self.fullscreen = not self.fullscreen
        self.save.data["fullscreen"] = self.fullscreen
        self.save.save()

    def add_message(self, message, color=C_TEXT, duration=2.0):
        self.message = message
        self.message_color = color
        self.message_timer = duration

    def rebuild_enemy_grid(self):
        grid = {}
        for enemy in self.enemies:
            if enemy.alive and not enemy.escaped:
                key = (int(enemy.pos.x) // SPATIAL_CELL, int(enemy.pos.y) // SPATIAL_CELL)
                grid.setdefault(key, []).append(enemy)
        self.enemy_grid = grid

    def query_enemies(self, pos, radius):
        if not self.enemy_grid:
            return self.enemies
        p = v2(pos)
        min_x = int((p.x - radius) // SPATIAL_CELL)
        max_x = int((p.x + radius) // SPATIAL_CELL)
        min_y = int((p.y - radius) // SPATIAL_CELL)
        max_y = int((p.y + radius) // SPATIAL_CELL)
        result = []
        for cy in range(min_y, max_y + 1):
            for cx in range(min_x, max_x + 1):
                result.extend(self.enemy_grid.get((cx, cy), ()))
        return result

    def add_floating(self, item):
        if len(self.floating_texts) < MAX_FLOATING_TEXTS:
            self.floating_texts.append(item)

    def add_burst(self, pos, color, count=8, speed=100):
        available = max(0, MAX_PARTICLES - len(self.particles))
        for _ in range(min(count, available)):
            ang = self.rng.random() * math.tau
            vel = pygame.Vector2(math.cos(ang), math.sin(ang)) * self.rng.uniform(speed * .35, speed)
            self.particles.append(Particle(pos, vel, color, self.rng.uniform(.25, .65), self.rng.uniform(2, 4)))

    def explode(self, pos, radius, damage, owner=None, burn=None, ignore_phase=False):
        hit_ids = set()
        for enemy in self.query_enemies(pos, radius + 36):
            if enemy.id in hit_ids or not enemy.alive or enemy.escaped:
                continue
            dist = enemy.pos.distance_to(pos)
            if dist <= radius + enemy.radius:
                hit_ids.add(enemy.id)
                falloff = .55 + .45 * (1 - clamp(dist / max(1, radius), 0, 1))
                enemy.take_damage(self, damage * falloff, armor_pen=4, source=owner,
                                  ignore_phase=ignore_phase, color=C_ORANGE)
                if burn:
                    enemy.apply_status("burn", burn[0], burn[1])
        self.add_burst(pos, C_ORANGE, 18, 210)
        if len(self.beams) < MAX_BEAMS:
            self.beams.append(BeamEffect([v2(pos) + (-radius*.6, 0), v2(pos) + (radius*.6, 0)], C_ORANGE, .18, 3))
        self.sound.play("explosion")
        self.screen_shake = max(self.screen_shake, 4)

    def damage_reactor(self, amount):
        if self.outcome_decided or self.reactor_health <= 0:
            return
        self.reactor_health = max(0, self.reactor_health - max(0, amount))
        self.core_flash = .35
        self.screen_shake = max(self.screen_shake, 8)
        self.sound.play("reactor")
        self.add_message(f"REACTOR HIT  -{amount}", C_RED)
        if self.reactor_health <= 0:
            self.trigger_defeat()

    def trigger_victory(self):
        if self.outcome_decided or self.reactor_health <= 0:
            return
        # Legacy guard retained for old state references; normal play advances levels instead.
        self.outcome_decided = True
        self.state = "victory"
        self.save.data["games_won"] += 1
        self.save.data["best_wave"] = max(self.save.data["best_wave"], total_campaign_wave(self.campaign_level, 30))
        self.save.data["best_level"] = max(self.save.data["best_level"], self.campaign_level)
        self.save.data["high_score"] = max(self.save.data["high_score"], int(self.score))
        self.save.data["campaign"] = None
        self.sound.play("victory")
        self.save.save()

    def trigger_defeat(self):
        if self.outcome_decided:
            return
        self.outcome_decided = True
        self.state = "defeat"
        self.save.data["games_lost"] += 1
        self.save.data["best_wave"] = max(
            self.save.data["best_wave"], total_campaign_wave(self.campaign_level, self.wave_manager.current_wave))
        self.save.data["best_level"] = max(self.save.data["best_level"], self.campaign_level)
        self.save.data["high_score"] = max(self.save.data["high_score"], int(self.score))
        self.save.data["campaign"] = None
        self.sound.play("defeat")
        self.save.save()

    def place_tower(self, kind, pad_index):
        if kind not in TOWER_SPECS or not (0 <= pad_index < len(self.map.pads)):
            return False
        if any(t.pad_index == pad_index for t in self.towers):
            self.add_message("Build pad occupied", C_RED)
            return False
        cost = int(round(TOWER_SPECS[kind]["cost"] * DIFFICULTY_DATA[self.difficulty]["tower_price"]))
        if self.credits < cost:
            self.add_message("Insufficient credits", C_RED)
            return False
        pos = pygame.Vector2(self.map.pads[pad_index])
        if not BOARD_RECT.collidepoint(pos):
            return False
        self.credits -= cost
        tower = Tower(kind, pos, pad_index, self.difficulty, level_cap=tower_level_cap(self.campaign_level))
        self.towers.append(tower)
        self.selected_tower = tower
        self.save.data["towers_built"] += 1
        self.sound.play("place")
        self.add_burst(pos, tower.spec["color"], 10, 120)
        self.capture_safe_snapshot()
        return True

    def upgrade_selected(self):
        tower = self.selected_tower
        if not tower or tower not in self.towers or tower.maxed:
            self.add_message("Tower is already at maximum level" if tower else "No tower selected", C_MUTED)
            return False
        cost = tower.upgrade_cost
        if self.credits < cost:
            self.add_message("Insufficient credits", C_RED)
            return False
        self.credits -= cost
        tower.level_up()
        self.save.data["towers_upgraded"] += 1
        self.sound.play("upgrade")
        self.add_burst(tower.pos, tower.spec["color"], 14, 150)
        self.capture_safe_snapshot()
        return True

    def sell_selected_request(self):
        if not self.selected_tower or self.selected_tower not in self.towers:
            return
        if self.pending_sell is self.selected_tower:
            self.sell_selected()
        else:
            self.pending_sell = self.selected_tower
            self.add_message("Press Delete or Sell again to confirm", C_YELLOW)

    def sell_selected(self):
        tower = self.selected_tower
        if not tower or tower not in self.towers:
            return False
        self.credits += tower.sale_value
        for drone in tower.drones:
            drone.alive = False
        for projectile in self.projectiles:
            if projectile.owner_id == tower.id:
                projectile.dead = True
        self.towers.remove(tower)
        self.sound.play("sell")
        self.add_message(f"Sold for {tower.sale_value}", C_GREEN)
        self.selected_tower = None
        self.pending_sell = None
        self.capture_safe_snapshot()
        return True

    def nearest_pad(self, pos, max_dist=34):
        best, dist = None, max_dist
        for i, p in enumerate(self.map.pads):
            d = pygame.Vector2(p).distance_to(pos)
            if d < dist:
                best, dist = i, d
        return best

    def virtual_mouse(self):
        mx, my = pygame.mouse.get_pos()
        ww, wh = self.window.get_size()
        scale = min(ww / DESIGN_W, wh / DESIGN_H)
        if scale <= 0:
            return (-9999, -9999)
        ox = (ww - DESIGN_W * scale) / 2
        oy = (wh - DESIGN_H * scale) / 2
        return ((mx - ox) / scale, (my - oy) / scale)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.request_quit()
            return
        if event.type == pygame.WINDOWFOCUSLOST and self.state == "game" and self.wave_manager.phase == "active":
            self.state = "paused"
        if event.type == pygame.VIDEORESIZE and not self.fullscreen and not self.headless:
            # Keep the virtual controls usable instead of allowing an unreadably tiny client area.
            size = (max(800, event.w), max(450, event.h))
            try:
                self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
            except pygame.error:
                pass
        if event.type == pygame.MOUSEMOTION:
            self.handle_menu_hover(self.virtual_mouse())
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                self.toggle_fullscreen(); return
            if event.key == pygame.K_m:
                self.toggle_mute(); return
            if self.dialog:
                if event.key in (pygame.K_ESCAPE, pygame.K_n): self.dialog["no"]()
                elif event.key in (pygame.K_RETURN, pygame.K_y): self.dialog["yes"]()
                return
            if self.state in ("game", "paused"):
                self.handle_game_key(event.key)
            elif event.key == pygame.K_ESCAPE:
                if self.state == "title": self.request_quit()
                else: self.enter_state("title")
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.virtual_mouse()
            if self.dialog:
                self.handle_dialog_click(pos)
            elif event.button == 1:
                self.handle_left_click(pos)
            elif event.button == 3 and self.state in ("game", "paused"):
                self.placing_kind = None
                self.abilities.cancel()


    def handle_menu_hover(self, pos):
        groups = {"title": self.title_buttons, "map": self.map_buttons, "difficulty": self.diff_buttons}
        hovered = None
        for i, button in enumerate(groups.get(self.state, [])):
            if button.enabled and button.rect.collidepoint(pos):
                hovered = (self.state, i)
                break
        if hovered is not None and hovered != self.menu_hover_id:
            self.sound.play("menu_move")
        self.menu_hover_id = hovered

    def handle_game_key(self, key):
        if key in (pygame.K_p, pygame.K_ESCAPE):
            if self.state == "paused": self.state = "game"
            elif self.placing_kind or self.abilities.targeting: self.placing_kind = None; self.abilities.cancel()
            else: self.state = "paused"
            return
        if self.state == "paused":
            return
        if key == pygame.K_SPACE:
            self.wave_manager.start_wave(self, early=True)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.speed = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3}[key]
        elif key == pygame.K_DELETE:
            self.sell_selected_request()
        elif key == pygame.K_u:
            self.upgrade_selected()
        elif key in (pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y):
            idx = (pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y).index(key)
            self.placing_kind = TOWER_KEYS[idx]
            self.abilities.cancel()

    def handle_left_click(self, pos):
        if self.state == "title":
            for b in self.title_buttons:
                if b.click(pos): self.sound.play("menu_confirm"); break
            return
        if self.state == "map":
            for b in self.map_buttons:
                if b.click(pos): break
            if pygame.Rect(30, 650, 120, 42).collidepoint(pos): self.enter_state("title")
            return
        if self.state == "difficulty":
            for b in self.diff_buttons:
                if b.click(pos): break
            if pygame.Rect(30, 650, 120, 42).collidepoint(pos): self.enter_state("map")
            return
        if self.state in ("instructions", "statistics", "victory", "defeat"):
            if pygame.Rect(30, 650, 160, 42).collidepoint(pos): self.enter_state("title")
            if self.state == "statistics" and pygame.Rect(1020, 650, 220, 42).collidepoint(pos): self.reset_save_confirm()
            if self.state in ("victory", "defeat") and pygame.Rect(505, 560, 270, 48).collidepoint(pos): self.enter_state("map")
            return
        if self.state == "paused":
            if pygame.Rect(500, 330, 280, 48).collidepoint(pos): self.state = "game"
            elif pygame.Rect(500, 390, 280, 48).collidepoint(pos): self.return_to_title()
            return
        if self.state != "game":
            return
        # UI is handled first so ability targeting can never fire through panels.
        if SIDE_RECT.collidepoint(pos) or pygame.Rect(0, 0, DESIGN_W, 72).collidepoint(pos):
            self.handle_game_ui_click(pos)
            return
        if self.abilities.targeting == "orbital":
            self.abilities.place_orbital(self, pos)
            return
        if self.placing_kind:
            pad = self.nearest_pad(pos)
            if pad is not None:
                if self.place_tower(self.placing_kind, pad):
                    self.placing_kind = None
            else:
                self.add_message("Select a highlighted build pad", C_RED)
            return
        clicked = None
        for tower in reversed(self.towers):
            if tower.pos.distance_to(pos) <= 24:
                clicked = tower; break
        self.selected_tower = clicked
        self.pending_sell = None

    def handle_game_ui_click(self, pos):
        # Top bar controls.
        if pygame.Rect(825, 18, 76, 34).collidepoint(pos): self.wave_manager.start_wave(self, early=True)
        elif pygame.Rect(910, 18, 55, 34).collidepoint(pos): self.state = "paused"
        elif pygame.Rect(975, 18, 50, 34).collidepoint(pos): self.speed = 1
        elif pygame.Rect(1030, 18, 50, 34).collidepoint(pos): self.speed = 2
        elif pygame.Rect(1085, 18, 50, 34).collidepoint(pos): self.speed = 3
        elif pygame.Rect(1145, 18, 95, 34).collidepoint(pos):
            self.wave_manager.auto_wave = not self.wave_manager.auto_wave
            self.save.data["auto_wave"] = self.wave_manager.auto_wave
        # Tower selection buttons.
        for i, kind in enumerate(TOWER_KEYS):
            rect = pygame.Rect(980 + (i % 2) * 136, 92 + (i // 2) * 58, 126, 50)
            if rect.collidepoint(pos):
                self.placing_kind = kind
                self.abilities.cancel()
                return
        # Abilities.
        for i, key in enumerate(("emp", "orbital", "repair")):
            rect = pygame.Rect(980, 285 + i * 48, 266, 40)
            if rect.collidepoint(pos):
                if key == "emp": self.abilities.activate_emp(self)
                elif key == "orbital": self.abilities.begin_orbital()
                else: self.abilities.repair(self)
                return
        if self.selected_tower:
            if pygame.Rect(980, 588, 126, 40).collidepoint(pos): self.upgrade_selected()
            elif pygame.Rect(1120, 588, 126, 40).collidepoint(pos): self.sell_selected_request()
            elif pygame.Rect(980, 640, 266, 34).collidepoint(pos):
                idx = (TARGET_MODES.index(self.selected_tower.targeting) + 1) % len(TARGET_MODES)
                self.selected_tower.targeting = TARGET_MODES[idx]
                self.capture_safe_snapshot()

    def handle_dialog_click(self, pos):
        if pygame.Rect(480, 420, 140, 44).collidepoint(pos): self.dialog["yes"]()
        elif pygame.Rect(660, 420, 140, 44).collidepoint(pos): self.dialog["no"]()

    def update_background(self, dt):
        for p in self.background_particles:
            p[0].y -= p[1] * dt
            p[0].x += math.sin(p[0].y * .01 + p[1]) * 4 * dt
            if p[0].y < -10:
                p[0].y = DESIGN_H + 10
                p[0].x = random.randrange(DESIGN_W)

    def update_simulation(self, dt):
        if self.state != "game" or self.outcome_decided:
            return
        self.play_session += dt
        self.message_timer = max(0, self.message_timer - dt)
        self.core_flash = max(0, self.core_flash - dt)
        self.screen_shake = max(0, self.screen_shake - dt * 20)
        self.wave_manager.update(self, dt)
        self.rebuild_enemy_grid()
        for tower in list(self.towers):
            tower.update(self, dt)
        for enemy in list(self.enemies):
            enemy.update(self, dt)
        if self.enemies_to_add:
            self.enemies.extend(self.enemies_to_add)
            self.enemies_to_add.clear()
        self.rebuild_enemy_grid()
        for projectile in list(self.projectiles):
            projectile.update(self, dt)
        self.projectiles = [p for p in self.projectiles if not p.dead]
        self.enemies = [e for e in self.enemies if e.alive]
        self.abilities.update(self, dt)
        for collection in (self.particles, self.floating_texts, self.beams):
            for item in list(collection):
                item.update(dt)
            collection[:] = [item for item in collection if not item.dead]
        self.score = max(0, int(self.score))
        self.credits = max(0, int(self.credits))
        self.save.data["best_wave"] = max(
            self.save.data["best_wave"], total_campaign_wave(self.campaign_level, min(30, self.wave_manager.current_wave)))
        self.save.data["best_level"] = max(self.save.data["best_level"], self.campaign_level)

    def update(self, real_dt):
        self.update_background(real_dt)
        if self.state == "game":
            self.accumulator += min(real_dt, MAX_FRAME_DT) * self.speed
            steps = 0
            while self.accumulator >= FIXED_STEP and steps < 12:
                self.update_simulation(FIXED_STEP)
                self.accumulator -= FIXED_STEP
                steps += 1
        else:
            self.accumulator = 0
        if self.message_timer > 0 and self.state != "game":
            self.message_timer = max(0, self.message_timer - real_dt)

    def draw_background(self):
        self.canvas.blit(self.background, (0, 0))
        overlay = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        overlay.fill((4, 8, 16, 165))
        self.canvas.blit(overlay, (0, 0))
        tick = pygame.time.get_ticks() * .02
        grid = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for x in range(-80, DESIGN_W + 80, 64):
            xx = x + int(tick) % 64
            pygame.draw.line(grid, (*C_CYAN, 17), (xx, 0), (xx - 160, DESIGN_H), 1)
        for y in range(0, DESIGN_H, 56):
            pygame.draw.line(grid, (*C_BLUE, 14), (0, y), (DESIGN_W, y), 1)
        self.canvas.blit(grid, (0, 0))
        for p, _, color, radius in self.background_particles:
            pygame.draw.circle(self.canvas, color, p, int(radius))

    def draw(self):
        self.draw_background()
        mouse = self.virtual_mouse()
        if self.state == "title": self.draw_title(mouse)
        elif self.state == "map": self.draw_map_select(mouse)
        elif self.state == "difficulty": self.draw_difficulty(mouse)
        elif self.state == "instructions": self.draw_instructions(mouse)
        elif self.state == "statistics": self.draw_statistics(mouse)
        elif self.state in ("game", "paused"): self.draw_game(mouse)
        elif self.state in ("victory", "defeat"): self.draw_outcome(mouse)
        if self.dialog: self.draw_dialog(mouse)
        self.present()

    def draw_title(self, mouse):
        text(self.canvas, self.fonts["title"], "TOWER-VISION", (DESIGN_W//2, 90), C_CYAN, "center")
        text(self.canvas, self.fonts["body"], "FrameVision Neon Defense Protocol", (DESIGN_W//2, 145), C_MUTED, "center")
        self.title_buttons[1].enabled = self.save.data.get("campaign") is not None
        for b in self.title_buttons:
            if b.label == "Sound": b.label = "Sound: OFF" if self.sound.muted else "Sound: ON"
            b.draw(self.canvas, self.fonts, mouse)
        text(self.canvas, self.fonts["small"], f"Volume: {int(self.sound.volume * 100)}%   •   Map: {self.maps[self.selected_map_id].name}   •   Difficulty: {self.difficulty}",
             (DESIGN_W//2, 607), C_GREEN, "center")
        text(self.canvas, self.fonts["tiny"], "F11 Fullscreen  •  M Mute  •  Endless escalating levels, 30 waves each", (DESIGN_W//2, 642), C_MUTED, "center")
        self.tooltip.update(mouse, self.title_buttons)
        self.tooltip.draw(self.canvas, self.fonts)

    def draw_map_select(self, mouse):
        text(self.canvas, self.fonts["h1"], "SELECT MAP", (DESIGN_W//2, 86), C_CYAN, "center")
        for i, b in enumerate(self.map_buttons):
            b.selected = MAP_IDS[i] == self.selected_map_id
            b.draw(self.canvas, self.fonts, mouse)
            map_def = self.maps[MAP_IDS[i]]
            preview = pygame.Rect(b.rect.left + 20, b.rect.top + 60, b.rect.width - 40, 125)
            pygame.draw.rect(self.canvas, (7, 13, 22), preview, border_radius=7)
            self.draw_mini_map(map_def, preview)
            lines = wrap_text(self.fonts["tiny"], map_def.description, b.rect.width - 30)
            for j, line in enumerate(lines[:3]):
                text(self.canvas, self.fonts["tiny"], line, (b.rect.centerx, b.rect.bottom - 54 + j*16), C_MUTED, "center", False)
        self.draw_back_button(mouse, "title")
        self.tooltip.update(mouse, self.map_buttons)
        self.tooltip.draw(self.canvas, self.fonts)

    def draw_mini_map(self, map_def, rect):
        all_points = [p for route in map_def.routes for p in route]
        minx, maxx = min(p[0] for p in all_points), max(p[0] for p in all_points)
        miny, maxy = min(p[1] for p in all_points), max(p[1] for p in all_points)
        sx = (rect.width - 20) / max(1, maxx - minx)
        sy = (rect.height - 20) / max(1, maxy - miny)
        scale = min(sx, sy)
        def tr(p): return (rect.left + 10 + (p[0]-minx)*scale, rect.top + 10 + (p[1]-miny)*scale)
        for route in map_def.routes:
            pygame.draw.lines(self.canvas, map_def.accent, False, [tr(p) for p in route], 4)
        for p in map_def.pads:
            pygame.draw.circle(self.canvas, C_GREEN, tr(p), 3, 1)
        pygame.draw.circle(self.canvas, C_CYAN, tr(map_def.routes[0][0]), 6)
        pygame.draw.circle(self.canvas, C_RED, tr(map_def.routes[0][-1]), 7)

    def draw_difficulty(self, mouse):
        text(self.canvas, self.fonts["h1"], "SELECT DIFFICULTY", (DESIGN_W//2, 90), C_CYAN, "center")
        for b in self.diff_buttons:
            b.selected = b.label == self.difficulty
            b.draw(self.canvas, self.fonts, mouse)
        text(self.canvas, self.fonts["body"], f"Map: {self.maps[self.selected_map_id].name}", (DESIGN_W//2, 560), C_GREEN, "center")
        self.draw_back_button(mouse, "map")
        self.tooltip.update(mouse, self.diff_buttons)
        self.tooltip.draw(self.canvas, self.fonts)

    def draw_back_button(self, mouse, state):
        rect = pygame.Rect(30, 650, 120 if state != "title" else 160, 42)
        pygame.draw.rect(self.canvas, C_PANEL_2, rect, border_radius=7)
        pygame.draw.rect(self.canvas, C_CYAN, rect, 2, border_radius=7)
        text(self.canvas, self.fonts["small"], "← Back", rect.center, C_TEXT, "center")

    def draw_instructions(self, mouse):
        text(self.canvas, self.fonts["h1"], "INSTRUCTIONS", (DESIGN_W//2, 42), C_CYAN, "center")
        columns = [
            (50, 95, "OBJECTIVE & WAVES", "Build on glowing pads and stop every enemy before it reaches the reactor. Each level contains exactly 30 waves, then the next level begins with stronger enemies and new tower upgrade tiers. Space or Start Wave begins early for a one-time bonus. Auto Wave starts each wave when preparation ends. Bosses arrive on waves 10, 20 and 30."),
            (440, 95, "TOWERS", "Q Pulse: balanced rapid fire. W Railgun: armor-piercing sniper. E Arc: chain lightning. R Cryo: slow and freeze. T Missile: area damage and burn. Y Drone Hub: autonomous orbiting drones. Click a tower to inspect, upgrade, sell and change targeting."),
            (830, 95, "TARGETING", "First and Last compare normalized route progress, including Split Junction branches. Strongest and Weakest use remaining health plus shield. Fastest uses current effective speed. Closest uses distance from the tower."),
            (50, 335, "ABILITIES", "EMP stuns normal enemies and briefly affects bosses. Orbital Strike enters targeting mode; click the board to place a delayed blast, or right-click/Escape to cancel. Orbital damage intentionally affects phased units. Emergency Repair restores reactor health but cannot be used at full health."),
            (440, 335, "ENEMIES & BOSSES", "Runners are fast; Tanks are armored; Shields regenerate after avoiding damage; Splitters create children; Regenerators heal; Phase Units periodically become untargetable. Bosses use resistance, tower disable, movement phases, shields and reinforcements."),
            (830, 335, "CONTROLS & SAVING", "Space: wave. P/Escape: pause. 1/2/3: speed. F11: fullscreen. M: mute. U: upgrade. Delete twice: confirm sale. Right-click: cancel placement. Each level contains 30 waves. Defeating the Core Devourer advances to a harder level, keeps your defenses, and unlocks two new tower tiers. The game saves statistics and safe wave-preparation snapshots; live enemies and projectiles are never serialized."),
        ]
        for x, y, heading, body in columns:
            rect = pygame.Rect(x, y, 350, 205)
            pygame.draw.rect(self.canvas, (8, 15, 25, 220), rect, border_radius=8)
            pygame.draw.rect(self.canvas, C_BLUE, rect, 1, border_radius=8)
            text(self.canvas, self.fonts["h2"], heading, (x+15, y+13), C_GREEN)
            for i, line in enumerate(wrap_text(self.fonts["small"], body, 320)):
                text(self.canvas, self.fonts["small"], line, (x+15, y+50+i*19), C_TEXT, shadow=False)
        self.draw_back_button(mouse, "title")

    def draw_statistics(self, mouse):
        text(self.canvas, self.fonts["h1"], "CAREER STATISTICS", (DESIGN_W//2, 70), C_CYAN, "center")
        stats = [
            ("High Score", self.save.data["high_score"]), ("Best Level", self.save.data.get("best_level", 1)),
            ("Highest Wave Reached", self.save.data["best_wave"]), ("Levels Cleared", self.save.data.get("levels_cleared", 0)),
            ("Enemies Defeated", self.save.data["total_enemies_defeated"]), ("Bosses Defeated", self.save.data["bosses_defeated"]),
            ("Games Won", self.save.data["games_won"]), ("Games Lost", self.save.data["games_lost"]),
            ("Towers Built", self.save.data["towers_built"]), ("Towers Upgraded", self.save.data["towers_upgraded"]),
            ("Credits Earned", self.save.data["credits_earned"]),
            ("Total Play Time", f"{int(self.save.data['total_play_time']//3600):02d}:{int(self.save.data['total_play_time']//60)%60:02d}:{int(self.save.data['total_play_time'])%60:02d}"),
        ]
        for i, (label, value) in enumerate(stats):
            x = 250 + (i % 2) * 410
            y = 150 + (i // 2) * 78
            rect = pygame.Rect(x, y, 370, 58)
            pygame.draw.rect(self.canvas, C_PANEL_2, rect, border_radius=8)
            pygame.draw.rect(self.canvas, C_GREEN if i % 2 == 0 else C_BLUE, rect, 1, border_radius=8)
            text(self.canvas, self.fonts["small"], label, (x+15, y+10), C_MUTED)
            text(self.canvas, self.fonts["h2"], value, (x+350, y+29), C_TEXT, "midright")
        self.draw_back_button(mouse, "title")
        reset = pygame.Rect(1020, 650, 220, 42)
        pygame.draw.rect(self.canvas, C_PANEL_2, reset, border_radius=7)
        pygame.draw.rect(self.canvas, C_RED, reset, 2, border_radius=7)
        text(self.canvas, self.fonts["small"], "Reset Save Data", reset.center, C_RED, "center")

    def draw_map_board(self):
        board_layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        pygame.draw.rect(board_layer, (5, 10, 17, 130), BOARD_RECT, border_radius=12)
        pygame.draw.rect(board_layer, (*self.map.accent, 95), BOARD_RECT, 2, border_radius=12)
        self.canvas.blit(board_layer, (0, 0))
        for block in self.map.blocked:
            pygame.draw.rect(self.canvas, (22, 31, 45), block, border_radius=6)
            pygame.draw.rect(self.canvas, (70, 92, 112), block, 2, border_radius=6)
            for x in range(block.left + 8, block.right - 4, 16):
                pygame.draw.line(self.canvas, (38, 52, 67), (x, block.top+5), (x-8, block.bottom-5), 2)
        path_layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for route in self.map.routes:
            pygame.draw.lines(path_layer, (7, 15, 23, 230), False, route, 34)
            pygame.draw.lines(path_layer, (*self.map.accent, 55), False, route, 25)
            pygame.draw.lines(path_layer, (*self.map.accent, 180), False, route, 3)
            offset = (pygame.time.get_ticks() // 35) % 32
            for a, b in zip(route, route[1:]):
                av, bv = v2(a), v2(b)
                length = av.distance_to(bv)
                if length <= 0: continue
                direction = (bv-av).normalize()
                for d in range(offset, int(length), 32):
                    p = av + direction*d
                    pygame.draw.circle(path_layer, (*C_WHITE, 90), p, 2)
        self.canvas.blit(path_layer, (0, 0))
        # Entrance portal and core.
        entrance = self.map.routes[0][0]
        core = self.map.routes[0][-1]
        draw_glow_circle(self.canvas, C_CYAN, entrance, 17 + int(3*math.sin(pygame.time.get_ticks()*.008)), 3, 220)
        text(self.canvas, self.fonts["tiny"], "ENTRY", (entrance[0]+10, entrance[1]-28), C_CYAN)
        core_color = C_WHITE if self.core_flash > 0 else C_RED
        draw_glow_circle(self.canvas, core_color, core, 23, 3, 230)
        pygame.draw.circle(self.canvas, C_PANEL, core, 13)
        text(self.canvas, self.fonts["tiny"], "REACTOR", (core[0]-8, core[1]+31), C_RED, "center")
        occupied = {t.pad_index for t in self.towers}
        for i, p in enumerate(self.map.pads):
            color = C_RED if i in occupied else C_GREEN
            pygame.draw.circle(self.canvas, (9, 18, 26), p, 19)
            pygame.draw.circle(self.canvas, color, p, 19, 2)
            if i not in occupied:
                pygame.draw.line(self.canvas, color, (p[0]-6, p[1]), (p[0]+6, p[1]), 2)
                pygame.draw.line(self.canvas, color, (p[0], p[1]-6), (p[0], p[1]+6), 2)

    def draw_game(self, mouse):
        shake = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            shake.update(self.rng.uniform(-self.screen_shake, self.screen_shake), self.rng.uniform(-self.screen_shake, self.screen_shake))
        world = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        original = self.canvas
        self.canvas = world
        self.draw_map_board()
        for tower in self.towers: tower.draw(self.canvas, tower is self.selected_tower)
        for enemy in self.enemies: enemy.draw(self.canvas)
        for projectile in self.projectiles: projectile.draw(self.canvas)
        for beam in self.beams: beam.draw(self.canvas)
        for particle in self.particles: particle.draw(self.canvas)
        for floating in self.floating_texts: floating.draw(self.canvas, self.fonts)
        self.abilities.draw(self.canvas, mouse)
        self.draw_placement_preview(mouse)
        self.canvas = original
        original.blit(world, shake)
        self.draw_hud(mouse)
        if self.state == "paused": self.draw_pause_overlay(mouse)
        if self.message_timer > 0:
            rect = pygame.Rect(DESIGN_W//2-230, 85, 460, 36)
            pygame.draw.rect(self.canvas, (4, 9, 16), rect, border_radius=7)
            pygame.draw.rect(self.canvas, self.message_color, rect, 1, border_radius=7)
            text(self.canvas, self.fonts["small"], self.message, rect.center, self.message_color, "center")

    def draw_placement_preview(self, mouse):
        if not self.placing_kind:
            return
        pad = self.nearest_pad(mouse)
        if pad is None:
            pos = v2(mouse)
            valid = False
        else:
            pos = v2(self.map.pads[pad])
            valid = not any(t.pad_index == pad for t in self.towers)
        spec = TOWER_SPECS[self.placing_kind]
        color = spec["color"] if valid else C_RED
        layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        pygame.draw.circle(layer, (*color, 25), pos, int(spec["range"]), 0)
        pygame.draw.circle(layer, (*color, 130), pos, int(spec["range"]), 1)
        pygame.draw.circle(layer, (*color, 170), pos, 18)
        self.canvas.blit(layer, (0, 0))
        cost = int(round(spec["cost"] * DIFFICULTY_DATA[self.difficulty]["tower_price"]))
        text(self.canvas, self.fonts["tiny"], f"{spec['name']}  {cost} cr", (pos.x, pos.y-30), color, "center")

    def draw_hud(self, mouse):
        top = pygame.Rect(0, 0, DESIGN_W, 68)
        pygame.draw.rect(self.canvas, (5, 10, 18), top)
        pygame.draw.line(self.canvas, C_CYAN, (0, 67), (DESIGN_W, 67), 2)
        text(self.canvas, self.fonts["h2"], "TOWER-VISION", (18, 16), C_CYAN)
        text(self.canvas, self.fonts["small"], f"Credits {self.credits}", (210, 13), C_GREEN)
        text(self.canvas, self.fonts["small"], f"Score {self.score}", (210, 37), C_TEXT)
        text(self.canvas, self.fonts["small"], f"High {max(self.save.data['high_score'], self.score)}", (340, 13), C_YELLOW)
        text(self.canvas, self.fonts["small"], f"Level {self.campaign_level}  •  Wave {self.wave_manager.current_wave}/30", (300, 37), C_TEXT)
        text(self.canvas, self.fonts["small"], f"Remaining {self.wave_manager.remaining(self)}", (475, 13), C_TEXT)
        text(self.canvas, self.fonts["small"], f"{self.difficulty} • {self.map.name}", (530, 37), C_MUTED)
        # Reactor bar.
        pygame.draw.rect(self.canvas, (24, 29, 36), (645, 18, 162, 17), border_radius=5)
        ratio = clamp(self.reactor_health / self.reactor_max, 0, 1)
        pygame.draw.rect(self.canvas, C_GREEN if ratio > .35 else C_RED, (645, 18, 162*ratio, 17), border_radius=5)
        text(self.canvas, self.fonts["tiny"], f"REACTOR {int(self.reactor_health)}/{int(self.reactor_max)}", (726, 26), C_WHITE, "center")
        # Top controls.
        controls = [((825,18,76,34), "START", self.wave_manager.phase == "prep", "Start the next wave early for a one-time credit bonus."),
                    ((910,18,55,34), "PAUSE", True, "Pause the entire simulation while keeping the UI responsive."),
                    ((975,18,50,34), "1×", True, "Normal simulation speed. Shortcut 1."),
                    ((1030,18,50,34), "2×", True, "Double simulation speed. Shortcut 2."),
                    ((1085,18,50,34), "3×", True, "Triple simulation speed. Shortcut 3."),
                    ((1145,18,95,34), "AUTO ON" if self.wave_manager.auto_wave else "AUTO OFF", True, "Automatically begin waves when preparation expires."),]
        tip_items = []
        for rect, label, enabled, tip in controls:
            r = pygame.Rect(rect); tip_items.append(type("Tip", (), {"rect": r, "tooltip": tip})())
            pygame.draw.rect(self.canvas, C_PANEL_2 if enabled else (29,34,40), r, border_radius=6)
            selected = (label.startswith(str(self.speed)) and "×" in label) or (label == "AUTO ON")
            pygame.draw.rect(self.canvas, C_GREEN if selected else C_BLUE, r, 2, border_radius=6)
            text(self.canvas, self.fonts["tiny"], label, r.center, C_TEXT if enabled else C_MUTED, "center")
        # Side panel.
        pygame.draw.rect(self.canvas, (6, 12, 21), SIDE_RECT, border_radius=10)
        pygame.draw.rect(self.canvas, C_BLUE, SIDE_RECT, 2, border_radius=10)
        text(self.canvas, self.fonts["h2"], "DEFENSE GRID", (1114, 84), C_CYAN, "midtop")
        for i, kind in enumerate(TOWER_KEYS):
            spec = TOWER_SPECS[kind]
            r = pygame.Rect(980 + (i % 2) * 136, 112 + (i // 2) * 58, 126, 50)
            cost = int(round(spec["cost"] * DIFFICULTY_DATA[self.difficulty]["tower_price"]))
            selected = self.placing_kind == kind
            pygame.draw.rect(self.canvas, C_PANEL_2, r, border_radius=7)
            pygame.draw.rect(self.canvas, spec["color"] if selected or r.collidepoint(mouse) else (65,85,105), r, 2, border_radius=7)
            draw_glow_circle(self.canvas, spec["color"], (r.left+18, r.centery), 7, alpha=150)
            text(self.canvas, self.fonts["tiny"], spec["name"].replace(" Tower", "").replace(" Battery", ""), (r.left+32, r.top+7), C_TEXT)
            text(self.canvas, self.fonts["tiny"], f"{cost} cr", (r.left+32, r.top+26), C_GREEN)
            tip_items.append(type("Tip", (), {"rect": r, "tooltip": f"{spec['name']}: {spec['desc']} {spec['special']} Hotkey {('Q','W','E','R','T','Y')[i]}."})())
        text(self.canvas, self.fonts["small"], "ACTIVE ABILITIES", (1114, 278), C_GREEN, "midtop")
        for i, key in enumerate(("emp", "orbital", "repair")):
            data = ABILITY_DATA[key]
            r = pygame.Rect(980, 305 + i*48, 266, 40)
            cd = self.abilities.cooldowns[key]
            ready = cd <= 0
            pygame.draw.rect(self.canvas, C_PANEL_2, r, border_radius=7)
            pygame.draw.rect(self.canvas, data["color"] if ready else (80,85,95), r, 2, border_radius=7)
            text(self.canvas, self.fonts["small"], data["name"], (r.left+12, r.centery), C_TEXT, "midleft")
            text(self.canvas, self.fonts["small"], "READY" if ready else f"{cd:.1f}s", (r.right-12, r.centery), data["color"] if ready else C_MUTED, "midright")
            tip_items.append(type("Tip", (), {"rect": r, "tooltip": data["desc"] + " Cooldowns pause with the simulation."})())
        # Wave preview / selection panel.
        y = 455
        if self.selected_tower and self.selected_tower in self.towers:
            t = self.selected_tower
            text(self.canvas, self.fonts["h2"], t.spec["name"], (1114, y), t.spec["color"], "midtop")
            text(self.canvas, self.fonts["small"], f"Tier {t.level+1}/{t.level_cap+1}   Target: {t.targeting}", (980, y+30), C_TEXT)
            text(self.canvas, self.fonts["tiny"], f"Damage {t.damage:.0f}   Range {t.range:.0f}   Fire rate {t.fire_rate:.2f}/s", (980, y+53), C_MUTED)
            special_lines = wrap_text(self.fonts["tiny"], t.spec["special"], 260)
            for j, line in enumerate(special_lines[:2]): text(self.canvas, self.fonts["tiny"], line, (980, y+73+j*16), C_MUTED, shadow=False)
            up = pygame.Rect(980, 588, 126, 40); sell = pygame.Rect(1120, 588, 126, 40); target = pygame.Rect(980, 640, 266, 34)
            for r, label, col in ((up, "MAX" if t.maxed else f"Upgrade {t.upgrade_cost}", C_GREEN),
                                  (sell, f"Sell {t.sale_value}", C_RED), (target, f"Targeting: {t.targeting} ›", C_CYAN)):
                pygame.draw.rect(self.canvas, C_PANEL_2, r, border_radius=6); pygame.draw.rect(self.canvas, col, r, 2, border_radius=6)
                text(self.canvas, self.fonts["tiny"], label, r.center, col, "center")
            tip_items.extend([
                type("Tip", (), {"rect": up, "tooltip": f"Upgrade immediately improves stats. This run currently unlocks tower tier {t.level_cap + 1}; two more tiers unlock after each cleared level."})(),
                type("Tip", (), {"rect": sell, "tooltip": "Sell refunds 68% of total investment. Click twice or press Delete twice to confirm."})(),
                type("Tip", (), {"rect": target, "tooltip": "Cycle First, Last, Strongest, Weakest, Fastest and Closest."})(),
            ])
        else:
            text(self.canvas, self.fonts["h2"], "NEXT WAVE", (1114, y), C_CYAN, "midtop")
            preview = ", ".join(name.title() for name in self.wave_manager.preview())
            preview_rect = pygame.Rect(976, y+28, 274, 67)
            for j, line in enumerate(wrap_text(self.fonts["small"], preview, 250)):
                text(self.canvas, self.fonts["small"], line, (980, y+35+j*20), C_TEXT)
            tip_items.append(type("Tip", (), {"rect": preview_rect, "tooltip": "Enemy guide: Runner—fast; Grunt—balanced; Tank—armored; Swarm—numerous; Shield—regenerating shield; Splitter—creates children; Regenerator—heals; Phase—periodically untargetable."})())
            threat = campaign_hp_scale(self.campaign_level)
            phase_text = (f"Prep: {max(0,self.wave_manager.prep_timer):.1f}s  •  Threat ×{threat:.2f}"
                          if self.wave_manager.phase == "prep" else f"Wave active  •  Threat ×{threat:.2f}")
            text(self.canvas, self.fonts["small"], phase_text, (980, y+92), C_YELLOW)
            pygame.draw.rect(self.canvas, (24,29,36), (980, y+122, 266, 12), border_radius=4)
            pygame.draw.rect(self.canvas, C_CYAN, (980, y+122, 266*self.wave_manager.progress(self), 12), border_radius=4)
            status_rect = pygame.Rect(976, y+143, 274, 32)
            text(self.canvas, self.fonts["tiny"], "Status effects: slow • freeze • burn • stun • armor break • disable", (980, y+151), C_MUTED)
            tip_items.append(type("Tip", (), {"rect": status_rect, "tooltip": "Slow uses the strongest capped effect; freeze and stun have strict duration caps; burn deals timed damage; armor break lowers defense temporarily; disable stops a tower until its timer ends."})())
        self.tooltip.update(mouse, tip_items)
        self.tooltip.draw(self.canvas, self.fonts)
        # Boss health bar.
        bosses = [e for e in self.enemies if e.alive and e.is_boss]
        if bosses:
            boss = bosses[0]
            r = pygame.Rect(250, 82, 470, 18)
            pygame.draw.rect(self.canvas, (18,20,28), r, border_radius=5)
            pygame.draw.rect(self.canvas, C_RED, (r.x, r.y, r.width*clamp(boss.hp/boss.max_hp,0,1), r.height), border_radius=5)
            text(self.canvas, self.fonts["tiny"], f"{boss.name}  {int(boss.hp)}/{int(boss.max_hp)}", r.center, C_WHITE, "center")

    def draw_pause_overlay(self, mouse):
        layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 165))
        self.canvas.blit(layer, (0, 0))
        box = pygame.Rect(450, 235, 380, 250)
        pygame.draw.rect(self.canvas, C_PANEL, box, border_radius=12)
        pygame.draw.rect(self.canvas, C_CYAN, box, 2, border_radius=12)
        text(self.canvas, self.fonts["h1"], "PAUSED", (640, 280), C_CYAN, "center")
        for r, label in ((pygame.Rect(500,330,280,48), "Resume"), (pygame.Rect(500,390,280,48), "Save & Return to Title")):
            pygame.draw.rect(self.canvas, C_PANEL_2, r, border_radius=7); pygame.draw.rect(self.canvas, C_GREEN, r, 2, border_radius=7)
            text(self.canvas, self.fonts["small"], label, r.center, C_TEXT, "center")

    def draw_outcome(self, mouse):
        victory = self.state == "victory"
        color = C_GREEN if victory else C_RED
        title_content = "REACTOR SECURED" if victory else "REACTOR LOST"
        text(self.canvas, self.fonts["title"], title_content, (DESIGN_W//2, 155), color, "center")
        text(self.canvas, self.fonts["h2"], f"Score {self.score}   •   Level {self.campaign_level}   •   Wave {self.wave_manager.current_wave}/30",
             (DESIGN_W//2, 245), C_TEXT, "center")
        summary = ("The Core Devourer and every remaining required enemy were eliminated." if victory else
                   "The reactor reached zero health. Rebuild your coverage, targeting and upgrade timing for the next endless run.")
        for i, line in enumerate(wrap_text(self.fonts["body"], summary, 650)):
            text(self.canvas, self.fonts["body"], line, (DESIGN_W//2, 315+i*28), C_MUTED, "center")
        r = pygame.Rect(505, 560, 270, 48)
        pygame.draw.rect(self.canvas, C_PANEL_2, r, border_radius=8); pygame.draw.rect(self.canvas, color, r, 2, border_radius=8)
        text(self.canvas, self.fonts["small"], "Start New Campaign", r.center, C_TEXT, "center")
        self.draw_back_button(mouse, "title")

    def draw_dialog(self, mouse):
        layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); layer.fill((0,0,0,170)); self.canvas.blit(layer,(0,0))
        box = pygame.Rect(380, 255, 520, 235)
        pygame.draw.rect(self.canvas, C_PANEL, box, border_radius=12); pygame.draw.rect(self.canvas, C_RED, box, 2, border_radius=12)
        text(self.canvas, self.fonts["h2"], self.dialog["title"], (640, 292), C_RED, "center")
        for i, line in enumerate(wrap_text(self.fonts["small"], self.dialog["message"], 450)):
            text(self.canvas, self.fonts["small"], line, (640, 340+i*20), C_TEXT, "center")
        for r,label,col in ((pygame.Rect(480,420,140,44),"Yes",C_RED),(pygame.Rect(660,420,140,44),"No",C_GREEN)):
            pygame.draw.rect(self.canvas,C_PANEL_2,r,border_radius=7); pygame.draw.rect(self.canvas,col,r,2,border_radius=7)
            text(self.canvas,self.fonts["small"],label,r.center,col,"center")

    def present(self):
        if self.headless:
            return
        ww, wh = self.window.get_size()
        scale = min(ww / DESIGN_W, wh / DESIGN_H)
        target = (max(1, int(DESIGN_W * scale)), max(1, int(DESIGN_H * scale)))
        # Integer-like nearest-neighbor scaling keeps the virtual canvas crisp and coordinate mapping stable.
        scaled = pygame.transform.scale(self.canvas, target)
        self.window.fill(C_BLACK)
        self.window.blit(scaled, ((ww-target[0])//2, (wh-target[1])//2))
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, MAX_FRAME_DT)
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw()
        self.save_settings()
        pygame.quit()


# -----------------------------------------------------------------------------
# Self tests
# -----------------------------------------------------------------------------
def run_selftest():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    game = Game(headless=True)
    # Never let developer tests touch the user's real FrameVision save.
    test_dir = Path(tempfile.mkdtemp(prefix="tower_vision_selftest_"))
    game.save.save_path = test_dir / "tower_defense_save.json"
    game.save.data = game.save.default_data()
    game.selected_map_id = "neon_circuit"
    game.difficulty = "Normal"
    game.reset_runtime()
    assert len(WAVE_DEFINITIONS) == 30
    assert set(game.maps) == set(MAP_IDS)
    for map_def in game.maps.values():
        assert map_def.routes and len(map_def.pads) >= 10
        for route in map_def.routes:
            assert len(route) >= 4
            assert all(BOARD_RECT.inflate(20,20).collidepoint(p) for p in route)
            assert sum(v2(route[i]).distance_to(route[i+1]) for i in range(len(route)-1)) > 500
        assert all(BOARD_RECT.collidepoint(p) for p in map_def.pads)
    # Placement transaction and no-double-charge checks.
    start = game.credits
    assert game.place_tower("pulse", 0)
    cost = game.towers[0].base_cost
    assert game.credits == start - cost
    assert not game.place_tower("pulse", 0)
    assert game.credits == start - cost
    # Upgrade and sale reconstruction.
    game.credits += 10000
    target_mode = "Strongest"
    game.towers[0].targeting = target_mode
    assert game.upgrade_selected()
    assert game.towers[0].targeting == target_mode
    game.capture_safe_snapshot()
    validated = game.save.validate_campaign(game.safe_snapshot)
    assert validated and validated["towers"][0]["level"] == 1
    # Route progress and high-speed endpoint safety on both branch routes.
    game.selected_map_id = "split_junction"; game.map = game.maps["split_junction"]
    for rid in range(2):
        e = Enemy("runner", game.map.route_vectors(rid), rid, "Hard", 30)
        for _ in range(500):
            if not e.alive: break
            e.update(game, .05)
        assert e.escape_processed
        assert e.progress >= .99
    # Split children are added once and count as required.
    game.reset_runtime(); game.state = "game"
    splitter = Enemy("splitter", game.map.route_vectors(0), 0, "Normal", 18)
    game.enemies = [splitter]
    splitter.take_damage(game, 1e9, ignore_phase=True)
    assert len(game.enemies_to_add) == 2 and all(c.required for c in game.enemies_to_add)
    # Chain lightning never repeats targets.
    game.reset_runtime(); game.state = "game"
    game.place_tower("arc", 0); arc = game.towers[0]
    game.enemies = [Enemy("grunt", game.map.route_vectors(0), 0, "Normal", 1, start_progress=.1+i*.01) for i in range(5)]
    hp_before = {e.id:e.hp for e in game.enemies}
    arc.fire_arc(game, game.enemies[0])
    assert all(e.hp <= hp_before[e.id] for e in game.enemies)
    # Explosion one-hit behavior and duplicate kill protection.
    game.reset_runtime(); game.state = "game"
    victim = Enemy("swarm", game.map.route_vectors(0), 0, "Normal", 1, start_progress=.1)
    game.enemies = [victim]
    initial_defeats = game.save.data["total_enemies_defeated"]
    game.explode(victim.pos, 100, 9999)
    game.explode(victim.pos, 100, 9999)
    assert game.save.data["total_enemies_defeated"] == initial_defeats + 1
    # Every tower type can acquire and damage a valid target.
    for index, kind in enumerate(TOWER_KEYS):
        game.reset_runtime(); game.state = "game"; game.credits = 999999
        assert game.place_tower(kind, index)
        tower = game.towers[0]
        victim_kind = "tank" if kind == "rail" else "grunt"
        victim = Enemy(victim_kind, game.map.route_vectors(0), 0, "Normal", 10)
        victim.pos = tower.pos + pygame.Vector2(min(60, tower.range / 2), 0)
        victim.spawn_protection = 0
        game.enemies = [victim]
        before = victim.durability
        for _ in range(360):
            game.update_simulation(FIXED_STEP)
        assert victim.durability < before or not victim.alive, kind
    # Core Devourer phase transitions fire once and reach the enraged phase.
    game.reset_runtime(); game.state = "game"
    boss = BossEnemy("devourer", game.map.route_vectors(0), 0, "Normal", 30)
    boss.spawn_protection = 0
    game.enemies = [boss]
    boss.take_damage(game, boss.shield + 1000, armor_pen=100, ignore_phase=True)
    boss.take_damage(game, 1, armor_pen=100, ignore_phase=True)
    assert boss.phase_stage >= 1
    boss.take_damage(game, boss.max_hp * .35, armor_pen=100, ignore_phase=True)
    assert boss.phase_stage == 2 and boss.enraged
    # Wave start is idempotent.
    game.reset_runtime(); game.state = "game"
    assert game.wave_manager.start_wave(game, early=True)
    queued = len(game.wave_manager.spawn_queue)
    assert not game.wave_manager.start_wave(game, early=True)
    assert len(game.wave_manager.spawn_queue) == queued
    # Final wave waits for every required enemy, then advances to a harder 30-wave level.
    game.reset_runtime(); game.state = "game"
    game.wave_manager.current_wave = 30
    game.wave_manager.phase = "active"
    game.wave_manager.spawn_queue = []
    blocker = Enemy("grunt", game.map.route_vectors(0), 0, "Normal", 30, campaign_level=1)
    blocker.spawn_protection = 0
    game.enemies = [blocker]
    game.wave_manager.update(game, FIXED_STEP)
    assert game.state == "game" and game.campaign_level == 1 and not game.outcome_decided
    blocker.die(game)
    game.enemies = []
    game.wave_manager.update(game, FIXED_STEP)
    assert game.state == "game" and game.campaign_level == 2
    assert game.wave_manager.current_wave == 0 and game.wave_manager.phase == "prep"
    assert tower_level_cap(2) == 5
    assert campaign_hp_scale(2) > campaign_hp_scale(1)
    # Defeat remains final and cannot be replaced by the legacy victory guard.
    game.reset_runtime(); game.reactor_health = 0; game.trigger_defeat(); state = game.state; game.trigger_victory(); assert game.state == state == "defeat"
    print("Tower-Vision selftest: PASS")
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--selftest", action="store_true", help="run noninteractive logic checks")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        Game().run()


if __name__ == "__main__":
    main()
